#!/usr/bin/env python3
"""
End-to-end training with answer loss flowing through state vector.

The gradient chain:
  answer_loss → logits → hidden_states → state_tokens → injector → state → perceiver

This is the experiment that matters.
"""

import os
import json
import random
import re
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    # Model
    "model_name": "meta-llama/Llama-3.2-1B-Instruct",
    "hf_token_file": "secrets/hf_token.txt",

    # Architecture (matches v18 spec)
    "num_layers": 16,        # Llama 3.2 1B has 16 layers
    "d_model": 2048,         # Llama hidden dim
    "state_size": 64,        # Tight bottleneck (not 512)
    "num_tokens": 4,         # 4 pseudo-tokens
    "num_queries": 8,        # 8 queries → 8 floats each → 64 total
    "num_perceiver_layers": 4,
    "d_perceiver": 512,

    # Thinking
    "max_passes": 3,         # Start conservative
    "alpha_init": 0.1,       # Residual accumulation strength

    # State scale warmup (critical for cold start)
    "scale_start": 0.1,      # Random perceiver barely affects generation
    "scale_end": 1.0,        # Full strength after warmup
    "scale_warmup_steps": 500,

    # Training
    "batch_size": 4,         # Small batch, accumulate gradients
    "grad_accum_steps": 4,   # Effective batch = 16
    "learning_rate": 1e-4,
    "epochs_per_level": 5,
    "max_seq_len": 256,
    "max_answer_tokens": 32,

    # Logging
    "log_every": 20,
    "eval_every": 100,
}


# ============================================================================
# Architecture Components
# ============================================================================

class AllLayerPerceiver(nn.Module):
    """
    Reads ALL transformer layers with pass-conditioned attention.
    Outputs a tight 64-float state vector.
    """
    def __init__(self, num_layers=16, d_model=2048, d_perceiver=512,
                 num_queries=8, num_perceiver_layers=4, state_size=64, max_passes=10):
        super().__init__()
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.state_size = state_size

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_perceiver) * 0.02)

        # Pass conditioning - queries shift based on which pass we're on
        self.pass_embed = nn.Embedding(max_passes, d_perceiver)

        # Layer gate - learns which transformer layers matter (pass-conditioned)
        self.layer_gate_base = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.layer_gate_pass = nn.Linear(d_perceiver, num_layers)

        # Project transformer hidden states to perceiver dim
        self.input_proj = nn.Linear(d_model, d_perceiver)

        # Cross-attention: queries attend to layer-weighted hidden states
        self.cross_attn = nn.MultiheadAttention(d_perceiver, num_heads=8, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_perceiver)

        # Self-attention layers for processing
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_perceiver, nhead=8, dim_feedforward=d_perceiver*4,
                                       batch_first=True, norm_first=True)
            for _ in range(num_perceiver_layers)
        ])

        # Project to state vector: num_queries * (state_size // num_queries) = 64
        self.output_proj = nn.Linear(d_perceiver, state_size // num_queries)

    def forward(self, all_hidden_states, pass_num=0):
        """
        Args:
            all_hidden_states: list of (batch, seq, d_model) for each layer
            pass_num: which thinking pass (0-indexed)
        Returns:
            state: (batch, 64) compressed state vector
        """
        batch_size = all_hidden_states[0].size(0)
        device = all_hidden_states[0].device

        # Stack all layers: (num_layers, batch, seq, d_model)
        stacked = torch.stack([h.float() for h in all_hidden_states], dim=0)

        # Pass-conditioned layer gate
        pass_idx = torch.tensor([pass_num], device=device)
        pass_emb = self.pass_embed(pass_idx)  # (1, d_perceiver)
        layer_gate_delta = self.layer_gate_pass(pass_emb).squeeze(0)  # (num_layers,)
        layer_weights = F.softmax(self.layer_gate_base + 0.1 * layer_gate_delta, dim=0)

        # Weighted combination of layers: (batch, seq, d_model)
        combined = (stacked * layer_weights.view(-1, 1, 1, 1)).sum(dim=0)

        # Pool over sequence (mean pool) → (batch, d_model)
        pooled = combined.mean(dim=1)

        # Project to perceiver dim
        x = self.input_proj(pooled)  # (batch, d_perceiver)

        # Expand queries for batch
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_queries, d_perceiver)

        # Add pass embedding to queries
        queries = queries + pass_emb.unsqueeze(0)

        # Cross-attention: queries attend to projected hidden states
        x_expanded = x.unsqueeze(1)  # (batch, 1, d_perceiver)
        attn_out, _ = self.cross_attn(queries, x_expanded, x_expanded)
        queries = self.cross_norm(queries + attn_out)

        # Self-attention processing
        for layer in self.layers:
            queries = layer(queries)

        # Project to state chunks and flatten
        state_chunks = self.output_proj(queries)  # (batch, num_queries, state_size//num_queries)
        state = state_chunks.flatten(start_dim=1)  # (batch, 64)

        return state


class StateInjector(nn.Module):
    """Converts 64-float state to 4 pseudo-tokens."""
    def __init__(self, state_size=64, d_model=2048, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.chunk_size = state_size // num_tokens  # 16
        self.project = nn.Linear(self.chunk_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, state, scale=1.0, dtype=torch.bfloat16):
        """
        Args:
            state: (batch, 64)
            scale: state scale for warmup (0.1 → 1.0)
            dtype: output dtype (must match transformer)
        Returns:
            tokens: (batch, 4, d_model)
        """
        # Reshape to chunks
        chunks = state.view(-1, self.num_tokens, self.chunk_size)  # (batch, 4, 16)
        # Project each chunk to d_model
        tokens = self.project(chunks)  # (batch, 4, d_model)
        # Add position embeddings
        tokens = tokens + self.pos_embed
        # Normalize and scale
        tokens = self.norm(tokens) * scale
        # Convert to transformer dtype
        return tokens.to(dtype)


class ThinkingModel(nn.Module):
    """
    Full thinking model: multiple passes through transformer,
    state accumulates, final answer generated.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        print(f"Loading {config['model_name']}...")
        self.transformer = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.bfloat16,
            token=config.get('hf_token'),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model_name'],
            token=config.get('hf_token'),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze transformer initially
        for p in self.transformer.parameters():
            p.requires_grad = False

        # Perceiver and injector
        self.perceiver = AllLayerPerceiver(
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            d_perceiver=config['d_perceiver'],
            num_queries=config['num_queries'],
            num_perceiver_layers=config['num_perceiver_layers'],
            state_size=config['state_size'],
            max_passes=config['max_passes'] + 5,
        )

        self.injector = StateInjector(
            state_size=config['state_size'],
            d_model=config['d_model'],
            num_tokens=config['num_tokens'],
        )

        # Residual accumulation strength
        self.alpha = nn.Parameter(torch.tensor(config['alpha_init']))

        self.state_size = config['state_size']
        self.max_passes = config['max_passes']

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable:,}")

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    def format_for_training(self, question, answer):
        """Format question + answer for teacher-forced training."""
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(answer)}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def format_for_generation(self, question):
        """Format question for generation (no answer)."""
        messages = [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def think_and_get_loss(self, question, answer, num_passes=None, scale=1.0):
        """
        Forward pass with teacher forcing.
        Returns loss that backprops through state vector into perceiver.

        This is the key: gradients flow from answer loss → perceiver.
        """
        if num_passes is None:
            num_passes = self.max_passes

        # Tokenize full sequence (question + answer)
        full_text = self.format_for_training(question, answer)
        tokens = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['max_seq_len'],
            padding=False,
        ).to(self.device)

        input_ids = tokens.input_ids

        # Find where the answer starts (after "assistant" marker)
        full_text_no_answer = self.format_for_generation(question)
        prompt_len = len(self.tokenizer(full_text_no_answer, add_special_tokens=False).input_ids)

        # Get embeddings
        embeds = self.transformer.get_input_embeddings()(input_ids)

        # Initialize state
        state = torch.zeros(1, self.state_size, device=self.device)

        # Multiple thinking passes
        for pass_num in range(num_passes):
            # Inject state as pseudo-tokens (with scale warmup)
            state_tokens = self.injector(state, scale=scale, dtype=embeds.dtype)

            # Combine: [state_tokens, prompt+answer_tokens]
            combined_embeds = torch.cat([state_tokens, embeds], dim=1)

            # Forward through transformer
            outputs = self.transformer(
                inputs_embeds=combined_embeds,
                output_hidden_states=True,
                use_cache=False,
            )

            # Get all hidden states (skip embedding layer)
            hidden_states = list(outputs.hidden_states[1:])

            # Update state via perceiver (this is where gradients flow!)
            delta = self.perceiver(hidden_states, pass_num=pass_num)
            state = state + self.alpha * delta

        # Compute loss on answer tokens only
        logits = outputs.logits  # (1, state_tokens + seq_len, vocab)

        # Shift for next-token prediction
        num_state_tokens = state_tokens.size(1)
        # Answer positions in the combined sequence
        answer_start = num_state_tokens + prompt_len - 1  # -1 for shift
        answer_end = logits.size(1) - 1

        if answer_end <= answer_start:
            # Answer too short or truncated
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Get logits and targets for answer region
        answer_logits = logits[0, answer_start:answer_end, :]
        answer_targets = input_ids[0, prompt_len:]

        # Truncate to same length
        min_len = min(answer_logits.size(0), answer_targets.size(0))
        if min_len == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        answer_logits = answer_logits[:min_len]
        answer_targets = answer_targets[:min_len]

        # Cross-entropy loss
        loss = F.cross_entropy(answer_logits, answer_targets)

        return loss

    @torch.no_grad()
    def generate(self, question, max_tokens=32):
        """Generate answer after thinking."""
        self.eval()

        # Think first
        prompt = self.format_for_generation(question)
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=128).to(self.device)
        embeds = self.transformer.get_input_embeddings()(tokens.input_ids)

        state = torch.zeros(1, self.state_size, device=self.device)

        for pass_num in range(self.max_passes):
            state_tokens = self.injector(state, scale=1.0, dtype=embeds.dtype)
            combined = torch.cat([state_tokens, embeds], dim=1)
            outputs = self.transformer(inputs_embeds=combined, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states[1:])
            delta = self.perceiver(hidden_states, pass_num=pass_num)
            state = state + self.alpha * delta

        # Generate with final state
        state_tokens = self.injector(state, scale=1.0, dtype=embeds.dtype)
        combined = torch.cat([state_tokens, embeds], dim=1)

        # Create attention mask
        attn_mask = torch.ones(1, combined.size(1), device=self.device)

        outputs = self.transformer.generate(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response


# ============================================================================
# Training Functions
# ============================================================================

def extract_answer(text):
    """Extract numerical answer from response."""
    # Look for boxed answer first
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()
    # Otherwise find last number
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        return nums[-1]
    return None


def check_answer(pred, gold):
    """Check if prediction matches gold answer."""
    if pred is None:
        return False
    try:
        pred_num = float(pred.replace(',', ''))
        gold_num = float(str(gold).replace(',', ''))
        return abs(pred_num - gold_num) < 0.01
    except:
        return str(pred).strip() == str(gold).strip()


def load_data(filepath):
    """Load JSONL data."""
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def get_scale(step, config):
    """Compute current state scale (warmup)."""
    warmup = config['scale_warmup_steps']
    if step >= warmup:
        return config['scale_end']
    progress = step / warmup
    return config['scale_start'] + progress * (config['scale_end'] - config['scale_start'])


def train_level(model, data, config, level_name, global_step=0):
    """Train on one curriculum level."""
    print(f"\n{'='*60}")
    print(f"Training: {level_name}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'],
        weight_decay=0.01,
    )

    # Split data
    random.shuffle(data)
    n_train = int(0.9 * len(data))
    train_data, eval_data = data[:n_train], data[n_train:]
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    best_acc = 0.0

    for epoch in range(config['epochs_per_level']):
        model.train()
        random.shuffle(train_data)

        epoch_loss = 0.0
        num_batches = 0
        correct, total = 0, 0

        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"E{epoch+1}")

        for i, item in pbar:
            try:
                # Current scale (warmup)
                scale = get_scale(global_step, config)

                # Forward pass with answer loss
                loss = model.think_and_get_loss(
                    item['question'],
                    item['answer'],
                    scale=scale
                )

                # Accumulate gradients
                loss = loss / config['grad_accum_steps']
                loss.backward()

                epoch_loss += loss.item() * config['grad_accum_steps']
                num_batches += 1

                # Update weights
                if (i + 1) % config['grad_accum_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                # Periodic evaluation during training
                if (i + 1) % config['log_every'] == 0:
                    with torch.no_grad():
                        output = model.generate(item['question'])
                        pred = extract_answer(output)
                        if check_answer(pred, item['answer']):
                            correct += 1
                        total += 1

                    pbar.set_postfix(
                        loss=f"{epoch_loss/num_batches:.3f}",
                        acc=f"{correct/max(total,1):.1%}",
                        scale=f"{scale:.2f}",
                        alpha=f"{model.alpha.item():.3f}"
                    )

            except Exception as e:
                print(f"\nError on item {i}: {e}")
                continue

        # Epoch evaluation
        model.eval()
        eval_correct = 0
        eval_total = min(100, len(eval_data))

        print(f"\nEvaluating on {eval_total} examples...")
        with torch.no_grad():
            for item in tqdm(eval_data[:eval_total], desc="Eval"):
                try:
                    output = model.generate(item['question'])
                    pred = extract_answer(output)
                    if check_answer(pred, item['answer']):
                        eval_correct += 1
                except:
                    pass

        eval_acc = eval_correct / eval_total
        avg_loss = epoch_loss / max(num_batches, 1)

        print(f"Epoch {epoch+1}: loss={avg_loss:.3f}, train_acc={correct/max(total,1):.1%}, eval_acc={eval_acc:.1%}")
        print(f"  scale={get_scale(global_step, config):.2f}, alpha={model.alpha.item():.3f}")

        # Save best
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save({
                'model_state': model.state_dict(),
                'config': config,
                'level': level_name,
                'epoch': epoch,
                'eval_acc': eval_acc,
                'global_step': global_step,
            }, f"checkpoints/{level_name}_best.pt")
            print(f"  New best! Saved checkpoint.")

        # Sample outputs
        print("\nSample outputs:")
        for item in eval_data[:3]:
            with torch.no_grad():
                out = model.generate(item['question'], max_tokens=48)
            pred = extract_answer(out)
            gold = item['answer']
            status = "OK" if check_answer(pred, gold) else "WRONG"
            print(f"  Q: {item['question'][:50]}...")
            print(f"  A: {out[:60]}...")
            print(f"  Pred: {pred}, Gold: {gold} [{status}]")
            print()

    return best_acc, global_step


def main():
    print("="*60)
    print("Mycelium v18 - End-to-End Training")
    print(f"Started: {datetime.now()}")
    print("="*60)

    # Load HF token
    if os.path.exists(CONFIG['hf_token_file']):
        with open(CONFIG['hf_token_file']) as f:
            CONFIG['hf_token'] = f.read().strip()
        login(token=CONFIG['hf_token'])
        print("Logged in to HuggingFace")
    else:
        raise ValueError(f"HF token not found at {CONFIG['hf_token_file']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    # Create model
    model = ThinkingModel(CONFIG).to(device)

    # Sanity check
    print("\nSanity check (before training):")
    model.eval()
    with torch.no_grad():
        test_q = "What is 2 + 3?"
        out = model.generate(test_q)
        print(f"  Q: {test_q}")
        print(f"  A: {out}")

    # Curriculum levels
    levels = [
        ("L0_single", "data/curriculum/level_0_single_step.jsonl"),
        ("L1_two", "data/curriculum/level_1_two_step.jsonl"),
        ("L2_three", "data/curriculum/level_2_three_step.jsonl"),
    ]

    global_step = 0
    results = {}

    for name, path in levels:
        if not os.path.exists(path):
            print(f"\nSkipping {name} - file not found: {path}")
            continue

        data = load_data(path)
        best_acc, global_step = train_level(model, data, CONFIG, name, global_step)
        results[name] = best_acc
        print(f"\n{name} complete. Best accuracy: {best_acc:.1%}")

        # Save final checkpoint for this level
        torch.save({
            'model_state': model.state_dict(),
            'config': CONFIG,
            'level': name,
            'final': True,
            'global_step': global_step,
        }, f"checkpoints/{name}_final.pt")

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    for name, acc in results.items():
        print(f"  {name}: {acc:.1%}")
    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
