#!/usr/bin/env python3
"""
Training script for Asymmetric Hourglass v19.

Architecture:
  DECOMPRESSOR (MLP, 1.3M): 64 floats → bias — EASY job
  TRANSFORMER (Llama 16L): Pristine, untouched
  COMPRESSOR (7L, 120M): 16 layers → 64 floats — HARD job

The gradient chain:
  answer_loss → logits → [bias + problem] → decompressor → state → compressor
"""

import os
import json
import random
import re
import math
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

    # Architecture (Asymmetric Hourglass)
    "num_transformer_layers": 16,
    "d_model": 2048,
    "state_size": 64,        # Hypersphere radius = √64 ≈ 8.0

    # Decompressor (lightweight MLP)
    "d_decompressor_hidden": 512,  # 64 → 512 → 512 → 2048

    # Compressor (heavy perceiver)
    "num_queries": 4,
    "num_compressor_layers": 7,
    "d_compressor": 1024,

    # Thinking
    "max_passes": 3,

    # State scale warmup (for bias injection)
    "scale_start": 0.01,
    "scale_end": 1.0,
    "scale_warmup_steps": 500,

    # Training
    "batch_size": 4,
    "grad_accum_steps": 4,
    "learning_rate": 1e-4,
    "epochs_per_level": 10,
    "max_seq_len": 256,
    "max_answer_tokens": 32,

    # Logging
    "log_every": 20,
    "eval_every": 100,
}


# ============================================================================
# Architecture Components
# ============================================================================

class Decompressor(nn.Module):
    """
    Lightweight MLP that projects 64-float state into input bias.
    EASY job: just project faithfully.

    Parameters: ~1.3M
    """
    def __init__(self, state_size=64, d_model=2048, d_hidden=512, max_passes=20):
        super().__init__()
        self.state_size = state_size
        self.d_model = d_model

        # Pass embedding
        self.pass_embed = nn.Embedding(max_passes, state_size)

        # Simple MLP: 64 → 512 → 512 → 2048
        self.mlp = nn.Sequential(
            nn.Linear(state_size, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

        # CRITICAL: Initialize FINAL layer to output zero initially
        # Random bias destroys Llama's output (even at 10% scale!)
        # Keep intermediate layers with small weights for gradient flow
        for module in self.mlp[:-1]:  # All but last layer
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
        # Final layer outputs zero → no bias initially
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, state, pass_num, scale=1.0):
        """
        Args:
            state: (batch, 64)
            pass_num: int
            scale: float (state warmup)
        Returns:
            bias: (batch, 1, 2048) to ADD to input embeddings
        """
        device = state.device
        pass_idx = torch.tensor(pass_num, device=device)
        pass_context = self.pass_embed(pass_idx)
        x = state + pass_context

        bias = self.mlp(x)
        bias = self.output_norm(bias)
        return bias.unsqueeze(1) * scale


class Compressor(nn.Module):
    """
    7-layer Perceiver that compresses all 16 transformer layers into 64 floats.
    HARD job: must select what matters.

    Parameters: ~120M
    """
    def __init__(self, num_transformer_layers=16, d_transformer=2048,
                 d_internal=1024, num_queries=4, num_layers=7,
                 state_size=64, max_passes=20):
        super().__init__()
        self.num_transformer_layers = num_transformer_layers
        self.num_queries = num_queries
        self.state_size = state_size

        # Learned queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_internal) * 0.02)
        self.pass_embed = nn.Embedding(max_passes, d_internal)

        # Layer gate
        self.layer_gate = nn.Sequential(
            nn.Linear(d_internal, 64),
            nn.ReLU(),
            nn.Linear(64, num_transformer_layers),
        )

        # Input projection
        self.input_project = nn.Linear(d_transformer, d_internal)

        # 7-layer perceiver
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(d_internal, num_heads=8, batch_first=True),
                'cross_norm': nn.LayerNorm(d_internal),
                'self_attn': nn.MultiheadAttention(d_internal, num_heads=8, batch_first=True),
                'self_norm': nn.LayerNorm(d_internal),
                'ffn': nn.Sequential(
                    nn.Linear(d_internal, d_internal * 4),
                    nn.GELU(),
                    nn.Linear(d_internal * 4, d_internal),
                ),
                'ffn_norm': nn.LayerNorm(d_internal),
            })
            for _ in range(num_layers)
        ])

        # Output projection: 1024 → 16 per query → 64 total
        self.project_out = nn.Linear(d_internal, state_size // num_queries)

        # Initialize layer gate to uniform
        nn.init.zeros_(self.layer_gate[0].weight)
        nn.init.zeros_(self.layer_gate[0].bias)
        nn.init.zeros_(self.layer_gate[2].weight)
        nn.init.zeros_(self.layer_gate[2].bias)

    def forward(self, all_layer_hidden_states, pass_num):
        """
        Args:
            all_layer_hidden_states: list of 16 tensors (batch, seq, 2048)
            pass_num: int
        Returns:
            state_delta: (batch, 64)
        """
        batch_size = all_layer_hidden_states[0].size(0)
        device = all_layer_hidden_states[0].device

        # Pass conditioning
        pass_idx = torch.tensor(pass_num, device=device)
        pass_context = self.pass_embed(pass_idx)

        queries = (self.queries + pass_context).unsqueeze(0).expand(batch_size, -1, -1)

        # Layer gate
        layer_logits = self.layer_gate(pass_context)
        layer_weights = F.softmax(layer_logits, dim=-1)

        # Weighted combination of layers (keep full sequence)
        stacked = torch.stack([h.float() for h in all_layer_hidden_states], dim=0)
        weights = layer_weights.view(self.num_transformer_layers, 1, 1, 1)
        combined = (stacked * weights).sum(dim=0)  # (batch, seq, d_model)

        # Project to perceiver dimension (KEEP full sequence for cross-attention)
        kv = self.input_project(combined.to(dtype=self.input_project.weight.dtype))  # (batch, seq, d_internal)

        queries = queries.to(dtype=kv.dtype)

        # 7 layers of processing
        for layer in self.layers:
            attended, _ = layer['cross_attn'](queries, kv, kv)
            queries = layer['cross_norm'](queries + attended)
            refined, _ = layer['self_attn'](queries, queries, queries)
            queries = layer['self_norm'](queries + refined)
            queries = layer['ffn_norm'](queries + layer['ffn'](queries))

        # Project to 64 floats
        state_chunks = self.project_out(queries)
        return state_chunks.flatten(start_dim=1)


class ThinkingModel(nn.Module):
    """
    Asymmetric Hourglass: DECOMPRESSOR → Llama → COMPRESSOR

    State lives on hypersphere of radius √64.
    Each pass rotates the state via: state = normalize(state + delta) * √64
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

        # Freeze transformer
        for p in self.transformer.parameters():
            p.requires_grad = False

        # Decompressor: state → bias (EASY, 1.3M params)
        self.decompressor = Decompressor(
            state_size=config['state_size'],
            d_model=config['d_model'],
            d_hidden=config['d_decompressor_hidden'],
            max_passes=config['max_passes'] + 5,
        )

        # Compressor: all layers → state (HARD, 120M params)
        self.compressor = Compressor(
            num_transformer_layers=config['num_transformer_layers'],
            d_transformer=config['d_model'],
            d_internal=config['d_compressor'],
            num_queries=config['num_queries'],
            num_layers=config['num_compressor_layers'],
            state_size=config['state_size'],
            max_passes=config['max_passes'] + 5,
        )

        self.state_size = config['state_size']
        self.state_radius = math.sqrt(config['state_size'])
        self.max_passes = config['max_passes']

        d_params = sum(p.numel() for p in self.decompressor.parameters())
        c_params = sum(p.numel() for p in self.compressor.parameters())
        trainable = d_params + c_params
        print(f"Decompressor: {d_params:,} ({d_params/1e6:.2f}M)")
        print(f"Compressor: {c_params:,} ({c_params/1e6:.1f}M)")
        print(f"Total trainable: {trainable:,} ({trainable/1e6:.1f}M)")
        print(f"Asymmetry ratio: {c_params/d_params:.0f}x")
        print(f"State hypersphere radius: {self.state_radius:.2f}")

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    def init_state(self, batch_size=1):
        """Initialize state on hypersphere."""
        state = torch.randn(batch_size, self.state_size, device=self.device)
        return F.normalize(state, dim=-1) * self.state_radius

    def update_state(self, state, delta):
        """Rotate state on hypersphere."""
        return F.normalize(state + delta, dim=-1) * self.state_radius

    def format_for_training(self, question, answer):
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(answer)}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def format_for_generation(self, question):
        messages = [{"role": "user", "content": question}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def think_and_get_loss(self, question, answer, num_passes=None, scale=1.0):
        """Forward pass with teacher forcing and bias injection."""
        if num_passes is None:
            num_passes = self.max_passes

        full_text = self.format_for_training(question, answer)
        tokens = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['max_seq_len'],
            padding=False,
        ).to(self.device)

        input_ids = tokens.input_ids
        full_text_no_answer = self.format_for_generation(question)
        prompt_len = len(self.tokenizer(full_text_no_answer, add_special_tokens=False).input_ids)

        # Get base embeddings
        embeds = self.transformer.get_input_embeddings()(input_ids)

        # Initialize state on hypersphere
        state = self.init_state(batch_size=1)

        cos_sims = []

        for pass_num in range(num_passes):
            # DECOMPRESS: state → bias
            bias = self.decompressor(state, pass_num, scale=scale)
            bias = bias.to(dtype=embeds.dtype)

            # Add bias to ALL positions (not prepend)
            input_embeds = embeds + bias

            # Forward through transformer
            outputs = self.transformer(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
                use_cache=False,
            )

            # COMPRESS: all 16 layers → delta
            hidden_states = list(outputs.hidden_states[1:])
            delta = self.compressor(hidden_states, pass_num=pass_num)

            # Update state on hypersphere
            old_state = state
            state = self.update_state(state, delta)

            # Track rotation
            cos_sim = F.cosine_similarity(old_state, state, dim=-1).item()
            cos_sims.append(cos_sim)

        # Compute loss on answer tokens
        logits = outputs.logits
        answer_start = prompt_len - 1
        answer_end = logits.size(1) - 1

        if answer_end <= answer_start:
            return torch.tensor(0.0, device=self.device, requires_grad=True), cos_sims

        answer_logits = logits[0, answer_start:answer_end, :]
        answer_targets = input_ids[0, prompt_len:]

        min_len = min(answer_logits.size(0), answer_targets.size(0))
        if min_len == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), cos_sims

        answer_logits = answer_logits[:min_len]
        answer_targets = answer_targets[:min_len]

        loss = F.cross_entropy(answer_logits, answer_targets)
        return loss, cos_sims

    @torch.no_grad()
    def generate(self, question, max_tokens=32):
        self.eval()

        prompt = self.format_for_generation(question)
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=128).to(self.device)
        embeds = self.transformer.get_input_embeddings()(tokens.input_ids)

        # Initialize on hypersphere
        state = self.init_state(batch_size=1)

        # Think
        for pass_num in range(self.max_passes):
            bias = self.decompressor(state, pass_num, scale=1.0)
            bias = bias.to(dtype=embeds.dtype)
            input_embeds = embeds + bias

            outputs = self.transformer(inputs_embeds=input_embeds, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states[1:])
            delta = self.compressor(hidden_states, pass_num=pass_num)
            state = self.update_state(state, delta)

        # Generate with final state (use last pass_num for consistency)
        bias = self.decompressor(state, pass_num=self.max_passes - 1, scale=1.0)
        bias = bias.to(dtype=embeds.dtype)
        input_embeds = embeds + bias

        attn_mask = torch.ones(1, input_embeds.size(1), device=self.device)

        outputs = self.transformer.generate(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        return response


# ============================================================================
# Training Functions
# ============================================================================

def extract_answer(text):
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).strip()
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        return nums[-1]
    return None


def check_answer(pred, gold):
    if pred is None:
        return False
    try:
        pred_num = float(pred.replace(',', ''))
        gold_num = float(str(gold).replace(',', ''))
        return abs(pred_num - gold_num) < 0.01
    except:
        return str(pred).strip() == str(gold).strip()


def load_data(filepath):
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def get_scale(step, config):
    warmup = config['scale_warmup_steps']
    if step >= warmup:
        return config['scale_end']
    progress = step / warmup
    return config['scale_start'] + progress * (config['scale_end'] - config['scale_start'])


def train_level(model, data, config, level_name, global_step=0):
    print(f"\n{'='*60}")
    print(f"Training: {level_name} (Asymmetric Hourglass)")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'],
        weight_decay=0.01,
    )

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
        avg_cos_sim = 0.0
        cos_sim_count = 0

        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_data), total=len(train_data), desc=f"E{epoch+1}")

        for i, item in pbar:
            try:
                scale = get_scale(global_step, config)

                loss, cos_sims = model.think_and_get_loss(
                    item['question'],
                    item['answer'],
                    scale=scale
                )

                loss = loss / config['grad_accum_steps']
                loss.backward()

                epoch_loss += loss.item() * config['grad_accum_steps']
                num_batches += 1

                if cos_sims:
                    avg_cos_sim += sum(cos_sims) / len(cos_sims)
                    cos_sim_count += 1

                if (i + 1) % config['grad_accum_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (i + 1) % config['log_every'] == 0:
                    with torch.no_grad():
                        output = model.generate(item['question'])
                        pred = extract_answer(output)
                        if check_answer(pred, item['answer']):
                            correct += 1
                        total += 1

                    avg_rot = avg_cos_sim / max(cos_sim_count, 1)
                    pbar.set_postfix(
                        loss=f"{epoch_loss/num_batches:.3f}",
                        acc=f"{correct/max(total,1):.1%}",
                        scale=f"{scale:.2f}",
                        cos=f"{avg_rot:.3f}"
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
        avg_rotation = avg_cos_sim / max(cos_sim_count, 1)

        print(f"Epoch {epoch+1}: loss={avg_loss:.3f}, train_acc={correct/max(total,1):.1%}, eval_acc={eval_acc:.1%}")
        print(f"  scale={get_scale(global_step, config):.2f}, avg_cos_sim={avg_rotation:.3f}")

        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save({
                'model_state': model.state_dict(),
                'config': config,
                'level': level_name,
                'epoch': epoch,
                'eval_acc': eval_acc,
                'global_step': global_step,
            }, f"checkpoints/{level_name}_hourglass_best.pt")
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
    print("Mycelium v19 - Asymmetric Hourglass Training")
    print(f"Decompressor: MLP (~1.3M) | Compressor: 7L Perceiver (~120M)")
    print(f"State radius: √{CONFIG['state_size']} = {math.sqrt(CONFIG['state_size']):.2f}")
    print(f"Started: {datetime.now()}")
    print("="*60)

    if os.path.exists(CONFIG['hf_token_file']):
        with open(CONFIG['hf_token_file']) as f:
            CONFIG['hf_token'] = f.read().strip()
        login(token=CONFIG['hf_token'])
        print("Logged in to HuggingFace")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    model = ThinkingModel(CONFIG).to(device)

    # Sanity check
    print("\nSanity check (before training):")
    model.eval()
    with torch.no_grad():
        test_q = "What is 2 + 3?"
        out = model.generate(test_q)
        print(f"  Q: {test_q}")
        print(f"  A: {out}")

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

    print("\n" + "="*60)
    print("TRAINING COMPLETE (Asymmetric Hourglass)")
    print("="*60)
    for name, acc in results.items():
        print(f"  {name}: {acc:.1%}")
    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
