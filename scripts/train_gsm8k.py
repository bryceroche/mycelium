"""
GSM8K word problem training with strategy channel v3.

Warm-started from three-step 73.6% checkpoint.
Parses step-by-step solutions for probe targets from <<expr=val>> annotations.
Fixed 3 thinking passes. Variable-step problems mapped to 3 passes:
  - 2-step problems: last target repeated for pass 3
  - 3-step problems: one target per pass
  - 4+ step problems: first 3 targets used

Base model GSM8K baseline: 4-5% (raw completion, no chain-of-thought)
Target: >10% (any improvement = architecture generalizes)
Paper territory: >20%
"""

import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/ubuntu/mycelium')

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.state_conditioned_lora_v3 import StateConditionedLoRA
from src.additive_lora import AdditiveLoRAManager


def parse_gsm8k_intermediates(answer_text):
    """
    Parse intermediate values from GSM8K step-by-step solutions.

    GSM8K format: "Natalia sold 48/2 = <<48/2=24>>24 clips in May."
    The <<expr=value>> annotations contain computed intermediates.
    Final answer after ####.

    Returns list of float intermediates and the final answer.
    """
    # Extract all <<...=value>> intermediates
    intermediates = []
    for match in re.finditer(r'<<[^>]*?=([^>]+)>>', answer_text):
        try:
            val = float(match.group(1))
            intermediates.append(val)
        except ValueError:
            continue

    # Extract final answer after ####
    final_match = re.search(r'####\s*(.+)', answer_text)
    final = None
    if final_match:
        try:
            final = float(final_match.group(1).strip().replace(',', ''))
        except ValueError:
            pass

    return intermediates, final


class GSM8KDataset(Dataset):
    """GSM8K with parsed probe targets, mapped to 3 passes."""

    def __init__(self, split='train', max_samples=None):
        ds = load_dataset("openai/gsm8k", "main", split=split)

        self.samples = []
        skipped = 0

        for ex in ds:
            intermediates, final = parse_gsm8k_intermediates(ex['answer'])

            if final is None or len(intermediates) < 2:
                skipped += 1
                continue

            # Map variable steps to exactly 3 probe targets
            if len(intermediates) == 2:
                targets = [intermediates[0], intermediates[1], intermediates[1]]
            elif len(intermediates) == 3:
                targets = intermediates[:3]
            else:  # 4+
                targets = intermediates[:3]

            # Format answer as string (integer if whole number)
            if final == int(final):
                answer_str = str(int(final))
            else:
                answer_str = str(final)

            self.samples.append({
                'question': ex['question'],
                'answer': answer_str,
                'final': final,
                'targets': targets,  # 3 probe targets
                'num_steps': len(intermediates),
            })

            if max_samples and len(self.samples) >= max_samples:
                break

        print(f"Loaded {len(self.samples)} GSM8K problems ({skipped} skipped, split={split})")

        # Stats
        step_counts = [s['num_steps'] for s in self.samples]
        if step_counts:
            from collections import Counter
            dist = Counter(step_counts)
            print(f"  Step distribution: {dict(sorted(dist.items()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def gsm8k_collate(batch):
    """Custom collate since targets is a list."""
    return {
        'question': [s['question'] for s in batch],
        'answer': [s['answer'] for s in batch],
        'final': [s['final'] for s in batch],
        'targets': [s['targets'] for s in batch],
    }


class ThinkingModelV3(nn.Module):
    """Same architecture as arithmetic — no changes for word problems."""

    def __init__(self, model_name='unsloth/Llama-3.2-1B'):
        super().__init__()

        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for param in self.transformer.parameters():
            param.requires_grad = False

        self.d_model = self.transformer.config.hidden_size
        self.num_layers = self.transformer.config.num_hidden_layers
        num_kv_heads = self.transformer.config.num_key_value_heads
        head_dim = self.d_model // self.transformer.config.num_attention_heads
        d_kv = num_kv_heads * head_dim

        self.state_size = 64
        self.strategy_size = 512
        self.state_radius = (self.state_size ** 0.5)

        self.compressor = Compressor(
            num_transformer_layers=self.num_layers,
            d_transformer=self.d_model,
            d_perceiver=1024,
            num_queries=4,
            num_perceiver_layers=7,
            state_size=self.state_size,
            strategy_size=self.strategy_size,
        )

        self.lora = StateConditionedLoRA(
            d_model=self.d_model,
            d_kv=d_kv,
            state_size=self.state_size,
            strategy_size=self.strategy_size,
            rank=4,
            num_layers=self.num_layers,
        )

        self.probe_head = nn.Linear(self.state_size, 1)

    def thinking_pass(self, input_ids, attention_mask, state, strategy, pass_num):
        lora_mods = self.lora(state, strategy)
        manager = AdditiveLoRAManager(self.transformer)
        manager.apply(lora_mods)
        try:
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
        finally:
            manager.remove()

        state_delta, new_strategy = self.compressor(hidden_states, pass_num)
        new_state = F.normalize(state + state_delta, dim=-1) * self.state_radius
        return new_state, new_strategy

    def compute_answer_loss(self, state, strategy, prompt_ids, prompt_mask, answer_ids):
        lora_mods = self.lora(state, strategy)
        manager = AdditiveLoRAManager(self.transformer)
        manager.apply(lora_mods)
        try:
            full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            # Extend attention mask for answer tokens
            answer_mask = torch.ones_like(answer_ids)
            full_mask = torch.cat([prompt_mask, answer_mask], dim=1)
            outputs = self.transformer(
                input_ids=full_ids, attention_mask=full_mask, use_cache=False,
            )
        finally:
            manager.remove()

        prompt_len = prompt_ids.size(1)
        logits = outputs.logits[:, prompt_len - 1:-1, :]
        targets = answer_ids

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        return loss

    def forward_train(self, questions, answers, targets_list, num_passes=3):
        """Training forward for word problems. Answer loss only (probe disabled)."""
        # Format: "Problem: {question}\nAnswer:"
        prompts = [f"Problem: {q}\nAnswer:" for q in questions]

        inputs = self.tokenizer(
            prompts, return_tensors='pt', padding=True,
            truncation=True, max_length=256,
        )
        input_ids = inputs['input_ids'].to(self.transformer.device)
        attention_mask = inputs['attention_mask'].to(self.transformer.device)

        answer_texts = [f" {a}" for a in answers]
        answer_inputs = self.tokenizer(
            answer_texts, return_tensors='pt', padding=True,
            add_special_tokens=False,
        )
        answer_ids = answer_inputs['input_ids'].to(self.transformer.device)

        batch_size = input_ids.size(0)
        device = input_ids.device

        state = torch.randn(batch_size, self.state_size, device=device)
        state = F.normalize(state, dim=-1) * self.state_radius
        strategy = torch.zeros(batch_size, self.strategy_size, device=device)

        all_states = []
        for pass_num in range(num_passes):
            state, strategy = self.thinking_pass(
                input_ids, attention_mask, state, strategy, pass_num
            )
            all_states.append(state)

        # Answer loss (deep supervision)
        answer_loss = torch.tensor(0.0, device=device)
        for i, s in enumerate(all_states):
            weight = (i + 1) / num_passes
            pass_loss = self.compute_answer_loss(s, strategy, input_ids, attention_mask, answer_ids)
            answer_loss = answer_loss + weight * pass_loss
        answer_loss = answer_loss / ((num_passes + 1) / 2)

        # Probe disabled for GSM8K — intermediate values range 0-100k+,
        # linear probe can't handle that. Answer loss only.
        total_loss = answer_loss
        return total_loss, answer_loss.item(), 0.0, all_states


def evaluate(model, eval_dataset, device, num_passes=3):
    """Evaluate by generating and extracting final answer."""
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for i in range(0, len(eval_dataset), 4):  # smaller batch for longer prompts
            batch_samples = eval_dataset.samples[i:i+4]
            questions = [s['question'] for s in batch_samples]
            golds = [s['final'] for s in batch_samples]

            prompts = [f"Problem: {q}\nAnswer:" for q in questions]
            inputs = model.tokenizer(
                prompts, return_tensors='pt', padding=True,
                truncation=True, max_length=256,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            bs = input_ids.size(0)

            state = torch.randn(bs, model.state_size, device=device)
            state = F.normalize(state, dim=-1) * model.state_radius
            strategy = torch.zeros(bs, model.strategy_size, device=device)

            for pass_num in range(num_passes):
                state, strategy = model.thinking_pass(
                    input_ids, attention_mask, state, strategy, pass_num
                )

            lora_mods = model.lora(state, strategy)
            manager = AdditiveLoRAManager(model.transformer)
            manager.apply(lora_mods)
            try:
                generated = model.transformer.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=50, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            for j in range(bs):
                prompt_len = input_ids[j].size(0)
                gen_text = model.tokenizer.decode(
                    generated[j][prompt_len:], skip_special_tokens=True
                ).strip()

                # Extract number: try multiple patterns
                pred = None
                # Pattern 1: first number in text
                num_match = re.search(r'-?\d+\.?\d*', gen_text)
                if num_match:
                    try:
                        pred = float(num_match.group())
                        if pred == int(pred):
                            pred = int(pred)
                    except ValueError:
                        pass

                gold = golds[j]
                if gold == int(gold):
                    gold = int(gold)

                if pred == gold:
                    correct += 1
                total += 1

    return 100.0 * correct / total if total > 0 else 0.0


def train():
    print("GSM8K WORD PROBLEMS: Strategy Channel v3 + Answer Loss")
    print("=" * 60)
    print("Warm start from three-step 73.6% checkpoint")
    print("3 thinking passes, probe targets from <<expr=val>> annotations")
    print("Base model baseline: 4-5%. Target: >10%. Paper: >20%.")
    print("=" * 60)

    device = torch.device('cuda')

    model = ThinkingModelV3()
    model.compressor = model.compressor.to(device)
    model.lora = model.lora.to(device)
    model.probe_head = model.probe_head.to(device)

    # Warm start from three-step checkpoint
    ckpt_path = '/home/ubuntu/mycelium/checkpoints/three_step_best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.compressor.load_state_dict(ckpt['compressor'])
    model.lora.load_state_dict(ckpt['lora'])
    model.probe_head.load_state_dict(ckpt['probe_head'])
    print(f"Loaded three-step checkpoint: epoch {ckpt['epoch']}, accuracy {ckpt['accuracy']:.1f}%")

    # Load GSM8K
    train_dataset = GSM8KDataset(split='train')
    eval_dataset = GSM8KDataset(split='test', max_samples=500)
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        collate_fn=gsm8k_collate,
    )

    # Fine-tuning LRs (lower for transfer). Probe disabled for GSM8K.
    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 3e-5},
        {'params': list(model.lora.A_templates) + list(model.lora.B_templates), 'lr': 3e-4},
        {'params': list(model.lora.state_to_scales.parameters()), 'lr': 3e-4},
    ])

    trainable = (list(model.compressor.parameters()) +
                 list(model.lora.parameters()))
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    num_epochs = 3
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_answer_loss = 0.0
        epoch_probe_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()

            total_loss, ans_loss, prb_loss, _ = model.forward_train(
                batch['question'], batch['answer'], batch['targets'], num_passes=3
            )
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            epoch_answer_loss += ans_loss
            epoch_probe_loss += prb_loss
            epoch_total_loss += total_loss.item()
            num_batches += 1

        avg_total = epoch_total_loss / num_batches
        avg_answer = epoch_answer_loss / num_batches
        avg_probe = epoch_probe_loss / num_batches

        accuracy = evaluate(model, eval_dataset, device, num_passes=3)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'lora': model.lora.state_dict(),
                'probe_head': model.probe_head.state_dict(),
                'accuracy': accuracy,
            }, 'checkpoints/gsm8k_best.pt')
            print(f"  -> Saved new best checkpoint")

        print(f"Epoch {epoch+1}: total={avg_total:.4f} answer={avg_answer:.4f} probe={avg_probe:.6f} | Acc={accuracy:.1f}% (best={best_accuracy:.1f}%)")

    print(f"\nFinal: {best_accuracy:.1f}% (base model baseline: 4-5%)")


if __name__ == '__main__':
    train()
