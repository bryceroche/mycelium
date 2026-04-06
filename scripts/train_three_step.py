"""
Three-step arithmetic training with strategy channel v3.

Warm-started from two-step 85.4% checkpoint (strategy_v3_85pct.pt).
Three passes for three operations: ((a op1 b) op2 c) op3 d

Target: >52% (previous best was 52% on SmolLM2, 22% on Llama without answer loss)
Ceiling: 92.4%^3 = 78.9% (based on two-step effective per-step)
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/ubuntu/mycelium')

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.state_conditioned_lora_v3 import StateConditionedLoRA
from src.additive_lora import AdditiveLoRAManager


class ThreeStepArithmeticDataset(Dataset):
    """((a op1 b) op2 c) op3 d — three chained operations."""

    def __init__(self, num_samples=10000, seed=42):
        random.seed(seed)
        self.samples = []

        for _ in range(num_samples):
            a = random.randint(2, 50)
            b = random.randint(2, 50)
            c = random.randint(2, 50)
            d = random.randint(2, 50)

            # Step 1
            if random.random() < 0.5:
                op1 = '+'
                v1 = a + b
            else:
                a = b * random.randint(2, 10)
                op1 = '/'
                v1 = a // b

            # Step 2
            if random.random() < 0.5:
                op2 = '+'
                v2 = v1 + c
            else:
                op2 = '-'
                v2 = v1 - c

            # Step 3
            if random.random() < 0.5:
                op3 = '+'
                v3 = v2 + d
            else:
                op3 = '-'
                v3 = v2 - d

            problem = f"(({a} {op1} {b}) {op2} {c}) {op3} {d} ="
            self.samples.append({
                'problem': problem,
                'answer': str(v3),
                'v1': v1,  # after step 1
                'v2': v2,  # after step 2
                'v3': v3,  # final answer
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ThinkingModelV3(nn.Module):
    """ThinkingModel with strategy channel — same architecture as two-step."""

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
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
        finally:
            manager.remove()

        state_delta, new_strategy = self.compressor(hidden_states, pass_num)

        new_state = state + state_delta
        new_state = F.normalize(new_state, dim=-1) * self.state_radius

        return new_state, new_strategy

    def compute_answer_loss(self, state, strategy, prompt_ids, answer_ids):
        lora_mods = self.lora(state, strategy)
        manager = AdditiveLoRAManager(self.transformer)
        manager.apply(lora_mods)

        try:
            full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            outputs = self.transformer(input_ids=full_ids, use_cache=False)
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

    def forward_train(self, problems, answers, v1_t, v2_t, v3_t, num_passes=3):
        """Three thinking passes + answer loss + probe loss at each step."""
        inputs = self.tokenizer(
            problems, return_tensors='pt', padding=True,
            truncation=True, max_length=128,
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

        # Random init on hypersphere (robust eval showed zero variance)
        state = torch.randn(batch_size, self.state_size, device=device)
        state = F.normalize(state, dim=-1) * self.state_radius
        strategy = torch.zeros(batch_size, self.strategy_size, device=device)

        all_states = []

        for pass_num in range(num_passes):
            state, strategy = self.thinking_pass(
                input_ids, attention_mask, state, strategy, pass_num
            )
            all_states.append(state)

        # --- ANSWER LOSS (deep supervision) ---
        answer_loss = torch.tensor(0.0, device=device)
        for i, s in enumerate(all_states):
            weight = (i + 1) / num_passes
            pass_answer_loss = self.compute_answer_loss(s, strategy, input_ids, answer_ids)
            answer_loss = answer_loss + weight * pass_answer_loss
        answer_loss = answer_loss / ((num_passes + 1) / 2)

        # --- PROBE LOSS (per-step gradient: 3 targets for 3 passes) ---
        pred1 = self.probe_head(all_states[0]).squeeze(-1)
        pred2 = self.probe_head(all_states[1]).squeeze(-1)
        pred3 = self.probe_head(all_states[2]).squeeze(-1)
        probe_loss = (F.mse_loss(pred1, v1_t) +
                      F.mse_loss(pred2, v2_t) +
                      F.mse_loss(pred3, v3_t))

        total_loss = answer_loss + 0.5 * probe_loss

        return total_loss, answer_loss.item(), probe_loss.item(), all_states


def evaluate(model, eval_dataset, device, num_passes=3):
    """Evaluate by generating answers and checking final number."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(eval_dataset), 8):
            batch_samples = eval_dataset.samples[i:i + 8]
            problems = [s['problem'] for s in batch_samples]
            gold_answers = [s['v3'] for s in batch_samples]

            inputs = model.tokenizer(
                problems, return_tensors='pt', padding=True,
                truncation=True, max_length=128,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            batch_size = input_ids.size(0)
            state = torch.randn(batch_size, model.state_size, device=device)
            state = F.normalize(state, dim=-1) * model.state_radius
            strategy = torch.zeros(batch_size, model.strategy_size, device=device)

            for pass_num in range(num_passes):
                state, strategy = model.thinking_pass(
                    input_ids, attention_mask, state, strategy, pass_num
                )

            lora_mods = model.lora(state, strategy)
            manager = AdditiveLoRAManager(model.transformer)
            manager.apply(lora_mods)

            try:
                generated = model.transformer.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            for j in range(batch_size):
                prompt_len = input_ids[j].size(0)
                gen_ids = generated[j][prompt_len:]
                gen_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                gen_text = gen_text.rstrip('.')

                try:
                    pred = int(gen_text.split()[0]) if gen_text else None
                except (ValueError, IndexError):
                    pred = None

                if pred == gold_answers[j]:
                    correct += 1
                total += 1

    return 100.0 * correct / total if total > 0 else 0.0


def train():
    print("THREE-STEP ARITHMETIC: Strategy Channel v3 + Answer Loss")
    print("=" * 60)
    print("Warm start from two-step 85.4% checkpoint")
    print("Architecture: same as two-step, 3 passes instead of 2")
    print("Target: >52% (prev best). Ceiling: 78.9% (92.4%^3)")
    print("=" * 60)

    device = torch.device('cuda')

    model = ThinkingModelV3()
    model.compressor = model.compressor.to(device)
    model.lora = model.lora.to(device)
    model.probe_head = model.probe_head.to(device)

    # Warm start: load two-step checkpoint
    ckpt_path = '/home/ubuntu/mycelium/checkpoints/strategy_v3_85pct.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.compressor.load_state_dict(ckpt['compressor'])
    model.lora.load_state_dict(ckpt['lora'])
    model.probe_head.load_state_dict(ckpt['probe_head'])
    print(f"Loaded two-step checkpoint: epoch {ckpt['epoch']}, accuracy {ckpt['accuracy']:.1f}%")

    train_dataset = ThreeStepArithmeticDataset(num_samples=5000, seed=42)
    eval_dataset = ThreeStepArithmeticDataset(num_samples=500, seed=123)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Lower LR for fine-tuning from warm start
    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 3e-5},
        {'params': list(model.lora.A_templates) + list(model.lora.B_templates), 'lr': 3e-4},
        {'params': list(model.lora.state_to_scales.parameters()), 'lr': 3e-4},
        {'params': list(model.probe_head.parameters()), 'lr': 3e-5},
    ])

    trainable = (list(model.compressor.parameters()) +
                 list(model.lora.parameters()) +
                 list(model.probe_head.parameters()))
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"LR: perceiver=3e-5, templates=3e-4, hypernetwork=3e-4 (0.3x two-step)")

    num_epochs = 10
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_answer_loss = 0.0
        epoch_probe_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            answers = batch['answer']
            v1_t = torch.tensor(
                [float(s) / 1000.0 for s in batch['v1']],
                dtype=torch.float32, device=device
            )
            v2_t = torch.tensor(
                [float(s) / 1000.0 for s in batch['v2']],
                dtype=torch.float32, device=device
            )
            v3_t = torch.tensor(
                [float(s) / 1000.0 for s in batch['v3']],
                dtype=torch.float32, device=device
            )

            optimizer.zero_grad()

            total_loss, ans_loss, prb_loss, _ = model.forward_train(
                problems, answers, v1_t, v2_t, v3_t, num_passes=3
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
            }, 'checkpoints/three_step_best.pt')
            print(f"  -> Saved new best checkpoint")

        print(f"Epoch {epoch+1}: total={avg_total:.4f} answer={avg_answer:.4f} probe={avg_probe:.6f} | Acc={accuracy:.1f}% (best={best_accuracy:.1f}%)")

    print(f"\nFinal: {best_accuracy:.1f}%")
    print(f"Previous three-step bests: 52% (SmolLM2), 22% (Llama no answer loss)")
    print(f"Theoretical ceiling: 78.9% (92.4%^3)")


if __name__ == '__main__':
    train()
