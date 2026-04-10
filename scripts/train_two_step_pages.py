"""
Two-step arithmetic smoke test for v21 page-based state accumulation.

Pages instead of overwriting state:
- state_pages = [] grows by one 64-float page per pass
- Per-page hypersphere normalize (each page on its own √64 sphere)
- PageAttentionHypernetwork cross-attends over all pages + strategy → LoRA scales
- First pass: no pages → zero scales (LoRA off, raw Llama forward)

Warm-starts perceiver + LoRA templates from a v20.1 checkpoint.
The hypernetwork is fresh (different shape — cross-attention vs Linear(576→256)).

Goal: match or beat 85.4% on two-step. If yes → page architecture preserves
what we already proved. If no → debug before GSM8K.

Usage:
    python scripts/train_two_step_pages.py --warm /path/to/strategy_v3_best.pt
"""
import argparse
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/mycelium')

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.compressor_v3 import Compressor
from src.page_attention_hypernetwork import PageAttentionHypernetwork
from src.additive_lora import AdditiveLoRAManager


class TwoStepArithmeticDataset(Dataset):
    def __init__(self, num_samples=10000, seed=42):
        random.seed(seed)
        self.samples = []
        for _ in range(num_samples):
            a = random.randint(2, 50)
            b = random.randint(2, 50)
            c = random.randint(2, 50)
            if random.random() < 0.5:
                op1 = '+'; intermediate = a + b
            else:
                a = b * random.randint(2, 10); op1 = '/'; intermediate = a // b
            if random.random() < 0.5:
                op2 = '+'; final = intermediate + c
            else:
                op2 = '-'; final = intermediate - c
            self.samples.append({
                'problem': f"({a} {op1} {b}) {op2} {c} =",
                'answer': str(final),
                'intermediate': intermediate,
                'final': final,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PageThinkingModel(nn.Module):
    def __init__(self, model_name='unsloth/Llama-3.2-1B'):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for p in self.transformer.parameters():
            p.requires_grad = False

        self.d_model = self.transformer.config.hidden_size
        self.num_layers = self.transformer.config.num_hidden_layers
        num_kv_heads = self.transformer.config.num_key_value_heads
        head_dim = self.d_model // self.transformer.config.num_attention_heads
        d_kv = num_kv_heads * head_dim

        self.page_size = 64
        self.strategy_size = 64
        self.page_radius = (self.page_size ** 0.5)

        self.compressor = Compressor(
            num_transformer_layers=self.num_layers,
            d_transformer=self.d_model,
            d_perceiver=1024,
            num_queries=4,
            num_perceiver_layers=7,
            state_size=self.page_size,
            strategy_size=self.strategy_size,
        )

        self.hypernet = PageAttentionHypernetwork(
            d_model=self.d_model,
            d_kv=d_kv,
            page_size=self.page_size,
            strategy_size=self.strategy_size,
            rank=4,
            num_layers=self.num_layers,
        )

        self.probe_head = nn.Linear(self.page_size, 1)

    def thinking_pass(self, input_ids, attention_mask, state_pages, strategy, pass_num):
        # Hypernet attends over current pages + strategy
        if len(state_pages) == 0:
            # First pass: zero LoRA, raw Llama forward
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
        else:
            lora_mods = self.hypernet(state_pages, strategy, pass_num=pass_num)
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

        page_delta, new_strategy = self.compressor(hidden_states, pass_num)
        # Per-page normalize on its own hypersphere
        page = F.normalize(page_delta, dim=-1) * self.page_radius
        return page, new_strategy

    def compute_answer_loss(self, state_pages, strategy, prompt_ids, answer_ids, pass_num=0):
        lora_mods = self.hypernet(state_pages, strategy, pass_num=pass_num)
        manager = AdditiveLoRAManager(self.transformer)
        manager.apply(lora_mods)
        try:
            full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            outputs = self.transformer(input_ids=full_ids, use_cache=False)
        finally:
            manager.remove()
        prompt_len = prompt_ids.size(1)
        logits = outputs.logits[:, prompt_len - 1:-1, :]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            answer_ids.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        return loss

    def forward_train(self, problems, answers, intermediates_t, finals_t, num_passes=2):
        inputs = self.tokenizer(
            problems, return_tensors='pt', padding=True,
            truncation=True, max_length=128,
        )
        input_ids = inputs['input_ids'].to(self.transformer.device)
        attention_mask = inputs['attention_mask'].to(self.transformer.device)

        answer_texts = [f" {a}" for a in answers]
        answer_inputs = self.tokenizer(
            answer_texts, return_tensors='pt', padding=True, add_special_tokens=False,
        )
        answer_ids = answer_inputs['input_ids'].to(self.transformer.device)

        batch_size = input_ids.size(0)
        device = input_ids.device

        state_pages = []
        strategy = torch.zeros(batch_size, self.strategy_size, device=device)

        for pass_num in range(num_passes):
            page, strategy = self.thinking_pass(
                input_ids, attention_mask, state_pages, strategy, pass_num,
            )
            state_pages.append(page)

        # Deep supervision: each prefix of pages tries to generate
        answer_loss = torch.tensor(0.0, device=device)
        for i in range(len(state_pages)):
            prefix = state_pages[: i + 1]
            weight = (i + 1) / num_passes
            pass_answer_loss = self.compute_answer_loss(prefix, strategy, input_ids, answer_ids)
            answer_loss = answer_loss + weight * pass_answer_loss
        answer_loss = answer_loss / ((num_passes + 1) / 2)

        # Probe loss on individual pages (per-step gradient)
        pred1 = self.probe_head(state_pages[0]).squeeze(-1)
        pred2 = self.probe_head(state_pages[1]).squeeze(-1)
        probe_loss = F.mse_loss(pred1, intermediates_t) + F.mse_loss(pred2, finals_t)

        total_loss = answer_loss + 0.5 * probe_loss
        return total_loss, answer_loss.item(), probe_loss.item()


def evaluate(model, eval_dataset, device, num_passes=2):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(eval_dataset), 8):
            batch_samples = [eval_dataset[j] for j in range(i, min(i + 8, len(eval_dataset)))]
            problems = [s['problem'] for s in batch_samples]
            gold_answers = [s['final'] for s in batch_samples]

            inputs = model.tokenizer(
                problems, return_tensors='pt', padding=True,
                truncation=True, max_length=128,
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            batch_size = input_ids.size(0)

            state_pages = []
            strategy = torch.zeros(batch_size, model.strategy_size, device=device)
            for pass_num in range(num_passes):
                page, strategy = model.thinking_pass(
                    input_ids, attention_mask, state_pages, strategy, pass_num,
                )
                state_pages.append(page)

            lora_mods = model.hypernet(state_pages, strategy, pass_num=num_passes)
            manager = AdditiveLoRAManager(model.transformer)
            manager.apply(lora_mods)
            try:
                generated = model.transformer.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=10, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            for j in range(batch_size):
                prompt_len = input_ids[j].size(0)
                gen_ids = generated[j][prompt_len:]
                gen_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip().rstrip('.')
                try:
                    pred = int(gen_text.split()[0]) if gen_text else None
                except (ValueError, IndexError):
                    pred = None
                if pred == gold_answers[j]:
                    correct += 1
                total += 1
    return 100.0 * correct / total if total > 0 else 0.0


def warm_start(model, ckpt_path):
    print(f"Warm-starting from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Compressor: same shape — load directly
    missing, unexpected = model.compressor.load_state_dict(ckpt['compressor'], strict=False)
    print(f"  compressor: missing={len(missing)} unexpected={len(unexpected)}")
    # LoRA templates: copy A_templates / B_templates from old StateConditionedLoRA
    old_lora = ckpt['lora']
    loaded_a = 0
    loaded_b = 0
    for i in range(len(model.hypernet.A_templates)):
        key = f'A_templates.{i}'
        if key in old_lora:
            model.hypernet.A_templates[i].data.copy_(old_lora[key])
            loaded_a += 1
    for i in range(len(model.hypernet.B_templates)):
        key = f'B_templates.{i}'
        if key in old_lora:
            model.hypernet.B_templates[i].data.copy_(old_lora[key])
            loaded_b += 1
    print(f"  templates: A loaded={loaded_a}/4, B loaded={loaded_b}/4")
    # Hypernet (page attention) stays fresh — different shape than old Linear(576→256)
    if 'probe_head' in ckpt:
        model.probe_head.load_state_dict(ckpt['probe_head'])
        print("  probe_head: loaded")


def train(args):
    print("=" * 60)
    print("Two-step arithmetic v21 PAGE smoke test")
    print("=" * 60)
    print("Pages: append-only, per-page normalized, cross-attention hypernet")
    print("Goal: match or beat 85.4% (v20.1 with overwriting state)")
    print("=" * 60)

    device = torch.device('cuda')
    model = PageThinkingModel()
    model.compressor = model.compressor.to(device)
    model.hypernet = model.hypernet.to(device)
    model.probe_head = model.probe_head.to(device)

    if args.warm:
        warm_start(model, args.warm)

    train_dataset = TwoStepArithmeticDataset(num_samples=5000, seed=42)
    eval_dataset = TwoStepArithmeticDataset(num_samples=500, seed=123)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 1e-4},
        {'params': list(model.hypernet.A_templates) + list(model.hypernet.B_templates), 'lr': 1e-3},
        {'params': (
            list(model.hypernet.page_project.parameters()) +
            [model.hypernet.page_query] +
            list(model.hypernet.page_attn.parameters()) +
            list(model.hypernet.page_norm.parameters()) +
            list(model.hypernet.combine.parameters())
        ), 'lr': 1e-3},
        {'params': list(model.probe_head.parameters()), 'lr': 1e-4},
    ])

    trainable = list(model.compressor.parameters()) + list(model.hypernet.parameters()) + list(model.probe_head.parameters())
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        ep_total = ep_ans = ep_prb = 0.0
        nb = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            answers = batch['answer']
            intermediates = torch.tensor(
                [float(s) / 1000.0 for s in batch['intermediate']],
                dtype=torch.float32, device=device,
            )
            finals = torch.tensor(
                [float(s) / 1000.0 for s in batch['final']],
                dtype=torch.float32, device=device,
            )
            optimizer.zero_grad()
            total_loss, ans, prb = model.forward_train(
                problems, answers, intermediates, finals, num_passes=2,
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            ep_total += total_loss.item(); ep_ans += ans; ep_prb += prb; nb += 1

        acc = evaluate(model, eval_dataset, device, num_passes=2)
        if acc > best:
            best = acc
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'probe_head': model.probe_head.state_dict(),
                'accuracy': acc,
            }, 'checkpoints/two_step_pages_best.pt')
            print(f"  -> Saved checkpoint (acc={acc:.1f}%)")
        print(f"Epoch {epoch+1}: total={ep_total/nb:.4f} ans={ep_ans/nb:.4f} prb={ep_prb/nb:.6f} | Acc={acc:.1f}% (best={best:.1f}%, target≥85)")

    print(f"\nFinal: {best:.1f}% | v20.1 baseline: 85.4%")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--warm', type=str, default=None, help='Path to v20.1 checkpoint to warm-start from')
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    train(args)
