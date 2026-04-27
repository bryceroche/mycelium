"""
Stepping Stones L2: word operations over numerals (v21.4).

Problems like "half of 48 plus 48", "double 25 minus 13", etc.
Fixed passes (3) with pass-conditioned hypernetwork + target-cos contrastive.
Generation via teacher-forced answer loss (same as arithmetic training).
Early stopping (patience=2).

Warm start from two_step_contrastive_best.pt (94.8%).
"""
import argparse
import random
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/mycelium')

from scripts.train_two_step_pages import PageThinkingModel
from scripts.train_two_step_contrastive import warm_start
from src.contrastive_page_loss import per_page_contrastive_loss
from src.additive_lora import AdditiveLoRAManager


class L2WordOpsDataset(Dataset):
    """
    L2: word operations over numerals with chain-of-thought targets.
    "half of 48 plus 48 =" → "half of 48 = 24. 24 plus 48 = 72. The answer is 72."
    "double 25 minus 13 =" → "double 25 = 50. 50 minus 13 = 37. The answer is 37."

    Answers constrained to [1, 200].
    """
    OPS = {
        'half of': lambda x: x // 2,
        'double': lambda x: x * 2,
        'triple': lambda x: x * 3,
        'the square of': lambda x: x * x,
    }
    COMBINERS = {
        'plus': lambda a, b: a + b,
        'minus': lambda a, b: a - b,
    }

    def __init__(self, num_samples=20000, seed=42):
        rng = random.Random(seed)
        self.samples = []
        attempts = 0
        while len(self.samples) < num_samples:
            attempts += 1
            if attempts > num_samples * 20:
                break
            op_name = rng.choice(list(self.OPS.keys()))
            op_fn = self.OPS[op_name]
            comb_name = rng.choice(list(self.COMBINERS.keys()))
            comb_fn = self.COMBINERS[comb_name]

            if op_name == 'half of':
                a = rng.randint(2, 100) * 2  # even
            elif op_name == 'double':
                a = rng.randint(2, 50)
            elif op_name == 'triple':
                a = rng.randint(2, 30)
            elif op_name == 'the square of':
                a = rng.randint(2, 14)
            else:
                a = rng.randint(2, 50)

            v1 = op_fn(a)
            b = rng.randint(1, 100)
            result = comb_fn(v1, b)

            if 1 <= result <= 200:
                problem = f"{op_name} {a} {comb_name} {b} ="
                # Chain-of-thought: "half of 48 = 24. 24 plus 48 = 72. The answer is 72."
                cot = f"{op_name} {a} = {v1}. {v1} {comb_name} {b} = {result}. The answer is {result}."
                self.samples.append({
                    'problem': problem,
                    'answer': cot,
                    'final': result,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def forward_train(model, problems, answers, finals_t, num_passes=3):
    """Forward pass with pass-conditioned hypernetwork."""
    device = model.transformer.device

    inputs = model.tokenizer(
        problems, return_tensors='pt', padding=True, truncation=True, max_length=128,
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    answer_texts = [f" {a}" for a in answers]
    answer_inputs = model.tokenizer(
        answer_texts, return_tensors='pt', padding=True, add_special_tokens=False,
    )
    answer_ids = answer_inputs['input_ids'].to(device)

    batch_size = input_ids.size(0)

    embed_layer = model.transformer.model.embed_tokens
    problem_embeds = embed_layer(input_ids)

    state_pages = []
    strategy = torch.zeros(batch_size, model.strategy_size, device=device)

    for pass_num in range(num_passes):
        if pass_num == 0:
            outputs = model.transformer(
                inputs_embeds=problem_embeds, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = list(outputs.hidden_states[1:])
        else:
            lora_mods = model.hypernet(state_pages, strategy, pass_num=pass_num)
            manager = AdditiveLoRAManager(model.transformer)
            manager.apply(lora_mods)
            try:
                outputs = model.transformer(
                    inputs_embeds=problem_embeds, attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = list(outputs.hidden_states[1:])
            finally:
                manager.remove()

        page_delta, strategy, _mid_states = model.compressor(hidden_states, pass_num)
        page = F.normalize(page_delta, dim=-1) * model.page_radius
        state_pages.append(page)

    # Answer loss (teacher-forced)
    lora_mods = model.hypernet(state_pages, strategy, pass_num=num_passes)
    manager = AdditiveLoRAManager(model.transformer)
    manager.apply(lora_mods)
    try:
        answer_embeds = embed_layer(answer_ids)
        full_embeds = torch.cat([problem_embeds, answer_embeds], dim=1)
        outputs = model.transformer(inputs_embeds=full_embeds, use_cache=False)
    finally:
        manager.remove()
    prompt_len = input_ids.size(1)
    logits = outputs.logits[:, prompt_len - 1:-1, :]
    answer_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        answer_ids.reshape(-1),
        ignore_index=model.tokenizer.pad_token_id,
    )

    # Target-cosine per page
    c_loss = per_page_contrastive_loss(
        state_pages, finals_t, cross_target=0.7, within_target=0.3,
    )

    # Diagnostics
    with torch.no_grad():
        last_page = state_pages[-1].float()
        normed = F.normalize(last_page, dim=-1)
        sim = normed @ normed.T
        B = sim.size(0)
        off_diag = sim - torch.eye(B, device=sim.device)
        page_cos_mean = off_diag.sum() / (B * (B - 1))
        if len(state_pages) >= 2:
            n1 = F.normalize(state_pages[-2].float(), dim=-1)
            n2 = F.normalize(state_pages[-1].float(), dim=-1)
            cross_cos = (n1 * n2).sum(dim=-1).mean()
        else:
            cross_cos = torch.tensor(0.0)

    return answer_loss, c_loss, page_cos_mean, cross_cos


def extract_answer(gen_text):
    """Extract final answer from chain-of-thought generation.
    Looks for 'The answer is X' first, falls back to last number."""
    import re
    # Try "The answer is X"
    m = re.search(r'The answer is (\d+)', gen_text)
    if m:
        return int(m.group(1))
    # Fallback: last number in text
    numbers = re.findall(r'\b(\d+)\b', gen_text)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass
    return None


def evaluate(model, eval_dataset, device, num_passes=3):
    """Evaluate accuracy on word problems via generation."""
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
                    max_new_tokens=50, do_sample=False,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
            finally:
                manager.remove()

            for j in range(batch_size):
                prompt_len = input_ids[j].size(0)
                gen_ids = generated[j][prompt_len:]
                gen_text = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                pred = extract_answer(gen_text)
                if pred == gold_answers[j]:
                    correct += 1
                total += 1
    return 100.0 * correct / total if total > 0 else 0.0


def train(args):
    print("=" * 60)
    print("Stepping Stones L2 — word operations over numerals")
    print("=" * 60)
    print(f"target_cos=0.7  within=0.3  lam={args.lam}  num_passes={args.num_passes}")
    print(f"batch={args.batch_size}  train_n={args.num_train}  patience={args.patience}")
    print(f"Warm: {args.warm}")
    print("Pass-conditioned hypernetwork + target-cos contrastive")
    print("=" * 60)

    device = torch.device('cuda')
    model = PageThinkingModel()
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.probe_head = model.probe_head.to(device)

    if args.warm:
        warm_start(model, args.warm)

    train_dataset = L2WordOpsDataset(num_samples=args.num_train, seed=42)
    eval_dataset = L2WordOpsDataset(num_samples=500, seed=123)
    print(f"\nSample problems:")
    for i in range(5):
        s = train_dataset[i]
        print(f"  {s['problem']} {s['answer']}")
    print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 5e-5},
        {'params': list(model.hypernet.A_templates) + list(model.hypernet.B_templates), 'lr': 5e-4},
        {'params': (
            list(model.hypernet.page_project.parameters())
            + [model.hypernet.page_query]
            + list(model.hypernet.page_attn.parameters())
            + list(model.hypernet.page_norm.parameters())
            + list(model.hypernet.combine.parameters())
            + list(model.hypernet.pass_embed.parameters())
        ), 'lr': 5e-4},
    ])
    trainable = list(model.compressor.parameters()) + list(model.hypernet.parameters())
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Baseline
    base_acc = evaluate(model, eval_dataset, device, num_passes=args.num_passes)
    print(f"Baseline accuracy (before training): {base_acc:.1f}%\n")

    lam = args.lam
    best = 0.0
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        ep_total = ep_ans = ep_ctr = ep_cos = ep_xcross = 0.0
        nb = 0
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            answers = batch['answer']
            finals_t = torch.tensor(
                [int(s) for s in batch['final']],
                dtype=torch.long, device=device,
            )
            optimizer.zero_grad()
            ans_loss, c_loss, page_cos, cross_cos = forward_train(
                model, problems, answers, finals_t, num_passes=args.num_passes,
            )
            total_loss = ans_loss + lam * c_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            ep_total += total_loss.item()
            ep_ans += ans_loss.item()
            ep_ctr += c_loss.item()
            ep_cos += page_cos.item()
            ep_xcross += cross_cos.item()
            nb += 1

        elapsed = time.time() - t0
        acc = evaluate(model, eval_dataset, device, num_passes=args.num_passes)
        if acc > best:
            best = acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'accuracy': acc,
                'level': 'L2',
            }, 'checkpoints/stepping_stones_L2_best.pt')
            print(f"  -> saved checkpoint (acc={acc:.1f}%)")
        else:
            patience_counter += 1
        print(
            f"Epoch {epoch+1}: total={ep_total/nb:.4f} ans={ep_ans/nb:.4f} "
            f"contrastive={ep_ctr/nb:.4f} page_cos={ep_cos/nb:.4f} "
            f"p2v3={ep_xcross/nb:.4f} | "
            f"Acc={acc:.1f}% (best={best:.1f}%, base={base_acc:.1f}%) "
            f"[{elapsed:.0f}s, {nb/elapsed:.1f} it/s]"
        )
        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} epochs")
            break

    print(f"\nFinal: {best:.1f}% (baseline {base_acc:.1f}%)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--warm', type=str, default='checkpoints/two_step_contrastive_best.pt')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lam', type=float, default=0.05)
    p.add_argument('--num_train', type=int, default=20000)
    p.add_argument('--num_passes', type=int, default=3)
    p.add_argument('--patience', type=int, default=3)
    args = p.parse_args()
    train(args)
