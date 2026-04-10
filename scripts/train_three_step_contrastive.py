"""
Three-step arithmetic with pass-conditioned hypernetwork + target-cos (v21.3).

Loss: answer_loss + lam * per_page_contrastive(cross=0.7, within=0.3)

Key innovation: pass-conditioned hypernetwork breaks the circular copy loop
that made pages 2&3 identical. The hypernetwork receives pass_num so different
passes produce different LoRA even with identical pages, breaking the cycle:
  same pages → same LoRA → same hidden states → same pages

Combined with target-cosine per page (proven self-stabilizing at 0.7) for
cross-problem differentiation. No SupCon (overshoots to page_cos=0.02).

Optimizations:
  - bf16 perceiver/hypernet (was fp32, 105M params)
  - cached token embeddings (embed once, reuse 3x)
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

from scripts.train_two_step_pages import PageThinkingModel, evaluate
from scripts.train_two_step_contrastive import warm_start
from src.contrastive_page_loss import per_page_contrastive_loss
from src.additive_lora import AdditiveLoRAManager


class ThreeStepPageDataset(Dataset):
    """((a op1 b) op2 c) op3 d — yields 'problem', 'answer', 'final'."""

    def __init__(self, num_samples=5000, seed=42):
        rng = random.Random(seed)
        self.samples = []
        for _ in range(num_samples):
            a = rng.randint(2, 50)
            b = rng.randint(2, 50)
            c = rng.randint(2, 50)
            d = rng.randint(2, 50)
            if rng.random() < 0.5:
                op1, v1 = '+', a + b
            else:
                a = b * rng.randint(2, 10)
                op1, v1 = '/', a // b
            if rng.random() < 0.5:
                op2, v2 = '+', v1 + c
            else:
                op2, v2 = '-', v1 - c
            if rng.random() < 0.5:
                op3, v3 = '+', v2 + d
            else:
                op3, v3 = '-', v2 - d
            problem = f"(({a} {op1} {b}) {op2} {c}) {op3} {d} ="
            self.samples.append({
                'problem': problem,
                'answer': str(v3),
                'final': v3,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def forward_train_fast(model, problems, answers, finals_t, num_passes=3):
    """
    Optimized forward: bf16 throughout, cached embeddings.
    """
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

    # Cache embeddings — same tokens embedded once, reused for all passes
    embed_layer = model.transformer.model.embed_tokens
    problem_embeds = embed_layer(input_ids)  # (B, seq, d_model), bf16

    state_pages = []
    strategy = torch.zeros(batch_size, model.strategy_size, device=device)

    for pass_num in range(num_passes):
        if pass_num == 0:
            # First pass: no LoRA, raw Llama forward with cached embeds
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

        page_delta, strategy = model.compressor(hidden_states, pass_num)
        page = F.normalize(page_delta, dim=-1) * model.page_radius
        state_pages.append(page)

    # Answer loss (teacher-forced) — use last pass_num for answer generation
    lora_mods = model.hypernet(state_pages, strategy, pass_num=num_passes)
    manager = AdditiveLoRAManager(model.transformer)
    manager.apply(lora_mods)
    try:
        # Embed answer tokens and concat with cached problem embeds
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

    # Target-cosine per page (cross=0.7) + soft anti-copying (threshold=0.7)
    c_loss = per_page_contrastive_loss(
        state_pages, finals_t, cross_target=0.7, within_target=0.3,
    )

    # Diagnostic: mean off-diagonal cosine similarity on last page
    with torch.no_grad():
        last_page = state_pages[-1].float()
        normed = F.normalize(last_page, dim=-1)
        sim = normed @ normed.T
        B = sim.size(0)
        off_diag = sim - torch.eye(B, device=sim.device)
        page_cos_mean = off_diag.sum() / (B * (B - 1))
        # Cross-page diagnostic: page 2 vs page 3
        n2 = F.normalize(state_pages[-2].float(), dim=-1)
        n3 = F.normalize(state_pages[-1].float(), dim=-1)
        cross_cos = (n2 * n3).sum(dim=-1).mean()

    return answer_loss, c_loss, page_cos_mean, cross_cos


def train(args):
    print("=" * 60)
    print("Three-step v21.3 — target-cos + pass-conditioned hypernetwork")
    print("=" * 60)
    print(f"target_cos=0.7  within=0.3  lam={args.lam}  num_passes=3  batch={args.batch_size}  train_n={args.num_train}")
    print(f"Warm: {args.warm}")
    print("Optimizations: bf16 perceiver/hypernet, cached embeddings")
    print("Loss: ans + lam * per_page_contrastive(cross=0.7, within=0.3)")
    print("NEW: pass-conditioned hypernetwork (breaks circular copy loop)")
    print("Target: > 83.4% previous best; ceiling ~92.5%")
    print("=" * 60)

    device = torch.device('cuda')
    model = PageThinkingModel()
    # Move perceiver/hypernet to bf16 (was fp32 — 105M params, 2x speedup)
    model.compressor = model.compressor.to(device=device, dtype=torch.bfloat16)
    model.hypernet = model.hypernet.to(device=device, dtype=torch.bfloat16)
    model.probe_head = model.probe_head.to(device)

    if args.warm:
        warm_start(model, args.warm)

    train_dataset = ThreeStepPageDataset(num_samples=args.num_train, seed=42)
    eval_dataset = ThreeStepPageDataset(num_samples=500, seed=123)
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
        ), 'lr': 5e-4},
    ])
    trainable = list(model.compressor.parameters()) + list(model.hypernet.parameters())
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"Compressor dtype: {next(model.compressor.parameters()).dtype}")
    print(f"Hypernet dtype: {next(model.hypernet.parameters()).dtype}")
    print(f"GPU mem after init: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    lam = args.lam
    best = 0.0
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
            ans_loss, c_loss, page_cos, cross_cos = forward_train_fast(
                model, problems, answers, finals_t, num_passes=3,
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
        acc = evaluate(model, eval_dataset, device, num_passes=3)
        if acc > best:
            best = acc
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'accuracy': acc,
            }, 'checkpoints/three_step_contrastive_best.pt')
            print(f"  -> saved checkpoint (acc={acc:.1f}%)")
        print(
            f"Epoch {epoch+1}: total={ep_total/nb:.4f} ans={ep_ans/nb:.4f} "
            f"contrastive={ep_ctr/nb:.4f} page_cos={ep_cos/nb:.4f} "
            f"p2v3={ep_xcross/nb:.4f} | "
            f"Acc={acc:.1f}% (best={best:.1f}%, prev 83.4%, ceiling 92.5%) "
            f"[{elapsed:.0f}s, {nb/elapsed:.1f} it/s]"
        )

    print(f"\nFinal: {best:.1f}% (prev 83.4%, ceiling 92.5%)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--warm', type=str, default='checkpoints/two_step_contrastive_best.pt')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--batch_size', type=int, default=48)
    p.add_argument('--lam', type=float, default=0.05, help='Contrastive lambda')
    p.add_argument('--num_train', type=int, default=20000)
    args = p.parse_args()
    train(args)
