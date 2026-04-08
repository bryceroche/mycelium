"""
Two-step arithmetic with CONTRASTIVE page loss (v21.2).

Fork of train_two_step_pages.py. Keeps the answer (generation) loss,
DROPS the probe loss, adds a contrastive loss on the last page:

    total = answer_loss + lam_contrastive * contrastive_page_loss(last_page, gold)

The contrastive loss pulls same-answer last-pages together and pushes
different-answer last-pages apart (margin 0.2). O(batch²) pairwise
gradient is strong enough to break the fixed-point collapse that the
probe-only / head-based losses could not.

Warm-starts from the existing page checkpoint so we can measure a clean
dip/recovery trajectory AND re-run the cosine similarity diagnostic
after training.
"""
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/mycelium')

from scripts.train_two_step_pages import (
    PageThinkingModel, TwoStepArithmeticDataset, evaluate,
)
from src.contrastive_page_loss import target_cos_page_loss


def forward_train_contrastive(model, problems, answers, finals_t, num_passes=2):
    """
    Replaces model.forward_train:
      - removes probe loss
      - adds contrastive loss on the last page
      - returns (total, answer_loss, contrastive_loss, page_cos_mean)
    """
    inputs = model.tokenizer(
        problems, return_tensors='pt', padding=True, truncation=True, max_length=128,
    )
    input_ids = inputs['input_ids'].to(model.transformer.device)
    attention_mask = inputs['attention_mask'].to(model.transformer.device)

    answer_texts = [f" {a}" for a in answers]
    answer_inputs = model.tokenizer(
        answer_texts, return_tensors='pt', padding=True, add_special_tokens=False,
    )
    answer_ids = answer_inputs['input_ids'].to(model.transformer.device)

    batch_size = input_ids.size(0)
    device = input_ids.device

    state_pages = []
    strategy = torch.zeros(batch_size, model.strategy_size, device=device)
    for pass_num in range(num_passes):
        page, strategy = model.thinking_pass(
            input_ids, attention_mask, state_pages, strategy, pass_num,
        )
        state_pages.append(page)

    # Answer loss on the final page set (teacher-forced, one forward).
    answer_loss = model.compute_answer_loss(
        state_pages, strategy, input_ids, answer_ids,
    )

    # Contrastive loss on the last page.
    last_page = state_pages[-1].to(torch.float32)
    c_loss = target_cos_page_loss(last_page, finals_t, target_cos=0.4)

    # Diagnostic: mean off-diagonal cosine similarity (lower is better).
    with torch.no_grad():
        normed = F.normalize(last_page, dim=-1)
        sim = normed @ normed.T
        B = sim.size(0)
        off_diag = sim - torch.eye(B, device=sim.device)
        page_cos_mean = off_diag.sum() / (B * (B - 1))

    return answer_loss, c_loss, page_cos_mean


def warm_start(model, ckpt_path):
    print(f"Warm-starting from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # Handle both page-based ckpts and v20.1 ckpts.
    if 'compressor' in ckpt:
        own = model.compressor.state_dict()
        keep = {k: v for k, v in ckpt['compressor'].items()
                if k in own and own[k].shape == v.shape}
        model.compressor.load_state_dict(keep, strict=False)
        print(f"  compressor: loaded {len(keep)}/{len(ckpt['compressor'])}")
    if 'hypernet' in ckpt:
        own = model.hypernet.state_dict()
        keep = {k: v for k, v in ckpt['hypernet'].items()
                if k in own and own[k].shape == v.shape}
        model.hypernet.load_state_dict(keep, strict=False)
        print(f"  hypernet: loaded {len(keep)}/{len(ckpt['hypernet'])}")


def train(args):
    print("=" * 60)
    print("Two-step arithmetic v21.2 — CONTRASTIVE page loss")
    print("=" * 60)
    print(f"target-cos loss: target=0.4  λ={args.lam} (constant)  "
          f"batch={args.batch_size}")
    print("Target: pages differentiate across problems AND accuracy ≥ 85%")
    print("=" * 60)

    device = torch.device('cuda')
    model = PageThinkingModel()
    model.compressor = model.compressor.to(device)
    model.hypernet = model.hypernet.to(device)
    model.probe_head = model.probe_head.to(device)

    # Force strategy_size=512 if warm-starting from the old 86.2% checkpoint,
    # which had combine.0 sized for 512 strategy. Current model defaults to 64,
    # so we'd hit shape mismatch. Inspect and rebuild if needed.
    if args.warm and '86' in args.warm or 'pages_best' in args.warm:
        ckpt = torch.load(args.warm, map_location='cpu', weights_only=False)
        strat = ckpt['compressor'].get('strategy_head.weight')
        if strat is not None and strat.shape[0] != model.strategy_size:
            print(f"  rebuilding model with strategy_size={strat.shape[0]} "
                  f"(was {model.strategy_size}) to match checkpoint")
            from src.compressor_v3 import Compressor
            from src.page_attention_hypernetwork import PageAttentionHypernetwork
            model.strategy_size = strat.shape[0]
            d_model = model.d_model
            num_layers = model.num_layers
            num_kv_heads = model.transformer.config.num_key_value_heads
            head_dim = d_model // model.transformer.config.num_attention_heads
            d_kv = num_kv_heads * head_dim
            model.compressor = Compressor(
                num_transformer_layers=num_layers, d_transformer=d_model,
                d_perceiver=1024, num_queries=4, num_perceiver_layers=7,
                state_size=64, strategy_size=model.strategy_size,
            ).to(device)
            model.hypernet = PageAttentionHypernetwork(
                d_model=d_model, d_kv=d_kv, page_size=64,
                strategy_size=model.strategy_size, rank=4, num_layers=num_layers,
            ).to(device)

    if args.warm:
        warm_start(model, args.warm)

    train_dataset = TwoStepArithmeticDataset(num_samples=5000, seed=42)
    eval_dataset = TwoStepArithmeticDataset(num_samples=500, seed=123)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW([
        {'params': list(model.compressor.parameters()), 'lr': 1e-4},
        {'params': list(model.hypernet.A_templates) + list(model.hypernet.B_templates), 'lr': 1e-3},
        {'params': (
            list(model.hypernet.page_project.parameters())
            + [model.hypernet.page_query]
            + list(model.hypernet.page_attn.parameters())
            + list(model.hypernet.page_norm.parameters())
            + list(model.hypernet.combine.parameters())
        ), 'lr': 1e-3},
    ])
    trainable = list(model.compressor.parameters()) + list(model.hypernet.parameters())
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    lam = args.lam

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        ep_total = ep_ans = ep_ctr = 0.0
        ep_cos = 0.0
        nb = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            problems = batch['problem']
            answers = batch['answer']
            finals_t = torch.tensor(
                [int(s) for s in batch['final']],
                dtype=torch.long, device=device,
            )
            optimizer.zero_grad()
            ans_loss, c_loss, page_cos = forward_train_contrastive(
                model, problems, answers, finals_t, num_passes=2,
            )
            total_loss = ans_loss + lam * c_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            ep_total += total_loss.item()
            ep_ans += ans_loss.item()
            ep_ctr += c_loss.item()
            ep_cos += page_cos.item()
            nb += 1

        acc = evaluate(model, eval_dataset, device, num_passes=2)
        if acc > best:
            best = acc
            torch.save({
                'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'accuracy': acc,
            }, 'checkpoints/two_step_contrastive_best.pt')
            print(f"  -> saved checkpoint (acc={acc:.1f}%)")
        print(
            f"Epoch {epoch+1}: total={ep_total/nb:.4f} ans={ep_ans/nb:.4f} "
            f"contrastive={ep_ctr/nb:.4f} page_cos={ep_cos/nb:.4f} | "
            f"Acc={acc:.1f}% (best={best:.1f}%, target ≥85)"
        )

    print(f"\nFinal: {best:.1f}% (baseline 85.4% / 86.2%)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--warm', type=str, default='checkpoints/two_step_pages_best.pt')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lam', type=float, default=0.05,
                   help='Constant contrastive weight (target-cos loss is bidirectional)')
    args = p.parse_args()
    train(args)
