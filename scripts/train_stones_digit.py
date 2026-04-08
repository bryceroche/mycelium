"""
Stepping-stones curriculum with the DIGIT answer head (categorical per-digit).

Same pipeline as train_stepping_stones.py, but swaps LogAnswerHead for
DigitAnswerHead. Categorical loss structurally cannot collapse to the mean,
so if the pages CAN encode answers, this head will pull them out.

L0-L4 are all bounded to [1, 200]. L5 (GSM8K) is skipped by default here
because the 3-digit head can't represent answers ≥ 1000; re-add after we
prove the pages encode.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/mycelium')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_gsm8k_answerhead import PageAnswerModel, collate
from scripts.train_stepping_stones import (
    SyntheticLevelDataset, load_level, warm_start,
)
from src.digit_answer_head import DigitAnswerHead


def swap_in_digit_head(model: PageAnswerModel, page_size: int = 64) -> None:
    model.answer_head = DigitAnswerHead(page_size=page_size).to(
        next(model.compressor.parameters()).device
    )

    def forward_train(questions, golds, num_passes=3):
        state_pages = model.run_thinking(questions, num_passes=num_passes)
        last_page = state_pages[-1].to(torch.float32)
        gold_t = torch.tensor(golds, dtype=torch.float32, device=last_page.device)
        loss, ll, dl = model.answer_head.compute_loss(last_page, gold_t)
        return loss, ll, dl

    model.forward_train = forward_train


def evaluate_digit(model, eval_dataset, device, num_passes=3, batch_size=8):
    model.eval()
    exact = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(eval_dataset), batch_size):
            batch = eval_dataset.samples[i:i + batch_size]
            questions = [s['question'] for s in batch]
            golds = [s['final'] for s in batch]
            state_pages = model.run_thinking(questions, num_passes=num_passes)
            last_page = state_pages[-1].to(torch.float32)
            preds = model.answer_head.decode(last_page).cpu().tolist()
            for pred, gold in zip(preds, golds):
                if int(pred) == int(gold):
                    exact += 1
                total += 1
    return 100.0 * exact / total


def train_level(model, level, train_ds, eval_ds, args, device, out_path):
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
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
        {'params': list(model.answer_head.parameters()), 'lr': 1e-3},
    ])
    trainable = (
        list(model.compressor.parameters())
        + list(model.hypernet.parameters())
        + list(model.answer_head.parameters())
    )

    best = 0.0
    stale = 0
    accum = args.grad_accum
    for epoch in range(args.max_epochs):
        model.train()
        ep_loss = ep_len = ep_dig = 0.0
        nb = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"L{level} ep{epoch+1}", leave=False)):
            loss, ll, dl = model.forward_train(
                batch['question'], batch['final'], num_passes=args.passes,
            )
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            ep_loss += loss.item(); ep_len += ll; ep_dig += dl; nb += 1

        acc = evaluate_digit(model, eval_ds, device, num_passes=args.passes,
                             batch_size=args.batch_size)
        improved = acc > best + 0.5
        print(f"L{level} ep{epoch+1}: loss={ep_loss/nb:.4f} "
              f"len={ep_len/nb:.4f} digit={ep_dig/nb:.4f} | exact={acc:.1f}% "
              f"(best={max(best,acc):.1f}%)")
        if acc > best:
            best = acc
            torch.save({
                'level': level, 'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'answer_head': model.answer_head.state_dict(),
                'exact': acc,
            }, out_path)
            print(f"  -> saved {out_path} (exact={acc:.1f}%)")
        stale = 0 if improved else stale + 1
        if stale >= args.patience:
            print(f"  plateau (patience={args.patience}), moving up")
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warm', type=str, default='checkpoints/two_step_pages_best.pt')
    ap.add_argument('--levels', type=str, default='0,1,2,3,4')
    ap.add_argument('--samples_per_level', type=int, default=5000)
    ap.add_argument('--max_epochs', type=int, default=5)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--grad_accum', type=int, default=2)
    ap.add_argument('--ckpt_dir', type=str, default='checkpoints')
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(',')]
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("=" * 60)
    print(f"Stones DIGIT head: levels={levels} "
          f"samples/level={args.samples_per_level} "
          f"max_epochs={args.max_epochs} patience={args.patience}")
    print("=" * 60)

    device = torch.device('cuda')
    model = PageAnswerModel()
    model.compressor = model.compressor.to(device)
    model.hypernet = model.hypernet.to(device)
    swap_in_digit_head(model, page_size=model.page_size)

    prev_ckpt = args.warm
    for level in levels:
        print("\n" + "=" * 60)
        print(f"LEVEL {level}")
        print("=" * 60)
        if prev_ckpt:
            warm_start(model, prev_ckpt)  # drops shape-mismatched (inc. answer_head)
        train_ds, eval_ds = load_level(level, args.samples_per_level)
        out_path = os.path.join(args.ckpt_dir, f'stones_digit_L{level}_best.pt')
        best = train_level(model, level, train_ds, eval_ds, args, device, out_path)
        print(f"LEVEL {level} final: exact={best:.1f}%")
        if os.path.exists(out_path):
            prev_ckpt = out_path

    print("\nCurriculum complete.")


if __name__ == '__main__':
    main()
