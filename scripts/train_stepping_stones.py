"""
Stepping-stones curriculum trainer with log-answer head.

Climbs L0 → L1 → L2 → L3 → L4 → L5 (GSM8K). Each level trains until eval
accuracy plateaus (patience), saves a checkpoint, and warm-starts the next.

All levels use the same PageAnswerModel (pages + log-answer head) from
train_gsm8k_answerhead. L0/L1 serve as a "refresh" for the re-initialized
strategy head (shrunk 512 → 64) and the fresh log-head before climbing
into word problems.
"""
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, '/home/ubuntu/mycelium')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_gsm8k_answerhead import PageAnswerModel, collate, evaluate, GSM8KDataset
from src import stepping_stones


class SyntheticLevelDataset(Dataset):
    def __init__(self, level: int, n: int, seed: int):
        self.samples = []
        for s in stepping_stones.generate(level, n, seed=seed):
            self.samples.append({
                'question': s['question'],
                'final': float(s['answer']),
                'is_int': True,
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def load_level(level: int, samples: int):
    if level == 5:
        train = GSM8KDataset(split='train', max_samples=samples)
        test = GSM8KDataset(split='test', max_samples=500)
        return train, test
    train = SyntheticLevelDataset(level, samples, seed=1000 + level)
    test = SyntheticLevelDataset(level, 500, seed=9000 + level)
    return train, test


def _filter_shape_compatible(target_module, sd):
    own = target_module.state_dict()
    keep, drop = {}, []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            keep[k] = v
        else:
            drop.append(k)
    return keep, drop


def warm_start(model, ckpt_path):
    print(f"Warm-starting from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'compressor' in ckpt:
        keep, drop = _filter_shape_compatible(model.compressor, ckpt['compressor'])
        miss, unex = model.compressor.load_state_dict(keep, strict=False)
        print(f"  compressor: loaded={len(keep)} dropped={len(drop)} missing={len(miss)}")
    if 'hypernet' in ckpt:
        keep, drop = _filter_shape_compatible(model.hypernet, ckpt['hypernet'])
        miss, unex = model.hypernet.load_state_dict(keep, strict=False)
        print(f"  hypernet: loaded={len(keep)} dropped={len(drop)} missing={len(miss)}")
    if 'answer_head' in ckpt:
        keep, drop = _filter_shape_compatible(model.answer_head, ckpt['answer_head'])
        model.answer_head.load_state_dict(keep, strict=False)
        print(f"  answer_head: loaded={len(keep)} dropped={len(drop)}")


def train_level(model, level, train_ds, eval_ds, args, device, out_path):
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
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

    best_exact = 0.0
    best_tol1 = 0.0
    stale = 0
    accum = args.grad_accum
    for epoch in range(args.max_epochs):
        model.train()
        ep_loss = ep_mag = ep_sign = 0.0; nb = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"L{level} ep{epoch+1}", leave=False)):
            loss, ml, sl = model.forward_train(
                batch['question'], batch['final'], num_passes=args.passes,
            )
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            ep_loss += loss.item(); ep_mag += ml; ep_sign += sl; nb += 1

        exact, tol1 = evaluate(model, eval_ds, device, num_passes=args.passes,
                               batch_size=args.batch_size)
        improved = exact > best_exact + 0.5
        print(f"L{level} ep{epoch+1}: loss={ep_loss/nb:.4f} mag={ep_mag/nb:.4f} "
              f"sign={ep_sign/nb:.4f} | exact={exact:.1f}% tol1%={tol1:.1f}%")
        if exact > best_exact:
            best_exact = exact
            torch.save({
                'level': level, 'epoch': epoch + 1,
                'compressor': model.compressor.state_dict(),
                'hypernet': model.hypernet.state_dict(),
                'answer_head': model.answer_head.state_dict(),
                'exact': exact, 'tol1': tol1,
            }, out_path)
            print(f"  -> saved {out_path} (exact={exact:.1f}%)")
        if tol1 > best_tol1:
            best_tol1 = tol1
        stale = 0 if improved else stale + 1
        if stale >= args.patience:
            print(f"  plateau (patience={args.patience}), moving up")
            break
    return best_exact, best_tol1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warm', type=str, default='checkpoints/two_step_pages_best.pt')
    ap.add_argument('--levels', type=str, default='0,1,2,3,4,5')
    ap.add_argument('--samples_per_level', type=int, default=5000)
    ap.add_argument('--max_epochs', type=int, default=4)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--passes', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--grad_accum', type=int, default=2)
    ap.add_argument('--ckpt_dir', type=str, default='checkpoints')
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(',')]
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print("=" * 60)
    print(f"Stepping stones: levels={levels} "
          f"samples/level={args.samples_per_level} "
          f"max_epochs={args.max_epochs} patience={args.patience}")
    print("=" * 60)

    device = torch.device('cuda')
    model = PageAnswerModel()
    model.compressor = model.compressor.to(device)
    model.hypernet = model.hypernet.to(device)
    model.answer_head = model.answer_head.to(device)

    prev_ckpt = args.warm
    for level in levels:
        print("\n" + "=" * 60)
        print(f"LEVEL {level}")
        print("=" * 60)
        if prev_ckpt:
            warm_start(model, prev_ckpt)
        n = args.samples_per_level if level < 5 else min(args.samples_per_level * 2, 7473)
        train_ds, eval_ds = load_level(level, n)
        out_path = os.path.join(args.ckpt_dir, f'stones_L{level}_best.pt')
        exact, tol1 = train_level(model, level, train_ds, eval_ds, args, device, out_path)
        print(f"LEVEL {level} final: exact={exact:.1f}% tol1%={tol1:.1f}%")
        if os.path.exists(out_path):
            prev_ckpt = out_path

    print("\nCurriculum complete.")


if __name__ == '__main__':
    main()
