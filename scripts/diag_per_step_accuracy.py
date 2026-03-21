#!/usr/bin/env python3
"""
Per-step accuracy diagnostic for AtomLoRA.

Measures accuracy after each thinking pass to see:
1. Which pass does the model start getting answers right?
2. Are any passes ready to graduate (>90% accuracy)?
3. How much does each additional pass help?

Usage:
    python scripts/diag_per_step_accuracy.py --checkpoint checkpoints/atom_lora_L5_best.pt --num_problems 50
"""

import os
import sys
import argparse
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from scripts.atom_lora import AtomLoRAModel, AnswerHead
from scripts.train_atom_lora import measure_per_step_accuracy, make_eval_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/atom_lora_L5_best.pt')
    parser.add_argument('--num_problems', type=int, default=50)
    parser.add_argument('--num_passes', type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = AtomLoRAModel(num_atoms=64, atom_rank=6)

    # Load answer head
    answer_head = AnswerHead(page_size=64)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Load model weights
    model.load_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    # Load answer head weights if present
    if 'answer_head' in ckpt:
        answer_head.load_state_dict(ckpt['answer_head'])
    answer_head.to(device)
    answer_head.eval()

    # Load eval data - use GSM8K test
    print(f"Loading {args.num_problems} GSM8K test problems...")
    from scripts.train_dual_lora_gsm8k import GSM8KDataset
    dataset = GSM8KDataset(split='test', max_samples=args.num_problems)

    print(f"\n{'='*60}")
    print("PER-STEP ACCURACY DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Problems: {args.num_problems}")
    print(f"Passes: {args.num_passes}")
    print(f"{'='*60}\n")

    # Measure per-step accuracy
    print("Running evaluation at each pass...")
    per_step_acc = measure_per_step_accuracy(
        model, answer_head, dataset, device,
        num_passes=args.num_passes, gsm8k_mode=True
    )

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for i, acc in enumerate(per_step_acc):
        acc_pct = acc * 100
        delta = f"+{(acc - per_step_acc[i-1])*100:.1f}" if i > 0 else "base"
        graduate = "✓ GRADUATE CANDIDATE" if acc_pct >= 90 else ""
        print(f"Pass {i+1}: {acc_pct:5.1f}%  ({delta:>6})  {graduate}")

    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    final_acc = per_step_acc[-1] * 100

    # Find biggest jump
    max_delta = 0
    max_delta_pass = 0
    for i in range(1, len(per_step_acc)):
        delta = (per_step_acc[i] - per_step_acc[i-1]) * 100
        if delta > max_delta:
            max_delta = delta
            max_delta_pass = i + 1

    if max_delta > 1:
        print(f"• Biggest jump: Pass {max_delta_pass-1}→{max_delta_pass} (+{max_delta:.1f}%)")

    # Check diminishing returns
    last_delta = (per_step_acc[-1] - per_step_acc[-2]) * 100 if len(per_step_acc) > 1 else 0
    if last_delta < 1:
        print(f"• Last pass adds only +{last_delta:.1f}% — diminishing returns")

    # Check for graduation candidates
    graduates = [i+1 for i, acc in enumerate(per_step_acc) if acc >= 0.9]
    if graduates:
        print(f"• Passes ready to graduate (≥90%): {graduates}")
        print(f"  → These passes could be cached to speed up training")
    else:
        print("• No passes at ≥90% yet — caching not beneficial yet")

    # Overall assessment
    if final_acc < 20:
        print("• Overall accuracy still climbing — keep training")
    elif final_acc >= 20 and graduates:
        print(f"• Consider enabling page cache for passes {graduates}")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
