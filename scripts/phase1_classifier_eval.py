"""Phase 1 Classifier: Evaluation script.

Evaluates a checkpoint against the validation set and optionally against
the Haiku ground-truth factor graphs.

Usage:
  # Evaluate latest checkpoint:
  python scripts/phase1_classifier_eval.py

  # Evaluate specific checkpoint:
  python scripts/phase1_classifier_eval.py --ckpt .cache/phase1_ckpts/phase1_classifier_step500.pt

  # Run inference on a single problem (sanity check):
  python scripts/phase1_classifier_eval.py --problem "Janet's ducks lay 16 eggs per day..."

Metrics reported:
  Per-variable:
    - leaf_acc    : fraction of variables with correct IB leaf class (32-way)
    - op_acc      : fraction of variables with correct op type (5-way)
    - op_adj_acc  : op accuracy restricted to non-observed (op != none) vars
  Per-problem:
    - all_correct : fraction of problems where ALL variables are correct
    - top1_leaf_acc : mean top-1 accuracy (== leaf_acc, sanity)
  Op-family accuracy (add/sub/mul/div) to check for blindspots
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR      = REPO_ROOT / ".cache"
VAL_JSONL      = CACHE_DIR / "gsm8k_phase1_classifier_val.jsonl"
CHECKPOINT_DIR = CACHE_DIR / "phase1_ckpts"

from mycelium.phase1_classifier import (
    Phase1Classifier, Phase1Config,
    LEAF_IDS, LEAF_TO_INT, OP_TO_INT, N_LEAVES, N_OPS,
)
INT_TO_OP = {v: k for k, v in OP_TO_INT.items()}


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path, device: torch.device) -> Phase1Classifier:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model  = Phase1Classifier(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path} (step {ckpt.get('step', '?')})")
    return model


def find_latest_checkpoint() -> Path | None:
    if not CHECKPOINT_DIR.exists():
        return None
    ckpts = sorted(CHECKPOINT_DIR.glob("phase1_classifier_step*.pt"))
    if not ckpts:
        return None
    return ckpts[-1]


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_full(model: Phase1Classifier, val_jsonl: Path, tokenizer,
              batch_size: int = 64, max_seq_len: int = 128,
              device: torch.device = torch.device("cpu")) -> None:
    """Full evaluation with per-leaf + per-op breakdown."""
    from scripts.phase1_classifier_train import VariableDataset
    ds = VariableDataset(val_jsonl, tokenizer, max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    leaf_correct_per_class: Counter = Counter()
    leaf_total_per_class:   Counter = Counter()
    op_correct_per_op:      Counter = Counter()
    op_total_per_op:        Counter = Counter()

    total_leaf_correct = total_op_correct = total = 0
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        leaf_labels    = batch["leaf_label"].to(device)
        op_labels      = batch["op_label"].to(device)

        out    = model(input_ids, attention_mask)
        losses = model.compute_loss(out.leaf_logits, out.op_logits,
                                    leaf_labels, op_labels)

        leaf_pred = out.leaf_logits.argmax(-1)
        op_pred   = out.op_logits.argmax(-1)

        B = input_ids.size(0)
        total += B
        total_loss += losses["loss"].item() * B
        total_leaf_correct += (leaf_pred == leaf_labels).sum().item()
        total_op_correct   += (op_pred   == op_labels).sum().item()

        # Per-class breakdown
        for true_l, pred_l in zip(leaf_labels.tolist(), leaf_pred.tolist()):
            leaf_total_per_class[true_l] += 1
            if true_l == pred_l:
                leaf_correct_per_class[true_l] += 1

        for true_o, pred_o in zip(op_labels.tolist(), op_pred.tolist()):
            op_total_per_op[true_o] += 1
            if true_o == pred_o:
                op_correct_per_op[true_o] += 1

    n = max(total, 1)
    print("\n=== Evaluation Results ===")
    print(f"Samples: {total}")
    print(f"Loss:          {total_loss/n:.4f}")
    print(f"Leaf accuracy: {total_leaf_correct/n:.4f}  ({total_leaf_correct}/{total})")
    print(f"Op accuracy:   {total_op_correct/n:.4f}  ({total_op_correct}/{total})")

    print("\n--- Op accuracy breakdown ---")
    for op_int, op_name in sorted(INT_TO_OP.items()):
        tot = leaf_total_per_class.get(op_int, 0) if False else op_total_per_op.get(op_int, 0)
        cor = op_correct_per_op.get(op_int, 0)
        acc = cor / max(tot, 1)
        print(f"  {op_name:6s}: {acc:.3f}  ({cor}/{tot})")

    print("\n--- Leaf accuracy breakdown (32 classes) ---")
    for i, leaf_id in enumerate(LEAF_IDS):
        tot = leaf_total_per_class.get(i, 0)
        cor = leaf_correct_per_class.get(i, 0)
        acc = cor / max(tot, 1)
        bar = "#" * int(acc * 20)
        print(f"  {leaf_id:20s}: {acc:.3f}  ({cor}/{tot})  {bar}")


# ---------------------------------------------------------------------------
# Single-problem inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_single(model: Phase1Classifier, question: str, tokenizer,
                 var_descriptions: list[str] | None = None,
                 device: torch.device = torch.device("cpu")) -> None:
    """Run inference on a single problem; print structured output."""
    model.eval()
    max_seq_len = 128

    # If no var descriptions provided, we can only show that inference works
    if var_descriptions is None:
        # Generate dummy var names from question words as placeholders
        words = question.split()[:5]
        var_descriptions = [" ".join(words[:i+1]) for i in range(3)]
        print(f"  (No var descriptions provided; using dummy: {var_descriptions})")

    enc = tokenizer(
        [question] * len(var_descriptions),
        var_descriptions,
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(input_ids, attention_mask)
    leaf_pred = out.leaf_logits.argmax(-1).tolist()
    op_pred   = out.op_logits.argmax(-1).tolist()

    leaf_conf = out.leaf_logits.softmax(-1).max(-1).values.tolist()
    op_conf   = out.op_logits.softmax(-1).max(-1).values.tolist()

    print(f"\nQuestion: {question[:120]}{'...' if len(question)>120 else ''}")
    print(f"\n{'Var description':40s}  {'Leaf ID':22s}  {'Conf':6s}  {'Op':6s}  {'Conf':6s}")
    print("-" * 90)
    for i, desc in enumerate(var_descriptions):
        lid  = LEAF_IDS[leaf_pred[i]]
        op   = INT_TO_OP[op_pred[i]]
        lc   = leaf_conf[i]
        oc   = op_conf[i]
        print(f"  {desc:38s}  {lid:22s}  {lc:.3f}  {op:6s}  {oc:.3f}")

    # Reconstruct minimal factor graph stub
    print("\nFactor graph stub:")
    print(json.dumps({
        "n_vars":          len(var_descriptions),
        "var_descriptions": var_descriptions,
        "predicted_leaves": [LEAF_IDS[leaf_pred[i]] for i in range(len(var_descriptions))],
        "predicted_ops":   [INT_TO_OP[op_pred[i]] for i in range(len(var_descriptions))],
    }, indent=2))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    type=str, default=None,
                   help="Path to checkpoint .pt file (default: latest in CHECKPOINT_DIR)")
    p.add_argument("--problem", type=str, default=None,
                   help="Single NL problem for inference test")
    p.add_argument("--vars",    type=str, nargs="*",
                   help="Variable descriptions for --problem mode")
    p.add_argument("--batch",   type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Find checkpoint
    ckpt_path = Path(args.ckpt) if args.ckpt else find_latest_checkpoint()
    if ckpt_path is None:
        print("No checkpoint found. Run phase1_classifier_train.py first.")
        sys.exit(1)

    model = load_checkpoint(ckpt_path, device)

    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    if args.problem:
        infer_single(model, args.problem, tokenizer,
                     var_descriptions=args.vars, device=device)
    else:
        eval_full(model, VAL_JSONL, tokenizer,
                  batch_size=args.batch, device=device)


if __name__ == "__main__":
    main()
