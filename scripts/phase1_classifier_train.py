"""Phase 1 Classifier: Training script.

Usage:
  # Smoke (50 steps, 100 records):
  python scripts/phase1_classifier_train.py --smoke

  # Full training:
  python scripts/phase1_classifier_train.py --steps 2000

Key env vars (all optional):
  STEPS=2000
  BATCH=32
  LR=3e-5
  FREEZE_BACKBONE=0     set 1 to train heads only (faster smoke)
  DEVICE=cpu            override device (cpu|cuda)
  CHECKPOINT_DIR=.cache/phase1_ckpts
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR      = REPO_ROOT / ".cache"
TRAIN_JSONL    = CACHE_DIR / "gsm8k_phase1_classifier_train.jsonl"
VAL_JSONL      = CACHE_DIR / "gsm8k_phase1_classifier_val.jsonl"
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", str(CACHE_DIR / "phase1_ckpts")))

sys.path.insert(0, str(REPO_ROOT))
from mycelium.phase1_classifier import (
    Phase1Classifier, Phase1Config,
    LEAF_TO_INT, OP_TO_INT, N_LEAVES, N_OPS,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VariableDataset(Dataset):
    """One sample per (problem, variable) pair.

    Each sample:
      input_ids, attention_mask, leaf_label (int), op_label (int)

    Encoding:  [CLS] question [SEP] var_description [SEP]
    """

    def __init__(self, jsonl_path: Path, tokenizer, max_seq_len: int = 128,
                 max_problems: int | None = None):
        from transformers import DistilBertTokenizerFast
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[dict] = []

        n_problems = 0
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                n_problems += 1
                if max_problems and n_problems > max_problems:
                    break
                question = r["question"]
                for v in r["variables"]:
                    self.samples.append({
                        "question":    question,
                        "var_text":    v["text"],
                        "leaf_label":  v["leaf_int"],
                        "op_label":    v["op_int"],
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        enc = self.tokenizer(
            s["question"], s["var_text"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "leaf_label":     torch.tensor(s["leaf_label"], dtype=torch.long),
            "op_label":       torch.tensor(s["op_label"],   dtype=torch.long),
        }


def build_leaf_counts(jsonl_path: Path, max_problems: int | None = None) -> list[int]:
    """Count occurrences of each leaf class in training data."""
    counts: Counter = Counter()
    n = 0
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            n += 1
            if max_problems and n > max_problems:
                break
            for v in r["variables"]:
                counts[v["leaf_int"]] += 1
    return [counts.get(i, 0) for i in range(N_LEAVES)]


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: Phase1Classifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    leaf_correct = op_correct = total = 0
    total_loss = leaf_loss_sum = op_loss_sum = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        leaf_labels    = batch["leaf_label"].to(device)
        op_labels      = batch["op_label"].to(device)

        out    = model(input_ids, attention_mask)
        losses = model.compute_loss(
            out.leaf_logits, out.op_logits, leaf_labels, op_labels
        )

        B = input_ids.size(0)
        total             += B
        total_loss        += losses["loss"].item()      * B
        leaf_loss_sum     += losses["leaf_loss"].item() * B
        op_loss_sum       += losses["op_loss"].item()   * B
        leaf_correct      += (out.leaf_logits.argmax(-1) == leaf_labels).sum().item()
        op_correct        += (out.op_logits.argmax(-1)   == op_labels).sum().item()

    n = max(total, 1)
    return {
        "loss":      total_loss    / n,
        "leaf_loss": leaf_loss_sum / n,
        "op_loss":   op_loss_sum   / n,
        "leaf_acc":  leaf_correct  / n,
        "op_acc":    op_correct    / n,
        "n_samples": total,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device_str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")

    steps         = int(os.environ.get("STEPS",  args.steps))
    batch_size    = int(os.environ.get("BATCH",  args.batch))
    lr            = float(os.environ.get("LR",   args.lr))
    freeze        = bool(int(os.environ.get("FREEZE_BACKBONE", "0")))
    max_seq_len   = 128
    eval_every    = args.eval_every
    save_every    = args.save_every

    max_problems  = 100 if args.smoke else None

    # Tokenizer
    from transformers import DistilBertTokenizerFast
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Datasets
    print("Building datasets...")
    train_ds = VariableDataset(TRAIN_JSONL, tokenizer, max_seq_len, max_problems)
    val_ds   = VariableDataset(VAL_JSONL,   tokenizer, max_seq_len,
                               max_problems=50 if args.smoke else None)
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False,
                              num_workers=0)

    # Leaf counts for class weighting
    leaf_counts = build_leaf_counts(TRAIN_JSONL, max_problems)
    print(f"  Leaf count range: min={min(leaf_counts)}, max={max(leaf_counts)}")

    # Model
    print("Building model...")
    config = Phase1Config(
        freeze_backbone=freeze,
        leaf_counts=leaf_counts,
        leaf_weight_strategy="inverse_sqrt",
    )
    model = Phase1Classifier(config).to(device)
    pcount = model.param_count()
    print(f"  Params: backbone={pcount['backbone']:,}, heads={pcount['heads']:,}, "
          f"total={pcount['total']:,}, trainable={pcount['trainable']:,}")

    # Optimizer with differential LR: backbone gets 10× lower LR
    if freeze:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
    else:
        backbone_params = list(model.backbone.parameters())
        head_params     = list(model.leaf_head.parameters()) + \
                          list(model.op_head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr / 10},
            {"params": head_params,     "lr": lr},
        ], weight_decay=0.01)

    # Cosine LR schedule with warmup
    warmup_steps = min(100, steps // 10)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    step        = 0
    train_iter  = iter(train_loader)
    t0          = time.time()
    log_loss    = log_leaf_loss = log_op_loss = 0.0
    log_n       = 0

    print(f"\nTraining for {steps} steps (batch={batch_size}, lr={lr}, freeze={freeze})")
    print(f"{'Step':>6}  {'loss':>8}  {'leaf_loss':>10}  {'op_loss':>8}  "
          f"{'lr':>8}  {'elapsed':>8}")

    while step < steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        leaf_labels    = batch["leaf_label"].to(device)
        op_labels      = batch["op_label"].to(device)

        optimizer.zero_grad()
        out    = model(input_ids, attention_mask)
        losses = model.compute_loss(
            out.leaf_logits, out.op_logits, leaf_labels, op_labels
        )
        losses["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1
        log_loss      += losses["loss"].item()
        log_leaf_loss += losses["leaf_loss"].item()
        log_op_loss   += losses["op_loss"].item()
        log_n         += 1

        if step % 10 == 0 or step == 1:
            cur_lr    = scheduler.get_last_lr()[-1]
            elapsed   = time.time() - t0
            avg_loss  = log_loss      / log_n
            avg_leaf  = log_leaf_loss / log_n
            avg_op    = log_op_loss   / log_n
            print(f"{step:>6}  {avg_loss:>8.4f}  {avg_leaf:>10.4f}  {avg_op:>8.4f}  "
                  f"{cur_lr:>8.2e}  {elapsed:>7.1f}s")
            log_loss = log_leaf_loss = log_op_loss = 0.0
            log_n = 0

        if step % eval_every == 0 or step == steps:
            val_metrics = evaluate(model, val_loader, device)
            print(f"\n  [eval step {step}]  "
                  f"loss={val_metrics['loss']:.4f}  "
                  f"leaf_acc={val_metrics['leaf_acc']:.3f}  "
                  f"op_acc={val_metrics['op_acc']:.3f}  "
                  f"(n={val_metrics['n_samples']})\n")
            model.train()

        if step % save_every == 0 or step == steps:
            ckpt_path = CHECKPOINT_DIR / f"phase1_classifier_step{step}.pt"
            # weights_only=False needed to load back the Phase1Config dataclass
            torch.save({
                "step":            step,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config":          config,
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    elapsed = time.time() - t0
    print(f"\nTraining complete: {steps} steps in {elapsed:.1f}s "
          f"({elapsed/steps:.2f}s/step)")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke",      action="store_true",
                   help="50-step smoke test on 100-problem subset")
    p.add_argument("--steps",      type=int,   default=2000)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-5)
    p.add_argument("--eval-every", type=int,   default=50,  dest="eval_every")
    p.add_argument("--save-every", type=int,   default=500, dest="save_every")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        args.steps      = 50
        args.eval_every = 25
        args.save_every = 50
        print("=== SMOKE MODE: 50 steps, 100-problem subset ===")
    train(args)
