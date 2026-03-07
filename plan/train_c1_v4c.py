"""
C1 Training v4c - Split Architecture

Trains two separate models:
- C1-A: Per-token boundary detector
- C1-B: Sequence analyzer (co-transition stats + BP depth)

Usage:
    python train_c1_v4c.py --data-dir /data/c1_training_v4 --output-dir /data/c1_model_v4c
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from tqdm import tqdm


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ============================================================================
# C1-A: Boundary Detector
# ============================================================================

class C1ABoundaryModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int = 896):
        super().__init__()
        self.backbone = backbone
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        boundary_logits = self.boundary_head(hidden_states).squeeze(-1)
        return boundary_logits


# ============================================================================
# C1-B: Sequence Analyzer
# ============================================================================

class C1BSequenceModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int = 896):
        super().__init__()
        self.backbone = backbone

        # Head 2: Co-transition statistics (4 values)
        self.co_transition_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

        # Head 3: BP depth classification (3 classes)
        self.bp_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Co-transition predictions with activations
        co_raw = self.co_transition_head(pooled)
        co_pred = torch.stack([
            F.softplus(co_raw[:, 0]),      # n_co_transitions
            torch.sigmoid(co_raw[:, 1]),    # reading_ratio
            F.softplus(co_raw[:, 2]),       # mean_spacing
            F.softplus(co_raw[:, 3]),       # burstiness
        ], dim=1)

        # BP depth logits
        bp_logits = self.bp_depth_head(pooled)

        return {"co_pred": co_pred, "bp_logits": bp_logits}


# ============================================================================
# Dataset
# ============================================================================

class C1Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 512,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        all_records = []
        data_path = Path(data_dir)

        for json_file in sorted(data_path.glob("*.json")):
            if "stats" in str(json_file) or "global" in str(json_file):
                continue
            with open(json_file) as f:
                chunk = json.load(f)
                all_records.extend(chunk.get("records", []))

        print(f"Loaded {len(all_records)} total records")

        # Stratified split by BP depth
        np.random.seed(seed)
        by_depth = {1: [], 2: [], 3: []}
        for i, r in enumerate(all_records):
            depth = r.get("bp_depth", 1)
            by_depth[depth].append(i)

        train_indices = []
        val_indices = []
        for depth, indices in by_depth.items():
            np.random.shuffle(indices)
            n_val = int(len(indices) * val_ratio)
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        self.indices = train_indices if split == "train" else val_indices
        self.all_records = all_records
        print(f"{split} set: {len(self.indices)} examples")

        # Compute z-score stats for co-transition targets (train only)
        if split == "train":
            self.co_stats = self._compute_co_stats(train_indices)
        else:
            self.co_stats = None

    def set_co_stats(self, stats: dict):
        """Set z-score stats from training set."""
        self.co_stats = stats

    def _compute_co_stats(self, indices: list) -> dict:
        """Compute mean/std for z-score normalization."""
        values = {k: [] for k in ["n_co_transitions", "reading_ratio", "mean_spacing", "burstiness"]}
        for idx in indices:
            r = self.all_records[idx]
            values["n_co_transitions"].append(r.get("n_co_transitions", 0))
            values["reading_ratio"].append(r.get("reading_ratio", 0.5))
            values["mean_spacing"].append(r.get("mean_spacing", 100))
            values["burstiness"].append(r.get("burstiness", 0))

        stats = {}
        for k, v in values.items():
            arr = np.array(v)
            stats[k] = {"mean": float(arr.mean()), "std": float(arr.std() + 1e-8)}
        return stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record = self.all_records[self.indices[idx]]

        text = record["problem_text"]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        actual_len = attention_mask.sum().item()

        # Boundary targets
        boundary_probs = record.get("boundary_probs", [])
        n_input = record.get("n_input_tokens", len(boundary_probs))
        boundary_target = torch.zeros(self.max_length)

        if n_input > 0 and actual_len > 0:
            scale = actual_len / n_input
            for i, prob in enumerate(boundary_probs):
                if prob > 0.5:
                    mapped_idx = min(int(i * scale), actual_len - 1)
                    boundary_target[mapped_idx] = 1.0

        # Co-transition targets (z-score normalized if stats available)
        raw_co = [
            record.get("n_co_transitions", 0),
            record.get("reading_ratio", 0.5),
            record.get("mean_spacing", 100),
            record.get("burstiness", 0),
        ]

        if self.co_stats:
            keys = ["n_co_transitions", "reading_ratio", "mean_spacing", "burstiness"]
            normalized = []
            for k, v in zip(keys, raw_co):
                normalized.append((v - self.co_stats[k]["mean"]) / self.co_stats[k]["std"])
            co_targets = torch.tensor(normalized, dtype=torch.float32)
        else:
            co_targets = torch.tensor(raw_co, dtype=torch.float32)

        # Raw targets for MAE computation
        co_targets_raw = torch.tensor(raw_co, dtype=torch.float32)

        bp_depth = record.get("bp_depth", 1) - 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "boundary_target": boundary_target,
            "co_targets": co_targets,
            "co_targets_raw": co_targets_raw,
            "bp_depth": bp_depth,
            "actual_len": actual_len,
        }


# ============================================================================
# Training Functions
# ============================================================================

def compute_boundary_metrics(logits: torch.Tensor, targets: torch.Tensor,
                             mask: torch.Tensor, thresholds: list = [0.3, 0.5, 0.7]) -> dict:
    """Compute boundary metrics at multiple thresholds."""
    probs = torch.sigmoid(logits)
    valid_mask = mask.bool()

    metrics = {}
    for thresh in thresholds:
        preds = (probs > thresh).float()
        preds_valid = preds[valid_mask]
        target_valid = targets[valid_mask]

        tp = ((preds_valid == 1) & (target_valid == 1)).sum().float()
        fp = ((preds_valid == 1) & (target_valid == 0)).sum().float()
        fn = ((preds_valid == 0) & (target_valid == 1)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        suffix = f"_{thresh}" if thresh != 0.5 else ""
        metrics[f"precision{suffix}"] = precision.item()
        metrics[f"recall{suffix}"] = recall.item()
        metrics[f"f1{suffix}"] = f1.item()
        metrics[f"n_pred{suffix}"] = preds_valid.sum().item()

    metrics["n_actual"] = targets[valid_mask].sum().item()
    return metrics


def train_c1a(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    args,
):
    """Train C1-A boundary detector."""
    print("\n" + "="*70)
    print("Training C1-A: Boundary Detector")
    print("="*70)

    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)

    # Separate LRs for LoRA and head
    lora_params = [p for n, p in model.named_parameters() if "lora" in n.lower()]
    head_params = [p for n, p in model.named_parameters() if "boundary_head" in n]

    optimizer = AdamW([
        {"params": lora_params, "lr": 2e-4},
        {"params": head_params, "lr": 5e-4},
    ], weight_decay=0.01)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    training_log = []
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"C1-A Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            boundary_target = batch["boundary_target"].to(device)

            logits = model(input_ids, attention_mask)

            valid_mask = attention_mask.bool()
            loss = focal_loss(logits[valid_mask], boundary_target[valid_mask])
            loss = loss / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        all_metrics = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                boundary_target = batch["boundary_target"].to(device)

                logits = model(input_ids, attention_mask)
                valid_mask = attention_mask.bool()
                loss = focal_loss(logits[valid_mask], boundary_target[valid_mask])
                val_loss += loss.item()

                metrics = compute_boundary_metrics(logits, boundary_target, attention_mask)
                all_metrics.append(metrics)

        val_loss /= len(val_loader)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        # Print summary
        print(f"\n{'='*50}")
        print(f"C1-A Epoch {epoch} Summary")
        print(f"{'='*50}")
        print(f"Train Loss: {total_loss/n_batches:.4f} | Val Loss: {val_loss:.4f}")
        print(f"\nBoundary Detection @ threshold 0.5:")
        print(f"  Precision: {avg_metrics['precision']:.4f}")
        print(f"  Recall:    {avg_metrics['recall']:.4f}")
        print(f"  F1:        {avg_metrics['f1']:.4f}")
        print(f"\nF1 @ other thresholds:")
        print(f"  0.3: {avg_metrics['f1_0.3']:.4f} | 0.7: {avg_metrics['f1_0.7']:.4f}")
        print(f"\nPredictions: {avg_metrics['n_pred']:.0f} | Actual: {avg_metrics['n_actual']:.0f}")

        epoch_log = {
            "epoch": epoch,
            "train_loss": total_loss / n_batches,
            "val_loss": val_loss,
            **avg_metrics,
        }
        training_log.append(epoch_log)

        # Save best model (by F1)
        if avg_metrics["f1"] > best_f1:
            best_f1 = avg_metrics["f1"]
            patience_counter = 0

            best_path = output_dir / "best_checkpoint"
            best_path.mkdir(parents=True, exist_ok=True)
            model.backbone.save_pretrained(best_path / "lora_adapters")
            torch.save({"boundary_head": model.boundary_head.state_dict()},
                      best_path / "head_weights.pt")
            print(f"  Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        with open(output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

    return {"best_f1": best_f1, "training_log": training_log}


def train_c1b(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    co_stats: dict,
    args,
):
    """Train C1-B sequence analyzer."""
    print("\n" + "="*70)
    print("Training C1-B: Sequence Analyzer")
    print("="*70)

    lora_params = [p for n, p in model.named_parameters() if "lora" in n.lower()]
    head_params = [p for n, p in model.named_parameters()
                   if "co_transition_head" in n or "bp_depth_head" in n]

    optimizer = AdamW([
        {"params": lora_params, "lr": 2e-4},
        {"params": head_params, "lr": 5e-4},
    ], weight_decay=0.01)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    training_log = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_co_loss = 0
        total_bp_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"C1-B Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            co_targets = batch["co_targets"].to(device)
            bp_target = batch["bp_depth"].to(device)

            outputs = model(input_ids, attention_mask)

            # Co-transition loss (Huber on z-scored targets)
            # Need to z-score the predictions too
            co_pred = outputs["co_pred"]
            co_pred_norm = torch.zeros_like(co_pred)
            keys = ["n_co_transitions", "reading_ratio", "mean_spacing", "burstiness"]
            for i, k in enumerate(keys):
                co_pred_norm[:, i] = (co_pred[:, i] - co_stats[k]["mean"]) / co_stats[k]["std"]

            co_loss = F.huber_loss(co_pred_norm, co_targets, delta=1.0)

            # BP depth loss
            bp_loss = F.cross_entropy(outputs["bp_logits"], bp_target)

            # Combined loss (1:1 weighting)
            loss = (co_loss + bp_loss) / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += (co_loss + bp_loss).item()
            total_co_loss += co_loss.item()
            total_bp_loss += bp_loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{(co_loss + bp_loss).item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        val_co_loss = 0
        val_bp_loss = 0

        # MAE accumulators
        mae_sums = {"n_co_transitions": 0, "reading_ratio": 0, "mean_spacing": 0, "burstiness": 0}
        bp_correct = 0
        bp_total = 0
        bp_per_class = {0: {"correct": 0, "total": 0}, 1: {"correct": 0, "total": 0}, 2: {"correct": 0, "total": 0}}
        n_val = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                co_targets = batch["co_targets"].to(device)
                co_targets_raw = batch["co_targets_raw"].to(device)
                bp_target = batch["bp_depth"].to(device)

                outputs = model(input_ids, attention_mask)
                co_pred = outputs["co_pred"]

                # Z-score predictions for loss
                co_pred_norm = torch.zeros_like(co_pred)
                keys = ["n_co_transitions", "reading_ratio", "mean_spacing", "burstiness"]
                for i, k in enumerate(keys):
                    co_pred_norm[:, i] = (co_pred[:, i] - co_stats[k]["mean"]) / co_stats[k]["std"]

                co_loss = F.huber_loss(co_pred_norm, co_targets, delta=1.0)
                bp_loss = F.cross_entropy(outputs["bp_logits"], bp_target)

                val_loss += (co_loss + bp_loss).item()
                val_co_loss += co_loss.item()
                val_bp_loss += bp_loss.item()

                # MAE on raw values
                for i, k in enumerate(keys):
                    mae_sums[k] += (co_pred[:, i] - co_targets_raw[:, i]).abs().sum().item()

                # BP accuracy
                bp_preds = outputs["bp_logits"].argmax(dim=1)
                bp_correct += (bp_preds == bp_target).sum().item()
                bp_total += len(bp_target)

                for c in range(3):
                    mask = bp_target == c
                    bp_per_class[c]["total"] += mask.sum().item()
                    bp_per_class[c]["correct"] += ((bp_preds == bp_target) & mask).sum().item()

                n_val += len(bp_target)

        val_loss /= len(val_loader)
        val_co_loss /= len(val_loader)
        val_bp_loss /= len(val_loader)

        mae = {k: v / n_val for k, v in mae_sums.items()}
        bp_acc = bp_correct / bp_total
        bp_per_class_acc = {c: d["correct"] / max(1, d["total"]) for c, d in bp_per_class.items()}

        # Print summary
        print(f"\n{'='*50}")
        print(f"C1-B Epoch {epoch} Summary")
        print(f"{'='*50}")
        print(f"Train Loss: {total_loss/n_batches:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Co-trans: {val_co_loss:.4f} | BP: {val_bp_loss:.4f}")
        print(f"\nCo-transition MAE:")
        print(f"  n_co_transitions: {mae['n_co_transitions']:.4f}")
        print(f"  reading_ratio:    {mae['reading_ratio']:.4f}")
        print(f"  mean_spacing:     {mae['mean_spacing']:.4f}")
        print(f"  burstiness:       {mae['burstiness']:.4f}")
        print(f"\nBP Depth Accuracy: {bp_acc:.4f}")
        print(f"  Class 0 (depth 1): {bp_per_class_acc[0]:.4f}")
        print(f"  Class 1 (depth 2): {bp_per_class_acc[1]:.4f}")
        print(f"  Class 2 (depth 3): {bp_per_class_acc[2]:.4f}")

        epoch_log = {
            "epoch": epoch,
            "train_loss": total_loss / n_batches,
            "val_loss": val_loss,
            "val_co_loss": val_co_loss,
            "val_bp_loss": val_bp_loss,
            "mae_n_co_transitions": mae["n_co_transitions"],
            "mae_reading_ratio": mae["reading_ratio"],
            "mae_mean_spacing": mae["mean_spacing"],
            "mae_burstiness": mae["burstiness"],
            "bp_accuracy": bp_acc,
            "bp_acc_class_0": bp_per_class_acc[0],
            "bp_acc_class_1": bp_per_class_acc[1],
            "bp_acc_class_2": bp_per_class_acc[2],
        }
        training_log.append(epoch_log)

        # Save best model (by val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_path = output_dir / "best_checkpoint"
            best_path.mkdir(parents=True, exist_ok=True)
            model.backbone.save_pretrained(best_path / "lora_adapters")
            torch.save({
                "co_transition_head": model.co_transition_head.state_dict(),
                "bp_depth_head": model.bp_depth_head.state_dict(),
            }, best_path / "head_weights.pt")
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        with open(output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

    return {
        "best_val_loss": best_val_loss,
        "best_bp_accuracy": max(log["bp_accuracy"] for log in training_log),
        "best_mae_n_co": min(log["mae_n_co_transitions"] for log in training_log),
        "training_log": training_log,
    }


def generate_comparison(c1a_results: dict, c1b_results: dict, output_dir: Path):
    """Generate comparison.json with v4b baseline."""

    # v4b baseline (from previous run)
    v4b_baseline = {
        "boundary_f1": 0.277,  # Best before degradation
        "bp_accuracy": 0.582,  # Peak
        "mae_n_co_transitions": 4.07,
    }

    comparison = {
        "v4b_baseline": v4b_baseline,
        "v4c_c1a": {
            "boundary_f1": c1a_results["best_f1"],
            "improvement": c1a_results["best_f1"] - v4b_baseline["boundary_f1"],
        },
        "v4c_c1b": {
            "bp_accuracy": c1b_results["best_bp_accuracy"],
            "mae_n_co_transitions": c1b_results["best_mae_n_co"],
            "bp_improvement": c1b_results["best_bp_accuracy"] - v4b_baseline["bp_accuracy"],
            "mae_improvement": v4b_baseline["mae_n_co_transitions"] - c1b_results["best_mae_n_co"],
        },
        "success_criteria": {
            "boundary_f1_exceeded": c1a_results["best_f1"] > v4b_baseline["boundary_f1"],
            "bp_accuracy_exceeded": c1b_results["best_bp_accuracy"] > v4b_baseline["bp_accuracy"],
            "mae_improved": c1b_results["best_mae_n_co"] < v4b_baseline["mae_n_co_transitions"],
        },
        "split_was_right_call": all([
            c1a_results["best_f1"] > v4b_baseline["boundary_f1"],
            c1b_results["best_bp_accuracy"] > v4b_baseline["bp_accuracy"],
            c1b_results["best_mae_n_co"] < v4b_baseline["mae_n_co_transitions"],
        ]),
    }

    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "="*70)
    print("COMPARISON: v4c (split) vs v4b (unified)")
    print("="*70)
    print(f"\nBoundary F1:       {c1a_results['best_f1']:.4f} vs {v4b_baseline['boundary_f1']:.4f} "
          f"({'BETTER' if c1a_results['best_f1'] > v4b_baseline['boundary_f1'] else 'WORSE'})")
    print(f"BP Accuracy:       {c1b_results['best_bp_accuracy']:.4f} vs {v4b_baseline['bp_accuracy']:.4f} "
          f"({'BETTER' if c1b_results['best_bp_accuracy'] > v4b_baseline['bp_accuracy'] else 'WORSE'})")
    print(f"Co-trans MAE:      {c1b_results['best_mae_n_co']:.4f} vs {v4b_baseline['mae_n_co_transitions']:.4f} "
          f"({'BETTER' if c1b_results['best_mae_n_co'] < v4b_baseline['mae_n_co_transitions'] else 'WORSE'})")
    print(f"\nSplit was right call: {comparison['split_was_right_call']}")

    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_path = Path(args.output_dir)
    c1a_output = output_path / "c1a_boundary"
    c1b_output = output_path / "c1b_sequence"
    c1a_output.mkdir(parents=True, exist_ok=True)
    c1b_output.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets (same split for both models)
    print(f"\nLoading data from {args.data_dir}...")
    train_dataset = C1Dataset(args.data_dir, tokenizer, args.max_length, split="train")
    val_dataset = C1Dataset(args.data_dir, tokenizer, args.max_length, split="val")
    val_dataset.set_co_stats(train_dataset.co_stats)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Save config
    config = {
        "model_name": args.model_name,
        "lora_rank": args.lora_rank,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "epochs": args.epochs,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "co_stats": train_dataset.co_stats,
    }

    # =========================================================================
    # Train C1-A: Boundary Detector
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: Training C1-A Boundary Detector")
    print("="*70)

    backbone_a = AutoModel.from_pretrained(args.model_name, trust_remote_code=True,
                                           torch_dtype=torch.float32)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    backbone_a = get_peft_model(backbone_a, lora_config)
    backbone_a.print_trainable_parameters()

    model_a = C1ABoundaryModel(backbone_a, hidden_dim=backbone_a.config.hidden_size)
    model_a = model_a.to(device)

    with open(c1a_output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    c1a_results = train_c1a(model_a, train_loader, val_loader, device, c1a_output, args)

    # Free memory
    del model_a, backbone_a
    torch.cuda.empty_cache()

    # =========================================================================
    # Train C1-B: Sequence Analyzer
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2: Training C1-B Sequence Analyzer")
    print("="*70)

    backbone_b = AutoModel.from_pretrained(args.model_name, trust_remote_code=True,
                                           torch_dtype=torch.float32)
    backbone_b = get_peft_model(backbone_b, lora_config)
    backbone_b.print_trainable_parameters()

    model_b = C1BSequenceModel(backbone_b, hidden_dim=backbone_b.config.hidden_size)
    model_b = model_b.to(device)

    with open(c1b_output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    c1b_results = train_c1b(model_b, train_loader, val_loader, device, c1b_output,
                            train_dataset.co_stats, args)

    # =========================================================================
    # Generate comparison
    # =========================================================================
    comparison = generate_comparison(c1a_results, c1b_results, output_path)

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"C1-A best F1: {c1a_results['best_f1']:.4f}")
    print(f"C1-B best BP accuracy: {c1b_results['best_bp_accuracy']:.4f}")
    print(f"C1-B best MAE: {c1b_results['best_mae_n_co']:.4f}")
    print(f"\nOutputs saved to {output_path}")


if __name__ == "__main__":
    main()
