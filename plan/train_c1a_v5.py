"""
C1-A v5: Boundary Detector with Soft Probability Targets

Key changes from v4:
- Uses v5 extraction data with transition-point boundaries
- Trains on soft probability targets (0.0-1.0) instead of hard 0/1
- BCE loss against soft targets for calibrated confidence
- Preserves transition direction metadata for later use
"""

import os
import json
import argparse
from pathlib import Path

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


class SoftFocalLoss(nn.Module):
    """Focal loss adapted for soft probability targets."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE against soft targets
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Focal weighting based on prediction confidence
        pt = torch.exp(-bce_loss)

        # Alpha weighting - weight by target probability (soft labels)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


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


class C1DatasetV5(Dataset):
    """Dataset for C1-A training on v5 soft probability targets."""

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

        print(f"Loaded {len(all_records)} total records from v5 data")

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

        # v5: Soft probability targets
        boundary_probs = record.get("boundary_probs", [])
        n_input = record.get("n_input_tokens", len(boundary_probs))

        # Create soft target tensor
        boundary_target = torch.zeros(self.max_length)

        if n_input > 0 and actual_len > 0 and len(boundary_probs) > 0:
            scale = actual_len / n_input
            for i, prob in enumerate(boundary_probs):
                if prob > 0.0:  # Only map non-zero probabilities
                    mapped_idx = min(int(i * scale), actual_len - 1)
                    # Use the soft probability value, not just 1.0
                    # Take max in case multiple source positions map to same target
                    boundary_target[mapped_idx] = max(boundary_target[mapped_idx].item(), prob)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "boundary_target": boundary_target,
        }


def compute_boundary_metrics(logits, targets, mask, thresholds=[0.3, 0.5, 0.7]):
    """Compute boundary metrics at multiple thresholds."""
    probs = torch.sigmoid(logits)
    valid_mask = mask.bool()

    metrics = {}
    for thresh in thresholds:
        preds = (probs > thresh).float()
        # For soft targets, threshold them too for metric computation
        hard_targets = (targets > thresh).float()

        preds_valid = preds[valid_mask]
        target_valid = hard_targets[valid_mask]

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

    # Count actual boundaries (soft target > 0.5)
    metrics["n_actual"] = (targets[valid_mask] > 0.5).sum().item()

    # Calibration: mean predicted prob vs mean target prob for boundary tokens
    boundary_mask = targets[valid_mask] > 0.1
    if boundary_mask.sum() > 0:
        pred_probs_at_boundaries = probs[valid_mask][boundary_mask]
        target_probs_at_boundaries = targets[valid_mask][boundary_mask]
        metrics["calibration_error"] = (pred_probs_at_boundaries.mean() - target_probs_at_boundaries.mean()).abs().item()
    else:
        metrics["calibration_error"] = 0.0

    return metrics


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
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading v5 data from {args.data_dir}...")
    train_dataset = C1DatasetV5(args.data_dir, tokenizer, args.max_length, split="train")
    val_dataset = C1DatasetV5(args.data_dir, tokenizer, args.max_length, split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    config = {
        "model_name": args.model_name,
        "lora_rank": args.lora_rank,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "epochs": args.epochs,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "data_version": "v5",
        "soft_targets": True,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nLoading {args.model_name}...")
    backbone = AutoModel.from_pretrained(args.model_name, trust_remote_code=True,
                                         torch_dtype=torch.float32)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    model = C1ABoundaryModel(backbone, hidden_dim=backbone.config.hidden_size)
    model = model.to(device)

    # Soft focal loss for soft probability targets
    soft_focal_loss = SoftFocalLoss(alpha=0.75, gamma=2.0)

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

    print(f"\nStarting C1-A v5 training for {args.epochs} epochs...")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"Using soft probability targets from v5 extraction")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"C1-A v5 Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            boundary_target = batch["boundary_target"].to(device)

            logits = model(input_ids, attention_mask)

            valid_mask = attention_mask.bool()
            loss = soft_focal_loss(logits[valid_mask], boundary_target[valid_mask])
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
                loss = soft_focal_loss(logits[valid_mask], boundary_target[valid_mask])
                val_loss += loss.item()

                metrics = compute_boundary_metrics(logits, boundary_target, attention_mask)
                all_metrics.append(metrics)

        val_loss /= len(val_loader)

        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        print(f"\n{'='*60}")
        print(f"C1-A v5 Epoch {epoch} Summary")
        print(f"{'='*60}")
        print(f"Train Loss: {total_loss/n_batches:.4f} | Val Loss: {val_loss:.4f}")
        print(f"\nBoundary Detection @ threshold 0.5:")
        print(f"  Precision: {avg_metrics['precision']:.4f}")
        print(f"  Recall:    {avg_metrics['recall']:.4f}")
        print(f"  F1:        {avg_metrics['f1']:.4f}")
        print(f"\nF1 @ other thresholds:")
        print(f"  0.3: {avg_metrics['f1_0.3']:.4f} | 0.7: {avg_metrics['f1_0.7']:.4f}")
        print(f"\nPredictions: {avg_metrics['n_pred']:.0f} | Actual: {avg_metrics['n_actual']:.0f}")
        print(f"Calibration Error: {avg_metrics['calibration_error']:.4f}")

        epoch_log = {
            "epoch": epoch,
            "train_loss": total_loss / n_batches,
            "val_loss": val_loss,
            **avg_metrics,
        }
        training_log.append(epoch_log)

        if avg_metrics["f1"] > best_f1:
            best_f1 = avg_metrics["f1"]
            patience_counter = 0

            best_path = output_path / "best_checkpoint"
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

        with open(output_path / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

    print(f"\nC1-A v5 Training Complete!")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Checkpoint saved to {output_path}")

    results = {"best_f1": best_f1, "final_epoch": epoch, "data_version": "v5"}
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
