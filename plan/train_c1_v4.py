"""
C1 Training Script v4 - Shadow Reader

Trains Qwen-0.5B + LoRA with 3 heads on distilled attention patterns:
- Head 1: Per-token boundary detection (binary focal loss)
- Head 2: Co-transition statistics (4 regression targets)
- Head 3: BP depth classification (3 classes)

Usage:
    python train_c1_v4.py --data-dir /data/c1_training_v4 --output-dir /data/c1_model_v4

Expects data downloaded from s3://mycelium-data/c1_training_v4/
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional
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
# Model Architecture
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in boundary detection."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)

        # Alpha weighting for positive class
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class C1Model(nn.Module):
    """
    C1: Structural analyzer with 3 prediction heads.

    Uses Qwen-0.5B backbone with LoRA adapters, plus three task-specific heads.
    """

    def __init__(
        self,
        backbone: nn.Module,
        hidden_dim: int = 896,  # Qwen2-0.5B hidden size
    ):
        super().__init__()
        self.backbone = backbone

        # Head 1: Per-token boundary probability
        # Higher dropout (0.3) to prevent overfitting with small dataset
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # Head 2: Co-transition statistics (4 values)
        # n_co_transitions, reading_ratio, mean_spacing, burstiness
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        # Get backbone hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Last hidden state: (batch, seq_len, hidden_dim)
        hidden_states = outputs.hidden_states[-1]

        # Head 1: Per-token boundary logits
        boundary_logits = self.boundary_head(hidden_states).squeeze(-1)

        # Pool for sequence-level predictions (mean over non-padding tokens)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Head 2: Co-transition statistics
        co_transition_raw = self.co_transition_head(pooled)
        # Apply appropriate activations:
        # [0] n_co_transitions: softplus (positive)
        # [1] reading_ratio: sigmoid (0-1)
        # [2] mean_spacing: softplus (positive)
        # [3] burstiness: softplus (positive)
        co_transition_pred = torch.stack([
            F.softplus(co_transition_raw[:, 0]),
            torch.sigmoid(co_transition_raw[:, 1]),
            F.softplus(co_transition_raw[:, 2]),
            F.softplus(co_transition_raw[:, 3]),
        ], dim=1)

        # Head 3: BP depth logits
        bp_depth_logits = self.bp_depth_head(pooled)

        return {
            "boundary_logits": boundary_logits,
            "co_transition_pred": co_transition_pred,
            "bp_depth_logits": bp_depth_logits,
        }


# ============================================================================
# Dataset
# ============================================================================

@dataclass
class C1Example:
    problem_text: str
    boundary_probs: list[float]
    n_co_transitions: float
    reading_ratio: float
    mean_spacing: float
    burstiness: float
    bp_depth: int


class C1Dataset(Dataset):
    """Dataset for C1 training."""

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

        # Load all records
        all_records = []
        data_path = Path(data_dir)

        for json_file in sorted(data_path.glob("*.json")):
            if "stats" in str(json_file) or "global" in str(json_file):
                continue

            with open(json_file) as f:
                chunk = json.load(f)
                records = chunk.get("records", [])
                all_records.extend(records)

        print(f"Loaded {len(all_records)} total records")

        # Stratified split by BP depth
        np.random.seed(seed)

        # Group by BP depth
        by_depth = {1: [], 2: [], 3: []}
        for i, r in enumerate(all_records):
            depth = r.get("bp_depth", 1)
            by_depth[depth].append(i)

        # Split each stratum
        train_indices = []
        val_indices = []

        for depth, indices in by_depth.items():
            np.random.shuffle(indices)
            n_val = int(len(indices) * val_ratio)
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        if split == "train":
            self.indices = train_indices
        else:
            self.indices = val_indices

        self.all_records = all_records
        print(f"{split} set: {len(self.indices)} examples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record = self.all_records[self.indices[idx]]

        # Tokenize
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

        # Get actual token count (before padding)
        actual_len = attention_mask.sum().item()

        # Boundary targets - align with tokens
        boundary_probs = record.get("boundary_probs", [])
        n_input = record.get("n_input_tokens", len(boundary_probs))

        # Create boundary target tensor
        boundary_target = torch.zeros(self.max_length)

        # Map original token boundaries to our tokenization
        # Heuristic: scale indices proportionally
        if n_input > 0 and actual_len > 0:
            scale = actual_len / n_input
            for i, prob in enumerate(boundary_probs):
                if prob > 0.5:  # Boundary token
                    mapped_idx = min(int(i * scale), actual_len - 1)
                    boundary_target[mapped_idx] = 1.0

        # Co-transition targets
        co_targets = torch.tensor([
            record.get("n_co_transitions", 0),
            record.get("reading_ratio", 0.5),
            record.get("mean_spacing", 100),
            record.get("burstiness", 0),
        ], dtype=torch.float32)

        # BP depth target (0-indexed)
        bp_depth = record.get("bp_depth", 1) - 1  # Convert 1,2,3 to 0,1,2

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "boundary_target": boundary_target,
            "co_targets": co_targets,
            "bp_depth": bp_depth,
            "actual_len": actual_len,
        }


# ============================================================================
# Training Loop
# ============================================================================

def compute_losses(
    outputs: dict,
    batch: dict,
    focal_loss_fn: FocalLoss,
) -> dict:
    """Compute all three head losses."""

    device = outputs["boundary_logits"].device

    # Head 1: Boundary focal loss (averaged over tokens)
    boundary_logits = outputs["boundary_logits"]
    boundary_target = batch["boundary_target"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Only compute loss on non-padding tokens
    valid_mask = attention_mask.bool()
    boundary_loss = focal_loss_fn(
        boundary_logits[valid_mask],
        boundary_target[valid_mask],
    )

    # Head 2: Co-transition regression (Huber loss)
    co_pred = outputs["co_transition_pred"]
    co_targets = batch["co_targets"].to(device)

    # Normalize targets for similar scale
    # n_co_transitions: divide by 20 (typical max ~40)
    # reading_ratio: already 0-1
    # mean_spacing: divide by 200 (typical range)
    # burstiness: divide by 10 (typical range)
    norm_factors = torch.tensor([20.0, 1.0, 200.0, 10.0], device=device)

    co_pred_norm = co_pred / norm_factors
    co_targets_norm = co_targets / norm_factors

    co_loss = F.huber_loss(co_pred_norm, co_targets_norm, delta=0.5)

    # Head 3: BP depth cross-entropy
    bp_logits = outputs["bp_depth_logits"]
    bp_target = batch["bp_depth"].to(device)
    bp_loss = F.cross_entropy(bp_logits, bp_target)

    return {
        "boundary_loss": boundary_loss,
        "co_loss": co_loss,
        "bp_loss": bp_loss,
        "total_loss": boundary_loss + co_loss + bp_loss,
    }


def compute_metrics(
    outputs: dict,
    batch: dict,
) -> dict:
    """Compute evaluation metrics."""

    device = outputs["boundary_logits"].device

    # Head 1: Boundary precision/recall/F1
    boundary_probs = torch.sigmoid(outputs["boundary_logits"])
    boundary_preds = (boundary_probs > 0.5).float()
    boundary_target = batch["boundary_target"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    valid_mask = attention_mask.bool()
    preds_valid = boundary_preds[valid_mask]
    target_valid = boundary_target[valid_mask]

    tp = ((preds_valid == 1) & (target_valid == 1)).sum().float()
    fp = ((preds_valid == 1) & (target_valid == 0)).sum().float()
    fn = ((preds_valid == 0) & (target_valid == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Head 2: Per-target MAE
    co_pred = outputs["co_transition_pred"]
    co_targets = batch["co_targets"].to(device)

    mae_n_co = (co_pred[:, 0] - co_targets[:, 0]).abs().mean()
    mae_reading = (co_pred[:, 1] - co_targets[:, 1]).abs().mean()
    mae_spacing = (co_pred[:, 2] - co_targets[:, 2]).abs().mean()
    mae_burstiness = (co_pred[:, 3] - co_targets[:, 3]).abs().mean()

    # Head 3: Accuracy
    bp_preds = outputs["bp_depth_logits"].argmax(dim=1)
    bp_target = batch["bp_depth"].to(device)
    bp_acc = (bp_preds == bp_target).float().mean()

    return {
        "boundary_precision": precision.item(),
        "boundary_recall": recall.item(),
        "boundary_f1": f1.item(),
        "mae_n_co_transitions": mae_n_co.item(),
        "mae_reading_ratio": mae_reading.item(),
        "mae_mean_spacing": mae_spacing.item(),
        "mae_burstiness": mae_burstiness.item(),
        "bp_accuracy": bp_acc.item(),
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    focal_loss_fn: FocalLoss,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_boundary_loss = 0
    total_co_loss = 0
    total_bp_loss = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        losses = compute_losses(outputs, batch, focal_loss_fn)

        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses["total_loss"].item()
        total_boundary_loss += losses["boundary_loss"].item()
        total_co_loss += losses["co_loss"].item()
        total_bp_loss += losses["bp_loss"].item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{losses['total_loss'].item():.4f}",
            "bound": f"{losses['boundary_loss'].item():.4f}",
        })

    return {
        "train_loss": total_loss / n_batches,
        "train_boundary_loss": total_boundary_loss / n_batches,
        "train_co_loss": total_co_loss / n_batches,
        "train_bp_loss": total_bp_loss / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    focal_loss_fn: FocalLoss,
    device: torch.device,
) -> dict:
    """Evaluate on validation set."""
    model.eval()

    total_loss = 0
    total_boundary_loss = 0
    total_co_loss = 0
    total_bp_loss = 0

    all_metrics = []
    n_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)
        losses = compute_losses(outputs, batch, focal_loss_fn)
        metrics = compute_metrics(outputs, batch)

        total_loss += losses["total_loss"].item()
        total_boundary_loss += losses["boundary_loss"].item()
        total_co_loss += losses["co_loss"].item()
        total_bp_loss += losses["bp_loss"].item()

        all_metrics.append(metrics)
        n_batches += 1

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    avg_metrics.update({
        "val_loss": total_loss / n_batches,
        "val_boundary_loss": total_boundary_loss / n_batches,
        "val_co_loss": total_co_loss / n_batches,
        "val_bp_loss": total_bp_loss / n_batches,
    })

    return avg_metrics


def print_epoch_summary(epoch: int, train_metrics: dict, val_metrics: dict):
    """Print formatted epoch summary."""
    print(f"\n{'='*70}")
    print(f"Epoch {epoch} Summary")
    print(f"{'='*70}")

    print(f"\n{'Losses':<20} {'Train':>12} {'Val':>12}")
    print(f"{'-'*44}")
    print(f"{'Total':<20} {train_metrics['train_loss']:>12.4f} {val_metrics['val_loss']:>12.4f}")
    print(f"{'Boundary':<20} {train_metrics['train_boundary_loss']:>12.4f} {val_metrics['val_boundary_loss']:>12.4f}")
    print(f"{'Co-transition':<20} {train_metrics['train_co_loss']:>12.4f} {val_metrics['val_co_loss']:>12.4f}")
    print(f"{'BP Depth':<20} {train_metrics['train_bp_loss']:>12.4f} {val_metrics['val_bp_loss']:>12.4f}")

    print(f"\n{'Head 1: Boundary Detection'}")
    print(f"{'-'*44}")
    print(f"{'Precision':<20} {val_metrics['boundary_precision']:>12.4f}")
    print(f"{'Recall':<20} {val_metrics['boundary_recall']:>12.4f}")
    print(f"{'F1':<20} {val_metrics['boundary_f1']:>12.4f}")

    print(f"\n{'Head 2: Co-transition MAE'}")
    print(f"{'-'*44}")
    print(f"{'n_co_transitions':<20} {val_metrics['mae_n_co_transitions']:>12.4f}")
    print(f"{'reading_ratio':<20} {val_metrics['mae_reading_ratio']:>12.4f}")
    print(f"{'mean_spacing':<20} {val_metrics['mae_mean_spacing']:>12.4f}")
    print(f"{'burstiness':<20} {val_metrics['mae_burstiness']:>12.4f}")

    print(f"\n{'Head 3: BP Depth'}")
    print(f"{'-'*44}")
    print(f"{'Accuracy':<20} {val_metrics['bp_accuracy']:>12.4f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr-lora", type=float, default=2e-4)
    parser.add_argument("--lr-heads", type=float, default=2e-4)  # Match LoRA LR to prevent head overfitting
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=10)  # More runway before early stopping
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer and backbone
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # A10G doesn't support bfloat16
    )

    # Get hidden size
    hidden_dim = backbone.config.hidden_size
    print(f"Backbone hidden size: {hidden_dim}")

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    # Create C1 model
    model = C1Model(backbone, hidden_dim=hidden_dim)
    model = model.to(device)

    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    train_dataset = C1Dataset(
        args.data_dir, tokenizer, args.max_length, split="train"
    )
    val_dataset = C1Dataset(
        args.data_dir, tokenizer, args.max_length, split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Setup optimizer with different LRs
    lora_params = [p for n, p in model.named_parameters() if "lora" in n.lower()]
    head_params = [p for n, p in model.named_parameters()
                   if any(h in n for h in ["boundary_head", "co_transition_head", "bp_depth_head"])]

    optimizer = AdamW([
        {"params": lora_params, "lr": args.lr_lora},
        {"params": head_params, "lr": args.lr_heads},
    ], weight_decay=0.01)

    # Learning rate schedule
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    # Loss function
    focal_loss_fn = FocalLoss(alpha=0.75, gamma=2.0)  # Higher alpha for sparse boundaries

    # Training loop
    training_log = []
    best_val_loss = float("inf")
    patience_counter = 0

    config = {
        "model_name": args.model_name,
        "lora_rank": args.lora_rank,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr_lora": args.lr_lora,
        "lr_heads": args.lr_heads,
        "max_length": args.max_length,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size} | Steps/epoch: {len(train_loader)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, focal_loss_fn, device, epoch
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, focal_loss_fn, device)

        print_epoch_summary(epoch, train_metrics, val_metrics)

        # Log
        epoch_log = {"epoch": epoch, **train_metrics, **val_metrics}
        training_log.append(epoch_log)

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            patience_counter = 0

            best_path = output_path / "best_checkpoint"
            best_path.mkdir(exist_ok=True)

            # Save LoRA adapters
            model.backbone.save_pretrained(best_path / "lora_adapters")

            # Save head weights
            torch.save({
                "boundary_head": model.boundary_head.state_dict(),
                "co_transition_head": model.co_transition_head.state_dict(),
                "bp_depth_head": model.bp_depth_head.state_dict(),
            }, best_path / "head_weights.pt")

            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Save log periodically
        with open(output_path / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

    # Save final checkpoint
    final_path = output_path / "final_checkpoint"
    final_path.mkdir(exist_ok=True)
    model.backbone.save_pretrained(final_path / "lora_adapters")
    torch.save({
        "boundary_head": model.boundary_head.state_dict(),
        "co_transition_head": model.co_transition_head.state_dict(),
        "bp_depth_head": model.bp_depth_head.state_dict(),
    }, final_path / "head_weights.pt")

    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {output_path}")


if __name__ == "__main__":
    main()
