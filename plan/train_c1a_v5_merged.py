"""
C1-A v5 Merged: Boundary Detector with Soft Probability Targets

Trains on merged JSONL with soft boundary probabilities.
Includes negative examples (no boundaries) for better calibration.
"""

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
    """Focal loss for soft probability targets."""
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

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        return self.boundary_head(hidden_states).squeeze(-1)


class C1DatasetMerged(Dataset):
    """Dataset for merged JSONL with soft boundaries."""

    def __init__(self, data_file: str, tokenizer, max_length: int = 512,
                 split: str = "train", val_ratio: float = 0.1, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load all records from JSONL
        print(f"Loading data from {data_file}...")
        all_records = []
        with open(data_file) as f:
            for line in f:
                if line.strip():
                    all_records.append(json.loads(line))

        n_positive = sum(1 for r in all_records if r.get("has_boundaries", False))
        n_negative = len(all_records) - n_positive
        print(f"  Loaded {len(all_records)} records ({n_positive} positive, {n_negative} negative)")

        # Stratified split - separate positive and negative
        np.random.seed(seed)
        positive_idx = [i for i, r in enumerate(all_records) if r.get("has_boundaries", False)]
        negative_idx = [i for i, r in enumerate(all_records) if not r.get("has_boundaries", False)]

        np.random.shuffle(positive_idx)
        np.random.shuffle(negative_idx)

        n_val_pos = int(len(positive_idx) * val_ratio)
        n_val_neg = int(len(negative_idx) * val_ratio)

        if split == "train":
            self.indices = positive_idx[n_val_pos:] + negative_idx[n_val_neg:]
        else:
            self.indices = positive_idx[:n_val_pos] + negative_idx[:n_val_neg]

        np.random.shuffle(self.indices)
        self.all_records = all_records
        print(f"  {split} set: {len(self.indices)} examples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record = self.all_records[self.indices[idx]]

        text = record["problem_text"]
        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length",
                                  truncation=True, return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        actual_len = attention_mask.sum().item()

        n_input = record.get("n_input_tokens", record.get("input_len", 100))

        # Create soft target tensor
        boundary_target = torch.zeros(self.max_length)

        # Extract soft boundaries from the new format
        if record.get("has_boundaries", False) and n_input > 0 and actual_len > 0:
            scale = actual_len / n_input
            for boundary in record.get("boundaries", []):
                for sb in boundary.get("soft_boundaries", []):
                    pos = sb["pos"]
                    prob = sb["prob"]
                    mapped_idx = min(int(pos * scale), actual_len - 1)
                    boundary_target[mapped_idx] = max(boundary_target[mapped_idx].item(), prob)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "boundary_target": boundary_target,
        }


def compute_metrics(logits, targets, mask, thresholds=[0.3, 0.5, 0.7]):
    probs = torch.sigmoid(logits)
    valid_mask = mask.bool()

    metrics = {}
    for thresh in thresholds:
        preds = (probs > thresh).float()
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

    metrics["n_actual"] = (targets[valid_mask] > 0.5).sum().item()

    # Calibration: mean pred prob at boundaries vs non-boundaries
    boundary_mask = targets[valid_mask] > 0.1
    non_boundary_mask = targets[valid_mask] <= 0.1

    if boundary_mask.sum() > 0:
        metrics["mean_pred_at_boundary"] = probs[valid_mask][boundary_mask].mean().item()
        metrics["mean_target_at_boundary"] = targets[valid_mask][boundary_mask].mean().item()
    if non_boundary_mask.sum() > 0:
        metrics["mean_pred_at_non_boundary"] = probs[valid_mask][non_boundary_mask].mean().item()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True, help="Merged JSONL file")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = C1DatasetMerged(args.data_file, tokenizer, args.max_length, "train")
    val_dataset = C1DatasetMerged(args.data_file, tokenizer, args.max_length, "val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    config = {
        "model_name": args.model_name, "lora_rank": args.lora_rank,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "train_size": len(train_dataset), "val_size": len(val_dataset),
        "data_version": "v5_merged",
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    backbone = AutoModel.from_pretrained(args.model_name, trust_remote_code=True,
                                         torch_dtype=torch.float32)
    lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_rank,
                             lora_alpha=args.lora_rank * 2, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    model = C1ABoundaryModel(backbone, hidden_dim=backbone.config.hidden_size).to(device)
    loss_fn = SoftFocalLoss(alpha=0.75, gamma=2.0)

    lora_params = [p for n, p in model.named_parameters() if "lora" in n.lower()]
    head_params = [p for n, p in model.named_parameters() if "boundary_head" in n]

    optimizer = AdamW([{"params": lora_params, "lr": 2e-4},
                       {"params": head_params, "lr": 5e-4}], weight_decay=0.01)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)
    ], milestones=[warmup_steps])

    training_log = []
    best_f1 = 0.0
    patience_counter = 0

    print(f"\nStarting C1-A v5 merged training: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"C1-A Epoch {epoch}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            boundary_target = batch["boundary_target"].to(device)

            logits = model(input_ids, attention_mask)
            valid_mask = attention_mask.bool()
            loss = loss_fn(logits[valid_mask], boundary_target[valid_mask]) / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            n_batches += 1

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
                val_loss += loss_fn(logits[valid_mask], boundary_target[valid_mask]).item()
                all_metrics.append(compute_metrics(logits, boundary_target, attention_mask))

        val_loss /= len(val_loader)
        avg_metrics = {k: np.mean([m[k] for m in all_metrics if k in m]) for k in all_metrics[0]}

        print(f"\n{'='*60}")
        print(f"C1-A v5 Merged Epoch {epoch}")
        print(f"{'='*60}")
        print(f"Train Loss: {total_loss/n_batches:.4f} | Val Loss: {val_loss:.4f}")
        print(f"F1@0.5: {avg_metrics['f1']:.4f} | F1@0.3: {avg_metrics['f1_0.3']:.4f} | F1@0.7: {avg_metrics['f1_0.7']:.4f}")
        print(f"Pred: {avg_metrics['n_pred']:.0f} | Actual: {avg_metrics['n_actual']:.0f}")
        if "mean_pred_at_boundary" in avg_metrics:
            print(f"Calibration: pred@boundary={avg_metrics['mean_pred_at_boundary']:.3f} vs target={avg_metrics['mean_target_at_boundary']:.3f}")

        epoch_log = {"epoch": epoch, "train_loss": total_loss/n_batches, "val_loss": val_loss, **avg_metrics}
        training_log.append(epoch_log)

        # Use F1@0.3 for early stopping - soft targets make 0.5 threshold too strict early on
        f1_for_stopping = avg_metrics["f1_0.3"]
        if f1_for_stopping > best_f1:
            best_f1 = f1_for_stopping
            patience_counter = 0
            best_path = output_path / "best_checkpoint"
            best_path.mkdir(parents=True, exist_ok=True)
            model.backbone.save_pretrained(best_path / "lora_adapters")
            torch.save({"boundary_head": model.boundary_head.state_dict()}, best_path / "head_weights.pt")
            print(f"  Saved best (F1@0.3: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        with open(output_path / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

    print(f"\nC1-A v5 Merged Complete! Best F1@0.3: {best_f1:.4f}")
    with open(output_path / "results.json", "w") as f:
        json.dump({"best_f1_0.3": best_f1, "final_epoch": epoch}, f, indent=2)


if __name__ == "__main__":
    main()
