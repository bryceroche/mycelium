"""
C1-B: Sequence Analyzer Training

Co-transition statistics regression + BP depth classification.
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


class C1BSequenceModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_dim: int = 896):
        super().__init__()
        self.backbone = backbone

        self.co_transition_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

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

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        co_raw = self.co_transition_head(pooled)
        co_pred = torch.stack([
            F.softplus(co_raw[:, 0]),
            torch.sigmoid(co_raw[:, 1]),
            F.softplus(co_raw[:, 2]),
            F.softplus(co_raw[:, 3]),
        ], dim=1)

        bp_logits = self.bp_depth_head(pooled)

        return {"co_pred": co_pred, "bp_logits": bp_logits}


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

        if split == "train":
            self.co_stats = self._compute_co_stats(train_indices)
        else:
            self.co_stats = None

    def set_co_stats(self, stats: dict):
        self.co_stats = stats

    def _compute_co_stats(self, indices: list) -> dict:
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

        co_targets_raw = torch.tensor(raw_co, dtype=torch.float32)
        bp_depth = record.get("bp_depth", 1) - 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "co_targets": co_targets,
            "co_targets_raw": co_targets_raw,
            "bp_depth": bp_depth,
        }


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

    print(f"\nLoading data from {args.data_dir}...")
    train_dataset = C1Dataset(args.data_dir, tokenizer, args.max_length, split="train")
    val_dataset = C1Dataset(args.data_dir, tokenizer, args.max_length, split="val")
    val_dataset.set_co_stats(train_dataset.co_stats)

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
        "co_stats": train_dataset.co_stats,
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

    model = C1BSequenceModel(backbone, hidden_dim=backbone.config.hidden_size)
    model = model.to(device)

    co_stats = train_dataset.co_stats

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

    print(f"\nStarting C1-B training for {args.epochs} epochs...")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

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

            co_pred = outputs["co_pred"]
            co_pred_norm = torch.zeros_like(co_pred)
            keys = ["n_co_transitions", "reading_ratio", "mean_spacing", "burstiness"]
            for i, k in enumerate(keys):
                co_pred_norm[:, i] = (co_pred[:, i] - co_stats[k]["mean"]) / co_stats[k]["std"]

            co_loss = F.huber_loss(co_pred_norm, co_targets, delta=1.0)
            bp_loss = F.cross_entropy(outputs["bp_logits"], bp_target)

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

        model.eval()
        val_loss = 0
        val_co_loss = 0
        val_bp_loss = 0

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

                co_pred_norm = torch.zeros_like(co_pred)
                keys = ["n_co_transitions", "reading_ratio", "mean_spacing", "burstiness"]
                for i, k in enumerate(keys):
                    co_pred_norm[:, i] = (co_pred[:, i] - co_stats[k]["mean"]) / co_stats[k]["std"]

                co_loss = F.huber_loss(co_pred_norm, co_targets, delta=1.0)
                bp_loss = F.cross_entropy(outputs["bp_logits"], bp_target)

                val_loss += (co_loss + bp_loss).item()
                val_co_loss += co_loss.item()
                val_bp_loss += bp_loss.item()

                for i, k in enumerate(keys):
                    mae_sums[k] += (co_pred[:, i] - co_targets_raw[:, i]).abs().sum().item()

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

        print(f"\n{'='*60}")
        print(f"C1-B Epoch {epoch} Summary")
        print(f"{'='*60}")
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_path = output_path / "best_checkpoint"
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

        with open(output_path / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

    best_bp_acc = max(log["bp_accuracy"] for log in training_log)
    best_mae = min(log["mae_n_co_transitions"] for log in training_log)

    print(f"\nC1-B Training Complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best BP accuracy: {best_bp_acc:.4f}")
    print(f"Best MAE n_co_transitions: {best_mae:.4f}")
    print(f"Checkpoint saved to {output_path}")

    results = {
        "best_val_loss": best_val_loss,
        "best_bp_accuracy": best_bp_acc,
        "best_mae_n_co_transitions": best_mae,
        "final_epoch": epoch,
    }
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
