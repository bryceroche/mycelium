"""
C1-B v5 Merged: Sequence Analyzer with Auxiliary Boundary Task

Uses merged JSONL for boundaries, v4 data for sequence-level targets.
Includes negative examples for better calibration.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

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
    """C1-B with auxiliary boundary task."""

    def __init__(self, backbone: nn.Module, hidden_dim: int = 896):
        super().__init__()
        self.backbone = backbone

        # Auxiliary: per-token boundary head
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))

        # Head 2: Co-transition statistics
        self.co_transition_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 4))

        # Head 3: BP depth classification
        self.bp_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 3))

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        boundary_logits = self.boundary_head(hidden_states).squeeze(-1)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        co_raw = self.co_transition_head(pooled)
        co_pred = torch.stack([
            F.softplus(co_raw[:, 0]), torch.sigmoid(co_raw[:, 1]),
            F.softplus(co_raw[:, 2]), F.softplus(co_raw[:, 3])
        ], dim=1)

        bp_logits = self.bp_depth_head(pooled)

        return {"boundary_logits": boundary_logits, "co_pred": co_pred, "bp_logits": bp_logits}


class C1BDatasetMerged(Dataset):
    """Dataset using merged JSONL for boundaries, v4 for sequence targets."""

    def __init__(self, data_file: str, v4_dir: str, tokenizer, max_length: int = 512,
                 split: str = "train", val_ratio: float = 0.1, seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load merged JSONL
        print(f"Loading merged data from {data_file}...")
        merged_records = {}
        with open(data_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    merged_records[str(r["problem_idx"])] = r

        print(f"  Loaded {len(merged_records)} problems from merged data")

        # Load v4 records for sequence-level targets
        print(f"Loading v4 records from {v4_dir}...")
        v4_records = {}
        v4_path = Path(v4_dir)

        for json_file in sorted(v4_path.glob("*.json")):
            if "stats" in str(json_file) or "global" in str(json_file):
                continue
            with open(json_file) as f:
                chunk = json.load(f)
                for r in chunk.get("records", []):
                    problem_id = str(r.get("problem_id", ""))
                    v4_records[problem_id] = r

        print(f"  Loaded {len(v4_records)} v4 records")

        # Join: merged (boundaries) + v4 (sequence targets)
        all_records = []
        for problem_idx, merged in merged_records.items():
            if problem_idx in v4_records:
                v4 = v4_records[problem_idx]
                all_records.append({
                    "problem_idx": problem_idx,
                    "problem_text": merged["problem_text"],
                    "n_input_tokens": merged.get("n_input_tokens", merged.get("input_len", 100)),
                    "has_boundaries": merged.get("has_boundaries", False),
                    "boundaries": merged.get("boundaries", []),
                    # Sequence targets from v4
                    "n_co_transitions": v4.get("n_co_transitions", 0),
                    "reading_ratio": v4.get("reading_ratio", 0.5),
                    "mean_spacing": v4.get("mean_spacing", 100),
                    "burstiness": v4.get("burstiness", 0),
                    "bp_depth": v4.get("bp_depth", 1),
                })

        n_positive = sum(1 for r in all_records if r["has_boundaries"])
        n_negative = len(all_records) - n_positive
        print(f"  Joined {len(all_records)} records ({n_positive} positive, {n_negative} negative)")

        # Compute z-score stats
        if split == "train":
            self.co_stats = self._compute_co_stats(all_records)
        else:
            self.co_stats = None

        # Stratified split by BP depth
        np.random.seed(seed)
        by_depth = {1: [], 2: [], 3: []}
        for i, r in enumerate(all_records):
            by_depth[r["bp_depth"]].append(i)

        train_indices, val_indices = [], []
        for indices in by_depth.values():
            np.random.shuffle(indices)
            n_val = int(len(indices) * val_ratio)
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        self.indices = train_indices if split == "train" else val_indices
        np.random.shuffle(self.indices)
        self.all_records = all_records
        print(f"  {split}: {len(self.indices)} examples")

    def _compute_co_stats(self, records):
        vals = {"n_co_transitions": [], "reading_ratio": [], "mean_spacing": [], "burstiness": []}
        for r in records:
            vals["n_co_transitions"].append(r["n_co_transitions"])
            vals["reading_ratio"].append(r["reading_ratio"])
            vals["mean_spacing"].append(r["mean_spacing"])
            vals["burstiness"].append(r["burstiness"])
        return {k: {"mean": np.mean(v), "std": np.std(v) + 1e-8} for k, v in vals.items()}

    def set_co_stats(self, stats):
        self.co_stats = stats

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        record = self.all_records[self.indices[idx]]

        encoding = self.tokenizer(record["problem_text"], max_length=self.max_length,
                                  padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        actual_len = attention_mask.sum().item()
        n_input = record["n_input_tokens"]

        # Soft boundary targets from merged data
        boundary_target = torch.zeros(self.max_length)
        if record["has_boundaries"] and n_input > 0 and actual_len > 0:
            scale = actual_len / n_input
            for boundary in record.get("boundaries", []):
                for sb in boundary.get("soft_boundaries", []):
                    pos = sb["pos"]
                    prob = sb["prob"]
                    mapped_idx = min(int(pos * scale), actual_len - 1)
                    boundary_target[mapped_idx] = max(boundary_target[mapped_idx].item(), prob)

        # Co-transition targets (z-scored)
        raw_co = [record["n_co_transitions"], record["reading_ratio"],
                  record["mean_spacing"], record["burstiness"]]
        if self.co_stats:
            keys = ["n_co_transitions", "reading_ratio", "mean_spacing", "burstiness"]
            co_targets = torch.tensor([(raw_co[i] - self.co_stats[k]["mean"]) / self.co_stats[k]["std"]
                                       for i, k in enumerate(keys)], dtype=torch.float32)
        else:
            co_targets = torch.tensor(raw_co, dtype=torch.float32)

        co_targets_raw = torch.tensor(raw_co, dtype=torch.float32)
        bp_depth = record["bp_depth"] - 1

        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "boundary_target": boundary_target, "co_targets": co_targets,
            "co_targets_raw": co_targets_raw, "bp_depth": bp_depth,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True, help="Merged JSONL file")
    parser.add_argument("--v4-dir", type=str, required=True, help="v4 data for sequence targets")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--aux-weight", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = C1BDatasetMerged(args.data_file, args.v4_dir, tokenizer, split="train")
    val_dataset = C1BDatasetMerged(args.data_file, args.v4_dir, tokenizer, split="val")
    val_dataset.set_co_stats(train_dataset.co_stats)
    co_stats = train_dataset.co_stats

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    config = {
        "model_name": args.model_name, "lora_rank": args.lora_rank,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "train_size": len(train_dataset), "val_size": len(val_dataset),
        "aux_weight": args.aux_weight, "co_stats": co_stats, "data_version": "v5_merged",
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=float)

    backbone = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.float32)
    lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=args.lora_rank,
                             lora_alpha=args.lora_rank * 2, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    model = C1BSequenceModel(backbone, hidden_dim=backbone.config.hidden_size).to(device)

    lora_params = [p for n, p in model.named_parameters() if "lora" in n.lower()]
    head_params = [p for n, p in model.named_parameters()
                   if any(h in n for h in ["boundary_head", "co_transition_head", "bp_depth_head"])]

    optimizer = AdamW([{"params": lora_params, "lr": 2e-4},
                       {"params": head_params, "lr": 5e-4}], weight_decay=0.01)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)
    ], milestones=[warmup_steps])

    training_log = []
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nStarting C1-B v5 merged training: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"Auxiliary boundary weight: {args.aux_weight}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_aux, total_co, total_bp = 0, 0, 0, 0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"C1-B Epoch {epoch}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            boundary_target = batch["boundary_target"].to(device)
            co_targets = batch["co_targets"].to(device)
            bp_target = batch["bp_depth"].to(device)

            outputs = model(input_ids, attention_mask)

            valid_mask = attention_mask.bool()
            aux_loss = F.binary_cross_entropy_with_logits(
                outputs["boundary_logits"][valid_mask], boundary_target[valid_mask])

            co_pred = outputs["co_pred"]
            keys = ["n_co_transitions", "reading_ratio", "mean_spacing", "burstiness"]
            co_pred_norm = torch.stack([(co_pred[:, i] - co_stats[k]["mean"]) / co_stats[k]["std"]
                                        for i, k in enumerate(keys)], dim=1)
            co_loss = F.huber_loss(co_pred_norm, co_targets, delta=1.0)

            bp_loss = F.cross_entropy(outputs["bp_logits"], bp_target)

            loss = (args.aux_weight * aux_loss + co_loss + bp_loss) / args.grad_accum
            loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += (args.aux_weight * aux_loss + co_loss + bp_loss).item()
            total_aux += aux_loss.item()
            total_co += co_loss.item()
            total_bp += bp_loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_loss, val_aux, val_co, val_bp = 0, 0, 0, 0
        mae_sums = {"n_co_transitions": 0, "reading_ratio": 0, "mean_spacing": 0, "burstiness": 0}
        bp_correct, bp_total = 0, 0
        aux_f1_sum, aux_count = 0, 0
        n_val = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                boundary_target = batch["boundary_target"].to(device)
                co_targets = batch["co_targets"].to(device)
                co_targets_raw = batch["co_targets_raw"].to(device)
                bp_target = batch["bp_depth"].to(device)

                outputs = model(input_ids, attention_mask)

                valid_mask = attention_mask.bool()
                aux_loss = F.binary_cross_entropy_with_logits(
                    outputs["boundary_logits"][valid_mask], boundary_target[valid_mask])

                co_pred = outputs["co_pred"]
                co_pred_norm = torch.stack([(co_pred[:, i] - co_stats[k]["mean"]) / co_stats[k]["std"]
                                            for i, k in enumerate(keys)], dim=1)
                co_loss = F.huber_loss(co_pred_norm, co_targets, delta=1.0)
                bp_loss = F.cross_entropy(outputs["bp_logits"], bp_target)

                val_loss += (co_loss + bp_loss).item()
                val_aux += aux_loss.item()
                val_co += co_loss.item()
                val_bp += bp_loss.item()

                for i, k in enumerate(keys):
                    mae_sums[k] += (co_pred[:, i] - co_targets_raw[:, i]).abs().sum().item()

                bp_preds = outputs["bp_logits"].argmax(dim=1)
                bp_correct += (bp_preds == bp_target).sum().item()
                bp_total += len(bp_target)

                preds = (torch.sigmoid(outputs["boundary_logits"]) > 0.5).float()
                tgt = (boundary_target > 0.5).float()
                tp = ((preds[valid_mask] == 1) & (tgt[valid_mask] == 1)).sum().float()
                fp = ((preds[valid_mask] == 1) & (tgt[valid_mask] == 0)).sum().float()
                fn = ((preds[valid_mask] == 0) & (tgt[valid_mask] == 1)).sum().float()
                prec = tp / (tp + fp + 1e-8)
                rec = tp / (tp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                aux_f1_sum += f1.item()
                aux_count += 1

                n_val += len(bp_target)

        val_loss /= len(val_loader)
        mae = {k: v / n_val for k, v in mae_sums.items()}
        bp_acc = bp_correct / bp_total
        aux_f1 = aux_f1_sum / aux_count

        print(f"\n{'='*60}")
        print(f"C1-B v5 Merged Epoch {epoch}")
        print(f"{'='*60}")
        print(f"Train Loss: {total_loss/n_batches:.4f} | Val Loss (no aux): {val_loss:.4f}")
        print(f"  Aux: {val_aux/len(val_loader):.4f} | Co: {val_co/len(val_loader):.4f} | BP: {val_bp/len(val_loader):.4f}")
        print(f"Co-trans MAE: n={mae['n_co_transitions']:.2f}, rr={mae['reading_ratio']:.3f}")
        print(f"BP Accuracy: {bp_acc:.4f}")
        print(f"Auxiliary Boundary F1: {aux_f1:.4f}")

        epoch_log = {
            "epoch": epoch, "train_loss": total_loss/n_batches, "val_loss": val_loss,
            "val_aux_loss": val_aux/len(val_loader), "val_co_loss": val_co/len(val_loader),
            "val_bp_loss": val_bp/len(val_loader), "bp_accuracy": bp_acc, "aux_boundary_f1": aux_f1,
            **{f"mae_{k}": v for k, v in mae.items()}
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
                "boundary_head": model.boundary_head.state_dict(),
            }, best_path / "head_weights.pt")
            print(f"  Saved best (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        with open(output_path / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

    best_bp = max(log["bp_accuracy"] for log in training_log)
    best_mae = min(log["mae_n_co_transitions"] for log in training_log)
    print(f"\nC1-B v5 Merged Complete! Best BP: {best_bp:.4f}, Best MAE: {best_mae:.4f}")

    with open(output_path / "results.json", "w") as f:
        json.dump({"best_val_loss": best_val_loss, "best_bp_accuracy": best_bp,
                   "best_mae_n_co_transitions": best_mae, "final_epoch": epoch}, f, indent=2)


if __name__ == "__main__":
    main()
