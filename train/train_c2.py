#!/usr/bin/env python3
"""
Mycelium v6: C2 Multi-Label Classifier with Heartbeat Auxiliary

Task: Given full problem text, predict the SET of operations needed + operation count.
Architecture:
  - Backbone: MiniLM-22M (frozen first 2 epochs, then unfreeze)
  - Head 1: Multi-label sigmoid over operation types (BCE loss)
  - Head 2: Regression predicting heartbeat count (MSE loss)
  - Joint loss: L = BCE(labels) + 0.1 * MSE(heartbeat_count)

Input: raw problem text (NO CoT, NO template tags)
Output: set of labels + heartbeat count
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import argparse
import sys
import boto3

# Use MiniLM for efficiency (22M params)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class C2Dataset(Dataset):
    """Dataset for C2 multi-label classification with heartbeat auxiliary."""

    def __init__(self, examples, label2id, tokenizer, max_length=256):
        self.examples = examples
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Tokenize
        encoding = self.tokenizer(
            ex["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Multi-hot label encoding
        label_vec = torch.zeros(self.num_labels)
        for lbl in ex["labels"]:
            if lbl in self.label2id:
                label_vec[self.label2id[lbl]] = 1.0

        # Heartbeat count
        heartbeat_count = float(ex.get("heartbeat_count", 0))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label_vec,
            "heartbeat_count": torch.tensor(heartbeat_count, dtype=torch.float32),
        }


class C2Model(nn.Module):
    """C2 classifier with dual heads: multi-label + heartbeat regression."""

    def __init__(self, backbone_name, num_labels, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size

        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )

        # Heartbeat count regression head
        self.heartbeat_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )

        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Dual heads
        logits = self.classifier(cls_output)
        heartbeat_pred = self.heartbeat_head(cls_output).squeeze(-1)

        return logits, heartbeat_pred

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone FROZEN")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  Backbone UNFROZEN")


def compute_metrics(labels, preds, heartbeat_true, heartbeat_pred):
    """Compute multi-label and regression metrics."""
    # Multi-label metrics
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    precision = precision_score(labels, preds, average='micro', zero_division=0)
    recall = recall_score(labels, preds, average='micro', zero_division=0)

    # Exact match
    exact_match = np.all(preds == labels, axis=1).mean()

    # Any correct (at least one predicted label is correct)
    any_correct = (((preds == 1) & (labels == 1)).sum(axis=1) > 0).mean()

    # Heartbeat MSE
    heartbeat_mse = np.mean((heartbeat_true - heartbeat_pred) ** 2)

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "exact_match": exact_match,
        "any_correct": any_correct,
        "heartbeat_mse": heartbeat_mse,
    }


def load_data_from_s3(s3_path):
    """Load training data from S3."""
    s3 = boto3.client('s3')
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])

    print(f"  Bucket: {bucket}, Key: {key}")
    response = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(response['Body'].read().decode('utf-8'))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="s3://mycelium-data/c2_training/c2_train_with_heartbeats.json")
    parser.add_argument("--output-dir", default="models/c2_heartbeat")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--heartbeat-weight", type=float, default=0.1)
    args = parser.parse_args()

    print("=" * 70)
    print("MYCELIUM V6: C2 MULTI-LABEL CLASSIFIER WITH HEARTBEAT AUXILIARY")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Freeze epochs: {args.freeze_epochs}")
    print(f"Heartbeat weight: {args.heartbeat_weight}")
    sys.stdout.flush()

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    sys.stdout.flush()

    if args.data_path.startswith("s3://"):
        data = load_data_from_s3(args.data_path)
    else:
        with open(args.data_path) as f:
            data = json.load(f)

    examples = data["examples"]
    labels_list = data["labels"]
    label2id = {l: i for i, l in enumerate(labels_list)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels_list)

    print(f"Loaded {len(examples)} examples")
    print(f"Labels ({num_labels}): {labels_list}")
    sys.stdout.flush()

    # Show label distribution
    print("\nLabel distribution:")
    for lbl, cnt in sorted(data.get('label_distribution', {}).items(), key=lambda x: -x[1]):
        pct = 100 * cnt / len(examples)
        print(f"  {lbl:12}: {cnt:4} ({pct:5.1f}%)")
    sys.stdout.flush()

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Train/val split (stratified by primary label)
    primary_labels = [ex["labels"][0] if ex["labels"] else "OTHER" for ex in examples]
    train_examples, val_examples = train_test_split(
        examples, test_size=0.1, random_state=42, stratify=primary_labels
    )

    print(f"\nTrain: {len(train_examples)}, Val: {len(val_examples)}")
    sys.stdout.flush()

    # Create datasets
    train_dataset = C2Dataset(train_examples, label2id, tokenizer, args.max_length)
    val_dataset = C2Dataset(val_examples, label2id, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    print(f"\nLoading model: {MODEL_NAME}")
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = C2Model(MODEL_NAME, num_labels)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    sys.stdout.flush()

    # Loss functions
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    # Start with frozen backbone
    model.freeze_backbone()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate * 10)  # Higher LR for heads only

    # Training loop
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    best_f1 = 0
    patience = 3
    patience_counter = 0

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    sys.stdout.flush()

    for epoch in range(args.epochs):
        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs:
            print(f"\n>>> Epoch {epoch}: UNFREEZING BACKBONE <<<")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            sys.stdout.flush()

        # Training
        model.train()
        train_loss = 0
        train_bce = 0
        train_mse = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            heartbeat_count = batch["heartbeat_count"].to(device)

            optimizer.zero_grad()

            logits, heartbeat_pred = model(input_ids, attention_mask)

            # Joint loss
            bce = bce_loss_fn(logits, labels)
            mse = mse_loss_fn(heartbeat_pred, heartbeat_count)
            loss = bce + args.heartbeat_weight * mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_bce += bce.item()
            train_mse += mse.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f} (BCE: {bce.item():.4f}, MSE: {mse.item():.4f})")
                sys.stdout.flush()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_bce = train_bce / len(train_loader)
        avg_train_mse = train_mse / len(train_loader)

        # Validation
        model.eval()
        all_labels = []
        all_preds = []
        all_heartbeat_true = []
        all_heartbeat_pred = []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                heartbeat_count = batch["heartbeat_count"].to(device)

                logits, heartbeat_pred = model(input_ids, attention_mask)

                bce = bce_loss_fn(logits, labels)
                mse = mse_loss_fn(heartbeat_pred, heartbeat_count)
                loss = bce + args.heartbeat_weight * mse
                val_loss += loss.item()

                # Sigmoid + threshold for predictions
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()

                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_heartbeat_true.append(heartbeat_count.cpu().numpy())
                all_heartbeat_pred.append(heartbeat_pred.cpu().numpy())

        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)
        all_heartbeat_true = np.concatenate(all_heartbeat_true)
        all_heartbeat_pred = np.concatenate(all_heartbeat_pred)

        avg_val_loss = val_loss / len(val_loader)
        metrics = compute_metrics(all_labels, all_preds, all_heartbeat_true, all_heartbeat_pred)

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce:.4f}, MSE: {avg_train_mse:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  F1 Micro:   {metrics['f1_micro']:.4f}")
        print(f"  F1 Macro:   {metrics['f1_macro']:.4f}")
        print(f"  Any-Correct: {metrics['any_correct']:.4f}")
        print(f"  Exact Match: {metrics['exact_match']:.4f}")
        print(f"  Heartbeat MSE: {metrics['heartbeat_mse']:.4f}")
        sys.stdout.flush()

        # Save best model
        if metrics['f1_micro'] > best_f1:
            best_f1 = metrics['f1_micro']
            patience_counter = 0

            save_path = f"{args.output_dir}/best"
            Path(save_path).mkdir(parents=True, exist_ok=True)

            torch.save({
                "model_state_dict": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "num_labels": num_labels,
                "metrics": metrics,
                "epoch": epoch,
            }, f"{save_path}/model.pt")

            tokenizer.save_pretrained(save_path)

            # Save config
            with open(f"{save_path}/config.json", "w") as f:
                json.dump({
                    "model_name": MODEL_NAME,
                    "num_labels": num_labels,
                    "labels": labels_list,
                    "label2id": label2id,
                    "id2label": {str(k): v for k, v in id2label.items()},
                    "heartbeat_weight": args.heartbeat_weight,
                    "best_f1_micro": best_f1,
                    "best_any_correct": metrics['any_correct'],
                }, f, indent=2)

            print(f"  >>> New best model saved (F1: {best_f1:.4f})")
            sys.stdout.flush()
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= args.freeze_epochs + 2:
                print(f"\nEarly stopping triggered (patience={patience})")
                break

    # Final evaluation
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best F1 Micro: {best_f1:.4f}")
    print(f"Model saved to: {args.output_dir}/best")
    sys.stdout.flush()

    # Upload to S3
    print("\nUploading model to S3...")
    s3 = boto3.client('s3')
    save_path = f"{args.output_dir}/best"

    for fname in Path(save_path).iterdir():
        if fname.is_file():
            s3_key = f"models/c2_heartbeat/{fname.name}"
            s3.upload_file(str(fname), "mycelium-data", s3_key)
            print(f"  Uploaded {fname.name} -> s3://mycelium-data/{s3_key}")

    print("\nDone!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
