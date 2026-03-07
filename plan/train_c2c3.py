"""
Train C2 (Operation Classifier) and C3 (Operand Extractor)

Both models are tiny MLPs that consume C1-A's cached hidden states.

Usage:
    python train_c2c3.py --train-c2   # Train C2 classifier
    python train_c2c3.py --train-c3   # Train C3 extractor
    python train_c2c3.py --evaluate   # Evaluate both
"""

import json
import io
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import boto3
from botocore.config import Config

config = Config(read_timeout=120, connect_timeout=30, retries={'max_attempts': 3})
s3 = boto3.client("s3", config=config)
BUCKET = "mycelium-data"

# Model parameters
HIDDEN_DIM = 896  # C1-A hidden size (Qwen2-0.5B)
W = 16  # Window size
N_C2_CLASSES = 26  # 25 IB clusters + NO_OP
MAX_OPERANDS = 4


class C2(nn.Module):
    """Per-segment operation classifier."""

    def __init__(self, input_dim=HIDDEN_DIM, n_classes=N_C2_CLASSES):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, features):
        # features: (batch, W, hidden_dim) - mean pool
        pooled = features.mean(dim=1)  # (batch, hidden_dim)
        return self.classifier(pooled)  # (batch, n_classes)


class C3(nn.Module):
    """Per-segment operand extractor."""

    def __init__(self, hidden_dim=HIDDEN_DIM, c2_embed_dim=32,
                 n_c2_classes=N_C2_CLASSES, max_operands=MAX_OPERANDS, window_size=W):
        super().__init__()
        self.c2_embedding = nn.Embedding(n_c2_classes, c2_embed_dim)
        self.max_operands = max_operands
        self.window_size = window_size

        # Per-operand slot scoring (W+1 positions: W real + 1 BACKREF)
        self.operand_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + c2_embed_dim, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, window_size + 1)  # W real positions + BACKREF
            )
            for _ in range(max_operands)
        ])

        # Predict operand count (0-4)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim + c2_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, max_operands + 1)
        )

    def forward(self, window_features, c2_label):
        """
        window_features: (batch, W, hidden_dim)
        c2_label: (batch,) int
        """
        c2_embed = self.c2_embedding(c2_label)  # (batch, c2_embed_dim)

        # Pool for count prediction
        pooled = window_features.mean(dim=1)  # (batch, hidden_dim)
        pooled_with_c2 = torch.cat([pooled, c2_embed], dim=-1)

        # Count prediction
        count_logits = self.count_head(pooled_with_c2)  # (batch, 5)

        # Position scores for each operand slot
        operand_logits = []
        for scorer in self.operand_scorers:
            # For each operand slot, score all positions
            logits = scorer(pooled_with_c2)  # (batch, W+1)
            operand_logits.append(logits)

        return count_logits, operand_logits


class C2Dataset(Dataset):
    """Dataset for C2 training."""

    def __init__(self, features_dict, window_labels):
        self.samples = []

        for pid, labels in window_labels.items():
            if pid not in features_dict:
                continue

            features = features_dict[pid]  # (n_windows, W, hidden_dim)
            n_windows = min(len(labels), features.shape[0])

            for w_idx in range(n_windows):
                self.samples.append({
                    "features": features[w_idx],  # (W, hidden_dim)
                    "label": labels[w_idx]["c2_cluster_id"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "features": torch.tensor(s["features"], dtype=torch.float32),
            "label": torch.tensor(s["label"], dtype=torch.long),
        }


class C3Dataset(Dataset):
    """Dataset for C3 training (non-NO_OP windows only)."""

    def __init__(self, features_dict, window_labels):
        self.samples = []

        for pid, labels in window_labels.items():
            if pid not in features_dict:
                continue

            features = features_dict[pid]
            n_windows = min(len(labels), features.shape[0])

            for w_idx in range(n_windows):
                label = labels[w_idx]
                if label["c2_cluster_id"] == 0:  # NO_OP
                    continue

                n_operands = min(label.get("n_operands", 0), MAX_OPERANDS)

                # For now, use n_operands as ground truth
                # Position targets would require per-operand position labels
                self.samples.append({
                    "features": features[w_idx],
                    "c2_label": label["c2_cluster_id"],
                    "n_operands": n_operands,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "features": torch.tensor(s["features"], dtype=torch.float32),
            "c2_label": torch.tensor(s["c2_label"], dtype=torch.long),
            "n_operands": torch.tensor(s["n_operands"], dtype=torch.long),
        }


def load_data():
    """Load window labels and cached features."""
    print("Loading data...")

    # Load window labels
    resp = s3.get_object(Bucket=BUCKET, Key="c2c3_training_ready/window_labels.jsonl")
    window_labels = {}
    for line in resp["Body"].iter_lines():
        data = json.loads(line.decode("utf-8"))
        window_labels[data["problem_idx"]] = data["window_labels"]

    print(f"  Window labels: {len(window_labels)} problems")

    # Load feature manifest
    resp = s3.get_object(Bucket=BUCKET, Key="c2c3_training_ready/cached_features/manifest.json")
    manifest = json.loads(resp["Body"].read().decode("utf-8"))
    print(f"  Features manifest: {manifest['n_problems']} problems")

    # Load features (this will be memory-intensive)
    print("  Loading feature files...")
    features_dict = {}
    for i, pid in enumerate(manifest["problems"]):
        try:
            resp = s3.get_object(
                Bucket=BUCKET,
                Key=f"c2c3_training_ready/cached_features/{pid}.npy"
            )
            features_dict[pid] = np.load(io.BytesIO(resp["Body"].read()))
            if (i + 1) % 500 == 0:
                print(f"    [{i+1}/{len(manifest['problems'])}]")
        except Exception as e:
            print(f"    Warning: failed to load {pid}: {e}")

    print(f"  Loaded features for {len(features_dict)} problems")

    return features_dict, window_labels


def train_c2(features_dict, window_labels, device, epochs=50, batch_size=256, lr=1e-3):
    """Train C2 classifier."""
    print("\n" + "=" * 60)
    print("TRAINING C2")
    print("=" * 60)

    # Create dataset
    dataset = C2Dataset(features_dict, window_labels)
    print(f"Total samples: {len(dataset)}")

    # Split by problem (prevent leakage)
    all_pids = list(set(s["features"].shape for s in dataset.samples))
    # Actually split samples directly with stratification
    labels = [s["label"] for s in dataset.samples]
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, stratify=labels, random_state=42
    )

    train_samples = [dataset.samples[i] for i in train_idx]
    val_samples = [dataset.samples[i] for i in val_idx]

    # Rebuild datasets
    train_ds = type('Dataset', (), {
        '__len__': lambda self: len(train_samples),
        '__getitem__': lambda self, i: {
            "features": torch.tensor(train_samples[i]["features"], dtype=torch.float32),
            "label": torch.tensor(train_samples[i]["label"], dtype=torch.long),
        }
    })()

    val_ds = type('Dataset', (), {
        '__len__': lambda self: len(val_samples),
        '__getitem__': lambda self, i: {
            "features": torch.tensor(val_samples[i]["features"], dtype=torch.float32),
            "label": torch.tensor(val_samples[i]["label"], dtype=torch.long),
        }
    })()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Class weights for imbalanced data
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    class_weights = torch.tensor([
        total / (len(label_counts) * label_counts.get(i, 1))
        for i in range(N_C2_CLASSES)
    ], dtype=torch.float32).to(device)

    # Model
    model = C2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0
    patience = 15
    patience_counter = 0
    training_log = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        correct_no_noop = 0
        total_no_noop = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                labels_batch = batch["label"].to(device)

                logits = model(features)
                loss = criterion(logits, labels_batch)
                val_loss += loss.item()

                preds = logits.argmax(dim=-1)
                correct += (preds == labels_batch).sum().item()

                # Accuracy excluding NO_OP
                mask = labels_batch != 0
                if mask.any():
                    correct_no_noop += (preds[mask] == labels_batch[mask]).sum().item()
                    total_no_noop += mask.sum().item()

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels_batch.cpu().numpy().tolist())

        val_loss /= len(val_loader)
        val_acc = correct / len(val_ds) * 100
        val_acc_no_noop = correct_no_noop / total_no_noop * 100 if total_no_noop > 0 else 0

        scheduler.step()

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_acc_no_noop": val_acc_no_noop,
        }
        training_log.append(log_entry)

        print(f"Epoch {epoch+1:3d}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"Acc {val_acc:.1f}% | Acc(ops) {val_acc_no_noop:.1f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save best model
    model.load_state_dict(best_state)
    save_c2_model(model, training_log)

    return model


def train_c3(features_dict, window_labels, device, epochs=50, batch_size=128, lr=1e-3):
    """Train C3 operand extractor."""
    print("\n" + "=" * 60)
    print("TRAINING C3")
    print("=" * 60)

    # Create dataset
    dataset = C3Dataset(features_dict, window_labels)
    print(f"Total samples (non-NO_OP): {len(dataset)}")

    if len(dataset) == 0:
        print("No C3 training samples! Check window labels.")
        return None

    # Split
    indices = list(range(len(dataset)))
    n_operands_list = [s["n_operands"] for s in dataset.samples]
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, stratify=n_operands_list, random_state=42
    )

    train_samples = [dataset.samples[i] for i in train_idx]
    val_samples = [dataset.samples[i] for i in val_idx]

    train_ds = type('Dataset', (), {
        '__len__': lambda self: len(train_samples),
        '__getitem__': lambda self, i: {
            "features": torch.tensor(train_samples[i]["features"], dtype=torch.float32),
            "c2_label": torch.tensor(train_samples[i]["c2_label"], dtype=torch.long),
            "n_operands": torch.tensor(train_samples[i]["n_operands"], dtype=torch.long),
        }
    })()

    val_ds = type('Dataset', (), {
        '__len__': lambda self: len(val_samples),
        '__getitem__': lambda self, i: {
            "features": torch.tensor(val_samples[i]["features"], dtype=torch.float32),
            "c2_label": torch.tensor(val_samples[i]["c2_label"], dtype=torch.long),
            "n_operands": torch.tensor(val_samples[i]["n_operands"], dtype=torch.long),
        }
    })()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = C3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience = 15
    patience_counter = 0
    training_log = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            c2_label = batch["c2_label"].to(device)
            n_operands = batch["n_operands"].to(device)

            optimizer.zero_grad()
            count_logits, _ = model(features, c2_label)
            loss = criterion(count_logits, n_operands)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        correct = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                c2_label = batch["c2_label"].to(device)
                n_operands = batch["n_operands"].to(device)

                count_logits, _ = model(features, c2_label)
                loss = criterion(count_logits, n_operands)
                val_loss += loss.item()

                preds = count_logits.argmax(dim=-1)
                correct += (preds == n_operands).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct / len(val_ds) * 100

        scheduler.step()

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "count_acc": val_acc,
        }
        training_log.append(log_entry)

        print(f"Epoch {epoch+1:3d}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"Count Acc {val_acc:.1f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Save best model
    model.load_state_dict(best_state)
    save_c3_model(model, training_log)

    return model


def save_c2_model(model, training_log):
    """Save C2 model to S3."""
    print("\nSaving C2 model...")

    # Save checkpoint
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    s3.put_object(
        Bucket=BUCKET,
        Key="models/c2_v1/best_checkpoint.pt",
        Body=buffer.read()
    )

    # Save training log
    s3.put_object(
        Bucket=BUCKET,
        Key="models/c2_v1/training_log.json",
        Body=json.dumps(training_log, indent=2).encode("utf-8")
    )

    # Save config
    config = {
        "input_dim": HIDDEN_DIM,
        "n_classes": N_C2_CLASSES,
        "architecture": "MLP(896->256->128->26)",
    }
    s3.put_object(
        Bucket=BUCKET,
        Key="models/c2_v1/config.json",
        Body=json.dumps(config, indent=2).encode("utf-8")
    )

    print(f"  Saved to s3://{BUCKET}/models/c2_v1/")


def save_c3_model(model, training_log):
    """Save C3 model to S3."""
    print("\nSaving C3 model...")

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    s3.put_object(
        Bucket=BUCKET,
        Key="models/c3_v1/best_checkpoint.pt",
        Body=buffer.read()
    )

    s3.put_object(
        Bucket=BUCKET,
        Key="models/c3_v1/training_log.json",
        Body=json.dumps(training_log, indent=2).encode("utf-8")
    )

    config = {
        "hidden_dim": HIDDEN_DIM,
        "c2_embed_dim": 32,
        "n_c2_classes": N_C2_CLASSES,
        "max_operands": MAX_OPERANDS,
        "window_size": W,
    }
    s3.put_object(
        Bucket=BUCKET,
        Key="models/c3_v1/config.json",
        Body=json.dumps(config, indent=2).encode("utf-8")
    )

    print(f"  Saved to s3://{BUCKET}/models/c3_v1/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-c2", action="store_true")
    parser.add_argument("--train-c3", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if args.train_c2 or args.train_c3:
        features_dict, window_labels = load_data()

        if args.train_c2:
            train_c2(features_dict, window_labels, device)

        if args.train_c3:
            train_c3(features_dict, window_labels, device)

    elif args.evaluate:
        print("Evaluation not implemented yet")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
