"""
Train Text-Aware Template Energy Model

Learns to score template compatibility based on C1-A hidden states.
Used as an energy term inside the factor graph.

Usage:
    python train_text_energy.py
"""

import json
import io
import argparse
from pathlib import Path
from collections import defaultdict, Counter

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
HIDDEN_DIM = 896
N_CLUSTERS = 26  # 25 IB clusters + NO_OP

# Super-category mapping for evaluation
SUPER_CATEGORIES = {
    0: "NO_OP",
    # SETUP: 1-2
    1: "SETUP", 2: "SETUP",
    # COMPUTE: 3-10 (evaluate, simplify, expand, factor)
    3: "COMPUTE", 4: "COMPUTE", 5: "COMPUTE", 6: "COMPUTE", 7: "COMPUTE",
    8: "COMPUTE", 9: "COMPUTE", 10: "COMPUTE",
    # SOLVE: 11-13
    11: "SOLVE", 12: "SOLVE", 13: "SOLVE",
    # SUBSTITUTE: 14-16
    14: "SUBST", 15: "SUBST", 16: "SUBST",
    # REASON: 17-22 (apply_theorem, count, compare)
    17: "REASON", 18: "REASON", 19: "REASON", 20: "REASON", 21: "REASON", 22: "REASON",
    # CONVERT + OTHER: 23-25
    23: "OTHER", 24: "OTHER", 25: "OTHER",
}


class TextTemplateEnergy(nn.Module):
    """
    Learned energy: how compatible is each template with
    the text content of this segment?
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, n_clusters=N_CLUSTERS):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_clusters)
        )

    def forward(self, window_features):
        """
        window_features: (batch, hidden_dim) - mean-pooled C1-A hidden states
        Returns: (batch, n_clusters) - log-compatibility score per cluster
        """
        return self.scorer(window_features)


class TextEnergyDataset(Dataset):
    """Dataset for text energy training."""

    def __init__(self, features_list, labels_list):
        self.features = features_list
        self.labels = labels_list

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_training_data():
    """Load cached features and window labels."""
    print("Loading training data...")

    # Load window labels
    print("  Loading window labels...")
    resp = s3.get_object(Bucket=BUCKET, Key="c2c3_training_ready/window_labels.jsonl")
    content = resp["Body"].read().decode("utf-8")

    window_labels = {}
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        window_labels[d["problem_idx"]] = d["window_labels"]

    print(f"  Loaded labels for {len(window_labels)} problems")

    # Load features and create training examples
    print("  Loading features...")
    features_list = []
    labels_list = []
    problem_ids = []  # Track which problem each example comes from

    problems_loaded = 0
    problems_skipped = 0

    for prob_idx, labels in window_labels.items():
        try:
            resp = s3.get_object(
                Bucket=BUCKET,
                Key=f"c2c3_training_ready/cached_features/{prob_idx}.npy"
            )
            features = np.load(io.BytesIO(resp["Body"].read()))  # (n_windows, W, hidden_dim)

            n_windows = min(len(labels), features.shape[0])

            for w_idx in range(n_windows):
                # Mean pool per-token features
                window_feat = features[w_idx].mean(axis=0)  # (W, 896) → (896,)
                cluster_id = labels[w_idx].get("c2_cluster_id", 0)

                features_list.append(window_feat)
                labels_list.append(cluster_id)
                problem_ids.append(prob_idx)

            problems_loaded += 1
            if problems_loaded % 500 == 0:
                print(f"    Loaded {problems_loaded} problems...")

        except Exception as e:
            problems_skipped += 1
            continue

    print(f"  Loaded {len(features_list)} examples from {problems_loaded} problems")
    print(f"  Skipped {problems_skipped} problems (missing features)")

    return features_list, labels_list, problem_ids


def compute_class_weights(labels):
    """Compute inverse frequency class weights."""
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(N_CLUSTERS)

    for c in range(N_CLUSTERS):
        if counts[c] > 0:
            weights[c] = total / (N_CLUSTERS * counts[c])
        else:
            weights[c] = 1.0

    # Normalize
    weights = weights / weights.sum() * N_CLUSTERS

    return weights


def evaluate(model, dataloader, device, class_weights=None):
    """Evaluate model and compute detailed metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits = model(features)

            # Loss with label smoothing
            loss = F.cross_entropy(
                logits, labels,
                weight=class_weights.to(device) if class_weights is not None else None,
                label_smoothing=0.1
            )
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    overall_acc = (all_preds == all_labels).mean()

    # Accuracy excluding NO_OP
    non_noop_mask = all_labels > 0
    if non_noop_mask.sum() > 0:
        non_noop_acc = (all_preds[non_noop_mask] == all_labels[non_noop_mask]).mean()
    else:
        non_noop_acc = 0.0

    # Top-3 accuracy (for each example, is correct label in top 3 predictions?)
    # Need to re-run with logits
    model.eval()
    top3_correct = 0
    top3_total = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(features)

            _, top3 = logits.topk(3, dim=1)
            for i, label in enumerate(labels):
                if label in top3[i]:
                    top3_correct += 1
                top3_total += 1

    top3_acc = top3_correct / top3_total if top3_total > 0 else 0.0

    # Super-category accuracy
    super_preds = np.array([SUPER_CATEGORIES.get(p, "OTHER") for p in all_preds])
    super_labels = np.array([SUPER_CATEGORIES.get(l, "OTHER") for l in all_labels])
    super_acc = (super_preds == super_labels).mean()

    # Per super-category accuracy
    per_super = {}
    for super_cat in set(SUPER_CATEGORIES.values()):
        mask = super_labels == super_cat
        if mask.sum() > 0:
            per_super[super_cat] = (super_preds[mask] == super_labels[mask]).mean()
        else:
            per_super[super_cat] = 0.0

    # NO_OP precision (of predicted NO_OP, how many are actually NO_OP?)
    noop_pred_mask = all_preds == 0
    if noop_pred_mask.sum() > 0:
        noop_precision = (all_labels[noop_pred_mask] == 0).mean()
    else:
        noop_precision = 0.0

    return {
        "loss": total_loss / n_batches,
        "overall_acc": overall_acc,
        "non_noop_acc": non_noop_acc,
        "top3_acc": top3_acc,
        "super_acc": super_acc,
        "per_super": per_super,
        "noop_precision": noop_precision,
    }


def train(model, train_loader, val_loader, device, class_weights,
          n_epochs=100, patience=20, lr=1e-3):
    """Train the text energy model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(features)

            loss = F.cross_entropy(
                logits, labels,
                weight=class_weights.to(device),
                label_smoothing=0.1
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Validation
        val_metrics = evaluate(model, val_loader, device, class_weights)

        scheduler.step(val_metrics["loss"])

        # Logging
        print(f"Epoch {epoch+1:3d}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Overall Acc: {val_metrics['overall_acc']*100:.1f}%")
        print(f"  Acc excl NO_OP: {val_metrics['non_noop_acc']*100:.1f}%")
        print(f"  Top-3 Acc: {val_metrics['top3_acc']*100:.1f}%")
        print(f"  Super-cat Acc: {val_metrics['super_acc']*100:.1f}%")
        print(f"  NO_OP Precision: {val_metrics['noop_precision']*100:.1f}%")
        print(f"  Per super-category:")
        for cat, acc in sorted(val_metrics['per_super'].items()):
            print(f"    {cat}: {acc*100:.1f}%")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "overall_acc": val_metrics["overall_acc"],
            "non_noop_acc": val_metrics["non_noop_acc"],
            "top3_acc": val_metrics["top3_acc"],
            "super_acc": val_metrics["super_acc"],
            "noop_precision": val_metrics["noop_precision"],
        })

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    features_list, labels_list, problem_ids = load_training_data()

    # Problem-level train/val split
    unique_problems = list(set(problem_ids))
    train_problems, val_problems = train_test_split(
        unique_problems, test_size=0.1, random_state=42
    )
    train_problems = set(train_problems)
    val_problems = set(val_problems)

    # Split examples by problem
    train_features, train_labels = [], []
    val_features, val_labels = [], []

    for feat, label, prob_id in zip(features_list, labels_list, problem_ids):
        if prob_id in train_problems:
            train_features.append(feat)
            train_labels.append(label)
        else:
            val_features.append(feat)
            val_labels.append(label)

    print(f"\nTrain: {len(train_features)} examples from {len(train_problems)} problems")
    print(f"Val: {len(val_features)} examples from {len(val_problems)} problems")

    # Class distribution
    print("\nClass distribution (train):")
    train_counts = Counter(train_labels)
    for c in range(N_CLUSTERS):
        pct = train_counts[c] / len(train_labels) * 100
        print(f"  Class {c}: {train_counts[c]} ({pct:.1f}%)")

    # Compute class weights
    class_weights = compute_class_weights(train_labels)
    print(f"\nClass weights computed (range: {class_weights.min():.2f} - {class_weights.max():.2f})")

    # Create datasets
    train_dataset = TextEnergyDataset(train_features, train_labels)
    val_dataset = TextEnergyDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    model = TextTemplateEnergy().to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

    model, history = train(
        model, train_loader, val_loader, device, class_weights,
        n_epochs=args.epochs, lr=args.lr
    )

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    final_metrics = evaluate(model, val_loader, device, class_weights)
    print(f"Overall Accuracy: {final_metrics['overall_acc']*100:.1f}%")
    print(f"Accuracy excl NO_OP: {final_metrics['non_noop_acc']*100:.1f}%")
    print(f"Top-3 Accuracy: {final_metrics['top3_acc']*100:.1f}%")
    print(f"Super-category Accuracy: {final_metrics['super_acc']*100:.1f}%")
    print(f"NO_OP Precision: {final_metrics['noop_precision']*100:.1f}%")

    # Save model
    print("\nSaving model to S3...")

    # Save checkpoint
    checkpoint = model.state_dict()
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)

    s3.put_object(
        Bucket=BUCKET,
        Key="models/text_energy_v1/best_checkpoint.pt",
        Body=buffer.read()
    )

    # Save config
    config_data = {
        "hidden_dim": HIDDEN_DIM,
        "n_clusters": N_CLUSTERS,
        "final_metrics": {
            "overall_acc": final_metrics["overall_acc"],
            "non_noop_acc": final_metrics["non_noop_acc"],
            "top3_acc": final_metrics["top3_acc"],
            "super_acc": final_metrics["super_acc"],
            "noop_precision": final_metrics["noop_precision"],
        }
    }
    s3.put_object(
        Bucket=BUCKET,
        Key="models/text_energy_v1/config.json",
        Body=json.dumps(config_data, indent=2).encode("utf-8")
    )

    # Save training log
    s3.put_object(
        Bucket=BUCKET,
        Key="models/text_energy_v1/training_log.json",
        Body=json.dumps(history, indent=2).encode("utf-8")
    )

    print(f"Model saved to s3://{BUCKET}/models/text_energy_v1/")

    # Success criteria check
    print("\n" + "="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)
    criteria_met = 0

    if final_metrics["top3_acc"] > 0.30:
        print("✓ Top-3 accuracy > 30%")
        criteria_met += 1
    else:
        print(f"✗ Top-3 accuracy = {final_metrics['top3_acc']*100:.1f}% (need > 30%)")

    if final_metrics["super_acc"] > 0.40:
        print("✓ Super-category accuracy > 40%")
        criteria_met += 1
    else:
        print(f"✗ Super-category accuracy = {final_metrics['super_acc']*100:.1f}% (need > 40%)")

    if final_metrics["noop_precision"] > 0.80:
        print("✓ NO_OP precision > 80%")
        criteria_met += 1
    else:
        print(f"✗ NO_OP precision = {final_metrics['noop_precision']*100:.1f}% (need > 80%)")

    print(f"\nCriteria met: {criteria_met}/3")

    print("\nDone!")


if __name__ == "__main__":
    main()
