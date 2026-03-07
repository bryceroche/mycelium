"""
Train a classifier on IB 20-dim features to predict clusters.
This is the correct validation: can we LEARN the mapping from X to cluster?
"""

import io
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import boto3

s3 = boto3.client("s3")
BUCKET = "mycelium-data"

N_CLUSTERS = 25


class IBClassifier(nn.Module):
    """Small MLP to classify 20-dim IB features into 25 clusters."""
    def __init__(self, input_dim=20, n_clusters=N_CLUSTERS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_clusters)
        )

    def forward(self, x):
        return self.net(x)


class IBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"features": self.X[idx], "label": self.y[idx]}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load IB data
    print("\nLoading IB data...")
    resp = s3.get_object(Bucket=BUCKET, Key="ib_ready/X_matrix.npy")
    X = np.load(io.BytesIO(resp["Body"].read()))

    resp = s3.get_object(Bucket=BUCKET, Key="ib_results_math/cluster_assignments.npy")
    y = np.load(io.BytesIO(resp["Body"].read())).astype(np.int64)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Unique clusters: {len(np.unique(y))}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    # Class weights
    counts = Counter(y_train)
    weights = torch.zeros(N_CLUSTERS)
    for c in range(N_CLUSTERS):
        if counts[c] > 0:
            weights[c] = len(y_train) / (N_CLUSTERS * counts[c])
    weights = weights / weights.sum() * N_CLUSTERS

    # Datasets
    train_dataset = IBDataset(X_train, y_train)
    val_dataset = IBDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    # Model
    model = IBClassifier().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Training
    print("\n" + "="*50)
    print("TRAINING IB CLASSIFIER (20-dim → 25 clusters)")
    print("="*50)

    best_val_acc = 0
    best_state = None
    patience = 0

    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            y_batch = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y_batch, weight=weights.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        top3_correct = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                y_batch = batch["label"].to(device)

                logits = model(x)
                loss = F.cross_entropy(logits, y_batch, weight=weights.to(device))
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

                # Top-3
                _, top3 = logits.topk(3, dim=1)
                for i, label in enumerate(y_batch):
                    if label in top3[i]:
                        top3_correct += 1

        val_loss /= len(val_loader)
        val_acc = correct / total
        top3_acc = top3_correct / total

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train {train_loss:.4f} | Val {val_loss:.4f} | "
                  f"Acc {val_acc*100:.1f}% | Top-3 {top3_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Restore best
    model.load_state_dict(best_state)

    # Final eval
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["features"].to(device)
            logits = model(x)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    # Top-3
    model.eval()
    top3_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["features"].to(device)
            logits = model(x)
            _, top3 = logits.topk(3, dim=1)
            for i, label in enumerate(batch["label"]):
                if label in top3[i]:
                    top3_correct += 1

    top3_acc = top3_correct / len(all_labels)

    print(f"Accuracy: {accuracy*100:.1f}%")
    print(f"Top-3 Accuracy: {top3_acc*100:.1f}%")

    # Save
    print("\nSaving model...")
    buffer = io.BytesIO()
    torch.save(best_state, buffer)
    buffer.seek(0)

    s3.put_object(
        Bucket=BUCKET,
        Key="models/ib_classifier_v1/best_checkpoint.pt",
        Body=buffer.read()
    )

    config = {
        "input_dim": 20,
        "n_clusters": N_CLUSTERS,
        "accuracy": accuracy,
        "top3_accuracy": top3_acc,
    }
    s3.put_object(
        Bucket=BUCKET,
        Key="models/ib_classifier_v1/config.json",
        Body=json.dumps(config, indent=2).encode("utf-8")
    )

    print(f"Saved to s3://{BUCKET}/models/ib_classifier_v1/")

    # Verdict
    print("\n" + "="*50)
    print("VERDICT")
    print("="*50)
    if accuracy > 0.5:
        print(f"SUCCESS: {accuracy*100:.1f}% accuracy on 20-dim IB features!")
        print("The 20-dim features DO separate clusters via learned classifier.")
        print("Next: Train C0 to predict full 20-dim, not just 4-dim aggregate.")
    elif top3_acc > 0.5:
        print(f"PARTIAL: Top-3 accuracy {top3_acc*100:.1f}% is usable for factor graph.")
        print("The factor graph can narrow from 25 to ~8 candidates.")
    else:
        print(f"FAILED: Neither accuracy ({accuracy*100:.1f}%) nor top-3 ({top3_acc*100:.1f}%) is useful.")


if __name__ == "__main__":
    main()
