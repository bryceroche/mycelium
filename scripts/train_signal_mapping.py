#!/usr/bin/env python3
"""Train a mapping from MiniLM embeddings to Qwen attention signals."""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


class SignalMapper(nn.Module):
    """MLP to map MiniLM embeddings (384-dim) to Qwen signals (3-dim)."""

    def __init__(self, input_dim=384, hidden_dim=128, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # Signals are normalized to [0, 1]
        )

    def forward(self, x):
        return self.net(x)


def load_training_data(templates_path: str = "specialized_templates.json"):
    """Load embeddings and signals from specialized templates."""
    with open(templates_path, 'r') as f:
        templates = json.load(f)

    embeddings = []
    signals = []

    for key, t in templates.items():
        if 'embedding_centroid' not in t:
            continue
        if any(t.get(k) is None for k in ['attention_entropy', 'attention_received', 'attention_connection']):
            continue

        emb = t['embedding_centroid']
        entropy = t['attention_entropy']
        received = t['attention_received']
        connection = t['attention_connection']

        embeddings.append(emb)
        signals.append([entropy, received, connection])

    return np.array(embeddings, dtype=np.float32), np.array(signals, dtype=np.float32)


def train_mapper(embeddings, signals, epochs=100, batch_size=256, lr=0.001):
    """Train the signal mapper."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    # Split data
    n = len(embeddings)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.9 * n)]
    val_idx = indices[int(0.9 * n):]

    X_train = torch.tensor(embeddings[train_idx], device=device)
    y_train = torch.tensor(signals[train_idx], device=device)
    X_val = torch.tensor(embeddings[val_idx], device=device)
    y_val = torch.tensor(signals[val_idx], device=device)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"Signal stats - entropy: [{signals[:, 0].min():.3f}, {signals[:, 0].max():.3f}]")
    print(f"Signal stats - received: [{signals[:, 1].min():.3f}, {signals[:, 1].max():.3f}]")
    print(f"Signal stats - connection: [{signals[:, 2].min():.3f}, {signals[:, 2].max():.3f}]")

    # Create model
    model = SignalMapper().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={total_loss/len(train_loader):.6f}, val_loss={val_loss:.6f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Calculate correlations
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).cpu().numpy()
        val_true = y_val.cpu().numpy()

        correlations = []
        for i, name in enumerate(['entropy', 'received', 'connection']):
            corr = np.corrcoef(val_pred[:, i], val_true[:, i])[0, 1]
            correlations.append(corr)
            print(f"Correlation for {name}: {corr:.4f}")

    return model, {
        'val_loss': best_val_loss,
        'correlations': correlations,
        'signal_stats': {
            'entropy_range': [float(signals[:, 0].min()), float(signals[:, 0].max())],
            'received_range': [float(signals[:, 1].min()), float(signals[:, 1].max())],
            'connection_range': [float(signals[:, 2].min()), float(signals[:, 2].max())]
        }
    }


def main():
    print("Loading training data...")
    embeddings, signals = load_training_data()
    print(f"Loaded {len(embeddings)} samples")

    print("\nTraining signal mapper...")
    model, stats = train_mapper(embeddings, signals)

    # Save model
    model_path = Path("models/minilm_to_qwen_mapping.pt")
    model_path.parent.mkdir(exist_ok=True)

    torch.save({
        'model_state_dict': model.cpu().state_dict(),
        'stats': stats
    }, model_path)

    print(f"\nModel saved to {model_path}")
    print(f"Validation loss: {stats['val_loss']:.6f}")
    print(f"Correlations: {stats['correlations']}")


if __name__ == "__main__":
    main()
