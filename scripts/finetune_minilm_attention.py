#!/usr/bin/env python3
"""Fine-tune MiniLM to match Qwen attention patterns.

Uses attention distillation: train MiniLM's attention to match Qwen's
connectivity patterns on math word problems.

Key innovations:
1. Learned head weights - not all 12 heads are equal
2. Learned layer weights - optimal mixing of 6 layers
3. Small projection head - linear transform to Qwen's attention space

USAGE:
    python scripts/finetune_minilm_attention.py --data-dir ./data/qwen_attention --epochs 10
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


class QwenAttentionDataset(Dataset):
    """Dataset of (text, qwen_connectivity) pairs."""

    def __init__(self, data_dir: Path, max_batches: int = None):
        self.samples = []

        batch_id = 0
        while True:
            meta_path = data_dir / f"metadata_{batch_id:04d}.json"
            features_path = data_dir / f"features_{batch_id:04d}.npz"

            if not meta_path.exists():
                break

            if max_batches and batch_id >= max_batches:
                break

            metadata = json.load(open(meta_path))
            features = np.load(features_path, allow_pickle=True)
            connectivity = features["connectivity"]

            for i, problem in enumerate(metadata):
                if i < len(connectivity):
                    self.samples.append({
                        "text": problem["problem_text"],
                        "connectivity": connectivity[i],
                    })

            batch_id += 1

        print(f"Loaded {len(self.samples)} samples from {batch_id} batches")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AttentionDistillationModel(nn.Module):
    """MiniLM with learned head/layer weights for attention distillation."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name, output_attentions=True)

        # Freeze encoder initially (optional - can unfreeze later)
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        # Learned weights for combining attention heads (12 heads)
        self.head_weights = nn.Parameter(torch.ones(12))

        # Learned weights for combining layers (6 layers)
        self.layer_weights = nn.Parameter(torch.ones(6))

        # Small projection to align with Qwen's attention space
        # Project from seq_len x seq_len to same (learned transform)
        self.use_projection = True
        if self.use_projection:
            # We'll apply a learned scalar + bias per position
            self.proj_scale = nn.Parameter(torch.ones(1))
            self.proj_bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):
        """Forward pass returning processed attention connectivity."""

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

        # Stack attention from all layers: (num_layers, batch, heads, seq, seq)
        attentions = torch.stack(outputs.attentions, dim=0)

        # Apply learned layer weights
        layer_w = F.softmax(self.layer_weights, dim=0)
        # (num_layers,) -> (num_layers, 1, 1, 1, 1)
        layer_w = layer_w.view(-1, 1, 1, 1, 1)
        weighted_attn = (attentions * layer_w).sum(dim=0)  # (batch, heads, seq, seq)

        # Apply learned head weights
        head_w = F.softmax(self.head_weights, dim=0)
        head_w = head_w.view(1, -1, 1, 1)
        weighted_attn = (weighted_attn * head_w).sum(dim=1)  # (batch, seq, seq)

        # Compute connectivity: sqrt(attn * attn.T)
        connectivity = torch.sqrt(weighted_attn * weighted_attn.transpose(-1, -2) + 1e-10)

        # Apply projection
        if self.use_projection:
            connectivity = self.proj_scale * connectivity + self.proj_bias

        return connectivity

    def get_learned_weights(self):
        """Return the learned head and layer weights for inspection."""
        return {
            "head_weights": F.softmax(self.head_weights, dim=0).detach().cpu().numpy(),
            "layer_weights": F.softmax(self.layer_weights, dim=0).detach().cpu().numpy(),
        }


def compute_connectivity_loss(pred_conn, target_conn, mask=None):
    """Compute MSE loss between predicted and target connectivity matrices.

    Handles different sequence lengths by truncating to minimum.
    """
    min_len = min(pred_conn.shape[-1], target_conn.shape[-1])

    pred = pred_conn[..., :min_len, :min_len]
    target = target_conn[..., :min_len, :min_len]

    if mask is not None:
        mask = mask[:, :min_len, :min_len]
        loss = ((pred - target) ** 2 * mask).sum() / mask.sum()
    else:
        loss = F.mse_loss(pred, target)

    return loss


def collate_fn(batch, tokenizer, max_length=512):
    """Collate batch with tokenization and connectivity padding."""

    texts = [item["text"] for item in batch]
    connectivities = [item["connectivity"] for item in batch]

    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Pad connectivity matrices to same size
    max_seq = encoded["input_ids"].shape[1]
    padded_conn = []

    for conn in connectivities:
        # Truncate or pad to max_seq
        if conn.shape[0] > max_seq:
            conn = conn[:max_seq, :max_seq]
        elif conn.shape[0] < max_seq:
            pad_size = max_seq - conn.shape[0]
            conn = np.pad(conn, ((0, pad_size), (0, pad_size)), mode='constant')
        padded_conn.append(conn)

    connectivity_tensor = torch.tensor(np.stack(padded_conn), dtype=torch.float32)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "target_connectivity": connectivity_tensor,
    }


def evaluate(model, dataloader, device):
    """Evaluate model and return average loss + correlation."""
    model.eval()
    total_loss = 0
    correlations = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_conn = batch["target_connectivity"].to(device)

            pred_conn = model(input_ids, attention_mask)
            loss = compute_connectivity_loss(pred_conn, target_conn)
            total_loss += loss.item()

            # Compute correlation for each sample
            for i in range(pred_conn.shape[0]):
                p = pred_conn[i].cpu().numpy().flatten()
                t = target_conn[i].cpu().numpy().flatten()
                min_len = min(len(p), len(t))
                if np.std(p[:min_len]) > 0 and np.std(t[:min_len]) > 0:
                    corr = np.corrcoef(p[:min_len], t[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

    avg_loss = total_loss / len(dataloader)
    avg_corr = np.mean(correlations) if correlations else 0

    return avg_loss, avg_corr


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Load dataset
    data_dir = Path(args.data_dir)
    dataset = QwenAttentionDataset(data_dir, max_batches=args.max_batches)

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=0,
    )

    # Initialize model
    model = AttentionDistillationModel()
    model.to(device)

    # Optimizer - different LR for encoder vs new params
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr * 0.1},  # Lower LR for pretrained
        {"params": [model.head_weights, model.layer_weights], "lr": args.lr},
        {"params": [model.proj_scale, model.proj_bias], "lr": args.lr},
    ])

    # Training loop
    best_corr = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_conn = batch["target_connectivity"].to(device)

            optimizer.zero_grad()
            pred_conn = model(input_ids, attention_mask)
            loss = compute_connectivity_loss(pred_conn, target_conn)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        # Evaluate
        val_loss, val_corr = evaluate(model, val_loader, device)

        # Get learned weights
        weights = model.get_learned_weights()

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Correlation: {val_corr:.3f}")
        print(f"  Head weights: {weights['head_weights'].round(3)}")
        print(f"  Layer weights: {weights['layer_weights'].round(3)}")

        # Save best model
        if val_corr > best_corr:
            best_corr = val_corr
            save_path = Path(args.output_dir) / "best_model.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "head_weights": weights["head_weights"],
                "layer_weights": weights["layer_weights"],
                "correlation": val_corr,
                "epoch": epoch,
            }, save_path)
            print(f"  Saved best model (corr={val_corr:.3f})")

    print(f"\nTraining complete! Best correlation: {best_corr:.3f}")
    return best_corr


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiniLM with Qwen attention")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with Qwen attention data")
    parser.add_argument("--output-dir", type=str, default="./models/minilm_finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches to load (for testing)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
