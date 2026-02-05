#!/usr/bin/env python3
"""
Contrastive fine-tuning of MiniLM for operation classification.

Goal: Train embeddings to cluster by operation type (ADD, SUB, MUL, DIV, SET),
not by lexical similarity.

Training approach:
- Triplet loss: anchor, positive (same op), negative (different op)
- Online hard negative mining for efficiency
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict
import argparse


class TripletDataset(Dataset):
    """Dataset that generates triplets: (anchor, positive, negative)."""

    def __init__(self, templates: List[dict], samples_per_epoch: int = 10000, balanced: bool = True):
        self.samples_per_epoch = samples_per_epoch
        self.balanced = balanced

        # Group templates by operation
        self.by_operation: Dict[str, List[dict]] = defaultdict(list)
        for t in templates:
            op = t.get('dsl_expr', t.get('operation_type', 'unknown'))
            if op != 'unknown' and t.get('span_examples'):
                self.by_operation[op].append(t)

        self.operations = list(self.by_operation.keys())
        print(f"Operations: {self.operations}")
        for op in self.operations:
            print(f"  {op}: {len(self.by_operation[op])} templates")

        if balanced:
            # Use uniform sampling across operations (not weighted by count)
            print(f"  Using BALANCED sampling (uniform across operations)")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Pick random operation for anchor
        anchor_op = random.choice(self.operations)

        # Pick anchor template and span
        anchor_template = random.choice(self.by_operation[anchor_op])
        anchor_span = random.choice(anchor_template['span_examples'])

        # Pick positive (same operation, different template if possible)
        pos_templates = self.by_operation[anchor_op]
        if len(pos_templates) > 1:
            pos_template = random.choice([t for t in pos_templates if t != anchor_template])
        else:
            pos_template = anchor_template
        pos_span = random.choice(pos_template['span_examples'])

        # Pick negative (different operation)
        neg_op = random.choice([op for op in self.operations if op != anchor_op])
        neg_template = random.choice(self.by_operation[neg_op])
        neg_span = random.choice(neg_template['span_examples'])

        return anchor_span, pos_span, neg_span, anchor_op, neg_op


class ContrastiveModel(nn.Module):
    """MiniLM with projection head for contrastive learning."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.transformer.config.hidden_size

        # Optional projection head (helps with contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 128),  # Project to smaller space
        )

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take attention mask into account."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings with gradient tracking."""
        device = next(self.parameters()).device

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Get transformer output
        output = self.transformer(**encoded)

        # Mean pooling
        embeddings = self.mean_pooling(output, encoded['attention_mask'])

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Project for contrastive loss
        projected = self.projection(embeddings)
        return F.normalize(projected, p=2, dim=1)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts without projection (for inference)."""
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            output = self.transformer(**encoded)
            embeddings = self.mean_pooling(output, encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor,
                 margin: float = 0.5) -> torch.Tensor:
    """Triplet margin loss."""
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def infonce_loss(anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE contrastive loss."""
    # Similarity to positive
    pos_sim = F.cosine_similarity(anchor, positive, dim=1) / temperature

    # Similarity to negatives (batch serves as negatives)
    neg_sim = torch.mm(anchor, negatives.t()) / temperature

    # InfoNCE: log softmax over positive vs all negatives
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)


def train_epoch(model: ContrastiveModel, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                device: str, loss_type: str = "triplet") -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, (anchors, positives, negatives, anchor_ops, neg_ops) in enumerate(dataloader):
        optimizer.zero_grad()

        # Encode all texts
        all_texts = list(anchors) + list(positives) + list(negatives)
        all_embeddings = model(all_texts)

        batch_size = len(anchors)
        anchor_emb = all_embeddings[:batch_size]
        pos_emb = all_embeddings[batch_size:2*batch_size]
        neg_emb = all_embeddings[2*batch_size:]

        if loss_type == "triplet":
            loss = triplet_loss(anchor_emb, pos_emb, neg_emb, margin=0.5)
        else:
            loss = infonce_loss(anchor_emb, pos_emb, neg_emb, temperature=0.07)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: loss={loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate_clustering(model: ContrastiveModel, templates: List[dict],
                        samples_per_op: int = 100) -> Dict[str, float]:
    """Evaluate how well embeddings cluster by operation."""
    model.eval()

    # Group by operation
    by_op = defaultdict(list)
    for t in templates:
        op = t.get('dsl_expr', t.get('operation_type'))
        if op and t.get('span_examples'):
            by_op[op].extend(t['span_examples'][:3])  # Take up to 3 examples per template

    # Sample and encode
    embeddings_by_op = {}
    for op, spans in by_op.items():
        sampled = random.sample(spans, min(samples_per_op, len(spans)))
        with torch.no_grad():
            emb = model.encode(sampled)
        embeddings_by_op[op] = emb

    # Compute intra-class vs inter-class distances
    intra_dists = []
    inter_dists = []

    ops = list(embeddings_by_op.keys())
    for i, op1 in enumerate(ops):
        emb1 = embeddings_by_op[op1]

        # Intra-class: distances within same operation
        for j in range(len(emb1)):
            for k in range(j+1, min(j+10, len(emb1))):  # Sample pairs
                dist = np.linalg.norm(emb1[j] - emb1[k])
                intra_dists.append(dist)

        # Inter-class: distances to different operations
        for op2 in ops[i+1:]:
            emb2 = embeddings_by_op[op2]
            for j in range(min(20, len(emb1))):
                for k in range(min(20, len(emb2))):
                    dist = np.linalg.norm(emb1[j] - emb2[k])
                    inter_dists.append(dist)

    avg_intra = np.mean(intra_dists)
    avg_inter = np.mean(inter_dists)
    separation = avg_inter / avg_intra  # Higher is better

    return {
        "avg_intra_dist": avg_intra,
        "avg_inter_dist": avg_inter,
        "separation_ratio": separation,
    }


def main():
    parser = argparse.ArgumentParser(description="Contrastive fine-tuning for operation classification")
    parser.add_argument("--templates", default="specialized_templates.json", help="Path to templates")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--samples-per-epoch", type=int, default=10000, help="Triplets per epoch")
    parser.add_argument("--loss", choices=["triplet", "infonce"], default="triplet", help="Loss type")
    parser.add_argument("--output", default="models/minilm_contrastive.pt", help="Output path")
    parser.add_argument("--balanced", action="store_true", default=True, help="Use balanced sampling")
    parser.add_argument("--hard-negatives", action="store_true", help="Use hard negative mining")
    args = parser.parse_args()

    # Load templates
    project_root = Path(__file__).parent.parent
    templates_path = project_root / args.templates

    print(f"Loading templates from {templates_path}")
    with open(templates_path) as f:
        data = json.load(f)

    templates = data.get('templates', data) if isinstance(data, dict) else data
    if isinstance(templates, dict):
        templates = list(templates.values())
    print(f"Loaded {len(templates)} templates")

    # Create dataset
    dataset = TripletDataset(templates, samples_per_epoch=args.samples_per_epoch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ContrastiveModel()
    model.to(device)

    # Evaluate before training
    print("\n=== Before Training ===")
    metrics = evaluate_clustering(model, templates)
    print(f"Intra-class dist: {metrics['avg_intra_dist']:.4f}")
    print(f"Inter-class dist: {metrics['avg_inter_dist']:.4f}")
    print(f"Separation ratio: {metrics['separation_ratio']:.4f}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n=== Training ({args.epochs} epochs, {args.loss} loss) ===")
    best_separation = metrics['separation_ratio']

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, device, args.loss)
        print(f"Average loss: {avg_loss:.4f}")

        # Evaluate
        metrics = evaluate_clustering(model, templates)
        print(f"Intra-class dist: {metrics['avg_intra_dist']:.4f}")
        print(f"Inter-class dist: {metrics['avg_inter_dist']:.4f}")
        print(f"Separation ratio: {metrics['separation_ratio']:.4f}")

        # Save best model
        if metrics['separation_ratio'] > best_separation:
            best_separation = metrics['separation_ratio']
            output_path = project_root / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'projection_state_dict': model.projection.state_dict(),
                'metrics': metrics,
                'args': vars(args),
            }, output_path)
            print(f"Saved best model (separation={best_separation:.4f})")

    # Also save the transformer and tokenizer for easy loading
    encoder_path = project_root / "models" / "minilm_contrastive_encoder"
    encoder_path.mkdir(parents=True, exist_ok=True)
    model.transformer.save_pretrained(str(encoder_path))
    model.tokenizer.save_pretrained(str(encoder_path))
    print(f"\nSaved transformer to {encoder_path}")

    print("\n=== Training Complete ===")
    print(f"Best separation ratio: {best_separation:.4f}")


if __name__ == "__main__":
    main()
