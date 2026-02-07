#!/usr/bin/env python3
"""Fine-tune MiniLM on Qwen-extracted coarse-grained spans.

Uses the spans from extract_qwen_spans.py (hidden-state boundary detection)
as training data. Clusters spans by operational similarity using existing
atomic templates, then trains with MultipleNegativesRankingLoss so spans
doing the same operation cluster together in embedding space.

Result: MiniLM learns to produce attention patterns that respect the same
clause boundaries that Qwen identifies, enabling correct span detection
at inference without needing Qwen.

Usage:
    python scripts/finetune_coarse_spans.py \
        --spans data/qwen_coarse_spans.json \
        --templates data/atomic_templates_1k.json \
        --output models/minilm_coarse_finetuned
"""

import argparse
import json
import random
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def load_spans(path: str) -> List[dict]:
    """Load Qwen-extracted coarse spans."""
    with open(path) as f:
        data = json.load(f)
    spans = data["spans"] if isinstance(data, dict) else data
    print(f"Loaded {len(spans)} spans from {path}")
    return spans


def load_templates(path: str) -> Tuple[List[dict], np.ndarray]:
    """Load atomic templates and build centroid matrix for matching."""
    with open(path) as f:
        data = json.load(f)

    templates = data if isinstance(data, list) else data.get("atomic_spans", data.get("templates", []))

    # Build centroid matrix
    centroids = []
    valid_templates = []
    for t in templates:
        if t.get("embedding"):
            centroids.append(t["embedding"])
            valid_templates.append(t)

    centroid_matrix = np.array(centroids, dtype=np.float32)
    norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
    centroid_matrix = centroid_matrix / np.maximum(norms, 1e-10)

    print(f"Loaded {len(valid_templates)} templates with embeddings")
    return valid_templates, centroid_matrix


def assign_clusters(
    spans: List[dict],
    templates: List[dict],
    centroid_matrix: np.ndarray,
    model: SentenceTransformer,
    batch_size: int = 512,
) -> Dict[int, List[str]]:
    """Assign each span to the nearest template cluster.

    Returns dict mapping cluster_id -> list of span texts.
    """
    # Extract span texts
    span_texts = [s["span_text"] for s in spans if s.get("span_text", "").strip()]

    # Filter out very short spans (< 3 words)
    span_texts = [t for t in span_texts if len(t.split()) >= 2]
    print(f"Encoding {len(span_texts)} spans...")

    # Batch encode
    embeddings = model.encode(span_texts, batch_size=batch_size,
                              normalize_embeddings=True, show_progress_bar=True)

    # Match each span to nearest template
    print("Matching spans to templates...")
    by_cluster = defaultdict(list)
    similarities = []

    for i, (text, emb) in enumerate(zip(span_texts, embeddings)):
        sims = centroid_matrix @ emb
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        similarities.append(best_sim)
        by_cluster[best_idx].append(text)

    print(f"  Mean similarity: {np.mean(similarities):.4f}")
    print(f"  Clusters used: {len(by_cluster)}")
    print(f"  Cluster size range: {min(len(v) for v in by_cluster.values())}-{max(len(v) for v in by_cluster.values())}")

    # Show top clusters
    for cid in sorted(by_cluster.keys(), key=lambda k: -len(by_cluster[k]))[:10]:
        examples = by_cluster[cid][:3]
        pat = templates[cid].get("pattern", "?")
        print(f"  Cluster {cid} ({pat!r}): {len(by_cluster[cid])} spans, e.g. {[e[:40] for e in examples]}")

    return by_cluster


def build_training_pairs(by_cluster: Dict[int, List[str]], max_pairs: int = 200000) -> List[InputExample]:
    """Build positive pairs for contrastive learning.

    Each pair: two spans from the same cluster (same operation).
    MultipleNegativesRankingLoss uses in-batch negatives automatically.
    """
    examples = []

    # Only use clusters with >= 2 members
    valid_clusters = {k: v for k, v in by_cluster.items() if len(v) >= 2}
    print(f"\nBuilding pairs from {len(valid_clusters)} clusters (>= 2 members)")

    for cid, texts in valid_clusters.items():
        # Number of pairs proportional to cluster size, capped
        n_pairs = min(len(texts) * 3, 500)

        for _ in range(n_pairs):
            a, b = random.sample(texts, 2)
            examples.append(InputExample(texts=[a, b]))

    random.shuffle(examples)

    if len(examples) > max_pairs:
        examples = random.sample(examples, max_pairs)

    print(f"  Total training pairs: {len(examples)}")
    return examples


def evaluate(model: SentenceTransformer) -> Dict[str, float]:
    """Quick evaluation with hand-crafted test cases."""
    test_cases = {
        "SET": [
            "Janet has 3 apples",
            "There are 12 students",
            "Bob bought 5 oranges",
            "costs $ 100",
        ],
        "ADD": [
            "she got 4 more",
            "received an additional 7",
            "found 3 more shells",
            "plus 10 more",
        ],
        "SUB": [
            "gave away 2 cookies",
            "spent 10 dollars",
            "lost 5 marbles",
            "sold 3 of them",
        ],
        "MUL": [
            "each costs 3 dollars",
            "worked 4 hours at 12 per hour",
            "bought twice as many",
            "3 times as much",
        ],
        "DIV": [
            "split equally among 4",
            "divided by 3",
            "shared evenly between",
            "half of the total",
        ],
    }

    all_texts = []
    all_labels = []
    for label, texts in test_cases.items():
        for t in texts:
            all_texts.append(t)
            all_labels.append(label)

    embeddings = model.encode(all_texts, normalize_embeddings=True)
    sim_matrix = embeddings @ embeddings.T

    intra, inter = [], []
    for i in range(len(all_texts)):
        for j in range(i + 1, len(all_texts)):
            if all_labels[i] == all_labels[j]:
                intra.append(sim_matrix[i][j])
            else:
                inter.append(sim_matrix[i][j])

    return {
        "intra_sim": float(np.mean(intra)),
        "inter_sim": float(np.mean(inter)),
        "separation": float(np.mean(intra) - np.mean(inter)),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiniLM on coarse spans")
    parser.add_argument("--spans", default="data/qwen_coarse_spans.json")
    parser.add_argument("--templates", default="data/atomic_templates_1k.json")
    parser.add_argument("--output", default="models/minilm_coarse_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int, default=200000)
    args = parser.parse_args()

    print("=" * 70)
    print("COARSE-GRAINED MiniLM FINE-TUNING")
    print("=" * 70)

    # Load data
    spans = load_spans(args.spans)
    templates, centroid_matrix = load_templates(args.templates)

    # Load base model for cluster assignment
    print("\nLoading base MiniLM for cluster assignment...")
    base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Assign spans to clusters
    by_cluster = assign_clusters(spans, templates, centroid_matrix, base_model)

    # Build training pairs
    examples = build_training_pairs(by_cluster, max_pairs=args.max_pairs)

    if not examples:
        print("ERROR: No training pairs generated!")
        sys.exit(1)

    # Evaluate BEFORE training
    print("\n=== BEFORE Training ===")
    before = evaluate(base_model)
    print(f"  Intra-class sim: {before['intra_sim']:.4f}")
    print(f"  Inter-class sim: {before['inter_sim']:.4f}")
    print(f"  Separation: {before['separation']:.4f}")

    # Train
    print(f"\nTraining config:")
    print(f"  Pairs: {len(examples)}")
    print(f"  Batch: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=base_model)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training ({args.epochs} epochs, {total_steps} steps) ===")
    base_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=str(output_path),
        show_progress_bar=True,
    )

    # Evaluate AFTER training
    print("\n=== AFTER Training ===")
    trained_model = SentenceTransformer(str(output_path))
    after = evaluate(trained_model)
    print(f"  Intra-class sim: {after['intra_sim']:.4f}")
    print(f"  Inter-class sim: {after['inter_sim']:.4f}")
    print(f"  Separation: {after['separation']:.4f}")

    print(f"\n  Improvement: {after['separation'] - before['separation']:+.4f}")

    # Save stats
    stats = {
        "training_pairs": len(examples),
        "clusters_used": len(by_cluster),
        "total_spans": len(spans),
        "before": before,
        "after": after,
        "improvement": after["separation"] - before["separation"],
        "args": vars(args),
    }
    with open(output_path / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nModel saved to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
