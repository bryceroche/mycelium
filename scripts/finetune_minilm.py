#!/usr/bin/env python3
"""
Fine-tune MiniLM on atomic span clusters.

Uses supervised contrastive learning: same-cluster atoms are pulled together,
different-cluster atoms are pushed apart. This teaches MiniLM operational
semantics — "person buys n items" vs "person gave n items" become distinguishable
even though they're lexically similar.

Input: atomic_templates.json (1K clusters) + atomic_spans.json (106K labeled atoms)
Output: fine-tuned MiniLM model at ./minilm_finetuned/

USAGE:
    python scripts/finetune_minilm.py
    python scripts/finetune_minilm.py --epochs 5 --batch-size 512
"""

import argparse
import json
import random
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

OUTPUT_DIR = Path(__file__).parent.parent


def load_training_data(templates_path: Path, atoms_path: Path):
    """Load atoms and assign cluster labels from templates."""
    print("Loading data...")

    with open(templates_path) as f:
        templates = json.load(f)

    with open(atoms_path) as f:
        atoms = json.load(f)

    print(f"  Templates: {len(templates)}")
    print(f"  Atoms: {len(atoms)}")

    # Build pattern → cluster_id mapping from templates
    pattern_to_cluster = {}
    for tpl in templates:
        cid = tpl["template_id"]
        for pat in tpl.get("all_patterns", []):
            pattern_to_cluster[pat] = cid
        # Also map the representative pattern
        pattern_to_cluster[tpl["pattern"]] = cid

    # Assign cluster labels to atoms
    labeled = []
    unmatched = 0
    for a in atoms:
        text = a["atomic"]
        cid = pattern_to_cluster.get(text)
        if cid is not None:
            labeled.append({"text": text, "cluster_id": cid})
        else:
            unmatched += 1

    print(f"  Labeled atoms: {len(labeled)}")
    print(f"  Unmatched: {unmatched}")

    # Group by cluster
    cluster_groups = defaultdict(list)
    for item in labeled:
        cluster_groups[item["cluster_id"]].append(item["text"])

    # Deduplicate within clusters
    for cid in cluster_groups:
        cluster_groups[cid] = list(set(cluster_groups[cid]))

    usable_clusters = {cid: texts for cid, texts in cluster_groups.items()
                       if len(texts) >= 2}
    print(f"  Clusters with >= 2 unique texts: {len(usable_clusters)}")

    return cluster_groups, usable_clusters


def build_training_pairs(usable_clusters: Dict[str, List[str]],
                         max_pairs: int = 150000) -> List[Tuple[str, str]]:
    """Build (anchor, positive) pairs from same-cluster atoms.

    MultipleNegativesRankingLoss uses in-batch negatives, so we only need
    positive pairs — other examples in the batch serve as negatives.
    """
    print(f"\nBuilding training pairs (max {max_pairs})...")

    pairs = []
    cluster_ids = list(usable_clusters.keys())

    # Generate pairs proportional to cluster size
    for cid, texts in usable_clusters.items():
        n = len(texts)
        # Number of pairs proportional to cluster size
        # but capped so large clusters don't dominate
        n_pairs = min(n * 2, 200)
        for _ in range(n_pairs):
            a, b = random.sample(texts, 2)
            pairs.append((a, b))

    # Shuffle and cap
    random.shuffle(pairs)
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    print(f"  Training pairs: {len(pairs)}")
    return pairs


def build_eval_data(cluster_groups: Dict[str, List[str]],
                    n_eval: int = 2000) -> List[Tuple[str, str, float]]:
    """Build evaluation triplets: (text1, text2, label).
    label=1.0 for same cluster, 0.0 for different cluster.
    """
    cluster_ids = [cid for cid, texts in cluster_groups.items() if len(texts) >= 2]
    all_texts = [(text, cid) for cid, texts in cluster_groups.items() for text in texts]

    eval_data = []
    # Positive pairs
    for _ in range(n_eval // 2):
        cid = random.choice(cluster_ids)
        texts = cluster_groups[cid]
        if len(texts) >= 2:
            a, b = random.sample(texts[:20], 2)  # sample from first 20 to avoid bias
            eval_data.append((a, b, 1.0))

    # Negative pairs
    for _ in range(n_eval // 2):
        cid1, cid2 = random.sample(cluster_ids, 2)
        a = random.choice(cluster_groups[cid1][:10])
        b = random.choice(cluster_groups[cid2][:10])
        eval_data.append((a, b, 0.0))

    random.shuffle(eval_data)
    return eval_data


def train(pairs: List[Tuple[str, str]],
          eval_data: List[Tuple[str, str, float]],
          output_path: Path,
          epochs: int = 3,
          batch_size: int = 256,
          lr: float = 2e-5,
          warmup_ratio: float = 0.1):
    """Fine-tune MiniLM with MultipleNegativesRankingLoss."""
    from sentence_transformers import (
        SentenceTransformer, InputExample, losses, evaluation
    )
    from torch.utils.data import DataLoader

    print(f"\n{'='*60}")
    print(f"FINE-TUNING MiniLM")
    print(f"{'='*60}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Output: {output_path}")

    # Load base model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Build training examples
    train_examples = [InputExample(texts=[a, b]) for a, b in pairs]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    # Loss: MultipleNegativesRankingLoss — in-batch negatives
    # With batch_size=256, each positive pair gets 255 negatives for free
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluation
    eval_sentences1 = [e[0] for e in eval_data]
    eval_sentences2 = [e[1] for e in eval_data]
    eval_labels = [e[2] for e in eval_data]
    evaluator = evaluation.BinaryClassificationEvaluator(
        eval_sentences1, eval_sentences2, eval_labels,
        name="cluster-eval",
        show_progress_bar=False,
    )

    # Warmup steps
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # Pre-training eval
    print(f"\n  Pre-training evaluation:")
    pre_score = evaluator(model)
    if isinstance(pre_score, dict):
        for k, v in pre_score.items():
            print(f"    {k}: {v:.4f}")
    else:
        print(f"  Score: {pre_score:.4f}")

    # Train
    print(f"\n  Training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader) // 2,  # eval twice per epoch
        output_path=str(output_path),
        show_progress_bar=True,
        save_best_model=True,
    )

    # Post-training eval
    print(f"\n  Post-training evaluation:")
    # Reload best model
    model = SentenceTransformer(str(output_path))
    post_score = evaluator(model)
    if isinstance(post_score, dict):
        for k, v in post_score.items():
            print(f"    {k}: {v:.4f}")
    else:
        print(f"  Score: {post_score:.4f}")

    return model


def evaluate_clusters(model, cluster_groups: Dict[str, List[str]],
                      n_clusters: int = 50):
    """Evaluate cluster quality with the fine-tuned model."""
    print(f"\n{'='*60}")
    print(f"CLUSTER QUALITY EVALUATION")
    print(f"{'='*60}")

    # Sample clusters
    cluster_ids = [cid for cid, texts in cluster_groups.items()
                   if len(texts) >= 5]
    if len(cluster_ids) > n_clusters:
        cluster_ids = random.sample(cluster_ids, n_clusters)

    intra_sims = []
    inter_sims = []

    for cid in cluster_ids:
        texts = cluster_groups[cid][:20]  # cap per cluster
        if len(texts) < 2:
            continue

        embs = model.encode(texts, normalize_embeddings=True)

        # Intra-cluster similarity
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                sim = float(np.dot(embs[i], embs[j]))
                intra_sims.append(sim)

    # Inter-cluster similarity (sample pairs of clusters)
    for _ in range(min(500, len(cluster_ids) * 5)):
        cid1, cid2 = random.sample(cluster_ids, 2)
        t1 = random.choice(cluster_groups[cid1][:10])
        t2 = random.choice(cluster_groups[cid2][:10])
        embs = model.encode([t1, t2], normalize_embeddings=True)
        sim = float(np.dot(embs[0], embs[1]))
        inter_sims.append(sim)

    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0
    separation = intra_mean - inter_mean

    print(f"  Intra-cluster similarity: {intra_mean:.4f} (std={np.std(intra_sims):.4f})")
    print(f"  Inter-cluster similarity: {inter_mean:.4f} (std={np.std(inter_sims):.4f})")
    print(f"  Separation gap: {separation:.4f}")
    print(f"  (Higher separation = better discriminative power)")

    # Show some example clusters
    print(f"\n  Sample clusters (first 5 texts each):")
    for cid in cluster_ids[:10]:
        texts = cluster_groups[cid][:5]
        print(f"\n  Cluster {cid}:")
        for t in texts:
            print(f"    - {t}")

    return {
        "intra_sim": intra_mean,
        "inter_sim": inter_mean,
        "separation": separation,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MiniLM")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-pairs", type=int, default=150000)
    parser.add_argument("--templates", type=str, default=None)
    parser.add_argument("--atoms", type=str, default=None)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("MiniLM Fine-Tuning on Atomic Span Clusters")
    print("=" * 60)

    templates_path = Path(args.templates) if args.templates else OUTPUT_DIR / "atomic_templates.json"
    atoms_path = Path(args.atoms) if args.atoms else OUTPUT_DIR / "atomic_spans.json"
    output_path = OUTPUT_DIR / "minilm_finetuned"

    # Load and prep data
    cluster_groups, usable_clusters = load_training_data(templates_path, atoms_path)

    # Build training pairs
    pairs = build_training_pairs(usable_clusters, max_pairs=args.max_pairs)

    # Build eval data
    eval_data = build_eval_data(cluster_groups)
    print(f"  Eval pairs: {len(eval_data)}")

    # Pre-training cluster quality (base model)
    from sentence_transformers import SentenceTransformer
    print(f"\n--- BASE MODEL (before fine-tuning) ---")
    base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    base_metrics = evaluate_clusters(base_model, cluster_groups)
    del base_model

    # Train
    model = train(
        pairs, eval_data, output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Post-training cluster quality
    print(f"\n--- FINE-TUNED MODEL ---")
    ft_metrics = evaluate_clusters(model, cluster_groups)

    # Summary
    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model saved to: {output_path}")
    print(f"  Intra-cluster sim: {base_metrics['intra_sim']:.4f} → {ft_metrics['intra_sim']:.4f}")
    print(f"  Inter-cluster sim: {base_metrics['inter_sim']:.4f} → {ft_metrics['inter_sim']:.4f}")
    print(f"  Separation gap:    {base_metrics['separation']:.4f} → {ft_metrics['separation']:.4f}")


if __name__ == "__main__":
    main()
