#!/usr/bin/env python3
"""Validate correlation between Qwen and MiniLM attention patterns.

The key bet for Path C (hybrid): MiniLM's attention patterns should be
correlated enough with Qwen's to match templates learned from Qwen.

This script:
1. Loads Qwen attention patterns from extracted data
2. Runs MiniLM on the same problems
3. Compares attention patterns (entropy, span connectivity)
4. Reports correlation metrics

USAGE:
    python scripts/validate_attention_correlation.py --data-dir /path/to/qwen_data --num-samples 50
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys


def load_qwen_batch(data_dir: Path, batch_id: int) -> Tuple[List[Dict], Dict]:
    """Load a batch of Qwen attention data."""
    meta_path = data_dir / f"metadata_{batch_id:04d}.json"
    features_path = data_dir / f"features_{batch_id:04d}.npz"

    with open(meta_path) as f:
        metadata = json.load(f)

    features = np.load(features_path, allow_pickle=True)

    return metadata, features


def get_minilm_attention(text: str, model, tokenizer) -> Dict:
    """Extract attention from MiniLM for a text."""
    import torch

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention: (num_layers, num_heads, seq_len, seq_len)
    attention = torch.stack(outputs.attentions).squeeze(1)  # Remove batch dim

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return {
        "tokens": tokens,
        "attention": attention.numpy(),  # (layers, heads, seq, seq)
        "seq_length": len(tokens),
    }


def compute_attention_entropy(attention: np.ndarray) -> np.ndarray:
    """Compute entropy of attention distribution for each token.

    Args:
        attention: (heads, seq, seq) or (layers, heads, seq, seq)

    Returns:
        (seq,) entropy per token position
    """
    # Average over heads (and layers if present)
    if attention.ndim == 4:
        attn = attention.mean(axis=(0, 1))  # Average over layers and heads
    else:
        attn = attention.mean(axis=0)  # Average over heads

    # Compute entropy for each token's attention distribution
    # attn[i, :] is token i's attention over all tokens
    eps = 1e-10
    entropy = -np.sum(attn * np.log(attn + eps), axis=-1)

    return entropy


def compute_span_connectivity(attention: np.ndarray) -> np.ndarray:
    """Compute span connectivity matrix from attention.

    Connectivity[i,j] = how much tokens i and j attend to each other (mutual attention).

    Args:
        attention: (heads, seq, seq) or (layers, heads, seq, seq)

    Returns:
        (seq, seq) symmetric connectivity matrix
    """
    # Average over heads (and layers if present)
    if attention.ndim == 4:
        attn = attention.mean(axis=(0, 1))
    else:
        attn = attention.mean(axis=0)

    # Mutual attention: geometric mean of A[i,j] and A[j,i]
    connectivity = np.sqrt(attn * attn.T + 1e-10)

    return connectivity


def correlate_patterns(qwen_pattern: np.ndarray, minilm_pattern: np.ndarray) -> Dict:
    """Compute correlation metrics between two attention patterns."""
    # Flatten for correlation
    q_flat = qwen_pattern.flatten()
    m_flat = minilm_pattern.flatten()

    # Truncate to same length
    min_len = min(len(q_flat), len(m_flat))
    q_flat = q_flat[:min_len]
    m_flat = m_flat[:min_len]

    # Pearson correlation
    if np.std(q_flat) > 0 and np.std(m_flat) > 0:
        pearson = np.corrcoef(q_flat, m_flat)[0, 1]
    else:
        pearson = 0.0

    # Spearman rank correlation
    from scipy.stats import spearmanr
    spearman, _ = spearmanr(q_flat, m_flat)

    # Cosine similarity
    norm_q = np.linalg.norm(q_flat)
    norm_m = np.linalg.norm(m_flat)
    if norm_q > 0 and norm_m > 0:
        cosine = np.dot(q_flat, m_flat) / (norm_q * norm_m)
    else:
        cosine = 0.0

    return {
        "pearson": float(pearson),
        "spearman": float(spearman) if not np.isnan(spearman) else 0.0,
        "cosine": float(cosine),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Qwen vs MiniLM attention correlation")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with Qwen attention data")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of problems to compare")
    parser.add_argument("--batch-start", type=int, default=0, help="Starting batch ID")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load MiniLM
    print("Loading MiniLM...")
    from transformers import AutoModel, AutoTokenizer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    print(f"MiniLM loaded: {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")

    # Collect correlation metrics
    entropy_correlations = []
    connectivity_correlations = []

    samples_processed = 0
    batch_id = args.batch_start

    print(f"\nComparing attention patterns on {args.num_samples} samples...")
    print("=" * 60)

    while samples_processed < args.num_samples:
        try:
            metadata, features = load_qwen_batch(data_dir, batch_id)
        except FileNotFoundError:
            print(f"Batch {batch_id} not found, stopping.")
            break

        # Get feature arrays (keys: 'entropy', 'received', 'connectivity')
        qwen_entropies = features.get("entropy")
        qwen_connectivity = features.get("connectivity")

        for i, problem in enumerate(metadata):
            if samples_processed >= args.num_samples:
                break

            problem_text = problem.get("problem_text", "")
            if not problem_text:
                continue

            # Get MiniLM attention
            try:
                minilm_data = get_minilm_attention(problem_text, model, tokenizer)
            except Exception as e:
                print(f"  Error on problem: {e}")
                continue

            # Compute MiniLM patterns
            minilm_entropy = compute_attention_entropy(minilm_data["attention"])
            minilm_connectivity = compute_span_connectivity(minilm_data["attention"])

            # Get Qwen patterns for this problem
            if qwen_entropies is not None and i < len(qwen_entropies):
                qwen_entropy = qwen_entropies[i]
                # Truncate to matching length
                min_len = min(len(qwen_entropy), len(minilm_entropy))

                entropy_corr = correlate_patterns(
                    qwen_entropy[:min_len],
                    minilm_entropy[:min_len]
                )
                entropy_correlations.append(entropy_corr)

            if qwen_connectivity is not None and i < len(qwen_connectivity):
                qwen_conn = qwen_connectivity[i]
                # Truncate to matching size
                min_size = min(qwen_conn.shape[0], minilm_connectivity.shape[0])

                conn_corr = correlate_patterns(
                    qwen_conn[:min_size, :min_size],
                    minilm_connectivity[:min_size, :min_size]
                )
                connectivity_correlations.append(conn_corr)

            samples_processed += 1

            if samples_processed % 10 == 0:
                print(f"  Processed {samples_processed}/{args.num_samples} samples...")

        batch_id += 1

    # Report results
    print("\n" + "=" * 60)
    print("RESULTS: Qwen vs MiniLM Attention Correlation")
    print("=" * 60)

    if entropy_correlations:
        avg_entropy = {
            "pearson": np.mean([c["pearson"] for c in entropy_correlations]),
            "spearman": np.mean([c["spearman"] for c in entropy_correlations]),
            "cosine": np.mean([c["cosine"] for c in entropy_correlations]),
        }
        print(f"\nEntropy Correlation (n={len(entropy_correlations)}):")
        print(f"  Pearson:  {avg_entropy['pearson']:.3f}")
        print(f"  Spearman: {avg_entropy['spearman']:.3f}")
        print(f"  Cosine:   {avg_entropy['cosine']:.3f}")

    if connectivity_correlations:
        avg_conn = {
            "pearson": np.mean([c["pearson"] for c in connectivity_correlations]),
            "spearman": np.mean([c["spearman"] for c in connectivity_correlations]),
            "cosine": np.mean([c["cosine"] for c in connectivity_correlations]),
        }
        print(f"\nConnectivity Correlation (n={len(connectivity_correlations)}):")
        print(f"  Pearson:  {avg_conn['pearson']:.3f}")
        print(f"  Spearman: {avg_conn['spearman']:.3f}")
        print(f"  Cosine:   {avg_conn['cosine']:.3f}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")

    if entropy_correlations:
        ep = avg_entropy["pearson"]
        if ep > 0.5:
            print(f"  Entropy: STRONG correlation ({ep:.2f}) - MiniLM agrees with Qwen on 'important' tokens")
        elif ep > 0.3:
            print(f"  Entropy: MODERATE correlation ({ep:.2f}) - Some agreement on token importance")
        else:
            print(f"  Entropy: WEAK correlation ({ep:.2f}) - Different notions of importance")

    if connectivity_correlations:
        cp = avg_conn["pearson"]
        if cp > 0.5:
            print(f"  Connectivity: STRONG ({cp:.2f}) - MiniLM agrees with Qwen on span grouping")
        elif cp > 0.3:
            print(f"  Connectivity: MODERATE ({cp:.2f}) - Partial agreement on spans")
        else:
            print(f"  Connectivity: WEAK ({cp:.2f}) - Different span structure")

    print("\n" + "=" * 60)
    print("Path C viability:")
    if entropy_correlations and connectivity_correlations:
        overall = (avg_entropy["pearson"] + avg_conn["pearson"]) / 2
        if overall > 0.4:
            print(f"  PROMISING ({overall:.2f}) - Hybrid approach likely viable")
        elif overall > 0.25:
            print(f"  UNCERTAIN ({overall:.2f}) - May need attention signature matching")
        else:
            print(f"  CHALLENGING ({overall:.2f}) - May need different approach")


if __name__ == "__main__":
    main()
