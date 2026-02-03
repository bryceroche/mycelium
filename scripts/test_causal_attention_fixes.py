#!/usr/bin/env python3
"""Test various fixes for causal attention clustering issues.

PROBLEM: Qwen2.5-Math-7B-Instruct uses causal attention where all tokens
attend heavily to the first token (0.8-1.0 attention weight). This causes
clustering to fail - everything ends up in one cluster.

APPROACHES TESTED:
1. Middle layers instead of last layers (middle layers often have more semantic info)
2. Remove first-token attention column before clustering
3. Use attention entropy as a feature
4. Find semantic heads (heads with high variance in attention patterns)
5. Use hidden states instead of attention for clustering
6. Local attention windows (ignore long-range first-token attention)

Run on the VM with: python scripts/test_causal_attention_fixes.py
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ClusteringResult:
    """Result from a clustering approach."""
    name: str
    n_clusters_found: int
    cluster_assignments: List[int]
    tokens: List[str]
    quality_score: float  # Higher is better
    details: Dict[str, Any]


def load_model():
    """Load Qwen model with attention outputs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded.")
    return model, tokenizer


def get_all_attention_info(model, tokenizer, text: str):
    """Get attention from all layers and heads."""
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

    # attentions: tuple of (batch, heads, seq, seq) for each layer
    all_attentions = outputs.attentions
    hidden_states = outputs.hidden_states

    return {
        'tokens': tokens,
        'attentions': all_attentions,  # List of [1, n_heads, seq, seq]
        'hidden_states': hidden_states,  # List of [1, seq, hidden_dim]
        'n_layers': len(all_attentions),
        'n_heads': all_attentions[0].shape[1],
        'seq_len': len(tokens),
    }


def cluster_from_matrix(dist_matrix: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Run hierarchical clustering on a distance matrix."""
    # Ensure valid distance matrix
    dist_matrix = np.nan_to_num(dist_matrix, nan=1.0, posinf=1.0, neginf=0.0)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.clip(dist_matrix, 0, None)

    # Make symmetric
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    if dist_matrix.shape[0] < 3:
        return np.array([1] * dist_matrix.shape[0])

    dist_condensed = squareform(dist_matrix, checks=False)
    Z = linkage(dist_condensed, method="average")
    return fcluster(Z, t=n_clusters, criterion="maxclust")


def compute_cluster_quality(cluster_ids: np.ndarray, dist_matrix: np.ndarray) -> float:
    """Compute clustering quality - variance between clusters vs within."""
    if len(set(cluster_ids)) <= 1:
        return 0.0  # Everything in one cluster = bad

    # Within-cluster distances
    within = []
    between = []

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            d = dist_matrix[i, j]
            if cluster_ids[i] == cluster_ids[j]:
                within.append(d)
            else:
                between.append(d)

    if not within or not between:
        return 0.0

    # Quality = between / within ratio (higher is better)
    mean_within = np.mean(within)
    mean_between = np.mean(between)

    if mean_within < 1e-6:
        return 10.0  # Perfect within-cluster similarity

    return mean_between / mean_within


# =============================================================================
# APPROACH 1: Use middle layers instead of last layers
# =============================================================================
def approach_middle_layers(info: dict, layer_range: Tuple[int, int] = None) -> ClusteringResult:
    """Use middle layers which often have more semantic information."""
    n_layers = info['n_layers']

    if layer_range is None:
        # Use middle third of layers
        start = n_layers // 3
        end = 2 * n_layers // 3
    else:
        start, end = layer_range

    # Average attention across middle layers and all heads
    attn_layers = []
    for layer_idx in range(start, end):
        attn = info['attentions'][layer_idx][0].mean(dim=0).cpu().numpy()
        attn_layers.append(attn)

    attn = np.mean(attn_layers, axis=0)

    # Symmetrize and convert to distance
    attn_sym = (attn + attn.T) / 2
    dist = 1 - np.clip(attn_sym, 0, 1)

    clusters = cluster_from_matrix(dist, n_clusters=3)
    quality = compute_cluster_quality(clusters, dist)

    return ClusteringResult(
        name=f"middle_layers_{start}-{end}",
        n_clusters_found=len(set(clusters)),
        cluster_assignments=clusters.tolist(),
        tokens=info['tokens'],
        quality_score=quality,
        details={'layer_range': (start, end), 'mean_attn_to_first': attn[:, 0].mean()}
    )


# =============================================================================
# APPROACH 2: Remove first-token attention column
# =============================================================================
def approach_remove_first_token(info: dict) -> ClusteringResult:
    """Remove the first token column from attention before clustering."""
    # Average last 4 layers
    attn_layers = [info['attentions'][i][0].mean(dim=0).cpu().numpy()
                   for i in range(-4, 0)]
    attn = np.mean(attn_layers, axis=0)

    # Zero out first column (attention TO first token)
    attn_fixed = attn.copy()
    attn_fixed[:, 0] = 0

    # Renormalize rows
    row_sums = attn_fixed.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    attn_fixed = attn_fixed / row_sums

    # Symmetrize and convert to distance
    attn_sym = (attn_fixed + attn_fixed.T) / 2
    dist = 1 - np.clip(attn_sym, 0, 1)

    clusters = cluster_from_matrix(dist, n_clusters=3)
    quality = compute_cluster_quality(clusters, dist)

    return ClusteringResult(
        name="remove_first_token",
        n_clusters_found=len(set(clusters)),
        cluster_assignments=clusters.tolist(),
        tokens=info['tokens'],
        quality_score=quality,
        details={'original_first_attn': attn[:, 0].mean()}
    )


# =============================================================================
# APPROACH 3: Use attention entropy as features
# =============================================================================
def approach_attention_entropy(info: dict) -> ClusteringResult:
    """Use per-token attention entropy as a clustering feature."""
    # Average last 4 layers
    attn_layers = [info['attentions'][i][0].mean(dim=0).cpu().numpy()
                   for i in range(-4, 0)]
    attn = np.mean(attn_layers, axis=0)

    # Compute entropy for each token's attention distribution
    n_tokens = len(info['tokens'])
    entropies = []
    for i in range(n_tokens):
        # Attention from token i to all previous tokens
        row = attn[i, :i+1]
        if len(row) > 1 and row.sum() > 0:
            row = row / row.sum()  # Normalize
            h = entropy(row + 1e-10)
        else:
            h = 0
        entropies.append(h)

    # Also get attention variance per token
    variances = [attn[i, :i+1].var() if i > 0 else 0 for i in range(n_tokens)]

    # Create feature matrix: [entropy, variance, position]
    features = np.column_stack([
        entropies,
        variances,
        np.linspace(0, 1, n_tokens)  # Normalized position
    ])

    # Cluster on features
    if len(features) >= 3:
        dist = pdist(features, metric='euclidean')
        dist_matrix = squareform(dist)
        dist_matrix = dist_matrix / (dist_matrix.max() + 1e-10)  # Normalize
        clusters = cluster_from_matrix(dist_matrix, n_clusters=3)
        quality = compute_cluster_quality(clusters, dist_matrix)
    else:
        clusters = np.array([1] * n_tokens)
        quality = 0

    return ClusteringResult(
        name="attention_entropy",
        n_clusters_found=len(set(clusters)),
        cluster_assignments=clusters.tolist(),
        tokens=info['tokens'],
        quality_score=quality,
        details={'entropies': entropies, 'variances': variances}
    )


# =============================================================================
# APPROACH 4: Find semantic heads (high variance heads)
# =============================================================================
def approach_semantic_heads(info: dict, top_k: int = 8) -> ClusteringResult:
    """Find heads with highest attention variance (more semantic)."""
    n_layers = info['n_layers']
    n_heads = info['n_heads']
    n_tokens = info['seq_len']

    # Score each head by its attention variance
    head_scores = []
    for layer_idx in range(n_layers):
        layer_attn = info['attentions'][layer_idx][0].cpu().numpy()  # [heads, seq, seq]
        for head_idx in range(n_heads):
            head_attn = layer_attn[head_idx]
            # Variance across all attention values (excluding diagonal)
            mask = ~np.eye(n_tokens, dtype=bool)
            var = head_attn[mask].var()
            head_scores.append((layer_idx, head_idx, var, head_attn))

    # Sort by variance (highest first)
    head_scores.sort(key=lambda x: x[2], reverse=True)

    # Average top-k highest variance heads
    top_heads = head_scores[:top_k]
    attn = np.mean([h[3] for h in top_heads], axis=0)

    # Symmetrize and convert to distance
    attn_sym = (attn + attn.T) / 2
    dist = 1 - np.clip(attn_sym, 0, 1)

    clusters = cluster_from_matrix(dist, n_clusters=3)
    quality = compute_cluster_quality(clusters, dist)

    return ClusteringResult(
        name=f"semantic_heads_top{top_k}",
        n_clusters_found=len(set(clusters)),
        cluster_assignments=clusters.tolist(),
        tokens=info['tokens'],
        quality_score=quality,
        details={
            'top_heads': [(l, h, f"{v:.4f}") for l, h, v, _ in top_heads[:5]],
            'mean_attn_to_first': attn[:, 0].mean()
        }
    )


# =============================================================================
# APPROACH 5: Use hidden states for clustering
# =============================================================================
def approach_hidden_states(info: dict, layer: int = -1) -> ClusteringResult:
    """Use hidden state similarity for clustering (bypasses attention entirely)."""
    hidden = info['hidden_states'][layer][0].cpu().numpy()  # [seq, hidden_dim]

    # Compute cosine similarity between token representations
    norms = np.linalg.norm(hidden, axis=1, keepdims=True)
    hidden_normed = hidden / (norms + 1e-10)
    similarity = hidden_normed @ hidden_normed.T

    # Convert to distance
    dist = 1 - similarity
    dist = np.clip(dist, 0, 2)

    clusters = cluster_from_matrix(dist, n_clusters=3)
    quality = compute_cluster_quality(clusters, dist)

    n_layers = info['n_layers']
    actual_layer = layer if layer >= 0 else n_layers + layer + 1

    return ClusteringResult(
        name=f"hidden_states_layer{actual_layer}",
        n_clusters_found=len(set(clusters)),
        cluster_assignments=clusters.tolist(),
        tokens=info['tokens'],
        quality_score=quality,
        details={'layer': actual_layer, 'hidden_dim': hidden.shape[1]}
    )


# =============================================================================
# APPROACH 6: Local attention windows
# =============================================================================
def approach_local_attention(info: dict, window_size: int = 4) -> ClusteringResult:
    """Only use attention within a local window (ignore long-range to first token)."""
    # Average last 4 layers
    attn_layers = [info['attentions'][i][0].mean(dim=0).cpu().numpy()
                   for i in range(-4, 0)]
    attn = np.mean(attn_layers, axis=0)

    n_tokens = len(info['tokens'])

    # Create local attention matrix
    local_attn = np.zeros_like(attn)
    for i in range(n_tokens):
        start = max(0, i - window_size)
        end = min(n_tokens, i + window_size + 1)
        local_attn[i, start:end] = attn[i, start:end]

    # Renormalize rows
    row_sums = local_attn.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    local_attn = local_attn / row_sums

    # Symmetrize and convert to distance
    attn_sym = (local_attn + local_attn.T) / 2
    dist = 1 - np.clip(attn_sym, 0, 1)

    clusters = cluster_from_matrix(dist, n_clusters=3)
    quality = compute_cluster_quality(clusters, dist)

    return ClusteringResult(
        name=f"local_attention_w{window_size}",
        n_clusters_found=len(set(clusters)),
        cluster_assignments=clusters.tolist(),
        tokens=info['tokens'],
        quality_score=quality,
        details={'window_size': window_size}
    )


# =============================================================================
# APPROACH 7: Combined - remove first + middle layers + semantic heads
# =============================================================================
def approach_combined(info: dict) -> ClusteringResult:
    """Combine multiple fixes: middle layers + remove first token + semantic heads."""
    n_layers = info['n_layers']
    n_heads = info['n_heads']
    n_tokens = info['seq_len']

    # Use middle layers
    start = n_layers // 3
    end = 2 * n_layers // 3

    # Find high-variance heads in middle layers
    head_scores = []
    for layer_idx in range(start, end):
        layer_attn = info['attentions'][layer_idx][0].cpu().numpy()
        for head_idx in range(n_heads):
            head_attn = layer_attn[head_idx]
            # Variance excluding first column
            mask = np.ones((n_tokens, n_tokens), dtype=bool)
            mask[:, 0] = False
            np.fill_diagonal(mask, False)
            if mask.sum() > 0:
                var = head_attn[mask].var()
            else:
                var = 0
            head_scores.append((layer_idx, head_idx, var, head_attn))

    head_scores.sort(key=lambda x: x[2], reverse=True)
    top_heads = head_scores[:8]

    # Average top heads
    attn = np.mean([h[3] for h in top_heads], axis=0)

    # Remove first token column
    attn[:, 0] = 0
    row_sums = attn.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    attn = attn / row_sums

    # Symmetrize and convert to distance
    attn_sym = (attn + attn.T) / 2
    dist = 1 - np.clip(attn_sym, 0, 1)

    clusters = cluster_from_matrix(dist, n_clusters=3)
    quality = compute_cluster_quality(clusters, dist)

    return ClusteringResult(
        name="combined_middle_semantic_nofirst",
        n_clusters_found=len(set(clusters)),
        cluster_assignments=clusters.tolist(),
        tokens=info['tokens'],
        quality_score=quality,
        details={
            'layer_range': (start, end),
            'top_heads': [(l, h, f"{v:.4f}") for l, h, v, _ in top_heads[:3]]
        }
    )


# =============================================================================
# APPROACH 8: Gradient-weighted attention (if backward info available)
# =============================================================================
def approach_relative_attention(info: dict) -> ClusteringResult:
    """Use relative attention (subtract row mean) to reduce first-token bias."""
    # Average last 4 layers
    attn_layers = [info['attentions'][i][0].mean(dim=0).cpu().numpy()
                   for i in range(-4, 0)]
    attn = np.mean(attn_layers, axis=0)

    # Subtract row mean (relative attention)
    row_means = attn.mean(axis=1, keepdims=True)
    attn_relative = attn - row_means

    # Shift to positive
    attn_relative = attn_relative - attn_relative.min()
    attn_relative = attn_relative / (attn_relative.max() + 1e-10)

    # Symmetrize and convert to distance
    attn_sym = (attn_relative + attn_relative.T) / 2
    dist = 1 - np.clip(attn_sym, 0, 1)

    clusters = cluster_from_matrix(dist, n_clusters=3)
    quality = compute_cluster_quality(clusters, dist)

    return ClusteringResult(
        name="relative_attention",
        n_clusters_found=len(set(clusters)),
        cluster_assignments=clusters.tolist(),
        tokens=info['tokens'],
        quality_score=quality,
        details={'row_mean_subtracted': True}
    )


def visualize_clusters(result: ClusteringResult) -> str:
    """Create a text visualization of clustering."""
    tokens = result.tokens
    clusters = result.cluster_assignments

    # Color-code by cluster
    colors = ['[A]', '[B]', '[C]', '[D]', '[E]']
    output = []
    for tok, c in zip(tokens, clusters):
        color = colors[(c - 1) % len(colors)]
        tok_clean = tok.replace('Ġ', '_').replace('▁', '_')
        output.append(f"{color}{tok_clean}")

    return ' '.join(output)


def run_all_approaches(model, tokenizer, text: str) -> List[ClusteringResult]:
    """Run all approaches on a text."""
    print(f"\n{'='*60}")
    print(f"TEXT: \"{text}\"")
    print('='*60)

    info = get_all_attention_info(model, tokenizer, text)
    print(f"Tokens ({info['seq_len']}): {info['tokens']}")
    print(f"Model: {info['n_layers']} layers, {info['n_heads']} heads")

    results = []

    # Test each approach
    approaches = [
        lambda: approach_middle_layers(info),
        lambda: approach_middle_layers(info, (info['n_layers']//4, info['n_layers']//2)),
        lambda: approach_remove_first_token(info),
        lambda: approach_attention_entropy(info),
        lambda: approach_semantic_heads(info, top_k=4),
        lambda: approach_semantic_heads(info, top_k=8),
        lambda: approach_hidden_states(info, layer=-1),
        lambda: approach_hidden_states(info, layer=info['n_layers']//2),
        lambda: approach_local_attention(info, window_size=3),
        lambda: approach_local_attention(info, window_size=5),
        lambda: approach_combined(info),
        lambda: approach_relative_attention(info),
    ]

    for approach in approaches:
        try:
            result = approach()
            results.append(result)
        except Exception as e:
            print(f"Error in approach: {e}")

    return results


def print_results(results: List[ClusteringResult]):
    """Print results sorted by quality."""
    print(f"\n{'='*60}")
    print("RESULTS (sorted by quality score)")
    print('='*60)

    results_sorted = sorted(results, key=lambda r: r.quality_score, reverse=True)

    for i, result in enumerate(results_sorted):
        print(f"\n{i+1}. {result.name}")
        print(f"   Quality: {result.quality_score:.3f}")
        print(f"   Clusters found: {result.n_clusters_found}")
        print(f"   Visualization: {visualize_clusters(result)}")
        if result.details:
            for k, v in result.details.items():
                print(f"   {k}: {v}")


def main():
    """Main test function."""
    model, tokenizer = load_model()

    test_cases = [
        "Lisa has 5 more apples than John",
        "Mary has twice as many books as Tom",
        "She sold 5 apples",
        "He found 8 coins then spent 3",
        "John gave Mary 7 pencils",
        "The store has 12 oranges and 8 bananas",
    ]

    all_results = {}

    for text in test_cases:
        results = run_all_approaches(model, tokenizer, text)
        print_results(results)
        all_results[text] = results

    # Summary: which approach wins most often?
    print("\n" + "="*60)
    print("SUMMARY: Best approach per test case")
    print("="*60)

    approach_wins = {}
    for text, results in all_results.items():
        if results:
            best = max(results, key=lambda r: r.quality_score)
            approach_wins[best.name] = approach_wins.get(best.name, 0) + 1
            print(f"\n\"{text[:40]}...\"")
            print(f"  Winner: {best.name} (quality={best.quality_score:.3f})")

    print("\n" + "="*60)
    print("OVERALL WINS:")
    print("="*60)
    for name, wins in sorted(approach_wins.items(), key=lambda x: -x[1]):
        print(f"  {name}: {wins} wins")


if __name__ == "__main__":
    main()
