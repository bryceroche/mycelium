#!/usr/bin/env python3
"""End-to-end test for fine-tuned MiniLM span detection.

Tests the fine-tuned model's ability to detect semantic spans in math word problems,
matching the attention patterns of Qwen 7B.

USAGE:
    python test_end_to_end.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict


# ============================================================================
# Model Class (copied from training script)
# ============================================================================

class AttentionDistillationModel(nn.Module):
    """MiniLM with learned head/layer weights for attention distillation."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name, output_attentions=True)

        # Learned weights for combining attention heads (12 heads)
        self.head_weights = nn.Parameter(torch.ones(12))

        # Learned weights for combining layers (6 layers)
        self.layer_weights = nn.Parameter(torch.ones(6))

        # Small projection to align with Qwen's attention space
        self.use_projection = True
        if self.use_projection:
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


# ============================================================================
# Community Detection (Span Finding)
# ============================================================================

def find_spans_greedy(connectivity, tokens, threshold=0.15, min_span_size=2):
    """
    Find semantic spans using greedy community detection on connectivity matrix.
    
    Algorithm:
    1. Start with each token as its own community
    2. Merge communities with highest inter-community connectivity
    3. Stop when no pair exceeds threshold
    
    Returns list of spans: [(start_idx, end_idx, tokens_in_span), ...]
    """
    n = len(tokens)
    if n == 0:
        return []
    
    # Normalize connectivity to [0, 1]
    conn = connectivity.copy()
    conn = (conn - conn.min()) / (conn.max() - conn.min() + 1e-10)
    
    # Initialize: each token is its own community
    communities = [{i} for i in range(n)]
    
    # Iteratively merge most connected communities
    while len(communities) > 1:
        best_score = -1
        best_pair = None
        
        for i, comm_i in enumerate(communities):
            for j, comm_j in enumerate(communities):
                if i >= j:
                    continue
                
                # Average connectivity between communities
                total = 0
                count = 0
                for ti in comm_i:
                    for tj in comm_j:
                        total += conn[ti, tj]
                        count += 1
                avg_conn = total / count if count > 0 else 0
                
                if avg_conn > best_score:
                    best_score = avg_conn
                    best_pair = (i, j)
        
        if best_score < threshold or best_pair is None:
            break
            
        # Merge communities
        i, j = best_pair
        communities[i] = communities[i] | communities[j]
        communities.pop(j)
    
    # Convert communities to spans (contiguous regions)
    spans = []
    for comm in communities:
        if len(comm) < min_span_size:
            continue
        indices = sorted(list(comm))
        
        # Find contiguous runs
        runs = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx != prev + 1:
                runs.append((start, prev))
                start = idx
            prev = idx
        runs.append((start, prev))
        
        # Take the longest run
        longest = max(runs, key=lambda r: r[1] - r[0])
        span_tokens = [tokens[i] for i in range(longest[0], longest[1] + 1)]
        spans.append((longest[0], longest[1], span_tokens))
    
    # Sort by start position
    spans.sort(key=lambda x: x[0])
    return spans


def find_spans_spectral(connectivity, tokens, n_clusters=None, threshold=0.1):
    """
    Find semantic spans using spectral clustering on connectivity matrix.
    
    Uses eigendecomposition to find natural clusters in the attention graph.
    """
    n = len(tokens)
    if n == 0:
        return []
    
    # Normalize connectivity
    conn = connectivity.copy()
    conn = (conn - conn.min()) / (conn.max() - conn.min() + 1e-10)
    
    # Threshold to create adjacency
    adj = (conn > threshold).astype(float)
    np.fill_diagonal(adj, 0)
    
    # Compute degree matrix and Laplacian
    degree = np.diag(adj.sum(axis=1) + 1e-10)
    laplacian = degree - adj
    
    # Normalized Laplacian
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree)))
    norm_laplacian = d_inv_sqrt @ laplacian @ d_inv_sqrt
    
    # Eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(norm_laplacian)
    except:
        return find_spans_greedy(connectivity, tokens, threshold)
    
    # Find number of clusters from eigengap if not specified
    if n_clusters is None:
        gaps = np.diff(eigenvalues[:min(10, n)])
        n_clusters = int(np.argmax(gaps) + 1)
        n_clusters = max(2, min(n_clusters, n // 2))
    
    # Use first k eigenvectors for clustering
    features = eigenvectors[:, :n_clusters]
    
    # Simple k-means style clustering
    from sklearn.cluster import KMeans
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
    except:
        return find_spans_greedy(connectivity, tokens, threshold)
    
    # Convert labels to spans
    spans = []
    for cluster_id in range(n_clusters):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) < 2:
            continue
        
        # Find contiguous runs
        indices = sorted(indices)
        runs = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx != prev + 1:
                runs.append((start, prev))
                start = idx
            prev = idx
        runs.append((start, prev))
        
        # Take the longest contiguous run
        longest = max(runs, key=lambda r: r[1] - r[0])
        if longest[1] - longest[0] >= 1:  # At least 2 tokens
            span_tokens = [tokens[i] for i in range(longest[0], longest[1] + 1)]
            spans.append((longest[0], longest[1], span_tokens))
    
    spans.sort(key=lambda x: x[0])
    return spans


# ============================================================================
# Visualization
# ============================================================================

def visualize_connectivity_ascii(connectivity, tokens, max_display=30):
    """
    Print ASCII visualization of connectivity matrix.
    Shows token-token attention strengths.
    """
    n = min(len(tokens), max_display)
    conn = connectivity[:n, :n]
    
    # Normalize to 0-9 scale
    conn_norm = (conn - conn.min()) / (conn.max() - conn.min() + 1e-10)
    conn_int = (conn_norm * 9).astype(int)
    
    # Truncate tokens for display
    display_tokens = [t[:6].ljust(6) for t in tokens[:n]]
    
    print("\n" + "="*60)
    print("CONNECTIVITY MATRIX (0=low, 9=high attention)")
    print("="*60)
    
    # Header
    print(" " * 7, end="")
    for i, tok in enumerate(display_tokens):
        print(f"{i:3d}", end="")
    print()
    
    # Matrix rows
    for i, tok in enumerate(display_tokens):
        print(f"{i:2d} {tok}", end=" ")
        for j in range(n):
            val = conn_int[i, j]
            if val >= 7:
                char = "#"
            elif val >= 4:
                char = "+"
            elif val >= 2:
                char = "."
            else:
                char = " "
            print(f"  {char}", end="")
        print()


def visualize_spans(tokens, spans, problem_text):
    """Print detected spans with highlighting."""
    print("\n" + "="*60)
    print("DETECTED SEMANTIC SPANS")
    print("="*60)
    print(f"Problem: {problem_text}\n")
    
    if not spans:
        print("No spans detected.")
        return
    
    for i, (start, end, span_tokens) in enumerate(spans):
        span_text = " ".join(span_tokens)
        print(f"Span {i+1} [{start}:{end}]: '{span_text}'")
    
    # Show full tokenization with span markers
    print("\nFull tokenization with spans:")
    span_markers = {}
    for i, (start, end, _) in enumerate(spans):
        span_markers[start] = f"[S{i+1}:"
        span_markers[end] = f":S{i+1}]"
    
    result = []
    for i, tok in enumerate(tokens):
        if i in span_markers and span_markers[i].startswith("["):
            result.append(span_markers[i])
        result.append(tok)
        if i in span_markers and span_markers[i].startswith(":"):
            result.append(span_markers[i])
    
    print(" ".join(result))


# ============================================================================
# Main Test
# ============================================================================

def load_model(model_path):
    """Load fine-tuned model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AttentionDistillationModel()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Training correlation: {checkpoint.get('correlation', 'N/A'):.4f}")
    print(f"Trained for {checkpoint.get('epoch', 'N/A')+1} epochs")
    
    weights = model.get_learned_weights()
    print(f"\nLearned head weights: {weights['head_weights'].round(3)}")
    print(f"Learned layer weights: {weights['layer_weights'].round(3)}")
    
    return model, device


def process_problem(model, tokenizer, problem_text, device):
    """Process a single problem and return connectivity + tokens."""
    encoded = tokenizer(
        problem_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    with torch.no_grad():
        connectivity = model(input_ids, attention_mask)
    
    connectivity = connectivity[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    
    return connectivity, tokens


def load_qwen_sample(data_dir, sample_idx=0):
    """Load a sample from Qwen attention data for comparison."""
    meta_path = data_dir / "metadata_0000.json"
    features_path = data_dir / "features_0000.npz"
    
    if not meta_path.exists():
        print(f"Warning: Could not find {meta_path}")
        return None, None, None
    
    metadata = json.load(open(meta_path))
    features = np.load(features_path, allow_pickle=True)
    
    if sample_idx >= len(metadata) or sample_idx >= len(features["connectivity"]):
        sample_idx = 0
    
    problem_text = metadata[sample_idx]["problem_text"]
    qwen_conn = features["connectivity"][sample_idx]
    
    return problem_text, qwen_conn, metadata[sample_idx]


def compare_with_qwen(model, tokenizer, data_dir, device, num_samples=5):
    """Compare model predictions with Qwen ground truth."""
    print("\n" + "="*60)
    print("COMPARISON WITH QWEN 7B GROUND TRUTH")
    print("="*60)
    
    meta_path = data_dir / "metadata_0000.json"
    features_path = data_dir / "features_0000.npz"
    
    if not meta_path.exists():
        print("Could not find Qwen data.")
        return
    
    metadata = json.load(open(meta_path))
    features = np.load(features_path, allow_pickle=True)
    connectivity = features["connectivity"]
    
    correlations = []
    
    for i in range(min(num_samples, len(metadata))):
        problem_text = metadata[i]["problem_text"]
        qwen_conn = connectivity[i]
        
        # Get model prediction
        pred_conn, tokens = process_problem(model, tokenizer, problem_text, device)
        
        # Align sizes
        min_len = min(pred_conn.shape[0], qwen_conn.shape[0])
        pred_flat = pred_conn[:min_len, :min_len].flatten()
        qwen_flat = qwen_conn[:min_len, :min_len].flatten()
        
        # Compute correlation
        if np.std(pred_flat) > 0 and np.std(qwen_flat) > 0:
            corr = np.corrcoef(pred_flat, qwen_flat)[0, 1]
            correlations.append(corr)
            
            print(f"\nSample {i+1}: correlation = {corr:.4f}")
            print(f"  Problem: {problem_text[:80]}...")
    
    if correlations:
        avg_corr = np.mean(correlations)
        print(f"\n{'='*60}")
        print(f"AVERAGE CORRELATION: {avg_corr:.4f}")
        print(f"{'='*60}")


def main():
    print("="*60)
    print("FINE-TUNED MINILM SPAN DETECTION TEST")
    print("="*60)
    
    # Paths
    model_path = Path.home() / "models" / "minilm_finetuned" / "best_model.pt"
    data_dir = Path.home() / "qwen_data"
    
    # Load model
    print("\n[1] Loading fine-tuned model...")
    model, device = load_model(model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    # Test problems
    test_problems = [
        "Sally has 12 apples. She sold 5 apples. How many does she have now?",
        "Tom found 8 coins. He then found 3 more coins. How many coins does Tom have?",
        "A farmer has 20 eggs. She sold half of them. How many eggs remain?",
    ]
    
    print("\n[2] Testing span detection on sample problems...\n")
    
    for idx, problem in enumerate(test_problems):
        print(f"\n{'#'*60}")
        print(f"TEST PROBLEM {idx+1}")
        print(f"{'#'*60}")
        
        connectivity, tokens = process_problem(model, tokenizer, problem, device)
        
        # Visualize connectivity
        visualize_connectivity_ascii(connectivity, tokens)
        
        # Detect spans using greedy method
        spans_greedy = find_spans_greedy(connectivity, tokens, threshold=0.15)
        visualize_spans(tokens, spans_greedy, problem)
        
        # Also try spectral clustering if sklearn available
        try:
            spans_spectral = find_spans_spectral(connectivity, tokens, threshold=0.1)
            if spans_spectral:
                print("\nSpectral clustering spans:")
                for i, (start, end, span_tokens) in enumerate(spans_spectral):
                    print(f"  Span {i+1} [{start}:{end}]: '{' '.join(span_tokens)}'")
        except ImportError:
            print("(sklearn not available for spectral clustering)")
    
    # Compare with Qwen ground truth
    print("\n[3] Comparing with Qwen 7B ground truth...\n")
    compare_with_qwen(model, tokenizer, data_dir, device, num_samples=5)
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
