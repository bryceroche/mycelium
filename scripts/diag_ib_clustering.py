"""IB clustering diagnostic — discover natural math-operation clusters in GSM8K.

Goal: Replace our hand-picked 12 op_role codebook with DATA-DRIVEN clusters.
Approach:
  1. Embed each L2 NL action description via Pythia tokens + frozen embed.weight (mean-pool)
  2. For each OP (ADD/SUB/MUL/DIV), recursively K-means on the embeddings
  3. Stop splitting when silhouette drops or cluster is too small
  4. Print the cluster tree with sample examples per leaf
  5. Human can then label each leaf as a semantic role

No model training. CPU-only. Uses Pythia's frozen embedding layer to put steps
into the model's native representation space — this is the space the codebook
will live in once we train.

Usage:
  .venv/bin/python scripts/diag_ib_clustering.py [--src JSONL] [--out OUT_TXT]
"""
import os
import sys
import re
import json
import argparse
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad.nn.state import safe_load

from mycelium.data import load_tokenizer


# ---------------- Parsing v80 L2 step descriptions ----------------

# Format: "Step k: <NL action>. OP=<OP>. ARG=<val>."
L2_STEP_RE = re.compile(r"^Step \d+:\s*(.+?)\.\s*OP=(ADD|SUB|MUL|DIV)\.\s*ARG=([-\d.]+)\.\s*$")


def extract_steps(jsonl_path: str):
    """Yield {nl, op, arg, problem, pidx, sidx} per step."""
    with open(jsonl_path) as f:
        for pidx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "L2" not in r:
                continue
            l2 = r["L2"]
            for sidx, raw_ln in enumerate(l2.split("\n")):
                ln = raw_ln.strip()
                m = L2_STEP_RE.match(ln)
                if not m:
                    continue
                yield {
                    "nl": m.group(1).strip(),
                    "op": m.group(2),
                    "arg": m.group(3),
                    "problem": r.get("problem", "")[:120],
                    "pidx": pidx,
                    "sidx": sidx,
                }


# ---------------- Pythia embedding lookup ----------------

def load_pythia_embed_numpy(path: str) -> np.ndarray:
    """Load Pythia-410M embed weights as a numpy array (vocab, hidden).

    Pythia uses 'embed_in.weight' or 'gpt_neox.embed_in.weight'. Probe both.
    """
    sd = safe_load(path)
    key_candidates = [
        "gpt_neox.embed_in.weight",
        "embed_in.weight",
        "embed.weight",
    ]
    for k in key_candidates:
        if k in sd:
            t = sd[k]
            return t.numpy().astype(np.float32)
    available = sorted(sd.keys())[:20]
    raise KeyError(f"Could not find embed weight. First keys in safetensors: {available}")


def embed_text(text: str, tok, embed_w: np.ndarray) -> np.ndarray:
    """Mean-pool of token embeddings. Returns (hidden,) float32."""
    ids = tok.encode(text).ids
    if not ids:
        return np.zeros((embed_w.shape[1],), dtype=np.float32)
    vecs = embed_w[ids]  # (T, hidden)
    return vecs.mean(axis=0)


# ---------------- Numpy K-means + silhouette ----------------

def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ initialization. X: (N, D). Returns (k, D)."""
    N, D = X.shape
    centers = np.empty((k, D), dtype=X.dtype)
    idx0 = rng.integers(0, N)
    centers[0] = X[idx0]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = closest_sq / max(closest_sq.sum(), 1e-12)
        idx = rng.choice(N, p=probs)
        centers[i] = X[idx]
        new_sq = np.sum((X - centers[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centers


def kmeans(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 42):
    """Lloyd's algorithm. Returns (labels, centers, inertia)."""
    N = X.shape[0]
    if k >= N:
        return np.arange(N), X.copy(), 0.0
    rng = np.random.default_rng(seed)
    centers = kmeans_pp_init(X, k, rng)
    for _ in range(max_iter):
        # Assign
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(d2, axis=1)
        # Update
        new_centers = np.zeros_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                new_centers[c] = X[mask].mean(axis=0)
            else:
                # Re-seed empty cluster to the point farthest from any center
                farthest = np.argmax(np.min(d2, axis=1))
                new_centers[c] = X[farthest]
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
    # Inertia (sum of squared distances to assigned centers)
    inertia = float(np.sum((X - centers[labels]) ** 2))
    return labels, centers, inertia


def silhouette_score_np(X: np.ndarray, labels: np.ndarray, sample_max: int = 1500) -> float:
    """Numpy silhouette. Subsamples for speed when N is large."""
    N = X.shape[0]
    if N > sample_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(N, size=sample_max, replace=False)
        X = X[idx]
        labels = labels[idx]
        N = sample_max
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0
    # Pairwise distances (NxN). For N<=1500 this is ~9MB float32 — fine.
    dists = np.sqrt(np.maximum(
        np.sum(X**2, axis=1, keepdims=True)
        + np.sum(X**2, axis=1)[None, :]
        - 2.0 * X @ X.T, 0.0))
    s_vals = np.zeros(N, dtype=np.float32)
    for i in range(N):
        own = labels[i]
        own_mask = (labels == own)
        own_mask[i] = False
        if own_mask.sum() == 0:
            s_vals[i] = 0.0
            continue
        a_i = float(dists[i, own_mask].mean())
        b_i = float("inf")
        for other in unique:
            if other == own:
                continue
            m = labels == other
            if m.sum() > 0:
                b_i = min(b_i, float(dists[i, m].mean()))
        s_vals[i] = (b_i - a_i) / max(a_i, b_i, 1e-12)
    return float(s_vals.mean())


# ---------------- Hierarchical clustering ----------------

def _is_leaf(node: dict) -> bool:
    return node["type"] == "leaf"


def hierarchical_cluster(X: np.ndarray,
                          samples: list,
                          *,
                          depth: int = 0,
                          max_depth: int = 4,
                          min_size: int = 40,
                          max_k: int = 4,
                          sil_threshold: float = 0.05) -> dict:
    """Recursively k-means cluster X. Stop when small or low silhouette.

    samples: list of length N (one per row of X) for sample tracking.
    """
    N = X.shape[0]
    if N < 2 * min_size or depth >= max_depth:
        return {"type": "leaf", "size": N, "samples": samples, "X": X}
    # Try K=2..max_k, pick best silhouette
    best = None
    for k in range(2, min(max_k + 1, max(2, N // min_size + 1))):
        labels, centers, inertia = kmeans(X, k, seed=42)
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score_np(X, labels)
        if best is None or sil > best[0]:
            best = (sil, k, labels, centers)
    if best is None or best[0] < sil_threshold:
        return {"type": "leaf", "size": N, "samples": samples, "X": X}
    sil, k, labels, centers = best
    children = []
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        sub_samples = [samples[i] for i in range(N) if mask[i]]
        sub_X = X[mask]
        children.append(hierarchical_cluster(
            sub_X, sub_samples,
            depth=depth + 1,
            max_depth=max_depth,
            min_size=min_size,
            max_k=max_k,
            sil_threshold=sil_threshold,
        ))
    if len(children) <= 1:
        return {"type": "leaf", "size": N, "samples": samples, "X": X}
    return {"type": "internal", "k": k, "silhouette": float(sil),
            "centers": centers, "children": children}


# ---------------- Reporting ----------------

def _leaf_summary(node: dict, n_samples: int = 5) -> list[str]:
    """Pick N representative samples per leaf: take samples closest to the
    cluster's centroid in the embedding space."""
    X = node["X"]
    samples = node["samples"]
    centroid = X.mean(axis=0)
    dists = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
    order = np.argsort(dists)[:n_samples]
    return [samples[i]["nl"] for i in order]


def _count_leaves(node: dict) -> int:
    if _is_leaf(node):
        return 1
    return sum(_count_leaves(c) for c in node["children"])


def _node_size(node: dict) -> int:
    if _is_leaf(node):
        return node["size"]
    return sum(_node_size(c) for c in node["children"])


def print_tree(node: dict, depth: int = 0, path: str = "0", lines: list | None = None) -> None:
    indent = "  " * depth
    if lines is None:
        lines = []
    if _is_leaf(node):
        size = node["size"]
        line = f"{indent}LEAF [{path}] size={size}"
        lines.append(line)
        for nl in _leaf_summary(node):
            lines.append(f"{indent}  • {nl}")
    else:
        line = (f"{indent}NODE [{path}] k={node['k']} silhouette={node['silhouette']:.3f} "
                f"size={_node_size(node)}")
        lines.append(line)
        for i, c in enumerate(node["children"]):
            print_tree(c, depth + 1, f"{path}.{i}", lines)
    return lines


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".cache/gsm8k_steps_v80_train.jsonl",
                    help="Source JSONL with L2 step descriptions")
    ap.add_argument("--pythia", default=".cache/pythia-410m/model.safetensors")
    ap.add_argument("--out", default="/tmp/ib_clustering_tree.txt")
    ap.add_argument("--max-steps", type=int, default=20000,
                    help="Max total steps to cluster (across all OPs)")
    ap.add_argument("--min-size", type=int, default=40,
                    help="Don't split clusters smaller than this")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--max-k", type=int, default=4,
                    help="Max branching factor at each internal node")
    ap.add_argument("--sil-threshold", type=float, default=0.05)
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(f"[1/4] Loading Pythia embeddings from {args.pythia}...")
    embed_w = load_pythia_embed_numpy(args.pythia)
    print(f"  embed shape = {embed_w.shape}  dtype={embed_w.dtype}")
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    tok = load_tokenizer()

    print(f"\n[2/4] Extracting L2 steps from {args.src}...")
    all_steps = list(extract_steps(args.src))
    print(f"  found {len(all_steps)} steps")
    if len(all_steps) > args.max_steps:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_steps), size=args.max_steps, replace=False)
        all_steps = [all_steps[i] for i in idx]
        print(f"  subsampled to {len(all_steps)}")

    # Group by OP
    by_op = defaultdict(list)
    for s in all_steps:
        by_op[s["op"]].append(s)
    print(f"  per-OP counts:")
    for op in ("ADD", "SUB", "MUL", "DIV"):
        print(f"    {op}: {len(by_op[op])}")

    print(f"\n[3/4] Embedding step descriptions...")
    t1 = time.perf_counter()
    embeddings_by_op = {}
    for op, steps in by_op.items():
        if not steps:
            continue
        embs = np.zeros((len(steps), embed_w.shape[1]), dtype=np.float32)
        for i, s in enumerate(steps):
            embs[i] = embed_text(s["nl"], tok, embed_w)
        embeddings_by_op[op] = embs
        print(f"  {op}: {embs.shape}  ({time.perf_counter() - t1:.1f}s)")

    print(f"\n[4/4] Clustering (max_k={args.max_k}, max_depth={args.max_depth}, "
          f"min_size={args.min_size}, sil_threshold={args.sil_threshold})...")
    out_lines = []
    out_lines.append("=" * 70)
    out_lines.append("IB CLUSTERING TREE")
    out_lines.append("=" * 70)
    out_lines.append(f"source: {args.src}")
    out_lines.append(f"total steps: {len(all_steps)}")
    out_lines.append("")
    total_leaves = 0
    for op in ("ADD", "SUB", "MUL", "DIV"):
        if op not in embeddings_by_op:
            continue
        X = embeddings_by_op[op]
        steps = by_op[op]
        out_lines.append("")
        out_lines.append("=" * 60)
        out_lines.append(f"OP: {op}   N={len(steps)}")
        out_lines.append("=" * 60)
        tree = hierarchical_cluster(
            X, steps,
            max_depth=args.max_depth,
            min_size=args.min_size,
            max_k=args.max_k,
            sil_threshold=args.sil_threshold,
        )
        n_leaves = _count_leaves(tree)
        total_leaves += n_leaves
        out_lines.append(f"  leaves found: {n_leaves}")
        out_lines.append("")
        sub = []
        print_tree(tree, depth=0, path=op, lines=sub)
        out_lines.extend(sub)

    out_lines.append("")
    out_lines.append("=" * 70)
    out_lines.append(f"TOTAL LEAVES ACROSS ALL OPS: {total_leaves}")
    out_lines.append(f"(target was 12 for codebook alignment; we may need to tune)")
    out_lines.append("=" * 70)

    text = "\n".join(out_lines)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(f"\n[wrote {args.out}]   total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
