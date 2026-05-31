"""IB clustering pipeline for GSM8K factor graphs.

Clusters variable_descriptions from labeled factor graph records using
Pythia token embeddings + hierarchical K-means (same approach as
diag_ib_clustering.py / diag_ib_tree_export.py, adapted for the factor
graph schema).

Variable_descriptions for OUTPUT variables (those produced by a factor) are
labeled with the factor's operation type (add/sub/mul/div).  Input-only
variables (observed, no producing factor) are left unclustered or optionally
included in a global cluster pass.

Outputs:
  .cache/ib_tree_gsm8k.json          — tree structure + sample representatives
  .cache/ib_centroids_gsm8k.npz      — (N_leaves, 1024) float32 centroids
  .cache/var_descriptions_to_leaf.jsonl — per-variable leaf assignment

Usage (partial data, produces *_partial.* outputs):
  .venv/bin/python scripts/ib_cluster_factor_graphs.py --partial

Usage (full data):
  .venv/bin/python scripts/ib_cluster_factor_graphs.py
"""
import os
import sys
import json
import argparse
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad.nn.state import safe_load

from mycelium.data import load_tokenizer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_factor_graphs(jsonl_path: str):
    """Yield parsed factor graph records. Skip malformed lines."""
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] bad JSON at line {lineno}, skipping")
                continue
            if "var_descriptions" not in r:
                continue
            yield r


def extract_var_entries(records):
    """Yield {text, op, pidx, var_idx} for each variable description.

    op is the type of the factor that PRODUCES this variable (ADD/SUB/MUL/DIV),
    or None for input-only variables that have no producing factor.
    """
    for pidx, r in enumerate(records):
        descs = r["var_descriptions"]
        factor_types = r.get("factor_types", [])
        factor_args = r.get("factor_args", [])

        # Build: output_var_idx -> op
        out_to_op = {}
        for ft, fa in zip(factor_types, factor_args):
            out_var = fa[2]
            out_to_op[out_var] = ft.upper()  # normalise to uppercase

        for vi, text in enumerate(descs):
            if not text or not text.strip():
                continue
            op = out_to_op.get(vi, None)  # None = input-only / observed
            yield {
                "text": text.strip(),
                "op": op,
                "pidx": pidx,
                "var_idx": vi,
                "gsm8k_idx": r.get("gsm8k_idx", -1),
            }


# ─── Pythia embedding ────────────────────────────────────────────────────────

def load_pythia_embed_numpy(path: str) -> np.ndarray:
    """Load Pythia-410M embed_in weights as (vocab, hidden) float32."""
    sd = safe_load(path)
    for k in ("gpt_neox.embed_in.weight", "embed_in.weight", "embed.weight"):
        if k in sd:
            return sd[k].numpy().astype(np.float32)
    raise KeyError(f"Embed weight not found. Keys: {sorted(sd.keys())[:20]}")


def embed_text(text: str, tok, embed_w: np.ndarray) -> np.ndarray:
    """Mean-pool of token embeddings → (hidden,) float32."""
    ids = tok.encode(text).ids
    if not ids:
        return np.zeros(embed_w.shape[1], dtype=np.float32)
    return embed_w[ids].mean(axis=0)


# ─── Numpy K-means + silhouette ──────────────────────────────────────────────

def kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    N, D = X.shape
    centers = np.empty((k, D), dtype=X.dtype)
    centers[0] = X[rng.integers(0, N)]
    closest_sq = np.sum((X - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = closest_sq / max(closest_sq.sum(), 1e-12)
        centers[i] = X[rng.choice(N, p=probs)]
        new_sq = np.sum((X - centers[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return centers


def kmeans(X: np.ndarray, k: int, max_iter: int = 50, seed: int = 42):
    N = X.shape[0]
    if k >= N:
        return np.arange(N), X.copy(), 0.0
    rng = np.random.default_rng(seed)
    centers = kmeans_pp_init(X, k, rng)
    for _ in range(max_iter):
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(d2, axis=1)
        new_centers = np.zeros_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                new_centers[c] = X[mask].mean(axis=0)
            else:
                new_centers[c] = X[np.argmax(np.min(d2, axis=1))]
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
    inertia = float(np.sum((X - centers[labels]) ** 2))
    return labels, centers, inertia


def silhouette_score_np(X: np.ndarray, labels: np.ndarray, sample_max: int = 1500) -> float:
    N = X.shape[0]
    if N > sample_max:
        idx = np.random.default_rng(0).choice(N, size=sample_max, replace=False)
        X, labels = X[idx], labels[idx]
        N = sample_max
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0
    dists = np.sqrt(np.maximum(
        np.sum(X**2, axis=1, keepdims=True)
        + np.sum(X**2, axis=1)[None, :]
        - 2.0 * X @ X.T, 0.0))
    s_vals = np.zeros(N, dtype=np.float32)
    for i in range(N):
        own = labels[i]
        own_mask = labels == own
        own_mask[i] = False
        if own_mask.sum() == 0:
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


# ─── Hierarchical clustering ─────────────────────────────────────────────────

def hierarchical_cluster(X: np.ndarray, samples: list, *,
                          depth: int = 0, max_depth: int = 3,
                          min_size: int = 30, max_k: int = 3,
                          sil_threshold: float = 0.05) -> dict:
    N = X.shape[0]
    if N < 2 * min_size or depth >= max_depth:
        return {"type": "leaf", "size": N, "samples": samples, "X": X}
    best = None
    for k in range(2, min(max_k + 1, max(2, N // min_size + 1))):
        labels, centers, _ = kmeans(X, k, seed=42)
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score_np(X, labels)
        if best is None or sil > best[0]:
            best = (sil, k, labels, centers)
    if best is None or best[0] < sil_threshold:
        return {"type": "leaf", "size": N, "samples": samples, "X": X}
    sil, k, labels, _ = best
    children = []
    for c in range(k):
        mask = labels == c
        if mask.sum() == 0:
            continue
        sub_samples = [samples[i] for i in range(N) if mask[i]]
        child = hierarchical_cluster(
            X[mask], sub_samples, depth=depth + 1,
            max_depth=max_depth, min_size=min_size,
            max_k=max_k, sil_threshold=sil_threshold)
        children.append(child)
    if len(children) <= 1:
        return {"type": "leaf", "size": N, "samples": samples, "X": X}
    return {"type": "internal", "k": k, "silhouette": float(sil), "children": children}


def is_leaf(node: dict) -> bool:
    return node["type"] == "leaf"


def count_leaves(node: dict) -> int:
    return 1 if is_leaf(node) else sum(count_leaves(c) for c in node["children"])


def collect_leaves(root: dict, op: str) -> list:
    """Walk tree, return list of leaf dicts with centroid, samples, leaf_id."""
    leaves = []

    def walk(node, path):
        if is_leaf(node):
            X = node["X"]
            samples = node["samples"]
            centroid = X.mean(axis=0).astype(np.float32)
            # Pick 5 samples closest to centroid
            dists = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            order = np.argsort(dists)[:5]
            leaves.append({
                "leaf_id": path,
                "op": op,
                "size": int(node["size"]),
                "centroid": centroid,
                "sample_texts": [samples[i]["text"] for i in order],
                "X": X,
                "samples": samples,
            })
        else:
            for i, c in enumerate(node["children"]):
                walk(c, f"{path}.{i}")

    walk(root, op)
    return leaves


def per_depth_ids(leaf_id: str, max_depth: int) -> list:
    """Ancestry chain: ['math', op, op.0, op.0.1, ...]."""
    parts = leaf_id.split(".")
    out = ["math"]
    for d in range(1, len(parts) + 1):
        out.append(".".join(parts[:d]))
    while len(out) <= max_depth:
        out.append(out[-1])
    return out[:max_depth + 1]


# ─── Assignment ──────────────────────────────────────────────────────────────

def assign_nearest(emb: np.ndarray, centroids: np.ndarray,
                   op_mask: np.ndarray | None = None) -> int:
    """Nearest centroid by cosine similarity."""
    emb_n = emb / (np.linalg.norm(emb) + 1e-12)
    cent_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims = cent_n @ emb_n
    if op_mask is not None:
        sims = np.where(op_mask, sims, -np.inf)
    return int(np.argmax(sims))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".cache/gsm8k_factor_graphs_train.jsonl")
    ap.add_argument("--pythia", default=".cache/pythia-410m/model.safetensors")
    ap.add_argument("--partial", action="store_true",
                    help="Save to *_partial.* outputs (for runs on incomplete data)")
    ap.add_argument("--out-tree", default=None)
    ap.add_argument("--out-cent", default=None)
    ap.add_argument("--out-assign", default=None)
    ap.add_argument("--min-size", type=int, default=30,
                    help="Min cluster size before stopping splits")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-k", type=int, default=3,
                    help="Max branching factor per internal node")
    ap.add_argument("--sil-threshold", type=float, default=0.05)
    ap.add_argument("--max-vars", type=int, default=0,
                    help="Cap total vars per op (0=no cap)")
    args = ap.parse_args()

    suffix = "_partial" if args.partial else ""
    cache_dir = os.path.join(_PROJECT_ROOT, ".cache")
    out_tree = args.out_tree or os.path.join(cache_dir, f"ib_tree_gsm8k{suffix}.json")
    out_cent = args.out_cent or os.path.join(cache_dir, f"ib_centroids_gsm8k{suffix}.npz")
    out_assign = args.out_assign or os.path.join(cache_dir, f"var_descriptions_to_leaf{suffix}.jsonl")

    src = os.path.join(_PROJECT_ROOT, args.src) if not os.path.isabs(args.src) else args.src
    pythia = os.path.join(_PROJECT_ROOT, args.pythia) if not os.path.isabs(args.pythia) else args.pythia

    t0 = time.perf_counter()

    # ── 1. Load Pythia embeddings ──────────────────────────────────────────
    print(f"[1/5] Loading Pythia embeddings from {pythia}...")
    embed_w = load_pythia_embed_numpy(pythia)
    tok = load_tokenizer()
    print(f"  embed shape = {embed_w.shape}  ({time.perf_counter()-t0:.1f}s)")

    # ── 2. Load + extract variable descriptions ────────────────────────────
    print(f"\n[2/5] Loading factor graphs from {src}...")
    records = list(load_factor_graphs(src))
    print(f"  {len(records)} records loaded")
    all_vars = list(extract_var_entries(records))
    print(f"  {len(all_vars)} variable descriptions extracted")

    # Group by op (skip None = input-only)
    by_op = defaultdict(list)
    for v in all_vars:
        if v["op"] is not None:
            by_op[v["op"]].append(v)
    input_only = [v for v in all_vars if v["op"] is None]
    print(f"  per-op counts:")
    for op in ("ADD", "SUB", "MUL", "DIV"):
        print(f"    {op}: {len(by_op[op])}")
    print(f"  input-only (no op label): {len(input_only)}")

    # ── 3. Embed per-op ────────────────────────────────────────────────────
    print(f"\n[3/5] Embedding variable descriptions...")
    t1 = time.perf_counter()
    embs_by_op = {}
    for op in ("ADD", "SUB", "MUL", "DIV"):
        vars_list = by_op[op]
        if not vars_list:
            continue
        if args.max_vars > 0 and len(vars_list) > args.max_vars:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(vars_list), size=args.max_vars, replace=False)
            vars_list = [vars_list[i] for i in idx]
            by_op[op] = vars_list
        embs = np.zeros((len(vars_list), embed_w.shape[1]), dtype=np.float32)
        for i, v in enumerate(vars_list):
            embs[i] = embed_text(v["text"], tok, embed_w)
        embs_by_op[op] = embs
        print(f"  {op}: {embs.shape}  ({time.perf_counter()-t1:.1f}s)")

    # ── 4. Hierarchical clustering per op ─────────────────────────────────
    print(f"\n[4/5] Clustering (max_k={args.max_k}, max_depth={args.max_depth}, "
          f"min_size={args.min_size}, sil_threshold={args.sil_threshold})...")
    all_leaves = []
    for op in ("ADD", "SUB", "MUL", "DIV"):
        if op not in embs_by_op:
            continue
        X = embs_by_op[op]
        vars_list = by_op[op]
        tree = hierarchical_cluster(
            X, vars_list,
            max_depth=args.max_depth, min_size=args.min_size,
            max_k=args.max_k, sil_threshold=args.sil_threshold)
        leaves = collect_leaves(tree, op)
        all_leaves.extend(leaves)
        print(f"  {op}: {len(leaves)} leaves")

    n_leaves = len(all_leaves)
    print(f"  → TOTAL leaves: {n_leaves}")

    if n_leaves == 0:
        print("[ERROR] No leaves produced. Check data size and min_size parameter.")
        sys.exit(1)

    max_depth_full = max((l["leaf_id"].count(".") for l in all_leaves), default=1) + 1

    # ── 5. Save outputs ────────────────────────────────────────────────────
    print(f"\n[5/5] Saving outputs...")

    # 5a. Tree JSON
    tree_out = {
        "n_leaves": n_leaves,
        "max_depth": max_depth_full,
        "source": src,
        "n_records": len(records),
        "leaves": [
            {
                "leaf_id": l["leaf_id"],
                "op": l["op"],
                "size": l["size"],
                "sample_texts": l["sample_texts"],
                "per_depth_ids": per_depth_ids(l["leaf_id"], max_depth_full),
            }
            for l in all_leaves
        ],
    }
    with open(out_tree, "w") as f:
        json.dump(tree_out, f, indent=2)
    print(f"  tree → {out_tree}")

    # 5b. Centroids NPZ
    leaf_ids = np.array([l["leaf_id"] for l in all_leaves])
    centroids = np.stack([l["centroid"] for l in all_leaves], axis=0)
    np.savez_compressed(out_cent, leaf_ids=leaf_ids, centroids=centroids)
    print(f"  centroids → {out_cent}  shape={centroids.shape}")

    # 5c. Per-variable assignment JSONL
    # Build per-op centroid masks
    leaf_ops_arr = np.array([l["op"] for l in all_leaves])
    op_masks = {op: leaf_ops_arr == op for op in ("ADD", "SUB", "MUL", "DIV")}

    n_written = 0
    with open(out_assign, "w") as fout:
        for v in all_vars:
            emb = embed_text(v["text"], tok, embed_w)
            if v["op"] is not None:
                mask = op_masks.get(v["op"])
                if mask is not None and mask.any():
                    idx = assign_nearest(emb, centroids, op_mask=mask)
                else:
                    idx = assign_nearest(emb, centroids)
                leaf_id = all_leaves[idx]["leaf_id"]
            else:
                # Input-only: unconstrained nearest-leaf
                idx = assign_nearest(emb, centroids)
                leaf_id = all_leaves[idx]["leaf_id"]
            rec = {
                "text": v["text"],
                "op": v["op"],
                "leaf_id": leaf_id,
                "pidx": v["pidx"],
                "var_idx": v["var_idx"],
                "gsm8k_idx": v["gsm8k_idx"],
            }
            fout.write(json.dumps(rec) + "\n")
            n_written += 1
    print(f"  assignments → {out_assign}  ({n_written} rows)")

    # Print tree summary
    print(f"\n--- Cluster tree summary ---")
    for op in ("ADD", "SUB", "MUL", "DIV"):
        op_leaves = [l for l in all_leaves if l["op"] == op]
        if not op_leaves:
            continue
        print(f"\n{op}  ({len(op_leaves)} leaves)")
        for leaf in op_leaves:
            print(f"  [{leaf['leaf_id']}] size={leaf['size']}")
            for s in leaf["sample_texts"]:
                print(f"      • {s}")

    print(f"\nTotal leaves: {n_leaves}  (target ~32)")
    print(f"Total time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
