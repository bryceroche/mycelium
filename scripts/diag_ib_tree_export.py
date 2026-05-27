"""Export the IB clustering tree + centroids + per-depth cluster IDs as a
JSON manifest (+ npz with centroids), and run an assignment quality check.

Reuses the clustering machinery from diag_ib_clustering.py.

Outputs:
  .cache/ib_tree.json       — tree structure with per-leaf metadata
  .cache/ib_centroids.npz   — leaf_id → centroid (1024d)
  .cache/ib_per_depth_ids.json — per-depth cluster id (str) per leaf

Then runs an assignment quality check: samples 50 L2 descriptions, embeds them,
finds nearest leaf, walks up tree to print the cluster IDs at every depth.
"""
import os
import sys
import json
import argparse
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import the existing clustering script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_ib_clustering as ibc  # type: ignore


def collect_leaves_and_centroids(root: dict, op: str, path: str = "") -> tuple[list, dict]:
    """Walk the tree. Return:
      leaves: list of dicts {leaf_id, op, depth, size, centroid, samples}
      tree_meta: dict {leaf_id: {ancestor_at_depth_d: cluster_id_str}}
    """
    leaves = []
    tree_meta = {}

    def walk(node, depth, path):
        cluster_id = path  # use the path string as cluster_id at this depth
        if ibc._is_leaf(node):
            X = node["X"]
            samples = node["samples"]
            centroid = X.mean(axis=0).astype(np.float32)
            leaf_id = path
            leaves.append({
                "leaf_id": leaf_id,
                "op": op,
                "depth": depth,
                "size": int(node["size"]),
                "centroid": centroid,
                "sample_nls": [s["nl"] for s in samples[:5]],
            })
        else:
            for i, c in enumerate(node["children"]):
                walk(c, depth + 1, f"{path}.{i}" if path else f"{i}")

    walk(root, 0, op)
    return leaves


def per_depth_cluster_ids(leaf_id: str, max_depth: int) -> list[str]:
    """Walk DOWN the leaf path, yielding the cluster id at each depth (1..max).
    Depth 0 is 'math' (root). Depth 1 is the OP (e.g. 'DIV'). Deeper truncations
    of the leaf path give intermediate cluster ids.

    Example: leaf_id = 'DIV.0.2.1.0'
      depth 0: 'math'
      depth 1: 'DIV'
      depth 2: 'DIV.0'
      depth 3: 'DIV.0.2'
      depth 4: 'DIV.0.2.1'
      depth 5: 'DIV.0.2.1.0'
    """
    parts = leaf_id.split(".")
    out = ["math"]  # depth 0
    for d in range(1, len(parts) + 1):
        out.append(".".join(parts[:d]))
    while len(out) <= max_depth:
        out.append(out[-1])  # pad by repeating the leaf id
    return out[: max_depth + 1]


def assign_nearest(emb: np.ndarray, centroids: np.ndarray,
                    op_mask: np.ndarray | None = None) -> int:
    """Return index of nearest centroid (cosine similarity). If op_mask is
    provided (bool array over centroids), only search among masked centroids."""
    emb_n = emb / (np.linalg.norm(emb) + 1e-12)
    cent_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims = cent_n @ emb_n
    if op_mask is not None:
        sims = np.where(op_mask, sims, -np.inf)
    return int(np.argmax(sims))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".cache/gsm8k_steps_v80_train.jsonl")
    ap.add_argument("--pythia", default=".cache/pythia-410m/model.safetensors")
    ap.add_argument("--out-tree", default=".cache/ib_tree.json")
    ap.add_argument("--out-cent", default=".cache/ib_centroids.npz")
    ap.add_argument("--out-depths", default=".cache/ib_per_depth_ids.json")
    ap.add_argument("--out-summary", default="/tmp/ib_tree_assignment_check.txt")
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--min-size", type=int, default=150)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--sil-threshold", type=float, default=0.05)
    ap.add_argument("--sample-n", type=int, default=50,
                    help="Number of L2 descriptions to spot-check assignment on")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(f"[1] Loading Pythia embeddings...")
    embed_w = ibc.load_pythia_embed_numpy(args.pythia)
    tok = ibc.load_tokenizer()

    print(f"[2] Extracting L2 steps...")
    all_steps = list(ibc.extract_steps(args.src))
    if len(all_steps) > args.max_steps:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_steps), size=args.max_steps, replace=False)
        all_steps = [all_steps[i] for i in idx]
    print(f"  {len(all_steps)} steps")

    by_op = defaultdict(list)
    for s in all_steps:
        by_op[s["op"]].append(s)

    print(f"[3] Embedding + clustering per OP (min_size={args.min_size}, depth={args.max_depth})...")
    all_leaves = []  # list of leaf dicts
    for op in ("ADD", "SUB", "MUL", "DIV"):
        steps = by_op[op]
        if not steps:
            continue
        embs = np.zeros((len(steps), embed_w.shape[1]), dtype=np.float32)
        for i, s in enumerate(steps):
            embs[i] = ibc.embed_text(s["nl"], tok, embed_w)
        tree = ibc.hierarchical_cluster(
            embs, steps,
            max_depth=args.max_depth,
            min_size=args.min_size,
            max_k=args.max_k,
            sil_threshold=args.sil_threshold,
        )
        leaves = collect_leaves_and_centroids(tree, op)
        all_leaves.extend(leaves)
        print(f"  {op}: {len(leaves)} leaves")

    # Determine global max depth
    max_depth = max((l["leaf_id"].count(".") for l in all_leaves), default=1)
    max_depth_full = max_depth + 1   # depth 0 = math (root), so +1 levels above

    print(f"  global max depth: {max_depth_full} (root + {max_depth} levels)")

    # Save tree metadata
    tree_meta = {
        "n_leaves": len(all_leaves),
        "max_depth": max_depth_full,
        "leaves": [
            {
                "leaf_id": l["leaf_id"],
                "op": l["op"],
                "size": l["size"],
                "sample_nls": l["sample_nls"],
                "per_depth_ids": per_depth_cluster_ids(l["leaf_id"], max_depth_full),
            }
            for l in all_leaves
        ],
    }
    with open(args.out_tree, "w") as f:
        json.dump(tree_meta, f, indent=2)

    # Save centroids
    leaf_ids = [l["leaf_id"] for l in all_leaves]
    centroids = np.stack([l["centroid"] for l in all_leaves], axis=0)
    np.savez_compressed(args.out_cent, leaf_ids=np.array(leaf_ids), centroids=centroids)

    # Per-depth cluster ID lookup
    per_depth_map = {l["leaf_id"]: per_depth_cluster_ids(l["leaf_id"], max_depth_full)
                       for l in all_leaves}
    with open(args.out_depths, "w") as f:
        json.dump(per_depth_map, f, indent=2)

    print(f"\nSaved:")
    print(f"  {args.out_tree}  ({len(all_leaves)} leaves)")
    print(f"  {args.out_cent}  (centroids shape: {centroids.shape})")
    print(f"  {args.out_depths}")

    # ----------------- Per-depth distribution -----------------
    print(f"\n[4] Per-depth cluster vocabulary sizes:")
    for d in range(max_depth_full + 1):
        ids = {per_depth_map[l["leaf_id"]][d] for l in all_leaves}
        print(f"  depth {d}: {len(ids)} unique cluster ids")
        if d <= 2 and len(ids) <= 30:
            print(f"    ids: {sorted(ids)}")

    # ----------------- Assignment quality check -----------------
    print(f"\n[5] Assignment quality: sampling {args.sample_n} random L2 descriptions...")
    rng = np.random.default_rng(42)
    pick_idx = rng.choice(len(all_steps), size=args.sample_n, replace=False)

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    # Pre-compute per-OP centroid masks
    leaf_ops = np.array([l["op"] for l in all_leaves])
    op_masks = {op: leaf_ops == op for op in ("ADD", "SUB", "MUL", "DIV")}

    emit("\n=== Assignment quality check (UNCONSTRAINED vs OP-CONSTRAINED) ===")
    emit(f"{'NL action':<55} {'true':>4} {'free_leaf':<14} {'free_op':>3}  {'constr_leaf':<14}")
    emit("-" * 110)
    free_op_match = 0
    for i in pick_idx:
        step = all_steps[i]
        emb = ibc.embed_text(step["nl"], tok, embed_w)
        # Free (unconstrained) assignment
        free_idx = assign_nearest(emb, centroids)
        free_leaf = all_leaves[free_idx]
        free_match = "✓" if free_leaf["op"] == step["op"] else "✗"
        if free_leaf["op"] == step["op"]:
            free_op_match += 1
        # Constrained assignment (search only within the correct OP sub-tree)
        constr_idx = assign_nearest(emb, centroids, op_mask=op_masks[step["op"]])
        constr_leaf = all_leaves[constr_idx]
        emit(f"{step['nl'][:55]:<55} {step['op']:>4} {free_leaf['leaf_id']:<14} {free_match:>3}  {constr_leaf['leaf_id']:<14}")
    emit("")
    emit(f"Free OP match rate: {free_op_match}/{args.sample_n} = {free_op_match*100/args.sample_n:.1f}%")
    emit(f"OP-constrained: 100% (by construction)")
    emit("  → at training time, use OP-constrained assignment (OP comes from L2).")

    emit("")
    emit("=== Per-depth IDs for 5 sample assignments ===")
    for i in pick_idx[:5]:
        step = all_steps[i]
        emb = ibc.embed_text(step["nl"], tok, embed_w)
        idx = assign_nearest(emb, centroids)
        leaf = all_leaves[idx]
        depth_ids = per_depth_map[leaf["leaf_id"]]
        emit(f"\nNL: {step['nl']}  (true_op={step['op']})")
        emit(f"  leaf: {leaf['leaf_id']}")
        for d, did in enumerate(depth_ids):
            emit(f"    breath {d}: {did}")

    text = "\n".join(lines)
    with open(args.out_summary, "w") as f:
        f.write(text + "\n")

    print(f"\n[wrote {args.out_summary}]   total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
