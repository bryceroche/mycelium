"""IB clustering verification and analysis script.

Reads the output of ib_cluster_factor_graphs.py and produces:
  - Per-cluster sample inspection (5 texts per leaf)
  - Cluster purity by operation type
  - Intra/inter cluster distance ratio
  - Comparison against prior .cache/ib_tree.json if both exist
  - Saves analysis to .cache/ib_clustering_report_gsm8k.md

Usage:
  .venv/bin/python scripts/ib_verify_clustering.py [--partial]
"""
import os
import sys
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity."""
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - (an @ bn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="Read *_partial.* files from ib_cluster_factor_graphs.py")
    ap.add_argument("--tree", default=None)
    ap.add_argument("--centroids", default=None)
    ap.add_argument("--assign", default=None)
    ap.add_argument("--prior-tree", default=None,
                    help="Prior ib_tree.json for structural comparison")
    ap.add_argument("--out", default=None,
                    help="Output markdown report path")
    args = ap.parse_args()

    cache_dir = os.path.join(_PROJECT_ROOT, ".cache")
    suffix = "_partial" if args.partial else ""

    tree_path = args.tree or os.path.join(cache_dir, f"ib_tree_gsm8k{suffix}.json")
    cent_path = args.centroids or os.path.join(cache_dir, f"ib_centroids_gsm8k{suffix}.npz")
    assign_path = args.assign or os.path.join(cache_dir, f"var_descriptions_to_leaf{suffix}.jsonl")
    prior_tree_path = args.prior_tree or os.path.join(cache_dir, "ib_tree.json")
    out_path = args.out or os.path.join(cache_dir, f"ib_clustering_report_gsm8k{suffix}.md")

    # ── Load artifacts ─────────────────────────────────────────────────────
    if not os.path.exists(tree_path):
        print(f"[error] tree file not found: {tree_path}")
        print("  Run ib_cluster_factor_graphs.py first.")
        sys.exit(1)

    with open(tree_path) as f:
        tree = json.load(f)
    leaves = tree["leaves"]
    n_leaves = tree["n_leaves"]

    d = np.load(cent_path)
    centroids = d["centroids"]   # (N_leaves, 1024)
    leaf_ids = d["leaf_ids"].tolist()

    leaf_id_to_idx = {lid: i for i, lid in enumerate(leaf_ids)}

    # Load assignments
    assignments = []
    if os.path.exists(assign_path):
        with open(assign_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    assignments.append(json.loads(line))
    else:
        print(f"  [warn] assignment file not found: {assign_path}")

    # ── Collect report lines ───────────────────────────────────────────────
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# IB Clustering Report — GSM8K Factor Graphs{' (PARTIAL)' if args.partial else ''}")
    emit()
    emit(f"**Source:** `{tree.get('source', '?')}`  ")
    emit(f"**Records:** {tree.get('n_records', '?')}  ")
    emit(f"**Total leaves:** {n_leaves}  ")
    emit(f"**Centroids shape:** {centroids.shape}  ")
    emit()

    # ── 1. Per-leaf sample inspection ─────────────────────────────────────
    emit("## 1. Per-Leaf Sample Inspection")
    emit()
    for op in ("ADD", "SUB", "MUL", "DIV"):
        op_leaves = [l for l in leaves if l["op"] == op]
        if not op_leaves:
            continue
        emit(f"### OP: {op}  ({len(op_leaves)} leaves)")
        emit()
        for leaf in op_leaves:
            emit(f"**[{leaf['leaf_id']}]**  size={leaf['size']}")
            for s in leaf.get("sample_texts", []):
                emit(f"  - {s}")
            emit()

    # ── 2. Cluster purity by operation ────────────────────────────────────
    emit("## 2. Cluster Purity by Operation Type")
    emit()
    if assignments:
        # For assignments with a known op, check if the assigned leaf's op matches
        leaf_op_lookup = {l["leaf_id"]: l["op"] for l in leaves}
        match = 0
        total_labeled = 0
        op_confusion = defaultdict(Counter)  # true_op -> predicted_op
        for a in assignments:
            true_op = a.get("op")
            if true_op is None:
                continue
            assigned_leaf = a.get("leaf_id", "")
            assigned_op = leaf_op_lookup.get(assigned_leaf, "?")
            op_confusion[true_op][assigned_op] += 1
            total_labeled += 1
            if true_op == assigned_op:
                match += 1
        if total_labeled > 0:
            purity = 100.0 * match / total_labeled
            emit(f"**Op-constrained assignment purity: {purity:.1f}%** ({match}/{total_labeled})")
            emit("(Should be 100% for labeled vars — measures cluster coherence via assignment consistency)")
            emit()
            emit("Confusion matrix (rows=true_op, cols=assigned_leaf_op):")
            emit()
            all_ops = sorted(op_confusion.keys())
            header = "| true \\ assigned | " + " | ".join(all_ops) + " |"
            sep = "|" + "---|" * (len(all_ops) + 1)
            emit(header)
            emit(sep)
            for true_op in all_ops:
                row = " | ".join(str(op_confusion[true_op].get(o, 0)) for o in all_ops)
                emit(f"| {true_op} | {row} |")
            emit()
        else:
            emit("No labeled assignments found for purity computation.")
            emit()
    else:
        emit("No assignment file found.")
        emit()

    # ── 3. Intra / inter cluster distance ratio ────────────────────────────
    emit("## 3. Embedding-Space Distances")
    emit()
    if len(centroids) > 1:
        # Normalize centroids
        cent_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
        # Inter-cluster distances (cosine)
        sims = cent_n @ cent_n.T  # (N, N)
        N = len(centroids)
        # Upper triangle only (exclude diagonal)
        inter_sims = []
        intra_sims_per_op = {}
        for i in range(N):
            for j in range(i + 1, N):
                inter_sims.append(float(sims[i, j]))

        inter_mean = np.mean(inter_sims)
        inter_std = np.std(inter_sims)
        emit(f"**Inter-cluster cosine similarity:** mean={inter_mean:.3f}  std={inter_std:.3f}")
        emit(f"  (lower = more separated clusters; random ~0.0 for unit vectors in 1024d)")
        emit()

        # Within-op inter-cluster distances
        for op in ("ADD", "SUB", "MUL", "DIV"):
            op_idx = [i for i, l in enumerate(leaves) if l["op"] == op]
            if len(op_idx) < 2:
                continue
            within_sims = []
            for ii, i in enumerate(op_idx):
                for j in op_idx[ii + 1:]:
                    within_sims.append(float(sims[i, j]))
            emit(f"  {op} within-op inter-cluster sim: mean={np.mean(within_sims):.3f}  "
                 f"std={np.std(within_sims):.3f}  n_pairs={len(within_sims)}")
        emit()

        # Across-op distances
        cross_sims = []
        for i in range(N):
            for j in range(i + 1, N):
                if leaves[i]["op"] != leaves[j]["op"]:
                    cross_sims.append(float(sims[i, j]))
        if cross_sims:
            emit(f"  Cross-op similarity: mean={np.mean(cross_sims):.3f}  std={np.std(cross_sims):.3f}")
            emit(f"  (should be < within-op if op families are separable)")
        emit()

    # ── 4. Comparison with prior ib_tree.json ──────────────────────────────
    emit("## 4. Comparison with Prior IB Tree (L2 step descriptions)")
    emit()
    if os.path.exists(prior_tree_path):
        with open(prior_tree_path) as f:
            prior = json.load(f)
        prior_leaves = prior["leaves"]
        emit(f"Prior tree: **{prior['n_leaves']} leaves** (from `{prior_tree_path}`)")
        emit(f"New tree:   **{n_leaves} leaves**")
        emit()
        # Per-op leaf count comparison
        emit("| OP | Prior leaves | New leaves |")
        emit("|---|---|---|")
        for op in ("ADD", "SUB", "MUL", "DIV"):
            prior_count = sum(1 for l in prior_leaves if l.get("op", l.get("leaf_id", "").split(".")[0]) == op)
            new_count = sum(1 for l in leaves if l["op"] == op)
            emit(f"| {op} | {prior_count} | {new_count} |")
        emit()
        emit("Prior used **L2 step descriptions** (action-oriented NL: 'Calculate total income')")
        emit("New uses **variable_descriptions** (entity-oriented NL: 'total income', 'price per jar')")
        emit("Structural similarity: both cluster within-op; leaf counts may differ due to data distribution.")
        emit()
    else:
        emit(f"Prior tree not found at `{prior_tree_path}` — skipping comparison.")
        emit()

    # ── 5. Leaf size distribution ──────────────────────────────────────────
    emit("## 5. Leaf Size Distribution")
    emit()
    sizes = [l["size"] for l in leaves]
    emit(f"Min={min(sizes)}  Max={max(sizes)}  Mean={np.mean(sizes):.0f}  "
         f"Median={np.median(sizes):.0f}")
    small = sum(1 for s in sizes if s < 30)
    emit(f"Leaves with size < 30: {small} (potential noise clusters)")
    emit()
    # Per-op stats
    emit("| OP | Leaves | Min | Max | Mean |")
    emit("|---|---|---|---|---|")
    for op in ("ADD", "SUB", "MUL", "DIV"):
        op_sizes = [l["size"] for l in leaves if l["op"] == op]
        if not op_sizes:
            continue
        emit(f"| {op} | {len(op_sizes)} | {min(op_sizes)} | {max(op_sizes)} | {np.mean(op_sizes):.0f} |")
    emit()

    # ── 6. Summary verdict ────────────────────────────────────────────────
    emit("## 6. Summary Verdict")
    emit()
    emit(f"- **Leaves found:** {n_leaves} (target ~32)")
    if n_leaves >= 24 and n_leaves <= 40:
        emit("- Leaf count is in the expected 24–40 range.")
    elif n_leaves < 24:
        emit(f"- Leaf count is below target. Consider reducing `--min-size` "
             f"(currently based on {tree.get('n_records','?')} records).")
    else:
        emit("- Leaf count exceeds target. Consider increasing `--min-size` or reducing `--max-depth`.")

    if small > 3:
        emit(f"- {small} small leaves (<30 items) present; may indicate noise. "
             f"Consider merging or raising `--min-size`.")

    emit()

    # ── Write report ──────────────────────────────────────────────────────
    report_text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(report_text + "\n")
    print(f"\n[report saved → {out_path}]")


if __name__ == "__main__":
    main()
