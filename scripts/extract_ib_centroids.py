"""Agglomerative Information Bottleneck centroid extraction.

Collects internal reps from a champion model (e.g., v24c step 500), groups them
by ground-truth op label, then runs Slonim-Tishby agglomerative IB clustering.

The merge tree IS the hierarchy: at each step we merge the cluster pair whose
merge loses least info about op_label. Track I(T;Y) vs cluster count. Plateaus
in this curve reveal natural cluster levels — that's where the rep distribution
has discrete "kinds" of procedures.

Output:
- centroids at multiple plateau levels (4, 16, 64, ...)
- merge dendrogram (parent → children)
- info curve I(T;Y) vs N_clusters

Usage:
    DEV=PCI+AMD CKPT=/path/to/v24c.safetensors NUM_PROBLEMS=600 \\
        python scripts/extract_ib_centroids.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, encode_cycles, parse_int_answer
from mycelium.lookup_table import (
    eq_token_ids_for, find_eq_position, op_label_from_text,
)


def cast_fp32(model):
    def _c(o, a):
        t = getattr(o, a)
        if t.dtype == dtypes.half:
            setattr(o, a, t.cast(dtypes.float).contiguous().realize())
    _c(model.embed, "weight")
    _c(model, "embed_out")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _c(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _c(layer, a)


def collect_reps(model, tok, examples, n_loops, fixed_len, eq_ids, batch_size=16):
    """Collect (rep, op_label) for each cycle's = position across all examples."""
    reps_list = []
    labels_list = []
    Tensor.training = False
    for ex_idx, ex in enumerate(examples):
        cycle_encodings = encode_cycles(tok, ex)
        for cyc_idx, (ids_list, prefix_len, total_len) in enumerate(cycle_encodings):
            cycle_text = ex.gen_targets[cyc_idx] if cyc_idx < len(ex.gen_targets) else ""
            op_label = op_label_from_text(cycle_text)
            if op_label < 0 or op_label >= 4:
                continue
            ids = ids_list[:fixed_len]
            tokens_np = np.zeros((1, fixed_len), dtype=np.int32)
            tokens_np[0, :len(ids)] = ids
            tokens = Tensor(tokens_np, dtype=dtypes.int).realize()
            final, _, _ = model.breathe_with_lookup(tokens, n_loops)
            final_np = final.realize().numpy()[0]
            target_span = ids[prefix_len:]
            eq_offset = -1
            for i, t in enumerate(target_span):
                if t in eq_ids:
                    eq_offset = i
                    break
            if eq_offset < 0:
                continue
            eq_pos = prefix_len + eq_offset
            if eq_pos >= fixed_len:
                continue
            reps_list.append(final_np[eq_pos, :].astype(np.float32))
            labels_list.append(op_label)
        if (ex_idx + 1) % 50 == 0:
            print(f"  collected {len(reps_list)} reps from {ex_idx+1}/{len(examples)} examples", flush=True)
    return np.array(reps_list), np.array(labels_list, dtype=np.int32)


def js_divergence(p, q, eps=1e-12):
    """Jensen-Shannon divergence between two discrete probability distributions."""
    m = 0.5 * (p + q)
    def kl(a, b):
        a = np.maximum(a, eps); b = np.maximum(b, eps)
        return np.sum(a * np.log(a / b))
    return 0.5 * (kl(p, m) + kl(q, m))


def agglomerative_ib(reps, labels, n_classes):
    """Slonim-Tishby agglomerative information bottleneck.

    Returns:
        merge_order: list of (cluster_i, cluster_j, cost, n_remaining)
        info_curve: array of I(T; Y) at each n_remaining value
        cluster_history: dict mapping n_clusters → list of (cluster_indices) per group
    """
    n = len(reps)
    # Each cluster's marginal probability and label distribution
    # Initialize: each rep is its own cluster
    cluster_size = np.ones(n, dtype=np.float64)
    cluster_label_dist = np.zeros((n, n_classes), dtype=np.float64)
    for i, lbl in enumerate(labels):
        cluster_label_dist[i, lbl] = 1.0
    cluster_active = np.ones(n, dtype=bool)
    cluster_members = {i: [i] for i in range(n)}

    total_size = float(n)
    # Marginal label distribution
    p_y = np.bincount(labels, minlength=n_classes).astype(np.float64) / n

    def info_TY():
        """Current I(T; Y) given active clusters."""
        active_idx = np.where(cluster_active)[0]
        pi = cluster_size[active_idx] / total_size
        pyt = cluster_label_dist[active_idx]  # (n_active, n_classes), each row is p(y|t)
        # I(T;Y) = Σ_t π_t Σ_y p(y|t) log(p(y|t) / p(y))
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio = np.where(pyt > 0, np.log(pyt / np.maximum(p_y, 1e-12)), 0.0)
            return float(np.sum(pi[:, None] * pyt * log_ratio))

    merge_order = []
    info_curve = [(n, info_TY())]
    cluster_history = {n: [list(cluster_members[k]) for k in cluster_members if cluster_active[k]]}

    # Precompute initial similarity matrix to find candidate merges efficiently.
    # For 2000 reps this is ~16MB and feasible.
    print(f"  computing cosine similarities for {n} reps...", flush=True)
    reps_norm = reps / (np.linalg.norm(reps, axis=-1, keepdims=True) + 1e-9)
    sim_matrix = reps_norm @ reps_norm.T  # (n, n)
    np.fill_diagonal(sim_matrix, -np.inf)

    # IB merge cost: (π_i + π_j) * JS(p(y|i), p(y|j))
    # Greedy: at each step, find the pair with minimum merge cost
    # For efficiency, restrict to top-K nearest neighbors per cluster

    K_NEIGHBORS = 20  # candidate merges to consider per cluster

    def merge_cost(i, j):
        pi = cluster_size[i] / total_size
        pj = cluster_size[j] / total_size
        return (pi + pj) * js_divergence(cluster_label_dist[i], cluster_label_dist[j])

    print(f"  starting agglomerative merging (target n=1 from n={n})...", flush=True)
    t0 = time.perf_counter()
    next_milestone = n // 2
    while np.sum(cluster_active) > 1:
        # Find min-cost merge among neighbors of active clusters
        active_idx = np.where(cluster_active)[0]
        best_cost = np.inf
        best_pair = None
        for i in active_idx:
            # Top-K most similar active clusters as merge candidates
            sim_row = sim_matrix[i].copy()
            sim_row[~cluster_active] = -np.inf
            sim_row[i] = -np.inf
            top_k = np.argpartition(sim_row, -K_NEIGHBORS)[-K_NEIGHBORS:]
            for j in top_k:
                if not cluster_active[j] or j == i:
                    continue
                cost = merge_cost(i, j)
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        # Merge j into i
        new_size = cluster_size[i] + cluster_size[j]
        new_label_dist = (cluster_size[i] * cluster_label_dist[i] + cluster_size[j] * cluster_label_dist[j]) / new_size
        cluster_size[i] = new_size
        cluster_label_dist[i] = new_label_dist
        cluster_members[i] = cluster_members[i] + cluster_members[j]
        cluster_active[j] = False
        n_remaining = int(np.sum(cluster_active))
        merge_order.append((int(i), int(j), float(best_cost), n_remaining))
        info_curve.append((n_remaining, info_TY()))
        # Record cluster snapshots at log-spaced milestones
        if n_remaining in (4, 8, 16, 32, 64, 128) or n_remaining == next_milestone:
            cluster_history[n_remaining] = [list(cluster_members[k]) for k in cluster_members if cluster_active[k]]
            next_milestone = max(2, next_milestone // 2)
        if n_remaining % 100 == 0 or n_remaining < 30:
            elapsed = time.perf_counter() - t0
            print(f"    n={n_remaining:5d}  I(T;Y)={info_curve[-1][1]:.4f}  cost={best_cost:.4e}  ({elapsed:.1f}s)", flush=True)
    print(f"  done merging in {time.perf_counter()-t0:.1f}s", flush=True)
    return merge_order, info_curve, cluster_history


def compute_centroids(reps, cluster_members_list):
    """For each cluster (list of rep indices), compute the centroid."""
    centroids = []
    for members in cluster_members_list:
        if not members:
            centroids.append(np.zeros(reps.shape[1], dtype=np.float32))
        else:
            centroids.append(reps[members].mean(axis=0))
    return np.array(centroids)


def main():
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/l4_mixed_ckpts/l4_mixed_v24c_dual_notebook_step500.safetensors")
    num_problems = getenv("NUM_PROBLEMS", 600)  # ~1200 cycle-reps
    n_loops = getenv("LOOPS", 4)
    fixed_len = getenv("FIXED_LEN", 96)
    seed = getenv("SEED", 42)
    out_dir = getenv("OUT_DIR", ".cache/ib_centroids")
    space_digits = bool(getenv("SPACE_DIGITS", 1))

    cfg = Config()
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Agglomerative IB centroid extraction ===")
    print(f"  ckpt:        {ckpt}")
    print(f"  n_problems:  {num_problems}  (target ~{num_problems*2} cycle-reps)")
    print(f"  out_dir:     {out_dir}")
    print()

    print("loading model + ckpt...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd); cast_fp32(model); del sd
    state = safe_load(ckpt)
    model.load_state_dict(state, strict=False)
    Device[Device.DEFAULT].synchronize()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s")

    tok = load_tokenizer()
    eq_ids = eq_token_ids_for(tok)

    print(f"\ngenerating {num_problems} L4_MIXED problems...")
    examples = generate_math("L4_MIXED", num_problems + 100, seed=seed + 99, digit_spacing=space_digits)
    examples = examples[:num_problems]

    print(f"\ncollecting reps + labels...")
    t0 = time.perf_counter()
    reps, labels = collect_reps(model, tok, examples, n_loops, fixed_len, eq_ids)
    print(f"\nCollected {len(reps)} reps ({time.perf_counter()-t0:.1f}s)")
    op_counts = np.bincount(labels, minlength=4)
    print(f"Op distribution: + = {op_counts[0]}, - = {op_counts[1]}, * = {op_counts[2]}, / = {op_counts[3]}")

    print(f"\nRunning agglomerative IB clustering...")
    n_classes = 4
    merge_order, info_curve, cluster_history = agglomerative_ib(reps, labels, n_classes)

    # Save outputs
    info_arr = np.array(info_curve, dtype=np.float64)  # (n_steps, 2): (n, info)
    np.save(f"{out_dir}/info_curve.npy", info_arr)
    print(f"\nSaved info curve: {out_dir}/info_curve.npy  ({info_arr.shape})")

    # Save centroids at common plateaus
    for n_k in (4, 8, 16, 32, 64):
        if n_k in cluster_history:
            members = cluster_history[n_k]
            centroids = compute_centroids(reps, members)
            sizes = np.array([len(m) for m in members])
            np.save(f"{out_dir}/centroids_n{n_k}.npy", centroids)
            np.save(f"{out_dir}/sizes_n{n_k}.npy", sizes)
            # Per-cluster op purity
            for c_i, m in enumerate(members):
                lbl_dist = np.bincount(labels[m], minlength=n_classes)
                purity = lbl_dist.max() / lbl_dist.sum() if lbl_dist.sum() > 0 else 0
                top_op = ['+', '-', '*', '/'][int(lbl_dist.argmax())]
                if c_i < 5 or c_i >= len(members) - 5:
                    print(f"  n={n_k} cluster {c_i:3d}: size={len(m):4d}  top_op={top_op}  purity={purity:.3f}")
            print(f"Saved n={n_k} centroids: shape={centroids.shape}")

    # Show info curve summary at key points
    print(f"\nI(T;Y) at key cluster counts:")
    for n_k in (4, 8, 16, 32, 64, 128, 256, 512):
        match = info_arr[info_arr[:, 0].astype(int) == n_k]
        if len(match) > 0:
            print(f"  n={n_k:4d}: I(T;Y) = {match[0, 1]:.4f}")


if __name__ == "__main__":
    main()
