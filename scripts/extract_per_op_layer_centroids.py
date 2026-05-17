"""Per-(op, layer) centroid extraction.

Captures rep at each of the 4 phase-specific layer outputs (within the last
breath) for each cycle of each L4_MIXED problem. Each rep tagged with
(op_label, layer_idx). 4 ops × 4 layers = 16 natural groups.

Then runs agglomerative IB clustering on this richer (op, layer) tagged data.
Expects natural plateau at N=16 (one per op-layer combo) or coarser if some
layer-stages are op-agnostic.

Output:
- centroids_oplayer_n{N}.npy at multiple plateau levels
- info_curve.npy
- per-cluster (op, layer) purity stats

Usage:
    DEV=PCI+AMD CKPT=/path/to/v24c.safetensors NUM_PROBLEMS=600 \\
        python scripts/extract_per_op_layer_centroids.py
"""
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, encode_cycles
from mycelium.lookup_table import (
    eq_token_ids_for, op_label_from_text,
)
# Import the breathe_once helpers we need
from mycelium.breathing import (
    _sine_temp_baseline, BREATH_TIME_EMBED, PER_HEAD_PITCH,
    LAYER_PITCH_TARGET, CONSTANT_RADIUS, BREATH_NORM_OSC,
    PER_BREATH_TEMP, _per_layer_temp_within_breath,
    _per_layer_norm_scale_within_breath,
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


def run_breath_capture_layers(model, x, loop_idx, n_phases):
    """Replicate breathe_once but capture rep AFTER each of the 4 phase layers.
    Returns list of 4 tensors (per-layer outputs)."""
    alpha = model.block.rope._alpha_at(loop_idx, x.dtype)
    if BREATH_TIME_EMBED:
        x = x + model.block.breath_embed[loop_idx].reshape(1, 1, -1).cast(x.dtype)
    ac_base, asn_base = alpha
    layer_outputs = []
    for layer_idx, layer in enumerate(model.block.layers):
        if PER_HEAD_PITCH and layer_idx > 0:
            cos_o = model.block.per_head_pitch_cos[layer_idx].cast(x.dtype)
            sin_o = model.block.per_head_pitch_sin[layer_idx].cast(x.dtype)
            ac_layer = ac_base * cos_o - asn_base * sin_o
            asn_layer = ac_base * sin_o + asn_base * cos_o
            layer_alpha = (ac_layer, asn_layer)
        else:
            layer_alpha = alpha
        layer_temp = 1.0  # for extraction we don't worry about temp
        x = layer(x, loop_idx, temp_mult=layer_temp, alpha=layer_alpha)
        layer_outputs.append(x)
    return layer_outputs


def collect_per_layer_reps(model, tok, examples, n_loops, fixed_len, eq_ids):
    """For each cycle of each example, capture rep AT each phase-layer of the
    LAST breath (most informative). Returns reps shape (N, hidden) + tags
    (N, 2) of (op, layer)."""
    reps_list = []
    tag_list = []
    Tensor.training = False
    cfg = model.cfg
    n_phases = cfg.n_phases
    handoff = None

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

            # Manually run the breath loop, capturing per-layer states at the last breath.
            x = model.embed(tokens).cast(dtypes.half)
            # Run all breaths up to the last
            for l in range(n_loops - 1):
                x = model.block.breathe_once(x, l, temp_mult=_sine_temp_baseline(l, n_loops))
            # Capture per-layer states for the FINAL breath
            layer_outputs = run_breath_capture_layers(model, x, n_loops - 1, n_phases)
            # For each layer output, find the rep at "=" position
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
            for layer_idx, layer_out in enumerate(layer_outputs):
                layer_np = layer_out.realize().numpy()[0]   # (T, hidden)
                rep = layer_np[eq_pos, :].astype(np.float32)
                reps_list.append(rep)
                tag_list.append((op_label, layer_idx))
        if (ex_idx + 1) % 25 == 0:
            print(f"  collected {len(reps_list)} reps from {ex_idx+1}/{len(examples)} examples", flush=True)
    return np.array(reps_list), np.array(tag_list, dtype=np.int32)


def js_divergence(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    def kl(a, b):
        a = np.maximum(a, eps); b = np.maximum(b, eps)
        return np.sum(a * np.log(a / b))
    return 0.5 * (kl(p, m) + kl(q, m))


def agglomerative_ib(reps, labels, n_classes):
    """Same as before — agglomerative IB clustering."""
    n = len(reps)
    cluster_size = np.ones(n, dtype=np.float64)
    cluster_label_dist = np.zeros((n, n_classes), dtype=np.float64)
    for i, lbl in enumerate(labels):
        cluster_label_dist[i, lbl] = 1.0
    cluster_active = np.ones(n, dtype=bool)
    cluster_members = {i: [i] for i in range(n)}
    total_size = float(n)
    p_y = np.bincount(labels, minlength=n_classes).astype(np.float64) / n

    def info_TY():
        active_idx = np.where(cluster_active)[0]
        pi = cluster_size[active_idx] / total_size
        pyt = cluster_label_dist[active_idx]
        with np.errstate(divide='ignore', invalid='ignore'):
            log_ratio = np.where(pyt > 0, np.log(pyt / np.maximum(p_y, 1e-12)), 0.0)
            return float(np.sum(pi[:, None] * pyt * log_ratio))

    merge_order = []
    info_curve = [(n, info_TY())]
    cluster_history = {n: [list(cluster_members[k]) for k in cluster_members if cluster_active[k]]}

    reps_norm = reps / (np.linalg.norm(reps, axis=-1, keepdims=True) + 1e-9)
    sim_matrix = reps_norm @ reps_norm.T
    np.fill_diagonal(sim_matrix, -np.inf)

    K_NEIGHBORS = 20

    def merge_cost(i, j):
        pi = cluster_size[i] / total_size
        pj = cluster_size[j] / total_size
        return (pi + pj) * js_divergence(cluster_label_dist[i], cluster_label_dist[j])

    t0 = time.perf_counter()
    next_milestone = n // 2
    while np.sum(cluster_active) > 1:
        active_idx = np.where(cluster_active)[0]
        best_cost = np.inf
        best_pair = None
        for i in active_idx:
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
        new_size = cluster_size[i] + cluster_size[j]
        new_label_dist = (cluster_size[i] * cluster_label_dist[i] + cluster_size[j] * cluster_label_dist[j]) / new_size
        cluster_size[i] = new_size
        cluster_label_dist[i] = new_label_dist
        cluster_members[i] = cluster_members[i] + cluster_members[j]
        cluster_active[j] = False
        n_remaining = int(np.sum(cluster_active))
        merge_order.append((int(i), int(j), float(best_cost), n_remaining))
        info_curve.append((n_remaining, info_TY()))
        if n_remaining in (4, 8, 16, 24, 32, 48, 64, 128) or n_remaining == next_milestone:
            cluster_history[n_remaining] = [list(cluster_members[k]) for k in cluster_members if cluster_active[k]]
            next_milestone = max(2, next_milestone // 2)
        if n_remaining % 100 == 0 or n_remaining < 30:
            elapsed = time.perf_counter() - t0
            print(f"    n={n_remaining:5d}  I(T;Y)={info_curve[-1][1]:.4f}  cost={best_cost:.4e}  ({elapsed:.1f}s)", flush=True)
    return merge_order, info_curve, cluster_history


def compute_centroids(reps, cluster_members_list):
    centroids = []
    for members in cluster_members_list:
        if not members:
            centroids.append(np.zeros(reps.shape[1], dtype=np.float32))
        else:
            centroids.append(reps[members].mean(axis=0))
    return np.array(centroids)


def main():
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/l4_mixed_ckpts/l4_mixed_v24c_dual_notebook_step500.safetensors")
    num_problems = getenv("NUM_PROBLEMS", 600)
    n_loops = getenv("LOOPS", 4)
    fixed_len = getenv("FIXED_LEN", 96)
    seed = getenv("SEED", 42)
    out_dir = getenv("OUT_DIR", ".cache/ib_centroids_per_layer")

    cfg = Config()
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Per-(op, layer) centroid extraction ===")
    print(f"  ckpt: {ckpt}  out: {out_dir}")
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
    examples = generate_math("L4_MIXED", num_problems + 100, seed=seed + 99, digit_spacing=True)
    examples = examples[:num_problems]

    print(f"\ncollecting per-layer reps (each cycle yields 4 reps, one per phase layer)...")
    t0 = time.perf_counter()
    reps, tags = collect_per_layer_reps(model, tok, examples, n_loops, fixed_len, eq_ids)
    print(f"\nCollected {len(reps)} reps ({time.perf_counter()-t0:.1f}s)")
    # Compose (op, layer) joint label: op * 4 + layer  →  0..15
    joint_labels = tags[:, 0] * 4 + tags[:, 1]
    op_layer_counts = np.bincount(joint_labels, minlength=16)
    op_names = ['+', '-', '*', '/']
    print(f"\n(op, layer) distribution:")
    for ol in range(16):
        op_i = ol // 4
        ly_i = ol % 4
        print(f"  ({op_names[op_i]}, L{ly_i}): {op_layer_counts[ol]:5d}")

    print(f"\nRunning agglomerative IB on (op, layer) joint labels...")
    n_classes_joint = 16  # 4 ops × 4 layers
    merge_order, info_curve, cluster_history = agglomerative_ib(reps, joint_labels, n_classes_joint)

    info_arr = np.array(info_curve, dtype=np.float64)
    np.save(f"{out_dir}/info_curve.npy", info_arr)
    print(f"\nSaved info curve: {out_dir}/info_curve.npy  ({info_arr.shape})")

    for n_k in (4, 8, 16, 24, 32, 48, 64):
        if n_k in cluster_history:
            members = cluster_history[n_k]
            centroids = compute_centroids(reps, members)
            sizes = np.array([len(m) for m in members])
            np.save(f"{out_dir}/centroids_n{n_k}.npy", centroids)
            np.save(f"{out_dir}/sizes_n{n_k}.npy", sizes)
            print(f"\nn={n_k} clusters: sizes range {sizes.min()}..{sizes.max()}")
            for c_i, m in enumerate(members[:8]):
                lbl_dist = np.bincount(joint_labels[m], minlength=n_classes_joint)
                purity = lbl_dist.max() / max(1, lbl_dist.sum())
                top_jl = int(lbl_dist.argmax())
                top_op = op_names[top_jl // 4]
                top_ly = top_jl % 4
                print(f"  cluster {c_i:3d}: size={len(m):4d}  top=({top_op}, L{top_ly})  purity={purity:.3f}")

    print(f"\nI(T;Y) at key cluster counts:")
    for n_k in (4, 8, 16, 24, 32, 48, 64, 128):
        match = info_arr[info_arr[:, 0].astype(int) == n_k]
        if len(match) > 0:
            print(f"  n={n_k:4d}: I(T;Y) = {match[0, 1]:.4f}")


if __name__ == "__main__":
    main()
