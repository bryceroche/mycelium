"""Per-(op, layer, head) centroid extraction from v38 ckpt.

Captures per-head 64d attention outputs at each phase-layer of the last breath,
for each cycle of ARITH problems the model gets RIGHT. Each centroid tagged
with (op_label, layer_idx, head_idx). 4 ops × 4 layers × 16 heads = 256 entries.

The per-head 64d output is then projected to 1024d via W_O column block:
  head_h_contribution = head_h_64d @ W_O[64h:64(h+1), :]
This gives the head's "characteristic contribution" to the residual stream.

Output: centroids_per_op_layer_head_n256.npy shape (256, 1024).
        Indexed by entry_idx = op*64 + layer*16 + head.

Usage:
    DEV=PCI+AMD CKPT=/path/to/v38_step1500.safetensors NUM_PROBLEMS=600 \\
        python scripts/extract_per_op_layer_head_centroids.py
"""
import os
import sys
import time
import math

os.environ.setdefault("DEV", "PCI+AMD")
# v38's geometry must be matched for centroid extraction
os.environ.setdefault("PER_HEAD_PITCH", "1")
os.environ.setdefault("BFIELD_WAIST", "256")
os.environ.setdefault("LOOKUP_VALUE_INJECT", "1")
os.environ.setdefault("LOOKUP_VALUES_INIT_PATH", ".cache/ib_centroids_per_layer/centroids_n16.npy")
os.environ.setdefault("LOOKUP_TEMP", "20")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, encode_cycles, parse_int_answer
from mycelium.lookup_table import eq_token_ids_for, op_label_from_text
from mycelium.breathing import (
    _sine_temp_baseline, _layernorm,
    BREATH_TIME_EMBED, PER_HEAD_PITCH,
    LAYER_PITCH_TARGET, CONSTANT_RADIUS, BREATH_NORM_OSC,
    PER_BREATH_TEMP, _per_layer_temp_within_breath,
    _per_layer_norm_scale_within_breath,
    BFIELD_WAIST, BFIELD_END_OF_BREATH,
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


def layer_forward_with_per_head(layer, x, loop_idx, alpha, temp_mult=1.0):
    """Mirror of BreathingLayer._forward (kv_cache=None path), but ALSO returns
    the per-head attention output (B, n_heads, S, head_dim) BEFORE W_O concat.
    """
    cfg = layer.cfg
    B, S, H = x.shape

    attn_in = _layernorm(x, layer.shared.in_ln_g, layer.shared.in_ln_b, cfg.layer_norm_eps)
    mlp_in = _layernorm(x, layer.shared.post_ln_g, layer.shared.post_ln_b, cfg.layer_norm_eps)
    attn_in_dt = attn_in.cast(x.dtype) if attn_in.dtype != x.dtype else attn_in
    mlp_in_dt = mlp_in.cast(x.dtype) if mlp_in.dtype != x.dtype else mlp_in

    q = (attn_in_dt @ layer.wq + layer.bq).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    k = (attn_in_dt @ layer.wk + layer.bk).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
    v = (attn_in_dt @ layer.shared.wv + layer.shared.bv).reshape(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)

    scale = layer.attn_scale / float(temp_mult)
    q, k = layer.rope.apply(q, k, loop_idx, start_pos=0, alpha=alpha)
    scores = q @ k.transpose(-2, -1) * scale
    mask = Tensor.ones(S, S, dtype=scores.dtype).tril().reshape(1, 1, S, S)
    scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = scores.softmax(-1).cast(v.dtype)

    # Per-head output BEFORE concat: (B, n_heads, S, head_dim)
    per_head = attn @ v        # (B, n_heads, S, head_dim)

    # Continue normal forward path
    ctx = per_head.transpose(1, 2).reshape(B, S, H)
    attn_out = ctx @ layer.shared.wo + layer.shared.bo
    ff = (mlp_in_dt @ layer.w_in + layer.b_in).gelu()
    ffn_out = ff @ layer.shared.w_out + layer.shared.b_out
    out = x + attn_out + ffn_out
    return out, per_head


def run_breath_capture_per_head(model, x, loop_idx, n_phases):
    """Replicate breathe_once but capture per-head outputs at EACH of 4 layers.
    Returns list of 4 per-head tensors (B, n_heads, S, head_dim)."""
    alpha = model.block.rope._alpha_at(loop_idx, x.dtype)
    if BREATH_TIME_EMBED:
        x = x + model.block.breath_embed[loop_idx].reshape(1, 1, -1).cast(x.dtype)
    ac_base, asn_base = alpha
    per_head_list = []
    for layer_idx, layer in enumerate(model.block.layers):
        if PER_HEAD_PITCH and layer_idx > 0:
            cos_o = model.block.per_head_pitch_cos[layer_idx].cast(x.dtype)
            sin_o = model.block.per_head_pitch_sin[layer_idx].cast(x.dtype)
            ac_layer = ac_base * cos_o - asn_base * sin_o
            asn_layer = ac_base * sin_o + asn_base * cos_o
            layer_alpha = (ac_layer, asn_layer)
        else:
            layer_alpha = alpha
        x, per_head = layer_forward_with_per_head(layer, x, loop_idx, layer_alpha)
        per_head_list.append(per_head)
        # mid-breath waist if v38 mode (applies after L1)
        if BFIELD_WAIST > 0 and layer_idx == 1 and not BFIELD_END_OF_BREATH:
            x = model.block.apply_bfield_waist(x)
    return per_head_list


def batched_filter_correct(model, tok, examples, n_loops, fixed_len, batch_size=64):
    """Use the optimized batched accuracy eval to find correct examples.
    Returns list of (ex, op_label, parsed_pred) for examples the model got right."""
    from mycelium.l3_training import accuracy_at_loops_multi
    cache_max_len = fixed_len + 40
    acc, rows = accuracy_at_loops_multi(
        model, tok, examples, n_loops=n_loops,
        batch_size=batch_size, cache_max_len=cache_max_len,
    )
    print(f"  batched accuracy: {acc*100:.1f}%  ({len(rows)} rows)")
    correct = []
    for ex, parsed, _gen_text in rows:
        if parsed is None:
            continue
        if parsed != ex.answer:
            continue
        op_label = op_label_from_text(ex.problem)
        if op_label < 0 or op_label >= 4:
            continue
        correct.append((ex, op_label))
    return correct


def main():
    ckpt = getenv("CKPT", "/home/bryce/mycelium/.cache/arith_ckpts/v38_bfield_w256_step1500.safetensors")
    num_problems = getenv("NUM_PROBLEMS", 600)
    n_loops = getenv("LOOPS", 8)
    fixed_len = getenv("FIXED_LEN", 32)
    seed = getenv("SEED", 42)
    out_dir = getenv("OUT_DIR", ".cache/per_op_layer_head_centroids")
    os.makedirs(out_dir, exist_ok=True)

    cfg = Config()
    print(f"=== Per-(op, layer, head) centroid extraction ===")
    print(f"  ckpt: {os.path.basename(ckpt)}")
    print(f"  N: {num_problems}  loops: {n_loops}  fixed_len: {fixed_len}")
    print(f"  output: 4 ops × 4 layers × 16 heads = 256 centroids × 1024d")

    print("\nloading Pythia → breathing transformer...")
    t0 = time.perf_counter()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_fp32(model)
    Device[Device.DEFAULT].synchronize()
    print(f"  loaded base ({time.perf_counter()-t0:.1f}s)")
    print(f"\nloading v38 ckpt: {os.path.basename(ckpt)}")
    state = safe_load(ckpt)
    info = model.load_state_dict(state, strict=False)
    Device[Device.DEFAULT].synchronize()
    if info.get("missing"):
        print(f"  missing keys (expected): {len(info['missing'])}")

    tok = load_tokenizer()
    eq_ids = eq_token_ids_for(tok)

    print(f"\ngenerating {num_problems} ARITH problems...")
    examples = generate_math("ARITH", num_problems + 200, seed=seed, digit_spacing=True)

    print(f"\nFiltering to correct examples via batched eval (B=64)...")
    Tensor.training = False
    t1 = time.perf_counter()
    correct_all = batched_filter_correct(model, tok, examples, n_loops=n_loops,
                                          fixed_len=fixed_len, batch_size=64)
    # Balance per op: take at most num_problems // 4 + 25 per op
    cap_per_op = num_problems // 4 + 25
    correct_examples = []
    correct_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for ex, op_label in correct_all:
        if correct_counts[op_label] < cap_per_op:
            correct_examples.append((ex, op_label))
            correct_counts[op_label] += 1
        if len(correct_examples) >= num_problems:
            break
    print(f"\n  filtered: {len(correct_examples)} correct examples  counts={correct_counts}")
    print(f"  filter took {time.perf_counter()-t1:.1f}s")

    # Now capture per-(op, layer, head) outputs at "=" position for each correct example
    print(f"\nCapturing per-head 64d outputs at '=' position, last breath, all 4 layers...")
    n_heads = cfg.n_heads
    head_dim = cfg.head_dim
    # Accumulators: sums[(op, layer, head)] = (count, sum_64d)
    sums = np.zeros((4, 4, n_heads, head_dim), dtype=np.float64)
    counts = np.zeros((4, 4, n_heads), dtype=np.int64)

    t2 = time.perf_counter()
    for ex_idx, (ex, op_label) in enumerate(correct_examples):
        cycle_encodings = encode_cycles(tok, ex)
        if not cycle_encodings:
            continue
        # Sanity re-derive op from problem text
        if op_label_from_text(ex.problem) != op_label:
            continue
        ids_list, prefix_len, _ = cycle_encodings[0]
        ids = ids_list[:fixed_len]
        tokens_np = np.zeros((1, fixed_len), dtype=np.int32)
        tokens_np[0, :len(ids)] = ids
        tokens = Tensor(tokens_np, dtype=dtypes.int).realize()

        # Run all breaths up to the last, then capture per-head on the last breath
        x = model.embed(tokens).cast(dtypes.half)
        for l in range(n_loops - 1):
            x = model.block.breathe_once(x, l, temp_mult=_sine_temp_baseline(l, n_loops))
        # Last breath with per-head capture
        per_head_list = run_breath_capture_per_head(model, x, n_loops - 1, cfg.n_phases)

        # Find "=" position. For ARITH single-cycle the "=" is the last token of
        # the prompt (position prefix_len-1). Fall back to a full scan otherwise.
        eq_pos = -1
        for i, t in enumerate(ids):
            if t in eq_ids:
                eq_pos = i  # take FIRST occurrence (cycle 0 boundary)
                break
        if eq_pos < 0 or eq_pos >= fixed_len:
            continue

        # For each layer, extract per-head 64d at eq_pos
        for layer_idx, per_head_t in enumerate(per_head_list):
            ph_np = per_head_t.realize().numpy()[0]  # (n_heads, S, head_dim)
            for h in range(n_heads):
                rep_64 = ph_np[h, eq_pos, :].astype(np.float64)
                sums[op_label, layer_idx, h] += rep_64
                counts[op_label, layer_idx, h] += 1

        if (ex_idx + 1) % 25 == 0:
            print(f"  captured {ex_idx+1}/{len(correct_examples)} examples  "
                  f"({time.perf_counter()-t2:.1f}s)", flush=True)

    print(f"\n  capture took {time.perf_counter()-t2:.1f}s")

    # Compute means: (op, layer, head) → 64d
    centroids_64 = np.zeros((4, 4, n_heads, head_dim), dtype=np.float32)
    for o in range(4):
        for l in range(4):
            for h in range(n_heads):
                c = counts[o, l, h]
                if c > 0:
                    centroids_64[o, l, h] = (sums[o, l, h] / c).astype(np.float32)

    # Project each 64d centroid to 1024d via W_O column block of that head:
    #   head_h_contribution = head_h_64d @ W_O[64h:64(h+1), :]
    # Need to read W_O from the model (shared.wo, shape (hidden, hidden))
    # which represents Pythia's o_proj matrix.
    print(f"\nProjecting per-head 64d centroids → 1024d via W_O column blocks...")
    wo = model.block.shared.wo.cast(dtypes.float).realize().numpy()  # (hidden, hidden) = (1024, 1024)
    centroids_1024 = np.zeros((4, 4, n_heads, cfg.hidden), dtype=np.float32)
    for o in range(4):
        for l in range(4):
            for h in range(n_heads):
                # Head h's columns of W_O: rows [64h, 64(h+1)) (since input is concat of heads)
                wo_block = wo[h * head_dim:(h + 1) * head_dim, :]  # (head_dim, hidden) = (64, 1024)
                centroids_1024[o, l, h] = centroids_64[o, l, h] @ wo_block
    # Reshape to 256 × 1024d
    centroids_flat = centroids_1024.reshape(4 * 4 * n_heads, cfg.hidden)  # (256, 1024)
    print(f"  centroids shape: {centroids_flat.shape}")

    # Stats
    norms = np.linalg.norm(centroids_flat, axis=-1)
    print(f"  norms: mean={norms.mean():.4f}  min={norms.min():.4f}  max={norms.max():.4f}")
    cn = centroids_flat / (norms.reshape(-1, 1) + 1e-9)
    sim = cn @ cn.T
    # Mean off-diagonal cosine
    off = sim[np.triu_indices(256, k=1)]
    print(f"  cosine sims (off-diag): mean={off.mean():.4f}  median={np.median(off):.4f}  "
          f"min={off.min():.4f}  max={off.max():.4f}")

    out_path = f"{out_dir}/centroids_per_op_layer_head_n256.npy"
    np.save(out_path, centroids_flat)
    print(f"\nSaved: {out_path}  shape={centroids_flat.shape}")

    # Also save per-head 64d (in case we want per-head matching architecture later)
    out_64 = f"{out_dir}/centroids_per_op_layer_head_64d.npy"
    np.save(out_64, centroids_64.reshape(4 * 4 * n_heads, head_dim).astype(np.float32))
    print(f"Saved: {out_64}  shape=({4 * 4 * n_heads}, {head_dim})")


if __name__ == "__main__":
    main()
