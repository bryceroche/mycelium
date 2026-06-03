"""Decoder consistency diagnostic for v105.10 AR digit decoder.

Tests whether the AR digit decoder respects its conditioning. For each of N
unobserved cells:
  1. Run the breathing forward, capture cell_hidden.
  2. Greedy-decode digits via the AR chain.
  3. Re-run the AR decoder with d1 (the second-from-LSD digit, position 1)
     clamped to:
       (a) the gold d1
       (b) gold_d1 ± 1
       (c) gold_d1 + 5
  4. Observe how the d2 distribution changes across clamps. Compute the
     entropy of d2 across clamps and the KL divergence between d2 dists.

Interpretation:
  * If d2 responds to clamp changes → decoder learned conditional rules; a
    cheap MCTS over digits would work.
  * If d2 stays constant → decoder ignores its conditioning; MCTS would need
    a full breathing forward per rollout (much more expensive).

LSD-first convention (matches v105.5): position 0 = ones, position 1 = tens,
position 2 = hundreds, etc. The AR chain iterates 0 → N-1 (LSD-first), so d2
sees d1's soft embedding via cond_accum.

Usage:
  CKPT=.cache/fg_v105_5_ckpts/v105_10_prod_step5000.safetensors \
    .venv/bin/python scripts/diag_v105_10_decoder_consistency.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("V105_5_TASK", "1")
os.environ.setdefault("V105_8_PER_NUMBER_READOUT", "1")
os.environ.setdefault("V105_8_N_NUMBER_BINS", "200")
os.environ.setdefault("V105_9_AR_DIGIT_DECODER", "1")
os.environ.setdefault("V105_9_AR_COND_SCALE", "0.5")
os.environ.setdefault("V105_10_DUAL_READOUT", "1")
os.environ.setdefault("V105_10_DIGIT_WEIGHT", "0.3")

import numpy as np
from tinygrad import Device, Tensor, dtypes

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v105_5 import (
    V105_5_K_MAX, V105_5_N_MAX, V105_5_F_MAX, V105_5_N_DIGITS, V105_5_N_HEADS,
    V105_5_WAIST, V105_5_CODEBOOK_N, V105_9_AR_COND_SCALE,
    attach_fg_params_v105_5, load_ckpt_v105_5,
    fg_breathing_forward_v105_5,
)
from mycelium.factor_graph_data_v105_5 import FactorGraphLoaderV105_5


def cast_layers_fp32(model):
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def kl_div_np(p, q, eps=1e-9):
    """KL(p || q) along last axis. Assumes both sum to 1 along last axis."""
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=-1).mean())


def entropy_np(p, eps=1e-9):
    return float(-(p * np.log(p + eps)).sum(axis=-1).mean())


def ar_decode_one_clamp(
    cell_hidden_np: np.ndarray,        # (M, H) — M cells we care about
    digit_codebook_np: np.ndarray,     # (N_DIGITS, 10, H)
    n_digits: int,
    cond_scale: float,
    clamp_pos: int | None = None,
    clamp_val: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the LSD-first AR chain on a stack of cell_hiddens.

    If clamp_pos is set, replace the soft prob at that position with a hard
    one-hot at clamp_val and use that hard distribution to update cond_accum
    for subsequent positions.

    Returns:
      logits : (M, N_DIGITS, 10) — raw logits at each position.
      probs  : (M, N_DIGITS, 10) — softmax probabilities at each position.
    """
    M = cell_hidden_np.shape[0]
    H = cell_hidden_np.shape[1]
    logits = np.zeros((M, n_digits, 10), dtype=np.float32)
    probs  = np.zeros((M, n_digits, 10), dtype=np.float32)
    cond_accum = np.zeros((M, H), dtype=np.float32)
    for p in range(n_digits):
        cb_p = digit_codebook_np[p]                              # (10, H)
        pos_hidden = cell_hidden_np + cond_accum                  # (M, H)
        log_p = pos_hidden @ cb_p.T                               # (M, 10)
        prob_p = softmax_np(log_p, axis=-1)                       # (M, 10)
        logits[:, p, :] = log_p
        probs[:, p, :]  = prob_p
        if clamp_pos is not None and p == clamp_pos:
            # Replace soft prob with one-hot at clamp_val.
            hard = np.zeros((M, 10), dtype=np.float32)
            hard[:, int(clamp_val)] = 1.0
            cond_accum = cond_accum + cond_scale * (hard @ cb_p)
        else:
            cond_accum = cond_accum + cond_scale * (prob_p @ cb_p)
    return logits, probs


def main():
    CKPT      = os.environ.get(
        "CKPT", ".cache/fg_v105_5_ckpts/v105_10_prod_step5000.safetensors"
    )
    VAL_PATH  = os.environ.get(
        "VAL_PATH", ".cache/factor_graph_test_loguniform.jsonl"
    )
    BATCH     = int(os.environ.get("BATCH", "8"))
    N_CELLS   = int(os.environ.get("N_CELLS", "100"))
    SEED      = int(os.environ.get("SEED", "42"))

    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"CKPT not found: {CKPT}")
    if not os.path.exists(VAL_PATH):
        raise FileNotFoundError(f"VAL_PATH not found: {VAL_PATH}")

    K        = V105_5_K_MAX
    N_MAX    = V105_5_N_MAX
    F_MAX    = V105_5_F_MAX
    N_DIGITS = V105_5_N_DIGITS

    print(f"=== v105.10 decoder consistency diag ===")
    print(f"  ckpt:    {CKPT}")
    print(f"  val:     {VAL_PATH}")
    print(f"  n_cells: {N_CELLS}  batch: {BATCH}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print("loading Pythia-410M backbone...")
    sd_p  = _load_state()
    model = load_breathing(cfg, sd=sd_p)
    del sd_p
    cast_layers_fp32(model)

    attach_fg_params_v105_5(
        model, hidden=cfg.hidden,
        n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K,
        waist=V105_5_WAIST, n_code=V105_5_CODEBOOK_N,
    )
    Device[Device.DEFAULT].synchronize()
    load_ckpt_v105_5(model, CKPT)

    # Pull the digit_codebook (frozen for the duration of this diagnostic).
    digit_codebook_np = model.fg_v105_5_digit_codebook.numpy()  # (N_DIGITS, 10, H)

    loader = FactorGraphLoaderV105_5(
        VAL_PATH, batch_size=BATCH, difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V105_5_N_HEADS,
        n_digits=N_DIGITS, seed=SEED,
    )

    Tensor.training = False

    # Collect cell_hidden + gold_digits for up to N_CELLS unobserved cells
    # whose magnitude is at least 2 digits (so clamping d1 is meaningful).
    collected_cell_hidden = []
    collected_gold_digits = []
    collected_valid_mask  = []

    t0 = time.time()
    for batch in loader.iter_eval(batch_size=BATCH):
        digit_init   = batch["digit_init"]
        node_kinds   = batch["node_kinds"]
        staging_mask = batch["staging_mask"]
        head_op_mask = batch["head_op_mask"]
        gold_digits  = batch["gold_digits"]
        obs_mask     = batch["observed_mask"]
        valid_mask   = batch["digit_valid_mask"]
        picks        = batch["picks"]

        # Forward (eager — we need cell_hidden, which is not exposed by the
        # eval JIT). Build cell_hidden from final terminal_var_hidden.
        (dlh, _flh, _calib, _maglh, terminal_var_hidden,
         _num_logits_final, _digit_logits_pooled_final) = (
            fg_breathing_forward_v105_5(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
            )
        )
        # cell_hidden = mean of terminal var_tokens across digit positions
        # (mirrors what the AR decoder reads in the forward pass).
        cell_hidden = terminal_var_hidden.mean(axis=2)              # (B, N_MAX, H)
        ch_np  = cell_hidden.numpy()
        gd_np  = gold_digits.numpy()
        obs_np = obs_mask.numpy()
        vm_np  = valid_mask.numpy()

        for b in range(len(picks)):
            nv = int(batch["n_vars_total"][b])
            for vi in range(min(nv, N_MAX)):
                if obs_np[b, vi] != 0:
                    continue
                vv = vm_np[b, vi].astype(bool)
                # Need at least 2 valid digit positions (pos 0 and pos 1).
                if not vv[0] or not vv[1]:
                    continue
                collected_cell_hidden.append(ch_np[b, vi])
                collected_gold_digits.append(gd_np[b, vi])
                collected_valid_mask.append(vm_np[b, vi])
                if len(collected_cell_hidden) >= N_CELLS:
                    break
            if len(collected_cell_hidden) >= N_CELLS:
                break
        if len(collected_cell_hidden) >= N_CELLS:
            break

    if len(collected_cell_hidden) < N_CELLS:
        print(f"[warn] collected only {len(collected_cell_hidden)} cells "
              f"(target {N_CELLS}); continuing", flush=True)
    print(f"  collected {len(collected_cell_hidden)} cells in {time.time()-t0:.1f}s")

    if not collected_cell_hidden:
        print("[error] no cells collected. Aborting.", flush=True)
        return

    ch_stack = np.stack(collected_cell_hidden, axis=0)              # (M, H)
    gd_stack = np.stack(collected_gold_digits, axis=0)              # (M, N_DIGITS)

    M = ch_stack.shape[0]
    print(f"\nrunning AR decoder consistency test on M={M} cells...")
    print(f"  clamp position: 1 (LSD-first → tens digit)")
    print(f"  observe position: 2 (LSD-first → hundreds digit)")
    print()

    # Baseline — no clamp.
    _, probs_base = ar_decode_one_clamp(
        ch_stack, digit_codebook_np, n_digits=N_DIGITS,
        cond_scale=V105_9_AR_COND_SCALE,
    )
    # probs_base[:, 2, :] is the d2 distribution without clamping.

    # Clamp position 1 (tens digit) to various values relative to gold.
    gold_d1 = gd_stack[:, 1].astype(np.int32)
    clamps = {}  # label -> (M, 10) d2-distribution after clamping
    clamp_labels = []
    # For each cell we want different clamp values (per-cell), but the AR
    # decode is vectorized over the cell axis with a SINGLE clamp_val.
    # Workaround: for each kind (gold, gold+1, gold-1, gold+5), run all
    # cells with their per-cell clamp value baked in. Easiest path:
    # loop over the cells × 4 clamp-flavors and accumulate per-cell d2.
    flavors = [
        ("gold",       lambda gd1: gd1),
        ("gold_p1",    lambda gd1: (gd1 + 1) % 10),
        ("gold_m1",    lambda gd1: (gd1 - 1) % 10),
        ("gold_p5",    lambda gd1: (gd1 + 5) % 10),
    ]
    for label, fn in flavors:
        clamp_vals = np.array([int(fn(int(g))) for g in gold_d1], dtype=np.int32)
        d2_dist = np.zeros((M, 10), dtype=np.float32)
        # Run with a single global clamp_val per pass; loop the 10 distinct
        # values to amortize. (Could be vectorized further if needed.)
        for cv in range(10):
            sel = (clamp_vals == cv)
            if not sel.any():
                continue
            sub_ch = ch_stack[sel]
            _, sub_probs = ar_decode_one_clamp(
                sub_ch, digit_codebook_np, n_digits=N_DIGITS,
                cond_scale=V105_9_AR_COND_SCALE,
                clamp_pos=1, clamp_val=cv,
            )
            d2_dist[sel] = sub_probs[:, 2, :]
        clamps[label] = d2_dist
        clamp_labels.append(label)

    # ---- Analysis ----
    # 1. Entropy of d2 averaged across cells, per clamp flavor.
    print("d2 mean entropy per clamp flavor (lower = sharper):")
    base_ent = entropy_np(probs_base[:, 2, :])
    print(f"  baseline (no clamp): H[d2] = {base_ent:.4f}")
    for lab in clamp_labels:
        ent = entropy_np(clamps[lab])
        print(f"  clamp d1={lab:<10s}: H[d2] = {ent:.4f}")

    # 2. KL divergence between d2 distributions across clamps and baseline.
    print(f"\nmean per-cell KL[d2 | clamp || d2 | baseline]:")
    for lab in clamp_labels:
        kl = kl_div_np(clamps[lab], probs_base[:, 2, :])
        print(f"  {lab:<10s}: KL = {kl:.4f}")

    # 3. KL divergence between clamp-gold and clamp-non-gold d2 distributions.
    print(f"\nmean per-cell KL[d2 | clamp=non_gold || d2 | clamp=gold]:")
    for lab in clamp_labels:
        if lab == "gold":
            continue
        kl = kl_div_np(clamps[lab], clamps["gold"])
        print(f"  {lab:<10s} vs gold: KL = {kl:.4f}")

    # 4. Predicted d2 (argmax) under each clamp.
    print(f"\nd2 argmax change-rate (fraction of cells where argmax differs):")
    arg_base = probs_base[:, 2, :].argmax(axis=-1)
    arg_gold = clamps["gold"][:, :].argmax(axis=-1)
    for lab in clamp_labels:
        arg_clamp = clamps[lab][:, :].argmax(axis=-1)
        diff = float((arg_clamp != arg_gold).mean())
        diff_base = float((arg_clamp != arg_base).mean())
        print(f"  {lab:<10s}: vs gold-clamp = {diff:.3f}  vs no-clamp = {diff_base:.3f}")

    # 5. Verdict heuristic.
    print()
    max_kl = max(
        kl_div_np(clamps[lab], clamps["gold"])
        for lab in ("gold_p1", "gold_m1", "gold_p5")
    )
    if max_kl > 0.05:
        verdict = "RESPONSIVE"
        explanation = (
            "decoder learned conditional rules → cheap MCTS over digits is viable "
            "(no full breathing forward needed per rollout)."
        )
    elif max_kl > 0.005:
        verdict = "WEAKLY-RESPONSIVE"
        explanation = (
            "decoder shows some response to conditioning but signal is faint; "
            "MCTS would amplify noise more than signal."
        )
    else:
        verdict = "UNRESPONSIVE"
        explanation = (
            "decoder appears to ignore its d1 conditioning; cheap MCTS would "
            "not help — a search would need a full breathing forward per rollout."
        )
    print(f"verdict: {verdict}")
    print(f"  reason: max_KL[non_gold || gold] = {max_kl:.4f}")
    print(f"  {explanation}")


if __name__ == "__main__":
    main()
