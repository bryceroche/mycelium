"""Evaluate a v105.10 ckpt on the OOD 5-digit eval set.

Loads a v105.10 checkpoint (built from `v105_10_factor_graph_prod.sh`),
forwards the OOD val set, and reports:

  * 200-bin number readout accuracy on OOD
      (expected: 0% — every gold value > training range has no matching bin)
  * AR digit decoder per-digit accuracy on OOD
      (compositional — the test for v105.10's killer claim)
  * AR digit decoder per-cell accuracy on OOD
      (cell is correct iff all VALID digit positions match gold)

Usage:
  .venv/bin/python scripts/eval_v105_10_ood.py
  CKPT=.cache/fg_v105_5_ckpts/v105_10_prod_step5000.safetensors \
    .venv/bin/python scripts/eval_v105_10_ood.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# v105.10 dual-readout: ensure env vars are set BEFORE any v105.5 module import
# so the module-level constants pick them up.
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
    V105_5_WAIST, V105_5_CODEBOOK_N,
    attach_fg_params_v105_5, load_ckpt_v105_5,
    fg_breathing_forward_v105_5,
    _compile_jit_fg_eval_v105_8, _compile_jit_fg_eval_v105_9,
)
from mycelium.factor_graph_data_v105_5 import FactorGraphLoaderV105_5

# fp32 cast helper — same as training driver
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


def main():
    OOD_PATH    = os.environ.get(
        "OOD_PATH", ".cache/factor_graph_test_ood_5digit.jsonl"
    )
    CKPT        = os.environ.get(
        "CKPT", ".cache/fg_v105_5_ckpts/v105_10_prod_step5000.safetensors"
    )
    BATCH       = int(os.environ.get("BATCH", "8"))
    MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "100"))
    SEED        = int(os.environ.get("SEED", "42"))

    K        = V105_5_K_MAX
    N_MAX    = V105_5_N_MAX
    F_MAX    = V105_5_F_MAX
    N_DIGITS = V105_5_N_DIGITS
    WAIST    = V105_5_WAIST
    N_CODE   = V105_5_CODEBOOK_N

    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"CKPT not found: {CKPT}")
    if not os.path.exists(OOD_PATH):
        raise FileNotFoundError(f"OOD_PATH not found: {OOD_PATH}")

    print(f"=== v105.10 OOD eval ===")
    print(f"  ckpt:     {CKPT}")
    print(f"  ood:      {OOD_PATH}")
    print(f"  batch:    {BATCH}  max_batches: {MAX_BATCHES}")
    print(f"  device:   {Device.DEFAULT}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print(f"loading Pythia-410M backbone...")
    sd_p  = _load_state()
    model = load_breathing(cfg, sd=sd_p)
    del sd_p
    cast_layers_fp32(model)

    attach_fg_params_v105_5(
        model, hidden=cfg.hidden,
        n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K,
        waist=WAIST, n_code=N_CODE,
    )
    Device[Device.DEFAULT].synchronize()

    print(f"\nloading ckpt: {CKPT}")
    load_ckpt_v105_5(model, CKPT)

    # OOD loader
    loader = FactorGraphLoaderV105_5(
        OOD_PATH, batch_size=BATCH,
        difficulty_filter=None,
        curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V105_5_N_HEADS,
        n_digits=N_DIGITS, seed=SEED,
    )

    # Compile eval JITs. v8 (codebook) is only available if v105.8 readout
    # is attached (v105.10 path). v105.11 has no codebook — skip v8.
    Tensor.training = False
    has_v8_codebook = hasattr(model, "fg_v105_8_number_codebook")
    if has_v8_codebook:
        eval_fn_v8 = _compile_jit_fg_eval_v105_8(
            model, K=K, B=BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )
    else:
        eval_fn_v8 = None
        print("  v8 number codebook not attached — skipping v8 readout (v105.11 mode)")
    eval_fn_v9 = _compile_jit_fg_eval_v105_9(
        model, K=K, B=BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
    )

    # Aggregates
    agg_v8 = {}  # per difficulty: number-bin acc
    agg_v9 = {}  # per difficulty: digit-decoder acc
    # Digit-level diagnostics (across all OOD cells):
    n_digit_correct = 0
    n_digit_total   = 0
    # Per-position digit accuracy (5 positions, LSD-first)
    pos_correct = np.zeros(N_DIGITS, dtype=np.int64)
    pos_total   = np.zeros(N_DIGITS, dtype=np.int64)

    n_batches = 0
    t0 = time.time()
    for batch in loader.iter_eval(batch_size=BATCH):
        digit_init   = batch["digit_init"]
        node_kinds   = batch["node_kinds"]
        staging_mask = batch["staging_mask"]
        head_op_mask = batch["head_op_mask"]
        gold_digits  = batch["gold_digits"]
        obs_mask     = batch["observed_mask"]
        valid_mask   = batch["digit_valid_mask"]
        num_bin_tgt  = batch["number_bin_target"]
        query_idx_np = batch["query_idx"]
        picks        = batch["picks"]

        # v105.8 path (codebook) — only run if attached.
        if eval_fn_v8 is not None:
            pred_bin_t, _ = eval_fn_v8(
                digit_init, node_kinds, staging_mask, head_op_mask,
                num_bin_tgt, obs_mask, valid_mask,
            )
            pred_bin_np = pred_bin_t.numpy()
        else:
            pred_bin_np = None

        # v105.9 path
        pred_dg_t, _ = eval_fn_v9(
            digit_init, node_kinds, staging_mask, head_op_mask,
            gold_digits, obs_mask, valid_mask,
        )
        pred_dg_np = pred_dg_t.numpy()

        gold_bin_np = num_bin_tgt.numpy()
        gold_dg_np  = gold_digits.numpy()
        obs_np      = obs_mask.numpy()
        valid_np    = valid_mask.numpy()

        for b in range(len(picks)):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            for tag, dict_agg in (("v8", agg_v8), ("v9", agg_v9)):
                if diff not in dict_agg:
                    dict_agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                                       "query_correct": 0, "n_puzzles": 0}
            nv = int(batch["n_vars_total"][b])
            qi = int(query_idx_np[b])
            for vi in range(min(nv, N_MAX)):
                if obs_np[b, vi] != 0:
                    continue
                agg_v8[diff]["n_unobs"] += 1
                agg_v9[diff]["n_unobs"] += 1
                # v8 — 200-bin number readout (only if codebook attached)
                if pred_bin_np is not None and int(pred_bin_np[b, vi]) == int(gold_bin_np[b, vi]):
                    agg_v8[diff]["n_correct_unobs"] += 1
                # v9 — AR digit decoder per-cell
                v_valid = valid_np[b, vi].astype(bool)
                if v_valid.any():
                    if np.all(pred_dg_np[b, vi, v_valid] == gold_dg_np[b, vi, v_valid]):
                        agg_v9[diff]["n_correct_unobs"] += 1
                    # Per-position digit accuracy and grand-total digit accuracy.
                    for p in range(N_DIGITS):
                        if v_valid[p]:
                            pos_total[p] += 1
                            n_digit_total += 1
                            if pred_dg_np[b, vi, p] == gold_dg_np[b, vi, p]:
                                pos_correct[p] += 1
                                n_digit_correct += 1
            if qi < N_MAX:
                if pred_bin_np is not None and int(pred_bin_np[b, qi]) == int(gold_bin_np[b, qi]):
                    agg_v8[diff]["query_correct"] += 1
                q_valid = valid_np[b, qi].astype(bool)
                if q_valid.any():
                    if np.all(pred_dg_np[b, qi, q_valid] == gold_dg_np[b, qi, q_valid]):
                        agg_v9[diff]["query_correct"] += 1
            agg_v8[diff]["n_puzzles"] += 1
            agg_v9[diff]["n_puzzles"] += 1

        n_batches += 1
        if n_batches >= MAX_BATCHES:
            break

    dt = time.time() - t0

    print(f"\n=== OOD results ({n_batches} batches × B={BATCH} = "
          f"{n_batches * BATCH} ex, {dt:.1f}s) ===")
    print(f"\n[v105.8 path — 200-bin number readout]")
    print(f"  (expected near 0% — codebook bins only span [0, 9999])")
    for d in ("easy", "medium", "hard"):
        v = agg_v8.get(d)
        if v is None:
            continue
        cell = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        qa   = v["query_correct"] / max(v["n_puzzles"], 1)
        print(
            f"  val[{d:6s}]: cell_acc={cell:.4f} ({v['n_correct_unobs']}/{v['n_unobs']}) "
            f"query_acc={qa:.4f} ({v['query_correct']}/{v['n_puzzles']})"
        )

    print(f"\n[v105.9 path — AR digit decoder]")
    print(f"  (the compositionality test — any nonzero is the win)")
    for d in ("easy", "medium", "hard"):
        v = agg_v9.get(d)
        if v is None:
            continue
        cell = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        qa   = v["query_correct"] / max(v["n_puzzles"], 1)
        print(
            f"  val[{d:6s}]: cell_acc={cell:.4f} ({v['n_correct_unobs']}/{v['n_unobs']}) "
            f"query_acc={qa:.4f} ({v['query_correct']}/{v['n_puzzles']})"
        )

    print(f"\n[per-digit-position accuracy — AR decoder, LSD-first]")
    pos_labels = ["ones", "tens", "100s", "1000s", "10000s"]
    digit_acc_total = n_digit_correct / max(n_digit_total, 1)
    print(f"  overall digit_acc = {digit_acc_total:.4f} "
          f"({n_digit_correct}/{n_digit_total})")
    for p in range(N_DIGITS):
        pa = pos_correct[p] / max(int(pos_total[p]), 1)
        print(f"    pos{p} ({pos_labels[p]:>7s}): {pa:.4f} "
              f"({int(pos_correct[p])}/{int(pos_total[p])})")


if __name__ == "__main__":
    main()
