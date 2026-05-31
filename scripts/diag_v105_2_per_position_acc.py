"""Per-position accuracy diagnostic for v105.2 (LSD-first + valid mask).

LSD-first interpretation:
  position 0 = ones place (ALWAYS valid for any number)
  position 1 = tens place (valid for values ≥ 10)
  position 2 = hundreds (valid for ≥ 100)
  position 3 = thousands (valid for ≥ 1000)
  position 4 = ten-thousands (valid for ≥ 10000)

We restrict accuracy to VALID positions only (using digit_valid_mask), so
trivial padding positions don't inflate the metric.

Moment of truth: position 0 (ones) accuracy. If model is doing real digit
arithmetic, position 0 should beat random (10%).
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import Counter
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v105_2 import (
    fg_breathing_forward_v105_2,
    attach_fg_params_v105_2,
    load_ckpt_v105_2,
)
from mycelium.factor_graph_data_v105_2 import FactorGraphLoaderV105_2


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


K_MAX     = int(getenv("V105_2_K_MAX", "8"))
N_DIGITS  = int(getenv("V105_2_N_DIGITS", "5"))
N_MAX     = int(getenv("V105_2_N_MAX", "16"))
F_MAX     = int(getenv("V105_2_F_MAX", "8"))
WAIST     = int(getenv("V105_2_WAIST", "512"))
N_CODE    = int(getenv("V105_2_CODEBOOK_N", "32"))
N_HEADS   = int(getenv("V105_2_N_HEADS", "16"))
N_BATCHES = int(getenv("N_BATCHES", "30"))
BATCH     = int(getenv("BATCH", "4"))
CKPT      = getenv("CKPT", ".cache/fg_v105_2_ckpts/v105_2_smoke_final.safetensors")

POS_LABELS = ["ones", "tens", "hundreds", "thousands", "ten-thousands"]


def _collect(loader, model, label):
    Tensor.training = False
    pos_correct = np.zeros(N_DIGITS, dtype=np.int64)
    pos_total   = np.zeros(N_DIGITS, dtype=np.int64)  # VALID positions only
    pos_correct_all = np.zeros(N_DIGITS, dtype=np.int64)
    pos_total_all   = np.zeros(N_DIGITS, dtype=np.int64)  # all positions (incl padding)
    pred_hist   = [Counter() for _ in range(N_DIGITS)]
    gold_hist   = [Counter() for _ in range(N_DIGITS)]
    cells_all_correct = 0
    cells_total = 0
    nb = 0
    for batch in loader.iter_eval(batch_size=loader.batch_size):
        dig_lh, _, _ = fg_breathing_forward_v105_2(
            model, batch["digit_init"], batch["node_kinds"],
            batch["staging_mask"], batch["head_op_mask"],
            K=K_MAX, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )
        pred = dig_lh[-1].argmax(axis=-1).realize().numpy()
        gold = batch["gold_digits"].numpy()
        obs  = batch["observed_mask"].numpy()
        valid_mask = batch["digit_valid_mask"].numpy()  # (B, N_MAX, N_DIGITS)

        for b in range(pred.shape[0]):
            nv = int(batch["n_vars_total"][b])
            for v in range(min(nv, N_MAX)):
                if obs[b, v] == 1:
                    continue
                # Skip padding rows (valid_mask all zero)
                if valid_mask[b, v].sum() == 0:
                    continue
                cells_total += 1
                all_ok_valid = True
                for p in range(N_DIGITS):
                    pp = int(pred[b, v, p]); gg = int(gold[b, v, p])
                    pos_total_all[p] += 1
                    if pp == gg:
                        pos_correct_all[p] += 1
                    if valid_mask[b, v, p] > 0.5:
                        pred_hist[p][pp] += 1
                        gold_hist[p][gg] += 1
                        pos_total[p]   += 1
                        if pp == gg:
                            pos_correct[p] += 1
                        else:
                            all_ok_valid = False
                if all_ok_valid:
                    cells_all_correct += 1
        nb += 1
        if nb >= N_BATCHES:
            break
    Tensor.training = True
    cell_acc = cells_all_correct / max(cells_total, 1)
    print(f"[{label}] {nb} batches, {cells_total} real unobs cells, "
          f"cell_acc(valid)={cell_acc:.3f}", flush=True)
    return (pos_correct, pos_total, pos_correct_all, pos_total_all,
            pred_hist, gold_hist, cell_acc)


def main():
    print("=== v105.2 per-position diagnostic (LSD-first) ===", flush=True)
    print(f"K={K_MAX}  n_digits={N_DIGITS}  n_batches={N_BATCHES}  batch={BATCH}",
          flush=True)
    print(f"ckpt: {CKPT}", flush=True)
    print("LSD-first: position 0 = ones, position 4 = ten-thousands", flush=True)
    print("", flush=True)

    cfg   = Config()
    sd_p  = _load_state()
    model = load_breathing(cfg, sd=sd_p); del sd_p
    cast_layers_fp32(model)
    attach_fg_params_v105_2(
        model, hidden=cfg.hidden,
        n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K_MAX,
        waist=WAIST, n_code=N_CODE,
    )
    Device[Device.DEFAULT].synchronize()
    load_ckpt_v105_2(model, CKPT)
    print("loaded ckpt", flush=True)

    train_loader = FactorGraphLoaderV105_2(
        ".cache/factor_graph_train.jsonl", batch_size=BATCH,
        difficulty_filter="easy", curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K_MAX, n_heads=N_HEADS,
        n_digits=N_DIGITS, seed=42,
    )
    val_loader = FactorGraphLoaderV105_2(
        ".cache/factor_graph_test.jsonl", batch_size=BATCH,
        difficulty_filter="easy", curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K_MAX, n_heads=N_HEADS,
        n_digits=N_DIGITS, seed=43,
    )

    t_pc, t_pt, t_pca, t_pta, tp, tg, t_cell = _collect(train_loader, model, "TRAIN")
    v_pc, v_pt, v_pca, v_pta, vp, vg, v_cell = _collect(val_loader,   model, "VAL  ")

    print("", flush=True)
    print("=== PER-POSITION ACCURACY (VALID positions only) ===", flush=True)
    print(f"  cell_acc (all VALID digits correct): train={t_cell:.3f}  val={v_cell:.3f}",
          flush=True)
    print("", flush=True)
    print(f"  {'pos':>3}  {'label':>14}  {'train':>8}  {'val':>8}  {'gap':>8}  "
          f"{'train_n':>8}  {'val_n':>8}",
          flush=True)
    print("  " + "-" * 70, flush=True)
    for p in range(N_DIGITS):
        ta = t_pc[p] / max(t_pt[p], 1)
        va = v_pc[p] / max(v_pt[p], 1)
        print(f"  {p:>3}  {POS_LABELS[p]:>14}  {ta:>8.3f}  {va:>8.3f}  "
              f"{ta-va:>+8.3f}  {int(t_pt[p]):>8}  {int(v_pt[p]):>8}",
              flush=True)

    print("", flush=True)
    print("=== PER-POSITION ACCURACY (ALL positions, incl. padding) ===", flush=True)
    print("Lower = better only if mask is working — padding should match (gold=0,pred=0)",
          flush=True)
    print(f"  {'pos':>3}  {'label':>14}  {'train':>8}  {'val':>8}", flush=True)
    print("  " + "-" * 50, flush=True)
    for p in range(N_DIGITS):
        ta = t_pca[p] / max(t_pta[p], 1)
        va = v_pca[p] / max(v_pta[p], 1)
        print(f"  {p:>3}  {POS_LABELS[p]:>14}  {ta:>8.3f}  {va:>8.3f}",
              flush=True)

    print("", flush=True)
    print("=== GOLD DIGIT DISTRIBUTION at valid positions (top 5 per pos) ===",
          flush=True)
    for p in range(N_DIGITS):
        tt_top = tg[p].most_common(5)
        vv_top = vg[p].most_common(5)
        print(f"  pos{p} ({POS_LABELS[p]}):  train={tt_top}", flush=True)
        print(f"                  val  ={vv_top}", flush=True)

    print("", flush=True)
    print("=== PREDICTED DIGIT DISTRIBUTION at valid positions (top 5) ===",
          flush=True)
    for p in range(N_DIGITS):
        tt_top = tp[p].most_common(5)
        vv_top = vp[p].most_common(5)
        print(f"  pos{p} ({POS_LABELS[p]}):  train={tt_top}", flush=True)
        print(f"                  val  ={vv_top}", flush=True)

    print("", flush=True)
    print("=== MOMENT OF TRUTH ===", flush=True)
    pos0_train = t_pc[0] / max(t_pt[0], 1)
    pos0_val   = v_pc[0] / max(v_pt[0], 1)
    print(f"Position 0 (ones digit) val accuracy: {pos0_val:.3f}", flush=True)
    if pos0_val > 0.15:
        print("  → BREAKTHROUGH: model is learning ones-place arithmetic!", flush=True)
    elif pos0_val > 0.10:
        print("  → marginal: above random but barely. More training needed.", flush=True)
    elif pos0_val > 0.05:
        print("  → not yet: position 0 is below random. Architecture not engaging.",
              flush=True)
    else:
        print("  → bad: degenerate collapse or starvation persists.", flush=True)


if __name__ == "__main__":
    main()
