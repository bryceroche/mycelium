"""Per-position accuracy diagnostic for v105.3 (LSD-first array layout).

LSD-first interpretation:
  position 0 = ONES (ALWAYS valid for any number) ← moment of truth
  position 1 = tens (valid for ≥ 10)
  position 2 = hundreds (valid for ≥ 100)
  position 3 = thousands (valid for ≥ 1000)
  position 4 = ten-thousands (valid for ≥ 10000)

With LSD-first natural layout, the ones digit (array index 0) gets RoPE
position 0 (no rotation) directly — no reversal patch.  We expect position 0
(ones) to be the most learned position, mirroring how v105.1.2 v2's MSD layout
made array index 4 (ones via right-aligned RoPE) the most learned position.
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
from mycelium.factor_graph_v105_5 import (
    fg_breathing_forward_v105_5,
    attach_fg_params_v105_5,
    load_ckpt_v105_5,
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


K_MAX     = int(getenv("V105_5_K_MAX", "8"))
N_DIGITS  = int(getenv("V105_5_N_DIGITS", "5"))
N_MAX     = int(getenv("V105_5_N_MAX", "16"))
F_MAX     = int(getenv("V105_5_F_MAX", "8"))
WAIST     = int(getenv("V105_5_WAIST", "512"))
N_CODE    = int(getenv("V105_5_CODEBOOK_N", "32"))
N_HEADS   = 16
N_BATCHES = int(getenv("N_BATCHES", "30"))
BATCH     = int(getenv("BATCH", "4"))
CKPT      = getenv("CKPT", ".cache/fg_v105_5_ckpts/v105_5_smoke_final.safetensors")
TRAIN_PATH = getenv("TRAIN_PATH", ".cache/factor_graph_train.jsonl")
VAL_PATH   = getenv("VAL_PATH",   ".cache/factor_graph_test.jsonl")
DIFF_FILTER = getenv("DIFF_FILTER", "easy") or None

# LSD-first labels: position 0 = ones digit (most learned, anchored at RoPE pos 0).
POS_LABELS = ["ones", "tens", "hundreds", "thousands", "ten-thousands"]


def _collect(loader, model, label):
    Tensor.training = False
    pos_correct = np.zeros(N_DIGITS, dtype=np.int64)
    pos_total   = np.zeros(N_DIGITS, dtype=np.int64)
    pred_hist   = [Counter() for _ in range(N_DIGITS)]
    gold_hist   = [Counter() for _ in range(N_DIGITS)]
    cells_all_correct = 0
    cells_total = 0
    # NEW v105.5: magnitude prediction tracking
    N_MAG = 4
    mag_confusion = np.zeros((N_MAG, N_MAG), dtype=np.int64)   # (gold, pred)
    mag_correct   = 0
    mag_total     = 0
    nb = 0
    for batch in loader.iter_eval(batch_size=loader.batch_size):
        dig_lh, _, _, mag_lh, _, _ = fg_breathing_forward_v105_5(
            model, batch["digit_init"], batch["node_kinds"],
            batch["staging_mask"], batch["head_op_mask"],
            K=K_MAX, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )
        pred = dig_lh[-1].argmax(axis=-1).realize().numpy()
        mag_pred = mag_lh[-1].argmax(axis=-1).realize().numpy()    # (B, N_MAX)
        gold = batch["gold_digits"].numpy()
        obs  = batch["observed_mask"].numpy()
        valid_mask = batch["digit_valid_mask"].numpy()
        mag_target = batch["magnitude_target"].numpy()

        for b in range(pred.shape[0]):
            nv = int(batch["n_vars_total"][b])
            for v in range(min(nv, N_MAX)):
                if obs[b, v] == 1:
                    continue
                if valid_mask[b, v].sum() == 0:
                    continue
                cells_total += 1
                all_ok_valid = True
                for p in range(N_DIGITS):
                    if valid_mask[b, v, p] > 0.5:
                        pp = int(pred[b, v, p]); gg = int(gold[b, v, p])
                        pred_hist[p][pp] += 1
                        gold_hist[p][gg] += 1
                        pos_total[p]   += 1
                        if pp == gg:
                            pos_correct[p] += 1
                        else:
                            all_ok_valid = False
                if all_ok_valid:
                    cells_all_correct += 1

                # Magnitude tracking on unobserved real cells
                mg = int(mag_target[b, v])
                mp = int(mag_pred[b, v])
                if 0 <= mg < N_MAG and 0 <= mp < N_MAG:
                    mag_confusion[mg, mp] += 1
                    mag_total += 1
                    if mg == mp:
                        mag_correct += 1
        nb += 1
        if nb >= N_BATCHES:
            break
    Tensor.training = True
    cell_acc = cells_all_correct / max(cells_total, 1)
    mag_acc  = mag_correct / max(mag_total, 1)
    print(f"[{label}] {nb} batches, {cells_total} real unobs cells, "
          f"cell_acc(valid)={cell_acc:.3f}  mag_acc={mag_acc:.3f}", flush=True)
    return pos_correct, pos_total, pred_hist, gold_hist, cell_acc, mag_confusion, mag_acc


def main():
    print("=== v105.5 per-position + magnitude diagnostic (LSD-first) ===",
          flush=True)
    print(f"K={K_MAX}  n_digits={N_DIGITS}  n_batches={N_BATCHES}  batch={BATCH}",
          flush=True)
    print(f"ckpt: {CKPT}", flush=True)
    print("LSD-first layout: position 0 = ONES, position 4 = ten-thousands",
          flush=True)
    print("Natural RoPE: ones digit (idx 0) at RoPE pos 0 (no rotation)",
          flush=True)
    print("", flush=True)

    cfg   = Config()
    sd_p  = _load_state()
    model = load_breathing(cfg, sd=sd_p); del sd_p
    cast_layers_fp32(model)
    attach_fg_params_v105_5(
        model, hidden=cfg.hidden,
        n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K_MAX,
        waist=WAIST, n_code=N_CODE,
    )
    Device[Device.DEFAULT].synchronize()
    load_ckpt_v105_5(model, CKPT)
    print("loaded ckpt", flush=True)

    train_loader = FactorGraphLoaderV105_5(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFF_FILTER, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K_MAX, n_heads=N_HEADS,
        n_digits=N_DIGITS, seed=42,
    )
    val_loader = FactorGraphLoaderV105_5(
        VAL_PATH, batch_size=BATCH,
        difficulty_filter=DIFF_FILTER, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K_MAX, n_heads=N_HEADS,
        n_digits=N_DIGITS, seed=43,
    )

    tc, tt, tp, tg, t_cell, t_mag_conf, t_mag_acc = _collect(train_loader, model, "TRAIN")
    vc, vt, vp, vg, v_cell, v_mag_conf, v_mag_acc = _collect(val_loader,   model, "VAL  ")

    print("", flush=True)
    print("=== PER-POSITION ACCURACY (VALID positions only) ===", flush=True)
    print(f"  cell_acc (all VALID digits correct): train={t_cell:.3f}  val={v_cell:.3f}",
          flush=True)
    print("", flush=True)
    print(f"  {'pos':>3}  {'label':>14}  {'train':>8}  {'val':>8}  {'gap':>8}  "
          f"{'train_n':>8}  {'val_n':>8}", flush=True)
    print("  " + "-" * 70, flush=True)
    for p in range(N_DIGITS):
        ta = tc[p] / max(tt[p], 1)
        va = vc[p] / max(vt[p], 1)
        print(f"  {p:>3}  {POS_LABELS[p]:>14}  {ta:>8.3f}  {va:>8.3f}  "
              f"{ta-va:>+8.3f}  {int(tt[p]):>8}  {int(vt[p]):>8}",
              flush=True)

    print("", flush=True)
    print("=== GOLD DIGIT DISTRIBUTION at valid positions (top 5 per pos) ===",
          flush=True)
    for p in range(N_DIGITS):
        tt_top = tg[p].most_common(5)
        vv_top = vg[p].most_common(5)
        print(f"  pos{p} ({POS_LABELS[p]}):  train={tt_top}", flush=True)
        print(f"                     val  ={vv_top}", flush=True)

    print("", flush=True)
    print("=== PREDICTED DIGIT DISTRIBUTION at valid positions (top 5) ===",
          flush=True)
    for p in range(N_DIGITS):
        tt_top = tp[p].most_common(5)
        vv_top = vp[p].most_common(5)
        print(f"  pos{p} ({POS_LABELS[p]}):  train={tt_top}", flush=True)
        print(f"                     val  ={vv_top}", flush=True)

    # NEW v105.5: magnitude accuracy + confusion matrix
    print("", flush=True)
    print("=== MAGNITUDE HEAD ACCURACY (NEW v105.5) ===", flush=True)
    print(f"  train_mag_acc = {t_mag_acc:.3f}", flush=True)
    print(f"  val_mag_acc   = {v_mag_acc:.3f}", flush=True)
    mag_labels = ["1-digit", "2-digit", "3-digit", "4+ digit"]
    for name, conf in [("TRAIN", t_mag_conf), ("VAL  ", v_mag_conf)]:
        print(f"  --- {name} magnitude confusion matrix (rows=gold, cols=pred) ---",
              flush=True)
        header = "    " + " ".join(f"{lbl:>10s}" for lbl in mag_labels)
        print(header, flush=True)
        for g in range(4):
            row = " ".join(f"{int(conf[g, p]):>10d}" for p in range(4))
            print(f"    [{mag_labels[g]:>10s}] {row}", flush=True)

    print("", flush=True)
    print("=== MOMENT OF TRUTH (position 0 = ones digit in LSD layout) ===", flush=True)
    pos_ones_train = tc[0] / max(tt[0], 1)
    pos_ones_val   = vc[0] / max(vt[0], 1)
    print(f"Position 0 (ones digit) train acc: {pos_ones_train:.3f}",
          flush=True)
    print(f"Position 0 (ones digit) val acc:   {pos_ones_val:.3f}",
          flush=True)
    print(f"  v105.2 (LSD) reference at 50 steps: pos0 ones val = 0.142",
          flush=True)
    if pos_ones_val > 0.15:
        print("  → BREAKTHROUGH: real ones-place learning", flush=True)
    elif pos_ones_val > 0.10:
        print("  → marginal: above random; matches v105.2 region", flush=True)
    else:
        print("  → degenerate collapse OR architecture not engaging", flush=True)


if __name__ == "__main__":
    main()
