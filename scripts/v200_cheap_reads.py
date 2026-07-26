"""v200_cheap_reads.py — INSTRUMENT-VALIDATION reads on the weak ckpt
(2026-07-26; scope double-carved at 0ac231b: characterizes an UNDERTRAINED
ARTIFACT, validates instrumentation; CANNOT-LICENSE register applies — no
number here speaks for the architecture)."""
import sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import json
import numpy as np
from tinygrad import Tensor
from tinygrad.nn.state import safe_load
from mycelium.llama_loader import load_llama_weights, SMOLLM2_1_7B_CFG
from mycelium.llama_loader import attach_llama_layers
from mycelium.factor_graph_v200_runera import (
    attach_fg_params_v200, fg_v200_state_dict, fg_breathing_forward_v200,
    V200_K_MAX, V200_N_MAX, V200_F_MAX, V200_N_LATENTS, V200_N_VAR_LAT,
    V200_N_DIGITS, V200_STAGE2A_WAIST, V200_WAIST_DIM,
)
from mycelium.factor_graph_data_v107 import FactorGraphLoaderV107
from mycelium.factor_graph_v108 import bins_to_digits_msd
from dart_cluster_probe import auc_mann_whitney

K = V200_K_MAX; BATCH = 8
class M: pass
model = M()
sd_ll = load_llama_weights()
attach_llama_layers(model, n_layers=4, sd=sd_ll, cfg=SMOLLM2_1_7B_CFG, layer_offset=0)
attach_fg_params_v200(model, n_latents=V200_N_LATENTS, n_var_lat=V200_N_VAR_LAT,
    k_max=K, n_digits=V200_N_DIGITS, n_max=V200_N_MAX, f_max=V200_F_MAX,
    stage2a_waist=True, waist_dim=V200_WAIST_DIM)
sd = safe_load(".cache/fg_v200_ckpts/v200_run_final.safetensors")
tg = fg_v200_state_dict(model); loaded = 0
for n, d in tg.items():
    if n in sd:
        s = sd[n].to(d.device).realize()
        d.assign(s.cast(d.dtype) if s.dtype != d.dtype else s).realize(); loaded += 1
consumed = sum(1 for n in tg if n in sd)
assert consumed == len(sd), f"consumed {consumed} != ckpt {len(sd)}"
print(f"[v200-reads] extras at init: {sorted(set(tg)-set(sd))}", flush=True)
print(f"[v200-reads] loaded {loaded} params (all ckpt keys consumed)", flush=True)

val = FactorGraphLoaderV107(".cache/factor_graph_test.jsonl", batch_size=BATCH,
    n_max=V200_N_MAX, f_max=V200_F_MAX, difficulty_filter=None, curriculum=False)
acc_curve = np.zeros(K); acc_n = 0
w2r = r2w = flip_cells = tot_cells = 0
calib_scores = [[] for _ in range(K)]; calib_lab = [[] for _ in range(K)]
nb = 0
for batch in val.iter_eval():
    hist, calib = fg_breathing_forward_v200(model, batch["domain_init"], batch["node_kinds"],
        K=K, training=False, stage2a_waist=True)
    gold_d = bins_to_digits_msd(batch["gold_bins"].numpy(), n_digits=V200_N_DIGITS)
    vmask = batch["var_mask"].numpy().astype(bool) if "var_mask" in batch else np.ones(gold_d.shape[:2], bool)
    per_k_correct = []
    for k in range(K):
        pd = hist[k].argmax(axis=-1).realize().numpy()
        corr = (pd == gold_d).all(axis=-1) & vmask
        per_k_correct.append(corr)
        acc_curve[k] += corr.sum()
        item_corr = np.array([corr[b][vmask[b]].mean() if vmask[b].any() else 0.0 for b in range(corr.shape[0])])
        calib_scores[k].extend(list(calib[k].realize().numpy())); calib_lab[k].extend(list(item_corr > 0.5))
    acc_n += vmask.sum()
    C = np.stack(per_k_correct)                     # (K, B, N)
    for b in range(C.shape[1]):
        for v in np.where(vmask[b])[0]:
            tr = C[:, b, v]; tot_cells += 1
            d = np.diff(tr.astype(int))
            if (d != 0).any(): flip_cells += 1
            w2r += int((d == 1).sum()); r2w += int((d == -1).sum())
    nb += 1
    if nb >= 12: break
acc_curve /= max(acc_n, 1)
print("[read a] per-breath acc curve: " + " ".join(f"{a:.3f}" for a in acc_curve))
print(f"[read b] flip stats over {tot_cells} var-cells: flip_cells {flip_cells} ({flip_cells/max(tot_cells,1):.3f}) | wrong->right {w2r} | right->wrong {r2w}")
aucs = []
for k in range(K):
    s = np.array(calib_scores[k]); y = np.array(calib_lab[k])
    aucs.append(float(auc_mann_whitney(s, y)) if 0 < y.sum() < len(y) else None)
print("[read c] calib-vs-correct AUC by breath: " + " ".join(f"{a:.3f}" if a else "--" for a in aucs))
print("\n=== INSTRUMENT VERDICTS (undertrained artifact; cannot-license register applies) ===")
print(f"  clock machinery: {'PRODUCES CURVES' if acc_curve.max() > acc_curve.min() else 'FLAT'} | flip-detector: {'FIRES' if flip_cells else 'no events'} | free structure: {'calib AUC above 0.6 somewhere' if any(a and a > 0.6 for a in aucs) else 'none above 0.6'}")
json.dump({"acc_curve": list(acc_curve), "flips": {"cells": tot_cells, "flip_cells": flip_cells, "w2r": w2r, "r2w": r2w},
           "calib_auc": aucs, "SCOPE": "UNDERTRAINED ARTIFACT — instrument validation only; cannot-license register 0ac231b"},
          open(".cache/v200_cheap_reads.json", "w"), indent=1)
print("[v200-reads] banked -> .cache/v200_cheap_reads.json")
