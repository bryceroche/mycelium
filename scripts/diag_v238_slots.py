"""#238 slot-distinctness read (§2 WRITE spec precondition, pinned pre-data).

Pairwise cosine between the K notebook slot contents on the fixed diag batch,
computed from dense ckpts via the tapped canonical forward. Pre-committed
bands: DISTINCT = mean pairwise cos <= 0.8; DEGENERATE = mean >= 0.95
("eight photocopies" — read-back worthless regardless of gating; design fork
= slot KEYING); between = PARTIAL.

Runs on CPU while the GPU trains (warm kernel cache from the diag family).

Usage:
  DEV=CPU STEPS_LIST=500 .venv/bin/python scripts/diag_v238_slots.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

from mycelium.llama_loader import (
    attach_llama_layers, load_llama_weights,
    LLAMA_3_2_1B_CFG, SMOLLM2_1_7B_CFG,
)
from mycelium.factor_graph_v200 import (
    attach_fg_params_v200, fg_v200_state_dict, fg_breathing_forward_v200,
    fg_v200_empty_taps,
    V200_K_MAX, V200_N_MAX, V200_F_MAX, V200_N_VAR_LAT, V200_N_DIGITS,
    V200_WAIST_DIM,
)
from mycelium.factor_graph_data_v107 import FactorGraphLoaderV107

CKPT_GLOB = ".cache/v200_perceiver_ckpts/v200_perceiver_238_write8_step*.safetensors"
OUT_PATH  = ".cache/v200_smoke/slot_distinctness_238.json"
SEED      = 42


class _Obj:
    pass


def main() -> None:
    K = V200_K_MAX
    steps_env = os.environ.get("STEPS_LIST", "500")
    wanted = {int(s) for s in steps_env.split(",")}
    ckpts = sorted(
        (c for c in glob.glob(CKPT_GLOB)
         if int(re.search(r"step(\d+)\.safetensors", c).group(1)) in wanted),
        key=lambda p: int(re.search(r"step(\d+)\.safetensors", p).group(1)))
    assert ckpts, f"no ckpts matched {CKPT_GLOB} for {steps_env}"

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)
    llama32_path = ".cache/llama-3.2-1b-weights/model.safetensors"
    if os.path.exists(llama32_path):
        sd = safe_load(llama32_path); cfg = LLAMA_3_2_1B_CFG
    else:
        sd = load_llama_weights(); cfg = SMOLLM2_1_7B_CFG
    model = _Obj()
    attach_llama_layers(model, n_layers=4, sd=sd, cfg=cfg)
    del sd
    attach_fg_params_v200(
        model, n_latents=32, n_var_lat=V200_N_VAR_LAT, k_max=K,
        n_digits=V200_N_DIGITS, n_max=V200_N_MAX, f_max=V200_F_MAX,
        stage2a_waist=True, waist_dim=V200_WAIST_DIM,
    )
    for layer in model.llama_layers:
        for p in layer.parameters():
            if p.dtype != dtypes.float:
                p.assign(p.cast(dtypes.float)).realize()

    val_loader = FactorGraphLoaderV107(
        ".cache/factor_graph_test.jsonl", batch_size=8,
        difficulty_filter=None, curriculum=False,
        n_max=V200_N_MAX, f_max=V200_F_MAX, k_max=K, n_heads=16, seed=SEED + 2,
    )
    batch = next(val_loader.iter_eval())

    targets = fg_v200_state_dict(model)
    results = []
    for ck in ckpts:
        step = int(re.search(r"step(\d+)\.safetensors", ck).group(1))
        sd_ck = safe_load(ck)
        for name, dst in targets.items():
            if name in sd_ck:
                src = sd_ck[name].to(dst.device).realize()
                if src.dtype != dst.dtype:
                    src = src.cast(dst.dtype)
                dst.assign(src).realize()
        del sd_ck

        was_training = Tensor.training
        Tensor.training = False
        taps = fg_v200_empty_taps()
        fg_breathing_forward_v200(
            model, batch["domain_init"], batch["node_kinds"], K=K,
            n_max=V200_N_MAX, f_max=V200_F_MAX,
            training=False, stage2a_waist=True, taps=taps,
        )
        Tensor.training = was_training

        slots = [np.asarray(s.numpy(), dtype=np.float64) for s in taps["nb_slots"]]  # K × (B, H)
        S = np.stack(slots, axis=0)                       # (K, B, H)
        # Slot amplitude watch (tripwire-13 audit: W_write output enters
        # storage unnormed; read-back output is detached-normed → slot scale
        # is loss-visible only via softmax sharpness)
        slot_norms = [float(np.linalg.norm(s, axis=-1).mean()) for s in slots]
        Sn = S / (np.linalg.norm(S, axis=-1, keepdims=True) + 1e-12)
        # pairwise cosine between slots, per batch item, then averaged
        cos = np.einsum('kbh,lbh->klb', Sn, Sn)           # (K, K, B)
        iu = np.triu_indices(len(slots), k=1)
        pairwise = cos[iu[0], iu[1], :]                   # (n_pairs, B)
        mean_cos = float(pairwise.mean())
        min_cos = float(pairwise.min())
        if mean_cos >= 0.95:
            band = "DEGENERATE (photocopies) — design fork: slot KEYING"
        elif mean_cos <= 0.8:
            band = "DISTINCT"
        else:
            band = "PARTIAL"
        # adjacent-slot cosine (k, k+1) — differentiation structure
        adj = [float(cos[k, k + 1, :].mean()) for k in range(len(slots) - 1)]
        entry = {"step": step, "mean_pairwise_cos": mean_cos,
                 "min_pairwise_cos": min_cos, "adjacent_cos": adj,
                 "slot_norms": slot_norms,
                 # torsion-diag v1.1 dependency: raw slot vectors (batch-mean)
                 # persist so twist-direction-vs-slot-subspace overlap is
                 # computable post-hoc (P3-proper)
                 "slot_vectors_batchmean": [s.mean(axis=0).tolist() for s in slots],
                 "band": band,
                 "bands_pre_committed": "DISTINCT<=0.8; DEGENERATE>=0.95; else PARTIAL"}
        results.append(entry)
        print(f"[step {step:5d}] slot mean-cos={mean_cos:.4f} min={min_cos:.4f} "
              f"adjacent={['%.3f' % a for a in adj]}  norms={['%.1f' % n for n in slot_norms]}  {band}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump({"run_id": "238", "checkpoints": results}, f, indent=2)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
