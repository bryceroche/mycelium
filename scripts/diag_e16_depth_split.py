"""E.16 depth-split eval (pre-registered §1A.E.16) — decomposition's question.

Splits the existing hard test bin by graph DEPTH (longest path: leaves at 0,
factor j's result = 1 + max(arg depths)) — training mixes chains (depth =
n_factors, 6-7 at hard) and trees (~3) 50/50, so depth varies in-dist at
fixed size. Per depth bin, on a given checkpoint: cell_acc + per-breath eval
CE (descent + tail per C4'-form).

Pre-registered readings (E.16): COMPONENT-LEARNER = descent persists at
depth 6-7 (>=50% of shallow bins' descent); PATTERN-MEMORIZER = depth-7
descent < 25% of shallow or absent. Depth-gradient discriminator (per-node
read) is the sweep's follow-up; this script delivers the binding
descent-ratio read.

Usage:
  RUN_CKPT=.cache/v200_perceiver_ckpts/v200_perceiver_238_write8_step2000.safetensors \
    .venv/bin/python scripts/diag_e16_depth_split.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

from mycelium.llama_loader import (
    attach_llama_layers, load_llama_weights,
    LLAMA_3_2_1B_CFG, SMOLLM2_1_7B_CFG,
)
from mycelium.factor_graph_v200 import (
    attach_fg_params_v200, fg_v200_state_dict,
    V200_K_MAX, V200_N_MAX, V200_F_MAX, V200_N_VAR_LAT, V200_N_DIGITS,
    V200_WAIST_DIM,
)
from mycelium.factor_graph_data_v107 import FactorGraphLoaderV107

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "r2375", os.path.join(_SCRIPT_DIR, "v200_resmoke_237_5.py"))
_r = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_r)

TEST_PATH = ".cache/factor_graph_test.jsonl"
OUT_PATH  = ".cache/v200_smoke/e16_depth_split.json"
SEED      = 42


class _Obj:
    pass


def record_depth(rec: dict) -> int:
    """Longest path: leaves depth 0; factor j's result var = n_vars + j."""
    n_leaves = rec["n_vars"]
    depth = {i: 0 for i in range(n_leaves)}
    for j, args in enumerate(rec["factor_args"]):
        depth[n_leaves + j] = 1 + max(depth.get(a, 0) for a in args)
    return max(depth.values())


def main() -> None:
    K = V200_K_MAX
    ckpt = os.environ.get("RUN_CKPT")
    assert ckpt and os.path.exists(ckpt), f"RUN_CKPT missing: {ckpt}"

    # Bin HARD records by depth
    bins: dict = {}
    with open(TEST_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("difficulty") != "hard":
                continue
            d = record_depth(rec)
            bins.setdefault(d, []).append(line)
    print("hard-bin depth distribution:",
          {d: len(v) for d, v in sorted(bins.items())})

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
    sd_ck = safe_load(ckpt)
    targets = fg_v200_state_dict(model)
    for name, dst in targets.items():
        if name in sd_ck:
            src = sd_ck[name].to(dst.device).realize()
            if src.dtype != dst.dtype:
                src = src.cast(dst.dtype)
            dst.assign(src).realize()
    del sd_ck

    results = {}
    for d in sorted(bins):
        lines = bins[d]
        if len(lines) < 24:   # need a few batches of 8
            print(f"depth {d}: only {len(lines)} records — skipped")
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
            tf.writelines(lines)
            tmp = tf.name
        loader = FactorGraphLoaderV107(
            tmp, batch_size=8, difficulty_filter=None, curriculum=False,
            n_max=V200_N_MAX, f_max=V200_F_MAX, k_max=K, n_heads=16,
            seed=SEED + 2,
        )
        pb = _r._per_breath_ce_at_eval(
            model, loader, K=K, n_max=V200_N_MAX, f_max=V200_F_MAX,
            n_var_lat=V200_N_VAR_LAT, n_digits=V200_N_DIGITS,
            max_batches=8, stage2a_waist=True,
        )
        ev = _r._evaluate_v200(
            model, loader, K=K, n_max=V200_N_MAX, f_max=V200_F_MAX,
            n_var_lat=V200_N_VAR_LAT, n_digits=V200_N_DIGITS,
            max_batches=8, stage2a_waist=True,
        )
        ce = np.array(pb)
        descent = float(ce[0] - ce.min())
        tail = float(ce[-1] - ce.min())
        cell = float(ev.get("hard", {}).get("cell_acc", float("nan")))
        results[d] = {"n_records": len(lines), "per_breath_ce": pb,
                      "descent": descent, "tail_rise": tail,
                      "cell_acc_hard": cell}
        print(f"[depth {d}] n={len(lines)} descent={descent:.4f} "
              f"tail={tail:.4f} cell={cell:.3f} "
              f"ce={['%.4f' % v for v in pb]}", flush=True)
        os.unlink(tmp)

    # E.16 binding read: deep-vs-shallow descent ratio
    if results:
        ds = sorted(results)
        shallow = [results[d]["descent"] for d in ds if d <= 3]
        deep    = [results[d]["descent"] for d in ds if d >= 5]
        if shallow and deep:
            ratio = float(np.mean(deep) / (np.mean(shallow) + 1e-12))
            verdict = ("COMPONENT-LEARNER (graceful)" if ratio >= 0.5
                       else "PATTERN-MEMORIZER (cliff)" if ratio < 0.25
                       else "INTERMEDIATE")
            print(f"E.16 BINDING: deep/shallow descent ratio = {ratio:.3f} → {verdict}")
            results["_e16_binding"] = {"ratio": ratio, "verdict": verdict,
                                       "pre_commit": "graceful>=0.5; cliff<0.25"}
    with open(OUT_PATH, "w") as f:
        json.dump({"ckpt": ckpt, "bins": {str(k): v for k, v in results.items()}},
                  f, indent=2)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
