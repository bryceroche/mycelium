"""Eval script for v107 factor graph model (hybrid 200-bin codebook).

Evaluates a saved v107 checkpoint on:
  1. Synthetic test set (factor_graph_test.jsonl) — per-difficulty cell/query acc
  2. GSM8K factor graphs (gsm8k_factor_graphs_train.jsonl) — overall acc

Usage:
  V107_TASK=1 \
  CKPT=.cache/fg_v107_ckpts/v107_prod_step3000.safetensors \
  bash -c '.venv/bin/python scripts/eval_v107_factor_graph.py'

Or via env var shorthand:
  CKPT=... bash scripts/v107_factor_graph_smoke.sh  (evaluates at end of smoke)

Env vars:
  CKPT                      — checkpoint path (required)
  V107_K_MAX=10             — number of breaths
  V107_N_MAX=16
  V107_F_MAX=8
  EVAL_BATCH=8
  EVAL_BATCHES=50           — number of batches for synthetic eval
  GSM8K_EVAL_BATCHES=50     — number of batches for GSM8K eval
  V107_VAL=.cache/factor_graph_test.jsonl
  V107_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl
  PYTHIA_INIT=1
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v107 import (
    V107_K_MAX, V107_N_MAX, V107_F_MAX, V107_N_HEADS,
    V107_CODEBOOK_N, V107_IB_CENTROIDS,
    attach_fg_params_v107, fg_v107_parameters,
    fg_breathing_forward_v107,
    _compile_jit_fg_eval_v107,
    get_bin_values,
)
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, load_gsm8k_records_v107,
    _records_to_batch_v107, batch_to_tensors_v107,
)
from mycelium.factor_graph_v107 import fg_accuracy_v107
from scripts.v107_factor_graph_train import (
    cast_layers_fp32, load_ckpt_v107, DIFFICULTIES,
)


def main():
    CKPT         = getenv("CKPT", "")
    assert CKPT and os.path.exists(CKPT), f"CKPT={CKPT!r} does not exist"

    K            = int(getenv("V107_K_MAX",    str(V107_K_MAX)))
    N_MAX        = int(getenv("V107_N_MAX",    str(V107_N_MAX)))
    F_MAX        = int(getenv("V107_F_MAX",    str(V107_F_MAX)))
    N_CODE       = int(getenv("V107_CODEBOOK_N", str(V107_CODEBOOK_N)))
    IB_PATH      = getenv("V107_IB_CENTROIDS", V107_IB_CENTROIDS)
    EVAL_BATCH   = int(getenv("EVAL_BATCH",   "8"))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", "50"))
    GSM_BATCHES  = int(getenv("GSM8K_EVAL_BATCHES", "50"))
    PYTHIA_INIT  = int(getenv("PYTHIA_INIT",  "1")) > 0

    VAL_PATH     = getenv("V107_VAL",         ".cache/factor_graph_test.jsonl")
    GSM8K_PATH   = getenv("V107_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")

    print(f"=== v107 eval ===")
    print(f"ckpt={CKPT}")
    print(f"device={Device.DEFAULT}  K={K}  N_MAX={N_MAX}  F_MAX={F_MAX}")
    print()

    cfg = Config()
    if PYTHIA_INIT:
        sd = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)

    attach_fg_params_v107(model, hidden=cfg.hidden, n_max=N_MAX, f_max=F_MAX,
                          k_max=K, n_code=N_CODE, ib_centroids_path=IB_PATH)
    Device[Device.DEFAULT].synchronize()

    print(f"loading ckpt: {CKPT}")
    load_ckpt_v107(model, CKPT)

    bv = get_bin_values()

    Tensor.training = False
    eval_fn = _compile_jit_fg_eval_v107(model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX)

    # --- Synthetic eval ---
    print(f"\n[synthetic] evaluating {EVAL_BATCHES} batches × B={EVAL_BATCH}...")
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V107_N_HEADS,
        seed=99,
    )

    synth_agg = {}
    n_batches = 0
    for batch in val_loader.iter_eval(batch_size=EVAL_BATCH):
        pred_t, _ = eval_fn(
            batch["domain_init"], batch["node_kinds"],
            batch["staging_mask"], batch["head_op_mask"],
            batch["gold_bins"], batch["observed_mask"],
        )
        pred_np  = pred_t.numpy()
        gold_np  = batch["gold_bins"].numpy()
        obs_np   = batch["observed_mask"].numpy()
        picks    = batch["picks"]
        q_np     = batch["query_idx"]

        for b in range(len(picks)):
            diff = picks[b].get("difficulty", "easy")
            if diff not in synth_agg:
                synth_agg[diff] = {"n_unobs": 0, "n_correct": 0, "q_total": 0, "q_correct": 0}
            qi = int(q_np[b])
            nv = int(batch["n_vars_total"][b])
            for vi in range(min(nv, N_MAX)):
                if obs_np[b, vi] == 0:
                    synth_agg[diff]["n_unobs"] += 1
                    if pred_np[b, vi] == gold_np[b, vi]:
                        synth_agg[diff]["n_correct"] += 1
            if qi < N_MAX and pred_np[b, qi] == gold_np[b, qi]:
                synth_agg[diff]["q_correct"] += 1
            synth_agg[diff]["q_total"] += 1

        n_batches += 1
        if n_batches >= EVAL_BATCHES:
            break

    print("\nSynthetic results:")
    for d in DIFFICULTIES:
        if d not in synth_agg:
            continue
        a = synth_agg[d]
        ca = a["n_correct"] / max(a["n_unobs"], 1)
        qa = a["q_correct"] / max(a["q_total"], 1)
        print(f"  [{d:6s}] cell_acc={ca:.3f} query_acc={qa:.3f} n={a['q_total']}")

    # --- GSM8K eval ---
    print(f"\n[gsm8k] evaluating {GSM_BATCHES} batches × B={EVAL_BATCH}...")
    gsm8k_records = load_gsm8k_records_v107(GSM8K_PATH)
    if not gsm8k_records:
        print("  no GSM8K records found.")
    else:
        n_correct_unobs = 0
        n_unobs_total   = 0
        q_correct       = 0
        q_total         = 0
        n_batches       = 0

        for start in range(0, len(gsm8k_records), EVAL_BATCH):
            recs = gsm8k_records[start:start + EVAL_BATCH]
            while len(recs) < EVAL_BATCH:
                recs.append(gsm8k_records[0])
            batch = batch_to_tensors_v107(
                _records_to_batch_v107(recs, N_MAX, F_MAX, K, V107_N_HEADS)
            )
            batch["picks"] = recs

            pred_t, _ = eval_fn(
                batch["domain_init"], batch["node_kinds"],
                batch["staging_mask"], batch["head_op_mask"],
                batch["gold_bins"], batch["observed_mask"],
            )
            pred_np = pred_t.numpy()
            gold_np = batch["gold_bins"].numpy()
            obs_np  = batch["observed_mask"].numpy()
            q_np    = batch["query_idx"]
            nv_np   = batch["n_vars_total"]

            for b in range(len(recs)):
                qi = int(q_np[b])
                nv = int(nv_np[b])
                for vi in range(min(nv, N_MAX)):
                    if obs_np[b, vi] == 0:
                        n_unobs_total += 1
                        if pred_np[b, vi] == gold_np[b, vi]:
                            n_correct_unobs += 1
                if qi < N_MAX and pred_np[b, qi] == gold_np[b, qi]:
                    q_correct += 1
                q_total += 1

            n_batches += 1
            if n_batches >= GSM_BATCHES:
                break

        ca = n_correct_unobs / max(n_unobs_total, 1)
        qa = q_correct / max(q_total, 1)
        print(f"\nGSM8K results:")
        print(f"  cell_acc={ca:.3f} query_acc={qa:.3f} n_puzzles={q_total}")

    print("\ndone.")


if __name__ == "__main__":
    main()
