# FROZEN HISTORICAL (pre-#237 mask1a): the shared module mycelium/factor_graph_v200.py
# now attaches the §2 latent topology mask UNCONDITIONALLY. Re-running this script
# trains/evals WITH the mask and will NOT reproduce the original run; this script's
# arch_version/config_sig strings predate mask1a and would misreport the architecture.
# The original artifacts (+ metric_sha content hashes) are the record. (#237 review, Jun 11)
"""5-minute diagnostic: does the eval path produce the same accuracy as training?

Loads v200_smoke_2a_v3_final.safetensors and runs the EVAL forward pass on a
batch of TRAINING examples. Compares to what we'd expect from train cell_acc 0.66.

Three possible outcomes:
  Train batch via EVAL path: ~0.66 → eval path matches training, real gen gap
  Train batch via EVAL path: ~0.04 → eval path differs from train path, bug
  Train batch via EVAL path: ~0.30 → partial overfit or partial bug
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.state import safe_load

from mycelium.llama_loader import (
    attach_llama_layers, LlamaConfig, SMOLLM2_1_7B_CFG, load_llama_weights,
)
from mycelium.factor_graph_v200 import (
    V200_K_MAX, V200_N_LATENTS, V200_N_VAR_LAT, V200_N_DIGITS,
    V200_N_MAX, V200_F_MAX,
    V200_STAGE2A_WAIST, V200_WAIST_DIM,
    attach_fg_params_v200, fg_v200_state_dict,
    compile_jit_eval_v200,
)
from mycelium.factor_graph_v108 import bins_to_digits_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, DualDataLoaderV107, load_gsm8k_records_v107,
)


def main():
    K = V200_K_MAX
    BATCH = 8
    N_VAR_LAT = V200_N_VAR_LAT
    N_DIGITS = V200_N_DIGITS

    ckpt_path = ".cache/fg_v200_ckpts/v200_smoke_2a_v3_final.safetensors"
    print(f"Loading ckpt: {ckpt_path}")

    # Build model (same as training)
    class V200Model:
        pass
    model = V200Model()

    cfg = SMOLLM2_1_7B_CFG
    print("Loading Llama weights...")
    sd_llama = load_llama_weights()
    attach_llama_layers(model, n_layers=4, sd=sd_llama, cfg=cfg, layer_offset=0)

    attach_fg_params_v200(
        model,
        n_latents=V200_N_LATENTS, n_var_lat=N_VAR_LAT, k_max=K,
        n_digits=N_DIGITS, n_max=V200_N_MAX, f_max=V200_F_MAX,
        stage2a_waist=V200_STAGE2A_WAIST, waist_dim=V200_WAIST_DIM,
    )

    # Load ckpt
    sd = safe_load(ckpt_path)
    targets = fg_v200_state_dict(model)
    loaded = 0
    for name, dst in targets.items():
        if name in sd:
            src = sd[name].to(dst.device).realize()
            if src.dtype != dst.dtype:
                src = src.cast(dst.dtype)
            dst.assign(src).realize()
            loaded += 1
    print(f"Loaded {loaded}/{len(targets)} params from ckpt")

    # Compile eval JIT
    eval_fn = compile_jit_eval_v200(
        model, K=K, B=BATCH,
        n_max=V200_N_MAX, f_max=V200_F_MAX,
        n_var_lat=N_VAR_LAT, n_digits=N_DIGITS,
        stage2a_waist=V200_STAGE2A_WAIST,
    )

    # Load TRAIN dataset (same as training script)
    train_loader = FactorGraphLoaderV107(
        ".cache/factor_graph_train.jsonl",
        batch_size=BATCH, n_max=V200_N_MAX, f_max=V200_F_MAX,
        difficulty_filter=None, curriculum=False,
    )
    val_loader = FactorGraphLoaderV107(
        ".cache/factor_graph_test.jsonl",
        batch_size=BATCH, n_max=V200_N_MAX, f_max=V200_F_MAX,
        difficulty_filter=None, curriculum=False,
    )

    print("\n=== Train batch via EVAL path ===")
    # Sample a training batch and run through eval_fn
    Tensor.training = False
    n_correct_per_diff = {}
    n_total_per_diff = {}
    n_batches = 3
    for batch_idx in range(n_batches):
        batch = train_loader.sample_batch(step=batch_idx)
        outs = eval_fn(batch["domain_init"], batch["node_kinds"])
        final_logits = outs[0]
        pred_digits_np = final_logits.argmax(axis=-1).realize().numpy()

        gold_bins_np = batch["gold_bins"].numpy()
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=N_DIGITS)
        obs_np = batch["observed_mask"].numpy()
        picks = batch["picks"]

        for b in range(len(picks)):
            diff = picks[b].get("difficulty", "easy")
            if diff not in n_correct_per_diff:
                n_correct_per_diff[diff] = 0
                n_total_per_diff[diff] = 0
            nv = int(batch.get("n_vars_total", [N_VAR_LAT]*len(picks))[b])
            for vi in range(min(nv, N_VAR_LAT)):
                if obs_np[b, vi] == 0:
                    n_total_per_diff[diff] += 1
                    if (pred_digits_np[b, vi] == gold_digits_np[b, vi]).all():
                        n_correct_per_diff[diff] += 1

    for diff in sorted(n_correct_per_diff.keys()):
        n = n_correct_per_diff[diff]
        t = n_total_per_diff[diff]
        print(f"  TRAIN data via eval path  [{diff}]: cell_acc = {n/max(t,1):.3f}  ({n}/{t})")

    print("\n=== Val batch via EVAL path (baseline) ===")
    n_correct_per_diff = {}
    n_total_per_diff = {}
    for batch in val_loader.iter_eval(batch_size=BATCH):
        outs = eval_fn(batch["domain_init"], batch["node_kinds"])
        final_logits = outs[0]
        pred_digits_np = final_logits.argmax(axis=-1).realize().numpy()
        gold_bins_np = batch["gold_bins"].numpy()
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=N_DIGITS)
        obs_np = batch["observed_mask"].numpy()
        picks = batch["picks"]
        for b in range(len(picks)):
            diff = picks[b].get("difficulty", "easy")
            if diff not in n_correct_per_diff:
                n_correct_per_diff[diff] = 0
                n_total_per_diff[diff] = 0
            nv = int(batch.get("n_vars_total", [N_VAR_LAT]*len(picks))[b])
            for vi in range(min(nv, N_VAR_LAT)):
                if obs_np[b, vi] == 0:
                    n_total_per_diff[diff] += 1
                    if (pred_digits_np[b, vi] == gold_digits_np[b, vi]).all():
                        n_correct_per_diff[diff] += 1
        if sum(n_total_per_diff.values()) > 30:
            break

    for diff in sorted(n_correct_per_diff.keys()):
        n = n_correct_per_diff[diff]
        t = n_total_per_diff[diff]
        print(f"  VAL   data via eval path  [{diff}]: cell_acc = {n/max(t,1):.3f}  ({n}/{t})")

    print("\n=== Diagnosis ===")
    print("If TRAIN-via-eval ≈ 0.66 and VAL-via-eval ≈ 0.04: real generalization gap")
    print("If TRAIN-via-eval ≈ 0.04 and VAL-via-eval ≈ 0.04: eval path bug (matches gold differently)")
    print("If TRAIN-via-eval ≈ 0.30 and VAL-via-eval ≈ 0.04: partial — both real gap AND something off")


if __name__ == "__main__":
    main()
