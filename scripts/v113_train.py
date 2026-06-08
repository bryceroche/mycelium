"""v113 training driver — per-digit input tokens + v112b topology gate.

Architecture change from v112b:
  - Per-digit input tokens (N_MAX * n_digits separate tokens for variables)
  - RIGHT-ALIGNED RoPE (ones digit = rope_pos 0, frozen sinusoidal tables)
  - valid_mask: padding leading zeros blocked as keys in attention
  - T_new = N_MAX * n_digits + F_MAX  (default 16*5+8 = 88)
  - v112b topology tensor resized to T_new

Cold-start only: v112b ckpt shapes are incompatible.

Env vars (key ones):
  V113_N_MAX          — number of variable slots (default 16)
  V113_F_MAX          — number of factor slots (default 8)
  V113_K_MAX          — number of breaths (default 8)
  V113_N_DIGITS       — digits per variable (default 5)
  V113_WAIST_DIM      — waist dimension (default 512)
  V110_STEP3_*        — shared training hyperparams (same as v112b)
  BATCH               — batch size (default 8)
  STEPS               — training steps (default 100)
  LR                  — learning rate (default 3e-5)
"""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, safe_load

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.factor_graph_v110_step3 import (
    V110_STEP3_K_MAX, V110_STEP3_N_MAX, V110_STEP3_F_MAX, V110_STEP3_N_HEADS,
    V110_STEP3_N_DIGITS,
    V110_STEP3_WAIST_DIM, V110_STEP3_ALTERNATION, V110_STEP3_HARD_BREATH_LEVEL,
    V110_STEP3_VAR_LOSS_WEIGHT, V110_STEP3_CALIB_WEIGHT,
    V110_STEP3_FACTOR_AUX_WEIGHT, V110_STEP3_BALANCE_WEIGHT,
    V110_STEP3_UNCERTAINTY_MIN,
    V110_STEP3_CODEBOOK_N, V110_STEP3_IB_CENTROIDS, V110_STEP3_PHASE_SCALE,
    V110_STEP3_GATE_PROFILE, V110_STEP3_PHOTON_ALPHA,
)
from mycelium.factor_graph_v112b import (
    V112B_TOPOLOGY_DIM,
)
from mycelium.factor_graph_v113 import (
    V113_N_MAX, V113_F_MAX, V113_K_MAX, V113_N_DIGITS,
    V113_WAIST_DIM, V113_CODEBOOK_N, V113_IB_CENTROIDS, V113_TOPOLOGY_DIM,
    attach_fg_params_v113,
    fg_v113_parameters, fg_v113_state_dict,
    _compile_jit_fg_step_v113, compile_jit_eval_v113,
    build_digit_init_v113, build_valid_mask_np,
    _expand_mask_fast,
)
from mycelium.factor_graph_v108 import bins_to_digits_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, DualDataLoaderV107, load_gsm8k_records_v107,
)
from scripts.v108b_factor_graph_train import cast_layers_fp32

DIFFICULTIES = ["easy", "medium", "hard"]


def evaluate_v113(
    model, val_loader, K: int, max_batches: int = 5, eval_fn=None,
    n_max: int = V113_N_MAX, f_max: int = V113_F_MAX, n_digits: int = V113_N_DIGITS,
    n_heads: int = V110_STEP3_N_HEADS, k_max_data: int = V113_K_MAX,
) -> dict:
    Tensor.training = False
    agg: dict = {}
    n_batches = 0
    T_new = n_max * n_digits + f_max
    T_old = n_max + f_max

    for batch in val_loader.iter_eval():
        if n_batches >= max_batches:
            break
        n_batches += 1

        gold_bins_np  = batch["gold_bins"].numpy()
        obs_mask_np   = batch["observed_mask"].numpy()
        gold_vals_np  = batch.get("gold_values", batch["gold_bins"]).numpy()

        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=n_digits)

        # Build per-digit init + valid_mask
        di_np = build_digit_init_v113(gold_vals_np, obs_mask_np, n_digits=n_digits)
        vm_np = build_valid_mask_np(gold_vals_np, obs_mask_np, n_digits=n_digits)

        # Expand masks: T_old → T_new
        stg_np = batch["staging_mask"].numpy()   # (B, K, T_old, T_old)
        hop_np = batch["head_op_mask"].numpy()   # (B, N_HEADS, T_old, T_old)
        stg_new_np = _expand_mask_fast(stg_np,  n_max, f_max, n_digits)
        hop_new_np = _expand_mask_fast(hop_np,  n_max, f_max, n_digits)

        # node_kinds for T_new layout
        nk_old = batch["node_kinds"].numpy()     # (B, T_old) — 0=obs_var, 1=unobs_var, 2=fac
        B = nk_old.shape[0]
        nk_new = np.full((B, T_new), -1, dtype=np.int32)
        for b in range(B):
            for v in range(n_max):
                kind = int(nk_old[b, v])
                for p in range(n_digits):
                    nk_new[b, v * n_digits + p] = kind
            for fi in range(f_max):
                nk_new[b, n_max * n_digits + fi] = int(nk_old[b, n_max + fi])
        nk_new = np.clip(nk_new, 0, 2)

        # Tensors
        di_t  = Tensor(di_np,  dtype=dtypes.float).contiguous().realize()
        vm_t  = Tensor(vm_np.astype(np.float32), dtype=dtypes.float).contiguous().realize()
        nk_t  = Tensor(nk_new, dtype=dtypes.int).contiguous().realize()
        stg_t = Tensor(stg_new_np, dtype=dtypes.float).contiguous().realize()
        hop_t = Tensor(hop_new_np, dtype=dtypes.float).contiguous().realize()
        gd_t  = Tensor(gold_digits_np.astype(np.int64), dtype=dtypes.int).contiguous().realize()
        om_t  = batch["observed_mask"]

        if eval_fn is not None:
            pred_digits, cell_acc = eval_fn(di_t, vm_t, nk_t, stg_t, hop_t, gd_t, om_t)
        else:
            from mycelium.factor_graph_v113 import fg_breathing_forward_v113
            from mycelium.breathing import _layernorm
            T_noise = Tensor.zeros(K, B, T_new, 1024, dtype=dtypes.half).contiguous().realize()
            T_ns    = Tensor(np.array([0.0], dtype=np.float16), dtype=dtypes.half).contiguous().realize()
            tree_lh, _, _, _, _ = fg_breathing_forward_v113(
                model, di_t, vm_t, nk_t, stg_t, hop_t, T_noise, T_ns,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )
            pred_digits = tree_lh[-1].argmax(axis=-1)
            eq_per_pos  = (pred_digits == gd_t).cast(dtypes.float)
            eq          = eq_per_pos.prod(axis=-1)
            unobs       = (1 - om_t.cast(dtypes.float))
            cell_acc    = (eq * unobs).sum() / (unobs.sum() + 1e-8)

        ca = float(cell_acc.numpy()) if hasattr(cell_acc, "numpy") else float(cell_acc)

        picks = batch.get("picks", [])
        for pick in picks:
            d = pick.get("difficulty", "easy")
            if d not in agg:
                agg[d] = {"cell_acc_sum": 0.0, "n": 0}
            agg[d]["cell_acc_sum"] += ca
            agg[d]["n"] += 1

    Tensor.training = True
    results = {}
    for d, v in agg.items():
        if v["n"] > 0:
            results[d] = {"cell_acc": v["cell_acc_sum"] / v["n"], "n_puzzles": v["n"]}
    return results


def main():
    V113_TASK_LOCAL = int(getenv("V113_TASK", "0")) > 0
    assert V113_TASK_LOCAL, "V113_TASK=1 must be set"

    K          = int(getenv("V113_K_MAX",         str(V113_K_MAX)))
    N_MAX      = int(getenv("V113_N_MAX",          str(V113_N_MAX)))
    F_MAX      = int(getenv("V113_F_MAX",          str(V113_F_MAX)))
    N_DIGITS   = int(getenv("V113_N_DIGITS",       str(V113_N_DIGITS)))
    WAIST_DIM  = int(getenv("V113_WAIST_DIM",      str(V113_WAIST_DIM)))
    N_CODE     = int(getenv("V113_CODEBOOK_N",     str(V113_CODEBOOK_N)))
    IB_PATH    = getenv("V113_IB_CENTROIDS",       V113_IB_CENTROIDS)
    TOPO_DIM   = int(getenv("V113_TOPOLOGY_DIM",   str(V113_TOPOLOGY_DIM)))

    BATCH      = int(getenv("BATCH",               "8"))
    STEPS      = int(getenv("STEPS",               "100"))
    LR         = float(getenv("LR",               "3e-5"))
    CKPT_EVERY = int(getenv("CKPT_EVERY",          "500"))
    EVAL_EVERY = int(getenv("EVAL_EVERY",          "50"))
    LOG_EVERY  = int(getenv("LOG_EVERY",           "10"))
    PER_BREATH_EVERY = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY   = int(getenv("GC_EVERY",            "50"))
    CKPT_LABEL = getenv("CKPT_LABEL",              "v113_smoke")
    PYTHIA_INIT = int(getenv("PYTHIA_INIT",        "1")) > 0
    SEED       = int(getenv("SEED",                "42"))

    ALTERNATION   = int(getenv("V110_STEP3_ALTERNATION",    "1")) > 0
    HARD_LEVEL    = int(getenv("V110_STEP3_HARD_BREATH_LEVEL", "0")) > 0
    PHASE_SCALE   = float(getenv("V110_STEP3_PHASE_SCALE", str(V110_STEP3_PHASE_SCALE)))
    GATE_PROFILE  = getenv("V110_STEP3_GATE_PROFILE", V110_STEP3_GATE_PROFILE)
    PHOTON_ALPHA  = float(getenv("V110_STEP3_PHOTON_ALPHA", str(V110_STEP3_PHOTON_ALPHA)))
    BALANCE_WEIGHT = float(getenv("V110_STEP3_BALANCE_WEIGHT", str(V110_STEP3_BALANCE_WEIGHT)))
    UNCERTAINTY_MIN = float(getenv("V110_STEP3_UNCERTAINTY_MIN", str(V110_STEP3_UNCERTAINTY_MIN)))
    VW         = float(getenv("V110_STEP3_VAR_LOSS_WEIGHT",   str(V110_STEP3_VAR_LOSS_WEIGHT)))
    FW         = float(getenv("V110_STEP3_FACTOR_AUX_WEIGHT", str(V110_STEP3_FACTOR_AUX_WEIGHT)))
    AW         = float(getenv("V110_STEP3_CALIB_WEIGHT",      str(V110_STEP3_CALIB_WEIGHT)))

    SBP_NOISE  = float(getenv("V113_SBP_NOISE_SCALE", "0.0"))

    TRAIN_PATH   = getenv("V113_TRAIN",       ".cache/factor_graph_train.jsonl")
    VAL_PATH     = getenv("V113_VAL",         ".cache/factor_graph_test.jsonl")
    GSM8K_PATH   = getenv("V113_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO  = float(getenv("V113_GSM8K_RATIO", "0.5"))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", "5"))
    EVAL_BATCH   = int(getenv("EVAL_BATCH",   str(BATCH)))

    H     = 1024
    T_new = N_MAX * N_DIGITS + F_MAX
    T_old = N_MAX + F_MAX
    N_HEADS = V110_STEP3_N_HEADS

    print(f"=== v113 training — per-digit input tokens + v112b topology ===")
    print(f"  N_MAX={N_MAX} N_DIGITS={N_DIGITS} F_MAX={F_MAX}")
    print(f"  T_old={T_old}  T_new={T_new}  K={K}")
    print(f"  waist_dim={WAIST_DIM}  topo_dim={TOPO_DIM}")
    print(f"  gate_profile={GATE_PROFILE}  photon_alpha={PHOTON_ALPHA}")
    print(f"  alternation={ALTERNATION}  phase_scale={PHASE_SCALE}")
    print(f"  balance_weight={BALANCE_WEIGHT}  uncertainty_min={UNCERTAINTY_MIN}")
    print(f"  SBP_NOISE={SBP_NOISE}")
    print(f"  device={Device.DEFAULT}  B={BATCH}  steps={STEPS}  lr={LR}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print(f"loading Pythia-410M (PYTHIA_INIT={PYTHIA_INIT})...")
    if PYTHIA_INIT:
        sd = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)

    attach_fg_params_v113(
        model, hidden=cfg.hidden,
        n_max=N_MAX, f_max=F_MAX, k_max=K,
        n_digits=N_DIGITS, n_code=N_CODE, ib_centroids_path=IB_PATH,
        waist_dim=WAIST_DIM, topology_dim=TOPO_DIM,
    )
    Device[Device.DEFAULT].synchronize()

    params   = fg_v113_parameters(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    synth_loader = FactorGraphLoaderV107(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=N_HEADS, seed=SEED,
    )
    gsm8k_records = load_gsm8k_records_v107(GSM8K_PATH)
    dual_loader = DualDataLoaderV107(
        synth_loader, gsm8k_records, gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=N_HEADS, seed=SEED + 1,
    )
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=N_HEADS, seed=SEED + 2,
    )

    ckpt_dir = ".cache/fg_v113_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    ns_zero_np  = np.array([0.0],       dtype=np.float16)
    ns_noisy_np = np.array([SBP_NOISE], dtype=np.float16)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v113(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FW, calib_weight=AW, var_loss_weight=VW,
        balance_weight=BALANCE_WEIGHT, uncertainty_min=UNCERTAINTY_MIN,
        hard_breath_level=HARD_LEVEL, alternation=ALTERNATION,
        phase_scale=PHASE_SCALE,
        gate_profile=GATE_PROFILE, photon_alpha=PHOTON_ALPHA,
        n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS, grad_clip=1.0,
    )
    Tensor.training = True

    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_aux = log_bal = log_n = 0.0
    rng_noise = np.random.RandomState(SEED + 7)
    first_loss = None
    last_loss  = None

    for step in range(1, STEPS + 1):
        batch = dual_loader.sample_batch(step=step)

        gold_bins_np  = batch["gold_bins"].numpy()
        obs_mask_np   = batch["observed_mask"].numpy()
        gold_vals_np  = batch.get("gold_values", batch["gold_bins"]).numpy()

        # Digit decomposition
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=N_DIGITS)

        # Factor gold digits (for factor aux CE)
        fgb_np = batch["factor_gold_bin"].numpy()
        factor_gold_digits_np = bins_to_digits_msd(fgb_np, n_digits=N_DIGITS)

        # Per-digit input
        di_np = build_digit_init_v113(gold_vals_np, obs_mask_np, n_digits=N_DIGITS)
        vm_np = build_valid_mask_np(gold_vals_np, obs_mask_np, n_digits=N_DIGITS)

        # Expand masks: T_old → T_new
        stg_np = batch["staging_mask"].numpy()   # (B, K, T_old, T_old)
        hop_np = batch["head_op_mask"].numpy()   # (B, N_HEADS, T_old, T_old)
        stg_new_np = _expand_mask_fast(stg_np, N_MAX, F_MAX, N_DIGITS)
        hop_new_np = _expand_mask_fast(hop_np, N_MAX, F_MAX, N_DIGITS)

        # node_kinds for T_new layout (expand each var to n_digits tokens)
        nk_old = batch["node_kinds"].numpy()   # (B, T_old)
        B_actual = nk_old.shape[0]
        nk_new = np.full((B_actual, T_new), -1, dtype=np.int32)
        for b in range(B_actual):
            for v in range(N_MAX):
                kind = int(nk_old[b, v])
                for p in range(N_DIGITS):
                    nk_new[b, v * N_DIGITS + p] = kind
            for fi in range(F_MAX):
                nk_new[b, N_MAX * N_DIGITS + fi] = int(nk_old[b, N_MAX + fi])
        nk_new = np.clip(nk_new, 0, 2)

        # SBP noise
        if SBP_NOISE > 0.0 and (step % 2 == 1):
            noise_np = rng_noise.randn(K, BATCH, T_new, H).astype(np.float16)
            ns_np = ns_noisy_np
        else:
            noise_np = np.zeros((K, BATCH, T_new, H), dtype=np.float16)
            ns_np = ns_zero_np

        # Build tensors
        di_t   = Tensor(di_np,  dtype=dtypes.float).contiguous().realize()
        vm_t   = Tensor(vm_np.astype(np.float32), dtype=dtypes.float).contiguous().realize()
        nk_t   = Tensor(nk_new, dtype=dtypes.int).contiguous().realize()
        stg_t  = Tensor(stg_new_np, dtype=dtypes.float).contiguous().realize()
        hop_t  = Tensor(hop_new_np, dtype=dtypes.float).contiguous().realize()
        gd_t   = Tensor(gold_digits_np.astype(np.int64), dtype=dtypes.int).contiguous().realize()
        gb_t   = batch["gold_bins"]
        om_t   = batch["observed_mask"]
        fgd_t  = Tensor(factor_gold_digits_np.astype(np.int64), dtype=dtypes.int).contiguous().realize()
        fv_t   = batch["factor_valid"]
        noise_t = Tensor(noise_np).cast(dtypes.half).contiguous().realize()
        ns_t    = Tensor(ns_np).cast(dtypes.half).contiguous().realize()

        outs = step_fn(
            di_t, vm_t, nk_t, stg_t, hop_t,
            gd_t, gb_t, om_t, fgd_t, fv_t,
            noise_t, ns_t,
        )
        total_t, healthy_t = outs[0], outs[1]
        ce_t, aux_t, calib_t = outs[2], outs[3], outs[4]
        cell_acc_t, query_acc_t = outs[5], outs[6]
        balance_t = outs[7]
        pb_ce_ts  = outs[8:8 + K]
        step_mag_ts = outs[8 + K: 8 + 2 * K]

        loss_val = float(total_t.numpy())
        if first_loss is None:
            first_loss = loss_val
        last_loss = loss_val

        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}", flush=True)

        log_loss  += loss_val
        log_ce    += float(ce_t.numpy())
        log_calib += float(calib_t.numpy())
        log_aux   += float(aux_t.numpy())
        log_bal   += float(balance_t.numpy())
        log_n     += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            print(
                f"[step {step:5d}] loss={log_loss/log_n:.4f} "
                f"ce={log_ce/log_n:.4f} aux={log_aux/log_n:.4f} "
                f"calib={log_calib/log_n:.4f} bal={log_bal/log_n:.4f}  "
                f"cell_acc={float(cell_acc_t.numpy()):.3f}  "
                f"({dt:.1f}s, {dt/step:.2f}s/step)",
                flush=True,
            )
            log_loss = log_ce = log_calib = log_aux = log_bal = log_n = 0.0

        if step % PER_BREATH_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            step_mags = [float(t.numpy()) for t in step_mag_ts]
            pb_str = " ".join(f"{v:.3f}" for v in pb_ce)
            sm_str = " ".join(f"{v:.3f}" for v in step_mags)
            ca = float(cell_acc_t.numpy())
            qa = float(query_acc_t.numpy())
            print(f"  per_breath_ce[B0..B{K-1}]: {pb_str}  "
                  f"(cell_acc={ca:.3f} query_acc={qa:.3f})", flush=True)
            print(f"  step_mags[B0..B{K-1}]:      {sm_str}", flush=True)
            if K > 1 and len(pb_ce) >= 2:
                ladder_delta = pb_ce[0] - pb_ce[-1]
                tag = "OK" if ladder_delta > 0.1 else "target > 0.1"
                print(f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} ({tag})", flush=True)

        if step % EVAL_EVERY == 0:
            print(f"  quick eval skip in smoke mode (EVAL_EVERY={EVAL_EVERY})", flush=True)

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(fg_v113_state_dict(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(fg_v113_state_dict(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)
    print(f"first_loss={first_loss:.4f}  last_loss={last_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
