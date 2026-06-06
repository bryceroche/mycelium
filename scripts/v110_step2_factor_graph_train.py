"""v110-step2 factor graph training driver — CE-normalized Goldilocks penalty."""
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
from mycelium.factor_graph_v110_step2 import (
    V110_STEP2_K_MAX, V110_STEP2_N_MAX, V110_STEP2_F_MAX, V110_STEP2_N_HEADS,
    V110_STEP2_N_DIGITS,
    V110_STEP2_WAIST_DIM, V110_STEP2_ALTERNATION, V110_STEP2_HARD_BREATH_LEVEL,
    V110_STEP2_VAR_LOSS_WEIGHT, V110_STEP2_CALIB_WEIGHT,
    V110_STEP2_FACTOR_AUX_WEIGHT, V110_STEP2_BALANCE_WEIGHT,
    V110_STEP2_CODEBOOK_N, V110_STEP2_IB_CENTROIDS, V110_STEP2_PHASE_SCALE,
    V110_STEP2_GATE_PROFILE, V110_STEP2_PHOTON_ALPHA,
    attach_fg_params_v110_step2, fg_v110_step2_parameters, fg_v110_step2_state_dict,
    _compile_jit_fg_step_v110_step2, _compile_jit_fg_eval_v110_step2,
)
from mycelium.factor_graph_v108 import bins_to_digits_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, DualDataLoaderV107, load_gsm8k_records_v107,
)
from scripts.v108_factor_graph_train import cast_layers_fp32
from scripts.v110_acc_factor_graph_train import (
    collect_fg_params_v110_acc, model_state_dict_v110_acc,
    load_ckpt_v110_acc, evaluate_v110_acc,
)

DIFFICULTIES = ["easy", "medium", "hard"]

collect_fg_params_v110_step2 = collect_fg_params_v110_acc
model_state_dict_v110_step2 = model_state_dict_v110_acc
load_ckpt_v110_step2 = load_ckpt_v110_acc
evaluate_v110_step2 = evaluate_v110_acc


def main():
    V110_STEP2_TASK_LOCAL = int(getenv("V110_STEP2_TASK", 0)) > 0
    assert V110_STEP2_TASK_LOCAL, "V110_STEP2_TASK=1 must be set"

    K          = int(getenv("V110_STEP2_K_MAX",    str(V110_STEP2_K_MAX)))
    BATCH      = int(getenv("BATCH",               "8"))
    STEPS      = int(getenv("STEPS",               "500"))
    LR         = float(getenv("LR",                "3e-5"))
    CKPT_EVERY = int(getenv("CKPT_EVERY",          "250"))
    EVAL_EVERY = int(getenv("EVAL_EVERY",          "100"))
    LOG_EVERY  = int(getenv("LOG_EVERY",           "10"))
    PER_BREATH_EVERY = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY   = int(getenv("GC_EVERY",            "50"))
    CKPT_LABEL = getenv("CKPT_LABEL",              "v110_step2_smoke")
    RESUME_FROM = getenv("RESUME_FROM",            "")
    PYTHIA_INIT = int(getenv("PYTHIA_INIT",        "1")) > 0
    SEED       = int(getenv("SEED",                "42"))
    N_CODE     = int(getenv("V110_STEP2_CODEBOOK_N", str(V110_STEP2_CODEBOOK_N)))
    IB_PATH    = getenv("V110_STEP2_IB_CENTROIDS",   V110_STEP2_IB_CENTROIDS)
    N_DIGITS   = int(getenv("V110_STEP2_N_DIGITS",   str(V110_STEP2_N_DIGITS)))
    WAIST_DIM  = int(getenv("V110_STEP2_WAIST_DIM",  str(V110_STEP2_WAIST_DIM)))
    ALTERNATION = int(getenv("V110_STEP2_ALTERNATION", "1")) > 0
    HARD_LEVEL = int(getenv("V110_STEP2_HARD_BREATH_LEVEL", "0")) > 0
    PHASE_SCALE = float(getenv("V110_STEP2_PHASE_SCALE", str(V110_STEP2_PHASE_SCALE)))
    GATE_PROFILE = getenv("V110_STEP2_GATE_PROFILE", V110_STEP2_GATE_PROFILE)
    PHOTON_ALPHA = float(getenv("V110_STEP2_PHOTON_ALPHA", str(V110_STEP2_PHOTON_ALPHA)))
    BALANCE_WEIGHT = float(getenv("V110_STEP2_BALANCE_WEIGHT", str(V110_STEP2_BALANCE_WEIGHT)))
    VW         = float(getenv("V110_STEP2_VAR_LOSS_WEIGHT",  str(V110_STEP2_VAR_LOSS_WEIGHT)))
    FW         = float(getenv("V110_STEP2_FACTOR_AUX_WEIGHT", str(V110_STEP2_FACTOR_AUX_WEIGHT)))
    AW         = float(getenv("V110_STEP2_CALIB_WEIGHT",      str(V110_STEP2_CALIB_WEIGHT)))
    N_MAX      = int(getenv("V110_STEP2_N_MAX",    str(V110_STEP2_N_MAX)))
    F_MAX      = int(getenv("V110_STEP2_F_MAX",    str(V110_STEP2_F_MAX)))

    DIFFICULTY_FILTER = os.environ.get("V110_STEP2_DIFFICULTY_FILTER", "").strip() or None
    TRAIN_PATH   = getenv("V110_STEP2_TRAIN",       ".cache/factor_graph_train.jsonl")
    VAL_PATH     = getenv("V110_STEP2_VAL",         ".cache/factor_graph_test.jsonl")
    GSM8K_PATH   = getenv("V110_STEP2_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO  = float(getenv("V110_STEP2_GSM8K_RATIO", "0.5"))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", "10"))
    EVAL_BATCH   = int(getenv("EVAL_BATCH",   str(BATCH)))

    print(f"=== v110-step2 factor graph training (CE-normalized Goldilocks penalty) ===")
    print(f"  gate_profile={GATE_PROFILE}  photon_alpha={PHOTON_ALPHA}  balance_weight={BALANCE_WEIGHT}")
    print(f"  alternation={ALTERNATION}  phase_scale={PHASE_SCALE}")
    print(f"  device={Device.DEFAULT}  B={BATCH}  K={K}  N_DIGITS={N_DIGITS}  "
          f"WAIST={WAIST_DIM}  steps={STEPS}  lr={LR}")
    print(f"  loss weights:  var={VW}  factor_aux={FW}  calib={AW}  balance={BALANCE_WEIGHT}")
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

    attach_fg_params_v110_step2(
        model, hidden=cfg.hidden,
        n_max=N_MAX, f_max=F_MAX, k_max=K,
        n_digits=N_DIGITS, n_code=N_CODE, ib_centroids_path=IB_PATH,
        waist_dim=WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()

    params   = collect_fg_params_v110_step2(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"loading ckpt: {RESUME_FROM}")
        load_ckpt_v110_step2(model, RESUME_FROM)

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    synth_loader = FactorGraphLoaderV107(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V110_STEP2_N_HEADS, seed=SEED,
    )
    gsm8k_records = load_gsm8k_records_v107(GSM8K_PATH)
    dual_loader = DualDataLoaderV107(
        synth_loader, gsm8k_records, gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V110_STEP2_N_HEADS, seed=SEED + 1,
    )
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V110_STEP2_N_HEADS, seed=SEED + 2,
    )

    ckpt_dir = ".cache/fg_v110_step2_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v110_step2(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FW, calib_weight=AW, var_loss_weight=VW,
        balance_weight=BALANCE_WEIGHT,
        hard_breath_level=HARD_LEVEL, alternation=ALTERNATION,
        phase_scale=PHASE_SCALE,
        gate_profile=GATE_PROFILE, photon_alpha=PHOTON_ALPHA,
        n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS, grad_clip=1.0,
    )
    eval_fn = _compile_jit_fg_eval_v110_step2(
        model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        alternation=ALTERNATION, phase_scale=PHASE_SCALE,
        gate_profile=GATE_PROFILE, photon_alpha=PHOTON_ALPHA,
    )
    Tensor.training = True

    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_aux = log_bal = log_n = 0.0

    for step in range(1, STEPS + 1):
        batch = dual_loader.sample_batch(step=step)

        domain_init  = batch["domain_init"]
        node_kinds   = batch["node_kinds"]
        staging_mask = batch["staging_mask"]
        head_op_mask = batch["head_op_mask"]
        gold_bins    = batch["gold_bins"]
        obs_mask     = batch["observed_mask"]
        fgb_t        = batch["factor_gold_bin"]
        fv_t         = batch["factor_valid"]

        gold_bins_np   = gold_bins.numpy()
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=N_DIGITS)
        gold_digits_t  = Tensor(
            gold_digits_np.astype(np.int64), dtype=dtypes.int,
        ).contiguous().realize()

        outs = step_fn(
            domain_init, node_kinds, staging_mask, head_op_mask,
            gold_digits_t, gold_bins, obs_mask, fgb_t, fv_t,
        )
        total_t, healthy_t = outs[0], outs[1]
        ce_t, aux_t, calib_t = outs[2], outs[3], outs[4]
        cell_acc_t, query_acc_t = outs[5], outs[6]
        balance_t = outs[7]
        pb_ce_ts = outs[8:8 + K]
        step_mag_ts = outs[8 + K: 8 + 2 * K]

        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}", flush=True)

        log_loss  += float(total_t.numpy())
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
            ratios = [step_mags[k] / max(pb_ce[k], 1e-6) for k in range(K)]
            pb_str = " ".join(f"{v:.3f}" for v in pb_ce)
            sm_str = " ".join(f"{v:.3f}" for v in step_mags)
            r_str  = " ".join(f"{v:.1f}" for v in ratios)
            ca = float(cell_acc_t.numpy())
            qa = float(query_acc_t.numpy())
            print(f"  per_breath_ce[B0..B{K-1}]:  {pb_str}  "
                  f"(cell_acc={ca:.3f} query_acc={qa:.3f})", flush=True)
            print(f"  step_mags[B0..B{K-1}]:      {sm_str}  bal_loss={float(balance_t.numpy()):.4f}",
                  flush=True)
            print(f"  step/CE [B0..B{K-1}]:       {r_str}  "
                  f"(target: roughly constant across breaths)", flush=True)
            if K > 1 and len(pb_ce) >= 2:
                ladder_delta = pb_ce[0] - pb_ce[-1]
                tag = "OK" if ladder_delta > 0.1 else "target > 0.1"
                print(f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} ({tag})",
                      flush=True)

        if step % EVAL_EVERY == 0:
            print(f"  evaluating ({EVAL_BATCHES} batches × B={EVAL_BATCH})...", flush=True)
            results = evaluate_v110_step2(
                model, val_loader, K=K, max_batches=EVAL_BATCHES, eval_fn=eval_fn,
                n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
            )
            for d in DIFFICULTIES:
                if d not in results: continue
                v = results[d]
                pp = " ".join(f"{p:.2f}" for p in v["per_pos_acc"])
                print(f"  val[{d:6s}]: cell={v['cell_acc']:.3f} q={v['query_acc']:.3f} "
                      f"digit={v['digit_acc']:.3f} per_pos=[{pp}] "
                      f"n={v['n_puzzles']}", flush=True)

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(model_state_dict_v110_step2(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_v110_step2(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
