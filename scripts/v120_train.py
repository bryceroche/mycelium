"""v120 training driver — persistent cross-breath latent notebook (NOTEBOOK, not FILTER).

Key architectural change from v118/v119:
  Latents PERSIST across all K breaths within a forward pass.
  Each breath: READ (residual ← latents) → BREATH → WRITE (latents ← residual)
  IB-centroid init (first 16 of 32) makes cross-attention meaningful from step 0.
  Two zero-init W_out projections with (1+g) amplifier bootstrap gradient flow.

Diagnostic logging (v120-specific, logged every LOG_EVERY steps):
  read_gate       — scalar gate on read delta (should grow away from 0)
  write_gate      — scalar gate on write delta (should grow away from 0)
  read_W_out_norm — Frobenius norm of read output projection (bootstrap signal)
  write_W_out_norm— Frobenius norm of write output projection
  latent_drift    — mean norm of latents at end of K-th breath vs init latents
                    (measures whether latents are actually evolving)

Warm-starts from .cache/fg_v112b_ckpts/v112b_cont1_final.safetensors.
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
    V112B_TOPOLOGY_DIM, V112B_BIAS_SCALE_INIT,
)
from mycelium.factor_graph_v120 import (
    V120_N_LATENTS,
    attach_fg_params_v120,
    fg_v120_parameters, fg_v120_state_dict,
    _compile_jit_fg_step_v120,
    compile_jit_eval_v120,
)
from mycelium.factor_graph_v108 import bins_to_digits_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, DualDataLoaderV107, load_gsm8k_records_v107,
)
from scripts.v108_factor_graph_train import cast_layers_fp32
from scripts.v110_acc_factor_graph_train import (
    load_ckpt_v110_acc, evaluate_v110_acc,
)

DIFFICULTIES = ["easy", "medium", "hard"]

V112B_CKPT_PATH = ".cache/fg_v112b_ckpts/v112b_cont1_final.safetensors"


def load_ckpt_v120(model, path: str) -> None:
    """Load v110-step3/v112b ckpt + optionally v120-specific params if present."""
    load_ckpt_v110_acc(model, path)
    sd = safe_load(path)
    # v112b topology params
    for key in ("fg_v115_node_topology", "fg_v115_W_res_gate",
                "fg_v115_attn_bias_scale"):
        if key in sd:
            dst = getattr(model, key)
            src = sd[key].to(dst.device).realize()
            dst.assign(src).realize()
            print(f"  loaded v112b param: {key}", flush=True)
    # v120-specific params (only present in v120 ckpts, not v112b ckpts — that's OK)
    v120_keys = [
        "fg_v120_latents",
        "fg_v120_read_gate",
        "fg_v120_write_gate",
        "fg_v120_read_W_out",
        "fg_v120_write_W_out",
    ]
    for key in v120_keys:
        if key in sd and hasattr(model, key):
            dst = getattr(model, key)
            src = sd[key].to(dst.device).realize()
            dst.assign(src).realize()
            print(f"  loaded v120 param: {key}", flush=True)


def main():
    V110_STEP3_TASK_LOCAL = int(getenv("V110_STEP3_TASK", 0)) > 0
    assert V110_STEP3_TASK_LOCAL, "V110_STEP3_TASK=1 must be set"

    K          = int(getenv("V110_STEP3_K_MAX",    str(V110_STEP3_K_MAX)))
    BATCH      = int(getenv("BATCH",               "8"))
    STEPS      = int(getenv("STEPS",               "300"))
    LR         = float(getenv("LR",                "3e-5"))
    CKPT_EVERY = int(getenv("CKPT_EVERY",          "300"))
    EVAL_EVERY = int(getenv("EVAL_EVERY",          "300"))
    LOG_EVERY  = int(getenv("LOG_EVERY",           "10"))
    PER_BREATH_EVERY = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY   = int(getenv("GC_EVERY",            "50"))
    CKPT_LABEL = getenv("CKPT_LABEL",              "v120_run")
    RESUME_FROM = getenv("RESUME_FROM",            "")
    WARM_FROM  = getenv("WARM_FROM",               "")
    PYTHIA_INIT = int(getenv("PYTHIA_INIT",        "1")) > 0
    SEED       = int(getenv("SEED",                "42"))
    N_CODE     = int(getenv("V110_STEP3_CODEBOOK_N", str(V110_STEP3_CODEBOOK_N)))
    IB_PATH    = getenv("V110_STEP3_IB_CENTROIDS",   V110_STEP3_IB_CENTROIDS)
    N_DIGITS   = int(getenv("V110_STEP3_N_DIGITS",   str(V110_STEP3_N_DIGITS)))
    WAIST_DIM  = int(getenv("V110_STEP3_WAIST_DIM",  str(V110_STEP3_WAIST_DIM)))
    ALTERNATION = int(getenv("V110_STEP3_ALTERNATION", "1")) > 0
    HARD_LEVEL = int(getenv("V110_STEP3_HARD_BREATH_LEVEL", "0")) > 0
    PHASE_SCALE = float(getenv("V110_STEP3_PHASE_SCALE", str(V110_STEP3_PHASE_SCALE)))
    GATE_PROFILE = getenv("V110_STEP3_GATE_PROFILE", V110_STEP3_GATE_PROFILE)
    PHOTON_ALPHA = float(getenv("V110_STEP3_PHOTON_ALPHA", str(V110_STEP3_PHOTON_ALPHA)))
    BALANCE_WEIGHT = float(getenv("V110_STEP3_BALANCE_WEIGHT", str(V110_STEP3_BALANCE_WEIGHT)))
    UNCERTAINTY_MIN = float(getenv("V110_STEP3_UNCERTAINTY_MIN", str(V110_STEP3_UNCERTAINTY_MIN)))
    VW         = float(getenv("V110_STEP3_VAR_LOSS_WEIGHT",  str(V110_STEP3_VAR_LOSS_WEIGHT)))
    FW         = float(getenv("V110_STEP3_FACTOR_AUX_WEIGHT", str(V110_STEP3_FACTOR_AUX_WEIGHT)))
    AW         = float(getenv("V110_STEP3_CALIB_WEIGHT",      str(V110_STEP3_CALIB_WEIGHT)))
    N_MAX      = int(getenv("V110_STEP3_N_MAX",    str(V110_STEP3_N_MAX)))
    F_MAX      = int(getenv("V110_STEP3_F_MAX",    str(V110_STEP3_F_MAX)))
    SBP_NOISE  = float(getenv("V110_STEP3_SBP_NOISE_SCALE", "0.0"))
    TOPOLOGY_DIM    = int(getenv("V112B_TOPOLOGY_DIM",    str(V112B_TOPOLOGY_DIM)))
    BIAS_SCALE_INIT = float(getenv("V112B_BIAS_SCALE_INIT", str(V112B_BIAS_SCALE_INIT)))
    N_LATENTS  = int(getenv("V120_N_LATENTS",      str(V120_N_LATENTS)))

    DIFFICULTY_FILTER = os.environ.get("V110_STEP3_DIFFICULTY_FILTER", "").strip() or None
    TRAIN_PATH   = getenv("V110_STEP3_TRAIN",       ".cache/factor_graph_train.jsonl")
    VAL_PATH     = getenv("V110_STEP3_VAL",         ".cache/factor_graph_test.jsonl")
    GSM8K_PATH   = getenv("V110_STEP3_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO  = float(getenv("V110_STEP3_GSM8K_RATIO", "0.5"))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", "10"))
    EVAL_BATCH   = int(getenv("EVAL_BATCH",   str(BATCH)))

    H = 1024
    T = N_MAX + F_MAX

    sbp_mode = "OFF" if SBP_NOISE == 0.0 else f"ON σ={SBP_NOISE}"
    print(f"=== v120 training: persistent cross-breath latent notebook ===")
    print(f"  n_latents={N_LATENTS}  topology_dim={TOPOLOGY_DIM}  bias_scale_init={BIAS_SCALE_INIT}")
    print(f"  photon_alpha={PHOTON_ALPHA}  balance_weight={BALANCE_WEIGHT}  "
          f"uncertainty_min={UNCERTAINTY_MIN}  SBP={sbp_mode}")
    print(f"  alternation={ALTERNATION}  phase_scale={PHASE_SCALE}  gate_profile={GATE_PROFILE}")
    print(f"  device={Device.DEFAULT}  B={BATCH}  K={K}  N_DIGITS={N_DIGITS}  "
          f"WAIST={WAIST_DIM}  steps={STEPS}  lr={LR}")
    print(f"  T (n_max+f_max) = {T}  H = {H}")
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

    attach_fg_params_v120(
        model, hidden=cfg.hidden,
        n_max=N_MAX, f_max=F_MAX, k_max=K,
        n_digits=N_DIGITS, n_code=N_CODE, ib_centroids_path=IB_PATH,
        waist_dim=WAIST_DIM,
        n_latents=N_LATENTS,
    )
    Device[Device.DEFAULT].synchronize()

    params   = fg_v120_parameters(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    # Warm-start priority: RESUME_FROM > WARM_FROM (v112b ckpt) > cold start
    ckpt_to_load = ""
    if RESUME_FROM:
        ckpt_to_load = RESUME_FROM
    elif WARM_FROM == "v112b_cont1_final":
        ckpt_to_load = V112B_CKPT_PATH
    elif WARM_FROM:
        ckpt_to_load = WARM_FROM

    if ckpt_to_load:
        print(f"loading ckpt: {ckpt_to_load}")
        load_ckpt_v120(model, ckpt_to_load)

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    synth_loader = FactorGraphLoaderV107(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V110_STEP3_N_HEADS, seed=SEED,
    )
    gsm8k_records = load_gsm8k_records_v107(GSM8K_PATH)
    dual_loader = DualDataLoaderV107(
        synth_loader, gsm8k_records, gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V110_STEP3_N_HEADS, seed=SEED + 1,
    )
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V110_STEP3_N_HEADS, seed=SEED + 2,
    )

    ckpt_dir = ".cache/fg_v120_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    ns_zero_np  = np.array([0.0],       dtype=np.float16)
    ns_noisy_np = np.array([SBP_NOISE], dtype=np.float16)

    # Snapshot init latents for drift measurement
    init_latents_np = model.fg_v120_latents.numpy().copy()
    init_lat_norms  = np.linalg.norm(init_latents_np, axis=1).mean()

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v120(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FW, calib_weight=AW, var_loss_weight=VW,
        balance_weight=BALANCE_WEIGHT, uncertainty_min=UNCERTAINTY_MIN,
        hard_breath_level=HARD_LEVEL, alternation=ALTERNATION,
        phase_scale=PHASE_SCALE,
        gate_profile=GATE_PROFILE, photon_alpha=PHOTON_ALPHA,
        n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS, grad_clip=1.0,
    )
    eval_fn = compile_jit_eval_v120(
        model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        alternation=ALTERNATION, phase_scale=PHASE_SCALE,
        gate_profile=GATE_PROFILE, photon_alpha=PHOTON_ALPHA,
    )
    Tensor.training = True

    print(f"\ntraining (notebook: {N_LATENTS} persistent latents, SBP={sbp_mode})...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_aux = log_bal = log_n = 0.0
    log_noisy_n = 0
    log_det_n   = 0

    rng_noise = np.random.RandomState(SEED + 7)

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

        if SBP_NOISE > 0.0 and (step % 2 == 1):
            noise_np = rng_noise.randn(K, BATCH, T, H).astype(np.float16)
            ns_np = ns_noisy_np
            log_noisy_n += 1
        else:
            noise_np = np.zeros((K, BATCH, T, H), dtype=np.float16)
            ns_np = ns_zero_np
            log_det_n += 1
        noise_t = Tensor(noise_np).cast(dtypes.half).contiguous().realize()
        ns_t    = Tensor(ns_np).cast(dtypes.half).contiguous().realize()

        outs = step_fn(
            domain_init, node_kinds, staging_mask, head_op_mask,
            gold_digits_t, gold_bins, obs_mask, fgb_t, fv_t,
            noise_t, ns_t,
        )
        total_t, healthy_t = outs[0], outs[1]
        ce_t, aux_t, calib_t = outs[2], outs[3], outs[4]
        cell_acc_t, query_acc_t = outs[5], outs[6]
        balance_t = outs[7]
        pb_ce_ts = outs[8:8 + K]
        step_mag_ts = outs[8 + K: 8 + 2 * K]
        calib_per_breath_ts = outs[8 + 2 * K: 8 + 3 * K]

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
            noisy_frac = log_noisy_n / max(log_noisy_n + log_det_n, 1)

            # v112b topology diagnostics
            topo_np = model.fg_v115_node_topology.numpy()
            n0_norm = float(np.linalg.norm(topo_np[0])) + 1e-8
            n1_norm = float(np.linalg.norm(topo_np[1])) + 1e-8
            sim01   = float(np.dot(topo_np[0], topo_np[1]) / (n0_norm * n1_norm))
            scale   = float(model.fg_v115_attn_bias_scale.numpy()[0])
            wres_norm = float(np.linalg.norm(model.fg_v115_W_res_gate.numpy()))

            # v120 notebook diagnostics
            read_gate_val  = float(model.fg_v120_read_gate.numpy()[0])
            write_gate_val = float(model.fg_v120_write_gate.numpy()[0])
            read_W_out_norm  = float(np.linalg.norm(model.fg_v120_read_W_out.numpy()))
            write_W_out_norm = float(np.linalg.norm(model.fg_v120_write_W_out.numpy()))
            # Latent drift: how much have latents moved from their IB init?
            cur_lat_np = model.fg_v120_latents.numpy()
            cur_lat_norms = np.linalg.norm(cur_lat_np, axis=1).mean()
            lat_drift = abs(cur_lat_norms - init_lat_norms)

            print(
                f"[step {step:5d}] loss={log_loss/log_n:.4f} "
                f"ce={log_ce/log_n:.4f} aux={log_aux/log_n:.4f} "
                f"calib={log_calib/log_n:.4f} bal={log_bal/log_n:.4f}  "
                f"cell_acc={float(cell_acc_t.numpy()):.3f}  "
                f"topo_sim01={sim01:+.3f}  Wres={wres_norm:.3f}  "
                f"r_gate={read_gate_val:+.4f}  w_gate={write_gate_val:+.4f}  "
                f"rW={read_W_out_norm:.3f}  wW={write_W_out_norm:.3f}  "
                f"lat_drift={lat_drift:.4f}  "
                f"noisy={noisy_frac:.0%}  "
                f"({dt:.1f}s, {dt/step:.2f}s/step)",
                flush=True,
            )
            log_loss = log_ce = log_calib = log_aux = log_bal = log_n = 0.0
            log_noisy_n = log_det_n = 0

        if step % PER_BREATH_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            step_mags = [float(t.numpy()) for t in step_mag_ts]
            calibs = [float(t.numpy()) for t in calib_per_breath_ts]
            pb_str = " ".join(f"{v:.3f}" for v in pb_ce)
            sm_str = " ".join(f"{v:.3f}" for v in step_mags)
            cal_str = " ".join(f"{v:.3f}" for v in calibs)
            ca = float(cell_acc_t.numpy())
            qa = float(query_acc_t.numpy())
            print(f"  per_breath_ce[B0..B{K-1}]:    {pb_str}  "
                  f"(cell_acc={ca:.3f} query_acc={qa:.3f})", flush=True)
            print(f"  step_mags[B0..B{K-1}]:        {sm_str}  "
                  f"bal_loss={float(balance_t.numpy()):.4f}", flush=True)
            print(f"  calib[B0..B{K-1}]:            {cal_str}", flush=True)
            if K > 1 and len(pb_ce) >= 2:
                ladder_delta = pb_ce[0] - pb_ce[-1]
                tag = "OK" if ladder_delta > 0.1 else "target > 0.1"
                print(f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} ({tag})",
                      flush=True)

        if step % EVAL_EVERY == 0:
            print(f"  evaluating ({EVAL_BATCHES} batches x B={EVAL_BATCH})...", flush=True)
            results = evaluate_v110_acc(
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
            safe_save(fg_v120_state_dict(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(fg_v120_state_dict(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
