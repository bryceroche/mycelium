"""v110-acc factor graph training driver — v109pi + ACCUMULATE notebook."""
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
from mycelium.factor_graph_v110_acc import (
    V110_ACC_K_MAX, V110_ACC_N_MAX, V110_ACC_F_MAX, V110_ACC_N_HEADS,
    V110_ACC_N_DIGITS,
    V110_ACC_WAIST_DIM, V110_ACC_ALTERNATION, V110_ACC_HARD_BREATH_LEVEL,
    V110_ACC_VAR_LOSS_WEIGHT, V110_ACC_CALIB_WEIGHT, V110_ACC_FACTOR_AUX_WEIGHT,
    V110_ACC_CODEBOOK_N, V110_ACC_IB_CENTROIDS, V110_ACC_PHASE_SCALE,
    attach_fg_params_v110_acc, fg_v110_acc_parameters, fg_v110_acc_state_dict,
    fg_breathing_forward_v110_acc,
    _compile_jit_fg_step_v110_acc, _compile_jit_fg_eval_v110_acc,
)
from mycelium.factor_graph_v108 import bins_to_digits_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, DualDataLoaderV107, load_gsm8k_records_v107,
)
from scripts.v108_factor_graph_train import cast_layers_fp32
from scripts.v109_factor_graph_train import (
    collect_fg_params_v109, model_state_dict_v109,
    evaluate_v109,
)

DIFFICULTIES = ["easy", "medium", "hard"]


# v110-acc: collect = v109 backbone + v110-acc notebook params + v109 fg params
def collect_fg_params_v110_acc(model) -> list:
    """Backbone + v107/v108/v109 fg + v110-acc notebook params."""
    params = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += fg_v110_acc_parameters(model)  # includes v107/v108/v109 fg + new acc params
    return params


def model_state_dict_v110_acc(model) -> dict:
    """Full state_dict: backbone (shared+per-layer) + all fg families incl. v110-acc."""
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    sd.update(fg_v110_acc_state_dict(model))
    return sd


def load_ckpt_v110_acc(model, path: str):
    """Load a v109/v109pi/v110-acc ckpt. v110-acc keys fall back to fresh init."""
    sd = safe_load(path)
    targets = model_state_dict_v110_acc(model)
    loaded, missing = [], []
    for name, dst in targets.items():
        if name in sd:
            src = sd[name].to(dst.device).realize()
            if src.shape != dst.shape:
                missing.append(f"{name}(shape)")
                continue
            if src.dtype != dst.dtype:
                src = src.cast(dst.dtype)
            dst.assign(src).realize()
            loaded.append(name)
        else:
            missing.append(name)
    bb  = [k for k in loaded if k.startswith("shared.") or k.startswith("phase")]
    v7  = [k for k in loaded if k.startswith("fg_v107.")]
    v8  = [k for k in loaded if k.startswith("fg_v108.")]
    v9  = [k for k in loaded if k.startswith("fg_v109.")]
    v10 = [k for k in loaded if k.startswith("fg_v110_acc")]
    print(f"  loaded {len(bb)} backbone + {len(v7)} v107 + "
          f"{len(v8)} v108 + {len(v9)} v109 + {len(v10)} v110-acc keys", flush=True)
    if any(k.startswith("fg_v110_acc") for k in missing):
        print("  v110-acc notebook keys not in ckpt — using fresh init "
              "(W_o=0 → byte-identical to v109pi at step 0)", flush=True)


# evaluate is identical interface to v109pi
evaluate_v110_acc = evaluate_v109


def main():
    V110_ACC_TASK_LOCAL = int(getenv("V110_ACC_TASK", 0)) > 0
    assert V110_ACC_TASK_LOCAL, "V110_ACC_TASK=1 must be set"

    K          = int(getenv("V110_ACC_K_MAX",    str(V110_ACC_K_MAX)))
    BATCH      = int(getenv("BATCH",             "8"))
    STEPS      = int(getenv("STEPS",             "500"))
    LR         = float(getenv("LR",              "3e-5"))
    CKPT_EVERY = int(getenv("CKPT_EVERY",        "250"))
    EVAL_EVERY = int(getenv("EVAL_EVERY",        "100"))
    LOG_EVERY  = int(getenv("LOG_EVERY",         "10"))
    PER_BREATH_EVERY = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY   = int(getenv("GC_EVERY",          "50"))
    CKPT_LABEL = getenv("CKPT_LABEL",            "v110_acc_smoke")
    RESUME_FROM = getenv("RESUME_FROM",          "")
    PYTHIA_INIT = int(getenv("PYTHIA_INIT",      "1")) > 0
    SEED       = int(getenv("SEED",              "42"))
    N_CODE     = int(getenv("V110_ACC_CODEBOOK_N", str(V110_ACC_CODEBOOK_N)))
    IB_PATH    = getenv("V110_ACC_IB_CENTROIDS",   V110_ACC_IB_CENTROIDS)
    N_DIGITS   = int(getenv("V110_ACC_N_DIGITS",   str(V110_ACC_N_DIGITS)))
    WAIST_DIM  = int(getenv("V110_ACC_WAIST_DIM",  str(V110_ACC_WAIST_DIM)))
    ALTERNATION = int(getenv("V110_ACC_ALTERNATION", "1")) > 0
    HARD_LEVEL = int(getenv("V110_ACC_HARD_BREATH_LEVEL", "0")) > 0
    PHASE_SCALE = float(getenv("V110_ACC_PHASE_SCALE", str(V110_ACC_PHASE_SCALE)))
    VW         = float(getenv("V110_ACC_VAR_LOSS_WEIGHT",  str(V110_ACC_VAR_LOSS_WEIGHT)))
    FW         = float(getenv("V110_ACC_FACTOR_AUX_WEIGHT", str(V110_ACC_FACTOR_AUX_WEIGHT)))
    AW         = float(getenv("V110_ACC_CALIB_WEIGHT",      str(V110_ACC_CALIB_WEIGHT)))
    N_MAX      = int(getenv("V110_ACC_N_MAX",    str(V110_ACC_N_MAX)))
    F_MAX      = int(getenv("V110_ACC_F_MAX",    str(V110_ACC_F_MAX)))

    DIFFICULTY_FILTER = os.environ.get("V110_ACC_DIFFICULTY_FILTER", "").strip() or None
    TRAIN_PATH   = getenv("V110_ACC_TRAIN",       ".cache/factor_graph_train.jsonl")
    VAL_PATH     = getenv("V110_ACC_VAL",         ".cache/factor_graph_test.jsonl")
    GSM8K_PATH   = getenv("V110_ACC_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO  = float(getenv("V110_ACC_GSM8K_RATIO", "0.5"))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", "10"))
    EVAL_BATCH   = int(getenv("EVAL_BATCH",   str(BATCH)))

    print(f"=== v110-acc factor graph training (v109pi + ACCUMULATE notebook) ===")
    print(f"  alternation={ALTERNATION}  phase_scale={PHASE_SCALE}")
    print(f"  device={Device.DEFAULT}  B={BATCH}  K={K}  N_DIGITS={N_DIGITS}  "
          f"WAIST={WAIST_DIM}  steps={STEPS}  lr={LR}")
    print(f"  loss weights:  var={VW}  factor_aux={FW}  calib={AW}")
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

    attach_fg_params_v110_acc(
        model, hidden=cfg.hidden,
        n_max=N_MAX, f_max=F_MAX, k_max=K,
        n_digits=N_DIGITS, n_code=N_CODE, ib_centroids_path=IB_PATH,
        waist_dim=WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()

    params   = collect_fg_params_v110_acc(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"loading ckpt: {RESUME_FROM}")
        load_ckpt_v110_acc(model, RESUME_FROM)

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    synth_loader = FactorGraphLoaderV107(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V110_ACC_N_HEADS, seed=SEED,
    )
    gsm8k_records = load_gsm8k_records_v107(GSM8K_PATH)
    dual_loader = DualDataLoaderV107(
        synth_loader, gsm8k_records, gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V110_ACC_N_HEADS, seed=SEED + 1,
    )
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V110_ACC_N_HEADS, seed=SEED + 2,
    )

    ckpt_dir = ".cache/fg_v110_acc_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v110_acc(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FW, calib_weight=AW, var_loss_weight=VW,
        hard_breath_level=HARD_LEVEL, alternation=ALTERNATION,
        phase_scale=PHASE_SCALE,
        n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS, grad_clip=1.0,
    )
    eval_fn = _compile_jit_fg_eval_v110_acc(
        model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        alternation=ALTERNATION, phase_scale=PHASE_SCALE,
    )
    Tensor.training = True

    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_aux = log_n = 0.0

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
        pb_ce_ts = outs[7:7 + K]

        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}", flush=True)

        log_loss  += float(total_t.numpy())
        log_ce    += float(ce_t.numpy())
        log_calib += float(calib_t.numpy())
        log_aux   += float(aux_t.numpy())
        log_n     += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            print(
                f"[step {step:5d}] loss={log_loss/log_n:.4f} "
                f"ce={log_ce/log_n:.4f} aux={log_aux/log_n:.4f} "
                f"calib={log_calib/log_n:.4f}  "
                f"cell_acc={float(cell_acc_t.numpy()):.3f}  "
                f"({dt:.1f}s, {dt/step:.2f}s/step)",
                flush=True,
            )
            log_loss = log_ce = log_calib = log_aux = log_n = 0.0

        if step % PER_BREATH_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            pb_str = " ".join(f"{v:.3f}" for v in pb_ce)
            ca = float(cell_acc_t.numpy())
            qa = float(query_acc_t.numpy())
            print(f"  per_breath_ce[B0..B{K-1}]: {pb_str}  "
                  f"(cell_acc={ca:.3f} query_acc={qa:.3f})", flush=True)
            if K > 1 and len(pb_ce) >= 2:
                ladder_delta = pb_ce[0] - pb_ce[-1]
                tag = "OK" if ladder_delta > 0.1 else "target > 0.1"
                print(f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} ({tag})",
                      flush=True)

        if step % EVAL_EVERY == 0:
            print(f"  evaluating ({EVAL_BATCHES} batches × B={EVAL_BATCH})...", flush=True)
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
            safe_save(model_state_dict_v110_acc(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_v110_acc(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
