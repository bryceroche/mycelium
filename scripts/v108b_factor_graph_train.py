"""v108b factor graph training driver — digit-decomposed input + tree output.

Architecture: v108b
  Input:  digit_init (B, N_MAX, 5, 10) — one-hot per-digit-per-position for
          observed variables; uniform 1/10 for unobserved. Embedded as sum
          of learned digit_embed[pos, digit] across positions → ONE token
          per variable (BERT-style superposition, no L0 collapse).
  Output: 5-level tree codebook (same as v108).
  Per-breath supervision: SOFT (all levels every breath) or HARD (breath k
                          → level k).

Cold-start from Pythia (no v107 warm-start — v108b's input path is
incompatible with v107's domain_codebook).
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
from mycelium.factor_graph_v108b import (
    V108B_K_MAX, V108B_N_MAX, V108B_F_MAX, V108B_N_HEADS, V108B_N_DIGITS,
    V108B_HARD_BREATH_LEVEL, V108B_VAR_LOSS_WEIGHT,
    V108B_CALIB_WEIGHT, V108B_FACTOR_AUX_WEIGHT,
    V108B_CODEBOOK_N, V108B_IB_CENTROIDS,
    attach_fg_params_v108b, fg_v108b_parameters, fg_v108b_state_dict,
    fg_breathing_forward_v108b,
    _compile_jit_fg_step_v108b, _compile_jit_fg_eval_v108b,
    build_digit_init,
)
from mycelium.factor_graph_v108 import bins_to_digits_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, DualDataLoaderV107, load_gsm8k_records_v107,
)

DIFFICULTIES = ["easy", "medium", "hard"]


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


def collect_fg_params_v108b(model) -> list[Tensor]:
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += fg_v108b_parameters(model)
    return params


def model_state_dict_v108b(model) -> dict:
    sd = {"ln_f.g": model.ln_f_g, "ln_f.b": model.ln_f_b}
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    sd.update(fg_v108b_state_dict(model))
    return sd


def load_ckpt_v108b(model, path: str):
    sd = safe_load(path)
    targets = model_state_dict_v108b(model)
    loaded, missing = [], []
    for name, dst in targets.items():
        if name in sd:
            src = sd[name].to(dst.device).realize()
            if src.shape != dst.shape:
                missing.append(f"{name}(shape mismatch)")
                continue
            if src.dtype != dst.dtype:
                src = src.cast(dst.dtype)
            dst.assign(src).realize()
            loaded.append(name)
        else:
            missing.append(name)
    bb  = [k for k in loaded if k.startswith("shared.") or k.startswith("phase")]
    v8b = [k for k in loaded if k.startswith("fg_v108b.")]
    print(f"  loaded {len(bb)} backbone + {len(v8b)} v108b keys", flush=True)


def build_factor_digit_targets(factor_gold_bin_np: np.ndarray,
                               n_digits: int = V108B_N_DIGITS) -> np.ndarray:
    """Factor gold bin → digit decomposition for factor tree CE."""
    return bins_to_digits_msd(factor_gold_bin_np, n_digits=n_digits)


def evaluate_v108b(model, loader, K: int,
                   max_batches: int = 20, eval_fn=None,
                   n_max: int = V108B_N_MAX, f_max: int = V108B_F_MAX,
                   n_digits: int = V108B_N_DIGITS) -> dict:
    Tensor.training = False
    agg = {}
    n_batches = 0
    for batch in loader.iter_eval(batch_size=loader.batch_size):
        node_kinds   = batch["node_kinds"]
        staging_mask = batch["staging_mask"]
        head_op_mask = batch["head_op_mask"]
        gold_bins    = batch["gold_bins"]
        obs_mask     = batch["observed_mask"]
        query_idx_np = batch["query_idx"]
        picks        = batch["picks"]

        gold_bins_np   = gold_bins.numpy()
        obs_np         = obs_mask.numpy()
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=n_digits)
        digit_init_np  = build_digit_init(gold_digits_np, obs_np, n_digits=n_digits)
        gold_digits_t  = Tensor(gold_digits_np.astype(np.int64),
                                dtype=dtypes.int).contiguous().realize()
        digit_init_t   = Tensor(digit_init_np, dtype=dtypes.float).contiguous().realize()

        if eval_fn is not None:
            pred_t, _ = eval_fn(
                digit_init_t, node_kinds, staging_mask, head_op_mask,
                gold_digits_t, obs_mask,
            )
            pred_digits_np = pred_t.numpy()
        else:
            tree_lh, _, _ = fg_breathing_forward_v108b(
                model, digit_init_t, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )
            pred_digits_np = tree_lh[-1].argmax(axis=-1).realize().numpy()

        B_local = len(picks)
        for b in range(B_local):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {
                    "n_unobs": 0, "n_correct_unobs": 0,
                    "query_correct": 0, "n_puzzles": 0,
                    "digit_correct": 0, "digit_total": 0,
                    "per_pos_correct": np.zeros(n_digits, dtype=np.int64),
                    "per_pos_total":   np.zeros(n_digits, dtype=np.int64),
                }
            qi = int(query_idx_np[b])
            nv = int(batch["n_vars_total"][b])
            for vi in range(min(nv, n_max)):
                if obs_np[b, vi] != 0:
                    continue
                agg[diff]["n_unobs"] += 1
                all_correct = bool(np.all(
                    pred_digits_np[b, vi] == gold_digits_np[b, vi]
                ))
                if all_correct:
                    agg[diff]["n_correct_unobs"] += 1
                for p in range(n_digits):
                    agg[diff]["per_pos_total"][p] += 1
                    agg[diff]["digit_total"]    += 1
                    if pred_digits_np[b, vi, p] == gold_digits_np[b, vi, p]:
                        agg[diff]["per_pos_correct"][p] += 1
                        agg[diff]["digit_correct"]    += 1
            if qi < n_max:
                q_all = bool(np.all(pred_digits_np[b, qi] == gold_digits_np[b, qi]))
                if q_all:
                    agg[diff]["query_correct"] += 1
            agg[diff]["n_puzzles"] += 1
        n_batches += 1
        if n_batches >= max_batches:
            break

    out = {}
    for d, v in agg.items():
        n = v["n_puzzles"]
        if n == 0: continue
        cell = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        q    = v["query_correct"] / n
        digit = v["digit_correct"] / max(v["digit_total"], 1)
        per_pos = v["per_pos_correct"] / np.maximum(v["per_pos_total"], 1)
        out[d] = {"cell_acc": cell, "query_acc": q, "digit_acc": digit,
                  "per_pos_acc": per_pos.tolist(), "n_puzzles": n}
    Tensor.training = True
    return out


def main():
    V108B_TASK_LOCAL = int(getenv("V108B_TASK", 0)) > 0
    assert V108B_TASK_LOCAL, "V108B_TASK=1 must be set"

    K          = int(getenv("V108B_K_MAX",      str(V108B_K_MAX)))
    BATCH      = int(getenv("BATCH",            "8"))
    STEPS      = int(getenv("STEPS",            "500"))
    LR         = float(getenv("LR",             "3e-5"))
    CKPT_EVERY = int(getenv("CKPT_EVERY",       "250"))
    EVAL_EVERY = int(getenv("EVAL_EVERY",       "100"))
    LOG_EVERY  = int(getenv("LOG_EVERY",        "10"))
    PER_BREATH_EVERY = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY   = int(getenv("GC_EVERY",         "50"))
    CKPT_LABEL = getenv("CKPT_LABEL",           "v108b_smoke")
    RESUME_FROM = getenv("RESUME_FROM",         "")
    PYTHIA_INIT = int(getenv("PYTHIA_INIT",     "1")) > 0
    SEED       = int(getenv("SEED",             "42"))
    N_CODE     = int(getenv("V108B_CODEBOOK_N", str(V108B_CODEBOOK_N)))
    IB_PATH    = getenv("V108B_IB_CENTROIDS",   V108B_IB_CENTROIDS)
    N_DIGITS   = int(getenv("V108B_N_DIGITS",   str(V108B_N_DIGITS)))
    HARD_LEVEL = int(getenv("V108B_HARD_BREATH_LEVEL", "0")) > 0
    VW         = float(getenv("V108B_VAR_LOSS_WEIGHT",  str(V108B_VAR_LOSS_WEIGHT)))
    FW         = float(getenv("V108B_FACTOR_AUX_WEIGHT", str(V108B_FACTOR_AUX_WEIGHT)))
    AW         = float(getenv("V108B_CALIB_WEIGHT",      str(V108B_CALIB_WEIGHT)))
    N_MAX      = int(getenv("V108B_N_MAX",      str(V108B_N_MAX)))
    F_MAX      = int(getenv("V108B_F_MAX",      str(V108B_F_MAX)))

    DIFFICULTY_FILTER = os.environ.get("V108B_DIFFICULTY_FILTER", "").strip() or None
    CURRICULUM        = int(getenv("V108B_CURRICULUM",        "0")) > 0
    CURRICULUM_ANNEAL = int(getenv("V108B_CURRICULUM_ANNEAL", "1000"))

    TRAIN_PATH   = getenv("V108B_TRAIN",       ".cache/factor_graph_train.jsonl")
    VAL_PATH     = getenv("V108B_VAL",         ".cache/factor_graph_test.jsonl")
    GSM8K_PATH   = getenv("V108B_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO  = float(getenv("V108B_GSM8K_RATIO", "0.5"))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", "10"))
    EVAL_BATCH   = int(getenv("EVAL_BATCH",   str(BATCH)))

    mode_str = "HARD breath→level" if HARD_LEVEL else "SOFT (all levels every breath)"
    print(f"=== v108b factor graph training (digit-decomp input + tree out, {mode_str}) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  N_DIGITS={N_DIGITS}  steps={STEPS}  lr={LR}")
    print(f"N_MAX={N_MAX}  F_MAX={F_MAX}  T_MAX={N_MAX+F_MAX}")
    print(f"loss weights:  var={VW}  factor_aux={FW}  calib={AW}")
    print(f"train={TRAIN_PATH}  val={VAL_PATH}  gsm8k={GSM8K_PATH}  ratio={GSM8K_RATIO}")
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

    attach_fg_params_v108b(
        model, hidden=cfg.hidden,
        n_max=N_MAX, f_max=F_MAX, k_max=K,
        n_digits=N_DIGITS, n_code=N_CODE, ib_centroids_path=IB_PATH,
    )
    Device[Device.DEFAULT].synchronize()

    params   = collect_fg_params_v108b(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"loading ckpt: {RESUME_FROM}")
        load_ckpt_v108b(model, RESUME_FROM)
        print("  loaded.")

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    synth_loader = FactorGraphLoaderV107(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER,
        curriculum=CURRICULUM, curriculum_anneal_steps=CURRICULUM_ANNEAL,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V108B_N_HEADS,
        seed=SEED,
    )
    gsm8k_records = load_gsm8k_records_v107(GSM8K_PATH)
    dual_loader = DualDataLoaderV107(
        synth_loader, gsm8k_records, gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V108B_N_HEADS,
        seed=SEED + 1,
    )
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V108B_N_HEADS,
        seed=SEED + 2,
    )

    ckpt_dir = ".cache/fg_v108b_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v108b(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FW, calib_weight=AW, var_loss_weight=VW,
        hard_breath_level=HARD_LEVEL,
        n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        grad_clip=1.0,
    )
    eval_fn = _compile_jit_fg_eval_v108b(
        model, K=K, B=EVAL_BATCH,
        n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
    )
    Tensor.training = True

    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_aux = log_n = 0.0

    for step in range(1, STEPS + 1):
        batch = dual_loader.sample_batch(step=step)

        node_kinds   = batch["node_kinds"]
        staging_mask = batch["staging_mask"]
        head_op_mask = batch["head_op_mask"]
        gold_bins    = batch["gold_bins"]
        obs_mask     = batch["observed_mask"]
        fgb_t        = batch["factor_gold_bin"]
        fv_t         = batch["factor_valid"]

        # Derive gold_digits + digit_init + factor_gold_digits
        gold_bins_np   = gold_bins.numpy()
        obs_np         = obs_mask.numpy()
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=N_DIGITS)
        digit_init_np  = build_digit_init(gold_digits_np, obs_np, n_digits=N_DIGITS)
        fac_bins_np    = fgb_t.numpy()
        fac_digits_np  = build_factor_digit_targets(fac_bins_np, n_digits=N_DIGITS)

        gold_digits_t = Tensor(gold_digits_np.astype(np.int64),
                               dtype=dtypes.int).contiguous().realize()
        digit_init_t  = Tensor(digit_init_np, dtype=dtypes.float).contiguous().realize()
        fac_digits_t  = Tensor(fac_digits_np.astype(np.int64),
                               dtype=dtypes.int).contiguous().realize()

        outs = step_fn(
            digit_init_t, node_kinds, staging_mask, head_op_mask,
            gold_digits_t, obs_mask, fac_digits_t, fv_t,
        )
        total_t     = outs[0]
        healthy_t   = outs[1]
        ce_t        = outs[2]
        aux_t       = outs[3]
        calib_t     = outs[4]
        cell_acc_t  = outs[5]
        query_acc_t = outs[6]
        pb_ce_ts    = outs[7:7 + K]

        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}: CE step skipped", flush=True)

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
            print(f"  evaluating ({EVAL_BATCHES} batches × B={EVAL_BATCH})...",
                  flush=True)
            results = evaluate_v108b(
                model, val_loader, K=K,
                max_batches=EVAL_BATCHES,
                eval_fn=eval_fn,
                n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
            )
            for d in DIFFICULTIES:
                if d not in results: continue
                v = results[d]
                pp = " ".join(f"{p:.2f}" for p in v["per_pos_acc"])
                print(f"  val[{d:6s}]: cell={v['cell_acc']:.3f} "
                      f"q={v['query_acc']:.3f} digit={v['digit_acc']:.3f} "
                      f"per_pos=[{pp}] n={v['n_puzzles']}", flush=True)

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(model_state_dict_v108b(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_v108b(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
