"""v108 factor graph training driver — tree-structured digit codebook.

Architecture: v107 single-token-per-variable substrate + 5-level tree codebook
readout (5 × 10 entries). Per-breath per-level CE ladder.

Modes (V108_HARD_BREATH_LEVEL):
  0 (soft, default): all levels supervised every breath; per-breath ladder
                     weights breaths only.
  1 (hard):          breath k → level k for k < n_digits; refinement breaths
                     k >= n_digits supervise all levels.

Warm-start: from v107 ckpt (backbone shared). Tree codebook is freshly
Fourier-initialized.

Env vars:
  V108_TASK=1
  V108_K_MAX=8
  V108_N_DIGITS=5
  V108_HARD_BREATH_LEVEL=0
  V108_VAR_LOSS_WEIGHT=1.0
  V108_CALIB_WEIGHT=0.05
  V108_FACTOR_AUX_WEIGHT=0.5
  V108_CODEBOOK_N=32
  V108_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz
  V108_TRAIN=.cache/factor_graph_train.jsonl
  V108_VAL=.cache/factor_graph_test.jsonl
  V108_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl
  V108_GSM8K_RATIO=0.5
  V108_DIFFICULTY_FILTER=easy
  BATCH=8
  STEPS=500
  LR=3e-5
  CKPT_EVERY=250
  EVAL_EVERY=100
  LOG_EVERY=10
  PER_BREATH_CE_EVERY=50
  CKPT_LABEL=v108_smoke
  RESUME_FROM=.cache/fg_v107_ckpts/v107_prod_step1000.safetensors
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
from mycelium.factor_graph_v108 import (
    V108_K_MAX, V108_N_MAX, V108_F_MAX, V108_N_HEADS, V108_N_DIGITS,
    V108_HARD_BREATH_LEVEL, V108_VAR_LOSS_WEIGHT,
    V108_CALIB_WEIGHT, V108_FACTOR_AUX_WEIGHT,
    V108_CODEBOOK_N, V108_IB_CENTROIDS,
    attach_fg_params_v108, fg_v108_parameters, fg_v108_state_dict,
    fg_breathing_forward_v108,
    _compile_jit_fg_step_v108, _compile_jit_fg_eval_v108,
    bins_to_digits_msd, digits_to_value_msd,
)
from mycelium.factor_graph_v107 import get_bin_values
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, DualDataLoaderV107, load_gsm8k_records_v107,
)

DIFFICULTIES = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def cast_layers_fp32(model):
    """Cast L0-L3 + shared weights from fp16 to fp32 for stable training."""
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


def collect_fg_params_v108(model) -> list[Tensor]:
    """Trainable parameters: shared L0-L3 attn/FFN + v108 fg params."""
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += fg_v108_parameters(model)
    return params


def model_state_dict_v108(model) -> dict:
    sd = {
        "ln_f.g": model.ln_f_g,
        "ln_f.b": model.ln_f_b,
    }
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        sd[f"shared.{a}"] = getattr(sw, a)
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            sd[f"phase{i}.{a}"] = getattr(layer, a)
    sd.update(fg_v108_state_dict(model))
    return sd


def load_ckpt_v108(model, path: str):
    """Load a v107 (or v108) ckpt into v108 model.

    v107 backbone keys (shared.*, phase*.*, fg_v107.*) → loaded directly.
    fg_v108.tree_codebook → fresh Fourier init kept if absent in ckpt.
    """
    sd = safe_load(path)
    targets = model_state_dict_v108(model)
    loaded = []
    missing = []
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

    v107_keys = [k for k in loaded if k.startswith("fg_v107.")]
    v108_keys = [k for k in loaded if k.startswith("fg_v108.")]
    backbone  = [k for k in loaded if k.startswith("shared.") or k.startswith("phase")]
    print(f"  loaded {len(backbone)} backbone + {len(v107_keys)} v107 + "
          f"{len(v108_keys)} v108 keys", flush=True)
    if any(k.startswith("fg_v108.") for k in missing):
        print(f"  v108 keys not in ckpt — using fresh init (expected for v107 warm-start)",
              flush=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_v108(model, loader, K: int,
                  max_batches: int = 20, eval_fn=None,
                  n_max: int = V108_N_MAX, f_max: int = V108_F_MAX,
                  n_digits: int = V108_N_DIGITS) -> dict:
    """Eval using tree decoder — cell correct iff ALL digits match."""
    Tensor.training = False
    agg = {}
    n_batches = 0

    for batch in loader.iter_eval(batch_size=loader.batch_size):
        domain_init  = batch["domain_init"]
        node_kinds   = batch["node_kinds"]
        staging_mask = batch["staging_mask"]
        head_op_mask = batch["head_op_mask"]
        gold_bins    = batch["gold_bins"]
        obs_mask     = batch["observed_mask"]
        query_idx_np = batch["query_idx"]
        picks        = batch["picks"]

        # Derive gold_digits from gold_bins
        gold_bins_np   = gold_bins.numpy()
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=n_digits)
        gold_digits_t  = Tensor(
            gold_digits_np.astype(np.int64), dtype=dtypes.int
        ).contiguous().realize()

        if eval_fn is not None:
            pred_t, _ = eval_fn(
                domain_init, node_kinds, staging_mask, head_op_mask,
                gold_digits_t, obs_mask,
            )
            pred_digits_np = pred_t.numpy()
        else:
            tree_lh, _, _, _ = fg_breathing_forward_v108(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )
            pred_digits_np = tree_lh[-1].argmax(axis=-1).realize().numpy()

        obs_np = obs_mask.numpy()
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
                # All-digits-match cell acc
                all_correct = bool(np.all(
                    pred_digits_np[b, vi] == gold_digits_np[b, vi]
                ))
                if all_correct:
                    agg[diff]["n_correct_unobs"] += 1
                # Per-position digit stats
                for p in range(n_digits):
                    agg[diff]["per_pos_total"][p] += 1
                    agg[diff]["digit_total"]    += 1
                    if pred_digits_np[b, vi, p] == gold_digits_np[b, vi, p]:
                        agg[diff]["per_pos_correct"][p] += 1
                        agg[diff]["digit_correct"]    += 1
            if qi < n_max:
                q_all_correct = bool(np.all(
                    pred_digits_np[b, qi] == gold_digits_np[b, qi]
                ))
                if q_all_correct:
                    agg[diff]["query_correct"] += 1
            agg[diff]["n_puzzles"] += 1

        n_batches += 1
        if n_batches >= max_batches:
            break

    out = {}
    for d, v in agg.items():
        n = v["n_puzzles"]
        if n == 0:
            continue
        cell_acc  = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        q_acc     = v["query_correct"] / n
        digit_acc = v["digit_correct"] / max(v["digit_total"], 1)
        per_pos   = v["per_pos_correct"] / np.maximum(v["per_pos_total"], 1)
        out[d] = {
            "cell_acc": cell_acc, "query_acc": q_acc,
            "digit_acc": digit_acc,
            "per_pos_acc": per_pos.tolist(),
            "n_puzzles": n,
        }

    Tensor.training = True
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    V108_TASK_LOCAL = int(getenv("V108_TASK", 0)) > 0
    assert V108_TASK_LOCAL, "V108_TASK=1 must be set"

    K          = int(getenv("V108_K_MAX",      str(V108_K_MAX)))
    BATCH      = int(getenv("BATCH",           "8"))
    STEPS      = int(getenv("STEPS",           "500"))
    LR         = float(getenv("LR",            "3e-5"))
    CKPT_EVERY = int(getenv("CKPT_EVERY",      "250"))
    EVAL_EVERY = int(getenv("EVAL_EVERY",      "100"))
    LOG_EVERY  = int(getenv("LOG_EVERY",       "10"))
    PER_BREATH_EVERY = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY   = int(getenv("GC_EVERY",        "50"))
    CKPT_LABEL = getenv("CKPT_LABEL",          "v108_smoke")
    RESUME_FROM = getenv("RESUME_FROM",        "")
    PYTHIA_INIT = int(getenv("PYTHIA_INIT",    "1")) > 0
    SEED       = int(getenv("SEED",            "42"))
    N_CODE     = int(getenv("V108_CODEBOOK_N", str(V108_CODEBOOK_N)))
    IB_PATH    = getenv("V108_IB_CENTROIDS",   V108_IB_CENTROIDS)
    N_DIGITS   = int(getenv("V108_N_DIGITS",   str(V108_N_DIGITS)))
    HARD_LEVEL = int(getenv("V108_HARD_BREATH_LEVEL", "0")) > 0
    VW         = float(getenv("V108_VAR_LOSS_WEIGHT",  str(V108_VAR_LOSS_WEIGHT)))
    FW         = float(getenv("V108_FACTOR_AUX_WEIGHT", str(V108_FACTOR_AUX_WEIGHT)))
    AW         = float(getenv("V108_CALIB_WEIGHT",      str(V108_CALIB_WEIGHT)))
    N_MAX      = int(getenv("V108_N_MAX",      str(V108_N_MAX)))
    F_MAX      = int(getenv("V108_F_MAX",      str(V108_F_MAX)))

    DIFFICULTY_FILTER = os.environ.get("V108_DIFFICULTY_FILTER", "").strip() or None
    CURRICULUM        = int(getenv("V108_CURRICULUM",        "0")) > 0
    CURRICULUM_ANNEAL = int(getenv("V108_CURRICULUM_ANNEAL", "1000"))

    TRAIN_PATH   = getenv("V108_TRAIN",       ".cache/factor_graph_train.jsonl")
    VAL_PATH     = getenv("V108_VAL",         ".cache/factor_graph_test.jsonl")
    GSM8K_PATH   = getenv("V108_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO  = float(getenv("V108_GSM8K_RATIO", "0.5"))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", "10"))
    EVAL_BATCH   = int(getenv("EVAL_BATCH",   str(BATCH)))

    mode_str = "HARD breath→level" if HARD_LEVEL else "SOFT (all levels every breath)"
    print(f"=== v108 factor graph training (tree codebook, {mode_str}) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  N_DIGITS={N_DIGITS}  "
          f"steps={STEPS}  lr={LR}")
    print(f"N_MAX={N_MAX}  F_MAX={F_MAX}  T_MAX={N_MAX+F_MAX}")
    print(f"loss weights:  var={VW}  factor_aux={FW}  calib={AW}")
    print(f"difficulty_filter={DIFFICULTY_FILTER}  curriculum={CURRICULUM}")
    print(f"train={TRAIN_PATH}  val={VAL_PATH}")
    print(f"gsm8k={GSM8K_PATH}  gsm8k_ratio={GSM8K_RATIO}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer (PYTHIA_INIT={PYTHIA_INIT})...")
    if PYTHIA_INIT:
        sd = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)

    attach_fg_params_v108(
        model, hidden=cfg.hidden,
        n_max=N_MAX, f_max=F_MAX, k_max=K,
        n_digits=N_DIGITS, n_code=N_CODE, ib_centroids_path=IB_PATH,
    )
    Device[Device.DEFAULT].synchronize()

    params   = collect_fg_params_v108(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"loading ckpt: {RESUME_FROM}")
        load_ckpt_v108(model, RESUME_FROM)
        print("  loaded.")

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    # Data loaders (reuse v107's loader — gold_bins are computed there)
    synth_loader = FactorGraphLoaderV107(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER,
        curriculum=CURRICULUM,
        curriculum_anneal_steps=CURRICULUM_ANNEAL,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V108_N_HEADS,
        seed=SEED,
    )
    gsm8k_records = load_gsm8k_records_v107(GSM8K_PATH)
    dual_loader = DualDataLoaderV107(
        synth_loader, gsm8k_records,
        gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V108_N_HEADS,
        seed=SEED + 1,
    )
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V108_N_HEADS,
        seed=SEED + 2,
    )

    ckpt_dir = ".cache/fg_v108_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v108(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FW, calib_weight=AW, var_loss_weight=VW,
        hard_breath_level=HARD_LEVEL,
        n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        grad_clip=1.0,
    )
    eval_fn = _compile_jit_fg_eval_v108(
        model, K=K, B=EVAL_BATCH,
        n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
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

        # Compute gold_digits on host (numpy) then ship to device
        gold_bins_np   = gold_bins.numpy()
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=N_DIGITS)
        gold_digits_t  = Tensor(
            gold_digits_np.astype(np.int64), dtype=dtypes.int,
        ).contiguous().realize()

        outs = step_fn(
            domain_init, node_kinds, staging_mask, head_op_mask,
            gold_digits_t, gold_bins, obs_mask,
            fgb_t, fv_t,
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
                f"ce={log_ce/log_n:.4f} "
                f"aux={log_aux/log_n:.4f} "
                f"calib={log_calib/log_n:.4f}  "
                f"cell_acc={float(cell_acc_t.numpy()):.3f}  "
                f"({dt:.1f}s, {dt/step:.2f}s/step)",
                flush=True,
            )
            log_loss = log_ce = log_calib = log_aux = log_n = 0.0

        if step % PER_BREATH_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            if K <= 8:
                pb_str = " ".join(f"{v:.3f}" for v in pb_ce)
            else:
                head = " ".join(f"{v:.3f}" for v in pb_ce[:5])
                tail = " ".join(f"{v:.3f}" for v in pb_ce[-5:])
                pb_str = f"{head} ... {tail}"
            ca = float(cell_acc_t.numpy())
            qa = float(query_acc_t.numpy())
            print(
                f"  per_breath_ce[B0..B{K-1}]: {pb_str}  "
                f"(cell_acc={ca:.3f} query_acc={qa:.3f})",
                flush=True,
            )
            if K > 1 and len(pb_ce) >= 2:
                ladder_delta = pb_ce[0] - pb_ce[-1]
                tag = "OK" if ladder_delta > 0.1 else "target > 0.1"
                print(f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} ({tag})",
                      flush=True)

        if step % EVAL_EVERY == 0:
            print(f"  evaluating ({EVAL_BATCHES} batches × B={EVAL_BATCH})...",
                  flush=True)
            results = evaluate_v108(
                model, val_loader, K=K,
                max_batches=EVAL_BATCHES,
                eval_fn=eval_fn,
                n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
            )
            for d in DIFFICULTIES:
                if d not in results:
                    continue
                v = results[d]
                pp = " ".join(f"{p:.2f}" for p in v["per_pos_acc"])
                print(
                    f"  val[{d:6s}]: cell={v['cell_acc']:.3f} "
                    f"q={v['query_acc']:.3f} "
                    f"digit={v['digit_acc']:.3f} "
                    f"per_pos=[{pp}] "
                    f"n={v['n_puzzles']}",
                    flush=True,
                )

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(model_state_dict_v108(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_v108(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
