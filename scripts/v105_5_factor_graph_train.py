"""v105.5 factor graph training driver — v105.3 + hierarchical codebooks.

Extends v105.3 (LSD-first array + digit RoPE + projection waist + IB codebook
+ AR digit decoding) with four targeted additions:

  1. Magnitude head — per-cell 4-way classification of "how many digits is
     this number?" (1 / 2 / 3 / 4+).  Contributes to total loss via
     V105_5_MAGNITUDE_WEIGHT (default 0.3).
  2. Per-position digit codebooks — 5 distinct (10, hidden) codebooks
     instead of 1 shared.
  3. Hierarchical IB attention — 4-way family gate × 32-leaf attention.
  4. Soft magnitude-derived valid mask applied to factor_aux loss.

Warm-start from v104_prod_step3000.safetensors:
  - Copies backbone (shared.*, phase*.*, ln_f.*)
  - Copies fg_v104.codebook → fg_v105_5.ib_codebook
  - Fresh-inits per-position digit codebooks, family centroids, magnitude head
  - Zero-init delta_gate_quant + W_expand keep hierarchical IB + waist at zero
    residual contribution at step 0.

Env vars (subset; full list in factor_graph_v105_5.py module docstring):
  V105_5_TASK=1
  V105_5_K_MAX=8
  V105_5_N_DIGITS=5
  V105_5_N_MAX=16
  V105_5_F_MAX=8
  V105_5_WAIST=512
  V105_5_CODEBOOK_N=32
  V105_5_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz
  V105_5_IB_TREE=.cache/ib_tree_gsm8k_partial.json
  V105_5_MAGNITUDE_WEIGHT=0.3
  V105_5_AR_DIGITS=1
  V105_5_AR_MSD_FIRST=0
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
from mycelium.factor_graph_v105_5 import (
    V105_5_K_MAX, V105_5_N_MAX, V105_5_F_MAX, V105_5_N_DIGITS, V105_5_N_HEADS,
    V105_5_ENERGY_WEIGHT, V105_5_FACTOR_AUX_WEIGHT, V105_5_CALIB_WEIGHT,
    V105_5_MAGNITUDE_WEIGHT, V105_5_N_MAGNITUDE,
    V105_5_WAIST, V105_5_CODEBOOK_N,
    V105_AUX_DISTINCT_WEIGHT, V105_5_VAR_LOSS_WEIGHT,
    V105_8_PER_NUMBER_READOUT, V105_8_N_NUMBER_BINS,
    V105_9_AR_DIGIT_DECODER, V105_9_AR_COND_SCALE,
    V105_10_DUAL_READOUT, V105_10_DIGIT_WEIGHT,
    attach_fg_params_v105_5, fg_v105_5_parameters, fg_v105_5_state_dict,
    fg_breathing_forward_v105_5, load_ckpt_v105_5,
    _compile_jit_fg_step_v105_5, _compile_jit_fg_eval_v105_5,
    _compile_jit_fg_eval_v105_8, _compile_jit_fg_eval_v105_9,
    value_to_digits_lsd, digits_to_value_lsd,
)
# v105.5 data loaders: LSD-first encoding + digit_valid_mask + magnitude_target
from mycelium.factor_graph_data_v105_5 import (
    FactorGraphLoaderV105_5, DualDataLoaderV105_5, load_gsm8k_records_v105_5,
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


def collect_fg_params_v105_5(model) -> list[Tensor]:
    """Trainable params: shared L0-L3 attn/FFN + v105.5 fg-specific params."""
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += fg_v105_5_parameters(model)
    return params


def model_state_dict_v105_5(model) -> dict:
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
    sd.update(fg_v105_5_state_dict(model))
    return sd


def evaluate_v105_5(
    model, loader: FactorGraphLoaderV105_5,
    K: int, max_batches: int = 20,
    eval_fn=None,
    n_max: int = V105_5_N_MAX,
    f_max: int = V105_5_F_MAX,
    n_digits: int = V105_5_N_DIGITS,
) -> dict:
    """Run eval on up to max_batches batches. Returns per-difficulty stats.

    Variable is counted "correct" if all VALID digits match (invalid positions,
    leading-zero padding above the most-significant digit, are treated as
    automatically correct since the model isn't asked to predict them).
    """
    Tensor.training = False
    agg = {}
    n_batches = 0

    for batch in loader.iter_eval(batch_size=loader.batch_size):
        digit_init    = batch["digit_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        gold_digits   = batch["gold_digits"]
        obs_mask      = batch["observed_mask"]
        valid_mask    = batch["digit_valid_mask"]
        query_idx_np  = batch["query_idx"]
        picks         = batch["picks"]

        if eval_fn is not None:
            pred_dg_t, _cell_acc_t = eval_fn(
                digit_init, node_kinds, staging_mask, head_op_mask,
                gold_digits, obs_mask, valid_mask,
            )
            pred_dg_np = pred_dg_t.numpy()
        else:
            dig_lh, _, _, _, _, _, _ = fg_breathing_forward_v105_5(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )
            pred_dg_np = dig_lh[-1].argmax(axis=-1).realize().numpy()

        gold_dg_np = gold_digits.numpy()
        obs_np     = obs_mask.numpy()
        valid_np   = valid_mask.numpy()

        for b in range(len(picks)):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                             "query_correct": 0, "n_puzzles": 0}
            qi  = int(query_idx_np[b])
            nv  = int(batch["n_vars_total"][b])
            for vi in range(min(nv, n_max)):
                if obs_np[b, vi] == 0:
                    agg[diff]["n_unobs"] += 1
                    v_valid = valid_np[b, vi].astype(bool)
                    if v_valid.any():
                        if np.all(pred_dg_np[b, vi, v_valid] == gold_dg_np[b, vi, v_valid]):
                            agg[diff]["n_correct_unobs"] += 1
                    else:
                        # Edge case (shouldn't happen — every value uses at least pos 0).
                        if np.all(pred_dg_np[b, vi] == gold_dg_np[b, vi]):
                            agg[diff]["n_correct_unobs"] += 1
            if qi < n_max:
                q_valid = valid_np[b, qi].astype(bool)
                if q_valid.any():
                    if np.all(pred_dg_np[b, qi, q_valid] == gold_dg_np[b, qi, q_valid]):
                        agg[diff]["query_correct"] += 1
                else:
                    if np.all(pred_dg_np[b, qi] == gold_dg_np[b, qi]):
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
        cell_acc = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        q_acc    = v["query_correct"] / n
        out[d]   = {"cell_acc": cell_acc, "query_acc": q_acc, "n_puzzles": n}

    Tensor.training = True
    return out


def evaluate_v105_8(
    model, loader: FactorGraphLoaderV105_5,
    K: int, max_batches: int = 20,
    eval_fn=None,
    n_max: int = V105_5_N_MAX,
    f_max: int = V105_5_F_MAX,
    n_digits: int = V105_5_N_DIGITS,
) -> dict:
    """v105.8 eval — per-NUMBER bin classification accuracy.

    A cell is "correct" if predicted bin == gold bin (number_bin_target).
    No per-digit decoding: the model outputs a single bin per cell.
    """
    Tensor.training = False
    agg = {}
    n_batches = 0

    for batch in loader.iter_eval(batch_size=loader.batch_size):
        digit_init    = batch["digit_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        obs_mask      = batch["observed_mask"]
        valid_mask    = batch["digit_valid_mask"]
        num_bin_tgt   = batch["number_bin_target"]
        query_idx_np  = batch["query_idx"]
        picks         = batch["picks"]

        if eval_fn is not None:
            pred_bin_t, _cell_acc_t = eval_fn(
                digit_init, node_kinds, staging_mask, head_op_mask,
                num_bin_tgt, obs_mask, valid_mask,
            )
            pred_bin_np = pred_bin_t.numpy()
        else:
            _, _, _, _, _, num_logits, _ = fg_breathing_forward_v105_5(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )
            pred_bin_np = num_logits.argmax(axis=-1).realize().numpy()

        gold_bin_np = num_bin_tgt.numpy()
        obs_np      = obs_mask.numpy()

        for b in range(len(picks)):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                             "query_correct": 0, "n_puzzles": 0}
            qi  = int(query_idx_np[b])
            nv  = int(batch["n_vars_total"][b])
            for vi in range(min(nv, n_max)):
                if obs_np[b, vi] == 0:
                    agg[diff]["n_unobs"] += 1
                    if int(pred_bin_np[b, vi]) == int(gold_bin_np[b, vi]):
                        agg[diff]["n_correct_unobs"] += 1
            if qi < n_max:
                if int(pred_bin_np[b, qi]) == int(gold_bin_np[b, qi]):
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
        cell_acc = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        q_acc    = v["query_correct"] / n
        out[d]   = {"cell_acc": cell_acc, "query_acc": q_acc, "n_puzzles": n}

    Tensor.training = True
    return out


def evaluate_v105_9(
    model, loader: FactorGraphLoaderV105_5,
    K: int, max_batches: int = 20,
    eval_fn=None,
    n_max: int = V105_5_N_MAX,
    f_max: int = V105_5_F_MAX,
    n_digits: int = V105_5_N_DIGITS,
) -> dict:
    """v105.9 eval — per-cell digit accuracy via pooled-AR digit decoder.

    Mechanically identical to `evaluate_v105_5` (same gold_digits, same
    cell-correct rule), but the predictions come from the pooled-cell AR
    decoder rather than per-position hidden states. Per-digit accuracy
    measures whether digits decoded from cell_hidden match gold.
    """
    Tensor.training = False
    agg = {}
    n_batches = 0

    for batch in loader.iter_eval(batch_size=loader.batch_size):
        digit_init    = batch["digit_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        gold_digits   = batch["gold_digits"]
        obs_mask      = batch["observed_mask"]
        valid_mask    = batch["digit_valid_mask"]
        query_idx_np  = batch["query_idx"]
        picks         = batch["picks"]

        if eval_fn is not None:
            pred_dg_t, _cell_acc_t = eval_fn(
                digit_init, node_kinds, staging_mask, head_op_mask,
                gold_digits, obs_mask, valid_mask,
            )
            pred_dg_np = pred_dg_t.numpy()
        else:
            _, _, _, _, _, _, dlp = fg_breathing_forward_v105_5(
                model, digit_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            )
            pred_dg_np = dlp.argmax(axis=-1).realize().numpy()

        gold_dg_np = gold_digits.numpy()
        obs_np     = obs_mask.numpy()
        valid_np   = valid_mask.numpy()

        for b in range(len(picks)):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                             "query_correct": 0, "n_puzzles": 0}
            qi  = int(query_idx_np[b])
            nv  = int(batch["n_vars_total"][b])
            for vi in range(min(nv, n_max)):
                if obs_np[b, vi] == 0:
                    agg[diff]["n_unobs"] += 1
                    v_valid = valid_np[b, vi].astype(bool)
                    if v_valid.any():
                        if np.all(pred_dg_np[b, vi, v_valid] == gold_dg_np[b, vi, v_valid]):
                            agg[diff]["n_correct_unobs"] += 1
                    else:
                        if np.all(pred_dg_np[b, vi] == gold_dg_np[b, vi]):
                            agg[diff]["n_correct_unobs"] += 1
            if qi < n_max:
                q_valid = valid_np[b, qi].astype(bool)
                if q_valid.any():
                    if np.all(pred_dg_np[b, qi, q_valid] == gold_dg_np[b, qi, q_valid]):
                        agg[diff]["query_correct"] += 1
                else:
                    if np.all(pred_dg_np[b, qi] == gold_dg_np[b, qi]):
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
        cell_acc = v["n_correct_unobs"] / max(v["n_unobs"], 1)
        q_acc    = v["query_correct"] / n
        out[d]   = {"cell_acc": cell_acc, "query_acc": q_acc, "n_puzzles": n}

    Tensor.training = True
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    V105_5_TASK_LOCAL = int(getenv("V105_5_TASK", 0)) > 0
    assert V105_5_TASK_LOCAL, "V105_5_TASK=1 must be set"

    K        = int(getenv("V105_5_K_MAX",    str(V105_5_K_MAX)))
    N_DIGITS = int(getenv("V105_5_N_DIGITS", str(V105_5_N_DIGITS)))
    N_MAX    = int(getenv("V105_5_N_MAX",    str(V105_5_N_MAX)))
    F_MAX    = int(getenv("V105_5_F_MAX",    str(V105_5_F_MAX)))
    WAIST    = int(getenv("V105_5_WAIST",    str(V105_5_WAIST)))
    N_CODE   = int(getenv("V105_5_CODEBOOK_N", str(V105_5_CODEBOOK_N)))
    T_MAX    = N_MAX * N_DIGITS + F_MAX

    BATCH            = int(getenv("BATCH",              "8"))
    STEPS            = int(getenv("STEPS",              "3000"))
    LR               = float(getenv("LR",               "3e-5"))
    CKPT_EVERY       = int(getenv("CKPT_EVERY",         "500"))
    EVAL_EVERY       = int(getenv("EVAL_EVERY",         "250"))
    LOG_EVERY        = int(getenv("LOG_EVERY",          "10"))
    PER_BREATH_EVERY = int(getenv("PER_BREATH_CE_EVERY","50"))
    GC_EVERY         = int(getenv("GC_EVERY",           "50"))
    CKPT_LABEL       = getenv("CKPT_LABEL",             "v105_5_prod")
    RESUME_FROM      = getenv("RESUME_FROM",            "")
    PYTHIA_INIT      = int(getenv("PYTHIA_INIT",        "1")) > 0
    SEED             = int(getenv("SEED",               "42"))
    EVAL_BATCHES     = int(getenv("EVAL_BATCHES",       "20"))
    EVAL_BATCH       = int(getenv("EVAL_BATCH",         str(BATCH)))

    ENERGY_WEIGHT     = float(getenv("V105_5_ENERGY_WEIGHT",     str(V105_5_ENERGY_WEIGHT)))
    FACTOR_AUX_WEIGHT = float(getenv("V105_5_FACTOR_AUX_WEIGHT", str(V105_5_FACTOR_AUX_WEIGHT)))
    CALIB_WEIGHT      = float(getenv("V105_5_CALIB_WEIGHT",      str(V105_5_CALIB_WEIGHT)))
    MAGNITUDE_WEIGHT  = float(getenv("V105_5_MAGNITUDE_WEIGHT",  str(V105_5_MAGNITUDE_WEIGHT)))
    AUX_DISTINCT_WEIGHT = float(getenv("V105_AUX_DISTINCT_WEIGHT", str(V105_AUX_DISTINCT_WEIGHT)))
    VAR_LOSS_WEIGHT     = float(getenv("V105_5_VAR_LOSS_WEIGHT", str(V105_5_VAR_LOSS_WEIGHT)))

    # v105.8 — per-NUMBER readout. When enabled, force per-digit and
    # magnitude/energy/aux_distinct weights to 0 (Python-side override). The
    # per-NUMBER CE loss supersedes per-digit CE as the variable supervision.
    # factor_aux is KEPT (it's already per-NUMBER MSE — complementary to
    # per-NUMBER CE).
    if V105_8_PER_NUMBER_READOUT:
        VAR_LOSS_WEIGHT     = 0.0
        MAGNITUDE_WEIGHT    = 0.0
        ENERGY_WEIGHT       = 0.0
        AUX_DISTINCT_WEIGHT = 0.0
        print(
            f"[v105.8] PER_NUMBER_READOUT=1 → forcing var_loss_weight=0, "
            f"magnitude_weight=0, energy_weight=0, aux_distinct_weight=0. "
            f"factor_aux_weight={FACTOR_AUX_WEIGHT} (kept). "
            f"calib_weight={CALIB_WEIGHT} (kept). "
            f"n_number_bins={V105_8_N_NUMBER_BINS}.",
            flush=True,
        )

    # v105.9 — pooled-AR digit decoder. When enabled, force the per-POSITION
    # var_loss / energy / aux_distinct to 0. The pooled-AR per-digit CE is
    # added unconditionally inside the JIT step with weight 1.0; it is the
    # variable supervision. magnitude_weight is KEPT (magnitude is decoded
    # from cell_hidden via a separate head — complementary). factor_aux is
    # KEPT (per-NUMBER MSE over factor cells — complementary to pooled-digit
    # CE). energy depended on per-position digit logits → drop.
    if V105_9_AR_DIGIT_DECODER:
        VAR_LOSS_WEIGHT     = 0.0
        ENERGY_WEIGHT       = 0.0
        AUX_DISTINCT_WEIGHT = 0.0
        print(
            f"[v105.9] AR_DIGIT_DECODER=1 → forcing var_loss_weight=0, "
            f"energy_weight=0, aux_distinct_weight=0. "
            f"magnitude_weight={MAGNITUDE_WEIGHT} (kept). "
            f"factor_aux_weight={FACTOR_AUX_WEIGHT} (kept). "
            f"calib_weight={CALIB_WEIGHT} (kept). "
            f"ar_cond_scale={V105_9_AR_COND_SCALE}.",
            flush=True,
        )

    # v105.10 — DUAL READOUT (v105.8 + v105.9). When enabled, both v105.8 and
    # v105.9 paths are active. Apply the same Python-side overrides as v105.8/9:
    # zero out var_loss / magnitude / energy / aux_distinct (we already did this
    # above through V105_8_PER_NUMBER_READOUT and V105_9_AR_DIGIT_DECODER), and
    # KEEP factor_aux_weight=1.0. Pooled-AR digit CE is weighted inside the JIT
    # by V105_10_DIGIT_WEIGHT (default 0.3) instead of v105.9's default 1.0.
    if V105_10_DUAL_READOUT:
        VAR_LOSS_WEIGHT     = 0.0
        MAGNITUDE_WEIGHT    = 0.0
        ENERGY_WEIGHT       = 0.0
        AUX_DISTINCT_WEIGHT = 0.0
        print(
            f"[v105.10] DUAL_READOUT=1 (v105.8 + v105.9) → forcing "
            f"var_loss_weight=0, magnitude_weight=0, energy_weight=0, "
            f"aux_distinct_weight=0. "
            f"factor_aux_weight={FACTOR_AUX_WEIGHT} (kept). "
            f"calib_weight={CALIB_WEIGHT} (kept). "
            f"n_number_bins={V105_8_N_NUMBER_BINS} "
            f"digit_weight={V105_10_DIGIT_WEIGHT} "
            f"ar_cond_scale={V105_9_AR_COND_SCALE}.",
            flush=True,
        )

    DIFFICULTY_FILTER = os.environ.get("V105_DIFFICULTY_FILTER", "").strip() or None
    CURRICULUM        = int(getenv("V105_CURRICULUM",        "0")) > 0
    CURRICULUM_ANNEAL = int(getenv("V105_CURRICULUM_ANNEAL", "1000"))

    TRAIN_PATH  = getenv("V105_TRAIN",       ".cache/factor_graph_train.jsonl")
    VAL_PATH    = getenv("V105_VAL",         ".cache/factor_graph_test.jsonl")
    GSM8K_PATH  = getenv("V105_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO = float(getenv("V105_GSM8K_RATIO", "0.5"))

    print("=== v105.5 factor graph training (v105.3 + hierarchical codebooks + magnitude head) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  N_DIGITS={N_DIGITS}  steps={STEPS}  lr={LR}")
    print(f"N_MAX={N_MAX}  F_MAX={F_MAX}  T_MAX={T_MAX}  waist={WAIST}  n_code={N_CODE}")
    print(f"energy_weight={ENERGY_WEIGHT}  factor_aux_weight={FACTOR_AUX_WEIGHT}  "
          f"calib_weight={CALIB_WEIGHT}  magnitude_weight={MAGNITUDE_WEIGHT}  "
          f"aux_distinct_weight={AUX_DISTINCT_WEIGHT}")
    print(f"difficulty_filter={DIFFICULTY_FILTER}  curriculum={CURRICULUM}")
    print(f"train={TRAIN_PATH}  val={VAL_PATH}")
    print(f"gsm8k_train={GSM8K_PATH}  gsm8k_ratio={GSM8K_RATIO}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print(f"loading Pythia-410M → breathing transformer (PYTHIA_INIT={PYTHIA_INIT})...")
    if PYTHIA_INIT:
        sd    = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)

    attach_fg_params_v105_5(
        model, hidden=cfg.hidden,
        n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX, k_max=K,
        waist=WAIST, n_code=N_CODE,
    )
    Device[Device.DEFAULT].synchronize()

    if RESUME_FROM:
        print(f"warm-starting from: {RESUME_FROM}")
        load_ckpt_v105_5(model, RESUME_FROM)

    params   = collect_fg_params_v105_5(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    synth_loader = FactorGraphLoaderV105_5(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER,
        curriculum=CURRICULUM,
        curriculum_anneal_steps=CURRICULUM_ANNEAL,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V105_5_N_HEADS,
        n_digits=N_DIGITS, seed=SEED,
    )

    gsm8k_records = load_gsm8k_records_v105_5(
        GSM8K_PATH, n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX,
    )

    dual_loader = DualDataLoaderV105_5(
        synth_loader, gsm8k_records,
        gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V105_5_N_HEADS,
        n_digits=N_DIGITS, seed=SEED + 1,
    )

    val_loader = FactorGraphLoaderV105_5(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None,
        curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V105_5_N_HEADS,
        n_digits=N_DIGITS, seed=SEED + 2,
    )

    ckpt_dir = ".cache/fg_v105_5_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v105_5(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FACTOR_AUX_WEIGHT,
        calib_weight=CALIB_WEIGHT,
        energy_weight=ENERGY_WEIGHT,
        magnitude_weight=MAGNITUDE_WEIGHT,
        aux_distinct_weight=AUX_DISTINCT_WEIGHT,
        var_loss_weight=VAR_LOSS_WEIGHT,
        n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        n_magnitude=V105_5_N_MAGNITUDE,
        grad_clip=1.0,
    )
    # v105.10 dual readout: compile BOTH eval JITs so we can run side-by-side
    # eval at EVAL_EVERY (200-bin number readout + pooled-AR digit decoder).
    # For pure v105.8 or v105.9 only the relevant eval JIT is built.
    eval_fn_v8: callable | None = None
    eval_fn_v9: callable | None = None
    eval_fn: callable | None = None
    if V105_10_DUAL_READOUT:
        eval_fn_v8 = _compile_jit_fg_eval_v105_8(
            model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )
        eval_fn_v9 = _compile_jit_fg_eval_v105_9(
            model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )
    elif V105_8_PER_NUMBER_READOUT:
        eval_fn = _compile_jit_fg_eval_v105_8(
            model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )
    elif V105_9_AR_DIGIT_DECODER:
        eval_fn = _compile_jit_fg_eval_v105_9(
            model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )
    else:
        eval_fn = _compile_jit_fg_eval_v105_5(
            model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
        )
    Tensor.training = True

    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_aux = log_energy = log_mag = log_magacc = log_distinct = log_n = 0.0
    log_numce = log_numacc = 0.0
    log_pool_ce = log_pool_acc = 0.0

    for step in range(1, STEPS + 1):
        batch = dual_loader.sample_batch(step=step)

        digit_init             = batch["digit_init"]
        node_kinds             = batch["node_kinds"]
        staging_mask           = batch["staging_mask"]
        head_op_mask           = batch["head_op_mask"]
        gold_digits            = batch["gold_digits"]
        obs_mask               = batch["observed_mask"]
        factor_gold_dg         = batch["factor_gold_dg"]
        factor_valid           = batch["factor_valid"]
        factor_types           = batch["factor_types"]
        factor_args            = batch["factor_args"]
        digit_valid_mask        = batch["digit_valid_mask"]
        factor_digit_valid_mask = batch["factor_digit_valid_mask"]
        magnitude_target        = batch["magnitude_target"]
        number_bin_target       = batch["number_bin_target"]

        outs = step_fn(
            digit_init, node_kinds, staging_mask, head_op_mask,
            gold_digits, obs_mask, factor_gold_dg, factor_valid,
            factor_types, factor_args,
            digit_valid_mask, factor_digit_valid_mask,
            magnitude_target, number_bin_target,
        )
        total_t          = outs[0]
        healthy_t        = outs[1]
        ce_t             = outs[2]
        aux_t            = outs[3]
        calib_t          = outs[4]
        energy_t         = outs[5]
        magnitude_t      = outs[6]
        mag_acc_t        = outs[7]
        cell_acc_t       = outs[8]
        query_acc_t      = outs[9]
        aux_distinct_t   = outs[10]
        number_ce_t      = outs[11]
        number_acc_t     = outs[12]
        var_loss_pooled_t = outs[13]
        pooled_cell_acc_t = outs[14]
        pb_ce_ts         = outs[15:15 + K]

        if float(healthy_t.numpy()) < 0.5:
            print(f"[NaN-skip] step {step}: gradient step skipped", flush=True)

        log_loss     += float(total_t.numpy())
        log_ce       += float(ce_t.numpy())
        log_calib    += float(calib_t.numpy())
        log_aux      += float(aux_t.numpy())
        log_energy   += float(energy_t.numpy())
        log_mag      += float(magnitude_t.numpy())
        log_magacc   += float(mag_acc_t.numpy())
        log_distinct += float(aux_distinct_t.numpy())
        log_numce    += float(number_ce_t.numpy())
        log_numacc   += float(number_acc_t.numpy())
        log_pool_ce  += float(var_loss_pooled_t.numpy())
        log_pool_acc += float(pooled_cell_acc_t.numpy())
        log_n        += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            distinct_fragment = (
                f"distinct={log_distinct/log_n:.4f} " if AUX_DISTINCT_WEIGHT > 0 else ""
            )
            num_fragment = (
                f"num_ce={log_numce/log_n:.4f} num_acc={log_numacc/log_n:.3f} "
                if V105_8_PER_NUMBER_READOUT else ""
            )
            pool_fragment = (
                f"pool_ce={log_pool_ce/log_n:.4f} "
                f"pool_acc={log_pool_acc/log_n:.3f} "
                if V105_9_AR_DIGIT_DECODER else ""
            )
            print(
                f"[step {step:5d}] loss={log_loss/log_n:.4f} "
                f"ce={log_ce/log_n:.4f} "
                f"aux={log_aux/log_n:.4f} "
                f"mag={log_mag/log_n:.4f} "
                f"mag_acc={log_magacc/log_n:.3f} "
                f"calib={log_calib/log_n:.4f} "
                f"energy={log_energy/log_n:.4f} "
                f"{distinct_fragment}"
                f"{num_fragment}"
                f"{pool_fragment}"
                f"({dt:.1f}s, {dt/step:.2f}s/step)",
                flush=True,
            )
            log_loss = log_ce = log_calib = log_aux = log_energy = log_mag = log_magacc = log_distinct = log_n = 0.0
            log_numce = log_numacc = 0.0
            log_pool_ce = log_pool_acc = 0.0

        if step % PER_BREATH_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            pb_str = " ".join(f"{v:.3f}" for v in pb_ce)
            ca = float(cell_acc_t.numpy())
            qa = float(query_acc_t.numpy())
            ma = float(mag_acc_t.numpy())
            print(
                f"  per_breath_ce[B0..B{K-1}]: {pb_str}  "
                f"(cell_acc={ca:.3f} query_acc={qa:.3f} mag_acc={ma:.3f})",
                flush=True,
            )
            if K > 1 and len(pb_ce) >= 2:
                delta = pb_ce[0] - pb_ce[-1]
                status = "LADDER" if delta > 0.1 else "flat"
                print(
                    f"  [{status}] B0-B{K-1} delta = {delta:.3f} (target > 0.1)",
                    flush=True,
                )

        if step % EVAL_EVERY == 0:
            print(f"  evaluating ({EVAL_BATCHES} batches × B={EVAL_BATCH})...", flush=True)
            dual_done = False
            if V105_10_DUAL_READOUT:
                # Run BOTH evals: 200-bin number readout AND pooled-AR digit decoder.
                results_v8 = evaluate_v105_8(
                    model, val_loader, K=K,
                    max_batches=EVAL_BATCHES,
                    eval_fn=eval_fn_v8,
                    n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
                )
                results_v9 = evaluate_v105_9(
                    model, val_loader, K=K,
                    max_batches=EVAL_BATCHES,
                    eval_fn=eval_fn_v9,
                    n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
                )
                print("  [v105.10 dual readout eval — number-bin / pooled-digit side-by-side]", flush=True)
                for d in DIFFICULTIES:
                    v8 = results_v8.get(d)
                    v9 = results_v9.get(d)
                    if v8 is None and v9 is None:
                        continue
                    v8_str = (
                        f"num: cell={v8['cell_acc']:.3f} q={v8['query_acc']:.3f}"
                        if v8 is not None else "num: -"
                    )
                    v9_str = (
                        f"pool: cell={v9['cell_acc']:.3f} q={v9['query_acc']:.3f}"
                        if v9 is not None else "pool: -"
                    )
                    npuz = (v8['n_puzzles'] if v8 is not None else
                            (v9['n_puzzles'] if v9 is not None else 0))
                    print(
                        f"  val[{d:6s}]: {v8_str}  |  {v9_str}  n={npuz}",
                        flush=True,
                    )
                # Diagnostic: print delta_gate values (per-breath step sizes).
                try:
                    dg_main  = model.fg_v105_5_delta_gate.numpy().tolist()
                    dg_quant = model.fg_v105_5_delta_gate_quant.numpy().tolist()
                    print("  delta_gate      : "
                          + " ".join(f"{g:.3f}" for g in dg_main), flush=True)
                    print("  delta_gate_quant: "
                          + " ".join(f"{g:.3f}" for g in dg_quant), flush=True)
                except Exception as _e:
                    pass
                dual_done = True
                results = None
            elif V105_8_PER_NUMBER_READOUT:
                results = evaluate_v105_8(
                    model, val_loader, K=K,
                    max_batches=EVAL_BATCHES,
                    eval_fn=eval_fn,
                    n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
                )
            elif V105_9_AR_DIGIT_DECODER:
                results = evaluate_v105_9(
                    model, val_loader, K=K,
                    max_batches=EVAL_BATCHES,
                    eval_fn=eval_fn,
                    n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
                )
            else:
                results = evaluate_v105_5(
                    model, val_loader, K=K,
                    max_batches=EVAL_BATCHES,
                    eval_fn=eval_fn,
                    n_max=N_MAX, f_max=F_MAX, n_digits=N_DIGITS,
                )
            if not dual_done:
                for d in DIFFICULTIES:
                    if d not in results:
                        continue
                    v = results[d]
                    print(
                        f"  val[{d:6s}]: cell_acc={v['cell_acc']:.3f} "
                        f"query_acc={v['query_acc']:.3f} n={v['n_puzzles']}",
                        flush=True,
                    )
                # Diagnostic: print delta_gate values (per-breath step sizes).
                try:
                    dg_main  = model.fg_v105_5_delta_gate.numpy().tolist()
                    dg_quant = model.fg_v105_5_delta_gate_quant.numpy().tolist()
                    print(
                        "  delta_gate      : "
                        + " ".join(f"{g:.3f}" for g in dg_main),
                        flush=True,
                    )
                    print(
                        "  delta_gate_quant: "
                        + " ".join(f"{g:.3f}" for g in dg_quant),
                        flush=True,
                    )
                except Exception as _e:
                    pass

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(model_state_dict_v105_5(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    # Final save
    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_v105_5(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
