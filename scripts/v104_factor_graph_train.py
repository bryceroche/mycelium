"""v104 factor graph training driver — IB-anchored codebook initialization.

Architecture: v102-style 1024d residual codebook matching, initialized from
IB-derived semantic centroids (Pythia embeddings of variable descriptions,
clustered via hierarchical K-means per OP family).

Hypothesis: semantic codebook initialization > random codebook initialization.
Baseline (v102/v103 with random init): ~46.6% cell acc at K=10.
Target: ≥ 50% cell acc at K=10.

Dual dataset training (distribution alignment lever):
  V104_TRAIN       — primary dataset (synthetic, 50K records)
  V104_GSM8K_TRAIN — real GSM8K factor graphs filtered to [0,99] domain
                     (set to "" or unset to disable GSM8K mixing)
  V104_GSM8K_RATIO — fraction of each batch drawn from GSM8K (default 0.5)

GSM8K records are pre-processed at load time: factor graph is executed to
produce gold_values for all variables; records with values outside [0,99]
or non-integer results are discarded.

Env vars (set in launchers):
  V104_TASK=1
  V100_TASK=1
  V104_K_MAX=10
  V104_FACTOR_AUX_WEIGHT=0.5
  V104_CALIB_WEIGHT=0.05
  V104_N_MAX=16
  V104_F_MAX=8
  V104_CODEBOOK_N=32
  V104_IB_CENTROIDS=.cache/ib_centroids_gsm8k_partial.npz
  V104_TRAIN=.cache/factor_graph_train.jsonl
  V104_VAL=.cache/factor_graph_test.jsonl
  V104_GSM8K_TRAIN=.cache/gsm8k_factor_graphs_train.jsonl
  V104_GSM8K_RATIO=0.5
  BATCH=8
  STEPS=3000
  LR=3e-5
  CKPT_EVERY=500
  EVAL_EVERY=250
  CKPT_LABEL=v104_smoke
  RESUME_FROM=...
  PYTHIA_INIT=1
  V104_CURRICULUM=1
  V104_CURRICULUM_ANNEAL=1000
  V104_DIFFICULTY_FILTER=easy
"""
import gc
import json
import os
import random
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
from mycelium.factor_graph_v100 import (
    V100_K_MAX, V100_N_MAX, V100_F_MAX, V100_N_HEADS,
    attach_fg_params_v100, fg_v100_parameters, fg_v100_state_dict,
    kl_energy_diagnostic_np,
)
from mycelium.factor_graph_v104 import (
    V104_K_MAX, V104_FACTOR_AUX_WEIGHT, V104_CALIB_WEIGHT,
    V104_N_MAX, V104_F_MAX, V104_N_HEADS, V104_KL_DIAG, V104_CODEBOOK_N,
    V104_IB_CENTROIDS,
    attach_fg_params_v104, fg_v104_codebook_parameters, fg_v104_state_dict,
    fg_breathing_forward_v104,
    _compile_jit_fg_step_v104, _compile_jit_fg_eval_v104,
)
from mycelium.factor_graph_data_v100 import FactorGraphLoaderV100

DIFFICULTIES = ["easy", "medium", "hard"]

OP_MAP = {"add": 0, "sub": 1, "mul": 2, "div": 3}


# ---------------------------------------------------------------------------
# GSM8K record pre-processing
# ---------------------------------------------------------------------------

def _execute_fg(rec: dict) -> list[float] | None:
    """Execute a GSM8K factor graph to produce gold_values.

    Returns list of float values for all variables (leaves + results),
    or None if execution fails (cycle, missing value, division by zero).
    """
    ft_list  = rec.get("factor_types", [])
    fa_list  = rec.get("factor_args",  [])
    n_vars   = rec.get("n_vars", 0)
    obs_vals = list(rec.get("observed_values", []))

    vals = list(obs_vals)
    ops  = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b if b != 0 else None,
    }

    changed = True
    iters   = 0
    while changed and iters < 30:
        changed = False
        iters  += 1
        for ft, fa in zip(ft_list, fa_list):
            a1, a2, res = int(fa[0]), int(fa[1]), int(fa[2])
            if (a1 < len(vals) and a2 < len(vals) and res < len(vals)
                    and vals[a1] is not None and vals[a2] is not None
                    and vals[res] is None):
                try:
                    v = ops.get(ft, lambda a, b: None)(vals[a1], vals[a2])
                    if v is None:
                        continue
                    vals[res] = float(v)
                    changed = True
                except Exception:
                    pass
    if any(v is None for v in vals):
        return None
    return vals


def convert_gsm8k_record(rec: dict) -> dict | None:
    """Convert a GSM8K factor graph record to the synthetic format.

    Returns None if:
     - Factor graph cannot be executed to completion.
     - Any value is non-integer (e.g. 2.5).
     - Any value is outside [0, 99].

    On success, returns a dict with the same keys as synthetic records,
    including gold_values (list[int]) and difficulty='gsm8k'.
    """
    vals = _execute_fg(rec)
    if vals is None:
        return None

    # Require all values to be integer-valued
    try:
        int_vals = [int(round(v)) for v in vals]
    except Exception:
        return None
    if any(abs(int_vals[i] - vals[i]) > 0.01 for i in range(len(vals))):
        return None

    # Require all in [0, 99]
    if not all(0 <= v < 100 for v in int_vals):
        return None

    ft_list  = rec.get("factor_types", [])
    fa_list  = rec.get("factor_args",  [])
    obs_mask = rec.get("observed_mask", [])
    obs_vals = rec.get("observed_values", [])
    n_vars   = rec.get("n_vars", 0)
    n_facts  = rec.get("n_factors", 0)
    qi       = int(rec.get("query_idx", 0))

    # Build observed_values (int, not None) from gold for observed positions
    obs_vals_int = [int(round(obs_vals[i])) if obs_mask[i] == 1 else int_vals[i]
                    for i in range(len(obs_mask))]

    return {
        "gold_values":     int_vals,
        "observed_mask":   obs_mask,
        "observed_values": obs_vals_int,
        "factor_types":    ft_list,
        "factor_args":     fa_list,
        "n_factors":       n_facts,
        "query_idx":       qi,
        "difficulty":      "gsm8k",   # treated as an extra bucket; not in DIFFICULTIES for eval
    }


def load_gsm8k_records(path: str) -> list[dict]:
    """Load and filter GSM8K records to those convertible to [0,99] format."""
    if not path or not os.path.exists(path):
        return []
    records = []
    n_loaded = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            n_loaded += 1
            converted = convert_gsm8k_record(rec)
            if converted is not None:
                records.append(converted)
    print(
        f"[v104] GSM8K: loaded {n_loaded}, kept {len(records)} "
        f"({100*len(records)/max(n_loaded,1):.1f}%) after [0,99] filter",
        flush=True,
    )
    return records


# ---------------------------------------------------------------------------
# Dual data loader
# ---------------------------------------------------------------------------

class DualDataLoader:
    """Samples 50/50 (or configurable ratio) from synthetic + GSM8K records.

    Falls back to synthetic-only if no GSM8K records are available.
    """

    def __init__(
        self,
        synth_loader: FactorGraphLoaderV100,
        gsm8k_records: list[dict],
        gsm8k_ratio: float = 0.5,
        n_max: int = V104_N_MAX,
        f_max: int = V104_F_MAX,
        k_max: int = V104_K_MAX,
        n_heads: int = V104_N_HEADS,
        seed: int = 42,
    ):
        self.synth_loader   = synth_loader
        self.gsm8k_records  = gsm8k_records
        self.gsm8k_ratio    = gsm8k_ratio if gsm8k_records else 0.0
        self.n_max          = n_max
        self.f_max          = f_max
        self.k_max          = k_max
        self.n_heads        = n_heads
        self.rng            = random.Random(seed)
        print(
            f"[v104] DualDataLoader: synth={len(synth_loader)} gsm8k={len(gsm8k_records)} "
            f"ratio={self.gsm8k_ratio:.2f}",
            flush=True,
        )

    def sample_batch(self, step: int = 0) -> dict:
        """Build a batch mixing synthetic and GSM8K records at gsm8k_ratio."""
        from mycelium.factor_graph_data_v100 import _records_to_batch_v100, batch_to_tensors_v100

        synth_batch = self.synth_loader.sample_batch(step=step)
        synth_picks = synth_batch["picks"]
        B           = len(synth_picks)

        if self.gsm8k_ratio <= 0 or not self.gsm8k_records:
            return synth_batch

        # Decide how many slots go to GSM8K (Binomial sampling)
        n_gsm8k = sum(
            1 for _ in range(B) if self.rng.random() < self.gsm8k_ratio
        )
        n_synth = B - n_gsm8k

        mixed_picks: list[dict] = []
        if n_synth > 0:
            mixed_picks += self.rng.choices(synth_picks, k=n_synth)
        if n_gsm8k > 0:
            mixed_picks += self.rng.choices(self.gsm8k_records, k=n_gsm8k)
        self.rng.shuffle(mixed_picks)

        batch_np = _records_to_batch_v100(
            mixed_picks, self.n_max, self.f_max, self.k_max, self.n_heads,
        )
        batch_t = batch_to_tensors_v100(batch_np)
        batch_t["picks"] = mixed_picks
        return batch_t


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


def collect_fg_params_v104(model) -> list[Tensor]:
    """Trainable parameters: shared L0-L3 attn/FFN + v100 fg params + v104 codebook."""
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += fg_v100_parameters(model)
    params += fg_v104_codebook_parameters(model)
    return params


def model_state_dict_v104(model) -> dict:
    """State dict for v104 factor-graph training (v100 keys + v104 codebook keys)."""
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
    sd.update(fg_v100_state_dict(model))
    sd.update(fg_v104_state_dict(model))
    return sd


def load_ckpt_v104(model, path: str):
    """Load a v100 or v104 checkpoint.

    v100 keys loaded exactly; v104 codebook keys loaded if present;
    if missing (e.g. v100 warm-start), the IB-anchored init is kept.
    """
    sd = safe_load(path)
    targets = model_state_dict_v104(model)
    missing = []
    loaded  = []
    for name, dst in targets.items():
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            if (name in ("fg_v100.breath_embed", "fg_v100.delta_gate")
                    and src.ndim == dst.ndim
                    and src.shape[0] <= dst.shape[0]):
                k_old = int(src.shape[0])
                if src.dtype != dst.dtype:
                    src = src.cast(dst.dtype)
                cur = dst.numpy()
                src_np = src.numpy()
                cur[:k_old] = src_np[:k_old]
                from tinygrad import Tensor as _T
                dst.assign(_T(cur, dtype=dst.dtype, device=dst.device).contiguous()).realize()
                loaded.append(name)
                continue
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name}(shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
        loaded.append(name)

    v104_keys    = [k for k in targets if k.startswith("fg_v104.")]
    loaded_v104  = [k for k in loaded  if k.startswith("fg_v104.")]
    if not loaded_v104:
        print("  no v104 codebook keys in ckpt — IB-anchored init preserved (warm-start from v100)")
    else:
        print(f"  loaded {len(loaded_v104)}/{len(v104_keys)} v104 codebook keys")
    non_v104_missing = [k for k in missing if not k.startswith("fg_v104.")]
    if non_v104_missing:
        print(f"  ckpt missing {len(non_v104_missing)} non-v104 keys: {non_v104_missing[:5]}")


def evaluate_v104(model, loader: FactorGraphLoaderV100, K: int,
                  max_batches: int = 20, eval_fn=None,
                  n_max: int = V104_N_MAX, f_max: int = V104_F_MAX) -> dict:
    """Run eval on up to max_batches batches. Returns per-difficulty stats."""
    Tensor.training = False
    agg = {}
    n_batches = 0

    for batch in loader.iter_eval(batch_size=loader.batch_size):
        domain_init   = batch["domain_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        gold_values   = batch["gold_values"]
        obs_mask      = batch["observed_mask"]
        query_idx_np  = batch["query_idx"]
        picks         = batch["picks"]

        if eval_fn is not None:
            pred_t, _cell_acc_t = eval_fn(
                domain_init, node_kinds, staging_mask, head_op_mask,
                gold_values, obs_mask,
            )
            pred_np = pred_t.numpy()
        else:
            var_logits_history, _, _ = fg_breathing_forward_v104(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=n_max, f_max=f_max,
            )
            pred_np = var_logits_history[-1].argmax(axis=-1).realize().numpy()

        gold_np = gold_values.numpy()
        obs_np  = obs_mask.numpy()
        B_local = len(picks)

        for b in range(B_local):
            rec  = picks[b]
            diff = rec.get("difficulty", "easy")
            if diff not in agg:
                agg[diff] = {"n_unobs": 0, "n_correct_unobs": 0,
                             "query_correct": 0, "n_puzzles": 0}
            qi = int(query_idx_np[b])
            nv = int(batch["n_vars_total"][b])
            for vi in range(min(nv, n_max)):
                if obs_np[b, vi] == 0:
                    agg[diff]["n_unobs"] += 1
                    if pred_np[b, vi] == gold_np[b, vi]:
                        agg[diff]["n_correct_unobs"] += 1
            if qi < n_max and pred_np[b, qi] == gold_np[b, qi]:
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
        out[d]    = {"cell_acc": cell_acc, "query_acc": q_acc, "n_puzzles": n}

    Tensor.training = True
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    V104_TASK_LOCAL = int(getenv("V104_TASK", 0)) > 0
    assert V104_TASK_LOCAL, "V104_TASK=1 must be set"

    K      = int(getenv("V104_K_MAX",    str(V104_K_MAX)))
    BATCH  = int(getenv("BATCH",         "8"))
    STEPS  = int(getenv("STEPS",         "3000"))
    LR     = float(getenv("LR",          "3e-5"))
    CKPT_EVERY        = int(getenv("CKPT_EVERY",        "500"))
    EVAL_EVERY        = int(getenv("EVAL_EVERY",        "250"))
    LOG_EVERY         = int(getenv("LOG_EVERY",         "10"))
    PER_BREATH_EVERY  = int(getenv("PER_BREATH_CE_EVERY", "50"))
    GC_EVERY          = int(getenv("GC_EVERY",          "50"))
    CKPT_LABEL        = getenv("CKPT_LABEL",            "v104_smoke")
    RESUME_FROM       = getenv("RESUME_FROM",           "")
    PYTHIA_INIT       = int(getenv("PYTHIA_INIT",       "1")) > 0
    SEED              = int(getenv("SEED",              "42"))
    N_CODE            = int(getenv("V104_CODEBOOK_N",   str(V104_CODEBOOK_N)))
    IB_PATH           = getenv("V104_IB_CENTROIDS",     V104_IB_CENTROIDS)

    N_MAX  = int(getenv("V104_N_MAX",  str(V104_N_MAX)))
    F_MAX  = int(getenv("V104_F_MAX",  str(V104_F_MAX)))
    T_MAX  = N_MAX + F_MAX

    DIFFICULTY_FILTER = (
        os.environ.get("V104_DIFFICULTY_FILTER", "").strip()
        or os.environ.get("V100_DIFFICULTY_FILTER", "").strip()
        or None
    )
    CURRICULUM        = int(getenv("V104_CURRICULUM",       "0")) > 0
    CURRICULUM_ANNEAL = int(getenv("V104_CURRICULUM_ANNEAL", "1000"))

    TRAIN_PATH    = getenv("V104_TRAIN",       getenv("V100_TRAIN", ".cache/factor_graph_train.jsonl"))
    VAL_PATH      = getenv("V104_VAL",         getenv("V100_VAL",   ".cache/factor_graph_test.jsonl"))
    GSM8K_PATH    = getenv("V104_GSM8K_TRAIN", ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO   = float(getenv("V104_GSM8K_RATIO", "0.5"))
    EVAL_BATCHES  = int(getenv("EVAL_BATCHES", "20"))
    EVAL_BATCH    = int(getenv("EVAL_BATCH",   str(BATCH)))

    FACTOR_AUX_WEIGHT = float(getenv("V104_FACTOR_AUX_WEIGHT", str(V104_FACTOR_AUX_WEIGHT)))
    CALIB_WEIGHT      = float(getenv("V104_CALIB_WEIGHT",       str(V104_CALIB_WEIGHT)))

    KL_DIAG_EVERY   = int(getenv("KL_DIAG_EVERY", "100"))
    KL_DIAG_ENABLED = V104_KL_DIAG or (int(getenv("V104_KL_DIAG", "0")) > 0)

    print("=== v104 factor graph training (IB-anchored semantic codebook) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  steps={STEPS}  lr={LR}")
    print(f"N_MAX={N_MAX}  F_MAX={F_MAX}  T_MAX={T_MAX}  codebook_n={N_CODE}")
    print(f"ib_centroids={IB_PATH}")
    print(f"factor_aux_weight={FACTOR_AUX_WEIGHT}  calib_weight={CALIB_WEIGHT}")
    print(f"difficulty_filter={DIFFICULTY_FILTER}  curriculum={CURRICULUM}")
    print(f"train={TRAIN_PATH}  val={VAL_PATH}")
    print(f"gsm8k_train={GSM8K_PATH}  gsm8k_ratio={GSM8K_RATIO}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer (PYTHIA_INIT={PYTHIA_INIT})...")
    if PYTHIA_INIT:
        from mycelium.loader import _load_state
        sd = _load_state()
        model = load_breathing(cfg, sd=sd)
        del sd
    else:
        model = BreathingTransformer(cfg)
    cast_layers_fp32(model)

    # Attach v100 params first, then v104 IB-anchored codebook
    attach_fg_params_v100(model, hidden=cfg.hidden, n_heads=cfg.n_heads,
                          k_max=K, n_max=N_MAX, f_max=F_MAX)
    attach_fg_params_v104(model, hidden=cfg.hidden, n_code=N_CODE,
                          ib_centroids_path=IB_PATH)
    Device[Device.DEFAULT].synchronize()

    params   = collect_fg_params_v104(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")
    cb_params = fg_v104_codebook_parameters(model)
    n_cb = sum(int(np.prod(t.shape)) for t in cb_params)
    K_max = int(model.fg_v100_breath_embed.shape[0])
    print(f"  codebook params: {n_cb/1e6:.4f}M  "
          f"(codebook={N_CODE}×{cfg.hidden}, delta_gate_quant={K_max}, temperature=1)")

    if RESUME_FROM:
        print(f"resuming from ckpt: {RESUME_FROM}")
        load_ckpt_v104(model, RESUME_FROM)
        print("  loaded.")

    opt = AdamW(params, lr=LR, weight_decay=0.0)

    # Primary synthetic loader
    synth_loader = FactorGraphLoaderV100(
        TRAIN_PATH, batch_size=BATCH,
        difficulty_filter=DIFFICULTY_FILTER,
        curriculum=CURRICULUM,
        curriculum_anneal_steps=CURRICULUM_ANNEAL,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V104_N_HEADS,
        seed=SEED,
    )

    # GSM8K records (filtered to [0,99])
    gsm8k_records = load_gsm8k_records(GSM8K_PATH)

    # Dual loader (50/50 by default)
    dual_loader = DualDataLoader(
        synth_loader, gsm8k_records,
        gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V104_N_HEADS,
        seed=SEED + 1,
    )

    # Eval loader (synthetic test set only — clean comparison with prior baselines)
    val_loader = FactorGraphLoaderV100(
        VAL_PATH, batch_size=EVAL_BATCH,
        difficulty_filter=None,
        curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=V104_N_HEADS,
        seed=SEED + 2,
    )

    ckpt_dir = ".cache/fg_v104_ckpts"
    os.makedirs(ckpt_dir, exist_ok=True)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v104(
        model, opt, K=K, B=BATCH,
        factor_aux_weight=FACTOR_AUX_WEIGHT,
        calib_weight=CALIB_WEIGHT,
        n_max=N_MAX, f_max=F_MAX,
        grad_clip=1.0,
    )
    eval_fn = _compile_jit_fg_eval_v104(
        model, K=K, B=EVAL_BATCH, n_max=N_MAX, f_max=F_MAX,
    )
    Tensor.training = True

    print(f"\ntraining...\n")
    t0 = time.time()
    log_loss = log_ce = log_calib = log_aux = log_n = 0.0

    for step in range(1, STEPS + 1):
        batch = dual_loader.sample_batch(step=step)

        domain_init   = batch["domain_init"]
        node_kinds    = batch["node_kinds"]
        staging_mask  = batch["staging_mask"]
        head_op_mask  = batch["head_op_mask"]
        gold_values   = batch["gold_values"]
        obs_mask      = batch["observed_mask"]
        ft_np         = batch["factor_types"].numpy()
        fa_np         = batch["factor_args"].numpy()
        gold_np       = batch["gold_values"].numpy()

        # Pre-compute factor_gold and factor_valid in numpy (no .numpy() inside JIT)
        factor_gold_np  = np.zeros((BATCH, F_MAX), dtype=np.int32)
        factor_valid_np = np.zeros((BATCH, F_MAX), dtype=np.float32)
        for b in range(BATCH):
            for fi in range(F_MAX):
                op = int(ft_np[b, fi])
                if op < 0:
                    continue
                r_idx = int(fa_np[b, fi, 2])
                if r_idx < 0 or r_idx >= N_MAX:
                    continue
                factor_gold_np[b, fi]  = int(gold_np[b, r_idx])
                factor_valid_np[b, fi] = 1.0
        factor_gold_t  = Tensor(factor_gold_np,  dtype=dtypes.int).contiguous().realize()
        factor_valid_t = Tensor(factor_valid_np, dtype=dtypes.float).contiguous().realize()

        outs = step_fn(
            domain_init, node_kinds, staging_mask, head_op_mask,
            gold_values, obs_mask, factor_gold_t, factor_valid_t,
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
                if ladder_delta > 0.1:
                    print(f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} > 0.1", flush=True)
                else:
                    print(f"  [LADDER] B0-B{K-1} delta = {ladder_delta:.3f} (target > 0.1)",
                          flush=True)

        if KL_DIAG_ENABLED and step % KL_DIAG_EVERY == 0:
            Tensor.training = False
            var_lh_diag, _, _ = fg_breathing_forward_v104(
                model, domain_init, node_kinds, staging_mask, head_op_mask,
                K=K, n_max=N_MAX, f_max=F_MAX,
            )
            final_logits_np = var_lh_diag[-1].realize().numpy()
            kl_energy = kl_energy_diagnostic_np(
                final_logits_np, ft_np, fa_np, n_max=N_MAX, f_max=F_MAX,
            )
            print(f"  [KL_energy] step {step}: mean_kl_per_factor = {kl_energy:.4f}", flush=True)
            Tensor.training = True

        if step % EVAL_EVERY == 0:
            print(f"  evaluating ({EVAL_BATCHES} batches × B={EVAL_BATCH})...", flush=True)
            results = evaluate_v104(
                model, val_loader, K=K,
                max_batches=EVAL_BATCHES,
                eval_fn=eval_fn,
                n_max=N_MAX, f_max=F_MAX,
            )
            for d in DIFFICULTIES:
                if d not in results:
                    continue
                v = results[d]
                print(
                    f"  val[{d:6s}]: cell_acc={v['cell_acc']:.3f} "
                    f"query_acc={v['query_acc']:.3f} n={v['n_puzzles']}",
                    flush=True,
                )

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(model_state_dict_v104(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    # Final save
    ckpt_path = os.path.join(ckpt_dir, f"{CKPT_LABEL}_final.safetensors")
    safe_save(model_state_dict_v104(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
