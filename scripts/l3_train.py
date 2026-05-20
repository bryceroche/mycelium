"""Math curriculum training (L3 / L4 / L4.5) for the breathing transformer.

Standard mixed-loops training (random {1,2,4}), masked-loss CE on the answer
span only. Periodic accuracy evaluation at fixed loop counts {1,2,4,8} on a
held-out set.

Select the curriculum level with the LEVEL env var (default L3). FIXED_LEN
defaults to a level-appropriate value (L3=64, L4=96, L4.5=160). Checkpoints
land in .cache/{level_lower}_ckpts/.

Perf knobs (env vars worth setting at launch):
  BEAM=2            tinygrad kernel autotuner — slower first compile, faster
                    steady-state. Effect biggest on long runs.
  CTRL_TRAIN_EVERY  controller train every K main steps (default 4).
  CTRL_MAX_LOOPS    breaths in controller-train forward (default 2).
  EVAL_LOOPS        list of n_loops for accuracy eval. Fewer = faster evals.
  PROFILE=1         per-phase timing breakdown (encode/forward/backward+step)
                    averaged every PROFILE_EVERY steps. Forces sync at phase
                    boundaries so timings are accurate but slightly slower.
"""
import sys
import os
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes, GlobalCounters
from tinygrad.helpers import getenv
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, safe_load


def mem_log(label: str, force: bool = False):
    """Print current per-device VRAM via tinygrad's GlobalCounters. Cache info
    (allocator's buffer pool) is included separately — cached buffers still
    hold GPU memory but aren't in mem_used. Gated on MEM_LOG env var unless
    force=True."""
    if not (force or bool(getenv("MEM_LOG", 0))):
        return
    per_dev = dict(GlobalCounters.mem_used_per_device)
    parts = [f"{d}={v/1e9:.2f}GB" for d, v in per_dev.items()]
    cache_info = ""
    try:
        alloc = Device[Device.DEFAULT].allocator
        if hasattr(alloc, "cache"):
            cache_size = len(alloc.cache)
            cached_bytes = sum(opt.nbytes for opt in alloc.cache.keys() if hasattr(opt, 'nbytes')) if cache_size else 0
            cache_info = f"  cache_entries={cache_size}"
            if cached_bytes:
                cache_info += f"  cached={cached_bytes/1e9:.2f}GB"
    except Exception:
        pass
    print(f"[MEM] {label}: " + " ".join(parts) + cache_info, flush=True)

from mycelium import Config
from mycelium.loader import _load_state, load_breathing
from mycelium.data import load_tokenizer
from mycelium.l3_data import generate_math, split_train_eval
from mycelium.l3_training import (
    multi_cycle_train_step, multi_cycle_eval_loss, accuracy_at_loops_multi,
    controller_train_step, calibration_train_step,
)
from mycelium.l3_training import _JIT_TRAIN_CACHE
from mycelium.calibration import digit_token_ids_for
from mycelium.lookup_eval import lookup_eval


def cast_model_fp32(model):
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    _cast(model.embed, "weight")
    _cast(model, "embed_out")
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
        _cast(sw, a)
    for layer in model.block.layers:
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            _cast(layer, a)


def collect_params(model):
    nps = [model.embed.weight, model.embed_out, model.ln_f_g, model.ln_f_b]
    sw = model.block.shared
    nps += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
            sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        nps += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    # v44 doubled-layers Set B — only included in optimizer when actually in use
    # (would have undefined gradient otherwise since not touched in forward).
    from mycelium.breathing import DOUBLED_LAYERS as _DL_FLAG
    if _DL_FLAG:
        for layer in model.block.layers_b:
            nps += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    # Breath-time embedding (axial conditioning) — always included for ckpt symmetry.
    # When BREATH_TIME_EMBED=0 the L2 reg keeps the gradient defined; the param doesn't move.
    nps += [model.block.breath_embed]
    # Cross-breath handoff projection — always included for ckpt symmetry.
    # When CROSS_BREATH_HANDOFF=0 the L2 reg keeps gradient defined; weights don't move.
    nps += [model.block.handoff_w, model.block.handoff_b]
    # Helix pitch (single scalar) — learned when LEARN_PITCH=1.
    nps += [model.block.rope.pitch]
    # Constant-radius projection scalars — learned when CONSTANT_RADIUS=1.
    nps += [model.block.crp_mix_alpha, model.block.crp_target_norm]
    # v24 notebook projections — zero-init, learned when NOTEBOOK_V24=1.
    # Always included for ckpt symmetry; gradient is zero when notebook is off.
    nps += [model.block.notebook_write_w, model.block.notebook_write_b,
            model.block.notebook_read_w, model.block.notebook_read_b,
            model.block.notebook_write_query]
    # v24c dual notebook (REPLACE-semantics partner). Always included for ckpt
    # symmetry; gradient inert when NOTEBOOK_DUAL=0.
    nps += [model.block.notebook_rep_write_w, model.block.notebook_rep_write_b,
            model.block.notebook_rep_read_w, model.block.notebook_rep_read_b,
            model.block.notebook_rep_query]
    # v28 lookup table values — learned prototype reps per entry. Keys (lookup_table.weight)
    # stay frozen at random-orthogonal init (preserves op classification stability).
    # Values trained end-to-end via main CE loss when LOOKUP_VALUE_INJECT=1.
    # value_proj_up: identity when LOOKUP_IB_DIM=0, learnable (IB_DIM, hidden) when >0.
    nps += [model.lookup_table.values, model.lookup_table.value_proj_up]
    # v38 B-field IB bottleneck — learned when BFIELD_WAIST > 0. Always included
    # for ckpt symmetry; L2 reg in l3_training keeps gradient defined when off.
    nps += [model.block.bfield_proj_down, model.block.bfield_proj_up, model.block.bfield_bias]
    # v50 IB-waist codebook — learnable keys + values for op-discriminative compression.
    # Always included for ckpt symmetry; gradient inert when WAIST_CODEBOOK_N=0
    # (values are zero-init → contribution is zero → gradient through retrieved is zero).
    nps += [model.block.waist_codebook_keys, model.block.waist_codebook_values]
    # v39 waist head (512 → 4 op classifier) — trained when BFIELD_AUX_WEIGHT > 0
    # and BFIELD_END_OF_BREATH=1. L2 reg covers the off case.
    nps += [model.waist_head_w, model.waist_head_b]
    # Calibration head — trained on the main loss via REINFORCE in calibration_train_step.
    # Always included so opt.step() has a defined gradient for these params even when
    # CALIBRATION_MODE=0 (gradient is zero in that path, weights don't move).
    nps += model.confidence_head.parameters()
    # v54 WaistController — cross-attention text decoder over (waist, prompt).
    # Only added to params (and therefore optimizer-trained) when CONTROLLER_DECODE=1,
    # because the other train paths don't touch it and AdamW would assert on missing grads.
    from mycelium.breathing import CONTROLLER_DECODE as _CD
    if _CD:
        nps += model.waist_controller.parameters()
    return nps


def load_checkpoint(model, path: str):
    """Load a safetensors checkpoint via the model's state_dict interface.
    Tolerates missing keys (e.g., resuming an old ckpt that predates lookup_table).
    """
    sd = safe_load(path)
    info = model.load_state_dict(sd, strict=False)
    if info["missing"]:
        print(f"  (ckpt missing {len(info['missing'])} keys, kept default init: {info['missing'][:3]}...)")
    if info["unexpected"]:
        print(f"  (ignoring {len(info['unexpected'])} extra ckpt keys: {info['unexpected'][:3]}...)")


def named_state(model):
    """Backwards-compat shim — model.state_dict() is the source of truth now."""
    return model.state_dict()


DEFAULT_FIXED_LEN = {"ARITH": 32, "ARITH_HARD": 32, "ARITH_MIXED": 32, "ARITH_BORROW": 32, "L3": 64, "L4": 96, "L4_BORROW": 96, "L4_MIXED": 96, "L4.5": 160, "L4.7": 200, "GSM8K_SPACED": 320, "GSM8K_STEPS": 320}


def main():
    # Config overrides via env vars — enables scaling to Pythia-1B et al.
    # Defaults match Pythia-410M (current baseline).
    cfg_kwargs = {}
    if (n_lookup_entries := getenv("N_LOOKUP_ENTRIES", 0)):
        if int(n_lookup_entries) != 16:
            cfg_kwargs["n_lookup_entries"] = int(n_lookup_entries)
    for env_key, cfg_key in [("HIDDEN", "hidden"), ("N_HEADS", "n_heads"),
                              ("HEAD_DIM", "head_dim"), ("FFN", "ffn"),
                              ("CONTROLLER_HIDDEN", "controller_hidden")]:
        v = os.environ.get(env_key)
        if v is not None:
            cfg_kwargs[cfg_key] = int(v)
    cfg = Config(**cfg_kwargs)
    if cfg_kwargs:
        print(f"[Config overrides] {cfg_kwargs}")
    LEVEL = getenv("LEVEL", "L3")
    BATCH = getenv("BATCH", 16)
    FIXED_LEN = getenv("FIXED_LEN", DEFAULT_FIXED_LEN.get(LEVEL, 96))
    STEPS = getenv("STEPS", 500)
    LR = float(getenv("LR", "3e-5"))
    LOSS_EVAL_EVERY = getenv("LOSS_EVAL_EVERY", 100)
    ACC_EVAL_EVERY = getenv("ACC_EVAL_EVERY", 250)
    CKPT_EVERY = getenv("CKPT_EVERY", 250)
    NUM_PROBLEMS = getenv("NUM_PROBLEMS", 20000)
    NUM_EVAL = getenv("NUM_EVAL", 100)
    NUM_VAL_BATCHES = getenv("NUM_VAL_BATCHES", 4)
    TRAIN_LOOPS = [int(x) for x in getenv("TRAIN_LOOPS", "1,2,4").split(",")]   # Phase A choices
    EVAL_LOOPS = [int(x) for x in getenv("EVAL_LOOPS", "1,2,4,8").split(",")]   # Phase A test points
    PHASE_C_LOOPS = getenv("PHASE_C_LOOPS", 1)                                  # light breathing for execution cycles
    SEED = getenv("SEED", 42)
    RESUME_FROM = getenv("RESUME_FROM", "")
    SPACE_DIGITS = bool(getenv("SPACE_DIGITS", 0))   # digit-by-digit tokenization for arithmetic
    EVAL_BATCH = getenv("EVAL_BATCH", 64)            # batched accuracy eval (kept fixed → JIT reuse)
    # K/V cache length for eval. Default = FIXED_LEN + 40 (room for max_prompt up to
    # FIXED_LEN plus max_new=40 generated tokens at eval). Always fits and stays much
    # smaller than cfg.max_seq_len=512 (so ~4-7× memory savings on K/V buffers vs the
    # default cap). Override explicitly only when you know prompts + max_new are smaller.
    EVAL_CACHE_LEN = getenv("EVAL_CACHE_LEN", 0) or (FIXED_LEN + 40)
    LOOKUP_EVAL = getenv("LOOKUP_EVAL", 1)           # 1 = run per-checkpoint lookup-table classification eval
    LOOKUP_EVAL_LOOPS = getenv("LOOKUP_EVAL_LOOPS", 8)  # n_loops for the lookup eval (single value)
    # Joint training of the model's internal LookupTable via aux op-classification CE.
    # 0 = off (table stays at random orthogonal init), 0.1 = light supervision, 1.0 = strong.
    LOOKUP_AUX_WEIGHT = float(getenv("LOOKUP_AUX_WEIGHT", "0.1"))
    # Controller training (Step F). 0 = off (controller stays at random init).
    # When > 0, runs a separate controller_train_step alongside the main step.
    # CTRL_LR = LR for the controller's separate Adam optimizer (transformer-isolated).
    CTRL_TRAIN = bool(getenv("CTRL_TRAIN", 0))
    CTRL_LR = float(getenv("CTRL_LR", "1e-4"))
    CTRL_MAX_LOOPS = getenv("CTRL_MAX_LOOPS", 2)     # max breaths used for controller training (was 4 — halved for speed)
    CTRL_TRAIN_EVERY = getenv("CTRL_TRAIN_EVERY", 4) # update controller every K main steps (1=every step, 4=every 4)
    COMPUTE_PENALTY = float(getenv("COMPUTE_PENALTY", "0.0"))  # reward stop_logit > 0 in ctrl loss (0.0 = off)
    STOP_CALIB_WEIGHT = float(getenv("STOP_CALIB_WEIGHT", "0.01"))  # weight on per-example stop calibration (existing default 0.01)
    # Per-step optimal-stopping calibration training. When CALIBRATION_MODE=1, the main
    # train step switches from multi_cycle_train_step to calibration_train_step:
    # single-cycle encoding, per-step answer-CE at EVERY breath, plus BCE(confidence,
    # argmax_correct) to train calibration. Different steps can be claimed at different
    # breaths via the learned confidence_head at inference.
    CALIBRATION_MODE = bool(getenv("CALIBRATION_MODE", 0))
    CALIBRATION_WEIGHT = float(getenv("CALIBRATION_WEIGHT", "0.1"))  # λ on the BCE term
    CALIBRATION_LOOPS = getenv("CALIBRATION_LOOPS", 8)  # n_loops for the calibration forward pass
    PROFILE = bool(getenv("PROFILE", 0))             # 1 = print per-phase timing summary every PROFILE_EVERY steps
    PROFILE_EVERY = getenv("PROFILE_EVERY", 50)
    GC_EVERY = getenv("GC_EVERY", 50)                # gc.collect() every K steps to flush Python refs holding lazy buffers
    USE_JIT = bool(getenv("USE_JIT", 0))             # 1 = JIT the whole train step (forward+backward+opt.step)

    print(f"=== Math training — level {LEVEL} (three-phase: heavy A, light C) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  seq_len={FIXED_LEN}  steps={STEPS}  lr={LR}")
    print(f"corpus={NUM_PROBLEMS}, eval set={NUM_EVAL}, space_digits={SPACE_DIGITS}")
    print(f"phase_A_train_loops={TRAIN_LOOPS}  phase_A_eval_loops={EVAL_LOOPS}  phase_C_loops={PHASE_C_LOOPS}")
    print(f"eval batch={EVAL_BATCH}  cache_len={EVAL_CACHE_LEN}  lookup_eval={'on' if LOOKUP_EVAL else 'off'}@A={LOOKUP_EVAL_LOOPS}")
    print(f"lookup_aux_weight={LOOKUP_AUX_WEIGHT}  ctrl_train={'on' if CTRL_TRAIN else 'off'}  ctrl_lr={CTRL_LR}  ctrl_max_loops={CTRL_MAX_LOOPS}  ctrl_train_every={CTRL_TRAIN_EVERY}")
    print(f"weight_decay={float(getenv('WEIGHT_DECAY', '0.05'))}  label_smoothing={float(getenv('LABEL_SMOOTHING', '0.0'))}  stoch_depth_p={float(getenv('STOCH_DEPTH_P', '0.0'))}")
    print(f"use_jit={USE_JIT}  gc_every={GC_EVERY}")
    print()

    # v44 MIXED_LEVELS: comma-separated list of curriculum levels. When set,
    # each training step samples a random level and uses its corpus + fixed_len.
    # Eval still runs against the primary LEVEL. JIT cache compiles per (level,
    # n_loops, fixed_len) combination; reuses across the run.
    MIXED_LEVELS = getenv("MIXED_LEVELS", "")
    mixed_pool = []  # list of (level_name, train_examples, fixed_len) tuples
    if MIXED_LEVELS:
        levels_list = [s.strip() for s in MIXED_LEVELS.split(",") if s.strip()]
        print(f"=== MIXED-level training over {levels_list} ===")
        t0 = time.perf_counter()
        for lev in levels_list:
            ex_all = generate_math(lev, NUM_PROBLEMS, seed=SEED, digit_spacing=SPACE_DIGITS)
            ex_train, ex_eval = split_train_eval(ex_all, n_eval=NUM_EVAL, seed=SEED)
            fl_lev = DEFAULT_FIXED_LEN.get(lev, 96)
            mixed_pool.append((lev, ex_train, fl_lev))
            print(f"  {lev}: train={len(ex_train)} eval={len(ex_eval)} fixed_len={fl_lev}")
        # Primary level's eval set for accuracy tracking (LEVEL chosen separately).
        # Special-case GSM8K_SPACED: use GSM8K test set as the real benchmark eval,
        # not a held-out from train (train is already in the mixed pool).
        if LEVEL == "GSM8K_SPACED":
            from mycelium.l3_data import load_gsm8k_spaced
            train_examples = mixed_pool[0][1]   # placeholder; the mixed pool is used for training
            eval_examples = load_gsm8k_spaced("test")[:NUM_EVAL]
            print(f"  primary eval level {LEVEL} (GSM8K test): eval={len(eval_examples)}  ({time.perf_counter()-t0:.1f}s)")
        else:
            all_examples = generate_math(LEVEL, NUM_PROBLEMS, seed=SEED, digit_spacing=SPACE_DIGITS)
            train_examples, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=SEED)
            print(f"  primary eval level {LEVEL}: eval={len(eval_examples)}  ({time.perf_counter()-t0:.1f}s)")
    elif LEVEL == "GSM8K_SPACED":
        # GSM8K curriculum extension: load 7.5k train + 1.3k test from the cached
        # parquet. Test set is the eval set (the real benchmark). Always digit-
        # spaced (the loader applies space_digits internally — SPACE_DIGITS env
        # var is ignored for this level).
        from mycelium.l3_data import load_gsm8k_spaced
        print(f"loading GSM8K-spaced (train + test)...")
        t0 = time.perf_counter()
        train_examples = load_gsm8k_spaced("train")
        eval_examples_all = load_gsm8k_spaced("test")
        # Cap eval at NUM_EVAL for speed during training (full test set is run
        # via scripts/eval_ckpt_on_gsm8k.py against the final ckpt).
        eval_examples = eval_examples_all[:NUM_EVAL]
        print(f"  train={len(train_examples)}  eval={len(eval_examples)} (cap {NUM_EVAL} of {len(eval_examples_all)})  ({time.perf_counter()-t0:.1f}s)")
        ex0 = train_examples[0]
        print(f"  sample: {ex0.problem!r}")
        print(f"  target: {ex0.gen_targets[0][:150]!r}...")
    elif LEVEL == "GSM8K_STEPS":
        # GSM8K with Haiku-distilled per-step gen_targets (generated by
        # scripts/generate_gsm8k_step_targets.py). Required by per_breath_train_step
        # which asserts uniform K across a batch — so we bucket by K and each batch
        # is drawn from a single K bucket. JIT compiles a separate kernel per K
        # (the JIT cache key includes K).
        from mycelium.l3_data import load_gsm8k_steps
        GSM8K_STEPS_PATH = getenv("GSM8K_STEPS_PATH", ".cache/gsm8k_steps_v1_train.jsonl")
        GSM8K_STEPS_MIN_K = int(getenv("GSM8K_STEPS_MIN_K", 2))
        GSM8K_STEPS_MAX_K = int(getenv("GSM8K_STEPS_MAX_K", 6))  # K=7+ pushes FIXED_LEN; cap by default
        print(f"loading GSM8K-steps from {GSM8K_STEPS_PATH} (min_k={GSM8K_STEPS_MIN_K}, max_k={GSM8K_STEPS_MAX_K})...")
        t0 = time.perf_counter()
        gsm8k_step_buckets = load_gsm8k_steps(GSM8K_STEPS_PATH,
                                                min_k=GSM8K_STEPS_MIN_K, max_k=GSM8K_STEPS_MAX_K,
                                                bucket_by_k=True)
        # Hold out a small slice per bucket as eval — proportional to bucket size, capped at NUM_EVAL total.
        # The full GSM8K test set lives under GSM8K_SPACED; this slice is just for in-training acc tracking.
        eval_examples = []
        train_buckets = {}
        total_keep = sum(len(v) for v in gsm8k_step_buckets.values())
        eval_per_bucket = max(1, NUM_EVAL // max(1, len(gsm8k_step_buckets)))
        for k in sorted(gsm8k_step_buckets):
            bucket = gsm8k_step_buckets[k]
            n_eval_k = min(eval_per_bucket, max(1, len(bucket) // 10))
            eval_examples.extend(bucket[:n_eval_k])
            train_buckets[k] = bucket[n_eval_k:]
        train_examples = [ex for k in sorted(train_buckets) for ex in train_buckets[k]]  # flat fallback
        print(f"  loaded {total_keep} examples across K={sorted(gsm8k_step_buckets)} buckets:")
        for k in sorted(train_buckets):
            print(f"    K={k}: train={len(train_buckets[k])} eval={len([e for e in eval_examples if len(e.gen_targets) == k])}")
        print(f"  total: train={sum(len(v) for v in train_buckets.values())} eval={len(eval_examples)} ({time.perf_counter()-t0:.1f}s)")
        ex0 = train_examples[0] if train_examples else None
        if ex0:
            print(f"  sample (K={len(ex0.gen_targets)}): {ex0.problem[:100]}")
            for i, g in enumerate(ex0.gen_targets):
                print(f"    step {i+1}: {g[:120]}")
    else:
        print(f"generating {LEVEL} problems...")
        t0 = time.perf_counter()
        all_examples = generate_math(LEVEL, NUM_PROBLEMS, seed=SEED, digit_spacing=SPACE_DIGITS)
        train_examples, eval_examples = split_train_eval(all_examples, n_eval=NUM_EVAL, seed=SEED)
        print(f"  train={len(train_examples)}  eval={len(eval_examples)}  ({time.perf_counter()-t0:.1f}s)")
        if SPACE_DIGITS:
            ex0 = train_examples[0]
            print(f"  sample (digit-spaced): {ex0.problem!r} -> {ex0.gen!r}")

    tok = load_tokenizer()
    from mycelium.lookup_table import eq_token_ids_for
    eq_token_ids = eq_token_ids_for(tok)  # both " =" and "=" — BPE may produce either

    print("\nloading Pythia-410M -> breathing transformer...")
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    cast_model_fp32(model)
    Device[Device.DEFAULT].synchronize()
    n_params = sum(int(np.prod(t.shape)) for t in collect_params(model))
    print(f"  trainable params: {n_params/1e6:.1f}M")
    del sd
    mem_log("after model load + fp32 cast")

    if RESUME_FROM:
        print(f"\nresuming from checkpoint: {RESUME_FROM}")
        load_checkpoint(model, RESUME_FROM)
        print("  loaded.")
        mem_log("after ckpt resume")

    params = collect_params(model)
    # Bumped default 0.01 → 0.05 (2026-05-17) as part of the regularization pass
    # alongside stochastic depth + label smoothing. AdamW typical range for this scale.
    WEIGHT_DECAY = float(getenv("WEIGHT_DECAY", "0.05"))
    LABEL_SMOOTHING = float(getenv("LABEL_SMOOTHING", "0.0"))  # for log only — read by l3_training at import
    STOCH_DEPTH_P = float(getenv("STOCH_DEPTH_P", "0.0"))      # for log only — read by breathing at import
    opt = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    # Separate controller optimizer — closed-loop component #5 trains independently
    # via the lookup-CE signal flowing back through decisions. Created regardless
    # of CTRL_TRAIN so the ckpt round-trip is symmetric; only stepped when CTRL_TRAIN=1.
    ctrl_opt = AdamW(model.controller_parameters(), lr=CTRL_LR) if CTRL_TRAIN else None
    Tensor.training = True
    mem_log("after optimizer setup")

    rng = np.random.default_rng(SEED)
    py_rng = np.random.default_rng(SEED + 1)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_label_default = LEVEL.lower().replace(".", "_") + ("_spaced" if SPACE_DIGITS else "_abs")
    ckpt_label = getenv("CKPT_LABEL", ckpt_label_default)
    # ckpt_dir uses just the level prefix so spaced + abs share a directory
    ckpt_dir = os.path.join(project_root, ".cache", f"{LEVEL.lower().replace('.', '_')}_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # We don't pre-tokenize for multi-cycle — the encoder is called per-batch.
    # For a small corpus (L3 ~20K examples) this is fine.
    print()

    # Lazy state holder for calibration helpers (avoids re-computing digit_token_ids each step)
    class _CalibState: pass
    _calib_state = _CalibState()

    if CALIBRATION_MODE:
        print(f"[CALIBRATION] mode=ON  weight={CALIBRATION_WEIGHT}  loops={CALIBRATION_LOOPS}", flush=True)

    # Layer-pitch ramp env vars (re-read from os.environ to avoid coupling to breathing.py import order)
    # LAYER_PITCH_SLOPE: treadmill mode — angle = SLOPE * step, perpetual linear growth,
    #   no cap, no target. Adam's momentum aligns with a constant gradient direction
    #   ("adapt to slightly more rotation than last step"). No discontinuity when the
    #   ramp "ends" because it never ends. Takes precedence over LAYER_PITCH_TARGET.
    # LAYER_PITCH_TARGET (legacy): cosine/exp ramp to a fixed target, then hold.
    LAYER_PITCH_SLOPE = float(os.environ.get("LAYER_PITCH_SLOPE", "0.0"))
    LAYER_PITCH_TARGET = float(os.environ.get("LAYER_PITCH_TARGET", "0.0"))
    LAYER_PITCH_RAMP_STEPS = int(os.environ.get("LAYER_PITCH_RAMP_STEPS", "500"))
    if LAYER_PITCH_SLOPE > 0.0:
        print(f"[LAYER_PITCH] treadmill mode: slope={LAYER_PITCH_SLOPE:.2e} rad/step, no cap")
    elif LAYER_PITCH_TARGET > 0.0:
        print(f"[LAYER_PITCH] ramp mode: target={LAYER_PITCH_TARGET:.4f} rad, ramp_steps={LAYER_PITCH_RAMP_STEPS}, shape={os.environ.get('LAYER_PITCH_RAMP_SHAPE', 'cos')}")

    # v46b quadrature ramp — slowly grow the second-half-of-heads offset from 0 to π/2
    # over QUADRATURE_RAMP_STEPS so the model's W_O adapts to the diverging geometry.
    # See mycelium/breathing.py QUADRATURE_HEADS / QUADRATURE_RAMP_STEPS comments.
    from mycelium.breathing import (
        QUADRATURE_HEADS as _QH, QUADRATURE_RAMP_STEPS as _QRS,
        ACROSS_LAYER_PITCH_TARGET as _ALPT, ACROSS_LAYER_PITCH_RAMP_STEPS as _ALPRS,
    )
    if _QH and _QRS > 0:
        print(f"[QUADRATURE] ramp mode: second-half offset 0 → π/2 over {_QRS} steps, then hold")
    elif _QH:
        print(f"[QUADRATURE] full mode: second-half offset = π/2 from step 0 (no ramp)")
    if _ALPT > 0.0:
        import math as _alpt_math
        _base_step = (2 * _alpt_math.pi if bool(getenv("ROPE_FULL_CIRCLE", 0)) else _alpt_math.pi) / (cfg.n_phases * cfg.n_heads)
        if _ALPRS > 0:
            print(f"[ACROSS_LAYER_PITCH] ramp mode: per-layer step {_base_step:.4f} → {_ALPT:.4f} rad over {_ALPRS} steps")
        else:
            print(f"[ACROSS_LAYER_PITCH] fixed mode: per-layer step = {_ALPT:.4f} rad from step 0 (no ramp)")

    t_start = time.perf_counter()
    for step in range(STEPS):
        # Three-phase scheduling: cycle 0 (Phase A) gets heavy breathing,
        # subsequent cycles (Phase C) get light. The list is padded to actual cycle count
        # inside multi_cycle_train_step.
        phase_a_loops = int(py_rng.choice(TRAIN_LOOPS))
        loops_per_cycle = [phase_a_loops, PHASE_C_LOOPS]
        # v44 MIXED_LEVELS: per-step level sampling. Different levels have different
        # fixed_len (and n_cycles), so JIT compiles separately for each combination.
        if mixed_pool:
            chosen_level, lev_train, lev_fixed_len = mixed_pool[int(py_rng.integers(0, len(mixed_pool)))]
            cur_train = lev_train
            cur_fixed_len = lev_fixed_len
        elif LEVEL == "GSM8K_STEPS":
            # Sample a K-bucket uniformly, then BATCH problems from that bucket.
            # per_breath_train_step requires uniform K across the batch; this
            # guarantees that while still exercising all K values across training.
            bucket_keys = sorted(train_buckets)
            chosen_k = bucket_keys[int(py_rng.integers(0, len(bucket_keys)))]
            cur_train = train_buckets[chosen_k]
            cur_fixed_len = FIXED_LEN
        else:
            cur_train = train_examples
            cur_fixed_len = FIXED_LEN
        idx = rng.integers(0, len(cur_train), size=BATCH)
        batch_examples = [cur_train[i] for i in idx]

        # Update layer_pitch_scale every step. Assign in place so JIT graph identity is
        # preserved.
        #
        # Two modes (SLOPE takes precedence):
        #   LAYER_PITCH_SLOPE > 0  (treadmill): angle = SLOPE * step. Linear forever,
        #     no cap, no plateau. The gradient direction ("adapt to slightly more
        #     rotation") is constant in character — Adam's momentum can lock onto it.
        #     No discontinuity at any step. Tests the hypothesis that v16-v21
        #     collapses came from Adam momentum lagging a changing-then-frozen ramp.
        #   LAYER_PITCH_TARGET > 0 (legacy): cosine/exp ramp to fixed target, then
        #     hold. The "hold" point is where momentum mismatches stop being
        #     replenished — this is the suspected drift trigger.
        if LAYER_PITCH_SLOPE > 0.0:
            new_scale = LAYER_PITCH_SLOPE * step
            model.block.layer_pitch_scale.assign(
                Tensor([new_scale], dtype=dtypes.float).contiguous()
            )

        # v46b Quadrature ramp: per training step, recompute the per-head pitch
        # cos/sin tables at the current ramp scale and assign to buffers. JIT
        # graph captures the buffer reference and picks up new values on replay
        # (same pattern as layer_pitch_scale above and stoch_keep_mask below).
        if _QH and _QRS > 0:
            ramp_scale = min(step / float(_QRS), 1.0)
            import math as _qm
            half = cfg.n_heads // 2
            pitch_range_q = 2 * _qm.pi if bool(getenv("ROPE_FULL_CIRCLE", 0)) else _qm.pi
            ph_np = np.zeros((cfg.n_phases, cfg.n_heads), dtype=np.float32)
            for l in range(cfg.n_phases):
                base = l * pitch_range_q / (cfg.n_phases * cfg.n_heads)
                for h in range(cfg.n_heads):
                    extra = (_qm.pi / 2) * ramp_scale if h >= half else 0.0
                    ph_np[l, h] = base + extra
            cos_np = np.cos(ph_np).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1)
            sin_np = np.sin(ph_np).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1)
            model.block.per_head_pitch_cos.assign(Tensor(cos_np, dtype=dtypes.float).contiguous())
            model.block.per_head_pitch_sin.assign(Tensor(sin_np, dtype=dtypes.float).contiguous())

        # v47 Across-layer quadrature ramp: per-layer step grows from base
        # (π/64 default) to ACROSS_LAYER_PITCH_TARGET (π/2 for full quadrature)
        # over ACROSS_LAYER_PITCH_RAMP_STEPS. All heads within a layer keep the
        # same offset — only the layer-to-layer phase difference changes.
        # Mutually exclusive with QUADRATURE_HEADS for clarity.
        if _ALPT > 0.0 and _ALPRS > 0 and not _QH:
            import math as _alm
            pitch_range_a = 2 * _alm.pi if bool(getenv("ROPE_FULL_CIRCLE", 0)) else _alm.pi
            base_step = pitch_range_a / (cfg.n_phases * cfg.n_heads)
            ramp_scale = min(step / float(_ALPRS), 1.0)
            current_step = base_step + ramp_scale * (_ALPT - base_step)
            ph_np = np.zeros((cfg.n_phases, cfg.n_heads), dtype=np.float32)
            for l in range(cfg.n_phases):
                for h in range(cfg.n_heads):
                    ph_np[l, h] = l * current_step
            cos_np = np.cos(ph_np).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1)
            sin_np = np.sin(ph_np).reshape(cfg.n_phases, 1, cfg.n_heads, 1, 1)
            model.block.per_head_pitch_cos.assign(Tensor(cos_np, dtype=dtypes.float).contiguous())
            model.block.per_head_pitch_sin.assign(Tensor(sin_np, dtype=dtypes.float).contiguous())

        # Stochastic depth: resample per-breath keep mask each training step.
        # Mask values are 1/keep_prob when kept (Bernoulli) and 0 when dropped, so
        # E[breath contribution] is unchanged — output is an unbiased estimator of
        # the no-drop mean. Buffer is read inside the JIT'd graph; assign updates
        # data in place while preserving graph identity (same pattern as
        # layer_pitch_scale above).
        #
        # SAFEGUARDS (added 2026-05-18 after v45 take 1 collapsed):
        #   1. Skip stoch depth entirely at phase_a_loops < 2 — at n=1 dropping the
        #      sole breath zeros the integral, giving garbage output and an enormous
        #      gradient that damages the model. SD at n=1 has no regularization
        #      meaning anyway (can't regularize depth=1).
        #   2. At n>=2, ensure ≥1 of the first phase_a_loops slots is kept. With
        #      Bernoulli sampling at p, prob of all-dropped at n=2 is p² ≈ 1% (p=0.1).
        #      The safeguard forces a single random slot to "kept" in that case,
        #      preserving E[integral] unbiasedness only approximately but eliminating
        #      the catastrophic all-zero-integral event.
        if STOCH_DEPTH_P > 0.0 and phase_a_loops >= 2:
            keep_prob = 1.0 - STOCH_DEPTH_P
            mask_np = (py_rng.random(cfg.max_loops) < keep_prob).astype(np.float32) / keep_prob
            if mask_np[:phase_a_loops].sum() == 0:
                # All active breaths dropped — force one back on
                mask_np[int(py_rng.integers(0, phase_a_loops))] = 1.0 / keep_prob
            model.block.stoch_keep_mask.assign(
                Tensor(mask_np, dtype=dtypes.float).contiguous()
            )
        elif STOCH_DEPTH_P > 0.0:
            # phase_a_loops == 1 — reset mask to all-ones so the breath isn't dropped
            # (and isn't over-scaled by 1/keep_prob either).
            model.block.stoch_keep_mask.assign(
                Tensor.ones(cfg.max_loops, dtype=dtypes.float).contiguous()
            )
        elif LAYER_PITCH_TARGET > 0.0:
            import math as _m
            shape = os.environ.get("LAYER_PITCH_RAMP_SHAPE", "cos")
            if step < LAYER_PITCH_RAMP_STEPS:
                t_norm = step / LAYER_PITCH_RAMP_STEPS
                if shape == "exp":
                    k = float(os.environ.get("LAYER_PITCH_RAMP_K", "3.0"))
                    ramp_progress = (1.0 - _m.exp(-k * t_norm)) / (1.0 - _m.exp(-k))
                else:  # "cos" default
                    ramp_progress = 0.5 * (1.0 - _m.cos(_m.pi * t_norm))
            else:
                ramp_progress = 1.0
            new_scale = ramp_progress * LAYER_PITCH_TARGET
            model.block.layer_pitch_scale.assign(
                Tensor([new_scale], dtype=dtypes.float).contiguous()
            )

        t0 = time.perf_counter()
        calib_info = None
        per_breath_info = None
        # v52 Stage 1: per-breath supervision path (mutually exclusive with calibration)
        from mycelium.l3_training import per_breath_train_step
        from mycelium.breathing import PER_BREATH_DECODE as _PBD
        if _PBD and not CALIBRATION_MODE:
            from mycelium.l3_training import per_breath_train_step
            loss, per_breath_ce = per_breath_train_step(model, opt, batch_examples, tok,
                                                        fixed_len=cur_fixed_len,
                                                        lookup_aux_weight=LOOKUP_AUX_WEIGHT,
                                                        lookup_eq_token_id=eq_token_ids,
                                                        use_jit=USE_JIT)
            per_breath_info = per_breath_ce
        elif CALIBRATION_MODE:
            digit_ids_set = getattr(_calib_state, "digit_ids", None)
            if digit_ids_set is None:
                digit_ids_set = digit_token_ids_for(tok)
                _calib_state.digit_ids = digit_ids_set
            calib_info = calibration_train_step(model, opt, batch_examples, tok,
                                                digit_token_ids=digit_ids_set,
                                                eq_token_ids=eq_token_ids,
                                                n_loops=int(CALIBRATION_LOOPS),
                                                fixed_len=cur_fixed_len,
                                                calibration_weight=CALIBRATION_WEIGHT,
                                                use_jit=USE_JIT)
            loss = calib_info["loss"]
        elif PROFILE:
            loss, main_t = multi_cycle_train_step(model, opt, batch_examples, tok, loops_per_cycle, cur_fixed_len,
                                                  lookup_aux_weight=LOOKUP_AUX_WEIGHT,
                                                  lookup_eq_token_id=eq_token_ids,
                                                  profile=True, use_jit=USE_JIT)
        else:
            loss = multi_cycle_train_step(model, opt, batch_examples, tok, loops_per_cycle, cur_fixed_len,
                                          lookup_aux_weight=LOOKUP_AUX_WEIGHT,
                                          lookup_eq_token_id=eq_token_ids,
                                          use_jit=USE_JIT)
        # Controller training step (Step F) — throttled to every CTRL_TRAIN_EVERY
        # main steps. Cuts wall-clock per main step ~2× without dropping the
        # controller's actual learning rate (a less-frequent, more-stable signal
        # is fine; controller params are tiny vs the transformer).
        ctrl_loss = None
        ctrl_t = None
        if not CALIBRATION_MODE and ctrl_opt is not None and step % int(CTRL_TRAIN_EVERY) == 0:
            if PROFILE:
                ctrl_loss, ctrl_t = controller_train_step(model, ctrl_opt, batch_examples, tok,
                                                          eq_token_ids, max_loops=int(CTRL_MAX_LOOPS),
                                                          profile=True, compute_penalty=COMPUTE_PENALTY,
                                                          stop_calib_weight=STOP_CALIB_WEIGHT)
            else:
                ctrl_loss = controller_train_step(model, ctrl_opt, batch_examples, tok,
                                                  eq_token_ids, max_loops=int(CTRL_MAX_LOOPS),
                                                  compute_penalty=COMPUTE_PENALTY,
                                                  stop_calib_weight=STOP_CALIB_WEIGHT)
        dt = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_start

        # Accumulate per-phase profiles for periodic summary
        if PROFILE:
            if "_prof" not in dir():
                _prof = {"main_encode": [], "main_py": [], "main_gpu": [],
                         "ctrl_encode": [], "ctrl_py": [], "ctrl_gpu": [],
                         "wall": []}
            _prof["main_encode"].append(main_t["encode"])
            _prof["main_py"].append(main_t["py_overhead"])
            _prof["main_gpu"].append(main_t["gpu_compute"])
            if ctrl_t is not None:
                _prof["ctrl_encode"].append(ctrl_t["encode"])
                _prof["ctrl_py"].append(ctrl_t["py_overhead"])
                _prof["ctrl_gpu"].append(ctrl_t["gpu_compute"])
            _prof["wall"].append(dt)

        if step % 10 == 0 or step + 1 == STEPS:
            ctrl_str = f"  ctrl_loss={ctrl_loss:.4f}" if ctrl_loss is not None else ""
            if CALIBRATION_MODE and calib_info is not None:
                cpb = [int(x) for x in calib_info['correct_per_breath']]
                calib_str = (f"  ans_ce={calib_info['answer_ce']:.4f}  "
                             f"calib_bce={calib_info['calib_bce']:.4f}  "
                             f"cpb={cpb}  "
                             f"conf[+]={calib_info['mean_conf_correct']:.3f}  "
                             f"conf[-]={calib_info['mean_conf_wrong']:.3f}  "
                             f"(n+={calib_info['n_correct']} n-={calib_info['n_wrong']})")
                print(f"step {step:4d}  loops={CALIBRATION_LOOPS}  loss={calib_info['loss']:.4f}{calib_str}  ({dt:.2f}s, total {elapsed:.0f}s)", flush=True)
            elif per_breath_info is not None:
                # v52 Stage 1 per-breath logging: show CE for each breath
                pb_str = "  pb_ce=[" + " ".join(f"{c:.3f}" for c in per_breath_info) + "]"
                print(f"step {step:4d}  K={len(per_breath_info)}  loss={loss:.4f}{pb_str}{ctrl_str}  ({dt:.2f}s, total {elapsed:.0f}s)", flush=True)
            else:
                print(f"step {step:4d}  A={phase_a_loops} C={PHASE_C_LOOPS}  loss={loss:.4f}{ctrl_str}  ({dt:.2f}s, total {elapsed:.0f}s)", flush=True)
            # Log mem usage at step 0 (post first JIT compile) and every 50 steps after
            if step == 0 or (step % 50 == 0 and step > 0):
                mem_log(f"step {step:4d}")

        # Periodic gc.collect() to keep tinygrad's lazy-graph Python refs from
        # accumulating. Empirically we saw py_overhead grow 870ms → 2274ms over
        # 100 steps without this — the L3 trainer's "per-step time creeps up"
        # symptom. gc.collect() at K=50 keeps things bounded with negligible
        # cost (~10ms per collect on a 134M-param model).
        if (step + 1) % int(GC_EVERY) == 0:
            gc.collect()

        # Per-phase profile summary every PROFILE_EVERY steps
        if PROFILE and (step + 1) % int(PROFILE_EVERY) == 0:
            def _avg(xs): return sum(xs)/len(xs) if xs else 0.0
            print(f"  --- profile (last {len(_prof['wall'])} steps avg) ---")
            print(f"    main:  encode={_avg(_prof['main_encode'])*1000:.0f}ms  "
                  f"py_overhead={_avg(_prof['main_py'])*1000:.0f}ms  "
                  f"gpu={_avg(_prof['main_gpu'])*1000:.0f}ms")
            if _prof["ctrl_gpu"]:
                print(f"    ctrl:  encode={_avg(_prof['ctrl_encode'])*1000:.0f}ms  "
                      f"py_overhead={_avg(_prof['ctrl_py'])*1000:.0f}ms  "
                      f"gpu={_avg(_prof['ctrl_gpu'])*1000:.0f}ms  "
                      f"({len(_prof['ctrl_gpu'])}/{len(_prof['wall'])} steps)")
            print(f"    wall:  avg={_avg(_prof['wall'])*1000:.0f}ms/step", flush=True)
            for k in _prof: _prof[k].clear()

        # Cheap loss eval — Phase A loops vary, Phase C fixed.
        # Skipped for GSM8K_STEPS: multi_cycle_eval_loss assumes uniform K and
        # would IndexError on the variable-K eval batch. The per-breath paradigm
        # uses scripts/eval_ckpt_controller_segmented.py for real eval anyway.
        if LEVEL != "GSM8K_STEPS" and ((step + 1) % LOSS_EVAL_EVERY == 0 or step + 1 == STEPS):
            Tensor.training = False
            mem_log(f"step {step+1:4d}  before loss-eval (train JITs still resident)")
            # Clear training-side JITs so their resident graphs/buffers don't
            # contend with the eval JIT compile. Empirically: training JITs in
            # memory made the eval block's cached_generate_batch compile take
            # 20+ min instead of the expected ~10s warm replay. Clearing here
            # forces a one-time ~40s recompile of training graphs after each
            # eval block (acceptable trade for un-blocked eval).
            if USE_JIT:
                _JIT_TRAIN_CACHE.clear()
            gc.collect()
            mem_log(f"step {step+1:4d}  after JIT clear + gc")
            print(f"  --- loss eval at step {step+1} ---")
            for nl in EVAL_LOOPS:
                losses = []
                for _ in range(NUM_VAL_BATCHES):
                    eidx = rng.integers(0, len(eval_examples), size=BATCH)
                    eb = [eval_examples[i] for i in eidx]
                    losses.append(multi_cycle_eval_loss(model, eb, tok,
                                                       [nl, PHASE_C_LOOPS], FIXED_LEN))
                print(f"    val loss @ A={nl} C={PHASE_C_LOOPS}: {np.mean(losses):.4f}  (+-{np.std(losses):.3f})")
            mem_log(f"step {step+1:4d}  after loss-eval")
            Tensor.training = True

        # Expensive accuracy eval — same scheduling: Phase A varies, Phase C fixed
        _SKIP_FINAL_ACC = bool(int(os.environ.get("SKIP_FINAL_ACC", "0")))
        _is_final = (step + 1 == STEPS)
        _should_acc_eval = ((step + 1) % ACC_EVAL_EVERY == 0) or (_is_final and not _SKIP_FINAL_ACC)
        if _should_acc_eval:
            Tensor.training = False
            mem_log(f"step {step+1:4d}  before acc-eval (train JITs still resident)")
            if USE_JIT:
                _JIT_TRAIN_CACHE.clear()
            gc.collect()
            mem_log(f"step {step+1:4d}  after JIT clear + gc")
            print(f"  --- accuracy at step {step+1} (multi-cycle, {NUM_EVAL} held-out) ---")
            for nl in EVAL_LOOPS:
                t0 = time.perf_counter()
                acc, rows = accuracy_at_loops_multi(model, tok, eval_examples,
                                                    n_loops=[nl, PHASE_C_LOOPS],
                                                    batch_size=EVAL_BATCH,
                                                    cache_max_len=EVAL_CACHE_LEN)
                gt = time.perf_counter() - t0
                print(f"    acc @ A={nl} C={PHASE_C_LOOPS}: {acc*100:.1f}%  ({gt:.1f}s)", flush=True)
                if step + 1 == STEPS:
                    for ex, parsed, gen in rows[:2]:
                        print(f"      Q: {ex.problem}")
                        print(f"      gen: {gen.strip()!r}")
                        print(f"      parsed: {parsed}, gold: {ex.answer}, {'OK' if parsed == ex.answer else 'WRONG'}")
            # Per-checkpoint lookup-table eval — second axis of training signal.
            # Trains a fresh 16x1024 cosine-similarity table on op classification
            # from the integrated rep at "=" position. Reports held-out classification
            # accuracy + on-target count (out of 4) + per-op purity.
            if LOOKUP_EVAL:
                m = lookup_eval(model, tok, n_loops=int(LOOKUP_EVAL_LOOPS), verbose=False)
                pur = " ".join(f"{o}={m['purity_per_op'].get(o, 0):.2f}" for o in ["+","-","*","/"])
                print(f"    lookup-eval @ A={m['n_loops']}: trained={m['trained_acc']*100:.1f}%  "
                      f"ncm={m['ncm_acc']*100:.1f}%  on-target={m['on_target_count']}/4  "
                      f"purity[{pur}]  ({m['elapsed_s']:.1f}s)")
            mem_log(f"step {step+1:4d}  after acc-eval (eval JITs now resident)")
            Tensor.training = True

        # Periodic checkpoint
        if (step + 1) % CKPT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{ckpt_label}_step{step+1}.safetensors")
            safe_save(named_state(model), ckpt_path)
            print(f"  saved: {ckpt_path}")
            print()

    total = time.perf_counter() - t_start
    print(f"\n=== done. {STEPS} steps in {total:.0f}s ===")

    ckpt_path = os.path.join(ckpt_dir, f"{ckpt_label}_step{STEPS}.safetensors")
    safe_save(named_state(model), ckpt_path)
    print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
