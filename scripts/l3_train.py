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
    from mycelium.breathing import TWO_PHASE as _TP
    if _TP:
        # v68 TWO_PHASE: cast expand_shared and compress_shared (the active shared weights).
        # The legacy model.block.shared is an unused placeholder — skip it.
        for sw in (model.block.expand_shared, model.block.compress_shared):
            for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
                _cast(sw, a)
        for layer in list(model.block.expand_layers) + list(model.block.compress_layers):
            for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
                _cast(layer, a)
    else:
        sw = model.block.shared
        for a in ("wv", "bv", "wo", "bo", "w_out", "b_out"):
            _cast(sw, a)
        for layer in model.block.layers:
            for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
                _cast(layer, a)


def collect_params(model):
    # v85 (2026-05-27): v85 path doesn't use final ln_f / embed_out. Skip them
    # when V85_QUERYABLE=1 so AdamW doesn't assert on None grad.
    from mycelium.breathing import V85_QUERYABLE as _V85_Q_BASE
    if _V85_Q_BASE:
        nps = [model.embed.weight]  # used via embed_in for number-span pooling
    else:
        nps = [model.embed.weight, model.embed_out, model.ln_f_g, model.ln_f_b]
    from mycelium.breathing import DOUBLED_LAYERS as _DL_FLAG, TWO_PHASE as _TWO_PHASE
    if _TWO_PHASE:
        # v68 TWO_PHASE: include expand+compress sets; EXCLUDE old shared/layers/layers_b
        # which aren't touched in forward (would have None grad → optimizer assertion).
        for sw in (model.block.expand_shared, model.block.compress_shared):
            nps += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
                    sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
        for layer in list(model.block.expand_layers) + list(model.block.compress_layers):
            nps += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    else:
        sw = model.block.shared
        nps += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
                sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
        for layer in model.block.layers:
            nps += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
        # v44 doubled-layers Set B — only included in optimizer when actually in use
        # (would have undefined gradient otherwise since not touched in forward).
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
    # v85: lookup_table.values is touched via match_weights computation inside
    # breathe_with_lookup (running_normed → lookup_table); skip if v85.
    from mycelium.breathing import V85_QUERYABLE as _V85_Q_LT
    if not _V85_Q_LT:
        nps += [model.lookup_table.values, model.lookup_table.value_proj_up]
    # v38 B-field IB bottleneck — learned when BFIELD_WAIST > 0. Always included
    # for ckpt symmetry; L2 reg in l3_training keeps gradient defined when off.
    nps += [model.block.bfield_proj_down, model.block.bfield_proj_up, model.block.bfield_bias]
    # v50 IB-waist codebook — learnable keys + values for op-discriminative compression.
    # Always included for ckpt symmetry; gradient inert when WAIST_CODEBOOK_N=0
    # (values are zero-init → contribution is zero → gradient through retrieved is zero).
    nps += [model.block.waist_codebook_keys, model.block.waist_codebook_values]
    # v69 collapse — JPEG-style lossy compression. Only included when COLLAPSE_V69=1
    # (forward path uses these only then; otherwise they'd have None grad → opt.step fails).
    from mycelium.breathing import COLLAPSE_V69 as _COLLAPSE_V69
    if _COLLAPSE_V69:
        nps += [model.block.collapse_codebook_keys, model.block.collapse_codebook_values,
                model.block.collapse_gate_w, model.block.collapse_gate_b,
                model.block.collapse_proj_down, model.block.collapse_proj_up,
                model.block.collapse_alpha]
    # v70 collapse — waist-dim codebook + split gate (proto/breath) + budget sparsity.
    from mycelium.breathing import COLLAPSE_V70 as _COLLAPSE_V70
    if _COLLAPSE_V70:
        nps += [model.block.collapse_v70_codebook_keys, model.block.collapse_v70_codebook_values,
                model.block.collapse_v70_proj_down, model.block.collapse_v70_proj_up,
                model.block.collapse_v70_bias,
                model.block.collapse_v70_gate_w_proto, model.block.collapse_v70_gate_w_breath,
                model.block.collapse_v70_gate_b,
                model.block.collapse_v70_breath_embed, model.block.collapse_v70_alpha]
    # v71 collapse — refined v70 (10× sparsity weight, lower gate bias, k-means codebook,
    # cleaner controller signal). Only in optimizer when COLLAPSE_V71=1.
    from mycelium.breathing import COLLAPSE_V71 as _COLLAPSE_V71
    if _COLLAPSE_V71:
        nps += [model.block.collapse_v71_codebook_keys, model.block.collapse_v71_codebook_values,
                model.block.collapse_v71_proj_down, model.block.collapse_v71_proj_up,
                model.block.collapse_v71_bias,
                model.block.collapse_v71_gate_w_proto, model.block.collapse_v71_gate_w_breath,
                model.block.collapse_v71_gate_b,
                model.block.collapse_v71_breath_embed, model.block.collapse_v71_alpha]
    # v39 waist head (512 → 4 op classifier) — trained when BFIELD_AUX_WEIGHT > 0
    # and BFIELD_END_OF_BREATH=1. L2 reg covers the off case.
    # v85 (2026-05-27): v85 path doesn't touch this head; skip when V85_QUERYABLE.
    from mycelium.breathing import V85_QUERYABLE as _V85_Q_CP2
    if not _V85_Q_CP2:
        nps += [model.waist_head_w, model.waist_head_b]
        # Calibration head — trained on the main loss via REINFORCE in calibration_train_step.
        # Always included so opt.step() has a defined gradient for these params even when
        # CALIBRATION_MODE=0 (gradient is zero in that path, weights don't move).
        nps += model.confidence_head.parameters()
    # v78 per-head model codebook — only added to optimizer when V78_HEAD_CODEBOOK=1
    # (forward path skips it otherwise → grad would be None → AdamW assertion).
    # State_dict registration is separate (always present for ckpt symmetry).
    from mycelium.breathing import V78_HEAD_CODEBOOK as _V78_HC, TWO_PHASE as _TP_V78
    if _V78_HC:
        if _TP_V78:
            for layer in list(model.block.expand_layers) + list(model.block.compress_layers):
                nps += [layer.v78_head_codebook]
        else:
            for layer in model.block.layers:
                nps += [layer.v78_head_codebook]
            # v44 layers_b participate only when DOUBLED_LAYERS=1 (forward path uses Set B).
            from mycelium.breathing import DOUBLED_LAYERS as _DL_V78
            if _DL_V78:
                for layer in model.block.layers_b:
                    nps += [layer.v78_head_codebook]
    # v54 WaistController — cross-attention text decoder over (waist, prompt).
    # Only added to params (and therefore optimizer-trained) when CONTROLLER_DECODE=1,
    # because the other train paths don't touch it and AdamW would assert on missing grads.
    from mycelium.breathing import CONTROLLER_DECODE as _CD, WAIST_COPY as _WC, MULTI_HEAD_WAIST as _MHW
    from mycelium.breathing import V85_QUERYABLE as _V85_Q_CP
    # v85 (2026-05-27): v85 path doesn't fire the WaistController (it uses the
    # v85 slot decoder instead). Skip the WaistController params when V85_QUERYABLE
    # is on — they'd have None grad and AdamW would assert.
    if _CD and not _V85_Q_CP:
        # Get the base params, but EXCLUDE the v72 copy params from the default list:
        # they're tail-appended by WaistController.parameters() so we can include them
        # only when WAIST_COPY=1 (forward path skips them otherwise, AdamW would assert
        # on None grad). Match by identity to stay robust if list order changes.
        all_wc = model.waist_controller.parameters()
        copy_ids = {id(model.waist_controller.copy_q_w),
                    id(model.waist_controller.copy_k_w),
                    id(model.waist_controller.copy_gate_w),
                    id(model.waist_controller.copy_gate_b)}
        base_wc = [p for p in all_wc if id(p) not in copy_ids]
        nps += base_wc
        if _WC:
            nps += [model.waist_controller.copy_q_w,
                    model.waist_controller.copy_k_w,
                    model.waist_controller.copy_gate_w,
                    model.waist_controller.copy_gate_b]
        # v81 multi-head MLPs — only optimizer-trained when MULTI_HEAD_WAIST=1
        # (otherwise the forward path is the single-head legacy and the MLPs see
        # no gradient → AdamW would assert).
        if _MHW:
            nps += model.waist_controller.multi_head_parameters()
    # v85 (2026-05-27) Queryable slot decoder — only optimizer-trained when
    # V85_QUERYABLE=1. When OFF, the forward path skips the slot decoder so its
    # params would have None grad → AdamW would assert.
    from mycelium.breathing import V85_QUERYABLE as _V85_Q
    if _V85_Q:
        nps += model.v85_slot_decoder.parameters()
    # v96 (2026-05-28) Consolidation-table params — optimizer-trained when
    # V96_CONSOLIDATION=1. State-dict registration is separate (always there).
    from mycelium.breathing import V96_CONSOLIDATION as _V96
    if _V96:
        nps += [model.v96_gate_w, model.v96_gate_b,
                 model.v96_ops_codebook, model.v96_types_codebook,
                 model.v96_summary_proj, model.v96_table_kv_proj,
                 model.v96_table_alpha,    # v96.1
                 # v96.2 constraint-check heads (Bombe-inspired elimination).
                 # Active gradient via the v96 energy loss when V96_ENERGY_WEIGHT > 0;
                 # L2 reg keeps grad defined otherwise.
                 model.v96_ref_validity_head_w, model.v96_ref_validity_head_b,
                 model.v96_arg_order_head_w,    model.v96_arg_order_head_b]
    # v97 (2026-05-28) Calibration head — optimizer-trained when V97_CALIBRATION=1.
    # State-dict registration is always-on; AdamW would assert None grad otherwise
    # (forward path skips when V97=0). Pure aux loss — no residual-stream feedback.
    from mycelium.breathing import V97_CALIBRATION as _V97
    if _V97:
        nps += [model.v97_calib_head_w, model.v97_calib_head_b]
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
        #
        # v77 path (V77_DAG_TRAINING=1): swap the loader to load_gsm8k_v77. Each
        # example carries per_layer_target (length=6) instead of gen_targets; we
        # still bucket by n_steps for the batch-size assertion to hold within
        # per_breath_train_step (uniform-K assertion is satisfied since N_BREATHS
        # is fixed at 6 in the V77 path; the K bucketing serves a different purpose:
        # keeping problems of similar complexity together for stable training).
        V77_DAG_TRAINING = int(getenv("V77_DAG_TRAINING", "0")) > 0
        V85_QUERYABLE = int(getenv("V85_QUERYABLE", "0")) > 0
        GSM8K_STEPS_PATH = getenv("GSM8K_STEPS_PATH", ".cache/gsm8k_steps_v1_train.jsonl")
        GSM8K_STEPS_MIN_K = int(getenv("GSM8K_STEPS_MIN_K", 2))
        GSM8K_STEPS_MAX_K = int(getenv("GSM8K_STEPS_MAX_K", 6))  # K=7+ pushes FIXED_LEN; cap by default
        if V85_QUERYABLE:
            from mycelium.l3_data import load_gsm8k_v85 as _v85_loader
            print(f"loading v85 queryable-structures GSM8K from {GSM8K_STEPS_PATH} "
                  f"(min_k={GSM8K_STEPS_MIN_K}, max_k={GSM8K_STEPS_MAX_K})...")
            t0 = time.perf_counter()
            gsm8k_step_buckets = _v85_loader(GSM8K_STEPS_PATH,
                                              min_k=GSM8K_STEPS_MIN_K, max_k=GSM8K_STEPS_MAX_K,
                                              require_sympy_match=True,
                                              bucket_by_k=True)
        elif V77_DAG_TRAINING:
            from mycelium.l3_data import load_gsm8k_v77 as _v77_loader
            print(f"loading v77 DAG-layered GSM8K from {GSM8K_STEPS_PATH} (min_k={GSM8K_STEPS_MIN_K}, max_k={GSM8K_STEPS_MAX_K})...")
            t0 = time.perf_counter()
            gsm8k_step_buckets = _v77_loader(GSM8K_STEPS_PATH,
                                              min_k=GSM8K_STEPS_MIN_K, max_k=GSM8K_STEPS_MAX_K,
                                              require_sympy_match=True,
                                              bucket_by_k=True)
        else:
            from mycelium.l3_data import load_gsm8k_steps
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

        def _ex_k(e):
            if V85_QUERYABLE or V77_DAG_TRAINING:
                return getattr(e, "n_steps", None)
            return len(e.gen_targets)

        print(f"  loaded {total_keep} examples across K={sorted(gsm8k_step_buckets)} buckets:")
        for k in sorted(train_buckets):
            print(f"    K={k}: train={len(train_buckets[k])} eval={sum(1 for e in eval_examples if _ex_k(e) == k)}")
        print(f"  total: train={sum(len(v) for v in train_buckets.values())} eval={len(eval_examples)} ({time.perf_counter()-t0:.1f}s)")
        ex0 = train_examples[0] if train_examples else None
        if ex0:
            if V85_QUERYABLE:
                print(f"  sample v85 (n_steps={ex0.n_steps}, n_numbers={len(ex0.numbers)}, "
                      f"n_implicit={len(ex0.implicit_numbers)}, n_verbs={len(ex0.verbs)}): "
                      f"{ex0.problem[:100]}")
                for k, s in enumerate(ex0.dag_slots):
                    a1, a2 = s["args"]
                    print(f"    slot {k}: op={s['op']}  type={s['type_path']}  "
                          f"a1={a1['source']}[{a1['index']}]  a2={a2['source']}[{a2['index']}]")
            elif V77_DAG_TRAINING:
                print(f"  sample v77 (n_steps={ex0.n_steps}, n_layers={ex0.n_layers}): {ex0.problem[:100]}")
                for ell in range(ex0.n_layers):
                    print(f"    L{ell}: {ex0.per_layer_target[ell][:120]}")
            else:
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

    # v96 (2026-05-28) Optionally initialize the types codebook from IB centroids.
    # The IB centroids are 1024d (Pythia embedding space), normalized to scale 0.02,
    # so they form an op-discriminative starting basis. Only runs when v96 is on AND
    # the ckpt didn't have a saved v96_types_codebook (or to override).
    from mycelium.breathing import V96_CONSOLIDATION as _V96_INIT
    V96_INIT_FROM_IB = int(os.environ.get("V96_INIT_FROM_IB", "1")) > 0
    if _V96_INIT and V96_INIT_FROM_IB:
        from mycelium.v96 import load_ib_codebooks_into_v96 as _v96_ib_init
        ok = _v96_ib_init(model, hidden=cfg.hidden)
        if ok:
            print(f"[v96] types_codebook initialized from .cache/ib_centroids.npz "
                  f"(32 IB leaves × {cfg.hidden}d, normalized to scale 0.02)")
        else:
            print(f"[v96] IB centroid init skipped (file missing or shape mismatch)")

    # v77b orthogonal breath_embed override. Applied AFTER load_checkpoint so
    # the warm-start ckpt's tiny trained breath_embed values don't overwrite
    # the orthogonal init. The override targets model.block.breath_embed
    # in-place via .assign() to preserve tensor identity for the optimizer.
    BREATH_EMBED_ORTHO_INIT = float(getenv("BREATH_EMBED_ORTHO_INIT", "0.0"))
    if BREATH_EMBED_ORTHO_INIT > 0.0:
        from mycelium.breathing import _make_orthogonal_breath_embed
        ortho_np = _make_orthogonal_breath_embed(cfg.max_loops, cfg.hidden, BREATH_EMBED_ORTHO_INIT, seed=42)
        ortho_t = Tensor(ortho_np, dtype=model.block.breath_embed.dtype).to(model.block.breath_embed.device).contiguous()
        model.block.breath_embed.assign(ortho_t).realize()
        norms = np.linalg.norm(ortho_np, axis=1)
        print(f"  breath_embed ortho-init applied: L2 norms = [{', '.join(f'{n:.3f}' for n in norms)}]")

    # v87 (2026-05-27) Slot-symmetry fix.
    # Reinitialize slot_pos_embed and v86_args_slot_pos AFTER ckpt load when
    # warm-starting from a v86 ckpt whose saved values are at the small v85/v86
    # scale. The in-place .assign() preserves tensor identity for AdamW.
    V87_SLOT_POS_INIT_SCALE_LOCAL = float(getenv("V87_SLOT_POS_INIT_SCALE", "0.0"))
    V87_REINIT_SLOT_POS = int(getenv("V87_REINIT_SLOT_POS", "0")) > 0
    if V87_REINIT_SLOT_POS and V87_SLOT_POS_INIT_SCALE_LOCAL > 0.0:
        from mycelium.breathing import V85_QUERYABLE as _V85_Q_V87
        if _V85_Q_V87:
            dec = model.v85_slot_decoder
            K_max_v, H_v = dec.slot_pos_embed.shape
            # Independent uniform draws for the two params (same scale).
            new_pos = Tensor.uniform(
                K_max_v, H_v,
                low=-V87_SLOT_POS_INIT_SCALE_LOCAL, high=V87_SLOT_POS_INIT_SCALE_LOCAL,
                dtype=dec.slot_pos_embed.dtype).to(dec.slot_pos_embed.device).contiguous()
            dec.slot_pos_embed.assign(new_pos).realize()
            new_args_pos = Tensor.uniform(
                K_max_v, H_v,
                low=-V87_SLOT_POS_INIT_SCALE_LOCAL, high=V87_SLOT_POS_INIT_SCALE_LOCAL,
                dtype=dec.v86_args_slot_pos.dtype).to(dec.v86_args_slot_pos.device).contiguous()
            dec.v86_args_slot_pos.assign(new_args_pos).realize()
            print(f"[v87] reinitialized slot_pos_embed + v86_args_slot_pos at scale "
                  f"{V87_SLOT_POS_INIT_SCALE_LOCAL} (K_max={K_max_v}, H={H_v})")
        else:
            print("[v87] V87_REINIT_SLOT_POS=1 but V85_QUERYABLE=0 — skipping reinit")

    # v88 (2026-05-27) — args cross-attn K/V projection reinit.
    # The v87 diagnostic confirmed that even after fixing slot_pos diversity (Q
    # side), the K/V projections of the args cross-attn remained near-zero
    # (`v86_args_k_proj` L2 = 0.10 after v86 + v87 300 steps, `v86_args_v_proj`
    # L2 = 0.39). Because attn_scores = q @ k.T, K-near-zero collapses softmax
    # to uniform regardless of Q diversity — chicken-and-egg vanishing gradient.
    # v88 reinitializes K and V at scale V88_KV_PROJ_INIT_SCALE (default 0.02,
    # matching the Pythia randn-scale used for other slot decoder params) so the
    # attention pattern starts with meaningful score variation, breaking the
    # gradient deadlock. Same in-place assign pattern as v87.
    V88_KV_PROJ_INIT_SCALE_LOCAL = float(getenv("V88_KV_PROJ_INIT_SCALE", "0.02"))
    V88_REINIT_KV_PROJ = int(getenv("V88_REINIT_KV_PROJ", "0")) > 0
    if V88_REINIT_KV_PROJ and V88_KV_PROJ_INIT_SCALE_LOCAL > 0.0:
        from mycelium.breathing import V85_QUERYABLE as _V85_Q_V88
        if _V85_Q_V88:
            dec = model.v85_slot_decoder
            scale = V88_KV_PROJ_INIT_SCALE_LOCAL
            K_shape = dec.v86_args_k_proj.shape
            V_shape = dec.v86_args_v_proj.shape
            new_k = (Tensor.randn(*K_shape) * scale).cast(
                dec.v86_args_k_proj.dtype).to(dec.v86_args_k_proj.device).contiguous()
            dec.v86_args_k_proj.assign(new_k).realize()
            new_v = (Tensor.randn(*V_shape) * scale).cast(
                dec.v86_args_v_proj.dtype).to(dec.v86_args_v_proj.device).contiguous()
            dec.v86_args_v_proj.assign(new_v).realize()
            # Report post-reinit L2s for the smoke log.
            k_l2 = float((dec.v86_args_k_proj.cast(dtypes.float) ** 2).sum().sqrt().numpy())
            v_l2 = float((dec.v86_args_v_proj.cast(dtypes.float) ** 2).sum().sqrt().numpy())
            print(f"[v88] reinitialized v86_args_k_proj (shape {tuple(K_shape)}) and "
                  f"v86_args_v_proj (shape {tuple(V_shape)}) at randn-scale {scale}")
            print(f"[v88] post-reinit L2: k_proj={k_l2:.4f}  v_proj={v_l2:.4f}")
        else:
            print("[v88] V88_REINIT_KV_PROJ=1 but V85_QUERYABLE=0 — skipping reinit")

    # v89 (2026-05-27) — split args1/args2 cross-attn K/V projection inits.
    # When warm-starting from a v88 ckpt (single shared v86 K/V), the v89 split
    # projections start either as fresh-random (V89_INHERIT_V86=0) or copied
    # from v86 values (V89_INHERIT_V86=1) so we begin where v88's cross-attn
    # left off. Inheriting is recommended — the v88 K/V are already trained to
    # produce DIVERSE per-slot attention; we want to keep that diversity and
    # let the supervised loss reshape it toward the gold number positions.
    V89_SUPERVISED_ATTN_LOCAL = int(getenv("V89_SUPERVISED_ATTN", "0")) > 0
    V89_INHERIT_V86 = int(getenv("V89_INHERIT_V86", "0")) > 0
    if V89_SUPERVISED_ATTN_LOCAL and V89_INHERIT_V86:
        from mycelium.breathing import V85_QUERYABLE as _V85_Q_V89
        if _V85_Q_V89:
            dec = model.v85_slot_decoder
            # Copy v86 K/V into both args1 and args2 K/V projections.
            # The clone() ensures each receives its own storage (subsequent
            # gradient updates diverge).
            k_src = dec.v86_args_k_proj.detach().cast(dec.v89_args1_k_proj.dtype).to(dec.v89_args1_k_proj.device).contiguous()
            v_src = dec.v86_args_v_proj.detach().cast(dec.v89_args1_v_proj.dtype).to(dec.v89_args1_v_proj.device).contiguous()
            dec.v89_args1_k_proj.assign(k_src.contiguous()).realize()
            dec.v89_args2_k_proj.assign(k_src.contiguous()).realize()
            dec.v89_args1_v_proj.assign(v_src.contiguous()).realize()
            dec.v89_args2_v_proj.assign(v_src.contiguous()).realize()
            k1_l2 = float((dec.v89_args1_k_proj.cast(dtypes.float) ** 2).sum().sqrt().numpy())
            v1_l2 = float((dec.v89_args1_v_proj.cast(dtypes.float) ** 2).sum().sqrt().numpy())
            k2_l2 = float((dec.v89_args2_k_proj.cast(dtypes.float) ** 2).sum().sqrt().numpy())
            v2_l2 = float((dec.v89_args2_v_proj.cast(dtypes.float) ** 2).sum().sqrt().numpy())
            print(f"[v89] inherited v86 K/V into args1 + args2 split projections")
            print(f"[v89] post-inherit L2: a1.k={k1_l2:.4f} a1.v={v1_l2:.4f} a2.k={k2_l2:.4f} a2.v={v2_l2:.4f}")
        else:
            print("[v89] V89_INHERIT_V86=1 but V85_QUERYABLE=0 — skipping")

    # v90 (2026-05-27) — reset active-head BIAS to a negative value so the
    # starting prediction defaults to False (sigmoid(-1.0) ≈ 0.27 < 0.5).
    #
    # Context: v89 trained the active head with V86_ACTIVE_POS_WEIGHT=5.0 (5x FN
    # penalty) for 200 steps. Result was an over-firing active head: every
    # problem had active=[T,T,T,T,T,T,F,F,F,F] (exactly 6 True) regardless of
    # n_steps, polluting the DAG with phantom slots that inherited the
    # degenerate args2_pred = 20+k pattern.
    #
    # v90 plan:
    #   1. V86_ACTIVE_POS_WEIGHT=1.0 in the launcher (balances FP vs FN).
    #   2. V90_RESET_ACTIVE_HEAD=1 here — reset bias to -1.0 so step 0 default
    #      is "predict False" instead of inheriting the baked-in over-fire.
    # The weight matrix stays (preserves learned slot-query features).
    V90_RESET_ACTIVE_HEAD = int(getenv("V90_RESET_ACTIVE_HEAD", "0")) > 0
    V90_ACTIVE_BIAS = float(getenv("V90_ACTIVE_BIAS", "-1.0"))
    if V90_RESET_ACTIVE_HEAD:
        import math as _v90_math
        from mycelium.breathing import V85_QUERYABLE as _V85_Q_V90
        if _V85_Q_V90:
            dec = model.v85_slot_decoder
            old_b = float(dec.active_head_b.cast(dtypes.float).numpy().reshape(-1)[0])
            new_b = Tensor.full(dec.active_head_b.shape, V90_ACTIVE_BIAS, dtype=dec.active_head_b.dtype).to(
                dec.active_head_b.device).contiguous()
            dec.active_head_b.assign(new_b).realize()
            new_b_val = float(dec.active_head_b.cast(dtypes.float).numpy().reshape(-1)[0])
            print(f"[v90] reset active_head_b: {old_b:.4f} -> {new_b_val:.4f} "
                  f"(sigmoid bias = {1.0 / (1.0 + _v90_math.exp(-new_b_val)):.4f})")
        else:
            print("[v90] V90_RESET_ACTIVE_HEAD=1 but V85_QUERYABLE=0 — skipping")

    # v92 (2026-05-28) — arg_pos_emb scale reinit.
    # V91 introduced arg_pos_emb as a (2, H) zero-init tensor that distinguishes
    # args1 from args2. Empirically the v91 ckpt's rows ended near-identical
    # (cos 0.915, L2 ~10), causing args1 ≈ args2 → same pointer for both arg
    # positions. v92 reinitializes from uniform(-scale, scale) so the two rows
    # start orthogonal-ish and the args1/args2 supervised gradients pull them
    # in opposite directions. Same in-place .assign() pattern as v87/v88/v90.
    V92_ARG_POS_EMB_SCALE_LOCAL = float(getenv("V92_ARG_POS_EMB_SCALE", "0.0"))
    V92_REINIT_ARG_POS_EMB = int(getenv("V92_REINIT_ARG_POS_EMB", "0")) > 0
    if V92_REINIT_ARG_POS_EMB and V92_ARG_POS_EMB_SCALE_LOCAL > 0.0:
        from mycelium.breathing import V85_QUERYABLE as _V85_Q_V92A
        if _V85_Q_V92A:
            dec = model.v85_slot_decoder
            shape = dec.arg_pos_emb.shape  # (2, H)
            # Pre-reinit diagnostic: cosine between row 0 and row 1.
            old_np = dec.arg_pos_emb.cast(dtypes.float).numpy().reshape(shape[0], shape[1])
            old_norms = np.linalg.norm(old_np, axis=1) + 1e-9
            old_cos = float(np.dot(old_np[0], old_np[1]) / (old_norms[0] * old_norms[1]))
            old_l2 = float(np.linalg.norm(old_np))
            new_emb = Tensor.uniform(
                *shape,
                low=-V92_ARG_POS_EMB_SCALE_LOCAL, high=V92_ARG_POS_EMB_SCALE_LOCAL,
                dtype=dec.arg_pos_emb.dtype).to(dec.arg_pos_emb.device).contiguous()
            dec.arg_pos_emb.assign(new_emb).realize()
            new_np = dec.arg_pos_emb.cast(dtypes.float).numpy().reshape(shape[0], shape[1])
            new_norms = np.linalg.norm(new_np, axis=1) + 1e-9
            new_cos = float(np.dot(new_np[0], new_np[1]) / (new_norms[0] * new_norms[1]))
            new_l2 = float(np.linalg.norm(new_np))
            print(f"[v92] reinitialized arg_pos_emb at scale {V92_ARG_POS_EMB_SCALE_LOCAL} "
                  f"(shape {tuple(shape)})")
            print(f"[v92] arg_pos_emb row-cos: {old_cos:.4f} -> {new_cos:.4f}  "
                  f"L2: {old_l2:.4f} -> {new_l2:.4f}")
        else:
            print("[v92] V92_REINIT_ARG_POS_EMB=1 but V85_QUERYABLE=0 — skipping")

    # v92 (2026-05-28) — neutral active-head bias reset.
    # v90 set bias to -1.0 to default to "predict False" after the over-firing
    # v89 ckpt. v91 trained 100 more steps from that base; combined effect was
    # active_logits mean = -1.67 at eval (ALL slots False, only v90 slot-0
    # fallback rescued parse rate). v92 resets bias to 0.0 so the model starts
    # neutral and learns active=True/False from training-data balance directly.
    V92_RESET_ACTIVE_HEAD_NEUTRAL = int(getenv("V92_RESET_ACTIVE_HEAD_NEUTRAL", "0")) > 0
    if V92_RESET_ACTIVE_HEAD_NEUTRAL:
        from mycelium.breathing import V85_QUERYABLE as _V85_Q_V92B
        if _V85_Q_V92B:
            dec = model.v85_slot_decoder
            old_b = float(dec.active_head_b.cast(dtypes.float).numpy().reshape(-1)[0])
            new_b = Tensor.zeros(*dec.active_head_b.shape, dtype=dec.active_head_b.dtype).to(
                dec.active_head_b.device).contiguous()
            dec.active_head_b.assign(new_b).realize()
            new_b_val = float(dec.active_head_b.cast(dtypes.float).numpy().reshape(-1)[0])
            print(f"[v92] reset active_head_b NEUTRAL: {old_b:.4f} -> {new_b_val:.4f} "
                  f"(sigmoid bias = 0.5)")
        else:
            print("[v92] V92_RESET_ACTIVE_HEAD_NEUTRAL=1 but V85_QUERYABLE=0 — skipping")

    # v85 (2026-05-27): init types_codebook from IB centroids when V85_QUERYABLE=1.
    # The IB tree has 32 leaves in (4 ops × ~8 sub-clusters) arrangement. The centroids
    # were computed from Pythia embeddings of L2 NL step descriptions and are already
    # in the model's residual-stream space (1024d). Loading them gives the codebook
    # a meaningful starting point — semantic similarity in input space → similarity
    # in target space.
    from mycelium.breathing import V85_QUERYABLE as _V85_INIT
    if _V85_INIT and os.path.exists(".cache/ib_centroids.npz"):
        cent_data = np.load(".cache/ib_centroids.npz")
        cent_np = cent_data["centroids"].astype(np.float32)
        # Centroid scale may be very different from 0.02 random init. Project to a
        # similar scale so the slot decoder doesn't blow up at step 0.
        cent_norm = np.linalg.norm(cent_np, axis=1, keepdims=True) + 1e-6
        cent_normed = cent_np / cent_norm * 1.0   # unit-norm centroids
        # Match expected shape (V85_TYPES_N, hidden).
        target_shape = model.v85_slot_decoder.types_codebook.shape
        if cent_normed.shape == target_shape:
            t = Tensor(cent_normed, dtype=model.v85_slot_decoder.types_codebook.dtype).to(
                model.v85_slot_decoder.types_codebook.device).contiguous()
            model.v85_slot_decoder.types_codebook.assign(t).realize()
            print(f"  v85 types_codebook init from .cache/ib_centroids.npz "
                  f"({cent_normed.shape}, unit-normed)")
        else:
            print(f"  v85 types_codebook init SKIPPED (shape mismatch: "
                  f"centroids {cent_normed.shape} vs codebook {target_shape})")

    params = collect_params(model)
    # Bumped default 0.01 → 0.05 (2026-05-17) as part of the regularization pass
    # alongside stochastic depth + label smoothing. AdamW typical range for this scale.
    WEIGHT_DECAY = float(getenv("WEIGHT_DECAY", "0.05"))
    LABEL_SMOOTHING = float(getenv("LABEL_SMOOTHING", "0.0"))  # for log only — read by l3_training at import
    STOCH_DEPTH_P = float(getenv("STOCH_DEPTH_P", "0.0"))      # for log only — read by breathing at import
    GRAD_CLIP = float(getenv("GRAD_CLIP", "0.0"))              # global-norm gradient clip; 0 = off, 1.0 = standard
    # v84 (2026-05-27) — cosine LR decay to zero over STEPS. Set
    # LR_DECAY_TO_ZERO=1 to enable; default 0 (backwards-compatible flat LR).
    # Both the regular path and the SS path read opt.lr inside the JIT, so
    # in-place .assign() of the buffer propagates without recompile (same
    # pattern as layer_pitch_scale).
    LR_DECAY_TO_ZERO = bool(getenv("LR_DECAY_TO_ZERO", 0))
    if LR_DECAY_TO_ZERO:
        print(f"[LR_DECAY] cosine to 0 over {STEPS} steps (initial lr={LR})")
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

        # v84 (2026-05-27) — cosine LR decay to zero.
        # opt.lr is a single-element tensor; in-place .assign() updates the
        # buffer that the JIT graph already references (no recompile needed —
        # same pattern as layer_pitch_scale.assign(...) above). Realize the
        # new value before the JIT step picks it up.
        # Printed every 100 steps so we can verify the schedule visually.
        #
        # v92 (2026-05-28) — linear warmup over first V92_LR_WARMUP_STEPS steps.
        # The v91 args_ce 60→1 transient at steps 0-50 (initially-wrong args
        # codebook pulled toward gold positions) caused a loss spike around
        # step 50-60. Linear warmup from 0 to LR over the first N steps lets
        # the args projection settle without large early updates blowing up
        # the active head and waist pool projection.
        if LR_DECAY_TO_ZERO:
            import math as _lrm
            _v92_warmup = int(getenv("V92_LR_WARMUP_STEPS", "0"))
            if _v92_warmup > 0 and step < _v92_warmup:
                _lr_curr = LR * (step + 1) / float(_v92_warmup)
            else:
                _eff_step = step - _v92_warmup
                _eff_total = max(STEPS - _v92_warmup, 1)
                _lr_curr = LR * 0.5 * (1.0 + _lrm.cos(_lrm.pi * _eff_step / _eff_total))
            opt.lr.assign(Tensor([_lr_curr], dtype=opt.lr.dtype).contiguous()).realize()
            if step % 25 == 0 or step < 5:
                print(f"[lr_decay] step {step}: lr={_lr_curr:.6e}", flush=True)

        t0 = time.perf_counter()
        calib_info = None
        per_breath_info = None
        v85_components = None
        # v52 Stage 1: per-breath supervision path (mutually exclusive with calibration)
        from mycelium.l3_training import per_breath_train_step, v85_train_step
        from mycelium.breathing import PER_BREATH_DECODE as _PBD
        from mycelium.breathing import V85_QUERYABLE as _V85_Q_LIVE
        from mycelium.breathing import V85_K_MAX as _V85_K_MAX
        from mycelium.breathing import V85_N_MAX as _V85_N_MAX
        _per_breath_waist_norm = None
        if _V85_Q_LIVE and not CALIBRATION_MODE:
            # v85 path — structured slot supervision (replaces per_breath_train_step).
            K_v85 = int(getenv("TRAIN_LOOPS", "5").split(",")[0])
            loss, v85_components, per_breath_ce = v85_train_step(
                model, opt, batch_examples, tok,
                fixed_len=cur_fixed_len, K=K_v85,
                K_max=int(_V85_K_MAX), N_max=int(_V85_N_MAX),
                grad_clip=GRAD_CLIP, step_idx=step)
            per_breath_info = per_breath_ce
        elif _PBD and not CALIBRATION_MODE:
            from mycelium.l3_training import per_breath_train_step
            _pbs_result = per_breath_train_step(model, opt, batch_examples, tok,
                                                fixed_len=cur_fixed_len,
                                                lookup_aux_weight=LOOKUP_AUX_WEIGHT,
                                                lookup_eq_token_id=eq_token_ids,
                                                use_jit=USE_JIT,
                                                grad_clip=GRAD_CLIP,
                                                step_idx=step)
            # v66: per_breath_train_step always returns (loss, per_breath_ce, waist_norm)
            loss, per_breath_ce, _per_breath_waist_norm = _pbs_result
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
                # v66 waist norm (computed every step inside JIT; logged every 10 steps)
                wn_str = f"  waist_norm={_per_breath_waist_norm:.4f}" if _per_breath_waist_norm is not None else ""
                # v85 component CEs (ops / types / args / active / v89 attn_aux)
                if v85_components is not None:
                    wn_str += (f"  v85: ops={v85_components['ops_ce']:.3f}"
                               f" types={v85_components['types_ce']:.3f}"
                               f" args={v85_components['args_ce']:.3f}"
                               f" act={v85_components['active_ce']:.3f}")
                    if "attn_aux_ce" in v85_components:
                        wn_str += f" attn_aux={v85_components['attn_aux_ce']:.3f}"
                # v69 codebook match-weights entropy (every 100 steps; diagnostic for soft-VQ health)
                ent_str = ""
                try:
                    from mycelium.breathing import COLLAPSE_V69 as _CV69
                    if _CV69 and step % 100 == 0:
                        ent_tensor = getattr(model.block, "_collapse_last_match_entropy", None)
                        if ent_tensor is not None:
                            ent_val = float(ent_tensor.numpy())
                            import math
                            # max possible entropy for uniform softmax over N entries
                            max_ent = math.log(model.block.collapse_codebook_keys.shape[0])
                            ent_str = f"  cb_ent={ent_val:.3f}/{max_ent:.3f}"
                except Exception:
                    pass
                # v70 codebook entropy + gate importance mean
                try:
                    from mycelium.breathing import COLLAPSE_V70 as _CV70
                    if _CV70 and step % 100 == 0:
                        ent_tensor70 = getattr(model.block, "_collapse_v70_last_match_entropy", None)
                        imp_tensor70 = getattr(model.block, "_collapse_v70_last_importance_mean", None)
                        if ent_tensor70 is not None:
                            ent_val70 = float(ent_tensor70.numpy())
                            import math
                            max_ent70 = math.log(model.block.collapse_v70_codebook_keys.shape[0])
                            ent_str += f"  cb70={ent_val70:.3f}/{max_ent70:.3f}"
                        if imp_tensor70 is not None:
                            imp_val70 = float(imp_tensor70.numpy())
                            ent_str += f"  gate={imp_val70:.3f}"
                except Exception:
                    pass
                # v71 codebook entropy + gate importance mean (parallel to v70 logging)
                try:
                    from mycelium.breathing import COLLAPSE_V71 as _CV71
                    if _CV71 and step % 100 == 0:
                        ent_tensor71 = getattr(model.block, "_collapse_v71_last_match_entropy", None)
                        imp_tensor71 = getattr(model.block, "_collapse_v71_last_importance_mean", None)
                        if ent_tensor71 is not None:
                            ent_val71 = float(ent_tensor71.numpy())
                            import math
                            max_ent71 = math.log(model.block.collapse_v71_codebook_keys.shape[0])
                            ent_str += f"  cb71={ent_val71:.3f}/{max_ent71:.3f}"
                        if imp_tensor71 is not None:
                            imp_val71 = float(imp_tensor71.numpy())
                            ent_str += f"  gate71={imp_val71:.3f}"
                except Exception:
                    pass
                print(f"step {step:4d}  K={len(per_breath_info)}  loss={loss:.4f}{pb_str}{wn_str}{ent_str}{ctrl_str}  ({dt:.2f}s, total {elapsed:.0f}s)", flush=True)
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
