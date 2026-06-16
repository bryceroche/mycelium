"""KenKen v1 training driver (cold-start, residual-stream v98 paradigm).

Iterative-prefill breathing transformer on variable-N KenKen (N in {5,6,7}),
flattened to a FIXED N_max=7 grid (49 cells). Standalone driver — mirrors the
STRUCTURE and env/CLI conventions of scripts/sudoku_train.py, but adopts the
v200 where()-gated grad-guard (NOT sudoku's multiply-gate) in the JIT step.

Reuses:
  - BreathingTransformer's L0-L3 attention/FFN weights (Pythia-410M init)
  - mycelium/kenken.py    (forward / per-breath weighted CE / calib / accuracy /
                           convergence_instrument / attach_kenken_params)
  - mycelium/kenken_data.py (variable-N loader → N_max=7/49 cells, n_cages_max
                             pinned from corpus; per-cage op/target/size tensors)

THE JIT STEP grad-guard (PORT #1 / #2):
  After backward, NaN skip via where()-SELECT (not multiply):
      healthy_b = total.isfinite()                 # scalar, single kernel
      for p in jit_params:
          if p.grad is not None:
              p.grad = healthy_b.where(p.grad, Tensor.zeros_like(p.grad))
  NEVER `p.grad = p.grad * healthy` (NaN*0=NaN poisons Adam moments forever).
  NEVER a per-param isnan().any() loop (AM-driver segfault).

JIT cache key (PORT #3) includes EVERY shape/runtime-determining constant,
including the NEW one not present in sudoku: n_cages_max (=27/28; the per-cage
tensors cage_op/target/size are (B, n_cages_max)). A stale graph reading a
wrong-shaped cage buffer is silent corruption.

Graph stability (PORT #4 / #5): train AND eval loaders are constructed with the
SAME n_cages_max (max over BOTH corpora) so train-step and eval-step JIT graph
topology is identical. build_kenken_attn_bias + build_verification_inlet are
pure tensor ops and stay INSIDE the traced step (they re-trace with inputs).
convergence_instrument + per_breath_ce are eval-only (they .realize()/.numpy())
and stay OUT of the JIT step.

No fp32-carry (PORT #6): the inter-breath activation carry stays fp16 (the
validated v98 recipe; kenken.py already carries x in fp16, weights cast fp32).

Env vars (mirror sudoku_train.py):
  KENKEN_TASK=1
  KENKEN_K_MAX=20               number of iterative-prefill breaths
  KENKEN_CONSTRAINT_WEIGHT=0.3
  KENKEN_CALIB_WEIGHT=0.1
  BATCH=8
  STEPS=2000
  LR=3e-5
  CKPT_EVERY=200
  EVAL_EVERY=100
  LOG_EVERY=10
  PER_BREATH_CE_EVERY=50
  GC_EVERY=50
  CKPT_LABEL / RUN_NAME=kenken_v1
  RESUME_FROM=...               warm-start from a saved kenken ckpt (default: COLD)
  PYTHIA_INIT=1                 use Pythia weights for L0-L3 (default 1)
  SEED=42
  KENKEN_TRAIN=.cache/kenken_train.jsonl
  KENKEN_TEST=.cache/kenken_test.jsonl
  EVAL_BATCHES=20
  EVAL_BATCH=<BATCH>
  GRAD_CLIP=0.0

RUN-2 ANTI-OVERFIT REG STACK (the v45 stack; run-1 overfit train 0.90/test 0.51).
Env-gated; all default-ON for run-2, default values chosen to match v45:
  LABEL_SMOOTHING=0.1   per-breath CE label smoothing, MASKED to the legal value
                        domain (naive smoothing would spread mass onto the -1e4
                        illegal-value classes and inject ~143 of spurious loss per
                        cell — the value-domain-bias interaction; we smooth ONLY
                        over legal values 1..N, illegal classes get target 0 and
                        are masked out of the per-class term so -1e4 logp never
                        leaks). At ls=0 this is exactly the original NLL ladder.
  WEIGHT_DECAY=0.05     AdamW decoupled weight decay (was hardcoded 0.0).
  STOCH_DEPTH_P=0.1     per-BREATH stochastic depth: during TRAINING only, each
                        breath's residual update (gate_k * delta) is dropped with
                        prob p and the kept ones rescaled by 1/(1-p) (ResNet
                        unbiased-estimator form). This is the NATURAL granularity
                        for the K-breath shared-layer structure — it drops a whole
                        propagation step, mirroring the validated breathing.py
                        STOCH_DEPTH_P integral-drop idiom — NOT per-LAYER drop
                        (which would be awkward / too violent on a 4-layer breath,
                        the same reason the v300 perceiver trainer chose dropout
                        instead). At eval STOCH_DEPTH is OFF (full deterministic
                        K-breath forward via kenken_breathing_forward).
"""
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import safe_save, safe_load

from mycelium import Config, BreathingTransformer
from mycelium.loader import _load_state, load_breathing
from mycelium.kenken import (
    KENKEN_K_MAX, KENKEN_CONSTRAINT_WEIGHT, KENKEN_CALIB_WEIGHT,
    KENKEN_PI_ROPE, KENKEN_PI_ROPE_PHASE_SCALE,
    KENKEN_VALUE_REWEIGHT, KENKEN_VALUE_REWEIGHT_POW, KENKEN_VALUE_BIAS,
    KENKEN_CODEBOOK_ORTHO, KENKEN_CODEBOOK_ORTHO_STATE,
    KENKEN_MASK_GIVENS_P,
    KENKEN_HYP_MASK, KENKEN_HYP_RELAX, KENKEN_HYP_JITTER, KENKEN_HYP_MAX_ZNORM,
    KENKEN_HYP_GPHI, KENKEN_HYP_EUCLID,
    KENKEN_HYP_GPHI_DPOS, KENKEN_HYP_GPHI_WIDTH, KENKEN_HYP_GPHI_LAYERS,
    attach_kenken_params, attach_value_freq_weights, kenken_parameters,
    kenken_state_dict, kenken_breathing_forward, kenken_loss, per_breath_ce,
    kenken_accuracy, convergence_instrument, kenken_constraint_energy,
    codebook_ortho_penalty,
    kenken_hyp_coord_parameters, kenken_hyp_gphi_parameters, clamp_hyp_tangent_norms,
)
from mycelium.kenken_data import KenKenLoader, N_CELLS, N_MAX, load_jsonl

# ---- BACKBONE SELECTOR (KENKEN_BACKBONE=pythia(default)/llama) ----------------
# pythia => the existing run-2 path, BYTE-IDENTICAL (none of the llama imports or
# branches below execute). llama => 2 shared SmolLM2-1.7B (2048d) layers + 512d waist
# (mycelium/kenken_llama.py). The forward, param collection, state dict, fp32 cast,
# and provenance all branch on this single string; the v98 MECHANISM (masks, K-breath
# loop, delta_gate, codebook readout, calib, ladder, convergence instrument) is shared.
KENKEN_BACKBONE = getenv("KENKEN_BACKBONE", "pythia").strip().lower()
assert KENKEN_BACKBONE in ("pythia", "llama"), \
    f"KENKEN_BACKBONE must be pythia|llama, got {KENKEN_BACKBONE!r}"


# ---- fp32 cast of the transformer weights (mirror sudoku_train.cast_layers_fp32) ----

def cast_layers_fp32(model):
    """Cast L0-L3 layer + shared weights from fp16 to fp32 for stable training.

    NOTE: only the WEIGHTS are cast to fp32. The inter-breath activation carry
    (x) stays fp16 — kenken.py casts x to half in the forward (PORT #6: no
    fp32-carry; keep the validated v98 recipe).
    """
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


def cast_llama_layers_fp32(model):
    """Cast the SHARED Llama-2048 layer weights fp16 -> fp32 for stable training.

    The Llama backbone analogue of cast_layers_fp32. LlamaLayer weights are already
    allocated fp32 by attach_llama_layers (loader casts the safetensors to float), so
    this is a no-op guard for any fp16 weight that slipped through — the inter-breath
    activation carry (x) still stays fp16 (PORT #6: no fp32-carry; v98 recipe).
    """
    def _cast(obj, attr):
        t = getattr(obj, attr)
        if t.dtype == dtypes.half:
            setattr(obj, attr, t.cast(dtypes.float).contiguous().realize())
    for layer in model.llama_layers:
        for a in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down",
                  "attn_norm", "ffn_norm"):
            _cast(layer, a)


def collect_kenken_params(model) -> list[Tensor]:
    """Trainable parameters for KenKen training: shared L0-L3 attn/FFN + final
    LN + KenKen-specific embeddings/codebook/calib/verification-inlet.

    Skips token embedding, embed_out, lookup_table, controllers, notebooks —
    none of those are touched by the kenken forward.
    """
    params: list[Tensor] = []
    sw = model.block.shared
    params += [sw.wv, sw.bv, sw.wo, sw.bo, sw.w_out, sw.b_out,
               sw.in_ln_g, sw.in_ln_b, sw.post_ln_g, sw.post_ln_b]
    for layer in model.block.layers:
        params += [layer.wq, layer.bq, layer.wk, layer.bk, layer.w_in, layer.b_in]
    params += [model.ln_f_g, model.ln_f_b]
    params += kenken_parameters(model)
    return params


def collect_kenken_llama_params(model) -> list[Tensor]:
    """Trainable params for the Llama backbone: SHARED Llama-2048 layer weights +
    KenKen-specific heads (at 2048d) + the 512d waist.

    Mirror of collect_kenken_params but over the Llama layer stack instead of the
    Pythia shared/phase weights. The v98 heads + waist come from kenken_llama_parameters.
    """
    from mycelium.kenken_llama import kenken_llama_parameters
    params: list[Tensor] = []
    for layer in model.llama_layers:
        params.extend(layer.parameters())
    params += kenken_llama_parameters(model)
    return params


def model_state_dict_kenken(model) -> dict:
    """Save only what KenKen needs: shared + L0-L3 phase weights + final LN +
    kenken params (incl. verification inlet). Excludes embed_out/lookup/etc.
    """
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
    sd.update(kenken_state_dict(model))
    return sd


def model_state_dict_kenken_llama(model) -> dict:
    """Save the Llama backbone: shared Llama-2048 layer weights + kenken heads (2048d)
    + the waist. Mirror of model_state_dict_kenken for the llama backbone."""
    from mycelium.kenken_llama import kenken_llama_state_dict
    sd = {}
    for li, layer in enumerate(model.llama_layers):
        for a in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down",
                  "attn_norm", "ffn_norm"):
            sd[f"llama_layer_{li}.{a}"] = getattr(layer, a)
    sd.update(kenken_llama_state_dict(model))
    return sd


def _backbone_state_dict(model) -> dict:
    """Dispatch the checkpoint state dict by backbone (pythia path unchanged)."""
    if KENKEN_BACKBONE == "llama":
        return model_state_dict_kenken_llama(model)
    return model_state_dict_kenken(model)


def _backbone_forward(model, batch, K, stoch_keep=None):
    """Dispatch the breath forward by backbone. pythia => kenken_breathing_forward
    (BYTE-IDENTICAL); llama => kenken_llama_forward. Both return the SAME
    (cell_logits_history, calib_history) shapes so the ladder/loss/eval are shared."""
    if KENKEN_BACKBONE == "llama":
        from mycelium.kenken_llama import kenken_llama_forward
        return kenken_llama_forward(model, batch, K=K, stoch_keep=stoch_keep)
    return kenken_breathing_forward(model, batch, K=K, stoch_keep=stoch_keep)


def load_ckpt(model, path: str):
    sd = safe_load(path)
    targets = _backbone_state_dict(model)
    missing = []
    for name, dst in targets.items():
        if name not in sd:
            missing.append(name)
            continue
        src = sd[name].to(dst.device).realize()
        if src.shape != dst.shape:
            try:
                src = src.reshape(dst.shape)
            except Exception:
                missing.append(f"{name} (shape mismatch)")
                continue
        if src.dtype != dst.dtype:
            src = src.cast(dst.dtype)
        dst.assign(src).realize()
    if missing:
        print(f"  ckpt missing {len(missing)} kenken keys (kept init): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")


# ---- JIT train step — v200 where()-gated grad-guard (PORT #1/#2/#3/#5) --------

_JIT_KENKEN_CACHE: dict = {}


def _compile_jit_kenken_step(model, opt, K: int, B: int, n_cages_max: int,
                             constraint_weight: float, calib_weight: float,
                             grad_clip: float = 0.0,
                             label_smoothing: float = 0.0,
                             stoch_depth_p: float = 0.0,
                             value_reweight: float = 0.0,
                             value_freq_weight=None,
                             ortho_lambda: float = 0.0,
                             ortho_state: bool = False,
                             mask_givens_p: float = 0.0,
                             coord_grad_log: bool = False,
                             gphi_grad_log: bool = False):
    """Compile and return a TinyJit'd train step for the kenken breathing forward.

    JIT cache key (PORT #3) includes EVERY shape/runtime-determining constant:
      K, B, n_cages_max (NEW — the per-cage tensors cage_op/target/size are
      (B, n_cages_max)), constraint_weight, calib_weight, grad_clip, the run-2 reg
      knobs label_smoothing + stoch_depth_p (both change the traced graph body: ls
      switches the CE formula; stoch_depth_p > 0 adds the per-breath keep-mask
      input + the rescaled delta-blend), AND the two NEW mechanism knobs:
        ortho_lambda    — MECHANISM 1: > 0 appends the codebook-orthogonality
                          penalty term to total (graph body changes).
        ortho_state     — whether the state_embed value-rows are also penalized
                          (changes which codebooks enter the ortho term).
        mask_givens_p   — MECHANISM 2: > 0 activates the masked-givens self-
                          supervision branch (consumes the given_mask JIT input,
                          re-derives input_cells + the supervise set).
      ortho_lambda==0 AND mask_givens_p==0 => the two branches are skipped and the
      graph body is byte-identical to the current v98/K=8 trainer.

    Inputs (stable shapes; pass realized Tensors via a KenKenBatch):
      input_cells (B,49) int, gold (B,49) int, cell_valid (B,49) f,
      cage_mask (B,49,49) f, cell_cage_id (B,49) int, value_domain_mask
      (B,49,N_MAX) f, cage_op/cage_target/cage_size (B, n_cages_max) int,
      stoch_keep (K,) f, given_mask (B,49) f.

    given_mask is ALWAYS a JIT input (stable signature/shape) but is only READ when
    mask_givens_p>0; when off the caller passes all-zeros and the masked-givens
    branch is never entered, so the traced graph ignores it (byte-identical).

    Returns (each a realized scalar Tensor):
      total, healthy, cell_ce, energy, calib, cell_acc, puzzle_acc, ortho,
      *per_breath_ce[K], *per_breath_calib[K].

    build_kenken_attn_bias + build_verification_inlet are pure tensor ops and
    live INSIDE kenken_breathing_forward, so they re-trace with the per-batch
    inputs (PORT #5). No .numpy()/.realize() leaks into the traced body.
    """
    key = (id(model), id(opt), int(K), int(B), int(n_cages_max),
           float(constraint_weight), float(calib_weight), float(grad_clip),
           float(label_smoothing), float(stoch_depth_p), float(value_reweight),
           float(ortho_lambda), bool(ortho_state), float(mask_givens_p),
           bool(coord_grad_log), bool(gphi_grad_log))
    if key in _JIT_KENKEN_CACHE:
        return _JIT_KENKEN_CACHE[key]

    cw = float(constraint_weight)
    aw = float(calib_weight)
    gc_val = float(grad_clip)
    ls = float(label_smoothing)
    sd_p = float(stoch_depth_p)
    use_stoch = sd_p > 0.0
    vrw = float(value_reweight)
    use_reweight = vrw > 0.0 and value_freq_weight is not None
    # MECHANISM 1 / 2 compile-time switches. Both False => the two branches below
    # are dead and the traced graph body is byte-identical to current v98/K=8.
    olam = float(ortho_lambda)
    use_ortho = olam > 0.0
    use_mask_givens = float(mask_givens_p) > 0.0
    jit_params = opt.params
    # STAGE-1 RELAXATION coord grad-norm logging (the boundary-explosion watch, spec
    # §7/§8.2). When ON, after the NaN guard (pre-clip) the per-relation row/col/cage
    # coord grad norms + overall are appended to the JIT return. OFF (frozen / non-relax
    # path) => not computed, return signature byte-identical to the current trainer.
    use_coord_grad_log = bool(coord_grad_log)
    coord_grad_tensors = []
    if use_coord_grad_log:
        coord_grad_tensors = [
            ("row", model.kenken_hyp_v_row),
            ("col", model.kenken_hyp_v_col),
            ("cage", model.kenken_hyp_cage_anchors),
        ]
    # STAGE-2 g_phi grad-norm watch (spec §8.4 / ask #4): the AGGREGATE L2 grad norm over
    # ALL g_phi tensors (pos_emb + phi + rho), appended AFTER the coord (row/col/cage/
    # overall) tail when ON. OFF (Stage-1 / no g_phi) => not computed, signature unchanged.
    use_gphi_grad_log = bool(gphi_grad_log)
    gphi_grad_tensors = kenken_hyp_gphi_parameters(model) if use_gphi_grad_log else []

    print(f"[JIT] compile kenken step: K={K} B={B} n_cages_max={n_cages_max} "
          f"cw={cw} aw={aw} clip={gc_val} ls={ls} stoch_depth_p={sd_p} "
          f"value_reweight={vrw} ortho_lambda={olam} ortho_state={ortho_state} "
          f"mask_givens_p={mask_givens_p}...", flush=True)

    @TinyJit
    def _step(input_cells: Tensor, gold: Tensor, cell_valid: Tensor,
              cage_mask: Tensor, cell_cage_id: Tensor, value_domain_mask: Tensor,
              cage_op: Tensor, cage_target: Tensor, cage_size: Tensor,
              stoch_keep: Tensor, given_mask: Tensor):
        opt.zero_grad()

        # ---- MECHANISM 2: masked-given self-supervision (TRAIN only). When ON,
        # given_mask[b,i]=1.0 marks a GIVEN cell HIDDEN this step: its input token
        # is set to 0 (=unknown) so the embedding/forward never see its value, and
        # it joins the supervised-solve set below (the loss section re-derives
        # `observed` from input_cells, so masking here automatically supervises the
        # hidden givens). value_domain_mask is UNTOUCHED — the legal-value domain of
        # a masked given stays intact for the readout. OFF => input_cells unchanged
        # (byte-identical). given_mask is consumed ONLY in this branch. ----
        if use_mask_givens:
            keep_input = (1.0 - given_mask).cast(input_cells.dtype)        # (B,49)
            input_cells = input_cells * keep_input                          # hide -> token 0

        # Lightweight batch shim so kenken_breathing_forward / kenken_loss read
        # the per-instance tensors by attribute. All tensor attrs are the JIT
        # inputs (re-traced each replay); deduction_depth etc. are eval-only and
        # are NOT referenced inside the traced training body.
        class _B:
            pass
        batch = _B()
        batch.input_cells = input_cells
        batch.gold = gold
        batch.cell_valid = cell_valid
        batch.cage_mask = cage_mask
        batch.cell_cage_id = cell_cage_id
        batch.value_domain_mask = value_domain_mask
        batch.cage_op = cage_op
        batch.cage_target = cage_target
        batch.cage_size = cage_size

        # Forward: K constant → loop unrolls, graph topology fully static.
        # build_kenken_attn_bias + build_verification_inlet run INSIDE here.
        # Stochastic depth (run-2): pass the per-breath keep-scale Tensor when
        # STOCH_DEPTH_P>0; None (the compile-time branch) when off so the graph
        # is byte-identical to the pre-run-2 path. stoch_keep is a JIT INPUT so
        # the per-step random drop pattern can vary without recompiling.
        cell_logits_history, calib_history = _backbone_forward(
            model, batch, K=K, stoch_keep=(stoch_keep if use_stoch else None))

        # ---- Per-breath weighted CE ladder (the training loss; mirrors
        # kenken_loss but inlined to expose per-breath scalars in the JIT
        # return). Supervise VALID & UNOBSERVED cells, value-domain masked. ----
        # Run-2: MASKED label smoothing. The logits carry a -1e4 bias on illegal
        # (out-of-domain) values, so naive sparse_categorical_crossentropy(
        # label_smoothing>0) would spread smoothing mass onto the -1e4 classes
        # and inject ~alpha*1e4/N_MAX of spurious loss. We smooth ONLY over the
        # LEGAL value domain: target = (1-ls)*onehot(gold) + ls*(vdm/n_legal),
        # and mask the per-class (target*logp) term by vdm so the illegal -1e4
        # logp can never leak. ls=0 reduces EXACTLY to the original NLL ladder.
        observed = (input_cells > 0).cast(dtypes.float)                     # (B,49)
        supervise = cell_valid * (1.0 - observed)                          # (B,49)

        # ---- VALUE REWEIGHT (rare-value undertraining fix). When ON, scale each
        # supervised cell's CE by w(gold_value) (inverse-freq**pow), NORMALIZED to
        # mean 1.0 over the supervised cells (overall loss scale / LR coupling
        # unchanged). Folded into `supervise` so BOTH CE branches (ls>0 / ls==0)
        # pick it up unchanged. OFF => weight 1.0 everywhere => byte-identical CE. ----
        if use_reweight:
            gold_idx_rw = (gold - 1).clip(0, N_MAX - 1)                     # (B,49)
            gold_oh_rw = gold_idx_rw.one_hot(N_MAX).cast(dtypes.float)      # (B,49,N_MAX)
            vfw = value_freq_weight.cast(dtypes.float)                      # (N_MAX,)
            vw = gold_oh_rw @ vfw                                          # (B,49) raw w(gold)
            vw = vw * supervise                                            # zero non-supervised
            vw_mean = vw.sum() / (supervise.sum() + 1e-6)                  # mean over supervised
            cell_weight = vw / (vw_mean + 1e-12)                          # mean-1.0 reweight
            supervise = supervise * cell_weight                           # (B,49) reweighted mask
        sup_sum = supervise.sum() + 1e-6

        cell_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        weight_sum = 0.0
        per_breath_ce_losses = []
        if ls > 0.0:
            gold_idx2d = (gold - 1).clip(0, N_MAX - 1)                      # (B,49)
            gold_oh = gold_idx2d.one_hot(N_MAX).cast(dtypes.float)          # (B,49,N_MAX)
            vdm = value_domain_mask.cast(dtypes.float)                      # (B,49,N_MAX)
            n_legal = vdm.sum(axis=-1, keepdim=True) + 1e-6                 # (B,49,1)
            smooth_target = gold_oh * (1.0 - ls) + (vdm / n_legal) * ls     # (B,49,N_MAX)
            for k, logits in enumerate(cell_logits_history):
                weight_k = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
                logp = logits.cast(dtypes.float).log_softmax(axis=-1)       # (B,49,N_MAX)
                ce_per_cell = -(smooth_target * logp * vdm).sum(axis=-1)    # (B,49)
                ce_k = (ce_per_cell * supervise).sum() / sup_sum
                per_breath_ce_losses.append(ce_k)
                cell_loss_sum = cell_loss_sum + ce_k * weight_k
                weight_sum += weight_k
        else:
            gold_idx = (gold - 1).clip(0, N_MAX - 1).reshape(B * N_CELLS)   # (B*49,)
            supervise_flat = supervise.reshape(B * N_CELLS)
            for k, logits in enumerate(cell_logits_history):
                weight_k = 1.0 + float(k) / float(K - 1) if K > 1 else 1.0
                ce_elems = logits.reshape(B * N_CELLS, N_MAX).sparse_categorical_crossentropy(
                    gold_idx, reduction="none")                            # (B*49,)
                ce_k = (ce_elems * supervise_flat).sum() / sup_sum
                per_breath_ce_losses.append(ce_k)
                cell_loss_sum = cell_loss_sum + ce_k * weight_k
                weight_sum += weight_k
        cell_loss = cell_loss_sum / float(weight_sum)

        # Constraint energy on the final breath (row/col soft AllDiff).
        final_probs = cell_logits_history[-1].softmax(axis=-1)
        energy = kenken_constraint_energy(final_probs, batch).mean()

        # Calibration with detached argmax-correctness target (valid cells).
        final_argmax = (cell_logits_history[-1].argmax(axis=-1) + 1).detach()  # (B,49)
        eq = (final_argmax == gold).cast(dtypes.float)                      # (B,49)
        eq_valid = eq * cell_valid + (1.0 - cell_valid)                     # pad cells = match
        correct = eq_valid.prod(axis=-1)                                    # (B,) 0/1
        calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
        per_breath_calib_means = []
        for k, calib_k in enumerate(calib_history):
            progression = float(k) / float(K - 1) if K > 1 else 1.0
            target_k = 0.5 + (correct - 0.5) * progression
            calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
            per_breath_calib_means.append(calib_k.mean())
        calib_loss = calib_loss_sum / float(K)

        # Train accuracy (detached) over VALID cells.
        eq_v = eq * cell_valid                                              # (B,49)
        n_valid = cell_valid.sum() + 1e-6
        train_cell_acc = (eq_v.sum() / n_valid).detach()
        train_puzzle_acc = eq_valid.prod(axis=-1).mean().detach()

        total = cell_loss + cw * energy + aw * calib_loss

        # ---- MECHANISM 1: codebook-orthogonality penalty. When ON, add
        # olam * sum over the selected codebooks of mean-square off-diagonal cosine
        # (codebook_ortho_penalty). This ROTATES collinear value-codebook rows apart
        # (decorrelates the N=7 6<->7 pair), which the scalar value_reweight/bias
        # cannot. Pure tensor ops, fully inside the JIT graph. OFF (olam==0) => no
        # term, `ortho` reports 0.0, total byte-identical. ----
        if use_ortho:
            ortho = codebook_ortho_penalty(model.kenken_value_codebook)
            if ortho_state:
                # state_embed value-rows are rows 1..N_MAX (row 0 = unknown token);
                # they are codebook-aligned at init, so they share the collinearity.
                ortho = ortho + codebook_ortho_penalty(
                    model.kenken_state_embed[1:N_MAX + 1])
            total = total + olam * ortho
        else:
            ortho = Tensor.zeros((), dtype=dtypes.float).contiguous()
        total.backward()

        # ---- NaN guard — v200 where()-gated SELECT (PORT #1/#2) ----
        # multiply-gating passes NaN through (NaN × 0 = NaN), poisoning Adam
        # moments for the rest of the run. where() SELECTS: cond False → exact 0
        # regardless of NaN in the grad. Single-kernel isfinite() per param,
        # NO per-param isnan() loop (AM-driver safe).
        healthy_b = total.isfinite()
        healthy = healthy_b.cast(dtypes.float)
        for p in jit_params:
            if p.grad is not None:
                p.grad = healthy_b.where(p.grad, Tensor.zeros_like(p.grad))

        # ---- STAGE-1 RELAXATION coord grad-norm watch (the boundary-explosion read,
        # spec §7/§8.2). Compute the per-relation row/col/cage L2 grad norm + overall,
        # POST-NaN-guard / PRE-CLIP, so the logged magnitude is the honest "what the
        # 1/(1-|z|^2) backward produced" (a healthy step shows bounded norms; an
        # exploding boundary gradient shows up here BEFORE clipping tames it). Single
        # sq-sum kernel per coord tensor (AM-safe, no isnan loop). OFF => not computed,
        # return signature unchanged. ----
        # sqrt-guard epsilon tiny (1e-24 -> 1e-12 floor) so the TRUE coord grad
        # magnitude shows; a 1e-12-style floor would mask the genuinely small grads
        # that a faithful (near-fully-masked) bias produces.
        coord_grad_norms = []
        if use_coord_grad_log:
            overall_sq = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for _nm, ct in coord_grad_tensors:
                g = ct.grad
                if g is not None:
                    gsq = g.cast(dtypes.float).square().sum()
                else:
                    gsq = Tensor.zeros((), dtype=dtypes.float).contiguous()
                coord_grad_norms.append((gsq + 1e-24).sqrt())
                overall_sq = overall_sq + gsq
            coord_grad_norms.append((overall_sq + 1e-24).sqrt())  # overall last

        # ---- STAGE-2 g_phi grad-norm watch (spec §8.4 / ask #4). The AGGREGATE L2 norm
        # over ALL g_phi tensors (pos_emb + phi + rho), POST-NaN-guard / PRE-CLIP. Single
        # scalar appended AFTER the coord tail. The DeepSets backward (segment-mean ->
        # phi/rho MLP -> the d_hyp / euclid distance) is the new graph here; finite +
        # bounded == the headline Stage-2 safety read. OFF => not computed. ----
        gphi_grad_norms = []
        if use_gphi_grad_log:
            gphi_sq = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for gt in gphi_grad_tensors:
                g = gt.grad
                if g is not None:
                    gphi_sq = gphi_sq + g.cast(dtypes.float).square().sum()
            gphi_grad_norms.append((gphi_sq + 1e-24).sqrt())

        # Optional global-norm gradient clipping (single sq_sum kernel; AMD-safe).
        if gc_val > 0:
            sq_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
            for p in jit_params:
                if p.grad is not None:
                    sq_sum = sq_sum + p.grad.cast(dtypes.float).square().sum()
            grad_norm = (sq_sum + 1e-12).sqrt()
            clip_coef = (Tensor(gc_val, dtype=dtypes.float) / (grad_norm + 1e-6)).minimum(
                Tensor(1.0, dtype=dtypes.float))
            for p in jit_params:
                if p.grad is not None:
                    p.grad = p.grad * clip_coef.cast(p.grad.dtype)

        opt.step()

        return (
            total.realize(),
            healthy.realize(),
            cell_loss.realize(),
            energy.realize(),
            calib_loss.realize(),
            train_cell_acc.realize(),
            train_puzzle_acc.realize(),
            ortho.realize(),
            *(ce.realize() for ce in per_breath_ce_losses),
            *(c.realize() for c in per_breath_calib_means),
            # STAGE-1 RELAXATION coord grad-norm tail (row, col, cage, overall) — only
            # present when coord_grad_log is ON; the caller knows to read it via the
            # same flag, so OFF leaves the return tuple byte-identical.
            *(gn.realize() for gn in coord_grad_norms),
            # STAGE-2 g_phi grad-norm tail (single aggregate scalar) — only present when
            # gphi_grad_log is ON, AFTER the coord tail. The caller reads it via the same
            # flag, so OFF leaves the return tuple identical to the Stage-1 trainer.
            *(gn.realize() for gn in gphi_grad_norms),
        )

    _JIT_KENKEN_CACHE[key] = _step
    print(f"[JIT] kenken step ready (cache size={len(_JIT_KENKEN_CACHE)}); "
          f"first call compiles (~60-90s)…", flush=True)
    return _step


# ---- evaluation (eager forward; per_breath_ce + convergence_instrument are
#      eval-only and stay OUT of the JIT step — PORT #5) -------------------------

def evaluate(model, loader: KenKenLoader, K: int, max_batches: int,
             property2_path: str, step: int) -> dict:
    """Run forward over up to max_batches eval batches. Returns aggregate
    cell_acc / puzzle_acc / per-breath CE (over the first batch). Also runs
    convergence_instrument over EVERY eval batch and writes per-puzzle Property-2
    records to property2_path (JSONL). Does NOT compute the correlation.
    """
    Tensor.training = False

    cell_eq_sum = 0.0
    n_cells = 0
    puzzle_eq_sum = 0
    n_puzzles = 0
    # per-N aggregation (KenKen has N in {5,6,7}, not difficulty bands).
    by_N: dict[int, dict] = {}
    # per-BAND aggregation (curriculum givens-density band; the Property-2 gate).
    # Backward-compatible: band-less legacy corpora yield a single 'all' group, so
    # existing runs are unaffected. settled = convergence_instrument status_min
    # == "settled" (the gold-free PRIMARY settle); correct = whole-puzzle exact
    # (== puzzle_ok); both are tracked so the analyzer can read a settled set from
    # the easy band.
    by_band: dict[str, dict] = {}
    # per-(band, N) cross-tab (cheap: same loop). band is the priority; this is the
    # secondary breakout requested.
    by_band_N: dict[tuple, dict] = {}

    def _agg(d: dict, key, correct_cells, nv, puzzle_ok):
        if key not in d:
            d[key] = {"cell_eq": 0.0, "n_cells": 0, "puzzle_eq": 0,
                      "n_puzzles": 0, "settled": 0}
        e = d[key]
        e["cell_eq"] += correct_cells
        e["n_cells"] += nv
        e["puzzle_eq"] += puzzle_ok
        e["n_puzzles"] += 1

    pb_ce_first = None
    p2_records = []

    n_batches = 0
    for batch in loader.iter_eval(batch_size=loader.batch_size):
        cell_logits_history, _ = _backbone_forward(model, batch, K=K)
        final_logits = cell_logits_history[-1]

        cell_valid_np = batch.cell_valid.realize().numpy()                  # (B,49)
        gold_np = batch.gold.realize().numpy().astype(np.int32)             # (B,49)
        pred_np = (final_logits.argmax(axis=-1) + 1).realize().numpy().astype(np.int32)
        eq_np = ((pred_np == gold_np).astype(np.float32) * cell_valid_np)   # (B,49)

        # Property-2 convergence records over THIS eval batch (eval-only). Computed
        # here (before the per-puzzle accumulation loop) so the per-band 'settled'
        # count can read status_min from the SAME records that get written out.
        recs = convergence_instrument(cell_logits_history, batch)

        Bn = int(cell_valid_np.shape[0])
        for b in range(Bn):
            valid = cell_valid_np[b] > 0.5
            nv = int(valid.sum())
            if nv == 0:
                continue
            correct_cells = float(eq_np[b].sum())
            cell_eq_sum += correct_cells
            n_cells += nv
            puzzle_ok = int(np.all(pred_np[b][valid] == gold_np[b][valid]))
            puzzle_eq_sum += puzzle_ok
            n_puzzles += 1
            Ni = int(batch.N[b])
            bandi = str(batch.band[b])
            settled_b = 1 if recs[b].get("status_min") == "settled" else 0
            _agg(by_N, Ni, correct_cells, nv, puzzle_ok)
            _agg(by_band, bandi, correct_cells, nv, puzzle_ok)
            _agg(by_band_N, (bandi, Ni), correct_cells, nv, puzzle_ok)
            by_band[bandi]["settled"] += settled_b
            by_band_N[(bandi, Ni)]["settled"] += settled_b

        # Per-breath CE on the FIRST eval batch only (eval-only; .realize()s).
        if pb_ce_first is None:
            pb_ce_first = per_breath_ce(cell_logits_history, batch)

        # Attach band (+ N/step/core) to every per-puzzle Property-2 record so the
        # analyzer (scripts/analyze_kenken_property2.py) can read a settled set from
        # the easy band. band defaults to 'all' for band-less legacy corpora.
        for b, rec in enumerate(recs):
            rec = dict(rec)
            rec["N"] = int(batch.N[b])
            rec["band"] = str(batch.band[b])
            rec["step"] = int(step)
            rec["core"] = "v98_residual"  # comparison tag for the gated v300 perceiver-core swap-the-core run
            p2_records.append(rec)

        n_batches += 1
        if n_batches >= max_batches:
            break

    # Write Property-2 raw per-puzzle records (no correlation computed here).
    os.makedirs(os.path.dirname(property2_path), exist_ok=True)
    with open(property2_path, "w") as f:
        for rec in p2_records:
            f.write(json.dumps(rec) + "\n")

    out = {
        "cell_acc": cell_eq_sum / max(n_cells, 1),
        "puzzle_acc": puzzle_eq_sum / max(n_puzzles, 1),
        "n_puzzles": n_puzzles,
        "per_breath_ce": pb_ce_first or [],
        "by_N": {},
        "by_band": {},
        "by_band_N": {},
        "property2_n": len(p2_records),
        "property2_path": property2_path,
    }
    for Ni, v in by_N.items():
        out["by_N"][Ni] = {
            "cell_acc": v["cell_eq"] / max(v["n_cells"], 1),
            "puzzle_acc": v["puzzle_eq"] / max(v["n_puzzles"], 1),
            "n_puzzles": v["n_puzzles"],
        }
    for bandi, v in by_band.items():
        out["by_band"][bandi] = {
            "cell_acc": v["cell_eq"] / max(v["n_cells"], 1),
            "puzzle_acc": v["puzzle_eq"] / max(v["n_puzzles"], 1),
            "n_puzzles": v["n_puzzles"],
            # whole-puzzle exact == puzzle_eq (the 'correct' count); settled is the
            # gold-free PRIMARY status_min settle count (the analyzer's settled set).
            "correct": v["puzzle_eq"],
            "settled": v["settled"],
        }
    for (bandi, Ni), v in by_band_N.items():
        out["by_band_N"][f"{bandi}/N{Ni}"] = {
            "cell_acc": v["cell_eq"] / max(v["n_cells"], 1),
            "puzzle_acc": v["puzzle_eq"] / max(v["n_puzzles"], 1),
            "n_puzzles": v["n_puzzles"],
            "correct": v["puzzle_eq"],
            "settled": v["settled"],
        }
    Tensor.training = True
    return out


def _print_eval_table(res: dict, K: int, by_band_n: bool = False) -> None:
    """Print the OVERALL + per-band (+ optional per-band×N) eval table and the
    per-breath CE / Property-2 summary. Shared by the train-loop eval block and the
    eval-only path. The per-band table activates automatically when the corpus has
    a 'band' field (band-less corpora collapse to a single 'all' group)."""
    print(f"  test: cell_acc={res['cell_acc']:.3f} "
          f"puzzle_acc={res['puzzle_acc']:.3f} n={res['n_puzzles']}",
          flush=True)
    # ---- PER-BAND table (the Property-2 gate): n, cell_acc, puzzle_acc, settled,
    # correct (== whole-puzzle exact). Priority output. ----
    bands = res.get("by_band", {})
    if bands:
        # g40 (easy/shallow) .. g10 (hard); 'all' for legacy. Sort easy->hard.
        order = ["g40", "g30", "g20", "g10", "all"]
        band_keys = sorted(bands.keys(),
                           key=lambda b: order.index(b) if b in order else 99)
        print(f"  {'band':>5} {'n':>5} {'cell_acc':>9} {'puzzle_acc':>11} "
              f"{'settled':>8} {'correct':>8}", flush=True)
        for b in band_keys:
            v = bands[b]
            print(f"  {b:>5} {v['n_puzzles']:>5} {v['cell_acc']:>9.3f} "
                  f"{v['puzzle_acc']:>11.3f} {v['settled']:>8} {v['correct']:>8}",
                  flush=True)
    # ---- PER-N (existing; regression-safe) ----
    for Ni in sorted(res["by_N"].keys()):
        v = res["by_N"][Ni]
        print(f"    N={Ni}: cell_acc={v['cell_acc']:.3f} "
              f"puzzle_acc={v['puzzle_acc']:.3f} n={v['n_puzzles']}", flush=True)
    # ---- PER-(band, N) cross-tab (cheap secondary breakout; band is the priority) ----
    if by_band_n and res.get("by_band_N"):
        print("    per-band x N:", flush=True)
        for key in sorted(res["by_band_N"].keys()):
            v = res["by_band_N"][key]
            print(f"      {key:>8}: cell_acc={v['cell_acc']:.3f} "
                  f"puzzle_acc={v['puzzle_acc']:.3f} n={v['n_puzzles']} "
                  f"settled={v['settled']}", flush=True)
    pbe = res["per_breath_ce"]
    if pbe:
        if K <= 8:
            pbe_str = " ".join(f"{v:.2f}" for v in pbe)
        else:
            pbe_str = (" ".join(f"{v:.2f}" for v in pbe[:4]) + " ... "
                       + " ".join(f"{v:.2f}" for v in pbe[-4:]))
        print(f"    eval per_breath_ce[B0..B{K-1}]: {pbe_str}", flush=True)
    print(f"    Property-2: {res['property2_n']} per-puzzle records -> "
          f"{res['property2_path']}", flush=True)


def main():
    KENKEN_TASK_LOCAL = int(getenv("KENKEN_TASK", 0)) > 0
    assert KENKEN_TASK_LOCAL, "KENKEN_TASK=1 must be set"

    K = int(getenv("KENKEN_K_MAX", str(KENKEN_K_MAX)))
    BATCH = int(getenv("BATCH", 8))
    STEPS = int(getenv("STEPS", 2000))
    LR = float(getenv("LR", "3e-5"))
    CKPT_EVERY = int(getenv("CKPT_EVERY", 200))
    EVAL_EVERY = int(getenv("EVAL_EVERY", 100))
    LOG_EVERY = int(getenv("LOG_EVERY", 10))
    PER_BREATH_CE_EVERY = int(getenv("PER_BREATH_CE_EVERY", 50))
    GC_EVERY = int(getenv("GC_EVERY", 50))
    RUN_NAME = getenv("RUN_NAME", getenv("CKPT_LABEL", "kenken_v1"))
    # RESUME_FROM is the canonical name; KENKEN_RESUME is accepted as an alias.
    RESUME_FROM = getenv("RESUME_FROM", "") or getenv("KENKEN_RESUME", "")
    # EVAL-ONLY path: load the ckpt, run ONE eval over the test corpus (per-band
    # table + Property-2 records), print, exit. NO training, NO JIT train-step
    # compile, NO checkpoint overwrite. Activated by KENKEN_EVAL_ONLY=1 OR STEPS=0.
    EVAL_ONLY = int(getenv("KENKEN_EVAL_ONLY", 0)) > 0 or STEPS == 0
    PYTHIA_INIT = int(getenv("PYTHIA_INIT", 1)) > 0
    SEED = int(getenv("SEED", 42))
    GRAD_CLIP = float(getenv("GRAD_CLIP", "0.0"))

    KENKEN_TRAIN = getenv("KENKEN_TRAIN", ".cache/kenken_train.jsonl")
    KENKEN_TEST = getenv("KENKEN_TEST", ".cache/kenken_test.jsonl")
    EVAL_BATCHES = int(getenv("EVAL_BATCHES", 20))
    EVAL_BATCH = int(getenv("EVAL_BATCH", BATCH))

    CONSTRAINT_WEIGHT = float(getenv("KENKEN_CONSTRAINT_WEIGHT",
                                     str(KENKEN_CONSTRAINT_WEIGHT)))
    CALIB_WEIGHT = float(getenv("KENKEN_CALIB_WEIGHT", str(KENKEN_CALIB_WEIGHT)))

    # ---- RUN-2 anti-overfit reg stack (v45). Default-ON; env-overridable. ----
    LABEL_SMOOTHING = float(getenv("LABEL_SMOOTHING", "0.1"))
    WEIGHT_DECAY = float(getenv("WEIGHT_DECAY", "0.05"))
    STOCH_DEPTH_P = float(getenv("STOCH_DEPTH_P", "0.1"))

    # ---- TWO ORTHOGONAL MECHANISMS (both default OFF => byte-identical v98/K=8). ----
    # M1: codebook-orthogonality penalty (root-cause readout fix — rotates collinear
    #     value rows apart, the 6<->7 fix a bias/reweight cannot do).
    # M2: masked-given self-supervision (deep-chaining trainer — hides givens so the
    #     puzzle needs deeper deduction; hidden givens are predicted/supervised).
    CODEBOOK_ORTHO = float(getenv("KENKEN_CODEBOOK_ORTHO", str(KENKEN_CODEBOOK_ORTHO)))
    CODEBOOK_ORTHO_STATE = int(getenv(
        "KENKEN_CODEBOOK_ORTHO_STATE",
        "1" if (CODEBOOK_ORTHO > 0 and KENKEN_CODEBOOK_ORTHO_STATE) else "0")) > 0
    MASK_GIVENS_P = float(getenv("KENKEN_MASK_GIVENS_P", str(KENKEN_MASK_GIVENS_P)))

    # ---- STAGE-1 RELAXATION (spec §8.2/§8.6 Stage 1). KENKEN_HYP_RELAX=1 freezes the
    # v98 backbone and trains ONLY the ~3 hyperbolic coordinate tensors. Requires
    # KENKEN_HYP_MASK=1 (the generator + coords must be attached) AND a RESUME_FROM v98
    # ckpt (a frozen competent executor — the whole point of the attribution). Coord-only
    # optimizer with a small separate LR (KENKEN_HYP_LR) linearly warmed up from 0 over
    # KENKEN_HYP_WARMUP steps; coord-grad clip (KENKEN_HYP_GRAD_CLIP) + post-step tangent-
    # norm clamp are the live §7 boundary-gradient guards. OFF => none of this runs. ----
    HYP_RELAX = bool(KENKEN_HYP_RELAX)
    HYP_LR = float(getenv("KENKEN_HYP_LR", "1e-4"))
    HYP_WARMUP = int(getenv("KENKEN_HYP_WARMUP", "200"))
    HYP_GRAD_CLIP = float(getenv("KENKEN_HYP_GRAD_CLIP", "1.0"))
    HYP_MAX_ZNORM = float(getenv("KENKEN_HYP_MAX_ZNORM", str(KENKEN_HYP_MAX_ZNORM)))
    # ---- STAGE-2 (spec §8.1/§8.6 Stage 2): DeepSets g_phi structure encoder + the
    # capacity-matched Euclidean control. KENKEN_HYP_GPHI=1 attaches g_phi (the cage coord
    # becomes anchor[id] + g_phi(cell-set), zero-init -> t=0 == foothold mask). When BOTH
    # HYP_RELAX and HYP_GPHI are ON the g_phi tensors JOIN the coord-only optimizer (the
    # backbone stays frozen). KENKEN_HYP_EUCLID=1 (requires GPHI) is the Euclidean control
    # arm. OFF (default) => the Stage-1 slot-anchor path, byte-identical. ----
    HYP_GPHI = bool(KENKEN_HYP_GPHI)
    HYP_EUCLID = bool(KENKEN_HYP_EUCLID)

    print("=== KenKen v1 training (cold-start, v98 paradigm) ===")
    print(f"device={Device.DEFAULT}  B={BATCH}  K={K}  steps={STEPS}  lr={LR}")
    print(f"constraint_weight={CONSTRAINT_WEIGHT}  calib_weight={CALIB_WEIGHT}  "
          f"grad_clip={GRAD_CLIP}")
    print(f"REG STACK (v45): label_smoothing={LABEL_SMOOTHING}  "
          f"weight_decay={WEIGHT_DECAY}  stoch_depth_p={STOCH_DEPTH_P}")
    print(f"MECHANISMS: codebook_ortho={CODEBOOK_ORTHO} "
          f"(state_rows={CODEBOOK_ORTHO_STATE})  mask_givens_p={MASK_GIVENS_P}  "
          f"{'(both OFF => byte-identical v98/K=8)' if (CODEBOOK_ORTHO == 0 and MASK_GIVENS_P == 0) else ''}")
    if HYP_RELAX:
        assert KENKEN_HYP_MASK, \
            "KENKEN_HYP_RELAX=1 requires KENKEN_HYP_MASK=1 (the coords must be attached)"
        assert KENKEN_BACKBONE == "pythia", \
            "KENKEN_HYP_RELAX is Pythia-only (the hyperbolic mask generator lives in kenken.py)"
        assert RESUME_FROM, \
            "KENKEN_HYP_RELAX=1 requires RESUME_FROM=<v98 ckpt> (freeze a competent executor)"
        stage = "STAGE-2 (g_phi DeepSets encoder)" if HYP_GPHI else "STAGE-1 (slot-anchor)"
        print(f"{stage} RELAXATION: ON — FREEZE v98 backbone, train ONLY the "
              f"{'coords + g_phi encoder' if HYP_GPHI else '~3 hyperbolic coord'} tensors")
        print(f"  hyp: lr={HYP_LR} warmup={HYP_WARMUP} grad_clip={HYP_GRAD_CLIP} "
              f"jitter={KENKEN_HYP_JITTER} max_znorm={HYP_MAX_ZNORM} "
              f"alpha_margin(env)={os.environ.get('KENKEN_HYP_ALPHA_MARGIN','4.0')}")
    if HYP_GPHI:
        assert KENKEN_HYP_MASK, \
            "KENKEN_HYP_GPHI=1 requires KENKEN_HYP_MASK=1 (the cage path + g_phi attach)"
        assert KENKEN_BACKBONE == "pythia", \
            "KENKEN_HYP_GPHI is Pythia-only (the hyperbolic mask generator lives in kenken.py)"
        arm = "EUCLIDEAN control (||u-v||, no exp_0)" if HYP_EUCLID else "HYPERBOLIC (d_hyp)"
        print(f"  STAGE-2 g_phi: arm={arm}  d_pos={KENKEN_HYP_GPHI_DPOS} "
              f"width={KENKEN_HYP_GPHI_WIDTH} layers={KENKEN_HYP_GPHI_LAYERS}  "
              f"(cage_coord = anchor[id] + g_phi(cell-set), rho zero-init => t=0 foothold)")
    if HYP_EUCLID:
        assert HYP_GPHI, "KENKEN_HYP_EUCLID=1 requires KENKEN_HYP_GPHI=1 (same encoder)"
    print(f"train_path={KENKEN_TRAIN}  test_path={KENKEN_TEST}")
    print(f"warm-start={'COLD' if not RESUME_FROM else RESUME_FROM}")
    print()

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # ---- n_cages_max: PORT #4 — train AND eval loaders MUST share the SAME
    # n_cages_max so the train-step and eval-step JIT graph topology is identical.
    # KenKenLoader asserts n_cages_max >= its own corpus max; the test corpus may
    # have MORE cages than train (observed 28 vs 27), so pin to the max over BOTH.
    train_recs = load_jsonl(KENKEN_TRAIN)
    test_recs = load_jsonl(KENKEN_TEST)
    corpus_n_cages_max = max(
        max(len(r["cages"]) for r in train_recs),
        max(len(r["cages"]) for r in test_recs),
    )
    N_CAGES_MAX = int(getenv("KENKEN_N_CAGES_MAX", str(corpus_n_cages_max)))
    assert N_CAGES_MAX >= corpus_n_cages_max, (
        f"N_CAGES_MAX={N_CAGES_MAX} < corpus max {corpus_n_cages_max}")
    print(f"n_cages_max (train+test corpus max) = {N_CAGES_MAX}")

    # ---- model (backbone-selected). pythia => the run-2 path, BYTE-IDENTICAL.
    cfg = Config()
    if KENKEN_BACKBONE == "llama":
        # 2 shared SmolLM2-1.7B (2048d, 32-head) layers + 512d waist. The v98
        # MECHANISM is unchanged (masks/loop/gate/codebook/calib/ladder/instrument);
        # only the backbone is bigger/shallower + an IB waist (mycelium/kenken_llama.py).
        from mycelium.llama_loader import attach_llama_layers, SMOLLM2_1_7B_CFG
        from mycelium.kenken_llama import (
            attach_kenken_llama_params, KENKEN_N_LLAMA_LAYERS,
            KENKEN_WAIST_DIM, KENKEN_WAIST_MODE,
        )
        n_llama_layers = int(getenv("KENKEN_N_LLAMA_LAYERS", str(KENKEN_N_LLAMA_LAYERS)))
        waist_dim = int(getenv("KENKEN_WAIST_DIM", str(KENKEN_WAIST_DIM)))
        waist_mode = getenv("KENKEN_WAIST_MODE", KENKEN_WAIST_MODE).strip().lower()
        llama_cfg = SMOLLM2_1_7B_CFG
        print(f"loading SmolLM2-1.7B (2048d, {llama_cfg.num_attention_heads}-head) "
              f"-> {n_llama_layers} SHARED layers + {waist_dim}d waist "
              f"(mode={waist_mode})...")
        # A bare carrier object for the kenken heads / Llama layers — the Llama
        # backbone does NOT use the Pythia BreathingTransformer block at all.
        class _LlamaCarrier:
            pass
        model = _LlamaCarrier()
        attach_llama_layers(model, n_layers=n_llama_layers, cfg=llama_cfg)
        cast_llama_layers_fp32(model)
        attach_kenken_llama_params(
            model, hidden=llama_cfg.hidden_size,
            n_heads=llama_cfg.num_attention_heads, k_max=K,
            waist_dim=waist_dim, waist_mode=waist_mode)
        Device[Device.DEFAULT].synchronize()
        params = collect_kenken_llama_params(model)
    else:
        print(f"loading Pythia-410M -> breathing transformer (PYTHIA_INIT={PYTHIA_INIT})...")
        if PYTHIA_INIT:
            sd = _load_state()
            model = load_breathing(cfg, sd=sd)
            del sd
        else:
            model = BreathingTransformer(cfg)
        cast_layers_fp32(model)
        attach_kenken_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=K)
        # Value-frequency reweight table (rare-value undertraining fix), ONLY when
        # KENKEN_VALUE_REWEIGHT>0. Computed once from the train corpus solutions and
        # attached to the model (model.kenken_value_freq_weight). OFF => not attached,
        # forward/CE byte-identical to current v98.
        if KENKEN_VALUE_REWEIGHT > 0:
            attach_value_freq_weights(model, KENKEN_TRAIN,
                                      pow_=KENKEN_VALUE_REWEIGHT_POW)
            print(f"  value reweight ON (pow={KENKEN_VALUE_REWEIGHT_POW}); "
                  f"weights={np.array2string(model.kenken_value_freq_weight_np, precision=4)}",
                  flush=True)
        if KENKEN_VALUE_BIAS:
            print(f"  learnable per-value logit bias ON (zero-init)", flush=True)
        Device[Device.DEFAULT].synchronize()
        params = collect_kenken_params(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    print(f"  trainable params: {n_params/1e6:.1f}M")

    if RESUME_FROM:
        print(f"resuming from kenken ckpt: {RESUME_FROM}")
        load_ckpt(model, RESUME_FROM)
        print("  loaded.")

    # ---- STAGE-1 RELAXATION optimizer (spec §8.2): FREEZE the entire v98 backbone,
    # train ONLY the ~3 hyperbolic coordinate tensors. The backbone tensors are removed
    # from the optimizer AND have requires_grad=False (so backward never accumulates
    # into them — a clean freeze, not just a skipped step). The coords get requires_grad
    # =True. After this block, opt.params is EXACTLY {v_row, v_col, cage_anchors}. The
    # OFF path leaves opt = AdamW(params, ...) on the full backbone (byte-identical). ----
    backbone_freeze_probes = []
    if HYP_RELAX:
        coords = kenken_hyp_coord_parameters(model)                  # {row, col, cage}
        coord_params = [coords["row"], coords["col"], coords["cage"]]
        # STAGE-2: the g_phi encoder tensors (pos_emb + phi + rho) JOIN the coord-only
        # param group (spec §8.6 ask #3). Empty list when GPHI off (Stage-1 unchanged).
        gphi_params = kenken_hyp_gphi_parameters(model) if HYP_GPHI else []
        train_params = coord_params + gphi_params
        train_ids = {id(t) for t in train_params}
        # Freeze EVERYTHING the v98 forward touches (collect_kenken_params is the full
        # trainable set: shared L0-L3 attn/FFN + final LN + kenken heads/codebook/inlet).
        # The g_phi tensors are NOT in `params` (collect_kenken_params predates the
        # encoder), so they are never frozen here; they get requires_grad=True below.
        n_frozen = 0
        for t in params:
            if id(t) not in train_ids:
                t.requires_grad = False
                n_frozen += 1
        for t in train_params:
            t.requires_grad = True
        # The hyp coords default requires_grad=None at attach (frozen-foothold semantics);
        # make them + the g_phi tensors explicitly trainable (the ONLY trained tensors).
        opt = AdamW(train_params, lr=HYP_LR, weight_decay=0.0)
        # SANITY: the optimizer param set must be EXACTLY {coords (+ g_phi when GPHI)}.
        n_expected = len(train_params)
        assert len(opt.params) == n_expected, \
            f"coord+gphi opt must have {n_expected} params, got {len(opt.params)}"
        assert {id(p) for p in opt.params} == train_ids, \
            "opt params are not exactly the coord (+ g_phi) tensors"
        n_coord_scalars = sum(int(np.prod(t.shape)) for t in coord_params)
        n_gphi_scalars = sum(int(np.prod(t.shape)) for t in gphi_params)
        stage_tag = "STAGE-2" if HYP_GPHI else "STAGE-1"
        print(f"  [{stage_tag}] froze {n_frozen} backbone tensors; "
              f"optimizer trains EXACTLY {len(opt.params)} tensors "
              f"(coords {n_coord_scalars/1e3:.1f}K + g_phi {n_gphi_scalars/1e3:.1f}K scalars): "
              f"row{tuple(coord_params[0].shape)} col{tuple(coord_params[1].shape)} "
              f"cage{tuple(coord_params[2].shape)}"
              f"{' + g_phi[' + str(len(gphi_params)) + ' tensors]' if HYP_GPHI else ''}",
              flush=True)
        if HYP_GPHI:
            # ANCHOR-DISCIPLINE assert (spec §8.6): rho's OUTPUT layer must be zero-init so
            # g_phi == 0 at t=0 (=> cage_coord == anchor[id] => t=0 bias == foothold mask).
            rho_last_w = model.kenken_gphi_rho_w[-1]
            assert float(np.abs(rho_last_w.detach().numpy()).max()) == 0.0, \
                "g_phi rho OUTPUT layer is not zero-init (t=0 anchor discipline broken)"
            print(f"  [STAGE-2] g_phi rho output layer zero-init VERIFIED "
                  f"(g_phi==0 at t=0 => cage_coord==anchor[id])", flush=True)
        # Capture a couple backbone tensors for the freeze-verification probe (smoke):
        # value_codebook + a shared attn weight + the first phase wq. We snapshot their
        # numpy AFTER the freeze/load so the post-train compare proves they didn't move.
        backbone_freeze_probes = [
            ("kenken.value_codebook", model.kenken_value_codebook),
            ("shared.wq?-> phase0.wq", model.block.layers[0].wq),
            ("shared.wo", model.block.shared.wo),
        ]
    else:
        opt = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # ---- data (SAME n_cages_max passed to BOTH loaders — PORT #4)
    train_loader = KenKenLoader(KENKEN_TRAIN, batch_size=BATCH, seed=SEED,
                                n_cages_max=N_CAGES_MAX)
    test_loader = KenKenLoader(KENKEN_TEST, batch_size=EVAL_BATCH, seed=SEED + 1,
                               n_cages_max=N_CAGES_MAX)

    # ---- EVAL-ONLY path (KENKEN_EVAL_ONLY=1 or STEPS=0). Loads the resumed ckpt,
    # runs ONE eval over the test corpus (per-band table + Property-2 records),
    # prints, exits. NO JIT train-step compile, NO optimizer step, NO checkpoint
    # save/overwrite. The substrate laws are untouched (eval is the SAME eager
    # forward the training loop uses for its periodic eval). ----
    if EVAL_ONLY:
        run_dir = os.path.join(".cache/kenken_ckpts", RUN_NAME)
        os.makedirs(run_dir, exist_ok=True)
        Tensor.training = False
        p2_path = os.path.join(run_dir, "property2_eval_only.jsonl")
        print(f"\n=== EVAL-ONLY (no training; ckpt={RESUME_FROM or 'COLD'}) ===")
        print(f"  test corpus: {KENKEN_TEST}  K={K}  EVAL_BATCH={EVAL_BATCH}  "
              f"EVAL_BATCHES={EVAL_BATCHES}", flush=True)
        t_eval = time.time()
        res = evaluate(model, test_loader, K=K, max_batches=EVAL_BATCHES,
                       property2_path=p2_path, step=0)
        # band IS the priority; also break out per-band x N (cheap).
        _print_eval_table(res, K, by_band_n=True)
        print(f"  (eval-only done in {time.time() - t_eval:.1f}s; "
              f"NO ckpt written, NO training step run)", flush=True)
        return

    # ---- ckpt dir + provenance
    run_dir = os.path.join(".cache/kenken_ckpts", RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    # Backbone provenance (pythia => exactly the run-2 fields; llama adds the swap meta).
    if KENKEN_BACKBONE == "llama":
        from mycelium.kenken_llama import (
            KENKEN_N_LLAMA_LAYERS, KENKEN_WAIST_DIM, KENKEN_WAIST_MODE,
        )
        _n_llama = int(getenv("KENKEN_N_LLAMA_LAYERS", str(KENKEN_N_LLAMA_LAYERS)))
        _waist_dim = int(getenv("KENKEN_WAIST_DIM", str(KENKEN_WAIST_DIM)))
        _waist_mode = getenv("KENKEN_WAIST_MODE", KENKEN_WAIST_MODE).strip().lower()
        _base = "SmolLM2-1.7B"
        _backbone_meta = {
            "backbone": "llama",
            "n_llama_layers": _n_llama,
            "llama_hidden": 2048,
            "llama_n_heads": 32,
            "waist_dim": _waist_dim,
            "waist_mode": _waist_mode,
        }
    else:
        _base = "Pythia-410M"
        _backbone_meta = {
            "backbone": "pythia",
            "n_pythia_layers": 4,
            "pythia_hidden": cfg.hidden,
            "pythia_n_heads": cfg.n_heads,
        }
    measured_config = {
        "arch_version": "kenken_v1",
        "core": "v98_residual",  # comparison tag for the gated future v300 perceiver-core swap-the-core run
        **_backbone_meta,
        "K": K,
        "B": BATCH,
        "LR": LR,
        "steps": STEPS,
        "n_cages_max": N_CAGES_MAX,
        "seed": SEED,
        "warm_start": "none" if not RESUME_FROM else RESUME_FROM,
        "warm_start_mode": "cold" if not RESUME_FROM else "resume",
        "base": _base,
        "constraint_weight": CONSTRAINT_WEIGHT,
        "calib_weight": CALIB_WEIGHT,
        "grad_clip": GRAD_CLIP,
        # ---- v109pi per-breath Q-rotation (π-cycled), Pythia-backbone only ----
        # OFF by default => byte-identical to current v98 KenKen. ON => each breath k
        # rotates Q (Q-only, uniform across positions) by theta_k = scale·k·π/K_max.
        "pi_rope": {
            "enabled": bool(KENKEN_PI_ROPE),
            "phase_scale": KENKEN_PI_ROPE_PHASE_SCALE,
            "backbone_support": "pythia"
            if KENKEN_BACKBONE == "pythia"
            else "NOT-WIRED (llama delegates attn to LlamaLayer; pi_rope inert)",
            "mechanism": "per-breath Q-only pairwise head_dim rotation, theta_k=scale*k*pi/K_max (v109pi Option A)",
        },
        # ---- RUN-2 anti-overfit reg stack (v45) ----
        "reg_stack": {
            "label_smoothing": LABEL_SMOOTHING,
            "weight_decay": WEIGHT_DECAY,
            "stoch_depth_p": STOCH_DEPTH_P,
            "stoch_depth_granularity": "per-breath residual-update drop (ResNet unbiased, train-only)",
            "label_smoothing_form": "value-domain-masked (legal values 1..N only)",
        },
        # ---- VALUE READOUT-FIX (rare-value undertraining; diag_kenken_cell_difficulty) ----
        # OFF by default => byte-identical to current v98. value_reweight scales each
        # supervised cell's CE by inverse-freq**pow of its gold value (mean-1.0
        # normalized); value_bias is a learnable zero-init per-value logit offset.
        "value_readout_fix": {
            "value_reweight": KENKEN_VALUE_REWEIGHT,
            "value_reweight_pow": KENKEN_VALUE_REWEIGHT_POW,
            "value_bias": bool(KENKEN_VALUE_BIAS),
            "value_freq_weights": (
                model.kenken_value_freq_weight_np.tolist()
                if getattr(model, "kenken_value_freq_weight_np", None) is not None
                else None),
        },
        # ---- TWO ORTHOGONAL MECHANISMS. Both OFF (lambda==0 AND p==0) => the JIT
        # graph body + param set is byte-identical to the current v98/K=8 trainer. ----
        "mechanisms": {
            # M1: codebook-orthogonality penalty (root-cause readout fix). Adds
            # lambda * mean-square off-diagonal cosine of the value-codebook (and,
            # when ortho_state, the state_embed value-rows). ROTATES collinear value
            # rows apart (the 6<->7 fix a bias/reweight cannot perform). No new params.
            "codebook_ortho": {
                "lambda": CODEBOOK_ORTHO,
                "apply_state_rows": bool(CODEBOOK_ORTHO_STATE),
                "enabled": bool(CODEBOOK_ORTHO > 0),
                "form": "lambda * mean_{i!=j} cos(codebook_i, codebook_j)^2",
                "rationale": "diag: cos(v6,v7)=0.173 vs mean|cos|(v1-5)=0.025 (~7x); "
                             "scalar bias/reweight cannot rotate collinear rows apart",
            },
            # M2: masked-given self-supervision (deep-chaining trainer). TRAIN-only;
            # each given cell hidden w.p. p (input token -> 0 = unknown) AND added to
            # the supervised-solve set (predicted like any unobserved cell). EVAL off.
            "mask_givens": {
                "p": MASK_GIVENS_P,
                "enabled": bool(MASK_GIVENS_P > 0),
                "scope": "train-only; eval uses full givens (deterministic)",
                "form": "Bernoulli(p) over given cells; hidden givens removed from "
                        "input + added to CE supervise set; value_domain intact",
                "rationale": "deep-chaining ceiling (r1=0.74 -> r5+=0.37); fewer "
                             "givens => deeper deduction required; masks rare 6/7 too",
            },
        },
        "train_path": KENKEN_TRAIN,
        "test_path": KENKEN_TEST,
        "n_cells": N_CELLS,
        "n_max": N_MAX,
        "trainable_params_M": round(n_params / 1e6, 3),
        "ckpt_dir": run_dir,
        # ---- STAGE-1 RELAXATION (spec §8.2/§8.6 Stage 1). OFF => byte-identical to the
        # frozen-foothold / v98 trainer (no backbone freeze, full-param optimizer). ----
        "hyp_relax": {
            "enabled": bool(HYP_RELAX),
            "hyp_mask": bool(KENKEN_HYP_MASK),
            "lr": HYP_LR,
            "warmup": HYP_WARMUP,
            "grad_clip": HYP_GRAD_CLIP,
            "jitter": KENKEN_HYP_JITTER,
            "max_znorm": HYP_MAX_ZNORM,
            "alpha_margin": float(os.environ.get("KENKEN_HYP_ALPHA_MARGIN", "4.0")),
            "trains": "ONLY {kenken_hyp_v_row, kenken_hyp_v_col, kenken_hyp_cage_anchors}"
                      + (" + g_phi encoder (pos_emb+phi+rho)" if HYP_GPHI else "")
                      + "; v98 backbone FROZEN (requires_grad=False, not in optimizer)",
            "rowcol_in_graph": True,
            "guards": "coord grad clip + bounded tangent norms + where()-gated NaN guard",
        },
        # ---- STAGE-2 (spec §8.1/§8.6 Stage 2): DeepSets g_phi structure encoder + the
        # capacity-matched Euclidean control. OFF (gphi disabled) => the cage path is the
        # Stage-1 slot-anchor field, byte-identical. ----
        "hyp_stage2": {
            "gphi_enabled": bool(HYP_GPHI),
            "euclid_arm": bool(HYP_EUCLID),
            "d_pos": KENKEN_HYP_GPHI_DPOS,
            "width": KENKEN_HYP_GPHI_WIDTH,
            "layers": KENKEN_HYP_GPHI_LAYERS,
            "cage_coord": "anchor[id] + g_phi(cell-set)" if HYP_GPHI else "anchor[id] (slot)",
            "g_phi": "rho( MEAN_{cell in cage} phi(pos_emb[cell]) concat [|cage|/N_CELLS] )",
            "perm_invariance": "segment-mean over cell_cage_id (no cell ordering)",
            "pos_emb": "ENCODER-OWNED learned (49,d_pos) — NOT the frozen backbone's pos features",
            "zero_init": "rho OUTPUT layer zero-init => g_phi==0 at t=0 => t=0 bias == foothold mask",
            "mask_only_inputs": "cell positions + cage size ONLY (NEVER op-type/target)",
            "distance": "Euclidean ||u-v|| (drop exp_0)" if HYP_EUCLID else "Poincare d_hyp",
            "capacity_match": "Euclidean arm = SAME g_phi params/shapes; only distance+exp_0 differ",
        },
    }
    with open(os.path.join(run_dir, "measured_config.json"), "w") as f:
        json.dump(measured_config, f, indent=2)
    print(f"  wrote provenance: {os.path.join(run_dir, 'measured_config.json')}")

    # ---- JIT compile the train step (v200 grad-guard). First call inside the
    # loop blocks ~60-90s for compile, then ~1-2s/step steady state.
    Tensor.training = True
    # STAGE-1 RELAXATION: the coord-grad-clip is HYP_GRAD_CLIP (the boundary-gradient
    # guard); enable the coord grad-norm logging tail. OFF => GRAD_CLIP + no coord log
    # (byte-identical to the current trainer's JIT body).
    effective_grad_clip = HYP_GRAD_CLIP if HYP_RELAX else GRAD_CLIP
    step_fn = _compile_jit_kenken_step(
        model, opt, K=K, B=BATCH, n_cages_max=N_CAGES_MAX,
        constraint_weight=CONSTRAINT_WEIGHT, calib_weight=CALIB_WEIGHT,
        grad_clip=effective_grad_clip, label_smoothing=LABEL_SMOOTHING,
        stoch_depth_p=STOCH_DEPTH_P,
        value_reweight=KENKEN_VALUE_REWEIGHT,
        value_freq_weight=getattr(model, "kenken_value_freq_weight", None),
        ortho_lambda=CODEBOOK_ORTHO, ortho_state=CODEBOOK_ORTHO_STATE,
        mask_givens_p=MASK_GIVENS_P,
        coord_grad_log=HYP_RELAX,
        gphi_grad_log=(HYP_RELAX and HYP_GPHI),
    )

    # ---- train loop
    print("\ntraining...\n")
    t0 = time.time()
    # Deferred logging (perf fix #1): accumulate the log scalars ON-GPU — one .realize()
    # per step (enqueue, no host block) and a single .numpy() per LOG_EVERY — instead of a
    # GPU->CPU sync every step. Training math is untouched (it all lives inside step_fn's
    # JIT graph), so this is byte-exact for the weight trajectory; only the cadence of log
    # readback changes. A stale-ref is impossible here: .cat allocates a fresh buffer, so
    # log_acc never aliases a reused JIT output buffer. Worst case of any bug = wrong log
    # display (self-revealing at the first LOG_EVERY), never wrong training.
    log_acc = None  # (5,) running sum: [loss, cell_ce, energy, calib, healthy]
    log_n = 0

    # ---- STAGE-1 RELAXATION: snapshot the freeze-probe backbone tensors BEFORE any
    # step so the smoke can prove they are bit-identical post-train (the freeze proof),
    # and prime v_init for the optional drift dump. Also set up the base coord LR for
    # linear warmup. OFF => skipped (no overhead). ----
    freeze_probe_pre = {}
    coord_v_init = {}
    if HYP_RELAX:
        for nm, t in backbone_freeze_probes:
            freeze_probe_pre[nm] = t.detach().numpy().copy()
        coords_now = kenken_hyp_coord_parameters(model)
        for nm, t in coords_now.items():
            coord_v_init[nm] = t.detach().numpy().copy()
        print(f"  [STAGE-1] snapshotted {len(freeze_probe_pre)} backbone freeze-probes "
              f"+ {len(coord_v_init)} coord v_init tensors", flush=True)

    def _set_coord_lr(step_i: int):
        """Linear LR warmup from 0 -> HYP_LR over HYP_WARMUP steps (spec §8.2). Step is
        1-based; at step >= HYP_WARMUP the LR is HYP_LR. opt.lr is a Tensor -> assign."""
        if HYP_WARMUP > 0:
            frac = min(1.0, float(step_i) / float(HYP_WARMUP))
        else:
            frac = 1.0
        opt.lr.assign(Tensor(HYP_LR * frac, dtype=opt.lr.dtype)).realize()

    # Stochastic-depth keep-mask RNG (run-2). Drawn per step on the host, fed as a
    # JIT INPUT (K,) so the random drop pattern varies WITHOUT recompiling. Each
    # breath kept (1-p) of the time, kept ones rescaled by 1/(1-p) so E[update] is
    # unchanged (ResNet unbiased estimator). When STOCH_DEPTH_P==0 we pass all-ones
    # (the JIT branch ignores it; the graph is byte-identical to pre-run-2).
    sd_rng = np.random.RandomState(SEED + 99)
    ones_keep = Tensor(np.ones((K,), dtype=np.float32), dtype=dtypes.float).contiguous().realize()

    def _draw_stoch_keep():
        if STOCH_DEPTH_P <= 0.0:
            return ones_keep
        kept = (sd_rng.rand(K) >= STOCH_DEPTH_P).astype(np.float32)
        scale = kept / (1.0 - STOCH_DEPTH_P)
        return Tensor(scale.astype(np.float32), dtype=dtypes.float).contiguous().realize()

    # ---- MECHANISM 2: masked-given RNG (deep-chaining trainer). Per step, draw a
    # host-side Bernoulli(MASK_GIVENS_P) mask over the GIVEN positions of the batch
    # (a given cell ⇔ input_cells > 0). given_mask[b,i]=1.0 hides that given this
    # step. Fed as a JIT INPUT (B,49) so the random pattern varies WITHOUT
    # recompiling (same idiom as stoch_keep). When MASK_GIVENS_P<=0 we pass all-
    # zeros (the JIT branch ignores it; graph byte-identical to current v98). The
    # mask is restricted to given cells so unobserved/pad cells are never touched. ----
    mg_rng = np.random.RandomState(SEED + 137)
    zeros_given_mask = Tensor(
        np.zeros((BATCH, N_CELLS), dtype=np.float32), dtype=dtypes.float).contiguous().realize()

    def _draw_given_mask(batch):
        if MASK_GIVENS_P <= 0.0:
            return zeros_given_mask
        given_np = batch.input_cells.realize().numpy() > 0           # (B,49) given cells
        draw = mg_rng.rand(BATCH, N_CELLS) < MASK_GIVENS_P           # Bernoulli(p)
        gm = (given_np & draw).astype(np.float32)                    # hide givens only
        return Tensor(gm, dtype=dtypes.float).contiguous().realize()

    for step in range(1, STEPS + 1):
        batch = train_loader.sample_batch()

        # STAGE-1 RELAXATION: linear coord-LR warmup from 0 BEFORE the step (spec §8.2).
        if HYP_RELAX:
            _set_coord_lr(step)

        outs = step_fn(
            batch.input_cells, batch.gold, batch.cell_valid,
            batch.cage_mask, batch.cell_cage_id, batch.value_domain_mask,
            batch.cage_op, batch.cage_target, batch.cage_size,
            _draw_stoch_keep(), _draw_given_mask(batch),
        )
        total_t      = outs[0]
        healthy_t    = outs[1]
        cell_ce_t    = outs[2]
        energy_t     = outs[3]
        calib_t      = outs[4]
        cell_acc_t   = outs[5]
        puzzle_acc_t = outs[6]
        ortho_t      = outs[7]   # MECHANISM 1 scalar (0.0 when ortho OFF)
        pb_ce_ts     = outs[8:8 + K]
        pb_calib_ts  = outs[8 + K:8 + 2 * K]
        # STAGE-1 RELAXATION coord grad-norm tail (row, col, cage, overall) + (STAGE-2)
        # the single g_phi aggregate grad-norm, appended to the JIT return ONLY when the
        # respective log flags were ON. The coord tail is ALWAYS 4 scalars (when HYP_RELAX);
        # the g_phi scalar (when GPHI) is the LAST element. Read them, then enforce the
        # bounded-tangent-norm rim guard (spec §7) AFTER opt.step (done inside step_fn).
        tail = outs[8 + 2 * K:] if HYP_RELAX else ()
        coord_gn_ts = tail[:4] if HYP_RELAX else ()
        gphi_gn_t = tail[4] if (HYP_RELAX and HYP_GPHI and len(tail) > 4) else None
        if HYP_RELAX:
            clamp_hyp_tangent_norms(model, max_znorm=HYP_MAX_ZNORM)

        # Accumulate on-GPU (no per-step host sync). .cat -> fresh buffer (stale-ref safe);
        # .realize() enqueues without blocking the host, unlike .numpy().
        cur = total_t.reshape(1).cat(cell_ce_t.reshape(1), energy_t.reshape(1),
                                     calib_t.reshape(1), healthy_t.reshape(1))
        log_acc = cur.realize() if log_acc is None else (log_acc + cur).realize()
        log_n += 1

        if step % LOG_EVERY == 0:
            v = log_acc.numpy()  # the ONLY host sync in the hot loop: one per LOG_EVERY
            loss_a, cell_ce_a, energy_a, calib_a, healthy_a = (float(x) for x in v)
            n_skips = int(round(log_n - healthy_a))  # healthy_t is 1.0 per healthy step
            if n_skips > 0:
                print(f"[NaN-skip] {n_skips} step(s) in [{step-log_n+1}..{step}] had NaN "
                      f"grad — where()-gated, Adam moments protected", flush=True)
            dt = time.time() - t0
            print(f"[step {step:5d}] loss={loss_a/log_n:.4f} "
                  f"cell_ce={cell_ce_a/log_n:.4f} "
                  f"energy={energy_a/log_n:.4f} "
                  f"calib={calib_a/log_n:.4f}  "
                  f"({dt:.1f}s, {dt/step:.2f}s/step)", flush=True)
            # STAGE-1 RELAXATION: per-relation coord grad-norm (the boundary-explosion
            # watch, spec §7/§8.2) + the current warmed-up coord LR. coord_gn_ts is
            # (row, col, cage, overall). This is THIS step's grad (pre-clip), the honest
            # 1/(1-|z|^2) read; finite + bounded == the headline safety signal.
            if HYP_RELAX and coord_gn_ts:
                g_row, g_col, g_cage, g_all = (float(t.numpy()) for t in coord_gn_ts)
                cur_lr = float(np.asarray(opt.lr.numpy()).reshape(-1)[0])
                gphi_str = (f" gphi={float(gphi_gn_t.numpy()):.4e}"
                            if gphi_gn_t is not None else "")
                print(f"  [coord-grad] |g|: row={g_row:.4e} col={g_col:.4e} "
                      f"cage={g_cage:.4e} overall={g_all:.4e}{gphi_str}  "
                      f"(lr={cur_lr:.2e}, clip={HYP_GRAD_CLIP})",
                      flush=True)
            log_acc = None
            log_n = 0

        if step % PER_BREATH_CE_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            pb_calib = [float(t.numpy()) for t in pb_calib_ts]
            if K <= 8:
                pb_ce_str = " ".join(f"{v:.2f}" for v in pb_ce)
                pb_calib_str = " ".join(f"{v:.2f}" for v in pb_calib)
            else:
                pb_ce_str = (" ".join(f"{v:.2f}" for v in pb_ce[:4]) + " ... "
                             + " ".join(f"{v:.2f}" for v in pb_ce[-4:]))
                pb_calib_str = (" ".join(f"{v:.2f}" for v in pb_calib[:4]) + " ... "
                                + " ".join(f"{v:.2f}" for v in pb_calib[-4:]))
            ortho_str = (f" ortho={float(ortho_t.numpy()):.4f}"
                         if CODEBOOK_ORTHO > 0 else "")
            print(f"  per_breath_ce[B0..B{K-1}]:    {pb_ce_str}  "
                  f"(train cell_acc={float(cell_acc_t.numpy()):.3f} "
                  f"puzzle_acc={float(puzzle_acc_t.numpy()):.3f}{ortho_str})", flush=True)
            print(f"  per_breath_calib[B0..B{K-1}]: {pb_calib_str}", flush=True)

        if step % EVAL_EVERY == 0:
            p2_path = os.path.join(run_dir, f"property2_step{step}.jsonl")
            print(f"  evaluating on test ({EVAL_BATCHES} batches × B={EVAL_BATCH})...",
                  flush=True)
            res = evaluate(model, test_loader, K=K, max_batches=EVAL_BATCHES,
                           property2_path=p2_path, step=step)
            _print_eval_table(res, K)

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(run_dir, f"{RUN_NAME}_step{step}.safetensors")
            safe_save(_backbone_state_dict(model), ckpt_path)
            print(f"  saved {ckpt_path}", flush=True)
            # STAGE-1 RELAXATION: the backbone state dict does NOT carry the coords
            # (they are re-derived at attach in the foothold), so persist the LEARNED
            # coords separately + log |v - v_init| drift (spec §8.4 drift dump).
            if HYP_RELAX:
                coords_ck = kenken_hyp_coord_parameters(model)
                coord_path = os.path.join(run_dir, f"{RUN_NAME}_coords_step{step}.safetensors")
                # STAGE-2: persist the g_phi encoder tensors alongside the coords (they
                # are NOT in the v98 backbone state dict; init fresh on a fresh attach).
                save_sd = {f"hyp.{nm}": t for nm, t in coords_ck.items()}
                if HYP_GPHI:
                    for gi, gt in enumerate(kenken_hyp_gphi_parameters(model)):
                        save_sd[f"gphi.{gi}"] = gt
                safe_save(save_sd, coord_path)
                drift = {nm: float(np.linalg.norm(
                            t.detach().numpy() - coord_v_init[nm]))
                         for nm, t in coords_ck.items()}
                print(f"  [coord-ckpt] saved {coord_path}  drift |v-v_init|: "
                      f"row={drift['row']:.4e} col={drift['col']:.4e} "
                      f"cage={drift['cage']:.4e}", flush=True)

        if step % GC_EVERY == 0:
            gc.collect()

    # Final ckpt
    ckpt_path = os.path.join(run_dir, f"{RUN_NAME}_final.safetensors")
    safe_save(_backbone_state_dict(model), ckpt_path)
    print(f"\ndone. saved {ckpt_path}", flush=True)

    # ---- STAGE-1 RELAXATION: save final coords + freeze verification (the proof the
    # backbone never moved) + drift dump. The freeze check compares the snapshotted
    # backbone probe tensors pre/post training; bit-identical == the freeze held. ----
    if HYP_RELAX:
        coords_final = kenken_hyp_coord_parameters(model)
        coord_path = os.path.join(run_dir, f"{RUN_NAME}_coords_final.safetensors")
        save_sd = {f"hyp.{nm}": t for nm, t in coords_final.items()}
        if HYP_GPHI:
            for gi, gt in enumerate(kenken_hyp_gphi_parameters(model)):
                save_sd[f"gphi.{gi}"] = gt
        safe_save(save_sd, coord_path)
        stage_tag = "STAGE-2" if HYP_GPHI else "STAGE-1"
        print(f"  [{stage_tag}] saved final coords"
              f"{' + g_phi encoder' if HYP_GPHI else ''}: {coord_path}", flush=True)
        # Freeze verification: the backbone probe tensors must be bit-identical.
        all_frozen = True
        for nm, t in backbone_freeze_probes:
            post = t.detach().numpy()
            ident = bool(np.array_equal(post, freeze_probe_pre[nm]))
            maxabs = float(np.abs(post - freeze_probe_pre[nm]).max())
            print(f"  [freeze-check] {nm}: bit-identical={ident} max|delta|={maxabs:.3e}",
                  flush=True)
            all_frozen = all_frozen and ident
        print(f"  [freeze-check] BACKBONE FROZEN VERIFIED: {all_frozen}", flush=True)
        drift = {nm: float(np.linalg.norm(t.detach().numpy() - coord_v_init[nm]))
                 for nm, t in coords_final.items()}
        print(f"  [STAGE-1] final coord drift |v-v_init|: row={drift['row']:.4e} "
              f"col={drift['col']:.4e} cage={drift['cage']:.4e}", flush=True)


if __name__ == "__main__":
    main()
