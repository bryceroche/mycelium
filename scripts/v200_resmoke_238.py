"""v200 #238 — THE WRITE OPERATOR (§2 WRITE spec, promoted from v201 by the
#237/#237.5 carrier cells).

SINGLE EXPERIMENTAL VARIABLE vs #237.5: the shared K-slot notebook.
  WRITE: slot k = RMSNorm_detached(pool(z_k_settled)) @ W_write, written ONCE
         at breath end (fixed address = breath index; write-once by
         construction — appended, never reassigned; future breaths cannot
         overwrite). No gate on the write side.
  READ-BACK: breaths k>=1 attend to slots 0..k-1 (causal over breaths;
         pointer attention over <=7 keys — bootstrap-safe support). Enters as
         z += g_nb * RMSNorm_detached(nb_ctx) with g_nb ZERO-INIT (auxiliary
         at init per the §2 gate paragraph — the architecture is complete
         without it; exact-0 multiplier => step-0 forward byte-identical to
         #237.5, verified by microtest).
Design brief from the cells: cross-breath persistence needed (MEMORY twice),
protection from THINK's writes needed (bus-negative twice), stable address
unholdable by gradient (7/10 dim drift) — WRITE inverts all three.

PRE-REGISTERED PREDICTIONS (§2 WRITE spec, pinned before this run):
  P-W1 (BINDING, scored by C4' — §1A.E.15, pre-registered before any WRITE
       data): the U's tail flattens — erosion absorbed by protected slots.
       C4' passes at step 1000+, or tail-rise fraction < #237.5's 15%.
  P-W2 (the elegant one): the improvised carrier DECOMMISSIONS — carrier-
       projection ablation dCE at 2000 falls below #237.5's +0.0164 and/or
       overlap totality decays. The gradient stops building the substitute
       when given the real thing.
  P-W3: engagement — g_nb departs zero within ~500 steps (read in-run at
       grad captures); slot-read attention develops breath structure.
C4' (E.15, binds this run): shape (CE non-increasing through k=3) AND
descent >= 0.01 nats AND tail rise <= 50% of descent. Legacy slope reported
alongside as ladder_slope_legacy.

#237.5 is the control (same substrate, no WRITE). Same seed/steps/config.
Inherited #237.5 substrate description follows:

v200 #237.5 — SUBSTRATE RESTORATION (§1A.E.14): the backward-side fix bundle.

Three fixes, all spec-restoration class (bundled per the restoration-vs-
experimental-variable rule; none is a new mechanism), vs #237:
  1. DETACHED-SCALE SEAM NORMS (_rms_norm_detached at Seams 1/2/3): forward
     byte-identical (microtest max|diff|=0.0); closes the flat direction —
     standard RMSNorm has d(out)/d(amplitude)=0 exactly (measured), letting
     post-THINK race 0.97→10,765 while the backward divided every upstream
     gradient by that scale (organ frozen by step 600, fp16 exact-zero).
  2. FP32 INTER-BREATH CHAIN: no half casts on the latent loop (32×2048 —
     negligible memory) — no underflow cliff, no overflow ceiling at 65,504
     (the step-1828 NaN cascade's trigger).
  3. WHERE-GATED NaN GUARD: multiply-gating passed NaN to Adam's moments
     (NaN×0=NaN; #237's cascade poisoned weights straight through it);
     where() selects.
CLEAN OPTIMIZER STATE (fresh Adam moments — cold start, no resume).
Same seed (42), same 2000 steps, same K/LR/BATCH/data/masks as #237.
The #237 frozen-organ run is this run's CONTROL (accidental organ-plasticity
ablation); the diff scores every cell.

PRE-REGISTERED PREDICTIONS (§1A.E.14, pinned before this run):
  P1 (read IN-RUN at grad captures): breath_embed + alpha_read grad norms
     hold above 1e-5 through step 2000.
  P2 (THE RUN'S BINDING READ — carries the weight C4 carried in #237):
     the per-token ρ trajectory SUSTAINS past step 300 on the live organ.
     Sustains → starvation-causes-dissolution arrow PROVEN. Collapses anyway
     → objective-rejection returns from the dead. (Post-run sweep read.)
  P3 (sharpened post-#237-declaration): does the live organ REBUILD the
     carrier, and does it dissolve again on the same timescale? Recurrent
     form-and-dissolve = near-conclusive the architecture wants WRITE and
     cannot hold it. (#237 measured: MEMORY@1000 → DECORATIVE@1750.)

Inherited #237 configuration below (masks, instrumentation, reads):

Single experimental variable vs #236 WAS the (L=32, T=24) latent_topology_mask
(24 per-token + 4 per-op + 4 global) in READ cross-attention. Everything else
(K=8, LR=3e-4, BATCH=8, 6-RMSNorm substrate, delta_gate -2.0, alpha_read 1.0)
is #236's at-spec configuration. Cold-start from random init.

ADVISORY (spec deviation, §11 — surfaces in line 1 of the log AND the verdict
line): per-op mask group assigns tokens by token_index mod 4, NOT factor-graph
op_type routing as §2's partition table specifies. Consequence pre-pinned
BEFORE results: the per-op group's entropy-separation read does NOT bind
(its token slices are semantically arbitrary); the masks-help-vs-decorative
read is carried by the per-token and global groups + C4. Next at-spec pass
wires real op_type routing or the brief blesses the proxy.

Horizon: 2000 steps (E.10 reads bind at step 1000+; Stage-2 short-prod = 2000).
Eval checkpoints at steps {200, 500, 1000, 1500, 2000}; dense model ckpts
every 100 steps to 1K then every 250 (§9 — this run is the control-generator;
retroactive persistence is impossible).

Sequence:
  1. STEP-0 GATE (§1A.E.10/§7): per-mask-family cross-attn entropy at random
     init strictly below each family's log(support); all-latent mean strictly
     below log(24)=3.178. Assertion failure = mask not wired = fix and re-run,
     DO NOT train.
  2. Regenerate ALL reference curves under the masked architecture (§6: §2
     changed → fresh null; entropy curves now MEASURED from attention weights,
     not hardcoded constants).
  3. Train 2000 steps; per-checkpoint instrumentation via ONE tapped canonical
     forward (fg_breathing_forward_v200(taps=...) — §2 single-forward; the
     #236-era probe reimplementations are gone from this driver).
  4. Final read: C1-C6 (C5 = recalibrated metric: pre-norm waist contribution
     norm + consecutive even-breath delta-direction cosine, first deployment),
     §1A.E.4 grid, §1A.E.10 pre-committed masks reads, even-breath cosine
     creep, per-latent THINK entropy (#238 routing diagnostic).

Pre-committed E.10 reads (binding, written before the run):
  - mean-removed inter-position cosine stays diverse (~-0.03 to 0.3)
  - per-group entropy separates by step 1000+ (per-token ~0, global sharpens;
    per-op read non-binding per the ADVISORY above)
  - C4 ladder slope <= -0.05 by step 1000. C4 fails again => §1A.E.2 component
    checks are MANDATORY (three deferrals was the limit) and the pre-registered
    message-passing hypothesis (memory/project_v200_message_passing_hypothesis_jun11.md)
    is confirmed in pre-committed shape => #238 = THINK quotient-graph mask.
  - concentration drift stays ~4% at post-THINK site (same-site provenance).

Pre-committed #238-routing thresholds (per-latent THINK entropy, set before
results): uniform mean-field = std-across-latents < 0.05 nats at final
checkpoint AND |mean - log(31)| < 0.15; differentiated = std >= 0.05.

Pre-committed C5 recalibrated thresholds (first deployment, §1A.B.5): PASS =
min even-breath waist-contribution norm > 1e-6 AND coefficient-of-variation of
the contribution norms across even breaths > 0.01. Secondary (logged, not
gated): cosine between consecutive even-breath contribution directions.

Predicted per-element scale trajectory (§1A.E.9, carried from #236):
  post-norm_breath ~1.0; post-READ-add ~2.0 (NOTE #236 measured 1.30 — carried
  observation, not recalibrated); post-THINK ~4-50 (not a criterion);
  post-norm_blend ~1.0; post-blend ~1.0.

Output artifacts:
  .cache/v200_smoke/reference_curves/*.npz + provenance   (regenerated, mask1a arch)
  .cache/v200_smoke/reference_curves/xattn_entropy_per_family_random_init.npz  (NEW)
  .cache/v200_smoke/reference_curves/think_per_latent_entropy_random_init.npz  (NEW)
  .cache/v200_smoke/train_238.log               — first line ADVISORY: ...
  .cache/v200_smoke/step2000_eval_238.json      — final eval + all reads
  .cache/v200_smoke/instrumentation_238.json      — per-checkpoint metric timeseries
  .cache/v200_smoke/grad_norms_238.npz
  .cache/v200_smoke/persistence/step{N}_z_238.npz       — per eval checkpoint
  .cache/v200_perceiver_ckpts/v200_perceiver_238_write8_step{N}.safetensors

Usage:
  cd /home/bryce/mycelium
  bash scripts/v200_resmoke_238.sh
"""
from __future__ import annotations

import gc
import json
import os
import sys
import subprocess
import time
from pathlib import Path

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy.stats import spearmanr

from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import safe_save

from mycelium.llama_loader import (
    attach_llama_layers, load_llama_weights,
    LLAMA_3_2_1B_CFG, SMOLLM2_1_7B_CFG,
    LlamaConfig,
)
from mycelium.factor_graph_v200 import (
    attach_fg_params_v200, fg_v200_parameters, fg_v200_state_dict,
    _compile_jit_fg_step_v200, fg_breathing_forward_v200, compute_drift_v200,
    _collect_latent_snapshots, compute_latent_jsd_from_snapshots,
    fg_v200_empty_taps, xattn_entropy_per_family_per_breath,
    sa_per_latent_entropy_per_breath, verify_topology_mask_step0,
    V200_MASK_FAMILIES,
    V200_K_MAX, V200_N_MAX, V200_F_MAX, V200_N_VAR_LAT, V200_N_DIGITS,
    V200_STAGE2A_WAIST, V200_WAIST_DIM,
)
from mycelium.factor_graph_v108 import bins_to_digits_msd
from mycelium.factor_graph_data_v107 import (
    FactorGraphLoaderV107, DualDataLoaderV107, load_gsm8k_records_v107,
)
from mycelium.provenance import make_provenance, write_with_provenance

DIFFICULTIES = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# Arch-version string (§6 required field)
# ---------------------------------------------------------------------------

def _get_arch_version(config_sig: str = "K8_L32_prenorm5_seamthree_gate-2") -> str:
    """Build the §6 arch_version string: v200-{git_sha[:8]}-{config_sig}.

    #237 review fix: append '-dirty' when the working tree has uncommitted
    changes, so the sha8 cannot be mistaken for an exact code pin (HEAD does
    not contain the masked architecture until the #237 surface is committed).
    """
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=5,
        )
        sha8 = r.stdout.strip()[:8] if r.returncode == 0 else "unknown"
        rs = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=5,
        )
        if rs.returncode == 0 and rs.stdout.strip():
            sha8 += "-dirty"
    except Exception:
        sha8 = "unknown"
    return f"v200-{sha8}-{config_sig}"


def _get_metric_sha() -> str:
    """SHA of the factor_graph_v200.py file — used as metric_sha per §7."""
    path = os.path.join(_PROJECT_ROOT, "mycelium", "factor_graph_v200.py")
    try:
        r = subprocess.run(
            ["git", "hash-object", path],
            capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Helpers (thin wrappers; forward logic lives in factor_graph_v200.py)
# ---------------------------------------------------------------------------

class _Obj:
    pass


def _cast_llama_fp32(model: _Obj) -> None:
    for layer in model.llama_layers:
        for p in layer.parameters():
            if p.dtype != dtypes.float:
                p.assign(p.cast(dtypes.float)).realize()


def _make_n_vars_mask(n_vars_total: np.ndarray, n_var_lat: int) -> Tensor:
    B = len(n_vars_total)
    vi = np.arange(n_var_lat, dtype=np.int32)
    mask = (vi[None, :] < n_vars_total[:, None]).astype(np.float32)
    return Tensor(mask, dtype=dtypes.float).contiguous().realize()


def _param_group_grad_norms(model: _Obj, stage2a_waist: bool) -> dict[str, float]:
    def _l2(tensors):
        sq = 0.0; n = 0
        for t in tensors:
            if t is not None and t.grad is not None:
                g = t.grad.cast(dtypes.float).realize().numpy()
                sq += float((g ** 2).sum()); n += 1
        return float(np.sqrt(sq)) if n > 0 else 0.0

    groups: dict[str, list] = {
        "backbone_L0_L3": [], "latent_init": [], "cross_attn": [],
        "waist_down_proj": [], "waist_up_proj": [],
        "tree_readout": [], "calib_head": [], "delta_gate": [], "breath_embed": [],
        "alpha_read": [],       # #235: named separately for §10 row 5 strain detection
        "norm_blend_weight": [], # #236: Seam 3 learnable gain — strain check
        "notebook": [],          # #238: WRITE operator (W_write, wq, wo, norms, gate)
    }
    for layer in model.llama_layers:
        for p in layer.parameters():
            groups["backbone_L0_L3"].append(p)
    groups["latent_init"].append(model.fg_v200_latents)
    groups["breath_embed"].append(model.fg_v200_breath_embed)
    for attr in ["fg_v200_cross_wq", "fg_v200_cross_wk",
                 "fg_v200_cross_wv", "fg_v200_cross_wo",
                 "fg_v200_breath_norm_w",    # added #234 (§2 norm_breath)
                 "fg_v200_read_norm_w",
                 "fg_v200_read_ctx_norm_w",  # added #235 (§1A.E.4 READ-dominance fix)
                 "fg_v200_commit_norm_w",
                 "fg_v200_latent_norm_w"]:
        if hasattr(model, attr):
            groups["cross_attn"].append(getattr(model, attr))
    # alpha_read: named separately (§10 row 5 strain check)
    if hasattr(model, "fg_v200_alpha_read"):
        groups["alpha_read"].append(model.fg_v200_alpha_read)
    # norm_blend_weight: named separately (Seam 3 gain, #236 strain check)
    if hasattr(model, "fg_v200_blend_norm_w"):
        groups["norm_blend_weight"].append(model.fg_v200_blend_norm_w)
    # notebook: WRITE operator group (#238)
    for attr in ["fg_v200_nb_W_write", "fg_v200_nb_wq", "fg_v200_nb_wo",
                 "fg_v200_nb_write_norm_w", "fg_v200_nb_read_norm_w",
                 "fg_v200_nb_gate"]:
        if hasattr(model, attr):
            groups["notebook"].append(getattr(model, attr))
    if stage2a_waist:
        if hasattr(model, "fg_v200_W_compress"):
            groups["waist_down_proj"].append(model.fg_v200_W_compress)
        if hasattr(model, "fg_v200_W_expand"):
            groups["waist_up_proj"].append(model.fg_v200_W_expand)
        if hasattr(model, "fg_v200_waist_gate"):
            groups["waist_up_proj"].append(model.fg_v200_waist_gate)
    for attr in ["fg_v200_tree_codebook"]:
        if hasattr(model, attr):
            groups["tree_readout"].append(getattr(model, attr))
    for attr in ["fg_v200_calib_w", "fg_v200_calib_b"]:
        if hasattr(model, attr):
            groups["calib_head"].append(getattr(model, attr))
    groups["delta_gate"].append(model.fg_v200_delta_gate)
    return {grp: _l2(tensors) for grp, tensors in groups.items()}


def _evaluate_v200(model, val_loader, K, n_max, f_max, n_var_lat, n_digits,
                   max_batches, stage2a_waist):
    was_training = Tensor.training
    Tensor.training = False
    results = {d: {"n": 0, "cell_correct": 0, "cell_total": 0,
                   "query_correct": 0, "query_total": 0,
                   "per_pos_correct": np.zeros(n_digits),
                   "per_pos_total": np.zeros(n_digits)} for d in DIFFICULTIES}
    n_batches = 0
    for batch in val_loader.iter_eval():
        if n_batches >= max_batches:
            break
        n_batches += 1
        domain_init = batch["domain_init"]
        node_kinds  = batch["node_kinds"]
        gold_bins_np = batch["gold_bins"].realize().numpy()
        obs_np      = batch["observed_mask"].realize().numpy()
        n_vars_np   = batch["n_vars_total"]
        query_np    = batch["query_idx"]
        picks       = batch.get("picks", [])
        B_local     = domain_init.shape[0]
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=n_digits)
        n_eval = min(n_var_lat, n_max)
        tree_logits_hist, _ = fg_breathing_forward_v200(
            model, domain_init, node_kinds,
            K=K, n_max=n_max, f_max=f_max,
            n_var_lat=n_var_lat, n_digits=n_digits,
            training=False, stage2a_waist=stage2a_waist,
        )
        final_logits_np = tree_logits_hist[-1].realize().numpy()
        for b in range(B_local):
            diff = picks[b].get("difficulty", "easy") if picks else "easy"
            if diff not in results:
                diff = "easy"
            rv = results[diff]
            nv = int(n_vars_np[b])
            n_check = min(nv, n_eval)
            qi = int(query_np[b])
            for vi in range(n_check):
                if obs_np[b, vi] == 1:
                    continue
                rv["cell_total"] += 1
                pred = final_logits_np[b, vi].argmax(axis=-1)
                gold = gold_digits_np[b, vi]
                cell_ok = int(np.all(pred == gold))
                rv["cell_correct"] += cell_ok
                for d_idx in range(n_digits):
                    rv["per_pos_total"][d_idx] += 1
                    rv["per_pos_correct"][d_idx] += int(pred[d_idx] == gold[d_idx])
            if qi < n_check and obs_np[b, qi] == 0:
                pred_q = final_logits_np[b, qi].argmax(axis=-1)
                gold_q = gold_digits_np[b, qi]
                rv["query_total"] += 1
                rv["query_correct"] += int(np.all(pred_q == gold_q))
    Tensor.training = was_training
    out = {}
    for d in DIFFICULTIES:
        rv = results[d]
        if rv["cell_total"] == 0:
            continue
        ca = rv["cell_correct"] / rv["cell_total"]
        qa = (rv["query_correct"] / rv["query_total"]) if rv["query_total"] > 0 else 0.0
        ppa = [(rv["per_pos_correct"][i] / max(1, rv["per_pos_total"][i]))
               for i in range(n_digits)]
        out[d] = {"cell_acc": ca, "query_acc": qa, "digit_acc": float(np.mean(ppa)),
                  "per_pos_acc": ppa, "n_puzzles": rv["query_total"]}
    return out


def _per_breath_ce_at_eval(model, val_loader, K, n_max, f_max, n_var_lat, n_digits,
                            max_batches, stage2a_waist, carrier_dim_mask=None,
                            carrier_mask_site="boundary"):
    was_training = Tensor.training
    Tensor.training = False
    n_eval = min(n_var_lat, n_max)
    pb_sums = np.zeros(K); pb_n = 0
    for batch in val_loader.iter_eval():
        if pb_n >= max_batches:
            break
        pb_n += 1
        domain_init  = batch["domain_init"]
        node_kinds   = batch["node_kinds"]
        gold_bins_np = batch["gold_bins"].realize().numpy()
        obs_np       = batch["observed_mask"].realize().numpy()
        n_vars_np    = batch["n_vars_total"]
        B_local      = domain_init.shape[0]
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=n_digits)
        n_vars_mask_np = (np.arange(n_eval)[None, :] < n_vars_np[:, None]).astype(np.float32)
        tree_logits_hist, _ = fg_breathing_forward_v200(
            model, domain_init, node_kinds,
            K=K, n_max=n_max, f_max=f_max,
            n_var_lat=n_var_lat, n_digits=n_digits,
            training=False, stage2a_waist=stage2a_waist,
            carrier_dim_mask=carrier_dim_mask,   # §1A.E.13 carrier-projection ablation
            carrier_mask_site=carrier_mask_site,
        )
        gold_eval = gold_digits_np[:, :n_eval, :]
        obs_eval  = obs_np[:, :n_eval]
        real_mask = n_vars_mask_np
        unobs = (1 - obs_eval) * real_mask
        denom = unobs.sum() + 1e-8
        for k_idx in range(K):
            logits_k = tree_logits_hist[k_idx].realize().numpy()
            lk = logits_k - logits_k.max(axis=-1, keepdims=True)
            log_p = lk - np.log(np.exp(lk).sum(axis=-1, keepdims=True) + 1e-8)
            g_idx = gold_eval[:, :, :, np.newaxis]
            nll_k = -np.take_along_axis(log_p, g_idx, axis=-1).squeeze(-1)
            nll_pos = nll_k.mean(axis=-1)
            ce_k = (nll_pos * unobs).sum() / denom
            pb_sums[k_idx] += float(ce_k)
    Tensor.training = was_training
    if pb_n == 0:
        return [float('nan')] * K
    return (pb_sums / pb_n).tolist()


def _eager_grad_norm_step(model, domain_init, node_kinds, gold_digits, obs_mask,
                          n_vars_mask, opt, K, n_max, f_max, n_var_lat, n_digits,
                          calib_weight, stage2a_waist):
    opt.zero_grad()
    n_eval = min(n_var_lat, n_max)
    B = int(domain_init.shape[0])
    ladder_weights = [1.0 + k_idx / float(max(K - 1, 1)) for k_idx in range(K)]
    tree_logits_history, calib_history = fg_breathing_forward_v200(
        model, domain_init, node_kinds,
        K=K, n_max=n_max, f_max=f_max,
        n_var_lat=n_var_lat, n_digits=n_digits,
        training=True, stage2a_waist=stage2a_waist,
    )
    gold_eval  = gold_digits[:, :n_eval, :].cast(dtypes.int)
    real_mask  = n_vars_mask[:, :n_eval].cast(dtypes.float)
    unobs_mask = (1 - obs_mask[:, :n_eval].cast(dtypes.float)) * real_mask
    unobs_sum  = unobs_mask.sum() + 1e-8
    gd_flat  = gold_eval.reshape(B * n_eval * n_digits)
    gold_oh  = gd_flat.one_hot(10).cast(dtypes.float)
    var_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
    var_weight_sum = 0.0
    for k_idx in range(K):
        logits_k = tree_logits_history[k_idx]
        lk_flat  = logits_k.reshape(B * n_eval * n_digits, 10)
        log_p    = lk_flat.log_softmax(axis=-1)
        nll_flat = -(log_p * gold_oh).sum(axis=-1)
        nll_pos  = nll_flat.reshape(B, n_eval, n_digits).mean(axis=-1)
        ce_k     = (nll_pos * unobs_mask).sum() / unobs_sum
        var_loss_sum = var_loss_sum + ce_k * ladder_weights[k_idx]
        var_weight_sum += ladder_weights[k_idx]
    total_ce = var_loss_sum / float(var_weight_sum)
    final_tree = tree_logits_history[-1]
    pred_final = final_tree.argmax(axis=-1).detach()
    eq_per_pos = (pred_final == gold_eval).cast(dtypes.float)
    eq = eq_per_pos.prod(axis=-1)
    unobs_2d = (1 - obs_mask[:, :n_eval].cast(dtypes.float)) * real_mask
    n_unobs_per = unobs_2d.sum(axis=-1) + 1e-8
    correct = (eq * unobs_2d).sum(axis=-1) / n_unobs_per
    calib_loss_sum = Tensor.zeros((), dtype=dtypes.float).contiguous()
    for kc, calib_k in enumerate(calib_history):
        prog = float(kc) / float(max(K - 1, 1))
        target_k = 0.5 + (correct - 0.5) * prog
        calib_loss_sum = calib_loss_sum + ((calib_k - target_k) ** 2).mean()
    calib_loss = calib_loss_sum / float(K)
    total = total_ce + calib_weight * calib_loss
    total.backward()
    norms = _param_group_grad_norms(model, stage2a_waist=stage2a_waist)
    opt.zero_grad()
    return norms


# ---------------------------------------------------------------------------
# Reference curve generation (Sub-task 3)
# ---------------------------------------------------------------------------

def generate_reference_curves(
    model: _Obj,
    val_loader: FactorGraphLoaderV107,
    K: int, B_ref: int,
    n_max: int, f_max: int,
    stage2a_waist: bool,
    ref_dir: str,
    arch_version: str,
    metric_sha: str,
    seed: int = 42,
) -> dict:
    """Run random-init tapped forward on B_ref batches and save reference curves.

    #237 change: entropy reference curves are MEASURED from attention weights
    via run_instrumented_probe (the #236 version hardcoded np.full(K, log(24)) /
    np.full(K, log(32)) constants, which are factually wrong under the §2 mask —
    per-token family entropy ≈ 0, per-op ≤ log(6)). Adds two new reference
    files: per-mask-family READ entropy (3, K) and per-latent THINK entropy
    mean (K,) under the exclude-self log(31) convention.

    Saves 9 npz files + provenance sidecars per §5/§6/§7.
    Returns dict with paths to all saved files.
    """
    os.makedirs(ref_dir, exist_ok=True)
    print(f"\n[ref] Generating reference curves (random-init, K={K}, B≥{B_ref})...")
    print(f"      arch_version: {arch_version}")
    print(f"      metric_sha:   {metric_sha}")

    all_jsd:      list[list[float]] = []
    all_energy:   list[np.ndarray] = []
    all_xattn:    list[np.ndarray] = []
    all_xattn_fam: list[np.ndarray] = []
    all_sa:       list[np.ndarray] = []
    all_sa_std:   list[np.ndarray] = []
    all_ipc_mr:   list[np.ndarray] = []
    all_rdr:      list[np.ndarray] = []
    all_zmag:     list[np.ndarray] = []

    n_batches_done = 0
    for batch in val_loader.iter_eval():
        if n_batches_done >= B_ref:
            break
        domain_init = batch["domain_init"]
        node_kinds  = batch["node_kinds"]

        probe = run_instrumented_probe(
            model, domain_init, node_kinds, K, n_max, f_max, stage2a_waist,
        )
        snapshots = probe["snapshots"]
        jsd = probe["jsd"]
        all_jsd.append(jsd)

        energy_k = []
        for k_idx in range(K):
            delta = snapshots[k_idx + 1] - snapshots[k_idx]
            e = np.linalg.norm(delta, axis=-1).mean(axis=0)
            energy_k.append(e)
        all_energy.append(np.stack(energy_k, axis=0))

        zmag_k = []
        for k_idx in range(K + 1):
            zmag = np.linalg.norm(snapshots[k_idx], axis=-1).mean()
            zmag_k.append(float(zmag))
        all_zmag.append(np.array(zmag_k))

        all_ipc_mr.append(np.array(probe["ipc_mr"]))
        all_rdr.append(np.array(probe["rdr"]))

        # MEASURED entropy curves (#237; replaces hardcoded constants)
        all_xattn.append(np.array(probe["fam_ent"]["all_mean"]))
        all_xattn_fam.append(np.stack(
            [np.array(probe["fam_ent"][name]) for name, _, _ in V200_MASK_FAMILIES],
            axis=0,
        ))   # (3, K) — family axis ordered per V200_MASK_FAMILIES
        all_sa.append(np.array(probe["sa_ent"]["mean_per_breath"]))
        all_sa_std.append(np.array(probe["sa_ent"]["std_per_breath"]))

        n_batches_done += 1
        print(f"  batch {n_batches_done}/{B_ref}  jsd[0]={jsd[0]:.5f}  zmag[0]={zmag_k[0]:.4f}  "
              f"xattn_mean[0]={all_xattn[-1][0]:.4f}  sa_mean[0]={all_sa[-1][0]:.4f}",
              flush=True)

    jsd_arr   = np.array(all_jsd).mean(axis=0)
    energy_arr = np.stack(all_energy).mean(axis=0)
    zmag_arr   = np.stack(all_zmag).mean(axis=0)
    ipc_mr_arr = np.stack(all_ipc_mr).mean(axis=0)
    rdr_arr    = np.stack(all_rdr).mean(axis=0)
    xattn_arr  = np.stack(all_xattn).mean(axis=0)
    xattn_fam_arr = np.stack(all_xattn_fam).mean(axis=0)   # (3, K)
    sa_arr     = np.stack(all_sa).mean(axis=0)
    sa_std_arr = np.stack(all_sa_std).mean(axis=0)

    def _save_ref(data: np.ndarray, fname: str, metric_name: str, units: str,
                  description: str) -> str:
        path = os.path.abspath(os.path.join(ref_dir, fname))
        prov = make_provenance(
            metric=metric_name,
            units=units,
            shape=list(data.shape),
            ckpt=f"random_init_seed_{seed}",
            split="reference",
            seed=seed,
            step=0,
            env_vars={"K": str(K), "B_ref": str(B_ref), "arch": arch_version},
            output_path=os.path.abspath(path),
            key="data",
            arch_version=arch_version,
            metric_sha=metric_sha,
        )
        prov["what"]["description"] = description
        write_with_provenance({"data": data}, path, prov)
        print(f"  Saved {fname}  shape={data.shape}  mean={float(data.mean()):.5f}")
        return path

    paths = {}
    paths["jsd"]   = _save_ref(jsd_arr,   "latent_jsd_random_init.npz",
                                "latent_jsd_random_init",
                                "dimensionless (JSD)",
                                "Pairwise inter-position cosine fingerprint JSD, corrected metric")
    paths["energy"] = _save_ref(energy_arr, "energy_channel_random_init.npz",
                                 "energy_channel_random_init",
                                 "L2 norm (float)",
                                 "Per-latent delta-z per breath, averaged over batch")
    paths["xattn"]  = _save_ref(xattn_arr, "xattn_entropy_random_init.npz",
                                 "xattn_entropy_random_init",
                                 "nats (MEASURED, masked arch)",
                                 "Cross-attn entropy per breath, mean over all 32 latents, "
                                 "MEASURED under §2 mask1a (expect ~0.62 = "
                                 "(24*0 + 4*log6 + 4*log24)/32, NOT log(24))")
    paths["xattn_fam"] = _save_ref(xattn_fam_arr, "xattn_entropy_per_family_random_init.npz",
                                 "xattn_entropy_per_family_random_init",
                                 "nats (MEASURED, per mask family)",
                                 "Per-mask-family READ entropy per breath, shape (3, K); "
                                 "family axis ordered per V200_MASK_FAMILIES "
                                 "[per_token, per_op, global]; expected ~[0, log6, log24] at init")
    paths["sa"]     = _save_ref(sa_arr,    "self_attn_entropy_random_init.npz",
                                 "self_attn_entropy_random_init",
                                 "nats (MEASURED, exclude-self over 31 latents)",
                                 "Per-latent THINK attention entropy per breath, mean across "
                                 "32 query latents; exclude-self convention, random-init "
                                 "reference log(31)=3.434 (§7 #238-routing diagnostic)")
    paths["sa_std"] = _save_ref(sa_std_arr, "think_per_latent_entropy_random_init.npz",
                                 "think_per_latent_entropy_std_random_init",
                                 "nats (MEASURED, std across 32 latents)",
                                 "Std of per-latent THINK attention entropy across the 32 "
                                 "query latents per breath; uniform mean-field mixing = "
                                 "near-zero std (§7, routes #238)")
    paths["ipc_mr"] = _save_ref(ipc_mr_arr, "inter_pos_cos_mean_removed_random_init.npz",
                                 "inter_pos_cos_mean_removed_random_init",
                                 "cosine (dimensionless)",
                                 "Mean-removed inter-position cosine per breath (§1A.E.4)")
    paths["rdr"]    = _save_ref(rdr_arr,   "read_dominance_ratio_random_init.npz",
                                 "read_dominance_ratio_random_init",
                                 "ratio (dimensionless)",
                                 "delta-z / z_pre per breath proxy for READ dominance (§1A.E.4)")
    paths["zmag"]   = _save_ref(zmag_arr,  "z_magnitude_random_init.npz",
                                 "z_magnitude_random_init",
                                 "L2 norm (float)",
                                 "Mean z_k per breath — C6 baseline (max/min ratio should be <3)")
    return paths


# ---------------------------------------------------------------------------
# §1A.E.4 Position-collapse grid reading
# ---------------------------------------------------------------------------

def read_e4_grid(ipc_mr_trained: np.ndarray, rdr_trained: np.ndarray) -> dict:
    """Map re-smoke results to §1A.E.4 three-cell grid."""
    max_ipc = float(np.max(np.nan_to_num(ipc_mr_trained, nan=0.0)))
    mean_rdr = float(np.mean(np.nan_to_num(rdr_trained, nan=0.0)))

    if mean_rdr >= 10.0:
        cell = "READ dominance"
        interp = ("‖read_ctx‖/‖z‖ still > 10 post-fix. Substrate fix didn't contain "
                  "READ's contribution. READ output dominates regardless of pre-norm.")
        next_move = ("§2 follow-up: add a gated residual blend on READ "
                     "(zero-init scale on read_ctx per v109 discipline). Architectural fix in §2, not v1.1.")
    elif max_ipc > 0.8:
        cell = "Real consensus"
        interp = ("Mean-removed cosine still > 0.8 after breath 0. Llama self-attn is driving "
                  "all latents to consensus even with normalized scales. Principle 10 biting.")
        next_move = ("Promote v1.1 row 1 (per-position routing / topology tensor) to Stage 1B+, "
                     "with the diagnosis already locked.")
    else:
        cell = "Arithmetic collapse"
        interp = ("Diversity persists (ipc_mr <= 0.8 across breaths). The 0.9999998 in the "
                  "original smoke was shared-additive dominance, dissolves under §2 RMSNorm.")
        next_move = ("No architectural change. v1.1 row 1 stays queued. "
                     "Re-smoke verdict reads the 5+1 standard criteria.")

    return {
        "cell": cell,
        "interpretation": interp,
        "next_move": next_move,
        "raw_values": {
            "max_ipc_mr_across_breaths": max_ipc,
            "mean_rdr_across_breaths":   mean_rdr,
            "ipc_mr_per_breath":         list(float(v) for v in ipc_mr_trained),
            "rdr_per_breath":            list(float(v) for v in rdr_trained),
        },
    }


# ---------------------------------------------------------------------------
# C6 z-magnitude bounded check
# ---------------------------------------------------------------------------

def check_c4_prime(pb_ce: list) -> dict:
    """C4′ (§1A.E.15, pre-registered before any WRITE data; BINDS from #238).

    Three clauses, all required:
      shape: per-breath eval CE non-increasing through k=3 (tol 1e-4)
      magnitude: descent (breath-0 CE − min CE) >= 0.01 nats
      tail: tail rise (final CE − min CE) <= 50% of descent
    Validated on pre-existing runs: passes #237.5@200/2000, fails #237@1000
    (tail 66%) and #236 (anti-ladder). Legacy slope reported alongside.
    """
    ce = np.array(pb_ce, dtype=np.float64)
    shape_ok = bool(all(ce[i + 1] <= ce[i] + 1e-4 for i in range(min(3, len(ce) - 1))))
    descent = float(ce[0] - ce.min())
    tail = float(ce[-1] - ce.min())
    mag_ok = bool(descent >= 0.01)
    tail_ok = bool(descent > 0 and tail <= 0.5 * descent)
    return {"shape_ok": shape_ok, "descent": descent, "tail_rise": tail,
            "tail_frac": float(tail / descent) if descent > 0 else float("inf"),
            "magnitude_ok": mag_ok, "tail_ok": tail_ok,
            "pass": bool(shape_ok and mag_ok and tail_ok)}


def check_c6_z_magnitude(snapshots: list) -> dict:
    """Criterion 6: max_k(‖z_k‖) / min_k(‖z_k‖) < 3.0 across K breaths.

    Operates on post-breath states (snapshots[1:]), not the init snapshot.
    """
    init_mag = float(np.linalg.norm(snapshots[0], axis=-1).mean()) if snapshots else 0.0
    post_breath = snapshots[1:] if len(snapshots) > 1 else snapshots
    mags = [float(np.linalg.norm(s, axis=-1).mean()) for s in post_breath]
    max_mag = float(max(mags)) if mags else 0.0
    min_mag = float(min(mags)) if mags else 0.0
    ratio = max_mag / (min_mag + 1e-8)
    verdict = "PASS" if ratio < 3.0 else "FAIL"
    return {
        "ratio": ratio, "max_mag": max_mag, "min_mag": min_mag,
        "init_mag": init_mag,
        "per_breath": [init_mag] + mags,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# §1A.E.9 Five-checkpoint within-breath per-element scale trajectory (#236)
# ---------------------------------------------------------------------------

def _per_elem_scale(z: "np.ndarray") -> float:
    """Per-element scale: ‖z‖ / sqrt(numel), averaged over batch + latent positions."""
    numel = z.shape[-1]
    norms = np.linalg.norm(z, axis=-1)
    return float((norms / np.sqrt(numel)).mean())


def compute_ipc_rdr_from_snapshots(snapshots: list, K: int) -> tuple:
    """§1A.E.4 metrics from snapshots — identical math to #236's inline loop.

    Returns (ipc_mr_per_breath, rdr_per_breath) as two length-K float lists.
    """
    ipc_mr, rdr = [], []
    for k_idx in range(K):
        z_np  = snapshots[k_idx + 1]
        z_pre = snapshots[k_idx]
        z_mean = z_np.mean(axis=1, keepdims=True)
        z_c    = z_np - z_mean
        norms  = np.linalg.norm(z_c, axis=-1, keepdims=True)
        z_n    = z_c / (norms + 1e-8)
        gram   = np.einsum('bld,bmd->blm', z_n, z_n)
        L_dim  = z_np.shape[1]
        idx    = np.triu_indices(L_dim, k=1)
        upper  = gram[:, idx[0], idx[1]]
        ipc_mr.append(float(upper.mean()))
        delta      = z_np - z_pre
        zp_norm    = np.linalg.norm(z_pre,  axis=-1)
        delta_norm = np.linalg.norm(delta, axis=-1)
        rdr.append(float((delta_norm / (zp_norm + 1e-8)).mean()))
    return ipc_mr, rdr


def run_instrumented_probe(
    model: "_Obj",
    domain_init: "Tensor",
    node_kinds: "Tensor",
    K: int,
    n_max: int,
    f_max: int,
    stage2a_waist: bool,
) -> dict:
    """ONE tapped canonical forward → every #237 probe metric (§2 single-forward).

    Replaces #236's measure_within_breath_trajectory + measure_concentration_drift
    (which reimplemented the breath loop and silently dropped the topology mask)
    AND adds the #237 instrumentation: per-mask-family READ entropy, per-latent
    THINK entropy, C5 recalibrated waist-contribution metrics.

    All metrics below are computed from the SAME forward pass the model trains
    with (fg_breathing_forward_v200), so probe-vs-training divergence is
    structurally impossible.

    Returns dict with:
      snapshots        : K+1 × (B, L, H) fp32 — init + post-blend per breath
      jsd              : latent JSD trajectory (corrected fingerprint metric)
      c6               : check_c6_z_magnitude result
      ipc_mr, rdr      : §1A.E.4 per-breath lists
      ipc_mr_even_mean / ipc_mr_odd_mean : even-breath cosine creep inputs
      traj             : {"trajectory_per_breath", "trajectory_avg"} (§1A.E.9)
      conc             : concentration dict (#236 shape + measurement_site fields)
      fam_ent          : per-mask-family READ entropy dict (per breath)
      sa_ent           : per-latent THINK entropy dict (per_latent/mean/std)
      c5               : recalibrated waist metrics (contribution norms pre-norm,
                         CoV, consecutive even-breath direction cosines)
    """
    was_training = Tensor.training
    Tensor.training = False
    taps = fg_v200_empty_taps()
    tree_logits_history, calib_history = fg_breathing_forward_v200(
        model, domain_init, node_kinds, K=K, n_max=n_max, f_max=f_max,
        training=False, stage2a_waist=stage2a_waist, taps=taps,
    )
    Tensor.training = was_training

    wb_keys = ("wb_post_norm", "wb_post_read", "wb_post_think",
               "wb_post_norm_blend", "wb_post_blend")
    wb_np = {key: [np.asarray(t.numpy(), dtype=np.float32) for t in taps[key]]
             for key in wb_keys}

    # Snapshots: init + post-blend per breath (same points as _collect_latent_snapshots)
    snapshots = [np.asarray(taps["z_init"].numpy(), dtype=np.float32)]
    snapshots += wb_np["wb_post_blend"]

    # §1A.E.9 within-breath per-element trajectory
    traj_per_breath = [[_per_elem_scale(wb_np[key][k]) for key in wb_keys]
                       for k in range(K)]
    traj = {
        "trajectory_per_breath": traj_per_breath,
        "trajectory_avg": np.array(traj_per_breath).mean(axis=0).tolist(),
    }

    # §7 concentration drift — same sites as #236 (post-THINK pre-waist-pre-norm_blend;
    # post-delta-gate-blend), now measured on the MASKED canonical forward
    post_think_fracs = [_compute_concentration_top10(wb_np["wb_post_think"][k]) for k in range(K)]
    post_blend_fracs = [_compute_concentration_top10(wb_np["wb_post_blend"][k]) for k in range(K)]
    conc = {
        "post_think_top10_frac":  float(np.mean(post_think_fracs)),
        "post_blend_top10_frac":  float(np.mean(post_blend_fracs)),
        "post_think_per_breath":  post_think_fracs,
        "post_blend_per_breath":  post_blend_fracs,
        "measurement_site_think": "post-THINK-pre-waist-pre-norm_blend",
        "measurement_site_blend": "post-delta-gate-blend",
        "reference_llama_tokens":  0.98,
        "reference_v235_trained":  0.258,
        "reference_v236_step200":  0.0404,
        "reference_random_gauss":  0.048,
    }

    # #237 per-mask-family READ entropy + per-latent THINK entropy
    fam_ent_raw = xattn_entropy_per_family_per_breath(taps)
    fam_ent = {name: [float(v) for v in fam_ent_raw[name]]
               for name, _, _ in V200_MASK_FAMILIES}
    fam_ent["all_mean"] = [float(v) for v in fam_ent_raw["all_mean"]]
    fam_ent["per_latent"] = fam_ent_raw["per_latent"].tolist()

    sa_raw = sa_per_latent_entropy_per_breath(taps, exclude_self=True)
    sa_ent = {
        "mean_per_breath": [float(v) for v in sa_raw["mean"]],
        "std_per_breath":  [float(v) for v in sa_raw["std"]],
        "per_latent":      sa_raw["per_latent"].tolist(),
        "ref_log31":       sa_raw["ref_log31"],
    }

    # C5 recalibrated (§1A.B.5, first deployment): pre-norm waist contribution
    contribs = [np.asarray(t.numpy(), dtype=np.float32) for t in taps["waist_contrib"]]
    contrib_norms = [float(np.linalg.norm(c, axis=-1).mean()) for c in contribs]
    dirs = [c.reshape(-1) / (np.linalg.norm(c.reshape(-1)) + 1e-12) for c in contribs]
    consec_cos = [float(np.dot(dirs[i], dirs[i + 1])) for i in range(len(dirs) - 1)]
    cn = np.array(contrib_norms) if contrib_norms else np.zeros(1)
    c5 = {
        "waist_breaths": list(taps["waist_breaths"]),
        "contrib_norm_per_even_breath": contrib_norms,
        "contrib_norm_cov": float(cn.std() / (cn.mean() + 1e-12)),
        "consecutive_even_breath_dir_cosine": consec_cos,
        "site": "waist-module-output-minus-input, pre-norm_blend",
    }

    # §1A.E.4 + JSD + C6 from snapshots
    ipc_mr, rdr = compute_ipc_rdr_from_snapshots(snapshots, K)
    even_idx = [k for k in range(K) if k % 2 == 0]
    odd_idx  = [k for k in range(K) if k % 2 != 0]
    jsd = compute_latent_jsd_from_snapshots(snapshots)
    c6  = check_c6_z_magnitude(snapshots)

    # §5 persistence extras (2-sample subsets; data already realized)
    xattn_w_sample = [np.asarray(w.numpy(), dtype=np.float32)[:2]
                      for w in taps["xattn_weights"]]                     # K × (2, nh, L, T)
    # WRITE operator observables (#238)
    nb_attn_sample = [np.asarray(w.numpy(), dtype=np.float32)[:2]
                      for w in taps["nb_attn"]]                           # (K-1) × (2, nh, L, S_k)
    nb_slots_np = [np.asarray(s.numpy(), dtype=np.float32)
                   for s in (taps["nb_slots"] or [])]
    nb_slots_finite = bool(all(np.isfinite(s).all() for s in nb_slots_np))
    g_nb = float(model.fg_v200_nb_gate.cast(dtypes.float).realize().numpy().item())
    tree_logits_final = np.asarray(
        tree_logits_history[-1].realize().numpy(), dtype=np.float32)[:2]  # (2, n_var_lat, n_digits, 10)
    calib_per_breath = np.stack(
        [np.asarray(c.realize().numpy(), dtype=np.float32)[:2]
         for c in calib_history], axis=0)                                  # (K, 2)

    return {
        "snapshots": snapshots,
        "jsd": [float(v) for v in jsd],
        "c6": c6,
        "ipc_mr": ipc_mr,
        "rdr": rdr,
        "ipc_mr_even_mean": float(np.mean([ipc_mr[k] for k in even_idx])),
        "ipc_mr_odd_mean":  float(np.mean([ipc_mr[k] for k in odd_idx])) if odd_idx else float('nan'),
        "traj": traj,
        "conc": conc,
        "fam_ent": fam_ent,
        "sa_ent": sa_ent,
        "c5": c5,
        "xattn_weights_sample": xattn_w_sample,
        "tree_logits_final_sample": tree_logits_final,
        "calib_per_breath_sample": calib_per_breath,
        "nb_attn_sample": nb_attn_sample,
        "nb_slot_count": len(nb_slots_np),
        "nb_slots_finite": nb_slots_finite,
        "g_nb": g_nb,
    }


def _check_trajectory_match(
    measured_avg: list[float],
    predicted: list[float] = None,
    tol: float = 0.30,
) -> dict:
    """Check §1A.E.9 trajectory match within +/-30% of predicted.

    #236 recalibrated: 5 checkpoints, post-THINK is NOT a criterion (logged only).
    Criterion checkpoints: 1 (post_norm_breath), 2 (post_READ_add),
                           4 (post_norm_blend), 5 (post_blend).
    post-THINK (index 2) is logged but always treated as PASS (not criterion-gated).

    predicted defaults to [1.0, 2.0, None, 1.0, 1.0] (#236 recalibrated).
    """
    # Recalibrated: post_THINK (index 2) not a criterion — set predicted=None → always PASS
    if predicted is None:
        predicted = [1.0, 2.0, None, 1.0, 1.0]

    checkpoint_names = [
        "post_norm_breath",
        "post_READ_add",
        "post_THINK",        # not a criterion (#236); logged, always PASS
        "post_norm_blend",   # NEW #236
        "post_blend",
    ]

    matches = []
    first_miss = None
    for i, (meas, pred) in enumerate(zip(measured_avg, predicted)):
        if pred is None:
            # post-THINK is not a criterion — always pass, but log actual value
            matches.append(True)
            continue
        lo = pred * (1.0 - tol)
        hi = pred * (1.0 + tol)
        ok = (lo <= meas <= hi)
        matches.append(ok)
        if not ok and first_miss is None:
            first_miss = f"{checkpoint_names[i]} (measured={meas:.2f} vs predicted={pred:.2f})"

    trajectory_match = all(matches)
    return {
        "predicted_trajectory_per_elem":     predicted,
        "measured_trajectory_per_elem":      measured_avg,
        "measured_trajectory_average_breaths": measured_avg,
        "trajectory_match":                  trajectory_match,
        "trajectory_deviation_breath":       first_miss,
        "per_checkpoint_match":              {checkpoint_names[i]: bool(matches[i])
                                              for i in range(len(matches))},
        "post_THINK_not_criterion":          True,
    }


# ---------------------------------------------------------------------------
# §7 Concentration-drift metric (Sub-task 2)
# ---------------------------------------------------------------------------

def _compute_concentration_top10(z_np: np.ndarray) -> float:
    """Compute top-10/2048 dim energy fraction of post-THINK state.

    Args:
      z_np: (B, L, H) float32 array — post-THINK latent state

    Returns:
      float: fraction of squared energy in top-10 dimensions,
             averaged over (batch, latent_position)
    """
    # z_np: (B, L, H)
    B, L, H = z_np.shape
    fracs = []
    # Per (b, l) position
    z_flat = z_np.reshape(B * L, H)   # (B*L, H)
    sq = z_flat ** 2                   # (B*L, H)
    total_energy = sq.sum(axis=-1)     # (B*L,)
    # top-10 indices per sample
    top_k = min(10, H)
    top_sq = np.partition(sq, -top_k, axis=-1)[:, -top_k:]   # (B*L, 10)
    top_energy = top_sq.sum(axis=-1)  # (B*L,)
    frac = top_energy / (total_energy + 1e-8)   # (B*L,)
    return float(frac.mean())


# NOTE (#237): measure_concentration_drift and measure_within_breath_trajectory
# (the #236 breath-loop reimplementations, which silently dropped the topology
# mask) are REMOVED. All probe metrics come from run_instrumented_probe above,
# which uses the tapped canonical forward (§2 single-forward).


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    assert int(os.environ.get("V200_TASK", "0")) > 0, "V200_TASK=1 must be set"

    # ---- Config ----
    K          = int(getenv("V200_K_MAX",       str(V200_K_MAX)))
    BATCH      = int(getenv("BATCH",             "8"))
    STEPS      = int(getenv("STEPS",             "2000"))   # #237 horizon: E.10 reads bind at 1000+
    LR         = float(getenv("LR",              "3e-4"))
    LOG_EVERY  = int(getenv("LOG_EVERY",          "10"))
    PB_EVERY   = int(getenv("PER_BREATH_EVERY",  "50"))
    GC_EVERY   = int(getenv("GC_EVERY",          "50"))
    GRAD_EVERY = int(getenv("GRAD_NORM_EVERY",   "100"))
    EVAL_BATCHES = int(getenv("EVAL_BATCHES",    "8"))
    EVAL_BATCH = int(getenv("EVAL_BATCH",        str(BATCH)))
    SEED       = int(getenv("SEED",              "42"))
    CALIB_W    = float(getenv("V200_CALIB_WEIGHT", "0.05"))
    STAGE2A    = int(os.environ.get("V200_STAGE2A_WAIST", "0")) > 0
    WAIST_DIM  = int(getenv("V200_WAIST_DIM",    str(V200_WAIST_DIM)))
    N_LATENTS  = int(getenv("V200_N_LATENTS",    "32"))
    N_VAR_LAT  = int(getenv("V200_N_VAR_LAT",    str(V200_N_VAR_LAT)))
    N_DIGITS   = int(getenv("V200_N_DIGITS",     str(V200_N_DIGITS)))
    N_MAX      = int(getenv("V200_N_MAX",        str(V200_N_MAX)))
    F_MAX      = int(getenv("V200_F_MAX",        str(V200_F_MAX)))
    GRAD_CLIP  = float(getenv("GRAD_CLIP",       "1.0"))
    B_REF      = int(getenv("B_REF",             "4"))

    TRAIN_PATH = getenv("V200_TRAIN",  ".cache/factor_graph_train.jsonl")
    VAL_PATH   = getenv("V200_VAL",    ".cache/factor_graph_test.jsonl")
    GSM8K_PATH = getenv("V200_GSM8K",  ".cache/gsm8k_factor_graphs_train.jsonl")
    GSM8K_RATIO = float(getenv("V200_GSM8K_RATIO", "0.5"))
    CKPT_DIR   = getenv("CKPT_DIR",    ".cache/v200_perceiver_ckpts")
    CKPT_LABEL = getenv("CKPT_LABEL",  "v200_perceiver_238_write8")
    SMOKE_DIR  = getenv("SMOKE_DIR",   ".cache/v200_smoke")
    REF_DIR    = os.path.join(SMOKE_DIR, "reference_curves")

    SMOKE_LOG_PATH  = os.path.join(SMOKE_DIR, "train_238.log")
    EVAL_JSON_PATH  = os.path.join(SMOKE_DIR, f"step{STEPS}_eval_238.json")
    INSTR_JSON_PATH = os.path.join(SMOKE_DIR, "instrumentation_238.json")
    GRAD_NORMS_PATH = os.path.join(SMOKE_DIR, "grad_norms_238.npz")
    PERSIST_DIR     = os.path.join(SMOKE_DIR, "persistence")
    REF_JSD_PATH    = os.path.join(REF_DIR, "latent_jsd_random_init.npz")

    # #237 eval-checkpoint schedule (E.10 mid-training reads) + §9 dense ckpt cadence
    EVAL_AT = sorted(set(s for s in (200, 500, 1000, 1500, STEPS) if 1 <= s <= STEPS))

    def _ckpt_due(s: int) -> bool:
        # §9: every 100 steps for the first 1K, every 250 after (control-generator run)
        return (s <= 1000 and s % 100 == 0) or (s > 1000 and s % 250 == 0)

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SMOKE_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    os.makedirs(REF_DIR, exist_ok=True)

    # ---- Arch version + metric SHA (§6/§7) ----
    # #237: #236's substrate (prenorm5 + seamthree + gate-2) + mask1a
    #       (§2 per-latent topology mask, partition 1a). §6: new config_sig →
    #       comparison machinery refuses cross-version; reference curves
    #       regenerate under this arch_version below.
    config_sig  = f"K{K}_L{N_LATENTS}_prenorm5_seamthree_gate-2_mask1a_dsnorm-fp32chain-whereguard_write8"
    arch_version = _get_arch_version(config_sig)
    metric_sha   = _get_metric_sha()

    # ---- ADVISORY check (§11) ----
    advisory_deviations: list[str] = []
    if K != 8:
        advisory_deviations.append(f"K={K} (spec K=8)")
    if abs(LR - 3e-4) > 1e-6:
        advisory_deviations.append(f"LR={LR} (spec LR=3e-4)")
    # #237 known spec deviation (§11, surfaced in line 1 — never silent):
    # §2 partition table says per-op latents read "tokens of that op
    # (ADD/SUB/MUL/DIV)"; the implementation assigns token_index mod 4.
    # Pre-pinned consequence (see module docstring): per-op group's
    # entropy-separation read does not bind; per-token + global + C4 carry it.
    advisory_deviations.append(
        "per-op mask family: token_index-mod-4 proxy, not §2 op_type routing"
    )
    is_advisory = len(advisory_deviations) > 0
    advisory_prefix = f"ADVISORY: deviations={advisory_deviations}" if is_advisory else ""

    # ---- Logging ----
    log_fh = open(SMOKE_LOG_PATH, "w", buffering=1)

    def log(msg: str, also_print: bool = True) -> None:
        log_fh.write(msg + "\n")
        if also_print:
            print(msg, flush=True)

    if advisory_prefix:
        log(advisory_prefix)
    log("=" * 72)
    log("v200 #238 — THE WRITE OPERATOR (shared K-slot notebook, §2 WRITE spec)")
    log("  SINGLE VARIABLE vs #237.5: write-once-per-breath slots (fixed address = breath")
    log("  index), read-back via pointer attention over <=7 keys, g_nb ZERO-init (auxiliary;")
    log("  step-0 forward byte-identical to #237.5 by exact-0 multiplier).")
    log("  P-W1 (BINDING, C4' per E.15): the U's tail flattens — erosion absorbed by slots.")
    log("  P-W2: the improvised carrier DECOMMISSIONS (ablation dCE < +0.0164 at 2000).")
    log("  P-W3: g_nb departs zero within ~500 steps (in-run read at grad captures).")
    log("  C4' binds: non-increasing thru k=3 AND descent>=0.01 AND tail<=50% of descent.")
    log("  #237.5 (substrate, no WRITE) is this run's control; same seed/steps/config.")
    log(f"  Masks inherited from #237: latent_topology_mask (24 per-token + 4 per-op + 4 global)")
    log(f"  device={Device.DEFAULT}  B={BATCH}  K={K}  steps={STEPS}  lr={LR}  cold-start")
    log(f"  n_latents={N_LATENTS}  stage2a_waist={STAGE2A}  waist_dim={WAIST_DIM if STAGE2A else 'N/A'}")
    log(f"  arch_version={arch_version}")
    log(f"  metric_sha={metric_sha}")
    log(f"  eval checkpoints: {EVAL_AT}  |  dense ckpts: every 100 to 1K, every 250 after (§9)")
    log(f"  Substrate carried from #236: 6 RMSNorms (breath/read/read_ctx/commit/blend/readout),")
    log(f"  alpha_read init=1.0 (READ = information inlet, §1A.E.8), delta_gate init=-2.0")
    log(f"  STEP-0 GATE (§1A.E.10/§7): per-family entropy < log(support) [1/6/24], per-group never mean;")
    log(f"    all-latent mean strictly < log(24)=3.178. Fail = mask not wired = abort, fix, re-run.")
    log(f"  Pre-committed E.10 reads: ipc_mr stays diverse; per-group entropy separates by 1000+")
    log(f"    (per-op read NON-BINDING per mod-4 ADVISORY; per-token+global+C4 carry it);")
    log(f"    C4 slope <= -0.05 by 1000 — C4 fails again => §1A.E.2 checks MANDATORY (no 4th deferral)")
    log(f"    AND message-passing hypothesis confirmed in pre-committed shape => #238 = THINK quotient mask;")
    log(f"    concentration stays ~4% at post-THINK site (same-site provenance).")
    log(f"  #238-routing thresholds (pre-committed): uniform = sa-entropy std<0.05 nats AND |mean-log31|<0.15")
    log(f"  C5 recalibrated (first deployment): PASS = min contrib_norm>1e-6 AND CoV>0.01 across even breaths")
    log(f"  Predicted trajectory (§1A.E.9, carried from #236; #236 measured post_read=1.30):")
    log(f"    post_norm=1.0  post_read=2.0  post_think=4-50(not criterion)  post_norm_blend=1.0  post_blend=1.0")
    log(f"  Concentration-drift §7 refs: init-Llama=98%, v235=25.8%, v236=4.04%, random=4.8%")
    log("=" * 72)

    np.random.seed(SEED)
    Tensor.manual_seed(SEED)

    # ---- Model setup ----
    log("\nLoading Llama weights...")
    t_load_start = time.time()
    llama32_path = ".cache/llama-3.2-1b-weights/model.safetensors"
    use_llama32  = os.path.exists(llama32_path) and not os.environ.get("FORCE_SMOLLM2", "")
    if use_llama32:
        from tinygrad.nn.state import safe_load as _safe_load_fn
        sd  = _safe_load_fn(llama32_path)
        cfg = LLAMA_3_2_1B_CFG
        log(f"  Loaded Llama-3.2-1B from {llama32_path}")
    else:
        sd  = load_llama_weights()
        cfg = SMOLLM2_1_7B_CFG
        log(f"  Loaded SmolLM2-1.7B (fallback)")

    model = _Obj()
    attach_llama_layers(model, n_layers=4, sd=sd, cfg=cfg)
    del sd
    gc.collect()

    attach_fg_params_v200(
        model, n_latents=N_LATENTS, n_var_lat=N_VAR_LAT, k_max=K,
        n_digits=N_DIGITS, n_max=N_MAX, f_max=F_MAX,
        stage2a_waist=STAGE2A, waist_dim=WAIST_DIM,
    )
    Device[Device.DEFAULT].synchronize()
    _cast_llama_fp32(model)
    t_load = time.time() - t_load_start
    log(f"  Load time: {t_load:.1f}s")

    # ---- Data loaders ----
    synth_loader = FactorGraphLoaderV107(
        TRAIN_PATH, batch_size=BATCH, difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=16, seed=SEED,
    )
    gsm8k_records = load_gsm8k_records_v107(GSM8K_PATH) if os.path.exists(GSM8K_PATH) else []
    dual_loader = DualDataLoaderV107(
        synth_loader, gsm8k_records, gsm8k_ratio=GSM8K_RATIO,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=16, seed=SEED + 1,
    )
    val_loader = FactorGraphLoaderV107(
        VAL_PATH, batch_size=EVAL_BATCH, difficulty_filter=None, curriculum=False,
        n_max=N_MAX, f_max=F_MAX, k_max=K, n_heads=16, seed=SEED + 2,
    )

    # =====================================================================
    # #237 STEP-0 GATE — mask presence + §7 per-group entropy verification
    # =====================================================================

    # Hard assert (audit hardening): no getattr-None fallback for this run.
    assert hasattr(model, "fg_v200_latent_topology_mask"), (
        "#237 requires fg_v200_latent_topology_mask on the model — "
        "attach_fg_params_v200 did not attach it (early-return path?). ABORT."
    )
    mask_np = model.fg_v200_latent_topology_mask.cast(dtypes.float).realize().numpy()
    log(f"\n§2 topology mask attached: shape={mask_np.shape}  "
        f"row-sums per family: per_token={int(mask_np[0:24].sum(1).max())}  "
        f"per_op={int(mask_np[24:28].sum(1).max())}  global={int(mask_np[28:32].sum(1).max())}")

    # Fixed diagnostic batch — used at step 0 AND every eval checkpoint so the
    # instrumentation timeseries is same-batch comparable across training.
    diag_batch = next(val_loader.iter_eval())
    diag_domain_init = diag_batch["domain_init"]
    diag_node_kinds  = diag_batch["node_kinds"]

    log("\nSTEP-0 GATE: random-init tapped forward + per-group entropy assertion...")
    probe0 = run_instrumented_probe(
        model, diag_domain_init, diag_node_kinds, K, N_MAX, F_MAX, STAGE2A,
    )
    step0_verification = verify_topology_mask_step0(probe0["fam_ent"], mask_np)
    log("  STEP-0 MASK VERIFICATION PASSED (per-group, §7):")
    for fam_name, fam_res in step0_verification["families"].items():
        log(f"    {fam_name:10s} support={fam_res['support']:2d}  "
            f"H_breath0={fam_res['entropy_breath0']:.4f}  "
            f"bound=log({fam_res['support']})+eps={fam_res['log_support'] + step0_verification['eps']:.4f}  "
            f"[{'PASS' if fam_res['pass'] else 'FAIL'}]")
    log(f"    all-latent mean H_breath0={step0_verification['all_latent_mean_breath0']:.4f}  "
        f"< log(24)={step0_verification['all_latent_mean_bound_logT']:.4f}  "
        f"[{'PASS' if step0_verification['all_latent_mean_pass'] else 'FAIL'}]")
    log(f"  THINK per-latent entropy at init: mean={probe0['sa_ent']['mean_per_breath'][0]:.4f}  "
        f"std={probe0['sa_ent']['std_per_breath'][0]:.5f}  (ref log(31)={probe0['sa_ent']['ref_log31']:.4f})")

    # WRITE operator step-0 gate (§2 WRITE spec, runtime-verifiable form):
    # (a') g_nb exactly 0.0 → read-back contribution = exact-0 multiplier →
    #      forward identical to #237.5 by construction (bitwise identity of
    #      the gated term verified by the pre-launch review microtest).
    # (b)  write-once: slots appended-never-reassigned by construction;
    #      runtime asserts = K slots written, K-1 read-back attentions fired,
    #      all slot contents finite.
    assert probe0["g_nb"] == 0.0, (
        f"#238 STEP-0 GATE: g_nb must be exactly 0.0 at init (got {probe0['g_nb']}) — ABORT"
    )
    assert probe0["nb_slot_count"] == K, (
        f"#238 STEP-0 GATE: expected {K} slots written, got {probe0['nb_slot_count']} — ABORT"
    )
    assert len(probe0["nb_attn_sample"]) == K - 1, (
        f"#238 STEP-0 GATE: expected {K-1} read-back attentions, got "
        f"{len(probe0['nb_attn_sample'])} — ABORT"
    )
    assert probe0["nb_slots_finite"], "#238 STEP-0 GATE: non-finite slot content — ABORT"
    log(f"  WRITE STEP-0 GATE PASSED: g_nb=0.0 exactly (identity to #237.5 by "
        f"exact-0 multiplier); {K} slots written once; {K-1} read-backs wired; slots finite")

    # =====================================================================
    # SUB-TASK 3: Reference curves (masked architecture — §6 fresh null)
    # =====================================================================

    ref_paths = generate_reference_curves(
        model, val_loader, K=K, B_ref=B_REF,
        n_max=N_MAX, f_max=F_MAX, stage2a_waist=STAGE2A,
        ref_dir=REF_DIR, arch_version=arch_version,
        metric_sha=metric_sha, seed=SEED,
    )
    log(f"\nReference curves saved to {REF_DIR}  (regenerated under {arch_version})")

    # =====================================================================
    # SUB-TASK 4: 200-step training smoke at K=8, LR=3e-4
    # =====================================================================

    log(f"\n{'='*40}\nStarting training ({STEPS} steps)  LR={LR}  K={K}\n{'='*40}\n")

    params   = fg_v200_parameters(model)
    n_params = sum(int(np.prod(t.shape)) for t in params)
    log(f"Trainable params: {n_params/1e6:.1f}M")
    opt = Adam(params, lr=LR, b1=0.9, b2=0.95, eps=1e-8)

    Tensor.training = True
    step_fn = _compile_jit_fg_step_v200(
        model, opt, K=K, B=BATCH,
        n_max=N_MAX, f_max=F_MAX, n_var_lat=N_VAR_LAT, n_digits=N_DIGITS,
        calib_weight=CALIB_W, grad_clip=GRAD_CLIP,
        stage2a_waist=STAGE2A,
    )

    grad_norm_keys = [
        "backbone_L0_L3", "latent_init", "cross_attn",
        "waist_down_proj", "waist_up_proj",
        "tree_readout", "calib_head", "delta_gate", "breath_embed",
        "alpha_read",         # #235: named separately for §10 row 5 strain detection
        "norm_blend_weight",  # #236: Seam 3 learnable gain — strain check
        "notebook",           # #238: WRITE operator
    ]
    grad_norm_history: dict[str, list[float]] = {k: [] for k in grad_norm_keys}
    grad_norm_steps: list[int] = []

    # ---- #237 per-checkpoint instrumentation (E.10 timeseries) ----
    checkpoint_metrics: list[dict] = []
    latest: dict = {}

    def _entry_from_probe(probe: dict, step_num: int, eval_results=None,
                          pb_ce=None, ladder_slope=None) -> dict:
        return {
            "step": step_num,
            "fam_ent": {name: probe["fam_ent"][name] for name, _, _ in V200_MASK_FAMILIES},
            "fam_ent_all_mean": probe["fam_ent"]["all_mean"],
            "sa_ent_mean": probe["sa_ent"]["mean_per_breath"],
            "sa_ent_std": probe["sa_ent"]["std_per_breath"],
            "sa_ent_ref_log31": probe["sa_ent"]["ref_log31"],
            "ipc_mr": probe["ipc_mr"],
            "ipc_mr_even_mean": probe["ipc_mr_even_mean"],
            "ipc_mr_odd_mean": probe["ipc_mr_odd_mean"],
            "rdr": probe["rdr"],
            "jsd": probe["jsd"],
            "c6_ratio": probe["c6"]["ratio"],
            "traj_avg": probe["traj"]["trajectory_avg"],
            "concentration_post_think": probe["conc"]["post_think_top10_frac"],
            "concentration_post_blend": probe["conc"]["post_blend_top10_frac"],
            "concentration_post_think_per_breath": probe["conc"]["post_think_per_breath"],
            "measurement_site_think": probe["conc"]["measurement_site_think"],
            "measurement_site_blend": probe["conc"]["measurement_site_blend"],
            "c5_contrib_norms": probe["c5"]["contrib_norm_per_even_breath"],
            "c5_contrib_cov": probe["c5"]["contrib_norm_cov"],
            "c5_consec_cos": probe["c5"]["consecutive_even_breath_dir_cosine"],
            "eval": eval_results,
            "per_breath_ce_eval": pb_ce,
            "ladder_slope": ladder_slope,
            "g_nb": probe["g_nb"],   # P-W3 engagement, persisted per checkpoint (#238)
        }

    def _write_instr_json() -> None:
        with open(INSTR_JSON_PATH, "w") as f_ij:
            json.dump({
                "run_id": "238",
                "arch_version": arch_version,
                "metric_sha": metric_sha,
                "advisory": advisory_deviations,
                "eval_at": EVAL_AT,
                "diag_batch": "first val_loader.iter_eval() batch, fixed across checkpoints",
                "checkpoints": checkpoint_metrics,
            }, f_ij, indent=2)
        # §6 sidecar (review fix: JSON artifacts get provenance too)
        prov_ij = make_provenance(
            metric="instrumentation_timeseries", units="mixed (see keys)",
            shape=[len(checkpoint_metrics)], ckpt="cold-start-per-checkpoint",
            split="smoke-eval", seed=SEED,
            step=checkpoint_metrics[-1]["step"] if checkpoint_metrics else 0,
            env_vars={"K_MAX": str(K), "STEPS": str(STEPS)},
            output_path=os.path.abspath(INSTR_JSON_PATH), key="checkpoints",
            arch_version=arch_version, metric_sha=metric_sha,
        )
        with open(INSTR_JSON_PATH.replace(".json", ".provenance.json"), "w") as fp:
            json.dump(prov_ij, fp, indent=2)

    def _persist_z(probe: dict, step_num: int) -> None:
        # §5 persistence bundle per eval checkpoint (review fix: widened from
        # z-only to include cross-attn weights, calib, final tree logits, and
        # effective delta_gate — all 2-sample subsets, fp16 where large).
        # NOT included (proxied elsewhere, noted per review): raw read_ctx norm
        # and self-attn residual norm — the §1A.E.9 within-breath scales + rdr
        # cover the same seams.
        z_sample = np.stack([s[:2].astype(np.float32) for s in probe["snapshots"]], axis=0)
        eff_gate = 1.0 / (1.0 + np.exp(-np.asarray(
            model.fg_v200_delta_gate.cast(dtypes.float).realize().numpy(),
            dtype=np.float32)[:K]))
        z_path = os.path.join(PERSIST_DIR, f"step{step_num}_z_238.npz")
        np.savez(
            z_path,
            data=z_sample.astype(np.float16),
            xattn_weights=np.stack(probe["xattn_weights_sample"], axis=0).astype(np.float16),
            tree_logits_final=probe["tree_logits_final_sample"].astype(np.float16),
            calib_per_breath=probe["calib_per_breath_sample"],
            effective_delta_gate=eff_gate,
        )
        prov = make_provenance(
            metric="latent_z_per_breath_plus_bundle", units="float16/float32",
            shape=list(z_sample.shape), ckpt=f"cold-start-step{step_num}",
            split="smoke-eval", seed=SEED, step=step_num,
            env_vars={"K_MAX": str(K), "BATCH": str(BATCH), "STEPS": str(STEPS),
                      "V200_STAGE2A_WAIST": str(int(STAGE2A))},
            output_path=os.path.abspath(z_path), key="data",
            arch_version=arch_version, metric_sha=metric_sha,
        )
        prov["with_what"]["measurement_site"] = "init+post-delta-gate-blend-per-breath"
        prov["what"]["bundle_keys"] = [
            "data (K+1, 2, L, H) z fp16",
            "xattn_weights (K, 2, nh, L, T) fp16",
            "tree_logits_final (2, n_var_lat, n_digits, 10) fp16",
            "calib_per_breath (K, 2) fp32",
            "effective_delta_gate (K,) fp32 sigmoid",
        ]
        with open(z_path.replace(".npz", ".provenance.json"), "w") as fp:
            json.dump(prov, fp, indent=2)

    def _run_checkpoint_instrumentation(step_num: int) -> None:
        t_ck = time.time()
        Tensor.training = False
        probe = run_instrumented_probe(
            model, diag_domain_init, diag_node_kinds, K, N_MAX, F_MAX, STAGE2A,
        )
        eval_results = _evaluate_v200(
            model, val_loader, K=K, n_max=N_MAX, f_max=F_MAX,
            n_var_lat=N_VAR_LAT, n_digits=N_DIGITS,
            max_batches=EVAL_BATCHES, stage2a_waist=STAGE2A,
        )
        pb_ce = _per_breath_ce_at_eval(
            model, val_loader, K=K, n_max=N_MAX, f_max=F_MAX,
            n_var_lat=N_VAR_LAT, n_digits=N_DIGITS,
            max_batches=EVAL_BATCHES, stage2a_waist=STAGE2A,
        )
        ys = np.array(pb_ce, dtype=np.float64)
        if np.all(np.isfinite(ys)) and K >= 2:
            slope = float(np.polyfit(np.arange(K, dtype=np.float64), ys, 1)[0])
        else:
            slope = float('nan')
        entry = _entry_from_probe(probe, step_num, eval_results=eval_results,
                                  pb_ce=pb_ce, ladder_slope=slope)
        entry["c4_prime"] = check_c4_prime(pb_ce)   # E.15, binds from #238
        checkpoint_metrics.append(entry)
        latest["probe"] = probe
        latest["eval_results"] = eval_results
        latest["pb_ce"] = pb_ce
        latest["ladder_slope"] = slope
        _persist_z(probe, step_num)
        _write_instr_json()
        log(f"\n[checkpoint step {step_num}]  ({time.time() - t_ck:.1f}s)")
        for d in DIFFICULTIES:
            if d in eval_results:
                v = eval_results[d]
                log(f"  val[{d:6s}]: cell={v['cell_acc']:.3f}  q={v['query_acc']:.3f}")
        c4p_ck = entry["c4_prime"]
        log(f"  ladder_slope={slope:.5f} (legacy)  C4'={'PASS' if c4p_ck['pass'] else 'FAIL'} "
            f"(shape={c4p_ck['shape_ok']} descent={c4p_ck['descent']:.4f} "
            f"tail={c4p_ck['tail_frac']*100:.0f}%)  "
            f"per_breath_ce={[f'{v:.4f}' for v in pb_ce]}")
        log("  fam_ent[breath-mean]: " + "  ".join(
            f"{name}={float(np.mean(entry['fam_ent'][name])):.4f}"
            for name, _, _ in V200_MASK_FAMILIES))
        log(f"  sa_ent: mean={float(np.mean(entry['sa_ent_mean'])):.4f}  "
            f"std={float(np.mean(entry['sa_ent_std'])):.5f}  "
            f"(ref log31={probe['sa_ent']['ref_log31']:.3f})")
        log(f"  ipc_mr even={entry['ipc_mr_even_mean']:.4f}  odd={entry['ipc_mr_odd_mean']:.4f}  "
            f"conc_think={entry['concentration_post_think']:.4f}  c6_ratio={entry['c6_ratio']:.3f}")
        log(f"  c5: norms={[f'{v:.4f}' for v in entry['c5_contrib_norms']]}  "
            f"cov={entry['c5_contrib_cov']:.4f}  "
            f"consec_cos={[f'{v:.3f}' for v in entry['c5_consec_cos']]}")
        Tensor.training = True
        gc.collect()

    # Step-0 entry from the gate probe (no eval — model is random init)
    checkpoint_metrics.append(_entry_from_probe(probe0, 0))
    _persist_z(probe0, 0)
    _write_instr_json()

    # §9 review fix: dense cadence "from step 0" — save the init checkpoint too
    # (seed-deterministic, but a saved artifact beats a reconstruction argument).
    ck0_path = os.path.join(CKPT_DIR, f"{CKPT_LABEL}_step0.safetensors")
    safe_save(fg_v200_state_dict(model), ck0_path)
    log(f"[ckpt] saved {ck0_path} (step-0 init state)")

    # NaN-skip deferral flags (review fix: a skip at a due step defers the
    # ckpt/eval to the next healthy step instead of silently dropping it)
    deferred_ckpt = False
    deferred_eval = False

    log_loss = log_ce = log_calib = log_n = 0.0
    all_losses: list[float] = []
    start_loss = end_loss = None
    t0 = time.time()
    nan_skip_count = 0

    for step in range(1, STEPS + 1):
        Tensor.training = True
        batch = dual_loader.sample_batch(step=step)
        domain_init   = batch["domain_init"]
        node_kinds    = batch["node_kinds"]
        gold_bins_np  = batch["gold_bins"].realize().numpy()
        obs_mask      = batch["observed_mask"]
        n_vars_np     = batch["n_vars_total"]
        gold_digits_np = bins_to_digits_msd(gold_bins_np, n_digits=N_DIGITS)
        gold_digits_t = Tensor(
            gold_digits_np.astype(np.int64), dtype=dtypes.int,
        ).contiguous().realize()
        n_vars_mask_t = _make_n_vars_mask(n_vars_np, min(N_VAR_LAT, N_MAX))

        # Grad norm capture
        if (step % GRAD_EVERY == 0) or (step == STEPS and step % GRAD_EVERY != 0):
            Tensor.training = True
            norms = _eager_grad_norm_step(
                model, domain_init, node_kinds, gold_digits_t, obs_mask,
                n_vars_mask_t, opt, K=K, n_max=N_MAX, f_max=F_MAX,
                n_var_lat=N_VAR_LAT, n_digits=N_DIGITS,
                calib_weight=CALIB_W, stage2a_waist=STAGE2A,
            )
            for k_name in grad_norm_keys:
                grad_norm_history[k_name].append(norms.get(k_name, 0.0))
            grad_norm_steps.append(step)
            # P1 in-run read (§1A.E.14): the starvation signature was exact
            # zero by step 600 in #237 — watch it live, every capture
            p1_be = norms.get("breath_embed", 0.0)
            p1_ar = norms.get("alpha_read", 0.0)
            p1_bb = norms.get("backbone_L0_L3", 0.0)
            p1_ok = (p1_be > 1e-5) and (p1_ar > 1e-5)
            log(f"  [P1 §E.14] breath_embed={p1_be:.2e}  alpha_read={p1_ar:.2e}  "
                f"backbone={p1_bb:.2e}  hold>1e-5: "
                f"{'PASS' if p1_ok else 'FAIL — starvation recurring'}")
            # P-W3 (#238): notebook engagement — g_nb departs zero by ~500
            g_nb_val = float(model.fg_v200_nb_gate.cast(dtypes.float)
                             .realize().numpy().item())
            nb_grad = norms.get("notebook", 0.0)
            log(f"  [P-W3 #238] g_nb={g_nb_val:+.6f}  notebook_grad={nb_grad:.2e}  "
                f"({'ENGAGED' if abs(g_nb_val) > 1e-4 else 'closed'})")
            _gn_data = {k: np.array(v) for k, v in grad_norm_history.items()}
            _gn_data["steps"] = np.array(grad_norm_steps)
            np.savez(GRAD_NORMS_PATH, **_gn_data)

        # JIT training step
        Tensor.training = True
        outs = step_fn(domain_init, node_kinds, gold_digits_t, obs_mask, n_vars_mask_t)
        total_t, healthy_t = outs[0], outs[1]
        ce_t, calib_t      = outs[2], outs[3]
        cell_acc_t         = outs[4]
        pb_ce_ts           = outs[6: 6 + K]

        loss_val = float(total_t.numpy())
        ce_val   = float(ce_t.numpy())
        healthy  = float(healthy_t.numpy())

        if healthy < 0.5:
            nan_skip_count += 1
            log(f"[NaN-skip] step {step}  (total so far: {nan_skip_count})")
            if nan_skip_count >= 10 and LR >= 3e-4:
                log(f"[WARNING] >=10 NaN skips at spec LR={LR} — substrate fix may be incomplete")
            # Review fix: a NaN-skip must not silently drop a due dense ckpt or
            # E.10 eval checkpoint — defer to the next healthy step.
            if _ckpt_due(step):
                deferred_ckpt = True
            if step in EVAL_AT:
                deferred_eval = True
            continue

        all_losses.append(loss_val)
        if start_loss is None:
            start_loss = loss_val
        end_loss = loss_val
        log_loss  += loss_val
        log_ce    += ce_val
        log_calib += float(calib_t.numpy())
        log_n     += 1

        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            ca = float(cell_acc_t.numpy())
            log(
                f"[step {step:5d}] loss={log_loss/log_n:.4f}  "
                f"ce={log_ce/log_n:.4f}  calib={log_calib/log_n:.4f}  "
                f"cell_acc={ca:.3f}  ({dt:.1f}s  {dt/step:.2f}s/step)"
            )
            log_loss = log_ce = log_calib = log_n = 0.0

        if step % PB_EVERY == 0:
            pb_ce = [float(t.numpy()) for t in pb_ce_ts]
            pb_str = " ".join(f"{v:.3f}" for v in pb_ce)
            log(f"  per_breath_ce[k=0..{K-1}]: {pb_str}")
            if K > 1:
                slope_approx = (pb_ce[-1] - pb_ce[0]) / (K - 1)
                log(f"  [LADDER] slope approx={slope_approx:.4f}  (target <= -0.05)")

        if step % GC_EVERY == 0:
            gc.collect()

        # §9 dense checkpoint cadence (every 100 to 1K, every 250 after) —
        # this run is the control-generator; retroactive persistence impossible
        if _ckpt_due(step) or deferred_ckpt:
            ck_path = os.path.join(CKPT_DIR, f"{CKPT_LABEL}_step{step}.safetensors")
            safe_save(fg_v200_state_dict(model), ck_path)
            log(f"  [ckpt] saved {ck_path}" + ("  (deferred from NaN-skip step)" if deferred_ckpt and not _ckpt_due(step) else ""))
            deferred_ckpt = False

        # E.10 eval checkpoints (incl. the final step)
        if step in EVAL_AT or deferred_eval:
            _run_checkpoint_instrumentation(step)
            deferred_eval = False

    # Guard: a NaN-skip `continue` at an EVAL_AT step would silently drop that
    # checkpoint — ensure the final-step instrumentation always exists.
    if not checkpoint_metrics or checkpoint_metrics[-1]["step"] != STEPS:
        log(f"[guard] final checkpoint instrumentation missing (NaN-skip at an "
            f"eval step?) — running at step {STEPS} now")
        _run_checkpoint_instrumentation(STEPS)

    t_total = time.time() - t0
    log(f"\nTraining complete. {STEPS} steps ({nan_skip_count} NaN-skipped) in "
        f"{t_total:.1f}s ({t_total/STEPS:.2f}s/step)")

    # Save checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"{CKPT_LABEL}_step{STEPS}.safetensors")
    safe_save(fg_v200_state_dict(model), ckpt_path)
    log(f"Saved checkpoint: {ckpt_path}")

    # Flush grad norms with provenance
    _gn_data = {k: np.array(v) for k, v in grad_norm_history.items()}
    _gn_data["steps"] = np.array(grad_norm_steps)
    np.savez(GRAD_NORMS_PATH, **_gn_data)
    prov_gn = make_provenance(
        metric="per_param_group_grad_l2_norms",
        units="L2 norm (float)",
        shape=[len(grad_norm_steps), len(grad_norm_keys)],
        ckpt=f"cold-start-step{STEPS}",
        split="smoke-train",
        seed=SEED,
        step=STEPS,
        env_vars={"GRAD_NORM_EVERY": str(GRAD_EVERY)},
        output_path=os.path.abspath(GRAD_NORMS_PATH),
        key="<per-group keys>",
        arch_version=arch_version,
        metric_sha=metric_sha,
    )
    with open(GRAD_NORMS_PATH.replace(".npz", ".provenance.json"), "w") as f_gn:
        json.dump(prov_gn, f_gn, indent=2)

    # ---- Final reads come from the in-loop checkpoint at step==STEPS ----
    # (one tapped canonical forward per checkpoint; no separate probe passes)
    assert "probe" in latest, (
        "final checkpoint instrumentation did not run — STEPS must be in EVAL_AT"
    )
    final_probe  = latest["probe"]
    eval_results = latest["eval_results"]
    pb_ce_eval   = latest["pb_ce"]
    ladder_slope = latest["ladder_slope"]

    log("\nFinal eval (from in-loop checkpoint at step %d):" % STEPS)
    for d in DIFFICULTIES:
        if d not in eval_results:
            continue
        v = eval_results[d]
        pp = " ".join(f"{p:.3f}" for p in v["per_pos_acc"])
        log(f"  val[{d:6s}]: cell={v['cell_acc']:.3f}  q={v['query_acc']:.3f}  "
            f"digit={v['digit_acc']:.3f}  per_pos=[{pp}]  n={v['n_puzzles']}")
    log(f"  per_breath_ce (eval): {' '.join(f'{v:.4f}' for v in pb_ce_eval)}")
    log(f"  ladder slope (linear fit): {ladder_slope:.5f}  (criterion: <= -0.05)")

    trained_snapshots = final_probe["snapshots"]
    trained_jsd = final_probe["jsd"]
    log(f"\nLatent JSD trained trajectory:  {[f'{v:.5f}' for v in trained_jsd]}")

    # Load reference JSD for Criterion 2 (regenerated under mask1a arch above).
    # §6/§7 review fix: refuse mismatched arch_version/metric_sha via the
    # provenance sidecar — a stale (unmasked-arch) null must fail loudly.
    ref_jsd = None
    if os.path.exists(REF_JSD_PATH):
        ref_prov_path = REF_JSD_PATH.replace(".npz", ".provenance.json")
        with open(ref_prov_path) as f_rp:
            ref_prov = json.load(f_rp)
        ref_arch = ref_prov.get("with_what", {}).get("arch_version", "missing")
        ref_msha = ref_prov.get("what", {}).get("metric_sha", "missing")
        assert ref_arch == arch_version, (
            f"C2 reference arch_version mismatch: ref={ref_arch!r} vs run={arch_version!r} "
            "— stale null; §6 refuses cross-version comparison"
        )
        assert ref_msha == metric_sha, (
            f"C2 reference metric_sha mismatch: ref={ref_msha!r} vs run={metric_sha!r} "
            "— §7 JSD method identity violated"
        )
        ref_data = np.load(REF_JSD_PATH)
        ref_jsd  = ref_data["data"].tolist()
        log(f"Ref JSD random-init trajectory: {[f'{v:.5f}' for v in ref_jsd]}")
        log(f"  ref provenance verified: arch_version + metric_sha match")
    else:
        log(f"WARNING: reference JSD not found at {REF_JSD_PATH}")

    # C6: z-magnitude bounded
    log("\nChecking Criterion 6 — z-magnitude bounded...")
    c6_result = final_probe["c6"]
    log(f"  z_k per breath: {[f'{v:.3f}' for v in c6_result['per_breath']]}")
    log(f"  max/min ratio: {c6_result['ratio']:.3f}  (criterion: < 3.0)  [{c6_result['verdict']}]")
    log(f"  Note: with norm_blend in place, post-blend ~1 by construction → C6 expected PASS trivially")

    # ====================================================================
    # §1A.E.9 QUANTITATIVE TRAJECTORY MATCH (5-checkpoint within-breath, #236)
    # ====================================================================
    log("\n§1A.E.9 — Five-checkpoint within-breath per-element scale trajectory:")
    log("  Predicted (carried from #236; #236 measured post_read=1.30):")
    log("    post_norm=1.0  post_read=2.0  post_think=4-50(not criterion)  post_norm_blend=1.0  post_blend=1.0")
    log("  Tolerance: +/-30% on criterion checkpoints; post_think logged but not gated")
    traj_result = final_probe["traj"]
    traj_avg = traj_result["trajectory_avg"]
    log(f"  Measured avg over K breaths:")
    log(f"    post_norm={traj_avg[0]:.3f}  post_read={traj_avg[1]:.3f}  "
        f"post_think={traj_avg[2]:.3f}(informational)  post_norm_blend={traj_avg[3]:.3f}  "
        f"post_blend={traj_avg[4]:.3f}")
    log(f"  Per-breath trajectory:")
    for k_idx, row in enumerate(traj_result["trajectory_per_breath"]):
        log(f"    k={k_idx}: norm={row[0]:.3f}  read={row[1]:.3f}  "
            f"think={row[2]:.3f}  norm_blend={row[3]:.3f}  blend={row[4]:.3f}")
    traj_check = _check_trajectory_match(traj_avg)
    log(f"  Trajectory match (criterion checkpoints only): {traj_check['trajectory_match']}")
    if not traj_check["trajectory_match"]:
        log(f"  First miss: {traj_check['trajectory_deviation_breath']}")
    for ckpt, ok in traj_check["per_checkpoint_match"].items():
        is_crit = (ckpt != "post_THINK")
        suffix = "" if is_crit else " (informational only)"
        log(f"    {ckpt}: {'PASS' if ok else 'FAIL'}{suffix}")

    # ====================================================================
    # §1A.C §10 row 5 strain detection: alpha_read + norm_blend.weight
    # ====================================================================
    log("\n§10 strain detection — alpha_read scalar (§10 row 5):")
    alpha_val = float(model.fg_v200_alpha_read.cast(dtypes.float).realize().numpy().item())
    alpha_grad_traj = grad_norm_history.get("alpha_read", [])
    log(f"  alpha_read at step {STEPS}: {alpha_val:.6f}  (strain threshold: > 5.0)")
    log(f"  alpha_read grad norm trajectory: {[f'{v:.6f}' for v in alpha_grad_traj]}")
    if alpha_val > 5.0:
        strain_signal = f"STRAIN: alpha_read={alpha_val:.3f} > 5 at step {STEPS}"
        log(f"  {strain_signal}  -> queue §10 row 5 (per-breath alpha)")
    elif alpha_grad_traj and len(alpha_grad_traj) >= 2:
        diffs = [alpha_grad_traj[i+1] - alpha_grad_traj[i] for i in range(len(alpha_grad_traj)-1)]
        alternating = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
        max_ratio = max(alpha_grad_traj) / (min(alpha_grad_traj) + 1e-10) if min(alpha_grad_traj) > 0 else 0
        if alternating >= len(diffs) // 2 or max_ratio > 10:
            strain_signal = f"STRAIN: alpha_read grad alternating/wild (alternating={alternating}/{len(diffs)}, max_ratio={max_ratio:.1f})"
            log(f"  {strain_signal}  -> queue §10 row 5 (per-breath alpha)")
        else:
            strain_signal = "NO_STRAIN"
            log(f"  No strain detected — §10 row 5 stays queued but not promoted")
    else:
        strain_signal = "NO_STRAIN"
        log(f"  No strain detected — §10 row 5 stays queued but not promoted")

    log("\n§10 strain detection — norm_blend.weight scalar (#236 new):")
    blend_norm_val = None
    blend_norm_strain = "N/A"
    if hasattr(model, "fg_v200_blend_norm_w"):
        bnw = model.fg_v200_blend_norm_w.cast(dtypes.float).realize().numpy()
        blend_norm_mean = float(bnw.mean())
        blend_norm_std  = float(bnw.std())
        blend_norm_val  = {"mean": blend_norm_mean, "std": blend_norm_std}
        blend_norm_grad_traj = grad_norm_history.get("norm_blend_weight", [])
        log(f"  norm_blend.weight at step {STEPS}: mean={blend_norm_mean:.6f}  std={blend_norm_std:.6f}")
        log(f"  (expected near mean=1.0, std~0 at init; deviation = gain compensating post-THINK state)")
        log(f"  norm_blend grad norm trajectory: {[f'{v:.6f}' for v in blend_norm_grad_traj]}")
        if abs(blend_norm_mean - 1.0) > 0.5:
            blend_norm_strain = f"STRAIN: blend_norm_mean={blend_norm_mean:.3f} (>0.5 from 1.0)"
            log(f"  {blend_norm_strain}")
        else:
            blend_norm_strain = "NO_STRAIN"
            log(f"  No strain — gain near 1.0, seam is absorbing post-THINK state cleanly")
    else:
        log("  WARNING: fg_v200_blend_norm_w not found on model")

    # §1A.E.4 — inter-position cosine mean-removed + read dominance (from final probe)
    log("\nComputing §1A.E.4 position-collapse metrics...")
    ipc_mr_trained = final_probe["ipc_mr"]
    rdr_trained    = final_probe["rdr"]
    ipc_mr_arr_t = np.array(ipc_mr_trained)
    rdr_arr_t    = np.array(rdr_trained)
    log(f"  ipc_mr trained: {[f'{v:.5f}' for v in ipc_mr_trained]}")
    log(f"  rdr    trained: {[f'{v:.4f}' for v in rdr_trained]}")

    e4_result = read_e4_grid(ipc_mr_arr_t, rdr_arr_t)
    log(f"\n§1A.E.4 GRID CELL: {e4_result['cell']}")
    log(f"  Interpretation: {e4_result['interpretation']}")
    log(f"  Next move:      {e4_result['next_move']}")

    # ====================================================================
    # §7 Concentration-drift metric (from final probe; per-checkpoint
    # trajectory lives in instrumentation_237_5.json)
    # ====================================================================
    conc_result = final_probe["conc"]
    log(f"\n§7 Concentration-drift — top-10/2048 dim energy fraction at step {STEPS}:")
    log(f"  post-THINK  top-10/2048 frac: {conc_result['post_think_top10_frac']:.4f}  "
        f"(ref: init-Llama=0.98, v235=0.258, v236=0.0404, random=0.048)")
    log(f"  post-blend  top-10/2048 frac: {conc_result['post_blend_top10_frac']:.4f}")
    log(f"  post-THINK per breath: {[f'{v:.4f}' for v in conc_result['post_think_per_breath']]}")
    log(f"  post-blend per breath: {[f'{v:.4f}' for v in conc_result['post_blend_per_breath']]}")
    log(f"  sites: think={conc_result['measurement_site_think']}  blend={conc_result['measurement_site_blend']}")
    conc_traj = [(e["step"], e["concentration_post_think"]) for e in checkpoint_metrics]
    log(f"  §7 slow-drift trajectory (step, post_think_frac): {[(s, f'{v:.4f}') for s, v in conc_traj]}")

    # up_proj norm (Criterion 3)
    up_proj_norm = 0.0
    if STAGE2A and hasattr(model, "fg_v200_W_expand"):
        wp = model.fg_v200_W_expand.cast(dtypes.float).realize().numpy()
        up_proj_norm = float(np.sqrt((wp ** 2).sum()))
    log(f"\nup_proj (W_expand) L2 norm at step {STEPS}: {up_proj_norm:.6f}  (criterion: > 1e-4)")
    wup_grad_traj = grad_norm_history.get("waist_up_proj", [])
    log(f"  waist_up_proj grad norm trajectory: {[f'{v:.5f}' for v in wup_grad_traj]}")

    # Persistence bundle: z traces are saved per eval checkpoint by
    # _persist_z (persistence/step{N}_z_238.npz + provenance), incl. step 0
    # and the final step — no separate end-of-run save needed.
    log(f"\nPersistence: z bundles at {PERSIST_DIR}/step{{N}}_z_237.npz for N in [0] + {EVAL_AT}")

    # =====================================================================
    # SUB-TASK 5: §1A.B Criteria 1-6 + §1A.E reading
    # =====================================================================

    log("\n" + "=" * 40)
    log("TRAINING CONTRACT (§1A.B) VERIFICATION — ALL 6 CRITERIA")
    log("=" * 40)

    criteria_passed = []

    # C1: Loss monotonically decreasing
    log("\nCriterion 1 — Loss monotonically decreasing:")
    log(f"  start={start_loss:.5f}  end={end_loss:.5f}")
    if len(all_losses) >= 10:
        smooth = [float(np.mean(all_losses[max(0, i-5): i+5])) for i in range(len(all_losses))]
        log(f"  smoothed first 5: {[f'{v:.4f}' for v in smooth[:5]]}")
        log(f"  smoothed last 5:  {[f'{v:.4f}' for v in smooth[-5:]]}")
        crit1 = (smooth[-1] < smooth[0]) if smooth else False
    else:
        crit1 = (end_loss < start_loss) if (start_loss and end_loss) else False
    log(f"  VERDICT: {'YES' if crit1 else 'NO'}")
    criteria_passed.append(crit1)

    # C2: Latent JSD departs from reference
    log("\nCriterion 2 — Latent JSD departs from random-init reference:")
    log(f"  trained trajectory:   {[f'{v:.5f}' for v in trained_jsd]}")
    if ref_jsd is not None:
        log(f"  reference trajectory: {[f'{v:.5f}' for v in ref_jsd]}")
        min_len = min(len(trained_jsd), len(ref_jsd))
        tr = np.array(trained_jsd[:min_len])
        rf = np.array(ref_jsd[:min_len])
        if min_len >= 3:
            spear = float(spearmanr(tr, rf).correlation)
        else:
            spear = 1.0
        max_dep = float(np.max(np.abs(tr - rf)))
        ref_range = float(np.max(rf) - np.min(rf)) if len(rf) > 1 else 1e-8
        ratio_dep = max_dep / (ref_range + 1e-8)
        log(f"  Spearman correlation: {spear:.4f}  (PASS if < 0.9)")
        log(f"  max-abs-departure: {max_dep:.5f}  ref range: {ref_range:.5f}")
        log(f"  ratio: {ratio_dep:.3f}  (PASS if > 0.3)")
        if len(tr) > 0 and tr[0] > 0:
            eps_tr = 0.05 * tr[0]
            freeze_tr = next((i+1 for i, v in enumerate(tr) if v <= eps_tr), len(tr))
        else:
            freeze_tr = 0
        if ref_jsd and ref_jsd[0] > 0:
            eps_rf = 0.05 * ref_jsd[0]
            freeze_rf = next((i+1 for i, v in enumerate(ref_jsd[:min_len]) if v <= eps_rf), min_len)
        else:
            freeze_rf = 0
        log(f"  freeze_trained={freeze_tr}  freeze_ref={freeze_rf}  "
            f"half_K={K//2}")
        direction_ok = (freeze_tr >= freeze_rf)
        gate_b_ok    = (freeze_tr >= K // 2)
        magnitude_ok = (spear < 0.9) or (ratio_dep > 0.3)
        crit2 = magnitude_ok and direction_ok and gate_b_ok
        log(f"  magnitude_ok={magnitude_ok}  direction_ok={direction_ok}  gate_b_ok={gate_b_ok}")
    else:
        log("  reference not available — checking non-monotone only")
        diffs = [trained_jsd[i+1] - trained_jsd[i] for i in range(len(trained_jsd)-1)]
        crit2 = any(d > 0 for d in diffs) and any(d < 0 for d in diffs)
        spear = float('nan'); ratio_dep = float('nan')
    log(f"  VERDICT: {'YES' if crit2 else 'NO'}")
    criteria_passed.append(crit2)

    # C3: up_proj norm
    log("\nCriterion 3 — up_proj norm has moved off zero:")
    log(f"  step {STEPS} norm: {up_proj_norm:.6f}  grad-norm traj: {wup_grad_traj}")
    if STAGE2A:
        crit3 = (up_proj_norm > 1e-4) or (any(v > 0 for v in wup_grad_traj))
    else:
        crit3 = True
        log("  (no waist — C3 marked YES)")
    log(f"  VERDICT: {'YES' if crit3 else 'NO'}")
    criteria_passed.append(crit3)

    # C4′ (§1A.E.15, BINDS from #238 — pre-registered before any WRITE data):
    # shape (non-increasing thru k=3) AND descent >= 0.01 AND tail <= 50% of
    # descent. Binds at the step-1000+ checkpoint per the E.10 binding-site
    # rule; final reported as confirmatory. Legacy slope reported alongside.
    log("\nCriterion 4′ — shape + magnitude + tail (E.15; binds at step 1000):")
    c4_bind_entry = next(
        (e for e in checkpoint_metrics
         if e["step"] >= 1000 and e.get("c4_prime") is not None),
        None,
    )
    if c4_bind_entry is not None:
        c4_bind_step = c4_bind_entry["step"]
        c4p = c4_bind_entry["c4_prime"]
    else:
        c4_bind_step = STEPS
        c4p = check_c4_prime(pb_ce_eval)
        log("  WARNING: no checkpoint at step >= 1000 — binding on final step instead")
    log(f"  binding read (step {c4_bind_step}): shape={c4p['shape_ok']}  "
        f"descent={c4p['descent']:.4f} (>=0.01: {c4p['magnitude_ok']})  "
        f"tail={c4p['tail_rise']:.4f} ({c4p['tail_frac']*100:.0f}% of descent, "
        f"<=50%: {c4p['tail_ok']})")
    c4p_final = check_c4_prime(pb_ce_eval)
    log(f"  final (step {STEPS}, confirmatory): pass={c4p_final['pass']}  "
        f"descent={c4p_final['descent']:.4f}  tail_frac={c4p_final['tail_frac']*100:.0f}%")
    log(f"  per-breath CE (final): {[f'{v:.4f}' for v in pb_ce_eval]}")
    log(f"  ladder_slope_legacy (final): {ladder_slope:.5f}  (continuity only, does not bind)")
    slope_traj = [(e["step"], e["ladder_slope"]) for e in checkpoint_metrics
                  if e["ladder_slope"] is not None]
    log(f"  legacy slope trajectory: {[(s, f'{v:.5f}') for s, v in slope_traj]}")
    crit4 = bool(c4p["pass"])
    c4_bind_slope = c4_bind_entry["ladder_slope"] if c4_bind_entry else ladder_slope
    log(f"  VERDICT: {'YES' if crit4 else 'NO'}")
    criteria_passed.append(crit4)

    # C5: Waist alternation — RECALIBRATED metric (§1A.B.5, locked Jun 11,
    # first deployment in #237). The old even/odd 10× magnitude ratio is a dead
    # instrument on the bounded substrate (norm_blend erases the signal).
    # New metric: pre-norm waist contribution ‖α·(h_compressed − z_w)‖ read at
    # the waist module output BEFORE norm_blend (from the tapped forward).
    # Pre-committed thresholds (set before the run; see module docstring):
    #   PASS = min contrib_norm > 1e-6 AND CoV across even breaths > 0.01.
    # Secondary (logged, not gated): consecutive even-breath delta-direction
    # cosine — waist should produce direction-varying contributions.
    log("\nCriterion 5 — Waist contribution (RECALIBRATED, pre-norm site):")
    if STAGE2A:
        c5_data = final_probe["c5"]
        contrib_norms = c5_data["contrib_norm_per_even_breath"]
        contrib_cov   = c5_data["contrib_norm_cov"]
        consec_cos    = c5_data["consecutive_even_breath_dir_cosine"]
        log(f"  waist breaths: {c5_data['waist_breaths']}")
        log(f"  contrib norms (pre-norm site): {[f'{v:.5f}' for v in contrib_norms]}")
        log(f"  CoV across even breaths: {contrib_cov:.5f}  (pre-committed: > 0.01)")
        log(f"  consecutive even-breath direction cosines: {[f'{v:.4f}' for v in consec_cos]}")
        log(f"  site: {c5_data['site']}")
        nonzero_ok = bool(min(contrib_norms) > 1e-6) if contrib_norms else False
        varies_ok  = bool(contrib_cov > 0.01)
        crit5 = nonzero_ok and varies_ok
        log(f"  non-zero={nonzero_ok}  varies(CoV>0.01)={varies_ok}")
        log(f"  note: first deployment of the recalibrated metric — no prior-run reference;"
            f" values persisted per checkpoint in instrumentation_237_5.json")
    else:
        crit5 = True
        contrib_norms, contrib_cov, consec_cos = [], 0.0, []
        log("  (no waist — C5 marked YES)")
    log(f"  VERDICT: {'YES' if crit5 else 'NO'}")
    criteria_passed.append(crit5)

    # C6: z-magnitude bounded — expected PASS trivially with norm_blend
    log("\nCriterion 6 — z-magnitude bounded (max/min ratio < 3.0):")
    log(f"  per-breath z_k: {[f'{v:.3f}' for v in c6_result['per_breath']]}")
    log(f"  ratio: {c6_result['ratio']:.3f}  (criterion: < 3.0)")
    log(f"  §1A.B.6 note: with norm_blend, post-blend state is ~1 by RMSNorm construction")
    crit6 = c6_result["ratio"] < 3.0
    log(f"  VERDICT: {'YES' if crit6 else 'NO'}")
    criteria_passed.append(crit6)

    # ---- Summary ----
    n_pass = sum(int(c) for c in criteria_passed)
    log("\n" + "=" * 40)
    log(f"PASSED: {n_pass} / 6")

    # ====================================================================
    # §1A.E.10 pre-committed masks reads + #238 routing (#237)
    # ====================================================================
    log("\n" + "=" * 40)
    log("§1A.E.10 PRE-COMMITTED MASKS READS")
    log("=" * 40)
    step0_entry = checkpoint_metrics[0]
    final_entry = checkpoint_metrics[-1]
    trained_entries = [e for e in checkpoint_metrics if e["step"] > 0]

    # Read 1 — mean-removed inter-position cosine stays diverse (~-0.03 to 0.3)
    max_ipc_final = float(np.max(np.abs(ipc_mr_arr_t)))
    e10_read1_diverse = bool(max_ipc_final <= 0.8)
    log(f"\nRead 1 — ipc_mr stays diverse: max|ipc_mr|={max_ipc_final:.4f}  "
        f"[{'DIVERSE' if e10_read1_diverse else 'COLLAPSED — masks broke something; '
           'route component-specific'}]")

    # Read 2 — per-group entropy separation across training
    # (per-op group NON-BINDING per mod-4 ADVISORY, pre-pinned before the run;
    #  binding separation signal = global group sharpening >= 0.15 nats)
    e10_sep = {}
    for fam_name, _, _ in V200_MASK_FAMILIES:
        h0 = float(np.mean(step0_entry["fam_ent"][fam_name]))
        hf = float(np.mean(final_entry["fam_ent"][fam_name]))
        e10_sep[fam_name] = {"step0": h0, "final": hf, "drop": h0 - hf}
    e10_read2_separating = bool(e10_sep["global"]["drop"] >= 0.15)
    log(f"\nRead 2 — per-group entropy separation (step0 → step{STEPS}):")
    for fam_name, vals in e10_sep.items():
        binding = " (NON-BINDING: mod-4 ADVISORY)" if fam_name == "per_op" else ""
        log(f"  {fam_name:10s}: {vals['step0']:.4f} → {vals['final']:.4f}  "
            f"drop={vals['drop']:+.4f}{binding}")
    log(f"  binding signal (global drop >= 0.15 nats): "
        f"[{'SEPARATING — masks helping' if e10_read2_separating else 'UNIFORM — masks decorative (per binding signal)'}]")

    # Read 3 — C4 routing (the sharpest line; crit4 computed above)
    if crit4:
        c4_routing = (
            "C4 GREEN — ladder story closes; masks differentiated per-breath state. "
            "Pre-registered message-passing hypothesis DIES CHEAP (its falsification "
            "condition fired; see memory/project_v200_message_passing_hypothesis_jun11.md)."
        )
    else:
        c4_routing = (
            "C4 FAILED AGAIN (4th run) — §1A.E.2 component checks are NOW MANDATORY "
            "(per-breath weighted CE wiring; readout-receives-per-breath-state). Three "
            "deferrals was the limit. With step-0 wiring PASSED, the pre-registered "
            "message-passing hypothesis is confirmed in pre-committed shape if entropy "
            "separates and ipc_mr stays diverse → #238 = THINK quotient-graph mask "
            "(single variable, same partition source). NOTE: this run is ADVISORY "
            "(mod-4 per-op proxy) — bindingness of the C4 read awaits the §11 "
            "resolution (brief blesses proxy, or at-spec re-run)."
        )
    log(f"\nRead 3 — C4 routing: {c4_routing}")

    # Read 4 — concentration drift stays bounded (~4% regime, same site)
    conc_final_think = final_entry["concentration_post_think"]
    e10_read4_bounded = bool(conc_final_think < 0.10)
    log(f"\nRead 4 — concentration bounded: final post-THINK={conc_final_think:.4f}  "
        f"trajectory={[(e['step'], round(e['concentration_post_think'], 4)) for e in checkpoint_metrics]}")
    log(f"  [{'BOUNDED ~4% — registers-not-tokens thesis holds' if e10_read4_bounded else 'CLIMBING — thesis erosion, drift toward Llama-native 98%'}]")

    # E.4 pre-commit — even-breath cosine creep (waist-consensus channel)
    creep_detected = False
    if len(trained_entries) >= 2:
        even_first = trained_entries[0]["ipc_mr_even_mean"]
        even_final = trained_entries[-1]["ipc_mr_even_mean"]
        odd_first  = trained_entries[0]["ipc_mr_odd_mean"]
        odd_final  = trained_entries[-1]["ipc_mr_odd_mean"]
        creep_detected = bool((even_final - even_first > 0.2)
                              and (abs(odd_final - odd_first) < 0.1))
        log(f"\nEven-breath cosine creep (E.4 pre-commit): "
            f"even {even_first:.4f}→{even_final:.4f}  odd {odd_first:.4f}→{odd_final:.4f}")
        log(f"  [{'CREEP — waist-induced consensus; routes to waist width/gating, NOT masks/READ' if creep_detected else 'no creep'}]")

    # #238 routing — per-latent THINK entropy (pre-committed thresholds:
    # uniform = std < 0.05 nats AND |mean - log31| < 0.15 at final checkpoint)
    sa_std_final  = float(np.mean(final_entry["sa_ent_std"]))
    sa_mean_final = float(np.mean(final_entry["sa_ent_mean"]))
    log31_ref     = float(final_entry["sa_ent_ref_log31"])
    think_uniform = bool((sa_std_final < 0.05) and (abs(sa_mean_final - log31_ref) < 0.15))
    log(f"\n#238 routing — per-latent THINK attention entropy at step {STEPS}:")
    log(f"  mean={sa_mean_final:.4f}  std-across-latents={sa_std_final:.5f}  ref log(31)={log31_ref:.4f}")
    log(f"  trajectory (step, mean, std): "
        f"{[(e['step'], round(float(np.mean(e['sa_ent_mean'])), 4), round(float(np.mean(e['sa_ent_std'])), 5)) for e in checkpoint_metrics]}")
    if think_uniform:
        log("  [UNIFORM MEAN-FIELD CONSENSUS MIXING — if C4 also failed, #238 = THINK "
            "quotient-graph mask is the pre-registered next single variable]")
    else:
        log("  [DIFFERENTIATED — inter-latent selectivity emerging without explicit "
            "structure; THINK quotient-graph mask may not be needed]")

    hard_cell = eval_results.get("hard", {}).get("cell_acc", float('nan'))
    chain_saturation = 0.376

    traj_ok_str = "PASS" if traj_check["trajectory_match"] else f"FAIL:{traj_check['trajectory_deviation_breath']}"
    # §11: any spec deviation → the verdict line itself reads ADVISORY first.
    adv_line_prefix = f"ADVISORY: {'; '.join(advisory_deviations)} | " if advisory_deviations else ""
    common_tail = (
        f"steps={STEPS}  loss={end_loss:.4f}  "
        f"hard_cell={hard_cell:.3f}  "
        f"ladder_slope={ladder_slope:.4f}  "
        f"up_proj_norm={up_proj_norm:.2e}  "
        f"c6_ratio={c6_result['ratio']:.2f}  "
        f"e4_cell={e4_result['cell']!r}  "
        f"traj={traj_ok_str}  "
        f"alpha_read={alpha_val:.3f}  "
        f"alpha_strain={strain_signal}  "
        f"blend_norm_strain={blend_norm_strain}  "
        f"conc_think={conc_result['post_think_top10_frac']:.3f}  "
        f"conc_blend={conc_result['post_blend_top10_frac']:.3f}  "
        f"step0_mask=PASS  "
        f"e10_sep_global={'YES' if e10_read2_separating else 'NO'}  "
        f"think_uniform={'YES' if think_uniform else 'NO'}  "
        f"creep={'YES' if creep_detected else 'NO'}"
    )
    if n_pass == 6:
        final_line = (
            f"{adv_line_prefix}#238 WRITE RUN PASSED  run=238  " + common_tail + "  all6=YES"
        )
    else:
        failed = [f"C{i+1}" for i, c in enumerate(criteria_passed) if not c]
        final_line = (
            f"{adv_line_prefix}#238 WRITE RUN FAILED  run=238  failed={','.join(failed)}  "
            + common_tail
        )

    log(final_line)

    # ---- Save eval JSON ----
    eval_json = {
        "step": STEPS,
        "arch_version": arch_version,
        "metric_sha": metric_sha,
        "run_id": "238",
        "advisory": advisory_deviations,
        "advisory_note": (
            "mod-4 per-op mask proxy (§11): pass/fail bindingness awaits user "
            "resolution — bless proxy in brief §2, or re-run at-spec with true "
            "op_type routing. Runtime config identical under both; artifacts valid."
        ),
        "step0_mask_verification": step0_verification,
        "e10_reads": {
            "read1_ipc_mr_diverse": e10_read1_diverse,
            "read1_max_abs_ipc_mr": max_ipc_final,
            "read2_per_group_separation": e10_sep,
            "read2_separating_binding_global": e10_read2_separating,
            "read2_per_op_nonbinding_reason": "token_index-mod-4 proxy (ADVISORY)",
            "read3_c4_routing": c4_routing,
            "read4_concentration_bounded": e10_read4_bounded,
            "even_breath_cosine_creep": creep_detected,
        },
        "routing_238": {
            "think_entropy_mean_final": sa_mean_final,
            "think_entropy_std_final": sa_std_final,
            "ref_log31": log31_ref,
            "pre_committed_thresholds": "uniform = std<0.05 AND |mean-log31|<0.15",
            "think_uniform_mean_field": think_uniform,
            "hypothesis_memo": "memory/project_v200_message_passing_hypothesis_jun11.md",
        },
        "c5_recalibrated": {
            "contrib_norm_per_even_breath": contrib_norms,
            "contrib_norm_cov": contrib_cov,
            "consecutive_even_breath_dir_cosine": consec_cos,
            "site": "waist-module-output-minus-input, pre-norm_blend",
            "pre_committed_thresholds": "PASS = min>1e-6 AND CoV>0.01",
            "first_deployment": True,
        },
        "drift_floor_status": "borrowed_from_v110-step3",
        "drift_floor": 0.02,
        "instrumentation_timeseries": INSTR_JSON_PATH,
        "cell_acc":   {d: eval_results.get(d, {}).get("cell_acc", float('nan'))
                       for d in DIFFICULTIES},
        "query_acc":  {d: eval_results.get(d, {}).get("query_acc", float('nan'))
                       for d in DIFFICULTIES},
        "digit_acc":  {d: eval_results.get(d, {}).get("digit_acc", float('nan'))
                       for d in DIFFICULTIES},
        "per_pos_acc": {d: eval_results.get(d, {}).get("per_pos_acc", [])
                        for d in DIFFICULTIES},
        "per_breath_ce_eval": pb_ce_eval,
        "ladder_slope": ladder_slope,
        "c4_binding": {
            "binds_at_step": c4_bind_step,
            "binding_slope_legacy": float(c4_bind_slope),
            "c4_prime_binding": c4p,
            "c4_prime_final": c4p_final,
            "final_slope_confirmatory": ladder_slope,
            "pre_commit": "E.15 C4-prime: shape + descent>=0.01 + tail<=50% (binds from #238); legacy slope reported for continuity",
        },
        "latent_jsd_trained": trained_jsd,
        "latent_jsd_reference": ref_jsd,
        "up_proj_l2_norm_step200": up_proj_norm,
        "c6_z_magnitude": c6_result,
        "e4_grid": e4_result,
        # §1A.E.9 quantitative trajectory match (#236: 5 checkpoints, post_THINK not criterion)
        "predicted_trajectory_per_elem":     traj_check["predicted_trajectory_per_elem"],
        "measured_trajectory_per_elem":      traj_check["measured_trajectory_per_elem"],
        "measured_trajectory_average_breaths": traj_check["measured_trajectory_average_breaths"],
        "trajectory_match":                  traj_check["trajectory_match"],
        "trajectory_deviation_breath":       traj_check["trajectory_deviation_breath"],
        "trajectory_per_checkpoint_match":   traj_check["per_checkpoint_match"],
        "trajectory_per_breath":             traj_result["trajectory_per_breath"],
        "post_THINK_not_criterion":          True,
        # §10 row 5 strain detection for alpha_read
        "alpha_read_at_final":               alpha_val,
        "alpha_read_grad_norm_traj":         alpha_grad_traj,
        "alpha_read_strain_signal":          strain_signal,
        "alpha_read_strain_requires_row5":   (strain_signal != "NO_STRAIN"),
        # Seam 3 norm_blend.weight strain detection
        "norm_blend_weight_at_final":        blend_norm_val,
        "norm_blend_strain_signal":          blend_norm_strain,
        # §7 Concentration-drift metric — sites + per-checkpoint trajectory
        "concentration_drift": {
            **conc_result,
            "step": STEPS,
            "trajectory_step_vs_post_think": [
                [e["step"], e["concentration_post_think"]] for e in checkpoint_metrics
            ],
        },
        "criteria": {
            "C1_loss_decreasing": bool(criteria_passed[0]),
            "C2_jsd_departs":     bool(criteria_passed[1]),
            "C3_up_proj_nonzero": bool(criteria_passed[2]),
            "C4_ladder_slope":    bool(criteria_passed[3]),
            "C5_waist_alternation": bool(criteria_passed[4]),
            "C6_z_magnitude_bounded": bool(criteria_passed[5]),
            "n_pass": n_pass,
            "n_total": 6,
        },
        "cont_control": {
            "chain_saturation": chain_saturation,
            "metric_minus_chain_saturation_hard": (
                hard_cell - chain_saturation if not np.isnan(hard_cell) else None
            ),
            "drift_floor_status": "borrowed_from_v110-step3",
            "note": ("Drift floor ±0.02 BORROWED from v110-step3 (v200's own floor "
                     "UNMEASURED until two same-arch continuation segments exist). "
                     "#237 masks run; cold-start 2000 steps vs full chain saturation."),
        },
        "nan_skip_count": nan_skip_count,
    }
    with open(EVAL_JSON_PATH, "w") as f_ev:
        json.dump(eval_json, f_ev, indent=2)
    # §6 sidecar (review fix: JSON artifacts get provenance too)
    prov_ev = make_provenance(
        metric="eval_bundle", units="mixed (see keys)",
        shape=[1], ckpt=f"cold-start-step{STEPS}",
        split="smoke-eval", seed=SEED, step=STEPS,
        env_vars={"K_MAX": str(K), "BATCH": str(BATCH), "STEPS": str(STEPS)},
        output_path=os.path.abspath(EVAL_JSON_PATH), key="criteria",
        arch_version=arch_version, metric_sha=metric_sha,
    )
    with open(EVAL_JSON_PATH.replace(".json", ".provenance.json"), "w") as fp:
        json.dump(prov_ev, fp, indent=2)

    log(f"\nSaved eval JSON: {EVAL_JSON_PATH}")
    log_fh.flush()
    log_fh.close()

    print(f"\n{final_line}")
    print(f"§1A.E.4 cell: {e4_result['cell']}")
    print(f"§1A.E.9 trajectory match: {traj_check['trajectory_match']}  "
          f"avg={[f'{v:.2f}' for v in traj_avg]}")
    print(f"alpha_read final={alpha_val:.4f}  strain={strain_signal}")
    print(f"norm_blend strain={blend_norm_strain}  val={blend_norm_val}")
    print(f"§7 concentration-drift: post_think={conc_result['post_think_top10_frac']:.4f}  "
          f"post_blend={conc_result['post_blend_top10_frac']:.4f}")
    print(f"#238 routing: think_uniform={think_uniform}  (std={sa_std_final:.5f}, mean={sa_mean_final:.4f})")
    print(f"Log: {SMOKE_LOG_PATH}")
    print(f"Eval JSON: {EVAL_JSON_PATH}")
    print(f"Instrumentation timeseries: {INSTR_JSON_PATH}")
    print(f"Grad norms: {GRAD_NORMS_PATH}")
    print(f"Latent z bundles: {PERSIST_DIR}/step{{N}}_z_238.npz")


if __name__ == "__main__":
    main()
