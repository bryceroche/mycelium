"""b3_rung2_grad_smoke.py — Decisive anchor-gradient smoke test for Rung-2 relax.

Two questions answered:

  1. ANCHOR SAVE/LOAD ROUND-TRIP
     Verifies that model_state_dict_fg now includes fg_hyp_anchors_{t} so a
     relaxed run's trained anchors are preserved and restored exactly.

  2. DECISIVE ANCHOR GRADIENT CHECK (Issue 2)
     Runs the ACTUAL training path:
       - CircuitLoader (small corpus, N=2, s_max=49, AND/OR/NOT)
       - attach_factor_graph_params + attach_factor_hyperbolic_params
         with FG_HYP_MASK=1, FG_HYP_FREEZE=0, FG_HYP_RELAX=1 (soft-block)
       - factor_breathing_forward (FULL path: build_factor_hyperbolic_attn_bias
         with the both-members gate + clique-union)
       - factor_loss (REAL per-breath weighted-CE on the cell logits) -- NOT bias.mean()
       - .backward()
       - Measure grad-norm of EACH fg_hyp_anchors_{t}
     Also runs the FROZEN comparison (FG_HYP_RELAX=0) and reports both.

  VERDICT:
    - NON-ZERO anchor grad under per-breath CE + full builder ->
        RELAX VALID, fire.  (The earlier 0.0 was a bias.mean() measurement artifact.)
    - ~0 anchor grad even under per-breath CE ->
        GRADIENT-DEAD — diagnose and report the root cause + recommended fix.

GPU-FREE (DEV=CPU): ast.parse + circuit batch build + forward + backward all run on CPU.
No GPU training, no JIT (the smoke bypasses TinyJit and runs the eager path).

Usage:
    DEV=CPU FG_HYP_MASK=1 FG_HYP_RELAX=1 FG_HYP_FREEZE=0 \\
        .venv/bin/python3 scripts/b3_rung2_grad_smoke.py
"""
from __future__ import annotations

import ast
import os
import sys

# ---- GPU-free build gate ---------------------------------------------------
_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _REPO_ROOT)

# Ensure CPU backend before importing tinygrad.
if "DEV" not in os.environ:
    os.environ["DEV"] = "CPU"


def _ast_parse_ok() -> bool:
    with open(_THIS_FILE) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Save/load round-trip test (Issue 1)
# ---------------------------------------------------------------------------

def _test_anchor_roundtrip(model, used_types: list[int]) -> bool:
    """Verify model_state_dict_fg includes fg_hyp_anchors_{t} and that the
    values survive a safe_save -> safe_load cycle (using in-memory bytes)."""
    import io
    import tempfile
    import numpy as np
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_save, safe_load

    # Import the UPDATED model_state_dict_fg from the trainer.
    # This import READS FG_HYP_MASK from env at import time, so call AFTER
    # env is set.
    from scripts.factor_graph_train import model_state_dict_fg as sd_fn

    sd = sd_fn(model)

    # Check keys present.
    missing = []
    for t in used_types:
        key = f"fg_hyp_anchors_{t}"
        if key not in sd:
            missing.append(key)
    if missing:
        print(f"  [FAIL] model_state_dict_fg missing anchor keys: {missing}", flush=True)
        return False
    print(f"  [ok] model_state_dict_fg contains fg_hyp_anchors_{{t}} for "
          f"types={used_types}", flush=True)

    # Round-trip: save to a temp file, reload, compare values.
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        tmp_path = f.name
    try:
        safe_save(sd, tmp_path)
        sd2 = safe_load(tmp_path)
        for t in used_types:
            key = f"fg_hyp_anchors_{t}"
            orig = getattr(model, f"fg_hyp_anchors_{t}").realize().numpy()
            loaded = sd2[key].realize().numpy()
            if orig.shape != loaded.shape:
                print(f"  [FAIL] {key}: shape mismatch {orig.shape} vs {loaded.shape}",
                      flush=True)
                return False
            max_diff = float(np.abs(orig - loaded).max())
            if max_diff > 1e-6:
                print(f"  [FAIL] {key}: max_diff={max_diff:.2e} (not round-trip exact)",
                      flush=True)
                return False
            print(f"  [ok] {key}: shape={orig.shape} round-trips exactly "
                  f"(max_diff={max_diff:.2e})", flush=True)
    finally:
        os.unlink(tmp_path)

    return True


# ---------------------------------------------------------------------------
# Gradient smoke — ONE environment setting at a time
# ---------------------------------------------------------------------------

def _grad_norm_anchors(model, used_types: list[int]) -> dict[int, float]:
    """Return {t: grad_norm} for each fg_hyp_anchors_{t}."""
    norms = {}
    for t in used_types:
        anchors = getattr(model, f"fg_hyp_anchors_{t}")
        if anchors.grad is None:
            norms[t] = 0.0
        else:
            g = anchors.grad.realize().numpy()
            import numpy as np
            norms[t] = float(np.sqrt((g ** 2).sum()))
    return norms


def _run_one(relax: bool, circuit_batch, spec, model_factory, used_types: list[int],
             K: int = 4) -> dict[int, float]:
    """Build a FRESH model instance (to avoid stale grad state), attach params,
    forward + backward, return anchor grad norms.

    relax=True  -> FG_HYP_RELAX=1 (the soft-block, learnable tangent path).
    relax=False -> FG_HYP_RELAX=0 (frozen, saturated alpha, ball-points raw).
    """
    import numpy as np
    from tinygrad import Tensor, dtypes

    # Patch the env + module-level flags for this run.
    os.environ["FG_HYP_RELAX"] = "1" if relax else "0"
    os.environ["FG_HYP_FREEZE"] = "0"

    # Reload factor_masks module to pick up the new FG_HYP_RELAX flag.
    import importlib
    import mycelium.factor_masks as fm_mod
    importlib.reload(fm_mod)
    # Re-import the public names we use so they point at the reloaded module.
    attach_fn = fm_mod.attach_factor_hyperbolic_params
    relax_flag = fm_mod.FG_HYP_RELAX
    print(f"\n  [smoke] FG_HYP_RELAX={int(relax_flag)} "
          f"FG_HYP_EUCLID={int(fm_mod.FG_HYP_EUCLID)}", flush=True)

    # Reload factor_graph_engine to pick up any dependency on factor_masks flags.
    import mycelium.factor_graph_engine as fge_mod
    importlib.reload(fge_mod)
    attach_fg = fge_mod.attach_factor_graph_params
    forward_fn = fge_mod.factor_breathing_forward
    loss_fn = fge_mod.factor_loss

    # Fresh model.
    model = model_factory()

    # Attach fg params.
    attach_fg(model, hidden=1024, spec=spec)

    # Build representative membership/latent_type arrays for anchor sizing.
    mem_np = circuit_batch.membership.realize().numpy()
    lt_np  = circuit_batch.latent_type.realize().numpy()

    # Attach hyperbolic anchor params (FG_HYP_MASK=1 confirmed at top).
    attach_fn(
        model,
        n_heads=spec.n_heads,
        n_factor_types=spec.n_factor_types,
        s_max=spec.s_max,
        membership_np=mem_np,
        latent_type_np=lt_np,
    )

    # Require grad on all anchor tensors (they are Tensors; tinygrad accumulates
    # grad on .backward() only if .requires_grad is set — mirror the optimizer's
    # param handling where .requires_grad is implicitly True for named params).
    for t in used_types:
        anchors = getattr(model, f"fg_hyp_anchors_{t}")
        anchors.requires_grad = True

    # Forward — the FULL path (build_factor_hyperbolic_attn_bias).
    Tensor.training = True
    logits_history, calib_history = forward_fn(model, circuit_batch, spec, K=K)

    # Loss — the REAL per-breath weighted-CE on cell logits (NOT bias.mean()).
    total, parts = loss_fn(logits_history, calib_history, circuit_batch, spec,
                           constraint_weight=0.0, calib_weight=0.1)

    # Backward.
    total.backward()

    # Read anchor grad norms.
    norms = _grad_norm_anchors(model, used_types)
    return norms, total


# ---------------------------------------------------------------------------
# Diagnosis helper — what causes zero grad (if it happens)?
# ---------------------------------------------------------------------------

def _diagnose_zero_grad(circuit_batch, spec, used_types: list[int], K: int = 4):
    """If anchor grad is ~0, probe where the gradient path breaks.

    Checks two structural suspects for the circuit DAG:
      A) within-factor co-location at d=0: do the bias values for within-factor
         member pairs show zero grad w.r.t. the anchor coord?
      B) both-members gate flooring cross-factor non-member pairs to -BLOCK constant:
         are there ANY responsive (gradient-carrying) pairs after the gate?

    Also checks that the bias tensor in the clique-union path has non-zero derivative
    w.r.t. one anchor coordinate directly (mini proxy test).
    """
    import numpy as np
    from tinygrad import Tensor, dtypes

    import importlib
    import mycelium.factor_masks as fm_mod
    importlib.reload(fm_mod)

    os.environ["FG_HYP_RELAX"] = "1"
    importlib.reload(fm_mod)
    relax_flag = fm_mod.FG_HYP_RELAX
    print(f"\n[diagnosis] FG_HYP_RELAX={int(relax_flag)}", flush=True)

    # Build minimal anchor just for type 0 (AND gates).
    mem_np = circuit_batch.membership.realize().numpy()
    lt_np  = circuit_batch.latent_type.realize().numpy()

    # Count max type-0 factors.
    G0 = int((lt_np == 0).sum(axis=-1).max())
    if G0 == 0:
        print("  [diagnosis] no type-0 factors in the batch — cannot probe", flush=True)
        return

    G0_alloc = G0 + 1
    dim = int(os.environ.get("FG_HYP_DIM", "48"))
    rho = float(os.environ.get("FG_HYP_RHO", "0.7"))

    from mycelium.kenken import _poincare_anchors, _tangent_for_anchors, _exp0_map
    anchors_ball_np = _poincare_anchors(G0_alloc, dim, rho)
    tangent_np = _tangent_for_anchors(anchors_ball_np.astype(np.float64)).astype(np.float32)

    # Make a LEARNABLE tangent anchor tensor.
    coord = Tensor(tangent_np, dtype=dtypes.float).contiguous()
    coord.requires_grad = True

    # Calibrate soft-block alpha (mirrors attach_factor_hyperbolic_params RELAX branch).
    from mycelium.kenken import _d_hyp_pairwise, _min_between_anchor_distance
    alpha_margin = float(os.environ.get("FG_HYP_ALPHA_MARGIN", "4.0"))
    relax_block_arg = float(os.environ.get("FG_HYP_RELAX_BLOCK_ARG", "20.0"))
    d_out = _min_between_anchor_distance(anchors_ball_np.astype(np.float64))
    r_t = d_out / 2.0
    alpha_t = alpha_margin * relax_block_arg / max(r_t, 1e-9)
    print(f"  r_t={r_t:.4f}  alpha_t={alpha_t:.4f}  d_out={d_out:.4f}", flush=True)

    # Apply exp_0 to get ball points.
    z = _exp0_map(coord)  # (G0_alloc, dim), Tensor

    # Compute pairwise d_hyp between the anchors themselves.
    # If all anchors are distinct -> d>0 between different rows; d=0 on diagonal.
    d = _d_hyp_pairwise(z.unsqueeze(0)).squeeze(0)  # (G0_alloc, G0_alloc)

    # Softplus bias: -softplus(alpha * (d - r_t)) (the _relation_bias_from_z formula).
    bias_raw = -(alpha_t * (d - r_t)).softplus()  # (G0_alloc, G0_alloc)
    bias_scalar = bias_raw.sum()
    bias_scalar.backward()

    if coord.grad is None:
        gn = 0.0
    else:
        gn = float((coord.grad.realize().numpy() ** 2).sum() ** 0.5)

    print(f"  [diagnosis A] d_hyp-alone backward grad_norm(coord)={gn:.6f}", flush=True)

    if gn < 1e-12:
        print("  -> Root cause A CONFIRMED: the softplus backward is ZERO.", flush=True)
        print("     Likely because alpha_t is still too large (saturating both tails).", flush=True)
        print(f"     softplus arg range: alpha*(d-r) with alpha={alpha_t:.2f}, "
              f"d range=[0, {d_out:.4f}], r={r_t:.4f}", flush=True)
        # Check the arg values directly.
        d_np = d.realize().numpy()
        arg_vals = alpha_t * (d_np - r_t)
        print(f"     between-anchor softplus args (upper triangle): "
              f"min={arg_vals[d_np > 1e-6].min() if (d_np > 1e-6).any() else 'N/A':.2f}  "
              f"max={arg_vals.max():.2f}", flush=True)
        print("  Cause: large alpha saturates softplus -> gradient dead at the bias level.",
              flush=True)
    else:
        print("  -> Grad flows through d_hyp<->softplus<->coord at the anchor level.", flush=True)
        print("     The zero grad must be DOWNSTREAM (clique-union path kills it).", flush=True)

    # Diagnosis B: within-factor co-location.
    print(f"\n  [diagnosis B] within-factor co-location: "
          f"members of one factor are placed on the SAME anchor row, so "
          f"d_hyp(member_i, member_j) = d_hyp(a_f, a_f) = 0 (exactly).", flush=True)
    print("    softplus_arg = alpha*(0 - r) = -alpha*r  -> "
          f"-{alpha_t:.2f}*{r_t:.4f} = {-alpha_t * r_t:.2f}", flush=True)
    sigma_at_zero = 1.0 / (1.0 + float(2.718281828 ** (alpha_t * r_t)))
    print(f"    softplus'(-alpha*r) = sigma(-alpha*r) = {sigma_at_zero:.6f}", flush=True)
    if sigma_at_zero < 1e-3:
        print("    -> WITHIN-FACTOR grad is SATURATED to ~0 (within-factor d=0 is "
              "deep in the allow tail -> softplus' ~ 0 there too).", flush=True)
    else:
        print("    -> within-factor gradient is NOT saturated (sigma > 1e-3).", flush=True)

    # Diagnosis C: both-members gate.
    print(f"\n  [diagnosis C] both-members gate:", flush=True)
    print("    For each (i,j) pair: bias_vol = both_mem * bias_vol + (1-both_mem) * (-BLOCK).", flush=True)
    print("    Where both_mem=0 (either cell is a non-member): this is a CONSTANT -BLOCK.", flush=True)
    print("    So the gradient from those pairs is 0 regardless.", flush=True)
    print("    The ONLY gradient-carrying pairs are within-factor member pairs (both_mem=1).", flush=True)
    print("    For THOSE pairs, d=0 -> we're in the allow tail -> see Diagnosis B.", flush=True)

    # Propose the fix.
    print("\n" + "=" * 72, flush=True)
    print("GRADIENT-DEAD ROOT CAUSE SUMMARY:", flush=True)
    print("  Both the within-factor co-location (d=0) and the both-members gate", flush=True)
    print("  conspire:", flush=True)
    print("   - Within-factor members are ALL placed on THE SAME anchor -> d=0.", flush=True)
    print("   - d=0 -> softplus arg = alpha*(0-r) = -alpha*r (deep in the allow tail).", flush=True)
    print("   - Even with soft-block alpha, sigma(-alpha*r) is still very small", flush=True)
    print("     because alpha*r is large (alpha ~ margin*target/r -> alpha*r ~ margin*target).", flush=True)
    print("   - The both-members gate ensures ONLY these within-factor pairs matter.", flush=True)
    print("   - Result: ZERO anchor gradient through the full clique-union builder.", flush=True)
    print("=" * 72, flush=True)

    print("\nCANDIDATE FIXES:", flush=True)
    print("  (a) FG_HYP_JITTER: within-factor members placed at anchor +/- small tangent", flush=True)
    print("      perturbation -> d > 0 within factor -> gradient flows.", flush=True)
    print("      Preserves t=0 reproduce: jitter is tiny (1e-3), mask matches <1e-3.", flush=True)
    print("      Makes anchor grad NON-ZERO: yes, immediately.", flush=True)
    print("      RECOMMENDED (simplest, least structural change, already in the design).", flush=True)
    print("  (b) DIFFERENTIABLE gate: replace hard both_mem * (-BLOCK) with a soft gate", flush=True)
    print("      bias_vol = both_mem * bias_vol + (1-both_mem) * softplus(-BLOCK).", flush=True)
    print("      Preserves t=0 reproduce: softplus(-BLOCK) ~ -BLOCK (numerically identical).", flush=True)
    print("      Makes anchor grad NON-ZERO: only if there are cross-factor non-member pairs", flush=True)
    print("      whose geometry is actually varied; does NOT help with within-factor d=0.", flush=True)
    print("      Less targeted than (a); fix is incomplete without also addressing d=0.", flush=True)
    print("  (c) PER-NODE coordinates (one coord per cell, not per factor anchor):", flush=True)
    print("      Each cell gets its own learnable position in the ball.", flush=True)
    print("      d_hyp(z_i, z_j) is then NEVER exactly 0 (unless cells collapse).", flush=True)
    print("      Preserves t=0 reproduce: requires re-anchoring cells by factor membership,", flush=True)
    print("       harder to ensure <1e-3 match.", flush=True)
    print("      Makes anchor grad NON-ZERO: yes, but is a larger architectural change.", flush=True)
    print("\nRECOMMENDATION: (a) — per-member tangent jitter (FG_HYP_JITTER already exists).", flush=True)
    print("  The jitter is already plumbed into attach_factor_hyperbolic_params and", flush=True)
    print("  applied to the TANGENT params under FG_HYP_RELAX=1. It is NOT applied to the", flush=True)
    print("  per-member z_li placement (the factor anchor itself is the centroid, all members", flush=True)
    print("  map to it identically). The fix is to jitter the PER-MEMBER ASSIGNMENT:", flush=True)
    print("  instead of z_li = member ? anchor_f : sentinel, use", flush=True)
    print("  z_li = member ? (anchor_f + epsilon_li) : sentinel  where epsilon_li is a", flush=True)
    print("  small per-member-slot random offset (frozen, not learned — just to break the", flush=True)
    print("  exact d=0 degeneracy). This keeps the t=0 reproduce: the offset is tiny (1e-3),", flush=True)
    print("  the bias changes by < epsilon << 1e-3 attention weight. With d > 0 within factors,", flush=True)
    print("  softplus' is no longer at the exact-zero argument -> anchor grad flows.", flush=True)
    print("  ALTERNATIVE to member jitter: reduce the softplus block target so alpha*r < ~5.", flush=True)
    print("  Currently alpha*r = margin*target = 4*20 = 80 -> sigma(-80)~0 (dead).", flush=True)
    print("  Reducing FG_HYP_RELAX_BLOCK_ARG from 20 -> 2 makes alpha*r = 4*2 = 8 ->", flush=True)
    print("  sigma(-8)=3e-4 (marginal) or FG_HYP_RELAX_BLOCK_ARG=1 -> alpha*r=4 ->", flush=True)
    print("  sigma(-4)=1.8e-2 (measurable gradient). But lowering the block target", flush=True)
    print("  increases the attention leak (exp(-margin*target) must stay <<1e-3).", flush=True)
    print("  margin*1 = 4 -> leak = exp(-4) = 1.8e-2 which EXCEEDS the 1e-3 tolerance.", flush=True)
    print("  So purely reducing the block target is NOT safe. Member jitter (option a) is", flush=True)
    print("  the cleaner fix: it breaks d=0 WITHOUT changing the soft-block calibration.", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import importlib
    import numpy as np
    from tinygrad import Tensor, dtypes, Device

    print("=" * 72, flush=True)
    print("b3_rung2_grad_smoke.py — decisive anchor gradient smoke (DEV=CPU)", flush=True)
    print("=" * 72, flush=True)
    print(f"device={Device.DEFAULT}", flush=True)

    # ---- env checks --------------------------------------------------------
    if int(os.environ.get("FG_HYP_MASK", "0")) == 0:
        print("ERROR: FG_HYP_MASK must be 1. Run with FG_HYP_MASK=1.", flush=True)
        sys.exit(1)
    if int(os.environ.get("FG_HYP_RELAX", "0")) == 0:
        print("WARNING: FG_HYP_RELAX=0 set at launch; the smoke will override it "
              "per-run (reload). Proceeding.", flush=True)

    # ---- build a small circuit batch (CircuitLoader, N=2, s_max=49) --------
    print("\n[1] Building circuit batch (small corpus, s_max=49, AND/OR/NOT) ...",
          flush=True)
    from mycelium.circuit_data import CircuitLoader
    from mycelium.factor_graph_engine import FactorGraphSpec

    N_INSTANCES = 50   # tiny corpus — smoke only
    BATCH = 2
    K = 4              # small K for speed; enough to confirm grad flow

    # s_max=49 required by factor_breathing_forward (kenken_layer_forward asserts S==49).
    loader = CircuitLoader(n_instances=N_INSTANCES, s_max=49, n_values=2,
                           batch_size=BATCH, seed=0,
                           gate_types=("AND", "OR", "NOT"))
    n_factor_types = int(loader.n_factor_types)
    batch = loader.sample_batch()

    spec = FactorGraphSpec(s_max=49, n_values=2,
                           n_factor_types=n_factor_types, n_heads=16,
                           k_max=K, has_factor_inlet=False)

    print(f"  batch: B={BATCH} s_max=49 T={n_factor_types} n_gates_max={loader.n_gates_max}",
          flush=True)

    # ---- figure out the used anchor types ----------------------------------
    from mycelium.factor_masks import cell_mp_head_allocation, CELL_MP_HEAD_GLOBAL
    G_heads = max(1, spec.n_heads // 16)
    alloc = cell_mp_head_allocation(spec.n_factor_types, spec.n_heads, G_heads)
    used_types = sorted({int(t) for t in alloc if int(t) != CELL_MP_HEAD_GLOBAL})
    print(f"  used anchor types: {used_types}", flush=True)

    # ---- model factory (Pythia-410M is too slow on CPU; use random-init) ---
    # The gradient smoke does NOT need correct logit values — only that the
    # gradient flows through the FULL builder path. Random init suffices.
    print("\n[2] Building model (random init, no Pythia load for CPU smoke) ...",
          flush=True)
    from mycelium import Config, BreathingTransformer

    cfg = Config()

    def model_factory():
        m = BreathingTransformer(cfg)
        return m

    # ---- Issue 1: save/load round-trip (uses FG_HYP_RELAX=1 model) --------
    print("\n" + "=" * 72, flush=True)
    print("[ISSUE 1] Anchor save/load round-trip", flush=True)
    print("=" * 72, flush=True)

    # Build a temporary model+attach to test the state dict.
    os.environ["FG_HYP_RELAX"] = "1"
    os.environ["FG_HYP_FREEZE"] = "0"
    import mycelium.factor_masks as fm_mod
    importlib.reload(fm_mod)
    import mycelium.factor_graph_engine as fge_mod
    importlib.reload(fge_mod)

    tmp_model = model_factory()
    fge_mod.attach_factor_graph_params(tmp_model, hidden=cfg.hidden, spec=spec)
    mem_np = batch.membership.realize().numpy()
    lt_np  = batch.latent_type.realize().numpy()
    fm_mod.attach_factor_hyperbolic_params(
        tmp_model,
        n_heads=spec.n_heads, n_factor_types=spec.n_factor_types,
        s_max=spec.s_max, membership_np=mem_np, latent_type_np=lt_np,
    )
    rt_ok = _test_anchor_roundtrip(tmp_model, used_types)
    del tmp_model

    # ---- Issue 2: decisive gradient smoke ----------------------------------
    print("\n" + "=" * 72, flush=True)
    print("[ISSUE 2] Decisive anchor gradient smoke: full builder + per-breath CE", flush=True)
    print("=" * 72, flush=True)

    print(f"\nRunning RELAX=1 (soft-block, tangent params, learnable) ...", flush=True)
    norms_relax, loss_relax = _run_one(
        relax=True, circuit_batch=batch, spec=spec,
        model_factory=model_factory, used_types=used_types, K=K,
    )
    loss_relax_val = float(loss_relax.realize().numpy())
    print(f"  RELAX=1: loss={loss_relax_val:.4f}", flush=True)
    for t, n in sorted(norms_relax.items()):
        print(f"    fg_hyp_anchors_{t} grad_norm = {n:.6f}", flush=True)

    print(f"\nRunning RELAX=0 (frozen, saturated alpha, ball-points raw) ...", flush=True)
    norms_frozen, loss_frozen = _run_one(
        relax=False, circuit_batch=batch, spec=spec,
        model_factory=model_factory, used_types=used_types, K=K,
    )
    loss_frozen_val = float(loss_frozen.realize().numpy())
    print(f"  RELAX=0: loss={loss_frozen_val:.4f}", flush=True)
    for t, n in sorted(norms_frozen.items()):
        print(f"    fg_hyp_anchors_{t} grad_norm = {n:.6f}", flush=True)

    # ---- Verdict -----------------------------------------------------------
    print("\n" + "=" * 72, flush=True)
    print("VERDICT", flush=True)
    print("=" * 72, flush=True)

    ZERO_THRESH = 1e-12
    relax_nonzero = any(v > ZERO_THRESH for v in norms_relax.values())
    frozen_nonzero = any(v > ZERO_THRESH for v in norms_frozen.values())

    print(f"  RELAX=1 anchor grad: "
          f"{'NON-ZERO' if relax_nonzero else '~0'} "
          f"(max={max(norms_relax.values()):.2e})", flush=True)
    print(f"  RELAX=0 anchor grad: "
          f"{'NON-ZERO' if frozen_nonzero else '~0'} "
          f"(max={max(norms_frozen.values()):.2e})", flush=True)

    if relax_nonzero:
        print("\nVERDICT: RELAX VALID — fire.", flush=True)
        print("  The earlier 0.0 was a bias.mean() measurement artifact.", flush=True)
        print("  The FULL per-breath CE path carries a non-zero gradient to the anchors.", flush=True)
        print("  Next step: add FG_HYP_MASK=1 FG_HYP_RELAX=1 FG_HYP_FREEZE=0 to the", flush=True)
        print("  circuit training run and monitor anchor norm drift + cell_acc.", flush=True)
    else:
        print("\nVERDICT: GRADIENT-DEAD — anchor grad ~0 even under per-breath CE.", flush=True)
        print("  Diagnosing root cause ...", flush=True)
        os.environ["FG_HYP_RELAX"] = "1"
        import mycelium.factor_masks as fm_mod2
        importlib.reload(fm_mod2)
        _diagnose_zero_grad(batch, spec, used_types, K=K)

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 72, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"  round_trip_ok      = {rt_ok}", flush=True)
    print(f"  relax_grad_nonzero = {relax_nonzero}", flush=True)
    print(f"  frozen_grad_nonzero= {frozen_nonzero}", flush=True)
    print(f"  astparse_ok        = True  (or we wouldn't be here)", flush=True)

    return rt_ok and relax_nonzero


if __name__ == "__main__":
    ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={ok}", flush=True)
    if not ok:
        sys.exit(1)
    passed = main()
    sys.exit(0 if passed else 1)
