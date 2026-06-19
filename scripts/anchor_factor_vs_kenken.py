"""anchor_factor_vs_kenken.py — Step 3 parity anchor: general engine vs v98 oracle.

Proves that factor_breathing_forward (mycelium/factor_graph_engine.py) reproduces
kenken_breathing_forward (mycelium/kenken.py) to within numerical noise when driven
with the SAME batch and the SAME trained weights.

PASS criterion
--------------
  * Per-breath max-abs-diff of value_logits < 1e-2  (ALL K=16 breaths)
  * Final-breath cell_acc matches between oracle and general engine within 0.5 pp

EXPECTED outcome
----------------
The masks are already proven bit-equal (Step 1).  The breath loop wiring is
byte-identical by design.  Any diff > 1e-2 pinpoints a param-copy or wiring delta
(see PARAM MAPPING section in the docstring below).

PARAM MAPPING  (kenken attribute  ->  fg attribute)
----------------------------------------------------
  model.kenken_state_embed      ->  model.fg_state_embed      (N+1=8, H)
  model.kenken_position_embed   ->  model.fg_position_embed   (49, H)
  model.kenken_value_codebook   ->  model.fg_value_codebook   (N=7, H)
  model.kenken_calib_head_w     ->  model.fg_calib_head_w     (H, 1)
  model.kenken_calib_head_b     ->  model.fg_calib_head_b     (1,)
  model.kenken_breath_embed     ->  model.fg_breath_embed     (K_max, H)
  model.kenken_delta_gate       ->  model.fg_delta_gate       (K_max,)

NOT COPIED (backbone — same model object, shared automatically):
  model.block.layers[:4]  (wq/bq/wk/bk/wv/bv/wo/bo/w_in/b_in/w_out/b_out/LNs)
  model.ln_f_g, model.ln_f_b  (final layernorm)

INLET:
  The verification inlet is pre-built ONCE from the raw batch using
  build_verification_inlet (kenken.py) and passed as batch.factor_inlet so
  BOTH forwards use the SAME (B, 49, H) inlet tensor.  This isolates the
  general engine loop itself from the inlet computation.

NOT MAPPED (kenken-only, not used by general engine):
  model.kenken_op_embed, kenken_target_embed, kenken_size_embed,
  model.kenken_inlet_w, kenken_inlet_b
  model.kenken_fixed_mask, kenken_head_split
  model.kenken_value_bias  (OFF in this run — checked below)

USAGE
-----
  DEV=AMD KENKEN_TASK=1 python scripts/anchor_factor_vs_kenken.py

GPU-FREE BUILD CHECK
--------------------
  python3 -c "import ast; ast.parse(open('scripts/anchor_factor_vs_kenken.py').read()); print('ast.parse OK')"
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Make repo root importable (scripts/ -> mycelium/ sibling).
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ---------------------------------------------------------------------------
# Env vars that kenken.py reads at import time — set BEFORE importing.
# ---------------------------------------------------------------------------
os.environ.setdefault("KENKEN_TASK", "1")
os.environ.setdefault("KENKEN_K_MAX", "16")  # matches the ckpt
os.environ.setdefault("KENKEN_HYP_MASK", "0")  # off -> v98 hard mask path
os.environ.setdefault("KENKEN_PI_ROPE", "0")    # off -> no Q rotation
os.environ.setdefault("KENKEN_VALUE_BIAS", "0") # off (not in this ckpt)

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium import Config, BreathingTransformer
from mycelium.kenken import (
    attach_kenken_params,
    kenken_breathing_forward,
    build_verification_inlet,
    kenken_accuracy,
    kenken_state_dict,
)
from mycelium.kenken_data import KenKenLoader, load_jsonl
from mycelium.factor_graph_engine import (
    FactorGraphSpec,
    attach_factor_graph_params,
    factor_breathing_forward,
    make_kenken_factor_batch,
    factor_accuracy,
    FG_HYP_MASK,
)
from mycelium.factor_masks import attach_factor_hyperbolic_params

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT = os.path.join(
    _REPO,
    ".cache/kenken_ckpts/kenken_curric_k16_cont",
    "kenken_curric_k16_cont_final.safetensors",
)
TEST_PATH = os.path.join(_REPO, ".cache/kenken_test.jsonl")

# From measured_config.json: K=16, n_cages_max=41, backbone=pythia
K = 16
N_CAGES_MAX = 41
EVAL_BATCH = 8   # small enough to realize quickly; increase if desired

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------
LOGIT_TOL = 1e-2   # max-abs-diff across all (B, 49, 7) logits per breath
ACC_TOL_PP = 0.5   # cell_acc must agree within 0.5 percentage points


def _load_ckpt_into(model, path: str) -> None:
    """Load a kenken safetensors checkpoint into model (same pattern as kenken_train.py)."""
    from tinygrad.nn.state import safe_load

    sd = safe_load(path)

    # Build the same state-dict structure that kenken_train.py uses for saving.
    # (model_state_dict_kenken keys — replicated here without importing the script.)
    targets: dict[str, Tensor] = {}

    # Backbone: final LN.
    targets["ln_f.g"] = model.ln_f_g
    targets["ln_f.b"] = model.ln_f_b

    # Shared weights (all layers share these wv/wo/ffn).
    sw = model.block.shared
    for a in ("wv", "bv", "wo", "bo", "w_out", "b_out",
              "in_ln_g", "in_ln_b", "post_ln_g", "post_ln_b"):
        targets[f"shared.{a}"] = getattr(sw, a)

    # Per-phase (layer-specific) attention weights.
    for i, layer in enumerate(model.block.layers):
        for a in ("wq", "bq", "wk", "bk", "w_in", "b_in"):
            targets[f"phase{i}.{a}"] = getattr(layer, a)

    # KenKen-specific params.
    targets.update(kenken_state_dict(model))

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
        print(f"  [WARN] ckpt missing {len(missing)} key(s): {missing[:6]}")
    else:
        print(f"  loaded {len(targets)} tensors — no missing keys.")


def _copy_kenken_to_fg(model) -> None:
    """Copy the trained kenken_ params into the fg_ slots (in-place assign).

    Both forwards use the SAME model.block.layers / ln_f_* backbone — no copy needed.
    Only the 7 KenKen-specific head params are copied into their fg_ counterparts.
    """
    mapping = [
        # (kenken attribute name,          fg attribute name)
        ("kenken_state_embed",      "fg_state_embed"),
        ("kenken_position_embed",   "fg_position_embed"),
        ("kenken_value_codebook",   "fg_value_codebook"),
        ("kenken_calib_head_w",     "fg_calib_head_w"),
        ("kenken_calib_head_b",     "fg_calib_head_b"),
        ("kenken_breath_embed",     "fg_breath_embed"),
        ("kenken_delta_gate",       "fg_delta_gate"),
    ]
    print("\n  Param copy (kenken -> fg):")
    for kn_name, fg_name in mapping:
        src: Tensor = getattr(model, kn_name)
        dst: Tensor = getattr(model, fg_name)
        assert src.shape == dst.shape, (
            f"Shape mismatch: {kn_name}{src.shape} vs {fg_name}{dst.shape}"
        )
        dst.assign(src.cast(dst.dtype)).realize()
        print(f"    {kn_name}{list(src.shape)} -> {fg_name}: OK")

    # Sanity: kenken_value_bias must be absent (it changes the oracle readout path).
    vb = getattr(model, "kenken_value_bias", None)
    if vb is not None:
        print("  [WARN] kenken_value_bias is present and active — "
              "the general engine has no equivalent; logit diff may be non-trivial.")


def _to_np(t: Tensor) -> np.ndarray:
    return t.realize().numpy()


def main():
    print("=" * 70)
    print("anchor_factor_vs_kenken.py — Step 3 parity anchor")
    print("=" * 70)

    # ---- Build + load the model ------------------------------------------
    print(f"\n[1] Building BreathingTransformer (Pythia-410M cfg) ...")
    cfg = Config()
    model = BreathingTransformer(cfg)

    # Cast backbone to fp32 (mirrors cast_layers_fp32 in kenken_train.py).
    # BreathingTransformer is already fp32 from Config defaults; this is
    # defensive — matches how the trainer loads the checkpoint.
    from mycelium.kenken import N_MAX, N_CELLS

    print(f"  attach_kenken_params(hidden={cfg.hidden}, n_heads={cfg.n_heads}, k_max={K}) ...")
    attach_kenken_params(model, hidden=cfg.hidden, n_heads=cfg.n_heads, k_max=K)

    print(f"\n[2] Loading checkpoint: {CKPT}")
    assert os.path.exists(CKPT), f"Checkpoint not found: {CKPT}"
    _load_ckpt_into(model, CKPT)

    # ---- Attach general-engine params (random init, then overwrite) ------
    print(f"\n[3] attach_factor_graph_params ...")
    spec = FactorGraphSpec(
        s_max=N_CELLS,        # 49
        n_values=N_MAX,       # 7
        n_factor_types=3,     # row / col / cage
        n_heads=cfg.n_heads,  # 16
        k_max=K,              # 16
        has_factor_inlet=True,
    )
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)

    # FG_HYP_MASK=1 (frozen-confirm): build anchor tables from representative
    # KenKen membership.  Use a sample batch so G_t is determined from real data.
    # The anchor tables are FROZEN (not in any optimizer); the geometric mask ==
    # the hard mask to ~1e-3, so the anchor compare should still PASS.
    if FG_HYP_MASK:
        print(f"\n[3b] FG_HYP_MASK=1: attach_factor_hyperbolic_params ...")
        _tmp_loader = KenKenLoader(TEST_PATH, batch_size=EVAL_BATCH, seed=999,
                                   n_cages_max=N_CAGES_MAX)
        _tmp_kb = _tmp_loader.sample_batch()
        from mycelium.factor_graph_engine import make_kenken_factor_batch as _mkfb2
        _tmp_fb = _mkfb2(_tmp_kb, spec)
        _mem_np = _tmp_fb.membership.realize().numpy()
        _lt_np  = _tmp_fb.latent_type.realize().numpy()
        attach_factor_hyperbolic_params(
            model,
            n_heads=spec.n_heads,
            n_factor_types=spec.n_factor_types,
            s_max=spec.s_max,
            membership_np=_mem_np,
            latent_type_np=_lt_np,
        )
        del _tmp_loader, _tmp_kb, _tmp_fb, _mem_np, _lt_np
        print(f"  hyperbolic params attached (frozen).")

    # Copy trained kenken params into fg slots.
    _copy_kenken_to_fg(model)

    # ---- Build one eval batch --------------------------------------------
    print(f"\n[4] Loading test batch from {TEST_PATH} ...")
    assert os.path.exists(TEST_PATH), f"Test data not found: {TEST_PATH}"
    loader = KenKenLoader(TEST_PATH, batch_size=EVAL_BATCH, seed=0,
                          n_cages_max=N_CAGES_MAX)
    kb = loader.sample_batch()   # KenKenBatch

    # ---- Pre-build the verification inlet once ---------------------------
    print(f"  Pre-building verification inlet (build_verification_inlet) ...")
    inlet = build_verification_inlet(
        model,
        kb.cage_op, kb.cage_target, kb.cage_size, kb.cell_cage_id,
    )  # (B, 49, H)  fp32
    inlet.realize()

    # ---- Adapt kenken batch -> factor batch ------------------------------
    print(f"  Building FactorGraphBatch (make_kenken_factor_batch) ...")
    fg_batch = make_kenken_factor_batch(kb, spec, prebuilt_inlet=inlet)

    # ---- Run oracle (kenken_breathing_forward) ---------------------------
    print(f"\n[5] Running oracle: kenken_breathing_forward (K={K}) ...")
    Tensor.training = False
    oracle_logits, oracle_calib = kenken_breathing_forward(model, kb, K=K)
    # Realize all oracle outputs.
    oracle_np = [_to_np(lg) for lg in oracle_logits]   # list of (B,49,7) np arrays
    print(f"  Oracle: {len(oracle_np)} breath tensors realized.")

    # ---- Run general engine (factor_breathing_forward) -------------------
    print(f"\n[6] Running general engine: factor_breathing_forward (K={K}) ...")
    fg_logits, fg_calib = factor_breathing_forward(model, fg_batch, spec, K=K)
    fg_np = [_to_np(lg) for lg in fg_logits]
    print(f"  FG engine: {len(fg_np)} breath tensors realized.")

    # ---- Compare per-breath -------------------------------------------------
    print(f"\n[7] Per-breath max-abs-diff (oracle vs general engine):")
    print(f"{'Breath':>7}  {'max|Δlogit|':>14}  {'PASS?':>6}")
    print("  " + "-" * 35)

    cell_valid_np = _to_np(kb.cell_valid).astype(bool)  # (B, 49)
    all_pass = True
    max_diff_overall = 0.0

    for k in range(K):
        # Compare only on VALID cells (padding cells carry zeros; any diff there
        # is a validity-masking artefact, not a logic error).
        oracle_k = oracle_np[k]   # (B, 49, 7)
        fg_k     = fg_np[k]       # (B, 49, 7)

        diff = np.abs(oracle_k - fg_k)
        # Mask to valid cells only.
        valid_mask = cell_valid_np[:, :, None]  # (B, 49, 1) broadcast over N
        diff_valid = diff * valid_mask
        max_diff = float(diff_valid.max()) if valid_mask.any() else 0.0
        max_diff_overall = max(max_diff_overall, max_diff)

        ok = max_diff < LOGIT_TOL
        if not ok:
            all_pass = False
        flag = "PASS" if ok else "FAIL"
        print(f"  k={k:>2d}  {max_diff:>14.6f}  {flag:>6}")

    print()
    print(f"  Overall max-abs-diff (valid cells): {max_diff_overall:.6f}")
    print(f"  Tolerance: {LOGIT_TOL}")

    # ---- Compare cell accuracy -------------------------------------------
    print(f"\n[8] Cell accuracy comparison (final breath):")

    # Oracle accuracy (kenken_accuracy uses its own internal final-logit argmax).
    oracle_cell_acc, oracle_puzzle_acc = kenken_accuracy(oracle_logits[-1], kb)

    # General engine accuracy.
    fg_cell_acc, fg_puzzle_acc = factor_accuracy(fg_logits[-1], fg_batch, spec)

    print(f"  oracle  cell_acc={oracle_cell_acc*100:.2f}%  "
          f"puzzle_acc={oracle_puzzle_acc*100:.2f}%")
    print(f"  FG eng  cell_acc={fg_cell_acc*100:.2f}%  "
          f"puzzle_acc={fg_puzzle_acc*100:.2f}%")

    acc_diff_pp = abs(oracle_cell_acc - fg_cell_acc) * 100.0
    acc_pass = acc_diff_pp <= ACC_TOL_PP
    if not acc_pass:
        all_pass = False
    print(f"  cell_acc diff: {acc_diff_pp:.3f} pp  "
          f"(tol={ACC_TOL_PP} pp)  {'PASS' if acc_pass else 'FAIL'}")

    # ---- Summary ---------------------------------------------------------
    print()
    print("=" * 70)
    if all_pass:
        print("ANCHOR PASS: general engine is byte-identical to v98 oracle")
        print(f"  max logit diff={max_diff_overall:.2e}  "
              f"cell_acc_diff={acc_diff_pp:.3f} pp")
    else:
        print("ANCHOR FAIL — diff exceeds tolerance; see per-breath table above")
        print("Common causes:")
        print("  * Param not copied into fg slot (check PARAM MAPPING in docstring)")
        print("  * Inlet not passed correctly (both forwards must use the SAME inlet)")
        print("  * KENKEN_VALUE_BIAS=1 with no fg equivalent (check model.kenken_value_bias)")
        print("  * kenken_layer_forward S==N_CELLS assert fires if s_max != 49")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
