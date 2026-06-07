"""v106 → v110-step3 adapter: PUCT-on-digit-codebook search atop v110-step3 BP.

Adapts the original v106 MCTS engine (designed for v105's per-position digit
codebook architecture) to work with v110-step3:

  v110-step3 INPUT  : domain_init shape (B, N_MAX, 200) — single 200-bin
                       distribution per variable (the 200 bins represent integer
                       values 0-9999 via build_bin_values()).
  v110-step3 OUTPUT : tree_logits shape (B, N_MAX, 5, 10) per breath — same
                       per-position digit-codebook layout as v105.

The OUTPUT-side clamping translates directly from v106 (clamp at (var, pos) →
digit_value). The INPUT-side has two modes:

  FULL CLAMP   (all n_digits clamped for a variable):
    digits → integer value → nearest 200-bin → set domain_init[b, vi, :] one-hot
    at the bin index AND set node_kinds[b, vi] = 0 (observed). Strong cascade:
    info propagates through BP via the embedding pathway.

  PARTIAL CLAMP (some positions clamped for a variable):
    leave domain_init alone. After each breath's forward, override the
    tree_logits at clamped (vi, pos) to put all mass on the clamped digit (the
    final argmax decode honors the clamp). This is a softer clamp — info flows
    only at decode time, not through the BP layers.

PUCT selection preferentially "completes" variables (tries to fill all 5
positions for one variable before moving to another), because a complete
variable becomes a FULL clamp with a much stronger cascade.

The MCTS layer is pure Python / NumPy — no TinyJit. The BP forward is called
directly (not through the JIT cache) because the JIT compile keys on tensor
shape and we want one-pass-per-rollout simplicity.

Algorithm (mirrors original v106):
  Phase 0  : run BP once at K=k_breaths
  Phase 1  : if calib > calib_threshold → done (BP-only)
  Phase 2  : PUCT loop (n_rollouts iterations)
    SELECT    : UCB1 walk from root
    EXPAND    : clamp most-uncertain unclamped (var, digit_pos)
    SIMULATE  : re-run BP with clamps applied, score the fixed point
    BACKPROP  : accumulate (visits, value) up the path
  Return best-path clamps → final BP decode
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.engine.jit import TinyJit

from mycelium.factor_graph_v110_step import (
    V110_STEP_K_MAX,
    V110_STEP_N_MAX,
    V110_STEP_F_MAX,
    V110_STEP_N_HEADS,
    V110_STEP_N_DIGITS,
    V110_STEP_ALTERNATION,
    V110_STEP_PHASE_SCALE,
    V110_STEP_GATE_PROFILE,
    V110_STEP_PHOTON_ALPHA,
    fg_breathing_forward_v110_step,
)
from mycelium.factor_graph_v107 import get_bin_values, nearest_bin
from mycelium.factor_graph_v108 import (
    values_to_digits_msd, digits_to_value_msd, bins_to_digits_msd,
)
from mycelium.factor_graph_data import OP_MAP


# ---------------------------------------------------------------------------
# Configuration (env vars)
# ---------------------------------------------------------------------------

_V106_S3_N_ROLLOUTS      = int(os.environ.get("V106_S3_N_ROLLOUTS",      "30"))
_V106_S3_CALIB_THRESHOLD = float(os.environ.get("V106_S3_CALIB_THRESHOLD", "0.85"))
_V106_S3_K_BREATHS       = int(os.environ.get("V106_S3_K_BREATHS",       str(V110_STEP_K_MAX)))
# PUCT exploration constant (AlphaZero-style). Lower than UCB1's sqrt(2)
# because the prior already focuses search; c=1.5 is a common starting point.
_V106_S3_UCT_C           = float(os.environ.get("V106_S3_UCT_C",          "1.5"))
# Score weights (calib, energy, conf). DEFAULT CHANGED FROM 0.5/0.3/0.2 TO
# 0.0/1.0/0.0 after empirical sweep on n=50 hard puzzles found:
#   0.5/0.3/0.2 → Δ cell_acc = -0.0122 (regression)
#   0.0/0.8/0.2 → Δ cell_acc = -0.0092
#   0.0/1.0/0.0 → Δ cell_acc = -0.0061  ← least bad
#   0.0/0.6/0.4 → Δ cell_acc = -0.0122
# Calib head provides no discriminative signal on hard (range 0.44-0.64,
# never trips 0.85 threshold). Energy is the only useful score axis. None
# of the configs flip to positive Δ — see project_v106_puct_step3_negative.md
# for the structural explanation (local consistency ≠ global correctness).
_V106_S3_SCORE_W_STR     = os.environ.get("V106_S3_SCORE_W", "0.0,1.0,0.0")
_w_parts                 = [float(x) for x in _V106_S3_SCORE_W_STR.split(",")]
_V106_S3_W_CALIB, _V106_S3_W_ENERGY, _V106_S3_W_CONF = (
    _w_parts[0], _w_parts[1], _w_parts[2]
) if len(_w_parts) == 3 else (0.0, 1.0, 0.0)
# When true (default), use JIT-compiled BP forward at B=1. AM driver constraint:
# the eval JIT pattern in factor_graph_v110_step3 works fine at B=8, the same
# pattern applies at B=1.
_V106_S3_USE_JIT         = int(os.environ.get("V106_S3_USE_JIT", "1")) > 0


# Reverse OP_MAP for energy computation
_INV_OP_MAP = {v: k for k, v in OP_MAP.items()}
_OPS_NP = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b if b != 0 else a,
}


# ---------------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """One node in the MCTS tree.

    clamps: dict mapping (var_idx, digit_pos) → digit_value (int 0-9).
    children: dict mapping (var_idx, digit_pos, digit_value) → MCTSNode.
    prior: BP-derived policy prior for this child (probability that this
        digit value at (var, pos) is correct, given parent's clamps). Set
        when the child is created during EXPAND. Root has prior=1.0 (unused).
    """
    clamps:   dict  = field(default_factory=dict)
    visits:   int   = 0
    value:    float = 0.0
    children: dict  = field(default_factory=dict)
    prior:    float = 1.0

    @property
    def q_value(self) -> float:
        return self.value / self.visits if self.visits > 0 else 0.0


# ---------------------------------------------------------------------------
# PUCT selection (AlphaZero-style, true policy prior)
# ---------------------------------------------------------------------------

def uct_select(parent: MCTSNode, c: float = _V106_S3_UCT_C) -> MCTSNode:
    """Select the child that maximises AlphaZero PUCT.

    PUCT(child) = Q(child) + c * P(child) * sqrt(parent.visits) / (1 + child.visits)

    Where P(child) is the BP-derived policy prior (probability that this
    digit value is correct at the expanded (var, pos), given the parent's
    clamps). Unlike UCB1, unvisited children DO NOT get infinite priority —
    they get c * P(child) * sqrt(parent.visits) / 1, so a high-prior unvisited
    child beats a low-prior unvisited child. This focuses budget on plausible
    digits.
    """
    sqrt_parent = math.sqrt(max(parent.visits, 1))
    best_score = -float("inf")
    best_child = None
    for child in parent.children.values():
        u = c * child.prior * sqrt_parent / (1.0 + child.visits)
        puct = child.q_value + u
        if puct > best_score:
            best_score = puct
            best_child = child
    assert best_child is not None, "uct_select called on node with no children"
    return best_child


# ---------------------------------------------------------------------------
# Clamping helpers (adapted for v110-step3 input layout)
# ---------------------------------------------------------------------------

def _completed_vars(clamps: dict, n_digits: int) -> dict:
    """Return {var_idx: [digit0, digit1, ...]} for variables where ALL
    n_digits positions are clamped (eligible for full-clamp cascade)."""
    by_var: dict[int, dict[int, int]] = {}
    for (vi, dp), dv in clamps.items():
        by_var.setdefault(vi, {})[dp] = dv
    completed = {}
    for vi, dps in by_var.items():
        if len(dps) == n_digits and all(p in dps for p in range(n_digits)):
            completed[vi] = [dps[p] for p in range(n_digits)]
    return completed


def _apply_full_clamps_to_batch_np(
    batch_np: dict,
    clamps: dict,
    n_digits: int,
) -> dict:
    """Apply FULL clamps to a copy of the (B=1) batch.

    For each variable whose full digit-tuple is determined by clamps:
      - convert digits → integer value
      - find nearest bin index
      - set domain_init[0, vi, :] = 0 except bin_idx = 1.0
      - set node_kinds[0, vi] = 0 (observed)

    Partial clamps are NOT applied here; they are handled at decode time
    via tree_logits override (see _apply_partial_clamps_to_logits).
    """
    full = _completed_vars(clamps, n_digits=n_digits)
    if not full:
        return batch_np

    result = dict(batch_np)
    result["domain_init"] = batch_np["domain_init"].copy()
    result["node_kinds"]  = batch_np["node_kinds"].copy()

    bv = get_bin_values()
    n_max = int(batch_np["domain_init"].shape[1])

    for vi, digit_list in full.items():
        if not (0 <= vi < n_max):
            continue
        # MSD-first digit list → integer value
        value = 0
        n_d = len(digit_list)
        for p, d in enumerate(digit_list):
            value += d * (10 ** (n_d - 1 - p))
        # Clip to [0, 9999] just in case
        value = int(max(0, min(value, 9999)))
        bin_idx = nearest_bin(value, bv)

        result["domain_init"][0, vi, :] = 0.0
        result["domain_init"][0, vi, bin_idx] = 1.0
        result["node_kinds"][0, vi] = 0  # observed

    return result


def _apply_partial_clamps_to_logits(
    tree_logits_np: np.ndarray,
    clamps: dict,
    n_digits: int,
) -> np.ndarray:
    """Override tree_logits at partially-clamped (vi, pos) positions.

    For each clamp (vi, pos) → dv:
      - Set tree_logits_np[vi, pos, :] to a strongly peaked distribution
        (all mass on dv) so the argmax decode picks dv.

    This is the softer clamp: only affects readout, NOT the BP cascade.
    NB: For variables already fully clamped (handled by full clamp path),
    we ALSO override here for safety — final argmax will respect the clamp
    regardless of the BP fixed point.

    Returns a NEW (N_MAX, N_DIGITS, 10) numpy array.
    """
    if not clamps:
        return tree_logits_np
    out = tree_logits_np.copy()
    n_max = out.shape[0]
    for (vi, dp), dv in clamps.items():
        if not (0 <= vi < n_max and 0 <= dp < n_digits and 0 <= dv <= 9):
            continue
        out[vi, dp, :] = -1e4
        out[vi, dp, dv] = 1e4
    return out


# ---------------------------------------------------------------------------
# BP runner (single problem, B=1) — JIT-compiled forward for fast rollouts
# ---------------------------------------------------------------------------

# Cache: maps (id(model), K, n_max, f_max, n_digits, alternation, phase_scale,
#              gate_profile, photon_alpha) → JIT'd forward closure.
_JIT_V106_S3_BP_CACHE: dict = {}


def _compile_jit_bp_single_v106_step3(
    model: Any,
    K: int,
    n_max: int,
    f_max: int,
    n_digits: int,
    alternation: bool,
    phase_scale: float,
    gate_profile: str,
    photon_alpha: float,
):
    """Build (or fetch from cache) a JIT'd B=1 BP forward.

    Returns a callable taking the 4 input tensors (B=1, fp32 for floats / i32
    for ints) and returning `(final_tree_logits_realized, final_calib_realized)`
    as realized tinygrad Tensors. The MCTS Python layer reads `.numpy()` on
    the returned tensors.

    AM driver constraint: the only casts inside this JIT are those already
    present in `fg_breathing_forward_v110_step`. We do NOT add any new
    `.cast(dtypes.float32)` or per-param `.isnan()` calls.

    JIT cache keys on tensor SHAPE not VALUE. Clamp value changes flow in via
    `domain_init` / `node_kinds` values and are fine; only shape changes
    (i.e. different K, n_max, f_max, n_digits) force a recompile.
    """
    key = ("bp_v106_s3", id(model), int(K), int(n_max), int(f_max),
           int(n_digits), bool(alternation), float(phase_scale),
           str(gate_profile), float(photon_alpha))
    if key in _JIT_V106_S3_BP_CACHE:
        return _JIT_V106_S3_BP_CACHE[key]

    print(
        f"[JIT] compile v106-step3 BP B=1: K={K} n_max={n_max} f_max={f_max} "
        f"n_digits={n_digits} profile={gate_profile} alpha={photon_alpha}...",
        flush=True,
    )

    @TinyJit
    def _bp_b1(
        domain_init: Tensor,
        node_kinds: Tensor,
        staging_mask: Tensor,
        head_op_mask: Tensor,
    ):
        tree_lh, _, _, calib_h, _ = fg_breathing_forward_v110_step(
            model, domain_init, node_kinds, staging_mask, head_op_mask,
            K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
        )
        final_tree = tree_lh[-1]              # (1, n_max, n_digits, 10)
        final_calib = calib_h[-1]             # (1,)
        return final_tree.realize(), final_calib.realize()

    _JIT_V106_S3_BP_CACHE[key] = _bp_b1
    print(
        f"[JIT] v106-step3 BP B=1 ready (cache={len(_JIT_V106_S3_BP_CACHE)})",
        flush=True,
    )
    return _bp_b1


def _batch_np_to_tensors_for_forward(batch_np: dict) -> dict:
    """Convert the relevant numpy arrays to Tensors for v110_step forward."""
    return {
        "domain_init":  Tensor(
            batch_np["domain_init"], dtype=dtypes.float,
        ).contiguous().realize(),
        "node_kinds":   Tensor(
            batch_np["node_kinds"], dtype=dtypes.int,
        ).contiguous().realize(),
        "staging_mask": Tensor(
            batch_np["staging_mask"], dtype=dtypes.float,
        ).contiguous().realize(),
        "head_op_mask": Tensor(
            batch_np["head_op_mask"], dtype=dtypes.float,
        ).contiguous().realize(),
    }


def _compute_energy_np(
    pred_values: np.ndarray,         # (N_MAX,) int — argmax-decoded values
    factor_types: np.ndarray,        # (F_MAX,) int — op code (or -1 for pad)
    factor_args: np.ndarray,         # (F_MAX, 3) int — (a, b, r) indices
    factor_valid: np.ndarray,        # (F_MAX,) float — 1 for real factor
) -> float:
    """Numpy constraint energy: relative error per factor, clipped to 10.

    For each valid factor (op, a, b, r):
      expected_r = op(pred[a], pred[b])
      err = |expected_r - pred[r]| / max(|expected_r|, 1)
      err = min(err, 10.0)
    Returns sum of errs over valid factors.
    """
    total = 0.0
    n_max = int(pred_values.shape[0])
    f_max = int(factor_types.shape[0])
    for fi in range(f_max):
        if factor_valid[fi] < 0.5:
            continue
        ft = int(factor_types[fi])
        if ft < 0 or ft not in _INV_OP_MAP:
            continue
        op_name = _INV_OP_MAP[ft]
        op_fn = _OPS_NP.get(op_name)
        if op_fn is None:
            continue
        ai = int(factor_args[fi, 0])
        bi = int(factor_args[fi, 1])
        ri = int(factor_args[fi, 2])
        if not (0 <= ai < n_max and 0 <= bi < n_max and 0 <= ri < n_max):
            continue
        a_val = int(pred_values[ai])
        b_val = int(pred_values[bi])
        r_val = int(pred_values[ri])
        try:
            expected = float(op_fn(float(a_val), float(b_val)))
        except Exception:
            continue
        denom = max(abs(expected), 1.0)
        err = abs(expected - float(r_val)) / denom
        err = min(err, 10.0)
        total += err
    return float(total)


def _run_bp_single(
    model: Any,
    batch_np: dict,
    clamps: dict,
    K: int,
    n_max: int,
    f_max: int,
    n_digits: int,
    alternation: bool,
    phase_scale: float,
    gate_profile: str,
    photon_alpha: float,
    use_jit: bool = True,
) -> tuple[np.ndarray, float, float, float]:
    """Run BP on a single problem (B=1) with the given clamps.

    Returns:
      tree_logits_np : (N_MAX, N_DIGITS, 10) final-breath logits (with partial
                       clamps applied as a logits override)
      calib_score    : final-breath calibration (0-1)
      energy_val     : constraint energy (>=0; lower = better)
      confidence     : mean max-prob over unclamped (var, pos) — minus 0.1

    When `use_jit=True`, calls the JIT-compiled B=1 forward (first call per
    shape config takes a few seconds to compile; subsequent calls are 5-10×
    faster than eager). Otherwise falls back to direct eager forward.
    """
    # Apply FULL clamps to the input (modifies domain_init, node_kinds)
    bnp = _apply_full_clamps_to_batch_np(batch_np, clamps, n_digits=n_digits)

    tens = _batch_np_to_tensors_for_forward(bnp)

    Tensor.training = False
    if use_jit:
        bp_fn = _compile_jit_bp_single_v106_step3(
            model, K=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
        )
        final_tree, final_calib = bp_fn(
            tens["domain_init"],
            tens["node_kinds"],
            tens["staging_mask"],
            tens["head_op_mask"],
        )
        final_logits_np = final_tree.numpy()[0]            # (N_MAX, n_digits, 10)
        calib_score = float(final_calib.numpy()[0])
    else:
        tree_lh, _, _, calib_h, _ = fg_breathing_forward_v110_step(
            model,
            tens["domain_init"],
            tens["node_kinds"],
            tens["staging_mask"],
            tens["head_op_mask"],
            K=K,
            n_max=n_max,
            f_max=f_max,
            n_digits=n_digits,
            alternation=alternation,
            phase_scale=phase_scale,
            gate_profile=gate_profile,
            photon_alpha=photon_alpha,
        )

        final_tree = tree_lh[-1]                                   # (B, N_MAX, n_digits, 10)
        final_logits_np = final_tree.realize().numpy()[0]          # (N_MAX, n_digits, 10)
        calib_score = float(calib_h[-1].realize().numpy()[0])

    # Apply PARTIAL clamps to logits (override at clamped positions)
    final_logits_np = _apply_partial_clamps_to_logits(
        final_logits_np, clamps, n_digits=n_digits,
    )

    # Argmax-decode → (N_MAX, n_digits) digits
    pred_digits = final_logits_np.argmax(axis=-1).astype(np.int64)
    pred_values = digits_to_value_msd(pred_digits).astype(np.int64)  # (N_MAX,)

    # Constraint energy
    energy_val = _compute_energy_np(
        pred_values,
        bnp["factor_types"][0],
        bnp["factor_args"][0],
        bnp["factor_valid"][0],
    )

    # Confidence: mean max-prob over unclamped (var, pos) where var is unobserved
    obs_mask_np = bnp["observed_mask"][0]  # (N_MAX,)
    # NB: full-clamped vars now have observed_mask=1 logically (we set node_kinds=0),
    # but observed_mask is the ORIGINAL puzzle observation status. For confidence,
    # we want positions we DIDN'T explicitly clamp on UNOBSERVED variables.
    logits = final_logits_np
    log_max = logits.max(axis=-1, keepdims=True)
    p = np.exp(logits - log_max)
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
    max_probs = p.max(axis=-1)  # (N_MAX, n_digits)

    unclamped_mask = np.zeros((n_max, n_digits), dtype=bool)
    for vi in range(n_max):
        if obs_mask_np[vi] == 0:
            for dp in range(n_digits):
                if (vi, dp) not in clamps:
                    unclamped_mask[vi, dp] = True

    if unclamped_mask.any():
        mean_max_prob = float((max_probs * unclamped_mask).sum() / unclamped_mask.sum())
        confidence_score = mean_max_prob - 0.1
    else:
        confidence_score = 1.0

    return final_logits_np, calib_score, energy_val, confidence_score


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def score_state(
    calib_score: float,
    energy_val: float,
    confidence_score: float,
    w_calib: float = _V106_S3_W_CALIB,
    w_energy: float = _V106_S3_W_ENERGY,
    w_conf: float = _V106_S3_W_CONF,
) -> float:
    """Score a BP fixed point (higher = better)."""
    energy_score = 1.0 / (1.0 + energy_val)
    return w_calib * calib_score + w_energy * energy_score + w_conf * confidence_score


# ---------------------------------------------------------------------------
# Position-selection: pick the (var, pos) to expand next
# ---------------------------------------------------------------------------

def _position_to_expand(
    tree_logits_np: np.ndarray,
    clamps: dict,
    obs_mask_np: np.ndarray,
    n_max: int,
    n_digits: int,
) -> tuple[int, int, float]:
    """Pick the (var, pos) to expand: HIGHEST entropy among unclamped unobserved
    positions, with a STRONG preference for variables that are already
    partially clamped (so PUCT prefers to "complete" a variable, achieving a
    full-clamp cascade).

    Returns (var_idx, digit_pos, entropy). Returns (-1, -1, 0.0) if all positions
    on unobserved variables are clamped.
    """
    # Vars that are partially clamped: bonus for completing them
    by_var: dict[int, int] = {}
    for (vi, dp) in clamps.keys():
        by_var[vi] = by_var.get(vi, 0) + 1

    best_var, best_pos, best_H = -1, -1, -float("inf")
    for vi in range(n_max):
        if obs_mask_np[vi] != 0:
            continue
        # Variables already fully clamped: skip
        n_clamped_for_v = by_var.get(vi, 0)
        if n_clamped_for_v >= n_digits:
            continue

        # Preference bonus: positive iff variable already partially clamped
        # (encourages completion before opening new variables)
        completion_bonus = 0.5 * n_clamped_for_v / max(n_digits, 1)

        for dp in range(n_digits):
            if (vi, dp) in clamps:
                continue
            logits = tree_logits_np[vi, dp]  # (10,)
            log_max = logits.max()
            p_ = np.exp(logits - log_max)
            p_ = p_ / (p_.sum() + 1e-12)
            H = -float((p_ * np.log(p_ + 1e-12)).sum())  # Shannon entropy
            score = H + completion_bonus
            if score > best_H:
                best_H, best_var, best_pos = score, vi, dp

    if best_var < 0:
        return -1, -1, 0.0
    return best_var, best_pos, best_H


# ---------------------------------------------------------------------------
# Policy-prior extraction (BP-derived per-digit probability for child nodes)
# ---------------------------------------------------------------------------

def _policy_prior_at(
    tree_logits_np: np.ndarray,
    var_idx: int,
    digit_pos: int,
) -> np.ndarray:
    """Softmax over the 10 digits at (var_idx, digit_pos) of the BP output.

    Returns a length-10 numpy array of priors summing to 1. These are the
    AlphaZero-PUCT P(child) values when expanding 10 children at this
    (var, pos): one per digit value 0-9.
    """
    logits = tree_logits_np[var_idx, digit_pos]   # (10,)
    log_max = float(logits.max())
    e = np.exp(logits - log_max)
    s = float(e.sum()) + 1e-12
    return (e / s).astype(np.float32)


# ---------------------------------------------------------------------------
# Greedy "best-clamps" extraction
# ---------------------------------------------------------------------------

def _pick_best_clamps(root: MCTSNode) -> dict:
    """Greedily descend, picking the child with highest Q-value (visits>0).
    Stop at leaves or unvisited children."""
    node = root
    while node.children:
        visited = [(k, c) for k, c in node.children.items() if c.visits > 0]
        if not visited:
            break
        _, best = max(visited, key=lambda kc: kc[1].q_value)
        node = best
    return node.clamps


# ---------------------------------------------------------------------------
# Decoder: tree_logits → predicted integer values
# ---------------------------------------------------------------------------

def argmax_decode_values(tree_logits_np: np.ndarray) -> np.ndarray:
    """(N_MAX, n_digits, 10) → (N_MAX,) integer values via MSD-first decode."""
    pred_digits = tree_logits_np.argmax(axis=-1).astype(np.int64)  # (N_MAX, n_digits)
    return digits_to_value_msd(pred_digits).astype(np.int64)


# ---------------------------------------------------------------------------
# Main entry point: PUCT-on-digit-codebook search
# ---------------------------------------------------------------------------

def puct_solve(
    model: Any,
    batch_np: dict,
    n_rollouts: int = _V106_S3_N_ROLLOUTS,
    calib_threshold: float = _V106_S3_CALIB_THRESHOLD,
    k_breaths: int | None = None,
    c_uct: float = _V106_S3_UCT_C,
    n_max: int = V110_STEP_N_MAX,
    f_max: int = V110_STEP_F_MAX,
    n_digits: int = V110_STEP_N_DIGITS,
    max_clamp_depth: int | None = None,
    alternation: bool = V110_STEP_ALTERNATION,
    phase_scale: float = V110_STEP_PHASE_SCALE,
    gate_profile: str = V110_STEP_GATE_PROFILE,
    photon_alpha: float = V110_STEP_PHOTON_ALPHA,
    use_jit: bool = _V106_S3_USE_JIT,
) -> dict:
    """Solve one v110-step3 factor-graph problem with PUCT refinement.

    Parameters
    ----------
    model        : breathing transformer w/ v110-step3 params attached
    batch_np     : numpy batch dict for a SINGLE problem (B=1)
    n_rollouts   : MCTS rollouts (0 = BP-only)
    calib_threshold : if initial BP calibration > this, skip MCTS
    k_breaths    : K (defaults to V110_STEP_K_MAX)
    c_uct        : UCT exploration constant
    max_clamp_depth : max number of digit clamps (default: n_max * n_digits)
    alternation, phase_scale, gate_profile, photon_alpha : forwarded to BP

    Returns dict:
      predicted_values : (N_MAX,) np.ndarray of int
      tree_logits      : (N_MAX, n_digits, 10) final logits
      calib            : float
      energy           : float
      mcts_triggered   : bool
      rollouts_used    : int
      final_clamps     : dict
      wallclock_s      : float
    """
    t0 = time.perf_counter()

    K = k_breaths if k_breaths is not None else V110_STEP_K_MAX
    K = min(K, V110_STEP_K_MAX)

    if max_clamp_depth is None:
        max_clamp_depth = n_max * n_digits

    # Phase 0: initial BP
    init_logits, init_calib, init_energy, init_conf = _run_bp_single(
        model, batch_np, clamps={}, K=K,
        n_max=n_max, f_max=f_max, n_digits=n_digits,
        alternation=alternation, phase_scale=phase_scale,
        gate_profile=gate_profile, photon_alpha=photon_alpha,
        use_jit=use_jit,
    )

    if n_rollouts == 0 or init_calib >= calib_threshold:
        return {
            "predicted_values": argmax_decode_values(init_logits),
            "tree_logits":      init_logits,
            "calib":            init_calib,
            "energy":           init_energy,
            "mcts_triggered":   False,
            "rollouts_used":    0,
            "final_clamps":     {},
            "wallclock_s":      time.perf_counter() - t0,
        }

    obs_mask_np = batch_np["observed_mask"][0]  # (N_MAX,) — original observation status

    # Phase 1: PUCT rollouts
    root = MCTSNode(clamps={})
    root.visits = 1
    root.value  = score_state(init_calib, init_energy, init_conf)

    rollouts_executed = 0
    for _ in range(n_rollouts):
        # SELECTION
        node = root
        path = [node]
        while node.children:
            node = uct_select(node, c=c_uct)
            path.append(node)

        # SIMULATION: re-run BP with current clamps
        cur_logits, cur_calib, cur_energy, cur_conf = _run_bp_single(
            model, batch_np, clamps=node.clamps, K=K,
            n_max=n_max, f_max=f_max, n_digits=n_digits,
            alternation=alternation, phase_scale=phase_scale,
            gate_profile=gate_profile, photon_alpha=photon_alpha,
            use_jit=use_jit,
        )

        # EXPANSION
        depth = len(node.clamps)
        unclamped_var, unclamped_pos, max_H = _position_to_expand(
            cur_logits, node.clamps, obs_mask_np, n_max=n_max, n_digits=n_digits,
        )
        is_terminal = (unclamped_var < 0) or (max_H < 0.01) or (depth >= max_clamp_depth)

        if not is_terminal:
            # AlphaZero-PUCT: assign BP-derived policy prior to each of 10 children
            priors = _policy_prior_at(cur_logits, unclamped_var, unclamped_pos)
            for dv in range(10):
                key = (unclamped_var, unclamped_pos, dv)
                new_clamp = {**node.clamps, (unclamped_var, unclamped_pos): dv}
                node.children[key] = MCTSNode(
                    clamps=new_clamp,
                    prior=float(priors[dv]),
                )

        # BACKPROP
        sim_score = score_state(cur_calib, cur_energy, cur_conf)
        for n in path:
            n.visits += 1
            n.value  += sim_score

        rollouts_executed += 1

    # Pick best path → final BP with those clamps
    best_clamps = _pick_best_clamps(root)
    final_logits, final_calib, final_energy, final_conf = _run_bp_single(
        model, batch_np, clamps=best_clamps, K=K,
        n_max=n_max, f_max=f_max, n_digits=n_digits,
        alternation=alternation, phase_scale=phase_scale,
        gate_profile=gate_profile, photon_alpha=photon_alpha,
        use_jit=use_jit,
    )

    return {
        "predicted_values": argmax_decode_values(final_logits),
        "tree_logits":      final_logits,
        "calib":            final_calib,
        "energy":           final_energy,
        "mcts_triggered":   True,
        "rollouts_used":    rollouts_executed,
        "final_clamps":     best_clamps,
        "wallclock_s":      time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Helper: extract a single-problem (B=1) batch_np from a multi-problem batch
# ---------------------------------------------------------------------------

def extract_single_problem(batch_np: dict, b: int) -> dict:
    """Slice the b-th problem out of a multi-problem numpy batch.

    Returns a B=1 dict suitable for puct_solve.
    """
    out = {}
    for k, v in batch_np.items():
        if isinstance(v, np.ndarray):
            # Slice along axis 0 and keep batch dim of 1
            out[k] = v[b:b+1].copy()
        else:
            out[k] = v
    return out
