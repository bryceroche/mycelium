"""v106 factor graph — MCTS-on-digit-codebook inference engine.

Sits ATOP v105's breathing transformer (the fast evaluator).  When v105's BP
calibration head reports high confidence, we return immediately.  When
calibration is low, we run AlphaZero-style MCTS rollouts: each rollout clamps
one (variable, digit_position) → digit_value, re-runs BP, and scores the
resulting fixed point.

The MCTS layer is pure Python / NumPy — no TinyJit needed.  v105's
`fg_breathing_forward_v105` IS jit-compiled and called as a subroutine.

Algorithm
---------
  Phase 0  : run BP once (K=k_breaths)
  Phase 1  : if calib > calib_threshold → done (easy path)
  Phase 2  : MCTS loop (n_rollouts iterations)
    SELECT    : UCT walk from root
    EXPAND    : clamp most-uncertain unclamped (var, digit_pos)
    SIMULATE  : re-run BP with clamps, score fixed point
    BACKPROP  : accumulate (visits, value) up the path
  Return best-path clamps → final BP decode

Clamping mechanism
------------------
  clamps: dict  (var_idx: int, digit_pos: int) → digit_value: int
  Applied in `_apply_clamps`: replaces digit_init[b, vi, p, :] with one-hot
  at the clamped digit value and sets node_kinds[b, vi*n_digits+p] = 0.

Score function
--------------
  0.5 * calib_score + 0.3 * energy_score + 0.2 * confidence_score
  where
    calib_score      = calibration head output  (0–1)
    energy_score     = 1 / (1 + constraint_energy)  (0–1)
    confidence_score = mean over unclamped variables of (max_prob - 0.1)

Configuration env vars
----------------------
  V106_N_ROLLOUTS       default 50
  V106_CALIB_THRESHOLD  default 0.7
  V106_K_BREATHS        default 10  (same as V105_K_MAX)
  V106_UCT_C            default sqrt(2) ≈ 1.4142
  V106_SCORE_W          default "0.5,0.3,0.2"  calib/energy/confidence

NOTE: V106_K_BREATHS is capped at V105_K_MAX at runtime; model must be
      initialised with k_max = max(V106_K_BREATHS, V105_K_MAX).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes

from mycelium.factor_graph_v105 import (
    V105_K_MAX,
    V105_N_MAX,
    V105_F_MAX,
    V105_N_DIGITS,
    V105_N_HEADS,
    fg_breathing_forward_v105,
    constraint_energy_v105,
    value_to_digits,
    digits_to_value,
)
from mycelium.factor_graph_data_v105 import (
    build_staging_and_head_masks_v105_np,
    _records_to_batch_v105,
    batch_to_tensors_v105,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_V106_N_ROLLOUTS      = int(os.environ.get("V106_N_ROLLOUTS",      "50"))
_V106_CALIB_THRESHOLD = float(os.environ.get("V106_CALIB_THRESHOLD", "0.7"))
_V106_K_BREATHS       = int(os.environ.get("V106_K_BREATHS",       str(V105_K_MAX)))
_V106_UCT_C           = float(os.environ.get("V106_UCT_C",          str(math.sqrt(2))))
_V106_SCORE_W_STR     = os.environ.get("V106_SCORE_W", "0.5,0.3,0.2")
_w_parts              = [float(x) for x in _V106_SCORE_W_STR.split(",")]
_V106_W_CALIB, _V106_W_ENERGY, _V106_W_CONF = (
    _w_parts[0], _w_parts[1], _w_parts[2]
) if len(_w_parts) == 3 else (0.5, 0.3, 0.2)


# ---------------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """One node in the MCTS tree.

    clamps: dict mapping (var_idx, digit_pos) → digit_value (int 0-9).
    children: dict mapping (var_idx, digit_pos, digit_value) → MCTSNode.
    """
    clamps:   dict  = field(default_factory=dict)
    visits:   int   = 0
    value:    float = 0.0
    children: dict  = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        """Mean value of this node (Q in UCT)."""
        return self.value / self.visits if self.visits > 0 else 0.0


# ---------------------------------------------------------------------------
# UCT selection
# ---------------------------------------------------------------------------

def uct_select(parent: MCTSNode, c: float = _V106_UCT_C) -> MCTSNode:
    """Select the child that maximises UCB1.

    UCT(child) = Q(child) + c * sqrt(ln(parent.visits) / child.visits)

    Unvisited children have infinite UCT (exploration priority).
    """
    best_score = -float("inf")
    best_child = None
    log_parent = math.log(max(parent.visits, 1))
    for child in parent.children.values():
        if child.visits == 0:
            return child  # unvisited: always explore first
        uct = child.q_value + c * math.sqrt(log_parent / child.visits)
        if uct > best_score:
            best_score = uct
            best_child = child
    assert best_child is not None, "uct_select called on node with no children"
    return best_child


# ---------------------------------------------------------------------------
# Clamping helpers
# ---------------------------------------------------------------------------

def _apply_clamps_to_batch_np(
    batch_np: dict,
    clamps: dict,
    n_max: int = V105_N_MAX,
    n_digits: int = V105_N_DIGITS,
) -> dict:
    """Return a copy of batch_np with clamps applied (in-place on B=1 slice).

    For each (var_idx, digit_pos) → digit_value:
      - digit_init[0, var_idx, digit_pos, :] = one-hot at digit_value
      - node_kinds[0, var_idx * n_digits + digit_pos] = 0  (treat as observed)
    """
    if not clamps:
        return batch_np
    # Shallow copy top-level dict; deep copy the two mutable arrays.
    result = dict(batch_np)
    result["digit_init"]  = batch_np["digit_init"].copy()
    result["node_kinds"]  = batch_np["node_kinds"].copy()
    for (vi, dp), dv in clamps.items():
        if not (0 <= vi < n_max and 0 <= dp < n_digits and 0 <= dv <= 9):
            continue
        result["digit_init"][0, vi, dp, :] = 0.0
        result["digit_init"][0, vi, dp, dv] = 1.0
        result["node_kinds"][0, vi * n_digits + dp] = 0  # observed
    return result


# ---------------------------------------------------------------------------
# BP call with optional clamps (single problem, B=1)
# ---------------------------------------------------------------------------

def _run_bp_single(
    model: Any,
    batch_np: dict,
    clamps: dict,
    K: int,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    n_digits: int = V105_N_DIGITS,
) -> tuple[np.ndarray, float, float, float]:
    """Run BP on a single problem (batch_np has B=1).

    Returns:
      digit_logits_np  : (N_MAX, N_DIGITS, 10) final-breath logits
      calib_score      : calibration scalar 0-1
      energy_val       : constraint energy scalar
      confidence_score : mean max-prob of unclamped variable positions
    """
    if clamps:
        batch_np = _apply_clamps_to_batch_np(batch_np, clamps, n_max=n_max, n_digits=n_digits)

    batch_t = batch_to_tensors_v105(batch_np)
    Tensor.training = False

    dig_lh, _, calib_h = fg_breathing_forward_v105(
        model,
        batch_t["digit_init"],
        batch_t["node_kinds"],
        batch_t["staging_mask"],
        batch_t["head_op_mask"],
        K=K,
        n_max=n_max,
        f_max=f_max,
        n_digits=n_digits,
    )

    # Final-breath digit logits (B=1, N_MAX, N_DIGITS, 10) → (N_MAX, N_DIGITS, 10)
    final_logits_np = dig_lh[-1].realize().numpy()[0]   # (N_MAX, N_DIGITS, 10)

    # Calibration (B=1,) → scalar
    calib_score = float(calib_h[-1].realize().numpy()[0])

    # Constraint energy (B=1,) → scalar
    energy_t = constraint_energy_v105(
        dig_lh[-1],
        batch_t["factor_types"],
        batch_t["factor_args"],
        n_max=n_max,
        f_max=f_max,
        n_digits=n_digits,
    )
    energy_val = float(energy_t.realize().numpy())

    # Confidence: mean max-prob over unclamped variable digit positions
    # unclamped = node_kinds == 1 (unobserved) in updated batch
    obs_mask = batch_np["observed_mask"][0]      # (N_MAX,)
    logits   = final_logits_np                   # (N_MAX, N_DIGITS, 10)
    probs    = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs   /= probs.sum(axis=-1, keepdims=True)  # softmax
    max_probs = probs.max(axis=-1)               # (N_MAX, N_DIGITS)

    # Build unclamped mask: digit (vi, dp) is unclamped if obs_mask[vi]==0 AND not in clamps
    unclamped_mask = np.zeros((n_max, n_digits), dtype=bool)
    for vi in range(n_max):
        if obs_mask[vi] == 0:
            for dp in range(n_digits):
                if (vi, dp) not in clamps:
                    unclamped_mask[vi, dp] = True

    if unclamped_mask.any():
        confidence_score = float((max_probs * unclamped_mask).sum() / unclamped_mask.sum()) - 0.1
    else:
        confidence_score = 1.0  # all clamped → fully determined

    return final_logits_np, calib_score, energy_val, confidence_score


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def score_state(
    calib_score: float,
    energy_val: float,
    confidence_score: float,
    w_calib: float = _V106_W_CALIB,
    w_energy: float = _V106_W_ENERGY,
    w_conf: float = _V106_W_CONF,
) -> float:
    """Score a BP fixed point (higher = better).

    calib_score    : model's self-assessment (0-1)
    energy_val     : constraint energy (0-∞; lower = better)
    confidence_score: mean (max_prob - 0.1) of unclamped positions (-0.1–0.9)
    """
    energy_score = 1.0 / (1.0 + energy_val)
    return w_calib * calib_score + w_energy * energy_score + w_conf * confidence_score


# ---------------------------------------------------------------------------
# Entropy / uncertainty measure
# ---------------------------------------------------------------------------

def _position_entropy(digit_logits_np: np.ndarray, clamps: dict, n_max: int, n_digits: int) -> tuple[int, int, float]:
    """Return the (var_idx, digit_pos) with the highest Shannon entropy that is
    NOT already clamped, along with the entropy value.

    digit_logits_np: (N_MAX, N_DIGITS, 10)
    Returns (var_idx, digit_pos, entropy).  Returns (-1, -1, 0.0) if all are clamped.
    """
    best_var, best_pos, best_H = -1, -1, -1.0
    for vi in range(n_max):
        for dp in range(n_digits):
            if (vi, dp) in clamps:
                continue
            logits = digit_logits_np[vi, dp]           # (10,)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            H = -float((p * np.log(p + 1e-12)).sum())  # Shannon entropy ≥ 0
            if H > best_H:
                best_H, best_var, best_pos = H, vi, dp
    return best_var, best_pos, best_H


# ---------------------------------------------------------------------------
# Argmax decode
# ---------------------------------------------------------------------------

def argmax_decode(digit_logits_np: np.ndarray, n_digits: int = V105_N_DIGITS) -> list[int]:
    """Decode (N_MAX, N_DIGITS, 10) logits to a list of integer values."""
    pred_digits = digit_logits_np.argmax(axis=-1)  # (N_MAX, N_DIGITS)
    values = []
    for vi in range(pred_digits.shape[0]):
        values.append(digits_to_value(list(pred_digits[vi]), n_digits))
    return values


# ---------------------------------------------------------------------------
# MCTS main entry point
# ---------------------------------------------------------------------------

def mcts_solve(
    model: Any,
    batch_np: dict,
    n_rollouts: int = _V106_N_ROLLOUTS,
    calib_threshold: float = _V106_CALIB_THRESHOLD,
    k_breaths: int | None = None,
    c_uct: float = _V106_UCT_C,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    n_digits: int = V105_N_DIGITS,
    max_clamp_depth: int | None = None,
) -> dict:
    """Solve one factor-graph problem with optional MCTS refinement.

    Parameters
    ----------
    model      : v105 model (with fg_v105_* params attached)
    batch_np   : numpy batch dict for a SINGLE problem (B=1)
    n_rollouts : max MCTS rollouts (0 = BP-only)
    calib_threshold : if initial BP calibration > this, skip MCTS
    k_breaths  : K for fg_breathing_forward_v105 (default: V105_K_MAX)
    c_uct      : UCT exploration constant
    max_clamp_depth : maximum number of digits to clamp (default: n_max * n_digits)

    Returns
    -------
    dict with keys:
      predicted_values   : list[int] of length N_MAX
      digit_logits       : (N_MAX, N_DIGITS, 10) final logits
      calib              : float   final calibration score
      energy             : float   final constraint energy
      mcts_triggered     : bool    whether MCTS ran
      rollouts_used      : int     actual rollouts executed
      final_clamps       : dict    clamps applied at final decode
      wallclock_s        : float   total elapsed seconds
    """
    t0 = time.perf_counter()

    K = k_breaths if k_breaths is not None else V105_K_MAX
    K = min(K, V105_K_MAX)

    if max_clamp_depth is None:
        max_clamp_depth = n_max * n_digits

    # ------------------------------------------------------------------
    # Phase 0: Initial BP pass
    # ------------------------------------------------------------------
    init_logits, init_calib, init_energy, init_conf = _run_bp_single(
        model, batch_np, clamps={}, K=K, n_max=n_max, f_max=f_max, n_digits=n_digits
    )

    if n_rollouts == 0 or init_calib >= calib_threshold:
        return {
            "predicted_values": argmax_decode(init_logits, n_digits),
            "digit_logits":     init_logits,
            "calib":            init_calib,
            "energy":           init_energy,
            "mcts_triggered":   False,
            "rollouts_used":    0,
            "final_clamps":     {},
            "wallclock_s":      time.perf_counter() - t0,
        }

    # ------------------------------------------------------------------
    # Phase 1: MCTS rollouts
    # ------------------------------------------------------------------
    root = MCTSNode(clamps={})

    # Initialise root with the BP score from phase 0 to prime statistics.
    root.visits = 1
    root.value  = score_state(init_calib, init_energy, init_conf)

    for _rollout_idx in range(n_rollouts):
        # ------ SELECTION: walk with UCT ------
        node  = root
        path  = [node]
        while node.children:
            node = uct_select(node, c=c_uct)
            path.append(node)

        # ------ EXPANSION ------
        # Re-run BP with current clamps to get up-to-date logits.
        cur_logits, cur_calib, cur_energy, cur_conf = _run_bp_single(
            model, batch_np, clamps=node.clamps, K=K,
            n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        # Check terminal: all unobserved variables have low entropy OR depth limit.
        depth = len(node.clamps)
        unclamped_var, unclamped_pos, max_H = _position_entropy(
            cur_logits, node.clamps, n_max=n_max, n_digits=n_digits
        )
        is_terminal = (unclamped_var < 0) or (max_H < 0.01) or (depth >= max_clamp_depth)

        if not is_terminal:
            # Expand: create 10 children, one per digit value 0-9.
            for dv in range(10):
                key       = (unclamped_var, unclamped_pos, dv)
                new_clamp = {**node.clamps, (unclamped_var, unclamped_pos): dv}
                node.children[key] = MCTSNode(clamps=new_clamp)

        # ------ SIMULATION score ------
        sim_score = score_state(cur_calib, cur_energy, cur_conf)

        # ------ BACKPROPAGATION ------
        for n in path:
            n.visits += 1
            n.value  += sim_score

    # ------------------------------------------------------------------
    # Pick best leaf: the CHILD of root with the highest Q-value (most visited
    # tie-broken by value/visits).  Walk the greedy path to collect final clamps.
    # ------------------------------------------------------------------
    best_clamps = _pick_best_clamps(root)

    # Final BP with best clamps
    final_logits, final_calib, final_energy, final_conf = _run_bp_single(
        model, batch_np, clamps=best_clamps, K=K,
        n_max=n_max, f_max=f_max, n_digits=n_digits,
    )

    return {
        "predicted_values": argmax_decode(final_logits, n_digits),
        "digit_logits":     final_logits,
        "calib":            final_calib,
        "energy":           final_energy,
        "mcts_triggered":   True,
        "rollouts_used":    n_rollouts,
        "final_clamps":     best_clamps,
        "wallclock_s":      time.perf_counter() - t0,
    }


def _pick_best_clamps(root: MCTSNode) -> dict:
    """Greedily descend the tree, at each node picking the child with highest
    Q-value (visits × value), stopping at leaves or unvisited children.
    """
    node = root
    while node.children:
        # Filter to visited children; if none, stop.
        visited = [(k, c) for k, c in node.children.items() if c.visits > 0]
        if not visited:
            break
        _, best = max(visited, key=lambda kc: kc[1].q_value)
        node = best
    return node.clamps


# ---------------------------------------------------------------------------
# Single-problem batch preparation helper
# ---------------------------------------------------------------------------

def make_single_problem_batch_np(
    record: dict,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    k_max: int = V105_K_MAX,
    n_heads: int = V105_N_HEADS,
    n_digits: int = V105_N_DIGITS,
) -> dict:
    """Convert a single GSM8K factor graph record to a B=1 numpy batch dict.

    record must be in v105 format (keys: gold_values, observed_mask,
    observed_values, factor_types, factor_args, n_factors, query_idx).
    """
    from mycelium.factor_graph_data_v105 import _records_to_batch_v105
    return _records_to_batch_v105(
        [record], n_max=n_max, f_max=f_max, k_max=k_max,
        n_heads=n_heads, n_digits=n_digits,
    )


# ---------------------------------------------------------------------------
# Batch-level convenience wrapper (for smoke / eval scripts)
# ---------------------------------------------------------------------------

def bp_and_mcts_eval(
    model: Any,
    records: list[dict],
    n_rollouts: int = _V106_N_ROLLOUTS,
    calib_threshold: float = _V106_CALIB_THRESHOLD,
    k_breaths: int | None = None,
    n_max: int = V105_N_MAX,
    f_max: int = V105_F_MAX,
    n_digits: int = V105_N_DIGITS,
    verbose: bool = True,
) -> list[dict]:
    """Run BP+MCTS on a list of v105-format records, returning per-problem result dicts.

    Each returned dict also contains:
      gold_values   : list[int] from the record
      query_idx     : int
      bp_correct    : bool (BP-only answer correct for query variable)
      mcts_correct  : bool (MCTS answer correct for query variable)
    """
    K = k_breaths if k_breaths is not None else min(_V106_K_BREATHS, V105_K_MAX)
    results = []
    for i, rec in enumerate(records):
        batch_np = make_single_problem_batch_np(rec, n_max=n_max, f_max=f_max,
                                                 n_digits=n_digits)

        # BP-only baseline (n_rollouts=0 forces BP path)
        bp_res = mcts_solve(
            model, batch_np, n_rollouts=0, calib_threshold=calib_threshold,
            k_breaths=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        # BP + MCTS
        mcts_res = mcts_solve(
            model, batch_np, n_rollouts=n_rollouts, calib_threshold=calib_threshold,
            k_breaths=K, n_max=n_max, f_max=f_max, n_digits=n_digits,
        )

        gold  = rec["gold_values"]
        qi    = int(rec["query_idx"])

        def _query_correct(predicted_values: list[int]) -> bool:
            if qi < len(gold) and qi < len(predicted_values):
                return int(round(gold[qi])) == predicted_values[qi]
            return False

        entry = {
            "idx":            i,
            "gold_values":    gold,
            "query_idx":      qi,
            "bp_correct":     _query_correct(bp_res["predicted_values"]),
            "mcts_correct":   _query_correct(mcts_res["predicted_values"]),
            "mcts_triggered": mcts_res["mcts_triggered"],
            "rollouts_used":  mcts_res["rollouts_used"],
            "bp_calib":       bp_res["calib"],
            "mcts_calib":     mcts_res["calib"],
            "bp_energy":      bp_res["energy"],
            "mcts_energy":    mcts_res["energy"],
            "bp_wallclock_s": bp_res["wallclock_s"],
            "mcts_wallclock_s": mcts_res["wallclock_s"],
        }
        results.append(entry)

        if verbose and (i % 5 == 0 or i == len(records) - 1):
            mark_bp   = "Y" if entry["bp_correct"]   else "N"
            mark_mc   = "Y" if entry["mcts_correct"]  else "N"
            trig      = "MCTS" if entry["mcts_triggered"] else " BP "
            print(
                f"  [{i:3d}] {trig}  bp={mark_bp} mcts={mark_mc}  "
                f"calib={entry['bp_calib']:.2f}→{entry['mcts_calib']:.2f}  "
                f"energy={entry['bp_energy']:.3f}→{entry['mcts_energy']:.3f}  "
                f"t={entry['mcts_wallclock_s']:.1f}s",
                flush=True,
            )

    return results
