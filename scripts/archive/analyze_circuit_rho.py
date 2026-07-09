"""analyze_circuit_rho.py — Rung-2 VERDICT instrument: rho(|z|, depth) on a
RELAXED-geometry Boolean-circuit checkpoint.

THE RUNG-2 QUESTION. Rung-1 (eval_circuit_depth.py) PASSED: the breathing engine
deduces hierarchical Boolean circuits at ~0.97 cell acc with node_acc FLAT across
topological levels (no depth ceiling). Rung-2 asks the DEEPER, geometric claim of
the three-tier program (CLAUDE.md §0 / docs/hyperbolic_mask_generator_spec.md):
when the per-type Poincare anchor field is RELAXED on the circuit DAG (FG_HYP_MASK=1,
FG_HYP_FREEZE=0 so the anchors enter the optimizer), does each gate node's learned
RADIAL position |z| TRACK its topological deduction-depth lvl? That is the
"radial position = abstraction level" hypothesis — the literal payoff that makes
the breath cycle a geodesic engine. KenKen is flat (lateral cliques); a circuit is
a DAG, so this is the first place the radial-depth bloom CAN express.

WHAT THIS SCRIPT COMPUTES.
  Per gate node v (the OUTPUT of exactly one gate factor f(v)):
    - read its anchor a_{f(v)} from model.fg_hyp_anchors_{type(v)} (the relaxed
      per-type anchor table — the param the optimizer trained under FG_HYP_FREEZE=0),
    - |z_v| = the node's radial position in the Poincare ball.
    - lvl(v) = the per-node longest-path topological depth (CircuitBatch.lvl).
  Pool over ALL valid gate nodes in the TEST split, then:
    rho = Spearman(|z_v|, lvl(v))                       [the Rung-2 statistic]
  with the FULL rho_no_ceiling-discipline control suite:
    - DEPTH-SHUFFLE permutation null (>=10,000x): permute lvl labels WITHIN each
      instance, recompute rho -> one-sided permutation p + bootstrap lower-CI.
    - optionally evaluate a DEPTH-SHUFFLE-TRAINED ckpt (--shuffle-trained) as the
      trained-flat control.

READING |z| FROM THE ANCHORS — the exact map (mirrors the engine).
  factor_masks.attach_factor_hyperbolic_params builds, per type t, one anchor table
  model.fg_hyp_anchors_{t} of shape (G_t+1, dim). The k-th type-t factor row (0-based,
  in membership row order) is placed on anchor row k; the engine
  (build_factor_hyperbolic_attn_bias -> _relation_bias_from_z -> _d_hyp_pairwise)
  uses each anchor row DIRECTLY as a Poincare BALL POINT (no exp_0 inside the bias
  path for the circuit/factor_masks arm). So the FAITHFUL per-node radius is
      |z_v| = || a_{f(v)} ||           (raw L2 norm of the trained anchor row).
  The Rung-2 spec/kenken REFERENCE phrases the anchor as a TANGENT param mapped via
  exp_0; _exp0_map is a STRICTLY MONOTONE function of the norm (|exp_0(v)| =
  tanh(|v|)), so the Spearman RANK correlation rho(|z_v|, lvl) is IDENTICAL whether
  the radius is the raw anchor norm or the exp_0-mapped norm. We report the raw norm
  by DEFAULT (faithful to the factor_masks engine) and also report the exp_0 radius
  (--radius-map exp0) for the kenken-reference convention; the rho is invariant.

  Node -> factor: factor rows are created in increasing node-index order over the
  GATE (non-leaf) nodes (circuit_data.encode_instance), so the i-th valid non-leaf
  node (in index order) is the OUTPUT of membership row i. The anchor index within
  its type t is the rank of row i among type-t rows (= cumsum of the type mask, the
  same assignment build_factor_hyperbolic_attn_bias uses). FALLBACK (node in >1
  factor, or output identification ambiguous): membership-mean anchor over every
  factor v belongs to.

THE BAR (pre-registered, printed by the script).
  Rung-2 STANDS iff on the TRUE DAG:
    bootstrap lower-CI > 0.30  AND  permutation p < 0.01     (point-|rho| >= 0.50 STRONG)
  AND the sign is MONOTONE & CONSISTENT across circuit-depth bands D2..D5 (we
  pre-register |rho| LARGE + a CONSISTENT sign, NOT a fixed sign: the optimizer may
  drive depth either inward or outward — what matters is a monotone radial axis).
  The DEPTH-SHUFFLE null must give rho ~ 0 (lower-CI <= 0.15 and/or p >= 0.05).
  HONEST NULL: rho ~ 0 on BOTH the true DAG AND the shuffle is a REAL finding — the
  undirected deduction masks do NOT induce radial-depth organization — reported
  cleanly (NOT spun as a failure of the engine; KenKen/circuit deduction can be flat
  in radius and still solve, CLAUDE.md §5 "KenKen is flat" caveat generalized).

MODES.
  MODE A (GPU, --ckpt): build the engine (mirrors eval_circuit_depth.py), load the
    relaxed ckpt, attach the hyperbolic anchor tables, read per-node |z| from the
    anchors over the TEST split, compute rho + the shuffle null + the bar. The HUMAN
    runs this on the GPU.
  MODE SELFTEST (--selftest): CPU-only. Synthetic anchors with a planted radius<->lvl
    monotone relation -> the rho core + the depth-shuffle null must recover it; a
    flat-radius synthetic must read NULL; plus the node->factor mapping unit checks.

GPU-FREE BUILD: ast.parse + CPU-import + the selftest all run with no Device ops.
Run the real readback with:
  DEV=AMD FG_HYP_MASK=1 FG_N_INSTANCES=8000 \\
    .venv/bin/python3 scripts/analyze_circuit_rho.py \\
      --ckpt .cache/fg_ckpts/<relaxed_run>/<relaxed_run>_final.safetensors
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# Path + GPU-free build gate (CPU import — no tinygrad Device ops until MODE A)
# --------------------------------------------------------------------------
_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_FILE))
sys.path.insert(0, _REPO_ROOT)

# The rho statistics core is REUSED from the KenKen Property-2 analyzer (do NOT
# re-implement — that module is the pinned, self-tested Spearman/bootstrap/perm core).
from scripts.analyze_kenken_property2 import (  # noqa: E402
    spearman_rho,
    bootstrap_ci_spearman,
    permutation_p_spearman,
)

# Pre-registered bar (mirrors the Property-2 discipline, CLAUDE.md §5 / §8).
HILL_RHO_BAR = 0.30          # bootstrap lower-CI bar (|rho|)
STRONG_RHO = 0.50            # point-|rho| STRONG sub-tier
P_STRICT = 0.01              # permutation p threshold to STAND
NULL_RHO_LO = 0.15           # null: lower-CI |rho| <= 0.15
P_NULL = 0.05                # null: permutation p >= 0.05
N_PERM_DEFAULT = 10000
N_BOOT_DEFAULT = 10000
DEPTH_BAND_ORDER = ["D2", "D3", "D4", "D5"]


# --------------------------------------------------------------------------
# Two-sided / sign-agnostic rho stats (Rung-2 pre-registers |rho|, not a sign)
# --------------------------------------------------------------------------

def _abs_lower_ci(lo: float, hi: float) -> float:
    """Lower bound on |rho| from a [lo, hi] CI for rho.

    If the CI straddles 0 the |rho| lower bound is 0 (sign undetermined). Otherwise
    it is min(|lo|, |hi|) — the closest the magnitude gets to 0 across the interval.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    if lo <= 0.0 <= hi:
        return 0.0
    return float(min(abs(lo), abs(hi)))


def permutation_p_two_sided(z_radius: np.ndarray, lvl: np.ndarray,
                            inst_id: np.ndarray, n_perm: int,
                            rng: np.random.Generator) -> float:
    """Sign-AGNOSTIC depth-shuffle permutation p for |rho(|z|, lvl)|.

    THE CONTROL (rho_no_ceiling discipline, circuit form): the null permutes lvl
    labels WITHIN each instance (inst_id groups nodes by circuit), preserving the
    per-instance depth multiset (the matched NULL — exactly CircuitBatch.shuffle_depth
    but applied here so the SAME pooled radii are reused). Counts permuted |rho| >=
    observed |rho|. Two-sided because Rung-2 pre-registers a CONSISTENT-sign large
    |rho|, not a fixed direction.

    p = (1 + #{|perm_rho| >= |obs_rho|}) / (n_perm + 1)   [add-one; never 0].
    """
    obs = spearman_rho(z_radius, lvl)
    if not np.isfinite(obs):
        return 1.0
    obs_abs = abs(obs)
    insts = np.unique(inst_id)
    # Pre-split node indices per instance for fast within-instance permutation.
    groups = [np.where(inst_id == g)[0] for g in insts]
    count = 0
    lvl_perm = lvl.copy()
    for _ in range(n_perm):
        for idx in groups:
            if idx.size > 1:
                lvl_perm[idx] = lvl[idx][rng.permutation(idx.size)]
        rp = spearman_rho(z_radius, lvl_perm)
        if np.isfinite(rp) and abs(rp) >= obs_abs - 1e-12:
            count += 1
    return (1 + count) / (n_perm + 1)


def _depth_shuffle_pooled(lvl: np.ndarray, inst_id: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """One within-instance permutation of lvl (the matched depth-shuffle NULL)."""
    out = lvl.copy()
    for g in np.unique(inst_id):
        idx = np.where(inst_id == g)[0]
        if idx.size > 1:
            out[idx] = lvl[idx][rng.permutation(idx.size)]
    return out


def rho_block(z_radius: np.ndarray, lvl: np.ndarray, inst_id: np.ndarray,
              n_perm: int, n_boot: int, rng: np.random.Generator) -> dict:
    """Full Rung-2 statistic block for one pool of (|z|, lvl) node pairs.

    Returns rho (point), bootstrap CI, |rho| lower-CI, the within-instance
    depth-shuffle permutation p (sign-agnostic), and the pooled-null rho (one
    shuffle, descriptive) + the radius/lvl spread guards.
    """
    z_radius = np.asarray(z_radius, dtype=np.float64)
    lvl = np.asarray(lvl, dtype=np.float64)
    inst_id = np.asarray(inst_id)
    n = z_radius.size
    out: dict = {"n": int(n)}
    if n < 3:
        out.update({"rho": float("nan"), "ci_lower": float("nan"),
                    "ci_upper": float("nan"), "abs_lower_ci": float("nan"),
                    "p_perm": float("nan"), "radius_spread": False,
                    "lvl_spread": False, "null_rho_one_shuffle": float("nan")})
        return out

    radius_spread = float(np.unique(np.round(z_radius, 9)).size) > 1
    lvl_spread = int(np.unique(lvl).size) > 1

    rho = spearman_rho(z_radius, lvl)
    lo, hi, _ = bootstrap_ci_spearman(z_radius, lvl, n_boot=n_boot, rng=rng)
    abs_lo = _abs_lower_ci(lo, hi)
    p = permutation_p_two_sided(z_radius, lvl, inst_id, n_perm=n_perm, rng=rng)

    # One pooled depth-shuffle (descriptive: the null rho on a single matched shuffle).
    lvl_null = _depth_shuffle_pooled(lvl, inst_id, rng)
    null_rho = spearman_rho(z_radius, lvl_null)

    out.update({
        "rho": rho,
        "ci_lower": lo, "ci_upper": hi, "abs_lower_ci": abs_lo,
        "p_perm": p,
        "radius_spread": radius_spread, "lvl_spread": lvl_spread,
        "null_rho_one_shuffle": null_rho,
        "radius_min": float(z_radius.min()), "radius_max": float(z_radius.max()),
        "lvl_min": float(lvl.min()), "lvl_max": float(lvl.max()),
    })
    return out


def verdict_from_block(block: dict, shuffle_null_block: dict | None) -> dict:
    """Apply the pre-registered Rung-2 bar to the TRUE-DAG block + the shuffle null.

    STANDS : abs_lower_ci > 0.30 AND p_perm < 0.01     (STRONG if |rho| >= 0.50)
    NULL   : abs_lower_ci <= 0.15 OR p_perm >= 0.05     (and the spread guards pass)
    UNTESTABLE: radius has no spread (anchors collapsed) OR lvl has no spread.
    The shuffle null is required to read ~0 (its own abs_lower_ci <= 0.15 or p>=0.05);
    if the TRUE block STANDS but the shuffle null ALSO stands, that is a LEAK flag
    (the matched null should destroy the correlation).
    """
    rho = block.get("rho", float("nan"))
    abs_lo = block.get("abs_lower_ci", float("nan"))
    p = block.get("p_perm", float("nan"))
    radius_spread = bool(block.get("radius_spread", False))
    lvl_spread = bool(block.get("lvl_spread", False))

    if not radius_spread:
        return {"verdict": "UNTESTABLE",
                "reason": ("anchors collapsed: per-node |z| has NO spread across gate "
                           "nodes (the relaxed field did not differentiate by node) — "
                           "cannot test radial-depth organization")}
    if not lvl_spread:
        return {"verdict": "UNTESTABLE",
                "reason": "lvl has no spread in the pooled gate nodes (degenerate corpus)"}

    stands = (np.isfinite(abs_lo) and np.isfinite(p)
              and abs_lo > HILL_RHO_BAR and p < P_STRICT)
    is_null = ((np.isfinite(abs_lo) and abs_lo <= NULL_RHO_LO)
               or (np.isfinite(p) and p >= P_NULL))

    # Did the matched depth-shuffle null correctly collapse to ~0?
    null_ok = True
    null_note = "no shuffle-null block supplied"
    if shuffle_null_block is not None:
        n_abs = shuffle_null_block.get("abs_lower_ci", float("nan"))
        n_p = shuffle_null_block.get("p_perm", float("nan"))
        null_collapsed = ((np.isfinite(n_abs) and n_abs <= NULL_RHO_LO)
                          or (np.isfinite(n_p) and n_p >= P_NULL))
        null_ok = bool(null_collapsed)
        null_note = (f"shuffle-null rho={_fmt(shuffle_null_block.get('rho'))} "
                     f"abs_lo={_fmt(n_abs)} p={_fmt(n_p, 4)} "
                     f"{'(collapsed ~0, good)' if null_collapsed else '(DID NOT collapse — LEAK)'}")

    if stands:
        tier = "STRONG" if (np.isfinite(rho) and abs(rho) >= STRONG_RHO) else "STANDS"
        verdict = f"RUNG-2 {tier}"
        sign = "INWARD (deep=core)" if rho < 0 else "OUTWARD (deep=rim)"
        reason = (f"abs_lower_ci={_fmt(abs_lo)} > {HILL_RHO_BAR} AND "
                  f"p_perm={_fmt(p, 4)} < {P_STRICT}; rho={_fmt(rho)} ({sign})")
        if not null_ok:
            verdict = "LEAK-SUSPECT"
            reason += f"; BUT the depth-shuffle null did NOT collapse: {null_note}"
        return {"verdict": verdict, "reason": reason, "null_note": null_note}

    if is_null:
        return {"verdict": "NULL",
                "reason": (f"abs_lower_ci={_fmt(abs_lo)} (<= {NULL_RHO_LO}) and/or "
                           f"p_perm={_fmt(p, 4)} (>= {P_NULL}); rho={_fmt(rho)} ~ 0 — "
                           f"the relaxed undirected deduction masks do NOT induce "
                           f"radial-depth organization (an HONEST null finding)"),
                "null_note": null_note}

    return {"verdict": "WEAK/INCONCLUSIVE",
            "reason": (f"rho={_fmt(rho)} abs_lower_ci={_fmt(abs_lo)} p_perm={_fmt(p, 4)} "
                       f"clears neither the STAND bar (abs_lo>{HILL_RHO_BAR} & p<{P_STRICT}) "
                       f"nor the NULL bar"),
            "null_note": null_note}


# --------------------------------------------------------------------------
# PER-NODE |z| FROM THE ANCHORS (the load-bearing reader)
# --------------------------------------------------------------------------

def node_factor_anchor_index(membership_b: np.ndarray, latent_type_b: np.ndarray,
                             cell_valid_b: np.ndarray, is_leaf_b: np.ndarray,
                             n_factor_types: int) -> dict:
    """Map each VALID GATE node v -> (type t, anchor_index within type t).

    Mirrors build_factor_hyperbolic_attn_bias's anchor assignment EXACTLY:
      - factor rows are created in increasing node-index order over gate (non-leaf)
        nodes (circuit_data.encode_instance), so the i-th membership row's OUTPUT
        node is the i-th valid non-leaf node in index order.
      - within a type t, the k-th type-t factor row (0-based, row order) -> anchor k
        (= cumsum of the type-t mask, minus 1). This is identical to the engine's
        anchor_idx = cumsum(sel_t) - 1.

    Returns dict:
      node_type[v]   : int type t of the gate factor v is the OUTPUT of (or -1).
      node_anchor[v] : int anchor index within type t (or -1).
      membership_factor_rows[v] : list of (t, anchor_idx) for EVERY factor v belongs
                                  to (used by the fallback membership-mean path).
    membership_b  : (L, s_max) float.
    latent_type_b : (L,) int (type idx in 0..T-1, or sentinel == T for padding).
    cell_valid_b  : (s_max,) float.
    is_leaf_b     : (s_max,) float (1 leaf / 0 gate-or-pad).
    """
    L, s_max = membership_b.shape
    T = int(n_factor_types)
    sentinel = T

    # Per-type running rank (cumsum) -> anchor index for each factor row.
    type_rank = {t: -1 for t in range(T)}
    row_anchor: dict[int, tuple[int, int]] = {}   # row -> (type, anchor_idx)
    real_rows_in_order: list[int] = []            # factor rows with a real type
    for g in range(L):
        t = int(latent_type_b[g])
        if t == sentinel or t < 0 or t >= T:
            continue
        type_rank[t] += 1
        row_anchor[g] = (t, type_rank[t])
        real_rows_in_order.append(g)

    # OUTPUT node of factor row g = the g-th valid non-leaf node in index order.
    gate_nodes_in_order = [v for v in range(s_max)
                           if cell_valid_b[v] > 0.5 and is_leaf_b[v] < 0.5]

    node_type = np.full((s_max,), -1, dtype=np.int64)
    node_anchor = np.full((s_max,), -1, dtype=np.int64)
    # Pair up factor rows (in order) with gate output nodes (in order).
    for i, g in enumerate(real_rows_in_order):
        if i < len(gate_nodes_in_order):
            v = gate_nodes_in_order[i]
            t, a = row_anchor[g]
            node_type[v] = t
            node_anchor[v] = a

    # Membership fallback: every factor each node belongs to (member rows).
    membership_factor_rows: dict[int, list[tuple[int, int]]] = {}
    for v in range(s_max):
        if cell_valid_b[v] <= 0.5 or is_leaf_b[v] >= 0.5:
            continue
        belongs = []
        for g in real_rows_in_order:
            if membership_b[g, v] > 0.5:
                belongs.append(row_anchor[g])   # (type, anchor_idx)
        membership_factor_rows[v] = belongs

    return {"node_type": node_type, "node_anchor": node_anchor,
            "membership_factor_rows": membership_factor_rows,
            "gate_nodes_in_order": gate_nodes_in_order}


def _radius_from_anchor(anchor_vec: np.ndarray, radius_map: str,
                        exp0_fn=None) -> float:
    """|z| from a single anchor coordinate row.

    radius_map='raw'  : || a ||  (faithful to factor_masks: the anchor IS the ball
                        point, used directly in _d_hyp_pairwise — NO exp_0).
    radius_map='exp0' : || exp_0(a) || = tanh(|a|)  (kenken-reference tangent->ball
                        convention). STRICTLY MONOTONE in |a| -> the Spearman RANK
                        rho is identical to 'raw'; reported for convention parity.
    """
    if radius_map == "exp0" and exp0_fn is not None:
        from tinygrad import Tensor, dtypes
        z = exp0_fn(Tensor(anchor_vec.reshape(1, -1).astype(np.float32),
                           dtype=dtypes.float)).realize().numpy()[0]
        return float(np.linalg.norm(z))
    return float(np.linalg.norm(anchor_vec))


def per_node_radii(anchors_by_type: dict[int, np.ndarray],
                   membership_np: np.ndarray, latent_type_np: np.ndarray,
                   cell_valid_np: np.ndarray, is_leaf_np: np.ndarray,
                   lvl_np: np.ndarray, n_factor_types: int,
                   radius_map: str = "raw", exp0_fn=None) -> dict:
    """Pool per-node (|z|, lvl, instance_id, type, band-key-source) over a batch set.

    anchors_by_type : {t: (G_t+1, dim) np anchor table from model.fg_hyp_anchors_{t}}.
    membership_np   : (B, L, s_max), latent_type_np: (B, L),
    cell_valid_np / is_leaf_np / lvl_np : (B, s_max).
    Returns arrays z_radius, lvl, inst_id, node_type, plus a fallback counter.
    """
    B = membership_np.shape[0]
    z_list, lvl_list, inst_list, type_list = [], [], [], []
    n_fallback = 0
    n_primary = 0
    n_skipped = 0
    for b in range(B):
        mp = node_factor_anchor_index(
            membership_np[b], latent_type_np[b], cell_valid_np[b],
            is_leaf_np[b], n_factor_types)
        node_type = mp["node_type"]
        node_anchor = mp["node_anchor"]
        memrows = mp["membership_factor_rows"]
        s_max = membership_np.shape[2]
        for v in range(s_max):
            if cell_valid_np[b, v] <= 0.5 or is_leaf_np[b, v] >= 0.5:
                continue   # leaves are GIVEN (no learned anchor as an output)
            lv = int(lvl_np[b, v])
            if lv < 0:
                continue
            t = int(node_type[v])
            a = int(node_anchor[v])
            radius = None
            if t >= 0 and a >= 0 and t in anchors_by_type and a < anchors_by_type[t].shape[0]:
                radius = _radius_from_anchor(anchors_by_type[t][a], radius_map, exp0_fn)
                n_primary += 1
            else:
                # FALLBACK: membership-mean anchor over every factor v belongs to.
                rows = memrows.get(v, [])
                vecs = [anchors_by_type[tt][aa] for (tt, aa) in rows
                        if tt in anchors_by_type and aa < anchors_by_type[tt].shape[0]]
                if vecs:
                    mean_anchor = np.mean(np.stack(vecs, axis=0), axis=0)
                    radius = _radius_from_anchor(mean_anchor, radius_map, exp0_fn)
                    n_fallback += 1
            if radius is None:
                n_skipped += 1
                continue
            z_list.append(radius)
            lvl_list.append(lv)
            inst_list.append(b)           # within-CALL instance id (per batch row)
            type_list.append(t)
    return {
        "z_radius": np.array(z_list, dtype=np.float64),
        "lvl": np.array(lvl_list, dtype=np.float64),
        "inst_id": np.array(inst_list, dtype=np.int64),
        "node_type": np.array(type_list, dtype=np.int64),
        "n_primary": n_primary, "n_fallback": n_fallback, "n_skipped": n_skipped,
    }


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None:
        return "  -  "
    try:
        if not np.isfinite(x):
            return " nan "
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.{nd}f}"


def print_summary(result: dict) -> None:
    print("=" * 92, flush=True)
    print("CIRCUIT RUNG-2 — rho(|z|, depth) ON A RELAXED-GEOMETRY CIRCUIT CHECKPOINT", flush=True)
    print("=" * 92, flush=True)
    print(f"ckpt={result.get('ckpt')}", flush=True)
    print(f"radius_map={result.get('radius_map')}  K={result.get('K')}  "
          f"n_perm={result.get('n_perm')}  n_boot={result.get('n_boot')}  "
          f"seed={result.get('seed')}  shuffle_trained_ckpt={result.get('shuffle_trained')}",
          flush=True)
    pn = result.get("pool_meta", {})
    print(f"pooled gate nodes={pn.get('n')}  "
          f"primary={pn.get('n_primary')}  fallback={pn.get('n_fallback')}  "
          f"skipped={pn.get('n_skipped')}", flush=True)
    print("-" * 92, flush=True)

    print("PRE-REGISTERED BAR (printed before the read):", flush=True)
    print(f"  STANDS : |rho| bootstrap lower-CI > {HILL_RHO_BAR}  AND  perm p < {P_STRICT}"
          f"   (point-|rho| >= {STRONG_RHO} => STRONG)", flush=True)
    print(f"  sign   : pre-register a CONSISTENT sign + LARGE |rho| (NOT a fixed sign — "
          f"inward OR outward is fine, monotone is what matters)", flush=True)
    print(f"  NULL   : |rho| lower-CI <= {NULL_RHO_LO}  OR  perm p >= {P_NULL}; "
          f"the depth-SHUFFLE null must read ~0", flush=True)
    print("-" * 92, flush=True)

    tb = result["true_block"]
    print("TRUE DAG (all pooled gate nodes):", flush=True)
    print(f"  rho={_fmt(tb['rho'])}  CI=[{_fmt(tb['ci_lower'])}, {_fmt(tb['ci_upper'])}]  "
          f"|rho|_loCI={_fmt(tb['abs_lower_ci'])}  p_perm={_fmt(tb['p_perm'], 4)}  "
          f"n={tb['n']}", flush=True)
    print(f"  radius range=[{_fmt(tb.get('radius_min'))}, {_fmt(tb.get('radius_max'))}]  "
          f"lvl range=[{_fmt(tb.get('lvl_min'))}, {_fmt(tb.get('lvl_max'))}]  "
          f"radius_spread={tb.get('radius_spread')}  lvl_spread={tb.get('lvl_spread')}",
          flush=True)

    nb = result.get("shuffle_null_block")
    if nb is not None:
        print("\nDEPTH-SHUFFLE NULL (within-instance lvl permutation, the matched control):",
              flush=True)
        print(f"  rho={_fmt(nb['rho'])}  CI=[{_fmt(nb['ci_lower'])}, {_fmt(nb['ci_upper'])}]  "
              f"|rho|_loCI={_fmt(nb['abs_lower_ci'])}  p_perm={_fmt(nb['p_perm'], 4)}", flush=True)

    bands = result.get("band_blocks")
    if bands:
        print("\nPER CIRCUIT-DEPTH BAND (D2..D5):", flush=True)
        print(f"  {'band':<6} {'rho':>7} {'|rho|loCI':>10} {'p':>8} {'n':>7}", flush=True)
        for band in DEPTH_BAND_ORDER:
            blk = bands.get(band)
            if blk is None:
                continue
            print(f"  {band:<6} {_fmt(blk['rho']):>7} {_fmt(blk['abs_lower_ci']):>10} "
                  f"{_fmt(blk['p_perm'], 4):>8} {blk['n']:>7}", flush=True)
        signs = [np.sign(bands[b]["rho"]) for b in DEPTH_BAND_ORDER
                 if b in bands and np.isfinite(bands[b].get("rho", float("nan")))]
        consistent = (len(signs) >= 2 and len(set(signs)) == 1)
        print(f"  sign consistency across bands: "
              f"{'CONSISTENT' if consistent else 'MIXED/insufficient'} "
              f"(signs={[int(s) for s in signs]})", flush=True)

    pertype = result.get("type_blocks")
    if pertype:
        print("\nPER GATE-TYPE (descriptive — one anchor field per type):", flush=True)
        for t in sorted(pertype.keys()):
            blk = pertype[t]
            print(f"  type {t}: rho={_fmt(blk['rho'])}  |rho|loCI={_fmt(blk['abs_lower_ci'])}  "
                  f"p={_fmt(blk['p_perm'], 4)}  n={blk['n']}", flush=True)

    print("\n" + "=" * 92, flush=True)
    v = result["verdict"]
    print(f"VERDICT: {v['verdict']}", flush=True)
    print(f"  reason: {v['reason']}", flush=True)
    if "null_note" in v:
        print(f"  null  : {v['null_note']}", flush=True)
    print("=" * 92, flush=True)
    print("INTERPRETATION:", flush=True)
    print("  CONFIRMS radial-depth organization: TRUE-DAG STANDS (|rho|_loCI>0.30, p<0.01,"
          " consistent sign across bands) WHILE the depth-shuffle null reads ~0.", flush=True)
    print("  REFUTES it (HONEST NULL): rho ~ 0 on BOTH true + shuffle — the relaxed"
          " undirected deduction masks do not organize |z| by deduction-depth (a real,"
          " cleanly-reported finding, NOT an engine failure).", flush=True)
    print("  UNTESTABLE: the relaxed anchors collapsed (no |z| spread) — re-train / check"
          " the relaxation actually moved the anchors.", flush=True)
    print("=" * 92, flush=True)


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else str(k)): _json_sanitize(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# --------------------------------------------------------------------------
# MODE A — model over the relaxed circuit ckpt (GPU; mirrors eval_circuit_depth.py)
# --------------------------------------------------------------------------

def run_mode_a(ckpt_path: str, n_perm: int, n_boot: int, seed: int,
               radius_map: str, shuffle_trained: bool, out_path: str | None) -> dict:
    """Load the relaxed circuit ckpt, read per-node |z| from the anchors over the
    TEST split, compute rho + the depth-shuffle null + the bar.

    *** RUNS THE MODEL ON THE GPU. *** Mirrors eval_circuit_depth.py's model build +
    load + the CircuitLoader test split (seed=42 by default). A human runs this.
    """
    import gc
    from tinygrad import Tensor, Device, dtypes  # noqa: F401
    from tinygrad.helpers import getenv

    K = int(getenv("FG_K_MAX", getenv("K", "16")))
    EVAL_BATCH = int(getenv("EVAL_BATCH", getenv("BATCH", "8")))
    N_INSTANCES = int(getenv("FG_N_INSTANCES", "8000"))
    S_MAX = int(getenv("FG_S_MAX", "49"))
    N_VALUES = 2

    from mycelium import Config, BreathingTransformer  # noqa: F401
    from mycelium.loader import _load_state, load_breathing
    from mycelium.factor_graph_engine import (
        FactorGraphSpec, attach_factor_graph_params, FG_HYP_MASK,
    )
    from mycelium.factor_masks import attach_factor_hyperbolic_params
    from mycelium.circuit_data import CircuitLoader
    from mycelium.kenken import _exp0_map
    # Mirror eval_circuit_depth.py's ckpt loader (same key set).
    from scripts.eval_circuit_depth import cast_layers_fp32, load_ckpt

    if not FG_HYP_MASK:
        print("*** WARNING: FG_HYP_MASK is OFF — no hyperbolic anchors will be attached. "
              "Set FG_HYP_MASK=1 to read a relaxed-geometry checkpoint. ***", flush=True)

    gate_types_env = getenv("FG_CIRCUIT_GATE_TYPES", "").strip()
    use_xor = int(getenv("FG_CIRCUIT_XOR", "0")) > 0
    if gate_types_env:
        gate_types = tuple(g.strip().upper() for g in gate_types_env.split(",") if g.strip())
    elif use_xor:
        gate_types = ("AND", "OR", "NOT", "XOR")
    else:
        gate_types = ("AND", "OR", "NOT")

    print(f"[MODE A] CircuitLoader(n_instances={N_INSTANCES}, seed={seed}, "
          f"gate_types={gate_types}, shuffle_depth={shuffle_trained})", flush=True)
    # shuffle_depth=True only for the trained-flat control arm (its ckpt was trained
    # on shuffled depth labels; we read |z| the same way but lvl is the matched null).
    loader = CircuitLoader(
        n_instances=N_INSTANCES, s_max=S_MAX, n_values=N_VALUES,
        batch_size=EVAL_BATCH, seed=seed, gate_types=gate_types,
        shuffle_depth=shuffle_trained,
    )
    n_factor_types = int(loader.n_factor_types)
    print(f"  test set: {len(loader.test_records)} instances  T={n_factor_types}  "
          f"n_gates_max={loader.n_gates_max}", flush=True)

    spec = FactorGraphSpec(
        s_max=S_MAX, n_values=N_VALUES, n_factor_types=n_factor_types,
        n_heads=16, k_max=K, has_factor_inlet=False,
    )

    print("loading Pythia-410M -> BreathingTransformer...", flush=True)
    cfg = Config()
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    gc.collect()
    cast_layers_fp32(model)
    attach_factor_graph_params(model, hidden=cfg.hidden, spec=spec)

    print("[FG_HYP_MASK] building circuit anchor tables (sized from a ref batch) ...",
          flush=True)
    _ref = loader.sample_batch()
    _mem = _ref.membership.realize().numpy()
    _lt = _ref.latent_type.realize().numpy()
    attach_factor_hyperbolic_params(
        model, n_heads=spec.n_heads, n_factor_types=spec.n_factor_types,
        s_max=spec.s_max, membership_np=_mem, latent_type_np=_lt,
    )
    del _ref, _mem, _lt
    Device[Device.DEFAULT].synchronize()

    print(f"loading checkpoint (relaxed anchors): {ckpt_path}", flush=True)
    load_ckpt(model, ckpt_path)

    # Pull the (post-load) per-type anchor tables to numpy ONCE.
    from mycelium.factor_masks import cell_mp_head_allocation, CELL_MP_HEAD_GLOBAL
    G_heads = max(1, spec.n_heads // 16)
    alloc = cell_mp_head_allocation(spec.n_factor_types, spec.n_heads, G_heads)
    used_types = sorted({int(t) for t in alloc if int(t) != CELL_MP_HEAD_GLOBAL})
    anchors_by_type: dict[int, np.ndarray] = {}
    for t in used_types:
        at = getattr(model, f"fg_hyp_anchors_{t}", None)
        if at is not None:
            anchors_by_type[t] = at.realize().numpy().astype(np.float64)
            print(f"  fg_hyp_anchors_{t}: shape={anchors_by_type[t].shape}  "
                  f"|row| range=[{np.linalg.norm(anchors_by_type[t], axis=-1).min():.4f}, "
                  f"{np.linalg.norm(anchors_by_type[t], axis=-1).max():.4f}]", flush=True)

    # Accumulate per-node (|z|, lvl, instance, type, band) over the WHOLE test split.
    z_all, lvl_all, inst_all, type_all, band_all = [], [], [], [], []
    n_primary = n_fallback = n_skipped = 0
    global_inst = 0
    n_seen = 0
    total = len(loader.test_records)
    for batch in loader.iter_eval(batch_size=EVAL_BATCH):
        mem_np = batch.membership.realize().numpy()
        lt_np = batch.latent_type.realize().numpy()
        cv_np = batch.cell_valid.realize().numpy()
        leaf_np = batch.is_leaf
        lvl_np = batch.lvl
        bands_b = batch.band
        pooled = per_node_radii(
            anchors_by_type, mem_np, lt_np, cv_np, leaf_np, lvl_np,
            n_factor_types, radius_map=radius_map, exp0_fn=_exp0_map)
        B_cur = mem_np.shape[0]
        # Stop once we've covered the real corpus (iter_eval pads the last batch).
        for j in range(pooled["z_radius"].size):
            b_local = int(pooled["inst_id"][j])
            if (n_seen + b_local) >= total:
                continue
            z_all.append(pooled["z_radius"][j])
            lvl_all.append(pooled["lvl"][j])
            inst_all.append(global_inst + b_local)
            type_all.append(int(pooled["node_type"][j]))
            band_all.append(bands_b[b_local])
        n_primary += pooled["n_primary"]
        n_fallback += pooled["n_fallback"]
        n_skipped += pooled["n_skipped"]
        # advance instance ids only over the REAL rows of this batch
        real_rows = min(B_cur, max(0, total - n_seen))
        global_inst += real_rows
        n_seen += real_rows
        if n_seen >= total:
            break

    z_all = np.array(z_all, dtype=np.float64)
    lvl_all = np.array(lvl_all, dtype=np.float64)
    inst_all = np.array(inst_all, dtype=np.int64)
    type_all = np.array(type_all, dtype=np.int64)
    band_all = np.array(band_all)

    rng = np.random.default_rng(seed)
    true_block = rho_block(z_all, lvl_all, inst_all, n_perm, n_boot, rng)

    # The depth-shuffle null (recompute rho on a within-instance lvl permutation of
    # the SAME radii — the matched NULL).  Distinct from --shuffle-trained (which
    # loads a ckpt trained on shuffled labels).
    rng2 = np.random.default_rng(seed + 1)
    lvl_null = _depth_shuffle_pooled(lvl_all, inst_all, rng2)
    shuffle_null_block = rho_block(z_all, lvl_null, inst_all, n_perm, n_boot, rng2)

    # Per circuit-depth band.
    band_blocks: dict[str, dict] = {}
    for band in DEPTH_BAND_ORDER:
        m = (band_all == band)
        if int(m.sum()) >= 3:
            rb = np.random.default_rng(seed + 100 + DEPTH_BAND_ORDER.index(band))
            band_blocks[band] = rho_block(
                z_all[m], lvl_all[m], inst_all[m], n_perm, n_boot, rb)

    # Per gate-type (descriptive).
    type_blocks: dict[int, dict] = {}
    for t in sorted(set(int(x) for x in type_all.tolist())):
        m = (type_all == t)
        if int(m.sum()) >= 3:
            rt = np.random.default_rng(seed + 200 + t)
            type_blocks[t] = rho_block(z_all[m], lvl_all[m], inst_all[m],
                                       n_perm, n_boot, rt)

    verdict = verdict_from_block(true_block, shuffle_null_block)

    result = {
        "ckpt": ckpt_path, "radius_map": radius_map, "K": K,
        "n_perm": n_perm, "n_boot": n_boot, "seed": seed,
        "shuffle_trained": shuffle_trained,
        "pool_meta": {"n": int(z_all.size), "n_primary": n_primary,
                      "n_fallback": n_fallback, "n_skipped": n_skipped},
        "true_block": true_block,
        "shuffle_null_block": shuffle_null_block,
        "band_blocks": band_blocks,
        "type_blocks": type_blocks,
        "verdict": verdict,
    }
    print_summary(result)
    if out_path:
        with open(out_path, "w") as f:
            json.dump(_json_sanitize(result), f, indent=2)
        print(f"\n[wrote verdict JSON] {out_path}", flush=True)
    return result


# --------------------------------------------------------------------------
# SELF-TEST (CPU-only: synthetic radii + node->factor mapping unit checks)
# --------------------------------------------------------------------------

def _selftest() -> bool:
    print("[selftest] CPU-only synthetic + mapping unit checks ...", flush=True)
    ok = True

    # ---- 1. node->factor->anchor mapping on a tiny hand circuit ----
    # 5 nodes: 0,1 leaves; 2=AND(0,1) lvl1; 3=NOT(2) lvl2; 4=OR(0,3) lvl3.
    # gate_types order (AND,OR,NOT): AND idx 0, OR idx 1, NOT idx 2. T=3, sentinel=3.
    s_max = 8
    L = 4  # >= n_gates (3) with one pad row
    membership = np.zeros((L, s_max), dtype=np.float32)
    latent_type = np.full((L,), 3, dtype=np.int32)   # sentinel pad
    cell_valid = np.zeros((s_max,), dtype=np.float32)
    is_leaf = np.zeros((s_max,), dtype=np.float32)
    for v in (0, 1, 2, 3, 4):
        cell_valid[v] = 1.0
    is_leaf[0] = is_leaf[1] = 1.0
    # gate rows created in node order over gate nodes: 2 (AND), 3 (NOT), 4 (OR).
    membership[0, 2] = 1.0; membership[0, 0] = 1.0; membership[0, 1] = 1.0; latent_type[0] = 0  # AND
    membership[1, 3] = 1.0; membership[1, 2] = 1.0; latent_type[1] = 2                          # NOT
    membership[2, 4] = 1.0; membership[2, 0] = 1.0; membership[2, 3] = 1.0; latent_type[2] = 1  # OR

    mp = node_factor_anchor_index(membership, latent_type, cell_valid, is_leaf,
                                  n_factor_types=3)
    nt, na = mp["node_type"], mp["node_anchor"]
    # node 2 -> type AND(0), anchor 0; node 3 -> NOT(2), anchor 0; node 4 -> OR(1), anchor 0.
    assert nt[2] == 0 and na[2] == 0, (nt[2], na[2])
    assert nt[3] == 2 and na[3] == 0, (nt[3], na[3])
    assert nt[4] == 1 and na[4] == 0, (nt[4], na[4])
    # leaves carry no output anchor.
    assert nt[0] == -1 and nt[1] == -1
    # membership fallback: node 0 is an operand of AND(row0) and OR(row2) -> but node 0
    # is a LEAF so it's excluded; node 2 belongs to AND(out) + NOT(operand).
    assert (0, 0) in mp["membership_factor_rows"][2] and (2, 0) in mp["membership_factor_rows"][2]
    print("  [ok] node->factor->anchor mapping (hand circuit)", flush=True)

    # ---- 2. anchor-index ranks within a type (two factors of same type) ----
    s2 = 6; L2 = 3
    mem2 = np.zeros((L2, s2), dtype=np.float32)
    lt2 = np.full((L2,), 3, dtype=np.int32)
    cv2 = np.zeros((s2,), dtype=np.float32); leaf2 = np.zeros((s2,), dtype=np.float32)
    for v in (0, 1, 2, 3): cv2[v] = 1.0
    leaf2[0] = leaf2[1] = 1.0
    mem2[0, 2] = 1.0; mem2[0, 0] = 1.0; mem2[0, 1] = 1.0; lt2[0] = 0   # AND -> rank 0
    mem2[1, 3] = 1.0; mem2[1, 0] = 1.0; mem2[1, 2] = 1.0; lt2[1] = 0   # AND -> rank 1
    mp2 = node_factor_anchor_index(mem2, lt2, cv2, leaf2, n_factor_types=3)
    assert mp2["node_anchor"][2] == 0 and mp2["node_anchor"][3] == 1, \
        (mp2["node_anchor"][2], mp2["node_anchor"][3])
    print("  [ok] within-type anchor index ranks (cumsum assignment)", flush=True)

    # ---- 3. planted monotone radius<->lvl => rho STANDS; shuffle null ~0 ----
    rng = np.random.default_rng(0)
    n_inst = 60
    z_list, lvl_list, inst_list = [], [], []
    for b in range(n_inst):
        depth = rng.integers(2, 6)
        for lv in range(1, depth + 1):
            # planted: radius increases with lvl (outward) + small noise.
            z_list.append(0.1 * lv + 0.01 * rng.standard_normal())
            lvl_list.append(lv)
            inst_list.append(b)
    z_arr = np.array(z_list); lvl_arr = np.array(lvl_list, dtype=np.float64)
    inst_arr = np.array(inst_list)
    blk = rho_block(z_arr, lvl_arr, inst_arr, n_perm=2000, n_boot=2000,
                    rng=np.random.default_rng(1))
    null_blk = rho_block(z_arr, _depth_shuffle_pooled(lvl_arr, inst_arr,
                                                      np.random.default_rng(2)),
                         inst_arr, n_perm=2000, n_boot=2000, rng=np.random.default_rng(3))
    v_plant = verdict_from_block(blk, null_blk)
    assert v_plant["verdict"].startswith("RUNG-2"), (v_plant, blk)
    assert blk["abs_lower_ci"] > HILL_RHO_BAR and blk["p_perm"] < P_STRICT, blk
    # the shuffle null must collapse.
    assert null_blk["p_perm"] >= P_NULL or null_blk["abs_lower_ci"] <= NULL_RHO_LO, null_blk
    print(f"  [ok] planted monotone: rho={blk['rho']:.3f} |rho|loCI={blk['abs_lower_ci']:.3f} "
          f"p={blk['p_perm']:.4f} -> {v_plant['verdict']}; "
          f"null rho={null_blk['rho']:.3f} p={null_blk['p_perm']:.4f}", flush=True)

    # ---- 4. flat radius => NULL on BOTH (the honest-null path) ----
    z_flat = 0.5 + 0.001 * rng.standard_normal(z_arr.size)
    blk_flat = rho_block(z_flat, lvl_arr, inst_arr, n_perm=2000, n_boot=2000,
                         rng=np.random.default_rng(4))
    null_flat = rho_block(z_flat, _depth_shuffle_pooled(lvl_arr, inst_arr,
                                                        np.random.default_rng(5)),
                          inst_arr, n_perm=2000, n_boot=2000, rng=np.random.default_rng(6))
    v_flat = verdict_from_block(blk_flat, null_flat)
    assert v_flat["verdict"] in ("NULL", "WEAK/INCONCLUSIVE", "UNTESTABLE"), (v_flat, blk_flat)
    print(f"  [ok] flat radius: rho={blk_flat['rho']:.3f} "
          f"|rho|loCI={blk_flat['abs_lower_ci']:.3f} p={blk_flat['p_perm']:.4f} "
          f"-> {v_flat['verdict']}", flush=True)

    # ---- 5. collapsed anchors => UNTESTABLE (no radius spread) ----
    z_const = np.full(z_arr.size, 0.7)
    blk_const = rho_block(z_const, lvl_arr, inst_arr, n_perm=200, n_boot=200,
                          rng=np.random.default_rng(7))
    v_const = verdict_from_block(blk_const, None)
    assert v_const["verdict"] == "UNTESTABLE", (v_const, blk_const)
    print(f"  [ok] collapsed anchors -> {v_const['verdict']}", flush=True)

    # ---- 6. exp0 radius is rank-equivalent to raw radius (Spearman invariance) ----
    raw = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5])
    exp0 = np.tanh(raw)   # exp0 norm = tanh(raw norm), strictly monotone
    lv6 = np.array([1, 2, 3, 4, 5, 6, 7.])
    assert abs(spearman_rho(raw, lv6) - spearman_rho(exp0, lv6)) < 1e-12
    print("  [ok] exp0 (tanh) radius is Spearman-rank-equivalent to raw radius", flush=True)

    # ---- 7. two-sided perm null detects a NEGATIVE (inward) rho too ----
    z_neg = -z_arr  # invert -> strong negative rho
    blk_neg = rho_block(z_neg, lvl_arr, inst_arr, n_perm=2000, n_boot=2000,
                        rng=np.random.default_rng(8))
    assert blk_neg["rho"] < 0 and blk_neg["p_perm"] < P_STRICT, blk_neg
    assert blk_neg["abs_lower_ci"] > HILL_RHO_BAR, blk_neg
    print(f"  [ok] two-sided null detects inward (neg) rho: rho={blk_neg['rho']:.3f} "
          f"p={blk_neg['p_perm']:.4f}", flush=True)

    print("[selftest] ALL PASSED", flush=True)
    return ok


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _ast_parse_ok() -> bool:
    with open(_THIS_FILE) as f:
        src = f.read()
    try:
        ast.parse(src)
        return True
    except SyntaxError as e:
        print(f"[ast.parse] FAILED: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="relaxed circuit checkpoint (.safetensors). MODE A (GPU).")
    ap.add_argument("--radius-map", type=str, default="raw", choices=["raw", "exp0"],
                    help="raw: ||anchor|| (faithful to factor_masks engine, default). "
                         "exp0: ||exp_0(anchor)|| (kenken-ref convention; rank-identical).")
    ap.add_argument("--shuffle-trained", action="store_true",
                    help="the ckpt was trained on depth-SHUFFLED labels (the trained-flat "
                         "control arm); reads the test split with shuffle_depth=True.")
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--seed", type=int, default=42,
                    help="CircuitLoader seed (default 42, mirrors eval_circuit_depth.py).")
    ap.add_argument("--out", type=str, default=None,
                    help="verdict JSON output path (default beside the ckpt).")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only synthetic + mapping self-test (no GPU, no model).")
    args = ap.parse_args()

    parse_ok = _ast_parse_ok()
    print(f"[ast.parse] astparse_ok={parse_ok}", flush=True)
    if not parse_ok:
        sys.exit(1)

    if args.selftest:
        ok = _selftest()
        sys.exit(0 if ok else 1)

    if not args.ckpt:
        print("ERROR: provide --ckpt CKPT (MODE A, GPU) or --selftest (CPU).", flush=True)
        sys.exit(2)

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                "circuit_rho_verdict.json")
    run_mode_a(args.ckpt, args.n_perm, args.n_boot, args.seed,
               args.radius_map, args.shuffle_trained, out_path)


if __name__ == "__main__":
    main()
