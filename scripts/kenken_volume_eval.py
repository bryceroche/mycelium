#!/usr/bin/env python3
"""kenken_volume_eval.py — generate-and-verify VOLUME (best-of-N) for KenKen.

EVAL-ONLY. Measures how much test-time SYMMETRY augmentation ("darts") lifts the
trained deducer's solve rate via best-of-M, per givens band.

THE DART MECHANISM (symmetry TTA)
=================================
A "dart" = a random CELL permutation sigma over the REAL cells of each puzzle (pad
cells fixed), applied to ALL per-cell tensors + the membership cell axis +
cell_cage_id (exactly what scripts.factor_graph_train.permute_factor_batch does).
This relabels the factor graph to an ISOMORPHIC one. We run the deducer forward on
the permuted puzzle, INVERSE-map the per-cell prediction back to original cell order,
then verify on the ORIGINAL puzzle. Solution-preserving by construction (a graph
relabel), so any sigma is valid. Dart 0 = identity = the argmax baseline.

Per-(instance, dart) deterministic permutations via a seeded RNG (reproducible).
Different sigma -> different RoPE/positions/inlet-scatter -> diverse forward -> diverse
dart. This is the position-dependence we EXPLOIT at test time (we do NOT train for
equivariance).

VERIFIER
========
The curriculum puzzles are UNIQUE-solution (generator gates count_solutions==1), so a
complete assignment is exact-constraint-valid IFF it equals the gold solution on all
real cells. Primary success criterion = gold-match on real cells (a free exact verifier
here). We ALSO implement a real constraint verifier (_kenken_proper_np: rows/cols
all-different 1..N + cage_ok per cage) and cross-check gold-match <=> constraint-valid
on a sample (sanity).

REUSE (no existing file modified)
=================================
  scripts.factor_graph_train : model build/load helpers, permute_factor_batch, the
                               eager forward+argmax->puzzle_acc eval template.
  mycelium.kenken_data       : encode_puzzle / KenKenBatch / KenKenLoader (n_cages_max=41).
  mycelium.kenken            : build_verification_inlet, attach_kenken_params (READ-ONLY).
  mycelium.factor_graph_engine : factor_breathing_forward, make_kenken_factor_batch.
  scripts.build_kenken_data  : cage_ok (exact cage arithmetic checker).
  scripts.amortized_frontier_measure : _solve_rate_curve / _view_success_count_distribution.

PARITY GATE
===========
Identity-dart (dart 0) per-band gold-match rate MUST reproduce the trainer eval numbers
within sampling noise (n>=160): g10~0.059, g20~0.276, g30~0.427, g40~0.652. If it does
NOT match, the load / batch build / forward / inlet is wrong. This is the correctness
contract; it is asserted before any volume claim is reported.

USAGE
=====
  DEV=AMD .venv/bin/python3 scripts/kenken_volume_eval.py --n 160 --m 64
  DEV=AMD .venv/bin/python3 scripts/kenken_volume_eval.py --smoke           # n=64 m=8
  DEV=AMD .venv/bin/python3 scripts/kenken_volume_eval.py --parity-only     # identity only
env knobs: KK_N (per-band count), KK_M (darts), KK_BATCH (forward batch), KK_SEED.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

# repo root on sys.path so `scripts.*` / `mycelium.*` import cleanly.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tinygrad import Tensor, Device, dtypes  # noqa: E402

# ---- reuse: model build/load + the cell-permutation relabeler -----------------
# factor_graph_train's heavy work is under `if __name__ == "__main__"`, so importing
# these module-level helpers/classes is side-effect-free.
from mycelium import Config, BreathingTransformer            # noqa: E402
from mycelium.loader import _load_state, load_breathing      # noqa: E402
from scripts.factor_graph_train import (                     # noqa: E402
    cast_layers_fp32, load_ckpt, permute_factor_batch,
)
from mycelium.factor_graph_engine import (                   # noqa: E402
    FactorGraphSpec, factor_breathing_forward, make_kenken_factor_batch,
)
from mycelium.kenken import attach_kenken_params, build_verification_inlet  # noqa: E402
from mycelium.kenken_data import (                           # noqa: E402
    encode_puzzle, KenKenBatch, load_jsonl, N_MAX, N_CELLS,
)
from scripts.build_kenken_data import cage_ok                # noqa: E402
from scripts.amortized_frontier_measure import (             # noqa: E402
    _solve_rate_curve, _view_success_count_distribution,
)

# ---- pinned constants (the ckpt's training topology) -------------------------
CKPT = ".cache/fg_ckpts/fg_kenken_k16_reg/fg_kenken_k16_reg_final.safetensors"
N_CAGES_MAX = 41   # pinned: ckpt trained with this; g40 max cages = 40 <= 41.
K = 16
BANDS = ["g10", "g20", "g30", "g40"]
BAND_FILES = {b: f"/tmp/kk_test_{b}.jsonl" for b in BANDS}
# Trainer eval anchors (the parity contract).
PARITY_EXPECTED = {"g10": 0.059, "g20": 0.276, "g30": 0.427, "g40": 0.652}
# absolute tolerance on the identity-dart gold-match rate vs the anchor (n>=160 noise).
PARITY_ABS_TOL = 0.06


# =============================================================================
# model build/load — mirrors scripts.factor_graph_train.main()'s kenken path.
# =============================================================================
def build_model():
    cfg = Config()
    print(f"loading Pythia-410M -> breathing transformer (hidden={cfg.hidden} "
          f"n_heads={cfg.n_heads})...", flush=True)
    sd = _load_state()
    model = load_breathing(cfg, sd=sd)
    del sd
    cast_layers_fp32(model)
    hidden, n_heads = cfg.hidden, cfg.n_heads

    # kenken spec + the inlet tables (attach BEFORE the general fg params, as main() does).
    spec = FactorGraphSpec(s_max=49, n_values=7, n_factor_types=3,
                           n_heads=n_heads, k_max=K, has_factor_inlet=True)
    attach_kenken_params(model, hidden=hidden, n_heads=n_heads, k_max=K)

    # The general factor-graph params (fg_*). attach_factor_graph_params is invoked
    # from factor_graph_train.main(); import it from there to keep the exact path.
    from scripts.factor_graph_train import attach_factor_graph_params
    attach_factor_graph_params(model, hidden=hidden, spec=spec)

    Device[Device.DEFAULT].synchronize()
    print(f"resuming from fg ckpt: {CKPT}", flush=True)
    load_ckpt(model, CKPT)
    print("  loaded.", flush=True)
    return model, spec


# =============================================================================
# batch construction — stack a list of puzzle records into a KenKenBatch.
# (mirrors KenKenLoader._stack but for an explicit pick list; no loader churn.)
# =============================================================================
def stack_records(recs):
    encs = [encode_puzzle(r, N_CAGES_MAX) for r in recs]

    def stack_int(key):
        return Tensor(np.stack([e[key] for e in encs]).astype(np.int32),
                      dtype=dtypes.int).contiguous().realize()

    def stack_f(key):
        return Tensor(np.stack([e[key] for e in encs]).astype(np.float32),
                      dtype=dtypes.float).contiguous().realize()

    d = {
        "input_cells": stack_int("input_cells"),
        "gold": stack_int("gold"),
        "cell_valid": stack_f("cell_valid"),
        "cage_mask": stack_f("cage_mask"),
        "cell_cage_id": stack_int("cell_cage_id"),
        "cage_cell_count_per_cell": stack_int("cage_cell_count_per_cell"),
        "value_domain_mask": stack_f("value_domain_mask"),
        "cage_op": stack_int("cage_op"),
        "cage_target": stack_int("cage_target"),
        "cage_size": stack_int("cage_size"),
        "deduction_depth": [e["deduction_depth"] for e in encs],
        "N": [e["N"] for e in encs],
        "n_givens": [e["n_givens"] for e in encs],
        "band": [e["band"] for e in encs],
    }
    return KenKenBatch(d)


def base_factor_batch(kb, spec):
    """KenKenBatch -> FactorGraphBatch carrying the raw cage features so the eager
    inlet build (in run_forward) and permute_factor_batch's cell-axis gathers work.
    Mirrors factor_graph_train._build_kenken_task.to_factor_batch (inlet stays None)."""
    fb = make_kenken_factor_batch(kb, spec)   # factor_inlet None
    fb.kenken_cage_op = kb.cage_op
    fb.kenken_cage_target = kb.cage_target
    fb.kenken_cage_size = kb.cage_size
    fb.kenken_cell_cage_id = kb.cell_cage_id
    return fb


# =============================================================================
# the dart: per-instance cell permutation via permute_factor_batch with frac=1.0.
# permute_factor_batch already (a) permutes only REAL cells per instance, (b) gathers
# every per-cell tensor + membership cell axis + cell_cage_id by the same P, (c) passes
# per-cage op/target/size through unchanged. We recover P (the gather map) from
# cell_cage_id (or any per-cell tensor) by comparing pre/post — but it is cleaner and
# exact to apply our OWN permutation and track P explicitly so we can inverse-map.
# =============================================================================
def make_dart_perm(cell_valid_np, rng):
    """Per-instance gather map P (B, S): new position idx[j] takes old position idx[pi[j]].
    Identity on pad cells. Returns P (int64). This is exactly permute_factor_batch's
    convention (new[..., idx] = old[..., idx[pi]])."""
    Bn, S = cell_valid_np.shape
    P = np.tile(np.arange(S, dtype=np.int64), (Bn, 1))
    for b in range(Bn):
        idx = np.nonzero(cell_valid_np[b] > 0.5)[0]
        if idx.size <= 1:
            continue
        pi = rng.permutation(idx.size)
        P[b, idx] = idx[pi]
    return P


def apply_perm(fb, P):
    """Apply gather map P (B,S) to a base FactorGraphBatch -> a NEW permuted batch.
    Reuses the SAME relabel semantics as permute_factor_batch, but with an EXPLICIT,
    tracked P (so we can invert it). Gathers per-cell tensors + membership cell axis +
    cell_cage_id; passes per-cage features through unchanged."""
    ic = fb.input_cells.realize().numpy()
    cv = fb.cell_valid.realize().numpy()
    vdm = fb.value_domain_mask.realize().numpy()
    gold = fb.gold.realize().numpy()
    mem = fb.membership.realize().numpy()           # (B, L, S)
    kcid = fb.kenken_cell_cage_id.realize().numpy()  # (B, 49)

    new_ic = np.take_along_axis(ic, P, axis=1)
    new_cv = np.take_along_axis(cv, P, axis=1)
    new_gold = np.take_along_axis(gold, P, axis=1)
    new_vdm = np.take_along_axis(vdm, P[:, :, None], axis=1)
    new_mem = np.take_along_axis(mem, P[:, None, :], axis=2)
    new_kcid = np.take_along_axis(kcid, P, axis=1)

    class _Shim:
        pass
    out = _Shim()
    out.input_cells = Tensor(new_ic.astype(np.int32), dtype=dtypes.int).contiguous().realize()
    out.cell_valid = Tensor(new_cv.astype(np.float32), dtype=dtypes.float).contiguous().realize()
    out.value_domain_mask = Tensor(new_vdm.astype(np.float32), dtype=dtypes.float).contiguous().realize()
    out.gold = Tensor(new_gold.astype(np.int32), dtype=dtypes.int).contiguous().realize()
    out.membership = Tensor(new_mem.astype(np.float32), dtype=dtypes.float).contiguous().realize()
    out.latent_type = fb.latent_type
    out.factor_inlet = None
    # per-cage features are NOT on the cell axis -> unchanged by a cell relabel.
    out.kenken_cage_op = fb.kenken_cage_op
    out.kenken_cage_target = fb.kenken_cage_target
    out.kenken_cage_size = fb.kenken_cage_size
    out.kenken_cell_cage_id = Tensor(new_kcid.astype(np.int32), dtype=dtypes.int).contiguous().realize()
    return out


# =============================================================================
# forward — eager (eval). Build the verification inlet EAGERLY via the oracle.
# =============================================================================
def run_forward(model, fb, spec):
    """Eager forward -> (B, S) predicted value per cell (argmax+1), as numpy int32.
    Builds the verification inlet eagerly from the carried raw cage features (mirrors
    factor_graph_train.evaluate's kenken branch)."""
    Tensor.training = False
    fb.factor_inlet = build_verification_inlet(
        model, fb.kenken_cage_op, fb.kenken_cage_target,
        fb.kenken_cage_size, fb.kenken_cell_cage_id).realize()
    logits_history, _ = factor_breathing_forward(model, fb, spec, K=K)
    final_logits = logits_history[-1]                       # (B, S, N)
    pred = (final_logits.argmax(axis=-1) + 1).realize().numpy().astype(np.int32)
    return pred


# =============================================================================
# verifiers
# =============================================================================
def _flat_to_rc(f):
    return f // N_MAX, f % N_MAX


def _kenken_proper_np(pred_row, rec):
    """Real constraint verifier on ONE puzzle (original cell order).
    pred_row : (49,) predicted value per flat cell (argmax+1).
    rec      : the original puzzle record (N, cages, clues).
    Returns True iff rows all-different 1..N, cols all-different 1..N, and every cage
    satisfies cage_ok. Reads only REAL cells (the N x N top-left block per encode_puzzle's
    _rc_to_flat = r*N_MAX + c layout)."""
    N = int(rec["N"])
    grid = np.zeros((N, N), dtype=np.int64)
    for r in range(N):
        for c in range(N):
            grid[r, c] = int(pred_row[r * N_MAX + c])
    # value range 1..N
    if grid.min() < 1 or grid.max() > N:
        return False
    # rows / cols all-different
    for r in range(N):
        if len(set(grid[r, :].tolist())) != N:
            return False
    for c in range(N):
        if len(set(grid[:, c].tolist())) != N:
            return False
    # cages
    for cage, (op, tgt) in zip(rec["cages"], rec["clues"]):
        vals = [int(grid[int(rr), int(cc)]) for (rr, cc) in cage]
        if not cage_ok(op, tgt, vals):
            return False
    return True


def _gold_match_np(pred_row, gold_row, cell_valid_row):
    """gold-match success on real cells (the free exact verifier — unique-solution)."""
    valid = cell_valid_row > 0.5
    if int(valid.sum()) == 0:
        return False
    return bool(np.all(pred_row[valid] == gold_row[valid]))


# =============================================================================
# per-band measurement
# =============================================================================
def measure_band(model, spec, band, recs, M, batch_size, seed, crosscheck_budget):
    """Returns (mech_flags, p_argmax, crosscheck_disagreements).

    mech_flags : list (one per instance) of (M,) bool arrays — dart j solved instance i.
                 dart 0 = identity (argmax). darts 1..M-1 = symmetry permutations.
    p_argmax   : identity-dart gold-match rate (== the puzzle rate; the parity number).
    """
    n_inst = len(recs)
    # per-(instance, dart) deterministic RNG: seeded by (band, global seed).
    band_seed = (seed * 1_000_003) ^ (hash(band) & 0xFFFFFFFF)

    # mech_flags[i] = (M,) bool. Fill dart by dart (each dart = one full pass over insts).
    flags = np.zeros((n_inst, M), dtype=bool)
    # cross-check accounting (gold-match vs constraint-valid).
    xcheck_done = 0
    xcheck_disagree = 0
    xcheck_examples = []

    fwd_count = 0
    t0 = time.time()
    for dart in range(M):
        # dart 0 = identity. dart>=1 = a fresh per-instance permutation; the RNG is
        # advanced deterministically per (dart, batch) so results are reproducible.
        for start in range(0, n_inst, batch_size):
            picks = recs[start:start + batch_size]
            bn = len(picks)
            kb = stack_records(picks)
            fb = base_factor_batch(kb, spec)
            cv_np = fb.cell_valid.realize().numpy()        # (bn, S)
            gold_np = fb.gold.realize().numpy().astype(np.int32)

            if dart == 0:
                # identity dart: no permutation. P is identity -> pred is in orig order.
                pred = run_forward(model, fb, spec)        # (bn, S)
                pred_orig = pred
            else:
                rng = np.random.RandomState(
                    (band_seed + dart * 7919 + start * 31) & 0x7FFFFFFF)
                P = make_dart_perm(cv_np, rng)             # (bn, S) gather map
                fb_perm = apply_perm(fb, P)
                pred_perm = run_forward(model, fb_perm, spec)   # (bn, S) permuted order
                # INVERSE-map: pred_perm is in permuted order; new pos j holds the
                # value for ORIGINAL cell P[b, j]. So pred_orig[b, P[b,j]] = pred_perm[b,j].
                pred_orig = np.empty_like(pred_perm)
                for b in range(bn):
                    pred_orig[b, P[b]] = pred_perm[b]
            fwd_count += 1

            for b in range(bn):
                i = start + b
                ok_gold = _gold_match_np(pred_orig[b], gold_np[b], cv_np[b])
                flags[i, dart] = ok_gold
                # sanity cross-check: gold-match <=> real-constraint-valid.
                if xcheck_done < crosscheck_budget:
                    ok_proper = _kenken_proper_np(pred_orig[b], picks[b])
                    xcheck_done += 1
                    if ok_gold != ok_proper:
                        xcheck_disagree += 1
                        if len(xcheck_examples) < 5:
                            xcheck_examples.append(
                                (band, dart, i, ok_gold, ok_proper))
        print(f"    [{band}] dart {dart + 1}/{M} done "
              f"({time.time() - t0:.0f}s, {fwd_count} fwd)", flush=True)

    p_argmax = float(flags[:, 0].mean())
    per_inst = [flags[i] for i in range(n_inst)]
    return per_inst, p_argmax, (xcheck_done, xcheck_disagree, xcheck_examples)


# =============================================================================
# reporting
# =============================================================================
def print_band_report(band, per_inst, p_argmax, M, m_grid, out_lines):
    curve = _solve_rate_curve(per_inst, m_grid, coherent=True)
    vsc = _view_success_count_distribution(per_inst, M)

    def emit(s):
        print(s, flush=True)
        out_lines.append(s)

    emit("")
    emit(f"=== BAND {band}  (n={len(per_inst)}, M={M}) ===")
    emit(f"  p_argmax (identity dart, = puzzle rate) : {p_argmax:.4f}")
    emit(f"  p1_div (mean valid-rate over perturbed darts) : {curve['p1_div']:.4f}")
    emit("  best-of-M solve-rate curve (empirical vs independent-ideal):")
    emit(f"    {'M':>5} | {'empirical':>10} | {'indep-ideal':>11} | "
         f"{'div-ideal':>10}")
    for Mv in m_grid:
        emit(f"    {Mv:>5} | {curve['solve_emp'][Mv]:>10.4f} | "
             f"{curve['solve_ideal'][Mv]:>11.4f} | {curve['solve_ideal_div'][Mv]:>10.4f}")
    emit(f"  M_eff (empirical crosses 0.90)   : {curve['m_eff_emp']}")
    emit(f"  M_eff (indep-ideal crosses 0.90) : {curve['m_eff_ideal']}  "
         f"(ratio emp/ideal={curve['m_eff_ratio']})")
    emit("  fragility — instances by #darts that solved (of M):")
    for (label, c) in vsc["buckets"]:
        frac = c / max(vsc["n_inst"], 1)
        emit(f"    {label:>12} : {c:>5}  ({frac:6.3f})")
    emit(f"  mean #solving darts / inst : {vsc['mean_success']:.3f}   "
         f"median : {vsc['median_success']:.1f}")
    best = curve['solve_emp'][max(m_grid)]
    emit(f"  >>> best-of-{max(m_grid)} = {best:.4f}  "
         f"(lift over argmax = {best - p_argmax:+.4f})")
    return curve, vsc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=int(os.environ.get("KK_N", "160")),
                    help="puzzles per band")
    ap.add_argument("--m", type=int, default=int(os.environ.get("KK_M", "64")),
                    help="darts (best-of-M); dart 0 = identity")
    ap.add_argument("--batch", type=int, default=int(os.environ.get("KK_BATCH", "8")),
                    help="forward batch size")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("KK_SEED", "0")))
    ap.add_argument("--bands", type=str, default=",".join(BANDS))
    ap.add_argument("--smoke", action="store_true", help="n=64, m=8 (crash shakeout)")
    ap.add_argument("--parity-only", action="store_true",
                    help="identity dart only (M=1) — just the parity gate")
    ap.add_argument("--out", type=str, default="/tmp/kenken_volume_result.txt")
    ap.add_argument("--crosscheck", type=int, default=200,
                    help="#per-band gold-match<=>constraint cross-checks (sanity)")
    args = ap.parse_args()

    if args.smoke:
        args.n, args.m = 64, 8
    if args.parity_only:
        args.m = 1

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    m_grid = [m for m in (1, 2, 4, 8, 16, 32, 64) if m <= args.m]
    if args.m not in m_grid:
        m_grid.append(args.m)
    m_grid = sorted(set(m_grid))

    print(f"=== kenken_volume_eval ===", flush=True)
    print(f"  ckpt={CKPT}", flush=True)
    print(f"  n/band={args.n}  M={args.m}  batch={args.batch}  seed={args.seed}  "
          f"bands={bands}  m_grid={m_grid}", flush=True)

    model, spec = build_model()

    out_lines = []
    out_lines.append(f"kenken_volume_eval — ckpt={CKPT}")
    out_lines.append(f"n/band={args.n} M={args.m} batch={args.batch} seed={args.seed} "
                     f"bands={bands}")

    parity_results = {}
    band_curves = {}
    xcheck_total = (0, 0)
    xcheck_all_examples = []

    for band in bands:
        recs = load_jsonl(BAND_FILES[band])
        recs = recs[:args.n]
        print(f"\n[{band}] {len(recs)} puzzles "
              f"(N={recs[0]['N']}, max_cages={max(len(r['cages']) for r in recs)})",
              flush=True)
        per_inst, p_argmax, (xd, xdg, xex) = measure_band(
            model, spec, band, recs, args.m, args.batch, args.seed, args.crosscheck)
        parity_results[band] = p_argmax
        xcheck_total = (xcheck_total[0] + xd, xcheck_total[1] + xdg)
        xcheck_all_examples += xex
        if args.m > 1:
            curve, vsc = print_band_report(band, per_inst, p_argmax, args.m,
                                           m_grid, out_lines)
            band_curves[band] = (curve, vsc)

    # ---- cross-check report (gold-match <=> constraint-valid) ----
    print("\n=== CROSS-CHECK (gold-match <=> real-constraint-valid) ===", flush=True)
    print(f"  checked={xcheck_total[0]}  disagreements={xcheck_total[1]}", flush=True)
    out_lines.append("")
    out_lines.append(f"cross-check: checked={xcheck_total[0]} "
                     f"disagreements={xcheck_total[1]}")
    if xcheck_total[1] > 0:
        print("  !!! DISAGREEMENTS FOUND (band, dart, inst, gold_ok, proper_ok):",
              flush=True)
        for ex in xcheck_all_examples:
            print(f"    {ex}", flush=True)
            out_lines.append(f"  disagree: {ex}")

    # ---- PARITY GATE ----
    print("\n=== PARITY GATE (identity dart vs trainer eval anchors) ===", flush=True)
    out_lines.append("")
    out_lines.append("PARITY GATE (identity dart vs trainer eval anchors):")
    all_pass = True
    for band in bands:
        exp = PARITY_EXPECTED.get(band)
        got = parity_results[band]
        if exp is None:
            line = f"  {band}: got={got:.4f}  (no anchor)"
        else:
            ok = abs(got - exp) <= PARITY_ABS_TOL
            all_pass = all_pass and ok
            line = (f"  {band}: got={got:.4f}  expected~{exp:.3f}  "
                    f"|diff|={abs(got - exp):.4f}  tol={PARITY_ABS_TOL}  "
                    f"{'PASS' if ok else 'FAIL'}")
        print(line, flush=True)
        out_lines.append(line)
    verdict = "PARITY GATE: PASS" if all_pass else "PARITY GATE: FAIL"
    print(f"\n{verdict}", flush=True)
    out_lines.append(verdict)

    # ---- summary table ----
    if args.m > 1:
        print("\n=== SUMMARY (per band) ===", flush=True)
        hdr = (f"  {'band':>5} | {'p_argmax':>9} | {'best-of-' + str(args.m):>12} | "
               f"{'lift':>8} | {'M_eff':>6} | {'mean_darts':>10}")
        print(hdr, flush=True)
        out_lines.append("")
        out_lines.append("SUMMARY (per band):")
        out_lines.append(hdr)
        for band in bands:
            if band not in band_curves:
                continue
            curve, vsc = band_curves[band]
            best = curve['solve_emp'][max(m_grid)]
            p0 = parity_results[band]
            row = (f"  {band:>5} | {p0:>9.4f} | {best:>12.4f} | "
                   f"{best - p0:>+8.4f} | {str(curve['m_eff_emp']):>6} | "
                   f"{vsc['mean_success']:>10.3f}")
            print(row, flush=True)
            out_lines.append(row)

    with open(args.out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nsaved -> {args.out}", flush=True)

    if not all_pass:
        print("\nPARITY GATE FAILED — do not trust the volume numbers above.",
              flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
