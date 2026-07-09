#!/usr/bin/env python3
"""diag_kenken_reach_vs_capacity.py — is the KenKen ceiling CAPACITY or REACH?

QUESTION (gates a multigrid build)
==================================
The deducer (4-layer transformer x K=16 breaths = ~4 attention hops x 16 ≈ 64
propagation LEVELS) hits a ceiling on KenKen (~0.80 generalizing base). Two
competing explanations:

  REACH-LIMITED : the deducer can only propagate ~K hops, so deductions needing a
                  chain longer than its reach are STRUCTURALLY unreachable. Error
                  should concentrate on DEEP-CHAIN cells. => build multigrid.
  CAPACITY-LIMITED : error is diffuse / depth-independent once puzzle HARDNESS is
                  held fixed; reach budget is ample. => multigrid buys speed, not
                  the wall. defer.

THE CRUX = the reach-vs-hardness CONFOUND. deduction_depth correlates with
puzzle hardness (deeper chains => fewer givens => harder puzzle overall). So a raw
"error rises with depth" curve is AMBIGUOUS: it could be hardness (capacity) OR
reach. We therefore bucket error by per-cell depth BOTH raw AND CONTROLLED (within
a FIXED givens-density band and FIXED N). If error-vs-depth FLATTENS once hardness
is held constant => it was hardness (CAPACITY), not reach.

WHAT THIS PROBE COMPUTES
========================
1. Per-cell argmax-vs-gold on REAL NON-GIVEN cells (final breath), via the exact
   forward path from scripts.kenken_volume_eval (build_model / stack_records /
   base_factor_batch / run_forward). PARITY-GATED against the trainer anchors.
2. Per-cell DEDUCTION DEPTH = the propagation round each cell first becomes a
   SINGLETON, from an instrumented copy of build_kenken_data.propagate (same
   semantics; we additionally record the per-cell settle round). The puzzle-level
   deduction_depth field == rounds-to-fixpoint of the same routine.
3. Per-cell GRAPH-DISTANCE-FROM-GIVENS = BFS hops on the rook+cage graph
   (row/col rook edges + cage clique edges) from the set of given cells. A simpler,
   model-free reach proxy.
4. Reach-budget sanity: max per-cell deduction depth observed vs ~64.

OUTPUT: a verdict (CAPACITY vs REACH) + the controlled depth-vs-error curves.

USAGE
=====
  DEV=AMD .venv/bin/python3 scripts/diag_kenken_reach_vs_capacity.py
  DEV=AMD .venv/bin/python3 scripts/diag_kenken_reach_vs_capacity.py --n 1500
env: KK_BATCH (forward batch, default 8), KKR_N (puzzles, default all 8004).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# REUSE the validated forward path verbatim (no existing file modified).
from scripts.kenken_volume_eval import (   # noqa: E402
    build_model, stack_records, base_factor_batch, run_forward, PARITY_EXPECTED,
)
from mycelium.kenken_data import load_jsonl, N_MAX  # noqa: E402

# ---- match the propagate routine's max_rounds so depth labels agree ----------
_MAX_ROUNDS = 64


# =============================================================================
# Per-cell DEDUCTION DEPTH: instrumented copy of build_kenken_data.propagate.
# Identical propagation semantics (naked-single row/col elimination + hidden
# single + cage-support pruning) but we RECORD, per cell, the round its domain
# first collapses to a singleton. That round = the cell's deduction-chain depth.
#   round 0  = given (singleton from the start, before any propagation)
#   round r  = became determined during propagation pass r (r >= 1)
# Returns: settle_round dict {(r,c): int}, total_rounds, solved.
# =============================================================================
def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _cage_ok(typ, tgt, asg):
    if typ == "given":
        return asg[0] == tgt
    if typ == "add":
        return sum(asg) == tgt
    if typ == "mul":
        return _prod(asg) == tgt
    if typ == "sub":
        return abs(asg[0] - asg[1]) == tgt
    if typ == "div":
        a, b = asg
        return (b and a == b * tgt) or (a and b == a * tgt)
    return False


def propagate_per_cell(n, cages, clues, max_rounds=_MAX_ROUNDS):
    """Constraint propagation that ALSO records each cell's settle round.

    Mirrors build_kenken_data.propagate exactly (same three pruning rules, same
    iteration order) and additionally stamps settle_round[cell] the FIRST round
    its domain becomes a singleton. Givens get settle_round 0.
    """
    import itertools
    # JSON-loaded cages are lists-of-lists; normalize cells to hashable tuples so
    # they key the domain dict (matches the generator, which used (r,c) tuples).
    cages = [[(int(r), int(c)) for (r, c) in cage] for cage in cages]
    dom = {(r, c): set(range(1, n + 1)) for r in range(n) for c in range(n)}
    settle_round = {}
    for cage, (typ, tgt) in zip(cages, clues):
        if typ == "given":
            dom[cage[0]] = {tgt}
    # stamp givens (singletons before any propagation) as round 0
    for cell, d in dom.items():
        if len(d) == 1:
            settle_round[cell] = 0

    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        changed = False
        # naked single: a singleton cell removes its value from row/col peers
        for (r, c), d in list(dom.items()):
            if len(d) == 1:
                v = next(iter(d))
                for cc in range(n):
                    if cc != c and v in dom[(r, cc)]:
                        dom[(r, cc)].discard(v); changed = True
                for rr in range(n):
                    if rr != r and v in dom[(rr, c)]:
                        dom[(rr, c)].discard(v); changed = True
        # hidden single: a value with a unique home in a line
        for line in ([[(r, c) for c in range(n)] for r in range(n)] +
                     [[(r, c) for r in range(n)] for c in range(n)]):
            for v in range(1, n + 1):
                homes = [cell for cell in line if v in dom[cell]]
                if len(homes) == 1 and len(dom[homes[0]]) > 1:
                    dom[homes[0]] = {v}; changed = True
        # cage-support pruning
        for cage, (typ, tgt) in zip(cages, clues):
            doms = [sorted(dom[cell]) for cell in cage]
            if _prod([len(d) for d in doms]) > 20000:
                continue
            support = [set() for _ in cage]
            for combo in itertools.product(*doms):
                ok = True
                for i in range(len(cage)):
                    for j in range(i + 1, len(cage)):
                        if combo[i] == combo[j] and (
                                cage[i][0] == cage[j][0] or cage[i][1] == cage[j][1]):
                            ok = False; break
                    if not ok:
                        break
                if ok and _cage_ok(typ, tgt, combo):
                    for i, v in enumerate(combo):
                        support[i].add(v)
            for i, cell in enumerate(cage):
                if support[i] and (dom[cell] - support[i]):
                    dom[cell] &= support[i]; changed = True
        # stamp any cell that became a singleton THIS round
        for cell, d in dom.items():
            if len(d) == 1 and cell not in settle_round:
                settle_round[cell] = rounds
        if not changed:
            break
    solved = all(len(d) == 1 for d in dom.values())
    return settle_round, rounds, solved


# =============================================================================
# Per-cell GRAPH-DISTANCE-FROM-GIVENS: BFS on the rook+cage graph.
# Edges: same-row, same-col (rook), and same-cage (cage clique). Sources = the
# given cells. Distance = #hops to the nearest given. A model-free reach proxy.
# =============================================================================
def graph_dist_from_givens(n, cages, clues):
    cells = [(r, c) for r in range(n) for c in range(n)]
    # adjacency
    adj = defaultdict(set)
    for (r, c) in cells:
        for cc in range(n):
            if cc != c:
                adj[(r, c)].add((r, cc))
        for rr in range(n):
            if rr != r:
                adj[(r, c)].add((rr, c))
    for cage in cages:
        cg = [(int(r), int(c)) for (r, c) in cage]
        for a in cg:
            for b in cg:
                if a != b:
                    adj[a].add(b)
    givens = [cage[0] for cage, (typ, _t) in zip(cages, clues) if typ == "given"]
    givens = [(int(r), int(c)) for (r, c) in givens]
    dist = {cell: -1 for cell in cells}
    dq = deque()
    for g in givens:
        dist[g] = 0; dq.append(g)
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist  # -1 means unreachable from any given (no givens at all => all -1)


# =============================================================================
# Bucketed error reporting (a small helper that prints depth -> error, count).
# =============================================================================
def bucket_curve(depths, errors):
    """depths, errors : 1D np arrays (per cell). Returns sorted list of
    (depth, n_cells, n_errors, error_rate)."""
    out = []
    for d in sorted(set(depths.tolist())):
        m = depths == d
        n = int(m.sum())
        e = int(errors[m].sum())
        out.append((int(d), n, e, e / max(n, 1)))
    return out


def fmt_curve(curve, label):
    lines = [f"  {label}:"]
    lines.append(f"    {'depth':>6} | {'n_cells':>8} | {'n_err':>7} | {'err_rate':>8}")
    for (d, n, e, rate) in curve:
        lines.append(f"    {d:>6} | {n:>8} | {e:>7} | {rate:>8.4f}")
    return "\n".join(lines)


def slope_of_curve(curve, min_count=200):
    """Weighted linear-regression slope of error_rate vs depth (cells as weights),
    over buckets with >= min_count cells (drop tiny noisy buckets). Returns
    (slope, n_buckets_used, depth_lo, depth_hi)."""
    pts = [(d, rate, n) for (d, n, e, rate) in curve if n >= min_count]
    if len(pts) < 2:
        return None, len(pts), None, None
    d = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)
    w = np.array([p[2] for p in pts], dtype=np.float64)
    # weighted least squares slope
    dm = np.average(d, weights=w)
    ym = np.average(y, weights=w)
    cov = np.average((d - dm) * (y - ym), weights=w)
    var = np.average((d - dm) ** 2, weights=w)
    slope = cov / var if var > 0 else 0.0
    return float(slope), len(pts), int(d.min()), int(d.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=int(os.environ.get("KKR_N", "8004")),
                    help="number of test puzzles to run (default all)")
    ap.add_argument("--batch", type=int, default=int(os.environ.get("KK_BATCH", "8")))
    ap.add_argument("--test", type=str,
                    default=".cache/kenken_test_curriculum.jsonl")
    ap.add_argument("--out", type=str, default="/tmp/kenken_reach_vs_capacity.txt")
    args = ap.parse_args()

    print("=== diag_kenken_reach_vs_capacity ===", flush=True)
    recs = load_jsonl(args.test)
    recs = recs[:args.n]
    print(f"  loaded {len(recs)} test puzzles from {args.test}", flush=True)

    model, spec = build_model()

    # ---- run forward over all puzzles, collect per-cell records ----
    # Per real non-given cell we collect: correct(0/1), settle_round (deduction
    # depth), graph_dist, band, N, deduction_depth(puzzle), n_givens.
    records = []   # list of dicts (one per real non-given cell)
    # also: per-puzzle identity-dart gold-match for the parity gate, per band.
    band_puzzle_correct = defaultdict(list)
    max_settle = 0

    t0 = time.time()
    nproc = 0
    for start in range(0, len(recs), args.batch):
        picks = recs[start:start + args.batch]
        kb = stack_records(picks)
        fb = base_factor_batch(kb, spec)
        cv_np = fb.cell_valid.realize().numpy()             # (bn, 49)
        gold_np = fb.gold.realize().numpy().astype(np.int32)  # (bn, 49)
        ic_np = fb.input_cells.realize().numpy().astype(np.int32)  # (bn, 49) givens>0
        pred = run_forward(model, fb, spec)                 # (bn, 49) argmax+1

        for b, rec in enumerate(picks):
            n = int(rec["N"])
            band = rec["band"]
            pdepth = int(rec["deduction_depth"])
            ngiv = int(rec["n_givens"])
            # per-cell labels via instrumented propagate + graph BFS
            settle = propagate_per_cell(n, rec["cages"], rec["clues"])[0]
            gdist = graph_dist_from_givens(n, rec["cages"], rec["clues"])

            # puzzle-level gold-match on real cells (parity gate)
            valid = cv_np[b] > 0.5
            puzzle_ok = bool(np.all(pred[b][valid] == gold_np[b][valid])) if valid.sum() else False
            band_puzzle_correct[band].append(puzzle_ok)

            for r in range(n):
                for c in range(n):
                    f = r * N_MAX + c
                    # REAL non-given cell only (cell_valid==1 AND not a given input)
                    if cv_np[b, f] <= 0.5:
                        continue
                    if ic_np[b, f] > 0:    # given cell -> skip (trivially "known")
                        continue
                    sr = settle.get((r, c), _MAX_ROUNDS)   # uncollapsed -> cap
                    gd = gdist.get((r, c), -1)
                    max_settle = max(max_settle, sr if sr < _MAX_ROUNDS else 0)
                    correct = int(pred[b, f] == gold_np[b, f])
                    records.append({
                        "err": 1 - correct,
                        "settle": sr,
                        "gdist": gd,
                        "band": band,
                        "N": n,
                        "pdepth": pdepth,
                        "ngiv": ngiv,
                    })
        nproc += len(picks)
        if (start // args.batch) % 50 == 0:
            print(f"  ... {nproc}/{len(recs)} puzzles "
                  f"({time.time()-t0:.0f}s, {len(records)} cells)", flush=True)

    print(f"  forward done: {nproc} puzzles, {len(records)} real non-given cells, "
          f"{time.time()-t0:.0f}s", flush=True)

    # ---- to numpy arrays ----
    err = np.array([r["err"] for r in records], dtype=np.int64)
    settle = np.array([r["settle"] for r in records], dtype=np.int64)
    gdist = np.array([r["gdist"] for r in records], dtype=np.int64)
    band = np.array([r["band"] for r in records])
    Narr = np.array([r["N"] for r in records], dtype=np.int64)
    pdepth = np.array([r["pdepth"] for r in records], dtype=np.int64)
    ngiv = np.array([r["ngiv"] for r in records], dtype=np.int64)
    # CAPPED = cells local propagation never resolved to a singleton. These are NOT
    # deep-chain cells; they're cells where pure local propagation is insufficient
    # (the puzzle's uniqueness needs backtracking/global reasoning). Excluding them
    # from depth slopes is REQUIRED — otherwise the artificial x=64 cap flattens the
    # regression and mislabels them as "deepest". Reported as their own category.
    uncapped = settle < _MAX_ROUNDS

    overall_err = float(err.mean())
    overall_cell_acc = 1.0 - overall_err
    print(f"\n  overall per-cell acc (real non-given) = {overall_cell_acc:.4f} "
          f"(err {overall_err:.4f}, n={len(err)})", flush=True)

    out = []

    def emit(s):
        print(s, flush=True)
        out.append(s)

    emit("=== diag_kenken_reach_vs_capacity ===")
    emit(f"ckpt: fg_kenken_k16_reg_final  test: {args.test}  n_puzzles={nproc}")
    emit(f"overall per-cell acc (real non-given) = {overall_cell_acc:.4f}  "
         f"(n_cells={len(err)})")

    # ---- PARITY GATE: per-band puzzle gold-match vs trainer anchors ----
    emit("")
    emit("=== PARITY GATE (per-band puzzle gold-match vs trainer anchors) ===")
    parity_ok = True
    for bname in ["g10", "g20", "g30", "g40"]:
        if bname not in band_puzzle_correct:
            continue
        got = float(np.mean(band_puzzle_correct[bname]))
        exp = PARITY_EXPECTED.get(bname)
        if exp is None:
            emit(f"  {bname}: got={got:.4f} (no anchor)")
        else:
            ok = abs(got - exp) <= 0.06
            parity_ok = parity_ok and ok
            emit(f"  {bname}: got={got:.4f}  expected~{exp:.3f}  "
                 f"|diff|={abs(got-exp):.4f}  {'PASS' if ok else 'FAIL'}")
    emit(f"  PARITY GATE: {'PASS' if parity_ok else 'FAIL'}")

    # ---- REACH-BUDGET sanity ----
    emit("")
    emit("=== REACH-BUDGET SANITY ===")
    real_settle = settle[settle < _MAX_ROUNDS]
    n_uncollapsed = int((settle >= _MAX_ROUNDS).sum())
    budget = 4 * 16  # ~4 levels/breath x K=16
    emit(f"  reach budget (~4 levels/breath x K=16)           = {budget}")
    emit(f"  rook+cage graph diameter                          ~ 2 (rook structure)")
    emit(f"  max per-cell DEDUCTION DEPTH (settle round) obs   = {int(real_settle.max()) if real_settle.size else 'NA'}")
    emit(f"  99th pctile per-cell deduction depth              = {int(np.percentile(real_settle,99)) if real_settle.size else 'NA'}")
    emit(f"  max per-cell GRAPH-DIST from givens               = {int(gdist[gdist>=0].max()) if (gdist>=0).any() else 'NA'}")
    emit(f"  cells deeper than budget                          = "
         f"{int((real_settle > budget).sum())} / {real_settle.size}")
    emit(f"  cells never collapsed by propagate (capped)       = {n_uncollapsed}")
    reach_ample = (real_settle.size > 0 and int(real_settle.max()) < budget)
    emit(f"  => max depth << budget ? {reach_ample}  "
         f"(if True, reach cannot be the limit BY CONSTRUCTION)")

    # ---- CAPPED cells (local propagation insufficient) reported separately ----
    emit("")
    emit("=== CAPPED CELLS (local propagation never resolved -> NOT depth, a "
         "different failure mode) ===")
    if n_uncollapsed > 0:
        cap_err = float(err[~uncapped].mean())
        emit(f"  capped cells = {n_uncollapsed} ({100*n_uncollapsed/len(err):.1f}% of "
             f"real non-given cells)")
        emit(f"  err on capped cells   = {cap_err:.4f}")
        emit(f"  err on uncapped cells = {float(err[uncapped].mean()):.4f}")
        emit("  NOTE: capped cells are EXCLUDED from all depth slopes below (an "
             "x=64 cap would falsely flatten the regression).")

    # ---- RAW depth-vs-error (the ambiguous curve) — UNCAPPED cells only ----
    emit("")
    emit("=== RAW: error vs per-cell DEDUCTION DEPTH (settle round, UNCAPPED) ===")
    raw_curve = bucket_curve(settle[uncapped], err[uncapped])
    emit(fmt_curve(raw_curve, "raw error vs settle-round"))
    sl, nb, lo, hi = slope_of_curve(raw_curve)
    emit(f"    weighted slope (buckets>=200 cells, depth {lo}..{hi}): "
         f"{sl:.4f} err/depth-level over {nb} buckets")

    emit("")
    emit("=== RAW: error vs per-cell GRAPH-DIST from givens ===")
    gmask = gdist >= 0
    raw_g = bucket_curve(gdist[gmask], err[gmask])
    emit(fmt_curve(raw_g, "raw error vs graph-dist"))
    slg, nbg, log_, hig = slope_of_curve(raw_g)
    emit(f"    weighted slope (buckets>=200 cells, dist {log_}..{hig}): "
         f"{slg:.4f} err/hop over {nbg} buckets")

    # ---- error vs PUZZLE deduction_depth + error vs N (the hardness axes) ----
    emit("")
    emit("=== error vs PUZZLE deduction_depth (hardness proxy) ===")
    emit(fmt_curve(bucket_curve(pdepth, err), "error vs puzzle deduction_depth"))
    emit("")
    emit("=== error vs N (board size — error tracks chain length or cell count?) ===")
    emit(fmt_curve(bucket_curve(Narr, err), "error vs N"))
    emit("  (depth distribution per N below — does error track depth or raw N?)")
    for nv in sorted(set(Narr.tolist())):
        nm = Narr == nv
        sub = settle[nm]
        subr = sub[sub < _MAX_ROUNDS]
        emit(f"    N={nv}: n_cells={int(nm.sum())}  err={err[nm].mean():.4f}  "
             f"mean_depth={subr.mean():.2f}  med_depth={int(np.median(subr)) if subr.size else 'NA'}  "
             f"max_depth={int(subr.max()) if subr.size else 'NA'}")

    # ============================================================
    # THE CRUX: depth-vs-error CONTROLLED for hardness (fixed band + fixed N).
    # If the raw slope was driven by hardness, the slope should FLATTEN once we
    # hold band AND N constant (each cell within a (band,N) cell has the same
    # puzzle-hardness regime; only its per-cell chain depth varies).
    # ============================================================
    emit("")
    emit("=== CONTROLLED: error vs per-cell DEPTH, WITHIN fixed (band, N) ===")
    emit("  (the confound control: hardness held => residual depth-slope = REACH signal)")
    emit("  (UNCAPPED cells only — capped cells reported above)")
    controlled_slopes = []
    for bnd in ["g40", "g30", "g20", "g10"]:
        for nv in [5, 6, 7]:
            m = (band == bnd) & (Narr == nv) & uncapped
            if int(m.sum()) < 400:
                continue
            cur = bucket_curve(settle[m], err[m])
            sl_c, nb_c, lo_c, hi_c = slope_of_curve(cur, min_count=100)
            base_err = float(err[m].mean())
            if sl_c is not None:
                controlled_slopes.append((bnd, nv, sl_c, nb_c, lo_c, hi_c,
                                          base_err, int(m.sum())))
            emit("")
            emit(f"  --- band={bnd} N={nv}  (n_cells={int(m.sum())}, "
                 f"mean_err={base_err:.4f}) ---")
            emit(fmt_curve(cur, f"error vs depth | {bnd} N{nv}"))
            if sl_c is not None:
                emit(f"    controlled slope (buckets>=100, depth {lo_c}..{hi_c}): "
                     f"{sl_c:.4f} err/depth-level over {nb_c} buckets")

    # ---- TIGHTER CONTROL: fix N AND exact n_givens (the real hardness driver).
    # band is a coarse density bucket; n_givens within N is the continuous hardness
    # knob. Holding (N, n_givens) fixed removes residual hardness that band leaves.
    emit("")
    emit("=== TIGHTER CONTROL: error vs per-cell DEPTH, WITHIN fixed (N, n_givens) ===")
    emit("  (n_givens within N = the continuous hardness driver band only coarsely bins)")
    tight_slopes = []
    for nv in [5, 6, 7]:
        for gv in sorted(set(ngiv[(Narr == nv)].tolist())):
            m = (Narr == nv) & (ngiv == gv) & uncapped
            if int(m.sum()) < 500:
                continue
            cur = bucket_curve(settle[m], err[m])
            sl_t, nb_t, lo_t, hi_t = slope_of_curve(cur, min_count=120)
            base_err = float(err[m].mean())
            if sl_t is not None and nb_t >= 2:
                tight_slopes.append((nv, gv, sl_t, nb_t, lo_t, hi_t, base_err,
                                     int(m.sum())))
                emit(f"  N={nv} n_givens={gv}: slope={sl_t:+.4f}  "
                     f"(n_cells={int(m.sum())}, base_err={base_err:.4f}, "
                     f"buckets {lo_t}..{hi_t} x{nb_t})  "
                     f"curve={[(d,round(r,3)) for (d,_n,_e,r) in cur if _n>=120]}")

    # ---- single fixed-band (g20 only) collapse across N, a headline control ----
    emit("")
    emit("=== HEADLINE CONTROL: g20 only (single hardness band), depth-vs-error ===")
    m20 = (band == "g20") & uncapped
    cur20 = bucket_curve(settle[m20], err[m20])
    emit(fmt_curve(cur20, "g20 error vs depth"))
    sl20, nb20, lo20, hi20 = slope_of_curve(cur20, min_count=150)
    emit(f"    g20 controlled slope (buckets>=150, depth {lo20}..{hi20}): "
         f"{sl20:.4f} err/depth-level over {nb20} buckets")

    # ---- summary of controlled slopes vs the raw slope ----
    emit("")
    emit("=== CONTROLLED-SLOPE SUMMARY ===")
    emit(f"  RAW slope (all puzzles, depth-vs-error)  = {sl:.4f} err/depth-level")
    if controlled_slopes:
        cs = np.array([x[2] for x in controlled_slopes])
        wts = np.array([x[7] for x in controlled_slopes], dtype=np.float64)
        mean_c = float(np.average(cs, weights=wts))
        emit(f"  CONTROLLED slopes (per band x N cell):")
        for (bnd, nv, sl_c, nb_c, lo_c, hi_c, be, ncell) in controlled_slopes:
            emit(f"    {bnd} N{nv}: slope={sl_c:+.4f}  (n_cells={ncell}, "
                 f"base_err={be:.4f}, buckets {lo_c}..{hi_c})")
        emit(f"  cell-weighted MEAN controlled slope       = {mean_c:.4f} err/depth-level")
        emit(f"  g20-only slope                            = {sl20:.4f} err/depth-level")
        attenuation = (sl - mean_c) / sl if abs(sl) > 1e-9 else 0.0
        emit(f"  slope attenuation raw->controlled         = {attenuation*100:.0f}% "
             f"(how much of the raw depth-slope was HARDNESS)")
    else:
        mean_c = None

    # tighter (N, n_givens) control mean slope — the cleanest residual depth signal
    mean_t = None
    if tight_slopes:
        ts = np.array([x[2] for x in tight_slopes])
        tw = np.array([x[7] for x in tight_slopes], dtype=np.float64)
        mean_t = float(np.average(ts, weights=tw))
        emit(f"  TIGHTER (N, n_givens) mean slope          = {mean_t:.4f} "
             f"err/depth-level  (over {len(tight_slopes)} (N,n_givens) cells)")

    # ============================================================
    # VERDICT
    # ============================================================
    emit("")
    emit("=== VERDICT ===")
    # Heuristic decision rule, stated explicitly:
    #  - reach budget ample iff max settle depth << 64 (it is, by construction)
    #  - REACH-limited iff controlled depth-slope stays strongly positive AND the
    #    deepest cells fail systematically.
    #  - CAPACITY-limited iff controlled slope ~flat (most of raw slope was
    #    hardness) and no cell exceeds the budget.
    flat_thresh = 0.02   # err/depth-level: below this the controlled curve is ~flat
    # decide on the TIGHTEST available control (N, n_givens) if present, else (band,N).
    decision_slope = mean_t if mean_t is not None else mean_c
    if decision_slope is not None:
        if reach_ample and abs(decision_slope) < flat_thresh:
            verdict = "CAPACITY-LIMITED"
        elif decision_slope >= flat_thresh:
            verdict = "REACH-LIMITED"
        else:
            verdict = "CAPACITY-LIMITED (weak/negative controlled slope)"
    else:
        verdict = "INCONCLUSIVE (insufficient controlled buckets)"
    emit(f"  reach budget ample (max_depth << 64)        : {reach_ample}")
    emit(f"  raw depth-slope (uncapped)                  : {sl:.4f}")
    emit(f"  controlled depth-slope (band x N)           : "
         f"{mean_c if mean_c is not None else 'NA'}")
    emit(f"  TIGHTER depth-slope (N x n_givens)          : "
         f"{mean_t if mean_t is not None else 'NA'}")
    emit(f"  decision slope (tightest control)           : {decision_slope}")
    emit(f"  VERDICT: {verdict}")

    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nsaved -> {args.out}", flush=True)

    # machine-readable summary for the caller
    summary = {
        "overall_cell_acc": overall_cell_acc,
        "raw_slope": sl,
        "controlled_mean_slope_bandN": mean_c,
        "tighter_mean_slope_N_ngivens": mean_t,
        "decision_slope": decision_slope,
        "g20_slope": sl20,
        "max_settle_depth": int(real_settle.max()) if real_settle.size else None,
        "max_graph_dist": int(gdist[gdist >= 0].max()) if (gdist >= 0).any() else None,
        "reach_budget": budget,
        "reach_ample": bool(reach_ample),
        "cells_over_budget": int((real_settle > budget).sum()),
        "capped_cells": int(n_uncollapsed),
        "capped_frac": float(n_uncollapsed / max(len(err), 1)),
        "err_capped": float(err[~uncapped].mean()) if n_uncollapsed > 0 else None,
        "err_uncapped": float(err[uncapped].mean()) if uncapped.any() else None,
        "verdict": verdict,
        "raw_curve": raw_curve,
        "g20_curve": cur20,
        "tight_slopes": [list(x) for x in tight_slopes],
        "controlled_slopes": [list(x) for x in controlled_slopes],
        "n_cells": int(len(err)),
        "n_puzzles": int(nproc),
    }
    with open("/tmp/kenken_reach_vs_capacity.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved -> /tmp/kenken_reach_vs_capacity.json", flush=True)


if __name__ == "__main__":
    main()
