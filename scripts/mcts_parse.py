"""scripts/mcts_parse.py -- MCTS OVER PARSES (2026-10-05, zero-GPU, delegate; item 2 of
"WORD GIVEN FOR THE DISCRETE HALVES", ledger 19:24).

QUESTION: the panel picker (2026-10-05 12:48) enumerates ALL <=64 candidates per row
up front, solves every distinct one, then PICKS post-hoc (a judge or a trained panel).
This script asks whether a tree SEARCH over the same per-slot alternative space --
visiting the most-promising (lowest-margin) flips first, stopping the instant a
solved+unique node is found, never visiting a node whose decoded graph the solver has
already certified unsat -- can reach a comparable number of rows while looking at far
fewer candidates (a budget of <=64 SOLVER CALLS per row, not <=64 candidates; most
rows terminate in a handful of calls). Per CLAUDE.md Sec.6 ("neural-guided clean-CSP
search CLOSED") the tree is over how to READ THE TEXT -- which slot the model's own
decode is least sure of -- never over the solver's own moves; the June solver is
called once per node, as an oracle, never searched inside.

======================================================================
REUSE, NOT REIMPLEMENTATION (the task's standing rule: never edit phase1_algebra_head.py,
jit_read.py, loop_val.py or chain_acc.py; everything else is a LIBRARY, imported)
======================================================================
- scripts/combined_oracle.py (CO), imported as a module (never edited): CO.BO and CO.SK
  (the same beam_oracle.py / sinkhorn_claim.py module objects combined_oracle itself
  imports), CO.typeop_decisions / CO.apply_typeop_flips (the TYPE/OP axis, combined_oracle.py
  :265-292), CO.value_variants / CO.apply_value_variant (the VALUES axis, combined_oracle.py
  :299-346), CO.candidate_loglik (unused here -- the MCTS judge is solved+unique only, no
  likelihood tie-break; see "THE JUDGE" below), CO.UNIQ_BUDGET (:208).
- scripts/beam_oracle.py (via CO.BO), imported, never edited: BO.mask_row (:150-161, the
  numeral mask on every non-relation slot's digits), BO.slot_decisions (:164-182, the ARGS
  axis's margin-ranked decision list), BO.apply_flips (:185-195), BO.parse_sig (:198-201,
  the order-free decoded-graph signature used to dedup nodes that reach the same graph via
  different flip orders), BO.build_gv_nvv (:204-210).
- scripts/sinkhorn_claim.py (via CO.SK), imported, never edited: SK.given_slots (:94-103),
  SK.affinity_row (:111-116, reused here ONLY to score a value-variant's "regret" -- see
  value_variant_margins below; CO.value_variants computes the identical affinity matrix
  internally but does not return it, so this one quantity is legitimately recomputed, not
  the branching/collision/Hungarian LOGIC itself, which stays solely inside CO.value_variants).
- mycelium/doors.py's certify_unique (:33-44): the uniqueness door, called exactly as
  combined_oracle.py's _uniqueness_task calls it (same budget constant, same fresh-problem-
  rebuild-and-ban convention).
- phase1_algebra_head._decode_slots / N_DIG: imported, never forward/build_params (the
  task's standing rule; same as every oracle/picker script in this family).
- scripts/admit_annotation.py's solve_walled / _Timeout / _alarm, and
  scripts/alternator_bridge.py's problem_from_algebra3: imported inside the per-call
  functions below (_solve_parse, _check_unique), exactly as combined_oracle.py's
  _solve_task / _uniqueness_task do. THE ONE DIFFERENCE from combined_oracle's copies:
  those are written to run inside a multiprocessing.Pool worker (a batch of thousands of
  independent (row, candidate) solves, parallelized ACROSS candidates); this script's
  search is inherently SEQUENTIAL within one row (each node's evaluation decides which
  node is visited next), so _solve_parse / _check_unique are called in-process, directly,
  not wrapped as pool tasks. Parallelism here is instead ACROSS ROWS (each row's whole
  tree search is one pool task -- see _mcts_row_task / main).

======================================================================
THE TREE
======================================================================
Per row: base_row = BO.mask_row(raw_row, text) (the standard numeral-masked top-1 decode,
byte-identical to chain_acc's CA_MASK=1 and to combined_oracle's own base_row). The
ACTION SET (identical menu to combined_oracle's three axes, same defaults: K_ARGS<=3
lowest-margin arg decisions, M_TYPEOP<=2 lowest-margin ftype/op decisions, <=VAL_MAX-1
non-baseline value variants):
  args(slot)      -- flip that RELATION slot's weaker argument pointer to its runner-up
                      (BO.slot_decisions' own margin; BO.apply_flips applies it).
  typeop(slot,fld) -- flip that slot's ftype (or, for a currently-"rel" slot, its op) to
                      its runner-up (CO.typeop_decisions' own margin; CO.apply_typeop_flips
                      applies it).
  value(idx)       -- replace EVERY given slot's digits with ONE of CO.value_variants'
                      non-baseline reassignments (V1 the Hungarian optimum over colliding
                      givens, V2.. one colliding given's own runner-up numeral) -- this is
                      the one INTERPRETATION CALL this script makes beyond combined_oracle's
                      own stated ones (see combined_oracle.py's module docstring, "Scope
                      note"): a value variant touches possibly several given slots at once,
                      but the task brief's "flip ONE slot to its next-best alternative along
                      ONE axis" is read at the AXIS level for values (adopt this one
                      reassignment) rather than decomposed into N single-slot flips, because
                      that is the granularity combined_oracle's own value axis already
                      commits to (a single value_variants() call list, not a per-given-slot
                      bitmask) -- stated here, not patched.

A NODE is the SET of actions committed so far: (frozenset(args slots flipped),
frozenset((slot,field) typeop keys flipped), value-variant index or 0 for baseline). This
is exactly combined_oracle's own (args-bitmask, typeop-bitmask, value-variant) candidate
space -- the tree's nodes ARE combined_oracle's candidates, reached one flip at a time
instead of enumerated as a cross product. ROOT = ((), (), 0) = the top-1 decode. An EDGE
commits exactly one more action not already active at the parent (an args slot / typeop
key is used at most once per path; the value axis fires at most once per path, baseline
-> some V_i, never V_i -> V_j). DEPTH = number of committed actions (<=3, MP_DEPTH).

PRIOR: every action carries a MARGIN (args/typeop: BO.slot_decisions' / CO.typeop_decisions'
own float margin, the gap the model itself saw between its top choice and the runner-up;
value: value_variant_margins' regret below) -- always >= 0, 0 only at the root. A node's
PRIORITY is the sum of its path's action margins (a plain, deterministic, zero-exploration
prior -- see "MCTS vs best-first" below).

ROLLOUT: a node is evaluated by decoding its full graph (BO.apply_flips -> CO.
apply_typeop_flips -> CO.apply_value_variant -> _decode_slots, the SAME compose order
combined_oracle.py's main() and gen_candidates.build_row_candidates use), computing its
parse signature (BO.parse_sig -- a node reached via two different flip orders that decodes
to the identical graph is evaluated ONCE, from cache, free of budget), then calling the
June solver on the FULL graph (_solve_parse, one solve-budget unit) and, if solved,
certify_unique on it (_check_unique, one more unit) -- this IS "the solver's verdict on
the node's full graph" the task brief specifies; there is no partial/incremental CSP
check inside one node (csp_core's GAC/MRV/LCV stays wholly inside one solve_symbolic call,
per the June law -- the tree never reaches into the solver's own search).

TERMINAL / THE JUDGE: a node with status "solved" AND unique is True is a SUCCESS -- the
search stops immediately (adaptive stopping) and that node's value is the row's answer.
A node with status "unsat" (a TRUE solver-certified contradiction, never "budget" /
"timeout" / "unbuildable" -- those are inconclusive, not a certificate) is a DEAD LEAF:
no children are generated from it, but every OTHER node already on the frontier (siblings,
cousins, anything not a descendant) is untouched and the search continues -- this IS
backtracking, expressed as "the frontier's next-best node is popped" rather than as an
explicit stack unwind. If the budget or depth cap is exhausted with no success, the row
KEEPS TOP-1 (its root node's own solved value if solved, else refused/wrong) --
adaptive stopping never *removes* an answer the row already had.

BUDGET: every genuinely new (not cache-hit) solve OR uniqueness call counts one unit,
<=64 per row (MP_BUDGET); the search stops opening new nodes once the budget is spent
(the row keeps whatever it found, success or top-1).

======================================================================
MCTS vs plain best-first (the task invites exactly this comparison)
======================================================================
Textbook MCTS/UCT balances a static PRIOR against back-propagated value estimates from
REPEATED stochastic rollouts, visiting a promising-but-unconfirmed branch more as its
estimate firms up. Here every rollout is DETERMINISTIC (the June solver gives the same
verdict every time) and every node is evaluated AT MOST ONCE (results are cached by parse
signature) -- there is nothing to average, so UCT's exploration bonus (its sqrt(log N / n)
term) is identically the degenerate case n<=1 everywhere it would ever fire, and the
algorithm collapses EXACTLY to "always expand the frontier node with the smallest static
prior" -- a textbook uniform-cost / greedy best-first search, not an approximation of MCTS
but its zero-exploration limit. Since every action's margin is >= 0, a node's priority
(sum of its path's margins) is never smaller than any of its ancestors' -- the frontier
is therefore a valid admissible ordering (Dijkstra-consistent: no child is ever popped
before its parent), so a plain priority-queue best-first search visits precisely the nodes
a UCT tree with its exploration constant -> 0 would visit, in the same order. This script
BUILDS the simpler one (a heapq priority-queue best-first search over the node lattice),
but keeps PRIOR / ROLLOUT / BACKTRACK as named, separable steps (action_prior(), the
evaluate() closure's solve+uniqueness call, and the "unsat -> no children" branch) so a
later learned value function could replace action_prior() (or add a backed-up value to
evaluate()'s result) without restructuring the search.

======================================================================
Inputs / outputs
======================================================================
  MP_DUMP     raw per-row head dump, default .cache/rawslots_wild_PMS8_241.pkl
  MP_KARGS    max ARGS branching list length, default 3 (combined_oracle's K_ARGS)
  MP_MTYPEOP  max TYPE/OP branching list length, default 2 (combined_oracle's M_TYPEOP)
  MP_VALMAX   max VALUE variants incl. baseline, default 4 (combined_oracle's VAL_MAX)
  MP_DEPTH    max committed actions per path, default 3
  MP_BUDGET   max solver(+uniqueness) calls per row, default 64
  MP_WALL     solver wall per call (s), default 3
  MP_WORKERS  row-level spawn-pool workers, default 8
  MP_SELFGATE expected root-only correct count, default 14 (combined_oracle's self-gate)
  MP_LIMIT    row cap for a fast debug run, default 0 (no cap)
  MP_ABLATE   1/0, also run the budget x depth ablation sweep, default 1
Outputs: MP_OUT (default .cache/mcts_parse_<dump-suffix>.txt) + a sibling .pkl of
per-row records for an offline study.
"""
import os
import sys
import json
import time
import pickle
import heapq

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

import combined_oracle as CO     # the three-axis machinery (args x type/op x values), reused verbatim
BO = CO.BO                       # beam_oracle: mask_row, slot_decisions, apply_flips, parse_sig, build_gv_nvv
SK = CO.SK                       # sinkhorn_claim: given_slots, affinity_row, onehot_digit_logits

UNIQ_BUDGET = CO.UNIQ_BUDGET      # 5000 -- the mint gate's own uniqueness budget (CLAUDE.md Sec.1)

DUMP = os.environ.get("MP_DUMP", ".cache/rawslots_wild_PMS8_241.pkl")
_suffix = os.path.basename(DUMP).replace("rawslots_wild_", "").replace("rawslots_", "").replace(".pkl", "")
OUT_TXT = os.environ.get("MP_OUT", f".cache/mcts_parse_{_suffix}.txt")
OUT_PKL = OUT_TXT.replace(".txt", ".pkl")
K_ARGS = int(os.environ.get("MP_KARGS", "3"))
M_TYPEOP = int(os.environ.get("MP_MTYPEOP", "2"))
VAL_MAX = int(os.environ.get("MP_VALMAX", "4"))
DEPTH_MAX = int(os.environ.get("MP_DEPTH", "3"))
BUDGET_MAX = int(os.environ.get("MP_BUDGET", "64"))
WALL = float(os.environ.get("MP_WALL", "3"))
WORKERS = int(os.environ.get("MP_WORKERS", "8"))
SELF_GATE_EXPECT = int(os.environ.get("MP_SELFGATE", "14"))
LIMIT = int(os.environ.get("MP_LIMIT", "0"))
RUN_ABLATE = bool(int(os.environ.get("MP_ABLATE", "1")))


# ----------------------------------------------------------------------------------------------------------
# THE ROLLOUT'S TWO CALLS (copied, not imported -- see the module docstring's "reuse, not reimplementation"
# note: combined_oracle.py's _solve_task / _uniqueness_task, minus the multiprocessing-task-tuple wrapper,
# since this search calls them in-process, sequentially, one node at a time).
# ----------------------------------------------------------------------------------------------------------
def _solve_parse(parse, q, key, wall):
    from admit_annotation import solve_walled
    from alternator_bridge import problem_from_algebra3
    gv, nvv = BO.build_gv_nvv(parse, q)
    gmax = max([int(v) for v in gv.values()] + [1]); m0 = int(min(10001, max(300, 2 * gmax, 2 * key)))
    res = {"status": "?"}
    for m in ([m0] + ([10000] if m0 < 10000 else [])):
        try:
            res = solve_walled(problem_from_algebra3(nvv, parse, gv, m), budget=5000, wall=wall)
        except Exception:
            return "unbuildable", None, None
        if res.get("status") == "solved":
            return "solved", int(res["assignment"][q]), m
    return res.get("status", "?"), None, None


def _check_unique(parse, q, val, m, wall):
    import signal
    from admit_annotation import _Timeout, _alarm
    from alternator_bridge import problem_from_algebra3
    from mycelium.doors import certify_unique
    gv, nvv = BO.build_gv_nvv(parse, q)
    old = signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, wall)
    try:
        problem2 = problem_from_algebra3(nvv, parse, gv, m)
        uniq = certify_unique(problem2, q, val, budget=UNIQ_BUDGET, seed=0)
    except _Timeout:
        uniq = None
    except Exception:
        uniq = None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
    return uniq


# ----------------------------------------------------------------------------------------------------------
# VALUE-AXIS MARGINS: CO.value_variants (combined_oracle.py:299-333) builds the variant list but does not
# return the affinity matrix it uses internally to rank collisions -- this recomputes ONLY that matrix
# (SK.affinity_row + mycelium.rulebook.legal_values, the same two calls CO.value_variants itself makes) to
# score each variant's "regret" (affinity given up vs the model's own baseline argmax), so the value axis
# has a margin on the SAME scale (a log-probability gap, always >= 0, 0 at baseline) as the args/typeop
# axes' margins. The COLLISION DETECTION / HUNGARIAN ASSIGNMENT LOGIC is NOT duplicated here -- only this
# derived score, from a quantity CO.value_variants already computed but did not expose.
# ----------------------------------------------------------------------------------------------------------
def value_variant_margins(raw_row, text, nd, gslots, val_variants):
    from mycelium.rulebook import legal_values
    if not gslots:
        return [0.0] * len(val_variants)
    candidates = legal_values(text, nd)
    if not candidates:
        return [0.0] * len(val_variants)
    A = np.stack([SK.affinity_row(raw_row["dig"][j], candidates, nd) for j in gslots])
    baseline_idx = [int(np.argmax(A[k])) for k in range(len(gslots))]
    cand_to_idx = {v: i for i, v in enumerate(candidates)}
    margins = []
    for label, vvals, vprio in val_variants:
        if not vvals:
            margins.append(0.0); continue
        regret = 0.0
        for k, v in enumerate(vvals):
            if v is None:
                continue
            vi = cand_to_idx.get(v)
            if vi is None:
                continue
            regret += float(A[k, baseline_idx[k]] - A[k, vi])
        margins.append(max(0.0, regret))
    return margins


# ----------------------------------------------------------------------------------------------------------
# THE SEARCH, one row. Returns a report dict (never raises on a row that simply fails to improve).
# ----------------------------------------------------------------------------------------------------------
def mcts_row(rec, nd, k_args, m_typeop, val_max, depth_max, budget_max, wall, decode_fn):
    i, text, key = rec["i"], rec["text"], rec["key"]
    raw_row = {k: np.asarray(rec[k], dtype=np.float64) for k in ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in rec else ())}
    q = int(np.asarray(rec["q"]).argmax())
    base_row = BO.mask_row(raw_row, text)

    args_decs = sorted(BO.slot_decisions(base_row), key=lambda d: d["margin"])[:k_args]
    typeop_decs = sorted(CO.typeop_decisions(base_row), key=lambda d: d["margin"])[:m_typeop]
    gslots, val_variants = CO.value_variants(raw_row, text, nd, val_max)
    val_margins = value_variant_margins(raw_row, text, nd, gslots, val_variants)

    # THE ACTION SET + PRIOR (action_prior): (axis, action-key, margin). One entry per committable flip.
    actions = []
    for d in args_decs:
        actions.append(("args", d["slot"], float(d["margin"])))
    for d in typeop_decs:
        actions.append(("typeop", (d["slot"], d["field"]), float(d["margin"])))
    for idx in range(1, len(val_variants)):
        actions.append(("value", idx, float(val_margins[idx])))

    solved_cache = {}      # parse signature -> result dict (dedup across flip-order-equivalent nodes)
    budget_used = [0]
    node_seen = set()

    def evaluate(node_key):
        """THE ROLLOUT. Decodes node_key's full graph, solves it, checks uniqueness if solved.
        Cache hits (a different flip ORDER reaching the same decoded graph) cost no budget."""
        args_flipped, typeop_flipped, val_idx = node_key
        row = BO.apply_flips(base_row, args_decs, set(args_flipped))
        row = CO.apply_typeop_flips(row, typeop_decs, set(typeop_flipped))
        if val_idx:
            row = CO.apply_value_variant(row, gslots, val_variants[val_idx][1], nd)
        parse = decode_fn(row)
        sig = BO.parse_sig(parse)
        if sig in solved_cache:
            return parse, sig, solved_cache[sig], False
        if key is None or not parse:
            res = dict(status="refused", value=None, m=None, unique=None)
            solved_cache[sig] = res
            return parse, sig, res, False
        if budget_used[0] >= budget_max:
            return parse, sig, dict(status="budget_exhausted", value=None, m=None, unique=None), False
        budget_used[0] += 1
        status, val, m = _solve_parse(parse, q, key, wall)
        unique = None
        if status == "solved" and budget_used[0] < budget_max:
            budget_used[0] += 1
            unique = _check_unique(parse, q, val, m, wall)
        res = dict(status=status, value=val, m=m, unique=unique)
        solved_cache[sig] = res
        return parse, sig, res, True

    root_key = ((), (), 0)
    frontier = [(0.0, 0, root_key)]
    best = None
    root_eval = None
    trace = []   # (node_key, depth, cmargin, status, value, unique, correct, spent_budget)

    while frontier:
        cmargin, depth, node_key = heapq.heappop(frontier)
        if node_key in node_seen:
            continue
        node_seen.add(node_key)
        parse, sig, res, spent = evaluate(node_key)
        if node_key == root_key:
            root_eval = res
        trace.append(dict(node=node_key, depth=depth, cmargin=cmargin, status=res["status"],
                           value=res["value"], unique=res.get("unique"), correct=(res["value"] == key),
                           budget_at=budget_used[0]))
        if res["status"] == "solved" and res.get("unique") is True:
            best = dict(node=node_key, depth=depth, value=res["value"], budget_at=budget_used[0])
            break                                   # ADAPTIVE STOPPING: first solved+unique node wins
        if res["status"] == "unsat":
            continue                                # BACKTRACK: a certified dead leaf, no children
        if depth >= depth_max or budget_used[0] >= budget_max:
            continue                                # depth/budget cap: no children from here either
        args_flipped, typeop_flipped, val_idx = node_key
        for axis, akey, margin in actions:
            if axis == "args":
                if akey in args_flipped: continue
                child = (tuple(sorted(args_flipped + (akey,))), typeop_flipped, val_idx)
            elif axis == "typeop":
                if akey in typeop_flipped: continue
                child = (args_flipped, tuple(sorted(typeop_flipped + (akey,))), val_idx)
            else:
                if val_idx != 0: continue
                child = (args_flipped, typeop_flipped, akey)
            if child in node_seen:
                continue
            heapq.heappush(frontier, (cmargin + margin, depth + 1, child))

    if root_eval is None:
        # the root is always the cheapest (margin 0) frontier entry, so this should not happen; guard anyway
        _, _, root_eval, _ = evaluate(root_key)

    top1_status = ("correct" if (root_eval["status"] == "solved" and root_eval["value"] == key) else
                    "wrong" if root_eval["status"] == "solved" else "refused")
    if best is not None:
        final_status = "correct" if best["value"] == key else "wrong"
        final_value = best["value"]; depth_of_success = best["depth"]
    else:
        final_status = top1_status; final_value = root_eval["value"] if root_eval["status"] == "solved" else None
        depth_of_success = None

    return dict(i=i, key=key, q=q, top1_status=top1_status, top1_value=(root_eval["value"] if root_eval["status"] == "solved" else None),
                final_status=final_status, final_value=final_value, success=(best is not None),
                depth_of_success=depth_of_success, budget_used=budget_used[0], n_nodes_visited=len(trace),
                n_actions=len(actions), trace=trace)


def _mcts_row_task(args):
    """Pool-task wrapper (one row's WHOLE search, run in a spawned worker -- parallelism is ACROSS rows;
    the search INSIDE one row is sequential by construction, see the module docstring)."""
    import os as _os, sys as _sys
    _sys.path.insert(0, "."); _sys.path.insert(0, "scripts")
    rec, nd, k_args, m_typeop, val_max, depth_max, budget_max, wall = args
    from phase1_algebra_head import _decode_slots
    return mcts_row(rec, nd, k_args, m_typeop, val_max, depth_max, budget_max, wall, _decode_slots)


# ----------------------------------------------------------------------------------------------------------
def run_fixture(dump_path, nd, k_args, m_typeop, val_max, depth_max, budget_max, wall, workers, limit, log=print):
    t0 = time.time()
    recs = pickle.load(open(dump_path, "rb"))
    if limit:
        recs = recs[:limit]
    n = len(recs)
    assert nd == np.asarray(recs[0]["dig"]).shape[1], (
        f"N_DIG={nd} from the head but the dump's digits are {np.asarray(recs[0]['dig']).shape[1]} wide -- "
        f"run under the FAMILY ENV (ALG_WIDE=1 sets N_DIG=7)")
    tasks = [(rec, nd, k_args, m_typeop, val_max, depth_max, budget_max, wall) for rec in recs]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers) as pool:
        results = []
        for ri, res in enumerate(pool.imap(_mcts_row_task, tasks, chunksize=1)):
            results.append(res)
            if (ri + 1) % 50 == 0 or ri + 1 == n:
                log(f"[mcts-parse] {os.path.basename(dump_path)} depth<={depth_max} budget<={budget_max}: "
                    f"{ri+1}/{n} rows ({time.time()-t0:.0f}s)")
    log(f"[mcts-parse] {os.path.basename(dump_path)} depth<={depth_max} budget<={budget_max}: "
        f"{n} rows done in {time.time()-t0:.0f}s")
    return results


def summarize(results, label, top1_expect=None, judge_expect=None, ceiling_expect=None, P=print):
    n = len(results)
    top1_correct = sum(1 for r in results if r["top1_status"] == "correct")
    final_correct = sum(1 for r in results if r["final_status"] == "correct")
    successes = sum(1 for r in results if r["success"])
    regressions = [r["i"] for r in results if r["top1_status"] == "correct" and r["final_status"] != "correct"]
    fixes = [r["i"] for r in results if r["top1_status"] != "correct" and r["final_status"] == "correct"]
    budgets = np.array([r["budget_used"] for r in results], float)
    depths = [r["depth_of_success"] for r in results if r["depth_of_success"] is not None]
    P(f"-- {label}: n={n} --")
    P(f"  top1 correct: {top1_correct}/{n}" + (f"  (expected {top1_expect})" if top1_expect is not None else ""))
    P(f"  MCTS final correct: {final_correct}/{n}" +
      (f"   vs judge {judge_expect}, ceiling {ceiling_expect}" if judge_expect is not None else ""))
    P(f"  terminal successes (solved+unique node found): {successes}/{n}")
    P(f"  REGRESSIONS (top1 correct -> final not correct): {len(regressions)}  rows={regressions}")
    P(f"  FIXES       (top1 not correct -> final correct): {len(fixes)}  rows={fixes}")
    P(f"  budget-used histogram: mean={budgets.mean():.1f} median={np.median(budgets):.0f} "
      f"min={budgets.min():.0f} max={budgets.max():.0f} "
      f"p25={np.percentile(budgets,25):.0f} p75={np.percentile(budgets,75):.0f}")
    hist_edges = [0, 1, 2, 4, 8, 16, 32, 64, 1e9]
    counts, _ = np.histogram(budgets, bins=hist_edges)
    P("  budget buckets: " + ", ".join(f"[{hist_edges[i]:.0f},{hist_edges[i+1]:.0f})={counts[i]}" for i in range(len(counts) - 1)))
    if depths:
        dcounts = {d: depths.count(d) for d in sorted(set(depths))}
        P(f"  depth-of-success histogram (n={len(depths)} successes): {dcounts}")
    else:
        P("  depth-of-success histogram: no terminal successes")
    return dict(n=n, top1_correct=top1_correct, final_correct=final_correct, successes=successes,
                regressions=regressions, fixes=fixes, budgets=budgets.tolist(), depths=depths)


def main():
    from phase1_algebra_head import N_DIG
    lines = []
    def P(s=""):
        print(s); lines.append(s)

    P("=" * 90); P("MCTS OVER PARSES"); P("=" * 90)
    P(f"dump: {DUMP} | K_ARGS<={K_ARGS} M_TYPEOP<={M_TYPEOP} VAL_MAX<={VAL_MAX} | "
      f"DEPTH<={DEPTH_MAX} BUDGET<={BUDGET_MAX} | wall={WALL}s workers={WORKERS}")
    P("")
    P("reused, never edited: scripts/combined_oracle.py (CO.typeop_decisions combined_oracle.py:265-278,")
    P("CO.apply_typeop_flips :281-292, CO.value_variants :299-333, CO.apply_value_variant :336-346,")
    P("CO.UNIQ_BUDGET :208) -> scripts/beam_oracle.py (BO.mask_row :150-161, BO.slot_decisions :164-182,")
    P("BO.apply_flips :185-195, BO.parse_sig :198-201, BO.build_gv_nvv :204-210) and scripts/sinkhorn_claim.py")
    P("(SK.given_slots :94-103, SK.affinity_row :111-116); mycelium/doors.py's certify_unique :33-44;")
    P("scripts/admit_annotation.py's solve_walled/_Timeout/_alarm; scripts/alternator_bridge.py's")
    P("problem_from_algebra3. _solve_parse/_check_unique below are COPIES (not imports) of")
    P("combined_oracle.py's _solve_task/_uniqueness_task bodies, called in-process (see module docstring).")
    P("")

    t0 = time.time()
    results_wild = run_fixture(DUMP, N_DIG, K_ARGS, M_TYPEOP, VAL_MAX, DEPTH_MAX, BUDGET_MAX, WALL, WORKERS, LIMIT, log=P)
    P("")
    stats_wild = summarize(results_wild, f"WILD ({DUMP})", top1_expect=SELF_GATE_EXPECT,
                            judge_expect=18, ceiling_expect=43, P=P)
    gate_ok = (stats_wild["top1_correct"] == SELF_GATE_EXPECT) or LIMIT
    P("")
    P(f"SELF-GATE: top1 correct {stats_wild['top1_correct']} expected {SELF_GATE_EXPECT} -> "
      f"{'PASS' if gate_ok else 'MISMATCH -- investigate before trusting anything below'}" +
      (" (SKIPPED -- MP_LIMIT set)" if LIMIT else ""))
    P("")
    bar_ok_rows = stats_wild["final_correct"] >= 24
    bar_ok_regr = len(stats_wild["regressions"]) <= 2
    P(f"BAR (pinned 2026-10-05 19:24, the picker's own bar): rows >= 24 -> "
      f"{'PASS' if bar_ok_rows else 'MISS'} ({stats_wild['final_correct']}); "
      f"regressions <= 2 -> {'PASS' if bar_ok_regr else 'MISS'} ({len(stats_wild['regressions'])})")
    P(f"OVERALL: {'PASS' if (bar_ok_rows and bar_ok_regr) else 'MISS'}")
    P("")

    diet_dump = os.environ.get("MP_DIET_DUMP", ".cache/rawslots_slicevalid2_PMS8_241.pkl")
    stats_diet = None
    if os.path.exists(diet_dump) and not LIMIT:
        P("-" * 90); P("SECOND FIXTURE: the training-diet valid2 slice (never wild; a parity/robustness read only)"); P("-" * 90)
        results_diet = run_fixture(diet_dump, N_DIG, K_ARGS, M_TYPEOP, VAL_MAX, DEPTH_MAX, BUDGET_MAX, WALL, WORKERS, 0, log=P)
        P("")
        stats_diet = summarize(results_diet, f"DIET ({diet_dump})", top1_expect=310, judge_expect=296, P=P)
        P("")

    ablation = {}
    if RUN_ABLATE and not LIMIT:
        P("-" * 90); P("ABLATION (wild fixture): budget x depth, one knob varied at a time off the registered"); P("config (depth<=3, budget<=64, already reported above)"); P("-" * 90)
        for b in (16, 32, 64):
            if b == BUDGET_MAX:
                ablation[("budget", b)] = stats_wild
                continue
            r = run_fixture(DUMP, N_DIG, K_ARGS, M_TYPEOP, VAL_MAX, DEPTH_MAX, b, WALL, WORKERS, 0, log=P)
            ablation[("budget", b)] = summarize(r, f"budget<={b} depth<={DEPTH_MAX}", P=P)
            P("")
        for d in (1, 2, 3):
            if d == DEPTH_MAX:
                ablation[("depth", d)] = stats_wild
                continue
            r = run_fixture(DUMP, N_DIG, K_ARGS, M_TYPEOP, VAL_MAX, d, BUDGET_MAX, WALL, WORKERS, 0, log=P)
            ablation[("depth", d)] = summarize(r, f"budget<={BUDGET_MAX} depth<={d}", P=P)
            P("")
        P("  ABLATION SUMMARY TABLE:")
        P(f"    {'config':22s} {'final_correct':>14s} {'regressions':>12s} {'successes':>10s}")
        for b in (16, 32, 64):
            s = ablation[("budget", b)]
            P(f"    {'budget<='+str(b)+' depth<=3':22s} {s['final_correct']:>14d} {len(s['regressions']):>12d} {s['successes']:>10d}")
        for d in (1, 2, 3):
            s = ablation[("depth", d)]
            P(f"    {'budget<=64 depth<='+str(d):22s} {s['final_correct']:>14d} {len(s['regressions']):>12d} {s['successes']:>10d}")
        P("")

    P(f"TOTAL WALL TIME: {time.time()-t0:.0f}s")
    P("")
    P("READING: this is a SEARCH, not an enumeration -- it looks at as few candidates as the budget and the")
    P("depth cap allow before either finding a solved+unique node (adaptive stopping) or exhausting, in which")
    P("case the row keeps top-1 unchanged. Its rows-correct count sits wherever the bars above say it sits;")
    P("the budget/depth-of-success histograms are the honest cost of the search itself, not an argument for")
    P("or against it -- only the bar comparison is.")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")

    out = dict(wild=results_wild, diet=(results_diet if stats_diet is not None else None),
               stats_wild=stats_wild, stats_diet=stats_diet, ablation=ablation,
               config=dict(dump=DUMP, k_args=K_ARGS, m_typeop=M_TYPEOP, val_max=VAL_MAX,
                           depth_max=DEPTH_MAX, budget_max=BUDGET_MAX, wall=WALL))
    pickle.dump(out, open(OUT_PKL, "wb"))
    print(f"[mcts-parse] wrote {OUT_TXT} + {OUT_PKL} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
