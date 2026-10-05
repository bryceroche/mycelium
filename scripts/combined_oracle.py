"""combined_oracle.py -- THE COMBINED ORACLE BOUND WITH THE CONSISTENCY JUDGE
(2026-10-05, zero-GPU read, delegate).

QUESTION (the task brief): beam_oracle.py (2026-09-24) branched ARGUMENTS
alone (k lowest-margin decisions, 2^k parses, a perfect judge) and bounded
the ceiling at 18/311 (+4). sinkhorn_claim.py (2026-09-24) branched GIVEN
VALUES alone (Hungarian/Sinkhorn reassignment of colliding givens) and also
landed at 18/311 (+4), with no judge needed at all. THE ONE-HOT TOGGLE TREE
entry (2026-10-04 20:10) registered a third axis -- TYPE/OP (branch the k
lowest-margin ftype/op decisions) -- as "the zero-GPU read that decides
whether the toggle tree earns an arm", never built. This script is the union
of all three axes, bound together, with TWO reads:
  (a) the ORACLE ceiling: with a perfect judge, how many of the 311 wild rows
      does some candidate get right -- per axis ALONE and for the three axes
      COMBINED into one candidate set per row;
  (b) THE CONSISTENCY JUDGE: a judge that does NOT see the key -- among a
      row's candidates, it prefers ones the June solver calls CONSISTENT
      (status "solved") AND UNIQUE (mycelium.doors.certify_unique's budgeted
      certificate, the same door mint's uniqueness gate uses), tie-broken by
      the model's own log-likelihood of the candidate. This is the first of
      the three axis-reads to try an actual (keyless) picker rather than only
      bound what a perfect judge could buy.

======================================================================
REUSE, NOT REIMPLEMENTATION (per the task's instruction: never touch
scripts/phase1_algebra_head.py or any existing script; copy small pieces
with attribution where importing would be less clean than copying)
======================================================================
- _decode_slots, N_DIG: imported from phase1_algebra_head, exactly as
  beam_oracle.py / sinkhorn_claim.py / twin_gap.py already do (never
  forward/build_params -- the task's standing rule).
- mask_row (the standard CA_MASK numeral mask applied to every non-relation
  slot): imported from beam_oracle.py (byte-identical to sinkhorn_claim's
  mask_nonrel; beam_oracle's name is reused since this script already
  imports beam_oracle for slot_decisions/apply_flips/parse_sig/build_gv_nvv).
- slot_decisions, apply_flips, parse_sig, build_gv_nvv: imported from
  beam_oracle.py verbatim -- the ARGUMENTS axis is beam_oracle's own
  machinery, unmodified.
- given_slots, affinity_row, onehot_digit_logits: imported from
  sinkhorn_claim.py verbatim -- the VALUES axis's affinity/assignment
  machinery.
- certify_unique: imported from mycelium/doors.py -- THE uniqueness
  certificate (the same door mint's gate uses: bans the judge's candidate
  value from a FRESH problem's domain and re-solves; True only on a true
  'unsat' certificate, budget/solved-again both refuse by the door's own
  twelve-site lesson).
- _solve_task: COPIED (not imported) from chain_acc.py / beam_oracle.py /
  sinkhorn_claim.py -- every one of those three scripts carries its own
  copy of this exact function rather than importing it from one another
  (the established convention in this codebase for spawn-pool picklability:
  each script's copy does its own sys.path insertion inside the function
  body so a freshly spawned worker can import admit_annotation/
  alternator_bridge regardless of which module the task was queued from).
  This script's copy differs from the precedent in ONE way only: it also
  returns the domain rung `m` that solved the row, because the judge's
  uniqueness check (certify_unique) needs to rebuild an IDENTICAL fresh
  problem at the same m.

======================================================================
THE THREE AXES (per row, over the banked top-1 masked decode)
======================================================================
Let base_row = mask_row(raw_row, text) (the standard numeral mask on every
non-relation slot's digits -- byte-identical to chain_acc's CA_MASK=1 and
to beam_oracle's own self-gated baseline).

1. ARGUMENTS (beam_oracle, unmodified): for each present RELATION slot
   (ftype argmax == 0), the weaker of the top-2 (or top-1-vs-runner-up for
   ALG_DUP slots) argument pointer logits is a decision; margin = v[weaker]
   - v[runner-up]. Rank ascending, take the K_ARGS lowest-margin decisions,
   enumerate all 2^k swaps.

2. TYPE/OP (new this script -- THE ONE-HOT TOGGLE TREE's registered zero-GPU
   read, 2026-10-04 20:10, "branch the k lowest-margin ftype/op decisions on
   the banked dump, solve, count rows any branch gets right"): for every
   PRESENT slot, a ftype decision (swap the top-2 ftype classes' logits,
   margin = v[top1]-v[top2], over the family env's 9-way ftype: rel/given/
   mod/sel/pct/fdiv/macro-OP_APPLY/macro-FRAC_OF/macro-CHAIN_MUL -- classes
   6/7 ("dig2") and 3 ("sel") are NOT in this family's banked dump fields
   at all, so decode()'s own try/except (inside _decode_slots, unmodified)
   turns a flip into one of those classes into "no fact for this slot",
   exactly as a live flip into those classes would read in production); for
   every present slot whose ftype argmax == 0 ("rel") ALSO an op decision
   (swap add<->mul, the op head's only two classes, margin = |op[0]-op[1]|).
   Rank the combined ftype+op decision list ascending, take the M_TYPEOP
   lowest-margin ones, enumerate 2^m swaps (bit i's slot+field is read off
   the SAME swap-the-two-logit-values technique beam_oracle uses for args --
   no decode logic reimplemented, decode()'s own argmax/argsort does the
   work on an edited copy).

3. VALUES (sinkhorn_claim's machinery, new combination logic): given_slots
   of the RAW (pre-mask) row; candidates = legal_values(text, N_DIG); A =
   each given slot's raw digit-head affinity (log-prob) over candidates
   (sinkhorn_claim.affinity_row, verbatim). baseline_vals = per-slot argmax
   of A (== what the numeral mask already picks -- the self-gate's
   equality). COLLISIONS = >=2 given slots claiming the same baseline
   numeral. Value variants (a short list, not a bitmask power set -- each
   variant is ONE global reassignment of every given slot's value, applied
   via sinkhorn_claim.onehot_digit_logits):
     V0 baseline_vals (always)
     V1 the Hungarian optimal assignment (scipy linear_sum_assignment on
        -A; sinkhorn_claim found Hungarian and Sinkhorn T=0.5 tie at 18/311,
        so only the exact Hungarian is carried here -- "Sinkhorn" in the
        task's phrasing is satisfied by the identical-result exact solver,
        stated as an interpretation call below)
     V2.. for each COLLIDING given slot, independently: swap ONLY that
        slot's value to its own top-2 (runner-up) candidate by affinity,
        every other given slot held at baseline_vals -- "each colliding
        given's top-2 numeral by affinity" read literally. Colliding slots
        are ranked by their own (best-affinity - 2nd-affinity) gap
        ascending (the most locally uncertain swapped first) and capped at
        VAL_MAX-2 extra variants.
   Scope note (stated, not fixed): a TYPE/OP edit that flips a slot's ftype
   AWAY FROM "given" does not retract that slot's queued value edit --
   the value edit still overwrites that slot's dig field, which decode()
   then reads under WHATEVER ftype is in effect post-edit (e.g. as a mod's
   k, or not at all if the post-edit ftype's branch doesn't read dig) --
   composing a value edit with a type/op edit on the SAME slot is the one
   place the three axes are not perfectly orthogonal; it is rare (the
   type/op decision list is dominated by non-given slots in practice) and
   is accepted as this read's scope, not patched.

======================================================================
CANDIDATE SET PER ROW
======================================================================
Full cross product of (args bitmask in 0..2^k_args-1) x (typeop bitmask in
0..2^k_typeop-1) x (value variant index in 0..n_vals-1), decoded via
_decode_slots (unmodified) on a copy of base_row with all three edits
applied, deduplicated by beam_oracle's parse_sig (the wheel's own memo
key). Each candidate's RANK = args bitmask's bit_length() + typeop
bitmask's bit_length() + value variant's priority index (0 baseline, 1
Hungarian, 2.. the swaps in gap order) -- the combined "how many edits
away from top-1" measure used both to prioritize ties at the same
signature (keep the cheapest-edit route to a given parse) and, if the
deduplicated set still exceeds CAND_MAX, to truncate (lowest rank kept
first -- this is why every single-axis-alone combination, whose rank is
bounded by that one axis's own k_needed, always survives truncation).
Defaults: K_ARGS<=3 (<=8), M_TYPEOP<=2 (<=4), VAL_MAX<=4, CAND_MAX<=64 --
exactly the task brief's own example bound.

======================================================================
THE CONSISTENCY JUDGE (does not see the key)
======================================================================
For every row, after solving every distinct candidate parse (the June
solver, chain_acc's exact domain-ladder rule, a hang-proof spawn pool):
  SOLVED pool = candidates whose solve status == "solved".
  UNIQUE pool = the SOLVED pool restricted to candidates that pass
    certify_unique (a FRESH problem rebuilt at the same solving rung m,
    the found value banned from the query variable's domain, re-solved to
    a true 'unsat' certificate -- budget exhaustion or a second solution
    both REFUSE uniqueness, per the door's own law).
  JUDGE POOL = UNIQUE pool if non-empty, ELSE the SOLVED pool (stated
    fallback -- see "interpretations" below; a row with zero solved
    candidates has an empty judge pool and the judge REFUSES).
  JUDGE'S PICK = argmax over JUDGE POOL of candidate_loglik (the sum, over
    every fact in the candidate's decoded parse, of the RAW (pre-any-edit)
    model logits' log-softmax score at exactly the fields decode() reads
    for that fact's type -- ftype's chosen class, op's chosen class and
    both/the one argument position for a relation, or the digit-head
    log-probability of the chosen numeral for every type that carries a
    value -- ALWAYS scored against the row's ORIGINAL unedited logits, so
    every candidate in a row is comparable on one scale regardless of
    which axis's edits produced it).
Zero-GPU: only phase1_algebra_head._decode_slots / N_DIG are imported from
the head (never forward/build_params, the task's standing rule). Run under
DEV=CPU and the exact family env (sets N_DIG=7/L_FAC=24/K_VARS=24 to match
the banked dump).

Inputs:
  CO_DUMP      raw per-row head dump, default .cache/rawslots_wild_PMS8_241.pkl
  CO_ROWS      fixture (unused directly; present for parity with sinkhorn_claim)
  CO_KARGS     max ARGUMENTS branching depth, default 3
  CO_MTYPEOP   max TYPE/OP branching depth, default 2
  CO_VALMAX    max VALUES variants per row (incl. baseline), default 4
  CO_CANDMAX   max distinct candidates kept per row after dedup, default 64
  CO_WALL      solver wall per attempt (s), default 3 (chain_acc's default)
  CO_WORKERS   spawn-pool workers, default 4
  CO_UNIQ      1/0, run the uniqueness-certificate phase, default 1
  CO_SELFGATE  expected k=all-baseline correct count, default 14
  CO_LIMIT     row cap for a fast debug run, default 0 (no cap)
Outputs: .cache/combined_oracle_PMS8_241.txt (report) +
         .cache/combined_oracle_PMS8_241.pkl (per-row candidate records).
"""
import os
import sys
import json
import time
import pickle

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from scipy.optimize import linear_sum_assignment

import beam_oracle as BO          # slot_decisions, apply_flips, parse_sig, build_gv_nvv, mask_row (args axis)
import sinkhorn_claim as SK       # given_slots, affinity_row, onehot_digit_logits (values axis)

DUMP = os.environ.get("CO_DUMP", ".cache/rawslots_wild_PMS8_241.pkl")
ROWS_PATH = os.environ.get("CO_ROWS", ".cache/wild_admitted_holdout.jsonl")
OUT_TXT = os.environ.get("CO_OUT", ".cache/combined_oracle_" + os.path.basename(DUMP).replace("rawslots_wild_", "").replace(".pkl", "") + ".txt")
OUT_PKL = OUT_TXT.replace(".txt", ".pkl")
K_ARGS = int(os.environ.get("CO_KARGS", "3"))
M_TYPEOP = int(os.environ.get("CO_MTYPEOP", "2"))
VAL_MAX = int(os.environ.get("CO_VALMAX", "4"))
CAND_MAX = int(os.environ.get("CO_CANDMAX", "64"))
WALL = float(os.environ.get("CO_WALL", "3"))
WORKERS = int(os.environ.get("CO_WORKERS", "4"))
RUN_UNIQ = bool(int(os.environ.get("CO_UNIQ", "1")))
SELF_GATE_EXPECT = int(os.environ.get("CO_SELFGATE", "14"))
UNIQ_BUDGET = 5000   # the mint gate's own uniqueness budget (mycelium/doors.py's door; CLAUDE.md §1)


# ----------------------------------------------------------------------------------------------------------
# THE SOLVE (copied, not imported -- see the module docstring's "reuse, not reimplementation" note).
# chain_acc._solve_task / beam_oracle._solve_task / sinkhorn_claim._solve_task, verbatim semantics, with
# ONE addition: also returns the domain rung `m` that solved the row (the judge's uniqueness check needs
# to rebuild an identical fresh problem at that same m).
# ----------------------------------------------------------------------------------------------------------
def _solve_task(t):
    import os as _os, sys as _sys
    _sys.path.insert(0, "."); _sys.path.insert(0, "scripts")
    from admit_annotation import solve_walled
    from alternator_bridge import problem_from_algebra3
    task_id, q, key, parse, gv, nvv = t
    wall = float(_os.environ.get("CO_WALL", "3"))
    gmax = max([int(v) for v in gv.values()] + [1]); m0 = int(min(10001, max(300, 2 * gmax, 2 * key)))
    res = {"status": "?"}
    for m in ([m0] + ([10000] if m0 < 10000 else [])):
        try:
            res = solve_walled(problem_from_algebra3(nvv, parse, gv, m), budget=5000, wall=wall)
        except Exception:
            return task_id, "unbuildable", None, None
        if res.get("status") == "solved":
            return task_id, "solved", int(res["assignment"][q]), m
    return task_id, res.get("status", "?"), None, None


def _uniqueness_task(t):
    """THE CONSISTENCY JUDGE's second door: mycelium.doors.certify_unique on a FRESH problem rebuilt
    at the same (nvv, parse, gv, m) that solved -- bans the found value from q's domain, re-solves;
    True only on a true 'unsat' certificate (budget/solved-again both refuse, the door's own law).
    Wrapped in the same SIGALRM wall admit_annotation.solve_walled uses (certify_unique itself has no
    wall -- a budget-bounded search can still run long wall-clock on an adversarial edit)."""
    import os as _os, sys as _sys, signal as _signal
    _sys.path.insert(0, "."); _sys.path.insert(0, "scripts")
    from admit_annotation import _Timeout, _alarm
    from alternator_bridge import problem_from_algebra3
    from mycelium.doors import certify_unique
    task_id, q, parse, gv, nvv, m, val = t
    wall = float(_os.environ.get("CO_WALL", "3"))
    old = _signal.signal(_signal.SIGALRM, _alarm); _signal.setitimer(_signal.ITIMER_REAL, wall)
    try:
        problem2 = problem_from_algebra3(nvv, parse, gv, m)
        uniq = certify_unique(problem2, q, val, budget=UNIQ_BUDGET, seed=0)
    except _Timeout:
        uniq = None   # unknown (timeout) -- treated as NOT certified unique, stated below
    except Exception:
        uniq = None
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0); _signal.signal(_signal.SIGALRM, old)
    return task_id, uniq


# ----------------------------------------------------------------------------------------------------------
# AXIS 2: TYPE/OP decisions (new; the ftype 9-way top-2 swap + the rel-only add/mul swap)
# ----------------------------------------------------------------------------------------------------------
def typeop_decisions(row):
    decs = []
    pres, ftype, op = row["pres"], row["ftype"], row["op"]
    for j in range(pres.shape[0]):
        if pres[j] <= 0:
            continue
        v = ftype[j]; order = np.argsort(-v)
        a, b = int(order[0]), int(order[1])
        decs.append(dict(slot=int(j), field="ftype", idx_a=a, idx_b=b, margin=float(v[a] - v[b])))
        if a == 0:   # currently decodes as "rel" -> the op head applies (decode()'s own convention)
            ov = op[j]
            oa, ob = (0, 1) if ov[0] >= ov[1] else (1, 0)
            decs.append(dict(slot=int(j), field="op", idx_a=oa, idx_b=ob, margin=float(abs(ov[0] - ov[1]))))
    return decs


def apply_typeop_flips(row, decisions, flip_keys):
    """row with ftype[j]/op[j] value-swapped at (idx_a, idx_b) for every decision whose (slot, field)
    is in flip_keys -- the same swap-the-two-logits technique beam_oracle.apply_flips uses for args,
    extended to two fields (a slot can carry both a ftype and an op decision)."""
    if not flip_keys:
        return row
    row = dict(row); row["ftype"] = row["ftype"].copy(); row["op"] = row["op"].copy()
    for d in decisions:
        if (d["slot"], d["field"]) in flip_keys:
            arr = row[d["field"]]; j, a, b = d["slot"], d["idx_a"], d["idx_b"]
            arr[j][a], arr[j][b] = arr[j][b], arr[j][a]
    return row


# ----------------------------------------------------------------------------------------------------------
# AXIS 3: VALUES variants (sinkhorn_claim's affinity/assignment machinery, recombined as a short list of
# whole-row reassignments rather than a single committed pick)
# ----------------------------------------------------------------------------------------------------------
def value_variants(raw_row, text, nd, val_max):
    """Returns (gslots, [(label, values, prio), ...]) -- values is a list parallel to gslots."""
    from mycelium.rulebook import legal_values
    candidates = legal_values(text, nd)
    gslots = SK.given_slots(raw_row)
    G, M = len(gslots), len(candidates)
    if G == 0:
        return gslots, [("V0_baseline", [], 0)]
    A = np.stack([SK.affinity_row(raw_row["dig"][j], candidates, nd) for j in gslots]) if M else np.zeros((G, 0))
    baseline_vals = [candidates[int(np.argmax(A[k]))] for k in range(G)] if M else [None] * G
    variants = [("V0_baseline", list(baseline_vals), 0)]
    if M:
        # V1: the Hungarian optimum (sinkhorn_claim's own finding: Hungarian and Sinkhorn T=0.5 tie at
        # +4 rows -- only the exact solver is carried here; see the module docstring's interpretation note)
        row_ind, col_ind = linear_sum_assignment(-A)
        hmap = dict(zip(row_ind.tolist(), col_ind.tolist()))
        hvals = [candidates[hmap[k]] if k in hmap else baseline_vals[k] for k in range(G)]
        variants.append(("V1_hungarian", hvals, 1))
        # V2..: each COLLIDING given's own top-2 (runner-up) numeral, one swap at a time, others held at
        # baseline -- ranked by that slot's own (best-2nd) affinity gap ascending (most locally uncertain first)
        from collections import Counter
        counts = Counter(v for v in baseline_vals if v is not None)
        colliding = [k for k in range(G) if baseline_vals[k] is not None and counts[baseline_vals[k]] >= 2]
        gaps = []
        for k in colliding:
            order = np.argsort(-A[k])
            if len(order) < 2:
                continue
            gap = float(A[k, order[0]] - A[k, order[1]])
            gaps.append((gap, k, candidates[int(order[1])]))
        gaps.sort(key=lambda x: x[0])
        for prio, (gap, k, runner_up) in enumerate(gaps[:max(0, val_max - 2)], start=2):
            vals = list(baseline_vals); vals[k] = runner_up
            variants.append((f"V{prio}_swap_slot{gslots[k]}", vals, prio))
    return gslots, variants[:val_max]


def apply_value_variant(row, gslots, values, nd):
    """sinkhorn_claim.apply_given_assignment, verbatim (one-hot-rewrites every given slot's digits to
    the variant's assigned value; every other slot's dig is untouched -- it already carries base_row's
    ordinary numeral mask from mask_row)."""
    if not gslots:
        return row
    row = dict(row); row["dig"] = row["dig"].copy()
    for j, v in zip(gslots, values):
        if v is not None:
            row["dig"][j] = SK.onehot_digit_logits(v, nd, row["dig"][j])
    return row


# ----------------------------------------------------------------------------------------------------------
# THE JUDGE'S SCORE: candidate_loglik, scored ALWAYS against the row's RAW (pre-any-edit) logits so every
# candidate in a row is comparable on one scale -- see the module docstring's "THE CONSISTENCY JUDGE" note.
# ----------------------------------------------------------------------------------------------------------
_FTYPE_IDX = {"given": 1, "rel": 0, "mod": 2, "sel": 3, "pct": 4, "fdiv": 5}
_MACRO_IDX = {"OP_APPLY": 6, "FRAC_OF": 7, "CHAIN_MUL": 8}


def _digit_loglik(dig_logits, value, nd):
    from mycelium.rulebook import _lsm, digits_of
    ls = _lsm(dig_logits)
    return float(sum(ls[d, dd] for d, dd in enumerate(digits_of(int(value), nd))))


def fact_loglik(raw_row, fact, nd):
    from mycelium.rulebook import _lsm
    j = fact["_slot"]
    ftype_ls = _lsm(raw_row["ftype"][j])
    idx = _MACRO_IDX[fact["name"]] if fact["ftype"] == "macro" else _FTYPE_IDX.get(fact["ftype"])
    score = float(ftype_ls[idx]) if idx is not None else 0.0
    if fact["ftype"] == "rel":
        op_ls = _lsm(raw_row["op"][j]); score += float(op_ls[0 if fact["op"] == "add" else 1])
        args_ls = _lsm(raw_row["args"][j])
        a0, a1 = fact["args"][0], fact["args"][1]
        score += float(args_ls[a0]) if a0 == a1 else float(args_ls[a0] + args_ls[a1])
    elif fact["ftype"] == "given":
        score += _digit_loglik(raw_row["dig"][j], abs(fact["value"]), nd)
    elif fact["ftype"] == "mod":
        score += float(_lsm(raw_row["args"][j])[fact["var"]]) + _digit_loglik(raw_row["dig"][j], fact["k"], nd)
    elif fact["ftype"] == "pct":
        args_ls = _lsm(raw_row["args"][j])
        score += float(sum(args_ls[a] for a in fact["args"])) + _digit_loglik(raw_row["dig"][j], fact["p"], nd)
    elif fact["ftype"] == "fdiv":
        score += float(_lsm(raw_row["args"][j])[fact["var"]]) + _digit_loglik(raw_row["dig"][j], fact["k"], nd)
    elif fact["ftype"] == "sel":
        args_ls = _lsm(raw_row["args"][j])
        score += float(sum(args_ls[a] for a in fact["args"]))
    elif fact["ftype"] == "macro":
        args_ls = _lsm(raw_row["args"][j])
        if "xs" in fact:
            score += float(sum(args_ls[a] for a in fact["xs"]))
        if "x" in fact:
            score += float(args_ls[fact["x"]])
        if "y" in fact and "y" in raw_row:
            score += float(_lsm(raw_row["y"][j])[fact["y"]])
        if "a" in fact:
            score += _digit_loglik(raw_row["dig"][j], fact["a"], nd)
        if "k" in fact:
            score += _digit_loglik(raw_row["dig"][j], fact["k"], nd)
    return score


def candidate_loglik(raw_row, parse, nd):
    return sum(fact_loglik(raw_row, f, nd) for f in parse)


# ----------------------------------------------------------------------------------------------------------
def main():
    from phase1_algebra_head import _decode_slots, N_DIG
    t0 = time.time()
    recs = pickle.load(open(DUMP, "rb"))
    _limit = int(os.environ.get("CO_LIMIT", "0"))
    if _limit:
        recs = recs[:_limit]
    n = len(recs)
    assert N_DIG == np.asarray(recs[0]["dig"]).shape[1], (
        f"N_DIG={N_DIG} from the head but the dump's digits are {np.asarray(recs[0]['dig']).shape[1]} wide -- "
        f"run under the FAMILY ENV (ALG_WIDE=1 sets N_DIG=7)")
    nd = N_DIG
    print(f"[combined-oracle] {n} rows from {DUMP} | K_ARGS<={K_ARGS} M_TYPEOP<={M_TYPEOP} VAL_MAX<={VAL_MAX} "
          f"CAND_MAX<={CAND_MAX} | wall={WALL}s workers={WORKERS} uniq={int(RUN_UNIQ)}", flush=True)

    per_row = []
    solve_tasks = {}      # (row_idx, sig) -> (q, key, parse, gv, nvv)  [dedup across whole dataset too]
    n_cand_total = 0
    cand_counts = []

    for ri, rec in enumerate(recs):
        i, text, key = rec["i"], rec["text"], rec["key"]
        raw_row = {k: np.asarray(rec[k], dtype=np.float64) for k in ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in rec else ())}
        q = int(np.asarray(rec["q"]).argmax())
        base_row = BO.mask_row(raw_row, text)

        # axis 1: arguments
        args_decs = sorted(BO.slot_decisions(base_row), key=lambda d: d["margin"])
        k_args_row = min(K_ARGS, len(args_decs)); top_args = args_decs[:k_args_row]

        # axis 2: type/op
        typeop_decs = sorted(typeop_decisions(base_row), key=lambda d: d["margin"])
        k_typeop_row = min(M_TYPEOP, len(typeop_decs)); top_typeop = typeop_decs[:k_typeop_row]

        # axis 3: values
        gslots, val_variants = value_variants(raw_row, text, nd, VAL_MAX)

        branches = {}   # sig -> record
        for abit in range(1 << k_args_row):
            flip_slots = {top_args[b]["slot"] for b in range(k_args_row) if abit & (1 << b)}
            row_a = BO.apply_flips(base_row, top_args, flip_slots)
            a_k = abit.bit_length()
            for tbit in range(1 << k_typeop_row):
                flip_keys = {(top_typeop[b]["slot"], top_typeop[b]["field"]) for b in range(k_typeop_row) if tbit & (1 << b)}
                row_at = apply_typeop_flips(row_a, top_typeop, flip_keys)
                t_k = tbit.bit_length()
                for vlabel, vvals, vprio in val_variants:
                    row_final = apply_value_variant(row_at, gslots, vvals, nd)
                    parse = _decode_slots(row_final)
                    sig = BO.parse_sig(parse)
                    rank = a_k + t_k + vprio
                    if sig not in branches:
                        branches[sig] = dict(parse=parse, rank=rank, tags={(a_k, t_k, vprio)}, a_k=a_k, t_k=t_k)
                    else:
                        b = branches[sig]
                        b["tags"].add((a_k, t_k, vprio))
                        b["a_k"] = min(b["a_k"], a_k); b["t_k"] = min(b["t_k"], t_k)
                        b["rank"] = min(b["rank"], rank)

        # truncate to CAND_MAX by rank (keeps every single-axis-alone combo: their rank is bounded by
        # that one axis's own k_needed, see the module docstring)
        sigs_sorted = sorted(branches.keys(), key=lambda s: branches[s]["rank"])
        kept_sigs = sigs_sorted[:CAND_MAX]
        cand_counts.append(len(kept_sigs))
        n_cand_total += len(kept_sigs)

        row_branches = {}
        for sig in kept_sigs:
            b = branches[sig]
            gv, nvv = BO.build_gv_nvv(b["parse"], q)
            task_id = (ri, sig)
            if key is not None and b["parse"]:
                if task_id not in solve_tasks:
                    solve_tasks[task_id] = (q, key, b["parse"], gv, nvv)
                row_branches[sig] = dict(parse=b["parse"], rank=b["rank"], tags=b["tags"],
                                          a_k=b["a_k"], t_k=b["t_k"], task_id=task_id)
            else:
                row_branches[sig] = dict(parse=b["parse"], rank=b["rank"], tags=b["tags"],
                                          a_k=b["a_k"], t_k=b["t_k"], task_id=None,
                                          status="refused", value=None)

        per_row.append(dict(i=i, key=key, q=q, text=text, raw_row=raw_row, branches=row_branches,
                             k_args_row=k_args_row, k_typeop_row=k_typeop_row,
                             n_val_variants=len(val_variants)))
        if (ri + 1) % 50 == 0 or ri + 1 == n:
            print(f"[combined-oracle] decoded {ri+1}/{n} rows, {len(solve_tasks)} distinct solve tasks so far "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"[combined-oracle] {len(solve_tasks)} distinct (row,parse) solves queued "
          f"(candidates/row mean {np.mean(cand_counts):.1f}, max {max(cand_counts)}) -- solving...", flush=True)
    import multiprocessing as mp
    tasks = [(tid, *args) for tid, args in solve_tasks.items()]
    solved = {}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        it = pool.imap_unordered(_solve_task, tasks, chunksize=1)
        n_done = 0
        try:
            for _ in range(len(tasks)):
                task_id, st, val, m = it.next(timeout=WALL * 3 + 30)
                solved[task_id] = (st, val, m); n_done += 1
                if n_done % 500 == 0 or n_done == len(tasks):
                    print(f"[combined-oracle] solved {n_done}/{len(tasks)} ({time.time()-t0:.0f}s elapsed)", flush=True)
        except mp.TimeoutError:
            print(f"[combined-oracle] {len(tasks) - len(solved)} tasks never returned -- counted as refused", flush=True)
    t_solve = time.time() - t0

    # attach solve results back onto branches
    for row in per_row:
        for sig, b in row["branches"].items():
            if b.get("task_id") is not None:
                st, val, m = solved.get(b["task_id"], ("hung", None, None))
                b["status"] = st; b["value"] = val; b["m"] = m
            b["correct"] = (b.get("status") == "solved" and b.get("value") == row["key"])

    # ---------------- uniqueness phase (the judge's second door) ----------------
    uniq_results = {}
    if RUN_UNIQ:
        uniq_tasks = {}
        for row in per_row:
            for sig, b in row["branches"].items():
                if b.get("status") == "solved" and b.get("task_id") is not None:
                    if b["task_id"] not in uniq_tasks:
                        gv, nvv = BO.build_gv_nvv(b["parse"], row["q"])
                        uniq_tasks[b["task_id"]] = (row["q"], b["parse"], gv, nvv, b["m"], b["value"])
        u_list = [(tid, *args) for tid, args in uniq_tasks.items()]
        print(f"[combined-oracle] {len(u_list)} distinct solved parses -> uniqueness phase...", flush=True)
        with mp.get_context("spawn").Pool(WORKERS) as pool:
            it = pool.imap_unordered(_uniqueness_task, u_list, chunksize=1)
            n_done = 0
            try:
                for _ in range(len(u_list)):
                    task_id, uniq = it.next(timeout=WALL * 3 + 30)
                    uniq_results[task_id] = uniq; n_done += 1
                    if n_done % 500 == 0 or n_done == len(u_list):
                        print(f"[combined-oracle] uniqueness {n_done}/{len(u_list)} ({time.time()-t0:.0f}s elapsed)", flush=True)
            except mp.TimeoutError:
                print(f"[combined-oracle] {len(u_list) - len(uniq_results)} uniqueness checks never returned -- treated as not-unique", flush=True)
        for row in per_row:
            for sig, b in row["branches"].items():
                if b.get("status") == "solved":
                    b["unique"] = uniq_results.get(b["task_id"])
    t_uniq = time.time() - t0 - t_solve

    # ---------------- loglik (the judge's tie-break) ----------------
    for row in per_row:
        for sig, b in row["branches"].items():
            b["loglik"] = candidate_loglik(row["raw_row"], b["parse"], nd) if b["parse"] else float("-inf")

    # ================================================================================================
    # REPORT
    # ================================================================================================
    lines = []
    def P(s=""):
        print(s); lines.append(s)

    def baseline_branch(row):
        for sig, b in row["branches"].items():
            if (0, 0, 0) in b["tags"] or b["rank"] == 0:
                return b
        return None   # should not happen if the self-gate is to mean anything

    P("=" * 90); P("THE COMBINED ORACLE BOUND WITH THE CONSISTENCY JUDGE"); P("=" * 90)
    P(f"dump: {DUMP} ({n} rows) | K_ARGS<={K_ARGS} M_TYPEOP<={M_TYPEOP} VAL_MAX<={VAL_MAX} CAND_MAX<={CAND_MAX}")
    P(f"wall={WALL}s workers={WORKERS} | {len(tasks)} distinct solves ({t_solve:.0f}s) + "
      f"{len(uniq_results)} uniqueness checks ({t_uniq:.0f}s) = {time.time()-t0:.0f}s total")
    P("")

    # ---- self-gate ----
    top1_correct = top1_wrong = top1_refused = 0
    for row in per_row:
        b0 = baseline_branch(row)
        row["top1"] = b0
        if b0 is None:
            top1_refused += 1; row["top1_status"] = "refused"; continue
        if b0["status"] == "solved" and b0["correct"]:
            top1_correct += 1; row["top1_status"] = "correct"
        elif b0["status"] == "solved":
            top1_wrong += 1; row["top1_status"] = "wrong"
        else:
            top1_refused += 1; row["top1_status"] = "refused"
    gate_ok = (top1_correct == SELF_GATE_EXPECT)
    P(f"SELF-GATE (all three axes at baseline): CORRECT {top1_correct} | wrong {top1_wrong} | refused {top1_refused}")
    P(f"  expected CORRECT {SELF_GATE_EXPECT} (chain_acc's masked read) -> {'PASS' if gate_ok else 'MISMATCH -- investigate before trusting anything below'}")
    P("")

    # ---- oracle ceilings, per axis alone and combined ----
    def axis_alone_correct(row, which):
        any_right = False
        for sig, b in row["branches"].items():
            a_k, t_k, v_idx = None, None, None
            for (ak, tk, vi) in b["tags"]:
                if which == "args" and tk == 0 and vi == 0:
                    any_right = any_right or b["correct"]
                elif which == "typeop" and ak == 0 and vi == 0:
                    any_right = any_right or b["correct"]
                elif which == "values" and ak == 0 and tk == 0:
                    any_right = any_right or b["correct"]
        return any_right

    def combined_correct(row):
        return any(b["correct"] for b in row["branches"].values())

    P("-" * 90); P("ORACLE CEILING (a perfect judge; rows where SOME candidate matches the key)"); P("-" * 90)
    for which in ("args", "typeop", "values"):
        n_right = sum(1 for row in per_row if axis_alone_correct(row, which))
        n_recov = sum(1 for row in per_row if row["top1_status"] != "correct" and axis_alone_correct(row, which))
        P(f"  {which:8s} alone: {n_right}/{n}  (recoverable over top-1: +{n_recov})")
    n_right_c = sum(1 for row in per_row if combined_correct(row))
    n_recov_c = sum(1 for row in per_row if row["top1_status"] != "correct" and combined_correct(row))
    P(f"  {'combined':8s}:     {n_right_c}/{n}  (recoverable over top-1: +{n_recov_c})")
    P("")

    # ---- candidate-count distribution ----
    P("-" * 90); P("CANDIDATE-COUNT DISTRIBUTION (distinct deduped parses kept per row, post CAND_MAX truncation)"); P("-" * 90)
    cc = np.asarray(cand_counts, float)
    P(f"  n={len(cc)}  mean={cc.mean():.2f}  median={np.median(cc):.1f}  min={cc.min():.0f}  max={cc.max():.0f}  "
      f"p75={np.percentile(cc,75):.1f}  p90={np.percentile(cc,90):.1f}  at-cap(={CAND_MAX})={int((cc>=CAND_MAX).sum())}")
    P("")

    # ---- THE CONSISTENCY JUDGE ----
    def judge_pick(row):
        solved_bs = [(sig, b) for sig, b in row["branches"].items() if b.get("status") == "solved"]
        if not solved_bs:
            return None, "refused", "no_solved_candidate"
        unique_bs = [(sig, b) for sig, b in solved_bs if b.get("unique") is True] if RUN_UNIQ else []
        pool, fallback = (unique_bs, False) if unique_bs else (solved_bs, True)
        sig, b = max(pool, key=lambda sb: sb[1]["loglik"])
        return b, "solved", ("fallback_no_unique" if fallback else "unique_pool")

    judge_correct = judge_wrong = judge_refused = 0
    confusion = {}   # (top1_status, judge_status) -> count
    fallback_count = 0
    regressions = []   # top1 correct -> judge not correct
    fixes = []         # top1 not correct -> judge correct

    for row in per_row:
        jb, jstat, jnote = judge_pick(row)
        if jnote == "fallback_no_unique":
            fallback_count += 1
        if jb is None:
            judge_status = "refused"
        elif jb["correct"]:
            judge_status = "correct"
        else:
            judge_status = "wrong"
        row["judge_status"] = judge_status; row["judge_branch"] = jb
        if judge_status == "correct": judge_correct += 1
        elif judge_status == "wrong": judge_wrong += 1
        else: judge_refused += 1
        confusion[(row["top1_status"], judge_status)] = confusion.get((row["top1_status"], judge_status), 0) + 1
        t1 = row["top1"]
        differs = (jb is None) != (t1 is None) or (jb is not None and t1 is not None and
                   (jb.get("value"), jb.get("status")) != (t1.get("value"), t1.get("status")))
        if row["top1_status"] == "correct" and differs:
            regressions.append(row["i"])
        if row["top1_status"] != "correct" and judge_status == "correct":
            fixes.append(row["i"])

    P("-" * 90); P("THE CONSISTENCY JUDGE (does not see the key; prefers solved+unique, ties by log-likelihood)"); P("-" * 90)
    P(f"  JUDGE: correct {judge_correct}/{n} | wrong {judge_wrong} | refused {judge_refused}  (top-1 baseline: {top1_correct}/{n})")
    P(f"  judge pool fell back to solved-only (no unique candidate in the row) on {fallback_count}/{n} rows")
    P("")
    P("  CONFUSION (top1_status -> judge_status):")
    for (ts, js), cnt in sorted(confusion.items()):
        P(f"    {ts:8s} -> {js:8s} : {cnt}")
    P("")
    P(f"  PRECISION: of rows where the judge's pick DIFFERS from top-1 --")
    P(f"    FIXES   (top1 wrong/refused -> judge correct): {len(fixes)}  rows={fixes}")
    P(f"    REGRESSIONS (top1 correct -> judge not correct): {len(regressions)}  rows={regressions}")
    n_differ = sum(1 for row in per_row if (row["judge_branch"] is None) != (row["top1"] is None) or
                   (row["judge_branch"] is not None and row["top1"] is not None and
                    (row["judge_branch"].get("value"), row["judge_branch"].get("status")) !=
                    (row["top1"].get("value"), row["top1"].get("status"))))
    P(f"  of {n_differ} rows where the judge's pick differs from top-1: {len(fixes)} newly right, "
      f"{len(regressions)} newly wrong-or-refused (were right), "
      f"{n_differ - len(fixes) - len(regressions)} differ but stay wrong/refused either way")
    P("")

    P("READING: the oracle ceilings bound what a PERFECT judge could ever buy on this decode + these three")
    P("axes; the consistency judge is a judge that does not exist yet (no training, no key) built only from")
    P("the solver's own consistency+uniqueness signal and the model's own confidence -- its gap to the oracle")
    P("ceiling is the room an actual trained picker would have to close, and the precision table above is the")
    P("honest cost: whether a keyless judge, as built, helps or hurts relative to the top-1 decode alone.")

    with open(OUT_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")

    # per-row records for an offline study
    out_recs = []
    for row in per_row:
        out_recs.append(dict(
            row=row["i"], key=row["key"], top1_status=row["top1_status"], judge_status=row["judge_status"],
            n_candidates=len(row["branches"]),
            branches=[dict(parse=b["parse"], tags=sorted(b["tags"]), status=b.get("status"), value=b.get("value"),
                            unique=b.get("unique"), loglik=b.get("loglik"), correct=b["correct"])
                       for b in row["branches"].values()],
        ))
    pickle.dump(out_recs, open(OUT_PKL, "wb"))
    print(f"[combined-oracle] wrote {OUT_TXT} + {OUT_PKL} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
