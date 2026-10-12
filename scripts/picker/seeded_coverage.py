"""scripts/picker/seeded_coverage.py -- THE SEEDED READINGS, STEP 1: coverage + judges + agreement
(2026-10-12, zero-extra-GPU: reads the device grid's dumps, everything else on CPU). New file.

TWO SCORING CONVENTIONS, NEVER SILENTLY MIXED (stated once here, printed again at run start):
  (A) DN-SCORING (dn_common.py, 2026-10-12's own denoising-breath reads): per-slot, POSITIONAL
      (slot j's predicted fields vs gold's slot j), the SAME convention loop_val.py's own
      fac-exact headline uses. A row is "DN-right" iff EVERY gold-present slot is ok. Used here
      for ORACLE COVERAGE ("per dn_common's row scoring", the task's own words) -- it asks "does
      the RIGHT GRAPH appear among the readings", independent of whether it would solve to the
      right final answer.
  (B) SOLVE-SCORING (chain_acc.py / combined_oracle.py / pvalue_judge.py / minimality.py's own,
      established, convention): a candidate is "correct" iff it SOLVES (the June solver) and the
      solved query value equals the row's answer KEY. Used here for THE JUDGES (consistency judge,
      minimality, the solver-weighted vote) because that is what every judge script in this
      directory already means by "rows right" (pvalue_judge.py's BAR_ROWS=24, KILL_ROWS=20, the
      lattice's own "43/311" ceiling, combined_oracle.py's b["correct"]) -- reusing their own
      machinery rather than inventing a third convention for the vote.
  THE "COMBINED WITH THE LATTICE" NUMBER mixes (A) (our new rows) with the lattice's own
  PRE-COMPUTED (B)-scored 43 rows (.cache/combined_oracle_PMS8_241.pkl's own b["correct"] flags)
  -- a deliberate union of two differently-scored "is the right graph reachable from the first
  look" measurements, not a re-scoring of the lattice under (A) (the lattice's pickle does not
  carry post-flip raw arrays, only parses, so a dn_common re-score would have to reconstruct
  flipped per-slot arrays generically -- fragile, skipped; the mix is reported AS a mix, not
  disguised as one convention).

INPUT: .cache/seeded_dump_<BODY>_t<TEMP>_s<SEED>_h<HOLD>_sch<SCHED>.pkl (scripts/loop_val.py's new
LV_DUMP_FULL door, one record per row: {i, text, key, q, pres, ftype, op, dig, args, res[, dup]}).

Usage:
  DEV=CPU .venv/bin/python3 scripts/picker/seeded_coverage.py \
    [SC_BODIES=PMS8_241,SR_241] [SC_HOLD=0] [SC_SCHED=0] [SC_WALL=10] [SC_WORKERS=8]
    [SC_ROW_LIMIT=0] [SC_SKIP_JUDGES=0] [SC_SKIP_VOTE=0]
"""
import os
import sys
import glob
import pickle
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np

import dn_common as DN
import beam_oracle as BO
import combined_oracle as CO
import minimality as MIN

BODIES = os.environ.get("SC_BODIES", "PMS8_241,SR_241").split(",")
HOLD = os.environ.get("SC_HOLD", "0")
SCHED = os.environ.get("SC_SCHED", "0")
WALL = float(os.environ.get("SC_WALL", "10"))
WORKERS = int(os.environ.get("SC_WORKERS", "8"))   # RESERVED, not yet wired: judge_row() below
                                                     # solves single-process (signal.alarm walls,
                                                     # the minimality.py idiom); a full 311-row run
                                                     # would parallelize the same way gen_candidates.
                                                     # py's pool does -- left single-process here for
                                                     # correctness-first under time pressure.
ROW_LIMIT = int(os.environ.get("SC_ROW_LIMIT", "0"))
SKIP_JUDGES = bool(int(os.environ.get("SC_SKIP_JUDGES", "0")))
SKIP_VOTE = bool(int(os.environ.get("SC_SKIP_VOTE", "0")))
GOLD_NPZ = os.environ.get("SC_GOLD", ".cache/phase1_alg_states_wildhold.npz")
LATTICE_PKL = os.environ.get("SC_LATTICE", ".cache/picker/cand_wild_PMS8_241.pkl")
BANKED_RAW = os.environ.get("SC_BANKED_RAW", ".cache/rawslots_wild_{body}.pkl")
BAR_COVERAGE = 50     # combined with the lattice's 43
BAR_VOTE_ROWS = 24
BAR_VOTE_REGR = 2
KILL_MARGIN = 5        # coverage no better than top-1 + 5 rows = KILL
DEAD_FACT_WEIGHT = 0.5  # the vote's down-weight for a non-minimal (dead-fact-carrying) candidate


def P(s=""):
    print(s, flush=True)


# =====================================================================================
# loading
# =====================================================================================
def load_pkl(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def find_dumps(body, hold, sched):
    pat = f".cache/seeded_dump_{body}_t*_s*_h{hold}_sch{sched}.pkl"
    return sorted(glob.glob(pat))


def decode_record(rec, l_fac=24):
    """A dump record's own raw per-slot arrays -> (parse, is DN-right). _decode_slots is the
    head's own top-1 decode (phase1_algebra_head.py:613) applied to whatever arrays this
    particular reading produced -- reused verbatim, not reimplemented."""
    import phase1_algebra_head as H
    row = {k: np.asarray(rec[k]) for k in ("pres", "ftype", "op", "dig", "args", "res")
           + (("dup",) if "dup" in rec else ())}
    parse = H._decode_slots(row)
    return parse, row


def dn_right(rec, row, vg, l_fac=24):
    scored = DN.score_row_slots(row, vg, rec["i"], l_fac)
    return bool(scored) and all(v["ok"] for v in scored.values())


# =====================================================================================
# per-row candidate collection across the N seeded readings of one (body, hold, sched) arm
# =====================================================================================
def collect_row_candidates(dumps_by_path, row_limit=0):
    """Returns {row_i: {"text":, "key":, "q":, "readings": [(tag, parse, row_arrays, dn_ok), ...],
    "distinct": {sig: best_reading_idx}}} -- one entry per row seen in ANY dump."""
    by_row = {}
    for path, recs in dumps_by_path.items():
        tag = os.path.basename(path)
        for rec in recs:
            i = int(rec["i"])
            if row_limit and i >= row_limit:
                continue
            parse, row_arrays = decode_record(rec)
            sig = BO.parse_sig(parse)
            ent = by_row.setdefault(i, {"text": rec["text"], "key": rec["key"],
                                        "q": int(np.asarray(rec["q"]).argmax()),
                                        "readings": [], "sigs": {}})
            ridx = len(ent["readings"])
            ent["readings"].append((tag, parse, row_arrays))
            ent["sigs"].setdefault(sig, ridx)   # first occurrence wins (dedup by signature)
    return by_row


# =====================================================================================
# THE JUDGES (solve-scoring convention; reusing minimality.py's wall + gv/nvv builders)
# =====================================================================================
def solve_and_min(parse, q, key, wall):
    """One candidate -> dict(value, m, unique, minimal, n_dead, loglik_placeholder)."""
    value, m = MIN.solve_query(parse, q, m=None, wall=wall)
    if value is None:
        return dict(status="unsolvable", value=None)
    uniq = MIN.is_forced(parse, q, value, m, wall=wall)
    mn = MIN.minimality(parse, q, value, m, wall=wall)
    return dict(status="solved", value=value, m=m, unique=bool(uniq), minimal=mn["minimal"],
                n_dead=mn["n_dead"], n_wall=mn["n_wall"])


def judge_row(entry, wall):
    """Consistency judge (solved + unique, loglik tie-break) and the solver-weighted vote
    (drop unsolvable, down-weight non-minimal, vote the rest, loglik tie-break) over this row's
    DISTINCT candidate signatures. Returns dict(consistency=..., vote=...)."""
    q, key = entry["q"], entry["key"]
    cands = []   # (sig, parse, raw_row, loglik)
    for sig, ridx in entry["sigs"].items():
        _, parse, raw_row = entry["readings"][ridx]
        # nd = the number of DIGIT POSITIONS (N_DIG, =7 under ALG_WIDE=1), not the per-position
        # value cardinality (10) -- dig's shape is (L_FAC, N_DIG, 10); combined_oracle.py's own
        # main() takes nd = N_DIG from the head, matched here via shape[-2].
        ll = CO.candidate_loglik(raw_row, parse, raw_row["dig"].shape[-2]) if parse else float("-inf")
        cands.append((sig, parse, raw_row, ll))
    scored = []
    for sig, parse, raw_row, ll in cands:
        r = solve_and_min(parse, q, key, wall)
        r.update(sig=sig, loglik=ll)
        scored.append(r)

    # --- consistency judge: solved + unique, tie-break by loglik ---
    uniq_solved = [r for r in scored if r["status"] == "solved" and r["unique"]]
    cj = None
    if uniq_solved:
        cj = max(uniq_solved, key=lambda r: r["loglik"])

    # --- the solver-weighted vote: drop unsolvable, down-weight non-minimal, vote, loglik tie ---
    solved = [r for r in scored if r["status"] == "solved"]
    vote_weight = {}
    vote_best = {}
    for r in solved:
        w = 1.0 if r["minimal"] else DEAD_FACT_WEIGHT
        vote_weight[r["value"]] = vote_weight.get(r["value"], 0.0) + w
        if r["value"] not in vote_best or r["loglik"] > vote_best[r["value"]]["loglik"]:
            vote_best[r["value"]] = r
    vote = None
    if vote_weight:
        best_w = max(vote_weight.values())
        tied = [v for v, w in vote_weight.items() if w == best_w]
        winner = max(tied, key=lambda v: vote_best[v]["loglik"])
        vote = vote_best[winner]

    return dict(consistency=cj, vote=vote, n_distinct=len(scored),
                n_solved=len(solved), n_unique=len(uniq_solved))


# =====================================================================================
# main
# =====================================================================================
def main():
    P("=" * 100)
    P("THE SEEDED READINGS -- coverage + judges + agreement (2026-10-12)")
    P("DN-SCORING (dn_common.py) for ORACLE COVERAGE; SOLVE-SCORING (chain_acc/combined_oracle/"
      "pvalue_judge/minimality convention) for THE JUDGES -- see module docstring; never mixed "
      "silently.")
    P("=" * 100)

    vg = DN.load_gold(GOLD_NPZ) if os.path.exists(GOLD_NPZ) else None
    if vg is None:
        P(f"WARNING: gold npz {GOLD_NPZ} not found -- DN-SCORING coverage will be skipped "
          f"(SOLVE-SCORING judges still run).")

    lattice_rows_right = set()
    if os.path.exists(LATTICE_PKL):
        lat = load_pkl(LATTICE_PKL)
        for row in lat:
            if any(b.get("correct") for b in row["branches"].values()):
                lattice_rows_right.add(int(row["i"]))
        P(f"lattice {LATTICE_PKL}: {len(lattice_rows_right)} rows with a SOLVE-SCORING-correct "
          f"candidate among the first look's branches (the established '43/311' ceiling check: "
          f"{len(lattice_rows_right)}==43 -> {len(lattice_rows_right) == 43})")
    else:
        P(f"lattice {LATTICE_PKL} not found -- 'combined with the lattice' numbers unavailable")

    for body in BODIES:
        P("-" * 100)
        P(f"BODY {body} (HOLD={HOLD} SCHED={SCHED})")
        paths = find_dumps(body, HOLD, SCHED)
        if not paths:
            P(f"  NO DUMPS FOUND for pattern .cache/seeded_dump_{body}_t*_s*_h{HOLD}_sch{SCHED}.pkl "
              f"-- the device read grid has not produced this arm's dumps yet.")
            continue
        P(f"  {len(paths)} dump(s): {[os.path.basename(p) for p in paths]}")
        dumps = {p: load_pkl(p) for p in paths}
        by_row = collect_row_candidates(dumps, ROW_LIMIT)
        n_rows = len(by_row)
        n_readings_per_row = [len(e["readings"]) for e in by_row.values()]
        n_distinct_per_row = [len(e["sigs"]) for e in by_row.values()]
        P(f"  rows seen: {n_rows} | readings/row: mean {np.mean(n_readings_per_row):.1f} "
          f"(expect {len(paths)}) | distinct graphs/row: mean {np.mean(n_distinct_per_row):.2f} "
          f"max {max(n_distinct_per_row) if n_distinct_per_row else 0}")

        # ---------------- ORACLE COVERAGE (DN-scoring) ----------------
        dn_covered = set()
        if vg is not None:
            for i, ent in by_row.items():
                for tag, parse, raw_row in ent["readings"]:
                    if dn_right({"i": i}, raw_row, vg):
                        dn_covered.add(i)
                        break
            banked_path = BANKED_RAW.format(body=body)
            banked_dn_covered = set()
            if os.path.exists(banked_path):
                for rec in load_pkl(banked_path):
                    _, raw_row = decode_record(rec)
                    if dn_right(rec, raw_row, vg):
                        banked_dn_covered.add(int(rec["i"]))
            P(f"  ORACLE COVERAGE (DN-scoring, alone): {len(dn_covered)}/{n_rows} rows "
              f"(banked top-1 DN-right: {len(banked_dn_covered)})")
            combined = dn_covered | lattice_rows_right
            P(f"  ORACLE COVERAGE combined with the lattice (DN-rows UNION lattice's SOLVE-"
              f"scoring-right rows -- a deliberate cross-convention union, see module docstring): "
              f"{len(combined)}/{n_rows}")
            bar_pass = len(combined) >= BAR_COVERAGE
            kill = len(dn_covered) <= len(banked_dn_covered) + KILL_MARGIN
            P(f"  BAR (combined coverage >= {BAR_COVERAGE}): {'PASS' if bar_pass else 'FAIL'} "
              f"({len(combined)}) | KILL (alone <= banked top-1 + {KILL_MARGIN}): "
              f"{'TRIGGERED' if kill else 'no'} ({len(dn_covered)} vs {len(banked_dn_covered)})")

        # ---------------- THE JUDGES (solve-scoring) ----------------
        if SKIP_JUDGES:
            P("  (judges skipped: SC_SKIP_JUDGES=1)")
            continue
        banked_path = BANKED_RAW.format(body=body)
        banked_right = {}
        if os.path.exists(banked_path):
            for rec in load_pkl(banked_path):
                i = int(rec["i"]); parse, _ = decode_record(rec)
                gv, nvv = BO.build_gv_nvv(parse, int(np.asarray(rec["q"]).argmax()))
                val, m = MIN.solve_query(parse, int(np.asarray(rec["q"]).argmax()), wall=WALL)
                banked_right[i] = (val is not None and val == rec["key"])
        n_bank_right = sum(banked_right.values())
        P(f"  banked top-1 (SOLVE-scoring) right: {n_bank_right}/{len(banked_right)}")

        t0 = time.time()
        cj_right = cj_regr = vote_right = vote_regr = 0
        cj_rows = []; vote_rows = []
        for n_done, (i, ent) in enumerate(by_row.items()):
            jr = judge_row(ent, WALL)
            key = ent["key"]
            bank_ok = banked_right.get(i, False)
            if jr["consistency"] is not None:
                ok = jr["consistency"]["value"] == key
                if ok:
                    cj_right += 1; cj_rows.append(i)
                elif bank_ok:
                    cj_regr += 1
            if jr["vote"] is not None:
                ok = jr["vote"]["value"] == key
                if ok:
                    vote_right += 1; vote_rows.append(i)
                elif bank_ok:
                    vote_regr += 1
            if (n_done + 1) % 50 == 0:
                P(f"    judged {n_done + 1}/{n_rows} ({time.time() - t0:.0f}s)")
        P(f"  CONSISTENCY JUDGE: {cj_right} rows right, {cj_regr} regressions vs banked top-1 "
          f"(floor 18, bar 24, ceiling 43 -- PMS8_241's own banked numbers)")
        P(f"  SOLVER-WEIGHTED VOTE: {vote_right} rows right, {vote_regr} regressions vs banked "
          f"top-1 (DEAD_FACT_WEIGHT={DEAD_FACT_WEIGHT})")
        bar_vote = vote_right >= BAR_VOTE_ROWS and vote_regr <= BAR_VOTE_REGR
        P(f"  BAR (vote >= {BAR_VOTE_ROWS} rows AND regressions <= {BAR_VOTE_REGR}): "
          f"{'PASS' if bar_vote else 'FAIL'}")

    # ---------------- AGREEMENT (per-slot, across seeds) ----------------
    P("-" * 100)
    P("PER-SLOT AGREEMENT ACROSS SEEDS vs PER-SLOT CORRECTNESS (AUROC) -- 'agreement is the "
      "confidence interval'")
    for body in BODIES:
        for temp in ("0.5", "1.0"):
            paths = sorted(glob.glob(f".cache/seeded_dump_{body}_t{temp}_s*_h{HOLD}_sch{SCHED}.pkl"))
            if len(paths) < 2 or vg is None:
                P(f"  {body} temp={temp}: {len(paths)} dump(s) -- need >= 2 and gold; skipped")
                continue
            dumps = [load_pkl(p) for p in paths]
            by_i = [{int(r["i"]): r for r in d} for d in dumps]
            common_i = sorted(set.intersection(*[set(d.keys()) for d in by_i]))
            agree_vals = []; correct_vals = []
            for i in common_i:
                recs = [d[i] for d in by_i]
                argmaxes = np.stack([np.asarray(r["res"]).argmax(-1) for r in recs])  # (n_seeds, L_FAC)
                agree = (argmaxes == argmaxes[0]).mean(0)   # (L_FAC,) share agreeing with seed 0
                row0 = {k: np.asarray(recs[0][k]) for k in ("pres", "ftype", "op", "dig", "args", "res")
                        + (("dup",) if "dup" in recs[0] else ())}
                scored = DN.score_row_slots(row0, vg, i)
                for j, s in scored.items():
                    agree_vals.append(float(agree[j])); correct_vals.append(1.0 if s["ok"] else 0.0)
            if len(set(correct_vals)) < 2:
                P(f"  {body} temp={temp}: degenerate (all-same correctness); AUROC undefined")
                continue
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(correct_vals, agree_vals)
            P(f"  {body} temp={temp}: {len(paths)} seeds, {len(agree_vals)} slot-readings, "
              f"AUROC(agreement -> correct) = {auroc:.4f}")


if __name__ == "__main__":
    main()
