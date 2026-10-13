"""scripts/picker/agreement_gated_vote.py -- THE AGREEMENT-GATED VOTE (2026-10-12, coordinator's
follow-up on THE SEEDED READINGS STEP 1). Zero-GPU; reads the device grid's banked dumps + the
banked top-1 raw decode; writes only new .cache/agv_* artifacts; no ledger write.

THE ALGORITHM. Starting from the banked TOP-1 graph per row (.cache/rawslots_wild_<BODY>.pkl's
own decode), a SLOT may be overridden by the 8 seeded tau=1.0 readings' vote only where (i) the
cross-seed agreement on that slot's WINNING value is >= a threshold, and (ii) applying the
override keeps the row's graph SOLVABLE (key-free: the June solver, not the answer). Two
agreement variants: 'all8' (agreement over all 8 seeds) and 'solvable_only' (agreement over only
the seeds whose OWN full graph solves) -- a seed's full-graph solvability is itself a key-free
signal, unlike the key.

A SLOT's "value" is its decoded FACT signature (beam_oracle.parse_sig's own per-fact convention,
json-dumped minus "_slot") -- EXACT structural identity, not just a field. "__ABSENT__" is a
value too (no fact at that slot).

ONE PASS PER (ROW, VARIANT), monotonic across the threshold sweep: candidate slots (mode value !=
top-1's own value, among the variant's voters) are sorted by agreement DESCENDING and offered to
the solver ONE AT A TIME against the currently-accepted graph; a candidate that keeps the graph
solvable is ACCEPTED and its solve (value, m, unique) recorded; one that breaks solvability is
REJECTED and never retried (permanently skipped, at every threshold -- the graph state the
rejection was tested against does not depend on which threshold we are about to report). Because
acceptance is tried in strictly decreasing agreement order, the report for threshold `th` is just
the PREFIX of the accepted trace whose agreement >= th -- no re-solving needed per threshold.
"Prefer unique": certify_unique is run at every ACCEPTED step (not gating, a diagnostic column).

SLOT-LEVEL PRECISION: a flipped slot's post-flip correctness is scored via dn_common's own
per-slot "ok" (the SAME gold-slot field check loop_val.py's fac-exact headline uses), read off
the DONOR seed reading's own raw row at that slot position (not re-derived from the fact dict --
exact, zero re-derivation risk); the pre-flip (top-1's own) correctness is read the same way off
top-1's own raw row. "Became right" = post-flip ok (the literal ask); fixed/broken are also
reported (pre-flip ok vs post-flip ok) as a bonus diagnostic.

ROW-LEVEL regressions use the ESTABLISHED solve-scoring convention (status==solved and value==
key) -- the SAME convention chain_acc.py / combined_oracle.py / the earlier seeded_coverage.py
judges use.

Usage (parallel, the gen_candidates.py pool idiom):
  env <FAMILY ENV incl. ALG_WIDE=1> AGV_BODIES=PMS8_241,SR_241 AGV_ARMS=base,sched1 AGV_WALL=3
      AGV_WORKERS=10 .venv/bin/python3 scripts/picker/agreement_gated_vote.py
Writes .cache/agv_<body>_<arm>.pkl (per-row per-variant per-threshold records) and prints the
compact table to stdout (and .cache/agreement_gated_vote.log via the launcher).
"""
import os
import sys
import json
import glob
import pickle
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np

BODIES = os.environ.get("AGV_BODIES", "PMS8_241,SR_241").split(",")
ARMS = os.environ.get("AGV_ARMS", "base,sched1").split(",")           # base=h0_sch0, sched1=h0_sch1
ARM_HS = {"base": (0, 0), "sched1": (0, 1), "hold1": (1, 0)}
WALL = float(os.environ.get("AGV_WALL", "3"))
WORKERS = int(os.environ.get("AGV_WORKERS", "10"))
ROW_LIMIT = int(os.environ.get("AGV_ROW_LIMIT", "0"))
GOLD_NPZ = os.environ.get("AGV_GOLD", ".cache/phase1_alg_states_wildhold.npz")
BANKED_RAW = os.environ.get("AGV_BANKED_RAW", ".cache/rawslots_wild_{body}.pkl")
THRESH = [1.0, 0.875, 0.75, 0.625, 0.5]   # DECREASING -- the monotonic sweep order
L_FAC = 24

BAR_SR_ROWS = 24
BAR_SR_REGR = 2


def P(s=""):
    print(s, flush=True)


def fsig(f):
    if f is None:
        return "__ABSENT__"
    return json.dumps({k: v for k, v in f.items() if k != "_slot"}, sort_keys=True, default=str)


def load_pkl(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def row_arrays(rec):
    return {k: np.asarray(rec[k]) for k in ("pres", "ftype", "op", "dig", "args", "res")
            + (("dup",) if "dup" in rec else ())}


# =====================================================================================
# building per-row jobs (decode only -- cheap, done in the MAIN process)
# =====================================================================================
def build_jobs(body, hold, sched, row_limit=0):
    import phase1_algebra_head as H
    banked = load_pkl(BANKED_RAW.format(body=body))
    top1_by_i = {int(r["i"]): r for r in banked}
    seed_paths = sorted(glob.glob(f".cache/seeded_dump_{body}_t1.0_s*_h{hold}_sch{sched}.pkl"))
    assert len(seed_paths) == 8, f"expected 8 tau=1.0 seeds, found {len(seed_paths)}: {seed_paths}"
    seed_dumps = [load_pkl(p) for p in seed_paths]
    seed_by_i = [{int(r["i"]): r for r in d} for d in seed_dumps]
    common_i = sorted(set(top1_by_i) & set.intersection(*[set(d) for d in seed_by_i]))
    if row_limit:
        common_i = common_i[:row_limit]
    jobs = []
    for i in common_i:
        t1rec = top1_by_i[i]
        t1row = row_arrays(t1rec)
        t1parse = H._decode_slots(t1row)
        q = int(np.asarray(t1rec["q"]).argmax())
        key = t1rec["key"]
        s_rows = [row_arrays(seed_by_i[k][i]) for k in range(8)]
        s_parses = [H._decode_slots(r) for r in s_rows]
        jobs.append(dict(i=i, q=q, key=key, t1_parse=t1parse, t1_row=t1row,
                          s_parses=s_parses, s_rows=s_rows))
    return jobs, [os.path.basename(p) for p in seed_paths]


# =====================================================================================
# the worker (spawned subprocess): candidate build + the single monotonic acceptance pass,
# for BOTH agreement variants, + slot-level precision via dn_common off the donor's own row.
# =====================================================================================
_WVG = None   # per-worker gold dict (loaded once via the pool initializer)


def _init_worker(gold_npz):
    global _WVG
    sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
    os.environ.setdefault("DEV", "CPU")
    import dn_common as DN
    _WVG = DN.load_gold(gold_npz)


def _apply(parse, slot, fact):
    """parse with `slot`'s own fact replaced by `fact` (None = removed)."""
    out = [f for f in parse if f.get("_slot") != slot]
    if fact is not None:
        nf = dict(fact); nf["_slot"] = slot
        out.append(nf)
    return out


def _solve(parse, q, wall):
    import minimality as MIN
    val, m = MIN.solve_query(parse, q, wall=wall)
    if val is None:
        return None, None, None
    uniq = MIN.is_forced(parse, q, val, m, wall=wall)
    return val, m, bool(uniq)


def _worker(job):
    import dn_common as DN
    import minimality as MIN
    i, q, key = job["i"], job["q"], job["key"]
    t1_parse, t1_row = job["t1_parse"], job["t1_row"]
    s_parses, s_rows = job["s_parses"], job["s_rows"]
    wall = job["wall"]
    t1_map = {f["_slot"]: f for f in t1_parse}
    t1_ok = DN.score_row_slots(t1_row, _WVG, i, L_FAC)   # {j: {"ok":...}} gold-present slots only

    val0, m0, uniq0 = _solve(t1_parse, q, wall)

    # which seeds are 'solvable' (their OWN full graph, key-free)
    seed_solved = []
    for p in s_parses:
        v, _, _ = _solve(p, q, wall)
        seed_solved.append(v is not None)

    out = {}
    for variant in ("all8", "solvable_only"):
        idxs = list(range(8)) if variant == "all8" else [k for k in range(8) if seed_solved[k]]
        n_v = len(idxs)
        cand = []   # (slot, agreement, winning_fact, donor_k)
        if n_v > 0:
            s_maps = [{f["_slot"]: f for f in s_parses[k]} for k in idxs]
            for j in range(L_FAC):
                sigs = {}
                for pos, k in enumerate(idxs):
                    f = s_maps[pos].get(j)
                    s = fsig(f)
                    sigs.setdefault(s, []).append((k, f))
                best_sig, best_list = max(sigs.items(), key=lambda kv: len(kv[1]))
                agree = len(best_list) / n_v
                if best_sig != fsig(t1_map.get(j)):
                    donor_k, donor_f = best_list[0]
                    cand.append((j, agree, donor_f, donor_k))
        cand.sort(key=lambda c: -c[1])

        # ONE monotonic pass: accept/reject each candidate, in descending-agreement order
        cur_parse = list(t1_parse)
        trace = []   # (slot, agree, fact, donor_k, val, m, uniq)
        for j, agree, fact, donor_k in cand:
            trial = _apply(cur_parse, j, fact)
            v, m, u = _solve(trial, q, wall)
            if v is not None:
                cur_parse = trial
                trace.append((j, agree, fact, donor_k, v, m, u))
            # else: rejected, permanently (cur_parse unchanged)

        # per-threshold report = the prefix of `trace` whose agreement >= th
        per_th = {}
        for th in THRESH:
            prefix = [t for t in trace if t[1] >= th]
            if prefix:
                j, agree, fact, donor_k, v, m, u = prefix[-1]
                final_val, final_uniq = v, u
            else:
                final_val, final_uniq = val0, uniq0
            flips = []
            for j, agree, fact, donor_k, v, m, u in prefix:
                post_ok = DN.score_row_slots(s_rows[donor_k], _WVG, i, L_FAC).get(j, {}).get("ok")
                pre_ok = t1_ok.get(j, {}).get("ok")
                flips.append(dict(slot=j, agree=agree, pre_ok=pre_ok, post_ok=post_ok))
            per_th[th] = dict(val=final_val, unique=final_uniq, n_flips=len(prefix), flips=flips)
        out[variant] = per_th

    return dict(i=i, key=key, val0=val0, uniq0=uniq0, n_seed_solved=sum(seed_solved),
                variants=out)


# =====================================================================================
# main
# =====================================================================================
def run_one(body, arm):
    hold, sched = ARM_HS[arm]
    t0 = time.time()
    jobs, seed_files = build_jobs(body, hold, sched, ROW_LIMIT)
    P(f"BODY {body} ARM {arm} (hold={hold} sched={sched}): {len(jobs)} rows, "
      f"8 seeds {seed_files}")
    for j in jobs:
        j["wall"] = WALL

    import multiprocessing as mp
    results = []
    with mp.get_context("spawn").Pool(WORKERS, initializer=_init_worker, initargs=(GOLD_NPZ,)) as pool:
        it = pool.imap_unordered(_worker, jobs, chunksize=1)
        n_done = 0
        try:
            for _ in range(len(jobs)):
                r = it.next(timeout=WALL * 20 + 60)
                results.append(r); n_done += 1
                if n_done % 50 == 0 or n_done == len(jobs):
                    P(f"  {n_done}/{len(jobs)} rows ({time.time()-t0:.0f}s)")
        except mp.TimeoutError:
            P(f"  WARNING: {len(jobs) - n_done} rows never returned (counted as top-1-only)")

    out_path = f".cache/agv_{body}_{arm}.pkl"
    pickle.dump(results, open(out_path, "wb"))
    P(f"  -> {out_path} ({len(results)} rows, {time.time()-t0:.0f}s total)")

    # ---------------- the compact table ----------------
    n_top1_right = sum(1 for r in results if r["val0"] == r["key"])
    for variant in ("all8", "solvable_only"):
        for th in THRESH:
            rows_right = regr = corr = n_flipped_slots = n_flip_scored = n_flip_post_ok = 0
            for r in results:
                d = r["variants"][variant][th]
                right_now = (d["val"] == r["key"])
                was_right = (r["val0"] == r["key"])
                rows_right += int(right_now)
                if was_right and not right_now:
                    regr += 1
                if (not was_right) and right_now:
                    corr += 1
                for fl in d["flips"]:
                    n_flipped_slots += 1
                    if fl["post_ok"] is not None:
                        n_flip_scored += 1
                        n_flip_post_ok += int(bool(fl["post_ok"]))
            precision = (n_flip_post_ok / n_flip_scored) if n_flip_scored else float("nan")
            net = rows_right - n_top1_right
            bar = ""
            if body == "SR_241":
                bar = "PASS" if (rows_right >= BAR_SR_ROWS and regr <= BAR_SR_REGR) else "FAIL"
            else:
                bar = "PASS" if (rows_right >= n_top1_right and net >= 0) else "FAIL"
            P(f"  [{variant:14s} th={th:.3f}] rows_right={rows_right:3d}/{len(results)} "
              f"(top1={n_top1_right}, net={net:+d}) regr={regr:2d} corr={corr:2d} "
              f"slots_flipped={n_flipped_slots:4d} precision={precision:.3f} "
              f"({n_flip_post_ok}/{n_flip_scored}) BAR={bar}")
    return results


def main():
    P("=" * 110)
    P("THE AGREEMENT-GATED VOTE (2026-10-12): top-1 + solver-gated slot overrides from the "
      "tau=1.0 seeded vote")
    P(f"THRESH sweep = {THRESH} | WALL={WALL}s | WORKERS={WORKERS}")
    P("=" * 110)
    for body in BODIES:
        for arm in ARMS:
            run_one(body, arm)
            P("-" * 110)


if __name__ == "__main__":
    main()
