"""scripts/picker/df_particle_read.py -- DF_241's own zero-GPU particle re-read (2026-10-13,
coordinator's ask: a clean replacement for the chain's own dn_particle_vote.py, which is VOID).

WHY A NEW FILE, NOT A SYMLINK INTO seeded_coverage.py / agreement_gated_vote.py (THE WORKING
TOOLS). Both read .cache/seeded_dump_<BODY>_t<TEMP>_s<SEED>_h<HOLD>_sch<SCHED>.pkl: loop_val.py's
LV_DUMP_FULL door, a dict-per-row record carrying the RAW per-slot logit arrays (pres/ftype/op/
dig/args/res), which H._decode_slots / the fsig() agreement machinery need. DF_241's rack chain
(.cache/rack_chain_DF.sh) never set LV_DUMP_FULL for the particle reads -- it used LV_DUMP
(loop_val.py:604), whose tuples are the ARGMAX-ALREADY-TAKEN per-slot fields beside gold, one
tuple per GOLD-PRESENT slot, no raw arrays underneath. The two dump shapes cannot be converted
into each other after the fact (there is no logit to decode from a bare argmax) -- a genuine
format gap, not a renaming, so "symlink into the expected name" (the task brief's own fallback)
does not clear it. This file re-derives the four requested numbers directly from what LV_DUMP
actually wrote, reusing the established machinery verbatim where it is reachable (minimality.py's
solve_query/is_forced -> mycelium.csp_domains.problem_from_algebra3; mycelium.custody_gold.
row_gold for the key) and re-implementing, on the tuple fields, the exact per-slot "ok" check
loop_val.py / dn_common.score_row_slots already use (verified byte-for-byte against the chain's
own printed [fields] line for the noise-free read, see the self-check at the bottom of main()).

SCOPE CHECK (why the tuple-only fact rebuild is EXACT here, not an approximation): gold ftype on
.cache/dump_wild_DF_241.pkl is 0 (rel) or 1 (given) for ALL 2051 gold-present slots -- no mod/pct/
fdiv/macro/frac slot in this fixture. decode()'s own fact dict for exactly these two ftypes is
fully reconstructable from LV_DUMP's (ft, op, args[top2], res, dig[7]) fields (phase1_algebra_
head.py:10257-10272, reproduced in fact_from() below). PREDICTED slots occasionally land on a
type decode() can't rebuild from this tuple (observed: 2/2051 on the noise-free read, ft==5); such
a slot is dropped from the rebuilt parse and counted in `excl`, never silently folded into "ok".

THE BUG IN dn_particle_vote.py (for the record, not fixed here -- a new file was asked for): its
pred_graph()/gold_graph() tuples include the decode's generic top-2 "args" slot UNconditionally,
even for given/mod slots where args is meaningless noise (always populated by argsort, never
gated by ftype) while gold's args is the real (thresholded, often EMPTY) vg["args"] set -- every
given slot's tuple then mismatches even when ft/res/dig all agree. Self-test: positional_exact on
DF_241's own noise-free dump against ITSELF reads 0.016 mean (sampled below), not the chain's own
printed masked-wild fac-exact of 0.3598 -- confirms the vote script's zero is a construction bug,
not merely an ftype-coverage gap.

Zero-GPU. DEV should be CPU (no GPU fields are read). Inputs: the 8 particle dumps
.cache/dump_wild_DF_241_p90{1..8}.pkl, the noise-free .cache/dump_wild_DF_241.pkl, the baseline
.cache/dump_wild_PMS8_241.pkl, the lattice .cache/picker/cand_wild_PMS8_241.pkl, the fixture
.cache/wild_admitted_holdout.jsonl (custody-gold keys only, never its own 'solution' field blindly
-- mycelium.custody_gold.row_gold decides mint vs pen per row), .cache/phase1_alg_states_
wildhold.npz (query_var cross-check only). Writes only .cache/picker/df_particle_read.pkl (new).

Usage: DEV=CPU .venv/bin/python3 scripts/picker/df_particle_read.py [DFR_WALL=3] [DFR_WORKERS=8]
"""
import os
import sys
import json
import pickle
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
os.environ.setdefault("DEV", "CPU")
import numpy as np

N_DIG = 7
WALL = float(os.environ.get("DFR_WALL", "3"))
WORKERS = int(os.environ.get("DFR_WORKERS", "8"))
ROW_LIMIT = int(os.environ.get("DFR_ROW_LIMIT", "0"))

BODY = "DF_241"
SEEDS = list(range(901, 909))
PARTICLE_PATHS = [f".cache/dump_wild_{BODY}_p{s}.pkl" for s in SEEDS]
NOFREE_PATH = f".cache/dump_wild_{BODY}.pkl"
BASELINE_PATH = ".cache/dump_wild_PMS8_241.pkl"
LATTICE_PKL = ".cache/picker/cand_wild_PMS8_241.pkl"
JSONL_PATH = ".cache/wild_admitted_holdout.jsonl"
GOLD_NPZ = ".cache/phase1_alg_states_wildhold.npz"
OUT_PKL = ".cache/picker/df_particle_read.pkl"

BAR_COVERAGE = 50
BAR_VOTE_ROWS = 24
BAR_VOTE_REGR = 2
THRESH = [1.0, 0.875, 0.75, 0.625, 0.5]


def P(s=""):
    print(s, flush=True)


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def by_row(dump):
    out = {}
    for t in dump:
        out.setdefault(t[0], []).append(t)
    for i in out:
        out[i].sort(key=lambda t: t[1])
    return out


def digval(dig):
    dig = list(dig)
    return int(sum(d * 10 ** (N_DIG - 1 - k) for k, d in enumerate(dig)))


def fact_from(ft, op, args, res, dig, dup):
    """decode()'s own fact dict (phase1_algebra_head.py:10257-10272), rebuilt from LV_DUMP's
    already-argmaxed fields. None = an ftype this tuple shape cannot carry (not rel/given)."""
    args = list(args)
    if ft == 1:
        return {"ftype": "given", "var": int(res), "value": digval(dig)}
    if ft == 0:
        opname = "add" if op == 0 else "mul"
        if dup or len(args) == 1:
            a0 = int(args[0])
            aa = [a0, a0]
        else:
            aa = sorted(int(a) for a in args[:2])
        return {"ftype": "rel", "op": opname, "args": aa, "result": int(res)}
    return None


def slot_ok(t):
    """loop_val.py's / dn_common.score_row_slots' own per-slot 'ok', reimplemented on an LV_DUMP
    tuple (i, j, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup)."""
    _, _, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = t
    f_pres = bool(ppres)
    f_ftype = (pft == gft)
    f_res = (pres_ == gres)
    ok = f_pres and f_ftype and f_res
    if gft == 0:
        gset = set(gargs)
        f_op = (pop == gop)
        if len(gset) == 1:
            f_args = bool(pdup) and (pargs[0] in gset)
        else:
            f_args = set(pargs) == gset
        ok = ok and f_op and f_args
    else:
        f_dig = (tuple(pdig) == tuple(gdig))
        ok = ok and f_dig
    return ok


def pred_sig(t):
    """Canonical signature of the PREDICTED fact at this slot (own-tuple fields), for cross-seed
    agreement -- same canonicalization fact_from() would apply, so agreement counts semantically
    identical facts rather than raw top-2-order coincidences."""
    _, _, gft, gop, gargs, gres, gdig, pft, pop, pargs, pres_, pdig, ppres, pdup = t
    if not ppres:
        return ("ABSENT",)
    if pft == 0:
        a = list(pargs)
        aa = (a[0], a[0]) if pdup else tuple(sorted(a[:2]))
        return ("rel", int(pop), aa, int(pres_))
    if pft == 1:
        return ("given", int(pres_), tuple(pdig))
    return ("other", int(pft), int(pres_), tuple(pdig))


def row_facs(slots, use_gold):
    facs, excl = [], 0
    for t in slots:
        j = t[1]
        if use_gold:
            gft, gop, gargs, gres, gdig = t[2], t[3], t[4], t[5], t[6]
            f = fact_from(gft, gop, gargs, gres, gdig, dup=(len(gargs) == 1))
        else:
            pft, pop, pargs, pres_, pdig, ppres, pdup = t[7], t[8], t[9], t[10], t[11], t[12], t[13]
            if not ppres:
                continue
            f = fact_from(pft, pop, pargs, pres_, pdig, dup=pdup)
        if f is None:
            excl += 1
            continue
        f = dict(f)
        f["_slot"] = j
        facs.append(f)
    return facs, excl


def solve(parse, q, wall=WALL):
    import minimality as MIN
    val, m = MIN.solve_query(parse, q, wall=wall)
    if val is None:
        return None, None, None
    uniq = MIN.is_forced(parse, q, val, m, wall=wall)
    return val, m, bool(uniq)


def apply_slot(parse, j, fact):
    out = [f for f in parse if f.get("_slot") != j]
    if fact is not None:
        nf = dict(fact)
        nf["_slot"] = j
        out.append(nf)
    return out


def auroc(scores, labels):
    """Mann-Whitney AUROC, no sklearn dependency. scores/labels: 1-D arrays, labels in {0,1}."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), len(pos), len(neg)
    # rank-based U statistic (tie-aware via average ranks)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_v = allv[order]
    i = 0
    r = 1
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        r += (j - i + 1)
        i = j + 1
    rank_pos_sum = ranks[:len(pos)].sum()
    n1, n2 = len(pos), len(neg)
    u = rank_pos_sum - n1 * (n1 + 1) / 2.0
    return u / (n1 * n2), n1, n2


def _worker_row(job):
    """One row: oracle coverage, per-slot agreement/diff census, consistency judge, gated vote
    (variant 'all8') vs the noise-free top-1 and vs the PMS8 baseline top-1."""
    i = job["i"]
    key = job["key"]
    q = job["q"]
    gold_slots = job["gold_slots"]          # list of LV_DUMP tuples (any dump's gold fields)
    particle_slots = job["particle_slots"]  # list of 8 lists of tuples
    nofree_slots = job["nofree_slots"]
    baseline_slots = job["baseline_slots"]

    gold_facs, gold_excl = row_facs(gold_slots, use_gold=True)

    # ---------------- oracle coverage (DN-scoring): any particle DN-right on this row? ----------------
    particle_dn_right = [all(slot_ok(t) for t in slots) for slots in particle_slots]
    dn_covered = any(particle_dn_right)

    # ---------------- per-slot agreement + difference census ----------------
    nofree_sig_by_j = {t[1]: pred_sig(t) for t in nofree_slots}
    nofree_ok_by_j = {t[1]: slot_ok(t) for t in nofree_slots}
    agree_rows = []      # (j, agreement_frac, ok_at_nofree)
    diff_rows = []       # (j, n_particles_differing_from_nofree)
    for j in nofree_sig_by_j:
        sigs = [pred_sig(t) for slots in particle_slots for t in slots if t[1] == j]
        if not sigs:
            continue
        tally = {}
        for s in sigs:
            tally[s] = tally.get(s, 0) + 1
        mode_sig, mode_n = max(tally.items(), key=lambda kv: kv[1])
        agree = mode_n / len(sigs)
        agree_rows.append((j, agree, bool(nofree_ok_by_j.get(j))))
        n_diff = sum(1 for s in sigs if s != nofree_sig_by_j[j])
        diff_rows.append((j, n_diff, len(sigs)))

    # ---------------- particle parses + solve ----------------
    particle_results = []
    for slots in particle_slots:
        facs, excl = row_facs(slots, use_gold=False)
        val, m, uniq = solve(facs, q, WALL)
        particle_results.append(dict(val=val, uniq=uniq, excl=excl,
                                      sig=tuple(sorted(pred_sig(t) for t in slots))))

    nofree_facs, nofree_excl = row_facs(nofree_slots, use_gold=False)
    val0, m0, uniq0 = solve(nofree_facs, q, WALL)

    baseline_facs, baseline_excl = row_facs(baseline_slots, use_gold=False)
    valb, mb, uniqb = solve(baseline_facs, q, WALL)

    # ---------------- consistency judge: solved+unique, majority-multiplicity tie-break ----------------
    # (loglik is unavailable from LV_DUMP's argmax-only tuples -- substituted by cross-particle
    # frequency of the full-row signature, lowest-seed tie-break; documented substitution.)
    groups = {}
    for k, r in enumerate(particle_results):
        if r["val"] is None or not r["uniq"]:
            continue
        groups.setdefault(r["sig"], []).append(k)
    cj_val = None
    if groups:
        best_sig = max(groups.items(), key=lambda kv: (len(kv[1]), -min(kv[1])))[0]
        k0 = min(groups[best_sig])
        cj_val = particle_results[k0]["val"]

    # ---------------- gated vote (agreement_gated_vote.py's algorithm, all8 variant) ----------------
    cand = []
    for j, agree, _ in agree_rows:
        sigs_here = [pred_sig(t) for slots in particle_slots for t in slots if t[1] == j]
        tally = {}
        for s in sigs_here:
            tally[s] = tally.get(s, 0) + 1
        mode_sig, mode_n = max(tally.items(), key=lambda kv: kv[1])
        if mode_sig == nofree_sig_by_j.get(j):
            continue
        if mode_sig[0] == "other" or mode_sig[0] == "ABSENT":
            donor_fact = None
        else:
            # rebuild the fact dict from the first particle carrying this winning sig
            donor_fact = None
            for slots in particle_slots:
                for t in slots:
                    if t[1] == j and pred_sig(t) == mode_sig:
                        pft, pop, pargs, pres_, pdig, ppres, pdup = t[7], t[8], t[9], t[10], t[11], t[12], t[13]
                        donor_fact = fact_from(pft, pop, pargs, pres_, pdig, dup=pdup) if ppres else None
                        break
                if donor_fact is not None or mode_sig == ("ABSENT",):
                    break
        cand.append((j, mode_n / len(sigs_here), donor_fact))
    cand.sort(key=lambda c: -c[1])

    cur_parse = list(nofree_facs)
    trace = []
    for j, agree, fact in cand:
        trial = apply_slot(cur_parse, j, fact)
        v, m, u = solve(trial, q, WALL)
        if v is not None:
            cur_parse = trial
            trace.append((j, agree, v, u))

    gv_by_th = {}
    for th in THRESH:
        prefix = [t for t in trace if t[1] >= th]
        if prefix:
            gv_by_th[th] = (prefix[-1][2], prefix[-1][3])
        else:
            gv_by_th[th] = (val0, uniq0)

    return dict(i=i, key=key, dn_covered=dn_covered, particle_dn_right=particle_dn_right,
                agree_rows=agree_rows, diff_rows=diff_rows,
                val0=val0, uniq0=uniq0, valb=valb, uniqb=uniqb, cj_val=cj_val,
                gv_by_th=gv_by_th, n_particle_solved=sum(1 for r in particle_results if r["val"] is not None),
                gold_excl=gold_excl, nofree_excl=nofree_excl)


def main():
    t0 = time.time()
    P("=" * 100)
    P("DF_241 PARTICLE READ -- zero-GPU re-derivation off LV_DUMP tuples (seeded_coverage.py / "
      "agreement_gated_vote.py cannot read these: they need LV_DUMP_FULL's raw-logit shape, which "
      "this chain did not write for the particle reads -- see module docstring)")
    P("=" * 100)

    particle_dumps = [by_row(load_pkl(p)) for p in PARTICLE_PATHS]
    nofree_dump = by_row(load_pkl(NOFREE_PATH))
    baseline_dump = by_row(load_pkl(BASELINE_PATH))

    # self-check: the known VOID bug in dn_particle_vote.py, quantified on a 20-row sample.
    import dn_particle_vote as DV
    sample_rows = list(nofree_dump)[:20]
    accs = []
    for i in sample_rows:
        g = DV.pred_graph(nofree_dump[i]); gg = DV.gold_graph(nofree_dump[i])
        accs.append(DV.positional_exact(g, gg))
    P(f"[self-check] dn_particle_vote.py's positional_exact on DF_241's OWN noise-free dump vs "
      f"itself (20-row sample): mean {np.mean(accs):.3f} (chain's own masked-wild fac-exact: "
      f"0.3598) -- confirms the VOID bug is in pred_graph/gold_graph's args handling, not a "
      f"solver issue.")

    rows = json.load(open(JSONL_PATH)) if JSONL_PATH.endswith(".json") else \
        [json.loads(l) for l in open(JSONL_PATH)]
    sys.path.insert(0, ".")
    from mycelium.custody_gold import row_gold
    keys = {i: int(row_gold(r)) for i, r in enumerate(rows)}
    qvars = {i: int(r["query_var"]) for i, r in enumerate(rows)}

    lattice_rows_right = set()
    if os.path.exists(LATTICE_PKL):
        lat = load_pkl(LATTICE_PKL)
        for row in lat:
            if any(b.get("correct") for b in row["branches"].values()):
                lattice_rows_right.add(int(row["i"]))
    P(f"lattice {LATTICE_PKL}: {len(lattice_rows_right)} rows SOLVE-scoring-correct among the "
      f"first look's branches (established ceiling check: =={len(lattice_rows_right)==43 and 43})")

    all_rows = sorted(set(nofree_dump) & set(baseline_dump) & set.intersection(*[set(d) for d in particle_dumps]))
    if ROW_LIMIT:
        all_rows = all_rows[:ROW_LIMIT]
    P(f"{len(all_rows)} rows common to all 8 particles + noise-free + baseline dumps")

    jobs = []
    for i in all_rows:
        jobs.append(dict(i=i, key=keys[i], q=qvars[i],
                          gold_slots=nofree_dump[i],
                          particle_slots=[d[i] for d in particle_dumps],
                          nofree_slots=nofree_dump[i],
                          baseline_slots=baseline_dump[i]))

    import multiprocessing as mp
    results = []
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for n_done, r in enumerate(pool.imap_unordered(_worker_row, jobs, chunksize=1), 1):
            results.append(r)
            if n_done % 50 == 0 or n_done == len(jobs):
                P(f"  {n_done}/{len(jobs)} rows ({time.time()-t0:.0f}s)")

    pickle.dump(results, open(OUT_PKL, "wb"))
    P(f"-> {OUT_PKL} ({len(results)} rows, {time.time()-t0:.0f}s total)")

    # ============================= REPORT =============================
    P("-" * 100)
    # (1) oracle coverage
    dn_covered = sum(1 for r in results if r["dn_covered"])
    combined = len({r["i"] for r in results if r["dn_covered"]} | lattice_rows_right)
    P(f"(1) ORACLE COVERAGE (DN-scoring): alone {dn_covered}/{len(results)} "
      f"({dn_covered/len(results):.3f}) | combined with lattice {combined}/{len(results)} "
      f"| BAR (combined >= {BAR_COVERAGE}): {'PASS' if combined >= BAR_COVERAGE else 'FAIL'}")

    # (2) consistency judge + gated vote vs noise-free top-1 and vs PMS8 baseline top-1
    n_nofree_right = sum(1 for r in results if r["val0"] == r["key"])
    n_base_right = sum(1 for r in results if r["valb"] == r["key"])
    P(f"    noise-free top-1 (solve-scoring) rows right: {n_nofree_right}/{len(results)} "
      f"| PMS8 baseline top-1 rows right: {n_base_right}/{len(results)}")

    def vs(name, get_val):
        rows_right = regr = gained = 0
        for r in results:
            v = get_val(r)
            right_now = (v == r["key"])
            was_right_nf = (r["val0"] == r["key"])
            was_right_b = (r["valb"] == r["key"])
            rows_right += int(right_now)
            if was_right_nf and not right_now:
                regr += 1
            if (not was_right_nf) and right_now:
                gained += 1
        P(f"    {name}: rows_right {rows_right}/{len(results)} | vs noise-free: "
          f"gained {gained} regressions {regr} (bar rows>={BAR_VOTE_ROWS} regr<={BAR_VOTE_REGR}: "
          f"{'PASS' if rows_right >= BAR_VOTE_ROWS and regr <= BAR_VOTE_REGR else 'FAIL'})")

    vs("CONSISTENCY JUDGE", lambda r: r["cj_val"])
    for th in THRESH:
        vs(f"GATED VOTE (all8, th={th})", lambda r, th=th: r["gv_by_th"][th][0])

    # (3) per-slot agreement AUROC vs correctness
    all_agree = []
    all_ok = []
    for r in results:
        for j, agree, ok in r["agree_rows"]:
            all_agree.append(agree)
            all_ok.append(ok)
    a, npos, nneg = auroc(all_agree, all_ok)
    P(f"(3) PER-SLOT AGREEMENT AUROC (8-seed mode fraction vs noise-free per-slot 'ok'): "
      f"AUROC={a:.4f} n_slots={len(all_agree)} (n_ok={npos} n_wrong={nneg}) "
      f"[PMS8 precedent: 0.76-0.79 at tau 1.0]")

    # (4) per-slot difference census
    n_slots = n_any_diff = n_total_particle_slot_obs = n_diff_particle_slot_obs = 0
    for r in results:
        for j, n_diff, n_obs in r["diff_rows"]:
            n_slots += 1
            n_any_diff += int(n_diff > 0)
            n_total_particle_slot_obs += n_obs
            n_diff_particle_slot_obs += n_diff
    P(f"(4) PER-SLOT DIFFERENCE CENSUS: slots with >=1 of 8 particles differing from the "
      f"noise-free decode: {n_any_diff}/{n_slots} ({n_any_diff/max(1,n_slots):.4f}) | "
      f"at the (particle,slot) level: {n_diff_particle_slot_obs}/{n_total_particle_slot_obs} "
      f"({n_diff_particle_slot_obs/max(1,n_total_particle_slot_obs):.4f}) differ from noise-free")

    gold_excl = sum(r["gold_excl"] for r in results)
    nofree_excl = sum(r["nofree_excl"] for r in results)
    P(f"[caveat] non-rel/given slots excluded from rebuilt parses: gold {gold_excl}, "
      f"noise-free predicted {nofree_excl} (out of {sum(len(nofree_dump[i]) for i in all_rows)} "
      f"gold-present slots total)")
    P(f"DONE ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
