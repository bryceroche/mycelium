"""dn_read_b.py -- READ B: FLAGGER LOCALIZATION (ledger 2026-10-12 12:35, denoising-breath reads).

Zero-GPU, banked artifacts + cheap new CPU solves (bounded, walled). Universe: PMS8_241's own
top-1 decoded parse per wild row, from .cache/picker/cand_wild_PMS8_241.pkl /
.cache/picker/minimality_wild_PMS8_241.pkl (both already banked, 2026-10-05/11; the minimality
pkl is a superset of the cand pkl with 'minimality' certs added on solved+unique top-1 branches
-- no recompute needed for the DEAD FACTS flagger). Gold wrongness per slot comes from
score_row_slots (dn_common.py, loop_val.py's own "ok" convention) against the FIXED-tap final
breath of .cache/rawslots_breaths_wild_PMS8_241.pkl, restricted to the slots the model's own
top-1 parse actually carries (a slot gold says present but the model dropped cannot be flagged by
any of these three solver-side flaggers by construction -- a stated blind spot, not fixed here).

Three flaggers (all slot-granular, via parse[idx]['_slot']):
  UNSAT  -- top1 status == 'unsat': mycelium's alternator_bridge.refuse_and_core (banked machinery,
            not new) gives a minimal unsatisfiable core over the parse's own factor indices.
  DEAD   -- top1 status == 'solved' and unique == True: minimality_wild_PMS8_241.pkl's own
            'minimality' cert's 'dead' list (ALREADY COMPUTED, 2026-10-11 run; read, not rebuilt).
  FREE   -- top1 status == 'solved' and unique == False: one bounded re-solve banning the found
            value at q (certify_unique's own mechanism, extended to keep the alternate assignment)
            -- variables that differ between the two solutions are 'free'; a slot is flagged if it
            references a free variable (var/args/result depending on ftype).
  (skipped) VIEW DISAGREEMENT: no banked per-row/slot array exists for the 5-view TTA census on
  PMS8_241 (hammerhead_census.py prints a console census only, no npz/pkl) -- re-running the 5
  views costs ~4922s wall (2026-10-06 23:43 ledger entry) for the full 311 rows, which is a fresh
  heavy CPU pass, not a banked-artifact read; skipped and stated per the task's own allowance.

Output: .cache/dnread_read_b_PMS8_241.json + .txt
"""
import os
import sys
import json
import pickle
import signal
import collections

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/picker")
import numpy as np
from dn_common import load_gold, score_row_slots, index_breaths

CAND = ".cache/picker/minimality_wild_PMS8_241.pkl"   # superset of cand_wild_PMS8_241.pkl (+ minimality certs)
RAW_WILD = ".cache/rawslots_breaths_wild_PMS8_241.pkl"
GOLD_WILD = ".cache/phase1_alg_states_wildhold.npz"
OUT_JSON = ".cache/dnread_read_b_PMS8_241.json"
OUT_TXT = ".cache/dnread_read_b_PMS8_241.txt"

WALL_UNSAT = float(os.environ.get("DNB_WALL_UNSAT", "8"))
WALL_FREE = float(os.environ.get("DNB_WALL_FREE", "5"))
WORKERS = int(os.environ.get("DNB_WORKERS", "6"))


class _T(Exception):
    pass


def _alarm(signum, frame):
    raise _T()


def _unsat_core_task(args):
    i, parse, q, m = args
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    old = signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, WALL_UNSAT)
    try:
        from alternator_bridge import refuse_and_core
        from minimality import _build_gv_nvv
        _, nvv = _build_gv_nvv(parse, q)
        r = refuse_and_core(nvv, parse, m, budget=2000, seed=0)
        return i, r.get("status"), r.get("core", [])
    except _T:
        return i, "timeout", []
    except Exception as e:
        return i, f"error:{type(e).__name__}", []
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)


def _freevar_task(args):
    i, parse, q, value, m, orig_asg = args
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    old = signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, WALL_FREE)
    try:
        from minimality import _problem
        from mycelium.csp_core import solve_symbolic
        prob = _problem(parse, q, m)
        prob.domains0[q].discard(value)
        if not prob.domains0[q]:
            return i, "degenerate", []
        r2 = solve_symbolic(prob, budget=5000, seed=1)
        if r2.get("status") != "solved":
            return i, r2.get("status", "?"), []
        alt = list(r2["assignment"])
        n = min(len(alt), len(orig_asg))
        free_vars = [v for v in range(n) if int(alt[v]) != int(orig_asg[v])]
        return i, "ok", free_vars
    except _T:
        return i, "timeout", []
    except Exception as e:
        return i, f"error:{type(e).__name__}", []
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)


def _pool_map(fn, tasks, workers, wall):
    import multiprocessing as mp
    if not tasks:
        return []
    res = []
    with mp.get_context("spawn").Pool(workers) as pool:
        it = pool.imap_unordered(fn, tasks, chunksize=1)
        for _ in range(len(tasks)):
            try:
                res.append(it.next(timeout=wall * 20 + 60))
            except mp.TimeoutError:
                break
    return res


def slot_vars(fact):
    """Every variable index a parse fact (post-decode, carries _slot) references."""
    ft = fact["ftype"]
    vs = []
    if ft == "given":
        vs.append(fact.get("var"))
    elif ft == "rel":
        vs += list(fact.get("args", [])) + [fact.get("result")]
    elif ft in ("mod", "fdiv"):
        vs += [fact.get("var"), fact.get("result")]
    elif ft == "sel":
        vs += list(fact.get("args", [])) + [fact.get("result")]
    elif ft == "pct":
        vs += list(fact.get("args", []))
    return [v for v in vs if v is not None]


def main():
    print("[dn-read-b] loading cand/minimality + raw breath-F dump + gold...", flush=True)
    rows = pickle.load(open(CAND, "rb"))
    rb = pickle.load(open(RAW_WILD, "rb"))
    kbF = max(rb.keys())
    idxF = index_breaths(rb)[kbF]
    vg = load_gold(GOLD_WILD)

    # the gold-wrong universe: every slot the model's OWN top-1 parse carries
    universe = {}   # (i, j) -> wrong (bool)
    status_count = collections.Counter()
    unsat_tasks, free_tasks, dead_hits = [], [], {}
    row_meta = {}
    for row in rows:
        i = row["i"]; q = row["q"]
        b = row["branches"].get(row.get("top1_sig")) if row.get("top1_sig") else None
        if b is None:
            continue
        status = b.get("status")
        status_count[status] += 1
        parse = b["parse"]
        slots_i = score_row_slots(idxF[i], vg, i) if i in idxF else {}
        for f in parse:
            j = f["_slot"]
            if j in slots_i:
                wrong = not slots_i[j]["ok"]
            else:
                wrong = True   # model present, gold absent on this slot j -- a hallucination, wrong by definition
            universe[(i, j)] = wrong
        row_meta[i] = dict(status=status, q=q, m=b.get("m"), value=b.get("value"),
                           unique=b.get("unique"), parse=parse,
                           assignment=b.get("assignment"), minimality=b.get("minimality"))
        if status == "unsat":
            unsat_tasks.append((i, parse, q, b.get("m") or 300))
        elif status == "solved" and b.get("unique") is True and b.get("minimality"):
            for idx_, ft_ in b["minimality"].get("dead", []):
                dead_hits.setdefault(i, []).append(parse[idx_]["_slot"])
        elif status == "solved" and b.get("unique") is False:
            free_tasks.append((i, parse, q, b.get("value"), b.get("m"), b.get("assignment")))

    print(f"[dn-read-b] top1 status counts: {dict(status_count)}; universe slots {len(universe)}", flush=True)
    print(f"[dn-read-b] unsat-core tasks {len(unsat_tasks)}, dead-fact rows (precomputed) {len(dead_hits)}, "
          f"free-var tasks {len(free_tasks)}", flush=True)

    unsat_hits = {}
    for i, status, core in _pool_map(_unsat_core_task, unsat_tasks, WORKERS, WALL_UNSAT):
        if status == "unsat" and core:
            parse = row_meta[i]["parse"]
            unsat_hits[i] = [parse[idx_]["_slot"] for idx_ in core]
    print(f"[dn-read-b] unsat-core: {len(unsat_hits)}/{len(unsat_tasks)} rows produced a non-empty core", flush=True)

    free_hits = {}
    for i, status, free_vars in _pool_map(_freevar_task, free_tasks, WORKERS, WALL_FREE):
        if status == "ok" and free_vars:
            parse = row_meta[i]["parse"]
            fv = set(free_vars)
            flagged = [f["_slot"] for f in parse if set(slot_vars(f)) & fv]
            if flagged:
                free_hits[i] = flagged
    print(f"[dn-read-b] free-var: {len(free_hits)}/{len(free_tasks)} rows produced a non-empty flag set", flush=True)

    n_dead_applicable = sum(1 for row in rows for b in [row["branches"].get(row.get("top1_sig"))]
                            if b and b.get("status") == "solved" and b.get("unique") is True)
    flaggers = {"unsat_core": (unsat_hits, len(unsat_tasks)),
                "dead_facts": (dead_hits, n_dead_applicable),
                "free_vars": (free_hits, len(free_tasks))}

    n_wrong_total = sum(1 for v in universe.values() if v)
    n_slots_total = len(universe)
    print(f"[dn-read-b] universe: {n_slots_total} slots in top-1 parses, {n_wrong_total} wrong "
          f"({n_wrong_total/max(n_slots_total,1):.3f})", flush=True)

    report = {"n_slots_universe": n_slots_total, "n_wrong_universe": n_wrong_total,
              "status_counts": dict(status_count), "flaggers": {}}
    for name, (hits, n_applicable_rows) in flaggers.items():
        flagged_pairs = set((i, j) for i, js in hits.items() for j in js)
        tp = sum(1 for p in flagged_pairs if universe.get(p))
        fp = sum(1 for p in flagged_pairs if p in universe and not universe[p])
        fn = n_wrong_total - tp
        precision = tp / max(len(flagged_pairs), 1)
        recall = tp / max(n_wrong_total, 1)
        n_rows_fired = len(hits)
        report["flaggers"][name] = dict(
            n_applicable_rows=n_applicable_rows, n_rows_fired=n_rows_fired,
            n_slots_flagged=len(flagged_pairs), tp=tp, fp=fp, fn=fn,
            precision=precision, recall=recall,
            coverage_rows=n_rows_fired / max(n_applicable_rows, 1))

    # THE CEILING (per the task): recall x fix_rate_given_flag x wrong_slots - false_flags x tabu_break_rate
    FIX_RATES = [0.44, 1.0]
    BREAK_RATES = [1.0, 0.5]
    WILD_SLOTS = 2051   # the task's own frame (total gold slots on wild, CLAUDE.md / ledger convention)
    rescale = WILD_SLOTS / max(n_slots_total, 1)
    ceiling = {}
    for name, f in report["flaggers"].items():
        rows_ = {}
        for fr in FIX_RATES:
            for br in BREAK_RATES:
                gained = f["recall"] * fr * n_wrong_total
                lost = f["fp"] * br
                net = gained - lost
                net_rescaled = net * rescale          # net slots, rescaled from the 2015-slot parse-only
                                                        # universe to the task's 2051-wild-slot frame
                rows_[f"fix{fr}_break{br}"] = dict(gained=gained, lost=lost, net=net,
                                                    net_slots_per_2051=net_rescaled,
                                                    masked_wild_delta=net_rescaled / WILD_SLOTS)
        ceiling[name] = rows_
    report["ceiling"] = ceiling
    report["n_slots_universe_note"] = ("universe is the model's own top-1 PARSE slots on wild "
                                        f"({n_slots_total}, not the full 2051 gold slots); the ceiling table's "
                                        "net_per_2051 column rescales to the task's 2051-wild-slot frame "
                                        "for comparability, assuming the same wrong-rate holds off-parse")
    report["skipped"] = {"view_disagreement": "no banked per-row/slot array for the 5-view TTA census on "
                          "PMS8_241 exists (hammerhead_census.py prints console-only); a fresh 5-view pass "
                          "costs ~4922s wall (2026-10-06 23:43 ledger) -- skipped, not a banked-artifact read"}
    json.dump(report, open(OUT_JSON, "w"), indent=1)

    lines = []
    lines.append(f"READ B -- FLAGGER LOCALIZATION (PMS8_241, wildhold, top-1 decode, breath {kbF})")
    lines.append(f"universe: {n_slots_total} top-1-parse slots, {n_wrong_total} wrong ({n_wrong_total/max(n_slots_total,1):.3f})")
    lines.append(f"top1 status counts: {dict(status_count)}")
    lines.append("")
    lines.append(f"{'flagger':12s} {'applic':>7s} {'fired':>6s} {'cov':>6s} {'flagged':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'prec':>6s} {'recall':>6s}")
    for name, f in report["flaggers"].items():
        lines.append(f"{name:12s} {f['n_applicable_rows']:>7d} {f['n_rows_fired']:>6d} {f['coverage_rows']:>6.3f} "
                      f"{f['n_slots_flagged']:>8d} {f['tp']:>5d} {f['fp']:>5d} {f['fn']:>5d} {f['precision']:>6.3f} {f['recall']:>6.3f}")
    lines.append("")
    lines.append(f"CEILING (gained = recall x fix_rate x wrong_total; lost = FP x break_rate; rescaled {n_slots_total}->2051 slots):")
    for name, rows_ in ceiling.items():
        for cond, v in rows_.items():
            lines.append(f"  {name:12s} {cond:16s} gained={v['gained']:6.1f} lost={v['lost']:6.1f} net={v['net']:+7.1f}  "
                          f"net_slots/2051={v['net_slots_per_2051']:+7.1f}  implied masked-wild delta={v['masked_wild_delta']:+.4f}")
    lines.append("")
    lines.append("SKIPPED: " + report["skipped"]["view_disagreement"])
    txt = "\n".join(lines)
    open(OUT_TXT, "w").write(txt + "\n")
    print(txt)
    print(f"\n[dn-read-b] wrote {OUT_JSON} and {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
