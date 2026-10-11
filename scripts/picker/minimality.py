"""minimality.py -- THE MINIMALITY CERTIFICATE (for scripts/picker/; zero-GPU, key-free).

THE CLAIM. "The solver cannot refute a twin because both readings are consistent" holds only for
SAT and UNIQUE. It does not hold for MINIMAL. A word problem's gold graph is (nearly always) a
minimal determining set for its query: every number and every relation is needed to force the
answer. A stale-twin binding (a relation, or the query, reading the same entity at an EARLIER time)
skips a link of the state chain, and the skipped link becomes dead weight: drop it and the query
is still forced to the same value. The solver can test this with the door it already has
(mycelium.doors.certify_unique): relax one fact, ban the candidate's own answer, ask for UNSAT.

THE DUAL OF THE MUC. The minimal unsatisfiable core localizes a refusal. A dead fact localizes a
consistent-but-wrong parse, and it is exactly the class the consistency judge waves through.

WHAT IT CANNOT SEE (the honest half). A role / direction error (the res pointer's forward default
on an inverse form) or an op or value error leaves every fact needed. Those stay unrefutable by
structure. So the twin problem splits into a refutable half (time: skipped links) and an
unrefutable half (role: direction), and the read below measures the split.

SOUNDNESS. Budget exhaustion and walls make a fact look NEEDED (certify_unique refuses), never
dead, so search limits cannot manufacture a refutation. The only false refutation is a gold graph
that genuinely carries dead weight (a distractor number, an unused intermediate): read R1 prices it.

THE READ (pre-registered; the rule is fixed here, no parameters, so no train or calibrate pool):
  R1  gold minimal rate on a NON-wild pool first (MIN_GOLD = the diet slice's fixture), then wild.
      >= 0.95 -> a hard filter; 0.85-0.95 -> an angle only; < 0.85 -> the prior is wrong, stop.
  R2  top-1 decodes that are solved + unique: the non-minimal share among WRONG ones (the catch) vs
      among RIGHT ones (the false refutations). Prediction: catch >= 0.25, false <= 0.05.
  R3  THE MINIMAL-FIRST JUDGE (solved + unique + minimal, loglik tie-break) vs the consistency judge
      (solved + unique, loglik tie-break; parity line: 18 / 311 on PMS8_241 wild). Bar: rows >= 24
      with regressions <= 2 (the picker's own bar). Kill: R2's catch < 0.10.
  Diagnostic only, never an input: the catch split by twin_gap's own labels (twin vs non-twin).

Usage:
  python3 minimality.py --selftest                       # synthetic rows through the real solver
  MIN_CANDS=.cache/picker/cand_wild_PMS8_241.pkl [MIN_GOLD=<fixture .jsonl or .pkl>] \
    MIN_OUT=.cache/picker/minimality_wild_PMS8_241.pkl MIN_WORKERS=8 MIN_WALL=3 \
    .venv/bin/python3 scripts/picker/minimality.py
MIN_CANDS is gen_candidates.py's output: [{i, text, key, q, top1_sig, branches: {sig: {parse, status,
value, m, unique, loglik, correct}}}]. A gold fixture row needs 'factors' and 'query_var' (or 'query').
"""
import os
import sys
import json
import time
import pickle

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

UNIQ_BUDGET = 5000          # the mint gate's uniqueness budget (CLAUDE.md Sec.1; gen_candidates' own)
GIVEN_LIKE = ("given",)


def _build_gv_nvv(parse, q):
    """beam_oracle.build_gv_nvv, copied (5 lines) so this module never imports the head."""
    used = [f.get("var") for f in parse if f["ftype"] == "given"] + \
           [a for f in parse if f["ftype"] == "rel" for a in list(f["args"]) + [f["result"]]]
    used += [v for f in parse if f["ftype"] in ("mod", "fdiv") for v in (f["var"], f["result"])]
    used += [v for f in parse if f["ftype"] == "sel" for v in list(f["args"]) + [f["result"]]]
    used += [v for f in parse if f["ftype"] == "pct" for v in f["args"]]
    nvv = max([q + 1] + [int(v) + 1 for v in used if v is not None])
    gv = {f["var"]: f["value"] for f in parse if f["ftype"] == "given"}
    return gv, nvv


def _problem(parse, q, m):
    from mycelium.csp_domains import problem_from_algebra3
    gv, nvv = _build_gv_nvv(parse, q)
    return problem_from_algebra3(nvv, parse, gv, m)


def _walled(fn, wall):
    import signal

    class _T(Exception):
        pass

    def _alarm(signum, frame):
        raise _T()
    old = signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, wall)
    try:
        return fn()
    except _T:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)


def solve_query(parse, q, m=None, wall=3.0):
    """Key-free solve for rows that carry no precomputed value (gold rows): the row-sized rung, then 10^4."""
    from mycelium.csp_core import solve_symbolic
    gmax = max([int(f["value"]) for f in parse if f["ftype"] == "given"] + [1])
    for mm in ([m] if m else [int(min(10001, max(300, 2 * gmax))), 10000]):
        r = _walled(lambda: solve_symbolic(_problem(parse, q, mm), budget=UNIQ_BUDGET), wall)
        if r is not None and r.get("status") == "solved":
            return int(r["assignment"][q]), mm
    return None, None


def is_forced(parse, q, value, m, wall=3.0):
    """True iff q is forced to `value` (the uniqueness door: ban it, demand UNSAT). None on a wall."""
    from mycelium.doors import certify_unique
    return _walled(lambda: bool(certify_unique(_problem(parse, q, m), q, value, budget=UNIQ_BUDGET)), wall)


def minimality(parse, q, value, m, wall=3.0):
    """Per fact: needed (dropping it un-forces the query) or DEAD (the query stays forced to the
    same value without it). Returns dict(n_facts, n_dead, dead=[(index, ftype)], minimal, n_wall)."""
    dead, n_wall = [], 0
    for i, f in enumerate(parse):
        relaxed = parse[:i] + parse[i + 1:]
        if not relaxed:
            continue
        forced = is_forced(relaxed, q, value, m, wall)
        if forced is None:
            n_wall += 1                        # a wall counts as NEEDED (never a refutation)
        elif forced:
            dead.append((i, f["ftype"]))
    return dict(n_facts=len(parse), n_dead=len(dead), dead=dead, minimal=(len(dead) == 0), n_wall=n_wall)


# ----------------------------------------------------------------------------------------------- the read
def _cand_task(t):
    ri, sig, parse, q, value, m, wall = t
    try:
        return ri, sig, minimality(parse, q, value, m, wall)
    except Exception as e:                     # an unbuildable fact type: no certificate, never a refutation
        return ri, sig, dict(n_facts=len(parse), n_dead=0, dead=[], minimal=True, n_wall=0,
                             error=f"{type(e).__name__}: {e}")


def _gold_task(t):
    ri, parse, q, wall = t
    try:
        value, m = solve_query(parse, q, wall=wall)
        if value is None:
            return ri, None
        return ri, dict(value=value, m=m, **minimality(parse, q, value, m, wall))
    except Exception:
        return ri, None


def _load_gold(path):
    if path.endswith(".pkl"):
        rows = pickle.load(open(path, "rb"))
    else:
        rows = [json.loads(line) for line in open(path) if line.strip()]
    out = []
    for r in rows:
        q = r.get("query_var", r.get("query"))
        out.append((r["factors"], None if q is None else int(q)))
    return out


def _pool_map(fn, tasks, workers, wall):
    import multiprocessing as mp
    res = []
    with mp.get_context("spawn").Pool(workers) as pool:
        it = pool.imap_unordered(fn, tasks, chunksize=1)    # chunksize 1: the iterator keeps .next(timeout)
        for _ in range(len(tasks)):
            try:
                res.append(it.next(timeout=wall * 40 + 60))
            except mp.TimeoutError:
                break
    return res


def run_read():
    cands_path = os.environ["MIN_CANDS"]
    gold_path = os.environ.get("MIN_GOLD", "")
    out_path = os.environ.get("MIN_OUT", "")
    workers = int(os.environ.get("MIN_WORKERS", "8"))
    wall = float(os.environ.get("MIN_WALL", "3"))
    t0 = time.time()

    if gold_path:
        gold = _load_gold(gold_path)
        res = _pool_map(_gold_task, [(i, f, q, wall) for i, (f, q) in enumerate(gold) if q is not None],
                        workers, wall)
        ok = [r for _, r in res if r is not None]
        mins = [r["minimal"] for r in ok]
        kinds = {}
        for r in ok:
            for _, ft in r["dead"]:
                kinds[ft] = kinds.get(ft, 0) + 1
        print(f"[minimality] R1 GOLD ({gold_path}): minimal {sum(mins)}/{len(ok)} = "
              f"{np.mean(mins) if ok else float('nan'):.3f}  (unsolved {len(res) - len(ok)}; "
              f"dead facts by ftype {kinds})  -> hard filter if >= 0.95, angle if 0.85-0.95, stop if < 0.85",
              flush=True)

    rows = pickle.load(open(cands_path, "rb"))
    tasks = []
    for ri, row in enumerate(rows):
        for sig, b in row["branches"].items():
            if b.get("status") == "solved" and b.get("unique") is True and b.get("parse"):
                tasks.append((ri, sig, b["parse"], row["q"], int(b["value"]), int(b["m"]), wall))
    print(f"[minimality] {len(rows)} rows, {len(tasks)} solved+unique candidates -> certificates "
          f"({workers} workers, wall {wall}s)", flush=True)
    for ri, sig, cert in _pool_map(_cand_task, tasks, workers, wall):
        rows[ri]["branches"][sig]["minimality"] = cert
    print(f"[minimality] certificates done ({time.time() - t0:.0f}s)", flush=True)

    # R2: the top-1 decode, solved + unique
    right, wrong = [], []
    for row in rows:
        b = row["branches"].get(row.get("top1_sig")) if row.get("top1_sig") else None
        if b is None or "minimality" not in b:
            continue
        (right if b.get("correct") else wrong).append(not b["minimality"]["minimal"])
    print(f"[minimality] R2 TOP-1 solved+unique: non-minimal among WRONG {sum(wrong)}/{len(wrong)} = "
          f"{np.mean(wrong) if wrong else float('nan'):.3f} (the catch; pred >= 0.25, kill < 0.10) | "
          f"among RIGHT {sum(right)}/{len(right)} = {np.mean(right) if right else float('nan'):.3f} "
          f"(false refutations; pred <= 0.05)", flush=True)

    # R3: the judges
    def judge(row, minimal_first):
        pool = [b for b in row["branches"].values() if b.get("status") == "solved" and b.get("unique") is True]
        if minimal_first:
            mp_ = [b for b in pool if b.get("minimality", {}).get("minimal")]
            pool = mp_ or pool
        if not pool:
            return row["branches"].get(row.get("top1_sig")) if row.get("top1_sig") else None
        return max(pool, key=lambda b: b.get("loglik", float("-inf")))

    n_c = n_m = fixes = regs = 0
    for row in rows:
        bc, bm = judge(row, False), judge(row, True)
        c = bool(bc and bc.get("correct")); mm = bool(bm and bm.get("correct"))
        n_c += c; n_m += mm; fixes += (mm and not c); regs += (c and not mm)
    print(f"[minimality] R3 JUDGES: consistency {n_c}/{len(rows)} (parity: 18/311 on PMS8_241 wild) | "
          f"MINIMAL-FIRST {n_m}/{len(rows)} (+{fixes} fixes / -{regs} regressions vs consistency) "
          f"-> bar rows >= 24, regressions <= 2", flush=True)
    if out_path:
        pickle.dump(rows, open(out_path, "wb"))
        print(f"[minimality] wrote {out_path}", flush=True)


# ----------------------------------------------------------------------------------------------- selftest
def _g(v, x):
    return {"ftype": "given", "var": v, "value": x}


def _r(op, a, b, res):
    return {"ftype": "rel", "op": op, "args": [a, b], "result": res}


def selftest():
    """'Tom had 12 marbles. He lost 5. Then he found 3. How many does Tom have now?'  Answer 10.
    v0 Tom@t0=12, v1 lost=5, v2 found=3, v3 Tom@t1, v4 Tom@t2. The dialect's own convention:
    'lost' is a sub re-encoded as add with the NEW variable among the args (Tom@t1 + lost = Tom@t0)."""
    G = [_g(0, 12), _g(1, 5), _g(2, 3)]
    cases = [
        ("gold", G + [_r("add", 3, 1, 0), _r("add", 3, 2, 4)], 4),
        ("TIME twin: 'found' reads Tom@t0", G + [_r("add", 3, 1, 0), _r("add", 0, 2, 4)], 4),
        ("TIME twin: query reads Tom@t1", G + [_r("add", 3, 1, 0), _r("add", 3, 2, 4)], 3),
        ("TIME twin: query reads Tom@t0", G + [_r("add", 3, 1, 0), _r("add", 3, 2, 4)], 0),
        ("ARGS: 'found' reads 'lost' (other sentence)", G + [_r("add", 3, 1, 0), _r("add", 3, 1, 4)], 4),
        ("ROLE: res defaults forward", G + [_r("add", 0, 1, 3), _r("add", 3, 2, 4)], 4),
        ("VALUE: lost read as 6", [_g(0, 12), _g(1, 6), _g(2, 3), _r("add", 3, 1, 0), _r("add", 3, 2, 4)], 4),
        ("OP: found as mul", G + [_r("add", 3, 1, 0), _r("mul", 3, 2, 4)], 4),
        ("gold + a distractor ('Tom is 9')", G + [_g(5, 9), _r("add", 3, 1, 0), _r("add", 3, 2, 4)], 4),
        ("two unknowns (x+y=11, x=y+3)", [_g(0, 11), _g(1, 3), _r("add", 2, 3, 0), _r("add", 3, 1, 2)], 2),
    ]
    print(f"{'case':44s} {'answer':>6s}  {'minimal':>7s}  dead facts")
    for name, parse, q in cases:
        value, m = solve_query(parse, q)
        if value is None:
            print(f"{name:44s} {'unsat':>6s}")
            continue
        uniq = is_forced(parse, q, value, m)
        cert = minimality(parse, q, value, m)
        dead = ", ".join(f"#{i} {ft}" for i, ft in cert["dead"]) or "-"
        print(f"{name:44s} {value:>6d}  {('yes' if cert['minimal'] else 'NO'):>7s}  {dead}"
              f"{'' if uniq else '   (not unique: outside the judge pool)'}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_read()
