"""THE WHEEL'S ROW CLINIC (2026-09-12): replay a ST_WHEEL_DUMP fixture on CPU.
Per row: wall time to a verdict (with a long cap), status, n_factors — the
distribution that a wall-clock timeout truncates; the memo hit rate a
parse-keyed cache would get (same row re-solved across breaths / steps);
and, per candidate timeout, how many certified refusals (unsat + core)
would be LOST. usage: WHEEL_ROW_TIMEOUT=12 wheel_rows_analysis.py dump.pkl
"""
import os, sys, json, pickle, time
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

def _key(row):
    n_vars, parse, m = row
    return json.dumps([n_vars, m, sorted(json.dumps({k: v for k, v in f.items() if k != "_slot"},
                                                  sort_keys=True, default=str) for f in parse)])

def _timed(args):
    import alternator_bridge as ab
    t = time.time(); st, core = ab._core_worker(args); return st, core, time.time() - t

def main():
    path = sys.argv[1]
    breaths = []
    with open(path, "rb") as f:
        while True:
            try: breaths.append(pickle.load(f))
            except EOFError: break
    rows = [r for b in breaths for r in b["rows"]]; res0 = [r for b in breaths for r in b["res"]]
    print(f"[clinic] {len(breaths)} breaths, {len(rows)} rows; pool wall per breath: "
          f"{np.mean([b['t_cores'] for b in breaths]):.2f}s mean, max {max(b['t_cores'] for b in breaths):.2f}s")
    keys = [_key(r) for r in rows]; seen = set(); hits = 0
    for k in keys:
        if k in seen: hits += 1
        seen.add(k)
    print(f"[clinic] memo: {len(seen)} distinct parses of {len(rows)} rows -> hit rate {hits/len(rows):.3f} (within this fixture)")
    uniq = {}; 
    for k, r in zip(keys, rows): uniq.setdefault(k, r)
    import multiprocessing as mp
    os.environ.setdefault("WHEEL_ROW_TIMEOUT", "12")
    with mp.get_context("spawn").Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        out = dict(zip(uniq.keys(), pool.map(_timed, list(uniq.values()), chunksize=1)))
    ts = np.array([out[k][2] for k in keys]); sts = [out[k][0] for k in keys]
    nf = np.array([len(r[1]) for r in rows])
    print(f"[clinic] row time (cap {os.environ['WHEEL_ROW_TIMEOUT']}s): p50 {np.percentile(ts,50):.3f} p90 {np.percentile(ts,90):.3f} "
          f"p99 {np.percentile(ts,99):.3f} max {ts.max():.2f}s; mean {ts.mean():.3f}")
    from collections import Counter
    print(f"[clinic] status: {dict(Counter(sts))}; certified refusals (unsat+core): {sum(1 for k in keys if out[k][0]=='unsat' and out[k][1])}")
    for st in ("solved", "unsat", "timeout", "unbuildable", "budget"):
        m = np.array([s == st for s in sts])
        if m.any(): print(f"   {st:12s} n={m.sum():4d}  time p50 {np.percentile(ts[m],50):.3f} p90 {np.percentile(ts[m],90):.3f} max {ts[m].max():.2f}  factors mean {nf[m].mean():.1f}")
    cert = np.array([out[k][0] == "unsat" and bool(out[k][1]) for k in keys])
    print("[clinic] per timeout: refusals kept / breath wall (pool of 14, ideal) :")
    for to in (0.25, 0.5, 1.0, 2.0, 3.0, 6.0, 12.0):
        kept = int((cert & (ts <= to)).sum())
        walls = [max(min(out[_key(r)][2], to) for r in b["rows"]) for b in breaths]
        print(f"   timeout {to:5.2f}s: refusals kept {kept:4d}/{cert.sum()}  rows over {int((ts>to).sum()):4d}  breath wall(max row) mean {np.mean(walls):.2f}s")
    # what do the slow rows look like
    order = np.argsort(-ts)[:5]
    for i in order:
        r = rows[i]; print(f"[clinic] slow row {ts[i]:.2f}s status={sts[i]} n_vars={r[0]} m={r[2]} factors={len(r[1])}: "
                           + "; ".join(f"{f.get('ftype', f.get('type','?'))}:{f.get('op','')}" for f in r[1])[:160])

if __name__ == "__main__":
    main()
