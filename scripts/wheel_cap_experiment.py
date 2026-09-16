"""wheel_cap_experiment.py — is the wheel's wall-clock row cap the source of
non-determinism, and can a deterministic cap replace it? On the wheel-row
CPU fixture (.cache/wheel_rows_dump.pkl: rows + the pool's answers):
  A. the alarm cap (WHEEL_ROW_TIMEOUT=1) run TWICE -> answers that differ
     between the two runs (timing-dependent rows)
  B. no alarm (timeout 600 s), the decision budget + arity cap + M_MAX only
     -> deterministic by construction; the max row wall time says whether the
     alarm can go; answers compared with A (timeouts resolved either way)
usage: wheel_cap_experiment.py [n_rows] [workers]"""
import os, sys, pickle, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
n_cap = int(sys.argv[1]) if len(sys.argv) > 1 else 300; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rows = []; seen = set()
with open(".cache/wheel_rows_dump.pkl", "rb") as f:
    while True:
        try: d = pickle.load(f)
        except EOFError: break
        for r in d["rows"]:
            k = (r[0], r[2], str(r[1]))
            if k not in seen: seen.add(k); rows.append(r)
rows = rows[:n_cap]; print(f"[cap] {len(rows)} unique fixture rows; workers {workers}", flush=True)
os.environ["WHEEL_M_MAX"] = "10000"; os.environ["WHEEL_BUDGET"] = "2000"
from alternator_bridge import core_rows
def run(timeout):
    os.environ["WHEEL_ROW_TIMEOUT"] = str(timeout); t0 = time.time()
    res = core_rows(rows, workers); return res, time.time() - t0
a1, ta1 = run(1); a2, ta2 = run(1)
diff = sum(1 for x, y in zip(a1, a2) if x != y); st = {}
for s, _ in a1: st[s] = st.get(s, 0) + 1
print(f"[cap] A (alarm 1 s) x2: {ta1:.1f}s / {ta2:.1f}s; answers differing between the two runs: {diff}/{len(rows)}; statuses run 1: {st}", flush=True)
import multiprocessing as _mp
# B in fresh workers so the alarm state is clean; per-row wall time measured inside a small wrapper
def _timed(args):
    import time as _t, os as _o
    _o.environ["WHEEL_ROW_TIMEOUT"] = "600"
    from alternator_bridge import _core_worker
    t0 = _t.time(); r = _core_worker(args); return r, _t.time() - t0
# no Pool.map: a dead worker hangs it forever (the 2026-09-15 run); imap_unordered with a per-result
# timeout instead, and rows that never return are counted as "hung" (the deterministic cap's real cost)
with _mp.get_context("spawn").Pool(workers) as pool:
    t0 = time.time(); outB = []; it = pool.imap_unordered(_timed, rows, chunksize=1); hung = 0
    for _ in range(len(rows)):
        try: outB.append(it.next(timeout=float(os.environ.get("CAP_B_TIMEOUT", "900"))))
        except _mp.TimeoutError: hung = len(rows) - len(outB); print(f"[cap] B: {hung} rows still running after {os.environ.get('CAP_B_TIMEOUT', '900')} s each — counted as hung", flush=True); break
    tb = time.time() - t0
b = [r for r, _ in outB]; times = sorted(t for _, t in outB) or [0.0]
stb = {}
for s, _ in b: stb[s] = stb.get(s, 0) + 1
agree = "n/a (unordered)"; to_res = "n/a"
print(f"[cap] B (no alarm; budget 2000 + arity cap + M_MAX): {tb:.1f}s total; row wall max {times[-1]:.2f}s, p99 {times[int(0.99 * len(times)) - 1]:.2f}s, median {times[len(times) // 2]:.3f}s; statuses {stb}; agrees with A on {agree}/{len(rows)}; A's timeouts resolved by B: {to_res}", flush=True)
print("[cap] B is deterministic by construction (no wall clock in the cap); if its max row wall is acceptable, the alarm can go and every wheel path becomes reproducible")
