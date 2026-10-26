"""tree_db_compaction_bench.py -- COMPACTION's pinned bars C1-C3 (ledger 2026-10-26, "COMPACTION"). CPU, no training.
  C1 SCALING      D2's grid on v2 + compaction (TAIL_MAX 32 / TAIL_KEEP 16): add / roll back the newest 1-event batch,
                  t(1M) / t(10k) <= 3 each
  C2 EQUIVALENCE  400 random operations (add permanent / add provisional / promote / roll back -- compacted batches
                  included) on a compacting store and a twin that never compacts: leaves bit-identical after EVERY
                  operation; rebuild_all agrees on both. Run at the registered (32, 16) and at (4, 2) to exercise it hard
  C3 REGRESSION   tree_db_check passes; tree_db_v2_bench's D1 and D3 still pass"""
import os, shutil, statistics, subprocess, sys, tempfile, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mycelium import tree_db as T
import tree_db_v2_bench as V

OUT = ".cache/tree_db_compaction.txt"
GRID, CH, REPS, SPACE = [10_000, 100_000, 1_000_000], 10_000, 5, "node-evidence"


def leaves(db):
    return db.c.execute("SELECT key, n, mean, m2 FROM leaves ORDER BY key").fetchall()


def c1(say, d):
    rng = np.random.default_rng(0); res = {}
    for H in GRID:
        db = T.TreeDB(os.path.join(d, f"c1_{H}.db")); db.register_space(SPACE, "symbolic", 4, "synthetic")
        for b in range(0, H, CH):
            ks = rng.integers(0, 4, CH)
            db.add_evidence(f"hist{b}", [("hot", np.eye(4)[k], f"h{b + j}") for j, k in enumerate(ks)], "permanent", "synthetic", "bench", SPACE)
        ta, tr = [], []
        for r in range(REPS):
            t0 = time.perf_counter(); db.add_evidence(f"probe{r}", [("hot", np.eye(4)[0], "p")], "permanent", "synthetic", "bench", SPACE)
            ta.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); db.rollback(f"probe{r}"); tr.append(time.perf_counter() - t0)
        cp = db.c.execute("SELECT upto FROM checkpoints WHERE key='hot'").fetchone()
        tail = db.c.execute(f"SELECT COUNT(*) FROM summaries s JOIN batches b ON s.bid=b.bid WHERE s.key='hot' AND {T.LIVE} AND s.bid > ?",
                            (cp[0] if cp else 0,)).fetchone()[0]
        res[H] = (statistics.median(ta), statistics.median(tr))
        say(f"C1 history {H:>9,} ({H // CH} batches): add {1e3 * res[H][0]:.2f} ms, rollback {1e3 * res[H][1]:.2f} ms; "
            f"checkpoint upto {cp[0] if cp else None}, tail {tail}")
        db.close()
    ra, rr = res[GRID[-1]][0] / res[GRID[0]][0], res[GRID[-1]][1] / res[GRID[0]][1]
    say(f"C1 ratio t(1M)/t(10k): add {ra:.2f}, rollback {rr:.2f} (v2 without compaction: 2.21 / 11.50)")
    return ra <= 3 and rr <= 3


def c2(say, d, tail_max, tail_keep, n_ops=400):
    rng = np.random.default_rng(tail_max)
    a = T.TreeDB(os.path.join(d, f"c2a_{tail_max}.db"), tail_max=tail_max, tail_keep=tail_keep)
    b = T.TreeDB(os.path.join(d, f"c2b_{tail_max}.db"), tail_max=None)
    for x in (a, b):
        x.register_space("v3", "symbolic", 3, "synthetic")
    live, prov, ops, mismatch, slow = [], [], {}, 0, 0
    for i in range(n_ops):
        u = rng.random()
        if u < 0.80 and not prov:
            u = 0.0                                                         # nothing to promote: add instead
        if u < 0.70 or not live:
            op = "add_perm" if rng.random() < 0.8 else "add_prov"
            rows = [(f"k{int(k)}", rng.normal(size=3) + k, f"o{i}.{j}") for j, k in enumerate(rng.integers(0, 6, rng.integers(1, 20)))]
            for x in (a, b):
                x.add_evidence(f"b{i}", rows, "permanent" if op == "add_perm" else "provisional", "synthetic", "bench", "v3")
            live.append(f"b{i}")
            if op == "add_prov":
                prov.append(f"b{i}")
        elif u < 0.80:
            op = "promote"; nm = prov.pop(int(rng.integers(0, len(prov))))
            for x in (a, b):
                x.promote(nm)
        else:
            op = "rollback"; nm = live.pop(int(rng.integers(0, len(live))))
            if nm in prov:
                prov.remove(nm)
            bid = a._batch(nm)[0]
            slow += bool(a.c.execute("SELECT COUNT(*) FROM checkpoints WHERE upto >= ?", (bid,)).fetchone()[0])
            for x in (a, b):
                x.rollback(nm)
        ops[op] = ops.get(op, 0) + 1
        mismatch += leaves(a) != leaves(b)
    na, nb = a.rebuild_all(), b.rebuild_all(); same = leaves(a) == leaves(b)
    ncp = a.c.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    say(f"C2 (tail_max {tail_max}, keep {tail_keep}): ops {ops}; rollbacks reaching into a checkpoint {slow}; checkpoints at end {ncp}; "
        f"operations with any leaf mismatch {mismatch}/{n_ops}; rebuild_all OK on both ({na} / {nb} events), equal after {same}")
    a.close(); b.close()
    return mismatch == 0 and same and slow > 0


if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); d = tempfile.mkdtemp(prefix="compaction_", dir=os.environ.get("BENCH_TMP"))

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"pid {os.getpid()}; scratch {d}")
    r1 = c1(say, d)
    r2 = c2(say, d, 32, 16) & c2(say, d, 4, 2)
    chk = subprocess.run([sys.executable, "scripts/tree_db_check.py"], capture_output=True, text=True)
    say(f"C3 tree_db_check: {chk.stdout.strip().splitlines()[-1] if chk.returncode == 0 else 'FAILED ' + chk.stderr[-300:]}")
    ok1, _ = V.d1(say, d); ok3 = V.d3(say, d)
    r3 = chk.returncode == 0 and ok1 and ok3
    say("VERDICTS")
    say(f"C1 SCALING: {'PASS' if r1 else 'FAIL'}")
    say(f"C2 EQUIVALENCE: {'PASS' if r2 else 'FAIL'}")
    say(f"C3 REGRESSION: {'PASS' if r3 else 'FAIL'} (check {chk.returncode == 0}, D1 {ok1}, D3 {ok3})")
    shutil.rmtree(d); say(f"DONE in {time.time() - t0:.0f}s")
