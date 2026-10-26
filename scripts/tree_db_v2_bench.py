"""tree_db_v2_bench.py -- TREE_DB v2's pinned bars D1-D3 (ledger 2026-10-26 06:25; D0 = scripts/tree_db_check.py, D4 = the
T2w re-run). CPU, no training. Reads the banked v1 T2w stores read-only; every v2 store is built in a scratch dir.
  D1 EQUIVALENCE  (a) migrate_v1 on the three T2w stores: n equal, mean/M2 within 1e-9 relative of the v1 leaves;
                  (b) the s0 live events in 8 shards, prepared in 8 parallel processes, committed: within 1e-9 of serial
  D2 SCALING      a hot key's history 10k -> 100k -> 1M events (batches of 10k): median add / rollback of a 1-event batch,
                  t(1M) / t(10k) <= 3 each; v1 (its git version, 358e1957) timed on the same grid, reported
  D3 ISOLATION    base + 8 provisional shards of s0 events, one a saboteur (node k -> k+1): saboteur rejected, 7 promote,
                  leaves within 1e-9 of base + the 7 clean shards alone"""
import importlib.util, os, shutil, sqlite3, statistics, subprocess, sys, tempfile, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mycelium import tree_db as T

SMOKE = "--smoke" in sys.argv
OUT = ".cache/tree_db_v2_bench{}.txt".format("_smoke" if SMOKE else "")
V1 = [f".cache/tree_handshake_s{s}.db" for s in ((0,) if SMOKE else (0, 1, 2))]
CAP = 60_000 if SMOKE else None                                             # smoke: the first 60k events only
GRID = [1_000, 10_000] if SMOKE else [10_000, 100_000, 1_000_000]
CH, REPS, TOL, SHARDS, AUDIT = 10_000, 5, 1e-9, 8, 0.99
SPACE = "node-evidence"


def v1_leaves(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    return {k: (n, T._arr(m), T._arr(s)) for k, n, m, s in c.execute("SELECT key, n, mean, m2 FROM leaves")}


def v2_leaves(db):
    return {k: (n, T._arr(m), T._arr(s)) for k, n, m, s in db.c.execute("SELECT key, n, mean, m2 FROM leaves")}


def compare(a, b):
    """(same key set and n, max relative diff over mean and M2, bit-exact)."""
    if set(a) != set(b) or any(a[k][0] != b[k][0] for k in a):
        return False, float("inf"), False
    rel = max(float(np.max(np.abs(a[k][i] - b[k][i]) / np.maximum(1.0, np.abs(b[k][i])))) for k in a for i in (1, 2))
    exact = all(a[k][i].tobytes() == b[k][i].tobytes() for k in a for i in (1, 2))
    return True, rel, exact


def live_events(path):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = """SELECT e.key, e.vec, e.source FROM events e JOIN batches b ON e.batch=b.batch
           WHERE b.tier='permanent' AND b.status='live' ORDER BY e.id""" + (f" LIMIT {CAP}" if CAP else "")
    return [(k, T._arr(v), s) for k, v, s in c.execute(q)]


def fresh(d, name):
    db = T.TreeDB(os.path.join(d, name)); db.register_space(SPACE, "symbolic", 4, "Welford evidence over the tree's 4 leaf-level nodes")
    return db


def dir_bytes(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


def d1(say, d):
    ok_a, rels = True, []
    for p in V1:
        t0 = time.time()
        if CAP:                                                             # smoke: a capped v1 copy, migrated
            src = os.path.join(d, "cap_v1.db"); cs = sqlite3.connect(src); co = sqlite3.connect(p)
            cs.executescript(co.execute("SELECT group_concat(sql, ';') FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'").fetchone()[0])
            for t in ("spaces", "batches", "nodes", "centroids"):
                rows = co.execute(f"SELECT * FROM {t}").fetchall()
                if rows:
                    cs.executemany(f"INSERT INTO {t} VALUES ({','.join('?' * len(rows[0]))})", rows)
            cs.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", co.execute(f"SELECT * FROM events ORDER BY id LIMIT {CAP}").fetchall())
            used = {r[0] for r in cs.execute("SELECT DISTINCT batch FROM events")}
            cs.execute(f"DELETE FROM batches WHERE status='live' AND batch NOT IN ({','.join('?' * len(used))})", tuple(used))
            cs.commit(); cs.close()
            sys.path.insert(0, d); ref_db = _v1_module(d).TreeDB(src); ref_db.rebuild_all(); ref_db.close(); p_ref = src
        else:
            p_ref = p
        db = T.migrate_v1(p_ref, os.path.join(d, "mig_" + os.path.basename(p)))
        db.c.execute("PRAGMA wal_checkpoint(TRUNCATE)")                    # the WAL folded in before the size read
        same, rel, exact = compare(v2_leaves(db), v1_leaves(p_ref)); ok_a &= same and rel <= TOL; rels.append(rel)
        say(f"D1a migrate {p}: {db.n_events()} live events, {len(v2_leaves(db))} leaves; n equal {same}; max rel diff {rel:.2e}; "
            f"bit-exact {exact}; bytes v1 {os.path.getsize(p_ref):,} -> v2 db {os.path.getsize(db.path):,} + cold {dir_bytes(db.cold):,} "
            f"[{time.time() - t0:.0f}s]")
        db.close()
    ev = live_events(V1[0] if not CAP else os.path.join(d, "cap_v1.db"))
    serial = T.TreeDB(os.path.join(d, "mig_" + os.path.basename(V1[0])))
    shards = [ev[i::SHARDS] for i in range(SHARDS)]
    t0 = time.time()
    with Pool(SHARDS) as pool:
        preps = pool.starmap(T.prepare_shard, [(s, d) for s in shards])
    t_map = time.time() - t0; t0 = time.time()
    db = fresh(d, "sharded.db")
    for i, pr in enumerate(preps):
        db.add_prepared(f"shard{i}", pr, "permanent", "gold_resolution", "bench", SPACE)
    t_commit = time.time() - t0
    same, rel, exact = compare(v2_leaves(db), v2_leaves(serial)); ok_b = same and rel <= TOL
    say(f"D1b {len(ev)} events in {SHARDS} shards: map {t_map:.1f}s ({SHARDS} processes), commit {t_commit:.2f}s; vs serial: n equal {same}, "
        f"max rel diff {rel:.2e}, bit-exact {exact}")
    db.close(); serial.close()
    return ok_a and ok_b, max(rels + [rel])


def _v1_module(d):
    p = os.path.join(d, "tree_db_v1.py")
    if not os.path.exists(p):
        open(p, "w").write(subprocess.check_output(["git", "show", "358e1957:mycelium/tree_db.py"], text=True))
    spec = importlib.util.spec_from_file_location("tree_db_v1", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def d2(say, d):
    rng = np.random.default_rng(0); res = {}
    for ver in ("v2", "v1"):
        mod = T if ver == "v2" else _v1_module(d)
        for H in GRID:
            p = os.path.join(d, f"scale_{ver}_{H}.db"); db = mod.TreeDB(p)
            db.register_space(SPACE, "symbolic", 4, "synthetic")
            for b in range(0, H, CH):
                n = min(CH, H - b); ks = rng.integers(0, 4, n)
                db.add_evidence(f"hist{b}", [("hot", np.eye(4)[k], f"h{b + j}") for j, k in enumerate(ks)], "permanent", "synthetic", "bench", SPACE)
            ta, tr = [], []
            for r in range(REPS):
                t0 = time.perf_counter(); db.add_evidence(f"probe{r}", [("hot", np.eye(4)[0], "p")], "permanent", "synthetic", "bench", SPACE)
                ta.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); db.rollback(f"probe{r}"); tr.append(time.perf_counter() - t0)
            res[ver, H] = (statistics.median(ta), statistics.median(tr))
            say(f"D2 {ver} history {H:>9,}: add 1 event {1e3 * res[ver, H][0]:8.2f} ms, rollback {1e3 * res[ver, H][1]:8.2f} ms")
            db.close()
    lo, hi = GRID[0], GRID[-1]
    ra, rr = res["v2", hi][0] / res["v2", lo][0], res["v2", hi][1] / res["v2", lo][1]
    say(f"D2 v2 ratio t({hi:,})/t({lo:,}): add {ra:.2f}, rollback {rr:.2f} | v1: add {res['v1', hi][0] / res['v1', lo][0]:.1f}, "
        f"rollback {res['v1', hi][1] / res['v1', lo][1]:.1f}")
    return ra <= 3 and rr <= 3


def audition(db, batch):
    own = db.batch_summaries(batch); shared = [k for k in own if db.leaf(k) is not None]
    return (float(np.mean([own[k][1].argmax() == db.leaf(k)[1].argmax() for k in shared])) if shared else float("nan")), len(shared)


def d3(say, d):
    ev = live_events(V1[0] if not CAP else os.path.join(d, "cap_v1.db"))
    half = len(ev) // 2; base, rest = ev[:half], ev[half:]
    shards = [rest[i::SHARDS] for i in range(SHARDS)]; SAB = 5
    shards[SAB] = [(k, np.roll(v, 1), s) for k, v, s in shards[SAB]]       # node k -> k+1 mod 4
    db, ref = fresh(d, "iso.db"), fresh(d, "iso_ref.db")
    for x in (db, ref):
        x.add_evidence("base", base, "permanent", "gold_resolution", "bench", SPACE)
    verdicts = []
    for i, sh in enumerate(shards):
        db.add_evidence(f"shard{i}", sh, "provisional", "saboteur" if i == SAB else "gold_resolution", "bench", SPACE)
        a, nk = audition(db, f"shard{i}"); passed = a >= AUDIT
        (db.promote if passed else db.rollback)(f"shard{i}"); verdicts.append((i, round(a, 3), nk, passed))
        if i != SAB:
            ref.add_evidence(f"shard{i}", sh, "permanent", "gold_resolution", "bench", SPACE)
    same, rel, exact = compare(v2_leaves(db), v2_leaves(ref))
    ok = all(p == (i != SAB) for i, _, _, p in verdicts) and same and rel <= TOL
    say(f"D3 auditions (shard, agreement, shared keys, promoted): {verdicts}")
    say(f"D3 final leaves vs base + 7 clean shards: n equal {same}, max rel diff {rel:.2e}, bit-exact {exact}; audit {db.rebuild_all():,} events re-read OK")
    db.close(); ref.close()
    return ok


if __name__ == "__main__":
    assert SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); d = tempfile.mkdtemp(prefix="tree_db_v2_", dir=os.environ.get("BENCH_TMP"))

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {'smoke' if SMOKE else 'full'} (pid {os.getpid()}); v1 stores {V1}; scratch {d}")
    r1, rel = d1(say, d); r2 = d2(say, d); r3 = d3(say, d)
    say("VERDICTS")
    say(f"D1 EQUIVALENCE: {'PASS' if r1 else 'FAIL'} (max rel diff {rel:.2e} vs 1e-9)")
    say(f"D2 SCALING: {'PASS' if r2 else 'FAIL'}")
    say(f"D3 ISOLATION: {'PASS' if r3 else 'FAIL'}")
    shutil.rmtree(d); say(f"DONE in {time.time() - t0:.0f}s")
