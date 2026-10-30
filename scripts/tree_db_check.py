"""tree_db_check.py — the permanent tree's audit (mycelium/tree_db.py). Zero-GPU, seconds. Runs on a scratch DB file.
Checks: incremental Welford == numpy over the raw events; rebuild_all == incremental; provisional evidence stays out of the
aggregates until promote; rollback restores the aggregates exactly as if the batch never existed; a reused batch name
fails atomically (nothing partial lands); the centroid generation guard refuses a foreign generation; provenance per row.
v3: forests by kind + children (the walk); macros land and roll back with their batch; promote_recurrent admits only keys with
>= k distinct sources, supersedes its sources (a later promote of one is refused) and folds == numpy; constants() separates a
zero-variance key from a per-problem slot; a v2 store upgrades in place, additively, with its leaves bit-identical."""
import os, sqlite3, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mycelium.tree_db import TreeDB


def expect(db, raw):
    """raw: {key: [vec, ...]} of the evidence that SHOULD be in the aggregates."""
    for k, vs in raw.items():
        X = np.stack(vs); n, m, var = db.leaf(k)
        assert n == len(X), (k, n, len(X))
        assert np.allclose(m, X.mean(0)), k
        assert np.allclose(var, X.var(0, ddof=1) if len(X) > 1 else np.zeros(X.shape[1])), k
    assert set(db.load_leaves()) == {k for k, v in raw.items() if v}, "leaf set differs"


def main():
    rng = np.random.default_rng(0); path = os.path.join(tempfile.mkdtemp(), "tree_check.db")
    db = TreeDB(path); keys = [f"tok{i}" for i in range(12)]; raw = {}; per_batch = {}
    db.register_space("vec5", "symbolic", 5, "test space: 5-way evidence")
    for b in range(6):                                                      # six permanent batches, overlapping keys
        rows = [(k, rng.normal(size=5) + i, f"ep{b}.m{j}") for j, (i, k) in
                enumerate((int(i), keys[int(i)]) for i in rng.integers(0, 12, 40))]
        db.add_evidence(f"B{b}", rows, "permanent", "answer_key", "gen1", "vec5"); per_batch[f"B{b}"] = rows
        for k, v, _ in rows:
            raw.setdefault(k, []).append(v)
    expect(db, raw); print("incremental Welford == numpy over raw events: OK")
    db.rebuild_all(); expect(db, raw); print("rebuild_all == incremental: OK")

    prov = [(keys[0], rng.normal(size=5), "test.q1"), (keys[1], rng.normal(size=5), "test.q2")]
    db.add_evidence("P0", prov, "provisional", "titles", "gen1", "vec5"); expect(db, raw); print("provisional stays out of the aggregates: OK")
    db.promote("P0")
    for k, v, _ in prov:
        raw[k].append(v)
    expect(db, raw); print("promote brings it in: OK")

    db.rollback("B3")
    raw2 = {}
    for b, rows in per_batch.items():
        if b != "B3":
            for k, v, _ in rows:
                raw2.setdefault(k, []).append(v)
    for k, v, _ in prov:
        raw2.setdefault(k, []).append(v)
    expect(db, raw2); print("rollback == the batch never existed: OK")

    n_ev = db.n_events()
    try:
        db.add_evidence("B0", [(keys[0], np.zeros(5), "dup")], "permanent", "answer_key", "gen1", "vec5"); raise AssertionError("reuse accepted")
    except sqlite3.IntegrityError:
        pass
    assert db.n_events() == n_ev; expect(db, raw2)
    try:                                                                    # a bad row mid-batch must leave nothing behind
        db.add_evidence("BAD", [(keys[2], np.zeros(5), "ok"), (keys[3], np.zeros(7), "wrong dim")], "permanent", "answer_key", "gen1", "vec5")
        raise AssertionError("dim mismatch accepted")
    except (AssertionError, ValueError) as e:                              # the space guard catches it before any write
        assert "dim" in str(e) and "accepted" not in str(e), e
    assert db.n_events() == n_ev
    assert db.c.execute("SELECT COUNT(*) FROM batches WHERE batch='BAD'").fetchone()[0] == 0; expect(db, raw2)
    print("reused batch name / bad row mid-batch fail atomically: OK")

    db.register_space("model:gen1", "model", 32, "a model's head coordinates")
    for args, why in ((("M0", [(keys[0], np.zeros(32), "h")], "permanent", "x", "gen1", "model:gen1"), "model space"),
                      (("M1", [(keys[0], np.zeros(5), "h")], "permanent", "x", "gen1", "nope"), "unregistered"),
                      (("M2", [(keys[0], np.zeros(4), "h")], "permanent", "x", "gen1", "vec5"), "wrong dim")):
        try:
            db.add_evidence(*args); raise AssertionError(f"{why} accepted")
        except ValueError:
            pass
    db.register_space("vec4", "symbolic", 4, "another symbolic space")
    try:
        db.add_evidence("M3", [(keys[0], np.zeros(4), "h")], "permanent", "x", "gen1", "vec4"); raise AssertionError("key in two spaces")
    except ValueError:
        pass
    try:
        db.register_space("vec5", "symbolic", 6, "redefinition"); raise AssertionError("space redefined")
    except ValueError:
        pass
    assert db.n_events() == n_ev; expect(db, raw2)
    print("space guard (model space / unregistered / wrong dim / key in two spaces / redefinition refused): OK")

    db.save_centroids("gen1", np.ones(5), rng.normal(size=(5, 32)), np.ones(5))
    assert db.load_centroids("gen1", "gen1")[1].shape == (5, 32)
    try:
        db.load_centroids("gen1", "gen2"); raise AssertionError("foreign generation loaded")
    except ValueError:
        pass
    print("centroid generation guard: OK")

    db.add_node(0, None, 0, "root", "B0"); db.add_node(1, 0, 1, "person", "B1")
    assert [r[3] for r in db.nodes()] == ["root", "person"]
    pv = db.provenance(keys[0]); assert pv and all(len(r) == 6 for r in pv)
    print(f"shape + provenance: OK ({len(pv)} events behind {keys[0]}; e.g. {pv[0]})")
    snap = {k: (db.c.execute("SELECT mean, m2 FROM leaves WHERE key=?", (k,)).fetchone()) for k in db.load_leaves()}
    db.rebuild_all()
    assert snap == {k: (db.c.execute("SELECT mean, m2 FROM leaves WHERE key=?", (k,)).fetchone()) for k in db.load_leaves()}
    print("rebuild_all from the cold events == incremental, bit-exact: OK")
    bid = db.c.execute("SELECT bid FROM batches WHERE batch='B0'").fetchone()[0]; p = db._cold_path(bid)
    good = open(p, "rb").read(); open(p, "wb").write(good[:-9] + bytes(9))  # tamper with one live batch's cold file
    try:
        db.rebuild_all(); raise AssertionError("tampered cold file accepted")
    except ValueError as e:
        assert "SHA-256" in str(e), e
    open(p, "wb").write(good); db.rebuild_all(); print("a tampered cold file fails the audit loudly: OK")
    v3_checks(db, rng)
    db.close(); v2_upgrade_check(); print("ALL CHECKS PASS")


def v3_checks(db, rng):
    db.add_node(10, None, 0, "quantity", "B0", kind="type"); db.add_node(11, 10, 1, "money", "B1", kind="type")
    db.add_node(12, 10, 1, "count", "B1", kind="type")
    db.add_macro(20, None, 1, "percent-of", "mul(div(n,c100),n)", "gsm-ptr-1", "B2")
    assert [r[3] for r in db.nodes("type")] == ["quantity", "money", "count"] and [r[0] for r in db.children(10)] == [11, 12]
    assert {r[0] for r in db.children(None)} >= {0, 10, 20} and db.macros()[0][1:4] == ("percent-of", "mul(div(n,c100),n)", "gsm-ptr-1")
    db.rollback("B2"); assert db.macros() == [] and not [r for r in db.nodes() if r[0] == 20]
    print("v3 forests by kind + children + macros roll back with their batch: OK")
    db.register_space("const1", "symbolic", 1, "a numeric value carried by a word in a certified problem")
    eps = {"dozen": [12.0] * 5, "crate": [50.0, 24.0, 10.0, 36.0], "pentagon": [5.0, 5.0]}   # 5 / 4 / 2 distinct problems
    for j in range(5):
        rows = [(k, np.array([v[j]]), f"problem{j}") for k, v in eps.items() if j < len(v)]
        db.add_evidence(f"EP{j}", rows, "provisional", "solver", "gen1", "const1")
    keep, held = db.promote_recurrent("REC0", [f"EP{j}" for j in range(5)], 3, "answer_key", "gen1")
    assert keep == ["crate", "dozen"] and held == ["pentagon"], (keep, held)
    for k in keep:
        n, m, var = db.leaf(k); X = np.array(eps[k])
        assert n == len(X) and np.allclose(m, X.mean()) and np.allclose(var, X.var(ddof=1)), k
    assert db.leaf("pentagon") is None and db.c.execute("SELECT COUNT(*) FROM key_space WHERE key='pentagon'").fetchone()[0] == 0
    try:
        db.promote("EP0"); raise AssertionError("a superseded batch promoted (double count)")
    except AssertionError as e:
        assert "superseded" in str(e), e
    assert set(db.constants(1e-9, 3)) == {"dozen"} and np.allclose(db.constants(1e-9, 3)["dozen"], 12.0)
    db.rebuild_all(); db.rollback("REC0"); assert db.leaf("dozen") is None
    print("promote_recurrent (>= 3 distinct sources; sources superseded; == numpy) + constants() + rollback: OK")


def v2_upgrade_check():
    """a v2 store (the v2 schema verbatim, as gsmb_tree.db / tree.db are) upgrades in place; its leaves stay bit-identical."""
    import shutil, glob as _g
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "gsmb_tree.db")
    if not os.path.exists(src):
        print("v2 upgrade: no banked v2 store to copy (skipped)"); return
    d = tempfile.mkdtemp(); dst = os.path.join(d, "v2.db"); shutil.copy(src, dst); shutil.copytree(src + ".events", dst + ".events")
    for ext in ("-wal", "-shm"):
        if os.path.exists(src + ext): shutil.copy(src + ext, dst + ext)
    c = sqlite3.connect(dst); before = c.execute("SELECT key, n, mean, m2 FROM leaves ORDER BY key").fetchall()
    v = c.execute("SELECT v FROM meta WHERE k='schema_version'").fetchone()[0]; nn = c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]; c.close()
    assert v == "2", v
    db = TreeDB(dst); assert db.c.execute("SELECT key, n, mean, m2 FROM leaves ORDER BY key").fetchall() == before
    assert len(db.nodes("node")) == nn and db.c.execute("SELECT v FROM meta WHERE k='schema_version'").fetchone()[0] == "3"
    db.rebuild_all(); db.close(); db = TreeDB(dst); db.close()                 # re-open of an upgraded store is a no-op
    print(f"v2 store upgrades in place, additively ({len(before)} leaves bit-identical, {nn} nodes kind 'node'): OK")


if __name__ == "__main__":
    main()
