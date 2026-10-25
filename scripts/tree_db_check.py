"""tree_db_check.py — the permanent tree's audit (mycelium/tree_db.py). Zero-GPU, seconds. Runs on a scratch DB file.
Checks: incremental Welford == numpy over the raw events; rebuild_all == incremental; provisional evidence stays out of the
aggregates until promote; rollback restores the aggregates exactly as if the batch never existed; a reused batch name
fails atomically (nothing partial lands); the centroid generation guard refuses a foreign generation; provenance per row."""
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
    for b in range(6):                                                      # six permanent batches, overlapping keys
        rows = [(k, rng.normal(size=5) + i, f"ep{b}.m{j}") for j, (i, k) in
                enumerate((int(i), keys[int(i)]) for i in rng.integers(0, 12, 40))]
        db.add_evidence(f"B{b}", rows, "permanent", "answer_key", "gen1"); per_batch[f"B{b}"] = rows
        for k, v, _ in rows:
            raw.setdefault(k, []).append(v)
    expect(db, raw); print("incremental Welford == numpy over raw events: OK")
    db.rebuild_all(); expect(db, raw); print("rebuild_all == incremental: OK")

    prov = [(keys[0], rng.normal(size=5), "test.q1"), (keys[1], rng.normal(size=5), "test.q2")]
    db.add_evidence("P0", prov, "provisional", "titles", "gen1"); expect(db, raw); print("provisional stays out of the aggregates: OK")
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

    n_ev = db.c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    try:
        db.add_evidence("B0", [(keys[0], np.zeros(5), "dup")], "permanent", "answer_key", "gen1"); raise AssertionError("reuse accepted")
    except sqlite3.IntegrityError:
        pass
    assert db.c.execute("SELECT COUNT(*) FROM events").fetchone()[0] == n_ev; expect(db, raw2)
    try:                                                                    # a bad row mid-batch must leave nothing behind
        db.add_evidence("BAD", [(keys[2], np.zeros(5), "ok"), (keys[3], np.zeros(7), "wrong dim")], "permanent", "answer_key", "gen1")
        raise AssertionError("dim mismatch accepted")
    except AssertionError as e:
        assert "dim" in str(e), e
    assert db.c.execute("SELECT COUNT(*) FROM events").fetchone()[0] == n_ev
    assert db.c.execute("SELECT COUNT(*) FROM batches WHERE batch='BAD'").fetchone()[0] == 0; expect(db, raw2)
    print("reused batch name / bad row mid-batch fail atomically: OK")

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
    db.close(); os.remove(path); print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
