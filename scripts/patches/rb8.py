"""RB8 THE READING BREATHS DESCEND THE TREE (ledger 2026-11-03, word given for the whole queue): distill_reader.py gains DR_RB=1.

Today the reader's Tree (reader_db: BIRCH CFs keyed L0 tok / L1 prev+tok / L2 prev+tok+next) enters ONCE, at the input, as the
deepest-node backoff row; the breaths never revisit it. Under DR_RB=1 (needs DR_DB=1 and >= 3 breaths):
  - the input row is removed (the seal: the breaths are the only road to the Tree)
  - breath b (b = 0, 1, 2) adds level b's row through its own map w_db{b} (a mandatory road, no gain): coarse -> fine, one level
    deeper per breath; breath 3 reads nothing new
  - rows per level are computed from the store on the fly: TRAIN rows leave-one-out from the row's own teacher state (as the banked
    backoff rows were built), eval/export rows a plain lookup; a level without a node (N >= 3) reads zero
Applied by unit lfrb8 after lf8m's GSM8K export (that chain re-reads distill_reader.py). Asserts each anchor once.
"""
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def edit(f, pairs):
    p = os.path.join(R, f); s = open(p).read()
    for a, b in pairs:
        assert s.count(a) == 1, f"{f}: anchor not found exactly once: {a[:80]!r}"
        s = s.replace(a, b)
    open(p, "w").write(s); print(f"[rb8] {f}: {len(pairs)} edit(s)")


edit("distill_reader.py", [
    ('BR = int(os.environ.get("DR_BREATHS", "0"))',
     'RB = os.environ.get("DR_RB") == "1"   # (ledger 11-03 RB8) the reading breaths descend the Tree: breath b reads level b (no input row)\n'
     'BR = int(os.environ.get("DR_BREATHS", "0"))'),
    ('    p = {"w_in": nrm(H_TRUNK, D), **({"w_db": nrm(DBP, D)} if DBIN else {}),',
     '    p = {"w_in": nrm(H_TRUNK, D), **({"w_db": nrm(DBP, D)} if DBIN and not RB else {}), **({f"w_db{l}": nrm(DBP, D) for l in range(3)} if RB else {}),'),
    ('    if db is not None:\n        x = x + db.cast(dtypes.float) @ p["w_db"]\n    xs = []\n    for br in range(max(BR, 1)):\n',
     '    if db is not None and not RB:\n        x = x + db.cast(dtypes.float) @ p["w_db"]\n    xs = []\n    for br in range(max(BR, 1)):\n'
     '      if RB and br < 3:   # (ledger 11-03 RB8) this breath reads one level deeper in the Tree\n'
     '        x = x + db[:, :, br * DBP:(br + 1) * DBP].cast(dtypes.float) @ p[f"w_db{br}"]\n'),
    ('def addmask(tm):\n',
     'if RB:\n'
     '    assert DBIN and BR >= 3 and not GSM, "RB8 needs the DB student, >= 3 breaths, no GSM-training form"\n'
     '    import reader_db_build as RDB\n'
     '    RB_MU, RB_U, RB_SCALE, RB_STORE = RDB.load_store()\n\n\n'
     'def rb_rows(ids_r, m_r, teach_r=None):\n'
     '    """(ledger 11-03 RB8) the Tree\'s three levels as separate rows [B, T, 3 x DBP]; teach_r given = leave-one-out (TRAIN)"""\n'
     '    out = np.zeros((len(ids_r), T_ALG, 3 * DBP), np.float16)\n'
     '    for r in range(len(ids_r)):\n'
     '        n = int(m_r[r].sum()); k = RDB.row_keys(np.asarray(ids_r[r]), n)\n'
     '        z = None if teach_r is None else ((np.asarray(teach_r[r, :n], np.float32) - RB_MU) @ RB_U).astype(np.float16).astype(np.float32)\n'
     '        for li, lv in enumerate(RDB.LEVELS):\n'
     '            uk, N, LS, _ = RB_STORE[lv]; pos = np.clip(np.searchsorted(uk, k[li]), 0, len(uk) - 1); hit = uk[pos] == k[li]\n'
     '            nn_ = np.where(hit, N[pos] - (1 if z is not None else 0), 0.0); ok = nn_ >= RDB.MINN\n'
     '            m_ = (LS[pos] - (z if z is not None else 0)) / np.maximum(nn_, 1)[:, None]\n'
     '            out[r, :n, li * DBP:(li + 1) * DBP] = np.where(ok[:, None], m_, 0.0) * RB_SCALE\n'
     '    return out\n\n\n'
     'def addmask(tm):\n'),
    ('            Tensor(np.asarray(teach[idx]), dtype=dtypes.half)) + ((Tensor(np.asarray(tdb[idx]), dtype=dtypes.half),) if DBIN else ()) + (',
     '            Tensor(np.asarray(teach[idx]), dtype=dtypes.half)) + ((Tensor(rb_rows(ids[idx], mask[idx], teach[idx]) if RB else np.asarray(tdb[idx]), dtype=dtypes.half),) if DBIN else ()) + ('),
    ('        Q.put((EMB[ids[idx]], mask[idx], np.asarray(teach[idx])) + ((np.asarray(tdb[idx]),) if DBIN else ())',
     '        Q.put((EMB[ids[idx]], mask[idx], np.asarray(teach[idx])) + ((rb_rows(ids[idx], mask[idx], teach[idx]) if RB else np.asarray(tdb[idx]),) if DBIN else ())'),
    ('    if DBIN: assert edb.shape == (len(ids), T_ALG, DBP), f"db rows {edb.shape} vs {len(ids)} rows"',
     '    if DBIN: assert edb.shape == (len(ids), T_ALG, DBP), f"db rows {edb.shape} vs {len(ids)} rows"\n'
     '    if RB: edb = rb_rows(ids, mask)   # (RB8) the per-level plain lookup'),
    ('            if DBIN: db[r, :n] = RDB.lookup(store, RDB.row_keys(ids[r], n)) * dscale     # as built: backoff, N >= 3',
     '            if DBIN and not RB: db[r, :n] = RDB.lookup(store, RDB.row_keys(ids[r], n)) * dscale     # as built: backoff, N >= 3\n'
     '        if RB: db = rb_rows(ids, m)   # (RB8) the per-level plain lookup'),
])
