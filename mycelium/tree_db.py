"""tree_db.py — THE PERMANENT TREE (2026-10-25, Bryce's word: "word given for the sqlite db -- salute!"). SQLite,
embedded, .cache-resident, no daemons (campaign_db.py's pattern). v2 (2026-10-26, Bryce's word: "word given for the
tree_db fixes -- salute!"; ledger "TREE_DB v2").
The rulings it encodes (ledger 2026-10-25, T2p/T2q/T2r):
  - ROWS ARE KEPT FOREVER; the tree only accumulates evidence. The raw EVENTS are append-only and are the source of
    truth; every leaf aggregate (Welford n / mean / m2) is a materialization, always rebuildable from the events.
  - PROVENANCE: every event names its source (episode/mention), its batch, its certifier and its generation, so a bad
    batch is rolled back by dropping it from the fold and re-folding the keys it touched.
  - CUSTODY TIERS: 'permanent' (answer-key-certified) events enter the aggregates; 'provisional' (test-time) events are
    stored but stay OUT of the aggregates until their batch is promoted.
  - CENTROIDS ARE REBUILT PER RUN (never mix generations' coordinates): interior BIRCH cluster features may be cached
    tagged by generation, and a load for any other generation raises.
  - TREE SHAPE persists only with provenance: a node row names the batch that certified it.
  - THE SPACE GUARD (Bryce's word, 10-25): THE PERMANENT TREE SPEAKS ONLY IN SYMBOLS. Every batch declares a registered
    SPACE (e.g. 'node-evidence': a distribution over the tree's own node ids) with a fixed dim; only kind='symbolic'
    spaces may hold events. Model-space vectors (hidden states, head coordinates) never enter the events or leaves --
    only the generation-tagged CENTROIDS table, which is derived and disposable. So a new model needs no translator:
    it reads the symbols, re-anchors its own centroids, and writes on probation (provisional -> audition -> promote).
v2 LAYOUT (the v1 store re-scanned a key's whole event history on every add and every rollback):
  - SUMMARIES(batch, key -> n, mean, m2): the MapReduce combiner, materialized. A leaf is the Chan LEFT-FOLD of its key's
    live permanent summaries in batch order; every write refolds only the keys it touches (cost ~ batches per key, never
    events per key), and nothing is ever subtracted.
  - COLD EVENTS: each batch's raw events live in a zstd Parquet file (key, vec, source) beside the DB, its SHA-256 on the
    batch row. rebuild_all re-hashes every file, recomputes every summary from its events (bit-exact) and refolds.
    A rolled-back batch keeps its file (forensics) and leaves the fold.
  - KEY_SPACE(key -> space): the one-space check is a row lookup. WAL journal: readers never block the single writer.
  - THE MAP STEP IS PURE: prepare_shard(rows) -> (cold file, summaries) may run in any number of worker processes; the
    single writer commits prepared shards (add_prepared). One batch = one shard = one audition = one rollback unit.
The breath never queries SQLite: a run loads the leaves into memory (`load_leaves`) and snaps against in-memory centroids."""
import hashlib, os, sqlite3, tempfile, time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DB = os.path.join(os.path.dirname(__file__), "..", ".cache", "tree.db")
TIERS = ("permanent", "provisional")
VERSION = "2"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS spaces(
  name TEXT PRIMARY KEY, kind TEXT NOT NULL, dim INTEGER NOT NULL, meaning TEXT NOT NULL, created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batches(
  bid INTEGER PRIMARY KEY, batch TEXT UNIQUE NOT NULL, tier TEXT NOT NULL, certifier TEXT NOT NULL,
  generation TEXT NOT NULL, space TEXT NOT NULL REFERENCES spaces(name), note TEXT, created TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'live', n_events INTEGER NOT NULL, sha256 TEXT NOT NULL);     -- live | rolled_back
CREATE TABLE IF NOT EXISTS summaries(
  key TEXT NOT NULL, bid INTEGER NOT NULL REFERENCES batches(bid), n INTEGER NOT NULL, mean BLOB NOT NULL,
  m2 BLOB NOT NULL, PRIMARY KEY(key, bid)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS summaries_bid ON summaries(bid);
CREATE TABLE IF NOT EXISTS key_space(key TEXT PRIMARY KEY, space TEXT NOT NULL REFERENCES spaces(name)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS leaves(
  key TEXT PRIMARY KEY, dim INTEGER NOT NULL, n INTEGER NOT NULL, mean BLOB NOT NULL, m2 BLOB NOT NULL,
  first_seen TEXT, last_seen TEXT);                                              -- permanent-tier aggregates only
CREATE TABLE IF NOT EXISTS nodes(
  id INTEGER PRIMARY KEY, parent INTEGER REFERENCES nodes(id), level INTEGER NOT NULL, name TEXT NOT NULL,
  batch TEXT NOT NULL REFERENCES batches(batch));                               -- shape persists only with provenance
CREATE TABLE IF NOT EXISTS centroids(
  generation TEXT NOT NULL, node INTEGER NOT NULL, n REAL NOT NULL, ls BLOB NOT NULL, ss REAL NOT NULL, built TEXT NOT NULL,
  PRIMARY KEY(generation, node));
"""
LIVE = "b.tier='permanent' AND b.status='live'"


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _blob(a):
    return np.ascontiguousarray(a, dtype=np.float64).tobytes()


def _arr(b):
    return np.frombuffer(b, dtype=np.float64).copy()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def combine(keys, X):
    """The combiner: per-key (n, mean, m2) of one batch's events, in first-appearance order. Pure; deterministic."""
    uk, first, inv = np.unique(np.asarray(keys, dtype=object), return_index=True, return_inverse=True)
    order = np.argsort(inv, kind="stable"); starts = np.r_[0, np.flatnonzero(np.diff(inv[order])) + 1]
    Xs = X[order]; n = np.diff(np.r_[starts, len(Xs)])
    mean = np.add.reduceat(Xs, starts, axis=0) / n[:, None]
    m2 = np.add.reduceat((Xs - mean[inv[order]]) ** 2, starts, axis=0)
    return {str(uk[i]): (int(n[i]), mean[i], m2[i]) for i in np.argsort(first, kind="stable")}


def chan(a, b):
    """Merge two Welford summaries (n, mean, m2) -- associative and commutative up to rounding."""
    na, ma, sa = a; nb, mb, sb = b; nn_ = na + nb; d = mb - ma
    return nn_, ma + d * nb / nn_, sa + sb + d * d * na * nb / nn_


def _read_cold(path):
    t = pq.read_table(path)
    if t.num_rows == 0:
        return [], np.zeros((0, 0)), []
    X = np.asarray(t.column("vec").combine_chunks().flatten(), dtype=np.float64).reshape(t.num_rows, -1)
    return t.column("key").to_pylist(), X, t.column("source").to_pylist()


def prepare_shard(rows, tmpdir=None):
    """THE MAP STEP (pure, runs in any worker): rows [(key, vec, source)] -> {path, sha256, n, dim, summaries}. The cold
    file is written to a temp path; add_prepared moves it into the store when the batch commits."""
    rows = [(str(k), np.asarray(v, dtype=np.float64), str(s)) for k, v, s in rows]
    dims = {r[1].shape for r in rows}
    if len(dims) > 1 or any(len(d) != 1 for d in dims):
        raise ValueError(f"rows of one batch must share one vector dim: {sorted(dims)[:3]}")
    dim = dims.pop()[0] if dims else 0
    keys = [r[0] for r in rows]; X = np.stack([r[1] for r in rows]) if rows else np.zeros((0, dim))
    vec = pa.FixedSizeListArray.from_arrays(pa.array(X.reshape(-1), pa.float64()), dim) if dim else pa.array([], pa.list_(pa.float64(), 1))
    fd, path = tempfile.mkstemp(suffix=".parquet", dir=tmpdir); os.close(fd)
    pq.write_table(pa.table({"key": pa.array(keys, pa.string()), "vec": vec, "source": pa.array([r[2] for r in rows], pa.string())}),
                   path, compression="zstd")
    return dict(path=path, sha256=_sha(path), n=len(rows), dim=dim, summaries=combine(keys, X) if rows else {})


class TreeDB:
    def __init__(self, path=DB):
        self.path = os.path.abspath(path); self.cold = self.path + ".events"
        os.makedirs(os.path.dirname(self.path), exist_ok=True); os.makedirs(self.cold, exist_ok=True)
        self.c = sqlite3.connect(self.path)
        self.c.execute("PRAGMA journal_mode=WAL"); self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.execute("PRAGMA foreign_keys = ON")
        if self.c.execute("SELECT name FROM sqlite_master WHERE name='events'").fetchone():
            raise ValueError(f"{path} is a v1 store; convert it with migrate_v1(src, dst)")
        self.c.executescript(SCHEMA)
        with self.c:
            self.c.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (VERSION,))
        v = self.c.execute("SELECT v FROM meta WHERE k='schema_version'").fetchone()[0]
        assert v == VERSION, f"{path}: schema version {v}, this code is {VERSION}"

    def close(self):
        self.c.close()

    def _cold_path(self, bid):
        return os.path.join(self.cold, f"{bid:08d}.parquet")

    # ---- spaces (the guard) -------------------------------------------------------------------------------------
    def register_space(self, name, kind, dim, meaning):
        assert kind in ("symbolic", "model"), kind
        r = self.c.execute("SELECT kind, dim FROM spaces WHERE name=?", (name,)).fetchone()
        if r is not None:
            if r != (kind, dim):
                raise ValueError(f"space {name!r} is already ({r[0]}, dim {r[1]}); spaces are never redefined")
            return
        with self.c:
            self.c.execute("INSERT INTO spaces VALUES (?,?,?,?,?)", (name, kind, dim, meaning, _now()))

    def _check_space(self, space, dim, keys):
        r = self.c.execute("SELECT kind, dim FROM spaces WHERE name=?", (space,)).fetchone()
        if r is None:
            raise ValueError(f"unregistered space {space!r}")
        if r[0] != "symbolic":
            raise ValueError(f"space {space!r} is kind {r[0]!r}: the permanent tree speaks only in symbols")
        if keys and dim != r[1]:
            raise ValueError(f"space {space!r} has dim {r[1]}; batch rows have dim {dim}")
        keys = list(keys)
        for i in range(0, len(keys), 500):
            ch = keys[i:i + 500]
            clash = self.c.execute(f"SELECT key, space FROM key_space WHERE space != ? AND key IN ({','.join('?' * len(ch))})",
                                   (space, *ch)).fetchall()
            if clash:
                raise ValueError(f"keys already live in another space: {clash[:3]}")

    # ---- writes ---------------------------------------------------------------------------------------------------
    def add_evidence(self, batch, rows, tier, certifier, generation, space, note=None):
        """rows: iterable of (key, vec, source) in a registered SYMBOLIC space. One batch = one transaction; a batch
        name is never reused; a key lives in one space only. (= prepare_shard + add_prepared, in this process.)"""
        return self.add_prepared(batch, prepare_shard(rows, self.cold), tier, certifier, generation, space, note)

    def add_prepared(self, batch, prep, tier, certifier, generation, space, note=None, created=None):
        """THE SINGLE WRITER: commit one prepared shard as one batch (atomic: the batch row, its summaries and its cold
        file land together or not at all)."""
        assert tier in TIERS, tier
        dst = None
        try:
            if _sha(prep["path"]) != prep["sha256"]:
                raise ValueError(f"batch {batch!r}: cold file hash does not match its prepared hash")
            self._check_space(space, prep["dim"], prep["summaries"])
            with self.c:
                now = created or _now()
                bid = self.c.execute("INSERT INTO batches(batch, tier, certifier, generation, space, note, created, n_events, sha256) "
                                     "VALUES (?,?,?,?,?,?,?,?,?)",
                                     (batch, tier, certifier, generation, space, note, now, prep["n"], prep["sha256"])).lastrowid
                self.c.executemany("INSERT INTO summaries VALUES (?,?,?,?,?)",
                                   [(k, bid, n, _blob(m), _blob(s)) for k, (n, m, s) in prep["summaries"].items()])
                self.c.executemany("INSERT OR IGNORE INTO key_space VALUES (?,?)", [(k, space) for k in prep["summaries"]])
                if tier == "permanent":
                    self._refold(list(prep["summaries"]))
                dst = self._cold_path(bid); os.replace(prep["path"], dst)
        except BaseException:
            if dst is not None and os.path.exists(dst):
                os.remove(dst)                                               # the transaction rolled back: so does the file
            if os.path.exists(prep["path"]):
                os.remove(prep["path"])
            raise
        return prep["n"]

    def _refold(self, keys):
        """Each key's leaf := the Chan left-fold of its live permanent summaries in batch order (inside the caller's
        transaction). Deterministic: incremental writes and rebuild_all produce the same bits."""
        for i in range(0, len(keys), 500):
            ch = keys[i:i + 500]; q = ",".join("?" * len(ch))
            self.c.execute(f"DELETE FROM leaves WHERE key IN ({q})", ch)
            acc = {}
            for k, n, m, s, t in self.c.execute(f"""SELECT s.key, s.n, s.mean, s.m2, b.created FROM summaries s JOIN batches b
                                                    ON s.bid=b.bid WHERE s.key IN ({q}) AND {LIVE} ORDER BY s.key, s.bid""", ch):
                cur = (n, _arr(m), _arr(s))
                if k in acc:
                    a = acc[k]; acc[k] = (chan(a[0], cur), a[1], t)
                else:
                    acc[k] = (cur, t, t)
            self.c.executemany("INSERT INTO leaves VALUES (?,?,?,?,?,?,?)",
                               [(k, len(f[1]), f[0], _blob(f[1]), _blob(f[2]), t0, t1) for k, (f, t0, t1) in acc.items()])

    def _batch(self, batch):
        r = self.c.execute("SELECT bid, tier, status FROM batches WHERE batch=?", (batch,)).fetchone()
        assert r is not None, f"no batch {batch!r}"
        return r

    def _batch_keys(self, bid):
        return [r[0] for r in self.c.execute("SELECT key FROM summaries WHERE bid=?", (bid,))]

    def promote(self, batch):
        """provisional -> permanent: the batch's summaries enter the fold."""
        bid, tier, status = self._batch(batch)
        assert (tier, status) == ("provisional", "live"), f"cannot promote {batch}: {(tier, status)}"
        with self.c:
            self.c.execute("UPDATE batches SET tier='permanent' WHERE bid=?", (bid,))
            self._refold(self._batch_keys(bid))

    def rollback(self, batch):
        """Drop a batch from the fold and refold every key it touched. The batch row and its cold file stay, marked
        rolled_back; a key no live batch still holds releases its space claim."""
        bid, _, status = self._batch(batch)
        assert status == "live", f"cannot roll back {batch}: {status}"
        keys = self._batch_keys(bid)
        with self.c:
            self.c.execute("UPDATE batches SET status='rolled_back' WHERE bid=?", (bid,))
            self.c.execute("DELETE FROM nodes WHERE batch=?", (batch,))
            for i in range(0, len(keys), 500):
                ch = keys[i:i + 500]; q = ",".join("?" * len(ch))
                self.c.execute(f"""DELETE FROM key_space WHERE key IN ({q}) AND key NOT IN (SELECT s.key FROM summaries s
                                   JOIN batches b ON s.bid=b.bid WHERE b.status='live' AND s.key IN ({q}))""", ch + ch)
            self._refold(keys)
        return keys

    def rebuild_all(self):
        """THE AUDIT: every live batch's cold file re-hashed, every summary recomputed from its events (must be
        bit-exact with the stored one), every leaf refolded. Raises on any mismatch. Returns the number of events read."""
        n_ev = 0
        for bid, batch, sha, n in self.c.execute("SELECT bid, batch, sha256, n_events FROM batches WHERE status='live' ORDER BY bid").fetchall():
            p = self._cold_path(bid)
            if _sha(p) != sha:
                raise ValueError(f"batch {batch!r}: cold file {p} does not match its recorded SHA-256")
            keys, X, _ = _read_cold(p)
            if len(keys) != n:
                raise ValueError(f"batch {batch!r}: {len(keys)} events on disk, {n} recorded")
            fresh = combine(keys, X) if keys else {}
            stored = {k: (c, _arr(m), _arr(s)) for k, c, m, s in self.c.execute("SELECT key, n, mean, m2 FROM summaries WHERE bid=?", (bid,))}
            if set(fresh) != set(stored) or any(fresh[k][0] != stored[k][0] or fresh[k][1].tobytes() != stored[k][1].tobytes()
                                                or fresh[k][2].tobytes() != stored[k][2].tobytes() for k in fresh):
                raise ValueError(f"batch {batch!r}: summaries recomputed from its events differ from the stored ones")
            n_ev += len(keys)
        with self.c:
            keys = [r[0] for r in self.c.execute("SELECT key FROM summaries UNION SELECT key FROM leaves")]
            self._refold(sorted(keys))
        return n_ev

    # ---- reads ----------------------------------------------------------------------------------------------------
    def leaf(self, key):
        """(n, mean, var) of a key's permanent evidence, or None."""
        r = self.c.execute("SELECT n, mean, m2 FROM leaves WHERE key=?", (key,)).fetchone()
        if r is None:
            return None
        n, mean, m2 = r[0], _arr(r[1]), _arr(r[2])
        return n, mean, m2 / max(n - 1, 1)

    def load_leaves(self):
        """The run's working set: {key: (n, mean)} for every leaf (one query; the breath never touches SQLite)."""
        return {k: (n, _arr(m)) for k, n, m in self.c.execute("SELECT key, n, mean FROM leaves")}

    def batch_rows(self, batch):
        """[(key, vec, source)] of one batch (e.g. a provisional batch under audition), read from its cold file."""
        keys, X, src = _read_cold(self._cold_path(self._batch(batch)[0]))
        return list(zip(keys, X, src))

    def batch_summaries(self, batch):
        """{key: (n, mean, m2)} of one batch -- its combiner output (an audition can read this instead of the events)."""
        return {k: (n, _arr(m), _arr(s)) for k, n, m, s in
                self.c.execute("SELECT key, n, mean, m2 FROM summaries WHERE bid=?", (self._batch(batch)[0],))}

    def batches(self):
        return self.c.execute("SELECT batch, tier, certifier, generation, space, status FROM batches ORDER BY created, batch").fetchall()

    def n_events(self, live_only=True):
        return self.c.execute(f"SELECT COALESCE(SUM(n_events), 0) FROM batches {'WHERE status=' + repr('live') if live_only else ''}").fetchone()[0]

    def provenance(self, key):
        """[(source, batch, tier, certifier, generation, recorded)] of every live event behind `key` (cold read of only
        the batches whose summaries hold it)."""
        out = []
        for bid, batch, tier, cert, gen, created in self.c.execute(
                """SELECT b.bid, b.batch, b.tier, b.certifier, b.generation, b.created FROM summaries s JOIN batches b
                   ON s.bid=b.bid WHERE s.key=? AND b.status='live' ORDER BY b.bid""", (key,)).fetchall():
            keys, _, src = _read_cold(self._cold_path(bid))
            out += [(s, batch, tier, cert, gen, created) for k, s in zip(keys, src) if k == key]
        return out

    # ---- shape ----------------------------------------------------------------------------------------------------
    def add_node(self, node_id, parent, level, name, batch):
        with self.c:
            self.c.execute("INSERT INTO nodes VALUES (?,?,?,?,?)", (node_id, parent, level, name, batch))

    def nodes(self):
        return self.c.execute("SELECT id, parent, level, name, batch FROM nodes ORDER BY id").fetchall()

    # ---- centroids (generation-tagged; never mixed) --------------------------------------------------------------
    def save_centroids(self, generation, N, LS, SS):
        with self.c:
            self.c.execute("DELETE FROM centroids WHERE generation=?", (generation,))
            self.c.executemany("INSERT INTO centroids VALUES (?,?,?,?,?,?)",
                               [(generation, i, float(N[i]), _blob(LS[i]), float(SS[i]), _now()) for i in range(len(N))])

    def load_centroids(self, generation, current):
        """BIRCH CFs (N, LS, SS) of `generation`; refuses unless it IS the current generation."""
        if generation != current:
            raise ValueError(f"never mix generations: centroids of {generation!r} requested by generation {current!r}")
        rs = self.c.execute("SELECT n, ls, ss FROM centroids WHERE generation=? ORDER BY node", (generation,)).fetchall()
        if not rs:
            return None
        return np.array([r[0] for r in rs]), np.stack([_arr(r[1]) for r in rs]), np.array([r[2] for r in rs])


def migrate_v1(src, dst):
    """Re-write a v1 store (events table) as v2, batch by batch in insertion order. Rolled-back v1 batches had their
    events deleted, so they arrive as empty rolled-back batches. Returns the new TreeDB."""
    assert not os.path.exists(dst), f"refusing to overwrite {dst}"
    old = sqlite3.connect(f"file:{os.path.abspath(src)}?mode=ro", uri=True); db = TreeDB(dst)
    for name, kind, dim, meaning, _ in old.execute("SELECT * FROM spaces ORDER BY rowid"):
        db.register_space(name, kind, dim, meaning)
    for batch, tier, cert, gen, space, note, created, status in old.execute(
            "SELECT batch, tier, certifier, generation, space, note, created, status FROM batches ORDER BY rowid").fetchall():
        rows = [(k, _arr(v), s) for k, v, s in old.execute("SELECT key, vec, source FROM events WHERE batch=? ORDER BY id", (batch,))]
        if not rows:
            assert status == "rolled_back", f"v1 batch {batch!r} is live with no events"
        db.add_prepared(batch, prepare_shard(rows, db.cold), tier, cert, gen, space, note, created=created)
        if status == "rolled_back":
            db.rollback(batch)
    with db.c:
        db.c.executemany("INSERT INTO nodes VALUES (?,?,?,?,?)", old.execute("SELECT * FROM nodes ORDER BY id").fetchall())
        db.c.executemany("INSERT INTO centroids VALUES (?,?,?,?,?,?)", old.execute("SELECT * FROM centroids").fetchall())
    old.close()
    return db
