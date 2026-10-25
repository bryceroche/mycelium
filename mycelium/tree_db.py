"""tree_db.py — THE PERMANENT TREE (2026-10-25, Bryce's word: "word given for the sqlite db -- salute!"). SQLite,
embedded, .cache-resident, no daemons (campaign_db.py's pattern).
The rulings it encodes (ledger 2026-10-25, T2p/T2q/T2r):
  - ROWS ARE KEPT FOREVER; the tree only accumulates evidence. The EVENTS table is append-only and is the source of truth;
    every leaf aggregate (Welford n / mean / m2) is a materialization, always rebuildable from the events.
  - PROVENANCE: every event names its source (episode/mention), its batch, its certifier and its generation, so a bad
    batch is rolled back by deleting its events and re-aggregating the keys it touched.
  - CUSTODY TIERS: 'permanent' (answer-key-certified) events enter the aggregates; 'provisional' (test-time) events are
    stored but stay OUT of the aggregates until their batch is promoted.
  - CENTROIDS ARE REBUILT PER RUN (never mix generations' coordinates): interior BIRCH cluster features may be cached
    tagged by generation, and a load for any other generation raises.
  - TREE SHAPE persists only with provenance: a node row names the batch that certified it.
The breath never queries SQLite: a run loads the leaves into memory (`load_leaves`) and snaps against in-memory centroids."""
import os, sqlite3, time
import numpy as np

DB = os.path.join(os.path.dirname(__file__), "..", ".cache", "tree.db")
TIERS = ("permanent", "provisional")

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches(
  batch TEXT PRIMARY KEY, tier TEXT NOT NULL, certifier TEXT NOT NULL, generation TEXT NOT NULL,
  note TEXT, created TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'live');      -- live | rolled_back
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, vec BLOB NOT NULL, source TEXT NOT NULL,
  batch TEXT NOT NULL REFERENCES batches(batch), recorded TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_key ON events(key);
CREATE INDEX IF NOT EXISTS events_batch ON events(batch);
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


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _blob(a):
    return np.ascontiguousarray(a, dtype=np.float64).tobytes()


def _arr(b):
    return np.frombuffer(b, dtype=np.float64).copy()


class TreeDB:
    def __init__(self, path=DB):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.c = sqlite3.connect(path)
        self.c.execute("PRAGMA foreign_keys = ON")
        self.c.executescript(SCHEMA)

    def close(self):
        self.c.close()

    # ---- evidence -------------------------------------------------------------------------------------------------
    def add_evidence(self, batch, rows, tier, certifier, generation, note=None):
        """rows: iterable of (key, vec, source). One batch = one transaction; a batch name is never reused."""
        assert tier in TIERS, tier
        rows = [(str(k), np.asarray(v, dtype=np.float64), str(s)) for k, v, s in rows]
        with self.c:
            self.c.execute("INSERT INTO batches(batch, tier, certifier, generation, note, created) VALUES (?,?,?,?,?,?)",
                           (batch, tier, certifier, generation, note, _now()))
            now = _now()
            self.c.executemany("INSERT INTO events(key, vec, source, batch, recorded) VALUES (?,?,?,?,?)",
                               [(k, _blob(v), s, batch, now) for k, v, s in rows])
            if tier == "permanent":
                self._merge(rows, now)
        return len(rows)

    def _merge(self, rows, now):
        """Chan/Welford merge of each key's new vectors into its aggregate (inside the caller's transaction)."""
        by = {}
        for k, v, _ in rows:
            by.setdefault(k, []).append(v)
        for k, vs in by.items():
            X = np.stack(vs); kb = len(X); mb = X.mean(0); m2b = ((X - mb) ** 2).sum(0)
            r = self.c.execute("SELECT dim, n, mean, m2 FROM leaves WHERE key=?", (k,)).fetchone()
            if r is None:
                self.c.execute("INSERT INTO leaves VALUES (?,?,?,?,?,?,?)", (k, X.shape[1], kb, _blob(mb), _blob(m2b), now, now))
                continue
            dim, n, mean, m2 = r[0], r[1], _arr(r[2]), _arr(r[3])
            assert dim == X.shape[1], f"key {k}: dim {X.shape[1]} != stored {dim}"
            nn_ = n + kb; d = mb - mean
            mean = mean + d * kb / nn_; m2 = m2 + m2b + d * d * n * kb / nn_
            self.c.execute("UPDATE leaves SET n=?, mean=?, m2=?, last_seen=? WHERE key=?", (nn_, _blob(mean), _blob(m2), now, k))

    def _reaggregate(self, keys):
        """Rebuild the aggregates of `keys` from their live permanent events (inside the caller's transaction)."""
        for k in keys:
            self.c.execute("DELETE FROM leaves WHERE key=?", (k,))
            ev = self.c.execute("""SELECT e.vec, e.recorded FROM events e JOIN batches b ON e.batch=b.batch
                                   WHERE e.key=? AND b.tier='permanent' AND b.status='live' ORDER BY e.id""", (k,)).fetchall()
            if not ev:
                continue
            X = np.stack([_arr(v) for v, _ in ev]); m = X.mean(0)
            self.c.execute("INSERT INTO leaves VALUES (?,?,?,?,?,?,?)",
                           (k, X.shape[1], len(X), _blob(m), _blob(((X - m) ** 2).sum(0)), ev[0][1], ev[-1][1]))

    def _batch_keys(self, batch):
        return [r[0] for r in self.c.execute("SELECT DISTINCT key FROM events WHERE batch=?", (batch,))]

    def promote(self, batch):
        """provisional -> permanent: the batch's events enter the aggregates."""
        with self.c:
            r = self.c.execute("SELECT tier, status FROM batches WHERE batch=?", (batch,)).fetchone()
            assert r == ("provisional", "live"), f"cannot promote {batch}: {r}"
            self.c.execute("UPDATE batches SET tier='permanent' WHERE batch=?", (batch,))
            self._reaggregate(self._batch_keys(batch))

    def rollback(self, batch):
        """Remove a batch's evidence and re-aggregate every key it touched. The batch row stays, marked rolled_back."""
        with self.c:
            r = self.c.execute("SELECT status FROM batches WHERE batch=?", (batch,)).fetchone()
            assert r == ("live",), f"cannot roll back {batch}: {r}"
            keys = self._batch_keys(batch)
            self.c.execute("UPDATE batches SET status='rolled_back' WHERE batch=?", (batch,))
            self.c.execute("DELETE FROM nodes WHERE batch=?", (batch,))
            self.c.execute("DELETE FROM events WHERE batch=?", (batch,))
            self._reaggregate(keys)
        return keys

    def rebuild_all(self):
        """Recompute every aggregate from the events (the audit: must equal the incremental materialization)."""
        with self.c:
            keys = [r[0] for r in self.c.execute("SELECT DISTINCT key FROM events")] + \
                   [r[0] for r in self.c.execute("SELECT key FROM leaves")]
            self._reaggregate(sorted(set(keys)))

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

    def provenance(self, key):
        return self.c.execute("""SELECT e.source, e.batch, b.tier, b.certifier, b.generation, e.recorded FROM events e
                                 JOIN batches b ON e.batch=b.batch WHERE e.key=? ORDER BY e.id""", (key,)).fetchall()

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
