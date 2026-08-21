"""waist_miner.py — THE WAIST-PATTERN LEDGER (2026-08-20, the gut registered):
the schema miner lifted to the 512-d waist grain. Pools per-row silhouettes
from banked trunk states, clusters them online (leader algorithm, cosine),
and keeps Welford running stats (mean/m2) per cluster in campaign.db's
waist_patterns table — cheap matching, no recompute. Common clusters get
marked macro_candidate='PROPOSED'; the rank-never-admit door (elsewhere)
decides real admission into the macro registry.

FENCES (carved at registration, do not soften):
  (1) HARVEST-ONLY. The bank is the mouth's own lawful pattern: mining
      reads train/harvest states only. A test/bigtest source would let the
      miner's "recurring pattern" signal leak backward into what gets
      annotated next — contamination, not measurement. Asserted at load.
  (2) MATCHING ACCELERATES, NEVER DECIDES. Cosine-cluster membership is a
      similarity heuristic for counting, not a claim about correctness
      (temperature-perp-truth's cousin: a familiar pattern is not a
      correct parse). Nothing here touches the solver or the gate.
  (3) MACROS ADMITTED ONLY VIA THE REGISTRY DOOR. This script writes
      macro_candidate='PROPOSED' and nothing stronger. Admission into
      mycelium/campaign_db.py's macros table is a separate, deliberate act
      (rank-never-admit: proposal by counting, admission by design).

CLI (env-driven, zero-GPU, CPU-only):
  MINER_SRC      path to a banked states .npz (required)
  MINER_NPY      path to the paired _states.npy memmap (optional; derived
                 from MINER_SRC if omitted and the npz has no "states" key)
  MINER_REGISTER label string recorded against clusters touched this run
  MINER_CAP      optional row cap (smoke runs)
"""
import os
import sys
import sqlite3
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mycelium.campaign_db import conn  # noqa: E402

D_IN = 2048          # pooled trunk state width
D_WAIST = 512        # projected waist-grain width ("the waist grain")
COS_THRESH = 0.92    # leader-algorithm admission cosine
CLUSTER_CAP = 4096   # stop CREATING clusters past this; matching continues
FLUSH_EVERY = 500    # batch DB writes


def _now():
    return datetime.now(timezone.utc).isoformat()


def _assert_harvest_only(*paths):
    """FENCE (1): never mine a test/bigtest npz — harvest/train sources only."""
    for p in paths:
        if p is None:
            continue
        low = os.path.basename(p).lower()
        assert "test" not in low and "big" not in low, (
            f"waist_miner: refusing non-harvest source {p!r} "
            "(contamination fence — mine train/harvest states only)"
        )


def load_pool(src, npy=None, cap=None):
    """Load a banked states npz and return the L2-normalized 2048-d pooled
    silhouette per row: masked mean over tokens of the trunk state. Follows
    the pooled_npz idiom used across entourage scripts (e.g. entourage22.py)."""
    _assert_harvest_only(src, npy)
    z = np.load(src)
    if "states" in z.files:
        st = z["states"]
    else:
        if npy is None:
            npy = src[:-4] + "_states.npy" if src.endswith(".npz") else src + "_states.npy"
        _assert_harvest_only(npy)
        st = np.load(npy, mmap_mode="r")
    tk = z["tokmask"]
    n = st.shape[0] if cap is None else min(st.shape[0], int(cap))
    out = np.zeros((n, st.shape[2]), np.float32)
    for s0 in range(0, n, 256):
        sl = slice(s0, min(s0 + 256, n))
        a = np.asarray(st[sl]).astype(np.float32)
        m = tk[sl].astype(np.float32)
        out[sl] = (a * m[:, :, None]).sum(1) / np.maximum(m.sum(1)[:, None], 1)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.maximum(norms, 1e-8)
    return out


def project(vecs, seed=41, d_out=D_WAIST):
    """Fixed random projection 2048 -> 512 ("the waist grain" for mining).
    Seeded 41 so every mining run shares the same subspace — cluster means
    persisted in the DB stay comparable across runs/sources."""
    d_in = vecs.shape[1]
    rng = np.random.RandomState(seed)
    proj = rng.randn(d_in, d_out).astype(np.float32) / np.sqrt(d_in)
    out = vecs @ proj
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-8)


class ClusterBank:
    """Leader-algorithm online clustering with Welford stats, backed by
    campaign.db's waist_patterns table. Loads existing clusters at start
    (so mining is cumulative across runs/sources — count is a library-wide
    census, not a per-run one) and flushes updates back in batches.

    Cluster means/m2 live in preallocated (cap, D_WAIST) arrays so that
    matching is one matvec per row (no Python-level restacking) and new-
    cluster creation is O(D) (fill the next free row), not O(K*D)."""

    def __init__(self, db, register, cap=CLUSTER_CAP, thresh=COS_THRESH):
        self.db = db
        self.register = register
        self.cap = cap
        self.thresh = thresh
        rows = db.execute(
            "SELECT cluster_id, count, mean, m2 FROM waist_patterns ORDER BY cluster_id"
        ).fetchall()
        n0 = len(rows)
        alloc = max(cap, n0)
        self.M = np.zeros((alloc, D_WAIST), np.float32)      # cluster means
        self.M2 = np.zeros((alloc, D_WAIST), np.float32)     # Welford m2
        self.norms = np.zeros(alloc, np.float32)
        self.counts = np.zeros(alloc, np.int64)
        self.ids = []             # index -> cluster_id
        id_to_idx = {}
        for i, (cid, count, mean_b, m2_b) in enumerate(rows):
            self.M[i] = np.frombuffer(mean_b, dtype=np.float32)
            self.M2[i] = np.frombuffer(m2_b, dtype=np.float32)
            self.counts[i] = count
            self.norms[i] = max(np.linalg.norm(self.M[i]), 1e-8)
            self.ids.append(cid)
            id_to_idx[cid] = i
        self.id_to_idx = id_to_idx
        self.K = n0                                    # used rows
        self.next_id = (max(self.ids) + 1) if self.ids else 0
        self.new_this_run = 0
        self.dirty = set()    # indices touched since last flush
        self.new_idx = set()  # indices created this run (need INSERT not UPDATE)

    def assign(self, x):
        """x: unit-norm 512-d vector. Returns the cluster_id it lands in."""
        K = self.K
        if K:
            sims = (self.M[:K] @ x) / self.norms[:K]
            j = int(np.argmax(sims))
            best = float(sims[j])
        else:
            j, best = -1, -1.0

        if best >= self.thresh or (K >= self.cap and j >= 0):
            # matched (or forced-match at the cap — matching accelerates,
            # never decides membership creation once the cap is hit)
            self._update(j, x)
            return self.ids[j]

        # create a new cluster (only reachable while K < cap)
        i = self.K
        if i >= self.M.shape[0]:  # grow the backing store past its initial size
            grow = np.zeros((max(1024, self.M.shape[0]), D_WAIST), np.float32)
            self.M = np.concatenate([self.M, grow], axis=0)
            self.M2 = np.concatenate([self.M2, grow], axis=0)
            self.norms = np.concatenate([self.norms, np.zeros(grow.shape[0], np.float32)])
            self.counts = np.concatenate([self.counts, np.zeros(grow.shape[0], np.int64)])
        cid = self.next_id
        self.next_id += 1
        self.M[i] = x
        self.M2[i] = 0.0
        self.norms[i] = 1.0  # x is unit-norm
        self.counts[i] = 1
        self.ids.append(cid)
        self.id_to_idx[cid] = i
        self.K += 1
        self.new_idx.add(i)
        self.dirty.add(i)
        self.new_this_run += 1
        return cid

    def _update(self, j, x):
        """Welford online update for cluster row j."""
        self.counts[j] += 1
        n = self.counts[j]
        mean = self.M[j]
        delta = x - mean
        mean = mean + delta / n
        delta2 = x - mean
        self.M2[j] = self.M2[j] + delta * delta2
        self.M[j] = mean
        self.norms[j] = max(float(np.linalg.norm(mean)), 1e-8)
        self.dirty.add(j)

    def flush(self):
        if not self.dirty:
            return
        now = _now()
        for j in self.dirty:
            cid = self.ids[j]
            mean_b = self.M[j].astype(np.float32).tobytes()
            m2_b = self.M2[j].astype(np.float32).tobytes()
            count = int(self.counts[j])
            if j in self.new_idx:
                self.db.execute(
                    "INSERT INTO waist_patterns"
                    "(cluster_id, count, mean, m2, register, first_seen, last_seen, macro_candidate)"
                    " VALUES (?,?,?,?,?,?,?,NULL)"
                    " ON CONFLICT(cluster_id) DO UPDATE SET"
                    " count=excluded.count, mean=excluded.mean, m2=excluded.m2,"
                    " register=excluded.register, last_seen=excluded.last_seen",
                    (cid, count, mean_b, m2_b, self.register, now, now),
                )
            else:
                self.db.execute(
                    "UPDATE waist_patterns SET count=?, mean=?, m2=?, register=?, last_seen=?"
                    " WHERE cluster_id=?",
                    (count, mean_b, m2_b, self.register, now, cid),
                )
        self.db.commit()
        self.new_idx -= self.dirty  # rows just flushed are now persisted (UPDATE next time)
        self.dirty.clear()


def mine(src, npy=None, register="unspecified", cap=None):
    vecs2048 = load_pool(src, npy, cap)
    vecs = project(vecs2048)
    db = conn()
    bank = ClusterBank(db, register)
    n = len(vecs)
    for i in range(n):
        bank.assign(vecs[i])
        if (i + 1) % FLUSH_EVERY == 0:
            bank.flush()
    bank.flush()
    census(db, n, bank.new_this_run)
    db.close()
    return n, bank.new_this_run


def census(db, n_mined, clusters_created):
    total_rows = db.execute("SELECT count(*) FROM waist_patterns").fetchone()[0]
    top = db.execute(
        "SELECT cluster_id, count, m2 FROM waist_patterns ORDER BY count DESC LIMIT 10"
    ).fetchall()
    print(f"=== waist_miner census ===")
    print(f"rows mined this run: {n_mined}")
    print(f"clusters created this run: {clusters_created}")
    print(f"clusters in library (total): {total_rows}")
    print(f"top-10 clusters by count:")
    for cid, count, m2_b in top:
        m2 = np.frombuffer(m2_b, dtype=np.float32)
        std = float(np.sqrt(np.maximum(m2 / max(count, 1), 0)).mean())
        print(f"  cluster {cid:5d}  count={count:8d}  within-cluster std={std:.5f}")

    # FENCE (3): propose only. macro_candidate='PROPOSED' is a flag for the
    # rank-never-admit door elsewhere to consider — never an admission.
    thresh_count = max(1, int(0.01 * total_rows))
    proposed = db.execute(
        "SELECT cluster_id FROM waist_patterns WHERE count >= ?", (thresh_count,)
    ).fetchall()
    db.executemany(
        "UPDATE waist_patterns SET macro_candidate='PROPOSED' WHERE cluster_id=?",
        proposed,
    )
    db.commit()
    print(f"macro_candidate='PROPOSED' on {len(proposed)} clusters "
          f"(count >= {thresh_count} = 1% of {n_mined} mined rows)")


if __name__ == "__main__":
    src = os.environ["MINER_SRC"]
    npy = os.environ.get("MINER_NPY")
    register = os.environ.get("MINER_REGISTER", "unspecified")
    cap = os.environ.get("MINER_CAP")
    cap = int(cap) if cap else None
    mine(src, npy, register, cap)
