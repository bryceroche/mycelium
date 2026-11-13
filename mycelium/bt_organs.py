"""bt_organs.py -- THE ORGANS OUTSIDE THE BREATHING TRANSFORMER (ledger 2026-11-07 "THE TIERED BT ARCHITECTURE").

The BT (`mycelium/breathing_transformer.py`) is the compute engine; everything that persists, ticks, certifies or is stored lives here:
  - T6Clock / T9Clock: the breath rotor and the token-address rotor. Each has BOTH faces (the bridge): the continuous gears (fields
    for the BT's KeyLayout, turned into angles there) and the discrete ODOMETER (integer digits: the exact form for Tree keys).
    Speeds come from `rotor_clock` (one source); the clock-range rule is checked at construction.
  - Notebook: the per-problem slot table the solver writes back to (committed atom, radius = confidence, certificate features,
    the T6 stamp of the commit, the tree level, consumed marks). Certificates, never raw values, enter the reasoning state.
  - PhaseBus: the T256 rotational bus: bind by phase addition, unbind by the conjugate (never 1/z), codebook cleanup.
  - ChartTable / TreeTables: the Tree DB's chart tables, partitioned by (tier, head): head h of a tier's BT reads partition
    (tier, 'h{h}') on every charted road (3 tiers x 6 heads = 18 partitions, each its own centroids). A partition is maintained by
    WELFORD stats (n, mean, M2 per chart; batches merged by Chan's formula; n capped so the charts follow a still-training
    projection) and BIRCH math (the clustering features N, LS, SS; the threshold T grows from 0 by rebuild: a point beyond T
    spawns a chart, a full partition merges its two closest charts first). Only CERTIFIED points feed it (custody: a correct chain). Centroids are MODEL-space,
    so they live only in the store's generation-tagged chart tables (never mixed across generations); the symbolic write-back
    (which chart a certified key took) goes through custody as evidence ('provisional' until the answer key promotes
    it). The breath never queries SQLite: a run loads its tables into memory and hands them to the BT as data.
  - Solver (Protocol) + TierPort (Protocol) + Conductor: THE ALTERNATION. Each breath, every tier's BT proposes (coarse tier
    first), the solver disposes and writes back to the notebook, the clock ticks, the temperature falls. Tiers talk only through
    the notebook and the bus (the residual-seal law: no shared residual between tiers).
"""
from dataclasses import dataclass, field
import functools, math
from typing import Protocol

import numpy as np
from tinygrad import Tensor

from mycelium import rotor_clock as CK


# ---- THE CLOCKS ----------------------------------------------------------------------------------------------------------------------
class T6Clock:
    """The breath rotor. tick() advances one breath; fields() feeds the BT's 't6' factor; odometer() = the discrete face."""
    def __init__(self, speeds=CK.T6_SPEEDS, span=None, field_name="tick"):
        self.speeds, self.field = tuple(speeds), field_name
        if span is not None: CK.check_range(self.speeds, span)
        self.t = 0

    def tick(self):
        self.t += 1; return self.t

    def reset(self, t=0):
        self.t = t

    def fields(self, shape, t=None):
        return {self.field: np.full(shape, self.t if t is None else t, np.int64)}

    def odometer(self, t=None):
        return CK.t6_odometer(self.t if t is None else t, self.speeds)


class T9Clock:
    """The token-address rotor (the T9-NS reader's clock, RoPE retired 11-03): the six gears over the offset within the sentence +
    the slow gears over the absolute position. Sentence planes stay quiet (NS)."""
    def __init__(self, speeds=CK.T6_SPEEDS, slow_periods=(192.0, 768.0, 3072.0), slow_bins=4, off_field="off", pos_field="pos"):
        self.speeds, self.slow = tuple(speeds), tuple(float(p) for p in slow_periods)
        self.slow_bins, self.off_field, self.pos_field = slow_bins, off_field, pos_field

    def slow_speeds(self):
        return tuple(360.0 / p for p in self.slow)

    def fields(self, sent):
        """sent (B, L) int sentence ids -> {off, pos}: the offset since the sentence began and the absolute position"""
        sent = np.asarray(sent); B, L = sent.shape; i = np.broadcast_to(np.arange(L), (B, L))
        chg = np.ones_like(sent, bool); chg[:, 1:] = sent[:, 1:] != sent[:, :-1]
        off = i - np.maximum.accumulate(np.where(chg, i, 0), 1)
        return {self.off_field: off.astype(np.int64), self.pos_field: i.astype(np.int64).copy()}

    def odometer(self, off, pos):
        """(offset digits on the six gears) + (one bin per slow gear): the token's exact address"""
        slow = tuple(int((pos % p) * self.slow_bins // p) for p in self.slow)
        return CK.t6_odometer(off, self.speeds) + slow


# ---- THE NOTEBOOK ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Notebook:
    """Per-batch slot table. Arrays are numpy (host); the solver writes, the BT reads (enter_slots + KeyLayout fields)."""
    B: int
    K: int
    Lt: int
    d_cert: int = 3
    committed: np.ndarray = None
    atom: np.ndarray = None
    radius: np.ndarray = None
    cert: np.ndarray = None
    tick: np.ndarray = None
    level: np.ndarray = None
    consumed: np.ndarray = None                # (B, Lt) tokens consumed by committed slots
    breaker: np.ndarray = None                 # (B, K, Lt) the solver's circuit breaker: a PROOF closed this (slot, token)

    def __post_init__(self):
        B, K, Lt = self.B, self.K, self.Lt
        self.committed = np.zeros((B, K), bool); self.atom = np.zeros((B, K), np.int64); self.radius = np.zeros((B, K), np.float32)
        self.cert = np.zeros((B, K, self.d_cert), np.float32); self.tick = np.zeros((B, K), np.int64)
        self.level = np.zeros((B, K), np.int64); self.consumed = np.zeros((B, Lt), bool); self.breaker = np.zeros((B, K, Lt), bool)

    def write(self, b, k, atom, radius, cert, tick, level=0, consumes=()):
        """A commit (the solver's write-back): certificates, never raw values."""
        self.committed[b, k] = True; self.atom[b, k] = atom; self.radius[b, k] = radius
        self.cert[b, k] = cert; self.tick[b, k] = tick; self.level[b, k] = level
        for i in consumes: self.consumed[b, i] = True

    def close(self, b, k, i):
        """The circuit breaker: only a proof (UNSAT) closes a (slot, token) pair."""
        self.breaker[b, k, i] = True

    def slot_fields(self, beam=None):
        """The slots' composite-key fields (slot index, T6 stamp, level, beam)."""
        f = {"slot": np.broadcast_to(np.arange(self.K), (self.B, self.K)).copy(), "tick": self.tick.copy(), "level": self.level.copy()}
        if beam is not None: f["beam"] = np.broadcast_to(np.asarray(beam)[:, None], (self.B, self.K)).copy()
        return f

    def tensors(self):
        """-> kwargs for BreathingTransformer.enter_slots (atoms/belief come from the Tree / the last breath)"""
        return dict(committed=Tensor(self.committed), cert=Tensor(self.cert), atom=Tensor(self.atom), radius=Tensor(self.radius))


# ---- THE T256 BUS ----------------------------------------------------------------------------------------------------------------------
class PhaseBus:
    """The rotational bus: P planes per problem. write(key, value) superposes r e^{i(key + value)}; read(key) unbinds with the
    conjugate e^{-i key}; cleanup snaps the read to the nearest codebook entry (crosstalk ~ 1/sqrt(P) per stored pair).
    Keys / values may carry extra item axes: (B, ..., P) (e.g. one key per slot)."""
    def __init__(self, B, P=256):
        self.P = P; self.M = np.zeros((B, P), np.complex128)

    @staticmethod
    def phases(ids, P=256, salt=0):
        """int ids (...,) -> (..., P) quasi-random angles (Weyl: plane p turns frac(id x frac(sqrt(prime_p))); salt picks another
        code, e.g. one per writing tier, so tiers bind on disjoint keys)"""
        return 2 * math.pi * (((np.asarray(ids, np.float64)[..., None] + 1) * _weyl(P, salt)) % 1.0)   # id + 1: id 0 is no identity key

    def _flat(self, a):
        a = np.asarray(a); return a.reshape(a.shape[0], -1, self.P)

    def write(self, key, value, radius=1.0):
        """key, value: (B, ..., P) angles; radius: scalar or (B, ...) confidence (the radius is the confidence, never a gate)"""
        k, v = self._flat(key), self._flat(value)
        r = np.broadcast_to(np.asarray(radius, np.float64), np.asarray(key).shape[:-1]).reshape(k.shape[0], -1, 1)
        self.M += (r * np.exp(1j * (k + v))).sum(1)

    def read(self, key):
        """(B, ..., P) keys -> (B, ..., P) complex reads: M e^{-i key} (the conjugate mirror, never 1/z)"""
        key = np.asarray(key); return (self.M[:, None] * np.exp(-1j * self._flat(key))).reshape(key.shape)

    def features(self, key):
        """the BT's bus entry: (B, ..., 2P) float32 = (Re, Im) of the read (BreathingTransformer.enter_slots(bus=...))"""
        r = self.read(key); return np.concatenate([r.real, r.imag], -1).astype(np.float32)

    def cleanup(self, key, codebook):
        """codebook (N, P) angles -> (best index (B, ...), its score (B, ...)): Re <read, e^{i c}> / P"""
        s = (self.read(key)[..., None, :] * np.exp(-1j * np.asarray(codebook))).real.sum(-1) / self.P
        return s.argmax(-1), s.max(-1)


@functools.lru_cache(maxsize=None)
def _weyl(P, salt):
    return np.array([math.sqrt(q) % 1.0 for q in _primes(P, skip=1000 + salt * P)])


def _primes(n, skip=0):
    out, k = [], 2
    while len(out) < n + skip:
        if all(k % p for p in out if p * p <= k): out.append(k)
        k += 1
    return out[skip:]


# ---- THE TREE'S CHART TABLES ------------------------------------------------------------------------------------------------------------
class ChartTable:
    """One partition's charts, a two-level BIRCH CF tree kept with WELFORD stats (n, mean, M2 per dim; BIRCH's clustering features are
    derived: N = n, LS = n mean, SS = sum M2 + n |mean|^2; CF additivity = Chan's exact merge). THE LIVE OAK's growth rule (Bryce:
    "BIRCH decides when to add children; Welford stats maintain and grow it ... new branches plateau as the tree matures"):
      - a CHART is a BIRCH leaf node: it holds up to `n_micro` micro-CF entries, kept by textbook BIRCH inside the chart (a point
        joins its nearest entry within the chart's threshold T, else spawns one; a full node merges its closest entry pair and T rises
        to their distance). A chart's stats are the exact CF sum of its entries.
      - a chart SPLITS by BIRCH's node split (seeds = its farthest entry pair, entries redistributed by closeness, two Lloyd passes at
        the CF level) when the split PASSES THE BIC TEST: two diagonal Gaussians explain its points better than one after the
        (2d + 1)/2 log n parameter cost and the mixing entropy -- one Gaussian never splits, separated clouds do. The closest live
        chart pair MERGES when one Gaussian explains it better by the same test (charts that drift together rejoin).
      - capacity is a CEILING, not a target: a split on a full table first merges the closest pair, only if that pair passes the
        merge test; otherwise the split waits (`pressure` counts the waits). The tree grows from one root to the data's structure
        and plateaus there. (11-07: textbook BIRCH on the charts themselves, T from 0, always filled capacity: 16 charts over 6
        token types split each type on noise the query cannot see, and the sealed two-tier read paid log(splits) per pointer.)
    `cap` bounds each chart's n (its entries' n and M2 scaled together: variance kept, memory bounded) and `max_idle` (None = never)
    retires a chart that received no point for that many updates -- both for charts that follow a still-training projection. Only
    the charts persist; a loaded chart restarts as a single entry."""
    def __init__(self, capacity, dim, cap=None, n_micro=8, min_child=8, max_idle=None, var_floor=1e-9):
        self.C, self.d, self.cap, self.M, self.min_child, self.max_idle, self.vf = capacity, dim, cap, n_micro, min_child, max_idle, var_floor
        self.n = np.zeros(capacity); self.mean = np.zeros((capacity, dim)); self.M2 = np.zeros((capacity, dim))
        self.mn = np.zeros((capacity, n_micro)); self.mmean = np.zeros((capacity, n_micro, dim)); self.mM2 = np.zeros((capacity, n_micro, dim))
        self.mT2 = np.zeros(capacity); self.idle = np.zeros(capacity, int); self.pressure = 0

    @property
    def live(self):
        return self.n > 0

    @property
    def N(self): return self.n.copy()

    @property
    def LS(self): return self.n[:, None] * self.mean

    @property
    def SS(self): return self.M2.sum(1) + self.n * (self.mean ** 2).sum(1)

    def var(self):
        """(C, d) per-dim variance (0 for a chart with < 2 points)"""
        return np.where(self.n[:, None] > 1, self.M2 / np.maximum(self.n[:, None], 1), 0.0)

    def centroids(self):
        return self.mean.astype(np.float32)

    def _d2(self, x):
        d2 = ((x[:, None] - self.mean[None]) ** 2).sum(-1); d2[:, ~self.live] = np.inf; return d2

    def nearest(self, x):
        """the discrete mirror: each point's live chart (-1 when nothing is live)"""
        x = np.asarray(x, np.float64)
        return np.full(len(x), -1) if not self.live.any() else self._d2(x).argmin(-1)

    # -- CF algebra (exact) --
    @staticmethod
    def _sum(ns, means, M2s):
        """the exact CF sum of entries (n_i, mean_i, M2_i) -> (n, mean, M2)"""
        n = ns.sum()
        if n <= 0: return 0.0, np.zeros(means.shape[-1]), np.zeros(means.shape[-1])
        mu = (ns[:, None] * means).sum(0) / n
        return n, mu, (M2s + ns[:, None] * (means - mu) ** 2).sum(0)

    def _loglik(self, n, M2):
        """the diagonal Gaussian's maximized log-likelihood (up to the shared constant) from Welford stats"""
        return -0.5 * n * np.log(M2 / max(n, 1e-12) + self.vf).sum()

    def _bic_gain(self, groups):
        """BIC(one Gaussian on the union) - BIC(one per group) over [(n, mean, M2)]: > 0 = the groups are separate charts"""
        ns = np.array([g[0] for g in groups]); n, _, M2 = self._sum(ns, np.stack([g[1] for g in groups]), np.stack([g[2] for g in groups]))
        if n <= 1 or (ns <= 0).any(): return -np.inf
        ll2 = sum(self._loglik(g[0], g[2]) + g[0] * np.log(g[0] / n) for g in groups)
        return (ll2 - self._loglik(n, M2)) - 0.5 * (len(groups) - 1) * (2 * self.d + 1) * np.log(n)

    # -- the leaf level: micro entries inside one chart (textbook BIRCH, scoped to the chart) --
    def _sync(self, c):
        self.n[c], self.mean[c], self.M2[c] = self._sum(self.mn[c], self.mmean[c], self.mM2[c])

    def _entry_rebuild(self, c):
        """chart c is full of entries: merge its closest pair; T rises to their distance"""
        lv = np.flatnonzero(self.mn[c] > 0); m = self.mmean[c, lv]
        d2 = ((m[:, None] - m[None]) ** 2).sum(-1); np.fill_diagonal(d2, np.inf)
        i, j = np.unravel_index(d2.argmin(), d2.shape); a, b = lv[i], lv[j]; self.mT2[c] = max(self.mT2[c], float(d2[i, j]))
        n, mu, M2 = self._sum(self.mn[c, [a, b]], self.mmean[c, [a, b]], self.mM2[c, [a, b]])
        self.mn[c, a], self.mmean[c, a], self.mM2[c, a] = n, mu, M2; self.mn[c, b] = 0; self.mmean[c, b] = 0; self.mM2[c, b] = 0

    def _entries_d2(self, c, x):
        d2 = ((x[:, None] - self.mmean[c][None]) ** 2).sum(-1); d2[:, self.mn[c] <= 0] = np.inf; return d2

    def _absorb(self, c, xs):
        """chart c absorbs points xs: outliers beyond T spawn entries (full -> rebuild), the rest Chan-merge into their nearest entry"""
        idx = np.full(len(xs), -1)
        if not (self.mn[c] > 0).any(): self.mn[c, 0], self.mmean[c, 0], self.mM2[c, 0] = 1.0, xs[0].copy(), 0; idx[0] = 0
        for j in np.flatnonzero((idx < 0) & (self._entries_d2(c, xs).min(-1) > self.mT2[c])):
            if self._entries_d2(c, xs[j:j + 1]).min() <= self.mT2[c]: continue
            while (self.mn[c] > 0).all() and self.M > 1:
                self._entry_rebuild(c)
                if self._entries_d2(c, xs[j:j + 1]).min() <= self.mT2[c]: break
            if (self.mn[c] > 0).all() or self._entries_d2(c, xs[j:j + 1]).min() <= self.mT2[c]: continue
            e = int(np.flatnonzero(self.mn[c] <= 0)[0]); self.mn[c, e], self.mmean[c, e], self.mM2[c, e] = 1.0, xs[j].copy(), 0; idx[j] = e
        rest = idx < 0
        if rest.any():
            near = self._entries_d2(c, xs[rest]).argmin(-1)
            for e in np.unique(near):
                ys = xs[rest][near == e]; n, mu, M2 = self._sum(np.array([self.mn[c, e], len(ys)]), np.stack([self.mmean[c, e], ys.mean(0)]),
                                                                np.stack([self.mM2[c, e], ((ys - ys.mean(0)) ** 2).sum(0)]))
                self.mn[c, e], self.mmean[c, e], self.mM2[c, e] = n, mu, M2
        self._sync(c)

    # -- the node level: split / merge by BIC --
    def _node_split(self, c):
        """BIRCH's node split of chart c: farthest entry pair as seeds, entries to the nearer seed, two Lloyd passes -> (group masks)"""
        lv = np.flatnonzero(self.mn[c] > 0)
        if len(lv) < 2: return None
        m, w = self.mmean[c, lv], self.mn[c, lv]
        d2 = ((m[:, None] - m[None]) ** 2).sum(-1); i, j = np.unravel_index(d2.argmax(), d2.shape); sd = m[[i, j]]
        for _ in range(3):
            g = ((m[:, None] - sd[None]) ** 2).sum(-1).argmin(-1)
            if g.min() == g.max(): return None
            sd = np.stack([(w[g == k, None] * m[g == k]).sum(0) / w[g == k].sum() for k in (0, 1)])
        return [lv[g == k] for k in (0, 1)]

    def _try_split(self, c):
        gs = self._node_split(c)
        if gs is None: return False
        parts = [self._sum(self.mn[c, e], self.mmean[c, e], self.mM2[c, e]) for e in gs]
        if min(p[0] for p in parts) < self.min_child or self._bic_gain(parts) <= 0: return False
        if self.live.all() and not self._try_merge():
            self.pressure += 1; return False
        f = int(np.flatnonzero(~self.live)[0]); e1 = gs[1]; k = len(e1)
        self.mn[f, :k], self.mmean[f, :k], self.mM2[f, :k] = self.mn[c, e1], self.mmean[c, e1], self.mM2[c, e1]
        self.mn[c, e1] = 0; self.mmean[c, e1] = 0; self.mM2[c, e1] = 0
        self.mT2[f] = self.mT2[c]; self.idle[f] = 0; self._sync(c); self._sync(f); return True

    def _closest_pair(self):
        lv = np.flatnonzero(self.live)
        if len(lv) < 2: return None
        m = self.mean[lv]; d2 = ((m[:, None] - m[None]) ** 2).sum(-1); np.fill_diagonal(d2, np.inf)
        i, j = np.unravel_index(d2.argmin(), d2.shape); return int(lv[i]), int(lv[j])

    def _try_merge(self):
        """the closest live chart pair merges when one Gaussian explains it better (BIC)"""
        pr = self._closest_pair()
        if pr is None: return False
        a, b = pr
        if self._bic_gain([(self.n[a], self.mean[a], self.M2[a]), (self.n[b], self.mean[b], self.M2[b])]) > 0: return False
        self._fold(a, b); return True

    def _fold(self, a, b):
        """chart a takes chart b's entries (union, reduced to n_micro by entry rebuilds); b dies"""
        ea, eb = np.flatnonzero(self.mn[a] > 0), np.flatnonzero(self.mn[b] > 0)
        ns = np.concatenate([self.mn[a, ea], self.mn[b, eb]]); ms = np.concatenate([self.mmean[a, ea], self.mmean[b, eb]])
        M2s = np.concatenate([self.mM2[a, ea], self.mM2[b, eb]]); T2 = max(self.mT2[a], self.mT2[b])
        while len(ns) > self.M:                                                      # entry rebuilds on the union
            d2 = ((ms[:, None] - ms[None]) ** 2).sum(-1); np.fill_diagonal(d2, np.inf); i, j = np.unravel_index(d2.argmin(), d2.shape)
            T2 = max(T2, float(d2[i, j])); n, mu, M2 = self._sum(ns[[i, j]], ms[[i, j]], M2s[[i, j]])
            ns[i], ms[i], M2s[i] = n, mu, M2; ns, ms, M2s = np.delete(ns, j), np.delete(ms, j, 0), np.delete(M2s, j, 0)
        for t in (a, b): self.mn[t] = 0; self.mmean[t] = 0; self.mM2[t] = 0
        k = len(ns); self.mn[a, :k], self.mmean[a, :k], self.mM2[a, :k] = ns, ms, M2s; self.mT2[a] = T2; self.mT2[b] = 0
        self.idle[a] = min(self.idle[a], self.idle[b]); self._sync(a); self._sync(b)

    def update(self, x):
        """x (n, d) certified points -> (their charts after the update (n,), charts spawned by splits)"""
        x = np.asarray(x, np.float64); spawned = 0
        if not len(x): return np.zeros(0, int), 0
        if not self.live.any(): self._absorb(0, x[:1]); x_ = x[1:]                 # the root
        else: x_ = x
        got = np.zeros(self.C, bool)
        if len(x_):
            near = self._d2(x_).argmin(-1)
            for c in np.unique(near): self._absorb(c, x_[near == c]); got[c] = True
        self.idle = np.where(got, 0, self.idle + 1)
        if self.max_idle is not None:
            for c in np.flatnonzero(self.live & (self.idle > self.max_idle)):
                self.n[c] = 0; self.mean[c] = 0; self.M2[c] = 0; self.mn[c] = 0; self.mmean[c] = 0; self.mM2[c] = 0; self.mT2[c] = 0
        for c in np.flatnonzero(self.live): spawned += self._try_split(c)
        while self._try_merge(): pass
        if self.cap:
            for c in np.flatnonzero(self.n > self.cap):
                f = self.cap / self.n[c]; self.mn[c] *= f; self.mM2[c] *= f; self._sync(c)
        return self.nearest(x), spawned

    def state(self):
        return self.n.copy(), self.mean.copy(), self.M2.copy()

    def set_state(self, n, mean, M2):
        assert mean.shape == (self.C, self.d), (mean.shape, self.C, self.d)
        self.n, self.mean, self.M2 = np.asarray(n, np.float64).copy(), np.asarray(mean, np.float64).copy(), np.asarray(M2, np.float64).copy()
        self.mn[:] = 0; self.mmean[:] = 0; self.mM2[:] = 0; self.mT2[:] = 0; self.idle[:] = 0
        self.mn[:, 0], self.mmean[:, 0], self.mM2[:, 0] = self.n, self.mean, self.M2   # a loaded chart restarts as one entry


class TreeTables:
    """The BT's door to the SQLite Tree (`mycelium/tree_db.TreeDB`), partitioned by (tier, head) -- one ChartTable per partition.
    io(bt) hands a tier's BT its partitions' centroids + live masks (BreathIO.charts / chart_ok); observe(tier, pts, mask) folds a
    breath's CERTIFIED chart points (BreathingTransformer out['chart_pts']) into each head's own partition.
    Persistence: the store's generation-tagged chart tables, tag = '{generation}|{tier}|{part}' (Welford n / mean / M2 per chart;
    a load of another generation is refused). Evidence: symbolic only -- a certified key's chart as a one-hot over the partition's
    charts, in the registered space 'chart-evidence|{tier}|{part}', written 'provisional' (promotion is the caller's custody step)."""
    def __init__(self, db, generation, **table_kw):
        self.db, self.gen, self.kw = db, generation, table_kw; self.tables = {}

    def tag(self, part, generation=None):
        tier, p = part; return f"{generation or self.gen}|{tier}|{p}"

    def table(self, part, capacity, dim):
        """the in-memory table for a partition: the store's saved stats of THIS generation if any, else an empty one"""
        if part not in self.tables:
            t = ChartTable(capacity, dim, **self.kw); st = self.db.load_charts(self.tag(part), self.tag(part))
            if st is not None: t.set_state(*st)
            self.tables[part] = t
        t = self.tables[part]; assert (t.C, t.d) == (capacity, dim), (part, t.C, t.d, capacity, dim)
        return t

    def attach(self, bt):
        """create / load every partition of a tier's BT -> its tables in head order"""
        c = bt.c; return [self.table(p, c.n_charts, c.d_chart) for p in c.partitions()]

    def io(self, bt):
        """-> (charts (n_partitions, C, d_chart) Tensor, chart_ok (n_partitions, C) bool Tensor) for BreathIO"""
        ts = self.attach(bt)
        return Tensor(np.stack([t.centroids() for t in ts])), Tensor(np.stack([t.live for t in ts]))

    def observe(self, bt, pts, mask):
        """pts (B, n_partitions, L, d_chart) numpy (out['chart_pts'][stream]); mask (B, L) bool: the CERTIFIED points.
        Head h's points feed partition (tier, h) only. -> [(charts of the points, spawned)] per partition"""
        pts = np.asarray(pts); mask = np.asarray(mask, bool)
        return [t.update(pts[:, h][mask]) for h, t in enumerate(self.attach(bt))]

    def load_generation(self, part, generation):
        """an explicit read of another generation's table: refused (never mix generations' coordinates)"""
        return self.db.load_charts(self.tag(part, generation), self.tag(part))

    def save(self):
        for part, t in self.tables.items(): self.db.save_charts(self.tag(part), *t.state())

    def space(self, part):
        tier, p = part; return f"chart-evidence|{tier}|{p}"

    def record(self, part, batch, rows, certifier, custody="provisional", note=None):
        """rows: [(symbolic key, chart index, source)] -> one evidence batch (one-hot over the partition's charts)"""
        n = self.tables[part].C
        self.db.register_space(self.space(part), "symbolic", n, f"one-hot chart of a certified key, partition {part}")
        ev = [(k, np.eye(n, dtype=np.float32)[c], s) for k, c, s in rows]
        return self.db.add_evidence(batch, ev, custody, certifier, self.gen, self.space(part), note)


# ---- THE ALTERNATION -------------------------------------------------------------------------------------------------------------------
class Solver(Protocol):
    def step(self, tier: str, proposals, notebook: Notebook, breath: int) -> list:
        """execute + certify the proposals; write commits into the notebook (and breaker closes on proofs); return what was done"""


class TierPort(Protocol):
    """The tier-specific adapter: what this tier's BT reads from the organs (charts: organs.tree.io(bt)), what its outputs
    propose, and which of its chart points are CERTIFIED (stream -> (B, L) bool; {} = none: test-time reads never write the Tree)."""
    def io(self, bt, organs: "Organs", breath: int): ...
    def propose(self, bt, out: dict, organs: "Organs", breath: int): ...
    def certified(self, bt, out: dict, organs: "Organs", breath: int) -> dict: ...


@dataclass
class Organs:
    notebook: object                           # a Notebook (shared) or {tier: Notebook} (each tier its own slots)
    t6: T6Clock
    t9: T9Clock
    tree: TreeTables = None
    bus: PhaseBus = None
    extra: dict = field(default_factory=dict)


class Conductor:
    """The core rhythm. tiers: [(bt, port)] in commit order (coarse first: the business tier commits before the data tier reads).
    Every breath: each tier's BT breathes at the schedule's temperature, proposes, the solver disposes into the notebook, the
    certified chart points fold into the tier's own Tree partitions (Welford + BIRCH); then the T6 clock ticks. All tiers share one
    breath count (one clock)."""
    def __init__(self, tiers, solver: Solver):
        assert len({bt.c.n_breaths for bt, _ in tiers}) == 1, "every tier breathes on the same clock"
        assert len({bt.c.tier for bt, _ in tiers}) == len(tiers), "one BT per tier"
        self.tiers, self.solver = tiers, solver

    def run(self, organs: Organs):
        trace = []
        for b in range(self.tiers[0][0].c.n_breaths):
            for bt, port in self.tiers:
                out = bt.breath(port.io(bt, organs, b), b)
                props = port.propose(bt, out, organs, b)
                nb = organs.notebook[bt.c.tier] if isinstance(organs.notebook, dict) else organs.notebook   # one notebook per tier, or shared
                trace.append((b, bt.c.tier, self.solver.step(bt.c.tier, props, nb, organs.t6.t)))
                if organs.tree is not None and bt.c.n_partitions:
                    for s, msk in port.certified(bt, out, organs, b).items():
                        organs.tree.observe(bt, out["chart_pts"][s].numpy(), msk)
            organs.t6.tick()
        return trace
