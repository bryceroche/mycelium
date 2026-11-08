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
    projection) and BIRCH math (the clustering features N, LS, SS; a point farther than the threshold from every live chart
    SPAWNS a new chart while capacity lasts). Only CERTIFIED points feed it (custody: a correct chain). Centroids are MODEL-space,
    so they live only in the store's generation-tagged chart tables (never mixed across generations); the symbolic write-back
    (which chart a certified key took) goes through the custody tiers as evidence ('provisional' until the answer key promotes
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
    """One partition's charts: WELFORD stats per chart (n, mean, M2 per dim) + BIRCH growth. A chart is LIVE once spawned (n > 0).
    update(x) (certified points only): (1) BIRCH, one pass over the OUTLIERS (points beyond the threshold T from every live chart;
    T^2 = spawn_k^2 x the pooled mean squared radius, trusted after min_n points): in order, an outlier within T of a chart born in
    this pass joins it (its running mean moves), else it spawns a new chart while capacity and max_spawn last; the very first point
    spawns the root; (2) every point joins its nearest live chart; (3) each chart merges its batch by Chan's formula, then n is
    capped at `cap` (M2 scaled with it: the variance is kept, the memory is bounded -- the charts follow a moving projection).
    The BIRCH clustering features are derived: N = n, LS = n mean, SS = sum M2 + n |mean|^2."""
    def __init__(self, capacity, dim, cap=None, spawn_k=3.0, max_spawn=2, min_n=None):
        self.C, self.d, self.cap, self.spawn_k, self.max_spawn = capacity, dim, cap, spawn_k, max_spawn
        self.min_n = 2 * dim if min_n is None else min_n                          # pooled radius trusted only after this many points
        self.n = np.zeros(capacity); self.mean = np.zeros((capacity, dim)); self.M2 = np.zeros((capacity, dim))

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

    def threshold2(self):
        tot = self.n.sum()
        return None if tot < self.min_n else self.spawn_k ** 2 * self.M2.sum() / tot

    def centroids(self):
        return self.mean.astype(np.float32)

    def _d2(self, x):
        d2 = ((x[:, None] - self.mean[None]) ** 2).sum(-1); d2[:, ~self.live] = np.inf; return d2

    def nearest(self, x):
        """the discrete mirror: each point's live chart (-1 when nothing is live)"""
        x = np.asarray(x, np.float64)
        return np.full(len(x), -1) if not self.live.any() else self._d2(x).argmin(-1)

    def update(self, x):
        """x (n, d) certified points -> (their charts (n,), charts spawned)"""
        x = np.asarray(x, np.float64); spawned = 0
        if not len(x): return np.zeros(0, int), 0
        if not self.live.any():
            self.mean[0] = x[0]; self.n[0] = 1e-9; spawned = 1                      # spawned: live, holds no mass yet
        T2 = self.threshold2()
        if T2 is not None:
            out = np.flatnonzero(self._d2(x).min(-1) > T2); born, cnt = [], []
            for j in out:                                                            # the BIRCH pass, outliers only, in order
                if born:
                    dd = ((self.mean[born] - x[j]) ** 2).sum(-1); i = int(dd.argmin())
                    if dd[i] <= T2:
                        cnt[i] += 1; self.mean[born[i]] += (x[j] - self.mean[born[i]]) / cnt[i]; continue
                if spawned >= self.max_spawn or self.live.all(): break
                c = int(np.flatnonzero(~self.live)[0]); self.mean[c] = x[j]; self.n[c] = 1e-9
                born.append(c); cnt.append(1); spawned += 1
        idx = self._d2(x).argmin(-1)
        for c in np.unique(idx):                                                     # Chan: merge the batch into each chart
            xs = x[idx == c]; nb = len(xs); mb = xs.mean(0); m2b = ((xs - mb) ** 2).sum(0)
            na = self.n[c] if self.n[c] >= 1 else 0.0; dl = mb - self.mean[c]; nn_ = na + nb
            self.mean[c] = self.mean[c] + dl * nb / nn_ if na else mb
            self.M2[c] = self.M2[c] + m2b + dl ** 2 * na * nb / nn_ if na else m2b
            self.n[c] = nn_
        if self.cap:
            over = self.n > self.cap; f = np.where(over, self.cap / np.maximum(self.n, 1e-12), 1.0)
            self.n = self.n * f; self.M2 = self.M2 * f[:, None]
        return idx, spawned

    def state(self):
        return self.n.copy(), self.mean.copy(), self.M2.copy()

    def set_state(self, n, mean, M2):
        assert mean.shape == (self.C, self.d), (mean.shape, self.C, self.d)
        self.n, self.mean, self.M2 = np.asarray(n, np.float64).copy(), np.asarray(mean, np.float64).copy(), np.asarray(M2, np.float64).copy()


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

    def record(self, part, batch, rows, certifier, tier="provisional", note=None):
        """rows: [(symbolic key, chart index, source)] -> one evidence batch (one-hot over the partition's charts)"""
        n = self.tables[part].C
        self.db.register_space(self.space(part), "symbolic", n, f"one-hot chart of a certified key, partition {part}")
        ev = [(k, np.eye(n, dtype=np.float32)[c], s) for k, c, s in rows]
        return self.db.add_evidence(batch, ev, tier, certifier, self.gen, self.space(part), note)


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
