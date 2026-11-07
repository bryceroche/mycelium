"""bt_organs.py -- THE ORGANS OUTSIDE THE BREATHING TRANSFORMER (ledger 2026-11-07 "THE TIERED BT ARCHITECTURE").

The BT (`mycelium/breathing_transformer.py`) is the compute engine; everything that persists, ticks, certifies or is stored lives here:
  - T6Clock / T9Clock: the breath rotor and the token-address rotor. Each has BOTH faces (the bridge): the continuous gears (fields
    for the BT's KeyLayout, turned into angles there) and the discrete ODOMETER (integer digits: the exact form for Tree keys).
    Speeds come from `rotor_clock` (one source); the clock-range rule is checked at construction.
  - Notebook: the per-problem slot table the solver writes back to (committed atom, radius = confidence, certificate features,
    the T6 stamp of the commit, the tree level, consumed marks). Certificates, never raw values, enter the reasoning state.
  - PhaseBus: the T256 rotational bus: bind by phase addition, unbind by the conjugate (never 1/z), codebook cleanup.
  - ChartTable / TreeTables: the Tree DB's chart tables, partitioned by (tier, road/head). Centroids are MODEL-space, so they live
    only in the store's generation-tagged centroids table (BIRCH CFs N, LS, SS; never mixed across generations); the symbolic
    write-back (which chart a certified key took) goes through the custody tiers as evidence ('provisional' until the answer key
    promotes it). The breath never queries SQLite: a run loads its tables into memory.
  - Solver (Protocol) + TierPort (Protocol) + Conductor: THE ALTERNATION. Each breath, every tier's BT proposes (coarse tier
    first), the solver disposes and writes back to the notebook, the clock ticks, the temperature falls. Tiers talk only through
    the notebook and the bus (the residual-seal law: no shared residual between tiers).
"""
from dataclasses import dataclass, field
import math
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
    conjugate e^{-i key}; cleanup snaps the read to the nearest codebook entry (crosstalk ~ 1/sqrt(P) per stored pair)."""
    def __init__(self, B, P=256):
        self.P = P; self.M = np.zeros((B, P), np.complex128)

    def write(self, key, value, radius=1.0):
        """key, value: (B, P) angles; radius: (B,) or scalar confidence"""
        self.M += np.asarray(radius, np.float64).reshape(-1, 1) * np.exp(1j * (np.asarray(key) + np.asarray(value)))

    def read(self, key):
        return self.M * np.exp(-1j * np.asarray(key))                                            # the conjugate mirror, never 1/z

    def cleanup(self, key, codebook):
        """codebook (N, P) angles -> (best index (B,), its score (B,)): Re <read, e^{i c}> / P"""
        s = (self.read(key)[:, None, :] * np.exp(-1j * np.asarray(codebook))[None]).real.sum(-1) / self.P
        return s.argmax(-1), s.max(-1)


# ---- THE TREE'S CHART TABLES ------------------------------------------------------------------------------------------------------------
class ChartTable:
    """One partition's charts as BIRCH clustering features (N, LS, SS): centroid = LS / N. Fed only by rows the caller certified."""
    def __init__(self, n_charts, dim, init=None, rng=0):
        self.N = np.zeros(n_charts); self.SS = np.zeros(n_charts)
        self.LS = np.zeros((n_charts, dim)) if init is None else np.asarray(init, np.float64).copy()
        self._init = np.random.default_rng(rng).standard_normal((n_charts, dim)) * 0.1 if init is None else self.LS.copy()
        if init is not None: self.N[:] = 1.0; self.SS = (self.LS ** 2).sum(1)

    def centroids(self):
        """(C, d) numpy: LS / N (an empty chart keeps its init until it has evidence)"""
        c = self._init.copy(); m = self.N > 0
        c[m] = self.LS[m] / self.N[m, None]
        return c.astype(np.float32)

    def tensor(self):
        return Tensor(self.centroids())

    def update(self, x, idx):
        """x (n, d) points, idx (n,) their charts (the certified assignment): CF additivity"""
        x = np.asarray(x, np.float64); idx = np.asarray(idx)
        np.add.at(self.N, idx, 1.0); np.add.at(self.LS, idx, x); np.add.at(self.SS, idx, (x ** 2).sum(1))

    def nearest(self, x):
        """the discrete mirror: each point's chart = its nearest centroid"""
        return ((np.asarray(x)[:, None] - self.centroids()[None]) ** 2).sum(-1).argmin(-1)


class TreeTables:
    """The BT's door to the SQLite Tree (`mycelium/tree_db.TreeDB`), partitioned by (tier, road/head).
    Centroids: the store's generation-tagged centroids table, tag = '{generation}|{tier}|{part}' (load refuses another generation).
    Evidence: symbolic only -- a certified key's chart as a one-hot over the partition's charts, in the registered space
    'chart-evidence|{tier}|{part}', written 'provisional' (the answer key's promotion is the caller's custody step)."""
    def __init__(self, db, generation):
        self.db, self.gen = db, generation; self.tables = {}

    def tag(self, part, generation=None):
        tier, p = part; return f"{generation or self.gen}|{tier}|{p}"

    def table(self, part, n_charts, dim, init=None):
        """the in-memory table for a partition: the store's centroids of THIS generation if saved, else a fresh one"""
        if part not in self.tables:
            cf = self.db.load_centroids(self.tag(part), self.tag(part))
            if cf is None: self.tables[part] = ChartTable(n_charts, dim, init)
            else:
                N, LS, SS = cf; t = ChartTable(n_charts, dim); assert LS.shape == (n_charts, dim), (part, LS.shape)
                t.N, t.LS, t.SS = N.astype(np.float64), LS.astype(np.float64), SS.astype(np.float64); self.tables[part] = t
        return self.tables[part]

    def load_generation(self, part, generation):
        """an explicit read of another generation's table: refused (never mix generations' coordinates)"""
        return self.db.load_centroids(self.tag(part, generation), self.tag(part))

    def save(self):
        for part, t in self.tables.items(): self.db.save_centroids(self.tag(part), t.N, t.LS, t.SS)

    def space(self, part):
        tier, p = part; return f"chart-evidence|{tier}|{p}"

    def record(self, part, batch, rows, certifier, tier="provisional", note=None):
        """rows: [(symbolic key, chart index, source)] -> one evidence batch (one-hot over the partition's charts)"""
        n = self.tables[part].N.shape[0]
        self.db.register_space(self.space(part), "symbolic", n, f"one-hot chart of a certified key, partition {part}")
        ev = [(k, np.eye(n, dtype=np.float32)[c], s) for k, c, s in rows]
        return self.db.add_evidence(batch, ev, tier, certifier, self.gen, self.space(part), note)


# ---- THE ALTERNATION -------------------------------------------------------------------------------------------------------------------
class Solver(Protocol):
    def step(self, tier: str, proposals, notebook: Notebook, breath: int) -> list:
        """execute + certify the proposals; write commits into the notebook (and breaker closes on proofs); return what was done"""


class TierPort(Protocol):
    """The tier-specific adapter: what this tier's BT reads from the organs, and what its outputs propose."""
    def io(self, bt, organs: "Organs", breath: int): ...
    def propose(self, bt, out: dict, organs: "Organs", breath: int): ...


@dataclass
class Organs:
    notebook: Notebook
    t6: T6Clock
    t9: T9Clock
    tree: TreeTables = None
    bus: PhaseBus = None
    extra: dict = field(default_factory=dict)


class Conductor:
    """The core rhythm. tiers: [(bt, port)] in commit order (coarse first: the business tier commits before the data tier reads).
    Every breath: each tier's BT breathes at the schedule's temperature, proposes, the solver disposes into the notebook; then the
    T6 clock ticks. All tiers share one breath count (one clock)."""
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
                trace.append((b, bt.c.tier, self.solver.step(bt.c.tier, props, organs.notebook, organs.t6.t)))
            organs.t6.tick()
        return trace
