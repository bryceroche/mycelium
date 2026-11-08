"""breathing_transformer.py -- THE BREATHING TRANSFORMER, the official class (tinygrad; ledger 2026-11-07 "THE TIERED BT ARCHITECTURE").

Bryce, 11-07: "lets build an official Breathing Transformer class in python for Tinygrad. It must follow all the specs we outlined above,
be fully composable so abstract out any magic numbers or parameters ... use this class to implement the Breathing transformer for every
tier going forward. It needs to be able to communicate the sqlite db and all the other organs."

THE SPEC (each line is enforced by construction here; `scripts/test_breathing_transformer.py` checks it):
  - The BT is the core COMPUTE engine. The organs live OUTSIDE it (`mycelium/bt_organs.py`): the T6 + T9 clocks, the notebook, the T256
    bus, the Tree DB, the CSP solver. A breath is a pure function of what the organs hand it; whatever persists between breaths
    goes through an organ (the alternation ruling), never through a hidden residual.
  - One BT per TIER (the tier's invariance contract is its `BTConfig.tier`). BREATHS are weight-tied (never specialize); HEADS
    specialize, each head reads its own partition (tier, head) of the Tree's tables: head h of every charted road in a tier reads
    partition (tier, h) through the tier's chart projection for head h (so 3 tiers x 6 heads = 18 partitions, 18 centroid sets).
    The partitions' centroids are maintained OUTSIDE the BT (`bt_organs.ChartTable`: Welford stats + BIRCH spawning) from the
    points the breath hands back (`out['chart_pts']`), and enter the next breath as data (`BreathIO.charts`, never parameters).
  - No RoPE: position and time enter only as composite-key phases from the clocks. No MLP: the only maps are linear projections,
    LayerNorm and attention (no activation function exists in this file).
  - THE U-NET every breath: the query stream passes the waist (d_model -> d_waist -> d_model); KEYS NEVER PASS IT (content keys and
    values read the full state); the composite keys skip the waist and rejoin the query on their own planes.
  - COMPOSITE KEYS: factors (token_id, slot_id, T6 stamp, T9 stamp, table, level, beam by default; any set by config), each on its
    OWN DISJOINT complex planes (`KeyLayout`). A head matches a subset of factors (the rest are wildcards: their planes carry zero
    weight). The match is the conjugate unbind, Re(e^{i q} e^{-i k}) = cos(q - k), averaged over the factor's planes. The query's
    angle = its own stamp + a learned turn (zero-init): queries rotate, keys never do.
  - THREE SEPARATE ROADS per breath block, each its own softmax, heads, tier tables and contract: token self-attention, slot
    self-attention (relations), slot -> token cross-attention (a road may run on a subset of breaths: RoadSpec.breaths; token
    self-attention defaults to breath 0). POINTER roads use the same scorer and return log-probs instead of
    writing a residual.
  - THE TWO-TIER SOFTMAX on a road with charts: log p(key) = log p(chart(key) | q) + log p(key | chart, q). Tier 1 (macro) is the
    waist's query against the head's Tree chart centroids; tier 2 (micro) is content + composite-key match within the chart. The
    micro score cannot move mass between charts (the residual-seal law: the CS form). A key's CHART is the macro view of it: its
    silhouette (the key stream's waist z) through the same head's chart projection, snapped to the nearest live centroid (the
    discrete mirror; no gradient crosses the snap) -- unless the organs hand an exact chart (`BreathIO.key_charts`, e.g. the
    Tree's symbolic lookup). The MICRO match (content + composite keys) never passes the waist.
  - THE CHART BEAM (`BTConfig.chart_beam` = W > 0; the theory line "between charts the bridge (beam + solver) chooses at the
    junctions"): the macro keeps each query's top-W charts (a discrete prune: keys in the other charts get no mass -- the road stays
    mandatory, the micro can never reach a pruned chart); inside the beam the keys COMPETE across charts: log p(key) =
    log_softmax over the beam's keys of [log p(chart(key) | q) + micro(key)]. (11-07, SMb/SMc: the sealed read normalizes the micro
    within ONE chart, so a wrong key alone in its chart never meets the gold key in another; the keys' charts carry structure finer
    than the query resolves -- word ids under a type -- and the sealed read paid for it.) Training passes `BreathIO.beam_keep` (the
    gold keys) so the gold key's chart is always in the beam (gold-in-beam); W >= the live charts = the full product of experts.
  - DIFFUSION DYNAMICS: every logit (chart and micro) is divided by the breath's temperature; tau falls over the breaths
    (`BTConfig.tau_first -> tau_last`, geometric or linear), so the representation sharpens slowly.

Shapes: B batch, Lt tokens, Ls slots, H heads per road, Dh head dim, P key planes, C charts. Integer key fields arrive as numpy (host
data); the layout turns them into angles once per batch (float64 on the host, so large ids keep their phase precision).
"""
from dataclasses import dataclass, field
import math

import numpy as np
from tinygrad import Tensor, dtypes, nn

from mycelium import rotor_clock as CK

STREAMS = ("token", "slot")                    # query streams (each has a waist)
KEY_STREAMS = STREAMS + ("const",)             # key streams: + the learned constants (no stamps, no waist)


# ---- CONFIG ------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class KeyFactor:
    """One composite-key factor on its own planes. kind 'clock': angles = sum over gears (field value x speed in degrees), one plane
    per speed (the continuous face; the odometer digits live in the clock organ). kind 'id': a quasi-random phase per id (Weyl
    sequence: plane p turns frac(id * alpha_p) of a circle, alpha_p = frac(sqrt(prime_p))), so any id range needs no table."""
    name: str
    kind: str                                  # 'clock' | 'id'
    gears: tuple = ()                          # clock: ((field, speeds_deg), ...)
    field: str = ""                            # id: the integer field
    n_planes: int = 0                          # id: planes (clock: implied by the gears)

    @property
    def planes(self):
        return sum(len(s) for _, s in self.gears) if self.kind == "clock" else self.n_planes

    @property
    def fields(self):
        return tuple(f for f, _ in self.gears) if self.kind == "clock" else (self.field,)


T9_SLOW = tuple(360.0 / p for p in (192.0, 768.0, 3072.0))   # the T9 slow gears 7-9 (periods in tokens; scripts/distill_reader.py)


def default_factors(id_planes=8, slot_speeds=CK.T6_SPEEDS, small_planes=4):
    """The seven factors of the 11-07 ruling. slot_id is a CLOCK over the slot index (so 'the slot before me' is a rotation);
    T6 = the breath tick on the six gears; T9 = the token offset on the six gears + the absolute position on the slow gears."""
    return (KeyFactor("token_id", "id", field="token_id", n_planes=id_planes),
            KeyFactor("slot_id", "clock", gears=(("slot", tuple(slot_speeds)),)),
            KeyFactor("t6", "clock", gears=(("tick", tuple(CK.T6_SPEEDS)),)),
            KeyFactor("t9", "clock", gears=(("off", tuple(CK.T6_SPEEDS)), ("pos", T9_SLOW))),
            KeyFactor("table", "id", field="table", n_planes=small_planes),
            KeyFactor("level", "id", field="level", n_planes=small_planes),
            KeyFactor("beam", "id", field="beam", n_planes=small_planes))


@dataclass(frozen=True)
class RoadSpec:
    """A road: queries from `query` stream (through the waist), keys/values from `key` stream (full state).
    match[h] = the factors head h matches (wildcards elsewhere); charts = True -> the two-tier softmax, head h on partition (tier, h)."""
    name: str
    query: str                                 # 'token' | 'slot'
    key: str
    n_heads: int
    match: tuple                               # per head: tuple of factor names
    charts: bool = False                       # the two-tier softmax over the tier's Tree partitions (head h -> partition h)
    residual: bool = True                      # False = a POINTER road (returns log-probs, writes nothing)
    breaths: tuple = None                      # the breaths this road runs on (None = every breath); the weights are the same
    d_head: int = 0                            # this road's head width (0 = BTConfig.d_head; e.g. one full-width pointer head)

    def runs(self, b):
        return self.breaths is None or b in self.breaths

    def charted(self):
        return self.charts


def default_roads(n_heads=6, charts=False, pointer_heads=1, token_breaths=(0,)):
    """The three roads of the ruling + one pointer road. Token self-attention runs on breath 0 only by default (Bryce 11-07: "we dont
    have to run token self attention every breath"; token_breaths=None = every breath). Token self matches token address (T9) and identity; slot self matches slot
    order + breath time; cross matches the token address and the Tree table/level; every head of a road shares the default match
    (heads still specialize through their weights and their own Tree partitions). charts=True charts every road whose KEYS are tokens
    (token self, cross, pointer): partition h then holds head h's view of the tokens."""
    return (RoadSpec("token_self", "token", "token", n_heads, (("t9", "token_id"),) * n_heads, charts, breaths=token_breaths),
            RoadSpec("slot_self", "slot", "slot", n_heads, (("slot_id", "t6"),) * n_heads),
            RoadSpec("cross", "slot", "token", n_heads, (("t9", "table", "level"),) * n_heads, charts),
            RoadSpec("pointer", "slot", "token", pointer_heads, (("t9", "table", "level"),) * pointer_heads, charts, False))


@dataclass(frozen=True)
class BTConfig:
    tier: str                                  # the tier this BT serves (its Tree tables' partition prefix)
    d_in: int = 768                            # the reader states' width (token entry)
    d_model: int = 512
    d_waist: int = 64
    d_head: int = 64
    d_chart: int = 64                          # the macro chart space (read off the waist)
    n_partitions: int = 0                      # this tier's Tree partitions = its heads (head h of a charted road -> partition h)
    n_charts: int = 16                         # chart capacity per partition (BIRCH spawns live charts up to it)
    chart_beam: int = 0                        # 0 = the sealed two-tier read; W > 0 = THE CHART BEAM (top-W charts per query, keys compete)
    d_bus: int = 0                             # T256 bus planes the slots read (0 = no bus entry)
    d_cert: int = 3                            # the notebook's certificate features per slot
    d_aux: int = 0                             # extra token features entering without the reader LN (0 = none)
    d_mark: int = 0                            # per-breath mark features per stream (e.g. consumed marks; 0 = none)
    n_const: int = 0                           # learned constant keys (the 'const' key stream)
    skip_factors: tuple = ("token_id", "slot_id", "t6", "t9", "table", "level")   # the IDs the U-Net skip carries (beam: bookkeeping)
    soft_belief: bool = False                  # enter_slots reads a soft type belief (the soft notebook) for uncommitted slots
    n_atoms: int = 0                           # the Tree atoms the notebook commits slots to (0 = no atom entry)
    n_breaths: int = 6
    blocks_per_breath: int = 1                 # blocks inside one breath (distinct weights); breaths reuse them
    factors: tuple = field(default_factory=default_factors)
    roads: tuple = field(default_factory=default_roads)
    tau_first: float = 2.0                     # the diffusion schedule: logits / tau(breath)
    tau_last: float = 0.5
    tau_kind: str = "geometric"                # 'geometric' | 'linear' | 'constant'
    id_gain_init: float = 1.0                  # per head x factor gain on the composite-key match (gains learnable, speeds frozen)
    mask_neg: float = -1e4                     # a masked logit (exp underflows to 0 in fp32)
    logit_clip: float = 1e4
    ln_eps: float = 1e-5
    waist_at: str = "every"                    # where queries squeeze (11-08): 'every' road of every block (the 11-07 form) | 'last' block's
                                               # roads only | 'end' = no road; one waist per breath (the silhouette + the pointer roads)
    dropout: float = 0.0                       # regularization: each residual road's write + the token entry, TRAINING only (0 = off)
    weyl_seed: int = 0                         # offsets the primes behind the id phases (a different, still fixed, code)

    def __post_init__(self):
        assert self.tau_kind in ("geometric", "linear", "constant"), self.tau_kind
        assert self.waist_at in ("every", "last", "end"), self.waist_at
        assert self.n_breaths >= 1 and self.blocks_per_breath >= 1
        names = [f.name for f in self.factors]
        assert len(set(names)) == len(names), f"duplicate factor names: {names}"
        for r in self.roads:
            assert r.query in STREAMS and r.key in KEY_STREAMS, r
            assert len(r.match) == r.n_heads, f"road {r.name}: one match tuple per head"
            if r.charts:
                assert r.key in STREAMS, f"road {r.name}: charted keys need a waist (their chart is their silhouette's)"
                assert 0 < r.n_heads <= self.n_partitions, f"road {r.name}: {r.n_heads} heads, {self.n_partitions} partitions"
            for m in r.match:
                assert set(m) <= set(names), f"road {r.name}: unknown factor in {m}"
        assert len({r.name for r in self.roads}) == len(self.roads), "duplicate road names"

    def tau(self, breath):
        """The temperature of breath b (0-based): falls from tau_first to tau_last."""
        if self.tau_kind == "constant" or self.n_breaths == 1: return self.tau_first
        a = breath / (self.n_breaths - 1)
        if self.tau_kind == "linear": return self.tau_first + a * (self.tau_last - self.tau_first)
        return self.tau_first * (self.tau_last / self.tau_first) ** a

    def partition(self, head):
        """The Tree table partition head h reads (every charted road of this tier): (tier, 'h{head}')."""
        assert 0 <= head < self.n_partitions, (head, self.n_partitions)
        return (self.tier, f"h{head}")

    def partitions(self):
        return [self.partition(h) for h in range(self.n_partitions)]


# ---- THE COMPOSITE KEYS (disjoint planes) ---------------------------------------------------------------------------------------------
def _primes(n, skip=0):
    out, k = [], 2
    while len(out) < n + skip:
        if all(k % p for p in out if p * p <= k): out.append(k)
        k += 1
    return out[skip:]


class KeyLayout:
    """Factor -> its own contiguous, disjoint plane range. angles(fields) -> (angles (B, L, P), presence (B, L, F)) on the host."""
    def __init__(self, factors, weyl_seed=0):
        self.factors = tuple(factors); self.slices = {}; o = 0
        for f in self.factors:
            assert f.planes > 0, f"factor {f.name} has no planes"
            self.slices[f.name] = (o, o + f.planes); o += f.planes
        self.P = o
        self._alpha = {}
        n_id = sum(f.planes for f in self.factors if f.kind == "id")
        pr = iter(_primes(n_id, skip=weyl_seed))
        for f in self.factors:
            if f.kind == "id": self._alpha[f.name] = np.array([math.sqrt(p) % 1.0 for p in (next(pr) for _ in range(f.planes))])

    def owner(self):
        """(P,) the factor index of every plane (the disjointness, as data)"""
        own = np.full(self.P, -1)
        for i, f in enumerate(self.factors):
            a, b = self.slices[f.name]; assert (own[a:b] == -1).all(); own[a:b] = i
        assert (own >= 0).all()
        return own

    def angles(self, fields, shape):
        """fields: {field: int array (B, L)}; a factor whose fields are all absent is ABSENT for this stream (angle 0, presence 0)."""
        A = np.zeros(shape + (self.P,), np.float64); pres = np.zeros(shape + (len(self.factors),), np.float32)
        for i, f in enumerate(self.factors):
            a, b = self.slices[f.name]
            have = [fl in fields for fl in f.fields]
            if not any(have): continue
            assert all(have), f"factor {f.name}: fields {f.fields} must arrive together"
            if f.kind == "clock":
                cols = [np.asarray(fields[fl], np.float64)[..., None] * np.radians(np.asarray(sp, np.float64)) for fl, sp in f.gears]
                A[..., a:b] = np.concatenate(cols, -1)
            else:
                A[..., a:b] = 2 * math.pi * ((np.asarray(fields[f.field], np.float64)[..., None] * self._alpha[f.name]) % 1.0)
            pres[..., i] = 1.0
        return np.mod(A, 2 * math.pi).astype(np.float32), pres

    def selector(self, matches):
        """(H, F, P) numpy: 1 / planes_f on factor f's planes where head h matches f (wildcards 0). Built once per road and realized
        at construction: a constant made inside a TinyJit body is captured as a buffer the replay must not outlive."""
        sel = np.zeros((len(matches), len(self.factors), self.P), np.float32)
        for h, match in enumerate(matches):
            for i, f in enumerate(self.factors):
                if f.name in match:
                    a, b = self.slices[f.name]; sel[h, i, a:b] = 1.0 / f.planes
        return sel

    def presence_planes(self, pres):
        """(B, L, F) presence -> (B, L, P) (the owner map is realized once, at the layout's first use)"""
        if not hasattr(self, "_own_t"):
            self._own_t = Tensor(np.eye(len(self.factors), dtype=np.float32)[self.owner()].T.copy()).realize()
        return pres @ self._own_t


# ---- THE PIECES ------------------------------------------------------------------------------------------------------------------------
def _over_tau(x, tau):
    """x / tau: tau a float, or a (B,) Tensor (per-example breath, as in masked-diffusion training) broadcast over x's trailing dims"""
    if isinstance(tau, Tensor): return x / tau.reshape(-1, *([1] * (x.ndim - 1)))
    return x * (1 / tau)


def _zero_linear(i, o):
    l = nn.Linear(i, o); l.weight = Tensor.zeros(o, i).contiguous(); l.bias = Tensor.zeros(o).contiguous(); return l


class Waist:
    """THE U-NET for one stream: z = down(LN(h)) (the silhouette, d_waist); u = up(z) + skip(ID tags) (the query base).
    THE SKIP carries ONLY the stream's own composite-key phasors (cos / sin of its stamps on the present factors' planes), never the
    state: the IDs bypass the waist and rejoin after it (11-07 fix: without it every uncommitted slot was the same query)."""
    def __init__(self, c, P):
        self.ln = nn.LayerNorm(c.d_model, eps=c.ln_eps); self.down = nn.Linear(c.d_model, c.d_waist); self.up = nn.Linear(c.d_waist, c.d_model)
        self.skip = nn.Linear(2 * P, c.d_model, bias=False)

    def __call__(self, h, ids):
        z = self.down(self.ln(h)); return z, self.up(z) + self.skip(ids)


class Road:
    """One road: its own heads, softmax, charts and (unless a pointer) its own output projection."""
    def __init__(self, c, spec, layout):
        H, Dh = spec.n_heads, spec.d_head or c.d_head
        self.c, self.spec, self.L, self.Dh = c, spec, layout, Dh
        self.lnk = nn.LayerNorm(c.d_model, eps=c.ln_eps)
        self.q = nn.Linear(c.d_model, H * Dh)                       # from the waist's u
        self.k = nn.Linear(c.d_model, H * Dh)                       # from the FULL key state (keys never compress)
        self.turn = _zero_linear(c.d_model, H * layout.P)           # the query's learned turn per plane (zero-init: match my own stamp)
        self.gain = Tensor.full((H, len(layout.factors)), c.id_gain_init).contiguous()
        self.sel = Tensor(layout.selector(spec.match)).realize()      # (H, F, P) constant, realized once
        if spec.residual:
            self.v = nn.Linear(c.d_model, H * Dh); self.o = nn.Linear(H * Dh, c.d_model)

    def logp(self, u, hk, qa, ka, kpres, valid, tau, ch=None):
        """log p(key | query) per head: (B, H, Lq, Lk).
        u (B, Lq, D) the query's waist output; hk (B, Lk, D) the key state; qa (B, Lq, P) / ka (B, Lk, P) the stamps;
        kpres (B, Lk, F) key factor presence; valid (B, Lq, Lk) bool; ch (charted roads): dict(cq (B, H, Lq, dc) the queries' chart
        coords, cent (H, C, dc) the partitions' centroids, ok (H, C) live charts, kc (B, H, Lk) int each key's chart)."""
        c, S, H, Dh, P = self.c, self.spec, self.spec.n_heads, self.Dh, self.L.P
        B, Lq, Lk = u.shape[0], u.shape[1], hk.shape[1]
        q = self.q(u).reshape(B, Lq, H, Dh).transpose(1, 2) * (1 / math.sqrt(Dh))
        k = self.k(self.lnk(hk)).reshape(B, Lk, H, Dh).transpose(1, 2)
        a = qa.unsqueeze(2) + self.turn(u).reshape(B, Lq, H, P)                                   # the query rotates; the key never does
        a = a.transpose(1, 2)                                                                       # (B, H, Lq, P)
        w = (self.gain.unsqueeze(-1) * self.sel.detach()).sum(1)                                   # (H, P): wildcards weigh 0
        kp = self.L.presence_planes(kpres).unsqueeze(1)                                            # (B, 1, Lk, P)
        qid = Tensor.cat(a.cos() * w[None, :, None], a.sin() * w[None, :, None], dim=-1)
        kid = Tensor.cat(ka.cos().unsqueeze(1) * kp, ka.sin().unsqueeze(1) * kp, dim=-1)           # Re(e^{iq} conj e^{ik}) = cos(q - k)
        F = _over_tau(q @ k.transpose(-2, -1) + qid @ kid.transpose(-2, -1), tau)
        F = F.clip(-c.logit_clip, c.logit_clip)
        m = valid.unsqueeze(1)                                                                      # (B, 1, Lq, Lk)
        if not S.charted() or ch is None:
            return m.where(F, c.mask_neg).log_softmax(-1)
        return self._two_tier(F, m, tau, ch)

    def _two_tier(self, F, m, tau, ch):
        """log p(chart(t) | q) + F_t - lse_{chart(t)} F (the seal); every head on its own partition's live charts."""
        c = self.c; cent, ok, Cm = ch["cent"], ch["ok"], ch["cent"].shape[1]
        d = -((ch["cq"].unsqueeze(-2) - cent.unsqueeze(1)) ** 2).sum(-1) * (1 / math.sqrt(c.d_chart))     # (B, H, Lq, C)
        lc = ok.unsqueeze(1).where(_over_tau(d, tau), c.mask_neg)                                  # dead charts never draw mass
        oh = ch["kc"].one_hot(Cm).cast(dtypes.float)                                               # (B, H, Lk, C)
        F_ = m.where(F, c.mask_neg)
        inc = m.unsqueeze(-1) & (oh.unsqueeze(2) > 0.5)                                            # (B, H, Lq, Lk, Cm)
        fm = inc.where(F_.unsqueeze(-1), c.mask_neg).max(3)                                        # per-chart max
        Sm = inc.where(F_.unsqueeze(-1) - fm.unsqueeze(3), c.mask_neg).exp().sum(3)                # (B, H, Lq, Cm)
        lsec = fm + Sm.maximum(1e-30).log()                                                        # lse within each chart
        lchr = (Sm > 0.5).where(lc, c.mask_neg).log_softmax(-1)                                    # only charts with an open key
        ch["lchr"] = lchr                                                                           # handed back: out['macro']
        if c.chart_beam:                                                                            # THE CHART BEAM
            rank = (lchr.unsqueeze(-1) < lchr.unsqueeze(-2)).cast(dtypes.float).sum(-1)            # charts strictly ahead of each chart
            beam = (rank < c.chart_beam) & (Sm > 0.5)
            if ch.get("keep") is not None:                                                         # gold-in-beam: the kept keys' charts
                beam = beam | ((ch["keep"].unsqueeze(1).cast(dtypes.float) @ oh) > 0.5)
            kin = m & ((beam.cast(dtypes.float) @ oh.transpose(-1, -2)) > 0.5)                     # (B, H, Lq, Lk): a key in the beam
            lg = kin.where(lchr @ oh.transpose(-1, -2) + F, c.mask_neg).log_softmax(-1)
            return kin.where(lg, c.mask_neg)
        lp = (lchr - lsec) @ oh.transpose(-1, -2)                                                  # (B, H, Lq, Lk): each key's chart terms
        return m.where(lp + F_, c.mask_neg)

    def __call__(self, hq, u, hk, qa, ka, kpres, valid, tau, ch=None):
        lp = self.logp(u, hk, qa, ka, kpres, valid, tau, ch)
        if not self.spec.residual: return hq, lp
        B, Lk, H, Dh = hk.shape[0], hk.shape[1], self.spec.n_heads, self.Dh
        v = self.v(self.lnk(hk)).reshape(B, Lk, H, Dh).transpose(1, 2)
        any_ = valid.any(-1, keepdim=True).unsqueeze(1)                                             # a query with no open key reads 0
        o = (any_.where(lp.exp(), 0.0) @ v).transpose(1, 2).reshape(hq.shape[0], hq.shape[1], H * Dh)
        return hq + self.o(o).dropout(self.c.dropout), lp


class Block:
    """One block of a breath: every road once, in config order (residual roads update their query stream in place). Pointer roads
    (residual False) live in the LAST block only: they read the finished state (an earlier copy would never reach a loss)."""
    def __init__(self, c, layout, last):
        self.roads = {r.name: Road(c, r, layout) for r in c.roads if r.residual or last}


# ---- THE BREATHING TRANSFORMER --------------------------------------------------------------------------------------------------------
@dataclass
class BreathIO:
    """What the organs hand one breath (and what it hands back). Streams' key stamps arrive as angles (KeyLayout.angles)."""
    h: dict                                    # stream -> (B, L, D) state
    angles: dict                               # stream -> (B, L, P) composite-key angles
    presence: dict                             # stream -> (B, L, F) factor presence
    valid: dict                                # road name -> (B, Lq, Lk) bool (padding, consumed marks, the solver's breaker)
    charts: Tensor = None                            # (n_partitions, C, d_chart) this tier's partitions' centroids (TreeTables.io)
    chart_ok: Tensor = None                          # (n_partitions, C) bool: the live charts (BIRCH has spawned them)
    key_charts: dict = field(default_factory=dict)   # road name -> (B, H, Lk) int: an exact chart per key (else: the silhouette's)
    beam_keep: dict = field(default_factory=dict)    # road name -> (B, Lq, Lk) bool: keys whose charts the chart beam must keep (gold)
    marks: dict = field(default_factory=dict)        # stream -> (B, L, d_mark) per-breath marks (added at the breath's start)


class BreathingTransformer:
    """One tier's BT. Usage per problem batch (the organs drive it; see bt_organs.Conductor):
        h_t = bt.enter_tokens(x)                        # the reader states -> the token stream
        h_s = bt.enter_slots(notebook arrays, atoms)    # the notebook's write-back -> the slot stream (every breath)
        out = bt.breath(io, b)                          # one breath at the schedule's temperature
    out: {'h': streams, 'z': per stream silhouette (B, L, d_waist), 'logp': road -> (B, H, Lq, Lk), 'macro': charted road ->
    (log p(chart | q) (B, H, Lq, C), each key's chart (B, H, Lk) int) -- the macro's own distribution, for its own loss term (a new
    attention pathway needs direct supervision: SMd, gold-in-beam alone let the micro carry training and left the macro untrained)}."""
    def __init__(self, c: BTConfig):
        self.c = c; self.layout = KeyLayout(c.factors, c.weyl_seed)
        self.tok_in = nn.Linear(c.d_in, c.d_model); self.tok_ln = nn.LayerNorm(c.d_in, eps=c.ln_eps)
        if c.d_aux: self.aux_in = nn.Linear(c.d_aux, c.d_model)
        if c.d_mark: self.mark = {s: nn.Linear(c.d_mark, c.d_model) for s in STREAMS}
        if c.n_const: self.const = (Tensor.randn(c.n_const, c.d_model) * 0.02).contiguous()
        self.waist = {s: Waist(c, self.layout.P) for s in STREAMS}
        self.blocks = [Block(c, self.layout, i == c.blocks_per_breath - 1) for i in range(c.blocks_per_breath)]   # shared by EVERY breath
        self.sil_scale = Tensor.zeros(1).contiguous()                               # the silhouette classifier's learned scale (log)
        self.mask_slot = Tensor.zeros(c.d_model).contiguous()                       # an uncommitted slot's entry
        self.cert = nn.Linear(c.d_cert, c.d_model)
        if c.n_partitions:
            self.chart = nn.Linear(c.d_waist, c.n_partitions * c.d_chart)            # head h's view of a silhouette: partition h's space
        if c.d_bus: self.bus_in = nn.Linear(2 * c.d_bus, c.d_model, bias=False)       # the T256 bus read (Re, Im) per slot
        if c.waist_at != "every":                                                   # a full-width query's LN (unsqueezed roads)
            self.qln = {s: nn.LayerNorm(c.d_model, eps=c.ln_eps) for s in STREAMS}
        if c.n_atoms:
            self.reent = nn.Linear(c.d_waist, c.d_model)                             # committed: atom x radius, back up from the waist space
            if c.soft_belief: self.belief = nn.Linear(c.d_waist, c.d_model, bias=False)   # uncommitted: what the BT believes
            self.cond_in = nn.Linear(c.d_waist, c.d_model)                           # condition: a decided coarse atom x confidence

    # -- entries (the organs' data -> streams) --
    def enter_tokens(self, x, aux=None):
        """(B, Lt, d_in) reader states (+ (B, Lt, d_aux) features entering without the LN) -> (B, Lt, D)"""
        h = self.tok_in(self.tok_ln(x))
        h = h + self.aux_in(aux) if aux is not None else h
        return h.dropout(self.c.dropout)

    def condition(self, h, atom, radius, atoms):
        """A coarse decision written into a stream (the tier hierarchy inside one breath: decide the atom, then address under it):
        h (B, L, D) + cond_in(radius x atoms[atom]). atom (B, L) int, radius (B, L), atoms (A, d_waist)."""
        return h + self.cond_in(radius.unsqueeze(-1) * (atom.one_hot(atoms.shape[0]).cast(dtypes.float) @ atoms))

    def enter_slots(self, committed, cert, atom=None, radius=None, belief=None, atoms=None, bus=None):
        """The notebook's write-back -> the slot stream. committed (B, K) bool; cert (B, K, d_cert); atom (B, K) int, radius (B, K),
        belief (B, K, A) soft type distribution, atoms (A, d_waist) the Tree's atom centroids (needs n_atoms); bus (B, K, 2 d_bus) the
        T256 bus read at each slot's key (PhaseBus.features: what the other tiers bound there)."""
        B, K = committed.shape
        h = self.mask_slot.reshape(1, 1, -1).expand(B, K, self.c.d_model)
        if self.c.n_atoms and atoms is not None:
            if belief is not None:
                assert self.c.soft_belief, "belief given but BTConfig.soft_belief is False"
                h = h + self.belief(belief @ atoms)
            if atom is not None:
                re = self.reent(radius.unsqueeze(-1) * (atom.one_hot(atoms.shape[0]).cast(dtypes.float) @ atoms))
                h = committed.unsqueeze(-1).where(re, h)
        if bus is not None:
            assert self.c.d_bus, "bus read given but BTConfig.d_bus is 0"
            h = h + self.bus_in(bus)
        return h + self.cert(cert)

    def chart_coords(self, z):
        """(B, L, d_waist) silhouettes -> (B, n_partitions, L, d_chart): every head's view, in its partition's space"""
        B, L = z.shape[0], z.shape[1]
        return self.chart(z).reshape(B, L, self.c.n_partitions, self.c.d_chart).transpose(1, 2)

    def _charts(self, io, spec, zq, hk, ak, pk):
        """the two-tier inputs for a charted road: the queries' chart coords, the head partitions' centroids, each key's chart"""
        H = spec.n_heads
        assert io.charts is not None and io.chart_ok is not None, f"road {spec.name} is charted: the organs must hand BreathIO.charts"
        cent, ok = io.charts[:H], io.chart_ok[:H]
        root = Tensor.arange(ok.shape[1]).reshape(1, -1) == 0                      # an empty partition = one root chart (a flat read)
        ok = ok | (root & ok.logical_not().all(-1, keepdim=True))
        kc = io.key_charts.get(spec.name)
        if kc is None:                                                               # the key's silhouette, snapped (no gradient)
            zk = self.waist[spec.key](hk, self.ids(ak, pk))[0].detach()
            dk = ((self.chart_coords(zk)[:, :H].unsqueeze(-2) - cent.unsqueeze(1)) ** 2).sum(-1)
            kc = ok.unsqueeze(1).where(dk, float("inf")).argmin(-1).detach()        # (B, H, Lk): the nearest live chart (the snap: no grad)
        return dict(cq=self.chart_coords(zq)[:, :H], cent=cent, ok=ok, kc=kc, keep=io.beam_keep.get(spec.name))

    def tau(self, b):
        return self.c.tau(b)

    def breath(self, io: BreathIO, b, roads=None, tau=None):
        """ONE BREATH. b = the 0-based breath index (int: sets the schedule's temperature and which roads run; the weights are the same
        for every breath). roads = an explicit subset of road names (a port's two-phase breath, or the once-per-problem reader);
        tau = an explicit temperature (a float or a (B,) Tensor: per-example breaths in masked-diffusion training)."""
        tau = self.c.tau(b) if tau is None else tau; h = dict(io.h); lps = {}; mac = {}
        for s, mk in io.marks.items(): h[s] = h[s] + self.mark[s](mk)
        ang, pres = dict(io.angles), dict(io.presence)
        if self.c.n_const:
            B = next(iter(h.values())).shape[0]; nc, F = self.c.n_const, len(self.layout.factors)
            h["const"] = self.const.unsqueeze(0).expand(B, nc, self.c.d_model)
            ang.setdefault("const", Tensor.zeros(B, nc, self.layout.P).contiguous()); pres.setdefault("const", Tensor.zeros(B, nc, F).contiguous())
        nb = len(self.blocks)
        for bi, blk in enumerate(self.blocks):
            for spec in self.c.roads:
                if spec.name not in blk.roads: continue
                if (roads is None and not spec.runs(b)) or (roads is not None and spec.name not in roads): continue
                s, k = spec.query, spec.key
                idq = self.ids(ang[s], pres[s]); z, u = self.waist[s](h[s], idq)
                if not self._squeezes(spec, bi == nb - 1): u = self.qln[s](h[s]) + self.waist[s].skip(idq)   # full width, the IDs still rejoin
                ch = self._charts(io, spec, z, h[k], ang[k], pres[k]) if spec.charted() else None
                hq, lp = blk.roads[spec.name](h[s], u, h[k], ang[s], ang[k], pres[k], io.valid[spec.name], tau, ch)
                h[s] = hq; lps[spec.name] = lp
                if ch is not None and "lchr" in ch: mac[spec.name] = (ch["lchr"], ch["kc"])     # (the two-tier read ran)
        h.pop("const", None)
        zs = {s: self.waist[s](h[s], self.ids(ang[s], pres[s]))[0] for s in STREAMS if s in h}
        out = {"h": h, "z": zs, "logp": lps, "macro": mac}
        if self.c.n_partitions:                                                      # what the Tree's Welford/BIRCH fold reads
            out["chart_pts"] = {s: self.chart_coords(zs[s]).detach() for s in {r.key for r in self.c.roads if r.charted()} if s in zs}
        return out

    def _squeezes(self, spec, last):
        """does this road's query pass the waist? (waist_at; a pointer road always reads the silhouette in 'end')"""
        w = self.c.waist_at
        return w == "every" or (w == "last" and last) or (w == "end" and not spec.residual)

    def ids(self, ang, pres):
        """the ID tags the skip carries: (B, L, 2P) = cos / sin of the stream's own stamps, zero on absent factors' planes"""
        if not hasattr(self, "_skip_sel"):
            self._skip_sel = Tensor(np.array([f.name in self.c.skip_factors for f in self.layout.factors], np.float32)).realize()
        pp = self.layout.presence_planes(pres * self._skip_sel)
        return Tensor.cat(ang.cos() * pp, ang.sin() * pp, dim=-1)

    def silhouette_logp(self, z, atoms, tau):
        """The silhouette classified against the Tree's atoms (GRAVITY form: softmax over -distance / tau). z (..., d_waist)."""
        d = -((z.unsqueeze(-2) - atoms) ** 2).sum(-1) * (1 / math.sqrt(self.c.d_waist)) * self.sil_scale.exp()
        return _over_tau(d, tau).log_softmax(-1)

    CONSTANTS = ("sel", "_own_t", "_skip_sel")              # realized constant buffers (never trained, never saved)

    @classmethod
    def _is_const(cls, k):
        parts = k.split(".")
        return parts[0] == "layout" or "L" in parts or parts[-1] in cls.CONSTANTS

    def parameters(self):
        return list(self.state_dict().values())

    def load(self, sd):
        """Load saved weights: the keys must equal state_dict()'s exactly (no silent fallbacks), shapes must match."""
        own = self.state_dict(); sd = {k: v for k, v in sd.items() if not self._is_const(k)}   # (files saved before 11-07 15:40 carry _own_t)
        assert set(sd) == set(own), f"key mismatch: missing {sorted(set(own) - set(sd))[:5]}, unexpected {sorted(set(sd) - set(own))[:5]}"
        for k, v in own.items():
            assert v.shape == sd[k].shape, (k, v.shape, sd[k].shape)
            v.replace(sd[k].to(v.device)).realize()

    def state_dict(self):
        from tinygrad.nn.state import get_state_dict
        return {k: v for k, v in get_state_dict(self).items() if not self._is_const(k)}
