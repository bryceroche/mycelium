"""breathing_transformer.py -- THE BREATHING TRANSFORMER, the official class (tinygrad; ledger 2026-11-07 "THE TIERED BT ARCHITECTURE").

Bryce, 11-07: "lets build an official Breathing Transformer class in python for Tinygrad. It must follow all the specs we outlined above,
be fully composable so abstract out any magic numbers or parameters ... use this class to implement the Breathing transformer for every
tier going forward. It needs to be able to communicate the sqlite db and all the other organs."

THE SPEC (each line is enforced by construction here; `scripts/test_breathing_transformer.py` checks it):
  - The BT is the core COMPUTE engine. The organs live OUTSIDE it (`mycelium/bt_organs.py`): the T6 + T9 clocks, the notebook, the T256
    bus, the Tree DB, the CSP solver. A breath is a pure function of what the organs hand it; whatever persists between breaths
    goes through an organ (the alternation ruling), never through a hidden residual.
  - One BT per TIER (the tier's invariance contract is its `BTConfig.tier`). BREATHS are weight-tied (never specialize); HEADS
    specialize, each head reads its own partition (tier, head) of the Tree's tables.
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
    micro score cannot move mass between charts (the residual-seal law: the CS form).
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
    match[h] = the factors head h matches (wildcards elsewhere); n_charts[h] = 0 -> a flat softmax for that head."""
    name: str
    query: str                                 # 'token' | 'slot'
    key: str
    n_heads: int
    match: tuple                               # per head: tuple of factor names
    n_charts: tuple = ()                       # per head (empty = no charts on this road)
    residual: bool = True                      # False = a POINTER road (returns log-probs, writes nothing)
    breaths: tuple = None                      # the breaths this road runs on (None = every breath); the weights are the same

    def runs(self, b):
        return self.breaths is None or b in self.breaths

    def charted(self):
        return any(c > 0 for c in self.n_charts)


def default_roads(n_heads=8, n_charts=0, pointer_heads=1, token_breaths=(0,)):
    """The three roads of the ruling + one pointer road. Token self-attention runs on breath 0 only by default (Bryce 11-07: "we dont
    have to run token self attention every breath"; token_breaths=None = every breath). Token self matches token address (T9) and identity; slot self matches slot
    order + breath time; cross matches the token address and the Tree table/level; every head of a road shares the default match
    (heads still specialize through their weights and their own Tree partitions)."""
    cc = lambda h: (n_charts,) * h if n_charts else ()
    return (RoadSpec("token_self", "token", "token", n_heads, (("t9", "token_id"),) * n_heads, breaths=token_breaths),
            RoadSpec("slot_self", "slot", "slot", n_heads, (("slot_id", "t6"),) * n_heads),
            RoadSpec("cross", "slot", "token", n_heads, (("t9", "table", "level"),) * n_heads, cc(n_heads)),
            RoadSpec("pointer", "slot", "token", pointer_heads, (("t9", "table", "level"),) * pointer_heads, cc(pointer_heads), False))


@dataclass(frozen=True)
class BTConfig:
    tier: str                                  # the tier this BT serves (its Tree tables' partition prefix)
    d_in: int = 768                            # the reader states' width (token entry)
    d_model: int = 512
    d_waist: int = 64
    d_head: int = 64
    d_chart: int = 64                          # the macro chart space (read off the waist)
    d_cert: int = 3                            # the notebook's certificate features per slot
    d_aux: int = 0                             # extra token features entering without the reader LN (0 = none)
    d_mark: int = 0                            # per-breath mark features per stream (e.g. consumed marks; 0 = none)
    n_const: int = 0                           # learned constant keys (the 'const' key stream)
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
    weyl_seed: int = 0                         # offsets the primes behind the id phases (a different, still fixed, code)

    def __post_init__(self):
        assert self.tau_kind in ("geometric", "linear", "constant"), self.tau_kind
        assert self.n_breaths >= 1 and self.blocks_per_breath >= 1
        names = [f.name for f in self.factors]
        assert len(set(names)) == len(names), f"duplicate factor names: {names}"
        for r in self.roads:
            assert r.query in STREAMS and r.key in KEY_STREAMS, r
            assert len(r.match) == r.n_heads, f"road {r.name}: one match tuple per head"
            assert not r.n_charts or len(r.n_charts) == r.n_heads, f"road {r.name}: one chart count per head"
            for m in r.match:
                assert set(m) <= set(names), f"road {r.name}: unknown factor in {m}"
        assert len({r.name for r in self.roads}) == len(self.roads), "duplicate road names"

    def tau(self, breath):
        """The temperature of breath b (0-based): falls from tau_first to tau_last."""
        if self.tau_kind == "constant" or self.n_breaths == 1: return self.tau_first
        a = breath / (self.n_breaths - 1)
        if self.tau_kind == "linear": return self.tau_first + a * (self.tau_last - self.tau_first)
        return self.tau_first * (self.tau_last / self.tau_first) ** a

    def partition(self, road, head):
        """The Tree table partition a head reads: (tier, road/head)."""
        return (self.tier, f"{road}/h{head}")


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

    def plane_weights(self, match, gain):
        """(P,) Tensor: gain_f / planes_f on the matched factors' planes, 0 on the wildcards. gain: (F,) Tensor."""
        sel = np.zeros((len(self.factors), self.P), np.float32)
        for i, f in enumerate(self.factors):
            if f.name in match:
                a, b = self.slices[f.name]; sel[i, a:b] = 1.0 / f.planes
        return gain @ Tensor(sel)

    def presence_planes(self, pres):
        """(B, L, F) presence -> (B, L, P)"""
        own = self.owner()
        return pres @ Tensor(np.eye(len(self.factors), dtype=np.float32)[own].T.copy())


# ---- THE PIECES ------------------------------------------------------------------------------------------------------------------------
def _over_tau(x, tau):
    """x / tau: tau a float, or a (B,) Tensor (per-example breath, as in masked-diffusion training) broadcast over x's trailing dims"""
    if isinstance(tau, Tensor): return x / tau.reshape(-1, *([1] * (x.ndim - 1)))
    return x * (1 / tau)


def _zero_linear(i, o):
    l = nn.Linear(i, o); l.weight = Tensor.zeros(o, i).contiguous(); l.bias = Tensor.zeros(o).contiguous(); return l


class Waist:
    """THE U-NET's waist for one stream: z = down(LN(h)) (the silhouette, d_waist), u = up(z) (the compressed query base)."""
    def __init__(self, c):
        self.ln = nn.LayerNorm(c.d_model, eps=c.ln_eps); self.down = nn.Linear(c.d_model, c.d_waist); self.up = nn.Linear(c.d_waist, c.d_model)

    def __call__(self, h):
        z = self.down(self.ln(h)); return z, self.up(z)


class Road:
    """One road: its own heads, softmax, charts and (unless a pointer) its own output projection."""
    def __init__(self, c, spec, layout):
        H, Dh = spec.n_heads, c.d_head
        self.c, self.spec, self.L = c, spec, layout
        self.lnk = nn.LayerNorm(c.d_model, eps=c.ln_eps)
        self.q = nn.Linear(c.d_model, H * Dh)                       # from the waist's u
        self.k = nn.Linear(c.d_model, H * Dh)                       # from the FULL key state (keys never compress)
        self.turn = _zero_linear(c.d_model, H * layout.P)           # the query's learned turn per plane (zero-init: match my own stamp)
        self.gain = Tensor.full((H, len(layout.factors)), c.id_gain_init).contiguous()
        if spec.residual:
            self.v = nn.Linear(c.d_model, H * Dh); self.o = nn.Linear(H * Dh, c.d_model)
        if spec.charted():
            self.zq = nn.Linear(c.d_waist, H * c.d_chart)            # the macro query lives in the waist

    def logp(self, z, u, hk, qa, ka, kpres, valid, tau, charts=None, key_charts=None):
        """log p(key | query) per head: (B, H, Lq, Lk).
        z (B, Lq, d_waist), u (B, Lq, D) the query's waist; hk (B, Lk, D) the key state; qa (B, Lq, P) / ka (B, Lk, P) the stamps;
        kpres (B, Lk, F) key factor presence; valid (B, Lq, Lk) bool; charts: per head (C_h, d_chart) Tensors (the Tree's centroids),
        key_charts: (B, H, Lk) int (the Tree's chart of every key, per head partition)."""
        c, S, H, Dh, P = self.c, self.spec, self.spec.n_heads, self.c.d_head, self.L.P
        B, Lq, Lk = u.shape[0], u.shape[1], hk.shape[1]
        q = self.q(u).reshape(B, Lq, H, Dh).transpose(1, 2) * (1 / math.sqrt(Dh))
        k = self.k(self.lnk(hk)).reshape(B, Lk, H, Dh).transpose(1, 2)
        a = qa.unsqueeze(2) + self.turn(u).reshape(B, Lq, H, P)                                   # the query rotates; the key never does
        a = a.transpose(1, 2)                                                                       # (B, H, Lq, P)
        w = Tensor.stack(*[self.L.plane_weights(S.match[h], self.gain[h]) for h in range(H)])     # (H, P): wildcards weigh 0
        kp = self.L.presence_planes(kpres).unsqueeze(1)                                            # (B, 1, Lk, P)
        qid = Tensor.cat(a.cos() * w[None, :, None], a.sin() * w[None, :, None], dim=-1)
        kid = Tensor.cat(ka.cos().unsqueeze(1) * kp, ka.sin().unsqueeze(1) * kp, dim=-1)           # Re(e^{iq} conj e^{ik}) = cos(q - k)
        F = _over_tau(q @ k.transpose(-2, -1) + qid @ kid.transpose(-2, -1), tau)
        F = F.clip(-c.logit_clip, c.logit_clip)
        m = valid.unsqueeze(1)                                                                      # (B, 1, Lq, Lk)
        if not S.charted() or charts is None:
            return m.where(F, c.mask_neg).log_softmax(-1)
        return self._two_tier(F, z, m, tau, charts, key_charts)

    def _two_tier(self, F, z, m, tau, charts, key_charts):
        """log p(chart(t) | q) + F_t - lse_{chart(t)} F (the seal); per head with its own chart count (padded to the max)."""
        c, S, H = self.c, self.spec, self.spec.n_heads
        B, Lq = z.shape[0], z.shape[1]; Cm = max(S.n_charts)
        zq = self.zq(z).reshape(B, Lq, H, c.d_chart).transpose(1, 2)                               # (B, H, Lq, dc)
        dist = []
        for h in range(H):
            Ch = charts[h]; n = Ch.shape[0]; assert n == S.n_charts[h], f"road {S.name} head {h}: {n} charts, spec {S.n_charts[h]}"
            d = _over_tau(-((zq[:, h].unsqueeze(-2) - Ch) ** 2).sum(-1) * (1 / math.sqrt(c.d_chart)), tau)   # (B, Lq, n)
            dist.append(d.pad((None, None, (0, Cm - n)), value=c.mask_neg) if n < Cm else d)
        lc = Tensor.stack(*dist, dim=1)                                                             # (B, H, Lq, Cm)
        oh = key_charts.one_hot(Cm).cast(dtypes.float)                                             # (B, H, Lk, Cm)
        F_ = m.where(F, c.mask_neg)
        inc = m.unsqueeze(-1) & (oh.unsqueeze(2) > 0.5)                                            # (B, H, Lq, Lk, Cm)
        fm = inc.where(F_.unsqueeze(-1), c.mask_neg).max(3)                                        # per-chart max
        Sm = inc.where(F_.unsqueeze(-1) - fm.unsqueeze(3), c.mask_neg).exp().sum(3)                # (B, H, Lq, Cm)
        lsec = fm + Sm.maximum(1e-30).log()                                                        # lse within each chart
        lchr = (Sm > 0.5).where(lc, c.mask_neg).log_softmax(-1)                                    # only charts with an open key
        lp = (lchr - lsec) @ oh.transpose(-1, -2)                                                  # (B, H, Lq, Lk): each key's chart terms
        return m.where(lp + F_, c.mask_neg)

    def __call__(self, hq, z, u, hk, qa, ka, kpres, valid, tau, charts=None, key_charts=None):
        lp = self.logp(z, u, hk, qa, ka, kpres, valid, tau, charts, key_charts)
        if not self.spec.residual: return hq, lp
        B, Lk, H, Dh = hk.shape[0], hk.shape[1], self.spec.n_heads, self.c.d_head
        v = self.v(self.lnk(hk)).reshape(B, Lk, H, Dh).transpose(1, 2)
        any_ = valid.any(-1, keepdim=True).unsqueeze(1)                                             # a query with no open key reads 0
        o = (any_.where(lp.exp(), 0.0) @ v).transpose(1, 2).reshape(hq.shape[0], hq.shape[1], H * Dh)
        return hq + self.o(o), lp


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
    charts: dict = field(default_factory=dict)       # road name -> per head (C_h, d_chart) Tensor (the Tree's centroids)
    key_charts: dict = field(default_factory=dict)   # road name -> (B, H, Lk) int
    marks: dict = field(default_factory=dict)        # stream -> (B, L, d_mark) per-breath marks (added at the breath's start)


class BreathingTransformer:
    """One tier's BT. Usage per problem batch (the organs drive it; see bt_organs.Conductor):
        h_t = bt.enter_tokens(x)                        # the reader states -> the token stream
        h_s = bt.enter_slots(notebook arrays, atoms)    # the notebook's write-back -> the slot stream (every breath)
        out = bt.breath(io, b)                          # one breath at the schedule's temperature
    out: {'h': streams, 'z': per stream silhouette (B, L, d_waist), 'logp': road -> (B, H, Lq, Lk)}."""
    def __init__(self, c: BTConfig):
        self.c = c; self.layout = KeyLayout(c.factors, c.weyl_seed)
        self.tok_in = nn.Linear(c.d_in, c.d_model); self.tok_ln = nn.LayerNorm(c.d_in, eps=c.ln_eps)
        if c.d_aux: self.aux_in = nn.Linear(c.d_aux, c.d_model)
        if c.d_mark: self.mark = {s: nn.Linear(c.d_mark, c.d_model) for s in STREAMS}
        if c.n_const: self.const = (Tensor.randn(c.n_const, c.d_model) * 0.02).contiguous()
        self.waist = {s: Waist(c) for s in STREAMS}
        self.blocks = [Block(c, self.layout, i == c.blocks_per_breath - 1) for i in range(c.blocks_per_breath)]   # shared by EVERY breath
        self.sil_scale = Tensor.zeros(1).contiguous()                               # the silhouette classifier's learned scale (log)
        self.mask_slot = Tensor.zeros(c.d_model).contiguous()                       # an uncommitted slot's entry
        self.cert = nn.Linear(c.d_cert, c.d_model)
        if c.n_atoms:
            self.reent = nn.Linear(c.d_waist, c.d_model)                             # committed: atom x radius, back up from the waist space
            self.belief = nn.Linear(c.d_waist, c.d_model, bias=False)                # uncommitted: what the BT believes (soft notebook)
            self.cond_in = nn.Linear(c.d_waist, c.d_model)                           # condition: a decided coarse atom x confidence

    # -- entries (the organs' data -> streams) --
    def enter_tokens(self, x, aux=None):
        """(B, Lt, d_in) reader states (+ (B, Lt, d_aux) features entering without the LN) -> (B, Lt, D)"""
        h = self.tok_in(self.tok_ln(x))
        return h + self.aux_in(aux) if aux is not None else h

    def condition(self, h, atom, radius, atoms):
        """A coarse decision written into a stream (the tier hierarchy inside one breath: decide the atom, then address under it):
        h (B, L, D) + cond_in(radius x atoms[atom]). atom (B, L) int, radius (B, L), atoms (A, d_waist)."""
        return h + self.cond_in(radius.unsqueeze(-1) * (atom.one_hot(atoms.shape[0]).cast(dtypes.float) @ atoms))

    def enter_slots(self, committed, cert, atom=None, radius=None, belief=None, atoms=None):
        """The notebook's write-back -> the slot stream. committed (B, K) bool; cert (B, K, d_cert); atom (B, K) int, radius (B, K),
        belief (B, K, A) soft type distribution, atoms (A, d_waist) the Tree's atom centroids (needs n_atoms)."""
        B, K = committed.shape
        h = self.mask_slot.reshape(1, 1, -1).expand(B, K, self.c.d_model)
        if self.c.n_atoms and atoms is not None:
            if belief is not None: h = h + self.belief(belief @ atoms)
            if atom is not None:
                re = self.reent(radius.unsqueeze(-1) * (atom.one_hot(atoms.shape[0]).cast(dtypes.float) @ atoms))
                h = committed.unsqueeze(-1).where(re, h)
        return h + self.cert(cert)

    def tau(self, b):
        return self.c.tau(b)

    def breath(self, io: BreathIO, b, roads=None, tau=None):
        """ONE BREATH. b = the 0-based breath index (int: sets the schedule's temperature and which roads run; the weights are the same
        for every breath). roads = an explicit subset of road names (a port's two-phase breath, or the once-per-problem reader);
        tau = an explicit temperature (a float or a (B,) Tensor: per-example breaths in masked-diffusion training)."""
        tau = self.c.tau(b) if tau is None else tau; h = dict(io.h); lps = {}
        for s, mk in io.marks.items(): h[s] = h[s] + self.mark[s](mk)
        ang, pres = dict(io.angles), dict(io.presence)
        if self.c.n_const:
            B = next(iter(h.values())).shape[0]; nc, F = self.c.n_const, len(self.layout.factors)
            h["const"] = self.const.unsqueeze(0).expand(B, nc, self.c.d_model)
            ang.setdefault("const", Tensor.zeros(B, nc, self.layout.P)); pres.setdefault("const", Tensor.zeros(B, nc, F))
        for blk in self.blocks:
            for spec in self.c.roads:
                if spec.name not in blk.roads: continue
                if (roads is None and not spec.runs(b)) or (roads is not None and spec.name not in roads): continue
                s, k = spec.query, spec.key
                z, u = self.waist[s](h[s])
                hq, lp = blk.roads[spec.name](h[s], z, u, h[k], ang[s], ang[k], pres[k], io.valid[spec.name],
                                              tau, io.charts.get(spec.name), io.key_charts.get(spec.name))
                h[s] = hq; lps[spec.name] = lp
        h.pop("const", None)
        zs = {s: self.waist[s](h[s])[0] for s in STREAMS if s in h}
        return {"h": h, "z": zs, "logp": lps}

    def silhouette_logp(self, z, atoms, tau):
        """The silhouette classified against the Tree's atoms (GRAVITY form: softmax over -distance / tau). z (..., d_waist)."""
        d = -((z.unsqueeze(-2) - atoms) ** 2).sum(-1) * (1 / math.sqrt(self.c.d_waist)) * self.sil_scale.exp()
        return _over_tau(d, tau).log_softmax(-1)

    def parameters(self):
        from tinygrad.nn.state import get_parameters
        return get_parameters(self)

    def state_dict(self):
        from tinygrad.nn.state import get_state_dict
        return {k: v for k, v in get_state_dict(self).items() if not k.startswith("layout")}
