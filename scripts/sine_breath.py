"""sine_breath.py -- the U-Net's concepts on the looped MHA blocks, as a standalone
tinygrad sketch. Nothing here touches scripts/phase1_algebra_head.py; in the head it
would enter as an env door (e.g. ALG_SINE=128,224,384,224) with unset = bit-identical.

THE SHAPE (one period per breath; the sine is in LOG-width; trough on the breath boundary)

    crystal --> k0 --> k1 --> k2 --> k3 --> squeeze --> crystal'
      128       128    224    384    224    (the polar waist's job, sealed)
                       |___ skip ___|       <- around the PEAK, never around the trough

  * k0 runs INSIDE the crystal: the slots attend to each other in 128-d. Today the waist is a
    linear 384->128->384 with no processing at 128; attention at the trough is the new organ.
  * every step is x1.73 (log-sine). A linear sine samples 128/256/384/256: 2.0x into the trough,
    1.5x off the peak, and the asymmetry explodes as the trough deepens (trough 32: 6.5x vs 1.8x).
  * the 128 clock dims ride every block at full width, steer Q/K, and are never written
    (bitwise -- the waist already leaves them alone).
  * lawful bypasses of the trough: the clock (WHEN) and the notebook (which reads the crystal).
    A skip AROUND the trough is the residual-seal law's bypass; skip="boundary" exists only as
    that placebo.

ARMS (make_schedule)
  SINE4   [128, 224, 384, 224]  the proposal
  FLAT4   [384, 384, 384, 384]  the registered depth pair's 4-layer arm (more params)
  ANTI4   [384, 224, 128, 224]  same widths half a period out: the textbook U (trough mid-breath,
                                the skip around it, a 3x jump at the boundary) -- matched params
  NAZARE  trough 128 -> 64 over the six breaths, peak fixed: the literal funnel in the deduction
          path. Block weights are Matryoshka-sliced (whole heads) because the same block runs at
          different widths in different breaths. Fenced in the reply; kept here so it can be read.
commit_damp() is the funnel's lawful home: commit side, per slot, keyed to evidence.

Run:  DEV=CPU python3 sine_breath.py      (prints the tables and runs the contract checks)
"""
import math
import numpy as np
from tinygrad import Tensor

HEAD_DIM = 32     # fixed head size: a block's width sets its head count (384 -> 12, 224 -> 7, 128 -> 4)
CLOCK_D = 128     # the clock planes (64 planes x 2): never in the ramp
CONTENT_D = 384   # the content planes at full width


# ----------------------------------------------------------------------------- schedules

def _quant(x, q):
    return max(q, int(round(x / q)) * q) if q else x


def log_sine_widths(trough, peak, n_blocks=4, quantum=HEAD_DIM):
    """Block k sits at phase 2*pi*k/n: k=0 is the trough (beside the boundary), k=n/2 the peak.
    Sine in log-width, so every step has the same ratio (peak/trough)**(2/n) for n=4."""
    mu = 0.5 * (math.log(peak) + math.log(trough))
    amp = 0.5 * (math.log(peak) - math.log(trough))
    return [_quant(math.exp(mu - amp * math.cos(2 * math.pi * k / n_blocks)), quantum)
            for k in range(n_blocks)]


def linear_sine_widths(trough, peak, n_blocks=4, quantum=None):
    mid, amp = 0.5 * (peak + trough), 0.5 * (peak - trough)
    return [_quant(mid - amp * math.cos(2 * math.pi * k / n_blocks), quantum) for k in range(n_blocks)]


def step_ratios(widths):
    """Cyclic: block k -> k+1, and the last block -> the next breath's first."""
    n = len(widths)
    return [max(widths[k], widths[(k + 1) % n]) / min(widths[k], widths[(k + 1) % n]) for k in range(n)]


def nazare_troughs(n_breaths=6, start=128, end=64, quantum=HEAD_DIM):
    """The literal funnel: the trough deepens geometrically, the peak stays -- the wave's
    amplitude grows toward the shore."""
    return [_quant(start * (end / start) ** (b / max(n_breaths - 1, 1)), quantum) for b in range(n_breaths)]


def make_schedule(arm, n_breaths=6, trough=128, peak=CONTENT_D, quantum=HEAD_DIM):
    sine = log_sine_widths(trough, peak, quantum=quantum)
    if arm == "SINE4":
        rows = [(trough, sine)] * n_breaths
    elif arm == "FLAT4":
        rows = [(trough, [peak] * 4)] * n_breaths
    elif arm == "ANTI4":
        rows = [(trough, sine[2:] + sine[:2])] * n_breaths
    elif arm == "NAZARE":
        rows = [(t, log_sine_widths(t, peak, quantum=quantum))
                for t in nazare_troughs(n_breaths, start=trough, quantum=quantum)]
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return [{"crystal": t, "blocks": list(ws)} for t, ws in rows]


# ----------------------------------------------------------------------------- organs

def _orth(rng, a, b):
    """Semi-orthogonal init: an isometry going up (a <= b), a projection going down."""
    q, _ = np.linalg.qr(rng.standard_normal((max(a, b), max(a, b))))
    return Tensor(np.ascontiguousarray(q[:a, :b]).astype(np.float32), requires_grad=True)


def _gauss(rng, shape, scale):
    return Tensor((rng.standard_normal(shape) * scale).astype(np.float32), requires_grad=True)


class Resample:
    """A learned width change between levels (the U-Net's down / up), prefix-sliced so one
    allocation serves every breath's width."""

    def __init__(self, a_max, b_max, rng):
        self.W = _orth(rng, a_max, b_max)

    def __call__(self, h, a, b):
        return h @ self.W[:a, :b]

    def params(self):
        return [self.W]


class SlotMHA:
    """One MHA block over the slots at live width w <= w_max. Heads are contiguous HEAD_DIM
    column blocks, so slicing a prefix of width w keeps w // head_dim whole heads (Matryoshka
    on heads). Pre-RMSNorm, the clock steers Q/K only, the output writes content only."""

    def __init__(self, w_max, rng, clock_d, head_dim, o_init="zero"):
        assert w_max % head_dim == 0
        self.head_dim = head_dim
        s, sc = 1.0 / math.sqrt(w_max), 0.5 / math.sqrt(clock_d)
        self.g = Tensor.ones(w_max, requires_grad=True)
        self.Wq, self.Wk, self.Wv = (_gauss(rng, (w_max, w_max), s) for _ in range(3))
        self.Cq, self.Ck = _gauss(rng, (clock_d, w_max), sc), _gauss(rng, (clock_d, w_max), sc)
        # zero-born output = the chassis's W_bo convention (silent birth). The block is still a
        # ROAD: the level it sits on is mandatory, only its delta starts silent.
        self.Wo = (Tensor.zeros(w_max, w_max, requires_grad=True) if o_init == "zero"
                   else _gauss(rng, (w_max, w_max), 0.5 * s))

    def params(self):
        return [self.g, self.Wq, self.Wk, self.Wv, self.Cq, self.Ck, self.Wo]

    def __call__(self, h, clock, w, mask=None, taps=None, key=None):
        B, L, _ = h.shape
        nh, hd = w // self.head_dim, self.head_dim
        x = h * (h.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * self.g[:w]
        q = x @ self.Wq[:w, :w] + clock @ self.Cq[:, :w]
        k = x @ self.Wk[:w, :w] + clock @ self.Ck[:, :w]
        v = x @ self.Wv[:w, :w]
        q, k, v = (t.reshape(B, L, nh, hd).transpose(1, 2) for t in (q, k, v))
        sc = (q @ k.transpose(-2, -1) / math.sqrt(hd)).clip(-1e4, 1e4)
        if mask is not None:
            sc = mask.reshape(B, 1, L, L).where(sc, -1e4)
        att = sc.softmax(-1)
        if taps is not None:
            taps[key] = att          # (B, nh, L, L): run the shuffle read per block on these
        out = (att @ v).transpose(1, 2).reshape(B, L, w)
        return h + out @ self.Wo[:w, :w]


class SineBreath:
    """Four MHA blocks looped over the breaths with a per-breath width schedule.

    skip: "peak"     the lawful U-Net skip: k1's output rides around the expansion into k3
          "none"     no skip
          "boundary" THE SEAL PLACEBO: last breath's falling state rides around the crystal into
                     this breath's k1 -- predicted to idle the trough (the residual-seal law)
    keep: a leak IN CRYSTAL COORDINATES, crystal' = new + keep * (old - new); it never leaves the
          trough, so it is not a bypass (the chassis's cur + g*(h - cur), moved to the trough)."""

    def __init__(self, schedule, *, content_d=CONTENT_D, clock_d=CLOCK_D, head_dim=HEAD_DIM,
                 skip="peak", keep=0.0, o_init="zero", seed=0):
        assert skip in ("peak", "none", "boundary"), skip
        assert all(len(r["blocks"]) == 4 for r in schedule), "the skip pairing assumes a 4-block period"
        for r in schedule:
            assert all(w % head_dim == 0 for w in r["blocks"]), r
            assert r["blocks"][1] == r["blocks"][3], "the U-Net skip pairs equal widths (k1 <-> k3)"
        self.sched, self.skip, self.keep, self.content_d = schedule, skip, keep, content_d
        rng = np.random.default_rng(seed)
        wmax = [max(r["blocks"][k] for r in schedule) for k in range(4)]
        tmax = max(r["crystal"] for r in schedule)
        self.blocks = [SlotMHA(wmax[k], rng, clock_d, head_dim, o_init) for k in range(4)]
        self.lift_in = (Resample(tmax, wmax[0], rng)                     # crystal -> k0, only if they differ
                        if any(r["crystal"] != r["blocks"][0] for r in schedule) else None)
        self.rs = [Resample(wmax[k], wmax[k + 1], rng)                   # k -> k+1 level changes
                   if any(r["blocks"][k] != r["blocks"][k + 1] for r in schedule) else None
                   for k in range(3)]
        self.squeeze = Resample(wmax[3], tmax, rng)    # k3 -> the next crystal (the waist's W_down)
        self.entry = Resample(content_d, tmax, rng)    # breath 0's state -> the first crystal
        self.lift = Resample(tmax, content_d, rng)     # what heads / notebook ink read (the waist's W_up)

    def params(self):
        ps = [p for blk in self.blocks for p in blk.params()]
        for org in [self.lift_in, *self.rs, self.squeeze, self.entry, self.lift]:
            if org is not None:
                ps += org.params()
        return ps

    def n_params(self):
        return int(sum(np.prod(p.shape) for p in self.params()))

    def _rs(self, k, h, a, b):
        if self.rs[k] is None:
            assert a == b
            return h
        return self.rs[k](h, a, b)

    def breath(self, crystal, b, clock, mask=None, carry=None, taps=None):
        """One breath. Returns (the next crystal, k3's output = the falling state)."""
        r = self.sched[b]
        t, (w0, w1, w2, w3) = r["crystal"], r["blocks"]
        t_next = self.sched[b + 1]["crystal"] if b + 1 < len(self.sched) else t
        h = self.lift_in(crystal, t, w0) if self.lift_in is not None else crystal
        h = self.blocks[0](h, clock, w0, mask, taps, (b, 0))   # the trough: the slots confer in the crystal
        h = self._rs(0, h, w0, w1)
        if self.skip == "boundary" and carry is not None:
            h = h + carry[..., :w1]                             # THE SEAL PLACEBO (bypasses the crystal)
        h = self.blocks[1](h, clock, w1, mask, taps, (b, 1))
        rise = h
        h = self._rs(1, h, w1, w2)
        h = self.blocks[2](h, clock, w2, mask, taps, (b, 2))   # the peak: the full-width work
        h = self._rs(2, h, w2, w3)
        if self.skip == "peak":
            h = h + rise                                        # the lawful skip: around the expansion
        h = self.blocks[3](h, clock, w3, mask, taps, (b, 3))
        new = self.squeeze(h, w3, t_next)                       # the boundary: everything crosses the crystal
        if self.keep:
            new = new + self.keep * (crystal[..., :t_next] - new)
        return new, h

    def __call__(self, content0, clocks, mask=None, taps=None, damp=None):
        """content0 (B, L, content_d): breath 0's state (the grounding read, outside time).
        clocks: one (B, L, clock_d) tensor per loop breath. damp(old, new, b) -> crystal: the
        commit-side hook (see commit_damp). Returns one read per breath: (B, L, content_d +
        clock_d), the lifted crystal beside the untouched clock -- what the ladder grades."""
        crystal = self.entry(content0, self.content_d, self.sched[0]["crystal"])
        carry, reads = None, []
        for b in range(len(self.sched)):
            nxt, h3 = self.breath(crystal, b, clocks[b], mask, carry, taps)
            if damp is not None:
                nxt = damp(crystal[..., :nxt.shape[-1]], nxt, b)
            crystal, carry = nxt, h3
            reads.append(self.lift(crystal, crystal.shape[-1], self.content_d).cat(clocks[b], dim=-1))
        return reads


def commit_damp(old, new, evidence, bands=((0, 16, 0.3), (16, 80, 0.6), (80, 128, None)),
                return_share=False):
    """THE NAZARE FUNNEL ON THE COMMIT SIDE (CLOCK_SPEC's phase fence: the wave touches commit /
    memory ordering, never the deduction path). Per slot, per band, the share of the OLD crystal
    kept rises with that slot's evidence -- clock v1's operator, own-sentence completion
    c in [0, 1], shape (B, L, 1). Band (lo, hi, theta): share = clip((c - theta) / (1 - theta), 0, 1);
    theta None = never settles. The default 16 / 64 / 48 split is the band-sizing read's WHAT /
    WHICH / LEAF, and its order (root first, leaf open) is HS_241's 2,4,0 re-keyed from breath
    index to evidence: the swell is the same every breath, the canyon is each slot's own.
    Inference-time routing only; c never enters a loss (the Goodhart fence)."""
    B, L, t = new.shape
    assert bands[-1][1] == t, f"bands must tile the crystal ({bands[-1][1]} != {t})"
    shares = []
    for lo, hi, theta in bands:
        if theta is None:
            shares.append(Tensor.zeros(B, L, hi - lo))
        else:
            shares.append(((evidence - theta) / (1.0 - theta)).clip(0.0, 1.0).expand(B, L, hi - lo))
    share = shares[0].cat(*shares[1:], dim=-1)
    out = new + share * (old - new)
    return (out, share) if return_share else out


def toy_clock(b, B, L, clock_d=CLOCK_D):
    """Stand-in for rotor_clock's table: planes on three wheels turning 60 / 120 / 0 degrees per
    breath (breath hand, parity, pass), fixed per-plane offsets. Swap in the real table."""
    p = np.arange(clock_d // 2)
    inc = np.array([math.pi / 3, 2 * math.pi / 3, 0.0])[p % 3]
    ang = p * 2.399963 + (b + 1) * inc
    row = np.stack([np.cos(ang), np.sin(ang)], -1).reshape(-1).astype(np.float32)
    return Tensor(np.broadcast_to(row, (B, L, clock_d)).copy())


# ----------------------------------------------------------------------------- contract checks

def _seal_rank(skip, eps=1e-3):
    """Numerical seal test on a tiny body (content 16, clock 4, head 4, widths [4, 8, 16, 8]):
    the Jacobian of breath 2's falling state w.r.t. breath 1's. Sealed, everything that reaches
    the next breath crossed the crystal, so rank <= L * trough. With the boundary skip it is not."""
    L, t = 3, 4
    sched = make_schedule("SINE4", n_breaths=2, trough=t, peak=16, quantum=4)
    m = SineBreath(sched, content_d=16, clock_d=4, head_dim=4, skip=skip, o_init="small", seed=1)
    w3 = sched[0]["blocks"][3]
    x0 = np.random.default_rng(2).standard_normal((L, w3)).astype(np.float32)
    n = x0.size
    # every +eps / -eps perturbation rides the batch axis: one forward, not 2n
    xs = np.repeat(x0[None], 2 * n, axis=0).reshape(2 * n, n)
    xs[np.arange(n) * 2, np.arange(n)] += eps
    xs[np.arange(n) * 2 + 1, np.arange(n)] -= eps
    xt = Tensor(xs.reshape(2 * n, L, w3))
    crystal = m.squeeze(xt, w3, sched[1]["crystal"])
    _, h3 = m.breath(crystal, 1, toy_clock(1, 2 * n, L, clock_d=4), carry=xt)
    y = h3.numpy().reshape(2 * n, -1).astype(np.float64)
    J = ((y[0::2] - y[1::2]) / (2 * eps)).T                       # (outputs, inputs)
    sv = np.linalg.svd(J, compute_uv=False)
    return int((sv > 1e-2 * sv[0]).sum()), n, L * t


def main():
    rng = np.random.default_rng(0)
    print("== the wave: widths and step ratios (cyclic, incl. the boundary) ==")
    for tr in (128, 32):
        lin, log = linear_sine_widths(tr, 384), log_sine_widths(tr, 384, quantum=None)
        print(f"  trough {tr:3d}  linear {[round(w) for w in lin]}  ratios {[round(r, 2) for r in step_ratios(lin)]}")
        print(f"  trough {tr:3d}  log    {[round(w) for w in log]}  ratios {[round(r, 2) for r in step_ratios(log)]}")
    print(f"  quantized to whole heads: SINE4 {log_sine_widths(128, 384)}  "
          f"Nazare troughs {nazare_troughs()}")

    B, L = 2, 24
    content0 = Tensor(rng.standard_normal((B, L, CONTENT_D)).astype(np.float32))
    clocks = [toy_clock(b, B, L) for b in range(6)]
    n_valid = 20                                    # 4 padded slots: attend to self only
    mk = np.zeros((B, L, L), bool)
    mk[:, :, :n_valid] = True
    mk[:, np.arange(L), np.arange(L)] = True
    mask = Tensor(mk)

    print("\n== arms: schedule per breath, params, forward ==")
    for arm in ("SINE4", "FLAT4", "ANTI4", "NAZARE"):
        sched = make_schedule(arm)
        m = SineBreath(sched, skip="peak")
        taps = {}
        reads = m(content0, clocks, mask=mask, taps=taps)
        reads[0].realize(*reads[1:])                  # one schedule for the whole loop
        out = [r.numpy() for r in reads]
        assert all(o.shape == (B, L, CONTENT_D + CLOCK_D) and np.isfinite(o).all() for o in out)
        clock_ok = all(np.array_equal(o[..., CONTENT_D:], c.numpy()) for o, c in zip(out, clocks))
        assert clock_ok, "a block wrote the clock"
        heads = sorted({tuple(a.shape[1] for (bb, kk), a in taps.items() if bb == b) for b in range(6)})
        rows = " | ".join(f"{r['crystal']}:{r['blocks']}" for r in (sched[0], sched[-1]))
        print(f"  {arm:6s} params {m.n_params():>9,}  first|last breath {rows}  heads/block {heads}  clock bitwise: ok")

    print("\n== two-terminal: every parameter gets a gradient (o_init='small' so the deltas speak) ==")
    for arm in ("SINE4", "NAZARE"):                    # two breaths keep tinygrad's backward quick;
        m = SineBreath(make_schedule(arm, n_breaths=2), o_init="small", seed=3)   # NAZARE still slices (128 -> 64)
        target = Tensor(rng.standard_normal((B, L, CONTENT_D)).astype(np.float32))
        reads = m(content0, clocks[:2], mask=mask)
        w = [1.0, 2.0]                                  # the ladder's 1 -> 2 rung weights
        loss = sum(wb * (r[..., :CONTENT_D] - target).square().mean() for wb, r in zip(w, reads))
        loss.backward()
        bad = [i for i, p in enumerate(m.params()) if p.grad is None or not np.isfinite(p.grad.numpy()).all()
               or float(np.abs(p.grad.numpy()).sum()) == 0.0]
        print(f"  {arm:6s} loss {float(loss.numpy()):.4f}  params {len(m.params())}  dead/None grads: {bad or 'none'}")
        assert not bad

    print("\n== the commit-side funnel: per-slot evidence freezes coarse bands first ==")
    m = SineBreath(make_schedule("SINE4"), o_init="small", seed=4)
    ev = [Tensor(np.clip(rng.uniform(0, 1, (B, L, 1)) * (b + 1) / 6 * 1.6, 0, 1).astype(np.float32)) for b in range(6)]
    frozen = []

    def damp(old, new, b):
        out, share = commit_damp(old, new, ev[b], return_share=True)
        sh = share.numpy()                                        # (B, L, t): the old crystal's kept share
        frozen.append([float(sh[..., lo:hi].mean()) for lo, hi in ((0, 16), (16, 80), (80, 128))])
        return out

    m(content0, clocks, mask=mask, damp=damp)
    for b, fr in enumerate(frozen):
        print(f"  breath {b + 1}: mean evidence {float(ev[b].mean().numpy()):.2f}   share kept "
              f"root {fr[0]:.2f} / branch {fr[1]:.2f} / leaf {fr[2]:.2f}")

    print("\n== the seal, numerically (tiny body; Jacobian rank of breath 2's falling state w.r.t. breath 1's) ==")
    for skip in ("peak", "none", "boundary"):
        rk, n, bound = _seal_rank(skip)
        verdict = "sealed: all of it crossed the crystal" if rk <= bound else "BYPASSED: the crystal is optional"
        print(f"  skip={skip:8s} rank {rk:2d} / {n}  (crystal bound {bound})  {verdict}")

    print("\n== chart rows (content width per block application) ==")
    for arm in ("SINE4", "NAZARE"):
        print(f"  {arm}: {[w for r in make_schedule(arm) for w in r['blocks']]}")


if __name__ == "__main__":
    main()
