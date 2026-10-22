"""cross_join_toy.py -- T3, THE CROSS-JOIN TOY (2026-10-22; branch binding-comprehension; Bryce: "our true superpower is
lossless binding and unbinding so AI doesn't memorize the SQL cross join -- NOT superposition"; "adding the snap sounds
good"; "save composite keys with meta data to multiple registers to avoid losing data"; Opus's corrections; ledger "T3").

THE CLAIM UNDER TEST. Factorization: learn 32 owners, 8 objects, 20 values, 7 days as separate pieces; build the
composite key (owner (x) object) at read time; never store the cross join. 'Lossless' in its exact form (Opus): any ONE
factor is recovered exactly given the others (unbind by the inverse), which is what a query needs.

THE WORLD. An episode = K registers (one per record). A record = (key = owner (x) object, partner, value, day[meta]).
A depth-D query is a CHAIN: start at (e0, o); read the matched record's PARTNER e1; match (e1, o); ...; at the D-th
record read the VALUE. Depth 1 = a plain lookup. (owner, object) is unique per episode; the chain's object is shared.
COVERAGE: per object, only a fraction c of the 32 owners ever own it in training (50 / 25 / 12.5 %); UNSEEN eval chains
use only uncovered (owner, object) pairs (distractors anywhere). Equal memory: every arm stores 128 floats per register
except TENSOR (the lossless, width-growing ceiling: 512) and REDUNDANT_2X (256).

ARMS
  analytic (fixed random codes; nothing to train; coverage cannot touch them by construction -- LABELED CONSTRUCTION):
    PHASOR   64 complex planes, unit-phase codes, register = sum_f role_f * filler_f, key filler = e * o; exact unbind
             (multiply by the conjugate role).
    MAP      128 real +-1 codes, elementwise product binding (self-inverse, exact).
    HRR      128 real Gaussian codes, circular-convolution binding, correlation unbind (APPROXIMATE).
    CONCAT   four 32-float slots side by side (lossless per slot, the width split instead of superposed).
    TENSOR   four 128-float slots (Smolensky with orthogonal roles = full-width slots): the lossless ceiling, 4x width.
    REDUNDANT     Bryce's form: each record written to TWO registers of W/4 complex planes (independent roles +
                  codebooks) WITH ITS METADATA (the day field doubles as a checksum: a copy whose day field does not
                  decode cleanly, best similarity < 0.5, is treated as lost); reads sum the surviving copies. Equal floats.
    REDUNDANT_2X  the same with two registers of W/2 planes (2x floats).
WIDTH: every arm is run at W = 32 (the BAR width: crosstalk bites) and W = 128 (reported; the smoke saturated there).
Analytic sizes scale with W (PHASOR W/2 planes, MAP/HRR W, CONCAT W/4 per slot, TENSOR W per slot).
  trained (coverage matters):
    PHASOR_L  PHASOR's algebra with LEARNED code phases (all four codebooks). Its init is itself a valid random code, so
              it STARTS factored: the arm asks whether training on 12.5 % of the cross join breaks the imposed product
              structure, i.e. X1/X2 race IMPOSED structure (the algebra) against FREE-FORM learned memories (FLAT, COMPRESS).
    COMPRESS  fixed MAP fillers, the 4x128 concatenation compressed to 128 by a LEARNED linear map, learned field
              decoders and query map (Opus: the fair fixed-width rival; lossy).
    FLAT      learned embeddings -> MLP register; learned query MLP; learned field heads (memorizes, if anything does).
  Trained arms: soft record selection + soft snap at train, hard at eval ('train wide, read narrow'); supervised at
  every hop (record choice + partner + value; the attention-bootstrap law).
SNAP: after each unbind the partner estimate is either SNAPPED to the nearest codebook entry (snap) or carried RAW into the
next key (raw). Record selection is argmax in every arm at eval.
LOSS OF DATA (Bryce): at eval each register FAILS with probability p in {0, 0.1, 0.3} (replaced by noise of equal norm);
REDUNDANT's copies fail independently.
"""
import json, math, os, sys, time
mode = "smoke" if "--smoke" in sys.argv else "full"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

NE, NO, NV, ND = 32, 8, 20, 7
W = int(os.environ.get("CJ_W", "128"))                  # floats per register (the bars pin at W = 32; 128 reported)
STEPS = int(os.environ.get("CJ_STEPS", "200" if mode == "smoke" else "3000"))
BATCH, N_EVAL = 64, (200 if mode == "smoke" else 1500)
SEEDS = [0] if mode == "smoke" else [0, 1, 2]
COVS = (0.5, 0.25, 0.125)
PFAIL = (0.0, 0.1, 0.3)
DEPTHS = (1, 2, 3)
ANALYTIC = ("PHASOR", "MAP", "HRR", "CONCAT", "TENSOR", "REDUNDANT", "REDUNDANT_2X")
TRAINED = ("PHASOR_L", "COMPRESS", "FLAT")
ARMS = os.environ.get("CJ_ARMS", ",".join(ANALYTIC + TRAINED)).split(",")
LR = 2e-3


def coverage(c, seed):
    rng = np.random.default_rng(1000 + seed)
    k = max(3, int(round(c * NE)))
    return {o: set(int(x) for x in rng.permutation(NE)[:k]) for o in range(NO)}


def episode(rng, D, cov, split):
    """-> records [(e, o, partner, value, day)], chain start (e0, o), answer value, gold record index per hop."""
    o = int(rng.integers(NO))
    pool = sorted(cov[o]) if split in ("train", "seen") else sorted(set(range(NE)) - cov[o])
    chain = [int(x) for x in rng.permutation(pool)[:D]]
    recs = []
    for i, e in enumerate(chain):
        p = chain[i + 1] if i + 1 < D else int(rng.integers(NE))
        recs.append((e, o, p, int(rng.integers(NV)), int(rng.integers(ND))))
    K = int(rng.integers(8, 11)); used = {(e, o) for e in chain}
    while len(recs) < K:
        oo = int(rng.integers(NO))
        if split == "train":
            ee = int(rng.choice(sorted(cov[oo])))
        else:
            ee = int(rng.integers(NE))
        if (ee, oo) in used:
            continue
        used.add((ee, oo)); recs.append((ee, oo, int(rng.integers(NE)), int(rng.integers(NV)), int(rng.integers(ND))))
    perm = rng.permutation(len(recs)); recs = [recs[i] for i in perm]
    inv = {int(j): i for i, j in enumerate(perm)}
    return recs, (chain[0], o), recs[inv[D - 1]][3], [inv[i] for i in range(D)]


def batch(rng, B, D, cov, split):
    eps = [episode(rng, D, cov, split) for _ in range(B)]
    K = max(len(x[0]) for x in eps)
    R = np.zeros((B, K, 5), np.int64); M = np.zeros((B, K), bool)
    for b, (recs, _, _, _) in enumerate(eps):
        R[b, :len(recs)] = recs; M[b, :len(recs)] = True
    t = torch.from_numpy
    return dict(R=t(R), M=t(M), e0=torch.tensor([x[1][0] for x in eps]), o=torch.tensor([x[1][1] for x in eps]),
                y=torch.tensor([x[2] for x in eps]), hops=torch.tensor([x[3] for x in eps]), D=D)


# ---------------------------------------------------------------- algebras (fixed codes)
class Phasor:
    def __init__(self, P, g, learn=False):
        self.P = P
        mk = lambda n: torch.rand(n, P, generator=g) * 2 * math.pi
        self.th = {"e": mk(NE), "o": mk(NO), "v": mk(NV), "d": mk(ND)}; self.rl = mk(4)
        if learn:
            self.th = {k: nn.Parameter(v) for k, v in self.th.items()}

    def code(self, k, idx):
        return torch.polar(torch.ones(()), self.th[k][idx])

    def role(self, f):
        return torch.polar(torch.ones(self.P), self.rl[f])

    def bind(self, a, b):
        return a * b

    def unbind(self, x, a):
        return x * a.conj()

    def sim(self, x, C):                                  # x (..., P), C (n, P) -> (..., n)
        return torch.einsum("...p,np->...n", x, C.conj()).real / self.P

    def book(self, k):
        return self.code(k, torch.arange(self.th[k].shape[0]))


class MAP:
    def __init__(self, n, g):
        self.P = n
        mk = lambda m: torch.randint(0, 2, (m, n), generator=g).float() * 2 - 1
        self.c = {"e": mk(NE), "o": mk(NO), "v": mk(NV), "d": mk(ND)}; self.rl = mk(4)

    def code(self, k, idx):
        return self.c[k][idx]

    def role(self, f):
        return self.rl[f]

    def bind(self, a, b):
        return a * b

    def unbind(self, x, a):
        return x * a

    def sim(self, x, C):
        return x @ C.T / self.P

    def book(self, k):
        return self.c[k]


class HRR:
    def __init__(self, n, g):
        self.P = n
        mk = lambda m: torch.randn(m, n, generator=g) / math.sqrt(n)
        self.c = {"e": mk(NE), "o": mk(NO), "v": mk(NV), "d": mk(ND)}; self.rl = mk(4)

    def code(self, k, idx):
        return self.c[k][idx]

    def role(self, f):
        return self.rl[f]

    def bind(self, a, b):
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=self.P)

    def unbind(self, x, a):                               # circular correlation (the approximate inverse)
        return torch.fft.irfft(torch.fft.rfft(x) * torch.fft.rfft(a).conj(), n=self.P)

    def sim(self, x, C):
        return x @ C.T

    def book(self, k):
        return self.c[k]


FIELDS = {"key": 0, "par": 1, "val": 2, "day": 3}


class VSAMemory(nn.Module):
    """One or more superposed registers per record in a given algebra. Fields: key (e (x) o), partner, value, day."""

    def __init__(self, algs, learn=False, check=False):
        super().__init__(); self.algs = algs; self.check = check
        if learn:
            self.params_ = nn.ParameterList([p for a in algs for p in a.th.values()])
        self.beta = nn.Parameter(torch.tensor(math.log(10.0)))

    def write(self, R):
        regs = []
        for a in self.algs:
            e, o, p, v, d = (a.code(k, R[..., i]) for i, k in enumerate("eoevd"))
            regs.append(a.bind(a.role(0), a.bind(e, o)) + a.bind(a.role(1), p) + a.bind(a.role(2), v) + a.bind(a.role(3), d))
        return regs

    def match(self, regs, keys):                          # keys: per alg (B, P) composite key -> (B, K)
        sc = 0
        for a, r, k in zip(self.algs, regs, keys):
            q = a.bind(a.role(0), k)
            sc = sc + (torch.einsum("bkp,bp->bk", r, q.conj()).real if torch.is_complex(r) else torch.einsum("bkp,bp->bk", r, q)) / a.P
        return sc

    def read(self, regs, w, f, book):                     # w (B, K) record weights -> logits over the codebook
        out = 0
        for a, r in zip(self.algs, regs):
            x = torch.einsum("bk,bkp->bp", w.to(r.dtype), r)
            out = out + a.sim(a.unbind(x, a.role(FIELDS[f])), a.book(book))
        return out

    def raw(self, regs, w, f):
        return [a.unbind(torch.einsum("bk,bkp->bp", w.to(r.dtype), r), a.role(FIELDS[f])) for a, r in zip(self.algs, regs)]

    def run(self, bt, snap, train, pfail=0.0, rng=None):
        regs = self.write(bt["R"])
        if pfail > 0:
            regs = [corrupt(r, pfail, rng) for r in regs]
        if self.check:                                    # THE METADATA CHECK: a copy whose meta field does not decode
            # cleanly (best day-code similarity < 0.5) is treated as lost and contributes nothing to match or read
            regs = [r * (a.sim(a.unbind(r, a.role(FIELDS["day"])), a.book("d")).max(-1).values > 0.5)[..., None].to(r.dtype)
                    for a, r in zip(self.algs, regs)]
        B = bt["R"].shape[0]; ar = torch.arange(B)
        cur = [a.code("e", bt["e0"]) for a in self.algs]; obj = [a.code("o", bt["o"]) for a in self.algs]
        sel_logits, par_logits = [], []
        for h in range(bt["D"]):
            sc = self.match(regs, [a.bind(c, ob) for a, c, ob in zip(self.algs, cur, obj)]).masked_fill(~bt["M"], -1e9)
            sel_logits.append(sc * self.beta.exp())
            w = F.softmax(sc * self.beta.exp(), -1) if train else F.one_hot(sc.argmax(-1), sc.shape[-1]).float()
            if h < bt["D"] - 1:
                lg = self.read(regs, w, "par", "e") * self.beta.exp(); par_logits.append(lg)
                if snap:
                    pw = F.softmax(lg, -1) if train else F.one_hot(lg.argmax(-1), NE).float()
                    cur = [torch.einsum("bn,np->bp", pw.to(a.book("e").dtype), a.book("e")) for a in self.algs]
                else:
                    cur = self.raw(regs, w, "par")
            else:
                return self.read(regs, w, "val", "v") * self.beta.exp(), sel_logits, par_logits


def corrupt(r, p, rng):
    fail = torch.from_numpy(rng.random(r.shape[:2]) < p)
    if torch.is_complex(r):
        noise = torch.polar(r.abs().mean() * torch.ones(r.shape), torch.from_numpy(rng.random(r.shape) * 2 * math.pi).float())
    else:
        noise = torch.from_numpy(rng.standard_normal(r.shape)).float() * r.norm(dim=-1, keepdim=True) / math.sqrt(r.shape[-1])
    return torch.where(fail[..., None], noise.to(r.dtype), r)


class SlotMemory(nn.Module):
    """CONCAT (four 32-float slots) / TENSOR (four 128-float slots): lossless per slot; key slot holds MAP(e * o)."""

    def __init__(self, width, g):
        super().__init__(); self.alg = MAP(width, g); self.beta = nn.Parameter(torch.tensor(math.log(10.0)))

    def run(self, bt, snap, train, pfail=0.0, rng=None):
        a = self.alg; R = bt["R"]
        reg = torch.stack([a.bind(a.code("e", R[..., 0]), a.code("o", R[..., 1])), a.code("e", R[..., 2]),
                           a.code("v", R[..., 3]), a.code("d", R[..., 4])], 2)            # (B, K, 4, w)
        if pfail > 0:
            reg = corrupt(reg.flatten(2), pfail, rng).view(reg.shape)
        cur = a.code("e", bt["e0"]); ob = a.code("o", bt["o"]); sel, par = [], []
        for h in range(bt["D"]):
            sc = (torch.einsum("bkp,bp->bk", reg[:, :, 0], a.bind(cur, ob)) / a.P).masked_fill(~bt["M"], -1e9)
            sel.append(sc * self.beta.exp())
            w = F.softmax(sc * self.beta.exp(), -1) if train else F.one_hot(sc.argmax(-1), sc.shape[-1]).float()
            if h < bt["D"] - 1:
                x = torch.einsum("bk,bkp->bp", w, reg[:, :, 1]); lg = a.sim(x, a.book("e")) * self.beta.exp(); par.append(lg)
                cur = a.book("e")[lg.argmax(-1)] if snap else x
            else:
                return a.sim(torch.einsum("bk,bkp->bp", w, reg[:, :, 2]), a.book("v")) * self.beta.exp(), sel, par


class Compress(nn.Module):
    def __init__(self, g):
        super().__init__(); self.alg = MAP(W, g)
        self.A = nn.Linear(4 * W, W); self.Q = nn.Linear(W, W); self.dec = nn.ModuleList([nn.Linear(W, W) for _ in range(3)])
        self.beta = nn.Parameter(torch.tensor(math.log(10.0)))

    def run(self, bt, snap, train, pfail=0.0, rng=None):
        a = self.alg; R = bt["R"]
        full = torch.cat([a.bind(a.code("e", R[..., 0]), a.code("o", R[..., 1])), a.code("e", R[..., 2]),
                          a.code("v", R[..., 3]), a.code("d", R[..., 4])], -1)
        reg = self.A(full)
        if pfail > 0:
            reg = corrupt(reg, pfail, rng)
        cur = a.code("e", bt["e0"]); ob = a.code("o", bt["o"]); sel, par = [], []
        for h in range(bt["D"]):
            q = self.Q(a.bind(cur, ob))
            sc = (torch.einsum("bkp,bp->bk", reg, q) / math.sqrt(W)).masked_fill(~bt["M"], -1e9)
            sel.append(sc)
            w = F.softmax(sc, -1) if train else F.one_hot(sc.argmax(-1), sc.shape[-1]).float()
            x = torch.einsum("bk,bkp->bp", w, reg)
            if h < bt["D"] - 1:
                xp = self.dec[0](x); lg = a.sim(xp, a.book("e")) * self.beta.exp(); par.append(lg)
                if snap:
                    pw = F.softmax(lg, -1) if train else F.one_hot(lg.argmax(-1), NE).float()
                    cur = pw @ a.book("e")
                else:
                    cur = xp
            else:
                return a.sim(self.dec[1](x), a.book("v")) * self.beta.exp(), sel, par


class Flat(nn.Module):
    def __init__(self):
        super().__init__(); d = 64
        self.E = nn.ModuleDict({k: nn.Embedding(n, d) for k, n in (("e", NE), ("o", NO), ("v", NV), ("d", ND))})
        self.wr = nn.Sequential(nn.Linear(5 * d, 256), nn.GELU(), nn.Linear(256, W))
        self.qk = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, W))
        self.hp = nn.Sequential(nn.Linear(W, 256), nn.GELU(), nn.Linear(256, d))
        self.hv = nn.Sequential(nn.Linear(W, 256), nn.GELU(), nn.Linear(256, d))

    def run(self, bt, snap, train, pfail=0.0, rng=None):
        R = bt["R"]; E = self.E
        reg = self.wr(torch.cat([E["e"](R[..., 0]), E["o"](R[..., 1]), E["e"](R[..., 2]), E["v"](R[..., 3]), E["d"](R[..., 4])], -1))
        if pfail > 0:
            reg = corrupt(reg, pfail, rng)
        cur = E["e"](bt["e0"]); ob = E["o"](bt["o"]); sel, par = [], []
        for h in range(bt["D"]):
            sc = (torch.einsum("bkp,bp->bk", reg, self.qk(torch.cat([cur, ob], -1))) / math.sqrt(W)).masked_fill(~bt["M"], -1e9)
            sel.append(sc)
            w = F.softmax(sc, -1) if train else F.one_hot(sc.argmax(-1), sc.shape[-1]).float()
            x = torch.einsum("bk,bkp->bp", w, reg)
            if h < bt["D"] - 1:
                xp = self.hp(x); lg = xp @ E["e"].weight.T; par.append(lg)
                if snap:
                    pw = F.softmax(lg, -1) if train else F.one_hot(lg.argmax(-1), NE).float()
                    cur = pw @ E["e"].weight
                else:
                    cur = xp
            else:
                return self.hv(x) @ E["v"].weight.T, sel, par


def make(arm, seed):
    g = torch.Generator().manual_seed(7000 + seed); torch.manual_seed(seed)
    if arm == "PHASOR":
        return VSAMemory([Phasor(W // 2, g)])
    if arm == "PHASOR_L":
        return VSAMemory([Phasor(W // 2, g, learn=True)], learn=True)
    if arm == "MAP":
        return VSAMemory([MAP(W, g)])
    if arm == "HRR":
        return VSAMemory([HRR(W, g)])
    if arm == "REDUNDANT":
        return VSAMemory([Phasor(W // 4, g), Phasor(W // 4, g)], check=True)
    if arm == "REDUNDANT_2X":
        return VSAMemory([Phasor(W // 2, g), Phasor(W // 2, g)], check=True)
    if arm == "CONCAT":
        return SlotMemory(W // 4, g)
    if arm == "TENSOR":
        return SlotMemory(W, g)
    if arm == "COMPRESS":
        return Compress(g)
    return Flat()


def train(net, cov, seed, snap):
    if not list(net.parameters()):
        return
    rng = np.random.default_rng(seed); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        D = int(rng.integers(1, 4)); bt = batch(rng, BATCH, D, cov, "train")
        lv, sel, par = net.run(bt, snap, True)
        loss = F.cross_entropy(lv, bt["y"])
        for h, s in enumerate(sel):
            loss = loss + F.cross_entropy(s, bt["hops"][:, h])
        for h, pl in enumerate(par):
            loss = loss + F.cross_entropy(pl, bt["R"][torch.arange(len(bt["y"])), bt["hops"][:, h], 2])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()


def evaluate(net, cov, seed, snap):
    out = {}
    with torch.no_grad():
        for split in ("seen", "unseen"):
            for D in DEPTHS:
                bt = batch(np.random.default_rng(80_000 + seed * 10 + D + (5 if split == "unseen" else 0)), N_EVAL, D, cov, split)
                for p in PFAIL:
                    lv, _, _ = net.run(bt, snap, False, p, np.random.default_rng(int(p * 1000) + seed))
                    out[f"{split}_D{D}_p{p}"] = float((lv.argmax(-1) == bt["y"]).float().mean())
    return out


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = ("_smoke" if mode == "smoke" else "") + f"_w{W}"
    assert mode == "smoke" or not os.path.exists(f".cache/cross_join{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/cross_join{tag}.txt", "w"); allres = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS} W {W}")
    A = {}
    for arm in ARMS:
        for snap in (True, False):
            for c in (COVS if arm in TRAINED else COVS[-1:]):
                rs = []
                for s in SEEDS:
                    cov = coverage(c, s); net = make(arm, s); train(net, cov, s, snap); rs.append(evaluate(net, cov, s, snap))
                a = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}; A[(arm, snap, c)] = a
                allres.append(dict(arm=arm, snap=snap, cov=c, seeds=rs, agg=a))
                say(f"{arm:12s} {'snap' if snap else 'raw '} cov {c:.3f} | " + " | ".join(
                    f"{sp} " + " ".join(f"D{D} {a[f'{sp}_D{D}_p0.0']:.3f}" for D in DEPTHS) for sp in ("seen", "unseen"))
                    + " | unseen D1 p.1/.3 " + f"{a['unseen_D1_p0.1']:.3f}/{a['unseen_D1_p0.3']:.3f}"
                    + " D3 p.1/.3 " + f"{a['unseen_D3_p0.1']:.3f}/{a['unseen_D3_p0.3']:.3f}" + f"  [{time.time() - t0:.0f}s]")
    json.dump([dict(r, snap=bool(r["snap"])) for r in allres], open(f".cache/cross_join{tag}.json", "w"), indent=1)
    if mode == "full":                                                   # THE PINNED BARS (ledger 2026-10-22, T3)
        g = lambda arm, snap, k, c=0.125: A[(arm, snap, c)][k]
        say(f"VERDICTS at W {W} (3-seed means). REAL BARS = trained arms, pinned at W 128. Analytic arms are CONSTRUCTION "
            "(fixed codes; the smoke already showed them) -> reported as 3-seed confirmations, not claims.")
        if W == 128:
            pl, fl, cp = (g(x, True, "unseen_D1_p0.0") for x in ("PHASOR_L", "FLAT", "COMPRESS"))
            say("(X1) " + ("LEARNED PHASOR CODES FACTOR: BEAT FLAT BY 0.30 ON UNSEEN PAIRS AT 12.5 % COVERAGE" if pl >= fl + 0.30 else "LEARNED PHASOR CODES DO NOT BEAT FLAT BY 0.30")
                + f"  (unseen D1 snap: PHASOR_L {pl:.3f}, FLAT {fl:.3f}, COMPRESS {cp:.3f})")
            pl3, cp3 = g("PHASOR_L", True, "unseen_D3_p0.0"), g("COMPRESS", True, "unseen_D3_p0.0")
            say("(X2) " + ("LEARNED BINDING BEATS THE LEARNED FIXED-WIDTH COMPRESSION AT DEPTH 3" if pl3 >= cp3 + 0.15 else "LEARNED BINDING DOES NOT BEAT COMPRESSION BY 0.15")
                + f"  (unseen D3 snap: PHASOR_L {pl3:.3f} vs COMPRESS {cp3:.3f})")
        say("CONSTRUCTION CONFIRMATIONS (unseen, cov 0.125):")
        for D in DEPTHS:
            say(f"   D{D} p0 snap/raw: " + ", ".join(f"{x} {g(x, True, f'unseen_D{D}_p0.0'):.3f}/{g(x, False, f'unseen_D{D}_p0.0'):.3f}" for x in ANALYTIC if x in ARMS))
        for pp in PFAIL[1:]:
            say(f"   LOSS p {pp} snap D1/D3: " + ", ".join(f"{x} {g(x, True, f'unseen_D1_p{pp}'):.3f}/{g(x, True, f'unseen_D3_p{pp}'):.3f}" for x in ANALYTIC if x in ARMS))
        if W == 128:
            say("COVERAGE BREAK (trained arms, unseen D1 / D3 snap, cov 0.5/0.25/0.125): " + ", ".join(
                f"{x} " + "/".join(f"{g(x, True, 'unseen_D1_p0.0', c):.3f}:{g(x, True, 'unseen_D3_p0.0', c):.3f}" for c in COVS) for x in TRAINED))
    say(f"DONE in {time.time() - t0:.0f}s")
