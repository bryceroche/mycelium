"""time_content_rung.py -- THE TIME-VS-CONTENT RUNG: FFT WHERE IT BELONGS, AND THE CLOCK AS A RUNNING FOURIER TRANSFORM
(2026-10-20; Bryce's word: 'combine the T6 clock with FFT'; Opus: add an LRU arm). CPU only.

THE CLAIM UNDER TEST: Fourier mixing over the SEQUENCE axis is shift-equivariant, so it should do POSITION-addressed lookup
('the event k back') and fail CONTENT-addressed lookup ('the value of key X'); a memory of clock-stamped events,
M = sum_t clock(t) (*) e(x_t), IS a running DFT of the timeline, so 'k back' is one rotation (the shift theorem).
TASKS (fresh sequences every batch; one answer token per sequence):
  TIME     N events (40 entity tokens), then a query token 'k back' (k in 1..8): the answer is the event k before the
           last one (k = 1 -> the second-to-last... precisely: events x_0..x_{N-1}; answer x_{N-1-k}). Train N ~ U[12, 64].
  CONTENT  MQAR (Zoology): N pairs (key in 40 keys, value in 40 values; keys distinct), then a query key: answer its
           value. Train N ~ U[4, 32].
  Eval per N: TIME at N = 16 / 32 / 64; CONTENT at N = 8 / 16 / 32.
MIXER ARMS (2 layers, D 128, SwiGLU MLP, pre-LN residual; token + nothing else unless stated; readout at the last token):
  ATT     softmax attention with RoPE (the strong baseline; relative position is native to it)
  FNET    Re(FFT_seq(FFT_hidden(x))): fixed, parameter-free mixing (FNet)
  GF      global filter (GFNet): rfft over the sequence (zero-padded to 2x length = linear, not circular, convolution),
          times a LEARNED complex filter per channel and frequency, irfft
  GGF     gated global filter (Hyena order-2 style): GF(x W_v) (*) (x W_g)
MEMORY ARMS (TIME only; 256 complex planes; entity codes LEARNED unit-init phasors; read by an exact rotation, then a
soft snap over the 40 entities (cosine, learned beta)); content lookup is not representable in them by construction
(no content key: the (who, when) task binds content in, the third step):
  CLOCK_full  fixed frequencies uniform in [0, 2 pi) (full band: a hash in time)
  CLOCK_ms    fixed frequencies geometric, periods 2..1000 ticks (multiscale, RoPE-style)
  LRU_freq    LEARNED frequencies (init = CLOCK_full, the same draw: a paired comparison; the smoke showed the ms init
              starts LRU at CLOCK_ms's weak level), no decay (|lambda| = 1)
  LRU         LEARNED frequencies and decay, lambda = exp(-nu + i omega) (Orvieto et al. 2023); read h (*) lambda^-(k+... )
  memory at the end: h = sum_t lambda^(N-1-t) e(x_t); answer x_{N-1-k}: u = h (*) lambda^(-k)  (|lambda| = 1: conj^k).
Bars pin in the ledger after --feas and before the full run.
Usage: .venv/bin/python3 scripts/toys/time_content_rung.py [--feas | --smoke] -> .cache/time_content_rung{tag}.*
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mode = "feas" if "--feas" in sys.argv else "smoke" if "--smoke" in sys.argv else "full"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

NE, KMAX, D, NL, MP = 40, 8, 128, 2, 256
TIME_N, CONT_N = (12, 64), (4, 32)
TIME_EVAL, CONT_EVAL = (16, 32, 64, 128), (8, 16, 32)          # TIME N 128 = LENGTH EXTRAPOLATION (trained <= 64)
STEPS = 300 if mode == "smoke" else int(os.environ.get("TC_STEPS", "4000"))
BATCH, N_EVAL = 64, (300 if mode == "smoke" else 1000)
SEEDS = [0] if mode != "full" else [0, 1, 2]
MIXERS = ["ATT", "FNET", "GF", "GGF"]
MEMS = ["CLOCK_full", "CLOCK_ms", "LRU_freq", "LRU"]
ARMS = os.environ.get("TC_ARMS", ",".join(MIXERS + MEMS)).split(",")
# vocab: events 0..39 | keys 40..79 | values 80..119 | 'k back' 120..127 | PAD 128
V_KEY, V_VAL, V_K, PADT, VOCAB = 40, 80, 120, 128, 129


def batch_time(B, rng, n=None):
    ns = rng.integers(TIME_N[0], TIME_N[1] + 1, size=B) if n is None else np.full(B, n)
    L = int(ns.max()) + 1; x = np.full((B, L), PADT, np.int64); ev = np.zeros((B, L - 1), np.int64)
    k = rng.integers(1, KMAX + 1, size=B); t = np.zeros(B, np.int64)
    for b in range(B):
        e = rng.integers(NE, size=ns[b]); x[b, :ns[b]] = e; x[b, ns[b]] = V_K + k[b] - 1; t[b] = e[ns[b] - 1 - k[b]]
        ev[b, :ns[b]] = e
    return tuple(torch.from_numpy(a) for a in (x, ns, k, t, ev))


def batch_content(B, rng, n=None):
    ns = rng.integers(CONT_N[0], CONT_N[1] + 1, size=B) if n is None else np.full(B, n)
    L = 2 * int(ns.max()) + 1; x = np.full((B, L), PADT, np.int64); t = np.zeros(B, np.int64)
    for b in range(B):
        ks = rng.choice(NE, ns[b], replace=False); vs = rng.integers(NE, size=ns[b]); i = rng.integers(ns[b])
        x[b, 0:2 * ns[b]:2] = V_KEY + ks; x[b, 1:2 * ns[b]:2] = V_VAL + vs; x[b, 2 * ns[b]] = V_KEY + ks[i]; t[b] = vs[i]
    return torch.from_numpy(x), torch.from_numpy(ns), None, torch.from_numpy(t), None


# ---------------------------------------------------------------- mixers
def rope(x):                                                                 # x (B, H, L, hd)
    L, hd = x.shape[-2], x.shape[-1]
    f = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd)); a = torch.arange(L).float()[:, None] * f[None]
    c, s = torch.cos(a), torch.sin(a); x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Mix(nn.Module):
    def __init__(self, kind, Lmax):
        super().__init__(); self.kind = kind
        if kind == "ATT":
            self.qkv, self.o = nn.Linear(D, 3 * D), nn.Linear(D, D)
        elif kind in ("GF", "GGF"):
            nf = Lmax + 1                                                    # rfft of a 2*Lmax zero-padded sequence
            self.filt = nn.Parameter(torch.randn(nf, D, 2) * 0.02); self.filt.data[..., 0] += 1.0 / Lmax
            if kind == "GGF": self.v, self.g, self.o = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        self.Lmax = Lmax

    def forward(self, h, pad):
        B, L, _ = h.shape
        if self.kind == "ATT":
            q, k, v = (t.view(B, L, 4, D // 4).transpose(1, 2) for t in self.qkv(h).chunk(3, -1))
            s = rope(q) @ rope(k).transpose(-1, -2) / math.sqrt(D // 4)
            a = F.softmax(s.masked_fill(pad[:, None, None, :], -1e4), -1)
            return self.o((a @ v).transpose(1, 2).reshape(B, L, D))
        if self.kind == "FNET":
            return torch.fft.fft(torch.fft.fft(h.to(torch.complex64), dim=-1), dim=-2).real
        u = self.v(h) if self.kind == "GGF" else h
        u = u.masked_fill(pad[..., None], 0.0)
        U = torch.fft.rfft(u, n=2 * self.Lmax, dim=1)                        # linear convolution (no wrap)
        W = torch.view_as_complex(self.filt.contiguous())
        y = torch.fft.irfft(U * W[None], n=2 * self.Lmax, dim=1)[:, :L]
        return self.o(y * self.g(h)) if self.kind == "GGF" else y


class Block(nn.Module):
    def __init__(self, kind, Lmax):
        super().__init__(); self.mix = Mix(kind, Lmax)
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.w1, self.w3, self.w2 = nn.Linear(D, 2 * D), nn.Linear(D, 2 * D), nn.Linear(2 * D, D)

    def forward(self, h, pad):
        h = h + self.mix(self.ln1(h), pad); z = self.ln2(h)
        return h + self.w2(F.silu(self.w1(z)) * self.w3(z))


class MixNet(nn.Module):
    def __init__(self, kind, Lmax):
        super().__init__(); self.emb = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList([Block(kind, Lmax) for _ in range(NL)]); self.lnf, self.out = nn.LayerNorm(D), nn.Linear(D, NE)

    def forward(self, x, ns, k, task):
        pad = x == PADT; h = self.emb(x)
        for b in self.blocks: h = b(h, pad)
        last = (ns if task == "time" else 2 * ns)                            # the query token's position
        return self.out(self.lnf(h[torch.arange(len(x)), last]))


# ---------------------------------------------------------------- memory arms (time only)
class ClockMem(nn.Module):
    def __init__(self, kind, seed):
        super().__init__(); g = torch.Generator().manual_seed(4000 + seed); self.kind = kind
        if kind in ("CLOCK_full", "LRU_freq", "LRU"):                       # the LRUs start FROM the full-band clock (Orvieto's
            om = torch.rand(MP, generator=g) * 2 * math.pi                    # max-phase 2 pi ring), so learned-vs-fixed is paired
        else:
            om = 2 * math.pi / torch.exp(torch.linspace(math.log(2.0), math.log(1000.0), MP))
        if kind.startswith("LRU"):
            self.om = nn.Parameter(om)
        else:
            self.register_buffer("om", om)
        if kind == "LRU":
            self.lognu = nn.Parameter(torch.full((MP,), math.log(1e-3)))
        te = torch.rand(NE, MP, generator=g) * 2 * math.pi
        self.ent = nn.Parameter(torch.cat([torch.cos(te), torch.sin(te)], -1))
        self.log_beta = nn.Parameter(torch.tensor(math.log(2.0)))

    def forward(self, x, ns, k, task, ev=None):
        B, L = ev.shape
        age = (ns[:, None] - 1 - torch.arange(L)[None]).float()              # N-1-t: ticks before the last event
        live = (age >= 0).float()
        nu = self.lognu.exp() if self.kind == "LRU" else torch.zeros(MP)
        ang = age[..., None] * self.om; mag = torch.exp(-age[..., None].clamp(min=0) * nu) * live[..., None]
        e = self.ent[ev]; er, ei = e[..., :MP], e[..., MP:]
        c, s = torch.cos(ang) * mag, torch.sin(ang) * mag                    # lambda^age
        hr, hi = (er * c - ei * s).sum(1), (er * s + ei * c).sum(1)
        kk = k.float()[:, None]                                              # read: h (*) lambda^-k
        ra, rm = -kk * self.om, torch.exp(kk * nu)
        cr, ci = torch.cos(ra) * rm, torch.sin(ra) * rm
        ur, ui = hr * cr - hi * ci, hr * ci + hi * cr
        u = torch.cat([ur, ui], -1)
        return self.log_beta.exp() * F.normalize(u, dim=-1) @ F.normalize(self.ent, dim=-1).T


def build(arm, seed):
    torch.manual_seed(seed)
    if arm in MIXERS: return MixNet(arm, TIME_EVAL[-1] + 1)                 # filters sized for the N 128 read
    return ClockMem(arm, seed)


def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(arm, seed, lr, task):
    rng = np.random.default_rng(seed * 7 + (0 if task == "time" else 1)); erng = np.random.default_rng(10_000 + seed)
    mk = batch_time if task == "time" else batch_content
    ev = {n: mk(N_EVAL, erng, n) for n in (TIME_EVAL if task == "time" else CONT_EVAL)}
    net = build(arm, seed); mem = arm in MEMS
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.1)
    fwd = lambda bt: net(bt[0], bt[1], bt[2], task, bt[4]) if mem else net(bt[0], bt[1], bt[2], task)
    for step in range(STEPS):
        bt = mk(BATCH, rng); loss = F.cross_entropy(fwd(bt), bt[3])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); res = {}
    with torch.no_grad():
        for n, bt in ev.items():
            pr = fwd(bt).argmax(-1); res[f"N{n}"] = float((pr == bt[3]).float().mean())
    res["mean"] = float(np.mean([res[f"N{n}"] for n in ev if n <= (TIME_N if task == "time" else CONT_N)[1]]))   # in-range only
    res["params"] = nparams(net); res["final_loss"] = float(loss.detach())
    if arm == "LRU": res["mean_decay_per_tick"] = float(net.lognu.detach().exp().mean())
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"feas": "_feas", "smoke": "_smoke", "full": ""}[mode] + os.environ.get("TC_TAG", "")
    assert mode == "smoke" or not os.path.exists(f".cache/time_content_rung{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/time_content_rung{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS}; params: " + ", ".join(f"{a} {nparams(build(a, 0))}" for a in ARMS))
    for task in ("time", "content"):
        for arm in ARMS:
            if task == "content" and arm in MEMS: continue
            if mode == "full":
                sc = {lr: run(arm, 0, lr, task)["mean"] for lr in (1e-3, 3e-3)}; lr = max(sc, key=sc.get)
                say(f"lr sweep {task} {arm}: {sc} -> {lr}  [{time.time() - t0:.0f}s]")
            else:
                lr = 3e-3
            rs = [run(arm, s, lr, task) for s in SEEDS]
            A = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            res.append(dict(task=task, arm=arm, lr=lr, seeds=rs, agg=A))
            say(f"{task:<7} {arm:<10} params {rs[0]['params']:>7} loss {A['final_loss']:.3f}  " +
                "  ".join(f"{k} {v:.3f}" for k, v in A.items() if k.startswith("N") or k in ("mean", "mean_decay_per_tick"))
                + f"  [{time.time() - t0:.0f}s]")
    json.dump(res, open(f".cache/time_content_rung{tag}.json", "w"), indent=1)
    R = {(r["task"], r["arm"]): r["agg"] for r in res}
    if mode == "feas":
        ok = max(a["mean"] for (t, _), a in R.items() if t == "time") >= 0.80 and max(a["mean"] for (t, _), a in R.items() if t == "content") >= 0.80
        say(f"FEASIBILITY: some arm >= 0.80 on each task: {ok} -> {'FEASIBLE (pin the bars)' if ok else 'NOT FEASIBLE'}")
    say(f"DONE in {time.time() - t0:.0f}s")
