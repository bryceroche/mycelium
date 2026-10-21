"""time_content_rung.py -- THE TIME-VS-CONTENT RUNG: FFT WHERE IT BELONGS, AND THE CLOCK AS A RUNNING FOURIER TRANSFORM
(2026-10-20; Bryce's word: 'combine the T6 clock with FFT'; Opus: add an LRU arm). CPU only.

THE CLAIM UNDER TEST: Fourier mixing over the SEQUENCE axis is shift-equivariant, so it should do POSITION-addressed lookup
('the event k back') and fail CONTENT-addressed lookup ('the value of key X'); a memory of clock-stamped events,
M = sum_t clock(t) (*) e(x_t), IS a running DFT of the timeline, so 'k back' is one rotation (the shift theorem).
TASKS (fresh sequences every batch; one answer token per sequence):
  TIME     N events (40 entity tokens), then a query token 'k back' (k in 1..8): the answer is the event k before the
           last one (k = 1 -> the second-to-last... precisely: events x_0..x_{N-1}; answer x_{N-1-k}). Train N ~ U[12, 64].
  CONTENT  MQAR (Zoology): N pairs (key in 40 keys, value in 40 values; keys distinct), then a query key: answer its
           value. Train N ~ U[4, 32]. FUSED since feas v3: token = key + value embeddings; 8 supervised queries.
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
THE REGISTERS EXTENSION (2026-10-20 ~20:45; Bryce: 'separate registers so it doesn't fill up'; Opus: learned routing,
an equal-memory attention rival, a log-spread LRU, the decay bar):
  TIMELONG  as TIME but k ~ U[1, min(N-1, 63)] (k tokens up to 63 are all seen in training; N 128 = length extrapolation
            with long reach, never unseen k). Vocab grew to 63 k-tokens for every arm (the feas reads used 8).
  CLOCK_reg   CEILING ARM: events stored in registers of W = 16 by ABSOLUTE time (register floor(t/W)), the fixed clock
              inside each; the read picks register floor((N-1-k)/W) BY HAND (the address supplied: not a win).
  CLOCK_regL  the registers with LEARNED soft routing: each register's key = the mean SLOW-clock stamp of its events
              (16 planes, learned frequencies, init periods 256..2048 ticks: no aliasing over 128), the query = the slow
              stamp of the target time N-1-k, weights softmax(beta_r cos); bands as the ADDRESS, registers as STORAGE.
  LRU_spread  learned frequencies, decay FIXED log-spaced per plane in [1e-4, 0.3]/tick: the strongest single vector.
  ATT_m8      ATT with an EQUAL-MEMORY cache: one head, k and v of 8 dims per layer, 2 layers -> 32 floats per event,
              = CLOCK_reg's 512 floats per 16 events. Memory floats per sequence are reported with the parameters.
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

NE, KMAX, KLONG, D, NL, MP, NQ, W, MS = 40, 8, 63, 128, 2, 256, 8, 16, 16
TIME_N, CONT_N = (12, 64), (4, 32)
TIME_EVAL, CONT_EVAL = (16, 32, 64, 128), (8, 16, 32)          # TIME N 128 = LENGTH EXTRAPOLATION (trained <= 64)
STEPS = 300 if mode == "smoke" else int(os.environ.get("TC_STEPS", "4000"))
BATCH, N_EVAL = 64, (300 if mode == "smoke" else 1000)
SEEDS = [0] if mode != "full" else [0, 1, 2]
MIXERS = ["ATT", "FNET", "GF", "GGF"]
MIXERS = MIXERS + ["ATT_m8"]
MEMS = ["CLOCK_full", "CLOCK_ms", "LRU_freq", "LRU", "LRU_spread", "CLOCK_reg", "CLOCK_regL"]
ARMS = os.environ.get("TC_ARMS", ",".join(MIXERS + MEMS)).split(",")
# vocab: events 0..39 | keys 40..79 | values 80..119 | 'k back' 120..182 | PAD 183
V_KEY, V_VAL, V_K, PADT, VOCAB = 40, 80, 120, 120 + KLONG, 121 + KLONG


def batch_time(B, rng, n=None, long=False):
    ns = rng.integers(TIME_N[0], TIME_N[1] + 1, size=B) if n is None else np.full(B, n)
    L = int(ns.max()) + 1; x = np.full((B, L), PADT, np.int64); ev = np.zeros((B, L - 1), np.int64)
    k = (np.array([rng.integers(1, min(m - 1, KLONG) + 1) for m in ns]) if long else rng.integers(1, KMAX + 1, size=B))
    t = np.zeros(B, np.int64)
    for b in range(B):
        e = rng.integers(NE, size=ns[b]); x[b, :ns[b]] = e; x[b, ns[b]] = V_K + k[b] - 1; t[b] = e[ns[b] - 1 - k[b]]
        ev[b, :ns[b]] = e
    return tuple(torch.from_numpy(a) for a in (x, ns, k, t, ev))


def batch_content(B, rng, n=None):                                           # FUSED fact tokens (feas v3): token i = key_i + value_i
    ns = rng.integers(CONT_N[0], CONT_N[1] + 1, size=B) if n is None else np.full(B, n)   # (embeddings summed), then NQ
    L = int(ns.max()) + NQ; x = np.full((B, L), PADT, np.int64); xv = np.full((B, L), -1, np.int64)   # supervised query keys.
    t = np.zeros((B, NQ), np.int64)                                          # v1/v2 (key, value as two tokens) sat on the
    for b in range(B):                                                       # induction plateau in ALL nets incl. stock torch:
        ks = rng.choice(NE, ns[b], replace=False); vs = rng.integers(NE, size=ns[b]); qi = rng.integers(ns[b], size=NQ)
        x[b, :ns[b]] = V_KEY + ks; xv[b, :ns[b]] = V_VAL + vs                # that read tests a 2-hop copy circuit, not lookup
        x[b, ns[b]:ns[b] + NQ] = V_KEY + ks[qi]; t[b] = vs[qi]
    return torch.from_numpy(x), torch.from_numpy(ns), torch.from_numpy(xv), torch.from_numpy(t), None


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
            self.qkv, self.o, self.nh, self.dk = nn.Linear(D, 3 * D), nn.Linear(D, D), 4, D
        elif kind == "ATT_m8":                                               # equal-memory cache: k, v of 8 dims, one head
            self.qkv, self.o, self.nh, self.dk = nn.Linear(D, 3 * 8), nn.Linear(8, D), 1, 8
        elif kind in ("GF", "GGF"):
            nf = Lmax + 1                                                    # rfft of a 2*Lmax zero-padded sequence
            self.filt = nn.Parameter(torch.randn(nf, D, 2) * 0.02); self.filt.data[..., 0] += 1.0 / Lmax
            if kind == "GGF": self.v, self.g, self.o = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        self.Lmax = Lmax

    def forward(self, h, pad):
        B, L, _ = h.shape
        if self.kind in ("ATT", "ATT_m8"):
            nh, dk = self.nh, self.dk
            q, k, v = (t.reshape(B, L, nh, dk // nh).transpose(1, 2) for t in self.qkv(h).chunk(3, -1))
            s = rope(q) @ rope(k).transpose(-1, -2) / math.sqrt(dk // nh)
            a = F.softmax(s.masked_fill(pad[:, None, None, :], -1e4), -1)
            return self.o((a @ v).transpose(1, 2).reshape(B, L, dk))
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
        if task == "content": h = h + self.emb(k.clamp(min=0)) * (k >= 0)[..., None]   # k carries the fused value channel
        for b in self.blocks: h = b(h, pad)
        if task != "content":
            return self.out(self.lnf(h[torch.arange(len(x)), ns]))           # the query token's position
        qp = ns[:, None] + torch.arange(NQ)[None]                            # the NQ query positions
        return self.out(self.lnf(h[torch.arange(len(x))[:, None], qp]))


# ---------------------------------------------------------------- memory arms (time only)
class ClockMem(nn.Module):
    def __init__(self, kind, seed):
        super().__init__(); g = torch.Generator().manual_seed(4000 + seed); self.kind = kind
        if kind in ("CLOCK_full", "LRU_freq", "LRU", "LRU_spread", "CLOCK_reg", "CLOCK_regL"):                       # the LRUs start FROM the full-band clock (Orvieto's
            om = torch.rand(MP, generator=g) * 2 * math.pi                    # max-phase 2 pi ring), so learned-vs-fixed is paired
        else:
            om = 2 * math.pi / torch.exp(torch.linspace(math.log(2.0), math.log(1000.0), MP))
        if kind.startswith("LRU"):
            self.om = nn.Parameter(om)
        else:
            self.register_buffer("om", om)
        if kind == "LRU":
            self.lognu = nn.Parameter(torch.full((MP,), math.log(1e-3)))
        if kind == "LRU_spread":                                             # fixed log-spread decay; frequencies learned
            self.register_buffer("lognu", torch.linspace(math.log(1e-4), math.log(0.3), MP)[torch.randperm(MP, generator=g)])
        if kind == "CLOCK_regL":                                             # the slow address band (learned frequencies)
            self.oms = nn.Parameter(2 * math.pi / torch.exp(torch.linspace(math.log(256.0), math.log(2048.0), MS)))
            self.log_beta_r = nn.Parameter(torch.tensor(math.log(5.0)))
        te = torch.rand(NE, MP, generator=g) * 2 * math.pi
        self.ent = nn.Parameter(torch.cat([torch.cos(te), torch.sin(te)], -1))
        self.log_beta = nn.Parameter(torch.tensor(math.log(2.0)))

    def forward(self, x, ns, k, task, ev=None):
        B, L = ev.shape
        age = (ns[:, None] - 1 - torch.arange(L)[None]).float()              # N-1-t: ticks before the last event
        live = (age >= 0).float()
        nu = self.lognu.exp() if self.kind in ("LRU", "LRU_spread") else torch.zeros(MP)
        ang = age[..., None] * self.om; mag = torch.exp(-age[..., None].clamp(min=0) * nu) * live[..., None]
        e = self.ent[ev]; er, ei = e[..., :MP], e[..., MP:]
        c, s = torch.cos(ang) * mag, torch.sin(ang) * mag                    # lambda^age
        pr, pi = er * c - ei * s, er * s + ei * c                            # each event, stamped (B, L, MP)
        kk = k.float()[:, None]                                              # read: h (*) lambda^-k
        ra, rm = -kk * self.om, torch.exp(kk * nu)
        cr, ci = torch.cos(ra) * rm, torch.sin(ra) * rm
        if self.kind in ("CLOCK_reg", "CLOCK_regL"):
            R = (L + W - 1) // W; tt = torch.arange(L); reg = F.one_hot(tt // W, R).float()   # (L, R) absolute-time registers
            hr, hi = torch.einsum("blm,lr->brm", pr, reg), torch.einsum("blm,lr->brm", pi, reg)
            tgt = (ns - 1 - k).float()                                       # the target's absolute time
            if self.kind == "CLOCK_reg":
                a = F.one_hot((tgt.long() // W).clamp(max=R - 1), R).float()  # the address BY HAND (ceiling)
            else:
                sa = tt.float()[:, None] * self.oms[None]                    # slow stamps (L, MS)
                sk = torch.einsum("blr,ls->brs", live[..., None] * reg[None], torch.cos(sa)), \
                     torch.einsum("blr,ls->brs", live[..., None] * reg[None], torch.sin(sa))
                key = F.normalize(torch.cat(sk, -1), dim=-1)                 # (B, R, 2MS): the register's mean slow stamp
                qa = tgt[:, None] * self.oms[None]
                q = F.normalize(torch.cat([torch.cos(qa), torch.sin(qa)], -1), dim=-1)
                filled = reg.T[None].mul(live[:, None, :]).sum(-1) > 0      # (B, R) registers holding any event
                lg = (self.log_beta_r.exp() * (key @ q[..., None]).squeeze(-1)).masked_fill(~filled, -1e4)
                a = F.softmax(lg, -1)
            hr, hi = (a[..., None] * hr).sum(1), (a[..., None] * hi).sum(1)
        else:
            hr, hi = pr.sum(1), pi.sum(1)
        ur, ui = hr * cr - hi * ci, hr * ci + hi * cr
        u = torch.cat([ur, ui], -1)
        return self.log_beta.exp() * F.normalize(u, dim=-1) @ F.normalize(self.ent, dim=-1).T


def build(arm, seed):
    torch.manual_seed(seed)
    if arm in MIXERS: return MixNet(arm, TIME_EVAL[-1] + 1)                 # filters sized for the N 128 read
    return ClockMem(arm, seed)


def memfloats(arm, n):                                                       # what the arm holds about a length-n sequence
    if arm == "ATT": return NL * 2 * D * n
    if arm == "ATT_m8": return NL * 2 * 8 * n
    if arm in ("CLOCK_reg", "CLOCK_regL"): return -(-n // W) * 2 * MP
    if arm in MEMS: return 2 * MP
    return NL * D * n                                                        # the filters' hidden state (no cache)


def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def run(arm, seed, lr, task):
    rng = np.random.default_rng(seed * 7 + {"time": 0, "content": 1, "timelong": 2}[task]); erng = np.random.default_rng(10_000 + seed)
    mk = {"time": batch_time, "timelong": lambda B, r, n=None: batch_time(B, r, n, long=True), "content": batch_content}[task]
    ev = {n: mk(N_EVAL, erng, n) for n in (CONT_EVAL if task == "content" else TIME_EVAL)}
    net = build(arm, seed); mem = arm in MEMS
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.1)
    fwd = lambda bt: net(bt[0], bt[1], bt[2], task, bt[4]) if mem else net(bt[0], bt[1], bt[2], task)
    for step in range(STEPS):
        bt = mk(BATCH, rng); loss = F.cross_entropy(fwd(bt).reshape(-1, NE), bt[3].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); res = {}
    with torch.no_grad():
        for n, bt in ev.items():
            pr = fwd(bt).argmax(-1); res[f"N{n}"] = float((pr == bt[3]).float().mean())
    res["mean"] = float(np.mean([res[f"N{n}"] for n in ev if n <= (CONT_N if task == "content" else TIME_N)[1]]))   # in-range only
    res["params"] = nparams(net); res["final_loss"] = float(loss.detach())
    if arm == "LRU": res["mean_decay_per_tick"] = float(net.lognu.detach().exp().mean())
    res["mem_floats_N128"] = memfloats(arm, 128)
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"feas": "_feas", "smoke": "_smoke", "full": ""}[mode] + os.environ.get("TC_TAG", "")
    assert mode == "smoke" or not os.path.exists(f".cache/time_content_rung{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/time_content_rung{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS}; params: " + ", ".join(f"{a} {nparams(build(a, 0))}" for a in ARMS))
    TASKS = os.environ.get("TC_TASKS", "time,timelong,content").split(",")
    for task in TASKS:
        for arm in ARMS:
            if task == "content" and (arm in MEMS or arm == "ATT_m8"): continue
            if mode == "full":
                sc = {lr: run(arm, 0, lr, task)["mean"] for lr in (1e-3, 3e-3)}; lr = max(sc, key=sc.get)
                say(f"lr sweep {task} {arm}: {sc} -> {lr}  [{time.time() - t0:.0f}s]")
            else:
                lr = float(os.environ.get("TC_LR", "3e-3"))
            rs = [run(arm, s, lr, task) for s in SEEDS]
            A = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            res.append(dict(task=task, arm=arm, lr=lr, seeds=rs, agg=A))
            say(f"{task:<7} {arm:<10} params {rs[0]['params']:>7} loss {A['final_loss']:.3f}  " +
                "  ".join(f"{k} {v:.3f}" for k, v in A.items() if k.startswith("N") or k in ("mean", "mean_decay_per_tick", "mem_floats_N128"))
                + f"  [{time.time() - t0:.0f}s]")
    json.dump(res, open(f".cache/time_content_rung{tag}.json", "w"), indent=1)
    R = {(r["task"], r["arm"]): r["agg"] for r in res}
    if mode == "feas":
        ok = all(max(a["mean"] for (t, _), a in R.items() if t == tk) >= 0.80 for tk in TASKS)
        say(f"FEASIBILITY: some arm >= 0.80 on each task: {ok} -> {'FEASIBLE (pin the bars)' if ok else 'NOT FEASIBLE'}")
    if mode == "full":                                                       # THE PINNED BARS (ledger 2026-10-20 ~20:55)
        g = lambda t, a, k: R[(t, a)][k]
        m2 = lambda t, a: 0.5 * (g(t, a, "N64") + g(t, a, "N128"))
        v = lambda c, y, n: y if c else n
        say("VERDICTS (3-seed means)")
        say("T1 " + v(min(g("time", a, "mean") for a in ("GF", "GGF")) >= g("time", "ATT", "mean") - 0.03, "FILTERS DO POSITION", "FILTERS DO NOT TIE ATT ON TIME"))
        say("T2 " + v(g("time", "LRU_freq", "mean") - g("time", "CLOCK_full", "mean") >= 0.05, "LEARNED FREQUENCIES NEEDED", "LEARNED FREQUENCIES NOT NEEDED"))
        say("T3 " + v(g("time", "LRU", "N128") - g("time", "LRU_freq", "N128") >= 0.10, "DECAY HELPS (short k)", "DECAY DOES NOT HELP"))
        say("(a) CEILING CHECK " + v(all(g("timelong", "CLOCK_reg", f"N{n}") >= 0.95 for n in TIME_EVAL), "OK", "FAILED (a bug, not a finding)"))
        say("(a') " + v(all(g("timelong", "CLOCK_regL", f"N{n}") >= 0.90 for n in TIME_EVAL), "REGISTERS HOLD", "REGISTERS DO NOT HOLD")
            + f"  (routing cost long-k N128 {g('timelong', 'CLOCK_reg', 'N128') - g('timelong', 'CLOCK_regL', 'N128'):+.3f})")
        asked = g("timelong", "LRU", "mean_decay_per_tick") <= 0.5 * g("time", "LRU", "mean_decay_per_tick")
        say("(e) " + v(asked, "THE TASK ASKED (long-k decay <= half the short-k decay)", "UNASKED (the LRU's decay did not move)")
            + f"  (nu short {g('time', 'LRU', 'mean_decay_per_tick'):.4f}, long {g('timelong', 'LRU', 'mean_decay_per_tick'):.4f})")
        say("(b) " + (v(m2("timelong", "LRU") <= m2("timelong", "CLOCK_regL") - 0.20, "ONE VECTOR'S PRICE PAID", "NO PRICE") if asked else "UNASKED")
            + f"  (LRU {m2('timelong', 'LRU'):.3f} vs CLOCK_regL {m2('timelong', 'CLOCK_regL'):.3f}, N64-128)")
        say("(b') " + v(m2("timelong", "LRU_spread") >= m2("timelong", "CLOCK_regL") - 0.05, "THE SINGLE VECTOR SUFFICES (registers = memory size only)", "REGISTERS BEAT THE STRONGEST SINGLE VECTOR")
            + f"  (LRU_spread {m2('timelong', 'LRU_spread'):.3f})")
        d = m2("timelong", "CLOCK_regL") - m2("timelong", "ATT_m8")
        say("(c) " + v(g("time", "CLOCK_regL", "mean") >= g("time", "LRU", "mean") - 0.03 and g("time", "CLOCK_regL", "N128") >= g("time", "LRU", "N128") - 0.03, "NO SHORT-K COST", "REGISTERS COST ON SHORT K"))
        say("(f) " + ("REGISTERS BEAT EQUAL-MEMORY ATTENTION" if d >= 0.05 else "THE MEMORY CLAIM DIES" if d <= -0.05 else "TIE AT EQUAL MEMORY") + f"  (CLOCK_regL - ATT_m8 N64-128 {d:+.3f})")
        ca, cf, cg = (g("content", a, "mean") for a in ("ATT", "GF", "GGF"))
        say("(d) " + v(ca - max(cf, cg) >= 0.40, "CONTENT NEEDS ATTENTION", "FILTERS DO CONTENT") + "; " + v(cg - cf < 0.5 * (ca - cf), "GATING PARTIAL", "GATING RESCUES CONTENT"))
        acc = all(g(t, "CLOCK_regL", "mean") >= g(t, "ATT", "mean") - 0.03 for t in ("time", "timelong"))
        ln = all(g(t, "CLOCK_regL", "N128") >= g(t, "ATT", "N128") + 0.05 for t in ("time", "timelong"))
        say(f"BRYCE'S PREDICTION 'THE CLOCK WINS THE TIME TRIAL': accuracy {'TIE' if acc else 'LOSS'}; parameters WIN by construction ("
            f"{nparams(build('CLOCK_regL', 0))} vs {nparams(build('ATT', 0))}); length {'WIN' if ln else 'NO WIN'} -> "
            + v(acc and ln, "THE CLOCK WINS", "THE CLOCK DOES NOT WIN OUTRIGHT"))
    say(f"DONE in {time.time() - t0:.0f}s")
