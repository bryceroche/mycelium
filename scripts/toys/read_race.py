"""read_race.py -- THE READ RACE: BLEND, SNAP, MARGINALIZE, OR A BEAM? (2026-10-21; Bryce's word; Opus's known-posterior
design and top-K arm; the coordinator's calibration read). CPU only.

WHY: race 4 measured the soft-choose / hard-read split (router 1.000 while the answer fell to 0.777: blending leaked
candidates' stamps corrupts the read). Under GENUINE ambiguity the right answer is a probability, so the question is which
READ to put behind the gate-then-compare router.
TASK (the (who, when) task with KNOWN ambiguity): events = actor + action (+ a DECOY flag). Surface-A events at t >= k:
s_clear in {0, 1} clear sales (always real) and s_dec in {1, 2} DECOY-flagged sales, each REAL with known probability q
(hidden; resampled until at least one real). Query: 'who acted k before the first/last REAL A?'. The TRUE posterior over
the anchor is the product with g = 1 (clear), q (decoy), 0 (other): p(anchor = t) ~ g_t prod_{u keyed later} (1 - g_u)
(mirrored for first); the true answer distribution = sum_t p(t) delta(actor at t - k). So the BAYES accuracy and the
BAYES log loss are computed exactly per sequence, before any training. q in {0.3, 0.7}.
ROUTER (shared): the race-3 gate-then-compare product (content gate g = sigmoid(kappa (c - theta)); order key = the 1296
wheel's angle on [0, 2 pi), which ties the index within one turn); registers W 16 stamped by a fixed 256-plane clock.
READS (the only difference between arms):
  blend   mix the candidates' target stamps (and register weights), then unbind once (the read so far)
  snap    read the top candidate only (straight-through: forward = argmax, backward = as blended logits)
  marg    read EVERY candidate, mix the ANSWERS: p(y) = sum_t alpha_t softmax(read_t)
  beam2 / beam4   marginalize over the top K candidates (renormalized): the beam
Loss: the arm's own negative log likelihood of the true answer (marg/beam: the mixture likelihood). lr fixed 3e-3 (the
race 3/4 sweeps chose 1e-3 and 3e-3 within 0.001 for the two-stage router). Train N 12-64; eval N 16/32/64/128/256.
REPORTED per arm, q, N: accuracy, log loss (nats), router hit, the gate LEARNED on decoy events (vs the true q).
Usage: .venv/bin/python3 scripts/toys/read_race.py [--smoke] -> .cache/read_race{tag}.*
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mode = "smoke" if "--smoke" in sys.argv else "full"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

NA, NACT, KMAX, MP, W = 40, 8, 8, 256, 16
V_ACT, V_QA, V_WH, V_K, DEC, PADT, VOCAB = 40, 48, 56, 58, 66, 67, 68
TRAIN_N, EVAL_N = (12, 64), (16, 32, 64, 128, 256)
STEPS = int(os.environ.get("RD_STEPS", "300" if mode == "smoke" else "4000"))
BATCH, N_EVAL = 64, (200 if mode == "smoke" else 2000)
SEEDS = [0] if mode == "smoke" else [0, 1, 2]
ARMS = os.environ.get("RD_ARMS", "blend,snap,marg,beam2,beam4").split(",")
QS = [float(x) for x in os.environ.get("RD_QS", "0.3,0.7").split(",")]
LR = 3e-3


def true_post(g, which):                                                     # the product posterior over the anchor
    l1 = np.log1p(-np.minimum(g, 1 - 1e-9)); c = np.cumsum(l1[::-1])[::-1]
    later = np.append(c[1:], 0.0); earlier = np.cumsum(l1) - l1
    a = g * np.exp(later if which == 1 else earlier)
    return a / a.sum()


def batch(B, rng, q, n=None):
    ns = rng.integers(TRAIN_N[0], TRAIN_N[1] + 1, size=B) if n is None else np.full(B, n)
    L = int(ns.max()) + 1
    x1, x2, x3 = np.full((B, L), PADT), np.full((B, L), -1), np.full((B, L), -1)
    actor = np.zeros((B, L - 1), np.int64); kq = np.zeros(B, np.int64); y = np.zeros(B, np.int64); anc = np.zeros(B, np.int64)
    dec = np.zeros((B, L - 1), bool); blp = np.zeros(B); bacc = np.zeros(B)
    for b in range(B):
        m = ns[b]; A = rng.integers(NACT); k = rng.integers(1, KMAX + 1); which = rng.integers(2)
        acts = rng.integers(NACT - 1, size=m); acts[acts >= A] += 1
        sc, sd = rng.integers(2), 1 + rng.integers(2)
        pos = rng.choice(np.arange(k, m), size=sc + sd, replace=False); cl, dc = pos[:sc], pos[sc:]
        while True:
            real_d = dc[rng.random(sd) < q]
            real = np.concatenate([cl, real_d])
            if len(real): break
        acts[pos] = A
        a = rng.integers(NA, size=m)
        x1[b, :m] = a; x2[b, :m] = V_ACT + acts; x3[b, dc] = DEC; dec[b, dc] = True
        x1[b, m] = V_QA + A; x2[b, m] = V_WH + which; x3[b, m] = V_K + k - 1
        t_anc = real.min() if which == 0 else real.max()
        actor[b, :m] = a; anc[b] = t_anc; kq[b] = k; y[b] = a[t_anc - k]
        g = np.zeros(m); g[cl] = 1.0; g[dc] = q
        p = true_post(g, which); pa = np.zeros(NA)
        for t in np.nonzero(p > 0)[0]: pa[a[t - k]] += p[t]
        blp[b] = -math.log(max(pa[y[b]], 1e-12)); bacc[b] = float(pa.argmax() == y[b])
    f = lambda z: torch.from_numpy(np.asarray(z))
    return dict(x1=f(x1).long(), x2=f(x2).long(), x3=f(x3).long(), ns=f(ns).long(), actor=f(actor).long(), k=f(kq).long(),
                y=f(y).long(), anc=f(anc).long(), dec=f(dec), blp=f(blp).float(), bacc=f(bacc).float())


class Reader(nn.Module):
    def __init__(self, kind, seed):
        super().__init__(); self.kind = kind; g = torch.Generator().manual_seed(6000 + seed)
        self.emb = nn.Embedding(VOCAB, 64); self.q, self.kk = nn.Linear(64, 64), nn.Linear(64, 64)
        self.kappa_log, self.theta = nn.Parameter(torch.tensor(math.log(4.0))), nn.Parameter(torch.tensor(0.0))
        self.register_buffer("om", torch.rand(MP, generator=g) * 2 * math.pi)
        te = torch.rand(NA, MP, generator=g) * 2 * math.pi
        self.ent = nn.Parameter(torch.cat([torch.cos(te), torch.sin(te)], -1))
        self.log_beta = nn.Parameter(torch.tensor(math.log(float(os.environ.get("RD_BETA0", "30")))))   # a register superposes 16 events: the right cos ~0.25, so a confident read needs beta ~40; init 2 left every arm read-capped (feas: marg ll 1.21 vs Bayes 0.32)

    def embed(self, bt):
        h = self.emb(bt["x1"])
        for c in ("x2", "x3"):
            z = bt[c]; h = h + self.emb(z.clamp(min=0)) * (z >= 0)[..., None]
        return h

    def forward(self, bt):                                                   # -> log p(answer) (B, NA)
        x = bt["x1"]; B, L1 = x.shape; L = L1 - 1; ns = bt["ns"]; ar = torch.arange(B)
        live = (torch.arange(L)[None] < ns[:, None]).float(); tt = torch.arange(L).float()
        h = self.embed(bt)
        c = (self.kk(h[:, :L]) @ self.q(h[ar, ns])[..., None]).squeeze(-1) / 8.0
        gte = torch.sigmoid(self.kappa_log.exp() * (c - self.theta)) * live
        self.gate_dec = gte[bt["dec"][:, :L]].detach()
        which = bt["x2"][ar, ns] - V_WH
        key = torch.remainder(2 * math.pi * tt / 1296.0, 2 * math.pi)
        later = (key[None, :] > key[:, None]).float()
        l1g = torch.log((1 - gte).clamp(min=1e-6))
        veto = torch.where(which[:, None] == 1, l1g @ later.T, l1g @ later)
        a = torch.exp(torch.log(gte.clamp(min=1e-6)) + veto) * live
        alpha = a / (a.sum(-1, keepdim=True) + 1e-6)
        self.router_hit = (alpha.argmax(-1) == bt["anc"]).float()
        # registers: actors stamped by the fast clock, W 16 by absolute tick
        e = self.ent[bt["actor"]]; er, ei = e[..., :MP], e[..., MP:]
        wa = tt[:, None] * self.om; cw, sw = torch.cos(wa), torch.sin(wa)
        pr, pi = (er * cw - ei * sw) * live[..., None], (er * sw + ei * cw) * live[..., None]
        R = (L + W - 1) // W; reg = F.one_hot(torch.arange(L) // W, R).float()
        hr, hi = torch.einsum("blm,lr->brm", pr, reg), torch.einsum("blm,lr->brm", pi, reg)
        tau = (tt[None] - bt["k"][:, None].float()).clamp(min=0)              # each candidate's target time (B, L)
        ta = tau[..., None] * self.om; tc, ts = torch.cos(ta), torch.sin(ta)  # target stamps (B, L, MP)
        ent_n = F.normalize(self.ent, dim=-1); beta = self.log_beta.exp()
        if self.kind == "blend":                                             # mix the stamps, then unbind once
            sc, ss = (alpha[..., None] * tc).sum(1), (alpha[..., None] * ts).sum(1)
            wr = torch.einsum("bl,blr->br", alpha, F.one_hot((tau.long() // W).clamp(max=R - 1), R).float())
            ur0, ui0 = (wr[..., None] * hr).sum(1), (wr[..., None] * hi).sum(1)
            u = torch.cat([ur0 * sc + ui0 * ss, ui0 * sc - ur0 * ss], -1)
            return F.log_softmax(beta * F.normalize(u, dim=-1) @ ent_n.T, -1)
        ri = (tau.long() // W).clamp(max=R - 1)                               # every candidate's own read
        gr, gi = hr[ar[:, None], ri], hi[ar[:, None], ri]                     # (B, L, MP)
        u = torch.cat([gr * tc + gi * ts, gi * tc - gr * ts], -1)
        lg = beta * F.normalize(u, dim=-1) @ ent_n.T                          # (B, L, NA)
        if self.kind == "snap":                                              # straight-through argmax
            hard = F.one_hot(alpha.argmax(-1), L).float(); ast = hard + alpha - alpha.detach()
            return F.log_softmax((ast[..., None] * lg).sum(1), -1)
        if self.kind.startswith("beam"):
            K = int(self.kind[4:]); top = alpha.topk(K, -1)
            alpha = torch.zeros_like(alpha).scatter(-1, top.indices, top.values)
            alpha = alpha / (alpha.sum(-1, keepdim=True) + 1e-9)
        return torch.logsumexp(torch.log(alpha.clamp(min=1e-12))[..., None] + F.log_softmax(lg, -1), 1)


def run(kind, seed, q):
    rng = np.random.default_rng(seed * 13 + int(q * 100)); erng = np.random.default_rng(40_000 + seed + int(q * 100))
    ev = {n: batch(N_EVAL, erng, q, n) for n in EVAL_N}
    torch.manual_seed(seed); net = Reader(kind, seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = batch(BATCH, rng, q); loss = F.nll_loss(net(bt), bt["y"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); res = {}
    with torch.no_grad():
        for n, full in ev.items():
            lp, hit, gd = [], [], []
            for i in range(0, N_EVAL, 250):
                bt = {k_: v_[i:i + 250] for k_, v_ in full.items()}; lp.append(net(bt)); hit.append(net.router_hit); gd.append(net.gate_dec)
            lp = torch.cat(lp); y = full["y"]
            res[f"acc{n}"] = float((lp.argmax(-1) == y).float().mean())
            res[f"ll{n}"] = float(-lp[torch.arange(len(y)), y].mean())
            res[f"bacc{n}"], res[f"bll{n}"] = float(full["bacc"].mean()), float(full["blp"].mean())
            res[f"route{n}"] = float(torch.cat(hit).mean()); res[f"gate{n}"] = float(torch.cat(gd).mean())
    ins = [n for n in EVAL_N if n <= TRAIN_N[1]]
    for m in ("acc", "ll", "bacc", "bll", "gate"):
        res[f"{m}_in"] = float(np.mean([res[f"{m}{n}"] for n in ins]))
    res["final_loss"] = float(loss.detach())
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"smoke": "_smoke", "full": ""}[mode] + os.environ.get("RD_TAG", "")
    assert mode == "smoke" or not os.path.exists(f".cache/read_race{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/read_race{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS} lr {LR}")
    R = {}
    for q in QS:
        for kind in ARMS:
            rs = [run(kind, s, q) for s in SEEDS]
            A = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}; R[(q, kind)] = A
            res.append(dict(q=q, arm=kind, seeds=rs, agg=A))
            say(f"q {q} {kind:<6} in-range acc {A['acc_in']:.3f} ll {A['ll_in']:.3f} (Bayes {A['bacc_in']:.3f} / {A['bll_in']:.3f}) | "
                + "  ".join(f"N{n} {A[f'acc{n}']:.3f}/{A[f'll{n}']:.3f}" for n in (128, 256))
                + f" | router {A['route64']:.3f}/{A['route256']:.3f} | decoy gate {A['gate_in']:.3f} (q {q})  [{time.time() - t0:.0f}s]")
    json.dump(res, open(f".cache/read_race{tag}.json", "w"), indent=1)
    if mode == "full":                                                       # THE PINNED BARS (ledger 2026-10-21)
        say("VERDICTS (3-seed means; in-range = N 16/32/64; log loss in nats)")
        for q in QS:
            g = lambda a, k: R[(q, a)][k]
            say(f"(M1) q {q}: " + ("MARGINALIZE IS CALIBRATED" if g("marg", "ll_in") <= g("marg", "bll_in") + 0.05 else "MARGINALIZE IS NOT CALIBRATED")
                + f"  (marg {g('marg', 'll_in'):.3f} vs Bayes {g('marg', 'bll_in'):.3f})")
            say(f"(M2) q {q}: " + ("SNAP IS OVERCONFIDENT" if g("snap", "ll_in") >= g("marg", "ll_in") + 0.10 else "SNAP IS NOT OVERCONFIDENT")
                + f"  (snap ll {g('snap', 'll_in'):.3f} vs marg {g('marg', 'll_in'):.3f}; acc snap {g('snap', 'acc_in'):.3f} vs marg {g('marg', 'acc_in'):.3f})")
            say(f"(M3) q {q}: " + ("BLEND LOSES" if g("blend", "acc256") <= g("marg", "acc256") - 0.03 and g("blend", "ll256") >= g("marg", "ll256") + 0.05 else "BLEND DOES NOT LOSE")
                + f"  (N256 acc {g('blend', 'acc256'):.3f} vs {g('marg', 'acc256'):.3f}, ll {g('blend', 'll256'):.3f} vs {g('marg', 'll256'):.3f})")
            b2 = all(abs(g("beam2", f"acc{n}") - g("marg", f"acc{n}")) <= 0.02 and g("beam2", f"ll{n}") <= 1.10 * g("marg", f"ll{n}") for n in EVAL_N)
            say(f"(M4) q {q}: " + ("TOP-2 SUFFICES" if b2 else "TOP-2 DOES NOT SUFFICE") + "  (" + ", ".join(
                f"N{n} {g('beam2', f'acc{n}'):.3f}/{g('beam2', f'll{n}'):.3f} vs {g('marg', f'acc{n}'):.3f}/{g('marg', f'll{n}'):.3f}" for n in EVAL_N) + ")")
            say(f"(M5) q {q}: " + ("MARGINALIZING TEACHES CALIBRATION" if abs(g("marg", "gate_in") - q) <= 0.10 else "MARGINALIZING DOES NOT CALIBRATE THE GATE")
                + "  (decoy gate: " + ", ".join(f"{a} {g(a, 'gate_in'):.3f}" for a in ARMS) + f"; true {q})")
    say(f"DONE in {time.time() - t0:.0f}s")
