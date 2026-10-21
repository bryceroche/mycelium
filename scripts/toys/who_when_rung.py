"""who_when_rung.py -- THE (WHO, WHEN) RUNG: THE CLOCK MUST FIND ITS OWN ADDRESS
(2026-10-21; Bryce's word; Opus's design adds: repeated content, oracle-tick ceiling, scrambled-stamp placebo). CPU only.

WHY: the time-vs-content rung's register clock read 0.999 at N 128, but every memory arm was HANDED the target's time.
Here nothing gives a time: the anchor is found by CONTENT, the step back is a clock rotation, the answer is unbound.
TASK (fresh sequences every batch): N events, each a FUSED token (actor in 40) + (action in 8). A query token =
(anchor action A, which in {first, last}, k in 1..8): 'who acted k events before the first/last A?'. A occurs ONCE or
TWICE (p 1/2 each; repeated content, so the content match alone is ambiguous); the other events draw from the other 7
actions; the chosen occurrence sits at t >= k. Answer: the actor at t_anchor - k. Train N ~ U[12, 64]; eval N 16 / 32
/ 64 / 128 (128 = length extrapolation; the in-range mean excludes it).
ARMS (every arm sees the same tokens; k is part of the QUESTION, given to all; no arm is given the anchor's time):
  ATT      2-layer RoPE transformer (D 128, 4 heads), readout at the query token (time_content_rung's Block)
  ATT_m8   the same with a small cache (one head, k and v of 8 dims per layer: the time rung's register budget, 4,096)
  ATT_m24  k and v of 24 dims: EQUAL MEMORY with the hybrid (12,288 floats at N 128 = registers 4,096 + the router's
           64-d key per event 8,192: a content search needs a per-event key; registers compress only the VALUES)
  HYB      THE HYBRID: a 1-layer RoPE attention router (one head, 64 d, learned sharpness) finds the anchor tick by
           content -> alpha_t; the anchor's clock stamp = sum_t alpha_t c(t) (fast: 256 planes uniform; slow: 16 planes,
           learned, periods 256..2048); rotate both back by k; the slow stamp picks the register (W 16, softmax over the
           registers' mean slow stamps); the fast stamp UNBINDS the actor from it (registers hold sum_t c(t) (*) actor_t);
           snap over the 40 actors (cosine, learned beta). Actor codes learned unit-init phasors.
  HYB_1vec the hybrid with ONE register (all events in one vector): the capacity comparator under learned routing
  CACHE    THE FAIR RIVAL (feas: every plain attention arm, 2 or 4 layers, sat at the floor ~0.07-0.10, so the rival
           gets the hybrid's INTERFACE, the depth run's lesson): the same router and the same clock as the address, but
           each event's actor stored in its OWN 32-d slot (no superposition), read by a sharp positional attention at the
           target stamp (cosine over the 256 fast planes, learned sharpness). Memory = the router's keys 8,192 + slots
           4,096 = 12,288 = the hybrid's. The clock addresses both; they differ only in STORAGE (superposed vs slots).
  ORACLE   the hybrid handed the TRUE anchor tick (alpha one-hot): the routing ceiling (a ceiling, not a win)
  PLACEBO  the hybrid with SCRAMBLED write stamps (each sequence's events stamped by a random permutation of the ticks):
           if the clock carries the read, this collapses
Usage: .venv/bin/python3 scripts/toys/who_when_rung.py [--feas | --smoke] -> .cache/who_when_rung{tag}.*
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mode = "feas" if "--feas" in sys.argv else "smoke" if "--smoke" in sys.argv else "full"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import time_content_rung as T  # noqa: E402  (Block, rope)

NA, NACT, KMAX, D, MP, MS, W = 40, 8, 8, 128, 256, 16, 16
NL = int(os.environ.get("WW_ATT_NL", "2"))                                 # the attention rivals' depth
TRAIN_N, EVAL_N = (12, 64), (16, 32, 64, 128)
STEPS = 300 if mode == "smoke" else int(os.environ.get("WW_STEPS", "4000"))
BATCH, N_EVAL = 64, (300 if mode == "smoke" else 1000)
SEEDS = [0] if mode != "full" else [0, 1, 2]
ALL = ["ATT", "ATT_m24", "HYB", "HYB_1vec", "CACHE", "ORACLE", "PLACEBO"]
ARMS = os.environ.get("WW_ARMS", ",".join(ALL)).split(",")
# vocab (summed channels): actors 0..39 | actions 40..47 | query-action 48..55 | which 56..57 | k 58..65 | PAD 66
V_ACT, V_QA, V_WH, V_K, PADT, VOCAB = 40, 48, 56, 58, 66, 67
T.D = D


def batch(B, rng, n=None):
    ns = rng.integers(TRAIN_N[0], TRAIN_N[1] + 1, size=B) if n is None else np.full(B, n)
    L = int(ns.max()) + 1
    x1, x2, x3 = np.full((B, L), PADT), np.full((B, L), -1), np.full((B, L), -1)
    actor = np.zeros((B, L - 1), np.int64); anc = np.zeros(B, np.int64); kq = np.zeros(B, np.int64); y = np.zeros(B, np.int64)
    for b in range(B):
        m = ns[b]; A = rng.integers(NACT); k = rng.integers(1, KMAX + 1); which = rng.integers(2)
        acts = rng.integers(NACT - 1, size=m); acts[acts >= A] += 1          # no A by default
        c = 1 + rng.integers(2)
        pos = rng.choice(np.arange(k, m), size=1) if c == 1 else None
        if c == 2:                                                           # the chosen occurrence at >= k; the other anywhere
            p = rng.integers(k, m); q = rng.choice(np.setdiff1d(np.arange(m), [p]))
            pos = np.sort([p, q])
            if (which == 0 and p > q) or (which == 1 and p < q):            # make p the CHOSEN one under 'which'
                which = 1 - which
        acts[pos] = A
        t_anc = pos[0] if which == 0 else pos[-1]
        a = rng.integers(NA, size=m)
        x1[b, :m] = a; x2[b, :m] = V_ACT + acts
        x1[b, m] = V_QA + A; x2[b, m] = V_WH + which; x3[b, m] = V_K + k - 1
        actor[b, :m] = a; anc[b] = t_anc; kq[b] = k; y[b] = a[t_anc - k]
    f = lambda z: torch.from_numpy(np.asarray(z, np.int64))
    return dict(x1=f(x1), x2=f(x2), x3=f(x3), ns=f(ns), actor=f(actor), anc=f(anc), k=f(kq), y=f(y))


def embed(emb, bt):
    h = emb(bt["x1"])
    for c in ("x2", "x3"):
        z = bt[c]; h = h + emb(z.clamp(min=0)) * (z >= 0)[..., None]
    return h


class MixNet(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.emb = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList([T.Block(kind, 0) for _ in range(NL)]); self.lnf, self.out = nn.LayerNorm(D), nn.Linear(D, NA)

    def forward(self, bt):
        x = bt["x1"]; pad = x == PADT; h = embed(self.emb, bt)
        for b in self.blocks: h = b(h, pad)
        return self.out(self.lnf(h[torch.arange(len(x)), bt["ns"]]))


class Hybrid(nn.Module):
    def __init__(self, kind, seed):
        super().__init__(); self.kind = kind; g = torch.Generator().manual_seed(5000 + seed)
        self.emb = nn.Embedding(VOCAB, 64); self.q, self.kk = nn.Linear(64, 64), nn.Linear(64, 64)
        self.log_beta_a = nn.Parameter(torch.tensor(math.log(4.0)))
        self.register_buffer("om", torch.rand(MP, generator=g) * 2 * math.pi)
        self.oms = nn.Parameter(2 * math.pi / torch.exp(torch.linspace(math.log(256.0), math.log(2048.0), MS)))
        self.log_beta_r = nn.Parameter(torch.tensor(math.log(5.0)))
        te = torch.rand(NA, MP, generator=g) * 2 * math.pi
        self.ent = nn.Parameter(torch.cat([torch.cos(te), torch.sin(te)], -1))
        self.log_beta = nn.Parameter(torch.tensor(math.log(2.0)))
        if kind == "CACHE":
            self.val, self.vout = nn.Embedding(NA, 32), nn.Linear(32, NA)
            self.log_beta_p = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, bt):
        x = bt["x1"]; B, L1 = x.shape; L = L1 - 1; ns = bt["ns"]; ar = torch.arange(B)
        live = (torch.arange(L)[None] < ns[:, None]).float()
        tt = torch.arange(L).float()
        # ROUTER: the query token attends over the events by content (+ RoPE relative position for first/last)
        h = embed(self.emb, bt)
        qv = self.q(h[ar, ns])[:, None, None, :]                             # (B, 1, 1, 64)
        kv = self.kk(h)[:, None]                                             # (B, 1, L1, 64)
        # rotate keys by their position relative to the query (RoPE): the query sits at ns
        rel = (tt[None] - ns[:, None].float())                               # (B, L) <= -1 for events
        fq = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64)); ang = rel[..., None] * fq
        k1, k2 = kv[..., :L, 0::2].squeeze(1), kv[..., :L, 1::2].squeeze(1)
        kr = torch.stack([k1 * torch.cos(ang) - k2 * torch.sin(ang), k1 * torch.sin(ang) + k2 * torch.cos(ang)], -1).flatten(-2)
        s = (kr @ qv.squeeze(1).squeeze(1)[..., None]).squeeze(-1) / 8.0     # (B, L)
        s = (self.log_beta_a.exp() * s).masked_fill(live == 0, -1e4)
        alpha = F.one_hot(bt["anc"], L).float() if self.kind == "ORACLE" else F.softmax(s, -1)
        self.router_hit = (alpha.argmax(-1) == bt["anc"]).float()
        # WRITE: actors stamped by the fast clock (absolute tick), in registers by absolute tick
        ws = tt[None].expand(B, L)
        if self.kind == "PLACEBO":
            ws = torch.stack([torch.randperm(L).float() for _ in range(B)])  # scrambled stamps
        e = self.ent[bt["actor"]]; er, ei = e[..., :MP], e[..., MP:]
        wa = ws[..., None] * self.om; c, sn = torch.cos(wa) * live[..., None], torch.sin(wa) * live[..., None]
        pr, pi = er * c - ei * sn, er * sn + ei * c                          # (B, L, MP)
        R = 1 if self.kind == "HYB_1vec" else (L + W - 1) // W
        reg = F.one_hot((torch.arange(L) // W).clamp(max=R - 1), R).float()  # (L, R)
        hr, hi = torch.einsum("blm,lr->brm", pr, reg), torch.einsum("blm,lr->brm", pi, reg)
        # READ: the anchor's stamps (soft), rotated back by k
        kf = bt["k"].float()[:, None]
        fa = tt[None, :, None] * self.om                                     # fast stamps of every tick
        ar_, ai_ = (alpha[..., None] * torch.cos(fa)).sum(1), (alpha[..., None] * torch.sin(fa)).sum(1)
        rb = -kf * self.om; cr, ci = torch.cos(rb), torch.sin(rb)
        tr, ti = ar_ * cr - ai_ * ci, ar_ * ci + ai_ * cr                    # the target's fast stamp
        if self.kind == "CACHE":                                             # slots: a positional read at the target stamp
            sc = (torch.cos(fa)[0] @ tr[..., None] + torch.sin(fa)[0] @ ti[..., None]).squeeze(-1) / MP   # (B, L)
            w = F.softmax((self.log_beta_p.exp() * sc).masked_fill(live == 0, -1e4), -1)
            return self.vout((w[..., None] * self.val(bt["actor"])).sum(1))
        if R > 1:
            sa = tt[:, None] * self.oms[None]                                # (L, MS)
            kc = torch.einsum("bl,lr,ls->brs", live, reg, torch.cos(sa)); ks_ = torch.einsum("bl,lr,ls->brs", live, reg, torch.sin(sa))
            key = F.normalize(torch.cat([kc, ks_], -1), dim=-1)
            qa_c, qa_s = alpha @ torch.cos(sa), alpha @ torch.sin(sa)        # the anchor's slow stamp
            sr = -kf * self.oms; qc = qa_c * torch.cos(sr) - qa_s * torch.sin(sr); qs = qa_c * torch.sin(sr) + qa_s * torch.cos(sr)
            q = F.normalize(torch.cat([qc, qs], -1), dim=-1)
            filled = torch.einsum("bl,lr->br", live, reg) > 0
            a = F.softmax((self.log_beta_r.exp() * (key @ q[..., None]).squeeze(-1)).masked_fill(~filled, -1e4), -1)
            hr, hi = (a[..., None] * hr).sum(1), (a[..., None] * hi).sum(1)
        else:
            hr, hi = hr[:, 0], hi[:, 0]
        ur, ui = hr * tr + hi * ti, hi * tr - hr * ti                        # unbind: h (*) conj(target stamp)
        u = torch.cat([ur, ui], -1)
        return self.log_beta.exp() * F.normalize(u, dim=-1) @ F.normalize(self.ent, dim=-1).T


def build(arm, seed):
    torch.manual_seed(seed)
    if arm.startswith("ATT"): return MixNet(arm)
    return Hybrid(arm, seed)


def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def memfloats(arm, n):
    if arm == "ATT": return NL * 2 * D * n
    if arm == "ATT_m8": return NL * 2 * 8 * n
    if arm == "ATT_m24": return NL * 2 * 24 * n
    if arm == "HYB_1vec": return 2 * MP + 64 * n
    if arm == "CACHE": return 32 * n + 64 * n                             # the vector + the router's keys
    return -(-n // W) * 2 * MP + 64 * n


def run(arm, seed, lr):
    rng = np.random.default_rng(seed * 11 + 3); erng = np.random.default_rng(20_000 + seed)
    ev = {n: batch(N_EVAL, erng, n) for n in EVAL_N}
    net = build(arm, seed)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = batch(BATCH, rng); loss = F.cross_entropy(net(bt), bt["y"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); res = {}
    with torch.no_grad():
        for n, bt in ev.items():
            res[f"N{n}"] = float((net(bt).argmax(-1) == bt["y"]).float().mean())
            if hasattr(net, "router_hit"): res[f"route{n}"] = float(net.router_hit.mean())
    res["mean"] = float(np.mean([res[f"N{n}"] for n in EVAL_N if n <= TRAIN_N[1]]))
    res["params"] = nparams(net); res["final_loss"] = float(loss.detach()); res["mem_floats_N128"] = memfloats(arm, 128)
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"feas": "_feas", "smoke": "_smoke", "full": ""}[mode] + os.environ.get("WW_TAG", "")
    assert mode == "smoke" or not os.path.exists(f".cache/who_when_rung{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/who_when_rung{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS}; params: " + ", ".join(f"{a} {nparams(build(a, 0))}" for a in ARMS))
    for arm in ARMS:
        if mode == "full":
            sc = {lr: run(arm, 0, lr)["mean"] for lr in (1e-3, 3e-3)}; lr = max(sc, key=sc.get)
            say(f"lr sweep {arm}: {sc} -> {lr}  [{time.time() - t0:.0f}s]")
        else:
            lr = float(os.environ.get("WW_LR", "3e-3"))
        rs = [run(arm, s, lr) for s in SEEDS]
        A = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        res.append(dict(arm=arm, lr=lr, seeds=rs, agg=A))
        say(f"{arm:<9} params {rs[0]['params']:>7} loss {A['final_loss']:.3f}  "
            + "  ".join(f"{k} {v:.3f}" for k, v in A.items() if k.startswith(("N", "route")) or k == "mean")
            + f"  mem {A['mem_floats_N128']:.0f}  [{time.time() - t0:.0f}s]")
    json.dump(res, open(f".cache/who_when_rung{tag}.json", "w"), indent=1)
    R = {r["arm"]: r["agg"] for r in res}
    if mode == "feas":
        ok = max(a["mean"] for k_, a in R.items() if k_ not in ("ORACLE", "PLACEBO")) >= 0.80   # a ceiling cannot pass it
        say(f"FEASIBILITY: some arm >= 0.80 in range: {ok} -> {'FEASIBLE (pin the bars)' if ok else 'NOT FEASIBLE'}")
    if mode == "full":                                                       # THE PINNED BARS (ledger 2026-10-21 ~06:35)
        g = lambda a, k: R[a][k]
        v = lambda c, y, n: y if c else n
        say("VERDICTS (3-seed means)")
        say("(a) " + v(g("HYB", "N128") >= 0.90, "THE HYBRID HOLDS AT 128", "THE HYBRID DOES NOT HOLD AT 128") + f"  ({g('HYB', 'N128'):.3f})")
        say("(b) " + v(g("ORACLE", "N128") - g("HYB", "N128") <= 0.05, "ROUTING IS CHEAP", "ROUTING COSTS") + f"  (oracle - hybrid at 128 {g('ORACLE', 'N128') - g('HYB', 'N128'):+.3f})")
        dN, dm = g("HYB", "N128") - g("CACHE", "N128"), g("HYB", "mean") - g("CACHE", "mean")
        say("(c) " + ("SUPERPOSITION TIES SLOTS AT EQUAL MEMORY" if abs(dN) <= 0.03 and abs(dm) <= 0.03 else
                      "SUPERPOSITION BEATS SLOTS" if dN >= 0.05 else
                      "SLOTS BEAT SUPERPOSITION" if dN <= -0.05 or dm <= -0.05 else "NEAR (within 0.05)")
            + f"  (hybrid - cache: 128 {dN:+.3f}, in range {dm:+.3f})")
        say("(d) " + v(g("PLACEBO", "mean") <= 0.30, "THE CLOCK CARRIES THE READ (placebo collapses)", "THE PLACEBO SURVIVES") + f"  ({g('PLACEBO', 'mean'):.3f})")
        say("(e) " + v(g("HYB", "N128") - g("HYB_1vec", "N128") >= 0.20, "REGISTERS NEEDED UNDER LEARNED ROUTING", "REGISTERS NOT NEEDED") + f"  ({g('HYB', 'N128') - g('HYB_1vec', 'N128'):+.3f})")
        say("(f) " + v(g("HYB", "route128") <= 0.90 and abs(g("HYB", "N128") - g("HYB", "route128")) <= 0.05, "THE ROUTER IS THE LENGTH WALL", "THE ROUTER IS NOT THE (WHOLE) WALL")
            + f"  (route128 {g('HYB', 'route128'):.3f}, N128 {g('HYB', 'N128'):.3f})")
        say("CONTEXT " + v(max(g("ATT", "mean"), g("ATT_m24", "mean")) <= 0.30, "PLAIN ATTENTION DOES NOT LEARN THE COMPOSITION", "PLAIN ATTENTION LEARNS IT")
            + f"  (ATT {g('ATT', 'mean'):.3f}, ATT_m24 {g('ATT_m24', 'mean'):.3f})")
    say(f"DONE in {time.time() - t0:.0f}s")
