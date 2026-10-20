"""complex_unbind_machine.py -- RUNG 1T, FORM 2: THE UNBINDING CHAIN AS THE ONLY ROAD (2026-10-19; Bryce's word on rung
1T; Opus's three build notes). CPU only. The task is complex_unbind_toy's (imported: same scenes, held-out table, seeds).

WHY FORM 2: form 1 wired the unbinding head into the residual stream as an ADDITION; the oracle-write arm then memorized
(value (*) role) pairs through the ordinary layers (in-dist hop 1 0.87, held-out 0.02) -- the mandatory-road law. Here the
machine has NO bypass: a pointer p_0 = e(X); per hop h < d:
    a = softmax(tau * Re<W_q p, W_k m_j> / sqrt)   over the facts' memory m_j = e(owner) + bind(e(filler), role(alias))
    u = unbind(sum_j a_j m_j, role(alias_h))
    s_c = beta * cos(u, e_c) over the scene's entities;  p_{h+1} = sum_c softmax(s)_c e_c    (the soft snap)
and the answer is hop d's snap logits. The analytic version (exact owner match, exact unbind, hard snap) reads held-out
0.984 / 0.981 / 0.966 at 96 planes (complex_unbind_ceiling.py, CEIL_M=96).
OPUS'S NOTES: (1) the snap starts SOFT (beta init 2, learned); (2) the owner attention gets DIRECT SUPERVISION (the
attention-bootstrap law): ATT_W * -log(sum of a over the current node's facts), in EVERY machine arm alike; (3) the arms
are read in pairs.
ARMS (W_q / W_k initialized to the identity in every machine):
  PH_oracle  complex, 96 planes: entity codes FIXED random unit phasors, alias -> its role's FIXED phasor, exact binding
  PH_learn   complex, 96 planes: entity codes LEARNED, alias -> code choice LEARNED over 8 FIXED phasor role codes
  MX_learn   real, 192 dims: entity codes LEARNED, role = LEARNED mix of 8 LEARNED 192x192 role matrices; bind f @ R,
             unbind r @ R^T (tensor-product style; the same machine, the same supervision)
  T4_width   plain transformer, 4 layers (hops + 1: Opus's depth caveat), D = 192, pointer readout over the scene
  T4_params  the same at the width closest to PH_learn's parameter count
PAIRS: PH_oracle vs PH_learn = can writing be learned? | PH_learn vs MX_learn = do PHASES beat learned binding? (the
phase claim) | best machine vs best T4 = does explicit structure beat none? (context: the machines are handed structure)
Usage: .venv/bin/python3 scripts/toys/complex_unbind_machine.py [--feas | --smoke] -> .cache/complex_unbind_machine{tag}.*
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_mode = "feas" if "--feas" in sys.argv else "smoke" if "--smoke" in sys.argv else "full"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import complex_unbind_toy as T  # noqa: E402

mode = _mode
MP, DR = 96, 192                                                             # complex planes / real dims
ATT_W = float(os.environ.get("UM_ATT_W", "0.3"))
STEPS = 300 if mode == "smoke" else int(os.environ.get("UM_STEPS", "4000"))
SEEDS = [0] if mode != "full" else [0, 1, 2]
ARMS = os.environ.get("UM_ARMS", "PH_oracle,PH_learn,MX_learn,T4_width,T4_params").split(",")
T.N_LAYERS = 4                                                               # the plain arms' depth (hops + 1)


def gold_nodes(facts, query):
    """The chain's owner node at each hop (B, 3), from the facts themselves."""
    cur = query[:, 0].clone(); nodes = []
    for h in range(3):
        nodes.append(cur.clone())
        hit = (facts[..., 0] == cur[:, None]) & (facts[..., 1] // T.N_ALIAS_PER == (query[:, 1 + h] // T.N_ALIAS_PER)[:, None])
        nxt = (hit.long() * facts[..., 2]).sum(1)
        cur = torch.where(hit.any(1) & (query[:, 4] > h), nxt, cur)
    return torch.stack(nodes, 1)


class Machine(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        g = torch.Generator().manual_seed(2000 + seed); self.kind = kind
        self.cx = kind.startswith("PH")
        n = 2 * MP if self.cx else DR; self.n = n
        if self.cx:
            te = torch.rand(T.N_ENT, MP, generator=g) * 2 * math.pi
            self.ent = nn.Parameter(torch.cat([torch.cos(te), torch.sin(te)], -1), requires_grad=(kind == "PH_learn"))
            th = torch.rand(T.N_CODES, MP, generator=g) * 2 * math.pi           # FIXED phasor role codes
            self.register_buffer("Pr", torch.cos(th)); self.register_buffer("Pi", torch.sin(th))
            self.wq = nn.ParameterList([nn.Parameter(torch.eye(MP)), nn.Parameter(torch.zeros(MP, MP))])
            self.wk = nn.ParameterList([nn.Parameter(torch.eye(MP)), nn.Parameter(torch.zeros(MP, MP))])
        else:
            self.ent = nn.Parameter(torch.randn(T.N_ENT, DR, generator=g))
            self.R = nn.Parameter(torch.linalg.qr(torch.randn(T.N_CODES, DR, DR, generator=g))[0])  # LEARNED role matrices
            self.wq, self.wk = nn.Parameter(torch.eye(DR)), nn.Parameter(torch.eye(DR))
        self.rho = nn.Embedding(T.N_ALIAS + 1, T.N_CODES)
        self.log_tau = nn.Parameter(torch.zeros(())); self.log_beta = nn.Parameter(torch.tensor(math.log(2.0)))

    # ---- the operators (complex = [re | im] halves)
    def cmul(self, a, b, conj=False):
        ar, ai, br, bi = a[..., :MP], a[..., MP:], b[..., :MP], b[..., MP:]
        if conj: bi = -bi
        return torch.cat([ar * br - ai * bi, ar * bi + ai * br], -1)

    def lin(self, w, x):
        if not self.cx: return x @ w.T
        xr, xi = x[..., :MP], x[..., MP:]
        return torch.cat([xr @ w[0].T - xi @ w[1].T, xr @ w[1].T + xi @ w[0].T], -1)

    def role_w(self, a):
        if self.kind == "PH_oracle":
            return F.one_hot((a // T.N_ALIAS_PER).clamp(max=T.N_CODES - 1), T.N_CODES).float()
        return F.softmax(self.rho(a), -1)

    def bind(self, f, a, inverse=False):
        w = self.role_w(a)
        if self.cx:
            pr, pi = w @ self.Pr, w @ self.Pi; nrm = torch.sqrt(pr * pr + pi * pi + 1e-6)
            return self.cmul(f, torch.cat([pr / nrm, pi / nrm], -1), conj=inverse)
        fR = torch.einsum("...j,kij->...ki", f, self.R) if inverse else torch.einsum("...i,kij->...kj", f, self.R)
        return torch.einsum("...k,...kd->...d", w, fR)                        # linear in w: mix after applying each R_k

    def forward(self, facts, query):
        E = self.ent
        mem = E[facts[..., 0]] + self.bind(E[facts[..., 2]], facts[..., 1])
        scene = F.one_hot(facts[..., 0], T.N_ENT).amax(1).bool()
        p = E[query[:, 0]]; atts, outs = [], []
        kx = self.lin(self.wk, mem)
        for h in range(3):
            s = torch.einsum("bd,btd->bt", self.lin(self.wq, p), kx) / math.sqrt(self.n) * self.log_tau.exp()
            a = F.softmax(s, -1); atts.append(a)
            u = self.bind(torch.einsum("bt,btd->bd", a, mem), query[:, 1 + h], inverse=True)
            cos = F.normalize(u, dim=-1) @ F.normalize(E, dim=-1).T
            lg = (self.log_beta.exp() * cos).masked_fill(~scene, -1e4); outs.append(lg)
            p = F.softmax(lg, -1) @ E
        lg = torch.stack(outs, 1)[torch.arange(len(query)), query[:, 4] - 1]
        return lg, torch.stack(atts, 1)


def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def t4_params_width():
    target = nparams(Machine("PH_learn", 0))
    return min(range(16, 260, 4), key=lambda D: abs(nparams(T.RNet(D, False, 0)) - target))


def build(arm, seed, Dp):
    if arm.startswith("T4"):
        return T.RNet(DR if arm == "T4_width" else Dp, False, seed)
    return Machine(arm, seed)


def run(arm, seed, lr, Dp):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); held = T.held_table(seed)
    erng = np.random.default_rng(10_000 + seed)
    ev = {"val": T.make_batch(T.N_EVAL, held, False, erng), "heldout": T.make_batch(T.N_EVAL, held, True, erng)}
    net = build(arm, seed, Dp); mach = isinstance(net, Machine)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        fa, q, t = T.make_batch(T.BATCH, held, False, rng)
        if mach:
            lg, att = net(fa, q); loss = F.cross_entropy(lg, t)
            nodes = gold_nodes(fa, q); own = fa[:, None, :, 0] == nodes[:, :, None]           # (B, 3, T)
            live = (q[:, 4, None] > torch.arange(3)[None]).float()
            la = -torch.log((att * own).sum(-1) + 1e-6)
            loss = loss + ATT_W * (la * live).sum() / live.sum()
        else:
            loss = F.cross_entropy(net(fa, q), t)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); res = {}
    with torch.no_grad():
        for split, (fa, q, t) in ev.items():
            out = net(fa, q); pr = (out[0] if mach else out).argmax(-1)
            res[split] = float((pr == t).float().mean())
            for d in (1, 2, 3):
                m = q[:, 4] == d; res[f"{split}_hop{d}"] = float((pr[m] == t[m]).float().mean())
            if mach:
                nodes = gold_nodes(fa, q); own = fa[:, None, :, 0] == nodes[:, :, None]
                live = q[:, 4, None] > torch.arange(3)[None]
                res[f"{split}_attmass"] = float((out[1] * own).sum(-1)[live].mean())
    res["params"] = nparams(net); res["final_loss"] = float(loss.detach())
    if mach: res["beta"] = float(net.log_beta.detach().exp())
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"feas": "_feas", "smoke": "_smoke", "full": ""}[mode] + os.environ.get("UM_TAG", "")
    out = open(f".cache/complex_unbind_machine{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    Dp = t4_params_width()
    say(f"mode {mode} steps {STEPS} ATT_W {ATT_W}; T4_params width {Dp}; trainable params: " +
        ", ".join(f"{a} {nparams(build(a, 0, Dp))}" for a in ARMS))
    lrs = {}
    for arm in ARMS:
        if mode == "full":
            sc = {lr: run(arm, 0, lr, Dp)["val"] for lr in (1e-3, 3e-3)}
            lrs[arm] = max(sc, key=sc.get); say(f"lr sweep {arm}: {sc} -> {lrs[arm]}  [{time.time() - t0:.0f}s]")
        else:
            lrs[arm] = 3e-3
    for arm in ARMS:
        rs = [run(arm, s, lrs[arm], Dp) for s in SEEDS]
        agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        sd = float(np.std([r["heldout"] for r in rs]))
        res.append(dict(arm=arm, lr=lrs[arm], seeds=rs, agg=agg, sd=sd))
        extra = (f"  attmass val {agg['val_attmass']:.3f} held {agg['heldout_attmass']:.3f}  beta {agg['beta']:.2f}"
                 if "beta" in agg else "")
        say(f"{arm:<10} params {rs[0]['params']:>7}  loss {agg['final_loss']:.3f}  val {agg['val']:.3f} "
            f"(h1 {agg['val_hop1']:.3f} h2 {agg['val_hop2']:.3f} h3 {agg['val_hop3']:.3f})  HELDOUT {agg['heldout']:.3f}±{sd:.3f} "
            f"(h1 {agg['heldout_hop1']:.3f} h2 {agg['heldout_hop2']:.3f} h3 {agg['heldout_hop3']:.3f}){extra}  [{time.time() - t0:.0f}s]")
    json.dump(res, open(f".cache/complex_unbind_machine{tag}.json", "w"), indent=1)
    A = {r["arm"]: r["agg"] for r in res}
    h23 = lambda a: (A[a]["heldout_hop2"] + A[a]["heldout_hop3"]) / 2
    say("HELDOUT hops 2-3: " + " | ".join(f"{a} {h23(a):.4f}" for a in A))
    if all(a in A for a in ("PH_oracle", "PH_learn", "MX_learn", "T4_width", "T4_params")):
        say(f"PAIR can writing be learned (PH_learn - PH_oracle): {h23('PH_learn') - h23('PH_oracle'):+.4f}")
        say(f"PAIR do phases beat learned binding (PH_learn - MX_learn): {h23('PH_learn') - h23('MX_learn'):+.4f}")
        say(f"PAIR structure vs none (max machine - max T4): "
            f"{max(h23(a) for a in ('PH_oracle', 'PH_learn', 'MX_learn')) - max(h23('T4_width'), h23('T4_params')):+.4f}")
    if mode == "feas":
        learns = max(A[a]["val"] for a in A) >= 0.80
        say(f"FEASIBILITY: some arm val >= 0.80: {learns} -> {'FEASIBLE (pin the bars)' if learns else 'NOT FEASIBLE'}")
    say(f"DONE in {time.time() - t0:.0f}s")
