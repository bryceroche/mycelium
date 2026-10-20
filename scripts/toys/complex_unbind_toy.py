"""complex_unbind_toy.py -- RUNG 1T: THE UNBINDING TASK (2026-10-19; Bryce's word; Opus's three catches folded in).
CPU only, never the card. Design: docs/complex_planes.md §9. Bars pin in the ledger AFTER the feasibility read
(--feas) and BEFORE the grid.

THE QUESTION: when the task REQUIRES unbinding (role-filler records queried by role, multi-hop, held-out role x filler
combinations at EVERY hop), does PHASE unbinding (fixed phasor role codes, exact conj inverse, zero parameters) beat
LEARNED unbinding (role matrices, tensor-product style) and plain width?

THE TASK (fresh scenes every batch: no finite pool, so nothing can be memorized -- v2's death).
  40 entities; 6 roles, each written with one of 3 alias tokens (18 aliases). A scene: 6 entities, each with a record of
  K facts (K distinct roles), every filler an ENTITY of the scene (so chains continue; catch 3). One token per fact:
  (owner, alias, filler), the facts shuffled. Query: (X, alias_1, alias_2, alias_3, depth d): "role_d of ... of
  (role_1 of X)", d in 1..3, hop 1 applied first. The answer is an entity.
  HELD OUT: per role, 25% of the entities never appear as that role's filler in training (every role and every entity is
  seen; only the combination is new). TEST: every hop of the chain lands on a held-out (role, filler) pair; the other
  facts are drawn from the allowed pairs. VAL: fresh scenes from the training distribution.

ARMS (3 layers, 4 heads, readout at the query position; stream width FIXED at 96 reals = 48 planes except R_params):
  C_clean+U  native complex (rung 0's CLayer, cleanup MLP) + UNBINDING HEADS: facts are WRITTEN bound,
             m = e(owner) + e(filler) (*) role(alias), role = unit-per-plane(sum_k rho_k P_k), P = 8 FIXED random phasor
             codes, rho = a learned alias -> code choice; layer l's unbinding head (query position only, used when l < d)
             attends over the memory m (learned q/k), reads r = sum a_j m_j, and adds r (*) conj(role(alias_l)) to the
             query's residual. One entity table (owner and filler): unbinding returns a vector the next hop can match.
  C_split+U  the same organs; the MLP is SiLU on Re/Im separately (breaks the algebra inside the layer)
  R_tpr+U    real D=96, the same organs with LEARNED role matrices (catch 1): role = sum_k rho_k R_k (8 learned D x D),
             write e(filler) @ role, unbind r @ role^T. Same entity table, same memory, same head placement.
  R_width    real D=96 plain transformer (v1's layer, SwiGLU x1): fact token = own(o) + alias(a) + fil(f) (two entity
             tables: an additive bundle cannot carry order otherwise); query = own(X) + slot_h(alias_h) + depth
  R_params   R_width at the width whose parameter count is closest to C_clean+U's (chosen at run time, multiple of 4)
COMPARISONS REPORTED SEPARATELY (catch 2): C_clean+U - C_split+U (does the cleanup memory need the algebra?);
  C_clean+U - R_tpr+U (does unbinding need the PHASES?); C_clean+U - max(R_width, R_params) (does it beat width?).
Usage: .venv/bin/python3 scripts/toys/complex_unbind_toy.py [--feas | --smoke]  -> .cache/complex_unbind_toy{tag}.txt/.json
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mode = "feas" if "--feas" in sys.argv else "smoke" if "--smoke" in sys.argv else "full"
sys.argv = [sys.argv[0]] + (["--smoke"] if mode != "full" else [])
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import complex_native_toy as R0  # noqa: E402

N_ENT, N_ROLE, N_ALIAS_PER, N_SCENE = 40, 6, 3, 6
N_ALIAS = N_ROLE * N_ALIAS_PER; PAD_A = N_ALIAS
HELD_FRAC, N_CODES, N_LAYERS, N_HEADS = 0.25, 8, 3, 4
KF = int(os.environ.get("UT_K", "3"))
STEPS = 300 if mode == "smoke" else int(os.environ.get("UT_STEPS", "4000"))
BATCH, N_EVAL = 128, (500 if mode == "smoke" else 2000)
SEEDS = [0] if mode != "full" else [0, 1, 2]
ARMS = ["C_clean+U", "C_split+U", "R_tpr+U", "R_width", "R_params"]
M = R0.M                                                                     # 48 planes


# ---------------------------------------------------------------- the task
def held_table(seed):
    rng = np.random.default_rng(500 + seed)
    held = np.zeros((N_ROLE, N_ENT), bool)
    for r in range(N_ROLE):
        held[r, rng.choice(N_ENT, int(HELD_FRAC * N_ENT), replace=False)] = True
    return held


def make_batch(B, held, test, rng):
    facts = np.zeros((B, N_SCENE * KF, 3), np.int64); query = np.zeros((B, 5), np.int64); tgt = np.zeros(B, np.int64)
    alias = lambda r: r * N_ALIAS_PER + rng.integers(N_ALIAS_PER)
    for b in range(B):
        while True:
            ents = rng.choice(N_ENT, N_SCENE, replace=False); d = int(rng.integers(1, 4))
            path = ents[:d + 1]                                              # distinct chain nodes
            roles, ok = [], True
            for h in range(d):
                cand = [r for r in range(N_ROLE) if held[r, path[h + 1]] == test]
                if not cand: ok = False; break
                roles.append(int(rng.choice(cand)))
            if not ok: continue
            rec = {int(e): {} for e in ents}
            for h in range(d): rec[int(path[h])][roles[h]] = int(path[h + 1])
            for e in ents:
                e = int(e)
                for r in rng.permutation([r for r in range(N_ROLE) if r not in rec[e]]):
                    if len(rec[e]) == KF: break
                    fl = [int(x) for x in ents if x != e and not held[r, x]]
                    if fl: rec[e][int(r)] = int(rng.choice(fl))
            if all(len(v) == KF for v in rec.values()): break                  # else redraw the scene (rare)
        fs = [(o, alias(r), f) for o in rec for r, f in rec[o].items()]
        facts[b] = np.array(fs)[rng.permutation(len(fs))]
        qa = [alias(r) for r in roles] + [PAD_A] * (3 - d)
        query[b] = [path[0], *qa, d]; tgt[b] = path[d]
    return torch.from_numpy(facts), torch.from_numpy(query), torch.from_numpy(tgt)


# ---------------------------------------------------------------- the real arms
class RLayer(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.D, self.hd = D, D // N_HEADS
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(D, D) for _ in range(4))
        self.w1, self.w3, self.w2 = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)

    def forward(self, x):
        B_, T_, D = x.shape; h = self.ln1(x)
        sp = lambda t: t.view(B_, T_, N_HEADS, self.hd).transpose(1, 2)
        a = F.softmax(sp(self.wq(h)) @ sp(self.wk(h)).transpose(-1, -2) / math.sqrt(self.hd), -1)
        x = x + self.wo((a @ sp(self.wv(h))).transpose(1, 2).reshape(B_, T_, D))
        h = self.ln2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class RNet(nn.Module):                                                       # R_width / R_params / R_tpr+U
    def __init__(self, D, unbind, seed):
        super().__init__()
        self.D, self.unbind = D, unbind
        self.layers = nn.ModuleList([RLayer(D) for _ in range(N_LAYERS)])
        self.alias = nn.Embedding(N_ALIAS + 1, D); self.slot = nn.Parameter(torch.randn(3, N_ALIAS + 1, D) * 0.02)
        self.depth = nn.Embedding(4, D); self.lnf = nn.LayerNorm(D); self.out = nn.Linear(D, N_ENT)
        if unbind:
            self.ent = nn.Embedding(N_ENT, D)
            self.rho = nn.Embedding(N_ALIAS + 1, N_CODES)
            self.R = nn.Parameter(torch.randn(N_CODES, D, D) / math.sqrt(D))  # LEARNED role matrices
            self.uq = nn.ModuleList([nn.Linear(D, D) for _ in range(N_LAYERS)])
            self.uk = nn.ModuleList([nn.Linear(D, D) for _ in range(N_LAYERS)])
            self.un = nn.ModuleList([nn.LayerNorm(D) for _ in range(N_LAYERS)])
        else:
            self.own, self.fil = nn.Embedding(N_ENT, D), nn.Embedding(N_ENT, D)

    def role(self, a):
        return torch.einsum("...k,kij->...ij", F.softmax(self.rho(a), -1), self.R)

    def forward(self, facts, query):
        q0 = sum(self.slot[h][query[:, 1 + h]] for h in range(3)) + self.depth(query[:, 4])
        if self.unbind:
            mem = self.ent(facts[..., 0]) + torch.einsum("bti,btij->btj", self.ent(facts[..., 2]), self.role(facts[..., 1]))
            qx = self.ent(query[:, 0]) + q0
            x = torch.cat([mem, qx[:, None]], 1)
        else:
            fx = self.own(facts[..., 0]) + self.alias(facts[..., 1]) + self.fil(facts[..., 2])
            x = torch.cat([fx, (self.own(query[:, 0]) + q0)[:, None]], 1)
        for l, L in enumerate(self.layers):
            if self.unbind:
                h = self.un[l](x[:, -1])
                s = torch.einsum("bd,btd->bt", self.uq[l](h), self.uk[l](mem)) / math.sqrt(self.D)
                r = torch.einsum("bt,btd->bd", F.softmax(s, -1), mem)
                u = torch.einsum("bj,bij->bi", r, self.role(query[:, 1 + l]))        # r @ role^T
                g = (query[:, 4] > l).float()[:, None]
                x = torch.cat([x[:, :-1], (x[:, -1] + g * u)[:, None]], 1)
            x = L(x)
        return self.out(self.lnf(x[:, -1]))


# ---------------------------------------------------------------- the complex arms
class CUNet(nn.Module):                                                      # C_clean+U / C_split+U
    def __init__(self, mlp, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(1000 + seed)
        self.layers = nn.ModuleList([R0.CLayer(mlp, gen) for _ in range(N_LAYERS)])
        self.ent = nn.Embedding(N_ENT, 2 * M); self.slot = nn.Parameter(torch.randn(3, N_ALIAS + 1, 2 * M) * 0.02)
        self.depth = nn.Embedding(4, 2 * M); self.rho = nn.Embedding(N_ALIAS + 1, N_CODES)
        th = torch.rand(N_CODES, M, generator=gen) * 2 * math.pi                # FIXED phasor role codes, never trained
        self.register_buffer("Pr", torch.cos(th)); self.register_buffer("Pi", torch.sin(th))
        self.uq = nn.ModuleList([R0.CLinear(M, M) for _ in range(N_LAYERS)])
        self.uk = nn.ModuleList([R0.CLinear(M, M) for _ in range(N_LAYERS)])
        self.un = nn.ModuleList([R0.CNorm(M) for _ in range(N_LAYERS)])
        self.nf = R0.CNorm(M); self.out = nn.Linear(2 * M, N_ENT)

    def role(self, a):                                                       # unit phasor per plane
        w = F.softmax(self.rho(a), -1); pr, pi = w @ self.Pr, w @ self.Pi
        n = torch.sqrt(pr * pr + pi * pi + 1e-6)
        return pr / n, pi / n

    def forward(self, facts, query):
        e = lambda i: (self.ent(i)[..., :M], self.ent(i)[..., M:])
        (orr, oi), (fr, fi), (rr, ri) = e(facts[..., 0]), e(facts[..., 2]), self.role(facts[..., 1])
        mr, mi = orr + fr * rr - fi * ri, oi + fr * ri + fi * rr                 # owner + filler (*) role
        q0 = sum(self.slot[h][query[:, 1 + h]] for h in range(3)) + self.depth(query[:, 4]) + self.ent(query[:, 0])
        xr, xi = torch.cat([mr, q0[:, None, :M]], 1), torch.cat([mi, q0[:, None, M:]], 1)
        for l, L in enumerate(self.layers):
            hr, hi = self.un[l](xr[:, -1], xi[:, -1])
            qr, qi = self.uq[l](hr, hi); kr, ki = self.uk[l](mr, mi)
            s = (torch.einsum("bd,btd->bt", qr, kr) + torch.einsum("bd,btd->bt", qi, ki)) / math.sqrt(2 * M)
            a = F.softmax(s, -1)
            rr_, ri_ = torch.einsum("bt,btd->bd", a, mr), torch.einsum("bt,btd->bd", a, mi)
            pr, pi = self.role(query[:, 1 + l])
            ur, ui = rr_ * pr + ri_ * pi, ri_ * pr - rr_ * pi                     # read (*) conj(role)
            g = (query[:, 4] > l).float()[:, None]
            xr = torch.cat([xr[:, :-1], (xr[:, -1] + g * ur)[:, None]], 1)
            xi = torch.cat([xi[:, :-1], (xi[:, -1] + g * ui)[:, None]], 1)
            xr, xi = L(xr, xi)
        xr, xi = self.nf(xr, xi)
        return self.out(torch.cat([xr[:, -1], xi[:, -1]], -1))


def nparams(m): return sum(p.numel() for p in m.parameters())


def r_params_width():
    target = nparams(CUNet("clean", 0))
    return min(range(16, 200, 4), key=lambda D: abs(nparams(RNet(D, False, 0)) - target))


def build(arm, seed, Dp=None):
    return {"C_clean+U": lambda: CUNet("clean", seed), "C_split+U": lambda: CUNet("split", seed),
            "R_tpr+U": lambda: RNet(2 * M, True, seed), "R_width": lambda: RNet(2 * M, False, seed),
            "R_params": lambda: RNet(Dp, False, seed)}[arm]()


def run(arm, seed, lr, Dp):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); held = held_table(seed)
    erng = np.random.default_rng(10_000 + seed)
    ev = {"val": make_batch(N_EVAL, held, False, erng), "heldout": make_batch(N_EVAL, held, True, erng)}
    net = build(arm, seed, Dp); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        fa, q, t = make_batch(BATCH, held, False, rng)
        loss = F.cross_entropy(net(fa, q), t)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); res = {}
    with torch.no_grad():
        for split, (fa, q, t) in ev.items():
            pr = net(fa, q).argmax(-1); res[split] = float((pr == t).float().mean())
            for d in (1, 2, 3):
                m = q[:, 4] == d; res[f"{split}_hop{d}"] = float((pr[m] == t[m]).float().mean())
    res["params"] = nparams(net); res["final_loss"] = float(loss.detach())
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"feas": "_feas", "smoke": "_smoke", "full": ""}[mode] + (f"_K{KF}" if KF != 3 else "")
    out = open(f".cache/complex_unbind_toy{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    Dp = r_params_width()
    say(f"mode {mode} K={KF} steps {STEPS}; R_params width {Dp}; params: " +
        ", ".join(f"{a} {nparams(build(a, 0, Dp))}" for a in ARMS))
    fa, q, t = make_batch(4, held_table(0), True, np.random.default_rng(0))
    say(f"sample (held-out) facts[0] {fa[0].tolist()} query {q[0].tolist()} -> {int(t[0])}")
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
        say(f"{arm:<10} params {rs[0]['params']:>7}  loss {agg['final_loss']:.3f}  val {agg['val']:.3f} "
            f"(h1 {agg['val_hop1']:.3f} h2 {agg['val_hop2']:.3f} h3 {agg['val_hop3']:.3f})  HELDOUT {agg['heldout']:.3f}±{sd:.3f} "
            f"(h1 {agg['heldout_hop1']:.3f} h2 {agg['heldout_hop2']:.3f} h3 {agg['heldout_hop3']:.3f})  [{time.time() - t0:.0f}s]")
    json.dump(res, open(f".cache/complex_unbind_toy{tag}.json", "w"), indent=1)
    A = {r["arm"]: r["agg"] for r in res}
    h23 = lambda a: (A[a]["heldout_hop2"] + A[a]["heldout_hop3"]) / 2
    say("HELDOUT hops 2-3: " + " | ".join(f"{a} {h23(a):.4f}" for a in ARMS))
    say(f"CMP cleanup needs the algebra (C_clean+U - C_split+U): {h23('C_clean+U') - h23('C_split+U'):+.4f}")
    say(f"CMP unbinding needs the phases (C_clean+U - R_tpr+U): {h23('C_clean+U') - h23('R_tpr+U'):+.4f}")
    say(f"CMP beats width (C_clean+U - max(R_width, R_params)): {h23('C_clean+U') - max(h23('R_width'), h23('R_params')):+.4f}")
    if mode == "feas":
        learns = max(A[a]["val"] for a in ARMS) >= 0.80; open_ = A["R_width"]["heldout_hop2"] <= 0.90
        say(f"FEASIBILITY: some arm val >= 0.80: {learns}; R_width held-out hop2 <= 0.90: {open_} -> "
            f"{'FEASIBLE (pin the bars)' if learns and open_ else 'NOT FEASIBLE (redesign before any bars)'}")
    say(f"DONE in {time.time() - t0:.0f}s")
