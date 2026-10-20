"""tree_unbind_machine.py -- RUNG 1T-TREE: PATH-ADDRESSED REGISTERS IN THE TRAINED MACHINE (2026-10-20; Bryce's 'tree with
infinite leaves, a register per node'; Opus's coarse-only query; Bryce's 'the T6 clock might help'). CPU only.

THE QUESTION: a tree with unbounded leaves cannot keep a code table; a node's code is COMPUTED from its path,
code(child) = code(parent) (*) branch. Phasor products commute, so with ONE shared branch book the path a->b and b->a get the
same code (the tree loses its order). Which path codes, inside a trained lookup machine, (1) generalize to unseen paths,
(2) keep order, (3) answer a coarse-only query ('the leaf under a')? The tree race (10-16, analytic) is the prior:
products are exact but erase family resemblance; path-product sums route fuzzy; registers beat one bucket.

THE TASK (fresh scenes every batch): 8 branch labels, used at EVERY level (so order matters); leaves at depth 3; a scene =
S = 12 distinct paths (k1, k2, k3), each holding a filler entity (40 entities, learned codes). One token per leaf.
  EXACT query: the full path -> its filler. HELD OUT: 25% of the 512 triples never appear in training scenes (every label
  is seen at every level; only the combination is new); HELDOUT = the queried leaf is a held-out triple.
  SWAP (eval): the scene holds both (a, b, c) and (b, a, c), a != b, different fillers; the query is one of them.
  COARSE query: a prefix (k1) or (k1, k2) under which the scene holds exactly ONE leaf -> that leaf's filler. 25% of
  training queries are coarse (so a coarse failure is representational, not unseen).
  COARSE_HELD (eval): a coarse query whose unique leaf is a HELD-OUT triple. The tree is finite here (512 leaves), so a
  learned W_k can MEMORIZE which branch each seen code belongs to (the smoke caught PH_level at 0.93 on seen coarse);
  'infinite leaves' means the prefix must find a leaf the machine never saw.
  BUCKET (eval): EXACT at S = 24 (the static grows with the bucket in FLAT, not in per-leaf slots).
THE MACHINE (form 2's skeleton, the only road): key_j = code(path_j); q = code(query prefix); a = softmax(tau Re<W_q q,
  W_k key_j>) (W init identity, owner-attention supervised on the gold leaf, ATT_W); read v = sum_j a_j e(f_j); soft snap
  (beta, learned) over the scene's fillers. FLAT arm: one superposed bucket M = sum_j key_j (*) e(f_j), read M (*) conj(q).
ARMS (96 planes / 192 reals; phasor books FIXED random, entity codes LEARNED):
  PH_level     per-level books P[l, k]; key = prod_l P[l, k_l]                          (Bryce's per-level registers)
  PH_shared    one book P[k]; key = prod_l P[k_l]                                       (the order placebo: commutes)
  PH_clockfreq one book; level l uses harmonic (l+1): key = prod_l P[k_l]^(l+1)          (the clock as FREQUENCY per level)
  PH_clockoff  one book times a level clock phasor: key = prod_l C[l] P[k_l]            (the clock as OFFSET: commutes)
  PH_pathsum   per-level books; key = sum_{p<=depth} prod_{l<=p} P[l, k_l] (equal weights, the tree race's law)
  PH_flat      PH_level's codes, ONE superposed bucket (the static arm vs PH_level's per-leaf slots)
  MX_orth      one book of LEARNED orthogonal matrices (non-commutative); key = R[k3] R[k2] R[k1] r0
  T6_width     plain transformer, 6 layers, D 192: leaf token sum_l emb_l(k_l) + val(f); query sum_{l<=p} emb_l(k_l)
Bars pin in the ledger after --feas and before the full run.
Usage: .venv/bin/python3 scripts/toys/tree_unbind_machine.py [--feas | --smoke] -> .cache/tree_unbind_machine{tag}.*
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mode = "feas" if "--feas" in sys.argv else "smoke" if "--smoke" in sys.argv else "full"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import complex_unbind_toy as T  # noqa: E402   (RLayer; its N_LAYERS)

NB, L, N_ENT, S = 8, 3, 40, 12
MP, DR = 96, 192
ATT_W = 0.3
STEPS = 300 if mode == "smoke" else int(os.environ.get("TU_STEPS", "4000"))
BATCH, N_EVAL = 128, (500 if mode == "smoke" else 2000)
SEEDS = [0] if mode != "full" else [0, 1, 2]
ARMS = os.environ.get("TU_ARMS", "PH_level,PH_shared,PH_clockfreq,PH_clockoff,PH_pathsum,PH_flat,MX_orth,T6_width").split(",")
T.N_LAYERS = 6
PAD = NB                                                                     # label pad (coarse queries)


# ---------------------------------------------------------------- the task
def held_table(seed):
    rng = np.random.default_rng(700 + seed); h = np.zeros(NB ** L, bool)
    h[rng.choice(NB ** L, NB ** L // 4, replace=False)] = True
    return h                                                                 # indexed by k1*64 + k2*8 + k3


tid = lambda p: p[0] * NB * NB + p[1] * NB + p[2]


def make_batch(B, held, kind, rng, s=S, coarse_frac=0.25):
    """kind: 'train' | 'val' | 'heldout' | 'swap' | 'coarse' | 'coarse_held' (the unique leaf is a HELD-OUT triple). Returns leaves (B, s, 4) = (k1, k2, k3, f),
    query (B, 4) = (k1, k2, k3, depth) with PAD beyond depth, target filler (B,), gold leaf index (B,)."""
    allowed = np.where(~held)[0]; heldidx = np.where(held)[0]
    lv = np.zeros((B, s, 4), np.int64); q = np.zeros((B, 4), np.int64); t = np.zeros(B, np.int64); g = np.zeros(B, np.int64)
    for b in range(B):
        while True:
            trip = set(rng.choice(allowed, s, replace=False).tolist())
            if kind in ("heldout", "coarse_held"):
                trip.pop(); trip.add(int(rng.choice(heldidx)))
            paths = [(x // 64, (x // 8) % 8, x % 8) for x in trip]
            if kind == "swap":
                cand = [p for p in paths if p[0] != p[1] and not held[tid((p[1], p[0], p[2]))]
                        and (p[1], p[0], p[2]) not in paths]
                if not cand: continue
                p0 = cand[rng.integers(len(cand))]; tw = (p0[1], p0[0], p0[2])
                paths = [p for p in paths if p != p0][: s - 2] + [p0, tw]
            paths = [paths[i] for i in rng.permutation(len(paths))]
            fil = rng.integers(N_ENT, size=s)
            if kind == "swap":
                i0 = paths.index(p0); i1 = paths.index(tw)
                while fil[i1] == fil[i0]: fil[i1] = rng.integers(N_ENT)
                gi, depth = i0, 3
            elif kind == "heldout":
                gi = [i for i, p in enumerate(paths) if held[tid(p)]][0]; depth = 3
            elif kind == "coarse_held":
                gi = [i for i, p in enumerate(paths) if held[tid(p)]][0]; depth = int(rng.integers(1, 3))
                if sum(pp[:depth] == paths[gi][:depth] for pp in paths) != 1: continue
            else:
                depth = 3
                if kind == "coarse" or (kind == "train" and rng.random() < coarse_frac):
                    depth = int(rng.integers(1, 3))
                gi = int(rng.integers(s))
                if depth < 3:
                    uniq = [i for i, p in enumerate(paths) if sum(pp[:depth] == p[:depth] for pp in paths) == 1]
                    if not uniq:
                        if kind == "coarse": continue
                        depth = 3
                    else:
                        gi = uniq[rng.integers(len(uniq))]
            break
        lv[b] = [(*p, f) for p, f in zip(paths, fil)]
        q[b] = [*(paths[gi][l] if l < depth else PAD for l in range(L)), depth]; t[b] = fil[gi]; g[b] = gi
    return tuple(torch.from_numpy(x) for x in (lv, q, t, g))


# ---------------------------------------------------------------- the machine
class Machine(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(3000 + seed); self.kind = kind; self.cx = kind.startswith("PH")
        if self.cx:
            nbook = L if kind in ("PH_level", "PH_pathsum", "PH_flat") else 1
            th = torch.rand(nbook, NB, MP, generator=gen) * 2 * math.pi
            self.register_buffer("th", th)
            self.register_buffer("clk", torch.rand(L, MP, generator=gen) * 2 * math.pi)  # the offset clock's level phases
            te = torch.rand(N_ENT, MP, generator=gen) * 2 * math.pi
            self.ent = nn.Parameter(torch.cat([torch.cos(te), torch.sin(te)], -1))
            self.wq = nn.ParameterList([nn.Parameter(torch.eye(MP)), nn.Parameter(torch.zeros(MP, MP))])
            self.wk = nn.ParameterList([nn.Parameter(torch.eye(MP)), nn.Parameter(torch.zeros(MP, MP))])
            self.n = 2 * MP
        else:
            self.register_buffer("Q", torch.linalg.qr(torch.randn(NB, DR, DR, generator=gen))[0])
            self.A = nn.Parameter(torch.zeros(NB, DR, DR)); self.r0 = nn.Parameter(torch.randn(DR, generator=gen))
            self.ent = nn.Parameter(torch.randn(N_ENT, DR, generator=gen))
            self.wq, self.wk = nn.Parameter(torch.eye(DR)), nn.Parameter(torch.eye(DR)); self.n = DR
        self.log_tau = nn.Parameter(torch.zeros(())); self.log_beta = nn.Parameter(torch.tensor(math.log(2.0)))

    def lin(self, w, x):
        if not self.cx: return x @ w.T
        xr, xi = x[..., :MP], x[..., MP:]
        return torch.cat([xr @ w[0].T - xi @ w[1].T, xr @ w[1].T + xi @ w[0].T], -1)

    def code(self, ks, depth):
        """ks (..., 3) labels (PAD past depth), depth (...,) -> key code. Phase arms: [re | im]."""
        if not self.cx:
            R = self.Q @ torch.matrix_exp(self.A - self.A.transpose(-1, -2))
            v = self.r0.expand(*ks.shape[:-1], DR)
            for l in range(L):
                live = (depth > l)[..., None]
                v = torch.where(live, torch.einsum("...j,...ij->...i", v, R[ks[..., l].clamp(max=NB - 1)]), v)
            return v
        k = self.kind; ph = torch.zeros(*ks.shape[:-1], MP); acc = torch.zeros(*ks.shape[:-1], 2 * MP)
        for l in range(L):
            live = (depth > l)[..., None].float(); kl = ks[..., l].clamp(max=NB - 1)
            book = self.th[l] if self.th.shape[0] == L else self.th[0]
            step = book[kl] * (l + 1 if k == "PH_clockfreq" else 1) + (self.clk[l] if k == "PH_clockoff" else 0)
            ph = ph + live * step
            if k == "PH_pathsum":
                acc = acc + live * torch.cat([torch.cos(ph), torch.sin(ph)], -1)
        if k == "PH_pathsum":
            return acc / depth[..., None].float().sqrt()
        return torch.cat([torch.cos(ph), torch.sin(ph)], -1)

    def cmul(self, a, b, conj=False):
        ar, ai, br, bi = a[..., :MP], a[..., MP:], b[..., :MP], b[..., MP:]
        if conj: bi = -bi
        return torch.cat([ar * br - ai * bi, ar * bi + ai * br], -1)

    def forward(self, lv, q):
        E = self.ent; key = self.code(lv[..., :3], torch.full(lv.shape[:2], L)); qc = self.code(q[:, :3], q[:, 3])
        val = E[lv[..., 3]]; mask = F.one_hot(lv[..., 3], N_ENT).amax(1).bool()
        if self.kind == "PH_flat":
            u = self.cmul(self.cmul(key, val).sum(1), qc, conj=True); a = None
        else:
            s = torch.einsum("bd,btd->bt", self.lin(self.wq, qc), self.lin(self.wk, key)) / math.sqrt(self.n)
            a = F.softmax(s * self.log_tau.exp() * 4, -1); u = torch.einsum("bt,btd->bd", a, val)
        cos = F.normalize(u, dim=-1) @ F.normalize(E, dim=-1).T
        return (self.log_beta.exp() * cos).masked_fill(~mask, -1e4), a


class Plain(nn.Module):
    def __init__(self, D, seed):
        super().__init__(); torch.manual_seed(seed)
        self.lab = nn.Parameter(torch.randn(L, NB + 1, D) * 0.02); self.val = nn.Embedding(N_ENT, D)
        self.dep = nn.Embedding(4, D); self.layers = nn.ModuleList([T.RLayer(D) for _ in range(T.N_LAYERS)])
        self.lnf, self.out = nn.LayerNorm(D), nn.Linear(D, D)

    def forward(self, lv, q):
        x = torch.cat([sum(self.lab[l][lv[..., l]] for l in range(L)) + self.val(lv[..., 3]),
                       (sum(self.lab[l][q[:, l]] for l in range(L)) + self.dep(q[:, 3]))[:, None]], 1)
        for Ly in self.layers: x = Ly(x)
        mask = F.one_hot(lv[..., 3], N_ENT).amax(1).bool()
        return (self.out(self.lnf(x[:, -1])) @ self.val.weight.T).masked_fill(~mask, -1e4), None


def nparams(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build(arm, seed): return Plain(DR, seed) if arm == "T6_width" else Machine(arm, seed)


def run(arm, seed, lr):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); held = held_table(seed)
    erng = np.random.default_rng(10_000 + seed)
    ev = {k: make_batch(N_EVAL, held, k, erng) for k in ("val", "heldout", "swap", "coarse", "coarse_held")}
    ev["bucket24"] = make_batch(N_EVAL, held, "val", erng, s=24)
    net = build(arm, seed)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        lv, q, t, g = make_batch(BATCH, held, "train", rng)
        lg, a = net(lv, q); loss = F.cross_entropy(lg, t)
        if a is not None:
            loss = loss + ATT_W * -torch.log(a[torch.arange(len(g)), g] + 1e-6).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); res = {}
    with torch.no_grad():
        for k, (lv, q, t, g) in ev.items():
            lg, a = net(lv, q); res[k] = float((lg.argmax(-1) == t).float().mean())
            if k.startswith("coarse"):
                for d in (1, 2):
                    m = q[:, 3] == d; res[f"{k}_d{d}"] = float((lg.argmax(-1)[m] == t[m]).float().mean())
            if a is not None: res[f"{k}_attgold"] = float(a[torch.arange(len(g)), g].mean())
    res["params"] = nparams(net); res["final_loss"] = float(loss.detach())
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"feas": "_feas", "smoke": "_smoke", "full": ""}[mode] + os.environ.get("TU_TAG", "")
    assert mode == "smoke" or not os.path.exists(f".cache/tree_unbind_machine{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/tree_unbind_machine{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS}; trainable params: " + ", ".join(f"{a} {nparams(build(a, 0))}" for a in ARMS))
    lrs = {}
    for arm in ARMS:
        if mode == "full":
            sc = {lr: run(arm, 0, lr)["val"] for lr in (1e-3, 3e-3)}
            lrs[arm] = max(sc, key=sc.get); say(f"lr sweep {arm}: {sc} -> {lrs[arm]}  [{time.time() - t0:.0f}s]")
        else:
            lrs[arm] = 3e-3
    for arm in ARMS:
        rs = [run(arm, s, lrs[arm]) for s in SEEDS]
        A = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        res.append(dict(arm=arm, lr=lrs[arm], seeds=rs, agg=A))
        say(f"{arm:<12} params {rs[0]['params']:>7} loss {A['final_loss']:.3f}  val {A['val']:.3f}  HELDOUT {A['heldout']:.3f}"
            f"±{np.std([r['heldout'] for r in rs]):.3f}  SWAP {A['swap']:.3f}  COARSE {A['coarse']:.3f} (d1 {A['coarse_d1']:.3f} "
            f"d2 {A['coarse_d2']:.3f})  COARSE_HELD {A['coarse_held']:.3f}  S24 {A['bucket24']:.3f}  [{time.time() - t0:.0f}s]")
    json.dump(res, open(f".cache/tree_unbind_machine{tag}.json", "w"), indent=1)
    A = {r["arm"]: r["agg"] for r in res}
    g = lambda a, k: A[a][k] if a in A else float("nan")
    if mode == "feas":
        ok = max(A[a]["val"] for a in A) >= 0.80
        say(f"FEASIBILITY: some arm val >= 0.80: {ok} -> {'FEASIBLE (pin the bars)' if ok else 'NOT FEASIBLE'}")
    else:   # THE BARS (pinned 10-20 in the ledger before the full run)
        say(f"B1 ORDER NEEDS LEVELS: PH_shared SWAP {g('PH_shared', 'swap'):.3f} (<= 0.60) and PH_level SWAP "
            f"{g('PH_level', 'swap'):.3f} (>= 0.90) -> {g('PH_shared', 'swap') <= 0.60 and g('PH_level', 'swap') >= 0.90}")
        say(f"B2 THE CLOCK AS FREQUENCY CARRIES ORDER: PH_clockfreq SWAP {g('PH_clockfreq', 'swap'):.3f} (>= 0.90) -> "
            f"{g('PH_clockfreq', 'swap') >= 0.90}; THE CLOCK AS OFFSET COMMUTES: PH_clockoff SWAP {g('PH_clockoff', 'swap'):.3f} "
            f"(<= 0.60) -> {g('PH_clockoff', 'swap') <= 0.60}")
        say(f"B3 COARSE NEEDS SUMS (on UNSEEN leaves): PH_pathsum COARSE_HELD {g('PH_pathsum', 'coarse_held'):.3f} (>= 0.80) and "
            f"PH_level COARSE_HELD {g('PH_level', 'coarse_held'):.3f} (<= 0.40) -> "
            f"{g('PH_pathsum', 'coarse_held') >= 0.80 and g('PH_level', 'coarse_held') <= 0.40}")
        say(f"B3' PRODUCT KEYS MEMORIZE THEIR PREFIXES (seen minus unseen coarse): PH_level {g('PH_level', 'coarse') - g('PH_level', 'coarse_held'):+.3f}"
            f" | MX_orth {g('MX_orth', 'coarse') - g('MX_orth', 'coarse_held'):+.3f} | PH_pathsum {g('PH_pathsum', 'coarse') - g('PH_pathsum', 'coarse_held'):+.3f}")
        g12, g24 = g("PH_level", "val") - g("PH_flat", "val"), g("PH_level", "bucket24") - g("PH_flat", "bucket24")
        say(f"B4 SLOTS BEAT THE BUCKET, AND THE STATIC GROWS WITH IT: PH_level - PH_flat S12 {g12:+.3f}, S24 {g24:+.3f} "
            f"(S24 gap >= 0.05 and >= S12 gap + 0.05) -> {g24 >= 0.05 and g24 >= g12 + 0.05}")
        say(f"B5 PATHS GENERALIZE: PH_level HELDOUT {g('PH_level', 'heldout'):.3f} >= val - 0.05 ({g('PH_level', 'val'):.3f}) -> "
            f"{g('PH_level', 'heldout') >= g('PH_level', 'val') - 0.05}")
        say(f"CONTEXT (no bars): HELDOUT  " + " | ".join(f"{a} {A[a]['heldout']:.3f}" for a in A)
            + f"\nCONTEXT (no bars): PH_pathsum - PH_level EXACT val {g('PH_pathsum', 'val') - g('PH_level', 'val'):+.3f} "
            f"(the price of fuzzy keys in slots); MX_orth SWAP {g('MX_orth', 'swap'):.3f} COARSE {g('MX_orth', 'coarse'):.3f}")
    say(f"DONE in {time.time() - t0:.0f}s")
