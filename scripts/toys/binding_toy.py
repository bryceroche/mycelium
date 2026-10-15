"""binding_toy.py -- THE BINDING TOY (2026-10-15; Bryce's phase-binding thesis, Opus/Fable brainstorm; CPU only, torch,
never the card, never the repo's head). Self-contained.

THE QUESTION: does a parameter-free PHASE BINDER (fixed random unit phasors, learned role choice) replace part of what an
MLP learns -- (1) generalization to role-filler COMBINATIONS never seen in training (the 2^N claim, relocated from storage to
systematicity), (2) multi-hop survival, (3) facts stored per parameter (the knowledge control)?

PART 1 -- IN-CONTEXT ENTITY TRACKING (the compositional test). A scene: 6 entities drawn from N_NAMES names; each gets 3
attributes from N_ATTR (value 0..9, random per scene) and one PARTNER (a derangement within the scene). One fact = one
token whose embedding is the SUM name + relation + value (an additive bundle: the twin problem by construction -- a fact
sharing only the name, or only the attribute, half-matches a query). Query token = name + relation-asked + hop flag.
Hops: 1 = attr(X); 2 = attr(partner(X)); 3 = attr(partner(partner(X))). COVERAGE f: training scenes use ONLY (name, attr)
pairs in an allowed set (fraction f of all pairs; every name and every attr appears); the TEST queries a held-out pair
(the queried fact's (name, attr) never co-occurred in any training scene). An in-distribution val (allowed pairs) is
reported beside it.
ARMS (one backbone: d=64, 3 pre-norm layers of 4-head attention + SwiGLU MLP, readout at the query token):
  A   standard, MLP ratio 4                      (the reference transformer -- attention binds by matching)
  A4  standard, MLP ratio 1                      (the MLP cut 4x; no binder: the matched-capacity control)
  B   A4 + PHASE-BOUND q/k: q = bind(Wq x, sum_r softmax(Gq x)_r P_r), the same for k; P_r fixed random unit phasors
      over d/2 complex planes per head (R=16 roles), never trained; bind = complex elementwise product
  E   A4 + LEARNED MULTIPLICATIVE q/k: q = (Wq1 x) * (Wq2 x) (the 'any multiplication' rival; ~matched params to B)
DECLARED CONFOUND: at low f the queried entity has fewer SAME-NAME facts (few of its pairs are allowed), so scenes get
somewhat easier as f falls; the out-of-scene padding holds the TOTAL real-fact count fixed. READ ARM GAPS WITHIN EACH f
(identical data for every arm); the across-f slope carries difficulty drift.
FAIRNESS (pinned): every arm gets the same lr sweep {1e-3, 3e-3} picked on the IN-DISTRIBUTION val at f=0.5 seed 0;
same steps, batch, data; 3 seeds per cell; params reported per arm.

PART 2 -- THE KNOWLEDGE CONTROL (no context: memorize a table (name, attr) -> value in 0..9). Addressing into one hidden
SwiGLU layer of width h: SUM (e_name + e_attr), CAT ([e_name, e_attr]), BOUND (e_name (*) phasor(attr), fixed phasors),
MULT (learned (W1 e_name) * (W2 e_attr)). Conditions: INDEPENDENT values; CORRELATED (names in clusters of 8 sharing 70 %
of their attribute values). Metric: the largest table (names x 16 attrs) stored at >= 95 % recall, as BITS PER PARAMETER
(table cells x log2(10) / params), per h.

PREDICTIONS (pinned before running; both authors):
  Part 1, random/in-dist val: all arms ~equal.  Held-out combinations: B > A4 by a wide margin as f shrinks (0.5 -> 0.02);
    B >= E (Fable) / B ~ E (Opus); A between; the B advantage grows with hop depth.
  Part 2: Fable -- BOUND/MULT beat SUM at small h (1.3-2x bits/param), converging at large h; largest in CORRELATED.
          Opus  -- no gain from BOUND anywhere (the ~2 bits/param ceiling; learned embeddings absorb the decorrelation).
READING: B >> A4 on held-out and B > E -> fixed unitary binding buys systematicity (the thesis, outside Mycelium);
  B ~ E > A4 -> multiplication matters, phases are a form of it; no gap -> no binding tax at toy scale (a scoped null);
  BOUND wins Part 2 at every h -> Opus wrong on the knowledge share.
Usage: .venv/bin/python3 scripts/toys/binding_toy.py [--smoke] [--part 1|2|all]  -> .cache/binding_toy_*.json/.txt
"""
import argparse, json, math, os, time, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--part", default="all")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default=".cache/binding_toy")
args = ap.parse_args()
torch.set_num_threads(args.threads)

N_NAMES, N_ATTR, N_VAL, E_SCENE, A_PER = 60, 12, 10, 6, 3
REL_PARTNER = N_ATTR                    # relation ids: 0..11 attrs, 12 partner
D, N_LAYERS, N_HEADS, R_ROLES = 64, 3, 4, 16
STEPS = 300 if args.smoke else 6000
BATCH = 128
COVER = [0.5, 0.1] if args.smoke else [0.5, 0.2, 0.1, 0.05, 0.02]
SEEDS = [0] if args.smoke else [0, 1, 2]
ARMS = ["A", "A4", "B", "E"]


# ------------------------------------------------------------------ data (part 1)
def allowed_pairs(f, rng):
    """fraction f of (name, attr) pairs; every name and every attr kept (>= 1 pair each)"""
    allp = [(n, a) for n in range(N_NAMES) for a in range(N_ATTR)]
    k = max(int(round(f * len(allp))), N_NAMES)
    perm = rng.permutation(len(allp))
    chosen = set()
    for n in range(N_NAMES):                          # cover every name
        chosen.add((n, int(rng.integers(N_ATTR))))
    for a in range(N_ATTR):                           # cover every attr
        chosen.add((int(rng.integers(N_NAMES)), a))
    for i in perm:
        if len(chosen) >= k: break
        chosen.add(allp[i])
    A = np.zeros((N_NAMES, N_ATTR), bool)
    for n, a in chosen: A[n, a] = True
    return A


NULL_NAME = N_NAMES                     # padding facts carry a name no query can ask (distractors, never matched)


def make_batch(B, allowed, heldout_query, rng):
    """facts: (B, NF, 3) [name, rel, val] (val = digit for attrs, name id for partner); query: (B, 3) [name, rel, hop];
    target: (B,) digit. Each entity carries up to A_PER of ITS allowed attrs (>= 1; low coverage leaves names with one),
    padded to a fixed NF with real facts of out-of-scene names (allowed pairs). heldout_query: the queried (final entity, attr) pair is NOT
    allowed (test) / allowed (train); the held-out pair is the only non-allowed pair in a test scene."""
    NF = E_SCENE * (A_PER + 1)
    facts = np.zeros((B, NF, 3), np.int64); query = np.zeros((B, 3), np.int64); tgt = np.zeros(B, np.int64)
    for b in range(B):
        while True:
            names = rng.choice(N_NAMES, E_SCENE, replace=False)
            hop = int(rng.integers(1, 4))
            perm = rng.permutation(E_SCENE)
            while (perm == np.arange(E_SCENE)).any(): perm = rng.permutation(E_SCENE)
            q_ent = int(rng.integers(E_SCENE)); fin = q_ent
            for _ in range(hop - 1): fin = int(perm[fin])
            fname = names[fin]
            cand = np.where(~allowed[fname] if heldout_query else allowed[fname])[0]
            if len(cand) == 0: continue
            q_attr = int(rng.choice(cand))
            rows = []
            for i, n in enumerate(names):
                pool = [a for a in np.where(allowed[n])[0] if not (i == fin and a == q_attr)]
                k = min(len(pool), A_PER - 1 if i == fin else A_PER)
                attrs = list(rng.choice(pool, k, replace=False)) if k > 0 else []
                if i == fin: attrs = [q_attr] + attrs
                for a in attrs:
                    v = int(rng.integers(N_VAL)); rows.append((n, int(a), v))
                    if i == fin and a == q_attr: tgt[b] = v
                rows.append((n, REL_PARTNER, int(names[perm[i]])))
            # pad with REAL facts of names OUTSIDE the scene (allowed pairs only): the same count of real, half-matching
            # distractors at every coverage f (null padding made low-f scenes EASIER -- the smoke's confound, fixed)
            outside = np.setdiff1d(np.arange(N_NAMES), names)
            while len(rows) < NF:
                n = int(rng.choice(outside)); pa = np.where(allowed[n])[0]
                rows.append((n, int(rng.choice(pa)), int(rng.integers(N_VAL))))
            rows = [rows[i] for i in rng.permutation(len(rows))]
            facts[b] = np.array(rows); query[b] = (names[q_ent], q_attr, hop)
            break
    return torch.from_numpy(facts), torch.from_numpy(query), torch.from_numpy(tgt)


_POOLS = {}


def pools(f, seed):
    """ONE data pool per (f, seed), shared by every arm (identical data; generated once)"""
    key = (f, seed)
    if key not in _POOLS:
        allowed = allowed_pairs(f, np.random.default_rng(100 + seed))
        n_train = 4096 if args.smoke else 120_000
        tr = make_batch(n_train, allowed, False, np.random.default_rng(seed))
        erng = np.random.default_rng(10_000 + seed)
        ev = {"val": make_batch(2000, allowed, False, erng), "heldout": make_batch(2000, allowed, True, erng)}
        _POOLS[key] = (tr, ev)
    return _POOLS[key]


# ------------------------------------------------------------------ model (part 1)
def bind(x, ph):
    """x (..., 2m) real -> complex m planes; ph (..., m) complex unit; returns real (..., 2m)"""
    m = x.shape[-1] // 2
    z = torch.complex(x[..., :m], x[..., m:]) * ph
    return torch.cat([z.real, z.imag], -1)


class Layer(nn.Module):
    def __init__(self, arm, gen):
        super().__init__()
        self.arm = arm; hd = D // N_HEADS; self.hd = hd
        ratio = 4 if arm == "A" else 1
        self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
        self.wq = nn.Linear(D, D); self.wk = nn.Linear(D, D); self.wv = nn.Linear(D, D); self.wo = nn.Linear(D, D)
        if arm == "B":
            self.gq = nn.Linear(D, N_HEADS * R_ROLES); self.gk = nn.Linear(D, N_HEADS * R_ROLES)
            th = torch.rand(N_HEADS, R_ROLES, hd // 2, generator=gen) * 2 * math.pi   # FIXED, never trained
            self.register_buffer("P", torch.polar(torch.ones_like(th), th))
        if arm == "E":
            self.wq2 = nn.Linear(D, D); self.wk2 = nn.Linear(D, D)
        self.w1 = nn.Linear(D, ratio * D); self.w3 = nn.Linear(D, ratio * D); self.w2 = nn.Linear(ratio * D, D)

    def qk(self, h, w, g, w2):
        B_, T_, _ = h.shape
        x = w(h).view(B_, T_, N_HEADS, self.hd)
        if self.arm == "B":
            r = F.softmax(g(h).view(B_, T_, N_HEADS, R_ROLES), -1)            # learned role choice
            ph = torch.einsum("bthr,hrm->bthm", r.to(torch.complex64), self.P)
            ph = ph / (ph.abs() + 1e-6)                                         # unit phasor per plane
            x = bind(x, ph)
        elif self.arm == "E":
            x = x * w2(h).view(B_, T_, N_HEADS, self.hd)
        return x.transpose(1, 2)

    def forward(self, x):
        h = self.ln1(x)
        q = self.qk(h, self.wq, getattr(self, "gq", None), getattr(self, "wq2", None))
        k = self.qk(h, self.wk, getattr(self, "gk", None), getattr(self, "wk2", None))
        B_, T_, _ = h.shape
        v = self.wv(h).view(B_, T_, N_HEADS, self.hd).transpose(1, 2)
        a = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd), -1)
        x = x + self.wo((a @ v).transpose(1, 2).reshape(B_, T_, D))
        h = self.ln2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Net(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(1000 + seed)
        self.name = nn.Embedding(N_NAMES + 1, D); self.rel = nn.Embedding(N_ATTR + 1, D)
        self.val = nn.Embedding(N_VAL, D); self.hop = nn.Embedding(4, D)
        self.layers = nn.ModuleList([Layer(arm, gen) for _ in range(N_LAYERS)])
        self.lnf = nn.LayerNorm(D); self.out = nn.Linear(D, N_VAL)

    def forward(self, facts, query):
        isp = facts[..., 1] == REL_PARTNER
        valemb = torch.where(isp[..., None], self.name(facts[..., 2]), self.val(facts[..., 2].clamp(max=N_VAL - 1)))
        fx = self.name(facts[..., 0]) + self.rel(facts[..., 1]) + valemb
        qx = self.name(query[:, 0]) + self.rel(query[:, 1]) + self.hop(query[:, 2])
        x = torch.cat([fx, qx[:, None]], 1)
        for L in self.layers: x = L(x)
        return self.out(self.lnf(x[:, -1]))


def nparams(m): return sum(p.numel() for p in m.parameters())


def run_part1(arm, f, seed, lr):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    (tf, tq, tt), ev = pools(f, seed)
    net = Net(arm, seed); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        ix = torch.from_numpy(rng.integers(len(tt), size=BATCH))
        loss = F.cross_entropy(net(tf[ix], tq[ix]), tt[ix])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); res = {}
    with torch.no_grad():
        for split in ("val", "heldout"):
            fa, q, t = ev[split]
            pr = net(fa, q).argmax(-1)
            res[split] = float((pr == t).float().mean())
            for hop in (1, 2, 3):
                m = q[:, 2] == hop
                res[f"{split}_hop{hop}"] = float((pr[m] == t[m]).float().mean())
    res["params"] = nparams(net); res["final_loss"] = float(loss.detach())
    return res


# ------------------------------------------------------------------ part 2: the knowledge control
class Mem(nn.Module):
    def __init__(self, mode, n_names, h, seed):
        super().__init__()
        self.mode = mode; d = 32
        g = torch.Generator().manual_seed(2000 + seed)
        self.en = nn.Embedding(n_names, d); self.ea = nn.Embedding(16, d)
        din = 2 * d if mode == "CAT" else d
        if mode == "BOUND":
            th = torch.rand(16, d // 2, generator=g) * 2 * math.pi
            self.register_buffer("P", torch.polar(torch.ones_like(th), th))
            self.ea = None
        if mode == "MULT":
            self.w1 = nn.Linear(d, d, bias=False); self.w2 = nn.Linear(d, d, bias=False)
        self.u = nn.Linear(din, h); self.g = nn.Linear(din, h); self.o = nn.Linear(h, 10)

    def forward(self, n, a):
        e = self.en(n)
        if self.mode == "SUM": x = e + self.ea(a)
        elif self.mode == "CAT": x = torch.cat([e, self.ea(a)], -1)
        elif self.mode == "BOUND": x = bind(e, self.P[a])
        else: x = self.w1(e) * self.w2(self.ea(a))
        return self.o(F.silu(self.g(x)) * self.u(x))


def table(n_names, cond, rng):
    V = rng.integers(10, size=(n_names, 16))
    if cond == "CORRELATED":
        for c0 in range(0, n_names, 8):
            proto = rng.integers(10, size=16); share = rng.random((min(8, n_names - c0), 16)) < 0.7
            V[c0:c0 + 8] = np.where(share, proto[None], V[c0:c0 + 8])
    return V


def recall(mode, n_names, h, cond, seed, steps):
    torch.manual_seed(seed); V = torch.from_numpy(table(n_names, cond, np.random.default_rng(seed)))
    n = torch.arange(n_names).repeat_interleave(16); a = torch.arange(16).repeat(n_names); y = V.reshape(-1)
    m = Mem(mode, n_names, h, seed); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        loss = F.cross_entropy(m(n, a), y); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): acc = float((m(n, a).argmax(-1) == y).float().mean())
    # params that scale with storage: everything (embeddings included -- they are where a table can hide)
    return acc, nparams(m)


def run_part2(out):
    HS = [16, 64] if args.smoke else [16, 32, 64, 128]
    SIZES = [32, 64, 128] if args.smoke else [16, 32, 64, 128, 256, 512, 1024]
    steps = 300 if args.smoke else 3000
    rows = []
    for cond, mode, h in itertools.product(["INDEPENDENT", "CORRELATED"], ["SUM", "CAT", "BOUND", "MULT"], HS):
        best = None
        for nn_ in SIZES:
            accs = [recall(mode, nn_, h, cond, s, steps) for s in SEEDS]
            acc = float(np.mean([x[0] for x in accs])); P = accs[0][1]
            rows.append(dict(cond=cond, mode=mode, h=h, names=nn_, acc=acc, params=P))
            if acc >= 0.95: best = (nn_, P)
        bits = (best[0] * 16 * math.log2(10) / best[1]) if best else 0.0
        line = f"[part2] {cond:<11} {mode:<5} h={h:<4} max table @95% = {best[0] if best else 0:>5} names x16 -> bits/param {bits:.3f}"
        print(line, flush=True); out.write(line + "\n"); out.flush()
        rows.append(dict(cond=cond, mode=mode, h=h, summary=True, max_names=best[0] if best else 0, bits_per_param=bits))
    return rows


if __name__ == "__main__":
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    out = open(f"{args.out}{tag}.txt", "w"); res = {"part1": [], "part2": []}
    t0 = time.time()
    if args.part in ("1", "all"):
        # the lr sweep, the same for every arm, on the in-distribution val at f=0.5 seed 0
        lrs = {}
        for arm in ARMS:
            sc = {lr: run_part1(arm, 0.5, 0, lr)["val"] for lr in ([3e-3] if args.smoke else [1e-3, 3e-3])}
            lrs[arm] = max(sc, key=sc.get)
            line = f"[part1] lr sweep {arm}: {sc} -> {lrs[arm]}"; print(line, flush=True); out.write(line + "\n")
        for f in COVER:
            for arm in ARMS:
                rs = [run_part1(arm, f, s, lrs[arm]) for s in SEEDS]
                agg = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in rs[0]}
                res["part1"].append(dict(arm=arm, f=f, lr=lrs[arm], seeds=rs, agg=agg))
                line = (f"[part1] f={f:<5} {arm:<3} params {rs[0]['params']:>7}  val {agg['val'][0]:.3f}  HELDOUT {agg['heldout'][0]:.3f}"
                        f"±{agg['heldout'][1]:.3f}  (hop1 {agg['heldout_hop1'][0]:.3f} hop2 {agg['heldout_hop2'][0]:.3f} "
                        f"hop3 {agg['heldout_hop3'][0]:.3f})  [{time.time()-t0:.0f}s]")
                print(line, flush=True); out.write(line + "\n"); out.flush()
    if args.part in ("2", "all"):
        res["part2"] = run_part2(out)
    json.dump(res, open(f"{args.out}{tag}.json", "w"), indent=1)
    print(f"[toy] DONE in {time.time()-t0:.0f}s -> {args.out}{tag}.txt/.json", flush=True)
