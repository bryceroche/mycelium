"""binding_toy_v2.py -- THE SCALING TOY (2026-10-15; Bryce's word; the merged Opus/Fable design, guards pinned). CPU only,
torch, never the card, never the repo's head. Self-contained.

THE THESIS (Bryce): "Give the network reusable concepts, reliable relationships, and operations so it can construct
solutions to problems it has never seen." Framed as INDUCTIVE BIAS, not capability: a transformer CAN learn binding; the
claim is that explicit structure (registers) and a binding operator (fixed phases / learned tensor product) make it CHEAP
-- the samples needed for unseen combinations grow slowly in K instead of geometrically.

THE TASK (transfer chains, roles from SYNTAX). K entities ("X has c ." -- c is noise); each GIVES once and RECEIVES once (a
random derangement), K transfer events in four surface forms where the SAME verb token flips roles by syntax (guard 1):
  F0  G gave R m .            F1  R was gave m by G .       F2  R got m from G .       F3  m was gave to R by G .
Query "X gave|got d ?": follow the giving / receiving chain d hops (d in 1..3) -> a NAME. OPERATIONS = CHAINED RELATIONAL
LOOKUPS (the parser's job; arithmetic is the solver's -- neural proposes, symbolic disposes). v2's first form asked mod-10
sums and its feasibility read (K=2, N=32k, 6000 steps) found in-dist val 0.45 for T and PHR alike, d3 ~0.2: arithmetic, not
binding, was the bottleneck, every cell would have been censored -- replaced before the full run.
THE UNSEEN COMBINATIONS: per name, 3 of its 8 (role, form) cells are HELD OUT of all training (each cell is trained with other
names: every concept, relation and form is seen -- only the combination is new). TEST: both events touching the queried
entity use its held-out cells. An in-distribution val is reported beside it.

ARMS (guard 2: each pair differs in exactly one thing). Shared encoder: token+position embeddings, 2 pre-norm layers (4-head
attention, SwiGLU x1), d=64.
  T     the encoder + 1 more layer, SwiGLU x4, readout at the query token        (the baseline transformer)
  PH    the encoder with PHASE-BOUND q/k (fixed random phasors, learned role choice; v1's arm B), readout at the query
  SLOT  encoder + REGISTERS: one register per entity (keyed by its intro name token -- structural, shared by every register
        arm); each token writes w_t into registers by learned soft addressing; the query reads its register; decoder MLP.
        WRITE OPERATOR: w_t = c_t (no binding)
  TPR   the same registers; w_t = c_t (16-d) (x) sum_k rho_tk r_k (4-d, LEARNED role vectors) = learned linear binding
  PHR   the same registers; w_t = c_t (64-d) (*) unit(sum_k rho_tk P_k), P_k FIXED random phasors over 32 planes
  (SLOT/TPR/PHR: identical encoder, addressing, register count, decoder; only the write operator differs.)
METRIC (guard 3/4): for each K in {2,4,6,8} and train-set size N in {2k, 8k, 32k, 128k} (nested subsets of one pool per
(K, seed), identical for every arm), train STEPS steps (batch 128, the same for every arm and N; lr from a matched sweep);
SAMPLES-TO-95% = the smallest N whose held-out accuracy (seed mean) >= 0.95; never reached = CENSORED (">128k"), never a
number. Curves per depth d reported beside. The knowledge control is v1's Part 2 (binding_toy.py), not repeated here.
PREDICTIONS (pinned before the first run):
  Opus: T's samples-to-95% grows geometrically in K; PHR polynomially; PHR beats TPR and SLOT (the thesis at full strength).
  Fable: STRUCTURE is the main effect -- the register arms (SLOT/TPR/PHR) all beat T and PH on held-out, flatter in K; the
         binding operator is second-order (PHR ~ TPR, within 2x samples), SLOT behind TPR/PHR (roles must be split to follow
         a chain in the right direction); PH ~ T; every arm's held-out falls with depth d.
READING: PHR < TPR, SLOT (fewer samples) -> phases matter; PHR ~ TPR > SLOT -> binding matters, phases are one form of it;
  SLOT ~ TPR ~ PHR > T -> registers (the notebook) carry it; nothing beats T -> no structural tax at this scale (scoped null).
Usage: .venv/bin/python3 scripts/toys/binding_toy_v2.py [--smoke] [--threads N]  -> .cache/binding_toy_v2{_smoke}.{txt,json}
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default=".cache/binding_toy_v2")
ap.add_argument("--role_sup", type=int, default=1)
args = ap.parse_args()
torch.set_num_threads(args.threads)
ROLE_SUP = args.role_sup

N_NAMES = 40
NUM0 = N_NAMES                                     # number tokens 0..9
W = {w: NUM0 + 10 + i for i, w in enumerate(["has", "gave", "was", "by", "got", "from", "to", "how", "many",
                                              "received", ".", "?", "PAD"])}
VOCAB = NUM0 + 10 + len(W)
D, H_HEADS, MAXLEN = 64, 4, 96
KS = [2, 4] if args.smoke else [2, 4, 6, 8]
NS = [2000, 8000] if args.smoke else [2000, 8000, 32000, 128000]
SEEDS = [0] if args.smoke else [0, 1]
STEPS = 300 if args.smoke else 6000
BATCH = 128
ARMS = ["T", "PH", "SLOT", "TPR", "PHR"]
QTYPES = ["gave", "got"]
N_HELD = 3                                         # held-out (role, form) cells per name, of 8


def cells_for(seed):
    """held[name] = set of (role, form) cells never trained for that name; every cell kept trained for some name."""
    rng = np.random.default_rng(500 + seed)
    held = {}
    for n in range(N_NAMES):
        cs = [(r, f) for r in (0, 1) for f in range(4)]
        held[n] = set(map(tuple, np.array(cs)[rng.choice(8, N_HELD, replace=False)].tolist()))
    return held


def event_tokens(form, g, r, m):
    G, R, M = g, r, NUM0 + m
    if form == 0: return [G, W["gave"], R, M, W["."]]
    if form == 1: return [R, W["was"], W["gave"], M, W["by"], G, W["."]]
    if form == 2: return [R, W["got"], M, W["from"], G, W["."]]
    return [M, W["was"], W["gave"], W["to"], R, W["by"], G, W["."]]


def make_scene(K, held, test, rng):
    """K entities, each GIVES once and RECEIVES once: a random derangement pi (entity i gives m to pi(i)), K events in the
    four surface forms. Query [X, gave|got, d, ?]: follow the giving (gave: X -> pi^d(X)) or the receiving (got: pi^-d(X))
    chain d hops (d in 1..min(3, K-1)); the answer is a NAME. TEST: both events touching the queried entity use one of ITS
    held-out (role, form) cells (the partner's cell allowed); every other event uses allowed cells.
    returns tokens, target (name id), depth, qtype (0 gave / 1 got), intro positions"""
    while True:
        names = rng.choice(N_NAMES, K, replace=False)
        counts = rng.integers(10, size=K)
        pi = rng.permutation(K)
        while (pi == np.arange(K)).any(): pi = rng.permutation(K)
        inv = np.argsort(pi)
        qi = int(rng.integers(K)); qt = int(rng.integers(2)); d = int(rng.integers(1, min(3, K - 1) + 1))
        evs = []; ok = True
        for g in range(K):
            r = int(pi[g])
            if g == qi or r == qi:
                role = 0 if g == qi else 1; other = r if g == qi else g
                qcells = [f for f in range(4) if ((role, f) in held[names[qi]]) == test]
                fs = [f for f in qcells if (1 - role, f) not in held[names[other]]]
            else:
                fs = [f for f in range(4) if (0, f) not in held[names[g]] and (1, f) not in held[names[r]]]
            if not fs: ok = False; break
            evs.append((int(rng.choice(fs)), g, r, int(rng.integers(1, 6))))
        if not ok: continue
        evs = [evs[i] for i in rng.permutation(len(evs))]
        toks, intro = [], []
        for i in range(K):
            intro.append(len(toks)); toks += [int(names[i]), W["has"], NUM0 + int(counts[i]), W["."]]
        rl = [-100] * len(toks)
        for f, g, r, m in evs:
            et = event_tokens(f, int(names[g]), int(names[r]), m)
            for t_ in et:
                rl.append(0 if t_ == int(names[g]) else 1 if t_ == int(names[r]) else -100)
            toks += et
        e = qi
        for _ in range(d): e = int(pi[e]) if qt == 0 else int(inv[e])
        toks += [int(names[qi]), W["gave"] if qt == 0 else W["got"], NUM0 + d, W["?"]]
        rl += [-100] * 4
        return toks, int(names[e]), d, qt, intro, rl


def make_pool(K, n, held, test, seed):
    rng = np.random.default_rng(seed)
    X = np.full((n, MAXLEN), W["PAD"], np.int64); L = np.zeros(n, np.int64); Y = np.zeros(n, np.int64)
    Dp = np.zeros(n, np.int64); Q = np.zeros(n, np.int64); I = np.zeros((n, 8), np.int64)
    RL = np.full((n, MAXLEN), -100, np.int64)
    for i in range(n):
        t, y, d, qt, intro, rl = make_scene(K, held, test, rng)
        assert len(t) <= MAXLEN
        X[i, :len(t)] = t; L[i] = len(t); Y[i] = y; Dp[i] = d; Q[i] = qt; I[i, :K] = intro; RL[i, :len(t)] = rl
    return [torch.from_numpy(a) for a in (X, L, Y, Dp, Q, I, RL)]


# ------------------------------------------------------------------ model
def bind(x, ph):
    m = x.shape[-1] // 2
    z = torch.complex(x[..., :m], x[..., m:]) * ph
    return torch.cat([z.real, z.imag], -1)


class Block(nn.Module):
    def __init__(self, ratio=1, phase=False, gen=None):
        super().__init__()
        self.phase = phase; hd = D // H_HEADS; self.hd = hd
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(D, D) for _ in range(4))
        if phase:
            self.gq, self.gk = nn.Linear(D, H_HEADS * 16), nn.Linear(D, H_HEADS * 16)
            th = torch.rand(H_HEADS, 16, hd // 2, generator=gen) * 2 * math.pi
            self.register_buffer("P", torch.polar(torch.ones_like(th), th))
        self.w1, self.w3, self.w2 = nn.Linear(D, ratio * D), nn.Linear(D, ratio * D), nn.Linear(ratio * D, D)

    def proj(self, h, w, g):
        B_, T_, _ = h.shape
        x = w(h).view(B_, T_, H_HEADS, self.hd)
        if self.phase:
            r = F.softmax(g(h).view(B_, T_, H_HEADS, 16), -1).to(torch.complex64)
            ph = torch.einsum("bthr,hrm->bthm", r, self.P); ph = ph / (ph.abs() + 1e-6)
            x = bind(x, ph)
        return x.transpose(1, 2)

    def forward(self, x, keymask):
        h = self.ln1(x); B_, T_, _ = h.shape
        q = self.proj(h, self.wq, getattr(self, "gq", None)); k = self.proj(h, self.wk, getattr(self, "gk", None))
        v = self.wv(h).view(B_, T_, H_HEADS, self.hd).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.hd) + keymask[:, None, None, :]
        x = x + self.wo((F.softmax(s, -1) @ v).transpose(1, 2).reshape(B_, T_, D))
        h = self.ln2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Net(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        self.arm = arm; gen = torch.Generator().manual_seed(1000 + seed)
        self.tok = nn.Embedding(VOCAB, D); self.pos = nn.Embedding(MAXLEN, D)
        self.enc = nn.ModuleList([Block(1, phase=(arm == "PH"), gen=gen) for _ in range(2)])
        if arm == "T": self.enc.append(Block(4))
        self.lnf = nn.LayerNorm(D)
        self.tag = nn.Linear(D, 2)            # THE PRIMITIVE SUPERVISION (every arm): giver/receiver per event name token
        if arm in ("SLOT", "TPR", "PHR"):
            self.aq, self.ak = nn.Linear(D, D), nn.Linear(D, D)          # soft write/read addressing over registers
            self.rq = nn.Linear(D, D)
            if arm == "TPR":
                self.c = nn.Linear(D, 16); self.rho = nn.Linear(D, 4); self.roles = nn.Parameter(torch.randn(4, 4) * 0.5)
            else:
                self.c = nn.Linear(D, D)
            if arm == "PHR":
                self.rho = nn.Linear(D, 4)
                th = torch.rand(4, D // 2, generator=gen) * 2 * math.pi
                self.register_buffer("P", torch.polar(torch.ones_like(th), th))   # FIXED, never trained
            self.dec = nn.Sequential(nn.Linear(2 * D, 2 * D), nn.SiLU(), nn.Linear(2 * D, N_NAMES))
        else:
            self.dec = nn.Linear(D, N_NAMES)

    def forward(self, X, L, I, K):
        B_, T_ = X.shape
        ar = torch.arange(T_)
        keymask = torch.where(ar[None] < L[:, None], 0.0, -1e9)
        x = self.tok(X) + self.pos(ar)[None]
        for b in self.enc: x = b(x, keymask)
        h = self.lnf(x); self._h = h
        hq = h[torch.arange(B_), L - 4]                                   # the queried NAME token's state
        if self.arm in ("T", "PH"):
            return self.dec(h[torch.arange(B_), L - 1])
        keys = self.ak(h[torch.arange(B_)[:, None], I[:, :K]])            # (B, K, D) register keys from intro names
        a = F.softmax(torch.einsum("btd,bkd->btk", self.aq(h), keys) / math.sqrt(D), -1)
        a = a * (ar[None] < L[:, None]).float()[..., None]
        if self.arm == "SLOT":
            w = self.c(h)
        elif self.arm == "TPR":
            role = F.softmax(self.rho(h), -1) @ self.roles                  # (B, T, 4) learned role vectors
            w = torch.einsum("bti,btj->btij", self.c(h), role).reshape(B_, T_, 64)
        else:
            ph = (F.softmax(self.rho(h), -1).to(torch.complex64) @ self.P)
            w = bind(self.c(h), ph / (ph.abs() + 1e-6))
        M = torch.einsum("btk,btd->bkd", a, w)                            # registers: per-entity sums of bound writes
        al = F.softmax(torch.einsum("bd,bkd->bk", self.rq(hq), keys) / math.sqrt(D), -1)
        m = torch.einsum("bk,bkd->bd", al, M)
        return self.dec(torch.cat([m, h[torch.arange(B_), L - 1]], -1))


def nparams(m): return sum(p.numel() for p in m.parameters())


def train_eval(arm, K, N, seed, lr, pools):
    tr, ev = pools
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(arm, seed); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.1)
    X, L, Y, Dp, Q, I, RL = tr
    for _ in range(STEPS):
        ix = torch.from_numpy(rng.integers(N, size=BATCH))
        Lb = L[ix]; T_ = int(Lb.max())
        loss = F.cross_entropy(net(X[ix, :T_], Lb, I[ix], K), Y[ix])
        if ROLE_SUP:
            loss = loss + F.cross_entropy(net.tag(net._h).reshape(-1, 2), RL[ix, :T_].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
    net.eval(); out = {"params": nparams(net)}
    with torch.no_grad():
        for split, (X, L, Y, Dp, Q, I, RL) in ev.items():
            pr = torch.cat([net(X[s:s + 500, :int(L[s:s + 500].max())], L[s:s + 500], I[s:s + 500], K).argmax(-1)
                            for s in range(0, len(Y), 500)])
            ok = (pr == Y).float()
            out[split] = float(ok.mean())
            for d in (1, 2, 3): out[f"{split}_d{d}"] = float(ok[Dp == d].mean())
            for qi, qn in enumerate(QTYPES): out[f"{split}_{qn}"] = float(ok[Q == qi].mean())
    return out


if __name__ == "__main__":
    tag = "_smoke" if args.smoke else ""
    fo = open(f"{args.out}{tag}.txt", "w"); res = []; t0 = time.time()
    def P(s): print(s, flush=True); fo.write(s + "\n"); fo.flush()
    for seed in SEEDS:
        held = cells_for(seed)
        for K in KS:
            n_eval = 600 if args.smoke else 3000
            tr = make_pool(K, max(NS), held, False, 10 * seed + K)
            ev = {"val": make_pool(K, n_eval, held, False, 7000 + 10 * seed + K),
                  "heldout": make_pool(K, n_eval, held, True, 9000 + 10 * seed + K)}
            P(f"[pool] seed {seed} K={K}: train {max(NS)}, mean len {float(tr[1].float().mean()):.1f} [{time.time()-t0:.0f}s]")
            res.append(dict(seed=seed, K=K, pools_built=True))
            for arm in ARMS:
                # the matched lr sweep: the same two lrs for every arm, picked on in-dist VAL at the largest N (never held-out)
                if seed == SEEDS[0] and K == KS[0] and not args.smoke:
                    sc = {lr: train_eval(arm, K, max(NS), seed, lr, (tr, ev))["val"] for lr in (1e-3, 3e-3)}
                    LR = res.append(dict(arm=arm, lr_sweep=sc)) or max(sc, key=sc.get)
                    globals().setdefault("LRS", {})[arm] = LR; P(f"[lr] {arm}: {sc} -> {LR}")
                lr = globals().get("LRS", {}).get(arm, 3e-3)
                for N in NS:
                    r = train_eval(arm, K, N, seed, lr, (tr, ev))
                    res.append(dict(arm=arm, K=K, N=N, seed=seed, lr=lr, **r))
                    P(f"[run] seed {seed} K={K} N={N:>6} {arm:<4} params {r['params']:>7} val {r['val']:.3f} HELDOUT {r['heldout']:.3f} "
                      f"(d1 {r['heldout_d1']:.2f} d2 {r['heldout_d2']:.2f} d3 {r['heldout_d3']:.2f} | gave {r['heldout_gave']:.2f} got {r['heldout_got']:.2f}) [{time.time()-t0:.0f}s]")
                    json.dump(res, open(f"{args.out}{tag}.json", "w"))
    # SAMPLES-TO-95% (seed mean), censored when never reached
    P("\n== SAMPLES-TO-95% on HELD-OUT combinations (seed mean; '>' = censored) ==")
    P("K    " + "  ".join(f"{a:>8}" for a in ARMS))
    for K in KS:
        row = []
        for arm in ARMS:
            hit = None
            for N in NS:
                v = [r["heldout"] for r in res if r.get("arm") == arm and r.get("K") == K and r.get("N") == N]
                if v and np.mean(v) >= 0.95: hit = N; break
            row.append(f"{hit:>8}" if hit else f"{'>' + str(max(NS)):>8}")
        P(f"{K:<4} " + "  ".join(row))
    P(f"[toy2] DONE in {time.time()-t0:.0f}s")
