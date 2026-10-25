"""leaf_lookup_toy.py -- T2z, THE ID TAG AS A KEY: LEAF-SPECIFIC LOOKUP ON THE EXPAND STEP (2026-10-25; Bryce's word; ledger
"T2z"). CPU. T2v/T2x tested the easy expand (the parked tag IS the answer's address). Real facts are leaf-specific:
"Taylor's spouse" is a row about Taylor. Here every hop is (t, i) -> (sigma(t), f(t, i)): the TYPE moves by a category
rule (episode type facts, MHA, snapped -- the bottleneck), the INDEX moves by a LEAF fact keyed by the full source address
(t, i) (64 leaf facts per episode, f random per episode). The expand step must build a key from (type from the
bottleneck, tag from the skip) and look the row up. Two hops (breaths); per-breath trace supervision.
HELD-OUT LEAVES (t + i) % 4 == 0 are never a hop's SOURCE in training; the test has a held-out source on some hop.
Codes: contentless fixed random unit phasors (P planes). The key arms differ ONLY in how the address is composed:
  BIND_KEY     key = T_t * I_i (planewise product; fact keys composed the same way) -- fixed algebra
  ADD_KEY      key = T_t + I_i (fact keys the same) -- fixed, linear
  ATTN_LINEAR  q = Wq [T_t, I_i], k = Wk [T_f, I_f] -- a plain learned attention (linear maps, bilinear score)
  MLP_KEY      q = MLP([T_t, I_i]), k = MLP([T_f, I_f]) (one shared MLP) -- learned NONLINEAR composition
  MLP_SPLIT    q = MLPq([T_t, I_i]), k = MLPk([T_f, I_f]) (separate MLPs) -- AMENDMENT after the feasibility read: a SHARED
               composer on both sides of a lookup is a hash (self-match needs only injectivity), so it cannot memorize
The read: softmax over the 64 leaf facts -> the row's index code -> snap to the index codebook."""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = "--smoke" in sys.argv
ARMS = os.environ.get("TZ_ARMS", "BIND_KEY,ADD_KEY,ATTN_LINEAR,MLP_KEY,MLP_SPLIT").split(",")
NT, NI, P, D, NH, NB = 8, 8, 64, 64, 4, 2
NL, DH = NT * NI, D // NH
BATCH, LR = 128, 3e-3
STEPS = 300 if SMOKE else 6000
SEEDS = [0] if SMOKE else [0, 1, 2]
N_EVAL = 500 if SMOKE else 3000
OUT = ".cache/leaf_lookup{}.txt".format("_smoke" if SMOKE else "")
HELD = np.array([[(t + i) % 4 == 0 for i in range(NI)] for t in range(NT)])
_g = torch.Generator().manual_seed(2027)
TC = torch.polar(torch.ones(NT, P), 2 * math.pi * torch.rand(NT, P, generator=_g))
IC = torch.polar(torch.ones(NI, P), 2 * math.pi * torch.rand(NI, P, generator=_g))
ALL_T = torch.arange(NL) // NI; ALL_I = torch.arange(NL) % NI


def ri(z):
    return torch.cat([z.real, z.imag], -1)


def episodes(rng, n, split):
    out = []; got = 0
    while got < n:
        m = 4 * n
        sig = np.stack([rng.permutation(NT) for _ in range(m)]); f = rng.integers(0, NI, (m, NT, NI))
        t = [rng.integers(0, NT, m)]; i = [rng.integers(0, NI, m)]; ar = np.arange(m)
        for _ in range(NB):
            t.append(sig[ar, t[-1]]); i.append(f[ar, t[-2], i[-1]])
        t, i = np.stack(t, 1), np.stack(i, 1)                                # (m, NB+1)
        src_held = HELD[t[:, :NB], i[:, :NB]]
        keep = ~src_held.any(1) if split == "train" else src_held.any(1)
        k = np.nonzero(keep)[0][: n - got]; got += len(k)
        out.append((sig[k], f[k], t[k], i[k]))
    sig, f, t, i = (np.concatenate(x) for x in zip(*out))
    L = lambda a: torch.from_numpy(a).long()
    return dict(sig=L(sig), f=L(f.reshape(len(f), NL)), t=L(t), i=L(i))


class Net(nn.Module):
    def __init__(self, arm):
        super().__init__(); self.arm = arm
        self.rq, self.rk, self.rv = nn.Linear(2 * P, D), nn.Linear(2 * P, D), nn.Linear(2 * P, D)   # rule step (shared form)
        self.ro = nn.Linear(D, 2 * P)
        self.ls = nn.Parameter(torch.tensor(math.log(10.0))); self.lk = nn.Parameter(torch.tensor(math.log(10.0)))
        if arm == "ATTN_LINEAR":
            self.wq, self.wk = nn.Linear(4 * P, D), nn.Linear(4 * P, D)
        if arm in ("MLP_KEY", "MLP_SPLIT"):
            self.mk = nn.Sequential(nn.Linear(4 * P, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D))
        if arm == "MLP_SPLIT":
            self.mq = nn.Sequential(nn.Linear(4 * P, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D))

    def key(self, t, i):
        T, I = TC[t], IC[i]
        if self.arm == "BIND_KEY":
            return ri(T * I) / math.sqrt(P)
        if self.arm == "ADD_KEY":
            return ri(T + I) / math.sqrt(2 * P)
        x = torch.cat([ri(T), ri(I)], -1)
        return self.mk(x) if self.arm in ("MLP_KEY", "MLP_SPLIT") else x

    @staticmethod
    def st(lg, C):
        soft = lg.softmax(-1); hard = F.one_hot(lg.argmax(-1), C.shape[0]).float()
        return hard + soft - soft.detach()

    def forward(self, bt):
        Bn = bt["sig"].shape[0]; t, i = bt["t"][:, 0], bt["i"][:, 0]
        ht = F.one_hot(t, NT).float(); hi = F.one_hot(i, NI).float()
        fk = self.key(ALL_T, ALL_I)                                         # (NL, d): the leaf facts' composed addresses
        if self.arm == "ATTN_LINEAR":
            fk = self.wk(fk)
        fv = F.one_hot(bt["f"], NI).float()                                 # (B, NL, NI): each row's index
        res = TC[bt["sig"]]; lts, lis = [], []
        for b in range(NB):
            Tq = torch.einsum("bn,np->bp", ht.to(torch.complex64), TC)
            Iq = torch.einsum("bn,np->bp", hi.to(torch.complex64), IC)
            # bottleneck: the category rule over the type facts (learned attention), snapped to the type codebook
            Q = self.rq(ri(Tq)).view(Bn, NH, DH); K = self.rk(ri(TC).expand(Bn, NT, 2 * P)).view(Bn, NT, NH, DH).transpose(1, 2)
            V = self.rv(ri(res)).view(Bn, NT, NH, DH).transpose(1, 2)    # fact t -> sigma(t): key the source, read the result
            a = torch.softmax(torch.einsum("bhd,bhld->bhl", Q, K) / math.sqrt(DH), -1)
            r = self.ro(torch.einsum("bhl,bhld->bhd", a, V).reshape(Bn, D))
            lt = F.normalize(r, dim=-1) @ F.normalize(ri(TC), dim=-1).T * self.ls.exp()
            # expand: the source address (type, tag) composed into a key, the leaf row looked up, snapped to the tag codebook
            if self.arm in ("BIND_KEY", "ADD_KEY"):
                q = ri(Tq * Iq) / math.sqrt(P) if self.arm == "BIND_KEY" else ri(Tq + Iq) / math.sqrt(2 * P)
            else:
                x = torch.cat([ri(Tq), ri(Iq)], -1)
                q = self.mk(x) if self.arm == "MLP_KEY" else self.mq(x) if self.arm == "MLP_SPLIT" else self.wq(x)
            sc = q @ fk.T if self.arm in ("BIND_KEY", "ADD_KEY") else F.normalize(q, dim=-1) @ F.normalize(fk, dim=-1).T
            w = torch.softmax(sc * self.lk.exp(), -1)
            li = torch.log(torch.einsum("bl,bln->bn", w, fv) + 1e-6)
            ht, hi = self.st(lt, TC), self.st(li, IC)
            lts.append(lt); lis.append(li)
        return torch.stack(lts, 1), torch.stack(lis, 1)


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    ev = {"train": episodes(erng, N_EVAL, "train"), "held": episodes(erng, N_EVAL, "held")}
    torch.manual_seed(seed); net = Net(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = episodes(rng, BATCH, "train"); lt, li = net(bt)
        loss = F.cross_entropy(lt.reshape(-1, NT), bt["t"][:, 1:].reshape(-1)) + F.nll_loss(li.reshape(-1, NI), bt["i"][:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); r = {"loss_end": float(loss.detach())}
    with torch.no_grad():
        for sp, bt in ev.items():
            lt, li = net(bt)
            okt = (lt.argmax(-1) == bt["t"][:, 1:]).all(-1); oki = (li.argmax(-1) == bt["i"][:, 1:]).all(-1)
            r[f"{sp}_acc"] = float((okt & oki).float().mean()); r[f"{sp}_type"] = float(okt.float().mean()); r[f"{sp}_index"] = float(oki.float().mean())
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    assert SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    e = episodes(np.random.default_rng(0), 300, "train")
    assert not HELD[e["t"][:, :NB].numpy(), e["i"][:, :NB].numpy()].any(), "a held-out source leaked into training"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {'smoke' if SMOKE else 'full'} steps {STEPS} seeds {SEEDS} arms {ARMS} tree {NT}x{NI} hops {NB} P {P}")
    for arm in ARMS:
        rs = [run(arm, s) for s in SEEDS]
        ag = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}; A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        say(f"{arm:12s} in-dist {ag['train_acc']:.3f} | HELD-OUT SOURCE {ag['held_acc']:.3f} (seeds {[round(x['held_acc'], 3) for x in rs]}) "
            f"type {ag['held_type']:.3f} index {ag['held_index']:.3f}  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if not SMOKE and all(a in A for a in ("BIND_KEY", "ATTN_LINEAR", "MLP_KEY")):   # PINNED (ledger 2026-10-25, T2z)
        bk = A["BIND_KEY"]; sd = lambda a: [x["held_acc"] for x in allres[ARMS.index(a)]["seeds"]]
        say("VERDICTS (3-seed means)")
        say(f"CONTROL: BIND_KEY in-dist {bk['train_acc']:.3f} -> {'OK' if bk['train_acc'] >= 0.95 else 'VOID'}")
        say(f"Z1 THE TAG AS A KEY GENERALIZES (fixed binding): {'PASS' if bk['held_acc'] >= 0.95 else 'FAIL'} (BIND_KEY held-out {bk['held_acc']:.3f})")
        for nm, z in (("ATTN_LINEAR", "Z2 FIXED ALGEBRA BEATS LINEAR LEARNED ATTENTION"), ("MLP_KEY", "Z3 FIXED ALGEBRA BEATS NONLINEAR LEARNED COMPOSITION"),
                      ("MLP_SPLIT", "Z4 (amendment) FIXED ALGEBRA BEATS SPLIT NONLINEAR COMPOSERS")):
            if nm not in A:
                continue
            d = [a - b for a, b in zip(sd("BIND_KEY"), sd(nm))]
            ok = bk["held_acc"] - A[nm]["held_acc"] >= 0.10 and all(x > 0 for x in d)
            say(f"{z}: {'PASS' if ok else 'FAIL'} (BIND_KEY - {nm} held-out {bk['held_acc'] - A[nm]['held_acc']:+.3f}; per seed {[round(x, 3) for x in d]}; "
                f"{nm} in-dist {A[nm]['train_acc']:.3f})")
        if "ADD_KEY" in A:
            say(f"READ ADD_KEY: in-dist {A['ADD_KEY']['train_acc']:.3f} held-out {A['ADD_KEY']['held_acc']:.3f}")
    say(f"DONE in {time.time() - t0:.0f}s")
