"""reread_toy.py -- T3e RE-READ vs ONE-SHOT (ledger 2026-10-26, word given). CPU. Bryce's alternation question: does the
reasoner gain from taking its CURRENT question back to the text every breath, vs being handed one package at handoff?
The MEGAZORD machine (T6 clock gate, tree read, snap) with only the RULE READ varied; the "text" is the in-context rule
sigma (NT facts t -> sigma(t)).
ARMS:
  REREAD  MEGAZORD's read: every breath the content heads attend over the NT facts with a query from the current type
  BUS     one-shot, fixed algebra: the reader binds the whole rule into ONE P-plane phasor vector (the T256 bus form),
          s = sum_t TC[t] * rho(TC[sigma(t)]) / sqrt(NT), rho a fixed plane permutation (the VALUE role); every breath the
          reasoner UNBINDS with its current type, rho^-1(conj(Tq) * s). Without rho the bind is commutative and the unbind
          returns sigma(t) and sigma^-1(t) alike (feasibility: one-hop 0.62 at any width; with rho 0.994 at P=64, 1.000 at 256)
  POOL    one-shot, learned (the vanilla handoff): two learned pooling queries per head summarize the facts once into a
          2D-wide vector (= BUS's width); every breath an MLP reads (current type, summary)
SPLITS: indist / HELD_RULE (the rule contains a held type pair t -> t' that an active hop crosses; no training rule
contains any held pair) / HELD_H / LONG / EDGE / SWAP (world B's tree). Trees: the banked T3b6 (NB 12) / T3d (NB 24)
stores, read-only."""
import json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import megazord_toy as M
from mycelium import tree_db as TDB

ARMS = os.environ.get("TE_ARMS", "REREAD,BUS,POOL").split(",")
SEEDS = [int(s) for s in os.environ.get("TM_SEEDS", "0,1,2,3,4").split(",")]
TREE_TAG = {12: "_t3b6", 24: "_t3d"}[M.NB]
OUT = f".cache/reread_t3e_nb{M.NB}{'_smoke' if M.SMOKE else ''}.txt"
HELD_PAIRS = {(1, 4), (3, 5), (5, 6), (7, 0)}                              # never in a training rule
HP = np.zeros((M.NT, M.NT), bool)
for a, b in HELD_PAIRS:
    HP[a, b] = True


def load_R(world):
    db = TDB.TreeDB(M.DBP.format(TREE_TAG, world)); rows = db.load_leaves(); db.close()
    return np.stack([rows[f"{t}:{i}"][1] for t in range(M.NT) for i in range(M.NI)])


def episodes(rng, n, hs, split, f):
    """train: no held rule pair anywhere in sigma, no held leaf sourced; held_rule: an ACTIVE hop crosses a held pair
    (no held leaf sourced); any: no filter (swap)."""
    import torch
    parts = []
    for h, nh in zip(hs, np.bincount(rng.integers(0, len(hs), n), minlength=len(hs))):
        got, acc = 0, []
        while got < nh:
            sig, t, i, _ = M.chains(rng, 8 * nh + 32, h, f); ar = np.arange(len(t))[:, None]
            leaf = M.HELD[t[:, :h], i[:, :h]].any(1)
            rule_any = HP[np.arange(M.NT)[None, :], sig].any(1)
            rule_act = HP[t[:, :h], sig[ar, t[:, :h]]].any(1)
            keep = {"train": ~leaf & ~rule_any, "held_rule": ~leaf & rule_act, "any": np.ones(len(t), bool)}[split]
            k = np.nonzero(keep)[0][: nh - got]; got += len(k); acc.append((sig[k], t[k], i[k], np.full(len(k), h)))
        parts += acc
    sig, t, i, hh = (np.concatenate(x) for x in zip(*parts))
    L = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long()
    return dict(sig=L(sig), t=L(t), i=L(i), h=L(hh))


def build_net():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    Base, _ = M.build_net()
    g = torch.Generator().manual_seed(2027)                                 # the SAME codes as megazord_toy.build_net
    TC = torch.polar(torch.ones(M.NT, M.P), 2 * math.pi * torch.rand(M.NT, M.P, generator=g))
    IC = torch.polar(torch.ones(M.NI, M.P), 2 * math.pi * torch.rand(M.NI, M.P, generator=g))
    ALL_T, ALL_I = torch.arange(M.NL) // M.NI, torch.arange(M.NL) % M.NI
    ri = lambda z: torch.cat([z.real, z.imag], -1)
    P, D, NH, DH, NT = M.P, M.D, M.NH, M.DH, M.NT

    def rot(z, th):
        c, s = torch.cos(th), torch.sin(th); re, im = z[..., 0::2], z[..., 1::2]
        return torch.stack([re * c - im * s, re * s + im * c], -1).flatten(-2)

    def st(lg_or_p, is_prob=False):
        soft = lg_or_p if is_prob else lg_or_p.softmax(-1)
        return F.one_hot(soft.argmax(-1), soft.shape[-1]).float() + soft - soft.detach()

    class Net(Base):
        def __init__(self, arm):
            super().__init__("MEGAZORD"); self.arm = arm
            if arm == "BUS":
                self.rb = nn.Linear(2 * P, 2 * P)
                gp = torch.Generator().manual_seed(7); pr = torch.randperm(P, generator=gp)
                self.register_buffer("rho", pr); self.register_buffer("rho_inv", pr.argsort())
            if arm == "POOL":
                self.pq = nn.Parameter(torch.randn(2, NH, DH) / math.sqrt(DH))
                self.pm = nn.Sequential(nn.Linear(2 * P + 2 * D, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, D))

        def forward(self, bt, R):
            Bn = bt["sig"].shape[0]; h = bt["h"]; tcn = F.normalize(ri(TC), dim=-1); wD = self.w.repeat(NH)
            ht = F.one_hot(bt["t"][:, 0], NT).float(); hi = F.one_hot(bt["i"][:, 0], M.NI).float()
            K = self.rk(ri(TC)).expand(Bn, NT, D).reshape(Bn, NT, NH, DH).transpose(1, 2)
            V = self.rv(ri(TC[bt["sig"]])).view(Bn, NT, NH, DH).transpose(1, 2)
            hopk = rot(self.ck.expand(Bn, D), wD[None] * h[:, None].float())
            fk = ri(TC[ALL_T] * IC[ALL_I]) / math.sqrt(P)
            if self.arm == "BUS":                                           # THE HANDOFF, once: the whole rule in one vector
                s = (TC[None] * TC[bt["sig"]][..., self.rho]).sum(1) / math.sqrt(NT)
            if self.arm == "POOL":                                          # THE HANDOFF, once: a learned summary
                a0 = torch.softmax(torch.einsum("qhd,bhld->bqhl", self.pq, K) / math.sqrt(DH), -1)
                s = torch.einsum("bqhl,bhld->bqhd", a0, V).reshape(Bn, 2 * D)
            lts, lis = [], []
            for b in range(M.NB):
                Tq = ht.to(torch.complex64) @ TC; Iq = hi.to(torch.complex64) @ IC
                p_go = torch.sigmoid((hopk @ rot(self.cq, wD * b))[:, None] * self.cs.exp() + self.cb)
                if self.arm == "REREAD":                                    # back to the text with the current question
                    Q = self.rq(ri(Tq)).view(Bn, NH, DH)
                    a = torch.softmax(torch.einsum("bhd,bhld->bhl", Q, K) / math.sqrt(DH), -1)
                    msg = self.ro(torch.einsum("bhl,bhld->bhd", a, V).reshape(Bn, D))
                elif self.arm == "BUS":
                    msg = self.rb(ri((Tq.conj() * s)[..., self.rho_inv]))
                else:
                    msg = self.ro(self.pm(torch.cat([ri(Tq), s], -1)))
                r = ri(Tq) + p_go * msg
                cos_t = F.normalize(r, dim=-1) @ tcn.T; lt = cos_t * self.ls.exp(); pt = lt.softmax(-1)
                g = 1 - (pt * ht).sum(-1, keepdim=True)
                dist = torch.softmax((ri(Tq * Iq) / math.sqrt(P)) @ fk.T * self.lk.exp(), -1) @ R
                pi = g * dist + (1 - g) * hi
                ht, hi = st(lt), st(pi / pi.sum(-1, keepdim=True), is_prob=True)
                lts.append(lt); lis.append(torch.log(pi + 1e-6))
            return torch.stack(lts, 1), torch.stack(lis, 1)

    return Net, F


def run(arm, seed, RA, RB):
    import torch
    Net, F = build_net()
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed); fA, fB = M.WORLDS["A"], M.WORLDS["B"]
    ev = {"indist": {h: episodes(erng, M.N_EV, [h], "train", fA) for h in M.TRAIN_H},
          "held_rule": {h: episodes(erng, M.N_EV, [h], "held_rule", fA) for h in M.TRAIN_H},
          "held_h": {h: episodes(erng, M.N_EV, [h], "train", fA) for h in M.HELD_H},
          "edge": {h: episodes(erng, M.N_EV, [h], "train", fA) for h in M.EDGE_H},
          "swap": {h: episodes(erng, M.N_EV, [h], "any", fB) for h in range(1, M.NB)}}
    torch.manual_seed(seed); net = Net(arm); tA, tB = torch.tensor(RA, dtype=torch.float32), torch.tensor(RB, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=M.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=M.LR, total_steps=M.STEPS, pct_start=0.1)
    for step in range(M.STEPS):
        bt = episodes(rng, M.BATCH, M.TRAIN_H, "train", fA); lt, li = net(bt, tA)
        loss = F.cross_entropy(lt.reshape(-1, M.NT), bt["t"][:, 1:].reshape(-1)) + F.nll_loss(li.reshape(-1, M.NI), bt["i"][:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); r = {"loss_end": float(loss.detach())}
    with torch.no_grad():
        for sp, byh in ev.items():
            accs = []
            for hh, bt in byh.items():
                lt, li = net(bt, tB if sp == "swap" else tA)
                ok = (lt[:, -1].argmax(-1) == bt["t"][:, -1]) & (li[:, -1].argmax(-1) == bt["i"][:, -1])
                r[f"{sp}_h{hh}"] = float(ok.float().mean()); accs.append(r[f"{sp}_h{hh}"])
            r[sp] = float(np.mean(accs))
    r["long"] = float(np.mean([r[f"indist_h{hh}"] for hh in M.LONG_H]))
    return r


if __name__ == "__main__":
    assert M.SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    out = open(OUT, "w"); t0 = time.time(); RA, RB = load_R("A"), load_R("B")

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"T3e pid {os.getpid()} NB {M.NB} steps {M.STEPS} seeds {SEEDS} arms {ARMS} clock {M.CLOCK} trees {TREE_TAG} held pairs {sorted(HELD_PAIRS)}")
    res = []
    for arm in ARMS:
        rs = [run(arm, s, RA, RB) for s in SEEDS]; res.append(dict(arm=arm, seeds=rs))
        for s, r in zip(SEEDS, rs):
            say(f"{arm:6s} s{s}: " + " ".join(f"{k} {r[k]:.3f}" for k in ("indist", "held_rule", "held_h", "long", "edge", "swap")) + f"  [{time.time() - t0:.0f}s]")
    json.dump(dict(nb=M.NB, arms=res), open(OUT.replace(".txt", ".json"), "w"), indent=1)
    say(f"DONE in {time.time() - t0:.0f}s")
