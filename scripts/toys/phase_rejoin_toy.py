"""phase_rejoin_toy.py -- T2x, THE NAZARE REJOIN: RULE AND ID TAG REJOIN IN PHASE (2026-10-25; Bryce's word; ledger "T2x"). CPU.
Bryce: "the logic rule traveling through the bottleneck and the ID tags through the skip connection need to rejoin in
phase. if they align they stack into a massive signal otherwise cancel out?" Phasor binding: bind(T, I) = T * I
(planewise complex product). Unbinding with the RIGHT type, z * conj(T), lines every plane of that pair's index code up
in phase (coherent sum ~ P); the other pair's planes arrive with random phases (incoherent sum ~ sqrt(P)).
WHERE IT MATTERS: when the state holds SEVERAL (type, id) pairs in ONE vector (the binding problem). Additive
superposition T1 + I1 + T2 + I2 is symmetric in the pairing -- no reader can tell (t1,i1),(t2,i2) from (t1,i2),(t2,i1).
TASK: two entities (t1, i1), (t2, i2) (distinct types, unordered) held as ONE state vector z. Episode facts are TYPE-level
(a permutation sigma). Query: a type u = sigma(t_q). The RULE step (category level, shared by every arm): MHA over the
type facts recovers the source type t_q, snapped hard to the type codebook. The EXPAND step rejoins it with z to read
the index i_q. Contentless ID codes: fixed random unit phasors, P planes.
HELD-OUT PAIRS (t + i) % 4 == 0 never occur in training (either entity); the test queries a held-out pair.
Arms:
  ADD          z = sum_e (T_e + I_e); learned MLP reader on [z, T_hat] (control: structurally ambiguous, <= ~0.53)
  BIND         z = sum_e T_e * I_e; expand = z * conj(T_hat) scored against the index codes (the Nazare rejoin)
  BIND_MLP     bound z, but a learned MLP reader on [z, T_hat] (read: does a learned reader exploit the binding?)
  LEARNED      z = sum_e MLP([T_e, I_e]) (DeepSets: learned binding); learned MLP reader (the cross-join learner)"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = "--smoke" in sys.argv
ARMS = os.environ.get("TX_ARMS", "ADD,BIND,BIND_MLP,LEARNED").split(",")
NT, NI, P, D, NH = 8, 8, 64, 64, 4
DH = D // NH
BATCH, LR = 128, 3e-3
STEPS = 300 if SMOKE else 6000
SEEDS = [0] if SMOKE else [0, 1, 2]
N_EVAL = 500 if SMOKE else 4000
OUT = ".cache/phase_rejoin{}.txt".format("_smoke" if SMOKE else "")
HELD = np.array([[(t + i) % 4 == 0 for i in range(NI)] for t in range(NT)])
_g = torch.Generator().manual_seed(2026)
TC = torch.polar(torch.ones(NT, P), 2 * math.pi * torch.rand(NT, P, generator=_g))      # contentless type codes
IC = torch.polar(torch.ones(NI, P), 2 * math.pi * torch.rand(NI, P, generator=_g))      # contentless ID tags


def ri(z):
    return torch.cat([z.real, z.imag], -1)


def episodes(rng, n, split):
    out = []; got = 0
    while got < n:
        m = 4 * n
        sig = np.stack([rng.permutation(NT) for _ in range(m)])
        t = np.stack([rng.choice(NT, 2, replace=False) for _ in range(m)]); i = rng.integers(0, NI, (m, 2))
        q = rng.integers(0, 2, m); ar = np.arange(m); h = HELD[t, i]
        keep = ~h.any(1) if split == "train" else h[ar, q]
        k = np.nonzero(keep)[0][: n - got]; got += len(k)
        out.append((sig[k], t[k], i[k], q[k]))
    sig, t, i, q = (np.concatenate(x) for x in zip(*out)); ar = np.arange(len(q))
    T_ = lambda a: torch.from_numpy(a).long()
    return dict(sig=T_(sig), t=T_(t), i=T_(i), u=T_(sig[ar, t[ar, q]]), tq=T_(t[ar, q]), y=T_(i[ar, q]))


class Net(nn.Module):
    def __init__(self, arm):
        super().__init__(); self.arm = arm
        self.q, self.k, self.v = nn.Linear(2 * P, D), nn.Linear(2 * P, D), nn.Linear(2 * P, D)    # the rule step (shared form)
        self.o = nn.Linear(D, 2 * P)
        if arm == "LEARNED":
            self.enc = nn.Sequential(nn.Linear(4 * P, 256), nn.GELU(), nn.Linear(256, 2 * P))
        if arm != "BIND":
            self.rd = nn.Sequential(nn.Linear(4 * P, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, NI))
        self.ls = nn.Parameter(torch.tensor(math.log(10.0)))

    def state(self, bt):
        T, I = TC[bt["t"]], IC[bt["i"]]                                     # (B, 2, P) complex
        if self.arm == "ADD":
            return ri((T + I).sum(1))
        if self.arm in ("BIND", "BIND_MLP"):
            return (T * I).sum(1)
        return self.enc(torch.cat([ri(T), ri(I)], -1)).sum(1)

    def forward(self, bt):
        Bn = bt["u"].shape[0]; z = self.state(bt)
        src = TC.expand(Bn, NT, P); res = TC[bt["sig"]]                     # fact t -> sigma(t)
        Q = self.q(ri(TC[bt["u"]])).view(Bn, NH, DH)
        K = self.k(ri(res)).view(Bn, NT, NH, DH).transpose(1, 2); V = self.v(ri(src)).view(Bn, NT, NH, DH).transpose(1, 2)
        a = torch.softmax(torch.einsum("bhd,bhld->bhl", Q, K) / math.sqrt(DH), -1)
        r = self.o(torch.einsum("bhl,bhld->bhd", a, V).reshape(Bn, D))
        lt = F.normalize(r, dim=-1) @ F.normalize(ri(TC), dim=-1).T * self.ls.exp()              # snap to the type codebook
        soft = lt.softmax(-1); hard = F.one_hot(lt.argmax(-1), NT).float()
        That = torch.einsum("bn,np->bp", (hard + soft - soft.detach()).to(torch.complex64), TC)
        if self.arm == "BIND":
            u = z * That.conj()                                             # the rejoin: right type -> that pair's tag in phase
            li = (u[:, None] * IC.conj()[None]).real.sum(-1) / P * self.ls.exp()
        else:
            li = self.rd(torch.cat([z if not torch.is_complex(z) else ri(z), ri(That)], -1))
        return lt, li


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    ev = {"train": episodes(erng, N_EVAL, "train"), "held": episodes(erng, N_EVAL, "held")}
    torch.manual_seed(seed); net = Net(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = episodes(rng, BATCH, "train"); lt, li = net(bt)
        loss = F.cross_entropy(li, bt["y"]) + F.cross_entropy(lt, bt["tq"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); r = {"loss_end": float(loss.detach())}
    with torch.no_grad():
        for sp, bt in ev.items():
            lt, li = net(bt); diff = bt["i"][:, 0] != bt["i"][:, 1]
            ok = li.argmax(-1) == bt["y"]
            r[f"{sp}_acc"] = float(ok.float().mean()); r[f"{sp}_acc_diff"] = float(ok[diff].float().mean())
            r[f"{sp}_rule"] = float((lt.argmax(-1) == bt["tq"]).float().mean())
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    assert SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    e = episodes(np.random.default_rng(0), 300, "train"); assert not HELD[e["t"].numpy(), e["i"].numpy()].any(), "held pair leaked"
    # the physics: right-type unbinding is coherent (~1), wrong-type / other-pair crosstalk incoherent (~1/sqrt(P))
    z = TC[0] * IC[3] + TC[5] * IC[6]
    c = lambda t, i: float(((z * TC[t].conj()) * IC[i].conj()).real.mean())
    coh, cross = c(0, 3), max(abs(c(0, 6)), abs(c(5, 3)), abs(c(0, 1)))
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {'smoke' if SMOKE else 'full'} steps {STEPS} seeds {SEEDS} arms {ARMS} P {P} held pairs {int(HELD.sum())}/64")
    say(f"PHYSICS: aligned rejoin {coh:.3f}; worst misaligned {cross:.3f} (1/sqrt(P) = {1 / math.sqrt(P):.3f})")
    for arm in ARMS:
        rs = [run(arm, s) for s in SEEDS]
        ag = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}; A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        say(f"{arm:9s} in-dist {ag['train_acc']:.3f} (diff-index {ag['train_acc_diff']:.3f}) | HELD-OUT PAIR {ag['held_acc']:.3f} "
            f"(seeds {[round(x['held_acc'], 3) for x in rs]}; diff-index {ag['held_acc_diff']:.3f}) | rule {ag['held_rule']:.3f}  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if not SMOKE and all(a in A for a in ("ADD", "BIND", "LEARNED")):    # PINNED (ledger 2026-10-25, T2x)
        ad, bi, le = A["ADD"], A["BIND"], A["LEARNED"]
        sd = lambda a: [x["held_acc"] for x in allres[ARMS.index(a)]["seeds"]]
        say("VERDICTS (3-seed means)")
        c1 = ad["train_acc_diff"] <= 0.6 and bi["train_acc"] >= 0.95
        say(f"CONTROL: ADD in-dist diff-index {ad['train_acc_diff']:.3f} (ambiguity real if <= 0.6); BIND in-dist {bi['train_acc']:.3f} -> {'OK' if c1 else 'VOID'}")
        say(f"X1 THE PHASE REJOIN GENERALIZES TO UNSEEN PAIRS: {'PASS' if bi['held_acc'] >= 0.95 else 'FAIL'} (BIND held-out {bi['held_acc']:.3f})")
        d = [a - b for a, b in zip(sd("BIND"), sd("LEARNED"))]
        x2 = bi["held_acc"] - le["held_acc"] >= 0.10 and all(x > 0 for x in d)
        say(f"X2 PHASE BINDING BEATS LEARNED BINDING ON THE CROSS JOIN: {'PASS' if x2 else 'FAIL'} (BIND - LEARNED held-out "
            f"{bi['held_acc'] - le['held_acc']:+.3f}; per seed {[round(x, 3) for x in d]}; LEARNED in-dist {le['train_acc']:.3f})")
        if "BIND_MLP" in A:
            say(f"READ BIND_MLP: in-dist {A['BIND_MLP']['train_acc']:.3f} held-out {A['BIND_MLP']['held_acc']:.3f}")
    say(f"DONE in {time.time() - t0:.0f}s")
