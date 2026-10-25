"""breath_transfer_toy.py -- T2u, THE 12-BREATH LOOP: DOES THE ROTATION TRANSFER, DOES THE SNAP HOLD? (2026-10-25; Bryce's word;
ledger "T2u"). CPU. T2t: at matched speeds a query ROTATION and a gear LABEL tie on every in-distribution breath rule. The case
left for the rotation is RELATIVE PHASE (RoPE's property): rotate the query by w*b and the hop token's key by w*h, and the
stop score depends on (b - h) alone, so a hop count never trained on uses offsets that were. A label must learn the
(b, h) comparison pair by pair.
TASK: T2s's pointer chase, 12 breaths always, h in 1..12. TRAIN h in {1,2,3,5,6,7,9,10,11}; HELD h in {4, 8} (interior:
every offset b - h they need was trained); EDGE h = 12 (offset -12 never trained; read only).
Speeds {180, 60, 30, 15} deg/breath (periods 2/6/12/24): the 15-degree plane is v109pi's no-wrap sweep over 12 breaths, so
offsets -11..11 are all distinct (with periods <= 12 the offsets +6 and -6 collide).
Arms (MHA-only reader, straight-through hard snap, per-breath trace supervision unless noted; 12000 steps, 3 seeds):
  MICRO_M        gear LABELS: g(b) stamped on the query, g(h) stamped on the hop token
  ROT_M          gear ROTATIONS: query planes by w*b, hop-token key planes by w*h (relative phase)
  UNTIED         per-breath weights + the hop-token label (ceiling on trained h)
  ROT_M_NOSNAP   ROT_M without the snap
  ROT_ANS        ROT_M trained on the FINAL ANSWER only (read: the clean snap test, if it trains at all)
  ROT_ANS_NOSNAP ROT_ANS without the snap (read)"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = "--smoke" in sys.argv
ARMS = os.environ.get("TU_ARMS", "MICRO_M,ROT_M,UNTIED,ROT_M_NOSNAP,ROT_ANS,ROT_ANS_NOSNAP").split(",")
NE, KE, NB, D, NH = 48, 16, 12, 64, 4
DH = D // NH
TAU, BATCH, LR = 0.1, 128, 3e-3
STEPS = 300 if SMOKE else 12000
SEEDS = [0] if SMOKE else [0, 1, 2]
N_PER_H = 100 if SMOKE else 500
TRAIN_H, HELD_H, EDGE_H = [1, 2, 3, 5, 6, 7, 9, 10, 11], [4, 8], [12]
SPEEDS = (180.0, 60.0, 30.0, 15.0)
OUT = ".cache/breath_transfer{}.txt".format("_smoke" if SMOKE else "")
_d = np.arange(-11, 12)
_ph = np.stack([np.exp(1j * np.deg2rad(s * _d)) for s in SPEEDS], 1)
assert len({tuple(np.round(r, 6)) for r in _ph}) == len(_d), "offsets -11..11 must have distinct phases"


def episodes(rng, n, hs):
    ents = np.stack([rng.choice(NE, KE, replace=False) for _ in range(n)])
    perm = np.stack([rng.permutation(KE) for _ in range(n)])
    start = rng.integers(0, KE, n); h = np.asarray(hs)[rng.integers(0, len(hs), n)] if len(hs) > 1 else np.full(n, hs[0])
    cur = start.copy(); traj = []; ar = np.arange(n)
    for k in range(NB):
        cur = np.where(h > k, perm[ar, cur], cur); traj.append(ents[ar, cur])
    t = lambda a: torch.from_numpy(np.asarray(a)).long()
    return dict(facts=t(np.stack([ents, np.take_along_axis(ents, perm, 1)], -1)), x=t(ents[ar, start]), h=t(h),
                y=t(traj[-1]), traj=t(np.stack(traj, 1)))


def gear(n):
    """(..., 2*len(SPEEDS)) gear phasor features of a tick count tensor n."""
    a = n[..., None].float() * torch.tensor(SPEEDS) * math.pi / 180
    return torch.cat([a.cos(), a.sin()], -1)


def rot_planes(z, th):
    c, s = torch.cos(th), torch.sin(th)
    re, im = z[..., 0::2], z[..., 1::2]
    return torch.stack([re * c - im * s, re * s + im * c], -1).flatten(-2)


class Net(nn.Module):
    def __init__(self, arm):
        super().__init__(); self.arm = arm
        self.C = nn.Parameter(torch.randn(NE, D) / math.sqrt(D))
        self.read = nn.Linear(2 * D, D)
        self.stay = nn.Parameter(torch.randn(D) / math.sqrt(D))
        self.Hb = nn.Parameter(torch.randn(D) / math.sqrt(D))
        nw = NB if arm == "UNTIED" else 1
        self.q, self.k, self.v, self.o = (nn.ModuleList(nn.Linear(D, D) for _ in range(nw)) for _ in range(4))
        g = torch.Generator().manual_seed(6)
        self.register_buffer("Pq", F.normalize(torch.randn(2 * len(SPEEDS), D, generator=g), dim=-1))
        self.register_buffer("Pk", F.normalize(torch.randn(2 * len(SPEEDS), D, generator=g), dim=-1))
        self.register_buffer("w", torch.tensor([SPEEDS[p % len(SPEEDS)] for p in range(DH // 2)]) * math.pi / 180)
        self.rot = arm.startswith("ROT")
        self.snap = not arm.endswith("NOSNAP")

    def forward(self, bt):
        f = bt["facts"]; Bn = f.shape[0]; Cn = F.normalize(self.C, dim=-1); h = bt["h"]
        Hm = self.Hb.expand(Bn, D)
        if not self.rot:
            Hm = Hm + gear(h) @ self.Pk / math.sqrt(len(SPEEDS))
        mem = torch.cat([self.read(torch.cat([Cn[f[..., 0]], Cn[f[..., 1]]], -1)), self.stay.expand(Bn, 1, D), Hm[:, None]], 1)
        L = mem.shape[1]; s = Cn[bt["x"]]; lgs = []
        for b in range(NB):
            wi = b if self.arm == "UNTIED" else 0
            qin = s
            if self.arm == "MICRO_M":
                qin = qin + gear(torch.tensor(b)) @ self.Pq / math.sqrt(len(SPEEDS))
            Q = self.q[wi](qin).view(Bn, NH, DH); K = self.k[wi](mem).view(Bn, L, NH, DH).transpose(1, 2)
            V = self.v[wi](mem).view(Bn, L, NH, DH).transpose(1, 2)
            if self.rot:
                Q = rot_planes(Q, self.w * b)
                kh = rot_planes(K[:, :, -1], self.w[None, None] * h[:, None, None].float())    # the hop token's key at its phase h
                K = torch.cat([K[:, :, :-1], kh[:, :, None]], 2)
            att = torch.softmax(torch.einsum("bhd,bhld->bhl", Q, K) / math.sqrt(DH), -1)
            s2 = s + self.o[wi](torch.einsum("bhl,bhld->bhd", att, V).reshape(Bn, D))
            lg = F.normalize(s2, dim=-1) @ Cn.T / TAU
            if self.snap:
                soft = lg.softmax(-1); hard = F.one_hot(lg.argmax(-1), NE).float()
                s = (hard + soft - soft.detach()) @ Cn
            else:
                s = s2
            lgs.append(lg)
        return lg, torch.stack(lgs, 1)


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    ev = {h: episodes(erng, N_PER_H, [h]) for h in range(1, NB + 1)}
    torch.manual_seed(seed); net = Net(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = episodes(rng, BATCH, TRAIN_H); lg, lgs = net(bt)
        loss = F.cross_entropy(lg, bt["y"]) if arm.startswith("ROT_ANS") else F.cross_entropy(lgs.reshape(-1, NE), bt["traj"].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); r = {"loss_end": float(loss.detach())}
    with torch.no_grad():
        for h, bt in ev.items():
            lg, _ = net(bt); r[f"acc_h{h}"] = float((lg.argmax(-1) == bt["y"]).float().mean())
    mean = lambda hs: float(np.mean([r[f"acc_h{h}"] for h in hs]))
    r.update(train=mean(TRAIN_H), held=mean(HELD_H), edge=mean(EDGE_H), short=mean([1, 2, 3]), long=mean([9, 10, 11]))
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    assert SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    # relative phase: the rotated hop-token score must depend on b - h only
    torch.manual_seed(0); nt = Net("ROT_M"); q = torch.randn(1, NH, DH); k = torch.randn(1, NH, DH)
    sc = lambda b, h: float((rot_planes(q, nt.w * b) * rot_planes(k, nt.w * h)).sum())
    assert abs(sc(5, 2) - sc(9, 6)) < 1e-4 and abs(sc(3, 8) - sc(0, 5)) < 1e-4, "rotation score is not relative"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {'smoke' if SMOKE else 'full'} steps {STEPS} seeds {SEEDS} arms {ARMS} breaths {NB} train h {TRAIN_H} held {HELD_H} edge {EDGE_H}")
    for arm in ARMS:
        rs = [run(arm, s) for s in SEEDS]
        ag = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}; A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        say(f"{arm:14s} train {ag['train']:.3f} HELD {ag['held']:.3f} (seeds {[round(x['held'], 3) for x in rs]}) edge {ag['edge']:.3f} | "
            "h1-12 " + " ".join(f"{ag[f'acc_h{h}']:.2f}" for h in range(1, NB + 1)) + f" | loss {ag['loss_end']:.3f}  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if not SMOKE and all(a in A for a in ("MICRO_M", "ROT_M", "UNTIED", "ROT_M_NOSNAP")):   # PINNED (ledger 2026-10-25, T2u)
        sd = lambda a, k: [x[k] for x in allres[ARMS.index(a)]["seeds"]]
        mi, ro, un, ns = A["MICRO_M"], A["ROT_M"], A["UNTIED"], A["ROT_M_NOSNAP"]
        say("VERDICTS (3-seed means)")
        say(f"CEILING: UNTIED trained-h {un['train']:.3f} -> {'OK' if un['train'] >= 0.95 else 'VOID'}")
        u1 = ro["held"] >= 0.95 and ro["held"] - mi["held"] >= 0.05
        say(f"U1 THE ROTATION TRANSFERS TO UNSEEN HOP COUNTS: {'PASS' if u1 else 'FAIL'} (ROT_M held {ro['held']:.3f} vs MICRO_M "
            f"{mi['held']:.3f}; UNTIED {un['held']:.3f})")
        d = [a - b for a, b in zip(sd("ROT_M", "long"), sd("ROT_M_NOSNAP", "long"))]
        u2 = ro["long"] >= ro["short"] - 0.03 and ro["long"] - ns["long"] >= 0.02 and all(x > 0 for x in d)
        say(f"U2 THE SNAP HOLDS LONG CHAINS: {'PASS' if u2 else 'FAIL'} (ROT_M h1-3 {ro['short']:.3f} -> h9-11 {ro['long']:.3f}; "
            f"NOSNAP h9-11 {ns['long']:.3f}; snap - nosnap per seed {[round(x, 3) for x in d]})")
        for a in ("ROT_ANS", "ROT_ANS_NOSNAP"):
            if a in A:
                say(f"READ {a}: train {A[a]['train']:.3f} held {A[a]['held']:.3f} h1-3 {A[a]['short']:.3f} h9-11 {A[a]['long']:.3f}")
        say(f"READ EDGE h=12: MICRO_M {mi['edge']:.3f} ROT_M {ro['edge']:.3f} UNTIED {un['edge']:.3f}")
    say(f"DONE in {time.time() - t0:.0f}s")
