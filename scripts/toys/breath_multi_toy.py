"""breath_multi_toy.py -- T2t, THE BREATH-TIME RoPE: THE CLOCK'S GEARS AS QUERY ROTATIONS (2026-10-25; Bryce's word; ledger "T2t").
CPU. T2s (breath_rotor_toy) found the period-6 LABEL orients the 6-breath loop (1.000) and the single-speed 60-degree query
ROTATION mostly does (0.918, weak at h=1). Geometry: with one speed every score is A cos(60b) + B sin(60b), so a breath
rule is one line cutting the 6-point circle (margins: h3 0.50, h2/h4 0.43, h1/h5 0.25 -- the wrap makes breath 5 the
neighbour of breath 0). Several speeds add harmonics; speeds {180, 60, 30} deg/breath (parity / Bryce's full circle /
v109pi's half circle) span every function of the 6 breaths (rank 6, asserted below).
TASK A (T2s's pointer chase, h in 1..6, follow then stay): ROT60_Q and MICRO re-run (reproduction), ROT_MULTI.
TASK B (DIFFERENT OPERATIONS PER BREATH): every entity has an A-successor and a B-successor (two permutations); six hops
always, breath b follows relation SCHED[b] = A A B A B B (fixed; not an arc of the circle, so ONE speed cannot carve it in
one head -- the best single-frequency line gets 4 of 6 breaths; four heads could compose arcs). Arms:
  NO_ROT       floor (no breath signal)
  MICRO        the period-6 gear as a label (one speed)
  MICRO_MULTI  the three gears {180,60,30} as labels (stamp normalised to the state's norm)
  ROT60_Q      the query rotated 60 deg/breath in every plane (one speed)
  ROT_MULTI    the query's complex planes rotated at 180 / 60 / 30 deg per breath (plane p -> speed p % 3)
  UNTIED       per-breath weights (ceiling)
Reader, snap, training (per-breath trace supervision, 12000 steps, 3 seeds) = breath_rotor_toy's, imported."""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_argv = sys.argv[:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import breath_rotor_toy as B
sys.argv = _argv
SMOKE = B.SMOKE
NE, KE, NB, D, NH, DH, TAU = B.NE, B.KE, B.NB, B.D, B.NH, B.DH, B.TAU
SPEEDS = (180.0, 60.0, 30.0)
SCHED = [0, 0, 1, 0, 1, 1]                                                  # A A B A B B
RUNS = [("A", a) for a in os.environ.get("TT_A", "ROT60_Q,MICRO,ROT_MULTI").split(",") if a] + \
       [("B", a) for a in os.environ.get("TT_B", "NO_ROT,MICRO,MICRO_MULTI,ROT60_Q,ROT_MULTI,UNTIED").split(",") if a]
OUT = ".cache/breath_multi{}.txt".format("_smoke" if SMOKE else "")
_bb = np.arange(NB)
_F = np.array([np.ones(NB)] + [f(np.deg2rad(s * _bb)) for s in SPEEDS for f in (np.cos, np.sin)])
assert np.linalg.matrix_rank(_F, tol=1e-9) == NB, "the three speeds must span every function of the six breaths"


def episodes_b(rng, n):
    ents = np.stack([rng.choice(NE, KE, replace=False) for _ in range(n)])
    pa = np.stack([rng.permutation(KE) for _ in range(n)]); pb = np.stack([rng.permutation(KE) for _ in range(n)])
    start = rng.integers(0, KE, n); cur = start.copy(); traj = []; ar = np.arange(n)
    for b in range(NB):
        cur = (pa if SCHED[b] == 0 else pb)[ar, cur]; traj.append(ents[ar, cur])
    fa = np.stack([ents, np.take_along_axis(ents, pa, 1), np.zeros_like(ents)], -1)
    fb = np.stack([ents, np.take_along_axis(ents, pb, 1), np.ones_like(ents)], -1)
    t = lambda a: torch.from_numpy(np.asarray(a)).long()
    return dict(facts=t(np.concatenate([fa, fb], 1)), x=t(ents[ar, start]), h=t(np.full(n, NB)), y=t(traj[-1]),
                traj=t(np.stack(traj, 1)))


def rot_planes(z, th):
    """rotate complex plane p (pairs of the last dim) by th[p]."""
    c, s = torch.cos(th), torch.sin(th)
    re, im = z[..., 0::2], z[..., 1::2]
    return torch.stack([re * c - im * s, re * s + im * c], -1).flatten(-2)


class Net(B.Net):
    def __init__(self, arm, task):
        base = {"ROT_MULTI": "NO_ROT", "MICRO_MULTI": "NO_ROT"}.get(arm, arm)
        super().__init__(base); self.arm2, self.task = arm, task           # base params drawn first: task-A arms = T2s's init
        if task == "B":
            self.R = nn.Parameter(torch.randn(2, D) / math.sqrt(D))
        if arm == "MICRO_MULTI":
            g = torch.Generator().manual_seed(66)
            self.register_buffer("gearP6", F.normalize(torch.randn(2 * len(SPEEDS), D, generator=g), dim=-1))
        self.register_buffer("pspeed", torch.tensor([SPEEDS[p % 3] for p in range(DH // 2)]) * math.pi / 180)

    def forward(self, bt):
        f = bt["facts"]; Bn = f.shape[0]; Cn = self.cn()
        fm = self.read(torch.cat([Cn[f[..., 0]], Cn[f[..., 1]]], -1))
        if self.task == "B":
            fm = fm + self.R[f[..., 2]]
        mem = torch.cat([fm, self.stay.expand(Bn, 1, D), self.H[bt["h"] - 1][:, None]], 1)
        s = Cn[bt["x"]]; lgs = []; a = self.arm2
        for b in range(NB):
            w = b if a == "UNTIED" else 0
            qin = s
            if a == "MICRO":
                th = 2 * math.pi * b / 6
                qin = qin + torch.tensor([math.cos(th), math.sin(th)]) @ self.gearP
            if a == "MICRO_MULTI":
                ph = torch.tensor([v for sp in SPEEDS for v in (math.cos(math.radians(sp * b)), math.sin(math.radians(sp * b)))])
                qin = qin + (ph @ self.gearP6) / math.sqrt(len(SPEEDS))
            Q = self.q[w](qin).view(Bn, NH, DH); K = self.k[w](mem).view(Bn, -1, NH, DH).transpose(1, 2)
            V = self.v[w](mem).view(Bn, -1, NH, DH).transpose(1, 2)
            if a == "ROT60_Q":
                Q = B.rot(Q, math.pi / 3 * b)
            if a == "ROT_MULTI":
                Q = rot_planes(Q, self.pspeed * b)
            att = torch.softmax(torch.einsum("bhd,bhld->bhl", Q, K) / math.sqrt(DH), -1)
            s2 = s + self.o[w](torch.einsum("bhl,bhld->bhd", att, V).reshape(Bn, D))
            lg = F.normalize(s2, dim=-1) @ Cn.T / TAU
            soft = lg.softmax(-1); hard = F.one_hot(lg.argmax(-1), NE).float()
            s = (hard + soft - soft.detach()) @ Cn
            lgs.append(lg)
        return lg, torch.stack(lgs, 1)


def run(task, arm, seed):
    gen = B.episodes if task == "A" else episodes_b
    rng = np.random.default_rng(seed); ev = gen(np.random.default_rng(91_000 + seed), B.N_EVAL)
    torch.manual_seed(seed); net = Net(arm, task)
    opt = torch.optim.AdamW(net.parameters(), lr=B.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=B.LR, total_steps=B.STEPS, pct_start=0.1)
    for step in range(B.STEPS):
        bt = gen(rng, B.BATCH); _, lgs = net(bt)
        loss = F.cross_entropy(lgs.reshape(-1, NE), bt["traj"].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        lg, lgs = net(ev)
    ok = (lg.argmax(-1) == ev["y"]).float(); step_ok = (lgs.argmax(-1) == ev["traj"]).float()
    r = {"acc": float(ok.mean()), "trace_acc": float(step_ok.all(-1).float().mean()), "loss_end": float(loss.detach())}
    if task == "A":
        for h in range(1, B.HMAX + 1):
            r[f"acc_h{h}"] = float(ok[ev["h"] == h].mean())
    else:                                                                   # per-breath step accuracy given a correct prefix
        pre = torch.ones(step_ok.shape[0])
        for b in range(NB):
            r[f"step_b{b}"] = float(step_ok[pre.bool(), b].mean()) if pre.sum() else float("nan"); pre = pre * step_ok[:, b]
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    assert SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    torch.manual_seed(0); a0 = B.Net("ROT60_Q"); torch.manual_seed(0); a1 = Net("ROT60_Q", "A"); a0.eval(); a1.eval()
    bt0 = B.episodes(np.random.default_rng(3), 16)
    with torch.no_grad():
        assert torch.equal(a0(bt0)[1], a1(bt0)[1]), "task-A ROT60_Q differs from T2s's"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {'smoke' if SMOKE else 'full'} steps {B.STEPS} seeds {B.SEEDS} runs {RUNS} speeds {SPEEDS} sched {SCHED}")
    for task, arm in RUNS:
        rs = [run(task, arm, s) for s in B.SEEDS]
        ag = {k: float(np.nanmean([x[k] for x in rs])) for k in rs[0]}; A[(task, arm)] = ag
        allres.append(dict(task=task, arm=arm, seeds=rs, agg=ag))
        cols = [f"acc_h{h}" for h in range(1, B.HMAX + 1)] if task == "A" else [f"step_b{b}" for b in range(NB)]
        say(f"{task} {arm:11s} acc {ag['acc']:.3f} (seeds {[round(x['acc'], 3) for x in rs]}) trace {ag['trace_acc']:.3f} | "
            + ("by hops " if task == "A" else "step|prefix ") + " ".join(f"{ag[k]:.3f}" for k in cols) + f"  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    need = [("A", "ROT60_Q"), ("A", "MICRO"), ("A", "ROT_MULTI"), ("B", "MICRO_MULTI"), ("B", "ROT_MULTI"), ("B", "UNTIED")]
    if not SMOKE and all(k in A for k in need):                             # PINNED (ledger 2026-10-25, T2t)
        g = lambda t, a: A[(t, a)]
        sd = lambda t, a: [x["acc"] for x in next(r for r in allres if r["task"] == t and r["arm"] == a)["seeds"]]
        say("VERDICTS (3-seed means)")
        c = abs(g("A", "ROT60_Q")["acc"] - 0.918) <= 0.02 and g("A", "MICRO")["acc"] >= 0.99
        say(f"CONTROL A (reproduce T2s): ROT60_Q {g('A', 'ROT60_Q')['acc']:.3f} (banked 0.918), MICRO {g('A', 'MICRO')['acc']:.3f} -> {'OK' if c else 'VOID'}")
        un = g("B", "UNTIED")["acc"]
        say(f"CEILING B: UNTIED {un:.3f} -> {'OK' if un >= 0.95 else 'VOID'}")
        m = g("A", "ROT_MULTI")
        say(f"M1 MULTI-SPEED CLOSES THE GAP (A): {'PASS' if m['acc'] >= 0.99 and m['acc_h1'] >= 0.97 else 'FAIL'} "
            f"(ROT_MULTI {m['acc']:.3f}, h1 {m['acc_h1']:.3f})")
        rm = g("B", "ROT_MULTI")["acc"]
        say(f"M2 MULTI-SPEED ROTATION RUNS DIFFERENT OPS (B): {'PASS' if rm >= un - 0.02 else 'FAIL'} (ROT_MULTI {rm:.3f} vs UNTIED {un:.3f})")
        d = [x - y for x, y in zip(sd("B", "ROT_MULTI"), sd("B", "MICRO_MULTI"))]
        m3 = rm - g("B", "MICRO_MULTI")["acc"] >= 0.02 and all(x > 0 for x in d)
        say(f"M3 ACTION BEATS LABEL AT MATCHED SPEEDS (B): {'PASS' if m3 else 'FAIL'} (ROT_MULTI - MICRO_MULTI "
            f"{rm - g('B', 'MICRO_MULTI')['acc']:+.3f}; per seed {[round(x, 3) for x in d]})")
        r6 = g("A", "ROT60_Q")
        say("READ margin table (A, ROT60_Q by h; predicted easiest h3, hardest h1/h5): "
            + " ".join(f"h{h} {r6[f'acc_h{h}']:.3f}" for h in range(1, B.HMAX + 1)))
    say(f"DONE in {time.time() - t0:.0f}s")
