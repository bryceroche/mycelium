"""breath_rotor_toy.py -- T2s, THE 60-DEGREE BREATH (2026-10-25; Bryce's word; ledger "T2s"). CPU, self-contained.
Bryce: the breathing transformer loops six times; each breath rotates the attention space by 60 degrees (six turns = one
circle); ONLY THE QUERY rotates (q and k together cancel: (Rq).(Rk) = q.k); snap-to-centroid saves the representation;
the T6 clock carries both macro and micro steps on SEPARATE gears (a gear never counts two things).
TASK: pointer chasing. An episode is a random permutation over KE entities drawn from a tree of NE (each entity has
exactly one outgoing fact a -> b), plus a query (start x, hop count h in 1..6). Answer = x after h hops. Six breaths
always run; h < 6 means the loop must FOLLOW on h breaths and STAY on the rest -- i.e. it must know where it is.
READER (every arm): MHA only, no MLP. A one-shot reading layer composes each fact's key/value from (a, b); a STAY token
and the hop-count token H_h sit in the same memory. Each breath: the state (one entity) queries the memory (4 heads),
state' = state + Wo(attn), then SNAP: straight-through hard snap to the entity tree (cos / tau over the NE centroids).
Arms (shared weights across breaths unless noted):
  NO_ROT      no breath signal at all (floor: the snapped state cannot carry a counter)
  ROT_QK      Q and K rotated together by 60 deg x b (PLACEBO: must equal NO_ROT -- the rotation cancels)
  MICRO       the micro gear (period 6) stamped on the query input: a breath LABEL, no rotation
  ROT60_Q     the query (only) rotated by 60 deg x b in every complex plane: an ACTION, no label
  FULL        ROT60_Q + MICRO (Bryce's design)
  UNTIED      per-breath weights, no rotation, no label (the ceiling)
  FULL_NOSNAP FULL without the snap (read: does the snap carry the chain?)
Macro gears are not used: the facts here carry no story time (stated in the registration)."""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = "--smoke" in sys.argv
ARMS = os.environ.get("TS_ARMS", "NO_ROT,ROT_QK,MICRO,ROT60_Q,FULL,UNTIED,FULL_NOSNAP").split(",")
NE, KE, HMAX, NB = 48, 16, 6, 6                                              # tree size, entities/episode, max hops, breaths
D, NH = 64, 4
DH = D // NH
TAU = 0.1
BATCH, LR = 128, 3e-3
STEPS = 300 if SMOKE else 6000
SEEDS = [0] if SMOKE else [0, 1, 2]
N_EVAL = 600 if SMOKE else 3000
OUT = ".cache/breath_rotor{}.txt".format("_smoke" if SMOKE else "")


def episodes(rng, n):
    """facts (n,KE,2) entity ids; query x (n,); hops h (n,); answer y (n,)."""
    ents = np.stack([rng.choice(NE, KE, replace=False) for _ in range(n)])
    perm = np.stack([rng.permutation(KE) for _ in range(n)])
    src, dst = ents, np.take_along_axis(ents, perm, 1)
    start = rng.integers(0, KE, n); h = rng.integers(1, HMAX + 1, n)
    cur = start.copy()
    for k in range(HMAX):
        cur = np.where(h > k, perm[np.arange(n), cur], cur)
    t = lambda a: torch.from_numpy(np.asarray(a)).long()
    return dict(facts=t(np.stack([src, dst], -1)), x=t(src[np.arange(n), start]), h=t(h), y=t(ents[np.arange(n), cur]))


def rot(z, th):
    """rotate every complex plane (consecutive pairs of the last dim) by angle th."""
    c, s = math.cos(th), math.sin(th)
    re, im = z[..., 0::2], z[..., 1::2]
    return torch.stack([re * c - im * s, re * s + im * c], -1).flatten(-2)


_z = torch.randn(5, 8); _r = _z
for _ in range(6):
    _r = rot(_r, math.pi / 3)
assert torch.allclose(_r, _z, atol=1e-5), "six 60-degree turns must be the identity"


class Net(nn.Module):
    def __init__(self, arm):
        super().__init__(); self.arm = arm
        self.C = nn.Parameter(torch.randn(NE, D) / math.sqrt(D))           # the entity tree (centroids)
        self.read = nn.Linear(2 * D, D)                                      # one-shot reading layer: fact (a,b) -> memory
        self.stay = nn.Parameter(torch.randn(D) / math.sqrt(D))
        self.H = nn.Parameter(torch.randn(HMAX, D) / math.sqrt(D))
        nw = NB if arm == "UNTIED" else 1
        self.q, self.k, self.v, self.o = (nn.ModuleList(nn.Linear(D, D) for _ in range(nw)) for _ in range(4))
        g = torch.Generator().manual_seed(6)                                 # the micro gear: period 6, fixed projection
        self.register_buffer("gearP", torch.randn(2, D, generator=g) / math.sqrt(2))

    def cn(self):
        return F.normalize(self.C, dim=-1)

    def forward(self, bt):
        f = bt["facts"]; B = f.shape[0]; Cn = self.cn()
        mem = torch.cat([self.read(torch.cat([Cn[f[..., 0]], Cn[f[..., 1]]], -1)),
                         self.stay.expand(B, 1, D), self.H[bt["h"] - 1][:, None]], 1)         # (B, KE+2, D)
        s = Cn[bt["x"]]; snaps = []
        for b in range(NB):
            w = b if self.arm == "UNTIED" else 0
            qin = s
            if self.arm in ("MICRO", "FULL", "FULL_NOSNAP"):
                th = 2 * math.pi * b / 6
                qin = qin + torch.tensor([math.cos(th), math.sin(th)]) @ self.gearP
            Q = self.q[w](qin).view(B, NH, DH); K = self.k[w](mem).view(B, -1, NH, DH).transpose(1, 2)
            V = self.v[w](mem).view(B, -1, NH, DH).transpose(1, 2)
            th = math.pi / 3 * b
            if self.arm in ("ROT60_Q", "FULL", "FULL_NOSNAP", "ROT_QK"):
                Q = rot(Q, th)
            if self.arm == "ROT_QK":
                K = rot(K, th)
            a = torch.softmax(torch.einsum("bhd,bhld->bhl", Q, K) / math.sqrt(DH), -1)
            s2 = s + self.o[w](torch.einsum("bhl,bhld->bhd", a, V).reshape(B, D))
            lg = F.normalize(s2, dim=-1) @ Cn.T / TAU
            if self.arm == "FULL_NOSNAP":
                s = s2
            else:
                soft = lg.softmax(-1); hard = F.one_hot(lg.argmax(-1), NE).float()
                s = (hard + soft - soft.detach()) @ Cn                          # straight-through hard snap
            snaps.append(lg.argmax(-1))
        return lg, torch.stack(snaps, 1)


def run(arm, seed):
    rng = np.random.default_rng(seed); ev = episodes(np.random.default_rng(91_000 + seed), N_EVAL)
    torch.manual_seed(seed); net = Net(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = episodes(rng, BATCH); lg, _ = net(bt); loss = F.cross_entropy(lg, bt["y"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        lg, sn = net(ev)
    ok = (lg.argmax(-1) == ev["y"]).float()
    r = {"acc": float(ok.mean()), "loss_end": float(loss)}
    for h in range(1, HMAX + 1):
        r[f"acc_h{h}"] = float(ok[ev["h"] == h].mean())
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    assert SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {'smoke' if SMOKE else 'full'} steps {STEPS} seeds {SEEDS} arms {ARMS} NE {NE} KE {KE} breaths {NB}")
    hk = [f"acc_h{h}" for h in range(1, HMAX + 1)]
    for arm in ARMS:
        rs = [run(arm, s) for s in SEEDS]
        ag = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}; A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        say(f"{arm:11s} acc {ag['acc']:.3f} (seeds {[round(x['acc'], 3) for x in rs]}) | by hops "
            + " ".join(f"{ag[k]:.3f}" for k in hk) + f" | loss {ag['loss_end']:.3f}  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if not SMOKE and all(a in A for a in ("NO_ROT", "ROT_QK", "MICRO", "ROT60_Q", "FULL", "UNTIED")):   # PINNED (ledger 2026-10-25, T2s)
        sd = lambda a: [x["acc"] for x in allres[ARMS.index(a)]["seeds"]]
        n, p, m, r6, fu, un = (A[a] for a in ("NO_ROT", "ROT_QK", "MICRO", "ROT60_Q", "FULL", "UNTIED"))
        say("VERDICTS (3-seed means)")
        say(f"PLACEBO: |ROT_QK - NO_ROT| = {abs(p['acc'] - n['acc']):.3f} -> {'OK' if abs(p['acc'] - n['acc']) <= 0.01 else 'VOID (code)'}")
        say(f"CEILING: UNTIED {un['acc']:.3f} -> {'OK' if un['acc'] >= 0.95 else 'VOID (task/regime)'}")
        r1 = r6["acc"] >= un["acc"] - 0.02 and r6["acc"] >= n["acc"] + 0.30
        say(f"R1 THE ROTATION ORIENTS: {'PASS' if r1 else 'FAIL'} (ROT60_Q {r6['acc']:.3f} vs UNTIED {un['acc']:.3f}, NO_ROT {n['acc']:.3f})")
        d = [a - b for a, b in zip(sd("FULL"), sd("MICRO"))]
        r2 = fu["acc"] - m["acc"] >= 0.02 and all(x > 0 for x in d)
        say(f"R2 ROTATION BEYOND THE LABEL: {'PASS' if r2 else 'FAIL'} (FULL - MICRO {fu['acc'] - m['acc']:+.3f}; per seed {[round(x, 3) for x in d]})")
        r3 = fu["acc_h6"] >= fu["acc_h1"] - 0.03
        say(f"R3 NO DEGRADATION WITH CHAIN LENGTH: {'PASS' if r3 else 'FAIL'} (FULL h1 {fu['acc_h1']:.3f} -> h6 {fu['acc_h6']:.3f})")
        if "FULL_NOSNAP" in A:
            ns = A["FULL_NOSNAP"]
            say(f"READ NOSNAP: acc {ns['acc']:.3f}; h1 {ns['acc_h1']:.3f} -> h6 {ns['acc_h6']:.3f} (FULL {fu['acc']:.3f})")
    say(f"DONE in {time.time() - t0:.0f}s")
