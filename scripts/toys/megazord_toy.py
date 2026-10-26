"""megazord_toy.py -- T3a, MEGAZORD: THE INTEGRATED MACHINE (2026-10-26; Bryce's word: "word given for the compaction fix and
Megazord integrated test -- salute!"; ledger "T3a MEGAZORD"). CPU. One breathing loop carrying every organ the T2s-T2z
toys passed one at a time, each with its own held-out axis, plus one ablation arm per organ.
WORLD: a tree of NT types x NI indices; a leaf's FACT f(t, i) is PERMANENT world knowledge that lives ONLY in the tree
(tree_db v2: key "t:i" -> Welford evidence over the NI indices). EPISODE: an in-context type RULE sigma (a derangement of
the types), a start leaf, a hop count h in 1..NB; NB breaths; breath b < h moves (t, i) -> (sigma(t), f(t, i)), else stays.
THE BREATH (MEGAZORD):
  bottleneck  MHA over [type facts, stay, hop] at the TYPE level; the query rotated at SPEEDS per breath and the hop token's
              key rotated by its count (T2u relative phase); snapped to the type codebook (hard straight-through)
  skip        the ID tag (the index code) only
  expand      the SOURCE address composed by FIXED BINDING (type code * tag code, planewise unit phasors) keys the tree's
              leaf rows (keys composed the same way); the row is read and snapped to the index codebook (T2z BIND_KEY)
  gate        algebra: sigma has no fixed points, so the tag moves iff the type moved (g = 1 - p(type unchanged))
  tree        harvested from a NOISY reader, written only from answer-key-verified chains (MapReduce: 8 shards prepared
              in 8 processes, committed as permanent batches; a saboteur shard on probation); read once per run
  perceiver   snap tension per breath, logged (registered loss-never)
ARMS: MEGAZORD / NO_ROT (gear labels, T2u MICRO_M) / NO_SNAP / MLP_SPLIT (learned nonlinear address composers, T2z Z4) /
MLP_MEM (no tree: an MLP maps (type, tag) -> index and must memorize the facts -- the traditional FFN)."""
import json, math, os, shutil, sys, time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from mycelium import tree_db as TDB

SMOKE = "--smoke" in sys.argv
ARMS = os.environ.get("TM_ARMS", "MEGAZORD,NO_ROT,NO_SNAP,MLP_SPLIT,MLP_MEM").split(",")
NT, NI, P, D, NH, NB = 8, 8, 64, 64, 4, 12
NL, DH = NT * NI, D // NH
BATCH, LR = 128, 3e-3
STEPS = int(os.environ.get("TM_STEPS", 300 if SMOKE else 12000))
SEEDS = [0] if SMOKE else [int(x) for x in os.environ.get("TM_SEEDS", "0,1,2").split(",")]
N_EV = 60 if SMOKE else 300
TRAIN_H, HELD_H, EDGE_H, LONG_H = [1, 2, 3, 5, 6, 7, 9, 10, 11], [4, 8], [12], [9, 10, 11]
SPEEDS = (180.0, 60.0, 30.0, 15.0)
SUF = ("_smoke" if SMOKE else "") + os.environ.get("TM_TAG", "")
CLOCK = tuple(float(x) for x in os.environ.get("TM_CLOCK", ",".join(map(str, SPEEDS))).split(","))   # the clock head's planes (T3b)
OUT = f".cache/megazord{SUF}.txt"
DBP = ".cache/megazord{}_world{}.db"
SPACE, NOISE, N_HARVEST, SHARDS, AUDIT = "leaf-fact", 0.2, (4000 if SMOKE else 20000), 8, 0.99
HELD = np.array([[(t + i) % 4 == 0 for i in range(NI)] for t in range(NT)])
WORLDS = {"A": np.random.default_rng(101).integers(0, NI, (NT, NI)), "B": np.random.default_rng(202).integers(0, NI, (NT, NI))}
assert (WORLDS["A"] != WORLDS["B"]).mean() > 0.7


def derangements(rng, m):
    out = []
    while sum(len(x) for x in out) < m:
        s = rng.random((2 * m, NT)).argsort(1)
        out.append(s[(s != np.arange(NT)).all(1)])
    return np.concatenate(out)[:m]


def chains(rng, m, h, f, noise=0.0):
    """m episodes at hop count h under world f: (sig, t (m, NB+1), i (m, NB+1), observed i (m, NB+1) if noise)."""
    sig = derangements(rng, m); ar = np.arange(m)
    t = [rng.integers(0, NT, m)]; i = [rng.integers(0, NI, m)]; o = [i[0].copy()]
    for b in range(NB):
        go = b < h
        t.append(np.where(go, sig[ar, t[-1]], t[-1]))
        i.append(np.where(go, f[t[-2], i[-1]], i[-1]))
        if noise:                                                           # the noisy reader follows its OWN observed chain
            j = f[t[-2], o[-1]]; wrong = rng.random(m) < noise
            j = np.where(wrong, (j + rng.integers(1, NI, m)) % NI, j)
            o.append(np.where(go, j, o[-1]))
    return sig, np.stack(t, 1), np.stack(i, 1), (np.stack(o, 1) if noise else None)


def episodes(rng, n, hs, split, f):
    """split 'train': no ACTIVE hop sources a held leaf; 'held': at least one does; 'any'. Stratified over hs."""
    import torch
    parts = []
    for h, nh in zip(hs, np.bincount(rng.integers(0, len(hs), n), minlength=len(hs))):
        got, acc = 0, []
        while got < nh:
            sig, t, i, _ = chains(rng, 4 * nh + 16, h, f)
            hs_ = HELD[t[:, :h], i[:, :h]].any(1)
            keep = np.ones(len(t), bool) if split == "any" else (~hs_ if split == "train" else hs_)
            k = np.nonzero(keep)[0][: nh - got]; got += len(k); acc.append((sig[k], t[k], i[k], np.full(len(k), h)))
        parts += acc
    sig, t, i, hh = (np.concatenate(x) for x in zip(*parts))
    L = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long()
    return dict(sig=L(sig), t=L(t), i=L(i), h=L(hh))


# ---- the tree: harvest -> shards -> commit -> probation ------------------------------------------------------------
def harvest(world, seed):
    """Noisy-reader episodes; ADMITTED iff the observed final answer equals the answer key; every hop fact of an admitted
    chain is evidence. Returns (admitted events, all events, admitted fraction)."""
    rng = np.random.default_rng(seed); f = WORLDS[world]; adm, alln, n_adm = [], [], 0
    for h in range(1, NB + 1):
        m = N_HARVEST // NB; sig, t, i, o = chains(rng, m, h, f, NOISE)
        ok = (o[:, h] == i[:, h])
        for e in range(m):
            ev = [(f"{t[e, b]}:{o[e, b]}", np.eye(NI)[o[e, b + 1]], f"w{world}.h{h}.e{e}.b{b}") for b in range(h)]
            alln += ev
            if ok[e]:
                adm += ev; n_adm += 1
    return adm, alln, n_adm / (NB * (N_HARVEST // NB))


def build_tree(world, say):
    p = DBP.format(SUF, world)
    for q in (p, p + "-wal", p + "-shm"):
        if os.path.exists(q):
            assert SMOKE, f"refusing to overwrite {q}"
            os.remove(q)
    if os.path.isdir(p + ".events"):
        shutil.rmtree(p + ".events")
    adm, alln, frac = harvest(world, {"A": 11, "B": 22}[world])
    shards = [adm[s::SHARDS] for s in range(SHARDS)]
    with get_context("fork").Pool(SHARDS) as pool:                          # THE MAP STEP: 8 processes
        preps = pool.starmap(TDB.prepare_shard, [(s, None) for s in shards])
    db = TDB.TreeDB(p); db.register_space(SPACE, "symbolic", NI, "Welford evidence over a leaf fact's target index")
    for s, pr in enumerate(preps):
        db.add_prepared(f"harvest{s}", pr, "permanent", "answer_key", f"harvest.{world}", SPACE)
    ser_p = p + ".serial"; ser = TDB.TreeDB(ser_p); ser.register_space(SPACE, "symbolic", NI, "serial twin")
    for s, sh in enumerate(shards):
        ser.add_evidence(f"harvest{s}", sh, "permanent", "answer_key", f"harvest.{world}", SPACE)
    q_ = "SELECT key, n, mean, m2 FROM leaves ORDER BY key"
    sharded_eq = db.c.execute(q_).fetchall() == ser.c.execute(q_).fetchall(); ser.close()
    for x in (ser_p, ser_p + "-wal", ser_p + "-shm"):
        if os.path.exists(x):
            os.remove(x)
    shutil.rmtree(ser_p + ".events")
    pre = db.c.execute(q_).fetchall()                                       # the saboteur: every target shifted j -> j+1
    db.add_evidence("saboteur", [(k, np.roll(v, 1), s + ".sab") for k, v, s in shards[0]], "provisional", "saboteur", f"sab.{world}", SPACE)
    own = db.batch_summaries("saboteur"); shared = [k for k in own if db.leaf(k) is not None]
    aud = float(np.mean([own[k][1].argmax() == db.leaf(k)[1].argmax() for k in shared]))
    if aud < AUDIT:
        db.rollback("saboteur")
    else:
        db.promote("saboteur")
    sab_ok = aud < AUDIT and db.c.execute(q_).fetchall() == pre
    rows = db.load_leaves(); f = WORLDS[world]
    R = np.zeros((NL, NI)); right = 0
    for t in range(NT):
        for i in range(NI):
            r = rows.get(f"{t}:{i}")
            assert r is not None, f"world {world}: leaf {t}:{i} never harvested"
            R[t * NI + i] = r[1]; right += int(r[1].argmax() == f[t, i])
    agg = TDB.combine([k for k, _, _ in alln], np.stack([v for _, v, _ in alln]))
    right_all = sum(int(agg[f"{t}:{i}"][1].argmax() == f[t, i]) for t in range(NT) for i in range(NI) if f"{t}:{i}" in agg)
    say(f"TREE world {world}: harvest {N_HARVEST} noisy chains, admitted {frac:.3f}, {len(adm)} events in {SHARDS} shards -> "
        f"{len(rows)} leaves; argmax right {right}/{NL} (admit-all read: {right_all}/{NL}); mean row confidence "
        f"{R.max(1).mean():.3f}; 8-shard == serial bit-exact {sharded_eq}; saboteur audition {aud:.3f} -> "
        f"{'REJECTED' if aud < AUDIT else 'PROMOTED'}, store unchanged {sab_ok}")
    db.close()
    return R, dict(right=right, right_all=right_all, admitted=frac, sharded_eq=sharded_eq, sab_ok=sab_ok, audit=aud)


# ---- the machine ---------------------------------------------------------------------------------------------------
def build_net():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(2027)
    TC = torch.polar(torch.ones(NT, P), 2 * math.pi * torch.rand(NT, P, generator=g))
    IC = torch.polar(torch.ones(NI, P), 2 * math.pi * torch.rand(NI, P, generator=g))
    ALL_T, ALL_I = torch.arange(NL) // NI, torch.arange(NL) % NI
    ri = lambda z: torch.cat([z.real, z.imag], -1)

    def gear(n):
        a = n[..., None].float() * torch.tensor(SPEEDS) * math.pi / 180
        return torch.cat([a.cos(), a.sin()], -1)

    def rot(z, th):
        c, s = torch.cos(th), torch.sin(th); re, im = z[..., 0::2], z[..., 1::2]
        return torch.stack([re * c - im * s, re * s + im * c], -1).flatten(-2)

    def st(lg_or_p, is_prob=False):
        soft = lg_or_p if is_prob else lg_or_p.softmax(-1)
        return F.one_hot(soft.argmax(-1), soft.shape[-1]).float() + soft - soft.detach()

    def mlp(i, o):
        return nn.Sequential(nn.Linear(i, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, o))

    class Net(nn.Module):
        def __init__(self, arm):
            super().__init__(); self.arm = arm
            self.rq, self.rk, self.rv = nn.Linear(2 * P, D), nn.Linear(2 * P, D), nn.Linear(2 * P, D)
            self.ro = nn.Linear(D, 2 * P)
            self.cq = nn.Parameter(torch.randn(D) / math.sqrt(D)); self.ck = nn.Parameter(torch.randn(D) / math.sqrt(D))    # the clock head
            self.cb = nn.Parameter(torch.zeros(1)); self.cs = nn.Parameter(torch.tensor(math.log(5.0)))
            self.ls = nn.Parameter(torch.tensor(math.log(10.0))); self.lk = nn.Parameter(torch.tensor(math.log(10.0)))
            self.register_buffer("w", torch.tensor([CLOCK[p % len(CLOCK)] for p in range(DH // 2)]) * math.pi / 180)
            gg = torch.Generator().manual_seed(6)
            self.register_buffer("Pq", F.normalize(torch.randn(2 * len(SPEEDS), D, generator=gg), dim=-1))
            self.register_buffer("Pk", F.normalize(torch.randn(2 * len(SPEEDS), D, generator=gg), dim=-1))
            if arm == "MLP_SPLIT":
                self.mq, self.mk = mlp(4 * P, D), mlp(4 * P, D)
            if arm == "MLP_MEM":
                self.mm = mlp(4 * P, NI)
            self.rotate, self.snap = arm != "NO_ROT", arm != "NO_SNAP"

        def forward(self, bt, R):
            Bn = bt["sig"].shape[0]; h = bt["h"]; tcn = F.normalize(ri(TC), dim=-1); wD = self.w.repeat(NH)   # D/2 clock planes
            ht = F.one_hot(bt["t"][:, 0], NT).float(); hi = F.one_hot(bt["i"][:, 0], NI).float()
            K = self.rk(ri(TC)).expand(Bn, NT, D).reshape(Bn, NT, NH, DH).transpose(1, 2)   # fact t -> sigma(t): key the source
            V = self.rv(ri(TC[bt["sig"]])).view(Bn, NT, NH, DH).transpose(1, 2)
            if self.rotate:                                                 # the hop token's key at its own phase h
                hopk = rot(self.ck.expand(Bn, D), wD[None] * h[:, None].float())
            else:                                                           # NO_ROT: a gear LABEL stamped on the hop key
                hopk = self.ck + gear(h) @ self.Pk / math.sqrt(len(SPEEDS))
            if self.arm in ("MEGAZORD", "NO_ROT", "NO_SNAP"):
                fk = ri(TC[ALL_T] * IC[ALL_I]) / math.sqrt(P)
            elif self.arm == "MLP_SPLIT":
                fk = F.normalize(self.mk(torch.cat([ri(TC[ALL_T]), ri(IC[ALL_I])], -1)), dim=-1)
            lts, lis, mt, mi = [], [], [], []
            for b in range(NB):
                Tq = ht.to(torch.complex64) @ TC; Iq = hi.to(torch.complex64) @ IC
                # THE CLOCK HEAD: its query rotated by the breath, the hop key at phase h -> p_go = f(b - h) (relative phase)
                cq = rot(self.cq, wD * b) if self.rotate else self.cq + gear(torch.tensor(b)) @ self.Pq / math.sqrt(len(SPEEDS))
                p_go = torch.sigmoid((hopk @ cq)[:, None] * self.cs.exp() + self.cb)
                # THE BOTTLENECK: content heads read the type rule; the move enters ONLY through the gate (a road, not a sum)
                Q = self.rq(ri(Tq)).view(Bn, NH, DH)
                a = torch.softmax(torch.einsum("bhd,bhld->bhl", Q, K) / math.sqrt(DH), -1)
                r = ri(Tq) + p_go * self.ro(torch.einsum("bhl,bhld->bhd", a, V).reshape(Bn, D))
                cos_t = F.normalize(r, dim=-1) @ tcn.T; lt = cos_t * self.ls.exp(); pt = lt.softmax(-1)
                g = 1 - (pt * ht).sum(-1, keepdim=True)                                       # the gate is algebra
                if self.arm == "MLP_MEM":
                    dist = self.mm(torch.cat([ri(Tq), ri(Iq)], -1)).softmax(-1)
                else:
                    if self.arm == "MLP_SPLIT":
                        sc = F.normalize(self.mq(torch.cat([ri(Tq), ri(Iq)], -1)), dim=-1) @ fk.T
                    else:
                        sc = (ri(Tq * Iq) / math.sqrt(P)) @ fk.T
                    dist = torch.softmax(sc * self.lk.exp(), -1) @ R                          # read the tree's row
                pi = g * dist + (1 - g) * hi
                li = torch.log(pi + 1e-6)
                if self.snap:
                    ht, hi = st(lt), st(pi / pi.sum(-1, keepdim=True), is_prob=True)
                else:
                    ht, hi = pt, pi / pi.sum(-1, keepdim=True)
                with torch.no_grad():
                    tc2 = cos_t.topk(2, -1).values; ti2 = pi.topk(2, -1).values
                    mt.append(tc2[:, 0] - tc2[:, 1]); mi.append(ti2[:, 0] - ti2[:, 1])
                lts.append(lt); lis.append(li)
            self.tension = torch.minimum(torch.stack(mt, 1), torch.stack(mi, 1))             # the perceiver: read-only
            return torch.stack(lts, 1), torch.stack(lis, 1)

    return Net, F


def auroc(score, pos):
    s, p = np.asarray(score, float), np.asarray(pos, bool)
    if p.all() or not p.any():
        return float("nan")
    r = s.argsort().argsort().astype(float)
    for v in np.unique(s):
        m = s == v; r[m] = r[m].mean()
    n1, n0 = p.sum(), (~p).sum()
    return float((r[p].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def run(arm, seed, RA, RB):
    import torch
    from mycelium.diagnostic_register import assert_not_supervised
    Net, F = build_net()
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed); fA, fB = WORLDS["A"], WORLDS["B"]
    ev = {"indist": {h: episodes(erng, N_EV, [h], "train", fA) for h in TRAIN_H},
          "held_leaf": {h: episodes(erng, N_EV, [h], "held", fA) for h in TRAIN_H},
          "held_h": {h: episodes(erng, N_EV, [h], "train", fA) for h in HELD_H},
          "joint": {h: episodes(erng, N_EV, [h], "held", fA) for h in HELD_H},
          "edge": {h: episodes(erng, N_EV, [h], "train", fA) for h in EDGE_H},
          "swap": {h: episodes(erng, N_EV, [h], "any", fB) for h in range(1, NB)}}
    torch.manual_seed(seed); net = Net(arm); tA, tB = torch.tensor(RA, dtype=torch.float32), torch.tensor(RB, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    assert_not_supervised(["trace_ce"])                                     # the door: snap tension never enters the loss
    for step in range(STEPS):
        bt = episodes(rng, BATCH, TRAIN_H, "train", fA); lt, li = net(bt, tA)
        loss = F.cross_entropy(lt.reshape(-1, NT), bt["t"][:, 1:].reshape(-1)) + F.nll_loss(li.reshape(-1, NI), bt["i"][:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); r = {"loss_end": float(loss.detach())}; tens, errs = [], []
    if os.environ.get("TM_SAVE"):                                           # T3b: state dicts for the gate-margin read
        torch.save(net.state_dict(), f".cache/megazord{SUF}_{arm}_s{seed}.pt")
    with torch.no_grad():
        for sp, byh in ev.items():
            accs = []
            for h, bt in byh.items():
                lt, li = net(bt, tB if sp == "swap" else tA)
                ok = (lt[:, -1].argmax(-1) == bt["t"][:, -1]) & (li[:, -1].argmax(-1) == bt["i"][:, -1])
                r[f"{sp}_h{h}"] = float(ok.float().mean()); accs.append(r[f"{sp}_h{h}"])
                tens += net.tension.min(1).values.tolist(); errs += (~ok).tolist()
            r[sp] = float(np.mean(accs))
    r["long"] = float(np.mean([r[f"indist_h{h}"] for h in LONG_H]))
    r["tension_auroc"] = auroc(-np.array(tens), np.array(errs)); r["n_err"] = int(np.sum(errs))
    return r


if __name__ == "__main__":
    assert SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {'smoke' if SMOKE else 'full'} (pid {os.getpid()}) steps {STEPS} seeds {SEEDS} arms {ARMS} tree {NT}x{NI} breaths {NB} "
        f"clock {CLOCK} train h {TRAIN_H} held h {HELD_H} held leaves {int(HELD.sum())} noise {NOISE}")
    RA, treeA = build_tree("A", say); RB, treeB = build_tree("B", say)      # before torch spins up threads (fork)
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    A, allres = {}, []
    for arm in ARMS:
        rs = [run(arm, s, RA, RB) for s in SEEDS]
        ag = {k: float(np.nanmean([x[k] for x in rs])) for k in rs[0]}; A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        sd = lambda k: [round(x[k], 3) for x in rs]
        say(f"{arm:9s} in-dist {ag['indist']:.3f} | HELD_LEAF {ag['held_leaf']:.3f} {sd('held_leaf')} HELD_H {ag['held_h']:.3f} {sd('held_h')} "
            f"JOINT {ag['joint']:.3f} SWAP {ag['swap']:.3f} {sd('swap')} LONG {ag['long']:.3f} EDGE {ag['edge']:.3f} | tension AUROC "
            f"{ag['tension_auroc']:.3f} on {ag['n_err']:.0f} errors | loss {ag['loss_end']:.3f}  [{time.time() - t0:.0f}s]")
        say("          in-dist by h " + " ".join(f"{ag[f'indist_h{h}']:.2f}" for h in TRAIN_H) +
            " | swap by h " + " ".join(f"{ag[f'swap_h{h}']:.2f}" for h in range(1, NB)))
    json.dump(dict(trees=dict(A=treeA, B=treeB), arms=allres), open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if not SMOKE and all(a in A for a in ("MEGAZORD", "NO_ROT", "NO_SNAP", "MLP_SPLIT", "MLP_MEM")):   # PINNED (ledger 2026-10-26, T3a)
        mz = A["MEGAZORD"]; sds = lambda a, k: [x[k] for x in allres[ARMS.index(a)]["seeds"]]
        say("VERDICTS (3-seed means)")
        say(f"CONTROL: MEGAZORD in-dist {mz['indist']:.3f} -> {'OK' if mz['indist'] >= 0.95 else 'VOID'}")
        m1 = {k: mz[k] >= 0.95 for k in ("held_leaf", "held_h", "joint", "swap")}
        say(f"M1 THE MACHINE COMPOSES: {'PASS' if all(m1.values()) else 'FAIL'} " + ", ".join(f"{k} {mz[k]:.3f}" for k in m1))
        m2 = {}
        for arm, axis, bar in (("NO_ROT", "held_h", 0.10), ("MLP_SPLIT", "held_leaf", 0.10), ("MLP_MEM", "swap", 0.10), ("NO_SNAP", "long", 0.05)):
            d = [x - y for x, y in zip(sds("MEGAZORD", axis), sds(arm, axis))]
            m2[arm] = mz[axis] - A[arm][axis] >= bar and all(x > 0 for x in d)
            say(f"   M2 {arm:9s} on {axis:9s}: MEGAZORD - arm {mz[axis] - A[arm][axis]:+.3f} (bar {bar}; per seed {[round(x, 3) for x in d]}; "
                f"arm in-dist {A[arm]['indist']:.3f}) -> {'PASS' if m2[arm] else 'FAIL'}")
        say(f"M2 EVERY ORGAN IS LOAD-BEARING ON ITS AXIS: {'PASS' if all(m2.values()) else 'FAIL'}")
        m3 = all(t["right"] >= 63 and t["sab_ok"] and t["sharded_eq"] for t in (treeA, treeB))
        say(f"M3 THE TREE'S CUSTODY: {'PASS' if m3 else 'FAIL'} (right A {treeA['right']}/64 B {treeB['right']}/64; saboteur rejected+unchanged "
            f"{treeA['sab_ok']}/{treeB['sab_ok']}; sharded == serial {treeA['sharded_eq']}/{treeB['sharded_eq']})")
        say(f"READ admit-all tree: A {treeA['right_all']}/64 B {treeB['right_all']}/64; admitted A {treeA['admitted']:.3f} B {treeB['admitted']:.3f}")
        say(f"READ MLP_MEM vs MEGAZORD: in-dist {A['MLP_MEM']['indist']:.3f} vs {mz['indist']:.3f}; HELD_LEAF {A['MLP_MEM']['held_leaf']:.3f} vs "
            f"{mz['held_leaf']:.3f}; SWAP {A['MLP_MEM']['swap']:.3f} vs {mz['swap']:.3f}")
    say(f"DONE in {time.time() - t0:.0f}s")
