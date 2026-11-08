"""bt_organs_smoke.py -- THE ORGANS SMOKE (ledger 2026-11-07 19:25 "THE ORGANS, WIRED"; Bryce: "properly setup and smoke our new
Breathing Transformer class to integrate with our external organs"). Wiring bars SM-J/T/C/A/R, not claims.

Synthetic typed tokens (no GSM8K): NTY token types, each a prototype + noise in d_in; a token's word id is drawn from its type's
words; sentences of SENT tokens (the T9 clock). THREE TIERS (business / data / surface), each its own BT with 6 heads and 6 Tree
partitions (18 in all). Tier t's slot k points at the first token of type (k + t) mod NTY. Per step every tier trains one masked-
diffusion breath (a random breath b, tau(b) per example, slots committed with probability ~ b / NCYC carrying their commit breath as
their T6 stamp and their target's type as the certificate); the business tier's committed types ride the T256 bus (salt 0) into
every tier's slot entry. After each step the tier's certified token chart points fold into its own partitions (Welford + BIRCH).
Eval: one Conductor pass, 6 breaths x 3 tiers, solver = commit the most confident pointers (no Tree writes: uncertified).
Env: SM_STEPS (600), SM_B (32), SM_SEED (0), SM_BEAM (the chart beam W, default 2; 0 = the sealed read of SMa-SMc), SM_DB (tree db path; default a fresh file under .cache). -> .cache/bt_organs_smoke.json
"""
import json, math, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.nn.optim import AdamW

from mycelium import breathing_transformer as BTM
from mycelium import bt_organs as O
from mycelium.tree_db import TreeDB

STEPS, B, SEED = int(os.environ.get("SM_STEPS", "600")), int(os.environ.get("SM_B", "32")), int(os.environ.get("SM_SEED", "0"))
BEAM = int(os.environ.get("SM_BEAM", "2"))
DBP = os.environ.get("SM_DB", f".cache/bt_organs_smoke_s{SEED}.db")
TIERS, NH, K, L, SENT, NTY, DIN, NCYC, PBUS = ("business", "data", "surface"), 6, 6, 32, 8, 6, 64, 6, 256
GEN = f"smoke-s{SEED}"
rng = np.random.default_rng(SEED); PROTO = rng.standard_normal((NTY, DIN)).astype(np.float32)
BOOK = O.PhaseBus.phases(np.arange(NTY), PBUS, salt=7)                              # the type codebook on the bus


def config(tier):
    return BTM.BTConfig(tier=tier, d_in=DIN, d_model=256, d_waist=64, d_head=32, d_chart=32, d_cert=NTY, n_partitions=NH, n_charts=16, chart_beam=BEAM,
                        d_bus=PBUS, n_breaths=NCYC, roads=BTM.default_roads(n_heads=NH, charts=True))


def problems(n, r):
    """-> token types (n, L), features (n, L, DIN), word ids, sentence ids; every type present"""
    ty = r.integers(0, NTY, (n, L))
    for b in range(n): ty[b, r.choice(L, NTY, replace=False)] = np.arange(NTY)
    x = PROTO[ty] + 0.5 * r.standard_normal((n, L, DIN)).astype(np.float32)
    wid = ty * 1000 + r.integers(0, 5, (n, L)); sent = np.broadcast_to(np.arange(L) // SENT, (n, L)).copy()
    return ty, x, wid, sent


def gold(ty, tier):
    t = TIERS.index(tier); want = (np.arange(K)[None] + t) % NTY                     # (1, K)
    return (ty[:, None, :] == want[..., None]).argmax(-1)                            # first token of the wanted type (n, K)


def token_io(bt, wid, sent):
    tf = {**O.T9Clock().fields(sent), "token_id": wid, "table": np.zeros_like(wid), "level": np.zeros_like(wid)}
    return bt.layout.angles(tf, wid.shape)


def slot_io(bt, nb, now):
    """the slots' composite keys: slot index, T6 stamp = the commit breath (committed) or now (open), level"""
    f = nb.slot_fields(); f["tick"] = np.where(nb.committed, nb.tick, np.asarray(now).reshape(-1, 1))
    return bt.layout.angles(f, (nb.B, nb.K))


def bus_read(nbB, n):
    """the bus as every tier reads it: the business tier's committed types bound at its slot keys (salt 0)"""
    bus = O.PhaseBus(n, PBUS); key = O.PhaseBus.phases(np.broadcast_to(np.arange(K), (n, K)), PBUS, salt=0)
    typ = nbB.cert.argmax(-1); bus.write(key, BOOK[typ], radius=nbB.radius * nbB.committed)
    return bus, key


def forward(bt, x, ta, tp, sa, sp, vt, vc, vs, charts, ok, com, cert, bus, tau, keep=None):
    io = BTM.BreathIO(h={"token": bt.enter_tokens(x)}, angles={"token": ta}, presence={"token": tp}, valid={"token_self": vt},
                      charts=charts, chart_ok=ok)
    ht = bt.breath(io, 0, roads=("token_self",))["h"]["token"]                       # the reader (breath 0's token road)
    hs = bt.enter_slots(com, cert, bus=bus)
    io = BTM.BreathIO(h={"token": ht, "slot": hs}, angles={"token": ta, "slot": sa}, presence={"token": tp, "slot": sp},
                      valid={"slot_self": vs, "cross": vc, "pointer": vc}, charts=charts, chart_ok=ok,
                      beam_keep={} if keep is None else {"pointer": keep})                       # training: gold-in-beam
    return bt.breath(io, None, roads=("slot_self", "cross", "pointer"), tau=tau)


class Tier:
    def __init__(self, tier):
        self.bt = BTM.BreathingTransformer(config(tier))
        self.params = [v for k, v in self.bt.state_dict().items() if k != "sil_scale"]   # no silhouette loss in this smoke
        for p in self.params: p.requires_grad = True
        self.opt = AdamW(self.params, lr=3e-4, weight_decay=0.01); self.jit = TinyJit(self._step)

    def _step(self, x, ta, tp, sa, sp, vt, vc, vs, charts, ok, com, cert, bus, tau, g):
        with Tensor.train():
            self.opt.zero_grad()
            o = forward(self.bt, x, ta, tp, sa, sp, vt, vc, vs, charts, ok, com, cert, bus, tau, keep=g.one_hot(L) > 0)
            lp = o["logp"]["pointer"][:, 0]                                          # (B, K, L)
            lo = -(lp * g.one_hot(L).cast(dtypes.float)).sum(-1).mean()
            lo.backward()
            gp = [p for p in self.params if p.grad is not None]
            gn = Tensor.stack(*[(p.grad ** 2).sum() for p in gp]).sum().sqrt()
            coef = (1.0 / (gn + 1e-6)).minimum(1.0) * lo.isfinite().cast(dtypes.float)
            for p in gp: p.grad = p.grad * coef
            pts = o["chart_pts"]["token"].contiguous()
            Tensor.realize(lo, pts, *[p.grad for p in gp]); self.opt.step()
            return lo.realize(), pts


def masks(n):
    vt = np.ones((n, L, L), bool); vc = np.ones((n, K, L), bool); vs = np.ones((n, K, K), bool)
    return vt, vc, vs


def main():
    T = lambda a: Tensor(np.ascontiguousarray(a))
    if os.path.exists(DBP): os.remove(DBP)
    db = TreeDB(DBP); tt = O.TreeTables(db, GEN, cap=4000, max_idle=50)
    Tensor.manual_seed(SEED); tiers = {t: Tier(t) for t in TIERS}
    for t in TIERS: tt.attach(tiers[t].bt)
    r = np.random.default_rng(SEED + 1); los = {t: [] for t in TIERS}; t0 = time.time(); vt, vc, vs = masks(B)
    for step in range(STEPS):
        ty, x, wid, sent = problems(B, r)
        b = r.integers(1, NCYC, B); tau = np.array([config("x").tau(int(c)) for c in b], np.float32)
        nbs = {}
        for t in TIERS:                                                              # the notebooks: masked-diffusion commits
            g = gold(ty, t); nb = O.Notebook(B, K, L, NTY); pc = b / NCYC
            for i in range(B):
                for k in range(K):
                    if r.random() < pc[i]:
                        nb.write(i, k, atom=0, radius=1.0, cert=np.eye(NTY)[ty[i, g[i, k]]], tick=int(r.integers(0, b[i])), consumes=(g[i, k],))
            nbs[t] = nb
        bus, key = bus_read(nbs["business"], B); bf = bus.features(key)
        for t in TIERS:
            tr = tiers[t]; ta, tp = token_io(tr.bt, wid, sent); sa, sp = slot_io(tr.bt, nbs[t], b)
            charts, ok = tt.io(tr.bt); nb = nbs[t]
            lo, pts = tr.jit(T(x), T(ta), T(tp), T(sa), T(sp), T(vt), T(vc), T(vs), charts.realize(), ok.realize(), T(nb.committed),
                             T(nb.cert), T(bf), T(tau), T(gold(ty, t).astype(np.int32)))
            lv = float(lo.numpy()); los[t].append(lv); assert math.isfinite(lv), (t, step, lv)
            tt.observe(tr.bt, pts.numpy(), np.ones((B, L), bool))                     # synthetic gold: every token certified
        if step % 100 == 0 or step == STEPS - 1:
            live = {t: [int(tt.tables[p].live.sum()) for p in tiers[t].bt.c.partitions()] for t in TIERS}
            print(f"[sm] step {step} " + " ".join(f"{t} {np.mean(los[t][-50:]):.3f}" for t in TIERS) + f" live {live} ({time.time() - t0:.0f}s)", flush=True)

    res = {}
    res["SM-J"] = {t: [float(np.mean(los[t][:50])), float(np.mean(los[t][-50:]))] for t in TIERS}
    parts = [p for t in TIERS for p in tiers[t].bt.c.partitions()]
    nlive = {"|".join(p): int(tt.tables[p].live.sum()) for p in parts}
    cs = [tt.tables[p].centroids()[tt.tables[p].live] for p in parts]
    same = [(i, j) for i in range(len(cs)) for j in range(i + 1, len(cs)) if cs[i].shape == cs[j].shape and np.allclose(cs[i], cs[j], atol=1e-4)]
    res["SM-T"] = dict(n_partitions=len(parts), live=nlive, equal_pairs=same)

    # beside: purity of each partition's charts vs token type, head-vs-head agreement (ARI) within a tier
    re_ = np.random.default_rng(99); ty, x, wid, sent = problems(64, re_); vt6, vc6, vs6 = masks(64)
    pur, ari, hit = {}, {}, {}
    for t in TIERS:
        tr = tiers[t]; ta, tp = token_io(tr.bt, wid, sent); nb = O.Notebook(64, K, L, NTY); sa, sp = slot_io(tr.bt, nb, np.zeros(64))
        charts, ok = tt.io(tr.bt); bus0 = np.zeros((64, K, 2 * PBUS), np.float32)
        o = forward(tr.bt, T(x), T(ta), T(tp), T(sa), T(sp), T(vt6), T(vc6), T(vs6), charts, ok, T(nb.committed), T(nb.cert), T(bus0), 1.0)
        pts = o["chart_pts"]["token"].numpy(); assign = []
        for h, p in enumerate(tr.bt.c.partitions()):
            a = tt.tables[p].nearest(pts[:, h].reshape(-1, pts.shape[-1])); assign.append(a)
            ct = np.zeros((16, NTY)); np.add.at(ct, (a, ty.reshape(-1)), 1); pur["|".join(p)] = float(ct.max(1).sum() / ct.sum())
        ari[t] = float(np.mean([_ari(assign[i], assign[j]) for i in range(NH) for j in range(i + 1, NH)]))
        # chart hit: the pointer road (head 0 -> partition h0): does the query's top chart hold the gold token's chart? (tau 1)
        tb = tt.tables[tr.bt.c.partition(0)]; cent = tb.centroids(); g = gold(ty, t)
        cq = tr.bt.chart_coords(o["z"]["slot"]).numpy()[:, 0]                                       # (64, K, dc)
        kc = tb.nearest(pts[:, 0][np.arange(64)[:, None], g].reshape(-1, pts.shape[-1])).reshape(64, K)
        d = -((cq[:, :, None] - cent[None, None]) ** 2).sum(-1) / math.sqrt(cent.shape[-1]); d[..., ~tb.live] = -np.inf
        pc = np.exp(d - d.max(-1, keepdims=True)); pc /= pc.sum(-1, keepdims=True)
        top2 = np.argsort(-d, -1)[..., :2]
        hit[t] = dict(hit=float((d.argmax(-1) == kc).mean()), top2=float((top2 == kc[..., None]).any(-1).mean()),
                      mass=float(np.take_along_axis(pc, kc[..., None], -1).mean()))
    res["purity"], res["head_ari"], res["chart_hit"] = pur, ari, hit

    # SM-C + SM-A: one Conductor pass, uncertified, 6 breaths x 3 tiers
    nE = 64; ty, x, wid, sent = problems(nE, np.random.default_rng(7))
    n_before = {p: tt.tables[p].n.sum() for p in parts}
    org = O.Organs(notebook={t: O.Notebook(nE, K, L, NTY) for t in TIERS}, t6=O.T6Clock(), t9=O.T9Clock(), tree=tt,
                   bus=O.PhaseBus(nE, PBUS), extra=dict(x=x, wid=wid, sent=sent, ty=ty))
    tr_ = O.Conductor([(tiers[t].bt, EvalPort()) for t in TIERS], Solver(org)).run(org)
    res["SM-C"] = dict(changed=[ "|".join(p) for p in parts if tt.tables[p].n.sum() != n_before[p]])
    nbB = org.notebook["business"]; key = O.PhaseBus.phases(np.broadcast_to(np.arange(K), (nE, K)), PBUS, salt=0)
    got, _ = org.bus.cleanup(key, BOOK)
    stamps = {t: float(np.mean([len(set(org.notebook[t].tick[i].tolist())) for i in range(nE)])) for t in TIERS}
    res["SM-A"] = dict(all_committed={t: bool(org.notebook[t].committed.all()) for t in TIERS}, stamps_per_problem=stamps,
                       bus_cleanup=float((got == nbB.cert.argmax(-1)).mean()),
                       ptr_acc={t: float(np.mean(org.extra["pick"][t] == gold(ty, t))) for t in TIERS}, trace=len(tr_))

    # SM-R: the Tree round trip
    tt.save(); tt2 = O.TreeTables(TreeDB(DBP), GEN, cap=4000)
    rt = all(all(np.allclose(a, b) for a, b in zip(tt2.table(p, 16, 32).state(), tt.tables[p].state())) for p in parts)
    try: tt2.load_generation(parts[0], "smoke-other"); refused = False
    except ValueError: refused = True
    res["SM-R"] = dict(roundtrip=bool(rt), foreign_refused=refused)
    json.dump(res, open(".cache/bt_organs_smoke.json", "w"), indent=1)
    verdict(res)


def _ari(a, b):
    ct = np.zeros((a.max() + 1, b.max() + 1)); np.add.at(ct, (a, b), 1); c2 = lambda v: v * (v - 1) / 2
    s = c2(ct).sum(); sa, sb, n = c2(ct.sum(1)).sum(), c2(ct.sum(0)).sum(), c2(len(a)); e = sa * sb / n
    return float((s - e) / (0.5 * (sa + sb) - e)) if 0.5 * (sa + sb) != e else 1.0


class EvalPort:
    """per tier: tokens read once (breath 0's token road), slots from the tier's notebook + the bus, charts from the Tree"""
    def io(self, bt, organs, b):
        e = organs.extra; n = e["x"].shape[0]; nb = organs.notebook[bt.c.tier]; T = lambda a: Tensor(np.ascontiguousarray(a))
        ta, tp = token_io(bt, e["wid"], e["sent"]); sa, sp = slot_io(bt, nb, np.full(n, organs.t6.t))
        key = O.PhaseBus.phases(np.broadcast_to(np.arange(K), (n, K)), PBUS, salt=0)
        ht = e.get(("h", bt.c.tier), bt.enter_tokens(T(e["x"])))
        cons = np.broadcast_to(~nb.consumed[:, None, :], (n, K, L)).copy()
        charts, ok = organs.tree.io(bt); vt, _, vs = masks(n)
        return BTM.BreathIO(h={"token": ht, "slot": bt.enter_slots(T(nb.committed), T(nb.cert), bus=T(organs.bus.features(key)))},
                            angles={"token": T(ta), "slot": T(sa)}, presence={"token": T(tp), "slot": T(sp)},
                            valid={"token_self": T(vt), "slot_self": T(vs), "cross": T(cons), "pointer": T(cons)}, charts=charts, chart_ok=ok)

    def propose(self, bt, out, organs, b):
        organs.extra[("h", bt.c.tier)] = out["h"]["token"]
        return np.exp(out["logp"]["pointer"][:, 0].numpy())                         # (n, K, L)

    def certified(self, bt, out, organs, b):
        return {}                                                                    # test-time: nothing is certified


class Solver:
    """commit the most confident open slots (ceil(K (b+1) / NCYC) by breath b); the certificate = the pointed token's type; the
    business tier's commits are bound on the bus (salt 0, radius = confidence)"""
    def __init__(self, org):
        self.org = org; org.extra["pick"] = {t: np.full((org.extra["x"].shape[0], K), -1) for t in TIERS}

    def step(self, tier, p, nb, tick):
        org = self.org; ty = org.extra["ty"]; tgt = math.ceil(K * (tick + 1) / NCYC); done = 0
        for i in range(nb.B):
            conf = p[i].max(-1); need = tgt - nb.committed[i].sum()
            for k in [k for k in np.argsort(-conf) if not nb.committed[i, k]][:max(need, 0)]:
                j = int(p[i, k].argmax()); nb.write(i, k, atom=0, radius=float(conf[k]), cert=np.eye(NTY)[ty[i, j]], tick=tick, consumes=(j,))
                org.extra["pick"][tier][i, k] = j; done += 1
                if tier == "business":                                               # bind (slot key, type) for problem i only
                    kk = np.zeros((nb.B, PBUS)); vv = np.zeros((nb.B, PBUS)); rr = np.zeros(nb.B)
                    kk[i] = O.PhaseBus.phases(np.array([k]), PBUS, salt=0)[0]; vv[i] = BOOK[ty[i, j]]; rr[i] = conf[k]
                    org.bus.write(kk, vv, rr)
        return done


def verdict(res):
    j = res["SM-J"]; Lj = [f"SM-J  loss first/last 50 " + " ".join(f"{t} {a:.3f}->{b:.3f}" for t, (a, b) in j.items())
                         + f" -> {'PASS' if all(b <= 0.7 * a for a, b in j.values()) else 'FAIL'}"]
    t_ = res["SM-T"]; mn = min(t_["live"].values())
    Lj.append(f"SM-T  {t_['n_partitions']} partitions, live charts min {mn} max {max(t_['live'].values())}, equal pairs {len(t_['equal_pairs'])} -> "
              f"{'PASS' if t_['n_partitions'] == 18 and mn >= 2 and not t_['equal_pairs'] else 'FAIL'}")
    Lj.append(f"SM-C  partitions changed by the uncertified pass: {res['SM-C']['changed']} -> {'PASS' if not res['SM-C']['changed'] else 'FAIL'}")
    a = res["SM-A"]; sm = min(a["stamps_per_problem"].values())
    Lj.append(f"SM-A  all committed {a['all_committed']}, T6 stamps/problem {a['stamps_per_problem']}, bus cleanup {a['bus_cleanup']:.4f} -> "
              f"{'PASS' if all(a['all_committed'].values()) and sm >= 2 and a['bus_cleanup'] >= 0.95 else 'FAIL'}")
    r = res["SM-R"]; Lj.append(f"SM-R  round trip {r['roundtrip']}, foreign refused {r['foreign_refused']} -> {'PASS' if r['roundtrip'] and r['foreign_refused'] else 'FAIL'}")
    pa = a["ptr_acc"]; Lj.append(f"SM-P  eval pointer accuracy {pa} (>= 0.90 all three tiers) -> {'PASS' if min(pa.values()) >= 0.90 else 'FAIL'}")
    Lj.append(f"  beside: head ARI per tier {res['head_ari']}; chart hit (pointer query's top chart = gold's chart, mass) {res['chart_hit']}")
    lv_ = res["SM-T"]["live"]; Lj.append(f"  beside: live charts per partition {sorted(lv_.values())} (capacity 16)")
    pu = res["purity"]; Lj.append("  beside: purity per tier (mean over heads) " + " ".join(
        f"{t} {np.mean([v for k_, v in pu.items() if k_.startswith(t)]):.3f}" for t in TIERS) + f" (chance ~{1 / NTY:.3f} + chart count)")
    print("\n".join(Lj)); open(".cache/bt_organs_smoke_verdict.txt", "w").write("\n".join(Lj) + "\n")


if __name__ == "__main__":
    main()
