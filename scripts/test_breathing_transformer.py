"""test_breathing_transformer.py -- the spec of the official BT (ledger 2026-11-07 "THE TIERED BT ARCHITECTURE"), checked mechanically.
CPU, tiny dims, no data. Run: DEV=CPU .venv/bin/python3 scripts/test_breathing_transformer.py
"""
import math, os, re, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tinygrad import Tensor

from mycelium import breathing_transformer as BTM
from mycelium import bt_organs as O
from mycelium.tree_db import TreeDB

B, LT, K, C = 2, 7, 3, 3
CFG = dict(tier="business", d_in=16, d_model=32, d_waist=8, d_head=8, d_chart=8, d_cert=3, n_atoms=5, n_breaths=6, soft_belief=True,
           n_partitions=2, n_charts=C, d_bus=16, roads=BTM.default_roads(n_heads=2, charts=True))
OK = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}"); OK.append(bool(cond))


def inputs(bt, seed=0, beam=None, sent=None):
    rng = np.random.default_rng(seed); L = bt.layout
    t9 = O.T9Clock(); nb = O.Notebook(B, K, LT, 3)
    nb.write(0, 0, atom=2, radius=0.9, cert=[1, 0, 0], tick=0, level=1, consumes=(1,))
    sent = np.array([[0, 0, 0, 1, 1, 1, 1]] * B) if sent is None else sent
    tf = {**t9.fields(sent), "token_id": rng.integers(0, 50000, (B, LT)), "table": np.zeros((B, LT), np.int64), "level": np.ones((B, LT), np.int64)}
    if beam is not None: tf["beam"] = np.full((B, LT), beam)
    ta, tp = L.angles(tf, (B, LT)); sa, spr = L.angles(nb.slot_fields(beam=None if beam is None else [beam] * B), (B, K))
    x = Tensor(rng.standard_normal((B, LT, 16)).astype(np.float32))
    atoms = Tensor(rng.standard_normal((5, 8)).astype(np.float32))
    ht = bt.enter_tokens(x); hs = bt.enter_slots(**nb.tensors(), atoms=atoms)
    pad = np.ones((B, LT), bool); pad[1, -2:] = False
    vt = Tensor(np.broadcast_to(pad[:, None, :], (B, LT, LT)).copy()); vs = Tensor(np.ones((B, K, K), bool))
    vc = Tensor(np.broadcast_to((pad & ~nb.consumed)[:, None, :], (B, K, LT)).copy())
    kc = Tensor(rng.integers(0, C, (B, 2, LT)))
    io = BTM.BreathIO(h={"token": ht, "slot": hs}, angles={"token": Tensor(ta), "slot": Tensor(sa)},
                      presence={"token": Tensor(tp), "slot": Tensor(spr)},
                      valid={"token_self": vt, "slot_self": vs, "cross": vc, "pointer": vc},
                      charts=Tensor(rng.standard_normal((2, C, 8)).astype(np.float32)), chart_ok=Tensor(np.ones((2, C), bool)),
                      key_charts={"cross": kc, "pointer": kc[:, :1]})
    return io, pad & ~nb.consumed


# 1. config + the diffusion schedule
c = BTM.BTConfig(**CFG); taus = [c.tau(b) for b in range(c.n_breaths)]
check("tau falls monotonically from tau_first to tau_last", all(a > b for a, b in zip(taus, taus[1:])) and abs(taus[0] - 2) < 1e-9
      and abs(taus[-1] - 0.5) < 1e-9, " ".join(f"{t:.3f}" for t in taus))

# 2. disjoint planes + the conjugate unbind
L = BTM.KeyLayout(c.factors)
own = L.owner(); check("every factor owns its own disjoint planes", len(own) == L.P and len(set(own)) == len(c.factors), f"P = {L.P}")
a1, _ = L.angles({"token_id": np.array([[7, 7, 8]])}, (1, 3)); s, e = L.slices["token_id"]
same, diff = np.cos(a1[0, 0, s:e] - a1[0, 1, s:e]).mean(), np.cos(a1[0, 0, s:e] - a1[0, 2, s:e]).mean()
check("unbind by the conjugate: same id matches 1, another id does not", abs(same - 1) < 1e-6 and diff < 0.9, f"{same:.3f} vs {diff:.3f}")
check("an absent factor's planes stay empty", np.all(a1[0, :, :s] == 0) and np.all(a1[0, :, e:] == 0))

# 3. no MLP, no RoPE (structure: no activation function in the BT at all)
src = open(BTM.__file__).read()
code = "\n".join(l.split("#")[0] for l in re.sub(r'"""[\s\S]*?"""', "", src).splitlines())   # docstrings + comments out
acts = re.findall(r"\.(relu|gelu|silu|swish|tanh|sigmoid|softplus|elu|leakyrelu|mish|quick_gelu)\(", code)
check("no MLP: no activation function anywhere in the BT", not acts, str(acts))
check("no RoPE: no rotary position table in the BT", not re.findall(r"\b(rope|rotary|_build_rope_freqs)\b", code.lower()))

# 4. breaths are weight-tied; heads specialize
n6 = sum(p.numel() for p in BTM.BreathingTransformer(c).parameters())
n12 = sum(p.numel() for p in BTM.BreathingTransformer(BTM.BTConfig(**{**CFG, "n_breaths": 12})).parameters())
check("weight-tied breaths: params independent of the breath count", n6 == n12, f"{n6} params at 6 and 12 breaths")
pp = {(r.name, h): c.partition(h) for r in c.roads if r.charted() for h in range(r.n_heads)}
check("each head reads its own Tree partition (tier, head), the same one on every charted road",
      len(set(pp.values())) == c.n_partitions and pp[("cross", 0)] == pp[("pointer", 0)] == ("business", "h0") and pp[("cross", 1)] == ("business", "h1"))

# 5. a breath: shapes, normalization, the seal
Tensor.manual_seed(0); bt = BTM.BreathingTransformer(c)
io, open_ = inputs(bt); out = bt.breath(io, 0)
lp = out["logp"]["pointer"].numpy()
check("pointer log-probs: (B, H, K, Lt)", lp.shape == (B, 1, K, LT), str(lp.shape))
mass = np.exp(lp).sum(-1)
check("every road's distribution sums to 1", all(np.allclose(np.exp(out["logp"][r].numpy()).sum(-1), 1, atol=1e-4) for r in out["logp"]),
      f"pointer {mass.min():.5f}..{mass.max():.5f}")
check("closed keys get no mass (padding + consumed)", np.exp(lp)[np.broadcast_to(~open_[:, None, None, :], lp.shape)].max() < 1e-6)
check("silhouette = d_waist", out["z"]["slot"].shape == (B, K, 8))


def chart_mass(o, kc):
    p = np.exp(o["logp"]["cross"].numpy()); oh = np.eye(C)[kc]                                 # (B, H, Lt, C)
    return np.einsum("bhql,bhlc->bhqc", p, oh)


kc = io.key_charts["cross"].numpy()
m0 = chart_mass(out, kc)
cr = bt.blocks[0].roads["cross"]; w0 = cr.k.weight.numpy()
cr.k.weight = Tensor(w0 + np.random.default_rng(1).standard_normal(w0.shape).astype(np.float32)).contiguous()
m1 = chart_mass(bt.breath(io, 0), kc)
check("THE SEAL: the micro score cannot move mass between charts", np.allclose(m0, m1, atol=1e-5), f"max shift {np.abs(m0 - m1).max():.2e}")
cr.k.weight = Tensor(w0).contiguous()

# 5b. THE CHART BEAM (W = 2): pruned charts get no mass; inside the beam the micro moves mass across charts; gold-in-beam; W >= C = PoE
Tensor.manual_seed(0); btb = BTM.BreathingTransformer(BTM.BTConfig(**{**c.__dict__, "chart_beam": 2})); btb.load(bt.state_dict())
ob = btb.breath(io, 0); pb_ = np.exp(ob["logp"]["cross"].numpy()); mb0 = chart_mass(ob, kc)
check("chart beam: every query's mass sits on at most W = 2 charts, and sums to 1",
      (np.sum(mb0 > 1e-6, -1) <= 2).all() and np.allclose(pb_.sum(-1), 1, atol=1e-4), f"charts per query max {np.sum(mb0 > 1e-6, -1).max()}")
crb = btb.blocks[0].roads["cross"]; crb.k.weight = Tensor(w0 + np.random.default_rng(1).standard_normal(w0.shape).astype(np.float32)).contiguous()
mb1 = chart_mass(btb.breath(io, 0), kc); crb.k.weight = Tensor(w0).contiguous()
check("chart beam: the micro moves mass between the beam's charts, never into a pruned chart",
      np.abs(mb0 - mb1).max() > 1e-3 and ((mb0 > 1e-6) | (mb1 <= 1e-6)).all(), f"max shift {np.abs(mb0 - mb1).max():.2e}")
gk = np.zeros((B, K, LT), bool); pr = np.argwhere(mb0[:, 0] <= 1e-6)                            # (b, query, chart) pruned for head 0
bq = next((b_, q_, c_) for b_, q_, c_ in pr if ((kc[b_, 0] == c_) & open_[b_]).any())
kk = int(np.flatnonzero((kc[bq[0], 0] == bq[2]) & open_[bq[0]])[0]); gk[bq[0], bq[1], kk] = True
io.beam_keep = {"cross": Tensor(gk)}; mk_ = chart_mass(btb.breath(io, 0), kc); io.beam_keep = {}
check("chart beam: gold-in-beam keeps a pruned key's chart reachable (training)", mk_[bq[0], 0, bq[1], bq[2]] > 1e-6, f"mass {mk_[bq[0], 0, bq[1], bq[2]]:.3e}")
lm, km = ob["macro"]["cross"]; lm, km = lm.numpy(), km.numpy()
check("out['macro']: log p(chart | q) per charted road sums to 1 over charts; each key's chart = the read's key charts",
      np.allclose(np.exp(lm).sum(-1), 1, atol=1e-4) and (km == kc).all() and set(ob["macro"]) == {r.name for r in btb.c.roads if r.charts}, f"{lm.shape}")
Tensor.manual_seed(0); btf = BTM.BreathingTransformer(BTM.BTConfig(**{**c.__dict__, "chart_beam": C})); btf.load(bt.state_dict())
of = btf.breath(io, 0, roads=("cross",)); lpf = of["logp"]["cross"].numpy()
lps = bt.breath(io, 0, roads=("cross",))["logp"]["cross"].numpy(); same = []
for b_ in range(B):
    for c_ in np.unique(kc[b_, 0]):
        ks = np.flatnonzero((kc[b_, 0] == c_) & open_[b_])
        if len(ks) >= 2: same.append(np.abs((lpf[b_, 0][:, ks] - lpf[b_, 0][:, ks[:1]]) - (lps[b_, 0][:, ks] - lps[b_, 0][:, ks[:1]])).max())
check("chart beam W >= C = the product of experts: sums to 1; inside one chart the key ratios equal the sealed read's (the chart term cancels)",
      np.allclose(np.exp(lpf).sum(-1), 1, atol=1e-4) and len(same) > 0 and max(same) < 1e-3, f"{len(same)} charts, max ratio gap {max(same):.1e}")

o1 = bt.breath(io, 1)
check("token self-attention runs on breath 0 only (default)", "token_self" in out["logp"] and "token_self" not in o1["logp"]
      and np.allclose(o1["h"]["token"].numpy(), io.h["token"].numpy()))

# 6. keys never pass the waist; queries do (a BT with the cross road only)
c1 = BTM.BTConfig(**{**CFG, "roads": tuple(r for r in CFG["roads"] if r.name in ("cross",))}); Tensor.manual_seed(0)
b1 = BTM.BreathingTransformer(c1); io1, _ = inputs(b1)
base = b1.breath(io1, 0)["logp"]["cross"].numpy()
wd = b1.waist["token"].down.weight; b1.waist["token"].down.weight = (wd * 3 + 1).contiguous()
kt = b1.breath(io1, 0)["logp"]["cross"].numpy(); b1.waist["token"].down.weight = wd
ws = b1.waist["slot"].down.weight; b1.waist["slot"].down.weight = (ws * 3 + 1).contiguous()
qs = b1.breath(io1, 0)["logp"]["cross"].numpy(); b1.waist["slot"].down.weight = ws
check("keys bypass the waist (the key stream's waist changes nothing)", np.allclose(base, kt, atol=1e-6))
check("queries pass the waist (the query stream's waist changes the read)", np.abs(base - qs).max() > 1e-3, f"{np.abs(base - qs).max():.3f}")

# 6b. THE SKIP breaks slot symmetry: identical slot states, different slot ids -> different queries; the skip reads IDs only
io_s, _ = inputs(bt); io_s.h["slot"] = Tensor.zeros(B, K, 32).contiguous() + 0.1
zs_ = bt.breath(io_s, 1, roads=("slot_self", "cross"))["logp"]["cross"].numpy()
check("THE SKIP: identical slot states with distinct slot ids read differently", np.abs(zs_[:, :, 0] - zs_[:, :, 1]).max() > 1e-4,
      f"{np.abs(zs_[:, :, 0] - zs_[:, :, 1]).max():.2e}")
check("THE SKIP carries only the ID tags (its input width = 2 x planes)", bt.waist["slot"].skip.weight.shape == (32, 2 * bt.layout.P))

# 7. wildcards: a factor no head matches is invisible; a matched one is not
o_a = bt.breath(inputs(bt, beam=0)[0], 0)["logp"]["cross"].numpy(); o_b = bt.breath(inputs(bt, beam=3)[0], 0)["logp"]["cross"].numpy()
check("wildcard factor (beam) leaves the read unchanged", np.allclose(o_a, o_b, atol=1e-6))
sent2 = np.array([[0, 1, 1, 1, 2, 2, 2]] * B)
o_c = bt.breath(inputs(bt, sent=sent2)[0], 0)["logp"]["cross"].numpy()
check("matched factor (T9 address) changes the read", np.abs(out["logp"]["cross"].numpy() - o_c).max() > 1e-4)

# 8. diffusion sharpening: lower temperature, lower entropy
ent = lambda o: float(-(np.exp(o) * np.where(np.exp(o) > 0, o, 0)).sum(-1).mean())
e0, e5 = ent(bt.breath(io, 0)["logp"]["pointer"].numpy()), ent(bt.breath(io, 5)["logp"]["pointer"].numpy())
check("the read sharpens across breaths (entropy breath 0 > breath 5)", e0 > e5, f"{e0:.3f} -> {e5:.3f}")

# 9. two terminals: every parameter receives a gradient
Tensor.manual_seed(0); bt2 = BTM.BreathingTransformer(c)
ps = bt2.parameters()
for p in ps: p.requires_grad = True
with Tensor.train():
    io2, _ = inputs(bt2); o2 = bt2.breath(io2, 0)
    nb_belief = Tensor(np.full((B, K, 5), 0.2, np.float32))
    nbx = O.Notebook(B, K, LT); nbx.write(0, 0, atom=1, radius=0.5, cert=[0, 1, 0], tick=0)
    hs2 = bt2.enter_slots(**nbx.tensors(), belief=nb_belief, atoms=Tensor(np.ones((5, 8), np.float32)), bus=Tensor(np.ones((B, K, 32), np.float32)))
    loss = sum(v.exp().mul(v.clip(-50, 0)).sum() for v in o2["logp"].values()) + o2["h"]["token"].square().mean() \
        + o2["h"]["slot"].square().mean() + o2["z"]["slot"].square().mean() + hs2.square().mean() \
        + bt2.silhouette_logp(o2["z"]["slot"], Tensor(np.ones((5, 8), np.float32)), 1.0)[..., 0].mean() \
        + bt2.condition(o2["h"]["slot"], Tensor(np.zeros((B, K), np.int32)), Tensor.ones(B, K), Tensor(np.ones((5, 8), np.float32))).square().mean()
    loss.backward()
sd = bt2.state_dict(); idp = {id(v): k for k, v in sd.items()}
none = [idp.get(id(p), "?") for p in ps if p.grad is None]
check("every parameter gets a gradient (the two-terminal law)", not none, ", ".join(none[:6]))

# 9b. per-example temperature, constants, marks, a road subset
cx = BTM.BTConfig(**{**CFG, "n_const": 4, "d_mark": 1, "roads": CFG["roads"] + (BTM.RoadSpec("ptr_const", "slot", "const", 1, ((),), residual=False),)})
Tensor.manual_seed(0); bx = BTM.BreathingTransformer(cx); iox, _ = inputs(bx)
iox.valid["ptr_const"] = Tensor.ones(B, K, 4).cast("bool")
ox = bx.breath(iox, 0, tau=Tensor([2.0, 0.5]))
lc = ox["logp"]["ptr_const"].numpy()
check("const key stream: a pointer over the learned constants", lc.shape == (B, 1, K, 4) and np.allclose(np.exp(lc).sum(-1), 1, atol=1e-4))
e_ = lambda v: -(np.exp(v) * v).sum(-1).mean(axis=(1, 2))
ep = e_(ox["logp"]["pointer"].numpy())
check("per-example tau: the tau 0.5 example reads sharper than the tau 2.0 one", ep[1] < ep[0], f"{ep[0]:.3f} vs {ep[1]:.3f}")
o_nm = bx.breath(iox, 0, roads=("cross",))
iox.marks = {"token": Tensor(np.ones((B, LT, 1), np.float32))}
om = bx.breath(iox, 0, roads=("cross",))
check("a road subset runs only those roads; marks change the read", set(om["logp"]) == {"cross"}
      and np.abs(om["logp"]["cross"].numpy() - o_nm["logp"]["cross"].numpy()).max() > 1e-4)

# 9c. constants are never parameters, never saved, even after a forward built them
ks = list(bt2.state_dict())
check("no constant buffer in the parameters or the saved state", not [k for k in ks if k.endswith(("sel", "_own_t"))], str([k for k in ks if "own" in k][:3]))

# 10. the organs
t6 = O.T6Clock(span=6)
check("T6 odometer: 96 distinct readings per revolution", len({t6.odometer(t) for t in range(96)}) == 96)
t9 = O.T9Clock(); f9 = t9.fields(np.array([[0, 0, 0, 1, 1, 2]]))
check("T9 fields: offset restarts at each sentence", f9["off"].tolist() == [[0, 1, 2, 0, 1, 0]] and f9["pos"].tolist() == [[0, 1, 2, 3, 4, 5]])
check("T9 odometer: distinct addresses for distinct (offset, position)", len({t9.odometer(o, p) for o in range(6) for p in range(0, 3000, 200)}) == 90)
rng = np.random.default_rng(0); bus = O.PhaseBus(1, 256); keys = rng.uniform(0, 2 * np.pi, (3, 256)); book = rng.uniform(0, 2 * np.pi, (10, 256))
for j, v in enumerate((4, 7, 1)): bus.write(keys[j:j + 1], book[v:v + 1])
got = [int(bus.cleanup(keys[j:j + 1], book)[0][0]) for j in range(3)]
check("T256 bus: bind three pairs, unbind by the conjugate, clean up", got == [4, 7, 1], str(got))
with tempfile.TemporaryDirectory() as d:
    db = TreeDB(os.path.join(d, "t.db")); tt = O.TreeTables(db, "gen-test"); part = c.partition(0)
    tab = tt.table(part, C, 8); tab.update(rng.standard_normal((40, 8))); tab.update(rng.standard_normal((40, 8)) + 6); st = tab.state(); tt.save()
    tt2 = O.TreeTables(db, "gen-test"); st2 = tt2.table(part, C, 8).state()
    check("Tree tables: a partition's Welford charts round-trip through SQLite", all(np.allclose(a_, b_) for a_, b_ in zip(st, st2)))
    try: tt2.load_generation(part, "gen-other"); refused = False
    except ValueError: refused = True
    check("Tree tables: another generation's centroids are refused", refused)
    n = tt2.record(part, "b1", [("type:add", 1, "p0"), ("type:mul", 2, "p1")], certifier="answer-key")
    check("Tree tables: symbolic evidence lands provisional", n == 2 and db.batches()[0][1] == "provisional")
    db.close()


# 11. the alternation
class Port:
    def __init__(self, cert=True): self.cert = cert
    def io(self, bt_, organs, b):
        io_ = inputs(bt_)[0]
        if organs.tree is not None: io_.charts, io_.chart_ok = organs.tree.io(bt_); io_.key_charts = {}
        return io_
    def propose(self, bt_, out_, organs, b): return out_["logp"]["pointer"].numpy().argmax(-1)
    def certified(self, bt_, out_, organs, b): return {"token": np.ones((B, LT), bool)} if self.cert else {}


class Solver:
    def step(self, tier, props, notebook, tick):
        k = int(notebook.committed[0].sum())
        if k < notebook.K: notebook.write(0, k, atom=0, radius=1.0, cert=[1, 0, 0], tick=tick, consumes=(int(props[0, 0, k]),))
        return k


c_dt = BTM.BTConfig(**{**CFG, "tier": "data"})
org = O.Organs(notebook=O.Notebook(B, K, LT), t6=O.T6Clock(), t9=O.T9Clock())
tr = O.Conductor([(bt, Port()), (BTM.BreathingTransformer(c_dt), Port())], Solver()).run(org)
check("Conductor: every breath, every tier, coarse tier first; the clock ticks", len(tr) == 12 and [t[1] for t in tr[:2]] == ["business", "data"]
      and org.t6.t == 6 and org.notebook.committed[0].all())


# 12. THE TREE PARTITIONS: Welford + BIRCH (the BIC growth rule), (tier, head) partitions, live charts, certified writes only
rng = np.random.default_rng(5); t = O.ChartTable(4, 3); xs = []; ys = []
for j in range(5):
    x = rng.standard_normal((30, 3)) + (8 * (j % 2)); xs.append(x); ys.append(np.full(30, j % 2)); t.update(x)
X, Y = np.concatenate(xs), np.concatenate(ys); idx = t.nearest(X); live = np.flatnonzero(t.live)
pure = all(len(set(Y[idx == c])) == 1 for c in np.unique(idx))
check("BIRCH: two separated clouds -> exactly two live charts (the plateau, capacity 4 is a ceiling), each pure",
      len(live) == 2 and pure and len(np.unique(idx)) == 2, f"live {live.tolist()}, n {t.n[live].round(1).tolist()}")
r1 = np.random.default_rng(11); g1 = O.ChartTable(16, 8); g2 = O.ChartTable(16, 8); sc = np.array([6, 1, 1, 1, 1, 1, 1, 1.0])
for _ in range(20): g1.update(r1.standard_normal((100, 8))); g2.update(r1.standard_normal((100, 8)) * sc)
check("BIRCH: one Gaussian never splits (isotropic, and stretched 6x on one axis), 2000 points, capacity 16",
      g1.live.sum() == 1 and g2.live.sum() == 1, f"live {int(g1.live.sum())} / {int(g2.live.sum())}")
t6c = O.ChartTable(16, 8); r6 = np.random.default_rng(6); cen6 = r6.standard_normal((6, 8)) * 6; yy = []
first = cen6[r6.integers(0, 6, 300)] + r6.standard_normal((300, 8)); t6c.update(first)            # the root's first batch holds every cluster
for _ in range(10):
    y_ = r6.integers(0, 6, 200); yy.append(y_); t6c.update(cen6[y_] + r6.standard_normal((200, 8)))
y_ = r6.integers(0, 6, 600); a6 = t6c.nearest(cen6[y_] + r6.standard_normal((600, 8)))
ct = np.zeros((16, 6)); np.add.at(ct, (a6, y_), 1); pur6 = ct.max(1).sum() / ct.sum()
check("BIRCH grows past a root that absorbed every cluster and PLATEAUS at the structure: 6 clouds -> 6 live charts of 16, purity >= 0.95",
      t6c.live.sum() == 6 and pur6 >= 0.95, f"live {int(t6c.live.sum())}, purity {pur6:.3f}")
dm = O.ChartTable(8, 3, cap=60); r7 = np.random.default_rng(7)
for _ in range(6): dm.update(np.concatenate([r7.standard_normal((40, 3)) - 6, r7.standard_normal((40, 3)) + 6]))
n_before = int(dm.live.sum())
for sh in np.linspace(6, 0, 30).tolist() + [0] * 10:                                       # the clouds drift together
    dm.update(np.concatenate([r7.standard_normal((40, 3)) - sh, r7.standard_normal((40, 3)) + sh]))
check("BIRCH: charts that drift together merge back (2 -> 1 as two clouds converge)", n_before == 2 and dm.live.sum() == 1,
      f"live {n_before} -> {int(dm.live.sum())}")
ab = O.ChartTable(8, 3, max_idle=5); r8 = np.random.default_rng(8)
for _ in range(6): ab.update(np.concatenate([r8.standard_normal((40, 3)) - 6, r8.standard_normal((40, 3)) + 6]))
for _ in range(10): ab.update(r8.standard_normal((40, 3)) + 6)
check("max_idle retires a chart its source abandoned (2 -> 1); max_idle None keeps it", ab.live.sum() == 1, f"live {int(ab.live.sum())}")
# Welford exactness: one chart (capacity 1: BIRCH can only merge into it) fed in batches equals the direct mean / var
t2 = O.ChartTable(1, 3)
for x in xs: t2.update(x)
ok_w = np.allclose(t2.mean[0], X.mean(0)) and np.allclose(t2.var()[0], X.var(0))
check("Welford: batch-merged mean / variance equal the direct ones (Chan; capacity 1 = the root holds everything)", ok_w)
cf_ok = np.allclose(t2.LS[0], X.sum(0)) and np.isclose(t2.SS[0], (X ** 2).sum())
check("BIRCH CF derived exactly: LS = sum x, SS = sum |x|^2", cf_ok)
t3 = O.ChartTable(2, 3, cap=50)
for x in xs: t3.update(x)
check("the cap bounds memory (n <= cap) and keeps the variance", t3.n.max() <= 50 + 1e-9 and np.all(t3.var()[t3.live] > 0.3))

tiers = ("business", "data", "surface"); NH6 = 6
cfg6 = lambda tr: BTM.BTConfig(**{**CFG, "tier": tr, "n_partitions": NH6, "n_charts": 8, "roads": BTM.default_roads(n_heads=NH6, charts=True)})
Tensor.manual_seed(0); bts = [BTM.BreathingTransformer(cfg6(tr)) for tr in tiers]
with tempfile.TemporaryDirectory() as d:
    db = TreeDB(os.path.join(d, "t.db")); tt = O.TreeTables(db, "gen-smoke")
    for bt_ in bts:
        io_, _ = inputs(bt_); io_.charts, io_.chart_ok = tt.io(bt_); io_.key_charts = {}
        check(f"{bt_.c.tier}: a fresh partition has no live chart (the BIRCH root grows from certified data)", not io_.chart_ok.numpy().any()) if bt_ is bts[0] else None
        for j in range(4):
            io_, _ = inputs(bt_, seed=j); io_.charts, io_.chart_ok = tt.io(bt_); io_.key_charts = {}
            o_ = bt_.breath(io_, 1); tt.observe(bt_, o_["chart_pts"]["token"].numpy(), np.ones((B, LT), bool))
    parts = sorted(tt.tables)
    check("3 tiers x 6 heads = 18 distinct Tree partitions", len(parts) == 18 and len({p[0] for p in parts}) == 3, str(parts[:3]))
    cents = [tt.tables[p].centroids()[tt.tables[p].live] for p in parts]
    gap = min(np.abs(a_[:1] - b_[:1]).max() for i, a_ in enumerate(cents) for b_ in cents[i + 1:])
    check("every partition holds its own centroids (min pairwise gap of first charts)", all(len(x_) for x_ in cents) and gap > 1e-3, f"{gap:.3e}")
    io_, _ = inputs(bts[0], seed=9); io_.charts, io_.chart_ok = tt.io(bts[0]); io_.key_charts = {}
    o_ = bts[0].breath(io_, 1, roads=("cross",))
    pts = o_["chart_pts"]["token"].numpy(); okm = io_.chart_ok.numpy(); cen = io_.charts.numpy()
    kc_ = np.stack([np.where(okm[h][None, None], ((pts[:, h][:, :, None] - cen[h][None, None]) ** 2).sum(-1), np.inf).argmin(-1)
                    for h in range(NH6)], 1)
    io_.key_charts = {"cross": Tensor(kc_)}; o2_ = bts[0].breath(io_, 1, roads=("cross",))
    check("a key's chart = its silhouette's nearest LIVE centroid in its head's partition",
          np.allclose(o_["logp"]["cross"].numpy(), o2_["logp"]["cross"].numpy(), atol=1e-5))
    hm = 0; cen = np.random.default_rng(3).standard_normal(cen.shape).astype(np.float32) * 0.3; okm = np.ones_like(okm)
    dead = okm.copy(); dead[hm, 0] = False
    io_.charts = Tensor(cen); io_.chart_ok = Tensor(dead); io_.key_charts = {}
    o3_ = bts[0].breath(io_, 1, roads=("cross",)); pts3 = o3_["chart_pts"]["token"].numpy()
    kc3 = np.stack([np.where(dead[h][None, None], ((pts3[:, h][:, :, None] - cen[h][None, None]) ** 2).sum(-1), np.inf).argmin(-1)
                    for h in range(NH6)], 1)
    io_.key_charts = {"cross": Tensor(kc3)}; o4_ = bts[0].breath(io_, 1, roads=("cross",))
    check("a dead chart is never a key's chart (the internal snap skips charts BIRCH has not spawned)",
          okm[hm].sum() >= 2 and np.allclose(o3_["logp"]["cross"].numpy(), o4_["logp"]["cross"].numpy(), atol=1e-5), f"head {hm}: {dead[hm].sum()} live, chart 0 dead; keys on charts {np.unique(kc3[:, hm]).tolist()}")
    io_.chart_ok = Tensor(np.zeros_like(okm)); io_.key_charts = {}; o5_ = bts[0].breath(io_, 1, roads=("cross",))
    fl = BTM.BTConfig(**{**cfg6("business").__dict__, "roads": BTM.default_roads(n_heads=NH6, charts=False)})
    rd = bts[0].blocks[0].roads["cross"]; rd.spec, sp0 = fl.roads[2], rd.spec
    o6_ = bts[0].breath(io_, 1, roads=("cross",)); rd.spec = sp0
    p5 = np.exp(o5_["logp"]["cross"].numpy())
    check("an empty partition (fresh Tree) reads as one root chart = the flat softmax", np.allclose(p5.sum(-1), 1, atol=1e-4)
          and np.allclose(p5, np.exp(o6_["logp"]["cross"].numpy()), atol=1e-5))
    n_before = {p: tt.tables[p].n.sum() for p in parts}
    tt.save(); tt2 = O.TreeTables(db, "gen-smoke")
    same = all(np.allclose(tt2.table(p, 8, 8).centroids(), tt.tables[p].centroids()) for p in parts)
    check("all 18 partitions round-trip through SQLite (generation-tagged)", same)

    # the alternation writes ONLY certified points
    org = O.Organs(notebook=O.Notebook(B, K, LT), t6=O.T6Clock(), t9=O.T9Clock(), tree=tt)
    tt_n = lambda: sum(tt.tables[p].n.sum() for p in parts)
    n0 = tt_n(); O.Conductor([(bts[0], Port(cert=False)), (bts[1], Port(cert=False))], Solver()).run(org)
    n1 = tt_n(); org2 = O.Organs(notebook=O.Notebook(B, K, LT), t6=O.T6Clock(), t9=O.T9Clock(), tree=tt)
    O.Conductor([(bts[0], Port()), (bts[1], Port())], Solver()).run(org2); n2 = tt_n()
    check("Conductor: uncertified breaths never write the Tree; certified ones fold into their tier's partitions", n1 == n0 and n2 > n1,
          f"{n0:.0f} -> {n1:.0f} -> {n2:.0f}")
    check("the surface tier's partitions were untouched by the business/data run", all(tt.tables[p].n.sum() == n_before[p] for p in parts if p[0] == "surface"))
    db.close()

# 13. THE T256 BUS between tiers: per-slot keys, a tier salt, the BT's bus entry
busB = O.PhaseBus(B, 256); kb = O.PhaseBus.phases(np.broadcast_to(np.arange(K), (B, K)), 256, salt=0)
book = O.PhaseBus.phases(np.arange(5), 256, salt=7); typ = np.array([[3, 1, 4], [0, 2, 2]])
busB.write(kb, book[typ], radius=np.ones((B, K)))
got, sc = busB.cleanup(kb, book)
kd = O.PhaseBus.phases(np.broadcast_to(np.arange(K), (B, K)), 256, salt=1)
check("bus: each slot's binding unbinds at its own key (per-slot, conjugate) and cleans up", (got == typ).all(), str(got.tolist()))
check("bus: another tier's salt reads noise, not the binding", np.abs(busB.cleanup(kd, book)[1]).max() < 0.35 < sc.min(), f"{sc.min():.2f}")
cb = BTM.BTConfig(**{**CFG, "d_bus": 256}); Tensor.manual_seed(0); bb = BTM.BreathingTransformer(cb)
nbb = O.Notebook(B, K, LT); f_ = busB.features(kb)
h_a = bb.enter_slots(**nbb.tensors(), atoms=Tensor(np.ones((5, 8), np.float32))).numpy()
h_b = bb.enter_slots(**nbb.tensors(), atoms=Tensor(np.ones((5, 8), np.float32)), bus=Tensor(f_)).numpy()
check("the BT reads the bus: slot entries change with the bus read (2P features)", f_.shape == (B, K, 512) and np.abs(h_a - h_b).max() > 1e-3)

# 14. dropout (the regularization knob): training only; off = the no-dropout model exactly
Tensor.manual_seed(0); b0 = BTM.BreathingTransformer(BTM.BTConfig(**CFG)); Tensor.manual_seed(0); b5 = BTM.BreathingTransformer(BTM.BTConfig(**{**CFG, "dropout": 0.5}))
xd = Tensor(np.random.default_rng(3).standard_normal((B, LT, 16)).astype(np.float32))
e0, e5 = b0.enter_tokens(xd).numpy(), b5.enter_tokens(xd).numpy()
with Tensor.train(): e5t = b5.enter_tokens(xd).numpy()
check("dropout: eval = the no-dropout model exactly; training zeroes about p of the entry", np.array_equal(e0, e5) and 0.35 < (e5t == 0).mean() < 0.65,
      f"zeroed {(e5t == 0).mean():.2f}")

# 15. waist_at (11-08): where the queries squeeze. 'every' keeps the 11-07 form's weights exactly; 'last' with one block = 'every';
# 'end': no residual road reads the waist (perturbing it moves only the pointer road + the silhouette); the IDs still rejoin
def mk(**kw):
    Tensor.manual_seed(0); return BTM.BreathingTransformer(BTM.BTConfig(**{**CFG, **kw}))
be, bl1 = mk(), mk(waist_at="last")
check("waist_at 'every' = the 11-07 weights (no new parameters); 'last'/'end' add only the query LNs",
      set(be.state_dict()) == set(mk(waist_at="every").state_dict())
      and set(bl1.state_dict()) - set(be.state_dict()) == {k for k in bl1.state_dict() if k.startswith("qln.")} != set())
same_init = all(np.array_equal(v_.numpy(), bl1.state_dict()[k_].numpy()) for k_, v_ in be.state_dict().items())   # the LNs draw no rng
ioq, _ = inputs(be); oe, ol = be.breath(ioq, 0), bl1.breath(ioq, 0)
check("waist_at 'last' with one block = 'every' exactly", same_init and all(np.array_equal(oe["logp"][r].numpy(), ol["logp"][r].numpy()) for r in oe["logp"]))
bn = mk(waist_at="end"); ion, _ = inputs(bn); o0 = bn.breath(ion, 0)
wd0 = bn.waist["slot"].up.weight.numpy(); bn.waist["slot"].up.weight = Tensor(wd0 + 0.5).contiguous()
wt0 = bn.waist["token"].up.weight.numpy(); bn.waist["token"].up.weight = Tensor(wt0 + 0.5).contiguous()
o1_ = bn.breath(ion, 0)
res_same = all(np.array_equal(o0["logp"][r].numpy(), o1_["logp"][r].numpy()) for r in ("token_self", "slot_self", "cross"))
ptr_moved = not np.allclose(o0["logp"]["pointer"].numpy(), o1_["logp"]["pointer"].numpy())
check("waist_at 'end': the residual roads never read the waist; the pointer road does", res_same and ptr_moved)
hs_ = Tensor.zeros(B, K, CFG["d_model"]).contiguous(); ion.h = {**ion.h, "slot": hs_}
q2 = np.exp(bn.breath(ion, 0, roads=("slot_self",))["logp"]["slot_self"].numpy())
check("waist_at 'end': identical open slots still ask different queries (the ID skip rejoins full-width queries)",
      np.abs(q2[:, :, 0] - q2[:, :, 1]).max() > 1e-4, f"{np.abs(q2[:, :, 0] - q2[:, :, 1]).max():.2e}")

# 16. THE FUNNEL TIER's levers (11-08, Bryce's word): tied layers, clock-turned queries, shared K/V, shared pointer projections, the
# Tree consulted only at the squeeze
bu, bt2 = mk(blocks_per_breath=3), mk(blocks_per_breath=3, tie_layers=True)
r0, r1 = bt2.blocks[0].roads["cross"], bt2.blocks[1].roads["cross"]
np_u, np_t = sum(p.numel() for p in bu.parameters()), sum(p.numel() for p in bt2.parameters())
check("tie_layers: the blocks share q/k/v/o/turn, keep their own LNs + ID gains; parameters() lists each tensor once",
      r0.q is r1.q and r0.k is r1.k and r0.v is r1.v and r0.o is r1.o and r0.turn is r1.turn and r0.lnk is not r1.lnk and r0.gain is not r1.gain
      and len({id(p) for p in bt2.parameters()}) == len(bt2.parameters()) and np_t < np_u, f"params {np_u} -> {np_t}")
from tinygrad.nn.optim import AdamW
ps_ = bt2.parameters(); [setattr(p, "requires_grad", True) for p in ps_]; opt_ = AdamW(ps_, lr=1e-2); iot, _ = inputs(bt2)
with Tensor.train():
    opt_.zero_grad(); lo_ = -bt2.breath(iot, 0)["logp"]["pointer"].max(-1).sum(); lo_.backward()
    for p in ps_:
        if p.grad is None: p.grad = p.zeros_like()
    w_before = r0.q.weight.numpy().copy(); opt_.step()
check("tie_layers: one optimizer step moves the shared projection once (trains through every block)", not np.allclose(w_before, r0.q.weight.numpy()))
bc = mk(query_turn="clock", n_partitions=0, roads=(BTM.RoadSpec("slot_self", "slot", "slot", 2, (("slot_id",),) * 2),), id_gain_init=40.0)
check("query_turn 'clock': no content turn; one fixed rotation per head (H x P)", not hasattr(bc.blocks[0].roads["slot_self"], "turn")
      and bc.blocks[0].roads["slot_self"].hturn.shape == (2, bc.layout.P))
nbk = O.Notebook(B, K, LT); sa_, sp_ = bc.layout.angles(nbk.slot_fields(), (B, K)); sl, se = bc.layout.slices["slot_id"]
dlt = sa_[0, 1, sl:se] - sa_[0, 0, sl:se]; ht_ = np.zeros((2, bc.layout.P), np.float32); ht_[1, sl:se] = -dlt
bc.blocks[0].roads["slot_self"].hturn = Tensor(ht_).contiguous()
ioc = BTM.BreathIO(h={"slot": Tensor.zeros(B, K, CFG["d_model"]).contiguous()}, angles={"slot": Tensor(sa_)}, presence={"slot": Tensor(sp_)},
                   valid={"slot_self": Tensor(np.ones((B, K, K), bool))})
pk = bc.breath(ioc, 0)["logp"]["slot_self"].numpy().argmax(-1)                                   # (B, H, K)
check("query_turn 'clock': the gears turn the query -- head 0 reads its own slot, head 1 (turned one slot back) reads slot k-1",
      (pk[:, 0] == np.arange(K)).all() and (pk[:, 1, 1:] == np.arange(K - 1)).all(), str(pk[0].tolist()))
bk = mk(shared_kv=True); rk = bk.blocks[0].roads["cross"]; iok, _ = inputs(bk); ok_ = bk.breath(iok, 0)["logp"]
check("shared_kv: one key/value head per road (D x d_head), every head still its own query and distribution",
      rk.k.weight.shape == (CFG["d_head"], CFG["d_model"]) and rk.v.weight.shape == (CFG["d_head"], CFG["d_model"])
      and np.allclose(np.exp(ok_["cross"].numpy()).sum(-1), 1, atol=1e-4) and not np.allclose(ok_["cross"].numpy()[:, 0], ok_["cross"].numpy()[:, 1]))
rds = (BTM.RoadSpec("pa", "slot", "token", 1, (("t9",),), residual=False, share_q="a", share_k="t"),
       BTM.RoadSpec("pb", "slot", "token", 1, (("t9",),), residual=False, share_q="b", share_k="t"),
       BTM.RoadSpec("pa2", "slot", "token", 1, (("t9",),), residual=False, share_q="a", share_k="t2"))
bp = mk(roads=rds, n_partitions=0); R_ = bp.blocks[0].roads
check("share_q / share_k: the shared pointer -- one query per operand, one key per key kind",
      R_["pa"].q is R_["pa2"].q and R_["pa"].q is not R_["pb"].q and R_["pa"].k is R_["pb"].k and R_["pa"].k is not R_["pa2"].k)
calls = []
for wa, nbl in (("every", 2), ("last", 2)):
    bw = mk(blocks_per_breath=nbl, waist_at=wa); f0 = bw._charts
    def spy(*a_, _f=f0, **k_): calls.append(wa); return _f(*a_, **k_)
    bw._charts = spy; iow, _ = inputs(bw); bw.breath(iow, 0)
check("the Tree is consulted only where the query squeezes: waist_at 'last' reads charts in the last block only",
      calls.count("every") == 5 and calls.count("last") == 3, f"every {calls.count('every')} / last {calls.count('last')}")

# -- the FIT-SPEED forms (ledger 11-08 audit) --
rq = (BTM.RoadSpec("pa", "slot", "token", 1, (("t9",),), residual=False, d_head=CFG["d_model"]),
      BTM.RoadSpec("pu", "slot", "token", 1, (("t9",),), residual=False, d_head=CFG["d_model"], q_scale=1.0))
bq = mk(roads=rq, n_partitions=0, id_gain_init=0.0); Rq = bq.blocks[0].roads; Rq["pu"].q, Rq["pu"].k = Rq["pa"].q, Rq["pa"].k
ioq, _ = inputs(bq); ioq.valid["pa"] = ioq.valid["pu"] = ioq.valid["cross"]; oq = bq.breath(ioq, 0, tau=1.0)["logp"]
vq_ = ioq.valid["cross"].numpy()[:, None]; fa = oq["pa"].numpy() - oq["pa"].numpy().max(-1, keepdims=True); fu = oq["pu"].numpy() - oq["pu"].numpy().max(-1, keepdims=True)
check("RoadSpec.q_scale=1.0: the unscaled dot product (= the default's logits x sqrt(d_head), same weights)",
      np.allclose(np.exp(oq["pu"].numpy()).sum(-1), 1, atol=1e-4) and np.isfinite(fu).all()
      and np.allclose(*[np.where(vq_, f_ - np.where(vq_, f_, 0).sum(-1, keepdims=True) / vq_.sum(-1, keepdims=True), 0) * w_
                        for f_, w_ in ((fu, 1.0), (fa, math.sqrt(CFG["d_model"])))], atol=1e-2))
NA_, G_ = 5, 2; bz = mk(n_atoms=NA_, n_coarse=G_); par_ = np.zeros((G_, NA_), np.float32); par_[[0, 0, 1, 1, 1], np.arange(NA_)] = 1
zz_ = Tensor.randn(B, K, CFG["d_waist"]); at_ = Tensor.randn(NA_, CFG["d_waist"]); co_ = Tensor.randn(G_, CFG["d_waist"])
lz = bz.silhouette_zoom_logp(zz_, at_, co_, Tensor(par_), 1.0).numpy(); pz = np.exp(lz)
pg = pz @ par_.T; lc_ = bz.silhouette_logp(zz_, co_, 1.0).numpy()
check("ZOOM: log p(type) = log p(op) + log p(type | op) -- a distribution whose op marginal is the coarse classifier",
      np.allclose(pz.sum(-1), 1, atol=1e-4) and np.allclose(pg, np.exp(lc_), atol=1e-4))

p2_ = np.zeros((1, G_), np.float32); p2_[0] = 1; bz3 = mk(n_atoms=NA_, n_coarse=G_, n_coarse2=1)
c2_ = Tensor.randn(1, CFG["d_waist"]); bz3.sil_scale_c = bz.sil_scale_c; bz3.sil_scale = bz.sil_scale
lz3 = bz3.silhouette_zoom_logp(zz_, at_, co_, Tensor(par_), 1.0, c2_, Tensor(p2_)).numpy()
check("ZOOM3: three levels by the chain rule -- a distribution; one grandparent over everything = the two-level zoom",
      np.allclose(np.exp(lz3).sum(-1), 1, atol=1e-4) and np.allclose(lz3, lz, atol=1e-4))

# -- MULTIPLICATIVE BINDING (ledger 11-08): content q / k rotate by the matched clock angles --
import dataclasses
rb_ = tuple(dataclasses.replace(r_, charts=False) for r_ in BTM.default_roads(n_heads=2))
bu_, bb_ = mk(roads=rb_, n_partitions=0), mk(roads=tuple(dataclasses.replace(r_, bind=True) for r_ in rb_), n_partitions=0)
check("bind adds no parameters (a constant plane map per road)", set(bu_.state_dict()) == set(bb_.state_dict()))
wdiff = max(float(np.abs(v_.numpy() - bb_.state_dict()[k_].numpy()).max()) for k_, v_ in bu_.state_dict().items())
for k_, v_ in bu_.state_dict().items(): bb_.state_dict()[k_].assign(v_.numpy()).realize()      # the same weights, by construction
iob, _ = inputs(bb_); [iob.valid.__setitem__(k_, iob.valid["cross"]) for k_ in ("pointer",)]
z0 = {s_: Tensor.zeros(*iob.angles[s_].shape) for s_ in iob.angles}
io0 = dataclasses.replace(iob, angles=z0)
ou0, ob0 = bu_.breath(io0, 0, tau=1.0)["logp"], bb_.breath(io0, 0, tau=1.0)["logp"]
check("bind with every stamp at 0 = the unbound road exactly", all(np.allclose(ou0[r_].numpy(), ob0[r_].numpy(), atol=1e-5) for r_ in ou0),
      f"init weight diff {wdiff:.1e}; " + " ".join(f"{r_} {np.abs(ou0[r_].numpy() - ob0[r_].numpy()).max():.1e}" for r_ in ou0))
ou1, ob1 = bu_.breath(iob, 0, tau=1.0)["logp"], bb_.breath(iob, 0, tau=1.0)["logp"]
check("bind with live stamps moves the content match (not a no-op)", not np.allclose(ou1["cross"].numpy(), ob1["cross"].numpy(), atol=1e-3))
rs_ = bb_.blocks[0].roads["slot_self"]; u_ = Tensor.randn(B, K, CFG["d_model"]); hk_ = Tensor.randn(B, K, CFG["d_model"])
sa_, sp_ = iob.angles["slot"], iob.presence["slot"]; vs_ = Tensor(np.ones((B, K, K), bool))
l0_, l1_ = (rs_.logp(u_, hk_, sa_ + d_, sa_ + d_, sp_, vs_, 1.0).numpy() for d_ in (0.0, 0.7))
check("bind is RELATIVE (the road's score): one common turn of query and key stamps leaves the match unchanged",
      np.allclose(l0_, l1_, atol=1e-4), f"{np.abs(l0_ - l1_).max():.1e}")
M_ = BTM._bind_map(bb_.layout.selector((("t9", "token_id"),)), np.array([bb_.layout.factors[i].kind == "clock" for i in bb_.layout.owner()]), 4)
t9a, t9b = bb_.layout.slices["t9"]
check("bind turns content by CLOCK planes only (the word-hash id stays additive)",
      M_.sum(-1)[0, t9a:t9b].sum() == 4 and M_[0, slice(*bb_.layout.slices["token_id"])].sum() == 0)
print(f"\n{sum(OK)}/{len(OK)} PASS"); sys.exit(0 if all(OK) else 1)
