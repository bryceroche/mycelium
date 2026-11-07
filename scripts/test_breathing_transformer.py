"""test_breathing_transformer.py -- the spec of the official BT (ledger 2026-11-07 "THE TIERED BT ARCHITECTURE"), checked mechanically.
CPU, tiny dims, no data. Run: DEV=CPU .venv/bin/python3 scripts/test_breathing_transformer.py
"""
import os, re, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tinygrad import Tensor

from mycelium import breathing_transformer as BTM
from mycelium import bt_organs as O
from mycelium.tree_db import TreeDB

B, LT, K, C = 2, 7, 3, 3
CFG = dict(tier="business", d_in=16, d_model=32, d_waist=8, d_head=8, d_chart=8, d_cert=3, n_atoms=5, n_breaths=6,
           roads=BTM.default_roads(n_heads=2, n_charts=C))
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
    charts = {r: [Tensor(rng.standard_normal((C, 8)).astype(np.float32)) for _ in range(2 if r == "cross" else 1)] for r in ("cross", "pointer")}
    io = BTM.BreathIO(h={"token": ht, "slot": hs}, angles={"token": Tensor(ta), "slot": Tensor(sa)},
                      presence={"token": Tensor(tp), "slot": Tensor(spr)},
                      valid={"token_self": vt, "slot_self": vs, "cross": vc, "pointer": vc},
                      charts=charts, key_charts={"cross": kc, "pointer": kc[:, :1]})
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
check("each head reads its own Tree partition", len({c.partition(r.name, h) for r in c.roads for h in range(r.n_heads)})
      == sum(r.n_heads for r in c.roads))

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
    hs2 = bt2.enter_slots(**nbx.tensors(), belief=nb_belief, atoms=Tensor(np.ones((5, 8), np.float32)))
    loss = sum(v.exp().mul(v.clip(-50, 0)).sum() for v in o2["logp"].values()) + o2["h"]["token"].square().mean() \
        + o2["h"]["slot"].square().mean() + o2["z"]["slot"].square().mean() + hs2.square().mean()
    loss.backward()
sd = bt2.state_dict(); idp = {id(v): k for k, v in sd.items()}
none = [idp.get(id(p), "?") for p in ps if p.grad is None]
check("every parameter gets a gradient (the two-terminal law)", not none, ", ".join(none[:6]))

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
    db = TreeDB(os.path.join(d, "t.db")); tt = O.TreeTables(db, "gen-test"); part = c.partition("cross", 0)
    tab = tt.table(part, C, 8); tab.update(rng.standard_normal((20, 8)), rng.integers(0, C, 20)); cen = tab.centroids(); tt.save()
    tt2 = O.TreeTables(db, "gen-test")
    check("Tree tables: a partition's centroids round-trip through SQLite", np.allclose(tt2.table(part, C, 8).centroids(), cen, atol=1e-6))
    try: tt2.load_generation(part, "gen-other"); refused = False
    except ValueError: refused = True
    check("Tree tables: another generation's centroids are refused", refused)
    n = tt2.record(part, "b1", [("type:add", 1, "p0"), ("type:mul", 2, "p1")], certifier="answer-key")
    check("Tree tables: symbolic evidence lands provisional", n == 2 and db.batches()[0][1] == "provisional")
    db.close()


# 11. the alternation
class Port:
    def io(self, bt_, organs, b): return inputs(bt_)[0]
    def propose(self, bt_, out_, organs, b): return out_["logp"]["pointer"].numpy().argmax(-1)


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

print(f"\n{sum(OK)}/{len(OK)} PASS"); sys.exit(0 if all(OK) else 1)
