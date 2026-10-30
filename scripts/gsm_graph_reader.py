"""gsm_graph_reader.py -- THE GRAPH ARM (ledger 2026-10-29 20:29, Bryce "word given"): idea C as a WALK over the problem's own graph,
routed by phase switches, on top of NOTEBOOK (gsm_trunk_reader.py imported UNCHANGED as RD_ARM=NOTEBOOK: same trunk states, same
alternation, same training loop settings).
  tags    : owner / event ids (.cache/gsmb_graph_keys.json) -> FRESH RANDOM PHASES per problem, added (Linear([cos, sin])) to the
            input of numeral words and of their MENTION words; kind -> an embedding of the k=3 forest class (the tree channel)
  walk    : per slot and operand an owner head and an event head emit a phase code; dot product with the problem's owner / event
            codes -> choice; the numeral pointer is MASKED to numerals whose (owner, event) match (unknown = wildcard; earlier slots
            and constants always open) = the switches
  train   : the teacher's walk = the key of each operand's first gold numeral member (CE owner + CE event + the pointer loss under
            the gold mask)
  score   : the same model three ways -- LEARNED (the walk chooses), ORACLE (the gold key opens the switch), OPEN (no switches)
Output .cache/gsmb_graph_{TAG}.json."""
import json, math, os, sys, time

os.environ["RD_ARM"] = "NOTEBOOK"
SEED = int(os.environ.get("RD_SEED", "0")); os.environ.setdefault("RD_TAG", f"GRAPH_s{SEED}")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
import gsm_trunk_reader as R

NG, PB, K = 12, R.PB, R.K
MODES = ("LEARNED", "ORACLE", "OPEN")


def load_keys(Dt):
    G = json.load(open(GD.V1.C("gsmb_graph_keys.json")))
    for sp, its in Dt.items():
        for it, g in zip(its, G["keys"][sp]):
            assert len(g["own"]) == len(it["toks"]), sp
            it["g"] = g
    return G["n_kind"]


def gold_key(it, S):
    """the (owner, event) of the first gold numeral member of a valid set, or (-1, -1)."""
    if not S["t"]: return -1, -1
    j = S["t"][0]; return it["g"]["own"][j], it["g"]["ev"][j]


def gbatch(its, t, rng):
    """graph fields for a batch laid out by R.batch (question words at 0..n-1)."""
    import torch
    B, L = t["pad"].shape; f = {k: np.full((B, L), -1) for k in ("own", "ev", "kind", "tago", "tage")}
    gk = np.full((B, K, 2, 2), -1)                                            # slot, operand (a/b), (owner, event)
    for b, it in enumerate(its):
        g = it["g"]; n = len(it["toks"])
        f["own"][b, :n] = g["own"]; f["ev"][b, :n] = g["ev"]; f["kind"][b, :n] = g["kind"]
        f["tago"][b, :n] = [o if o >= 0 else m for o, m in zip(g["own"], g["own_m"])]
        f["tage"][b, :n] = [e if e >= 0 else m for e, m in zip(g["ev"], g["ev_m"])]
        for k, (_, A, Bs) in enumerate(it["prog"][:K]):
            gk[b, k, 0] = gold_key(it, A); gk[b, k, 1] = gold_key(it, Bs)
    out = {k: torch.as_tensor(np.minimum(v, NG - 1) if k != "kind" else v) for k, v in f.items()}
    out["gk"] = torch.as_tensor(np.minimum(gk, NG - 1))
    out["tho"] = torch.as_tensor(rng.random((B, NG, PB)) * 2 * math.pi, dtype=torch.float32)
    out["the"] = torch.as_tensor(rng.random((B, NG, PB)) * 2 * math.pi, dtype=torch.float32)
    return out


def make_graph_net(n_kind):
    import torch, torch.nn as nn
    D = R.D

    class Inp(nn.Module):                                                     # R's input + the graph tags (set per call)
        def __init__(s, base):
            super().__init__(); s.base = base; s.extra = None
        def forward(s, x): return s.base(x) + s.extra

    class Cap(nn.Module):                                                     # R's final LayerNorm, output captured for the walk heads
        def __init__(s, base):
            super().__init__(); s.base = base; s.out = None
        def forward(s, x): s.out = s.base(x); return s.out

    class GNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = R.make_net(); s.net.inp = Inp(s.net.inp); s.net.ln = Cap(s.net.ln)
            s.tpo = nn.Linear(2 * PB, D); s.tpe = nn.Linear(2 * PB, D); s.kemb = nn.Embedding(n_kind + 1, D)
            s.heads = nn.Linear(D, 4 * 2 * PB)                                # owner a, owner b, event a, event b codes
            s.temp = nn.Parameter(torch.tensor(3.0))

        def codes(s, th, ids):
            c = torch.cat([th.cos(), th.sin()], -1)                           # (B, NG, 2PB)
            return torch.gather(c, 1, ids.clamp(min=0)[..., None].expand(-1, -1, 2 * PB)), c

        def forward(s, t, g, mode, gkey=None):
            """-> op, pa, pb (switch-masked per mode), owner/event logits (B, K, 2, NG) x 2."""
            co, Co = s.codes(g["tho"], g["tago"]); ce, Ce = s.codes(g["the"], g["tage"])
            s.net.inp.extra = (s.tpo(co) * (g["tago"] >= 0)[..., None] + s.tpe(ce) * (g["tage"] >= 0)[..., None]
                               + s.kemb(g["kind"] + 1))
            op, pa, pb = s.net(t)
            B, L = t["pad"].shape; h = torch.gather(s.net.ln.out, 1, t["slot"][..., None].expand(-1, -1, D))
            q = s.heads(h).reshape(B, K, 4, 2 * PB)
            ar = torch.arange(NG)                                             # (a scatter of the -1 rows would overwrite id 0)
            present_o = (g["own"][..., None] == ar).any(1); present_e = (g["ev"][..., None] == ar).any(1)
            lo = torch.einsum("bkxd,bnd->bkxn", q[:, :, :2], Co) / math.sqrt(PB) * s.temp
            le = torch.einsum("bkxd,bnd->bkxn", q[:, :, 2:], Ce) / math.sqrt(PB) * s.temp
            lo = lo.masked_fill(~present_o[:, None, None], -1e4); le = le.masked_fill(~present_e[:, None, None], -1e4)
            if mode != "OPEN":
                if mode == "LEARNED":
                    ko = torch.where(present_o.any(1)[:, None, None], lo.argmax(-1), -1)
                    ke = torch.where(present_e.any(1)[:, None, None], le.argmax(-1), -1)
                else:                                                         # ORACLE / teacher: the gold key
                    ko, ke = gkey[..., 0], gkey[..., 1]
                own, ev = g["own"][:, None, None, :], g["ev"][:, None, None, :]     # (B, 1, 1, L)
                ok = (((own == ko[..., None]) | (own < 0) | (ko[..., None] < 0)) &
                      ((ev == ke[..., None]) | (ev < 0) | (ke[..., None] < 0)))     # (B, K, 2, L)
                pa = torch.cat([pa[..., :L].masked_fill(~ok[:, :, 0], -1e4), pa[..., L:]], -1)
                pb = torch.cat([pb[..., :L].masked_fill(~ok[:, :, 1], -1e4), pb[..., L:]], -1)
            return op, pa, pb, lo, le
    return GNet()


def gloss(out, t, g):
    import torch.nn.functional as F
    op, pa, pb, lo, le = out; lo_ = R.loss_fn((op, pa, pb), t); am = t["am"]
    for lg, j in ((lo, 0), (le, 1)):
        tgt = g["gk"][..., j]; m = am[..., None] & (tgt >= 0)
        if m.any(): lo_ = lo_ + F.cross_entropy(lg[m], tgt[m])
    return lo_


def score(net, its, states, tree, mode, seed=0):
    """R.score's NOTEBOOK path (sequential, one slot per pass, same ID phases every pass) with the walk."""
    import torch
    res = dict(ans=[], step_all=[], owner_ok=[], event_ok=[])
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]; progs = [[] for _ in part]; done = [False] * len(part)
            for k in range(K):
                rng = np.random.default_rng(10_000 * seed + s0)
                t, E = R.batch(part, states, tree, rng, progs); g = gbatch(part, t, rng)
                op, pa, pb, lo, le = net(t, g, mode, g["gk"]); L = t["pad"].shape[1]
                po = op.argmax(-1).numpy(); ia = pa.argmax(-1).numpy(); ib = pb.argmax(-1).numpy()
                dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
                if k == 0: PO, IA, IB = po.copy(), ia.copy(), ib.copy(); LO, LE = lo.argmax(-1).numpy(), le.argmax(-1).numpy(); GK = g["gk"].numpy()
                PO[:, k], IA[:, k], IB[:, k] = po[:, k], ia[:, k], ib[:, k]
                LO[:, k], LE[:, k] = lo.argmax(-1).numpy()[:, k], le.argmax(-1).numpy()[:, k]
                for b in range(len(part)):
                    if done[b]: continue
                    if R.OPS[po[b, k]] == ".": done[b] = True; continue
                    progs[b].append((R.OPS[po[b, k]], dec(ia[b, k]), dec(ib[b, k])))
            for b, (it, e) in enumerate(zip(part, E)):
                v = GD.execute(progs[b], it["vals"] + [None] * (L - len(it["vals"])))
                res["ans"].append(v is not None and GD.V1.near(v, it["final"]))
                for k in range(e["m"]):
                    res["step_all"].append(bool(PO[b, k] == e["op"][k] and t["va"][b, k, IA[b, k]] and t["vb"][b, k, IB[b, k]]))
                    for x in range(2):
                        if GK[b, k, x, 0] >= 0: res["owner_ok"].append(bool(LO[b, k, x] == GK[b, k, x, 0]))
                        if GK[b, k, x, 1] >= 0: res["event_ok"].append(bool(LE[b, k, x] == GK[b, k, x, 1]))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in res.items()} | {"per": [int(x) for x in res["ans"]]}


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8"))); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    Dt, states = R.load(); tree = R.load_tree(); n_kind = load_keys(Dt); train = Dt["TRAIN"]
    net = make_graph_net(n_kind); opt = torch.optim.AdamW(net.parameters(), lr=R.LR); npar = sum(p.numel() for p in net.parameters())
    print(f"[graph] {R.TAG} train {len(train)} | params {npar}", flush=True); t0 = time.time(); curve = []
    for step in range(1, R.STEPS + 1):
        its = [train[int(i)] for i in rng.integers(0, len(train), R.BATCH)]
        t, _ = R.batch(its, states, tree, rng); g = gbatch(its, t, rng)
        loss = gloss(net(t, g, "TEACH", g["gk"]), t, g)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == R.STEPS:
            net.eval(); r = {m: score(net, Dt["TEST"][:300], states, tree, m) for m in MODES}; net.train()
            curve.append(dict(step=step, loss=float(loss.detach()), **{f"TEST_{m}_{k}": v[k] for m, v in r.items() for k in ("step_all", "owner_ok")}))
            print(f"[graph] {R.TAG} step {step} loss {float(loss.detach()):.4f} " +
                  " ".join(f"{m} {v['step_all']:.3f}" for m, v in r.items()) + f" owner {r['LEARNED']['owner_ok']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval(); out = dict(tag=R.TAG, seed=SEED, params=npar, steps=R.STEPS, curve=curve)
    for sp in R.SPLITS:
        for m in MODES:
            r = score(net, Dt[sp], states, tree, m); out[f"{sp}_{m}"] = {k: v for k, v in r.items() if k != "per"}
            print(f"[graph] {R.TAG} FINAL {sp} {m} ans {r['ans']:.3f} step {r['step_all']:.3f} owner {r['owner_ok']:.3f} event {r['event_ok']:.3f}", flush=True)
    json.dump(out, open(f".cache/gsmb_graph_{R.TAG}.json", "w"))
