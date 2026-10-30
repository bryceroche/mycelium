"""gsm_trunk_reader.py -- STEP 0 of the GSM8K reader bench (ledger 2026-10-29): the reader arms on FROZEN Llama-3.2-1B L0-L3 states
(.cache/gsmb_trunk_states.npy, scripts/gsm_trunk_states.py), the ladder rebuilt under THE NO-MLP RULING. Task, gold, executor and
splits are gsm_reader.py's (the pointer program; .cache/gsmb_data.json). A word's input = Linear(LayerNorm(its trunk state)).
  VANILLA   CONTROL ONLY (the ruling's benchmark exception): learned absolute positions, pre-LN MHA + MLP blocks
  BASE      attention-only blocks (no MLP), learned absolute positions, width raised to VANILLA's parameter count (RD_D)
  ID        BASE + a PERSON-name word's input is an ID TAG only (fresh random phases per name per question); the trunk state of the
            name word is dropped (context words still carry what Llama attended to -- the leak is measured, not zero)
  TREE      ID + the name's gender from its SQLite tree node (.cache/gsmb_tree.db)
  CLOCK     TREE without absolute positions: offset-in-sentence embedding + q/k rotated by the T6 gears x sentence index
  NOTEBOOK  CLOCK + THE ALTERNATION (reader <-> executor; ruling 10-29): slots are decoded ONE AT A TIME; after each step the executor
            runs it and writes back (a) a CONSUMED bit on every operand it used (question numeral or earlier step) and (b) the step's
            certificate (executed / integer / non-negative) + its op onto the next slot; never a value. Question words do not attend
            to slots (their reading is fixed while the program is written); slot k sees slots <= k. The consumed bits enter every
            block's slot-row attention (a learned per-head bias) and the pointer logits (a learned bias). Training = teacher forcing
            on the gold program (consumed = the first member of each gold valid set); scoring = sequential greedy decoding.
  TICK      NOTEBOOK + A CLOCK TICK PER STEP (ledger 2026-10-30, Bryce "T6 clock to the rescue"): slot k sits at tick S + 1 + k (S =
            the last sentence) instead of every slot at S + 1, and the step pointer's query / key are rotated by the T6 gears x
            their slots' ticks, so "the result one step back" is a fixed relative rotation. No new weights.
Output .cache/gsmb_trunk_{TAG}.json."""
import json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
from mycelium import rotor_clock as CK

ARM = os.environ.get("RD_ARM", "VANILLA"); SEED = int(os.environ.get("RD_SEED", "0")); TAG = os.environ.get("RD_TAG", f"{ARM}_s{SEED}")
STEPS = int(os.environ.get("RD_STEPS", "6000")); BATCH = int(os.environ.get("RD_BATCH", "64")); LR = float(os.environ.get("RD_LR", "3e-4"))
ARMS = ("VANILLA", "BASE", "ID", "TREE", "CLOCK", "NOTEBOOK", "TICK")
D_DEFAULT = {"VANILLA": 128}                                      # every MLP-free arm: D=192 (BASE 1,197,765 vs VANILLA 1,163,909)
D = int(os.environ.get("RD_D", D_DEFAULT.get(ARM, 192))); NH = 4; NL = int(os.environ.get("RD_NL", "4")); PB = 64; NNAME = 12; OMAX = 64
K = GD.KMAX; LMAX = GD.LQMAX + K; NC = len(GD.CONSTS); OPS = GD.OPS + "."; H = 2048
USE_MLP = ARM == "VANILLA"; USE_ID = ARM in ("ID", "TREE", "CLOCK", "NOTEBOOK", "TICK"); USE_TREE = ARM in ("TREE", "CLOCK", "NOTEBOOK", "TICK")
USE_CLOCK = ARM in ("CLOCK", "NOTEBOOK", "TICK"); NOTE = ARM in ("NOTEBOOK", "TICK"); TICK = ARM == "TICK"
SPLITS = ("ORIG", "CFNUM", "CFNAME", "TEST")
TREE_DB = ".cache/gsmb_tree.db"


def load():
    Dt = json.load(open(GD.OUT)); idx = json.load(open(GD.V1.C("gsmb_trunk_index.json")))
    for sp, its in Dt.items():
        for it, s in zip(its, idx[sp]):
            it["st0"] = s
            nl = {n.lower(): n for n in it["names"]}; it["surf"] = [nl.get(x, x) if f else x for x, f in zip(it["toks"], it["isn"])]
    return Dt, np.load(GD.V1.C("gsmb_trunk_states.npy"), mmap_mode="r")


def load_tree():
    from mycelium import tree_db as TDB
    db = TDB.TreeDB(TREE_DB); L = db.load_leaves(); db.close()
    return {k: np.asarray(v[1], np.float32) for k, v in L.items()}


def first(S):
    return ("t", S["t"][0]) if S["t"] else (("s", S["s"][0]) if S["s"] else ("c", S["c"][0]))


def cert(v):
    """the executor's certificate of a step result: executed / integer / non-negative (never the value)."""
    try: v = None if v is None else float(v)                            # an all-constant chain is a Python int: may exceed int64 / float
    except OverflowError: v = None
    if v is None or not np.isfinite(v): return (0.0, 0.0, 0.0)
    return (1.0, float(GD.V1.isint(v)), float(v >= -1e-9))


def run_steps(prog, vals):
    out = []
    for op, a, b in prog:
        try:
            av, bv = [vals[i] if k == "t" else (out[i] if k == "s" else GD.CONSTS[i]) for k, i in (a, b)]
            if av is None or bv is None or (op == "/" and bv == 0): out.append(None); continue
            out.append(av + bv if op == "+" else av - bv if op == "-" else av * bv if op == "*" else av / bv)
        except (IndexError, TypeError):
            out.append(None)
    return out


def notebook_feats(prog, vals, n):
    """teacher/decoded program -> per-slot write-back: consumed (K x [n tokens | K slots]) and slot input (prev op, certificate)."""
    cons = np.zeros((K, n + K), bool); pop = np.full(K, len(OPS)); cf = np.zeros((K, 3), np.float32)
    res = run_steps(prog, vals)
    for k, (op, a, b) in enumerate(prog[:K - 1]):
        cons[k + 1:] |= cons[k]
        for kind, i in (a, b):
            if kind == "t" and i < n: cons[k + 1:, i] = True
            elif kind == "s": cons[k + 1:, n + i] = True
        pop[k + 1] = OPS.index(op); cf[k + 1] = cert(res[k])
    return cons, pop, cf


def encode(it, states, tree):
    n = len(it["toks"]); L = n + K
    x = np.zeros((L, H), np.float32); x[:n] = states[it["st0"]:it["st0"] + n]
    isn = np.array(it["isn"] + [False] * K); isnum = np.array([v is not None for v in it["vals"]] + [False] * K)
    sid = np.array(it["sid"] + ([it["sid"][-1] + 1 + k for k in range(K)] if TICK else [it["sid"][-1] + 1] * K)); off = np.zeros(L, np.int64)
    for i in range(1, n): off[i] = off[i - 1] + 1 if sid[i] == sid[i - 1] else 0
    off[n:] = np.arange(K)
    nm = list(dict.fromkeys(t for t, f in zip(it["surf"], it["isn"]) if f))
    nid = np.array([min(nm.index(t), NNAME - 1) if f else -1 for t, f in zip(it["surf"], it["isn"])] + [-1] * K)
    g = np.zeros((L, 3), np.float32)
    for i, (t, f) in enumerate(zip(it["surf"], it["isn"])):
        if f and t in tree: g[i, :2] = tree[t]; g[i, 2] = 1
    P = it["prog"]; m = min(len(P), K)
    op = np.array([OPS.index(p[0]) for p in P[:K]] + ([OPS.index(".")] if len(P) < K else []))
    va = np.zeros((K, LMAX + K + NC), bool); vb = np.zeros_like(va)
    for k, (_, A, B) in enumerate(P[:K]):
        for V, S in ((va, A), (vb, B)):
            V[k, S["t"]] = True; V[k, [LMAX + j for j in S["s"]]] = True; V[k, [LMAX + K + c for c in S["c"]]] = True
    teach = [(o, first(A), first(B)) for o, A, B in P[:K]]
    return dict(x=x, isn=isn, isnum=isnum, sid=sid, off=np.minimum(off, OMAX - 1), nid=nid, g=g, n=n, op=op, va=va, vb=vb, m=m, teach=teach)


def batch(its, states, tree, rng, progs=None):
    """progs: per item the program whose write-back the NOTEBOOK slots read (None = the teacher's gold program)."""
    import torch
    E = [encode(it, states, tree) for it in its]; B = len(E); L = max(e["n"] for e in E) + K
    out = dict(x=np.zeros((B, L, H), np.float32), isn=np.zeros((B, L), bool), isnum=np.zeros((B, L), bool), sid=np.zeros((B, L), np.int64),
               off=np.zeros((B, L), np.int64), nid=np.full((B, L), -1), g=np.zeros((B, L, 3), np.float32), pad=np.ones((B, L), bool),
               slot=np.zeros((B, K), np.int64), op=np.zeros((B, K), np.int64), om=np.zeros((B, K), bool), am=np.zeros((B, K), bool),
               va=np.zeros((B, K, L + K + NC), bool), vb=np.zeros((B, K, L + K + NC), bool),
               cons=np.zeros((B, K, L), bool), pop=np.full((B, K), len(OPS)), cf=np.zeros((B, K, 3), np.float32))
    for b, (it, e) in enumerate(zip(its, E)):
        n = e["n"]; idx = list(range(n)) + list(range(L - K, L))
        for k in ("x", "isn", "isnum", "sid", "off", "nid", "g"): out[k][b, idx] = e[k]
        out["pad"][b, idx] = False; out["slot"][b] = np.arange(L - K, L)
        out["op"][b, :len(e["op"])] = e["op"]; out["om"][b, :len(e["op"])] = True; out["am"][b, :e["m"]] = True
        for V in ("va", "vb"):
            out[V][b, :, :n] = e[V][:, :n]; out[V][b, :, L:] = e[V][:, LMAX:]
        if NOTE:
            c, p, f = notebook_feats(e["teach"] if progs is None else progs[b], it["vals"], n)
            out["cons"][b][:, idx] = c; out["pop"][b] = p; out["cf"][b] = f
    t = {k: torch.as_tensor(v) for k, v in out.items()}
    t["theta"] = torch.as_tensor(rng.random((B, NNAME, PB)) * 2 * math.pi, dtype=torch.float32)
    return t, E


def make_net():
    import torch, torch.nn as nn, torch.nn.functional as F
    DH = D // NH; w = torch.as_tensor(np.radians(CK.t6_plane_speeds(DH // 2)), dtype=torch.float32)
    wp = torch.as_tensor(np.radians(CK.t6_plane_speeds(D // 2)), dtype=torch.float32)       # TICK: the step pointer's planes

    def rot(z, a):
        c, s_ = a.cos()[:, None], a.sin()[:, None]; re, im = z[..., 0::2], z[..., 1::2]
        return torch.stack([re * c - im * s_, re * s_ + im * c], -1).flatten(-2)

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.ln1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D)
            s.ff = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)) if USE_MLP else None
            s.cb = nn.Parameter(torch.zeros(NH)) if NOTE else None

        def forward(s, x, mask, ang, cons_rows):
            B, L, _ = x.shape
            q, k, v = (t.reshape(B, L, NH, DH).transpose(1, 2) for t in s.qkv(s.ln1(x)).chunk(3, -1))
            if ang is not None: q, k = rot(q, ang), rot(k, ang)
            if cons_rows is not None: mask = mask + s.cb[None, :, None, None] * cons_rows[:, None]
            x = x + s.o(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, L, D))
            return x + s.ff(x) if s.ff is not None else x

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.inp = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, D)); s.isn = nn.Embedding(2, D); s.isnum = nn.Embedding(2, D)
            s.slotk = nn.Embedding(K, D)
            s.pos = nn.Embedding(LMAX, D) if not USE_CLOCK else None
            s.off = nn.Embedding(OMAX, D) if USE_CLOCK else None
            s.idp = nn.Linear(2 * PB, D) if USE_ID else None
            s.gp = nn.Linear(3, D) if USE_TREE else None
            s.pop = nn.Embedding(len(OPS) + 1, D) if NOTE else None; s.cfp = nn.Linear(3, D) if NOTE else None
            s.pc = nn.Parameter(torch.zeros(1)) if NOTE else None
            s.blocks = nn.ModuleList([Block() for _ in range(NL)]); s.ln = nn.LayerNorm(D)
            s.hop = nn.Linear(D, len(OPS)); s.qa = nn.Linear(D, D); s.qb = nn.Linear(D, D); s.kt = nn.Linear(D, D); s.ks = nn.Linear(D, D)
            s.kc = nn.Parameter(torch.randn(NC, D) * 0.02)

        def forward(s, t):
            x = s.inp(t["x"])
            if USE_ID:
                th = torch.gather(t["theta"], 1, t["nid"].clamp(min=0)[..., None].expand(-1, -1, PB))
                x = torch.where(t["isn"][..., None], s.idp(torch.cat([th.cos(), th.sin()], -1)), x)
            x = x + s.isn(t["isn"].long()) + s.isnum(t["isnum"].long())
            if USE_TREE: x = x + s.gp(t["g"])
            B, L = t["pad"].shape
            sl = s.slotk.weight[None].expand(B, -1, -1)
            if NOTE: sl = sl + s.pop(t["pop"]) + s.cfp(t["cf"])
            x = x.scatter_add(1, t["slot"][..., None].expand(-1, -1, D), sl)
            if USE_CLOCK: x = x + s.off(t["off"]); ang = t["sid"][..., None].float() * w
            else:
                pos = torch.cumsum((~t["pad"]).long(), 1) - 1
                x = x + s.pos(pos.clamp(min=0)); ang = None
            mask = torch.zeros(B, 1, L, L).masked_fill(t["pad"][:, None, None], -1e4)
            cons_rows = None
            if NOTE:
                isslot = torch.zeros(B, L, dtype=torch.bool).scatter(1, t["slot"], True)
                mask = mask.masked_fill((~isslot)[:, None, :, None] & isslot[:, None, None, :], -1e4)        # question rows !-> slots
                sk = torch.arange(K)
                later = (sk[None, :] > sk[:, None])                                                     # slot row k !-> slot col j>k
                blk = torch.zeros(B, L, L, dtype=torch.bool)
                ri = t["slot"][:, :, None].expand(-1, -1, K); ci = t["slot"][:, None, :].expand(-1, K, -1)
                blk[torch.arange(B)[:, None, None], ri, ci] = later[None].expand(B, -1, -1)
                mask = mask.masked_fill(blk[:, None], -1e4)
                cons_rows = torch.zeros(B, L, L)                                                         # slot rows: consumed cols
                cons_rows[torch.arange(B)[:, None], t["slot"]] = t["cons"].float()
                cs = torch.gather(t["cons"].float(), 2, t["slot"][:, None, :].expand(-1, K, -1))          # consumed earlier slots
                cons_rows[torch.arange(B)[:, None, None], t["slot"][:, :, None], t["slot"][:, None, :]] = cs
            for blk_ in s.blocks: x = blk_(x, mask, ang, cons_rows)
            x = s.ln(x); h = torch.gather(x, 1, t["slot"][..., None].expand(-1, -1, D))
            ct = t["cons"].float() if NOTE else None
            def lt(q):
                z = torch.einsum("bkd,bld->bkl", q, s.kt(x))
                if NOTE: z = z + s.pc * ct * math.sqrt(D)
                return z.masked_fill(~t["isnum"][:, None], -1e4)
            def ls(q):
                kk = s.ks(h)
                if TICK:                                                                                  # relative time between slots
                    a = torch.gather(t["sid"], 1, t["slot"])[..., None].float() * wp                     # (B, K, D/2)
                    c, s_ = a.cos(), a.sin()
                    r = lambda z: torch.stack([z[..., 0::2] * c - z[..., 1::2] * s_, z[..., 0::2] * s_ + z[..., 1::2] * c], -1).flatten(-2)
                    q, kk = r(q), r(kk)
                z = torch.einsum("bkd,bjd->bkj", q, kk)
                if NOTE: z = z + s.pc * cs * math.sqrt(D)
                return z.masked_fill(~torch.ones(K, K, dtype=torch.bool).tril(-1), -1e4)
            lc = lambda q: torch.einsum("bkd,cd->bkc", q, s.kc)
            ptr = lambda q: torch.cat([lt(q), ls(q), lc(q)], -1) / math.sqrt(D)
            return s.hop(h), ptr(s.qa(h)), ptr(s.qb(h))
    return Net()


def loss_fn(out, t):
    import torch.nn.functional as F
    op, pa, pb = out; om, am = t["om"], t["am"]
    lo = F.cross_entropy(op[om], t["op"][om])
    for p, V in ((pa, t["va"]), (pb, t["vb"])):
        lo = lo + (p.logsumexp(-1) - p.masked_fill(~V, -1e4).logsumexp(-1))[am].mean()
    return lo


def score(net, its, states, tree, seed=0):
    import torch
    rng = np.random.default_rng(10_000 + seed); res = dict(ans=[], prog=[], step_op=[], step_ptr=[], step_all=[], first_all=[], lag1_ok=[], num_for_slot=[])
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]
            progs = [[] for _ in part]; done = [False] * len(part); PO, IA, IB = [], [], []
            for k in range(K if NOTE else 1):                               # NOTEBOOK: one slot per pass (the alternation)
                t, E = batch(part, states, tree, np.random.default_rng(10_000 * seed + s0), progs if NOTE else None)   # same ID phases every pass
                op, pa, pb = net(t); L = t["pad"].shape[1]
                po = op.argmax(-1).numpy(); ia = pa.argmax(-1).numpy(); ib = pb.argmax(-1).numpy()
                dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
                if not NOTE:
                    for b in range(len(part)):
                        for kk in range(K):
                            if OPS[po[b, kk]] == ".": break
                            progs[b].append((OPS[po[b, kk]], dec(ia[b, kk]), dec(ib[b, kk])))
                    PO, IA, IB = po, ia, ib
                else:
                    if k == 0: PO, IA, IB = po.copy(), ia.copy(), ib.copy()
                    PO[:, k], IA[:, k], IB[:, k] = po[:, k], ia[:, k], ib[:, k]
                    for b in range(len(part)):
                        if done[b]: continue
                        if OPS[po[b, k]] == ".": done[b] = True; continue
                        progs[b].append((OPS[po[b, k]], dec(ia[b, k]), dec(ib[b, k])))
            for b, (it, e) in enumerate(zip(part, E)):
                v = GD.execute(progs[b], it["vals"] + [None] * (L - len(it["vals"])))
                res["ans"].append(v is not None and GD.V1.near(v, it["final"]))
                m = e["m"]
                ok = len(progs[b]) == m and all(PO[b, k] == e["op"][k] and t["va"][b, k, IA[b, k]] and t["vb"][b, k, IB[b, k]] for k in range(m))
                res["prog"].append(bool(ok))
                for k in range(m):
                    o = PO[b, k] == e["op"][k]; pt = bool(t["va"][b, k, IA[b, k]] and t["vb"][b, k, IB[b, k]])
                    res["step_op"].append(o); res["step_ptr"].append(pt); res["step_all"].append(o and pt)
                    if k == 0: res["first_all"].append(o and pt)
                    for V, I in ((t["va"], IA), (t["vb"], IB)):                     # reads: gold holds slot k-1 / gold is slot-only
                        Vb = V[b, k].numpy(); i = int(I[b, k])
                        if k >= 1 and Vb[L + k - 1]: res["lag1_ok"].append(bool(Vb[i]))
                        if Vb[L:L + K].any() and not Vb[:L].any(): res["num_for_slot"].append(bool(i < L))
    return {k: np.asarray(v) for k, v in res.items()}


if __name__ == "__main__":
    import torch
    assert ARM in ARMS, ARM
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8"))); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    Dt, states = load(); tree = load_tree(); train = Dt["TRAIN"]
    net = make_net(); opt = torch.optim.AdamW(net.parameters(), lr=LR); npar = sum(p.numel() for p in net.parameters())
    print(f"[gsmt] {TAG} arm {ARM} D {D} | train {len(train)} | " + " ".join(f"{k} {len(Dt[k])}" for k in SPLITS) + f" | params {npar}", flush=True)
    t0 = time.time(); curve = []; EV = os.environ.get("RD_EVAL", "ORIG,TEST").split(",")
    for step in range(1, STEPS + 1):
        its = [train[int(i)] for i in rng.integers(0, len(train), BATCH)]
        t, _ = batch(its, states, tree, rng); loss = loss_fn(net(t), t)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == STEPS:
            net.eval(); r = {k: score(net, Dt[k][:600], states, tree) for k in EV}; net.train()
            curve.append(dict(step=step, loss=float(loss.detach()), **{f"{k}_{m}": float(v[m].mean()) for k, v in r.items() for m in ("ans", "step_all")}))
            print(f"[gsmt] {TAG} step {step} loss {float(loss.detach()):.4f} " +
                  " ".join(f"{k} ans {v['ans'].mean():.3f} step {v['step_all'].mean():.3f}" for k, v in r.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    net.eval(); R = {k: score(net, Dt[k], states, tree) for k in os.environ.get("RD_FINAL", ",".join(SPLITS)).split(",")}
    json.dump(dict(tag=TAG, arm=ARM, seed=SEED, d=D, params=npar, steps=STEPS, curve=curve,
                   **{f"{k}_{m}": float(v[m].mean()) for k, v in R.items() for m in v},
                   **{f"{k}_per": [int(x) for x in v["ans"]] for k, v in R.items()}), open(f".cache/gsmb_trunk_{TAG}.json", "w"))
    print(f"[gsmt] {TAG} FINAL " + " | ".join(f"{k} ans {v['ans'].mean():.3f} step {v['step_all'].mean():.3f}" for k, v in R.items()), flush=True)
