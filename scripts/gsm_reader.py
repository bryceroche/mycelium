"""gsm_reader.py -- THE GSM8K READER BENCH, the reader arms (ledger 2026-10-29, Bryce "word given for both"; data scripts/gsm_bench_data.py).
The NL binding bench's arms (scripts/nl_reader.py: VANILLA / ID / TREE / CLOCK / NOMLP, one organ added at a time, same width / depth /
heads / steps) on GSM8K TEXT. The reader emits a POINTER PROGRAM: KMAX step slots appended to the question; per slot an op (+ - * / STOP)
and two operand pointers over [numeral tokens of the question | earlier slots | constants]. Numerals are the token <num> (no number
crosses); the executor reads values off the text and runs the program -> graded against the harvest final (or final_cf).
  VANILLA  learned word embeddings (names included), learned ABSOLUTE positions, pre-LN MHA + MLP blocks
  ID       a PERSON-name token's embedding is an ID TAG only (fresh random phases per distinct name per question; nl_reader's micro)
  TREE     ID + the name's gender from its SQLite tree node (.cache/gsmb_tree.db, built once by --build-tree from the names corpus,
           a static lexicon: no answers, no labels; names absent from the lexicon carry an evidence bit 0)
  CLOCK    TREE without absolute positions: offset-in-sentence embedding + q/k rotated by the T6 gears x SENTENCE index (slots sit
           one sentence after the question, offset = slot index)
  NOMLP    CLOCK without the feed-forward (RD_D sets width; NOMLPW = RD_D=208, parameter-matched)
Output .cache/gsmb_{TAG}.json."""
import json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
from mycelium import rotor_clock as CK

ARM = os.environ.get("RD_ARM", "VANILLA"); SEED = int(os.environ.get("RD_SEED", "0")); TAG = os.environ.get("RD_TAG", f"{ARM}_s{SEED}")
STEPS = int(os.environ.get("RD_STEPS", "6000")); BATCH = int(os.environ.get("RD_BATCH", "64")); LR = float(os.environ.get("RD_LR", "3e-4"))
D = int(os.environ.get("RD_D", "128")); NH = 4; NL = int(os.environ.get("RD_NL", "4")); PB = 64; NNAME = 12; OMAX = 64
K = GD.KMAX; LMAX = GD.LQMAX + K; NC = len(GD.CONSTS); OPS = GD.OPS + "."          # '.' = STOP
TREE_DB = ".cache/gsmb_tree.db"
ARMS = ("VANILLA", "ID", "TREE", "CLOCK", "NOMLP")
USE_ID, USE_TREE, USE_CLOCK, USE_MLP = ARM != "VANILLA", ARM in ("TREE", "CLOCK", "NOMLP"), ARM in ("CLOCK", "NOMLP"), ARM != "NOMLP"
SPLITS = ("ORIG", "CFNUM", "CFNAME", "TEST")


def load():
    Dt = json.load(open(GD.OUT)); cnt = {}
    for it in Dt["TRAIN"]:
        for w in it["toks"]: cnt[w] = cnt.get(w, 0) + 1
    vocab = ["<pad>", "<unk>", "<slot>"] + sorted(w for w, c in cnt.items() if c >= 2)
    return Dt, {w: i for i, w in enumerate(vocab)}


def build_tree():
    from mycelium import tree_db as TDB
    import gsm8k_counterfactual as V1
    assert not os.path.exists(TREE_DB), f"refusing to rebuild {TREE_DB}"
    Dt, _ = load(); names = sorted({n for sp in Dt.values() for it in sp for n in it["names"]})
    rows = [(n, np.eye(2, dtype=np.float32)[g], f"lex.{n}.{g}") for n in names for g, lst in enumerate((V1.NF, V1.NM)) if n in lst]
    db = TDB.TreeDB(TREE_DB); db.register_space("name-gender", "symbolic", 2, "names-corpus membership (female, male) per first name")
    db.add_evidence("names-corpus", rows, "permanent", "name-list", "gsmb", "name-gender", note="static lexicon; no answers, no labels")
    L = db.load_leaves(); print(f"[gsmb] tree built: {len(L)}/{len(names)} PERSON names have a lexicon node", flush=True); db.close()


def load_tree():
    from mycelium import tree_db as TDB
    db = TDB.TreeDB(TREE_DB); L = db.load_leaves(); db.close()
    return {k: np.asarray(v[1], np.float32) for k, v in L.items()}


def encode(it, wid, tree):
    n = len(it["toks"]); L = n + K
    w = np.array([wid.get(x, 1) for x in it["toks"]] + [2] * K)
    isn = np.array(it["isn"] + [False] * K); isnum = np.array([v is not None for v in it["vals"]] + [False] * K)
    sid = np.array(it["sid"] + [it["sid"][-1] + 1] * K); off = np.zeros(L, np.int64)
    for i in range(1, n): off[i] = off[i - 1] + 1 if sid[i] == sid[i - 1] else 0
    off[n:] = np.arange(K)
    nm = [x for x in dict.fromkeys(t for t, f in zip(it["surf"], it["isn"]) if f)]
    nid = np.array([min(nm.index(t), NNAME - 1) if f else -1 for t, f in zip(it["surf"], it["isn"])] + [-1] * K)
    g = np.zeros((L, 3), np.float32)
    for i, (t, f) in enumerate(zip(it["surf"], it["isn"])):
        if f and t in tree: g[i, :2] = tree[t]; g[i, 2] = 1
    P = it["prog"]; m = len(P)
    op = np.array([OPS.index(p[0]) for p in P[:K]] + ([OPS.index(".")] if m < K else []))
    va = np.zeros((K, LMAX + K + NC), bool); vb = np.zeros_like(va)
    for k, (_, A, B) in enumerate(P[:K]):
        for V, S in ((va, A), (vb, B)):
            V[k, S["t"]] = True; V[k, [LMAX + j for j in S["s"]]] = True; V[k, [LMAX + K + c for c in S["c"]]] = True
    return dict(w=w, isn=isn, isnum=isnum, sid=sid, off=np.minimum(off, OMAX - 1), nid=nid, g=g, n=n, op=op, va=va, vb=vb, m=min(m, K))


def batch(its, wid, tree, rng):
    import torch
    E = [encode(it, wid, tree) for it in its]; B = len(E); L = max(e["n"] for e in E) + K
    out = dict(w=np.zeros((B, L), np.int64), isn=np.zeros((B, L), bool), isnum=np.zeros((B, L), bool), sid=np.zeros((B, L), np.int64),
               off=np.zeros((B, L), np.int64), nid=np.full((B, L), -1), g=np.zeros((B, L, 3), np.float32), pad=np.ones((B, L), bool),
               slot=np.zeros((B, K), np.int64), op=np.zeros((B, K), np.int64), om=np.zeros((B, K), bool), am=np.zeros((B, K), bool),
               va=np.zeros((B, K, L + K + NC), bool), vb=np.zeros((B, K, L + K + NC), bool))
    for b, e in enumerate(E):
        n = e["n"]; idx = list(range(n)) + list(range(L - K, L))          # question left-aligned, slots at the right end
        for k in ("w", "isn", "isnum", "sid", "off", "nid", "g"): out[k][b, idx] = e[k]
        out["pad"][b, idx] = False; out["slot"][b] = np.arange(L - K, L)
        out["op"][b, :len(e["op"])] = e["op"]; out["om"][b, :len(e["op"])] = True; out["am"][b, :e["m"]] = True
        for V in ("va", "vb"):                                          # remap [LMAX | K | NC] -> [L | K | NC]
            out[V][b, :, :n] = e[V][:, :n]; out[V][b, :, L:] = e[V][:, LMAX:]
    t = {k: torch.as_tensor(v) for k, v in out.items()}
    t["theta"] = torch.as_tensor(rng.random((B, NNAME, PB)) * 2 * math.pi, dtype=torch.float32)
    return t, E


def make_net(nvocab):
    import torch, torch.nn as nn, torch.nn.functional as F
    DH = D // NH; w = torch.as_tensor(np.radians(CK.t6_plane_speeds(DH // 2)), dtype=torch.float32)

    def rot(z, a):
        c, s_ = a.cos()[:, None], a.sin()[:, None]; re, im = z[..., 0::2], z[..., 1::2]
        return torch.stack([re * c - im * s_, re * s_ + im * c], -1).flatten(-2)

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.ln1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D)
            s.ff = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)) if USE_MLP else None

        def forward(s, x, mask, ang):
            B, L, _ = x.shape
            q, k, v = (t.reshape(B, L, NH, DH).transpose(1, 2) for t in s.qkv(s.ln1(x)).chunk(3, -1))
            if ang is not None: q, k = rot(q, ang), rot(k, ang)
            x = x + s.o(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, L, D))
            return x + s.ff(x) if s.ff is not None else x

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(nvocab, D); s.isn = nn.Embedding(2, D); s.isnum = nn.Embedding(2, D); s.slotk = nn.Embedding(K, D)
            s.pos = nn.Embedding(LMAX, D) if not USE_CLOCK else None
            s.off = nn.Embedding(OMAX, D) if USE_CLOCK else None
            s.idp = nn.Linear(2 * PB, D) if USE_ID else None
            s.gp = nn.Linear(3, D) if USE_TREE else None
            s.blocks = nn.ModuleList([Block() for _ in range(NL)]); s.ln = nn.LayerNorm(D)
            s.hop = nn.Linear(D, len(OPS)); s.qa = nn.Linear(D, D); s.qb = nn.Linear(D, D); s.kt = nn.Linear(D, D); s.ks = nn.Linear(D, D)
            s.kc = nn.Parameter(torch.randn(NC, D) * 0.02)

        def forward(s, t):
            x = s.emb(t["w"])
            if USE_ID:
                th = torch.gather(t["theta"], 1, t["nid"].clamp(min=0)[..., None].expand(-1, -1, PB))
                x = torch.where(t["isn"][..., None], s.idp(torch.cat([th.cos(), th.sin()], -1)), x)
            x = x + s.isn(t["isn"].long()) + s.isnum(t["isnum"].long())
            if USE_TREE: x = x + s.gp(t["g"])
            B, L = t["w"].shape
            x = x.scatter_add(1, t["slot"][..., None].expand(-1, -1, D), s.slotk.weight[None].expand(B, -1, -1))
            if USE_CLOCK: x = x + s.off(t["off"]); ang = t["sid"][..., None].float() * w
            else:
                pos = torch.cumsum((~t["pad"]).long(), 1) - 1                 # slots continue the question's positions
                x = x + s.pos(pos.clamp(min=0)); ang = None
            mask = torch.zeros(B, 1, 1, L).masked_fill(t["pad"][:, None, None], -1e4)
            for blk in s.blocks: x = blk(x, mask, ang)
            x = s.ln(x); h = torch.gather(x, 1, t["slot"][..., None].expand(-1, -1, D))
            lt = lambda q: torch.einsum("bkd,bld->bkl", q, s.kt(x)).masked_fill(~t["isnum"][:, None], -1e4)
            ls = lambda q: torch.einsum("bkd,bjd->bkj", q, s.ks(h)).masked_fill(~torch.ones(K, K, dtype=torch.bool).tril(-1), -1e4)
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


def score(net, its, wid, tree, seed=0):
    import torch
    rng = np.random.default_rng(10_000 + seed); res = dict(ans=[], prog=[])
    with torch.no_grad():
        for s0 in range(0, len(its), 128):
            part = its[s0:s0 + 128]; t, E = batch(part, wid, tree, rng); op, pa, pb = net(t)
            L = t["w"].shape[1]; po = op.argmax(-1).numpy(); ia = pa.argmax(-1).numpy(); ib = pb.argmax(-1).numpy()
            dec = lambda i: ("t", int(i)) if i < L else (("s", int(i - L)) if i < L + K else ("c", int(i - L - K)))
            for b, (it, e) in enumerate(zip(part, E)):
                prog = []
                for k in range(K):
                    if OPS[po[b, k]] == ".": break
                    prog.append((OPS[po[b, k]], dec(ia[b, k]), dec(ib[b, k])))
                v = GD.execute(prog, it["vals"] + [None] * (L - len(it["vals"])))
                res["ans"].append(v is not None and GD.V1.near(v, it["final"]))
                m = e["m"]; ok = len(prog) == m and all(po[b, k] == e["op"][k] and t["va"][b, k, ia[b, k]] and t["vb"][b, k, ib[b, k]]
                                                         for k in range(m))
                res["prog"].append(bool(ok))
    return {k: np.asarray(v) for k, v in res.items()}


def prep(Dt):
    """attach surface tokens (case kept) for the name channel: a token is a name iff its lowercase form matches a PERSON name."""
    for sp in Dt.values():
        for it in sp:
            nl = {n.lower(): n for n in it["names"]}
            it["surf"] = [nl.get(x, x) if f else x for x, f in zip(it["toks"], it["isn"])]
    return Dt


if __name__ == "__main__":
    if "--build-tree" in sys.argv: build_tree(); sys.exit()
    import torch
    assert ARM in ARMS, ARM
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8"))); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    Dt, wid = load(); Dt = prep(Dt); tree = load_tree(); train = Dt["TRAIN"]
    net = make_net(len(wid)); opt = torch.optim.AdamW(net.parameters(), lr=LR); npar = sum(p.numel() for p in net.parameters())
    print(f"[gsmb] {TAG} arm {ARM} | train {len(train)} | " + " ".join(f"{k} {len(Dt[k])}" for k in SPLITS) + f" | vocab {len(wid)} | params {npar}", flush=True)
    t0 = time.time(); curve = []
    for step in range(1, STEPS + 1):
        its = [train[int(i)] for i in rng.integers(0, len(train), BATCH)]
        t, _ = batch(its, wid, tree, rng); loss = loss_fn(net(t), t)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == STEPS:
            net.eval(); r = {k: score(net, Dt[k][:600], wid, tree)["ans"].mean() for k in ("ORIG", "TEST")}; net.train()
            curve.append(dict(step=step, loss=float(loss.detach()), **{k: float(v) for k, v in r.items()}))
            print(f"[gsmb] {TAG} step {step} loss {float(loss.detach()):.4f} ORIG {r['ORIG']:.3f} TEST {r['TEST']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval(); R = {k: score(net, Dt[k], wid, tree) for k in SPLITS}
    json.dump(dict(tag=TAG, arm=ARM, seed=SEED, d=D, params=npar, steps=STEPS, curve=curve,
                   **{f"{k}_{m}": float(v[m].mean()) for k, v in R.items() for m in v},
                   **{f"{k}_per": [int(x) for x in v["ans"]] for k, v in R.items()}), open(f".cache/gsmb_{TAG}.json", "w"))
    print(f"[gsmb] {TAG} FINAL " + " | ".join(f"{k} ans {v['ans'].mean():.3f} prog {v['prog'].mean():.3f}" for k, v in R.items()), flush=True)
