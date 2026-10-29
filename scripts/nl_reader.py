"""nl_reader.py -- THE NL BINDING BENCH, the reader arms (ledger 2026-10-29; generator + census scripts/nl_bench_gen.py). CPU torch.
The reader reads a story and predicts, per mention anchor: the sentence op, the mention's role and its OWNER (pointer to an occurrence
of the owner's name); the assembler + forced propagation grade the answer against the custody key. Arms (RD_ARM), one organ added at
a time, every arm the same width / depth / heads / steps:
  VANILLA  learned word embeddings (names included), learned ABSOLUTE positions, pre-LN MHA + MLP blocks (NL layers)
  ID       VANILLA with organ 4's micro channel: a NAME token's embedding is an ID TAG only = Linear([cos th, sin th]) of a fresh random
           phase vector per distinct name per story (P planes) -- no learned per-name vector exists, so an unseen name is no different
           from a seen one, and nothing about a name (its gender) survives in the micro channel
  TREE     ID + organ 3 as organ 4's macro channel: every name token gets the TYPE its SQLite tree node holds (gender, Welford mean of
           one-hot title evidence, + an evidence bit) -- .cache/nlb_tree.db (mycelium/tree_db.py), built once by --build-tree:
           titles in 20,000 training stories = a PERMANENT batch; titles in the test stories = a PROVISIONAL batch promoted only
           if no key's evidence conflicts (the audition); the evidence is surface titles, never answers
  CLOCK    TREE with time by organs 7 + 8: the learned absolute positions are removed; a token carries only its offset inside its
           sentence (always in the trained range), and every block's queries and keys are rotated by the T6 gears x the token's
           SENTENCE index (relative phase between sentences = how many sentences ago; rotor_clock.t6_plane_speeds per head plane)
  NOMLP    CLOCK without the feed-forward (Bryce's MHA-without-MLP block)
Output .cache/nlb_{TAG}.json (TAG default {ARM}_s{SEED})."""
import json, math, os, sys, time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import nl_bench_gen as G
from mycelium import rotor_clock as CK

ARM = os.environ.get("RD_ARM", "VANILLA"); SEED = int(os.environ.get("RD_SEED", "0")); TAG = os.environ.get("RD_TAG", f"{ARM}_s{SEED}")
STEPS = int(os.environ.get("RD_STEPS", "6000")); BATCH = int(os.environ.get("RD_BATCH", "64")); LR = float(os.environ.get("RD_LR", "3e-4"))
D = int(os.environ.get("RD_D", "128")); NH = 4; NL = int(os.environ.get("RD_NL", "4")); PB = 64; LMAX = 192; OMAX = 32
TREE_DB = ".cache/nlb_tree.db"
ARMS = ("VANILLA", "ID", "TREE", "CLOCK", "NOMLP")
USE_ID, USE_TREE, USE_CLOCK, USE_MLP = ARM != "VANILLA", ARM in ("TREE", "CLOCK", "NOMLP"), ARM in ("CLOCK", "NOMLP"), ARM != "NOMLP"


def title_rows(stories, tag):
    out = []
    for k, st in enumerate(stories):
        for i, w in enumerate(st["toks"][:-1]):
            if w in G.TITLE: out.append((st["toks"][i + 1], np.eye(2, dtype=np.float32)[G.TITLE.index(w)], f"{tag}.{k}.{i}"))
    return out


def build_tree():
    from mycelium import tree_db as TDB
    assert not os.path.exists(TREE_DB), f"refusing to rebuild {TREE_DB}"
    train, S, _, _ = G.test_sets(); rng = np.random.default_rng(7)
    db = TDB.TreeDB(TREE_DB); db.register_space("name-gender", "symbolic", 2, "one-hot title evidence (ms., mr.) per name")
    tr = [G.make_story(train[int(rng.integers(len(train)))], G.TRAIN_NAMES, rng) for _ in range(20000)]
    db.add_evidence("train-titles", title_rows(tr, "train"), "permanent", "surface-title", "nlb", "name-gender")
    db.add_evidence("test-titles", title_rows([st for v in S.values() for st in v], "test"), "provisional", "surface-title", "nlb",
                    "name-gender", note="test-time stored procedure: surface titles only")
    conf = [k for k, (n, mean, m2) in db.batch_summaries("test-titles").items() if 0 < mean[0] < 1]
    assert not conf, f"audition failed: conflicting title evidence for {conf}"
    db.promote("test-titles")
    L = db.load_leaves(); print(f"[nlb] tree built: {len(L)} name nodes | train-pool {sum(n in L for n in G.TRAIN_NAMES)}/60 | "
                                f"held-pool {sum(n in L for n in G.HELD_NAMES)}/30 (promoted from test titles)", flush=True); db.close()


def load_tree():
    from mycelium import tree_db as TDB
    db = TDB.TreeDB(TREE_DB); L = db.load_leaves(); db.close()
    return {k: np.asarray(v[1], np.float32) for k, v in L.items()}


def encode(st, tree):
    T = st["toks"]; L = len(T); assert L <= LMAX
    w = np.array([G.WID[x] for x in T]); isn = np.array(st["is_name"])
    sid = np.zeros(L, np.int64); off = np.zeros(L, np.int64)
    for k, s in enumerate(st["sents"]): sid[s["start"]:s["end"]] = k; off[s["start"]:s["end"]] = np.arange(s["end"] - s["start"])
    names = sorted(st["name_pos"], key=lambda n: st["name_pos"][n][0]); nid = np.array([names.index(x) if x in G.GENDER else -1 for x in T])
    g = np.zeros((L, 3), np.float32)
    for i, x in enumerate(T):
        if x in G.GENDER and x in tree: g[i, :2] = tree[x]; g[i, 2] = 1
    A = st["anchors"]; anc = np.array([a["tok"] for a in A]); op = np.array([G.OPS.index(a["op"]) for a in A])
    role = np.array([G.ROLES.index(a["role"]) for a in A]); own = np.zeros((len(A), L), bool)
    for j, a in enumerate(A): own[j, st["name_pos"][a["owner"]]] = True
    return dict(w=w, isn=isn, sid=sid, off=np.minimum(off, OMAX - 1), nid=nid, g=g, anc=anc, op=op, role=role, own=own,
                pron=np.array([a["pron"] for a in A]), need=np.array([a["need"] for a in A]))


def batch(sts, tree, rng):
    import torch
    E = [encode(st, tree) for st in sts]; B = len(E); L = max(len(e["w"]) for e in E); A = max(len(e["anc"]) for e in E)
    out = dict(w=np.zeros((B, L), np.int64), isn=np.zeros((B, L), bool), sid=np.zeros((B, L), np.int64), off=np.zeros((B, L), np.int64),
               nid=np.full((B, L), -1), g=np.zeros((B, L, 3), np.float32), pad=np.ones((B, L), bool), anc=np.zeros((B, A), np.int64),
               op=np.zeros((B, A), np.int64), role=np.zeros((B, A), np.int64), own=np.zeros((B, A, L), bool), am=np.zeros((B, A), bool))
    for b, e in enumerate(E):
        n, a = len(e["w"]), len(e["anc"])
        for k in ("w", "isn", "sid", "off", "nid", "g"): out[k][b, :n] = e[k]
        out["pad"][b, :n] = False
        for k in ("anc", "op", "role"): out[k][b, :a] = e[k]
        out["own"][b, :a, :n] = e["own"]; out["am"][b, :a] = True
    t = {k: torch.as_tensor(v) for k, v in out.items()}
    t["theta"] = torch.as_tensor(rng.random((B, 8, PB)) * 2 * math.pi, dtype=torch.float32)   # fresh ID phases per story per name
    return t, E


def make_net():
    import torch, torch.nn as nn, torch.nn.functional as F
    DH = D // NH; w = torch.as_tensor(np.radians(CK.t6_plane_speeds(DH // 2)), dtype=torch.float32)

    def rot(z, a):   # z: B x NH x L x DH (interleaved planes); a: B x L x DH/2
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
            s.emb = nn.Embedding(len(G.VOCAB), D); s.isn = nn.Embedding(2, D)
            s.pos = nn.Embedding(LMAX, D) if not USE_CLOCK else None
            s.off = nn.Embedding(OMAX, D) if USE_CLOCK else None
            s.idp = nn.Linear(2 * PB, D) if USE_ID else None
            s.gp = nn.Linear(3, D) if USE_TREE else None
            s.blocks = nn.ModuleList([Block() for _ in range(NL)]); s.ln = nn.LayerNorm(D)
            s.hop = nn.Linear(D, len(G.OPS)); s.hrole = nn.Linear(D, len(G.ROLES)); s.q = nn.Linear(D, D); s.k = nn.Linear(D, D)

        def forward(s, t):
            x = s.emb(t["w"])
            if USE_ID:                                                       # micro: the ID tag only
                th = torch.gather(t["theta"], 1, t["nid"].clamp(min=0)[..., None].expand(-1, -1, PB))
                x = torch.where(t["isn"][..., None], s.idp(torch.cat([th.cos(), th.sin()], -1)), x)
            x = x + s.isn(t["isn"].long())
            if USE_TREE: x = x + s.gp(t["g"])                                # macro: the tree node's type
            B, L = t["w"].shape
            if USE_CLOCK: x = x + s.off(t["off"]); ang = t["sid"][..., None].float() * w
            else: x = x + s.pos(torch.arange(L))[None]; ang = None
            mask = torch.zeros(B, 1, 1, L).masked_fill(t["pad"][:, None, None], -1e4)
            for blk in s.blocks: x = blk(x, mask, ang)
            x = s.ln(x); ha = torch.gather(x, 1, t["anc"][..., None].expand(-1, -1, D))
            own = torch.einsum("bad,bld->bal", s.q(ha), s.k(x)) / math.sqrt(D)
            own = own.masked_fill(~t["isn"][:, None], -1e4)
            return s.hop(ha), s.hrole(ha), own
    return Net()


def loss_fn(out, t):
    import torch.nn.functional as F
    op, role, own = out; am = t["am"]
    lo = F.cross_entropy(op[am], t["op"][am]) + F.cross_entropy(role[am], t["role"][am])
    num = own.masked_fill(~t["own"], -1e4).logsumexp(-1); den = own.logsumexp(-1)
    return lo + (den - num)[am].mean()


def score(net, sts, tree, seed=0):
    import torch
    rng = np.random.default_rng(10_000 + seed); res = dict(ans=[], own=[], own_pron=[], own_need=[], op=[], role=[])
    with torch.no_grad():
        for s0 in range(0, len(sts), 128):
            its = sts[s0:s0 + 128]; t, E = batch(its, tree, rng); op, role, own = net(t)
            for b, (st, e) in enumerate(zip(its, E)):
                a = len(e["anc"]); po = op[b, :a].argmax(-1).numpy(); pr = role[b, :a].argmax(-1).numpy(); pw = own[b, :a].argmax(-1).numpy()
                names = [st["toks"][i] for i in pw]; ok_own = e["own"][np.arange(a), pw]
                res["ans"].append(G.assemble(st, [G.OPS[i] for i in po], [G.ROLES[i] for i in pr], names) == Fraction(st["gold"]))
                res["own"] += list(ok_own); res["own_pron"] += list(ok_own[e["pron"]]); res["own_need"] += list(ok_own[e["need"]])
                res["op"] += list(po == e["op"]); res["role"] += list(pr == e["role"])
    return {k: np.asarray(v) for k, v in res.items()}


if __name__ == "__main__":
    if "--build-tree" in sys.argv: build_tree(); sys.exit()
    import torch
    assert ARM in ARMS, ARM
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8"))); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    train, S, _, _ = G.test_sets(); tree = load_tree(); net = make_net(); opt = torch.optim.AdamW(net.parameters(), lr=LR)
    npar = sum(p.numel() for p in net.parameters())
    print(f"[nlb] {TAG} arm {ARM} | train structures {len(train)} | IN/HELD {len(S['IN'])} LONG {len(S['LONG'])} | params {npar}", flush=True)
    t0 = time.time(); curve = []
    for step in range(1, STEPS + 1):
        sts = [G.make_story(train[int(rng.integers(len(train)))], G.TRAIN_NAMES, rng) for _ in range(BATCH)]
        t, _ = batch(sts, tree, rng); loss = loss_fn(net(t), t)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if step % 1000 == 0 or step == STEPS:
            net.eval(); r = {k: score(net, S[k], tree)["ans"].mean() for k in ("IN", "HELD")}; net.train()
            curve.append(dict(step=step, loss=float(loss.detach()), **{k: float(v) for k, v in r.items()}))
            print(f"[nlb] {TAG} step {step} loss {float(loss.detach()):.4f} IN {r['IN']:.3f} HELD {r['HELD']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    net.eval(); R = {k: score(net, v, tree) for k, v in S.items()}
    json.dump(dict(tag=TAG, arm=ARM, seed=SEED, params=npar, steps=STEPS, curve=curve,
                   **{f"{k}_{m}": float(v[m].mean()) for k, v in R.items() for m in v},
                   **{f"{k}_per": [int(x) for x in v["ans"]] for k, v in R.items()}), open(f".cache/nlb_{TAG}.json", "w"))
    print(f"[nlb] {TAG} FINAL " + " | ".join(f"{k} ans {v['ans'].mean():.3f} owner {v['own'].mean():.3f} pron {v['own_pron'].mean():.3f} "
                                           f"need-gender {v['own_need'].mean():.3f}" for k, v in R.items()), flush=True)
