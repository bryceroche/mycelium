"""composite_entity_toy.py -- T2b, COMPOSITE ENTITIES WITH TREE-CODED TYPES (2026-10-22; branch binding-comprehension;
Bryce: "pair the hierarchical embeddings with bindings for problem-specific composite entities: Janet bound to apples
creates a new composite entity, same with Bob's apples"; ledger "T2b").

T2's records are already composite keys (owner x object x verb, as a channel table or a phase binding). What T2b adds is
the TREE inside the object channel: 4 categories x 5 leaves = 20 objects; one leaf per category is NEVER SEEN in
training. A composite key then must answer (a) fine queries about an unseen leaf and (b) COARSE queries ('how many FRUIT
does Janet have in total?') that need category membership -- the tree rung's B3 (path codes reach unseen leaves) carried
into composite entities.

OBJECT EMBEDDINGS:
  FLATEMB  every object token has its own learned embedding (an unseen leaf's is its random init); category tokens
           learned separately (membership must be learned from data).
  TREEEMB  object embedding = C_cat[category] (learned, shared by siblings) + C_leaf[leaf] (FIXED random); a category
           token's embedding IS C_cat[category]: the tree's path code, aligned with the query's coarse level.
READS (both feature-supervised, T2's winning form): TABLE = STRUCT_S (owner / object / verb channels, gate-multiply);
BIND = BIND_S (the same channels as phase-bound complex planes, unbound for supervision). The object channel is
supervised at BOTH levels (category CE + leaf CE), in every arm.
QUERIES: LOOKUP (owner, verb, object) -> 1 pointer; LEFT (owner, object) -> HAS and OUT pointers, a calculator subtracts;
COARSE (owner, CATEGORY, TOTAL) -> a multi-select over records (sigmoid per record), a calculator sums the selected HAS
values; correct = the exact set (only HAS records count toward a total).
SPLITS: IID (seen leaves only) / UNSEEN (every story contains held-out leaves and every query touches one).
No LONG split: T2's feasibility read showed length failing in every arm (a trunk property, not a read property).
"""
import json, math, os, sys, time
mode = "smoke" if "--smoke" in sys.argv else "full"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

NE, NDESC, NCAT, NLEAF, NNUM = 16, 8, 4, 5, 20
NOBJ = NCAT * NLEAF
T_PAD, T_S, T_DOT, T_Q, T_QM, T_IS, T_HAS, T_BUYS, T_LOSES, T_GIVES, T_LEFT, T_TOTAL, T_HE, T_SHE = range(14)
O_NAME = 14; O_DESC = O_NAME + NE; O_OBJ = O_DESC + NDESC; O_CAT = O_OBJ + NOBJ; O_NUM = O_CAT + NCAT
VOCAB = O_NUM + NNUM + 1
VERB_CLS = {T_HAS: 0, T_BUYS: 1, T_LOSES: 2, T_GIVES: 2}
D, NH, NL = 64, 4, 2
STEPS = int(os.environ.get("CE_STEPS", "300" if mode == "smoke" else "4000"))
BATCH, N_EVAL = 64, (300 if mode == "smoke" else 2000)
SEEDS = [0] if mode == "smoke" else [0, 1, 2]
ARMS = os.environ.get("CE_ARMS", "FLATEMB_TABLE,TREEEMB_TABLE,FLATEMB_BIND,TREEEMB_BIND").split(",")
LR = 1e-3
GENDER = np.array([0] * 8 + [1] * 8)
UNSEEN = {c * NLEAF + NLEAF - 1 for c in range(NCAT)}            # the last leaf of every category
cat_of = lambda o: o // NLEAF


def story(rng, n_ev, split):
    ents = rng.choice(NE, size=int(rng.integers(3, 5)), replace=False)
    desc = {int(e): int(d) for e, d in zip(ents, rng.permutation(NDESC)[: len(ents)])}
    intro = {int(e): rng.random() < 0.6 for e in ents}
    toks, recs, seen, last_by_g, introduced, pairs = [], [], set(), {}, set(), {}
    objs = [o for o in range(NOBJ) if o not in UNSEEN] if split == "train" else list(range(NOBJ))

    def mention(e):
        if e not in seen:
            seen.add(e); last_by_g[GENDER[e]] = e
            if intro[e]:
                toks.extend([T_S, O_NAME + e, T_IS, O_DESC + desc[e], T_DOT]); introduced.add(e)
            return [O_NAME + e], "name"
        opts = [("name", [O_NAME + e])]
        if last_by_g.get(GENDER[e]) == e:
            opts.append(("pronoun", [T_HE if GENDER[e] == 0 else T_SHE]))
        if e in introduced:
            opts.append(("desc", [O_DESC + desc[e]]))
        form, tk = opts[int(rng.integers(len(opts)))]; last_by_g[GENDER[e]] = e
        return tk, form

    tries = 0
    while len(recs) < n_ev and tries < 300:
        tries += 1
        open_ = [p for p, s in pairs.items() if len(s) < 3]
        if open_ and rng.random() < 0.5:
            e, o = open_[int(rng.integers(len(open_)))]
        else:
            e = int(rng.choice(ents))
            if split == "unseen" and rng.random() < 0.5:
                o = int(rng.choice(sorted(UNSEEN)))
            elif rng.random() < 0.5 and pairs:                         # siblings under one owner: coarse totals need >1
                pe = [p for p in pairs if p[0] == e]
                o = int(cat_of(pe[0][1]) * NLEAF + rng.integers(NLEAF)) if pe else int(rng.choice(objs))
                if split == "train" and o in UNSEEN:
                    continue
            else:
                o = int(rng.choice(objs))
        st = pairs.setdefault((e, o), set())
        if 0 not in st:
            verb = T_HAS
        else:
            cand = [v for v, c in ((T_BUYS, 1), (T_LOSES, 2), (T_GIVES, 2)) if c not in st]
            if not cand:
                continue
            verb = cand[int(rng.integers(len(cand)))]
        n = int(rng.integers(1, NNUM + 1))
        mk, form = mention(e); toks.extend([T_S] + mk + [verb])
        if verb == T_GIVES:
            others = [int(x) for x in ents if int(x) != e]; rk, _ = mention(others[int(rng.integers(len(others)))]); toks.extend(rk)
        recs.append((len(toks), e, o, VERB_CLS[verb], n, form)); toks.extend([O_NUM + n - 1, O_OBJ + o, T_DOT]); st.add(VERB_CLS[verb])
    return toks, recs, desc, introduced


def example(rng, split):
    while True:
        toks, recs, desc, introduced = story(rng, int(rng.integers(4, 7)), split)
        pool = [r for r in recs if (split != "unseen" or r[2] in UNSEEN)]
        if pool:
            break
    tgt = pool[int(rng.integers(len(pool)))]; e, o = tgt[1], tgt[2]
    same = {r[3]: r for r in recs if r[1] == e and r[2] == o}
    qo = [O_DESC + desc[e]] if (e in introduced and rng.random() < 0.5) else [O_NAME + e]
    u = rng.random()
    if u < 0.34:                                                        # COARSE total over the category
        c = cat_of(o); sel = [r for r in recs if r[1] == e and cat_of(r[2]) == c and r[3] == 0]
        q = [T_Q] + qo + [O_CAT + c, T_TOTAL, T_QM]; return dict(toks=toks + q, recs=recs, ops=[], sel=sel, kind="coarse", tgt=tgt)
    if 2 in same and u < 0.67:
        q = [T_Q] + qo + [O_OBJ + o, T_LEFT, T_QM]; return dict(toks=toks + q, recs=recs, ops=[same[0], same[2]], sel=None, kind="left", tgt=tgt)
    vt = {0: T_HAS, 1: T_BUYS, 2: None}[tgt[3]]
    if vt is None:
        vt = T_GIVES if any(t == T_GIVES for t in toks[tgt[0] - 4: tgt[0]]) else T_LOSES
    q = [T_Q] + qo + [vt, O_OBJ + o, T_QM]; return dict(toks=toks + q, recs=recs, ops=[tgt], sel=None, kind="lookup", tgt=tgt)


def batch(rng, B, split):
    exs = [example(rng, split) for _ in range(B)]
    L = max(len(x["toks"]) for x in exs); R = max(len(x["recs"]) for x in exs)
    tok = np.full((B, L), T_PAD); rpos = np.zeros((B, R), np.int64); rmask = np.zeros((B, R), bool)
    rfeat = np.zeros((B, R, 4), np.int64); qpos = np.zeros(B, np.int64); tgt = np.full((B, 2), -1)
    sel = np.zeros((B, R), np.float32); coarse = np.zeros(B, bool); qfeat = np.full((B, 2, 4), -1)
    for b, x in enumerate(exs):
        tok[b, :len(x["toks"])] = x["toks"]; qpos[b] = len(x["toks"]) - 1
        for k, r in enumerate(x["recs"]):
            rpos[b, k] = r[0]; rmask[b, k] = True; rfeat[b, k] = (r[1], cat_of(r[2]), r[2], r[3])
        for i, op in enumerate(x["ops"]):
            tgt[b, i] = x["recs"].index(op); qfeat[b, i] = (op[1], cat_of(op[2]), op[2], op[3])
        if x["kind"] == "coarse":
            coarse[b] = True
            for r in x["sel"]:
                sel[b, x["recs"].index(r)] = 1.0
    t = lambda a: torch.from_numpy(np.asarray(a))
    return dict(tok=t(tok), rpos=t(rpos), rmask=t(rmask), rfeat=t(rfeat), qpos=t(qpos), tgt=t(tgt), qfeat=t(qfeat),
                sel=t(sel), coarse=t(coarse), kind=[x["kind"] for x in exs])


def sinus(L):
    p = torch.arange(L)[:, None].float(); i = torch.arange(0, D, 2).float()
    a = p / (10000 ** (i / D)); pe = torch.zeros(L, D); pe[:, 0::2] = torch.sin(a); pe[:, 1::2] = torch.cos(a)
    return pe


class Net(nn.Module):
    CH = (21, 21, 22)                                                    # owner / object (cat + leaf) / verb
    NCLS = (NE, NCAT, NOBJ, 3)                                           # supervised: owner, category, leaf, verb

    def __init__(self, arm):
        super().__init__(); self.arm = arm; self.tree = arm.startswith("TREE"); self.bindr = arm.endswith("BIND")
        self.emb = nn.Embedding(VOCAB, D)
        if self.tree:
            g = torch.Generator().manual_seed(31337)
            self.c_cat = nn.Parameter(torch.randn(NCAT, D) * 0.5)
            self.register_buffer("c_leaf", torch.randn(NLEAF, D, generator=g) * 0.5)
        lay = nn.TransformerEncoderLayer(D, NH, 4 * D, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(lay, NL, enable_nested_tensor=False)
        self.qop = nn.Linear(D, 3 * D)                                   # operand 1 / operand 2 / the coarse query
        if self.bindr:
            self.P = 32; g = torch.Generator().manual_seed(4242)
            self.register_buffer("role", torch.rand(3, self.P, generator=g) * 2 * math.pi)
            self.wq = nn.ModuleList([nn.Linear(D, 2 * self.P) for _ in range(3)]); self.wk = nn.ModuleList([nn.Linear(D, 2 * self.P) for _ in range(3)])
            self.log_beta = nn.Parameter(torch.tensor(math.log(8.0))); cd = (2 * self.P,) * 3
        else:
            self.wq, self.wk = nn.Linear(D, sum(self.CH)), nn.Linear(D, sum(self.CH))
            self.kappa = nn.Parameter(torch.full((3,), math.log(5.0))); self.theta = nn.Parameter(torch.zeros(3)); cd = self.CH
        chan = (0, 1, 1, 2)                                              # which channel each supervised class reads
        self.chan = chan
        self.cls = nn.ParameterList([nn.Parameter(torch.randn(n, cd[c]) * 0.3) for n, c in zip(self.NCLS, chan)])
        self.sel_bias = nn.Parameter(torch.tensor(0.0)); self.sel_scale = nn.Parameter(torch.tensor(1.0))

    def embed(self, tok):
        h = self.emb(tok)
        if self.tree:
            o = (tok - O_OBJ).clamp(0, NOBJ - 1); is_o = ((tok >= O_OBJ) & (tok < O_CAT))[..., None]
            c = (tok - O_CAT).clamp(0, NCAT - 1); is_c = ((tok >= O_CAT) & (tok < O_NUM))[..., None]
            h = torch.where(is_o, self.c_cat[o // NLEAF] + self.c_leaf[o % NLEAF], h)
            h = torch.where(is_c, self.c_cat[c], h)
        return h

    def channels(self, x, lin):
        if not self.bindr:
            return list(torch.split(lin(x), self.CH, -1)), None
        out, parts = 0, []
        for c in range(3):
            u = lin[c](x); ur, ui = u[..., :self.P], u[..., self.P:]
            cr, sr = torch.cos(self.role[c]), torch.sin(self.role[c])
            out = out + torch.complex(ur * cr - ui * sr, ur * sr + ui * cr)
        for c in range(3):                                                # the unbound views, for supervision
            rot = torch.polar(torch.ones(self.P), -self.role[c]); z = out * rot
            parts.append(torch.cat([z.real, z.imag], -1))
        return parts, out

    def forward(self, bt):
        tok = bt["tok"]; B, L = tok.shape
        h = self.enc(self.embed(tok) + sinus(L)[None], src_key_padding_mask=(tok == T_PAD))
        r = h[torch.arange(B)[:, None], bt["rpos"]]
        q = self.qop(h[torch.arange(B), bt["qpos"]]).view(B, 3, D)
        qs, qb = self.channels(q, self.wq); rs, rb = self.channels(r, self.wk)
        if self.bindr:
            sc = self.log_beta.exp() * torch.einsum("bop,brp->bor", qb, rb.conj()).real / self.P
        else:
            sc = 0
            for c in range(3):
                cs = torch.einsum("bod,brd->bor", F.normalize(qs[c], dim=-1), F.normalize(rs[c], dim=-1))
                sc = sc + F.logsigmoid(self.kappa[c].exp() * (cs - self.theta[c]))
        m = bt["rmask"][:, None, :]
        lp = F.log_softmax(sc[:, :2].masked_fill(~m, -1e9), -1)            # operand pointers
        sel_logit = (self.sel_scale * sc[:, 2] + self.sel_bias).masked_fill(~bt["rmask"], -30.0)   # the coarse multi-select
        aux = 0
        rm = bt["rmask"]; qm = bt["qfeat"][..., 0] >= 0
        for k in range(4):
            c = self.chan[k]; tab = F.normalize(self.cls[k], dim=-1)
            aux = aux + F.cross_entropy(F.normalize(rs[c], dim=-1)[rm] @ tab.T * 10, bt["rfeat"][..., k][rm])
            if qm.any():
                aux = aux + F.cross_entropy(F.normalize(qs[c][:, :2], dim=-1)[qm] @ tab.T * 10, bt["qfeat"][..., k][qm])
        return lp, sel_logit, aux


def step_loss(net, bt):
    lp, sl, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
    loss = aux + (F.nll_loss(lp[m], tg[m]) if m.any() else 0)
    cm = bt["coarse"]
    if cm.any():
        rm = bt["rmask"][cm]
        loss = loss + F.binary_cross_entropy_with_logits(sl[cm][rm], bt["sel"][cm][rm])
    return loss


def evaluate(net, ev):
    net.eval(); out = {}
    with torch.no_grad():
        for sp, bt in ev.items():
            lp, sl, _ = net(bt); pick = lp.argmax(-1)
            okp = ((pick == bt["tgt"]) | (bt["tgt"] < 0)).all(-1)
            oks = (((sl > 0).float() == bt["sel"]) | ~bt["rmask"]).all(-1)
            ok = torch.where(bt["coarse"], oks, okp).float()
            out[f"acc_{sp}"] = float(ok.mean())
            for k in ("lookup", "left", "coarse"):
                mm = torch.tensor([x == k for x in bt["kind"]])
                out[f"acc_{sp}_{k}"] = float(ok[mm].mean()) if mm.any() else float("nan")
    net.train(); return out


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    ev = {sp: batch(erng, N_EVAL, sp) for sp in ("iid", "unseen")}
    torch.manual_seed(seed); net = Net(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        loss = step_loss(net, batch(rng, BATCH, "train"))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    res = evaluate(net, ev); res["final_loss"] = float(loss.detach()); return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"smoke": "_smoke", "full": ""}[mode]
    assert mode == "smoke" or not os.path.exists(f".cache/composite_entity{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/composite_entity{tag}.txt", "w"); allres = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS} lr {LR}")
    A = {}
    for arm in ARMS:
        rs = [run(arm, s) for s in SEEDS]
        a = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}; A[arm] = a; allres.append(dict(arm=arm, seeds=rs, agg=a))
        say(f"{arm:14s} " + " | ".join(f"{sp} {a[f'acc_{sp}']:.3f} (lookup {a[f'acc_{sp}_lookup']:.3f} left {a[f'acc_{sp}_left']:.3f} coarse {a[f'acc_{sp}_coarse']:.3f})"
                                         for sp in ("iid", "unseen")) + f"  loss {a['final_loss']:.3f} [{time.time() - t0:.0f}s]")
    json.dump(allres, open(f".cache/composite_entity{tag}.json", "w"), indent=1)
    if mode == "full":                                                        # THE PINNED BARS (ledger 2026-10-22)
        g = lambda a, k: A[a][k]
        say("VERDICTS (3-seed means)")
        for rd in ("TABLE", "BIND"):
            t_, f_ = f"TREEEMB_{rd}", f"FLATEMB_{rd}"
            h1 = g(t_, "acc_unseen_coarse") >= g(f_, "acc_unseen_coarse") + 0.20 and g(t_, "acc_unseen_coarse") >= 0.80 * g(t_, "acc_iid_coarse")
            say(f"(H1) {rd}: " + ("TREE CODES CARRY COARSE QUERIES TO UNSEEN LEAVES" if h1 else "TREE CODES DO NOT CARRY COARSE QUERIES AS PINNED")
                + f"  (unseen coarse tree {g(t_, 'acc_unseen_coarse'):.3f} vs flat {g(f_, 'acc_unseen_coarse'):.3f}; tree iid coarse {g(t_, 'acc_iid_coarse'):.3f})")
            d2 = g(t_, "acc_unseen_lookup") - g(f_, "acc_unseen_lookup")
            say(f"(H2) {rd}: " + ("FINE LOOKUPS NEED NO TREE" if abs(d2) <= 0.05 else "FINE LOOKUPS DIFFER BY EMBEDDING") + f"  (unseen lookup tree − flat {d2:+.3f})")
        for e_ in ("FLATEMB", "TREEEMB"):
            d3 = max(abs(g(f"{e_}_BIND", f"acc_{sp}") - g(f"{e_}_TABLE", f"acc_{sp}")) for sp in ("iid", "unseen"))
            say(f"(H3) {e_}: " + ("COMPOSITE BINDING TIES THE COMPOSITE TABLE" if d3 <= 0.03 else "COMPOSITE BINDING DOES NOT TIE THE TABLE") + f"  (max |BIND − TABLE| over splits {d3:.3f})")
    say(f"DONE in {time.time() - t0:.0f}s")
