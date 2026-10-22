"""supervision_split_toy.py -- T2c, WHERE DOES THE +0.28 LIVE? (2026-10-22; branch binding-comprehension; Bryce's word;
Opus's three points; ledger "T2c"). T2 supervised BOTH sides at once: the RECORD side ('this 16 is Janet's') and the QUERY
side ('this operand wants Janet's apples'). The main line's measured wall is the query side, so split them:
  STRUCT_U   three-channel read, no feature supervision (the same-read-form control; tied FLAT exactly in T2)
  SUP_REC    record channels supervised only
  SUP_QRY    query channels supervised only
  SUP_BOTH   both (= T2's STRUCT_S)
  FLAT       the unstructured read (reported)
T2's questions NAME the owner, so the query side has little to learn (Opus). New here: an IMPLICIT query form where the
owner must be INFERRED from the story -- 'Q WHO OBJ LEFT ?' / 'Q WHO HAS OBJ ?' = the one entity that lost or gave OBJ
(generated only when exactly one entity has an OUT event on OBJ). One query in three is implicit in every split, train
included. The query-side label for an implicit query is the gold owner (a generator fact, never a measurement tool).
Also the S4 probe: a record whose owner mention is the entity's FIRST mention is split out of 'name' ('first').
Story generator, trunk, reads and splits otherwise = T2's.
"""
import json, math, os, sys, time
mode = "smoke" if "--smoke" in sys.argv else ("feas" if "--feas" in sys.argv else "full")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

NE, NDESC, NOBJ, NNUM = 16, 8, 8, 20
# vocab
T_PAD, T_S, T_DOT, T_Q, T_QM, T_IS, T_HAS, T_BUYS, T_LOSES, T_GIVES, T_LEFT, T_HE, T_SHE = range(13)
O_NAME = 13; O_DESC = O_NAME + NE; O_OBJ = O_DESC + NDESC; O_NUM = O_OBJ + NOBJ; T_WHO = O_NUM + NNUM + 1; VOCAB = T_WHO + 1
VERB_CLS = {T_HAS: 0, T_BUYS: 1, T_LOSES: 2, T_GIVES: 2}
D, NH, NL = 64, 4, 2
STEPS = int(os.environ.get("BM_STEPS", "300" if mode == "smoke" else "4000"))
BATCH, N_EVAL = 64, (300 if mode == "smoke" else 2000)
SEEDS = [0] if mode in ("smoke", "feas") else [0, 1, 2]
ARMS = os.environ.get("BM_ARMS", "FLAT,STRUCT_U,SUP_REC,SUP_QRY,SUP_BOTH").split(",")
LR = float(os.environ.get("BM_LR", "1e-3"))
GENDER = np.array([0] * 8 + [1] * 8)                     # entity -> he (0) / she (1)
_hold = np.random.default_rng(777).permutation(NE * NOBJ)[: NE * NOBJ // 4]
HELD = {(int(p) // NOBJ, int(p) % NOBJ) for p in _hold}  # (entity, object) never owned in training


def story(rng, n_ev, split):
    """-> tokens (list), numeral records [(pos, owner, obj, vcls, value, owner_form)], queries pool."""
    ents = rng.choice(NE, size=int(rng.integers(3, 5)), replace=False)
    desc = {int(e): int(d) for e, d in zip(ents, rng.permutation(NDESC)[: len(ents)])}
    intro = {int(e): rng.random() < 0.6 for e in ents}
    toks, recs, seen, last_by_g, introduced = [], [], set(), {}, set()
    pairs = {}

    def mention(e):
        e = int(e)
        if e not in seen:
            seen.add(e); last_by_g[GENDER[e]] = e
            if intro[e]:
                toks.extend([T_S, O_NAME + e, T_IS, O_DESC + desc[e], T_DOT]); introduced.add(e)
            return [O_NAME + e], "first"
        opts = [("name", [O_NAME + e])]
        if last_by_g.get(GENDER[e]) == e:
            opts.append(("pronoun", [T_HE if GENDER[e] == 0 else T_SHE]))
        if e in introduced:
            opts.append(("desc", [O_DESC + desc[e]]))
        form, tk = opts[int(rng.integers(len(opts)))]
        last_by_g[GENDER[e]] = e
        return tk, form

    tries = 0
    while len(recs) < n_ev and tries < 200:
        tries += 1
        open_ = [p for p, s in pairs.items() if len(s) < 3]
        if open_ and rng.random() < 0.6:                                   # most later events revisit an owned pair
            e, o = open_[int(rng.integers(len(open_)))]
        else:
            e = int(rng.choice(ents)); o = int(rng.integers(NOBJ))
        if (split == "train" and (e, o) in HELD) or (split == "combo" and (e, o) not in HELD and rng.random() < 0.7):
            continue
        st = pairs.setdefault((e, o), set())
        if 0 not in st:
            verb = T_HAS
        else:
            cand = [v for v, c in ((T_BUYS, 1), (T_LOSES, 2), (T_GIVES, 2)) if c not in st]
            if not cand:
                continue
            verb = cand[int(rng.integers(len(cand)))]
        if verb == T_GIVES:
            others = [int(x) for x in ents if int(x) != e]
            r = others[int(rng.integers(len(others)))]
        n = int(rng.integers(1, NNUM + 1))
        mk, form = mention(e)
        toks.extend([T_S] + mk + [verb])
        if verb == T_GIVES:
            rk, _ = mention(r); toks.extend(rk)
        recs.append((len(toks), e, o, VERB_CLS[verb], n, form))
        toks.extend([O_NUM + n - 1, O_OBJ + o, T_DOT])
        st.add(VERB_CLS[verb])
    return toks, recs, desc, introduced


def example(rng, split):
    n_ev = int(rng.integers(8, 11)) if split == "long" else int(rng.integers(3, 6))
    while True:
        toks, recs, desc, introduced = story(rng, n_ev, split)
        if split == "combo":
            pool = [r for r in recs if (r[1], r[2]) in HELD]
        else:
            pool = list(recs)
        if pool:
            break
    outs = {}
    for r in recs:
        if r[3] == 2:
            outs.setdefault(r[2], []).append(r[1])
    ipool = [r for r in pool if len(outs.get(r[2], [])) == 1 and outs[r[2]][0] == r[1]]
    if ipool and rng.random() < 1 / 3:                                 # IMPLICIT: the owner is 'the one who lost/gave OBJ'
        tgt = ipool[int(rng.integers(len(ipool)))]; e, o = tgt[1], tgt[2]
        same = {r[3]: r for r in recs if r[1] == e and r[2] == o}
        if rng.random() < 0.5:
            q = [T_Q, T_WHO, O_OBJ + o, T_LEFT, T_QM]; ops = [same[0], same[2]]; kind = "left"
        else:
            q = [T_Q, T_WHO, T_HAS, O_OBJ + o, T_QM]; ops = [same[0]]; kind = "lookup"
        return dict(toks=toks + q, recs=recs, ops=ops, kind=kind, qform="implicit")
    tgt = pool[int(rng.integers(len(pool)))]
    e, o = tgt[1], tgt[2]
    same = {r[3]: r for r in recs if r[1] == e and r[2] == o}
    qform = "desc" if (e in introduced and rng.random() < 0.5) else "name"
    qo = [O_DESC + desc[e]] if qform == "desc" else [O_NAME + e]
    if 2 in same and rng.random() < 0.5:                               # LEFT: has - out
        q = [T_Q] + qo + [O_OBJ + o, T_LEFT, T_QM]; ops = [same[0], same[2]]; kind = "left"
    else:
        vt = {0: T_HAS, 1: T_BUYS, 2: None}[tgt[3]]
        if vt is None:                                                 # the OUT verb as written in the story
            vt = T_GIVES if any(t == T_GIVES for t in toks[tgt[0] - 4: tgt[0]]) else T_LOSES
        q = [T_Q] + qo + [vt, O_OBJ + o, T_QM]; ops = [tgt]; kind = "lookup"
    return dict(toks=toks + q, recs=recs, ops=ops, kind=kind, qform=qform)


def batch(rng, B, split):
    exs = [example(rng, split) for _ in range(B)]
    L = max(len(x["toks"]) for x in exs); R = max(len(x["recs"]) for x in exs)
    tok = np.full((B, L), T_PAD); rpos = np.zeros((B, R), np.int64); rmask = np.zeros((B, R), bool)
    rfeat = np.zeros((B, R, 3), np.int64); qpos = np.zeros(B, np.int64)
    tgt = np.full((B, 2), -1); nop = np.zeros(B, np.int64)
    rform = []
    for b, x in enumerate(exs):
        tok[b, :len(x["toks"])] = x["toks"]; qpos[b] = len(x["toks"]) - 1
        for k, r in enumerate(x["recs"]):
            rpos[b, k] = r[0]; rmask[b, k] = True; rfeat[b, k] = (r[1], r[2], r[3])
        for i, op in enumerate(x["ops"]):
            tgt[b, i] = x["recs"].index(op)
        nop[b] = len(x["ops"])
        rform.append(x["ops"][0][5])
    t = lambda a: torch.from_numpy(np.asarray(a))
    qfeat = np.full((B, 2, 3), -1)
    for b, x in enumerate(exs):
        for i, op in enumerate(x["ops"]):
            qfeat[b, i] = (op[1], op[2], op[3])
    return dict(tok=t(tok), rpos=t(rpos), rmask=t(rmask), rfeat=t(rfeat), qpos=t(qpos), tgt=t(tgt), nop=t(nop),
                qfeat=t(qfeat), kind=[x["kind"] for x in exs], rform=rform, qform=[x["qform"] for x in exs])


def sinus(L):
    p = torch.arange(L)[:, None].float(); i = torch.arange(0, D, 2).float()
    a = p / (10000 ** (i / D)); pe = torch.zeros(L, D); pe[:, 0::2] = torch.sin(a); pe[:, 1::2] = torch.cos(a)
    return pe


class Net(nn.Module):
    CH = (21, 21, 22)
    NCLS = (NE, NOBJ, 3)

    def __init__(self, arm):
        super().__init__(); self.arm = arm
        self.emb = nn.Embedding(VOCAB, D)
        lay = nn.TransformerEncoderLayer(D, NH, 4 * D, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(lay, NL)
        self.qop = nn.Linear(D, 2 * D)                                     # two operand queries from '?'
        if arm == "FLAT":
            self.wq, self.wk = nn.Linear(D, D), nn.Linear(D, D)
        elif arm.startswith("STRUCT") or arm.startswith("SUP"):
            self.wq, self.wk = nn.Linear(D, sum(self.CH)), nn.Linear(D, sum(self.CH))
            self.kappa = nn.Parameter(torch.full((3,), math.log(5.0))); self.theta = nn.Parameter(torch.zeros(3))
            self.cls = nn.ParameterList([nn.Parameter(torch.randn(n, c) * 0.3) for n, c in zip(self.NCLS, self.CH)])
        else:                                                              # BIND
            self.P = 32
            g = torch.Generator().manual_seed(4242)
            self.register_buffer("role", torch.rand(3, self.P, generator=g) * 2 * math.pi)
            self.wq = nn.ModuleList([nn.Linear(D, 2 * self.P) for _ in range(3)])
            self.wk = nn.ModuleList([nn.Linear(D, 2 * self.P) for _ in range(3)])
            self.log_beta = nn.Parameter(torch.tensor(math.log(8.0)))
            self.cls = nn.ParameterList([nn.Parameter(torch.randn(n, 2 * self.P) * 0.3) for n in self.NCLS])

    def bind(self, h, lin):
        out = 0
        for c in range(3):
            u = lin[c](h); ur, ui = u[..., :self.P], u[..., self.P:]
            cr, sr = torch.cos(self.role[c]), torch.sin(self.role[c])
            out = out + torch.complex(ur * cr - ui * sr, ur * sr + ui * cr)
        return out

    def forward(self, bt):
        tok = bt["tok"]; B, L = tok.shape
        h = self.enc(self.emb(tok) + sinus(L)[None], src_key_padding_mask=(tok == T_PAD))
        r = h[torch.arange(B)[:, None], bt["rpos"]]                       # (B, R, D)
        q = self.qop(h[torch.arange(B), bt["qpos"]]).view(B, 2, D)         # (B, 2, D)
        aux = 0.0
        if self.arm == "FLAT":
            sc = torch.einsum("bod,brd->bor", self.wq(q), self.wk(r)) / math.sqrt(D)
        elif self.arm.startswith("STRUCT") or self.arm.startswith("SUP"):
            qs, rs = torch.split(self.wq(q), self.CH, -1), torch.split(self.wk(r), self.CH, -1)
            sc = 0
            for c in range(3):
                cs = torch.einsum("bod,brd->bor", F.normalize(qs[c], dim=-1), F.normalize(rs[c], dim=-1))
                sc = sc + F.logsigmoid(self.kappa[c].exp() * (cs - self.theta[c]))
            if self.arm.startswith("SUP"):
                aux = self.sup(qs, rs, bt, lambda x, c: F.normalize(x, dim=-1) @ F.normalize(self.cls[c], dim=-1).T * 10)
        else:
            qb, rb = self.bind(q, self.wq), self.bind(r, self.wk)
            sc = self.log_beta.exp() * torch.einsum("bop,brp->bor", qb, rb.conj()).real / self.P
            if self.arm == "BIND_S":
                def unb(x, c):
                    rot = torch.polar(torch.ones(self.P), -self.role[c])
                    return torch.cat([(x * rot).real, (x * rot).imag], -1)
                qs = [unb(qb, c) for c in range(3)]; rs = [unb(rb, c) for c in range(3)]
                aux = self.sup(qs, rs, bt, lambda x, c: F.normalize(x, dim=-1) @ F.normalize(self.cls[c], dim=-1).T * 10)
        sc = sc.masked_fill(~bt["rmask"][:, None, :], -1e9)
        return F.log_softmax(sc, -1), aux

    def sup(self, qs, rs, bt, logit):
        loss = 0
        m = bt["rmask"]; qm = bt["qfeat"][..., 0] >= 0
        for c in range(3):
            lr_ = logit(rs[c], c); lq = logit(qs[c], c)
            if self.arm in ("SUP_REC", "SUP_BOTH"):
                loss = loss + F.cross_entropy(lr_[m], bt["rfeat"][..., c][m])
            if self.arm in ("SUP_QRY", "SUP_BOTH"):
                loss = loss + F.cross_entropy(lq[qm], bt["qfeat"][..., c][qm])
        return loss


def evaluate(net, ev):
    net.eval(); out = {}
    with torch.no_grad():
        for sp, bt in ev.items():
            lp, _ = net(bt); pick = lp.argmax(-1)                            # (B, 2)
            okop = (pick == bt["tgt"]) | (bt["tgt"] < 0)
            ok = okop.all(-1).float()
            out[f"acc_{sp}"] = float(ok.mean())
            for k in ("lookup", "left"):
                m = torch.tensor([x == k for x in bt["kind"]])
                out[f"acc_{sp}_{k}"] = float(ok[m].mean()) if m.any() else float("nan")
            for f_ in ("first", "name", "pronoun", "desc"):
                m = torch.tensor([x == f_ for x in bt["rform"]])
                out[f"acc_{sp}_r{f_}"] = float(okop[:, 0][m].float().mean()) if m.any() else float("nan")
                out[f"n_{sp}_r{f_}"] = int(m.sum())
            for f_ in ("name", "desc", "implicit"):
                m = torch.tensor([x == f_ for x in bt["qform"]])
                out[f"acc_{sp}_q{f_}"] = float(ok[m].mean()) if m.any() else float("nan")
                out[f"n_{sp}_q{f_}"] = int(m.sum())
            m = torch.tensor([x != "implicit" for x in bt["qform"]])
            out[f"acc_{sp}_qexplicit"] = float(ok[m].mean())
    net.train(); return out


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(90_000 + seed)
    ev = {sp: batch(erng, N_EVAL, sp) for sp in ("iid", "combo", "long")}
    torch.manual_seed(seed); net = Net(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = batch(rng, BATCH, "train"); lp, aux = net(bt)
        tg = bt["tgt"]; m = tg >= 0
        loss = F.nll_loss(lp[m], tg[m]) + (aux if isinstance(aux, torch.Tensor) else 0.0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    res = evaluate(net, ev); res["final_loss"] = float(loss.detach()); return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"smoke": "_smoke", "feas": "_feas", "full": ""}[mode]
    assert mode == "smoke" or not os.path.exists(f".cache/supervision_split{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/supervision_split{tag}.txt", "w"); allres = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS} lr {LR}")
    A = {}
    for arm in ARMS:
        rs = [run(arm, s) for s in SEEDS]
        a = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}; A[arm] = a; allres.append(dict(arm=arm, seeds=rs, agg=a))
        say(f"{arm:9s} " + " | ".join(f"{sp} {a[f'acc_{sp}']:.3f} (query name/desc/IMPLICIT {a[f'acc_{sp}_qname']:.3f}/{a[f'acc_{sp}_qdesc']:.3f}/"
            f"{a[f'acc_{sp}_qimplicit']:.3f}; rec first/name/pron/desc {a[f'acc_{sp}_rfirst']:.3f}/{a[f'acc_{sp}_rname']:.3f}/"
            f"{a[f'acc_{sp}_rpronoun']:.3f}/{a[f'acc_{sp}_rdesc']:.3f})" for sp in ("iid", "combo", "long"))
            + f"  loss {a['final_loss']:.3f} [{time.time() - t0:.0f}s]")
        say(f"          n combo query name/desc/implicit {a['n_combo_qname']:.0f}/{a['n_combo_qdesc']:.0f}/{a['n_combo_qimplicit']:.0f}; "
            f"rec first/name/pron/desc {a['n_combo_rfirst']:.0f}/{a['n_combo_rname']:.0f}/{a['n_combo_rpronoun']:.0f}/{a['n_combo_rdesc']:.0f}")
    json.dump(allres, open(f".cache/supervision_split{tag}.json", "w"), indent=1)
    if mode == "feas":
        best = max(A[a]["acc_iid"] for a in A)
        ok = best >= 0.70 and A[ARMS[0]]["n_iid_qimplicit"] >= 200
        say(f"FEASIBILITY GATE (some arm IID >= 0.70 at seed 0, >= 200 implicit queries in IID): {'PASS' if ok else 'FAIL -> bars void, redesign'} (best {best:.3f})")
    if mode == "full":                                                        # THE PINNED BARS (ledger 2026-10-22, T2c)
        g = lambda a, k: A[a][k]
        say("VERDICTS (3-seed means, COMBO split)")
        r, q, b, u = (g(x, "acc_combo_qexplicit") for x in ("SUP_REC", "SUP_QRY", "SUP_BOTH", "STRUCT_U"))
        q1 = r >= 0.95 and q <= u + 0.10
        say("(Q1) " + ("EXPLICIT QUERIES: THE RECORD SIDE CARRIES IT" if q1 else "EXPLICIT QUERIES: NOT THE RECORD SIDE ALONE")
            + f"  (explicit-query COMBO: SUP_REC {r:.3f} [bar >= 0.95], SUP_QRY {q:.3f} vs STRUCT_U {u:.3f} [bar <= +0.10], SUP_BOTH {b:.3f})")
        ri, qi, bi, ui = (g(x, "acc_combo_qimplicit") for x in ("SUP_REC", "SUP_QRY", "SUP_BOTH", "STRUCT_U"))
        q2 = bi - ri >= 0.10
        say("(Q2) " + ("IMPLICIT QUERIES: QUERY SUPERVISION EARNS ITS KEEP" if q2 else "IMPLICIT QUERIES: QUERY SUPERVISION ADDS < 0.10 OVER RECORDS")
            + f"  (implicit COMBO: SUP_BOTH {bi:.3f} − SUP_REC {ri:.3f} = {bi - ri:+.3f}; SUP_QRY {qi:.3f}; STRUCT_U {ui:.3f})")
        q3 = bi >= 0.90
        say("(Q3) " + ("IMPLICIT QUERIES ARE SOLVABLE WITH BOTH SIDES SUPERVISED" if q3 else "IMPLICIT QUERIES STAY HARD EVEN FULLY SUPERVISED")
            + f"  (SUP_BOTH implicit COMBO {bi:.3f} [bar >= 0.90] vs its explicit {b:.3f})")
        fl = "FLAT"
        fi, nl, pr, de = (g(fl, f"acc_combo_r{k}") for k in ("first", "name", "pronoun", "desc"))
        p4 = fi <= nl - 0.05 and abs(nl - (pr + de) / 2) <= 0.05
        say("(P4) " + ("FIRST MENTIONS EXPLAIN S4" if p4 else "FIRST MENTIONS DO NOT EXPLAIN S4")
            + f"  (FLAT COMBO record forms: first {fi:.3f}, later name {nl:.3f}, pronoun {pr:.3f}, desc {de:.3f})")
        say("(S5) LENGTH (reported): " + ", ".join(f"{a} {g(a, 'acc_long'):.3f}" for a in ARMS))
    say(f"DONE in {time.time() - t0:.0f}s")
