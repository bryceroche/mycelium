"""bound_memory_toy.py -- T2, THE BOUND WORKING MEMORY (2026-10-22; branch binding-comprehension; Bryce's word; Opus's
three additions; ledger "T2"). Does a STRUCTURED query (owner / object / verb expectations) fix the 'blind wrong pick'
the main line measured (filter_f, 10-21), does supervision matter, and does BINDING beat a plain slot table?

STORIES. 16 entities with names (8 he / 8 she), 8 descriptors ('the farmer'), 8 objects, numbers 1..20. Events:
  HAS:   S <owner> HAS n OBJ .          BUYS:  S <owner> BUYS n OBJ .
  LOSES: S <owner> LOSES n OBJ .        GIVES: S <owner> GIVES <recipient> n OBJ .
  INTRO: S NAME IS DESC .   (an alias introduced in the text; the pairing is random per story)
A mention after the first is the NAME, a PRONOUN (only when the most recent mentioned entity of that gender is this one)
or the DESCRIPTOR (only once introduced). Each (owner, object) has one HAS, at most one BUYS and at most one OUT
(GIVES or LOSES), so every lookup is unique.
QUERIES (appended after the story; the query's owner is a NAME or an introduced DESCRIPTOR):
  LOOKUP  Q <owner> VERB OBJ ?        -> one pointer: the numeral of that event
  LEFT    Q <owner> OBJ LEFT ?        -> two pointers, operand 1 = the HAS numeral, operand 2 = the OUT numeral; a
                                         CALCULATOR returns v1 - v2 (the reading -> calculator bridge). Correct = both.
Every arm points only at numeral tokens (the main line's numeral legality). A value no token carries is NOT a lookup
target here (true derived addressing = T2b; scope said plainly).

THE TRUNK (shared): token embeddings + fixed sinusoidal positions + a 2-layer bidirectional transformer (d 64, 4 heads).
Records = the trunk states at numeral tokens; operand queries = learned operand projections of the state at '?'.
READS (equal memory per record, 64 floats):
  FLAT      score = (Wq q) . (Wk r)                                   (the main line's unstructured query)
  STRUCT_U  three channels (owner 21 / object 21 / verb 22 dims); score = sum_c log sigmoid(kappa (cos(q_c, r_c) - theta))
            (gate-then-multiply across channels), no feature supervision
  STRUCT_S  STRUCT_U + every channel of every record AND every operand query classified against a shared class table
            (owner entity / object / verb class {HAS, BUYS, OUT}) with gold CE  (the attention-bootstrap law)
  BIND      records and queries written as sum_c P_c * u_c (32 complex planes, fixed random role phasors per channel);
            score = beta * Re<q, r> / 32; no feature supervision
  BIND_S    BIND + each channel UNBOUND (conj(P_c) * r) and classified against the class table with gold CE
Splits: IID (3-5 owner events), COMBO (25 % of (entity, object) pairs never own that object in training; queries on
them), LONG (8-10 events). Breakdown by the queried record's owner MENTION form (name / pronoun / descriptor) and by the
query's owner form (name / descriptor).
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
O_NAME = 13; O_DESC = O_NAME + NE; O_OBJ = O_DESC + NDESC; O_NUM = O_OBJ + NOBJ; VOCAB = O_NUM + NNUM + 1
VERB_CLS = {T_HAS: 0, T_BUYS: 1, T_LOSES: 2, T_GIVES: 2}
D, NH, NL = 64, 4, 2
STEPS = int(os.environ.get("BM_STEPS", "300" if mode == "smoke" else "4000"))
BATCH, N_EVAL = 64, (300 if mode == "smoke" else 2000)
SEEDS = [0] if mode in ("smoke", "feas") else [0, 1, 2]
ARMS = os.environ.get("BM_ARMS", "FLAT,STRUCT_U,STRUCT_S,BIND,BIND_S").split(",")
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
            return [O_NAME + e], "name"
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
        elif arm.startswith("STRUCT"):
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
        elif self.arm.startswith("STRUCT"):
            qs, rs = torch.split(self.wq(q), self.CH, -1), torch.split(self.wk(r), self.CH, -1)
            sc = 0
            for c in range(3):
                cs = torch.einsum("bod,brd->bor", F.normalize(qs[c], dim=-1), F.normalize(rs[c], dim=-1))
                sc = sc + F.logsigmoid(self.kappa[c].exp() * (cs - self.theta[c]))
            if self.arm == "STRUCT_S":
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
            loss = loss + F.cross_entropy(lr_[m], bt["rfeat"][..., c][m]) + F.cross_entropy(lq[qm], bt["qfeat"][..., c][qm])
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
            for f_ in ("name", "pronoun", "desc"):
                m = torch.tensor([x == f_ for x in bt["rform"]])
                out[f"acc_{sp}_r{f_}"] = float(okop[:, 0][m].float().mean()) if m.any() else float("nan")
                out[f"n_{sp}_r{f_}"] = int(m.sum())
            for f_ in ("name", "desc"):
                m = torch.tensor([x == f_ for x in bt["qform"]])
                out[f"acc_{sp}_q{f_}"] = float(ok[m].mean()) if m.any() else float("nan")
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
    assert mode == "smoke" or not os.path.exists(f".cache/bound_memory{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/bound_memory{tag}.txt", "w"); allres = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS} lr {LR}")
    A = {}
    for arm in ARMS:
        rs = [run(arm, s) for s in SEEDS]
        a = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}; A[arm] = a; allres.append(dict(arm=arm, seeds=rs, agg=a))
        say(f"{arm:9s} " + " | ".join(f"{sp} {a[f'acc_{sp}']:.3f} (lookup {a[f'acc_{sp}_lookup']:.3f} left {a[f'acc_{sp}_left']:.3f}; "
            f"rec name/pron/desc {a[f'acc_{sp}_rname']:.3f}/{a[f'acc_{sp}_rpronoun']:.3f}/{a[f'acc_{sp}_rdesc']:.3f}; "
            f"query name/desc {a[f'acc_{sp}_qname']:.3f}/{a[f'acc_{sp}_qdesc']:.3f})" for sp in ("iid", "combo", "long"))
            + f"  loss {a['final_loss']:.3f} [{time.time() - t0:.0f}s]")
    json.dump(allres, open(f".cache/bound_memory{tag}.json", "w"), indent=1)
    if mode == "feas":
        best = max(A[a]["acc_iid"] for a in A)
        say(f"FEASIBILITY GATE (some arm IID >= 0.70 at seed 0): {'PASS' if best >= 0.70 else 'FAIL -> bars void, redesign'} (best {best:.3f})")
    if mode == "full":                                                        # THE PINNED BARS (ledger 2026-10-22)
        g = lambda a, k: A[a][k]
        say("VERDICTS (3-seed means)")
        d1 = g("STRUCT_S", "acc_combo") - g("FLAT", "acc_combo")
        say(f"(S1) " + ("SUPERVISED STRUCTURE WINS" if d1 >= 0.15 else "SUPERVISED STRUCTURE DOES NOT WIN BY 0.15") + f"  (COMBO STRUCT_S {g('STRUCT_S', 'acc_combo'):.3f} vs FLAT {g('FLAT', 'acc_combo'):.3f}, {d1:+.3f})")
        d2 = g("STRUCT_U", "acc_combo") - g("FLAT", "acc_combo")
        say(f"(S2) " + ("STRUCTURE ALONE TIES" if abs(d2) <= 0.03 else ("STRUCTURE ALONE HELPS" if d2 > 0 else "STRUCTURE ALONE HURTS")) + f"  (STRUCT_U − FLAT {d2:+.3f})")
        d3a = g("BIND_S", "acc_combo") - g("STRUCT_S", "acc_combo"); d3b = g("BIND", "acc_combo") - g("FLAT", "acc_combo")
        say(f"(S3) " + ("BINDING TIES THE TABLE" if abs(d3a) <= 0.03 and abs(d3b) <= 0.03 else "BINDING DOES NOT TIE THE TABLE") + f"  (BIND_S − STRUCT_S {d3a:+.3f}; BIND − FLAT {d3b:+.3f})")
        lows = {a: min(("name", "pronoun", "desc"), key=lambda f_: g(a, f"acc_combo_r{f_}")) for a in ARMS}
        say(f"(S4) " + ("PRONOUNS COST MOST" if all(v == "pronoun" for v in lows.values()) else "PRONOUNS DO NOT COST MOST IN EVERY ARM") + "  (lowest form per arm on COMBO: " + ", ".join(f"{a} {v}" for a, v in lows.items()) + ")")
        say("(S5) LENGTH (reported): " + ", ".join(f"{a} {g(a, 'acc_long'):.3f} vs iid {g(a, 'acc_iid'):.3f}" for a in ARMS))
    say(f"DONE in {time.time() - t0:.0f}s")
