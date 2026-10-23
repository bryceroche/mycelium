"""tree_type_toy.py -- T2l, TYPE FROM THE PERMANENT TREE (2026-10-23; Bryce's word; ledger 16:23). T2k's toy with the slot's
he/she/they/it type read off its head-token EMBEDDING replaced by the slot's bound TREE NODE (written once at slot creation):
TREE (one-hot node), TREE_BOUND (node stored by phasor binding to the slot's composite key slot*time*level and recovered by
unbind + cleanup; mycelium/complex_tensor.py numpy backend), TREE_N10 (10 % of persons bound to the wrong gender node).
--- T2k's docstring follows ---
role_structure_toy.py -- T2k, ROLE AS STRUCTURE vs DEPTH (2026-10-23; Bryce's word; ledger 14:14 "T2k"). A copy of T2j's
learned_resolver_toy.py with the generator, seeds, eval sets, reader, query hop and HYBRID resolver UNCHANGED (episodes are
byte-identical: the previous sentence's (subject, recipient) is recorded without any rng draw). Arms: HYBRID (control),
HYBRID_NL4 (depth), HYBRID_ROLE (+ per-slot is-prev-subject / is-prev-recipient flags on pronoun mentions, gold parse, learned
weights), HYBRID_ROLE_N15 (flags noised 15 %: swapped when both exist, else dropped; own generator).
--- T2j's docstring follows ---
learned_resolver_toy.py -- T2j, A LEARNED RESOLVER ON REALISTIC MENTIONS (2026-10-23; Bryce's word; Opus's three cells;
ledger "T2j").  T2i closed the blind query with entity slots, but resolved names/descriptions BY CONSTRUCTION (frozen-embedding
equality).  Here mentions are realistic: two-token names (FIRST SUR) with aliases (FIRST / MR|MS SUR / SUR), families sharing
a surname (title gender separates them), possessive groups (FIRST 'S TEAM -> the possessor trap), businesses (HEAD, 'it'),
groups ('they'), and pronouns resolved by ROLE: a pronoun refers to an entity of the PREVIOUS sentence, the subject winning
when subject and recipient share the pronoun (the same-gender cell, where token recency points at the recipient).
FIRST names, surnames and descriptions use FROZEN random embeddings (first names carry a +-2 gender direction); 4 first names
are HELD OUT of training (the heldout split's giver).  The reader and query hop are T2i's SLOT (blind questions keyed by the
OUT record's slot distribution, hard snap at read).  Only the RESOLVER differs between arms:
  EXACT    the strongest hand rule we would ship: token OVERLAP of the mention's content tokens against the slot's keys
           (first/surname/desc, or possessor/head) + a learned slot-kind bias + a learned TYPE read (he/she/they/it from the
           slot's head token; titles MR/MS use it too) + a recency rank (last key-token occurrence) for pronouns.  T2i's form,
           generalized.
  LEARNED  contextual only: (Wm mean_h(mention span)) . (Ws mean_h(slot's first-mention span)); no overlap, no rank feature.
  HYBRID   EXACT + LEARNED logits summed.
Slots are created at FIRST mention (always the full form) -- the span is structural (spaCy gives NPs); only later mentions
are resolved.  Cells: full / first / title / title_family / sur / desc / poss / head / pron_gender / pron_role / they / it.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from mycelium import complex_tensor as CT

mode = "smoke" if "--smoke" in sys.argv else "full"
STEPS = int(os.environ.get("BM_STEPS", "300" if mode == "smoke" else "4000"))
BATCH, LR = 64, 1e-3
N_EVAL = 300 if mode == "smoke" else 6000
SEEDS = [0] if mode == "smoke" else [0, 1, 2]
ARMS = os.environ.get("TL_ARMS", "HYBRID_ROLE,HYBRID_ROLE_TREE,HYBRID_ROLE_TREEB,HYBRID_ROLE_TREE_N10").split(",")
OUT_TXT = ".cache/tree_type{}.txt"
D, NH, NL = 64, 4, 2
NF, NSUR, NDESC, NG, NB, NOBJ, NNUM = 16, 8, 8, 3, 3, 8, 20
(T_PAD, T_S, T_DOT, T_Q, T_QM, T_IS, T_HAS, T_BUYS, T_LOSES, T_GIVES, T_LEFT, T_HE, T_SHE, T_THEY, T_IT, T_MR, T_MS,
 T_POSS) = range(18)
O_F = 18; O_SUR = O_F + NF; O_DESC = O_SUR + NSUR; O_GH = O_DESC + NDESC; O_BH = O_GH + NG; O_OBJ = O_BH + NB
O_NUM = O_OBJ + NOBJ; VOCAB = O_NUM + NNUM
VERB_CLS = {T_HAS: 0, T_BUYS: 1, T_LOSES: 2, T_GIVES: 2}
GENDER = [0] * 8 + [1] * 8
HELDOUT = {6, 7, 14, 15}
S, MK, MC = 6, 4, 3                                                       # slots, keys per slot, content tokens per mention
PRONS = (T_HE, T_SHE, T_THEY, T_IT)
CELLS = ["full", "first", "title", "title_family", "sur", "desc", "poss", "head", "pron_gender", "pron_role", "they", "it"]
ALIAS = ["first", "title", "title_family", "sur", "poss", "head"]

_pg = np.random.default_rng(4049); PB = 64                                 # the tree's phasor codebooks (fixed)
PH_SLOT = CT.phasor(_pg.uniform(0, 2 * np.pi, (6, PB))); PH_TIME = CT.phasor(_pg.uniform(0, 2 * np.pi, (16, PB)))
PH_LEVEL = CT.phasor(_pg.uniform(0, 2 * np.pi, (3, PB))); PH_NODE = CT.phasor(_pg.uniform(0, 2 * np.pi, (4, PB)))
NODE_LEVEL = [2, 2, 1, 1]                                                  # he/she under person (level 2); they/it level 1
RECOV = dict(n=0, ok=0)
_g = torch.Generator().manual_seed(2027)
FROZEN = torch.randn(VOCAB, D, generator=_g)
GDIR = F.normalize(torch.randn(D, generator=_g), dim=0)
for f in range(NF):
    FROZEN[O_F + f] += 2.0 * (1 if GENDER[f] == 0 else -1) * GDIR
ISFROZEN = torch.zeros(VOCAB, dtype=torch.bool); ISFROZEN[O_F:O_DESC + NDESC] = True


def episode(rng, split):
    npers = int(rng.integers(2, 5))
    if split == "heldout":
        g0 = int(rng.choice(sorted(HELDOUT)))
        firsts = [g0] + [int(x) for x in rng.choice([f for f in range(NF) if f != g0], npers - 1, replace=False)]
    else:
        firsts = [int(x) for x in rng.choice([f for f in range(NF) if f not in HELDOUT], npers, replace=False)]
    surs = [int(x) for x in rng.integers(NSUR, size=npers)]
    if rng.random() < 0.5:
        i, j = rng.choice(npers, 2, replace=False); surs[j] = surs[i]
    descs = rng.permutation(NDESC)
    ents = [dict(kind="p", f=f, sur=s, pron=T_HE if GENDER[f] == 0 else T_SHE, desc=int(descs[i]) if rng.random() < 0.6 else None)
            for i, (f, s) in enumerate(zip(firsts, surs))]
    if rng.random() < 0.45:
        ents.append(dict(kind="g", head=O_GH + int(rng.integers(NG)), poss=int(rng.integers(npers)), pron=T_THEY))
    if rng.random() < 0.45:
        ents.append(dict(kind="b", head=O_BH + int(rng.integers(NB)), pron=T_IT))
    E = list(range(len(ents)))
    O = int(rng.integers(NOBJ)); k = int(rng.integers(2, min(3, len(E)) + 1))
    if split == "heldout":
        g = 0; owners = [0] + [int(x) for x in rng.choice(E[1:], k - 1, replace=False)]
    else:
        owners = [int(x) for x in rng.choice(E, k, replace=False)]; g = owners[int(rng.integers(k))]
    others = [o for o in range(NOBJ) if o != O]
    fill = [(int(rng.choice(E)), int(rng.choice(others)), int(rng.choice([T_HAS, T_BUYS, T_GIVES]))) for _ in range(int(rng.integers(1, 4)))]
    pre = [(e, O, T_HAS) for e in owners] + fill
    pre = [pre[i] for i in rng.permutation(len(pre))]
    last_has = max(i for i, ev in enumerate(pre) if ev[1] == O)
    cut = int(rng.integers(last_has + 1, len(pre) + 1))
    evs = pre[:cut] + [(g, O, T_GIVES if rng.random() < 0.5 else T_LOSES)] + pre[cut:]

    toks, slot_of, slot_span, st = [], {}, [], dict(P=None)

    def full(e):
        x = ents[e]
        if x["kind"] == "p":
            return [O_F + x["f"], O_SUR + x["sur"]]
        return [O_F + ents[x["poss"]]["f"], T_POSS, x["head"]] if x["kind"] == "g" else [x["head"]]

    def mention(e, allow_pron=True, push=False):
        x = ents[e]
        if e not in slot_of:
            slot_of[e] = len(slot_span); a = len(toks); toks.extend(full(e)); slot_span.append((a, len(toks)))
            return a, len(toks), "new"
        opts = []
        if x["kind"] == "p":
            ps = [y for y in ents if y["kind"] == "p"]
            fam = sum(y["sur"] == x["sur"] for y in ps) > 1
            opts += [("full", full(e)), ("first", [O_F + x["f"]])]
            if sum(y["sur"] == x["sur"] and y["pron"] == x["pron"] for y in ps) == 1:
                opts.append(("title_family" if fam else "title", [T_MR if x["pron"] == T_HE else T_MS, O_SUR + x["sur"]]))
            if not fam:
                opts.append(("sur", [O_SUR + x["sur"]]))
            if x.get("intro"):
                opts.append(("desc", [O_DESC + x["desc"]]))
        else:
            opts += [("poss", full(e))] if x["kind"] == "g" else []
            opts += [("head", [x["head"]])]
        P = st["P"]
        if allow_pron and P is not None and e in P and (e == P[0] or ents[P[0]]["pron"] != x["pron"]):
            role = e == P[0] and len(P) > 1 and ents[P[1]]["pron"] == x["pron"]
            form = {"g": "they", "b": "it"}.get(x["kind"], "pron_role" if role else "pron_gender")
            opts += [(form, [x["pron"]])] * 2
        pr = [i for i, (f_, _) in enumerate(opts) if f_ in ("pron_role", "pron_gender", "they", "it")]
        form, tk = opts[pr[0]] if (push and pr and rng.random() < 0.7) else opts[int(rng.integers(len(opts)))]
        a = len(toks); toks.extend(tk)
        return a, len(toks), form

    recs = []; ops = [None, None]
    evs = [(e, o, v, False) for e, o, v in evs]
    while evs:
        e, o, v, push = evs.pop(0)
        if v == T_GIVES:                                                   # same-pronoun recipients half the time (the role cell)
            same = [y for y in E if y != e and ents[y]["pron"] == ents[e]["pron"]]
            r = int(rng.choice(same)) if same and rng.random() < 0.5 else int(rng.choice([y for y in E if y != e]))
        else:
            r = -1
        for z in (e, r):                                                   # intro sentences for unseen described persons
            if z >= 0 and z not in slot_of and ents[z]["kind"] == "p" and ents[z]["desc"] is not None:
                toks.append(T_S); mention(z); toks.extend([T_IS, O_DESC + ents[z]["desc"], T_DOT]); ents[z]["intro"] = True; st["P"] = (z,)
        Pm = st["P"]                                                       # the previous sentence's (subject, recipient)
        toks.append(T_S); ma, mb, form = mention(e, push=push); toks.append(v)
        if r >= 0:
            mention(r)
        n = int(rng.integers(1, NNUM + 1))
        rec = dict(pos=len(toks), e=e, o=o, vc=VERB_CLS[v], ms=(ma, mb), form=form, r=r, v=v, P=Pm)
        recs.append(rec); toks.extend([O_NUM + n - 1, O_OBJ + o, T_DOT])
        st["P"] = (e,) if r < 0 else (e, r)
        if r >= 0 and ents[r]["pron"] == ents[e]["pron"] and rng.random() < 0.6:   # a continuation by the giver, pronoun-pushed
            evs.insert(0, (e, int(rng.choice(others)), int(rng.choice([T_HAS, T_BUYS])), True))
        if e == g and o == O:
            if v == T_HAS:
                ops[0] = rec
            elif v in (T_GIVES, T_LOSES):
                ops[1] = rec
    blind = rng.random() < 0.5; toks.append(T_Q); qs = (-1, -1)
    if not blind:
        a = len(toks); toks.extend(full(g)); qs = (a, len(toks))
    toks.extend([O_OBJ + O, T_LEFT, T_QM])
    keys = []
    for e in sorted(slot_of, key=lambda z: slot_of[z]):
        x = ents[e]
        kk = ([O_F + x["f"], O_SUR + x["sur"]] + ([O_DESC + x["desc"]] if x.get("intro") else [])) if x["kind"] == "p" else \
            ([O_F + ents[x["poss"]]["f"], x["head"]] if x["kind"] == "g" else [x["head"]])
        keys.append(kk)
    ptok = [full(e)[0] if ents[e]["kind"] == "p" else ents[e]["head"] for e in sorted(slot_of, key=lambda z: slot_of[z])]
    kinds = [{"p": 0, "g": 1, "b": 2}[ents[e]["kind"]] for e in sorted(slot_of, key=lambda z: slot_of[z])]
    stypes = [PRONS.index(ents[e]["pron"]) for e in sorted(slot_of, key=lambda z: slot_of[z])]   # the gold tree node
    return dict(toks=toks, recs=recs, ops=ops, qform="blind" if blind else "explicit", k=k, qs=qs, slot_of=slot_of,
                slot_span=slot_span, keys=keys, ptok=ptok, kinds=kinds, g=g, stypes=stypes)


def batch(rng, B, split, noise=0.0, nrng=None, tnoise=0.0, bound=False, trng=None):
    exs = [episode(rng, split) for _ in range(B)]
    L = max(len(x["toks"]) for x in exs); R = max(len(x["recs"]) for x in exs)
    tok = np.full((B, L), T_PAD); rpos = np.zeros((B, R), np.int64); rmask = np.zeros((B, R), bool)
    rfeat = np.zeros((B, R, 2), np.int64); qpos = np.zeros(B, np.int64); tgt = np.full((B, 2), -1)
    rspan = np.zeros((B, R, L), np.float32); sspan = np.zeros((B, S, L), np.float32); qspan = np.zeros((B, L), np.float32)
    rslot = np.zeros((B, R), np.int64); rform = np.full((B, R), -1); rrec = np.full((B, R), -1); rgives = np.zeros((B, R), bool)
    smask = np.zeros((B, S), bool); skeys = np.full((B, S, MK), -1); sptok = np.zeros((B, S), np.int64); skind = np.zeros((B, S), np.int64)
    mcont = np.full((B, R, MC), -1); mtitle = np.zeros((B, R), np.int64); mpron = np.full((B, R), -1)
    stype = np.full((B, S), -1, np.int64)
    rsubj = np.zeros((B, R, S), np.float32); rrecp = np.zeros((B, R, S), np.float32)
    rank = np.full((B, R, S), S); qcont = np.full((B, MC), -1); qslot = np.zeros(B, np.int64); isexp = np.zeros(B, bool)
    for b, x in enumerate(exs):
        tk = x["toks"]; tok[b, :len(tk)] = tk; qpos[b] = len(tk) - 1
        for j, (a, z) in enumerate(x["slot_span"]):
            smask[b, j] = True; sspan[b, j, a:z] = 1.0 / (z - a); skeys[b, j, :len(x["keys"][j])] = x["keys"][j]
            sptok[b, j] = x["ptok"][j]; skind[b, j] = x["kinds"][j]
            ty = x["stypes"][j]
            if tnoise > 0 and ty < 2 and trng.random() < tnoise:
                ty = 1 - ty                                                # a gazetteer miss: the wrong gender node
            if bound:                                                      # composite key (slot, time, level) bound to the node
                key = PH_SLOT[j] * PH_TIME[j % 16] * PH_LEVEL[NODE_LEVEL[ty]]
                rec = int(CT.cleanup(CT.unbind(CT.bind(PH_NODE[ty], key), key)[None], PH_NODE)[0])
                RECOV["n"] += 1; RECOV["ok"] += rec == ty; ty = rec
            stype[b, j] = ty
        ks = [set(kk) for kk in x["keys"]]

        def content(a, z):
            c = [t for t in tk[a:z] if t not in (T_MR, T_MS, T_POSS) + PRONS]
            return (c + [-1] * MC)[:MC]
        for i, r in enumerate(x["recs"]):
            a, z = r["ms"]; rpos[b, i] = r["pos"]; rmask[b, i] = True; rfeat[b, i] = (r["o"], r["vc"])
            rspan[b, i, a:z] = 1.0 / (z - a); rslot[b, i] = x["slot_of"][r["e"]]
            rform[b, i] = CELLS.index(r["form"]) if r["form"] in CELLS else -1
            rrec[b, i] = x["slot_of"][r["r"]] if r["r"] >= 0 else -1; rgives[b, i] = r["v"] == T_GIVES
            mcont[b, i] = content(a, z)
            mtitle[b, i] = {T_MR: 1, T_MS: 2}.get(tk[a], 0)
            mpron[b, i] = PRONS.index(tk[a]) if tk[a] in PRONS else -1
            P = r["P"]
            if P is not None and tk[a] in PRONS:
                sj, rc = x["slot_of"][P[0]], (x["slot_of"][P[1]] if len(P) > 1 else None)
                if noise > 0 and nrng.random() < noise:
                    sj, rc = (rc, sj) if rc is not None else (None, None)
                if sj is not None: rsubj[b, i, sj] = 1.0
                if rc is not None: rrecp[b, i, rc] = 1.0
            last = [max([p for p in range(a) if tk[p] in kk] or [-1]) for kk in ks]
            for rr, j in enumerate(sorted([j for j in range(len(ks)) if last[j] >= 0], key=lambda j: -last[j])):
                rank[b, i, j] = rr
        for i, op in enumerate(x["ops"]):
            tgt[b, i] = x["recs"].index(op)
        if x["qs"][0] >= 0:
            a, z = x["qs"]; isexp[b] = True; qspan[b, a:z] = 1.0 / (z - a); qcont[b] = content(a, z); qslot[b] = x["slot_of"][x["g"]]
    t = lambda v: torch.from_numpy(np.asarray(v))
    return dict(tok=t(tok), rpos=t(rpos), rmask=t(rmask), rfeat=t(rfeat), qpos=t(qpos), tgt=t(tgt), rspan=t(rspan), sspan=t(sspan),
                qspan=t(qspan), rslot=t(rslot), rform=t(rform), rrec=t(rrec), rgives=t(rgives), smask=t(smask), skeys=t(skeys),
                sptok=t(sptok), skind=t(skind), mcont=t(mcont), mtitle=t(mtitle), mpron=t(mpron), rank=t(rank), qcont=t(qcont),
                qslot=t(qslot), isexp=t(isexp), rsubj=t(rsubj), rrecp=t(rrecp), stype=t(stype), qform=[x["qform"] for x in exs], k=t(np.array([x["k"] for x in exs])))


def sinus(L):
    p = torch.arange(L)[:, None].float(); i = torch.arange(0, D, 2).float()
    a = p / (10000 ** (i / D)); pe = torch.zeros(L, D); pe[:, 0::2] = torch.sin(a); pe[:, 1::2] = torch.cos(a)
    return pe


class MixEmb(nn.Module):
    def __init__(self):
        super().__init__(); self.tr = nn.Embedding(VOCAB, D); self.register_buffer("fz", FROZEN.clone()); self.register_buffer("isf", ISFROZEN.clone())

    def forward(self, tok):
        return torch.where(self.isf[tok][..., None], self.fz[tok], self.tr(tok))


class Net(nn.Module):
    C = 32

    def __init__(self, arm):
        super().__init__(); self.arm = arm; self.emb = MixEmb(); self.role = "ROLE" in arm; self.tree = "TREE" in arm
        base = arm.split("_")[0]; nl = 4 if "NL4" in arm else NL
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, NH, 4 * D, dropout=0.0, batch_first=True), nl)
        self.qop = nn.Linear(D, 2 * D); self.wq = nn.Linear(D, 2 * self.C); self.wk = nn.Linear(D, 2 * self.C)
        self.kappa = nn.Parameter(torch.full((2,), math.log(5.0))); self.theta = nn.Parameter(torch.zeros(2))
        self.cls = nn.ParameterList([nn.Parameter(torch.randn(n, self.C) * 0.3) for n in (NOBJ, 3)])
        if self.role:
            self.wr = nn.Parameter(torch.zeros(2))                         # is-prev-subject / is-prev-recipient weights
        if base in ("EXACT", "HYBRID"):
            self.a = nn.Parameter(torch.tensor(5.0)); self.kb = nn.Parameter(torch.zeros(3))
            self.typ = nn.Linear(D, 4); self.rho = nn.Parameter(torch.zeros(S + 1)); self.wt = nn.Parameter(torch.ones(2))
        if base in ("LEARNED", "HYBRID"):
            self.wm = nn.Linear(D, D); self.ws = nn.Linear(D, D); self.ls = nn.Parameter(torch.tensor(math.log(4.0)))

    def exact(self, cont, title, pron, rank, bt):
        """cont (B,X,MC) content tokens, title (B,X) 0/1/2, pron (B,X) -1..3, rank (B,X,S) -> (B,X,S)."""
        E = self.emb
        ec = F.normalize(E(cont.clamp(min=0)), dim=-1) * (cont >= 0)[..., None]
        ek = F.normalize(E(bt["skeys"].clamp(min=0)), dim=-1) * (bt["skeys"] >= 0)[..., None]
        ov = torch.einsum("bxcd,bskd->bxsck", ec, ek).amax(-1).sum(-1)    # overlap: per mention token, its best key match
        tl = self.typ(E(bt["sptok"]))                                      # (B,S,4) he/she/they/it read of the slot's head token
        lg = self.a * ov + self.kb[bt["skind"]][:, None, :]
        pt = torch.gather(tl[:, None].expand(-1, pron.shape[1], -1, -1), 3, pron.clamp(min=0)[..., None, None].expand(-1, -1, S, 1))[..., 0]
        if self.tree:                                                       # read the bound tree node, not the embedding
            pt = (bt["stype"][:, None, :] == pron[..., None]).float()
        lg = lg + (pron >= 0)[..., None] * (self.wt[0] * pt + self.rho[rank])
        tt = torch.gather(tl[:, None].expand(-1, title.shape[1], -1, -1), 3, (title - 1).clamp(min=0)[..., None, None].expand(-1, -1, S, 1))[..., 0]
        if self.tree:
            tt = (bt["stype"][:, None, :] == (title - 1)[..., None]).float()
        return lg + (title > 0)[..., None] * self.wt[1] * tt

    def learned(self, span, h, hs):
        hm = torch.einsum("bxl,bld->bxd", span, h)
        return self.ls.exp() * torch.einsum("bxd,bsd->bxs", F.normalize(self.wm(hm), dim=-1), F.normalize(self.ws(hs), dim=-1))

    def resolve(self, h, bt):
        hs = torch.einsum("bsl,bld->bsd", bt["sspan"], h)
        B = h.shape[0]; z = torch.zeros(B, 1, dtype=torch.long)
        rl = 0; ql = 0
        base = self.arm.split("_")[0]
        if base in ("EXACT", "HYBRID"):
            rl = rl + self.exact(bt["mcont"], bt["mtitle"], bt["mpron"], bt["rank"], bt)
            ql = ql + self.exact(bt["qcont"][:, None], z, z - 1, torch.full((B, 1, S), S), bt)
        if self.role:
            rl = rl + (bt["mpron"] >= 0)[..., None] * (self.wr[0] * bt["rsubj"] + self.wr[1] * bt["rrecp"])
        if base in ("LEARNED", "HYBRID"):
            rl = rl + self.learned(bt["rspan"], h, hs); ql = ql + self.learned(bt["qspan"][:, None], h, hs)
        m = ~bt["smask"][:, None, :]
        return rl.masked_fill(m, -1e9), ql.masked_fill(m, -1e9)[:, 0]

    def chan2(self, q, rs):
        sc = 0
        for c in range(2):
            cs = torch.einsum("bd,brd->br", F.normalize(q[c], dim=-1), F.normalize(rs[c], dim=-1))
            sc = sc + F.logsigmoid(self.kappa[c].exp() * (cs - self.theta[c]))
        return sc

    def forward(self, bt):
        tok = bt["tok"]; B, L = tok.shape; M = bt["rmask"]; ar = torch.arange(B)
        h = self.enc(self.emb(tok) + sinus(L)[None], src_key_padding_mask=(tok == T_PAD))
        rs = torch.split(self.wk(h[ar[:, None], bt["rpos"]]), self.C, -1)
        q = self.qop(h[ar, bt["qpos"]]).view(B, 2, D)
        q0s, q1s = torch.split(self.wq(q[:, 0]), self.C, -1), torch.split(self.wq(q[:, 1]), self.C, -1)
        self._rl, ql = self.resolve(h, bt); prec = F.softmax(self._rl, -1); pq = F.softmax(ql, -1)
        ex = bt["isexp"].float()
        sc1 = (self.chan2(q1s, rs) + ex[:, None] * torch.log(torch.einsum("bs,brs->br", pq, prec) + 1e-6)).masked_fill(~M, -1e9)
        p1 = F.softmax(sc1, -1) if self.training else F.one_hot(sc1.argmax(-1), sc1.shape[-1]).float()
        key = torch.einsum("br,brs->bs", p1, prec)
        if not self.training:
            key = F.one_hot(key.argmax(-1), S).float()                        # the snap, onto the episode's slots
        key = ex[:, None] * pq + (1 - ex[:, None]) * key
        sc0 = (self.chan2(q0s, rs) + torch.log(torch.einsum("bs,brs->br", key, prec) + 1e-6)).masked_fill(~M, -1e9)
        mx = bt["isexp"]
        aux = F.cross_entropy(self._rl[M], bt["rslot"][M]) + (F.cross_entropy(ql[mx], bt["qslot"][mx]) if mx.any() else 0.0)
        for c in range(2):
            aux = aux + F.cross_entropy((F.normalize(rs[c], dim=-1) @ F.normalize(self.cls[c], dim=-1).T * 10)[M], bt["rfeat"][..., c][M])
        return F.log_softmax(torch.stack([sc0, sc1], 1), -1), aux


def evaluate(net, ev):
    out = {}; net.eval()
    with torch.no_grad():
        for sp, bt in ev.items():
            lp, _ = net(bt); okop = lp.argmax(-1) == bt["tgt"]; ok = okop.all(-1).float()
            mb = torch.tensor([x == "blind" for x in bt["qform"]])
            out[f"{sp}_acc_blind"] = float(ok[mb].mean()); out[f"{sp}_acc_explicit"] = float(ok[~mb].mean())
            pick = net._rl.argmax(-1); okr = pick == bt["rslot"]; M = bt["rmask"]
            for ci, c in enumerate(CELLS):
                m = M & (bt["rform"] == ci); out[f"{sp}_n_{c}"] = int(m.sum()); out[f"{sp}_res_{c}"] = float(okr[m].float().mean()) if m.any() else float("nan")
            for nm, cs in (("alias", ALIAS), ("all", CELLS)):
                m = M & torch.isin(bt["rform"], torch.tensor([CELLS.index(c) for c in cs])); out[f"{sp}_res_{nm}"] = float(okr[m].float().mean())
            mg = M & bt["rgives"] & (bt["rform"] >= 0)
            out[f"{sp}_gives_err"] = float((~okr[mg]).float().mean())
            out[f"{sp}_gives_on_recipient"] = float((pick[mg] == bt["rrec"][mg]).float().mean())
            ar = torch.arange(len(mb)); t1 = bt["tgt"][:, 1]
            out[f"{sp}_outres_blind"] = float((pick[ar, t1] == bt["rslot"][ar, t1])[mb].float().mean())
    net.train(); return out


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    nz = 0.15 if arm.endswith("_N15") else 0.0; nrng = np.random.default_rng(77_000 + seed); envg = np.random.default_rng(93_000 + seed)
    tn = 0.10 if arm.endswith("_N10") else 0.0; bd = arm.endswith("TREEB"); trng = np.random.default_rng(55_000 + seed); tev = np.random.default_rng(57_000 + seed)
    ev = {"indist": batch(erng, N_EVAL, "train", nz, envg, tn, bd, tev), "heldout": batch(erng, N_EVAL, "heldout", nz, envg, tn, bd, tev)}
    torch.manual_seed(seed); net = Net(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = batch(rng, BATCH, "train", nz, nrng, tn, bd, trng); lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
        loss = F.nll_loss(lp[m], tg[m]) + aux
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return evaluate(net, ev)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    OUT = OUT_TXT.format("_smoke" if mode == "smoke" else "")
    assert mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS} seeds {SEEDS} heldout first names {sorted(HELDOUT)}")
    for arm in ARMS:
        rs = [run(arm, s) for s in SEEDS]
        a = {kk: float(np.nanmean([x[kk] for x in rs])) for kk in rs[0]}; A[arm] = a; allres.append(dict(arm=arm, seeds=rs, agg=a))
        for sp in ("indist", "heldout"):
            say(f"{arm:15s} {sp:7s} | blind {a[f'{sp}_acc_blind']:.3f} explicit {a[f'{sp}_acc_explicit']:.3f} | OUT-owner {a[f'{sp}_outres_blind']:.3f}"
                f" | resolution all {a[f'{sp}_res_all']:.3f} alias {a[f'{sp}_res_alias']:.3f} | GIVES err {a[f'{sp}_gives_err']:.3f}"
                f" (on recipient {a[f'{sp}_gives_on_recipient']:.3f})")
            say("        cells: " + " ".join(f"{c} {a[f'{sp}_res_{c}']:.3f}(n{a[f'{sp}_n_{c}']:.0f})" for c in CELLS) + f"  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    say(f"TREE_BOUND type recovery by unbind+cleanup: {RECOV['ok']}/{RECOV['n']} = {RECOV['ok'] / max(1, RECOV['n']):.4f}")
    if mode == "full" and all(x in A for x in ARMS):                       # PINNED (ledger 16:23, T2l)
        c, t, tb, tn = (A[k] for k in ARMS)
        say("VERDICTS (3-seed means)")
        l1 = (t["heldout_res_pron_role"] >= 0.90 and all(t[f"indist_res_{k}"] >= c[f"indist_res_{k}"] - 0.01 for k in CELLS)
              and t["heldout_res_all"] >= c["heldout_res_all"] - 0.02)
        say(f"L1 tree type: {'PASS' if l1 else 'FAIL'} (held-out role {t['heldout_res_pron_role']:.3f} vs control {c['heldout_res_pron_role']:.3f}; "
            f"worst in-dist cell delta {min(t[f'indist_res_{k}'] - c[f'indist_res_{k}'] for k in CELLS):+.3f}; held-out all {t['heldout_res_all']:.3f} vs {c['heldout_res_all']:.3f})")
        l2 = (abs(tb["heldout_res_pron_role"] - t["heldout_res_pron_role"]) <= 0.01 and abs(tb["indist_res_all"] - t["indist_res_all"]) <= 0.01
              and RECOV["ok"] == RECOV["n"])
        say(f"L2 lossless binding: {'PASS' if l2 else 'FAIL'} (held-out role {tb['heldout_res_pron_role']:.3f} vs {t['heldout_res_pron_role']:.3f}; "
            f"in-dist all {tb['indist_res_all']:.3f} vs {t['indist_res_all']:.3f}; recovery {RECOV['ok']}/{RECOV['n']})")
        say(f"L3 gazetteer misses: {'PASS' if tn['heldout_res_pron_role'] >= 0.85 else 'FAIL'} (N10 held-out role {tn['heldout_res_pron_role']:.3f})")
    say(f"DONE in {time.time() - t0:.0f}s")
