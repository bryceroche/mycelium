"""gsm_ladder_read.py -- THE LADDER READ (ledger 2026-11-10 pin; Bryce: "different levels of the Tree DB should hold different levels
of abstraction ... 'fresh duck eggs' should sit on the lowest layer and share a parent or grandparent with 'rotten chicken egg'").
Zero GPU, zero training. A Tree of IS-A edges over step keys: L0 = the exact IMPULSE2 span key; L1/L2/L3 = nested clusters (2048 /
256 / 32) of the reader's banked word states over that key; L4 = the operand-kind prior. Each node keeps op counts per operand kind
(TRAIN gold steps only). A query descends root -> leaf with shrinkage n/(n+kappa), kappa 0.5. The THEN-A edges are never stored.
TREE-PLACEBO = the same ladder with TRAIN vectors shuffled across steps. -> .cache/gsm_ladder.txt"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_bench_data as GD
import gsm_cj_probe_skel as SK
import gsm_trunk_reader as TR
import gsm_trunk_states as GTS

GATE = float(os.environ.get("LADDER_GATE", "0")); KSEED = int(os.environ.get("LADDER_SEED", "0"))   # (ledger 11-11 GATED / LADDER-NN)
K = 12; PUN = set(".,;:?!'\"()’"); KAPPA = 0.5; KS = (2048, 256, 32); WINDOW = 0.5047
D, ST = TR.load(); seen = SK.seen(D["TRAIN"])   # D carries st0 per item (the states' layout); GD.OUT order
OPS = sorted({o for it in D["TRAIN"] for o, _, _ in it["prog"]}); OI = {o: i for i, o in enumerate(OPS)}

# --- the IMPULSE2 leaves (identical rule to gsm_span_read.py SPAN_SRC=impulse SPAN_LOCMIN=1) ---
_, Q = GTS.texts(); IMP = np.load(".cache/gsmb_impulse_rb8.npy"); _tx = json.load(open(".cache/gsmb_impulse_rb8.npy.json"))["texts"]
_st = np.cumsum([0] + [len(GD.tokenize(q, set())[0]) for q in _tx]); _pos = {q: int(_st[i]) for i, q in enumerate(_tx)}
assert _st[-1] == len(IMP)
for sp in ("TRAIN", "TEST"):
    for it, q in zip(D[sp], Q[sp]):
        it["imp0"] = _pos[q]; it["ch"] = [m.span() for m in GD.TOK.finditer(q)]; it["q"] = q; assert len(it["ch"]) == len(it["toks"])
SRC = os.environ.get("LADDER_STATES", "full"); PARENT = os.environ.get("LADDER_PARENT", "geo")   # (ledger 11-10 LOCAL/ROLE)
LOCALF = os.environ.get("LADDER_LOCALF", ".cache/gsmb_local_rb8.npy" if os.environ.get("LADDER_SPANS", "impulse") == "impulse" else f".cache/gsmb_local_rb8_{os.environ['LADDER_SPANS']}.npy"); DIAL = os.environ.get("LADDER_DIAL") == "1"; NOVEL = os.environ.get("LADDER_NOVEL") == "1"; LTXT = []; LIDX = {}
if SRC == "local":
    LV = np.load(LOCALF); LIDX = {t: i for i, t in enumerate(json.load(open(LOCALF + ".json"))["texts"])}


def words(it): return ["<num>" if v is not None else ("<name>" if n else w) for w, v, n in zip(it["toks"], it["vals"], it["isn"])]
def cpl(it, i): s0 = it["imp0"]; return (IMP[s0 + i, 0] + IMP[s0 + i + 1, 1]) / 2
SPANS = os.environ.get("LADDER_SPANS", "impulse")   # (ledger 11-11 MRI) impulse | mri | agree (a cut only where both witnesses cut)
if SPANS != "impulse":
    MRI = np.load(".cache/gsmb_mri_rb8.npy"); assert len(MRI) == len(IMP)
def mcp(it, i): return MRI[it["imp0"] + i]


def cut(f, it, i, n):   # a boundary after word i where the coupling f dips below both neighbouring pairs
    c = f(it, i); return (i == 0 or c < f(it, i - 1)) and (i + 2 >= n or c < f(it, i + 1))


def span_ids(it):
    w = words(it); sid = it["sid"]; out, cur = [], 0
    for i in range(len(w)):
        out.append(cur)
        if i + 1 >= len(w) or w[i] in PUN or w[i + 1] in PUN or sid[i] != sid[i + 1]: cur += 1; continue
        ci = SPANS != "mri" and cut(cpl, it, i, len(w)); cm = SPANS != "impulse" and cut(mcp, it, i, len(w))
        if {"impulse": ci, "mri": cm, "agree": ci and cm}[SPANS]: cur += 1
    return out


def steps(it):
    w = words(it); sid = it["sid"]; sp = span_ids(it); last = max(sid); res = []
    for o, A, B in it["prog"][:K]:
        if A["t"] or B["t"]:
            pos = [[j for i in S["t"][:1] for j in range(len(w)) if sp[j] in (sp[i], sp[i] + 1)] for S in (A, B)]
            key = " ".join(f"{sd}:{w[j]}" for sd, P in zip("ab", pos) for j in P)
        else:
            pos = [[j for j in range(len(w)) if sid[j] == last], []]; key = "q: " + " ".join(w[j] for j in pos[0])
        v = []
        for P in pos:
            txt = it["q"][it["ch"][min(P)][0]:it["ch"][max(P)][1]] if P else None   # the key's exact substring (local form)
            if txt is not None: LTXT.append(txt)
            if SRC == "local": x = LV[LIDX[txt]].astype(np.float32) if P else np.zeros(LV.shape[1], np.float32)
            else: x = np.asarray(ST[it["st0"] + np.array(P, int)], np.float32).mean(0) if P else np.zeros(ST.shape[1], np.float32)
            v.append(x / (np.linalg.norm(x) + 1e-6))
        kd = (SK.first_kind(A), SK.first_kind(B))   # (GLUE) one row per phrase: the side's own key, vector, kind and role
        sides = [(("q: " if key.startswith("q:") else "") + " ".join(f"{'ab'[r]}:{w[j]}" for j in P), v[r], kd[r], r) for r, P in enumerate(pos) if P]
        refs = tuple(S["s"][0] if S["s"] and not S["t"] else None for S in (A, B))   # (CHAIN) which earlier step feeds each side
        res.append((key, kd, OI[o], np.concatenate(v), sides, refs, not (A["t"] or B["t"])))
    return res


tr = [s for it in D["TRAIN"] for s in steps(it)]; te = [s for it in D["TEST"] for s in steps(it)]
if os.environ.get("LADDER_DUMP") == "1":   # the local texts for distill_reader.py --local
    LT = sorted(set(LTXT)); json.dump(dict(texts=LT), open(".cache/gsm_ladder_local_texts" + ("" if SPANS == "impulse" else "_" + SPANS) + ".json", "w")); print(f"[ladder] {len(LT)} local texts"); sys.exit(0)
u = np.concatenate([[SK.skeleton(it) not in seen] * min(len(it["prog"]), K) for it in D["TEST"]]).astype(bool)
assert len(u) == len(te)
Xtr = np.stack([s[3] for s in tr]); Xte = np.stack([s[3] for s in te]); ytr = np.array([s[2] for s in tr]); yte = np.array([s[2] for s in te])
kd_tr = [s[1] for s in tr]; kd_te = [s[1] for s in te]
pca = PCA(64, random_state=0).fit(Xtr); Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
print(f"[ladder] {len(tr)} TRAIN steps, {len(te)} TEST steps (TEST-U {u.sum()}), PCA-64 var {pca.explained_variance_ratio_.sum():.3f}", flush=True)


def build(Z):
    """nested levels on TRAIN vectors Z -> (assign fn, train node ids per level L1..L3)"""
    km = MiniBatchKMeans(KS[0], random_state=KSEED, n_init=3, batch_size=4096).fit(Z); c1 = km.labels_
    maps = [np.arange(KS[0])]; C = km.cluster_centers_
    if PARENT == "role":   # (ROLE) parents in an op-discriminant space: spans that FEED THE SAME OP group together
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        C = LinearDiscriminantAnalysis().fit(Z, ytr).transform(C)
    for k in KS[1:]: maps.append(AgglomerativeClustering(k, linkage="ward").fit(C).labels_)
    def assign(X): l1 = km.predict(X); return [m[l1] for m in maps]
    return assign, [m[c1] for m in maps]


def tables(tr_nodes):
    """per level: (node, kind) -> op counts; L0 keyed by exact text; L4 by kind alone"""
    T = [defaultdict(lambda: np.zeros(len(OPS)))]
    for s in tr: T[0][(s[0], s[1])][s[2]] += 1
    for lv, nodes in enumerate(tr_nodes, 1):
        T.append(defaultdict(lambda: np.zeros(len(OPS))))
        for n, s in zip(nodes, tr): T[lv][(int(n), s[1])][s[2]] += 1
    T.append(defaultdict(lambda: np.zeros(len(OPS))))
    for s in tr: T[-1][s[1]][s[2]] += 1
    return T


def conf(c): return (c.max() + 1) / (c.sum() + len(OPS))   # (DIAL) Laplace-smoothed majority share: a radius that knows n


def read(T, keys, kinds, nodes):
    """shrinkage descent + per-level census (each level alone; an empty node falls back to its parent)"""
    L = len(T); lad, lvl = [], [[] for _ in range(L)]
    for r, (k, kd) in enumerate(zip(keys, kinds)):
        cnt = [T[0].get((k, kd))] + [T[lv].get((int(nodes[lv - 1][r]), kd)) for lv in range(1, L - 1)] + [T[-1].get(kd)]
        p = np.ones(len(OPS)) / len(OPS); alone = None; par = None
        hit = cnt[0] is not None and cnt[0].sum() > 0   # (NOVEL) the query's exact key recurs: memorize, never climb
        for lv in range(L - 1, -1, -1):
            c = cnt[lv]
            if c is not None and c.sum() > 0:
                alone = int(c.argmax())
                if lv < L - 1 and GATE and not (NOVEL and hit) and c.max() / c.sum() < GATE: lvl[lv].append(alone); continue   # (GATED) a split node: climb
                if lv < L - 1 and DIAL and conf(c) < conf(par): lvl[lv].append(alone); continue   # (DIAL) the parent is surer: climb
                p = (c + KAPPA * p) / (c.sum() + KAPPA)
            if c is not None and c.sum() > 0: par = c
            lvl[lv].append(alone if alone is not None else -1)
        lad.append(int(p.argmax()))
    return np.array(lad), [np.array(x)[::1] for x in lvl]


def run(Z_tr, tag):
    assign, trn = build(Z_tr); T = tables(trn)
    lad, lvl = read(T, [s[0] for s in te], kd_te, assign(Zte))
    rl, _ = read(T, [s[0] for s in tr], kd_tr, trn)
    acc = lambda p, m: float((p[m] == yte[m]).mean())
    cen = [(acc(x, u), acc(x, ~u)) for x in lvl]
    return dict(tag=tag, U=acc(lad, u), S=acc(lad, ~u), ALL=acc(lad, np.ones_like(u)), REC=float((rl == ytr).mean()), census=cen)


GLUE = os.environ.get("LADDER_GLUE") == "1"
if GLUE:   # (ledger 11-11 GLUE-L) no pair rows: each phrase is its own row on a tree built over single-phrase vectors; glued at read time
    sd = lambda S: [x for s in S for x in s[4]]
    Str, Ste = sd(tr), sd(te); own = [len(s[4]) for s in te]
    pca1 = PCA(64, random_state=0).fit(np.stack([x[1] for x in Str])); Gtr = pca1.transform(np.stack([x[1] for x in Str])); Gte = pca1.transform(np.stack([x[1] for x in Ste]))
    gy = np.concatenate([[s[2]] * len(s[4]) for s in tr]); base = np.log(np.bincount(ytr, minlength=len(OPS)) / len(ytr) + 1e-9)
    def glue_run(G, tag):
        global ytr
        km = MiniBatchKMeans(KS[0], random_state=KSEED, n_init=3, batch_size=4096).fit(G); maps = [np.arange(KS[0])] + [AgglomerativeClustering(k, linkage="ward").fit(km.cluster_centers_).labels_ for k in KS[1:]]
        nodes = lambda X: [m[km.predict(X)] for m in maps]; ntr, nte = nodes(Gtr), nodes(Gte)
        T = [defaultdict(lambda: np.zeros(len(OPS))) for _ in range(5)]
        for i, (k, _, kd, r) in enumerate(Str):
            T[0][(k, kd, r)][gy[i]] += 1; T[4][(kd, r)][gy[i]] += 1
            for l in range(3): T[l + 1][(int(ntr[l][i]), kd, r)][gy[i]] += 1
        def desc(k, kd, r, nd):
            p = np.ones(len(OPS)) / len(OPS)
            for c in [T[4].get((kd, r))] + [T[l + 1].get((int(nd[l]), kd, r)) for l in (2, 1, 0)] + [T[0].get((k, kd, r))]:
                if c is not None and c.sum() > 0: p = (c + KAPPA * p) / (c.sum() + KAPPA)
            return np.log(p + 1e-9)
        def pred(S, steps_, nd):
            out, i = [], 0
            for s in steps_:
                lp = sum(desc(S[i + j][0], S[i + j][2], S[i + j][3], [n[i + j] for n in nd]) for j in range(len(s[4])))
                if len(s[4]) == 2: lp = lp - base
                if len(s[4]) == 0: c = T[4].get((s[1][0], 0)); lp = np.log((c if c is not None else np.ones(len(OPS))) + 1e-9)
                out.append(int(np.argmax(lp))); i += len(s[4])
            return np.array(out)
        p = pred(Ste, te, nte); rp = pred(Str, tr, ntr)
        return dict(tag=tag, U=float((p[u] == yte[u]).mean()), S=float((p[~u] == yte[~u]).mean()), REC=float((rp == ytr).mean()), rows=len(T[0]))
    gR = glue_run(Gtr, "GLUE"); gP = glue_run(Gtr[np.random.default_rng(0).permutation(len(Gtr))], "GLUE-PLACEBO")
HYB = os.environ.get("LADDER_HYB") == "1"
if HYB:   # (ledger 11-11 HYBRID) three tables: nodes (IS-A over single phrases), edges (class x class, level-gated), junctions (volatile)
    HK, HM, PURE = 2, 2, 0.9   # diversity: >= HK distinct leaf pairs from >= HM problems; an edge splits down only if its parent is < PURE
    pid = np.concatenate([[i] * min(len(it["prog"]), K) for i, it in enumerate(D["TRAIN"])]); assert len(pid) == len(tr)
    hS = lambda S: [x for s in S for x in s[4]]
    pcaH = PCA(64, random_state=0).fit(np.stack([x[1] for x in hS(tr)]))
    Htr, Hte = pcaH.transform(np.stack([x[1] for x in hS(tr)])), pcaH.transform(np.stack([x[1] for x in hS(te)]))
    def hyb_run(rules, tag, Hfit, chain="off"):
        km = MiniBatchKMeans(KS[0], random_state=KSEED, n_init=3, batch_size=4096).fit(Hfit)
        maps = [np.arange(KS[0])] + [AgglomerativeClustering(k, linkage="ward").fit(km.cluster_centers_).labels_ for k in KS[1:]]
        def junctions(S, H):   # per step: the volatile junction rows -> (A path, B path) by role; path = [leaf key, L1, L2, L3]
            c = km.predict(H); out, i = [], 0
            for s in S:
                ab = [None, None]
                for j, (k, _, _, r) in enumerate(s[4]): ab[r] = [k] + [int(m[c[i + j]]) for m in maps]
                out.append(ab); i += len(s[4])
            return out
        def ekey(ab, kd, l): return (l, kd) if l == 4 else (l, kd) + tuple(None if x is None else x[l] for x in ab)
        Jtr, Jte = junctions(tr, Hfit), junctions(te, Hte)   # (placebo) TRAIN sides take shuffled vectors
        # (CHAIN, ledger 11-11) a side fed by an earlier step gets that step's RESULT CLASS: a bounded symbol = its op + its inputs'
        # classes (256 at L1, 32 at L2, op alone at L3); a nested result collapses to its op, so the symbol set stays finite
        nst = lambda sp: [min(len(it["prog"]), K) for it in D[sp]]
        cl = lambda y, l: None if y is None else (y[3] if isinstance(y[3], tuple) else y[l])
        refpath = lambda x, op: [("s", op, tuple(None if y is None else y[0] for y in x)), ("s", op, cl(x[0], 2), cl(x[1], 2)), ("s", op, cl(x[0], 3), cl(x[1], 3)), ("s", op)]
        def fill(ab, st, dab, dop):
            if chain == "off": return ab
            ab = [None, None] if st[6] and any(r is not None for r in st[5]) else list(ab)   # a no-text step reads its inputs, not the question
            for r, ref in enumerate(st[5]):
                if ref is not None and ref < len(dab) and ab[r] is None: ab[r] = refpath(dab[ref], dop[ref])
            return ab
        ytr_ref = np.random.default_rng(7).permutation(ytr) if chain == "placebo" else ytr   # (CHAIN-PLACEBO) result classes carry a shuffled op
        if chain != "off":
            out_, g = [], 0
            for n in nst("TRAIN"):
                dab, dop = [], []
                for _ in range(n): ab = fill(Jtr[g], tr[g], dab, dop); out_.append(ab); dab.append(ab); dop.append(int(ytr_ref[g])); g += 1
            Jtr = out_
        cnt = defaultdict(lambda: np.zeros(len(OPS))); lp_ = defaultdict(set); pr_ = defaultdict(set)
        levels = (4, 3, 2, 1) if rules else (4, 3, 2, 1, 0)
        for ab, s, q in zip(Jtr, tr, pid):
            for l in levels:
                e = ekey(ab, s[1], l); cnt[e][s[2]] += 1; lp_[e].add(tuple(None if x is None else x[0] for x in ab)); pr_[e].add(int(q))
        if rules:   # admission top-down: the top always; below, diversity AND a split (impure) admitted parent
            adm = {e for e in cnt if e[0] == 4}
            for l in (3, 2, 1):
                for ab, s in zip(Jtr, tr):
                    e = ekey(ab, s[1], l)
                    if e in adm: continue
                    par = ekey(ab, s[1], l + 1); c = cnt[par]
                    if len(lp_[e]) >= HK and len(pr_[e]) >= HM and par in adm and (l == 3 or c.max() / c.sum() < PURE): adm.add(e)
        else: adm = set(cnt)
        def dist(ab, kd):
            p = np.ones(len(OPS)) / len(OPS)
            for l in levels:
                e = ekey(ab, kd, l)
                if e in adm: c = cnt[e]; p = (c + KAPPA * p) / (c.sum() + KAPPA)
            return p
        desc = lambda ab, kd: int(dist(ab, kd).argmax())
        def pred(J, S, sp="TEST", mode="pred"):   # sequential per problem: mode pred = the parse feeds its OWN ops forward; tf = gold ops
            out, g = [], 0; yg = yte if sp == "TEST" else ytr
            for n in nst(sp):
                dab, dop = [], []
                for _ in range(n):
                    ab = fill(J[g], S[g], dab, dop); q = desc(ab, S[g][1]); out.append(q); dab.append(ab); dop.append(q if mode == "pred" else int(yg[g])); g += 1
            return np.array(out)
        BEAM = int(os.environ.get("LADDER_BEAM", "0")); MOTIF = os.environ.get("LADDER_MOTIF", "")   # "" | on | placebo
        LAM = float(os.environ.get("LADDER_MOTIF_LAM", "1.0")); MPROB = 3
        # (MOTIF, ledger 11-12) the motif table: recurring parse FRAGMENTS over classes, mined from TRAIN junctions (gold ops = key-verified).
        # key = (ops of the previous two steps, the link signature: how far back each side's feeding step is (cap 3) + no-text flag,
        # kinds); backoff M2 -> M1 (one previous op) -> M0 (shape only); a row needs >= MPROB distinct problems (no leaf ever enters)
        sig = lambda st, j: tuple(None if r is None else min(j - r, 3) for r in st[5]) + (st[6],)
        mkeys = lambda dop, st, j: [(0, sig(st, j), st[1]), (1, tuple(dop[-1:]), sig(st, j), st[1]), (2, tuple(dop[-2:]), sig(st, j), st[1])]
        mcnt = defaultdict(lambda: np.zeros(len(OPS))); mpr = defaultdict(set)
        if MOTIF in ("on", "placebo"):
            ym = np.random.default_rng(11).permutation(ytr) if MOTIF == "placebo" else ytr   # (MOTIF-PLACEBO) fragments point at shuffled ops
            g = 0
            for q, n in enumerate(nst("TRAIN")):
                for j in range(n):
                    for mk in mkeys([int(x) for x in ytr[g - j:g]], tr[g], j): mcnt[mk][ym[g]] += 1; mpr[mk].add(q)
                    g += 1
        lbase = np.log(np.bincount(ytr, minlength=len(OPS)) / len(ytr) + 1e-9)
        # (CMOTIF, ledger 11-12) class-keyed motifs: a fragment = producer's input classes -> (its result) -> meets the consumer's other
        # side; NO op in the key (nested results collapse to "s"), so a wrong guess of the producer's op cannot change the row read
        isref = lambda x: x is not None and isinstance(x[0], tuple)
        strip = lambda c: "s" if isinstance(c, tuple) else c
        def ckeys(ab, kd):
            out = []
            for r in (0, 1):
                if not isref(ab[r]): continue
                o = ab[1 - r]; lf = (ab[r][0][2], None if o is None else o[0])
                for pl, ol in ((2, 3), (1, 2)):   # coarse: producer inputs at 32 classes (refpath[2]) + other side at 32; fine: 256 / 256
                    oc = None if o is None else ("s" if isref(o) else o[ol])
                    out.append(((ol, kd, r) + tuple(strip(c) for c in ab[r][pl][2:]) + (oc,), lf))
            return out
        ccnt = defaultdict(lambda: np.zeros(len(OPS))); cpr = defaultdict(set); clf = defaultdict(set)
        if MOTIF in ("class", "classplacebo"):
            yc = np.random.default_rng(13).permutation(ytr) if MOTIF == "classplacebo" else ytr
            for g, (ab, st, q) in enumerate(zip(Jtr, tr, pid)):
                for mk, lf in ckeys(ab, st[1]): ccnt[mk][yc[g]] += 1; cpr[mk].add(int(q)); clf[mk].add(lf)
        def cscore(ab, kd):   # sum over linked sides of (log p_motif - log base), each a 32 -> 256 shrinkage descent over admitted rows
            tot, ks = 0.0, ckeys(ab, kd)
            for i in range(0, len(ks), 2):
                p = np.ones(len(OPS)) / len(OPS)
                for mk, _ in ks[i:i + 2]:
                    if len(cpr.get(mk, ())) >= MPROB and len(clf[mk]) >= 2: c = ccnt[mk]; p = (c + KAPPA * p) / (c.sum() + KAPPA)
                tot = tot + np.log(p + 1e-12) - lbase
            return tot
        def mdist(dop, st, j):
            p = np.ones(len(OPS)) / len(OPS)
            for mk in mkeys(dop, st, j):
                if len(mpr.get(mk, ())) >= MPROB: c = mcnt[mk]; p = (c + KAPPA * p) / (c.sum() + KAPPA)
            return p
        def pred_beam(J, S, B):   # (BEAM, ledger 11-12) top-B whole parses per problem, scored by joint log-probability; never commit early
            out, g = [], 0
            for n in nst("TEST"):
                hyps = [(0.0, [], [])]
                for _ in range(n):
                    nh = []
                    for lp, dab, dop in hyps:
                        ab = fill(J[g], S[g], dab, dop); sc = np.log(dist(ab, S[g][1]) + 1e-12)
                        if MOTIF in ("on", "placebo"): sc = sc + LAM * (np.log(mdist(dop, S[g], len(dop)) + 1e-12) - lbase)   # product of experts: edge x motif / base
                        elif MOTIF: sc = sc + LAM * cscore(ab, S[g][1])
                        for o in np.argsort(-sc)[:B]: nh.append((lp + float(sc[o]), dab + [ab], dop + [int(o)]))
                    hyps = sorted(nh, key=lambda h: -h[0])[:B]; g += 1
                out.extend(hyps[0][2])
            return np.array(out)
        pe = pred_beam(Jte, te, BEAM) if BEAM and chain != "off" else pred(Jte, te)
        pr = pred(Jtr, tr, "TRAIN", "tf"); ptf = pred(Jte, te, "TEST", "tf")
        if os.environ.get("LADDER_DUMPPRED"): np.savez(os.environ["LADDER_DUMPPRED"] + f"_{tag}.npz", pe=pe, ptf=ptf, y=yte, u=u, Jte=np.array([repr(x) for x in Jte]))
        per = Counter(e[0] for e in adm); rf = np.array([any(r is not None for r in st[5]) for st in te])
        return dict(tag=tag, U=float((pe[u] == yte[u]).mean()), S=float((pe[~u] == yte[~u]).mean()), REC=float((pr == ytr).mean()),
                    TF=float((ptf[u] == yte[u]).mean()), UREF=float((pe[u & rf] == yte[u & rf]).mean()), nref=int((u & rf).sum()),
                    edges={l: per[l] for l in sorted(per)})
    hR = hyb_run(True, "HYBRID", Htr)
    if os.environ.get("LADDER_MOTIF"):   # (MOTIF) arms: BEAM (motif off) is hR's role; CHAIN = BEAM + motifs; CHAIN-PLACEBO = BEAM + shuffled motifs
        CM = os.environ["LADDER_MOTIF"] == "class"
        os.environ["LADDER_MOTIF"] = ""; hR = hyb_run(True, "BEAM4", Htr, "pred")
        os.environ["LADDER_MOTIF"] = "class" if CM else "on"
        hCH = hyb_run(True, "BEAM4+CMOTIF" if CM else "BEAM4+MOTIF", Htr, "pred")
        os.environ["LADDER_MOTIF"] = "classplacebo" if CM else "placebo"; hCP = hyb_run(True, "MOTIF-PLACEBO", Htr, "pred"); os.environ["LADDER_MOTIF"] = "class" if CM else "on"
    elif os.environ.get("LADDER_CHAIN") == "1": hCH = hyb_run(True, "CHAIN" + (f"-BEAM{os.environ['LADDER_BEAM']}" if os.environ.get("LADDER_BEAM") else ""), Htr, "pred"); hCP = hyb_run(True, "CHAIN-PLACEBO", Htr, "placebo")
    else: hC = hyb_run(False, "CROSS", Htr); hP = hyb_run(True, "HYB-PLACEBO", Htr[np.random.default_rng(0).permutation(len(Htr))])
OUTF = ".cache/gsm_ladder" + ("" if SRC == "full" else "_" + SRC) + ("" if PARENT == "geo" else "_" + PARENT) + (f"_gate{GATE}" if GATE else "") + ("_dial" if DIAL else "") + ("_novel" if NOVEL else "") + ("" if SPANS == "impulse" else "_" + SPANS) + ("_" + os.path.basename(LOCALF)[5:-4] if "LADDER_LOCALF" in os.environ else "") + (f"_k{KSEED}" if KSEED else "") + ".txt"
R = run(Ztr, "LADDER"); P = run(Ztr[np.random.default_rng(0).permutation(len(Ztr))], "TREE-PLACEBO")
names = ["L0 leaf", "L1 2048", "L2 256", "L3 32", "L4 kind"]
_sl = np.mean([np.bincount(span_ids(it)).mean() for it in D["TEST"]])
out = [f"mean span {_sl:.2f} words | LADDER [states {SRC}, parents {PARENT}, gate {GATE}, dial {DIAL}, novel {NOVEL}, spans {SPANS}, localf {LOCALF}, kseed {KSEED}]: {len(tr)} TRAIN step entries, {len({s[0] for s in tr})} distinct leaves, levels {KS}, kappa {KAPPA}"]
for r in (R, P):
    out.append(f"  {r['tag']:12s}: TEST-U {r['U']:.4f} | seen {r['S']:.4f} | all {r['ALL']:.4f} | RECALL {r['REC']:.4f}")
    out.append("    census (each level alone; TEST-U / seen): " + " | ".join(f"{n} {a:.4f}/{b:.4f}" for n, (a, b) in zip(names, r["census"])))
bu = int(np.argmax([c[0] for c in R["census"]])); bs = int(np.argmax([c[1] for c in R["census"]]))
out.append(f"  window +-3 (STEPMEM2, banked): TEST-U 0.5047")
out.append(f"LADDER-x (TEST-U - 0.5047 >= +0.020): {R['U'] - WINDOW:+.4f} -> {'PASS' if R['U'] - WINDOW >= 0.020 else 'FAIL'} | "
           f"TREE-PLACEBO (ladder - placebo >= +0.020): {R['U'] - P['U']:+.4f} -> {'PASS' if R['U'] - P['U'] >= 0.020 else 'FAIL'} | "
           f"CROSSOVER (best seen = L0 {names[bs]}, best TEST-U coarser {names[bu]}): {'PASS' if bs == 0 and bu > 0 else 'FAIL'}")
if GLUE:
    for g in (gR, gP): out.append(f"  {g['tag']:12s}: TEST-U {g['U']:.4f} | seen {g['S']:.4f} | RECALL {g['REC']:.4f} | phrase rows {g['rows']}")
    out.append(f"GLUE-L (GLUE - JOIN ladder, TEST-U >= +0.020): {gR['U'] - R['U']:+.4f} | GLUE-SEEN (seen >= -0.020): {gR['S'] - R['S']:+.4f} | GLUE-PLACEBO (GLUE - placebo >= +0.020): {gR['U'] - gP['U']:+.4f}")
    OUTF = OUTF[:-4] + "_glue.txt"
if HYB and os.environ.get("LADDER_CHAIN") == "1":
    for g in (hR, hCH, hCP): out.append(f"  {g['tag']:13s}: TEST-U {g['U']:.4f} (gold-fed {g['TF']:.4f}) | seen {g['S']:.4f} | RECALL {g['REC']:.4f} | TEST-U steps fed by a step (n={g['nref']}) {g['UREF']:.4f} | edges {g['edges']}")
    out.append(f"CHAIN-x (CHAIN - HYBRID, self-fed, >= +0.020): {hCH['U'] - hR['U']:+.4f} | CHAIN-PLACEBO (CHAIN - placebo >= +0.020): {hCH['U'] - hCP['U']:+.4f} | "
               f"fed-steps CHAIN - HYBRID {hCH['UREF'] - hR['UREF']:+.4f} | CHAIN - JOIN {hCH['U'] - R['U']:+.4f}")
    OUTF = OUTF[:-4] + "_chain" + (f"_beam{os.environ['LADDER_BEAM']}" if os.environ.get("LADDER_BEAM") else "") + (f"_{'c' if os.environ.get('LADDER_MOTIF') == 'class' else ''}motif{os.environ.get('LADDER_MOTIF_LAM', '1.0')}" if os.environ.get("LADDER_MOTIF") else "") + ".txt"
elif HYB:
    for g in (hR, hC, hP): out.append(f"  {g['tag']:12s}: TEST-U {g['U']:.4f} | seen {g['S']:.4f} | RECALL {g['REC']:.4f} | edges per level {g['edges']}")
    out.append(f"HYB-NOLOSS (HYB - JOIN >= -0.010): {hR['U'] - R['U']:+.4f} | HYB-x (>= +0.020): {hR['U'] - R['U']:+.4f} | "
               f"RULES (HYB - CROSS >= +0.020): {hR['U'] - hC['U']:+.4f} | HYB-PLACEBO (HYB - placebo >= +0.020): {hR['U'] - hP['U']:+.4f} | seen HYB - JOIN {hR['S'] - R['S']:+.4f}")
    OUTF = OUTF[:-4] + "_hyb.txt"
txt = "\n".join(out); print(txt); open(OUTF, "w").write(txt + "\n")

if os.environ.get("LADDER_DIAG") == "1":   # (unbarred attribution, ledger 11-10) is it the states or the ladder? flat 1-NN on the same vectors
    Zn = Ztr / np.linalg.norm(Ztr, axis=1, keepdims=True); Qn = Zte / np.linalg.norm(Zte, axis=1, keepdims=True)
    byk = defaultdict(list)
    for i, kd in enumerate(kd_tr): byk[kd].append(i)
    pr = np.array([ytr[byk[kd]][int((Zn[byk[kd]] @ q).argmax())] if kd in byk else -1 for q, kd in zip(Qn, kd_te)])
    d = f"DIAG flat 1-NN on the same state vectors (no ladder): TEST-U {float((pr[u] == yte[u]).mean()):.4f} | seen {float((pr[~u] == yte[~u]).mean()):.4f}"
    print(d); open(OUTF, "a").write(d + "\n")
