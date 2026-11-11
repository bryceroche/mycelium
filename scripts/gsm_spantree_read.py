"""gsm_spantree_read.py -- THE SPAN TREE READ (ledger 2026-11-11 pin; Bryce: "the leaf node gets the highly specific MRI tagged span but
the parent node gets a genericized version of the span ... aggressively memorize spans on the Tree DB but dont memorize the problem
specific combinations how spans are assembled -- the SQL cross join").
Zero GPU, zero training. Leaves = exact operand phrases cut by the MRI spin detector (the operand's span + the next span, numbers and
names genericized). Parents = the SAME phrase genericized word by word: each word replaced by its word class, classes from BIRCH
branching over Welford means of the reader's states per word type (TRAIN only), Ward-nested 1024 > 256 > 64. "fresh duck egg" and
"rotten chicken egg" share a parent wherever fresh~rotten and duck~chicken share a class.
Two stores on the same tree (the only difference between the arms):
  GLUE  -- one row per PHRASE (role a/b, own operand kind). No pair is stored; a step reads A's phrase and B's phrase separately and
           glues them on the fly (product of experts, minus one copy of the role prior).
  JOIN  -- one row per PAIR (A phrase x B phrase, kind pair): the banked ladder's store, the cross join.
TREE-PLACEBO = word vectors shuffled across word types before the classes are built.  -> .cache/gsm_spantree{_SPANS}{_kN}.txt"""
import json, os, sys
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_bench_data as GD
import gsm_cj_probe_skel as SK
import gsm_trunk_reader as TR
import gsm_trunk_states as GTS

K = 12; PUN = set(".,;:?!'\"()’"); KAPPA = 0.5; CS = (1024, 256, 64); KSEED = int(os.environ.get("SPANTREE_SEED", "0"))
SPANS = os.environ.get("SPANTREE_SPANS", "mri")   # mri | impulse
D, ST = TR.load(); seen = SK.seen(D["TRAIN"])
OPS = sorted({o for it in D["TRAIN"] for o, _, _ in it["prog"]}); OI = {o: i for i, o in enumerate(OPS)}

_, Q = GTS.texts(); IMP = np.load(".cache/gsmb_impulse_rb8.npy"); MRI = np.load(".cache/gsmb_mri_rb8.npy"); assert len(MRI) == len(IMP)
_tx = json.load(open(".cache/gsmb_impulse_rb8.npy.json"))["texts"]
_st = np.cumsum([0] + [len(GD.tokenize(q, set())[0]) for q in _tx]); _pos = {q: int(_st[i]) for i, q in enumerate(_tx)}
for sp in ("TRAIN", "TEST"):
    for it, q in zip(D[sp], Q[sp]): it["imp0"] = _pos[q]


def words(it): return ["<num>" if v is not None else ("<name>" if n else w) for w, v, n in zip(it["toks"], it["vals"], it["isn"])]
def cpl(it, i): s0 = it["imp0"]; return (IMP[s0 + i, 0] + IMP[s0 + i + 1, 1]) / 2
def mcp(it, i): return MRI[it["imp0"] + i]
def cut(f, it, i, n): c = f(it, i); return (i == 0 or c < f(it, i - 1)) and (i + 2 >= n or c < f(it, i + 1))


def span_ids(it):   # identical rule to gsm_ladder_read.py (LADDER_SPANS)
    w = words(it); sid = it["sid"]; out, cur = [], 0
    for i in range(len(w)):
        out.append(cur)
        if i + 1 >= len(w) or w[i] in PUN or w[i + 1] in PUN or sid[i] != sid[i + 1]: cur += 1; continue
        if cut(mcp if SPANS == "mri" else cpl, it, i, len(w)): cur += 1
    return out


def steps(it):
    """per gold step: the A phrase and the B phrase as word tuples (None = no text operand), kinds, op; 'q:' steps carry the question"""
    w = words(it); sid = it["sid"]; sp = span_ids(it); last = max(sid); res = []
    for o, A, B in it["prog"][:K]:
        if A["t"] or B["t"]:
            ph = [tuple(w[j] for i in S["t"][:1] for j in range(len(w)) if sp[j] in (sp[i], sp[i] + 1)) or None for S in (A, B)]
        else:
            ph = [("q:",) + tuple(w[j] for j in range(len(w)) if sid[j] == last), None]
        res.append((ph, (SK.first_kind(A), SK.first_kind(B)), OI[o]))
    return res


tr = [s for it in D["TRAIN"] for s in steps(it)]; te = [s for it in D["TEST"] for s in steps(it)]
u = np.concatenate([[SK.skeleton(it) not in seen] * min(len(it["prog"]), K) for it in D["TEST"]]).astype(bool); assert len(u) == len(te)
ytr = np.array([s[2] for s in tr]); yte = np.array([s[2] for s in te])

# --- word types: Welford mean of the reader's state over every TRAIN occurrence (TEST words get their own TEST mean, classed by the
#     TRAIN tree; no labels touch the classes) ---
def welford(split):
    n, mu = defaultdict(int), {}
    for it in D[split]:
        for j, wd in enumerate(words(it)):
            x = np.asarray(ST[it["st0"] + j], np.float64); n[wd] += 1
            mu[wd] = x.copy() if n[wd] == 1 else mu[wd] + (x - mu[wd]) / n[wd]
    return n, mu
ntr, mtr = welford("TRAIN"); nte, mte = welford("TEST")
VOC = sorted(mtr); Vte = sorted(set(mte) - set(mtr))
pca = PCA(64, random_state=0).fit(np.stack([mtr[w] for w in VOC]))
def emb(M, ws): z = pca.transform(np.stack([M[w] for w in ws])); return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
Ztr = emb(mtr, VOC); Zte = emb(mte, Vte) if Vte else np.zeros((0, 64))


def classes(Z):
    """BIRCH branching (a new subcluster when a word falls outside every leaf radius) -> Ward-nested 1024 > 256 > 64"""
    th = 0.30
    while True:
        b = Birch(threshold=th, n_clusters=None, branching_factor=50).fit(Z)
        if len(b.subcluster_centers_) >= CS[0] * 1.5: break
        th *= 0.85
    sc = b.subcluster_centers_; lab = b.predict(Z)
    a1 = AgglomerativeClustering(CS[0], linkage="ward").fit(sc).labels_
    c1 = np.stack([sc[a1 == k].mean(0) for k in range(CS[0])])
    maps = [np.arange(CS[0])] + [AgglomerativeClustering(k, linkage="ward").fit(c1).labels_ for k in CS[1:]]
    def cls(ws_z): s = b.predict(ws_z); return [m[a1[s]] for m in maps]
    return cls, len(sc), th


def run(Z, tag):
    order = np.random.default_rng(KSEED).permutation(len(VOC))   # BIRCH is order-dependent: the seed is the insertion order
    cls_o, nsub, th = classes(Z[order]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
    ctr = cls_o(Z); cte = cls_o(Zte) if len(Vte) else [np.zeros(0, int)] * 3
    C = [dict(zip(VOC, ctr[l])) | dict(zip(Vte, cte[l])) for l in range(3)]
    gen = lambda ph, l: None if ph is None else tuple(w if w in ("q:", "<num>", "<name>") or w in PUN else C[l][w] for w in ph)
    keys = lambda ph: [ph] + [gen(ph, l) for l in range(3)]   # L0 exact .. L3 most generic
    out = {}
    for store in ("GLUE", "JOIN"):
        T = [defaultdict(lambda: np.zeros(len(OPS))) for _ in range(5)]
        def rows(s):
            ph, kd, _ = s
            if store == "JOIN": return [(tuple(zip(*[keys(p) for p in ph])), kd, "ab")]
            return [(keys(p), (kd[r], r), "ab"[r]) for r, p in enumerate(ph) if p is not None] or [([None] * 4, (kd[0], 0), "a")]
        for s in tr:
            for ks, kd, _ in rows(s):
                for l in range(4): T[l][(ks[l], kd)][s[2]] += 1
                T[4][kd][s[2]] += 1
        def desc(ks, kd, lvl):
            p = np.ones(len(OPS)) / len(OPS)
            for l in (4, 3, 2, 1, 0):
                c = T[l].get(kd) if l == 4 else T[l].get((ks[l], kd))
                if c is not None and c.sum() > 0:
                    p = (c + KAPPA * p) / (c.sum() + KAPPA)
                    if lvl is not None: lvl[l].append(int(c.argmax()))
                elif lvl is not None: lvl[l].append(-1)
            return p
        def predict(S, census):
            pr, lvl = [], [[] for _ in range(5)]
            for s in S:
                R = rows(s); lp = 0
                for i, (ks, kd, _) in enumerate(R):
                    lp = lp + np.log(desc(ks, kd, lvl if i == 0 else None) + 1e-9)
                if len(R) == 2:   # product of experts: remove one copy of the shared prior (the op base rate)
                    lp = lp - np.log(np.bincount(ytr, minlength=len(OPS)) / len(ytr) + 1e-9)
                pr.append(int(np.argmax(lp)))
            return np.array(pr), [np.array(x) for x in lvl]
        p, lvl = predict(te, True); rp, _ = predict(tr, False)
        acc = lambda x, m: float((x[m] == yte[m]).mean())
        out[store] = dict(U=acc(p, u), S=acc(p, ~u), REC=float((rp == ytr).mean()), census=[(acc(x, u), acc(x, ~u)) for x in lvl],
                          leaves=len(T[0]), nodes=[len(t) for t in T[:4]])
    out["nsub"], out["th"] = nsub, th
    return out


R = run(Ztr, "SPANTREE"); P = run(Ztr[np.random.default_rng(1).permutation(len(Ztr))], "TREE-PLACEBO")
names = ["L0 exact", "L1 c1024", "L2 c256", "L3 c64", "L4 kind"]
_sl = np.mean([np.bincount(span_ids(it)).mean() for it in D["TEST"]])
o = [f"SPANTREE [spans {SPANS}, mean span {_sl:.2f} words, kseed {KSEED}]: {len(VOC)} TRAIN word types ({len(Vte)} TEST-only), "
     f"BIRCH {R['nsub']} subclusters (threshold {R['th']:.3f}) -> classes {CS}, kappa {KAPPA}; {len(tr)} TRAIN / {len(te)} TEST steps (TEST-U {u.sum()})"]
for tag, r in (("SPANTREE", R), ("PLACEBO", P)):
    for st in ("GLUE", "JOIN"):
        x = r[st]
        o.append(f"  {tag:8s} {st}: TEST-U {x['U']:.4f} | seen {x['S']:.4f} | RECALL {x['REC']:.4f} | rows per level {x['nodes']}")
        o.append("    census (each level alone, first operand; TEST-U / seen): " + " | ".join(f"{n} {a:.4f}/{b:.4f}" for n, (a, b) in zip(names, x["census"])))
g, j, pg = R["GLUE"], R["JOIN"], P["GLUE"]
bu = int(np.argmax([c[0] for c in g["census"]])); bs = int(np.argmax([c[1] for c in g["census"]]))
o.append(f"GLUE-x (GLUE - JOIN, TEST-U >= +0.020): {g['U'] - j['U']:+.4f} -> {'PASS' if g['U'] - j['U'] >= 0.020 else 'FAIL'} | "
         f"GLUE-SEEN (GLUE - JOIN, seen >= -0.020): {g['S'] - j['S']:+.4f} -> {'PASS' if g['S'] - j['S'] >= -0.020 else 'FAIL'} | "
         f"SPANTREE-x (GLUE TEST-U - 0.5555 >= +0.020): {g['U'] - 0.5555:+.4f} -> {'PASS' if g['U'] - 0.5555 >= 0.020 else 'FAIL'} | "
         f"TREE-PLACEBO (GLUE - placebo GLUE >= +0.020): {g['U'] - pg['U']:+.4f} -> {'PASS' if g['U'] - pg['U'] >= 0.020 else 'FAIL'} | "
         f"CROSSOVER (best seen {names[bs]}, best TEST-U {names[bu]}): {'PASS' if bs == 0 and bu > 0 else 'FAIL'}")
OUTF = ".cache/gsm_spantree" + ("" if SPANS == "mri" else "_" + SPANS) + (f"_k{KSEED}" if KSEED else "") + ".txt"
txt = "\n".join(o); print(txt); open(OUTF, "w").write(txt + "\n")
if os.environ.get("SPANTREE_DEMO") == "1":   # who shares a parent with whom (the duck-egg check)
    cls_o, _, _ = classes(Ztr); c = cls_o(Ztr)
    for l in range(3):
        grp = defaultdict(list)
        for w, k in zip(VOC, c[l]): grp[k].append(w)
        for probe in ("duck", "eggs", "fresh", "dollars"):
            if probe in mtr: print(f"  L{l + 1} class of {probe!r}: {sorted(grp[c[l][VOC.index(probe)]], key=lambda w: -ntr[w])[:14]}")
