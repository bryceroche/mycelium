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
LOCALF = os.environ.get("LADDER_LOCALF", ".cache/gsmb_local_rb8.npy"); DIAL = os.environ.get("LADDER_DIAL") == "1"; NOVEL = os.environ.get("LADDER_NOVEL") == "1"; LTXT = []; LIDX = {}
if SRC == "local":
    LV = np.load(LOCALF); LIDX = {t: i for i, t in enumerate(json.load(open(LOCALF + ".json"))["texts"])}


def words(it): return ["<num>" if v is not None else ("<name>" if n else w) for w, v, n in zip(it["toks"], it["vals"], it["isn"])]
def cpl(it, i): s0 = it["imp0"]; return (IMP[s0 + i, 0] + IMP[s0 + i + 1, 1]) / 2


def span_ids(it):
    w = words(it); sid = it["sid"]; out, cur = [], 0
    for i in range(len(w)):
        out.append(cur)
        if i + 1 >= len(w) or w[i] in PUN or w[i + 1] in PUN or sid[i] != sid[i + 1]: cur += 1; continue
        c = cpl(it, i)
        if (i == 0 or c < cpl(it, i - 1)) and (i + 2 >= len(w) or c < cpl(it, i + 1)): cur += 1
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
        res.append((key, (SK.first_kind(A), SK.first_kind(B)), OI[o], np.concatenate(v)))
    return res


tr = [s for it in D["TRAIN"] for s in steps(it)]; te = [s for it in D["TEST"] for s in steps(it)]
if os.environ.get("LADDER_DUMP") == "1":   # the local texts for distill_reader.py --local
    LT = sorted(set(LTXT)); json.dump(dict(texts=LT), open(".cache/gsm_ladder_local_texts.json", "w")); print(f"[ladder] {len(LT)} local texts"); sys.exit(0)
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


OUTF = ".cache/gsm_ladder" + ("" if SRC == "full" else "_" + SRC) + ("" if PARENT == "geo" else "_" + PARENT) + (f"_gate{GATE}" if GATE else "") + ("_dial" if DIAL else "") + ("_novel" if NOVEL else "") + ("_" + os.path.basename(LOCALF)[5:-4] if "LADDER_LOCALF" in os.environ else "") + (f"_k{KSEED}" if KSEED else "") + ".txt"
R = run(Ztr, "LADDER"); P = run(Ztr[np.random.default_rng(0).permutation(len(Ztr))], "TREE-PLACEBO")
names = ["L0 leaf", "L1 2048", "L2 256", "L3 32", "L4 kind"]
out = [f"LADDER [states {SRC}, parents {PARENT}, gate {GATE}, dial {DIAL}, novel {NOVEL}, localf {LOCALF}, kseed {KSEED}]: {len(tr)} TRAIN step entries, {len({s[0] for s in tr})} distinct leaves, levels {KS}, kappa {KAPPA}"]
for r in (R, P):
    out.append(f"  {r['tag']:12s}: TEST-U {r['U']:.4f} | seen {r['S']:.4f} | all {r['ALL']:.4f} | RECALL {r['REC']:.4f}")
    out.append("    census (each level alone; TEST-U / seen): " + " | ".join(f"{n} {a:.4f}/{b:.4f}" for n, (a, b) in zip(names, r["census"])))
bu = int(np.argmax([c[0] for c in R["census"]])); bs = int(np.argmax([c[1] for c in R["census"]]))
out.append(f"  window +-3 (STEPMEM2, banked): TEST-U 0.5047")
out.append(f"LADDER-x (TEST-U - 0.5047 >= +0.020): {R['U'] - WINDOW:+.4f} -> {'PASS' if R['U'] - WINDOW >= 0.020 else 'FAIL'} | "
           f"TREE-PLACEBO (ladder - placebo >= +0.020): {R['U'] - P['U']:+.4f} -> {'PASS' if R['U'] - P['U'] >= 0.020 else 'FAIL'} | "
           f"CROSSOVER (best seen = L0 {names[bs]}, best TEST-U coarser {names[bu]}): {'PASS' if bs == 0 and bu > 0 else 'FAIL'}")
txt = "\n".join(out); print(txt); open(OUTF, "w").write(txt + "\n")

if os.environ.get("LADDER_DIAG") == "1":   # (unbarred attribution, ledger 11-10) is it the states or the ladder? flat 1-NN on the same vectors
    Zn = Ztr / np.linalg.norm(Ztr, axis=1, keepdims=True); Qn = Zte / np.linalg.norm(Zte, axis=1, keepdims=True)
    byk = defaultdict(list)
    for i, kd in enumerate(kd_tr): byk[kd].append(i)
    pr = np.array([ytr[byk[kd]][int((Zn[byk[kd]] @ q).argmax())] if kd in byk else -1 for q, kd in zip(Qn, kd_te)])
    d = f"DIAG flat 1-NN on the same state vectors (no ladder): TEST-U {float((pr[u] == yte[u]).mean()):.4f} | seen {float((pr[~u] == yte[~u]).mean()):.4f}"
    print(d); open(OUTF, "a").write(d + "\n")
