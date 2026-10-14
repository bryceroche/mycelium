"""depthprobe_read.py -- THE DEPTH PROBE's read (zero-GPU; registered ledger 10-13 14:03 / 14:50, decision table 10-14 14:27).
Reads .cache/depthprobe2_{index.npz, native.npy, deployed.npy, bidir.npy} (scripts/picker/depthprobe_collect.py).

Depth naming: Dk = after k Llama layers (D0 = embedding); D4 = THE DEPLOYED TAP ('L0-L3'; the July 'L4'; the decision
table's 'at L3' reads HERE). EV1..EV4 = the every-fourth stack (layers 0,4,8,12 composed). Row split 2/3 train / 1/3
test, by row, seeded; SE = bootstrap over TEST ROWS (B=300); the bar's difference uses the PAIRED bootstrap.

Per regime x tap:
 (a) OWNER PAIRS: numeral pairs in a row, both owners named (non-pronoun); same = same owner surface. Logistic probe on
     [|u-v|, u*v] in a train-fit PCA-128 space; the no-training cosine; AUROC on all test pairs, on SAME-NOUN pairs (the
     twins), distance-matched (mean AUROC within token-distance tertiles), a SHUFFLED-LABEL placebo; the position-only
     control (-|token distance|) printed once.
 (b) WHICH-TWIN-IS-LATER: same-noun numeral pairs; logistic on u-v (order randomized); D0 is the position-free floor
     (the embedding carries no position; RoPE enters in attention).
 (c) THE JULY VALUE MEASURE: digit-exact (3 digits, multinomial logistic on PCA-256) of the numeral's value <= 999 at
     its span; must reproduce the decline D4 > D8 (July: 0.998 / 0.953).
 (d) THE TREE PROBE (Hewitt & Manning) on the FULL rows: a rank-64 map B, ||B(h_i-h_j)||^2 regressed on spaCy tree
     distance within sentences (L1 loss, Adam); test DSpr (per-sentence Spearman), UUAS (MST) vs the adjacent-token
     baseline; LINKS: the numeral's spaCy head is its nearest sentence token (top-1) vs nearest-by-position; the OWNER
     (climb targets) among the row's nominals excluding the numeral's own head (top-1/top-3) vs position.
BARS (pinned 10-13): DEEPER-BINDS = best contiguous tap D5..D16 >= D4 + 0.05 AUROC and >= 2 SE (paired), read on the
NATIVE regime; FLAT = all within +-0.02; ALREADY-WRITTEN needs the sorting room's output (not in this pass).
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--regime", required=True, choices=["native", "deployed", "bidir"])
ap.add_argument("--prefix", default=".cache/depthprobe2")
ap.add_argument("--taps", default="")
ap.add_argument("--out", default="")
args = ap.parse_args()
rng0 = np.random.default_rng(0)
NB = 300

I = np.load(args.prefix + "_index.npz", allow_pickle=True)
TAPS = list(I["taps"])
taps = args.taps.split(",") if args.taps else TAPS
X = np.load(f"{args.prefix}_{args.regime}.npy", mmap_mode="r")
idx_row, idx_pos = I["idx_row"], I["idx_pos"]
loc = {(int(r), int(p)): i for i, (r, p) in enumerate(zip(idx_row, idx_pos))}
n_rows = len(I["pool_idx"]); n_full = int(I["n_full"])
perm = rng0.permutation(n_rows)
test_rows = set(perm[: n_rows // 3].tolist())
num = {k[4:]: I[k] for k in I.files if k.startswith("num_")}
NN = len(num["row"])
is_test_num = np.array([int(r) in test_rows for r in num["row"]])


def numvec(S, i):
    r, s, e = int(num["row"][i]), int(num["num_s"][i]), int(num["num_e"][i])
    return S[[loc[(r, p)] for p in range(s, e)]].mean(0)


def pca_fit(A, k):
    mu = A.mean(0); sd = A.std(0) + 1e-6
    Z = (A - mu) / sd
    _, _, Vt = np.linalg.svd(Z[rng0.choice(len(Z), min(len(Z), 6000), replace=False)], full_matrices=False)
    W = Vt[:k].T
    return lambda B: ((B - mu) / sd) @ W


def boot_auc(y, s, rows, B=NB, rng=None):
    rng = rng or np.random.default_rng(1)
    ur = np.unique(rows); by = {r: np.where(rows == r)[0] for r in ur}
    out = []
    for _ in range(B):
        ix = np.concatenate([by[r] for r in rng.choice(ur, len(ur))])
        if len(np.unique(y[ix])) == 2:
            out.append(roc_auc_score(y[ix], s[ix]))
    return np.array(out)


# ------------------------------------------------------------------ pair tables (tap-independent)
def ok_owner(i):
    t = num["owner_text"][i]
    return num["owner"][i] >= 0 and not num["owner_pron"][i] and t.isalpha() and t not in ("each", "every", "what", "which", "who", "that", "this", "it")
pairs = []          # (i, j, same_owner, same_noun, tokdist, row)
twins = []          # (i, j, row) same noun
by_row = {}
for i in range(NN): by_row.setdefault(int(num["row"][i]), []).append(i)
for r, ii in by_row.items():
    for a in range(len(ii)):
        for b in range(a + 1, len(ii)):
            i, j = ii[a], ii[b]
            same_noun = num["noun_text"][i] != "" and num["noun_text"][i] == num["noun_text"][j]
            if same_noun: twins.append((i, j, r))
            if ok_owner(i) and ok_owner(j):
                pairs.append((i, j, int(num["owner_text"][i] == num["owner_text"][j]), int(same_noun),
                              abs(int(num["num_s"][i]) - int(num["num_s"][j])), r))
P = np.array(pairs); TW = np.array(twins)
p_test = np.array([r in test_rows for r in P[:, 5]]); tw_test = np.array([r in test_rows for r in TW[:, 2]])
y_p = P[:, 2]
tert = np.quantile(P[:, 4], [1 / 3, 2 / 3]); dbin = np.digitize(P[:, 4], tert)
print(f"[read:{args.regime}] rows {n_rows} (test {len(test_rows)}), numerals {NN}, owner pairs {len(P)} "
      f"(same {y_p.mean():.2f}; test {p_test.sum()}; same-noun test {(P[p_test,3]==1).sum()}), twins {len(TW)}", flush=True)
pos_auc = roc_auc_score(y_p[p_test], -P[p_test, 4])
print(f"[read] position-only control (owner pairs, -|tokdist|): AUROC {pos_auc:.3f}", flush=True)

# tree-probe tables (FULL rows; spaCy at read; tap-independent)
import spacy
nlp = spacy.load("en_core_web_sm")
tree = []   # per full row: dict(pieces, sents=[(piece_idx_list, dist_matrix, toks)], nominal pieces, heads)
for r in range(n_full):
    text = I["texts"][r]; off = I["row_off"][r]
    doc = nlp(text)
    def piece_of(ch):
        for k, (a, b) in enumerate(off):
            if a <= ch < b and not (a == 0 and b == 0): return k
        return -1
    tokp = {t.i: piece_of(t.idx) for t in doc}
    sents = []
    for s in doc.sents:
        ts = [t for t in s if not t.is_punct and not t.is_space and tokp[t.i] >= 0]
        if len(ts) < 3: continue
        ids = {t.i: k for k, t in enumerate(ts)}
        adj = {t.i: set() for t in ts}
        for t in s:
            if t.i in adj:
                h = t.head
                while h.i not in adj and h.head.i != h.i: h = h.head
                if h.i in adj and h.i != t.i: adj[t.i].add(h.i); adj[h.i].add(t.i)
        D = np.full((len(ts), len(ts)), 99, np.int32)
        for t in ts:
            D[ids[t.i], ids[t.i]] = 0; fr = [t.i]; d = 0; seen = {t.i}
            while fr:
                d += 1; nx = []
                for u in fr:
                    for v in adj[u]:
                        if v not in seen: seen.add(v); D[ids[t.i], ids[v]] = d; nx.append(v)
                fr = nx
        heads = {}
        for t in ts:
            if t.like_num and t.head.i in ids and t.head.i != t.i: heads[ids[t.i]] = ids[t.head.i]
        sents.append(dict(p=[tokp[t.i] for t in ts], D=D, heads=heads))
    nominal = sorted({tokp[t.i] for t in doc if t.pos_ in ("NOUN", "PROPN", "PRON") and tokp[t.i] >= 0})
    tree.append(dict(sents=sents, nominal=nominal))
print(f"[read] tree rows {n_full}, sentences {sum(len(t['sents']) for t in tree)}", flush=True)
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import spearmanr


def tree_probe(S):
    tr = [r for r in range(n_full) if r not in test_rows]; te = [r for r in range(n_full) if r in test_rows]
    A = np.stack([S[loc[(r, p)]] for r in tr for s in tree[r]["sents"] for p in s["p"]])
    f = pca_fit(A, 512)
    def H(r, ps): return f(np.stack([S[loc[(r, p)]] for p in ps]))
    # training pairs
    di, dj, dd = [], [], []
    feats = []; base = 0
    for r in tr:
        for s in tree[r]["sents"]:
            h = H(r, s["p"]); feats.append(h); n = len(s["p"])
            a, b = np.triu_indices(n, 1)
            di.append(a + base); dj.append(b + base); dd.append(s["D"][a, b]); base += n
    F = np.concatenate(feats); di = np.concatenate(di); dj = np.concatenate(dj); dd = np.concatenate(dd).astype(np.float32)
    rng = np.random.default_rng(2)
    Bm = rng.normal(0, 0.05, (512, 64)).astype(np.float32); m = np.zeros_like(Bm); v = np.zeros_like(Bm)
    for step in range(1, 3001):
        ix = rng.integers(0, len(dd), 4096)
        Dl = (F[di[ix]] - F[dj[ix]]).astype(np.float32); Z = Dl @ Bm
        sq = (Z ** 2).sum(1); g = np.sign(sq - dd[ix])
        grad = 2 * Dl.T @ (g[:, None] * Z) / len(ix)
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
        Bm -= 1e-3 * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
    def proj(r, ps): return H(r, ps) @ Bm
    sp, uu, ub, hk, hb, hn = [], [], [], 0, 0, 0
    o1 = o3 = ob1 = ob3 = on = 0
    for r in te:
        for s in tree[r]["sents"]:
            Z = proj(r, s["p"]); n = len(Z)
            Dp = ((Z[:, None] - Z[None]) ** 2).sum(-1)
            if 5 <= n <= 50:
                a, b = np.triu_indices(n, 1); sp.append(spearmanr(Dp[a, b], s["D"][a, b]).correlation)
            mst = minimum_spanning_tree(Dp + 1e-9).toarray(); pe = {tuple(sorted(e)) for e in zip(*np.nonzero(mst))}
            ge = {tuple(sorted((x, y))) for x, y in zip(*np.nonzero(s["D"] == 1))}
            adjb = {(k, k + 1) for k in range(n - 1)}
            uu.append(len(pe & ge) / max(1, n - 1)); ub.append(len(adjb & ge) / max(1, n - 1))
            for t, hd in s["heads"].items():
                d = Dp[t].copy(); d[t] = np.inf; hk += int(np.argmin(d) == hd)
                pd = np.abs(np.arange(n) - t).astype(float); pd[t] = np.inf; hb += int(np.argmin(pd) == hd); hn += 1
        # owner among nominals (row level)
        rows_n = [i for i in np.where(num["row"] == r)[0] if num["owner"][i] >= 0]
        for i in rows_n:
            ownp = int(num["owner"][i]); nounp = int(num["noun"][i]); s0 = int(num["num_s"][i])
            cands = [p for p in tree[r]["nominal"] if p != nounp and not (s0 <= p < int(num["num_e"][i]))]
            if ownp not in cands or len(cands) < 2: continue
            z0 = proj(r, list(range(s0, int(num["num_e"][i])))).mean(0); Zc = proj(r, cands)
            rk = np.argsort(((Zc - z0) ** 2).sum(1)); rb = np.argsort(np.abs(np.array(cands) - s0))
            j = cands.index(ownp); o1 += int(rk[0] == j); o3 += int(j in rk[:3]); ob1 += int(rb[0] == j); ob3 += int(j in rb[:3]); on += 1
    return dict(dspr=float(np.nanmean(sp)), uuas=float(np.mean(uu)), uuas_adj=float(np.mean(ub)),
                head1=hk / max(1, hn), head1_pos=hb / max(1, hn), n_head=hn,
                own1=o1 / max(1, on), own3=o3 / max(1, on), own1_pos=ob1 / max(1, on), own3_pos=ob3 / max(1, on), n_own=on)


res = {}
for tap in taps:
    t0 = time.time()
    S = np.asarray(X[TAPS.index(tap)], dtype=np.float32)
    V = np.stack([numvec(S, i) for i in range(NN)])
    f = pca_fit(V[~is_test_num], 128); Z = f(V)
    out = {}
    # (a) owner pairs
    Fp = np.concatenate([np.abs(Z[P[:, 0]] - Z[P[:, 1]]), Z[P[:, 0]] * Z[P[:, 1]]], 1)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(Fp[~p_test], y_p[~p_test])
    s = clf.decision_function(Fp[p_test]); yt = y_p[p_test]; rt = P[p_test, 5]
    out["own_auc"] = roc_auc_score(yt, s); bs = boot_auc(yt, s, rt); out["own_se"] = float(bs.std())
    out["_own_s"] = s
    sn = P[p_test, 3] == 1
    out["own_auc_samenoun"] = roc_auc_score(yt[sn], s[sn]) if len(np.unique(yt[sn])) == 2 else float("nan")
    db = dbin[p_test]
    out["own_auc_distmatched"] = float(np.mean([roc_auc_score(yt[db == k], s[db == k]) for k in range(3) if len(np.unique(yt[db == k])) == 2]))
    Vc = V - V[~is_test_num].mean(0)
    cs = np.sum(Vc[P[:, 0]] * Vc[P[:, 1]], 1) / np.linalg.norm(Vc[P[:, 0]], axis=1) / np.linalg.norm(Vc[P[:, 1]], axis=1)
    out["own_cos_auc"] = roc_auc_score(yt, cs[p_test])
    ysh = np.random.default_rng(3).permutation(y_p[~p_test])
    out["own_placebo"] = roc_auc_score(yt, LogisticRegression(C=0.1, max_iter=2000).fit(Fp[~p_test], ysh).decision_function(Fp[p_test]))
    # (b) which-later
    flip = np.random.default_rng(4).random(len(TW)) < 0.5
    a_ = np.where(flip, TW[:, 1], TW[:, 0]); b_ = np.where(flip, TW[:, 0], TW[:, 1])
    yl = (num["num_s"][a_].astype(int) > num["num_s"][b_].astype(int)).astype(int)
    Fl = Z[a_] - Z[b_]
    cl = LogisticRegression(C=0.1, max_iter=2000).fit(Fl[~tw_test], yl[~tw_test])
    out["later_auc"] = roc_auc_score(yl[tw_test], cl.decision_function(Fl[tw_test]))
    # (c) value
    vm = num["value"] >= 0
    fv = pca_fit(V[vm & ~is_test_num], 256); Zv = fv(V)
    ok = np.ones(int((vm & is_test_num).sum()), bool)
    for dgt in (100, 10, 1):
        lab = (num["value"] // dgt) % 10
        c = LogisticRegression(C=1.0, max_iter=3000).fit(Zv[vm & ~is_test_num], lab[vm & ~is_test_num])
        ok &= c.predict(Zv[vm & is_test_num]) == lab[vm & is_test_num]
    out["value_exact"] = float(ok.mean()); out["n_value_test"] = int(ok.size)
    # (d) tree probe
    out.update({f"tree_{k}": v for k, v in tree_probe(S).items()})
    res[tap] = out
    print(f"[{args.regime} {tap:>4}] own {out['own_auc']:.3f}±{out['own_se']:.3f} cos {out['own_cos_auc']:.3f} "
          f"samenoun {out['own_auc_samenoun']:.3f} distm {out['own_auc_distmatched']:.3f} plac {out['own_placebo']:.3f} | "
          f"later {out['later_auc']:.3f} | value {out['value_exact']:.3f} | tree dspr {out['tree_dspr']:.3f} "
          f"uuas {out['tree_uuas']:.3f} (adj {out['tree_uuas_adj']:.3f}) head1 {out['tree_head1']:.3f} (pos {out['tree_head1_pos']:.3f}) "
          f"own1/3 {out['tree_own1']:.3f}/{out['tree_own3']:.3f} (pos {out['tree_own1_pos']:.3f}/{out['tree_own3_pos']:.3f}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

# THE BAR (native regime is the decision table's frame; printed for every regime)
if "D4" in res:
    yt = y_p[p_test]; rt = P[p_test, 5]; ur = np.unique(rt); by = {r: np.where(rt == r)[0] for r in ur}
    deep = [t for t in res if t.startswith("D") and int(t[1:]) >= 5]
    if deep:
        best = max(deep, key=lambda t: res[t]["own_auc"])
        rng = np.random.default_rng(5); diffs = []
        for _ in range(NB):
            ix = np.concatenate([by[r] for r in rng.choice(ur, len(ur))])
            if len(np.unique(yt[ix])) == 2:
                diffs.append(roc_auc_score(yt[ix], res[best]["_own_s"][ix]) - roc_auc_score(yt[ix], res["D4"]["_own_s"][ix]))
        d = res[best]["own_auc"] - res["D4"]["own_auc"]; se = float(np.std(diffs))
        allD = [res[t]["own_auc"] for t in res if t.startswith("D") and int(t[1:]) >= 1]
        verdict = ("DEEPER-BINDS" if d >= 0.05 and d >= 2 * se else
                   "FLAT" if max(allD) - min(allD) <= 0.04 and abs(d) <= 0.02 else "NEITHER (between the bars)")
        print(f"[{args.regime}] OWNER-PAIR BAR: best deep tap {best} {res[best]['own_auc']:.3f} vs D4 {res['D4']['own_auc']:.3f}: "
              f"diff {d:+.3f} paired SE {se:.3f} -> {verdict}", flush=True)
        bt = max(deep, key=lambda t: res[t]["tree_dspr"])
        print(f"[{args.regime}] TREE: D4 dspr {res['D4']['tree_dspr']:.3f} head1 {res['D4']['tree_head1']:.3f} "
              f"own1 {res['D4']['tree_own1']:.3f}; best deep {bt} dspr {res[bt]['tree_dspr']:.3f} head1 {res[bt]['tree_head1']:.3f} "
              f"own1 {res[bt]['tree_own1']:.3f}", flush=True)
for t in res: res[t].pop("_own_s", None)
json.dump(dict(regime=args.regime, position_only_auc=pos_auc, results=res),
          open(args.out or f".cache/depthprobe2_read_{args.regime}.json", "w"), indent=1)
