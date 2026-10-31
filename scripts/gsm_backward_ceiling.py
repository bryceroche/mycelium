"""gsm_backward_ceiling.py -- B-PLAN (ledger 2026-10-30, Bryce: "work backwards from the centroids to the tokens"). Zero-GPU.
Top-down (goal-first) planning priced in two halves on TEST of .cache/gsmb_data.json:
  P1 SKELETON: the gold program's expression tree from the final step (numeral leaves 'n', constants 'c<v>', + and * children
     sorted; shared steps duplicated), predicted by TF-IDF kNN over TRAIN (k 25, similarity-weighted vote) on the QUESTION
     SENTENCE alone; PLACEBO = the same kNN on the FIRST sentence; reads: whole text, the frequency prior, the root op.
  P2 LEAF FILL: given the gold skeleton, injective fills of the problem's numerals into its 'n' leaves -> distinct final values
     (the search width) and gold reachability.
Writes .cache/gsm_backward_ceiling.txt."""
import collections, itertools, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD

K, CAP = 25, 200_000


def first(S):
    return ("t", S["t"][0]) if S["t"] else (("s", S["s"][0]) if S["s"] else ("c", S["c"][0]))


def skeleton(prog, r=None):
    r = len(prog) - 1 if r is None else r; op, A, B = prog[r]
    def leaf(o):
        k, i = o
        return "n" if k == "t" else (f"c{GD.CONSTS[i]:g}" if k == "c" else skeleton(prog, i))
    ch = [leaf(first(A)), leaf(first(B))]
    if op in "+*": ch.sort()
    return f"{op}({ch[0]},{ch[1]})"


def leaves(prog, r=None):
    """-> numeral token ids in the skeleton's leaf order (matching skeleton()'s traversal, unsorted)"""
    r = len(prog) - 1 if r is None else r; op, A, B = prog[r]; out = []
    for S in (A, B):
        k, i = first(S)
        if k == "t": out.append(i)
        elif k == "s": out += leaves(prog, i)
    return out


def run(prog, fill, r=None, it=None):
    """evaluate the gold program with its numeral leaves replaced, in leaves() order, by fill (an iterator of values)"""
    r = len(prog) - 1 if r is None else r; op, A, B = prog[r]; v = []
    for S in (A, B):
        k, i = first(S)
        v.append(next(fill) if k == "t" else (GD.CONSTS[i] if k == "c" else run(prog, fill, i)))
    a, b = v
    return a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else (a / b if b != 0 else float("nan"))


def sents(it):
    sid = np.array(it["sid"]); toks = np.array(it["toks"], dtype=object)
    return [" ".join(t for t in toks[sid == s] if t != "<num>") for s in sorted(set(sid.tolist()))]


def tfidf(docs_tr, docs_te):
    df = collections.Counter(w for d in docs_tr for w in set(d.split())); n = len(docs_tr)
    vocab = {w: j for j, w in enumerate(df)}; idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in vocab])
    def mat(docs):
        M = np.zeros((len(docs), len(vocab)), np.float32)
        for i, d in enumerate(docs):
            for w, c in collections.Counter(d.split()).items():
                if w in vocab: M[i, vocab[w]] = c * idf[vocab[w]]
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return mat(docs_tr), mat(docs_te)


def knn_topk(Mtr, Mte, labels, k=K, top=5):
    out = []
    for s0 in range(0, len(Mte), 256):
        S = Mte[s0:s0 + 256] @ Mtr.T; nb = np.argsort(-S, 1)[:, :k]
        for i in range(len(S)):
            v = collections.Counter()
            for j in nb[i]: v[labels[j]] += float(S[i, j])
            out.append([l for l, _ in v.most_common(top)])
    return out


if __name__ == "__main__":
    D = json.load(open(GD.OUT)); TR, TE = D["TRAIN"], D["TEST"]
    sk_tr = [skeleton(it["prog"]) for it in TR]; sk_te = [skeleton(it["prog"]) for it in TE]
    root_tr = [it["prog"][-1][0] for it in TR]; root_te = [it["prog"][-1][0] for it in TE]
    S_tr = [sents(it) for it in TR]; S_te = [sents(it) for it in TE]
    L = [f"B-PLAN on TEST ({len(TE)} problems; TRAIN {len(TR)}); distinct TRAIN skeletons {len(set(sk_tr))}; "
         f"TEST skeletons seen in TRAIN {np.mean([s in set(sk_tr) for s in sk_te]):.3f}"]
    cov = {}
    for name, f in (("QUESTION", lambda s: s[-1]), ("FIRST (placebo)", lambda s: s[0]), ("WHOLE TEXT", lambda s: " ".join(s))):
        Mtr, Mte = tfidf([f(s) for s in S_tr], [f(s) for s in S_te])
        pk = knn_topk(Mtr, Mte, sk_tr); pr = knn_topk(Mtr, Mte, root_tr, top=1)
        t1 = np.mean([p[:1] == [g] for p, g in zip(pk, sk_te)]); t5 = np.mean([g in p for p, g in zip(pk, sk_te)])
        ro = np.mean([p[0] == g for p, g in zip(pr, root_te)]); cov[name] = t5
        L.append(f"  P1 {name:16s}: skeleton top-1 {t1:.3f} top-5 {t5:.3f} | root op {ro:.3f}")
    prior = [s for s, _ in collections.Counter(sk_tr).most_common(5)]
    rp = collections.Counter(root_tr).most_common(1)[0][0]
    L.append(f"  prior (5 most frequent TRAIN skeletons): top-5 {np.mean([g in prior for g in sk_te]):.3f} | root op prior "
             f"'{rp}' {np.mean([g == rp for g in root_te]):.3f}")
    by_len = collections.defaultdict(list)
    # P2: leaf fill
    reach, width, skipped = [], [], 0
    for it in TE:
        prog = it["prog"]; lv = leaves(prog); vals = it["vals"]
        nums = sorted({v for v in vals if v is not None}); k = len(lv)
        if math.perm(len(nums), k) > CAP: skipped += 1; continue
        gold = it["final"]; outs = set(); hit = False
        for fill in itertools.permutations(nums, k):
            try: v = run(prog, iter(fill))
            except (OverflowError, ZeroDivisionError): continue
            if v != v or abs(v) > 1e12: continue
            outs.add(round(v, 6)); hit |= GD.V1.near(v, gold)
        reach.append(hit); width.append(len(outs)); by_len[min(len(prog), 5)].append(len(outs))
    L.append(f"  P2 LEAF FILL (gold skeleton, injective numeral fill; {skipped} problems over {CAP} fills skipped): gold reachable "
             f"{np.mean(reach):.3f}; distinct answers median {np.median(width):.0f} (p25 {np.percentile(width, 25):.0f}, p75 "
             f"{np.percentile(width, 75):.0f}); median by program length " + " ".join(f"{k}{'+' if k == 5 else ''}:{np.median(v):.0f}"
             for k, v in sorted(by_len.items())))
    P1 = cov["QUESTION"] >= 0.50 and cov["QUESTION"] - cov["FIRST (placebo)"] >= 0.10
    P2 = np.mean(reach) >= 0.70 and np.median(width) <= 24
    L += [f"P1 THE GOAL CARRIES THE PLAN: question top-5 {cov['QUESTION']:.3f} (bar 0.50), vs placebo "
          f"{cov['QUESTION'] - cov['FIRST (placebo)']:+.3f} (bar +0.10) -> {'PASS' if P1 else 'FAIL'}",
          f"P2 THE FILL IS A SMALL SEARCH: reachable {np.mean(reach):.3f} (bar 0.70), median width {np.median(width):.0f} (bar 24) -> "
          f"{'PASS' if P2 else 'FAIL'}"]
    print("\n".join(L)); open(GD.V1.C("gsm_backward_ceiling.txt"), "w").write("\n".join(L) + "\n")
