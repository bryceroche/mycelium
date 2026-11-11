"""gsm_span_read.py -- THE SPAN READ (ledger 2026-11-10 pin; Bryce: "panama hats ... Jensen Shannon divergence"). Zero model, zero GPU.
JSD SEGMENTER (TRAIN text, numerals -> <num>, names -> <name>): a token a COHERES with what follows when its successor distribution
departs from the unigram: c(a) = JSD(P(. | a), P(.)) (bits). A boundary falls after token i when c(w_i) < the median c over TRAIN
tokens, and always at punctuation/sentence ends. Spans = the runs between boundaries.
YARDSTICK = the STEPMEM read with KEY = the span holding each operand + the span after it, side-tagged ("a:<num> eggs per day").
PLACEBO = random boundaries at the same rate (rng 0), same key. Steps with no token operand keep the question-sentence key.
-> .cache/gsm_span.txt"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_bench_data as GD
import gsm_cj_probe_skel as SK

K = 12; PUN = set(".,;:?!'\"()’")
D = json.load(open(GD.OUT)); seen = SK.seen(D["TRAIN"])


def words(it): return ["<num>" if v is not None else ("<name>" if n else w) for w, v, n in zip(it["toks"], it["vals"], it["isn"])]


uni = Counter(); big = defaultdict(Counter)
for it in D["TRAIN"]:
    w = words(it)
    for a, b in zip(w, w[1:]): uni[b] += 1; big[a][b] += 1
V = sorted(uni); vi = {w: i for i, w in enumerate(V)}; pu = np.array([uni[w] for w in V], float); pu /= pu.sum()


def jsd(p, q):
    m = (p + q) / 2; f = lambda x: np.sum(x[x > 0] * np.log2(x[x > 0] / m[x > 0]))
    return (f(p) + f(q)) / 2


coh = {}
for a, c in big.items():
    p = np.full(len(V), 0.0); idx = [vi[b] for b in c]; p[idx] = list(c.values()); p = 0.9 * p / p.sum() + 0.1 * pu   # smoothed toward unigram
    coh[a] = jsd(p, pu)
med = float(np.median([coh.get(w, 0.0) for it in D["TRAIN"] for w in words(it) if w not in PUN]))


def bounds(it, rng=None, rate=None):
    w = words(it); sid = it["sid"]
    b = [True] * len(w)
    for i in range(len(w) - 1):
        if w[i] in PUN or w[i + 1] in PUN or sid[i] != sid[i + 1]: b[i] = True
        elif rng is not None: b[i] = rng.random() < rate
        else: b[i] = coh.get(w[i], 0.0) < med
    return b                                                     # b[i]: a boundary after token i


def spans(it, b):
    sp, cur = [], 0; out = []
    for i in range(len(b)):
        out.append(cur)
        if b[i]: cur += 1
    return out                                                   # span id per token


def steps(it, sid_sp):
    w = words(it); sid = it["sid"]; last = max(sid); res = []
    for o, A, B in it["prog"][:K]:
        if A["t"] or B["t"]:
            key = " ".join(f"{sd}:{w[j]}" for sd, S in (("a", A), ("b", B)) for i in S["t"][:1]
                           for j in range(len(w)) if sid_sp[j] in (sid_sp[i], sid_sp[i] + 1))
        else:
            key = " ".join(x for x, s in zip(w, sid) if s == last)
        res.append((key, (SK.first_kind(A), SK.first_kind(B)), o))
    return res


def read(make):
    tr = [s for it in D["TRAIN"] for s in steps(it, make(it))]; te = [s for it in D["TEST"] for s in steps(it, make(it))]
    vec = TfidfVectorizer(token_pattern=r"[^ ]+", sublinear_tf=True).fit([s[0] for s in tr])
    Xtr, Xte = vec.transform([s[0] for s in tr]), vec.transform([s[0] for s in te]); op = np.array([s[2] for s in tr])
    byk = defaultdict(list)
    for i, s in enumerate(tr): byk[s[1]].append(i)
    prior = {k: Counter(op[ix]).most_common(1)[0][0] for k, ix in byk.items()}

    def nn(X, ss):
        return np.array([op[byk[s[1]]][int((Xtr[byk[s[1]]] @ X[r].T).toarray().argmax())] if s[1] in byk else None for r, s in enumerate(ss)], dtype=object)
    rec = (nn(Xtr, tr) == op).mean(); g = np.array([s[2] for s in te])
    u = np.concatenate([[SK.skeleton(it) not in seen] * min(len(it["prog"]), K) for it in D["TEST"]]).astype(bool)
    mem = nn(Xte, te) == g; pri = np.array([prior.get(s[1]) == s[2] for s in te])
    return rec, mem[u].mean(), pri[u].mean(), mem[~u].mean()


nb = [b for it in D["TRAIN"] for b in bounds(it)]; rate_free = None
# the placebo's rate = the JSD boundary rate on non-forced positions
forced = [w[i] in PUN or w[i + 1] in PUN or it["sid"][i] != it["sid"][i + 1] for it in D["TRAIN"] for w in [words(it)] for i in range(len(w) - 1)]
jb = [coh.get(words(it)[i], 0.0) < med for it in D["TRAIN"] for i in range(len(words(it)) - 1)]
rate = float(np.mean([b for b, f in zip(jb, forced) if not f]))
rng = np.random.default_rng(0)
J = read(lambda it: spans(it, bounds(it)))
P = read(lambda it: spans(it, bounds(it, rng, rate)))
ex = D["TEST"][0]; sp = spans(ex, bounds(ex)); w = words(ex)
demo = " | ".join(" ".join(x for x, s in zip(w, sp) if s == k) for k in range(max(sp) + 1))[:400]
out = [f"JSD segmenter: median coherence {med:.4f} bits; free-boundary rate {rate:.3f}; mean span len "
       f"{np.mean([np.bincount(spans(it, bounds(it))).mean() for it in D['TEST']]):.2f} tokens",
       f"  demo: {demo}",
       f"  JSD spans : RECALL {J[0]:.4f} | TEST-U MEM {J[1]:.4f} (prior {J[2]:.4f}, d {J[1] - J[2]:+.4f}) | seen MEM {J[3]:.4f}",
       f"  PLACEBO   : RECALL {P[0]:.4f} | TEST-U MEM {P[1]:.4f} (d {P[1] - P[2]:+.4f}) | seen MEM {P[3]:.4f}",
       f"  window +-3 (STEPMEM2, banked): RECALL 0.9582 | TEST-U MEM 0.5047",
       f"SPAN-x (JSD - window >= +0.020): {J[1] - 0.5047:+.4f} -> {'PASS' if J[1] - 0.5047 >= 0.020 else 'FAIL'} | "
       f"SPAN-PLACEBO (JSD - placebo >= +0.020): {J[1] - P[1]:+.4f} -> {'PASS' if J[1] - P[1] >= 0.020 else 'FAIL'} | RECALL >= 0.99 -> {'PASS' if J[0] >= 0.99 else 'FAIL'}"]
txt = "\n".join(out); print(txt); open(".cache/gsm_span.txt", "w").write(txt + "\n")
