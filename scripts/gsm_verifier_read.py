"""gsm_verifier_read.py -- THE STEP VERIFIER READ (ledger 2026-11-10 pin; Bryce: "memorize genericized spans into the Tree DB but not the
skeleton linking them"). Zero GPU, no training: the step memory (STEPMEM2's +-3 operand-window keys over TRAIN gold steps, numerals and
names masked) JUDGES each step of each candidate program from the SELECT pool (MACRO_s0: greedy + 16 draws, saved programs).
SUPPORT(step) = max TF-IDF cosine between the step's window key and the TRAIN entries with the SAME (op, kind a, kind b); a step with
no token operand carries no local evidence and is skipped. SUPPORT(program) = mean over its anchored steps (none -> 0).
No links are judged by memory -- the chain is the solver's (FCERT / CERT certificates, as in SELECT).
  VER   = argmax (FCERT, CERT, SUPPORT, LL)      VER-ONLY = argmax (SUPPORT, LL)      (ties: greedy first)
Graded on ANSWERS (the ORDER audit). -> .cache/gsm_verifier.txt"""
import json, os, sys
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_bench_data as GD
import gsm_cj_probe_skel as SK

K = 12; WIN = 3; ARM = "MACRO_s0"
D = json.load(open(GD.OUT)); its = D["TEST"]; seen = SK.seen(D["TRAIN"])


def words(it): return ["<num>" if v is not None else ("<name>" if n else w) for w, v, n in zip(it["toks"], it["vals"], it["isn"])]


def wkey(w, ia, ib):
    return " ".join(f"{sd}{d:+d}:{w[i + d]}" for sd, i in (("a", ia), ("b", ib)) if i is not None
                    for d in range(-WIN, WIN + 1) if d and 0 <= i + d < len(w))


tr = []
for it in D["TRAIN"]:
    w = words(it)
    for o, A, B in it["prog"][:K]:
        if A["t"] or B["t"]:
            tr.append((wkey(w, A["t"][0] if A["t"] else None, B["t"][0] if B["t"] else None), (o, SK.first_kind(A), SK.first_kind(B))))
vec = TfidfVectorizer(token_pattern=r"[^ ]+", sublinear_tf=True).fit([s[0] for s in tr]); Xtr = vec.transform([s[0] for s in tr])
byk = defaultdict(list)
for i, s in enumerate(tr): byk[s[1]].append(i)


def support(prog, it, w):
    sc = []
    for o, a, b in prog:
        ia = a[1] if a[0] == "t" and a[1] < len(w) else None; ib = b[1] if b[0] == "t" and b[1] < len(w) else None
        if ia is None and ib is None: continue
        ix = byk.get((o, a[0], b[0]))
        sc.append(0.0 if not ix else float((Xtr[ix] @ vec.transform([wkey(w, ia, ib)]).T).toarray().max()))
    return float(np.mean(sc)) if sc else 0.0


d = np.load(f".cache/gsm_select_{ARM}.npz"); P = json.load(open(f".cache/gsm_select_{ARM}_progs.json"))
A, F, LL = d["A"], d["F"], d["LL"]; C = A.shape[0]
SUP = np.array([[support(P[c][j], its[j], words(its[j])) for j in range(len(its))] for c in range(C)])
pu = np.array([SK.skeleton(it) not in seen for it in its])
sel = A[d["pick_SEL"], np.arange(len(its))]
out = [f"{ARM}: greedy ans {A[0].mean():.4f} | SEL (banked rule) ans {sel.mean():.4f} | BEST@{C} ans {A.any(0).mean():.4f} | memory "
       f"{len(tr)} anchored TRAIN steps",
       f"  support: answer-right candidates {SUP[A].mean():.4f} vs wrong {SUP[~A].mean():.4f}"]
res = {}
for name, keys in {"VER": (0, 1, "sup", "ll"), "VER-ONLY": ("sup", "ll")}.items():
    pick = np.array([max(range(C), key=lambda c: (tuple(SUP[c, j] if q == "sup" else LL[c, j] if q == "ll" else F[c, j, q] for q in keys), -c))
                     for j in range(len(its))])
    an = A[pick, np.arange(len(its))]; res[name] = an
    out.append(f"  {name}: ans {an.mean():.4f} (vs SEL {an.mean() - sel.mean():+.4f}, vs greedy {an.mean() - A[0].mean():+.4f}) | unseen-skeleton "
               f"ans {an[pu].mean():.4f} (SEL {sel[pu].mean():.4f}) | greedy kept {np.mean(pick == 0):.3f}")
v = res["VER"].mean() - sel.mean()
out.append(f"VER-x (VER - SEL ans >= +0.020): {v:+.4f} -> {'PASS' if v >= 0.020 else 'FAIL'}")
txt = "\n".join(out); print(txt); open(".cache/gsm_verifier.txt", "w").write(txt + "\n")
