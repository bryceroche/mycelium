"""gsm_stepmem_read.py -- THE STEP-MEMORY READ (ledger 2026-11-10 pin; Bryce: "overfit the microscopic step level with the Tree DB ...
do not memorize the whole problem skeleton"). Zero GPU, zero training: a pure memory at the STEP grain.
Entry per TRAIN gold step: KEY = the bag of words of the step's ANCHOR sentences (the sentences of its token operands; a step with no
token operand anchors on the question = the last sentence), numerals -> <num>, names -> <name> (local by construction: blind to the
other sentences, to numbers and to names = the role tier's contract); VALUE = the step's op. TF-IDF cosine, 1-NN.
  RECALL  TRAIN steps queried against the full memory, their own entry included ("if we saw it again"): 1 - key collisions
  MEM     TEST steps: the op of the nearest TRAIN entry with the same operand kinds (gold grounding given: an anchored read)
  PRIOR   the majority TRAIN op for those operand kinds (the structure alone)
-> .cache/gsm_stepmem.txt"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_bench_data as GD
import gsm_cj_probe_skel as SK

K = 12; WIN = int(os.environ.get("SM_WIN", "0"))                     # 0 = the sentence key (STEPMEM); 3 = STEPMEM2
D = json.load(open(GD.OUT))


def kind(S): return "t" if S["t"] else ("s" if S["s"] else "c")


def steps(it):
    sid = it["sid"]; last = max(sid)
    words = ["<num>" if v is not None else ("<name>" if n else w) for w, v, n in zip(it["toks"], it["vals"], it["isn"])]
    out = []
    for o, A, B in it["prog"][:K]:
        if WIN and (A["t"] or B["t"]):   # (STEPMEM2) the words at each operand, side- and offset-tagged
            key = " ".join(f"{sd}{d:+d}:{words[i + d]}" for sd, S in (("a", A), ("b", B)) for i in S["t"][:1]
                           for d in range(-WIN, WIN + 1) if d and 0 <= i + d < len(words))
        else:
            anc = sorted({sid[i] for S in (A, B) for i in S["t"]}) or [last]
            key = " ".join(w for w, s in zip(words, sid) if s in anc)
        out.append((key, (kind(A), kind(B)), o))
    return out


tr = [s for it in D["TRAIN"] for s in steps(it)]
te = [s for it in D["TEST"] for s in steps(it)]
useen = np.concatenate([[SK.skeleton(it) not in SK.seen(D["TRAIN"])] * min(len(it["prog"]), K) for it in D["TEST"]]).astype(bool)
vec = TfidfVectorizer(token_pattern=r"[^ ]+", sublinear_tf=True).fit([s[0] for s in tr])
Xtr = vec.transform([s[0] for s in tr]); Xte = vec.transform([s[0] for s in te])
op_tr = np.array([s[2] for s in tr]); kd_tr = [s[1] for s in tr]
by_kind = defaultdict(list)
for i, kd in enumerate(kd_tr): by_kind[kd].append(i)
prior = {kd: Counter(op_tr[ix]).most_common(1)[0][0] for kd, ix in by_kind.items()}


def nn(X, kinds):
    pred = []
    for r in range(X.shape[0]):
        ix = by_kind.get(kinds[r]); 
        if not ix: pred.append(None); continue
        sim = (Xtr[ix] @ X[r].T).toarray().ravel(); pred.append(op_tr[ix][int(sim.argmax())])
    return np.array(pred, dtype=object)


rec = nn(Xtr, kd_tr) == op_tr
gold = np.array([s[2] for s in te]); kte = [s[1] for s in te]
mem = nn(Xte, kte) == gold; pri = np.array([prior.get(k) == g for k, g in zip(kte, gold)])
noanc = np.array([k == ("s", "s") or "t" not in k for k in kte])
out = [f"STEP MEMORY: {len(tr)} TRAIN step entries ({len(set(s[0] for s in tr))} distinct keys) vs {len(SK.seen(D['TRAIN']))} TRAIN skeletons",
       f"  RECALL (TRAIN, own entry in memory): {rec.mean():.4f}",
       f"  TEST all  (n {len(te)}): MEM {mem.mean():.4f} PRIOR {pri.mean():.4f} d {mem.mean() - pri.mean():+.4f}",
       f"  TEST-U    (n {useen.sum()}): MEM {mem[useen].mean():.4f} PRIOR {pri[useen].mean():.4f} d {mem[useen].mean() - pri[useen].mean():+.4f}",
       f"  TEST seen (n {(~useen).sum()}): MEM {mem[~useen].mean():.4f} PRIOR {pri[~useen].mean():.4f}",
       f"  no text anchor (n {noanc.sum()}): MEM {mem[noanc].mean():.4f} PRIOR {pri[noanc].mean():.4f} | anchored: MEM {mem[~noanc].mean():.4f} PRIOR {pri[~noanc].mean():.4f}"]
d = mem[useen].mean() - pri[useen].mean()
out.append(f"MEM-U: {d:+.4f} (>= +0.050) -> {'NAMED' if d >= 0.05 else 'none named'} | RECALL >= 0.99 -> {'PASS' if rec.mean() >= 0.99 else 'FAIL'}")
txt = "\n".join(out); print(txt); open(f".cache/gsm_stepmem{'_w' + str(WIN) if WIN else ''}.txt", "w").write(txt + "\n")
