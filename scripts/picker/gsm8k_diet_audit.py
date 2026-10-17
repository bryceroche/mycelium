"""gsm8k_diet_audit.py -- WHERE DO 13.4k DISTINCT GSM8K TEXTS COME FROM? (2026-10-16, Bryce: 'run the audit'; zero GPU.)
GSM8K train has 7,473 problems. Every diet row tagged src=gsm8k is matched to the dataset by TEXT (whitespace-normalized):
  EXACT-TRAIN   the question verbatim                 EXACT-TEST  a GSM8K TEST question (must be 0: the test split is never trained)
  SENT-PERM     the same sentences in another order (a TTA-style view)
  SUBSET        every sentence is a train question's sentence (a truncated / sub-problem)
  OTHER         none of the above (paraphrase, number change, ...): sampled for eyeballing
Per class: rows, distinct texts, distinct source problems; by admission route. Also: wild/silver vs the diet's source problems.
Out: .cache/gsm8k_diet_audit.txt"""
import collections, glob, json, re
import pyarrow.parquet as pq

snap = glob.glob(".cache/gsm8k/datasets--openai--gsm8k/snapshots/*/main/")[0]
tr = pq.read_table(snap + "train-00000-of-00001.parquet").to_pydict(); te = pq.read_table(snap + "test-00000-of-00001.parquet").to_pydict()
norm = lambda s: re.sub(r"\s+", " ", s).strip()
sents = lambda s: tuple(x.strip() for x in re.split(r"(?<=[.?!])\s+", norm(s)) if x.strip())
TR = {norm(q): i for i, q in enumerate(tr["question"])}; TE = {norm(q): i for i, q in enumerate(te["question"])}
PERM = {tuple(sorted(sents(q))): i for i, q in enumerate(tr["question"])}
SENT = collections.defaultdict(set)
for i, q in enumerate(tr["question"]):
    for s in sents(q): SENT[s].add(i)
TES = collections.defaultdict(set)
for i, q in enumerate(te["question"]):
    for s in sents(q): TES[s].add(i)

def classify(text):
    t = norm(text)
    if t in TE: return "EXACT-TEST", TE[t]
    if t in TR: return "EXACT-TRAIN", TR[t]
    ss = sents(t)
    if tuple(sorted(ss)) in PERM: return "SENT-PERM", PERM[tuple(sorted(ss))]
    if ss and all(s in SENT for s in ss):
        common = set.intersection(*(SENT[s] for s in ss))
        if common: return "SUBSET", min(common)
    if ss and sum(s in TES for s in ss) >= max(2, len(ss) // 2): return "TEST-OVERLAP?", None
    return "OTHER", None

out = []
def say(s=""): print(s); out.append(s)
C = collections.Counter(); T = collections.defaultdict(set); P = collections.defaultdict(set); A = collections.Counter(); ex = collections.defaultdict(list)
allsrc = set()
for l in open(".cache/form_mix_pm35c_RO.jsonl"):
    r = json.loads(l); g = r.get("gen") or {}
    if not (isinstance(g, dict) and g.get("src") == "gsm8k"): continue
    k, i = classify(r["text"]); C[k] += 1; T[k].add(norm(r["text"])); A[(k, str(g.get("admission") or g.get("admitted_by") or "?")[:36])] += 1
    if i is not None: P[k].add(i); allsrc.add(i)
    if len(ex[k]) < 4: ex[k].append(r["text"][:160])
say(f"GSM8K-tagged diet rows: {sum(C.values())} | distinct texts {sum(len(v) for v in T.values())} | distinct GSM8K TRAIN source problems {len(allsrc)} of {len(tr["question"])}")
for k in ("EXACT-TRAIN", "SENT-PERM", "SUBSET", "OTHER", "TEST-OVERLAP?", "EXACT-TEST"):
    say(f"  {k:14s} rows {C[k]:6d} | texts {len(T[k]):6d} | source problems {len(P[k]):5d}")
say("  by admission route:")
for (k, a), v in sorted(A.items(), key=lambda x: -x[1]): say(f"    {k:14s} {a:36s} {v:6d}")
for k in ("SENT-PERM", "SUBSET", "OTHER", "TEST-OVERLAP?"):
    for e in ex[k]: say(f"  e.g. {k}: {e}")
for p in (".cache/wild_admitted_holdout.jsonl", ".cache/silver460_test_RO.jsonl"):
    cl = collections.Counter(); hit = 0; n = 0
    for l in open(p):
        r = json.loads(l); k, i = classify(r["text"]); cl[k] += 1; n += 1; hit += i in allsrc
    say(f"{p}: {dict(cl)} | rows whose source problem is ALSO in the diet: {hit}/{n}")
open(".cache/gsm8k_diet_audit.txt", "w").write("\n".join(out) + "\n")
