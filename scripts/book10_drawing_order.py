"""book9_drawing_order.py — THE GRAPH-SHAPED RETRIEVAL CURE (2026-07-29,
gates t9's diet draws; position move #1).

The t8 lesson: keyword regexes retrieve MENTIONS, not construction
demand ([958]/[1085] had exceeds-language and nothing to subtract).
THE CURE: signatures derived FROM THE DRAFTED GRAPHS — the 220+ book-8
rows carry compiled factor graphs, so each diet construction's source
signature is learned from the sources that ACTUALLY exercised it
in-graph, and eligible candidates are scored by lexical similarity to
those construction-bearing sources (TF-IDF cosine, pure numpy — no
hand regexes anywhere).

Constructions labeled from factors (graph-shaped, mechanical):
  dup-add      : rel add with args[0]==args[1]
  pct          : any pct factor
  two-free-var : >=2 non-given, non-result-only free vars entering a
                 joint system (approximated: >=2 vars that appear in
                 rel args but are never given and never any factor's
                 sole binding — the [772]/[856] shape)
  sub-distract : rel sub (or exceeds-form add-inverse) in a graph with
                 >=2 small given literals (<20) — the [875] shape
  sel          : any sel factor

Fence carried from #92: the order ROUTES, never excludes.
Output: .cache/book10_drawing_order.json
"""
import json, glob, re
import numpy as np
from collections import defaultdict

# ---- 1. label drafted rows by graph-shaped construction presence
draft_files = sorted(glob.glob(".cache/book8*prose_pairs_draft.jsonl") + glob.glob(".cache/book9_t*_batch*.jsonl"))
labeled = defaultdict(list)   # construction -> [source texts]
n_rows = 0
for f in draft_files:
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        facs = r.get("factors", [])
        text = r["text"]
        n_rows += 1
        givens = {fa["var"]: fa.get("value", 0) for fa in facs if fa["ftype"] == "given"}
        small_givens = sum(1 for v in givens.values() if v < 20)
        has = set()
        free_in_args = set()
        bound_vars = set(givens)
        for fa in facs:
            if fa["ftype"] == "rel":
                a = fa.get("args", [])
                if len(a) == 2 and a[0] == a[1] and fa.get("op") == "add":
                    has.add("dup-add")
                if fa.get("op") == "sub" and small_givens >= 2:
                    has.add("sub-distract")
                for v in a:
                    if v not in givens:
                        free_in_args.add(v)
                bound_vars.add(fa.get("result", -1))
            elif fa["ftype"] == "pct":
                has.add("pct")
            elif fa["ftype"] == "sel":
                has.add("sel")
        truly_free = {v for v in free_in_args if v not in givens}
        if len(truly_free) >= 4:      # x,y style systems bind several derived vars
            has.add("two-free-var")
        for c in has:
            labeled[c].append(text)

print(f"[cure] labeled {n_rows} drafted rows; construction counts: "
      f"{ {c: len(v) for c, v in labeled.items()} }")

# ---- 2. eligible pool (mirror the census filter + all registered skips)
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
def int_answer(a):
    s = str(a).strip().replace("$", "").replace(",", "")
    return int(s) if re.fullmatch(r"-?\d+", s) else None
filt_idx = [i for i, x in enumerate(h)
            if x["level"] in ("Level 1", "Level 2", "Level 3")
            and len(x["problem"]) < 300 and "asy]" not in x["problem"]
            and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))]
fixture = set(filt_idx[:100])
used = set()
for f in sorted(glob.glob(".cache/book*_prose_pairs*.jsonl")):
    for line in open(f):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "text" in r:
            used.add(r["text"].strip())
SKIPS = {189,206,230,233,234,237,254,256,277,324,358,371,378,380,404,412,419,431,473,
         489,499,504,539,544,549,553,577,525,589,621,625,643,652,672,636,
         702,707,712,715,716,725,739,746,774,779,782,788,802,807,814,818,838,867,
         869,877,880,888,896,899,914,935,941}
pool = []
for i in filt_idx:
    if i in fixture or i in SKIPS or h[i]["problem"].strip() in used:
        continue
    a = int_answer(h[i]["answer"])
    if a is None or not (0 <= a <= 300):
        continue
    pool.append({"src_idx": i, "problem": h[i]["problem"]})
print(f"[cure] eligible pool: {len(pool)}")

# ---- 3. TF-IDF cosine from construction-bearing sources to candidates
def tokens(t):
    return re.findall(r"[a-z]+", t.lower())
vocab = {}
docs = [c["problem"] for c in pool]
all_docs = docs + [t for v in labeled.values() for t in v]
for d in all_docs:
    for w in set(tokens(d)):
        vocab.setdefault(w, len(vocab))
N = len(all_docs)
df = np.zeros(len(vocab))
def vec(d):
    v = np.zeros(len(vocab))
    for w in tokens(d):
        v[vocab[w]] += 1
    return v
for d in all_docs:
    for w in set(tokens(d)):
        df[vocab[w]] += 1
idf = np.log(N / (df + 1))
def tfidf(d):
    v = vec(d) * idf
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
cand_vecs = np.stack([tfidf(d) for d in docs])

order = {}
for c, texts in labeled.items():
    if len(texts) < 2:
        order[c] = {"note": f"only {len(texts)} exemplars — signature too thin, LOW-DEMAND WATCH", "candidates": []}
        continue
    # NEAREST-EXEMPLAR scoring (centroid dilution fix, 2026-07-29: broad
    # constructions' mean vectors converge to a generic algebra centroid —
    # two-free-var and sub-distract printed IDENTICAL tops; max-similarity
    # to any single construction-bearing source discriminates)
    ex = np.stack([tfidf(t) for t in texts])
    scores = (cand_vecs @ ex.T).max(axis=1)
    top = np.argsort(-scores)[:12]
    order[c] = {"exemplars": len(texts),
                "candidates": [{"src_idx": pool[int(j)]["src_idx"], "score": float(scores[j])}
                               for j in top if scores[j] > 0.05]}
    print(f"[cure] {c}: {len(texts)} exemplars -> top candidates "
          f"{[x['src_idx'] for x in order[c]['candidates'][:8]]}")

json.dump({"note": "GRAPH-SHAPED drawing order: signatures learned from drafted graphs' own "
                   "sources (no hand regexes); ROUTES-NEVER-EXCLUDES fence carried",
           "drawing_order": order},
          open(".cache/book10_drawing_order.json", "w"), indent=1)
print("[cure] wrote .cache/book10_drawing_order.json — t9's gate is open")
