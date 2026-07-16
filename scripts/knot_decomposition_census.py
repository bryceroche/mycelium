"""knot_decomposition_census.py — GUT #25, READS (b)+(c) (2026-07-16).

THE TWO ALGEBRAS, declared in sentence one: WHOLE-KNOT identity is
hash_audit_iso.canon (values IN, query-marked — the contamination
instrument); SUB-KNOT shape is schema_miner.sub_canon classes (values
OUT, rooted, sizes 2-6 — the recurrence instrument). This census is the
map between them: each whole-knot decomposed over the prime inventory.

TWO VIEWS, both pinned, jurisdictions declared (the countersign's
design): the COVER — maximal non-overlapping factorization, greedy by
(size desc, digest lex asc), tie-break PINNED so the cover is canonical
— answers diversity/novelty questions; the PROFILE — the full
downward-closed class multiset — is the FEATURE algebra (the triage
bank wants everything at every scale, not a lossy cover).

READ (c), kill-only: cyclomatic number (E - N + C on the bipartite
factor/var graph) per corpus — if books sit at ~0 cycles while bigtest
carries the mass, twenty-four's 'redundancy' was cycle count wearing
units and the mint's minimal-mode dial is a CYCLE dial.

Deliverable 2 (the 58's novelty split) is REGISTERED, NOT FIRED: knotted
census items have no banked parses; the census parse bank rides the
next census run.
"""
import json, sys
from collections import Counter, defaultdict

sys.path.insert(0, "scripts")
from schema_miner import TRAIN_SOURCES, CAP, mine_graph
from hash_audit_iso import canon


def cover(factors):
    """Canonical maximal non-overlapping factorization over the prime
    inventory. Returns a sorted tuple of class digests (the multiset)."""
    subs = sorted(mine_graph(factors), key=lambda t: (-t[1], t[0]))
    taken, out = set(), []
    for dg, k, idxs in subs:
        if not (set(idxs) & taken):
            taken.update(idxs)
            out.append(dg)
    return tuple(sorted(out))


def cyclomatic(factors):
    """E - N + C on the bipartite factor/var graph."""
    edges, vs = 0, set()
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, f in enumerate(factors):
        fn = ("f", i)
        parent.setdefault(fn, fn)
        mem = ([("a", f["args"][0]), ("a", f["args"][1]), ("r", f["result"])]
               if f["ftype"] in ("rel", "sel") else
               [("v", f["var"])] if f["ftype"] == "given" else
               [("s", f["var"]), ("r", f["result"])] if f["ftype"] in ("mod", "fdiv") else
               [("p", f["args"][0]), ("b", f["args"][1])])
        for _, v in mem:
            vn = ("v", v)
            parent.setdefault(vn, vn)
            vs.add(vn)
            edges += 1
            union(fn, vn)
    n = len(factors) + len(vs)
    comps = len({find(x) for x in parent})
    return edges - n + comps


# ---- READ (b): compositional diversity on train --------------------------
train_rows = []
for src, path in TRAIN_SOURCES.items():
    try:
        train_rows += [json.loads(l) for l in open(path)][:CAP]
    except FileNotFoundError:
        pass
print(f"[census] train rows: {len(train_rows)} (cap {CAP}/source — logged)")

whole = set()
covers = set()
cover_of_knot = {}
for r in train_rows:
    dg, _ = canon(r)
    cv = cover(r["factors"])
    whole.add(dg)
    covers.add(cv)
    cover_of_knot.setdefault(dg, cv)

print(f"\n=== READ (b): THE DECOMPOSITION CENSUS (train) ===")
print(f"  whole-knot classes (values in):        {len(whole)}")
print(f"  distinct cover-multisets (values out): {len(covers)}")
print(f"  compression ratio: {len(whole)/max(len(covers),1):.1f} knots per multiset")
sizes = Counter(len(cv) for cv in covers)
print(f"  cover-size histogram: {dict(sorted(sizes.items()))}")

# the feature bank's first stock: profiles for books + bigtest
def profile(factors):
    return dict(Counter(dg for dg, k, _ in mine_graph(factors)))

books = [json.loads(l) for p in ["book1", "book2", "book3"]
         for l in open(f".cache/{p}_prose_pairs.jsonl")]
big = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")][:200]
bank = {"books": [{"cover": list(cover(r["factors"])),
                   "profile": profile(r["factors"])} for r in books],
        "bigtest": [{"cover": list(cover(r["factors"])),
                     "profile": profile(r["factors"])} for r in big]}

book_covers = {tuple(b["cover"]) for b in bank["books"]}
novel = sum(1 for cv in book_covers if cv not in covers)
print(f"\n  books: {len(book_covers)} distinct covers; {novel} NOT in train's "
      f"{len(covers)} (compositional novelty at the whole-cover level)")

# ---- READ (c): the cycle read, kill-only ---------------------------------
import numpy as np
bc = np.array([cyclomatic(r["factors"]) for r in books])
gc = np.array([cyclomatic(r["factors"]) for r in big])
print(f"\n=== READ (c): THE CYCLE READ (kill-only) ===")
print(f"  BOOKS   cyclomatic: median {np.median(bc):.0f}  mean {bc.mean():.2f}  "
      f"frac==0: {(bc==0).mean():.1%}")
print(f"  BIGTEST cyclomatic: median {np.median(gc):.0f}  mean {gc.mean():.2f}  "
      f"frac==0: {(gc==0).mean():.1%}")
verdict = ("CYCLES EXPLAIN IT" if np.median(bc) == 0 and np.median(gc) >= 1
           else "CYCLES DO NOT EXPLAIN IT — the kill fires")
print(f"  kill frame: books ~0 while bigtest carries mass? -> {verdict}")

json.dump({"train_whole_knots": len(whole), "train_covers": len(covers),
           "cover_sizes": {str(k): v for k, v in sizes.items()},
           "books_novel_covers": novel, "books_distinct_covers": len(book_covers),
           "cyc_books": {"median": float(np.median(bc)), "mean": float(bc.mean()),
                         "zero_frac": float((bc == 0).mean())},
           "cyc_bigtest": {"median": float(np.median(gc)), "mean": float(gc.mean()),
                           "zero_frac": float((gc == 0).mean())},
           "verdict_c": verdict, "feature_bank": bank},
          open(".cache/knot_decomposition_census.json", "w"))
print("\n[census] banked -> .cache/knot_decomposition_census.json "
      "(incl. the triage feature bank's first stock)")
