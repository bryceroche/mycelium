"""macro_admission_review.py — THE RECURSION'S FIRST ADMISSION REVIEW
(2026-07-16): OPERATION-APPLY, with CHAIN(k)/PREFIX-SUM read beside it.

Charter (ledger 2026-07-14/15): miner finds recurring subgraph classes ->
classes PROPOSED as macro-factors (rank-never-admit) -> admitted macros
enter the registry with DETERMINISTIC EXPANSION -> book N+1 annotates at
the macro level. Guard rails: macros expand before the solver sees
anything (the key grades in primitives, always); hand-quota constitutional.

This script is the EVIDENCE HALF of the review: structural detectors run
over every banked gold graph (train, capped+logged, and all three books'
prose pairs = the harvest at today's full volume, n=182). It counts, it
prices the factor-count savings, it pins WL digests. It admits nothing.

Shapes under review (concrete detectors, value-abstracted):
  OP-APPLY-2  r = k1*x (+|-) k2*y   crown: {given k1, mul, given k2, mul, add/sub} 5 -> 1
  OP-APPLY-1  r = x (+|-) k*y       crown: {given k, mul, add/sub}                 3 -> 1
  CHAIN(k)    linear rel path, each factor consuming the previous result
  PREFIX-SUM  add-tree accumulating >=3 leaf inputs
"""
import json, sys
from collections import defaultdict, Counter

sys.path.insert(0, "scripts")
from schema_miner import TRAIN_SOURCES, CAP, sub_canon, producer_of, inputs_of

BOOKS = {
    "book1": ".cache/book1_prose_pairs.jsonl",
    "book2": ".cache/book2_prose_pairs.jsonl",
    "book3": ".cache/book3_prose_pairs.jsonl",
}


def index(factors):
    prod = {}
    for i, f in enumerate(factors):
        pv = producer_of(f)
        if pv is not None and pv not in prod:
            prod[pv] = i
    given_fed = {f["var"] for f in factors if f["ftype"] == "given"}
    return prod, given_fed


def detect_op_apply(factors):
    """Yield (variant, root_idx, member_idxs) for OP-APPLY crowns."""
    prod, given_fed = index(factors)
    for i, f in enumerate(factors):
        if f["ftype"] != "rel" or f["op"] not in ("add", "sub"):
            continue
        legs = []          # (arg, mul_idx or None, k_given_idx or None)
        for a in f["args"]:
            pi = prod.get(a)
            if pi is not None and factors[pi]["ftype"] == "rel" and factors[pi]["op"] == "mul":
                m = factors[pi]
                ks = [x for x in m["args"] if x in given_fed]
                if ks:
                    legs.append((a, pi, prod[ks[0]]))
                    continue
            legs.append((a, None, None))
        muls = [l for l in legs if l[1] is not None]
        if len(muls) == 2:
            mem = [i] + [x for l in muls for x in (l[1], l[2])]
            yield ("OP-APPLY-2", i, sorted(set(mem)))
        elif len(muls) == 1:
            mem = [i, muls[0][1], muls[0][2]]
            yield ("OP-APPLY-1", i, sorted(set(mem)))


def detect_chains(factors):
    """Longest linear rel chains (each consumes the previous result)."""
    prod, _ = index(factors)
    lens = []
    for i, f in enumerate(factors):
        if f["ftype"] != "rel":
            continue
        L, cur = 1, f
        seen = {i}
        while True:
            nxt = None
            for a in inputs_of(cur):
                pi = prod.get(a)
                if pi is not None and pi not in seen and factors[pi]["ftype"] == "rel":
                    nxt = pi
                    break
            if nxt is None:
                break
            seen.add(nxt); cur = factors[nxt]; L += 1
        lens.append(L)
    return max(lens) if lens else 0


def detect_prefix_sum(factors):
    """Add-trees accumulating >=3 distinct non-given leaf inputs."""
    prod, given_fed = index(factors)
    hits = 0
    for i, f in enumerate(factors):
        if f["ftype"] != "rel" or f["op"] != "add":
            continue
        leaves, frontier, seen = set(), [i], {i}
        while frontier:
            fi = frontier.pop()
            for a in inputs_of(factors[fi]):
                pi = prod.get(a)
                if pi is not None and factors[pi]["ftype"] == "rel" and \
                        factors[pi]["op"] == "add" and pi not in seen:
                    seen.add(pi); frontier.append(pi)
                else:
                    leaves.add(a)
        if len(seen) >= 2 and len(leaves) >= 3:
            hits += 1
    return hits


def load(path, cap=None):
    rows = [json.loads(l) for l in open(path)]
    return rows[:cap] if cap else rows


def survey(rows, tag):
    occ = Counter(); items = Counter(); digests = defaultdict(Counter)
    saved = 0; chain_hist = Counter(); psum_items = 0
    for r in rows:
        fs = r["factors"]
        kinds = set()
        used = set()
        for variant, root, mem in detect_op_apply(fs):
            occ[variant] += 1
            kinds.add(variant)
            digests[variant][sub_canon([fs[j] for j in mem], mem.index(root))] += 1
            if not (set(mem) & used):          # non-overlapping savings
                saved += len(mem) - 1
                used |= set(mem)
        for k in kinds:
            items[k] += 1
        chain_hist[detect_chains(fs)] += 1
        if detect_prefix_sum(fs):
            psum_items += 1
    n = len(rows)
    print(f"\n=== {tag} (n={n}) ===")
    for v in ("OP-APPLY-2", "OP-APPLY-1"):
        print(f"  {v}: {occ[v]} occurrences in {items[v]} items "
              f"({items[v]/n:.1%} of items)")
        for dg, c in digests[v].most_common(3):
            print(f"      digest [{dg}] n={c}")
    print(f"  factor-count savings if macro-annotated (non-overlap): "
          f"{saved} factors across {n} items ({saved/n:.2f}/item)")
    long = sum(c for L, c in chain_hist.items() if L >= 4)
    print(f"  CHAIN: max-length hist {dict(sorted(chain_hist.items()))} | items with chain>=4: {long}")
    print(f"  PREFIX-SUM(>=3 leaves): {psum_items} items")
    return {"occ": dict(occ), "items": dict(items), "n": n,
            "digests": {v: dict(d) for v, d in digests.items()},
            "saved": saved, "chain_ge4": long, "prefix_sum_items": psum_items}


def main():
    out = {}
    train_rows = []
    for src, path in TRAIN_SOURCES.items():
        try:
            train_rows += load(path, CAP)
        except FileNotFoundError:
            print(f"  [warn] missing train source {path}")
    print(f"[review] train rows: {len(train_rows)} (cap {CAP}/source — logged, not silent)")
    out["train"] = survey(train_rows, "TRAIN (generated dialect)")

    all_books = []
    for b, path in BOOKS.items():
        rows = load(path)
        all_books += rows
        out[b] = survey(rows, f"HARVEST {b}")
    out["harvest_all"] = survey(all_books, "HARVEST all books joined")

    json.dump(out, open(".cache/macro_admission_review.json", "w"), indent=1)
    print("\n[review] evidence -> .cache/macro_admission_review.json "
          "(ranked and priced, never admitted)")


if __name__ == "__main__":
    main()
