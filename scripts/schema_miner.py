"""schema_miner.py — BRICK-M (2026-07-14): mine the recurring subgraph
schemas from every banked gold graph. The hierarchical library's
measurement-first entry: discovered-beats-designed — the second floor of
the library of primes gets PROPOSED by counting, admitted by design.

Units: rooted upstream-closed subgraphs (a factor + the producers of its
inputs, recursively) of size 2-6, WL-canonical with VALUES ABSTRACTED
(schemas are shapes, not numbers; ftype/op/sel retained). Train and
harvest mined separately, then joined.
"""
import json, sys, hashlib
from collections import defaultdict, Counter

TRAIN_SOURCES = {
    "nl":      ".cache/algebra_nl_train.jsonl",
    "alg2":    ".cache/algebra2_nl_train.jsonl",
    "alg3":    ".cache/algebra3_nl_train.jsonl",
    "alg4":    ".cache/algebra4_nl_train.jsonl",
    "verbose": ".cache/algv_train_verbose.jsonl",
    "dag6":    ".cache/dag_train.jsonl",
    "dag7":    ".cache/dag7_train.jsonl",
    "dag8":    ".cache/dag8_train.jsonl",
    "dag10":   ".cache/dag10_train.jsonl",
    "dag11":   ".cache/dag11_train.jsonl",
}
HARVEST = ".cache/book1_prose_pairs.jsonl"   # gold graphs of the book pairs
CAP = 2000                                    # rows per source (runtime cap; logged)


def skind(f):
    """Value-abstracted factor kind (shape, not number)."""
    ft = f["ftype"]
    if ft == "rel":
        return ("rel", f["op"])
    if ft == "sel":
        return ("sel", f["sel"])
    return (ft,)                              # given/mod/fdiv/pct — params abstracted


def members(f):
    ft = f["ftype"]
    if ft == "rel":
        return [("a", f["args"][0]), ("a", f["args"][1]), ("r", f["result"])]
    if ft == "given":
        return [("v", f["var"])]
    if ft in ("mod", "fdiv"):
        return [("s", f["var"]), ("r", f["result"])]
    if ft == "pct":
        return [("p", f["args"][0]), ("b", f["args"][1])]
    if ft == "sel":
        return [("a", f["args"][0]), ("a", f["args"][1]), ("r", f["result"])]
    raise ValueError(ft)


def inputs_of(f):
    ft = f["ftype"]
    if ft == "rel":
        return list(f["args"])
    if ft == "given":
        return []
    if ft in ("mod", "fdiv"):
        return [f["var"]]
    if ft == "pct":
        return [f["args"][1]]
    if ft == "sel":
        return list(f["args"])
    return []


def producer_of(f):
    ft = f["ftype"]
    if ft == "rel":
        return f["result"]
    if ft == "given":
        return f["var"]
    if ft in ("mod", "fdiv"):
        return f["result"]
    if ft == "pct":
        return f["args"][0]
    if ft == "sel":
        return f["result"]
    return None


def sub_canon(facs, root_i):
    """WL canonical digest of a small subgraph, root marked, values out."""
    vs = sorted({m for f in facs for _, m in members(f)})
    col = {v: 0 for v in vs}
    for _ in range(4):
        fcols = []
        for fi, f in enumerate(facs):
            aa = tuple(sorted(col[m] for r, m in members(f) if r == "a"))
            rest = tuple((r, col[m]) for r, m in members(f) if r != "a")
            fcols.append((skind(f), fi == root_i, aa, rest))
        inc = defaultdict(list)
        for f, fc in zip(facs, fcols):
            for r, m in members(f):
                inc[m].append((repr(fc), r))
        col2 = {v: repr((col[v], tuple(sorted(inc[v])))) for v in vs}
        ranks = {c: i for i, c in enumerate(sorted(set(col2.values())))}
        col = {v: ranks[col2[v]] for v in vs}
    sig = sorted(repr((skind(f), fi == root_i,
                       tuple(sorted(col[m] for r, m in members(f) if r == "a")),
                       tuple((r, col[m]) for r, m in members(f) if r != "a")))
                 for fi, f in enumerate(facs))
    return hashlib.sha256("|".join(sig).encode()).hexdigest()[:16]


def mine_graph(factors):
    """All rooted upstream-closed subgraphs, sizes 2-6, BFS-truncated."""
    prod = {}
    for i, f in enumerate(factors):
        pv = producer_of(f)
        if pv is not None and pv not in prod:
            prod[pv] = i
    out = []
    for root in range(len(factors)):
        seen = [root]
        frontier = [root]
        while frontier and len(seen) < 6:
            nxt = []
            for fi in frontier:
                for v in inputs_of(factors[fi]):
                    pi = prod.get(v)
                    if pi is not None and pi not in seen:
                        seen.append(pi)
                        nxt.append(pi)
                        if len(seen) >= 6:
                            break
                if len(seen) >= 6:
                    break
            frontier = nxt
        for k in range(2, len(seen) + 1):
            sub = [factors[i] for i in seen[:k]]
            out.append((sub_canon(sub, 0), k, seen[:k]))
    return out


def mine(rows, tag):
    classes = defaultdict(lambda: {"n": 0, "srcs": set(), "ex": None, "size": 0})
    for src, r in rows:
        for dg, k, idxs in mine_graph(r["factors"]):
            c = classes[dg]
            c["n"] += 1
            c["srcs"].add(src)
            c["size"] = k
            if c["ex"] is None:
                c["ex"] = (src, [(r["factors"][i]["ftype"],
                                  r["factors"][i].get("op", r["factors"][i].get("sel", "")))
                                 for i in idxs],
                           r.get("text", "")[:110])
    return classes


def main():
    train_rows = []
    for src, path in TRAIN_SOURCES.items():
        try:
            rows = [json.loads(l) for l in open(path)][:CAP]
        except FileNotFoundError:
            print(f"  [warn] missing {path}")
            continue
        train_rows += [(src, r) for r in rows]
    print(f"[miner] train rows: {len(train_rows)} (cap {CAP}/source — logged, not silent)")
    T = mine(train_rows, "train")

    hrows = [("book", json.loads(l)) for l in open(HARVEST)]
    H = mine(hrows, "harvest")
    print(f"[miner] harvest rows: {len(hrows)} | train classes: {len(T)} | harvest classes: {len(H)}")

    tot = sum(c["n"] for c in T.values())
    ranked = sorted(T.items(), key=lambda kv: -kv[1]["n"])
    print(f"\n=== P1: COVERAGE CURVE (train; {tot} subgraph occurrences, {len(T)} classes) ===")
    cum = 0
    for i, (dg, c) in enumerate(ranked[:20]):
        cum += c["n"]
        print(f"  #{i+1:2d} [{dg}] size {c['size']} n={c['n']:6d} srcs={len(c['srcs'])} "
              f"cum={cum/tot:.1%} | ex({c['ex'][0]}): {c['ex'][1]}")
    top15 = sum(c["n"] for _, c in ranked[:15]) / tot
    print(f"  P1 READ: top-15 cover {top15:.1%} (bar: >60%)")

    print(f"\n=== JOIN: harvest-vs-train (the diet instrument) ===")
    h_only = [(dg, c) for dg, c in sorted(H.items(), key=lambda kv: -kv[1]["n"])
              if dg not in T][:8]
    print(f"  harvest classes ABSENT from train (coverage gaps): {len([d for d in H if d not in T])}")
    for dg, c in h_only:
        print(f"    [{dg}] n={c['n']} size {c['size']} | ex: {c['ex'][1]} | {c['ex'][2]!r}")

    print(f"\n=== P3: THE CHRONIC FAMILY ===")
    by_item = {}
    for src, r in hrows:
        idx = r["gen"]["src_idx"]
        by_item[idx] = {dg for dg, k, _ in mine_graph(r["factors"])}
    if 45 in by_item and 7 in by_item:
        shared = by_item[45] & by_item[7]
        others = set()
        for i, s in by_item.items():
            if i not in (45, 7):
                others |= s
        distinctive = shared - others
        print(f"  [45] classes: {len(by_item[45])} | [7]: {len(by_item[7])} | "
              f"SHARED: {len(shared)} | shared-and-EXCLUSIVE to the pair: {len(distinctive)}")
        print(f"  P3 READ: {'SHARED CLASS EXISTS' if shared else 'no shared class'}"
              f"{' (and exclusive — the rate schema signature)' if distinctive else ''}")
    json.dump({dg: {"n": c["n"], "size": c["size"], "srcs": sorted(c["srcs"])}
               for dg, c in ranked[:50]},
              open(".cache/schema_mine_top50.json", "w"))
    print("\n[miner] top-50 classes -> .cache/schema_mine_top50.json (ranked, never admitted)")


if __name__ == "__main__":
    main()
