"""Zero-GPU census: nesting depth of non-given factors in gold factor graphs.
STRUCTURE ONLY (factors/args/result/var); never reads solution fields.
Depth: given var = 0. A non-given factor with exactly one undetermined variable
determines it at 1+max(depth of its other vars); factor nesting = max depth of the
consumed (other) vars. Factors processed lowest-depth-first. A factor whose vars are
all already determined when reached = 'redundant' (consistency check; reported apart).
A factor never reaching <=1 unknown = 'simultaneous'.
Sets: wild = wild_admitted_holdout (ledger 'wildhold'); mint = algebra_nl_test (test23)."""
import json, os, sys, collections, statistics
OUT = ".cache/nesting_census.txt"
assert not os.path.exists(OUT), "refuse overwrite " + OUT
SETS = [("wild", ".cache/wild_admitted_holdout.jsonl"),
        ("mint test23", ".cache/algebra_nl_test.jsonl"),
        ("mint dag8_test", ".cache/dag8_test.jsonl")]

def fvars(f):
    v = list(f.get("args", []))
    for k in ("var", "result"):
        if k in f and isinstance(f[k], int): v.append(f[k])
    if "sel" in f and isinstance(f["sel"], int) and f["ftype"] == "sel": v.append(f["sel"]) if False else None
    return sorted(set(v))

def analyze(d):
    fs = d["factors"]; depth = {}
    for f in fs:
        if f["ftype"] == "given": depth[f["var"]] = 0
    todo = [i for i, f in enumerate(fs) if f["ftype"] != "given"]
    res = {}  # i -> (nest, nconsumed_derived, kind)
    while todo:
        best = None
        for i in todo:
            vs = fvars(fs[i]); unk = [v for v in vs if v not in depth]
            if len(unk) <= 1:
                cons = [depth[v] for v in vs if v in depth]
                nest = max(cons) if cons else 0
                key = (nest, i)
                if best is None or key < best[0]: best = (key, i, unk, cons)
        if best is None: break
        _, i, unk, cons = best
        nest = max(cons) if cons else 0
        nder = sum(1 for c in cons if c >= 1)
        if unk:
            depth[unk[0]] = nest + 1; res[i] = (nest, nder, "ok")
        else:
            res[i] = (nest, nder, "redundant")
        todo.remove(i)
    for i in todo: res[i] = (None, 0, "simultaneous")
    return res

lines = []
def P(s=""): print(s); lines.append(s)
for name, path in SETS:
    rows = [json.loads(l) for l in open(path)]
    H = collections.Counter(); pm = collections.Counter(); ft = collections.Counter()
    nf = 0; ge2 = 0; two = 0; nonsim_nest = []; unk_types = collections.Counter()
    for d in rows:
        r = analyze(d); mx = -1; sim = False
        for i, (nest, nder, kind) in r.items():
            nf += 1; ft[d["factors"][i]["ftype"]] += 1
            if kind == "simultaneous": H["simultaneous"] += 1; sim = True; continue
            if kind == "redundant": H["redundant"] += 1; continue
            H[min(nest, 4)] += 1; nonsim_nest.append(nest); mx = max(mx, nest)
            if nest >= 2: ge2 += 1
            if nder >= 2: two += 1
        pm["sim" if sim and mx < 0 else (min(mx, 4) if mx >= 0 else "none")] += 1
    P(f"== {name}  ({path})  n_problems={len(rows)}  n_nongiven_factors={nf}  by ftype={dict(ft)}")
    P("factor nesting histogram (ok factors; 4 = 4+): " + "  ".join(f"{k}:{H[k]}" for k in [0,1,2,3,4,"redundant","simultaneous"]))
    tot_ok = len(nonsim_nest)
    P(f"share nesting>=2: {ge2}/{nf} = {ge2/max(nf,1):.4f} (of ok factors {ge2/max(tot_ok,1):.4f})")
    P(f"share with >=2 derived args consumed: {two}/{nf} = {two/max(nf,1):.4f}")
    P("per-problem max nesting hist: " + "  ".join(f"{k}:{pm[k]}" for k in [0,1,2,3,4,"sim","none"]) + "  (sim = problem whose only factors are simultaneous)")
    P(f"median factor nesting (ok): {statistics.median(nonsim_nest) if nonsim_nest else None}; "
      f"median per-problem chain length (max nesting+1 factors): see hist; mean nesting {sum(nonsim_nest)/max(tot_ok,1):.3f}")
    P()
open(OUT, "x").write("\n".join(lines) + "\n")
