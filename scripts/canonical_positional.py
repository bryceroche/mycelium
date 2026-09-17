"""canonical_positional.py — THE POSITIONAL LAW (2026-09-17). The pen dialect the machine learned
on prose (13,475 diet rows, 1.000; the holdout 0.997): every factor introduces EXACTLY ONE new
variable and that variable's index IS the factor's slot — a given at slot k is variable k; a
relation at slot k introduces variable k (its result, or the one unknown argument: "40 = partner
+ 10"). The machine decodes prose in this scheme. Sonnet's rows obey it on 76% in their own order;
the gold builder's span sort (mint's convention) breaks it on every spanned row. This script
orders a row's factors greedily in READING order — a given is ready at its numeral, a relation
when its statement ends (span end) — taking at each step the earliest ready factor that
introduces exactly one unseen variable; renumbers variables by slot; remaps args / result / query
/ mentions / solution; stamps gen.canonical = "positional" (the gold builder keeps a stamped row's
order); re-certifies through the gate. A row with no such order (a relation introducing two
unknowns at once, a variable introduced by no factor) is refused.
usage: canonical_positional.py in.jsonl out.jsonl"""
import json, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from admit_annotation import solver_verdict, key_of

def vars_of(f):
    return ([f["var"]] if "var" in f else []) + list(f.get("args", [])) + ([f["result"]] if "result" in f else [])

def ready_pos(f):
    sp = f.get("spans") or []
    if not sp: return 10**9
    return min(s for s, _ in sp) if f["ftype"] == "given" else max(e for _, e in sp)

def positional_order(r):
    F = r["factors"]; seen = set(); order = []; left = list(range(len(F)))
    while left:
        cands = [i for i in left if len(set(vars_of(F[i])) - seen) == 1]
        if not cands: return None, f"no_positional_order_at_slot_{len(order)}"
        i = min(cands, key=lambda i: (ready_pos(F[i]), i)); order.append(i); seen |= set(vars_of(F[i])); left.remove(i)
    if len(seen) != r["n_vars"]: return None, f"vars_{r['n_vars']}_introduced_{len(seen)}"
    return order, None

def canonicalize(r):
    order, why = positional_order(r)
    if order is None: return None, why
    F = r["factors"]; new = {}; seen = set()
    for k, i in enumerate(order):
        (v,) = set(vars_of(F[i])) - seen; new[v] = k; seen |= set(vars_of(F[i]))
    c = json.loads(json.dumps(r)); facs = []
    for i in order:
        f = dict(F[i])
        if "var" in f: f["var"] = new[f["var"]]
        if "args" in f: f["args"] = [new[a] for a in f["args"]]
        if "result" in f: f["result"] = new[f["result"]]
        facs.append(f)
    c["factors"] = facs; c["query_var"] = new[r["query_var"]]
    c["mentions"] = {str(new[int(v)]): sp for v, sp in (r.get("mentions") or {}).items() if int(v) in new}
    if r.get("solution"): c["solution"] = [r["solution"][old] for old, k in sorted(new.items(), key=lambda kv: kv[1])]
    g = dict(c.get("gen") or {}); g["canonical"] = "positional"; c["gen"] = g
    return c, None

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]; out = []; why = {}; reordered = 0
    for r in rows:
        c, w = canonicalize(r)
        if c is None: why[w.split("_at_")[0]] = why.get(w.split("_at_")[0], 0) + 1; continue
        reordered += [f["ftype"] for f in c["factors"]] != [f["ftype"] for f in r["factors"]] or any(f.get("var") != g.get("var") for f, g in zip(c["factors"], r["factors"]))
        key = r.get("key", key_of(r.get("answer_field"))); ok, d = solver_verdict(c, key)
        if not ok: why["gate:" + d.split("@")[0]] = why.get("gate:" + d.split("@")[0], 0) + 1; continue
        out.append(c)
    with open(sys.argv[2], "w") as f:
        for c in out: f.write(json.dumps(c) + "\n")
    print(f"[positional] {len(rows)} rows -> {len(out)} ({reordered} changed); refused {why} -> {sys.argv[2]}", flush=True)
