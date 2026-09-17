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

RULE = __import__("os").environ.get("CP_RULE", "asap")   # "asap": a relation is stated as soon as its arguments are known (the pen shape
                                                          # "g g r g r"; slot 2 = the first relation); "spanend": at the end of its last evidence span

def law_holds(F):
    seen = set()
    for k, f in enumerate(F):
        vs = set(vars_of(f))
        if vs - seen != {k}: return False
        seen |= vs
    return True

def positional_order(r):
    F = r["factors"]; seen = set(); order = []; left = list(range(len(F))); intro = {}
    if RULE == "keep" and law_holds(F): return list(range(len(F))), None   # the extractor's (rationale) order IS the machine's convention
    while left:
        cands = [i for i in left if len(set(vars_of(F[i])) - seen) == 1]
        if not cands: return None, f"no_positional_order_at_slot_{len(order)}"
        def key(i):
            f = F[i]
            if RULE == "asap" and f["ftype"] != "given":
                known = [intro[v] for v in vars_of(f) if v in intro]
                return (max(known) + 0.5 if known else ready_pos(f), ready_pos(f), i)   # right after its last known argument
            return (ready_pos(f), 0, i)
        i = min(cands, key=key); order.append(i)
        (v,) = set(vars_of(F[i])) - seen; intro[v] = key(i)[0]; seen |= set(vars_of(F[i])); left.remove(i)
    if len(seen) != r["n_vars"]: return None, f"vars_{r['n_vars']}_introduced_{len(seen)}"
    return order, None

def reencode_ops(r):
    """THE GRAMMAR'S TWO OPERATORS (2026-09-17): the pen dialect is add and mul only (26,631 / 19,877 in the
    diet; the head's op gold is binary and decodes add/mul). sub(a,b)->r is add(b,r)->a; div(a,b)->r is
    mul(b,r)->a — the known moves to the result side and the unknown becomes an argument (the pen form
    "40 = partner + 10"). Sonnet's 329 sub / 131 div relations had been written into the gold as mul."""
    c = json.loads(json.dumps(r))
    for f in c["factors"]:
        if f["ftype"] == "rel" and f.get("op") == "sub": a, b = f["args"]; f["op"] = "add"; f["args"] = [b, f["result"]]; f["result"] = a
        elif f["ftype"] == "rel" and f.get("op") == "div": a, b = f["args"]; f["op"] = "mul"; f["args"] = [b, f["result"]]; f["result"] = a
    return c

def canonicalize(r):
    r = reencode_ops(r)
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
        key = r.get("key", key_of(r.get("answer_field"))); ok, d = (True, "nogate") if __import__("os").environ.get("CP_NOGATE") else solver_verdict(c, key)
        if not ok: why["gate:" + d.split("@")[0]] = why.get("gate:" + d.split("@")[0], 0) + 1; continue
        out.append(c)
    with open(sys.argv[2], "w") as f:
        for c in out: f.write(json.dumps(c) + "\n")
    print(f"[positional] {len(rows)} rows -> {len(out)} ({reordered} changed); refused {why} -> {sys.argv[2]}", flush=True)
