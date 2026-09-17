"""canonical_silver.py — THE CANONICAL NUMBERING (2026-09-16 night). The machine's variable
bank is positional and it learned the pen rows' convention: variables numbered in order of
FIRST APPEARANCE in the text (mint: 1.00). Sonnet numbers the givens first (text order 0.30),
so 70% of silver rows carried gold pointers in a foreign numbering — the base reads them at
0.03 fac-exact. This renumbers every variable by its first mention (a given's numeral span or a
mention span); a variable never mentioned takes the position of the factor that produces it
(factors in span-start order, the gold builder's own sort), placed after the mentioned
variables that precede that factor; args/result/query_var/mentions/solution are remapped and
every row is re-certified by the admission gate (the key at the new query var).
usage: canonical_silver.py in.jsonl out.jsonl"""
import json, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from admit_annotation import solver_verdict, key_of

def canonical_order(r):
    F = r["factors"]; n = r["n_vars"]; first = {}
    for f in F:
        if f["ftype"] == "given":
            for s in f.get("spans", []): first[f["var"]] = min(first.get(f["var"], 1e9), s[0])
    for v, sp in (r.get("mentions") or {}).items():
        for s in sp: first[int(v)] = min(first.get(int(v), 1e9), s[0])
    order = sorted(first, key=lambda v: (first[v], v))
    # unmentioned variables: the producing factor's span start (the sort key the gold builder uses)
    prod = {}
    for f in F:
        if "result" in f and f.get("spans"): prod[f["result"]] = min(s for s, _ in f["spans"])
        if f["ftype"] == "fdiv" and f.get("spans") and "result" in f: prod[f["result"]] = min(s for s, _ in f["spans"])
    rest = [v for v in range(n) if v not in first]
    for v in sorted(rest, key=lambda v: (prod.get(v, 1e9), v)):
        pos = prod.get(v, 1e9); i = 0
        while i < len(order) and first.get(order[i], prod.get(order[i], 1e9)) <= pos: i += 1
        order.insert(i, v)
    assert sorted(order) == list(range(n)), (order, n)
    return order

def remap(r, order):
    new = {old: i for i, old in enumerate(order)}; c = json.loads(json.dumps(r))
    for f in c["factors"]:
        if "var" in f: f["var"] = new[f["var"]]
        if "args" in f: f["args"] = [new[a] for a in f["args"]]
        if "result" in f: f["result"] = new[f["result"]]
    c["query_var"] = new[r["query_var"]]
    c["mentions"] = {str(new[int(v)]): sp for v, sp in (r.get("mentions") or {}).items()}
    if r.get("solution"): c["solution"] = [r["solution"][old] for old in order]
    g = dict(c.get("gen") or {}); g["canonical"] = "first-mention"; c["gen"] = g
    return c

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]; out = []; changed = 0; refused = 0
    for r in rows:
        order = canonical_order(r); c = remap(r, order); changed += order != list(range(r["n_vars"]))
        key = r.get("key", key_of(r.get("answer_field"))); ok, d = solver_verdict(c, key)
        if not ok: refused += 1; print("  refused after renumbering:", d, "|", r["text"][:80]); continue
        out.append(c)
    with open(sys.argv[2], "w") as f:
        for c in out: f.write(json.dumps(c) + "\n")
    print(f"[canonical] {len(rows)} rows: {changed} renumbered, {refused} refused by the gate after renumbering -> {len(out)} -> {sys.argv[2]}")
