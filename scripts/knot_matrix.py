"""knot_matrix.py — THE KNOT-REHEARSAL MATRIX, first dietary use
(gen-10, 2026-07-12). Coverage report: distinct canonical classes per
DAG kind across the current training mix's dag rows; emits a dag10
booster quota targeting the THINNEST kinds (knot diversity, not row
count — the pigeonhole constraint made operational).
"""
import json, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from collections import defaultdict
from hash_audit_iso import canon

MIX = ".cache/algebra_mixed9b_train.jsonl"
BOOST_TOTAL = 2000

def kind_of(r):
    ks = set()
    givens = sum(1 for f in r["factors"] if f["ftype"] == "given")
    adds = sum(1 for f in r["factors"] if f["ftype"] == "rel" and f["op"] == "add")
    gvars = {f["var"] for f in r["factors"] if f["ftype"] == "given"}
    for f in r["factors"]:
        if f["ftype"] == "fdiv":
            ks.add("ifdiv" if f["var"] not in gvars else "fdiv")
        if f["ftype"] == "rel" and f["args"][0] == f["args"][1]:
            ks.add("isq" if f["args"][0] not in gvars else "sq")
    if givens >= 8 and adds >= 7:
        ks.add("ladder")
    lits = gvars
    muls = [f for f in r["factors"] if f["ftype"] == "rel" and f["op"] == "mul"]
    if sum(1 for f in muls if f["args"][0] in lits or f["args"][1] in lits) >= 2:
        ks.add("coupled")
    return ks or {"plain"}

rows_n = defaultdict(int)
knots = defaultdict(set)
for l in open(MIX):
    r = json.loads(l)
    if r.get("gen", {}).get("shape") not in ("dag", "dag7", "fdiv-tiny"):
        continue
    dg = canon(r)[0]
    for k in kind_of(r):
        rows_n[k] += 1
        knots[k].add(dg)

print("=== KNOT-REHEARSAL MATRIX (dag rows in the current mix) ===")
print(f"  {'kind':8s} {'rows':>6s} {'knots':>6s} {'knots/row':>9s}")
scores = {}
for k in sorted(knots, key=lambda k: len(knots[k])):
    n, kn = rows_n[k], len(knots[k])
    scores[k] = kn
    print(f"  {k:8s} {n:6d} {kn:6d} {kn/max(n,1):9.2f}")

# booster quota: inverse to knot count over the DAG7_QUOTA-mintable kinds
MINTABLE = ["ladder", "coupled", "sq", "fdiv", "isq", "ifdiv", "plain"]
inv = {k: 1.0 / max(scores.get(k, 1), 1) for k in MINTABLE}
tot = sum(inv.values())
quota = {k: max(100, int(BOOST_TOTAL * inv[k] / tot)) for k in MINTABLE}
# normalize to total
scale = BOOST_TOTAL / sum(quota.values())
quota = {k: int(v * scale) for k, v in quota.items()}
quota[MINTABLE[0]] += BOOST_TOTAL - sum(quota.values())
qstr = ",".join(f"{k}:{v}" for k, v in quota.items())
open(".cache/dag10_quota.txt", "w").write(qstr)
print(f"\n[knot-matrix] dag10 booster quota (thin kinds fed first): {qstr}")
