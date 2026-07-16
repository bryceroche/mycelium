"""test_macro_expansion.py — OPERATION-APPLY's admission exam (2026-07-16).

Four mechanical checks, all constitutional:
  1. LEVEL-INVARIANCE: the banked 3a+5b specimen's hand-primitive gold and
     the macro's expansion solve to the same answer through the same core
     (the key grades in primitives; certification is level-invariant).
  2. CROWN IDENTITY: the expansion's detected crown digest equals the
     pinned mined digest — the registry entry IS the harvested class.
  3. DETERMINISM: identical macro input -> byte-identical expansion.
  4. VARIANTS: sub op and k=1 (affine) legs expand and solve correctly.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts")
from mycelium.macros import expand_graph, OP_APPLY_CROWN_DIGEST, MACRO_GRAMMAR_VERSION
from macro_admission_review import detect_op_apply
from schema_miner import sub_canon
from tta_alg2_dials import solve2


def solve(facs, q, m=300):
    return solve2(facs, q, {"n_vars": 24, "m": m})


def macro_graph(op, k1, xv, k2, yv, x, y, result):
    return [
        {"ftype": "given", "var": x, "value": xv, "spans": []},
        {"ftype": "given", "var": y, "value": yv, "spans": []},
        {"ftype": "macro", "name": "OP_APPLY", "op": op,
         "k1": k1, "x": x, "k2": k2, "y": y, "result": result},
    ]


def crown_digest(facs):
    hits = list(detect_op_apply(facs))
    assert len(hits) == 1, hits
    variant, root, mem = hits[0]
    return variant, sub_canon([facs[j] for j in mem], mem.index(root))


# 1. LEVEL-INVARIANCE on the banked specimen (7 S 2 where a S b = 3a+5b)
spec = None
for line in open(".cache/book2_prose_pairs.jsonl"):
    r = json.loads(line)
    if "3a+5b" in r["text"] and "\\S" in r["text"]:
        spec = r
        break
assert spec is not None
gold_ans = solve(spec["factors"], spec["query_var"], spec["m"])
g, nv = expand_graph(macro_graph("add", 3, 7, 5, 2, x=0, y=1, result=2), n_vars=3)
macro_ans = solve(g, 2)
assert gold_ans == macro_ans == 31, (gold_ans, macro_ans)
print(f"[1] level-invariance: banked primitive gold = {gold_ans}, "
      f"macro expansion = {macro_ans} — same key, same answer")

# 2. CROWN IDENTITY against the pinned mined digest
variant, dg = crown_digest(g)
assert variant == "OP-APPLY-2" and dg == OP_APPLY_CROWN_DIGEST, (variant, dg)
print(f"[2] crown identity: expansion digest {dg} == pinned {OP_APPLY_CROWN_DIGEST}")

# 3. DETERMINISM (byte-level)
a = json.dumps(expand_graph(macro_graph("add", 3, 7, 5, 2, 0, 1, 2), 3))
b = json.dumps(expand_graph(macro_graph("add", 3, 7, 5, 2, 0, 1, 2), 3))
assert a == b
print("[3] determinism: byte-identical re-expansion")

# 3b. FLOOR-TWIN IDENTITY (gut #25, the floor-identity protocol): a
# macro-annotated row and its prime twin are ONE knot — canon() grades at
# level 0, so their digests match and verify_iso confirms exactly.
from hash_audit_iso import canon, verify_iso
macro_row = {"factors": macro_graph("add", 3, 7, 5, 2, 0, 1, 2),
             "n_vars": 3, "query_var": 2}
gp, gnv = expand_graph(macro_row["factors"], 3)
prime_row = {"factors": gp, "n_vars": gnv, "query_var": 2}
dg_m, _ = canon(macro_row)
dg_p, _ = canon(prime_row)
assert dg_m == dg_p, (dg_m, dg_p)
assert verify_iso(macro_row, prime_row)
print(f"[3b] floor-twin identity: macro digest == prime digest ({dg_m[:16]}), "
      f"verify_iso exact — one knot, two floors")

# 4. VARIANTS: sub, and the k=1 affine leg (sequence-style x + k*y)
g, _ = expand_graph(macro_graph("sub", 4, 10, 3, 2, 0, 1, 2), 3)
assert solve(g, 2) == 34
g, _ = expand_graph(macro_graph("add", 1, 2, 4, 3, 0, 1, 2), 3)
assert solve(g, 2) == 14
v, dg1 = crown_digest(g)
assert v == "OP-APPLY-1"
print(f"[4] variants: sub -> 34, affine k1=1 -> 14 (crown {dg1})")

print(f"\nADMISSION EXAM PASSED — OP_APPLY under grammar {MACRO_GRAMMAR_VERSION}")
