"""diet_corpus_schema_fix.py — THE SCHEMA REPAIR (2026-07-30, the deep
clean's first catch). The diet corpus v3 was minted with placeholder
fields that violate the mix row contract:
  decisions: []  -> build_gold expects the INT band label (the solver's
                    measured decision count; TypeError at mix row 78400)
  mentions:  []  -> build_gold expects a dict
  solution: [0]*24 -> mint rows carry LAWFUL solution vectors (custody
                    law); a zero vector is a lie waiting for a reader.
Repair, not workaround: decisions computed by the same v2 bridge every
consumer uses (solve_symbolic via problem_from_algebra3), solution
written from the solver's assignment, answer re-verified against the
mint's own solve2 check. Text and factors are UNTOUCHED (tokenization
and trunk states unchanged)."""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic
from tta_alg2_dials import solve2

PATH = ".cache/diet_fdiv_derived_corpus.jsonl"
rows = [json.loads(l) for l in open(PATH)]
assert len(rows) == 400
fixed = []
from collections import Counter
bands = Counter()
for r in rows:
    facs, q, nv, m = r["factors"], r["query_var"], r["n_vars"], r["m"]
    gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
    prob = problem_from_algebra3(nv, facs, gv, m)
    res = solve_symbolic(prob, budget=200_000, seed=0)
    assert res["status"] == "solved", r["text"]
    sol = [int(res["assignment"][v]) for v in range(nv)]
    a2 = solve2(facs, q, {"n_vars": nv, "m": m})
    assert a2 == sol[q], (a2, sol[q], r["text"])
    r["decisions"] = int(res.get("decisions", 0))
    r["mentions"] = {}
    r["solution"] = sol
    bands[r["decisions"]] += 1
    fixed.append(r)
with open(PATH, "w") as f:
    for r in fixed:
        f.write(json.dumps(r) + "\n")
print(f"[schema fix] 400/400 repaired; decisions bands: {dict(sorted(bands.items()))}")
print("[schema fix] mentions -> dict, solution -> solver assignment (custody-lawful)")
