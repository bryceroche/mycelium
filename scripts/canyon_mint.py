"""canyon_mint.py — the band-dose probe's substrate (2026-07-22): canyon-
shape rows minted through the standing gates — 2-digit divisors across
the range, clamp-floor neighbors, mag-3 operands: the three wild
specimens' shapes at volume."""
import json, sys, random
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from hash_audit_iso import canon
from tta_alg2_dials import solve2
rng = random.Random(5100)
LET = "abcdefghijklmnopqrstuvwx"
out, seen, attempts = [], set(), 0
while len(out) < 1500 and attempts < 60000:
    attempts += 1
    pat = rng.choice(["fdiv2", "fdiv2chain", "mag3", "modk"])
    if pat in ("fdiv2", "fdiv2chain"):
        k = rng.randint(10, 99)
        b = rng.randint(2, 300 // k) if k <= 150 else 1
        x = k * b + rng.randint(0, k - 1)
        if x > 300: continue
        facs = [{"ftype": "given", "var": 0, "value": x, "spans": []},
                {"ftype": "fdiv", "var": 0, "k": k, "result": 1, "spans": []}]
        q = 1
        text = f"Consider the numbers a, b, c. a is {x}. When a is divided by {k}, the quotient is b and the remainder is c. What is b?"
        if pat == "fdiv2chain" and b + rng.randint(1, 20) <= 300:
            d = rng.randint(1, 20)
            facs.append({"ftype": "given", "var": 2, "value": d, "spans": []})
            facs.append({"ftype": "rel", "op": "add", "args": [1, 2], "result": 3, "spans": []})
            q = 3
            text = (f"Consider the numbers a, b, c, d, e. a is {x}. When a is divided by {k}, "
                    f"the quotient is b and the remainder is c. d is {d}. b plus d equals e. What is e?")
    elif pat == "mag3":
        v1, v2 = rng.randint(100, 250), rng.randint(10, 50)
        facs = [{"ftype": "given", "var": 0, "value": v1, "spans": []},
                {"ftype": "given", "var": 1, "value": v2, "spans": []},
                {"ftype": "rel", "op": "add", "args": [0, 1], "result": 2, "spans": []}]
        q = 2
        if v1 + v2 > 300: continue
        text = f"Consider the numbers a, b, c. a is {v1}. b is {v2}. a plus b equals c. What is c?"
    else:
        k = rng.randint(10, 60)
        x = rng.randint(k + 1, 300)
        facs = [{"ftype": "given", "var": 0, "value": x, "spans": []},
                {"ftype": "mod", "var": 0, "k": k, "result": 1, "spans": []}]
        q = 1
        text = f"Consider the numbers a, b. a is {x}. When a is divided by {k}, the remainder is b. What is b?"
    a_ = solve2(facs, q, {"n_vars": 24, "m": 300})
    if a_ is None or not (0 <= a_ <= 300): continue
    dg, _ = canon({"factors": facs, "n_vars": 24, "query_var": q})
    if dg in seen: continue
    seen.add(dg)
    out.append({"text": text, "factors": facs, "mentions": {}, "n_vars": 24,
                "query_var": q, "decisions": 1, "m": 300, "solution": [0]*24})
with open(".cache/canyon_rows.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print(f"[canyon-mint] {len(out)} rows ({attempts} attempts)")
