"""gen_distractor_ladders.py — THE DISTRACTOR FIXTURE (2026-09-01).
Bryce's hypothesis: the masking null is an artifact of clean templates —
dilution can't be cured where it doesn't exist. Fixture: ladders at
depth {4,8,12,16} x teeth {0.0, 0.8} x 150 rows (held seed 4242).
Teeth: render's own dials (oblique/distractor/shuffled letters) + an
IRRELEVANT self-contained fragment (real gold factors, disjoint from
the query component — the parser must emit them; relevance is the
query's job). -> .cache/distval.jsonl with gen.bucket=(depth, teeth).
"""
import json
import random
import sys

sys.path.insert(0, 'scripts')
from algebra_nl_gen import gen_ladder, render, roundtrip

M = 300
rng = random.Random(4242)
rows, rej = [], 0
for depth in (4, 8, 12, 16):
    for teeth in (0.0, 0.8):
        got = 0
        while got < 150:
            n_vars, factors, sol, query = gen_ladder(
                rng, depth, M, inverse_p=0.25 if depth <= 8 else 0.10)
            if teeth > 0:
                # irrelevant fragment: tiny self-contained ladder over
                # fresh vars (2 anchors + 1 rung), disjoint from the query
                base = len(sol)
                a, b = rng.randint(1, M // 6), rng.randint(1, M // 6)
                sol += [a, b, a + b]
                factors.append({"ftype": "given", "var": base, "value": a,
                                "role": "anchor"})
                factors.append({"ftype": "given", "var": base + 1, "value": b,
                                "role": "anchor"})
                factors.append({"ftype": "rel", "op": "add",
                                "args": [base, base + 1],
                                "result": base + 2, "surface": "add"})
                n_vars = len(sol)
            if n_vars > 24:
                rej += 1
                continue
            text, gfactors, mentions = render(
                rng, n_vars, factors, query,
                shuffle_letters=(rng.random() < teeth * 0.5),
                oblique_prob=teeth * 0.35, distractor_prob=teeth * 0.4)
            ok, decisions = roundtrip(n_vars, gfactors, M, sol)
            if not ok:
                rej += 1
                continue
            rows.append({"n_vars": n_vars, "m": M, "text": text,
                         "factors": gfactors, "mentions": mentions,
                         "query_var": query, "solution": sol,
                         "decisions": decisions,
                         "gen": {"ladder": depth, "teeth": teeth,
                                 "bucket": f"d{depth}t{int(teeth * 10)}"}})
            got += 1
with open('.cache/distval.jsonl', 'w') as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"[distval] {len(rows)} rows ({rej} rejected) — depth x teeth grid, "
      f"held seed 4242")
