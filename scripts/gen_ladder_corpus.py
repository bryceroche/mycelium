"""gen_ladder_corpus.py — form9's ladder corpus (2026-08-31, the ladder
era). Generates N ladder rows (depth 2-6, inverse_p .25, m=300), gates
each through the standard roundtrip/uniqueness door, ASSERTS 2-of-3 gold
ladder density >= 0.85 on the corpus (the whole point), then builds the
declared-dose mix: form_mix9.jsonl = form_mix8 + ladders, shuffled
(ladder share printed — the dose law).
"""
import json
import random
import sys

sys.path.insert(0, 'scripts')
from algebra_nl_gen import gen_ladder, render, roundtrip

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24000
SEED, M = 909, 300
rng = random.Random(SEED)
rows, n_rej = [], 0
dens = []
while len(rows) < N:
    depth = rng.randint(2, 6)
    n_vars, factors, sol, query = gen_ladder(rng, depth, M)
    if n_vars > 24:
        n_rej += 1
        continue
    text, gfactors, mentions = render(rng, n_vars, factors, query)
    ok, decisions = roundtrip(n_vars, gfactors, M, sol)
    if not ok:
        n_rej += 1
        continue
    # 2-of-3 density on this row's gold
    giv = {f["var"] for f in gfactors if f["ftype"] == "given"}
    slots = [set(f.get("args", [])) | {f.get("result")}
             for f in gfactors if f["ftype"] == "rel"]
    det, fired = set(giv), 0
    for _ in range(8):
        for vv in slots:
            unk = [v for v in vv if v not in det]
            if len(unk) == 1:
                det.add(unk[0])
                fired += 1
    dens.append(fired / max(len(slots), 1))
    rows.append({"n_vars": n_vars, "m": M, "text": text, "factors": gfactors,
                 "mentions": mentions, "query_var": query, "solution": sol,
                 "decisions": decisions,
                 "gen": {"seed": SEED, "ladder": depth}})
d = sum(dens) / len(dens)
assert d >= 0.85, f"ladder density {d:.3f} < 0.85 — the corpus misses its point"
with open('.cache/ladder24k.jsonl', 'w') as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"[ladder] {len(rows)} rows ({n_rej} rejected), 2-of-3 density {d:.3f}")

base = [l for l in open('.cache/form_mix8.jsonl')]
lad = [json.dumps(r) + "\n" for r in rows]
mix = base + lad
random.Random(11).shuffle(mix)
with open('.cache/form_mix9.jsonl', 'w') as fh:
    fh.writelines(mix)
print(f"[mix] form_mix9: {len(mix)} rows; ladder share "
      f"{len(lad) / len(mix) * 100:.1f}% (the dose law: declared)")
