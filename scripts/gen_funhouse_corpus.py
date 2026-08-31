"""gen_funhouse_corpus.py — GENESIS's diet (2026-09-01, word given).
THE FUNHOUSE MINT: 8k ladder SHAPES (depth 2..16), each rendered THREE
times through different teeth (0.0 / 0.4 / 0.8 — clean face, light warp,
heavy warp) = 24k rows. The effective-n law applied deliberately: shapes
are n, surfaces are costumes; register invariance taught by construction.
Token-gated at birth. -> form_mix11 = form_mix8 + funhouse (share ~20%).
"""
import json
import random
import sys

sys.path.insert(0, 'scripts')
import os
os.environ.setdefault("ALG2", "1")
from algebra_nl_gen import gen_ladder, render, roundtrip
from phase1_algebra_head import TOKENIZER_JSON
from tokenizers import Tokenizer

_TOK = Tokenizer.from_file(TOKENIZER_JSON)
M = 300
rng = random.Random(1111)
rows, rej, shapes = [], 0, 0
while shapes < 8000:
    depth = rng.randint(2, 16)
    n_vars, factors, sol, query = gen_ladder(
        rng, depth, M, inverse_p=0.25 if depth <= 8 else 0.10)
    if n_vars > 24:
        rej += 1
        continue
    renders = []
    for teeth in (0.0, 0.4, 0.8):
        text, gfactors, mentions = render(
            rng, n_vars, [dict(f) for f in factors], query,
            shuffle_letters=(rng.random() < teeth * 0.5),
            oblique_prob=teeth * 0.35, distractor_prob=teeth * 0.4)
        ok, decisions = roundtrip(n_vars, gfactors, M, sol)
        if not ok or len(_TOK.encode(text).ids) > 250:
            renders = None
            break
        renders.append({"n_vars": n_vars, "m": M, "text": text,
                        "factors": gfactors, "mentions": mentions,
                        "query_var": query, "solution": sol,
                        "decisions": decisions,
                        "gen": {"ladder": depth, "teeth": teeth,
                                "shape_id": shapes}})
    if renders is None:
        rej += 1
        continue
    rows.extend(renders)
    shapes += 1
with open('.cache/funhouse24k.jsonl', 'w') as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"[funhouse] {shapes} shapes x 3 costumes = {len(rows)} rows "
      f"({rej} shapes rejected)")
base = [l for l in open('.cache/form_mix8.jsonl')]
mix = base + [json.dumps(r) + "\n" for r in rows]
random.Random(17).shuffle(mix)
with open('.cache/form_mix11.jsonl', 'w') as fh:
    fh.writelines(mix)
print(f"[mix] form_mix11: {len(mix)} rows; funhouse share "
      f"{len(rows) / len(mix) * 100:.1f}% (dose declared)")
