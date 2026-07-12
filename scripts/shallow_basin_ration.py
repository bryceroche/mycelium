"""shallow_basin_ration.py — THE SHALLOW-BASIN INSTRUMENT (gen-9,
2026-07-12; chartered on the physics night, motivated by [45]).

Vote entropy across TTA views = per-item effective temperature. This
instrument samples training rows, runs the 5-view gate under the gen-8
head, and identifies CORRECT-BUT-SHALLOW rows (top vote correct, vote
multiset non-unanimous) — the retraining-target class, self-identified
by its own temperature. Output: row texts to OVERSAMPLE (x2) in mixed9;
rehearsal deepens basins (measured: [71]/[7]/[72] vote-shy -> 5/5 after
one generation of rehearsal).

Env: SB_CKPT (default gen-8), SB_SAMPLE (default 3000), SB_SEED.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

rows = [json.loads(l) for l in open(".cache/algebra_mixed8_train.jsonl")]
rng = np.random.RandomState(int(os.environ.get("SB_SEED", "9")))
idxs = rng.choice(len(rows), int(os.environ.get("SB_SAMPLE", "3000")),
                  replace=False)

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
CKPT = os.environ.get("SB_CKPT", ".cache/phase1_gen8_head.safetensors")
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def parse_batch(texts):
    n = len(texts)
    N = ((n + 7) // 8) * 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    res = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + \
            (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res

shallow, deep, wrong, refused = [], 0, 0, 0
for ci, ri in enumerate(idxs):
    r = rows[int(ri)]
    gold = r["solution"][r["query_var"]]
    texts = [r["text"]] + [permuted_view(r["text"], 50000 + 10*ci + k)
                           for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": r.get("m", 60)})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    if top == gold and cnt == 5:
        deep += 1
    elif top == gold:                     # correct-but-shallow: the class
        shallow.append(r["text"])
    elif cnt >= 3:
        wrong += 1
    else:
        refused += 1
    if (ci + 1) % 500 == 0:
        print(f"  ...{ci+1}/{len(idxs)}: deep {deep} shallow {len(shallow)} "
              f"wrong {wrong} refused {refused}", flush=True)

print(f"\n[shallow-basin] deep {deep} | SHALLOW {len(shallow)} | wrong {wrong} "
      f"| refused {refused}  (n={len(idxs)}, ckpt {CKPT})")
json.dump(shallow, open(".cache/shallow_basin_texts.json", "w"))
print(f"[saved] .cache/shallow_basin_texts.json (oversample x2 in mixed9)")
