"""zone_read.py — the two-silhouette read (gut #56, first administration).

Splits a fixture's rows into UMBRA (unanimous-correct), PENUMBRA (gold
present among the 5 views but vote split or plurality), and DARK (no
view finds gold), against the key. Bars pinned in the ledger before
this ran: on the hundreds held fixture's misses under g17_armR —
penumbra >= 60% validates the 17b dose bet; dark >= 60% re-prices it.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

CKPT = os.environ.get("ZONE_CKPT", ".cache/g17_armR.safetensors")
FIXTURE = os.environ.get("ZONE_FIXTURE", ".cache/gen17_hundreds_held.jsonl")
OUT = os.environ.get("ZONE_OUT", ".cache/zone_read_hundreds_armR.json")

rows = [json.loads(l) for l in open(FIXTURE)]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[zone] ckpt {CKPT} | fixture {FIXTURE} ({len(rows)} rows)")


def parse_batch(texts):
    n = len(texts)
    N = ((n + 7) // 8) * 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    res = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "dig2", "args",
                            "res", "query", "sel", "dup", "y") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res


zones = Counter()
miss_zone = Counter()
detail = []
for ri, r in enumerate(rows):
    gold_ans = r["solution"][r["query_var"]]
    dia = r["text"]
    texts = [dia] + [permuted_view(dia, 720000 + 100 * ri + k) for k in range(1, 5)]
    votes = []
    for facs, q in parse_batch(texts):
        votes.append(solve2(facs, q, {"n_vars": 24, "m": r["m"]}))
    solid = [v for v in votes if v is not None]
    top, cnt = (Counter(solid).most_common(1)[0] if solid else (None, 0))
    if cnt == 5 and top == gold_ans:
        z = "umbra"
    elif gold_ans in solid:
        z = "penumbra"
    else:
        z = "dark"
    zones[z] += 1
    correct_by_vote = cnt >= 3 and top == gold_ans
    if not (top == gold_ans and cnt == len(votes) and None not in votes):
        pass
    # a MISS in the battery's single-view sense: classify misses by zone
    # (battery eval was single-view; here the miss set = rows where the
    # plurality vote fails the key)
    if not correct_by_vote:
        miss_zone[z] += 1
    detail.append(dict(i=ri, gold=gold_ans, votes=votes, zone=z))

n = len(rows)
print(f"[zone] full fixture: umbra {zones['umbra']} / penumbra {zones['penumbra']}"
      f" / dark {zones['dark']}  (n={n})")
nm = sum(miss_zone.values())
pen_share = miss_zone["penumbra"] / max(nm, 1)
dark_share = miss_zone["dark"] / max(nm, 1)
print(f"[zone] vote-miss set: {nm} rows -> penumbra {miss_zone['penumbra']}"
      f" ({pen_share:.0%}) / dark {miss_zone['dark']} ({dark_share:.0%})"
      f" / umbra-wrong {miss_zone['umbra']}")
if pen_share >= 0.60:
    verdict = "PENUMBRA-SHAPED — the dose bet VALIDATED (reps reach rows the rays already touch)"
elif dark_share >= 0.60:
    verdict = "DARK-SHAPED — reps buy nothing; the charter re-prices at the verdict"
else:
    verdict = "MIXED — neither zone >=60%; the gap takes both medicines, dose bet partially funded"
print(f"[zone] VERDICT: {verdict}")
json.dump(dict(ckpt=CKPT, fixture=FIXTURE, zones=dict(zones),
               miss_zones=dict(miss_zone), pen_share=pen_share,
               dark_share=dark_share, verdict=verdict, detail=detail),
          open(OUT, "w"), indent=1)
print(f"[zone] banked -> {OUT}")
