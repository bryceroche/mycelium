"""book7_certify.py — the certification pass (2026-07-27): the 24 Sonnet
drafts through the STANDARD chain — g21 parse x 5 views, vote >=3, answer
key (gold==0). The key judges; the annotator never did. [1470] EXCLUDED
(held for Bryce's faithfulness ruling)."""
import sys, os, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from mycelium.attestation import attest_quorum_v3
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/g21.safetensors")
assert set(sd.keys()) == set(p.keys())
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def parse_batch(texts):
    n = len(texts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    out_r = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi < n: out_r.append(decode({k: o[k][bi] for k in o}))
    return out_r

rows = [json.loads(l) for l in open(".cache/book7_prose_pairs_draft.jsonl")]
res = {"certified": [], "abstain": [], "wrong": [], "held_1470": []}
for i, r in enumerate(rows):
    if r.get("harvest_idx") == 1470 or "1470" in str(r.get("gen", "")) or r.get("meta", {}).get("harvest_index") == 1470:
        res["held_1470"].append(i); continue
    gold = r["solution"][r["query_var"]]
    m = r.get("m", 300)
    dialect = r["gen"]["dialect"]   # SCHEMA CORRECTION (2026-07-27): text=SOURCE, gen.dialect=the annotation — book5's own convention; the first pass parsed raw LaTeX
    texts = [dialect] + [permuted_view(dialect, 90000 + 10*i + k) for k in range(1, 5)]
    views = [(f, q, solve2(f, q, {"n_vars": 24, "m": m})) for f, q in parse_batch(texts)]
    votes = [a for _, _, a in views]
    nn = [a for a in votes if a is not None]
    c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
    if cnt >= 3 and plur == gold:
        att, _, _ = attest_quorum_v3(views, plur, dialect, m, solve2)
        res["certified" if att else "abstain"].append({"i": i, "votes": votes})
    elif cnt >= 3:
        res["wrong"].append({"i": i, "plur": plur, "gold": gold, "votes": votes})
    else:
        res["abstain"].append({"i": i, "votes": votes})
    print(f"  [{i+1}/{len(rows)}] votes {votes} gold {gold} -> {'CERT' if cnt>=3 and plur==gold else ('WRONG' if cnt>=3 else 'abstain')}", flush=True)
print(f"\n=== BOOK 7 CERTIFICATION: certified {len(res['certified'])} | abstain {len(res['abstain'])} | wrong {len(res['wrong'])} | held(1470) {len(res['held_1470'])} ===")
json.dump(res, open(".cache/book7_certification.json", "w"), indent=1, default=int)
