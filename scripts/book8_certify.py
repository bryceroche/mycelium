"""book8_certify.py — the certification pass (2026-07-28): the 29 Sonnet
drafts through the STANDARD chain — g21 parse x 5 views, vote >=3, answer
key, attestation v3. The key judges; the annotator never did.

CUSTODY CORRECTION over book7_certify.py: gold comes from the HARVEST
answer (via .cache/book8_candidates.json, keyed by src_idx), never from
the draft's own solution vector — the certifier's key must be independent
of the annotator's artifact (the row's solution descends from the draft's
own graph; using it as gold would let the pen grade itself)."""
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

cands = json.load(open(".cache/book8_candidates.json"))
harvest_gold = {int(c["src_idx"]): int(c["answer"]) for c in cands["tranche1"]}

rows = [json.loads(l) for l in open(".cache/book8_prose_pairs_draft.jsonl")]
res = {"certified": [], "abstain": [], "wrong": []}
for i, r in enumerate(rows):
    src = int(r["gen"]["src_idx"])
    gold = harvest_gold[src]                       # THE KEY: harvest answer, not the draft's artifact
    assert r["solution"][r["query_var"]] == gold, \
        f"draft [{src}] solution disagrees with harvest gold — pen/key divergence, audit before certifying"
    m = r.get("m", 300)
    dialect = r["gen"]["dialect"]   # schema law: text=SOURCE, gen.dialect=the annotation
    texts = [dialect] + [permuted_view(dialect, 91000 + 10*i + k) for k in range(1, 5)]
    views = [(f, q, solve2(f, q, {"n_vars": 24, "m": m})) for f, q in parse_batch(texts)]
    votes = [a for _, _, a in views]
    nn = [a for a in votes if a is not None]
    c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
    if cnt >= 3 and plur == gold:
        att, _, _ = attest_quorum_v3(views, plur, dialect, m, solve2)
        res["certified" if att else "abstain"].append({"i": i, "src_idx": src, "votes": votes})
    elif cnt >= 3:
        res["wrong"].append({"i": i, "src_idx": src, "plur": plur, "gold": gold, "votes": votes})
    else:
        res["abstain"].append({"i": i, "src_idx": src, "votes": votes})
    print(f"  [{i+1}/{len(rows)}] src {src} votes {votes} gold {gold} -> {'CERT' if cnt>=3 and plur==gold else ('WRONG' if cnt>=3 else 'abstain')}", flush=True)
print(f"\n=== BOOK 8 T1 CERTIFICATION: certified {len(res['certified'])} | abstain {len(res['abstain'])} | wrong {len(res['wrong'])} ===")
json.dump(res, open(".cache/book8_certification.json", "w"), indent=1, default=int)
