"""adversarial_walk.py — the manufactured species' promotion exam
(sentinel column 5). Usage: adversarial_walk.py <cand> <ckpt>.
Walks the 20 scope specimens through the candidate (5-view votes) and
the panama guard; banks per-specimen outcomes. The exam's question:
would any specimen CERTIFY (unanimous) while the guard is not wired?
And: does the guard flag all 20 (the wiring's precondition)?"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "7")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from panama_guard import load_lexicon, guard
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

cand, ckpt = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(ckpt)
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
lex = load_lexicon()

spec = [json.loads(l) for l in open(".cache/adversarial_scope_fixture.jsonl")]
out = []
for si, s in enumerate(spec):
    texts = [s["text"]] + [permuted_view(s["text"], 400000 + 10 * si + k)
                           for k in range(1, 5)]
    N = 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    o = forward(p, Tensor(st.astype(np.float32), dtype=dtypes.float),
                Tensor(msk.astype(np.float32), dtype=dtypes.float),
                Tensor(snt.astype(np.int32), dtype=dtypes.int))
    keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "dig2", "args",
                        "res", "query", "sel", "dup", "y") if k in o]
    onp = {k: o[k].realize().numpy() for k in keys}
    votes = []
    for bi in range(5):
        facs, q = decode({k: onp[k][bi] for k in keys})
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    flagged, _ = guard(s["text"], lex)
    out.append({"kind": s["kind"], "pair": s["pair"], "gold": s["gold"],
                "majority": top, "count": cnt, "unanimous": bool(cnt == 5),
                "wrong_unanimous": bool(cnt == 5 and top != s["gold"]),
                "guard_flagged": bool(flagged)})
wu = sum(o_["wrong_unanimous"] for o_ in out)
fl = sum(o_["guard_flagged"] for o_ in out)
print(f"[adversarial:{cand}] unanimous-wrong {wu}/20 | guard flags {fl}/20 "
      f"(wiring precondition {'HOLDS' if fl == 20 else 'FAILS'})")
json.dump(out, open(f".cache/adversarial_walk_{cand}.json", "w"))
