"""funnel_sweep.py — THE INDUSTRIAL FUNNEL, STAGE 1 (2026-08-28): g41
parses ALL remaining L2 raws directly; solve_forced vs the key. Yield
tiers: AUTO (raw parse hits key — machine-bankable after render+gate),
NEAR (solves, wrong answer — repair-lane), REFUSE (no solve — surgery).
Sizes the funnel before building the renderer."""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "8",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1"})
import numpy as np
from phase1_algebra_head import (T_ALG, build_params, forward, decode,
                                 sent_indices, TOKENIZER_JSON)
from beacon_closing_arm import recompute_states
from repair_replace_swap import solve_forced
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

L = json.load(open(".cache/book3_lanes.json"))
done = set()
for l in open('.cache/book3.jsonl'):
    done.add(json.loads(l)['raw'])
cands = [x for x in L if x.get('lane') == 'L2'
         and (x.get('problem') or x.get('raw', '')) not in done]
print(f"[fs] L2 remaining: {len(cands)}", flush=True)
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/g41_onemass_refold.safetensors")
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
auto = []; near = []; refuse = []
for s0 in range(0, len(cands), 8):
    sl = cands[s0:s0+8]
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
    snt = np.zeros((8, T_ALG), np.int32)
    for i, c in enumerate(sl):
        t = c.get('problem') or c.get('raw', '')
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    out = forward(p, Tensor(st.astype(np.float32), dtype=dtypes.float),
                  Tensor(msk, dtype=dtypes.float),
                  Tensor(snt.astype(np.int32), dtype=dtypes.int))
    keys = ("pres","ftype","op","islit","dig","args","res","query") + \
        tuple(k for k in ("sel","dup","sgn") if k in out)
    o = {k: out[k].realize().numpy() for k in keys}
    for i, c in enumerate(sl):
        if s0 + i >= len(cands): break
        facs, q = decode({k: o[k][i] for k in o})
        try:
            a = solve_forced(facs, q, {"n_vars": 24, "m": 300})
        except Exception:
            a = None
        gold = c.get('answer')
        if a is not None and a == gold:
            auto.append((c['idx'], gold, facs, q))
        elif a is not None:
            near.append((c['idx'], gold, a))
        else:
            refuse.append(c['idx'])
print(f"[fs] AUTO (raw parse hits key): {len(auto)}", flush=True)
print(f"[fs] NEAR (solves, wrong): {len(near)}", flush=True)
print(f"[fs] REFUSE (no solve): {len(refuse)}", flush=True)
print(f"[fs] auto idxs: {[a[0] for a in auto]}", flush=True)
json.dump([{'idx': i, 'answer': g,
            'facs': f, 'q': int(q)} for i, g, f, q in auto],
          open('.cache/funnel_auto.json', 'w'), default=int)
