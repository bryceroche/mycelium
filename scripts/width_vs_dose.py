"""width_vs_dose.py — GUT #44(a): are the crown's refusals WIDTH
casualties (simultaneous-binding budget) or FLUENCY casualties (dose)?
Per-item acceptance under crown_reader_v3, joined with expanded width.
PINNED: width wins if refused median expanded-factors >= accepted+2 AND
top-tertile refusal >= 2x bottom-tertile. Else dose stands."""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ["ALG_FTYPES"] = "8"
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/crown_reader_v3.safetensors")
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
        keys = [k for k in ("pres","ftype","op","islit","dig","dig2","args","res","query","sel","dup","y") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res

held = [json.loads(l) for l in open(".cache/fluency_mint_held.jsonl")]
out = []
for i, r in enumerate(held):
    texts = [r["macro"]["text"]] + [permuted_view(r["macro"]["text"], 700000+10*i+k) for k in range(1,5)]
    votes = []
    for facs, q in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None: votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None,0))
    ok = cnt >= 3 and top == r["answer"]
    pw = len(r["prime"]["factors"])
    out.append({"ok": bool(ok), "width": pw, "key": r["key"]})
W = np.array([o["width"] for o in out]); OK = np.array([o["ok"] for o in out])
med_r, med_a = np.median(W[~OK]), np.median(W[OK])
t1, t2 = np.percentile(W, 33), np.percentile(W, 66)
lo, hi = OK[W <= t1], OK[W > t2]
r_lo, r_hi = 1-lo.mean(), 1-hi.mean()
print(f"[width] refused median width {med_r:.0f} vs accepted {med_a:.0f} | "
      f"refusal rate: bottom tertile {r_lo:.0%} vs top {r_hi:.0%}")
width_wins = (med_r >= med_a + 2) and (r_hi >= 2*r_lo)
print(f"[width] VERDICT: {'WIDTH CASUALTIES — the binding budget caps; the fourth continuation is aimed wrong; macro-of-macro is the cure' if width_wins else 'DOSE STANDS — the fourth continuation proceeds as charted'}")
json.dump({"med_refused": float(med_r), "med_accepted": float(med_a),
           "r_lo": float(r_lo), "r_hi": float(r_hi), "width_wins": bool(width_wins),
           "items": out}, open(".cache/width_vs_dose.json","w"))
print("[width] banked")
