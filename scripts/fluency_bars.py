"""fluency_bars.py — THE FLUENCY RUN'S BARS (2026-07-21, per charter: per-key acceptance, dividend re-fire, canyon re-check, floor 1194).
(1) FRAC_OF acceptance on truly-held-out mints (the greedy arm's 90,
    knot-filtered against the trained corpus) — bar >=70% at 5-view >=3.
(2) THE DIVIDEND READ: [73]'s 5-factor macro form (10v/3fdiv -> 5/5) and
    [100]'s best crowned form (fdivs irreducible — the honest attempt).
(3) THE CANYON BAR: param-digit accuracy, crown forms vs matched chain
    forms — errors must drop with decode-site count.
(4) Displacement floor: bigtest >= 1182 (C1's 1197 − 15).
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ["ALG_FTYPES"] = "8"
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, L_FAC, N_DIG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from hash_audit_iso import canon
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/crown_reader_v2.safetensors")
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print("[bars] crown_reader loaded (FTYPES=8)")


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
                res.append((decode({k: o[k][bi] for k in o}),
                            {k: o[k][bi] for k in ("dig", "dig2", "ftype", "pres")}))
    return res


def vote5(text, answer, seed0):
    texts = [text] + [permuted_view(text, seed0 + k) for k in range(1, 5)]
    votes = []
    for (facs, q), _ in parse_batch(texts):
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    return (cnt >= 3 and top == answer), votes


# (1) per-key acceptance on the knot-disjoint held-out mints
held = [json.loads(l) for l in open(".cache/fluency_mint_held.jsonl")]
by_key = {}
for i, r in enumerate(held):
    ok, _ = vote5(r["macro"]["text"], r["answer"], 700000 + 10 * i)
    by_key.setdefault(r["key"], []).append(ok)
pooled = sum(sum(v) for v in by_key.values()) / len(held)
mins = {k: sum(v)/len(v) for k, v in by_key.items()}
print(f"[BAR 1] acceptance POOLED {pooled:.0%} "
      f"({'PASS' if pooled >= 0.70 else 'FAIL'} vs 70%) | per-key " +
      " ".join(f"{k}:{v:.0%}" for k, v in sorted(mins.items())) +
      f" | MIN-KEY {min(mins.values()):.0%} "
      f"({'PASS' if min(mins.values()) >= 0.50 else 'FAIL'} vs 50%)")
rate = pooled

# (2) THE DIVIDEND READ
D = "Consider the numbers "
d73 = (D + "a, b, c, d, e. a is 56. 3 times a, divided by 7, gives a quotient "
       "of b. When a is divided by 4, the quotient is c. b plus c equals d. "
       "When d is divided by 2, the quotient is e. What is e?")
ok73, v73 = vote5(d73, 19, 600000)
d100 = (D + "a, b, c, d, e, f, g, h, i. a is 3. When a is divided by 2, the "
        "quotient is b. c is 9. When c is divided by 4, the quotient is d. "
        "e is 81. When e is divided by 16, the quotient is f. g is 3. "
        "b plus d equals h. h plus f plus g equals i. What is i?")
ok100, v100 = vote5(d100, 11, 610000)
print(f"[BAR 2] THE DIVIDEND: [73] {'BANKS — THE WALL FALLS' if ok73 else 'misses'} "
      f"(votes {v73}) | [100] {'BANKS' if ok100 else 'misses (fdivs irreducible)'} "
      f"(votes {v100})")

# (3) THE CANYON BAR: matched crown-vs-chain param accuracy
import random
rng = random.Random(4321)
crown_texts, chain_texts, golds = [], [], []
for _ in range(60):
    x = rng.randint(3, 90); a = rng.choice([2, 3, 4, 5, 6, 7]); k = rng.choice([2, 3, 4, 5, 6])
    if a * x > 300:
        continue
    g_ = (a * x) // k
    crown_texts.append(D + f"a, b. a is {x}. {a} times a, divided by {k}, "
                       f"gives a quotient of b. What is b?")
    chain_texts.append(D + f"a, b, c, d. a is {a}. b is {x}. a times b equals c. "
                       f"When c is divided by {k}, the quotient is d. What is d?")
    golds.append(g_)
cr = parse_batch(crown_texts)
ch = parse_batch(chain_texts)
c_ok = sum(1 for (fp_, q), _ in cr
           for a_ in [solve2(fp_, q, {"n_vars": 24, "m": 300})]
           if a_ == golds[cr.index(((fp_, q), _))] if True) if False else None
c_right = sum(1 for i, ((f_, q), _) in enumerate(cr)
              if solve2(f_, q, {"n_vars": 24, "m": 300}) == golds[i])
h_right = sum(1 for i, ((f_, q), _) in enumerate(ch)
              if solve2(f_, q, {"n_vars": 24, "m": 300}) == golds[i])
n_c = len(golds)
print(f"[BAR 3] THE CANYON: crown-form answers {c_right}/{n_c} "
      f"({c_right/n_c:.0%}) vs chain-form {h_right}/{n_c} ({h_right/n_c:.0%}) — "
      f"{'DAMPING HOLDS (crown >= chain)' if c_right >= h_right else 'DAMPING FAILS'}")

# (4) displacement floor
z = np.load(".cache/phase1_alg_states_bigtest.npz")
rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
st, tkm, se = z["states"], z["tokmask"], z["sent"]
n_ans = 0
for s0 in range(0, len(rows), 8):
    sl = np.arange(s0, min(s0 + 8, len(rows)))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    out = forward(p, Tensor(st[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(tkm[sl_p].astype(np.float32), dtype=dtypes.float),
                  Tensor(se[sl_p].astype(np.int32), dtype=dtypes.int))
    keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "dig2", "args",
                        "res", "query", "sel", "dup", "y") if k in out]
    o = {k: out[k].realize().numpy() for k in keys}
    for bi, i in enumerate(sl):
        i = int(i)
        facs, q = decode({k: o[k][bi] for k in o})
        a_ = solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})
        n_ans += (a_ == rows[i]["solution"][rows[i]["query_var"]])
print(f"[BAR 4] displacement: bigtest {n_ans}/1500 "
      f"({'PASS' if n_ans >= 1194 else 'FAIL'} vs floor 1194)")
json.dump({"acceptance": rate, "div73": ok73, "div100": ok100,
           "canyon": [c_right, h_right, n_c], "bigtest": n_ans},
          open(".cache/fluency_bars.json", "w"))
print("[bars] banked -> .cache/crown_bars.json")
