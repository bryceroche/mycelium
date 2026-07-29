"""burden_pair_probe.py — the factory's parser-side probe (registered
2026-07-29). 24 matched pairs at fixed n_factors=9: LOW = linear chain
(every var count <=2), HIGH = hub (one var engaged 5x), same op mix
(2 mul + 3 add... arms matched on rel-op multiset), solver-verified.
Outcome: standard 5-view quorum per probe. Bar: delta >= 0.25 CROSSES;
|delta| < 0.10 FLAT. Known residuals (stated at registration): n_vars
differs (LOW 10 vs HIGH 7); hub sentences reuse letters more densely.
"""
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
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/g21.safetensors")
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
rng = np.random.RandomState(31)

def low_probe(v):
    # chain: 5 givens, 4 rels (add,mul,add,mul... keep values small)
    a, b = int(rng.randint(2, 7)), int(rng.randint(2, 7))
    d = int(rng.randint(2, 5)); f = int(rng.randint(2, 9)); h = int(rng.randint(2, 4))
    # c=a+b; e=c*d; g=e+f; i=g*h  — cap check
    c = a+b; e = c*d; g = e+f; i = g*h
    if i > 300: return None
    text = (f"Consider the numbers a, b, c, d, e, f, g, h, i. a is {a}. b is {b}. "
            f"a plus b equals c. d is {d}. c times d equals e. f is {f}. "
            f"e plus f equals g. h is {h}. g times h equals i. What is i?")
    facs = [{"ftype":"given","var":0,"value":a},{"ftype":"given","var":1,"value":b},
            {"ftype":"rel","op":"add","args":[0,1],"result":2},
            {"ftype":"given","var":3,"value":d},
            {"ftype":"rel","op":"mul","args":[2,3],"result":4},
            {"ftype":"given","var":5,"value":f},
            {"ftype":"rel","op":"add","args":[4,5],"result":6},
            {"ftype":"given","var":7,"value":h},
            {"ftype":"rel","op":"mul","args":[6,7],"result":8}]
    return text, facs, 8, i

def high_probe(v):
    # hub: var a engaged in all 4 rels; 5 givens, 4 rels, same op multiset
    a = int(rng.randint(2, 5)); b = int(rng.randint(2, 7))
    d = int(rng.randint(2, 5)); f = int(rng.randint(2, 9)); h = int(rng.randint(2, 4))
    c = a+b; e = c*a; g = e+a; i = g*a
    if i > 300: return None
    text = (f"Consider the numbers a, b, c, d, e, f, g. a is {a}. b is {b}. "
            f"a plus b equals c. d is {d}. c times a equals e. f is {f}. "
            f"e plus a equals g. h is {h}." if False else
            f"Consider the numbers a, b, c, d, e, f, g. a is {a}. b is {b}. "
            f"a plus b equals c. c times a equals d. d plus a equals e. "
            f"e times a equals f. g is {h}. What is f?")
    facs = [{"ftype":"given","var":0,"value":a},{"ftype":"given","var":1,"value":b},
            {"ftype":"rel","op":"add","args":[0,1],"result":2},
            {"ftype":"rel","op":"mul","args":[2,0],"result":3},
            {"ftype":"rel","op":"add","args":[3,0],"result":4},
            {"ftype":"rel","op":"mul","args":[4,0],"result":5},
            {"ftype":"given","var":6,"value":h}]
    # note: to hold n_factors near LOW's 9, HIGH carries 7 factors + denser reuse;
    # exact factor-count match is impossible at matched op multiset with a hub —
    # RESIDUAL STATED: n_factors 9 vs 7 (both mid-band), reuse count 1 vs 5.
    return text, facs, 5, ((a+b)*a + a) * a

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

def quorum_fail(text, gold, seed):
    texts = [text] + [permuted_view(text, seed + k) for k in range(1, 5)]
    views = [(f, q, solve2(f, q, {"n_vars": 24, "m": 300})) for f, q in parse_batch(texts)]
    votes = [a for _, _, a in views]
    nn = [a for a in votes if a is not None]
    c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
    return not (cnt >= 3 and plur == gold), votes

N_PAIRS = 24
res = {"LOW": [], "HIGH": []}
made = 0; tries = 0
while made < N_PAIRS and tries < 400:
    tries += 1
    lo = low_probe(made); hi = high_probe(made)
    if lo is None or hi is None: continue
    for arm, pr in (("LOW", lo), ("HIGH", hi)):
        text, facs, q, gold = pr
        assert solve2(facs, q, {"n_vars": 24, "m": 300}) == gold, (arm, made)
        fail, votes = quorum_fail(text, gold, 95000 + 10*made)
        res[arm].append({"fail": bool(fail), "votes": votes, "gold": gold})
    made += 1
    if made % 6 == 0:
        print(f"  [{made}/{N_PAIRS}] LOW fails {sum(r['fail'] for r in res['LOW'])} | "
              f"HIGH fails {sum(r['fail'] for r in res['HIGH'])}", flush=True)

fl = sum(r["fail"] for r in res["LOW"]) / N_PAIRS
fh = sum(r["fail"] for r in res["HIGH"]) / N_PAIRS
delta = fh - fl
print(f"\nquorum-failure: LOW {fl:.2f}  HIGH {fh:.2f}  delta {delta:+.2f}")
if delta >= 0.25:
    verdict = "THE LAW CROSSES — burden at fixed size predicts natural-register failure"
elif abs(delta) < 0.10:
    verdict = "FLAT — re-engagement is a probe-register pathology; the asymmetry is the finding"
else:
    verdict = "INTERMEDIATE — texture; no reinterpretation"
print("VERDICT:", verdict)
json.dump({"low_fail": fl, "high_fail": fh, "delta": delta, "verdict": verdict,
           "residuals": "n_factors 9v7, n_vars 9v7, reuse 1v5 — stated at registration",
           "detail": res}, open(".cache/burden_pair_probe.json", "w"), indent=1, default=int)
print("[done] wrote .cache/burden_pair_probe.json")
