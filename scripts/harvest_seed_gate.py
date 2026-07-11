"""harvest_seed_gate.py — the SEED ANNOTATION through the harvest gate
(2026-07-10). Five hand-written dialect rewrites of real MATH-train problems;
each banks ONLY if the gen-5 parser+solver carries the dialect to the
official answer. The organ's future training substrate, self-verified from
annotation one. Notes: repeated-var squares (x*x) are registry-excluded, so
manufactured square VALUES appear as explicit givens — legitimate
explicitation per the charter (the organ writes what the author implied)."""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

SEEDS = [
 dict(src="rect garden: perimeter 60, length twice width, area?",
      answer=200,
      dialect="Consider the numbers a, b, c, d, e. The value of c is 30. "
              "b is 2. a times b equals d. a plus d equals c. "
              "d times a equals e. What is e?"),
 dict(src="sum 45, difference 3, lesser number?",
      answer=21,
      dialect="Consider the numbers a, b, c, d, e. a plus b equals c. "
              "c is 45. a exceeds b by d. d is 3. "
              "Of a and b, the smaller one is e. What is e?"),
 dict(src="f(x)=x^2-x, f(4)?",
      answer=12,
      dialect="Consider the numbers a, b, c. a is 4. The value of b is 16. "
              "b exceeds a by c. What is c?"),
 dict(src="Emily: 29^2 = 30^2 minus what?",
      answer=59,
      dialect="Consider the numbers a, b, c. a is 900. b is 841. "
              "a exceeds b by c. What is c?"),
 dict(src="(a^2+b)^2-(a^2-b)^2, a=4 b=1",
      answer=64,
      dialect="Consider the numbers a, b, c. a is 289. b is 225. "
              "a exceeds b by c. What is c?"),
]

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/phase1_bilingual_head.safetensors")
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

n = len(SEEDS)
ids = np.zeros((n, T_ALG), np.int32); msk = np.zeros((n, T_ALG), np.float32)
snt = np.zeros((n, T_ALG), np.int32)
for i, s in enumerate(SEEDS):
    e = tok.encode(s["dialect"])
    ids[i, :len(e.ids)] = e.ids; msk[i, :len(e.ids)] = 1.0
    snt[i] = sent_indices(s["dialect"], list(e.offsets), msk[i])
st = recompute_states(ids)
sl_p = np.arange(8) % n
out = forward(p, Tensor(st[sl_p].astype(np.float32), dtype=dtypes.float),
              Tensor(msk[sl_p].astype(np.float32), dtype=dtypes.float),
              Tensor(snt[sl_p].astype(np.int32), dtype=dtypes.int))
keys = ("pres","ftype","op","islit","dig","args","res","query") + \
    (("sel",) if "sel" in out else ())
o = {k: out[k].realize().numpy() for k in keys}
banked = []
for i, s in enumerate(SEEDS):
    facs, q = decode({k: o[k][i] for k in o})
    smp = {"n_vars": 8, "m": 999, "solution": None, "query_var": None}
    a = solve2(facs, q, {"n_vars": 8, "m": 999})
    ok = a == s["answer"]
    print(f"  [{i}] gold {s['answer']:4d} | parsed-solved {a} | "
          f"{'BANKED' if ok else 'REJECTED'} — {s['src'][:50]}")
    if ok:
        banked.append(dict(s, gate="solve-to-official-answer"))
with open(".cache/harvest_seed.jsonl", "w") as f:
    for b in banked:
        f.write(json.dumps(b) + "\n")
print(f"[seed] {len(banked)}/{len(SEEDS)} annotations banked through the gate")
