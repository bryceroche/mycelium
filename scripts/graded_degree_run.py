"""graded_degree_run.py — THE GRADED-DEGREE RUN (registered 2026-07-29).
Pilot top rung first (degree 5, n=12; proceed only in [0.2,0.8]).
Chassis: 5 rels fixed op-mix (add,mul,add,mul,add); rung-k substitutes
the hub var 'a' into the second operand of the first (k-1) post-r1
rels, removing that rung's fresh given. Engagements 2/3/4/5; n_factors
falls 11->8 with degree (size direction PRE-STATED: conservative).
Bar: monotone fail(2)<=..<=fail(5), total rise >=0.20, >=2 adjacent
separations >=0.08 = CROSSES-ON-DOSE; |rise|<0.10 = residuals; else
texture. Standard 5-view quorum; probes solver-verified.
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
rng = np.random.RandomState(41)

def make_probe(degree):
    """degree = total engagements of hub 'a' (given + arg-uses = 1 + (degree-1))."""
    A = 2
    b = int(rng.randint(2, 6)); d = int(rng.randint(2, 5))
    f = int(rng.randint(2, 9)); h = int(rng.randint(2, 4)); j = int(rng.randint(2, 9))
    hub_uses = degree - 1              # arg substitutions beyond r1
    # slots: r1 second operand is ALWAYS b (r1: a+b -> first engagement as arg is a itself)
    # post-r1 second operands, in order: [d(mul), f(add), h(mul), j(add)]
    ops = ["mul", "add", "mul", "add"]
    fresh = [d, f, h, j]
    subs = [i < (hub_uses - 1) for i in range(4)]   # r1 gives one arg-use; remaining spread here
    letters = ["a", "b"]; facs = [{"ftype":"given","var":0,"value":A},
                                  {"ftype":"given","var":1,"value":b}]
    sents = [f"a is {A}.", f"b is {b}."]
    sents.append("a plus b equals c."); letters.append("c")
    facs.append({"ftype":"rel","op":"add","args":[0,1],"result":2})
    cur = 2; val = A + b
    for i in range(4):
        if subs[i]:
            operand_var, operand_val, operand_letter = 0, A, "a"
        else:
            operand_letter = chr(ord(letters[-1]) + 1)
            letters.append(operand_letter)
            operand_var = len(letters) - 1
            operand_val = fresh[i]
            sents.append(f"{operand_letter} is {operand_val}.")
            facs.append({"ftype":"given","var":operand_var,"value":operand_val})
        res_letter = chr(ord(letters[-1]) + 1)
        letters.append(res_letter); res_var = len(letters) - 1
        verb = "times" if ops[i] == "mul" else "plus"
        sents.append(f"{letters[cur]} {verb} {operand_letter} equals {res_letter}.")
        facs.append({"ftype":"rel","op":ops[i],"args":[cur,operand_var],"result":res_var})
        val = val * operand_val if ops[i] == "mul" else val + operand_val
        cur = res_var
    if val > 300: return None
    text = f"Consider the numbers {', '.join(letters)}. " + " ".join(sents) + f" What is {letters[cur]}?"
    return text, facs, cur, val

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

def qfail(text, gold, seed):
    texts = [text] + [permuted_view(text, seed + k) for k in range(1, 5)]
    views = [(f, q, solve2(f, q, {"n_vars": 24, "m": 300})) for f, q in parse_batch(texts)]
    votes = [a for _, _, a in views]
    nn = [a for a in votes if a is not None]
    c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
    return not (cnt >= 3 and plur == gold)

def run_rung(degree, n, seed0):
    fails = 0; made = 0; tries = 0
    while made < n and tries < 200:
        tries += 1
        pr = make_probe(degree)
        if pr is None: continue
        text, facs, q, gold = pr
        assert solve2(facs, q, {"n_vars": 24, "m": 300}) == gold
        fails += qfail(text, gold, seed0 + 10*made)
        made += 1
    return fails / n

# PILOT: top rung first
pilot = run_rung(5, 12, 97000)
print(f"[PILOT] degree-5 fail rate: {pilot:.2f}  (proceed iff in [0.2, 0.8])")
if not (0.2 <= pilot <= 0.8):
    print("PILOT OUT OF BAND — THE RUN STOPS; ladder shift is a fresh registration, not an improvisation")
    json.dump({"pilot": pilot, "stopped": True}, open(".cache/graded_degree_run.json","w"), indent=1)
    sys.exit(0)

rates = {}
for deg in (2, 3, 4, 5):
    rates[deg] = run_rung(deg, 24, 98000 + 1000*deg)
    print(f"[rung degree-{deg}] fail rate: {rates[deg]:.2f}")
seq = [rates[d] for d in (2,3,4,5)]
mono = all(seq[i+1] >= seq[i] - 1e-9 for i in range(3))
rise = seq[-1] - seq[0]
seps = [seq[i+1]-seq[i] for i in range(3)]
big_seps = sum(1 for s in seps if s >= 0.08)
print(f"\nladder: {' -> '.join(f'{r:.2f}' for r in seq)}  rise {rise:+.2f}  monotone {mono}  seps>=0.08: {big_seps}")
if mono and rise >= 0.20 and big_seps >= 2:
    verdict = "CROSSES-ON-DOSE — burden is a mechanism at natural size; the seam rewrite is earned"
elif abs(rise) < 0.10:
    verdict = "FLAT — the +0.21 was the residuals talking"
else:
    verdict = "TEXTURE — partial/non-monotone; mechanism probe; no reinterpretation"
print("VERDICT:", verdict)
json.dump({"pilot": pilot, "rates": {str(k): v for k, v in rates.items()},
           "monotone": bool(mono), "rise": rise, "big_seps": big_seps, "verdict": verdict},
          open(".cache/graded_degree_run.json", "w"), indent=1)
print("[done] wrote .cache/graded_degree_run.json")
