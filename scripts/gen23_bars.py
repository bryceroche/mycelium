"""gen23_bars.py — THE GEN-23 BARS (2026-08-03; both pinned before
their numbers: B1 = new-range arm >= old-range arm − 5pts (matched
shapes/dose, held-out seed 104000, n=120/arm, answer==gold); B2 =
bigtest(g23) >= 1214 (gen-22's 1226 − 1%, pinned pre-measurement).
B3 = no conversion bar exists."""
import os, sys, json, random, subprocess, signal
os.environ["ALG_WIDE"] = "1"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from answer_space_mint import gen_row
from algebra2_nl_gen import render2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic
from phase1_algebra_head import (T_ALG, build_params, forward, decode,
                                 sent_indices, TOKENIZER_JSON)
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok = Tokenizer.from_file(TOKENIZER_JSON)

# ---- fixture arms (held-out seed; same generator, matched shapes) ----
def mint_arm(cls, n, seed):
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        g = gen_row(rng, cls)
        if g is None:
            continue
        n_vars, facs, sol, query = g
        if n_vars > 24 or len(facs) > 24:
            continue
        text, gfactors, mentions, _ = render2(rng, n_vars, facs, query)
        givens = {f["var"]: f["value"] for f in gfactors if f["ftype"] == "given"}
        pr = problem_from_algebra3(n_vars, gfactors, givens, 10**6, signed=True)
        res = solve_symbolic(pr, budget=5000, seed=0)
        if res["status"] != "solved" or \
                [int(res["assignment"][v]) for v in range(n_vars)] != sol:
            continue
        rows.append({"text": text, "sol": sol, "query": query,
                     "n_vars": n_vars})
    return rows

old_arm = mint_arm("plain", 120, 104000)
new_arm = (mint_arm("sign", 40, 104100) + mint_arm("wide", 40, 104200)
           + mint_arm("both", 40, 104300))
print(f"[fixtures] old {len(old_arm)} / new {len(new_arm)} (held-out seeds)")

p = build_params(0)
sd = safe_load(".cache/g23.safetensors")
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def parse_solve(rows):
    ok = 0
    for s0 in range(0, len(rows), 8):
        batch = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for i, r in enumerate(batch):
            e = tok.encode(r["text"]); Ln = min(len(e.ids), T_ALG)
            ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
            snt[i] = sent_indices(r["text"], list(e.offsets), msk[i])
        st = recompute_states(ids)
        out = forward(p, Tensor(st.astype(np.float32), dtype=dtypes.float),
                      Tensor(msk, dtype=dtypes.float),
                      Tensor(snt, dtype=dtypes.int))
        keys = ("pres", "ftype", "op", "islit", "dig", "sgn", "args", "res",
                "query") + (("sel",) if "sel" in out else ()) + \
               (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for i, r in enumerate(batch):
            facs, q = decode({k: o[k][i] for k in o})
            # LANDMINE #3 GUARD: garbage parses + wide domains make each
            # DECISION expensive — budget counts decisions, not work.
            # Wall-clock cap per row; timeout counts as WRONG (an answer
            # the chain can't produce in bounded time is not an answer).
            def _to(sig, frm):
                raise TimeoutError
            try:
                signal.signal(signal.SIGALRM, _to)
                signal.alarm(10)
                giv = {f["var"]: f["value"] for f in facs
                       if f["ftype"] == "given"}
                pr = problem_from_algebra3(24, facs, giv, 10**6, signed=True)
                res = solve_symbolic(pr, budget=5000, seed=0)
                if res["status"] == "solved" and q is not None and \
                        int(res["assignment"][q]) == r["sol"][r["query"]]:
                    ok += 1
            except (Exception, TimeoutError):
                pass
            finally:
                signal.alarm(0)
    return ok

old_ok = parse_solve(old_arm)
new_ok = parse_solve(new_arm)
r_old, r_new = old_ok / len(old_arm), new_ok / len(new_arm)
b1 = r_new >= r_old - 0.05
print(f"[B1] old-range {old_ok}/{len(old_arm)} = {r_old:.1%}  "
      f"new-range {new_ok}/{len(new_arm)} = {r_new:.1%}  "
      f"-> {'PASS' if b1 else 'FAIL'} (bar: new >= old - 5pts)")

# ---- B2: bigtest under g23 ----
env = dict(os.environ)
env.update({"ALG_CKPT": ".cache/g23.safetensors",
            "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
            "ALG_TEST_NAME": "bigtest", "DEV": "AMD"})
r = subprocess.run([".venv/bin/python3", "scripts/phase1_algebra_head.py",
                    "--eval"], env=env, capture_output=True, text=True)
tail = r.stdout.strip().splitlines()[-6:]
print("[B2] bigtest eval tail:")
for l in tail:
    print("   ", l)
json.dump({"b1": {"old": old_ok, "new": new_ok, "pass": bool(b1)},
           "b2_tail": tail},
          open(".cache/gen23_bars.json", "w"), indent=1)
print("[saved] .cache/gen23_bars.json")
