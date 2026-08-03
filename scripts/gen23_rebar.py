"""gen23_rebar.py — THE SIGN-ONLY RE-BAR (word (c), 2026-08-03; pins in
ledger BEFORE these numbers: sign >= old - 5pts, fresh seeds 105000/
105100, n=120/arm, validity floor old >= 60%)."""
import os, sys, json, signal
os.environ["ALG_WIDE"] = "1"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from gen23_bars import mint_arm
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic
from phase1_algebra_head import (T_ALG, build_params, forward, decode,
                                 sent_indices, TOKENIZER_JSON)
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
tok = Tokenizer.from_file(TOKENIZER_JSON)

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
            def _to(sig, frm):
                raise TimeoutError
            try:
                signal.signal(signal.SIGALRM, _to)
                signal.alarm(10)
                used = sorted({v for f in facs for v in
                               ([f.get("var")] if f.get("var") is not None else [])
                               + list(f.get("args", []))
                               + ([f.get("result")] if f.get("result") is not None else [])
                               if isinstance(v, int)} |
                              ({q} if isinstance(q, int) else set()))
                cmp_ = {v: i2 for i2, v in enumerate(used)}
                def _rm(f):
                    f = dict(f)
                    for kk in ("var", "result"):
                        if isinstance(f.get(kk), int): f[kk] = cmp_[f[kk]]
                    if isinstance(f.get("args"), list):
                        f["args"] = [cmp_[a] for a in f["args"]]
                    return f
                cfacs = [_rm(f) for f in facs]
                giv = {f["var"]: f["value"] for f in cfacs
                       if f["ftype"] == "given"}
                pr = problem_from_algebra3(len(used), cfacs, giv, 10**6,
                                           signed=True)
                res = solve_symbolic(pr, budget=5000, seed=0)
                if res["status"] == "solved" and isinstance(q, int) and \
                        int(res["assignment"][cmp_[q]]) == r["sol"][r["query"]]:
                    ok += 1
            except (Exception, TimeoutError):
                pass
            finally:
                signal.alarm(0)
    return ok

old = mint_arm("plain", 120, 105000)
sgn = mint_arm("sign", 120, 105100)
print(f"[rebar fixtures] old {len(old)} / sign {len(sgn)} (fresh held-out)", flush=True)
old_ok = parse_solve(old); sgn_ok = parse_solve(sgn)
r_o, r_s = old_ok/len(old), sgn_ok/len(sgn)
if r_o < 0.60:
    v = "INSTRUMENT-INVALID (validity floor)"
else:
    v = "PASS" if r_s >= r_o - 0.05 else "FAIL"
print(f"[B1-prime] old {old_ok}/120 = {r_o:.1%}  sign {sgn_ok}/120 = {r_s:.1%} -> {v}", flush=True)
json.dump({"old": old_ok, "sign": sgn_ok, "verdict": v},
          open(".cache/gen23_rebar.json", "w"), indent=1)
print("[saved] .cache/gen23_rebar.json", flush=True)
