"""gen23_split_read.py — B1's per-arm split (the countersigned rider),
self-contained (gen23_bars' parse_solve lives under its main guard and
closes over module state — the import lesson's second clause: guard the
body, EXPORT the tools; this copy is the tool export)."""
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

res = {}
for cls, seed in (("sign", 104100), ("wide", 104200), ("both", 104300)):
    arm = mint_arm(cls, 40, seed)
    ok = parse_solve(arm)
    res[cls] = {"ok": ok, "n": len(arm)}
    print(f"[split] {cls}: {ok}/{len(arm)} = {ok/len(arm):.1%}", flush=True)
json.dump(res, open(".cache/gen23_split.json", "w"), indent=1)
print("[saved] .cache/gen23_split.json", flush=True)
