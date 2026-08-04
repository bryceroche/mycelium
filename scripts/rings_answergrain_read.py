"""rings_answergrain_read.py — THE GRADUATION READ (2026-08-04; pins in
ledger BEFORE this ran: min-mass primary, AUC bars 0.75/0.60)."""
import os, sys, json
os.environ["ALG_BREATH"] = "3"; os.environ["ALG_RINGS"] = "1"
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
os.environ.setdefault("ALG_WIDE", "1")
os.environ.setdefault("ALG_TEST", ".cache/algebra_nl_bigtest.jsonl")
os.environ.setdefault("ALG_TEST_NAME", "bigtest")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from phase1_algebra_head import (build_params, forward, load_alg, decode,
                                 build_slot_masks, L_FAC)
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic

samples, states, tokmask, gold, sent = load_alg("test")
p = build_params(0)
sd = safe_load(".cache/g24_rings_rings.safetensors")
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
n = len(samples)
rows = []
for s0 in range(0, n, 8):
    sl = np.arange(s0, min(s0 + 8, n))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    t_tr = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)
    t_tk = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
    t_se = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, t_tr, t_tk, t_se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, sent[sl_p])
    o = forward(p, t_tr, t_tk, t_se, slot_mask=Tensor(mk, dtype=dtypes.float))
    keys = ("pres", "ftype", "op", "islit", "dig", "sgn", "args", "res",
            "query", "cmt_m") + (("sel",) if "sel" in o else ()) + \
           (("dup",) if "dup" in o else ())
    onp = {k: o[k].realize().numpy() for k in keys}
    for bi, i in enumerate(sl):
        i = int(i)
        smp = samples[i]
        facs, q = decode({k: onp[k][bi] for k in onp if k != "cmt_m"})
        pm = [float(onp["cmt_m"][bi, j]) for j in range(L_FAC)
              if float(onp["pres"][bi, j]) > 0]
        if not pm:
            continue
        ok = False
        try:
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
            giv = {f["var"]: f["value"] for f in cfacs if f["ftype"] == "given"}
            pr = problem_from_algebra3(len(used), cfacs, giv, int(smp["m"]))
            res = solve_symbolic(pr, budget=5000, seed=0)
            ok = (res["status"] == "solved" and isinstance(q, int) and
                  int(res["assignment"][cmp_[q]]) ==
                  smp["solution"][smp["query_var"]])
        except Exception:
            ok = False
        rows.append((min(pm), float(np.mean(pm)), bool(ok)))
mn = np.array([r[0] for r in rows]); me = np.array([r[1] for r in rows])
y = np.array([r[2] for r in rows])
from scipy.stats import mannwhitneyu
def auc(x):
    a, b = x[y], x[~y]
    if not len(b): return float("nan")
    u, pv = mannwhitneyu(a, b, alternative="greater")
    return u / (len(a) * len(b))
a_min, a_mean = auc(mn), auc(me)
print(f"[grad] rows n={len(rows)} correct {int(y.sum())} wrong {int((~y).sum())}")
print(f"[grad] AUC(min-mass)={a_min:.3f} PRIMARY | AUC(mean-mass)={a_mean:.3f}")
v = ("GRADUATES" if a_min >= 0.75 else
     "WEAK" if a_min >= 0.60 else "FAILS TO GRADUATE")
print(f"VERDICT (pinned): {v}")
json.dump({"n": len(rows), "auc_min": float(a_min), "auc_mean": float(a_mean),
           "verdict": v}, open(".cache/rings_answergrain.json", "w"), indent=1)
print("[saved] .cache/rings_answergrain.json")
