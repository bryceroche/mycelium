"""alg4_drift_autopsy.py — the drift mechanism's autopsy (2026-07-24):
stable-loss vs flicker across the annealed family — lesion or churn-with-drift?

Per-row answer diff on alg4test between gen-16 (crown_reader_v4) and
the gen-17 family (g17_armR): which rows did the family lose, which
did it gain, and what shapes are they? Reuses the banked alg4test
states; single-view, the battery's own grain.
"""
import json, sys, os, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_DUP", "1")
os.environ["ALG_TEST"] = ".cache/algebra4_nl_test.jsonl"
os.environ["ALG_TEST_NAME"] = "alg4test"
from phase1_algebra_head import build_params, forward, decode, load_alg
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

CKPTS = {"v4": ".cache/crown_reader_v4.safetensors",
         "armR": ".cache/g17_armR.safetensors",
         "s3000": ".cache/g17c_s3000.safetensors",
         "s3500": ".cache/g17c_s3500.safetensors",
         "s4000": ".cache/g17c_s4000.safetensors"}

samples, states, tokmask, gold, sent = load_alg("test")
n = len(samples)
print(f"[autopsy] alg4test n={n}")


def eval_ckpt(ckpt):
    p = build_params(0)
    sd = safe_load(ckpt)
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    ok = np.zeros(n, bool)
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int))
        keys = [k for k in ("pres", "ftype", "op", "islit", "dig", "dig2", "args",
                            "res", "query", "sel", "dup", "y") if k in out]
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, ri in enumerate(sl):
            facs, q = decode({k: o[k][bi] for k in o})
            smp = samples[ri]
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}

            def fvars(f):
                if f["ftype"] in ("rel", "sel", "pct"):
                    return list(f["args"]) + ([f["result"]] if "result" in f else [])
                if f["ftype"] in ("mod", "fdiv"):
                    return [f["var"], f["result"]]
                return [f["var"]]
            try:
                nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in fvars(f)])
                res = solve_symbolic(problem_from_algebra3(nv, facs, gv, smp["m"]),
                                     budget=200_000, seed=0)
                if res["status"] == "solved":
                    sol = [int(res["assignment"][v]) for v in range(nv)]
                    ga = smp["solution"][smp["query_var"]]
                    if q < len(sol) and sol[q] == ga:
                        ok[ri] = True
            except Exception:
                pass
    return ok


res = {name: eval_ckpt(ck) for name, ck in CKPTS.items()}
for name, ok in res.items():
    print(f"[drift] {name}: {int(ok.sum())}/{n}")
fam = ["armR", "s3000", "s3500", "s4000"]
stable_lost = np.where(res["v4"] & ~res["armR"] & ~res["s3000"]
                       & ~res["s3500"] & ~res["s4000"])[0]
flicker = np.where(res["v4"] & (~res["armR"] | ~res["s3000"] | ~res["s3500"]
                                | ~res["s4000"])
                   & ~(np.isin(np.arange(n), stable_lost)))[0]
stable_gain = np.where(~res["v4"] & res["armR"] & res["s3000"]
                       & res["s3500"] & res["s4000"])[0]
print(f"[drift] STABLE LOSSES (v4-right, wrong in ALL 4 family ckpts): {len(stable_lost)}")
print(f"[drift] FLICKER (v4-right, wrong in SOME family ckpts): {len(flicker)}")
print(f"[drift] STABLE GAINS: {len(stable_gain)}")
lost = stable_lost
gained = stable_gain


def shape(ri):
    s = samples[ri]
    kinds = Counter(f.get("op", f["ftype"]) if f["ftype"] == "rel" else f["ftype"]
                    for f in s["factors"])
    vals = [f["value"] for f in s["factors"] if f["ftype"] == "given"]
    return dict(i=int(ri), n_vars=s["n_vars"], n_fac=len(s["factors"]),
                kinds=dict(kinds), max_given=max(vals) if vals else 0,
                text=s["text"][:110])


print("\n[drift] STABLE-LOSS ROW SHAPES:")
kind_tally = Counter()
for ri in lost:
    sh = shape(ri)
    for k, c in sh["kinds"].items():
        kind_tally[k] += c
    print(" ", json.dumps(sh))
print(f"\n[autopsy] lost-row factor-kind tally: {dict(kind_tally)}")
print("[autopsy] GAINED ROW SHAPES (first 6):")
for ri in gained[:6]:
    print(" ", json.dumps(shape(ri)))
json.dump(dict(stable_lost=[shape(int(r)) for r in lost],
               n_flicker=len(flicker),
               gained=[shape(int(r)) for r in gained],
               totals={k: int(v.sum()) for k, v in res.items()}),
          open(".cache/alg4_drift_autopsy.json", "w"), indent=1)
print("[drift] banked -> .cache/alg4_drift_autopsy.json")
