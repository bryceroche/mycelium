"""five_prediction_reads.py — the table's missing measurements (2026-07-10):
invisibles per capita (gen-4 head, all domains) for the prevention verdict.
"""
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import build_params, forward, decode
from tta_alg2_dials import solve2
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

p = build_params(0)
sd = safe_load(".cache/phase1_algebra4_head.safetensors")
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

DOMS = {"bigtest": (".cache/algebra_nl_bigtest.jsonl",
                    ".cache/phase1_alg_states_bigtest.npz"),
        "alg2test": (".cache/algebra2_nl_test.jsonl",
                     ".cache/phase1_alg_states_alg2test.npz"),
        "alg4test": (".cache/algebra4_nl_test.jsonl",
                     ".cache/phase1_alg_states_alg4test.npz")}
print("PREVENTION READ — one-shot forced-wrong (invisibles) per capita, gen-4 head")
print("  (priors: legacy head on bigtest 90/1500=6.0%; gen-2 head on alg2test 12/800=1.5%)")
for name, (jp, sp) in DOMS.items():
    samples = [json.loads(l) for l in open(jp)]
    z = np.load(sp)
    st, tk, se = z["states"], z["tokmask"], z["sent"]
    n = len(samples)
    inv = forced = 0
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(st[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tk[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(se[sl_p].astype(np.int32), dtype=dtypes.int))
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query") + (("sel",) if "sel" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, i in enumerate(sl):
            i = int(i)
            facs, q = decode({k: o[k][bi] for k in o})
            a = solve2(facs, q, samples[i])
            if a is not None:
                forced += 1
                if a != samples[i]["solution"][samples[i]["query_var"]]:
                    inv += 1
    print(f"  {name:9s}: invisibles {inv}/{n} = {inv/n:.3%} | forced {forced} "
          f"| forced-precision {(forced-inv)/max(forced,1):.3f}")
