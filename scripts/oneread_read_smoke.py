"""oneread_read_smoke.py -- OR_241's gate, THE READ SMOKE (8 wild rows, DEV=CPU). Fixed per
the coordinator's note (2026-10-13): no new trunk precompute -- reuses the already-precomputed
wildhold states (.cache/phase1_alg_states_wildhold.npz, via load_alg("test") under the usual
ALG_TEST=wild_admitted_holdout.jsonl / ALG_TEST_NAME=wildhold env) and just slices the first
ORS_N rows, exactly the technique scripts/picker/dn_read_c.py already uses (no separate .jsonl
fixture, no new .npz).

Env: the arm's own family env (SURF8 + role8 chassis + ALG_ONE_READ=1, ALG_TEST(_NAME)=wildhold)
+ ORS_CKPT (required). ORS_N (default 8)."""
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/picker")
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, K_VARS
from dn_common import load_gold, score_row_slots

CKPT = os.environ["ORS_CKPT"]
N = int(os.environ.get("ORS_N", "8"))

vs, vst, vtk, vg, vse = load_alg("test")
n = min(N, len(vs))
vg2 = load_gold(".cache/phase1_alg_states_wildhold.npz")

p = build_params(0)
sd = safe_load(CKPT)
assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

sl = np.arange(n)
nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl])
ma = np.array([vs[int(i)].get("m", 0) for i in sl])
se_np = vse[sl].astype(np.int32)
ts = Tensor(np.ascontiguousarray(vst[sl]), dtype=dtypes.half)
tk = Tensor(vtk[sl].astype(np.float32))
se = Tensor(se_np, dtype=dtypes.int)

o0 = forward(p, ts, tk, se)
onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
mk = build_slot_masks(onp0, se_np)
_ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
_oa = {**onp0, **{k: o0[k].numpy() for k in _ka}}
fb = alt2_fact_buf(_oa, se_np, nv, ma)
fact_t = Tensor(fb, dtype=dtypes.float)

o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t)
KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o else ())
final = {k: o[k].numpy() for k in KEYS}

n_ok = n_tot = 0
for bi, i in enumerate(sl):
    i = int(i)
    s = score_row_slots({k: final[k][bi] for k in KEYS}, vg2, i)
    for j, d in s.items():
        n_tot += 1
        n_ok += int(d["ok"])

print(f"[oneread-read-smoke] n_rows={n} fac-exact={n_ok/max(n_tot,1):.4f} ({n_ok}/{n_tot} slots)")
