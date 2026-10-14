"""oneread_decode_change.py -- THE DECODE-CHANGE-AFTER-BREATH-2 COUNT (OR_241's own chain,
2026-10-12 "THE ENGINE IS MUDDY"): scripts/decode_change_breath.py (branch breathstop) does
not exist on gen-weights/oneread yet (breathstop has not merged forward) -- this is the
"your own count from the per-breath heads" fallback the mission brief names.

Per wild row, present gold slot: compares the ARGMAX-LEVEL decode (ftype/op/res/args/dig,
the SAME decode_fields_equal convention scripts/picker/dn_read_c.py and this arm's own gate
use -- never raw float array_equal, per docs/replay_gym.md §3) at breath 2 vs the final
breath (K_B-1), using forward()'s own `out["heads_all"]` list (always populated,
unconditionally, by _forward_tail -- no snapshot/replay machinery needed here, just a plain
read). Reports the count and fraction of (row, slot) pairs whose decode differs.

Env: the arm's own family env (SURF8 + role8 chassis + ALG_ONE_READ=1) + ODCNG_CKPT
(required) + ALG_TEST/ALG_TEST_NAME (defaults to wildhold). DEV is whatever the caller sets
(CPU for a smoke, the real device for the chain's own post-training read)."""
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
import phase1_algebra_head as H
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, K_VARS

CKPT = os.environ["ODCNG_CKPT"]
BATCH = int(os.environ.get("ODCNG_BATCH", "8"))
N_ROWS = int(os.environ.get("ODCNG_N", "0")) or None

vs, vst, vtk, vg, vse = load_alg("test")
n = N_ROWS or len(vs)
n = min(n, len(vs))
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())


def decode_fields_equal(a, b, j):
    eq = True
    eq &= int(np.argmax(a["ftype"][j])) == int(np.argmax(b["ftype"][j]))
    eq &= int(np.argmax(a["res"][j])) == int(np.argmax(b["res"][j]))
    eq &= (a["pres"][j] > 0) == (b["pres"][j] > 0)
    eq &= int(np.argmax(a["op"][j])) == int(np.argmax(b["op"][j]))
    eq &= tuple(np.argsort(-a["args"][j])[:2].tolist()) == tuple(np.argsort(-b["args"][j])[:2].tolist())
    eq &= bool((np.argmax(a["dig"][j], axis=-1) == np.argmax(b["dig"][j], axis=-1)).all())
    return eq


n_total = 0
n_diff = 0
n_breaths = None
for s0 in range(0, n, BATCH):
    sl = np.arange(s0, min(s0 + BATCH, n))
    pad = BATCH - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    B = len(sl_p)
    nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
    ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
    se_np = vse[sl_p].astype(np.int32)
    ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
    tk = Tensor(vtk[sl_p].astype(np.float32))
    se = Tensor(se_np, dtype=dtypes.int)

    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, se_np)
    _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
    _oa = {**onp0, **{k: o0[k].numpy() for k in _ka}}
    fb = alt2_fact_buf(_oa, se_np, nv, ma)
    fact_t = Tensor(fb, dtype=dtypes.float)

    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t)
    heads_all = o["heads_all"]
    n_breaths = len(heads_all)
    h2 = {k: heads_all[2][k].numpy() for k in KEYS}
    hF = {k: heads_all[-1][k].numpy() for k in KEYS}
    for bi, i in enumerate(sl):
        i = int(i)
        for j in range(h2["pres"].shape[1]):
            if vg["presence"][i, j] < 0.5:
                continue
            n_total += 1
            if not decode_fields_equal({k: h2[k][bi] for k in KEYS}, {k: hF[k][bi] for k in KEYS}, j):
                n_diff += 1
    print(f"[decode-change] batch {s0}-{s0+len(sl)-1} done", flush=True)

print(f"[decode-change] K_B={n_breaths} breath2->breathF decode changes: {n_diff}/{n_total} "
      f"({n_diff/max(n_total,1):.4f}) on {os.environ.get('ALG_TEST_NAME','?')}")
