"""depth_curve.py — THE DEPTH CURVE (2026-09-01, the crossover's read).
fac-exact (loop-engaged, two-pass masked) bucketed by ladder depth on the
deepval fixture. Env: DC_CKPT + mode envs (ALG_BREATH for the K arm;
ALG_MASKRE/ALG_ALTMASK/etc. for organ arms). ALG_TEST must point at
.cache/deepval.jsonl (ALG_TEST_NAME=deepval).
PINNED (the subsumption law made falsifiable): K=7 heads collapse near
depth ~7; a K=8 head's knee sits right of a K=7 head's.
"""
import os, sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from phase1_algebra_head import (build_params, forward, load_alg,
                                 build_slot_masks, L_FAC)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

rows_meta = [json.loads(l) for l in open('.cache/deepval.jsonl')]
vs, vst, vtk, vg, vse = load_alg("test")
assert len(vs) == len(rows_meta), (len(vs), len(rows_meta))
p = build_params(0)
sd = safe_load(os.environ["DC_CKPT"])
for k in p:
    if k in sd:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
_fresh = sorted(set(p) - set(sd))
if _fresh: print(f"[dc] fresh-init: {_fresh}")
ok_b, tot_b = {}, {}
for s0 in range(0, len(vs), 8):
    sl = np.arange(s0, min(s0 + 8, len(vs)))
    pad = 8 - len(sl)
    sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
    onp = {k: o[k].realize().numpy() for k in
           (("pres", "ftype", "op", "islit", "dig", "args", "res")
            + (("dup",) if "h_dup" in p else ()))}
    for bi, i in enumerate(sl):
        i = int(i)
        bkt = rows_meta[i]["gen"]["bucket"]
        for j in range(L_FAC):
            if vg["presence"][i, j] < 0.5:
                continue
            tot_b[bkt] = tot_b.get(bkt, 0) + 1
            ok = (onp["pres"][bi, j] > 0)
            ok &= int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j]
            ok &= int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
            if vg["ftype"][i, j] == 0:
                ok &= int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                if len(gset) == 1 and "dup" in onp:
                    ok &= bool(onp["dup"][bi, j] > 0)
                    ok &= int(np.argmax(onp["args"][bi, j])) in gset
                else:
                    top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                    ok &= top2 == gset
            else:
                ok &= bool((onp["dig"][bi, j].argmax(-1) ==
                            vg["digits"][i, j]).all())
            ok_b[bkt] = ok_b.get(bkt, 0) + int(ok)
name = os.environ["DC_CKPT"].split("/")[-1]
K = os.environ.get("ALG_BREATH", "7")
line = " ".join(f"d{b}:{ok_b[b]/tot_b[b]:.3f}" for b in sorted(tot_b))
print(f"[depth-curve] {name} K={K} {line}")
