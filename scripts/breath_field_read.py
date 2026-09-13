"""THE DISH LINE READ (2026-09-12): per breath, per FIELD accuracy of the
decoded parse — which fields sharpen at which breath on a checkpoint (the
stages the machine already runs). CPU-friendly (eager forward, two-pass
read like loop_val, BF_N rows). usage: <family env> BF_CKPT=... ALG_TEST=...
BF_N=256 breath_field_read.py"""
import os, sys, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, L_FAC, K_VARS
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
vs, vst, vtk, vg, vse = load_alg("test")
N = min(int(os.environ.get("BF_N", "256")), len(vs)); rng = np.random.RandomState(0)
idx = np.sort(rng.choice(len(vs), N, replace=False))
p = build_params(0); sd = safe_load(os.environ["BF_CKPT"])
assert set(sd) == set(p), (sorted(set(sd) - set(p))[:3], sorted(set(p) - set(sd))[:3])
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
fields = ["pres", "ftype", "op", "args", "res", "dig"]
acc = None; cnt = None
for s0 in range(0, N, 8):
    sl = idx[s0:s0 + 8]; pad = 8 - len(sl); sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
    ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half); tk = Tensor(vtk[sl_p].astype(np.float32)); se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se); onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
    fact_t = None
    if int(os.environ.get("ALG_ALT2", "0")):
        from phase1_algebra_head import alt2_fact_buf
        _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p]); _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fact_t = Tensor(alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma), dtype=dtypes.float)
    o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fact_t)
    br = o["breaths"]; K = len(br)
    if acc is None: acc = np.zeros((K, len(fields))); cnt = np.zeros((K, len(fields)))
    for kb, ob in enumerate(br):
        onp = {k: ob[k].numpy() for k in ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in ob else ())}
        for bi, i in enumerate(sl):
            i = int(i)
            for j in range(L_FAC):
                if vg["presence"][i, j] < 0.5: continue
                isrel = vg["ftype"][i, j] == 0
                acc[kb, 0] += onp["pres"][bi, j] > 0; cnt[kb, 0] += 1
                acc[kb, 1] += int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j]; cnt[kb, 1] += 1
                acc[kb, 4] += int(onp["res"][bi, j].argmax()) == vg["res"][i, j]; cnt[kb, 4] += 1
                if isrel:
                    acc[kb, 2] += int(onp["op"][bi, j].argmax()) == vg["op"][i, j]; cnt[kb, 2] += 1
                    gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                    if len(gset) == 1 and "dup" in onp: ok = bool(onp["dup"][bi, j] > 0) and int(np.argmax(onp["args"][bi, j])) in gset
                    else: ok = set(np.argsort(-onp["args"][bi, j])[:2].tolist()) == gset
                    acc[kb, 3] += ok; cnt[kb, 3] += 1
                else:
                    acc[kb, 5] += bool((onp["dig"][bi, j].argmax(-1) == vg["digits"][i, j]).all()); cnt[kb, 5] += 1
name = os.path.basename(os.environ["BF_CKPT"]).replace("sharp_", "").replace(".safetensors", "")
print(f"[dish-line] {name} on {os.environ.get('ALG_TEST_NAME','?')} N={N}: per-breath field accuracy (rows = breath states 0..K-1)")
print("  breath  " + "  ".join(f"{f:>6s}" for f in fields))
for kb in range(acc.shape[0]):
    print(f"  b{kb:<6d} " + "  ".join(f"{acc[kb, f] / max(cnt[kb, f], 1):6.3f}" for f in range(len(fields))))
