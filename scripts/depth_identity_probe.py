"""depth_identity_probe.py -- THE LOOPED TRANSFORMER's identity proof (2026-10-09, the restated
bar for a road that REPLACES existing state: THE SORTING ROOM's own ledger entry, 2026-10-09
18:10). depth4 is NOT expected bit-identical to role8 at step 0 (the mixer is genuinely bypassed
-- a real architectural change, not a birth artifact); the identity instrument instead compares
TWO depth4 configs, BOTH with the mixer already bypassed (ALG_DEPTH=4 in both), that should
differ ONLY in whether the stack's own blocks run real compute:
  ON  (ALG_DEPTH_STUB=0): the real ReZero blocks run (real attention/FFN math on wq/wk/wv/xwq/
      xwk/xwv/ffn_w1, zero-init wo/xwo/ffn_w2 at birth)
  OFF (ALG_DEPTH_STUB=1): _depth_stack returns `cur` immediately, no compute at all
If the ReZero doors are truly exact zero at birth, ON and OFF must produce BIT-IDENTICAL forward()
outputs (every top-level tensor, every per-breath dict) -- the real compute mathematically
cancels to nothing, proving the mandatory-road law's AJAR form (a road that runs for real and
still contributes exactly zero until the gradient moves it) rather than a disguised bypass.

Usage (THREE separate processes -- ALG_DEPTH/ALG_DEPTH_STUB are read once at import):
  MODE=dump TAG=on  ALG_DEPTH=4 ALG_DEPTH_STUB=0 .venv/bin/python3 scripts/depth_identity_probe.py
  MODE=dump TAG=off ALG_DEPTH=4 ALG_DEPTH_STUB=1 .venv/bin/python3 scripts/depth_identity_probe.py
  MODE=diff                                     .venv/bin/python3 scripts/depth_identity_probe.py
Env (dump mode): GP_ROWS (default 0:8), SR_CKPT (optional warm checkpoint).
"""
import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np

MODE = os.environ.get("MODE", "dump")

if MODE == "dump":
    import phase1_algebra_head as H
    from phase1_algebra_head import build_params, forward, ident_build_array, T_ALG, load_alg
    from tinygrad import Tensor, dtypes

    assert int(os.environ.get("ALG_DEPTH", "0")) > 0, "ALG_DEPTH=4 required for this probe"
    TAG = os.environ["TAG"]
    OUT = f".cache/depth_identity_outputs_{TAG}.npz"

    vs, vst, vtk, vg, vse = load_alg("train")
    lo, hi = (int(x) for x in os.environ.get("GP_ROWS", "0:8").split(":"))
    sl = np.arange(lo, hi); B = len(sl)

    p = build_params(0)
    ckpt = os.environ.get("SR_CKPT", "")
    if ckpt:
        from tinygrad.nn.state import safe_load
        sd = safe_load(ckpt)
        for k in p:
            if k in sd and tuple(sd[k].shape) == tuple(p[k].shape):
                p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
        print(f"[depth-identity] TAG={TAG} warm from {ckpt}")
    else:
        print(f"[depth-identity] TAG={TAG} fresh init throughout")

    ts = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    _idt = Tensor(ident_build_array([vs[int(i)] for i in sl], T_ALG), dtype=dtypes.int) \
        if (H.ALG_BUSREG or H.ALG_IDKEY) else None

    o = forward(p, ts, tk, se, ident=_idt)
    dump = {}
    for k, v in o.items():
        if k == "breaths":
            continue
        if hasattr(v, "numpy"):
            dump[k] = v.numpy()
    if "breaths" in o:
        for bi, ob in enumerate(o["breaths"]):
            for k, v in ob.items():
                if hasattr(v, "numpy"):
                    dump[f"breath{bi}__{k}"] = v.numpy()
    np.savez(OUT, **dump)
    print(f"[depth-identity] TAG={TAG} ALG_DEPTH={H.ALG_DEPTH} ALG_DEPTH_STUB={H.ALG_DEPTH_STUB} "
          f"dumped {len(dump)} tensors -> {OUT}")

elif MODE == "diff":
    a = np.load(".cache/depth_identity_outputs_on.npz")
    b = np.load(".cache/depth_identity_outputs_off.npz")
    common = sorted(set(a.files) & set(b.files))
    only_a = sorted(set(a.files) - set(b.files))
    only_b = sorted(set(b.files) - set(a.files))
    maxd = 0.0; worst = None
    for k in common:
        d = float(np.abs(a[k].astype("float64") - b[k].astype("float64")).max()) if a[k].size else 0.0
        if d > maxd:
            maxd = d; worst = k
    print(f"[depth-identity] DIFF: maxdiff={maxd:.3e} over {len(common)}/{len(common)} common keys "
          f"(on {len(a.files)} keys, off {len(b.files)} keys; only-in-on={only_a}; only-in-off={only_b}; "
          f"worst key={worst})")
else:
    raise ValueError(f"unknown MODE={MODE!r}")
