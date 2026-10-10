"""scripts/dirtb_rawdump.py -- THE DIRECTION TIE-BREAK's solve-free raw per-slot logit dump
(2026-10-09). chain_acc.py's CA_RAWDUMP collects the exact same per-slot args/res/... logits but
ALSO solves every row with the CSP core afterward (slow, and unnecessary when only the logits are
wanted -- the tie-break's DIET read is ~775 rows and needs no solve at all). Same forward pass
(pass 0: slot_mask + fact_buf; pass 1: the final decode, mirroring chain_acc.py's non-ALT3,
non-HUD/tree/ident/dircue path exactly -- PMS8_241 and PC_241 use none of those ports), same KEYS,
no multiprocessing solve step, no CA_MASK (res/args/ftype/op/pres are IDENTICAL under mask 0 vs 1;
only "dig" differs, and this script does not apply the numeral mask at all -- raw "dig" logits only).

env: family env (as any rack_chain_*.sh's $FAM [+ $SURF8 [+ extra, e.g. $PCX]]) + ALG_TEST(_NAME)
+ CA_CKPT + OUT (output pickle path).
usage: env $FAM $SURF8 [$PCX] ALG_TEST=... ALG_TEST_NAME=... CA_CKPT=... OUT=... \\
         .venv/bin/python3 scripts/dirtb_rawdump.py
"""
import os
import pickle
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np


def main():
    t0 = time.time()
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, K_VARS
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load

    vs, vst, vtk, vg, vse = load_alg("test")
    n = len(vs)
    p = build_params(0)
    sd = safe_load(os.environ["CA_CKPT"])
    assert set(sd) == set(p), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "h_dup" in p else ())
    print(f"[dirtb-rawdump] {os.environ['CA_CKPT']} on {os.environ.get('ALG_TEST_NAME')} n={n} "
          f"keys={KEYS} ({time.time()-t0:.0f}s)", flush=True)

    out = []
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
        tk = Tensor(vtk[sl_p].astype(np.float32))
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, se.numpy())
        _oa = {**onp0, **{k: o0[k].numpy() for k in ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())}}
        nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb_t = Tensor(alt2_fact_buf(_oa, se.numpy(), nv, ma), dtype=dtypes.float)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=fb_t)
        onp = {k: o[k].numpy() for k in KEYS}
        for bi, i in enumerate(sl):
            i = int(i)
            out.append({"i": i, "text": vs[i]["text"], **{k: onp[k][bi].astype(np.float32) for k in KEYS}})
        if s0 % 64 == 0:
            print(f"[dirtb-rawdump] {s0 + len(sl)}/{n} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(".cache", exist_ok=True)
    pickle.dump(out, open(os.environ["OUT"], "wb"))
    print(f"[dirtb-rawdump] wrote {os.environ['OUT']} ({len(out)} rows, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
