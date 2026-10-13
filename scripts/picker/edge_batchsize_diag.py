"""edge_batchsize_diag.py -- scratch diagnostic (not part of the deliverable): is PLAIN forward()
(no snapshot/replay) itself batch-size sensitive on PMS8_241, or is the divergence found in
edge_read1_transfer.py's first cut specific to replay()/the snapshot path? Compares rows 0-7's
final decode computed as part of a B=64 call vs the SAME rows computed as a B=8 call.
"""
import os, sys, time
os.environ.setdefault("DEV", "CPU")
sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
import numpy as np
from adaptive_stop import FAMILY_ENVS
FAM = dict(FAMILY_ENVS["PMS8_241"]); FAM["DEV"] = "CPU"
for k, v in FAM.items():
    os.environ[k] = v
from dn_common import load_gold, score_row_slots
from dn_read_c import decode_fields_equal

def run(n):
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    import phase1_algebra_head as H
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, K_VARS
    p = build_params(0)
    sd = safe_load(".cache/sharp_PMS8_241.safetensors")
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    vs, vst_, vtk, vg, vse = load_alg("test")
    sl = np.arange(n)
    ts = Tensor(np.ascontiguousarray(vst_[sl]), dtype=dtypes.half)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    o0 = forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = build_slot_masks(onp0, se.numpy().astype(np.int32))
    slot_mask = Tensor(mk, dtype=dtypes.float)
    _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
    _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
    _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl])
    _ma = np.array([vs[int(i)].get("m", 0) for i in sl])
    fb = alt2_fact_buf(_oa, se.numpy().astype(np.int32), _nv, _ma)
    fact_t = Tensor(fb, dtype=dtypes.float)
    o = forward(p, ts, tk, se, slot_mask=slot_mask, fact_buf=fact_t)
    KEYS = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o else ())
    return {k: o[k].realize().numpy() for k in KEYS}, vg, vs


def main():
    t0 = time.time()
    big, vg, vs = run(64)
    small, _, _ = run(8)
    KEYS = list(small.keys())
    n_tot = n_diff = 0
    for i in range(8):
        a = {k: big[k][i] for k in KEYS}
        b = {k: small[k][i] for k in KEYS}
        s_a = score_row_slots(a, vg, i); s_b = score_row_slots(b, vg, i)
        for j in sorted(set(s_a) & set(s_b)):
            n_tot += 1
            if not decode_fields_equal(a, b, j):
                n_diff += 1
    print(f"[diag] plain forward(), B=64 rows0-7 vs B=8 rows0-7: {n_diff}/{n_tot} decode slots differ "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
