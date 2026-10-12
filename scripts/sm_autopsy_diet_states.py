"""scripts/sm_autopsy_diet_states.py -- THE SM_241 AUTOPSY's diet-states collector (2026-10-09,
zero training, one read). SM_241 autopsy task 2 (THE REPRESENTABILITY RE-READ, docs/
phase1_skeleton_spec.md 2026-10-09 "SM_241 ... DOES NOT FIRE") needs direction_probe.py's own
per-breath CONTENT-dims diet states, but for SM_241's body, not PMS8_241's -- and welford_atlas.py
hardcodes DEV=CPU + FAMILY_ENVS["PMS8_241"] + the PMS8_241 ckpt/output paths for its own
(much larger) three-act build. This is the MINIMAL slice of welford_atlas.build()'s own forward
loop (lines ~166-249, copied, not reimplemented, down to the states_all tensor -- everything after
that line there is library/retina construction this task does not need): one two-pass forward
over the diet slice, per body, writing ONLY states_all (content dims, all K_B breaths) +
admissible, in EXACTLY direction_probe.py's expected shape/keys so DIET_STATES_NPZ can point at
this file with no reader change beyond the BODY parameterization.

Unlike welford_atlas.py, this script does NOT force DEV=CPU -- it inherits whatever family env the
caller's `env $FAM ...` supplies (clock_band_probe.py's own convention), so it can run on the idle
card under DEV=PCI+AMD (the word given: "reads may use the GPU ... under flock .cache/gpu.lock").

usage (family env on the command line, exactly rack_chain_<ARM>.sh's $FAM [+ $SURF8 [+ extra]]):
  env $FAM $SURF8 ALG_SELFMATCH=1 CKPT=.cache/sharp_SM_241.safetensors \\
      OUT=.cache/sm_autopsy_diet_states_SM_241.npz .venv/bin/python3 scripts/sm_autopsy_diet_states.py
"""
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import numpy as np

DIET_SLICE = ".cache/form_pm35c_slice1024_valid2.jsonl"
CKPT = os.environ.get("CKPT", ".cache/sharp_PMS8_241.safetensors")
OUT = os.environ.get("OUT", f".cache/sm_autopsy_diet_states_{os.path.basename(CKPT).replace('sharp_', '').replace('.safetensors', '')}.npz")
BATCH = 16


def main():
    t0 = time.time()
    os.environ.setdefault("ALG_TEST", DIET_SLICE)
    os.environ.setdefault("ALG_TEST_NAME", "pm35cslicevalid2")
    import phase1_algebra_head as H
    from phase1_algebra_head import build_params, forward, load_alg, build_slot_masks, alt2_fact_buf, K_VARS, L_FAC
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.custody_gold import row_gold

    bands, clock_dims = H._hier_band_dims()
    CONTENT = np.sort(np.concatenate(bands))
    assert len(np.intersect1d(CONTENT, clock_dims)) == 0
    C = len(CONTENT)
    K_B = int(os.environ["ALG_BREATH"])

    vs, vst, vtk, vg, vse = load_alg("test")
    n = len(vs)
    print(f"[diet-states] {CKPT}: diet slice {DIET_SLICE} n={n} C={C}/512 K_B={K_B} DEV={os.environ.get('DEV')}", flush=True)

    admissible = np.ones(n, dtype=bool)
    for i in range(n):
        try:
            row_gold(vs[i])
        except Exception:
            admissible[i] = False
    print(f"[diet-states] custody door: {int((~admissible).sum())}/{n} rows failed row_gold()", flush=True)

    p = build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    print(f"[diet-states] ckpt loaded ({time.time()-t0:.0f}s)", flush=True)

    states_all = np.zeros((n, K_B, L_FAC, C), np.float16)
    for s0 in range(0, n, BATCH):
        sl = np.arange(s0, min(s0 + BATCH, n))
        pad = BATCH - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
        tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
        se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, vse[sl_p].astype(np.int32))
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        _nv = np.array([vs[int(i)].get("n_vars", K_VARS) for i in sl_p])
        _ma = np.array([vs[int(i)].get("m", 0) for i in sl_p])
        fb = alt2_fact_buf(_oa, vse[sl_p].astype(np.int32), _nv, _ma)
        H._CENSUS = []
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float), fact_buf=Tensor(fb, dtype=dtypes.float))
        o["fat"].realize()
        got = {kb: arr for (kb, tag, arr) in H._CENSUS if tag == "state"}
        H._CENSUS = None
        assert set(range(K_B)) <= set(got), sorted(got)
        for kb in range(K_B):
            states_all[sl, kb] = got[kb][:len(sl), :L_FAC, :][:, :, CONTENT].astype(np.float16)
        if s0 % (BATCH * 8) == 0:
            print(f"[diet-states] forward {s0 + len(sl)}/{n} ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(".cache", exist_ok=True)
    np.savez_compressed(OUT, states_all=states_all, admissible=admissible)
    print(f"[diet-states] wrote {OUT} states_all {states_all.shape} admissible {admissible.shape} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
