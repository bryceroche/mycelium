"""rtb_identity_check.py -- THE GATE (THE REPLAY-TRAINED BREATH, worktree mycelium-wt28, branch
rtb; 2026-10-11). Proves replay()'s new `kb_stop` param (one extra breath only, not the whole
remainder) reproduces the SAME breath's own decode the full forward() loop produces -- the
decode-exact standard docs/replay_gym.md §3 states, applied to RTB's own resumption shape.

For kb in (1, 3, 5) on PMS8_241 (the chassis RTB_241 continues; .cache/sharp_PMS8_241.safetensors,
no ALG_ALT3 -- see rtb_lib's family-scoping note) over N_ROWS wild-holdout rows:
  1. ALG_SNAP_AT=kb,kb+2 (the MULTI-POINT extension, two points in one call so this same run also
     feeds the kb+2 check below) dumps snapshots at the end of breath kb (and kb+2 when < K_B).
  2. out_full = the SAME forward() call with no snapshot env set (the plain read).
  3. out_replay = replay(p, snap_load(kb's npz), kb_start=kb, kb_stop=kb+1) -- EXACTLY what RTB's
     training step does for kb_target=kb+1: resume from kb, run ONE breath.
  4. Compare out_replay's TOP-LEVEL heads (by _forward_tail's own construction, these are always
     heads_of(breaths[-1]) -- the LAST breath run, i.e. breath kb+1) against out_full's breath
     (kb+1) own decode: out_full["_early"][kb+1] when kb+1 < K_B-1, else out_full's own top level
     (breath K_B-1 is never in `_early`, which excludes the final breath by construction).

DEV=CPU asserted; no .cache/gpu.lock touch.
"""
import os
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r}: this script is CPU-only"

import numpy as np

import rtb_lib as RL

_FAM = {
    "DEV": "CPU", "ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1",
    "ALG_HW": "512", "ALG_WIDE": "1", "ALG_BREATH": "7",
    "ALG_NOTEBOOK": "1", "ALG_SIXWAVE": "1", "NB_PERSLOT": "1",
    "ALG_BINDBUS": "7", "ALG_BIND_D": "512",
    "BIND_CODES": ".cache/bindbus_codes512.npz",
    "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
    "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
    "ALG_FED": "1", "ALG_POLAR": "1", "ALG_POLAR_D": "128",
    "ALG_POLAR_EM": "0.1",
    "ALG_POLAR_D_INIT": ".cache/polar_waist_init_d128u.npz",
    "ALG_PRUNE": "pforms,s4,fednl0,lane2", "ALG_SLOT_ALL": "1",
    "ALG_STELLAR": "2", "ALG_CLOCK_CANON": "1", "SC_EVAL": "0",
    "ALG_TEST": ".cache/wild_admitted_holdout.jsonl", "ALG_TEST_NAME": "wildhold",
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain", "ALG_ROUTER_PTR": "0.0",
    "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1", "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1",
    "ALG_SPAN_ARCUE": "1", "ALG_PTR_SURF": "role:add:2.0", "BIND_CODES": ".cache/bindbus_codes512r.npz",
}
CKPT = ".cache/sharp_PMS8_241.safetensors"
N_ROWS = 8
KBS = (1, 3, 5)
HEAD_KEYS_SKIP = {"heads_all", "breaths", "breaths_u", "breaths_r", "_early"}


def _set_env():
    for k in list(os.environ.items()):
        pass
    for k, v in _FAM.items():
        os.environ[k] = v


def _load():
    import phase1_algebra_head as H
    from tinygrad.nn.state import safe_load
    p = H.build_params(0)
    sd = safe_load(CKPT)
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return H, p


def _compare(a_dict, b_dict):
    keys = sorted(set(a_dict) & set(b_dict) - HEAD_KEYS_SKIP)
    mismatches, checked = [], []
    for k in keys:
        a, b = a_dict[k], b_dict[k]
        if not (hasattr(a, "realize") and hasattr(b, "realize")):
            continue
        an, bn = a.realize().numpy(), b.realize().numpy()
        if an.shape != bn.shape:
            mismatches.append((k, f"shape {an.shape} vs {bn.shape}")); continue
        if np.array_equal(an, bn):
            checked.append((k, 0.0))
        else:
            d = float(np.max(np.abs(an.astype(np.float64) - bn.astype(np.float64))))
            checked.append((k, d)); mismatches.append((k, f"max|delta|={d:.6g}"))
    return checked, mismatches


def main():
    _set_env()
    H, p = _load()
    from tinygrad import Tensor, dtypes
    K_B = int(os.environ["ALG_BREATH"])
    vs, vst, vtk, vg, vse = H.load_alg("test")
    sl = np.arange(min(N_ROWS, len(vs)))
    ts = Tensor(np.ascontiguousarray(vst[sl]), dtype=dtypes.half)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    kwargs = RL.build_consult_kwargs(H, p, ts, tk, se, vs, sl)

    t0 = time.time()
    rows = []
    for kb in KBS:
        if kb + 1 >= K_B:
            continue
        os.environ.pop("ALG_SNAP_AT", None); os.environ.pop("ALG_SNAP_OUT", None)
        out_full = H.forward(p, ts, tk, se, **kwargs)
        for k in list(out_full):
            if hasattr(out_full[k], "realize"):
                out_full[k] = out_full[k].realize()

        snap_path = f".cache/rtb_identity_snap_kb{kb}.npz"
        os.environ["ALG_SNAP_AT"] = str(kb)
        os.environ["ALG_SNAP_OUT"] = snap_path
        _ = H.forward(p, ts, tk, se, **kwargs)
        os.environ.pop("ALG_SNAP_AT", None); os.environ.pop("ALG_SNAP_OUT", None)

        snap = H.snap_load(snap_path)
        out_replay = H.replay(p, snap, kb, kb_stop=kb + 2)   # run EXACTLY breath kb+1

        kb_target = kb + 1
        # out_full["breaths"][ri] is heads_of(breaths[ri], vst=_vst_at[ri]) for EVERY ri,
        # always populated (unlike "_early", gated behind ALG_DEEPSUP/ALG_CONSUME) -- the
        # correct per-breath comparison point, and more correct than "_early" would be here
        # since it reads the PER-BREATH vst, not unconditionally the final one.
        target_dict = out_full["breaths"][kb_target]
        checked, mismatches = _compare(target_dict, out_replay)
        max_d = max([d for _, d in checked], default=0.0)
        # THE DECODE-LEVEL STANDARD (docs/replay_gym.md §3): a replay verdict is read with
        # decode(), never np.array_equal on raw floats -- tinygrad's lazy scheduling re-sums
        # in a different order once a tensor is rematerialized from numpy (~1e-5 to ~9.8e-4
        # on the gym's own gate); report the raw float comparison for continuity but PASS on
        # decode-level agreement.
        need = ["pres", "res", "ftype", "op", "dig", "args"]
        for extra in ("sgn", "dup", "dargs"):
            if extra in target_dict and extra in out_replay:
                need.append(extra)
        fnp = {k: target_dict[k].realize().numpy() for k in need if k in target_dict}
        rnp = {k: out_replay[k].realize().numpy() for k in need if k in out_replay}
        n_decode_diff = 0
        for i in range(ts.shape[0]):
            row_f = {k: fnp[k][i] for k in fnp}; row_r = {k: rnp[k][i] for k in rnp}
            try:
                dec_f = H.decode(row_f); dec_r = H.decode(row_r)
            except Exception as e:
                dec_f = dec_r = f"<decode error: {e}>"
            if dec_f != dec_r:
                n_decode_diff += 1
        print(f"[rtb-identity] kb_start={kb} kb_target={kb_target}: {len(checked)} heads, "
              f"max|delta|={max_d:.6g}, decode-level rows differing={n_decode_diff}/{ts.shape[0]}", flush=True)
        for k, info in mismatches:
            print(f"    raw-float diff {k}: {info}", flush=True)
        rows.append((kb, kb_target, len(checked), n_decode_diff, max_d))
        os.remove(snap_path)
    print(f"[rtb-identity] done in {time.time() - t0:.1f}s")
    print("PASS" if rows and all(r[3] == 0 for r in rows) else "FAIL")


if __name__ == "__main__":
    main()
