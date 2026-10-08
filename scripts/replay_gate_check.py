"""replay_gate_check.py -- THE GATE, item (3) of THE BUILD (2026-10-07, worktree mycelium-wt8,
branch replay; docs/phase1_skeleton_spec.md 2026-10-07 17:27 "THE SAVE POINT").

On 24 wild rows (.cache/wild_admitted_holdout.jsonl) with two bodies:
  PMS8_241  -- SURF8 only (role pointer bank), no consults (ALG_ALT3 unset): a single forward()
              call is already "the final read-out".
  RK_241    -- SURF8 + hierd (ALG_HIER_READ/WAIST/DAMP/TAU) + ALG_ALT3=1 ALG_CERT=2.0 + THE RACK
              (ALG_RACK=1 ALG_RACK_TESTS=given_unique ALG_RACK_FREEZE=leaf): THE THREE CONSULTS
              fire inside the loop, so the TRUE final `out` is membrane_rack.py's own 3-pass
              cycle's third call (facts3/facts5/cert3/cert5/rack3/rack5 all threaded as top-level
              forward() params) -- the same cycle this script reuses by import.

For kb in (1, 3, 5): run the body's final-pass forward() call with ALG_SNAP_AT=kb /
ALG_SNAP_OUT=<tmp npz>, snapshot-dump at the end of breath kb; separately run the SAME final-pass
call with ALG_SNAP_AT unset to get out_full; call replay(p, snap_load(path), kb) to get
out_replay; np.array_equal every realized head in out_full vs out_replay (skip "bank", "breaths",
"heads_all" -- live closures / diagnostic ladders out of this gate's stated scope, see
replay_harness_report.md -- wait, no report files; see the final response).

DEV=CPU is asserted; .cache/gpu.lock is never touched (no flock anywhere in this file).
"""
import os
import sys
import time

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
assert os.environ.get("DEV") == "CPU", f"DEV={os.environ.get('DEV')!r}: this script is CPU-only by brief"

import numpy as np

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
}
SURF8 = {
    "ALG_ROUTER": "2", "R_GAIN_INIT": "1.0", "ALG_FREEZE": "r_gain", "ALG_ROUTER_PTR": "0.0",
    "ALG_SPAN_ALL": "1", "ALG_SPAN_ARGS": "1", "ALG_SPAN_OP": "1", "ALG_SPAN_RCUE": "1",
    "ALG_SPAN_ARCUE": "1", "ALG_PTR_SURF": "role:add:2.0", "BIND_CODES": ".cache/bindbus_codes512r.npz",
}
HIERD = {"ALG_HIER_READ": "1", "ALG_HIER_WAIST": "1", "ALG_HIER_DAMP": "2,4,0", "ALG_HIER_TAU": "0"}
RACKY = {"ALG_ALT3": "1", "ALG_CERT": "2.0", "ALG_RACK": "1",
         "ALG_RACK_TESTS": "given_unique", "ALG_RACK_FREEZE": "leaf"}

BODIES = {
    "PMS8_241": {"ckpt": ".cache/sharp_PMS8_241.safetensors", "env": dict(_FAM, **SURF8), "consults": False},
    "RK_241":   {"ckpt": ".cache/sharp_RK_241.safetensors",
                 "env": dict(_FAM, **SURF8, **HIERD, **RACKY), "consults": True},
}

N_ROWS = 24
HEAD_KEYS_SKIP = {"heads_all", "breaths", "breaths_u", "breaths_r"}   # stated scope limits (see report)


def _set_env(env):
    for k, v in list(os.environ.items()):
        if k.startswith("ALG_") or k in ("BIND_CODES", "R_GAIN_INIT", "DEV", "SC_EVAL"):
            del os.environ[k]
    for k, v in env.items():
        os.environ[k] = v
    assert os.environ.get("DEV") == "CPU"


def _load_body(tag, cfg):
    _set_env(cfg["env"])
    import importlib
    import phase1_algebra_head as H
    importlib.reload(H)
    from tinygrad.nn.state import safe_load
    p = H.build_params(0)
    sd = safe_load(cfg["ckpt"])
    assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return H, p


def _rows(H, n):
    from tinygrad import Tensor, dtypes
    vs, vst, vtk, vg, vse = H.load_alg("test")
    n = min(n, len(vs))
    sl = np.arange(n)
    ts = Tensor(np.ascontiguousarray(vst[sl]), dtype=dtypes.half)
    tk = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
    return vs, sl, ts, tk, se


def _final_pass_kwargs(H, p, vs, sl, ts, tk, se, consults):
    """Returns the kwargs the TRUE final forward() call needs -- mirrors membrane_rack.py's
    collect() exactly (pass-1 unmasked -> slot_mask; ALT3 on: the 3-pass consult cycle with
    facts3/facts5/cert3/cert5/rack3/rack5 all derived; ALT3 off: pass-2's own ALT2 fact_buf)."""
    from tinygrad import Tensor, dtypes
    o0 = H.forward(p, ts, tk, se)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = H.build_slot_masks(onp0, se.numpy().astype(np.int32))
    slot_mask = Tensor(mk, dtype=dtypes.float)
    texts = [vs[int(i)]["text"] for i in sl]
    _nv = np.array([vs[int(i)].get("n_vars", H.K_VARS) for i in sl])
    _ma = np.array([vs[int(i)].get("m", 0) for i in sl])

    if not consults:
        _ka = ("pres", "ftype", "op", "dig") + (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        fb = H.alt2_fact_buf(_oa, se.numpy().astype(np.int32), _nv, _ma)
        return dict(slot_mask=slot_mask, fact_buf=Tensor(fb, dtype=dtypes.float))

    ck3 = ("pres", "ftype", "op", "dig", "args", "res") + (("dup",) if "dup" in o0 else ())

    def consult3(oo, rack_prev=None):
        onp3 = {k: oo[k].realize().numpy() for k in ck3}
        fb_ = H.alt2_fact_buf(onp3, se.numpy().astype(np.int32), _nv, _ma)
        cb_t = rk_t = rk = None
        rows3 = [{k: onp3[k][bi] for k in onp3} for bi in range(ts.shape[0])]
        if H.ALG_CERT or H.ALG_CERT_IMPLIED:
            cb_t = Tensor(H.certifier_bias(rows3, fb_, texts, H.T_ALG, H.ALG_CERT, H.ALG_CERT_IMPLIED), dtype=dtypes.float)
        if H.ALG_RACK:
            rk = H.rack_pack(rows3, fb_, texts, H.T_ALG, H.ALG_RACK_TESTS, prev=rack_prev)
            rk_t = Tensor(rk, dtype=dtypes.float)
        return Tensor(fb_, dtype=dtypes.float), cb_t, rk_t, rk

    oa3 = H.forward(p, ts, tk, se, slot_mask=slot_mask, stop_after=2)
    f3_t, c3_t, r3_t, r3_np = consult3(oa3)
    ob3 = H.forward(p, ts, tk, se, slot_mask=slot_mask, stop_after=4, facts3=f3_t, cert3=c3_t, rack3=r3_t)
    f5_t, c5_t, r5_t, r5_np = consult3(ob3, rack_prev=r3_np)
    return dict(slot_mask=slot_mask, facts3=f3_t, facts5=f5_t, cert3=c3_t, cert5=c5_t, rack3=r3_t, rack5=r5_t)


def _compare(out_full, out_replay, tag):
    keys = sorted(set(out_full) | set(out_replay))
    mismatches = []
    checked = []
    for k in keys:
        if k in HEAD_KEYS_SKIP:
            continue
        a = out_full.get(k); b = out_replay.get(k)
        if a is None or b is None:
            if a is not b:
                mismatches.append((k, "present-in-one-only"))
            continue
        if not (hasattr(a, "realize") and hasattr(a, "numpy")):
            continue   # not a tensor head (e.g. a plain python value)
        an = a.realize().numpy(); bn = b.realize().numpy()
        if an.shape != bn.shape:
            mismatches.append((k, f"shape {an.shape} vs {bn.shape}"))
            continue
        if np.array_equal(an, bn):
            checked.append((k, 0.0))
        else:
            d = float(np.max(np.abs(an.astype(np.float64) - bn.astype(np.float64))))
            mismatches.append((k, f"max|delta|={d:.6g}"))
            checked.append((k, d))
    return checked, mismatches


def _decode_agreement(H, out_full, out_replay, n_rows):
    """Row-by-row: does the tiny float residual change the DISCRETE parse at all? Reuses
    decode() (scripts/phase1_algebra_head.py:7616) -- the same function that turns heads
    into the factor graph the solver actually sees -- on both out dicts, per row."""
    need = ["pres", "res", "ftype", "op", "dig", "args"]
    for extra in ("sgn", "dup", "dargs"):
        if extra in out_full and extra in out_replay:
            need.append(extra)
    fnp = {k: out_full[k].realize().numpy() for k in need if k in out_full}
    rnp = {k: out_replay[k].realize().numpy() for k in need if k in out_replay}
    n_diff = 0
    for i in range(n_rows):
        row_f = {k: fnp[k][i] for k in fnp}
        row_r = {k: rnp[k][i] for k in rnp}
        try:
            dec_f = H.decode(row_f)
            dec_r = H.decode(row_r)
        except Exception as e:
            dec_f = dec_r = f"<decode error: {e}>"
        if dec_f != dec_r:
            n_diff += 1
    return n_diff


def run_one(tag, kb):
    cfg = BODIES[tag]
    snap_path = f".cache/replay_snap_{tag}_kb{kb}.npz"
    H, p = _load_body(tag, cfg)
    vs, sl, ts, tk, se = _rows(H, N_ROWS)

    final_kwargs = _final_pass_kwargs(H, p, vs, sl, ts, tk, se, cfg["consults"])

    os.environ.pop("ALG_SNAP_AT", None); os.environ.pop("ALG_SNAP_OUT", None)
    out_full = H.forward(p, ts, tk, se, **final_kwargs)
    for k in out_full:   # force realization before env/state changes under it
        if hasattr(out_full[k], "realize"):
            out_full[k] = out_full[k].realize()

    os.environ["ALG_SNAP_AT"] = str(kb)
    os.environ["ALG_SNAP_OUT"] = snap_path
    _ = H.forward(p, ts, tk, se, **final_kwargs)
    os.environ.pop("ALG_SNAP_AT", None); os.environ.pop("ALG_SNAP_OUT", None)

    snap = H.snap_load(snap_path)
    out_replay = H.replay(p, snap, kb)

    checked, mismatches = _compare(out_full, out_replay, tag)
    n_decode_diff = _decode_agreement(H, out_full, out_replay, ts.shape[0])
    return checked, mismatches, n_decode_diff


def main():
    t0 = time.time()
    rows = []
    for tag in ("PMS8_241", "RK_241"):
        for kb in (1, 3, 5):
            checked, mismatches, n_decode_diff = run_one(tag, kb)
            ok = not mismatches
            max_d = max([d for _, d in checked], default=0.0)
            print(f"[gate] {tag} kb={kb}: {len(checked)} heads checked, "
                  f"{'BIT-IDENTICAL' if ok else 'MISMATCH'} "
                  f"({len(mismatches)} mismatches, max|delta| overall={max_d:.6g}, "
                  f"decode-level rows differing={n_decode_diff}/{N_ROWS})", flush=True)
            for k, info in mismatches:
                print(f"    MISMATCH {k}: {info}", flush=True)
            rows.append((tag, kb, len(checked), len(mismatches), max_d, n_decode_diff))
    print(f"[gate] done in {time.time() - t0:.0f}s")
    print("tag,kb,n_checked,n_mismatch,max_delta,n_decode_diff")
    for tag, kb, nc, nm, md, ndd in rows:
        print(f"{tag},{kb},{nc},{nm},{md:.6g},{ndd}")


if __name__ == "__main__":
    main()
