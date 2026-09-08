"""polar_sink_smoke.py — THE KITCHEN SINK BIRTH SMOKE (CPU, zero GPU,
zero training, zero gradient descent; 2026-09-08).

Proves scripts/apply_polar_sink.py's contract against the STAGED source —
the patch is built in memory (apply_polar_sink.py forced to --check) and
exec'd into in-memory modules, the polar_birth_smoke.py idiom, so the real
head is NEVER written. Fixture: .cache/pc_row_smoke.py's (banked test23
states, first 8 rows, build_params(0) random init — this smoke tests
MECHANISM, not skill).

  GATE 1 — ENV-INERTNESS, twice (the CPU stand-in for eq A/B/C):
     1a  ALG_POLAR=1 with ALG_POLAR_D / ALG_POLAR_EM UNSET: the APPLIED
         head and the STAGED head, same 8 rows, np.array_equal on every
         emission key and on every breath page.
     1b  ALG_POLAR unset entirely: same claim, and the staged module must
         not even build the sink's tables.
  GATE 2 — ALG_POLAR_D=128 (the content-plane waist):
     2a  ORGAN LEVEL, the exact claim: _polar_waist applied to one state
         leaves all 128 CLOCK dims BITWISE unchanged and restores the
         content block's norm, so ||u|| is preserved to f32.
     2b  FORWARD: ||u|| == 1 per slot at every breath; at the FIRST
         breath (the only one where both runs share their input) the
         clock planes are BITWISE the D=0 run's. From breath 2 on the
         runs legitimately diverge — the waist changed the state and the
         state is carried forward; that divergence is REPORTED, not
         gated (asserting it away would be asserting the organ inert).
     2c  BIRTH RECONSTRUCTION on the CHAMPION'S OWN dumped states
         (.cache/polar_states_{mint,wild}.npz), measured THROUGH THE
         HEAD'S ORGAN (the meter law) and cross-checked against the
         numpy projection scripts/polar_waist_init.py banked.
  GATE 3 — ALG_POLAR_EM=0.1 (the E&B coupling):
     3a  ORGAN LEVEL: the 384 CONTENT dims come through BITWISE, the
         clock dims move, everything finite, and the clock block's norm
         is restored (so ||u|| survives).
     3b  THE NORM CHANGE BEFORE RENORMALIZATION — the number the ledger
         wants: the median relative change of the clocked dims at
         kappa = 0.1, measured with the renorm organ lifted.
     3c  FORWARD: finite; content planes bitwise at the first breath.
  GATE 4 — TWO-TERMINAL: a parse loss reaches polar_wd AND polar_wu
     (nonzero) under D=128; losses on the DIAGNOSTIC taps breaths_r /
     breaths_u reach NOTHING — not the sink's parameters, not anything.
  GATE 5 — the apply script's --check: anchors present and unique, ast
     OK, structural asserts + symtable audit PASS, idempotence guard
     refuses a second application, and nothing is written.

Run from the repo root:
  .venv/bin/python3 scripts/polar_sink_smoke.py
"""
import json
import os
import runpy
import shutil
import subprocess
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

os.environ["DEV"] = "CPU"                 # HARD: this smoke never touches the GPU
ENV = {"ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1", "ALG_HW": "512",
       "ALG_WIDE": "1", "ALG_BREATH": "7", "ALG_NOTEBOOK": "1",
       "ALG_SIXWAVE": "1", "NB_PERSLOT": "1", "ALG_BINDBUS": "7",
       "ALG_BIND_D": "512", "BIND_CODES": ".cache/bindbus_codes512.npz",
       "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
       "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
       "ALG_FED": "1", "ALG_TEST": ".cache/algebra_nl_test.jsonl",
       "ALG_TEST_NAME": "test23",
       "ALG_MINE_BREATHS": "1"}       # the tap: breaths_all / breaths_u
os.environ.update(ENV)
for _k in ("ALG_PC_MIX", "ALG_PC_LIVE", "ALG_POLAR",
           "ALG_POLAR_D", "ALG_POLAR_D_INIT", "ALG_POLAR_EM"):
    os.environ.pop(_k, None)
os.environ["SC_EVAL"] = "0"            # the machine's living regime — SEAL FENCE

import numpy as np                                              # noqa: E402
from tinygrad import Tensor, dtypes                             # noqa: E402

HEAD = os.path.join(_ROOT, "scripts", "phase1_algebra_head.py")
BANDS = json.load(open(os.path.join(_ROOT, ".cache", "polar_bands.json")))
D_WIDTH = 128
KAPPA = 0.1
B = 8
KEYS = ("pres", "ftype", "op", "args", "res", "dig")
TOL_U = 1e-4

CLOCK_PL = np.array(sorted({p for w in BANDS["wheels"] for p in w["planes"]}),
                    np.int64)
CONT_PL = np.array(sorted(set(range(BANDS["n_planes"])) - set(CLOCK_PL.tolist())),
                   np.int64)
CLOCK_D = np.sort(np.concatenate([2 * CLOCK_PL, 2 * CLOCK_PL + 1]))
CONT_D = np.sort(np.concatenate([2 * CONT_PL, 2 * CONT_PL + 1]))
assert len(CLOCK_D) == 128 and len(CONT_D) == 384


# ===========================================================================
# STAGING — build the would-be patched source without writing the head
# ===========================================================================
def staged_source():
    argv0 = sys.argv
    sys.argv = ["apply_polar_sink.py", "--check"]
    try:
        ns = runpy.run_path(os.path.join(_ROOT, "scripts",
                                         "apply_polar_sink.py"),
                            run_name="_polar_sink_staging")
    finally:
        sys.argv = argv0
    assert ns["CHECK"], "staging must run under --check (nothing written)"
    return ns["s"]


def load_module(src, name):
    mod = types.ModuleType(name)
    mod.__file__ = HEAD + f" ({name}, in-memory)"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def with_env(env, fn_):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items()})
    try:
        return fn_()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


APPLIED = open(HEAD).read()
STAGED = staged_source()
assert APPLIED != STAGED and "ALG_POLAR_D" not in APPLIED
assert "ALG_POLAR = int" in APPLIED, "phase 3 rides on the APPLIED phase 1/2"
print(f"[sink-smoke] staged +{STAGED.count(chr(10)) - APPLIED.count(chr(10))} "
      f"lines on the APPLIED head; head on disk UNTOUCHED", flush=True)

M_BASE_OFF = load_module(APPLIED, "applied_polar_off")
M_STAG_OFF = load_module(STAGED, "staged_polar_off")
M_BASE_ON = with_env({"ALG_POLAR": "1"},
                     lambda: load_module(APPLIED, "applied_polar_on"))
M_STAG_ON = with_env({"ALG_POLAR": "1"},
                     lambda: load_module(STAGED, "staged_polar_on"))
M_D = with_env({"ALG_POLAR": "1", "ALG_POLAR_D": str(D_WIDTH)},
               lambda: load_module(STAGED, "staged_polar_D"))
M_EM = with_env({"ALG_POLAR": "1", "ALG_POLAR_EM": str(KAPPA)},
                lambda: load_module(STAGED, "staged_polar_EM"))
assert not M_BASE_OFF.ALG_POLAR and not M_STAG_OFF.ALG_POLAR
assert M_BASE_ON.ALG_POLAR and M_STAG_ON.ALG_POLAR
assert M_STAG_OFF.POLAR_D == 0 and M_STAG_OFF.POLAR_EM == 0.0
assert M_STAG_ON.POLAR_D == 0 and M_STAG_ON.POLAR_EM == 0.0
assert M_D.POLAR_D == D_WIDTH and M_D.POLAR_EM == 0.0
assert M_EM.POLAR_D == 0 and M_EM.POLAR_EM == KAPPA
assert M_STAG_OFF._POLAR_SINK is None, \
    "the sink built its tables with ALG_POLAR unset (not inert)"


# ===========================================================================
# FIXTURE — the pc_row_smoke idiom
# ===========================================================================
vs, vst, vtk, vg, vse = M_BASE_OFF.load_alg("test")
sl = np.arange(B)
TS = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
TK = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
SE = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
SENT = vse[sl].astype(np.int32)
print(f"[sink-smoke] fixture test23 rows={B} dev={os.environ['DEV']} "
      f"d={D_WIDTH} kappa={KAPPA}", flush=True)


def masked_read(M, p, extra_keys=(), **fwd):
    """loop_val's two-pass shape (pass 0 open -> slot masks -> pass 1)."""
    o0 = M.forward(p, TS, TK, SE)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = M.build_slot_masks(onp0, SENT)
    o = M.forward(p, TS, TK, SE,
                  slot_mask=Tensor(mk, dtype=dtypes.float), **fwd)
    out = {k: o[k].realize().numpy() for k in KEYS}
    for k in extra_keys:
        out[k] = [b.realize().numpy() for b in o[k]]
    return out, mk


def slot_masks_for(M, p):
    o0 = M.forward(p, TS, TK, SE)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    return M.build_slot_masks(onp0, SENT)


# ===========================================================================
# GATE 1 — ENV-INERTNESS, twice
# ===========================================================================
XK = ("breaths_all", "breaths_u", "breaths_r")
for lab, MA, MB, xk in (("ALG_POLAR=1, sink doors UNSET",
                         M_BASE_ON, M_STAG_ON, XK),
                        ("ALG_POLAR unset entirely",
                         M_BASE_OFF, M_STAG_OFF, ())):
    PA = MA.build_params(0)
    PB = MB.build_params(0)
    assert set(PA) == set(PB), \
        f"GATE 1 FAIL ({lab}): the patch changed the parameter set {sorted(set(PA) ^ set(PB))[:4]}"
    assert all(np.array_equal(PA[k].numpy(), PB[k].numpy()) for k in PA), \
        f"GATE 1 FAIL ({lab}): the patch moved the init rng stream"
    oa, _ = masked_read(MA, PA, extra_keys=xk)
    ob, _ = masked_read(MB, PB, extra_keys=xk)
    bad = [k for k in KEYS if not np.array_equal(oa[k], ob[k])]
    for k in xk:
        bad += [f"{k}[{j}]" for j in range(len(oa[k]))
                if not np.array_equal(oa[k][j], ob[k][j])]
    assert not bad, f"GATE 1 FAIL ({lab}): NOT bit-identical ({bad[:6]})"
    n_cmp = len(KEYS) + sum(len(oa[k]) for k in xk)
    print(f"[sink-smoke] GATE 1 PASS [{lab}]: applied vs staged "
          f"BIT-IDENTICAL on {n_cmp} tensors ({len(KEYS)} emission keys"
          + (f" + {n_cmp - len(KEYS)} breath pages" if xk else "")
          + "), params identical  [CPU stand-in for eq A/B/C]", flush=True)

P_ON = M_STAG_ON.build_params(0)
P_D = M_D.build_params(0)
P_EM = M_EM.build_params(0)
assert set(P_D) - set(P_ON) == {"polar_wd", "polar_wu"}, \
    f"GATE 1 FAIL: D=128 adds {sorted(set(P_D) - set(P_ON))}, expected two"
assert set(P_EM) == set(P_ON), \
    f"GATE 1 FAIL: ALG_POLAR_EM added parameters {sorted(set(P_EM) - set(P_ON))}"
assert all(np.array_equal(P_ON[k].numpy(), P_D[k].numpy()) for k in P_ON) and \
    all(np.array_equal(P_ON[k].numpy(), P_EM[k].numpy()) for k in P_ON), \
    "GATE 1 FAIL: the sink's params moved the shared init rng stream"
_npar = int(P_D["polar_wd"].numpy().size + P_D["polar_wu"].numpy().size)
print(f"[sink-smoke] params: ALG_POLAR_D={D_WIDTH} adds polar_wd"
      f"{tuple(P_D['polar_wd'].shape)} + polar_wu{tuple(P_D['polar_wu'].shape)}"
      f" = {_npar} entries; ALG_POLAR_EM adds 0 (fixed kappa). Shared init "
      f"stream untouched.", flush=True)


# ===========================================================================
# GATE 2 — ALG_POLAR_D (the content-plane waist)
# ===========================================================================
# --- 2a ORGAN LEVEL: the exact bitwise claim -------------------------------
_rs = np.random.RandomState(11)
_u0 = _rs.randn(3, 32, M_D.H_W).astype(np.float32)
_u0 /= np.linalg.norm(_u0, axis=-1, keepdims=True)
_T0 = Tensor(_u0, dtype=dtypes.float)
_st = {}
_u1 = M_D._polar_waist(_T0, P_D, _st).realize().numpy()
assert np.isfinite(_u1).all(), "GATE 2a FAIL: waist output non-finite"
assert np.array_equal(_u1[..., CLOCK_D], _u0[..., CLOCK_D]), \
    "GATE 2a FAIL: the waist moved the CLOCK dims"
assert not np.array_equal(_u1[..., CONT_D], _u0[..., CONT_D]), \
    "GATE 2a FAIL: the waist is a no-op on the content dims"
_n_c0 = np.linalg.norm(_u0[..., CONT_D], axis=-1)
_n_c1 = np.linalg.norm(_u1[..., CONT_D], axis=-1)
_dn_c = float(np.abs(_n_c1 / np.maximum(_n_c0, 1e-12) - 1.0).max())
_dn_u = float(np.abs(np.linalg.norm(_u1, axis=-1) - 1.0).max())
assert _dn_c < 1e-5 and _dn_u < TOL_U, \
    f"GATE 2a FAIL: block norm {_dn_c:.2e}, ‖u‖ {_dn_u:.2e}"
assert set(_st) == {"polar_wd_eff", "polar_wu_eff"}, \
    f"GATE 2a FAIL: the waist wrote {sorted(_st)} into state"
_rk = int(np.linalg.matrix_rank(
    (P_D["polar_wd"].numpy() @ P_D["polar_wu"].numpy()).astype(np.float64),
    tol=1e-4))
assert _rk == D_WIDTH, f"GATE 2a FAIL: the bottleneck has rank {_rk} != {D_WIDTH}"
print(f"[sink-smoke] GATE 2a PASS: _polar_waist leaves all {len(CLOCK_D)} "
      f"clock dims BITWISE unchanged (the bottleneck's domain and range are "
      f"the {len(CONT_D)} content dims only — it cannot fight the rotation), "
      f"restores the content block's norm (max dev {_dn_c:.2e}) so ‖u‖ = 1 "
      f"survives (max |‖u‖-1| = {_dn_u:.2e}), caches ONE scatter per forward "
      f"in state, and has rank exactly {_rk}", flush=True)

# --- 2b FORWARD ------------------------------------------------------------
o_on, _mk_on = masked_read(M_STAG_ON, P_ON, extra_keys=("breaths_u",))
o_d, _mk_d = masked_read(M_D, P_D, extra_keys=("breaths_u", "breaths_r"))
K = len(o_d["breaths_u"])
assert K == 7, f"breath page count {K}"
for k in KEYS:
    assert np.isfinite(o_d[k]).all(), f"GATE 2b FAIL: {k} non-finite"
worst = max(float(np.abs(np.linalg.norm(u, axis=-1) - 1.0).max())
            for u in o_d["breaths_u"])
assert worst <= TOL_U, f"GATE 2b FAIL: max |‖u‖-1| = {worst:.3e}"
assert np.array_equal(o_d["breaths_u"][0], o_on["breaths_u"][0]), \
    "GATE 2b FAIL: breath 0 (outside the loop) moved"
_ck1 = np.array_equal(o_d["breaths_u"][1][..., CLOCK_D],
                      o_on["breaths_u"][1][..., CLOCK_D])
assert _ck1, ("GATE 2b FAIL: at the FIRST loop breath — the only one whose "
              "input both runs share — the waist moved the clock planes")
assert not np.array_equal(o_d["breaths_u"][1][..., CONT_D],
                          o_on["breaths_u"][1][..., CONT_D]), \
    "GATE 2b FAIL: the waist did nothing to the content planes in-graph"
_div = [float(np.abs(o_d["breaths_u"][j][..., CLOCK_D]
                     - o_on["breaths_u"][j][..., CLOCK_D]).max())
        for j in range(K)]
print(f"[sink-smoke] GATE 2b PASS: ‖u‖ == 1 per slot at all {K} breaths "
      f"(max |‖u‖-1| = {worst:.3e}); clock planes BITWISE the D=0 run's at "
      f"breath 0 and breath 1. REPORTED, not gated — clock-plane max|d| vs "
      f"the D=0 run per breath: "
      f"{[f'{x:.1e}' for x in _div]} — from breath 2 the runs diverge "
      f"BECAUSE the waist changed the state and the state is carried "
      f"forward; asserting that away would be asserting the organ inert.",
      flush=True)

# --- 2c BIRTH RECONSTRUCTION on the champion's own dumped states -----------
_Z = {n: np.load(f".cache/polar_states_{n}.npz")["u"] for n in ("mint", "wild")}
_UC = np.concatenate([z.reshape(-1, M_D.H_W) for z in _Z.values()], 0)
_sub = _UC[::37]                            # a strided read, ~2.3k slots
_st2 = {}
_rec = M_D._polar_waist(Tensor(_sub, dtype=dtypes.float).reshape(1, -1, M_D.H_W),
                        P_D, _st2).realize().numpy().reshape(-1, M_D.H_W)
_c = _sub[:, CONT_D].astype(np.float64)
_cr = _rec[:, CONT_D].astype(np.float64)
_den = np.linalg.norm(_c, axis=1)
_rel_org = float(np.mean(np.linalg.norm(_cr - _c, axis=1) / np.maximum(_den, 1e-12)))
_V = P_D["polar_wd"].numpy().astype(np.float64)
_praw = (_c @ _V) @ _V.T                    # the ORTHOGONAL projection itself
_rel_raw = float(np.mean(np.linalg.norm(_praw - _c, axis=1)
                         / np.maximum(_den, 1e-12)))
_pz = np.load(f".cache/polar_waist_init_d{D_WIDTH}.npz")
assert np.array_equal(_pz["W_down"], P_D["polar_wd"].numpy()) and \
    np.array_equal(_pz["W_up"], P_D["polar_wu"].numpy()), \
    "GATE 2c FAIL: the head did not load the banked PCA birth verbatim"
assert np.array_equal(np.asarray(_pz["content_dims"], np.int64), CONT_D), \
    "GATE 2c FAIL: the banked init's content dims are not the band file's"
assert abs(float(_pz["var_kept"]) - 0.9188) < 5e-3, \
    f"GATE 2c FAIL: var_kept {float(_pz['var_kept'])} (expected ~0.9188)"
assert _rel_org < 0.40, f"GATE 2c FAIL: reconstruction {_rel_org:.4f}"
print(f"[sink-smoke] GATE 2c PASS: birth on the CHAMPION'S OWN dumped "
      f"states (n={len(_sub)} slots strided from both fixtures x 7 breaths "
      f"x 24 slots) — var kept {float(_pz['var_kept']):.4f} at d={D_WIDTH}; "
      f"relative ‖c'-c‖/‖c‖ = {_rel_raw:.4f} for the bare orthogonal "
      f"projection and {_rel_org:.4f} THROUGH THE HEAD'S ORGAN (projection "
      f"+ the content-block norm restore). The head loaded the banked init "
      f"verbatim — no silent random init anywhere.", flush=True)


# ===========================================================================
# GATE 3 — ALG_POLAR_EM (the E&B coupling)
# ===========================================================================
MK = slot_masks_for(M_EM, P_EM)             # the REAL lanes, pass-0 masks
_L0 = MK.shape[1]
_MT = Tensor(MK, dtype=dtypes.float)
_ue = _rs.randn(MK.shape[0], _L0, M_EM.H_W).astype(np.float32)
_ue /= np.linalg.norm(_ue, axis=-1, keepdims=True)
_TE = Tensor(_ue, dtype=dtypes.float)
_ue1 = M_EM._polar_em(_TE, _MT, KAPPA).realize().numpy()
assert np.isfinite(_ue1).all(), "GATE 3a FAIL: EM output non-finite"
assert np.array_equal(_ue1[..., CONT_D], _ue[..., CONT_D]), \
    "GATE 3a FAIL: the coupling moved the CONTENT dims"
assert not np.array_equal(_ue1[..., CLOCK_D], _ue[..., CLOCK_D]), \
    "GATE 3a FAIL: the coupling is a no-op on the clock dims"
_nk0 = np.linalg.norm(_ue[..., CLOCK_D], axis=-1)
_nk1 = np.linalg.norm(_ue1[..., CLOCK_D], axis=-1)
_dnk = float(np.abs(_nk1 / np.maximum(_nk0, 1e-12) - 1.0).max())
_dnu = float(np.abs(np.linalg.norm(_ue1, axis=-1) - 1.0).max())
assert _dnk < 1e-5 and _dnu < TOL_U, \
    f"GATE 3a FAIL: clock block norm {_dnk:.2e}, ‖u‖ {_dnu:.2e}"
print(f"[sink-smoke] GATE 3a PASS: at kappa={KAPPA} the coupling leaves all "
      f"{len(CONT_D)} content dims BITWISE unchanged, moves the clock dims, "
      f"is finite, and restores the clock block's norm (max dev {_dnk:.2e}) "
      f"so ‖u‖ = 1 holds (max |‖u‖-1| = {_dnu:.2e}). Mask: the fixture's own "
      f"pass-0 lanes, mean degree {float(MK.sum(-1).mean()):.2f}/{_L0}.",
      flush=True)

# --- 3b THE NORM CHANGE BEFORE RENORMALIZATION, on the DUMPED states -------
# The number the ledger wants, measured where the task pins it: the
# CHAMPION'S OWN dumped polar directions (.cache/polar_states_*.npz, u =
# (256, 7, 24, 512)) — real slots, real per-slot phase spread, one item's
# 24 slots per lane graph. The renorm organ is LIFTED for this read (and
# restored immediately) so the raw update is visible; the coupling itself
# is still the head's own organ. Two lane graphs bracket the answer: the
# FULLY OPEN 24x24 mask (the mean-field limit — every slot averages every
# other) and a SPARSE 4x6 block mask (a sentence-clique stand-in).
_Uc = np.concatenate([z.reshape(-1, 24, M_EM.H_W) for z in _Z.values()], 0)
_Uc = _Uc[::7].astype(np.float32)               # a strided read over items
_MOPEN = np.ones((len(_Uc), 24, 24), np.float32)
_MBLK = np.zeros((len(_Uc), 24, 24), np.float32)
for _g in range(4):
    _MBLK[:, 6 * _g:6 * _g + 6, 6 * _g:6 * _g + 6] = 1.0
_TU = Tensor(_Uc, dtype=dtypes.float)
_nk_d = np.linalg.norm(_Uc[..., CLOCK_D], axis=-1)
_keep = M_EM._polar_keepnorm
_rows = []
try:
    M_EM._polar_keepnorm = lambda a, b, g: a    # lift the renorm to expose it
    for _lab, _Mn in (("fully open 24x24", _MOPEN),
                      ("sparse 4x6 blocks", _MBLK)):
        _raw = M_EM._polar_em(_TU, Tensor(_Mn, dtype=dtypes.float),
                              KAPPA).realize().numpy()
        assert np.array_equal(_raw[..., CONT_D], _Uc[..., CONT_D]), \
            "GATE 3b FAIL: the pre-renorm update touched the content dims"
        _dn = np.abs(np.linalg.norm(_raw[..., CLOCK_D], axis=-1)
                     / np.maximum(_nk_d, 1e-12) - 1.0)
        _mv = (np.linalg.norm(_raw[..., CLOCK_D] - _Uc[..., CLOCK_D], axis=-1)
               / np.maximum(_nk_d, 1e-12))
        _rows.append((_lab, float(np.median(_dn)), float(np.percentile(_dn, 90)),
                      float(_dn.max()), float(np.median(_mv)),
                      float(_Mn.sum(-1).mean())))
finally:
    M_EM._polar_keepnorm = _keep
print(f"[sink-smoke] GATE 3b — THE NORM CHANGE BEFORE RENORMALIZATION at "
      f"kappa={KAPPA}, on the CHAMPION'S dumped states ({_Uc.shape[0]} items "
      f"x 24 slots, both fixtures, all breaths):")
print("      lane graph          | deg  | median |‖clock'‖/‖clock‖ - 1| | "
      "p90    | max    | median ‖Δclock‖/‖clock‖")
for _lab, _md, _p9, _mx, _mv, _dg in _rows:
    print(f"      {_lab:<19} | {_dg:4.1f} |        {_md:.4f}            | "
          f"{_p9:.4f} | {_mx:.4f} |        {_mv:.4f}")
assert all(r[1] > 0 for r in _rows), \
    "GATE 3b FAIL: the coupling changed no norm at all"
print(f"[sink-smoke] GATE 3b PASS: the update IS norm-changing (z <- "
      f"(I - i*kappa*L)z has |eigenvalue| = sqrt(1 + kappa^2*lambda^2) > 1); "
      f"at kappa={KAPPA} it is a {_rows[0][1] * 100:.2f}% median growth of "
      f"the clocked block under the open lane graph and "
      f"{_rows[1][1] * 100:.2f}% under the sparse one — which is why the "
      f"clock block's norm is restored right after (the renorm is part of "
      f"the organ, not an afterthought).", flush=True)
# --- 3c FORWARD ------------------------------------------------------------
o_em, _ = masked_read(M_EM, P_EM, extra_keys=("breaths_u",))
for k in KEYS:
    assert np.isfinite(o_em[k]).all(), f"GATE 3c FAIL: {k} non-finite"
for j, u in enumerate(o_em["breaths_u"]):
    assert np.isfinite(u).all(), f"GATE 3c FAIL: breaths_u[{j}] non-finite"
_we = max(float(np.abs(np.linalg.norm(u, axis=-1) - 1.0).max())
          for u in o_em["breaths_u"])
assert _we <= TOL_U, f"GATE 3c FAIL: max |‖u‖-1| = {_we:.3e}"
assert np.array_equal(o_em["breaths_u"][1][..., CONT_D],
                      o_on["breaths_u"][1][..., CONT_D]), \
    "GATE 3c FAIL: at the first loop breath the coupling moved content planes"
assert not np.array_equal(o_em["breaths_u"][1][..., CLOCK_D],
                          o_on["breaths_u"][1][..., CLOCK_D]), \
    "GATE 3c FAIL: the coupling did nothing in-graph"
_divc = [float(np.abs(o_em["breaths_u"][j][..., CONT_D]
                      - o_on["breaths_u"][j][..., CONT_D]).max())
         for j in range(K)]
print(f"[sink-smoke] GATE 3c PASS: forward finite, ‖u‖ == 1 at all {K} "
      f"breaths (max dev {_we:.3e}); content planes BITWISE the EM=0 run's "
      f"at breaths 0 and 1. REPORTED: content max|d| per breath "
      f"{[f'{x:.1e}' for x in _divc]} — the same carried-state divergence "
      f"as GATE 2b, and the same reason.", flush=True)


# ===========================================================================
# GATE 4 — TWO-TERMINAL
# ===========================================================================
def grads_from(M, P, loss_of, keys):
    for t_ in P.values():
        t_.grad = None
    o0 = M.forward(P, TS, TK, SE)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = M.build_slot_masks(onp0, SENT)
    o = M.forward(P, TS, TK, SE, slot_mask=Tensor(mk, dtype=dtypes.float))
    loss_of(o).backward()
    out = {}
    for k in keys:
        g = P[k].grad
        out[k] = 0.0 if g is None else float(g.detach().abs().max().numpy())
    return out


PK = ("polar_wd", "polar_wu", "W_sil", "waist_w")
g_parse = grads_from(M_D, P_D, lambda o: o["res"].sum(), PK)
g_rad = grads_from(M_D, P_D, lambda o: sum(x.sum() for x in o["breaths_r"]), PK)
g_dir = grads_from(M_D, P_D, lambda o: sum(x.sum() for x in o["breaths_u"]), PK)
print("[sink-smoke] grad |max| by terminal (ALG_POLAR_D=128):")
for k in PK:
    print(f"      {k:10s} parse-loss={g_parse[k]:.3e}  "
          f"breaths_r={g_rad[k]:.3e}  breaths_u={g_dir[k]:.3e}")
assert g_parse["polar_wd"] > 0.0 and g_parse["polar_wu"] > 0.0, \
    ("GATE 4 FAIL: the parse loss does not reach the waist's parameters — "
     "the two-terminal contract is broken (emission AND gold feed, or the "
     "grad is None)")
assert g_parse["W_sil"] > 0.0 and g_parse["waist_w"] > 0.0, \
    "GATE 4 FAIL: a known-live parameter went dark under the sink"
assert all(g_rad[k] == 0.0 for k in PK) and all(g_dir[k] == 0.0 for k in PK), \
    ("GATE 4 FAIL: a DIAGNOSTIC tap carries gradient into the parameters — "
     "the radius/direction register is supervisable (the Goodhart fence)")
print("[sink-smoke] GATE 4 PASS: two-terminal — a parse loss reaches BOTH "
      "polar_wd and polar_wu through the polar state (live), and losses on "
      "breaths_r / breaths_u reach NOTHING, the sink's parameters included.",
      flush=True)


# ===========================================================================
# GATE 5 — the apply script's --check and its idempotence guard
# ===========================================================================
_py = sys.executable
_r = subprocess.run([_py, "scripts/apply_polar_sink.py", "--check"],
                    capture_output=True, text=True, cwd=_ROOT)
assert _r.returncode == 0, f"GATE 5 FAIL: --check exit {_r.returncode}\n{_r.stderr[-800:]}"
assert "3 anchors OK" in _r.stdout and "symtable free-var audit PASS" in _r.stdout \
    and "NOTHING written" in _r.stdout, f"GATE 5 FAIL:\n{_r.stdout}"
assert open(HEAD).read() == APPLIED, "GATE 5 FAIL: --check wrote to the head"
_tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "polar_sink_rehearsal.py")
shutil.copy2(HEAD, _tmp)
_e = dict(os.environ, PS_TARGET=_tmp)
_r2 = subprocess.run([_py, "scripts/apply_polar_sink.py"],
                     capture_output=True, text=True, cwd=_ROOT, env=_e)
assert _r2.returncode == 0 and "APPLIED" in _r2.stdout, \
    f"GATE 5 FAIL: rehearsal apply\n{_r2.stderr[-800:]}"
assert open(_tmp).read() == STAGED, \
    "GATE 5 FAIL: the applied rehearsal differs from the staged source"
_r3 = subprocess.run([_py, "scripts/apply_polar_sink.py", "--check"],
                     capture_output=True, text=True, cwd=_ROOT, env=_e)
assert _r3.returncode != 0 and "idempotence" in _r3.stderr, \
    f"GATE 5 FAIL: a second application was not refused\n{_r3.stdout}{_r3.stderr}"
os.remove(_tmp)
assert open(HEAD).read() == APPLIED, "GATE 5 FAIL: the head moved"
print("[sink-smoke] GATE 5 PASS: --check clean (3 anchors present+unique, "
      "ast OK, structural asserts + symtable audit PASS, nothing written); "
      "a rehearsal apply reproduces the staged source exactly and a second "
      "application is REFUSED by the idempotence guard.", flush=True)

print("[sink-smoke] ALL GATES PASS — both sink doors are byte-inert when "
      "unset (with ALG_POLAR=1 and with ALG_POLAR unset); the content "
      "waist never touches the clock and the E&B coupling never touches "
      "the content; ‖u‖ = 1 survives both; r is untouched; the parse loss "
      "reaches both new parameters and no diagnostic reaches anything. "
      "NOT PROVEN HERE (GPU, the lead's gate): eq A/B/C bit-identity, "
      "pc_row_smoke ALL GATES, and any claim about SKILL.", flush=True)
