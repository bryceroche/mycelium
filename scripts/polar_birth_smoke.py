"""polar_birth_smoke.py — THE POLAR WAIST BIRTH SMOKE (CPU, zero GPU,
zero training, zero gradient descent; 2026-09-07).

Runs spec S5 items 4 and 5 (docs/polar_waist_spec.md) against the
STAGED source of scripts/apply_polar_waist.py — the patch is built in
memory (apply_polar_waist.py forced to --check) and exec'd into
in-memory modules, exactly the seamtest_vector.py idiom, so the real
head is NEVER written. The row/forward idiom is .cache/pc_row_smoke.py's
(banked test23 states, first 8 rows, load_alg("test"), build_params(0)
random init: this smoke tests MECHANISM, not skill).

  GATE A — ENV-INERTNESS (the CPU stand-in for the eq A/B/C gate, which
     is GPU and is NOT run here). The PRISTINE head and the STAGED head
     with ALG_POLAR unset, same 8 rows, same two-pass masked read:
     np.array_equal on every emission key. This is the same claim eq
     A/B/C makes, measured on CPU at one batch — the lead still runs
     the real eq gate before trusting the applied patch.
  GATE B — SPEC S5 ITEM 4, birth with ALG_POLAR=1 on 8 rows:
     B1 every emission finite;
     B2 ||u|| == 1 per slot at EVERY breath (to float32 tolerance: the
        where-gate guards the DIVISION, so u is exactly unit for every
        non-zero slot and exactly zero at the origin);
     B3 the old coordinates are recovered: breaths_all == r * u;
     B4 THE ROTATION UNIT TEST (a test of the rotation, not the model):
        with the write ZEROED (drop=0 makes the breath an exact
        identity, door #52) the clock bands of u at breath k+1 are the
        bands at breath k turned by exactly the wheel's angle — 60 deg
        breath hand, 120 deg parity, 0 deg pass wheel — and the content
        planes are BITWISE unchanged;
     B6 the Q-side plane indexing on BOTH attention paths — the fed
        mixer (through its per-head reshape) and the main sc2 mixer
        (flat on 512 dims) must agree plane-for-plane with the state's
        clock, and ALG_POLAR_QROT must default to 2 (main + mixer);
     B5 THE BREATH PROBE at birth on breaths_u, using clock_read.py's
        OWN ridge_probe (the meter law: a check must call its organ),
        printed beside the same probe on breaths_all and clock_read's
        built-in norm-only control.
  GATE C — SPEC S5 ITEM 5, the gradient contract:
     C1 the where-gates: _polar_ru on a state with an exact-zero slot —
        forward r == 0 and u == 0 exactly, backward FINITE AND BOUNDED
        (the un-gated sqrt is what detonated the confidence stamp at
        step 5000; an epsilon denominator would keep the forward finite
        and hand the backward a 1/eps = 1e6 spike instead);
     C2 TWO-TERMINAL: a parse loss (out["res"]) reaches the parameters
        THROUGH the polar state (the live terminal), and a loss on the
        diagnostic taps out["breaths_r"] / out["breaths_u"] reaches NO
        parameter at all (the severed terminal) — r receives no loss
        gradient from any diagnostic.

Run from the repo root:
  .venv/bin/python3 scripts/polar_birth_smoke.py
"""
import json
import os
import runpy
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
for _k in ("ALG_PC_MIX", "ALG_PC_LIVE", "ALG_POLAR"):
    os.environ.pop(_k, None)
os.environ["SC_EVAL"] = "0"            # the machine's living regime (open
                                       # shelf circle) — THE SEAL FENCE

import numpy as np                                              # noqa: E402
from tinygrad import Tensor, dtypes                             # noqa: E402

HEAD = os.path.join(_ROOT, "scripts", "phase1_algebra_head.py")
BANDS = json.load(open(os.path.join(_ROOT, ".cache", "polar_bands.json")))
B = 8
KEYS = ("pres", "ftype", "op", "args", "res", "dig")
TOL_U = 1e-4          # ||u|| == 1 within float32 accumulation noise
TOL_ROT = 2e-4        # the rotation unit test (f32, 512-d, seven breaths)


# ===========================================================================
# STAGING — build the would-be patched source without writing the head
# ===========================================================================
def staged_source():
    argv0 = sys.argv
    sys.argv = ["apply_polar_waist.py", "--check"]
    try:
        ns = runpy.run_path(os.path.join(_ROOT, "scripts",
                                         "apply_polar_waist.py"),
                            run_name="_polar_waist_staging")
    finally:
        sys.argv = argv0
    assert ns["CHECK"], "staging must run under --check (nothing written)"
    return ns["s"]


def load_module(src, name):
    mod = types.ModuleType(name)
    mod.__file__ = HEAD + f" ({name}, in-memory)"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


PRISTINE = open(HEAD).read()
STAGED = staged_source()
assert PRISTINE != STAGED and "ALG_POLAR" not in PRISTINE
print(f"[polar-smoke] staged +{STAGED.count(chr(10)) - PRISTINE.count(chr(10))} "
      f"lines; head on disk UNTOUCHED", flush=True)

M_BASE = load_module(PRISTINE, "head_PRISTINE")
M_OFF = load_module(STAGED, "head_STAGED_polar_off")
os.environ["ALG_POLAR"] = "1"          # read at module level -> own module
M_ON = load_module(STAGED, "head_STAGED_polar_on")
os.environ.pop("ALG_POLAR", None)      # the process default stays OFF
assert M_BASE.ALG_BREATH if hasattr(M_BASE, "ALG_BREATH") else True
assert not M_OFF.ALG_POLAR and M_ON.ALG_POLAR


# ===========================================================================
# FIXTURE — the pc_row_smoke idiom: banked test23 states, first 8 rows
# ===========================================================================
vs, vst, vtk, vg, vse = M_BASE.load_alg("test")
sl = np.arange(B)
TS = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
TK = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
SE = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
SENT = vse[sl].astype(np.int32)
print(f"[polar-smoke] fixture test23 rows={B} dev={os.environ['DEV']}",
      flush=True)


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
    return out


# ===========================================================================
# GATE A — ENV-INERTNESS (the CPU stand-in for eq A/B/C)
# ===========================================================================
P_BASE = M_BASE.build_params(0)
P_OFF = M_OFF.build_params(0)
assert set(P_BASE) == set(P_OFF), "the patch changed the parameter set"
assert all(np.array_equal(P_BASE[k].numpy(), P_OFF[k].numpy())
           for k in P_BASE), "the patch moved the init rng stream"
o_base = masked_read(M_BASE, P_BASE)
o_off = masked_read(M_OFF, P_OFF)
bad = [k for k in KEYS if not np.array_equal(o_base[k], o_off[k])]
assert not bad, f"GATE A FAIL: ALG_POLAR unset is NOT bit-identical ({bad})"
assert "breaths_u" not in M_OFF.forward(P_OFF, TS, TK, SE), \
    "GATE A FAIL: the polar tap exists with ALG_POLAR unset"
print("[polar-smoke] GATE A PASS: pristine vs staged(ALG_POLAR unset) "
      f"BIT-IDENTICAL on {len(KEYS)} emission keys, params identical, no "
      "polar key emitted  [CPU stand-in for eq A/B/C — the GPU eq gate is "
      "still the contract]", flush=True)


# ===========================================================================
# GATE B — SPEC S5 ITEM 4 (birth with ALG_POLAR=1)
# ===========================================================================
P_ON = M_ON.build_params(0)
o_on = masked_read(M_ON, P_ON, extra_keys=("breaths_all", "breaths_u",
                                           "breaths_r"))
K = len(o_on["breaths_all"])
assert K == len(o_on["breaths_u"]) == len(o_on["breaths_r"]) == 7, \
    f"breath page count {K} (expected 7: breath-0 + six loop breaths)"

# --- B1 finite -------------------------------------------------------------
for k in KEYS:
    assert np.isfinite(o_on[k]).all(), f"GATE B1 FAIL: {k} non-finite"
for nm in ("breaths_all", "breaths_u", "breaths_r"):
    for j, a in enumerate(o_on[nm]):
        assert np.isfinite(a).all(), f"GATE B1 FAIL: {nm}[{j}] non-finite"
print("[polar-smoke] GATE B1 PASS: every emission and every breath page "
      "finite at birth under ALG_POLAR=1", flush=True)

# --- B2 ||u|| == 1 ---------------------------------------------------------
un = [np.linalg.norm(u, axis=-1) for u in o_on["breaths_u"]]
worst = max(float(np.abs(n - 1.0).max()) for n in un)
assert worst <= TOL_U, f"GATE B2 FAIL: max |‖u‖-1| = {worst:.3e} > {TOL_U}"
print(f"[polar-smoke] GATE B2 PASS: ‖u‖ == 1 per slot at all {K} breaths "
      f"(max |‖u‖-1| = {worst:.3e} — float32 accumulation only: the "
      f"where-gate guards the DIVISION, not the coordinate)", flush=True)

# --- B3 the old coordinates are recovered ----------------------------------
d_rec = max(float(np.abs(o_on["breaths_all"][j]
                         - o_on["breaths_u"][j] * o_on["breaths_r"][j]
                         .reshape(B, 24, -1)[:, :, :1]).max())
            for j in range(K)) if o_on["breaths_r"][0].shape[2] == 1 else None
assert d_rec is not None and d_rec <= 1e-3, \
    f"GATE B3 FAIL: breaths_all != r*u (max|d| = {d_rec})"
rad = [float(np.median(r)) for r in o_on["breaths_r"]]
print(f"[polar-smoke] GATE B3 PASS: breaths_all == r*u (max|d| = "
      f"{d_rec:.3e}); median radius per breath = "
      f"{[round(x, 3) for x in rad]}", flush=True)

# --- B4 THE ROTATION UNIT TEST (write zeroed via drop=0, door #52) ---------
DROP0 = Tensor(np.zeros((1,), np.float32), dtype=dtypes.float)
o_z = masked_read(M_ON, P_ON, extra_keys=("breaths_u",), drop=DROP0)
UZ = o_z["breaths_u"]
WHEEL = {w["name"]: (np.array(w["planes"], np.int64), w["wheel"])
         for w in BANDS["wheels"]}
content = np.array(sorted(set(range(BANDS["n_planes"]))
                          - {p for w in BANDS["wheels"] for p in w["planes"]}),
                   np.int64)
EXPECT = {"breath_hand": 60.0, "parity": 120.0, "pass_wheel": 0.0}


def fit_angle(a, b, planes):
    """Least-squares plane rotation angle (deg) taking a -> b on `planes`."""
    x, y = a[..., 2 * planes], a[..., 2 * planes + 1]
    xp, yp = b[..., 2 * planes], b[..., 2 * planes + 1]
    a = float(np.degrees(np.arctan2((x * yp - y * xp).sum(),
                                    (x * xp + y * yp).sum()))) % 360.0
    return 0.0 if a > 359.999 else a       # atan2(-0, +) wraps to 360


def resid(a, b, planes, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    x, y = a[..., 2 * planes], a[..., 2 * planes + 1]
    return float(np.abs(np.stack([x * c - y * s, x * s + y * c], -1)
                        - np.stack([b[..., 2 * planes],
                                    b[..., 2 * planes + 1]], -1)).max())


print("[polar-smoke] B4 rotation unit test (write ZEROED: drop=0):")
print("      transition | breath_hand | parity | pass_wheel | content max|d|")
for j in range(K - 1):
    row, ok = [], True
    for nm in ("breath_hand", "parity", "pass_wheel"):
        pl = WHEEL[nm][0]
        ang = fit_angle(UZ[j], UZ[j + 1], pl)
        exp = 0.0 if j == 0 else EXPECT[nm]   # breath 0 -> loop breath 1 is
        r_ = resid(UZ[j], UZ[j + 1], pl, exp)  # the ENTRY step: phase_of(1)=0
        ok = ok and r_ <= TOL_ROT
        row.append(f"{ang:7.2f}deg (r {r_:.1e})")
    cd = float(np.abs(UZ[j][..., 2 * content] - UZ[j + 1][..., 2 * content]).max())
    cd = max(cd, float(np.abs(UZ[j][..., 2 * content + 1]
                              - UZ[j + 1][..., 2 * content + 1]).max()))
    print(f"      b{j} -> b{j+1}   | " + " | ".join(row) + f" | {cd:.2e}")
    assert ok, f"GATE B4 FAIL: transition b{j}->b{j+1} is not the wheel angle"
    assert cd <= TOL_ROT, \
        f"GATE B4 FAIL: content planes moved ({cd:.2e}) with the write zeroed"
print("[polar-smoke] GATE B4 PASS: with the write zeroed the clock bands "
      "turn by EXACTLY the wheel angle every loop breath (60 / 120 / 0 deg; "
      "b0 -> b1 is the entry step, phase_of(1) = 0 by rotor_clock's "
      "contract) and the 200 content planes do not move", flush=True)

# --- B6 THE Q-SIDE INDEXING: state and attention are ONE clock -------------
# The Q rotation lives behind the mixer's ZERO-INIT gains, so at birth it
# is invisible in the emissions — it cannot be proven by a forward diff.
# What CAN be proven, and is the thing that would silently break, is the
# INDEXING: the head reshapes bq (B, L, 512) -> (B, MX_HEADS, L, 64) and
# rotates pairs per head, so plane p of the 512-d waist must land on head
# p // (P/MX_HEADS), pair p % (P/MX_HEADS). Replicated here in numpy from
# the module's OWN table, plane for plane.
_qdc, _qds, _qac, _qas, _qwof = M_ON._polar_tables()
_P = M_ON.H_W // 2
_hd2 = (M_ON.H_W // M_ON.MX_HEADS) // 2
_KBQ = 3                                    # loop breath 3 -> absolute 120 deg
_bq = np.zeros((1, 1, M_ON.H_W), np.float32)
_bq[0, 0, 0::2] = 1.0                       # every plane's x-component = 1
_q4 = _bq.reshape(1, 1, M_ON.MX_HEADS, -1).transpose(0, 2, 1, 3)
_rc = _qac[_KBQ - 1].reshape(M_ON.MX_HEADS, _hd2)[None, :, None, :]
_rs = _qas[_KBQ - 1].reshape(M_ON.MX_HEADS, _hd2)[None, :, None, :]
_qp = _q4.reshape(1, M_ON.MX_HEADS, 1, _hd2, 2)
_qx, _qy = _qp[..., 0], _qp[..., 1]
_flat = np.stack([_qx * _rc - _qy * _rs, _qx * _rs + _qy * _rc], -1) \
    .reshape(1, M_ON.MX_HEADS, 1, -1).transpose(0, 2, 1, 3).reshape(M_ON.H_W)
_bad = [p for p in range(_P)
        if abs(_flat[2 * p] - _qac[_KBQ - 1][p]) > 1e-6
        or abs(_flat[2 * p + 1] - _qas[_KBQ - 1][p]) > 1e-6]
assert not _bad, f"GATE B6 FAIL: mixer Q-side plane indexing wrong for {_bad[:8]}"
# THE MAIN sc2 PATH (ALG_POLAR_QROT >= 2, the default): no head reshape —
# _rot2 straight on the 512-d queries. It must produce the SAME per-plane
# phase as the mixer's reshaped path, or "one clock" is a story.
_mx, _my = _bq[0, 0, 0::2], _bq[0, 0, 1::2]
_mc, _ms = _qac[_KBQ - 1], _qas[_KBQ - 1]
_main = np.empty(M_ON.H_W, np.float32)
_main[0::2] = _mx * _mc - _my * _ms
_main[1::2] = _mx * _ms + _my * _mc
assert np.allclose(_main, _flat, atol=1e-6), \
    ("GATE B6 FAIL: the main sc2 path and the mixer disagree on per-plane "
     "phase — they are two clocks, not one")
_uncl = np.array([p for p in range(_P) if _qwof[p] < 0], np.int64)
assert np.array_equal(_main[2 * _uncl], _bq[0, 0, 2 * _uncl]) and \
    np.array_equal(_main[2 * _uncl + 1], _bq[0, 0, 2 * _uncl + 1]), \
    "GATE B6 FAIL: content planes moved on the main path"
assert M_ON.POLAR_QROT == 2, \
    f"GATE B6 FAIL: ALG_POLAR_QROT default is {M_ON.POLAR_QROT}, expected 2"
_nq = int((_qwof >= 0).sum())
print(f"[polar-smoke] GATE B6 PASS: BOTH Q paths land the SAME {_nq} "
      f"clocked plane indices as the state's clock — the mixer through its "
      f"(MX_HEADS, {_hd2}) reshape and the MAIN sc2 path flat on 512 dims "
      f"agree plane-for-plane (all {_P} planes at loop breath {_KBQ}; "
      f"{len(_uncl)} content planes exactly unrotated on both). QROT "
      f"default = {M_ON.POLAR_QROT} (main+mixer). NOTE the asymmetry: "
      f"the MIXER's turn is invisible in birth emissions (it sits behind "
      f"zero-init fed_mx_hg gains), while the MAIN path's turn is LIVE at "
      f"birth — it enters sc2 ungated, so it does move the parse. That is "
      f"the point of QROT=2, and it is why this is a new generation "
      f"measured as a twin, never asserted equivalent.", flush=True)


# --- B5 THE BREATH PROBE at birth (clock_read.py's own organ) --------------
import clock_read as CR                                          # noqa: E402


def probe(pages, label, per_slot):
    if per_slot:
        X = np.stack([p.reshape(-1, p.shape[-1]).astype(np.float64)
                      for p in pages])
    else:
        X = np.stack([p.mean(1).astype(np.float64) for p in pages])
    pr = CR.ridge_probe(X, 242)
    print(f"      {label:34s} n={X.shape[1]:4d} D={X.shape[2]:4d}  "
          f"probe={pr['acc']:.4f}   norm-only control={pr['norm_acc']:.4f}")
    return pr


print("[polar-smoke] B5 breath probe at BIRTH (clock_read.ridge_probe, "
      "random-init params, 8 rows):")
pr_u_s = probe(o_on["breaths_u"], "breaths_u  (per-slot samples)", True)
pr_a_s = probe(o_on["breaths_all"], "breaths_all (per-slot samples)", True)
pr_u_i = probe(o_on["breaths_u"], "breaths_u  (item-pooled, CR idiom)", False)
pr_a_i = probe(o_on["breaths_all"], "breaths_all (item-pooled, CR idiom)", False)
print("[polar-smoke] B5 REPORTED, NOT GATED at n=8: clock_read's 0.95 bar "
      "is defined at n=256 on a TRAINED checkpoint (spec S6). The n=8 "
      "item-pooled split is 5 train / 3 test items — a 3-sample test set "
      "cannot carry a 0.95 claim. The per-slot read (n=192) is the "
      "informative one here.", flush=True)


# ===========================================================================
# GATE D — the slotvec radius door (ALG_POLAR_R_MODE=slotvec), unit-tested
# ===========================================================================
# The twin fires under the default scalar radius, so the forward gates above
# exercise scalar only. slotvec is a coded door; unit-test the two things
# that could silently break it: per-GROUP unit norm, and the allocation
# invariant that makes a band rotation norm-preserving inside every group
# (a plane occupies dims 2p, 2p+1 — it must never straddle a group edge).
_G8 = 8
_xs = np.random.RandomState(7).randn(3, 5, M_ON.H_W).astype(np.float32)
_rs8, _us8 = M_ON._polar_ru(Tensor(_xs, dtype=dtypes.float), _G8)
_rn8, _un8 = _rs8.numpy(), _us8.numpy()
_ug8 = _un8.reshape(3, 5, _G8, -1)
_dn8 = float(np.abs(np.linalg.norm(_ug8, axis=-1) - 1.0).max())
_dr8 = float(np.abs(_ug8 * _rn8 - _xs.reshape(3, 5, _G8, -1)).max())
assert _dn8 < 1e-5, f"GATE D FAIL: group norms {_dn8:.3e}"
assert _dr8 < 1e-4, f"GATE D FAIL: r*u reconstruction {_dr8:.3e}"
assert (M_ON.H_W // _G8) % 2 == 0, \
    "GATE D FAIL: a slotvec group must contain whole planes"
print(f"[polar-smoke] GATE D PASS: ALG_POLAR_R_MODE=slotvec (g={_G8}) — "
      f"‖u‖ == 1 per GROUP (max dev {_dn8:.3e}), r*u reconstructs "
      f"(max|d| {_dr8:.3e}), and groups hold whole planes so the band "
      f"rotation preserves every group radius. Unit only: the twin fires "
      f"under the scalar default.", flush=True)


# ===========================================================================
# GATE C — SPEC S5 ITEM 5 (the gradient contract)
# ===========================================================================
# --- C1 the where-gate at the exact origin ---------------------------------
xz = np.random.RandomState(0).randn(2, 4, M_ON.H_W).astype(np.float32)
xz[0, 1] = 0.0                                   # an EXACT-zero slot
tz = Tensor(xz, dtype=dtypes.float, requires_grad=True).contiguous().realize()
tz.requires_grad = True
r_z, u_z = M_ON._polar_ru(tz, 1)
(u_z.sum() + r_z.sum()).backward()
r_np, u_np, g_np = r_z.numpy(), u_z.numpy(), tz.grad.numpy()
assert np.isfinite(g_np).all(), "GATE C1 FAIL: non-finite grad at the origin"
assert np.isfinite(r_np).all() and np.isfinite(u_np).all()
assert float(r_np[0, 1, 0, 0]) == 0.0 and not u_np[0, 1].any(), \
    f"GATE C1 FAIL: (r, u) at the zero slot = {float(r_np[0,1,0,0])}, u!=0"
assert np.array_equal((r_np[0, 1] * u_np[0, 1]).reshape(-1), xz[0, 1]), \
    "GATE C1 FAIL: r*u does not reconstruct the zero slot"
_gmax = float(np.abs(g_np).max())
_gzero = float(np.abs(g_np[0, 1]).max())          # the exact-zero slot's own
_rmin = float(r_np[r_np > 0].min())
_bound = max(1.0, 2.0 / _rmin)                    # 1 at the origin, 2/r away
assert _gzero <= 1.0 + 1e-6, \
    (f"GATE C1 FAIL: the zero slot's grad is {_gzero:.3e}, not 1.0 — the "
     f"denominator gate is not holding (an eps guard gives 1/eps here)")
assert _gmax <= _bound + 1e-6, \
    f"GATE C1 FAIL: grad {_gmax:.3e} exceeds the bound {_bound:.3e}"
print(f"[polar-smoke] GATE C1 PASS: doubly where-gated normalization — at "
      f"an exact-zero slot r == 0, u == 0, r*u == x, and its own backward "
      f"is EXACTLY {_gzero:.6f} (an epsilon denominator would make this "
      f"1/eps = 1e6); overall max|grad| = {_gmax:.3e} <= max(1, 2/r_min) = "
      f"{_bound:.3e}. NOTE the residual, reported not gated: away from the "
      f"origin d(u)/dx scales as 1/r, so a slot whose radius approaches "
      f"(but never reaches) zero still steepens; the measured state radius "
      f"is 7-12 and nothing drives it to the origin the way the live wire "
      f"drove the deposit.", flush=True)


# --- C2 two-terminal: live parse terminal vs severed diagnostic terminal ---
def grads_from(loss_of, keys):
    for t_ in P_ON.values():
        t_.grad = None
    o0 = M_ON.forward(P_ON, TS, TK, SE)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = M_ON.build_slot_masks(onp0, SENT)
    o = M_ON.forward(P_ON, TS, TK, SE,
                     slot_mask=Tensor(mk, dtype=dtypes.float))
    loss_of(o).backward()
    out = {}
    for k in keys:
        g = P_ON[k].grad
        out[k] = 0.0 if g is None else float(g.detach().abs().max().numpy())
    return out


# LIVE: parameters whose ONLY road from a parse loss runs THROUGH the
# polar state — W_sil is the notebook ink written from the post-polar cur,
# W_gq the garage query read from it at the next breath, waist_w the trunk
# projection every breath re-reads. DEAD-AT-COLD-INIT, asserted as such
# (the pc_row_smoke GATE-4 facts, not new): W_bind1's deposit is DETACHED
# with the live wire closed (its only teacher is out["bind"], excluded
# here), and W_bq speaks through the zero-init W_bo.
PK_LIVE = ("W_gq", "W_sil", "waist_w")
PK_DEAD = ("W_bind1", "W_bq")
PK = PK_LIVE + PK_DEAD
g_live = grads_from(lambda o: o["res"].sum(), PK)
g_rad = grads_from(lambda o: sum(x.sum() for x in o["breaths_r"]), PK)
g_dir = grads_from(lambda o: sum(x.sum() for x in o["breaths_u"]), PK)
print("[polar-smoke] grad |max| by terminal:")
for k in PK:
    print(f"      {k:10s} ({'live' if k in PK_LIVE else 'cold-dead'}) "
          f"parse-loss={g_live[k]:.3e}  breaths_r={g_rad[k]:.3e}  "
          f"breaths_u={g_dir[k]:.3e}")
assert all(g_live[k] > 0.0 for k in PK_LIVE), \
    ("GATE C2 FAIL: the parse loss does not reach the parameters through "
     "the polar state — the live terminal is broken")
assert all(g_live[k] == 0.0 for k in PK_DEAD), \
    ("GATE C2 FAIL: a cold-init-dead parameter woke — the detached deposit "
     "or the zero W_bo changed; re-read pc_row_smoke GATE 4")
assert all(g_rad[k] == 0.0 for k in PK) and all(g_dir[k] == 0.0 for k in PK), \
    ("GATE C2 FAIL: a DIAGNOSTIC tap carries gradient into the parameters "
     "— the radius/direction register is supervisable (Goodhart fence)")
print("[polar-smoke] GATE C2 PASS: two-terminal — the parse loss reaches "
      "every probed parameter THROUGH the polar state (live), and losses "
      "on breaths_r / breaths_u reach NOTHING (severed). r receives no "
      "loss gradient from any diagnostic.", flush=True)

print("[polar-smoke] ALL GATES PASS — ALG_POLAR unset is bit-identical on "
      "CPU; at birth the state is (r, u) with ‖u‖ = 1, the sextet turns "
      "the clock bands by exactly the wheel angles and leaves the content "
      "planes alone, the normalization is where-gated, and the radius is "
      "an unsupervisable diagnostic. NOT PROVEN HERE (GPU, the lead's "
      "gate): eq A/B/C bit-identity, pc_row_smoke ALL GATES, and the "
      "clock_read 0.95 breath-probe bar on a trained checkpoint.",
      flush=True)
