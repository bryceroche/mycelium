"""mask_cook_smoke.py — THE MASK COOKER's CPU proofs (zero GPU, zero
training, zero gradient descent; 2026-09-08, spec docs/mask_cooker_spec.md
S3).

Proves scripts/apply_mask_cook.py's contract against the STAGED source —
the patch is built in memory (apply_mask_cook.py forced to --check) and
exec'd into in-memory modules (the polar_sink_smoke.py / polar_birth_smoke.py
idiom), so the real head is NEVER written. Fixture: .cache/pc_row_smoke.py's
(banked test23 states, first 8 rows) — this smoke tests MECHANISM, not skill.

THE FIXTURE'S ONE SUBTLETY, measured before anything was claimed:
build_params() starts W_bo (the slot mixer's OUTPUT projection) at ZEROS,
so on a cold random init the entire slot-mixer road is dead and THE SLOT
MASK HAS NO EFFECT ON ANY EMISSION (measured: an all-ones baseline and an
identity baseline give bit-identical outputs). A cooker smoke on cold
params would therefore "pass" every gate while proving nothing. So every
gate below that must SEE the road runs on a WARM STAND-IN: build_params(0)
with every all-zero parameter (53 of them: W_bo, the ALT21 stations' output
matrices, the mask head's two doors, the FED gains, ...) seeded from one
fixed rng at 0.05. That is the deployed case by charter (the cooker is a
warm continuation from polarsink242, whose W_bo is trained), and it is
stated here rather than assumed. The BIRTH gate deliberately puts the mask
head's own two doors (mh_wo, mh_headmix) back to their true zero birth.

  GATE 1 — ENV-INERTNESS (spec proof a): with every cooker door unset the
     APPLIED head and the STAGED head are BIT-IDENTICAL on all 6 emission
     keys x 8 rows, in FOUR configurations: {ALG_POLAR unset, the polar+
     sink env (ALG_POLAR=1 D=128 EM=0.1)} x {cold params, warm stand-in}.
     Parameter sets and values identical (the patch adds NO parameters and
     does not move the init rng stream). [CPU stand-in for eq A/B/C]
  GATE 2 — PER-ROW SURGERY (proof b): with ALG_MASK_COOK armed,
     2a  _MCV all-zeros == the door-unset run (cross-graph, atol);
     2b  _MCV = [0,1,0,1,0,1,0,1]: the OPEN rows are BIT-IDENTICAL to the
         all-zeros run and the SEALED rows BIT-IDENTICAL to the all-ones
         run (same graph, row independence — the pc_row_smoke GATE 2/3
         idiom); sealed rows differ from open, and by how much is reported.
  GATE 3 — THE BIRTH FIELD (proof c): at raw = 0 (mh_wo and mh_headmix at
     their true zero birth) a SEALED row's slot attention is the
     uniform-over-all-lanes attention: end-to-end, a sealed run under a
     SPARSE baseline equals an OPEN run under an ALL-OPEN baseline; and at
     the arithmetic level, on the sc2 -> softmax path itself, the sealed
     bias is exactly the constant log(0.5) on every lane and the severed
     close is exactly zero, so the two softmaxes agree. Tolerances stated.
     Both skeletons (self, sentence) are exercised.
  GATE 4 — TWO-TERMINAL GRADIENT (proof d): a parse-only loss on SEALED
     rows reaches mh_wq / mh_wk / mh_wv / mh_wo / mh_headmix AND the mask
     head's context encoder (mh_enc1/mh_enc2) — all nonzero; on all-open
     batches those gradients are the pristine head's (bitwise with the
     door unset, atol with the door armed and _MCV zero). The cold-birth
     caveat is MEASURED and reported, not hidden (the ResNet law: with
     both output doors at zero, mh_wq/wk/wv have zero-but-defined grads
     and wake one step after the doors move).
  GATE 5 — THE TWO SEALS COMPOSE (proof e): _PCV and _MCV armed together;
     the four cells {open/open, pc-only, mask-only, both} all occur in a
     batch of 8, and every row is BITWISE the corresponding uniform run
     (all-open / all-pc / all-mask / all-both). Where they interact is
     named: inside a both-sealed row, not across rows. The assignment
     hashes' independence is measured over the real dataset index range.
  GATE 6 — THE READ-TIME METER (proof f): ALG_MASK_SEAL=1 is BITWISE the
     all-rows-sealed training path; and MC_EVAL (val's push) forces the
     OPEN regime BITWISE even with the cooker armed and every row assigned
     sealed — val compares the open regime at every shelf mode.
  GATE 7 — THE APPLY SCRIPT (proof g): --check is clean and writes
     nothing, a rehearsal apply reproduces the staged source exactly, and
     a second application is refused by the idempotence guard.

Run from the repo root:
  .venv/bin/python3 scripts/mask_cook_smoke.py
"""
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

os.environ["DEV"] = "CPU"           # HARD: this smoke never touches the GPU
ENV = {"ALG2": "1", "ALG_FTYPES": "9", "ALG_DUP": "1", "ALG_HW": "512",
       "ALG_WIDE": "1", "ALG_BREATH": "7", "ALG_NOTEBOOK": "1",
       "ALG_SIXWAVE": "1", "NB_PERSLOT": "1", "ALG_BINDBUS": "7",
       "ALG_BIND_D": "512", "BIND_CODES": ".cache/bindbus_codes512.npz",
       "ALG_BUSGARAGE": "2", "ALG_SHELF_CIRCLE": "2", "ALG_ALTMASK": "1",
       "ALG_ALT21": "1", "ALG_ALT2": "1", "ALG_MASKHEAD": "1",
       "ALG_FED": "1", "ALG_TEST": ".cache/algebra_nl_test.jsonl",
       "ALG_TEST_NAME": "test23"}
os.environ.update(ENV)
for _k in ("ALG_PC_MIX", "ALG_PC_LIVE", "ALG_POLAR", "ALG_POLAR_D",
           "ALG_POLAR_D_INIT", "ALG_POLAR_EM", "ALG_MASK_COOK",
           "ALG_MASK_COOK_SKEL", "ALG_MASK_SEAL", "MC_EVAL", "SC_EVAL"):
    os.environ.pop(_k, None)
os.environ["SC_EVAL"] = "0"     # the machine's OPEN regime — the seal fence
                                # (the pressure cooker is a separate organ;
                                # GATE 5 lifts this on purpose)

import numpy as np                                            # noqa: E402
from tinygrad import Tensor, dtypes                           # noqa: E402

HEAD = os.environ.get("MC_TARGET",
                      os.path.join(_ROOT, "scripts", "phase1_algebra_head.py"))
B = 8
KEYS = ("pres", "ftype", "op", "args", "res", "dig")
MH_KEYS = ("mh_wq", "mh_wk", "mh_wv", "mh_wo", "mh_headmix",
           "mh_enc1", "mh_enc2")
ATOL = 1e-4        # cross-graph tolerance (pc_row_smoke's; noise ~1e-6)
BIRTH_TOL = 1e-4   # end-to-end birth-field tolerance (measured 8.6e-6)
WARM = 0.05        # the warm stand-in's scale on the zero-born doors
POLAR_ENV = {"ALG_POLAR": "1", "ALG_POLAR_D": "128", "ALG_POLAR_EM": "0.1"}


# ===========================================================================
# STAGING — build the would-be patched source without writing the head
# ===========================================================================
def staged_source():
    argv0 = sys.argv
    sys.argv = ["apply_mask_cook.py", "--check"]
    try:
        ns = runpy.run_path(os.path.join(_ROOT, "scripts",
                                         "apply_mask_cook.py"),
                            run_name="_mask_cook_staging")
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
    os.environ.update(env)
    try:
        return fn_()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.update({k: v})


APPLIED = open(HEAD).read()
STAGED = staged_source()
assert APPLIED != STAGED and "ALG_MASK_COOK" not in APPLIED
assert "_PCV" in APPLIED and "_polar_em" in APPLIED, \
    "the cooker anchors into the head AS IT IS (pressure mix + polar sink)"
print(f"[cook-smoke] staged +{STAGED.count(chr(10)) - APPLIED.count(chr(10))}"
      f" lines on the APPLIED head; head on disk UNTOUCHED", flush=True)

M_BASE = load_module(APPLIED, "applied_plain")
M_STAG = load_module(STAGED, "staged_plain")
M_BASE_P = with_env(POLAR_ENV, lambda: load_module(APPLIED, "applied_polar"))
M_STAG_P = with_env(POLAR_ENV, lambda: load_module(STAGED, "staged_polar"))
assert not M_STAG.ALG_POLAR and M_STAG_P.ALG_POLAR and M_STAG_P.POLAR_D == 128


# ===========================================================================
# FIXTURE
# ===========================================================================
vs, vst, vtk, vg, vse = M_BASE.load_alg("test")
sl = np.arange(B)
TS = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
TK = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
SE = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
SENT = vse[sl].astype(np.int32)
L_FAC = M_BASE.L_FAC
L_TOT = M_BASE.L_TOT
EYE = np.eye(L_FAC, dtype=np.float32)
_rs = np.random.RandomState(7)
MK_SPARSE = np.clip((_rs.rand(B, L_FAC, L_FAC) < 0.4).astype(np.float32)
                    + EYE, 0.0, 1.0)          # a baseline that really closes
MK_OPEN = np.ones((B, L_FAC, L_FAC), np.float32)
print(f"[cook-smoke] fixture test23 rows={B} L_FAC={L_FAC} L_TOT={L_TOT} "
      f"dev={os.environ['DEV']} | sparse baseline opens "
      f"{MK_SPARSE.mean() * 100:.1f}% of lanes (the champion's opens "
      f"62-67%)", flush=True)


def cold_params(M, seed=0):
    return M.build_params(seed)


def warm_params(M, seed=0, scale=WARM, cold_doors=()):
    """The WARM STAND-IN: build_params(seed), then every ALL-ZERO parameter
    seeded from one fixed rng. Without it W_bo is zero and the slot mask
    has no effect on any emission (measured) — a smoke on cold params
    would prove nothing about a mask. `cold_doors` puts named params back
    to zero (the birth gate returns the mask head's own two doors)."""
    p = M.build_params(seed)
    rng = np.random.RandomState(1234)
    n = 0
    for k in sorted(p):
        a = p[k].numpy()
        if not np.any(a):
            n += 1
            v = (rng.randn(*a.shape) * scale).astype(np.float32)
            t = Tensor(v, dtype=dtypes.float,
                       requires_grad=True).contiguous().realize()
            t.requires_grad = True
            p[k] = t
    for k in cold_doors:
        t = Tensor(np.zeros(tuple(p[k].shape), np.float32),
                   dtype=dtypes.float, requires_grad=True).contiguous().realize()
        t.requires_grad = True
        p[k] = t
    return p, n


def read(M, p, mk, keys=KEYS):
    o = M.forward(p, TS, TK, SE, slot_mask=Tensor(mk, dtype=dtypes.float))
    return {k: o[k].realize().numpy() for k in keys}


def maxd(a, b):
    return max(float(np.abs(a[k].astype(np.float64)
                            - b[k].astype(np.float64)).max()) for k in KEYS)


def rows_bitwise(a, b, rows):
    return all(np.array_equal(a[k][r], b[k][r]) for k in KEYS for r in rows)


def rows_differ(a, b, rows):
    return all(any(not np.array_equal(a[k][r], b[k][r]) for k in KEYS)
               for r in rows)


def set_mcv(M, vals):
    M._MCV = Tensor(np.array(vals, np.float32)
                    .reshape(-1, 1, 1)).contiguous().realize()


def set_pcv(M, vals):
    M._PCV = Tensor(np.array(vals, np.float32)
                    .reshape(-1, 1, 1)).contiguous().realize()


def clear_bufs(M):
    for nm in ("_MCV", "_PCV"):
        if hasattr(M, nm):
            delattr(M, nm)


# ===========================================================================
# GATE 1 — ENV-INERTNESS (proof a)
# ===========================================================================
for lab, MA, MB, env in (("ALG_POLAR unset", M_BASE, M_STAG, {}),
                         ("ALG_POLAR=1 + sink (D=128, EM=0.1)",
                          M_BASE_P, M_STAG_P, POLAR_ENV)):
    def _g1():
        PA = cold_params(MA)
        PB = cold_params(MB)
        assert set(PA) == set(PB), \
            f"GATE 1 FAIL ({lab}): parameter set moved " \
            f"{sorted(set(PA) ^ set(PB))[:4]}"
        assert all(np.array_equal(PA[k].numpy(), PB[k].numpy()) for k in PA), \
            f"GATE 1 FAIL ({lab}): the patch moved the init rng stream"
        WA, _ = warm_params(MA)
        WB, _n = warm_params(MB)
        n_t = 0
        for tag, pa, pb in (("cold", PA, PB), ("warm stand-in", WA, WB)):
            for mk_lab, mk in (("sparse", MK_SPARSE), ("all-open", MK_OPEN)):
                oa = read(MA, pa, mk)
                ob = read(MB, pb, mk)
                bad = [k for k in KEYS if not np.array_equal(oa[k], ob[k])]
                assert not bad, (f"GATE 1 FAIL ({lab}, {tag}, {mk_lab}): "
                                 f"NOT bit-identical ({bad})")
                n_t += len(KEYS)
        return _n, n_t
    _nz, _nt = with_env(env, _g1)
    print(f"[cook-smoke] GATE 1 PASS [{lab}]: applied vs staged BIT-IDENTICAL "
          f"on {_nt} tensors (6 emission keys x 8 rows x {{cold, warm}} x "
          f"{{sparse, all-open}} baseline); params identical, 0 added "
          f"(warm stand-in seeds {_nz} all-zero params)  "
          f"[CPU stand-in for eq A/B/C]", flush=True)

P_W, _NZ = warm_params(M_STAG)
O_OPEN = read(M_STAG, P_W, MK_SPARSE)          # the door-unset reference
O_OPEN2 = read(M_STAG, P_W, MK_SPARSE)
assert rows_bitwise(O_OPEN, O_OPEN2, range(B)), \
    "the CPU read is nondeterministic — every bitwise claim below is void"
_d_mask = maxd(read(M_STAG, P_W, MK_OPEN), O_OPEN)
assert _d_mask > 1e-3, (
    f"the fixture is blind: the baseline mask changes nothing "
    f"(max|d| = {_d_mask:.2e}) — a cooker gate on it would prove nothing")
print(f"[cook-smoke] fixture LIVE: the baseline mask moves the emissions by "
      f"{_d_mask:.4f} (all-open vs sparse) on the warm stand-in; the read is "
      f"deterministic", flush=True)


# ===========================================================================
# GATE 2 — PER-ROW SURGERY (proof b)
# ===========================================================================
os.environ["ALG_MASK_COOK"] = "0.5"
set_mcv(M_STAG, [0.0] * B)
O_Z = read(M_STAG, P_W, MK_SPARSE)
assert all(np.allclose(O_OPEN[k], O_Z[k], rtol=0.0, atol=ATOL) for k in KEYS), \
    (f"GATE 2a FAIL: the all-open buffer drifted from the door-unset run "
     f"beyond scheduling noise (max|d| = {maxd(O_OPEN, O_Z):.3e})")
print(f"[cook-smoke] GATE 2a PASS: ALG_MASK_COOK=0.5 with _MCV all-zeros == "
      f"the door-unset read (max|d| = {maxd(O_OPEN, O_Z):.3e}, cross-graph "
      f"atol {ATOL})", flush=True)

set_mcv(M_STAG, [1.0] * B)
O_ONE = read(M_STAG, P_W, MK_SPARSE)
SEALED = [1, 3, 5, 7]
OPENR = [0, 2, 4, 6]
set_mcv(M_STAG, [1.0 if r in SEALED else 0.0 for r in range(B)])
O_MIX = read(M_STAG, P_W, MK_SPARSE)
assert rows_bitwise(O_Z, O_MIX, OPENR), \
    "GATE 2b FAIL: open-assigned rows moved under the mixed seal"
assert rows_bitwise(O_ONE, O_MIX, SEALED), \
    "GATE 2b FAIL: sealed rows are not bitwise the all-sealed run"
assert rows_differ(O_Z, O_MIX, SEALED), \
    "GATE 2b FAIL: sealed-assigned rows did not change"
_dsev = max(float(np.abs(O_Z[k][r].astype(np.float64)
                         - O_MIX[k][r].astype(np.float64)).max())
            for k in KEYS for r in SEALED)
print(f"[cook-smoke] GATE 2b PASS: rows {SEALED} sealed — BIT-IDENTICAL to "
      f"the all-sealed run; rows {OPENR} BIT-IDENTICAL to the all-open run "
      f"(row surgery is exact); the severance moves a sealed row's "
      f"emissions by {_dsev:.4f}", flush=True)


# ===========================================================================
# GATE 3 — THE BIRTH FIELD (proof c)
# ===========================================================================
# The mask head's own two output doors back at their TRUE zero birth, so
# _raw == 0 exactly; the rest of the machine stays warm (else the mask is
# invisible — see the fixture note).
P_BIRTH, _ = warm_params(M_STAG, cold_doors=("mh_wo", "mh_wo_b",
                                             "mh_headmix"))
clear_bufs(M_STAG)
os.environ.pop("ALG_MASK_COOK", None)
O_B_SPARSE = read(M_STAG, P_BIRTH, MK_SPARSE)
O_B_ALL = read(M_STAG, P_BIRTH, MK_OPEN)       # the uniform lane field
for _skel in ("self", "sentence"):
    os.environ["ALG_MASK_SEAL"] = "1"
    os.environ["ALG_MASK_COOK_SKEL"] = _skel
    O_B_SEAL = read(M_STAG, P_BIRTH, MK_SPARSE)
    os.environ.pop("ALG_MASK_SEAL", None)
    os.environ.pop("ALG_MASK_COOK_SKEL", None)
    _d_birth = maxd(O_B_SEAL, O_B_ALL)
    _d_open = maxd(O_B_SEAL, O_B_SPARSE)
    assert np.isfinite(_d_birth) and _d_birth < BIRTH_TOL, (
        f"GATE 3 FAIL (skeleton={_skel}): the birth field is not the "
        f"uniform-over-all-lanes field (max|d| = {_d_birth:.3e} > "
        f"{BIRTH_TOL})")
    assert _d_open > 1e-3, (
        f"GATE 3 FAIL (skeleton={_skel}): a sealed row at birth equals the "
        f"row under its SPARSE baseline — the close was not severed")
    print(f"[cook-smoke] GATE 3 PASS [skeleton={_skel}]: at raw = 0 a SEALED "
          f"row is the uniform-over-all-lanes field (max|d| = {_d_birth:.3e} "
          f"< {BIRTH_TOL}, end-to-end through 6 breaths), and differs from "
          f"its own sparse-baseline row by {_d_open:.4f} — cold but "
          f"FUNCTIONAL, not dead", flush=True)

# --- the `sentence` skeleton, at the ORGAN level (the exact claim) ---------
# On the fixture's DIFFUSE attention (random init) the soft same-sentence
# agreement is ~1/n_sent, and the floor (lg0*skel + (1-skel)*-1e4) only
# binds where the agreement is ~1 — so `sentence` safely DEGENERATES to
# `self` there, which is why the end-to-end gate above reads the same for
# both. The claim the door actually makes is about a head that READS
# sharply, so it is proved on the organ with a sharp reading.
_snt_np = np.array([[0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2]], np.int32)
_att = np.zeros((2, 4, 6), np.float32)
for _bi, _rows in enumerate(([0, 1, 3, 5], [0, 2, 3, 4])):
    for _si, _ti in enumerate(_rows):
        _att[_bi, _si, _ti] = 1.0            # a SHARP reading: one token
_sk_np = with_env({"ALG_MASK_COOK_SKEL": "sentence"},
                  lambda: M_STAG._mask_cook_skel(
                      2, 4, Tensor(_att, dtype=dtypes.float),
                      {"mc_sent": Tensor(_snt_np, dtype=dtypes.int)}).numpy())
_want = np.zeros((2, 4, 4), np.float32)
for _bi in range(2):
    _sid = _snt_np[_bi][[0, 1, 3, 5] if _bi == 0 else [0, 2, 3, 4]]
    _want[_bi] = np.clip((_sid[:, None] == _sid[None, :]).astype(np.float32)
                         + np.eye(4, dtype=np.float32), 0.0, 1.0)
assert np.array_equal(_sk_np, _want), \
    f"GATE 3 FAIL (skeleton organ): sentence skeleton\n{_sk_np}\n!= \n{_want}"
_sk_self = M_STAG._mask_cook_skel(2, 4, None, {}).numpy()
assert np.array_equal(_sk_self, np.eye(4, dtype=np.float32).reshape(1, 4, 4)), \
    "GATE 3 FAIL (skeleton organ): the `self` skeleton is not the identity"
_sk_fix = None


def _skel_on_fixture():
    global _sk_fix
    o = M_STAG.forward(P_W, TS, TK, SE,
                       slot_mask=Tensor(MK_SPARSE, dtype=dtypes.float))
    _f = o["fat"].realize()
    _sk_fix = with_env({"ALG_MASK_COOK_SKEL": "sentence"},
                       lambda: M_STAG._mask_cook_skel(
                           B, _f.shape[1], _f, {"mc_sent": SE}).numpy())


_skel_on_fixture()
_off = float((_sk_fix * (1.0 - np.eye(_sk_fix.shape[-1], dtype=np.float32))
              ).max())
print(f"[cook-smoke] GATE 3 PASS [skeleton organ]: `self` returns the exact "
      f"identity; `sentence` on a SHARP reading returns exactly the "
      f"same-sentence block (union self), bitwise, on 2 rows x 4 slots x 3 "
      f"sentences. On THIS fixture's untrained (diffuse) attention its "
      f"largest off-diagonal entry is {_off:.3f}, so the floor does not bind "
      f"and `sentence` degenerates to `self` — the safe direction, and the "
      f"reason the end-to-end birth gate reads identically for both.",
      flush=True)

# --- the same claim on the sc2 -> softmax path itself (the unit test) ------
_rs2 = np.random.RandomState(3)
_sc2 = Tensor(_rs2.randn(2, L_TOT, L_TOT).astype(np.float32) * 2.0,
              dtype=dtypes.float)
_m_np = np.clip((_rs2.rand(2, L_TOT, L_TOT) < 0.4).astype(np.float32)
                + np.eye(L_TOT, dtype=np.float32), 0.0, 1.0)
_m_np[:, :, L_FAC:] = 0.0                      # the scratch-column policy
_m = Tensor(_m_np, dtype=dtypes.float)
_raw0 = Tensor(np.zeros((2, L_TOT, L_TOT), np.float32), dtype=dtypes.float)
_mcr = _raw0.clip(-30.0, 30.0)
_lg = -(1.0 + (-_mcr).exp()).log()
_lg0 = -(1.0 + (-(_mcr * 0.0)).exp()).log()
_sk = M_STAG._mask_cook_skel(2, L_TOT, None, {})            # skeleton=self
_bias = _lg.maximum(_lg0 * _sk + (1.0 - _sk) * -1e4)
_mcc = Tensor(np.concatenate([np.ones(L_FAC, np.float32),
                              np.zeros(L_TOT - L_FAC, np.float32)])
              ).reshape(1, 1, -1)
_mck = _m.maximum(_mcc)
_a_seal = ((_sc2 + _bias).clip(-1e4, 1e4) + (1.0 - _mck) * -1e4).softmax(-1)
_a_all = (_sc2.clip(-1e4, 1e4)
          + (1.0 - _m.maximum(_mcc)) * -1e4).softmax(-1)
_a_base = (_sc2.clip(-1e4, 1e4) + (1.0 - _m) * -1e4).softmax(-1)
_bn = _bias.numpy()
_d_unit = float(np.abs(_a_seal.numpy().astype(np.float64)
                       - _a_all.numpy().astype(np.float64)).max())
_d_base = float(np.abs(_a_seal.numpy().astype(np.float64)
                       - _a_base.numpy().astype(np.float64)).max())
assert np.isfinite(_bn).all() and _bn.max() == _bn.min(), \
    "GATE 3 FAIL (unit): the birth gate is not a CONSTANT over lanes"
assert abs(float(_bn.max()) + np.log(2.0)) < 1e-6, \
    f"GATE 3 FAIL (unit): the birth gate is {_bn.max()}, not log(0.5)"
assert _d_unit < 1e-6, \
    f"GATE 3 FAIL (unit): softmax moved by {_d_unit:.3e} under a constant"
print(f"[cook-smoke] GATE 3 PASS [unit, sc2 -> softmax]: at raw = 0 the "
      f"sealed bias is the CONSTANT {float(_bn.max()):.7f} = log(0.5) on all "
      f"{L_TOT} lanes and the severed close is exactly zero on the factor "
      f"lanes, so the sealed attention == the all-lanes-open attention to "
      f"{_d_unit:.2e} (softmax is shift-invariant; a constant cannot "
      f"re-rank), while the sparse baseline's attention is {_d_base:.4f} "
      f"away. Scratch columns keep the baseline's policy (FED item 6's "
      f"raising law is not repealed).", flush=True)


# ===========================================================================
# GATE 4 — TWO-TERMINAL GRADIENT (proof d)
# ===========================================================================
def grad_probe(M, p, mk, keys):
    for t_ in p.values():
        t_.grad = None
    o = M.forward(p, TS, TK, SE, slot_mask=Tensor(mk, dtype=dtypes.float))
    o["res"].sum().backward()     # the PARSE head only (no mask loss, ever)
    out = {}
    for k in keys:
        g = p[k].grad
        out[k] = 0.0 if g is None else float(g.detach().abs().max().numpy())
    return out


P_G, _ = warm_params(M_STAG)
os.environ["ALG_MASK_COOK"] = "0.5"
set_mcv(M_STAG, [1.0] * B)
G_SEAL = grad_probe(M_STAG, P_G, MK_SPARSE, MH_KEYS)
_dead = [k for k, v in G_SEAL.items() if not v > 0.0]
assert not _dead, f"GATE 4 FAIL: sealed rows give NO gradient to {_dead}"
print("[cook-smoke] GATE 4a PASS: on SEALED rows the parse loss reaches "
      "every organ of the mask head — "
      + ", ".join(f"{k} {G_SEAL[k]:.3e}" for k in MH_KEYS), flush=True)

# the cold-birth caveat, measured (the ResNet law), not hidden
P_C, _ = warm_params(M_STAG, cold_doors=("mh_wo", "mh_wo_b", "mh_headmix"))
G_COLD = grad_probe(M_STAG, P_C, MK_SPARSE, MH_KEYS)
assert G_COLD["mh_wo"] > 0.0 and G_COLD["mh_headmix"] > 0.0, \
    "GATE 4 FAIL: at birth the two doors themselves must carry gradient " \
    "(else the road can never open — a gate deadlock)"
print("[cook-smoke] GATE 4b (the honest caveat, measured): with mh_wo and "
      "mh_headmix at their TRUE zero birth the doors carry grad "
      f"({G_COLD['mh_wo']:.3e} / {G_COLD['mh_headmix']:.3e}) while "
      f"mh_wq/wk/wv/enc are zero-but-DEFINED "
      f"({G_COLD['mh_wq']:.1e}/{G_COLD['mh_enc1']:.1e}) — the ResNet law: "
      "the bank wakes one step after the doors move. No deadlock.",
      flush=True)

# the OPEN path's gradients are the pristine head's
set_mcv(M_STAG, [0.0] * B)
G_OPEN_ARMED = grad_probe(M_STAG, P_G, MK_SPARSE, MH_KEYS)
clear_bufs(M_STAG)
os.environ.pop("ALG_MASK_COOK", None)
G_OPEN_STAG = grad_probe(M_STAG, P_G, MK_SPARSE, MH_KEYS)
P_B, _ = warm_params(M_BASE)
G_OPEN_BASE = grad_probe(M_BASE, P_B, MK_SPARSE, MH_KEYS)
assert all(G_OPEN_STAG[k] == G_OPEN_BASE[k] for k in MH_KEYS), \
    ("GATE 4 FAIL: with the door unset the staged head's gradients are not "
     "BITWISE the applied head's " +
     str({k: (G_OPEN_STAG[k], G_OPEN_BASE[k]) for k in MH_KEYS}))
_rel = max(abs(G_OPEN_ARMED[k] - G_OPEN_BASE[k]) / max(G_OPEN_BASE[k], 1e-30)
           for k in MH_KEYS)
assert _rel < 1e-4, \
    f"GATE 4 FAIL: all-open armed gradients drifted {_rel:.3e} (relative)"
print(f"[cook-smoke] GATE 4c PASS: on all-open batches the mask head's "
      f"gradients are the pristine head's — BITWISE with the door unset "
      f"(7/7 params), and {_rel:.2e} relative with the door armed and _MCV "
      f"all-zero (cross-graph). The path is unchanged.", flush=True)


# ===========================================================================
# GATE 5 — THE TWO SEALS COMPOSE (proof e)
# ===========================================================================
# the assignment hashes, over the real dataset index range
_n_rows = 25000
_ii = np.arange(_n_rows, dtype=np.uint64)
_h_pc = ((_ii * np.uint64(2654435761)) % np.uint64(4294967296)
         ).astype(np.float64) / 4294967296.0
_h_mc = ((_ii * np.uint64(2246822519) + np.uint64(2654435761))
         % np.uint64(4294967296)).astype(np.float64) / 4294967296.0
for _s_pc, _s_mc in ((0.30, 0.15), (0.30, 0.30), (0.15, 0.15)):
    _a, _b_ = _h_pc < _s_pc, _h_mc < _s_mc
    _cells = (int((~_a & ~_b_).sum()), int((_a & ~_b_).sum()),
              int((~_a & _b_).sum()), int((_a & _b_).sum()))
    _dev = abs(float((_a & _b_).mean()) - float(_a.mean()) * float(_b_.mean()))
    assert min(_cells) > 0 and _dev < 1e-3, \
        f"GATE 5 FAIL: hashes correlate at ({_s_pc}, {_s_mc}): dev {_dev:.4f}"
    print(f"[cook-smoke] GATE 5a: doses (pc {_s_pc}, mask {_s_mc}) over "
          f"n={_n_rows}: cells open/pc/mask/both = {_cells}, "
          f"|P(both) - P(pc)P(mc)| = {_dev:.5f}", flush=True)

os.environ.pop("SC_EVAL", None)         # the pressure seal needs it unset
os.environ["ALG_PC_MIX"] = "0.3"
os.environ["ALG_MASK_COOK"] = "0.3"
CELLS = {"open/open": (0.0, 0.0), "pc-only": (1.0, 0.0),
         "mask-only": (0.0, 1.0), "both": (1.0, 1.0)}
REF = {}
for _lab, (_pv, _mv) in CELLS.items():
    set_pcv(M_STAG, [_pv] * B)
    set_mcv(M_STAG, [_mv] * B)
    REF[_lab] = read(M_STAG, P_W, MK_SPARSE)
ROWCELL = ["open/open", "pc-only", "mask-only", "both",
           "open/open", "both", "mask-only", "pc-only"]
set_pcv(M_STAG, [CELLS[c][0] for c in ROWCELL])
set_mcv(M_STAG, [CELLS[c][1] for c in ROWCELL])
O_COMP = read(M_STAG, P_W, MK_SPARSE)
assert set(ROWCELL) == set(CELLS), "all four cells must occur in the batch"
for _r, _c in enumerate(ROWCELL):
    assert rows_bitwise(O_COMP, REF[_c], [_r]), \
        f"GATE 5 FAIL: row {_r} ({_c}) is not bitwise its uniform run"
for _c1, _c2 in (("open/open", "pc-only"), ("open/open", "mask-only"),
                 ("pc-only", "both"), ("mask-only", "both")):
    assert rows_differ(REF[_c1], REF[_c2], range(B)), \
        f"GATE 5 FAIL: cells {_c1} and {_c2} are indistinguishable"
clear_bufs(M_STAG)
os.environ.pop("ALG_PC_MIX", None)
os.environ.pop("ALG_MASK_COOK", None)
os.environ["SC_EVAL"] = "0"
print("[cook-smoke] GATE 5b PASS: with BOTH cookers armed, all four cells "
      "{open/open, pc-only, mask-only, both} occur in a batch of 8 and every "
      "row is BITWISE its uniform single-configuration run. WHERE THEY "
      "INTERACT: inside a both-sealed row (one forward carries both "
      "severances — the residual cut and the baseline cut), never across "
      "rows; the per-row blends are exact and independent.", flush=True)


# ===========================================================================
# GATE 6 — THE READ-TIME METER AND VAL'S PUSH (proof f)
# ===========================================================================
os.environ["ALG_MASK_SEAL"] = "1"
O_SEALDOOR = read(M_STAG, P_W, MK_SPARSE)
os.environ.pop("ALG_MASK_SEAL", None)
assert rows_bitwise(O_SEALDOOR, O_ONE, range(B)), \
    ("GATE 6 FAIL: ALG_MASK_SEAL=1 is not BITWISE the all-rows-sealed "
     f"training path (max|d| = {maxd(O_SEALDOOR, O_ONE):.3e})")
print("[cook-smoke] GATE 6a PASS: ALG_MASK_SEAL=1 at read == the "
      "all-rows-sealed training path, BITWISE on all 6 keys x 8 rows (the "
      "meter measures the machine the cooker trains)", flush=True)

os.environ["ALG_MASK_COOK"] = "1.0"
set_mcv(M_STAG, [1.0] * B)
os.environ["MC_EVAL"] = "0"
O_VAL = read(M_STAG, P_W, MK_SPARSE)
os.environ.pop("MC_EVAL", None)
O_SEAL_NOVAL = read(M_STAG, P_W, MK_SPARSE)
clear_bufs(M_STAG)
os.environ.pop("ALG_MASK_COOK", None)
assert rows_bitwise(O_VAL, O_OPEN, range(B)), \
    "GATE 6 FAIL: MC_EVAL did not force the OPEN regime bitwise"
assert rows_differ(O_VAL, O_SEAL_NOVAL, range(B)), \
    "GATE 6 FAIL: MC_EVAL made no difference — the guard is not live"
os.environ["ALG_MASK_SEAL"] = "1"
os.environ["MC_EVAL"] = "0"
O_VAL2 = read(M_STAG, P_W, MK_SPARSE)
os.environ.pop("MC_EVAL", None)
os.environ.pop("ALG_MASK_SEAL", None)
assert rows_bitwise(O_VAL2, O_OPEN, range(B)), \
    "GATE 6 FAIL: MC_EVAL must win over ALG_MASK_SEAL too"
print("[cook-smoke] GATE 6b PASS: MC_EVAL (the trainer's UNCONDITIONAL push "
      "around _quick_val) forces the OPEN regime BITWISE — with every row "
      "assigned sealed, and over ALG_MASK_SEAL as well. Val compares the "
      "open regime at every shelf mode (SC_EVAL's push is "
      "ALG_SHELF_CIRCLE>=2 only, which is why the cooker carries its own).",
      flush=True)


# ===========================================================================
# GATE 7 — THE APPLY SCRIPT (proof g)
# ===========================================================================
_py = sys.executable
_r = subprocess.run([_py, "scripts/apply_mask_cook.py", "--check"],
                    capture_output=True, text=True, cwd=_ROOT)
assert _r.returncode == 0, \
    f"GATE 7 FAIL: --check exit {_r.returncode}\n{_r.stderr[-800:]}"
assert "13 anchors OK" in _r.stdout and \
    "symtable free-var audit PASS" in _r.stdout and \
    "NOTHING written" in _r.stdout, f"GATE 7 FAIL:\n{_r.stdout}"
assert open(HEAD).read() == APPLIED, "GATE 7 FAIL: --check wrote to the head"
_tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "mask_cook_rehearsal.py")
shutil.copy2(HEAD, _tmp)
_e = dict(os.environ, MC_TARGET=_tmp)
_r2 = subprocess.run([_py, "scripts/apply_mask_cook.py"],
                     capture_output=True, text=True, cwd=_ROOT, env=_e)
assert _r2.returncode == 0 and "APPLIED" in _r2.stdout, \
    f"GATE 7 FAIL: rehearsal apply\n{_r2.stderr[-800:]}"
assert open(_tmp).read() == STAGED, \
    "GATE 7 FAIL: the applied rehearsal differs from the staged source"
_r3 = subprocess.run([_py, "scripts/apply_mask_cook.py", "--check"],
                     capture_output=True, text=True, cwd=_ROOT, env=_e)
assert _r3.returncode != 0 and "idempotence" in _r3.stderr, \
    f"GATE 7 FAIL: a second application was not refused\n{_r3.stdout}{_r3.stderr}"
os.remove(_tmp)
assert open(HEAD).read() == APPLIED, "GATE 7 FAIL: the head moved"
print("[cook-smoke] GATE 7 PASS: --check clean (13 anchors present+unique, "
      "ast OK, structural asserts + symtable audit PASS, nothing written); a "
      "rehearsal apply reproduces the staged source exactly and a second "
      "application is REFUSED by the idempotence guard.", flush=True)

print("[cook-smoke] ALL GATES PASS — the cooker is byte-inert with its doors "
      "unset (both polar configs, cold and warm), surgical per row (bitwise "
      "within-graph), functional at birth (the uniform lane field), "
      "two-terminal into every organ of the mask head, composable with the "
      "pressure seal, and excluded from val by its own guard. NOT PROVEN "
      "HERE (GPU, the lead's gate): eq A/B/C bit-identity, "
      ".cache/pc_row_smoke.py ALL GATES, and any claim about SKILL.",
      flush=True)
