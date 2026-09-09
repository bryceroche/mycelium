"""census_organs2_smoke.py — THE PORT CENSUS PHASE-2 SMOKE (CPU, zero
GPU, zero training, zero gradient descent; 2026-09-08).

Proves scripts/apply_census_organs2.py's contract against the STAGED
source. The BASE is the head as it stands with phase 1 applied — taken
from disk if the lead has already applied apply_census_organs.py, and
staged in memory from apply_census_organs.py otherwise, so this smoke is
correct on both sides of that apply. Phase 2 is then staged ON TOP of
that base via PS_TARGET pointing at a temp copy. The real head is NEVER
written. Fixture: .cache/pc_row_smoke.py's (banked test23 states, first
8 rows, build_params(0) random init: MECHANISM, not skill).

  GATE 1 — INERTNESS WITH THE HOOK UNARMED: base vs staged, same 8 rows,
     np.array_equal on EVERY emission key. Run twice: champion regime
     (ALG_POLAR unset) and the polar sink regime (ALG_POLAR=1,
     ALG_POLAR_D=128, ALG_POLAR_EM=0.1).
  GATE 2 — ARMING IS CROSS-GRAPH, and phase 2 adds no new kind of
     disturbance: the BASE head's own armed-vs-unarmed deviation is the
     baseline, the staged head's must be the same order (both <= ATOL).
  GATE 3 — COVERAGE, in THREE regimes: champion (NB_PERSLOT=1), the
     polar sink, and NB_PERSLOT=0 — the third exists because the gen-1
     hook only wired the per-slot arm of the notebook read, so a
     blurred-lane config censused the DOMINANT PORT as absent. All three
     must record `notebook`; the two lane-2 organs and the two ALT21
     stations must appear wherever their organ exists.
  GATE 4 — GAIN LEGIBILITY, and its absence stated: with fed_nb_g poked
     off zero, rms(notebook2) == |fed_nb_g| * rms(notebook2_pre) to f32.
     ALT21 stations 3-4 have NO scalar gain (their door is the zero-init
     output matrix), so the gate asserts the opposite claim: no
     `alt21_*_pre` organ is emitted anywhere, and with the output
     matrices poked the two stations are nonzero and distinct.
  GATE 5 — the apply script's --check: anchors present and unique, ast
     OK, structural asserts + symtable audit PASS, the head on disk is
     BYTE-UNCHANGED, and a second application is refused (idempotence).

Run from the repo root:
  .venv/bin/python3 scripts/census_organs2_smoke.py
"""
import hashlib
import os
import runpy
import subprocess
import sys
import tempfile
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

os.environ["DEV"] = "CPU"             # HARD: this smoke never touches the GPU
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
           "ALG_POLAR_D_INIT", "ALG_POLAR_EM", "ALG_MINE_BREATHS"):
    os.environ.pop(_k, None)
os.environ["SC_EVAL"] = "0"           # the machine's living regime — SEAL FENCE

import numpy as np                                              # noqa: E402
from tinygrad import Tensor, dtypes                             # noqa: E402

HEAD = os.environ.get("PS_TARGET",
                      os.path.join(_ROOT, "scripts", "phase1_algebra_head.py"))
B = 8
D_WIDTH = 128
KAPPA = 0.1
ATOL = 1e-4        # the pc_row_smoke cross-graph convention (logit scale ~1)
POLAR_ENV = {"ALG_POLAR": "1", "ALG_POLAR_D": str(D_WIDTH),
             "ALG_POLAR_EM": str(KAPPA)}
BLUR_ENV = {"NB_PERSLOT": "0"}
NEW_ORGANS = ("alt21_s3", "alt21_s4", "notebook2", "notebook2_pre")
PHASE1_ORGANS = ("maskhead", "maskhead_pre", "alt", "alt_pre",
                 "mixer", "mixer_pre", "altfact", "altfact_pre")
BASELINES = ("state", "state_slot", "state_hslot")
_TD = tempfile.TemporaryDirectory()


# ===========================================================================
# STAGING — phase 1 (from disk or in memory), then phase 2 on top of it
# ===========================================================================
def stage(script, target):
    argv0 = sys.argv
    old = os.environ.get("PS_TARGET")
    sys.argv = [script, "--check"]
    os.environ["PS_TARGET"] = target
    try:
        ns = runpy.run_path(os.path.join(_ROOT, "scripts", script),
                            run_name="_census2_staging")
    finally:
        sys.argv = argv0
        if old is None:
            os.environ.pop("PS_TARGET", None)
        else:
            os.environ["PS_TARGET"] = old
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
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


DISK = open(HEAD).read()
if '"maskhead"' in DISK:
    BASE = DISK
    _origin = "the head on disk (phase 1 already applied)"
else:
    BASE = stage("apply_census_organs.py", HEAD)
    _origin = "phase 1 staged in memory (the head on disk predates it)"
BASE_FILE = os.path.join(_TD.name, "phase1_algebra_head.py")
open(BASE_FILE, "w").write(BASE)
STAGED = stage("apply_census_organs2.py", BASE_FILE)
assert BASE != STAGED and '"alt21_s3"' not in BASE
print(f"[organ2-smoke] base = {_origin}; staged "
      f"+{STAGED.count(chr(10)) - BASE.count(chr(10))} lines on it; head on "
      f"disk UNTOUCHED", flush=True)

M_BASE = load_module(BASE, "base_plain")
M_STAG = load_module(STAGED, "staged_plain")
M_BASE_P = with_env(POLAR_ENV, lambda: load_module(BASE, "base_polar"))
M_STAG_P = with_env(POLAR_ENV, lambda: load_module(STAGED, "staged_polar"))
M_STAG_B = with_env(BLUR_ENV, lambda: load_module(STAGED, "staged_blur"))
assert not M_STAG.ALG_POLAR and M_STAG_P.ALG_POLAR
assert M_STAG.NB_PERSLOT and not M_STAG_B.NB_PERSLOT, \
    "the blurred-lane regime did not take (NB_PERSLOT)"
assert M_STAG.FED_SHELF, "FED_SHELF is off — lane 2 would not exist to census"


# ===========================================================================
# FIXTURE — the pc_row_smoke idiom, plus loop_val's pass-1 conditioning
# ===========================================================================
vs, vst, vtk, vg, vse = M_BASE.load_alg("test")
sl = np.arange(B)
TS = Tensor(vst[sl].astype(np.float32), dtype=dtypes.float)
TK = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
SE = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
SENT = vse[sl].astype(np.int32)
NV = np.array([vs[int(i)].get("n_vars", M_BASE.K_VARS) for i in sl])
MA = np.array([vs[int(i)].get("m", 0) for i in sl])
print(f"[organ2-smoke] fixture test23 rows={B} dev={os.environ['DEV']} "
      f"d={D_WIDTH} kappa={KAPPA}", flush=True)


def masked_read(M, p):
    """loop_val's two-pass shape, WITH the fact conditioning pass 1 gets in
    the deployed read path (ALG_ALT2)."""
    o0 = M.forward(p, TS, TK, SE)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk = M.build_slot_masks(onp0, SENT)
    fact_t = None
    if int(os.environ.get("ALG_ALT2", "0")):
        _ka = ("pres", "ftype", "op", "dig") + \
            (("dup",) if "dup" in o0 else ())
        _oa = {**onp0, **{k: o0[k].realize().numpy() for k in _ka}}
        fact_t = Tensor(M.alt2_fact_buf(_oa, SENT, NV, MA), dtype=dtypes.float)
    o = M.forward(p, TS, TK, SE, slot_mask=Tensor(mk, dtype=dtypes.float),
                  fact_buf=fact_t)
    return flat(o)


def flat(o):
    out = {}
    for k, v in o.items():
        if isinstance(v, list):
            for j, e in enumerate(v):
                if isinstance(e, dict):
                    for k2, v2 in e.items():
                        out[f"{k}[{j}].{k2}"] = v2.realize().numpy()
                else:
                    out[f"{k}[{j}]"] = e.realize().numpy()
        else:
            out[k] = v.realize().numpy()
    return out


def dev_of(a, b):
    return max(float(np.abs(a[k] - b[k]).max()) for k in a
               if np.issubdtype(a[k].dtype, np.floating))


# ===========================================================================
# GATE 1 — INERTNESS WITH THE HOOK UNARMED (both polar configs)
# ===========================================================================
for lab, MA_, MB_, env in (
        ("champion regime, ALG_POLAR unset", M_BASE, M_STAG, {}),
        (f"polar sink regime (D={D_WIDTH}, EM={KAPPA})",
         M_BASE_P, M_STAG_P, POLAR_ENV)):
    def _run():
        pa = MA_.build_params(0)
        pb = MB_.build_params(0)
        assert set(pa) == set(pb), f"GATE 1 FAIL ({lab}): parameter set changed"
        assert all(np.array_equal(pa[k].numpy(), pb[k].numpy()) for k in pa), \
            f"GATE 1 FAIL ({lab}): the patch moved the init rng stream"
        MA_._CENSUS = None
        MB_._CENSUS = None
        return masked_read(MA_, pa), masked_read(MB_, pb)
    oa, ob = with_env(env, _run)
    assert set(oa) == set(ob), f"GATE 1 FAIL ({lab}): emission keys differ"
    bad = [k for k in oa if not np.array_equal(oa[k], ob[k])]
    assert not bad, f"GATE 1 FAIL ({lab}): NOT bit-identical ({bad[:6]})"
    print(f"[organ2-smoke] GATE 1 PASS [{lab}]: base vs staged BIT-IDENTICAL "
          f"on {len(oa)} emission tensors, hook UNARMED; params identical",
          flush=True)


# ===========================================================================
# GATE 2 + 3 — ARMING IS CROSS-GRAPH; COVERAGE IN THREE REGIMES
# ===========================================================================
def census_read(M, p, env):
    def _run():
        M._CENSUS = None
        base = masked_read(M, p)
        M._CENSUS = []
        armed = masked_read(M, p)
        rec = M._CENSUS
        M._CENSUS = None
        return base, armed, rec
    return with_env(env, _run)


_p1 = M_BASE.build_params(0)
_b1, _a1, _r1 = census_read(M_BASE, _p1, {})
DEV_BASE = dev_of(_b1, _a1)
assert DEV_BASE <= ATOL, \
    f"GATE 2 FAIL: the BASE head already deviates {DEV_BASE:.3g} > {ATOL}"
print(f"[organ2-smoke] GATE 2 baseline [phase-1 head]: arming moves the "
      f"emissions by {DEV_BASE:.3g} max-abs; {len(_r1)} records, "
      f"{len({o for _, o, _ in _r1})} organs.", flush=True)

REGIMES = (("champion, NB_PERSLOT=1", M_STAG, {}),
           (f"polar sink (D={D_WIDTH}, EM={KAPPA})", M_STAG_P, POLAR_ENV),
           ("champion, NB_PERSLOT=0 (the BLURRED lanes)", M_STAG_B, BLUR_ENV))
for lab, M, env in REGIMES:
    p = with_env(env, lambda: M.build_params(0))
    base, armed, rec = census_read(M, p, env)
    dev = dev_of(base, armed)
    assert dev <= ATOL, \
        (f"GATE 2 FAIL ({lab}): arming moves the forward by {dev:.3g} > "
         f"{ATOL} (base head {DEV_BASE:.3g})")
    seen = {}
    for (kb, organ, arr) in rec:
        assert np.isfinite(arr).all(), \
            f"GATE 3 FAIL ({lab}): {organ} at b{kb} is not finite"
        seen.setdefault(organ, []).append((kb, arr))
    print(f"[organ2-smoke] GATE 2 PASS [{lab}]: arming moves the emissions "
          f"by {dev:.3g} max-abs over {len(base)} tensors — same order as "
          f"the base head's {DEV_BASE:.3g}, both << {ATOL}", flush=True)

    # THE BLIND SPOT THIS PATCH CLOSES: the notebook must be censused in
    # EVERY regime, per-slot or blurred. Before phase 2 the blurred arm
    # reported the dominant port as absent.
    assert "notebook" in seen, \
        f"GATE 3 FAIL ({lab}): the notebook is not censused in this regime"
    for o in NEW_ORGANS + PHASE1_ORGANS:
        assert o in seen, f"GATE 3 FAIL ({lab}): organ {o} never recorded"
    for o in BASELINES[:2]:
        assert o in seen, f"GATE 3 FAIL ({lab}): baseline {o} missing"
    assert "state_hslot" in seen, \
        f"GATE 3 FAIL ({lab}): the h_slot baseline is missing"
    # bands: alt21 + lane 2 are STATE band, and h_slot-shaped where it counts
    for o in ("alt21_s3", "alt21_s4", "state_hslot"):
        for kb, a in seen[o]:
            assert a.shape == (B, M.L_TOT, M.H_W), \
                f"GATE 3 FAIL ({lab}): {o} {a.shape} is not the h_slot band"
    for o in ("notebook2", "notebook2_pre", "notebook"):
        for kb, a in seen[o]:
            assert a.ndim == 3 and a.shape[-1] == M.H_W, \
                f"GATE 3 FAIL ({lab}): {o} {a.shape} is not state band"
    _nbshape = sorted({a.shape[1] for _, a in seen["notebook"]})
    _arm = ("per-slot" if M.NB_PERSLOT
            else "BLURRED — the arm gen-1 never hooked")
    print(f"[organ2-smoke] GATE 3 PASS [{lab}]: {len(seen)} organs "
          f"({', '.join(sorted(seen))}); notebook slot-axis {_nbshape} "
          f"({_arm}); all finite, all in-band", flush=True)


# ===========================================================================
# GATE 4 — GAIN LEGIBILITY, AND ITS PRINCIPLED ABSENCE
# ===========================================================================
M = M_STAG
p4 = M.build_params(0)
_rs = np.random.RandomState(9082)


def _poke(key, scale):
    """Open a zero-init door BY HAND (no training, no gradient) so the
    organ has something to say. Mechanism, not skill."""
    assert key in p4, f"GATE 4 FAIL: {key} does not exist in this config"
    sh = tuple(int(d) for d in p4[key].shape)
    p4[key].assign(Tensor((_rs.randn(*sh) * scale).astype(np.float32),
                          dtype=dtypes.float)).realize()


for _k, _sc in (("fed_nb_g", 0.25), ("alt21_attn_wo", 0.02),
                ("alt21_W_bo", 0.02), ("W_bo", 0.05)):
    _poke(_k, _sc)
_b4, _a4, _rec4 = census_read(M, p4, {})
seen4 = {}
for (kb, organ, arr) in _rec4:
    seen4.setdefault(organ, {})[kb] = arr

# (a) lane 2 HAS a scalar gain — the usual two-coordinate identity
_g = float(np.abs(p4["fed_nb_g"].numpy()).reshape(-1)[0])
_live = 0
for kb, post in seen4["notebook2"].items():
    pre = seen4["notebook2_pre"][kb]
    rq = float(np.sqrt((pre ** 2).mean()))
    rp = float(np.sqrt((post ** 2).mean()))
    if rq == 0.0:
        assert float(np.abs(post).max()) == 0.0, \
            f"GATE 4 FAIL: notebook2 b{kb} post nonzero on a zero pre"
        continue
    assert abs(rp - _g * rq) <= 1e-5 * max(rp, 1e-6) + 1e-9, \
        (f"GATE 4 FAIL: notebook2 b{kb} rms(post)={rp:.6g} != "
         f"fed_nb_g={_g:.6g} * rms(pre)={rq:.6g}")
    _live += 1
assert _live > 0, "GATE 4 FAIL: lane 2 never spoke"
print(f"[organ2-smoke] GATE 4 PASS [notebook2]: rms(post) == fed_nb_g "
      f"({_g:.4g}) x rms(pre) on {_live} breath(s)", flush=True)

# (b) ALT21 has NO scalar gain — assert the ABSENCE, do not fake a ratio
assert not [o for o in seen4 if o.startswith("alt21") and o.endswith("_pre")], \
    ("GATE 4 FAIL: an alt21 _pre organ was emitted — stations 3-4 ride "
     "zero-init OUTPUT MATRICES, not a scalar gain; a _pre there is a fake "
     "ratio in a different space")
assert "alt21_s3_pre" not in STAGED and "alt21_s4_pre" not in STAGED
_n3 = {kb: float(np.sqrt((a ** 2).mean())) for kb, a in seen4["alt21_s3"].items()}
_n4 = {kb: float(np.sqrt((a ** 2).mean())) for kb, a in seen4["alt21_s4"].items()}
assert all(v > 0 for v in _n3.values()) and all(v > 0 for v in _n4.values()), \
    "GATE 4 FAIL: an ALT21 station is silent with its output matrix poked"
assert _n3 != _n4, "GATE 4 FAIL: the two stations are indistinguishable"
_hs = {kb: float(np.sqrt((a ** 2).mean()))
       for kb, a in seen4["state_hslot"].items()}
print(f"[organ2-smoke] GATE 4 PASS [alt21_s3 / alt21_s4]: NO scalar gain "
      f"on this path (asserted: no _pre organ anywhere). With the zero-init "
      f"output matrices poked, both stations are live and distinct; "
      f"rms/state_hslot at b6 = {_n3[max(_n3)] / _hs[max(_hs)]:.4g} / "
      f"{_n4[max(_n4)] / _hs[max(_hs)]:.4g}", flush=True)


# ===========================================================================
# GATE 5 — the apply script's --check, and idempotence
# ===========================================================================
_h0 = hashlib.sha256(open(HEAD, "rb").read()).hexdigest()
r = subprocess.run([sys.executable, "scripts/apply_census_organs2.py",
                    "--check"], capture_output=True, text=True,
                   env=dict(os.environ, PS_TARGET=BASE_FILE))
assert r.returncode == 0, f"GATE 5 FAIL: --check errored\n{r.stdout}{r.stderr}"
assert "NOTHING written" in r.stdout, "GATE 5 FAIL: --check did not say so"
assert hashlib.sha256(open(HEAD, "rb").read()).hexdigest() == _h0, \
    "GATE 5 FAIL: --check WROTE to the head"
assert hashlib.sha256(open(BASE_FILE, "rb").read()).hexdigest() == \
    hashlib.sha256(BASE.encode()).hexdigest(), \
    "GATE 5 FAIL: --check WROTE to the staged base"
_twice = os.path.join(_TD.name, "twice.py")
open(_twice, "w").write(STAGED)
r2 = subprocess.run([sys.executable, "scripts/apply_census_organs2.py",
                     "--check"], capture_output=True, text=True,
                    env=dict(os.environ, PS_TARGET=_twice))
assert r2.returncode != 0 and "idempotence" in r2.stderr, \
    f"GATE 5 FAIL: a second application was not refused\n{r2.stderr[-400:]}"
print(f"[organ2-smoke] GATE 5 PASS: --check writes nothing (head sha "
      f"{_h0[:12]} unchanged); a second application is REFUSED "
      f"(idempotence guard)", flush=True)

print("[organ2-smoke] RESIDUE after phase 2 (additive injections still "
      "uncensused): the sync receiver oscillator (ALG_SYNC), the BEXIT "
      "commit-mass bias (ALG_BEXIT), the per-forward biases (sixwave "
      "sw_g, pmask, FED waist2), the _IMP systems-ID kick, and the state "
      "REPLACEMENTS (stellar / circle / the pressure seal), which are "
      "not injections.")
print("[organ2-smoke] ALL GATES PASS — CPU only, no GPU, no training, "
      "head on disk untouched")
