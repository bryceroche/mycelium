"""tok_seal_smoke.py — THE TOKEN SEAL's CPU proofs (zero GPU, zero
training, zero gradient descent; 2026-09-09, the registered TOKEN-ATTENTION
SEVERANCE PROBE).

Proves scripts/apply_tok_seal.py's contract against the STAGED source —
the patch is built in memory (apply_tok_seal.py forced to --check) and
exec'd into in-memory modules (the mask_cook_smoke.py / polar_sink_smoke.py
idiom), so the real head is NEVER written. Fixture: the banked test23
states, first 8 rows. This smoke tests MECHANISM, not skill.

  GATE 1 — ENV-INERTNESS (proof a): with the door unset the APPLIED head
     and the STAGED head are BIT-IDENTICAL on EVERY tensor-valued emission
     key x 8 rows, in FOUR configurations: {ALG_POLAR unset, the polar+
     sink env (ALG_POLAR=1 D=128 EM=0.1)} x {cold params, warm stand-in};
     parameter sets and values identical (0 params added, the init rng
     stream unmoved). Plus: ALG_TOK_SEAL=0 explicitly, and
     ALG_TOK_SEAL_S3 set either way with the main door unset, are all
     bitwise the pristine read — a companion door cannot seal anything on
     its own.
  GATE 2 — THE MISTYPED DOOR IS LOUD: ALG_TOK_SEAL=1 (the value the
     ledger's registration text uses!) raises instead of reading as OFF.
  GATE 3 — ALG_TOK_SEAL=loop (proof b): breath 0's quantities are
     BITWISE the pristine run's (fst, qst, vst, vat and the raw breath-0
     bank attention), while every LOOP breath's slot->token attention is
     exactly uniform over the row's real tokens: max|w - 1/n_tok| on real
     tokens, 0 on every pad, and every slot's row IDENTICAL (so the value
     read IS the token mean for every slot). The seal's uniform organ is
     called once per sealed road per breath (6 bank + 6 ALT21 station 3);
     ALG_TOK_SEAL_S3=0 drops it to 6 and changes the emissions — the
     narrow arm is real and the second road was live.
  GATE 4 — ALG_TOK_SEAL=all (proof c): breath 0's factor-bank attention
     is uniform too, and the VAR and QUERY banks are measured NOT uniform
     (the registered exclusion, shown rather than asserted in prose).
  GATE 5 — THE GUARD (proof d): do_train with the door set raises at its
     FIRST executable statement — proved by EXECUTION (do_train is called;
     it raises before load_alg, which is monkeypatched to a sentinel that
     is never reached) and by AST (the guard is body[0] after the imports).
     With the door unset the same call reaches the sentinel — the guard is
     a guard, not a wall.
  GATE 6 — THE APPLY SCRIPT (proof e): --check is clean and writes
     nothing, a rehearsal apply reproduces the staged source exactly, a
     second application is refused by the idempotence guard, and
     mycelium.jit_read's env-name miner finds ALG_TOK_SEAL in the applied
     source (so the JIT read key carries the door automatically).

Run from the repo root:
  .venv/bin/python3 scripts/tok_seal_smoke.py
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
           "ALG_MASK_COOK_SKEL", "ALG_MASK_SEAL", "MC_EVAL",
           "ALG_TOK_SEAL", "ALG_TOK_SEAL_S3", "ALG_MINE_BREATHS",
           "ALG_CLOCK", "ALG_JIT_READ"):
    os.environ.pop(_k, None)
os.environ["SC_EVAL"] = "0"     # the machine's OPEN regime

import numpy as np                                            # noqa: E402
from tinygrad import Tensor, dtypes                           # noqa: E402

HEAD = os.environ.get("TS_TARGET",
                      os.path.join(_ROOT, "scripts", "phase1_algebra_head.py"))
B = 8
WARM = 0.05        # the warm stand-in's scale on the zero-born doors
UNI_TOL = 1e-6     # the registered uniformity bar
ATOL = 1e-4        # cross-SCHEDULE tolerance (mask_cook_smoke's; noise ~1e-6)
POLAR_ENV = {"ALG_POLAR": "1", "ALG_POLAR_D": "128", "ALG_POLAR_EM": "0.1"}


# ===========================================================================
# STAGING — build the would-be patched source without writing the head
# ===========================================================================
def staged_source():
    argv0 = sys.argv
    sys.argv = ["apply_tok_seal.py", "--check"]
    try:
        ns = runpy.run_path(os.path.join(_ROOT, "scripts",
                                         "apply_tok_seal.py"),
                            run_name="_tok_seal_staging")
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
assert APPLIED != STAGED and "ALG_TOK_SEAL" not in APPLIED
assert "_mask_cook_skel" in APPLIED and "_polar_em" in APPLIED, \
    "the seal anchors into the head AS IT IS (mask cooker + polar sink)"
print(f"[tok-smoke] staged +{STAGED.count(chr(10)) - APPLIED.count(chr(10))}"
      f" lines on the APPLIED head; head on disk UNTOUCHED", flush=True)

M_BASE = load_module(APPLIED, "applied_plain")
M_STAG = load_module(STAGED, "staged_plain")
M_BASE_P = with_env(POLAR_ENV, lambda: load_module(APPLIED, "applied_polar"))
M_STAG_P = with_env(POLAR_ENV, lambda: load_module(STAGED, "staged_polar"))
assert not M_STAG.ALG_POLAR and M_STAG_P.ALG_POLAR and M_STAG_P.POLAR_D == 128


# ===========================================================================
# FIXTURE
# ===========================================================================
vs, vst_np, vtk, vg, vse = M_BASE.load_alg("test")
sl = np.arange(B)
TS = Tensor(vst_np[sl].astype(np.float32), dtype=dtypes.float)
TK = Tensor(vtk[sl].astype(np.float32), dtype=dtypes.float)
SE = Tensor(vse[sl].astype(np.int32), dtype=dtypes.int)
TKN = vtk[sl].astype(np.float32)                 # (B, T) the real-token mask
NTOK = TKN.sum(-1)                               # tokens per row
L_FAC = M_BASE.L_FAC
L_TOT = M_BASE.L_TOT
K_VARS = M_BASE.K_VARS
K_B = int(os.environ["ALG_BREATH"])
EYE = np.eye(L_FAC, dtype=np.float32)
_rs = np.random.RandomState(7)
MK_SPARSE = np.clip((_rs.rand(B, L_FAC, L_FAC) < 0.4).astype(np.float32)
                    + EYE, 0.0, 1.0)
assert TKN.min() == 0.0 and set(np.unique(TKN)) <= {0.0, 1.0}, \
    "tokmask must be 0/1 (the uniform's exactness rests on it)"
print(f"[tok-smoke] fixture test23 rows={B} L_FAC={L_FAC} L_TOT={L_TOT} "
      f"K_VARS={K_VARS} breaths={K_B} dev={os.environ['DEV']} | tokens per "
      f"row {NTOK.astype(int).tolist()} of T={TKN.shape[1]}", flush=True)


def cold_params(M, seed=0):
    return M.build_params(seed)


def warm_params(M, seed=0, scale=WARM):
    """The WARM STAND-IN: build_params(seed), then every ALL-ZERO parameter
    seeded from one fixed rng — the zero-born output doors (W_bo, the ALT21
    stations' output matrices, the mask head's doors, the FED gains, ...)
    are the deployed case by charter (a warm continuation from
    polarsink242) and without them whole roads are dead and a severance
    gate would 'pass' while proving nothing."""
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
    return p, n


def read(M, p, mk=None):
    """Every TENSOR-valued emission key, as numpy (all emission keys, not
    a hand-picked six)."""
    o = M.forward(p, TS, TK, SE,
                  slot_mask=Tensor(MK_SPARSE if mk is None else mk,
                                   dtype=dtypes.float))
    return {k: v.realize().numpy() for k, v in o.items()
            if isinstance(v, Tensor)}


def diff_keys(a, b):
    ks = sorted(set(a) & set(b))
    assert set(a) == set(b), f"key sets differ: {sorted(set(a) ^ set(b))}"
    return [k for k in ks if not np.array_equal(a[k], b[k])]


def maxd(a, b, keys=None):
    ks = keys or sorted(set(a) & set(b))
    return max(float(np.abs(a[k].astype(np.float64)
                            - b[k].astype(np.float64)).max()) for k in ks)


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
        n_k = 0
        for tag, pa, pb in (("cold", PA, PB), ("warm stand-in", WA, WB)):
            oa = read(MA, pa)
            ob = read(MB, pb)
            bad = diff_keys(oa, ob)
            assert not bad, (f"GATE 1 FAIL ({lab}, {tag}): NOT bit-identical "
                             f"({bad})")
            n_k = len(oa)
            n_t += len(oa)
            # the companion door cannot seal anything with the main door off
            for _extra in ({"ALG_TOK_SEAL": "0"},
                           {"ALG_TOK_SEAL_S3": "0"},
                           {"ALG_TOK_SEAL_S3": "1"},
                           {"ALG_TOK_SEAL": "0", "ALG_TOK_SEAL_S3": "0"}):
                ob2 = with_env(_extra, lambda: read(MB, pb))
                bad2 = diff_keys(oa, ob2)
                assert not bad2, (f"GATE 1 FAIL ({lab}, {tag}, {_extra}): "
                                  f"the door is not inert ({bad2})")
                n_t += len(ob2)
        return _n, n_t, n_k
    _nz, _nt, _nk = with_env(env, _g1)
    print(f"[tok-smoke] GATE 1 PASS [{lab}]: applied vs staged BIT-IDENTICAL "
          f"on {_nt} tensors ({_nk} emission keys x 8 rows x {{cold, warm}} x "
          f"{{door unset, =0, S3=0, S3=1, both}}); params identical, 0 added "
          f"(warm stand-in seeds {_nz} all-zero params)", flush=True)

P_W, _NZ = warm_params(M_STAG)
O_OPEN = read(M_STAG, P_W)
assert not diff_keys(O_OPEN, read(M_STAG, P_W)), \
    "the CPU read is nondeterministic — every bitwise claim below is void"
EKEYS = sorted(O_OPEN)
print(f"[tok-smoke] emission keys under test ({len(EKEYS)}): "
      f"{', '.join(EKEYS)}", flush=True)


# ===========================================================================
# GATE 2 — THE MISTYPED DOOR IS LOUD
# ===========================================================================
for _bad in ("1", "loops", "ALL", "true"):
    try:
        with_env({"ALG_TOK_SEAL": _bad}, lambda: read(M_STAG, P_W))
        raise SystemExit(f"GATE 2 FAIL: ALG_TOK_SEAL={_bad!r} did not raise")
    except AssertionError as e:
        assert "not one of 0 / loop / all" in str(e), \
            f"GATE 2 FAIL: wrong error for {_bad!r}: {e}"
print("[tok-smoke] GATE 2 PASS: ALG_TOK_SEAL in {'1','loops','ALL','true'} "
      "RAISES (a mistyped door must never read as OFF — note that the "
      "ledger's registration text writes the door as ALG_TOK_SEAL=1, so "
      "this is the value a reader is most likely to type)", flush=True)


# ===========================================================================
# INSTRUMENTATION (numerically neutral — proved below)
# ===========================================================================
CAP = []          # (tag, nq, flat, attention as numpy)
FLAT_CALLS = [0]
_orig_make_bank = M_STAG._make_bank
_orig_tok_flat = M_STAG._tok_flat


def _cap_make_bank(p, waist, tokmask, Bn):
    _b = _orig_make_bank(p, waist, tokmask, Bn)

    def _wrapped(queries, nq, extra=None, pbias=None, rbias=None, flat=False):
        st, at = _b(queries, nq, extra=extra, pbias=pbias, rbias=rbias,
                    flat=flat)
        CAP.append(("loop" if extra is not None else
                    ("vars" if nq == K_VARS else
                     ("query" if nq == 1 else "breath0")),
                    nq, flat, at.realize().numpy()))
        return st, at
    return _wrapped


def _cap_tok_flat(tokmask, Bn):
    FLAT_CALLS[0] += 1
    return _orig_tok_flat(tokmask, Bn)


M_STAG._make_bank = _cap_make_bank
M_STAG._tok_flat = _cap_tok_flat


def instrumented_read(env):
    CAP.clear()
    FLAT_CALLS[0] = 0
    M_STAG._STEP_TAP = {}          # the step trainer's stage-0 seam: fst,
    try:                           # qst, vst, waist and the live ctx, with
        o = with_env(env, lambda: read(M_STAG, P_W))    # no behaviour change
        tap = dict(M_STAG._STEP_TAP)
    finally:
        M_STAG._STEP_TAP = None
    return o, list(CAP), FLAT_CALLS[0], tap


# (i) the _STEP_TAP seam alone is FREE — bitwise, so fst/qst/vst read
#     through it are the pristine run's own tensors
M_STAG._STEP_TAP = {}
O_TAPONLY = read(M_STAG, P_W)
M_STAG._STEP_TAP = None
assert not diff_keys(O_OPEN, O_TAPONLY), \
    "the _STEP_TAP seam is not free — it must be a pure read"

# (ii) the bank capture realizes `at` mid-graph, which RESCHEDULES kernels;
#      that is cross-schedule, so the bar is atol, not bitwise (the head's
#      own cross-graph tolerance). Every BITWISE claim below is made
#      between two INSTRUMENTED runs, which share the schedule.
O_OPEN_I, CAP_OPEN, NF_OPEN, TAP_OPEN = instrumented_read({})
_dn = maxd(O_OPEN, O_OPEN_I)
assert _dn <= ATOL, \
    f"the instrumentation moved the machine by {_dn:.3e} > {ATOL} — every " \
    f"capture below would be describing a different machine"
assert NF_OPEN == 0, "the uniform organ ran with the door unset"
_tags = [c[0] for c in CAP_OPEN]
assert _tags == ["vars", "breath0", "query"] + ["loop"] * (K_B - 1), \
    f"unexpected bank call order {_tags}"
print(f"[tok-smoke] instrumentation: the _STEP_TAP seam is BITWISE free; "
      f"the bank capture (which realizes `at` mid-graph, forcing a "
      f"different kernel schedule) moves the emissions by {_dn:.3e} "
      f"<= {ATOL} — so every BITWISE claim below is made between two "
      f"instrumented runs. Bank calls per forward: {_tags}; _STEP_TAP "
      f"keys {sorted(TAP_OPEN)}", flush=True)


def uniformity(at):
    """(max|w - 1/n_tok| over REAL tokens, max|w| over PADS, max spread
    across slots) for a (B, nq, T) attention."""
    tgt = (TKN / NTOK.reshape(-1, 1)).reshape(B, 1, -1)
    a64 = at.astype(np.float64)
    real = float(np.abs((a64 - tgt) * TKN.reshape(B, 1, -1)).max())
    pad = float(np.abs(a64 * (1.0 - TKN.reshape(B, 1, -1))).max())
    spread = float(np.abs(a64 - a64[:, :1, :]).max())
    return real, pad, spread


# ===========================================================================
# GATE 3 — ALG_TOK_SEAL=loop (proof b)
# ===========================================================================
O_LOOP, CAP_LOOP, NF_LOOP, TAP_LOOP = instrumented_read({"ALG_TOK_SEAL": "loop"})

# (b.1) breath 0 is untouched — bitwise
for _nm in ("fst", "qst", "vst", "vst_base"):
    assert np.array_equal(TAP_OPEN[_nm].realize().numpy(),
                          TAP_LOOP[_nm].realize().numpy()), \
        f"GATE 3 FAIL: breath-0 quantity {_nm} moved under ALG_TOK_SEAL=loop"
for _i, (_t, _nq, _fl, _at) in enumerate(CAP_LOOP[:3]):
    assert not _fl and np.array_equal(_at, CAP_OPEN[_i][3]), \
        f"GATE 3 FAIL: the {_t} bank attention moved under `loop`"
assert np.array_equal(O_OPEN_I["fat"], O_LOOP["fat"]) and \
    np.array_equal(O_OPEN_I["vat"], O_LOOP["vat"]), \
    "GATE 3 FAIL: the emitted breath-0 attentions (fat/vat) moved under `loop`"
print("[tok-smoke] GATE 3a PASS: under `loop` breath 0 is BITWISE the "
      "pristine run — fst, qst, vst, vst_base (via _STEP_TAP), the raw "
      "breath-0/vars/query bank attentions, and the emitted fat/vat. The "
      "grounding is intact; only the RE-reading is severed.", flush=True)

# (b.2) every loop breath's attention is exactly uniform over real tokens
_worst = (0.0, 0.0, 0.0)
for _i, (_t, _nq, _fl, _at) in enumerate(CAP_LOOP[3:], start=1):
    assert _t == "loop" and _fl, f"GATE 3 FAIL: loop breath {_i} not sealed"
    _r, _p, _s = uniformity(_at)
    _worst = (max(_worst[0], _r), max(_worst[1], _p), max(_worst[2], _s))
    assert _r <= UNI_TOL, (f"GATE 3 FAIL: breath {_i} attention off uniform "
                           f"by {_r:.3e} > {UNI_TOL}")
    assert _p == 0.0, f"GATE 3 FAIL: breath {_i} puts {_p:.3e} on PADS"
    assert _s == 0.0, (f"GATE 3 FAIL: breath {_i} slots disagree by {_s:.3e} "
                       f"— the value read is not the token mean for EVERY slot")
assert len(CAP_LOOP) == 3 + (K_B - 1)
print(f"[tok-smoke] GATE 3b PASS: all {K_B - 1} loop breaths read the tokens "
      f"UNIFORMLY — max|w - 1/n_tok| on real tokens {_worst[0]:.3e} "
      f"(bar {UNI_TOL}), max weight on a PAD {_worst[1]:.1e} (exactly 0), "
      f"max disagreement BETWEEN slots {_worst[2]:.1e} (exactly 0: every "
      f"slot's value read is the same token mean)", flush=True)

# (b.3) the severance is live, and the second road was live too
_bad = diff_keys(O_OPEN_I, O_LOOP)
assert len(_bad) >= len(EKEYS) - 3, \
    f"GATE 3 FAIL: the seal barely moved anything (only {_bad} changed) — " \
    f"the fixture would be blind"
assert NF_LOOP == 2 * (K_B - 1), \
    (f"GATE 3 FAIL: the uniform organ ran {NF_LOOP} times, expected "
     f"{2 * (K_B - 1)} (one per sealed road per loop breath: the fq bank "
     f"and ALT21 station 3)")
O_NARROW, CAP_N, NF_N, _ = instrumented_read({"ALG_TOK_SEAL": "loop",
                                              "ALG_TOK_SEAL_S3": "0"})
assert NF_N == K_B - 1, f"GATE 3 FAIL: narrow arm called the organ {NF_N}x"
assert diff_keys(O_NARROW, O_LOOP), \
    "GATE 3 FAIL: sealing ALT21 station 3 changed nothing — either the " \
    "station is off in this env or the seal missed it"
print(f"[tok-smoke] GATE 3c PASS: the severance is LIVE — it moves "
      f"{len(_bad)}/{len(EKEYS)} emission keys (max|d| "
      f"{maxd(O_OPEN_I, O_LOOP, _bad):.4f}). The uniform organ ran "
      f"{NF_LOOP} = 2 x {K_B - 1} times (fq bank + ALT21 station 3 per loop "
      f"breath); ALG_TOK_SEAL_S3=0 drops it to {NF_N} and CHANGES the "
      f"emissions (max|d| {maxd(O_NARROW, O_LOOP):.4f}) — station 3 is a "
      f"live second slots<-tokens road, and a fq-only seal would have left "
      f"the grounding BYPASSABLE (the headroom corollary's whole point).",
      flush=True)


# ===========================================================================
# GATE 4 — ALG_TOK_SEAL=all (proof c)
# ===========================================================================
O_ALL, CAP_ALL, NF_ALL, TAP_ALL = instrumented_read({"ALG_TOK_SEAL": "all"})
_t0 = [c for c in CAP_ALL if c[0] == "breath0"]
assert len(_t0) == 1 and _t0[0][2], "GATE 4 FAIL: breath 0 was not sealed"
_r0, _p0, _s0 = uniformity(_t0[0][3])
assert _r0 <= UNI_TOL and _p0 == 0.0 and _s0 == 0.0, \
    f"GATE 4 FAIL: breath 0 attention not uniform ({_r0:.3e}, {_p0}, {_s0})"
_vars = [c for c in CAP_ALL if c[0] == "vars"][0]
_qry = [c for c in CAP_ALL if c[0] == "query"][0]
assert not _vars[2] and not _qry[2], \
    "GATE 4 FAIL: the var/query banks were sealed — registered exclusion"
_rv, _pv, _sv = uniformity(_vars[3])
assert _rv > 1e-3, \
    f"GATE 4 FAIL: the VAR bank reads uniformly anyway ({_rv:.3e}) — the " \
    f"exclusion would be meaningless and the claim about it false"
assert np.array_equal(_vars[3], CAP_OPEN[0][3]) and \
    np.array_equal(_qry[3], CAP_OPEN[2][3]), \
    "GATE 4 FAIL: the var/query bank attentions moved under `all`"
assert NF_ALL == 2 * (K_B - 1) + 1, \
    f"GATE 4 FAIL: organ ran {NF_ALL}x, expected {2 * (K_B - 1) + 1}"
assert diff_keys(O_ALL, O_LOOP), "GATE 4 FAIL: `all` == `loop`"
assert not np.array_equal(TAP_ALL["fst"].realize().numpy(),
                          TAP_OPEN["fst"].realize().numpy()), \
    "GATE 4 FAIL: fst did not move under `all`"
print(f"[tok-smoke] GATE 4 PASS: under `all` breath 0's factor bank is "
      f"uniform too (max|w - 1/n_tok| {_r0:.3e}, pads {_p0:.1e}, slot "
      f"spread {_s0:.1e}) and fst moves; the VAR bank stays live and "
      f"BITWISE pristine (its distance from uniform: {_rv:.4f}) — the "
      f"pointer target space is not deleted — as does the query bank. "
      f"`all` differs from `loop` on {len(diff_keys(O_ALL, O_LOOP))} keys "
      f"(max|d| {maxd(O_ALL, O_LOOP):.4f}); organ calls {NF_ALL}.",
      flush=True)


# ===========================================================================
# GATE 5 — THE GUARD (proof d): BY EXECUTION and BY AST
# ===========================================================================
import ast                                                     # noqa: E402
_tree = ast.parse(STAGED)
_dt = [n for n in _tree.body
       if isinstance(n, ast.FunctionDef) and n.name == "do_train"][0]
_first = [b for b in _dt.body
          if not isinstance(b, (ast.Import, ast.ImportFrom))][0]
assert isinstance(_first, ast.Assert) and \
    '_tok_seal_mode() == "0"' in (ast.get_source_segment(STAGED, _first) or ""), \
    "GATE 5 FAIL (ast): the guard is not do_train's first executable statement"


class _Sentinel(Exception):
    pass


def _sentinel_load_alg(*a, **k):
    raise _Sentinel("load_alg reached")


_real_load_alg = M_STAG.load_alg
M_STAG.load_alg = _sentinel_load_alg
try:
    for _m in ("loop", "all"):
        try:
            with_env({"ALG_TOK_SEAL": _m},
                     lambda: M_STAG.do_train(1, 1e-4, B, 0))
            raise SystemExit(f"GATE 5 FAIL: do_train ran with "
                             f"ALG_TOK_SEAL={_m}")
        except _Sentinel:
            raise SystemExit(f"GATE 5 FAIL: do_train reached load_alg with "
                             f"ALG_TOK_SEAL={_m} — the guard is too late")
        except AssertionError as e:
            assert "READ door" in str(e) and "COOKER" in str(e), \
                f"GATE 5 FAIL: wrong guard message: {e}"
    try:
        M_STAG.do_train(1, 1e-4, B, 0)
        raise SystemExit("GATE 5 FAIL: do_train did not reach load_alg with "
                         "the door unset")
    except _Sentinel:
        pass
finally:
    M_STAG.load_alg = _real_load_alg
print("[tok-smoke] GATE 5 PASS: do_train RAISES on ALG_TOK_SEAL in "
      "{loop, all} — BY EXECUTION (do_train called; it raises before "
      "load_alg, which was monkeypatched to a sentinel that is never "
      "reached, so nothing loads, nothing captures, no _quick_val can see "
      "the door) and BY AST (the guard is the first statement after the "
      "imports). With the door unset the SAME call reaches the sentinel — "
      "a guard, not a wall.", flush=True)


# ===========================================================================
# GATE 6 — THE APPLY SCRIPT (proof e)
# ===========================================================================
_py = sys.executable
_r = subprocess.run([_py, "scripts/apply_tok_seal.py", "--check"],
                    capture_output=True, text=True, cwd=_ROOT)
assert _r.returncode == 0, \
    f"GATE 6 FAIL: --check exit {_r.returncode}\n{_r.stderr[-800:]}"
assert "9 anchors OK" in _r.stdout and \
    "symtable free-var audit PASS" in _r.stdout and \
    "NOTHING written" in _r.stdout, f"GATE 6 FAIL:\n{_r.stdout}"
assert open(HEAD).read() == APPLIED, "GATE 6 FAIL: --check wrote to the head"
_tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tok_seal_rehearsal.py")
shutil.copy2(HEAD, _tmp)
_e = dict(os.environ, TS_TARGET=_tmp)
_r2 = subprocess.run([_py, "scripts/apply_tok_seal.py"],
                     capture_output=True, text=True, cwd=_ROOT, env=_e)
assert _r2.returncode == 0 and "APPLIED" in _r2.stdout, \
    f"GATE 6 FAIL: rehearsal apply\n{_r2.stderr[-800:]}"
assert open(_tmp).read() == STAGED, \
    "GATE 6 FAIL: the applied rehearsal differs from the staged source"
_r3 = subprocess.run([_py, "scripts/apply_tok_seal.py", "--check"],
                     capture_output=True, text=True, cwd=_ROOT, env=_e)
assert _r3.returncode != 0 and "idempotence" in _r3.stderr, \
    f"GATE 6 FAIL: a second application was not refused\n{_r3.stdout}{_r3.stderr}"

from mycelium.jit_read import env_names                        # noqa: E402
_names = env_names(_tmp)
assert "ALG_TOK_SEAL" in _names and "ALG_TOK_SEAL_S3" in _names, \
    f"GATE 6 FAIL: jit_read's env miner missed the door: {_names[:5]}..."
_names0 = env_names(HEAD)
assert "ALG_TOK_SEAL" not in _names0, "the pristine head must not know it"
os.remove(_tmp)
assert open(HEAD).read() == APPLIED, "GATE 6 FAIL: the head moved"
print(f"[tok-smoke] GATE 6 PASS: --check clean (9 anchors present+unique, "
      f"ast OK, structural asserts + symtable audit PASS, nothing written); "
      f"a rehearsal apply reproduces the staged source exactly and a second "
      f"application is REFUSED. mycelium.jit_read.env_names() finds "
      f"ALG_TOK_SEAL and ALG_TOK_SEAL_S3 in the applied source "
      f"({len(_names)} names, was {len(_names0)}) — the JIT read key "
      f"carries the door automatically; NO manual addition needed.",
      flush=True)

print("[tok-smoke] ALL GATES PASS — the token seal is byte-inert with its "
      "door unset (both polar configs, cold and warm, every emission key), "
      "loud when mistyped, exactly uniform over real tokens at every "
      "sealed read, breath-0-preserving under `loop` and breath-0-flat "
      "under `all`, blocked at do_train's door, and idempotence-guarded. "
      "NOT PROVEN HERE (GPU, the lead's read): any number about SKILL — "
      "the headroom itself.", flush=True)
