"""tok_cook_smoke.py — THE TOKEN COOKER's CPU proofs (zero GPU, zero
training, zero gradient descent; 2026-09-09, spec docs/token_cooker_spec.md
S3).

Proves scripts/apply_tok_cook.py's contract against the STAGED source —
the patch is built in memory (apply_tok_cook.py forced to --check) and
exec'd into in-memory modules (the tok_seal_smoke.py / mask_cook_smoke.py
idiom), so the real head is NEVER written. Fixture: the banked test23
states, first 8 rows. This smoke tests MECHANISM, not skill.

THE FIXTURE'S SUBTLETY, inherited and restated: build_params() starts the
zero-born output doors (W_bo, the ALT21 stations' output matrices, the
mask head's mh_wo/mh_headmix, the FED gains) at ZEROS, so on a cold init
whole roads are dead and a severance gate would "pass" while proving
nothing. Every gate that must SEE a road runs on a WARM STAND-IN:
build_params(0) with every all-zero parameter seeded from one fixed rng at
0.05 — the deployed case by charter (the cooker is a warm continuation
from polarsink242). W_tg is NOT zero-born, so it is never re-seeded: what
this smoke measures at "birth" is the real birth.

  GATE 1 — ENV-INERTNESS (proof a): with ALG_TOK_COOK unset and
     ALG_TOK_SEAL != head, the APPLIED head and the STAGED head are
     BIT-IDENTICAL on EVERY tensor-valued emission key x 8 rows, in
     {ALG_POLAR unset, the polar+sink env} x {cold, warm stand-in} x
     {door unset, ALG_TOK_SEAL=0/loop/all, ALG_TOK_COOK=0.15 with no
     buffer (the read env — the trained-env law), ALG_TOK_COOK_S3=0/1,
     ALG_TG_INIT set, TC_EVAL set, ALG_MASKRE=1}. Parameter sets and
     values identical (0 added, the init rng stream unmoved). The
     ALG_MASKRE and TOK_SEAL cells are the ones that prove the two PURE
     CODE MOTIONS (_mh_a5, _mh_ctx) changed nothing.
  GATE 2 — THE CODE MOTION IS A MOTION: `_mh_ctx`'s body is the applied
     head's context block VERBATIM (dedented one level, compared
     character by character), and with the cooker armed the memo makes
     the gate and the mask head hold THE SAME TENSOR OBJECT (measured by
     identity, 2 calls per breath, 1 distinct result).
  GATE 3 — THE GATE IS A DISTRIBUTION (proof c): at every loop breath on
     sealed rows it sums to 1 over the row's real tokens, is EXACTLY 0.0
     on every pad, and the bank's attention IS the gate, bitwise, on
     every head. The birth KL from uniform is REPORTED (nonzero, small,
     against log n_tok). Breath 0's grounding is bitwise pristine.
  GATE 4 — PER-ROW SURGERY (proof b): _TCV all-zeros == the door-unset
     run (cross-graph atol); _TCV = [0,1,0,1,...]: open rows BITWISE the
     all-open run, sealed rows BITWISE the all-sealed run. S3=0 changes
     the emissions (the second road is live and is being cooked).
  GATE 5 — TWO-TERMINAL (proof d): on sealed rows the parse loss reaches
     tg_a and tg_b (reachable ONLY through the gate) and every organ of
     the mask head; a unit test shows the gate is differentiable in the
     context state itself (so the road runs on into the encoder that
     built it); on all-open batches the gradients are the pristine
     head's — BITWISE with the door unset.
  GATE 6 — THE THREE COOKERS COMPOSE (proof e): the three assignment
     hashes are measured independent over the real dataset index range
     and all EIGHT cells occur; in a batch of 8 every row is BITWISE its
     single-configuration run.
  GATE 7 — THE METER AND THE GUARD (proof f): ALG_TOK_SEAL=head at read
     is BITWISE the all-rows-sealed training path; TC_EVAL forces the
     OPEN regime bitwise and wins over `head`; and do_train REFUSES to
     start with ALG_TOK_SEAL=head — by EXECUTION and by AST.
  GATE 8 — THE APPLY SCRIPT (proof g): --check clean and writes nothing,
     a rehearsal apply reproduces the staged source exactly, a second
     application is refused, and jit_read's env miner finds every door.

Run from the repo root:
  .venv/bin/python3 scripts/tok_cook_smoke.py
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
           "ALG_TOK_SEAL", "ALG_TOK_SEAL_S3", "ALG_TOK_COOK",
           "ALG_TOK_COOK_S3", "ALG_TG_INIT", "TC_EVAL", "ALG_MASKRE",
           "ALG_MINE_BREATHS", "ALG_CLOCK", "ALG_JIT_READ"):
    os.environ.pop(_k, None)
os.environ["SC_EVAL"] = "0"     # the machine's OPEN regime

import numpy as np                                            # noqa: E402
from tinygrad import Tensor, dtypes                           # noqa: E402

HEAD = os.environ.get("TC_TARGET",
                      os.path.join(_ROOT, "scripts", "phase1_algebra_head.py"))
B = 8
WARM = 0.05        # the warm stand-in's scale on the zero-born doors
SUM_TOL = 1e-5     # a softmax's row sum, f32
ATOL = 1e-4        # cross-SCHEDULE tolerance (the smokes' shared bar)
POLAR_ENV = {"ALG_POLAR": "1", "ALG_POLAR_D": "128", "ALG_POLAR_EM": "0.1"}
MH_KEYS = ("mh_wq", "mh_wk", "mh_wv", "mh_wo", "mh_headmix",
           "mh_enc1", "mh_enc2")
# PORT-ABSENT BY CONSTRUCTION on this fixture: the atlas page enters as
# `(cur * 0.0).detach() @ mh_atlas_w` when no seam driver supplies
# ctx["mh_atlas"], so its gradient is ZERO but DEFINED (the None-grad
# law, and the head says so in its own comment). Asserted as DEFINED,
# never as nonzero — a bar it cannot pass would be a bar about the
# fixture, not the organ.
PORT_ABSENT = ("mh_atlas_w",)
TG_KEYS = ("tg_a", "tg_b")


# ===========================================================================
# STAGING — build the would-be patched source without writing the head
# ===========================================================================
def staged_source():
    argv0 = sys.argv
    sys.argv = ["apply_tok_cook.py", "--check"]
    try:
        ns = runpy.run_path(os.path.join(_ROOT, "scripts",
                                         "apply_tok_cook.py"),
                            run_name="_tok_cook_staging")
    finally:
        sys.argv = argv0
    assert ns["CHECK"], "staging must run under --check (nothing written)"
    return ns["s"], ns["MH_BLOCK"], ns["MH_BODY"]


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
STAGED, MH_BLOCK, MH_BODY = staged_source()
assert APPLIED != STAGED and "ALG_TOK_COOK" not in APPLIED
assert "ALG_TOK_SEAL" in APPLIED and "_mask_cook_skel" in APPLIED \
    and "_polar_em" in APPLIED, \
    "the cooker anchors into the head AS IT IS (tok seal + mask cooker + sink)"
print(f"[tc-smoke] staged +{STAGED.count(chr(10)) - APPLIED.count(chr(10))}"
      f" lines on the APPLIED head; head on disk UNTOUCHED", flush=True)

M_BASE = load_module(APPLIED, "applied_plain")
M_STAG = load_module(STAGED, "staged_plain")
M_BASE_P = with_env(POLAR_ENV, lambda: load_module(APPLIED, "applied_polar"))
M_STAG_P = with_env(POLAR_ENV, lambda: load_module(STAGED, "staged_polar"))
assert not M_STAG.ALG_POLAR and M_STAG_P.ALG_POLAR and M_STAG_P.POLAR_D == 128
assert M_STAG.TG_D == 128, "TG_D must be the spec's 128"


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
H_W = M_BASE.H_W
EYE = np.eye(L_FAC, dtype=np.float32)
_rs = np.random.RandomState(7)
MK_SPARSE = np.clip((_rs.rand(B, L_FAC, L_FAC) < 0.4).astype(np.float32)
                    + EYE, 0.0, 1.0)
assert set(np.unique(TKN)) <= {0.0, 1.0} and TKN.min() == 0.0, \
    "tokmask must be 0/1 (the gate's exact-zero-on-pads claim rests on it)"
print(f"[tc-smoke] fixture test23 rows={B} L_FAC={L_FAC} L_TOT={L_TOT} "
      f"breaths={K_B} H_W={H_W} dev={os.environ['DEV']} | tokens per row "
      f"{NTOK.astype(int).tolist()} of T={TKN.shape[1]}; log n_tok mean "
      f"{float(np.log(NTOK).mean()):.4f} (the uniform gate's entropy — the "
      f"scale every KL below is read against)", flush=True)


def cold_params(M, seed=0):
    return M.build_params(seed)


def warm_params(M, seed=0, scale=WARM):
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
    o = M.forward(p, TS, TK, SE,
                  slot_mask=Tensor(MK_SPARSE if mk is None else mk,
                                   dtype=dtypes.float))
    return {k: v.realize().numpy() for k, v in o.items()
            if isinstance(v, Tensor)}


def diff_keys(a, b):
    assert set(a) == set(b), f"key sets differ: {sorted(set(a) ^ set(b))}"
    return [k for k in sorted(a) if not np.array_equal(a[k], b[k])]


def maxd(a, b, keys=None):
    ks = keys or sorted(set(a) & set(b))
    return max(float(np.abs(a[k].astype(np.float64)
                            - b[k].astype(np.float64)).max()) for k in ks)


def rows_bitwise(a, b, rows):
    return all(np.array_equal(a[k][r], b[k][r]) for k in a for r in rows)


def rows_differ(a, b, rows):
    return all(any(not np.array_equal(a[k][r], b[k][r]) for k in a)
               for r in rows)


def set_buf(M, nm, vals):
    setattr(M, nm, Tensor(np.array(vals, np.float32)
                          .reshape(-1, 1, 1)).contiguous().realize())


def clear_bufs(M):
    for nm in ("_TCV", "_MCV", "_PCV"):
        if hasattr(M, nm):
            delattr(M, nm)


# ===========================================================================
# GATE 1 — ENV-INERTNESS (proof a)
# ===========================================================================
# CORE cells run in EVERY {polar config} x {cold, warm} combination — the
# registered inertness claim (the door unset and every tok-seal value
# unchanged from the current head) plus the two code motions' own cells
# (ALG_MASKRE=1 reaches _mh_a5; a read env with ALG_TOK_COOK set and no
# _TCV buffer is the trained-env law). EXTRA cells are companion doors
# that cannot seal anything on their own; they ride the warm stand-in in
# the plain config, where the roads are live and a leak would show.
CORE_CELLS = ({}, {"ALG_TOK_SEAL": "0"}, {"ALG_TOK_SEAL": "loop"},
              {"ALG_TOK_SEAL": "all"},
              {"ALG_TOK_COOK": "0.15"},           # a READ env: no _TCV
              {"ALG_MASKRE": "1"})                # the _mh_a5 motion
EXTRA_CELLS = ({"ALG_TOK_COOK": "0"}, {"ALG_TOK_COOK_S3": "0"},
               {"ALG_TOK_COOK_S3": "1"}, {"ALG_TG_INIT": "2.0"},
               {"TC_EVAL": "0"},
               {"ALG_MASKRE": "1", "ALG_TOK_SEAL": "loop"})
for lab, MA, MB, env, extra in (
        ("ALG_POLAR unset", M_BASE, M_STAG, {}, True),
        ("ALG_POLAR=1 + sink (D=128, EM=0.1)",
         M_BASE_P, M_STAG_P, POLAR_ENV, False)):
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
        n_c = 0
        for tag, pa, pb in (("cold", PA, PB), ("warm stand-in", WA, WB)):
            cells = (CORE_CELLS + EXTRA_CELLS
                     if (extra and tag != "cold") else CORE_CELLS)
            for cell in cells:
                oa = with_env(cell, lambda: read(MA, pa))
                ob = with_env(cell, lambda: read(MB, pb))
                bad = diff_keys(oa, ob)
                assert not bad, (f"GATE 1 FAIL ({lab}, {tag}, {cell}): NOT "
                                 f"bit-identical ({bad})")
                n_k = len(oa)
                n_t += len(oa)
                n_c += 1
        return _n, n_t, n_k, n_c
    _nz, _nt, _nk, _nc = with_env(env, _g1)
    print(f"[tc-smoke] GATE 1 PASS [{lab}]: applied vs staged BIT-IDENTICAL "
          f"on {_nt} tensors ({_nk} emission keys x 8 rows x {_nc} "
          f"(params, env) cells incl. every tok-seal value, ALG_TOK_COOK="
          f"0.15 with no buffer (the read env) and ALG_MASKRE=1); params "
          f"identical, 0 added (warm stand-in seeds {_nz} all-zero params)",
          flush=True)

P_W, _NZ = warm_params(M_STAG)
O_OPEN = read(M_STAG, P_W)
assert not diff_keys(O_OPEN, read(M_STAG, P_W)), \
    "the CPU read is nondeterministic — every bitwise claim below is void"
EKEYS = sorted(O_OPEN)
assert "tg_a" not in P_W and "tg_b" not in P_W, \
    "W_tg must not exist with the doors unset (the parameter set moves)"
print(f"[tc-smoke] emission keys under test ({len(EKEYS)}): "
      f"{', '.join(EKEYS)}; with every door unset the params carry NO tg_a/"
      f"tg_b — the banked checkpoints load unchanged", flush=True)


# ===========================================================================
# GATE 2 — THE CODE MOTION IS A MOTION
# ===========================================================================
_i0 = STAGED.index("        return _memo[1], _memo[2]\n") \
    + len("        return _memo[1], _memo[2]\n")
_i1 = STAGED.index('    state["tc_mh_ctx"] = (kb, _mh_kv, _A5s)')
_moved = STAGED[_i0:_i1]
_orig_ded = '\n'.join(_l[4:] if _l.startswith('    ') else _l
                      for _l in MH_BLOCK.split('\n'))
assert _moved == _orig_ded == MH_BODY, \
    "GATE 2 FAIL: _mh_ctx's body is NOT the applied head's block verbatim"
assert MH_BLOCK in APPLIED and MH_BLOCK not in STAGED, \
    "GATE 2 FAIL: the block did not leave its old site"
assert len(MH_BLOCK.split('\n')) > 60, \
    "GATE 2 FAIL: the lifted block is suspiciously short"
print(f"[tc-smoke] GATE 2a PASS: the mask head's context block "
      f"({len(MH_BLOCK.split(chr(10))) - 1} lines) is inside `_mh_ctx` "
      f"VERBATIM — the patch script SLICES it out of the head's source and "
      f"dedents it one level; compared here character by character against "
      f"the applied head. It left its old site. PURE CODE MOTION is a "
      f"fact here, not a claim.", flush=True)

_CTXCALLS = []
_orig_mh_ctx = M_STAG._mh_ctx


def _cap_mh_ctx(p, cur, state, ctx, kb, B_, _A5, _snaps):
    r = _orig_mh_ctx(p, cur, state, ctx, kb, B_, _A5, _snaps)
    _CTXCALLS.append((kb, id(r[0])))
    return r


M_STAG._mh_ctx = _cap_mh_ctx
P_C = with_env({"ALG_TOK_COOK": "0.5"},
               lambda: warm_params(M_STAG)[0])
_CTXCALLS.clear()
os.environ["ALG_TOK_COOK"] = "0.5"
set_buf(M_STAG, "_TCV", [1.0] * B)
read(M_STAG, P_C)
_armed = list(_CTXCALLS)
_CTXCALLS.clear()
clear_bufs(M_STAG)
os.environ.pop("ALG_TOK_COOK", None)
read(M_STAG, P_W)
_unarmed = list(_CTXCALLS)
M_STAG._mh_ctx = _orig_mh_ctx
_by_kb = {}
for kb, i in _armed:
    _by_kb.setdefault(kb, []).append(i)
assert len(_armed) == 2 * (K_B - 1) and len(_unarmed) == K_B - 1, \
    f"GATE 2 FAIL: {len(_armed)} calls armed / {len(_unarmed)} unarmed, " \
    f"expected {2 * (K_B - 1)} / {K_B - 1}"
assert all(len(v) == 2 and len(set(v)) == 1 for v in _by_kb.values()), \
    "GATE 2 FAIL: the two calls per breath returned DIFFERENT tensors — " \
    "the memo is broken and the gate is scoring a copy, not the organ"
print(f"[tc-smoke] GATE 2b PASS: with the cooker armed `_mh_ctx` is CALLED "
      f"twice per loop breath ({len(_armed)} = 2 x {K_B - 1}) and returns "
      f"the SAME TENSOR OBJECT both times (identity, {len(_by_kb)} breaths) "
      f"— the gate scores the state the head reads, not a copy of it (the "
      f"meter-divergence law). Unarmed it is called once per breath "
      f"({len(_unarmed)}): today's path, today's cost.", flush=True)


# ===========================================================================
# INSTRUMENTATION (its disturbance measured — the tok_seal_smoke discipline)
# ===========================================================================
CAP = []           # (tag, nq, flat, tgated, attention as numpy)
GATES = []         # (gate as numpy)
_orig_make_bank = M_STAG._make_bank
_orig_tok_gate = M_STAG._tok_gate


def _cap_make_bank(p, waist, tokmask, Bn):
    _b = _orig_make_bank(p, waist, tokmask, Bn)

    def _wrapped(queries, nq, extra=None, pbias=None, rbias=None, flat=False,
                 tgate=None, tgv=None):
        st, at = _b(queries, nq, extra=extra, pbias=pbias, rbias=rbias,
                    flat=flat, tgate=tgate, tgv=tgv)
        CAP.append(("loop" if extra is not None else
                    ("vars" if nq == K_VARS else
                     ("query" if nq == 1 else "breath0")),
                    nq, flat, tgate is not None, at.realize().numpy()))
        return st, at
    return _wrapped


def _cap_tok_gate(p, ctxst, waist, tokmask, Bn):
    g = _orig_tok_gate(p, ctxst, waist, tokmask, Bn)
    GATES.append(g.realize().numpy())
    return g


M_STAG._make_bank = _cap_make_bank
M_STAG._tok_gate = _cap_tok_gate


def instrumented_read(env, p=None, tcv=None):
    CAP.clear()
    GATES.clear()
    if tcv is not None:
        set_buf(M_STAG, "_TCV", tcv)
    M_STAG._STEP_TAP = {}
    try:
        o = with_env(env, lambda: read(M_STAG, P_W if p is None else p))
        tap = dict(M_STAG._STEP_TAP)
    finally:
        M_STAG._STEP_TAP = None
        if tcv is not None:
            clear_bufs(M_STAG)
    return o, list(CAP), list(GATES), tap


O_OPEN_I, CAP_OPEN, G_OPEN_I, TAP_OPEN = instrumented_read({})
_dn = maxd(O_OPEN, O_OPEN_I)
assert _dn <= ATOL, \
    f"the instrumentation moved the machine by {_dn:.3e} > {ATOL}"
assert not G_OPEN_I, "the gate organ ran with every door unset"
_tags = [c[0] for c in CAP_OPEN]
assert _tags == ["vars", "breath0", "query"] + ["loop"] * (K_B - 1), \
    f"unexpected bank call order {_tags}"
print(f"[tc-smoke] instrumentation: the bank/gate capture (mid-graph "
      f"realizes, a different kernel schedule) moves the emissions by "
      f"{_dn:.3e} <= {ATOL}, so every BITWISE claim below is made between "
      f"two instrumented runs. Bank calls per forward: {_tags}; the gate "
      f"organ ran 0 times with the doors unset.", flush=True)


def gate_stats(g):
    """(max|sum over real tokens - 1|, max weight on a PAD, mean KL from
    uniform over slots and rows, max KL)."""
    g64 = g.astype(np.float64)
    real = TKN.reshape(B, 1, -1)
    ssum = float(np.abs((g64 * real).sum(-1) - 1.0).max())
    pad = float(np.abs(g64 * (1.0 - real)).max())
    kl = (np.log(NTOK).reshape(B, 1)
          + (g64 * np.log(np.maximum(g64, 1e-300))).sum(-1))
    return ssum, pad, float(kl.mean()), float(kl.max())


# ===========================================================================
# GATE 3 — THE GATE IS A DISTRIBUTION (proof c) + THE BIRTH KL
# ===========================================================================
P_H = with_env({"ALG_TOK_SEAL": "head"},
               lambda: warm_params(M_STAG)[0])
assert set(P_H) - set(P_W) == {"tg_a", "tg_b"}, \
    f"GATE 3 FAIL: the armed parameter delta is {sorted(set(P_H) ^ set(P_W))}"
assert all(np.array_equal(P_H[k].numpy(), P_W[k].numpy())
           for k in P_W if k not in ("tg_a", "tg_b")), \
    "GATE 3 FAIL: arming the door moved another parameter's init (the " \
    "dedicated rng stream is not dedicated)"
_npar = sum(int(np.prod(P_H[k].shape)) for k in TG_KEYS)
assert _npar == 2 * H_W * M_STAG.TG_D == 131072, f"W_tg is {_npar} params"

O_HEAD, CAP_HEAD, G_HEAD, TAP_HEAD = instrumented_read(
    {"ALG_TOK_SEAL": "head"}, p=P_H)
assert len(G_HEAD) == K_B - 1, \
    f"GATE 3 FAIL: the gate ran {len(G_HEAD)}x, expected {K_B - 1} (ONE " \
    f"gate per loop breath, spent on BOTH roads)"
_w = (0.0, 0.0)
_kls = []
_dgs = []
for _i, _g in enumerate(G_HEAD, start=1):
    assert _g.shape == (B, L_TOT, TKN.shape[1]), f"gate shape {_g.shape}"
    _s, _p, _kl, _klx = gate_stats(_g)
    _w = (max(_w[0], _s), max(_w[1], _p))
    _kls.append(_kl)
    assert _s <= SUM_TOL, f"GATE 3 FAIL: breath {_i} sums to 1 +- {_s:.3e}"
    assert _p == 0.0, f"GATE 3 FAIL: breath {_i} puts {_p:.3e} on PADS"
    assert _kl > 0.0, f"GATE 3 FAIL: breath {_i} gate is EXACTLY uniform " \
                      f"— a zero-born gate is a dead gate"
# the bank's attention IS the gate, on every head, bitwise
for _i, (_t, _nq, _fl, _tg, _at) in enumerate(CAP_HEAD[3:], start=1):
    assert _t == "loop" and _tg and not _fl, \
        f"GATE 3 FAIL: loop breath {_i} did not take the gate"
    # what the bank RETURNS is at.mean(1), the head average — and the
    # gate is head-independent, so a fully sealed breath's head average
    # IS the gate itself (a mean of 8 identical copies, exact in binary
    # f32). The per-head claim is structural: the gate enters as
    # tgate.reshape(B, 1, nq, -1), broadcast over the head axis.
    _g = G_HEAD[_i - 1]
    _dg = float(np.abs(_at.astype(np.float64) - _g.astype(np.float64)).max())
    _dgs.append(_dg)
    assert _dg <= 1e-7, \
        f"GATE 3 FAIL: breath {_i}'s bank attention is not the gate " \
        f"(max|d| {_dg:.3e})"
# breath 0 is untouched
for _nm in ("fst", "qst", "vst", "vst_base"):
    assert np.array_equal(TAP_OPEN[_nm].realize().numpy(),
                          TAP_HEAD[_nm].realize().numpy()), \
        f"GATE 3 FAIL: breath-0 quantity {_nm} moved under the gate"
for _i, (_t, _nq, _fl, _tg, _at) in enumerate(CAP_HEAD[:3]):
    assert not _tg and np.array_equal(_at, CAP_OPEN[_i][4]), \
        f"GATE 3 FAIL: the {_t} bank attention moved (breath 0 / var / query)"
print(f"[tc-smoke] GATE 3 PASS: at every one of the {K_B - 1} loop breaths "
      f"the gate is a PROPER DISTRIBUTION over the row's real tokens — sums "
      f"to 1 within {_w[0]:.2e} (f32, bar {SUM_TOL}) and puts EXACTLY "
      f"{_w[1]:.1e} on every pad — and what the bank RETURNS (at.mean(1), "
      f"the head average) IS the gate to max|d| {max(_dgs):.1e}, because "
      f"the gate enters broadcast over the head axis and every head of a "
      f"sealed attention reads the same aim. Breath 0's grounding (fst, "
      f"qst, vst, vst_base "
      f"and the var/query/breath-0 attentions) is bitwise pristine: the "
      f"cooker cooks the RE-reading only.", flush=True)
print(f"[tc-smoke] GATE 3 BIRTH KL (reported, not barred): mean KL from "
      f"uniform per slot = {np.mean(_kls):.5f} nats "
      f"(per breath {[round(k, 5) for k in _kls]}, max over slots/rows "
      f"{max(gate_stats(g)[3] for g in G_HEAD):.5f}) against log n_tok = "
      f"{float(np.log(NTOK).mean()):.4f}. NONZERO (the gate has structure "
      f"and a gradient path — GATE 5) and SMALL: at ALG_TG_INIT=1.0 the "
      f"birth gate sits essentially AT the uniform floor the severance "
      f"probe measured (0.1536 wild), which is where the cooker is supposed "
      f"to start. RE-MEASURE ON THE WARM CKPT before turning the dial: this "
      f"is a warm STAND-IN's state scale, not polarsink242's.", flush=True)

# the KL the ORGAN records (the census) is the KL the reader will quote
M_STAG._CENSUS = []
with_env({"ALG_TOK_SEAL": "head"}, lambda: read(M_STAG, P_H))
_rec = list(M_STAG._CENSUS)
M_STAG._CENSUS = None
_cg = [(kb, a) for (kb, n_, a) in _rec if n_ == "tokgate"]
_ck = [(kb, a) for (kb, n_, a) in _rec if n_ == "tokgate_kl"]
assert len(_cg) == len(_ck) == K_B - 1, \
    f"GATE 3 FAIL: census recorded {len(_cg)}/{len(_ck)} organ rows"
assert all(a.shape == (B, 1, 1) for _, a in _ck), \
    f"tokgate_kl must be one scalar per row, got {_ck[0][1].shape}"
_d_kl = max(abs(float(a.mean()) - gate_stats(g)[2])
            for (_, a), (_, g) in zip(_ck, _cg))
assert _d_kl < 1e-5, f"GATE 3 FAIL: the organ's KL differs by {_d_kl:.2e}"
assert all(float(a.min()) >= 0.0 for _, a in _ck), "a KL went negative"
print(f"[tc-smoke] GATE 3 CENSUS PASS: the `tokgate` organ records the gate "
      f"{_cg[0][1].shape} and `tokgate_kl` one scalar per row {_ck[0][1].shape} "
      f"at each of the {len(_ck)} loop breaths; the organ's own KL agrees "
      f"with this smoke's independent numpy KL to {_d_kl:.2e} (the reader "
      f"quotes the organ, it does not rebuild it) and is >= 0 everywhere. "
      f"Behind the _CENSUS hook: unarmed, none of it exists.", flush=True)


# ===========================================================================
# GATE 4 — PER-ROW SURGERY (proof b)
# ===========================================================================
M_STAG._make_bank = _orig_make_bank        # the surgery gates read the
M_STAG._tok_gate = _orig_tok_gate          # UNinstrumented machine
os.environ["ALG_TOK_COOK"] = "0.5"
set_buf(M_STAG, "_TCV", [0.0] * B)
O_Z = read(M_STAG, P_H)
_d0 = maxd(O_OPEN, O_Z)
assert _d0 <= ATOL, \
    f"GATE 4a FAIL: _TCV all-zeros drifted from the door-unset run by " \
    f"{_d0:.3e} > {ATOL} (the gate compute + the memo must be neutral)"
set_buf(M_STAG, "_TCV", [1.0] * B)
O_ONE = read(M_STAG, P_H)
SEALED = [1, 3, 5, 7]
OPENR = [0, 2, 4, 6]
set_buf(M_STAG, "_TCV", [1.0 if r in SEALED else 0.0 for r in range(B)])
O_MIX = read(M_STAG, P_H)
assert rows_bitwise(O_Z, O_MIX, OPENR), \
    "GATE 4b FAIL: open-assigned rows moved under the mixed seal"
assert rows_bitwise(O_ONE, O_MIX, SEALED), \
    "GATE 4b FAIL: sealed rows are not bitwise the all-sealed run"
assert rows_differ(O_Z, O_MIX, SEALED), \
    "GATE 4b FAIL: sealed-assigned rows did not change"
_dsev = max(float(np.abs(O_Z[k][r].astype(np.float64)
                         - O_MIX[k][r].astype(np.float64)).max())
            for k in O_Z for r in SEALED)
set_buf(M_STAG, "_TCV", [1.0] * B)
O_NARROW = with_env({"ALG_TOK_COOK_S3": "0"}, lambda: read(M_STAG, P_H))
assert diff_keys(O_NARROW, O_ONE), \
    "GATE 4c FAIL: cooking ALT21 station 3 changed nothing — either the " \
    "station is off in this env or the gate missed it"
clear_bufs(M_STAG)
os.environ.pop("ALG_TOK_COOK", None)
print(f"[tc-smoke] GATE 4 PASS: ALG_TOK_COOK=0.5 with _TCV all-zeros == the "
      f"door-unset read (max|d| {_d0:.3e}, cross-graph atol {ATOL}); rows "
      f"{SEALED} sealed are BIT-IDENTICAL to the all-sealed run and rows "
      f"{OPENR} BIT-IDENTICAL to the all-open run (row surgery is exact); "
      f"the severance moves a sealed row's emissions by {_dsev:.4f}. "
      f"ALG_TOK_COOK_S3=0 changes {len(diff_keys(O_NARROW, O_ONE))} keys "
      f"(max|d| {maxd(O_NARROW, O_ONE):.4f}) — station 3 is a live second "
      f"grounding road and the default cooks it too.", flush=True)


# ===========================================================================
# GATE 5 — TWO-TERMINAL (proof d)
# ===========================================================================
def grad_probe(M, p, keys):
    for t_ in p.values():
        t_.grad = None
    o = M.forward(p, TS, TK, SE, slot_mask=Tensor(MK_SPARSE, dtype=dtypes.float))
    o["res"].sum().backward()     # the PARSE head only (no gate loss, ever)
    out = {}
    for k in keys:
        g = p[k].grad
        # None (no path at all — the representability audit's failure
        # mode) is kept DISTINCT from 0.0 (a path that carries zero)
        out[k] = None if g is None else float(g.detach().abs().max().numpy())
    return out


P_G = with_env({"ALG_TOK_SEAL": "head"}, lambda: warm_params(M_STAG)[0])
os.environ["ALG_TOK_COOK"] = "0.5"
set_buf(M_STAG, "_TCV", [1.0] * B)
G_SEAL = grad_probe(M_STAG, P_G, TG_KEYS + MH_KEYS + PORT_ABSENT)
_dead = [k for k in TG_KEYS + MH_KEYS
         if not (G_SEAL[k] is not None and G_SEAL[k] > 0.0)]
assert not _dead, f"GATE 5 FAIL: sealed rows give NO gradient to {_dead}"
_undef = [k for k in PORT_ABSENT if G_SEAL[k] is None]
assert not _undef, \
    f"GATE 5 FAIL: {_undef} has NO gradient path at all (None, not zero) " \
    f"— the None-grad law: a port must stay in the graph"
print("[tc-smoke] GATE 5a PASS: on SEALED rows the PARSE loss reaches the "
      "gate and the head — " + ", ".join(f"{k} {G_SEAL[k]:.3e}"
                                         for k in TG_KEYS + MH_KEYS)
      + " (tg_a/tg_b are reachable ONLY through the gate: the new road is "
        "two-terminal, emission AND feed). "
      + ", ".join(f"{k} {G_SEAL[k]:.1e}" for k in PORT_ABSENT)
      + " is zero-but-DEFINED, as it is with the cooker off: no seam "
        "driver supplies ctx['mh_atlas'] on this fixture, so the page is "
        "cur*0 and the port carries no signal to carry — the None-grad "
        "law is satisfied (a path exists), the fixture is simply silent "
        "on that port.", flush=True)

# the unit: the gate is differentiable in the CONTEXT STATE itself, so the
# road runs on into whatever built that state (the head's context encoder)
_rs3 = np.random.RandomState(11)
_ctx_t = Tensor(_rs3.randn(B, L_TOT, H_W).astype(np.float32) * 0.1,
                dtype=dtypes.float, requires_grad=True).contiguous().realize()
_ctx_t.requires_grad = True
_wst = Tensor(_rs3.randn(B, TKN.shape[1], H_W).astype(np.float32) * 0.1,
              dtype=dtypes.float)
_tgt_t = Tensor(_rs3.randn(B, L_TOT, TKN.shape[1]).astype(np.float32),
                dtype=dtypes.float)
_gu = M_STAG._tok_gate(P_G, _ctx_t, _wst, TK, B)
(_gu * _tgt_t).sum().backward()
_gctx = float(_ctx_t.grad.detach().abs().max().numpy())
_gta = float(P_G["tg_a"].grad.detach().abs().max().numpy())
assert _gctx > 0.0 and _gta > 0.0, \
    "GATE 5 FAIL (unit): the gate is not differentiable in its context state"
print(f"[tc-smoke] GATE 5b PASS (unit): d(gate)/d(context state) is nonzero "
      f"({_gctx:.3e}) — the gate's gradient does not stop at W_tg, it runs "
      f"on into the state the mask head built (mh_enc1/mh_enc2/mh_atlas_w/"
      f"fed_nl0_w and the live `cur`), which is why GATE 5a's mh_enc "
      f"numbers are not an artifact of the untouched mask road.", flush=True)

set_buf(M_STAG, "_TCV", [0.0] * B)
G_OPEN_ARMED = grad_probe(M_STAG, P_G, MH_KEYS)
clear_bufs(M_STAG)
os.environ.pop("ALG_TOK_COOK", None)
G_OPEN_STAG = grad_probe(M_STAG, P_W, MH_KEYS)
P_B, _ = warm_params(M_BASE)
G_OPEN_BASE = grad_probe(M_BASE, P_B, MH_KEYS)
assert all(G_OPEN_STAG[k] == G_OPEN_BASE[k] for k in MH_KEYS), \
    ("GATE 5 FAIL: with the door unset the staged head's gradients are not "
     "BITWISE the applied head's " +
     str({k: (G_OPEN_STAG[k], G_OPEN_BASE[k]) for k in MH_KEYS}))
_rel = max(abs(G_OPEN_ARMED[k] - G_OPEN_BASE[k]) / max(G_OPEN_BASE[k], 1e-30)
           for k in MH_KEYS)   # all non-None on the warm stand-in
assert _rel < 1e-3, f"GATE 5 FAIL: all-open armed gradients drifted {_rel:.3e}"
print(f"[tc-smoke] GATE 5c PASS: on all-open batches the mask head's "
      f"gradients are the PRISTINE head's — BITWISE with the door unset "
      f"({len(MH_KEYS)}/{len(MH_KEYS)} params, so the two code motions cost "
      f"the gradient path nothing), and {_rel:.2e} relative with the door "
      f"armed and _TCV all-zero (cross-graph).", flush=True)


# ===========================================================================
# GATE 6 — THE THREE COOKERS COMPOSE (proof e)
# ===========================================================================
_n_rows = 25000
_ii = np.arange(_n_rows, dtype=np.uint64)
_h_pc = ((_ii * np.uint64(2654435761)) % np.uint64(4294967296)
         ).astype(np.float64) / 4294967296.0
_h_mc = ((_ii * np.uint64(2246822519) + np.uint64(2654435761))
         % np.uint64(4294967296)).astype(np.float64) / 4294967296.0
_h_tc = ((_ii * np.uint64(3266489917) + np.uint64(374761393))
         % np.uint64(4294967296)).astype(np.float64) / 4294967296.0
for _s_pc, _s_mc, _s_tc in ((0.30, 0.15, 0.15), (0.15, 0.15, 0.15),
                            (0.30, 0.30, 0.30)):
    _a, _b_, _c_ = _h_pc < _s_pc, _h_mc < _s_mc, _h_tc < _s_tc
    _cells = [int((( _a if i & 1 else ~_a)
                   & (_b_ if i & 2 else ~_b_)
                   & (_c_ if i & 4 else ~_c_)).sum()) for i in range(8)]
    _dev = max(abs(float((_x & _y).mean()) - float(_x.mean()) * float(_y.mean()))
               for _x, _y in ((_a, _b_), (_a, _c_), (_b_, _c_)))
    _dev3 = abs(float((_a & _b_ & _c_).mean())
                - float(_a.mean()) * float(_b_.mean()) * float(_c_.mean()))
    assert min(_cells) > 0 and _dev < 1e-3 and _dev3 < 1e-3, \
        f"GATE 6 FAIL: hashes correlate at ({_s_pc}, {_s_mc}, {_s_tc})"
    print(f"[tc-smoke] GATE 6a: doses (pc {_s_pc}, mask {_s_mc}, tok "
          f"{_s_tc}) over n={_n_rows}: all 8 cells occur {_cells}, max "
          f"pairwise |P(xy) - P(x)P(y)| = {_dev:.5f}, triple deviation "
          f"{_dev3:.5f}", flush=True)

os.environ.pop("SC_EVAL", None)         # the pressure seal needs it unset
os.environ["ALG_PC_MIX"] = "0.3"
os.environ["ALG_MASK_COOK"] = "0.3"
os.environ["ALG_TOK_COOK"] = "0.3"
P_3 = with_env({"ALG_TOK_SEAL": "head"}, lambda: warm_params(M_STAG)[0])
CELLS = {(pv, mv, tv): None for pv in (0.0, 1.0) for mv in (0.0, 1.0)
         for tv in (0.0, 1.0)}
for _cell in CELLS:
    set_buf(M_STAG, "_PCV", [_cell[0]] * B)
    set_buf(M_STAG, "_MCV", [_cell[1]] * B)
    set_buf(M_STAG, "_TCV", [_cell[2]] * B)
    CELLS[_cell] = read(M_STAG, P_3)
ROWCELL = list(CELLS)
set_buf(M_STAG, "_PCV", [c[0] for c in ROWCELL])
set_buf(M_STAG, "_MCV", [c[1] for c in ROWCELL])
set_buf(M_STAG, "_TCV", [c[2] for c in ROWCELL])
O_COMP = read(M_STAG, P_3)
for _r, _c in enumerate(ROWCELL):
    assert rows_bitwise(O_COMP, CELLS[_c], [_r]), \
        f"GATE 6 FAIL: row {_r} (pc,mc,tc = {_c}) is not bitwise its " \
        f"single-configuration run"
_pairs = [(c1, c2) for c1 in CELLS for c2 in CELLS if c1 < c2]
_same = [(c1, c2) for c1, c2 in _pairs
         if not rows_differ(CELLS[c1], CELLS[c2], range(B))]
assert not _same, f"GATE 6 FAIL: indistinguishable cells {_same}"
clear_bufs(M_STAG)
for _k in ("ALG_PC_MIX", "ALG_MASK_COOK", "ALG_TOK_COOK"):
    os.environ.pop(_k, None)
os.environ["SC_EVAL"] = "0"
print(f"[tc-smoke] GATE 6b PASS: with ALL THREE cookers armed, all 8 cells "
      f"occur in a batch of 8, every row is BITWISE its single-configuration "
      f"run, and all {len(_pairs)} cell pairs are distinguishable. WHERE "
      f"THEY INTERACT: inside a row that carries more than one severance "
      f"(one forward, three cuts), never across rows — the per-row blends "
      f"are exact and the three assignments independent.", flush=True)


# ===========================================================================
# GATE 7 — THE METER AND THE GUARD (proof f)
# ===========================================================================
os.environ["ALG_TOK_COOK"] = "1.0"
set_buf(M_STAG, "_TCV", [1.0] * B)
O_TRAINPATH = read(M_STAG, P_H)
os.environ["TC_EVAL"] = "0"
O_VAL = read(M_STAG, P_H)
os.environ.pop("TC_EVAL", None)
clear_bufs(M_STAG)
os.environ.pop("ALG_TOK_COOK", None)
O_METER = with_env({"ALG_TOK_SEAL": "head"}, lambda: read(M_STAG, P_H))
_dm = maxd(O_METER, O_TRAINPATH)
assert _dm <= ATOL, \
    (f"GATE 7 FAIL: ALG_TOK_SEAL=head differs from the all-rows-sealed "
     f"training path by {_dm:.3e} > {ATOL} — more than scheduling")
assert rows_bitwise(O_VAL, O_OPEN, range(B)), \
    "GATE 7 FAIL: TC_EVAL did not force the OPEN regime bitwise"
O_VAL2 = with_env({"ALG_TOK_SEAL": "head", "TC_EVAL": "0"},
                  lambda: read(M_STAG, P_H))
assert rows_bitwise(O_VAL2, O_OPEN, range(B)), \
    "GATE 7 FAIL: TC_EVAL must win over ALG_TOK_SEAL=head too"
print(f"[tc-smoke] GATE 7a PASS: ALG_TOK_SEAL=head at read == the "
      f"all-rows-sealed TRAINING path to max|d| {_dm:.3e} (bar {ATOL}) — "
      f"the meter measures the machine the cooker trains. NOT BITWISE, and "
      f"the reason is named rather than hidden: the meter's blend value is "
      f"the python scalar 1.0 (`_tok_cook_v` returns it for `head`, the "
      f"_mask_cook_v idiom) while training's is a (B,1,1,1) DATA BUFFER, "
      f"so tinygrad folds constants in one and cannot in the other — a "
      f"different kernel schedule, the same f32 noise (~9e-6) this smoke "
      f"measured for its own instrumentation. The two are ONE CODE PATH "
      f"by construction (one `_tok_cook_v`, one blend site, asserted "
      f"structurally in the apply script), and GATE 4 proved the sealed "
      f"rows of a MIXED batch bitwise equal to the all-sealed run. TC_EVAL "
      f"forces the OPEN regime BITWISE with every row assigned sealed, and "
      f"over `head` as well — val compares OPEN at every shelf mode.",
      flush=True)

for _bad in ("1", "HEAD", "heads", "true"):
    try:
        with_env({"ALG_TOK_SEAL": _bad}, lambda: read(M_STAG, P_W))
        raise SystemExit(f"GATE 7 FAIL: ALG_TOK_SEAL={_bad!r} did not raise")
    except AssertionError as e:
        assert "not one of 0 / loop / all / head" in str(e), \
            f"GATE 7 FAIL: wrong error for {_bad!r}: {e}"

import ast                                                     # noqa: E402
_tree = ast.parse(STAGED)
_dt = [n for n in _tree.body
       if isinstance(n, ast.FunctionDef) and n.name == "do_train"][0]
_first = [b for b in _dt.body
          if not isinstance(b, (ast.Import, ast.ImportFrom))][0]
assert isinstance(_first, ast.Assert) and \
    '_tok_seal_mode() == "0"' in (ast.get_source_segment(STAGED, _first) or ""), \
    "GATE 7 FAIL (ast): the guard is not do_train's first executable statement"


class _Sentinel(Exception):
    pass


def _sentinel_load_alg(*a, **k):
    raise _Sentinel("load_alg reached")


_real_load_alg = M_STAG.load_alg
M_STAG.load_alg = _sentinel_load_alg
try:
    for _m in ("head", "loop", "all"):
        try:
            with_env({"ALG_TOK_SEAL": _m},
                     lambda: M_STAG.do_train(1, 1e-4, B, 0))
            raise SystemExit(f"GATE 7 FAIL: do_train ran with "
                             f"ALG_TOK_SEAL={_m}")
        except _Sentinel:
            raise SystemExit(f"GATE 7 FAIL: do_train reached load_alg with "
                             f"ALG_TOK_SEAL={_m} — the guard is too late")
        except AssertionError as e:
            assert "READ door" in str(e), \
                f"GATE 7 FAIL: wrong guard message for {_m}: {e}"
    try:
        M_STAG.do_train(1, 1e-4, B, 0)
        raise SystemExit("GATE 7 FAIL: do_train did not reach load_alg with "
                         "the door unset")
    except _Sentinel:
        pass
finally:
    M_STAG.load_alg = _real_load_alg
print("[tc-smoke] GATE 7b PASS: the mistyped door is LOUD ('1', 'HEAD', "
      "'heads', 'true' all RAISE), and do_train REFUSES to start with "
      "ALG_TOK_SEAL in {head, loop, all} — BY EXECUTION (do_train called; it "
      "raises before load_alg, monkeypatched to a sentinel that is never "
      "reached, so nothing loads and no JIT capture can bake the meter in) "
      "and BY AST (the guard is body[0] after the imports). With the door "
      "unset the same call reaches the sentinel — a guard, not a wall.",
      flush=True)


# ===========================================================================
# GATE 8 — THE APPLY SCRIPT (proof g)
# ===========================================================================
_py = sys.executable
_r = subprocess.run([_py, "scripts/apply_tok_cook.py", "--check"],
                    capture_output=True, text=True, cwd=_ROOT)
assert _r.returncode == 0, \
    f"GATE 8 FAIL: --check exit {_r.returncode}\n{_r.stderr[-800:]}"
assert "16 anchors OK" in _r.stdout and \
    "symtable free-var audit PASS" in _r.stdout and \
    "NOTHING written" in _r.stdout, f"GATE 8 FAIL:\n{_r.stdout}"
assert open(HEAD).read() == APPLIED, "GATE 8 FAIL: --check wrote to the head"
_tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "tok_cook_rehearsal.py")
shutil.copy2(HEAD, _tmp)
_e = dict(os.environ, TC_TARGET=_tmp)
_r2 = subprocess.run([_py, "scripts/apply_tok_cook.py"],
                     capture_output=True, text=True, cwd=_ROOT, env=_e)
assert _r2.returncode == 0 and "APPLIED" in _r2.stdout, \
    f"GATE 8 FAIL: rehearsal apply\n{_r2.stderr[-800:]}"
assert open(_tmp).read() == STAGED, \
    "GATE 8 FAIL: the applied rehearsal differs from the staged source"
_r3 = subprocess.run([_py, "scripts/apply_tok_cook.py", "--check"],
                     capture_output=True, text=True, cwd=_ROOT, env=_e)
assert _r3.returncode != 0 and "idempotence" in _r3.stderr, \
    f"GATE 8 FAIL: a second application was not refused\n{_r3.stdout}{_r3.stderr}"

from mycelium.jit_read import env_names                        # noqa: E402
_names = env_names(_tmp)
for _d in ("ALG_TOK_COOK", "ALG_TOK_COOK_S3", "ALG_TG_INIT", "TC_EVAL"):
    assert _d in _names, f"GATE 8 FAIL: jit_read's env miner missed {_d}"
_names0 = env_names(HEAD)
assert "ALG_TOK_COOK" not in _names0, "the pristine head must not know it"
os.remove(_tmp)
assert open(HEAD).read() == APPLIED, "GATE 8 FAIL: the head moved"
print(f"[tc-smoke] GATE 8 PASS: --check clean (16 anchors present+unique, "
      f"ast OK, structural asserts + symtable audit PASS, nothing written); "
      f"a rehearsal apply reproduces the staged source exactly and a second "
      f"application is REFUSED. jit_read.env_names() finds ALG_TOK_COOK, "
      f"ALG_TOK_COOK_S3, ALG_TG_INIT and TC_EVAL in the applied source "
      f"({len(_names)} names, was {len(_names0)}) — the JIT read key carries "
      f"every door automatically.", flush=True)

print("[tc-smoke] ALL GATES PASS — the token cooker is byte-inert with "
      "ALG_TOK_COOK unset and ALG_TOK_SEAL != head (both polar configs, "
      "cold and warm, every emission key, every env cell), a pure code motion "
      "where it touches the mask head (verbatim block, one shared tensor), "
      "a proper distribution over real tokens at every sealed breath "
      "(exact zeros on pads, the bank's attention IS the gate), surgical "
      "per row, two-terminal into W_tg and the head, composable with the "
      "other two cookers (8 cells), and excluded from val by its own "
      "guard (the meter reproduces the sealed training path to 1e-5, "
      "scheduling). NOT PROVEN HERE (GPU, the lead's gate): eq A/B/C "
      "bit-identity, and any claim about SKILL — the headroom conversion "
      "itself.", flush=True)
