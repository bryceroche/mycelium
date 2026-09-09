"""census_organs_smoke.py — THE PORT CENSUS ORGAN-PASS SMOKE (CPU, zero
GPU, zero training, zero gradient descent; 2026-09-08).

Proves scripts/apply_census_organs.py's contract against the STAGED
source — the patch is built in memory (apply_census_organs.py forced to
--check) and exec'd into in-memory modules, the polar_sink_smoke.py
idiom — so the real head is NEVER written. Fixture: .cache/pc_row_smoke
.py's (banked test23 states, first 8 rows, build_params(0) random init:
this smoke tests MECHANISM, not skill).

  GATE 1 — INERTNESS WITH THE HOOK UNARMED (the contract that lets this
     patch ride the deployed stack): applied head vs staged head, same 8
     rows, np.array_equal on EVERY emission key of the two-pass read.
     Run twice: (1a) champion regime, ALG_POLAR unset; (1b) the polar
     sink regime, ALG_POLAR=1 ALG_POLAR_D=128 ALG_POLAR_EM=0.1.
  GATE 2 — ARMING IS A CROSS-GRAPH READ, AND THE ORGAN PASS ADDS NO NEW
     KIND OF DISTURBANCE. Arming _CENSUS forces mid-graph realize()s,
     which move KERNEL BOUNDARIES and therefore f32 scheduling noise —
     this is a property of the gen-1 hook (2026-09-01), not of this
     patch, and the gate measures it as such: the APPLIED head's own
     armed-vs-unarmed deviation is the baseline, and the STAGED head's
     must be the same order (both <= ATOL). Stated, not hidden: the
     census reading is not bit-identical to the ordinary read.
  GATE 3 — COVERAGE: every named organ is recorded, at every breath it
     can fire, finite, with the shape its band implies. The champion
     regime must yield the 8 non-sink organs; the sink regime all 10.
     Also reports what the census does NOT see (the honest residue).
  GATE 4 — GAIN LEGIBILITY (the point of the pass): with the zero-init
     doors opened by hand (mh_wo / mh_headmix / fed_mx_hg / W_bo given
     small random values — no training, no gradient), rms(post) ==
     gain * rms(pre) to f32 for maskhead, alt, altfact and mixer, on
     every breath whose pre is not itself exactly zero (at random init
     the snap adjacency can be empty; there post is exact zeros too and
     0/0 is not a ratio — counted and reported, never divided). This is
     what makes "silent" and "loud-times-tiny" different readings.
  GATE 5 — the apply script's --check: anchors present and unique, ast
     OK, structural asserts + symtable audit PASS, the head on disk is
     BYTE-UNCHANGED, and a second application is refused (idempotence).

Run from the repo root:
  .venv/bin/python3 scripts/census_organs_smoke.py
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
POLAR_ENV = {"ALG_POLAR": "1", "ALG_POLAR_D": str(D_WIDTH),
             "ALG_POLAR_EM": str(KAPPA)}
NEW_ORGANS = ("altfact", "altfact_pre", "maskhead", "maskhead_pre",
              "alt", "alt_pre", "mixer", "mixer_pre",
              "sink_waist", "sink_em")
# the three the ledger quotes — these MUST survive the extension untouched
LEDGER_ORGANS = ("breath_emb", "notebook", "garage")
# the other two gen-1 organs are env-gated (detwave: ALG_DETWAVE, router:
# ALG_ROUTER) and are DORMANT in the champion family — reported, not asserted
OLD_ORGANS = LEDGER_ORGANS + ("detwave", "router(bank)")


# ===========================================================================
# STAGING — build the would-be patched source without writing the head
# ===========================================================================
def staged_source():
    argv0 = sys.argv
    sys.argv = ["apply_census_organs.py", "--check"]
    try:
        ns = runpy.run_path(os.path.join(_ROOT, "scripts",
                                         "apply_census_organs.py"),
                            run_name="_census_organs_staging")
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
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


APPLIED = open(HEAD).read()
STAGED = staged_source()
assert APPLIED != STAGED and '"maskhead"' not in APPLIED
assert '_CENSUS.append((kb, "state"' in APPLIED, \
    "the gen-1 census hook must already be applied — this patch extends it"
print(f"[organ-smoke] staged +{STAGED.count(chr(10)) - APPLIED.count(chr(10))}"
      f" lines on the APPLIED head; head on disk UNTOUCHED", flush=True)

M_BASE = load_module(APPLIED, "applied_plain")
M_STAG = load_module(STAGED, "staged_plain")
M_BASE_P = with_env(POLAR_ENV, lambda: load_module(APPLIED, "applied_polar"))
M_STAG_P = with_env(POLAR_ENV, lambda: load_module(STAGED, "staged_polar"))
assert not M_STAG.ALG_POLAR and M_STAG_P.ALG_POLAR
assert M_STAG_P.POLAR_D == D_WIDTH and M_STAG_P.POLAR_EM == KAPPA


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
print(f"[organ-smoke] fixture test23 rows={B} dev={os.environ['DEV']} "
      f"d={D_WIDTH} kappa={KAPPA}", flush=True)


def masked_read(M, p):
    """loop_val's two-pass shape, WITH the fact conditioning pass 1 gets in
    the deployed read path (ALG_ALT2): without it the alternator injection
    is skipped entirely and the mask head sees a zero fact port — the census
    would then measure a configuration the machine never runs."""
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
    """Every emission the forward returns, flattened to name -> ndarray."""
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


# ===========================================================================
# GATE 1 — INERTNESS WITH THE HOOK UNARMED
# ===========================================================================
for lab, MA_, MB_, env in (
        ("champion regime, ALG_POLAR unset", M_BASE, M_STAG, {}),
        (f"polar sink regime (D={D_WIDTH}, EM={KAPPA})",
         M_BASE_P, M_STAG_P, POLAR_ENV)):
    def _run():
        pa = MA_.build_params(0)
        pb = MB_.build_params(0)
        assert set(pa) == set(pb), \
            f"GATE 1 FAIL ({lab}): parameter set changed"
        assert all(np.array_equal(pa[k].numpy(), pb[k].numpy()) for k in pa), \
            f"GATE 1 FAIL ({lab}): the patch moved the init rng stream"
        MA_._CENSUS = None
        MB_._CENSUS = None
        return MA_, pa, MB_, pb, masked_read(MA_, pa), masked_read(MB_, pb)
    _ma, _pa, _mb2, _pb, oa, ob = with_env(env, _run)
    assert set(oa) == set(ob), f"GATE 1 FAIL ({lab}): emission keys differ"
    bad = [k for k in oa if not np.array_equal(oa[k], ob[k])]
    assert not bad, f"GATE 1 FAIL ({lab}): NOT bit-identical ({bad[:6]})"
    print(f"[organ-smoke] GATE 1 PASS [{lab}]: applied vs staged "
          f"BIT-IDENTICAL on {len(oa)} emission tensors, hook UNARMED; "
          f"params identical", flush=True)


# ===========================================================================
# GATE 2 + 3 — ARMING IS CROSS-GRAPH; EVERY ORGAN IS RECORDED
# ===========================================================================
ATOL = 1e-4        # the pc_row_smoke cross-graph convention (logit scale ~1)


def dev_of(a, b):
    return max(float(np.abs(a[k] - b[k]).max()) for k in a
               if np.issubdtype(a[k].dtype, np.floating))


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


# the BASELINE: the gen-1 hook's own armed-vs-unarmed deviation, on the
# APPLIED head — 5 organs' realize()s, no organ pass anywhere near it
_p1 = M_BASE.build_params(0)
_b1, _a1, _r1 = census_read(M_BASE, _p1, {})
DEV_GEN1 = dev_of(_b1, _a1)
assert DEV_GEN1 <= ATOL, \
    (f"GATE 2 FAIL: the GEN-1 hook alone already deviates {DEV_GEN1:.3g} "
     f"> {ATOL} — the instrument, not this patch")
print(f"[organ-smoke] GATE 2 baseline [gen-1 hook, APPLIED head]: arming "
      f"moves the emissions by {DEV_GEN1:.3g} max-abs (mid-graph realize() "
      f"moves kernel boundaries; f32 scheduling noise). "
      f"{len(_r1)} records, {len({o for _, o, _ in _r1})} organs.", flush=True)

REGIMES = (("champion (ALG_POLAR unset)", M_STAG, {},
            tuple(o for o in NEW_ORGANS if not o.startswith("sink"))),
           (f"polar sink (D={D_WIDTH}, EM={KAPPA})", M_STAG_P, POLAR_ENV,
            NEW_ORGANS))
CENSUS = {}
for lab, M, env, expect in REGIMES:
    p = with_env(env, lambda: M.build_params(0))
    base, armed, rec = census_read(M, p, env)
    dev = dev_of(base, armed)
    assert dev <= ATOL, \
        (f"GATE 2 FAIL ({lab}): arming the organ pass moves the forward by "
         f"{dev:.3g} > {ATOL} (gen-1 baseline {DEV_GEN1:.3g})")
    print(f"[organ-smoke] GATE 2 PASS [{lab}]: arming moves the emissions "
          f"by {dev:.3g} max-abs over {len(base)} tensors — same order as "
          f"the gen-1 baseline {DEV_GEN1:.3g}, both << {ATOL}", flush=True)

    seen = {}
    for (kb, organ, arr) in rec:
        assert np.isfinite(arr).all(), \
            f"GATE 3 FAIL ({lab}): {organ} at b{kb} is not finite"
        seen.setdefault(organ, []).append((kb, arr))
    CENSUS[lab] = seen
    missing = [o for o in expect if o not in seen]
    assert not missing, f"GATE 3 FAIL ({lab}): organs never recorded {missing}"
    for o in LEDGER_ORGANS:
        assert o in seen, \
            f"GATE 3 FAIL ({lab}): gen-1 organ {o} lost by the extension"
    dormant = [o for o in OLD_ORGANS if o not in seen]
    # the two band baselines a ratio needs a denominator from
    assert "state" in seen and "state_slot" in seen, \
        f"GATE 3 FAIL ({lab}): a band baseline is missing"
    for kb, a in seen["state_slot"]:
        assert a.shape[-2:] == (M.L_TOT, M.L_TOT), \
            f"GATE 3 FAIL ({lab}): state_slot {a.shape} is not the sc2 band"
    assert any(kb == 0 for kb, _ in seen["state"]), \
        f"GATE 3 FAIL ({lab}): no kb=0 vst baseline for the fact injection"
    # bands: the shapes the names promise
    K_LOOP = int(os.environ["ALG_BREATH"]) - 1
    for o in ("mixer", "mixer_pre"):
        for kb, a in seen[o]:
            assert a.shape == (B, M.L_TOT, M.H_W), \
                f"GATE 3 FAIL ({lab}): {o} shape {a.shape} is not state-band"
    for o in ("maskhead", "maskhead_pre", "alt", "alt_pre"):
        for kb, a in seen[o]:
            assert a.shape[-2:] == (M.L_TOT, M.L_TOT), \
                f"GATE 3 FAIL ({lab}): {o} shape {a.shape} is not score-band"
    for o in ("altfact", "altfact_pre"):
        assert all(kb == 0 for kb, _ in seen[o]), \
            f"GATE 3 FAIL ({lab}): {o} must be recorded at kb=0 (pre-loop)"
        for kb, a in seen[o]:
            assert a.shape == (B, M.K_VARS, M.H_W), \
                f"GATE 3 FAIL ({lab}): {o} shape {a.shape} is not vst-band"
    for o in ("sink_waist", "sink_em"):
        if o in seen:
            kbs = sorted(kb for kb, _ in seen[o])
            assert kbs == list(range(1, K_LOOP + 1)), \
                f"GATE 3 FAIL ({lab}): {o} fired at {kbs}, expected every breath"
            for kb, a in seen[o]:
                assert a.shape == (B, M.L_TOT, M.H_W), \
                    f"GATE 3 FAIL ({lab}): {o} shape {a.shape} is not state-band"
    print(f"[organ-smoke] GATE 3 PASS [{lab}]: {len(seen)} organs recorded "
          f"({', '.join(sorted(seen))}) — all finite, all in-band; "
          f"env-dormant gen-1 organs (not created by this env, so nothing "
          f"to record): {dormant or 'none'}", flush=True)


# ===========================================================================
# GATE 4 — GAIN LEGIBILITY: rms(post) == gain * rms(pre)
# ===========================================================================
M = M_STAG_P
p4 = with_env(POLAR_ENV, lambda: M.build_params(0))
_rs = np.random.RandomState(9081)


def _poke(key, scale):
    """Open a zero-init door BY HAND (no training, no gradient) so the
    post/pre ratio has something to divide. Mechanism, not skill."""
    if key in p4:
        sh = tuple(int(d) for d in p4[key].shape)
        p4[key].assign(Tensor((_rs.randn(*sh) * scale).astype(np.float32),
                              dtype=dtypes.float)).realize()


for _k, _sc in (("mh_wo", 0.05), ("mh_headmix", 0.05),
                ("fed_mx_hg", 0.30), ("W_bo", 0.05)):
    _poke(_k, _sc)
_base4, _armed4, _rec4 = census_read(M, p4, POLAR_ENV)
seen4 = {}
for (kb, organ, arr) in _rec4:
    seen4.setdefault(organ, {})[kb] = arr
GAINS = {"maskhead": "mh_gain", "alt": "alt_g"}
for organ, gkey in GAINS.items():
    g = float(np.abs(p4[gkey].numpy()).reshape(-1)[0])
    live = dead = 0
    for kb, post in seen4[organ].items():
        pre = seen4[organ + "_pre"][kb]
        rp = float(np.sqrt((post ** 2).mean()))
        rq = float(np.sqrt((pre ** 2).mean()))
        if rq == 0.0:
            # the organ's OWN INPUT is exactly zero this breath — at random
            # init the snap one-hots can have no overlap at all, so the
            # adjacency _A5 is all zeros. post must then be exact zeros too
            # (that IS the reading: silent because there is nothing to say,
            # not silent because the gain closed it), and 0/0 is not a
            # ratio. Asserted, counted, and excluded from the identity.
            assert float(np.abs(post).max()) == 0.0, \
                f"GATE 4 FAIL: {organ} b{kb} post nonzero on a zero pre"
            dead += 1
            continue
        assert abs(rp - g * rq) <= 1e-5 * max(rp, 1e-6) + 1e-9, \
            (f"GATE 4 FAIL: {organ} b{kb} rms(post)={rp:.6g} != "
             f"{gkey}={g:.6g} * rms(pre)={rq:.6g}")
        live += 1
    assert live > 0, \
        f"GATE 4 FAIL: {organ} never had a nonzero pre — nothing measured"
    print(f"[organ-smoke] GATE 4 PASS [{organ}]: rms(post) == {gkey} "
          f"({g:.4g}) x rms(pre) on {live} breath(s)"
          + (f"; {dead} breath(s) had an exactly-zero input (post zero too)"
             if dead else ""), flush=True)
# ALTFACT AT THE ORGAN. In the forward, fact_buf comes from the pass-0
# PARSE, and a random-init head parses no known vars — the buffer is all
# zeros and the organ has nothing to say (a skill fact, not a mechanism
# fact). So the identity is tested where the organ lives, with a synthetic
# fact buffer: _fact_inject is called directly, which is also the exact
# entry point the step trainer uses per seam.
M._CENSUS = []
M._fact_inject(p4,
               Tensor(np.zeros((B, M.K_VARS, M.H_W), np.float32),
                      dtype=dtypes.float),
               Tensor(_rs.randn(B, M.K_VARS, 4).astype(np.float32),
                      dtype=dtypes.float)).realize()
_rf = {o: a for _, o, a in M._CENSUS}
M._CENSUS = None
assert set(_rf) == {"state", "altfact", "altfact_pre"}, \
    f"GATE 4 FAIL: _fact_inject recorded {sorted(_rf)}"
_g2 = float(np.abs(p4["alt2_g"].numpy()).reshape(-1)[0])
_rp = float(np.sqrt((_rf["altfact"] ** 2).mean()))
_rq = float(np.sqrt((_rf["altfact_pre"] ** 2).mean()))
assert _rq > 0 and abs(_rp - _g2 * _rq) <= 1e-5 * max(_rp, 1e-6) + 1e-9, \
    f"GATE 4 FAIL: altfact rms(post)={_rp:.6g} != alt2_g={_g2:.6g} * {_rq:.6g}"
print(f"[organ-smoke] GATE 4 PASS [altfact, at the organ]: rms(post) == "
      f"alt2_g ({_g2:.4g}) x rms(pre); the kb=0 vst baseline rides with it",
      flush=True)

# the mixer's gain is per-head, so the identity is per-head, not scalar:
_hg = p4["fed_mx_hg"].numpy().reshape(-1)
_mx_live = 0
for kb in sorted(seen4["mixer"]):
    rp = float(np.sqrt((seen4["mixer"][kb] ** 2).mean()))
    rq = float(np.sqrt((seen4["mixer_pre"][kb] ** 2).mean()))
    if rq == 0.0:
        assert rp == 0.0, f"GATE 4 FAIL: mixer b{kb} post nonzero on zero pre"
        continue
    assert rp > 0 and rp != rq, \
        f"GATE 4 FAIL: mixer post/pre indistinguishable at b{kb}"
    _mx_live += 1
assert _mx_live > 0, "GATE 4 FAIL: the mixer never spoke"
print(f"[organ-smoke] GATE 4 PASS [mixer]: per-head gains "
      f"(|fed_mx_hg| mean {np.abs(_hg).mean():.4g}) — post and pre both "
      f"live and distinct; the ratio is a per-head mixture, reported not "
      f"asserted", flush=True)


# ===========================================================================
# GATE 5 — the apply script's --check, and idempotence
# ===========================================================================
_h0 = hashlib.sha256(open(HEAD, "rb").read()).hexdigest()
_env5 = dict(os.environ, PS_TARGET=HEAD)
r = subprocess.run([sys.executable, "scripts/apply_census_organs.py",
                    "--check"], capture_output=True, text=True, env=_env5)
assert r.returncode == 0, f"GATE 5 FAIL: --check errored\n{r.stdout}{r.stderr}"
assert "NOTHING written" in r.stdout, "GATE 5 FAIL: --check did not say so"
assert hashlib.sha256(open(HEAD, "rb").read()).hexdigest() == _h0, \
    "GATE 5 FAIL: --check WROTE to the head"
with tempfile.TemporaryDirectory() as td:
    _tmp = os.path.join(td, "phase1_algebra_head.py")
    open(_tmp, "w").write(STAGED)
    r2 = subprocess.run([sys.executable, "scripts/apply_census_organs.py",
                         "--check"], capture_output=True, text=True,
                        env=dict(os.environ, PS_TARGET=_tmp))
    assert r2.returncode != 0 and "idempotence" in r2.stderr, \
        f"GATE 5 FAIL: a second application was not refused\n{r2.stderr[-400:]}"
print("[organ-smoke] GATE 5 PASS: --check writes nothing (head sha "
      f"{_h0[:12]} unchanged); a second application is REFUSED "
      "(idempotence guard)", flush=True)


# ===========================================================================
# THE RESIDUE — what the census STILL does not see (stated, not hidden)
# ===========================================================================
print("[organ-smoke] RESIDUE (additive injections into cur / q_extra / "
      "the score space that remain UNCENSUSED after this pass): "
      "notebook lane 2 (fed_nb_g), the sync receiver oscillator, the "
      "BEXIT commit-mass bias, ALT21 stations 3-4 (_d21a/_d21b), the "
      "_IMP systems-ID kick, and the STELLAR/CIRCLE/seal state "
      "REPLACEMENTS (which are not injections). See the report.")
print("[organ-smoke] ALL GATES PASS — CPU only, no GPU, no training, "
      "head on disk untouched")
