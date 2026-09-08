"""apply_polar_waist.py — THE POLAR WAIST, staged patch (2026-09-07,
word given; ledger: "WORD GIVEN: THE POLAR WAIST GENERATION"; spec:
docs/polar_waist_spec.md, which is the contract this script implements
sections 1-4 of).

WHY (the reasons, all measured, all in the ledger of 2026-09-07):
  * THE CLOCK READ (rung 1, FAIL): breath-identity probe 0.6011 mint /
    0.5325 wild against a 0.95 bar; Procrustes eigen-angles put 0.000
    of the variance mass in [50,80) deg and 0.98 in [0,20). The state
    CONSOLIDATES (per-breath norm 7.0 -> 11.9, monotone) and does not
    TURN. The machine knows breath 0 and breath 1 and then loses count.
  * RUNG 0a: the only rotor in the head (fed item 7a) is a Q-side 60-deg
    turn on 8 of 32 pairs per head, behind mixer gains fed_mx_hg whose
    abs-mean woke to 0.023 and SHRANK under the cooker. A whisper the
    state never heard. Neither BREATH_BAND_PAIRS nor ROT_BANDS ever
    governed a trained head (the bands note).
  * THE AMPLITUDE READ: champion stamps median 412, max 1.9e4, against
    a state radius of 7-12 — three to four orders of magnitude louder
    than a trained stamp (~4); and 0 under the live wire without a
    floor. Amplitude was never a calibrated coordinate.
A rotation is invisible on a growing radius. The sextet has nowhere to
turn. So: make the radius an explicit coordinate and give the sextet a
sphere to turn on.

WHAT THIS PATCH DOES (spec S1-S4; ALG_POLAR unset = byte-identical):
  1. THE POLAR STATE. At the per-breath update the loop state becomes
     r * u: u on the unit sphere of the T^256 torus (256 planes of the
     512-d waist — the bus's own geometry), r an explicit non-negative
     radius channel (the consolidation coordinate; DIAGNOSTIC REGISTER,
     never in a loss). The re-parametrization sits AFTER the pawl and
     AFTER the seal and BEFORE the notebook ink / garage write /
     breaths append, so every organ downstream reads ONE state in ONE
     clock, and the sealed crossing's state carries (r, u) too.
  2. THE GUARANTEED SEXTET. u's CLOCK BANDS turn by rotor_clock's
     frozen wheel increments every loop breath kb = 1..6 (breath hand
     60 deg, parity 120 deg, pass wheel static), no gains, no learnable
     rate, no schedule; breath 0 is outside time. The state applies
     DELTAS (it compounds), so its accumulated phase at breath k equals
     the master clock's ABSOLUTE phase — asserted against
     rotor_clock.wheel_table() at table-build time. The SAME allocation
     drives a Q-side rotation of the mixer attention (ABSOLUTE angles:
     the query is rebuilt from cur each breath, no compounding; K
     unrotated — the v109pi relative-phase precedent), REPLACING fed
     item 7a's gated 8-pair whisper when ALG_POLAR is set and leaving
     it byte-identical when unset.
  3. FIX B, THE DEPOSIT'S RADIUS. The confidence stamp becomes the
     deposit's radius channel, calibrated to the slot's own state
     radius: w' = w*R/(R+w), R detached. w' -> w for w << R (the
     healthy regime untouched), w' -> R for w >> R (a deposit can never
     shout louder than the state that wrote it), and dw'/dw =
     (R/(R+w))^2 is bounded and NEVER zero. ALG_PC_FLOOR rides
     underneath unchanged: the floor sets the bottom, the radius the
     top.
  4. THE TAP. breaths_u (and breaths_r) beside breaths_all, both
     DETACHED — clock_read.py can probe the direction without the
     radius confound, and no diagnostic can teach r.
Numerics per CLAUDE.md S5: the normalization is where-gated TWICE —
the sqrt (the NaN-safe idiom already proven on the confidence stamp)
AND the denominator, which is replaced by 1.0 rather than by an
epsilon, so an exact-zero slot yields u = 0 and du/dx = 1 instead of a
1/eps = 1e6 backward spike at precisely the state the guard exists
for. No dtype literals inside the JIT (dtypes.float only, the item-7a
idiom); content planes multiply by cos=1 / sin=0 and pass through
bitwise.

DOORS (spec S3), all printed once at build_params, all inert unset:
  ALG_POLAR=1            master
  ALG_POLAR_BANDS=path   default .cache/polar_bands.json (READ by the
                         head — the bands note's lesson: a band table
                         that no head reads never governed anything)
  ALG_POLAR_R_MODE       scalar (default) | slotvec
  ALG_POLAR_RG           slotvec group count (default 8)
  ALG_POLAR_QROT=1       Q-side mixer rotation (default on)
  ALG_POLAR_STAMP=1      the deposit-radius fix B (default on)

--check: loads the file, asserts every anchor present and unique,
builds the would-be result, ast-parses it, runs the symtable free-
variable audit (the apply_pressure_mix.py idiom), and writes NOTHING.
PW_TARGET may point at a copy (rehearsal); default is
scripts/phase1_algebra_head.py. The CPU proofs of spec S5 items 4-5
live in scripts/polar_birth_smoke.py, which stages this patch in
memory exactly as seamtest_vector.py does.
"""
import ast
import builtins
import json
import os
import symtable
import sys

fn = os.environ.get("PW_TARGET", 'scripts/phase1_algebra_head.py')
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

# ---------------------------------------------------------------- guards
assert 'ALG_POLAR' not in s and '_polar_ru' not in s, \
    "polar waist already present — patch was applied; refuse (idempotence)"
assert '_dep4 = _canon4 / _cn4 * _wn4' in s, \
    "canonical-shelf deposit anchor missing — wrong vintage of the head"
assert 'ALG_PC_FLOOR' in s, \
    "the soft floor is missing — wrong vintage (fix B rides on top of fix A)"
assert '_PCV' in s, \
    "the pressure mix is missing — wrong vintage of the head"
assert 'from mycelium.rotor_clock import breath_qk_angles' in s, \
    "rotor_clock is not imported by the head — wrong vintage"

# --------------------------------------------------- the band file must exist
BANDS = os.environ.get("ALG_POLAR_BANDS", ".cache/polar_bands.json")
assert os.path.exists(BANDS), \
    f"band allocation {BANDS} missing — the head READS it (spec S2)"
_bj = json.load(open(BANDS))
assert _bj["waist"] == 512 and _bj["n_planes"] == 256 and len(_bj["wheels"]) == 3
_seen = set()
for _w in _bj["wheels"]:
    for _p in _w["planes"]:
        assert 0 <= _p < 256 and _p not in _seen, "band overlap in the json"
        _seen.add(_p)
assert len(_seen) == _bj["n_clocked"]

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# ===========================================================================
# 1. MODULE LEVEL — the doors, the band reader, the frozen wheel tables,
#    and the two polar organs (_polar_ru / _polar_ru_join). rotor_clock
#    becomes the single source of truth BY IMPORT (spec S0).
# ===========================================================================
patch(1, "module: polar doors + band reader + frozen sextet tables + (r,u) organs",
      '''    _FED_ROT_C[:, 24:32] = np.cos(_fed_ang).astype(np.float32)
    _FED_ROT_S[:, 24:32] = np.sin(_fed_ang).astype(np.float32)
SENT_MAX = 32''',
      '''    _FED_ROT_C[:, 24:32] = np.cos(_fed_ang).astype(np.float32)
    _FED_ROT_S[:, 24:32] = np.sin(_fed_ang).astype(np.float32)
# ===========================================================================
# THE POLAR WAIST (apply_polar_waist.py, 2026-09-07; docs/polar_waist_spec
# .md). The loop state as r * u: u on the unit sphere of the T^256 torus,
# r an explicit radius channel. The sextet turns u's CLOCK BANDS every
# loop breath with FROZEN, UNGAINED angles from mycelium/rotor_clock.py —
# which is the single source of truth here BY IMPORT, not by prose.
# Every line below is inert with ALG_POLAR unset (the band file is not
# even opened): the equivalence contract is eq A/B/C bit-identical.
# ===========================================================================
from mycelium.rotor_clock import (N_LOOP as _RC_N_LOOP,           # noqa: E402
                                  N_WHEELS as _RC_N_WHEELS,
                                  QUANTUM as _RC_QUANTUM,
                                  wheel_table as _rc_wheel_table)
ALG_POLAR = int(os.environ.get("ALG_POLAR", "0"))
POLAR_BANDS = os.environ.get("ALG_POLAR_BANDS", ".cache/polar_bands.json")
POLAR_R_MODE = os.environ.get("ALG_POLAR_R_MODE", "scalar")   # scalar|slotvec
POLAR_RG = int(os.environ.get("ALG_POLAR_RG", "8"))           # slotvec groups
POLAR_QROT = int(os.environ.get("ALG_POLAR_QROT", "1"))       # Q-side rotation
POLAR_STAMP = int(os.environ.get("ALG_POLAR_STAMP", "1"))     # fix B
_POLAR_TAB = None           # (delta_cos, delta_sin, abs_cos, abs_sin, wheel_of)
_POLAR_SHOWN = False


def _polar_groups():
    """The radius channel's width: 1 scalar per slot (default) or
    POLAR_RG group radii (ALG_POLAR_R_MODE=slotvec). Groups are whole
    numbers of PLANES, so a band rotation never straddles a group and
    every group norm is rotation-invariant."""
    if POLAR_R_MODE == "scalar":
        return 1
    assert POLAR_R_MODE == "slotvec", \\
        f"ALG_POLAR_R_MODE={POLAR_R_MODE} (scalar|slotvec)"
    assert H_W % POLAR_RG == 0 and (H_W // POLAR_RG) % 2 == 0, \\
        f"ALG_POLAR_RG={POLAR_RG} must split H_W={H_W} into even-sized groups"
    return POLAR_RG


def _polar_tables():
    """THE FROZEN WHEEL TABLES, built once from the band allocation the
    head READS (POLAR_BANDS) and the angles rotor_clock OWNS.
    Returns (dc, ds, ac, as_, wheel_of):
      dc/ds  (N_LOOP, P) cos/sin of the per-breath INCREMENT — what the
             STATE applies (it compounds: u_k = rot(u_{k-1}, delta_k));
      ac/as_ (N_LOOP, P) cos/sin of the ABSOLUTE phase — what the
             Q-SIDE applies (the query is rebuilt from cur each breath,
             so there is nothing to compound);
      wheel_of (P,) the owning wheel per plane, -1 = content.
    Content planes carry cos=1, sin=0: bit-identical passthrough.
    THE COMPOUNDING CONTRACT is asserted here, not asserted in prose:
    the accumulated increments must reproduce rotor_clock's absolute
    table exactly."""
    global _POLAR_TAB
    if _POLAR_TAB is None:
        import json as _pjs
        _P = H_W // 2
        assert _P == MX_HEADS * ((H_W // MX_HEADS) // 2), \\
            "plane count must reshape to (MX_HEADS, pairs/head) for the Q side"
        with open(POLAR_BANDS) as _pf:
            _pb = _pjs.load(_pf)
        assert int(_pb["waist"]) == H_W and int(_pb["n_planes"]) == _P, \\
            f"{POLAR_BANDS}: waist/plane mismatch (head H_W={H_W})"
        assert len(_pb["wheels"]) == _RC_N_WHEELS, \\
            f"{POLAR_BANDS}: {len(_pb['wheels'])} wheels, clock has {_RC_N_WHEELS}"
        _wof = np.full(_P, -1, np.int64)
        for _wi, _wd in enumerate(_pb["wheels"]):
            assert int(_wd["wheel"]) == _wi, "wheels out of order"
            for _pl in _wd["planes"]:
                _pl = int(_pl)
                assert 0 <= _pl < _P, f"plane {_pl} out of range"
                assert _wof[_pl] < 0, \\
                    f"plane {_pl} claimed twice — SEPARATE BANDS is a law"
                _wof[_pl] = _wi
        _abs = _rc_wheel_table()                      # (N_LOOP, N_WHEELS)
        _dlt = np.zeros_like(_abs)
        _dlt[0] = _abs[0]
        _dlt[1:] = _abs[1:] - _abs[:-1]
        assert np.allclose(np.cos(np.cumsum(_dlt, 0)), np.cos(_abs), atol=1e-6) \\
            and np.allclose(np.sin(np.cumsum(_dlt, 0)), np.sin(_abs), atol=1e-6), \\
            "state increments do not accumulate to rotor_clock's absolute phase"
        assert np.allclose(np.cos(_dlt[1:, 0]), math.cos(_RC_QUANTUM)) \\
            and np.allclose(np.sin(_dlt[1:, 0]), math.sin(_RC_QUANTUM)), \\
            "the breath hand must advance exactly one quantum (60 deg)/breath"
        _dc = np.ones((_RC_N_LOOP, _P), np.float32)
        _ds = np.zeros((_RC_N_LOOP, _P), np.float32)
        _ac = np.ones((_RC_N_LOOP, _P), np.float32)
        _as = np.zeros((_RC_N_LOOP, _P), np.float32)
        for _pl in range(_P):
            _wi = int(_wof[_pl])
            if _wi < 0:
                continue                              # content: identity
            _dc[:, _pl] = np.cos(_dlt[:, _wi]); _ds[:, _pl] = np.sin(_dlt[:, _wi])
            _ac[:, _pl] = np.cos(_abs[:, _wi]); _as[:, _pl] = np.sin(_abs[:, _wi])
        _POLAR_TAB = (_dc, _ds, _ac, _as, _wof)
    return _POLAR_TAB


def _polar_ru(x, g):
    """THE POLAR DECOMPOSITION — the ONE organ every caller uses (the
    meter-divergence law: a check must call its organ). x: (B, L, H_W).
    Returns (r, u) with r (B, L, g, 1) the per-group radius and u the
    unit direction; x == r*u EXACTLY, including at the origin.
    TWO WHERE-GATES, both load-bearing (CLAUDE.md S5; the confidence
    stamp's own NaN-safe idiom, extended):
      * the SQRT gate — an exact-zero slot must not send grad/(2*sqrt(0))
        to NaN (the stamp's step-5000 detonation);
      * the DENOMINATOR gate — the radius is replaced by 1.0 (never by
        an epsilon) in the division, so at the origin u is exactly zero
        and du/dx is exactly 1. An epsilon guard is NOT enough here: it
        keeps the forward finite but hands the backward a 1/eps = 1e6
        spike at precisely the state the guard exists for. The reported
        radius stays the TRUE radius (0 at the origin) — the guard
        lives in the division, never in the coordinate.
    r is a DIAGNOSTIC-REGISTER coordinate: read it, never supervise it."""
    _xg = x.reshape(x.shape[0], x.shape[1], g, -1)
    _ss = _xg.pow(2).sum(-1, keepdim=True)
    _sp = _ss > 0
    _r = _sp.where(_sp.where(_ss, 1.0).sqrt(), 0.0)   # true radius, 0 at origin
    _rd = _sp.where(_r, 1.0)                          # the guarded denominator
    return _r, (_xg / _rd).reshape(x.shape)


def _polar_ru_join(u, r, g):
    """r * u back to the old coordinates — what every existing organ
    reads (spec S1.3: reads are polar OR cartesian, the caller's choice;
    with ALG_POLAR unset this function is never called at all)."""
    return (u.reshape(u.shape[0], u.shape[1], g, -1) * r).reshape(u.shape)
SENT_MAX = 32''')

# ===========================================================================
# 2. build_params — every door prints its state ONCE (spec S3).
# ===========================================================================
patch(2, "build_params: the polar door prints its state once",
      '''def build_params(seed=0):
    from tinygrad import Tensor, dtypes
    rng = np.random.RandomState(seed)''',
      '''def build_params(seed=0):
    from tinygrad import Tensor, dtypes
    rng = np.random.RandomState(seed)
    global _POLAR_SHOWN
    if ALG_POLAR and not _POLAR_SHOWN:
        # THE POLAR DOOR (spec S3): one line, once, naming the frozen
        # allocation it is about to run under. Silent when unset.
        _POLAR_SHOWN = True
        _pd0, _pd1, _pa0, _pa1, _pwof = _polar_tables()
        _pn = [int((_pwof == _wi).sum()) for _wi in range(_RC_N_WHEELS)]
        print(f"[polar] ALG_POLAR=1 bands={POLAR_BANDS} "
              f"planes={H_W // 2} clocked={sum(_pn)} "
              f"(breath-hand {_pn[0]} / parity {_pn[1]} / pass {_pn[2]}) "
              f"content={H_W // 2 - sum(_pn)} | R_MODE={POLAR_R_MODE}"
              f"({_polar_groups()} group(s)) QROT={POLAR_QROT} "
              f"STAMP={POLAR_STAMP} FLOOR={os.environ.get('ALG_PC_FLOOR', '0')} "
              f"| sextet = mycelium/rotor_clock.wheel_table() "
              f"(frozen, ungained, breath-0 outside time)", flush=True)''')

# ===========================================================================
# 3. breath_step — THE POLAR STATE and THE GUARANTEED SEXTET (spec S1.1-1.3
#    and the S4 anchor "cur = m_c*anchor + (1-m_c)*cur_new / breaths.append").
# ===========================================================================
patch(3, "breath_step: the polar state (r, u) + the sextet on u's clock bands",
      '''        cur = m_c * anchor + (1.0 - m_c) * cur_new
    else:
        cur = cur_new
    if ALG_NOTEBOOK:''',
      '''        cur = m_c * anchor + (1.0 - m_c) * cur_new
    else:
        cur = cur_new
    _pol_r = None
    if ALG_POLAR:
        # THE POLAR WAIST (apply_polar_waist.py, 2026-09-07; spec S1).
        # The loop state becomes r * u — u on the unit sphere of the
        # T^256 torus (256 planes of the 512-d waist, the bus's own
        # geometry), r the explicit radius channel (consolidation; the
        # measured 7 -> 12 growth becomes a COORDINATE instead of a
        # swamp). THE GUARANTEED SEXTET: u's clock bands turn by
        # rotor_clock's frozen wheel INCREMENTS (breath hand 60 deg,
        # parity 120 deg, pass wheel static) at every loop breath
        # 1..6 — no gains, no learnable rate, no schedule, breath 0
        # outside time; content planes multiply by cos=1/sin=0 and pass
        # through BITWISE. The write that just happened (cur_new) lands
        # in u, which is re-normalized WHERE-GATED, and in r, which
        # rides its own channel: writes tangential, reads polar.
        # PLACEMENT is load-bearing: after the pawl and after the seal
        # (so a SEALED row's crossing state carries (r, u) and the
        # clock too — the seal's own 46x norm jump becomes a radius
        # reading rather than a coordinate break), and before the
        # notebook ink, the garage write and breaths.append (so every
        # downstream organ reads ONE state in ONE frame).
        _pg = _polar_groups()
        _pol_r, _pol_u = _polar_ru(cur, _pg)
        if 1 <= kb <= _RC_N_LOOP:
            from tinygrad import Tensor as _Tp, dtypes as _dp
            _pdc, _pds, _pac, _pas, _pwof = _polar_tables()
            _pol_u = _rot2(_pol_u,
                           _Tp(_pdc[kb - 1], dtype=_dp.float),
                           _Tp(_pds[kb - 1], dtype=_dp.float))
        cur = _polar_ru_join(_pol_u, _pol_r, _pg)
        if int(os.environ.get("ALG_MINE_BREATHS", "0")):
            # THE TAP (spec S4): u and r beside r*u, DETACHED — the
            # clock read probes the direction without the radius
            # confound, and no diagnostic can teach the radius
            # (the Goodhart fence; the two-terminal proof is
            # scripts/polar_birth_smoke.py item 5).
            state.setdefault("u_all", []).append(_pol_u.detach())
            state.setdefault("r_all", []).append(_pol_r.detach())
    if ALG_NOTEBOOK:''')

# ===========================================================================
# 4. breath_step — FIX B: the deposit's radius (spec S1.4 / S4).
# ===========================================================================
patch(4, "breath_step: fix B — the confidence stamp becomes the deposit's radius",
      '''            _cn4 = _canon4.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6
            _dep4 = _canon4 / _cn4 * _wn4''',
      '''            _cn4 = _canon4.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6
            if ALG_POLAR and POLAR_STAMP and _pol_r is not None:
                # THE DEPOSIT'S RADIUS — FIX B (spec S1.4; the amplitude
                # read of 2026-09-07: champion stamps median 412 / max
                # 1.9e4 against a state radius of 7-12, and 0 under the
                # live wire without the floor). The stamp becomes the
                # deposit's RADIUS CHANNEL, calibrated to THIS slot's own
                # state radius R:
                #        w' = w * R / (R + w)
                # w' -> w for w << R (the healthy regime is untouched),
                # w' -> R for w >> R (a deposit can never shout louder
                # than the state that wrote it), and dw'/dw = (R/(R+w))^2
                # is bounded and NEVER zero. The algebraic saturation is
                # chosen over tanh precisely here: tanh's gradient at the
                # champion's w/R ~ 40 is exp(-80) — a live wire DEAD at
                # birth, the very pathology the pressure campaign cured.
                # R is DETACHED: it is a SCALE, not a signal (an un-
                # detached R lets the committer earn a loud stamp by
                # shrinking the state — a Goodhart loop). ALG_PC_FLOOR
                # rides underneath, unchanged: the floor sets the bottom,
                # the radius sets the top; amplitude can neither shout
                # nor vanish.
                _rst4 = _pol_r.mean(-2).detach()          # (B, L, 1)
                _wn4 = _wn4 * _rst4 / (_rst4 + _wn4)
            _dep4 = _canon4 / _cn4 * _wn4''')

# ===========================================================================
# 5. breath_step (the mixer) — THE SEXTET, Q-SIDE: replaces fed item 7a's
#    gated 8-pair rotation when ALG_POLAR is set; item 7a stays byte-
#    identical when it is not (spec S1.2 / S4).
# ===========================================================================
patch(5, "breath_step: Q-side sextet on the mixer (replaces fed item 7a under ALG_POLAR)",
      '''        if FED_ROTOR and _FED_ROT_C is not None and 1 <= kb <= 6:''',
      '''        if ALG_POLAR and POLAR_QROT and 1 <= kb <= _RC_N_LOOP:
            # THE SEXTET, Q-SIDE (spec S1.2): the SAME plane allocation
            # as the state's clock, reshaped (MX_HEADS, pairs/head) —
            # state and attention are ONE clock, not two. ABSOLUTE
            # angles here (the query is rebuilt from cur every breath:
            # nothing compounds), K UNROTATED (the v109pi precedent:
            # one table on both sides cancels — relative phase is the
            # signal). UNCONDITIONAL: no gains, no learnable rate. This
            # REPLACES fed item 7a, whose 60 deg on 8 of 32 pairs sat
            # behind mixer gains of 0.023 and shrank to 0.013 under the
            # cooker (rung 0a) — a whisper the state never heard. With
            # ALG_POLAR unset item 7a below runs byte-identically.
            from tinygrad import Tensor as _Tq, dtypes as _dq
            _qdc, _qds, _qac, _qas, _qwof = _polar_tables()
            _rcq = _Tq(_qac[kb - 1].reshape(MX_HEADS, _mx_hd // 2),
                       dtype=_dq.float).reshape(1, MX_HEADS, 1, -1)
            _rsq = _Tq(_qas[kb - 1].reshape(MX_HEADS, _mx_hd // 2),
                       dtype=_dq.float).reshape(1, MX_HEADS, 1, -1)
            _qp8 = _mx_q.reshape(B, MX_HEADS, L_TOT, _mx_hd // 2, 2)
            _qx8, _qy8 = _qp8[..., 0], _qp8[..., 1]
            _mx_q = Tensor.stack(_qx8 * _rcq - _qy8 * _rsq,
                                 _qx8 * _rsq + _qy8 * _rcq, dim=-1) \\
                .reshape(B, MX_HEADS, L_TOT, _mx_hd)
        elif FED_ROTOR and _FED_ROT_C is not None and 1 <= kb <= 6:''')

# ===========================================================================
# 6. forward — the atlas tap keeps r*u and gains u (spec S4, breaths_all).
# ===========================================================================
patch(6, "forward: breaths_u / breaths_r exposed beside breaths_all",
      '''        out["breaths_all"] = [_fed_core(_b9) for _b9 in out_breaths]''',
      '''        out["breaths_all"] = [_fed_core(_b9) for _b9 in out_breaths]
        if ALG_POLAR:
            # SPEC S4: the atlas tap keeps the OLD coordinates (r*u) in
            # breaths_all — every banked atlas stays readable — and the
            # DIRECTION arrives beside it so clock_read.py can probe
            # both (angle = identity, radius = consolidation: the two-
            # channel law, now two keys). Breath 0 is OUTSIDE TIME
            # (rotor_clock's contract): unrotated, and its (r, u) come
            # from the SAME organ the loop uses — one meter, one
            # caller. Both taps are DETACHED: a diagnostic terminal
            # that cannot teach (the Goodhart fence).
            _pg0 = _polar_groups()
            _r0p, _u0p = _polar_ru(out_breaths[0], _pg0)
            out["breaths_u"] = [_fed_core(_x9) for _x9 in
                                ([_u0p.detach()]
                                 + ((_bs_state or {}).get("u_all") or []))]
            out["breaths_r"] = [_fed_core(_x9) for _x9 in
                                ([_r0p.detach()]
                                 + ((_bs_state or {}).get("r_all") or []))]''')

for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# ===========================================================================
# STRUCTURAL ASSERTS on the would-be module (cheap, no import, no GPU)
# ===========================================================================
# -- the organs exist exactly once, and every caller uses THEM (meter law)
for _d in ('def _polar_groups(', 'def _polar_tables(', 'def _polar_ru(',
           'def _polar_ru_join('):
    assert s.count(_d) == 1, f"{_d} not defined exactly once"
assert s.count('_polar_ru(') == 3, \
    "the (r, u) organ must have exactly its def + 2 callers (loop, breath-0)"
assert s.count('_polar_ru_join(') == 2, "def + the single recomposition site"
# -- rotor_clock is the source of truth BY IMPORT
assert s.count('from mycelium.rotor_clock import') == 2 and \
    '_rc_wheel_table' in s, "the wheel table must come from rotor_clock"
assert 'wheel_table()' not in s.split('_rc_wheel_table()')[0].replace(
    'wheel_table as _rc_wheel_table', ''), "no second wheel table anywhere"
# -- the sextet is ungained and unlearnable: no parameter is created here
assert 'p["polar' not in s and "p['polar" not in s, \
    "the polar waist adds ZERO parameters (frozen frequencies, no gains)"
# -- ordering: the polar block sits AFTER the pawl/seal and BEFORE the
#    notebook ink, the breaths append and the garage write
_i_pol = s.index('    _pol_r = None\n    if ALG_POLAR:')
_i_pawl = s.index('        cur = m_c * anchor + (1.0 - m_c) * cur_new')
_i_nb = s.index('    if ALG_NOTEBOOK:\n        _nb.append(')
_i_app = s.index('    breaths.append(cur)')
_i_gar = s.index('        # GARAGE WRITE (drop-off)')
_i_seal = s.index('            _cur_seal = cur * 0.0 + _inj4')
assert _i_seal < _i_pawl < _i_pol < _i_nb < _i_app < _i_gar, \
    "polar block misplaced (must be after pawl+seal, before ink/append/garage)"
# -- fix B must sit between the stamp and the deposit, and must not
#    disturb the floor
_i_floor = s.index('_wn4 = (_ss4 + _pcf4 * _pcf4).sqrt() + 1e-6')
_i_fixb = s.index('_rst4 = _pol_r.mean(-2).detach()')
_i_dep = s.index('            _dep4 = _canon4 / _cn4 * _wn4')
assert _i_floor < _i_fixb < _i_dep, "fix B must sit after the floor, before _dep4"
assert s.count('_wn4 = _wn4 * _rst4 / (_rst4 + _wn4)') == 1
assert '.tanh()' not in s.split('THE DEPOSIT\'S RADIUS')[1][:2000], \
    "fix B uses the algebraic saturation, never tanh (the vanishing-grad trap)"
# -- the Q side replaces item 7a rather than stacking on it
assert s.count('elif FED_ROTOR and _FED_ROT_C is not None') == 1 and \
    s.count('if FED_ROTOR and _FED_ROT_C is not None') == 1, \
    "item 7a must become the ELIF arm (replaced, not stacked)"
assert s.index('if ALG_POLAR and POLAR_QROT') < \
    s.index('elif FED_ROTOR and _FED_ROT_C is not None'), \
    "the polar Q rotation must take precedence over item 7a"
# -- every diagnostic tap is detached (the Goodhart fence, two-terminal)
for _t in ('state.setdefault("u_all", []).append(_pol_u.detach())',
           'state.setdefault("r_all", []).append(_pol_r.detach())',
           '[_u0p.detach()]', '[_r0p.detach()]'):
    assert s.count(_t) == 1, f"undetached diagnostic tap: {_t}"
# -- the normalization is where-gated, and no dtype literal enters the JIT
assert '_sp = _ss > 0\n    _r = _sp.where(_sp.where(_ss, 1.0).sqrt(), 0.0)' in s \
    and '_rd = _sp.where(_r, 1.0)' in s and '(_xg / _rd)' in s, \
    ("the polar normalization must be where-gated TWICE (CLAUDE.md S5): "
     "the sqrt (no NaN) and the denominator (no 1/eps backward spike)")
assert 'POLAR_EPS' not in s, \
    "an epsilon denominator is the refuted cell here — use the where-gate"
assert 'dtypes.float32' not in s, "no float32 dtype literal anywhere"
# -- env-inertness: every new runtime line is under an ALG_POLAR guard
for _line in ('_pol_r, _pol_u = _polar_ru(cur, _pg)',
              'cur = _polar_ru_join(_pol_u, _pol_r, _pg)',
              'out["breaths_u"] = [_fed_core(_x9) for _x9 in',
              '_rst4 = _pol_r.mean(-2).detach()'):
    _blk = s[:s.index(_line)]
    assert 'ALG_POLAR' in _blk[-2600:], \
        f"unguarded polar line (no ALG_POLAR gate above): {_line}"

# ===========================================================================
# THE SYMTABLE FREE-VARIABLE AUDIT (the apply_mask_head.py idiom)
# ===========================================================================
mod_tbl = symtable.symtable(s, fn, 'exec')
module_names = set(mod_tbl.get_identifiers())
DYNAMIC_OK = {'_CENSUS', '_IMP', '_SEV', '_SGC', '_BINDC', '_PCV'}
BUILTIN = set(dir(builtins))


def audit(tbl, fname):
    bad = set()
    for sym in tbl.get_symbols():
        n_ = sym.get_name()
        if sym.is_global() and n_ not in module_names \
                and n_ not in DYNAMIC_OK and n_ not in BUILTIN:
            bad.add(n_)
    for ch in tbl.get_children():
        bad |= audit(ch, fname)
    assert not bad, f"{fname}: unresolved free variables {sorted(bad)}"
    return set()


AUDITED = ('breath_step', 'do_train', 'forward', 'build_params',
           '_polar_groups', '_polar_tables', '_polar_ru', '_polar_ru_join')
for child in mod_tbl.get_children():
    if child.get_name() in AUDITED:
        audit(child, child.get_name())

# ===========================================================================
# REPORT
# ===========================================================================
print(f"[polar waist] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
_n0 = len(_bj['wheels'][0]['planes'])
_n1 = len(_bj['wheels'][1]['planes'])
_n2 = len(_bj['wheels'][2]['planes'])
print(f"[polar waist] bands {BANDS}: {_bj['n_planes']} planes -> "
      f"breath-hand {_n0} / parity {_n1} / pass {_n2} "
      f"(clocked {_bj['n_clocked']}, content {_bj['n_content']}); "
      f"disjoint, frozen at birth, READ by the head")
print("[polar waist] symtable free-var audit PASS "
      f"({', '.join(AUDITED)})")
print("[polar waist] NEW params: 0 — frozen frequencies, no gains, no "
      "learnable schedule (a parametrization and a clock, not capacity)")
print("[polar waist] contract: ALG_POLAR unset = byte-identical (eq A/B/C); "
      "set = a NEW GENERATION measured as a twin, never asserted equivalent")
if CHECK:
    print("[polar waist] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[polar waist] APPLIED ({fn}); ast OK — run the eq pre/post "
          "A/B/C gate (ALG_POLAR unset) + .cache/pc_row_smoke.py + "
          "scripts/polar_birth_smoke.py before trusting")
