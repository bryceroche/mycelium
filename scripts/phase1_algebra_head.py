"""phase1_algebra_head.py — the ALGEBRA delta head: the §11 layout, built.

THE OBJECT: parse algebra-in-words into arith3 graphs — the first head whose output
feeds a domain where the symbolic jaw has GENUINE deciding to do (per-sample band
labels 0-3 decisions). Layout per the registered §11 frame:

  TWO SLOT BANKS over the shared waist (frozen Llama L0-L3 trunk, T=256):
    VARIABLE slots (24): identity = positional (slot i <-> letter i); text anchoring
      supervised by the generator's MENTION spans (name->slot binding AS STRUCTURE,
      the §6 law applied prospectively — the text-NACK lesson).
    FACTOR slots (24): span-supervised like the KenKen head.
  POINTERS are BILINEAR (factor-state x variable-state): args = 2-hot BCE over var
  slots; result = CE over var slots. Directly supervised from gold — the attention-
  bootstrap law's sanctioned escape, third deployment.
  RESULT = UNION TYPE: is-literal mode bit; rel factors -> result POINTER; given
  factors -> var pointer + value DIGITS (3 x 10-way, MSD-first, transplanted).
  QUERY POINTER: one global head over variable slots (solving is not answering).

EVAL IS PER-BAND from run one (registered): factor-exact / graph-solve / ANSWER
rate logged per decisions-band — the parse-vs-solve factorization question answers
itself. ANSWER rate = the honest end metric: decode -> problem_from_algebra ->
solve_symbolic -> solution[predicted query] == gold answer.

USAGE:
  Selftest:    .venv/bin/python3 scripts/phase1_algebra_head.py --selftest
  Precompute:  DEV=PCI+AMD .venv/bin/python3 scripts/phase1_algebra_head.py --precompute
  Train:       DEV=PCI+AMD STEPS=8000 .venv/bin/python3 scripts/phase1_algebra_head.py --train
  Eval:        DEV=PCI+AMD .venv/bin/python3 scripts/phase1_algebra_head.py --eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

T_ALG = 256
ALG_REF = int(os.environ.get("ALG_REF", "0"))   # E-FLOOR referent supervision
ALG_DIAL = int(os.environ.get("ALG_DIAL", "0"))  # door #45: the dialect reader
ALG_VALATT = int(os.environ.get("ALG_VALATT", "0"))  # door #61: given-binding aid
ALG_SIXWAVE = int(os.environ.get("ALG_SIXWAVE", "0"))  # door #62: six-wave slot phasing
ALG_LSENT = int(os.environ.get("ALG_LSENT", "0"))    # V2: letter-keyed partition input
ALG_SYNC = int(os.environ.get("ALG_SYNC", "0"))      # sync-complete: one clock, both sides, ticking
ALG_CONSUME = int(os.environ.get("ALG_CONSUME", "0"))  # consume-once credit (any breath, once)
ALG_NOTEBOOK = int(os.environ.get("ALG_NOTEBOOK", "0"))  # the cathedral notebook
ALG_CIRCLE = int(os.environ.get("ALG_CIRCLE", "0"))      # the traffic circle
ALG_STELLAR = int(os.environ.get("ALG_STELLAR", "0"))    # cell-3b: helical handoff
NB_PERSLOT = int(os.environ.get("NB_PERSLOT", "0"))      # per-slot lanes: sharp ink
ALG_SEPHASE_Q = int(os.environ.get("ALG_SEPHASE_Q", "0"))   # identity channel
SEPHASE_Q_SCRAMBLE = int(os.environ.get("SEPHASE_Q_SCRAMBLE", "0"))
NB_FOCAL = float(os.environ.get("NB_FOCAL", "0"))        # magnifying glass: read sharpening
SEPHASE_B_BAND = int(os.environ.get("SEPHASE_B_BAND", "0"))  # banded-overlap stamps
ALG_SEPHASE_W = int(os.environ.get("ALG_SEPHASE_W", "0"))   # waist channel
ALG_SEPHASE_PAIR = int(os.environ.get("ALG_SEPHASE_PAIR", "0"))  # transceiver
ALG_SEPHASE_SETTLE = int(os.environ.get("ALG_SEPHASE_SETTLE", "0"))  # settle transceiver


def phase_alphabet(n, dims, scale, rng, noise=0.2):
    """THE SHARED SIX-PHASE ALPHABET (2026-08-19): one coordinate code for
    every seeded channel — row k rides phase (k mod 6)*pi/3 at frequency
    family k+1, native scale, brownian seasoning (the receipt's recipe)."""
    _k = np.arange(n)
    _d = np.arange(dims)
    _ph = (_k % 6) * (np.pi / 3.0)
    _pat = scale * np.sqrt(2.0) * np.cos(
        _ph[:, None] + _d[None, :] * (2 * np.pi / dims) * (_k[:, None] + 1))
    return (_pat + rng.randn(n, dims) * scale * noise).astype(np.float32)
NB_H = int(os.environ.get("NB_H", "4"))                  # calibrated horizon
H_TRUNK = 2048
H_W = int(os.environ.get("ALG_HW", "512"))  # capacity-probe dial (2026-07-11)
K_VARS = 24
L_FAC = 24
# ANSWER_SPACE_SPEC E1+E3 (2026-08-03): ALG_WIDE=1 widens the digit bank
# to 7 (values to 1e6, MSD-first as always) and adds the sign head.
ALG_WIDE = int(os.environ.get("ALG_WIDE", "0"))
N_DIG = 7 if ALG_WIDE else 3
N_HEADS = 8
MH_HEADS = int(os.environ.get("MH_HEADS", "4"))  # MASK HEAD bank width
                                                 # (4 default; 8 = the
                                                 # registered scale axis)
assert H_W % MH_HEADS == 0, \
    f"MH_HEADS={MH_HEADS} must divide H_W={H_W} (head reshape)"
MH_CTX_F = 22   # mask-context features: 12 fact (arg1/arg2/res x 4) +
                # 3 domain-mass port + 1 given-flag + 2 adjacency
                # row/col mass + 2 prev-breath row/col + 2 breath phase
# ===========================================================================
# THE FED MIND (apply_fed_mind.py, 2026-09-05, word given): ONE env
# family. ALG_FED=1 turns all items on; ALG_FED_<ITEM>=0 ablates one
# (MIXER/POINTERS/WAIST/FFN/MACRO/SCRATCH/ROTOR/NL0/SHELF). Family
# unset = byte-identical (every new tensor and line is guarded).
# ===========================================================================
ALG_FED = int(os.environ.get("ALG_FED", "0"))


def _fed_sub(_n):
    return bool(ALG_FED and int(os.environ.get("ALG_FED_" + _n, "1")))


FED_MIXER = _fed_sub("MIXER")
FED_POINTERS = _fed_sub("POINTERS")
FED_WAIST = _fed_sub("WAIST")
FED_FFN = _fed_sub("FFN")
FED_MACRO = _fed_sub("MACRO")
FED_SCRATCH = _fed_sub("SCRATCH")
FED_ROTOR = _fed_sub("ROTOR")
FED_NL0 = _fed_sub("NL0")
FED_SHELF = _fed_sub("SHELF")
MX_HEADS = int(os.environ.get("MX_HEADS", "8"))   # fed mixer head count
assert H_W % MX_HEADS == 0, \
    f"MX_HEADS={MX_HEADS} must divide H_W={H_W} (head reshape)"
PF_FORMS = int(os.environ.get("PF_FORMS", "3"))   # pointer/macro forms
N_SCR = 8 if FED_SCRATCH else 0                   # scratch slot rows
L_TOT = L_FAC + N_SCR                             # bank rows incl. scratch
NB_ROWS = 16 if FED_SHELF else 8                  # shelf stamp rows (item 9)
if FED_SCRATCH:
    # read-back for scratch rows lives ONLY behind the fed mixer's
    # zero-init door (the raising law: no cold births — fully-open
    # columns would inject 8 cold states into a converged circuit)
    assert FED_MIXER, \
        "ALG_FED_SCRATCH requires ALG_FED_MIXER (scratch read-back door)"
    assert not int(os.environ.get("ALG_RINGS", "0")), \
        "scratch + RINGS unsupported (the pawl grades slots; loud door)"
# FED item 7: THE BREATH ROTOR — mycelium/rotor_clock.py gets its FIRST
# importer here (the clock-audit debt: sync enforced by the import
# graph, finally true). Top-level and unconditional: numpy-only module,
# no circularity, negligible cost. FREQUENCIES FROZEN, GAINS LEARNABLE
# (the gains are fed_mx_hg) — zero parameters live here.
from mycelium.rotor_clock import breath_qk_angles as _rc_breath_qk_angles
_FED_ROT_C = _FED_ROT_S = None
if FED_ROTOR:
    assert (H_W // MX_HEADS) % 2 == 0 and (H_W // MX_HEADS) // 2 >= 32, \
        "fed rotor band table needs >=32 pairs per head (64-d heads)"
    _fed_ang = _rc_breath_qk_angles()      # (6, 8): legacy band 24..31
    _FED_ROT_C = np.ones((_fed_ang.shape[0], (H_W // MX_HEADS) // 2),
                         np.float32)
    _FED_ROT_S = np.zeros_like(_FED_ROT_C)
    _FED_ROT_C[:, 24:32] = np.cos(_fed_ang).astype(np.float32)
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
POLAR_QROT = int(os.environ.get("ALG_POLAR_QROT", "2"))       # 0/1/2: off,
                                                # mixer only, main+mixer
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
    assert POLAR_R_MODE == "slotvec", \
        f"ALG_POLAR_R_MODE={POLAR_R_MODE} (scalar|slotvec)"
    assert H_W % POLAR_RG == 0 and (H_W // POLAR_RG) % 2 == 0, \
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
        assert _P == MX_HEADS * ((H_W // MX_HEADS) // 2), \
            "plane count must reshape to (MX_HEADS, pairs/head) for the Q side"
        with open(POLAR_BANDS) as _pf:
            _pb = _pjs.load(_pf)
        assert int(_pb["waist"]) == H_W and int(_pb["n_planes"]) == _P, \
            f"{POLAR_BANDS}: waist/plane mismatch (head H_W={H_W})"
        assert len(_pb["wheels"]) == _RC_N_WHEELS, \
            f"{POLAR_BANDS}: {len(_pb['wheels'])} wheels, clock has {_RC_N_WHEELS}"
        _wof = np.full(_P, -1, np.int64)
        for _wi, _wd in enumerate(_pb["wheels"]):
            assert int(_wd["wheel"]) == _wi, "wheels out of order"
            for _pl in _wd["planes"]:
                _pl = int(_pl)
                assert 0 <= _pl < _P, f"plane {_pl} out of range"
                assert _wof[_pl] < 0, \
                    f"plane {_pl} claimed twice — SEPARATE BANDS is a law"
                _wof[_pl] = _wi
        _abs = _rc_wheel_table()                      # (N_LOOP, N_WHEELS)
        _dlt = np.zeros_like(_abs)
        _dlt[0] = _abs[0]
        _dlt[1:] = _abs[1:] - _abs[:-1]
        assert np.allclose(np.cos(np.cumsum(_dlt, 0)), np.cos(_abs), atol=1e-6) \
            and np.allclose(np.sin(np.cumsum(_dlt, 0)), np.sin(_abs), atol=1e-6), \
            "state increments do not accumulate to rotor_clock's absolute phase"
        assert np.allclose(np.cos(_dlt[1:, 0]), math.cos(_RC_QUANTUM)) \
            and np.allclose(np.sin(_dlt[1:, 0]), math.sin(_RC_QUANTUM)), \
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


# ===========================================================================
# THE KITCHEN SINK (apply_polar_sink.py, 2026-09-08) — two organs on the
# polar DIRECTION u, each behind its own door, each byte-inert when unset.
#   ALG_POLAR_D  the CONTENT-PLANE WAIST: collapse & expand the 192
#                content planes' 384 dims through a d-wide bottleneck,
#                once per breath ("expand & collapse x7"). The 64 CLOCK
#                planes are in neither its domain nor its range, so the
#                bottleneck CANNOT fight the rotation.
#   ALG_POLAR_EM the E&B COUPLING: one discrete Maxwell-like exchange
#                step per breath on the CLOCK planes, along the slot-mask
#                lanes (the mask actually in force this breath). kappa is
#                FIXED and unlearnable — the zero-born-gain pattern died
#                in fed_mx_hg (0.023 at best, 0.013 under the cooker).
# Both are norm-changing and both restore the norm OF THEIR OWN BLOCK
# through the one where-gated organ below, so ||u|| == 1 still holds and
# neither door can rescale the other's channel (a single GLOBAL renorm
# would let the collapsing content block amplify the clock every breath
# — the bottleneck fighting the rotation by the back door).
# ===========================================================================
POLAR_D = int(os.environ.get("ALG_POLAR_D", "0"))       # content-waist width
POLAR_D_INIT = os.environ.get("ALG_POLAR_D_INIT", "")   # PCA-birth override
POLAR_EM = float(os.environ.get("ALG_POLAR_EM", "0"))   # E&B kappa (FIXED)
assert ALG_POLAR or not (POLAR_D or POLAR_EM), \
    ("ALG_POLAR_D / ALG_POLAR_EM ride on the polar direction u — they "
     "require ALG_POLAR=1 (refusing a door that would do nothing)")
assert POLAR_D >= 0 and POLAR_EM == POLAR_EM, "bad ALG_POLAR_D/ALG_POLAR_EM"
_POLAR_SINK = None          # (cdim, sel, gate_content, gate_clock, gate_pl)
_POLAR_SINK_SHOWN = False


def _polar_sink():
    """The dim tables and constant tensors BOTH sink organs share, built
    ONCE from the `wheel_of` map _polar_tables owns — the SAME band
    allocation the sextet turns. No second band table exists (the meter-
    divergence law: a check must call its organ). Returns
      cdim   (C,)      the CONTENT dims; plane p owns dims 2p, 2p+1
      sel    (H_W, C)  the one-hot gather/scatter constant: x @ sel is
                       the content block, sel @ W scatters a content-
                       shaped matrix back into waist coordinates with
                       EXACT zeros on every clock dim
      g_c    (H_W,)    1.0 on content dims, 0.0 on clock dims
      g_k    (H_W,)    its complement (1.0 on clock dims)
      g_p    (P,)      1.0 on CLOCKED planes (the EM step's gate)
    Built lazily and cached module-side (the _SGC pattern): the tensors
    are constants, so the JIT sees the same buffers every call."""
    global _POLAR_SINK
    if _POLAR_SINK is None:
        from tinygrad import Tensor as _Tk, dtypes as _dk
        _wof = _polar_tables()[4]                    # (P,) -1 = content
        _P = H_W // 2
        _cp = np.flatnonzero(_wof < 0).astype(np.int64)
        assert 0 < len(_cp) < _P, \
            "the sink needs BOTH a content band and a clock band"
        _cd = np.empty(2 * len(_cp), np.int64)
        _cd[0::2] = 2 * _cp
        _cd[1::2] = 2 * _cp + 1
        _sl = np.zeros((H_W, len(_cd)), np.float32)
        _sl[_cd, np.arange(len(_cd))] = 1.0
        _gc = np.zeros(H_W, np.float32)
        _gc[_cd] = 1.0
        _POLAR_SINK = (_cd, _Tk(_sl, dtype=_dk.float),
                       _Tk(_gc, dtype=_dk.float),
                       _Tk(1.0 - _gc, dtype=_dk.float),
                       _Tk((_wof >= 0).astype(np.float32), dtype=_dk.float))
    return _POLAR_SINK


def _polar_keepnorm(u_new, u_old, gate):
    """Restore the norm of the GATE'd block of u_new to the norm u_old's
    same block carried, and leave every dim OUTSIDE the gate multiplied by
    EXACTLY 1.0 (bitwise passthrough — this is what makes each door's
    bit-identity proof possible). ||u|| == 1 survives because each organ
    restores its own block and the blocks partition the waist.
    WHERE-GATED TWICE, the _polar_ru idiom: the sqrt (no grad/(2*sqrt 0))
    and the denominator (replaced by 1.0, never by an epsilon — an eps
    guard keeps the forward finite and hands the backward a 1/eps spike
    at precisely the state the guard exists for)."""
    _s0 = (u_old * u_old * gate).sum(-1, keepdim=True)
    _s1 = (u_new * u_new * gate).sum(-1, keepdim=True)
    _p0 = _s0 > 0
    _p1 = _s1 > 0
    _n0 = _p0.where(_p0.where(_s0, 1.0).sqrt(), 0.0)     # true block norm
    _n1 = _p1.where(_p1.where(_s1, 1.0).sqrt(), 1.0)     # guarded denom
    return u_new * (1.0 + gate * (_n0 / _n1 - 1.0))


def _polar_waist(u, p, state):
    """ALG_POLAR_D — THE CONTENT-PLANE WAIST. c' = c @ W_down @ W_up on
    the 384 content dims; the 128 clock dims are in neither the domain
    nor the range. The two (C, d) / (d, C) parameters are scattered into
    waist coordinates ONCE PER FORWARD (cached in `state`, which is a
    fresh dict per forward and shared across the breaths) so the seven
    breaths pay one scatter, not seven; the scatter constant carries
    EXACT zeros, so the clock columns of the effective W_up are exactly
    0.0 and the clock dims of c' are exactly 0.0 -> u * g_k + c' leaves
    every clock dim BITWISE untouched. Then the content block's norm is
    restored, so ||u|| == 1 still holds and the clock keeps its share."""
    _cd, _sel, _gc, _gk, _gp = _polar_sink()
    _wd = state.get("polar_wd_eff")
    if _wd is None:
        _wd = _sel @ p["polar_wd"]                       # (H_W, d)
        state["polar_wd_eff"] = _wd
        state["polar_wu_eff"] = p["polar_wu"] @ _sel.transpose(-2, -1)
    _wu = state["polar_wu_eff"]                          # (d, H_W)
    _cn = (u @ _wd) @ _wu                # exact zeros on every clock dim
    return _polar_keepnorm(u * _gk + _cn, u, _gc)


def _polar_em(u, msk, kappa):
    """ALG_POLAR_EM — THE E&B COUPLING on the clock planes, along the
    slot-mask lanes. With (x, y) the coordinates of a clocked plane and
    Mhat the ROW-NORMALIZED mask actually in force this breath (so kappa
    is dimensionless):
        x_i += kappa * sum_j Mhat_ij (y_j - y_i)
        y_i -= kappa * sum_j Mhat_ij (x_j - x_i)
    i.e. z <- (I - i*kappa*L)z with L the row-normalized graph Laplacian:
    the two field components exchange along the OPEN lanes. kappa is a
    python float baked into the graph — FIXED, declared once, never a
    parameter (no zero-born gain: fed_mx_hg's grave).
    CONTENT PLANES: their delta is multiplied by an exact 0.0, so they
    pass through bitwise; the clock block's norm is then restored (the
    update is norm-CHANGING: |1 - i*kappa*lambda| > 1)."""
    from tinygrad import Tensor as _Te
    _cd, _sel, _gc, _gk, _gp = _polar_sink()
    _B, _L, _W = u.shape[0], u.shape[1], u.shape[2]
    _uv = u.reshape(_B, _L, _W // 2, 2)
    _x = _uv[..., 0]
    _y = _uv[..., 1]                                     # (B, L, P)
    _dg = msk.sum(-1, keepdim=True)                      # (B, L, 1) degree
    _pg = _dg > 0
    _dn = _pg.where(_dg, 1.0)              # the guarded denominator (1.0)
    _lx = (msk @ _x) / _dn - _x            # sum_j Mhat_ij (x_j - x_i)
    _ly = (msk @ _y) / _dn - _y
    _kg = _gp.reshape(1, 1, -1) * kappa    # CLOCKED planes only
    _un = _Te.stack(_x + _kg * _ly, _y - _kg * _lx, dim=-1).reshape(_B, _L, _W)
    return _polar_keepnorm(_un, u, _gk)
SENT_MAX = 32


# ===========================================================================
# THE TERMINAL REGISTRY — the buffer-spec door, v1 (2026-08-04; earned by
# five dens in one day). ONE authority describing every trainable terminal:
# its params, its emission key, its gold keys, and the env gate that brings
# it into existence. v1 is the ASSERT arm of derive-or-assert (#128): the
# wiring below stays hand-written (the deployed gate's code is not
# refactored the week it promoted); every trainer ASSERTS its buffer set
# and every build can assert its emission set against THIS table. The
# DERIVE arm (generating forward/loss/buffers from the table) waits for a
# proper refactor window. A sixth terminal added without updating this
# table fails LOUDLY at build, not at the optimizer.
# ===========================================================================
def _ft(): return int(os.environ.get("ALG_FTYPES", "4"))
TERMINALS = {
    "pres":   {"params": ["h_pres", "h_pres_b"],   "emit": "pres",  "gold": ["presence"], "when": lambda: True},
    "ftype":  {"params": ["h_ftype", "h_ftype_b"], "emit": "ftype", "gold": ["ftype"],    "when": lambda: True},
    "op":     {"params": ["h_op", "h_op_b"],       "emit": "op",    "gold": ["op"],       "when": lambda: True},
    "islit":  {"params": ["h_islit", "h_islit_b"], "emit": "islit", "gold": ["is_lit_f"], "when": lambda: True},
    "dig":    {"params": ["h_dig", "h_dig_b"],     "emit": "dig",   "gold": ["digits"],   "when": lambda: True},
    "args":   {"params": ["W_args"],               "emit": "args",  "gold": ["args"],     "when": lambda: True},
    "dargs":  {"params": ["W_dargs"],              "emit": "dargs", "gold": ["args", "arg_dup"],
               "when": lambda: int(os.environ.get("ALG_DUPPTR", "0")) > 0},
    "res":    {"params": ["W_res"],                "emit": "res",   "gold": ["res"],      "when": lambda: True},
    "query":  {"params": ["W_query"],              "emit": "query", "gold": ["query"],    "when": lambda: True},
    "sel":    {"params": ["h_sel", "h_sel_b"],     "emit": "sel",   "gold": ["sel"],      "when": lambda: _ft() >= 5},
    "dup":    {"params": ["h_dup", "h_dup_b"],     "emit": "dup",   "gold": ["arg_dup"],  "when": lambda: int(os.environ.get("ALG_DUP", "0")) > 0},
    "dig2":   {"params": ["h_dig2", "h_dig2_b"],   "emit": "dig2",  "gold": ["digits2", "is_macro"], "when": lambda: int(os.environ.get("ALG2", "0")) and _ft() >= 7},
    "y":      {"params": ["W_y"],                  "emit": "y",     "gold": ["y"],        "when": lambda: int(os.environ.get("ALG2", "0")) and _ft() >= 7},
    "sgn":    {"params": ["h_sgn", "h_sgn_b"],     "emit": "sgn",   "gold": ["sign"],     "when": lambda: ALG_WIDE},
    "depth":  {"params": ["h_depth", "h_depth_b"], "emit": "depth", "gold": ["depth"],
               "when": lambda: int(os.environ.get("ALG_POSCH", "0")) > 0},
    "term":   {"params": ["h_term", "h_term_b"],   "emit": "term",  "gold": ["term"],
               "when": lambda: int(os.environ.get("ALG_POSCH", "0")) > 0},
    "opc":    {"params": ["W_opc1", "W_opc1_b", "W_opc2", "W_opc2_b"],
               "emit": "opc", "gold": ["opc"],
               "when": lambda: int(os.environ.get("ALG_OPCOUNT", "0")) > 0},
    "router": {"params": ["W_rs", "W_ra", "W_rb", "r_gain"],
               "emit": "rbias", "gold": ["fspan"],
               "when": lambda: int(os.environ.get("ALG_ROUTER", "0")) > 0},
    "bindbus": {"params": ["W_bind1", "W_bind1_b", "W_bind2"],
                # tabula-rasa cleanup 2026-08-31: monolith only (the
                # sharing monotone's winner); v1/v5/v6 in git history
                "emit": "bind", "gold": (["bind_ids"] if int(os.environ.get("ALG_BINDBUS", "0")) >= 5
                                         else ["bindvec", "bind_ids"] if int(os.environ.get("ALG_BINDBUS", "0")) >= 3 else ["bindvec"]),
                "when": lambda: int(os.environ.get("ALG_BINDBUS", "0")) > 0},
    "cmt":    {"params": ["W_cmt", "W_cmt_b"],     "emit": "cmt",   "gold": ["ftype", "res"],
               "when": lambda: int(os.environ.get("ALG_RINGS", "0")) and int(os.environ.get("ALG_BREATH", "1")) > 1},
    "cmtreg": {"params": ["w_cmt_reg"],            "emit": "cmt",   "gold": ["ftype", "res"],
               "when": lambda: int(os.environ.get("ALG_RINGS", "0")) and int(os.environ.get("ALG_CMT_REG", "0"))
               and int(os.environ.get("ALG_BREATH", "1")) > 1},
}


def assert_terminals(p=None, emitted=None, gold_keys=None, site="build"):
    """THE DOOR'S ASSERT: every active terminal has (a) its params built,
    (b) its emission present (when `emitted` given), (c) its gold buffers
    present (when `gold_keys` given). Raises with the two-terminal law's
    own words; a missing entry fails at BUILD, never at the optimizer."""
    for name, t in TERMINALS.items():
        if not t["when"]():
            continue
        if p is not None:
            missing = [k for k in t["params"] if k not in p]
            assert not missing, (
                f"TERMINAL '{name}' params missing at {site}: {missing} "
                f"(the two-terminal law — the registry says this terminal "
                f"exists under the current env)")
        if emitted is not None:
            assert t["emit"] in emitted, (
                f"TERMINAL '{name}' NOT EMITTED at {site} — a forward "
                f"variant left it out (the fifth-den shape); grad will be "
                f"None at the optimizer unless every emission ships")
        if gold_keys is not None:
            missing = [g for g in t["gold"] if g not in gold_keys]
            assert not missing, (
                f"TERMINAL '{name}' gold buffers missing at {site}: "
                f"{missing} (the fourth-den shape — without these the "
                f"terminal leaves the graph: None grads)")


ALG_TRAIN = os.environ.get("ALG_TRAIN", ".cache/algebra_nl_train.jsonl")
ALG_TEST = os.environ.get("ALG_TEST", ".cache/algebra_nl_test.jsonl")
TEST_NAME = os.environ.get("ALG_TEST_NAME", "test")   # states-file key for the test slice
TRAIN_NAME = os.environ.get("ALG_TRAIN_NAME", "train")  # states-file key for the train slice
STATES_NPZ = ".cache/phase1_alg_states_{split}.npz"
STATES_NPY = ".cache/phase1_alg_states_{split}_states.npy"  # memmap sibling (gen-7+)
ALG_CKPT = os.environ.get("ALG_CKPT", ".cache/phase1_algebra_head.safetensors")
TOKENIZER_JSON = ".cache/llama-3.2-1b-weights/tokenizer.json"


# ===========================================================================
# GOLD (CPU): jsonl + offsets -> slot tensors per the §11 layout
# ===========================================================================

def _spans_to_tokmask(spans, offs, out):
    for (cs, ce) in spans:
        for ti, (ts, te) in enumerate(offs):
            if ti >= T_ALG:
                break
            if ts < ce and te > cs:
                out[ti] = 1.0


def _pos_meta(r):
    """per-factor (graph depth 0-5, terminality) — the position channel's
    gold (2026-08-24; labels derived from the factor list, key-lawful)."""
    facs = r["factors"]; q = r.get("query_var", r.get("query", 0))
    def var_of(f): return f.get("result", f.get("var", 0))
    def dep(fi, seen):
        f = facs[fi]
        if f["ftype"] == "given": return 0
        srcs = f.get("args", []) if "args" in f else [f.get("var", 0)]
        ds = []
        for v in srcs:
            for fj, gg in enumerate(facs):
                if fj != fi and var_of(gg) == v and fj not in seen:
                    ds.append(dep(fj, seen | {fj}))
        return 1 + max(ds, default=0)
    return [(min(dep(fi, {fi}), 5), 1.0 if var_of(f) == q else 0.0)
            for fi, f in enumerate(facs)]


OPC_CLASSES = ["add", "sub", "mul", "div", "sq", "opa", "fr",
               "given", "mod", "sel", "pct", "fdiv"]
OPC_CAP = 7


def _opc_meta(r):
    """per-row op-class counts (lever 3, 2026-08-25) — the SAME op-grain
    mapping as chain_decode's grader (rel->op / dup-mul->sq / macro->opa|fr
    / frac->fr / else ftype); gold at the ROW grain, where exactness lives."""
    from collections import Counter
    cnt = Counter()
    for f in r["factors"]:
        if f["ftype"] == "rel":
            if f.get("op") == "mul" and len(set(f.get("args", []))) == 1:
                cnt["sq"] += 1
            else:
                cnt[f.get("op", "add")] += 1
        elif f["ftype"] == "macro":
            cnt["opa" if f.get("name") == "OP_APPLY" else "fr"] += 1
        elif f["ftype"] == "frac":
            cnt["fr"] += 1
        else:
            cnt[f["ftype"]] += 1
    return [min(cnt.get(c, 0), OPC_CAP) for c in OPC_CLASSES]


def build_gold(samples, offsets):
    n = len(samples)
    g = {
        "presence": np.zeros((n, L_FAC), np.float32),
        "ftype": np.zeros((n, L_FAC), np.int32),         # 0=rel 1=given 2=mod 3=sel
        "op": np.zeros((n, L_FAC), np.int32),            # 0=add 1=mul
        "args": np.zeros((n, L_FAC, K_VARS), np.float32),
        "res": np.zeros((n, L_FAC), np.int32),
        "is_lit": np.zeros((n, L_FAC), np.float32),
        "digits": np.zeros((n, L_FAC, N_DIG), np.int32),
        "sign": np.zeros((n, L_FAC), np.float32),        # E1: 1.0 = negative literal
        "fspan": np.zeros((n, L_FAC, T_ALG), np.float32),
        "vspan": np.zeros((n, K_VARS, T_ALG), np.float32),
        **({"refvar": np.full((n, T_ALG), -1, np.int8)} if ALG_REF else {}),
        **({"is_ind": np.zeros((n, L_FAC), np.float32)} if ALG_DIAL else {}),
        "query": np.zeros((n,), np.int32),
        "band": np.zeros((n,), np.int32),
        # the tranche (2026-07-09): explicit per-kind masks (rel mask was
        # (1-is_lit) — wrong once mod/sel exist) + selector-type gold
        "sel": np.zeros((n, L_FAC), np.int32),           # SEL_TO_ID
        "is_rel": np.zeros((n, L_FAC), np.float32),
        "is_mod": np.zeros((n, L_FAC), np.float32),
        "is_sel": np.zeros((n, L_FAC), np.float32),
        # tranche 2 (2026-07-10): pct=4, fdiv=5 — the mod head-shape
        "is_pct": np.zeros((n, L_FAC), np.float32),
        "is_fdiv": np.zeros((n, L_FAC), np.float32),
        # gen-9 (2026-07-12): arg multiplicity — args=[a,a] was
        # UNREPRESENTABLE (multi-hot gold + top-2-distinct decode); the
        # [85] fix. 1.0 on rel factors whose two args are the same var.
        "arg_dup": np.zeros((n, L_FAC), np.float32),
        # gen-15 (2026-07-20): OP_APPLY macro floor (ALG_FTYPES=7) — second
        # digit bank (k2) + ordered second operand pointer (y). Structural
        # entry per the pointer law; gold-fed from birth (two-terminal law).
        "digits2": np.zeros((n, L_FAC, N_DIG), np.int32),
        "is_macro": np.zeros((n, L_FAC), np.float32),
        "is_frac": np.zeros((n, L_FAC), np.float32),   # mg2: FRAC_OF (ftype 7)
        "is_chain": np.zeros((n, L_FAC), np.float32),  # mg3: CHAIN_MUL (ftype 8)
        **({"valspan": np.zeros((n, L_FAC, T_ALG), np.float32)} if ALG_VALATT else {}),
        **({"lsent": np.zeros((n, K_VARS, T_ALG), np.float32)} if ALG_LSENT else {}),
        "y": np.zeros((n, L_FAC), np.int32),
    }
    for i, (smp, offs) in enumerate(zip(samples, offsets)):
        # gen-10 prose rows: factors may carry NO spans (raw prose has no
        # letter anchors) — sort them last; span losses auto-mask on zeros
        facs = sorted(smp["factors"],
                      key=lambda f: (min(s for s, _ in f["spans"])
                                     if f.get("spans") else 10 ** 9))
        assert len(facs) <= L_FAC and smp["n_vars"] <= K_VARS
        g["query"][i] = smp["query_var"]
        g["band"][i] = smp["decisions"]
        _indvars = set()
        for v_str, spans in smp["mentions"].items():
            _spans_to_tokmask(spans, offs, g["vspan"][i, int(v_str)])
            if any(sp[1] - sp[0] > 2 for sp in spans):
                _indvars.add(int(v_str))
            if ALG_REF:   # E-FLOOR (door #43): indirect sites only (char>2)
                _ind = [sp for sp in spans if sp[1] - sp[0] > 2]
                if _ind:
                    _m = np.zeros(T_ALG, np.float32)
                    _spans_to_tokmask(_ind, offs, _m)
                    g["refvar"][i][_m > 0] = int(v_str)
        if ALG_LSENT:
            import re as _re2
            _parts=[(m.start(),m.end()) for m in _re2.finditer(r"[^.]+\.", smp["text"])]
            for _v in range(min(smp.get("n_vars",K_VARS),K_VARS)):
                _nm=chr(ord("a")+_v)
                for _a,_b in _parts:
                    if _re2.search(r"\b"+_nm+r"\b", smp["text"][_a:_b]):
                        _spans_to_tokmask([(_a,_b)], offs, g["lsent"][i, _v])
        for j, f in enumerate(facs):
            g["presence"][i, j] = 1.0
            if ALG_DIAL and f["ftype"] == "rel" and any(a in _indvars for a in f.get("args", [])):
                g["is_ind"][i, j] = 1.0
            _spans_to_tokmask(f.get("spans") or [], offs, g["fspan"][i, j])
            if f["ftype"] == "rel":
                g["ftype"][i, j] = 0
                g["is_rel"][i, j] = 1.0
                g["op"][i, j] = 0 if f["op"] == "add" else 1
                for a in f["args"]:
                    g["args"][i, j, a] = 1.0
                if f["args"][0] == f["args"][1]:
                    g["arg_dup"][i, j] = 1.0
                g["res"][i, j] = f["result"]
            elif f["ftype"] == "given":
                g["ftype"][i, j] = 1
                g["is_lit"][i, j] = 1.0
                if ALG_VALATT and f.get("spans"):
                    _va, _vb = f["spans"][0]
                    import re as _re
                    for _vmm in _re.finditer(r"\d+", smp["text"][_va:_vb]):
                        if int(_vmm.group()) == int(f["value"]):
                            _spans_to_tokmask([(_va + _vmm.start(),
                                                _va + _vmm.end())],
                                              offs, g["valspan"][i, j])
                            break
                g["res"][i, j] = f["var"]
                t = int(f["value"])
                if t < 0:                    # E1: sign gold; digits carry |t|
                    assert ALG_WIDE, "negative literal outside ALG_WIDE"
                    g["sign"][i, j] = 1.0
                    t = -t
                assert t < 10 ** N_DIG
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "mod":
                g["ftype"][i, j] = 2
                g["is_mod"][i, j] = 1.0
                g["args"][i, j, f["var"]] = 1.0
                g["res"][i, j] = f["result"]
                t = int(f["k"])                # modulus via the digit head
                assert t < 10 ** N_DIG
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "sel":
                from mycelium.csp_domains import SEL_TO_ID
                g["ftype"][i, j] = 3
                g["is_sel"][i, j] = 1.0
                g["sel"][i, j] = SEL_TO_ID[f["sel"]]
                for a in f["args"]:
                    g["args"][i, j, a] = 1.0
                g["res"][i, j] = f["result"]
            elif f["ftype"] == "pct":
                g["ftype"][i, j] = 4
                g["is_pct"][i, j] = 1.0
                g["args"][i, j, f["args"][0]] = 1.0
                g["res"][i, j] = f["args"][1]
                t = int(f["p"])
                assert t < 10 ** N_DIG
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "fdiv":
                g["ftype"][i, j] = 5
                g["is_fdiv"][i, j] = 1.0
                g["args"][i, j, f["var"]] = 1.0
                g["res"][i, j] = f["result"]
                t = int(f["k"])
                for d in range(N_DIG):
                    g["digits"][i, j, d] = (t // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "macro" and f.get("name") == "FRAC_OF":
                g["ftype"][i, j] = 7
                g["is_frac"][i, j] = 1.0
                g["args"][i, j, f["x"]] = 1.0
                g["res"][i, j] = f["result"]
                for t_, arr in ((int(f["a"]), "digits"), (int(f["k"]), "digits2")):
                    assert t_ < 10 ** N_DIG
                    for d in range(N_DIG):
                        g[arr][i, j, d] = (t_ // 10 ** (N_DIG - 1 - d)) % 10
            elif f["ftype"] == "macro" and f.get("name") == "CHAIN_MUL":
                g["ftype"][i, j] = 8                     # mg3: the tower pilot
                g["is_chain"][i, j] = 1.0
                for v in f["xs"]:
                    g["args"][i, j, v] = 1.0             # multi-hot xs
                g["res"][i, j] = f["result"]
            elif f["ftype"] == "macro":
                assert f.get("name") == "OP_APPLY"
                g["ftype"][i, j] = 6
                g["is_macro"][i, j] = 1.0
                g["op"][i, j] = 0 if f["op"] == "add" else 1   # add/sub on macro slots
                g["args"][i, j, f["x"]] = 1.0                  # x via args argmax
                g["y"][i, j] = f["y"]                          # y via the fresh pointer
                g["res"][i, j] = f["result"]
                for t_, key, arr in ((int(f["k1"]), "k1", "digits"),
                                     (int(f["k2"]), "k2", "digits2")):
                    assert t_ < 10 ** N_DIG
                    for d in range(N_DIG):
                        g[arr][i, j, d] = (t_ // 10 ** (N_DIG - 1 - d)) % 10
            else:
                raise ValueError(f"unknown gold ftype {f['ftype']!r}")
    if int(os.environ.get("ALG_POSCH", "0")):
        g["depth"] = np.zeros((n, L_FAC), np.int32)
        g["term"] = np.zeros((n, L_FAC), np.float32)
        for ri, smp in enumerate(samples):
            try:
                for j, (d, t) in enumerate(_pos_meta(smp)[:L_FAC]):
                    g["depth"][ri, j] = d; g["term"][ri, j] = t
            except Exception:
                pass
    if int(os.environ.get("ALG_BINDBUS", "0")):
        # THE ROTATIONAL BINDING BUS (2026-08-28, word given): per-slot gold
        # bound vectors — role-phase rotations of var/op codes (VSA in the
        # Fourier domain; codes deterministic from .cache/bindbus_codes.npz)
        _bz = np.load(_bind_codes_path())
        _CB = _bz["CB"]; _P = _CB.shape[1] // 2
        def _rotv(v, th):
            v2 = v.reshape(_P, 2)
            c_, s_ = np.cos(th), np.sin(th)
            return np.stack([c_ * v2[:, 0] - s_ * v2[:, 1],
                             s_ * v2[:, 0] + c_ * v2[:, 1]], -1).reshape(-1)
        _TH = {r: _bz[f"theta_{r}"] for r in ("arg1", "arg2", "res", "op")}
        g["bindvec"] = np.zeros((n, L_FAC, _CB.shape[1]), np.float32)
        g["bind_ids"] = np.zeros((n, L_FAC, 4), np.int32)   # audit 2026-08-30:
        for i2 in range(n):            # the pipeline must produce what the
            for j2 in range(L_FAC):    # fence demands — no out-of-band injection
                if g["presence"][i2, j2] <= 0: continue
                aidx = np.where(g["args"][i2, j2] > 0)[0]
                if len(aidx) == 0: a1 = a2 = int(g["res"][i2, j2])
                elif len(aidx) == 1: a1 = a2 = int(aidx[0])
                else: a1, a2 = int(aidx[0]), int(aidx[1])
                r2 = int(g["res"][i2, j2]); opv = 24 + min(int(g["ftype"][i2, j2]), 7)
                z = (_rotv(_CB[a1], _TH["arg1"]) + _rotv(_CB[a2], _TH["arg2"])
                     + _rotv(_CB[r2], _TH["res"]) + _rotv(_CB[opv], _TH["op"]))
                g["bindvec"][i2, j2] = z
                g["bind_ids"][i2, j2] = (a1, a2, r2, opv)
    if int(os.environ.get("ALG_OPCOUNT", "0")):
        # feed-door completion (2026-08-26): gold is built here, but CACHED
        # npzs from before the surgery lack g_opc — load_alg's consumer
        # must hard-error rather than zero-train (the first opc fire's
        # lesson, now fenced at the missing-gold case too)
        g["opc"] = np.zeros((n, len(OPC_CLASSES)), np.int32)
        for ri, smp in enumerate(samples):
            try:
                g["opc"][ri] = _opc_meta(smp)
            except Exception:
                pass
    return g


def tokenize(path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    samples = [json.loads(l) for l in open(path)]
    ids = np.zeros((len(samples), T_ALG), np.int32)
    mask = np.zeros((len(samples), T_ALG), np.float32)
    offsets = []
    for i, s in enumerate(samples):
        e = tok.encode(s["text"])
        if len(e.ids) > T_ALG:
            raise RuntimeError(f"TRUNCATION {path}:{i} — {len(e.ids)} > {T_ALG}")
        ids[i, :len(e.ids)] = e.ids
        mask[i, :len(e.ids)] = 1.0
        offsets.append(list(e.offsets))
    return samples, ids, mask, offsets


def sent_indices(text, offs, mask_row):
    bounds = []
    i = text.find(". ")
    while i != -1:
        bounds.append(i + 1)
        i = text.find(". ", i + 1)
    out = np.zeros((T_ALG,), np.int32)
    ntk = int(mask_row.sum())
    arr = np.asarray(offs[:min(ntk, T_ALG)], dtype=np.int64)
    if len(arr):
        idx = np.searchsorted(np.asarray(bounds, dtype=np.int64), arr[:, 0], "right")
        out[:len(arr)] = np.minimum(idx, SENT_MAX - 1)
    return out


def tails_of(sent_rows):
    out = np.zeros_like(sent_rows, dtype=np.float32)
    for ri in range(sent_rows.shape[0]):
        s = sent_rows[ri]; i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
            out[ri, i + (j - i + 1) // 2:j + 1] = 1.0
            i = j + 1
    return out


def build_slot_masks(o_np, sent_rows):
    """Evidence-sharing slot mask (B, L, L) from the model's OWN breath-0
    outputs (deployable): same-sentence (attention-argmax sentence) OR
    shared-variable (top-2 args + res argmax overlap) OR self."""
    B = o_np["fat"].shape[0]
    masks = np.zeros((B, L_FAC, L_FAC), np.float32)
    for bi in range(B):
        tok_star = o_np["fat"][bi].argmax(-1)              # (L,)
        s_id = sent_rows[bi][np.minimum(tok_star, T_ALG - 1)]
        same = s_id[:, None] == s_id[None, :]
        M = np.zeros((L_FAC, K_VARS), bool)
        for j in range(L_FAC):
            for a in np.argsort(-o_np["args"][bi, j])[:2]:
                M[j, a] = True
            M[j, int(o_np["res"][bi, j].argmax())] = True
        shared = (M.astype(np.int32) @ M.astype(np.int32).T) > 0
        masks[bi] = (same | shared | np.eye(L_FAC, dtype=bool)).astype(np.float32)
    return masks


def _alt2_fact_buf_v0(onp, se_np, n_vars_arr, m_arr, theta=0.9,
                      mass_out=None):
    """THE PRE-VECTOR REFERENCE (kept for ALG_SEAM_V0=1 fallback A/B;
    scripts/apply_seam_vector.py, 2026-09-05). Per-item python loop —
    the SWEEP VERDICT's measured bottleneck (~0.12s/item, CPU-bound).
    Superseded by _alt2_fact_buf_v1 (batch-vectorized decode, bit-
    identical by construction; scripts/seamtest_vector.py verifies).
    Original docstring follows.

    ALTERNATOR V2 commit adapter + cycle driver (2026-09-01). Consumes
    the realized pass-1 output dict (decode()'s key conventions: pres/
    ftype/op/dig/args/res logits + optional dup), discretizes ONLY
    confident slots (presence sigmoid > theta; ftype/res softmax top-prob
    > theta; args sigmoid > theta — args is BCE-trained 2-hot, a softmax
    read is wrong there), commits given/rel factors in the mint grammar,
    calls the symbolic half (alternator_bridge.ping: GAC propagation
    only), and packs the forced facts into (B, 24, 4) float32:
    [1.0, h/9, t/9, o/9] per known var, zeros elsewhere. Contradiction
    (mass None) or ANY per-item exception -> zeros for that item —
    silence, never a crash (the bridge contract). Numpy in, numpy out:
    detached by construction (the dual-terminal law). se_np rides for
    signature symmetry with build_slot_masks (unused)."""
    from alternator_bridge import ping   # lazy — scripts/ is on sys.path
    B = onp["pres"].shape[0]
    buf = np.zeros((B, K_VARS, 4), np.float32)
    if mass_out is not None:
        # THE MASS THREAD (apply_mass_thread.py, 2026-09-05): default
        # fill = m+1 (full 0..m domain — nothing known); ping rows
        # overwrite below. Contradiction keeps the fill (silence).
        mass_out[:] = np.asarray(m_arr, np.float64)[:, None] + 1.0

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _smax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    for bi in range(B):
        try:
            ftp = _smax(onp["ftype"][bi])          # (L, nft)
            rsp = _smax(onp["res"][bi])            # (L, K_VARS)
            agp = _sig(onp["args"][bi])            # (L, K_VARS) 2-hot BCE
            facs = []
            for j in range(L_FAC):
                if _sig(onp["pres"][bi, j]) <= theta:
                    continue
                ft = int(ftp[j].argmax())
                if ftp[j, ft] <= theta:
                    continue
                res = int(rsp[j].argmax())
                if rsp[j, res] <= theta:
                    continue
                if ft == 1:                        # given: digits carry value
                    digs = onp["dig"][bi, j].argmax(-1)
                    v = int(sum(d * 10 ** (N_DIG - 1 - i2)
                                for i2, d in enumerate(digs)))
                    if "sgn" in onp and onp["sgn"][bi, j] > 0:
                        v = -v                     # E1 negative literal (decode's rule;
                                                   # ultrareview 2026-09-06: the adapter
                                                   # must not launder a sign it read)
                    facs.append({"ftype": "given", "var": res, "value": v})
                elif ft == 0:                      # rel (decode conventions)
                    op = "add" if onp["op"][bi, j].argmax() == 0 else "mul"
                    if "dup" in onp and onp["dup"][bi, j] > 0:
                        if "dargs" in onp:         # door #12: the dedicated dup pointer
                            dsp = _smax(onp["dargs"][bi, j])
                            a0 = int(np.argmax(dsp))
                            if dsp[a0] <= theta:
                                continue
                        else:
                            a0 = int(np.argmax(onp["args"][bi, j]))
                            if agp[j, a0] <= theta:
                                continue
                        args = [a0, a0]
                    else:
                        top2 = np.argsort(-onp["args"][bi, j])[:2]
                        if float(agp[j, top2].min()) <= theta:
                            continue
                        args = sorted(int(a) for a in top2)
                    facs.append({"ftype": "rel", "op": op,
                                 "args": args, "result": res})
                # other ftypes: never committed (bridge grammar: given/rel)
            if not facs:
                continue
            nv = max([int(n_vars_arr[bi])]        # do_eval's nv convention
                     + [v + 1 for f in facs for v in
                        ([f["var"]] if f["ftype"] == "given"
                         else list(f["args"]) + [f["result"]])])
            facts, mass, _r = ping(nv, facs, int(m_arr[bi]))
            if mass is None:                       # contradiction: silence
                continue
            if mass_out is not None:               # the mass thread:
                _nmv = min(len(mass), mass_out.shape[1])
                mass_out[bi, :_nmv] = mass[:_nmv]  # post-GAC domain sizes
            for v, val in facts.items():
                if 0 <= v < K_VARS and 0 <= val <= 999:
                    buf[bi, v] = (1.0, (val // 100) / 9.0,
                                  (val // 10 % 10) / 9.0, (val % 10) / 9.0)
        except Exception:
            buf[bi] = 0.0                          # per-item silence
            if mass_out is not None:               # silence for mass too
                mass_out[bi] = float(m_arr[bi]) + 1.0
    return buf


def _alt2_fact_buf_v1(onp, se_np, n_vars_arr, m_arr, theta=0.9,
                      mass_out=None):
    """VECTORIZED commit adapter (scripts/apply_seam_vector.py, 2026-09-05).
    Bit-identical to _alt2_fact_buf_v0 by construction (verified by
    scripts/seamtest_vector.py, np.array_equal on 200 realistic inputs):
    the decode phase (presence sigmoid, ftype/res softmax+argmax, args
    sigmoid, digit argmax, op argmax, dup sign, top-2 args by raw logit)
    runs as a HANDFUL of whole-(B, L_FAC, ...) numpy ops instead of a
    B*L_FAC python-level loop of per-slot numpy calls — every reduction
    here is independent per (item, slot), so batching it changes nothing
    about the floating-point result (same reduction, same axis, same
    order; numpy sorts/reduces each 1-D slice along an axis identically
    regardless of what else rides alongside it in the array).

    The per-item python loop SURVIVES, but now only walks the (typically
    few) slots that pass every gate (`keep`), to assemble the facs list
    in the ORIGINAL ascending-j order and call the symbolic half
    (alternator_bridge.ping — per-item by nature, cheap per the ledger,
    NOT touched by this patch). Contradiction or any per-item exception
    still zeros that item's buf row only — the bridge contract, preserved
    verbatim."""
    from alternator_bridge import ping   # lazy — scripts/ is on sys.path
    B = onp["pres"].shape[0]
    buf = np.zeros((B, K_VARS, 4), np.float32)
    if mass_out is not None:
        # THE MASS THREAD (apply_mass_thread.py, 2026-09-05): default
        # fill = m+1 (full 0..m domain — nothing known); ping rows
        # overwrite below. Contradiction keeps the fill (silence).
        mass_out[:] = np.asarray(m_arr, np.float64)[:, None] + 1.0
    has_dup = "dup" in onp

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _smax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    # ---- whole-batch decode: (B, L_FAC, ...) numpy ops, once each ------
    pres_sig = _sig(onp["pres"])                          # (B, L)
    ftp = _smax(onp["ftype"])                              # (B, L, nft)
    ft_am = ftp.argmax(-1)                                  # (B, L)
    ft_conf = np.take_along_axis(ftp, ft_am[..., None], -1)[..., 0]

    rsp = _smax(onp["res"])                                 # (B, L, K_VARS)
    res_am = rsp.argmax(-1)                                 # (B, L)
    res_conf = np.take_along_axis(rsp, res_am[..., None], -1)[..., 0]

    agp = _sig(onp["args"])                                 # (B, L, K_VARS) BCE 2-hot
    op_am = onp["op"].argmax(-1)                            # (B, L)

    digs = onp["dig"].argmax(-1)                             # (B, L, N_DIG)
    place = (10 ** np.arange(N_DIG - 1, -1, -1)).astype(np.int64)
    given_val = (digs.astype(np.int64) * place).sum(-1)      # (B, L)
    if "sgn" in onp:                                         # E1 negative literal (decode's rule)
        given_val = np.where(onp["sgn"] > 0, -given_val, given_val)

    if "dargs" in onp:                                       # door #12: dedicated dup pointer
        dsp = _smax(onp["dargs"])                            # (B, L, K_VARS) CE-trained
        raw_a0 = dsp.argmax(-1)
        a0_conf = np.take_along_axis(dsp, raw_a0[..., None], -1)[..., 0]
    else:
        raw_a0 = onp["args"].argmax(-1)                      # (B, L) raw-logit argmax (dup path)
        a0_conf = np.take_along_axis(agp, raw_a0[..., None], -1)[..., 0]

    top2 = np.argsort(-onp["args"], axis=-1)[..., :2]        # (B, L, 2) same per-row sort as the loop
    top2_conf_min = np.take_along_axis(agp, top2, axis=-1).min(-1)
    top2_sorted = np.sort(top2, axis=-1)                      # ascending pair (matches sorted(...))

    dup_on = (onp["dup"] > 0) if has_dup else np.zeros_like(pres_sig, dtype=bool)

    active = (pres_sig > theta) & (ft_conf > theta) & (res_conf > theta)
    is_given = active & (ft_am == 1)
    is_rel = active & (ft_am == 0)
    rel_dup_ok = is_rel & dup_on & (a0_conf > theta)
    rel_nondup_ok = is_rel & (~dup_on) & (top2_conf_min > theta)
    keep = is_given | rel_dup_ok | rel_nondup_ok           # (B, L) slots to commit
    keep_rows = keep.any(axis=1)

    # ---- per-item assembly (ONLY over surviving slots) + ping ----------
    for bi in range(B):
        if not keep_rows[bi]:
            continue
        try:
            facs = []
            for j in np.nonzero(keep[bi])[0]:               # ascending j, original order
                j = int(j)
                if is_given[bi, j]:
                    facs.append({"ftype": "given", "var": int(res_am[bi, j]),
                                 "value": int(given_val[bi, j])})
                else:
                    op = "add" if op_am[bi, j] == 0 else "mul"
                    if dup_on[bi, j]:
                        a0 = int(raw_a0[bi, j])
                        args = [a0, a0]
                    else:
                        args = [int(a) for a in top2_sorted[bi, j]]
                    facs.append({"ftype": "rel", "op": op,
                                 "args": args, "result": int(res_am[bi, j])})
            nv = max([int(n_vars_arr[bi])]        # do_eval's nv convention
                     + [v + 1 for f in facs for v in
                        ([f["var"]] if f["ftype"] == "given"
                         else list(f["args"]) + [f["result"]])])
            facts, mass, _r = ping(nv, facs, int(m_arr[bi]))
            if mass is None:                       # contradiction: silence
                continue
            if mass_out is not None:               # the mass thread:
                _nmv = min(len(mass), mass_out.shape[1])
                mass_out[bi, :_nmv] = mass[:_nmv]  # post-GAC domain sizes
            for v, val in facts.items():
                if 0 <= v < K_VARS and 0 <= val <= 999:
                    buf[bi, v] = (1.0, (val // 100) / 9.0,
                                  (val // 10 % 10) / 9.0, (val % 10) / 9.0)
        except Exception:
            buf[bi] = 0.0                          # per-item silence
            if mass_out is not None:               # silence for mass too
                mass_out[bi] = float(m_arr[bi]) + 1.0
    return buf


def alt2_fact_buf(onp, se_np, n_vars_arr, m_arr, theta=0.9,
                  mass_out=None):
    """Dispatcher (scripts/apply_seam_vector.py, 2026-09-05): the
    vectorized decode by default; ALG_SEAM_V0=1 selects the pre-vector
    reference implementation for A/B fallback ONLY (not a shipping
    config — the seamtest is the authority on equivalence, not this
    flag's existence)."""
    fn = _alt2_fact_buf_v0 if os.environ.get("ALG_SEAM_V0") else _alt2_fact_buf_v1
    return fn(onp, se_np, n_vars_arr, m_arr, theta=theta,
              mass_out=mass_out)


# ===========================================================================
# PRECOMPUTE (GPU once) — small corpus, plain npz
# ===========================================================================

def do_precompute():
    from tinygrad import Tensor, dtypes
    from mycelium.llama_loader import (
        attach_llama_layers, load_llama_weights, LLAMA_3_2_1B_CFG, _rms_norm)

    class _H:
        pass
    host = _H()
    sd = load_llama_weights(os.path.join(_ROOT, ".cache/llama-3.2-1b-weights/model.safetensors"))
    attach_llama_layers(host, n_layers=4, sd=sd, cfg=LLAMA_3_2_1B_CFG)
    del sd
    jobs = [(TRAIN_NAME, ALG_TRAIN), (TEST_NAME, ALG_TEST)]
    if os.environ.get("PRECOMPUTE_ONLY"):
        jobs = [(n_, p_) for n_, p_ in jobs if n_ == os.environ["PRECOMPUTE_ONLY"]]
    for split, path in jobs:
        samples, ids, mask, offsets = tokenize(path)
        n = len(samples)
        # states stream straight to a disk-backed memmap — holding the full
        # (n, T, 2048) fp16 array in RAM beside the AM driver's pinned pages
        # OOMs at gen-7 scale (three kills, all during the giant-array write)
        states = np.lib.format.open_memmap(
            STATES_NPY.format(split=split), mode="w+", dtype=np.float16,
            shape=(n, T_ALG, H_TRUNK))
        for s0 in range(0, n, 8):
            sl = slice(s0, min(s0 + 8, n))
            x = host.llama_embed[Tensor(ids[sl], dtype=dtypes.int)]
            for layer in host.llama_layers:
                x = layer(x, host.llama_rope_cos, host.llama_rope_sin)
            x = _rms_norm(x, host.llama_layers[-1].ffn_norm, host.llama_cfg.rms_norm_eps)
            c = x.cast(dtypes.float).realize().numpy()
            assert np.isfinite(c).all()
            states[sl] = c.astype(np.float16)
            if (s0 // 8) % 200 == 0:
                print(f"[precompute] {split}: batch {s0//8}/{(n+7)//8}", flush=True)
        states.flush()
        shape = states.shape
        del states
        gold = build_gold(samples, offsets)
        sent = np.stack([sent_indices(s["text"], o, mask[i])
                         for i, (s, o) in enumerate(zip(samples, offsets))])
        np.savez(STATES_NPZ.format(split=split),
                 tokmask=mask.astype(np.uint8), sent=sent.astype(np.int8),
                 **{f"g_{k}": v for k, v in gold.items()})
        print(f"[precompute] {split}: {shape} (states -> memmap npy)", flush=True)


def load_alg(split):
    is_train = split == "train"
    split = TRAIN_NAME if is_train else (TEST_NAME if split == "test" else split)
    z = np.load(STATES_NPZ.format(split=split))
    _mix_path = ALG_TRAIN if is_train else ALG_TEST
    # THE SHA-FENCE (2026-08-10): staged arrays are a JOIN against the mix
    # file, keyed by TRAIN_NAME — a name, not content. Three fossils rode
    # that gap (the ALG_TRAIN_NAME replay trained on another mix's arrays
    # to the digit). Stamped npz -> hard-assert; unstamped -> say so loudly.
    if "mix_sha" in z.files:
        from mycelium.era import mix_sha16
        _want = str(z["mix_sha"])
        _got = mix_sha16(_mix_path)
        assert _got == _want, (
            f"SHA-FENCE: staged arrays were built from a different mix "
            f"(npz stamp {_want} != {_mix_path} sha {_got}) — the "
            f"name-key points at the wrong content; re-assemble.")
    elif is_train:
        print(f"[sha-fence] UNSTAMPED train arrays for '{split}' — "
              f"fence blind here until re-assembled", flush=True)
    samples = [json.loads(l) for l in open(_mix_path)]
    # CUSTODY-GOLD GUARD (deep clean 2026-07-28): load_alg is the single door
    # every battery consumer's gold flows through, and ALG_TRAIN/ALG_TEST env
    # is the only redirection mechanism. Pen rows (delegated book annotations,
    # gen.src_idx present) carry solution vectors that are pen-side scratch —
    # NEVER custody-side gold (543/550 banked book rows hold all-zero vectors
    # that disagree with harvest gold). Downstream code reads
    # sample["solution"][query_var] as gold in ~60 places; rather than patch
    # sixty readers, refuse the poison at the door.
    # Train side: pen rows are LAWFUL diet members (prose dose; supervision
    # comes from the states-file gold arrays, not the solution field) — warn
    # loudly so the count is visible. Test side: solution-as-gold judges
    # verdicts downstream, so pen rows REFUSE at the door.
    n_pen = sum(1 for s in samples if "src_idx" in s.get("gen", {}))
    if n_pen and is_train:
        # WARN-SIDE FENCE (bench 2026-07-28): an ambient warning becomes the
        # new silent hazard in six weeks. Pen rows on the train side are an
        # ERROR unless the launch explicitly acknowledges the diet via
        # ALG_ALLOW_PEN_TRAIN=1 (set by diet-aware trainer launches only).
        if os.environ.get("ALG_ALLOW_PEN_TRAIN") == "1":
            print(f"[load_alg] diet acknowledged: {n_pen} pen rows in "
                  f"ALG_TRAIN (solution fields are NOT gold — custody law)")
        else:
            raise RuntimeError(
                f"load_alg: {n_pen} PEN rows in ALG_TRAIN without "
                f"ALG_ALLOW_PEN_TRAIN=1 — if this is a deliberate diet mix, "
                f"set the env in the launch script; otherwise this path is "
                f"battery-adjacent and pen rows are an error (custody-gold "
                f"law, warn-side fence)")
    elif n_pen:
        # THE ROW_GOLD DOOR (2026-09-01): the guard's prescribed route,
        # built. Pen rows are admitted to a TEST fixture IFF every row's
        # solution[query_var] verifies against the INDEPENDENT key via
        # the custody organ (text identity; MATH harvest + GSM8K key
        # table). One miss or mismatch = hard error, fixture refused.
        from mycelium.custody_gold import row_gold, is_pen_row
        n_v = 0
        for smp in samples:
            if not is_pen_row(smp):
                continue
            gk = row_gold(smp)     # hard-errors on missing/non-int key
            sv = smp["solution"][smp["query_var"]]
            if int(gk) != int(sv):
                raise RuntimeError(
                    f"load_alg: pen row (gen.src_idx="
                    f"{smp['gen'].get('src_idx')}) solution[query]={sv} "
                    f"CONTRADICTS independent key {gk} — custody-gold "
                    f"law, fixture refused")
            n_v += 1
        print(f"[load_alg] custody: {n_v} pen rows KEY-VERIFIED via "
              f"row_gold (test fixture admitted; solution fields "
              f"independently confirmed)")
    gold = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
    if os.path.exists(STATES_NPY.format(split=split)):
        states = np.load(STATES_NPY.format(split=split), mmap_mode="r")
        # SAMPLES-STATES DESYNC GUARD (deep clean 2026-07-30): samples come
        # from ALG_TRAIN/ALG_TEST while states come from ALG_TRAIN_NAME's
        # files — two env vars with no cross-check. A forgotten
        # ALG_TRAIN_NAME would silently pair states[i] with samples[i] from
        # two unrelated corpora. Fail loudly instead.
        if len(states) != len(samples):
            raise RuntimeError(
                f"load_alg({split}): states file has {len(states)} rows but "
                f"the samples jsonl has {len(samples)} — ALG_TRAIN_NAME and "
                f"ALG_TRAIN point at different corpora (desync guard)")
    else:
        states = z["states"]   # legacy artifacts (gen<=6): states inside npz
    return samples, states, z["tokmask"], gold, z["sent"]


# ===========================================================================
# MODEL — two slot banks + bilinear pointers (all supervised)
# ===========================================================================

def build_params(seed=0):
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
              f"({_polar_groups()} group(s)) QROT={POLAR_QROT}"
              f"({'off' if not POLAR_QROT else ('mixer' if POLAR_QROT == 1 else 'main+mixer')}) "
              f"STAMP={POLAR_STAMP} FLOOR={os.environ.get('ALG_PC_FLOOR', '0')} "
              f"| sextet = mycelium/rotor_clock.wheel_table() "
              f"(frozen, ungained, breath-0 outside time)", flush=True)

    def t(a):
        x = Tensor(a.astype(np.float32), dtype=dtypes.float,
                   requires_grad=True).contiguous().realize()
        x.requires_grad = True
        return x

    def lin(i, o):
        return t(rng.randn(i, o) / math.sqrt(i)), t(np.zeros((o,)))

    p = {}
    _rngF = np.random.RandomState(seed + 9000)   # FED stream: fed
                                                 # tensors never move
                                                 # the base rng stream
    p["waist_w"], p["waist_b"] = lin(H_TRUNK, H_W)
    if ALG_SEPHASE_W:
        _ww = p["waist_w"].numpy()
        _ww += phase_alphabet(H_TRUNK, H_W, 1.0 / math.sqrt(H_TRUNK), rng) * 0.5
        p["waist_w"] = t(_ww)
    if int(os.environ.get("ALG_SEPHASE", "0")):
        # the properly-wired receiver (2026-08-17): six-phase structure seeded
        # INTO the learned sync channel (init, not bias — the model may keep
        # or reshape it; amplitude matches the native init scale)
        if int(os.environ.get("SEPHASE_SCRAMBLE", "0")):
            _s6 = np.random.RandomState(9).uniform(0, 2 * np.pi, SENT_MAX)
        else:
            _s6 = np.arange(SENT_MAX) % 6 * (np.pi / 3.0)
        _d6 = np.arange(H_W) * (2 * np.pi / H_W)
        _se = 0.1 * np.sqrt(2.0) * np.cos(_s6[:, None] + _d6[None, :] * 6)
        p["sent_emb"] = t(_se + rng.randn(SENT_MAX, H_W) * 0.02)
    else:
        p["sent_emb"] = t(rng.randn(SENT_MAX, H_W) * 0.1)
    if ALG_SEPHASE_Q:
        if SEPHASE_Q_SCRAMBLE:
            _rs = np.random.RandomState(227)
            def _scram(n, dims, scale):
                _ph = _rs.uniform(0, 2 * np.pi, n)
                _d = np.arange(dims)
                _pt = scale * np.sqrt(2.0) * np.cos(_ph[:, None] + _d[None, :] * (2 * np.pi / dims) * (np.arange(n)[:, None] + 1))
                return (_pt + rng.randn(n, dims) * scale * 0.2).astype(np.float32)
            p["vq"] = t(_scram(K_VARS, H_W, 0.02))
            p["fq"] = t(_scram(L_FAC, H_W, 0.02))
        else:
            p["vq"] = t(phase_alphabet(K_VARS, H_W, 0.02, rng))
            p["fq"] = t(phase_alphabet(L_FAC, H_W, 0.02, rng))
    else:
        p["vq"] = t(rng.randn(K_VARS, H_W) * 0.02)
        p["fq"] = t(rng.randn(L_FAC, H_W) * 0.02)
    p["qq"] = t(rng.randn(1, H_W) * 0.02)
    for nm in ("wq", "wk", "wv", "wo"):
        p[f"attn_{nm}"], p[f"attn_{nm}_b"] = lin(H_W, H_W)
    p["ffn_w1"], p["ffn_b1"] = lin(H_W, 2 * H_W)
    p["ffn_w2"], p["ffn_b2"] = lin(2 * H_W, H_W)
    if FED_FFN:
        # FED item 4: FFN 4x — extend by CONCATENATION so the base rng
        # stream is untouched. New w1 columns: fresh features (fed
        # stream, native scale); new b1: zeros; new w2 ROWS: ZEROS (the
        # door — new hidden units speak through zeros, birth-identical
        # in exact arithmetic, live grads from step one). PAD-WARM
        # (do_train's loader) lands trained 2x weights on the prefix.
        _fw1 = np.concatenate(
            [p["ffn_w1"].detach().numpy(),
             (_rngF.randn(H_W, 2 * H_W) / math.sqrt(H_W))
             .astype(np.float32)], 1)
        _fb1 = np.concatenate(
            [p["ffn_b1"].detach().numpy(),
             np.zeros(2 * H_W, np.float32)])
        _fw2 = np.concatenate(
            [p["ffn_w2"].detach().numpy(),
             np.zeros((2 * H_W, H_W), np.float32)], 0)   # ZERO door
        p["ffn_w1"] = t(_fw1)
        p["ffn_b1"] = t(_fb1)
        p["ffn_w2"] = t(_fw2)
    p["h_pres"], p["h_pres_b"] = lin(H_W, 1)
    # ALG2=1 -> tranche geometry (4-way ftype + selector head). Default keeps
    # the legacy 2-way build BYTE-COMPATIBLE with deployed checkpoints — every
    # lattice script loads the old ckpt through here.
    if int(os.environ.get("ALG2", "0")):
        nft = int(os.environ.get("ALG_FTYPES", "4"))  # 6 = +pct/fdiv (T2)
        p["h_ftype"], p["h_ftype_b"] = lin(H_W, nft)
        p["h_sel"], p["h_sel_b"] = lin(H_W, 4)       # larger/smaller/even/odd
    else:
        p["h_ftype"], p["h_ftype_b"] = lin(H_W, 2)
    p["h_op"], p["h_op_b"] = lin(H_W, 2)
    if int(os.environ.get("ALG_DUP", "0")):   # gen-9: arg-multiplicity bit
        p["h_dup"], p["h_dup_b"] = lin(H_W, 1)
    p["h_islit"], p["h_islit_b"] = lin(H_W, 1)
    p["h_dig"], p["h_dig_b"] = lin(H_W, N_DIG * 10)
    if ALG_WIDE:                              # E1: the sign terminal
        p["h_sgn"], p["h_sgn_b"] = lin(H_W, 1)
    if ALG_REF:                               # E-FLOOR: token-grain referent
        p["h_ref"], p["h_ref_b"] = lin(H_W, K_VARS)
    if int(os.environ.get("ALG2", "0")) and \
            int(os.environ.get("ALG_FTYPES", "4")) >= 7:   # gen-15: OP_APPLY
        p["h_dig2"], p["h_dig2_b"] = lin(H_W, N_DIG * 10)
        p["W_y"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    if K_B > 1:   # BRICK-P: masked slot-to-slot breathing (2026-07-09)
        if ALG_SEPHASE_SETTLE:   # the settle transceiver: slot-to-slot
            _stp = phase_alphabet(H_W, H_W, 1.0 / math.sqrt(H_W), rng)
            p["W_bq"] = t(_stp + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
            p["W_bq_b"] = t(np.zeros((H_W,)))
            p["W_bk"] = t(_stp + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
            p["W_bk_b"] = t(np.zeros((H_W,)))
        else:
            p["W_bq"], p["W_bq_b"] = lin(H_W, H_W)
        if not ALG_SEPHASE_SETTLE:
            p["W_bk"], p["W_bk_b"] = lin(H_W, H_W)
        p["W_bv"], p["W_bv_b"] = lin(H_W, H_W)
        p["W_bo"] = t(np.zeros((H_W, H_W)))          # zero-init: breath deltas
        p["W_bo_b"] = t(np.zeros(H_W))               # start silent
        if int(os.environ.get("ALG_SEPHASE_B", "0")):
            # sephase-B (2026-08-18): the temporal sibling — breath_emb born
            # with orthogonal phase stamps at native amplitude (init, the
            # winning idiom; SEPHASE_B_SCRAMBLE randomizes phases, placebo)
            _bk = np.arange(K_B)
            if int(os.environ.get("SEPHASE_B_SCRAMBLE", "0")):
                _bph = np.random.RandomState(227).uniform(0, 2 * np.pi, K_B)
            else:
                _bph = _bk * np.pi / max(K_B, 1)
            _bd = np.arange(H_W)
            if SEPHASE_B_BAND == 2:  # the 1/3 dose: stride-2 bands
                _be = 0.02 * np.sqrt(2.0 / 3.0) * sum(   # {2k,2k+1,2k+2} —
                    np.cos(_bph[:, None] + _bd[None, :] * (2 * np.pi / H_W)
                           * (2 * _bk[:, None] + 1 + _f)) for _f in range(3))
            elif SEPHASE_B_BAND:  # the orientation form: bands {k,k+1,k+2} —
                _be = 0.02 * np.sqrt(2.0 / 3.0) * sum(   # adjacent corr ~2/3
                    np.cos(_bph[:, None] + _bd[None, :] * (2 * np.pi / H_W)
                           * (_bk[:, None] + 1 + _f)) for _f in range(3))
            else:
                _be = 0.02 * np.sqrt(2.0) * np.cos(
                    _bph[:, None] + _bd[None, :] * (2 * np.pi / H_W) * (_bk[:, None] + 1))
            p["breath_emb"] = t(_be + rng.randn(K_B, H_W) * 0.004)
        else:
            p["breath_emb"] = t(rng.randn(K_B, H_W) * 0.02)
        p["breath_gate"] = t(np.full(K_B, float(os.environ.get(
            "BREATH_GATE_INIT", "-2.0"))))   # init-closed convex blend;
            # door #44-RESCUE: the organ does not self-open (gates at init
            # after 4k) — BREATH_GATE_INIT=0.0 opens the door at birth
        if int(os.environ.get("ALG_RINGS", "0")):    # RUNG-3 v1: the soft pawl
            p["W_cmt"] = t(np.zeros((H_W, 1)))       # zero-init commit head
            p["W_cmt_b"] = t(np.full(1, float(os.environ.get(
                "CMT_B_INIT", "-4.0"))))             # init: commit ~nothing;
                                                     # door #54-R: bias-open
        if int(os.environ.get("ALG_ALT21", "0")):
            # ALTERNATOR v2.1 (2026-09-02, word given): the INTEGRATE
            # station pair — a SECOND bank-attention (slots<-tokens) +
            # a SECOND slot-mixer per breath_step (stations 3-4 of the
            # four-layer step; the June engine's 4-per-breath precedent).
            # INIT LAW (ResNet/V11 zero-init): each block's OUTPUT
            # projection starts at ZERO so the ALG_ALT21=1 forward is
            # identical to baseline at birth (rung 1 of the bring-up
            # ladder); every other tensor copies its original's idiom.
            p["alt21_attn_wq"], p["alt21_attn_wq_b"] = lin(H_W, H_W)
            p["alt21_attn_wk"], p["alt21_attn_wk_b"] = lin(H_W, H_W)
            p["alt21_attn_wv"], p["alt21_attn_wv_b"] = lin(H_W, H_W)
            p["alt21_attn_wo"] = t(np.zeros((H_W, H_W)))   # ZERO: silent birth
            p["alt21_attn_wo_b"] = t(np.zeros(H_W))
            if ALG_SEPHASE_SETTLE:   # mirror the settle-transceiver idiom
                _stp21 = phase_alphabet(H_W, H_W, 1.0 / math.sqrt(H_W), rng)
                p["alt21_W_bq"] = t(_stp21 + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
                p["alt21_W_bq_b"] = t(np.zeros((H_W,)))
                p["alt21_W_bk"] = t(_stp21 + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
                p["alt21_W_bk_b"] = t(np.zeros((H_W,)))
            else:
                p["alt21_W_bq"], p["alt21_W_bq_b"] = lin(H_W, H_W)
                p["alt21_W_bk"], p["alt21_W_bk_b"] = lin(H_W, H_W)
            p["alt21_W_bv"], p["alt21_W_bv_b"] = lin(H_W, H_W)
            p["alt21_W_bo"] = t(np.zeros((H_W, H_W)))      # ZERO: silent birth
            p["alt21_W_bo_b"] = t(np.zeros(H_W))
        if int(os.environ.get("ALG_MASKHEAD", "0")):
            # THE MASK HEAD (2026-09-05, word given): the fourth trained
            # organ — the learned PRECISION channel (mask_head_spec.md).
            # A dedicated MH_HEADS-head attention bank over H_W with its
            # own Wq/Wk/Wv, a gelu integration layer (mh_wu), a pair-key
            # projection (mh_wp), a mask-context encoder (facts +
            # graded adjacency + mass port + breath phase -> kv space),
            # and an atlas-page port projection (mh_atlas_w). TWO
            # ZERO-INIT output doors: mh_wo (output projection — the
            # ResNet law) and mh_headmix (per-head score combiner), so
            # the emitted bias is EXACTLY zero at birth; mh_gain is
            # AJAR (0.02, gate-deadlock corollary) — the softplus slope
            # gain*sigmoid(raw)*open is nonzero from step one, so both
            # doors self-open. Sized UP per the word: ~1.97M at 4 heads
            # (compute), state["mh_prev"] (storage), parse-CE-only
            # training through the re-masked pass (learnable data).
            p["mh_wq"], p["mh_wq_b"] = lin(H_W, H_W)
            p["mh_wk"], p["mh_wk_b"] = lin(H_W, H_W)
            p["mh_wv"], p["mh_wv_b"] = lin(H_W, H_W)
            p["mh_wu"], p["mh_wu_b"] = lin(H_W, H_W)
            p["mh_wo"] = t(np.zeros((H_W, H_W)))    # ZERO door 1: birth
            p["mh_wo_b"] = t(np.zeros(H_W))         # is bit-identical
            p["mh_wp"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
            p["mh_enc1"], p["mh_enc1_b"] = lin(MH_CTX_F, 256)
            p["mh_enc2"], p["mh_enc2_b"] = lin(256, H_W)
            p["mh_atlas_w"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
            p["mh_headmix"] = t(np.zeros(MH_HEADS)) # ZERO door 2
            p["mh_gain"] = t(np.full(1, 0.02))      # AJAR (the law)
        if FED_MIXER:
            # FED item 1: per-head ZERO-INIT gains — the twin path's
            # single door (heads reshape the trained W_bq/W_bk/W_bv;
            # output speaks through the trained W_bo)
            p["fed_mx_hg"] = t(np.zeros(MX_HEADS))
        if FED_NL0 and int(os.environ.get("ALG_MASKHEAD", "0")):
            # FED item 8: the breath-0 invariant feed's ZERO door into
            # the mask-head context (inert when the organ is off)
            p["fed_nl0_w"] = t(np.zeros((H_W, H_W)))
        pass
    if int(os.environ.get("ALG_BINDBUS", "0")):
        _bd = int(os.environ.get("ALG_BIND_D", "128"))
        # tabula-rasa cleanup: the monolith is the only living bind head
        p["W_bind1"] = t(rng.randn(H_W, 512) / math.sqrt(H_W))
        p["W_bind1_b"] = t(np.zeros(512))
        p["W_bind2"] = t(rng.randn(512, _bd) / math.sqrt(512))
    if int(os.environ.get("ALG_RINGS", "0")):
        if True:
            if int(os.environ.get("ALG_CMT_REG", "0")):
                p["w_cmt_reg"] = t(np.zeros(1))      # zero-init: register
                                                     # enters at no-effect
    p["W_args"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    if int(os.environ.get("ALG_DUPPTR", "0")):
        p["W_dargs"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    if ALG_DIAL:                              # door #45: dialect args pointer
        p["W_iargs"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    p["W_res"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    p["W_query"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    if FED_POINTERS:
        # FED item 2: PF_FORMS extra bilinears per pointer, zero gains
        for _pfn in ("args", "res", "query"):
            p["fed_pf_" + _pfn + "_W"] = t(np.stack(
                [_rngF.randn(H_W, H_W).astype(np.float32)
                 / math.sqrt(H_W) for _ in range(PF_FORMS)]))
            p["fed_pf_" + _pfn + "_g"] = t(np.zeros(PF_FORMS))
    if FED_MACRO and "h_dig2" in p:
        # FED item 5: the macro value system's forms (W_y bilinear,
        # h_dig2 linear) — the audit's previously-unflagged organ
        p["fed_pf_y_W"] = t(np.stack(
            [_rngF.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W)
             for _ in range(PF_FORMS)]))
        p["fed_pf_y_g"] = t(np.zeros(PF_FORMS))
        p["fed_pf_dig2_W"] = t(np.stack(
            [_rngF.randn(H_W, N_DIG * 10).astype(np.float32)
             / math.sqrt(H_W) for _ in range(PF_FORMS)]))
        p["fed_pf_dig2_g"] = t(np.zeros(PF_FORMS))
    if FED_WAIST:
        # FED item 3: residual waist layer, ZERO output door
        p["fed_w2a"] = t(_rngF.randn(H_W, H_W).astype(np.float32)
                         / math.sqrt(H_W))
        p["fed_w2a_b"] = t(np.zeros(H_W))
        p["fed_w2b"] = t(np.zeros((H_W, H_W)))   # ZERO door (ResNet law)
        p["fed_w2b_b"] = t(np.zeros(H_W))
    if FED_SCRATCH:
        # FED item 6: +8 scratch slot embeds appended to fq (pad-warm
        # loads the trained 24; the doctrine: factor slots stay 24,
        # scratch scales with the tier)
        p["fq"] = t(np.concatenate(
            [p["fq"].detach().numpy(),
             (_rngF.randn(N_SCR, H_W) * 0.02).astype(np.float32)], 0))
    if ALG_SIXWAVE:      # door #62: carrier gate — structure enters at zero
        p["sw_g"] = t(np.zeros((1,)))
    if int(os.environ.get("ALG_BUSGARAGE", "0")):
        # THE PARKING GARAGE (2026-08-30, word given): typed relational
        # mail — deposits are role-bound wires; retrieval is content-
        # addressed attention (drop-off semantics; the coordination law).
        # Gate AJAR (gate-deadlock corollary — learnable path behind it);
        # W_busr decode-init arrives via the warm file.
        _bd4 = int(os.environ.get("ALG_BIND_D", "128"))
        p["W_gq"] = t(rng.randn(H_W, _bd4) / math.sqrt(H_W))
        p["W_busr"] = t(rng.randn(4 * _bd4, H_W) / math.sqrt(4 * _bd4))
        p["bus_g"] = t(np.full(1, 0.02))
    if int(os.environ.get("ALG_DETWAVE", "0")):
        # ALTERNATOR v1a THE DETERMINATION WAVE (2026-08-31, word given):
        # per-breath solvability closure over the committed graph — givens
        # seed, forced moves propagate (3 sweeps = multi-hop content no
        # single attention step derives). "determination" is a registered
        # DIAGNOSTIC (never supervised); as an INPUT it is lawful.
        p["W_det"] = t(rng.randn(3, H_W) / math.sqrt(3))
        p["det_g"] = t(np.full(1, 0.02))
    if int(os.environ.get("ALG_ALT2", "0")):
        # ALTERNATOR V2 (2026-09-01, word given): cycle-level ping-pong —
        # pass-1 commits confident slots, the bridge propagates
        # (alternator_bridge.ping; meter-divergence law: the check calls
        # its organ), forced facts re-enter pass-2 as var-slot
        # conditioning. W_fact is the ONLY path for facts: small-random
        # init (the W_det idiom, NEVER zero); gate ajar (0.02, the law).
        p["W_fact"] = t(rng.randn(4, H_W) / math.sqrt(4))
        p["alt2_g"] = t(np.full(1, 0.02))
    if int(os.environ.get("ALG_ROUTER", "0")):
        assert int(os.environ.get("ALG_BREATH", "1")) > 1, \
            "ROUTER emits only inside the breath loop (no-silent-fallbacks)"
        # v3 THE ROUTER HEAD (2026-09-01, word given): learned token-grain
        # routing — snap-conditioned slot queries vs waist keys, soft bias
        # into the BANK's reads (the measured artery); trained on its OWN
        # span loss (bootstrap law; the dual-terminal contract by design)
        p["W_rs"] = t(rng.randn(73, H_W) / math.sqrt(73))
        p["W_ra"] = t(rng.randn(H_W, 64) / math.sqrt(H_W))
        p["W_rb"] = t(rng.randn(H_W, 64) / math.sqrt(H_W))
        p["r_gain"] = t(np.full(1, float(os.environ.get("R_GAIN_INIT", "0.02"))))
        # rescue 2026-09-01: default aligned to the AJAR law (0.02);
        # sweepable via R_GAIN_INIT (the 0.1 deviation was unswept)
    if int(os.environ.get("ALG_ALTMASK", "0")):
        # THE ALTERNATOR v0 (2026-08-30, word given): committed adjacency
        # (producer->consumer edges from the lattice snaps) reshapes the
        # slot-mixer per breath — the mask BREATHES and its content is
        # SYMBOLIC. Ajar gain (escape-valve lesson).
        p["alt_g"] = t(np.full(1, 0.02))
    if ALG_NOTEBOOK:     # the cathedral (2026-08-18)
        if ALG_SEPHASE_PAIR:   # the transceiver: ink and query born in the
            _shared = phase_alphabet(H_W, H_W, 1.0 / math.sqrt(H_W), rng)
            p["W_sil"] = t(_shared + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
            p["W_nq"] = t(_shared + rng.randn(H_W, H_W).astype(np.float32) / math.sqrt(H_W) * 0.2)
        else:                   # same coordinate code
            p["W_sil"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
            p["W_nq"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
        if FED_SHELF:
            # FED item 9: the second ink lane (2 rows/breath — the
            # lawful shelf-16; stamps rows 8..15). Read rides the
            # ZERO-INIT gain; fed_sil2 wakes through it (item-2 law).
            p["fed_sil2"] = t(_rngF.randn(H_W, H_W).astype(np.float32)
                              / math.sqrt(H_W))
            p["fed_nb_g"] = t(np.zeros(1))
    if int(os.environ.get("ALG_POSCH", "0")):
        # THE POSITION CHANNEL (2026-08-24, word given; deficit twice-banked:
        # ladder depth-at-chance + A0 balanced .222): supervised depth/term
        # terminals — position TAUGHT into the states (gold outcomes, the
        # lawful side of the Goodhart boundary)
        p["h_depth"] = t(rng.randn(H_W, 6) / math.sqrt(H_W))
        p["h_depth_b"] = t(np.zeros(6))
        p["h_term"] = t(rng.randn(H_W, 1) / math.sqrt(H_W))
        p["h_term_b"] = t(np.zeros(1))
    if int(os.environ.get("ALG_OPCOUNT", "0")):
        # LEVER 3 (2026-08-25, word given): the op-multiset COUNT head —
        # pooled waist -> per-op-class count logits, supervised at the ROW
        # grain from mint-source gold (whole-row exactness is where
        # enumeration lives; per-slot decode compounds, a count read doesn't)
        p["W_opc1"] = t(rng.randn(H_W, 256) / math.sqrt(H_W))
        p["W_opc1_b"] = t(np.zeros(256))
        p["W_opc2"] = t(rng.randn(256, len(OPC_CLASSES) * (OPC_CAP + 1))
                        / math.sqrt(256))
        p["W_opc2_b"] = t(np.zeros(len(OPC_CLASSES) * (OPC_CAP + 1)))
    if int(os.environ.get("ALG_TRUNK_LORA", "0")):
        # THE TRUNK-COPY LoRA (2026-08-22, constitutional word): rank-r
        # adapters on a RUNTIME copy of L0-L3 (wq/wo/wdown — the hook
        # LlamaLayer already carries). Base trunk safetensors NEVER touched;
        # research lineage only; zero-init B = zero delta at step 0.
        _lr_r = int(os.environ.get("ALG_LORA_R", "16"))
        _span = os.environ.get("ALG_LORA_SPAN", "0123")   # pool axis: layers
        _proj = os.environ.get("ALG_LORA_PROJ", "all")    # pool axis: all|wq|wod
        _pset = {"all": ("wq", "wo", "wdown"), "wq": ("wq",),
                 "wod": ("wo", "wdown")}[_proj]
        _lrng = np.random.RandomState(seed + 7000)
        for _li in range(4):
            if str(_li) not in _span: continue
            for _nm, _din in (("wq", 2048), ("wo", 2048), ("wdown", 8192)):
                if _nm not in _pset: continue
                p[f"lora{_li}_{_nm}_A"] = t(_lrng.randn(_din, _lr_r) * 0.01)
                p[f"lora{_li}_{_nm}_B"] = t(np.zeros((_lr_r, 2048)))
    if ALG_POLAR and POLAR_D:
        # THE CONTENT-PLANE WAIST's two parameters. BIRTH = the champion's
        # OWN manifold: W_down = V, W_up = V^T with V the top-d principal
        # directions of the content dims of the dumped fedon242 polar
        # directions (scripts/polar_waist_init.py; both fixtures, all
        # breaths, all slots, centered). A random init here would be a
        # fresh 98k-parameter organ dropped into a warm continuation —
        # blur, not compute. A MISSING FILE IS A HARD ERROR: no silent
        # random init, ever (the no-silent-fallbacks rule).
        _cdim = _polar_sink()[0]
        _pwi = POLAR_D_INIT or f".cache/polar_waist_init_d{POLAR_D}.npz"
        assert os.path.exists(_pwi), (
            f"ALG_POLAR_D={POLAR_D} but {_pwi} is missing — the content "
            f"waist is born as the PCA projection of the champion's own "
            f"manifold, never at random. Build it with "
            f"scripts/polar_waist_init.py (ALG_POLAR_D_INIT overrides "
            f"the path).")
        _pz = np.load(_pwi)
        _wd0 = np.asarray(_pz["W_down"], np.float32)
        _wu0 = np.asarray(_pz["W_up"], np.float32)
        assert _wd0.shape == (len(_cdim), POLAR_D) \
            and _wu0.shape == (POLAR_D, len(_cdim)), (
                f"{_pwi}: W_down{_wd0.shape} / W_up{_wu0.shape} do not "
                f"match ({len(_cdim)}, {POLAR_D}) / ({POLAR_D}, "
                f"{len(_cdim)}) — wrong width or wrong band allocation")
        assert np.array_equal(np.asarray(_pz["content_dims"], np.int64),
                              _cdim), (
            f"{_pwi} was built against a DIFFERENT band allocation than "
            f"{POLAR_BANDS} — the waist would collapse the wrong dims")
        p["polar_wd"] = t(_wd0)
        p["polar_wu"] = t(_wu0)
    global _POLAR_SINK_SHOWN
    if ALG_POLAR and (POLAR_D or POLAR_EM) and not _POLAR_SINK_SHOWN:
        # THE SINK DOOR: one line, once, naming what is about to run.
        # Silent when both doors are shut.
        _POLAR_SINK_SHOWN = True
        _nc = len(_polar_sink()[0])
        _dtxt = "off"
        if POLAR_D:
            _dtxt = ("%d content dims -> %d -> %d, %d params, init %s"
                     % (_nc, POLAR_D, _nc, 2 * _nc * POLAR_D,
                        POLAR_D_INIT
                        or (".cache/polar_waist_init_d%d.npz" % POLAR_D)))
        _etxt = ("off" if not POLAR_EM else
                 "kappa=%g on the clock planes along the slot-mask lanes, "
                 "FIXED (not learnable), 0 params" % POLAR_EM)
        print("[polar-sink] ALG_POLAR_D=%d (%s) | ALG_POLAR_EM=%g (%s) | "
              "the waist never touches the %d clock dims and the coupling "
              "never touches the %d content dims; each organ restores its "
              "OWN block norm, so ||u|| == 1 still holds and r is untouched"
              % (POLAR_D, _dtxt, POLAR_EM, _etxt, H_W - _nc, _nc),
              flush=True)
    return p


NB_STAMPS = None
if ALG_NOTEBOOK:
    assert int(os.environ.get("ALG_BREATH", "1")) <= 8, "NB_STAMPS holds 8 rows (audit #11)"
    _ks = np.arange(NB_ROWS)   # FED item 9: 16 rows under the
    _ds = np.arange(512)           # family; rows 0..7 are bitwise the
    NB_STAMPS = np.cos(_ks[:, None] * np.pi / 8.0 * 7   # legacy table
                       + _ds[None, :] * (2 * np.pi / 512)
                       * (_ks[:, None] + 1)).astype(np.float32)
    NB_STAMPS /= np.linalg.norm(NB_STAMPS, axis=1, keepdims=True)
    _cc = np.abs(NB_STAMPS @ NB_STAMPS.T - np.eye(NB_ROWS)).max()
    assert _cc < 0.35, f"sharpness assert FAILED: stamp cos {_cc:.3f}"


def _bind_codes_path():
    """audit 2026-09-01: the default DERIVED from ALG_BIND_D (launcher
    memory is not a dependency mechanism — the prose-promotions law)."""
    d = int(os.environ.get("ALG_BIND_D", "128"))
    default = {128: ".cache/bindbus_codes.npz",
               256: ".cache/bindbus_codes256.npz",
               512: ".cache/bindbus_codes512.npz"}.get(d,
                    ".cache/bindbus_codes.npz")
    path = os.environ.get("BIND_CODES", default)
    import numpy as _np_bc
    _cb = _np_bc.load(path)["CB"]
    assert _cb.shape[1] == d, \
        (f"BIND_CODES pair-count mismatch: {path} carries D={_cb.shape[1]} "
         f"but ALG_BIND_D={d} — stale codebook")
    return path



def _fed_core(x):
    """FED item 6 (scratch): emission, grading, gold indexing, decode
    and every loss live on the FIRST L_FAC rows only — scratch rows are
    attention citizens, never supervised (the Goodhart fence: graded
    scratch stops being scratch). No-op when scratch is off or x is not
    slot-major (nothing else in the stack is 32-wide)."""
    if N_SCR and x.shape[1] == L_TOT:
        return x[:, :L_FAC]
    return x


def _fed_pf(p, name, s, vst, base):
    """FED items 2/5: multi-form emission — base + sum_f g_f * form_f
    with ZERO-INIT gains. Grad-aliveness (verified): at g=0 the form
    weights carry zero-but-DEFINED grads (dL/dW_f = g_f * .. = 0, never
    None) while dL/dg_f = <upstream, form_f> != 0 generically — one
    optimizer step opens the gate; no deadlock. At g=0 the added term
    is exact zeros (0 * finite), so birth is bit-identical. vst=None
    means a plain linear form (h_dig2's shape)."""
    gk = "fed_pf_" + name + "_g"
    if not (ALG_FED and gk in p):
        return base
    W = p["fed_pf_" + name + "_W"]
    g = p[gk]
    extra = None
    for f in range(PF_FORMS):
        form = ((s @ W[f]) @ vst.transpose(-2, -1)) if vst is not None \
            else (s @ W[f])
        term = g[f] * form
        extra = term if extra is None else extra + term
    return base + extra


_STEP_TAP = None    # the step trainer's stage-0 seam (the _CENSUS/_IMP
                    # hook pattern): None everywhere except under
                    # scripts/step_trainer.py — inert in every other path


def _make_bank(p, waist, tokmask, B):
    """forward()'s bank attention, factored BY PURE CODE MOTION
    (apply_step_trainer.py, 2026-09-03) so the step trainer can rebuild
    the closure over ITS OWN waist tensor. forward's call sites are
    unchanged; behavior bit-identical by construction."""
    def bank(queries, nq, extra=None, pbias=None, rbias=None):
        q_in = queries.unsqueeze(0) + (extra if extra is not None else 0)
        q = q_in @ p["attn_wq"] + p["attn_wq_b"]
        k = waist @ p["attn_wk"] + p["attn_wk_b"]
        v = waist @ p["attn_wv"] + p["attn_wv_b"]
        hd = H_W // N_HEADS
        qh = q.reshape(B if extra is not None else 1, nq, N_HEADS, hd).permute(0, 2, 1, 3)
        kh = k.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
        vh = v.reshape(B, -1, N_HEADS, hd).permute(0, 2, 1, 3)
        sc = (qh @ kh.transpose(-2, -1)) / math.sqrt(hd)
        if pbias is not None:   # door #62: six-wave phase-resonance bias
            sc = sc + pbias
        if rbias is not None:   # v3: the router's soft token bias (never
            sc = sc + rbias.unsqueeze(1) * p["r_gain"].reshape(1, 1, 1, 1)
                                # hard -inf — A0's grave)
        sc = sc.clip(-1e4, 1e4) + (1.0 - tokmask.reshape(B, 1, 1, -1)) * -1e4
        at = sc.softmax(-1)
        st = (at @ vh).permute(0, 2, 1, 3).reshape(B, nq, H_W)
        st = st @ p["attn_wo"] + p["attn_wo_b"] + q_in.reshape(-1, nq, H_W)
        st = st + ((st @ p["ffn_w1"] + p["ffn_b1"]).gelu() @ p["ffn_w2"] + p["ffn_b2"])
        return st, at.mean(1)

    return bank


def _heads_of(p, s, vst, B):
    """forward()'s emission heads, factored BY PURE CODE MOTION
    (apply_step_trainer.py, 2026-09-03): the step trainer runs these on
    intermediate breath states at every seam (commit adapter) and on the
    final state with the seam-current vst. Single source of truth."""
    s = _fed_core(s)   # FED scratch: grade only the true factor rows
    return {
        "pres": (s @ p["h_pres"] + p["h_pres_b"]).squeeze(-1),
        "ftype": s @ p["h_ftype"] + p["h_ftype_b"],
        "op": s @ p["h_op"] + p["h_op_b"],
        **({"sel": s @ p["h_sel"] + p["h_sel_b"]} if "h_sel" in p else {}),
        **({"dup": (s @ p["h_dup"] + p["h_dup_b"]).squeeze(-1)}
           if "h_dup" in p else {}),
        "islit": (s @ p["h_islit"] + p["h_islit_b"]).squeeze(-1),
        "dig": (s @ p["h_dig"] + p["h_dig_b"]).reshape(B, L_FAC, N_DIG, 10),
        **({"sgn": (s @ p["h_sgn"] + p["h_sgn_b"]).squeeze(-1)}
           if "h_sgn" in p else {}),
        "args": _fed_pf(p, "args", s, vst,
                        (s @ p["W_args"]) @ vst.transpose(-2, -1)),
        **({"dargs": (s @ p["W_dargs"]) @ vst.transpose(-2, -1)}
           if "W_dargs" in p else {}),
        **({"iargs": (s @ p["W_iargs"]) @ vst.transpose(-2, -1)}
           if "W_iargs" in p else {}),
        "res": _fed_pf(p, "res", s, vst,
                       (s @ p["W_res"]) @ vst.transpose(-2, -1)),
        **({"dig2": _fed_pf(p, "dig2", s, None,
                            s @ p["h_dig2"] + p["h_dig2_b"])
            .reshape(B, L_FAC, N_DIG, 10),
            "y": _fed_pf(p, "y", s, vst,
                         (s @ p["W_y"]) @ vst.transpose(-2, -1))}
           if "h_dig2" in p else {}),
    }


def _fact_inject(p, vst, fact_buf):
    """The ALT2 injection (one line, factored so the trainer's per-seam
    vst update calls the SAME organ — the meter-divergence law)."""
    return vst + (fact_buf @ p["W_fact"]) * p["alt2_g"].reshape(1, 1, 1)


def breath_step(p, state, kb, ctx):
    """THE BREATH STEP — forward()'s K-breath loop BODY, factored to
    module level BY PURE CODE MOTION (apply_step_trainer.py, 2026-09-03;
    the final-boss ruling). forward() calls this in its loop: behavior
    bit-identical by construction. The inner-step trainer
    (scripts/step_trainer.py) walks the same function one breath at a
    time with solver pings between dispatches.

    state — the TRUE cross-breath set (mutated in place, also returned):
      cur       (B, L_FAC, H_W) gradient-carrying slot state
      breaths   list; cur appended per breath (the ladder loss feed)
      nb, nb_st notebook shelf (ink list, GRADIENT-CARRYING) + stamps;
                born at kb == 1 (nb/nb_st enter as None)
      garage    deposit shelf (list; DETACHED under ALG_BUSGARAGE >= 2)
      snaps, snaps_g  lattice snap tuples (DETACHED); only [-1] is read
      rb_last   router bias (output-only)
      m_c, anchor, cmt_logits, x_rel  RINGS pawl state (None when off)
    ctx — per-forward constants: B, K_B, waist, tokmask, slot_mask,
      bank, rot2, sync, drop, gmod, revoke, tail, reg, RINGS, XOUT,
      XARM, XR_GRADED, XR_ELASTIC."""
    from tinygrad import Tensor
    global _CENSUS, _IMP
    try: _CENSUS
    except NameError: _CENSUS = None
    try: _IMP
    except NameError: _IMP = None
    B = ctx["B"]; K_B = ctx["K_B"]
    waist = ctx["waist"]; tokmask = ctx["tokmask"]
    slot_mask = ctx["slot_mask"]; bank = ctx["bank"]; _rot2 = ctx["rot2"]
    _sync = ctx["sync"]; drop = ctx["drop"]; gmod = ctx["gmod"]
    revoke = ctx["revoke"]; tail = ctx["tail"]; reg = ctx["reg"]
    RINGS = ctx["RINGS"]; XOUT = ctx["XOUT"]; XARM = ctx["XARM"]
    XR_GRADED = ctx["XR_GRADED"]; XR_ELASTIC = ctx["XR_ELASTIC"]
    cur = state["cur"]; breaths = state["breaths"]
    _nb = state["nb"]; _nb_st = state["nb_st"]
    _garage = state["garage"]; _snaps = state["snaps"]
    _snaps_g = state["snaps_g"]; _rb_last = state["rb_last"]
    m_c = state["m_c"]; anchor = state["anchor"]
    cmt_logits = state["cmt_logits"]; x_rel = state["x_rel"]
    if _IMP is not None and kb == _IMP[0]:
        cur = cur + _IMP[1]          # the kick
    if ALG_NOTEBOOK and kb == 1:
        from tinygrad import Tensor as _T2, dtypes as _dt2
        _nb_st = _T2(NB_STAMPS, dtype=_dt2.float)
        _nb = [(cur @ p["W_sil"]) if NB_PERSLOT
               else (_fed_core(cur).mean(1) @ p["W_sil"])]   # sharp vs blurred
        if FED_SHELF and "fed_sil2" in p:
            # FED item 9: lane 2 born at the same breath (2 rows per
            # breath — the structural coupling's lawful expansion)
            state["nb2"] = [(cur @ p["fed_sil2"]) if NB_PERSLOT
                            else (_fed_core(cur).mean(1) @ p["fed_sil2"])]
    q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1)
    if _CENSUS is not None:
        _CENSUS.append((kb, "state", cur.realize().numpy()))
        _CENSUS.append((kb, "breath_emb",
                        p["breath_emb"][kb].realize().numpy()
                        .reshape(1, 1, -1)))
    if ALG_NOTEBOOK:
        if NB_PERSLOT:      # per-slot lanes: each slot queries the
            _q = cur @ p["W_nq"]              # shelf and reads ITS OWN
            _sc = (_q @ _nb_st[:len(_nb)].transpose(1, 0)) / math.sqrt(H_W)
            if NB_FOCAL > 0:
                _sc = _sc * NB_FOCAL          # the magnifying glass
            _at = _sc.softmax(-1)             # (B, L, k)
            _rd = sum(_at[:, :, j:j + 1] * _nb[j] for j in range(len(_nb)))
            q_extra = q_extra + _rd           # (B, L, H) — no blur
            if _CENSUS is not None:
                _CENSUS.append((kb, "notebook", _rd.realize().numpy()))
        else:
            _q = _fed_core(cur).mean(1) @ p["W_nq"]
            _sc = (_q @ _nb_st[:len(_nb)].transpose(1, 0)) / math.sqrt(H_W)
            if NB_FOCAL > 0:
                _sc = _sc * NB_FOCAL          # the magnifying glass
            _at = _sc.softmax(-1)
            _rd = sum(_at[:, j:j + 1] * _nb[j] for j in range(len(_nb)))
            q_extra = q_extra + _rd.reshape(B, 1, -1)
        _nb2 = state.get("nb2")
        if FED_SHELF and "fed_sil2" in p and _nb2:
            # FED item 9: LANE-2 READ — stamps rows 8..8+k of the
            # 16-row alphabet (own address space), entering through
            # the ZERO-INIT gain door: birth bit-identical; the gain's
            # live grad wakes fed_sil2 (item-2 law). Same query _q as
            # lane 1 (per-slot or blurred, whichever branch ran).
            _sc2r = (_q @ _nb_st[8:8 + len(_nb2)].transpose(1, 0)) \
                / math.sqrt(H_W)
            if NB_FOCAL > 0:
                _sc2r = _sc2r * NB_FOCAL
            _at2 = _sc2r.softmax(-1)
            if NB_PERSLOT:
                _rd2 = sum(_at2[:, :, _j2:_j2 + 1] * _nb2[_j2]
                           for _j2 in range(len(_nb2)))
                q_extra = q_extra + _rd2 * p["fed_nb_g"].reshape(1, 1, 1)
            else:
                _rd2 = sum(_at2[:, _j2:_j2 + 1] * _nb2[_j2]
                           for _j2 in range(len(_nb2)))
                q_extra = q_extra + _rd2.reshape(B, 1, -1) \
                    * p["fed_nb_g"].reshape(1, 1, 1)
        if ALG_STELLAR:                        # cell-3b: the twist in
            _w = math.cos(kb * math.pi / (2 * K_B)) ** 2   # geometry —
            cur = _w * cur + (1 - _w) * _rd.reshape(B, 1, -1)
            q_extra = cur + p["breath_emb"][kb].reshape(1, 1, -1) + _rd.reshape(B, 1, -1)
                                                  # no cliff, no gate
        if ALG_CIRCLE and kb == NB_H + 1:     # the traffic circle:
            cur = cur * 0.0 + _rd.reshape(B, 1, -1)   # residual severed
            q_extra = (cur + p["breath_emb"][kb].reshape(1, 1, -1)
                       + _rd.reshape(B, 1, -1))       # memory the road
    if _garage is not None and len(_garage) > 0:
        # GARAGE READ (drop-off): content-addressed attention over
        # the deposit shelf — the reader needs NO tick knowledge;
        # then per-role conj unbind of the retrieved wire
        _gq4 = cur @ p["W_gq"]
        _sc4 = Tensor.cat(*[(_gq4 * _d4).sum(-1, keepdim=True)
                            / math.sqrt(float(_gq4.shape[-1]))
                            for _d4 in _garage], dim=-1)
        _at4 = _sc4.softmax(-1)
        _rd4 = sum(_at4[:, :, _j4:_j4 + 1] * _garage[_j4]
                   for _j4 in range(len(_garage)))
        _rds4 = [_rot2(_rd4, _rc4, _rs4)
                 for (_rc4, _rs4) in _SGC[0].values()]
        _inj4 = Tensor.cat(*_rds4, dim=-1) @ p["W_busr"]
        if _CENSUS is not None:
            _CENSUS.append((kb, "garage",
                            (_inj4 * p["bus_g"].reshape(1, 1, 1))
                            .realize().numpy()))
        _scm = int(os.environ.get("ALG_SHELF_CIRCLE", "0"))
        if _scm and kb == int(os.environ.get("SC_KB", "4")):
            # THE PRESSURE COOKER (2026-08-30, word given):
            # residual SEVERED at this breath — committed facts
            # are the only road across. Ungated, full gradient.
            # mode 1 = constant seal (the boundary condition);
            # mode 2 = THE PULSE (per-step Bernoulli seal via the
            # _SEV data buffer — MASK_GOLD idiom; SC_EVAL forces
            # a mode at read time; sealed-mode val IS the
            # capability meter).
            _cur_seal = cur * 0.0 + _inj4
            _q_seal = _cur_seal + p["breath_emb"][kb].reshape(1, 1, -1)
            _q_open = q_extra + _inj4 * p["bus_g"].reshape(1, 1, 1)
            _pcv4 = (globals().get("_PCV")
                     if float(os.environ.get("ALG_PC_MIX", "0")) > 0.0
                     and not os.environ.get("SC_EVAL", "") else None)
            if _pcv4 is not None:
                # THE PRESSURE MIX (apply_pressure_mix.py, 2026-09-06):
                # per-row mixed seal — _PCV is a (B,1,1) data buffer
                # (the _SEV/MASK_GOLD idiom: one JIT graph, dynamic
                # value). 1.0 rows get the mode-1 constant severance
                # (residual dead, shelf crossing the only road), 0.0
                # rows the open path — same blend arithmetic as the
                # SC_EVAL forms, per row (exact at v in {0,1}: 1.0*x =
                # x, 0.0*finite = 0, x+0 = x). The trainer assigns it
                # per step from the STABLE index-hash assignment (flat
                # mix, never re-rolled); _quick_val is excluded by the
                # existing SC_EVAL="0" push (val compares OPEN mode).
                _pvt4 = _pcv4.reshape(-1, 1, 1)
                cur = cur * (1.0 - _pvt4) + _cur_seal * _pvt4
                q_extra = _q_open * (1.0 - _pvt4) + _q_seal * _pvt4
            elif _scm >= 2:
                _sce = os.environ.get("SC_EVAL", "")
                if _sce:
                    _sv = float(_sce)
                    cur = cur * (1.0 - _sv) + _cur_seal * _sv
                    q_extra = _q_open * (1.0 - _sv) + _q_seal * _sv
                else:
                    global _SEV
                    try: _SEV
                    except NameError: _SEV = None
                    if _SEV is None:
                        _SEV = Tensor([1.0]).contiguous().realize()
                    _svt = _SEV.reshape(1, 1, 1)
                    cur = cur * (1.0 - _svt) + _cur_seal * _svt
                    q_extra = _q_open * (1.0 - _svt) + _q_seal * _svt
            else:
                cur = _cur_seal
                q_extra = _q_seal
        else:
            q_extra = q_extra + _inj4 * p["bus_g"].reshape(1, 1, 1)
    if _snaps and "W_det" in p:
        # THE 2-OF-3 FIELD (the ladder era, 2026-08-31): true
        # forced moves — two determined roles force the third,
        # forward ladder AND inverse anchor alike; seed at
        # givens, 3 sweeps; features [ndet/3, res-det, fires]
        _a6, _b6, _sr6, _gv6 = _snaps[-1]
        _det6 = (_gv6.unsqueeze(-1) * _sr6).max(1)          # (B,24)
        for _ in range(3):
            _r16 = (_a6 @ _det6.unsqueeze(-1)).squeeze(-1).clip(0, 1)
            _r26 = (_b6 @ _det6.unsqueeze(-1)).squeeze(-1).clip(0, 1)
            _r36 = (_sr6 @ _det6.unsqueeze(-1)).squeeze(-1).clip(0, 1)
            _nd6 = _r16 + _r26 + _r36
            _fi6 = ((_nd6 >= 2).float()
                    * (1.0 - _gv6))                          # (B,L)
            _new6 = None
            for _oh6, _rr6 in ((_a6, _r16), (_b6, _r26), (_sr6, _r36)):
                _c6 = ((_fi6 * (1.0 - _rr6)).unsqueeze(-1)
                       * _oh6).max(1)
                _new6 = _c6 if _new6 is None else _new6 + _c6
            _det6 = (_det6 + _new6).clip(0, 1)
        _fe6 = Tensor.stack((_nd6 / 3.0).clip(0, 1), _r36, _fi6,
                            dim=-1)                          # (B,L,3)
        _dinj = (_fe6 @ p["W_det"]) * p["det_g"].reshape(1, 1, 1)
        q_extra = q_extra + _dinj
        if _CENSUS is not None:
            _CENSUS.append((kb, "detwave", _dinj.realize().numpy()))
    if _sync is not None:   # sync-complete: transmitter ON during
        q_extra = q_extra + _sync[1](kb)     # settle; receiver locked
    _rb7 = None
    if "W_ra" in p:
        if _snaps:
            _src7 = (_snaps_g[-1] if (_snaps_g and
                     int(os.environ.get("ALG_ROUTER_GRADED", "0")))
                     else _snaps[-1])
            _sf7 = Tensor.cat(_src7[0], _src7[1],
                              _src7[2],
                              _src7[3].unsqueeze(-1), dim=-1)
            _cq7 = cur + _sf7 @ p["W_rs"]
        else:
            _cq7 = cur
        _rb7 = ((_cq7 @ p["W_ra"])
                @ (waist @ p["W_rb"]).transpose(-2, -1)) / 8.0
        _rb_last = _rb7
        if _CENSUS is not None:
            _CENSUS.append((kb, "router(bank)",
                            (_rb7 * p["r_gain"].reshape(1, 1, 1))
                            .realize().numpy()))
    h_tok, fat_cur = bank(p["fq"], L_TOT, extra=q_extra,
                          pbias=(_sync[0](kb) if _sync is not None
                                 else None),
                          rbias=_rb7)
    if int(os.environ.get("ALG_MINE_BREATHS", "0")):
        # NL TAP (apply_nl_tap.py, 2026-09-05, the paired atlas):
        # read-only capture of this breath's READING — head-avg
        # token attention, slot-averaged to one distribution,
        # pooling the SAME waist the bank read (attention-weighted
        # mean over tokens -> (B, H_W)). Lazy tensors on state;
        # realized only by miners/readers (the _CENSUS discipline:
        # inert unless armed; training never sets this env).
        _nlw = _fed_core(fat_cur).mean(1)            # (B, T) read
        state.setdefault("nl_all", []).append(
            (_nlw.unsqueeze(1) @ waist).squeeze(1))  # (B, H_W)
        state.setdefault("nlat_all", []).append(_nlw)
    bq = cur @ p["W_bq"] + p["W_bq_b"]
    bk = cur @ p["W_bk"] + p["W_bk_b"]
    bv = cur @ p["W_bv"] + p["W_bv_b"]
    _bq2 = bq
    if ALG_POLAR and POLAR_QROT >= 2 and 1 <= kb <= _RC_N_LOOP:
        # THE SEXTET ON THE ATTENTION SPACE (spec S1.2; lead's ruling
        # 2026-09-07). The mixer's rotation rides behind fed_mx_hg
        # gains that woke to 0.023 and SHRANK to 0.013 under the cooker
        # — a whisper. The slot mixer's OWN queries are where the
        # attention actually speaks, so they turn here on the SAME 256
        # planes, by the SAME rotor_clock wheel table, ABSOLUTE angles
        # (the query is rebuilt from cur each breath: nothing
        # compounds), K UNROTATED (the v109pi relative-phase
        # precedent), no gains and no learnable rate. It rides its OWN
        # tensor: bq itself stays untouched, so the mixer below builds
        # _mx_q from the unrotated queries and applies its own turn —
        # each attention is rotated exactly ONCE at QROT=2.
        from tinygrad import Tensor as _Tm, dtypes as _dm
        _mdc, _mds, _mac, _mas, _mwof = _polar_tables()
        _bq2 = _rot2(bq,
                     _Tm(_mac[kb - 1], dtype=_dm.float),
                     _Tm(_mas[kb - 1], dtype=_dm.float))
    sc2 = (_bq2 @ bk.transpose(-2, -1)) / math.sqrt(H_W)
    _sm_kb = slot_mask
    _A5 = None
    if _snaps and ("alt_g" in p
                   or int(os.environ.get("ALG_MASKRE", "0"))):
        _sa5 = _snaps[-1][0] + _snaps[-1][1]
        _sr5 = _snaps[-1][2]
        _A5 = _sr5 @ _sa5.transpose(-2, -1)
        if int(os.environ.get("ALG_MASKRE", "0")):
            # v2 THE MASK RE-FORMATION (2026-09-01, word given):
            # the HARD mask rebuilt per breath — OPEN-BY-
            # COMMITMENT (committed producer->consumer edges may
            # attend across the first-pass mask; additive-optional
            # per the ensemble law; NEVER tightens — A0's grave
            # stays honored)
            _sm_kb = (slot_mask
                      + ((_A5 + _A5.transpose(-2, -1)) > 0.5)
                      .float()).clip(0, 1)
    _mb = None
    if int(os.environ.get("ALG_MASKHEAD", "0")) and "mh_wo" in p:
        # THE MASK HEAD (2026-09-05): the learned precision channel at
        # the RELATE seam. Reads what the >0.5 reflex throws away — the
        # GRADED _A5 (confidences), solver fact_buf, the previous
        # breath's adjacency (state storage), breath phase, and the
        # domain-mass / atlas-page ports — and emits a soft OPEN-ONLY
        # bias over the slot mixer. ALL metadata enters DETACHED (the
        # dual-terminal law); the live terminal is cur (queries + kv
        # stream). NO mask loss exists anywhere — trained ONLY by
        # downstream parse CE through sc2 -> softmax -> h_slot -> cur
        # -> emissions (Goodhart fence: a supervised mask teaches
        # concealment; assert_not_supervised in spirit — no mask
        # signal enters any loss, ever). EQUIVALENCE AT BIRTH: mh_wo
        # and mh_headmix are ZERO-INIT, so _raw == 0 everywhere and
        # _mb = gain*(softplus(_raw) - softplus(_raw*0))*open == exact
        # zeros (identical kernels cancel bitwise). OPEN-ONLY: _mb is
        # gated to the already-open region _sm_kb (committed MASKRE
        # edges included) and bounded below by -gain*ln2 (the birth-
        # plateau reference) — the -1e4 close and the base-mask
        # SUPPORT are untouchable (A0's grave honored); a bounded
        # signed bias within the open region is the alt_g precedent.
        _z1 = (cur[:, :, :1] * 0.0).detach()
        if _A5 is not None:
            _A5s = (_A5 + _A5.transpose(-2, -1)).detach()
            _mh_a = _snaps[-1][0]           # detached snap one-hots
            _mh_b = _snaps[-1][1]
            _mh_r = _snaps[-1][2]
            _mh_g = _snaps[-1][3].unsqueeze(-1)
            _mh_row = _A5s.mean(-1, keepdim=True)
            _mh_col = _A5s.transpose(-2, -1).mean(-1, keepdim=True)
        else:
            _A5s = None
            _mh_a = _mh_b = _mh_r = None
            _mh_g = _z1; _mh_row = _z1; _mh_col = _z1
        _mh_f = ctx.get("fact_buf")         # (B, K_VARS, 4) solver
        if _mh_f is not None and _mh_a is not None:   # facts, detached
            _mh_ff = Tensor.cat(_mh_a @ _mh_f, _mh_b @ _mh_f,
                                _mh_r @ _mh_f, dim=-1)   # (B, L, 12):
            # what the solver knows about MY args and MY result
        else:
            _mh_ff = Tensor.cat(*([_z1] * 12), dim=-1)
        # DOMAIN-MASS PORT (documented, 2026-09-05): (B, K_VARS, 1)
        # per-var matryoshka radius (alternator_bridge.ping returns
        # mass; not yet threaded into the fused graph — only fact_buf
        # is in-graph today). A seam driver may set ctx["mh_mass"];
        # absent -> zeros, graph shape unchanged, grads stay defined.
        _mh_m = ctx.get("mh_mass")
        if _mh_m is not None and _mh_a is not None:
            _mh_fm = Tensor.cat(_mh_a @ _mh_m, _mh_b @ _mh_m,
                                _mh_r @ _mh_m, dim=-1)   # (B, L, 3)
        else:
            _mh_fm = Tensor.cat(*([_z1] * 3), dim=-1)
        _mh_p = state.get("mh_prev")        # STORAGE READ: the organ
        if _mh_p is not None:               # sees the commitment FLOW
            _mh_pr = _mh_p.mean(-1, keepdim=True)
            _mh_pc = _mh_p.transpose(-2, -1).mean(-1, keepdim=True)
        else:
            _mh_pr = _z1; _mh_pc = _z1
        _mh_bs = _z1 + math.sin(kb * math.pi / 3.0)   # breath phase
        _mh_bc = _z1 + math.cos(kb * math.pi / 3.0)   # (60-deg clock)
        _mh_cf = Tensor.cat(_mh_ff, _mh_fm, _mh_g, _mh_row, _mh_col,
                            _mh_pr, _mh_pc, _mh_bs, _mh_bc,
                            dim=-1)          # (B, L, MH_CTX_F) DETACHED
        _mh_ce = ((_mh_cf @ p["mh_enc1"] + p["mh_enc1_b"]).gelu()
                  @ p["mh_enc2"] + p["mh_enc2_b"])    # (B, L, H_W)
        # ATLAS-PAGE PORT (documented, 2026-09-05): (B, H_W) or
        # (B, L, H_W) detached page(s) from mycelium/step_atlas.consult
        # at a seam (the fused loop cannot consult mid-graph — consult
        # is numpy); a seam driver may set ctx["mh_atlas"]; absent ->
        # zeros from cur*0 keep mh_atlas_w in-graph (defined zero
        # grads — the None-grad law; degrade gracefully).
        _mh_ap = ctx.get("mh_atlas")
        if _mh_ap is None and ctx.get("mh_atlas_traj") is not None:
            # ATLAS TRAJECTORY PORT (apply_mass_thread.py,
            # 2026-09-05): (B, K_STEPS, H_W) per-row class pages;
            # kb is a python int (the breath loop is unrolled) so
            # this slice is static per jitted step. Page kb feeds
            # breath kb (page 0 = intake, never consumed here —
            # breath_step runs kb>=1). Consult-by-similarity is
            # the read-time upgrade (seam drivers set "mh_atlas").
            _mh_ap = ctx["mh_atlas_traj"][:, kb:kb + 1, :]
        if _mh_ap is None:
            _mh_ap = (cur * 0.0).detach()
        _mh_ce = _mh_ce + _mh_ap.reshape(B, -1, H_W) @ p["mh_atlas_w"]
        _mh_nl = ctx.get("fed_nl0")
        if _mh_nl is not None and "fed_nl0_w" in p:
            # FED item 8: the breath-0 invariant page through its ZERO
            # door — exact zeros at birth, live grads on fed_nl0_w
            _mh_ce = _mh_ce + (_mh_nl.reshape(B, 1, H_W)
                               @ p["fed_nl0_w"])
        _mh_kv = cur + _mh_ce      # LIVE stream + detached context
        _mh_q = cur @ p["mh_wq"] + p["mh_wq_b"]
        _mh_k = _mh_kv @ p["mh_wk"] + p["mh_wk_b"]
        _mh_v = _mh_kv @ p["mh_wv"] + p["mh_wv_b"]
        _mh_hd = H_W // MH_HEADS
        _mh_qh = _mh_q.reshape(B, L_TOT, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)
        _mh_kh = _mh_k.reshape(B, L_TOT, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)
        _mh_vh = _mh_v.reshape(B, L_TOT, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)
        _mh_sc = ((_mh_qh @ _mh_kh.transpose(-2, -1))
                  / math.sqrt(_mh_hd)).clip(-1e4, 1e4)   # (B, M, L, L)
        _mh_at = (_mh_sc
                  + (1.0 - _sm_kb.unsqueeze(1)) * -1e4).softmax(-1)
        _mh_gt = (_mh_at @ _mh_vh).permute(0, 2, 1, 3) \
            .reshape(B, L_TOT, H_W)
        _mh_u = (_mh_gt @ p["mh_wu"] + p["mh_wu_b"]).gelu()
        _mh_o = _mh_u @ p["mh_wo"] + p["mh_wo_b"]     # ZERO door 1
        _mh_rp = (_mh_o @ (_mh_kv @ p["mh_wp"]).transpose(-2, -1)) \
            / math.sqrt(H_W)                # value-informed pair logits
        _mh_rh = (_mh_sc * p["mh_headmix"].reshape(1, MH_HEADS, 1, 1)) \
            .sum(1)                         # ZERO door 2: direct head
        _raw = (_mh_rp + _mh_rh).clip(-30.0, 30.0)    # finite softplus
        _mh_sp = (1.0 + _raw.exp()).log()             # softplus(raw)
        _mh_sp0 = (1.0 + (_raw * 0.0).exp()).log()    # birth plateau
        _mb = (p["mh_gain"].reshape(1, 1, 1)
               * (_mh_sp - _mh_sp0) * _sm_kb)
        if _A5s is not None:                # STORAGE WRITE: this
            state["mh_prev"] = _A5s         # breath's consumed
                                            # adjacency, detached
        sc2 = sc2 + _mb        # the injection site: BEFORE the close
    sc2 = sc2.clip(-1e4, 1e4) + (1.0 - _sm_kb) * -1e4
    if _A5 is not None and "alt_g" in p:
        # v0 soft bias rides alongside (facts wire attention)
        sc2 = sc2 + (_A5 + _A5.transpose(-2, -1)) \
            * p["alt_g"].reshape(1, 1, 1)
    if RINGS and int(os.environ.get("ALG_BEXIT", "0")):
        # BEAM EXIT (door #8): committed slots leave the mixer as
        # keys, proportional to mass — soft, init-closed (m starts 0)
        sc2 = sc2 + m_c.reshape(B, 1, L_FAC) * -8.0
    h_slot = (sc2.softmax(-1) @ bv) @ p["W_bo"] + p["W_bo_b"]
    if FED_MIXER and "fed_mx_hg" in p:
        # FED item 1: MIXER MULTI-HEAD — the twin-kernel form (chosen,
        # not fallback: a score-level combine keeps ONE softmax = one
        # geometry; per-head DISTRIBUTIONS are the multi-head win).
        # The SAME W_bq/W_bk/W_bv reshaped into MX_HEADS heads (free
        # reinterpretation — warm-load keys unchanged); the same
        # mask-head bias, close, and alt bias stack as sc2; outputs
        # gated by ZERO-INIT per-head gains, spoken through the
        # TRAINED W_bo (no second bias). At zero gains the twin term
        # is exact zeros -> birth bitwise = the single-head path;
        # dL/dg_h = <dL/dh_slot @ W_bo^T, head_h> != 0 at WARM birth
        # (port242's W_bo is trained/nonzero — measured ALIVE 1.6e-1
        # in the warm-sim grad smoke). COLD-init caveat, stated
        # honestly: build_params starts W_bo at zeros, so gains' grads
        # are zero-defined for exactly as long as W_bo itself is zero
        # (W_bo moves at step 1 via the base path; gains wake step 2 —
        # no deadlock; the fed mind is a warm-continuation package by
        # charter, so the warm case is the deployed case).
        # (BEXIT's -8 soft exit is not mirrored: door #8 is off in the
        # champion family; documented deferral.)
        _mx_hd = H_W // MX_HEADS
        _mx_q = bq.reshape(B, L_TOT, MX_HEADS, _mx_hd).permute(0, 2, 1, 3)
        _mx_k = bk.reshape(B, L_TOT, MX_HEADS, _mx_hd).permute(0, 2, 1, 3)
        _mx_v = bv.reshape(B, L_TOT, MX_HEADS, _mx_hd).permute(0, 2, 1, 3)
        if ALG_POLAR and POLAR_QROT >= 1 and 1 <= kb <= _RC_N_LOOP:
            # THE SEXTET, Q-SIDE (spec S1.2): the SAME plane allocation
            # as the state's clock, reshaped (MX_HEADS, pairs/head) —
            # state and attention are ONE clock, not two. ABSOLUTE
            # angles here (the query is rebuilt from cur every breath:
            # nothing compounds), K UNROTATED (the v109pi precedent:
            # one table on both sides cancels — relative phase is the
            # signal). UNCONDITIONAL: no gains, no learnable rate. The
            # mixer builds _mx_q from the UNROTATED bq (the main path's
            # turn rides its own tensor, _bq2), so QROT=2 rotates each
            # attention ONCE — never twice. This REPLACES fed item 7a,
            # whose 60 deg on 8 of 32 pairs sat
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
                                 _qx8 * _rsq + _qy8 * _rcq, dim=-1) \
                .reshape(B, MX_HEADS, L_TOT, _mx_hd)
        elif FED_ROTOR and _FED_ROT_C is not None and 1 <= kb <= 6:
            # FED item 7a: THE BREATH ROTOR INSTALLS HERE —
            # 60deg/breath sextet rotation (rotor_clock's legacy band,
            # pairs 24..31 of each 64d head), Q-SIDE ONLY (the v109pi
            # precedent: one table on both sides cancels — relative
            # phase is the signal). Behind the zero gains, so birth
            # equivalence is free. kb -> tick kb-1 (breath-0 is
            # outside time, phase_of's contract; kb > 6 unclocked).
            from tinygrad import Tensor as _T7, dtypes as _d7
            _rc7 = _T7(_FED_ROT_C[kb - 1], dtype=_d7.float) \
                .reshape(1, 1, 1, -1)
            _rs7 = _T7(_FED_ROT_S[kb - 1], dtype=_d7.float) \
                .reshape(1, 1, 1, -1)
            _qp7 = _mx_q.reshape(B, MX_HEADS, L_TOT, _mx_hd // 2, 2)
            _qx7, _qy7 = _qp7[..., 0], _qp7[..., 1]
            _mx_q = Tensor.stack(_qx7 * _rc7 - _qy7 * _rs7,
                                 _qx7 * _rs7 + _qy7 * _rc7, dim=-1) \
                .reshape(B, MX_HEADS, L_TOT, _mx_hd)
        _mx_sc = (_mx_q @ _mx_k.transpose(-2, -1)) / math.sqrt(_mx_hd)
        if _mb is not None:                 # the same mask-head bias
            _mx_sc = _mx_sc + _mb.unsqueeze(1)
        _sm_tw = _sm_kb
        if N_SCR:
            # FED item 6 read-back: scratch COLUMNS open ONLY here —
            # behind the zero gains (the raising law's route)
            _sm_tw = Tensor.cat(_sm_tw[:, :, :L_FAC],
                                _sm_tw[:, :, L_FAC:] * 0.0 + 1.0,
                                dim=2)
        _mx_sc = (_mx_sc.clip(-1e4, 1e4)
                  + (1.0 - _sm_tw.unsqueeze(1)) * -1e4)
        if _A5 is not None and "alt_g" in p:   # same v0 bias as sc2
            _mx_sc = _mx_sc + ((_A5 + _A5.transpose(-2, -1))
                               * p["alt_g"].reshape(1, 1, 1)).unsqueeze(1)
        _mx_o = (_mx_sc.softmax(-1) @ _mx_v) \
            * p["fed_mx_hg"].reshape(1, MX_HEADS, 1, 1)   # ZERO gains
        h_slot = h_slot + _mx_o.permute(0, 2, 1, 3) \
            .reshape(B, L_TOT, H_W) @ p["W_bo"]
    # ABLATION arms (2026-07-10): zero-mult keeps every param in the
    # graph (defined zero grads — the None-grad lesson, applied)
    arm = os.environ.get("ALG_BREATH_ARM", "both")
    if arm == "tok":
        h_slot = h_slot * 0.0
    elif arm == "slot":
        h_tok = h_tok * 0.0
    elif arm == "depth":
        # the decider control: plain per-slot MLP second pass — same
        # params repurposed, NO attention, no mask, no re-read
        h_tok = h_tok * 0.0
        h_slot = h_slot * 0.0 + ((cur @ p["W_bq"] + p["W_bq_b"])
                                 .gelu() @ p["W_bv"] + p["W_bv_b"]) \
            @ p["W_bo"] + p["W_bo_b"]
    if int(os.environ.get("ALG_ALT21", "0")) and "alt21_W_bo" in p:
        # ALTERNATOR v2.1 STATIONS 3-4 (2026-09-02): the INTEGRATE
        # pair, between GATHER+RELATE above and the gate/commit
        # below. Each block writes ADDITIVELY through its ZERO-INIT
        # output projection, identity path untouched
        # (out = out + block(out)). EQUIVALENCE BY CONSTRUCTION:
        # at init alt21_attn_wo and alt21_W_bo are zeros, so
        # _d21a = _d21b = exact zeros and h_slot (hence forward)
        # equals baseline exactly at birth; env unset skips the
        # whole block (the chain verifies via eq_check pre/post).
        _s21 = h_tok + h_slot            # the stream after 1-2
        # STATION 3: second bank-attention (slots<-tokens), live
        # query = slot codes + stream + this breath's conditioning
        _qx21 = p["fq"].unsqueeze(0) + _s21 + (q_extra - cur)
        _q21 = _qx21 @ p["alt21_attn_wq"] + p["alt21_attn_wq_b"]
        _k21 = waist @ p["alt21_attn_wk"] + p["alt21_attn_wk_b"]
        _v21 = waist @ p["alt21_attn_wv"] + p["alt21_attn_wv_b"]
        _hd21 = H_W // N_HEADS
        _qh21 = _q21.reshape(B, L_TOT, N_HEADS, _hd21).permute(0, 2, 1, 3)
        _kh21 = _k21.reshape(B, -1, N_HEADS, _hd21).permute(0, 2, 1, 3)
        _vh21 = _v21.reshape(B, -1, N_HEADS, _hd21).permute(0, 2, 1, 3)
        _sa21 = (_qh21 @ _kh21.transpose(-2, -1)) / math.sqrt(_hd21)
        if _sync is not None:            # the same breath rotation
            _sa21 = _sa21 + _sync[0](kb)
        if _rb7 is not None:             # the same router bias
            _sa21 = _sa21 + _rb7.unsqueeze(1) * p["r_gain"].reshape(1, 1, 1, 1)
        _sa21 = _sa21.clip(-1e4, 1e4) + (1.0 - tokmask.reshape(B, 1, 1, -1)) * -1e4
        _st21 = (_sa21.softmax(-1) @ _vh21).permute(0, 2, 1, 3).reshape(B, L_TOT, H_W)
        _d21a = _st21 @ p["alt21_attn_wo"] + p["alt21_attn_wo_b"]
        _s21 = _s21 + _d21a              # exact zero at birth
        # STATION 4: second slot-mixer over the SAME breathed mask
        _bq21 = _s21 @ p["alt21_W_bq"] + p["alt21_W_bq_b"]
        _bk21 = _s21 @ p["alt21_W_bk"] + p["alt21_W_bk_b"]
        _bv21 = _s21 @ p["alt21_W_bv"] + p["alt21_W_bv_b"]
        _sm21 = (_bq21 @ _bk21.transpose(-2, -1)) / math.sqrt(H_W)
        if _mb is not None:        # MASK HEAD: the same open-only
            _sm21 = _sm21 + _mb    # bias, station-4 mixer (before
                                   # ITS close — one geometry/breath)
        _sm21 = _sm21.clip(-1e4, 1e4) + (1.0 - _sm_kb) * -1e4
        if _A5 is not None and "alt_g" in p:   # v0 bias, as station 2
            _sm21 = _sm21 + (_A5 + _A5.transpose(-2, -1)) \
                * p["alt_g"].reshape(1, 1, 1)
        if RINGS and int(os.environ.get("ALG_BEXIT", "0")):
            _sm21 = _sm21 + m_c.reshape(B, 1, L_FAC) * -8.0
        _d21b = (_sm21.softmax(-1) @ _bv21) @ p["alt21_W_bo"] \
            + p["alt21_W_bo_b"]
        h_slot = h_slot + _d21a + _d21b  # additive; zeros at birth
    g = p["breath_gate"][kb].sigmoid()
    if drop is not None:            # door #52: BREATH DROPOUT —
        g = g * drop                # per-STEP coin; drop=0 makes the
                                    # breath an exact identity (silent)
    if gmod is not None:            # NAZARE (B)-site smoke: per-slot
        g = g * gmod                # authority INSIDE the loop
    cur_new = cur + g * (h_tok + h_slot - cur)
    if RINGS:  # the soft pawl: monotone commitment mass + anchor
        cl = (cur_new @ p["W_cmt"] + p["W_cmt_b"])     # (B,L_FAC,1)
        if reg is not None and "w_cmt_reg" in p:
            # REGISTER-AWARE COMMITMENT (2026-08-26, word given):
            # the mouth's length-corrected read as the pawl's INPUT
            # — commit boldly in-register, reluctantly on the
            # frontier. The mouth stays out of every loss (Goodhart
            # fence); the pawl is handed the map, not the meter.
            cl = cl + reg.reshape(B, 1, 1) * p["w_cmt_reg"]
        cmt_logits.append(cl.squeeze(-1))
        if XOUT:  # release BEFORE the pawl: the same-breath commit
            # pressure is the graded arm's RESISTING term (#150)
            rel = m_c * 0.0
            if XARM == "elastic":            # standing leak toward
                rel = rel + XR_ELASTIC * m_c  # rest; self-resetting
            if revoke is not None:
                rv = revoke.reshape(B, L_FAC, 1)
                if XARM == "dump":
                    rel = rel + m_c * rv      # instant
                else:                         # graded|elastic
                    rel = rel + XR_GRADED * m_c * rv
            rel = rel.minimum(m_c)            # never below zero mass
            m_c = m_c - rel
            x_rel = x_rel + rel
        dm = (1.0 - m_c) * cl.sigmoid()
        if int(os.environ.get("ALG_CLOCK", "0")) and tail is not None:
            # CLOCK v1 (door #10): commit gated on OWN-SENTENCE
            # COMPLETION — attention mass on sentence-tail tokens
            c_j = (fat_cur * tail.reshape(B, 1, -1)).sum(-1, keepdim=True)                         / (fat_cur.sum(-1, keepdim=True) + 1e-6)
            _fl = float(os.environ.get("ALG_CLOCK_FLOOR", "0"))
            dm = dm * (_fl + (1.0 - _fl) * c_j)
        anchor = (m_c * anchor + dm * cur_new) / (m_c + dm + 1e-6)
        m_c = m_c + dm
        cur = m_c * anchor + (1.0 - m_c) * cur_new
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
        if POLAR_EM:
            # (B) THE E&B COUPLING (apply_polar_sink.py, 2026-09-08).
            # AFTER the sextet's turn, BEFORE the content waist: one
            # discrete Maxwell-like exchange step on the CLOCK planes
            # along the lanes of `_sm_kb` — the slot mask THIS breath's
            # own attention closes sc2 with (MASKRE re-formation
            # included), reused rather than rebuilt. kappa is fixed and
            # unlearnable. Content planes pass through bitwise; the
            # clock block's norm is restored inside the organ.
            _pol_u = _polar_em(_pol_u, _sm_kb, POLAR_EM)
        if POLAR_D:
            # (A) THE CONTENT-PLANE WAIST ("expand & collapse x7"). The
            # 192 content planes' 384 dims collapse to ALG_POLAR_D and
            # expand back, born as the champion's own top-d content
            # subspace. The 128 clock dims are in neither the domain nor
            # the range — the bottleneck cannot fight the rotation — and
            # the content block's norm is restored inside the organ, so
            # ||u|| == 1 still holds and r is untouched.
            _pol_u = _polar_waist(_pol_u, p, state)
        cur = _polar_ru_join(_pol_u, _pol_r, _pg)
        if int(os.environ.get("ALG_MINE_BREATHS", "0")):
            # THE TAP (spec S4): u and r beside r*u, DETACHED — the
            # clock read probes the direction without the radius
            # confound, and no diagnostic can teach the radius
            # (the Goodhart fence; the two-terminal proof is
            # scripts/polar_birth_smoke.py item 5).
            state.setdefault("u_all", []).append(_pol_u.detach())
            state.setdefault("r_all", []).append(_pol_r.detach())
    if ALG_NOTEBOOK:
        _nb.append((cur @ p["W_sil"]) if NB_PERSLOT
                   else (_fed_core(cur).mean(1) @ p["W_sil"]))
        if FED_SHELF and "fed_sil2" in p and state.get("nb2") is not None:
            state["nb2"].append((cur @ p["fed_sil2"]) if NB_PERSLOT
                                else (_fed_core(cur).mean(1)
                                      @ p["fed_sil2"]))
    breaths.append(cur)
    if _garage is not None:
        # GARAGE WRITE (drop-off): the refined state's role-bound
        # wire; the list IS the parking separation
        _wg4 = ((cur @ p["W_bind1"] + p["W_bind1_b"]).gelu()
                @ p["W_bind2"])
        if int(os.environ.get("ALG_BUSGARAGE", "0")) >= 2:
            # THE CANONICAL SHELF (2026-08-30, word given): snap to
            # the lattice — per-role cleanup, re-bind the cleaned
            # codes BY CONSTRUCTION; the deposit is a FACT
            # (detached, gradient-free) carrying the pre-snap
            # magnitude as its confidence stamp (wrong wires run
            # quiet — the measured confession de-weights them in
            # the shelf's own attention). Two jaws, in the loop.
            _cj4, _pl4, _CBt4 = _SGC
            _canon4 = None
            _snap_a5 = _snap_b5 = _snap_r5 = None
            _snap_g5 = None
            for _rn4 in ("arg1", "arg2", "res", "op"):
                _zc4 = _rot2(_wg4, *_cj4[_rn4])
                _lg4 = _zc4 @ _CBt4.T
                _oh4 = (_lg4 == _lg4.max(-1, keepdim=True)).float()
                _oh4 = _oh4 / (_oh4.sum(-1, keepdim=True) + 1e-9)
                if _rn4 == "arg1":
                    _snap_a5 = _oh4[..., :24]
                elif _rn4 == "arg2":
                    _snap_b5 = _oh4[..., :24]
                elif _rn4 == "res":
                    _snap_r5 = _oh4[..., :24]
                elif _rn4 == "op":
                    _snap_g5 = _oh4[..., 25]   # ftype 'given' code
                if int(os.environ.get("ALG_ROUTER_GRADED", "0")):
                    _sg4 = _lg4.softmax(-1)   # GRADED: the
                    # confidence distribution the argmax throws away
                    if _rn4 == "arg1": _grad_a5 = _sg4[..., :24]
                    elif _rn4 == "arg2": _grad_b5 = _sg4[..., :24]
                    elif _rn4 == "res": _grad_r5 = _sg4[..., :24]
                    elif _rn4 == "op": _grad_g5 = _sg4[..., 25]
                _cb4 = _rot2(_oh4 @ _CBt4, *_pl4[_rn4])
                _canon4 = _cb4 if _canon4 is None else _canon4 + _cb4
            if "alt_g" in p or "W_det" in p:
                _snaps.append((_snap_a5.detach(), _snap_b5.detach(),
                               _snap_r5.detach(), _snap_g5.detach()))
                if int(os.environ.get("ALG_ROUTER_GRADED", "0")):
                    _snaps_g.append((_grad_a5.detach(),
                        _grad_b5.detach(), _grad_r5.detach(),
                        _grad_g5.detach()))
            # THE CONFIDENCE STAMP, NaN-SAFE (2026-09-07, the dose-0.15 death):
            # tinygrad's sqrt backward is grad/(2*sqrt(x)); an exact-zero
            # deposit vector (the live wire shrinking a useless commitment to
            # float32 zero) made this 0/0 -> NaN at step ~5000. The +1e-6 sat
            # OUTSIDE the sqrt and guarded nothing. where()-gate (the CLAUDE.md
            # idiom): forward BIT-IDENTICAL (sqrt(s)+1e-6 for s>0; 1e-6 at s=0),
            # backward finite (zero into the s=0 branch).
            _ss4 = _wg4.pow(2).sum(-1, keepdim=True)
            _pcf4 = float(os.environ.get("ALG_PC_FLOOR", "0"))
            if _pcf4 > 0.0:
                # THE SOFT FLOOR (2026-09-07, word given; fix A for the
                # amplitude collapse): stamp = sqrt(||w||^2 + f^2) >= f —
                # the deposit (canon direction x stamp) can never vanish
                # from the garage; gradient w/sqrt(||w||^2+f^2) is bounded
                # and nonzero except at the exact origin (a point). Env-
                # gated: unset = the where-gated form below, bit-identical.
                _wn4 = (_ss4 + _pcf4 * _pcf4).sqrt() + 1e-6
            else:
                _sp4 = _ss4 > 0
                _wn4 = _sp4.where(_sp4.where(_ss4, 1.0).sqrt(), 0.0) + 1e-6
            _cn4 = _canon4.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6
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
            _dep4 = _canon4 / _cn4 * _wn4
            _wg4 = _dep4.detach()
            _pcl4 = (globals().get("_PCV")
                     if float(os.environ.get("ALG_PC_MIX", "0")) > 0.0
                     and int(os.environ.get("ALG_PC_LIVE", "1"))
                     and not os.environ.get("SC_EVAL", "") else None)
            if _pcl4 is not None:
                # THE LIVE WIRE (apply_pressure_mix.py, 2026-09-06):
                # for SEALED training rows the shelf crossing carries
                # gradient (per-row blend of live/detached — VALUES
                # identical either way; detach only cuts the tape).
                # Because _canon4 rides an argmax one-hot (grad-dead
                # direction) the surviving live channel into the
                # committer (W_bind1/2, upstream cur) is exactly the
                # confidence stamp _wn4: commitment AMPLITUDE learns
                # from downstream use; fact IDENTITY stays discrete.
                # Dual-terminal contract: _snaps (the solver facts,
                # detached above) untouched; open rows detached as
                # today; val/reads detached via the SC_EVAL gate.
                _plv4 = _pcl4.reshape(-1, 1, 1)
                _wg4 = _dep4 * _plv4 + _dep4.detach() * (1.0 - _plv4)
        _garage.append(_wg4)
    state["cur"] = cur; state["nb"] = _nb; state["nb_st"] = _nb_st
    state["rb_last"] = _rb_last
    state["m_c"] = m_c; state["anchor"] = anchor; state["x_rel"] = x_rel
    return state


def forward(p, trunk, tokmask, sent, slot_mask=None, revoke=None, tail=None, drop=None, anchor=None, amask=None, gmod=None, pmask=None, lsent=None, reg=None, fact_buf=None, mh_mass=None, mh_atlas_traj=None):
    from tinygrad import Tensor, dtypes   # audit 2026-09-01: was a
    # SCOPE ACCIDENT (bound only via the sixwave/sync branches — any
    # SIXWAVE-off config killed five organs at step 1)
    B = trunk.shape[0]
    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]
    if FED_WAIST and "fed_w2b" in p:
        # FED item 3: waist2 = waist + MLP(waist), output ZERO-INIT —
        # exact zeros at birth; every downstream organ (bank closure,
        # breath ctx, step-trainer tap) inherits the rebound name
        waist = waist + ((waist @ p["fed_w2a"] + p["fed_w2a_b"]).gelu()
                         @ p["fed_w2b"] + p["fed_w2b_b"])

    bank = _make_bank(p, waist, tokmask, B)
    if N_SCR and slot_mask is not None:
        # FED item 6 mask ruling: scratch rows (queries) OPEN to all;
        # scratch columns CLOSED here (no cold read-back at birth —
        # the raising law; the fed mixer's zero door is the channel)
        _f6c = slot_mask[:, :, :1] * 0.0            # (B, L_FAC, 1) zeros
        _f6top = Tensor.cat(slot_mask,
                            *([_f6c] * N_SCR), dim=2)  # (B, L_FAC, L_TOT)
        _f6r = _f6top[:, :1, :] * 0.0 + 1.0         # (B, 1, L_TOT) ones
        slot_mask = Tensor.cat(_f6top,
                               *([_f6r] * N_SCR), dim=1)  # (B, L_TOT, L_TOT)
    _lb = None
    if lsent is not None:               # V2: letter-keyed partition — imposed
        _lb = lsent.reshape(B, 1, K_VARS, -1) * float(os.environ.get("LS_A", "1.0"))
    vst, vat = bank(p["vq"], K_VARS, pbias=_lb)
    _vst_base = vst   # pre-injection tap (step trainer reads this)
    if int(os.environ.get("ALG_ALT2", "0")) and fact_buf is not None:
        # ALTERNATOR V2 injection: symbolic facts ((B, 24, 4): known flag
        # + MSD digits/9) condition the var-slot states that args/res/y/
        # query pointers and every breath read against. BOTH guards are
        # load-bearing: env unset -> byte-identical baseline; fact_buf
        # None -> byte-identical too (the injection is skipped entirely).
        vst = _fact_inject(p, vst, fact_buf)
    _pb = None
    _pb_prior = None
    _sync = None
    if ALG_SYNC:
        from tinygrad import Tensor, dtypes
        _A = float(os.environ.get("SYNC_A", "1.0"))
        _phi = np.pi / 3.0 * (np.arange(L_TOT) % 6)
        _cph = Tensor(np.cos(_phi).astype(np.float32), dtype=dtypes.float)
        _sph = Tensor(np.sin(_phi).astype(np.float32), dtype=dtypes.float)
        _th0 = (sent - (sent // 6) * 6).float() * (math.pi / 3.0)
        _cth, _sth = _th0.cos(), _th0.sin()
        _scr = None
        if int(os.environ.get("SYNC_SCRAMBLE", "0")):
            _scr = np.random.RandomState(227).uniform(0, 2 * np.pi, 8)
        def _mk_pb(kb):
            _d = float(_scr[kb]) if _scr is not None else kb * (math.pi / 3.0)
            ck, sk = math.cos(_d), math.sin(_d)
            cthk = _cth * ck - _sth * sk
            sthk = _sth * ck + _cth * sk
            return (_cph.reshape(1, 1, L_TOT, 1) * cthk.reshape(B, 1, 1, -1)
                    + _sph.reshape(1, 1, L_TOT, 1)
                    * sthk.reshape(B, 1, 1, -1)) * _A
        _php = np.zeros((L_TOT, H_W), np.float32)
        def _mk_osc(kb):   # the receiver's local oscillator — same clock
            _d = float(_scr[kb]) if _scr is not None else kb * (math.pi / 3.0)
            _o = _php.copy()
            _o[:, 0] = np.cos(_phi + _d) * _A
            _o[:, 1] = np.sin(_phi + _d) * _A
            return Tensor(_o, dtype=dtypes.float).reshape(1, L_TOT, H_W)
        _sync = (_mk_pb, _mk_osc)
        _pb = _mk_pb(0)
    if ALG_SIXWAVE:
        from tinygrad import Tensor, dtypes
        # six helical carriers: token phase from sentence index (mod 6, 60
        # deg apart, antiphase pairs); slot phase from slot index mod 6.
        # Resonance bias cos(phi_slot - theta_tok) enters the factor bank's
        # scores through the zero-init gate — structure, never supervision.
        if int(os.environ.get("SW_SCRAMBLE", "0")):
            _tab = Tensor(np.random.RandomState(227).randint(0, 6, 16)
                          .astype(np.float32) * (math.pi / 3.0), dtype=dtypes.float)
            _th = _tab[sent - (sent // 16) * 16]
        else:
            _th = (sent - (sent // 6) * 6).float() * (math.pi / 3.0)
        _phi = np.pi / 3.0 * (np.arange(L_TOT) % 6)
        _cph = Tensor(np.cos(_phi).astype(np.float32), dtype=dtypes.float)
        _sph = Tensor(np.sin(_phi).astype(np.float32), dtype=dtypes.float)
        _sw_term = (_cph.reshape(1, 1, L_TOT, 1) * _th.cos().reshape(B, 1, 1, -1)
               + _sph.reshape(1, 1, L_TOT, 1)
               * _th.sin().reshape(B, 1, 1, -1)) * p["sw_g"].reshape(1, 1, 1, 1)
        _pb = _sw_term if _pb is None else _pb + _sw_term  # audit #6: adds
    if pmask is not None:                 # A0: imposed route-mask (wiring,
        _pb = pmask if _pb is None else _pb + pmask   # not knobs — no grad)
    fst, fat = bank(p["fq"], L_TOT, pbias=_pb)
    qst, _qa = bank(p["qq"], 1)

    # BRICK-P breathing (2026-07-09): K-1 refinement passes. Each breath
    # re-reads the text conditioned on current beliefs (bank-with-extra) +
    # MASKED slot-to-slot settling — evidence-sharing topology AS STRUCTURE
    # (the v98 escape; free-form slot attention is the perceiver trap and is
    # not built). Deltas enter via zero-init W_bo + init-closed gates:
    # at init the K-breath output is byte-identical to the incumbent.
    K_B = int(os.environ.get("ALG_BREATH", "1"))
    breaths = [fst]
    RINGS = int(os.environ.get("ALG_RINGS", "0")) and "W_cmt" in p
    # ORGAN-2: the reverse gear (spec §2). Release DYNAMICS only — the
    # revoke signal is an INPUT PORT: transport arrives solver-side, from
    # outside the neural partition (#152's leading candidate). Rates are
    # PINNED (2-3 breath release scale; #136), never tuned by feel. No
    # new params: no new terminal; the register clause holds (no settle,
    # no entropy — revoke is a named contradiction or nothing).
    XOUT = RINGS and int(os.environ.get("ALG_XOUT", "0"))
    XARM = os.environ.get("ALG_XARM", "dump")        # dump|graded|elastic
    XR_GRADED, XR_ELASTIC = 0.5, 0.15                # pinned, breath-scale
    if RINGS:
        m_c = (fst * 0.0).sum(-1, keepdim=True)      # (B,L_FAC,1) zeros
        anchor = fst
        cmt_logits = []
        x_rel = m_c                                   # released-mass ledger
    def _rot2(v, c, s):        # interleaved-real phasor rotation
        vr = v.reshape(*v.shape[:-1], v.shape[-1] // 2, 2)
        x, y = vr[..., 0], vr[..., 1]
        return Tensor.stack(x * c - y * s, x * s + y * c, dim=-1).reshape(*v.shape)

    _bus_reg = None
    _rb_last = None
    _garage = None
    global _CENSUS                  # the port census hook (inert unless
    try: _CENSUS                    # port_census.py arms it — same
    except NameError: _CENSUS = None  # pattern as _IMP below)
    global _IMP                     # the impulse hook (systems-ID probe;
    try: _IMP                       # None everywhere except under
    except NameError: _IMP = None   # impulse_response.py — inert in training)
    if (int(os.environ.get("ALG_BUSGARAGE", "0")) and "W_gq" in p
            and "W_bind2" in p):
        assert int(os.environ.get("ALG_BUSGARAGE", "0")) >= 2, \
            "garage v1 (raw wires) is dead — canonical shelf only"
        global _SGC
        try: _SGC
        except NameError: _SGC = None
        if _SGC is None:
            import numpy as _np4
            from tinygrad import Tensor as _Ts4
            _bz4 = _np4.load(_bind_codes_path())
            _conj4, _plus4 = {}, {}
            for _rn4 in ("arg1", "arg2", "res", "op"):
                _th4 = _bz4[f"theta_{_rn4}"]
                _conj4[_rn4] = (_Ts4(_np4.cos(-_th4).astype(_np4.float32)),
                                _Ts4(_np4.sin(-_th4).astype(_np4.float32)))
                _plus4[_rn4] = (_Ts4(_np4.cos(_th4).astype(_np4.float32)),
                                _Ts4(_np4.sin(_th4).astype(_np4.float32)))
            _SGC = (_conj4, _plus4, _Ts4(_bz4["CB"].astype(_np4.float32)))
        _garage = []
    _snaps = []
    _snaps_g = []   # router graded-input repair: softmax snap tuple (ALG_ROUTER_GRADED)
    _bs_ctx = _bs_state = None
    if K_B > 1 and slot_mask is not None and "W_bo" in p:
        cur = fst
        # FINAL BOSS rung 0 (2026-09-03): the loop BODY lives in
        # module-level breath_step (pure code motion — bit-identical by
        # construction); state carries what crosses breath boundaries,
        # ctx the per-forward constants. Under _STEP_TAP hold the fused
        # loop is SKIPPED — the step trainer drives the walk itself.
        _fed_nl0 = None
        if FED_NL0 and int(os.environ.get("ALG_MASKHEAD", "0")) \
                and "fed_nl0_w" in p:
            # FED item 8: the breath-0 invariant NL page (two-tap law)
            # — the nl-tap's pooled read, computed ONCE (the fq bank
            # pass has no cur/fact/mask reach), DETACHED at entry (the
            # mask head's metadata contract)
            _fed_nl0 = (_fed_core(fat).mean(1).unsqueeze(1)
                        @ waist).squeeze(1).detach()
        _bs_ctx = {"B": B, "K_B": K_B, "waist": waist, "tokmask": tokmask,
                   "slot_mask": slot_mask, "bank": bank, "rot2": _rot2,
                   "sync": _sync, "drop": drop, "gmod": gmod,
                   "revoke": revoke, "tail": tail, "reg": reg,
                   # MASK HEAD metadata (2026-09-05; plumbing only — a
                   # dict key, zero compute when ALG_MASKHEAD unset).
                   # ctx also serves the OPTIONAL per-seam ports read
                   # via ctx.get: "mh_mass" (per-var domain-mass from
                   # the solver ping) and "mh_atlas" (step_atlas
                   # consult page) — populated by seam drivers only.
                   "fact_buf": fact_buf, "mh_mass": mh_mass,
                   "fed_nl0": _fed_nl0,
                   "mh_atlas_traj": mh_atlas_traj,
                   "RINGS": RINGS, "XOUT": XOUT, "XARM": XARM,
                   "XR_GRADED": XR_GRADED, "XR_ELASTIC": XR_ELASTIC}
        _bs_state = {"cur": cur, "breaths": breaths, "nb": None,
                     # MASK HEAD storage (2026-09-05): the graded
                     # adjacency the organ consumed at the previous
                     # breath_step (detached) — Δ-visibility into the
                     # commitment FLOW; the notebook-threading contract
                     "mh_prev": None,
                     # FED item 9: shelf lane-2 ink (born at kb == 1)
                     "nb2": None,
                     "nb_st": None, "garage": _garage, "snaps": _snaps,
                     "snaps_g": _snaps_g, "rb_last": _rb_last,
                     "m_c": m_c if RINGS else None,
                     "anchor": anchor if RINGS else None,
                     "cmt_logits": cmt_logits if RINGS else None,
                     "x_rel": x_rel if RINGS else None}
        if not (_STEP_TAP is not None and _STEP_TAP.get("hold")):
            for kb in range(1, K_B):
                breath_step(p, _bs_state, kb, _bs_ctx)
            cur = _bs_state["cur"]
            _rb_last = _bs_state["rb_last"]
            if RINGS:
                m_c = _bs_state["m_c"]
                anchor = _bs_state["anchor"]
                cmt_logits = _bs_state["cmt_logits"]
                x_rel = _bs_state["x_rel"]

    def heads_of(s, vst=vst):
        return _heads_of(p, s, vst, B)
    if _STEP_TAP is not None:
        # the step trainer's stage-0 seam: everything the per-step walk
        # needs, single source (inert when None — the _CENSUS pattern)
        _STEP_TAP.update(ctx=_bs_ctx, state=_bs_state, waist=waist,
                         vst=vst, vst_base=_vst_base, fst=fst, qst=qst,
                         heads_of=heads_of, B=B)
    if int(os.environ.get("ALG_MINE_BREATHS", "0")):
        out_breaths = breaths          # v3: the dialect ladder's raw states
    _s_final = breaths[-1]
    if anchor is not None and amask is not None:     # FORM (A): state-side
        _s_final = amask * anchor + (1.0 - amask) * _s_final   # anchors —
                                                     # structural re-entry the
                                                     # forward cannot ignore
    out = heads_of(_s_final)
    if _rb_last is not None:
        out["rbias"] = _rb_last
    if int(os.environ.get("ALG_MINE_BREATHS", "0")) and K_B > 1 and slot_mask is not None:
        out["breaths_all"] = [_fed_core(_b9) for _b9 in out_breaths]
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
                                 + ((_bs_state or {}).get("r_all") or []))]
        # NL TAP (apply_nl_tap.py): the seven-page reading — breath
        # 0 is the same fq bank pass fst came from (fat); breaths
        # 1..K-1 were appended by breath_step under the same env.
        _nl0w = _fed_core(fat).mean(1)
        out["nl_all"] = ([(_nl0w.unsqueeze(1) @ waist).squeeze(1)]
                         + ((_bs_state or {}).get("nl_all") or []))
        out["nlat_all"] = ([_nl0w]
                           + ((_bs_state or {}).get("nlat_all") or []))
    if (int(os.environ.get("ALG_MINE_BREATHS", "0"))
            or int(os.environ.get("ALG_MH_XPRIOR", "0"))):
        # breath-0 NL state for the CROSS-ATLAS PRIOR — identical
        # in pass-1 and pass-2 (fq bank: no cur/fact/mask reach)
        out["nl0"] = (_fed_core(fat).mean(1)
                      .unsqueeze(1) @ waist).squeeze(1)
    if (int(os.environ.get("ALG_DEEPSUP", "0")) or ALG_CONSUME) and K_B > 1 and len(breaths) > 1:
        out["_early"] = [heads_of(b) for b in breaths[:-1]]   # V2: the whole
                                                  # supply chain gets gradient
    if int(os.environ.get("ALG_INV", "0")):
        out["fst_s"] = breaths[-1]        # ORGAN: invariance fire — the pair
                                          # agreement term reads the waist here
    if RINGS and K_B > 1 and slot_mask is not None and "W_bo" in p:
        out["cmt"] = cmt_logits[0].stack(*cmt_logits[1:], dim=1) if len(cmt_logits) > 1 \
            else cmt_logits[0].unsqueeze(1)               # (B, K_B-1, L_FAC)
        out["cmt_m"] = m_c.squeeze(-1)                  # final mass (B, L_FAC)
        if XOUT:
            out["xrel"] = x_rel.squeeze(-1)             # released-mass ledger
                                                        # (revisability meter's
                                                        # raw feed; audit-side)
    out["query"] = _fed_pf(
        p, "query", qst, vst,
        (qst @ p["W_query"]) @ vst.transpose(-2, -1)).reshape(B, K_VARS)
    if "h_depth" in p:
        out["depth"] = _fed_core(fst) @ p["h_depth"] + p["h_depth_b"]
        out["term"] = (_fed_core(fst) @ p["h_term"]
                       + p["h_term_b"]).squeeze(-1)
    if "W_bind2" in p:
        # THE TAP (2026-08-29): ALG_BINDTAP=1 or v>=7 reads _s_final (the
        # refined post-breath state, like every parse head); default fst =
        # breath-0 (the bus era's historical tap, kept for lineage compat)
        _bsrc = (_s_final if int(os.environ.get("ALG_BINDTAP", "0"))
                 else fst)   # tap dial ORTHOGONAL to version (v7a verdict:
                             # invariance lives at breath-0; default stays)
        out["bind"] = (_fed_core(_bsrc) @ p["W_bind1"]
                       + p["W_bind1_b"]).gelu() @ p["W_bind2"]
    if "W_opc1" in p:
        if int(os.environ.get("ALG_OPCOUNT", "0")) == 2:
            # rescue variant (registered pre-fire; mechanism-cleared): pool
            # over the FIXED 24 slots — constant denominator keeps the
            # readout EXTENSIVE (counts survive; token-mean normalized
            # them away — the intensive-readout exclusion)
            _pool = _fed_core(fst).mean(1)
        else:
            _pool = ((waist * tokmask.unsqueeze(-1)).sum(1)
                     / (tokmask.sum(1, keepdim=True) + 1e-6))
        out["opc"] = ((_pool @ p["W_opc1"] + p["W_opc1_b"]).gelu()
                      @ p["W_opc2"] + p["W_opc2_b"]).reshape(
                          B, len(OPC_CLASSES), OPC_CAP + 1)
    out["fat"], out["vat"] = _fed_core(fat), vat
    if "h_ref" in p:
        out["ref"] = waist @ p["h_ref"] + p["h_ref_b"]   # (B, T_ALG, K_VARS)
    if len(breaths) > 1:
        out["breaths"] = [heads_of(s) for s in breaths]
    return out


def loss_fn(o, g):
    """Ladder wrapper: with breaths, per-breath weighted CE (1 + k/(K-1)) —
    the v98 ladder; the shared query/span terms ride the final breath only."""
    if "breaths" in o:
        K_B = len(o["breaths"])
        tot = None
        for kb, ob in enumerate(o["breaths"]):
            full = dict(o, **ob)
            w = 1.0 + kb / max(K_B - 1, 1)
            term = _loss_single(full, g) * w
            tot = term if tot is None else tot + term
        if int(os.environ.get("BREATH_NORM", "0")):
            _wsum = sum(1.0 + kb / max(K_B - 1, 1) for kb in range(K_B))
            return tot / _wsum          # door #50: TRUE-scale ladder — the
                                        # 1.5x heat confound removed at birth
        return tot / K_B
    return _loss_single(o, g)


def _loss_single(o, g):
    from tinygrad import Tensor
    pres = g["presence"]
    n_p = pres.sum() + 1e-6
    # explicit per-kind masks (tranche); legacy callers without them fall back
    # to the old two-kind arithmetic (byte-identical behavior)
    is_rel = g["is_rel"] if "is_rel" in g else (1.0 - g["is_lit_f"])
    is_mod = g["is_mod"] if "is_mod" in g else g["is_lit_f"] * 0.0
    is_sel = g["is_sel"] if "is_sel" in g else g["is_lit_f"] * 0.0
    is_pct = g["is_pct"] if "is_pct" in g else g["is_lit_f"] * 0.0
    is_fdiv = g["is_fdiv"] if "is_fdiv" in g else g["is_lit_f"] * 0.0
    rel = pres * is_rel
    n_rel = rel.sum() + 1e-6

    def bce(lg, tg):
        return lg.maximum(0) - lg * tg + (1 + (-lg.abs()).exp()).log()

    def ce(lg, tg):
        return (lg.log_softmax(-1) * -1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)

    l = bce(o["pres"], pres).mean()
    l = l + (ce(o["ftype"], g["ftype"]) * pres).sum() / n_p
    l = l + (ce(o["op"], g["op"]) * rel).sum() / n_rel
    l = l + bce(o["islit"], g["is_lit_f"]).mean()
    if "depth" in o and "depth" in g:      # the position channel (gold-fed)
        l = l + (ce(o["depth"], g["depth"]) * pres).sum() / n_p
        l = l + (bce(o["term"], g["term"]) * pres).sum() / n_p
    if "opc" in o and "opc" in g:          # lever 3: row-grain count CE
        l = l + ce(o["opc"], g["opc"]).mean()
    if "bind" in o and "bindvec" in g and int(os.environ.get("ALG_BINDBUS", "0")) < 5:
        # the rotational binding bus (v5 excluded: pure CE, no wire loss)
        _e = o["bind"]; _t = g["bindvec"]
        _cos = (_e * _t).sum(-1) / ((_e.pow(2).sum(-1).sqrt() + 1e-6)
                                    * (_t.pow(2).sum(-1).sqrt() + 1e-6))
        _w = 1.0 if int(os.environ.get("ALG_BINDBUS", "0")) < 3 else 0.2
        l = l + _w * ((1.0 - _cos) * pres).sum() / n_p
    if "rbias" in o and "fspan" in g:
        # v3 router span loss (bootstrap law: new attention pathways get
        # DIRECT supervision — gold factor spans, per slot)
        l = l + 0.5 * (bce(o["rbias"], g["fspan"]).mean(-1) * pres).sum() / n_p
    if "bind" in o and "bind_ids" in g and int(os.environ.get("ALG_BINDBUS", "0")) >= 3:
        # v3 THE ROLE-FACTORED LOSS: supervise each role's unbound cleanup
        # directly — conjugate-rotate the emission, CE against the codebook
        _e = o["bind"]        # own binding: the cosine branch (which also
                              # sets _e) is skipped in pure-CE modes (v7+)
        global _BINDC
        try: _BINDC
        except NameError: _BINDC = None
        if _BINDC is None:
            import numpy as _np
            from tinygrad import Tensor as _T3
            _bz = _np.load(_bind_codes_path())
            _CBt = _T3(_bz["CB"].astype(_np.float32))
            _rc = {}
            for _ri, _r in enumerate(("arg1", "arg2", "res", "op")):
                _th = _bz[f"theta_{_r}"]
                _rc[_ri] = (_T3(_np.cos(-_th).astype(_np.float32)),
                            _T3(_np.sin(-_th).astype(_np.float32)))
            _BINDC = (_CBt, _rc)
        _CBt, _rc = _BINDC
        _P2 = _e.shape[-1] // 2
        _er = _e.reshape(*_e.shape[:-1], _P2, 2)
        _ex, _ey = _er[..., 0], _er[..., 1]
        for _ri in range(4):
            _c3, _s3 = _rc[_ri]
            _ux = _ex * _c3 - _ey * _s3
            _uy = _ex * _s3 + _ey * _c3
            _u = Tensor.stack(_ux, _uy, dim=-1).reshape(*_e.shape)
            _lg = _u @ _CBt.T                       # (B, L_FAC, 32)
            _y = g["bind_ids"][..., _ri]
            l = l + 0.5 * (ce(_lg, _y) * pres).sum() / n_p
    is_macro = g["is_macro"] if "is_macro" in g else is_mod * 0.0
    is_frac = g["is_frac"] if "is_frac" in g else is_mod * 0.0
    dm = g["is_lit_f"] + is_mod + is_pct + is_fdiv + is_macro + is_frac
    l = l + (ce(o["dig"], g["digits"]).mean(-1) * dm).sum() / (dm.sum() + 1e-6) \
        * float(os.environ.get("OBJW_DIG", "1.0"))       # pool axis OBJW
    if "sgn" in o and "sign" in g:              # E1: sign BCE on value slots
        l = l + (bce(o["sgn"], g["sign"]) * dm).sum() / (dm.sum() + 1e-6)
    if "cmt" in o:      # RUNG-3: commit-when-correct (self-labeled, DETACHED
                        # targets from gold-match — settle appears nowhere)
        ok_ft = (o["ftype"].argmax(-1) == g["ftype"]).float()
        ok_res = (o["res"].argmax(-1) == g["res"]).float()
        correct = (ok_ft * ok_res * pres).detach()      # (B, L_FAC)
        ck = o["cmt"]                                    # (B, KB-1, L_FAC)
        l = l + (bce(ck, correct.unsqueeze(1).expand(*ck.shape)) *
                 pres.unsqueeze(1)).sum() / (pres.sum() * ck.shape[1] + 1e-6)
    if "sel" in o and "sel" in g:               # selector-type CE (closed vocab)
        sm = pres * is_sel
        l = l + (ce(o["sel"], g["sel"]) * sm).sum() / (sm.sum() + 1e-6)
    if "dup" in o and "arg_dup" in g:           # gen-9: arg-multiplicity BCE
        l = l + (bce(o["dup"], g["arg_dup"]) * rel).sum() / n_rel
    if "dargs" in o and "arg_dup" in g:         # door #12: the dedicated dup
        dm2 = rel * g["arg_dup"]                # pointer — single-target gold
        l = l + (ce(o["dargs"], g["args"].argmax(-1)) * dm2).sum() / (dm2.sum() + 1e-6)
    if "valspan" in g:                          # door #61: given-binding aid —
        _vs = g["valspan"]                       # the pointer law's structural
        _vm = (_vs.sum(-1) > 0).float()          # entry at the value grain
        _vsn = _vs / (_vs.sum(-1, keepdim=True) + 1e-6)
        l = l + float(os.environ.get("VALATT_W", "1.0")) * (
            (-(o["fat"] + 1e-9).log() * _vsn).sum(-1) * _vm).sum() / (_vm.sum() + 1e-6)
    if "iargs" in o and "is_ind" in g:          # door #45: dialect reader —
        im = is_rel * g["is_ind"] * pres        # indirect-population gold only
        l = l + float(os.environ.get("DIAL_W", "1.0")) * (
            (bce(o["iargs"], g["args"]).mean(-1) * im).sum() / (im.sum() + 1e-6))
    if "dig2" in o and "is_macro" in g:        # gen-15: OP_APPLY terms
        if "is_frac" in g:                     # mg2: frac k rides the dig2 CE
            mac2 = pres * (g["is_macro"] + g["is_frac"])
            l = l + (ce(o["dig2"], g["digits2"]) .mean(-1) * mac2).sum() / (mac2.sum() + 1e-6)
        mac = pres * g["is_macro"]
        n_mac = mac.sum() + 1e-6
        l = l + (ce(o["op"], g["op"]) * mac).sum() / n_mac
        l = l + (ce(o["dig2"], g["digits2"]).mean(-1) * mac).sum() / n_mac
        l = l + (ce(o["y"], g["y"]) * mac).sum() / n_mac * 2.0
    args_w = 1.0 + 4.0 * g["args"]
    is_chain = g["is_chain"] if "is_chain" in g else is_mod * 0.0
    am = pres * (is_rel + is_sel + is_mod + is_pct + is_fdiv + is_macro + is_frac + is_chain)
    n_am = am.sum() + 1e-6
    _ow_ptr = float(os.environ.get("OBJW_PTR", "1.0"))   # pool axis OBJW
    l = l + ((bce(o["args"], g["args"]) * args_w).mean(-1) * am).sum() / n_am * 2.0 * _ow_ptr
    l = l + (ce(o["res"], g["res"]) * pres).sum() / n_p * 2.0 * _ow_ptr
    l = l + ce(o["query"], g["query"]).mean() * 2.0 * _ow_ptr
    fsn = g["fspan"] / (g["fspan"].sum(-1, keepdim=True) + 1e-6)
    # FAT_W (routing-canvas dose probe, gut #55 amended): the fat-CE canvas
    # has hung here at weight 1 all along; the probe doses it, never adds it.
    fat_w = float(os.environ.get("FAT_W", "1"))
    l = l + fat_w * ((-(o["fat"] + 1e-9).log() * fsn).sum(-1) * pres).sum() / n_p
    vsn = g["vspan"] / (g["vspan"].sum(-1, keepdim=True) + 1e-6)
    vmask = (g["vspan"].sum(-1) > 0).float()
    l = l + ((-(o["vat"] + 1e-9).log() * vsn).sum(-1) * vmask).sum() / (vmask.sum() + 1e-6)
    if "ref" in o and "refoh" in g:
        _rm = g["refoh"].sum(-1)                          # (B, T_ALG) site mask
        _rce = (-(o["ref"].log_softmax(-1)) * g["refoh"]).sum(-1)
        l = l + float(os.environ.get("REF_W", "0.1")) * (_rce * _rm).sum() / (_rm.sum() + 1e-6)
    return l


# ===========================================================================
# DECODE + PER-BAND EVAL (factor-exact / graph-solve / ANSWER, per band)
# ===========================================================================

def decode(o_np):
    from mycelium.csp_domains import ID_TO_SEL
    facs = []
    for j in range(L_FAC):
        if o_np["pres"][j] <= 0:
            continue
        res = int(o_np["res"][j].argmax())
        ft = int(o_np["ftype"][j].argmax()) if o_np["ftype"].shape[-1] >= 4 \
            else (1 if o_np["islit"][j] > 0 else 0)

        def digval():
            digs = o_np["dig"][j].argmax(-1)
            return int(sum(d * 10 ** (N_DIG - 1 - i)
                           for i, d in enumerate(digs)))
        if ft == 1:
            v = digval()
            if "sgn" in o_np and o_np["sgn"][j] > 0:   # E1: negative literal
                v = -v
            facs.append({"ftype": "given", "var": res, "value": v})
        elif ft == 0:
            op = "add" if o_np["op"][j].argmax() == 0 else "mul"
            if "dup" in o_np and o_np["dup"][j] > 0:
                a0 = int(np.argmax(o_np["dargs"][j])) if "dargs" in o_np \
                    else int(np.argmax(o_np["args"][j]))   # door #12 / gen-9
                args = [a0, a0]
            else:
                args = sorted(int(a) for a in
                              np.argsort(-o_np["args"][j])[:2])
            facs.append({"ftype": "rel", "op": op,
                         "args": args, "result": res})
        elif ft == 2:
            var = int(np.argmax(o_np["args"][j]))
            facs.append({"ftype": "mod", "var": var, "k": max(digval(), 2),
                         "result": res})
        elif ft == 4:
            facs.append({"ftype": "pct",
                         "args": [int(np.argmax(o_np["args"][j])), res],
                         "p": max(digval(), 1)})
        elif ft == 5:
            facs.append({"ftype": "fdiv",
                         "var": int(np.argmax(o_np["args"][j])),
                         "k": max(digval(), 2), "result": res})
        elif ft == 8:
            xs = sorted(int(v) for v in np.where(o_np["args"][j] > 0)[0].tolist())
            if len(xs) >= 3:
                facs.append({"ftype": "macro", "name": "CHAIN_MUL",
                             "xs": xs, "result": res})
            continue
        elif ft == 7 and "dig2" in o_np:
            k_ = int(sum(d * 10 ** (N_DIG - 1 - i2) for i2, d in
                         enumerate(o_np["dig2"][j].argmax(-1))))
            facs.append({"ftype": "macro", "name": "FRAC_OF",
                         "a": max(digval(), 1), "k": max(k_, 2),
                         "x": int(np.argmax(o_np["args"][j])), "result": res})
        elif ft == 6 and "dig2" in o_np:
            k2 = int(sum(d * 10 ** (N_DIG - 1 - i2) for i2, d in
                         enumerate(o_np["dig2"][j].argmax(-1))))
            facs.append({"ftype": "macro", "name": "OP_APPLY",
                         "op": "add" if o_np["op"][j].argmax() == 0 else "sub",
                         "k1": max(digval(), 1), "x": int(np.argmax(o_np["args"][j])),
                         "k2": max(k2, 1), "y": int(np.argmax(o_np["y"][j])),
                         "result": res})
        else:
            args = list(np.argsort(-o_np["args"][j])[:2])
            sel = ID_TO_SEL[int(o_np["sel"][j].argmax())]
            facs.append({"ftype": "sel", "sel": sel,
                         "args": sorted(int(a) for a in args), "result": res})
    return facs, int(o_np["query"].argmax())


def do_eval():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)
    if int(os.environ.get("ALG_TRUNK_LORA", "0")):
        # audit 2026-08-22 #3: banked states are PURE-trunk; a LoRA ckpt
        # must be ringed through its adapters or the number is a no-op
        from beacon_closing_arm import _trunk_host
        from mycelium.llama_loader import _rms_norm as _lrms
        _host = _trunk_host()
        _sc = float(os.environ.get("ALG_LORA_SCALE", "8.0"))
        _ro = int(os.environ.get("ALG_ROPE_OFF", "0"))
        _LD = [{f"{_nm}_{_ab}": (p[f"lora{_li}_{_nm}_{_ab}"] * (_sc if _ab == "B" else 1.0))
                for _nm in ("wq", "wo", "wdown") for _ab in ("A", "B")
                if f"lora{_li}_{_nm}_A" in p} or None
               for _li in range(4)]
        _samples2, _ids2, _msk2, _off2 = tokenize(ALG_TEST)
        assert len(_samples2) == n, "eval tokenize/load_alg desync"
        _st2 = np.zeros((n, T_ALG, H_TRUNK), np.float16)
        for _s0 in range(0, n, 8):
            _sl = slice(_s0, min(_s0 + 8, n))
            _x = _host.llama_embed[Tensor(_ids2[_sl], dtype=dtypes.int)]
            for _li, _layer in enumerate(_host.llama_layers):
                _x = _layer(_x, _host.llama_rope_cos[_ro:] if _ro else _host.llama_rope_cos,
                            _host.llama_rope_sin[_ro:] if _ro else _host.llama_rope_sin,
                            lora=_LD[_li])
            _x = _lrms(_x, _host.llama_layers[-1].ffn_norm,
                       _host.llama_cfg.rms_norm_eps)
            _st2[_sl] = _x.cast(dtypes.float).realize().numpy().astype(np.float16)
        states = _st2
        print(f"[eval] TRUNK_LORA: states recomputed through adapters "
              f"(scale {_sc}) — HONEST LoRA-ring", flush=True)
    per_band = {}
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        t_tr = Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float)
        t_tk = Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float)
        t_se = Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int)
        _tl = Tensor(tails_of(sent[sl_p]), dtype=dtypes.float) \
            if int(os.environ.get("ALG_CLOCK", "0")) else None
        out = forward(p, t_tr, t_tk, t_se, tail=_tl)
        if int(os.environ.get("ALG_BREATH", "1")) > 1 and "W_bo" in p \
                and not int(os.environ.get("BREATH_SILENT", "0")):
            o0 = {k: out[k].realize().numpy() for k in ("fat", "args", "res")}
            mk = build_slot_masks(o0, sent[sl_p])
            out = forward(p, t_tr, t_tk, t_se, tail=_tl,
                          slot_mask=Tensor(mk, dtype=dtypes.float))
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ()) \
            + (("dargs",) if "dargs" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, i in enumerate(sl):
            i = int(i)
            smp = samples[i]
            band = int(gold["band"][i])
            st = per_band.setdefault(band, {"n": 0, "fac_ok": 0, "fac_tot": 0,
                                            "solve": 0, "answer": 0, "query_ok": 0})
            st["n"] += 1
            facs, q_pred = decode({k: o[k][bi] for k in o})

            def fkey(f):
                if f["ftype"] == "rel":
                    return ("rel", f["op"], tuple(sorted(f["args"])),
                            f["result"])
                if f["ftype"] == "given":
                    return ("given", f["var"], f["value"])
                if f["ftype"] == "mod":
                    return ("mod", f["var"], f["k"], f["result"])
                if f["ftype"] == "pct":
                    return ("pct", tuple(f["args"]), f["p"])
                if f["ftype"] == "fdiv":
                    return ("fdiv", f["var"], f["k"], f["result"])
                return ("sel", f["sel"], tuple(sorted(f["args"])), f["result"])
            gset = set(fkey(f) for f in smp["factors"])
            st["fac_ok"] += len(gset & set(fkey(f) for f in facs))
            st["fac_tot"] += len(gset)
            st["query_ok"] += int(q_pred == smp["query_var"])
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}

            def fvars(f):
                if f["ftype"] in ("rel", "sel", "pct"):
                    return list(f["args"]) + ([f["result"]]
                                              if "result" in f else [])
                if f["ftype"] in ("mod", "fdiv"):
                    return [f["var"], f["result"]]
                return [f["var"]]
            try:
                from mycelium.csp_domains import problem_from_algebra3
                nv = max([smp["n_vars"]] + [v + 1 for f in facs
                                            for v in fvars(f)])
                res = solve_symbolic(problem_from_algebra3(nv, facs, gv,
                                                           smp["m"]),
                                     budget=200_000, seed=0)
                if res["status"] == "solved":
                    sol = [int(res["assignment"][v]) for v in range(nv)]
                    if sol[:smp["n_vars"]] == smp["solution"]:
                        st["solve"] += 1
                    gold_ans = smp["solution"][smp["query_var"]]
                    if q_pred < len(sol) and sol[q_pred] == gold_ans:
                        st["answer"] += 1
            except Exception:
                pass

    print(f"\n[algebra eval] per-BAND (the registered stratification):")
    print(f"  band |  n | fac-F1ish | query | graph-solve | ANSWER")
    tot = {"n": 0, "solve": 0, "answer": 0}
    for b in sorted(per_band):
        st = per_band[b]
        print(f"    {b:2d} | {st['n']:2d} |   {st['fac_ok']/max(st['fac_tot'],1):.3f}   "
              f"|  {st['query_ok']/max(st['n'],1):.2f} |     {st['solve']:2d}      |   {st['answer']:2d}")
        for k in tot:
            tot[k] += st[k] if k != "n" else st["n"]
    print(f"  TOTAL: {tot['solve']}/{tot['n']} graph-solve, "
          f"{tot['answer']}/{tot['n']} ANSWER")
    print(f"  (factorization read: flat fac-exact across bands = reading and"
          f" reasoning on independent axes)")




# ===========================================================================
# TAXONOMY (--errors): prediction #2-algebra's grader (spec §11)
# ===========================================================================
# Classes: CORRECT | DETECT_malformed | DETECT_unsat | DETECT_multi (ban-and-resolve
# uniqueness probe, gold-free) | SILENT (unique-solves to the WRONG answer). Literal
# errors attributed by the gold given's ROLE: chain_k vs pair_sum/pair_diff — the
# registered ordering: chain-literals >> coupled-literals > structural (~0).

def do_errors():
    from tinygrad import Tensor, dtypes
    from tinygrad.nn.state import safe_load
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic

    samples, states, tokmask, gold, sent = load_alg("test")
    p = build_params(0)
    sd = safe_load(ALG_CKPT)
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    n = len(samples)
    cats = {"CORRECT": 0, "DETECT_malformed": 0, "DETECT_unsat": 0,
            "DETECT_multi": 0, "SILENT": 0}
    lit_err = {}     # class -> {role: count} for wrong-literal cases
    for s0 in range(0, n, 8):
        sl = np.arange(s0, min(s0 + 8, n))
        pad = 8 - len(sl)
        sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
        out = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                      Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int),
                      lsent=(Tensor(gold["lsent"][sl_p].astype(np.float32),
                                    dtype=dtypes.float) if ALG_LSENT and "lsent" in gold else None))
        keys = ("pres", "ftype", "op", "islit", "dig", "args", "res",
                "query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ()) \
            + (("dargs",) if "dargs" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi, i in enumerate(sl):
            i = int(i)
            smp = samples[i]
            facs, q_pred = decode({k: o[k][bi] for k in o})
            # wrong-literal attribution (pred given value != gold given value, by var)
            gold_gv = {f["var"]: (f["value"], f.get("role", "?"))
                       for f in smp["factors"] if f["ftype"] == "given"}
            wrong_lits = [(v, gold_gv[v][1]) for f in facs if f["ftype"] == "given"
                          for v in [f["var"]] if v in gold_gv
                          and f["value"] != gold_gv[v][0]]
            rels = [(f["op"], f["args"][0], f["args"][1], f["result"])
                    for f in facs if f["ftype"] == "rel"]
            gv = {f["var"]: f["value"] for f in facs if f["ftype"] == "given"}
            gold_ans = smp["solution"][smp["query_var"]]
            cat = None
            try:
                nv = max([smp["n_vars"]] + [v + 1 for f in facs for v in
                         ((list(f["args"]) + [f["result"]]) if f["ftype"] == "rel"
                          else [f["var"]])])
                if nv > 26:
                    cat = "DETECT_malformed"
                else:
                    res = solve_symbolic(problem_from_algebra(nv, rels, gv, smp["m"]),
                                         budget=200_000, seed=0)
                    if res["status"] != "solved":
                        cat = "DETECT_unsat"
                    else:
                        sol = [int(res["assignment"][v]) for v in range(nv)]
                        ans_ok = q_pred < len(sol) and sol[q_pred] == gold_ans
                        if ans_ok and not wrong_lits:
                            cat = "CORRECT"
                        else:
                            # uniqueness probe on the PREDICTED graph (gold-free)
                            multi = False
                            for v in range(nv):
                                if v in gv:
                                    continue
                                p2 = problem_from_algebra(nv, rels, gv, smp["m"])
                                p2.domains0[v].discard(sol[v])
                                if p2.domains0[v]:
                                    r2 = solve_symbolic(p2, budget=100_000, seed=0)
                                    if r2["status"] != "unsat":  # deep clean 2026-07-30: budget is not a uniqueness certificate
                                        multi = True
                                        break
                            if ans_ok:
                                cat = "CORRECT"   # right answer, benign literal drift
                            elif multi:
                                cat = "DETECT_multi"
                            else:
                                cat = "SILENT"
            except Exception:
                cat = "DETECT_malformed"
            cats[cat] += 1
            if wrong_lits and cat in ("SILENT", "DETECT_unsat", "DETECT_multi"):
                for _v, role in wrong_lits:
                    lit_err.setdefault(cat, {}).setdefault(role, 0)
                    lit_err[cat][role] += 1

    wrong = n - cats["CORRECT"]
    det = cats["DETECT_malformed"] + cats["DETECT_unsat"] + cats["DETECT_multi"]
    print(f"\n[algebra errors] n={n}")
    for k, v in cats.items():
        print(f"  {k:18s} {v}")
    if wrong:
        print(f"  DETECTABLE fraction: {det}/{wrong} = {det/wrong:.2f} "
              f"(KenKen was 1.00 seven times — the INVERSION is the prediction)")
    print(f"  wrong-literal attribution by role x class: {lit_err}")
    print(f"  (registered ordering: chain_k literals >> pair literals (parity ~half-"
          f"caught) > structural ~0)")


# ===========================================================================
# TRAIN
# ===========================================================================

# ===========================================================================
# THE MASK-PREP CACHE (apply_maskprep_cache.py, 2026-09-08) — a perf
# organ, no science. Everything here is dark unless ALG_MASKPREP_CACHE
# is set; see the apply script's docstring for the full contract.
# ===========================================================================
_MP_CTRL = ("ALG_MASKPREP_CACHE", "ALG_MASKPREP_DIR", "ALG_MASKPREP_IGNORE")
_MP_EXTRA_ENV = ("DEV", "BEAM", "JIT", "NOOPT")
_MP_ARRAYS = ("MASKS", "FACTS", "MASSB", "NL0")
# Files at or under this size are keyed by their BYTES; bigger ones by
# (size, mtime_ns) plus whatever DECLARED stamp they carry. The staged
# train npz is 14 GB at form-scale and the states memmap 140 GB — hashing
# them would cost more than the pass this cache exists to skip, and on a
# box whose RAM is already holding a training run it would be an OOM, not
# a slowdown. Which mode was used is RECORDED in the fingerprint, so a
# bucket can always be autopsied for what it actually checked.
_MP_HASH_CAP = 256 << 20
# the dials read ONLY after the pass has run — the difference between two
# arms that share a diet and differ in LENGTH. Excluded from the key so
# those arms share a bucket; the exclusion is CHECKED against the source
# on every key build (_maskprep_trainer_only), never asserted in prose.
_MP_TRAINER_ONLY = ("STEPS", "LR", "BATCH", "VAL_EVERY", "SNAP_EVERY",
                    "PRECOMPUTE_ONLY")
# everything the mask-prep pass actually calls (the reachability set the
# exclusion check is run against)
_MP_PASS_FUNCS = ("forward", "build_params", "build_slot_masks",
                  "alt2_fact_buf", "_alt2_fact_buf_v0", "_alt2_fact_buf_v1",
                  "load_alg", "tokenize", "build_gold", "sent_indices",
                  "cross_prior")


def _maskprep_env_names(src):
    """Every env var name the head source itself reads. Mechanical, so a
    door added tomorrow enters the key without anyone remembering to add
    it here (the hand-maintained list is the thing that goes stale)."""
    import re
    return set(re.findall(
        r'os\.environ(?:\.get)?[\(\[]\s*"([A-Za-z0-9_]+)"', src))


def _maskprep_file_fp(path):
    """A file's fingerprint: its bytes when that is affordable, else its
    (size, mtime_ns). Never silent about which — the record says so."""
    from mycelium.era import mix_sha16
    if not path or not os.path.isfile(path):
        return None
    st = os.stat(path)
    if st.st_size <= _MP_HASH_CAP:
        return {"sha": mix_sha16(path), "size": int(st.st_size)}
    return {"sha": None, "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns), "why": "over the hash cap"}


def _maskprep_trainer_only(src):
    """Prove, from the source, that every name in _MP_TRAINER_ONLY is read
    only AFTER the mask-prep pass: never at module scope, never inside a
    function the pass calls, and never in do_train ahead of the pass. If
    one of them ever moves into that region this raises and the name goes
    back into the key — the exclusion cannot rot silently."""
    import ast
    tree = ast.parse(src)
    seg = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in _MP_PASS_FUNCS:
                seg.append(ast.get_source_segment(src, node) or "")
        elif not isinstance(node, ast.ClassDef):
            seg.append(ast.get_source_segment(src, node) or "")   # module scope
    _dt = src.index("def do_train(")
    seg.append(src[_dt:src.index('print(f"[breath] masks ready', _dt)])
    read = _maskprep_env_names(chr(10).join(seg))   # the READ syntax, not
                                                   # a prose mention
    for nm in _MP_TRAINER_ONLY:
        assert nm not in read, (
            f"[maskprep] {nm} is read by code the mask-prep pass runs — it "
            f"can no longer be excluded from the cache key. Remove it from "
            f"_MP_TRAINER_ONLY (the arms that differ in it will re-key).")
    return set(_MP_TRAINER_ONLY)


def _maskprep_ignored():
    return sorted({x.strip() for x in
                   os.environ.get("ALG_MASKPREP_IGNORE", "").split(",")
                   if x.strip()})


def _maskprep_fingerprints(p, n, seed):
    """The full dependency record of the mask-prep pass, JSON-able. The
    cache key is a sha over this; the record itself is banked beside the
    arrays so a stale bucket can be autopsied instead of guessed at."""
    import hashlib
    import mycelium
    from mycelium.era import mix_sha16   # the ledger's ONE file-hash
                                         # organ (meter-divergence law)
    src_path = os.path.abspath(__file__)
    src = open(src_path).read()
    fp = {"v": 1, "head": mix_sha16(src_path)}
    # the mycelium modules this head imports — a code change one level
    # out must invalidate too
    import re as _re
    _mroot = os.path.dirname(os.path.abspath(mycelium.__file__))
    fp["mods"] = {}
    for _m in sorted(set(_re.findall(
            r'mycelium\.([a-z0-9_]+)', src))):
        _mp = os.path.join(_mroot, _m + ".py")
        if os.path.exists(_mp):
            fp["mods"][_m] = mix_sha16(_mp)
    # the parameters the pass actually runs (post RESUME/WARM/PAD-WARM)
    _hp = hashlib.sha256()
    for k in sorted(p):
        _a = np.ascontiguousarray(p[k].detach().numpy())
        _hp.update(k.encode())
        _hp.update(f"{_a.shape}{_a.dtype}".encode())
        _hp.update(_a.tobytes())
    fp["params"] = _hp.hexdigest()[:16]
    fp["n_params"] = len(p)
    fp["seed"] = int(seed)
    _wf = (ALG_CKPT if (int(os.environ.get("RESUME", "0"))
                        and os.path.exists(ALG_CKPT))
           else os.environ.get("WARM_FROM", ""))
    fp["warm"] = {"path": _wf, "file": _maskprep_file_fp(_wf)}
    # the train arrays, identified as load_alg identifies them
    _npz = STATES_NPZ.format(split=TRAIN_NAME)
    _npy = STATES_NPY.format(split=TRAIN_NAME)
    _z = np.load(_npz)     # lazy: reading one member, not the archive
    fp["train"] = {
        "name": TRAIN_NAME, "mix": ALG_TRAIN,
        "mix_sha16": mix_sha16(ALG_TRAIN),          # THE SHA-FENCE's key
        "npz": _npz, "npz_file": _maskprep_file_fp(_npz),
        "npz_stamp": (str(_z["mix_sha"]) if "mix_sha" in _z.files
                      else None),                   # the fence's stamp
        "npy": _npy, "npy_file": _maskprep_file_fp(_npy),
        "n": int(n)}
    # the environment
    _ign = set(_maskprep_ignored())
    _names = ((_maskprep_env_names(src)
               | {k for k in os.environ if k.startswith("ALG_")}
               | set(_MP_EXTRA_ENV))
              - set(_MP_CTRL) - _ign - _maskprep_trainer_only(src))
    fp["env"] = {k: os.environ.get(k) for k in sorted(_names)}
    fp["env_files"] = {k: _maskprep_file_fp(v)
                       for k, v in sorted(fp["env"].items())
                       if v and os.path.isfile(v)}
    fp["ignored"] = sorted(_ign)
    fp["trainer_only"] = sorted(_MP_TRAINER_ONLY)
    fp["pass"] = {"batch": 8}
    return fp


def _maskprep_path(key):
    d = os.environ.get("ALG_MASKPREP_DIR", ".cache")
    return os.path.join(d, f"maskprep_{key}.npz")


def _maskprep_ver_starts(n):
    """Batch starts the pass RE-RUNS on a hit: the first two, plus the
    tail batch when a declared ignore list is live (the ignore door's
    second wall)."""
    st = [s0 for s0 in (0, 8) if s0 < n]
    if _maskprep_ignored():
        _last = ((max(n, 1) - 1) // 8) * 8
        if _last not in st:
            st.append(_last)
    return st


def _maskprep_lookup(p, n, seed):
    """(key, fingerprints, cached-or-None). Door unset -> all None, and
    the call site's iterator stays range(0, n, 8)."""
    if not int(os.environ.get("ALG_MASKPREP_CACHE", "0")):
        return None, None, None
    import hashlib
    import json
    import time
    _t0 = time.time()
    fp = _maskprep_fingerprints(p, n, seed)
    key = hashlib.sha256(
        json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    path = _maskprep_path(key)
    _ig = _maskprep_ignored()
    if _ig:
        print(f"[maskprep] key EXCLUSIONS declared: {','.join(_ig)} "
              f"(verification adds the tail batch)", flush=True)
    if not os.path.exists(path):
        print(f"[maskprep] cache MISS key={key} ({time.time() - _t0:.1f}s "
              f"to key) -> running the pass, will bank {path}", flush=True)
        return key, fp, None
    z = np.load(path, allow_pickle=False)
    cached = {k: z[k] for k in z.files if k in _MP_ARRAYS}
    print(f"[maskprep] cache candidate {path} "
          f"({', '.join(f'{k}{cached[k].shape}' for k in sorted(cached))}) "
          f"— verifying", flush=True)
    return key, fp, cached


def _maskprep_finish(key, fp, cached, n, starts, arrays):
    """HIT: assert the recomputed batches equal the cached rows and hand
    back the cached arrays. MISS: bank what the pass just built. Either
    way the caller's arrays are replaced by the returned ones."""
    import json
    if cached is None:
        out = {k: v for k, v in arrays.items() if v is not None}
        path = _maskprep_path(key)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + f".tmp{os.getpid()}.npz"   # np.savez appends .npz
                                                # to any other suffix
        np.savez_compressed(
            tmp, _key=np.array(key),
            _fingerprints=np.array(json.dumps(fp, sort_keys=True, indent=1)),
            **out)
        os.replace(tmp, path)   # atomic: a killed run never leaves half
        print(f"[maskprep] cached -> {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB, arrays "
              f"{'+'.join(sorted(out))})", flush=True)
        return arrays
    rows = np.concatenate([np.arange(s0, min(s0 + 8, n)) for s0 in starts])
    live = {k for k, v in arrays.items() if v is not None}
    assert live == set(cached), (
        f"[maskprep] STALE CACHE {_maskprep_path(key)}: it holds "
        f"{sorted(cached)} but this run's doors fill {sorted(live)} — "
        f"the key is incomplete; delete the bucket and re-key")
    for nm in sorted(live):
        cur, c = arrays[nm], cached[nm]
        assert c.shape == cur.shape and c.dtype == cur.dtype, (
            f"[maskprep] STALE CACHE {_maskprep_path(key)}: {nm} is "
            f"{c.shape}/{c.dtype}, this run wants {cur.shape}/{cur.dtype}")
        if not np.array_equal(c[rows], cur[rows]):
            _bad = rows[[not np.array_equal(c[r], cur[r]) for r in rows]]
            raise RuntimeError(
                f"[maskprep] CACHE VERIFICATION FAILED on {nm}: "
                f"{len(_bad)}/{len(rows)} recomputed rows differ from the "
                f"cached ones (first: row {int(_bad[0])}) — bucket "
                f"{_maskprep_path(key)} is stale or the key is missing a "
                f"dependency. NOT falling back silently: fix the key (or "
                f"delete the bucket) and re-run.")
    _tail = "" if len(starts) <= 2 else " + tail (ignore-list live)"
    print(f"[maskprep] CACHE HIT key={key} verified on "
          f"{min(len(starts), 2)} batches{_tail}", flush=True)
    return {k: (cached[k] if k in cached else None) for k in arrays}


def do_train(steps, lr, batch, seed):
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.jit import TinyJit
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_save

    samples, states, tokmask, gold, sent = load_alg("train")
    n = states.shape[0]
    p = build_params(seed)
    TRUNK_LORA = int(os.environ.get("ALG_TRUNK_LORA", "0"))
    if TRUNK_LORA:
        from beacon_closing_arm import _trunk_host
        HOST = _trunk_host()
        _npz_p = STATES_NPY.format(split=os.environ.get("ALG_TRAIN_NAME", "train")).replace("_states.npy", ".npz")
        try:
            _z = np.load(_npz_p)
            IDS_ALL = _z["ids"]     # perf audit 2026-08-23 #4: assembly already
            print(f"[lora] ids from sidecar {_npz_p}", flush=True)   # tokenized
        except Exception:
            _, IDS_ALL, _, _ = tokenize(os.environ["ALG_TRAIN"])
        LORA_SCALE = float(os.environ.get("ALG_LORA_SCALE", "8.0"))
        ROPE_OFF = int(os.environ.get("ALG_ROPE_OFF", "0"))
        print(f"[lora] trunk-in-the-loop: r={os.environ.get('ALG_LORA_R','16')} "
              f"scale={LORA_SCALE} rows={len(IDS_ALL)}", flush=True)
    if int(os.environ.get("RESUME", "0")) and not os.path.exists(ALG_CKPT):
        # RESUME GUARD (deep clean 2026-07-30): a missing ckpt under RESUME=1
        # previously fell through to FRESH RANDOM INIT silently — a
        # silently-wrong segment is far costlier than a loud stop.
        raise RuntimeError(f"RESUME=1 but {ALG_CKPT} does not exist — "
                           f"refusing to cold-start silently (resume guard)")
    if int(os.environ.get("RESUME", "0")) and os.path.exists(ALG_CKPT):
        from tinygrad.nn.state import safe_load as _sl
        import shutil
        shutil.copy2(ALG_CKPT, ALG_CKPT + ".pre_resume")   # house rule
        # (2026-08-16): RESUME never clobbers the only copy — the pre-swap
        # checkpoint stays readable after the continuation overwrites in place
        sd0 = _sl(ALG_CKPT)
        assert set(sd0.keys()) == set(p.keys()), "resume key mismatch (hard error)"
        for k in p:
            p[k].assign(sd0[k].to(p[k].device).cast(p[k].dtype)).realize()
        print(f"[train] RESUMED from {ALG_CKPT}", flush=True)
    elif os.environ.get("WARM_FROM"):
        # tranche warm-start: load every shape-matching key from a LEGACY ckpt;
        # skips are EXPLICIT AND PRINTED (warm-start may skip loudly — eval
        # loads must hard-error; this is the training-side allowance)
        from tinygrad.nn.state import safe_load as _sl
        sd0 = _sl(os.environ["WARM_FROM"])
        n_load = 0
        for k in p:
            if k in sd0 and tuple(sd0[k].shape) == tuple(p[k].shape):
                p[k].assign(sd0[k].to(p[k].device).cast(p[k].dtype)).realize()
                n_load += 1
            elif k in sd0 and len(sd0[k].shape) == len(p[k].shape) and all(
                    o <= n_ for o, n_ in zip(sd0[k].shape, p[k].shape)):
                # PAD-WARM (2026-07-10, the ftype-router lesson): old shape is
                # a prefix of new — copy the trained slice, keep fresh init on
                # the new rows. Discarding a trained ROUTER forces relearning
                # inside a converged circuit (the bootstrap-trap family).
                import numpy as _np
                cur = p[k].detach().numpy()
                old = sd0[k].to(p[k].device).cast(p[k].dtype).numpy()
                sl = tuple(slice(0, o) for o in old.shape)
                cur[sl] = old
                p[k].assign(Tensor(cur, dtype=p[k].dtype)).realize()
                n_load += 1
                print(f"[warm] PAD-WARM {k} {tuple(old.shape)} -> "
                      f"{tuple(cur.shape)}", flush=True)
            else:
                print(f"[warm] SKIP {k} (fresh init: "
                      f"{'missing' if k not in sd0 else 'shape'})", flush=True)
        print(f"[train] WARM from {os.environ['WARM_FROM']}: "
              f"{n_load}/{len(p)} keys", flush=True)
        if "W_bo" in p and int(os.environ.get("BREATH_WARM_BO", "0")) and "attn_wo" in sd0:
            p["W_bo"].assign(sd0["attn_wo"].to(p["W_bo"].device)
                             .cast(p["W_bo"].dtype)).realize()   # door #50:
                             # the pipe warm-seeded from the trained router
                             # (never discard a trained router — the W_bo
                             # zero-saddle's named key)
        if "W_iargs" in p and "W_iargs" not in sd0 and "W_args" in sd0:
            p["W_iargs"].assign(sd0["W_args"].to(p["W_iargs"].device)
                                .cast(p["W_iargs"].dtype)).realize()
        if "W_dargs" in p and "W_dargs" not in sd0 and "W_args" in sd0:
            p["W_dargs"].assign(sd0["W_args"].to(p["W_dargs"].device)
                                .cast(p["W_dargs"].dtype)).realize()
            print("[warm] W_dargs seeded from trained W_args (door #12)", flush=True)
    _frz = [k_ for k_ in (("h_dup", "h_dup_b") if int(os.environ.get("ALG_FREEZE_DUP", "0")) else ())]
    if os.environ.get("ALG_TRAIN_ONLY"):
        # the frozen-parse wake (2026-08-25): train ONLY the named params;
        # everything else frozen — drift eliminated by construction, new
        # organs learn around a byte-identical parse (clean attribution)
        _only = set(os.environ["ALG_TRAIN_ONLY"].split(","))
        _missing = _only - set(p.keys())
        assert not _missing, f"ALG_TRAIN_ONLY names absent params: {_missing}"
        _frz = [k_ for k_ in p if k_ not in _only]
    if _frz: print(f"[freeze] excluded from optimizer: {len(_frz)} params (freeze-or-refold law)", flush=True)
    opt = AdamW([v_ for k_, v_ in p.items() if k_ not in _frz], lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)

    K_B = int(os.environ.get("ALG_BREATH", "1"))
    CLOCK = int(os.environ.get("ALG_CLOCK", "0"))
    TAILS = None
    if CLOCK:
        TAILS = np.zeros_like(sent, dtype=np.uint8)
        for _ri in range(sent.shape[0]):
            _s = sent[_ri]; _i = 0
            while _i < len(_s):
                _j = _i
                while _j + 1 < len(_s) and _s[_j + 1] == _s[_i]: _j += 1
                _h = _i + (_j - _i + 1) // 2
                TAILS[_ri, _h:_j + 1] = 1
                _i = _j + 1
        print(f"[clock] tails ready ({TAILS.mean():.2f} of tokens)", flush=True)
    MASKS = None
    ALT2 = int(os.environ.get("ALG_ALT2", "0"))
    FACTS = np.zeros((n, K_VARS, 4), np.float32) if ALT2 else None
    # MASK HEAD round 2 (apply_mass_thread.py, 2026-09-05): the two
    # dark senses. MASSB = per-var domain-mass banked at mask-prep
    # (same vintage as FACTS), normalized /301 -> [0,1].
    MH_MASS = int(os.environ.get("ALG_MH_MASS", "0"))
    MASSB = np.zeros((n, K_VARS), np.float32) \
        if (ALT2 and MH_MASS) else None
    ATLAS_TAB = ATLAS_IDX = None
    assert not int(os.environ.get("ALG_MH_XPRIOR", "0")) \
        or int(os.environ.get("ALG_MH_ATLAS", "0")), \
        ("ALG_MH_XPRIOR requires ALG_MH_ATLAS=1 (the trajectory "
         "port it retrieves into) — refusing a silently dark prior")
    if int(os.environ.get("ALG_MH_ATLAS", "0")):
        # THE ATLAS FEED: per-row class trajectory pages (the
        # research-manifest loud door; missing file = HARD error,
        # never a silent dark port). Zero page for absent classes.
        from mycelium.step_atlas import load_atlas, atlas_class
        _amp = os.environ.get("MH_ATLAS_MANIFEST",
                              ".cache/RESEARCH_MANIFEST.json")
        _apath = os.environ.get("MH_ATLAS",
                                ".cache/step_atlas_current.npz")
        _atl = load_atlas(_apath, manifest_path=_amp)
        _acls = {c: i for i, c in enumerate(_atl["classes"])}
        _tab = np.ascontiguousarray(
            _atl["means"].transpose(1, 0, 2)).astype(np.float32)
        assert _tab.shape[1] >= K_B and _tab.shape[2] == H_W, \
            (_tab.shape, K_B, H_W)
        ATLAS_TAB = np.concatenate(
            [_tab, np.zeros((1,) + _tab.shape[1:], np.float32)])
        ATLAS_IDX = np.array(
            [_acls.get(atlas_class(smp.get("gen")), len(_acls))
             for smp in samples], np.int64)
        print(f"[mh-atlas] trajectory feed live: {_apath} "
              f"classes={sorted(_acls)} zero-page rows="
              f"{int((ATLAS_IDX == len(_acls)).sum())}/{n}",
              flush=True)
        # THE CROSS-ATLAS PRIOR (apply_cross_prior.py, 2026-09-05):
        # 1 = retrieve for UNKNOWN-class rows only; 2 = retrieve
        # for ALL rows (deployable; gen-label mode = oracle upper
        # bound, training scaffolding). NL0 fills at mask-prep.
        XPRIOR = int(os.environ.get("ALG_MH_XPRIOR", "0"))
        NL0 = None
        if XPRIOR:
            from mycelium.step_atlas import cross_prior
            assert _atl.get("nl_means") is not None, \
                ("ALG_MH_XPRIOR needs the PAIRED atlas (nl chart) "
                 "— re-mine with the paired miner")
            assert K_B > 1, "xprior rides the mask-prep pass"
            NL0 = np.zeros((n, H_W), np.float32)
    if K_B > 1:
        # mask-prep pass: masks from the WARM-STARTED head's own breath-0
        # parses (deployable-from-birth; frozen for training efficiency)
        print("[breath] mask-prep pass ...", flush=True)
        MASKS = np.zeros((n, L_FAC, L_FAC), np.float32)
        # THE MASK-PREP CACHE (apply_maskprep_cache.py, 2026-09-08).
        # Door unset: _maskprep_lookup returns (None, None, None), the
        # iterator is the literal old range, and _maskprep_finish is
        # never called — this pass is the old pass. HIT: only the
        # verification batches run, and the recomputed rows are
        # asserted against the cache below before it is trusted.
        _mp_key, _mp_fp, _mp_cached = _maskprep_lookup(p, n, seed)
        _mp_starts = (range(0, n, 8) if _mp_cached is None
                      else _maskprep_ver_starts(n))
        for s0 in _mp_starts:
            sl = np.arange(s0, min(s0 + 8, n))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            out0 = forward(p, Tensor(states[sl_p].astype(np.float32), dtype=dtypes.float),
                           Tensor(tokmask[sl_p].astype(np.float32), dtype=dtypes.float),
                           Tensor(sent[sl_p].astype(np.int32), dtype=dtypes.int),
                           lsent=(Tensor(gold["lsent"][sl_p].astype(np.float32),
                                         dtype=dtypes.float)
                                  if ALG_LSENT and "lsent" in gold else None))
            o0 = {k: out0[k].realize().numpy() for k in ("fat", "args", "res")}
            MASKS[sl] = build_slot_masks(o0, sent[sl_p])[:len(sl)]
            if ATLAS_TAB is not None and NL0 is not None:
                # breath-0 NL state (the tap; pass-1 == pass-2)
                NL0[sl] = out0["nl0"].realize().numpy()[:len(sl)]
            if FACTS is not None:
                # ALTERNATOR V2 pass-1 commit: the same realized parse the
                # masks come from; facts banked like MASKS (frozen for
                # training efficiency, rebuilt with the head at each prep)
                _ka2 = (("pres", "ftype", "op", "dig")
                        + (("dup",) if "dup" in out0 else ()))
                _oa2 = {**o0, **{k: out0[k].realize().numpy() for k in _ka2}}
                _nv2 = np.array([samples[int(i)].get("n_vars", K_VARS)
                                 for i in sl_p])
                _ma2 = np.array([samples[int(i)].get("m", 0)
                                 for i in sl_p])
                _mo2 = (np.zeros((len(sl_p), K_VARS), np.float32)
                        if MASSB is not None else None)
                FACTS[sl] = alt2_fact_buf(_oa2, sent[sl_p], _nv2,
                                          _ma2,
                                          mass_out=_mo2)[:len(sl)]
                if MASSB is not None:
                    # normalize [0,1]: /301 (values<=300 law)
                    MASSB[sl] = np.clip(_mo2[:len(sl)] / 301.0,
                                        0.0, 1.0)
        if _mp_key is not None:
            _mp_out = _maskprep_finish(
                _mp_key, _mp_fp, _mp_cached, n, _mp_starts,
                {"MASKS": MASKS, "FACTS": FACTS, "MASSB": MASSB,
                 "NL0": (NL0 if ATLAS_TAB is not None else None)})
            MASKS, FACTS, MASSB = (_mp_out["MASKS"], _mp_out["FACTS"],
                                   _mp_out["MASSB"])
            if ATLAS_TAB is not None:
                NL0 = _mp_out["NL0"]
        print(f"[breath] masks ready (mean degree "
              f"{MASKS.sum(-1).mean():.1f}/{L_FAC})", flush=True)
        if ATLAS_TAB is not None and NL0 is not None:
            # retrieval instead of oracle labels (mode semantics in
            # the atlas block above); the b_mha feed needs no change
            _xci, _ = cross_prior(_atl, NL0, return_traj=False)
            _unk = (ATLAS_IDX == len(_acls))
            _rep = _unk if XPRIOR == 1 else np.ones(n, bool)
            _agree = int((_xci[_rep] == ATLAS_IDX[_rep]).sum())
            ATLAS_IDX = np.where(_rep, _xci, ATLAS_IDX)
            print(f"[mh-xprior] mode={XPRIOR}: {int(_rep.sum())}/"
                  f"{n} rows fed RETRIEVED trajectories "
                  f"({int(_unk.sum())} unknown-class; retrieval "
                  f"agrees with gen label on {_agree} of the "
                  f"replaced)", flush=True)
    MG = None
    if int(os.environ.get("ALG_MASK_GOLD", "0")) and MASKS is not None:
        # M3 (2026-08-26, word given; the nazare constitution): TRAINING
        # masks from GOLD wiring — the key-gated truth, the one fully
        # independent ocean; the head's own decoded wirings NEVER
        # masquerade as mask gold (recirculation's sibling). Mixed with
        # the heuristic mask per row (flat mix — no curriculum) so
        # inference never meets an unseen regime.
        _A = gold["args"]; _R = gold["res"].astype(np.int64)
        _P = gold["presence"]
        _n = len(_P)
        MG = np.zeros((_n, L_FAC, L_FAC), np.uint8)
        _roh = np.zeros((_n, L_FAC, K_VARS), np.float32)
        np.put_along_axis(_roh, _R[:, :, None], 1.0, axis=2)
        _roh *= _P[:, :, None]
        for i in range(_n):
            MG[i] = ((_A[i] @ _roh[i].T) > 0) & (_P[i][:, None] > 0) \
                & (_P[i][None, :] > 0)
            np.fill_diagonal(MG[i], 1)
        print(f"[m3] gold-wiring masks ready (mean degree "
              f"{MG.sum(-1).mean():.1f}/{L_FAC}, mix p="
              f"{os.environ.get('ALG_MASK_GOLD_P', '0.5')})", flush=True)

    def fix(a, dt):
        return Tensor(a, dtype=dt).contiguous().realize()
    b_tr = fix(np.zeros((batch, T_ALG, H_TRUNK), np.float32), dtypes.float)
    b_ids = fix(np.zeros((batch, T_ALG), np.int32), dtypes.int) if TRUNK_LORA else None
    b_tk = fix(np.zeros((batch, T_ALG), np.float32), dtypes.float)
    b_se = fix(np.zeros((batch, T_ALG), np.int32), dtypes.int)
    b_mask = fix(np.zeros((batch, L_FAC, L_FAC), np.float32), dtypes.float) \
        if K_B > 1 else None
    b_fact = fix(np.zeros((batch, K_VARS, 4), np.float32), dtypes.float) \
        if ALT2 else None   # ALT2: fixed shape, ALWAYS fed (zeros when no
                            # facts) — the jitted step's signature is stable
    b_mhm = fix(np.zeros((batch, K_VARS, 1), np.float32), dtypes.float) \
        if MASSB is not None else None   # mask-head mass port (b_fact idiom)
    b_mha = fix(np.zeros((batch, ATLAS_TAB.shape[1], H_W), np.float32),
                dtypes.float) if ATLAS_TAB is not None else None
    b_tail = fix(np.zeros((batch, T_ALG), np.float32), dtypes.float) if CLOCK else None
    _GOLD_ALIAS = {"is_lit_f": "is_lit", "refoh": "refvar"}
    _GOLD_OPTIONAL = {"opspan", "arg_dup", "sel", "sign", "y", "digits2",
                      "is_macro", "is_frac", "is_chain"}
    for _tn, _t in TERMINALS.items():
        if not _t["when"](): continue
        for _gk in _t["gold"]:
            if _gk in _GOLD_OPTIONAL: continue
            _gk2 = _GOLD_ALIAS.get(_gk, _gk)
            assert _gk2 in gold, \
                (f"terminal {_tn!r} is ACTIVE but gold {_gk2!r} is missing from "
                 f"the loaded states npz — a stale cache would zero-train it "
                 f"silently (the feed-door fence, GENERALIZED after the "
                 f"bindvec starvation). Re-precompute or inject g_{_gk}.")
    # THE SIDE-DOOR FENCE (audit 2026-08-30): gold consumers living OUTSIDE
    # TERMINALS get the same guard — a stale cache + an active flag would
    # zero-train them silently (the feed dict's "if key in gold" gating
    # omits the feed; the fixed buffer stays zero; grads flow; val looks
    # fine — the opc specimen's exact shape, through the side door).
    for _on2, _gk3 in ((int(os.environ.get("ALG_FTYPES", "4")) >= 9, "is_chain"),
                       (bool(ALG_REF), "refvar"),
                       (bool(ALG_DIAL), "is_ind"),
                       (bool(ALG_VALATT), "valspan")):
        assert (not _on2) or _gk3 in gold, \
            (f"feature ACTIVE but gold {_gk3!r} missing from the states npz "
             f"(the side-door fence) — re-precompute this cache.")
    b_reg = fix(np.zeros((batch,), np.float32), dtypes.float) \
        if int(os.environ.get("ALG_CMT_REG", "0")) else None
    REG = np.load(os.environ.get("ALG_REG_NPY", ".cache/reg_form8.npy")) \
        if b_reg is not None else None
    if REG is not None:
        assert len(REG) == len(samples), \
            f"reg npy {len(REG)} rows vs {len(samples)} samples (desync)"
    b_drop = fix(np.ones((1,), np.float32), dtypes.float) \
        if os.environ.get("BREATH_DROPOUT") else None   # door #52 coin buffer
    b_ls = fix(np.zeros((batch, K_VARS, T_ALG), np.float32), dtypes.float) \
        if ALG_LSENT else None                          # V2: letter partition
    if ALG_CONSUME:
        DEFINER = np.full((len(samples), K_VARS), -1, np.int32)
        ARGM = gold["args"] > 0.5
        for _i in range(len(samples)):
            for _j in range(L_FAC):
                if gold["presence"][_i, _j] > 0:
                    DEFINER[_i, int(gold["res"][_i, _j])] = _j
        PARENTS = np.zeros((len(samples), L_FAC, L_FAC), np.float32)
        for _i in range(len(samples)):
            for _j in range(L_FAC):
                if gold["presence"][_i, _j] > 0 and gold["is_rel"][_i, _j] > 0:
                    for _v in np.where(ARGM[_i, _j])[0]:
                        _d = DEFINER[_i, _v]
                        if _d >= 0 and _d != _j:
                            PARENTS[_i, _j, _d] = 1.0
        print(f"[consume] parent DAG built (mean parents "
              f"{PARENTS.sum(-1)[gold['presence']>0].mean():.2f})", flush=True)
    bg = {}
    for k, shape, dt in (("presence", (L_FAC,), dtypes.float),
                         ("is_lit_f", (L_FAC,), dtypes.float),
                         ("args", (L_FAC, K_VARS), dtypes.float),
                         ("fspan", (L_FAC, T_ALG), dtypes.float),
                         ("vspan", (K_VARS, T_ALG), dtypes.float),
                         *((("refoh", (T_ALG, K_VARS), dtypes.float),) if ALG_REF else ()),
                         *((("is_ind", (L_FAC,), dtypes.float),) if ALG_DIAL else ()),
                         ("ftype", (L_FAC,), dtypes.int),
                         ("op", (L_FAC,), dtypes.int),
                         ("res", (L_FAC,), dtypes.int),
                         ("digits", (L_FAC, N_DIG), dtypes.int),
                         ("sel", (L_FAC,), dtypes.int),
                         ("is_rel", (L_FAC,), dtypes.float),
                         ("is_mod", (L_FAC,), dtypes.float),
                         ("is_sel", (L_FAC,), dtypes.float),
                         ("is_pct", (L_FAC,), dtypes.float),
                         ("is_fdiv", (L_FAC,), dtypes.float),
                         ("arg_dup", (L_FAC,), dtypes.float),
                         # gen-15: OP_APPLY gold buffers (two-terminal law —
                         # without these, h_dig2/W_y leave the graph: None grads)
                         *((("is_macro", (L_FAC,), dtypes.float),
                            ("digits2", (L_FAC, N_DIG), dtypes.int),
                            ("y", (L_FAC,), dtypes.int))
                           if int(os.environ.get("ALG_FTYPES", "4")) >= 7 else ()),
                         *((("is_frac", (L_FAC,), dtypes.float),)
                           if int(os.environ.get("ALG_FTYPES", "4")) >= 8 else ()),
                         *((("is_chain", (L_FAC,), dtypes.float),)
                           if int(os.environ.get("ALG_FTYPES", "4")) >= 9 else ()),
                         *((("valspan", (L_FAC, T_ALG), dtypes.float),)
                           if ALG_VALATT else ()),
                         # E1 (2026-08-03): sign gold buffer — two-terminal
                         # law, the same lesson as the gen-15 comment above
                         # (without it h_sgn leaves the graph: None grad)
                         *((("sign", (L_FAC,), dtypes.float),)
                           if ALG_WIDE else ()),
                         *((("depth", (L_FAC,), dtypes.int),
                            ("term", (L_FAC,), dtypes.float))
                           if int(os.environ.get("ALG_POSCH", "0")) else ()),
                         *((("opc", (len(OPC_CLASSES),), dtypes.int),)
                           if int(os.environ.get("ALG_OPCOUNT", "0")) else ()),
                         *((("bindvec", (L_FAC, int(os.environ.get("ALG_BIND_D", "128"))), dtypes.float),)
                           if 0 < int(os.environ.get("ALG_BINDBUS", "0")) < 5 else ()),
                         *((("bind_ids", (L_FAC, 4), dtypes.int),)
                           if int(os.environ.get("ALG_BINDBUS", "0")) >= 3 else ()),
                         ("query", (), dtypes.int)):
        npdt = np.float32 if dt == dtypes.float else np.int32
        bg[k] = fix(np.zeros((batch,) + shape, npdt), dt)
    if ALG_CONSUME:
        bg["parents"] = fix(np.zeros((batch, L_FAC, L_FAC), np.float32), dtypes.float)
        bg["claimed"] = fix(np.zeros((batch, L_FAC), np.float32), dtypes.float)
        CLAIMED = np.zeros((len(samples), L_FAC), np.float32)  # the persistent
        # ledger: once per fact per RUN — the sharpening subsidy excluded
    if int(os.environ.get("ALG_OPATT", "0")):
        bg["opspan"] = fix(np.zeros((batch, L_FAC, T_ALG), np.float32), dtypes.float)
    assert_terminals(p=p, gold_keys=set(bg.keys()), site="do_train buffers")

    XOUT_TR = int(os.environ.get("ALG_XOUT", "0")) and \
        int(os.environ.get("ALG_RINGS", "0"))
    INV_TR = int(os.environ.get("ALG_INV", "0"))
    INV_PAIRS_ARR = np.load(os.environ["INV_PAIRS"]) if INV_TR else None
    OPATT = int(os.environ.get("ALG_OPATT", "0"))
    OPGOLD = np.load(os.environ["OPATT_GOLD"], mmap_mode="r") if OPATT else None

    _NEWCL = [None]
    @TinyJit
    def step():
        Tensor.training = True
        if TRUNK_LORA:
            from mycelium.llama_loader import _rms_norm as _ll_rms
            _x = HOST.llama_embed[b_ids].detach()
            _rc = HOST.llama_rope_cos[ROPE_OFF:] if ROPE_OFF else HOST.llama_rope_cos
            _rs = HOST.llama_rope_sin[ROPE_OFF:] if ROPE_OFF else HOST.llama_rope_sin
            for _li, _layer in enumerate(HOST.llama_layers):
                _ld = {f"{_nm}_{_ab}": (p[f"lora{_li}_{_nm}_{_ab}"] * (LORA_SCALE if _ab == "B" else 1.0))
                       for _nm in ("wq", "wo", "wdown") for _ab in ("A", "B")
                       if f"lora{_li}_{_nm}_A" in p}
                _x = _layer(_x, _rc, _rs,
                            lora=(_ld if _ld else None))
            _x = _ll_rms(_x, HOST.llama_layers[-1].ffn_norm, HOST.llama_cfg.rms_norm_eps)
            s_tr = _x.cast(dtypes.float)
        else:
            s_tr = b_tr
        if XOUT_TR:
            # ORGAN-2 fire (registered 2026-08-05): two-pass — the first
            # read finds wrong bindings (revoke gold = solver-refuted
            # commits, self-labeled from gold like the commit loss,
            # DETACHED); the second trains under live release dynamics.
            o0 = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail,
                         reg=b_reg)
            ok = ((o0["ftype"].argmax(-1) == bg["ftype"]).float()
                  * (o0["res"].argmax(-1) == bg["res"]).float())
            rv = (bg["presence"] * (1.0 - ok)).detach()
            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, revoke=rv,
                        tail=b_tail, reg=b_reg, fact_buf=b_fact,
                        mh_mass=b_mhm, mh_atlas_traj=b_mha)
        elif int(os.environ.get("NAZ_TRAIN", "0")):
            # NAZARÉ TRAINING (door #55): the organ-2 two-forward pattern —
            # pre-pass yields the intra-pass event field IN-GRAPH (detached);
            # the training pass runs FOCUSED. Training-time criterion
            # (declared per the criterion-key clause, distinct from the
            # read-time dup-aware argpair): argmax-change OR dup-flip on
            # rel-typed present slots, breath-0 vs final.
            o0 = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail)
            _b0 = o0["breaths"][0]
            _relmask = (o0["pres"].squeeze(-1) > 0).float() \
                * (o0["ftype"].argmax(-1) == 0).float()
            _ev = (((_b0["args"].argmax(-1) != o0["args"].argmax(-1)).float()
                    + ((_b0["dup"].squeeze(-1) > 0) != (o0["dup"].squeeze(-1) > 0)).float())
                   .clip(0, 1) * _relmask).detach()
            _bgauth = float(os.environ.get("NAZ_BG", "0.05"))
            _gm = (_bgauth + (1.0 - _bgauth) * _ev).unsqueeze(-1).detach()
            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail,
                        gmod=_gm, fact_buf=b_fact,
                        mh_mass=b_mhm, mh_atlas_traj=b_mha)
        else:
            _bd = os.environ.get("BREATH_DROPOUT")
            o = forward(p, s_tr, b_tk, b_se, slot_mask=b_mask, tail=b_tail,
                        drop=(b_drop if _bd else None), lsent=b_ls, reg=b_reg,
                        fact_buf=b_fact,
                        mh_mass=b_mhm, mh_atlas_traj=b_mha)
        l = loss_fn(o, bg)
        if ALG_CONSUME and "_early" in o:   # support-gated consume-once:
            # any breath claims, each fact pays once; eligibility = the DAG
            _cw = float(os.environ.get("CO_W", "0.5"))
            def _ce(lg, tg):
                return (lg.log_softmax(-1) * -1).gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            def _bce(lg, tg):
                return lg.maximum(0) - lg * tg + (1 + (-lg.abs()).exp()).log()
            _stages = list(o["_early"]) + [o]
            _prev = bg["claimed"] * 1.0   # the ledger seeds prev: paid facts
                                           # never pay again this run
            _P = bg["parents"]              # (B, L, L) adjacency
            _need = _P.sum(-1)
            for _ok in _stages:
                _corr = ((_ok["ftype"].argmax(-1) == bg["ftype"]).float()
                         * (_ok["res"].argmax(-1) == bg["res"]).float())
                _elig = ((_P * _prev.unsqueeze(1)).sum(-1) >= _need - 1e-3).float()
                _first = _corr * _elig * (1.0 - _prev) * bg["presence"]
                _n1 = _first.sum() + 1e-6
                l = l + _cw * ((_ce(_ok["ftype"], bg["ftype"]) * _first).sum()
                               + (_ce(_ok["res"], bg["res"]) * _first).sum()
                               + (_bce(_ok["args"], bg["args"]).mean(-1) * _first).sum()) / _n1
                _prev = (_prev + _first).clip(0, 1)
            _newcl = (_prev - bg["claimed"]).clip(0, 1).realize()
            _NEWCL[0] = _newcl
        if "_early" in o and not ALG_CONSUME:  # deepsup (convicted; kept gated)
            _dw = float(os.environ.get("DEEPSUP_W", "0.3"))   # every breath
            for _ok in o["_early"]:
                l = l + _dw * loss_fn({**_ok, "fat": o["fat"], "vat": o["vat"],
                                       "query": o["query"]}, bg)
        if INV_TR and "fst_s" in o:
            _d = o["fst_s"][0:4:2] - o["fst_s"][1:4:2]
            _pm = bg["presence"][0:4:2].unsqueeze(-1)
            l = l + 0.1 * (_d * _d * _pm).sum() / (_pm.sum() * o["fst_s"].shape[-1] + 1e-6)
        if OPATT:  # dup_staging_cure arm 1: operand-attention TARGET
            _op = bg["opspan"]
            _dm = (_op.sum(-1) > 0).float()
            _opn = _op / (_op.sum(-1, keepdim=True) + 1e-6)
            l = l + float(os.environ.get("OPATT_W", "1")) * ((-(o["fat"] + 1e-9).log() * _opn).sum(-1) * _dm).sum() / (_dm.sum() + 1e-6)
        opt.zero_grad()
        l.backward()
        opt.step()
        return l.realize()

    # HYGIENE (the stack-at-convergence protocol): cosine LR decay + periodic
    # validation on the SMALL test slice (bigtest stays untouched as measurement
    # set) + PICK-BEST-BY-VAL. The overnight constant-lr spike taught this.
    lr_min = lr / 30.0
    val_every = int(os.environ.get("VAL_EVERY", "4000"))

    def _quick_val():
        vs, vst, vtk, vg, vse = load_split_val
        _vaidx = (np.array(
            [_acls.get(atlas_class(smp.get("gen")), len(_acls))
             for smp in vs], np.int64)
            if ATLAS_TAB is not None else None)
        _xpv = (int(os.environ.get("ALG_MH_XPRIOR", "0"))
                if ATLAS_TAB is not None else 0)
        n_ok = n_tot = 0
        for s0 in range(0, len(vs), 8):
            sl = np.arange(s0, min(s0 + 8, len(vs)))
            pad = 8 - len(sl)
            sl_p = np.concatenate([sl, sl[:1].repeat(pad)]) if pad else sl
            _t1 = Tensor(vst[sl_p].astype(np.float32), dtype=dtypes.float)
            _t2 = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
            _t3 = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
            o = forward(p, _t1, _t2, _t3)
            if int(os.environ.get("ALG_ALT2", "0")):
                # ALTERNATOR V2 val two-pass: masked pass-2 + LIVE facts
                # (recomputed from this checkpoint's own pass-1, not the
                # banked FACTS — val measures the deployable cycle)
                _kv = (("pres", "ftype", "op", "dig", "fat", "args", "res")
                       + (("dup",) if "dup" in o else ()))
                _ov = {k: o[k].realize().numpy() for k in _kv}
                _mkv = build_slot_masks(_ov, vse[sl_p].astype(np.int32))
                _nvv = np.array([vs[int(i)].get("n_vars", K_VARS)
                                 for i in sl_p])
                _mav = np.array([vs[int(i)].get("m", 0) for i in sl_p])
                _mov = (np.zeros((len(sl_p), K_VARS), np.float32)
                        if MASSB is not None else None)
                _fbv = alt2_fact_buf(_ov, vse[sl_p], _nvv, _mav,
                                     mass_out=_mov)
                _vmh = (Tensor(np.clip(_mov / 301.0, 0.0, 1.0)
                               [:, :, None].astype(np.float32),
                               dtype=dtypes.float)
                        if _mov is not None else None)
                _vai = (_vaidx[sl_p].copy()
                        if _vaidx is not None else None)
                if _xpv and _vai is not None:
                    # cross-atlas prior on the val cycle (same
                    # retrieval the trainer fed; pass-1 nl0)
                    from mycelium.step_atlas import cross_prior
                    _xcv, _ = cross_prior(
                        _atl, o["nl0"].realize().numpy(),
                        return_traj=False)
                    _rpv = ((_vai == len(_acls)) if _xpv == 1
                            else np.ones(len(_vai), bool))
                    _vai = np.where(_rpv, _xcv, _vai)
                _vat = (Tensor(ATLAS_TAB[_vai],
                               dtype=dtypes.float)
                        if ATLAS_TAB is not None else None)
                o = forward(p, _t1, _t2, _t3,
                            slot_mask=Tensor(_mkv, dtype=dtypes.float),
                            fact_buf=Tensor(_fbv, dtype=dtypes.float),
                            mh_mass=_vmh, mh_atlas_traj=_vat)
            onp = {k: o[k].realize().numpy() for k in
                   (("pres", "ftype", "op", "islit", "dig", "args", "res") + (("dup",) if "h_dup" in p else ()))}
            for bi, i in enumerate(sl):
                i = int(i)
                for j in range(L_FAC):
                    if vg["presence"][i, j] < 0.5:
                        continue
                    n_tot += 1
                    ok = (onp["pres"][bi, j] > 0)
                    ok &= int(onp["ftype"][bi, j].argmax()) == vg["ftype"][i, j]
                    ok &= int(onp["res"][bi, j].argmax()) == vg["res"][i, j]
                    if vg["ftype"][i, j] == 0:
                        ok &= int(onp["op"][bi, j].argmax()) == vg["op"][i, j]
                        gset = set(np.where(vg["args"][i, j] > .5)[0].tolist())
                        if len(gset) == 1 and "dup" in onp:
                            # gen-9: repeated-arg rel — dup bit + top-1 match
                            ok &= bool(onp["dup"][bi, j] > 0)
                            ok &= int(np.argmax(onp["args"][bi, j])) in gset
                        else:
                            top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                            ok &= top2 == gset
                    else:
                        ok &= bool((onp["dig"][bi, j].argmax(-1) ==
                                    vg["digits"][i, j]).all())
                    n_ok += ok
        return n_ok / max(n_tot, 1)

    load_split_val = load_alg("test")
    best_val, best_snap = -1.0, None

    # CURRICULUM=1 (2026-07-10 ablation): coarse->fine by SAMPLE ORDERING.
    # Teeth score from sample artifacts (post-hoc, deployable-blind): oblique
    # (mention span >2 chars) + shuffled (letter != LETTERS[v]) + irrelevant.
    # Phase 1/3: score==0 only; 2/3: score<=1; 3/3: full mix.
    pools = None
    if int(os.environ.get("CURRICULUM", "0")):
        from characterize_survivors import sample_teeth
        score = np.array([int(t["oblique"]) + int(t["shuffled"])
                          + int(t["irrelevant"])
                          for t in (sample_teeth(s_) for s_ in samples)])
        pools = [np.where(score == 0)[0], np.where(score <= 1)[0],
                 np.arange(n)]
        print(f"[curriculum] pools: easy={len(pools[0])} "
              f"mid={len(pools[1])} full={n}", flush=True)

    # RATION (gen-18 charter, inputs #4/#5): hot-phase upweighting of a
    # marked row set — the dilution law's antidote carries mass AND
    # placement (the boundary toll charges in the hot phase).
    ration_w = None
    if os.environ.get("RATION_FILE"):
        r_idx = json.load(open(os.environ["RATION_FILE"]))
        ration_w = np.ones(n, np.float64)
        ration_w[np.array(r_idx, int)] = float(os.environ.get("RATION_W", "1.5"))
        print(f"[ration] {len(r_idx)} rows upweighted x{os.environ.get('RATION_W', '1.5')} in hot phase", flush=True)
        if os.environ.get("RATION_FILE2"):
            r2 = json.load(open(os.environ["RATION_FILE2"]))
            ration_w[np.array(r2, int)] = np.maximum(
                ration_w[np.array(r2, int)], float(os.environ.get("RATION_W2", "1.5")))
            print(f"[ration2] {len(r2)} rows upweighted x{os.environ.get('RATION_W2', '1.5')} (max-combine on overlap)", flush=True)

    # THE STRAW (2026-08-22, Bryce's air-mattress gut; env ALG_STRAW=1):
    # when the front of the diet collapses (easy mass deflates), uniform
    # sampling seals off the back — a channel keeps pulling from rows
    # that still hold air. v1: class-prior x visit-decay weights riding
    # the ration_w fitting (dialect trickle 0.15, wild 1.0, each visit
    # deflates its row by 1/sqrt(1+visits)). Curriculum-grave cited at
    # registration; regime-tagged to trunk-LoRA joint fires. v2 (loss-EMA
    # straw) deferred pending per-row loss emission from step().
    STRAW = int(os.environ.get("ALG_STRAW", "0"))
    if STRAW:
        _sw_base = np.ones(n, np.float64)
        _sw_h = float(os.environ.get("STRAW_HUMAN", "3.0"))
        def _tier(smp):
            g = smp.get("gen")
            if not (isinstance(g, str) and g.startswith("b22")): return 0.15
            return _sw_h if g == "b22human" else 1.0   # straw v2: three tiers
        _sw_wild = np.array([_tier(smp) for smp in samples], np.float64)
        _sw_visits = np.zeros(n, np.float64)
        ration_w = _sw_base * _sw_wild
        print(f"[straw] armed: wild rows {(1.0 == _sw_wild).sum()} @1.0, "
              f"base {(0.15 == _sw_wild).sum()} @0.15, visit-decay live",
              flush=True)

    _pc_mix = float(os.environ.get("ALG_PC_MIX", "0"))
    _pc_assign = None
    if _pc_mix > 0.0:
        # THE PRESSURE MIX (2026-09-06): per-row seal assignment is a
        # DETERMINISTIC hash of the dataset row index (Knuth
        # multiplicative) — stable across epochs/steps/restarts (flat
        # mix law: a constant sealed subpopulation, not a per-epoch
        # coin). The seal needs the shelf road to exist and SC_EVAL
        # unset (else the mode-2 branch bakes OPEN at JIT capture —
        # the exact silent no-op the forensic caught in the champion).
        assert (int(os.environ.get("ALG_BUSGARAGE", "0")) >= 2
                and int(os.environ.get("ALG_SHELF_CIRCLE", "0")) >= 2), (
            "ALG_PC_MIX needs ALG_BUSGARAGE>=2 + ALG_SHELF_CIRCLE>=2 (the per-row "
            "blend rides the mode-2 SC_EVAL forms; _quick_val's OPEN push is "
            "mode>=2 only — at mode 1 a stale _PCV would leak into val)")
        assert not os.environ.get("SC_EVAL", ""), (
            "ALG_PC_MIX with SC_EVAL set would bake the seal shut at "
             "JIT capture (the champion's silent no-op) — unset SC_EVAL; "
             "val forces OPEN by itself")
        _pc_h = ((np.arange(n, dtype=np.uint64) * np.uint64(2654435761))
                 % np.uint64(4294967296)).astype(np.float64) / 4294967296.0
        _pc_assign = (_pc_h < _pc_mix).astype(np.float32)
        globals()["_PCV"] = Tensor(
            np.zeros((batch, 1, 1), np.float32)).contiguous().realize()
        print(f"[pressure] mixed-seal armed: share={_pc_mix} -> "
              f"{int(_pc_assign.sum())}/{n} rows sealed (stable "
              f"index-hash); live-wire="
              f"{os.environ.get('ALG_PC_LIVE', '1')}", flush=True)
    t0 = time.time()
    for s in range(steps):
        cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * s / steps))
        opt.lr.assign(Tensor([cur_lr], dtype=dtypes.float)).realize()
        pool = (pools[min(3 * s // steps, 2)] if pools is not None
                else np.arange(n))
        if STRAW:
            ration_w = _sw_wild / np.sqrt(1.0 + _sw_visits)
        if ration_w is not None and (STRAW or cur_lr > 0.5 * (lr + lr_min)):
            pw = ration_w[pool] / ration_w[pool].sum()
            idx = rng.choice(pool, batch, replace=False, p=pw)
        else:
            idx = rng.choice(pool, batch, replace=False)
        if STRAW:
            _sw_visits[idx] += 1.0
        if INV_TR:  # inv-fire v2: 2 pairs (pos 0-3) + carrier rows (pos 4-7);
            # pairs from INV_PAIRS side file — the carrier mix rides (the
            # substrate lesson: never train on the patch alone)
            pk = rng.choice(len(INV_PAIRS_ARR), 2, replace=False)
            idx = np.concatenate([INV_PAIRS_ARR[pk].reshape(-1),
                                  rng.choice(n, batch - 4, replace=False)])
        if not TRUNK_LORA:   # audit #15: b_tr is dead under the in-graph trunk
            b_tr.assign(Tensor(states[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        _rl = [b_tk.assign(Tensor(tokmask[idx].astype(np.float32), dtype=dtypes.float).contiguous()),
               b_se.assign(Tensor(sent[idx].astype(np.int32), dtype=dtypes.int).contiguous())]
        if TRUNK_LORA:
            _rl.append(b_ids.assign(Tensor(IDS_ALL[idx].astype(np.int32), dtype=dtypes.int).contiguous()))
        Tensor.realize(*_rl)   # perf audit #2: one combined schedule, not N dispatches
        if ALG_CONSUME:
            bg["parents"].assign(Tensor(PARENTS[idx], dtype=dtypes.float).contiguous()).realize()
            bg["claimed"].assign(Tensor(CLAIMED[idx], dtype=dtypes.float).contiguous()).realize()
        if b_ls is not None:
            b_ls.assign(Tensor(gold["lsent"][idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        if b_mask is not None:
            _mfeed = MASKS[idx]
            if MG is not None:
                _coin = np.random.RandomState(1000 + s).random(len(idx)) \
                    < float(os.environ.get("ALG_MASK_GOLD_P", "0.5"))
                _mfeed = _mfeed.copy()
                _mfeed[_coin] = MG[idx][_coin].astype(np.float32)
            b_mask.assign(Tensor(_mfeed, dtype=dtypes.float).contiguous()).realize()
        if b_fact is not None:
            b_fact.assign(Tensor(FACTS[idx], dtype=dtypes.float).contiguous()).realize()
        if b_mhm is not None:
            b_mhm.assign(Tensor(MASSB[idx][:, :, None],
                                dtype=dtypes.float).contiguous()).realize()
        if b_mha is not None:
            b_mha.assign(Tensor(ATLAS_TAB[ATLAS_IDX[idx]],
                                dtype=dtypes.float).contiguous()).realize()
        if b_tail is not None:
            b_tail.assign(Tensor(TAILS[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        if b_reg is not None:
            b_reg.assign(Tensor(REG[idx].astype(np.float32), dtype=dtypes.float).contiguous()).realize()
        if b_drop is not None:
            b_drop.assign(Tensor(np.array(
                [1.0 if rng.rand() >= float(os.environ["BREATH_DROPOUT"]) else 0.0],
                np.float32), dtype=dtypes.float).contiguous()).realize()
        feed = {"presence": gold["presence"][idx], "is_lit_f": gold["is_lit"][idx],
                **({"opspan": OPGOLD[idx].astype(np.float32)} if OPATT else {}),
                "args": gold["args"][idx], "fspan": gold["fspan"][idx],
                "vspan": gold["vspan"][idx], "ftype": gold["ftype"][idx],
                **({"refoh": (np.eye(K_VARS, dtype=np.float32)[
                        np.maximum(gold["refvar"][idx].astype(np.int32), 0)]
                        * (gold["refvar"][idx] >= 0)[..., None])} if ALG_REF and "refvar" in gold else {}),
                **({"is_ind": gold["is_ind"][idx]} if ALG_DIAL and "is_ind" in gold else {}),
                "op": gold["op"][idx], "res": gold["res"][idx],
                "digits": gold["digits"][idx], "query": gold["query"][idx],
                "sel": gold["sel"][idx], "is_rel": gold["is_rel"][idx],
                "is_mod": gold["is_mod"][idx], "is_sel": gold["is_sel"][idx],
                "is_pct": gold["is_pct"][idx], "is_fdiv": gold["is_fdiv"][idx],
                **({"is_chain": gold["is_chain"][idx]}
                   if "is_chain" in gold
                   and int(os.environ.get("ALG_FTYPES", "4")) >= 9 else {}),
                **({"valspan": gold["valspan"][idx]} if ALG_VALATT and "valspan" in gold else {}),
                "arg_dup": (gold["arg_dup"][idx] if "arg_dup" in gold
                            else np.zeros_like(gold["is_rel"][idx])),
                **({"is_macro": gold["is_macro"][idx],
                    "digits2": gold["digits2"][idx],
                    "y": gold["y"][idx]} if "is_macro" in bg else {}),
                **({"is_frac": gold["is_frac"][idx]} if "is_frac" in bg else {}),
                **({"sign": gold["sign"][idx]} if "sign" in bg else {})}
        # THE FEED DOOR (2026-08-25): the buffer list and the feed dict are
        # TWO doors — a terminal registered in one but not the other trains
        # against silent zeros (the opc/posch specimen: emission + gold
        # buffer both present, grads flowing, gold all-zero). Any bg buffer
        # with a same-named gold array rides automatically; new terminals
        # can never miss the feed again.
        for k in bg:
            if k not in feed and k in gold:
                feed[k] = gold[k][idx]
        for k, v in feed.items():
            npdt = np.float32 if bg[k].dtype == dtypes.float else np.int32
            bg[k].assign(Tensor(v.astype(npdt), dtype=bg[k].dtype).contiguous()).realize()
        if int(os.environ.get("ALG_SHELF_CIRCLE", "0")) >= 2:
            _sevb = globals().get("_SEV")
            if _sevb is not None:      # the pulse: reseal per step
                _sevb.assign(Tensor([1.0 if np.random.rand() <
                                     float(os.environ.get("SC_P", "0.5"))
                                     else 0.0],
                                    dtype=_sevb.dtype)).realize()
        if _pc_assign is not None:
            globals()["_PCV"].assign(Tensor(
                _pc_assign[idx].reshape(-1, 1, 1),
                dtype=globals()["_PCV"].dtype)).realize()
        lv = step()
        if ALG_CONSUME and _NEWCL[0] is not None:
            CLAIMED[idx] = np.clip(CLAIMED[idx] + _NEWCL[0].numpy(), 0, 1)
        if s % 500 == 0 or s == steps - 1:
            v = float(lv.numpy())
            assert np.isfinite(v)
            print(f"  step {s:5d} loss={v:.4f} lr={cur_lr:.1e} "
                  f"({(time.time()-t0)/(s+1):.2f}s/step)", flush=True)
        if (s + 1) % val_every == 0 or s == steps - 1:
            if int(os.environ.get("ALG_SHELF_CIRCLE", "0")) >= 2:
                os.environ["SC_EVAL"] = "0"     # val compares OPEN mode
            fv = _quick_val()
            if int(os.environ.get("ALG_SHELF_CIRCLE", "0")) >= 2:
                os.environ.pop("SC_EVAL", None)
            mark = ""
            if fv > best_val:
                best_val = fv
                best_snap = {k: t.detach().numpy().copy() for k, t in p.items()}
                mark = "  <-- BEST"
            print(f"  [val @{s+1}] fac-exact-proxy={fv:.4f}{mark}", flush=True)
        # SNAP_EVERY (gut #57's snapshot rider): periodic trajectory ckpts for
        # the wobble census. Snapshots NEVER read bar fixtures (val picks,
        # bars judge); they exist so checkpoint variance can be measured.
        snap_every = int(os.environ.get("SNAP_EVERY", "0"))
        if snap_every and (s + 1) % snap_every == 0:
            sp = ALG_CKPT.replace(".safetensors", f"_s{s+1}.safetensors")
            safe_save(p, sp)
            print(f"  [snap @{s+1}] -> {sp}", flush=True)
    if best_snap is not None and TRUNK_LORA:
        # audit 2026-08-22 #3: _quick_val reads PURE-trunk states — under
        # LoRA it is a mismatched proxy; save FINAL-step params instead
        # (snapshots carry the trajectory for selection)
        print("[train] TRUNK_LORA: best-by-val restore SKIPPED (proxy is "
              "pure-trunk; final-step params saved)", flush=True)
    elif best_snap is not None:
        for k in p:
            p[k].assign(Tensor(best_snap[k], dtype=p[k].dtype)).realize()
        print(f"[train] restored BEST ckpt (val {best_val:.4f})", flush=True)
    safe_save(p, ALG_CKPT)
    print(f"[train] saved {ALG_CKPT} "
          f"({'final-step (TRUNK_LORA)' if int(os.environ.get('TRUNK_LORA', '0')) else 'best-by-val'})",
          flush=True)


# ===========================================================================
# SELFTEST
# ===========================================================================

def selftest():
    os.environ.setdefault("DEV", "CPU")
    from tinygrad import Tensor, dtypes
    p = build_params(0)
    B = 2
    o = forward(p, Tensor(np.random.RandomState(0).randn(B, T_ALG, H_TRUNK).astype(np.float32) * .1),
                Tensor(np.ones((B, T_ALG), np.float32)),
                Tensor(np.zeros((B, T_ALG), np.int32), dtype=dtypes.int))
    assert o["args"].shape == (B, L_FAC, K_VARS) and o["query"].shape == (B, K_VARS)
    g = {"presence": Tensor(np.ones((B, L_FAC), np.float32)),
         "is_lit_f": Tensor(np.zeros((B, L_FAC), np.float32)),
         "args": Tensor(np.zeros((B, L_FAC, K_VARS), np.float32)),
         "fspan": Tensor(np.ones((B, L_FAC, T_ALG), np.float32)),
         "vspan": Tensor(np.ones((B, K_VARS, T_ALG), np.float32)),
         "ftype": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
         "op": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
         "res": Tensor(np.zeros((B, L_FAC), np.int32), dtype=dtypes.int),
         "digits": Tensor(np.zeros((B, L_FAC, N_DIG), np.int32), dtype=dtypes.int),
         "query": Tensor(np.zeros((B,), np.int32), dtype=dtypes.int)}
    l = loss_fn(o, g)
    lv = float(l.numpy())
    assert np.isfinite(lv)
    l.backward()
    assert float(p["W_res"].grad.abs().max().numpy()) > 0
    print(f"  [OK] forward/loss/backward (loss={lv:.3f}); pointer grads flow")
    # decode round trip
    onp = {"pres": np.full((L_FAC,), -9., np.float32),
           "ftype": np.zeros((L_FAC, 2), np.float32),
           "op": np.zeros((L_FAC, 2), np.float32),
           "islit": np.full((L_FAC,), -9., np.float32),
           "dig": np.zeros((L_FAC, N_DIG, 10), np.float32),
           "args": np.full((L_FAC, K_VARS), -9., np.float32),
           "res": np.zeros((L_FAC, K_VARS), np.float32),
           "query": np.zeros((K_VARS,), np.float32)}
    onp["pres"][0] = 9.
    onp["args"][0, [2, 5]] = 9.
    onp["res"][0, 7] = 9.
    onp["query"][5] = 9.
    facs, q = decode(onp)
    assert facs == [{"ftype": "rel", "op": "add", "args": [2, 5], "result": 7}]
    assert q == 5
    print("  [OK] decode round trip (rel factor + query pointer)")
    # gold builder on a real corpus line
    samples, ids, mask, offsets = tokenize(ALG_TEST)
    g2 = build_gold(samples[:2], offsets[:2])
    assert g2["vspan"][0].sum() > 0 and g2["fspan"][0].sum() > 0
    print("  [OK] gold builder on real corpus (mentions + factor spans)")
    print("[selftest] PASS")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--errors", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
    elif args.precompute:
        do_precompute()
    elif args.train:
        do_train(int(os.environ.get("STEPS", "8000")),
                 float(os.environ.get("LR", "3e-4")),
                 int(os.environ.get("BATCH", "8")),
                 int(os.environ.get("SEED", "0")))
    elif args.eval:
        do_eval()
    elif args.errors:
        do_errors()
    else:
        # no-silent-fallbacks law (2026-09-01): a modeless invocation once
        # burned a whole training chain — argparse help exits 0 and set -e
        # sails past. Chains must die loudly here.
        ap.print_help()
        ap.error("no mode flag (--train/--eval/--precompute/--selftest/"
                 "--errors) — refusing to exit 0")


if __name__ == "__main__":
    main()
