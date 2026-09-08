"""apply_polar_sink.py — THE KITCHEN SINK, phase 3 of the polar-waist
generation (2026-09-08). A STAGED patch: it anchors into the APPLIED head
(scripts/phase1_algebra_head.py, which already carries the polar waist and
its sextet, apply_polar_waist.py 2026-09-07) and under --check writes
NOTHING. The lead applies.

WHAT THIS PATCH ADDS — two organs bolted onto the polar DIRECTION u, both
BYTE-INERT when their door is unset (proved by scripts/polar_sink_smoke.py
against the staged source, with ALG_POLAR=1 and with ALG_POLAR unset):

(A) ALG_POLAR_D — THE CONTENT-PLANE WAIST ("expand & collapse x7").
    Once per breath, after the sextet has turned u, the 192 CONTENT
    planes' 384 dims are collapsed and expanded

        c' = c @ W_down @ W_up      W_down (384, d), W_up (d, 384)

    with d = ALG_POLAR_D (0 = off; 128 = the twin's width). The 64 CLOCK
    planes' 128 dims are NOT in the bottleneck's domain OR its range, so
    the waist can never fight the rotation — the whole point of the split
    (the sextet is a frozen, ungained clock; a learned map that could
    read or write the clock planes would be able to cancel it, and the
    clock would stop being a guarantee).
    BIRTH = THE CHAMPION'S OWN MANIFOLD. W_down = V, W_up = V^T with V
    the top-d principal directions of the content dims of the champion's
    dumped polar directions (.cache/polar_states_{mint,wild}.npz, key
    `u` = breaths_u under fedon242, ALG_POLAR=1, open regime), pooled
    over both fixtures, all seven breaths and all 24 slots, centered on
    the mean. So birth is the ORTHOGONAL PROJECTION onto the subspace
    the champion already lives in — not a fresh random organ dropped into
    a warm continuation (the refuted-cell map: blur, not compute). The
    init is BANKED (.cache/polar_waist_init_d{d}.npz, built by
    scripts/polar_waist_init.py) and the head LOADS it; a missing file is
    a HARD ERROR, never a silent random init (ALG_POLAR_D_INIT=path
    overrides the default name).
      d=64  var kept 0.8140   d=128  0.9188   d=256  0.9820   (measured)

(B) ALG_POLAR_EM — THE E&B COUPLING on the clock planes, along the slot-
    mask lanes. On each CLOCKED plane p, with (x_i, y_i) the plane's
    coordinates for slot i, ONE discrete Maxwell-like exchange step per
    breath across neighbouring slots:

        x_i += kappa * sum_j Mhat_ij (y_j - y_i)
        y_i -= kappa * sum_j Mhat_ij (x_j - x_i)

    where M is THE SLOT MASK ACTUALLY IN FORCE THIS BREATH (`_sm_kb` —
    the same tensor the breath's own slot attention closes sc2 with,
    MASKRE re-formation included; not a second copy, the meter-divergence
    law), row-normalized by the neighbour count so kappa is dimensionless.
    In complex form z = x + iy this is z <- (I - i*kappa*L)z with L the
    row-normalized graph Laplacian: the two field components exchange
    ALONG THE OPEN LANES — a slot's phase is nudged by the phase its
    neighbours are carrying. kappa = ALG_POLAR_EM is FIXED, declared
    once, and NOT LEARNABLE: a zero-born learnable gain is exactly the
    pattern that died in fed_mx_hg (abs-mean 0.023 at its best, SHRANK
    to 0.013 under the cooker — a whisper the state never heard).
    CONTENT PLANES ARE UNTOUCHED (their delta is multiplied by an exact
    0.0 and added: bitwise passthrough).

NORMS. Both organs are NORM-CHANGING, and both restore the norm OF THEIR
OWN BLOCK afterwards through ONE where-gated organ (`_polar_keepnorm`):
the waist restores the content block's norm, the EM step restores the
clock block's norm. The consequence is exactly what the spec asks for and
what the birth proofs pin:
  * ||u|| == 1 per slot still holds at every breath (both blocks keep
    their norms, so the sum of squares is unchanged);
  * the clock planes are BITWISE what the sextet left them when only the
    waist is open, and the content planes are BITWISE unchanged when only
    the EM step is open — the two doors cannot contaminate each other,
    and neither can silently rescale the other's channel.
A single GLOBAL renormalization would have been the other reading of
"re-normalize the whole u"; it is rejected here BECAUSE it fails both
bit-identity proofs — and, worse, it would hand the content bottleneck a
lever on the clock's share of the energy every breath (as c collapses,
the global rescale would AMPLIFY the clock), which is the bottleneck
fighting the rotation by the back door. Block-local renormalization is
the same statement (||u|| = 1) with that lever removed. r is untouched
by both organs (it is the diagnostic register; never supervised).

PLACEMENT (asserted structurally, not in prose): inside the existing
ALG_POLAR block of breath_step, AFTER u is formed and rotated by the
sextet, in the order  rotate -> EM -> waist -> renorm(s) -> r*u join,
so every organ downstream still reads ONE state in ONE frame.

DOORS (all inert unset; all printed once at build_params):
  ALG_POLAR_D=128        content-waist width (0 = off)
  ALG_POLAR_D_INIT=path  override .cache/polar_waist_init_d{d}.npz
  ALG_POLAR_EM=0.1       the coupling kappa (0 = off; FIXED, not learned)
Both REQUIRE ALG_POLAR=1 (they ride on u) — asserted loudly at import.

NEW PARAMETERS: two, and only under ALG_POLAR_D>0 —
  polar_wd (384, d)  polar_wu (d, 384)     = 2*384*d entries
  d=128 -> 98,304 parameters (d=64 -> 49,152; d=256 -> 196,608).
ALG_POLAR_EM adds ZERO parameters by construction.

--check: loads the file, asserts every anchor present and unique, builds
the would-be result, ast-parses it, runs the structural asserts and the
symtable free-variable audit, and writes NOTHING. PS_TARGET may point at
a copy (rehearsal); default is scripts/phase1_algebra_head.py.
The CPU proofs live in scripts/polar_sink_smoke.py, which stages this
patch in memory exactly as polar_birth_smoke.py stages phase 1/2.
"""
import ast
import builtins
import json
import os
import symtable
import sys

fn = os.environ.get("PS_TARGET", 'scripts/phase1_algebra_head.py')
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

# ---------------------------------------------------------------- guards
assert 'ALG_POLAR_D' not in s and '_polar_sink' not in s, \
    "the polar sink is already present — patch was applied; refuse (idempotence)"
assert 'ALG_POLAR = int(os.environ.get("ALG_POLAR", "0"))' in s, \
    "the POLAR WAIST is not applied to this head — phase 3 rides on phase 1/2"
assert 'def _polar_ru(x, g):' in s and 'def _polar_ru_join(u, r, g):' in s, \
    "the polar organs are missing — wrong vintage of the head"
assert '_pol_u = _rot2(_pol_u,' in s, \
    "the sextet's state rotation is missing — wrong vintage"
assert '    _sm_kb = slot_mask\n' in s, \
    "the effective slot mask `_sm_kb` is missing — the EM step reuses it"
assert 'p["polar' not in s, "polar parameters already exist — wrong vintage"

# ------------------------------------------- the band file must exist (v2)
BANDS = os.environ.get("ALG_POLAR_BANDS", ".cache/polar_bands.json")
assert os.path.exists(BANDS), \
    f"band allocation {BANDS} missing — the head READS it (spec S2)"
_bj = json.load(open(BANDS))
assert _bj["waist"] == 512 and _bj["n_planes"] == 256, BANDS
_clk = set()
for _w in _bj["wheels"]:
    for _p in _w["planes"]:
        assert _p not in _clk, "band overlap in the json"
        _clk.add(int(_p))
assert len(_clk) == int(_bj["n_clocked"]) == 64, \
    f"{BANDS}: expected the 64-plane v2 allocation (ruling 2026-09-08)"
N_CONTENT = 256 - len(_clk)
assert N_CONTENT == int(_bj["n_content"]) == 192, BANDS
C_DIMS = 2 * N_CONTENT

# ------------------- the banked PCA birth must exist for the twin's width
_D_DEFAULT = 128
_INIT = f".cache/polar_waist_init_d{_D_DEFAULT}.npz"
assert os.path.exists(_INIT), (
    f"{_INIT} missing — the content waist is born as the PCA projection "
    f"of the champion's own manifold, never random; run "
    f"scripts/polar_waist_init.py first")

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# ===========================================================================
# 1. MODULE LEVEL — the two doors and the four organs they share. Placed
#    immediately after the phase-1/2 polar organs so ONE block owns the
#    whole polar parametrization (and _polar_tables stays the single band
#    authority: _polar_sink derives its dim tables from ITS wheel_of).
# ===========================================================================
patch(1, "module: ALG_POLAR_D / ALG_POLAR_EM doors + sink dim tables + "
         "the block-renorm, waist and E&B organs",
      '''    return (u.reshape(u.shape[0], u.shape[1], g, -1) * r).reshape(u.shape)
SENT_MAX = 32''',
      '''    return (u.reshape(u.shape[0], u.shape[1], g, -1) * r).reshape(u.shape)


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
assert ALG_POLAR or not (POLAR_D or POLAR_EM), \\
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
        assert 0 < len(_cp) < _P, \\
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
SENT_MAX = 32''')

# ===========================================================================
# 2. build_params — the two parameters (PCA birth, loaded LOUDLY) and the
#    sink door's one-line print. One anchor: `p` and the `t()` helper are
#    both in scope at the end of the function.
# ===========================================================================
patch(2, "build_params: polar_wd/polar_wu born as the champion's own top-d "
         "content subspace (banked PCA init, hard error if absent) + door",
      '''    return p


NB_STAMPS = None''',
      '''    if ALG_POLAR and POLAR_D:
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
        assert _wd0.shape == (len(_cdim), POLAR_D) \\
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


NB_STAMPS = None''')

# ===========================================================================
# 3. breath_step — the two organs, in the pinned order, inside the existing
#    ALG_POLAR block: rotate -> EM -> waist -> r*u join.
# ===========================================================================
patch(3, "breath_step: E&B coupling then the content waist, between the "
         "sextet's turn and the r*u recomposition",
      '''            _pol_u = _rot2(_pol_u,
                           _Tp(_pdc[kb - 1], dtype=_dp.float),
                           _Tp(_pds[kb - 1], dtype=_dp.float))
        cur = _polar_ru_join(_pol_u, _pol_r, _pg)''',
      '''            _pol_u = _rot2(_pol_u,
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
        cur = _polar_ru_join(_pol_u, _pol_r, _pg)''')

for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# ===========================================================================
# STRUCTURAL ASSERTS on the would-be module (cheap, no import, no GPU)
# ===========================================================================
# -- the organs exist exactly once and every caller uses THEM (meter law)
for _d in ('def _polar_sink(', 'def _polar_keepnorm(', 'def _polar_waist(',
           'def _polar_em('):
    assert s.count(_d) == 1, f"{_d} not defined exactly once"
assert s.count('_polar_waist(') == 2 and s.count('_polar_em(') == 2, \
    "each organ must have exactly its def + ONE call site in breath_step"
assert s.count('_polar_keepnorm(') == 3, \
    ("the block renorm: def + the waist's call + the EM step's call and "
     "nothing else (ONE renorm organ, two callers)")
# -- ONE band authority: the sink derives its dims from _polar_tables()
assert s.count('_polar_tables()') == 6, \
    ("_polar_tables must gain exactly ONE new caller (_polar_sink) on top "
     "of phase 1/2's five — no second band table anywhere")
assert '_wof = _polar_tables()[4]' in s, \
    "the sink must read _polar_tables' wheel_of, never re-open the band file"
assert 'polar_bands' not in s.split('def _polar_sink(')[1].split('def _polar_keepnorm(')[0], \
    "the sink must NOT open the band file itself — one reader, one table"
# -- ORDERING (the phase-1 idiom): rotate -> EM -> waist -> join, all
#    inside the existing ALG_POLAR block, before the notebook/append/garage
_i_pol = s.index('    _pol_r = None\n    if ALG_POLAR:')
_i_rot = s.index('            _pol_u = _rot2(_pol_u,')
_i_em = s.index('            _pol_u = _polar_em(_pol_u, _sm_kb, POLAR_EM)')
_i_wst = s.index('            _pol_u = _polar_waist(_pol_u, p, state)')
_i_join = s.index('        cur = _polar_ru_join(_pol_u, _pol_r, _pg)')
_i_msk = s.index('\n    _sm_kb = slot_mask\n')
_i_nb = s.index('    if ALG_NOTEBOOK:\n        _nb.append(')
_i_app = s.index('    breaths.append(cur)')
assert _i_msk < _i_pol < _i_rot < _i_em < _i_wst < _i_join < _i_nb < _i_app, \
    ("sink misplaced: the order is  _sm_kb defined -> polar block -> "
     "sextet rotation -> E&B -> content waist -> r*u join -> ink/append")
# -- the EM step reuses the breath's OWN mask tensor, never a rebuild
assert '_polar_em(_pol_u, _sm_kb, POLAR_EM)' in s and \
    'build_slot_masks' not in s[_i_pol:_i_join], \
    "the E&B coupling must reuse _sm_kb (the mask in force this breath)"
# -- kappa is FIXED: no parameter, no zero-born gain, no schedule
assert 'p["polar_em' not in s and "p['polar_em" not in s, \
    "ALG_POLAR_EM adds ZERO parameters (fixed kappa — fed_mx_hg's grave)"
# -- exactly TWO new parameters, both under the ALG_POLAR_D door
import re as _re                                        # noqa: E402
_pkeys = _re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(', s)
assert sorted(k for k in _pkeys if k.startswith('polar')) == \
    ['polar_wd', 'polar_wu'], \
    f"exactly two new parameters expected, found {sorted(set(_pkeys))[:6]}"
assert s.count('p["polar_wd"] = t(_wd0)') == 1 and \
    s.count('p["polar_wu"] = t(_wu0)') == 1, \
    "the two parameters must be created exactly once, from the banked init"
# -- no silent random init: the banked PCA birth or a hard error
assert 'assert os.path.exists(_pwi)' in s and 'rng.randn' not in \
    s[s.index('if ALG_POLAR and POLAR_D:'):s.index('p["polar_wu"] = t(_wu0)')], \
    "the waist must load its PCA birth or hard-error — never random-init"
# -- the clock planes are outside the waist's domain AND range
assert '_cn = (u @ _wd) @ _wu' in s and 'u * _gk + _cn' in s, \
    ("the waist must scatter through the one-hot constant (exact zeros on "
     "clock dims) and add to the gated state — not touch the clock")
# -- the renormalization is where-gated TWICE, no epsilon denominator
assert '_n0 = _p0.where(_p0.where(_s0, 1.0).sqrt(), 0.0)' in s and \
    '_n1 = _p1.where(_p1.where(_s1, 1.0).sqrt(), 1.0)' in s, \
    "the block renorm must be where-gated twice (sqrt AND denominator)"
assert '_dn = _pg.where(_dg, 1.0)' in s, \
    "the EM degree denominator must be where-gated (isolated slots exist)"
assert 'dtypes.float32' not in s, "no float32 dtype literal anywhere"
assert 'POLAR_EPS' not in s and '+ 1e-6)' not in \
    s[s.index('def _polar_keepnorm('):s.index('def _polar_waist(')], \
    "an epsilon denominator is the refuted cell here — use the where-gate"
# -- r is untouched by both organs (the diagnostic register)
_body = s[_i_rot:_i_join]
assert '_pol_r' not in _body, \
    "the sink must not touch the radius channel (diagnostic register)"
# -- env-inertness: every new runtime line sits under its own door
for _line in ('_pol_u = _polar_em(_pol_u, _sm_kb, POLAR_EM)',
              '_pol_u = _polar_waist(_pol_u, p, state)',
              'p["polar_wd"] = t(_wd0)'):
    _blk = s[:s.index(_line)]
    assert 'POLAR_D' in _blk[-1800:] or 'POLAR_EM' in _blk[-1800:], \
        f"unguarded sink line (no door above): {_line}"
assert 'assert ALG_POLAR or not (POLAR_D or POLAR_EM)' in s, \
    "both doors must refuse to be set without ALG_POLAR"

# ===========================================================================
# THE SYMTABLE FREE-VARIABLE AUDIT (the apply_polar_waist.py idiom)
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
           '_polar_groups', '_polar_tables', '_polar_ru', '_polar_ru_join',
           '_polar_sink', '_polar_keepnorm', '_polar_waist', '_polar_em')
_seen_children = set()
for child in mod_tbl.get_children():
    if child.get_name() in AUDITED:
        _seen_children.add(child.get_name())
        audit(child, child.get_name())
assert _seen_children == set(AUDITED), \
    f"audit missed {sorted(set(AUDITED) - _seen_children)}"

# ===========================================================================
# REPORT
# ===========================================================================
print(f"[polar sink] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines, 0 deleted):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
print(f"[polar sink] bands {BANDS} v{_bj['version']}: {len(_clk)} clocked "
      f"planes ({2 * len(_clk)} dims, the sextet's — the waist never "
      f"touches them) / {N_CONTENT} content planes ({C_DIMS} dims, the "
      f"waist's domain and range)")
print(f"[polar sink] NEW params: 2 tensors, polar_wd ({C_DIMS}, d) + "
      f"polar_wu (d, {C_DIMS}) = {2 * C_DIMS}*d entries — "
      f"{2 * C_DIMS * 128} at d=128 (d=64: {2 * C_DIMS * 64}; d=256: "
      f"{2 * C_DIMS * 256}); ALG_POLAR_EM adds ZERO (fixed kappa)")
print(f"[polar sink] birth: {_INIT} (PCA of the champion's own manifold; "
      f"a missing file is a HARD ERROR, never a random init)")
print("[polar sink] symtable free-var audit PASS "
      f"({', '.join(AUDITED)})")
print("[polar sink] contract: ALG_POLAR_D unset AND ALG_POLAR_EM unset = "
      "byte-identical to the applied head (with ALG_POLAR=1 and with "
      "ALG_POLAR unset); set = a NEW GENERATION, measured as a twin")
if CHECK:
    print("[polar sink] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[polar sink] APPLIED ({fn}); ast OK — run the eq pre/post "
          "A/B/C gate (all sink doors unset) + .cache/pc_row_smoke.py + "
          "scripts/polar_sink_smoke.py before trusting")
