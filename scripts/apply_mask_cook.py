"""apply_mask_cook.py — THE MASK COOKER, staged patch (2026-09-08;
spec docs/mask_cooker_spec.md, registered by THE ORGAN CENSUS; the run
needs the word). A STAGED patch: it anchors into the APPLIED head
(scripts/phase1_algebra_head.py at a0d7b75 — polar waist + kitchen sink
+ pressure mix + mask-prep cache + census hooks all present) and under
--check writes NOTHING. The lead applies. MC_TARGET may point at a copy
(rehearsal); default scripts/phase1_algebra_head.py.

WHY (the census, ledger 2026-09-08): the mask head — 2.0M params, "the
player" — injects at ~1e-5x of the slot-score band; its PRE-gain organ
is at 1-9% of it and its gain fell 0.02 -> 0.0011 -> 0.0002 while its
pre-gain output GREW 2-5x. Three structural reasons, all in the code:
two zero-born output doors, a gain the loss can close, and the OPEN-ONLY
constitution AS IMPLEMENTED — `_mb = gain*(softplus(raw)-softplus(0))
*_sm_kb`: the head may only re-weight lanes the baseline already opened,
and the baseline opens 62-67% of them. THE MANDATORY-ROAD LAW: an organ
enters as a road with no bypass, never as an invitation through a gain.

THE BUILD (spec S1-2). On a SHARE of TRAINING ROWS (`ALG_MASK_COOK`,
dose 0.15) the baseline slot mask stops being the road:

  * the CLOSE that the baseline imposes on the slot mixer is LIFTED for
    sealed rows (all factor lanes structurally open; scratch columns keep
    the baseline's policy — FED item 6's raising law is not repealed by
    this patch), and
  * the mask head's RAW logits become the mask over ALL lanes, as a soft
    gate `sc2 + log(sigmoid(raw))` = `sc2 - softplus(-raw)`, UNGATED by
    mh_gain and UNMULTIPLIED by the baseline, and
  * the SKELETON of the severed baseline survives as a FLOOR: on the
    skeleton's lanes (ALG_MASK_COOK_SKEL=self -> the diagonal;
    =sentence -> the breath's own same-sentence agreement, self
    included) the gate can never fall below its BIRTH value, so a slot
    can always read itself and no row can be talked into having no
    lanes at all.
  * OPEN rows keep today's additive, open-only, gain-scaled `_mb` and
    today's close BIT-FOR-BIT (the pressure mix's per-row arithmetic:
    x_open*(1-v) + x_sealed*v at v in {0,1} is exact — 1.0*x = x,
    0.0*finite = 0, x + 0 = x).

THE READING THIS PATCH TOOK (stated because the spec admits two, and the
registered proofs decide between them). "The baseline becomes the
identity (self lanes only)" cannot mean the identity is used as the HARD
CLOSE for sealed rows: `(1 - I) * -1e4` kills every off-diagonal lane,
the head's +-30 gate can never reopen one, and the spec's own birth
claim ("raw ~ 0 -> a uniform half-open field ... attention over all
lanes at once") would be false. So the severance is: THE HARD CLOSE
GOES, the skeleton stays as the guaranteed-open FLOOR, and the soft gate
is the only thing that opens or closes a lane. That makes the birth
field EXACTLY the all-lanes-open field (a constant log(0.5) added to
every lane leaves softmax invariant — proof (c), bitwise), and it makes
`ALG_MASK_COOK_SKEL` mean something real: which lanes the head is not
allowed to close.

BIRTH IS FUNCTIONAL, NOT DEAD: mh_wo and mh_headmix are zero-init, so
`_raw == 0` exactly and the gate is the constant log(0.5) on every lane;
the floor equals it exactly (same kernel chain on `_raw*0`), so
`maximum` returns it bitwise. A sealed row at birth reads the whole
lane field. NO -inf, NO NaN: `_raw` is re-clipped to [-30, 30] before
the log-sigmoid (the head already clips it there; the second clip is
this organ's own guard, so the gate does not depend on an upstream clip
staying), log-sigmoid is computed as -softplus(-raw) (finite at -30:
-29.999998), and the floor is a plain blend of two finite constants —
no epsilon denominators, no where-gates needed because nothing divides.

ONE MASK, ONE ROAD (the polar B6 lesson: two consumers of one tensor,
rotate/gate each exactly ONCE). `_mb` feeds THREE mixers — `sc2` (the
main slot attention), `_mx_sc` (the FED mixer twin) and `_sm21` (ALT21
station 4). The blend happens ONCE, at `_mb`'s definition, so all three
consumers see the same road by construction; and the close is severed
ONCE, into `_mck`, which all three closes read.

WHAT THIS PATCH DELIBERATELY DOES NOT SEVER (registered, not hidden):
 (1) the mask head's OWN attention (`_mh_at`, closed by `_sm_kb`) — that
     is the organ's INPUT, not the road. Severing it would leave the
     head reading only itself and the road it must draw would be drawn
     blind. The head is allowed to READ the baseline; it may no longer
     be MULTIPLIED by it.
 (2) the polar sink's E&B coupling (`_polar_em(_pol_u, _sm_kb,
     POLAR_EM)`). The sink's contract calls `_sm_kb` "the mask actually
     in force this breath"; under the cooker the mask in force on a
     sealed row is the head's gate, so on those rows the two diverge.
     Feeding the EM `_mck` instead would make a sealed row's clock
     couple every slot to every slot — an unregistered change to the
     clock physics with the twin's clock bar (probe >= 0.95) in play.
     The divergence is REGISTERED here and asserted structurally below
     (the EM must keep reading `_sm_kb`) so nobody "fixes" it silently.

THE READ-TIME METER. `ALG_MASK_SEAL=1` seals EVERY row at read (the
`_MCV` buffer never exists outside do_train): parse accuracy carried by
the mask head's lanes alone — the new meter mask-sealed-wild /
mask-sealed-mint via loop_val. The champion reads ~0 there by
construction (an identity floor plus a 1e-5x additive head = no lanes);
that 0 is the baseline the cooker opens.

VAL/READ HYGIENE — ITS OWN GUARD, NOT SC_EVAL's. The trainer's SC_EVAL
push around `_quick_val` is conditional on ALG_SHELF_CIRCLE >= 2, so it
cannot be borrowed to exclude an attention organ at every shelf mode.
This patch adds `MC_EVAL`, pushed UNCONDITIONALLY around `_quick_val`
(the SC_EVAL idiom, one door of its own): non-empty -> the per-row
cooker is OFF for that call, so val always compares the OPEN regime and
stays comparable across arms. Only "0" is ever pushed; any non-empty
value means OPEN. Arming asserts MC_EVAL is unset at train time (a stale
MC_EVAL would bake the cooker OPEN at JIT capture — THE UNLIT STOVE).

INDEPENDENCE FROM THE PRESSURE COOKER. `_MCV` is armed exactly like
`_PCV` (a (B,1,1) data buffer, created before the first step() capture,
assigned in place per step from the batch's DATASET row indices) but the
assignment hash uses a DIFFERENT Knuth multiplier and a different addend:
  _PCV:  h(i) = (i * 2654435761) mod 2^32 / 2^32
  _MCV:  h(i) = (i * 2246822519 + 2654435761) mod 2^32 / 2^32
MEASURED (n = 25000): |P(both) - P(pc)P(mc)| <= 1e-4 at (0.30, 0.15),
(0.30, 0.30) and (0.15, 0.15); all four cells {open/open, pc-only,
mask-only, both} occur. A row may be sealed by either cooker, both, or
neither, and each seal is exact per row.

NEW PARAMETERS: ZERO. The cooker is a data buffer, a lifted close and a
gradient road, not capacity.

--check: loads the file, asserts every anchor present and unique, builds
the would-be result, ast-parses it, runs the structural asserts and the
symtable free-variable audit, and writes NOTHING. The CPU proofs live in
scripts/mask_cook_smoke.py, which stages this patch in memory exactly as
polar_sink_smoke.py stages the sink.
"""
import ast
import builtins
import os
import symtable
import sys

fn = os.environ.get("MC_TARGET", 'scripts/phase1_algebra_head.py')
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

# ---------------------------------------------------------------- guards
assert 'ALG_MASK_COOK' not in s and '_MCV' not in s, \
    "the mask cooker is already present — patch was applied; refuse (idempotence)"
assert '_mb = (p["mh_gain"].reshape(1, 1, 1)' in s, \
    "the MASK HEAD is not in this head — the cooker severs its baseline"
assert '_PCV' in s and 'ALG_PC_MIX' in s, \
    "the PRESSURE MIX is missing — the cooker is its per-row twin and " \
    "arms beside it (wrong vintage of the head)"
assert '_polar_em(_pol_u, _sm_kb, POLAR_EM)' in s, \
    "the polar sink is missing — anchor into the head AS IT IS (a0d7b75)"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# ---------------------------------------------------------------------
# 1. module scope: the once-print flag (the _POLAR_SHOWN idiom)
# ---------------------------------------------------------------------
patch(1, "module: _MC_SHOWN (the doors' once-print flag)",
      '''_POLAR_SHOWN = False''',
      '''_POLAR_SHOWN = False
_MC_SHOWN = False        # THE MASK COOKER's doors: printed once''')

# ---------------------------------------------------------------------
# 2. module scope: the cooker's two helpers, immediately above breath_step
# ---------------------------------------------------------------------
patch(2, "module: _mask_cook_v + _mask_cook_skel (the cooker's organs)",
      '''def breath_step(p, state, kb, ctx):''',
      '''def _mask_cook_v():
    """THE MASK COOKER's per-row seal value (apply_mask_cook.py,
    2026-09-08; spec docs/mask_cooker_spec.md). Returns

      None      nothing is sealed — today's path, bit-for-bit;
      1.0       EVERY row is sealed: ALG_MASK_SEAL=1, the READ-TIME
                meter (mask-sealed-wild / mask-sealed-mint), the
                baseline severed for all rows at read;
      (B,1,1)   the `_MCV` data buffer do_train arms — the _PCV idiom:
                one JIT graph, dynamic value, a STABLE index-hash
                assignment (flat mix, never re-rolled per epoch).

    VAL/READ HYGIENE: MC_EVAL non-empty forces the OPEN regime and wins
    over every other door. The trainer pushes MC_EVAL="0" around
    _quick_val — its OWN guard, not SC_EVAL's, because that push is
    conditional on ALG_SHELF_CIRCLE >= 2 while the mask road must be
    excluded from val at every shelf mode. Outside do_train `_MCV` never
    exists, so ALG_MASK_COOK in a read env is inert by construction (the
    trained-env law; the _PCV precedent)."""
    if os.environ.get("MC_EVAL", ""):
        return None                     # val/read compares OPEN
    if int(os.environ.get("ALG_MASK_SEAL", "0")):
        return 1.0                      # the read-time meter: all rows
    if float(os.environ.get("ALG_MASK_COOK", "0")) <= 0.0:
        return None
    _v = globals().get("_MCV")          # armed only inside do_train
    return None if _v is None else _v.reshape(-1, 1, 1)


def _mask_cook_skel(B, L, fat_cur, ctx):
    """The SEVERED baseline's SKELETON: the lanes whose openness is
    GUARANTEED on a sealed row — the floor the head is not allowed to
    close (a slot can always read itself; no row can be talked into
    having no lanes). Values in [0, 1]; DETACHED (structure, never a
    gradient road — the mask head's metadata contract).

      ALG_MASK_COOK_SKEL=self      (default) the diagonal.
      ALG_MASK_COOK_SKEL=sentence  same-sentence lanes, self included:
        the breath's OWN reading (fat_cur, this breath's head-averaged
        token attention) pushed through the token->sentence map, so
        P = attn @ onehot(sent) is each slot's sentence distribution and
        P @ P^T its pairwise agreement (the meter-divergence law: the
        skeleton CALLS the organ in force, it does not rebuild a second
        copy of build_slot_masks' `same` from stale numpy).

    The birth proof does not depend on which: the floor is
    logsigmoid(0)*skel + (1-skel)*(-1e4), which is <= logsigmoid(0) for
    ANY skel in [0, 1], so at raw = 0 the maximum returns the constant
    log(0.5) on every lane — the all-lanes-open field, bitwise."""
    from tinygrad import Tensor, dtypes
    _mode = os.environ.get("ALG_MASK_COOK_SKEL", "self")
    _eye = Tensor(np.eye(L, dtype=np.float32),
                  dtype=dtypes.float).reshape(1, L, L)
    if _mode == "self":
        return _eye
    assert _mode == "sentence", (
        f"ALG_MASK_COOK_SKEL={_mode!r}: the skeleton is 'self' "
        f"(the diagonal) or 'sentence' (same-sentence lanes)")
    _snt = ctx.get("mc_sent")
    assert _snt is not None, (
        "ALG_MASK_COOK_SKEL=sentence needs ctx['mc_sent'] (forward "
        "passes it); a seam driver calling breath_step directly must "
        "supply it — refusing to fall back to 'self' silently")
    _ar = Tensor(np.arange(SENT_MAX, dtype=np.float32),
                 dtype=dtypes.float).reshape(1, 1, SENT_MAX)
    _oh = (_snt.float().reshape(B, -1, 1) == _ar).float()   # (B, T, S)
    _pp = fat_cur @ _oh                                     # (B, L, S)
    return (_pp @ _pp.transpose(-2, -1) + _eye).clip(0.0, 1.0).detach()


def breath_step(p, state, kb, ctx):''')

# ---------------------------------------------------------------------
# 3. breath_step: `_mck`, the close mask (the baseline until severed)
# ---------------------------------------------------------------------
patch(3, "breath_step: _mck (the close mask) + the ALG_MASK_SEAL guard",
      '''    _mb = None
    if int(os.environ.get("ALG_MASKHEAD", "0")) and "mh_wo" in p:''',
      '''    _mb = None
    # THE MASK COOKER (apply_mask_cook.py, 2026-09-08): the CLOSE mask.
    # `_sm_kb` until the cooker severs it per sealed row; all THREE slot
    # mixers close with it (sc2, the FED twin, ALT21 station 4) — one
    # mask, one road. With every cooker door unset it IS `_sm_kb`, the
    # same object: env-inertness by identity, not by value.
    _mck = _sm_kb
    assert not (int(os.environ.get("ALG_MASK_SEAL", "0"))
                and not (int(os.environ.get("ALG_MASKHEAD", "0"))
                         and "mh_wo" in p)), (
        "ALG_MASK_SEAL needs the mask head (ALG_MASKHEAD=1 and mh_wo in "
        "the checkpoint): with no head there are no lanes to seal to, "
        "and the meter would silently read the OPEN machine")
    if int(os.environ.get("ALG_MASKHEAD", "0")) and "mh_wo" in p:''')

# ---------------------------------------------------------------------
# 4. breath_step: the severance itself — one blend, at `_mb`'s definition
# ---------------------------------------------------------------------
patch(4, "breath_step: the per-row severance (gate + floor + lifted close)",
      '''        _mb = (p["mh_gain"].reshape(1, 1, 1)
               * (_mh_sp - _mh_sp0) * _sm_kb)''',
      '''        _mc_pre = (_mh_sp - _mh_sp0) * _sm_kb   # the census PRE tap
        _mb = (p["mh_gain"].reshape(1, 1, 1)
               * (_mh_sp - _mh_sp0) * _sm_kb)
        _mcv = _mask_cook_v()
        if _mcv is not None:
            # THE MASK COOKER (apply_mask_cook.py, 2026-09-08; spec
            # docs/mask_cooker_spec.md; the mandatory-road law).
            # SEALED rows: the baseline stops being the road. Its hard
            # CLOSE is lifted (all factor lanes structurally open;
            # scratch columns keep the baseline's policy — FED item 6's
            # raising law stands), its SKELETON survives as the floor
            # the head may not close, and the head's RAW logits become
            # the mask over ALL lanes: sc2 + log(sigmoid(raw)),
            # UNGATED by mh_gain, UNMULTIPLIED by the baseline.
            # OPEN rows: today's `_mb` and today's close, bit-for-bit —
            # the pressure mix's arithmetic, per row, exact at v in
            # {0,1} (1.0*x = x, 0.0*finite = 0, x + 0 = x).
            # BIRTH: mh_wo and mh_headmix are zero-init -> _raw == 0 ->
            # the gate is the CONSTANT log(0.5) on every lane and the
            # floor equals it bitwise (the same kernel chain on
            # _raw*0 — the _mh_sp0 idiom), so a sealed row's attention
            # is the softmax over ALL lanes: cold but FUNCTIONAL, and
            # the loss sharpens it. No -inf/NaN: _raw is re-clipped to
            # [-30, 30] here (this organ's own guard, not a borrowed
            # one) and log-sigmoid is -softplus(-raw), finite at -30.
            # ONE MASK, ONE ROAD: the blend happens ONCE, here, so all
            # three consumers of `_mb` (sc2, the FED mixer, ALT21
            # station 4) see the same road by construction.
            _mcr = _raw.clip(-30.0, 30.0)
            _mclg = -(1.0 + (-_mcr).exp()).log()            # log sigmoid
            _mclg0 = -(1.0 + (-(_mcr * 0.0)).exp()).log()   # its birth
            _mcsk = _mask_cook_skel(B, L_TOT, fat_cur, ctx)
            _mcb = _mclg.maximum(_mclg0 * _mcsk + (1.0 - _mcsk) * -1e4)
            _mcc = Tensor(np.concatenate(
                [np.ones(L_FAC, np.float32),
                 np.zeros(L_TOT - L_FAC, np.float32)])).reshape(1, 1, -1)
            _mb = _mb * (1.0 - _mcv) + _mcb * _mcv
            _mc_pre = _mc_pre * (1.0 - _mcv) + _mcb * _mcv
            _mck = _sm_kb * (1.0 - _mcv) + _sm_kb.maximum(_mcc) * _mcv''')

# ---------------------------------------------------------------------
# 5. breath_step: the census PRE tap reads the road actually taken
# ---------------------------------------------------------------------
patch(5, "breath_step: census maskhead_pre reads _mc_pre (meter law)",
      '''            _CENSUS.append((kb, "maskhead_pre",
                            ((_mh_sp - _mh_sp0) * _sm_kb)
                            .realize().numpy()))''',
      '''            _CENSUS.append((kb, "maskhead_pre",
                            _mc_pre.realize().numpy()))
                            # (the SAME expression when the cooker is
                            # off; on sealed rows PRE == POST, because
                            # the gate is ungained — the meter must
                            # call the organ in force, not a copy)''')

# ---------------------------------------------------------------------
# 6-8. breath_step: the three closes read `_mck`
# ---------------------------------------------------------------------
patch(6, "breath_step: sc2's close reads _mck",
      '''    sc2 = sc2.clip(-1e4, 1e4) + (1.0 - _sm_kb) * -1e4''',
      '''    sc2 = sc2.clip(-1e4, 1e4) + (1.0 - _mck) * -1e4   # cooker: _mck''')

patch(7, "breath_step: the FED mixer's close reads _mck",
      '''        _sm_tw = _sm_kb''',
      '''        _sm_tw = _mck              # cooker: the severed close''')

patch(8, "breath_step: ALT21 station 4's close reads _mck",
      '''        _sm21 = _sm21.clip(-1e4, 1e4) + (1.0 - _sm_kb) * -1e4''',
      '''        _sm21 = _sm21.clip(-1e4, 1e4) + (1.0 - _mck) * -1e4  # cooker''')

# ---------------------------------------------------------------------
# 9. forward: the sentence-skeleton port (a dict key; zero compute)
# ---------------------------------------------------------------------
patch(9, "forward: ctx['mc_sent'] (the sentence skeleton's port)",
      '''                   "XR_GRADED": XR_GRADED, "XR_ELASTIC": XR_ELASTIC}''',
      '''                   "XR_GRADED": XR_GRADED, "XR_ELASTIC": XR_ELASTIC,
                   # THE MASK COOKER's sentence-skeleton port
                   # (ALG_MASK_COOK_SKEL=sentence): the token->sentence
                   # ids. A dict key only — zero compute when the door
                   # is unset or the skeleton is `self`.
                   "mc_sent": sent}''')

# ---------------------------------------------------------------------
# 10. do_train: arm _MCV (after the mask-prep pass, before the capture)
# ---------------------------------------------------------------------
patch(10, "do_train: arm _MCV + the independent stable index-hash",
      '''    t0 = time.time()
    for s in range(steps):''',
      '''    _mc_mix = float(os.environ.get("ALG_MASK_COOK", "0"))
    _mc_assign = None
    if _mc_mix > 0.0:
        # THE MASK COOKER (2026-09-08, docs/mask_cooker_spec.md): the
        # per-row severance of the BASELINE slot mask. Armed exactly
        # like the pressure mix (a (B,1,1) buffer created BEFORE the
        # first step() capture — the JIT law) and INDEPENDENT of it: a
        # different Knuth multiplier AND a different addend, so a row
        # may be sealed by either cooker, both, or neither. Measured at
        # n = 25000: |P(both) - P(pc)P(mc)| <= 1e-4 at the doses in
        # play; all four cells occur.
        assert int(os.environ.get("ALG_MASKHEAD", "0")) and "mh_wo" in p, (
            "ALG_MASK_COOK needs the mask head (ALG_MASKHEAD=1 and "
            "mh_wo in the params): the cooker severs the baseline so "
            "the HEAD's lanes are the only road — with no head a "
            "sealed row would have no mask at all")
        assert not int(os.environ.get("ALG_MASK_SEAL", "0")), (
            "ALG_MASK_SEAL is the READ-TIME meter (every row sealed); "
            "with ALG_MASK_COOK it would seal training entirely and "
            "the dose would be a lie — unset it")
        assert not os.environ.get("MC_EVAL", ""), (
            "ALG_MASK_COOK with MC_EVAL set would bake the cooker OPEN "
            "at JIT capture (THE UNLIT STOVE: training that never "
            "sealed) — unset MC_EVAL; val pushes it by itself")
        _mc_skel = os.environ.get("ALG_MASK_COOK_SKEL", "self")
        assert _mc_skel in ("self", "sentence"), (
            f"ALG_MASK_COOK_SKEL={_mc_skel!r}: 'self' or 'sentence'")
        _mc_h = ((np.arange(n, dtype=np.uint64) * np.uint64(2246822519)
                  + np.uint64(2654435761)) % np.uint64(4294967296)
                 ).astype(np.float64) / 4294967296.0
        _mc_assign = (_mc_h < _mc_mix).astype(np.float32)
        globals()["_MCV"] = Tensor(
            np.zeros((batch, 1, 1), np.float32)).contiguous().realize()
        _mc_both = (float((_mc_assign * _pc_assign).sum()) / max(n, 1)
                    if _pc_assign is not None else 0.0)
        print(f"[maskcook] baseline severance armed: share={_mc_mix} -> "
              f"{int(_mc_assign.sum())}/{n} rows sealed (stable "
              f"index-hash, multiplier 2246822519); skeleton={_mc_skel}"
              f"; both-seals={_mc_both:.4f} of rows (pressure share "
              f"{_pc_mix}); the head's raw logits are the mask on those "
              f"rows (sc2 + log sigmoid(raw), ungained)", flush=True)
    t0 = time.time()
    for s in range(steps):''')

# ---------------------------------------------------------------------
# 11. do_train loop: the per-step assign, beside _PCV's
# ---------------------------------------------------------------------
patch(11, "do_train loop: per-step _MCV assign from the batch indices",
      '''        if _pc_assign is not None:
            globals()["_PCV"].assign(Tensor(
                _pc_assign[idx].reshape(-1, 1, 1),
                dtype=globals()["_PCV"].dtype)).realize()
        lv = step()''',
      '''        if _pc_assign is not None:
            globals()["_PCV"].assign(Tensor(
                _pc_assign[idx].reshape(-1, 1, 1),
                dtype=globals()["_PCV"].dtype)).realize()
        if _mc_assign is not None:
            globals()["_MCV"].assign(Tensor(
                _mc_assign[idx].reshape(-1, 1, 1),
                dtype=globals()["_MCV"].dtype)).realize()
        lv = step()''')

# ---------------------------------------------------------------------
# 12. do_train: the val push — the cooker's OWN guard
# ---------------------------------------------------------------------
patch(12, "do_train: MC_EVAL push around _quick_val (val stays OPEN)",
      '''            if int(os.environ.get("ALG_SHELF_CIRCLE", "0")) >= 2:
                os.environ["SC_EVAL"] = "0"     # val compares OPEN mode
            fv = _quick_val()
            if int(os.environ.get("ALG_SHELF_CIRCLE", "0")) >= 2:
                os.environ.pop("SC_EVAL", None)''',
      '''            if int(os.environ.get("ALG_SHELF_CIRCLE", "0")) >= 2:
                os.environ["SC_EVAL"] = "0"     # val compares OPEN mode
            # THE MASK COOKER's own val guard (2026-09-08): pushed
            # UNCONDITIONALLY, because the SC_EVAL push above is
            # conditional on ALG_SHELF_CIRCLE >= 2 and an attention
            # organ must be excluded from val at every shelf mode. Any
            # non-empty value means OPEN; only "0" is ever pushed.
            os.environ["MC_EVAL"] = "0"       # ... and the OPEN mask
            fv = _quick_val()
            os.environ.pop("MC_EVAL", None)
            if int(os.environ.get("ALG_SHELF_CIRCLE", "0")) >= 2:
                os.environ.pop("SC_EVAL", None)''')

# ---------------------------------------------------------------------
# 13. build_params: the doors, printed once
# ---------------------------------------------------------------------
patch(13, "build_params: print every cooker door's state once",
      '''    global _POLAR_SHOWN''',
      '''    global _POLAR_SHOWN, _MC_SHOWN
    if not _MC_SHOWN and (float(os.environ.get("ALG_MASK_COOK", "0")) > 0.0
                          or int(os.environ.get("ALG_MASK_SEAL", "0"))
                          or os.environ.get("ALG_MASK_COOK_SKEL", "")
                          or os.environ.get("MC_EVAL", "")):
        # THE MASK COOKER's doors (spec S2): one line, once, naming
        # every one of them. Silent when all are unset.
        _MC_SHOWN = True
        print(f"[maskcook] doors: ALG_MASK_COOK="
              f"{os.environ.get('ALG_MASK_COOK', '0')} (train dose) "
              f"ALG_MASK_COOK_SKEL="
              f"{os.environ.get('ALG_MASK_COOK_SKEL', 'self')} (floor) "
              f"ALG_MASK_SEAL={os.environ.get('ALG_MASK_SEAL', '0')} "
              f"(read-time meter: ALL rows) "
              f"MC_EVAL={os.environ.get('MC_EVAL', '') or '(unset)'} "
              f"(non-empty = OPEN) | severed baseline -> skeleton floor "
              f"+ sc2 + log sigmoid(raw), ungained", flush=True)''')

for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# ===========================================================================
# STRUCTURAL ASSERTS on the would-be module (cheap, no import, no GPU)
# ===========================================================================
# -- ZERO new parameters: the cooker is a buffer and a road, not capacity
import re as _re                                              # noqa: E402
_p0 = set(_re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(',
                      open(fn).read()))
_p1 = set(_re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(', s))
assert _p0 == _p1, f"the cooker added parameters: {sorted(_p1 - _p0)}"

# -- ONE blend, at _mb's definition: every consumer inherits it
assert s.count('_mb = _mb * (1.0 - _mcv) + _mcb * _mcv') == 1, \
    "the road must be blended exactly ONCE (the polar B6 lesson)"
assert s.count('_mck = _sm_kb * (1.0 - _mcv)') == 1 and \
    s.count('_mck = _sm_kb\n') == 1, \
    "the close must be severed exactly once, from one default"
_i_mb = s.index('_mb = _mb * (1.0 - _mcv) + _mcb * _mcv')
for _consumer in ('sc2 = sc2 + _mb', '_mx_sc = _mx_sc + _mb.unsqueeze(1)',
                  '_sm21 = _sm21 + _mb'):
    assert s.count(_consumer) == 1 and s.index(_consumer) > _i_mb, \
        f"consumer {_consumer!r} must read the BLENDED road (one road)"
for _close in ('sc2 = sc2.clip(-1e4, 1e4) + (1.0 - _mck) * -1e4',
               '_sm_tw = _mck',
               '_sm21 = _sm21.clip(-1e4, 1e4) + (1.0 - _mck) * -1e4'):
    assert s.count(_close) == 1, f"close {_close!r} lost"
assert '(1.0 - _sm_kb) * -1e4' not in s, \
    "a close still reads the UNSEVERED baseline — one mask, one road"

# -- the gate: the head's RAW logits, re-clipped, log-sigmoid, no epsilon
assert '_mcr = _raw.clip(-30.0, 30.0)' in s, \
    "the gate must clip the raw logits itself (no borrowed guard)"
assert '_mclg = -(1.0 + (-_mcr).exp()).log()' in s, \
    "log(sigmoid(raw)) must be computed as -softplus(-raw) (finite at -30)"
assert '_mclg0 = -(1.0 + (-(_mcr * 0.0)).exp()).log()' in s, \
    "the floor's birth value must ride the SAME kernel chain on raw*0 " \
    "(the _mh_sp0 idiom) — else the birth field is not bitwise"
assert 'p["mh_gain"]' not in s[s.index('_mcr = _raw.clip'):
                               s.index('_mck = _sm_kb * (1.0 - _mcv)')], \
    "the sealed road must be UNGATED by mh_gain (the mandatory road)"
_i_raw = s.index('_raw = (_mh_rp + _mh_rh).clip(-30.0, 30.0)')
assert _i_raw < s.index('_mcr = _raw.clip'), "the gate reads _raw after it exists"

# -- what the cooker deliberately does NOT sever (registered above)
assert '_polar_em(_pol_u, _sm_kb, POLAR_EM)' in s, \
    "the polar E&B coupling must keep reading _sm_kb (registered scope: " \
    "feeding it _mck would rewire the clock on sealed rows)"
assert '+ (1.0 - _sm_kb.unsqueeze(1)) * -1e4).softmax(-1)' in s, \
    "the mask head's OWN attention must keep reading _sm_kb (its input, " \
    "not the road)"

# -- val hygiene: the push exists, is unconditional, and brackets the call
_i_push = s.index('os.environ["MC_EVAL"] = "0"')
_i_val = s.index('fv = _quick_val()')
_i_pop = s.index('os.environ.pop("MC_EVAL", None)')
assert _i_push < _i_val < _i_pop, \
    "MC_EVAL must be pushed before _quick_val and popped after"
assert s[s.rindex('\n', 0, _i_push) + 1:_i_push] == ' ' * 12, \
    "the MC_EVAL push must be UNCONDITIONAL (it sits at val's indent, " \
    "not inside the ALG_SHELF_CIRCLE branch)"
assert s.count('if os.environ.get("MC_EVAL", ""):\n        return None') == 1, \
    "MC_EVAL must win over every other door in _mask_cook_v"

# -- arming: after the mask-prep pass, before the first capture; per-step feed
assert s.index('print(f"[breath] masks ready') < s.index('_mc_assign = (_mc_h'), \
    "the buffer must be armed AFTER the mask-prep pass"
assert s.index('_mc_assign = (_mc_h') < s.index('    t0 = time.time()'), \
    "arming must precede the step loop (the JIT law)"
assert s.index('_mc_assign[idx]') < s.index('lv = step()'), \
    "the per-step assign must precede step()"
assert s.index('_pc_assign[idx]') < s.index('_mc_assign[idx]'), \
    "the _MCV assign rides beside (after) the _PCV assign"

# -- the two hashes are DIFFERENT (independence is the whole point)
assert '2654435761' in s and '2246822519' in s, "both hash seeds present"
assert s.count('np.uint64(2246822519)') == 1 and \
    'np.uint64(2246822519)\n                  + np.uint64(2654435761)' in s, \
    "the mask assignment must use its own multiplier AND an addend"

# -- housekeeping the head's own laws impose
assert 'dtypes.float32' not in s, "no float32 dtype literal anywhere"
assert '1e-6)' not in s[s.index('def _mask_cook_v('):
                        s.index('def breath_step(')], \
    "no epsilon denominators in the cooker (nothing divides)"

# ===========================================================================
# THE SYMTABLE FREE-VARIABLE AUDIT (the apply_polar_sink.py idiom)
# ===========================================================================
mod_tbl = symtable.symtable(s, fn, 'exec')
module_names = set(mod_tbl.get_identifiers())
DYNAMIC_OK = {'_CENSUS', '_IMP', '_SEV', '_SGC', '_BINDC', '_PCV', '_MCV'}
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
           '_mask_cook_v', '_mask_cook_skel')
_seen = set()
for child in mod_tbl.get_children():
    if child.get_name() in AUDITED:
        _seen.add(child.get_name())
        audit(child, child.get_name())
assert _seen == set(AUDITED), f"audit missed {sorted(set(AUDITED) - _seen)}"

# ===========================================================================
# REPORT
# ===========================================================================
print(f"[mask cook] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines, 0 deleted):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
print("[mask cook] symtable free-var audit PASS "
      f"({', '.join(AUDITED)})")
print("[mask cook] NEW params: 0 (a data buffer, a lifted close and a "
      "gradient road — not capacity); ALG_MASK_COOK unset + "
      "ALG_MASK_SEAL unset = byte-identical (the eq A/B/C gate and "
      "scripts/mask_cook_smoke.py GATE 1 prove it, both polar configs)")
print("[mask cook] the road: SEALED rows lose the baseline's close, keep "
      "its SKELETON as a floor, and take sc2 + log sigmoid(raw) over ALL "
      "lanes — ungained, unmultiplied; OPEN rows are bit-for-bit today")
print("[mask cook] doors: ALG_MASK_COOK=<share> ALG_MASK_COOK_SKEL="
      "self|sentence ALG_MASK_SEAL=1 (read-time meter) MC_EVAL (val's "
      "OPEN push, the cooker's own — SC_EVAL's is shelf-conditional)")
print("[mask cook] seals compose: _MCV's hash is (i*2246822519 + "
      "2654435761) mod 2^32 vs _PCV's (i*2654435761) — measured "
      "independent to 1e-4 at n=25000; all four cells occur")
if CHECK:
    print("[mask cook] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[mask cook] APPLIED ({fn}); ast OK — run the eq pre/post "
          "A/B/C gate (all cooker doors unset) + .cache/pc_row_smoke.py "
          "+ scripts/mask_cook_smoke.py before trusting")
