"""apply_cross_prior.py — THE CROSS-ATLAS PRIOR, feed-side (2026-09-05,
the paired atlas). STAGED patch ON TOP OF the APPLIED
apply_mass_thread.py code in scripts/phase1_algebra_head.py and
scripts/loop_val.py (anchors are the mass-thread's own applied text —
this patch does NOT touch apply_mass_thread.py itself; it layers).
Running-invocation law: applied only by the staged paired-atlas chain
after round-2b exits.

ENV ALG_MH_XPRIOR (requires ALG_MH_ATLAS=1 and a PAIRED atlas):
  0 (default) — gen-label feed, unchanged: each row's atlas trajectory
      comes from atlas_class(row.gen). THE ORACLE UPPER BOUND — gen
      metadata is training scaffolding; a wild deployment has none.
  1 — RESCUE mode: rows whose gen class is UNKNOWN (the zero page
      today) get a RETRIEVED trajectory instead — cross_prior on the
      row's breath-0 NL state (out["nl0"], the NL tap) against the NL
      chart's page 0; known rows keep the gen label.
  2 — DEPLOYABLE mode: EVERY row's trajectory is retrieved. No oracle
      labels anywhere in the feed — the honest path the registration
      names ("at breath 0 the NL match pre-fetches the kind's math
      trajectory").

WHERE IT FEEDS (each site already runs the pass-1 forward whose nl0 is
free — the tap's pass-1==pass-2 identity):
  do_train: NL0 banked during the mask-prep pass (same vintage as
      MASKS/FACTS/MASSB); ATLAS_IDX overridden per mode after the pass;
      the existing b_mha batch feed then serves retrieved pages with
      zero further change.
  _quick_val + loop_val: per-batch retrieval off the pass-1 forward,
      same mode semantics (trained-env law: the read legs see what the
      trainer saw).

LOUD DOORS: ALG_MH_XPRIOR without ALG_MH_ATLAS is a hard assert;
a single-chart atlas (no nl bank) is cross_prior's own RuntimeError.
Conditioning only — the retrieved page rides the SAME Goodhart-fenced
port the gen-label page rode; no new loss terms exist.

DEPENDS ON: apply_nl_tap.py (out["nl0"]) and apply_paired_atlas.py
(cross_prior) — asserted at apply time.

BOTH-ENVS-UNSET = byte-identical (every addition is behind
ALG_MH_XPRIOR, itself behind ALG_MH_ATLAS; the chain's eq gate A/B/C
verifies).

--check: builds both would-be results, ast-parses them, writes NOTHING.
"""
import ast
import sys

CHECK = '--check' in sys.argv
ANCHORS = []


def note(desc):
    ANCHORS.append(desc)


def sub(s, old, new, n=1, desc=""):
    assert s.count(old) == n, \
        f"anchor MISSING/NOT-UNIQUE (want {n}, have {s.count(old)}): {desc}"
    note(f"{desc} (x{n})")
    return s.replace(old, new, n)


# ===========================================================================
# FILE 1: scripts/phase1_algebra_head.py (do_train + _quick_val)
# ===========================================================================
fn1 = 'scripts/phase1_algebra_head.py'
s = open(fn1).read()
n_lines0 = s.count('\n')

assert 'cross_prior' not in s and '_xci' not in s, \
    "cross-prior feed already present — refuse (idempotence guard)"
if not CHECK:
    assert '"nl0"' in s, \
        "apply_nl_tap.py must be applied FIRST (out['nl0'] missing)"
    assert 'cross_prior' in open('mycelium/step_atlas.py').read(), \
        "apply_paired_atlas.py must be applied FIRST (cross_prior missing)"

# --- 1. do_train: the loud door + XPRIOR init in the atlas block -----------
s = sub(s,
        '    ATLAS_TAB = ATLAS_IDX = None\n',
        '    ATLAS_TAB = ATLAS_IDX = None\n'
        '    assert not int(os.environ.get("ALG_MH_XPRIOR", "0")) \\\n'
        '        or int(os.environ.get("ALG_MH_ATLAS", "0")), \\\n'
        '        ("ALG_MH_XPRIOR requires ALG_MH_ATLAS=1 (the trajectory "\n'
        '         "port it retrieves into) — refusing a silently dark prior")\n',
        1, "do_train: XPRIOR-requires-ATLAS loud door")

s = sub(s,
        '        print(f"[mh-atlas] trajectory feed live: {_apath} "\n'
        '              f"classes={sorted(_acls)} zero-page rows="\n'
        '              f"{int((ATLAS_IDX == len(_acls)).sum())}/{n}",\n'
        '              flush=True)\n',
        '        print(f"[mh-atlas] trajectory feed live: {_apath} "\n'
        '              f"classes={sorted(_acls)} zero-page rows="\n'
        '              f"{int((ATLAS_IDX == len(_acls)).sum())}/{n}",\n'
        '              flush=True)\n'
        '        # THE CROSS-ATLAS PRIOR (apply_cross_prior.py, 2026-09-05):\n'
        '        # 1 = retrieve for UNKNOWN-class rows only; 2 = retrieve\n'
        '        # for ALL rows (deployable; gen-label mode = oracle upper\n'
        '        # bound, training scaffolding). NL0 fills at mask-prep.\n'
        '        XPRIOR = int(os.environ.get("ALG_MH_XPRIOR", "0"))\n'
        '        NL0 = None\n'
        '        if XPRIOR:\n'
        '            from mycelium.step_atlas import cross_prior\n'
        '            assert _atl.get("nl_means") is not None, \\\n'
        '                ("ALG_MH_XPRIOR needs the PAIRED atlas (nl chart) "\n'
        '                 "— re-mine with the paired miner")\n'
        '            assert K_B > 1, "xprior rides the mask-prep pass"\n'
        '            NL0 = np.zeros((n, H_W), np.float32)\n',
        1, "do_train: XPRIOR init + paired-atlas loud door")

# --- 2. mask-prep: bank NL0 (same vintage as MASKS/FACTS) ------------------
s = sub(s,
        '            o0 = {k: out0[k].realize().numpy() for k in ("fat", "args", "res")}\n'
        '            MASKS[sl] = build_slot_masks(o0, sent[sl_p])[:len(sl)]\n',
        '            o0 = {k: out0[k].realize().numpy() for k in ("fat", "args", "res")}\n'
        '            MASKS[sl] = build_slot_masks(o0, sent[sl_p])[:len(sl)]\n'
        '            if ATLAS_TAB is not None and NL0 is not None:\n'
        '                # breath-0 NL state (the tap; pass-1 == pass-2)\n'
        '                NL0[sl] = out0["nl0"].realize().numpy()[:len(sl)]\n',
        1, "mask-prep: NL0 banked beside MASKS/FACTS")

# --- 3. after mask-prep: the retrieval override ----------------------------
s = sub(s,
        '        print(f"[breath] masks ready (mean degree "\n'
        '              f"{MASKS.sum(-1).mean():.1f}/{L_FAC})", flush=True)\n',
        '        print(f"[breath] masks ready (mean degree "\n'
        '              f"{MASKS.sum(-1).mean():.1f}/{L_FAC})", flush=True)\n'
        '        if ATLAS_TAB is not None and NL0 is not None:\n'
        '            # retrieval instead of oracle labels (mode semantics in\n'
        '            # the atlas block above); the b_mha feed needs no change\n'
        '            _xci, _ = cross_prior(_atl, NL0, return_traj=False)\n'
        '            _unk = (ATLAS_IDX == len(_acls))\n'
        '            _rep = _unk if XPRIOR == 1 else np.ones(n, bool)\n'
        '            _agree = int((_xci[_rep] == ATLAS_IDX[_rep]).sum())\n'
        '            ATLAS_IDX = np.where(_rep, _xci, ATLAS_IDX)\n'
        '            print(f"[mh-xprior] mode={XPRIOR}: {int(_rep.sum())}/"\n'
        '                  f"{n} rows fed RETRIEVED trajectories "\n'
        '                  f"({int(_unk.sum())} unknown-class; retrieval "\n'
        '                  f"agrees with gen label on {_agree} of the "\n'
        '                  f"replaced)", flush=True)\n',
        1, "do_train: ATLAS_IDX override by breath-0 NL retrieval")

# --- 4. _quick_val: same retrieval on the val cycle ------------------------
s = sub(s,
        '        _vaidx = (np.array(\n'
        '            [_acls.get(atlas_class(smp.get("gen")), len(_acls))\n'
        '             for smp in vs], np.int64)\n'
        '            if ATLAS_TAB is not None else None)\n',
        '        _vaidx = (np.array(\n'
        '            [_acls.get(atlas_class(smp.get("gen")), len(_acls))\n'
        '             for smp in vs], np.int64)\n'
        '            if ATLAS_TAB is not None else None)\n'
        '        _xpv = (int(os.environ.get("ALG_MH_XPRIOR", "0"))\n'
        '                if ATLAS_TAB is not None else 0)\n',
        1, "_quick_val: mode flag beside the label indices")

s = sub(s,
        '                _vat = (Tensor(ATLAS_TAB[_vaidx[sl_p]],\n'
        '                               dtype=dtypes.float)\n'
        '                        if ATLAS_TAB is not None else None)\n',
        '                _vai = (_vaidx[sl_p].copy()\n'
        '                        if _vaidx is not None else None)\n'
        '                if _xpv and _vai is not None:\n'
        '                    # cross-atlas prior on the val cycle (same\n'
        '                    # retrieval the trainer fed; pass-1 nl0)\n'
        '                    from mycelium.step_atlas import cross_prior\n'
        '                    _xcv, _ = cross_prior(\n'
        '                        _atl, o["nl0"].realize().numpy(),\n'
        '                        return_traj=False)\n'
        '                    _rpv = ((_vai == len(_acls)) if _xpv == 1\n'
        '                            else np.ones(len(_vai), bool))\n'
        '                    _vai = np.where(_rpv, _xcv, _vai)\n'
        '                _vat = (Tensor(ATLAS_TAB[_vai],\n'
        '                               dtype=dtypes.float)\n'
        '                        if ATLAS_TAB is not None else None)\n',
        1, "_quick_val: retrieved pages on the deployable cycle")

ast.parse(s)
assert s.count('ALG_MH_XPRIOR') >= 4 and s.count('cross_prior') >= 4

# ===========================================================================
# FILE 2: scripts/loop_val.py (same transaction — the read legs must obey
# the same mode envs the trainer saw, or the ON arm is read with a lie)
# ===========================================================================
fn2 = 'scripts/loop_val.py'
s2 = open(fn2).read()
n2_lines0 = s2.count('\n')
assert 'ALG_MH_XPRIOR' not in s2 and 'cross_prior' not in s2, \
    "loop_val cross-prior already present — refuse (idempotence guard)"

s2 = sub(s2,
         '_ATAB = _AIDX = None\n',
         '_ATAB = _AIDX = None\n'
         '_XPV = int(os.environ.get("ALG_MH_XPRIOR", "0"))\n'
         'assert not _XPV or int(os.environ.get("ALG_MH_ATLAS", "0")), \\\n'
         '    "ALG_MH_XPRIOR requires ALG_MH_ATLAS=1 (loud, never dark)"\n',
         1, "loop_val: mode flag + loud door")

s2 = sub(s2,
         '    _AIDX = np.array(\n'
         '        [_acls.get(atlas_class(s.get("gen")), len(_acls))\n'
         '         for s in vs], np.int64)\n',
         '    _AIDX = np.array(\n'
         '        [_acls.get(atlas_class(s.get("gen")), len(_acls))\n'
         '         for s in vs], np.int64)\n'
         '    if _XPV:\n'
         '        from mycelium.step_atlas import cross_prior\n'
         '        assert _atl.get("nl_means") is not None, \\\n'
         '            ("ALG_MH_XPRIOR needs the PAIRED atlas (nl chart) — "\n'
         '             "re-mine with the paired miner")\n',
         1, "loop_val: paired-atlas loud door")

s2 = sub(s2,
         '    _mha_t = (Tensor(_ATAB[_AIDX[sl_p]], dtype=dtypes.float)\n'
         '              if _ATAB is not None else None)\n',
         '    _lvai = _AIDX[sl_p].copy() if _AIDX is not None else None\n'
         '    if _XPV and _lvai is not None:\n'
         '        # THE CROSS-ATLAS PRIOR (apply_cross_prior.py): retrieval\n'
         '        # off the pass-1 breath-0 NL state (the tap) — mode 1 =\n'
         '        # unknown rows only, mode 2 = every row (deployable)\n'
         '        _xlv, _ = cross_prior(_atl, o0["nl0"].realize().numpy(),\n'
         '                              return_traj=False)\n'
         '        _rlv = ((_lvai == len(_acls)) if _XPV == 1\n'
         '                else np.ones(len(_lvai), bool))\n'
         '        _lvai = np.where(_rlv, _xlv, _lvai)\n'
         '    _mha_t = (Tensor(_ATAB[_lvai], dtype=dtypes.float)\n'
         '              if _ATAB is not None else None)\n',
         1, "loop_val: retrieved pages on the read cycle")

ast.parse(s2)

print(f"[cross prior] {len(ANCHORS)} anchors OK "
      f"(phase1 +{s.count(chr(10)) - n_lines0} lines, "
      f"loop_val +{s2.count(chr(10)) - n2_lines0} lines):")
for i, desc in enumerate(ANCHORS, 1):
    print(f"  {i:2d}. {desc}")
if CHECK:
    print("[cross prior] --check: ast OK on both would-be results; "
          "NOTHING written")
else:
    open(fn1, 'w').write(s)
    open(fn2, 'w').write(s2)
    print("[cross prior] APPLIED (phase1_algebra_head.py + loop_val.py); "
          "envs unset = byte-identical — run the eq gate (A/B/C) before "
          "trusting")
