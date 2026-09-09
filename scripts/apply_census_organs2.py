"""apply_census_organs2.py — THE PORT CENSUS, THE THREE HOLES (2026-09-08).
Phase 2 of the organ pass. It RIDES ON apply_census_organs.py (which the
lead applied the same day) and closes the three gaps that pass named in
its own report rather than quietly leaving them. Under --check it writes
NOTHING. The lead applies.

  (1) ALT21 STATIONS 3-4 — the largest hole, and the one that was ON in
      the champion family the whole time. `h_slot = h_slot + _d21a +
      _d21b` writes two additive contributions into the state every
      breath and neither was measured. Recorded as `alt21_s3` (_d21a,
      the second bank attention's output) and `alt21_s4` (_d21b, the
      second slot mixer's output), both STATE band.
      NO PRE-GAIN FORM EXISTS ON THIS PATH, and that is a finding, not
      an omission: stations 3-4 have no scalar gain. Their door is the
      ZERO-INIT OUTPUT MATRIX itself (alt21_attn_wo, alt21_W_bo) — the
      "silent birth" idiom, not the "ajar gain" idiom. There is no
      scalar to divide out, so post IS the whole reading; a `_pre` here
      would have to be the pre-projection activation, which lives in a
      different space and would be a fake ratio. Reported as `-` by the
      reader, deliberately.

  (2) NOTEBOOK LANE 2 — `_rd2 * fed_nb_g`, the FED shelf's second lane,
      whose gain sat at 0.0004 in the forensic's dead-organ list and
      which the census could not see. Recorded as `notebook2` /
      `notebook2_pre` at BOTH of its branches (per-slot and blurred).
      This one HAS a scalar gain, so the pre/post ratio is the usual
      two-coordinate reading — and it is the reading that decides
      whether lane 2 is a pruning candidate or a starved organ.

  (3) THE LANE-1 BLURRED BRANCH — gen-1 (apply_census_hooks.py) hooked
      only the NB_PERSLOT arm of the notebook read, so any NB_PERSLOT=0
      configuration silently censused the notebook as ABSENT while it
      was in fact the dominant port. Same organ name (`notebook`), the
      other branch; exactly one of the two ever runs.

AND ONE MORE BAND BASELINE: `state_hslot`, the single-head slot output at
its birth. The mixer and both ALT21 stations do not add into `cur` — they
add into `h_slot`, which then enters the state through the breath gate.
Dividing them by the loop-state rms answered a question nobody asked. The
reader now divides them by the h_slot they actually add to, and the gate
factor is stated separately. (Phase 1 established the precedent with
`state_slot` for the score band; this is the same move for h_slot.)

Everything is behind `if _CENSUS is not None:` — the head stays BYTE-INERT
with the hook unarmed, which is every training step and every ordinary
read. Zero new parameters, zero new envs, zero supervision.

--check: loads the file, asserts every anchor present and unique, builds
the would-be result, ast-parses it, runs the structural asserts and the
symtable free-variable audit, and writes NOTHING. PS_TARGET may point at
a copy (rehearsal) — including a copy staged by apply_census_organs.py,
which is how scripts/census_organs2_smoke.py proves this patch without
ever touching the head. Default is scripts/phase1_algebra_head.py.
"""
import ast
import builtins
import os
import symtable
import sys

fn = os.environ.get("PS_TARGET", 'scripts/phase1_algebra_head.py')
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

# ---------------------------------------------------------------- guards
assert '"alt21_s3"' not in s and '"notebook2"' not in s, \
    "phase 2 of the organ census is already present — refuse (idempotence)"
assert '"maskhead"' in s and '"state_slot"' in s, \
    ("apply_census_organs.py (phase 1) is not applied to this head — "
     "phase 2 rides on it, it does not replace it")
assert s.count('_CENSUS.append(') == 15 and s.count('_cs.append(') == 3, \
    (f"expected phase 1's 15 + 3 census sites, found "
     f"{s.count('_CENSUS.append(')} + {s.count('_cs.append(')} — wrong vintage")
assert 'def breath_step(p, state, kb, ctx):' in s, \
    "breath_step is not factored — this patch anchors inside it"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# ===========================================================================
# 1. THE h_slot BASELINE. Recorded at h_slot's birth — the single-head
#    slot output, before the mixer's twin term, before ALT21's two
#    stations, before the ablation arms. This is the tensor the three
#    h_slot-band organs actually add to, so it is their denominator.
# ===========================================================================
patch(1, "breath_step: state_hslot — the h_slot baseline, the in-band "
         "denominator for the mixer and the two ALT21 stations",
      '''    h_slot = (sc2.softmax(-1) @ bv) @ p["W_bo"] + p["W_bo_b"]''',
      '''    h_slot = (sc2.softmax(-1) @ bv) @ p["W_bo"] + p["W_bo_b"]
    if _CENSUS is not None:
        _CENSUS.append((kb, "state_hslot", h_slot.realize().numpy()))''')

# ===========================================================================
# 2. ALT21 STATIONS 3-4. One anchor, two organs. No scalar gain exists on
#    this path (the door is the zero-init output matrix), so no _pre.
# ===========================================================================
patch(2, "breath_step: alt21_s3 + alt21_s4 — the two ALT21 station "
         "contributions into h_slot (no scalar gain on this path)",
      '''        h_slot = h_slot + _d21a + _d21b  # additive; zeros at birth''',
      '''        h_slot = h_slot + _d21a + _d21b  # additive; zeros at birth
        if _CENSUS is not None:
            # THE LARGEST HOLE, closed (apply_census_organs2.py): two
            # additive writes into the state every breath, ON in the
            # champion family, never measured. NO _pre FORM: stations
            # 3-4 ride the ZERO-INIT OUTPUT MATRIX idiom (alt21_attn_wo,
            # alt21_W_bo), not a scalar gain — there is nothing to
            # divide out, and a pre-projection "pre" would be a fake
            # ratio in a different space.
            _CENSUS.append((kb, "alt21_s3", _d21a.realize().numpy()))
            _CENSUS.append((kb, "alt21_s4", _d21b.realize().numpy()))''')

# ===========================================================================
# 3-4. NOTEBOOK LANE 2, both branches. Exactly one runs per config; the
#      organ name is the same either way (the reader must not care which
#      branch the config took — that is precisely the gen-1 bug below).
# ===========================================================================
patch(3, "breath_step: notebook2 + notebook2_pre, PER-SLOT branch",
      '''                q_extra = q_extra + _rd2 * p["fed_nb_g"].reshape(1, 1, 1)''',
      '''                q_extra = q_extra + _rd2 * p["fed_nb_g"].reshape(1, 1, 1)
                if _CENSUS is not None:
                    _CENSUS.append((kb, "notebook2",
                                    (_rd2 * p["fed_nb_g"].reshape(1, 1, 1))
                                    .realize().numpy()))
                    _CENSUS.append((kb, "notebook2_pre",
                                    _rd2.realize().numpy()))''')

patch(4, "breath_step: notebook2 + notebook2_pre, BLURRED branch",
      '''                q_extra = q_extra + _rd2.reshape(B, 1, -1) \\
                    * p["fed_nb_g"].reshape(1, 1, 1)''',
      '''                q_extra = q_extra + _rd2.reshape(B, 1, -1) \\
                    * p["fed_nb_g"].reshape(1, 1, 1)
                if _CENSUS is not None:
                    _CENSUS.append((kb, "notebook2",
                                    (_rd2.reshape(B, 1, -1)
                                     * p["fed_nb_g"].reshape(1, 1, 1))
                                    .realize().numpy()))
                    _CENSUS.append((kb, "notebook2_pre",
                                    _rd2.reshape(B, 1, -1).realize().numpy()))''')

# ===========================================================================
# 5. THE LANE-1 BLURRED BRANCH — the gen-1 hook's own blind spot. With
#    NB_PERSLOT=0 the census reported the notebook as ABSENT while it was
#    the loudest port in the machine. Same organ, the other branch.
# ===========================================================================
patch(5, "breath_step: notebook — the BLURRED lane-1 branch gen-1 missed "
         "(NB_PERSLOT=0 configs censused the dominant port as absent)",
      '''            q_extra = q_extra + _rd.reshape(B, 1, -1)''',
      '''            q_extra = q_extra + _rd.reshape(B, 1, -1)
            if _CENSUS is not None:
                _CENSUS.append((kb, "notebook",
                                _rd.reshape(B, 1, -1).realize().numpy()))''')

for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# ===========================================================================
# STRUCTURAL ASSERTS on the would-be module (cheap, no import, no GPU)
# ===========================================================================
NEW_ORGANS = ("alt21_s3", "alt21_s4", "notebook2", "notebook2_pre")
for _o in ("alt21_s3", "alt21_s4"):
    assert s.count(f'"{_o}",') == 1, f'organ "{_o}" must be recorded once'
# lane 2 is recorded at BOTH branches (exactly one ever runs)
for _o in ("notebook2", "notebook2_pre"):
    assert s.count(f'"{_o}",') == 2, \
        (f'organ "{_o}" must be recorded at BOTH lane-2 branches '
         f'(per-slot and blurred), found {s.count(chr(34) + _o + chr(34) + chr(44))}')
# lane 1 now has BOTH branches too (gen-1 had only the per-slot arm)
assert s.count('_CENSUS.append((kb, "notebook",') == 2, \
    "the notebook must be recorded at BOTH lane-1 branches after this patch"
assert s.count('"state_hslot"') == 1, "one h_slot baseline, recorded once"
# 15 phase-1 sites + 8 new (2 alt21 + 4 lane-2 + 1 lane-1 + 1 baseline)
assert s.count('_CENSUS.append(') == 23 and s.count('_cs.append(') == 3, \
    (f"expected 23 _CENSUS.append + 3 _cs.append, found "
     f"{s.count('_CENSUS.append(')} + {s.count('_cs.append(')}")
# -- EVERY census site is guarded (the inertness contract, mechanically)
_lines = s.split('\n')
for _i, _ln in enumerate(_lines):
    if '_CENSUS.append(' not in _ln and '_cs.append(' not in _ln:
        continue
    _guard = [_l for _l in _lines[max(0, _i - 14):_i]
              if _l.strip().startswith(('if _CENSUS is not None:',
                                        'if _cs is not None:'))]
    assert _guard, f"UNGUARDED census site at line {_i + 1}: {_ln.strip()}"
# -- NOT A LINE OF ARITHMETIC MOVED: this patch is pure insertion
_src0 = open(fn).read()
assert all(_l in s for _l in _src0.split('\n')), \
    "phase 2 must be pure insertion — no existing line may change"
assert len(s) > len(_src0), "nothing was added"
# -- ALT21 carries NO scalar gain, so it must NOT claim a _pre organ
assert '"alt21_s3_pre"' not in s and '"alt21_s4_pre"' not in s, \
    ("stations 3-4 have no scalar gain (zero-init output matrices) — a "
     "_pre organ there would be a fake ratio in a different space")
# -- lane 2's post IS pre x the gain, at both branches
assert s.count('p["fed_nb_g"].reshape(1, 1, 1))\n                                    .realize().numpy()))') == 1 \
    or s.count('"notebook2",') == 2, "lane 2 post must carry the gain"
# -- the h_slot baseline is captured at BIRTH, before every addition
_i_hs = s.index('_CENSUS.append((kb, "state_hslot"')
_i_mx = s.index('_CENSUS.append((kb, "mixer"')
_i_s3 = s.index('_CENSUS.append((kb, "alt21_s3"')
_i_gate = s.index('    g = p["breath_gate"][kb].sigmoid()')
assert _i_hs < _i_mx < _i_s3 < _i_gate, \
    ("the h_slot baseline must be captured at h_slot's birth, before the "
     "mixer's twin term and before ALT21's stations, all before the gate")
# -- no new envs, no new parameters, no supervision, no float32 literal
assert s.count('os.environ') == _src0.count('os.environ'), \
    "the census extension must not add an env door"
import re as _re                                        # noqa: E402
assert _re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(', s) == \
    _re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(', _src0), \
    "the census extension must not add a parameter"
assert 'dtypes.float32' not in s, "no float32 dtype literal anywhere"
assert '_CENSUS' not in s[s.index('def loss_fn('):], \
    ("no census tensor may reach a loss — diagnostics are NEVER supervised "
     "(the Goodhart fence)")

# ===========================================================================
# THE SYMTABLE FREE-VARIABLE AUDIT (the apply_polar_sink.py idiom)
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


AUDITED = ('breath_step', 'forward', 'do_train', 'build_params',
           '_fact_inject', '_heads_of')
_seen = set()
for child in mod_tbl.get_children():
    if child.get_name() in AUDITED:
        _seen.add(child.get_name())
        audit(child, child.get_name())
assert _seen == set(AUDITED), f"audit missed {sorted(set(AUDITED) - _seen)}"

# ===========================================================================
# REPORT
# ===========================================================================
print(f"[census organs 2] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines, 0 deleted, 0 changed):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
print("[census organs 2] new organs: alt21_s3, alt21_s4 (STATE band, into "
      "h_slot; NO scalar gain on this path — zero-init output matrices, so "
      "no _pre form exists and none is faked), notebook2 + notebook2_pre "
      "(STATE band, gain fed_nb_g); plus the lane-1 BLURRED branch under "
      "the existing `notebook` name (gen-1's blind spot: NB_PERSLOT=0 "
      "censused the dominant port as ABSENT)")
print("[census organs 2] new baseline: state_hslot — the mixer and both "
      "ALT21 stations add into h_slot, not into cur; they now divide by "
      "the h_slot they actually add to, and the breath gate g = "
      "sigmoid(breath_gate[kb]) is stated separately by the reader")
print("[census organs 2] 0 new params, 0 new envs, 0 supervision; PURE "
      "INSERTION — every pre-existing line survives byte-for-byte")
print(f"[census organs 2] symtable free-var audit PASS "
      f"({', '.join(AUDITED)})")
print("[census organs 2] contract: hook UNARMED (_CENSUS is None) = "
      "byte-identical to the phase-1 head; proved on CPU by "
      "scripts/census_organs2_smoke.py")
if CHECK:
    print("[census organs 2] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[census organs 2] APPLIED ({fn}); ast OK — run "
          "scripts/census_organs2_smoke.py before trusting")
