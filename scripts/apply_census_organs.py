"""apply_census_organs.py — THE PORT CENSUS, EVERY PORT (2026-09-08).
A STAGED patch closing the instrument gap the gain-table entry named:
"the census covers 3 of the ~8 injections into the state ... builder
extending the hook (post- AND pre-gain per organ, so the gain's effect
is legible)". Under --check it writes NOTHING. The lead applies.

WHAT THIS PATCH ADDS — ten new `_CENSUS.append` records, every one behind
`if _CENSUS is not None:` (or the `globals().get("_CENSUS")` form where a
`global` declaration cannot legally be moved), so the head is BYTE-INERT
with the hook unarmed — which is every training step and every ordinary
read. Zero new parameters, zero new envs, zero GPU cost when unarmed.

  altfact / altfact_pre  the ALTERNATOR V2 fact injection, inside its own
        organ `_fact_inject` (so the step trainer's per-seam calls are
        censused by the SAME organ — the meter-divergence law). Recorded
        at kb=0 (it lands on `vst` BEFORE the breath loop) together with a
        `state` baseline for that band, so the ratio the census prints is
        against the var-slot state the injection actually modifies, never
        against a foreign denominator.
  maskhead / maskhead_pre  THE MASK HEAD's emitted bias `_mb`, captured at
        `sc2 = sc2 + _mb` — the first of its THREE injection sites (sc2,
        the fed mixer's `_mx_sc`, ALT21 station 4's `_sm21`); one tensor,
        one record. post = mh_gain * (softplus(raw) - softplus(0)) * mask;
        pre = the same without mh_gain. SCORE band (B, L, L), not state.
  alt / alt_pre  the alt_g-gated symmetric fact bias into sc2 (same three
        sites, same one tensor). SCORE band.
  mixer / mixer_pre  the FED mixer's contribution INTO THE STATE: the
        fed_mx_hg-gated per-head read, spoken through W_bo — i.e. exactly
        the tensor added to h_slot. STATE band. NOTE (printed by the
        reader): h_slot-band injections are further scaled by the breath
        gate g = sigmoid(breath_gate[kb]) before they reach cur.
  sink_waist  ALG_POLAR_D's content-plane delta, IN STATE COORDINATES:
        r * (u_after - u_before), through `_polar_ru_join` (the organ).
  sink_em  ALG_POLAR_EM's clock-plane delta, same form, same organ.

AND TWO BAND BASELINES, because a ratio needs a denominator in its own
units. `state` at kb=0 is the var-slot state the fact injection modifies;
`state_slot` is the slot mixer's RAW scores at their birth — the sc2 the
mask head and the alt bias are added to. Without the second one the
registered mask-head bar ("< 0.02x of the base state") has no in-band
denominator: a LOGIT rms over a STATE rms is a ratio of incommensurables,
and the census must not print one as if it were a fraction.

PRE-GAIN IS THE POINT. A gain's value is not its organ's volume (the
garage: bus_g 0.0077 and yet 0.35-0.46x of the state, because the stamps
it multiplies ride at 1e3-1e4). Recording post AND pre makes gain x
organ-scale legible as two coordinates instead of one confound, which is
the exact question the mask head's mh_gain 0.0002 poses: silent, or
loud-times-tiny like the garage?

WHAT THIS PATCH DOES NOT DO: it adds no organ, changes no arithmetic, and
touches no gain. Three lines are FACTORED (`_fi`, `_mx_raw`, `_mx_inj`) by
pure code motion so the pre-gain tensor has a name — the same move
apply_census_hooks.py made for `_dinj` in 2026-09-01. Byte-identity with
the hook unarmed is proved on CPU by scripts/census_organs_smoke.py
against the STAGED source (np.array_equal on every emission key, 8 rows,
DEV=CPU, with ALG_POLAR unset and with ALG_POLAR=1 + the sink open).

--check: loads the file, asserts every anchor present and unique, builds
the would-be result, ast-parses it, runs the structural asserts and the
symtable free-variable audit, and writes NOTHING. PS_TARGET may point at
a copy (rehearsal); default is scripts/phase1_algebra_head.py.
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
assert '"maskhead"' not in s and '"altfact"' not in s, \
    "the organ census is already present — patch was applied; refuse (idempotence)"
assert '_CENSUS.append((kb, "state", cur.realize().numpy()))' in s, \
    ("the port census hook is not applied to this head — this patch EXTENDS "
     "apply_census_hooks.py (2026-09-01), it does not replace it")
assert s.count('_CENSUS.append(') == 6, \
    (f"expected the 6 gen-1 census sites (state/breath_emb/notebook/garage/"
     f"detwave/router), found {s.count('_CENSUS.append(')} — wrong vintage")
assert 'def _fact_inject(p, vst, fact_buf):' in s, \
    "the alternator fact organ is missing — wrong vintage of the head"
assert 'def breath_step(p, state, kb, ctx):' in s, \
    "breath_step is not factored — this patch anchors inside it"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# ===========================================================================
# 1. THE ALTERNATOR FACT INJECTION — censused INSIDE its own organ.
#    Two reasons this lives here and not at the call site in forward():
#    (a) the meter-divergence law — the step trainer calls _fact_inject once
#        per seam with ITS OWN vst, and a call-site meter would miss those;
#    (b) forward() declares `global _CENSUS` LATER in its body, and Python
#        refuses a read prior to a global declaration in the same scope
#        (SyntaxError), so a call-site capture would have to move a line
#        this patch has no business moving.
#    `_fi` is pure code motion: `vst + (fact_buf @ W) * g` becomes
#    `_fi = fact_buf @ W;  vst + _fi * g` — same ops, same order.
#    kb=0: this injection lands BEFORE the breath loop. The `state` record
#    at kb=0 is the vst baseline, so the reader's rel-column divides by the
#    band's OWN state (the var slots), never by the loop state.
# ===========================================================================
patch(1, "_fact_inject: altfact + altfact_pre (+ the kb=0 vst baseline), "
         "captured inside the organ so per-seam calls are censused too",
      '''    return vst + (fact_buf @ p["W_fact"]) * p["alt2_g"].reshape(1, 1, 1)''',
      '''    _fi = fact_buf @ p["W_fact"]          # PRE-gain (named for the census)
    _cs = globals().get("_CENSUS")       # the port census hook, inert when
    if _cs is not None:                  # None (forward's `global _CENSUS`
        # THE PORT CENSUS, organ pass (apply_census_organs.py, 2026-09-08).
        # kb=0: this injection lands on the VAR SLOTS, before the breath
        # loop — so the state baseline recorded here is vst's, and the
        # reader's rel-column divides by the band the organ actually
        # modifies. post AND pre: gain x organ-scale, two coordinates.
        _cs.append((0, "state", vst.realize().numpy()))
        _cs.append((0, "altfact",
                    (_fi * p["alt2_g"].reshape(1, 1, 1)).realize().numpy()))
        _cs.append((0, "altfact_pre", _fi.realize().numpy()))
    return vst + _fi * p["alt2_g"].reshape(1, 1, 1)''')

# ===========================================================================
# 2. THE MASK HEAD — at the sc2 injection site. _mb is ONE tensor with
#    THREE consumers (sc2, the fed mixer's _mx_sc, ALT21 station 4's
#    _sm21); one record covers all three. SCORE band (B, L_TOT, L_TOT):
#    the reader labels the band and never pretends it is state-space.
# ===========================================================================
patch(2, "breath_step: maskhead + maskhead_pre at `sc2 = sc2 + _mb`",
      '''        sc2 = sc2 + _mb        # the injection site: BEFORE the close''',
      '''        sc2 = sc2 + _mb        # the injection site: BEFORE the close
        if _CENSUS is not None:
            # post-gain = what actually biases the scores; pre-gain = the
            # same organ with mh_gain lifted out, so "silent" and
            # "loud-times-tiny" stop being the same reading. _mb feeds
            # THREE mixers (sc2, _mx_sc, _sm21) — one tensor, one record.
            _CENSUS.append((kb, "maskhead", _mb.realize().numpy()))
            _CENSUS.append((kb, "maskhead_pre",
                            ((_mh_sp - _mh_sp0) * _sm_kb)
                            .realize().numpy()))''')

# ===========================================================================
# 3. THE alt_g PATH — the symmetric snap-adjacency bias into sc2. Same
#    three-consumer story as the mask head; the pre-gain tensor is the
#    raw symmetrized adjacency.
# ===========================================================================
patch(3, "breath_step: alt + alt_pre at the alt_g bias into sc2",
      '''        sc2 = sc2 + (_A5 + _A5.transpose(-2, -1)) \\
            * p["alt_g"].reshape(1, 1, 1)''',
      '''        sc2 = sc2 + (_A5 + _A5.transpose(-2, -1)) \\
            * p["alt_g"].reshape(1, 1, 1)
        if _CENSUS is not None:
            _CENSUS.append((kb, "alt",
                            ((_A5 + _A5.transpose(-2, -1))
                             * p["alt_g"].reshape(1, 1, 1))
                            .realize().numpy()))
            _CENSUS.append((kb, "alt_pre",
                            (_A5 + _A5.transpose(-2, -1)).realize().numpy()))''')

# ===========================================================================
# 4. THE FED MIXER — its contribution INTO THE STATE (through W_bo), the
#    only one of the four that is genuinely state-band. `_mx_raw` and
#    `_mx_inj` are pure code motion (the `_dinj` precedent).
# ===========================================================================
patch(4, "breath_step: mixer + mixer_pre — the fed_mx_hg-gated contribution "
         "into h_slot, spoken through W_bo",
      '''        _mx_o = (_mx_sc.softmax(-1) @ _mx_v) \\
            * p["fed_mx_hg"].reshape(1, MX_HEADS, 1, 1)   # ZERO gains
        h_slot = h_slot + _mx_o.permute(0, 2, 1, 3) \\
            .reshape(B, L_TOT, H_W) @ p["W_bo"]''',
      '''        _mx_raw = _mx_sc.softmax(-1) @ _mx_v          # PRE-gain (named)
        _mx_o = _mx_raw \\
            * p["fed_mx_hg"].reshape(1, MX_HEADS, 1, 1)   # ZERO gains
        _mx_inj = _mx_o.permute(0, 2, 1, 3) \\
            .reshape(B, L_TOT, H_W) @ p["W_bo"]
        h_slot = h_slot + _mx_inj
        if _CENSUS is not None:
            # STATE band. The reader's grammar line says the rest: an
            # h_slot-band injection is further scaled by the breath gate
            # g = sigmoid(breath_gate[kb]) before it reaches cur.
            _CENSUS.append((kb, "mixer", _mx_inj.realize().numpy()))
            _CENSUS.append((kb, "mixer_pre",
                            (_mx_raw.permute(0, 2, 1, 3)
                             .reshape(B, L_TOT, H_W) @ p["W_bo"])
                            .realize().numpy()))''')

# ===========================================================================
# 5-6. THE KITCHEN SINK's two organs, as deltas IN STATE COORDINATES.
#      Both go through `_polar_ru_join` — the organ every other reader of
#      the polar state uses (the meter law) — so what the census reports
#      is r * (u_after - u_before): the injection the downstream organs
#      actually see, not a direction-space quantity nothing consumes.
#      `_cs_u0/_cs_u1` are plain python rebinds under the hook's guard:
#      no tensor op, no graph node, nothing when the hook is None.
# ===========================================================================
patch(5, "breath_step: sink_em — the E&B clock delta in state coordinates",
      '''            _pol_u = _polar_em(_pol_u, _sm_kb, POLAR_EM)''',
      '''            _cs_u0 = _pol_u if _CENSUS is not None else None
            _pol_u = _polar_em(_pol_u, _sm_kb, POLAR_EM)
            if _CENSUS is not None:
                _CENSUS.append((kb, "sink_em",
                                _polar_ru_join(_pol_u - _cs_u0, _pol_r,
                                               _pg).realize().numpy()))''')

patch(6, "breath_step: sink_waist — the content-waist delta in state "
         "coordinates",
      '''            _pol_u = _polar_waist(_pol_u, p, state)''',
      '''            _cs_u1 = _pol_u if _CENSUS is not None else None
            _pol_u = _polar_waist(_pol_u, p, state)
            if _CENSUS is not None:
                _CENSUS.append((kb, "sink_waist",
                                _polar_ru_join(_pol_u - _cs_u1, _pol_r,
                                               _pg).realize().numpy()))''')

# ===========================================================================
# 7. THE SLOT-SCORE BASELINE. The mask head and the alt bias are not
#    state-band organs: they are LOGIT-band, and dividing their rms by the
#    state's rms is a ratio of incommensurables. So the census records the
#    slot mixer's raw scores at their birth — the sc2 the biases are added
#    to — as `state_slot`, the baseline of THAT band. Without it the
#    registered mask-head bar ("< 0.02x of the base state") has no
#    in-band denominator to be evaluated against.
# ===========================================================================
patch(7, "breath_step: state_slot — the slot-mixer score baseline, the "
         "in-band denominator for the score-band organs",
      '''    sc2 = (_bq2 @ bk.transpose(-2, -1)) / math.sqrt(H_W)''',
      '''    sc2 = (_bq2 @ bk.transpose(-2, -1)) / math.sqrt(H_W)
    if _CENSUS is not None:
        _CENSUS.append((kb, "state_slot", sc2.realize().numpy()))''')

for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# ===========================================================================
# STRUCTURAL ASSERTS on the would-be module (cheap, no import, no GPU)
# ===========================================================================
NEW_ORGANS = ("altfact", "altfact_pre", "maskhead", "maskhead_pre",
              "alt", "alt_pre", "mixer", "mixer_pre",
              "sink_waist", "sink_em")
OLD_ORGANS = ("breath_emb", "notebook", "garage", "detwave", "router(bank)")
for _o in NEW_ORGANS:
    assert s.count(f'"{_o}",') == 1, \
        f'organ "{_o}" must be recorded exactly once, found {s.count(chr(34) + _o + chr(34) + chr(44))}'
# -- the gen-1 organs are UNTOUCHED: same name, same single capture site
for _o in OLD_ORGANS:
    assert s.count(f'_CENSUS.append((kb, "{_o}"') == 1, \
        f'gen-1 organ "{_o}" moved or changed arity — the extension must not touch it'
assert s.count('_CENSUS.append((kb, "state", cur.realize().numpy()))') == 1, \
    "the loop-state baseline record must survive untouched"
assert s.count('"state_slot"') == 1 and s.count('_cs.append((0, "state"') == 1, \
    "exactly one slot-score baseline and one vst baseline"
# -- 6 gen-1 sites + 12 new records (10 organs + TWO band baselines: the
#    kb=0 vst state and the per-breath slot-score state);
#    3 of the new ones ride `_cs` (the globals() handle inside _fact_inject)
assert s.count('_CENSUS.append(') == 15 and s.count('_cs.append(') == 3, \
    (f"expected 15 _CENSUS.append + 3 _cs.append after the patch, found "
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
# -- the two pre-gain rebinds cost nothing when the hook is unarmed
for _v in ('_cs_u0', '_cs_u1'):
    assert s.count(f'{_v} = _pol_u if _CENSUS is not None else None') == 1, \
        f"{_v} must be a guarded python rebind, never an unconditional tensor op"
# -- CODE MOTION ONLY: the three factored expressions keep their arithmetic
assert '_fi = fact_buf @ p["W_fact"]' in s and \
    'return vst + _fi * p["alt2_g"].reshape(1, 1, 1)' in s, \
    "the fact organ's arithmetic must be code-motion identical"
assert '_mx_raw = _mx_sc.softmax(-1) @ _mx_v' in s and \
    'h_slot = h_slot + _mx_inj' in s, \
    "the mixer's arithmetic must be code-motion identical"
assert '(fact_buf @ p["W_fact"]) * p["alt2_g"]' not in s, \
    "the old un-factored fact expression must be gone (one source, one meter)"
# -- POST and PRE for every gained organ (the whole point of the pass)
for _post, _pre in (("maskhead", "maskhead_pre"), ("alt", "alt_pre"),
                    ("altfact", "altfact_pre"), ("mixer", "mixer_pre")):
    _ip = s.index(f'"{_post}",')
    _iq = s.index(f'"{_pre}",')
    assert _ip < _iq, f"{_post} must be recorded before {_pre} (post, then pre)"
# -- the meter calls the ORGAN: the sink deltas go through _polar_ru_join
assert s.count('_polar_ru_join(_pol_u - _cs_u0, _pol_r,') == 1 and \
    s.count('_polar_ru_join(_pol_u - _cs_u1, _pol_r,') == 1, \
    "the sink census must express its deltas through _polar_ru_join"
# -- ORDERING inside breath_step: maskhead -> alt -> mixer, all before the
#    gate, and the sink records inside the polar block before the r*u join
_i_mb = s.index('_CENSUS.append((kb, "maskhead"')
_i_alt = s.index('_CENSUS.append((kb, "alt",')
_i_mx = s.index('_CENSUS.append((kb, "mixer"')
_i_gate = s.index('    g = p["breath_gate"][kb].sigmoid()')
_i_slot = s.index('_CENSUS.append((kb, "state_slot"')
_i_em = s.index('_CENSUS.append((kb, "sink_em"')
_i_wst = s.index('_CENSUS.append((kb, "sink_waist"')
_i_join = s.index('        cur = _polar_ru_join(_pol_u, _pol_r, _pg)')
assert _i_slot < _i_mb < _i_alt < _i_mx < _i_gate < _i_em < _i_wst < _i_join, \
    ("census misplaced: the slot-score baseline -> maskhead -> alt -> mixer "
     "-> breath gate -> sink_em -> sink_waist -> the r*u join")
# -- the score-band baseline is the RAW sc2, captured before any bias lands
assert s.index('_CENSUS.append((kb, "state_slot"') < \
    s.index('    sc2 = sc2.clip(-1e4, 1e4)'), \
    ("state_slot must be the UNBIASED slot-mixer score — captured at sc2's "
     "birth, before the mask head, the alt bias, BEXIT or the close")
# -- no new envs, no new parameters, no supervision, no float32 literal
assert s.count('os.environ') == open(fn).read().count('os.environ'), \
    "the census extension must not add an env door"
import re as _re                                        # noqa: E402
_pk0 = _re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(', open(fn).read())
_pk1 = _re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(', s)
assert _pk0 == _pk1, "the census extension must not add a parameter"
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
           '_fact_inject', '_heads_of', '_polar_ru_join')
_seen = set()
for child in mod_tbl.get_children():
    if child.get_name() in AUDITED:
        _seen.add(child.get_name())
        audit(child, child.get_name())
assert _seen == set(AUDITED), f"audit missed {sorted(set(AUDITED) - _seen)}"

# ===========================================================================
# REPORT
# ===========================================================================
print(f"[census organs] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines, 0 deleted):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
print(f"[census organs] organs after the patch: "
      f"3 gen-1 read by the ledger (breath_emb, notebook, garage) "
      f"+ 2 gen-1 unread (detwave, router(bank)) + "
      f"{len(NEW_ORGANS)} new ({', '.join(NEW_ORGANS)})")
print("[census organs] bands: STATE (mixer, sink_waist, sink_em, notebook, "
      "garage, detwave, breath_emb) | SCORE (maskhead, alt, router(bank)) | "
      "VST/var-slot, kb=0 (altfact) — the reader labels them and never "
      "cross-compares a score-band cosine with a state-band one")
print("[census organs] 0 new params, 0 new envs, 0 supervision; three "
      "expressions factored by pure code motion (_fi, _mx_raw, _mx_inj)")
print("[census organs] symtable free-var audit PASS "
      f"({', '.join(AUDITED)})")
print("[census organs] contract: hook UNARMED (_CENSUS is None) = "
      "byte-identical to the applied head — every training step and every "
      "ordinary read pays nothing; proved on CPU by "
      "scripts/census_organs_smoke.py")
if CHECK:
    print("[census organs] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[census organs] APPLIED ({fn}); ast OK — run "
          "scripts/census_organs_smoke.py before trusting")
