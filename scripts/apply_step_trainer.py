"""apply_step_trainer.py — FINAL BOSS rung 0: the breath-step factoring
(2026-09-03; the final-boss ruling, countersigned).

Factors forward()'s breath-loop BODY into a module-level function
    breath_step(p, state, kb, ctx)
by PURE CODE MOTION: the body text is EXTRACTED from the file between
asserted anchors, dedented mechanically, and re-embedded under a
prologue (unpack state/ctx into the exact local names the body already
uses) and an epilogue (pack the rebound names back). forward() then
calls breath_step in its loop — behavior bit-identical by construction
(and verified bitwise by scripts/step_trainer.py --cpuprobe).

Also factored by the same code-motion discipline (the trainer needs
them callable with ITS tensors, single source of truth):
  _make_bank(p, waist, tokmask, B)  -> the bank closure (verbatim)
  _heads_of(p, s, vst, B)           -> the emission heads (verbatim);
                                       forward keeps heads_of(s, vst=vst)
  _fact_inject(p, vst, fact_buf)    -> the ALT2 injection line (verbatim)
And one inert hook in the _CENSUS/_IMP pattern:
  _STEP_TAP = None   module global; when a dict with {"hold": True} is
  installed, forward() builds ctx/state as always but SKIPS the fused
  loop and fills the tap with {ctx, state, waist, vst, vst_base, fst,
  qst, heads_of, B} — the step trainer's stage-0 seam. None (the
  default) = byte-identical behavior everywhere.

THE TRUE CROSS-BREATH STATE SET (discovered by reading the loop; the
state dict carries exactly this):
  cur                 (B, L_FAC, H_W)  gradient-carrying slot state
  breaths             list, appended per breath (the ladder loss feed)
  nb, nb_st           notebook shelf: list of inks (GRADIENT-CARRYING,
                      grows 1 -> K_B) + stamp table; born at kb == 1
  garage              deposit shelf (grows per breath; DETACHED under
                      ALG_BUSGARAGE >= 2 — gradient-free by law)
  snaps, snaps_g      lattice snap tuples (DETACHED); only [-1] is read
  rb_last             router bias, output-only (W_ra heads)
  m_c, anchor, cmt_logits, x_rel   RINGS pawl state (None when off)
ctx (per-forward constants): B, K_B, waist, tokmask, slot_mask, bank,
rot2, sync, drop, gmod, revoke, tail, reg, RINGS, XOUT, XARM,
XR_GRADED, XR_ELASTIC.

RUN ONLY AFTER any running chain exits (the running-chain law). --check
loads the file, asserts every anchor, builds the would-be result,
ast-parses it, runs the symtable free-variable audit, writes NOTHING.
"""
import ast
import builtins
import symtable
import sys

fn = 'scripts/phase1_algebra_head.py'
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

assert '_STEP_TAP' not in s and 'def breath_step(' not in s, \
    "step-trainer patch already present — refuse (idempotence guard)"

ANCHORS = []


def note(num, desc):
    ANCHORS.append((num, desc))


# ===========================================================================
# EXTRACTION (from the pristine text, before any replacement)
# ===========================================================================

# ---- anchor 1: the breath-loop header (region start) ----
A_LOOP = ('    if K_B > 1 and slot_mask is not None and "W_bo" in p:\n'
          '        cur = fst\n'
          '        for kb in range(1, K_B):\n')
assert s.count(A_LOOP) == 1, "anchor 1 MISSING/NOT-UNIQUE (loop header)"
note(1, "forward: breath-loop header (extraction start)")

# ---- anchor 2: the loop terminator (region end) ----
A_HEADS = '\n    def heads_of(s):'
assert s.count(A_HEADS) == 1, "anchor 2 MISSING/NOT-UNIQUE (heads_of def)"
note(2, "forward: heads_of def (extraction end)")

i0 = s.index(A_LOOP) + len(A_LOOP)
i1 = s.index(A_HEADS)
body_raw = s[i0:i1]
assert body_raw.rstrip().endswith('_garage.append(_wg4)'), \
    "body tail drifted — re-read the loop before patching"

ded = []
for ln in body_raw.split('\n'):
    if ln.strip() == '':
        ded.append('')
        continue
    assert ln.startswith(' ' * 12), f"under-indented body line: {ln[:50]!r}"
    ded.append(ln[8:])
body_ded = '\n'.join(ded)
# round-trip: re-indenting must reproduce the original body exactly
assert '\n'.join((' ' * 8 + l if l else '') for l in ded) == body_raw, \
    "dedent round-trip failed — refuse"
for bad in ('return', 'yield'):
    assert not any(l.strip().startswith(bad) for l in ded), \
        f"body contains {bad!r} — not a pure block, refuse"

# ---- anchor 3: the bank closure (extraction) ----
A_BANK0 = '    def bank(queries, nq, extra=None, pbias=None, rbias=None):\n'
A_BANK1 = '\n    _lb = None'
assert s.count(A_BANK0) == 1, "anchor 3 MISSING/NOT-UNIQUE (bank def)"
assert s.count(A_BANK1) == 1, "anchor 3b MISSING/NOT-UNIQUE (bank end)"
note(3, "forward: bank closure block (extraction)")
j0 = s.index(A_BANK0)
j1 = s.index(A_BANK1)
bank_block = s[j0:j1] + '\n'          # def at indent 4, body at 8 (verbatim)

# ---- anchor 4: the heads_of block (extraction) ----
A_HEADS_END = ('\n    if int(os.environ.get("ALG_MINE_BREATHS", "0")):\n'
               '        out_breaths = breaths')
assert s.count(A_HEADS_END) == 1, "anchor 4 MISSING/NOT-UNIQUE (heads end)"
note(4, "forward: heads_of body (extraction end)")
k0 = s.index(A_HEADS) + 1             # start of '    def heads_of(s):'
k1 = s.index(A_HEADS_END)
heads_block = s[k0:k1] + '\n'
heads_body_lines = heads_block.split('\n')[1:]   # drop the def line
hded = []
for ln in heads_body_lines:
    if ln.strip() == '':
        hded.append('')
        continue
    assert ln.startswith(' ' * 8), f"heads body indent drift: {ln[:50]!r}"
    hded.append(ln[4:])
heads_body_ded = '\n'.join(hded)

# ---- anchor 5: the ALT2 injection line (extraction + replacement) ----
A_INJ = ('        vst = vst + (fact_buf @ p["W_fact"]) '
         '* p["alt2_g"].reshape(1, 1, 1)\n')
assert s.count(A_INJ) == 1, "anchor 5 MISSING/NOT-UNIQUE (ALT2 injection)"
note(5, "forward: ALT2 fact-injection line")

# ---- anchor 6: vst birth line (for the pre-injection tap) ----
A_VST = '    vst, vat = bank(p["vq"], K_VARS, pbias=_lb)\n'
assert s.count(A_VST) == 1, "anchor 6 MISSING/NOT-UNIQUE (vst birth)"
note(6, "forward: vst birth (pre-injection tap point)")

# ---- anchor 7: module insertion point ----
A_FWD = '\ndef forward(p, trunk, tokmask, sent,'
assert s.count(A_FWD) == 1, "anchor 7 MISSING/NOT-UNIQUE (def forward)"
note(7, "module: insertion point before def forward")

# ===========================================================================
# CONSTRUCTION
# ===========================================================================

MODULE_BLOCK = '''
_STEP_TAP = None    # the step trainer's stage-0 seam (the _CENSUS/_IMP
                    # hook pattern): None everywhere except under
                    # scripts/step_trainer.py — inert in every other path


def _make_bank(p, waist, tokmask, B):
    """forward()'s bank attention, factored BY PURE CODE MOTION
    (apply_step_trainer.py, 2026-09-03) so the step trainer can rebuild
    the closure over ITS OWN waist tensor. forward's call sites are
    unchanged; behavior bit-identical by construction."""
''' + bank_block + '''    return bank


def _heads_of(p, s, vst, B):
    """forward()'s emission heads, factored BY PURE CODE MOTION
    (apply_step_trainer.py, 2026-09-03): the step trainer runs these on
    intermediate breath states at every seam (commit adapter) and on the
    final state with the seam-current vst. Single source of truth."""
''' + heads_body_ded + '''

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
''' + body_ded + '''
    state["cur"] = cur; state["nb"] = _nb; state["nb_st"] = _nb_st
    state["rb_last"] = _rb_last
    state["m_c"] = m_c; state["anchor"] = anchor; state["x_rel"] = x_rel
    return state

'''

NEW_LOOP = '''    _bs_ctx = _bs_state = None
    if K_B > 1 and slot_mask is not None and "W_bo" in p:
        cur = fst
        # FINAL BOSS rung 0 (2026-09-03): the loop BODY lives in
        # module-level breath_step (pure code motion — bit-identical by
        # construction); state carries what crosses breath boundaries,
        # ctx the per-forward constants. Under _STEP_TAP hold the fused
        # loop is SKIPPED — the step trainer drives the walk itself.
        _bs_ctx = {"B": B, "K_B": K_B, "waist": waist, "tokmask": tokmask,
                   "slot_mask": slot_mask, "bank": bank, "rot2": _rot2,
                   "sync": _sync, "drop": drop, "gmod": gmod,
                   "revoke": revoke, "tail": tail, "reg": reg,
                   "RINGS": RINGS, "XOUT": XOUT, "XARM": XARM,
                   "XR_GRADED": XR_GRADED, "XR_ELASTIC": XR_ELASTIC}
        _bs_state = {"cur": cur, "breaths": breaths, "nb": None,
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
'''

NEW_HEADS = '''    def heads_of(s, vst=vst):
        return _heads_of(p, s, vst, B)
    if _STEP_TAP is not None:
        # the step trainer's stage-0 seam: everything the per-step walk
        # needs, single source (inert when None — the _CENSUS pattern)
        _STEP_TAP.update(ctx=_bs_ctx, state=_bs_state, waist=waist,
                         vst=vst, vst_base=_vst_base, fst=fst, qst=qst,
                         heads_of=heads_of, B=B)
'''

# ===========================================================================
# REPLACEMENT (each site exactly once)
# ===========================================================================

# 1+2: the loop region -> breath_step calls
s = s.replace(A_LOOP + body_raw, NEW_LOOP, 1)
# 3: the bank closure -> _make_bank call
s = s.replace(bank_block, '    bank = _make_bank(p, waist, tokmask, B)\n', 1)
# 4: heads_of -> thin wrapper (vst default arg) + tap fill
s = s.replace(heads_block, NEW_HEADS, 1)
# 5: injection line -> the factored organ
s = s.replace(A_INJ, '        vst = _fact_inject(p, vst, fact_buf)\n', 1)
# 6: pre-injection tap point
s = s.replace(A_VST, A_VST + '    _vst_base = vst   # pre-injection tap '
              '(step trainer reads this)\n', 1)
# 7: module block before forward
s = s.replace(A_FWD, '\n' + MODULE_BLOCK + '\ndef forward(p, trunk, '
              'tokmask, sent,', 1)

# ===========================================================================
# VERIFICATION (cheap, no import, no GPU)
# ===========================================================================

tree = ast.parse(s)                       # the would-be result must parse

# structural asserts
assert s.count('_STEP_TAP') == 6, \
    f"expected 6 _STEP_TAP sites (def, loop comment+guard x2, tap fill x2), got {s.count('_STEP_TAP')}"
assert 'breath_step(p, _bs_state, kb, _bs_ctx)' in s
assert 'bank = _make_bank(p, waist, tokmask, B)' in s
assert 'return _heads_of(p, s, vst, B)' in s
assert 'vst = _fact_inject(p, vst, fact_buf)' in s
assert s.count('def breath_step(') == 1
assert body_ded in s, "factored body text lost in re-embedding"
assert s.index('def breath_step(') < s.index('def forward(p, trunk'), \
    "breath_step must precede forward"

# the symtable free-variable audit: every name breath_step/_make_bank/
# _heads_of reads globally must be a module-level name (or a known
# dynamic global, or a builtin) — the loud door against a missed local
mod_tbl = symtable.symtable(s, fn, 'exec')
module_names = set(mod_tbl.get_identifiers())
DYNAMIC_OK = {'_CENSUS', '_IMP', '_SEV', '_SGC', '_BINDC'}
BUILTIN = set(dir(builtins))


def audit(tbl, fname):
    bad = set()
    for sym in tbl.get_symbols():
        n = sym.get_name()
        if sym.is_global() and n not in module_names \
                and n not in DYNAMIC_OK and n not in BUILTIN:
            bad.add(n)
    for ch in tbl.get_children():
        bad |= audit(ch, fname)
    assert not bad, f"{fname}: unresolved free variables {sorted(bad)}"
    return set()


for child in mod_tbl.get_children():
    if child.get_name() in ('breath_step', '_make_bank', '_heads_of',
                            '_fact_inject'):
        audit(child, child.get_name())

print(f"[step-trainer patch] {len(ANCHORS)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines):")
for num, desc in ANCHORS:
    print(f"  {num:2d}. {desc}")
print("[step-trainer patch] symtable free-var audit PASS "
      "(breath_step/_make_bank/_heads_of/_fact_inject)")
if CHECK:
    print("[step-trainer patch] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print("[step-trainer patch] APPLIED; ast OK — run step_trainer.py "
          "--cpuprobe (bitwise fused-vs-partitioned on CPU) before trusting")
