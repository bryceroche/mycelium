"""apply_tok_seal.py — THE TOKEN-ATTENTION SEVERANCE PROBE, staged patch
(2026-09-09; registered in the ledger the same day, "REGISTERED: THE
TOKEN-ATTENTION SEVERANCE PROBE (the headroom read that precedes any
token cooker)"). A STAGED patch: it anchors into the APPLIED head
(scripts/phase1_algebra_head.py as it stands — polar waist + kitchen sink
+ pressure mix + mask cooker + census hooks all present) and under
--check writes NOTHING. The lead applies. TS_TARGET may point at a copy
(rehearsal); default scripts/phase1_algebra_head.py.

WHY (THE HEADROOM COROLLARY, ledger 2026-09-09). The mask cooker woke the
player from 1e-5x to 0.84x of the score band and NOTHING moved, because
the slot mask's removal cost is zero (sever it: 0.2511 -> 0.2521). The
law that came out of it: a cooker converts HEADROOM, and headroom is the
BYPASS'S REMOVAL COST, measured FIRST — sever, read the drop, THEN cook.
This patch is that read for the next candidate road: the SLOT->TOKEN
attention, the grounding itself (the hypothesis: dynamic attention has no
job among 24 slots that already share a notebook and a shelf; its whole
job is aiming 24 slots at 100-300 mostly-filler tokens, and the wild
failure mode is the mis-aimed pointer).

THIS IS A READ DOOR, NOT AN ORGAN. It adds no parameter, no loss term and
no training path: do_train REFUSES to start with it set (the guard is the
FIRST statement of do_train, ahead of load_alg — so the answer to "may it
be used inside do_train's _quick_val?" is NO, structurally, and a stale
door can never be baked into a JIT capture — THE UNLIT STOVE).

THE DOOR — ALG_TOK_SEAL in {0 (default, off), loop, all}:

  loop  breaths kb = 1..K_B-1 (1..6 in the champion) read the tokens
        through the UNIFORM distribution over the prompt's real tokens.
        Breath 0's grounding (`fst, fat = bank(p["fq"], L_TOT, ...)`) is
        UNTOUCHED: the slots are still grounded once, and only the
        re-reading is severed.
  all   breath 0's factor-bank read is flattened too — the floor: the
        machine never once aims a factor slot at a token.

The weights are flattened AT THEIR SOURCE — `at = sc.softmax(-1)` inside
the bank closure — so BOTH consumers of that tensor inherit the flat
attention by construction: the value read (`at @ vh` -> h_tok / fst) and
the returned head-average (`at.mean(1)` -> fat_cur / fat). ONE ROAD,
CONSISTENTLY (the polar B6 lesson, the mask cooker's `_mb` precedent).
The value read then IS the token mean for every slot at every sealed
breath; every downstream consumer of `fat_cur` sees the same flat field:

  1. h_tok  — the bank's own value read (the road; the reason the
              severance is a severance at all).
  2. c_j    — ALG_CLOCK's sentence-completion clock,
              `(fat_cur * tail).sum(-1) / (fat_cur.sum(-1) + 1e-6)`.
              FLATTENED (it becomes the tail-token FRACTION of the
              prompt, one number per row): the clock is a READING of the
              attention, and a probe that left it reading a live
              attention while the road was flat would be measuring a
              machine that does not exist. (OFF in the champion env —
              ALG_CLOCK is unset in mask_cook_chain.sh — so on the
              registered read it is moot; it is flattened anyway because
              the source is the honest place to cut.)
  3. _nlw   — the NL tap (`ALG_MINE_BREATHS`), the paired atlas's miner.
              FLATTENED; a miner must see the machine that ran, and the
              tap is inert unless armed (never in training).
  4. _mcsk  — the mask cooker's `sentence` skeleton, which pushes
              fat_cur through the token->sentence map. FLATTENED; under
              a flat reading every slot's sentence distribution is the
              corpus-frequency one and the skeleton degenerates toward
              all-lanes — correct, because that IS the reading in force
              (the meter-divergence law: a check must CALL its organ).
              Only live under ALG_MASK_COOK_SKEL=sentence + a seal.

THE SECOND SLOT->TOKEN ROAD, AND WHY IT IS SEALED TOO (registered, not
hidden). ALT21 STATION 3 (`ALG_ALT21=1`, ON in the champion family) is a
SECOND bank-attention, slots<-tokens, with its own qkv over the same
waist and the same tokmask — it does not produce `fat_cur`, so a naive
"flatten fat_cur" seal would leave a fully live grounding road open, and
outcome (B) "no headroom" could then be an artifact of the BYPASS rather
than a fact about grounding. The headroom corollary is precisely about
removal cost, so the removal must be complete. Station 3 is therefore
flattened by the SAME organ (`_tok_flat`) whenever a loop breath is
sealed. `ALG_TOK_SEAL_S3=0` re-opens it for the NARROW arm (fq bank
only) — the texture read, free, on the same build.

WHAT IS DELIBERATELY NOT SEALED (registered):
 (1) the VAR bank `vst, vat = bank(p["vq"], K_VARS, pbias=_lb)` and the
     QUERY bank `qst, _qa = bank(p["qq"], 1)`. The var slots are the
     POINTER TARGET SPACE (args/res point into vst, letters bind there);
     flattening them would not sever a reading, it would delete the
     alphabet, and the drop would answer a question nobody asked. The
     probe is about which WORDS a FACTOR slot reads.
 (2) the slot<->slot mixers (sc2, the FED twin, station 4), the notebook,
     the garage, the polar block, the mask head, the parse heads: every
     one of them is untouched. This patch cuts exactly one kind of edge.

THE UNIFORM. `_tok_flat(tokmask, B)` = tokmask / max(tokmask.sum(-1), 1),
shape (B, 1, 1, T), broadcast over heads and slots: exactly 1/n_tok on
every real token, exactly 0 on every pad. The guard is `.maximum(1.0)`
rather than a `+1e-6` denominator ON PURPOSE and the deviation is
registered: a +1e-6 guard puts the error at 1e-6/(n(n+1e-6)) which AT
n = 1 is ~1e-6 — exactly the proof's tolerance, i.e. a bar that passes
by luck. Token counts are integer-valued floats, so `.maximum(1.0)` is
the IDENTITY on every row with at least one token (all of them) and
yields 0/1 = 0 on an empty row: exact where it matters, guarded where it
does not. Nothing else divides, so nothing else is where-gated.

The substitution is `at = at * 0.0 + _tok_flat(...)`, not `at = _tok_flat
(...)`: the zero-multiply keeps every parameter of the query/key path in
the graph with DEFINED (zero) gradients — the head's own ablation idiom
(`ALG_BREATH_ARM`: `h_tok = h_tok * 0.0`) and the None-grad lesson. A
softmax output is finite in [0, 1], so `at * 0.0` is exactly zero.

--check: loads the file, asserts every anchor present and unique, builds
the would-be result, ast-parses it, runs the structural asserts (including
an AST proof that the do_train guard is the function's first executable
statement and precedes load_alg) and the symtable free-variable audit,
and writes NOTHING. The CPU proofs live in scripts/tok_seal_smoke.py,
which stages this patch in memory exactly as mask_cook_smoke.py does.
"""
import ast
import builtins
import os
import symtable
import sys

fn = os.environ.get("TS_TARGET", 'scripts/phase1_algebra_head.py')
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

# ---------------------------------------------------------------- guards
assert 'ALG_TOK_SEAL' not in s and '_tok_flat' not in s, \
    "the token seal is already present — patch was applied; refuse (idempotence)"
assert 'def bank(queries, nq, extra=None, pbias=None, rbias=None):' in s, \
    "the bank closure is not in this head — the seal cuts its softmax"
assert 'h_tok, fat_cur = bank(p["fq"], L_TOT, extra=q_extra,' in s, \
    "the per-breath token attention is missing (wrong vintage of the head)"
assert '_st21 = (_sa21.softmax(-1) @ _vh21)' in s, \
    "ALT21 station 3 is missing — the second slot<-token road the seal " \
    "must close (anchor into the head AS IT IS)"
assert '_mask_cook_skel' in s and '_polar_em(_pol_u, _sm_kb, POLAR_EM)' in s, \
    "the mask cooker / polar sink are missing — anchor into the APPLIED head"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# ---------------------------------------------------------------------
# 1. module scope: the once-print flag (the _POLAR_SHOWN / _MC_SHOWN idiom)
# ---------------------------------------------------------------------
patch(1, "module: _TS_SHOWN (the door's once-print flag)",
      '''_MC_SHOWN = False        # THE MASK COOKER's doors: printed once''',
      '''_MC_SHOWN = False        # THE MASK COOKER's doors: printed once
_TS_SHOWN = False        # THE TOKEN SEAL's door: printed once''')

# ---------------------------------------------------------------------
# 2. module scope: the door's three helpers, immediately above _make_bank
# ---------------------------------------------------------------------
patch(2, "module: _tok_seal_mode / _tok_seal_on / _tok_flat (the door)",
      '''def _make_bank(p, waist, tokmask, B):''',
      '''def _tok_seal_mode():
    """THE TOKEN SEAL's door (apply_tok_seal.py, 2026-09-09; the
    registered TOKEN-ATTENTION SEVERANCE PROBE — the headroom read that
    precedes any token cooker). A READ door; do_train refuses to start
    with it set.

      "0"    (default) off — byte-inert, the same objects, the same
             kernels, the same bytes.
      "loop" loop breaths 1..K_B-1 read the tokens uniformly; breath 0's
             grounding is untouched.
      "all"  breath 0's factor-bank read is flattened too (the floor).

    A mistyped door must never read as OFF (the ALG_JIT_READ idiom), so
    anything else raises here rather than silently measuring the open
    machine."""
    _v = os.environ.get("ALG_TOK_SEAL", "0") or "0"
    assert _v in ("0", "loop", "all"), (
        f"ALG_TOK_SEAL={_v!r} is not one of 0 / loop / all — a mistyped "
        f"door must never read as OFF (it would silently report the "
        f"UNSEVERED machine as the severance read)")
    return _v


def _tok_seal_on(kb):
    """Is THIS breath's slot->token attention flattened? kb = 0 is
    breath 0's grounding read (only `all` cuts it); kb >= 1 is a loop
    breath (both `loop` and `all` cut it). One organ, both call sites —
    the road is decided in exactly one place."""
    _m = _tok_seal_mode()
    return _m == "all" or (_m == "loop" and kb >= 1)


def _tok_flat(tokmask, B):
    """THE UNIFORM READING: 1/n_tok on every real token of the row, 0 on
    every pad, shaped (B, 1, 1, T) to broadcast over heads and slots.

    NO epsilon denominator (registered): token counts are integer-valued
    floats >= 1, so `.maximum(1.0)` is the exact identity on every row
    that has tokens and turns the empty row into 0/1 = 0. A `+ 1e-6`
    guard would put the uniform off by ~1e-6 at n = 1 — the proof's own
    tolerance, i.e. a bar passed by luck. Nothing else here divides."""
    _u = tokmask.reshape(B, 1, 1, -1)
    return _u / _u.sum(-1, keepdim=True).maximum(1.0)


def _make_bank(p, waist, tokmask, B):''')

# ---------------------------------------------------------------------
# 3. the bank closure: the `flat` port (default False = today's path)
# ---------------------------------------------------------------------
patch(3, "bank: the `flat` port (default False — every old call site inert)",
      '''    def bank(queries, nq, extra=None, pbias=None, rbias=None):''',
      '''    def bank(queries, nq, extra=None, pbias=None, rbias=None, flat=False):''')

# ---------------------------------------------------------------------
# 4. the bank closure: THE SEVERANCE, at the weights' source
# ---------------------------------------------------------------------
patch(4, "bank: at = uniform over the real tokens (the severance)",
      '''        at = sc.softmax(-1)''',
      '''        at = sc.softmax(-1)
        if flat:
            # THE TOKEN SEAL (apply_tok_seal.py, 2026-09-09). The
            # slot->token attention WEIGHTS become the uniform
            # distribution over the prompt's real tokens (0 on pads),
            # so the value read below is the TOKEN MEAN for every slot.
            # Cut AT THE SOURCE: both consumers of `at` — the value
            # read `at @ vh` and the returned head-average `at.mean(1)`
            # (fat_cur / fat, and through it the clock c_j, the NL tap
            # and the cooker's sentence skeleton) — inherit the flat
            # field by construction. ONE ROAD, CONSISTENTLY.
            # The zero-multiply (not a plain rebind) keeps the q/k path
            # in the graph with defined zero grads: the ALG_BREATH_ARM
            # idiom, the None-grad lesson. softmax output is finite, so
            # `at * 0.0` is exactly zero.
            at = at * 0.0 + _tok_flat(tokmask, B)''')

# ---------------------------------------------------------------------
# 5. breath_step: the per-breath read takes the door
# ---------------------------------------------------------------------
patch(5, "breath_step: the loop breath's bank read passes flat=",
      '''    h_tok, fat_cur = bank(p["fq"], L_TOT, extra=q_extra,
                          pbias=(_sync[0](kb) if _sync is not None
                                 else None),
                          rbias=_rb7)''',
      '''    h_tok, fat_cur = bank(p["fq"], L_TOT, extra=q_extra,
                          pbias=(_sync[0](kb) if _sync is not None
                                 else None),
                          rbias=_rb7,
                          # THE TOKEN SEAL: this breath's RE-READING of
                          # the text. `loop` and `all` both cut it; the
                          # grounding at breath 0 is decided in
                          # forward(). Every consumer of fat_cur (the
                          # clock c_j, the NL tap, the cooker's sentence
                          # skeleton) inherits the flat field from here.
                          flat=_tok_seal_on(kb))''')

# ---------------------------------------------------------------------
# 6. breath_step: ALT21 station 3 — the SECOND slot<-token road
# ---------------------------------------------------------------------
patch(6, "breath_step: ALT21 station 3's token attention takes the door",
      '''        _st21 = (_sa21.softmax(-1) @ _vh21).permute(0, 2, 1, 3).reshape(B, L_TOT, H_W)''',
      '''        _a21 = _sa21.softmax(-1)
        if _tok_seal_on(kb) and int(os.environ.get("ALG_TOK_SEAL_S3", "1")):
            # THE TOKEN SEAL, second road (registered scope). Station 3
            # is a SECOND slots<-tokens bank attention over the same
            # waist and the same tokmask; leaving it live would leave
            # the grounding BYPASSABLE and a "no headroom" verdict would
            # be an artifact of the bypass, not a fact about grounding
            # (THE HEADROOM COROLLARY: removal cost must be the cost of
            # removing the road, all of it). ALG_TOK_SEAL_S3=0 gives the
            # NARROW arm (fq bank only) for the texture read.
            _a21 = _a21 * 0.0 + _tok_flat(tokmask, B)
        _st21 = (_a21 @ _vh21).permute(0, 2, 1, 3).reshape(B, L_TOT, H_W)''')

# ---------------------------------------------------------------------
# 7. forward: breath 0's grounding — cut only under `all`
# ---------------------------------------------------------------------
patch(7, "forward: breath 0's factor bank takes the door (`all` only)",
      '''    fst, fat = bank(p["fq"], L_TOT, pbias=_pb)
    qst, _qa = bank(p["qq"], 1)''',
      '''    # THE TOKEN SEAL (apply_tok_seal.py, 2026-09-09): breath 0's
    # GROUNDING. `loop` leaves it intact (the slots are grounded once
    # and only the re-reading is severed); `all` flattens it too — the
    # floor, where no factor slot ever aims at a token.
    # NOT SEALED, registered: the VAR bank above (vst — the pointer
    # TARGET space; flattening it deletes the alphabet rather than
    # severing a reading) and the QUERY bank below.
    fst, fat = bank(p["fq"], L_TOT, pbias=_pb, flat=_tok_seal_on(0))
    qst, _qa = bank(p["qq"], 1)''')

# ---------------------------------------------------------------------
# 8. do_train: THE GUARD — a read door never rides a training run
# ---------------------------------------------------------------------
patch(8, "do_train: refuse to train with the read door set (first statement)",
      '''def do_train(steps, lr, batch, seed):
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.jit import TinyJit
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_save

    samples, states, tokmask, gold, sent = load_alg("train")''',
      '''def do_train(steps, lr, batch, seed):
    from tinygrad import Tensor, dtypes
    from tinygrad.engine.jit import TinyJit
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_save

    # THE TOKEN SEAL IS A READ DOOR (apply_tok_seal.py, 2026-09-09).
    # It is a SEVERANCE for measuring the grounding's removal cost — not
    # an organ, not a regime, not a regularizer. Trained through, it
    # would (a) teach the machine to do without the reading, which is a
    # cooker and needs its own registration and its own dose, and (b)
    # bake the flat attention into the JIT capture where nothing would
    # ever print it again (THE UNLIT STOVE). The guard is do_train's
    # FIRST executable statement — ahead of load_alg, ahead of the
    # mask-prep pass, ahead of every capture — so `_quick_val` inside
    # this process cannot see the door either. Loud, like ALG_MASK_SEAL's.
    assert _tok_seal_mode() == "0", (
        f"ALG_TOK_SEAL={os.environ.get('ALG_TOK_SEAL')!r} is set and this "
        f"is a TRAINING run — the token seal is a READ door (the headroom "
        f"probe, ledger 2026-09-09). Training through it is a token "
        f"COOKER: a different thing, with its own dose, its own per-row "
        f"buffer and its own registration. Unset ALG_TOK_SEAL.")
    samples, states, tokmask, gold, sent = load_alg("train")''')

# ---------------------------------------------------------------------
# 9. build_params: the door, printed once
# ---------------------------------------------------------------------
patch(9, "build_params: print the door's state once",
      '''    global _POLAR_SHOWN, _MC_SHOWN''',
      '''    global _POLAR_SHOWN, _MC_SHOWN, _TS_SHOWN
    if not _TS_SHOWN and _tok_seal_mode() != "0":
        # THE TOKEN SEAL's door: one line, once, naming the severance
        # and its scope. Silent when the door is unset.
        _TS_SHOWN = True
        _ts_m = _tok_seal_mode()
        print(f"[tokseal] ALG_TOK_SEAL={_ts_m} — slot->token attention "
              f"WEIGHTS = uniform over the prompt's real tokens (0 on "
              f"pads) at "
              f"{'breath 0 AND loop breaths 1..K-1' if _ts_m == 'all' else 'loop breaths 1..K-1 (breath 0 grounding INTACT)'}"
              f"; ALG_TOK_SEAL_S3="
              f"{os.environ.get('ALG_TOK_SEAL_S3', '1')} (ALT21 station "
              f"3, the second slots<-tokens road: 1 = sealed too, 0 = "
              f"narrow fq-only arm). NOT sealed: the var bank (pointer "
              f"targets), the query bank, every slot<->slot mixer. "
              f"READ-ONLY: do_train refuses to start with this set.",
              flush=True)''')

for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# ===========================================================================
# STRUCTURAL ASSERTS on the would-be module (cheap, no import, no GPU)
# ===========================================================================
import re as _re                                              # noqa: E402

# -- ZERO new parameters: a read door is not capacity
_p0 = set(_re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(', open(fn).read()))
_p1 = set(_re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(', s))
assert _p0 == _p1, f"the seal added parameters: {sorted(_p1 - _p0)}"

# -- the severance happens at the WEIGHTS' source, exactly once per road
assert s.count('at = at * 0.0 + _tok_flat(tokmask, B)') == 1, \
    "the bank's weights must be flattened exactly ONCE, at their source " \
    "(so the value read AND at.mean(1) both inherit it — one road)"
assert s.count('_a21 = _a21 * 0.0 + _tok_flat(tokmask, B)') == 1, \
    "station 3's weights must be flattened exactly once"
assert s.count('def _tok_flat(') == 1, \
    "ONE uniform organ: both roads must call the same one (the " \
    "meter-divergence law — a check must CALL its organ, not rebuild it)"
_i_at = s.index('at = sc.softmax(-1)')
_i_flat = s.index('at = at * 0.0 + _tok_flat(tokmask, B)')
_i_use = s.index('st = (at @ vh)')
_i_ret = s.index('return st, at.mean(1)')
assert _i_at < _i_flat < _i_use < _i_ret, \
    "the flatten must sit between the softmax and BOTH of its consumers"
_i_s3f = s.index('_a21 = _a21 * 0.0 + _tok_flat(tokmask, B)')
assert s.index('_a21 = _sa21.softmax(-1)') < _i_s3f < s.index('_st21 = (_a21 @ _vh21)'), \
    "station 3's flatten must sit between its softmax and its value read"

# -- the uniform: no epsilon denominator, exact on non-empty rows
assert '_u / _u.sum(-1, keepdim=True).maximum(1.0)' in s, \
    "the uniform must be guarded by maximum(1.0), not a +1e-6 denominator " \
    "(1e-6 is the proof's own tolerance at n = 1 — a bar passed by luck)"
_helper_code = []
for _n in tree.body:
    if isinstance(_n, ast.FunctionDef) and _n.name in (
            "_tok_seal_mode", "_tok_seal_on", "_tok_flat"):
        for _b in _n.body:                       # docstrings are not code
            if isinstance(_b, ast.Expr) and isinstance(_b.value, ast.Constant):
                continue
            _helper_code.append(ast.get_source_segment(s, _b) or "")
_helper_code = chr(10).join(_helper_code)
assert '1e-6' not in _helper_code and '1e-9' not in _helper_code, \
    "no epsilon anywhere in the seal's helper CODE (only one thing " \
    "divides and it is guarded exactly, by maximum(1.0))"
assert _helper_code.count('_u.sum(-1, keepdim=True)') == 1, \
    "the uniform's normalizer is the token COUNT, computed once"

# -- the door is decided in ONE place, and both call sites ask it
assert s.count('def _tok_seal_on(') == 1 \
    and s.count('_tok_seal_on(kb)') == 3 \
    and s.count('_tok_seal_on(0)') == 1, \
    "the road must be decided by ONE organ: its def plus exactly two " \
    "loop call sites (the bank read + ALT21 station 3), plus breath 0 " \
    "in forward (which `all` alone cuts)"
assert 'flat=_tok_seal_on(kb))' in s and 'pbias=_pb, flat=_tok_seal_on(0))' in s, \
    "both bank call sites must pass the door"

# -- what is NOT sealed (registered scope, asserted structurally)
assert 'vst, vat = bank(p["vq"], K_VARS, pbias=_lb)' in s, \
    "the VAR bank must keep its live attention (it is the pointer TARGET " \
    "space — flattening it deletes the alphabet, it does not sever a read)"
assert 'qst, _qa = bank(p["qq"], 1)' in s, \
    "the QUERY bank must keep its live attention (registered scope)"
assert s.count('flat=') == 3 and s.count('flat=False') == 1, \
    "exactly two call sites take the door, plus the closure's default"

# -- the door defaults to OFF and is byte-inert unset
assert 'os.environ.get("ALG_TOK_SEAL", "0")' in s, \
    "the door must default to 0 (off) and be read by NAME (the jit_read " \
    "and mask-prep key builders mine env names from this source text)"
assert 'if flat:' in s, \
    "the seal must be behind a plain False default — no tensor work, no " \
    "kernel, nothing at all when the door is unset"

# -- THE GUARD: do_train's FIRST executable statement, ahead of load_alg
_dt = None
for _n in tree.body:
    if isinstance(_n, ast.FunctionDef) and _n.name == "do_train":
        _dt = _n
assert _dt is not None, "do_train vanished"
_body = [b for b in _dt.body
         if not isinstance(b, (ast.Import, ast.ImportFrom, ast.Expr))]
assert isinstance(_body[0], ast.Assert), \
    f"the guard must be do_train's first executable statement, got " \
    f"{type(_body[0]).__name__}"
_guard_src = ast.get_source_segment(s, _body[0]) or ""
assert '_tok_seal_mode() == "0"' in _guard_src, \
    "do_train's first statement must be the token-seal guard"
_i_guard = s.index('assert _tok_seal_mode() == "0"')
_i_load = s.index('samples, states, tokmask, gold, sent = load_alg("train")')
assert _i_guard < _i_load, "the guard must precede load_alg (nothing runs first)"
assert s.index('print(f"[breath] masks ready', _i_guard) > _i_guard, \
    "the guard must precede the mask-prep pass and every JIT capture"

# -- housekeeping the head's own laws impose
assert 'dtypes.float32' not in s, "no float32 dtype literal anywhere"
assert s.count('_TS_SHOWN = True') == 1 and s.count('[tokseal] ') == 1, \
    "the door prints exactly once, from one place"

# ===========================================================================
# THE SYMTABLE FREE-VARIABLE AUDIT (the apply_mask_cook.py idiom)
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
           '_make_bank', '_tok_seal_mode', '_tok_seal_on', '_tok_flat')
_seen = set()
for child in mod_tbl.get_children():
    if child.get_name() in AUDITED:
        _seen.add(child.get_name())
        audit(child, child.get_name())
assert _seen == set(AUDITED), f"audit missed {sorted(set(AUDITED) - _seen)}"

# ===========================================================================
# REPORT
# ===========================================================================
print(f"[tok seal] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines, 0 deleted):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
print("[tok seal] symtable free-var audit PASS "
      f"({', '.join(AUDITED)})")
print("[tok seal] NEW params: 0; NEW losses: 0; NEW training paths: 0 — a "
      "READ door. ALG_TOK_SEAL unset = byte-identical (scripts/"
      "tok_seal_smoke.py GATE 1, both polar configs, cold and warm)")
print("[tok seal] the severance: slot->token attention WEIGHTS -> uniform "
      "over the prompt's real tokens (0 on pads), cut at `at = sc.softmax"
      "(-1)` so the value read AND at.mean(1) (fat_cur -> clock c_j, NL "
      "tap, cooker sentence skeleton) both inherit it — one road")
print("[tok seal] doors: ALG_TOK_SEAL=0|loop|all (loop = breaths 1..K-1, "
      "breath 0 grounding intact; all = breath 0 too) + ALG_TOK_SEAL_S3=1 "
      "(default: ALT21 station 3, the SECOND slots<-tokens road, sealed "
      "too — else the grounding is bypassable; 0 = narrow fq-only arm)")
print("[tok seal] NOT sealed (registered): the var bank (pointer targets), "
      "the query bank, every slot<->slot mixer, the notebook, the garage, "
      "the polar block, the mask head, the parse heads")
print("[tok seal] the guard: do_train's FIRST executable statement refuses "
      "to start with the door set (ahead of load_alg, the mask-prep pass "
      "and every JIT capture — so _quick_val cannot see it either)")
if CHECK:
    print("[tok seal] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[tok seal] APPLIED ({fn}); ast OK — run "
          "scripts/tok_seal_smoke.py before trusting")
