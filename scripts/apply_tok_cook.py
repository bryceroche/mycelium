"""apply_tok_cook.py — THE TOKEN COOKER, staged patch (2026-09-09; spec
docs/token_cooker_spec.md, registered by THE TOKEN-ATTENTION SEVERANCE
PROBE the same day; the run needs the word). A STAGED patch: it anchors
into the APPLIED head (scripts/phase1_algebra_head.py at 8792773 — polar
waist + kitchen sink + census hooks + mask cooker + the tok-seal door all
present) and under --check writes NOTHING. The lead applies. TC_TARGET may
point at a copy (rehearsal); default scripts/phase1_algebra_head.py.

WHY (THE HEADROOM COROLLARY, obeyed). The mask cooker woke its organ from
1e-5x to 0.84x of the score band and NOTHING moved, because the slot
mask's removal cost was zero. So this cooker's pot was MEASURED FIRST, by
the severance probe: sever the loop's slot->token re-reading and wild
falls 0.2511 -> 0.1536, mint 0.9556 -> 0.7704 (station 3 carries 2/3 of
the wild headroom, the main bank 1/3). THERE IS A POT. This patch is the
cooker that converts it: on a dose of rows the mask head's own token gate
becomes the ONLY grounding road, and the loss must teach the head to aim.

THE BUILD (spec S1-2).

  THE GATE. A new projection W_tg — FACTORED, `tg_a` (H_W x TG_D) and
  `tg_b` (H_W x TG_D) at TG_D = 128, 2 x 512 x 128 = 131,072 params —
  scores the MASK HEAD's per-slot context state against the token bank's
  per-token states:

      g = (ctx @ tg_a) @ (waist @ tg_b)^T / sqrt(TG_D)        (B, L_TOT, T)
      gate = softmax(g.clip(-1e4, 1e4) + (1 - tokmask) * -1e4)

  and `gate` is a proper distribution over the row's REAL tokens: exactly
  0 on pads (exp(-1e4) underflows to 0.0, the head's own pad idiom — no
  -inf, no NaN, nothing divides), sums to 1 over the rest.

  THE SLOT STATE SCORED — `_mh_kv = cur + _mh_ce` (the mask head's
  per-slot context state), NOT `_mh_o`. Registered, because the choice is
  the difference between a live organ and a dead one: `_mh_o` is the
  head's pair-logit output and rides mh_wo, a ZERO-INIT door, so a gate
  scored on it would be EXACTLY uniform at birth AND have exactly zero
  gradient into tg_a (dL/dtg_a is proportional to the state scored) — the
  spec's "zero-born makes the gate uniform AND dead", precisely. `_mh_kv`
  is the state the head's own slot-pair logits are built from (`_mh_rp =
  _mh_o @ (_mh_kv @ mh_wp)^T`), it is nonzero at birth, and it carries
  exactly what the spec names: the polar/loop state `cur` (live, attached
  — detached from nothing) plus `_mh_ce`, the encoded context — solver
  facts, the graded committed edges, the domain mass, the previous
  breath's adjacency, the breath phase, and the atlas needle at rung 3.

  THE TOKEN STATE SCORED — `waist`, the token states BOTH banks project
  their keys from (`k = waist @ attn_wk`, `_k21 = waist @ alt21_attn_wk`).
  Registered, with the argument: scoring the bank's keys `k` instead
  would be the SAME hypothesis class — (ctx @ A)(waist @ Wk @ B)^T is
  (ctx @ A)(waist @ B')^T with B' = Wk B, plus a bias row — so `waist`
  loses no expressiveness, costs no duplicated (B,T,H_W)x(H_W,H_W)
  matmul, decouples the head's aim from the bank's own key path (the
  point: a SECOND pair of eyes, not a reparameterization of the first),
  and gives ONE gate that serves BOTH grounding roads. The head aims with
  its own eyes at the same words.

  BIRTH IS FUNCTIONAL, NOT DEAD. tg_a/tg_b are SMALL RANDOM (scale
  ALG_TG_INIT, default 1.0, on the 1/sqrt(H_W) rows) drawn from a
  DEDICATED rng stream (seed + 9200 — the FED law: a new tensor never
  moves the base stream, so every other parameter is bit-identical with
  the door on or off). The birth gate is therefore a mild, structured
  field near the uniform floor (its KL from uniform is measured and
  reported by scripts/tok_cook_smoke.py, not asserted in prose), with a
  live gradient into tg_a/tg_b from step one.

  THE SURGERY. On SEALED rows (`_TCV` = 1) at loop breaths 1..K_B-1 the
  main bank's `at` is REPLACED by the gate, at the WEIGHTS' SOURCE inside
  the bank closure — so BOTH consumers inherit it (the value read
  `at @ vh` AND the returned head-average `at.mean(1)` -> fat_cur, and
  through fat_cur the clock c_j, the NL tap and the mask cooker's
  sentence skeleton). ONE ROAD, CONSISTENTLY (the tok-seal precedent, the
  polar B6 lesson). Because the gate is head-independent, on a fully
  sealed row fat_cur IS the gate. Under ALG_TOK_COOK_S3=1 (default — it
  carries 2/3 of the wild headroom) ALT21 station 3's `_a21` is replaced
  by the SAME gate tensor. OPEN rows keep the banks' q.k bit-for-bit (the
  per-row blend at v in {0,1} is exact: 1.0*x = x, 0.0*finite = 0,
  x + 0 = x). BREATH 0 IS NEVER SEALED — forward()'s grounding read is
  not even wired to the port, so it is structural, not conditional.

  THE ORDERING PROBLEM, AND THE CODE MOTION THAT SOLVES IT (registered,
  because it is the one place this patch touches the mask head's own
  path). The bank read happens EARLY in breath_step; the mask head's
  context state is built LATE. The gate needs the state before the read.
  So the context construction is factored to module level BY PURE CODE
  MOTION into `_mh_ctx` — the block is lifted VERBATIM by this script
  (it slices the source, it does not retype it) — and MEMOIZED on the
  breath state per kb. With the cooker armed the gate site computes it
  and the mask head takes the memo HIT: ONE tensor, scored by the gate
  and read by the head (the meter-divergence law in its strongest form —
  the check does not rebuild its organ, it holds the same object). With
  every door unset the gate site never runs and `_mh_ctx` is called
  exactly once, with today's ops in today's order: bit-identical. The
  memo is safe by construction — `cur` is not rebound between the two
  sites (verified), `state["mh_prev"]` is written AFTER both, and the
  key is kb. `_A5` is factored the same way (`_mh_a5`) so the gate site
  and the mask head compute the graded adjacency from one definition.

THE DOORS (spec S2), all printed once:
  ALG_TOK_COOK=<share>   train dose (0 = off = byte-identical)
  ALG_TOK_COOK_S3=1      station 3 sealed too (default 1)
  ALG_TG_INIT=1.0        the birth scale of W_tg. At the default the birth
                         gate sits essentially AT the uniform floor the
                         severance probe measured (KL from uniform ~2e-3
                         nats against log n_tok ~4.7 on the CPU fixture) —
                         BY DESIGN: the floor (0.1536 wild) is where the
                         cooker starts and the bar is floor + 0.05. It is
                         small-random rather than ZERO because zero is
                         DEAD, not because a peaked birth would be better.
                         Re-measure the birth KL on the warm ckpt (a free
                         CPU read: ALG_TOK_SEAL=head + the census) before
                         turning this dial.
  TC_EVAL                pushed UNCONDITIONALLY around _quick_val — val
                         always compares the OPEN regime, at every shelf
                         mode. Non-empty = OPEN, wins over every door.
                         Arming asserts it is UNSET (a stale TC_EVAL
                         would bake the cooker OPEN into the JIT capture
                         — THE UNLIT STOVE, the forensic's own specimen).
  ALG_TOK_SEAL=head      the READ-TIME meter, a fourth value beside
                         0/loop/all: the head's gate is the grounding on
                         ALL rows at loop breaths 1..K_B-1. Barred in
                         do_train by the tok seal's existing FIRST
                         statement (`_tok_seal_mode() == "0"`), like
                         loop and all — a read door never rides a run.

THE CENSUS ORGAN `tokgate` (spec S3, the twin's bar 2). Under the
existing `_CENSUS` hook — inert on every training step and every ordinary
read — each sealed breath records the gate tensor (`tokgate`, token band,
rms by the reader's own grammar) and its mean KL FROM UNIFORM per slot,
one scalar per row (`tokgate_kl`). The KL is computed HERE, by the organ,
not rebuilt by the reader: log n_tok + sum g log g, with n_tok guarded by
.maximum(1.0) (the tok seal's registered guard — token counts are
integer-valued floats >= 1, so it is the exact identity) and log's
argument by .maximum(1e-30), which is EXACT at g = 0 (the term is
0 * anything = 0, on every pad) and the identity for every g >= 1e-30.
NO DIAGNOSTIC IS SUPERVISED: the only loss remains the parse loss (THE
GOODHART FENCE, mycelium/diagnostic_register.py).

NEW PARAMETERS: tg_a + tg_b = 131,072, and ONLY when a door is armed
(`ALG_TOK_COOK > 0` or `ALG_TOK_SEAL=head`) — so with the doors unset the
parameter SET is identical too and every banked checkpoint loads exactly
as before. Under WARM_FROM the two new keys take the "SKIP (fresh init)"
path by the pad-warm law; under do_eval's strict loader a checkpoint
without them hard-errors, which is correct and loud.

THE READ-ENV LAW FOR THIS ORGAN (stated because it is a footgun, and a
LOUD one by design). A checkpoint trained with ALG_TOK_COOK carries
tg_a/tg_b, and scripts/loop_val.py asserts `set(sd.keys()) ==
set(p.keys())`. So EVERY read of a cooked checkpoint must carry a door
that BIRTHS the gate: `ALG_TOK_COOK=<the training dose>` (completely
inert at read time — `_TCV` exists only inside do_train) or
`ALG_TOK_SEAL=head`. Without one the read stops on a key-set mismatch,
loudly, before it can report a number from the wrong machine. This is the
trained-env law's normal shape (a read env mirrors the training env, as
it already must for ALG_MASKHEAD, ALG_ALT21, the polar doors); it is
written here so the first person to run the guard read does not have to
rediscover it. The exception is scripts/port_census.py, which loads
tolerantly (`if k in sd`) and prints `[census] fresh-init: ['tg_a',
'tg_b']` — which is exactly how to read THE BIRTH GATE on the warm
checkpoint before spending a GPU hour: a free CPU census on
sharp_polarsink242 with ALG_TOK_SEAL=head reports the birth `aim`
(KL / log n_tok) on the real warm state, and that is the number that
should decide ALG_TG_INIT.

INDEPENDENCE FROM THE OTHER TWO COOKERS. `_TCV` is armed exactly like
`_PCV` and `_MCV` (a (B,1,1) data buffer created before the first step()
capture, assigned in place per step from the batch's DATASET row indices)
with a THIRD independent multiplier and addend:
  _PCV:  h(i) = (i * 2654435761)              mod 2^32 / 2^32
  _MCV:  h(i) = (i * 2246822519 + 2654435761) mod 2^32 / 2^32
  _TCV:  h(i) = (i * 3266489917 +  374761393) mod 2^32 / 2^32
so a row may be sealed by any subset of the three; all EIGHT cells occur
and each seal is exact per row (scripts/tok_cook_smoke.py GATE 6).

--check: loads the file, asserts every anchor present and unique, builds
the would-be result, ast-parses it, runs the structural asserts and the
symtable free-variable audit, and writes NOTHING. The CPU proofs live in
scripts/tok_cook_smoke.py, which stages this patch in memory exactly as
tok_seal_smoke.py stages the seal.
"""
import ast
import builtins
import os
import symtable
import sys

fn = os.environ.get("TC_TARGET", 'scripts/phase1_algebra_head.py')
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

# ---------------------------------------------------------------- guards
assert 'ALG_TOK_COOK' not in s and '_TCV' not in s and '_tok_gate' not in s, \
    "the token cooker is already present — patch was applied; refuse (idempotence)"
assert 'ALG_TOK_SEAL' in s and '_tok_flat' in s and '_tok_seal_mode' in s, \
    "the TOK SEAL door is missing — the cooker builds ON it (its `flat` " \
    "port, its mode alphabet, its do_train guard). Apply " \
    "scripts/apply_tok_seal.py first (wrong vintage of the head)"
assert '_MCV' in s and 'ALG_MASK_COOK' in s and '_PCV' in s, \
    "the mask cooker / pressure mix are missing — the token cooker is " \
    "their per-row twin and arms beside them"
assert '_mh_kv = cur + _mh_ce' in s and 'p["mh_enc1"]' in s, \
    "the MASK HEAD is not in this head — the gate is ITS token road"
assert '_polar_em(_pol_u, _sm_kb, POLAR_EM)' in s, \
    "the polar sink is missing — anchor into the APPLIED head (8792773)"
assert 'def bank(queries, nq, extra=None, pbias=None, rbias=None, flat=False):' in s, \
    "the bank closure's `flat` port is missing (the tok seal's signature)"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# ---------------------------------------------------------------------
# THE CODE MOTION, LIFTED FROM THE SOURCE (never retyped): the mask
# head's context block, verbatim, dedented one level into a function
# body. Slicing the file is what makes "PURE CODE MOTION" a fact rather
# than a claim — a transcription typo cannot survive it.
# ---------------------------------------------------------------------
_MH_B0 = '        _z1 = (cur[:, :, :1] * 0.0).detach()\n'
_MH_B1 = '        _mh_kv = cur + _mh_ce      # LIVE stream + detached context\n'
assert s.count(_MH_B0) == 1 and s.count(_MH_B1) == 1, \
    "the mask-head context block's boundaries are not unique"
_i0 = s.index(_MH_B0)
_i1 = s.index(_MH_B1) + len(_MH_B1)
assert _i0 < _i1, "the context block's end precedes its start"
MH_BLOCK = s[_i0:_i1]
for _need in ('_mh_cf = Tensor.cat(', '_mh_ce = ((_mh_cf @ p["mh_enc1"]',
              'ctx.get("fact_buf")', 'ctx.get("mh_mass")',
              'state.get("mh_prev")', 'ctx.get("mh_atlas")',
              'p["mh_atlas_w"]', 'ctx.get("fed_nl0")'):
    assert _need in MH_BLOCK, f"the lifted block is missing {_need!r}"
assert 'mh_wq' not in MH_BLOCK and '_sm_kb' not in MH_BLOCK, \
    "the lifted block must be the CONTEXT only — no attention, no mask"
MH_BODY = '\n'.join(_l[4:] if _l.startswith('    ') else _l
                    for _l in MH_BLOCK.split('\n'))

# ---------------------------------------------------------------------
# 1. module scope: the once-print flag (the _TS_SHOWN / _MC_SHOWN idiom)
# ---------------------------------------------------------------------
patch(1, "module: _TC_SHOWN (the doors' once-print flag)",
      '''_TS_SHOWN = False        # THE TOKEN SEAL's door: printed once''',
      '''_TS_SHOWN = False        # THE TOKEN SEAL's door: printed once
_TC_SHOWN = False        # THE TOKEN COOKER's doors: printed once''')

# ---------------------------------------------------------------------
# 2. the seal's mode alphabet grows a fourth value: `head`
# ---------------------------------------------------------------------
patch(2, "_tok_seal_mode: the alphabet grows `head` (the cooker's meter)",
      '''      "all"  breath 0's factor-bank read is flattened too (the floor).

    A mistyped door must never read as OFF (the ALG_JIT_READ idiom), so
    anything else raises here rather than silently measuring the open
    machine."""
    _v = os.environ.get("ALG_TOK_SEAL", "0") or "0"
    assert _v in ("0", "loop", "all"), (
        f"ALG_TOK_SEAL={_v!r} is not one of 0 / loop / all — a mistyped "''',
      '''      "all"  breath 0's factor-bank read is flattened too (the floor).
      "head" THE TOKEN COOKER's read-time meter (apply_tok_cook.py,
             2026-09-09): loop breaths 1..K_B-1 read the tokens through
             THE MASK HEAD'S TOKEN GATE on ALL rows — not the uniform
             floor, the head's own aim. It is the capability meter the
             cooker trains toward (bar: >= 0.1536 + 0.05 wild). It seals
             NOTHING flat: `_tok_seal_on` stays False for it, and the
             substitution happens through `_tok_cook_v`.

    A mistyped door must never read as OFF (the ALG_JIT_READ idiom), so
    anything else raises here rather than silently measuring the open
    machine."""
    _v = os.environ.get("ALG_TOK_SEAL", "0") or "0"
    assert _v in ("0", "loop", "all", "head"), (
        f"ALG_TOK_SEAL={_v!r} is not one of 0 / loop / all / head — a "
        f"mistyped "''')

# ---------------------------------------------------------------------
# 3. module scope: the cooker's four organs, above _make_bank
# ---------------------------------------------------------------------
patch(3, "module: TG_D/TG_INIT + _tok_cook_v / _tok_gate / _mh_a5 / _mh_ctx",
      '''def _make_bank(p, waist, tokmask, B):''',
      '''TG_D = 128       # THE TOKEN GATE's factored rank (spec S2: W_tg is
                 # (H_W x TG_D)(TG_D x H_W) = 2 x 512 x 128 = 131,072)


def _tok_cook_arm():
    """Is the TOKEN COOKER armed AT ALL (params must exist)? Either the
    train dose or the read-time meter. False -> zero new parameters, zero
    new tensors, zero new kernels: the head is byte-identical."""
    return (float(os.environ.get("ALG_TOK_COOK", "0")) > 0.0
            or _tok_seal_mode() == "head")


def _tok_cook_v():
    """THE TOKEN COOKER's per-row seal value (apply_tok_cook.py,
    2026-09-09; spec docs/token_cooker_spec.md). Returns

      None        nothing is sealed — today's path, bit-for-bit;
      1.0         EVERY row is sealed: ALG_TOK_SEAL=head, the READ-TIME
                  meter (token-sealed-wild / token-sealed-mint), the
                  head's gate as the grounding on all rows;
      (B,1,1,1)   the `_TCV` data buffer do_train arms — the _PCV/_MCV
                  idiom: one JIT graph, dynamic value, a STABLE
                  index-hash assignment (flat mix, never re-rolled).

    Shaped (B,1,1,1) to broadcast over an attention's (B, heads, slots,
    tokens) — the main bank's `at` and ALT21 station 3's `_a21` alike.

    VAL/READ HYGIENE: TC_EVAL non-empty forces the OPEN regime and wins
    over every other door (the MC_EVAL idiom — its own guard, because
    SC_EVAL's push is ALG_SHELF_CIRCLE>=2 only and a grounding organ must
    be excluded from val at every shelf mode). Outside do_train `_TCV`
    never exists, so ALG_TOK_COOK in a read env is inert by construction
    (the trained-env law; the _PCV/_MCV precedent)."""
    if os.environ.get("TC_EVAL", ""):
        return None                     # val/read compares OPEN
    if _tok_seal_mode() == "head":
        return 1.0                      # the read-time meter: all rows
    if float(os.environ.get("ALG_TOK_COOK", "0")) <= 0.0:
        return None
    _v = globals().get("_TCV")          # armed only inside do_train
    return None if _v is None else _v.reshape(-1, 1, 1, 1)


def _tok_gate(p, ctxst, waist, tokmask, B):
    """THE TOKEN GATE (spec S1): the mask head's per-slot distribution
    over the prompt's REAL tokens — the grounding road on sealed rows.

        g    = (ctxst @ tg_a) @ (waist @ tg_b)^T / sqrt(TG_D)
        gate = softmax(g.clip(-1e4, 1e4) + (1 - tokmask) * -1e4)

    ctxst is `_mh_kv` (the head's per-slot context state, attached);
    waist is the token states BOTH banks project their keys from. The
    pad mask is the head's OWN idiom — a -1e4 addend before the softmax,
    never a hard negative sentinel — so pads come out EXACTLY 0.0
    (exp(-1e4) underflows to zero, and every score is finite)
    and nothing here divides, so nothing needs a where-gate or an
    epsilon. Returns (B, L_TOT, T), head-independent by construction:
    every head of a sealed attention reads the same aim."""
    _q = ctxst @ p["tg_a"]                      # (B, L_TOT, TG_D)
    _k = waist @ p["tg_b"]                      # (B, T, TG_D)
    _g = (_q @ _k.transpose(-2, -1)) / math.sqrt(TG_D)
    _g = _g.clip(-1e4, 1e4) + (1.0 - tokmask.reshape(B, 1, -1)) * -1e4
    return _g.softmax(-1)


def _mh_a5(p, snaps):
    """The GRADED committed adjacency `_A5` (`_sr5 @ _sa5^T`), factored
    to module level BY PURE CODE MOTION (apply_tok_cook.py, 2026-09-09)
    so the token gate — which is computed BEFORE the bank read, earlier
    in breath_step than the mask head — and the mask head itself get it
    from ONE definition. Returns None exactly where the inline code did:
    no snaps, or neither alt_g nor ALG_MASKRE asking for it."""
    if snaps and ("alt_g" in p or int(os.environ.get("ALG_MASKRE", "0"))):
        _sa5 = snaps[-1][0] + snaps[-1][1]
        _sr5 = snaps[-1][2]
        return _sr5 @ _sa5.transpose(-2, -1)
    return None


def _mh_ctx(p, cur, state, ctx, kb, B, _A5, _snaps):
    """THE MASK HEAD'S PER-SLOT CONTEXT STATE (the live loop state plus
    its encoded context), factored to module level BY PURE CODE
    MOTION (apply_tok_cook.py, 2026-09-09; the step-trainer precedent — the block below is
    LIFTED from the head's source by the patch script, never retyped).

    WHY IT MOVED: the token gate must score this state, and the bank read
    it replaces happens EARLIER in breath_step than the mask head that
    builds it. One organ, two call sites, ONE tensor — the memo below
    keyed by kb makes the gate site's computation the ONE the mask head
    then reads (the meter-divergence law: a check must CALL its organ,
    and here it holds the very same object). The memo is safe because
    `cur` is not rebound between the two sites and `state["mh_prev"]` —
    the only state this block READS that the mask head WRITES — is
    written after both. With every cooker door unset the gate site never
    runs, this is the only call, and the ops are today's in today's
    order: bit-identical.

    Returns (_mh_kv, _A5s): the context state (ATTACHED — `cur` is the
    live stream; the metadata inside `_mh_ce` is detached at its own
    terminals, as it always was) and the detached symmetrized adjacency
    the caller stores as `state["mh_prev"]`."""
    from tinygrad import Tensor
    _memo = state.get("tc_mh_ctx")
    if _memo is not None and _memo[0] == kb:
        return _memo[1], _memo[2]
''' + MH_BODY + '''    state["tc_mh_ctx"] = (kb, _mh_kv, _A5s)
    return _mh_kv, _A5s


def _make_bank(p, waist, tokmask, B):''')

# ---------------------------------------------------------------------
# 4. the bank closure: the `tgate` / `tgv` ports (default None = inert)
# ---------------------------------------------------------------------
patch(4, "bank: the tgate/tgv ports (default None — every old call site inert)",
      '''    def bank(queries, nq, extra=None, pbias=None, rbias=None, flat=False):''',
      '''    def bank(queries, nq, extra=None, pbias=None, rbias=None, flat=False,
             tgate=None, tgv=None):''')

# ---------------------------------------------------------------------
# 5. the bank closure: THE SUBSTITUTION, at the weights' source
# ---------------------------------------------------------------------
patch(5, "bank: at = the head's gate on sealed rows (the cooker's road)",
      '''            at = at * 0.0 + _tok_flat(tokmask, B)
        st = (at @ vh).permute(0, 2, 1, 3).reshape(B, nq, H_W)''',
      '''            at = at * 0.0 + _tok_flat(tokmask, B)
        if tgate is not None:
            # THE TOKEN COOKER (apply_tok_cook.py, 2026-09-09; spec
            # docs/token_cooker_spec.md; the mandatory-road law + the
            # headroom corollary). On SEALED rows the slot->token
            # attention WEIGHTS become THE MASK HEAD'S TOKEN GATE — the
            # head's own aim is the only grounding road; open rows keep
            # the q.k weights bit-for-bit (the blend is exact at
            # v in {0,1}: 1.0*x = x, 0.0*finite = 0, x + 0 = x).
            # Cut AT THE SOURCE, like the seal: both consumers of `at`
            # — the value read `at @ vh` and the head-average
            # `at.mean(1)` (fat_cur, and through it the clock c_j, the
            # NL tap and the mask cooker's sentence skeleton) — inherit
            # the gate. ONE ROAD, CONSISTENTLY. The gate is
            # head-independent, so on a fully sealed row fat_cur IS the
            # gate. The zero-multiply on the open side keeps the q/k
            # path in the graph with defined grads (the ALG_BREATH_ARM
            # idiom, the None-grad lesson).
            assert extra is not None, (
                "the token gate is a LOOP-breath road: breath 0's "
                "grounding read (extra=None, batch dim 1) is never "
                "sealed and must never be handed a (B, ...) gate")
            at = at * (1.0 - tgv) + tgate.reshape(B, 1, nq, -1) * tgv
        st = (at @ vh).permute(0, 2, 1, 3).reshape(B, nq, H_W)''')

# ---------------------------------------------------------------------
# 6. breath_step: _A5 through the factored organ (one definition)
# ---------------------------------------------------------------------
patch(6, "breath_step: _A5 = _mh_a5(p, _snaps) (the factored adjacency)",
      '''    _A5 = None
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
                      .float()).clip(0, 1)''',
      '''    # THE TOKEN COOKER (apply_tok_cook.py, 2026-09-09): the graded
    # adjacency through `_mh_a5`, ONE definition, because the token
    # gate needs it EARLIER in this function than the mask head does
    # (see _mh_ctx). Same expression, same None cases, bit-identical.
    _A5 = _mh_a5(p, _snaps)
    if _A5 is not None and int(os.environ.get("ALG_MASKRE", "0")):
        # v2 THE MASK RE-FORMATION (2026-09-01, word given):
        # the HARD mask rebuilt per breath — OPEN-BY-
        # COMMITMENT (committed producer->consumer edges may
        # attend across the first-pass mask; additive-optional
        # per the ensemble law; NEVER tightens — A0's grave
        # stays honored)
        _sm_kb = (slot_mask
                  + ((_A5 + _A5.transpose(-2, -1)) > 0.5)
                  .float()).clip(0, 1)''')

# ---------------------------------------------------------------------
# 7. breath_step: the mask head's context block -> the factored organ
# ---------------------------------------------------------------------
patch(7, "breath_step: _mh_kv, _A5s = _mh_ctx(...) (pure code motion)",
      MH_BLOCK,
      '''        # THE TOKEN COOKER (apply_tok_cook.py, 2026-09-09): the
        # head's per-slot CONTEXT STATE, factored to module level BY
        # PURE CODE MOTION (the block moved verbatim into `_mh_ctx`,
        # sliced from this file by the patch script). MEMOIZED per
        # breath, so when the token gate built it a few lines above the
        # bank read this call is the memo HIT and the head and the gate
        # score the SAME tensor; with the cooker's doors unset this is
        # the only call and the ops are today's, in today's order.
        _mh_kv, _A5s = _mh_ctx(p, cur, state, ctx, kb, B, _A5, _snaps)
''')

# ---------------------------------------------------------------------
# 8. breath_step: the gate, computed once, BEFORE the bank read
# ---------------------------------------------------------------------
patch(8, "breath_step: the token gate (+ the tokgate census organ)",
      '''    h_tok, fat_cur = bank(p["fq"], L_TOT, extra=q_extra,''',
      '''    # THE TOKEN COOKER (apply_tok_cook.py, 2026-09-09; spec
    # docs/token_cooker_spec.md). THE GATE, computed ONCE per breath and
    # spent on BOTH grounding roads (this bank read and ALT21 station 3)
    # — one aim, not two. Inert (None) unless a door is armed, and then
    # only inside do_train unless ALG_TOK_SEAL=head is the read.
    _tcv = _tok_cook_v()
    _tgt = None
    if _tcv is not None:
        assert (int(os.environ.get("ALG_MASKHEAD", "0")) and "mh_wo" in p
                and "tg_a" in p), (
            "the token cooker needs the MASK HEAD (ALG_MASKHEAD=1, "
            "mh_wo in the params) and its token gate (tg_a/tg_b): the "
            "gate IS the head's road, and with no head there is no "
            "context state to aim from — a sealed row would read the "
            "tokens through an organ that does not exist")
        _tc_kv, _ = _mh_ctx(p, cur, state, ctx, kb, B,
                            _mh_a5(p, _snaps), _snaps)
        _tgt = _tok_gate(p, _tc_kv, waist, tokmask, B)
        if _CENSUS is not None:
            # THE CENSUS ORGAN `tokgate` (spec S3; the twin's bar 2).
            # The gate itself (token band — the reader's rms grammar)
            # and its mean KL FROM UNIFORM per slot, one scalar per row,
            # computed BY THE ORGAN so the reader quotes it rather than
            # rebuilding it. n_tok is guarded by .maximum(1.0) (the tok
            # seal's registered guard: token counts are integer-valued
            # floats >= 1, so it is the exact identity) and log's
            # argument by .maximum(1e-30), EXACT at g = 0 (the term is
            # 0 * anything = 0 on every pad) and the identity above it.
            # NOT SUPERVISED, ever: no diagnostic enters any loss.
            _tg_n = tokmask.sum(-1, keepdim=True).maximum(1.0)
            _tg_kl = (_tg_n.reshape(B, 1, 1).log()
                      + (_tgt * _tgt.maximum(1e-30).log())
                      .sum(-1, keepdim=True))
            _CENSUS.append((kb, "tokgate", _tgt.realize().numpy()))
            _CENSUS.append((kb, "tokgate_kl",
                            _tg_kl.mean(1, keepdim=True).realize().numpy()))
    h_tok, fat_cur = bank(p["fq"], L_TOT, extra=q_extra,''')

patch(9, "breath_step: the bank read takes the gate",
      '''                          flat=_tok_seal_on(kb))''',
      '''                          flat=_tok_seal_on(kb),
                          # THE TOKEN COOKER: the head's aim replaces
                          # this breath's q.k weights on SEALED rows,
                          # at the weights' source (so fat_cur and the
                          # value read inherit it together).
                          tgate=_tgt, tgv=_tcv)''')

# ---------------------------------------------------------------------
# 10. breath_step: ALT21 station 3 — the SECOND grounding road
# ---------------------------------------------------------------------
patch(10, "breath_step: ALT21 station 3 takes the SAME gate",
      '''        _st21 = (_a21 @ _vh21).permute(0, 2, 1, 3).reshape(B, L_TOT, H_W)''',
      '''        if _tgt is not None and int(os.environ.get("ALG_TOK_COOK_S3", "1")):
            # THE TOKEN COOKER, second road (spec S1; the severance
            # probe measured station 3 carrying 2/3 of the WILD
            # headroom). The SAME gate tensor — one aim, both roads —
            # blended per row exactly as the main bank's. S3=0 gives
            # the narrow fq-only arm. Leaving it live would leave the
            # grounding BYPASSABLE and the cooker would be cooking a
            # road the machine can walk around (the headroom
            # corollary's whole point, and the seal's own scope).
            _a21 = _a21 * (1.0 - _tcv) + _tgt.reshape(B, 1, L_TOT, -1) * _tcv
        _st21 = (_a21 @ _vh21).permute(0, 2, 1, 3).reshape(B, L_TOT, H_W)''')

# ---------------------------------------------------------------------
# 11. build_params: W_tg, born only when a door is armed
# ---------------------------------------------------------------------
patch(11, "build_params: tg_a/tg_b (131,072 params; dedicated rng stream)",
      '''            p["mh_gain"] = t(np.full(1, 0.02))      # AJAR (the law)''',
      '''            p["mh_gain"] = t(np.full(1, 0.02))      # AJAR (the law)
            if _tok_cook_arm():
                # THE TOKEN COOKER's W_tg (apply_tok_cook.py,
                # 2026-09-09; spec docs/token_cooker_spec.md S1-2), the
                # FACTORED form: tg_a (H_W x TG_D) and tg_b (H_W x
                # TG_D), 2 x 512 x 128 = 131,072 params. SMALL RANDOM,
                # NOT ZERO — the spec's own ruling: a zero-born W_tg
                # makes the gate exactly uniform AND exactly dead (the
                # gradient into tg_a is proportional to the state it
                # scores through tg_b, and vice versa), so the organ
                # would be born in the gate deadlock the mandatory-road
                # law exists to prevent. A DEDICATED rng stream (the
                # FED law: seed + 9000 for FED, + 7000 for LoRA,
                # + 9200 here) so arming the door moves no other
                # parameter's init by a single bit.
                _rngT = np.random.RandomState(seed + 9200)
                _tg_i = float(os.environ.get("ALG_TG_INIT", "1.0"))
                p["tg_a"] = t(_rngT.randn(H_W, TG_D)
                              / math.sqrt(H_W) * _tg_i)
                p["tg_b"] = t(_rngT.randn(H_W, TG_D)
                              / math.sqrt(H_W) * _tg_i)''')

# ---------------------------------------------------------------------
# 12. build_params: the doors, printed once
# ---------------------------------------------------------------------
patch(12, "build_params: print every cooker door's state once",
      '''    global _POLAR_SHOWN, _MC_SHOWN, _TS_SHOWN''',
      '''    global _POLAR_SHOWN, _MC_SHOWN, _TS_SHOWN, _TC_SHOWN
    if not _TC_SHOWN and (_tok_cook_arm()
                          or os.environ.get("ALG_TOK_COOK_S3", "")
                          or os.environ.get("ALG_TG_INIT", "")
                          or os.environ.get("TC_EVAL", "")):
        # THE TOKEN COOKER's doors (spec S2): one line, once, naming
        # every one of them. Silent when all are unset.
        _TC_SHOWN = True
        print(f"[tokcook] doors: ALG_TOK_COOK="
              f"{os.environ.get('ALG_TOK_COOK', '0')} (train dose) "
              f"ALG_TOK_COOK_S3="
              f"{os.environ.get('ALG_TOK_COOK_S3', '1')} (ALT21 station "
              f"3, the second grounding road: 2/3 of the wild headroom) "
              f"ALG_TOK_SEAL={_tok_seal_mode()} "
              f"('head' = the read-time meter, ALL rows) "
              f"TC_EVAL={os.environ.get('TC_EVAL', '') or '(unset)'} "
              f"(non-empty = OPEN) ALG_TG_INIT="
              f"{os.environ.get('ALG_TG_INIT', '1.0')} | on SEALED rows "
              f"at loop breaths 1..K-1 the slot->token attention IS the "
              f"mask head's gate softmax((mh_kv @ tg_a)(waist @ tg_b)^T"
              f"/sqrt({TG_D})) over the real tokens; breath 0 never "
              f"sealed; W_tg = 2 x {H_W} x {TG_D} = {2 * H_W * TG_D} "
              f"params, born only with a door armed", flush=True)''')

patch(13, "build_params: the SEAL's print excludes `head` (it flattens nothing)",
      '''    if not _TS_SHOWN and _tok_seal_mode() != "0":''',
      '''    if not _TS_SHOWN and _tok_seal_mode() not in ("0", "head"):
        # `head` is THE TOKEN COOKER's meter, not a flattening: the
        # [tokseal] line below would describe a machine that is not
        # running (the meter-divergence law — a print is a check too).
        # The [tokcook] line above names it instead.''')

# ---------------------------------------------------------------------
# 14. do_train: arm _TCV (beside _MCV, before the first capture)
# ---------------------------------------------------------------------
patch(14, "do_train: arm _TCV + the THIRD independent stable index-hash",
      '''    t0 = time.time()
    for s in range(steps):''',
      '''    _tc_mix = float(os.environ.get("ALG_TOK_COOK", "0"))
    _tc_assign = None
    if _tc_mix > 0.0:
        # THE TOKEN COOKER (2026-09-09, docs/token_cooker_spec.md): the
        # per-row severance of the BANKS' slot->token grounding, which
        # the mask head's gate then replaces. Armed exactly like the
        # pressure mix and the mask cooker (a (B,1,1) buffer created
        # BEFORE the first step() capture — the JIT law) and INDEPENDENT
        # of both: a THIRD Knuth multiplier AND a third addend, so a row
        # may be sealed by any subset of the three cookers.
        assert int(os.environ.get("ALG_MASKHEAD", "0")) and "mh_wo" in p, (
            "ALG_TOK_COOK needs the mask head (ALG_MASKHEAD=1 and "
            "mh_wo in the params): the gate is the HEAD's road, scored "
            "from the head's own context state")
        assert "tg_a" in p and "tg_b" in p, (
            "ALG_TOK_COOK without W_tg — build_params was called before "
            "the door was set (the params are born only when armed)")
        assert _tok_seal_mode() == "0", (
            "ALG_TOK_SEAL is the READ-TIME meter; do_train's first "
            "statement already refuses it — this is the second wall")
        assert not os.environ.get("TC_EVAL", ""), (
            "ALG_TOK_COOK with TC_EVAL set would bake the cooker OPEN "
            "at JIT capture (THE UNLIT STOVE: training that never "
            "sealed) — unset TC_EVAL; val pushes it by itself")
        _tc_h = ((np.arange(n, dtype=np.uint64) * np.uint64(3266489917)
                  + np.uint64(374761393)) % np.uint64(4294967296)
                 ).astype(np.float64) / 4294967296.0
        _tc_assign = (_tc_h < _tc_mix).astype(np.float32)
        globals()["_TCV"] = Tensor(
            np.zeros((batch, 1, 1), np.float32)).contiguous().realize()
        _tc_pc = (float((_tc_assign * _pc_assign).sum()) / max(n, 1)
                  if _pc_assign is not None else 0.0)
        _tc_mc = (float((_tc_assign * _mc_assign).sum()) / max(n, 1)
                  if _mc_assign is not None else 0.0)
        print(f"[tokcook] grounding severance armed: share={_tc_mix} -> "
              f"{int(_tc_assign.sum())}/{n} rows sealed (stable "
              f"index-hash, multiplier 3266489917); S3="
              f"{os.environ.get('ALG_TOK_COOK_S3', '1')}; both-seals "
              f"pressure={_tc_pc:.4f} mask={_tc_mc:.4f} of rows; on "
              f"those rows the mask head's gate is the ONLY slot->token "
              f"road at loop breaths 1..K-1 (breath 0 untouched)",
              flush=True)
    t0 = time.time()
    for s in range(steps):''')

# ---------------------------------------------------------------------
# 15. do_train loop: the per-step assign, beside _MCV's
# ---------------------------------------------------------------------
patch(15, "do_train loop: per-step _TCV assign from the batch indices",
      '''        if _mc_assign is not None:
            globals()["_MCV"].assign(Tensor(
                _mc_assign[idx].reshape(-1, 1, 1),
                dtype=globals()["_MCV"].dtype)).realize()
        lv = step()''',
      '''        if _mc_assign is not None:
            globals()["_MCV"].assign(Tensor(
                _mc_assign[idx].reshape(-1, 1, 1),
                dtype=globals()["_MCV"].dtype)).realize()
        if _tc_assign is not None:
            globals()["_TCV"].assign(Tensor(
                _tc_assign[idx].reshape(-1, 1, 1),
                dtype=globals()["_TCV"].dtype)).realize()
        lv = step()''')

# ---------------------------------------------------------------------
# 16. do_train: the val push — the cooker's OWN guard
# ---------------------------------------------------------------------
patch(16, "do_train: TC_EVAL push around _quick_val (val stays OPEN)",
      '''            os.environ["MC_EVAL"] = "0"       # ... and the OPEN mask
            fv = _quick_val()
            os.environ.pop("MC_EVAL", None)''',
      '''            os.environ["MC_EVAL"] = "0"       # ... and the OPEN mask
            # THE TOKEN COOKER's own val guard (2026-09-09), pushed
            # UNCONDITIONALLY beside the mask cooker's and for the same
            # reason: SC_EVAL's push is ALG_SHELF_CIRCLE>=2 only, and a
            # grounding organ must be excluded from val at every shelf
            # mode. Any non-empty value means OPEN; only "0" is pushed.
            os.environ["TC_EVAL"] = "0"       # ... and the OPEN reading
            fv = _quick_val()
            os.environ.pop("TC_EVAL", None)
            os.environ.pop("MC_EVAL", None)''')

for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# ===========================================================================
# STRUCTURAL ASSERTS on the would-be module (cheap, no import, no GPU)
# ===========================================================================
import re as _re                                              # noqa: E402

# -- EXACTLY two new parameters, and they are the gate's
_p0 = set(_re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(', open(fn).read()))
_p1 = set(_re.findall(r'p\["([A-Za-z_0-9]+)"\]\s*=\s*t\(', s))
assert _p1 - _p0 == {"tg_a", "tg_b"} and not _p0 - _p1, \
    f"the cooker's parameter delta is not exactly W_tg: {sorted(_p1 ^ _p0)}"
assert s.count('if _tok_cook_arm():\n') == 1, \
    "W_tg must be born ONLY with a door armed (else the parameter SET " \
    "moves for every existing run and every banked ckpt)"
assert '_rngT = np.random.RandomState(seed + 9200)' in s, \
    "W_tg must draw from its OWN rng stream (the FED law: a new tensor " \
    "never moves the base stream — arming the door must not re-init the head)"

# -- the substitution happens at the WEIGHTS' source, exactly once per road
assert s.count('at = at * (1.0 - tgv) + tgate.reshape(B, 1, nq, -1) * tgv') == 1, \
    "the bank's weights must take the gate exactly ONCE, at their source " \
    "(so the value read AND at.mean(1) both inherit it — one road)"
assert s.count('_a21 = _a21 * (1.0 - _tcv) + _tgt.reshape(B, 1, L_TOT, -1) * _tcv') == 1, \
    "station 3's weights must take the gate exactly once"
assert s.count('def _tok_gate(') == 1 and s.count('_tok_gate(p, _tc_kv') == 1, \
    "ONE gate organ, computed ONCE per breath and spent on both roads"
_i_at = s.index('at = sc.softmax(-1)')
_i_flat = s.index('at = at * 0.0 + _tok_flat(tokmask, B)')
_i_tg = s.index('at = at * (1.0 - tgv) + tgate.reshape(B, 1, nq, -1) * tgv')
_i_use = s.index('st = (at @ vh)')
_i_ret = s.index('return st, at.mean(1)')
assert _i_at < _i_flat < _i_tg < _i_use < _i_ret, \
    "the gate must sit between the softmax and BOTH of its consumers, " \
    "after the seal's flatten (the seal is the floor; the gate is the road)"
_i_s3 = s.index('_a21 = _a21 * (1.0 - _tcv)')
assert s.index('_a21 = _sa21.softmax(-1)') < _i_s3 < s.index('_st21 = (_a21 @ _vh21)'), \
    "station 3's gate must sit between its softmax and its value read"

# -- the gate is computed BEFORE the read it replaces, and once
_i_gate = s.index('_tgt = _tok_gate(p, _tc_kv')
_i_bank = s.index('h_tok, fat_cur = bank(p["fq"], L_TOT, extra=q_extra,')
assert _i_gate < _i_bank < _i_s3, \
    "the gate must be computed before the bank read and reused at station 3"
assert s.count('_tcv = _tok_cook_v()') == 1, \
    "the per-row seal value is read ONCE per breath (one decision, two roads)"

# -- the pad mask is the head's own idiom; nothing divides; no -inf
_gate_src = s[s.index('def _tok_gate('):s.index('def _mh_a5(')]
assert '(1.0 - tokmask.reshape(B, 1, -1)) * -1e4' in _gate_src \
    and 'inf' not in _gate_src, \
    "the gate's pad mask must be the -1e4 addend (never -inf)"
assert _gate_src.count('/') == 2 and '/ math.sqrt(TG_D)' in _gate_src, \
    "the only division in the gate is the score scale (no epsilon guards, " \
    "because nothing else divides)"
assert '.clip(-1e4, 1e4)' in _gate_src, "the scores must be clipped (the head's law)"

# -- the code motion is a MOTION: the block left one site and entered one
assert s.count('_mh_kv = cur + _mh_ce') == 1 and \
    s.count('_mh_kv, _A5s = _mh_ctx(p, cur, state, ctx, kb, B, _A5, _snaps)') == 1, \
    "the context block must exist exactly once, inside _mh_ctx, and be " \
    "called from exactly one site in breath_step"
assert s.count('def _mh_ctx(') == 1 and s.count('_mh_ctx(') == 3, \
    "one definition, two call sites (the gate's and the mask head's)"
assert s.count('state["tc_mh_ctx"] = (kb, _mh_kv, _A5s)') == 1 and \
    s.count('_memo = state.get("tc_mh_ctx")') == 1, \
    "the memo is written and read in exactly one place each"
assert s.count('def _mh_a5(') == 1 and s.count('_mh_a5(p, _snaps)') == 2, \
    "the graded adjacency has one definition and two call sites"
assert '_sr5 @ _sa5.transpose(-2, -1)' in s and \
    s.count('_sr5 @ _sa5.transpose(-2, -1)') == 1, \
    "_A5's expression must survive the motion unchanged, once"

# -- breath 0 is NEVER sealed: forward's read is not wired to the port
_fwd = s[s.index('    fst, fat = bank(p["fq"], L_TOT, pbias=_pb'):]
_fwd = _fwd[:_fwd.index('\n')]
assert 'tgate' not in _fwd and 'flat=_tok_seal_on(0)' in _fwd, \
    "breath 0's grounding read must not take the gate port at all — the " \
    "floor of everything is structural here, not conditional"
assert s.count('tgate=') == 2 and s.count('tgate=None') == 1, \
    "exactly ONE call site takes the gate, plus the closure's default"

# -- val hygiene: the push exists, is unconditional, and brackets the call
_i_push = s.index('os.environ["TC_EVAL"] = "0"')
_i_val = s.index('fv = _quick_val()')
_i_pop = s.index('os.environ.pop("TC_EVAL", None)')
assert _i_push < _i_val < _i_pop, \
    "TC_EVAL must be pushed before _quick_val and popped after"
assert s[s.rindex('\n', 0, _i_push) + 1:_i_push] == ' ' * 12, \
    "the TC_EVAL push must be UNCONDITIONAL (it sits at val's indent, not " \
    "inside the ALG_SHELF_CIRCLE branch)"
assert s.count('if os.environ.get("TC_EVAL", ""):\n        return None') == 1, \
    "TC_EVAL must win over every other door in _tok_cook_v"

# -- arming: after the mask-prep pass, before the first capture; per-step feed
assert s.index('print(f"[breath] masks ready') < s.index('_tc_assign = (_tc_h'), \
    "the buffer must be armed AFTER the mask-prep pass"
assert s.index('_tc_assign = (_tc_h') < s.index('    t0 = time.time()'), \
    "arming must precede the step loop (the JIT law)"
assert s.index('_tc_assign[idx]') < s.index('lv = step()'), \
    "the per-step assign must precede step()"
assert s.index('_mc_assign[idx]') < s.index('_tc_assign[idx]'), \
    "the _TCV assign rides beside (after) the _PCV/_MCV assigns"

# -- three DIFFERENT hashes (independence is the whole point)
assert s.count('np.uint64(3266489917)') == 1 and \
    'np.uint64(3266489917)\n                  + np.uint64(374761393)' in s, \
    "the token assignment must use its own multiplier AND its own addend"
for _mult in ('2654435761', '2246822519', '3266489917'):
    assert _mult in s, f"hash seed {_mult} lost"

# -- THE READ DOOR IS STILL BARRED IN do_train (now for `head` too)
_dt = [n for n in tree.body
       if isinstance(n, ast.FunctionDef) and n.name == "do_train"][0]
_body = [b for b in _dt.body
         if not isinstance(b, (ast.Import, ast.ImportFrom, ast.Expr))]
assert isinstance(_body[0], ast.Assert) and \
    '_tok_seal_mode() == "0"' in (ast.get_source_segment(s, _body[0]) or ""), \
    "the tok seal's guard must remain do_train's first executable " \
    "statement — it is what bars ALG_TOK_SEAL=head from a training run"
assert '"0", "loop", "all", "head"' in s, "the fourth door value must be legal"
assert '_tok_seal_mode() not in ("0", "head")' in s and \
    s.count('if not _TS_SHOWN and') == 1, \
    "the [tokseal] line must NOT fire for `head` — it would describe a " \
    "flattening that is not happening (the meter-divergence law)"

# -- the census organ is present, hook-gated, and NEVER supervised
assert s.count('_CENSUS.append((kb, "tokgate"') == 1 and \
    s.count('_CENSUS.append((kb, "tokgate_kl"') == 1, \
    "the tokgate census organ records the gate and its KL, once each"
_i_hook = s.index('        if _CENSUS is not None:\n            # THE CENSUS ORGAN')
assert _i_hook < s.index('_CENSUS.append((kb, "tokgate"'), \
    "the census records must sit behind the hook (byte-inert unarmed)"
_tg_kl_src = s[s.index('_tg_n = tokmask.sum'):s.index('h_tok, fat_cur = bank')]
assert '.maximum(1.0)' in _tg_kl_src and '.maximum(1e-30)' in _tg_kl_src, \
    "the KL's two guards must both be the exact-identity kind"
assert 'backward' not in _tg_kl_src and 'loss' not in _tg_kl_src, \
    "a diagnostic never enters a loss (THE GOODHART FENCE)"

# -- housekeeping the head's own laws impose
assert 'dtypes.float32' not in s, "no float32 dtype literal anywhere"
assert s.count('_TC_SHOWN = True') == 1 and s.count('[tokcook] doors:') == 1, \
    "the doors print exactly once, from one place"
assert '1e-6' not in s[s.index('def _tok_cook_arm('):s.index('def _make_bank(')], \
    "no epsilon denominators in the cooker's organs"

# ===========================================================================
# THE SYMTABLE FREE-VARIABLE AUDIT (the apply_tok_seal.py idiom)
# ===========================================================================
mod_tbl = symtable.symtable(s, fn, 'exec')
module_names = set(mod_tbl.get_identifiers())
DYNAMIC_OK = {'_CENSUS', '_IMP', '_SEV', '_SGC', '_BINDC', '_PCV', '_MCV',
              '_TCV'}
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
           '_make_bank', '_tok_cook_arm', '_tok_cook_v', '_tok_gate',
           '_mh_a5', '_mh_ctx')
_seen = set()
for child in mod_tbl.get_children():
    if child.get_name() in AUDITED:
        _seen.add(child.get_name())
        audit(child, child.get_name())
assert _seen == set(AUDITED), f"audit missed {sorted(set(AUDITED) - _seen)}"

# ===========================================================================
# REPORT
# ===========================================================================
print(f"[tok cook] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines, 0 deleted; the mask-head "
      f"context block was LIFTED from the source, not retyped):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
print("[tok cook] symtable free-var audit PASS "
      f"({', '.join(AUDITED)})")
print("[tok cook] NEW params: tg_a + tg_b = 2 x H_W x 128 = 131,072, born "
      "ONLY when ALG_TOK_COOK > 0 or ALG_TOK_SEAL=head (so with the doors "
      "unset the parameter SET is unchanged and every banked ckpt loads); "
      "drawn from a dedicated rng stream (seed + 9200)")
print("[tok cook] the road: on SEALED rows at loop breaths 1..K-1 the main "
      "bank's `at` AND (S3=1) ALT21 station 3's are REPLACED by the mask "
      "head's token gate softmax((mh_kv @ tg_a)(waist @ tg_b)^T/sqrt(128) "
      "+ pad*-1e4); OPEN rows bit-for-bit today; breath 0 never sealed")
print("[tok cook] doors: ALG_TOK_COOK=<share> ALG_TOK_COOK_S3=1 "
      "ALG_TG_INIT=1.0 TC_EVAL (val's OPEN push, unconditional) + the "
      "read-time meter ALG_TOK_SEAL=head (all rows), barred in do_train by "
      "the tok seal's first statement")
print("[tok cook] census: `tokgate` (the gate, token band) + `tokgate_kl` "
      "(mean KL from uniform per slot, one scalar per row), behind the "
      "_CENSUS hook, computed by the organ — never supervised")
print("[tok cook] three cookers, three hashes: _PCV (i*2654435761), _MCV "
      "(i*2246822519 + 2654435761), _TCV (i*3266489917 + 374761393)")
if CHECK:
    print("[tok cook] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[tok cook] APPLIED ({fn}); ast OK — run the eq pre/post A/B/C "
          "gate (all cooker doors unset) + scripts/tok_cook_smoke.py "
          "before trusting")
