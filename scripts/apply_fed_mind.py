"""apply_fed_mind.py — THE FED MIND, staged patch (2026-09-05, word
given; ledger: "WORD GIVEN: THE FED MIND"). The registered ~2.5x
capacity package + the three-rotor integration + the two riders, ONE
env family: ALG_FED=1 turns everything on; per-item sub-envs
(ALG_FED_MIXER / POINTERS / WAIST / FFN / MACRO / SCRATCH / ROTOR /
NL0 / SHELF, each default 1 inside the family) exist for ablation arms
later; family unset = byte-identical (every new tensor and every new
line lives behind the family guard).

THE TEN ITEMS (7-10 per Bryce's addenda):
 1. MIXER MULTI-HEAD — the slot mixer's single 512-wide QK^T gains a
    TWIN multi-head path (MX_HEADS, default 8, of 64d) built by
    RESHAPING THE SAME W_bq/W_bk/W_bv (free reinterpretation:
    warm-load keys unchanged), per-head softmax under the same
    bias/close stack, head outputs gated by ZERO-INIT per-head gains
    (fed_mx_hg) and spoken through the TRAINED W_bo. DESIGN RULING
    (the "design this carefully" clause): a score-level head-combiner
    cannot deliver per-head attention DISTRIBUTIONS (one softmax is
    one geometry), so the sanctioned alt21 twin-kernel form is chosen
    ON PURPOSE, not as a fallback — old path untouched, new path
    behind a zero door. At zero gains the twin term is exact zeros
    (0 * finite = 0; zeros @ W_bo = zeros), so birth == the old
    single-head path bitwise. Gains' grads are ALIVE at birth
    (dL/dg_h = <dL/dh_slot @ W_bo^T, head_h out> != 0); W_bq/bk/bv
    keep their old-path gradient throughout. (Station-4's own mixer
    stays single-head — audit-noted, deferred.)
 2. POINTER MULTI-FORM — beside each single bilinear (args/res/query),
    PF_FORMS (default 3) additional bilinears summed through a
    ZERO-INIT per-form gain vector. GRAD-ALIVENESS (verified, the
    mask-head precedent's reasoning): at g_f = 0 the form weights get
    zero-but-DEFINED grads (dL/dW_f = g_f * (..) = 0, never None —
    the two-terminal law) while the gain's own grad
    dL/dg_f = <upstream, form_f> is generically nonzero, so one
    optimizer step opens the gate and the forms wake — no deadlock.
 3. WAIST RESIDUAL LAYER — waist += MLP(waist) (512->512->512, gelu)
    with a ZERO-INIT output projection (the ResNet/V11 law): exact
    zeros at birth, live grads on the zero door itself.
 4. FFN 4x — 512->2048->512. Built WITHOUT moving the base rng stream
    (the fed stream _rngF extends the 2x tensors by concatenation):
    w1 gains 1024 fresh random columns, b1 zeros, w2 gains 1024 ZERO
    rows (the zero door: new hidden units speak through zeros ->
    birth-identical in exact arithmetic). WARM: the existing loader's
    PAD-WARM branch (prefix-shape copy, do_train ~L2543) confirmed to
    handle (512,1024)->(512,2048), (1024,)->(2048,), and
    (1024,512)->(2048,512); the trained slab lands on the prefix and
    the zero/fresh extension is preserved.
 5. MACRO-VALUE MULTI-FORM — the same treatment for the macro value
    system (audit's unflagged organ): W_y (bilinear forms, as item 2)
    and h_dig2 (linear forms, s @ W_f through zero gains).
 6. SCRATCH SLOTS — +8 appended factor-bank rows (L_TOT = 32 under
    ALG_FED_SCRATCH=1; slot-scaling doctrine: var/factor slots stay
    24). fq extends by 8 fresh rows (pad-warm loads the trained 24).
    Scratch rows are attention citizens: they read tokens and all
    slots every breath (bank, mixer, stations, mask head, notebook
    lanes, garage per-slot lanes — they ride cur everywhere).
    MASK RULING (documented choice): scratch ROWS are fully open
    (never masked, both mixers + mask head); scratch COLUMNS are
    CLOSED in the base masks and open ONLY inside the fed mixer's
    twin path (_sm_tw) — the READ-BACK channel rides the zero-init
    head gains. This is the raising law applied to slots ("no cold
    births"): fully-open columns would inject 8 cold random states
    into a converged circuit at step 0; behind the door, birth is
    bitwise-identical and the circuit opens the channel at its own
    pace. Hence the module-level assert: SCRATCH requires MIXER.
    Emission/loss/gold/decode all stay on the first 24 rows via
    _fed_core() (site enumeration in the patch list + report);
    Goodhart fence: scratch content is never supervised.
    RINGS + scratch is refused loudly (the pawl grades slots).
 7. ROTOR INTEGRATION (addendum 1) — (a) the BREATH ROTOR installs
    into the new mixer heads: 60deg/breath sextet rotation
    (mycelium/rotor_clock.breath_qk_angles, legacy band pairs 24..31
    of each 64d head) applied Q-SIDE ONLY (the v109pi precedent:
    rotating Q and K by one table cancels — relative phase is the
    signal), inside the zero-gated twin path, so birth equivalence is
    free. kb maps to tick kb-1; breath-0 is outside time (phase_of's
    contract — the loop starts at kb=1); kb>6 stays unclocked.
    ** rotor_clock.py GETS ITS FIRST IMPORTER HERE — the
    sync-by-import-graph claim becomes true (top-level, unconditional
    import: numpy-only module, no circularity, negligible cost). **
    (b) pointer forms are breath-AGNOSTIC (computed once on _s_final
    in _heads_of — no kb in scope): per-breath phase conditioning is
    honestly SKIPPED there, per the addendum's own out.
    (c) bus-rotor compat: the new forms parallel W_args/W_res/W_query/
    W_y only; they touch neither the bindbus codebook/thetas nor the
    garage snap path — no bypass, no double-binding (report states).
    (d) scratch rows inherit per-breath conditioning via q_extra
    (breath_emb + notebook + garage + sixwave all broadcast over
    L_TOT rows) — confirmed, nothing to wire. Zero new params
    (FREQUENCIES FROZEN, GAINS LEARNABLE — the gains are fed_mx_hg).
 8. BREATH-0 INVARIANT FEED (addendum 2) — nl0 (the nl-tap's pooled
    breath-0 NL state, the two-tap law's invariant read) enters the
    MASK HEAD's context through a separate ZERO-INIT projection
    (fed_nl0_w) added to _mh_ce (the 22-feature encoder's input dim
    is baked; the additive path is the sanctioned alternative).
    DETACHED at entry (the mask head's metadata contract). Inert when
    ALG_MASKHEAD is off (param not built). REQUIRES the nl-tap patch
    applied first — asserted loudly below (queue order guarantees it
    at chain time).
 9. NOTEBOOK SHELF 16 (addendum 2) — the audit's storage flag. The
    coupling IS structural (ink entries are indexed by breath id:
    _nb[j] = breath j's ink, stamps _nb_st[:len(_nb)]), so the lawful
    expansion is 2 ROWS PER BREATH: a SECOND INK LANE (fed_sil2)
    stamped on rows 8..NB_ROWS-1 of the NB_ROWS=16 stamp alphabet
    (same construction, verified sharp: 16-row max off-diag |cos| =
    0.0000 < 0.35; rows 0..7 bitwise-identical to the old table, so
    lane 1 is untouched). Lane-2 read enters q_extra through a
    ZERO-INIT gain (fed_nb_g) — birth bit-identical; W_sil2 wakes
    through the gain (item-2 reasoning). Blurred-ink sites also gain
    _fed_core (scratch rows stay out of the blurred mean).
10. POST-TWIN ATLAS RE-MINE — chain-only (no patch): the paired miner
    re-runs against the FED winner into .cache/step_atlas_fed.npz +
    .cache/RESEARCH_MANIFEST_FED.json (never overwriting the port242
    atlas — vintage hygiene / never-mix-coordinates), then
    atlas_collapse_check reads it. See .cache/fed_mind_chain.sh.

ORDERING CONTRACT: this patch layers on the APPLIED mask-head +
alt21 + mass-thread code AND the paired-atlas queue's nl-tap text —
apply_nl_tap.py must have run first (asserted below). RUN ONLY AFTER
mh-round2b.service and paired-atlas.service are inactive (the
running-chain law — live units have this module imported).

RNG DISCIPLINE: every fed tensor draws from a SEPARATE stream
(_rngF = RandomState(seed + 9000)); the FFN/fq growth extends by
concatenation. The base stream is never moved, so an ALG_FED=1 build
shares every base tensor bitwise with the ALG_FED=0 build of the same
seed (never-mix-coordinates, applied at birth; also what makes the
CPU birth-equivalence smoke a real gate).

JIT discipline: no dtypes.float32 literals; every new score tensor
carries clip(-1e4, 1e4); metadata enters detached; every fed param is
in-graph whenever its envs are on (defined, possibly zero, grads —
the None-grad law).

--check: loads the file, asserts every anchor, ast-parses the
would-be result, runs the symtable free-variable audit, prints the
per-item new-param table, writes NOTHING. FED_TARGET env may point at
a copy (rehearsal); default scripts/phase1_algebra_head.py.
"""
import ast
import builtins
import math
import os
import symtable
import sys

fn = os.environ.get("FED_TARGET", 'scripts/phase1_algebra_head.py')
CHECK = '--check' in sys.argv
s = open(fn).read()
n_lines0 = s.count('\n')

assert 'ALG_FED' not in s and 'fed_mx_hg' not in s, \
    "fed mind already present — patch was applied; refuse (idempotence)"
assert '"nl0"' in s and 'nl_all' in s, \
    ("apply_nl_tap.py must be applied FIRST (the fed mind layers on the "
     "paired-atlas tap; the queue order guarantees this at chain time — "
     "run the paired-atlas chain, then this)")
assert 'mh_wo' in s and 'alt21_W_bo' in s and 'mh_atlas_traj' in s, \
    "mask-head/alt21/mass-thread code missing — wrong vintage of the head"

PATCHES = []


def patch(num, desc, old, new):
    PATCHES.append((num, desc, old, new))


# 1. module consts: the family dial, sub-dials, dims, the rotor import
#    (rotor_clock's FIRST importer — sync by import graph, finally true)
#    and the frozen rotation tables.
patch(1, "module consts: ALG_FED family + dials + rotor_clock import",
      '''MH_CTX_F = 22   # mask-context features: 12 fact (arg1/arg2/res x 4) +
                # 3 domain-mass port + 1 given-flag + 2 adjacency
                # row/col mass + 2 prev-breath row/col + 2 breath phase
SENT_MAX = 32''',
      '''MH_CTX_F = 22   # mask-context features: 12 fact (arg1/arg2/res x 4) +
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
assert H_W % MX_HEADS == 0, \\
    f"MX_HEADS={MX_HEADS} must divide H_W={H_W} (head reshape)"
PF_FORMS = int(os.environ.get("PF_FORMS", "3"))   # pointer/macro forms
N_SCR = 8 if FED_SCRATCH else 0                   # scratch slot rows
L_TOT = L_FAC + N_SCR                             # bank rows incl. scratch
NB_ROWS = 16 if FED_SHELF else 8                  # shelf stamp rows (item 9)
if FED_SCRATCH:
    # read-back for scratch rows lives ONLY behind the fed mixer's
    # zero-init door (the raising law: no cold births — fully-open
    # columns would inject 8 cold states into a converged circuit)
    assert FED_MIXER, \\
        "ALG_FED_SCRATCH requires ALG_FED_MIXER (scratch read-back door)"
    assert not int(os.environ.get("ALG_RINGS", "0")), \\
        "scratch + RINGS unsupported (the pawl grades slots; loud door)"
# FED item 7: THE BREATH ROTOR — mycelium/rotor_clock.py gets its FIRST
# importer here (the clock-audit debt: sync enforced by the import
# graph, finally true). Top-level and unconditional: numpy-only module,
# no circularity, negligible cost. FREQUENCIES FROZEN, GAINS LEARNABLE
# (the gains are fed_mx_hg) — zero parameters live here.
from mycelium.rotor_clock import breath_qk_angles as _rc_breath_qk_angles
_FED_ROT_C = _FED_ROT_S = None
if FED_ROTOR:
    assert (H_W // MX_HEADS) % 2 == 0 and (H_W // MX_HEADS) // 2 >= 32, \\
        "fed rotor band table needs >=32 pairs per head (64-d heads)"
    _fed_ang = _rc_breath_qk_angles()      # (6, 8): legacy band 24..31
    _FED_ROT_C = np.ones((_fed_ang.shape[0], (H_W // MX_HEADS) // 2),
                         np.float32)
    _FED_ROT_S = np.zeros_like(_FED_ROT_C)
    _FED_ROT_C[:, 24:32] = np.cos(_fed_ang).astype(np.float32)
    _FED_ROT_S[:, 24:32] = np.sin(_fed_ang).astype(np.float32)
SENT_MAX = 32''')

# 2. module helpers: the scratch grading fence + the multi-form summer,
#    ahead of _STEP_TAP so every consumer sees them.
patch(2, "module helpers: _fed_core (scratch fence) + _fed_pf (forms)",
      '''_STEP_TAP = None    # the step trainer's stage-0 seam (the _CENSUS/_IMP''',
      '''def _fed_core(x):
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
        form = ((s @ W[f]) @ vst.transpose(-2, -1)) if vst is not None \\
            else (s @ W[f])
        term = g[f] * form
        extra = term if extra is None else extra + term
    return base + extra


_STEP_TAP = None    # the step trainer's stage-0 seam (the _CENSUS/_IMP''')

# 3. _heads_of: the scratch grading fence at the door — every emission
#    (and therefore every loss, decode, eval, slot-mask build and gold
#    index) sees exactly the first L_FAC rows.
patch(3, "_heads_of: scratch fence on s (one cut covers all emissions)",
      '''    final state with the seam-current vst. Single source of truth."""
    return {''',
      '''    final state with the seam-current vst. Single source of truth."""
    s = _fed_core(s)   # FED scratch: grade only the true factor rows
    return {''')

# 4-6. _heads_of: multi-form args/res (+ macro y/dig2 in their guard).
patch(4, "_heads_of: args multi-form (item 2)",
      '''        "args": (s @ p["W_args"]) @ vst.transpose(-2, -1),''',
      '''        "args": _fed_pf(p, "args", s, vst,
                        (s @ p["W_args"]) @ vst.transpose(-2, -1)),''')
patch(5, "_heads_of: res multi-form (item 2)",
      '''        "res": (s @ p["W_res"]) @ vst.transpose(-2, -1),''',
      '''        "res": _fed_pf(p, "res", s, vst,
                       (s @ p["W_res"]) @ vst.transpose(-2, -1)),''')
patch(6, "_heads_of: macro y + dig2 multi-form (item 5)",
      '''        **({"dig2": (s @ p["h_dig2"] + p["h_dig2_b"])
            .reshape(B, L_FAC, N_DIG, 10),
            "y": (s @ p["W_y"]) @ vst.transpose(-2, -1)}
           if "h_dig2" in p else {}),''',
      '''        **({"dig2": _fed_pf(p, "dig2", s, None,
                            s @ p["h_dig2"] + p["h_dig2_b"])
            .reshape(B, L_FAC, N_DIG, 10),
            "y": _fed_pf(p, "y", s, vst,
                         (s @ p["W_y"]) @ vst.transpose(-2, -1))}
           if "h_dig2" in p else {}),''')

# 7. build_params: the fed rng stream (base stream never moves — an
#    ALG_FED=1 build shares every base tensor bitwise with the
#    ALG_FED=0 build of the same seed).
patch(7, "build_params: _rngF fed stream (coordinate law at birth)",
      '''    p = {}
    p["waist_w"], p["waist_b"] = lin(H_TRUNK, H_W)''',
      '''    p = {}
    _rngF = np.random.RandomState(seed + 9000)   # FED stream: fed
                                                 # tensors never move
                                                 # the base rng stream
    p["waist_w"], p["waist_b"] = lin(H_TRUNK, H_W)''')

# 8. build_params: FFN 4x by concatenation (item 4). w2's new rows are
#    the ZERO DOOR; warm-load pad-warms the trained 2x slab onto the
#    prefix and preserves the extension.
patch(8, "build_params: FFN 2x->4x extension (item 4, zero-door rows)",
      '''    p["ffn_w1"], p["ffn_b1"] = lin(H_W, 2 * H_W)
    p["ffn_w2"], p["ffn_b2"] = lin(2 * H_W, H_W)''',
      '''    p["ffn_w1"], p["ffn_b1"] = lin(H_W, 2 * H_W)
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
        p["ffn_w2"] = t(_fw2)''')

# 9. build_params: mixer gains + nl0 door inside the K_B > 1 block
#    (the mixer and mask head live only there).
patch(9, "build_params: fed_mx_hg (item 1) + fed_nl0_w (item 8)",
      '''            p["mh_gain"] = t(np.full(1, 0.02))      # AJAR (the law)
        pass''',
      '''            p["mh_gain"] = t(np.full(1, 0.02))      # AJAR (the law)
        if FED_MIXER:
            # FED item 1: per-head ZERO-INIT gains — the twin path's
            # single door (heads reshape the trained W_bq/W_bk/W_bv;
            # output speaks through the trained W_bo)
            p["fed_mx_hg"] = t(np.zeros(MX_HEADS))
        if FED_NL0 and int(os.environ.get("ALG_MASKHEAD", "0")):
            # FED item 8: the breath-0 invariant feed's ZERO door into
            # the mask-head context (inert when the organ is off)
            p["fed_nl0_w"] = t(np.zeros((H_W, H_W)))
        pass''')

# 10. build_params: shelf lane-2 ink + gain (item 9), inside the
#     notebook block.
patch(10, "build_params: fed_sil2 + fed_nb_g (item 9, shelf lane 2)",
      '''        else:                   # same coordinate code
            p["W_sil"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
            p["W_nq"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))''',
      '''        else:                   # same coordinate code
            p["W_sil"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
            p["W_nq"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
        if FED_SHELF:
            # FED item 9: the second ink lane (2 rows/breath — the
            # lawful shelf-16; stamps rows 8..15). Read rides the
            # ZERO-INIT gain; fed_sil2 wakes through it (item-2 law).
            p["fed_sil2"] = t(_rngF.randn(H_W, H_W).astype(np.float32)
                              / math.sqrt(H_W))
            p["fed_nb_g"] = t(np.zeros(1))''')

# 11. build_params: pointer forms, macro forms, waist residual, scratch
#     rows — anchored on the pointer tail.
patch(11, "build_params: pf forms + waist MLP + scratch fq rows (2/3/5/6)",
      '''    p["W_res"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
    p["W_query"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))''',
      '''    p["W_res"] = t(rng.randn(H_W, H_W) / math.sqrt(H_W))
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
             (_rngF.randn(N_SCR, H_W) * 0.02).astype(np.float32)], 0))''')

# 12. NB_STAMPS: row count env-driven (rows 0..7 bitwise-unchanged; the
#     16-row table verified sharp: max off-diag |cos| = 0.0000 < 0.35).
patch(12, "NB_STAMPS: NB_ROWS-driven stamp table (item 9)",
      '''    _ks = np.arange(8)
    _ds = np.arange(512)
    NB_STAMPS = np.cos(_ks[:, None] * np.pi / 8.0 * 7
                       + _ds[None, :] * (2 * np.pi / 512)
                       * (_ks[:, None] + 1)).astype(np.float32)
    NB_STAMPS /= np.linalg.norm(NB_STAMPS, axis=1, keepdims=True)
    _cc = np.abs(NB_STAMPS @ NB_STAMPS.T - np.eye(8)).max()''',
      '''    _ks = np.arange(NB_ROWS)   # FED item 9: 16 rows under the
    _ds = np.arange(512)           # family; rows 0..7 are bitwise the
    NB_STAMPS = np.cos(_ks[:, None] * np.pi / 8.0 * 7   # legacy table
                       + _ds[None, :] * (2 * np.pi / 512)
                       * (_ks[:, None] + 1)).astype(np.float32)
    NB_STAMPS /= np.linalg.norm(NB_STAMPS, axis=1, keepdims=True)
    _cc = np.abs(NB_STAMPS @ NB_STAMPS.T - np.eye(NB_ROWS)).max()''')

# 13. forward: waist residual layer (item 3) — before the bank closure
#     captures waist.
patch(13, "forward: waist residual through the zero door (item 3)",
      '''    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]''',
      '''    waist = (trunk @ p["waist_w"] + p["waist_b"]).gelu() + p["sent_emb"][sent]
    if FED_WAIST and "fed_w2b" in p:
        # FED item 3: waist2 = waist + MLP(waist), output ZERO-INIT —
        # exact zeros at birth; every downstream organ (bank closure,
        # breath ctx, step-trainer tap) inherits the rebound name
        waist = waist + ((waist @ p["fed_w2a"] + p["fed_w2a_b"]).gelu()
                         @ p["fed_w2b"] + p["fed_w2b_b"])''')

# 14. forward: the scratch mask geometry (item 6) — scratch ROWS fully
#     open, scratch COLUMNS closed in the base masks (read-back rides
#     the fed mixer's door only). Base 24x24 support untouched bitwise.
patch(14, "forward: slot_mask 24->32 pad (rows open, columns closed)",
      '''    bank = _make_bank(p, waist, tokmask, B)''',
      '''    bank = _make_bank(p, waist, tokmask, B)
    if N_SCR and slot_mask is not None:
        # FED item 6 mask ruling: scratch rows (queries) OPEN to all;
        # scratch columns CLOSED here (no cold read-back at birth —
        # the raising law; the fed mixer's zero door is the channel)
        _f6c = slot_mask[:, :, :1] * 0.0            # (B, L_FAC, 1) zeros
        _f6top = Tensor.cat(slot_mask,
                            *([_f6c] * N_SCR), dim=2)  # (B, L_FAC, L_TOT)
        _f6r = _f6top[:, :1, :] * 0.0 + 1.0         # (B, 1, L_TOT) ones
        slot_mask = Tensor.cat(_f6top,
                               *([_f6r] * N_SCR), dim=1)  # (B, L_TOT, L_TOT)''')

# 15-19. forward: sync/sixwave phase tables extend to L_TOT (scratch
#     slots are phase citizens; unset = same value, same bits).
patch(15, "forward: sync phase table over L_TOT",
      '''        _A = float(os.environ.get("SYNC_A", "1.0"))
        _phi = np.pi / 3.0 * (np.arange(L_FAC) % 6)''',
      '''        _A = float(os.environ.get("SYNC_A", "1.0"))
        _phi = np.pi / 3.0 * (np.arange(L_TOT) % 6)''')
patch(16, "forward: sync _mk_pb reshapes over L_TOT",
      '''            return (_cph.reshape(1, 1, L_FAC, 1) * cthk.reshape(B, 1, 1, -1)
                    + _sph.reshape(1, 1, L_FAC, 1)
                    * sthk.reshape(B, 1, 1, -1)) * _A''',
      '''            return (_cph.reshape(1, 1, L_TOT, 1) * cthk.reshape(B, 1, 1, -1)
                    + _sph.reshape(1, 1, L_TOT, 1)
                    * sthk.reshape(B, 1, 1, -1)) * _A''')
patch(17, "forward: sync oscillator table over L_TOT",
      '''        _php = np.zeros((L_FAC, H_W), np.float32)''',
      '''        _php = np.zeros((L_TOT, H_W), np.float32)''')
patch(18, "forward: sync oscillator reshape over L_TOT",
      '''            return Tensor(_o, dtype=dtypes.float).reshape(1, L_FAC, H_W)''',
      '''            return Tensor(_o, dtype=dtypes.float).reshape(1, L_TOT, H_W)''')
patch(19, "forward: sixwave phase table + term over L_TOT",
      '''        _phi = np.pi / 3.0 * (np.arange(L_FAC) % 6)
        _cph = Tensor(np.cos(_phi).astype(np.float32), dtype=dtypes.float)
        _sph = Tensor(np.sin(_phi).astype(np.float32), dtype=dtypes.float)
        _sw_term = (_cph.reshape(1, 1, L_FAC, 1) * _th.cos().reshape(B, 1, 1, -1)
               + _sph.reshape(1, 1, L_FAC, 1)
               * _th.sin().reshape(B, 1, 1, -1)) * p["sw_g"].reshape(1, 1, 1, 1)''',
      '''        _phi = np.pi / 3.0 * (np.arange(L_TOT) % 6)
        _cph = Tensor(np.cos(_phi).astype(np.float32), dtype=dtypes.float)
        _sph = Tensor(np.sin(_phi).astype(np.float32), dtype=dtypes.float)
        _sw_term = (_cph.reshape(1, 1, L_TOT, 1) * _th.cos().reshape(B, 1, 1, -1)
               + _sph.reshape(1, 1, L_TOT, 1)
               * _th.sin().reshape(B, 1, 1, -1)) * p["sw_g"].reshape(1, 1, 1, 1)''')

# 20. forward: the factor bank call widens to L_TOT.
patch(20, "forward: fst bank call over L_TOT (item 6)",
      '''    fst, fat = bank(p["fq"], L_FAC, pbias=_pb)''',
      '''    fst, fat = bank(p["fq"], L_TOT, pbias=_pb)''')

# 21. forward: nl0 computed once (detached) + ctx plumbing (item 8).
patch(21, "forward: fed_nl0 compute before the breath ctx (item 8)",
      '''        _bs_ctx = {"B": B, "K_B": K_B, "waist": waist, "tokmask": tokmask,''',
      '''        _fed_nl0 = None
        if FED_NL0 and int(os.environ.get("ALG_MASKHEAD", "0")) \\
                and "fed_nl0_w" in p:
            # FED item 8: the breath-0 invariant NL page (two-tap law)
            # — the nl-tap's pooled read, computed ONCE (the fq bank
            # pass has no cur/fact/mask reach), DETACHED at entry (the
            # mask head's metadata contract)
            _fed_nl0 = (_fed_core(fat).mean(1).unsqueeze(1)
                        @ waist).squeeze(1).detach()
        _bs_ctx = {"B": B, "K_B": K_B, "waist": waist, "tokmask": tokmask,''')
patch(22, "forward: fed_nl0 rides the breath ctx (item 8 plumbing)",
      '''                   "fact_buf": fact_buf, "mh_mass": mh_mass,''',
      '''                   "fact_buf": fact_buf, "mh_mass": mh_mass,
                   "fed_nl0": _fed_nl0,''')
patch(23, "forward: nb2 plumbing slot in _bs_state (item 9)",
      '''                     "mh_prev": None,''',
      '''                     "mh_prev": None,
                     # FED item 9: shelf lane-2 ink (born at kb == 1)
                     "nb2": None,''')

# 24. breath_step: the breath bank call widens to L_TOT (the nl-tap
#     block after it is untouched; its fat_cur wrap is patch 25).
patch(24, "breath_step: bank call over L_TOT (item 6)",
      '''    h_tok, fat_cur = bank(p["fq"], L_FAC, extra=q_extra,''',
      '''    h_tok, fat_cur = bank(p["fq"], L_TOT, extra=q_extra,''')

# 25-27. nl-tap pages stay on the graded 24 rows (scratch reading never
#     enters the atlas charts — chart vintage stays slot-lawful).
patch(25, "nl-tap (breath): page pooled over the true factor rows",
      '''        _nlw = fat_cur.mean(1)                       # (B, T) read''',
      '''        _nlw = _fed_core(fat_cur).mean(1)            # (B, T) read''')
patch(26, "nl-tap (forward): breath-0 page over the true factor rows",
      '''        _nl0w = fat.mean(1)''',
      '''        _nl0w = _fed_core(fat).mean(1)''')
patch(27, "nl-tap (forward): nl0 over the true factor rows",
      '''        out["nl0"] = (fat.mean(1).unsqueeze(1) @ waist).squeeze(1)''',
      '''        out["nl0"] = (_fed_core(fat).mean(1)
                      .unsqueeze(1) @ waist).squeeze(1)''')
patch(28, "forward: breaths_all mined on the graded rows (atlas lawful)",
      '''        out["breaths_all"] = out_breaths''',
      '''        out["breaths_all"] = [_fed_core(_b9) for _b9 in out_breaths]''')

# 29-31. breath_step notebook: lane-2 ink (item 9) + blurred means
#     fenced from scratch rows.
patch(29, "breath_step: notebook birth — lane-2 ink born beside lane 1",
      '''    if ALG_NOTEBOOK and kb == 1:
        from tinygrad import Tensor as _T2, dtypes as _dt2
        _nb_st = _T2(NB_STAMPS, dtype=_dt2.float)
        _nb = [(cur @ p["W_sil"]) if NB_PERSLOT
               else (cur.mean(1) @ p["W_sil"])]   # sharp vs blurred ink''',
      '''    if ALG_NOTEBOOK and kb == 1:
        from tinygrad import Tensor as _T2, dtypes as _dt2
        _nb_st = _T2(NB_STAMPS, dtype=_dt2.float)
        _nb = [(cur @ p["W_sil"]) if NB_PERSLOT
               else (_fed_core(cur).mean(1) @ p["W_sil"])]   # sharp vs blurred
        if FED_SHELF and "fed_sil2" in p:
            # FED item 9: lane 2 born at the same breath (2 rows per
            # breath — the structural coupling's lawful expansion)
            state["nb2"] = [(cur @ p["fed_sil2"]) if NB_PERSLOT
                            else (_fed_core(cur).mean(1) @ p["fed_sil2"])]''')
patch(30, "breath_step: notebook blurred read fenced + lane-2 read",
      '''            _q = cur.mean(1) @ p["W_nq"]
            _sc = (_q @ _nb_st[:len(_nb)].transpose(1, 0)) / math.sqrt(H_W)
            if NB_FOCAL > 0:
                _sc = _sc * NB_FOCAL          # the magnifying glass
            _at = _sc.softmax(-1)
            _rd = sum(_at[:, j:j + 1] * _nb[j] for j in range(len(_nb)))
            q_extra = q_extra + _rd.reshape(B, 1, -1)''',
      '''            _q = _fed_core(cur).mean(1) @ p["W_nq"]
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
            _sc2r = (_q @ _nb_st[8:8 + len(_nb2)].transpose(1, 0)) \\
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
                q_extra = q_extra + _rd2.reshape(B, 1, -1) \\
                    * p["fed_nb_g"].reshape(1, 1, 1)''')
patch(31, "breath_step: notebook append — lane-2 ink written per breath",
      '''    if ALG_NOTEBOOK:
        _nb.append((cur @ p["W_sil"]) if NB_PERSLOT
                   else (cur.mean(1) @ p["W_sil"]))''',
      '''    if ALG_NOTEBOOK:
        _nb.append((cur @ p["W_sil"]) if NB_PERSLOT
                   else (_fed_core(cur).mean(1) @ p["W_sil"]))
        if FED_SHELF and "fed_sil2" in p and state.get("nb2") is not None:
            state["nb2"].append((cur @ p["fed_sil2"]) if NB_PERSLOT
                                else (_fed_core(cur).mean(1)
                                      @ p["fed_sil2"]))''')

# 32-35. breath_step mask head: L_TOT reshapes + the nl0 door (item 8).
patch(32, "mask head: q reshape over L_TOT",
      '''        _mh_qh = _mh_q.reshape(B, L_FAC, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)''',
      '''        _mh_qh = _mh_q.reshape(B, L_TOT, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)''')
patch(33, "mask head: k/v reshapes over L_TOT",
      '''        _mh_kh = _mh_k.reshape(B, L_FAC, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)
        _mh_vh = _mh_v.reshape(B, L_FAC, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)''',
      '''        _mh_kh = _mh_k.reshape(B, L_TOT, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)
        _mh_vh = _mh_v.reshape(B, L_TOT, MH_HEADS, _mh_hd).permute(0, 2, 1, 3)''')
patch(34, "mask head: gather reshape over L_TOT",
      '''        _mh_gt = (_mh_at @ _mh_vh).permute(0, 2, 1, 3) \\
            .reshape(B, L_FAC, H_W)''',
      '''        _mh_gt = (_mh_at @ _mh_vh).permute(0, 2, 1, 3) \\
            .reshape(B, L_TOT, H_W)''')
patch(35, "mask head: nl0 context door (item 8, zero-init)",
      '''        _mh_ce = _mh_ce + _mh_ap.reshape(B, -1, H_W) @ p["mh_atlas_w"]''',
      '''        _mh_ce = _mh_ce + _mh_ap.reshape(B, -1, H_W) @ p["mh_atlas_w"]
        _mh_nl = ctx.get("fed_nl0")
        if _mh_nl is not None and "fed_nl0_w" in p:
            # FED item 8: the breath-0 invariant page through its ZERO
            # door — exact zeros at birth, live grads on fed_nl0_w
            _mh_ce = _mh_ce + (_mh_nl.reshape(B, 1, H_W)
                               @ p["fed_nl0_w"])''')

# 36. breath_step: THE FED MIXER (items 1 + 7a) + scratch read-back.
patch(36, "breath_step: fed mixer twin path + breath rotor (items 1/6/7)",
      '''    h_slot = (sc2.softmax(-1) @ bv) @ p["W_bo"] + p["W_bo_b"]''',
      '''    h_slot = (sc2.softmax(-1) @ bv) @ p["W_bo"] + p["W_bo_b"]
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
        if FED_ROTOR and _FED_ROT_C is not None and 1 <= kb <= 6:
            # FED item 7a: THE BREATH ROTOR INSTALLS HERE —
            # 60deg/breath sextet rotation (rotor_clock's legacy band,
            # pairs 24..31 of each 64d head), Q-SIDE ONLY (the v109pi
            # precedent: one table on both sides cancels — relative
            # phase is the signal). Behind the zero gains, so birth
            # equivalence is free. kb -> tick kb-1 (breath-0 is
            # outside time, phase_of's contract; kb > 6 unclocked).
            from tinygrad import Tensor as _T7, dtypes as _d7
            _rc7 = _T7(_FED_ROT_C[kb - 1], dtype=_d7.float) \\
                .reshape(1, 1, 1, -1)
            _rs7 = _T7(_FED_ROT_S[kb - 1], dtype=_d7.float) \\
                .reshape(1, 1, 1, -1)
            _qp7 = _mx_q.reshape(B, MX_HEADS, L_TOT, _mx_hd // 2, 2)
            _qx7, _qy7 = _qp7[..., 0], _qp7[..., 1]
            _mx_q = Tensor.stack(_qx7 * _rc7 - _qy7 * _rs7,
                                 _qx7 * _rs7 + _qy7 * _rc7, dim=-1) \\
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
        _mx_o = (_mx_sc.softmax(-1) @ _mx_v) \\
            * p["fed_mx_hg"].reshape(1, MX_HEADS, 1, 1)   # ZERO gains
        h_slot = h_slot + _mx_o.permute(0, 2, 1, 3) \\
            .reshape(B, L_TOT, H_W) @ p["W_bo"]''')

# 37-38. alt21 stations: reshapes widen to L_TOT (station-4's mixer
#     itself stays single-head — audit-noted deferral).
patch(37, "alt21 station 3: q reshape over L_TOT",
      '''        _qh21 = _q21.reshape(B, L_FAC, N_HEADS, _hd21).permute(0, 2, 1, 3)''',
      '''        _qh21 = _q21.reshape(B, L_TOT, N_HEADS, _hd21).permute(0, 2, 1, 3)''')
patch(38, "alt21 station 3: gather reshape over L_TOT",
      '''        _st21 = (_sa21.softmax(-1) @ _vh21).permute(0, 2, 1, 3).reshape(B, L_FAC, H_W)''',
      '''        _st21 = (_sa21.softmax(-1) @ _vh21).permute(0, 2, 1, 3).reshape(B, L_TOT, H_W)''')

# 39. forward: query pointer multi-form (item 2).
patch(39, "forward: query multi-form (item 2)",
      '''    out["query"] = ((qst @ p["W_query"]) @ vst.transpose(-2, -1)).reshape(B, K_VARS)''',
      '''    out["query"] = _fed_pf(
        p, "query", qst, vst,
        (qst @ p["W_query"]) @ vst.transpose(-2, -1)).reshape(B, K_VARS)''')

# 40-43. forward: remaining emission-side scratch fences.
patch(40, "forward: fat emission fenced to the graded rows",
      '''    out["fat"], out["vat"] = fat, vat''',
      '''    out["fat"], out["vat"] = _fed_core(fat), vat''')
patch(41, "forward: depth/term heads fenced to the graded rows",
      '''        out["depth"] = fst @ p["h_depth"] + p["h_depth_b"]
        out["term"] = (fst @ p["h_term"] + p["h_term_b"]).squeeze(-1)''',
      '''        out["depth"] = _fed_core(fst) @ p["h_depth"] + p["h_depth_b"]
        out["term"] = (_fed_core(fst) @ p["h_term"]
                       + p["h_term_b"]).squeeze(-1)''')
patch(42, "forward: bind emission fenced to the graded rows",
      '''        out["bind"] = (_bsrc @ p["W_bind1"] + p["W_bind1_b"]).gelu() @ p["W_bind2"]''',
      '''        out["bind"] = (_fed_core(_bsrc) @ p["W_bind1"]
                       + p["W_bind1_b"]).gelu() @ p["W_bind2"]''')
patch(43, "forward: opc slot pool fenced (extensive readout law kept)",
      '''            _pool = fst.mean(1)''',
      '''            _pool = _fed_core(fst).mean(1)''')


for num, desc, old, new in PATCHES:
    assert old in s, f"anchor {num} MISSING ({desc}) — read the file, adjust"
    assert s.count(old) == 1, f"anchor {num} NOT UNIQUE ({desc})"
    s = s.replace(old, new, 1)

tree = ast.parse(s)                       # the would-be result must parse

# structural asserts on the would-be module (cheap, no import, no GPU)
for key in ("fed_mx_hg", "fed_pf_y_W", "fed_pf_y_g", "fed_pf_dig2_W",
            "fed_pf_dig2_g", "fed_w2a", "fed_w2b", "fed_nl0_w",
            "fed_sil2", "fed_nb_g"):
    assert f'p["{key}"]' in s, f"patched tree lost {key}"
assert 'p["fed_pf_" + _pfn + "_W"]' in s \
    and 'p["fed_pf_" + _pfn + "_g"]' in s, \
    "patched tree lost the pointer form loop (args/res/query)"
assert s.count('def _fed_core') == 1 and s.count('def _fed_pf') == 1
assert s.count('from mycelium.rotor_clock import') == 1, \
    "rotor_clock must be imported exactly once (the first importer)"
assert s.index('ALG_FED = int(') < s.index('def _fed_core'), \
    "family dial must precede the helpers"
assert s.index('h_slot = (sc2.softmax(-1) @ bv)') \
    < s.index('if FED_MIXER and "fed_mx_hg" in p:') \
    < s.index('g = p["breath_gate"][kb].sigmoid()'), \
    "fed mixer must land between the old path and the gate"
assert s.index('_sm_tw = _sm_kb') > s.index('sc2 = sc2.clip(-1e4, 1e4)'), \
    "scratch read-back must live in the twin path, after the base close"
for probe in ('"fed_nl0": _fed_nl0,', '"nb2": None,',
              'ctx.get("fed_nl0")', 'state.get("nb2")',
              '_nb_st[8:8 + len(_nb2)]',
              'bank(p["fq"], L_TOT, pbias=_pb)',
              'bank(p["fq"], L_TOT, extra=q_extra,'):
    assert probe in s, f"patched tree lost the {probe} wiring"
assert 'reshape(B, L_FAC, MH_HEADS' not in s, \
    "a mask-head reshape still reads L_FAC (scratch would break loudly)"
assert 'reshape(B, L_FAC, N_HEADS' not in s, \
    "an alt21 reshape still reads L_FAC (scratch would break loudly)"

# the symtable free-variable audit (the apply_mask_head.py idiom)
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
    if child.get_name() in ('breath_step', 'build_params', 'forward',
                            '_heads_of', '_fed_core', '_fed_pf'):
        audit(child, child.get_name())

# the per-item new-param table (H_W=512, N_DIG=7, PF_FORMS=3 defaults) —
# capacity is a registered claim, not a vibe
H, F, ND10 = 512, 3, 70
ITEMS = [
    ("1 mixer multi-head (fed_mx_hg)",            8),
    ("2 pointer multi-form (args/res/query)",     3 * (F * H * H + F)),
    ("3 waist residual layer",                    2 * H * H + 2 * H),
    ("4 FFN 2x->4x extension",                    H * 2 * H + 2 * H + 2 * H * H),
    ("5 macro multi-form (y + dig2)",             (F * H * H + F)
                                                  + (F * H * ND10 + F)),
    ("6 scratch slots (+8 fq rows)",              8 * H),
    ("7 rotor integration (frozen freqs)",        0),
    ("8 nl0 door (fed_nl0_w)",                    H * H),
    ("9 shelf lane 2 (fed_sil2 + gain)",          H * H + 1),
]
total = sum(n for _, n in ITEMS)
print(f"[fed mind] {len(PATCHES)} anchors OK "
      f"(+{s.count(chr(10)) - n_lines0} lines):")
for num, desc, _o, _n in PATCHES:
    print(f"  {num:2d}. {desc}")
print("[fed mind] symtable free-var audit PASS (breath_step, build_params, "
      "forward, _heads_of, helpers)")
print("[fed mind] per-item NEW params (defaults MX=8, PF_FORMS=3, WIDE):")
for desc, n in ITEMS:
    print(f"    item {desc:44s} {n:>9,}")
print(f"    {'TOTAL NEW':49s} {total:>9,}")
print("[fed mind] champion-stack projection: ~11.9M (port242 organs + "
      "mask head) + %.2fM = ~%.1fM trained — inside the 14-18M landing "
      "zone" % (total / 1e6, 11.9 + total / 1e6))
print("[fed mind] item 10 (atlas re-mine) is chain-only: "
      ".cache/fed_mind_chain.sh")
if CHECK:
    print("[fed mind] --check: ast OK on the would-be result; "
          "NOTHING written")
else:
    open(fn, 'w').write(s)
    print(f"[fed mind] APPLIED ({fn}); ast OK — run the eq pre/post "
          "A/B/C gate before trusting (equivalence contract, rung 1)")
