"""credit_census.py -- THE CREDIT CENSUS (2026-10-08; Bryce's "Aaron Judge"; zero GPU, DEV=CPU).

Question: in the training step, WHO does the gradient on the SHARED parameters come from? The
loss is a sum over slots (24 per row) and breaths (the ladder's 7 rungs, out["breaths"]); the
parameters (the bank's token->slot projections, the mixer, the polar waist, the notebook, and
every head: pres/ftype/op/args/res/dig) are shared across every slot and every breath. This script
measures, on a few real diet batches under the record body's own checkpoint and env, the SHARE of
the gradient each slot CLASS is responsible for, per parameter group -- not an aggregate loss
number, a gradient-norm partition.

Method (no chain-imported file edited; this is a new, free-standing read, the dirbit_grad_probe.py
/ grad_cosine_census.py pattern): build_params() + WARM_FROM-style load of the checkpoint, load_alg
on the real diet slice, the SAME open-pass -> build_slot_masks -> masked-pass two-step every
trainer step takes, then for EACH class a FRESH forward() call (a backward() consumes the graph --
the codebase's own convention, grad_cosine_census.py's comment) with a GOLD DICT whose presence
AND kind fields (is_rel/is_lit_f/is_mod/is_sel/is_pct/is_fdiv/is_macro/is_frac/is_chain) are zeroed
for every slot OUTSIDE the class -- "zero the other slots' loss terms" literally: every per-slot
term in _loss_single that gates on `pres` alone (ftype, res, the span/canvas terms, bind) or on a
kind-sum (dig's dm, args' am, op/caric's rel) goes to exactly zero for an excluded slot, because
presence is zeroed AND the kind flags that are not already presence-gated are zeroed too. The two
EXCEPTIONS, named loudly because they corrupt rather than silently vanish: the "pres" head's own
BCE term and the "islit" head's own BCE term are UNWEIGHTED (`.mean()` over all 24 slots, using
presence/is_lit_f as the regression TARGET, not a gate) -- zeroing presence/is_lit_f for excluded
slots LIES to these two terms ("this slot is not here"), so their class-conditioned gradient is a
genuine confound, not a clean share; isolated into their own groups below so the leak is visible
and not smeared into "rest". The "query" head's CE term is per-ROW (not per-slot) and cannot be
split by slot class at all -- it rides every class's forward pass identically, an irreducible
shared residual (noted, not fixed).

RIGHT/WRONG: loop_val.py's own `ok` criterion (pres/ftype/res/op/args or pres/ftype/res/dig),
recomputed field-by-field from this script's own two-pass forward output, under THE NUMERAL MASK
(mycelium.rulebook.legal_digit_logits, loop_val's LV_LEGAL=num path) on literal slots whose
predicted ftype agrees with gold on being non-relation -- the exact gate loop_val's own line uses.
GIVEN/RELATION: g["ftype"]==1 / g["ftype"]==0 (the 8/9-way ftype's first two classes; the minority
mod/sel/pct/fdiv/macro/frac/chain slots belong to neither and are excluded from this split, counted
separately, so GIVEN+RELATION undershoots FULL by their share). FORWARD/INVERSE: ported from
polarity_census.classify_row (verbatim import, never copied, per the module's own "training gold
and the census can never drift" discipline applied to prose), restricted to is_rel==1 slots;
classify_row's "other" bucket (the single-new-variable law not holding, ~0.2% on wild) is excluded
from both, counted separately.

Per-breath split: the ladder's own weighting (w_kb = 1 + kb/(K_B-1), summed then /K_B, loss_fn's
own formula, replicated here with no slot-class mask at all) -- one backward per rung in isolation;
these must sum to the FULL unmasked gradient near-exactly (a clean linear decomposition, no
pres/islit/query leak, the strongest sanity check this script has).

THE OOM REALITY (found running this on a live host): this body's checkpoint is big enough
(512-d, 7 breaths, FED/NOTEBOOK/SIXWAVE/POLAR/STELLAR/BINDBUS all on) that EAGER forward+backward
(no TinyJit -- needed because every class/breath target is its own one-off loss formula) pays a
one-time-per-shape kernel-compile tax of MINUTES per distinct backward graph (one per class, one
per breath-in-isolation): ~2-12 min the first time a shape+graph combo is hit, ~2 min on repeat.
Worse: a CONCURRENT GPU training chain on the same host (gpu.lock's own holder, never touched by
this script) can use the bulk of system RAM independently of anything this script does, and the
OOM killer prefers to kill a low-priority `systemd --user` transient unit over that chain -- the
SAFE outcome (a training run must never be the casualty) but it means this census must be
RESUMABLE, not monolithic. So it runs as a WORKER + AGGREGATE split: CC_TARGET (+ CC_BIDX) makes
this script compute exactly ONE (batch, class-or-breath) backward, save its grad vector + loss +
that batch's class-size census to CC_SAVE_DIR, and exit -- a driver shell loop invokes one fresh
worker PROCESS per combo (full OS memory reclaim between combos) and retries a combo whose result
file is missing, indefinitely (the failure mode is transient memory pressure from the other chain,
which clears on its own). CC_AGGREGATE=1 is the other door: no model, no diet, no safetensors load
-- read every saved combo back off disk (one pair of vectors in memory at a time) and print the
tables + the six-line reading to CC_OUT.

Env (worker mode): CC_TARGET (one of FULL/RIGHT/WRONG/GIVEN/RELATION/FORWARD/INVERSE/B0../B6),
CC_BIDX (which batch, 0-based), CC_CKPT, CC_BATCH, CC_NBATCH, CC_ROW0, CC_SAVE_DIR. Env (aggregate
mode): CC_AGGREGATE=1, CC_CKPT, CC_BATCH, CC_NBATCH, CC_ROW0, CC_SAVE_DIR, CC_OUT. The caller
supplies every model env (DEV=CPU, the family block, PMS8's SURF8/role8 block, ALG_TRAIN/
ALG_TRAIN_NAME pointed at the diet slice) exactly as a training launch would in worker mode; none
of it is needed (or read) in aggregate mode.

usage (worker): DEV=CPU <family + SURF8 envs> ALG_TRAIN=.cache/form_pm35c_slice1024_valid2.jsonl \\
       ALG_TRAIN_NAME=pm35cslicevalid2 CC_CKPT=.cache/sharp_PMS8_241.safetensors \\
       CC_TARGET=FULL CC_BIDX=0 .venv/bin/python3 scripts/credit_census.py
usage (aggregate): CC_AGGREGATE=1 CC_CKPT=.cache/sharp_PMS8_241.safetensors \\
       .venv/bin/python3 scripts/credit_census.py
"""
import collections
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice")   # dig-head on
# RELATION-only / op-head+args-head on GIVEN-only are legitimately all-zero
# gradients (those heads have no term on that slot kind at all) -- nanmean
# over an all-nan cosine/ratio array is the CORRECT answer (nan), not a bug;
# the warning is cosmetic noise in the log, silenced here, not papered over
# (the nan still prints in every table exactly where it belongs).

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import numpy as np

T0 = time.time()
CKPT = os.environ.get("CC_CKPT", ".cache/sharp_PMS8_241.safetensors")
BATCH = int(os.environ.get("CC_BATCH", "8"))
NBATCH = int(os.environ.get("CC_NBATCH", "4"))
ROW0 = int(os.environ.get("CC_ROW0", "0"))
OUT = os.environ.get("CC_OUT", ".cache/credit_census_PMS8_241.txt")
K_B_ENV = int(os.environ.get("ALG_BREATH", "7"))
BREATH_NORM = int(os.environ.get("BREATH_NORM", "0"))
CLASS_NAMES = ["FULL", "RIGHT", "WRONG", "GIVEN", "RELATION", "FORWARD", "INVERSE"]
TARGETS = CLASS_NAMES + [f"B{kb}" for kb in range(K_B_ENV)]

SAVE_DIR = os.environ.get("CC_SAVE_DIR", ".cache/credit_census_tmp")
CKPT_TAG = os.path.splitext(os.path.basename(CKPT))[0]
TARGET = os.environ.get("CC_TARGET", "")
BIDX = int(os.environ.get("CC_BIDX", "0"))
AGGREGATE = int(os.environ.get("CC_AGGREGATE", "0"))

KIND_FIELDS = ["is_rel", "is_lit_f", "is_mod", "is_sel", "is_pct", "is_fdiv",
               "is_macro", "is_frac", "is_chain", "is_ind"]

HEAD_GROUPS = {
    "pres-head": ["h_pres", "h_pres_b"],
    "islit-head": ["h_islit", "h_islit_b"],   # the other presence-TARGET leak, isolated
    "ftype-head": ["h_ftype", "h_ftype_b"],
    "op-head": ["h_op", "h_op_b"],
    "args-head": ["W_args"],
    "res-head": ["W_res"],
    "dig-head": ["h_dig", "h_dig_b"],
}
ORGAN_GROUPS = {
    "bank": ["attn_wq", "attn_wq_b", "attn_wk", "attn_wk_b", "attn_wv", "attn_wv_b"],
    "mixer": ["W_bq", "W_bq_b", "W_bk", "W_bk_b", "W_bv", "W_bv_b", "W_bo", "W_bo_b"],
    "polar-waist": ["polar_wd", "polar_wu"],
    "notebook": ["W_sil", "W_nq"],
}


def log(msg):
    print(f"[credit {time.time()-T0:7.1f}s] {msg}", flush=True)


def combo_path(bidx, target):
    return os.path.join(SAVE_DIR, f"{CKPT_TAG}_b{bidx}_{target}.npz")


def meta_path(bidx):
    return os.path.join(SAVE_DIR, f"{CKPT_TAG}_b{bidx}_meta.json")


def manifest_path():
    return os.path.join(SAVE_DIR, f"{CKPT_TAG}_manifest.json")


# ===========================================================================
# AGGREGATE MODE: no model, no diet -- read saved combos off disk and report
# ===========================================================================
def aggregate_and_report():
    man = json.load(open(manifest_path()))
    SHARED = man["shared"]
    SPANS = {k: tuple(v) for k, v in man["spans"].items()}
    TOTAL = man["total"]
    GROUPS = man["groups"]
    K_B = man["k_b"]
    log(f"manifest: {len(SHARED)} shared params, {TOTAL} scalars, K_B={K_B}, "
        f"groups: " + ", ".join(f"{g}({len(ks)})" for g, ks in GROUPS.items()))

    def group_slice(v, keys):
        if not keys:
            return np.zeros(0, np.float64)
        idx = np.concatenate([np.arange(*SPANS[k]) for k in keys])
        return v[idx]

    def group_norm(v, keys):
        return float(np.linalg.norm(group_slice(v, keys)))

    def group_cos(va, vb, keys):
        a, b = group_slice(va, keys), group_slice(vb, keys)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        return float(np.dot(a, b) / (na * nb))

    def load_vec(bidx, target):
        z = np.load(combo_path(bidx, target))
        return z["grad"].astype(np.float64), float(z["loss"]), int(z["n"])

    ROWS = collections.defaultdict(int)
    LOSSES = collections.defaultdict(list)
    missing = []
    for bidx in range(NBATCH):
        mp = meta_path(bidx)
        if not os.path.exists(mp):
            missing.append(f"meta b{bidx}")
            continue
        meta = json.load(open(mp))
        for k, v in meta.items():
            ROWS[k] += v
        for name in TARGETS:
            if not os.path.exists(combo_path(bidx, name)):
                missing.append(f"b{bidx}:{name}")
    if missing:
        log(f"NOT READY: {len(missing)} combo(s) missing: {missing[:12]}"
            + (" ..." if len(missing) > 12 else ""))
        sys.exit(1)

    for bidx in range(NBATCH):
        for name in TARGETS:
            _, lv, _ = load_vec(bidx, name)
            LOSSES[name].append(lv)

    lines = []

    def emit(s):
        print(s)
        lines.append(s)

    emit("=" * 100)
    emit(f"CREDIT CENSUS -- {CKPT} -- {NBATCH} batches x {BATCH} rows (rows {ROW0}:{ROW0 + NBATCH*BATCH}) "
         f"of {man.get('alg_train', '?')}")
    emit(f"K_B (breaths) = {K_B}  BREATH_NORM={BREATH_NORM}  shared params = {len(SHARED)} ({TOTAL} scalars)")
    emit("")
    emit("CLASS SIZES (summed over all batches, gold slots):")
    for k in ("TOTAL_PRES", "RIGHT", "WRONG", "GIVEN", "RELATION", "FORWARD", "INVERSE",
              "OTHER_REL", "OTHER_FTYPE"):
        emit(f"  {k:12s} {ROWS.get(k, 0):5d}")
    emit("")
    emit("PER-BATCH LOSS (the masked ladder loss_fn value) AND CLASS SIZE:")
    for name in CLASS_NAMES:
        losses = LOSSES[name]
        emit(f"  {name:9s} loss/batch = " + ", ".join(f"{l:.4f}" for l in losses) +
             f"   mean={np.mean(losses):.4f}")
    emit("")

    emit("GRADIENT-NORM SHARE PER PARAMETER GROUP (normalised: FULL's own group norm = 1.0; mean over "
         f"the {NBATCH} batches of ratio |grad_class|/|grad_FULL| on that group):")
    header = "  {:12s}".format("group") + "".join(f"{n:>10s}" for n in CLASS_NAMES)
    emit(header)
    GROUP_SHARE = {}
    full_vecs = {}
    for gname, keys in GROUPS.items():
        row = []
        shares = {}
        for name in CLASS_NAMES:
            ratios = []
            for bidx in range(NBATCH):
                vfull, _, _ = full_vecs.get((bidx, "FULL")) or (None, None, None)
                if vfull is None:
                    vfull, _, _ = load_vec(bidx, "FULL")
                    full_vecs[(bidx, "FULL")] = (vfull, None, None)
                vcls, _, _ = load_vec(bidx, name)
                nf = group_norm(vfull, keys)
                nc = group_norm(vcls, keys)
                ratios.append(nc / nf if nf > 1e-12 else float("nan"))
            shares[name] = float(np.nanmean(ratios))
            row.append(shares[name])
        GROUP_SHARE[gname] = shares
        emit("  {:12s}".format(gname) + "".join(f"{r:10.4f}" for r in row))
    emit("")

    emit("ADDITIVITY CHECK A (slot classes): |RIGHT+WRONG| / |FULL| and |GIVEN+RELATION| / |FULL| and "
         "|FORWARD+INVERSE| / |RELATION|, per group, mean over batches (1.0 = exact linear partition; "
         "expected SHORTFALL on pres-head/islit-head/rest from the named leaks, and on GIVEN+RELATION / "
         "FORWARD+INVERSE from the excluded OTHER_* residue):")
    emit("  {:12s} {:>14s} {:>18s} {:>16s}".format("group", "RIGHT+WRONG", "GIVEN+RELATION", "FWD+INV/REL"))
    for gname, keys in GROUPS.items():
        rws, grs, fis = [], [], []
        for bidx in range(NBATCH):
            vfull, _, _ = load_vec(bidx, "FULL")
            vrel, _, _ = load_vec(bidx, "RELATION")
            vr, _, _ = load_vec(bidx, "RIGHT"); vw, _, _ = load_vec(bidx, "WRONG")
            vg_, _, _ = load_vec(bidx, "GIVEN")
            vf, _, _ = load_vec(bidx, "FORWARD"); vi, _, _ = load_vec(bidx, "INVERSE")
            nf = group_norm(vfull, keys)
            nr = group_norm(vrel, keys)
            rws.append(group_norm(vr + vw, keys) / nf if nf > 1e-12 else float("nan"))
            grs.append(group_norm(vg_ + vrel, keys) / nf if nf > 1e-12 else float("nan"))
            fis.append(group_norm(vf + vi, keys) / nr if nr > 1e-12 else float("nan"))
        emit("  {:12s} {:14.4f} {:18.4f} {:16.4f}".format(
            gname, float(np.nanmean(rws)), float(np.nanmean(grs)), float(np.nanmean(fis))))
    emit("")

    emit("PER-BREATH GRADIENT SHARE (|grad_breath_kb| / |grad_FULL| per group, mean over batches) AND "
         "ADDITIVITY CHECK B (breath sum vs FULL -- the clean one, no slot-class leak):")
    header2 = "  {:12s}".format("group") + "".join(f"  b{kb}" for kb in range(K_B)) + "   sum/FULL"
    emit(header2)
    for gname, keys in GROUPS.items():
        shares_b = []
        sums = []
        for bidx in range(NBATCH):
            vfull, _, _ = load_vec(bidx, "FULL")
            nf = group_norm(vfull, keys)
            vsum = None
            row_shares = []
            for kb in range(K_B):
                vb, _, _ = load_vec(bidx, f"B{kb}")
                vsum = vb if vsum is None else vsum + vb
                row_shares.append(group_norm(vb, keys) / nf if nf > 1e-12 else float("nan"))
            shares_b.append(row_shares)
            sums.append(group_norm(vsum, keys) / nf if nf > 1e-12 else float("nan"))
        mean_shares = np.nanmean(np.array(shares_b, dtype=float), axis=0)
        emit("  {:12s}".format(gname) + "".join(f"{s:4.2f}" for s in mean_shares) +
             f"   {np.nanmean(sums):.4f}")
    emit("")

    emit("COSINE(RIGHT-grad, WRONG-grad) per group, per batch, on the shared organs (the tug-of-war "
         "question, per slot class instead of per breath):")
    emit("  {:12s}".format("group") + "".join(f"  batch{bidx}" for bidx in range(NBATCH)) + "   mean")
    rw_cos_cache = {}
    for gname, keys in GROUPS.items():
        cosines = []
        for bidx in range(NBATCH):
            vr, _, _ = load_vec(bidx, "RIGHT"); vw, _, _ = load_vec(bidx, "WRONG")
            c = group_cos(vr, vw, keys)
            cosines.append(c)
        rw_cos_cache[gname] = cosines
        emit("  {:12s}".format(gname) + "".join(f"{c:9.4f}" for c in cosines) +
             f"   {np.nanmean(cosines):.4f}")
    emit("")

    emit("COSINE(GIVEN-grad, RELATION-grad) and COSINE(FORWARD-grad, INVERSE-grad) per group, mean over "
         "batches:")
    emit("  {:12s} {:>14s} {:>14s}".format("group", "GIVEN/RELATION", "FWD/INV"))
    for gname, keys in GROUPS.items():
        c1s, c2s = [], []
        for bidx in range(NBATCH):
            vg_, _, _ = load_vec(bidx, "GIVEN"); vrel, _, _ = load_vec(bidx, "RELATION")
            vf, _, _ = load_vec(bidx, "FORWARD"); vi, _, _ = load_vec(bidx, "INVERSE")
            c1s.append(group_cos(vg_, vrel, keys))
            c2s.append(group_cos(vf, vi, keys))
        emit("  {:12s} {:14.4f} {:14.4f}".format(gname, float(np.nanmean(c1s)), float(np.nanmean(c2s))))
    emit("")

    RES_HEAD_FWD_SHARE = GROUP_SHARE["res-head"]["FORWARD"]
    RES_HEAD_INV_SHARE = GROUP_SHARE["res-head"]["INVERSE"]
    SHARED_ORGANS = [g for g in ("bank", "mixer", "polar-waist", "notebook") if g in GROUP_SHARE]
    WRONG_SHARE_SHARED_MEAN = float(np.mean([GROUP_SHARE[g]["WRONG"] for g in SHARED_ORGANS]))
    RIGHT_SHARE_SHARED_MEAN = float(np.mean([GROUP_SHARE[g]["RIGHT"] for g in SHARED_ORGANS]))
    INV_FRAC_DIET = ROWS.get("INVERSE", 0) / max(ROWS.get("RELATION", 1), 1)
    RW_COS_SHARED_MEAN = float(np.mean([np.nanmean(rw_cos_cache[g]) for g in SHARED_ORGANS]))

    emit("THE SIX-LINE READING:")
    emit(f"1. WHO: on the shared organs (bank/mixer/polar-waist/notebook), WRONG slots carry "
         f"{WRONG_SHARE_SHARED_MEAN:.3f}x FULL's gradient norm and RIGHT slots {RIGHT_SHARE_SHARED_MEAN:.3f}x "
         f"(class sizes: {ROWS.get('RIGHT', 0)} right / {ROWS.get('WRONG', 0)} wrong of {ROWS.get('TOTAL_PRES', 0)} "
         f"present slots) -- {'WRONG DOMINATES' if WRONG_SHARE_SHARED_MEAN > RIGHT_SHARE_SHARED_MEAN else 'RIGHT DOMINATES' if RIGHT_SHARE_SHARED_MEAN > WRONG_SHARE_SHARED_MEAN else 'ABOUT EVEN'} the shared organs' gradient on this slice.")
    emit(f"2. INVERSE AND THE RES HEAD: inverse relations are {INV_FRAC_DIET:.3f} of this slice's relation "
         f"slots (diet; wild runs ~0.27-0.36 per the polarity census); the res-head gradient share is "
         f"{RES_HEAD_FWD_SHARE:.3f}x FULL from FORWARD vs {RES_HEAD_INV_SHARE:.3f}x FULL from INVERSE -- "
         f"{'INVERSE gets LESS than its slot share of the res-head gradient' if (RES_HEAD_INV_SHARE < RES_HEAD_FWD_SHARE * INV_FRAC_DIET / max(1 - INV_FRAC_DIET, 1e-6)) else 'INVERSE gets at least its slot share'} "
         f"of the res-head's own gradient.")
    emit(f"3. TUG-OF-WAR: RIGHT vs WRONG grad cosine on the shared organs averages {RW_COS_SHARED_MEAN:+.4f} "
         f"({'a real conflict (negative)' if RW_COS_SHARED_MEAN < -0.02 else 'near-orthogonal' if abs(RW_COS_SHARED_MEAN) <= 0.02 else 'pulling the SAME way (positive)'}) -- "
         f"see the per-group cosine table for which organ disagrees most.")
    emit("4. ADDITIVITY: the breath-sum vs FULL check (table above, column 'sum/FULL') is the clean "
         "linear decomposition (no slot-class leak) and should read ~1.0 on every group; the slot-class "
         "checks (RIGHT+WRONG, GIVEN+RELATION, FORWARD+INVERSE) are expected to read BELOW 1.0 specifically "
         "on pres-head/islit-head (the two mean()-over-all-slots terms this script's masking corrupts "
         "rather than zeros) and on GIVEN+RELATION/FORWARD+INVERSE everywhere (the excluded mod/sel/pct/"
         "fdiv/macro/frac/chain and classify_row 'other' residue) -- see the additivity table for the "
         "actual numbers per group, which is the finding, not a bug to silently patch around.")
    emit("5. UNVERIFIED / FLAGGED: BATCH/NBATCH=" + f"{BATCH}/{NBATCH} rows {ROW0}:{ROW0+NBATCH*BATCH} of "
         f"{man.get('alg_train', '?')} is a SMALL slice (not the full diet) -- class-size noise is "
         "real at this n (see CLASS SIZES above); the pres-head/islit-head leak's absolute size is reported "
         "via the additivity shortfall, not independently decomposed from the genuine presence-conditioned "
         "signal; the 'query' head's per-row CE term is NOT split by slot class at all (rides every class "
         "identically) and is absorbed into whichever group holds W_query (here: 'rest').")
    emit(f"6. SEE THE READING ABOVE for the organ-by-organ breakdown; this ran under checkpoint {CKPT} "
         f"({CKPT_TAG}); per-combo results read from {SAVE_DIR}.")

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"wrote {OUT}")


if AGGREGATE:
    aggregate_and_report()
    sys.exit(0)

# ===========================================================================
# WORKER MODE: one (batch, target) backward, saved to disk, then exit
# ===========================================================================
assert os.environ.get("DEV", "") == "CPU", (
    "credit_census.py is a zero-GPU census (THE PAWLS: never take .cache/gpu.lock) -- "
    "set DEV=CPU explicitly, every launch, no default.")
assert TARGET == "ALL" or TARGET in TARGETS, (
    f"CC_TARGET must be one of {TARGETS} or 'ALL' (every target for CC_BIDX, warm-cache, "
    f"skipping combos already on disk); got {TARGET!r}")
os.makedirs(SAVE_DIR, exist_ok=True)

import phase1_algebra_head as H
from phase1_algebra_head import (build_params, forward, _loss_single, load_alg,
                                  build_slot_masks, L_FAC, K_VARS, T_ALG,
                                  ident_build_array)
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from mycelium.rulebook import legal_digit_logits
import polarity_census as PC

K_B = int(os.environ.get("ALG_BREATH", "7"))
assert K_B == K_B_ENV

samples, states, tokmask, gold, sent = load_alg("train")
n = states.shape[0]
log(f"loaded diet split: {n} rows, L_FAC={L_FAC} K_VARS={K_VARS} K_B(ALG_BREATH)={K_B}")
need = ROW0 + BATCH * NBATCH
assert n >= need, f"only {n} rows staged, need {need} (ROW0={ROW0} BATCH={BATCH} NBATCH={NBATCH})"
assert 0 <= BIDX < NBATCH, f"CC_BIDX={BIDX} out of range [0, {NBATCH})"

p = build_params(0)
sd = safe_load(CKPT)
miss = [k for k in p if k not in sd]
for k in p:
    if k in sd:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
log(f"WARM_FROM {CKPT}: {len(p) - len(miss)}/{len(p)} keys loaded; stay-at-init: {miss[:8]}")

SHARED = sorted(p.keys())   # autograd sees every param; ALG_FREEZE only excludes the OPTIMIZER step
_off = 0
SPANS = {}
for k in SHARED:
    nk = int(np.prod(p[k].shape))
    SPANS[k] = (_off, _off + nk)
    _off += nk
TOTAL = _off

GROUPS = {**HEAD_GROUPS, **ORGAN_GROUPS}
_assigned = set(k for v in GROUPS.values() for k in v if k in p)
GROUPS["rest"] = [k for k in SHARED if k not in _assigned]
for gname, keys in GROUPS.items():
    present = [k for k in keys if k in p]
    missing = [k for k in keys if k not in p]
    if missing:
        log(f"group {gname}: params not in this checkpoint's build: {missing}")
    GROUPS[gname] = present
log("groups: " + ", ".join(f"{g}({len(ks)})" for g, ks in GROUPS.items()))

# the manifest: written every worker invocation (cheap, deterministic) so aggregate mode never
# needs to build the model at all
with open(manifest_path(), "w") as f:
    json.dump({"shared": SHARED, "spans": {k: list(v) for k, v in SPANS.items()},
               "total": TOTAL, "groups": GROUPS, "k_b": K_B,
               "alg_train": os.environ.get("ALG_TRAIN", "?")}, f)


def zero_grads():
    for t in p.values():
        t.grad = None


def grad_vec():
    return np.concatenate([
        (p[k].grad.detach().numpy().reshape(-1) if p[k].grad is not None
         else np.zeros(int(np.prod(p[k].shape)), np.float32))
        for k in SHARED]).astype(np.float32)   # float32 on disk -- half the bytes, plenty of precision for norms/cosines


def batch_tensors(sl):
    ts = Tensor(states[sl].astype(np.float32), dtype=dtypes.float)
    tk = Tensor(tokmask[sl].astype(np.float32), dtype=dtypes.float)
    se = Tensor(sent[sl].astype(np.int32), dtype=dtypes.int)
    idt = None
    if H.ALG_BUSREG or H.ALG_IDKEY:
        idt = Tensor(ident_build_array([samples[int(i)] for i in sl], T_ALG), dtype=dtypes.int)
    gb_np = {}
    for k, a in gold.items():
        gb_np[k] = np.asarray(a[sl])
    if "is_lit_f" not in gb_np and "is_lit" in gb_np:
        gb_np["is_lit_f"] = gb_np["is_lit"]
    return ts, tk, se, idt, gb_np


def to_gold_tensors(gb_np):
    g = {}
    for k, a in gb_np.items():
        g[k] = Tensor(a.astype(np.float32) if a.dtype.kind == "f" else a.astype(np.int32),
                       dtype=dtypes.float if a.dtype.kind == "f" else dtypes.int)
    return g


def open_pass_mask(ts, tk, se, idt, sl_sent):
    o0 = forward(p, ts, tk, se, ident=idt)
    onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
    mk_np = build_slot_masks(onp0, sl_sent.astype(np.int32))
    return Tensor(mk_np, dtype=dtypes.float)


def decode_pass(ts, tk, se, idt, mk, gb_np, texts):
    """The SECOND (masked) forward, no grad -- loop_val's own `ok`, field by field,
    under THE NUMERAL MASK on literal slots both sides agree are non-relation."""
    o = forward(p, ts, tk, se, slot_mask=mk, ident=idt)
    keys = ["pres", "ftype", "op", "islit", "dig", "args", "res"] + (["dup"] if "h_dup" in p else [])
    onp = {k: o[k].realize().numpy() for k in keys}
    B = onp["pres"].shape[0]
    ok = np.zeros((B, L_FAC), bool)
    vg = gb_np
    for bi in range(B):
        text = texts[bi]
        for j in range(L_FAC):
            if vg["presence"][bi, j] < 0.5:
                continue
            gft = int(vg["ftype"][bi, j])
            if gft != 0 and int(onp["ftype"][bi, j].argmax()) != 0:
                fake = legal_digit_logits(onp["dig"][bi, j], text)
                if fake is not None:
                    onp["dig"][bi, j] = fake
            f_pres = bool(onp["pres"][bi, j] > 0)
            f_ftype = int(onp["ftype"][bi, j].argmax()) == gft
            f_res = int(onp["res"][bi, j].argmax()) == int(vg["res"][bi, j])
            row_ok = f_pres and f_ftype and f_res
            if gft == 0:
                gset = set(np.where(vg["args"][bi, j] > .5)[0].tolist())
                f_op = int(onp["op"][bi, j].argmax()) == int(vg["op"][bi, j])
                if len(gset) == 1 and "dup" in onp:
                    f_args = bool(onp["dup"][bi, j] > 0) and int(np.argmax(onp["args"][bi, j])) in gset
                else:
                    top2 = set(np.argsort(-onp["args"][bi, j])[:2].tolist())
                    f_args = top2 == gset
                row_ok = row_ok and f_op and f_args
            else:
                f_dig = bool((onp["dig"][bi, j].argmax(-1) == vg["digits"][bi, j]).all())
                row_ok = row_ok and f_dig
            ok[bi, j] = row_ok
    return ok


def mask_gold(gb_np, keep):
    """keep: (B, L_FAC) bool. Zero presence + every kind-indicator field for excluded
    slots on a COPY (the input dict is never mutated -- every class reads the same base)."""
    out = dict(gb_np)
    m = keep.astype(np.float32)
    for k in ("presence",) + tuple(KIND_FIELDS):
        if k in out:
            a = out[k]
            mm = m.reshape(m.shape + (1,) * (a.ndim - 2)) if a.ndim > 2 else m
            out[k] = (a * mm).astype(a.dtype)
    return out


def full_ladder_loss(o, g, only_breath=None):
    """Replica of loss_fn's ladder sum (THE SURFACE: no change to behavior -- same formula,
    same weights, same /K_B normalization under BREATH_NORM unset), optionally isolating
    ONE rung (only_breath) so its gradient alone can be read."""
    tot = None
    for kb, ob in enumerate(o["breaths"]):
        if only_breath is not None and kb != only_breath:
            continue
        full = dict(o, **ob)
        w = 1.0 + kb / max(K_B - 1, 1)
        term = _loss_single(full, g) * w
        tot = term if tot is None else tot + term
    if BREATH_NORM:
        wsum = sum(1.0 + kb / max(K_B - 1, 1) for kb in range(K_B))
        return tot / wsum
    return tot / K_B


def backward_grad(ts, tk, se, idt, mk, g, only_breath=None):
    o = forward(p, ts, tk, se, slot_mask=mk, ident=idt)
    loss = full_ladder_loss(o, g, only_breath=only_breath)
    zero_grads()
    loss.backward()
    lv = float(loss.numpy())   # AFTER backward() -- the tinygrad quirk (dirbit_grad_probe.py's
                                # own note, reproduced here): .numpy() before .backward() detaches
                                # every grad.
    assert np.isfinite(lv), f"loss not finite: {lv}"
    return grad_vec(), lv


# ----- this invocation's batch: one target, or (CC_TARGET=ALL) every target for this batch in
# one process -- the warm-cache speedup (tinygrad's eager mode caches compiled kernels per
# distinct shape+graph; a fresh process pays the compile tax again on its FIRST call, so doing
# every target for a batch in ONE process amortizes that tax over ~14 calls instead of paying it
# ~14 times) -- combos already saved on disk are skipped, so a killed mid-batch run resumes
# clean (no progress lost beyond whichever single combo was mid-flight when it died).
lo = ROW0 + BIDX * BATCH
hi = lo + BATCH
sl = np.arange(lo, hi)
run_targets = TARGETS if TARGET == "ALL" else [TARGET]
pending = [t for t in run_targets if not os.path.exists(combo_path(BIDX, t))]
if not pending:
    log(f"batch {BIDX}: every requested target already on disk, nothing to do")
    sys.exit(0)
log(f"batch {BIDX} rows {lo}:{hi} -- building tensors ({len(pending)}/{len(run_targets)} targets pending: {pending})")
ts, tk, se, idt, gb_np = batch_tensors(sl)
mk = open_pass_mask(ts, tk, se, idt, sent[sl])
texts = [samples[int(i)]["text"] for i in sl]
log(f"batch {BIDX} -- decode pass (no grad) for RIGHT/WRONG")
ok = decode_pass(ts, tk, se, idt, mk, gb_np, texts)

pres = gb_np["presence"] > 0.5
is_rel = gb_np.get("is_rel", np.zeros_like(pres, np.float32)) > 0.5
ftype = gb_np["ftype"]
given = pres & (ftype == 1)
relation = pres & is_rel

dirclass = np.full((BATCH, L_FAC), None, dtype=object)
for bi, i in enumerate(sl):
    factors = samples[int(i)]["factors"]
    cls = PC.classify_row(factors)
    for k, c in enumerate(cls):
        if k < L_FAC:
            dirclass[bi, k] = c
fwd = relation & (dirclass == "fwd")
inv = relation & (dirclass == "inv")
other_rel = relation & ~fwd & ~inv

right = pres & ok
wrong = pres & ~ok

masks = {
    "FULL": pres, "RIGHT": right, "WRONG": wrong, "GIVEN": given,
    "RELATION": relation, "FORWARD": fwd, "INVERSE": inv,
}

# this batch's class-size census -- written every invocation (cheap, deterministic, idempotent)
meta = {name: int(m.sum()) for name, m in masks.items()}
meta["OTHER_REL"] = int(other_rel.sum())
meta["OTHER_FTYPE"] = int((pres & ~given & ~relation).sum())
meta["TOTAL_PRES"] = int(pres.sum())
with open(meta_path(BIDX), "w") as f:
    json.dump(meta, f)

for this_target in pending:
    if this_target.startswith("B"):
        kb = int(this_target[1:])
        g = to_gold_tensors(gb_np)
        v, lv = backward_grad(ts, tk, se, idt, mk, g, only_breath=kb)
        n_this = meta["TOTAL_PRES"]
    else:
        keep = masks[this_target]
        gmasked = mask_gold(gb_np, keep)
        g = to_gold_tensors(gmasked)
        v, lv = backward_grad(ts, tk, se, idt, mk, g)
        n_this = int(keep.sum())
    log(f"batch {BIDX} target {this_target:9s} n={n_this:4d} loss={lv:.4f} |grad|={np.linalg.norm(v):.4f}")
    np.savez(combo_path(BIDX, this_target), grad=v.astype(np.float32), loss=np.float64(lv), n=np.int64(n_this))
    log(f"saved {combo_path(BIDX, this_target)}")
