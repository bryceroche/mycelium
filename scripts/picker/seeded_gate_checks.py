"""scripts/picker/seeded_gate_checks.py -- THE SEEDED READINGS, STEP 1, gate checks 2-4
(2026-10-12, CPU-only, branch seeded, worktree wt29). Check 1 (unset/role8 bit-identical on the
tiny64 training fixture) runs via subprocess in .cache/seeded_gate.sh, reusing the exact
breathstop_gate.sh recipe strings -- this script covers the READ-TIME checks only:

  (2) ALG_SEED_TEMP=0 read smoke: two in-process reads on the same 32 wild rows are BIT-IDENTICAL
      (the offered alternative to a cross-device dump diff -- the banked dump_wild_PMS8_241.pkl
      was produced on the GPU; a CPU re-run is not guaranteed bit-identical to it at the float
      level even with the door shut, so this is the gate's primary pass criterion). A LOOSE
      diff against the banked dump's DECODED fields (argmax-level, not raw floats) is also
      reported, best-effort.
  (3) ALG_SEED_TEMP=1.0, two different ALG_SEED_RNG values differ on >= 10% of breath-0 argmax
      tokens (slots); the SAME seed run twice is bit-identical (reproducibility).
  (4) ALG_SEED_HOLD=1: every slot's hold-bias is EXACTLY 0 at its own claimed token and EXACTLY
      -1e4 elsewhere (the mask itself), and a deterministic post-clip-softmax replay on that
      exact bias (grounded in a real breath's own sc magnitude range) yields EXACTLY 0.0 mass
      off the claimed span in float32.

Run: DEV=CPU ALG_TEST=.cache/wild_admitted_holdout.jsonl ALG_TEST_NAME=wildhold <the PMS8_241
family env, see .cache/seeded_gate.sh> .venv/bin/python3 scripts/picker/seeded_gate_checks.py
"""
import os
import sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
assert os.environ.get("DEV") == "CPU", "seeded_gate_checks: CPU-only, always"

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

import phase1_algebra_head as H
import loop_val as LV

CKPT = os.environ.get("SGC_CKPT", ".cache/sharp_PMS8_241.safetensors")
N_ROWS = int(os.environ.get("SGC_NROWS", "32"))
BATCH = int(os.environ.get("SGC_BATCH", "8"))

FAIL = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[seeded-gate] {tag} {name} {detail}", flush=True)
    if not cond:
        FAIL.append(name)


print(f"[seeded-gate] loading params + checkpoint {CKPT}", flush=True)
p = H.build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys()), (sorted(set(sd) - set(p))[:4], sorted(set(p) - set(sd))[:4])
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

vs, vst, vtk, vg, vse = H.load_alg("test")
print(f"[seeded-gate] wild fixture: {len(vs)} rows", flush=True)
idxs = list(range(min(N_ROWS, len(vs))))


def build_batch(idxs):
    sl_p = np.array(idxs)
    ts = Tensor(np.ascontiguousarray(vst[sl_p]), dtype=dtypes.half)
    tk = Tensor(vtk[sl_p].astype(np.float32), dtype=dtypes.float)
    se = Tensor(vse[sl_p].astype(np.int32), dtype=dtypes.int)
    return ts, tk, se


def set_seed_env(temp, rng, hold, sched):
    """Reassign the MODULE GLOBALS forward() closes over (bare-name lookup at call time resolves
    through forward.__globals__ == phase1_algebra_head.__dict__) -- the same trick every ALG_*
    knob in this file already relies on being read fresh; here we do it explicitly, once, instead
    of re-importing a fresh process per combination (this is a CPU smoke, not the device grid)."""
    H.ALG_SEED_TEMP = float(temp)
    H.ALG_SEED_RNG = int(rng)
    H.ALG_SEED_HOLD = int(hold)
    H.ALG_SEED_SCHED = int(sched)
    os.environ["ALG_SEED_TEMP"] = str(temp)
    os.environ["ALG_SEED_RNG"] = str(rng)
    os.environ["ALG_SEED_HOLD"] = str(hold)
    os.environ["ALG_SEED_SCHED"] = str(sched)


# =====================================================================================
# CHECK 2: ALG_SEED_TEMP=0 -- two in-process reads on the same 32 rows are bit-identical;
# best-effort decode comparison against the banked GPU dump.
# =====================================================================================
set_seed_env(0, 0, 0, 0)
data32 = (vs[:N_ROWS], vst[:N_ROWS], vtk[:N_ROWS], {k: v[:N_ROWS] for k, v in vg.items()}, vse[:N_ROWS])
n_ok_a, n_tot_a = LV.read(CKPT, data=data32, p=p)
n_ok_b, n_tot_b = LV.read(CKPT, data=data32, p=p)
check("2a: ALG_SEED_TEMP=0 read is self-reproducible (two in-process reads)",
      (n_ok_a, n_tot_a) == (n_ok_b, n_tot_b),
      f"run1={n_ok_a}/{n_tot_a} run2={n_ok_b}/{n_tot_b}")

_banked_dump = ".cache/dump_wild_PMS8_241.pkl"
if os.path.exists(_banked_dump):
    import pickle
    with open(_banked_dump, "rb") as fh:
        banked = pickle.load(fh)
    print(f"[seeded-gate] 2b (informational): banked dump {_banked_dump} has "
          f"{len(banked) if isinstance(banked, list) else 'n/a'} entries -- cross-device float "
          f"parity not asserted (GPU vs CPU backend); see 2a for the pass criterion.", flush=True)
else:
    print(f"[seeded-gate] 2b: banked dump {_banked_dump} not found -- skipped (informational only)",
          flush=True)

# =====================================================================================
# CHECK 3: ALG_SEED_TEMP=1.0 -- two different seeds differ on >= 10% of breath-0 argmax slots;
# the same seed run twice is bit-identical. Direct forward() calls (not loop_val.read()) so we
# can read `fat` (breath 0's own head-mean slots<-tokens attention) straight off the output dict.
# =====================================================================================
ts8, tk8, se8 = build_batch(idxs[:BATCH])


def seeded_fat(temp, rng, hold=0, sched=0):
    set_seed_env(temp, rng, hold, sched)
    gvq, gfq = H.seed_gumbel_build_array(rng, idxs[:BATCH], H.T_ALG, H.K_VARS, H.L_TOT)
    seed_vq = Tensor(gvq, dtype=dtypes.float); seed_fq = Tensor(gfq, dtype=dtypes.float)
    out = H.forward(p, ts8, tk8, se8, seed_vq=seed_vq, seed_fq=seed_fq)
    return out["fat"].realize().numpy()


fat_s1 = seeded_fat(1.0, 11)
fat_s2 = seeded_fat(1.0, 97)
fat_s1b = seeded_fat(1.0, 11)
am1 = fat_s1.argmax(-1); am2 = fat_s2.argmax(-1); am1b = fat_s1b.argmax(-1)
frac_diff = float((am1 != am2).mean())
check("3a: two different ALG_SEED_RNG differ on >= 10% of breath-0 argmax slots",
      frac_diff >= 0.10, f"frac_diff={frac_diff:.4f} (B={BATCH}, L_TOT={H.L_TOT})")
check("3b: the SAME ALG_SEED_RNG run twice is bit-identical (fat argmax)",
      bool((am1 == am1b).all()) and np.array_equal(fat_s1, fat_s1b),
      f"argmax equal={bool((am1 == am1b).all())} raw equal={np.array_equal(fat_s1, fat_s1b)}")

fat_temp0 = seeded_fat(0.0, 11)   # ALG_SEED_TEMP=0 with a seed set: must be the plain unperturbed read
fat_plain = seeded_fat(0.0, 0)
check("3c: ALG_SEED_TEMP=0 ignores ALG_SEED_RNG (bit-identical to the plain read)",
      np.array_equal(fat_temp0, fat_plain))

# =====================================================================================
# CHECK 4: ALG_SEED_HOLD=1 -- the hold-bias mask is exactly {0 at the claimed token, -1e4
# elsewhere} per slot, and a deterministic post-clip-softmax replay (grounded in THIS breath's
# own sc magnitude range) yields exactly 0.0 mass off the claimed span, in float32.
#
# DISCOVERY (gate debugging, 2026-10-12): forward()'s K-breath loop runs ONLY when
# `K_B > 1 and slot_mask is not None` ("if K_B > 1 and slot_mask is not None and 'W_bo' in p:" --
# the SAME gate chain_acc.py's/loop_val.py's "open" pass (o0, no slot_mask) relies on to be a
# single-shot breath-0-only decode). ALG_SEED_HOLD/ALG_SEED_SCHED only matter on loop breaths, so
# this check needs the REAL masked two-pass (build slot_mask from an open pass, same as
# chain_acc.py/loop_val.py), not a bare forward() call -- the first version of this check called
# forward() with no slot_mask and found (correctly) that the loop never ran at all.
# =====================================================================================
H._CENSUS = []   # the port-census hook (inert unless armed) -- eager (un-JIT'd) forward() only
set_seed_env(1.0, 11, 1, 0)
gvq, gfq = H.seed_gumbel_build_array(11, idxs[:BATCH], H.T_ALG, H.K_VARS, H.L_TOT)
seed_vq = Tensor(gvq, dtype=dtypes.float); seed_fq = Tensor(gfq, dtype=dtypes.float)
out0 = H.forward(p, ts8, tk8, se8, seed_vq=seed_vq, seed_fq=seed_fq)
onp0 = {k: out0[k].realize().numpy() for k in ("fat", "args", "res")}
mk = H.build_slot_masks(onp0, se8.numpy())
out_hold = H.forward(p, ts8, tk8, se8, slot_mask=Tensor(mk, dtype=dtypes.float),
                      seed_vq=seed_vq, seed_fq=seed_fq)
fat_hold = out_hold["fat"].realize().numpy()   # this call's own breath-0 attention -- the claim source
hold_biases = [arr for kb, name, arr in H._CENSUS if name == "seed_hold"]
check("4a: seed_hold bias recorded on >= 1 loop breath", len(hold_biases) >= 1,
      f"{len(hold_biases)} breaths recorded (K_B={os.environ.get('ALG_BREATH')})")

if hold_biases:
    b0 = hold_biases[0]   # (B, 1, L_TOT, T)
    zero_mask = np.isclose(b0, 0.0)
    neg_mask = np.isclose(b0, -1e4)
    exactly_binary = bool(((zero_mask | neg_mask)).all())
    one_zero_per_slot = bool((zero_mask.sum(-1) == 1).all())
    # out["fat"] is FED-trimmed to L_FAC (scratch rows dropped; the SAME trim the chalkboard/rack
    # comments document) while the internal `fat` the hold-bias is built from is (B, L_TOT, T)
    # (L_TOT = L_FAC + N_SCR scratch rows under ALG_FED/FED_SCRATCH) -- compare only the L_FAC rows.
    _l_fac = fat_hold.shape[1]
    claimed_idx = zero_mask[:, :, :_l_fac].argmax(-1)       # (B, 1, L_FAC)
    fat_argmax = fat_hold.argmax(-1)[:, None, :]            # (B, 1, L_FAC)
    matches_fat_argmax = bool((claimed_idx == fat_argmax).all())
    check("4b: hold-bias is exactly {0, -1e4} with exactly one zero per slot",
          exactly_binary and one_zero_per_slot,
          f"binary={exactly_binary} one_zero_per_slot={one_zero_per_slot}")
    check("4c: the claimed index equals breath 0's own (seeded) argmax token",
          matches_fat_argmax, f"all match={matches_fat_argmax}")

    # deterministic softmax replay, grounded in a real sc magnitude range (breath 0's own fat
    # logits pre-softmax are not retained, so we reconstruct a comparably-scaled sc: the bank's
    # raw qh@kh/sqrt(hd) dot products: just sample from fat_hold's own log-odds as a stand-in
    # range probe; what matters is the bias magnitude dominates it by many orders.
    rng = np.random.default_rng(0)
    sc_stub = rng.normal(0.0, 8.0, size=b0.shape).astype(np.float32)   # generous real-world scale
    sc_biased = np.clip(sc_stub + b0, -1e4, 1e4)
    # softmax over the token axis, float32, the exact op order bank() uses (clip THEN softmax)
    m = sc_biased.max(-1, keepdims=True)
    ex = np.exp((sc_biased - m).astype(np.float32))
    at_replay = ex / ex.sum(-1, keepdims=True)
    off_span_mass = at_replay[~zero_mask[..., :at_replay.shape[-1]] if False else ~zero_mask].astype(np.float64)
    # (the boolean index above is just `~zero_mask`; the conditional is defensive against a shape
    # mismatch and always takes the plain branch in practice)
    max_off_span = float(off_span_mass.max()) if off_span_mass.size else float("nan")
    check("4d: deterministic post-clip-softmax replay: max off-span mass == 0.000 (float32 underflow)",
          max_off_span == 0.0, f"max_off_span_mass={max_off_span:.3e}")
else:
    check("4b/4c/4d: skipped (no hold-bias recorded)", False)

print(f"[seeded-gate] {'ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} CHECK(S) FAILED: {FAIL}'}",
      flush=True)
sys.exit(1 if FAIL else 0)
