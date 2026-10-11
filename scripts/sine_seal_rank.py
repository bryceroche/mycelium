"""sine_seal_rank.py -- THE SEAL RANK (2026-10-10), ported from scripts/sine_breath.py's own
_seal_rank numeric Jacobian test to the REAL head's _depth_stack_sine. Run on a SMALL SYNTHETIC
body (B=1, L=3 slots, T=5 token positions -- not the production L_TOT=24/T_ALG shape: the
sketch's own method is a full-basis finite-difference SVD, n = L * H_W = 1536 directions,
batched as 2n = 3072 forward rows in ONE call; at the production L=24 this would be n = 12288,
a 24576-row batch -- intractable on CPU for a diagnostic. The SMALL body is a deliberate
tractability simplification (documented here, not hidden): it still exercises the real
_depth_stack_sine code path (same params, same breath-to-breath `state["sine_boundary_carry"]`
threading), just at a shrunk L.

METHOD: call _depth_stack_sine TWICE, threading one `state` dict -- breath b (producing the
first falling state and, if skip=="boundary", writing state["sine_boundary_carry"]), then
breath b+1 on a FRESH random `cur_b1` (b+1's own grounding input, independent of b's output --
exactly the sketch's own setup, where only the CARRY, not cur itself, connects the breaths).
Perturb breath b's OUTPUT (the thing that *could* flow into breath b+1 only through whatever
road skip provides) by +-eps along every standard basis direction of its (L, H_W) shape,
re-run breath b+1 with EACH perturbed carry, and take the Jacobian of breath (b+1)'s own output
w.r.t. breath b's output. With skip="peak" or "none" there is no carry at all (boundary_in is
None), so breath b's own output cannot reach breath b+1 by this road AT ALL -- the Jacobian is
then measuring breath b+1's sensitivity to its own (fixed) grounding input only, i.e. it should
read as a CONSTANT (rank 0) w.r.t. this particular perturbation, by construction (the sealed
bound is trivially 0, not L*128 -- see the printed verdict). With skip="boundary" the carry
*is* breath b's output, so the Jacobian is exactly d(breath b+1 output)/d(breath b's own carry),
rank BOUNDED ONLY by min(L*H_W, L*H_W) = unbounded by the crystal (the placebo bypasses it).

Requires the arm's own env (ALG_SINE=<sine|anti|flat>, ALG_DEPTH=4, ALG_POLAR=1 ALG_POLAR_D=128,
ALG_HW=512) exported by the caller. The ReZero output doors (dp{i}_wo/xwo/ffn_w2) are forced to
SMALL-RANDOM (not their zero birth value) for this numeric test only -- the sketch's own
o_init="small" convention: at literal ReZero birth every block's delta is exactly zero and the
rank would trivially read 0 everywhere, telling us nothing about the ARCHITECTURE's own bound.
"""
import os
import sys

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("DEV", "CPU")
import numpy as np

import phase1_algebra_head as H
from tinygrad import Tensor

assert H.ALG_SINE, "ALG_SINE=<sine|anti|flat> required"
assert H.ALG_DEPTH == 4, "ALG_SINE requires ALG_DEPTH=4"

EPS = 1e-3
L, T, H_W = 3, 5, H.H_W
TROUGH_W = 128 if H._SINE_TROUGH_I is not None else 0
BOUND = L * TROUGH_W


def _small_rezero(p, seed=7):
    rng = np.random.RandomState(seed)
    for i in range(H.ALG_DEPTH):
        w = H._SINE_WIDTHS[i]
        for nm, shp in ((f"dp{i}_wo", (w, w)), (f"dp{i}_xwo", (w, w)),
                        (f"dp{i}_ffn_w2", (H.DEPTH_FFN_HID, w))):
            p[nm].assign(Tensor((rng.randn(*shp) * (0.5 / np.sqrt(shp[0]))).astype(np.float32))).realize()


def _run_breath(p, cur_np, kb, state, waist_np, tokmask_np, slot_mask_np):
    B = cur_np.shape[0]
    cur = Tensor(cur_np.astype(np.float32))
    waist = Tensor(waist_np.astype(np.float32))
    tokmask = Tensor(tokmask_np.astype(np.float32))
    slot_mask = Tensor(slot_mask_np.astype(np.float32)) if slot_mask_np is not None else None
    out = H._depth_stack_sine(p, cur, slot_mask, waist, tokmask, B, kb, state)
    return out


def seal_rank(skip):
    os.environ["ALG_SINE_SKIP"] = skip
    import importlib
    importlib.reload(H)
    assert H.ALG_SINE_SKIP == skip
    p = H.build_params(0)
    _small_rezero(p)

    rng = np.random.default_rng(3)
    cur_b0 = rng.standard_normal((1, L, H_W)).astype(np.float32)
    cur_b1_base = rng.standard_normal((1, L, H_W)).astype(np.float32)
    waist_np = rng.standard_normal((1, T, H_W)).astype(np.float32)
    tokmask_np = np.ones((1, T), np.float32)
    slot_mask_np = np.ones((1, L, L), np.float32)

    # breath b: run once to get the falling state and (if boundary) seed the carry
    state_b = {}
    out_b = _run_breath(p, cur_b0, 0, state_b, waist_np, tokmask_np, slot_mask_np)
    y0_b = out_b.numpy().astype(np.float64)

    n = L * H_W
    xs = np.repeat(cur_b0.reshape(1, -1), 2 * n, axis=0)
    xs[np.arange(n) * 2, np.arange(n)] += EPS
    xs[np.arange(n) * 2 + 1, np.arange(n)] -= EPS
    xs = xs.reshape(2 * n, L, H_W)

    # re-run breath b for every perturbed row (batched), to get the perturbed carry
    state_b_batched = {}
    out_b_pert = _run_breath(p, xs, 0,
                              state_b_batched,
                              np.repeat(waist_np, 2 * n, axis=0),
                              np.repeat(tokmask_np, 2 * n, axis=0),
                              np.repeat(slot_mask_np, 2 * n, axis=0))
    carries = out_b_pert.numpy().astype(np.float64)   # (2n, L, H_W): breath b's own (perturbed) output

    # breath b+1 on a FIXED grounding input, carrying EACH perturbed carry forward
    cur_b1 = np.repeat(cur_b1_base, 2 * n, axis=0)
    state_b1 = {"sine_boundary_carry": Tensor(carries.astype(np.float32))} if skip == "boundary" else {}
    out_b1 = _run_breath(p, cur_b1, 1, state_b1,
                          np.repeat(waist_np, 2 * n, axis=0),
                          np.repeat(tokmask_np, 2 * n, axis=0),
                          np.repeat(slot_mask_np, 2 * n, axis=0))
    y1 = out_b1.numpy().reshape(2 * n, -1).astype(np.float64)

    J = ((y1[0::2] - y1[1::2]) / (2 * EPS)).T   # (outputs, n): d(breath b+1 out)/d(breath b out)
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int((sv > 1e-2 * max(sv[0], 1e-12)).sum())
    return rank, n, sv[:5]


def main():
    print(f"[seal-rank] ALG_SINE={H.ALG_SINE} trough_i={H._SINE_TROUGH_I} widths={H._SINE_WIDTHS} "
          f"L={L} T={T} H_W={H_W} crystal bound (L*128)={BOUND} full n={L * H_W}")
    for skip in ("peak", "none", "boundary"):
        rank, n, sv5 = seal_rank(skip)
        verdict = ("sealed: rank bounded by the crystal" if rank <= BOUND
                   else "BYPASSED: the crystal is optional")
        print(f"[seal-rank] skip={skip:8s} rank {rank:4d} / {n}  (crystal bound {BOUND})  "
              f"top sv {[round(float(x), 4) for x in sv5]}  {verdict}")


if __name__ == "__main__":
    main()
