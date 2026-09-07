"""seamtest_adapter_contract.py — v0/v1 commit-adapter equivalence on the
LIVE head (2026-09-06, ultrareview chapter A fix): the sign branch (E1
negative literal -> value negated, decode's rule) and the door-#12 dargs
branch (dedicated dup pointer preferred over the shared args argmax).
Reuses seamtest_vector.py's fixtures (synthetic consistent problems +
real form_mix3 rows), then INJECTS 'sgn' and 'dargs' heads so both new
branches fire, and asserts np.array_equal(v0, v1) on every batch plus
that each branch actually changed something vs. the no-key run.
No GPU (DEV=CPU), no tinygrad graph — numpy adapter only. Zero writes."""
import os, sys
os.environ.setdefault("DEV", "CPU")
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import seamtest_vector as sv                      # fixtures only (main() not run on import)
import phase1_algebra_head as H

def run(onp, se, nv, m, v0):
    os.environ["ALG_SEAM_V0"] = "1" if v0 else "0"
    if not v0: os.environ.pop("ALG_SEAM_V0", None)
    return H.alt2_fact_buf(onp, se, nv, m, theta=0.9)

def main():
    rng = np.random.RandomState(7)
    total = 0; sgn_hits = 0; dargs_hits = 0
    for seed in range(6):
        onp, nv, m = sv.synthetic_batch(16, seed) if seed % 2 == 0 else sv.real_batch(16, seed)
        se = None                                    # unused by the adapter (signature symmetry)
        base0 = run(dict(onp), se, nv, m, True); base1 = run(dict(onp), se, nv, m, False)
        assert np.array_equal(base0, base1), f"no-key v0/v1 drift (seed {seed})"
        # inject sgn: fire on ~30% of slots; dargs: a pointer that disagrees with args argmax
        B, L = onp["pres"].shape
        onp2 = dict(onp)
        onp2["sgn"] = np.where(rng.rand(B, L) < 0.3, 4.0, -4.0).astype(np.float32)
        d = rng.randn(B, L, H.K_VARS).astype(np.float32)
        alt = (onp["args"].argmax(-1) + 1) % H.K_VARS
        np.put_along_axis(d, alt[..., None], 9.0, axis=-1)          # confident, different pick
        onp2["dargs"] = d
        v0 = run(onp2, se, nv, m, True); v1 = run(onp2, se, nv, m, False)
        assert np.array_equal(v0, v1), f"keyed v0/v1 drift (seed {seed})"
        sgn_hits += int((v0 != base0).any(axis=(1, 2)).sum())
        # dargs alone
        onp3 = dict(onp); onp3["dargs"] = d
        d0 = run(onp3, se, nv, m, True); d1 = run(onp3, se, nv, m, False)
        assert np.array_equal(d0, d1), f"dargs v0/v1 drift (seed {seed})"
        dargs_hits += int((d0 != base0).any(axis=(1, 2)).sum())
        total += B
    assert sgn_hits > 0 and dargs_hits > 0, "a branch never fired — test is blind"
    print(f"[adapter-contract] PASS: v0 == v1 on {total} items x {{none, sgn+dargs, dargs}}; "
          f"sign branch changed {sgn_hits} items, dargs branch changed {dargs_hits} items")

if __name__ == "__main__":
    main()
