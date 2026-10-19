"""complex_lowcov_toy.py -- RUNG 0b: THE LOW-COVERAGE AXIS (2026-10-19; Bryce's word; bars pinned in the ledger first).
Rung 0's gap grew as coverage fell (C_clean - A4: +0.002 / +0.015 / +0.051 at f 0.5 / 0.2 / 0.1), but with 12 attributes
the coverage floor is exactly 0.100 (one pair per name plus the attr cover step: v1's identical f=0.05/0.02 rows). With 48
attributes the floor is 0.0368, so f in {0.2, 0.1, 0.05, floor} are real, distinct cells (realized coverage printed).
Everything else is rung 0's: v1's data code and training loop, the arms' code (complex_native_toy.CNet), 6000 steps,
batch 128, seeds 0/1/2. Arms A4 / C_clean / C_split (C_mod dropped: it lost rung 0). lr re-swept {1e-3, 3e-3} per arm
on the f=0.2 seed-0 val (the attribute count changed, so rung 0's picks do not carry over).
Out: .cache/complex_lowcov_toy{_smoke}.txt/.json"""
import json, os, sys, time
smoke = "--smoke" in sys.argv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import complex_native_toy as R0  # noqa: E402
V1 = R0.V1
V1.N_ATTR = 48; V1.REL_PARTNER = 48                     # read as module globals by make_batch / allowed_pairs / Net
FLOOR = 0.0368
COVER = [0.2, FLOOR] if smoke else [0.2, 0.1, 0.05, FLOOR]
ARMS = ["A4", "C_clean", "C_split"]

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = "_smoke" if smoke else ""
    out = open(f".cache/complex_lowcov_toy{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    for f in COVER:
        say(f"cell f={f}: realized coverage {V1.allowed_pairs(f, np.random.default_rng(100)).mean():.4f} (seed 0's allowed set)")
    lrs = {}
    for arm in ARMS:
        sc = {lr: R0.run(arm, 0.2, 0, lr)["val"] for lr in ([3e-3] if smoke else [1e-3, 3e-3])}
        lrs[arm] = max(sc, key=sc.get); say(f"lr sweep {arm}: {sc} -> {lrs[arm]}  [{time.time() - t0:.0f}s]")
    for f in COVER:
        for arm in ARMS:
            rs = [R0.run(arm, f, s, lrs[arm]) for s in V1.SEEDS]
            agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            sd = float(np.std([r["heldout"] for r in rs]))
            res.append(dict(arm=arm, f=f, lr=lrs[arm], seeds=rs, agg=agg, sd=sd))
            say(f"f={f:<6} {arm:<8} params {rs[0]['params']:>7}  val {agg['val']:.3f}  HELDOUT {agg['heldout']:.3f}±{sd:.3f}  "
                f"(hop1 {agg['heldout_hop1']:.3f} hop2 {agg['heldout_hop2']:.3f} hop3 {agg['heldout_hop3']:.3f})  [{time.time() - t0:.0f}s]")
    json.dump(res, open(f".cache/complex_lowcov_toy{tag}.json", "w"), indent=1)
    H = {(r["arm"], r["f"]): r["agg"]["heldout"] for r in res}
    gap = lambda arm, f: H[(arm, f)] - H[("A4", f)]
    for arm in ("C_clean", "C_split"):
        say(f"GAP {arm} - A4 by f: " + " | ".join(f"{f}: {gap(arm, f):+.4f}" for f in COVER))
    top, lo = COVER[0], COVER[-1]
    win = all(gap(a, lo) >= 0.05 and gap(a, lo) - gap(a, top) >= 0.03 for a in ("C_clean", "C_split"))
    dead = all(gap(a, lo) <= 0.01 for a in ("C_clean", "C_split"))
    say(f"VERDICT (pinned: STRUCTURE GENERALIZES = both complex arms >= A4 + 0.05 at the floor AND each gap widens >= +0.03 "
        f"from f={top}; DEAD = both <= A4 + 0.01 at the floor): {'STRUCTURE GENERALIZES' if win else 'DEAD' if dead else 'WEAK (between the bars)'}")
    say(f"DONE in {time.time() - t0:.0f}s")
