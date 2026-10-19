"""complex_width_ctrl.py -- THE WIDTH CONTROL for rungs 0/0b (2026-10-19; Fable's catch, Bryce's word for the queue).
At matched PARAMETERS the complex arms carry a WIDER stream (48 planes = 96 reals per token vs A4's 64), so their gain
could be equivariance (weights forced to commute with per-plane rotation) or just width-per-parameter. A_wide = v1's A4
at D = 96, unconstrained (about 2x the params). The same 48-attribute data as rung 0b (identical code and seeds);
cells f = 0.2 and the floor 0.0368; lr swept {1e-3, 3e-3} on the f=0.2 seed-0 val. Read against rung 0b's arms.
Out: .cache/complex_width_ctrl.txt/.json"""
import json, os, sys, time
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch  # noqa: E402
import binding_toy as V1  # noqa: E402
V1.N_ATTR = 48; V1.REL_PARTNER = 48; V1.D = 96
COVER = [0.2, 0.0368]
torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
out = open(".cache/complex_width_ctrl.txt", "w"); res = []; t0 = time.time()
def say(s):
    print(s, flush=True); out.write(s + "\n"); out.flush()
sc = {lr: V1.run_part1("A4", 0.2, 0, lr)["val"] for lr in (1e-3, 3e-3)}
lr = max(sc, key=sc.get); say(f"lr sweep A_wide (A4 at D=96): {sc} -> {lr}")
for f in COVER:
    rs = [V1.run_part1("A4", f, s, lr) for s in V1.SEEDS]
    agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
    res.append(dict(arm="A_wide", f=f, lr=lr, seeds=rs, agg=agg))
    say(f"f={f:<6} A_wide   params {rs[0]['params']:>7}  val {agg['val']:.3f}  HELDOUT {agg['heldout']:.3f}±{np.std([r['heldout'] for r in rs]):.3f}  "
        f"(hop1 {agg['heldout_hop1']:.3f} hop2 {agg['heldout_hop2']:.3f} hop3 {agg['heldout_hop3']:.3f})  [{time.time() - t0:.0f}s]")
json.dump(res, open(".cache/complex_width_ctrl.json", "w"), indent=1)
say(f"DONE in {time.time() - t0:.0f}s (verdict read against .cache/complex_lowcov_toy.json; bars in the ledger)")
