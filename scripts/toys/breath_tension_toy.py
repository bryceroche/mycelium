"""breath_tension_toy.py -- T2y, THE ORTHODONTIST: DOES SNAP TENSION PREDICT ERRORS, AND WHERE IS IT BLIND? (2026-10-25;
Bryce's word; ledger "T2y"). CPU. A READ-ONLY instrument on T2u's 12-breath loop (breath_transfer_toy, imported): every
breath logs the snap's TENSION -- the top1 - top2 centroid margin (PRIMARY, pinned) and the pre-snap distance
1 - cos(state, snapped centroid) (secondary) -- per episode, per breath, and on the training batch every 500 steps
(the consolidation curve). The tension is registered in mycelium/diagnostic_register.py ("snap_tension", loss-never);
the training door asserts the loss reads only the trace CE.
Arms (12000 steps, 3 seeds, per-breath trace supervision, T2u's train/held split):
  ROT_M    T2u's rotation arm (honest; few errors)
  MICRO_M  T2u's label arm (honest; fails on held hop counts -> many honest errors to detect)
  PRIVATE  ROT_M trained on a SYSTEMATICALLY RELABELED trace: every target entity in the low half (< NE/2) is replaced by a
           fixed derangement pi of that half -- a consistent private code, T2r HUNCH's shape. Judged against the TRUTH.
Episode tension = the MIN margin over the breaths (the tightest snap). AUROC(error | -tension)."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

_argv = sys.argv[:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import breath_transfer_toy as B
sys.argv = _argv
from mycelium.diagnostic_register import assert_not_supervised

SMOKE = B.SMOKE
ARMS = os.environ.get("TY_ARMS", "ROT_M,MICRO_M,PRIVATE").split(",")
OUT = ".cache/breath_tension{}.txt".format("_smoke" if SMOKE else "")
NE, NB = B.NE, B.NB
_p = np.random.default_rng(7).permutation(NE // 2); _pi = np.empty_like(_p); _pi[_p] = np.roll(_p, 1)   # one cycle: no fixed point
assert not (_pi == np.arange(NE // 2)).any()
PI = torch.from_numpy(np.concatenate([_pi, np.arange(NE // 2, NE)])).long()   # derangement of the low half, identity above


class Net(B.Net):
    """T2u's Net; forward additionally returns per-breath (margin, distance) -- observational, no gradient path is added."""
    def forward(self, bt):
        lg, lgs = super().forward(bt)
        with torch.no_grad():
            Cn = F.normalize(self.C, dim=-1)
            cos = lgs * B.TAU                                               # (B, NB, NE): cos(pre-snap state, centroid)
            top = cos.topk(2, -1).values
            self.margin = top[..., 0] - top[..., 1]; self.dist = 1 - top[..., 0]
        return lg, lgs


def auroc(score, pos):
    """P(score of a positive > score of a negative), ties half."""
    s, p = np.asarray(score, float), np.asarray(pos, bool)
    if p.all() or not p.any():
        return float("nan")
    r = s.argsort().argsort().astype(float)
    for v in np.unique(s):                                                  # average ranks over ties
        m = s == v
        r[m] = r[m].mean()
    n1, n0 = p.sum(), (~p).sum()
    return float((r[p].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def run(arm, seed):
    base = "ROT_M" if arm == "PRIVATE" else arm
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    ev = {h: B.episodes(erng, B.N_PER_H, [h]) for h in range(1, NB + 1)}
    torch.manual_seed(seed); net = Net(base)
    opt = torch.optim.AdamW(net.parameters(), lr=B.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=B.LR, total_steps=B.STEPS, pct_start=0.1)
    assert_not_supervised(["trace_ce"])                                     # the door: the loss reads the trace CE only
    curve = []
    for step in range(B.STEPS):
        bt = B.episodes(rng, B.BATCH, B.TRAIN_H); _, lgs = net(bt)
        tgt = PI[bt["traj"]] if arm == "PRIVATE" else bt["traj"]
        loss = F.cross_entropy(lgs.reshape(-1, NE), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0 or step == B.STEPS - 1:
            curve.append((step, float(net.margin.min(1).values.mean()), float(net.dist.max(1).values.mean())))
    net.eval(); rows = []
    with torch.no_grad():
        for h, bt in ev.items():
            lg, _ = net(bt); ok = (lg.argmax(-1) == bt["y"]).numpy()
            lowhalf = (bt["traj"] < NE // 2).any(1).numpy()                 # PRIVATE's code touched this episode
            for j in range(len(ok)):
                rows.append((h, bool(ok[j]), float(net.margin[j].min()), float(net.dist[j].max()), bool(lowhalf[j]),
                             net.margin[j].numpy().round(4).tolist()))
    ok = np.array([r[1] for r in rows]); mg = np.array([r[2] for r in rows]); ds = np.array([r[3] for r in rows])
    held = np.array([r[0] in B.HELD_H for r in rows]); err = ~ok
    r = dict(acc=float(ok.mean()), n_err=int(err.sum()), auroc_margin=auroc(-mg, err), auroc_dist=auroc(ds, err),
             auroc_margin_held=auroc(-mg[held], err[held]), curve=curve,
             margin_ok=float(mg[ok].mean()) if ok.any() else float("nan"), margin_err=float(mg[err].mean()) if err.any() else float("nan"),
             settle=np.mean([r[5] for r in rows], 0).round(4).tolist())
    if arm == "PRIVATE":
        lo = np.array([r[4] for r in rows])
        r.update(acc_untouched=float(ok[~lo].mean()), acc_coded=float(ok[lo].mean()),
                 margin_coded=float(mg[lo].mean()), margin_untouched=float(mg[~lo].mean()))
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    assert SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    assert abs(auroc([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-9 and abs(auroc([1, 1, 1, 1], [0, 1, 0, 1]) - 0.5) < 1e-9
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {'smoke' if SMOKE else 'full'} steps {B.STEPS} seeds {B.SEEDS} arms {ARMS} breaths {NB}")
    for arm in ARMS:
        rs = [run(arm, s) for s in B.SEEDS]
        sc = [k for k in rs[0] if isinstance(rs[0][k], (int, float))]
        ag = {k: float(np.nanmean([x[k] for x in rs])) for k in sc}; A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        say(f"{arm:8s} acc {ag['acc']:.3f} errors {ag['n_err']:.0f} | AUROC(error) margin {ag['auroc_margin']:.3f} "
            f"(seeds {[round(x['auroc_margin'], 3) for x in rs]}) dist {ag['auroc_dist']:.3f} held-only {ag['auroc_margin_held']:.3f} | "
            f"margin right {ag['margin_ok']:.3f} wrong {ag['margin_err']:.3f}  [{time.time() - t0:.0f}s]")
        if arm == "PRIVATE":
            say(f"         private code: acc on coded episodes {ag['acc_coded']:.3f} vs untouched {ag['acc_untouched']:.3f}; "
                f"margin coded {ag['margin_coded']:.3f} vs untouched {ag['margin_untouched']:.3f}")
        say(f"         settle (mean margin by breath, seed 0): {rs[0]['settle']}")
        say(f"         consolidation (step, mean min-margin, mean max-dist on the training batch, seed 0): "
            f"{[(s_, round(m, 3), round(d, 3)) for s_, m, d in rs[0]['curve'][::4]]}")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if not SMOKE and all(a in A for a in ("MICRO_M", "PRIVATE")):         # PINNED (ledger 2026-10-25, T2y)
        mi, pv = A["MICRO_M"], A["PRIVATE"]
        say("VERDICTS (3-seed means)")
        y1 = mi["auroc_margin"] >= 0.80 and (A["ROT_M"]["auroc_margin"] >= 0.80 or A["ROT_M"]["n_err"] < 30)
        say(f"Y1 TENSION DETECTS HONEST ERRORS: {'PASS' if y1 else 'FAIL'} (MICRO_M AUROC {mi['auroc_margin']:.3f} on "
            f"{mi['n_err']:.0f} errors; ROT_M {A['ROT_M']['auroc_margin']:.3f} on {A['ROT_M']['n_err']:.0f})")
        y2 = pv["auroc_margin"] <= 0.65
        say(f"Y2 TENSION IS BLIND TO A PRIVATE CODE: {'PASS' if y2 else 'FAIL'} (PRIVATE AUROC {pv['auroc_margin']:.3f}; acc vs truth "
            f"{pv['acc']:.3f}; coded episodes {pv['acc_coded']:.3f})")
    say(f"DONE in {time.time() - t0:.0f}s")
