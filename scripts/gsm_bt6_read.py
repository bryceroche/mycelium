"""BT6 THE CONFIDENCE-GATED COLLAPSE (ledger 2026-11-02 18:22): a zero-training read on banked BT weights.

Run with the weights' own BT_* env (BT4 or BT5 config) and BT_LOAD=.cache/gsmb_breath_{TAG}.pt.
  1. fit rho0 on TRAIN rows (Dt["TRAIN"][:600]) over the pinned grid, max step_all  (never on a scored split)
  2. score ORIG and TEST under the schedule (BT_GATE "") and under the gate (rho0) on the same weights
  -> .cache/gsmb_bt6_{TAG}.json + one summary line. Verdict: scripts/gsm_bt6_verdict.py.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import gsm_breath_reader as BR

GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
TAG = os.path.basename(BR.LOAD).replace("gsmb_breath_", "").replace(".pt", "")
assert BR.LOAD and os.path.exists(BR.LOAD) and not BR.GATE, "need BT_LOAD weights and no BT_GATE in the env (the read sets it)"
torch.set_num_threads(int(os.environ.get("TOY_THREADS", "3")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
if BR.ROLE: BR.RF = np.load(BR.GD.V1.C(os.environ.get("BT_ROLEF", "gsmb_role_feats.npy")), mmap_mode="r")
net = BR.make_net(); net.load_state_dict(torch.load(BR.LOAD)); net.eval()


def run(gate, items):
    BR.GATE = gate
    return BR.score(net, items, states, tree)


fit = {r: float(run(str(r), Dt["TRAIN"][:600])["step_all"].mean()) for r in GRID}
rho = max(GRID, key=lambda r: fit[r])
print(f"[bt6] {TAG} TRAIN fit step_all " + " ".join(f"{r}:{v:.3f}" for r, v in fit.items()) + f" -> rho0 {rho}", flush=True)
out = dict(tag=TAG, load=BR.LOAD, soft=BR.SOFT, grid=GRID, fit=fit, rho0=rho)
for split in ("ORIG", "TEST"):
    S, G = run("", Dt[split]), run(str(rho), Dt[split])
    assert len(S["step_all"]) == len(G["step_all"])
    d = G["step_all"].astype(float) - S["step_all"].astype(float)
    sh = lambda R: np.bincount(R["commit_cycle"][R["commit_cycle"] >= 0], minlength=BR.NCYC).tolist()
    out[split] = dict(sched={m: float(S[m].mean()) for m in ("ans", "step_all", "type_ok")}, gate={m: float(G[m].mean()) for m in ("ans", "step_all", "type_ok")},
                      diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), n=len(d), share_sched=sh(S), share_gate=sh(G))
    o = out[split]
    print(f"[bt6] {TAG} {split} step sched {o['sched']['step_all']:.4f} gate {o['gate']['step_all']:.4f} diff {o['diff']:+.4f} "
          f"(paired SE over steps {o['se']:.4f}, n {o['n']}) | ans {o['sched']['ans']:.3f}->{o['gate']['ans']:.3f} "
          f"type {o['sched']['type_ok']:.3f}->{o['gate']['type_ok']:.3f} | share {o['share_sched']} -> {o['share_gate']}", flush=True)
json.dump(out, open(f".cache/gsmb_bt6_{TAG}.json", "w"), indent=1)
