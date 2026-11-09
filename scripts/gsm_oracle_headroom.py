"""gsm_oracle_headroom.py -- THE PLAN HEADROOM READ (ledger 2026-11-09 pin): banked BTC weights scored on TEST with the GOLD step type
given at every slot (BT_ORACLE=1: a perfect macro plan + its residual kinds), split by skeleton seen / unseen as the CJ probe.
If the pointers under a perfect plan still fail on unseen skeletons, a plan-level organ cannot carry TEST-U alone.
Arm: HR_ARM (a gsm_btc tag with .safetensors + atoms). -> .cache/gsm_hr_{ARM}.npz + a summary line (never touches the CJ npz)."""
import os, sys

os.environ["BT_ORACLE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_cj_probe as CP                     # its skeleton + the gsm_btc import order
BTC, BR = CP.BTC, CP.BR
import numpy as np
from tinygrad.nn.state import safe_load

ARM = os.environ["HR_ARM"]; K = BR.K
assert BR.ORACLE
Dt, states = BR.TR.load(); tree = BR.TR.load_tree(); BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
seen = {CP.skeleton(it) for it in Dt["TRAIN"]}; its = Dt["TEST"]
t0_, E0 = BR.batch(Dt["TRAIN"][:2], states, tree, np.random.default_rng(0)); P0 = BTC.port_batch(t0_, E0, Dt["TRAIN"][:2], 64)
bt = BTC.BTM.BreathingTransformer(BTC.config(P0["x"].shape[-1], P0["aux"].shape[-1]))
sd = safe_load(f".cache/gsmb_btc_{ARM}.safetensors"); own = bt.state_dict()
assert set(sd) == set(own), f"key mismatch: {sorted(set(sd) ^ set(own))[:8]}"
for k, v in own.items(): v.assign(sd[k].to(v.device)).realize()
atoms = BTC.Atoms(BTC.NA, BTC.DB, BR.NCAP); atoms.C = np.load(f".cache/gsmb_btc_{ARM}_atoms.npy")
r = BTC.score(bt, atoms, its, states, tree)
m = [min(len(it["prog"]), K) for it in its]
ok = np.asarray(r["step_all"], bool); ty = np.asarray(r["type_ok"], bool); assert len(ok) == sum(m)
sn = np.concatenate([[CP.skeleton(it) in seen] * mm for it, mm in zip(its, m)]).astype(bool)
np.savez(f".cache/gsm_hr_{ARM}.npz", ok=ok, seen=sn, type_ok=ty, ptr=np.asarray(r["step_ptr"], bool))
print(f"[hr] {ARM} ORACLE type_ok {ty.mean():.4f} | TEST step_all {ok.mean():.4f} seen {ok[sn].mean():.4f} TEST-U {ok[~sn].mean():.4f} "
      f"| ans {np.mean(r['ans']):.4f}", flush=True)
