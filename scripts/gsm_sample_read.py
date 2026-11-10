"""gsm_sample_read.py -- THE SAMPLE READ (ledger 2026-11-10 pin; Bryce: "monte carlo pi estimate ... white noise"): banked BTC weights
scored on TEST N times with Gumbel noise on every commit choice (BR.score noise=(tau, rng): plan op, type, pointers), then VOTED.
  VOTE   per step: the majority (op, ptr a, ptr b) over the N draws (ties: the first drawn), graded as the scorer grades it.
  ANS    per problem: the majority executed value over the draws (None excluded; all None = wrong), graded by near(final).
  BEST@N per step / per problem: any draw right (the headroom a verifier could buy; uses the key -- descriptive only).
Arm: SR_ARM (+ the arm's own flags in the env). Greedy re-score must reproduce the banked TEST step_all within 0.003, or VOID.
-> .cache/gsm_sample_{ARM}.npz + .cache/gsm_sample_{ARM}.txt"""
import json, os, sys, time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_cj_probe as CP
BTC, BR = CP.BTC, CP.BR
import numpy as np
from tinygrad.nn.state import safe_load

ARM = os.environ["SR_ARM"]; N = int(os.environ.get("SR_N", "16")); TAUS = (1.0, 0.5); K = BR.K
Dt, states = BR.TR.load(); tree = BR.TR.load_tree(); BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
seen = {CP.skeleton(it) for it in Dt["TRAIN"]}; its = Dt["TEST"]
t0_, E0 = BR.batch(Dt["TRAIN"][:2], states, tree, np.random.default_rng(0)); P0 = BTC.port_batch(t0_, E0, Dt["TRAIN"][:2], 64)
bt = BTC.BTM.BreathingTransformer(BTC.config(P0["x"].shape[-1], P0["aux"].shape[-1]))
sd = safe_load(f".cache/gsmb_btc_{ARM}.safetensors"); own = bt.state_dict()
assert set(sd) == set(own), f"key mismatch: {sorted(set(sd) ^ set(own))[:8]}"
for k, v in own.items(): v.assign(sd[k].to(v.device)).realize()
atoms = BTC.Atoms(BTC.NA, BTC.DB, BR.NCAP); atoms.C = np.load(f".cache/gsmb_btc_{ARM}_atoms.npy")
m = [min(len(it["prog"]), K) for it in its]
sn = np.concatenate([[CP.skeleton(it) in seen] * mm for it, mm in zip(its, m)]).astype(bool)
pu = np.array([CP.skeleton(it) not in seen for it in its])                                  # problem-level unseen
banked = json.load(open(f".cache/gsmb_btc_{ARM}.json"))["TEST_step_all"]


def run(noise):
    t = time.time(); r = BTC.score(bt, atoms, its, states, tree, noise=noise); assert len(r["step_all"]) == sum(m)
    print(f"[sr] {ARM} {noise and noise[0]} step {np.mean(r['step_all']):.4f} ({time.time() - t:.0f}s)", flush=True)
    return r, time.time() - t


def vote_steps(rs):
    ok = np.zeros(sum(m), bool)
    for i in range(sum(m)):
        g = {}; c = Counter()
        for r in rs: c[r["pred"][i]] += 1; g.setdefault(r["pred"][i], bool(r["step_all"][i]))
        ok[i] = g[c.most_common(1)[0][0]]
    return ok


def vote_ans(rs):
    ok = np.zeros(len(its), bool)
    for j, it in enumerate(its):
        vs = [round(float(r["val"][j]), 6) for r in rs if r["val"][j] is not None and np.isfinite(float(r["val"][j]))]
        ok[j] = bool(vs) and BR.GD.V1.near(Counter(vs).most_common(1)[0][0], it["final"])
    return ok


g, dt = run(None); gok, gans = np.asarray(g["step_all"], bool), np.asarray(g["ans"], bool)
out = [f"{ARM} GREEDY TEST step {gok.mean():.4f} (banked {banked:.4f}, |d| {abs(gok.mean() - banked):.4f} "
       f"{'VALID' if abs(gok.mean() - banked) <= 0.003 else 'VOID'}) TEST-U {gok[~sn].mean():.4f} | ans {gans.mean():.4f} "
       f"ans-U {gans[pu].mean():.4f} | {dt:.0f}s/score"]
sv = dict(greedy=gok, greedy_ans=gans, seen=sn, pu=pu)
for tau in TAUS:
    rs = [run((tau, np.random.default_rng(1000 + i)))[0] for i in range(N)]
    S = np.stack([np.asarray(r["step_all"], bool) for r in rs]); A = np.stack([np.asarray(r["ans"], bool) for r in rs])
    vs, va = vote_steps(rs), vote_ans(rs); bs, ba = S.any(0), A.any(0)
    sv |= {f"t{tau}_S": S, f"t{tau}_A": A, f"t{tau}_vs": vs, f"t{tau}_va": va}
    out.append(f"  tau {tau}: one draw step {S.mean():.4f} | VOTE step {vs.mean():.4f} ({vs.mean() - gok.mean():+.4f}) TEST-U {vs[~sn].mean():.4f} "
               f"({vs[~sn].mean() - gok[~sn].mean():+.4f}) | ANS-VOTE {va.mean():.4f} ({va.mean() - gans.mean():+.4f}) | BEST@{N} step "
               f"{bs.mean():.4f} TEST-U {bs[~sn].mean():.4f} ans {ba.mean():.4f} | gap best-vote step {bs.mean() - vs.mean():+.4f}")
np.savez(f".cache/gsm_sample_{ARM}.npz", **sv)
txt = "\n".join(out); print(txt, flush=True); open(f".cache/gsm_sample_{ARM}.txt", "w").write(txt + "\n")
