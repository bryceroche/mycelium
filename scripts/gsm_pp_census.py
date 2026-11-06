"""gsm_pp_census.py -- the PHASE-BOUND POINTERS' census (ledger 2026-11-06 06:38, the pin's beside; the pre/post knob law). Zero GPU.

Banked PP s0 / s1 (BT_PHPTR=1, BT4RB8's flags) on TEST[:256] with the READER PER-POINTER protocol (gold slots < k committed, cond = gold types,
cycle floor(k NCYC / K)). By exact gain ablation on the token logits at numeral positions of each read slot: base = all gains 0; the relation
m's post-gain contribution = (only g_m,r live) - base; pre-gain = post / g_m,r. Reports the std over (slot, numeral) of base and of each
contribution per role (numerals the kind mask leaves open), and the trained gains (init qn_a +1.6, prev_b +2.8, sib_b -3.0, rest 0).  -> .cache/gsm_pp_census.json + printed table
"""
import json, math, os, sys

for k_, v_ in dict(BT_D="512", BT_NH="8", BT_DB="64", BT_CYCLOCK="1", BT_ZOOM="1", BT_ROLE="1", BT_T6OFF="1", BT_PHPTR="1",
                   RD_STATES="gsmb_rb8_states.npy", BT_ROLEF="gsmb_role_feats_rb8.npy").items():
    os.environ[k_] = v_
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gsm_breath_reader as BR          # FIRST (the 11-04 import rule)
import torch

torch.set_num_threads(int(os.environ.get("TOY_THREADS", "6")))
Dt, states = BR.TR.load(); tree = BR.TR.load_tree()
BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
K, NCYC = BR.K, BR.NCYC; TAG = os.environ.get("PC_TAG", "PP"); SEEDS = [int(s) for s in os.environ.get("PC_SEEDS", "0,1").split(",")]
MN = ("qn", "prev", "sib")


def logits(net, t, E, xq, progs, gt, k, B, L):
    com = np.zeros((B, K), bool); com[:, :k] = True
    nb = BR.nb_tensors(BR.notebook(E, L, progs, com, gt, np.ones((B, K), np.float32)))
    _, pa, pb, _, _ = net.breath(t, xq, nb, torch.as_tensor(gt), torch.full((B,), min(k * NCYC // K, NCYC - 1)))
    return pa[:, k], pb[:, k]


o = {}
for s in SEEDS:
    net = BR.make_net(); net.load_state_dict(torch.load(f".cache/gsmb_breath_{TAG}_s{s}.pt")); net.eval(); g0 = net.pg.detach().clone()
    acc = {(r, m): [] for r in (0, 1) for m in ("base",) + MN}
    with torch.no_grad():
        for s0 in range(0, 256, 128):
            part = Dt["TEST"][s0:s0 + 128]
            t, E = BR.batch(part, states, tree, np.random.default_rng(s0)); L = t["pad"].shape[1]; B = len(part); Lq = L - K
            for b, it in enumerate(part): E[b]["vals"] = it["vals"]
            progs, gt = BR.gold(E); xq = net.read(t); m_ = np.array([e["m"] for e in E])
            for k in range(K):
                rows = m_ > k
                if not rows.any(): continue
                net.pg.zero_(); base = logits(net, t, E, xq, progs, gt, k, B, L)
                live = [t["isnum"][rows, :Lq] & (base[r][rows, :Lq] > -1e4 / math.sqrt(BR.D) + 1) for r in (0, 1)]   # numerals the kind mask leaves open
                for r in (0, 1): acc[r, "base"].append(base[r][rows, :Lq][live[r]].numpy())
                for j, mn in enumerate(MN):
                    net.pg.zero_(); net.pg[j] = g0[j]; one = logits(net, t, E, xq, progs, gt, k, B, L)
                    for r in (0, 1): acc[r, mn].append((one[r] - base[r])[rows, :Lq][live[r]].numpy())
                net.pg.copy_(g0)
    sd = {f"{'ab'[r]}_{m}": float(np.concatenate(v).std()) for (r, m), v in acc.items()}
    o[f"s{s}"] = dict(gains=g0.tolist(), std=sd,
                      pre={f"{'ab'[r]}_{m}": (sd[f"{'ab'[r]}_{m}"] / abs(float(g0[j, r])) if abs(float(g0[j, r])) > 1e-9 else None)
                           for r in (0, 1) for j, m in enumerate(MN)})
    print(f"[census] {TAG}_s{s} gains (qn / prev / sib x a, b) {np.round(g0.numpy(), 3).tolist()}", flush=True)
    for r in (0, 1):
        print(f"  role {'ab'[r]}: std base {sd[f'{chr(97 + r)}_base']:.3f} | post-gain " + " ".join(f"{m} {sd[f'{chr(97 + r)}_{m}']:.3f} "
              f"({sd[f'{chr(97 + r)}_{m}'] / sd[f'{chr(97 + r)}_base']:.2f}x base)" for m in MN), flush=True)
json.dump(o, open(f".cache/gsm_pp_census{'' if TAG == 'PP' else '_' + TAG}.json", "w"), indent=1)
