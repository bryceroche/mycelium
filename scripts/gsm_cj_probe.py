"""gsm_cj_probe.py -- THE COMBINATION-SPLIT PROBE (ledger 2026-11-08; Bryce: "word given for the combination split probe").
Serves: "the SQL cross join is the nemesis" -- does a reader carry TEST on combinations it has SEEN in TRAIN?

Census (zero-model, TRAIN gold programs): at the step grains TEST is covered (type 0/3978 unseen, type bigram 25, structure 26);
only the PROBLEM SKELETON (the gold type sequence) is novel: 2362/3978 TEST steps sit in problems whose skeleton no TRAIN problem has.
So the probe splits TEST steps by skeleton seen / unseen and reads step_all on each, stratified by program length m (unseen skeletons
run longer): CJ = sum_m w_m (acc_seen_m - acc_unseen_m) / sum_m w_m, w_m = n_seen n_unseen / (n_seen + n_unseen).

Banked weights, re-scored by the arm's own scorer (BR.score; per-step results in item order, e["m"] steps each).
VALID gate: the re-scored TEST step_all reproduces the banked FINAL within 0.003, or the read is VOID.
Arm (CJ_ARM): BT4RB8_s{0,1} (torch, .pt, BT4RB8's flags) | BTC2_s{0,1} (tinygrad, .safetensors + atoms; gsm_btc defaults = BTC2).
-> .cache/gsm_cj_{ARM}.npz (per-step ok + skeleton-seen + m + k) and a summary line; scripts/gsm_cj_verdict.py reads the frame.
"""
import json, os, sys

ARM = os.environ["CJ_ARM"]
TG = os.path.exists(f".cache/gsmb_btc_{ARM}.safetensors")                   # a gsm_btc arm (tinygrad weights) vs a BR arm (.pt)
if not TG and ARM.startswith("BT4RB8"):
    for k_, v_ in dict(BT_D="512", BT_NH="8", BT_DB="64", BT_CYCLOCK="1", BT_ZOOM="1", BT_ROLE="1", BT_T6OFF="1",
                       RD_STATES="gsmb_rb8_states.npy", BT_ROLEF="gsmb_role_feats_rb8.npy").items():
        os.environ[k_] = v_                                                     # BT4RB8's own flags (as gsm_reader_ptr_read.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if TG:
    import gsm_btc as BTC
    BR = BTC.BR
else:
    import gsm_breath_reader as BR      # FIRST (the 11-04 import rule)
import numpy as np

K = BR.K


def skeleton(it):
    return tuple((o, BR.TR.first(A)[0], BR.TR.first(B)[0]) for o, A, B in it["prog"][:K])


def main():
    Dt, states = BR.TR.load(); tree = BR.TR.load_tree(); BR.RF = np.load(BR.GD.V1.C(BR.rolef_name()), mmap_mode="r")
    seen = {skeleton(it) for it in Dt["TRAIN"]}
    its = Dt["TEST"]
    if TG:
        from tinygrad.nn.state import safe_load
        t0_, E0 = BR.batch(Dt["TRAIN"][:2], states, tree, np.random.default_rng(0)); P0 = BTC.port_batch(t0_, E0, Dt["TRAIN"][:2], 64)
        bt = BTC.BTM.BreathingTransformer(BTC.config(P0["x"].shape[-1], P0["aux"].shape[-1]))
        sd = safe_load(f".cache/gsmb_btc_{ARM}.safetensors"); own = bt.state_dict()
        assert set(sd) == set(own), f"key mismatch: {sorted(set(sd) ^ set(own))[:8]}"           # eval loads hard-error on mismatch
        for k, v in own.items(): v.assign(sd[k].to(v.device)).realize()
        atoms = BTC.Atoms(BTC.NA, BTC.DB, BR.NCAP); atoms.C = np.load(f".cache/gsmb_btc_{ARM}_atoms.npy")
        r = BTC.score(bt, atoms, its, states, tree); banked = json.load(open(f".cache/gsmb_btc_{ARM}.json"))["TEST_step_all"]
    else:
        import torch
        torch.set_num_threads(int(os.environ.get("TOY_THREADS", "6")))
        net = BR.make_net(); net.load_state_dict(torch.load(f".cache/gsmb_breath_{ARM}.pt")); net.eval()
        r = BR.score(net, its, states, tree); banked = json.load(open(f".cache/gsmb_breath_{ARM}.json"))["TEST_step_all"]
    m = [min(len(it["prog"]), K) for it in its]
    ok = np.asarray(r["step_all"], bool); assert len(ok) == sum(m), (len(ok), sum(m))
    sn = np.concatenate([[skeleton(it) in seen] * mm for it, mm in zip(its, m)]).astype(bool)
    mm_ = np.concatenate([[mm] * mm for mm in m]); kk = np.concatenate([np.arange(mm) for mm in m])
    np.savez(f".cache/gsm_cj_{ARM}.npz", ok=ok, seen=sn, m=mm_, k=kk, banked=banked)
    print(f"[cj] {ARM} TEST step_all {ok.mean():.4f} (banked {banked:.4f}, |d| {abs(ok.mean() - banked):.4f}) | seen {ok[sn].mean():.4f} "
          f"(n {sn.sum()}) unseen {ok[~sn].mean():.4f} (n {(~sn).sum()})", flush=True)


if __name__ == "__main__":
    main()
