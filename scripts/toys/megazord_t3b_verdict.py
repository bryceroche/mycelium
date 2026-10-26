"""megazord_t3b_verdict.py -- T3b THE SIX-WAVE CLOCK, the pinned bars (ledger 2026-10-26 11:23). Reads the banked runs
(.cache/megazord_t3b6.json, .cache/megazord_t3b4.json) and their state dicts; zero training.
  C6 CONTROL   six-wave MEGAZORD in-dist >= 0.95 on every seed (0-4)
  K1 CURE      six-wave MEGAZORD >= 0.95 on every seed on HELD_LEAF, HELD_H, JOINT, SWAP, EDGE
  E1 SNAP      among six-wave NO_SNAP seeds with in-dist >= 0.95: MEGAZORD EDGE - NO_SNAP EDGE >= 0.30 per same-seed pair;
               VOID if fewer than 2 land
  READS        four-wave base rate (seeds 3-4); NO_SNAP landing count; gate margin per seed (max p_go on stay cells vs
               min p_go on go cells over h 1..12, b 0..11)"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import megazord_toy as M
from megazord_autopsy import gate_grid

OUT = ".cache/megazord_t3b_verdict.txt"
BAR, SPLITS = 0.95, ("held_leaf", "held_h", "joint", "swap", "edge")


def seeds(path, arm):
    return [s for r in json.load(open(path))["arms"] if r["arm"] == arm for s in r["seeds"]]


def margin(tag, arm, seed):
    import torch
    Net, _ = M.build_net(); net = Net(arm); net.load_state_dict(torch.load(f".cache/megazord{tag}_{arm}_s{seed}.pt")); net.eval()
    G, _ = gate_grid(net); H, B = np.meshgrid(np.arange(1, M.NB + 1), np.arange(M.NB), indexing="ij"); go = B < H
    return float(G[~go].max()), float(G[go].min())


if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w")

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    mz, ns, mz4 = seeds(".cache/megazord_t3b6.json", "MEGAZORD"), seeds(".cache/megazord_t3b6.json", "NO_SNAP"), seeds(".cache/megazord_t3b4.json", "MEGAZORD")
    assert len(mz) == 5 and len(ns) == 5 and len(mz4) == 2, (len(mz), len(ns), len(mz4))
    fmt = lambda s: f"in-dist {s['indist']:.3f} " + " ".join(f"{k} {s[k]:.3f}" for k in SPLITS) + \
        " | h1-3 " + " ".join(f"{s[f'indist_h{h}']:.2f}" for h in (1, 2, 3))
    for name, tag, arm, rs, sd in (("MZ6", "_t3b6", "MEGAZORD", mz, range(5)), ("NO_SNAP6", "_t3b6", "NO_SNAP", ns, range(5)),
                                   ("MZ4", "_t3b4", "MEGAZORD", mz4, (3, 4))):
        for s, r in zip(sd, rs):
            st, gm = margin(tag, arm, s)
            say(f"{name:8s} s{s}: {fmt(r)} | gate: stays max {st:.3f}, go min {gm:.3f}")
    c6 = all(s["indist"] >= BAR for s in mz)
    k1 = all(s[k] >= BAR for s in mz for k in SPLITS)
    landed = [i for i, s in enumerate(ns) if s["indist"] >= BAR]
    gaps = [mz[i]["edge"] - ns[i]["edge"] for i in landed]
    e1 = "VOID" if len(landed) < 2 else ("PASS" if all(g >= 0.30 for g in gaps) else "FAIL")
    say("VERDICTS (per seed)")
    say(f"C6 CONTROL: {'PASS' if c6 else 'FAIL'} (in-dist {[round(s['indist'], 3) for s in mz]})")
    say(f"K1 THE CLOCK IS THE CURE: {'PASS' if k1 else 'FAIL'} (min over seeds: " + ", ".join(f"{k} {min(s[k] for s in mz):.3f}" for k in SPLITS) + ")")
    say(f"E1 THE SNAP'S AXIS IS EXTRAPOLATION: {e1} (NO_SNAP landed seeds {landed}; EDGE gaps {[round(g, 3) for g in gaps]})")
    say(f"READ four-wave base rate, seeds 3-4: in-dist {[round(s['indist'], 3) for s in mz4]}; landed {sum(s['indist'] >= BAR for s in mz4)}/2 "
        f"(T3a seeds 0-2: 2/3)")
    say(f"READ NO_SNAP6 landing: {len(landed)}/5")
