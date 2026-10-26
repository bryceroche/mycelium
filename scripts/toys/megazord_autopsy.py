"""megazord_autopsy.py -- T3a-AUTOPSY (ledger 2026-10-26 11:09). CPU. Re-train MEGAZORD exactly as T3a (seeds 0, 1, 2; trees read
from the banked T3a stores, read-only) and read the clock gate: p_go over (h in 1..12, b in 0..11) against go <=> b < h, the gate
as a function of d = h - b, per-speed plane amplitudes, and per-h accuracy. Hypothesis WRAP: seed 1's go region wraps around the
15-degree plane's 24-breath circle onto the stay cells at d <= -8."""
import json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import megazord_toy as M
from mycelium import tree_db as TDB

OUT = ".cache/megazord_autopsy.txt"
SEEDS = [int(s) for s in os.environ.get("TA_SEEDS", "0,1,2").split(",")]


def load_R(world):
    db = TDB.TreeDB(M.DBP.format("", world)); rows = db.load_leaves(); db.close()
    return np.stack([rows[f"{t}:{i}"][1] for t in range(M.NT) for i in range(M.NI)])


def train(seed, RA):
    import torch
    Net, F = M.build_net(); rng = np.random.default_rng(seed)
    torch.manual_seed(seed); net = Net("MEGAZORD"); tA = torch.tensor(RA, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=M.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=M.LR, total_steps=M.STEPS, pct_start=0.1)
    for step in range(M.STEPS):
        bt = M.episodes(rng, M.BATCH, M.TRAIN_H, "train", M.WORLDS["A"]); lt, li = net(bt, tA)
        loss = F.cross_entropy(lt.reshape(-1, M.NT), bt["t"][:, 1:].reshape(-1)) + F.nll_loss(li.reshape(-1, M.NI), bt["i"][:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval()
    return net, float(loss.detach())


def gate_grid(net):
    """p_go[h-1, b] for h in 1..NB, b in 0..NB-1 -- the same arithmetic as the forward pass."""
    import torch
    with torch.no_grad():
        wD = net.w.repeat(M.NH); G = np.zeros((M.NB, M.NB))
        for h in range(1, M.NB + 1):
            th = wD * h
            c, s = torch.cos(th), torch.sin(th); re, im = net.ck[0::2], net.ck[1::2]
            hopk = torch.stack([re * c - im * s, re * s + im * c], -1).flatten()
            for b in range(M.NB):
                th = wD * b; c, s = torch.cos(th), torch.sin(th); re, im = net.cq[0::2], net.cq[1::2]
                cq = torch.stack([re * c - im * s, re * s + im * c], -1).flatten()
                G[h - 1, b] = float(torch.sigmoid((hopk @ cq) * net.cs.exp() + net.cb))
        amp = {}
        for p, w in enumerate(wD.tolist()):
            sp = round(w * 180 / math.pi, 2)
            amp[sp] = amp.get(sp, 0.0) + float(net.ck[2 * p:2 * p + 2].norm() * net.cq[2 * p:2 * p + 2].norm() * net.cs.exp())
    return G, amp


def acc_by_h(net, RA, seed):
    import torch
    erng = np.random.default_rng(91_000 + seed); tA = torch.tensor(RA, dtype=torch.float32); r = {}
    with torch.no_grad():
        for h in range(1, M.NB + 1):
            bt = M.episodes(erng, M.N_EV, [h], "train", M.WORLDS["A"]); lt, li = net(bt, tA)
            r[h] = float(((lt[:, -1].argmax(-1) == bt["t"][:, -1]) & (li[:, -1].argmax(-1) == bt["i"][:, -1])).float().mean())
    return r


if __name__ == "__main__":
    assert not os.path.exists(OUT), "refusing to overwrite a banked read"
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    out = open(OUT, "w"); t0 = time.time(); RA = load_R("A"); res = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"T3a-AUTOPSY pid {os.getpid()} seeds {SEEDS} steps {M.STEPS}")
    H, B = np.meshgrid(np.arange(1, M.NB + 1), np.arange(M.NB), indexing="ij"); want = B < H; Dd = H - B
    for seed in SEEDS:
        net, loss = train(seed, RA)
        torch.save(net.state_dict(), f".cache/megazord_autopsy_s{seed}.pt")
        G, amp = gate_grid(net); a = acc_by_h(net, RA, seed); wrong = (G > 0.5) != want
        bad_go = int((wrong & want).sum()); bad_stay = [(int(h), int(b), int(h - b), round(float(G[h - 1, b]), 3))
                                                          for h, b in zip(H[wrong & ~want], B[wrong & ~want])]
        byd = {int(d): round(float(G[Dd == d].mean()), 3) for d in range(-(M.NB - 1), M.NB + 1)}
        say(f"seed {seed}: loss {loss:.3f}; acc by h " + " ".join(f"{h}:{a[h]:.2f}" for h in a))
        say(f"   gate wrong cells {int(wrong.sum())}/{wrong.size}: go cells wrong {bad_go}; stay cells wrong {len(bad_stay)} (h, b, d, p_go) {bad_stay}")
        say(f"   p_go by d " + " ".join(f"{d}:{v:.2f}" for d, v in byd.items()))
        say(f"   plane amplitude x scale by speed (deg/breath) {{{', '.join(f'{k}: {v:.2f}' for k, v in amp.items())}}}; bias {float(net.cb):.2f}")
        res.append(dict(seed=seed, loss=loss, acc=a, grid=G.tolist(), bad_go=bad_go, bad_stay=bad_stay, by_d=byd, amp=amp))
    json.dump(res, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    say("READ (WRAP, as registered)")
    for r in res:
        ds = [d for _, _, d, _ in r["bad_stay"]]
        say(f"   seed {r['seed']}: go cells wrong {r['bad_go']}; stay cells wrong {len(ds)} at d {sorted(set(ds))}; "
            f"all at d <= -8: {bool(ds) and all(d <= -8 for d in ds)}")
    say(f"DONE in {time.time() - t0:.0f}s")
