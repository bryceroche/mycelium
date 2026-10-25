"""tree_unet_toy.py -- T2v, THE U-NET BREATH: ZOOM OUT ON THE TREE (2026-10-25; Bryce's word: "Unet to zoom out on the tree";
ledger "T2v"). CPU, self-contained.
Each breath = compress (UNBIND the identity, climb to the coarse node), reason at the TYPE level, expand (REBIND: walk back
down to the leaf, the identity carried by a NARROW skip that holds only the address under the parent, never content --
the residual-seal law).
TREE: NT types x NI indices = 64 leaves; leaf (t, i). TASK: each episode's facts are TYPE-level (a random permutation
sigma over the types, NT facts "t -> sigma(t)"); query = a start leaf (t, i); three breaths, each hop maps (t, i) ->
(sigma(t), i). Answer (sigma^3(t), i). Per-breath trace supervision (as T2s-T2u).
HELD-OUT LEAVES: (t + i) % 4 == 0 (16 of 64; every type and every index still appears). Training episodes whose trace
touches a held-out leaf (start or any hop) are rejected; the test split is episodes whose ANSWER is a held-out leaf.
Arms (MHA-only breath, 4 heads, hard straight-through snap):
  FLAT_FREE      flat snap over 64 free leaf centroids (the cross-join memorizer)
  FLAT_FACTORED  flat snap over 64 COMPOSED centroids normalize(Ct[t] + Ci[i]); the residual stream may carry the index
  UNET           hierarchical: the state is (type code, index pointer); the breath's query reads the TYPE code only, the
                 update writes the type code only, the snap is to the NT type centroids; the index pointer rides the skip
                 untouched; the leaf read = type logits + index logits (the walk back down)"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = "--smoke" in sys.argv
ARMS = os.environ.get("TV_ARMS", "FLAT_FREE,FLAT_FACTORED,UNET").split(",")
NT, NI, NB, D, NH = 8, 8, 3, 64, 4
NL, DH = NT * NI, D // NH
TAU, BATCH, LR = 0.1, 128, 3e-3
STEPS = 300 if SMOKE else 12000
SEEDS = [0] if SMOKE else [0, 1, 2]
N_EVAL = 500 if SMOKE else 3000
OUT = ".cache/tree_unet{}.txt".format("_smoke" if SMOKE else "")
HELD = np.array([[(t + i) % 4 == 0 for i in range(NI)] for t in range(NT)])
assert HELD.any(0).all() and HELD.any(1).all() and (~HELD).any(0).all() and (~HELD).any(1).all()


def episodes(rng, n, split):
    """split 'train': no trace leaf held out; 'held': the answer leaf is held out."""
    out = {k: [] for k in ("sig", "t", "i", "traj")}; got = 0
    while got < n:
        m = 4 * n
        sig = np.stack([rng.permutation(NT) for _ in range(m)]); t0 = rng.integers(0, NT, m); i0 = rng.integers(0, NI, m)
        ts = [t0]
        for _ in range(NB):
            ts.append(sig[np.arange(m), ts[-1]])
        ts = np.stack(ts, 1)                                                # (m, NB+1)
        h = HELD[ts, i0[:, None]]
        keep = ~h.any(1) if split == "train" else h[:, -1]
        k = np.nonzero(keep)[0][: n - got]; got += len(k)
        out["sig"].append(sig[k]); out["t"].append(t0[k]); out["i"].append(i0[k]); out["traj"].append(ts[k, 1:] * NI + i0[k, None])
    c = {k: torch.from_numpy(np.concatenate(v)).long() for k, v in out.items()}
    return dict(sig=c["sig"], x=c["t"] * NI + c["i"], traj=c["traj"], y=c["traj"][:, -1])


class Net(nn.Module):
    def __init__(self, arm):
        super().__init__(); self.arm = arm
        self.Ct = nn.Parameter(torch.randn(NT, D) / math.sqrt(D))          # type centroids (the coarse level)
        self.Ci = nn.Parameter(torch.randn(NI, D) / math.sqrt(D))          # index centroids (the address under a parent)
        if arm == "FLAT_FREE":
            self.Cl = nn.Parameter(torch.randn(NL, D) / math.sqrt(D))
        self.read = nn.Linear(2 * D, D)                                      # type facts (t, sigma t) -> memory
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))

    def leaves(self):
        if self.arm == "FLAT_FREE":
            return F.normalize(self.Cl, dim=-1)
        return F.normalize(self.Ct[:, None] + self.Ci[None], dim=-1).reshape(NL, D)

    def attend(self, s, mem):
        Bn, L = s.shape[0], mem.shape[1]
        Q = self.q(s).view(Bn, NH, DH); K = self.k(mem).view(Bn, L, NH, DH).transpose(1, 2)
        V = self.v(mem).view(Bn, L, NH, DH).transpose(1, 2)
        a = torch.softmax(torch.einsum("bhd,bhld->bhl", Q, K) / math.sqrt(DH), -1)
        return self.o(torch.einsum("bhl,bhld->bhd", a, V).reshape(Bn, D))

    @staticmethod
    def st(lg, C):
        soft = lg.softmax(-1); hard = F.one_hot(lg.argmax(-1), C.shape[0]).float()
        return (hard + soft - soft.detach()) @ C

    def forward(self, bt):
        Bn = bt["x"].shape[0]; Ctn = F.normalize(self.Ct, dim=-1); Cin = F.normalize(self.Ci, dim=-1)
        src = torch.arange(NT).expand(Bn, -1)
        mem = self.read(torch.cat([Ctn[src], Ctn[bt["sig"]]], -1))          # (B, NT, D): type-level facts only
        lgs = []
        if self.arm == "UNET":
            tc = Ctn[bt["x"] // NI]; ip = bt["x"] % NI                     # compress: the type code; the skip: the index pointer
            for b in range(NB):
                t2 = tc + self.attend(tc, mem)                              # reason at the type level only
                lt = F.normalize(t2, dim=-1) @ Ctn.T / TAU
                tc = self.st(lt, Ctn)                                       # snap at the coarse level
                li = F.one_hot(ip, NI).float() * (1 / TAU)                  # expand: the skip's address under the new parent
                lgs.append((F.log_softmax(lt, -1)[:, :, None] + F.log_softmax(li, -1)[:, None]).reshape(Bn, NL))
            return lgs[-1], torch.stack(lgs, 1)
        Cl = self.leaves(); s = Cl[bt["x"]]
        for b in range(NB):
            s2 = s + self.attend(s, mem)
            lg = F.normalize(s2, dim=-1) @ Cl.T / TAU
            s = self.st(lg, Cl); lgs.append(F.log_softmax(lg, -1))
        return lgs[-1], torch.stack(lgs, 1)


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    ev = {"train": episodes(erng, N_EVAL, "train"), "held": episodes(erng, N_EVAL, "held")}
    torch.manual_seed(seed); net = Net(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = episodes(rng, BATCH, "train"); _, lgs = net(bt)
        loss = F.nll_loss(lgs.reshape(-1, NL), bt["traj"].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); r = {"loss_end": float(loss.detach())}
    with torch.no_grad():
        for sp, bt in ev.items():
            lg, lgs = net(bt); pr = lg.argmax(-1)
            r[f"{sp}_acc"] = float((pr == bt["y"]).float().mean())
            r[f"{sp}_type"] = float((pr // NI == bt["y"] // NI).float().mean())
            r[f"{sp}_index"] = float((pr % NI == bt["y"] % NI).float().mean())
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    assert SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    e = episodes(np.random.default_rng(0), 200, "train")
    assert not HELD.reshape(-1)[torch.cat([e["x"][:, None], e["traj"]], 1).numpy()].any(), "a held-out leaf leaked into training"
    e = episodes(np.random.default_rng(1), 200, "held"); assert HELD.reshape(-1)[e["y"].numpy()].all()
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {'smoke' if SMOKE else 'full'} steps {STEPS} seeds {SEEDS} arms {ARMS} tree {NT}x{NI} breaths {NB} held leaves {int(HELD.sum())}")
    for arm in ARMS:
        rs = [run(arm, s) for s in SEEDS]
        ag = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}; A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        say(f"{arm:13s} in-dist {ag['train_acc']:.3f} | HELD-OUT LEAF {ag['held_acc']:.3f} (seeds {[round(x['held_acc'], 3) for x in rs]}) "
            f"type {ag['held_type']:.3f} index {ag['held_index']:.3f} | loss {ag['loss_end']:.3f}  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if not SMOKE and all(a in A for a in ("FLAT_FREE", "FLAT_FACTORED", "UNET")):   # PINNED (ledger 2026-10-25, T2v)
        u, ff, fr = A["UNET"], A["FLAT_FACTORED"], A["FLAT_FREE"]
        sd = lambda a: [x["held_acc"] for x in allres[ARMS.index(a)]["seeds"]]
        say("VERDICTS (3-seed means)")
        c = u["train_acc"] >= 0.95 and ff["train_acc"] >= 0.95
        say(f"CONTROL: in-dist UNET {u['train_acc']:.3f}, FLAT_FACTORED {ff['train_acc']:.3f} -> {'OK' if c else 'VOID'}")
        say(f"V1 THE U-NET BREATH GENERALIZES TO UNSEEN LEAVES: {'PASS' if u['held_acc'] >= 0.95 else 'FAIL'} (UNET held-out {u['held_acc']:.3f})")
        d = [a - b for a, b in zip(sd("UNET"), sd("FLAT_FACTORED"))]
        v2 = u["held_acc"] - ff["held_acc"] >= 0.05 and all(x > 0 for x in d)
        say(f"V2 BEYOND A FACTORED FLAT CODEBOOK: {'PASS' if v2 else 'FAIL'} (UNET - FLAT_FACTORED {u['held_acc'] - ff['held_acc']:+.3f}; "
            f"per seed {[round(x, 3) for x in d]})")
        say(f"READ FLAT_FREE (cross-join memorizer): in-dist {fr['train_acc']:.3f}, held-out {fr['held_acc']:.3f}")
    say(f"DONE in {time.time() - t0:.0f}s")
