"""cj_followup.py -- T3 follow-ups (2026-10-22; Bryce's word 15:1x; ledger "T3 FOLLOW-UPS").
(1) PARITY REGISTERS (Opus: RAID 5 instead of mirroring). Records are grouped in threes; a fourth register per group
    stores the SUM of the group's three registers. Equal floats: every register is shrunk to 3/4 width (PHASOR planes =
    3/8 W). A data register that fails its METADATA CHECK (its day field won't decode, best sim < 0.5) is rebuilt as
    parity − the other two, if the parity register survived and it is the group's only failure; otherwise it is dropped.
    The parity register's own failure is taken as DETECTED (an idealization: a real one would carry its own checksum).
    Analytic (fixed codes, hard reads; nothing trained). Compared with PHASOR and REDUNDANT (mirroring) at W 32 / 128.
(2) THE PER-HOP TABLE (Opus/coordinator: is FLAT's depth-3 collapse independent hops, clustered, or later hops worse?).
    Retrain FLAT / COMPRESS / PHASOR_L at W 128, 12.5 % coverage, snap (3 seeds, T3's recipe) and read, on UNSEEN depth-3
    chains: P(match 1), P(partner 1 | match 1), P(match 2 | all before), P(partner 2 | ...), P(match 3 | ...),
    P(value | all before), and the product vs the observed end-to-end accuracy.
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
os.environ.setdefault("CJ_W", "128")
import cross_join_toy as CJ  # noqa: E402

OUT = ".cache/cj_followup.txt"


class ParityMemory(CJ.VSAMemory):
    def __init__(self, alg):
        super().__init__([alg], check=False)

    def run(self, bt, snap, train, pfail=0.0, rng=None):
        a = self.algs[0]; reg = self.write(bt["R"])[0]                              # (B, K, P)
        B, K, P = reg.shape; G = (K + 2) // 3
        pad = torch.zeros(B, G * 3, P, dtype=reg.dtype); pad[:, :K] = reg
        par = pad.view(B, G, 3, P).sum(2)                                          # (B, G, P)
        if pfail > 0:
            reg = CJ.corrupt(reg, pfail, rng)
            pfl = torch.from_numpy(rng.random((B, G)) < pfail)                       # parity failures (detected)
        else:
            pfl = torch.zeros(B, G, dtype=torch.bool)
        bad = a.sim(a.unbind(reg, a.role(CJ.FIELDS["day"])), a.book("d")).max(-1).values <= 0.5   # (B, K) metadata check
        bad = bad & bt["M"]
        padr = torch.zeros(B, G * 3, P, dtype=reg.dtype); padr[:, :K] = reg
        badp = torch.zeros(B, G * 3, dtype=torch.bool); badp[:, :K] = bad
        gb = badp.view(B, G, 3); nbad = gb.sum(-1)
        others = (padr.view(B, G, 3, P) * (~gb)[..., None]).sum(2)                 # sum of the intact members
        rebuilt = (par - others)                                                   # = the one lost member
        fix = gb & ((nbad == 1) & ~pfl)[..., None]                                 # recoverable members
        new = torch.where(fix[..., None], rebuilt[:, :, None, :].expand(B, G, 3, P), padr.view(B, G, 3, P))
        new = torch.where((gb & ~fix)[..., None], torch.zeros_like(new), new)      # unrecoverable -> dropped
        regs = [new.view(B, G * 3, P)[:, :K]]
        # the read itself = VSAMemory.run's loop on prepared registers
        ar = torch.arange(B); cur = [a.code("e", bt["e0"])]; obj = [a.code("o", bt["o"])]
        for h in range(bt["D"]):
            sc = self.match(regs, [a.bind(cur[0], obj[0])]).masked_fill(~bt["M"], -1e9)
            w = F.one_hot(sc.argmax(-1), sc.shape[-1]).float()
            if h < bt["D"] - 1:
                lg = self.read(regs, w, "par", "e")
                cur = [a.book("e")[lg.argmax(-1)]] if snap else self.raw(regs, w, "par")
            else:
                return self.read(regs, w, "val", "v"), None, None


def parity_part(say):
    say("(1) PARITY vs MIRRORING vs SINGLE (analytic, unseen, cov 0.125, snap; acc at loss p 0 / 0.1 / 0.3)")
    for W in (32, 128):
        for name in ("PHASOR", "REDUNDANT", "PARITY"):
            rows = []
            for s in (0, 1, 2):
                g = torch.Generator().manual_seed(7000 + s); cov = CJ.coverage(0.125, s)
                if name == "PHASOR":
                    net = CJ.VSAMemory([CJ.Phasor(W // 2, g)])
                elif name == "REDUNDANT":
                    net = CJ.VSAMemory([CJ.Phasor(W // 4, g), CJ.Phasor(W // 4, g)], check=True)
                else:
                    net = ParityMemory(CJ.Phasor(W * 3 // 8, g))
                r = {}
                with torch.no_grad():
                    for D in (1, 3):
                        bt = CJ.batch(np.random.default_rng(80_000 + s * 10 + D + 5), CJ.N_EVAL, D, cov, "unseen")
                        for p in CJ.PFAIL:
                            lv, _, _ = net.run(bt, True, False, p, np.random.default_rng(int(p * 1000) + s))
                            r[(D, p)] = float((lv.argmax(-1) == bt["y"]).float().mean())
                rows.append(r)
            m = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
            say(f"   W {W:3d} {name:9s} D1 " + "/".join(f"{m[(1, p)]:.3f}" for p in CJ.PFAIL) + "   D3 " + "/".join(f"{m[(3, p)]:.3f}" for p in CJ.PFAIL))


def perhop_part(say):
    say("(2) PER-HOP TABLE (W 128, cov 0.125, snap, UNSEEN depth 3; conditional on every earlier step right)")
    for arm in ("PHASOR_L", "COMPRESS", "FLAT"):
        acc = []
        for s in (0, 1, 2):
            cov = CJ.coverage(0.125, s); net = CJ.make(arm, s); CJ.train(net, cov, s, True)
            bt = CJ.batch(np.random.default_rng(80_000 + s * 10 + 3 + 5), CJ.N_EVAL, 3, cov, "unseen")
            with torch.no_grad():
                lv, sel, par = net.run(bt, True, False)
            ar = torch.arange(len(bt["y"])); ok = torch.ones(len(bt["y"]), dtype=torch.bool); steps = []
            for h in range(3):
                mh = sel[h].argmax(-1) == bt["hops"][:, h]; steps.append(float(mh[ok].float().mean())); ok = ok & mh
                if h < 2:
                    ph = par[h].argmax(-1) == bt["R"][ar, bt["hops"][:, h], 2]; steps.append(float(ph[ok].float().mean())); ok = ok & ph
            vh = lv.argmax(-1) == bt["y"]; steps.append(float(vh[ok].float().mean())); ok = ok & vh
            steps.append(float(ok.float().mean())); steps.append(float(vh.float().mean())); acc.append(steps)
        m = np.mean(np.array(acc), 0)
        say(f"   {arm:9s} match1 {m[0]:.3f} | partner1 {m[1]:.3f} | match2 {m[2]:.3f} | partner2 {m[3]:.3f} | match3 {m[4]:.3f} | value {m[5]:.3f}"
            f" || product {np.prod(m[:6]):.3f} = all-steps-right {m[6]:.3f}; end-to-end answer {m[7]:.3f}")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    assert CJ.mode == "smoke" or not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    out = open(OUT if CJ.mode != "smoke" else OUT.replace(".txt", "_smoke.txt"), "w"); t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {CJ.mode} steps {CJ.STEPS}")
    parity_part(say); perhop_part(say)
    say(f"DONE in {time.time() - t0:.0f}s")
