"""cj_readout_toy.py -- T3d, WHICH HALF OF BINDING BEATS THE CROSS JOIN: THE WRITE OR THE READ? (2026-10-24; Bryce's word;
ledger "T3d"). CPU. T3's world, recipe and splits UNCHANGED (imports cross_join_toy: 32 owners x 8 objects, depth 1-3
chains, coverage 50 / 25 / 12.5 %, unseen = uncovered (owner, object) pairs, W 128, snap). T3 raced arms that were
algebraic on BOTH sides (PHASOR_L: bind write + conjugate unbind + snap; unseen depth 3 = 1.000) against arms free-form on
BOTH sides (FLAT 0.414, COMPRESS 0.394). Here the WRITE is PHASOR_L's (learned phases, the same superposed register) in
every arm, and one READ is replaced by a learned network:
  PHASOR_L   the control (T3's arm, re-run)
  BIND_MLP   the FIELD read (partner / value out of the selected register) by an MLP over the register (Re, Im) -> code
  BIND_MLPD  the field read by a PER-DIMENSION MLP shared over the planes, given each plane's role phase (a learned read
             that respects the planes' independence)
  BIND_MLPQ  the QUERY composition: the key (current owner (x) object) composed by an MLP from the two codes, in place of
             the phase product; the match against the register stays the conjugate dot
The record selection (match) is the conjugate dot in every arm; the final snap onto the codebook is shared by all arms."""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_join_toy as T                                                  # mode / W / STEPS / SEEDS from its own argv+env

ARMS = os.environ.get("T3D_ARMS", "PHASOR_L,BIND_MLP,BIND_MLPD,BIND_MLPQ").split(",")
H = 256


def cplx(y, P):
    return torch.complex(y[..., :P], y[..., P:])


class ReadMemory(T.VSAMemory):
    def __init__(self, arm, g):
        a = T.Phasor(T.W // 2, g, learn=True); super().__init__([a], learn=True); self.arm = arm; P = a.P
        if arm == "BIND_MLP":
            self.rd = nn.ModuleDict({f: nn.Sequential(nn.Linear(2 * P, H), nn.GELU(), nn.Linear(H, H), nn.GELU(), nn.Linear(H, 2 * P))
                                     for f in ("par", "val")})
        if arm == "BIND_MLPD":
            self.rd = nn.ModuleDict({f: nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 2))
                                     for f in ("par", "val")})
        if arm == "BIND_MLPQ":
            self.qc = nn.Sequential(nn.Linear(4 * P, H), nn.GELU(), nn.Linear(H, H), nn.GELU(), nn.Linear(H, 2 * P))

    def read(self, regs, w, f, book):
        if self.arm not in ("BIND_MLP", "BIND_MLPD"):
            return super().read(regs, w, f, book)
        a, r = self.algs[0], regs[0]
        x = torch.einsum("bk,bkp->bp", w.to(r.dtype), r)
        if self.arm == "BIND_MLP":
            y = cplx(self.rd[f](torch.cat([x.real, x.imag], -1)), a.P)
        else:
            rl = a.rl[T.FIELDS[f]]
            z = torch.stack([x.real, x.imag, rl.cos().expand_as(x.real), rl.sin().expand_as(x.real)], -1)   # (B,P,4)
            y = self.rd[f](z); y = torch.complex(y[..., 0], y[..., 1])
        return a.sim(y, a.book(book))

    def match(self, regs, keys):
        return super().match(regs, keys)

    def compose(self, a, c, ob):
        if self.arm != "BIND_MLPQ":
            return a.bind(c, ob)
        y = self.qc(torch.cat([c.real, c.imag, ob.real, ob.imag], -1))
        k = cplx(y, a.P)
        return k / (k.abs() + 1e-6)                                         # unit phasors, as the product would be

    def run(self, bt, snap, train, pfail=0.0, rng=None):
        """VSAMemory.run with the key composition routed through compose() (verbatim otherwise; no data loss here)."""
        regs = self.write(bt["R"])
        if pfail > 0:
            regs = [T.corrupt(r, pfail, rng) for r in regs]
        cur = [a.code("e", bt["e0"]) for a in self.algs]; obj = [a.code("o", bt["o"]) for a in self.algs]
        sel_logits, par_logits = [], []
        for h in range(bt["D"]):
            sc = self.match(regs, [self.compose(a, c, ob) for a, c, ob in zip(self.algs, cur, obj)]).masked_fill(~bt["M"], -1e9)
            sel_logits.append(sc * self.beta.exp())
            w = F.softmax(sc * self.beta.exp(), -1) if train else F.one_hot(sc.argmax(-1), sc.shape[-1]).float()
            if h < bt["D"] - 1:
                lg = self.read(regs, w, "par", "e") * self.beta.exp(); par_logits.append(lg)
                if snap:
                    pw = F.softmax(lg, -1) if train else F.one_hot(lg.argmax(-1), T.NE).float()
                    cur = [torch.einsum("bn,np->bp", pw.to(a.book("e").dtype), a.book("e")) for a in self.algs]
                else:
                    cur = self.raw(regs, w, "par")
            else:
                return self.read(regs, w, "val", "v") * self.beta.exp(), sel_logits, par_logits


def make(arm, seed):
    g = torch.Generator().manual_seed(7000 + seed); torch.manual_seed(seed)
    return ReadMemory(arm, g)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    tag = "_smoke" if T.mode == "smoke" else ""
    OUT = f".cache/cj_readout{tag}.txt"
    assert T.mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {T.mode} steps {T.STEPS} W {T.W} seeds {T.SEEDS} arms {ARMS} (snap only; T3 world)")
    assert T.W == 128, "T3d is pinned at W 128"
    # the control's equivalence: ReadMemory('PHASOR_L') must be T3's PHASOR_L, bit for bit, at init and on one forward
    ref, mine = T.make("PHASOR_L", 0), make("PHASOR_L", 0)
    bt = T.batch(np.random.default_rng(5), 32, 3, T.coverage(0.125, 0), "unseen")
    assert torch.equal(ref.run(bt, True, False)[0], mine.run(bt, True, False)[0]), "control differs from T3's PHASOR_L"
    for arm in ARMS:
        for c in T.COVS:
            rs = []
            for s in T.SEEDS:
                cov = T.coverage(c, s); net = make(arm, s); T.train(net, cov, s, True); rs.append(T.evaluate(net, cov, s, True))
            a = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}; A[(arm, c)] = a
            allres.append(dict(arm=arm, cov=c, seeds=rs, agg=a))
            say(f"{arm:10s} cov {c:.3f} | " + " | ".join(f"{sp} " + " ".join(f"D{D} {a[f'{sp}_D{D}_p0.0']:.3f}" for D in T.DEPTHS)
                                                       for sp in ("seen", "unseen"))
                + f" | seeds unseen D3 {[round(r['unseen_D3_p0.0'], 3) for r in rs]}  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if T.mode == "full" and all(x in ARMS for x in ("PHASOR_L", "BIND_MLP", "BIND_MLPQ")):   # PINNED (ledger 2026-10-24, T3d)
        g = lambda arm, k, c=0.125: A[(arm, c)][k]
        say("VERDICTS (3-seed means; unseen pairs, depth 3, cov 0.125, snap)")
        pl = g("PHASOR_L", "unseen_D3_p0.0")
        say(f"CONTROL: PHASOR_L unseen D3 {pl:.3f} (T3 banked 1.000) -> {'OK' if pl >= 0.97 else 'VOID (control does not reproduce)'}")
        for arm, nm in (("BIND_MLP", "R1 THE FIELD READ"), ("BIND_MLPQ", "R2 THE QUERY COMPOSITION")):
            seen = g(arm, "seen_D3_p0.0"); gap = pl - g(arm, "unseen_D3_p0.0")
            v = ("VOID (undertrained: seen D3 < 0.90)" if seen < 0.90 or pl < 0.97 else
                 "SUPPORTED (the learned read re-admits the cross join)" if gap >= 0.20 else
                 "REFUTED (the learned read is not the lever)" if gap <= 0.05 else "BETWEEN")
            say(f"{nm}: {v}  (PHASOR_L - {arm} = {gap:+.3f}; {arm} unseen D3 {g(arm, 'unseen_D3_p0.0'):.3f}, seen D3 {seen:.3f})")
        if "BIND_MLPD" in ARMS:
            say(f"READ (no bar): BIND_MLPD unseen D3 {g('BIND_MLPD', 'unseen_D3_p0.0'):.3f}, seen D3 {g('BIND_MLPD', 'seen_D3_p0.0'):.3f}")
        say("COVERAGE CURVES (unseen D1:D3 at cov 0.5/0.25/0.125): " + ", ".join(
            f"{x} " + "/".join(f"{g(x, 'unseen_D1_p0.0', c):.3f}:{g(x, 'unseen_D3_p0.0', c):.3f}" for c in T.COVS) for x in ARMS))
    say(f"DONE in {time.time() - t0:.0f}s")
