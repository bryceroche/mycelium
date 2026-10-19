"""complex_native_toy.py -- RUNG 0 OF THE COMPLEX-PLANES BRANCH (2026-10-19; Bryce's word; Opus: cleanup first, modReLU
as the control in the same build; bars pinned in docs/phase1_skeleton_spec.md BEFORE this runs). CPU only, never the card.

THE QUESTION: does a NATIVE complex transformer (the residual stream itself in C^M: no real path around the planes, so
the mandatory-road objection to toy v1's arm B does not apply) generalize better to held-out role-filler combinations
and multi-hop chains than a real transformer with the same parameter count?

THE TASK: toy v1's Part 1, imported (scripts/toys/binding_toy.py: make_batch / pools / run_part1, the same seeds), so the
data is byte-identical to the banked v1 run. A4 is rerun here as the REPRODUCTION GATE and the REAL TWIN.

ARMS (3 layers, 4 heads, readout at the query token; M = 48 complex planes = 96 reals, ~matched to A4's 94.5k params):
  A4       v1's arm, real D=64, SwiGLU ratio 1                       (the real twin; the gate)
  C_clean  native complex; MLP = CLEANUP MEMORY: a complex input projection, then per group of planes (4 groups x 12)
           a match against a FIXED random unit-phasor codebook (48 codes per group, never trained); keep the top 2
           codes, softmax over them, emit their LEARNED complex values; plus the residual
  C_mod    native complex; MLP = CLinear -> modReLU (bends the radius, keeps the phase) -> CLinear
  C_split  native complex; MLP = CLinear -> SiLU on the real and imaginary parts SEPARATELY -> CLinear
           (the part-wise control: it breaks the phase algebra; same linear structure as C_mod)
Every complex arm: complex embeddings (the additive fact bundle kept, as in v1); CNorm = divide by the RMS radius,
phase kept; attention scores Re(q^H k) / sqrt(2 * planes per head); complex values mixed by the real softmax weights;
readout = Linear([Re, Im]).
NOTE (stated in docs/complex_planes.md): Re(q^H k) is a real dot product over [Re, Im], and a complex linear layer is a
real linear layer with the block structure [[Wr, -Wi], [Wi, Wr]]. The arms differ from a real transformer by WEIGHT
STRUCTURE (rotation-equivariant maps) and by the NONLINEARITY: the inductive bias is the whole claim.
FAIRNESS: v1's protocol: the lr sweep {1e-3, 3e-3} per arm on the in-distribution val at f=0.5 seed 0; 6000 steps,
batch 128; seeds 0, 1, 2; f in {0.5, 0.2, 0.1} (v1's 0.05 / 0.02 rows hit the allowed_pairs floor and were identical).
Usage: .venv/bin/python3 scripts/toys/complex_native_toy.py [--smoke]  -> .cache/complex_native_toy{_smoke}.txt/.json
"""
import json, math, os, sys, time
smoke = "--smoke" in sys.argv
sys.argv = [sys.argv[0]] + (["--smoke"] if smoke else [])          # binding_toy parses its own args at import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import binding_toy as V1  # noqa: E402

M, H, G, K = 48, 4, 4, 48
ARMS = ["A4", "C_clean", "C_mod", "C_split"]
COVER = [0.5, 0.1] if smoke else [0.5, 0.2, 0.1]
SEEDS = V1.SEEDS


class CLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        s = 1 / math.sqrt(2 * i)
        self.wr = nn.Parameter(torch.randn(o, i) * s); self.wi = nn.Parameter(torch.randn(o, i) * s)
        self.br = nn.Parameter(torch.zeros(o)); self.bi = nn.Parameter(torch.zeros(o))

    def forward(self, xr, xi):
        return xr @ self.wr.T - xi @ self.wi.T + self.br, xr @ self.wi.T + xi @ self.wr.T + self.bi


class CNorm(nn.Module):
    def __init__(self, m):
        super().__init__(); self.g = nn.Parameter(torch.ones(m))

    def forward(self, xr, xi):
        r = torch.sqrt((xr * xr + xi * xi).mean(-1, keepdim=True) + 1e-6)
        return xr / r * self.g, xi / r * self.g


class CLayer(nn.Module):
    def __init__(self, mlp, gen):
        super().__init__()
        self.mlp = mlp; self.hp = M // H
        self.n1, self.n2 = CNorm(M), CNorm(M)
        self.q, self.k, self.v, self.o = CLinear(M, M), CLinear(M, M), CLinear(M, M), CLinear(M, M)
        self.w1 = CLinear(M, M)
        if mlp == "clean":
            th = torch.rand(G, K, M // G, generator=gen) * 2 * math.pi       # FIXED codebook, never trained
            self.register_buffer("kr", torch.cos(th)); self.register_buffer("ki", torch.sin(th))
            self.vr = nn.Parameter(torch.randn(G, K, M // G) * 0.02); self.vi = nn.Parameter(torch.randn(G, K, M // G) * 0.02)
        else:
            self.w2 = CLinear(M, M)
            if mlp == "mod":
                self.b = nn.Parameter(torch.zeros(M))

    def heads(self, x):
        B_, T_, _ = x.shape
        return x.view(B_, T_, H, self.hp).transpose(1, 2)

    def forward(self, xr, xi):
        hr, hi = self.n1(xr, xi)
        qr, qi = map(self.heads, self.q(hr, hi)); kr, ki = map(self.heads, self.k(hr, hi)); vr, vi = map(self.heads, self.v(hr, hi))
        a = F.softmax((qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)) / math.sqrt(2 * self.hp), -1)
        B_, T_, _ = xr.shape
        orr, oi = (a @ vr).transpose(1, 2).reshape(B_, T_, M), (a @ vi).transpose(1, 2).reshape(B_, T_, M)
        dr, di = self.o(orr, oi); xr, xi = xr + dr, xi + di
        hr, hi = self.n2(xr, xi)
        ur, ui = self.w1(hr, hi)
        if self.mlp == "clean":
            gr, gi = ur.view(B_, T_, G, M // G), ui.view(B_, T_, G, M // G)
            s = (torch.einsum("btgp,gkp->btgk", gr, self.kr) + torch.einsum("btgp,gkp->btgk", gi, self.ki)) / math.sqrt(M // G)
            top, idx = s.topk(2, -1)                                           # keep the top 2 codes per group
            w = F.softmax(top, -1)
            gidx = torch.arange(G).view(1, 1, G, 1).expand_as(idx)
            yr = (w[..., None] * self.vr[gidx, idx]).sum(-2).reshape(B_, T_, M)
            yi = (w[..., None] * self.vi[gidx, idx]).sum(-2).reshape(B_, T_, M)
        else:
            if self.mlp == "mod":
                mag = torch.sqrt(ur * ur + ui * ui + 1e-6); f = F.relu(mag + self.b) / mag
                ur, ui = ur * f, ui * f
            else:
                ur, ui = F.silu(ur), F.silu(ui)
            yr, yi = self.w2(ur, ui)
        return xr + yr, xi + yi


class CNet(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        gen = torch.Generator().manual_seed(1000 + seed)
        self.name = nn.Embedding(V1.N_NAMES + 1, 2 * M); self.rel = nn.Embedding(V1.N_ATTR + 1, 2 * M)
        self.val = nn.Embedding(V1.N_VAL, 2 * M); self.hop = nn.Embedding(4, 2 * M)
        self.layers = nn.ModuleList([CLayer(arm.split("_")[1], gen) for _ in range(V1.N_LAYERS)])
        self.nf = CNorm(M); self.out = nn.Linear(2 * M, V1.N_VAL)

    def forward(self, facts, query):
        isp = facts[..., 1] == V1.REL_PARTNER
        valemb = torch.where(isp[..., None], self.name(facts[..., 2]), self.val(facts[..., 2].clamp(max=V1.N_VAL - 1)))
        fx = self.name(facts[..., 0]) + self.rel(facts[..., 1]) + valemb
        qx = self.name(query[:, 0]) + self.rel(query[:, 1]) + self.hop(query[:, 2])
        x = torch.cat([fx, qx[:, None]], 1)
        xr, xi = x[..., :M], x[..., M:]
        for L in self.layers:
            xr, xi = L(xr, xi)
        xr, xi = self.nf(xr, xi)
        return self.out(torch.cat([xr[:, -1], xi[:, -1]], -1))


def run(arm, f, seed, lr):
    if arm == "A4":
        return V1.run_part1("A4", f, seed, lr)
    V1.Net = lambda a, s: CNet(arm, s)                                          # v1's exact training/eval loop
    try:
        return V1.run_part1(arm, f, seed, lr)
    finally:
        V1.Net = _V1Net


_V1Net = V1.Net

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = "_smoke" if smoke else ""
    out = open(f".cache/complex_native_toy{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    lrs = {}
    for arm in ARMS:
        sc = {lr: run(arm, 0.5, 0, lr)["val"] for lr in ([3e-3] if smoke else [1e-3, 3e-3])}
        lrs[arm] = max(sc, key=sc.get); say(f"lr sweep {arm}: {sc} -> {lrs[arm]}  [{time.time() - t0:.0f}s]")
    for f in COVER:
        for arm in ARMS:
            rs = [run(arm, f, s, lrs[arm]) for s in SEEDS]
            agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            sd = float(np.std([r["heldout"] for r in rs]))
            res.append(dict(arm=arm, f=f, lr=lrs[arm], seeds=rs, agg=agg))
            say(f"f={f:<4} {arm:<8} params {rs[0]['params']:>7}  val {agg['val']:.3f}  HELDOUT {agg['heldout']:.3f}±{sd:.3f}  "
                f"(hop1 {agg['heldout_hop1']:.3f} hop2 {agg['heldout_hop2']:.3f} hop3 {agg['heldout_hop3']:.3f})  [{time.time() - t0:.0f}s]")
    json.dump(res, open(f".cache/complex_native_toy{tag}.json", "w"), indent=1)
    # THE VERDICT (the bars as pinned in the ledger 2026-10-19)
    mean = lambda arm, k="heldout": float(np.mean([r["agg"][k] for r in res if r["arm"] == arm]))
    per_f = lambda arm: {r["f"]: r["agg"]["heldout"] for r in res if r["arm"] == arm}
    a4f = [r for r in res if r["arm"] == "A4" and r["f"] == 0.5][0]["agg"]["heldout"]
    gate = abs(a4f - 0.744) <= 0.02
    say(f"GATE: A4 held-out at f=0.5 {a4f:.3f} vs banked v1 0.744 -> {'PASS' if gate else 'VOID'}")
    for arm in ARMS:
        say(f"MEAN {arm:<8} held-out {mean(arm):.4f} | hop2 {mean(arm, 'heldout_hop2'):.4f} | hop3 {mean(arm, 'heldout_hop3'):.4f} | val {mean(arm, 'val'):.4f}")
    d = mean("C_clean") - mean("A4"); every = all(per_f("C_clean")[f] >= per_f("A4")[f] for f in COVER)
    v = ("VOID" if not gate else "NATIVE PHASE WINS" if (d >= 0.03 and every) else "DEAD" if d <= 0 else "WEAK (between the bars)")
    say(f"VERDICT C_clean vs A4 (pinned: WINS >= +0.03 mean held-out AND >= at every f; DEAD <= 0): {d:+.4f}, every f {every} -> {v}")
    best_c = max(mean("C_clean"), mean("C_mod"))
    say(f"ATTRIBUTION phase algebra (max(C_clean, C_mod) - C_split >= +0.02): {best_c - mean('C_split'):+.4f} -> "
        f"{'ATTRIBUTED' if best_c - mean('C_split') >= 0.02 else 'NOT ATTRIBUTED'}")
    say(f"ATTRIBUTION cleanup memory (C_clean - C_mod >= +0.02): {mean('C_clean') - mean('C_mod'):+.4f} -> "
        f"{'ATTRIBUTED' if mean('C_clean') - mean('C_mod') >= 0.02 else 'NOT ATTRIBUTED'}")
    say(f"DONE in {time.time() - t0:.0f}s")
