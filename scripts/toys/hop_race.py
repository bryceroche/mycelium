"""hop_race.py -- THE MULTI-HOP BEAM RACE (2026-10-21; Bryce's word; Opus's three additions; ledger "THE MULTI-HOP
BEAM RACE"). The read race (one hop) found: marginalize over a beam that covers the live mass. Across HOPS the questions
change: does ambiguity compound, can a later hop rule out an earlier choice (the garden path), and what does a
fallible filter cost?

TASK. A query key q1 and a hop count H in {2, 3}. Events are (key, flag, filter bit, link, actor). Layer 1 = the n events
with key q1; layers 2..H = BR = 3 keys each, n events per key. A layer-h event (h < H) LINKS to one of layer h+1's keys,
or (prob D_DEAD) to a DEAD key no event carries: a garden path, invisible until the next hop. 'Follow H links from q1;
who acted at the end?' Each event's flag is CLEAR or WEAK (prior weight 1 / R_WEAK). The TRUE path is drawn from
p(path) ~ prod prior(flag) over VALID paths (those reaching layer H); then, in the FALLIBLE regimes, a filter bit per
event: 'yes' w.p. F on the true path's events, w.p. 1 - F on every other event. Hence the exact posterior over paths is
prod prior(flag) * L(bit) over valid paths, L(yes) / L(no) = (F / (1 - F))^2, and the BAYES accuracy and log loss are
computed exactly per sequence by a forward pass before training.

REGIMES (n per key, from the candidate census 10-21): FILT n ~ U{1..4} (behind owner + type + event); UNF n ~ U{3..10}
(no filter); FALL F 0.9 / 0.7 (UNF's n plus the fallible filter bit).

THE MACHINE (learned): event vector h_u = E(key) + E(flag) + E(bit); match M(t -> u) = sigmoid(kappa * ((Wk h_u) . (Wq
E(link_t))) / 8 - theta)), hop 1 from Wq E(q1); a GATE per flag and per bit (log-weights, learned scalars); the answer
read = beta * cos(ent[actor_u], ent) (learned ent, beta). Forward: f_1(u) = M(q1 -> u) g_u; f_{h+1}(u) = sum_t f_h(t)
M(t -> u) g_u; normalized each hop.
ARMS (each trained on its own answer NLL; lr 3e-3; mixed H 2/3 in training):
  exact      every path (the forward pass above), marginalized over final answers
  collapse   the accidental build (Opus): each intermediate hop's alpha blends the LINK vectors into ONE query for the next
             hop (mixtures, not paths); the final hop marginalizes
  snap       beam 1, straight-through at every hop
  beam2/4    keep the top K events of f_h at every hop (paths merge at an event: the event is the sufficient state)
  mass       keep events in order of f_h until 0.95 of its mass (cap 16); reports reads per hop
  hard_exact / hard_mass   (FALL only) a HARD filter: events whose bit says 'no' are dropped (gate 0)
Eval per regime and H: accuracy, log loss, Bayes floors, reads per hop, learned weak-flag ratio (true R_WEAK) and the bit
log-odds (true 2 log(F / (1 - F))).
"""
import json, math, os, sys, time
mode = "smoke" if "--smoke" in sys.argv else ("bayes" if "--bayes" in sys.argv else "full")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

NKEY, NA, BR, HMAX = 160, 16, 3, 3
R_WEAK, D_DEAD = float(os.environ.get("HR_RWEAK", "0.3")), float(os.environ.get("HR_DEAD", "0.4"))
REGIMES = {"FILT": ((1, 4), None), "UNF": ((3, 10), None), "FALL0.9": ((3, 10), 0.9), "FALL0.7": ((3, 10), 0.7)}
STEPS = int(os.environ.get("HR_STEPS", "300" if mode == "smoke" else "4000"))
BATCH, N_EVAL = 64, (200 if mode == "smoke" else 2000)
SEEDS = [0] if mode == "smoke" else [0, 1, 2]
BASE_ARMS = os.environ.get("HR_ARMS", "exact,collapse,snap,beam2,beam4,mass").split(",")
HARD_ARMS = ["hard_exact", "hard_mass"]
LR, MASS, MCAP = 3e-3, 0.95, 16
FL_CLEAR, FL_WEAK, B_NONE, B_YES, B_NO = 0, 1, 0, 1, 2


def gen(rng, nr, fbit, H):
    """one sequence -> dict of event arrays + the exact Bayes answer posterior + the sampled truth."""
    while True:
        keys = rng.choice(NKEY, size=1 + BR * (H - 1) + 3 * 30, replace=False)
        q1, layer_keys, dead = keys[0], [[keys[0]]], list(keys[1 + BR * (H - 1):])
        for h in range(1, H):
            layer_keys.append(list(keys[1 + BR * (h - 1): 1 + BR * h]))
        ev = []                                                              # (layer, key, flag, link)
        for h, ks in enumerate(layer_keys):
            for k in ks:
                for _ in range(rng.integers(nr[0], nr[1] + 1)):
                    if h == H - 1:
                        link = -1
                    elif rng.random() < D_DEAD:
                        link = dead.pop()
                    else:
                        link = layer_keys[h + 1][rng.integers(BR)]
                    ev.append((h, k, int(rng.random() < 0.5), link))
        L = len(ev)
        lay = np.array([e[0] for e in ev]); key = np.array([e[1] for e in ev]); flag = np.array([e[2] for e in ev])
        link = np.array([e[3] for e in ev])
        prior = np.where(flag == FL_WEAK, R_WEAK, 1.0)
        T = (link[:, None] == key[None, :]).astype(float)                    # t -> u
        # backward: b(t) = total prior weight of valid completions from t
        b = np.where(lay == H - 1, 1.0, 0.0)
        for _ in range(H - 1):
            b = np.where(lay == H - 1, 1.0, T @ (prior * b))
        start = (key == q1).astype(float)
        if (start * prior * b).sum() > 0:
            break
    # sample the true path
    path, w = [], start * prior * b
    for h in range(H):
        u = rng.choice(L, p=w / w.sum()); path.append(u)
        if h < H - 1:
            w = T[u] * prior * b
    actor = rng.integers(NA, size=L)
    bit = np.full(L, B_NONE)
    lik = np.ones(L)
    if fbit is not None:
        on = np.zeros(L, bool); on[path] = True
        yes = np.where(on, rng.random(L) < fbit, rng.random(L) < 1 - fbit)
        bit = np.where(yes, B_YES, B_NO)
        lik = np.where(yes, fbit / (1 - fbit), (1 - fbit) / fbit)
    # exact Bayes answer posterior: forward with the true weights
    f = start * prior * lik
    for _ in range(H - 1):
        f = (f @ T) * prior * lik
    f = f * (lay == H - 1)
    pa = np.zeros(NA)
    np.add.at(pa, actor, f)
    pa /= pa.sum()
    y = actor[path[-1]]
    return dict(L=L, q1=q1, key=key, flag=flag, bit=bit, link=link, actor=actor, H=H, y=y,
                blp=-math.log(max(pa[y], 1e-12)), bacc=float(pa.argmax() == y))


def batch(B, rng, regime, H=None):
    nr, fbit = REGIMES[regime]
    seqs = [gen(rng, nr, fbit, H if H is not None else int(rng.integers(2, HMAX + 1))) for _ in range(B)]
    Lm = max(s["L"] for s in seqs)
    out = {k: np.zeros((B, Lm), np.int64) for k in ("key", "flag", "bit", "link", "actor")}
    out["live"] = np.zeros((B, Lm), np.float32)
    for i, s in enumerate(seqs):
        for k in ("key", "flag", "bit", "link", "actor"):
            out[k][i, :s["L"]] = s[k]
        out["link"][i, :s["L"]] = np.where(s["link"] < 0, NKEY, s["link"])  # NKEY = 'no link' token
        out["live"][i, :s["L"]] = 1
    t = {k: torch.from_numpy(v) for k, v in out.items()}
    for k in ("q1", "H", "y"):
        t[k] = torch.tensor([s[k] for s in seqs])
    t["blp"] = torch.tensor([s["blp"] for s in seqs]).float(); t["bacc"] = torch.tensor([s["bacc"] for s in seqs]).float()
    return t


class Hopper(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.kind = kind
        self.ek, self.ef, self.eb = nn.Embedding(NKEY + 1, 64), nn.Embedding(2, 64), nn.Embedding(3, 64)
        self.wq, self.wk = nn.Linear(64, 64), nn.Linear(64, 64)
        self.kappa_log, self.theta = nn.Parameter(torch.tensor(math.log(4.0))), nn.Parameter(torch.tensor(0.0))
        self.g_flag, self.g_bit = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(3))
        self.ent = nn.Parameter(torch.randn(NA, 64))
        self.log_beta = nn.Parameter(torch.tensor(math.log(10.0)))

    def match(self, qv, ku):                                                  # qv (B, ..., 64), ku (B, L, 64)
        c = torch.einsum("bsd,bld->bsl", qv, ku) / 8.0
        return torch.sigmoid(self.kappa_log.exp() * (c - self.theta))

    def forward(self, bt):
        B, L = bt["key"].shape; live = bt["live"]; H = bt["H"]
        hu = self.ek(bt["key"]) + self.ef(bt["flag"]) + self.eb(bt["bit"])
        ku = self.wk(hu); ql = self.wq(self.ek(bt["link"]))                   # (B, L, 64) each event's link as a query
        lg = self.g_flag[bt["flag"]] + self.g_bit[bt["bit"]]
        if self.kind.startswith("hard"):
            lg = lg.masked_fill(bt["bit"] == B_NO, -30.0)
        g = lg.exp() * live
        Mtu = self.match(ql, ku)                                              # (B, L, L)
        f = self.match(self.wq(self.ek(bt["q1"]))[:, None], ku)[:, 0] * g    # hop 1
        reads, finals = [], []
        for h in range(HMAX):
            a = f / (f.sum(-1, keepdim=True) + 1e-12)
            k = self.kind.replace("hard_", "")
            if k == "snap":
                a = F.one_hot(a.argmax(-1), L).float() + a - a.detach()
            elif k.startswith("beam"):
                top = a.topk(min(int(k[4:]), L), -1)
                a = torch.zeros_like(a).scatter(-1, top.indices, top.values); a = a / (a.sum(-1, keepdim=True) + 1e-12)
            elif k == "mass":
                srt, idx = a.sort(-1, descending=True)
                keep = ((srt.cumsum(-1) - srt) < MASS) & (torch.arange(L)[None] < MCAP)
                a = torch.zeros_like(a).scatter(-1, idx, srt * keep); a = a / (a.sum(-1, keepdim=True) + 1e-12)
            reads.append((a > 0).float().sum(-1) if k != "snap" else torch.ones(B))
            finals.append(a)
            if h == HMAX - 1:
                break
            if k == "collapse":                                               # mixtures, not paths
                qv = torch.einsum("bl,bld->bd", a, ql)[:, None]
                f = self.match(qv, ku)[:, 0] * g
            else:
                f = torch.einsum("bt,btu->bu", a, Mtu) * g
        sel = H - 1
        A = torch.stack(finals, 1)[torch.arange(B), sel]                     # (B, L) the final hop's weights
        self.reads = torch.stack(reads, 1)                                    # (B, HMAX)
        en = F.normalize(self.ent, dim=-1)
        lgt = self.log_beta.exp() * en[bt["actor"]] @ en.T                   # (B, L, NA)
        if self.kind.replace("hard_", "") == "snap":
            return F.log_softmax((A[..., None] * lgt).sum(1), -1)
        return torch.logsumexp(torch.log(A.clamp(min=1e-12))[..., None] + F.log_softmax(lgt, -1), 1)


def run(kind, seed, regime):
    rid = list(REGIMES).index(regime)
    rng = np.random.default_rng(seed * 17 + rid); erng = np.random.default_rng(50_000 + seed * 7 + rid)
    ev = {H: batch(N_EVAL, erng, regime, H) for H in (2, 3)}
    torch.manual_seed(seed); net = Hopper(kind)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = batch(BATCH, rng, regime); loss = F.nll_loss(net(bt), bt["y"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    net.eval(); res = {}
    with torch.no_grad():
        for H, full in ev.items():
            lp, rd = [], []
            for i in range(0, N_EVAL, 250):
                bt = {k_: v_[i:i + 250] for k_, v_ in full.items()}; lp.append(net(bt)); rd.append(net.reads[:, :H].mean(-1))
            lp = torch.cat(lp); y = full["y"]
            res[f"acc{H}"] = float((lp.argmax(-1) == y).float().mean())
            res[f"ll{H}"] = float(-lp[torch.arange(len(y)), y].mean())
            res[f"bacc{H}"], res[f"bll{H}"] = float(full["bacc"].mean()), float(full["blp"].mean())
            res[f"reads{H}"] = float(torch.cat(rd).mean())
        res["weak_ratio"] = float((net.g_flag[1] - net.g_flag[0]).exp())
        res["bit_logodds"] = float(net.g_bit[B_YES] - net.g_bit[B_NO])
    res["final_loss"] = float(loss.detach())
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    if mode == "bayes":                                                      # the design check: data properties only
        rng = np.random.default_rng(1)
        for rg in REGIMES:
            for H in (2, 3):
                bt = batch(1000, rng, rg, H)
                print(f"{rg:8s} H{H}: Bayes acc {float(bt['bacc'].mean()):.3f} ll {float(bt['blp'].mean()):.3f}  "
                      f"events {float(bt['live'].sum(-1).mean()):.1f}")
        sys.exit(0)
    tag = {"smoke": "_smoke", "full": ""}[mode] + os.environ.get("HR_TAG", "")
    assert mode == "smoke" or not os.path.exists(f".cache/hop_race{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/hop_race{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS} lr {LR} R_WEAK {R_WEAK} D_DEAD {D_DEAD}")
    RR = {}
    for rg in os.environ.get("HR_REGIMES", ",".join(REGIMES)).split(","):
        for kind in BASE_ARMS + (HARD_ARMS if REGIMES[rg][1] is not None else []):
            rs = [run(kind, s, rg) for s in SEEDS]
            A = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}; RR[(rg, kind)] = A
            res.append(dict(regime=rg, arm=kind, seeds=rs, agg=A))
            say(f"{rg:8s} {kind:<10} " + "  ".join(
                f"H{H} acc {A[f'acc{H}']:.3f} ll {A[f'll{H}']:.3f} (Bayes {A[f'bacc{H}']:.3f}/{A[f'bll{H}']:.3f}) reads {A[f'reads{H}']:.2f}"
                for H in (2, 3)) + f" | weak {A['weak_ratio']:.3f} bit {A['bit_logodds']:.2f}  [{time.time() - t0:.0f}s]")
    json.dump(res, open(f".cache/hop_race{tag}.json", "w"), indent=1)
    if mode == "full":                                                       # THE PINNED BARS (ledger 2026-10-21)
        say("VERDICTS (3-seed means; log loss in nats)")
        rgs = sorted({k[0] for k in RR}, key=list(REGIMES).index)
        g = lambda rg, a, k: RR[(rg, a)][k]
        for rg in rgs:
            p1 = all(g(rg, "exact", f"ll{H}") <= g(rg, "exact", f"bll{H}") + 0.05 for H in (2, 3))
            say(f"(P1) {rg}: " + ("PATHS ARE CALIBRATED" if p1 else "PATHS ARE NOT CALIBRATED") + "  (" + ", ".join(
                f"H{H} {g(rg, 'exact', f'll{H}'):.3f} vs Bayes {g(rg, 'exact', f'bll{H}'):.3f}" for H in (2, 3)) + ")")
            gap = {H: g(rg, "collapse", f"ll{H}") - g(rg, "exact", f"ll{H}") for H in (2, 3)}
            p2 = gap[3] >= 0.10 and gap[3] >= gap[2]
            say(f"(P2) {rg}: " + ("COLLAPSE LOSES AND WORSENS WITH HOPS" if p2 else "COLLAPSE DOES NOT LOSE AS PINNED")
                + f"  (ll gap H2 {gap[2]:+.3f} H3 {gap[3]:+.3f}; acc H3 {g(rg, 'collapse', 'acc3'):.3f} vs {g(rg, 'exact', 'acc3'):.3f})")
            if rg in ("FILT", "UNF"):
                sg = {H: g(rg, "exact", f"acc{H}") - g(rg, "snap", f"acc{H}") for H in (2, 3)}
                say(f"(P3) {rg}: " + ("THE GARDEN PATH TAKES THE SNAP" if sg[3] >= 0.10 and sg[3] > sg[2] else "THE SNAP SURVIVES THE GARDEN PATH")
                    + f"  (snap acc gap H2 {sg[2]:+.3f} H3 {sg[3]:+.3f})")
            ok4 = all(abs(g(rg, "beam4", f"acc{H}") - g(rg, "exact", f"acc{H}")) <= 0.02
                      and g(rg, "beam4", f"ll{H}") <= g(rg, "exact", f"ll{H}") + 0.05 for H in (2, 3))
            say(f"(P4) {rg}: " + ("BEAM-4 SUFFICES" if ok4 else "BEAM-4 DOES NOT SUFFICE") + ("" if rg == "FILT" else " [reported, not barred]")
                + "  (" + ", ".join(f"H{H} {g(rg, 'beam4', f'acc{H}'):.3f}/{g(rg, 'beam4', f'll{H}'):.3f} vs {g(rg, 'exact', f'acc{H}'):.3f}/{g(rg, 'exact', f'll{H}'):.3f}" for H in (2, 3)) + ")")
            ok5 = all(abs(g(rg, "mass", f"acc{H}") - g(rg, "exact", f"acc{H}")) <= 0.02
                      and g(rg, "mass", f"ll{H}") <= g(rg, "exact", f"ll{H}") + 0.05 for H in (2, 3))
            say(f"(P5) {rg}: " + ("THE MASS BEAM TRACKS EXACT" if ok5 else "THE MASS BEAM DOES NOT TRACK EXACT") + "  (" + ", ".join(
                f"H{H} {g(rg, 'mass', f'acc{H}'):.3f}/{g(rg, 'mass', f'll{H}'):.3f}, reads/hop {g(rg, 'mass', f'reads{H}'):.2f}" for H in (2, 3)) + ")")
            if REGIMES[rg][1] is not None:
                fb = REGIMES[rg][1]
                d = g(rg, "mass", "acc3") - g(rg, "hard_mass", "acc3")
                say(f"(P6) {rg}: " + ("SOFT FILTER BEATS HARD" if d >= 0.10 else "SOFT FILTER DOES NOT BEAT HARD BY 0.10")
                    + ("" if fb == 0.7 else " [reported, not barred]")
                    + f"  (H3 mass soft {g(rg, 'mass', 'acc3'):.3f} vs hard {g(rg, 'hard_mass', 'acc3'):.3f}; hard_exact {g(rg, 'hard_exact', 'acc3'):.3f}; arithmetic ceiling F^3 = {fb ** 3:.3f})")
            ok7 = abs(g(rg, "exact", "weak_ratio") - R_WEAK) <= 0.10
            line = f"(P7) {rg}: " + ("THE FLAG GATE CALIBRATES" if ok7 else "THE FLAG GATE DOES NOT CALIBRATE") + f"  (exact weak ratio {g(rg, 'exact', 'weak_ratio'):.3f} vs {R_WEAK}"
            if REGIMES[rg][1] is not None:
                tb = 2 * math.log(REGIMES[rg][1] / (1 - REGIMES[rg][1]))
                line += f"; bit log-odds {g(rg, 'exact', 'bit_logodds'):.2f} vs {tb:.2f} -> " + ("BIT CALIBRATES" if abs(g(rg, "exact", "bit_logodds") - tb) <= 0.5 else "BIT DOES NOT CALIBRATE")
            say(line + "; other arms: " + ", ".join(f"{a} {g(rg, a, 'weak_ratio'):.2f}" for a in BASE_ARMS if a != "exact") + ")")
    say(f"DONE in {time.time() - t0:.0f}s")
