"""hop_read.py -- TRAIN WIDE, READ NARROW (2026-10-21; Opus's design after the hop race; ledger "TRAIN WIDE, READ
NARROW"). The hop race found that a beam narrower than the live ambiguity cannot be TRAINED through. Here only the exact
(all-paths) arm trains; every cheap read is evaluated on its weights, beside the same read run on the TRUE weights (the
oracle) on the same eval sequences.

Changes from hop_race.py's exact arm (both touch the arm everything is read from, hence bar RN0):
  - the forward runs in LOG SPACE (logsumexp over kept events: no underflow, no 1e-12 renormalization);
  - the filter bit enters ONLY through its scalar gate (hop_race's E(bit) in the key vector was a second door).
Reads, each with and without the LEFTOVER BUCKET (the mass a read prunes is kept, compounded across hops, and spread
evenly over the answers at the end, instead of being renormalized away): exact, snap, beam2, beam4, mass (0.95, cap 16).
Checkpoints: .cache/hop_read_{regime}_s{seed}.pt. Data, eval seeds and training seeds = hop_race.py's.
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mode = "smoke" if "--smoke" in sys.argv else "full"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import hop_race as HR  # noqa: E402

STEPS = int(os.environ.get("HD_STEPS", "300" if mode == "smoke" else "4000"))
N_EVAL = 200 if mode == "smoke" else 2000
SEEDS = [0] if mode == "smoke" else [0, 1, 2]
READS = ["exact", "snap", "beam2", "beam4", "mass"]
NEG = -1e9


def read(logs, logM, logg, H, kind, bucket):
    """logs (B, L) hop-1 log weights (match from q1, before the gate); logM (B, L, L) log match t -> u; logg (B, L) log
    gates. -> (log a_final (B, L) normalized over kept events, log leftover (B,), reads per hop (B, HMAX))."""
    B, L = logs.shape
    lf = logs + logg
    left = torch.zeros(B)                                                     # pruned fraction so far (linear)
    finals, lefts, reads = [], [], []
    for h in range(HR.HMAX):
        la = lf - torch.logsumexp(lf, -1, keepdim=True)
        if kind != "exact":
            a = la.exp()
            if kind == "mass":
                srt, idx = a.sort(-1, descending=True)
                keep_s = ((srt.cumsum(-1) - srt) < HR.MASS) & (torch.arange(L)[None] < HR.MCAP)
                keep = torch.zeros_like(a, dtype=torch.bool).scatter(-1, idx, keep_s)
            else:
                K = 1 if kind == "snap" else int(kind[4:])
                keep = torch.zeros_like(a, dtype=torch.bool).scatter(-1, la.topk(min(K, L), -1).indices, True)
            kept = (a * keep).sum(-1)
            left = left + (1 - left) * (1 - kept).clamp(min=0)
            la = torch.where(keep, la, torch.full_like(la, NEG))
            la = la - torch.logsumexp(la, -1, keepdim=True)
            reads.append(keep.float().sum(-1))
        else:
            reads.append((la > -30).float().sum(-1))
        finals.append(la); lefts.append(left.clone())
        if h < HR.HMAX - 1:
            lf = torch.logsumexp(la[:, :, None] + logM, 1) + logg
    sel = H - 1; ar = torch.arange(B)
    lfin = torch.stack(finals, 1)[ar, sel]
    lleft = torch.stack(lefts, 1)[ar, sel] if bucket else torch.zeros(B)
    return lfin, lleft, torch.stack(reads, 1)


def answer(lfin, lleft, lans, bucket):
    """lans (B, L, NA) log p(answer | event). -> log p(answer) (B, NA)."""
    lp = torch.logsumexp(lfin[..., None] + lans, 1)
    if not bucket:
        return lp
    left = lleft.clamp(0, 1 - 1e-6)[:, None]
    return torch.log((1 - left) * lp.exp() + left / HR.NA)


class Exact(nn.Module):
    def __init__(self):
        super().__init__()
        self.ek, self.ef = nn.Embedding(HR.NKEY + 1, 64), nn.Embedding(2, 64)
        self.wq, self.wk = nn.Linear(64, 64), nn.Linear(64, 64)
        self.kappa_log, self.theta = nn.Parameter(torch.tensor(math.log(4.0))), nn.Parameter(torch.tensor(0.0))
        self.g_flag, self.g_bit = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(3))
        self.ent = nn.Parameter(torch.randn(HR.NA, 64))
        self.log_beta = nn.Parameter(torch.tensor(math.log(10.0)))

    def parts(self, bt):
        live = bt["live"] > 0
        ku = self.wk(self.ek(bt["key"]) + self.ef(bt["flag"]))                # the bit is NOT in the key: one door
        ql = self.wq(self.ek(bt["link"]))
        k = self.kappa_log.exp()
        logM = F.logsigmoid(k * (torch.einsum("btd,bud->btu", ql, ku) / 8.0 - self.theta))
        logs = F.logsigmoid(k * ((self.wq(self.ek(bt["q1"]))[:, None] * ku).sum(-1) / 8.0 - self.theta))
        logg = (self.g_flag[bt["flag"]] + self.g_bit[bt["bit"]]).masked_fill(~live, NEG)
        en = F.normalize(self.ent, dim=-1)
        lans = F.log_softmax(self.log_beta.exp() * en[bt["actor"]] @ en.T, -1)
        return logs, logM, logg, lans

    def forward(self, bt, kind="exact", bucket=False):
        logs, logM, logg, lans = self.parts(bt)
        lfin, lleft, rd = read(logs, logM, logg, bt["H"], kind, bucket)
        self.reads = rd
        return answer(lfin, lleft, lans, bucket)


def true_parts(bt, fbit):
    live = bt["live"] > 0
    logs = torch.where(bt["key"] == bt["q1"][:, None], 0.0, NEG)
    logM = torch.where(bt["link"][:, :, None] == bt["key"][:, None, :], 0.0, NEG)
    lg = torch.where(bt["flag"] == HR.FL_WEAK, math.log(HR.R_WEAK), 0.0)
    if fbit is not None:
        lr = math.log(fbit / (1 - fbit))
        lg = lg + torch.where(bt["bit"] == HR.B_YES, lr, -lr)
    lans = torch.where(F.one_hot(bt["actor"], HR.NA) > 0, 0.0, NEG)
    return logs, logM, lg.masked_fill(~live, NEG), lans


def run(seed, regime):
    rid = list(HR.REGIMES).index(regime); fbit = HR.REGIMES[regime][1]
    rng = np.random.default_rng(seed * 17 + rid); erng = np.random.default_rng(50_000 + seed * 7 + rid)
    ev = {H: HR.batch(N_EVAL, erng, regime, H) for H in (2, 3)}
    torch.manual_seed(seed); net = Exact()
    opt = torch.optim.AdamW(net.parameters(), lr=HR.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=HR.LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = HR.batch(HR.BATCH, rng, regime); loss = F.nll_loss(net(bt), bt["y"])
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    ck = f".cache/hop_read_{regime}_s{seed}{'_smoke' if mode == 'smoke' else ''}.pt"
    assert mode == "smoke" or not os.path.exists(ck), f"refusing to overwrite {ck}"
    torch.save(net.state_dict(), ck)
    net.eval(); res = {}
    with torch.no_grad():
        for H, full in ev.items():
            y = full["y"]; n = len(y)
            res[f"bacc{H}"], res[f"bll{H}"] = float(full["bacc"].mean()), float(full["blp"].mean())
            tp = true_parts(full, fbit)
            for kind in READS:
                for bucket in (False, True):
                    tag = f"{kind}{'_b' if bucket else ''}{H}"
                    lp, rd = [], []
                    for i in range(0, n, 250):
                        bt = {k_: v_[i:i + 250] for k_, v_ in full.items()}
                        lp.append(net(bt, kind, bucket)); rd.append(net.reads[:, :H].mean(-1))
                    lp = torch.cat(lp)
                    res[f"acc_{tag}"] = float((lp.argmax(-1) == y).float().mean())
                    res[f"ll_{tag}"] = float(-lp[torch.arange(n), y].mean())
                    res[f"reads_{tag}"] = float(torch.cat(rd).mean())
                lfin, lleft, rd = read(tp[0], tp[1], tp[2], full["H"], kind, False)
                lp = torch.logsumexp(lfin[..., None] + tp[3], 1)
                res[f"oacc_{kind}{H}"] = float((lp.argmax(-1) == y).float().mean())
        res["weak_ratio"] = float((net.g_flag[1] - net.g_flag[0]).exp())
        res["bit_logodds"] = float(net.g_bit[HR.B_YES] - net.g_bit[HR.B_NO])
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"smoke": "_smoke", "full": ""}[mode]
    assert mode == "smoke" or not os.path.exists(f".cache/hop_read{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/hop_read{tag}.txt", "w"); allres = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS}")
    RR = {}
    rgs = os.environ.get("HD_REGIMES", ",".join(HR.REGIMES)).split(",")
    for rg in rgs:
        rs = [run(s, rg) for s in SEEDS]
        A = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}; RR[rg] = A; allres.append(dict(regime=rg, seeds=rs, agg=A))
        for H in (2, 3):
            say(f"{rg:8s} H{H} Bayes {A[f'bacc{H}']:.3f}/{A[f'bll{H}']:.3f} | " + "  ".join(
                f"{k} {A[f'acc_{k}{H}']:.3f}/{A[f'll_{k}{H}']:.3f} (+b {A[f'acc_{k}_b{H}']:.3f}/{A[f'll_{k}_b{H}']:.3f}; oracle {A[f'oacc_{k}{H}']:.3f}; reads {A[f'reads_{k}{H}']:.2f})"
                for k in READS))
        say(f"{rg:8s} gates: weak {A['weak_ratio']:.3f} bit {A['bit_logodds']:.2f}  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(f".cache/hop_read{tag}.json", "w"), indent=1)
    if mode == "full":                                                       # THE PINNED BARS (ledger 2026-10-21)
        say("VERDICTS (3-seed means; nats)")
        for rg in rgs:
            A = RR[rg]
            r0 = all(A[f"ll_exact{H}"] <= A[f"bll{H}"] + 0.01 for H in (2, 3))
            say(f"(RN0) {rg}: " + ("THE RETRAINED EXACT ARM IS ON THE FLOOR" if r0 else "THE RETRAINED EXACT ARM IS OFF THE FLOOR -> RN1-RN3 VOID HERE")
                + "  (" + ", ".join(f"H{H} {A[f'll_exact{H}']:.3f} vs {A[f'bll{H}']:.3f}" for H in (2, 3)) + ")")
            r1 = all(abs(A[f"acc_{k}{H}"] - A[f"oacc_{k}{H}"]) <= 0.02 for k in READS[1:] for H in (2, 3))
            say(f"(RN1) {rg}: " + ("CHEAP READS REACH THEIR ORACLES" if r1 else "CHEAP READS DO NOT REACH THEIR ORACLES") + "  (" + ", ".join(
                f"{k} H{H} {A[f'acc_{k}{H}'] - A[f'oacc_{k}{H}']:+.3f}" for k in READS[1:] for H in (2, 3)) + ")")
            if rg in ("FILT", "UNF"):
                g = {H: A[f"acc_exact{H}"] - A[f"acc_snap{H}"] for H in (2, 3)}
                say(f"(RN2) {rg}: " + ("THE GARDEN PATH TAKES THE SNAP" if g[3] >= 0.10 and g[3] > g[2] else "THE SNAP SURVIVES THE GARDEN PATH")
                    + f"  (exact - snap acc H2 {g[2]:+.3f} H3 {g[3]:+.3f})")
            r3 = all(A[f"ll_{k}_b{H}"] <= A[f"ll_exact{H}"] + 0.05 for k in ("beam4", "mass") for H in (2, 3))
            r3a = all(abs(A[f"acc_{k}_b{H}"] - A[f"acc_{k}{H}"]) <= 0.005 for k in READS for H in (2, 3))
            say(f"(RN3) {rg}: " + ("THE BUCKET CLEARS THE BARS" if r3 else "THE BUCKET DOES NOT CLEAR THE BARS") + "; accuracy "
                + ("UNCHANGED" if r3a else "CHANGED") + "  (" + ", ".join(
                f"{k} H{H} ll {A[f'll_{k}{H}']:.3f} -> {A[f'll_{k}_b{H}']:.3f} vs exact {A[f'll_exact{H}']:.3f}" for k in ("beam4", "mass") for H in (2, 3)) + ")")
            if HR.REGIMES[rg][1] is not None:
                tb = 2 * math.log(HR.REGIMES[rg][1] / (1 - HR.REGIMES[rg][1]))
                say(f"(RN4) {rg}: " + ("THE ONE-DOOR BIT CALIBRATES" if abs(A["bit_logodds"] - tb) <= 0.5 else "THE ONE-DOOR BIT DOES NOT CALIBRATE")
                    + f"  (bit log-odds {A['bit_logodds']:.2f} vs {tb:.2f}; weak {A['weak_ratio']:.3f} vs {HR.R_WEAK})")
    say(f"DONE in {time.time() - t0:.0f}s")
