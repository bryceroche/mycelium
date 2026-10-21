"""router_race.py -- THE ROUTER RACE: WHAT CARRIES THE FIRST/LAST JUDGMENT PAST THE TRAINED LENGTH?
(2026-10-21; Bryce's word; Opus's three tweaks taken.) CPU only. The (who, when) hybrid (who_when_rung.HYB: clock address +
registers + snap) with ONLY the router's position signal swapped. The decomposition showed: one sale is found perfectly at
every distance; every miss is a two-sale ORDER judgment that breaks when either candidate sits beyond the trained 64.
ROUTERS (all: content score q.k WITHOUT RoPE, plus a position bias per 'which' (first/last), except R_rope):
  R_rope    the banked router (RoPE on q/k): the baseline
  R_none    no position at all: content only (should be perfect on one sale, ~chance on two)
  R_alibi   bias = s_which * t (learned signed slope on RAW t). NOTE: ALiBi's s*(N - t) and the raw-t scalar differ by
            a per-query constant that cancels in the softmax: they are ONE arm. Nothing is divided by N.
  R_slowL   bias = w_which . [cos, sin](Omega t), 16 LEARNED slow frequencies (init periods 256..2048; the rung's band)
  R_gear    THE T6 GEAR CLOCK: bias = w_which . [cos, sin](2 pi t / P), P in {3, 6, 36, 216, 1296} FROZEN (rotor_clock's
            breath hand 6 + parity 3, extended as an odometer: each wheel turns once per turn of the one below;
            frequencies frozen per the rotor_clock law; only the readout weights learn)
RACE 2 (2026-10-21 ~09:55, Bryce's word; RR_RACE=2): the gear clock's order horizon (69-92) stopped short because the learned
readout mixed the 216 wheel and the fast wheels into the order judgment. The rotor-clock design gives each wheel a job:
  R_gearS   ORDER READS THE SLOWEST GEAR ONLY: bias = w_which . [cos, sin](2 pi t / 1296), frozen (the fast wheels keep
            their job in the read's address clock, untouched)
  R_alibiC  THE BOUNDED-RAMP RIVAL: bias = s_which * tanh(t / tau), tau = 1296 / 2 pi FIXED (the slow wheel's own scale):
            is boundedness alone enough, or does the clock's shape matter?
RACE 3 (Opus's structural fix; RR_RACE=3): ORDER AS A COMPARISON AMONG CANDIDATES, never a bias added to content.
  R_2stage  stage 1 CONTENT GATE g_t = sigmoid(kappa (c_t - theta)) (c = the content score; kappa, theta learned);
            stage 2 THE CLOCK COMPARES: alpha_t ~ g_t * prod_{u: phi_u later than phi_t} (1 - g_u) ('last': no candidate
            with a later slow-wheel phase; 'first' mirrored), phi = angle of the 1296 wheel, FROZEN, nothing learned in
            stage 2. An additive form (log g + gamma phi) was rejected on paper: a non-candidate 240 ticks later outbids
            any finite gate. The ANGLE is read on [0, 2 pi) (Bryce: 'track distance around the circle'; the smoke's
            atan2 on (-pi, pi] wrapped at HALF a turn): horizon = one full turn, 1296 ticks.
  R_2st_idx  the same product with order from the RAW INDEX (no clock): the honest rival
  R_2st_odo  THE ODOMETER (Bryce's pass wheel): order key = sector of a 7776 wheel (60 deg per full turn of the 1296
            wheel, rotor_clock's pass-wheel rule) x 2 pi + the 1296 wheel's angle: horizon 7776, frozen, nothing learned
  R_2st_fz   R_2stage on FUZZY content: Gaussian noise (sigma FZ_SIGMA per dim) on every event's router embedding, train
            and eval, so the gate must be soft (Opus: a leak eps per later event multiplies the anchor by (1-eps)^k)
  R_2st_fzv  the same fuzzy content, the dead zone on the VETO side only (reported, no bar)
  R_2st_fzs  the same fuzzy content, the VETO side SNAPPED: a gate below 0.5 vetoes nothing (snapped to 0; straight-
            through), above 0.5 its soft veto stands; the same dead zone on the OWN gate (the smoke found a 2nd leak: N x eps of own
            weight dilutes the anchor in the normalization). Two harder snaps died at init.
EVAL: N 16/32/64 (in range), 128, 256 (4000 sequences each at 128 and 256: ~50 gap-1-2 two-sale cases). Router anchor accuracy + answer accuracy;
two-sale splits by GAP between the sales (1-2 / 3-8 / 9-32 / 33+) and by query-anchor distance (<= 64 / > 64).
ORDER HORIZON (Opus's turning point, made robust to the fast gears' wiggle; the smoke read the raw turning point as 0 for
R_gear): for the band arms, the learned bias b_which(t), t = 0..4000; the horizon = the first t2 at which some t1 <= t2 - 3
is ranked WRONGLY (for 'last' b must rise: b(t2) <= max b(t <= t2-3) is a failure; 'first' mirrored). ALiBi: linear, none.
Usage: .venv/bin/python3 scripts/toys/router_race.py [--feas | --smoke] -> .cache/router_race{tag}.*
"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mode = "feas" if "--feas" in sys.argv else "smoke" if "--smoke" in sys.argv else "full"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import who_when_rung as W  # noqa: E402

STEPS = 300 if mode == "smoke" else int(os.environ.get("RR_STEPS", "4000"))
SEEDS = [0] if mode != "full" else [0, 1, 2]
EVAL_N, N_BIG = (16, 32, 64, 128, 256), (300 if mode == "smoke" else 4000)
RACE = int(os.environ.get("RR_RACE", "1"))
ALL = {1: ["R_rope", "R_none", "R_alibi", "R_slowL", "R_gear"], 2: ["R_gearS", "R_alibiC"],
       3: ["R_2stage", "R_2st_idx", "R_2st_odo", "R_2st_fz", "R_2st_fzv", "R_2st_fzs"]}[RACE]
FZ_SIGMA = float(os.environ.get("FZ_SIGMA", "1.0"))
if RACE == 3: EVAL_N = EVAL_N + (1024, 1536)                                    # the horizon reads (one turn = 1296)
ARMS = os.environ.get("RR_ARMS", ",".join(ALL)).split(",")
GEARS = (3.0, 6.0, 36.0, 216.0, 1296.0)
IN_RANGE_BANKED = 0.995                                                       # HYB in-range mean, .cache/who_when_rung.txt


class Router(W.Hybrid):
    def __init__(self, arm, seed):
        super().__init__("HYB", seed); self.router = arm[2:].lower(); self.arm = arm
        if arm == "R_gearS": self.router = "gear"
        if arm.startswith("R_2st"):
            self.router = "2stage"
            self.kappa_log, self.theta = nn.Parameter(torch.tensor(math.log(4.0))), nn.Parameter(torch.tensor(0.0))
        if arm in ("R_alibi", "R_alibiC"): self.slope = nn.Parameter(torch.zeros(2))
        if arm == "R_gearS":
            self.register_buffer("om_r", 2 * math.pi / torch.tensor([1296.0])); self.wr = nn.Parameter(torch.zeros(2, 2))
        if arm == "R_slowL":
            self.om_r = nn.Parameter(2 * math.pi / torch.exp(torch.linspace(math.log(256.0), math.log(2048.0), 16)))
            self.wr = nn.Parameter(torch.zeros(2, 32))
        if arm == "R_gear":
            self.register_buffer("om_r", 2 * math.pi / torch.tensor(GEARS)); self.wr = nn.Parameter(torch.zeros(2, 2 * len(GEARS)))

    def feats(self, t):                                                      # t: (..., L) float
        a = t[..., None] * self.om_r
        return torch.cat([torch.cos(a), torch.sin(a)], -1)

    def bias(self, which, t):                                                # which (B,), t (L,) or (T,)
        if self.router == "none": return torch.zeros(len(which), len(t))
        if self.router == "alibi": return self.slope[which][:, None] * t[None] / 64.0
        if self.router == "alibic": return self.slope[which][:, None] * torch.tanh(t[None] / (1296.0 / (2 * math.pi)))
        return self.wr[which] @ self.feats(t).T                            # (B, len(t))

    def two_stage(self, bt, c, live, tt):
        which = bt["x2"][torch.arange(len(bt["ns"])), bt["ns"]] - W.V_WH        # 0 first, 1 last
        g = torch.sigmoid(self.kappa_log.exp() * (c - self.theta)) * live     # stage 1: content candidates
        self.gate = g
        # SNAP the VETO side only (R_2st_fzs; the leak lives in (1 - g_u)), straight-through IN LOG SPACE: forward = the hard
        # factor, backward = the soft gradient (a snapped own gate, and a snap through the clamp, both died in the smoke)
        if self.arm == "R_2st_idx":
            key = tt
        else:
            key = torch.remainder(2 * math.pi * tt / 1296.0, 2 * math.pi)     # the 1296 wheel's angle on [0, 2 pi)
            if self.arm == "R_2st_odo":                                       # + the pass wheel's sector (60 deg per turn)
                sector = torch.floor(torch.remainder(2 * math.pi * tt / 7776.0, 2 * math.pi) / (math.pi / 3) + 1e-9)
                key = sector * 2 * math.pi + key
        later = (key[None, :] > key[:, None]).float()                         # later[t, u]: u's key is later than t's
        l1g = torch.log((1 - g).clamp(min=1e-6))
        if self.arm in ("R_2st_fzs", "R_2st_fzv"):
            hard = torch.where(g > 0.5, l1g, torch.zeros_like(l1g))           # DEAD ZONE: a sub-threshold gate vetoes nothing
            l1g = l1g + (hard - l1g).detach()                                 # (a full {0,1} snap died at init: ~half the gates
                                                                              # start > 0.5 and veto every candidate)
        veto = torch.where(which[:, None] == 1, l1g @ later.T, l1g @ later)   # sum of log(1-g_u) over later / earlier u
        own = torch.log(g.clamp(min=1e-6))
        if self.arm == "R_2st_fzs":                                           # the dead zone on the OWN gate too: the smoke showed
            own = own + (torch.where(g > 0.5, own, torch.full_like(own, -30.0)) - own).detach()   # a 2nd leak (N x eps of own mass)
        a = torch.exp(own + veto) * live
        return a / (a.sum(-1, keepdim=True) + 1e-6)

    def fuzz(self, h):
        return h + FZ_SIGMA * torch.randn_like(h) if self.arm in ("R_2st_fz", "R_2st_fzs", "R_2st_fzv") else h

    def pos_bias(self, bt, tt):
        which = bt["x2"][torch.arange(len(bt["ns"])), bt["ns"]] - W.V_WH
        return self.bias(which, tt)


def build(arm, seed):
    torch.manual_seed(seed)
    if arm == "R_rope": return W.Hybrid("HYB", seed)
    return Router(arm, seed)


def turning(net):                                                            # the ORDER HORIZON per which (first, last)
    if not isinstance(net, Router) or net.router in ("none", "alibi", "alibic", "2stage"): return [None, None]
    t = torch.arange(0, 4001).float(); res = []
    with torch.no_grad():
        for w in (0, 1):
            b = net.bias(torch.tensor([w]), t)[0].numpy()
            b = b if w == 1 else -b                                          # 'last' wants b rising; 'first' wants it falling
            runmax = np.maximum.accumulate(b)
            bad = np.nonzero(b[3:] <= runmax[:-3])[0]
            res.append(int(bad[0]) + 3 if len(bad) else None)
    return res


def evaluate(net, seed):
    er = np.random.default_rng(30_000 + seed); res = {}; net.eval()
    with torch.no_grad():
        for n in EVAL_N:
            nb = {1024: 500, 1536: 300}.get(n, N_BIG if n >= 128 else W.N_EVAL)
            if mode == "smoke": nb = min(nb, 100)
            prs, hits, ys, aws, eps = [], [], [], [], []
            full = W.batch(nb, er, n)
            for i in range(0, nb, 100):
                bt_ = {k_: v_[i:i + 100] for k_, v_ in full.items()}; prs.append(net(bt_).argmax(-1)); hits.append(net.router_hit.numpy())
                if hasattr(net, "gate"):
                    g_ = net.gate; anc = bt_["anc"]; ar_ = torch.arange(len(anc))
                    aws.append(net.last_alpha[ar_, anc].numpy())
                    x2_ = bt_["x2"]; A_ = bt_["x1"][ar_, bt_["ns"]] - W.V_QA
                    nonm = (x2_[:, :n] != W.V_ACT + A_[:, None]); eps.append(float(g_[nonm].mean()))
            bt = full; pr = torch.cat(prs); hit = np.concatenate(hits); ok = (pr == bt["y"]).float().numpy()
            if aws: res[f"aw{n}"], res[f"eps{n}"] = float(np.concatenate(aws).mean()), float(np.mean(eps))
            res[f"N{n}"], res[f"route{n}"] = float(ok.mean()), float(hit.mean())
            if n >= 64:
                ns = bt["ns"].numpy(); x2 = bt["x2"].numpy(); A = bt["x1"].numpy()[np.arange(nb), ns] - W.V_QA
                isA = x2[:, :n] == W.V_ACT + A[:, None]; cnt = isA.sum(1); dist = ns - bt["anc"].numpy()
                gap = np.array([np.ptp(np.nonzero(r)[0]) if r.sum() == 2 else 0 for r in isA])
                two = cnt == 2; res[f"one{n}"] = float(hit[~two].mean()); res[f"two{n}"] = float(hit[two].mean())
                for lo, hi in ((1, 2), (3, 8), (9, 32), (33, 10_000)):
                    m = two & (gap >= lo) & (gap <= hi); res[f"gap{lo}_{n}"] = float(hit[m].mean()) if m.any() else float("nan")
                res[f"twofar{n}"] = float(hit[two & (dist > 64)].mean())
    res["mean"] = float(np.mean([res[f"N{n}"] for n in EVAL_N if n <= W.TRAIN_N[1]]))
    res["route_mean"] = float(np.mean([res[f"route{n}"] for n in EVAL_N if n <= W.TRAIN_N[1]]))
    return res


def run(arm, seed, lr):
    W.STEPS = STEPS
    rng = np.random.default_rng(seed * 11 + 3); net = build(arm, seed)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        bt = W.batch(W.BATCH, rng); loss = F.cross_entropy(net(bt), bt["y"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    res = evaluate(net, seed); res["final_loss"] = float(loss.detach())
    tp = turning(net); res["turn_first"], res["turn_last"] = tp
    if arm == "R_slowL": res["min_period"] = float((2 * math.pi / net.om_r.detach().abs()).min())
    return res


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = {"feas": "_feas", "smoke": "_smoke", "full": ""}[mode] + ("" if RACE == 1 else f"_race{RACE}") + os.environ.get("RR_TAG", "")
    assert mode == "smoke" or not os.path.exists(f".cache/router_race{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/router_race{tag}.txt", "w"); res = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {mode} steps {STEPS}")
    for arm in ARMS:
        if mode == "full":
            sc = {lr: run(arm, 0, lr)["mean"] for lr in (1e-3, 3e-3)}; lr = max(sc, key=sc.get)
            say(f"lr sweep {arm}: {sc} -> {lr}  [{time.time() - t0:.0f}s]")
        else:
            lr = 3e-3
        rs = [run(arm, s, lr) for s in SEEDS]
        num = [k for k in rs[0] if isinstance(rs[0][k], float)]
        A = {k: float(np.nanmean([r[k] for r in rs])) for k in num}
        res.append(dict(arm=arm, lr=lr, seeds=rs, agg=A))
        say(f"{arm:<8} in-range {A['mean']:.3f} (router {A['route_mean']:.3f}) | N128 {A['N128']:.3f} router {A['route128']:.3f} "
            f"(one {A['one128']:.3f} two {A['two128']:.3f} two-far {A['twofar128']:.3f}) | N256 {A['N256']:.3f} router {A['route256']:.3f} "
            f"(one {A['one256']:.3f} two {A['two256']:.3f})  [{time.time() - t0:.0f}s]")
        if RACE == 3:
            say("         long: " + "  ".join(f"N{n} router {A[f'route{n}']:.3f} (one {A[f'one{n}']:.3f} two {A[f'two{n}']:.3f})" for n in (1024, 1536))
                + " | anchor weight / gate leak eps: " + "  ".join(f"N{n} {A.get(f'aw{n}', float('nan')):.3f}/{A.get(f'eps{n}', float('nan')):.4f}" for n in (64, 128, 256, 1024)))
        say(f"         gaps@128 1-2 {A['gap1_128']:.3f}  3-8 {A['gap3_128']:.3f}  9-32 {A['gap9_128']:.3f}  33+ {A['gap33_128']:.3f} | "
            f"order horizons (first, last) per seed: {[(r['turn_first'], r['turn_last']) for r in rs]}"
            + (f" | min learned period {[round(r['min_period']) for r in rs]}" if arm == "R_slowL" else ""))
    json.dump(res, open(f".cache/router_race{tag}.json", "w"), indent=1)
    R = {r["arm"]: r["agg"] for r in res}
    if mode == "feas":
        say(f"FEASIBILITY (some arm in-range >= 0.80): {max(a['mean'] for a in R.values()) >= 0.80}")
    if mode == "full" and RACE == 3:                                         # RACE 3 BARS (ledger 2026-10-21)
        g = lambda a, k: R[a][k]
        say("VERDICTS (3-seed means; router anchor accuracy)")
        t1 = all(g("R_2stage", f"{w}128") >= 0.95 and g("R_2stage", f"{w}256") >= 0.90 for w in ("one", "two")) and g("R_2stage", "mean") >= 0.985
        say("(T1) " + ("THE COMPARISON HOLDS" if t1 else "THE COMPARISON DOES NOT HOLD") + f"  (one/two at 128 {g('R_2stage', 'one128'):.3f}/{g('R_2stage', 'two128'):.3f}, 256 {g('R_2stage', 'one256'):.3f}/{g('R_2stage', 'two256'):.3f}, in-range {g('R_2stage', 'mean'):.3f})")
        wrap = g("R_2stage", "two1024") >= 0.90 and g("R_2stage", "two1536") <= g("R_2st_idx", "two1536") - 0.10
        say("(T2) " + ("ONE TURN IS THE HORIZON (holds at 1024, wraps by 1536)" if wrap else "NO WRAP AS PREDICTED") + f"  (two 1024 {g('R_2stage', 'two1024'):.3f}, 1536 {g('R_2stage', 'two1536'):.3f} vs index {g('R_2st_idx', 'two1536'):.3f})")
        tie = all(abs(g("R_2stage", f"route{n}") - g("R_2st_idx", f"route{n}")) <= 0.02 for n in (128, 256, 1024))
        say("(T3) " + ("THE CLOCK IS A SUFFICIENT ORDER KEY WITHIN ONE TURN (ties the index)" if tie else "THE CLOCK AND THE INDEX DIFFER WITHIN ONE TURN"))
        say("(T4) " + ("THE PASS WHEEL UNWRAPS" if g("R_2st_odo", "two1536") >= 0.95 else "THE PASS WHEEL DOES NOT UNWRAP") + f"  (odometer two-sale 1536 {g('R_2st_odo', 'two1536'):.3f})")
        decay = g("R_2st_fz", "aw64") - g("R_2st_fz", "aw256") >= 0.10 and g("R_2st_fz", "route256") < g("R_2st_fz", "route64")
        say("(F1) " + ("THE SOFT GATE LEAKS WITH LENGTH" if decay else "NO LENGTH LEAK") + f"  (anchor weight 64 {g('R_2st_fz', 'aw64'):.3f} -> 256 {g('R_2st_fz', 'aw256'):.3f} -> 1024 {g('R_2st_fz', 'aw1024'):.3f}; eps {g('R_2st_fz', 'eps256'):.4f})")
        f2 = g("R_2st_fzs", "route128") >= 0.95 and g("R_2st_fzs", "route256") >= 0.90
        say("(F2) " + ("THE SNAPPED GATE HOLDS ON FUZZY CONTENT" if f2 else "THE SNAPPED GATE DOES NOT HOLD") + f"  (router 128 {g('R_2st_fzs', 'route128'):.3f}, 256 {g('R_2st_fzs', 'route256'):.3f}; veto-only {g('R_2st_fzv', 'route128'):.3f}/{g('R_2st_fzv', 'route256'):.3f}; soft {g('R_2st_fz', 'route128'):.3f}/{g('R_2st_fz', 'route256'):.3f})")
        dz, ds = g("R_2st_fz", "route64") - g("R_2st_fz", "route1024"), g("R_2st_fzs", "route64") - g("R_2st_fzs", "route1024")
        say("(F3) " + ("THE SNAP STOPS THE LENGTH DECAY" if ds <= 0.5 * dz and dz > 0.05 else "THE SNAP DOES NOT STOP THE DECAY")
            + f"  (router 64 -> 1024 drop: soft {dz:+.3f}, snapped {ds:+.3f}, veto-only {g('R_2st_fzv', 'route64') - g('R_2st_fzv', 'route1024'):+.3f})")
    if mode == "full" and RACE == 2:                                         # RACE 2 BARS (ledger 2026-10-21 ~09:57)
        say("VERDICTS (3-seed means; router anchor accuracy)")
        ok = {a: R[a]["route128"] >= 0.95 and R[a]["route256"] >= 0.90 and R[a]["mean"] >= IN_RANGE_BANKED - 0.01 for a in ALL}
        for a in ALL:
            say(f"(S1) {a:<8} " + ("HOLDS" if ok[a] else "DOES NOT HOLD") + f"  (router128 {R[a]['route128']:.3f}, router256 {R[a]['route256']:.3f}, in-range {R[a]['mean']:.3f})")
        d = R["R_gearS"]["route128"] - R["R_alibiC"]["route128"]
        say("(S2) " + ("THE CLOCK EARNS THE TIME SENSE" if ok["R_gearS"] and d >= 0.03 else
                       "BOUNDEDNESS IS ENOUGH (the clock = a bounded frozen ramp)" if ok["R_gearS"] and ok["R_alibiC"] and abs(d) < 0.03 else
                       "THE BOUNDED RAMP BEATS THE CLOCK" if ok["R_alibiC"] and d <= -0.03 else
                       "THE TIME SENSE IS NOT EARNED") + f"  (gearS - alibiC router128 {d:+.3f})")
        for a in ALL:
            say(f"(S3) {a:<8} gap 1-2 {R[a]['gap1_128']:.3f} vs 33+ {R[a]['gap33_128']:.3f}; one {R[a]['one128']:.3f} two {R[a]['two128']:.3f}")
    if mode == "full" and RACE == 1:                                         # THE PINNED BARS (ledger 2026-10-21)
        say("VERDICTS (3-seed means; router anchor accuracy)")
        for a in ALL:
            ok = R[a]["route128"] >= 0.95 and R[a]["mean"] >= IN_RANGE_BANKED - 0.01
            say(f"(R1) {a:<8} " + ("HOLDS AT 128" if ok else "DOES NOT HOLD AT 128") + f"  (router128 {R[a]['route128']:.3f}, in-range {R[a]['mean']:.3f})")
        d = R["R_gear"]["route128"] - R["R_alibi"]["route128"]
        gear_ok = R["R_gear"]["route128"] >= 0.95 and R["R_gear"]["mean"] >= IN_RANGE_BANKED - 0.01
        say("(R2) " + ("THE CLOCK IS THE TIME SENSE" if gear_ok and d >= 0.03 else "THE TIME SENSE IS NOT EARNED BY THE CLOCK")
            + f"  (gear - alibi router128 {d:+.3f})")
        say("(R3) HORIZONS router256: " + ", ".join(f"{a} {R[a]['route256']:.3f}" for a in ALL)
            + f"; learned vs frozen band (slowL - gear) router128 {R['R_slowL']['route128'] - R['R_gear']['route128']:+.3f}")
        for a in ALL:
            if R[a]["route128"] >= 0.95:
                g1, g33 = R[a]["gap1_128"], R[a]["gap33_128"]
                say(f"(R4) {a:<8} " + ("GAP-1 WEAKEST" if g1 <= g33 - 0.05 else "GAP DOES NOT MATTER") + f"  (gap 1-2 {g1:.3f} vs 33+ {g33:.3f})")
    say(f"DONE in {time.time() - t0:.0f}s")
