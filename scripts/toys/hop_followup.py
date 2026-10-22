"""hop_followup.py -- two zero-training reads on the banked hop_read checkpoints (2026-10-21; Bryce's word; ledger
"ESCALATION + THE LEAK READ").

(1) ESCALATION (Opus): run a cheap read (beam4 + bucket, mass + bucket); escalate a query to the exact read when a
    trigger says the cheap read is unsure. Triggers: LEFTOVER (the pruned mass the bucket carries; the coordinator's
    proposal) and MARGIN (1 - the top answer's probability; Opus's first form). Budgets: escalate the top 10 / 25 / 50 %
    by the trigger. Reported: combined log loss and accuracy vs the exact read.
(2) THE LEAK READ: does the trained snap beat its oracle because the learned match LEAKS past dead ends? Trace the snap's
    chosen event per hop; a DEAD END = a chosen event (hop < H) whose link matches no event's key. Split accuracy by
    dead end vs not, model vs oracle (the oracle guesses after a dead end, as registered).
Eval sets = hop_read's (same seeds and order). No training; checkpoints are only read.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import hop_race as HR  # noqa: E402
import hop_read as RD  # noqa: E402

OUT = ".cache/hop_followup.txt"
BUDGETS = (0.10, 0.25, 0.50)


def snap_trace(logs, logM, logg, H, link, key, live):
    """-> chosen event per hop (B, H) and dead-end flag (B,)."""
    B, L = logs.shape; lf = logs + logg; chosen = []
    for h in range(H):
        la = lf - torch.logsumexp(lf, -1, keepdim=True); c = la.argmax(-1); chosen.append(c)
        hard = torch.full_like(la, RD.NEG).scatter(-1, c[:, None], 0.0)
        if h < H - 1:
            lf = torch.logsumexp(hard[:, :, None] + logM, 1) + logg
    ch = torch.stack(chosen, 1); ar = torch.arange(B)
    dead = torch.zeros(B, dtype=torch.bool)
    for h in range(H - 1):
        lk = link[ar, ch[:, h]]
        dead |= ~((key == lk[:, None]) & (live > 0)).any(-1)
    return ch, dead


def main():
    assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    L = []; R = {}

    def say(s):
        print(s, flush=True); L.append(s)
    for rg in HR.REGIMES:
        rid = list(HR.REGIMES).index(rg); fbit = HR.REGIMES[rg][1]
        acc = {}
        for seed in (0, 1, 2):
            erng = np.random.default_rng(50_000 + seed * 7 + rid)
            ev = {H: HR.batch(RD.N_EVAL, erng, rg, H) for H in (2, 3)}
            net = RD.Exact(); net.load_state_dict(torch.load(f".cache/hop_read_{rg}_s{seed}.pt")); net.eval()
            with torch.no_grad():
                for H, bt in ev.items():
                    y = bt["y"]; n = len(y); ar = torch.arange(n)
                    logs, logM, logg, lans = net.parts(bt)
                    lp_ex = RD.answer(*RD.read(logs, logM, logg, bt["H"], "exact", False)[:2], lans, False)
                    nll_ex = -lp_ex[ar, y]
                    for base in ("beam4", "mass"):
                        lfin, lleft, _ = RD.read(logs, logM, logg, bt["H"], base, True)
                        lp = RD.answer(lfin, lleft, lans, True); nll = -lp[ar, y]
                        trig = {"leftover": lleft, "margin": 1 - lp.exp().max(-1).values}
                        for tn, t in trig.items():
                            order = torch.argsort(t, descending=True)
                            for bud in BUDGETS:
                                esc = torch.zeros(n, dtype=torch.bool); esc[order[:int(round(bud * n))]] = True
                                comb = torch.where(esc, nll_ex, nll)
                                ca = torch.where(esc, lp_ex.argmax(-1), lp.argmax(-1)) == y
                                acc.setdefault((H, base, tn, bud), []).append((float(comb.mean()), float(ca.float().mean())))
                        acc.setdefault((H, base, "none", 0.0), []).append((float(nll.mean()), float((lp.argmax(-1) == y).float().mean())))
                        acc.setdefault((H, base, "leftover_sd", 0.0), []).append((float(lleft.std()), float(lleft.mean())))
                    acc.setdefault((H, "exact", "none", 0.0), []).append((float(nll_ex.mean()), float((lp_ex.argmax(-1) == y).float().mean())))
                    # the leak read
                    tp = RD.true_parts(bt, fbit)
                    lp_sn = RD.answer(*RD.read(logs, logM, logg, bt["H"], "snap", False)[:2], lans, False)
                    lf_o, _, _ = RD.read(tp[0], tp[1], tp[2], bt["H"], "snap", False)
                    lp_o = torch.logsumexp(lf_o[..., None] + tp[3], 1)
                    _, dead_m = snap_trace(logs, logM, logg, H, bt["link"], bt["key"], bt["live"])
                    _, dead_o = snap_trace(tp[0], tp[1], tp[2], H, bt["link"], bt["key"], bt["live"])
                    am, ao = (lp_sn.argmax(-1) == y).float(), (lp_o.argmax(-1) == y).float()
                    for nm, m in (("dead_o", dead_o), ("live_o", ~dead_o)):
                        acc.setdefault((H, "leak", nm, 0.0), []).append(
                            (float(m.float().mean()), float(am[m].mean()) if m.any() else float("nan"),
                             float(ao[m].mean()) if m.any() else float("nan"), float(dead_m[m].float().mean()) if m.any() else float("nan")))
                    acc.setdefault((H, "leak", "all", 0.0), []).append((float(am.mean()), float(ao.mean())))
        A = {k: tuple(float(np.nanmean([v[i] for v in vs])) for i in range(len(vs[0]))) for k, vs in acc.items()}; R[rg] = A
        for H in (2, 3):
            ex = A[(H, "exact", "none", 0.0)]
            say(f"{rg:8s} H{H} exact ll {ex[0]:.3f} acc {ex[1]:.3f}")
            for base in ("beam4", "mass"):
                nb = A[(H, base, "none", 0.0)]; sd = A[(H, base, "leftover_sd", 0.0)]
                say(f"   {base:5s}+bucket ll {nb[0]:.3f} acc {nb[1]:.3f} (leftover mean {sd[1]:.3f} sd {sd[0]:.3f}) | " + "  ".join(
                    f"{tn} " + " ".join(f"{int(b * 100)}%:{A[(H, base, tn, b)][0]:.3f}" for b in BUDGETS) for tn in ("leftover", "margin")))
            d, lv, al = A[(H, "leak", "dead_o", 0.0)], A[(H, "leak", "live_o", 0.0)], A[(H, "leak", "all", 0.0)]
            say(f"   LEAK snap acc model {al[0]:.3f} vs oracle {al[1]:.3f} | oracle dead-ends {d[0]:.3f} of queries: model acc {d[1]:.3f} vs oracle {d[2]:.3f} "
                f"(model also dead there {d[3]:.3f}) | no oracle dead-end: model {lv[1]:.3f} vs oracle {lv[2]:.3f}")
    say("VERDICTS (3-seed means; nats)")
    for rg in HR.REGIMES:
        A = R[rg]
        e1 = all(A[(H, "mass", "leftover", 0.25)][0] <= A[(H, "exact", "none", 0.0)][0] + 0.02 for H in (2, 3))
        e2 = all(A[(H, "beam4", "leftover", 0.25)][0] < A[(H, "beam4", "margin", 0.25)][0] for H in (2, 3))
        e3 = all(A[(H, "mass", "margin", 0.25)][0] <= A[(H, "exact", "none", 0.0)][0] + 0.02 for H in (2, 3))
        say(f"(E1) {rg}: " + ("MASS + LEFTOVER TRIGGER @25% IS CALIBRATED" if e1 else "MASS + LEFTOVER TRIGGER @25% IS NOT CALIBRATED")
            + "; (E2) " + ("LEFTOVER BEATS MARGIN FOR BEAM4" if e2 else "LEFTOVER DOES NOT BEAT MARGIN FOR BEAM4")
            + "; (E3) " + ("MASS + MARGIN TRIGGER @25% IS CALIBRATED" if e3 else "MASS + MARGIN TRIGGER @25% IS NOT CALIBRATED"))
        if HR.REGIMES[rg][1] is not None or rg == "FILT":
            d = A[(3, "leak", "dead_o", 0.0)]; al = A[(3, "leak", "all", 0.0)]
            surplus = al[0] - al[1]; part = d[0] * (d[1] - d[2])
            say(f"(L1) {rg} H3: " + ("THE LEAK RECOVERS DEAD ENDS" if d[1] - d[2] >= 0.10 and (surplus <= 0 or part >= 0.5 * surplus) else "THE LEAK DOES NOT EXPLAIN IT")
                + f"  (dead-end queries {d[0]:.3f}: model {d[1]:.3f} vs oracle {d[2]:.3f}; their share of the surplus {part:+.3f} of {surplus:+.3f})")
    open(OUT, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
