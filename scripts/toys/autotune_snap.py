"""autotune_snap.py -- T3c, WHAT AUTO-TUNE KNOWS ABOUT THE SNAP (2026-10-22; Bryce's word; Opus's guards; ledger "T3c").
Auto-Tune snaps a noisy pitch to a codebook in real time; four of its knobs map onto the hop read:
  SCALE   snap only to notes in the KEY  -> hard snap restricted to the entities present in THIS episode
  RETUNE  retune speed                   -> partial snap: next = raw + alpha (code - raw), alpha in {0.25, 0.5, 0.75}
  GATED   don't correct unvoiced sounds  -> hard snap only where the top-1 vs top-2 margin clears tau, raw elsewhere
  (HARD = alpha 1 = T3's snap; RAW = alpha 0.)
Memories (T3 recipe, unseen pairs, 12.5 % coverage, depth 3): trained FLAT / COMPRESS / PHASOR_L at W 128 (trained once,
with T3's soft snap at train -- 'train wide'; the policies differ only at READ), and the analytic PHASOR at W 32 (where the
raw read has room to fail).
OPUS'S GUARDS: tau is CHOSEN ON A VALIDATION SPLIT (unseen chains, another seed) by a rule pinned here -- the tau among the
margin deciles {0, 10, ..., 90 %} or +inf (never snap) with the best validation depth-3 accuracy, ties to the lower tau -- then SCORED on the
test split; and the gate's CALIBRATION is reported (when it snaps, how often is the snapped partner right).
THE FREE READOUT FIRST (Opus): where do HARD's wrong partner snaps land -- on another entity IN the episode, or outside it?
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
os.environ.setdefault("CJ_W", "128")
import cross_join_toy as CJ  # noqa: E402

OUT = ".cache/autotune_snap.txt"
ALPHAS = (0.25, 0.5, 0.75)


def parts(net):
    """-> (kind, helpers) so one read loop serves all memory types."""
    if isinstance(net, CJ.Flat):
        return "flat"
    if isinstance(net, CJ.Compress):
        return "compress"
    return "vsa"


def read(net, bt, policy, tau=None, alpha=None, log=None):
    """One depth-D read under a hop policy. Returns answer logits; fills log with per-hop partner stats."""
    kind = parts(net); R = bt["R"]; B = R.shape[0]; ar = torch.arange(B)
    present = (F.one_hot(R[..., 0], CJ.NE) * bt["M"][..., None]).sum(1) > 0        # the episode's owners = the 'key'
    if kind == "flat":
        E = net.E
        reg = net.wr(torch.cat([E["e"](R[..., 0]), E["o"](R[..., 1]), E["e"](R[..., 2]), E["v"](R[..., 3]), E["d"](R[..., 4])], -1))
        book = E["e"].weight; cur = E["e"](bt["e0"]); ob = E["o"](bt["o"])
    elif kind == "compress":
        a = net.alg
        reg = net.A(torch.cat([a.bind(a.code("e", R[..., 0]), a.code("o", R[..., 1])), a.code("e", R[..., 2]),
                               a.code("v", R[..., 3]), a.code("d", R[..., 4])], -1))
        book = a.book("e"); cur = a.code("e", bt["e0"]); ob = a.code("o", bt["o"])
    else:
        a = net.algs[0]; regs = net.write(R); book = a.book("e"); cur = a.code("e", bt["e0"]); ob = a.code("o", bt["o"])
    for h in range(bt["D"]):
        if kind == "flat":
            sc = torch.einsum("bkp,bp->bk", reg, net.qk(torch.cat([cur, ob], -1)))
        elif kind == "compress":
            sc = torch.einsum("bkp,bp->bk", reg, net.Q(a.bind(cur, ob)))
        else:
            sc = net.match(regs, [a.bind(cur, ob)])
        w = F.one_hot(sc.masked_fill(~bt["M"], -1e9).argmax(-1), sc.shape[-1]).float()
        if kind == "vsa":
            raw = net.raw(regs, w, "par")[0]; xv = None
        else:
            x = torch.einsum("bk,bkp->bp", w, reg)
        if h == bt["D"] - 1:
            if kind == "flat":
                return net.hv(x) @ E["v"].weight.T
            if kind == "compress":
                return a.sim(net.dec[1](x), a.book("v"))
            return net.read(regs, w, "val", "v")
        if kind == "flat":
            raw = net.hp(x); lg = raw @ book.T
        elif kind == "compress":
            raw = net.dec[0](x); lg = a.sim(raw, book)
        else:
            lg = a.sim(raw, book)
        if policy == "scale":
            lg = lg.masked_fill(~present, -1e9)
        top2 = lg.topk(2, -1).values; marg = top2[:, 0] - top2[:, 1]
        pick = lg.argmax(-1); code = book[pick]
        truth = R[ar, sc.masked_fill(~bt["M"], -1e9).argmax(-1), 2]                 # the matched record's true partner
        if log is not None:
            log.setdefault(h, []).append(dict(right=(pick == truth), inside=present[ar, pick], marg=marg))
        if policy in ("hard", "scale"):
            cur = code
        elif policy == "raw":
            cur = raw
        elif policy == "retune":
            cur = raw + alpha * (code - raw)
        elif policy == "gated":
            g = (marg > tau)[:, None]
            cur = torch.where(g, code, raw)
            if log is not None:
                log[h][-1]["gate"] = marg > tau


def acc(net, bt, policy, **kw):
    with torch.no_grad():
        return float((read(net, bt, policy, **kw).argmax(-1) == bt["y"]).float().mean())


def main():
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    assert CJ.mode == "smoke" or not os.path.exists(OUT), f"refusing to overwrite {OUT}"
    out = open(OUT if CJ.mode != "smoke" else OUT.replace(".txt", "_smoke.txt"), "w"); t0 = time.time(); R_ = {}

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {CJ.mode} steps {CJ.STEPS}")
    mems = [("FLAT", 128), ("COMPRESS", 128), ("PHASOR_L", 128), ("PHASOR", 32)]
    for arm, W in mems:
        CJ.W = W
        rows = []
        for s in CJ.SEEDS:
            cov = CJ.coverage(0.125, s); net = CJ.make(arm, s)
            if arm != "PHASOR":
                CJ.train(net, cov, s, True)
            net.eval()
            val = CJ.batch(np.random.default_rng(60_000 + s), CJ.N_EVAL, 3, cov, "unseen")
            test = CJ.batch(np.random.default_rng(80_000 + s * 10 + 3 + 5), CJ.N_EVAL, 3, cov, "unseen")
            r = {"hard": acc(net, test, "hard"), "raw": acc(net, test, "raw"), "scale": acc(net, test, "scale")}
            for al in ALPHAS:
                r[f"retune{al}"] = acc(net, test, "retune", alpha=al)
            lg = {}
            with torch.no_grad():
                read(net, val, "hard", log=lg)
            m = torch.cat([d["marg"] for h in lg for d in lg[h]])
            taus = [float(torch.quantile(m, q)) for q in np.arange(0, 1.0, 0.1)] + [float("inf")]   # + never snap (= RAW)
            vals = [acc(net, val, "gated", tau=t) for t in taus]
            best = int(np.argmax(vals))                                              # ties -> the lower tau (first max)
            r["gated"] = acc(net, test, "gated", tau=taus[best]); r["gate_q"] = best / 10
            gl = {}
            with torch.no_grad():
                read(net, test, "gated", tau=taus[best], log=gl)
            snapped = torch.cat([d["gate"] for h in gl for d in gl[h]]); right = torch.cat([d["right"] for h in gl for d in gl[h]])
            r["gate_snap_share"] = float(snapped.float().mean())
            r["gate_snap_right"] = float(right[snapped].float().mean()) if snapped.any() else float("nan")
            r["gate_raw_right"] = float(right[~snapped].float().mean()) if (~snapped).any() else float("nan")
            hl = {}
            with torch.no_grad():
                read(net, test, "hard", log=hl)
            rt = torch.cat([d["right"] for h in hl for d in hl[h]]); ins = torch.cat([d["inside"] for h in hl for d in hl[h]])
            r["hard_wrong"] = float((~rt).float().mean())
            r["wrong_inside"] = float(ins[~rt].float().mean()) if (~rt).any() else float("nan")
            sl = {}
            with torch.no_grad():
                read(net, test, "scale", log=sl)
            r["scale_wrong"] = float((~torch.cat([d["right"] for h in sl for d in sl[h]])).float().mean())
            rows.append(r)
        a = {k: float(np.nanmean([x[k] for x in rows])) for k in rows[0]}; R_[f"{arm}@{W}"] = dict(seeds=rows, agg=a)
        say(f"{arm:9s} W {W:3d} | D3 unseen: HARD {a['hard']:.3f} RAW {a['raw']:.3f} SCALE {a['scale']:.3f} | RETUNE "
            + " ".join(f"{al}:{a[f'retune{al}']:.3f}" for al in ALPHAS)
            + f" | GATED {a['gated']:.3f} (val-picked decile {a['gate_q']:.2f}; snaps {a['gate_snap_share']:.2f} of hops, right when snapping {a['gate_snap_right']:.3f}, right when raw {a['gate_raw_right']:.3f})"
            + f" | HARD wrong partner snaps {a['hard_wrong']:.3f}, of them inside the episode {a['wrong_inside']:.3f}; SCALE wrong {a['scale_wrong']:.3f}  [{time.time() - t0:.0f}s]")
    json.dump(R_, open(OUT.replace(".txt", ".json") if CJ.mode != "smoke" else OUT.replace(".txt", "_smoke.json"), "w"), indent=1)
    if CJ.mode != "smoke":                                                        # THE PINNED BARS (ledger 2026-10-22, T3c)
        say("VERDICTS (3-seed means, unseen depth 3)")
        g1 = {k: v["agg"]["gated"] >= max(v["agg"]["hard"], v["agg"]["raw"]) - 0.01 for k, v in R_.items()}
        say("(G1) " + ("THE GATE IS NEVER WORSE THAN THE BETTER OF SNAP AND RAW" if all(g1.values()) else "THE GATE LOSES TO SNAP OR RAW SOMEWHERE")
            + "  (" + ", ".join(f"{k} gated {v['agg']['gated']:.3f} vs max {max(v['agg']['hard'], v['agg']['raw']):.3f}" for k, v in R_.items()) + ")")
        g2 = all(v["agg"]["gate_snap_right"] >= 0.90 for v in R_.values())
        say("(G2) " + ("THE GATE IS CALIBRATED: WHEN IT SNAPS, IT IS RIGHT >= 0.90 IN EVERY MEMORY" if g2 else "THE GATE IS NOT CALIBRATED EVERYWHERE")
            + "  (" + ", ".join(f"{k} {v['agg']['gate_snap_right']:.3f}" for k, v in R_.items()) + ")")
        f = R_["FLAT@128"]["agg"]
        say("(S1) " + ("SCALE HALVES FLAT'S WRONG SNAPS" if f["scale_wrong"] <= 0.5 * f["hard_wrong"] else "SCALE DOES NOT HALVE FLAT'S WRONG SNAPS")
            + f"  (wrong partner snaps HARD {f['hard_wrong']:.3f} -> SCALE {f['scale_wrong']:.3f}; HARD's wrong snaps inside the episode {f['wrong_inside']:.3f})")
        say("RETUNE (reported): best alpha per memory: " + ", ".join(
            f"{k} " + max([("0(raw)", v["agg"]["raw"])] + [(str(al), v["agg"][f"retune{al}"]) for al in ALPHAS] + [("1(hard)", v["agg"]["hard"])], key=lambda t: t[1])[0]
            for k, v in R_.items()))
    say(f"DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
