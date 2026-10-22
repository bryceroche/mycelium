"""label_noise_toy.py -- T2e, LABEL NOISE (2026-10-22; branch binding-comprehension; Bryce's word 15:1x; ledger "T2e").
T2/T2c's gain came from PERFECT record labels. The spaCy sample (LLM-judged, 15:12) says the main line's owner labels are
~45 % wrong as-is and ~16 % wrong with pronouns resolved, and the errors follow RULES. How fast does the record-side gain
fall with wrong OWNER labels (object/verb labels stay clean)?
  STRUCT_U            no supervision (the floor)
  REC_RANDOM_r        SUP_REC; each record's owner label replaced with prob r by a uniformly random OTHER entity
  REC_STRUCT_r        SUP_REC; with prob r by a RULE-FOLLOWING wrong owner (Opus): a GIVES record takes its RECIPIENT
                      (spaCy's giver/recipient swap); any other record BORROWS the owner of the nearest record with a
                      different owner (the coordinated-price error); none such -> a random other entity
  r in {0, 0.15, 0.30, 0.45}. Training labels only; eval reads answers. Generator/trunk/reads = T2c (SS_DISTRACT off).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import supervision_split_toy as SS  # noqa: E402

RATES = (0.0, 0.15, 0.30, 0.45)


def corrupt(bt, r, kind, rng):
    own = bt["rfeat"][..., 0].numpy().copy(); m = bt["rmask"].numpy(); rec = bt["rrecip"].numpy()
    B, R = own.shape; out = own.copy()
    for b in range(B):
        n = int(m[b].sum())
        for k in range(n):
            if rng.random() >= r:
                continue
            wrong = -1
            if kind == "struct":
                if rec[b, k] >= 0:
                    wrong = int(rec[b, k])
                else:
                    for d in range(1, n):
                        for j in (k - d, k + d):
                            if 0 <= j < n and own[b, j] != own[b, k]:
                                wrong = int(own[b, j]); break
                        if wrong >= 0:
                            break
            if wrong < 0:
                wrong = int((own[b, k] + 1 + rng.integers(SS.NE - 1)) % SS.NE)
            out[b, k] = wrong
    bt = dict(bt); rf = bt["rfeat"].clone(); rf[..., 0] = torch.from_numpy(out); bt["rfeat"] = rf
    return bt


def run(arm, r, kind, seed):
    rng = np.random.default_rng(seed); nrng = np.random.default_rng(5_000 + seed); erng = np.random.default_rng(90_000 + seed)
    ev = {sp: SS.batch(erng, SS.N_EVAL, sp) for sp in ("iid", "combo")}
    torch.manual_seed(seed); net = SS.Net(arm)
    opt = torch.optim.AdamW(net.parameters(), lr=SS.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SS.LR, total_steps=SS.STEPS, pct_start=0.1)
    for step in range(SS.STEPS):
        bt = SS.batch(rng, SS.BATCH, "train")
        if r > 0:
            bt = corrupt(bt, r, kind, nrng)
        lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
        loss = F.nll_loss(lp[m], tg[m]) + (aux if isinstance(aux, torch.Tensor) else 0.0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return SS.evaluate(net, ev)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = "_smoke" if SS.mode == "smoke" else ""
    assert SS.mode == "smoke" or not os.path.exists(f".cache/label_noise{tag}.txt"), "refusing to overwrite a banked read"
    out = open(f".cache/label_noise{tag}.txt", "w"); allres = []; t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {SS.mode} steps {SS.STEPS} seeds {SS.SEEDS}")
    cells = [("STRUCT_U", 0.0, "none")] + [("SUP_REC", r, k) for k in ("random", "struct") for r in RATES if not (r == 0 and k == "struct")]
    A = {}
    for arm, r, k in cells:
        rs = [run(arm, r, k, s) for s in SS.SEEDS]
        a = {kk: float(np.nanmean([x[kk] for x in rs])) for kk in rs[0]}; A[(arm, r, k)] = a
        allres.append(dict(arm=arm, rate=r, kind=k, seeds=rs, agg=a))
        say(f"{arm:8s} {k:6s} r {r:.2f} | combo {a['acc_combo']:.3f} (explicit {a['acc_combo_qexplicit']:.3f} implicit {a['acc_combo_qimplicit']:.3f}) iid {a['acc_iid']:.3f}  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(f".cache/label_noise{tag}.json", "w"), indent=1)
    if SS.mode == "full":                                                 # THE PINNED BARS (ledger 2026-10-22, T2e)
        g = lambda r, k: A[("SUP_REC", r, k)]["acc_combo"]
        floor, top = A[("STRUCT_U", 0.0, "none")]["acc_combo"], g(0.0, "random")
        kept = lambda r, k: (g(r, k) - floor) / (top - floor)
        say(f"VERDICTS (3-seed means, COMBO; gain = SUP_REC clean {top:.3f} − STRUCT_U {floor:.3f} = {top - floor:+.3f})")
        say("(N1) " + ("THE GAIN SURVIVES 15 % RANDOM NOISE" if kept(0.15, "random") >= 0.80 else "15 % RANDOM NOISE COSTS > 20 % OF THE GAIN") + f"  (kept {kept(0.15, 'random'):.2f})")
        say("(N2) " + ("45 % STRUCTURED NOISE DESTROYS MOST OF THE GAIN" if kept(0.45, "struct") < 0.50 else "THE GAIN SURVIVES 45 % STRUCTURED NOISE") + f"  (kept {kept(0.45, 'struct'):.2f})")
        n3 = all((top - g(r, "struct")) >= 2 * (top - g(r, "random")) for r in (0.30, 0.45))
        say("(N3) " + ("STRUCTURED NOISE COSTS >= 2x RANDOM" if n3 else "STRUCTURED NOISE DOES NOT COST 2x RANDOM") + "  (" + ", ".join(
            f"r {r}: random −{top - g(r, 'random'):.3f} struct −{top - g(r, 'struct'):.3f}" for r in (0.15, 0.30, 0.45)) + ")")
        say("CURVE kept-share random / struct: " + ", ".join(f"r {r}: {kept(r, 'random'):.2f} / {kept(r, 'struct') if r else 1.0:.2f}" for r in RATES))
    say(f"DONE in {time.time() - t0:.0f}s")
