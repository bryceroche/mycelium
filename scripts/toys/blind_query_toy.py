"""blind_query_toy.py -- T2g, THE BLIND QUERY (2026-10-22; Bryce's word; Opus's design + shortcut-probe gate; ledger "T2g").
T2c's implicit query leaked (the object alone pinned the owner) and T2d's failed its validity check. Here the query names NO
owner and is defined by its ROLE in the arithmetic (Opus): k = 2-3 entities each HAVE the same object; afterwards exactly one
of them loses or gives some away; the question is 'Q OBJ LEFT ?' = HAS(the one who lost/gave) - OUT. The OUT record is
surface-findable (the only OUT on OBJ); the HAS operand is NOT -- it is the HAS record whose owner equals the OUT's owner,
a join on identity (the giver may be written as a name, a description or a pronoun).
THE GATE (Opus, sharpened): a SHORTCUT PROBE -- the FLAT arm trained and read on stories with EVERY owner mention (names,
descriptions, he/she, recipients) replaced by one blank token: it sees objects, verbs, numbers, order, and nothing about who.
Validity: its HAS-operand accuracy on blind queries must sit within +0.10 of chance (mean 1/k). If it doesn't, the toy leaks
and the real arms are NOT trained (the run stops and says REDESIGN).
Arms after the gate: FLAT, STRUCT_U, SUP_REC, SUP_QRY, SUP_BOTH (T2c's Net). Half the episodes ask the blind question, half
the EXPLICIT control ('Q NAME OBJ LEFT ?', same operands). Filler events on other objects keep stories T2c-sized.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import supervision_split_toy as SS  # noqa: E402

T_ENT = SS.VOCAB; SS.VOCAB = SS.VOCAB + 1
OUT_TXT = ".cache/blind_query{}.txt"
ARMS = ["PROBE", "FLAT", "STRUCT_U", "SUP_REC", "SUP_QRY", "SUP_BOTH"]


def episode(rng):
    """-> dict(toks, recs, ops, qform, k). recs = (pos, owner, obj, vcls, value, form, recip)."""
    ents = [int(e) for e in rng.choice(SS.NE, size=int(rng.integers(3, 5)), replace=False)]
    desc = {e: int(d) for e, d in zip(ents, rng.permutation(SS.NDESC)[: len(ents)])}
    intro = {e: rng.random() < 0.6 for e in ents}
    O = int(rng.integers(SS.NOBJ)); k = int(rng.integers(2, min(3, len(ents)) + 1))
    owners = [int(x) for x in rng.choice(ents, size=k, replace=False)]; g = owners[int(rng.integers(k))]
    others = [o for o in range(SS.NOBJ) if o != O]
    fill = [(int(rng.choice(ents)), int(rng.choice(others)), int(rng.choice([SS.T_HAS, SS.T_BUYS]))) for _ in range(int(rng.integers(1, 4)))]
    pre = [(e, O, SS.T_HAS) for e in owners] + fill
    order = rng.permutation(len(pre)); pre = [pre[i] for i in order]
    last_has = max(i for i, ev in enumerate(pre) if ev[1] == O)
    cut = int(rng.integers(last_has + 1, len(pre) + 1))                        # OUT comes after every HAS of O
    out_verb = SS.T_GIVES if rng.random() < 0.5 else SS.T_LOSES
    evs = pre[:cut] + [(g, O, out_verb)] + pre[cut:]
    toks, recs, seen, last_by_g, introduced = [], [], set(), {}, set()

    def mention(e):
        if e not in seen:
            seen.add(e); last_by_g[SS.GENDER[e]] = e
            if intro[e]:
                toks.extend([SS.T_S, SS.O_NAME + e, SS.T_IS, SS.O_DESC + desc[e], SS.T_DOT]); introduced.add(e)
            return [SS.O_NAME + e], "first"
        opts = [("name", [SS.O_NAME + e])]
        if last_by_g.get(SS.GENDER[e]) == e:
            opts.append(("pronoun", [SS.T_HE if SS.GENDER[e] == 0 else SS.T_SHE]))
        if e in introduced:
            opts.append(("desc", [SS.O_DESC + desc[e]]))
        form, tk = opts[int(rng.integers(len(opts)))]
        last_by_g[SS.GENDER[e]] = e
        return tk, form

    has_g = out_g = None
    for e, o, v in evs:
        mk, form = mention(e); toks.extend([SS.T_S] + mk + [v]); r = -1
        if v == SS.T_GIVES:
            r = int(rng.choice([x for x in ents if x != e])); rk, _ = mention(r); toks.extend(rk)
        n = int(rng.integers(1, SS.NNUM + 1))
        rec = (len(toks), e, o, SS.VERB_CLS[v], n, form, r); recs.append(rec)
        toks.extend([SS.O_NUM + n - 1, SS.O_OBJ + o, SS.T_DOT])
        if e == g and o == O:
            if v == SS.T_HAS:
                has_g = rec
            elif v in (SS.T_GIVES, SS.T_LOSES):
                out_g = rec
    blind = rng.random() < 0.5
    q = [SS.T_Q, SS.O_OBJ + O, SS.T_LEFT, SS.T_QM] if blind else [SS.T_Q, SS.O_NAME + g, SS.O_OBJ + O, SS.T_LEFT, SS.T_QM]
    return dict(toks=toks + q, recs=recs, ops=[has_g, out_g], qform="blind" if blind else "explicit", k=k)


ERASE = set(range(SS.O_NAME, SS.O_NAME + SS.NE)) | set(range(SS.O_DESC, SS.O_DESC + SS.NDESC)) | {SS.T_HE, SS.T_SHE}


def batch(rng, B, erase=False):
    exs = [episode(rng) for _ in range(B)]
    L = max(len(x["toks"]) for x in exs); R = max(len(x["recs"]) for x in exs)
    tok = np.full((B, L), SS.T_PAD); rpos = np.zeros((B, R), np.int64); rmask = np.zeros((B, R), bool)
    rfeat = np.zeros((B, R, 3), np.int64); qpos = np.zeros(B, np.int64); tgt = np.full((B, 2), -1); qfeat = np.full((B, 2, 3), -1)
    for b, x in enumerate(exs):
        tk = [T_ENT if (erase and t in ERASE) else t for t in x["toks"]]
        tok[b, :len(tk)] = tk; qpos[b] = len(tk) - 1
        for j, r in enumerate(x["recs"]):
            rpos[b, j] = r[0]; rmask[b, j] = True; rfeat[b, j] = (r[1], r[2], r[3])
        for i, op in enumerate(x["ops"]):
            tgt[b, i] = x["recs"].index(op); qfeat[b, i] = (op[1], op[2], op[3])
    t = lambda a: torch.from_numpy(np.asarray(a))
    return dict(tok=t(tok), rpos=t(rpos), rmask=t(rmask), rfeat=t(rfeat), qpos=t(qpos), tgt=t(tgt), nop=t(np.full(B, 2)),
                qfeat=t(qfeat), qform=[x["qform"] for x in exs], k=t(np.array([x["k"] for x in exs])))


def evaluate(net, bt):
    net.eval(); out = {}
    with torch.no_grad():
        lp, _ = net(bt); pick = lp.argmax(-1); okop = pick == bt["tgt"]; ok = okop.all(-1).float()
        for f in ("blind", "explicit"):
            m = torch.tensor([x == f for x in bt["qform"]])
            out[f"acc_{f}"] = float(ok[m].mean()); out[f"has_{f}"] = float(okop[m, 0].float().mean()); out[f"out_{f}"] = float(okop[m, 1].float().mean())
        mb = torch.tensor([x == "blind" for x in bt["qform"]])
        out["chance_has_blind"] = float((1.0 / bt["k"][mb].float()).mean())
    net.train(); return out


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(90_000 + seed); erase = arm == "PROBE"
    ev = batch(erng, 6000 if SS.mode == "full" else SS.N_EVAL, erase)
    torch.manual_seed(seed); net = SS.Net("FLAT" if erase else arm)
    opt = torch.optim.AdamW(net.parameters(), lr=SS.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SS.LR, total_steps=SS.STEPS, pct_start=0.1)
    for step in range(SS.STEPS):
        bt = batch(rng, SS.BATCH, erase); lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
        loss = F.nll_loss(lp[m], tg[m]) + (aux if isinstance(aux, torch.Tensor) else 0.0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return evaluate(net, ev)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    OUT = OUT_TXT.format("_smoke" if SS.mode == "smoke" else "")
    assert SS.mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {SS.mode} steps {SS.STEPS} seeds {SS.SEEDS}")
    for arm in ARMS:
        rs = [run(arm, s) for s in SS.SEEDS]
        a = {kk: float(np.mean([x[kk] for x in rs])) for kk in rs[0]}; A[arm] = a; allres.append(dict(arm=arm, seeds=rs, agg=a))
        say(f"{arm:8s} | BLIND {a['acc_blind']:.3f} (HAS op {a['has_blind']:.3f}, OUT op {a['out_blind']:.3f}; HAS chance {a['chance_has_blind']:.3f}) | "
            f"EXPLICIT {a['acc_explicit']:.3f} (HAS {a['has_explicit']:.3f}, OUT {a['out_explicit']:.3f})  [{time.time() - t0:.0f}s]")
        if arm == "PROBE":
            gate = a["has_blind"] <= a["chance_has_blind"] + 0.10
            say("(B0, THE GATE) " + ("THE OWNER-BLIND PROBE SITS NEAR CHANCE -> THE QUERY DOES NOT LEAK" if gate else "THE OWNER-BLIND PROBE BEATS CHANCE BY > 0.10 -> THE TOY LEAKS: REDESIGN, NO ARMS TRAINED")
                + f"  (probe HAS-operand {a['has_blind']:.3f} vs chance {a['chance_has_blind']:.3f})")
            if SS.mode == "full" and not gate:
                break
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if SS.mode == "full" and "SUP_BOTH" in A:                              # THE PINNED BARS (ledger 2026-10-22, T2g)
        g = lambda arm, k: A[arm][k]
        say("VERDICTS (3-seed means)")
        b3 = g("SUP_REC", "acc_blind") <= g("SUP_REC", "acc_explicit") - 0.05
        say("(B3, validity) " + ("BLIND QUERIES ARE HARDER THAN EXPLICIT WITHOUT QUERY SUPERVISION" if b3 else "BLIND QUERIES ARE NOT HARDER -> B1 UNTESTED")
            + f"  (SUP_REC blind {g('SUP_REC', 'acc_blind'):.3f} vs explicit {g('SUP_REC', 'acc_explicit'):.3f})")
        d = g("SUP_BOTH", "acc_blind") - g("SUP_REC", "acc_blind")
        say("(B1) " + ("QUERY SUPERVISION EARNS ITS KEEP ON BLIND QUERIES" if d >= 0.10 else "QUERY SUPERVISION ADDS < 0.10 ON BLIND QUERIES")
            + f"  (SUP_BOTH {g('SUP_BOTH', 'acc_blind'):.3f} − SUP_REC {g('SUP_REC', 'acc_blind'):.3f} = {d:+.3f}; SUP_QRY {g('SUP_QRY', 'acc_blind'):.3f}; STRUCT_U {g('STRUCT_U', 'acc_blind'):.3f}; FLAT {g('FLAT', 'acc_blind'):.3f})")
        say("(B2) " + ("BLIND QUERIES ARE SOLVABLE FULLY SUPERVISED" if g("SUP_BOTH", "acc_blind") >= 0.90 else "BLIND QUERIES STAY HARD FULLY SUPERVISED")
            + f"  (SUP_BOTH blind {g('SUP_BOTH', 'acc_blind'):.3f}; best arm {max(A[x]['acc_blind'] for x in A if x != 'PROBE'):.3f})")
    say(f"DONE in {time.time() - t0:.0f}s")
