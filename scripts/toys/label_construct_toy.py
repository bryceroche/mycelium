"""label_construct_toy.py -- T2f, CONSTRUCTION-TIED LABEL ERRORS (2026-10-22; Bryce's word; ledger "T2f").
T2e flipped owner labels PER RECORD, so the right owner stayed every context's majority label and supervision averaged the
errors away. spaCy's errors are tied to CONSTRUCTIONS (the samples: pronoun owners 34 % wild / 43 % GSM8K-train, never
resolved; ownerless numerals ~7 %, always given an invented owner). Here the error is deterministic per construction:
  PRONOUN records (the owner is written as he/she): label policy TRUE / WRONG (a consistent rule: the most recently
      mentioned OTHER entity -- a resolver that ignores gender) / MASK (no owner loss on the record; Opus's arm)
  OWNERLESS records (new: 'S THERE IS n OBJ .', owner = the NONE class, never a query target; the object is the previous
      record's object half the time, else one already in the story): label policy NONE (true) / INVENT (the previous
      record's owner -- spaCy borrows a nearby subject) / MASK
Object and verb labels stay clean. Generator = T2c's (SS_DISTRACT off) plus: a pronoun is used whenever available with
prob PRON_P (raises the toy's pronoun share from 0.20 to ~0.35), and an ownerless record follows an event with prob NONE_P.
Readouts on COMBO: answer accuracy overall, by the target record's form (pronoun / name), and on queries whose story has an
ownerless record sharing an operand's object ('distracted'); the OWNER DECODE on pronoun records (share decoded as the true
owner / as the wrong rule's owner) and on ownerless records (share decoded NONE).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import supervision_split_toy as SS  # noqa: E402

PRON_P, NONE_P, SHARE_P = 0.75, 0.09, 0.6
T_THERE = SS.VOCAB; SS.VOCAB = SS.VOCAB + 1; NONE_CLS = SS.NE
GENDER = SS.GENDER
CELLS = [  # (arm, pronoun policy, ownerless policy)
    ("STRUCT_U", "true", "none"), ("SUP_REC", "true", "none"),
    ("SUP_REC", "wrong", "none"), ("SUP_REC", "mask", "none"),
    ("SUP_REC", "true", "invent"), ("SUP_REC", "true", "mask"),
    ("SUP_REC", "mask", "mask"), ("SUP_REC", "wrong", "invent"),
    ("SUP_REC", "allmask", "mask")]                                        # V0: no owner labels at all (validity)


def story(rng, n_ev, split):
    """T2c's story + pronoun bias + ownerless records. Records gain [7] = the wrong-rule owner (pronoun) / invented owner
    (ownerless), -1 if none."""
    ents = rng.choice(SS.NE, size=int(rng.integers(3, 5)), replace=False)
    desc = {int(e): int(d) for e, d in zip(ents, rng.permutation(SS.NDESC)[: len(ents)])}
    intro = {int(e): rng.random() < 0.6 for e in ents}
    toks, recs, seen, last_by_g, introduced, order = [], [], set(), {}, set(), []
    pairs = {}

    def mention(e):
        e = int(e)
        if e not in seen:
            seen.add(e); last_by_g[GENDER[e]] = e; order.append(e)
            if intro[e]:
                toks.extend([SS.T_S, SS.O_NAME + e, SS.T_IS, SS.O_DESC + desc[e], SS.T_DOT]); introduced.add(e)
            return [SS.O_NAME + e], "first", -1
        opts = [("name", [SS.O_NAME + e])]
        if e in introduced:
            opts.append(("desc", [SS.O_DESC + desc[e]]))
        pron = last_by_g.get(GENDER[e]) == e
        if pron and rng.random() < PRON_P:
            form, tk = "pronoun", [SS.T_HE if GENDER[e] == 0 else SS.T_SHE]
        else:
            if pron:
                opts.append(("pronoun", [SS.T_HE if GENDER[e] == 0 else SS.T_SHE]))
            form, tk = opts[int(rng.integers(len(opts)))]
        wrong = next((x for x in reversed(order) if x != e), -1) if form == "pronoun" else -1
        last_by_g[GENDER[e]] = e; order.append(e)
        return tk, form, wrong

    tries = 0
    while len([r for r in recs if r[1] != NONE_CLS]) < n_ev and tries < 200:
        tries += 1
        open_ = [p for p, s in pairs.items() if len(s) < 3]
        if open_ and rng.random() < 0.6:
            e, o = open_[int(rng.integers(len(open_)))]
        else:
            e = int(rng.choice(ents)); o = int(rng.integers(SS.NOBJ))
            if pairs and rng.random() < SHARE_P:                              # T2d's sharing: the owner must decide
                o = int(list(pairs)[int(rng.integers(len(pairs)))][1])
        if (split == "train" and (e, o) in SS.HELD) or (split == "combo" and (e, o) not in SS.HELD and rng.random() < 0.7):
            continue
        st = pairs.setdefault((e, o), set())
        if 0 not in st:
            verb = SS.T_HAS
        else:
            cand = [v for v, c in ((SS.T_BUYS, 1), (SS.T_LOSES, 2), (SS.T_GIVES, 2)) if c not in st]
            if not cand:
                continue
            verb = cand[int(rng.integers(len(cand)))]
        if verb == SS.T_GIVES:
            others = [int(x) for x in ents if int(x) != e]
            r = others[int(rng.integers(len(others)))]
        n = int(rng.integers(1, SS.NNUM + 1))
        mk, form, wrong = mention(e)
        toks.extend([SS.T_S] + mk + [verb])
        if verb == SS.T_GIVES:
            rk, _, _ = mention(r); toks.extend(rk)
        recs.append((len(toks), e, o, SS.VERB_CLS[verb], n, form, r if verb == SS.T_GIVES else -1, wrong))
        toks.extend([SS.O_NUM + n - 1, SS.O_OBJ + o, SS.T_DOT])
        st.add(SS.VERB_CLS[verb])
        if rng.random() < NONE_P:                                          # an ownerless numeral (a rate / generic price)
            oo = o if rng.random() < 0.5 else int(list(pairs)[int(rng.integers(len(pairs)))][1])
            nn = int(rng.integers(1, SS.NNUM + 1))
            toks.extend([SS.T_S, T_THERE, SS.T_IS])
            recs.append((len(toks), NONE_CLS, oo, 0, nn, "none", -1, e))   # invented owner = the previous record's owner
            toks.extend([SS.O_NUM + nn - 1, SS.O_OBJ + oo, SS.T_DOT])
    return toks, recs, desc, introduced


def example(rng, split):
    n_ev = int(rng.integers(3, 6))
    while True:
        toks, recs, desc, introduced = story(rng, n_ev, split)
        owned = [r for r in recs if r[1] != NONE_CLS]
        pool = [r for r in owned if (r[1], r[2]) in SS.HELD] if split == "combo" else list(owned)
        if pool:
            break
    outs = {}
    for r in owned:
        if r[3] == 2:
            outs.setdefault(r[2], []).append(r[1])
    ipool = [r for r in pool if len(outs.get(r[2], [])) == 1 and outs[r[2]][0] == r[1]]
    if ipool and rng.random() < 1 / 3:
        tgt = ipool[int(rng.integers(len(ipool)))]; e, o = tgt[1], tgt[2]
        same = {r[3]: r for r in owned if r[1] == e and r[2] == o}
        if rng.random() < 0.5:
            q = [SS.T_Q, SS.T_WHO, SS.O_OBJ + o, SS.T_LEFT, SS.T_QM]; ops = [same[0], same[2]]; kind = "left"
        else:
            q = [SS.T_Q, SS.T_WHO, SS.T_HAS, SS.O_OBJ + o, SS.T_QM]; ops = [same[0]]; kind = "lookup"
        return dict(toks=toks + q, recs=recs, ops=ops, kind=kind, qform="implicit")
    tgt = pool[int(rng.integers(len(pool)))]
    e, o = tgt[1], tgt[2]
    same = {r[3]: r for r in owned if r[1] == e and r[2] == o}
    qform = "desc" if (e in introduced and rng.random() < 0.5) else "name"
    qo = [SS.O_DESC + desc[e]] if qform == "desc" else [SS.O_NAME + e]
    if 2 in same and rng.random() < 0.5:
        q = [SS.T_Q] + qo + [SS.O_OBJ + o, SS.T_LEFT, SS.T_QM]; ops = [same[0], same[2]]; kind = "left"
    else:
        vt = {0: SS.T_HAS, 1: SS.T_BUYS, 2: None}[tgt[3]]
        if vt is None:
            vt = SS.T_GIVES if any(t == SS.T_GIVES for t in toks[tgt[0] - 4: tgt[0]]) else SS.T_LOSES
        q = [SS.T_Q] + qo + [vt, SS.O_OBJ + o, SS.T_QM]; ops = [tgt]; kind = "lookup"
    return dict(toks=toks + q, recs=recs, ops=ops, kind=kind, qform=qform)


def batch(rng, B, split):
    exs = [example(rng, split) for _ in range(B)]
    L = max(len(x["toks"]) for x in exs); R = max(len(x["recs"]) for x in exs)
    tok = np.full((B, L), SS.T_PAD); rpos = np.zeros((B, R), np.int64); rmask = np.zeros((B, R), bool)
    rfeat = np.zeros((B, R, 3), np.int64); qpos = np.zeros(B, np.int64); ralt = np.full((B, R), -1)
    rkind = np.zeros((B, R), np.int64)                                     # 0 plain, 1 pronoun, 2 ownerless
    tgt = np.full((B, 2), -1); nop = np.zeros(B, np.int64); qfeat = np.full((B, 2, 3), -1); rform = []; distr = []; cont = []
    for b, x in enumerate(exs):
        tok[b, :len(x["toks"])] = x["toks"]; qpos[b] = len(x["toks"]) - 1
        for k, r in enumerate(x["recs"]):
            rpos[b, k] = r[0]; rmask[b, k] = True; rfeat[b, k] = (r[1], r[2], r[3]); ralt[b, k] = r[7]
            rkind[b, k] = {"pronoun": 1, "none": 2}.get(r[5], 0)
        for i, op in enumerate(x["ops"]):
            tgt[b, i] = x["recs"].index(op); qfeat[b, i] = (op[1], op[2], op[3])
        nop[b] = len(x["ops"]); rform.append(x["ops"][0][5])
        cont.append(any(r[1] not in (op[1], NONE_CLS) and r[2] == op[2] and r[3] == op[3] for op in x["ops"] for r in x["recs"]))
        distr.append(any(r[1] == NONE_CLS and r[2] in {op[2] for op in x["ops"]} for r in x["recs"]))
    t = lambda a: torch.from_numpy(np.asarray(a))
    return dict(tok=t(tok), rpos=t(rpos), rmask=t(rmask), rfeat=t(rfeat), qpos=t(qpos), tgt=t(tgt), nop=t(nop), qfeat=t(qfeat),
                ralt=t(ralt), rkind=t(rkind), kind=[x["kind"] for x in exs], rform=rform, qform=[x["qform"] for x in exs],
                distr=t(np.array(distr)), cont=t(np.array(cont)))


class Net(SS.Net):
    NCLS = (SS.NE + 1, SS.NOBJ, 3)                                         # + the NONE owner class


def label(bt, ppol, npol):
    """The training owner labels under the policies; -100 = masked (ignored by cross_entropy)."""
    own = bt["rfeat"][..., 0].clone(); k = bt["rkind"]; alt = bt["ralt"]
    if ppol == "allmask":
        own[:] = -100
    elif ppol == "wrong":
        m = (k == 1) & (alt >= 0); own[m] = alt[m]
        own[(k == 1) & (alt < 0)] = -100
    elif ppol == "mask":
        own[k == 1] = -100
    if npol == "invent":
        m = (k == 2) & (alt >= 0); own[m] = alt[m]
    elif npol == "mask":
        own[k == 2] = -100
    bt = dict(bt); rf = bt["rfeat"].clone(); rf[..., 0] = own; bt["rfeat"] = rf
    return bt


def decode(net, bt):
    """Owner-channel decode per record (the class the record's owner channel points at)."""
    tok = bt["tok"]; B, L = tok.shape
    h = net.enc(net.emb(tok) + SS.sinus(L)[None], src_key_padding_mask=(tok == SS.T_PAD))
    r = h[torch.arange(B)[:, None], bt["rpos"]]
    rs = torch.split(net.wk(r), net.CH, -1)[0]
    return (F.normalize(rs, dim=-1) @ F.normalize(net.cls[0], dim=-1).T).argmax(-1)


def evaluate(net, bt):
    out = SS.evaluate(net, {"combo": bt})
    net.eval()
    with torch.no_grad():
        lp, _ = net(bt); ok = ((lp.argmax(-1) == bt["tgt"]) | (bt["tgt"] < 0)).all(-1).float()
        d = bt["distr"]
        out["acc_distr"] = float(ok[d].mean()) if d.any() else float("nan"); out["n_distr"] = int(d.sum())
        out["acc_undistr"] = float(ok[~d].mean())
        c = bt["cont"]; pr = torch.tensor([x == "pronoun" for x in bt["rform"]]); nmf = torch.tensor([x == "name" for x in bt["rform"]])
        for nm_, m_ in (("cont", c), ("cont_pron", c & pr), ("cont_name", c & nmf), ("uncont", ~c)):
            out[f"acc_{nm_}"] = float(ok[m_].mean()) if m_.any() else float("nan"); out[f"n_{nm_}"] = int(m_.sum())
        if net.arm != "STRUCT_U" and getattr(net, "ppol", "") != "allmask":
            dec = decode(net, bt); m = bt["rmask"]; k = bt["rkind"]; tr = bt["rfeat"][..., 0]; alt = bt["ralt"]
            p = m & (k == 1) & (alt >= 0)
            out["pron_dec_true"] = float((dec[p] == tr[p]).float().mean()); out["pron_dec_wrongrule"] = float((dec[p] == alt[p]).float().mean())
            pl = m & (k == 0)
            out["plain_dec_true"] = float((dec[pl] == tr[pl]).float().mean())
            nm = m & (k == 2)
            out["none_dec_none"] = float((dec[nm] == NONE_CLS).float().mean()); out["none_dec_invent"] = float((dec[nm] == alt[nm]).float().mean())
    net.train(); return out


def run(arm, ppol, npol, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(90_000 + seed)
    ev = batch(erng, 6000 if SS.mode == "full" else SS.N_EVAL, "combo")                # contested pronoun targets are ~4 %
    torch.manual_seed(seed); net = Net(arm); net.ppol = ppol
    opt = torch.optim.AdamW(net.parameters(), lr=SS.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SS.LR, total_steps=SS.STEPS, pct_start=0.1)
    for step in range(SS.STEPS):
        bt = label(batch(rng, SS.BATCH, "train"), ppol, npol)
        lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
        loss = F.nll_loss(lp[m], tg[m]) + (aux if isinstance(aux, torch.Tensor) else 0.0)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
    return evaluate(net, ev)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "8")))
    tag = "_smoke" if SS.mode == "smoke" else ""
    OUT = f".cache/label_construct{tag}.txt"
    assert SS.mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    sh = batch(np.random.default_rng(7), 2000, "train"); km = sh["rkind"][sh["rmask"]]
    say(f"mode {SS.mode} steps {SS.STEPS} seeds {SS.SEEDS} | train record shares: plain {float((km == 0).float().mean()):.3f} "
        f"pronoun {float((km == 1).float().mean()):.3f} ownerless {float((km == 2).float().mean()):.3f}")
    for arm, pp, npol in CELLS:
        rs = [run(arm, pp, npol, s) for s in SS.SEEDS]
        a = {k: float(np.nanmean([x[k] for x in rs])) for k in rs[0]}; A[(arm, pp, npol)] = a
        allres.append(dict(arm=arm, pron=pp, ownerless=npol, seeds=rs, agg=a))
        dec = (f" | decode: pronoun true {a['pron_dec_true']:.3f} wrong-rule {a['pron_dec_wrongrule']:.3f}; plain true {a['plain_dec_true']:.3f}; "
               f"ownerless NONE {a['none_dec_none']:.3f} invented {a['none_dec_invent']:.3f}") if "pron_dec_true" in a else ""
        say(f"{arm:8s} pron {pp:5s} ownerless {npol:6s} | combo {a['acc_combo']:.3f} (target rec pronoun {a['acc_combo_rpronoun']:.3f} "
            f"name {a['acc_combo_rname']:.3f}; CONTESTED {a['acc_cont']:.3f} n {a['n_cont']:.0f}: pronoun {a['acc_cont_pron']:.3f} n {a['n_cont_pron']:.0f} name {a['acc_cont_name']:.3f}; uncontested {a['acc_uncont']:.3f}; distracted {a['acc_distr']:.3f} n {a['n_distr']:.0f} / undistracted {a['acc_undistr']:.3f})"
            + dec + f"  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if SS.mode == "full":                                                # THE PINNED BARS (ledger 2026-10-22, T2f)
        g = lambda pp, npol: A[("SUP_REC", pp, npol)]; cl = g("true", "none"); fl = A[("STRUCT_U", "true", "none")]
        nol = g("allmask", "mask"); w = g("wrong", "none"); mk = g("mask", "none")
        say("VERDICTS (3-seed means, COMBO; CONTESTED = another owner holds an operand's object + verb class)")
        v0 = cl["acc_cont"] >= nol["acc_cont"] + 0.05
        say("(V0, validity) " + ("OWNER LABELS MATTER ON CONTESTED QUERIES" if v0 else "OWNER LABELS DON'T MATTER EVEN WHEN CONTESTED -> C1b, C2 VOID")
            + f"  (contested acc clean {cl['acc_cont']:.3f} vs no owner labels {nol['acc_cont']:.3f} [bar >= +0.05])")
        c1 = w["pron_dec_wrongrule"] >= 0.80
        say("(C1) " + ("WRONG PRONOUN LABELS ARE LEARNED AS THE RULE" if c1 else "WRONG PRONOUN LABELS ARE NOT LEARNED AS THE RULE")
            + f"  (P_WRONG decodes pronoun records as the wrong rule's owner {w['pron_dec_wrongrule']:.3f} [bar >= 0.80], as the true owner {w['pron_dec_true']:.3f})")
        c1b = w["acc_cont_pron"] <= nol["acc_cont_pron"] + 0.05 and w["acc_cont_name"] >= cl["acc_cont_name"] - 0.03
        say("(C1b) " + ("VOID: " if not v0 else "") + ("CONTESTED PRONOUN TARGETS FALL TO THE NO-LABEL FLOOR, NAME TARGETS UNTOUCHED" if c1b else "NOT (CONTESTED PRONOUN TARGETS TO THE FLOOR AND NAME TARGETS UNTOUCHED)")
            + f"  (contested pronoun-target acc P_WRONG {w['acc_cont_pron']:.3f} vs no-label {nol['acc_cont_pron']:.3f} [bar <= +0.05]; "
            f"contested name-target {w['acc_cont_name']:.3f} vs clean {cl['acc_cont_name']:.3f} [bar >= -0.03])")
        c2 = mk["acc_cont_pron"] >= cl["acc_cont_pron"] - 0.05
        say("(C2) " + ("VOID: " if not v0 else "") + ("MASKING PRONOUN LABELS IS NEARLY AS GOOD AS CLEAN" if c2 else "MASKING PRONOUN LABELS COSTS > 0.05 ON CONTESTED PRONOUN TARGETS")
            + f"  (contested pronoun-target acc P_MASK {mk['acc_cont_pron']:.3f} vs clean {cl['acc_cont_pron']:.3f} vs P_WRONG {w['acc_cont_pron']:.3f}; "
            f"P_MASK decodes pronoun records true {mk['pron_dec_true']:.3f})")
        inv, nm = g("true", "invent"), g("true", "mask")
        c3 = inv["acc_distr"] <= cl["acc_distr"] - 0.05
        say("(C3) " + ("INVENTED OWNERS HURT DISTRACTED QUERIES" if c3 else "INVENTED OWNERS COST < 0.05 ON DISTRACTED QUERIES")
            + f"  (distracted acc N_INVENT {inv['acc_distr']:.3f} vs clean {cl['acc_distr']:.3f})")
        c4 = nm["acc_distr"] >= cl["acc_distr"] - 0.02
        say("(C4) " + ("MASKING OWNERLESS LABELS IS AS GOOD AS A NONE CLASS" if c4 else "MASKING OWNERLESS LABELS COSTS > 0.02 VS A NONE CLASS")
            + f"  (distracted acc N_MASK {nm['acc_distr']:.3f} vs clean {cl['acc_distr']:.3f})")
        sm, sr = g("mask", "mask"), g("wrong", "invent")
        say(f"REPORTED: combo / contested -- spaCy+masking {sm['acc_combo']:.3f} / {sm['acc_cont']:.3f}, spaCy as-is {sr['acc_combo']:.3f} / {sr['acc_cont']:.3f}, "
            f"clean {cl['acc_combo']:.3f} / {cl['acc_cont']:.3f}, no owner labels {nol['acc_combo']:.3f} / {nol['acc_cont']:.3f}, floor {fl['acc_combo']:.3f} / {fl['acc_cont']:.3f}")
    say(f"DONE in {time.time() - t0:.0f}s")
