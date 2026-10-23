"""entity_slot_toy.py -- T2i, ENTITY SLOTS ON THE WRITE SIDE (2026-10-23; Bryce's word; ledger "T2i").
T2h: the imposed query hop is exact (OUT pick 1.000, HAS pick given the key 0.998); the ceiling is the OUT record's owner
(decode 0.836; DESCRIPTIONS 0.668, GIVES decodes drift to the recipient). Resolving a mention is itself a join. Here the
write side gets an IMPOSED hop too, onto per-episode ENTITY SLOTS (the pointer form; the main line's positional var slots),
and the query hop matches slot distributions (no global name codebook anywhere in SLOT).
REGIME CHANGE for every arm (the main line's frozen trunk knows names): name and description tokens use FROZEN random
embeddings (names carry a +-2 gender direction); every other token embedding trains. 4 of the 16 names are HELD OUT of
training (ids 6, 7, 14, 15: two per gender); the HELDOUT eval split makes the giver a held-out name.
Arms:
  SUP_BOTH  T2g's one-hop reader (record + query labels; global name codebook)
  HOP2_I    T2h's imposed query hop (record labels; global codebook decode + snap)
  SLOT      write hop: each record's owner MENTION (its token position is given -- a span, structural entry; spaCy gives
            subjects) is resolved over the episode's slots (entities in first-mention order) by
              logit_j = a cos(E(m), E(name_j)) + b cos(E(m), E(desc_j)) + [m is a pronoun] (w_m g(name_j) + rho[rank_j])
            (equality by frozen embeddings; g = a learned linear gender read of the frozen name; rank_j = recency rank of
            slot j's last NAME/DESC mention before m, by token equality, no gold); a, b, w, g, rho are learned.
            Supervised by the gold SLOT index (record labels in slot form). Query hop: pick the OUT by object/verb channels,
            take its slot distribution (soft at train, hard snap at read) as the key; the HAS score adds log(key . p_slot).
            Explicit questions resolve the question's name to a slot the same way.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import hop_query_toy as HQ  # noqa: E402
BQ, SS = HQ.BQ, HQ.SS

HELDOUT = {6, 7, 14, 15}; S = 4
ARMS = ["SUP_BOTH", "HOP2_I", "SLOT"]
OUT_TXT = ".cache/entity_slot{}.txt"
_g = torch.Generator().manual_seed(2026)
FROZEN = torch.randn(SS.VOCAB, SS.D, generator=_g)
GDIR = F.normalize(torch.randn(SS.D, generator=_g), dim=0)
for e in range(SS.NE):
    FROZEN[SS.O_NAME + e] += 2.0 * (1 if SS.GENDER[e] == 0 else -1) * GDIR
ISFROZEN = torch.zeros(SS.VOCAB, dtype=torch.bool); ISFROZEN[SS.O_NAME:SS.O_NAME + SS.NE] = True; ISFROZEN[SS.O_DESC:SS.O_DESC + SS.NDESC] = True


class MixEmb(nn.Module):
    def __init__(self):
        super().__init__(); self.tr = nn.Embedding(SS.VOCAB, SS.D); self.register_buffer("fz", FROZEN.clone()); self.register_buffer("isf", ISFROZEN.clone())

    def forward(self, tok):
        return torch.where(self.isf[tok][..., None], self.fz[tok], self.tr(tok))


def episode(rng, split):
    pool = [e for e in range(SS.NE) if e not in HELDOUT] if split == "train" else list(range(SS.NE))
    if split == "heldout":
        g = int(rng.choice(sorted(HELDOUT)))
        rest = [e for e in range(SS.NE) if e != g]
        ents = [g] + [int(x) for x in rng.choice(rest, size=int(rng.integers(2, 4)), replace=False)]
    else:
        ents = [int(e) for e in rng.choice(pool, size=int(rng.integers(3, 5)), replace=False)]
    desc = {e: int(d) for e, d in zip(ents, rng.permutation(SS.NDESC)[: len(ents)])}
    intro = {e: rng.random() < 0.6 for e in ents}
    O = int(rng.integers(SS.NOBJ)); k = int(rng.integers(2, min(3, len(ents)) + 1))
    if split == "heldout":
        owners = [ents[0]] + [int(x) for x in rng.choice(ents[1:], size=k - 1, replace=False)]; g = ents[0]
    else:
        owners = [int(x) for x in rng.choice(ents, size=k, replace=False)]; g = owners[int(rng.integers(k))]
    others = [o for o in range(SS.NOBJ) if o != O]
    fill = [(int(rng.choice(ents)), int(rng.choice(others)), int(rng.choice([SS.T_HAS, SS.T_BUYS]))) for _ in range(int(rng.integers(1, 4)))]
    pre = [(e, O, SS.T_HAS) for e in owners] + fill
    order = rng.permutation(len(pre)); pre = [pre[i] for i in order]
    last_has = max(i for i, ev in enumerate(pre) if ev[1] == O)
    cut = int(rng.integers(last_has + 1, len(pre) + 1))
    out_verb = SS.T_GIVES if rng.random() < 0.5 else SS.T_LOSES
    evs = pre[:cut] + [(g, O, out_verb)] + pre[cut:]
    toks, recs, seen, last_by_g, introduced, first = [], [], set(), {}, set(), []

    def mention(e):
        if e not in seen:
            seen.add(e); last_by_g[SS.GENDER[e]] = e; first.append(e)
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
        toks.append(SS.T_S); mk, form = mention(e); mpos = len(toks); toks.extend(mk + [v]); r = -1
        if v == SS.T_GIVES:
            r = int(rng.choice([x for x in ents if x != e])); rk, _ = mention(r); toks.extend(rk)
        n = int(rng.integers(1, SS.NNUM + 1))
        rec = (len(toks), e, o, SS.VERB_CLS[v], n, form, r, mpos); recs.append(rec)
        toks.extend([SS.O_NUM + n - 1, SS.O_OBJ + o, SS.T_DOT])
        if e == g and o == O:
            if v == SS.T_HAS:
                has_g = rec
            elif v in (SS.T_GIVES, SS.T_LOSES):
                out_g = rec
    blind = rng.random() < 0.5
    q = [SS.T_Q, SS.O_OBJ + O, SS.T_LEFT, SS.T_QM] if blind else [SS.T_Q, SS.O_NAME + g, SS.O_OBJ + O, SS.T_LEFT, SS.T_QM]
    qnpos = -1 if blind else len(toks) + 1
    return dict(toks=toks + q, recs=recs, ops=[has_g, out_g], qform="blind" if blind else "explicit", k=k, qnpos=qnpos,
                slots=first, desc={e: desc[e] for e in introduced})


def batch(rng, B, split):
    exs = [episode(rng, split) for _ in range(B)]
    L = max(len(x["toks"]) for x in exs); R = max(len(x["recs"]) for x in exs)
    tok = np.full((B, L), SS.T_PAD); rpos = np.zeros((B, R), np.int64); rmask = np.zeros((B, R), bool)
    rfeat = np.zeros((B, R, 3), np.int64); qpos = np.zeros(B, np.int64); tgt = np.full((B, 2), -1); qfeat = np.full((B, 2, 3), -1)
    mpos = np.zeros((B, R), np.int64); rslot = np.zeros((B, R), np.int64); rform = np.zeros((B, R), np.int64)
    sname = np.full((B, S), SS.T_PAD); sdesc = np.full((B, S), -1); smask = np.zeros((B, S), bool)
    rank = np.full((B, R, S), S); qrank = np.full((B, S), S); qtok = np.full(B, SS.T_PAD); isexp = np.zeros(B, bool)
    FORMS = {"first": 0, "name": 1, "desc": 2, "pronoun": 3}
    for b, x in enumerate(exs):
        tk = x["toks"]; tok[b, :len(tk)] = tk; qpos[b] = len(tk) - 1
        sl = {e: j for j, e in enumerate(x["slots"])}
        for j, e in enumerate(x["slots"]):
            sname[b, j] = SS.O_NAME + e; smask[b, j] = True
            if e in x["desc"]:
                sdesc[b, j] = SS.O_DESC + x["desc"][e]
        keys = [(sname[b, j], sdesc[b, j]) for j in range(len(x["slots"]))]

        def ranks(m):                                                      # recency of each slot's NAME/DESC mention before m
            last = [max([p for p in range(m) if tk[p] in kk] or [-1]) for kk in keys]
            ordr = sorted([j for j in range(len(keys)) if last[j] >= 0], key=lambda j: -last[j])
            out = np.full(S, S)
            for rr, j in enumerate(ordr):
                out[j] = rr
            return out
        for i, r in enumerate(x["recs"]):
            rpos[b, i] = r[0]; rmask[b, i] = True; rfeat[b, i] = (r[1], r[2], r[3])
            mpos[b, i] = r[7]; rslot[b, i] = sl[r[1]]; rform[b, i] = FORMS[r[5]]; rank[b, i] = ranks(r[7])
        for i, op in enumerate(x["ops"]):
            tgt[b, i] = x["recs"].index(op); qfeat[b, i] = (op[1], op[2], op[3])
        if x["qnpos"] >= 0:
            isexp[b] = True; qtok[b] = tk[x["qnpos"]]; qrank[b] = ranks(x["qnpos"])
    t = lambda a: torch.from_numpy(np.asarray(a))
    return dict(tok=t(tok), rpos=t(rpos), rmask=t(rmask), rfeat=t(rfeat), qpos=t(qpos), tgt=t(tgt), nop=t(np.full(B, 2)),
                qfeat=t(qfeat), qform=[x["qform"] for x in exs], k=t(np.array([x["k"] for x in exs])), mpos=t(mpos), rslot=t(rslot),
                rform=t(rform), sname=t(sname), sdesc=t(sdesc), smask=t(smask), rank=t(rank), qrank=t(qrank), qtok=t(qtok), isexp=t(isexp))


class SlotNet(SS.Net):
    def __init__(self):
        super().__init__("SUP_REC"); self.emb = MixEmb()
        self.a = nn.Parameter(torch.tensor(5.0)); self.b = nn.Parameter(torch.tensor(5.0))
        self.gl = nn.Linear(SS.D, 1); self.wpron = nn.Parameter(torch.zeros(2)); self.rho = nn.Parameter(torch.zeros(S + 1))

    def resolve(self, mtok, rank, bt):
        """mtok (B, X) mention tokens, rank (B, X, S) -> slot logits (B, X, S)."""
        fz = self.emb.fz; em = F.normalize(fz[mtok], dim=-1)
        en = F.normalize(fz[bt["sname"]], dim=-1); ed = F.normalize(fz[bt["sdesc"].clamp(min=0)], dim=-1) * (bt["sdesc"] >= 0)[..., None]
        lg = self.a * torch.einsum("bxd,bsd->bxs", em, en) + self.b * torch.einsum("bxd,bsd->bxs", em, ed)
        pr = (mtok == SS.T_HE) | (mtok == SS.T_SHE)
        w = torch.where(mtok == SS.T_HE, self.wpron[0], self.wpron[1])
        gs = self.gl(fz[bt["sname"]]).squeeze(-1)                            # (B, S) learned gender read
        lg = lg + pr[..., None] * (w[..., None] * gs[:, None, :] + self.rho[rank])
        return lg.masked_fill(~bt["smask"][:, None, :], -1e9)

    def chan2(self, qs_o, rs):
        sc = 0
        for c in (1, 2):
            cs = torch.einsum("bd,brd->br", F.normalize(qs_o[c], dim=-1), F.normalize(rs[c], dim=-1))
            sc = sc + F.logsigmoid(self.kappa[c].exp() * (cs - self.theta[c]))
        return sc

    def forward(self, bt):
        tok = bt["tok"]; B, L = tok.shape; M = bt["rmask"]
        h = self.enc(self.emb(tok) + SS.sinus(L)[None], src_key_padding_mask=(tok == SS.T_PAD))
        r = h[torch.arange(B)[:, None], bt["rpos"]]
        q = self.qop(h[torch.arange(B), bt["qpos"]]).view(B, 2, SS.D)
        rs = torch.split(self.wk(r), self.CH, -1)
        mtok = tok[torch.arange(B)[:, None], bt["mpos"]]
        self._rl = self.resolve(mtok, bt["rank"], bt); prec = F.softmax(self._rl, -1)                    # (B, R, S)
        ql = self.resolve(bt["qtok"][:, None], bt["qrank"][:, None], bt)[:, 0]; pq = F.softmax(ql, -1)      # (B, S)
        ex = bt["isexp"].float()
        q1s = torch.split(self.wq(q[:, 1]), self.CH, -1); q0s = torch.split(self.wq(q[:, 0]), self.CH, -1)
        own_q = torch.log(torch.einsum("bs,brs->br", pq, prec) + 1e-6)
        sc1 = (self.chan2(q1s, rs) + ex[:, None] * own_q).masked_fill(~M, -1e9)
        p1 = F.softmax(sc1, -1) if self.training else F.one_hot(sc1.argmax(-1), sc1.shape[-1]).float()
        key = torch.einsum("br,brs->bs", p1, prec)
        if not self.training:
            key = F.one_hot(key.argmax(-1), S).float()                        # the snap, onto the episode's slots
        key = ex[:, None] * pq + (1 - ex[:, None]) * key
        self._key = key
        sc0 = (self.chan2(q0s, rs) + torch.log(torch.einsum("bs,brs->br", key, prec) + 1e-6)).masked_fill(~M, -1e9)
        aux = F.cross_entropy(self._rl[M], bt["rslot"][M])
        for c in (1, 2):
            aux = aux + F.cross_entropy((F.normalize(rs[c], dim=-1) @ F.normalize(self.cls[c], dim=-1).T * 10)[M], bt["rfeat"][..., c][M])
        return F.log_softmax(torch.stack([sc0, sc1], 1), -1), aux


def make(arm):
    if arm == "SLOT":
        return SlotNet()
    net = SS.Net(arm) if arm == "SUP_BOTH" else HQ.HopNet("I")
    net.emb = MixEmb(); return net


def evaluate(net, ev):
    out = {}
    for sp, bt in ev.items():
        o = BQ.evaluate(net, bt)
        for kk in ("acc_blind", "acc_explicit"):
            out[f"{sp}_{kk}"] = o[kk]
        net.eval()
        with torch.no_grad():
            lp, _ = net(bt); tg = bt["tgt"]; ar = torch.arange(len(tg)); mb = torch.tensor([x == "blind" for x in bt["qform"]])
            if isinstance(net, SlotNet):
                ok = net._rl.argmax(-1) == bt["rslot"]; M = bt["rmask"]
                for f, nm in enumerate(("first", "name", "desc", "pronoun")):
                    m = M & (bt["rform"] == f); out[f"{sp}_res_{nm}"] = float(ok[m].float().mean())
                gk = bt["rslot"][ar, tg[:, 1]]
                out[f"{sp}_outres_blind"] = float((net._rl[ar, tg[:, 1]].argmax(-1) == gk)[mb].float().mean())
            elif isinstance(net, HQ.HopNet):
                own = bt["rfeat"][ar, tg[:, 1], 0]
                out[f"{sp}_outres_blind"] = float((net._dl.argmax(-1) == own)[mb].float().mean())
        net.train()
    return out


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(90_000 + seed)
    n_ev = 6000 if SS.mode == "full" else SS.N_EVAL
    ev = {"indist": batch(erng, n_ev, "train"), "heldout": batch(erng, n_ev, "heldout")}
    torch.manual_seed(seed); net = make(arm)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=SS.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SS.LR, total_steps=SS.STEPS, pct_start=0.1)
    for step in range(SS.STEPS):
        bt = batch(rng, SS.BATCH, "train"); lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
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
    say(f"mode {SS.mode} steps {SS.STEPS} seeds {SS.SEEDS} heldout names {sorted(HELDOUT)}")
    for arm in ARMS:
        rs = [run(arm, s) for s in SS.SEEDS]
        a = {kk: float(np.mean([x[kk] for x in rs])) for kk in rs[0]}; A[arm] = a; allres.append(dict(arm=arm, seeds=rs, agg=a))
        res = (" | slot resolution in-dist first/name/desc/pronoun " + " / ".join(f"{a[f'indist_res_{n}']:.3f}" for n in ("first", "name", "desc", "pronoun"))
               + "; held-out " + " / ".join(f"{a[f'heldout_res_{n}']:.3f}" for n in ("first", "name", "desc", "pronoun"))) if arm == "SLOT" else ""
        oo = f" | OUT-owner resolution blind in-dist {a['indist_outres_blind']:.3f} held-out {a['heldout_outres_blind']:.3f}" if "indist_outres_blind" in a else ""
        say(f"{arm:8s} | IN-DIST blind {a['indist_acc_blind']:.3f} (seeds {' / '.join(f'{x['indist_acc_blind']:.3f}' for x in rs)}) explicit {a['indist_acc_explicit']:.3f}"
            f" | HELD-OUT blind {a['heldout_acc_blind']:.3f} explicit {a['heldout_acc_explicit']:.3f}" + oo + res + f"  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if SS.mode == "full":                                                    # THE PINNED BARS (ledger 2026-10-23, T2i)
        s, i = A["SLOT"], A["HOP2_I"]
        say("VERDICTS (3-seed means)")
        say("(I1) " + ("ENTITY SLOTS SOLVE BLIND QUERIES" if s["indist_acc_blind"] >= 0.95 else "ENTITY SLOTS DO NOT REACH 0.95 ON BLIND QUERIES")
            + f"  (SLOT in-dist blind {s['indist_acc_blind']:.3f} [bar >= 0.95]; HOP2_I {i['indist_acc_blind']:.3f}; SUP_BOTH {A['SUP_BOTH']['indist_acc_blind']:.3f})")
        say("(I2) " + ("DESCRIPTIONS RESOLVE" if s["indist_res_desc"] >= 0.95 else "DESCRIPTIONS STILL DON'T RESOLVE")
            + f"  (SLOT description-mention resolution {s['indist_res_desc']:.3f} [bar >= 0.95]; pronoun {s['indist_res_pronoun']:.3f})")
        d = s["heldout_acc_blind"] - i["heldout_acc_blind"]
        say("(I3) " + ("SLOTS SURVIVE HELD-OUT NAMES WHERE THE GLOBAL CODEBOOK DOESN'T" if d >= 0.10 else "SLOTS DO NOT BEAT THE GLOBAL CODEBOOK BY 0.10 ON HELD-OUT NAMES")
            + f"  (held-out blind SLOT {s['heldout_acc_blind']:.3f} − HOP2_I {i['heldout_acc_blind']:.3f} = {d:+.3f}; SUP_BOTH {A['SUP_BOTH']['heldout_acc_blind']:.3f})")
    say(f"DONE in {time.time() - t0:.0f}s")
