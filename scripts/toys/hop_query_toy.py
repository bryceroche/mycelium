"""hop_query_toy.py -- T2h, IMPOSE THE SECOND HOP (2026-10-23; Bryce's word; Opus's learned-two-hop rival; ledger "T2h").
T2g (leak-free) put blind questions at 0.838 fully supervised vs 0.998 explicit: one query vector can't do two lookups in
sequence. Same episodes, seeds and generator as T2g (its owner-erased probe gate, B0, passed on exactly this generator).
Arms (all with RECORD labels only -- SUP_REC's aux; no query labels, which bought nothing three times):
  SUP_BOTH  T2g's best one-hop reader, retrained (the rival; record + query labels)
  HOP2_L    LEARNED two-hop (Opus): pick operand 2 (the OUT) first, read the picked record's state, update operand 1's
            query from [query, read] by a learned MLP, then pick operand 1. No owner decode, no snap.
  HOP2_I    IMPOSED hop: pick the OUT; read ITS owner channel; decode it against the owner codebook; snap (soft at train,
            hard at read); that code REPLACES operand 1's owner-channel query. Object and verb channels stay learned.
Mechanism (HOP2_I, blind): P(OUT pick right) x P(owner decoded right | pick right) x P(HAS pick right | both right).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import blind_query_toy as BQ  # noqa: E402
SS = BQ.SS

ARMS = ["SUP_BOTH", "HOP2_L", "HOP2_I"]
OUT_TXT = ".cache/hop_query{}.txt"


class HopNet(SS.Net):
    def __init__(self, mode):
        super().__init__("SUP_REC"); self.mode = mode                        # STRUCT channels + record-side aux only
        if mode == "L":
            self.upd = nn.Sequential(nn.Linear(2 * SS.D, 2 * SS.D), nn.GELU(), nn.Linear(2 * SS.D, SS.D))

    def chan(self, qs_o, rs):
        """qs_o: per-channel (B, c) query for one operand -> (B, R) scores (T2c's form)."""
        sc = 0
        for c in range(3):
            cs = torch.einsum("bd,brd->br", F.normalize(qs_o[c], dim=-1), F.normalize(rs[c], dim=-1))
            sc = sc + F.logsigmoid(self.kappa[c].exp() * (cs - self.theta[c]))
        return sc

    def forward(self, bt):
        tok = bt["tok"]; B, L = tok.shape; M = bt["rmask"]
        h = self.enc(self.emb(tok) + SS.sinus(L)[None], src_key_padding_mask=(tok == SS.T_PAD))
        r = h[torch.arange(B)[:, None], bt["rpos"]]
        q = self.qop(h[torch.arange(B), bt["qpos"]]).view(B, 2, SS.D)
        rs = torch.split(self.wk(r), self.CH, -1)
        q1s = torch.split(self.wq(q[:, 1]), self.CH, -1)
        sc1 = self.chan(q1s, rs).masked_fill(~M, -1e9)                      # operand 2: the OUT, picked first
        p1 = F.softmax(sc1, -1) if self.training else F.one_hot(sc1.argmax(-1), sc1.shape[-1]).float()
        if self.mode == "L":
            read = torch.einsum("br,brd->bd", p1, r)
            q0s = list(torch.split(self.wq(q[:, 0] + self.upd(torch.cat([q[:, 0], read], -1))), self.CH, -1))
        else:
            q0s = list(torch.split(self.wq(q[:, 0]), self.CH, -1))
            x = torch.einsum("br,brd->bd", p1, rs[0])                         # the picked record's owner channel
            book = F.normalize(self.cls[0], dim=-1)
            self._dl = F.normalize(x, dim=-1) @ book.T * 10
            pd = F.softmax(self._dl, -1) if self.training else F.one_hot(self._dl.argmax(-1), book.shape[0]).float()
            q0s[0] = pd @ book                                                # the snapped owner keys operand 1
        sc0 = self.chan(q0s, rs).masked_fill(~M, -1e9)
        self._sc1 = sc1
        aux = self.sup([torch.stack([q0s[c], q1s[c]], 1) for c in range(3)], rs, bt,
                       lambda x_, c: F.normalize(x_, dim=-1) @ F.normalize(self.cls[c], dim=-1).T * 10)
        return F.log_softmax(torch.stack([sc0, sc1], 1), -1), aux


def evaluate(net, bt):
    out = BQ.evaluate(net, bt)
    if isinstance(net, HopNet) and net.mode == "I":
        net.eval()
        with torch.no_grad():
            lp, _ = net(bt); pick = lp.argmax(-1); tg = bt["tgt"]; B = len(tg); ar = torch.arange(B)
            mb = torch.tensor([x == "blind" for x in bt["qform"]])
            out_owner = bt["rfeat"][ar, tg[:, 1], 0]
            p1 = pick[:, 1] == tg[:, 1]; dec = net._dl.argmax(-1) == out_owner; p0 = pick[:, 0] == tg[:, 0]
            out["f_pick_out"] = float(p1[mb].float().mean())
            out["f_decode_given"] = float(dec[mb & p1].float().mean())
            out["f_has_given"] = float(p0[mb & p1 & dec].float().mean())
        net.train()
    return out


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(90_000 + seed)
    ev = BQ.batch(erng, 6000 if SS.mode == "full" else SS.N_EVAL)
    torch.manual_seed(seed)
    net = SS.Net(arm) if arm == "SUP_BOTH" else HopNet(arm[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=SS.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=SS.LR, total_steps=SS.STEPS, pct_start=0.1)
    for step in range(SS.STEPS):
        bt = BQ.batch(rng, SS.BATCH); lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
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
        fac = (f" | factors: OUT pick {a['f_pick_out']:.3f} x owner decode {a['f_decode_given']:.3f} x HAS pick {a['f_has_given']:.3f} "
               f"= {a['f_pick_out'] * a['f_decode_given'] * a['f_has_given']:.3f}") if "f_pick_out" in a else ""
        say(f"{arm:8s} | BLIND {a['acc_blind']:.3f} (seeds {' / '.join(f'{x['acc_blind']:.3f}' for x in rs)}) | EXPLICIT {a['acc_explicit']:.3f}" + fac
            + f"  [{time.time() - t0:.0f}s]")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if SS.mode == "full":                                                    # THE PINNED BARS (ledger 2026-10-23, T2h)
        i, l_, b = A["HOP2_I"]["acc_blind"], A["HOP2_L"]["acc_blind"], A["SUP_BOTH"]["acc_blind"]
        say("VERDICTS (3-seed means, blind queries)")
        say("(H1) " + ("THE IMPOSED HOP SOLVES BLIND QUERIES" if i >= 0.95 else "THE IMPOSED HOP DOES NOT REACH 0.95") + f"  (HOP2_I {i:.3f} [bar >= 0.95])")
        say("(H2) " + ("THE IMPOSED HOP BEATS THE ONE-HOP READER BY >= 0.10" if i - b >= 0.10 else "THE IMPOSED HOP BEATS THE ONE-HOP READER BY < 0.10") + f"  (HOP2_I − SUP_BOTH = {i - b:+.3f})")
        say("(H3, Opus's separator) " + ("IMPOSED COMPOSITION BEATS A LEARNED SECOND HOP" if i - l_ >= 0.03 else "A LEARNED SECOND HOP DOES AS WELL -> 'GIVE THE QUERY A SECOND STEP'; IMPOSITION OPTIONAL")
            + f"  (HOP2_I {i:.3f} − HOP2_L {l_:.3f} = {i - l_:+.3f} [bar >= +0.03])")
    say(f"DONE in {time.time() - t0:.0f}s")
