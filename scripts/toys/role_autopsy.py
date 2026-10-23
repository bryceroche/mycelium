"""role_autopsy.py -- THE HELD-OUT ROLE AUTOPSY (2026-10-23; Bryce's word; ledger 15:21). Diagnostic, CPU.
Retrains HYBRID_ROLE (T2k recipe, seeds 0-2, same episodes), then on the HELD-OUT role-pronoun cell splits the resolver logit
into its terms (overlap / kind bias / pronoun type read / recency rank / LEARNED context / role flag) and, per ERROR, reads each
term's margin (wrong pick - gold slot). Criteria pinned in the ledger: OUTVOTE CONFIRMED if in >= 0.70 of errors the role term
favors gold AND the other terms' summed margin exceeds it; NOT CONFIRMED if the role term favors the wrong slot in >= 0.30."""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.argv = [sys.argv[0]]                                                   # full-mode constants from the toy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import role_structure_toy as K

OUT = ".cache/role_autopsy.txt"
assert not os.path.exists(OUT), "refusing to overwrite a banked read"
TERMS = ["overlap", "kind", "type", "rank", "learned", "role"]


def terms(net, bt):
    """-> (B,R,S) logits per term, reproducing Net.resolve for HYBRID_ROLE on record mentions."""
    E = net.emb; tok = bt["tok"]; L = tok.shape[1]
    h = net.enc(E(tok) + K.sinus(L)[None], src_key_padding_mask=(tok == K.T_PAD))
    hs = torch.einsum("bsl,bld->bsd", bt["sspan"], h)
    cont, title, pron, rank = bt["mcont"], bt["mtitle"], bt["mpron"], bt["rank"]
    ec = F.normalize(E(cont.clamp(min=0)), dim=-1) * (cont >= 0)[..., None]
    ek = F.normalize(E(bt["skeys"].clamp(min=0)), dim=-1) * (bt["skeys"] >= 0)[..., None]
    ov = torch.einsum("bxcd,bskd->bxsck", ec, ek).amax(-1).sum(-1)
    tl = net.typ(E(bt["sptok"]))
    S = K.S; isp = (pron >= 0)[..., None].float()
    pt = torch.gather(tl[:, None].expand(-1, pron.shape[1], -1, -1), 3, pron.clamp(min=0)[..., None, None].expand(-1, -1, S, 1))[..., 0]
    tt = torch.gather(tl[:, None].expand(-1, title.shape[1], -1, -1), 3, (title - 1).clamp(min=0)[..., None, None].expand(-1, -1, S, 1))[..., 0]
    out = dict(overlap=net.a * ov,
               kind=net.kb[bt["skind"]][:, None, :].expand_as(ov),
               type=isp * net.wt[0] * pt + (title > 0)[..., None] * net.wt[1] * tt,
               rank=isp * net.rho[rank],
               learned=net.learned(bt["rspan"], h, hs),
               role=isp * (net.wr[0] * bt["rsubj"] + net.wr[1] * bt["rrecp"]))
    return out


def main():
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    f = open(OUT, "w")
    say = lambda s: (print(s, flush=True), f.write(s + "\n"), f.flush())
    agg = dict(n_err=0, n=0, role_favors_gold=0, outvoted=0, role_favors_wrong=0, margins={t: [] for t in TERMS})
    for seed in (0, 1, 2):
        # the T2k training loop, verbatim, keeping the net
        rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
        envg = np.random.default_rng(93_000 + seed); nrng = np.random.default_rng(77_000 + seed)
        ev_in = K.batch(erng, K.N_EVAL, "train", 0.0, envg); ev = K.batch(erng, K.N_EVAL, "heldout", 0.0, envg)
        torch.manual_seed(seed); net = K.Net("HYBRID_ROLE")
        opt = torch.optim.AdamW(net.parameters(), lr=K.LR, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=K.STEPS, pct_start=0.1)
        for step in range(K.STEPS):
            bt = K.batch(rng, K.BATCH, "train", 0.0, nrng); lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
            loss = F.nll_loss(lp[m], tg[m]) + aux
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            net(ev); full = net._rl
            T = terms(net, ev)
            tot = sum(T.values()).masked_fill(~ev["smask"][:, None, :], -1e9)
            assert torch.allclose(tot.argmax(-1), full.argmax(-1)), "term split does not reproduce the resolver"
            cell = ev["rmask"] & (ev["rform"] == K.CELLS.index("pron_role"))
            pick = full.argmax(-1); gold = ev["rslot"]
            acc = float((pick == gold)[cell].float().mean())
            say(f"seed {seed}: held-out role acc {acc:.3f} (n {int(cell.sum())}); weights wr {net.wr.tolist()} a {float(net.a):.2f} "
                f"wt {net.wt.tolist()}")
            err = cell & (pick != gold)
            b, r = err.nonzero(as_tuple=True)
            for bi, ri in zip(b.tolist(), r.tolist()):
                w, g = int(pick[bi, ri]), int(gold[bi, ri])
                mg = {t: float(T[t][bi, ri, w] - T[t][bi, ri, g]) for t in TERMS}
                for t in TERMS: agg["margins"][t].append(mg[t])
                others = sum(mg[t] for t in TERMS if t != "role")
                agg["n_err"] += 1
                if mg["role"] < 0:
                    agg["role_favors_gold"] += 1
                    if others > -mg["role"]: agg["outvoted"] += 1
                elif mg["role"] > 0:
                    agg["role_favors_wrong"] += 1
            agg["n"] += int(cell.sum())
    ne = max(1, agg["n_err"])
    say(f"\nHELD-OUT ROLE ERRORS: {agg['n_err']} of {agg['n']} (acc {1 - agg['n_err'] / max(1, agg['n']):.3f}; banked 0.703)")
    say(f"role term favors GOLD in {agg['role_favors_gold'] / ne:.3f} of errors; OUTVOTED there in {agg['outvoted'] / ne:.3f}; "
        f"role favors the WRONG slot in {agg['role_favors_wrong'] / ne:.3f}")
    say("mean margin (wrong - gold) per term over errors; share of errors where the term pushes wrong:")
    for t in TERMS:
        a = np.array(agg["margins"][t])
        say(f"   {t:8s} mean {a.mean():+.3f}  pushes-wrong {float((a > 0).mean()):.3f}")
    void = abs((1 - agg["n_err"] / max(1, agg["n"])) - 0.703) > 0.03
    conf = agg["outvoted"] / ne >= 0.70
    notc = agg["role_favors_wrong"] / ne >= 0.30
    culprit = max((t for t in TERMS if t != "role"), key=lambda t: np.mean(agg["margins"][t]))
    say(f"VERDICT: {'VOID (re-run misses the banked read)' if void else ('OUTVOTE CONFIRMED' if conf and not notc else 'NOT CONFIRMED')}"
        f"; culprit (largest mean wrong-push) = {culprit}")
    json.dump(dict(agg, margins={t: list(map(float, v)) for t, v in agg["margins"].items()}), open(OUT.replace(".txt", ".json"), "w"))


if __name__ == "__main__":
    main()
