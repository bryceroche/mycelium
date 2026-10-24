"""tree_role_autopsy.py -- THE TREE ARM'S HELD-OUT ROLE AUTOPSY (2026-10-24; Bryce's word; ledger "T2l term split"). Diagnostic, CPU.
role_autopsy.py's term split, ported to T2l's HYBRID_ROLE_TREE: the type term is now the slot's bound tree node (exact), so
~0.11 of held-out role pronouns that still miss are carried by some OTHER term. Training is tree_type_toy.run VERBATIM (its
evaluate() is wrapped to keep the net and the eval batch); on the held-out pron_role cell the resolver logit is split into
overlap / kind / type / rank / learned / role and, per ERROR, each term's margin (wrong pick - gold slot) is read.
Criteria pinned in the ledger (2026-10-24, T2l term split) BEFORE the run:
  SINGLE CULPRIT X  if X pushes wrong in >= 0.70 of errors AND X's margin alone exceeds the role term's gold-favor in >= 0.50
  TWO-TERM X+Y      else, if the top-two terms' summed margin exceeds the role term's gold-favor in >= 0.70 of errors
  DIFFUSE           otherwise
  VOID              if the re-run's held-out role acc misses the banked 0.894 by > 0.03
  INTEGRITY         the term split must reproduce the resolver's argmax; type must push wrong in 0.000 of errors (tree is exact)."""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.argv = [sys.argv[0]]                                                   # full-mode constants from the toy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tree_type_toy as K

SMOKE = os.environ.get("AUTOPSY_SMOKE") == "1"                            # plumbing check only: 1 seed, BM_STEPS
OUT = ".cache/tree_role_autopsy{}.txt".format("_smoke" if SMOKE else "")
if SMOKE: K.SEEDS = [0]
ARM = "HYBRID_ROLE_TREE"
BANKED = 0.894
TERMS = ["overlap", "kind", "type", "rank", "learned", "role"]


def terms(net, bt):
    """-> {term: (B,R,S)}, reproducing Net.resolve for HYBRID_ROLE_TREE on record mentions."""
    E = net.emb; tok = bt["tok"]; L = tok.shape[1]
    h = net.enc(E(tok) + K.sinus(L)[None], src_key_padding_mask=(tok == K.T_PAD))
    hs = torch.einsum("bsl,bld->bsd", bt["sspan"], h)
    cont, title, pron, rank = bt["mcont"], bt["mtitle"], bt["mpron"], bt["rank"]
    ec = F.normalize(E(cont.clamp(min=0)), dim=-1) * (cont >= 0)[..., None]
    ek = F.normalize(E(bt["skeys"].clamp(min=0)), dim=-1) * (bt["skeys"] >= 0)[..., None]
    ov = torch.einsum("bxcd,bskd->bxsck", ec, ek).amax(-1).sum(-1)
    isp = (pron >= 0)[..., None].float()
    pt = (bt["stype"][:, None, :] == pron[..., None]).float()             # the bound tree node
    tt = (bt["stype"][:, None, :] == (title - 1)[..., None]).float()
    return dict(overlap=net.a * ov,
                kind=net.kb[bt["skind"]][:, None, :].expand_as(ov),
                type=isp * net.wt[0] * pt + (title > 0)[..., None] * net.wt[1] * tt,
                rank=isp * net.rho[rank],
                learned=net.learned(bt["rspan"], h, hs),
                role=isp * (net.wr[0] * bt["rsubj"] + net.wr[1] * bt["rrecp"]))


def main():
    assert SMOKE or not os.path.exists(OUT), "refusing to overwrite a banked read"
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    f = open(OUT, "w")
    say = lambda s: (print(s, flush=True), f.write(s + "\n"), f.flush())
    kept = {}
    orig_eval = K.evaluate

    def keep_eval(net, ev):
        kept["net"], kept["ev"] = net, ev
        return orig_eval(net, ev)
    K.evaluate = keep_eval
    agg = dict(n_err=0, n=0, role_favors_gold=0, role_favors_wrong=0, wrong_is_prev_recp=0, margins={t: [] for t in TERMS},
               role_m=[], seeds=[])
    for seed in K.SEEDS:
        res = K.run(ARM, seed); net, ev = kept["net"], kept["ev"]["heldout"]
        net.eval()
        with torch.no_grad():
            net(ev); full = net._rl
            T = terms(net, ev)
            tot = sum(T.values()).masked_fill(~ev["smask"][:, None, :], -1e9)
            cell = ev["rmask"] & (ev["rform"] == K.CELLS.index("pron_role"))
            pick = full.argmax(-1); gold = ev["rslot"]
            assert torch.equal(tot.argmax(-1)[cell], pick[cell]), "term split does not reproduce the resolver"
            acc = float((pick == gold)[cell].float().mean())
            assert abs(acc - res["heldout_res_pron_role"]) < 1e-6, "autopsy eval differs from the toy's own evaluate()"
            say(f"seed {seed}: held-out role acc {acc:.4f} (n {int(cell.sum())}); wr {[round(x, 3) for x in net.wr.tolist()]} "
                f"wt {[round(x, 3) for x in net.wt.tolist()]} a {float(net.a):.2f} learned scale {float(net.ls.exp()):.2f}")
            agg["seeds"].append(dict(seed=seed, acc=acc, n=int(cell.sum())))
            err = cell & (pick != gold)
            for bi, ri in zip(*[x.tolist() for x in err.nonzero(as_tuple=True)]):
                w, g = int(pick[bi, ri]), int(gold[bi, ri])
                mg = {t: float(T[t][bi, ri, w] - T[t][bi, ri, g]) for t in TERMS}
                for t in TERMS: agg["margins"][t].append(mg[t])
                agg["role_m"].append(mg["role"]); agg["n_err"] += 1
                agg["role_favors_gold"] += mg["role"] < 0; agg["role_favors_wrong"] += mg["role"] > 0
                agg["wrong_is_prev_recp"] += bool(ev["rrecp"][bi, ri, w] > 0)
            agg["n"] += int(cell.sum())
    ne = max(1, agg["n_err"]); acc = 1 - agg["n_err"] / max(1, agg["n"])
    M = {t: np.array(agg["margins"][t]) for t in TERMS}; gf = -np.array(agg["role_m"])   # role's gold-favor (>0 = favors gold)
    say(f"\nHELD-OUT ROLE ERRORS: {agg['n_err']} of {agg['n']} (acc {acc:.3f}; banked {BANKED})")
    say(f"role favors GOLD in {agg['role_favors_gold'] / ne:.3f} of errors; WRONG in {agg['role_favors_wrong'] / ne:.3f}; "
        f"the wrong pick is the previous RECIPIENT in {agg['wrong_is_prev_recp'] / ne:.3f}")
    say("per term over errors: mean margin (wrong - gold) | pushes-wrong share | alone outvotes role share:")
    alone = {}
    for t in TERMS:
        alone[t] = float((M[t] > gf).mean()) if t != "role" else float("nan")
        say(f"   {t:8s} mean {M[t].mean():+.3f} | pushes-wrong {float((M[t] > 0).mean()):.3f} | alone {alone[t]:.3f}")
    cand = [t for t in TERMS if t != "role"]
    top = sorted(cand, key=lambda t: M[t].mean(), reverse=True)
    single = [t for t in cand if (M[t] > 0).mean() >= 0.70 and alone[t] >= 0.50]
    two = float((M[top[0]] + M[top[1]] > gf).mean())
    integ = float((M["type"] > 0).mean()) == 0.0
    void = abs(acc - BANKED) > 0.03
    if void: v = "VOID (re-run misses the banked read)"
    elif single: v = f"SINGLE CULPRIT = {max(single, key=lambda t: M[t].mean())}"
    elif two >= 0.70: v = f"TWO-TERM = {top[0]}+{top[1]} (summed outvotes role in {two:.3f})"
    else: v = f"DIFFUSE (top two {top[0]}+{top[1]} outvote role in {two:.3f})"
    say(f"INTEGRITY: type pushes wrong in {float((M['type'] > 0).mean()):.3f} of errors -> {'OK' if integ else 'FLAG'}")
    say(f"VERDICT: {v}")
    say("COORDINATOR PREDICTION (pinned): SINGLE CULPRIT = learned")
    json.dump(dict(agg, margins={t: list(map(float, v)) for t, v in agg["margins"].items()}, role_m=list(map(float, agg["role_m"])),
                   verdict=v), open(OUT.replace(".txt", ".json"), "w"))


if __name__ == "__main__":
    main()
