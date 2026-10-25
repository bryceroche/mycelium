"""tree_volume_toy.py -- T2r, THE VOLUME TEST: WHICH EVIDENCE MAY GROW THE TREE? (2026-10-25; Bryce's word; ledger "T2r"). CPU.
Bryce: "we would get WAY more data if we allow the tree to update on each data point that is connected to a logic chain that
gets the correct answer instead of only rewarding confirmed data points. the welford stats should statistically drown out
the noise." And: "should we let the AI save its own internal hunches?"
T2p's machinery (tree_train_toy: the tree as a mandatory road, Welford leaves, decayed BIRCH CFs, the walk's first-sight
prior; the tree starts EMPTY, so early nodes are the untrained walk's guesses) with the training-time CERTIFIER varied:
  GOLD_RES     T2p: title + pronoun evidence, resolved by the episode's GOLD resolution (the clean judge; re-run)
  ANSWER_ALL   Bryce's volume: title + pronoun evidence resolved by the RESOLVER'S OWN PICK, from every mention of every
               episode whose FINAL ANSWER (both query hops) the model got right
  ANSWER_PATH  ANSWER_ALL restricted to mentions on the answer path (the two records the question reads; a structural proxy
               for a counterfactual flip, disclosed)
  ANY_ALL      no certification at all: resolver-picked evidence from every episode (what the answer filter is worth)
  TIERS        hard tier = gold-resolution evidence on a random 10 % of episodes; soft tier = ANSWER_ALL on all;
               a row decides by hard when it has any, else soft
  HUNCH        ANSWER_ALL + the walk's posterior saved into the row as an event on EVERY sighting (the AI's own hunches)
Test-time evidence for every arm = T2p's TITLES mode (scored) and PRIOR (read)."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

_argv = sys.argv[:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tree_train_toy as T
sys.argv = _argv
O, K = T.O, T.K

ARMS = os.environ.get("TR_ARMS", "GOLD_RES,ANSWER_ALL,ANSWER_PATH,ANY_ALL,TIERS,HUNCH").split(",")
OUT_TXT = ".cache/tree_volume{}.txt"
S, CHECK = K.S, (50, 200, 1000)
HELD = T.HELD


def truth(t):
    if K.O_F <= t < K.O_F + K.NF:
        return 0 if t - K.O_F < 8 else 1
    return 2 if K.O_GH <= t < K.O_GH + K.NG else 3


class Tree2(T.Tree):
    """T2p's tree with a SOFT tier per row (used by TIERS); node = hard if any hard evidence, else soft, else prior."""
    def node(self, tok):
        lf = self.leaf[tok]
        if lf["n"] > 0:
            return int(lf["mean"].argmax())
        if lf.get("sn", 0) > 0:
            return int(lf["smean"].argmax())
        return int(lf["prior"].argmax())

    def add(self, tok, vec, soft=False):
        lf = self.leaf[tok]; a, b = ("sn", "smean") if soft else ("n", "mean")
        lf[a] = lf.get(a, 0) + 1; m = lf.get(b, np.zeros(4)); lf[b] = m + (vec - m) / lf[a]

    def copy(self):
        t = Tree2(); t.leaf = {k: {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()} for k, v in self.leaf.items()}
        t.N, t.LS, t.SS = self.N.copy(), self.LS.copy(), self.SS.copy()
        return t


def certified(tree, bt):
    tok, sm = bt["sptok"].numpy(), bt["smask"].numpy(); lab = np.full(tok.shape, -1)
    for b, j in zip(*np.nonzero(sm)):
        lf = tree.leaf.get(int(tok[b, j]))
        if lf is not None and (lf["n"] > 0 or lf.get("sn", 0) > 0):
            lab[b, j] = tree.node(int(tok[b, j]))
    return torch.from_numpy(lab)


def events(bt, pick, allow):
    """(b, mention, slot, type) for title / pronoun mentions allowed by the (B,R) mask."""
    M, ti, pr = (bt["rmask"] & allow).numpy(), bt["mtitle"].numpy(), bt["mpron"].numpy(); out = []
    for b, i in zip(*np.nonzero(M)):
        k = ti[b, i] - 1 if ti[b, i] > 0 else pr[b, i]
        if k >= 0:
            out.append((b, i, int(pick[b, i]), int(k)))
    return out


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    nrng = np.random.default_rng(77_000 + seed); envg = np.random.default_rng(93_000 + seed)
    trng = np.random.default_rng(55_000 + seed); tev = np.random.default_rng(57_000 + seed); hrng = np.random.default_rng(66_000 + seed)
    ev = {"indist": O.batch(erng, K.N_EVAL, "train", 0.0, envg, 0.0, False, tev),
          "heldout": O.batch(erng, K.N_EVAL, "heldout", 0.0, envg, 0.0, False, tev)}
    torch.manual_seed(seed); net = O.Net("SEARCH"); tree = Tree2()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=K.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=K.STEPS, pct_start=0.1)
    r = dict(ev_n=0, ev_ok=0, hunch_n=0, hunch_disagree=0, wrong_seen=0, slots=0, answer_right=0, episodes=0)
    for step in range(K.STEPS):
        bt = O.batch(rng, K.BATCH, "train", 0.0, nrng, 0.0, False, trng)
        gold = bt["stype"]; bt = dict(bt, stype=T.read_nodes(net, tree, bt))
        r["wrong_seen"] += int((bt["stype"] != gold)[bt["smask"]].sum()); r["slots"] += int(bt["smask"].sum())
        lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
        loss = F.nll_loss(lp[m], tg[m]) + aux
        lab = certified(tree, bt); loss = loss + T.proto_loss(net, lab)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        T.update_cf(net, tree, lab)
        with torch.no_grad():                                               # the certifiers (all after the step, as in T2p)
            right = ((lp.argmax(-1) == tg) | ~m).all(-1)                     # the final answer: both query hops
            r["answer_right"] += int(right.sum()); r["episodes"] += len(right)
            pick = net._rl.argmax(-1).numpy(); B, R = bt["rmask"].shape; tok = bt["sptok"].numpy(); g = gold.numpy()
            ok_ep = right[:, None].expand(B, R)
            path = torch.zeros(B, R, dtype=torch.bool); path[torch.arange(B)[:, None], tg.clamp(min=0)] = True
            full = torch.ones(B, R, dtype=torch.bool)
            hard_ep = torch.from_numpy(hrng.random(B) < 0.10)[:, None].expand(B, R)
            plan = {"GOLD_RES": [(bt["rslot"].numpy(), full, False)],
                    "ANSWER_ALL": [(pick, ok_ep, False)], "HUNCH": [(pick, ok_ep, False)],
                    "ANSWER_PATH": [(pick, ok_ep & path, False)],
                    "ANY_ALL": [(pick, full, False)],
                    "TIERS": [(bt["rslot"].numpy(), hard_ep, False), (pick, ok_ep, True)]}[arm]
            for pk, allow, soft in plan:
                for b, i, j, k in events(bt, pk, allow):
                    t = int(tok[b, j]); tree.add(t, np.eye(4)[k], soft)
                    r["ev_n"] += 1; r["ev_ok"] += int(k == g[b, j])
            if arm == "HUNCH":                                              # the AI's own hunches, saved as events
                post = net.walk(net._v).numpy()
                for b, j in zip(*np.nonzero(bt["smask"].numpy())):
                    t = int(tok[b, j]); tree.add(t, post[b, j]); r["hunch_n"] += 1
                    r["hunch_disagree"] += int(post[b, j].argmax() != g[b, j])
        if step + 1 in CHECK:
            r[f"rows_wrong_{step + 1}"] = sum(int(tree.node(t) != truth(t)) for t in tree.leaf)
    r["rows_wrong_end"] = sum(int(tree.node(t) != truth(t)) for t in tree.leaf); r["rows"] = len(tree.leaf)
    r["ev_precision"] = r["ev_ok"] / max(1, r["ev_n"]); r["train_wrong_node_rate"] = r["wrong_seen"] / max(1, r["slots"])
    r["answer_right_rate"] = r["answer_right"] / max(1, r["episodes"])
    r["hunch_disagree_rate"] = r["hunch_disagree"] / max(1, r["hunch_n"])
    gl = {sp: bt["stype"].clone() for sp, bt in ev.items()}
    for mode in ("TITLES", "PRIOR"):
        t = tree.copy(); evs = {}
        for sp, bt in ev.items():
            if mode == "PRIOR" and sp == "indist":
                continue
            st, _ = T.stream_eval(net, t, bt, "TITLES" if mode == "TITLES" else "NONE")
            held = torch.isin(bt["sptok"], HELD) & bt["smask"]
            r[f"{mode}_{sp}_node_heldname"] = float((st == gl[sp])[held].float().mean()) if held.any() else float("nan")
            evs[sp] = dict(bt, stype=st)
        res = K.evaluate(net, evs)
        r.update(res if mode == "TITLES" else {f"PRIOR_{k}": v for k, v in res.items()})
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    OUT = OUT_TXT.format("_smoke" if K.mode == "smoke" else "")
    assert K.mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    T.Tree = Tree2                                                          # stream_eval / read_nodes insert into Tree2 rows
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {K.mode} steps {K.STEPS} seeds {K.SEEDS} arms {ARMS}")
    for arm in ARMS:
        rs = [run(arm, s) for s in K.SEEDS]
        ag = {kk: float(np.nanmean([x[kk] for x in rs])) for kk in rs[0] if all(kk in x for x in rs)}
        A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        for sp in ("indist", "heldout"):
            say(f"{arm:11s} {sp:7s} | role {ag[f'{sp}_res_pron_role']:.3f} (seeds {[round(x[f'{sp}_res_pron_role'], 3) for x in rs]}) | all "
                f"{ag[f'{sp}_res_all']:.3f} | blind {ag[f'{sp}_acc_blind']:.3f} explicit {ag[f'{sp}_acc_explicit']:.3f}")
            say("        cells: " + " ".join(f"{c} {ag[f'{sp}_res_{c}']:.3f}" for c in K.CELLS) + f"  [{time.time() - t0:.0f}s]")
        say(f"        EVIDENCE admitted {ag['ev_n']:.0f} precision {ag['ev_precision']:.4f} | answer right {ag['answer_right_rate']:.3f} | "
            f"wrong rows at step 50/200/1000/end " + "/".join(str([x.get(k) for x in rs]) for k in ("rows_wrong_50", "rows_wrong_200", "rows_wrong_1000", "rows_wrong_end"))
            + f" of {ag['rows']:.0f} | resolver saw wrong node {ag['train_wrong_node_rate']:.4f}")
        if arm == "HUNCH":
            say(f"        HUNCHES saved {ag['hunch_n']:.0f}, disagreeing with the truth {ag['hunch_disagree_rate']:.4f}")
        say(f"        held-out names node: TITLES {ag['TITLES_heldout_node_heldname']:.3f} PRIOR {ag['PRIOR_heldout_node_heldname']:.3f} | "
            f"PRIOR held-out role {ag['PRIOR_heldout_res_pron_role']:.3f}")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if K.mode == "full" and all(x in A for x in ("GOLD_RES", "ANSWER_ALL")):  # PINNED (ledger 2026-10-25, T2r)
        g, a = A["GOLD_RES"], A["ANSWER_ALL"]
        say("VERDICTS (3-seed means)")
        say(f"CONTROL: GOLD_RES held-out role {g['heldout_res_pron_role']:.3f} (T2p banked 0.996) -> "
            f"{'OK' if abs(g['heldout_res_pron_role'] - 0.996) <= 0.02 else 'VOID'}")
        worst = min(a[f"indist_res_{k}"] - g[f"indist_res_{k}"] for k in K.CELLS)
        endw = [x["rows_wrong_end"] for x in allres[ARMS.index("ANSWER_ALL")]["seeds"]]
        v1 = sum(endw) == 0 and a["heldout_res_pron_role"] >= g["heldout_res_pron_role"] - 0.01 and worst >= -0.01
        say(f"V1 FINAL-ANSWER VOLUME GROWS A CLEAN TREE: {'PASS' if v1 else 'FAIL'} (ANSWER_ALL wrong rows at end {endw}; held-out role "
            f"{a['heldout_res_pron_role']:.3f} vs {g['heldout_res_pron_role']:.3f}; worst in-dist cell {worst:+.3f}; "
            f"evidence {a['ev_n']:.0f} at precision {a['ev_precision']:.4f} vs GOLD_RES {g['ev_n']:.0f} at {g['ev_precision']:.4f})")
    say(f"DONE in {time.time() - t0:.0f}s")
