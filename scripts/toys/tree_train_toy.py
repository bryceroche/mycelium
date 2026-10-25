"""tree_train_toy.py -- T2p, THE TREE IN TRAINING, GROWN ON HARD EVIDENCE ONLY (2026-10-25; Bryce: "we only update the welford
stats and birch math on hard evidence. the tree must be included in the training"; ledger "T2p"). CPU.
T2o's world (tree_search_toy: real gear stamps in the key, the PLANE term, the walk) with the tree made a MANDATORY ROAD:
  - The resolver NEVER sees a gold node, in training or at test. Its node is the tree's: the slot's head token is looked up
    in the literal leaves; a token never seen before is INSERTED with the walk's posterior as its FIRST-SIGHT PRIOR.
  - A leaf's node = argmax of its HARD-EVIDENCE statistic (exact running mean of evidence one-hots, count-based Welford) once
    it has any evidence; until then, its first-sight prior. The walk's re-reads are never evidence (T2o's lesson: averaging
    the same biased read locked a misplaced name in).
  - HARD EVIDENCE = a gendered title (MR/MS) or a typed pronoun (he/she/they/it) whose mention is resolved to the slot.
    Training: resolved by the episode's gold resolution (the answer key gates training data). Test: resolved by the
    resolver's own pick; scored mode TITLES = title evidence only (the surname overlap carries the pick); read mode TP =
    titles + pronouns picked with probability >= 0.9 (circular: the pick already used the node; disclosed).
  - The interior centroids are BIRCH cluster features (N, LS, SS), decayed per step because reader coordinates drift,
    updated ONLY from slots whose leaf is evidence-certified, labelled by the leaf's evidence node; the search head's
    prototype loss uses the same certified labels.
  - Streaming: training batches and eval chunks (100 episodes) are processed in order; nodes are read from the tree state
    before the chunk, evidence is written after it.
Arms: GEARS (control: T2o's GEARS, gold node) / TREE (the above)."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

_argv = sys.argv[:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tree_search_toy as O                                               # imports tree_type_toy / plane_term_toy; argv-safe
sys.argv = _argv
K = O.K

ARMS = os.environ.get("TP_ARMS", "GEARS,TREE").split(",")
OUT_TXT = ".cache/tree_train{}.txt"
S, PERSON, LAMBDA, CHUNK, PCERT = K.S, O.PERSON, 0.95, 100, 0.9
HELD = torch.tensor(sorted(O.HELD_TOK))


class Tree:
    """The permanent tree: literal leaves {token: prior, n, mean} + BIRCH CFs for the 5 interior/type nodes."""
    def __init__(self):
        self.leaf = {}; self.N = np.zeros(5); self.LS = np.zeros((5, O.DS)); self.SS = np.zeros(5)

    def copy(self):
        t = Tree(); t.leaf = {k: dict(v) for k, v in self.leaf.items()}; t.N, t.LS, t.SS = self.N.copy(), self.LS.copy(), self.SS.copy()
        return t

    def node(self, tok):
        lf = self.leaf[tok]
        return int(lf["mean"].argmax()) if lf["n"] > 0 else int(lf["prior"].argmax())

    def evidence(self, tok, k):
        lf = self.leaf[tok]; e = np.eye(4)[k]; lf["n"] += 1; lf["mean"] = lf["mean"] + (e - lf["mean"]) / lf["n"]   # Welford


def walk_post(net, bt):
    net_was = net.training; net.eval()
    with torch.no_grad():
        net(bt)
    p = net.walk(net._v).numpy(); net.train(net_was)
    return p


def read_nodes(net, tree, bt):
    """INSERT unseen tokens (first-sight prior from the walk), then read every slot's node from the tree."""
    tok, sm = bt["sptok"].numpy(), bt["smask"].numpy()
    if any(int(t) not in tree.leaf for t in tok[sm]):
        post = walk_post(net, bt)
        for b, j in zip(*np.nonzero(sm)):
            t = int(tok[b, j])
            if t not in tree.leaf:
                tree.leaf[t] = dict(prior=post[b, j], n=0, mean=np.zeros(4))
    st = np.full(tok.shape, -1)
    for b, j in zip(*np.nonzero(sm)):
        st[b, j] = tree.node(int(tok[b, j]))
    return torch.from_numpy(st)


def write_evidence(tree, bt, pick, prob, mode):
    """Hard evidence from resolved title / pronoun mentions. pick (B,R) slot, prob (B,R) or None (gold = certified)."""
    M, ti, pr, tok = bt["rmask"].numpy(), bt["mtitle"].numpy(), bt["mpron"].numpy(), bt["sptok"].numpy()
    n = 0
    for b, i in zip(*np.nonzero(M)):
        k = -1
        if mode == "NONE":
            break
        if ti[b, i] > 0:
            k = ti[b, i] - 1
        elif mode == "TP" and pr[b, i] >= 0 and (prob is None or prob[b, i] >= PCERT):
            k = pr[b, i]
        if k >= 0:
            tree.evidence(int(tok[b, pick[b, i]]), int(k)); n += 1
    return n


def certified(tree, bt):
    tok, sm = bt["sptok"].numpy(), bt["smask"].numpy(); lab = np.full(tok.shape, -1)
    for b, j in zip(*np.nonzero(sm)):
        lf = tree.leaf.get(int(tok[b, j]))
        if lf is not None and lf["n"] > 0:
            lab[b, j] = int(lf["mean"].argmax())
    return torch.from_numpy(lab)


def proto_loss(net, lab):
    v = net._v; ok = lab >= 0
    if not ok.any():
        return 0.0
    t1 = torch.where(lab < 2, 0, lab - 1)
    l = F.cross_entropy(-O.SC * ((v[..., None, :] - net.mu[O.L1]) ** 2).sum(-1)[ok], t1[ok])
    mp = ok & (lab < 2)
    return l + (F.cross_entropy(-O.SC * ((v[..., None, :] - net.mu[[0, 1]]) ** 2).sum(-1)[mp], lab[mp]) if mp.any() else 0.0)


@torch.no_grad()
def update_cf(net, tree, lab):
    """BIRCH CF update from certified slots only (decayed), then the search centroids = LS / N."""
    v = net._v.detach().numpy(); lb = lab.numpy()
    tree.N *= LAMBDA; tree.LS *= LAMBDA; tree.SS *= LAMBDA
    for c in range(5):
        mm = (lb >= 0) & ((lb < 2) if c == PERSON else (lb == c))
        if mm.any():
            x = v[mm]; tree.N[c] += len(x); tree.LS[c] += x.sum(0); tree.SS[c] += (x ** 2).sum()
    for c in range(5):
        if tree.N[c] > 0:
            net.mu[c] = torch.from_numpy(tree.LS[c] / tree.N[c]).float()


def sub(bt, a, z):
    return {k: (v[a:z] if torch.is_tensor(v) else v[a:z]) for k, v in bt.items()}


def stream_eval(net, tree, bt, mode):
    """Stream the split in chunks: read nodes from the tree, resolve, write test-time evidence (resolver picks)."""
    B = bt["tok"].shape[0]; st = torch.full((B, S), -1, dtype=torch.long); nev = 0
    net.eval()
    for a in range(0, B, CHUNK):
        c = sub(bt, a, a + CHUNK)
        s_ = read_nodes(net, tree, c); st[a:a + CHUNK] = s_
        c = dict(c, stype=s_)
        with torch.no_grad():
            net(c)
        p = F.softmax(net._rl, -1); nev += write_evidence(tree, c, p.argmax(-1).numpy(), p.amax(-1).numpy(), mode)
    net.train()
    return st, nev


def run(arm, seed):
    rng = np.random.default_rng(seed); erng = np.random.default_rng(91_000 + seed)
    nrng = np.random.default_rng(77_000 + seed); envg = np.random.default_rng(93_000 + seed)
    trng = np.random.default_rng(55_000 + seed); tev = np.random.default_rng(57_000 + seed)
    ev = {"indist": O.batch(erng, K.N_EVAL, "train", 0.0, envg, 0.0, False, tev),
          "heldout": O.batch(erng, K.N_EVAL, "heldout", 0.0, envg, 0.0, False, tev)}
    torch.manual_seed(seed); net = O.Net("SEARCH" if arm == "TREE" else arm); tree = Tree()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=K.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=K.STEPS, pct_start=0.1)
    wrong_train = 0; nslots = 0
    for step in range(K.STEPS):
        bt = O.batch(rng, K.BATCH, "train", 0.0, nrng, 0.0, False, trng)
        if arm == "TREE":
            gold = bt["stype"]; bt = dict(bt, stype=read_nodes(net, tree, bt))          # the mandatory road: tree nodes only
            wrong_train += int((bt["stype"] != gold)[bt["smask"]].sum()); nslots += int(bt["smask"].sum())
        lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
        loss = F.nll_loss(lp[m], tg[m]) + aux
        if arm == "TREE":
            lab = certified(tree, bt); loss = loss + proto_loss(net, lab)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if arm == "TREE":
            update_cf(net, tree, lab)
            write_evidence(tree, bt, bt["rslot"].numpy(), None, "TP")                    # certified by the gold resolution
    if arm != "TREE":
        return K.evaluate(net, ev)
    r = dict(train_wrong_node_rate=wrong_train / max(1, nslots), train_leaves=len(tree.leaf),
             train_leaves_wrong=sum(int(tree.node(t) != (0 if t - K.O_F < 8 else 1)) for t in tree.leaf if K.O_F <= t < K.O_F + K.NF))
    gold = {sp: bt["stype"].clone() for sp, bt in ev.items()}
    for sp, bt in ev.items():                                               # the walk alone (T2o's S1 read), every slot a miss
        wk = torch.from_numpy(walk_post(net, bt).argmax(-1)); held = torch.isin(bt["sptok"], HELD) & bt["smask"]
        r[f"{sp}_walk_all"] = float((wk == gold[sp])[bt["smask"]].float().mean())
        r[f"{sp}_walk_heldname"] = float((wk == gold[sp])[held].float().mean()) if held.any() else float("nan")
    for mode in ("TITLES", "TP", "PRIOR"):
        t = tree.copy(); evs = {}
        for sp, bt in ev.items():
            if mode != "TITLES" and sp == "indist":
                continue
            st, nev = stream_eval(net, t, bt, mode if mode != "PRIOR" else "NONE")
            held = torch.isin(bt["sptok"], HELD) & bt["smask"]
            r[f"{mode}_{sp}_node_heldname"] = float((st == gold[sp])[held].float().mean()) if held.any() else float("nan")
            r[f"{mode}_{sp}_node_all"] = float((st == gold[sp])[bt["smask"]].float().mean())
            r[f"{mode}_{sp}_evidence"] = nev
            r[f"{mode}_{sp}_names"] = [(int(k) - K.O_F, t.node(k), t.leaf[k]["n"]) for k in sorted(O.HELD_TOK) if k in t.leaf]
            evs[sp] = dict(bt, stype=st)
        res = K.evaluate(net, evs)
        r.update(res if mode == "TITLES" else {f"{mode}_{k}": v for k, v in res.items()})
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    OUT = OUT_TXT.format("_smoke" if K.mode == "smoke" else "")
    assert K.mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {K.mode} steps {K.STEPS} seeds {K.SEEDS} arms {ARMS}")
    for arm in ARMS:
        rs = [run(arm, s) for s in K.SEEDS]
        ag = {kk: (float(np.nanmean([x[kk] for x in rs])) if isinstance(rs[0][kk], (int, float)) else [x[kk] for x in rs]) for kk in rs[0]}
        A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        for sp in ("indist", "heldout"):
            say(f"{arm:6s} {sp:7s} | role {ag[f'{sp}_res_pron_role']:.3f} (seeds {[round(x[f'{sp}_res_pron_role'], 3) for x in rs]}) | all "
                f"{ag[f'{sp}_res_all']:.3f} alias {ag[f'{sp}_res_alias']:.3f} | blind {ag[f'{sp}_acc_blind']:.3f} explicit {ag[f'{sp}_acc_explicit']:.3f}")
            say("        cells: " + " ".join(f"{c} {ag[f'{sp}_res_{c}']:.3f}" for c in K.CELLS) + f"  [{time.time() - t0:.0f}s]")
        if arm == "TREE":
            say(f"        TRAINING: wrong-node rate seen by the resolver {ag['train_wrong_node_rate']:.4f} | leaves {ag['train_leaves']:.0f} "
                f"| first-name leaves wrong at end {ag['train_leaves_wrong']:.2f}")
            say(f"        WALK (every slot a miss): in-dist {ag['indist_walk_all']:.3f} | held-out names {ag['heldout_walk_heldname']:.3f} "
                f"(seeds {[round(x['heldout_walk_heldname'], 3) for x in rs]})")
            for mode in ("TITLES", "TP", "PRIOR"):
                hr = ag["heldout_res_pron_role"] if mode == "TITLES" else ag[f"{mode}_heldout_res_pron_role"]
                say(f"        {mode:6s}: held-out-name node acc {ag[f'{mode}_heldout_node_heldname']:.3f} (seeds "
                    f"{[round(x[f'{mode}_heldout_node_heldname'], 3) for x in rs]}) | held-out role {hr:.3f} | evidence {ag[f'{mode}_heldout_evidence']:.0f}")
                say(f"                names (first, node, n evidence) per seed: {ag[f'{mode}_heldout_names']}")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if K.mode == "full" and all(x in A for x in ("GEARS", "TREE")):          # PINNED (ledger 2026-10-25, T2p)
        g, t = A["GEARS"], A["TREE"]
        say("VERDICTS (3-seed means)")
        say(f"CONTROL: GEARS held-out role {g['heldout_res_pron_role']:.3f} (T2o banked 0.995) -> "
            f"{'OK' if abs(g['heldout_res_pron_role'] - 0.995) <= 0.02 else 'VOID'}")
        worst = min(t[f"indist_res_{k}"] - g[f"indist_res_{k}"] for k in K.CELLS)
        t1 = t["heldout_res_pron_role"] >= 0.95 and worst >= -0.01
        say(f"T1 THE TREE IN TRAINING, HARD EVIDENCE: {'PASS' if t1 else 'FAIL'} (TREE held-out role {t['heldout_res_pron_role']:.3f}; "
            f"worst in-dist cell delta vs GEARS {worst:+.3f})")
        n2 = t["TITLES_heldout_node_heldname"]
        say(f"T2 THE LEAF: {'PASS' if n2 >= 0.95 else 'FAIL'} (held-out-name node acc, streamed, titles-only evidence: {n2:.3f})")
    say(f"DONE in {time.time() - t0:.0f}s")
