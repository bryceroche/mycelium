"""tree_inherit_toy.py -- T2q, A NEW MODEL INHERITS THE TREE (2026-10-25; Bryce's word; ledger "T2q"). CPU.
Bryce: "ideally we never delete the tree between training runs. it only accumulates evidence ... we need a way to train new
models to become familiar with the existing tree." The test makes knowledge live ONLY in the tree:
  - MODEL A trains exactly as T2p's TREE arm (tree_train_toy: mandatory road, hard evidence only) on the standard training
    names, which include the RESERVED names. Its tree's ROWS are kept.
  - MODEL B: fresh init, its data drawn with the RESERVED names removed (they are never in B's training text). Arms:
      B_GOLD     B with the gold node (GEARS form): the ceiling the tree should reach
      B_INHERIT  B trained against A's tree: rows inherited (evidence counts intact), centroid CFs RESET and rebuilt from
                 B's own certified slots (Bryce's ruling: keep rows, rebuild centroids per run); B's training keeps adding
                 hard evidence to the rows
      B_EMPTY    B trained against an empty tree (T2p's TREE on B's data)
  - TEST: episodes whose protagonist is a RESERVED name (B's weights never saw it; A's rows know it), streamed with NO
    test-time evidence (PRIOR mode: the tree as it stands), so only inheritance can place a reserved name. TITLES mode
    (test-time title evidence, T2p's scored mode) is reported as a read.
Generator, reader, resolver and tree mechanics are tree_train_toy's / tree_search_toy's, imported unchanged; the name split is
done by setting tree_type_toy's HELDOUT set at generation time."""
import contextlib, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

_argv = sys.argv[:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tree_train_toy as T
sys.argv = _argv
O, K = T.O, T.K

ARMS = os.environ.get("TQ_ARMS", "B_GOLD,B_INHERIT,B_EMPTY").split(",")
OUT_TXT = ".cache/tree_inherit{}.txt"
HELD0 = set(K.HELDOUT)
RESERVED = {4, 5, 12, 13}                                                   # 2 he + 2 she training names, withheld from B
RES_TOK = torch.tensor(sorted(K.O_F + f for f in RESERVED))
assert not RESERVED & HELD0


@contextlib.contextmanager
def names_held(s):
    old = K.HELDOUT; K.HELDOUT = set(s)
    try:
        yield
    finally:
        K.HELDOUT = old


def train(kind, tseed, dseed, held, tree):
    """tree_train_toy.run's training loop, with the training names restricted by `held` and a given starting tree."""
    rng = np.random.default_rng(dseed); nrng = np.random.default_rng(77_000 + dseed); trng = np.random.default_rng(55_000 + dseed)
    torch.manual_seed(tseed); net = O.Net("SEARCH" if kind == "TREE" else "GEARS")
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=K.LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=K.LR, total_steps=K.STEPS, pct_start=0.1)
    wrong = n = 0
    for step in range(K.STEPS):
        with names_held(held):
            bt = O.batch(rng, K.BATCH, "train", 0.0, nrng, 0.0, False, trng)
        if kind == "TREE":
            gold = bt["stype"]; bt = dict(bt, stype=T.read_nodes(net, tree, bt))
            wrong += int((bt["stype"] != gold)[bt["smask"]].sum()); n += int(bt["smask"].sum())
        lp, aux = net(bt); tg = bt["tgt"]; m = tg >= 0
        loss = F.nll_loss(lp[m], tg[m]) + aux
        if kind == "TREE":
            lab = T.certified(tree, bt); loss = loss + T.proto_loss(net, lab)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if kind == "TREE":
            T.update_cf(net, tree, lab); T.write_evidence(tree, bt, bt["rslot"].numpy(), None, "TP")
    return net, wrong / max(1, n)


def gender(t):
    return 0 if t - K.O_F < 8 else 1


def run(arm, seed, treeA):
    erng = np.random.default_rng(91_000 + seed); envg = np.random.default_rng(93_000 + seed); tev = np.random.default_rng(57_000 + seed)
    with names_held(HELD0 | RESERVED):
        indist = O.batch(erng, K.N_EVAL, "train", 0.0, envg, 0.0, False, tev)
    with names_held(RESERVED):
        reserved = O.batch(erng, K.N_EVAL, "heldout", 0.0, envg, 0.0, False, tev)
    ev = {"indist": indist, "reserved": reserved}
    for x in indist["sptok"][indist["smask"]].tolist():
        assert x not in RES_TOK.tolist(), "a reserved name leaked into B's in-distribution split"
    r = {}
    if arm == "B_GOLD":
        net, _ = train("GOLD", 500 + seed, 500 + seed, HELD0 | RESERVED, None)
        r.update(K.evaluate(net, ev)); return r
    tree = T.Tree()
    if arm == "B_INHERIT":
        tree.leaf = {k: dict(v, mean=v["mean"].copy()) for k, v in treeA.leaf.items()}   # ROWS inherited; CFs start empty
    before = {t: tree.node(t) for t in tree.leaf}
    net, wr = train("TREE", 500 + seed, 500 + seed, HELD0 | RESERVED, tree)
    r["train_wrong_node_rate"] = wr; r["rows_inherited"] = len(before)
    r["rows_flipped"] = sum(int(tree.node(t) != k) for t, k in before.items())
    r["reserved_rows_evidence_gain"] = sum(int(tree.leaf[t]["n"] - treeA.leaf[t]["n"]) for t in RES_TOK.tolist()
                                           if arm == "B_INHERIT" and t in tree.leaf)
    gold = ev["reserved"]["stype"].clone(); held = torch.isin(ev["reserved"]["sptok"], RES_TOK) & ev["reserved"]["smask"]
    wk = torch.from_numpy(T.walk_post(net, ev["reserved"]).argmax(-1))
    r["walk_reserved"] = float((wk == gold)[held].float().mean())
    for mode in ("PRIOR", "TITLES"):
        t = tree.copy(); evs = {}
        for sp, bt in ev.items():
            if mode == "TITLES" and sp == "indist":
                continue
            st, nev = T.stream_eval(net, t, bt, "NONE" if mode == "PRIOR" else "TITLES")
            if sp == "reserved":
                r[f"{mode}_node_reserved"] = float((st == gold)[held].float().mean())
                r[f"{mode}_names"] = [(int(k) - K.O_F, t.node(k), gender(k)) for k in RES_TOK.tolist() if k in t.leaf]
            evs[sp] = dict(bt, stype=st)
        res = K.evaluate(net, evs)
        r.update(res if mode == "PRIOR" else {f"TITLES_{k}": v for k, v in res.items()})
    return r


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    OUT = OUT_TXT.format("_smoke" if K.mode == "smoke" else "")
    assert K.mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w"); t0 = time.time(); A = {}; allres = []

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"mode {K.mode} steps {K.STEPS} seeds {K.SEEDS} arms {ARMS} reserved first names {sorted(RESERVED)}")
    trees = {}
    for s in K.SEEDS:                                                       # MODEL A per seed: T2p's TREE on the standard names
        tr = T.Tree(); _, wr = train("TREE", s, s, HELD0, tr); trees[s] = tr
        bad = [t - K.O_F for t in tr.leaf if K.O_F <= t < K.O_F + K.NF and tr.node(t) != gender(t)]
        say(f"MODEL A seed {s}: rows {len(tr.leaf)}; reserved rows {[(f, tr.node(K.O_F + f), tr.leaf[K.O_F + f]['n']) for f in sorted(RESERVED)]}; "
            f"wrong first-name rows {bad}; train wrong-node {wr:.4f}  [{time.time() - t0:.0f}s]")
    for arm in ARMS:
        rs = [run(arm, s, trees[s]) for s in K.SEEDS]
        ag = {kk: (float(np.nanmean([x[kk] for x in rs])) if isinstance(rs[0][kk], (int, float)) else [x[kk] for x in rs]) for kk in rs[0]}
        A[arm] = ag; allres.append(dict(arm=arm, seeds=rs, agg=ag))
        for sp in ("indist", "reserved"):
            say(f"{arm:9s} {sp:8s} | role {ag[f'{sp}_res_pron_role']:.3f} (seeds {[round(x[f'{sp}_res_pron_role'], 3) for x in rs]}) | all "
                f"{ag[f'{sp}_res_all']:.3f} alias {ag[f'{sp}_res_alias']:.3f} | blind {ag[f'{sp}_acc_blind']:.3f} explicit {ag[f'{sp}_acc_explicit']:.3f}")
            say("        cells: " + " ".join(f"{c} {ag[f'{sp}_res_{c}']:.3f}" for c in K.CELLS) + f"  [{time.time() - t0:.0f}s]")
        if arm != "B_GOLD":
            say(f"        rows inherited {ag['rows_inherited']:.0f} | rows flipped by B {[x['rows_flipped'] for x in rs]} | evidence B added to "
                f"reserved rows {ag['reserved_rows_evidence_gain']:.0f} | train wrong-node {ag['train_wrong_node_rate']:.4f}")
            say(f"        reserved-name node acc: PRIOR {ag['PRIOR_node_reserved']:.3f} (seeds {[round(x['PRIOR_node_reserved'], 3) for x in rs]}) | "
                f"TITLES {ag['TITLES_node_reserved']:.3f} | walk alone {ag['walk_reserved']:.3f}")
            say(f"        reserved names (first, node, true) per seed, PRIOR: {ag['PRIOR_names']}")
            say(f"        TITLES mode reserved role {ag['TITLES_reserved_res_pron_role']:.3f}")
    json.dump(allres, open(OUT.replace(".txt", ".json"), "w"), indent=1)
    if K.mode == "full" and all(x in A for x in ("B_GOLD", "B_INHERIT", "B_EMPTY")):   # PINNED (ledger 2026-10-25, T2q)
        g, i, e = A["B_GOLD"], A["B_INHERIT"], A["B_EMPTY"]
        say("VERDICTS (3-seed means; reserved split, PRIOR mode = no test-time evidence)")
        say(f"CONTROL: B_GOLD reserved role {g['reserved_res_pron_role']:.3f} -> {'OK' if g['reserved_res_pron_role'] >= 0.95 else 'VOID'}")
        worst = min(i[f"indist_res_{k}"] - g[f"indist_res_{k}"] for k in K.CELLS)
        q1 = (i["reserved_res_pron_role"] >= g["reserved_res_pron_role"] - 0.01 and i["PRIOR_node_reserved"] >= 0.99 and worst >= -0.01)
        say(f"Q1 INHERITANCE: {'PASS' if q1 else 'FAIL'} (B_INHERIT reserved role {i['reserved_res_pron_role']:.3f} vs B_GOLD "
            f"{g['reserved_res_pron_role']:.3f}; reserved node {i['PRIOR_node_reserved']:.3f}; worst in-dist cell vs B_GOLD {worst:+.3f})")
        fl = [x["rows_flipped"] for x in allres[ARMS.index("B_INHERIT")]["seeds"]]
        say(f"Q2 NO CORRUPTION: {'PASS' if sum(fl) == 0 else 'FAIL'} (inherited rows flipped by B's training per seed {fl})")
        say(f"READ B_EMPTY: reserved role {e['reserved_res_pron_role']:.3f}, reserved node {e['PRIOR_node_reserved']:.3f}; "
            f"INHERIT - EMPTY role {i['reserved_res_pron_role'] - e['reserved_res_pron_role']:+.3f}")
    say(f"DONE in {time.time() - t0:.0f}s")
