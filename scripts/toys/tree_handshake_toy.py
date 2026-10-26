"""tree_handshake_toy.py -- T2w, THE HANDSHAKE ACROSS PROCESSES (2026-10-25; Bryce's word; ledger "T2w"). CPU.
Bryce: "the sooner we move to sqlite the better ... a clean way to introduce a new model to an existing permanent database
... we dont want to rebuild the DB every time a new model connects." The protocol (ledger 10-25):
  1. connect read-only: load_leaves() -> the rows (symbols only: literal key -> Welford evidence over node ids)
  2. train with the tree in the loop (mandatory road); rows from the store, centroids RESET and re-anchored in the new
     model's own space (T2q's B_INHERIT, verbatim: tree_inherit_toy.run)
  3. the new model's evidence enters as a PROVISIONAL batch; AUDITION (its own per-key verdicts vs the permanent rows on
     shared keys >= 0.99 agreement) -> promote, else roll back
Two PROCESSES (TW_PHASE=A, then TW_PHASE=B): nothing passes between them but the SQLite file (and phase A's report json).
  A: per seed, T2q's model A (T2p TREE, standard names incl. the RESERVED ones); every hard-evidence event is logged and
     written to .cache/tree_handshake_s{seed}.db as permanent batches (certifier gold_resolution, generation A.s{seed});
     A's centroids saved under its generation. FIDELITY: the store's aggregates == A's in-memory rows.
  B: a fresh process; per seed, connect; the GUARD must refuse A's centroids and a model-space write; T2q's B_INHERIT run
     with the rows read from the store; B's training evidence -> provisional batch -> audition; a SABOTEUR batch (B's
     events with every node shifted k -> k+1 mod 4) must fail the audition and roll back leaving the store unchanged."""
import json, os, shutil, sys, time
import numpy as np

_argv = sys.argv[:]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import tree_inherit_toy as Q
sys.argv = _argv
from mycelium.tree_db import TreeDB

T, O, K = Q.T, Q.O, Q.K
PHASE = os.environ["TW_PHASE"]
SUF = ("_smoke" if K.mode == "smoke" else "") + os.environ.get("TW_TAG", "")   # TW_TAG=_v2: the tree_db v2 re-run (D4)
OUT = f".cache/tree_handshake{SUF}.txt"
DBP = ".cache/tree_handshake{}_s{}.db"
AJSON = f".cache/tree_handshake{SUF}_A.json"
SPACE = "node-evidence"
AUDIT = 0.99
LOG = []


class LogTree(T.Tree):
    """T2p's Tree; the TRAINING tree logs every hard-evidence event (eval copies do not)."""
    def __init__(self, log=True):
        super().__init__(); self.log = log; self.buf = []
        if log:
            LOG.append(self)

    def copy(self):
        t = LogTree(log=False); t.leaf = {k: dict(v) for k, v in self.leaf.items()}
        t.N, t.LS, t.SS = self.N.copy(), self.LS.copy(), self.SS.copy()
        return t

    def evidence(self, tok, k):
        super().evidence(tok, k)
        if self.log:
            self.buf.append((int(tok), int(k)))


def onehot(k):
    return np.eye(4)[k]


def open_db(seed, fresh):
    p = DBP.format(SUF, seed)
    if fresh:
        assert K.mode == "smoke" or not (os.path.exists(p) or os.path.exists(p + ".events")), f"refusing to overwrite {p}"
        for q in (p, p + "-wal", p + "-shm"):
            if os.path.exists(q):
                os.remove(q)
        if os.path.isdir(p + ".events"):
            shutil.rmtree(p + ".events")
    db = TreeDB(p)
    db.register_space(SPACE, "symbolic", 4, "Welford evidence over the tree's 4 leaf-level nodes (he/she/they/it)")
    return db


def audition(db, batch):
    """Agreement of a provisional batch's own per-key verdict with the permanent rows, on shared keys."""
    own = {}
    for k, v, _ in db.batch_rows(batch):
        own.setdefault(k, []).append(v)
    shared = [k for k in own if db.leaf(k) is not None]
    agree = [int(np.mean(own[k], 0).argmax() == db.leaf(k)[1].argmax()) for k in shared]
    return (float(np.mean(agree)) if agree else float("nan")), len(shared)


def snapshot(db):
    return {k: (n, m.round(12).tobytes()) for k, (n, m) in db.load_leaves().items()}


def phase_a(say):
    rep = {}
    for s in K.SEEDS:
        LOG.clear(); T.Tree = LogTree
        tr = LogTree(); _, wr = Q.train("TREE", s, s, Q.HELD0, tr)
        db = open_db(s, fresh=True); gen = f"A.s{s}"; ev = tr.buf; CH = 5000
        for i in range(0, len(ev), CH):
            db.add_evidence(f"{gen}.b{i // CH:03d}", [(str(t), onehot(k), f"{gen}.e{i + j}") for j, (t, k) in enumerate(ev[i:i + CH])],
                            "permanent", "gold_resolution", gen, SPACE)
        db.save_centroids(gen, tr.N, tr.LS, tr.SS)
        st = db.load_leaves(); ok = True
        for t, lf in tr.leaf.items():
            if lf["n"] > 0:
                n, m = st[str(t)]; ok &= (n == lf["n"]) and bool(np.allclose(m, lf["mean"]))
            else:
                ok &= str(t) not in st                                       # first-sight priors are never stored
        ok &= len(st) == sum(lf["n"] > 0 for lf in tr.leaf.values())
        res = {f: (st[str(K.O_F + f)][0], int(st[str(K.O_F + f)][1].argmax())) for f in sorted(Q.RESERVED) if str(K.O_F + f) in st}
        rep[s] = dict(fidelity=bool(ok), events=len(ev), rows=len(st), batches=len(db.batches()), train_wrong=wr, reserved=res)
        say(f"A seed {s}: {len(ev)} events in {len(db.batches())} permanent batches -> {len(st)} rows; FIDELITY store == in-memory "
            f"{'OK' if ok else 'MISMATCH'}; reserved rows (first: n, node) {res}; train wrong-node {wr:.4f}")
        db.close()
    json.dump(rep, open(AJSON, "w"), indent=1)


def phase_b(say):
    A = {int(k): v for k, v in json.load(open(AJSON)).items()}; rs = []
    for s in K.SEEDS:
        db = open_db(s, fresh=False); gen = f"B.s{s}"; r = {}
        try:
            db.load_centroids(f"A.s{s}", current=gen); r["guard_centroids"] = False
        except ValueError:
            r["guard_centroids"] = True
        db.register_space(f"model:{gen}", "model", O.DS, "model B's head coordinates")
        try:
            db.add_evidence(f"{gen}.leak", [("x", np.zeros(O.DS), "h")], "permanent", "none", gen, f"model:{gen}"); r["guard_space"] = False
        except ValueError:
            r["guard_space"] = True
        rows = db.load_leaves()                                             # 1. connect read-only: the rows, symbols only
        fake = T.Tree()
        fake.leaf = {int(k): dict(prior=np.full(4, 0.25), n=n, mean=m) for k, (n, m) in rows.items()}
        LOG.clear(); T.Tree = LogTree
        r.update(Q.run("B_INHERIT", s, fake))                               # 2. T2q's B_INHERIT, rows from the store
        btree = [t for t in LOG if t.log]; assert len(btree) == 1, len(btree)
        ev = btree[0].buf; pre = snapshot(db)
        db.add_evidence(f"{gen}.train", [(str(t), onehot(k), f"{gen}.e{i}") for i, (t, k) in enumerate(ev)],
                        "provisional", "gold_resolution", gen, SPACE)       # 3. probation
        r["provisional_unchanged"] = snapshot(db) == pre
        r["audit_b"], r["audit_b_keys"] = audition(db, f"{gen}.train")
        r["promoted"] = r["audit_b"] >= AUDIT
        if r["promoted"]:
            db.promote(f"{gen}.train")
        else:
            db.rollback(f"{gen}.train")
        inc = snapshot(db); db.rebuild_all(); r["rebuild_equal"] = snapshot(db) == inc
        pre = snapshot(db)
        db.add_evidence(f"{gen}.saboteur", [(str(t), onehot((k + 1) % 4), f"{gen}.sab{i}") for i, (t, k) in enumerate(ev)],
                        "provisional", "saboteur", gen, SPACE)
        r["audit_sab"], _ = audition(db, f"{gen}.saboteur")
        r["sab_rejected"] = r["audit_sab"] < AUDIT
        db.rollback(f"{gen}.saboteur"); r["sab_store_unchanged"] = snapshot(db) == pre
        r["b_events"] = len(ev); rs.append(r); db.close()
        say(f"B seed {s}: guard centroids {r['guard_centroids']} space {r['guard_space']} | rows read {len(rows)}, flipped by B "
            f"{r['rows_flipped']} | reserved node PRIOR {r['PRIOR_node_reserved']:.3f} TITLES {r['TITLES_node_reserved']:.3f} | "
            f"indist role {r['indist_res_pron_role']:.3f} reserved role {r['reserved_res_pron_role']:.3f} (TITLES {r['TITLES_reserved_res_pron_role']:.3f})")
        say(f"        probation: {len(ev)} events provisional (store unchanged {r['provisional_unchanged']}); audition {r['audit_b']:.3f} on "
            f"{r['audit_b_keys']} keys -> {'PROMOTED' if r['promoted'] else 'ROLLED BACK'}; rebuild == incremental {r['rebuild_equal']} | "
            f"saboteur audition {r['audit_sab']:.3f} -> {'rejected' if r['sab_rejected'] else 'ACCEPTED'}, store unchanged {r['sab_store_unchanged']}")
    json.dump(dict(A=A, B=rs), open(OUT.replace(".txt", ".json"), "w"), indent=1, default=str)
    if K.mode == "full":                                                    # PINNED (ledger 2026-10-25, T2w)
        say("VERDICTS (every seed)")
        w0 = all(A[s]["fidelity"] for s in K.SEEDS)
        say(f"W0 FIDELITY (store == model A's rows): {'PASS' if w0 else 'FAIL'} {[A[s]['fidelity'] for s in K.SEEDS]}")
        w1 = all(r["PRIOR_node_reserved"] >= 0.99 and r["rows_flipped"] == 0 for r in rs)
        say(f"W1 HANDSHAKE (fresh process, rows from SQLite): {'PASS' if w1 else 'FAIL'} (reserved node "
            f"{[round(r['PRIOR_node_reserved'], 3) for r in rs]}; flipped {[r['rows_flipped'] for r in rs]}; T2q B_INHERIT banked 1.000 / 0)")
        w2 = all(r["promoted"] and r["sab_rejected"] and r["sab_store_unchanged"] and r["provisional_unchanged"] and r["rebuild_equal"] for r in rs)
        say(f"W2 PROBATION: {'PASS' if w2 else 'FAIL'} (B audition {[round(r['audit_b'], 3) for r in rs]}; saboteur "
            f"{[round(r['audit_sab'], 3) for r in rs]}; store unchanged after rejection {[r['sab_store_unchanged'] for r in rs]})")
        w3 = all(r["guard_centroids"] and r["guard_space"] for r in rs)
        say(f"W3 GUARDS: {'PASS' if w3 else 'FAIL'} (foreign centroids refused {[r['guard_centroids'] for r in rs]}; model-space write "
            f"refused {[r['guard_space'] for r in rs]})")


if __name__ == "__main__":
    import torch
    torch.set_num_threads(int(os.environ.get("TOY_THREADS", "16")))
    if PHASE == "A":
        assert K.mode == "smoke" or not os.path.exists(OUT), "refusing to overwrite a banked read"
    out = open(OUT, "w" if PHASE == "A" else "a"); t0 = time.time()

    def say(s):
        print(s, flush=True); out.write(s + "\n"); out.flush()
    say(f"PHASE {PHASE} (pid {os.getpid()}) mode {K.mode} steps {K.STEPS} seeds {K.SEEDS} reserved {sorted(Q.RESERVED)}")
    (phase_a if PHASE == "A" else phase_b)(say)
    say(f"PHASE {PHASE} DONE in {time.time() - t0:.0f}s")
