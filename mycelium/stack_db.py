"""stack_db.py -- THE STACK'S TABLES IN THE PERMANENT TREE (TDB-STACK, ledger 2026-11-12; Bryce: "word given").
The carved line (docs/CORE_THEORIES.md, 11-11): at the bottom of the Tree DB the cross join is a cheating trap; at the top it is
the grammar of mathematics. So the store holds:
  - NODES: the L1 -> L2 -> L3 class tree (parents only; ids level*10000 + class) and the L1 centroids (generation-tagged; model
    space, rebuilt per generation). A leaf phrase is never a node of this forest and never part of a key.
  - EDGES (space 'edge-op'): admitted class x class rows (level >= 1), one event per TRAIN step, vec = one-hot op, source = problem.
  - MOTIFS (space 'cmotif-op'): admitted class-keyed fragments (producer input classes -> consumer other side), same events.
JUNCTIONS (which phrase met which, per problem) are volatile: the builder sees them to apply admission, the store never does.
Built OFFLINE from answer-key-gated TRAIN; read-only during a run; grows between runs by provisional batches the key promotes."""
import ast

import numpy as np

from mycelium.tree_db import TreeDB

SPACES = {"edge-op": "admitted class x class edge rows (level >= 1): a step's op, one-hot; source = problem",
          "cmotif-op": "admitted class-keyed motif rows (producer input classes -> consumer other side): a step's op, one-hot"}


def _guard(key, vocab):
    """NO-LEAF-PAIR: a stored key may hold only ints, None, bools, 's' and the kind vocabulary; level 0 never."""
    if key[0] == 0:
        raise ValueError(f"L0 key refused (the cross join of leaves): {key!r}")
    stack = [key]
    while stack:
        x = stack.pop()
        if isinstance(x, tuple):
            stack.extend(x)
        elif isinstance(x, str) and x != "s" and x not in vocab:
            raise ValueError(f"key carries a non-class string {x!r}: {key!r}")
        elif not isinstance(x, (str, int, bool, type(None))):
            raise ValueError(f"key element {type(x).__name__} does not round-trip: {key!r}")


def write_stack(path, generation, centers, n_assigned, ss, maps, edges, motifs, nops, vocab, batch="stack-train"):
    """maps: [L1 ident, L1 -> L2, L1 -> L3] (Ward cuts of one tree, so nested). edges / motifs: {key: [(op, problem), ...]}, the
    TRAIN events of ADMITTED rows only (the builder applies admission; no leaf pair is ever passed in)."""
    db = TreeDB(path, tail_max=None)
    for s, m in SPACES.items():
        db.register_space(s, "symbolic", nops, m)
    eye = np.eye(nops)
    for space, rows, tag in (("edge-op", edges, "edges"), ("cmotif-op", motifs, "motifs")):
        ev = []
        for k, evs in rows.items():
            _guard(k, vocab); sk = repr(k)
            if ast.literal_eval(sk) != k:
                raise ValueError(f"key does not round-trip: {sk}")
            ev += [(sk, eye[o], str(q)) for o, q in evs]
        db.add_evidence(f"{batch}-{tag}", ev, "permanent", "answer-key|diversity", generation, space,
                        note="TDB-STACK offline build from TRAIN (gold ops)")
    b = f"{batch}-edges"
    L1, L2, L3 = maps
    for c in set(int(x) for x in L2):
        if len(set(L3[L2 == c].tolist())) != 1:
            raise ValueError(f"L2 class {c} straddles L3 classes: the cuts are not nested")
    for c in sorted(set(int(x) for x in L3)):
        db.add_node(30000 + c, None, 3, f"L3:{c}", b)
    for c in sorted(set(int(x) for x in L2)):
        db.add_node(20000 + c, 30000 + int(L3[np.flatnonzero(L2 == c)[0]]), 2, f"L2:{c}", b)
    for c in range(len(L1)):
        db.add_node(10000 + c, 20000 + int(L2[c]), 1, f"L1:{c}", b)
    db.save_centroids(generation, n_assigned, centers * n_assigned[:, None], ss)
    db.close()


def load_stack(path, generation):
    """-> centers [K, d], maps [L1 ident, L1 -> L2, L1 -> L3], edges {key: counts}, motifs {key: counts}."""
    db = TreeDB(path, tail_max=None)
    N, LS, _ = db.load_centroids(generation, generation)
    centers = LS / N[:, None]
    par = {i: p for i, p, _, _, _ in db.nodes()}
    K = len(N); l2 = np.array([par[10000 + c] - 20000 for c in range(K)]); l3 = np.array([par[20000 + c] - 30000 for c in l2])
    space = dict(db.c.execute("SELECT key, space FROM key_space").fetchall())
    tabs = {"edge-op": {}, "cmotif-op": {}}
    for k, n, m in db.c.execute("SELECT key, n, mean FROM leaves").fetchall():
        tabs[space[k]][ast.literal_eval(k)] = np.round(np.frombuffer(m, dtype=np.float64) * n)
    db.close()
    return centers, [np.arange(K), l2, l3], tabs["edge-op"], tabs["cmotif-op"]
