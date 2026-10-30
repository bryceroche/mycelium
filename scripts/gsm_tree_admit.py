"""gsm_tree_admit.py -- the queue's tree writes (ledger 2026-10-29, Bryce "word given for the whole queue"), into the GSM bench's
store (.cache/gsmb_tree.db, upgraded in place to tree_db v3 on open): ONE store, three forests.
  kind 'name'  : (already there as leaves: the name-gender lexicon; untouched)
  kind 'type'  : idea B's k=3 type forest (class roots at level 0, member nouns at level 1); evidence space 'noun-class'
                 (dim = number of classes, one-hot) -- one event per (noun, TRAIN problem) behind a kept union edge.
  kind 'macro' : idea D's HAND-ADMITTED macros (the quota: fully-leafed shapes from the miner's ranking, named by hand) with
                 primitive bodies (grammar 'gsm-ptr-1': add/sub/mul/div over n | c<value>); evidence space 'macro-use' (dim 1)
                 -- one event per TRAIN problem carrying the shape.
Every write is answer-key custody (TRAIN gold programs reproduce their finals; gsm_bench_data asserts it). Idempotence: refuses
if the batches exist (a batch name is never reused)."""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import gsm_bench_data as GD
import gsm_macro_miner as MM
from mycelium.tree_db import TreeDB

STORE = ".cache/gsmb_tree.db"
ADMIT = {                                                   # the hand quota: name -> fully-leafed body (miner rank 10-29)
    "two-item-total": "add(mul(n,n),mul(n,n))",
    "percent-of": "mul(div(n,c100),n)",
    "discount": "sub(n,mul(div(n,c100),n))",
    "increase-by-percent": "add(mul(div(n,c100),n),n)",
    "change-after-buying": "sub(n,mul(n,n))",
    "unit-rate-times": "mul(div(n,n),n)",
}

if __name__ == "__main__":
    fo = json.load(open(GD.V1.C("gsm_type_forest_k3.json"))); Dt = json.load(open(GD.OUT))
    roots = sorted(set(fo["parent"].values())); cid = {r: i for i, r in enumerate(roots)}
    db = TreeDB(STORE)
    db.register_space("noun-class", "symbolic", len(roots), f"one-hot over the {len(roots)} k=3 type-forest classes (gsm_type_forest)")
    db.register_space("macro-use", "symbolic", 1, "1.0 per TRAIN problem whose gold program carries the macro's shape")
    rows = []
    for e, pids in fo["edges"].items():
        for noun in set(e.split("|")):
            v = np.zeros(len(roots)); v[cid[fo["parent"][noun]]] = 1.0
            rows += [(f"noun:{noun}", v, f"gsm-train:{Dt['TRAIN'][p]['gid']}") for p in pids]
    db.add_evidence("gsm-type-forest-k3", rows, "permanent", "answer_key", "gsmb-10-29", "noun-class",
                    note="idea B: unions of + / - operands' nouns carried by >= 3 distinct TRAIN problems")
    nid = 1_000_000
    for r in roots:
        db.add_node(nid + cid[r], None, 0, f"class:{r}", "gsm-type-forest-k3", kind="type")
    for j, noun in enumerate(sorted(fo["parent"])):
        db.add_node(nid + len(roots) + j, nid + cid[fo["parent"][noun]], 1, noun, "gsm-type-forest-k3", kind="type")
    inv = {b: a for a, b in ADMIT.items()}; mrows = []
    for it in Dt["TRAIN"]:
        for s in {s for s, _ in MM.instances(it)} & set(inv):
            mrows.append((f"macro:{inv[s]}", np.ones(1), f"gsm-train:{it['gid']}"))
    db.add_evidence("gsm-macros-hand-10-29", mrows, "permanent", "answer_key", "gsmb-10-29", "macro-use",
                    note="idea D: hand-admitted fully-leafed macros (the quota) from gsm_macro_miner's TRAIN ranking")
    for j, (name, body) in enumerate(ADMIT.items()):
        db.add_macro(2_000_000 + j, None, 1, name, body, "gsm-ptr-1", "gsm-macros-hand-10-29")
    n_type = len(db.nodes("type")); M = db.macros(); L = db.load_leaves(); db.rebuild_all(); db.close()
    print(f"[admit] type forest: {len(roots)} classes + {n_type - len(roots)} nouns ({len(rows)} events); macros: "
          + "; ".join(f"{m[1]} = {m[2]} ({int(L['macro:' + m[1]][0])} problems)" for m in M))
