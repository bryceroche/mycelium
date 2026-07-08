"""brickp_rel_bars.py — the REL-SIDE split bars (2026-07-10, registered
before built): where Brick-P's +32 apparently lives.

PREDICTION (relay, pinned): the thinned rel-side errors are disproportionately
COLLISION-TYPE — mutually inconsistent relational claims — vs LONE rel
misbindings. Lone-thinning instead = the mechanism is NOT negotiation
(candidates pre-listed: ladder-as-regularizer; settle-as-implicit-TTA) and the
breathing thesis takes its third formulation.

COUNTERS (rel/sel factors + query; gold result vars are unique per factor by
generator construction, so duplicates are collisions by definition):
  COLL   — slots in a duplicate-RESULT or duplicate-SIGNATURE group
           (two rel/sel factors claiming the same result var, or identical
           op+args with conflicting results)
  LONE   — wrong rel/sel factor (>=2-of-3 component match to an unmatched
           gold factor) NOT in any collision group
  MISS   — unmatched gold rel/sel factors
  PHANT  — unpairable predicted rel/sel factors outside collisions
  QUERY  — q_pred != gold query var

USAGE: DEV=AMD .venv/bin/python3 scripts/brickp_rel_bars.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import numpy as np

from brickp_split_bars import run, HEADS, CORPORA  # noqa: E402


def rkey(f):
    if f["ftype"] == "rel":
        return ("rel", f["op"], tuple(sorted(f["args"])), f["result"])
    return ("sel", f["sel"], tuple(sorted(f["args"])), f["result"])


def classify_rel(samples, parses):
    c = {"coll": 0, "lone": 0, "miss": 0, "phant": 0, "query": 0,
         "rel_total": 0}
    for i, smp in enumerate(samples):
        facs, q = parses[i]
        pred = [f for f in facs if f["ftype"] in ("rel", "sel")]
        goldr = [f for f in smp["factors"] if f["ftype"] in ("rel", "sel")]
        c["rel_total"] += len(goldr)
        if q != smp["query_var"]:
            c["query"] += 1
        # collisions among predictions
        res_ct = Counter(f["result"] for f in pred)
        sig_ct = Counter((f.get("op", f.get("sel")),
                          tuple(sorted(f["args"]))) for f in pred)
        coll_idx = set()
        for xi, f in enumerate(pred):
            if res_ct[f["result"]] > 1 or \
               sig_ct[(f.get("op", f.get("sel")),
                       tuple(sorted(f["args"])))] > 1:
                coll_idx.add(xi)
        c["coll"] += len(coll_idx)
        # exact matches out (multiset)
        gset = Counter(rkey(f) for f in goldr)
        unmatched_pred = []
        for xi, f in enumerate(pred):
            k = rkey(f)
            if gset[k] > 0:
                gset[k] -= 1
            else:
                unmatched_pred.append((xi, f))
        gold_left = list(gset.elements())
        used = [False] * len(gold_left)
        for xi, f in unmatched_pred:
            if xi in coll_idx:
                continue          # counted as collision
            k = rkey(f)
            best, best_n = -1, 1
            for gi, gk in enumerate(gold_left):
                if used[gi] or gk[0] != k[0]:
                    continue
                nm = (gk[1] == k[1]) + (gk[2] == k[2]) + (gk[3] == k[3])
                if nm > best_n:
                    best, best_n = gi, nm
            if best >= 0:
                used[best] = True
                c["lone"] += 1
            else:
                c["phant"] += 1
        c["miss"] += sum(1 for u in used if not u)
    return c


def main():
    for cname, (jpath, spath) in CORPORA.items():
        samples = [json.loads(l) for l in open(jpath)]
        z = np.load(spath)
        states, tokmask, sent = z["states"], z["tokmask"], z["sent"]
        print(f"\n=== {cname} (n={len(samples)}, "
              f"rel-side population) ===")
        print(f"  head       | COLL | LONE | MISS | PHANT | QUERY")
        base = None
        for hname, (ckpt, kb) in HEADS.items():
            parses, answers, _mv = run(ckpt, kb, samples, states,
                                       tokmask, sent)
            c = classify_rel(samples, parses)
            print(f"  {hname:10s} | {c['coll']:4d} | {c['lone']:4d} |"
                  f" {c['miss']:4d} | {c['phant']:4d}  | {c['query']:4d}")
            if base is None:
                base = c
            else:
                for k in ("coll", "lone", "miss", "query"):
                    d = (c[k] - base[k]) / max(base[k], 1)
                    print(f"    delta {k:5s}: {d:+.1%}")
    print(f"\n  REGISTERED: COLL thins disproportionately vs LONE ="
          f" negotiation confirmed; LONE thins instead = third formulation"
          f" (ladder-regularizer / settle-as-implicit-TTA).")


if __name__ == "__main__":
    main()
