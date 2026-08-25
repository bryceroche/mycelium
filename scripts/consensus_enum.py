"""consensus_enum.py — THE EQUATION'S TEST (2026-08-25): op-multiset
consensus across the deepened bank -> enumeration on agreed rows only.
Consensus = >=K readers emit the IDENTICAL sorted op-multiset (exact
tuples; the Anna-Karenina grain for ops). Metrics vs the single-reader
baseline (coverage ~15-25%): consensus-row coverage, unique-emit
rights/lies, and the coverage-when-agreed number that decides whether
consensus exactness unlocks assembly. Env: CE_MIN_K (default 5).
"""
import os, sys, json, re, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from enum_assembly import reachable

NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.])")
K_MIN = int(os.environ.get("CE_MIN_K", "5"))

def main():
    files = sorted(glob.glob('.cache/audition_R*.json'))
    reads = {}
    for f in files:
        d = json.load(open(f))
        if d["rows"] and "ops" in d["rows"][0]:
            reads[d["id"]] = d["rows"]
    ids = sorted(reads)
    print(f"[ce] readers with ops field: {len(ids)}", flush=True)
    if len(ids) < K_MIN: print("[ce] not enough"); return
    n = len(reads[ids[0]])
    # rebuild fixture texts (same order as audition fixtures)
    import importlib
    aro = open('scripts/audition_read_one.py').read()
    ns = {"json": json, "glob": glob, "np": np}
    exec(aro[aro.index("def fixtures():"):aro.index("rows = fixtures()")], ns)
    rows = ns["fixtures"]()
    assert len(rows) == n, (len(rows), n)
    T = {t: {"n": 0, "agreed": 0, "cover": 0, "uniq": 0, "uright": 0,
             "ulies": 0} for t in ("gold", "wv", "held", "cen")}
    for i in range(n):
        tag = rows[i]["tag"]
        if tag.startswith("anc"): continue
        tag = tag if tag in T else "cen"
        t = T[tag]; t["n"] += 1
        votes = Counter()
        for c in ids:
            o = reads[c][i].get("ops")
            if o is not None: votes[tuple(o)] += 1
        if not votes: continue
        (best_ops, k) = votes.most_common(1)[0]
        if k < K_MIN: continue
        ops = [l for l in best_ops if l in ("add", "sub", "mul", "sq", "fr")]
        if len(ops) != len(best_ops) or not ops: continue   # opa etc: refuse
        nums = [int(m.group(1)) for m in NUM.finditer(rows[i]["original"])]
        if not nums or len(nums) > 8 or len(ops) > 6: continue
        t["agreed"] += 1
        roots, blown = reachable(nums, ops)
        key = rows[i]["answer"]
        if key in roots: t["cover"] += 1
        if len(roots) == 1:
            t["uniq"] += 1
            if key in roots: t["uright"] += 1
            else: t["ulies"] += 1
    for tag, t in T.items():
        if t["n"] == 0: continue
        print(f"[ce {tag}] n={t['n']} AGREED {t['agreed']} "
              f"coverage-when-agreed {t['cover']}/{t['agreed']} "
              f"unique {t['uniq']} (right {t['uright']} lies {t['ulies']})",
              flush=True)

if __name__ == "__main__":
    main()
