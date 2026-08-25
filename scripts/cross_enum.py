"""cross_enum.py — LEVER 2: CROSS-MECHANISM CONSENSUS (the Nazare
interference). Gate: the atlas-CHAIN multiset must EQUAL the reader-
parse consensus multiset (two different decode mechanisms constructively
interfering) -> enumeration fires only on double-sworn multisets.
Metrics vs single-mechanism baselines (chain-only enum ~15-25% coverage;
parse-consensus ~14%).
"""
import os, sys, json, re, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter
from enum_assembly import reachable

NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.])")
K_MIN = int(os.environ.get("CE_MIN_K", "5"))

def main():
    chain = json.load(open('.cache/chain_ops.json'))
    reads = {}
    for f in sorted(glob.glob('.cache/audition_R*.json')):
        d = json.load(open(f))
        if d["rows"] and "ops" in d["rows"][0]:
            reads[d["id"]] = d["rows"]
    ids = sorted(reads)
    aro = open('scripts/audition_read_one.py').read()
    ns = {"json": json, "glob": glob, "np": np}
    exec(aro[aro.index("def fixtures():"):aro.index("rows = fixtures()")], ns)
    rows = ns["fixtures"]()
    n = len(rows)
    T = {t: {"n": 0, "sworn": 0, "cover": 0, "uniq": 0, "ur": 0, "ul": 0}
         for t in ("gold", "wv", "held", "cen")}
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
        best_parse, k = votes.most_common(1)[0]
        cops = tuple(chain[i]["ops"])
        if k < K_MIN or best_parse != cops: continue      # no interference
        ops = [l for l in cops if l in ("add", "sub", "mul", "sq", "fr")]
        if len(ops) != len(cops) or not ops: continue
        nums = [int(m.group(1)) for m in NUM.finditer(rows[i]["original"])]
        if not nums or len(nums) > 8 or len(ops) > 6: continue
        t["sworn"] += 1
        roots, _ = reachable(nums, ops)
        key = rows[i]["answer"]
        if key in roots: t["cover"] += 1
        if len(roots) == 1:
            t["uniq"] += 1
            if key in roots: t["ur"] += 1
            else: t["ul"] += 1
    for tag, t in T.items():
        if t["n"] == 0: continue
        print(f"[cx {tag}] n={t['n']} DOUBLE-SWORN {t['sworn']} "
              f"coverage {t['cover']}/{t['sworn']} unique {t['uniq']} "
              f"(right {t['ur']} lies {t['ul']})", flush=True)

if __name__ == "__main__":
    main()
