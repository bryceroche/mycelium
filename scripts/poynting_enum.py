"""poynting_enum.py — LEVER 2.5: THE POYNTING PRIOR (2026-08-25, word
given; E x B = flow). Enumerate wirings WITH provenance and SCORE each
complete tree by flow-affinity: an op-instance consuming a surface
number pays that slot's fat-affinity to it; consuming another op's
result pays the slots' attention OVERLAP. EMIT THE ARGMAX wiring on
every double-sworn row (coverage -> emission). Gate: cross-mechanism
(chain multiset == parse consensus). Metrics: top-1 rights/lies vs the
unique-only baseline.
"""
import os, sys, json, re, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from collections import Counter

NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.])")
K_MIN = int(os.environ.get("CE_MIN_K", "5"))
OPF = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
       "mul": lambda a, b: a * b}

def best_wiring(nums, slots, ovl, cap=400000):
    """DFS over op-instance applications with provenance; returns
    (best_score, best_root) or None. avail: list of (value, tag) where
    tag = ('n', num_idx) or ('s', slot_idx)."""
    calls = [0]; best = [None]
    order = list(range(len(slots)))
    def rec(avail, left, score):
        calls[0] += 1
        if calls[0] > cap: return
        if not left:
            for v, _ in avail:
                if best[0] is None or score > best[0][0]:
                    best[0] = (score, v)
            return
        si = left[0]; rest = left[1:]
        op = slots[si]["op"]
        n = len(avail)
        def aff(tag):
            if tag[0] == 'n':
                a = slots[si]["aff"]
                return a[tag[1]] if tag[1] < len(a) else 0.0
            return ovl[si][tag[1]] if tag[1] < len(ovl) else 0.0
        if op == "sq":
            for i in range(n):
                v, tg = avail[i]
                nv = v * v
                if nv > 300: continue
                rec(avail[:i] + [(nv, ('s', si))] + avail[i + 1:],
                    rest, score + aff(tg))
        elif op == "fr":
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    (a, ta), (k, tk2) = avail[i], avail[j]
                    if k < 2 or a % k: continue
                    na = [x for t2, x in enumerate(avail) if t2 not in (i, j)]
                    rec(na + [(a // k, ('s', si))], rest,
                        score + aff(ta) + 0.5 * aff(tk2))
        elif op in OPF:
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    (a, ta), (b, tb) = avail[i], avail[j]
                    v = OPF[op](a, b)
                    if not (0 <= v <= 300): continue
                    na = [x for t2, x in enumerate(avail) if t2 not in (i, j)]
                    rec(na + [(v, ('s', si))], rest,
                        score + aff(ta) + aff(tb))
    rec([(v, ('n', i)) for i, v in enumerate(nums)], order, 0.0)
    return best[0]

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
    T = {t: {"n": 0, "sworn": 0, "emit": 0, "right": 0, "lies": 0}
         for t in ("gold", "wv", "held", "cen")}
    for i in range(len(rows)):
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
        if k < K_MIN or best_parse != cops: continue
        slots = [s2 for s2 in chain[i].get("slots", [])
                 if s2["op"] in ("add", "sub", "mul", "sq", "fr")]
        if not slots or len(slots) != len([o for o in cops
                                           if o in ("add", "sub", "mul", "sq", "fr")]):
            continue
        if any(o == "opa" for o in cops): continue
        nums = [int(m.group(1)) for m in NUM.finditer(rows[i]["original"])]
        if not nums or len(nums) > 8 or len(slots) > 6: continue
        t["sworn"] += 1
        ovl = chain[i].get("ovl", [])
        r2 = best_wiring(nums, slots, ovl)
        if r2 is None: continue
        t["emit"] += 1
        if r2[1] == rows[i]["answer"]: t["right"] += 1
        else: t["lies"] += 1
    for tag, t in T.items():
        if t["n"] == 0: continue
        print(f"[poy {tag}] n={t['n']} sworn {t['sworn']} EMIT {t['emit']} "
              f"right {t['right']} lies {t['lies']}", flush=True)

if __name__ == "__main__":
    main()
