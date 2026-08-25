"""enum_assembly.py — WIRING ROAD (d): ENUMERATIVE ASSEMBLY (2026-08-24,
word given; the two-jaws law one level up — neural proposes OPS,
symbolic search disposes WIRING). Per row: decoded op multiset (chains)
+ surface-number leaves -> enumerate value-trees (ops in any order,
operands from {unused leaves + previous results}, domain 0..300 pruned,
sub-nonneg, fr-exact; leaves may go unused) -> the set of reachable
roots. METRICS: (1) COVERAGE — key in reachable set (decides the
program: high = disambiguation is the remaining game; low = ops/tree
convention insufficient); (2) UNIQUE-EMIT — rows with exactly one
reachable root: emit, grade rights/lies. opa-labeled rows refuse (v1).
"""
import os, sys, json, glob
os.environ.setdefault("ALG_MINE_BREATHS", "1")
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.setdefault("ATLAS_TABLE", "waist_patterns_op")
os.environ.setdefault("ATLAS_TRANS", "op_transitions")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import re
import numpy as np
from functools import lru_cache
from phase1_algebra_head import (build_params, forward, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg,
                                 build_slot_masks, L_FAC)
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from beacon_closing_arm import recompute_states
from chain_decode import load_atlas

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.])")

def reachable(leaves, ops, cap=200000):
    """set of reachable roots: ops applied in any order over
    {unused leaves + intermediate results}; each value node single-use."""
    calls = [0]
    seen_states = set()
    roots = set()
    def rec(avail, ops_left):
        calls[0] += 1
        if calls[0] > cap: return
        key = (tuple(sorted(avail)), tuple(sorted(ops_left)))
        if key in seen_states: return
        seen_states.add(key)
        if not ops_left:
            for v in avail: roots.add(v)
            return
        n = len(avail)
        for oi, op in enumerate(set(ops_left)):
            rest = list(ops_left); rest.remove(op)
            if op == "sq":
                for i in range(n):
                    v = avail[i] * avail[i]
                    if v > 300: continue
                    rec(avail[:i] + (v,) + avail[i + 1:], tuple(rest))
            elif op == "fr":
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        a, k = avail[i], avail[j]
                        if k < 2 or a % k: continue
                        na = tuple(x for t2, x in enumerate(avail)
                                   if t2 not in (i, j)) + (a // k,)
                        rec(na, tuple(rest))
            else:
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        a, b = avail[i], avail[j]
                        v = a + b if op == "add" else (a - b if op == "sub"
                                                      else a * b)
                        if not (0 <= v <= 300): continue
                        na = tuple(x for t2, x in enumerate(avail)
                                   if t2 not in (i, j)) + (v,)
                        rec(na, tuple(rest))
    rec(tuple(leaves), tuple(ops))
    return roots, calls[0] > cap

def main():
    cents, ckinds, trans = load_atlas()
    cycles = sorted(cents)
    p = build_params(0)
    sd = safe_load('.cache/gsb227_real.safetensors')
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    rows = [{"original": v["original"], "answer": v["answer"], "tag": "gold"}
            for k, v in sorted(byid.items()) if k not in sk]
    wv = [json.loads(l) for l in open('.cache/g55_wildval.jsonl')]
    rows += [{"original": r["original"], "answer": r["answer"], "tag": "wv"}
             for r in wv]
    dd = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
    h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
    drafted = set(byid) | sk | set(r["src_idx"] for r in dd)
    for seed in (99, 299):
        rg = np.random.default_rng(seed)
        rows += [{"original": h[i]["problem"],
                  "answer": int(str(h[i]["answer"]).strip()), "tag": "held"}
                 for i in rg.permutation(len(h)) if i not in drafted
                 and str(h[i]["answer"]).strip().isdigit()][:10]
    T = {t: {"n": 0, "attempted": 0, "cover": 0, "uniq": 0, "uright": 0,
             "ulies": 0, "blown": 0} for t in ("gold", "wv", "held")}
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        for li, r in enumerate(sl):
            e = tok.encode(r["original"])
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(r["original"], list(e.offsets), msk[li])
        sts = np.asarray(recompute_states(ids)).astype(np.float32)
        ts = Tensor(sts, dtype=dtypes.float)
        tk = Tensor(msk, dtype=dtypes.float)
        se = Tensor(snt.astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
        Bst = [b.realize().numpy() for b in o["breaths_all"]]
        pres = o["pres"].realize().numpy()
        for li, r in enumerate(sl):
            t = T[r["tag"]]; t["n"] += 1
            nums = [int(m.group(1)) for m in NUM.finditer(r["original"])]
            labs = []
            for j in range(L_FAC):
                if pres[li, j] <= 0.5: continue
                chain = []
                for ci, cyc in enumerate(cycles[:len(Bst)]):
                    v = Bst[min(ci, len(Bst) - 1)][li, j]
                    v = v / (np.linalg.norm(v) + 1e-9)
                    bank = cents[cyc]; bids = list(bank.keys())
                    sims = np.array([float(v @ bank[b]) for b in bids])
                    order = np.argsort(-sims)[:3]
                    if chain:
                        tr2 = trans.get(cyc, {}).get(chain[-1], {})
                        cand = [(sims[oi] + 0.2 * np.log1p(tr2.get(bids[oi], 0)), oi)
                                for oi in order]
                        _, oi = max(cand)
                    else:
                        oi = order[0]
                    chain.append(bids[oi])
                for cid in reversed(chain):
                    if cid in ckinds and ckinds[cid]:
                        labs.append(max(ckinds[cid], key=ckinds[cid].get))
                        break
            ops = [l for l in labs if l in ("add", "sub", "mul", "sq", "fr")]
            if not nums or not ops or len(nums) > 8 or len(ops) > 6 or \
               any(l == "opa" for l in labs):
                continue                           # refuse (v1 scope)
            t["attempted"] += 1
            roots, blown = reachable(nums, ops)
            if blown: t["blown"] += 1
            if r["answer"] in roots: t["cover"] += 1
            if len(roots) == 1:
                t["uniq"] += 1
                if r["answer"] in roots: t["uright"] += 1
                else: t["ulies"] += 1
    for tag, t in T.items():
        print(f"[enum {tag}] n={t['n']} attempted {t['attempted']} "
              f"COVERAGE {t['cover']}/{t['attempted']} "
              f"unique {t['uniq']} (right {t['uright']} lies {t['ulies']}) "
              f"cap-blown {t['blown']}", flush=True)

if __name__ == "__main__":
    main()
