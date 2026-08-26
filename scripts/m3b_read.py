"""m3b_read.py — M3b: SOLVER-CERTIFIED INFERENCE MASKS (2026-08-26; the
nazare constitution's clause 2, built after attempt 1 proved wall-trust
needs wall-truth). Eligible rows = chain op-multiset all enum-grade +
enumeration finds a UNIQUE reachable root over anchor-law surface
numbers: a witness TREE for that root becomes the canyon — tree ops
matched to rel-family slots by op label, leaves matched to given slots
by value; matched edges -> directed mask rows (consumer attends
producer + self); unmatched slots keep their heuristic row. Ineligible
rows keep the standard mask (no regime shift).
Read on BOTH gsb227_m3 (canyon-trained) and gsb227_sharp10k (untrained
control): BARS (pinned) — per ckpt, eligible-row net M3B > RAW-same-rows;
THE PAIRING QUESTION: m3's M3B gain > sharp10k's = canyon literacy is
real and lacked only truth. MINT watched (rights above same-row RAW).
"""
import os, sys, json, glob
os.environ.setdefault("ALG_MINE_BREATHS", "1")
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.setdefault("NB_PERSLOT", "1")
os.environ.setdefault("ATLAS_TABLE", "waist_patterns_sharp")
os.environ.setdefault("ATLAS_TRANS", "sharp_transitions")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import re
import numpy as np
from collections import Counter
from phase1_algebra_head import (build_params, forward, decode, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg,
                                 build_slot_masks, L_FAC)
from repair_replace_swap import solve_forced
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from beacon_closing_arm import recompute_states
from chain_decode import load_atlas
from iter_a0 import chain_labels
from enum_assembly import reachable

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")
NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.])")
ENUM_OPS = ("add", "sub", "mul", "sq", "fr")

def find_tree(leaves, ops, target, cap=200000):
    """first witness tree for `target`: steps [(op, src1, src2|None, val)]
    with src = ('n', leaf_idx) | ('t', step_idx). Mirrors reachable()'s
    exact semantics (sq in-place <=300; fr exact k>=2; add/sub/mul
    ordered, 0..300)."""
    calls = [0]
    avail0 = tuple((v, ('n', i)) for i, v in enumerate(leaves))
    out = []
    def rec(avail, ops_left, steps):
        calls[0] += 1
        if calls[0] > cap or out: return
        if not ops_left:
            for v, src in avail:
                if v == target:
                    out.append((steps, src)); return
            return
        n = len(avail)
        for op in sorted(set(ops_left)):
            rest = list(ops_left); rest.remove(op)
            if op == "sq":
                for i in range(n):
                    v = avail[i][0] * avail[i][0]
                    if v > 300: continue
                    st = steps + [("sq", avail[i][1], None, v)]
                    na = avail[:i] + ((v, ('t', len(steps))),) + avail[i+1:]
                    rec(na, tuple(rest), st)
                    if out: return
            elif op == "fr":
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        a, k2 = avail[i][0], avail[j][0]
                        if k2 < 2 or a % k2: continue
                        v = a // k2
                        st = steps + [("fr", avail[i][1], avail[j][1], v)]
                        na = tuple(x for t2, x in enumerate(avail)
                                   if t2 not in (i, j)) + ((v, ('t', len(steps))),)
                        rec(na, tuple(rest), st)
                        if out: return
            else:
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        a, b = avail[i][0], avail[j][0]
                        v = a + b if op == "add" else (a - b if op == "sub"
                                                      else a * b)
                        if not (0 <= v <= 300): continue
                        st = steps + [(op, avail[i][1], avail[j][1], v)]
                        na = tuple(x for t2, x in enumerate(avail)
                                   if t2 not in (i, j)) + ((v, ('t', len(steps))),)
                        rec(na, tuple(rest), st)
                        if out: return
    rec(avail0, tuple(ops), [])
    return out[0] if out else None

def canyon_mask(base_row, facs, present, lab, nums, tree_steps):
    """tree -> directed slot mask. rel-family slots matched to tree ops
    by chain label (greedy in order); given slots matched to leaves by
    VALUE. Matched consumer attends its producers; unmatched slots keep
    their heuristic row."""
    m = base_row.copy()
    slot_of_fac = {fi: present[fi] for fi in range(len(facs))
                   if fi < len(present)}
    op_slots = {}          # op label -> [slot ids] in slot order
    giv_val = {}           # value -> [slot ids]
    for fi, f in enumerate(facs):
        if fi not in slot_of_fac: continue
        j = slot_of_fac[fi]
        cl = lab.get(j)
        if cl in ENUM_OPS:
            op_slots.setdefault(cl, []).append(j)
        elif f["ftype"] == "given":
            v = f.get("value")
            if v is not None:
                giv_val.setdefault(int(v), []).append(j)
    step_slot = {}
    used = {k2: 0 for k2 in op_slots}
    for si, (op, s1, s2, v) in enumerate(tree_steps):
        pool = op_slots.get(op, [])
        if used.get(op, 0) < len(pool):
            step_slot[si] = pool[used[op]]; used[op] += 1
    def src_slot(src):
        if src is None: return None
        kind, idx = src
        if kind == 't': return step_slot.get(idx)
        v = nums[idx]
        lst = giv_val.get(v, [])
        return lst[0] if lst else None
    wired = set()
    for si, (op, s1, s2, v) in enumerate(tree_steps):
        j = step_slot.get(si)
        if j is None: continue
        row = np.zeros(L_FAC, np.float32); row[j] = 1.0
        for src in (s1, s2):
            k2 = src_slot(src)
            if k2 is not None: row[k2] = 1.0
        m[j] = row; wired.add(j)
    return m, len(wired)

def main():
    cents, ckinds, trans = load_atlas()
    cycles = sorted(cents)
    ck = os.environ.get("M3B_CKPT", "gsb227_m3")
    p = build_params(0)
    sd = safe_load(f'.cache/{ck}.safetensors')
    assert set(sd.keys()) == set(p.keys())
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    gold = [{"original": v["original"], "answer": v["answer"], "tag": "gold"}
            for k, v in sorted(byid.items()) if k not in sk]
    wv = [json.loads(l) for l in open('.cache/g55_wildval.jsonl')]
    never = [{"original": r["original"], "answer": r["answer"], "tag": "wv"}
             for r in wv]
    dd = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
    h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
    drafted = set(byid) | sk | set(r["src_idx"] for r in dd)
    for seed in (99, 299):
        rg = np.random.default_rng(seed)
        never += [{"original": h[i]["problem"],
                   "answer": int(str(h[i]["answer"]).strip()), "tag": "held"}
                  for i in rg.permutation(len(h)) if i not in drafted
                  and str(h[i]["answer"]).strip().isdigit()][:10]
    rows = gold + never
    T = {t: {"raw": [0, 0, 0], "m3b": [0, 0, 0], "elig": 0}
         for t in ("gold", "wv", "held")}
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
        onp0 = {k2: o0[k2].realize().numpy() for k2 in ("fat", "args", "res")}
        M0 = build_slot_masks(onp0, snt)
        o1 = forward(p, ts, tk, se, slot_mask=Tensor(M0, dtype=dtypes.float))
        ex = tuple(k2 for k2 in ("sel", "dup", "sgn") if k2 in o1)
        onp1 = {k2: o1[k2].realize().numpy() for k2 in K + ex}
        Bst = [b.realize().numpy() for b in o1["breaths_all"]]
        MC = M0.copy(); any_elig = [False] * 8
        for li, r in enumerate(sl):
            lab = chain_labels(Bst, cents, ckinds, trans, cycles, li,
                               onp1["pres"][li])
            facs, q = decode({k2: onp1[k2][li] for k2 in onp1})
            present = [j for j in range(L_FAC) if onp1["pres"][li, j] > 0.5]
            ops = [lab[j] for j in present if lab.get(j) in ENUM_OPS]
            allops = [lab[j] for j in present if lab.get(j) is not None
                      and lab[j] not in ("given", "mod", "sel", "pct", "fdiv")]
            if not ops or len(ops) != len(allops) or len(ops) > 6: continue
            nums = [int(mm.group(1)) for mm in NUM.finditer(r["original"])]
            if not nums or len(nums) > 8: continue
            roots, blown = reachable(nums, ops)
            if blown or len(roots) != 1: continue
            tr = find_tree(nums, ops, next(iter(roots)))
            if tr is None: continue
            MC[li], nw = canyon_mask(MC[li], facs, present, lab, nums, tr[0])
            if nw:
                any_elig[li] = True
        o2 = forward(p, ts, tk, se, slot_mask=Tensor(MC, dtype=dtypes.float))
        onp2 = {k2: o2[k2].realize().numpy() for k2 in K + ex}
        for li, r in enumerate(sl):
            if not any_elig[li]: continue
            t = T[r["tag"]]; t["elig"] += 1
            for arm, onp in (("raw", onp1), ("m3b", onp2)):
                facs, q = decode({k2: onp[k2][li] for k2 in onp})
                try:
                    a = solve_forced(facs, q, {"n_vars": 24, "m": 300})
                except Exception:
                    a = None
                if a is not None:
                    t[arm][0] += 1
                    if a == r["answer"]: t[arm][1] += 1
                    else: t[arm][2] += 1
    for tag in ("gold", "wv", "held"):
        t = T[tag]
        rw, mb = t["raw"], t["m3b"]
        print(f"[m3b {ck} {tag}] eligible {t['elig']}  RAW f{rw[0]} r{rw[1]} "
              f"l{rw[2]} (net {rw[1]-rw[2]})  |  M3B f{mb[0]} r{mb[1]} "
              f"l{mb[2]} (net {mb[1]-mb[2]})", flush=True)

if __name__ == "__main__":
    main()
