"""true_assembly.py — RUNG 2: TRUE ASSEMBLY, A1 (2026-08-24, word
given). Factors BORN from decomposition: op label from the op-atlas
chain (the classifier that reads 0.531 above bar); given VALUES from
the slot's fat-span surface numbers (the anchor law as binder — the
head's weak digit emissions bypassed); slot WIRING (args/res pointers)
from the head — demoted to proposer. opa/fr slots fall back to full
head decode (v1 scope). Solve -> grade RAW vs A0-filter vs A1-true on
gold-143 + wild-val 20 + held-out 20. Bars: A1 lies <= A0 lies on the
never-seen 40; rights delta reported.
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
import numpy as np
import re
from phase1_algebra_head import (build_params, forward, decode, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg,
                                 build_slot_masks, L_FAC)
from repair_replace_swap import solve_forced
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from beacon_closing_arm import recompute_states
from chain_decode import load_atlas

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")
NUM = re.compile(r"(?<![\d.])(\d+)(?![\d.])")

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
    tallies = {t: {m: [0, 0, 0] for m in ("raw", "true")}
               for t in ("gold", "wv", "held")}
    OPMAP = {"add": ("rel", "add"), "sub": ("rel", "sub"),
             "mul": ("rel", "mul"), "sq": ("rel", "mul")}
    for s0 in range(0, len(rows), 8):
        sl = rows[s0:s0 + 8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        snt = np.zeros((8, T_ALG), np.int32)
        offs = [None] * 8
        for li, r in enumerate(sl):
            e = tok.encode(r["original"])
            if len(e.ids) > T_ALG: continue
            ids[li, :len(e.ids)] = e.ids; msk[li, :len(e.ids)] = 1.0
            snt[li] = sent_indices(r["original"], list(e.offsets), msk[li])
            offs[li] = list(e.offsets)
        sts = np.asarray(recompute_states(ids)).astype(np.float32)
        ts = Tensor(sts, dtype=dtypes.float)
        tk = Tensor(msk, dtype=dtypes.float)
        se = Tensor(snt.astype(np.int32), dtype=dtypes.int)
        o0 = forward(p, ts, tk, se)
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
        ex = tuple(k2 for k2 in ("sel", "dup", "sgn", "fat") if k2 in o)
        onp = {k2: o[k2].realize().numpy() for k2 in K + ex}
        Bst = [b.realize().numpy() for b in o["breaths_all"]]
        fat = onp["fat"]; pres = onp["pres"]
        for li, r in enumerate(sl):
            if offs[li] is None: continue
            text = r["original"]
            # surface numbers with token index sets
            nums = []
            for m in NUM.finditer(text):
                toks = [ti for ti, (a, b) in enumerate(offs[li])
                        if a < m.end() and b > m.start()]
                if toks: nums.append((int(m.group(1)), toks))
            # chain op per slot
            chain_lab = {}
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
                        tr = trans.get(cyc, {}).get(chain[-1], {})
                        cand = [(sims[oi] + 0.2 * np.log1p(tr.get(bids[oi], 0)), oi)
                                for oi in order]
                        _, oi = max(cand)
                    else:
                        oi = order[0]
                    chain.append(bids[oi])
                for cid in reversed(chain):
                    if cid in ckinds and ckinds[cid]:
                        chain_lab[j] = max(ckinds[cid], key=ckinds[cid].get)
                        break
            facs_raw, q = decode({k2: onp[k2][li] for k2 in onp})
            # A2: EXCLUSIVE VALUE BINDING — greedy 1:1 matching of given-
            # slots x surface numbers by fat mass; each number spent once
            _present = [j for j in range(L_FAC) if pres[li, j] > 0.5]
            _gslots = [(_present[fi] if fi < len(_present) else None, fi)
                       for fi, f in enumerate(facs_raw)]
            _cand = []
            for j, fi in _gslots:
                if j is None: continue
                lab = None  # chain label known below; score all, filter later
                for ni, (nv, toks) in enumerate(nums):
                    _cand.append((float(fat[li, j, toks].sum()), fi, ni))
            if os.environ.get("A2_BIND", "fat") == "order":
                _assign = {}; _ni = 0
                for fi, f in enumerate(facs_raw):
                    if f["ftype"] == "given" and _ni < len(nums):
                        _assign[fi] = _ni; _ni += 1
            else:
                _cand.sort(reverse=True)
                _assign = {}; _used = set()
                for sc, fi, ni in _cand:
                    if fi in _assign or ni in _used: continue
                    _assign[fi] = ni; _used.add(ni)
            try:
                a_raw = solve_forced(facs_raw, q, {"n_vars": 24, "m": 300})
            except Exception:
                a_raw = None
            # TRUE ASSEMBLY: rebuild each present slot's factor
            present = [j for j in range(L_FAC) if pres[li, j] > 0.5]
            born = []
            ok = True
            for fi, f in enumerate(facs_raw):
                j = present[fi] if fi < len(present) else None
                lab = chain_lab.get(j)
                if lab == "given" or (lab is None and f["ftype"] == "given"):
                    if fi in _assign:
                        born.append({"ftype": "given", "var": f.get("var", 0),
                                     "value": nums[_assign[fi]][0]})
                    else:
                        born.append(f)   # no exclusive number left: head value
                elif lab in OPMAP:
                    ft, op = OPMAP[lab]
                    g = {"ftype": ft, "op": op,
                         "args": f.get("args", [0, 0]) if f["ftype"] == "rel"
                         else [f.get("x", 0), f.get("y", 0)],
                         "result": f.get("result", 0)}
                    if lab == "sq" and len(g["args"]) == 2:
                        g["args"] = [g["args"][0], g["args"][0]]
                    born.append(g)
                else:
                    born.append(f)      # opa/fr/unknown: head's proposal
            if ok and born:
                try:
                    a_true = solve_forced(born, q, {"n_vars": 24, "m": 300})
                except Exception:
                    a_true = None
            else:
                a_true = None
            T = tallies[r["tag"]]
            for name, a in (("raw", a_raw), ("true", a_true)):
                if a is not None:
                    T[name][0] += 1
                    if a == r["answer"]: T[name][1] += 1
                    else: T[name][2] += 1
    for t, T in tallies.items():
        n = {"gold": 143, "wv": 20, "held": 20}[t]
        print(f"[asm2 {t}] RAW forced {T['raw'][0]}/{n} right {T['raw'][1]} "
              f"lies {T['raw'][2]}  |  TRUE forced {T['true'][0]} right "
              f"{T['true'][1]} lies {T['true'][2]}", flush=True)

if __name__ == "__main__":
    main()
