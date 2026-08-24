"""assembly_game.py — THE ASSEMBLY GAME, A0 (2026-08-24, word given):
the chain-consistency filter. gsb227 parses; v4's atlas-chains audit
each present slot; slots whose chain endpoint kind DISAGREES with the
head's decoded ftype are AMPUTATED (atlas-powered surgery); the
survivor graph solve_forced's or refuses. Graded RAW vs FILTERED on the
143 gold + wild-val 20 + held-out 20: assembly's value = rights gained
+ lies cut. Bars: filtered lies <= raw lies on the 40 never-seen;
rights delta reported with the honest floor.
"""
import os, sys, json, glob
os.environ.setdefault("ALG_MINE_BREATHS", "1")
os.environ.setdefault("ALG_BREATH", "7")
os.environ.setdefault("ALG_NOTEBOOK", "1")
os.environ.setdefault("ALG_SIXWAVE", "1")
os.environ.update({"DEV": "AMD", "ALG2": "1", "ALG_FTYPES": "9",
                   "ALG_DUP": "1", "ALG_HW": "512", "ALG_WIDE": "1",
                   "ALG_TEST": ".cache/algebra_nl_bigtest.jsonl",
                   "ALG_TEST_NAME": "bigtest"})
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import sqlite3
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

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
K = ("pres", "ftype", "op", "islit", "dig", "args", "res", "query")

def main():
    cents, ckinds, trans = load_atlas()
    cycles = sorted(cents)
    p = build_params(0)
    sd = safe_load('.cache/gsb227_real.safetensors')
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
    tallies = {t: {"raw": [0, 0, 0], "filt": [0, 0, 0]}
               for t in ("gold", "wv", "held")}   # [forced, right, lies]
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
        ex = tuple(k2 for k2 in ("sel", "dup", "sgn") if k2 in o)
        onp = {k2: o[k2].realize().numpy() for k2 in K + ex}
        Bst = [b.realize().numpy() for b in o["breaths_all"]]
        pres = onp["pres"]
        for li, r in enumerate(sl):
            # chain label per present slot
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
            # raw parse
            facs, q = decode({k2: onp[k2][li] for k2 in onp})
            try:
                a_raw = solve_forced(facs, q, {"n_vars": 24, "m": 300})
            except Exception:
                a_raw = None
            # chain-consistency filter: drop slots whose head ftype
            # disagrees with the chain label (order of present slots
            # matches decode's factor order)
            present = [j for j in range(L_FAC) if pres[li, j] > 0.5]
            keep = []
            for fi, f in enumerate(facs):
                j = present[fi] if fi < len(present) else None
                cl = chain_lab.get(j)
                if cl is None or cl == f["ftype"]:
                    keep.append(f)
            try:
                a_f = solve_forced(keep, q, {"n_vars": 24, "m": 300})
            except Exception:
                a_f = None
            T = tallies[r["tag"]]
            for name, a in (("raw", a_raw), ("filt", a_f)):
                if a is not None:
                    T[name][0] += 1
                    if a == r["answer"]: T[name][1] += 1
                    else: T[name][2] += 1
    for t, T in tallies.items():
        n = {"gold": 143, "wv": 20, "held": 20}[t]
        print(f"[asm {t}] RAW forced {T['raw'][0]}/{n} right {T['raw'][1]} "
              f"lies {T['raw'][2]}  |  FILTERED forced {T['filt'][0]} right "
              f"{T['filt'][1]} lies {T['filt'][2]}", flush=True)

if __name__ == "__main__":
    main()
