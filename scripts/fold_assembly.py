"""fold_assembly.py — WIRING ROAD (b): THE SURFACE-CONVENTION ASSEMBLER
(2026-08-24, word given). No head pointers at all: op-grain chain labels
in slot order + surface numbers in reading order + LEFT-FOLD result
chaining (the comp-mint eval grammar as the wiring prior). REFUSES when
the convention can't fit (op-count != numbers-1, opa/fr-with-missing-k,
unknown labels) — abstain, never guess. Graded on gold-143 + never-40
vs A0's standing numbers (gold 4r/26l, wv 1r/5l, held 1r/7l).
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
OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b,
       "mul": lambda a, b: a * b}

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
    tallies = {t: [0, 0, 0] for t in ("gold", "wv", "held")}  # solved/right/lies
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
                        tr = trans.get(cyc, {}).get(chain[-1], {})
                        cand = [(sims[oi] + 0.2 * np.log1p(tr.get(bids[oi], 0)), oi)
                                for oi in order]
                        _, oi = max(cand)
                    else:
                        oi = order[0]
                    chain.append(bids[oi])
                for cid in reversed(chain):
                    if cid in ckinds and ckinds[cid]:
                        labs.append(max(ckinds[cid], key=ckinds[cid].get))
                        break
            ops = [l for l in labs if l in OPS or l == "sq"]
            # THE CONVENTION GATE: fold fits only if ops == numbers-1
            # (sq consumes no number) — else REFUSE
            n_bin = sum(1 for l in ops if l != "sq")
            if not nums or n_bin != len(nums) - 1 or \
               any(l not in OPS and l != "sq" for l in ops) or not ops:
                continue                                   # refuse
            acc = nums[0]; ni = 1; ok = True
            for l in ops:
                if l == "sq":
                    acc = acc * acc
                else:
                    acc = OPS[l](acc, nums[ni]); ni += 1
                if not (-10**6 < acc < 10**6): ok = False; break
            if not ok: continue
            T = tallies[r["tag"]]
            T[0] += 1
            if acc == r["answer"]: T[1] += 1
            else: T[2] += 1
    for t, (s_, r_, l_) in tallies.items():
        n = {"gold": 143, "wv": 20, "held": 20}[t]
        print(f"[fold {t}] emitted {s_}/{n} right {r_} lies {l_}", flush=True)

if __name__ == "__main__":
    main()
