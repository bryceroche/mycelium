"""chain_decode.py — HMM v4: THE ATLAS-GUIDED SLOT-CHAIN DECODER
(2026-08-24; the MLIR x breaths superpower). The breathing head IS the
segment-and-classify machine: per slot per cycle, match the slot state
to the atlas's (cluster, cycle) centroids; Viterbi each slot's cluster
CHAIN through the empirical v3_transitions lowering graph; the chain's
endpoint kind_counts vote the op label; chain path-score = confidence.
Instrument: gsb227_real K=7 with ALG_MINE_BREATHS (the atlas's OWN
coordinates and machinery — waist_miner_v3's exact recipe).
Grade: op-multiset F1 on the 143 gold vs the flat-HMM curve
(v1 .373 / v2 .343 / v3 .296; bar 0.5).
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
from phase1_algebra_head import (build_params, forward, T_ALG,
                                 TOKENIZER_JSON, sent_indices, load_alg,
                                 build_slot_masks, L_FAC)
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from beacon_closing_arm import recompute_states

_ = load_alg("test")
tok = Tokenizer.from_file(TOKENIZER_JSON)
K_B = int(os.environ["ALG_BREATH"])
KINDS = ["rel", "given", "mod", "sel", "pct", "fdiv", "macro", "frac"]

def load_atlas():
    c = sqlite3.connect('.cache/campaign.db')
    cents = {}; kinds = {}
    for cid, cyc, cnt, mean, kc in c.execute(
            "SELECT cluster_id,breath_cycle,count,mean,kind_counts "
            "FROM waist_patterns_v3 WHERE count >= 3"):
        v = np.frombuffer(mean, np.float32).copy()
        cents.setdefault(cyc, {})[cid] = v / (np.linalg.norm(v) + 1e-9)
        if kc:
            kinds[cid] = json.loads(kc)
    trans = {}
    for cyc, a, b, cnt in c.execute("SELECT cycle,from_id,to_id,count FROM v3_transitions"):
        trans.setdefault(cyc, {}).setdefault(a, {})[b] = cnt
    c.close()
    return cents, kinds, trans

def main():
    cents, ckinds, trans = load_atlas()
    cycles = sorted(cents)
    print(f"[chain] atlas: cycles {cycles}, "
          f"{sum(len(v) for v in cents.values())} (cluster,cycle) centroids, "
          f"{sum(len(v) for v in trans.values())} transition rows", flush=True)
    p = build_params(0)
    sd = safe_load('.cache/gsb227_real.safetensors')
    assert set(sd.keys()) == set(p.keys()), \
        f"gsb227 key mismatch under current envs: {len(set(sd)-set(p))}/{len(set(p)-set(sd))}"
    for k in p:
        p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    byid = {}
    for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl')):
        for l in open(f): r = json.loads(l); byid[r["src_idx"]] = r
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); byid[r["src_idx"]] = r
    sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    gold = [v for k, v in sorted(byid.items()) if k not in sk]
    print(f"[chain] gold rows: {len(gold)}", flush=True)

    def gold_kinds(facs):
        return Counter(f["ftype"] for f in facs)

    f1s = []
    for s0 in range(0, len(gold), 8):
        sl = gold[s0:s0 + 8]
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
        onp = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp, snt)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
        assert "breaths_all" in o, "hook not engaged"
        Bst = [b.realize().numpy() for b in o["breaths_all"]]  # K x (B,L,H)
        pres = o["pres"].realize().numpy()
        for li, r in enumerate(sl):
            dec = []
            for j in range(L_FAC):
                if pres[li, j] <= 0.5: continue
                # per-cycle nearest cluster + chain score through transitions
                chain = []; score = 0.0
                for ci, cyc in enumerate(cycles[:len(Bst)]):
                    v = Bst[min(ci, len(Bst) - 1)][li, j]
                    v = v / (np.linalg.norm(v) + 1e-9)
                    bank = cents[cyc]
                    bids = list(bank.keys())
                    sims = np.array([float(v @ bank[b]) for b in bids])
                    order = np.argsort(-sims)[:3]
                    if chain:
                        prev = chain[-1]
                        tr = trans.get(cyc, {}).get(prev, {})
                        cand = [(sims[oi] + 0.2 * np.log1p(tr.get(bids[oi], 0)), oi)
                                for oi in order]
                        sbest, oi = max(cand)
                    else:
                        oi = order[0]; sbest = sims[oi]
                    chain.append(bids[oi]); score += float(sbest)
                # endpoint kind vote (walk back until a labeled cluster)
                lab = None
                for cid in reversed(chain):
                    if cid in ckinds and ckinds[cid]:
                        lab = max(ckinds[cid], key=ckinds[cid].get); break
                if lab: dec.append(lab)
            d = Counter(dec)
            g = gold_kinds(r["factors"])
            inter = sum((d & g).values())
            f1s.append(2 * inter / max(sum(d.values()) + sum(g.values()), 1))
    f1s = np.array(f1s)
    print(f"[chain] V4 KIND-MULTISET F1: mean {f1s.mean():.3f} median "
          f"{np.median(f1s):.3f} (flat-HMM curve .373/.343/.296; bar 0.5)  "
          f"rows>=0.5: {(f1s >= 0.5).sum()}/{len(f1s)}", flush=True)
    print("[chain] VERDICT: " + ("THE SUPERPOWER LANDS — slot-chain decode "
          "beats flat HMM" if f1s.mean() > 0.373 else
          "chains do not beat flat emissions at kind grain"), flush=True)

if __name__ == "__main__":
    main()
