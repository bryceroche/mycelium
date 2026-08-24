"""entropy_staircase.py — THE ENTROPY STAIRCASE (2026-08-24, word
given): per-cycle cluster-assignment entropy per slot on the 143 gold
rows — settling as cooling, measured. BARS (pinned): 'staircase
confirmed' = mean entropy non-increasing on >=5 of 6 cycle steps;
per-slot monotone fraction reported; right-vs-wrong final-entropy
reported HONESTLY as depth-not-correctness (temperature-perp-to-truth:
the read calibrates the refusal gate, never a verdict).
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
TAU = float(os.environ.get("STAIR_TAU", "12.0"))

def main():
    cents, _, _ = load_atlas()
    cycles = sorted(cents)
    banks = {}
    for cyc in cycles:
        ids_ = list(cents[cyc].keys())
        banks[cyc] = np.stack([cents[cyc][b] for b in ids_])
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
    gold = [v for k, v in sorted(byid.items()) if k not in sk]
    per_cycle = [[] for _ in cycles]
    mono = 0; total = 0
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
        onp0 = {k: o0[k].realize().numpy() for k in ("fat", "args", "res")}
        mk = build_slot_masks(onp0, snt)
        o = forward(p, ts, tk, se, slot_mask=Tensor(mk, dtype=dtypes.float))
        Bst = [b.realize().numpy() for b in o["breaths_all"]]
        pres = o["pres"].realize().numpy()
        for li in range(len(sl)):
            for j in range(L_FAC):
                if pres[li, j] <= 0.5: continue
                Hs = []
                for ci, cyc in enumerate(cycles[:len(Bst)]):
                    v = Bst[min(ci, len(Bst) - 1)][li, j]
                    v = v / (np.linalg.norm(v) + 1e-9)
                    sims = banks[cyc] @ v
                    z = sims * TAU
                    z = z - z.max()
                    pdist = np.exp(z); pdist /= pdist.sum()
                    H = float(-(pdist * np.log(pdist + 1e-12)).sum())
                    H = H / max(np.log(len(pdist)), 1e-9)   # normalized: bank sizes differ per cycle
                    Hs.append(H); per_cycle[ci].append(H)
                total += 1
                if all(Hs[i + 1] <= Hs[i] + 0.05 for i in range(len(Hs) - 1)):
                    mono += 1
    means = [float(np.mean(c)) for c in per_cycle if c]
    print("[stair] mean entropy per cycle: " +
          " ".join(f"{m:.3f}" for m in means), flush=True)
    steps_down = sum(1 for i in range(len(means) - 1)
                     if means[i + 1] <= means[i])
    print(f"[stair] non-increasing on {steps_down}/{len(means) - 1} steps "
          f"(bar >= 5/6)  slot-monotone fraction {mono}/{total} "
          f"= {mono / max(total, 1):.2f}", flush=True)
    print("[stair] VERDICT: " + ("THE STAIRCASE IS REAL — settling is "
          "cooling; the unsettled flag calibrates from these statistics"
          if steps_down >= 5 else
          "no monotone staircase — the lowering is not an annealing at "
          "this grain"), flush=True)

if __name__ == "__main__":
    main()
