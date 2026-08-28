"""mouth_recal_gen9b.py — THE MOUTH RECALIBRATION (2026-07-13, entourage
rebuild step 1, sequenced FIRST per the relay's adjudication).

The mouth is INPUT-SPACE (pooled frozen-trunk geometry): the gen-5 bank
was drawn from alg4train — the family as of 2026-07-10. The native family
has since grown by dag6/7/7b/8/9/booster corpora. Recalibration = re-draw
the bank from the CURRENT family (m9btrain), re-derive the native
threshold on held-out native text (dag8test), and run the FREE
RETROACTIVE READ: harvest odometer + census-pool distances under BOTH
lenses — the zero-point corrected and the history re-scored from disk.

Precision note (correcting the record): the census's knotted counts never
consult the mouth (parse+vote+key only) — the 76->89 trend is real parse
behavior. What the stale mouth DID skew: the deployment chain-of-custody,
the harvest odometer (0.2488), and book-1's diversity guards.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, sent_indices, TOKENIZER_JSON, STATES_NPZ, STATES_NPY
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer

def pooled_split(split, sample=None, seed=0):
    z = np.load(STATES_NPZ.format(split=split))
    tk = z["tokmask"].astype(np.float32)
    npy = STATES_NPY.format(split=split)
    st = (np.load(npy, mmap_mode="r") if os.path.exists(npy)
          else z["states"])
    idx = np.arange(st.shape[0])
    if sample and sample < len(idx):
        idx = np.random.RandomState(seed).choice(len(idx), sample, replace=False)
        idx.sort()
    v = np.zeros((len(idx), st.shape[-1]), np.float32)
    for i, j in enumerate(idx):
        s = np.asarray(st[j], np.float32)
        m = tk[j][:, None]
        v[i] = (s * m).sum(0) / max(m.sum(), 1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def pooled_texts(texts):
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    out = []
    for s0 in range(0, len(texts), 8):
        chunk = texts[s0:s0+8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        for i, t in enumerate(chunk):
            e = tok.encode(t)
            L = min(len(e.ids), T_ALG)
            ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        st = recompute_states(ids).astype(np.float32)
        v = (st * msk[:, :, None]).sum(1) / np.maximum(msk.sum(1)[:, None], 1)
        out.append(v[:len(chunk)])
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def knn(V, bank):
    d = 1.0 - V @ bank.T
    return np.sort(d, axis=1)[:, :8].mean(1)

print("[recal] drawing the gen-9b bank from m9btrain (the current family)...")
bank_new = pooled_split("m9btrain", sample=2000, seed=0)
old = np.load(".cache/recognition_mouth_gen5.npz")
bank_old = old["bank"]

native = pooled_split("dag8test")
thr_new = float(np.percentile(knn(native, bank_new), 99))
print(f"[recal] native threshold: gen-5 {float(old['thr_knn']):.4f} -> "
      f"gen-9b {thr_new:.4f} (99th pct held-out native)")

print("[recal] retroactive read: the harvest odometer under both lenses...")
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
hv = pooled_texts([x["problem"] for x in h])
d_old, d_new = knn(hv, bank_old), knn(hv, bank_new)
print(f"  HARVEST (n={len(h)}): mean kNN gen-5 lens {d_old.mean():.4f} "
      f"(banked zero-point 0.2488) -> gen-9b lens {d_new.mean():.4f}")
print(f"  read-foreign rate: gen-5 {float((d_old > old['thr_knn']).mean()):.1%} "
      f"-> gen-9b {float((d_new > thr_new).mean()):.1%}")

pool = [x for x in h if x["level"] in ("Level 1", "Level 2", "Level 3")][:100]
pv = pooled_texts([x["problem"] for x in pool])
print(f"  CENSUS POOL (100): mean gen-5 {knn(pv, bank_old).mean():.4f} -> "
      f"gen-9b {knn(pv, bank_new).mean():.4f}")

book = [json.loads(l) for l in open(".cache/book1.jsonl")]
bv = pooled_texts([b["raw"] for b in book])
print(f"  BOOK-1 RAW (n={len(book)}): mean gen-5 {knn(bv, bank_old).mean():.4f} -> "
      f"gen-9b {knn(bv, bank_new).mean():.4f}")

np.savez(".cache/recognition_mouth_gen9b.npz", bank=bank_new,
         thr_knn=np.float64(thr_new))
print("[recal] saved .cache/recognition_mouth_gen9b.npz — the odometer re-zeroed")
