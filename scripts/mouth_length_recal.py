"""mouth_length_recal.py — the length-controlled recalibration
(2026-07-14, gut #16's fix propagated to all three consumers).
Fit: distance ~ a + b/len on NATIVE (dag8test); the residual is the
length-controlled distance. Recalibrate: (1) deployment threshold
(99th pct native residual); (2) harvest odometer zero-point
(length-controlled); (3) book-1 diversity guard re-read.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from phase1_algebra_head import STATES_NPZ, STATES_NPY, T_ALG, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer

bank = np.load(".cache/recognition_mouth_gen9b.npz")["bank"]
def knn(V): return np.sort(1.0 - V @ bank.T, axis=1)[:, :8].mean(1)

z = np.load(STATES_NPZ.format(split="dag8test"))
st = np.load(STATES_NPY.format(split="dag8test"), mmap_mode="r")
tk = z["tokmask"].astype(np.float32)
V = np.zeros((st.shape[0], st.shape[-1]), np.float32)
for i in range(st.shape[0]):
    m = tk[i][:, None]
    V[i] = (np.asarray(st[i], np.float32) * m).sum(0) / max(m.sum(), 1)
V /= np.linalg.norm(V, axis=1, keepdims=True)
dn, Ln = knn(V), tk.sum(1)
X = np.stack([np.ones_like(Ln), 1.0 / Ln], 1)
coef, *_ = np.linalg.lstsq(X, dn, rcond=None)
res_n = dn - X @ coef
thr = float(np.percentile(res_n, 99))
print(f"[recal-L] fit d ~ {coef[0]:.4f} + {coef[1]:.2f}/len on native "
      f"(r after control: {np.corrcoef(Ln, res_n)[0,1]:+.3f})")
print(f"[recal-L] length-controlled native threshold: {thr:.4f} (was 0.0347 raw)")

tok = Tokenizer.from_file(TOKENIZER_JSON)
def pooled_texts(texts):
    out = []
    for s0 in range(0, len(texts), 8):
        chunk = texts[s0:s0+8]
        ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32)
        for i, t in enumerate(chunk):
            e = tok.encode(t); L = min(len(e.ids), T_ALG)
            ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        s = recompute_states(ids).astype(np.float32)
        v = (s * msk[:, :, None]).sum(1) / np.maximum(msk.sum(1)[:, None], 1)
        out.append((v[:len(chunk)], msk.sum(1)[:len(chunk)]))
    V = np.concatenate([a for a, _ in out]); L = np.concatenate([b for _, b in out])
    return V / np.linalg.norm(V, axis=1, keepdims=True), L

h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
hv, hl = pooled_texts([x["problem"] for x in h])
res_h = knn(hv) - (coef[0] + coef[1] / hl)
print(f"[recal-L] HARVEST length-controlled zero-point: {res_h.mean():.4f} "
      f"(raw was 0.2431); read-foreign at new thr: {(res_h > thr).mean():.1%}")

book = [json.loads(l) for l in open(".cache/book2.jsonl") if "raw" in l]
bv, bl = pooled_texts([b["raw"] for b in book[:60]])
res_b = knn(bv) - (coef[0] + coef[1] / bl)
print(f"[recal-L] BOOK-2 raws (n={len(bl)}): length-controlled mean {res_b.mean():.4f} "
      f"vs harvest {res_h.mean():.4f} (the diversity guard, straightened)")
np.savez(".cache/mouth_length_correction.npz", coef=coef, thr=np.float64(thr))
print("[recal-L] correction saved -> .cache/mouth_length_correction.npz")
