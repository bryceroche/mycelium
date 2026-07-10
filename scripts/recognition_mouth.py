"""recognition_mouth.py — THE FUNNEL'S MOUTH (2026-07-10): input-space OOD
recognition, the fourth jurisdiction. Zero-parameter: pooled trunk-state
geometry against the training family. SELECTION-SAFE by position (upstream
of the parse; no pipeline pressure shapes errors against it).

REGISTERED: (P1) foreign (MATH-500) vs native (held-out generator text)
separates at AUC >= 0.95 (they are different LANGUAGES, not hard cases);
(P2) at the 99th-percentile-native threshold, the mouth refuses >= 90% of
MATH-500 — including ~all 160 of the anchor's false certifications;
(P3) gradation preview (the atlas's leaf-vs-forest read): MATH-500 algebra
subjects read NEARER the family than geometry/precalculus.
Artifact: .cache/recognition_mouth.npz (joins the manifest at next bump).
"""
import json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from survivor_multiplicity import midrank_auc
from tokenizers import Tokenizer

def pooled(npz_path):
    z = np.load(npz_path)
    st, tk = z["states"].astype(np.float32), z["tokmask"].astype(np.float32)
    v = (st * tk[:, :, None]).sum(1) / np.maximum(tk.sum(1)[:, None], 1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

print("[mouth] pooling the training family + native draws (banked npz)...")
fam = pooled(".cache/phase1_alg_states_alg4train.npz")
native = pooled(".cache/phase1_alg_states_alg4test.npz")
native2 = pooled(".cache/phase1_alg_states_bigtest.npz")
cent = fam.mean(0); cent /= np.linalg.norm(cent)
rng = np.random.RandomState(0)
bank = fam[rng.choice(len(fam), 2000, replace=False)]

def scores(V):
    c = 1.0 - V @ cent                                  # centroid distance
    d = 1.0 - (V @ bank.T)                              # kNN (cosine) distance
    knn = np.sort(d, axis=1)[:, :8].mean(1)
    return c, knn

print("[mouth] computing MATH-500 trunk states (foreign)...")
rows = [json.loads(l) for l in open(".cache/math500_test.jsonl")]
tok = Tokenizer.from_file(TOKENIZER_JSON)
ids = np.zeros((len(rows), T_ALG), np.int32)
msk = np.zeros((len(rows), T_ALG), np.float32)
kept = []
for i, r in enumerate(rows):
    e = tok.encode(r["problem"])
    if len(e.ids) > T_ALG:
        continue
    kept.append(i)
    ids[i, :len(e.ids)] = e.ids; msk[i, :len(e.ids)] = 1.0
st = recompute_states(ids)
fv = (st.astype(np.float32) * msk[:, :, None]).sum(1) / \
    np.maximum(msk.sum(1)[:, None], 1)
fv = fv / np.linalg.norm(fv, axis=1, keepdims=True)
foreign = fv[kept]

nc, nk = scores(np.vstack([native, native2]))
fc, fk = scores(foreign)
print(f"\n=== THE MOUTH (foreign n={len(foreign)}, native n={len(nc)}) ===")
for nm, ns, fs in (("centroid", nc, fc), ("kNN-8", nk, fk)):
    auc = midrank_auc(fs, ns)
    thr = np.percentile(ns, 99)
    catch = float((fs > thr).mean())
    print(f"  {nm:9s}: AUC {auc:.4f} | thr@99pct-native {thr:.4f} | "
          f"foreign refused {catch:.1%} | native false-foreign "
          f"{float((ns > thr).mean()):.1%}")

# P2's teeth: what would the mouth have done to the anchor's certifications?
thr = np.percentile(nk, 99)
out = json.load(open(".cache/math500_anchor_outcomes.json"))
kept_set = {i: j for j, i in enumerate(kept)}
refused_cert = tot_cert = 0
for o in out:
    if o["decision"] == "certify":
        tot_cert += 1
        j = kept_set.get(o["i"])
        if j is not None and fk[j] > thr:
            refused_cert += 1
print(f"\n  P2 read: mouth refuses {refused_cert}/{tot_cert} of the anchor's "
      f"certifications at the kNN threshold")

# P3: gradation by subject (leaf-vs-forest preview)
subj = {}
for j, i in enumerate(kept):
    subj.setdefault(rows[i]["subject"], []).append(fk[j])
print(f"  P3 gradation (mean kNN distance by subject):")
for s, v in sorted(subj.items(), key=lambda kv: np.mean(kv[1])):
    print(f"    {s:24s} {np.mean(v):.4f}")
np.savez(".cache/recognition_mouth.npz", centroid=cent, bank=bank,
         thr_knn=thr, thr_centroid=np.percentile(nc, 99))
print(f"  [saved] .cache/recognition_mouth.npz (the organ's artifact)")
