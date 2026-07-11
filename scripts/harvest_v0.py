"""harvest_v0.py — the real-prose HARVEST opens (2026-07-10). Source: MATH
TRAIN split (disjoint from the MATH-500 examiner). Deliverables: the filtered
in-reach corpus + the mouth's odometer ZERO-POINT + zero-shot lattice
baseline. The seed annotation (dialect rewrites, solve-to-answer-key gated)
follows by hand."""
import json, re, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer

rows = json.load(open(".cache/math_train_raw.json"))
tok = Tokenizer.from_file(TOKENIZER_JSON)
def boxed(sol):
    m = re.search(r"\\boxed\{([^{}]+)\}", sol)
    return m.group(1) if m else None
harvest = []
for r in rows:
    a = boxed(r.get("solution", ""))
    if not a or not a.strip().lstrip("-").isdigit(): continue
    v = int(a.strip())
    if not (0 <= v <= 999): continue
    if len(tok.encode(r["problem"]).ids) > T_ALG: continue
    harvest.append({"problem": r["problem"], "answer": v,
                    "level": r["level"], "subject": r["subject"]})
with open(".cache/math_harvest_v0.jsonl", "w") as f:
    for h in harvest:
        f.write(json.dumps(h) + "\n")
print(f"[harvest] in-reach corpus: {len(harvest)} problems "
      f"(integer 0-999, <=256 tokens)")

# odometer zero-point
mo = np.load(".cache/recognition_mouth_gen5.npz")
n = len(harvest)
ids = np.zeros((n, T_ALG), np.int32); msk = np.zeros((n, T_ALG), np.float32)
for i, h in enumerate(harvest):
    e = tok.encode(h["problem"])
    ids[i, :len(e.ids)] = e.ids; msk[i, :len(e.ids)] = 1.0
st = recompute_states(ids)
v = (st.astype(np.float32) * msk[:, :, None]).sum(1) / \
    np.maximum(msk.sum(1)[:, None], 1)
v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
d = 1.0 - v @ mo["bank"].T
knn = np.sort(d, axis=1)[:, :8].mean(1)
np.save(".cache/harvest_states_L4.npy", st.astype(np.float16))
print(f"[odometer] zero-point: harvest mean kNN {knn.mean():.4f} "
      f"(native thr {float(mo['thr_knn']):.4f}; MATH-500 was 0.2480) | "
      f"read-foreign {float((knn > mo['thr_knn']).mean()):.1%}")
by_lvl = {}
for i, h in enumerate(harvest):
    by_lvl.setdefault(h["level"], []).append(knn[i])
for l in sorted(by_lvl):
    print(f"  {l}: mean {np.mean(by_lvl[l]):.4f} (n={len(by_lvl[l])})")
