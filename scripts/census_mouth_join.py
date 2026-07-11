"""census_mouth_join.py — THE MOUTH-DISTANCE JOIN (2026-07-11, registered
before measurement in docs/phase1_skeleton_spec.md).

The bootstrap census's knotted column (72/100) conflates two walls: style
(the mouth reads the prose as foreign — books cure it) and structure (the
problem is genuinely knotted — only the organ cures it). This join crosses
every census outcome with its prose's mouth kNN-distance to attribute them.

Deterministic re-run of the census loop (same pool slice, same permuted-view
seeds 700*xi+k, same gen-6 ckpt) capturing BOTH columns this time.

PRE-REGISTERED DECISION RULES (pinned in the ledger before this ran):
 (1) AUC (rank-sum) of mouth distance, KNOTTED vs PARSE-CARRIED (banked+near).
     AUC >= 0.60 -> style-wall attribution holds; ~0.5 -> the 72 stays
     unattributed (honest negative; weakens books-will-recover).
 (2) PATIENT LIST: knotted items at mouth distance <= carried-group MEDIAN
     are the early knotted candidates — the organ's first visible patients.
 (3) BOOK-1 PREDICTION: knotted items ABOVE the carried median are claimed
     style-recoverable; book 1 must recover them at a higher rate than the
     below-median tier or the attribution was wrong.
Split is relative (carried median) because the calibrated native line
(0.0443) is unusable — the whole pool reads foreign (mean 0.254).
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
import re

h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
pool = [x for x in h if x["level"] in ("Level 1", "Level 2", "Level 3")
        and len(x["problem"]) < 300 and "asy]" not in x["problem"]
        and all(int(n) <= 300 for n in re.findall(r"\d+", x["problem"]))]
pool = pool[:100]
print(f"[join] pool n={len(pool)} (identical slice to the census)")

tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(".cache/phase1_gen6_head.safetensors")
assert set(sd.keys()) == set(p.keys())
for k in p:
    p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()

def parse_batch(texts):
    n = len(texts)
    N = ((n + 7) // 8) * 8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32)
    snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    res = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + \
            (("sel",) if "sel" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0 + bi < n:
                res.append(decode({k: o[k][bi] for k in o}))
    return res

mo = np.load(".cache/recognition_mouth_gen5.npz")
def mouth(texts):
    n = len(texts)
    ids = np.zeros((n, T_ALG), np.int32); msk = np.zeros((n, T_ALG), np.float32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
    st = recompute_states(ids)
    v = (st.astype(np.float32)*msk[:,:,None]).sum(1)/np.maximum(msk.sum(1)[:,None],1)
    v = v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9)
    d = 1.0 - v @ mo["bank"].T
    return np.sort(d, axis=1)[:, :8].mean(1)

rows = []
for xi, x in enumerate(pool):
    texts = [x["problem"]] + [permuted_view(x["problem"], 700*xi+k) for k in range(1, 5)]
    parses = parse_batch(texts)
    votes = []
    for facs, q in parses:
        a = solve2(facs, q, {"n_vars": 24, "m": 300})
        if a is not None:
            votes.append(a)
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    if cnt >= 3 and top == x["answer"]:
        census = "banked"
    elif len(votes) >= 2:
        census = "near"
    else:
        census = "knotted"
    rows.append(dict(idx=xi, census=census, level=x["level"],
                     problem=x["problem"], answer=x["answer"]))
    if (xi + 1) % 25 == 0:
        c = Counter(r["census"] for r in rows)
        print(f"  ...{xi+1}/{len(pool)}: banked {c['banked']} near {c['near']} knotted {c['knotted']}")

md = mouth([r["problem"] for r in rows])
for r, d in zip(rows, md):
    r["mouth_d"] = float(d)

c = Counter(r["census"] for r in rows)
print(f"\n[join] census replay: banked {c['banked']} near {c['near']} knotted {c['knotted']}"
      f" (original 2/26/72 — must match)")

knot = np.array([r["mouth_d"] for r in rows if r["census"] == "knotted"])
carr = np.array([r["mouth_d"] for r in rows if r["census"] != "knotted"])

# rank-sum AUC: P(knotted distance > carried distance)
diffs = knot[:, None] - carr[None, :]
auc = float((diffs > 0).mean() + 0.5 * (diffs == 0).mean())
cmed = float(np.median(carr))

print(f"\n=== THE JOIN (pre-registered rules) ===")
print(f"  mouth distance — knotted: mean {knot.mean():.4f} median {np.median(knot):.4f} (n={len(knot)})")
print(f"  mouth distance — carried: mean {carr.mean():.4f} median {cmed:.4f} (n={len(carr)})")
print(f"  RULE 1 — AUC(knotted reads more foreign than carried): {auc:.3f}"
      f"  [bar 0.60: {'STYLE-WALL ATTRIBUTION HOLDS' if auc >= 0.60 else 'UNATTRIBUTED — honest negative' if auc < 0.55 else 'weak/inconclusive'}]")

patients = sorted([r for r in rows if r["census"] == "knotted" and r["mouth_d"] <= cmed],
                  key=lambda r: r["mouth_d"])
style_tier = [r for r in rows if r["census"] == "knotted" and r["mouth_d"] > cmed]
print(f"  RULE 2 — PATIENT LIST (knotted at <= carried median {cmed:.4f}): {len(patients)}")
for r in patients:
    print(f"    [{r['idx']:2d}] d={r['mouth_d']:.4f} {r['level']:8s} gold {r['answer']:>4} | {r['problem'][:90]!r}")
print(f"  RULE 3 — claimed STYLE-RECOVERABLE (above carried median): {len(style_tier)}"
      f"  <- book-1 must recover these at a higher rate than the patient tier")

json.dump(rows, open(".cache/census_mouth_join.json", "w"))
print(f"  [saved] .cache/census_mouth_join.json (n={len(rows)}, columns: census, mouth_d)")
