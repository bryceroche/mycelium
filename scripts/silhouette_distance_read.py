"""silhouette_distance_read.py — THE LAST CHEAP AXIS (2026-07-31,
registered before firing; bars pinned in the ledger). The interior:
per-item silhouettes in the gate's own fst space (compute_fst), present
slots pooled; family = the 263 certified book rows (the book of primes
at the waist); soft distance = kNN-8 cosine (the antagonism law's
surviving form). LOO calibration first; then the wild ledger's 124
answered items priced by distance: quartile table + AUC(correct vs
wrong). SIGNAL: AUC >= 0.70. FAIL: AUC < 0.60 — the cheap axes are
fully exhausted and grading costs training. One generation's
coordinates only (g22)."""
import sys, os, json, glob
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter
from phase1_algebra_head import T_ALG, build_params, sent_indices, TOKENIZER_JSON
from waist_abstention_probe import compute_fst
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

MAN = json.load(open(".cache/GENERATION.json"))
CKPT = MAN["parser_ckpt"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
assert set(sd.keys()) == set(p.keys())
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
print(f"[silhouette] gate from manifest: {CKPT} (one generation's coordinates)")

def silhouettes(texts):
    """Item vectors: mean over PRESENT slots of fst rows, L2-normalized."""
    n = len(texts)
    ids = np.zeros((n, T_ALG), np.int32); msk = np.zeros((n, T_ALG), np.float32)
    snt = np.zeros((n, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids).astype(np.float16)
    fst = compute_fst(p, st, msk, snt, list(range(n)))   # (n, L_FAC, H_W)
    hp, hpb = p["h_pres"].detach().numpy(), p["h_pres_b"].detach().numpy()
    out = np.zeros((n, fst.shape[-1]), np.float32)
    n_pres = np.zeros(n, np.int32)
    for i in range(n):
        pres = fst[i] @ hp[:, 0] + hpb[0] > 0
        rows = fst[i][pres] if pres.any() else fst[i]
        n_pres[i] = int(pres.sum())
        v = rows.mean(0)
        out[i] = v / max(np.linalg.norm(v), 1e-8)
    return out, n_pres

# family: the 263 certified book rows
fam_texts = []
for draft in sorted(glob.glob(".cache/book8_*prose_pairs_draft.jsonl")):
    certf = draft.replace("prose_pairs_draft.jsonl", "certification.json")
    if not os.path.exists(certf): certf = ".cache/book8_certification.json"
    rows = [json.loads(l) for l in open(draft)]
    fam_texts += [rows[e["i"]]["gen"]["dialect"] for e in json.load(open(certf))["certified"]]
print(f"[silhouette] family: {len(fam_texts)} certified rows")
F, fam_pres = silhouettes(fam_texts)

# LOO calibration: family self-distance (kNN-8, self excluded)
S = F @ F.T
np.fill_diagonal(S, -2.0)
loo = np.sort(1.0 - S, axis=1)[:, :8].mean(1)
print(f"[LOO] family self-distance: median {np.median(loo):.4f}  "
      f"p90 {np.percentile(loo, 90):.4f}  max {loo.max():.4f}")

# wild: the ledger's answered items
recs = [json.loads(l) for l in open(".cache/wild_ledger_v1.jsonl")]
ans = [r for r in recs if r["tier"] == "answered"]
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
W, wild_pres = silhouettes([h[r["harvest_idx"]]["problem"] for r in ans])
d = np.sort(1.0 - W @ F.T, axis=1)[:, :8].mean(1)
correct = np.array([bool(r["correct"]) for r in ans])
print(f"[wild] answered n={len(ans)}  distance: median {np.median(d):.4f}  "
      f"(family LOO median {np.median(loo):.4f})")

# quartile table
order = np.argsort(d)
print("\n[quartile table] distance-to-family vs precision:")
qn = len(ans) // 4
for qi in range(4):
    sl = order[qi*qn: (qi+1)*qn if qi < 3 else len(ans)]
    pr = correct[sl].mean()
    print(f"  Q{qi+1} dist [{d[sl].min():.4f},{d[sl].max():.4f}]  n={len(sl)}  precision {pr:.3f}")

# AUC (correct ranked LOWER distance = positive signal)
from itertools import product
pos = d[correct]; neg = d[~correct]
auc = float(np.mean([(1.0 if a < b else 0.5 if a == b else 0.0)
                     for a, b in product(pos, neg)]))
print(f"\n=== AUC (correct nearer the family): {auc:.3f}  "
      f"(n_correct={len(pos)}, n_wrong={len(neg)}) ===")
verdict = ("SIGNAL — the interior grades; the tier ladder revives" if auc >= 0.70
           else "FAIL — the cheap axes are FULLY exhausted; grading costs training (the decision point)" if auc < 0.60
           else "MIXED — directional only; richer interior reads may be priced, not presumed")
print(f"=== VERDICT (pinned): {verdict} ===")

# secondary, report-only: present-slot counts as a crude complexity proxy
print(f"\n[secondary] present slots: family mean {fam_pres.mean():.1f}  "
      f"wild-correct {np.array(wild_pres)[correct].mean():.1f}  "
      f"wild-wrong {np.array(wild_pres)[~correct].mean():.1f}")
json.dump({"ckpt": CKPT, "family_loo_median": float(np.median(loo)),
           "wild_median": float(np.median(d)), "auc": auc, "verdict": verdict,
           "distances": [{"harvest_idx": r["harvest_idx"], "d": float(d[i]),
                          "correct": bool(correct[i])} for i, r in enumerate(ans)]},
          open(".cache/silhouette_distance.json", "w"), indent=1)
print("[saved] .cache/silhouette_distance.json")
