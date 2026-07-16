"""ood_decomposition_audit.py — GUT #22: THE OOD DECOMPOSITION AUDIT
(2026-07-16, registered as amended; all zero-GPU on held artifacts).

One word, four debts, four owners: style (mouth/books), structure
(mint/annotation), prime (registry), domain (solver caps). The scalar
"foreign" collapses a joint structure to a projection; this audit prints
the joint tables, regime-tagged, and tests the pinned predictions.

PINNED BEFORE ANY PRINT (the registration's bars):
  READ A prediction: the 56 panel-dissent items concentrate on RARER
    STRUCTURE than the 912 certified (style is constant: bigtest is
    native by construction). Metric: rank AUC of structure-rarity,
    dissent vs certified. AUC >= 0.60 in the predicted direction =
    prediction lands (decomposition is mechanism); 0.45-0.60 = scatter
    (taxonomy, not mechanism — banked honestly); < 0.45 = inverted.
  READ B bars (census calibration, n=100, labels vintage 2026-07-11,
    distances CURRENT vintage gen-13 mouth + its length correction):
    (b1) monotonicity: P(in-reach) non-increasing across ascending
    corrected-distance quartiles, one inversion <= 5pts tolerated at
    n=100; (b2) informativeness: AUC(corrected distance -> in-reach)
    >= 0.60 = the ruler reads in the gray zone; < 0.55 = flat where
    the campaign walks next (the wall's sharpness was the forest being
    far away).
Machinery vintages are printed with every table (the third-appearance
clause: outcome labels inherit their gate's generation).
"""
import json, re, sys, os
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, "scripts")
from schema_miner import TRAIN_SOURCES, CAP, mine_graph

# ---------------------------------------------------------------- READ A
print("=" * 72)
print("READ A — THE JOINT TABLE (population 1: bigtest, regime gen-14")
print("lattice; style constant/native by construction; structure axis =")
print("min train-frequency over the item's mined subgraph classes)")
print("=" * 72)

cls_cache = ".cache/train_class_counts.json"
if os.path.exists(cls_cache):
    train_freq = json.load(open(cls_cache))
else:
    train_freq = Counter()
    n_rows = 0
    for src, path in TRAIN_SOURCES.items():
        try:
            rows = [json.loads(l) for l in open(path)][:CAP]
        except FileNotFoundError:
            continue
        n_rows += len(rows)
        for r in rows:
            for dg, k, _ in mine_graph(r["factors"]):
                train_freq[dg] += 1
    print(f"[A] mined {n_rows} train rows -> {len(train_freq)} classes (cached)")
    json.dump(dict(train_freq), open(cls_cache, "w"))

rows = [json.loads(l) for l in open(".cache/algebra_nl_bigtest.jsonl")]
gold = [r["solution"][r["query_var"]] for r in rows]
gate = json.load(open(".cache/lattice_gate.json"))["bigtest"]
armb = json.load(open(".cache/lattice_armB.json"))["bigtest"]
c2x = json.load(open(".cache/lattice_cap2x.json"))["bigtest"]


def maj(v):
    vs = [x for x in v if x is not None]
    if not vs:
        return None, 0
    return Counter(vs).most_common(1)[0]


def rarity(r):
    """min train frequency over the item's subgraph classes, size<=4
    (large-k classes are near-unique by construction and would dominate)."""
    fs = [train_freq.get(dg, 0) for dg, k, _ in mine_graph(r["factors"]) if k <= 4]
    return min(fs) if fs else 0


chan, rar = [], []
for i, r in enumerate(rows):
    gt, gc = maj(gate[i]); at, _ = maj(armb[i]); ct, _ = maj(c2x[i])
    if gc == 5 and at == gt and ct == gt:
        ch = "certify"
    elif gc == 5:
        ch = "panel-dissent"
    elif gc >= 3:
        ch = "answer"
    else:
        ch = "vote-abstain"
    chan.append(ch)
    rar.append(rarity(r))
rar = np.array(rar, float)


def rank_auc(a, b):
    """P(random a < random b): a=dissent rarities, b=certified (predicted:
    dissent rarer -> smaller freq -> AUC > 0.5)."""
    a, b = np.asarray(a), np.asarray(b)
    order = np.argsort(np.concatenate([a, b]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    # midranks for ties
    allv = np.concatenate([a, b])
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    ra = ranks[:len(a)].sum()
    u = ra - len(a) * (len(a) + 1) / 2
    return 1.0 - u / (len(a) * len(b))


for ch in ("certify", "panel-dissent", "answer", "vote-abstain"):
    m = np.array([c == ch for c in chan])
    print(f"  {ch:14s} n={m.sum():4d}  structure-rarity median "
          f"{np.median(rar[m]):8.0f}  (25th {np.percentile(rar[m],25):7.0f})")
dis = rar[np.array([c == "panel-dissent" for c in chan])]
cer = rar[np.array([c == "certify" for c in chan])]
auc = rank_auc(dis, cer)
verdict = ("LANDS (mechanism)" if auc >= 0.60 else
           "SCATTER (taxonomy, not mechanism)" if auc >= 0.45 else "INVERTED")
print(f"\n  PINNED PREDICTION: dissent rarer than certified -> AUC {auc:.3f}"
      f"  => {verdict}")
abst = rar[np.array([c == "vote-abstain" for c in chan])]
print(f"  context: abstain-vs-certified rarity AUC {rank_auc(abst, cer):.3f}")

# ------------------------------------------------- READ A pop 2 + READ B
print()
print("=" * 72)
print("READ A pop-2 + READ B — CENSUS (labels VINTAGE 2026-07-11 join;")
print("distances CURRENT: gen-13 mouth bank + gen-13 length coef;")
print("gen-14 disjoint aggregate for comparison: banked 1/near 24/knotted 61 on n=86)")
print("=" * 72)

cj = json.load(open(".cache/census_mouth_join.json"))
harv = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
by_text = {h["problem"]: i for i, h in enumerate(harv)}
idxs = [by_text[c["problem"]] for c in cj]        # hard-error on mismatch

mouth = np.load(".cache/recognition_mouth_gen13.npz")
bank, coef = mouth["bank"].astype(np.float32), mouth["coef"]
thr = float(mouth["thr_knn"])
st = np.load(".cache/harvest_states_L4.npy", mmap_mode="r")

from tokenizers import Tokenizer
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
tok = Tokenizer.from_file(TOKENIZER_JSON)

V, L = [], []
for i in idxs:
    ids = tok.encode(harv[i]["problem"]).ids
    l = min(len(ids), T_ALG)
    v = st[i, :l].astype(np.float32).mean(0)
    V.append(v / np.linalg.norm(v)); L.append(l)
V, L = np.stack(V), np.array(L, float)
d_raw = np.sort(1.0 - V @ bank.T, axis=1)[:, :8].mean(1)
d = d_raw - (coef[0] + coef[1] / L)               # length-corrected, per law

label = [c["census"] for c in cj]
inreach = np.array([lb in ("banked", "near") for lb in label])
print(f"  census n={len(cj)} | labels: {Counter(label)} | corrected thr {thr:.4f}")

q = np.quantile(d, [0.25, 0.5, 0.75])
band = np.digitize(d, q)
names = ["nearest", "mid-near", "mid-far", "farthest"]
print(f"\n  CALIBRATION (corrected distance quartiles vs in-reach):")
prev, mono_breaks, worst = None, 0, 0.0
for b in range(4):
    m = band == b
    p = inreach[m].mean()
    print(f"    {names[b]:9s} d in [{d[m].min():+.4f},{d[m].max():+.4f}]  "
          f"n={m.sum():3d}  P(in-reach) {p:.2f}")
    if prev is not None and p > prev + 1e-9:
        mono_breaks += 1; worst = max(worst, p - prev)
    prev = p
auc_b = rank_auc(d[inreach], d[~inreach])          # in-reach nearer -> AUC>0.5
b1 = "HOLDS" if (mono_breaks == 0 or worst <= 0.05) else "BREAKS"
b2 = ("READS" if auc_b >= 0.60 else
      "MARGINAL" if auc_b >= 0.55 else "FLAT")
print(f"\n  BAR b1 (monotonicity): {mono_breaks} inversions (worst +{worst:.2f}) => {b1}")
print(f"  BAR b2 (informativeness): AUC(distance -> in-reach) {auc_b:.3f} => {b2}")
fr = (d[inreach] > thr).mean()
print(f"  false-refusal on in-reach census at current thr: {fr:.1%}")

# joint table: mouth band x outcome x domain flag
big = np.array([any(int(x) > 300 for x in re.findall(r"\d+", c["problem"]))
                or (isinstance(c["answer"], int) and abs(c["answer"]) > 300)
                for c in cj])
print(f"\n  JOINT (band x outcome x domain>300) — cells with n>0:")
for b in range(4):
    for lb in ("banked", "near", "knotted"):
        for dv in (False, True):
            m = (band == b) & np.array([x == lb for x in label]) & (big == dv)
            if m.sum():
                print(f"    {names[b]:9s} {lb:8s} dom>{300 if dv else ''}"
                      f"{'>300' if dv else '<=300':>6s}  n={m.sum()}")

# ---------------------------------------------------------------- pop 3
print()
print("=" * 72)
print("READ A pop-3 — MATH-500 ANCHOR (decision x stratum; vintage:")
print("anchor outcomes as banked; no per-item mouth distance held — stated)")
print("=" * 72)
anc = json.load(open(".cache/math500_anchor_outcomes.json"))
tab = Counter((a["decision"], a["stratum"]) for a in anc)
for (dec, strat), c in sorted(tab.items()):
    print(f"    {dec:10s} {strat:14s} n={c}")

json.dump({"predA_auc": float(auc), "predA_verdict": verdict,
           "b1": b1, "b2": b2, "auc_b": float(auc_b),
           "false_refusal_inreach": float(fr),
           "census_labels_vintage": "2026-07-11",
           "distance_vintage": "gen-13 mouth + coef, length-corrected"},
          open(".cache/ood_decomposition_audit.json", "w"))
print("\n[ood-audit] banked -> .cache/ood_decomposition_audit.json")
