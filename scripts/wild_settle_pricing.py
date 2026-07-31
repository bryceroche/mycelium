"""wild_settle_pricing.py — THE FIFTH CHEAP AXIS PRICED (2026-07-31,
registered before firing; the word: wild first). Settle (interior
DYNAMICS — never swept; the four nulls were external-and-static) priced
on the wild ledger's 124 answered items with gold. Per present slot:
LATE WANDER = mean fst delta over the final third of prefix steps; item
score = mean over present slots. DIRECTION PRE-DECLARED: lower
late-wander -> correct; a reversed separation is a NULL. Baseline: the
flat 0.626 four axes couldn't beat, same 124 items. Scope on the
sheet: mouth-bypassed population — a hit is a SIGNAL, not a product."""
import sys, os, json, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from itertools import product
from phase1_algebra_head import T_ALG, build_params, sent_indices, TOKENIZER_JSON, L_FAC
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
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
hp, hpb = p["h_pres"].detach().numpy(), p["h_pres_b"].detach().numpy()
print(f"[wild-settle] gate from manifest: {CKPT}")

def fst_of(texts):
    n = len(texts)
    ids = np.zeros((n, T_ALG), np.int32); msk = np.zeros((n, T_ALG), np.float32)
    snt = np.zeros((n, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); L = min(len(e.ids), T_ALG)
        ids[i, :L] = e.ids[:L]; msk[i, :L] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids).astype(np.float16)
    return compute_fst(p, st, msk, snt, list(range(n)))

recs = [json.loads(l) for l in open(".cache/wild_ledger_v1.jsonl")]
ans = [r for r in recs if r["tier"] == "answered"]
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
print(f"[wild-settle] answered items: {len(ans)}")

scores, labels = [], []
skipped = 0
for j, r in enumerate(ans):
    text = h[r["harvest_idx"]]["problem"]
    sents = re.split(r"(?<=[.?!])\s+", text.strip())
    sents = [s_ for s_ in sents if s_]
    if len(sents) < 3:
        skipped += 1; continue
    prefixes = [" ".join(sents[:si+1]) for si in range(len(sents))]
    F = fst_of(prefixes)                                  # (steps, L_FAC, H_W)
    final = F[-1]
    pres = final @ hp[:, 0] + hpb[0] > 0
    slots = [k for k in range(L_FAC) if pres[k]]
    if not slots:
        skipped += 1; continue
    deltas = np.linalg.norm(np.diff(F[:, slots, :].astype(np.float32), axis=0), axis=2)  # (steps-1, n_slots)
    third = max(1, deltas.shape[0] // 3)
    late = deltas[-third:].mean()
    scores.append(float(late)); labels.append(bool(r["correct"]))
    if (j+1) % 25 == 0: print(f"  [{j+1}/{len(ans)}]", flush=True)

scores = np.array(scores); labels = np.array(labels)
pos = scores[labels]; neg = scores[~labels]        # pre-declared: correct LOWER
auc = float(np.mean([(1.0 if a < b else 0.5 if a == b else 0.0)
                     for a, b in product(pos, neg)]))
print(f"\n[wild-settle] scored {len(scores)} (skipped {skipped}: <3 sentences or no slots)")
print(f"[wild-settle] late-wander median: correct {np.median(pos):.3f}  wrong {np.median(neg):.3f}")
print(f"=== AUC (pre-declared direction: correct LOWER): {auc:.3f}  "
      f"(n_correct={len(pos)}, n_wrong={len(neg)}; baseline flat 0.626) ===")
order = np.argsort(scores)
qn = len(scores) // 4
print("[quartile table] settle score vs precision (n printed per bin):")
for qi in range(4):
    sl = order[qi*qn: (qi+1)*qn if qi < 3 else len(scores)]
    pr = labels[sl].mean()
    tag = "" if len(sl) >= 20 else "  (n<20: directional-only)"
    print(f"  Q{qi+1} [{scores[sl].min():.3f},{scores[sl].max():.3f}]  n={len(sl)}  precision {pr:.3f}{tag}")
verdict = ("SIGNAL — interior dynamics grades the frontier; the decision point REOPENS without training" if auc >= 0.70
           else "NULL — the fifth axis joins the four (reversed or flat); the decision point stands" if auc < 0.60
           else "MIXED — band unclaimed")
if auc < 0.40:
    verdict = "REVERSED — a null by pre-declaration (and evidence settle is population-specific)"
print(f"=== VERDICT (pinned): {verdict} ===")
print("[scope] mouth-bypassed population: a hit is a SIGNAL, not a shipped product (mouth admits 1.6%)")
json.dump({"ckpt": CKPT, "n": int(len(scores)), "skipped": skipped, "auc": auc,
           "median_correct": float(np.median(pos)), "median_wrong": float(np.median(neg)),
           "verdict": verdict},
          open(".cache/wild_settle_pricing.json", "w"), indent=1)
print("[saved] .cache/wild_settle_pricing.json")
