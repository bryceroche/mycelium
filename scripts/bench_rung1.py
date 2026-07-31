"""bench_rung1.py — BENCH RUNG 1: does readiness already exist in the
trained gate? (2026-07-31, registered before firing.) EVIDENCE-LOCK on
sentence-prefix trajectories: for each pct row (the matrix's cells,
same seeds — misbinding-rich) + clean-cell contrast, reveal the text
sentence by sentence, read the waist (fst) at each step, locate the pct
factor's slot in the final parse, and measure the slot's delta at the
percent-sentence step against its mean delta elsewhere.
PINNED: AUC(evidence-lock, correct-bound vs misbound) >= 0.70 ->
readiness is a READOUT problem; < 0.60 -> the driver must add it;
between -> MIXED. Zero training."""
import sys, os, json, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter
from itertools import product
from phase1_algebra_head import T_ALG, build_params, sent_indices, TOKENIZER_JSON, L_FAC
from waist_abstention_probe import compute_fst
from beacon_closing_arm import recompute_states
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load
from composition_matrix import mint_cell, KINDS

MAN = json.load(open(".cache/GENERATION.json"))
CKPT = MAN["parser_ckpt"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
hp, hpb = p["h_pres"].detach().numpy(), p["h_pres_b"].detach().numpy()
hf, hfb = p["h_ftype"].detach().numpy(), p["h_ftype_b"].detach().numpy()
print(f"[rung1] gate from manifest: {CKPT}")

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

# cells: all pct rows + two clean contrast cells
cells = [(A, None) for A in KINDS] + [(A, B) for A in KINDS for B in KINDS]
targets = [(ci, A, B) for ci, (A, B) in enumerate(cells)
           if (A == "pct" or B == "pct")]
specimens = []
for ci, A, B in targets:
    for r in mint_cell(A, B, 12, 31000 + ci):
        specimens.append((f"{A}->{B or 'base'}", r))
print(f"[rung1] pct specimens: {len(specimens)}")

lock_scores, labels = [], []
n_done = 0
for cell, r in specimens:
    text = r["text"]
    gold_pct = next(f for f in r["facs"] if f["ftype"] == "pct")
    sents = re.split(r"(?<=\.)\s+", text)
    # evidence step: the sentence containing "percent"
    ev = next((si for si, s_ in enumerate(sents) if "percent" in s_), None)
    if ev is None or len(sents) < 3: continue
    prefixes = [" ".join(sents[:si+1]) for si in range(len(sents))]
    F = fst_of(prefixes)                                   # (steps, L_FAC, H_W)
    # locate the pct slot in the FINAL parse + its binding correctness
    final = F[-1]
    pres = final @ hp[:, 0] + hpb[0] > 0
    slots = [j for j in range(L_FAC) if pres[j]]
    pct_slot, bound_ok = None, None
    for j in slots:
        ft = int(np.argmax(final[j] @ hf + hfb))
        if ft == 4:                                        # pct ftype index
            pct_slot = j
            break
    if pct_slot is None: continue                          # pct absent — skip (5% class)
    # binding correctness: re-decode this row's full parse via the standard path
    from phase1_algebra_head import forward, decode
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32); snt = np.zeros((8, T_ALG), np.int32)
    e = tok.encode(text); L = min(len(e.ids), T_ALG)
    ids[0, :L] = e.ids[:L]; msk[0, :L] = 1.0
    snt[0] = sent_indices(text, list(e.offsets), msk[0])
    out = forward(p, Tensor(recompute_states(ids).astype(np.float32), dtype=dtypes.float),
                  Tensor(msk, dtype=dtypes.float), Tensor(snt, dtype=dtypes.int))
    keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
    o = {k: out[k].realize().numpy() for k in keys}
    facs, _q = decode({k: o[k][0] for k in o})
    ppct = [f for f in facs if f.get("ftype") == "pct"]
    if not ppct: continue
    bound_ok = any(sorted(f.get("args", [])) == sorted(gold_pct["args"])
                   and int(f.get("p", -1)) == int(gold_pct["p"]) for f in ppct)
    # evidence-lock: slot delta at the evidence step vs mean elsewhere
    deltas = np.linalg.norm(np.diff(F[:, pct_slot, :], axis=0), axis=1)  # (steps-1,)
    if len(deltas) < 2 or ev == 0: continue
    ev_d = deltas[ev - 1]                                  # delta INTO the evidence step
    others = np.delete(deltas, ev - 1)
    lock = float(ev_d / max(others.mean(), 1e-6))
    lock_scores.append(lock); labels.append(bool(bound_ok))
    n_done += 1
    if n_done % 25 == 0: print(f"  [{n_done}]", flush=True)

lock_scores = np.array(lock_scores); labels = np.array(labels)
pos = lock_scores[labels]; neg = lock_scores[~labels]
print(f"\n[rung1] specimens scored {len(lock_scores)}  "
      f"(correct-bound {len(pos)}, misbound {len(neg)})")
print(f"[rung1] evidence-lock: correct-bound median {np.median(pos):.2f}  "
      f"misbound median {np.median(neg):.2f}")
auc = float(np.mean([(1.0 if a > b else 0.5 if a == b else 0.0)
                     for a, b in product(pos, neg)])) if len(pos) and len(neg) else float("nan")
print(f"=== AUC (correct-bound has HIGHER evidence-lock): {auc:.3f} ===")
verdict = ("READOUT — the readiness signal EXISTS in the trained gate; R1 is a readout problem before it is a training problem" if auc >= 0.70
           else "ABSENT — the driver must add the signal; the eager math now specifies what" if auc < 0.60
           else "MIXED — per-population map only")
print(f"=== VERDICT (pinned): {verdict} ===")
json.dump({"ckpt": CKPT, "n": int(len(lock_scores)),
           "n_correct": int(len(pos)), "n_misbound": int(len(neg)),
           "median_lock_correct": float(np.median(pos)) if len(pos) else None,
           "median_lock_misbound": float(np.median(neg)) if len(neg) else None,
           "auc": auc, "verdict": verdict},
          open(".cache/bench_rung1.json", "w"), indent=1)
print("[saved] .cache/bench_rung1.json")
