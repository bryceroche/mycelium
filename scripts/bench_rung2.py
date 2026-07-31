"""bench_rung2.py — BENCH RUNG 2: THE DRIVER MATH (2026-07-31,
registered before firing; still zero training). Rung 1 says the
readiness signal must be built; this rung asks what the driver computes
FROM. Candidates pre-declared in the ledger: D1 ANTICIPATION (primary —
cos(slot at pre-evidence step, final); correct-bound predicted HIGHER),
D2 LURCH (rung 1's baseline, known 0.587 flipped), D3 SETTLE
(post-evidence mean delta; misbound predicted HIGHER), and the
pre-declared no-fit combination z-mean(D1, -D3). Bars inherited:
any >= 0.70 -> the driver has its formula; all < 0.60 -> the signal
needs information NOT in the waist trajectories; between -> MIXED,
band unclaimed."""
import sys, os, json, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from itertools import product
from phase1_algebra_head import T_ALG, build_params, sent_indices, TOKENIZER_JSON, L_FAC, forward, decode
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
print(f"[rung2] gate from manifest: {CKPT}")

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

cells = [(A, None) for A in KINDS] + [(A, B) for A in KINDS for B in KINDS]
targets = [(ci, A, B) for ci, (A, B) in enumerate(cells) if (A == "pct" or B == "pct")]
specimens = []
for ci, A, B in targets:
    for r in mint_cell(A, B, 12, 31000 + ci):
        specimens.append(r)
print(f"[rung2] pct specimens: {len(specimens)}")

feats = {"D1_anticipation": [], "D2_lurch": [], "D3_settle": []}
labels = []
n_done = 0
for r in specimens:
    text = r["text"]
    gold_pct = next(f for f in r["facs"] if f["ftype"] == "pct")
    sents = re.split(r"(?<=\.)\s+", text)
    ev = next((si for si, s_ in enumerate(sents) if "percent" in s_), None)
    if ev is None or ev == 0 or len(sents) < 3: continue
    prefixes = [" ".join(sents[:si+1]) for si in range(len(sents))]
    F = fst_of(prefixes)
    final = F[-1]
    pres = final @ hp[:, 0] + hpb[0] > 0
    pct_slot = None
    for j in range(L_FAC):
        if pres[j] and int(np.argmax(final[j] @ hf + hfb)) == 4:
            pct_slot = j; break
    if pct_slot is None: continue
    # binding correctness via the standard decode path
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
    ok = any(sorted(f.get("args", [])) == sorted(gold_pct["args"])
             and int(f.get("p", -1)) == int(gold_pct["p"]) for f in ppct)
    traj = F[:, pct_slot, :].astype(np.float32)
    deltas = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    # D1: was the slot already positioned BEFORE its evidence?
    pre = traj[ev - 1]; fin = traj[-1]
    d1 = float(pre @ fin / (np.linalg.norm(pre) * np.linalg.norm(fin) + 1e-8))
    # D2: rung 1's lurch ratio
    others = np.delete(deltas, ev - 1)
    d2 = float(deltas[ev - 1] / max(others.mean(), 1e-6))
    # D3: post-evidence wander
    post = deltas[ev:] if ev < len(deltas) else np.array([0.0])
    d3 = float(post.mean()) if len(post) else 0.0
    feats["D1_anticipation"].append(d1)
    feats["D2_lurch"].append(d2)
    feats["D3_settle"].append(d3)
    labels.append(bool(ok))
    n_done += 1
    if n_done % 25 == 0: print(f"  [{n_done}]", flush=True)

labels = np.array(labels)
def auc_of(x, higher_is_correct=True):
    x = np.array(x)
    pos = x[labels] if higher_is_correct else -x[labels]
    neg = x[~labels] if higher_is_correct else -x[~labels]
    return float(np.mean([(1.0 if a > b else 0.5 if a == b else 0.0)
                          for a, b in product(pos, neg)]))
print(f"\n[rung2] scored {len(labels)} (correct {int(labels.sum())}, misbound {int((~labels).sum())})")
results = {}
for name, hic in (("D1_anticipation", True), ("D2_lurch", False), ("D3_settle", False)):
    a = auc_of(feats[name], hic)
    results[name] = a
    med_c = float(np.median(np.array(feats[name])[labels]))
    med_m = float(np.median(np.array(feats[name])[~labels]))
    print(f"  {name:16s} AUC {a:.3f}  (median correct {med_c:.3f} vs misbound {med_m:.3f})")
# pre-declared combination: z-mean(D1, -D3), no fitting
z = lambda v: (np.array(v) - np.mean(v)) / max(np.std(v), 1e-8)
combo = z(feats["D1_anticipation"]) - z(feats["D3_settle"])
a_combo = auc_of(combo, True)
results["combo_z(D1,-D3)"] = a_combo
print(f"  {'combo z(D1,-D3)':16s} AUC {a_combo:.3f}")
best = max(results.values())
verdict = ("THE DRIVER HAS ITS FORMULA — readiness is computable from waist trajectories; rung 3's proposal writes itself" if best >= 0.70
           else "SIGNAL NOT IN THE WAIST TRAJECTORIES — the driver must INJECT provenance; the training case gains specificity" if best < 0.60
           else "MIXED — band unclaimed (the reading note honored)")
print(f"=== best AUC {best:.3f} -> VERDICT (pinned): {verdict} ===")
json.dump({"ckpt": CKPT, "n": int(len(labels)), "aucs": results, "verdict": verdict},
          open(".cache/bench_rung2.json", "w"), indent=1)
print("[saved] .cache/bench_rung2.json")
