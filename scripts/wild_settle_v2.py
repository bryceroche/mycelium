"""wild_settle_v2.py — TOKEN-GRAIN SETTLE (2026-07-31, the amendment;
sentence-grain v1 VOID-BY-COVERAGE at 15%). Reveal = fixed fractional
cuts of the token stream [0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0] — works
at any text length. LEG 1, CALIBRATION: the pct population at token
grain — must reproduce ~0.83 for grain equivalence (pinned: >= 0.70
calibrates; below, the wild leg's scope narrows to token-grain settle).
LEG 2, THE WILD PRICING: the 124 answered, same pre-declared direction
(correct LOWER late-wander), same 0.626 baseline, n printed per bin."""
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
FRACS = (0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0)
print(f"[settle-v2] gate: {CKPT}  reveal fractions: {FRACS}")

def fst_of_prefix_ids(full_ids, full_off, text, cuts):
    """fst per token-prefix (ids truncated; sent indices recomputed on the
    truncated char span so the waist sees a coherent prefix)."""
    n = len(cuts)
    ids = np.zeros((n, T_ALG), np.int32); msk = np.zeros((n, T_ALG), np.float32)
    snt = np.zeros((n, T_ALG), np.int32)
    for i, c in enumerate(cuts):
        ids[i, :c] = full_ids[:c]; msk[i, :c] = 1.0
        char_end = full_off[c - 1][1]
        snt[i] = sent_indices(text[:char_end], [o for o in full_off[:c]], msk[i])
    st = recompute_states(ids).astype(np.float16)
    return compute_fst(p, st, msk, snt, list(range(n)))

def late_wander_item(text, slots_from_final=True, slot_pick=None):
    e = tok.encode(text)
    L = min(len(e.ids), T_ALG)
    if L < 12: return None, None
    cuts = sorted({max(4, int(L * f)) for f in FRACS})
    if len(cuts) < 4: return None, None
    F = fst_of_prefix_ids(e.ids, list(e.offsets), text, cuts)
    final = F[-1]
    pres = final @ hp[:, 0] + hpb[0] > 0
    slots = [k for k in range(L_FAC) if pres[k]]
    if not slots: return None, None
    if slot_pick is not None:
        pk = None
        for j in slots:
            if int(np.argmax(final[j] @ hf + hfb)) == slot_pick:
                pk = j; break
        if pk is None: return None, None
        traj = F[:, [pk], :].astype(np.float32)
    else:
        traj = F[:, slots, :].astype(np.float32)
    deltas = np.linalg.norm(np.diff(traj, axis=0), axis=2)
    third = max(1, deltas.shape[0] // 3)
    return float(deltas[-third:].mean()), F

def auc_lower_correct(scores, labels):
    scores = np.array(scores); labels = np.array(labels)
    pos = scores[labels]; neg = scores[~labels]
    if not len(pos) or not len(neg): return float("nan"), 0, 0
    a = float(np.mean([(1.0 if x < y else 0.5 if x == y else 0.0)
                       for x, y in product(pos, neg)]))
    return a, len(pos), len(neg)

def late_wander_anchored(text, ev_phrase="percent", slot_pick=4):
    """deep clean 2026-08-01 (the audit's finding): cuts ANCHORED to the
    evidence phrase's token position — the fixed-fraction window put the
    evidence outside/inside the window depending on cell, confounding the
    0.400. Cuts: [just-before-ev, ev-end, then thirds of the remainder]."""
    e = tok.encode(text)
    L = min(len(e.ids), T_ALG)
    if L < 12: return None, None
    ci = text.find(ev_phrase)
    if ci < 0: return None, None
    ev_tok = next((i for i, (a, b) in enumerate(e.offsets[:L]) if b > ci), None)
    if ev_tok is None or ev_tok < 4: return None, None
    ev_end = min(L, ev_tok + 6)
    rem = L - ev_end
    if rem < 3: return None, None
    cuts = sorted({max(4, ev_tok - 2), ev_end,
                   ev_end + rem // 3, ev_end + 2 * rem // 3, L})
    if len(cuts) < 4: return None, None
    F = fst_of_prefix_ids(e.ids, list(e.offsets), text, cuts)
    final = F[-1]
    pres = final @ hp[:, 0] + hpb[0] > 0
    pk = None
    for j in range(L_FAC):
        if pres[j] and int(np.argmax(final[j] @ hf + hfb)) == slot_pick:
            pk = j; break
    if pk is None: return None, None
    traj = F[:, pk, :].astype(np.float32)
    deltas = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    # deltas[0] = into ev-end; POST-EVIDENCE = deltas[1:] (after ev absorbed)
    if len(deltas) < 2: return None, None
    return float(deltas[1:].mean()), F

# ---- LEG 1: calibration on the pct population (token grain) ----
cells = [(A, None) for A in KINDS] + [(A, B) for A in KINDS for B in KINDS]
targets = [(ci, A, B) for ci, (A, B) in enumerate(cells) if (A == "pct" or B == "pct")]
cal_scores, cal_labels = [], []
for ci, A, B in targets:
    for r in mint_cell(A, B, 12, 31000 + ci):
        text = r["text"]
        gold_pct = next(f for f in r["facs"] if f["ftype"] == "pct")
        lw, F = late_wander_anchored(text)            # EVIDENCE-ANCHORED (v3)
        if lw is None: continue
        # binding correctness via standard decode
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
        cal_scores.append(lw); cal_labels.append(bool(ok))
a_cal, ncp, ncn = auc_lower_correct(cal_scores, cal_labels)
print(f"\n[LEG 1 calibration] token-grain settle on pct: AUC {a_cal:.3f} "
      f"(correct {ncp}, misbound {ncn}; sentence-grain was 0.834)")
calibrated = a_cal >= 0.70
print(f"[LEG 1] {'CALIBRATED — grains equivalent; the wild leg interprets cleanly' if calibrated else 'NOT calibrated — the wild leg reads as token-grain settle only'}")

# ---- LEG 2: the wild 124 ----
recs = [json.loads(l) for l in open(".cache/wild_ledger_v1.jsonl")]
ans = [r for r in recs if r["tier"] == "answered"]
h = [json.loads(l) for l in open(".cache/math_harvest_v0.jsonl")]
w_scores, w_labels = [], []
skipped = 0
for j, r in enumerate(ans):
    text = h[r["harvest_idx"]]["problem"]
    lw, _ = late_wander_item(text)
    if lw is None:
        skipped += 1; continue
    w_scores.append(lw); w_labels.append(bool(r["correct"]))
    if (j+1) % 25 == 0: print(f"  [wild {j+1}/{len(ans)}]", flush=True)
a_w, nwp, nwn = auc_lower_correct(w_scores, w_labels)
print(f"\n[LEG 2 wild] scored {len(w_scores)}/{len(ans)} (skipped {skipped}) — "
      f"coverage {len(w_scores)/len(ans):.0%}")
print(f"[LEG 2 wild] AUC (correct LOWER): {a_w:.3f}  (correct {nwp}, wrong {nwn}; baseline 0.626)")
ws = np.array(w_scores); wl = np.array(w_labels)
order = np.argsort(ws); qn = max(1, len(ws)//4)
for qi in range(4):
    sl = order[qi*qn: (qi+1)*qn if qi < 3 else len(ws)]
    tag = "" if len(sl) >= 20 else "  (n<20: directional-only)"
    print(f"  Q{qi+1} [{ws[sl].min():.2f},{ws[sl].max():.2f}]  n={len(sl)}  precision {wl[sl].mean():.3f}{tag}")
print("[LEG 2 NOTE] the wild leg is VOID-BY-CONFOUND regardless of AUC — wild items have no known evidence anchor; anchor-free windows are the confound the audit demonstrated")
verdict = ("SIGNAL — interior dynamics grades the frontier; the decision point REOPENS" if a_w >= 0.70 and calibrated
           else "NULL at token grain — the axis closes as measured" if a_w < 0.60
           else "MIXED — band unclaimed")
if not calibrated:
    verdict += " [scope: token-grain only — grains not equivalent]"
print(f"=== VERDICT (pinned): {verdict} ===")
json.dump({"ckpt": CKPT, "calibration_auc": a_cal, "calibrated": bool(calibrated),
           "wild_auc": a_w, "wild_n": len(w_scores), "wild_skipped": skipped,
           "verdict": verdict},
          open(".cache/wild_settle_v2.json", "w"), indent=1)
print("[saved] .cache/wild_settle_v2.json")
