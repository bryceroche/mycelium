"""bench_rung2b.py — RUNG 2b: SETTLE GENERALIZATION (2026-07-31,
registered; settle's valid domain — sentence grain, dialect text).
Second misbinding population: the dup-args binding-competition family
([655]/[1382]/[875] — the engage-slip species; books demand 51.7%, mix
12.3%). Mint: one dup rel ("a plus a equals c" / "a times a equals c")
under DISTRACTOR LOAD (extra givens), solver-verified. Binding correct
= the decoded rel carries args [a,a] on the gold variable. Settle: the
rel slot's post-evidence wander at sentence grain. PINNED: direction
inherited (correct LOWER); AUC >= 0.70 = settle GENERALIZES (a binding
property, not a pct artifact); < 0.60 = population-specific; between
MIXED. Support printed; imbalanced labels reported honestly."""
import sys, os, json, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from itertools import product
from phase1_algebra_head import T_ALG, build_params, sent_indices, TOKENIZER_JSON, L_FAC, forward, decode
from waist_abstention_probe import compute_fst
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

MAN = json.load(open(".cache/GENERATION.json"))
CKPT = os.environ.get("R2B_CKPT") or MAN["parser_ckpt"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
hp, hpb = p["h_pres"].detach().numpy(), p["h_pres_b"].detach().numpy()
hf, hfb = p["h_ftype"].detach().numpy(), p["h_ftype_b"].detach().numpy()
print(f"[rung2b] gate: {CKPT}")
L = "abcdefghij"

def mint_dup(n_target, seed):
    rng = np.random.RandomState(seed)
    rows, tries = [], 0
    while len(rows) < n_target and tries < 1200:
        tries += 1
        op = "add" if rng.rand() < 0.5 else "mul"
        x = int(rng.randint(2, 60)) if op == "add" else int(rng.randint(2, 13))
        n_dist = int(rng.randint(2, 5))                    # distractor load
        gv = [int(rng.randint(2, 90)) for _ in range(n_dist)]
        # vars: distractors first (the competition), then the dup var, result
        dv = n_dist; res = n_dist + 1
        sents = [f"{L[i]} is {gv[i]}." for i in range(n_dist)]
        sents.append(f"{L[dv]} is {x}.")
        word = "plus" if op == "add" else "times"
        sents.append(f"{L[dv]} {word} {L[dv]} equals {L[res]}.")
        gold = x + x if op == "add" else x * x
        if gold > 300: continue
        facs = [{"ftype": "given", "var": i, "value": gv[i]} for i in range(n_dist)]
        facs.append({"ftype": "given", "var": dv, "value": x})
        facs.append({"ftype": "rel", "op": op, "args": [dv, dv], "result": res})
        letters = ", ".join(L[:res+1])
        # shuffle the distractor sentences among the front (load variety)
        order = list(range(n_dist)); rng.shuffle(order)
        body = [sents[i] for i in order] + sents[n_dist:]
        text = f"Consider the numbers {letters}. " + " ".join(body) + f" What is {L[res]}?"
        if solve2(facs, res, {"n_vars": 24, "m": 300}) != gold: continue
        rows.append({"text": text, "dv": dv, "res": res, "op": op, "gold": gold,
                     "ev_word": word})
    return rows

rows = mint_dup(120, 41000)
print(f"[rung2b] minted {len(rows)} dup-args rows under distractor load")

def fst_of(texts):
    n = len(texts)
    ids = np.zeros((n, T_ALG), np.int32); msk = np.zeros((n, T_ALG), np.float32)
    snt = np.zeros((n, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids).astype(np.float16)
    return compute_fst(p, st, msk, snt, list(range(n)))

scores, labels = [], []
n_done = 0
for r in rows:
    text = r["text"]
    sents = re.split(r"(?<=\.)\s+", text)
    ev = next((si for si, s_ in enumerate(sents) if f" {r['ev_word']} " in s_), None)
    if ev is None or ev == 0 or len(sents) < 3: continue
    prefixes = [" ".join(sents[:si+1]) for si in range(len(sents))]
    F = fst_of(prefixes)
    final = F[-1]
    pres = final @ hp[:, 0] + hpb[0] > 0
    rel_slot = None
    for j in range(L_FAC):
        if pres[j] and int(np.argmax(final[j] @ hf + hfb)) == 0:   # rel
            rel_slot = j; break
    if rel_slot is None:
        # deep clean 2026-08-01: a parse with NO detectable rel slot is a
        # FAILURE, not a skip — dropping these silently excluded the
        # model's worst rows from its own baseline denominator (g22 scored
        # 91/120 while arms scored 120/120; the 53% understated)
        scores.append(float("nan")); labels.append(False)
        n_done += 1
        continue
    # binding correctness via standard decode
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32); snt = np.zeros((8, T_ALG), np.int32)
    e = tok.encode(text); Ln = min(len(e.ids), T_ALG)
    ids[0, :Ln] = e.ids[:Ln]; msk[0, :Ln] = 1.0
    snt[0] = sent_indices(text, list(e.offsets), msk[0])
    out = forward(p, Tensor(recompute_states(ids).astype(np.float32), dtype=dtypes.float),
                  Tensor(msk, dtype=dtypes.float), Tensor(snt, dtype=dtypes.int))
    keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
    o = {k: out[k].realize().numpy() for k in keys}
    facs, _q = decode({k: o[k][0] for k in o})
    rels = [f for f in facs if f.get("ftype") == "rel"]
    if not rels: continue
    ok = any(f.get("args") == [r["dv"], r["dv"]] and f.get("op") == r["op"]
             for f in rels)
    deltas = np.linalg.norm(np.diff(F[:, rel_slot, :].astype(np.float32), axis=0), axis=1)
    third = max(1, len(deltas) // 3)
    scores.append(float(deltas[-third:].mean())); labels.append(bool(ok))
    n_done += 1
    if n_done % 25 == 0: print(f"  [{n_done}]", flush=True)

scores = np.array(scores); labels = np.array(labels)
n_noslot = int(np.isnan(scores).sum())
if n_noslot:
    print(f"[rung2b] no-rel-slot rows counted as misbound: {n_noslot}")
valid = ~np.isnan(scores)
pos = scores[valid & labels]; neg = scores[valid & ~labels]
print(f"\n[rung2b] scored {len(scores)} (correct-bound {len(pos)}, misbound {len(neg)})")
if len(pos) and len(neg):
    auc = float(np.mean([(1.0 if a < b else 0.5 if a == b else 0.0)
                         for a, b in product(pos, neg)]))
    print(f"[rung2b] settle medians: correct {np.median(pos):.3f}  misbound {np.median(neg):.3f}")
    sup = min(len(pos), len(neg))
    tag = "" if sup >= 20 else f"  [SUPPORT {sup} < 20: directional-only]"
    print(f"=== AUC (correct LOWER): {auc:.3f}{tag} ===")
    verdict = ("SETTLE GENERALIZES — a binding property, not a pct artifact" if auc >= 0.70
               else "POPULATION-SPECIFIC — settle is a pct-frame phenomenon" if auc < 0.60
               else "MIXED — band unclaimed")
    if sup < 20: verdict += " [directional-only at this support]"
else:
    auc = float("nan"); verdict = "UNSCORABLE — a label class is empty (population too easy or too hard)"
    print("[rung2b] a label class is empty — the population did not yield both classes")
print(f"=== VERDICT (pinned): {verdict} ===")
json.dump({"ckpt": CKPT, "n": int(len(scores)), "n_correct": int(len(pos)),
           "n_misbound": int(len(neg)),
           "auc": None if np.isnan(auc) else auc, "verdict": verdict},
          open(".cache/bench_rung2b.json", "w"), indent=1)
print("[saved] .cache/bench_rung2b.json")
