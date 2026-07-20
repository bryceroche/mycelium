"""fingerpost_v01.py — GUT #33, READ (a) v0.1 (2026-07-20, countersigned):
THE FINGERPOST AT THE ARGUMENT FACTORY. One variable changed from v0:
POPULATION = the sampled lattice's top-2 contests (plurality vs
runner-up among solver-consistent candidates, per abstain item, standing
seeds). Same deterministic writer, same trunk adjudication, same Wood
fence (witness never judge), same bars (>=60% points; <=55% dies).
Locus rider inherits WITH ITS REGIME TAG: repair-lane-sampled — the
confusion matrix under temperature is a different object than under
determinism, and the tag keeps them from blurring.
"""
import json, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
from collections import Counter
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "6")
os.environ.setdefault("ALG_DUP", "1")
from phase1_algebra_head import T_ALG, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer

# importing the lattice module re-runs its pipeline by design (flat
# script, seed-deterministic) — we need its view_outs + sampler
from vote_sample_lattice import (fixture, rows, gold, sampled_decode,
                                 view_outs, T_SAMP, K_PER_VIEW, N_VIEWS)

z = np.load(".cache/phase1_alg_states_bigtest.npz")
states0, tokmask0 = z["states"], z["tokmask"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
LET = "abcdefghijklmnopqrstuvwx"


def render(facs, q):
    used = sorted({v for f in facs for v in
                   (list(f.get("args", [])) + [f[k] for k in ("result", "var")
                    if k in f])})
    L = {v: LET[j] for j, v in enumerate(used)}
    s = ["Consider the numbers " + ", ".join(L[v] for v in used) + "."]
    for f in facs:
        if f["ftype"] == "given":
            s.append(f"{L[f['var']]} is {f['value']}.")
        elif f["ftype"] == "rel":
            w = "plus" if f["op"] == "add" else "times"
            s.append(f"{L[f['args'][0]]} {w} {L[f['args'][1]]} equals {L[f['result']]}.")
        elif f["ftype"] == "mod":
            s.append(f"When {L[f['var']]} is divided by {f['k']}, the remainder is {L[f['result']]}.")
        elif f["ftype"] == "fdiv":
            s.append(f"When {L[f['var']]} is divided by {f['k']}, the quotient is {L[f['result']]}.")
        elif f["ftype"] == "pct":
            s.append(f"{L[f['args'][0]]} percent of {L[f['args'][1]]} is taken.")
        else:
            s.append(f"the larger of {L[f['args'][0]]} and {L[f['args'][1]]} is {L[f['result']]}.")
    s.append(f"What is {L.get(q, 'a')}?")
    return " ".join(s)


def fkey(f):
    if f["ftype"] == "rel":
        return ("rel", f["op"], tuple(sorted(f["args"])), f["result"])
    if f["ftype"] == "given":
        return ("given", f["var"], f["value"])
    return (f["ftype"], f.get("var"), f.get("k", f.get("p")), f.get("result"))


def pooled_batch(texts):
    ids = np.zeros((len(texts), T_ALG), np.int32)
    msk = np.zeros((len(texts), T_ALG), np.float32)
    for i, t in enumerate(texts):
        e = tok.encode(t)
        Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
    sts = recompute_states(ids)
    V = (sts.astype(np.float32) * msk[:, :, None]).sum(1) / \
        np.maximum(msk.sum(1)[:, None], 1)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


right = wrong = skipped = 0
locus = Counter()
details = []
batch_texts, batch_meta = [], []
for i in fixture:
    cands = {}
    for v in range(N_VIEWS):
        for k in range(K_PER_VIEW):
            rng = np.random.RandomState(70000 + i * 100 + v * 10 + k)
            facs, q = sampled_decode(view_outs[v][i], T_SAMP, rng)
            a = solve2(facs, q, {"n_vars": 24, "m": rows[i].get("m", 60)})
            if a is not None:
                cands.setdefault(a, {"count": 0, "parse": (facs, q)})
                cands[a]["count"] += 1
    top = sorted(cands.items(), key=lambda kv: -kv[1]["count"])[:2]
    if len(top) < 2 or gold[i] not in (top[0][0], top[1][0]):
        skipped += 1
        continue
    (a1, c1), (a2, c2) = top
    d1 = render(*c1["parse"])
    d2 = render(*c2["parse"])
    batch_texts += [d1, d2]
    batch_meta.append((i, a1, a2, c1["parse"], c2["parse"]))

print(f"[fingerpost-v01] gradable contests: {len(batch_meta)} "
      f"(skipped {skipped}: <2 candidates or gold absent)")

# adjudicate in batches
for s0 in range(0, len(batch_meta), 32):
    chunk = batch_meta[s0:s0 + 32]
    V = pooled_batch(batch_texts[2 * s0: 2 * s0 + 2 * len(chunk)])
    for ci, (i, a1, a2, p1, p2) in enumerate(chunk):
        v0 = (states0[i].astype(np.float32) * tokmask0[i][:, None]).sum(0) / \
            max(tokmask0[i].sum(), 1)
        v0 /= np.linalg.norm(v0)
        s1 = float(V[2 * ci] @ v0)
        s2 = float(V[2 * ci + 1] @ v0)
        point = a1 if s1 > s2 else a2
        ok = point == gold[i]
        right += ok
        wrong += (not ok)
        ka, kb = set(map(fkey, p1[0])), set(map(fkey, p2[0]))
        for f in (ka ^ kb):
            locus[f[0]] += 1
        details.append({"i": int(i), "gold": gold[i], "a1": a1, "a2": a2,
                        "s1": s1, "s2": s2, "point": point, "ok": bool(ok)})

n = right + wrong
acc = right / max(n, 1)
verdict = ("THE FINGERPOST POINTS" if acc >= 0.60 else
           "DIES AT THE ARGUMENT FACTORY" if acc <= 0.55 else "BETWEEN BARS")
print(f"\n=== THE FINGERPOST v0.1 (repair-lane contests, n={n}) ===")
print(f"  preference-for-truth: {right}/{n} = {acc:.3f}  => {verdict}")
print(f"  contested-binding census [regime: repair-lane-sampled]: "
      f"{dict(locus.most_common())}")
json.dump({"n": n, "right": right, "acc": acc, "verdict": verdict,
           "skipped": skipped, "locus_regime": "repair-lane-sampled",
           "locus": dict(locus), "details": details},
          open(".cache/fingerpost_v01.json", "w"))
print("[fingerpost-v01] banked -> .cache/fingerpost_v01.json")
