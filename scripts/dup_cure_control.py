"""dup_cure_control.py — GUT #114's CONTROL (2026-07-31, registered
before the verdict banks). The suspicious question: did the diet cure
the SPECIES or teach the PROBE'S TEMPLATE (probe and pool share a
generator lineage)? Two sides:
(a) book-8 dup-carrying certified rows (natural prose, different
    generator) — binding-grain read under g22 vs the 2%-dry arm; may
    ceiling both (reported either way);
(b) HELD-OUT-TEMPLATE mint — dup rows from families the diet pool never
    contained ("The sum of X and X is Y" phrasing; distractor counts
    5-6; letter range shifted by interleaving distractors AFTER the dup
    var) — under g22 AND the arm.
PINNED: CURE-REAL = arm's held-out misbinding <= half of g22's;
TEMPLATE-MEMORIZED = within 10 pts of g22; between = MIXED.
Waits for the fire to release the device."""
import sys, os, json, glob, subprocess, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")

while subprocess.run(["systemctl", "--user", "is-active", "dup-fire.service"],
                     capture_output=True, text=True).stdout.strip() == "active":
    time.sleep(30)

import numpy as np
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

tok = Tokenizer.from_file(TOKENIZER_JSON)
L = "abcdefghij"

def load_gate(ckpt):
    p = build_params(0)
    sd = safe_load(ckpt)
    for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
    return p

def decode_one(p, text):
    ids = np.zeros((8, T_ALG), np.int32); msk = np.zeros((8, T_ALG), np.float32); snt = np.zeros((8, T_ALG), np.int32)
    e = tok.encode(text); Ln = min(len(e.ids), T_ALG)
    ids[0, :Ln] = e.ids[:Ln]; msk[0, :Ln] = 1.0
    snt[0] = sent_indices(text, list(e.offsets), msk[0])
    out = forward(p, Tensor(recompute_states(ids).astype(np.float32), dtype=dtypes.float),
                  Tensor(msk, dtype=dtypes.float), Tensor(snt, dtype=dtypes.int))
    keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
    o = {k: out[k].realize().numpy() for k in keys}
    return decode({k: o[k][0] for k in o})[0]

# ---- (b) the held-out-template mint (never in the diet pool) ----
def mint_heldout(n_target, seed):
    rng = np.random.RandomState(seed)
    rows, tries = [], 0
    while len(rows) < n_target and tries < n_target * 15:
        tries += 1
        op = "add" if rng.rand() < 0.5 else "mul"
        x = int(rng.randint(2, 60)) if op == "add" else int(rng.randint(2, 13))
        n_dist = int(rng.randint(5, 7))                    # HELD OUT: 5-6 (pool was 2-4)
        gv = [int(rng.randint(2, 90)) for _ in range(n_dist)]
        gold = x + x if op == "add" else x * x
        if gold > 300: continue
        # HELD OUT: dup var FIRST (pool put distractors first), phrasing
        # "The sum of a and a is X." / "The product of a and a is X."
        dv = 0; res = n_dist + 1
        facs = [{"ftype": "given", "var": dv, "value": x}]
        sents = [f"{L[dv]} is {x}."]
        for i in range(n_dist):
            facs.append({"ftype": "given", "var": 1 + i, "value": gv[i]})
            sents.append(f"{L[1+i]} is {gv[i]}.")
        facs.append({"ftype": "rel", "op": op, "args": [dv, dv], "result": res})
        word = "sum" if op == "add" else "product"
        sents.append(f"The {word} of {L[dv]} and {L[dv]} is {L[res]}.")
        letters = ", ".join(L[:res+1])
        text = f"Consider the numbers {letters}. " + " ".join(sents) + f" What is {L[res]}?"
        if solve2(facs, res, {"n_vars": 24, "m": 300}) != gold: continue
        rows.append({"text": text, "dv": dv, "op": op, "res": res})
    return rows

held = mint_heldout(100, 61000)
print(f"[control] held-out-template rows: {len(held)} "
      f"(sum/product phrasing, 5-6 distractors, dup-var-first)")

def misbind_rate(p, rows):
    mis = tot = 0
    for r in rows:
        facs = decode_one(p, r["text"])
        rels = [f for f in facs if f.get("ftype") == "rel"]
        ok = any(f.get("args") == [r["dv"], r["dv"]] and f.get("op") == r["op"]
                 for f in rels)
        mis += (not ok); tot += 1
    return mis, tot

# ---- (a) book-8 dup-carrying certified rows ----
book_dup = []
for draft in sorted(glob.glob(".cache/book8_*prose_pairs_draft.jsonl")):
    certf = draft.replace("prose_pairs_draft.jsonl", "certification.json")
    if not os.path.exists(certf): certf = ".cache/book8_certification.json"
    rows = [json.loads(l) for l in open(draft)]
    for e in json.load(open(certf))["certified"]:
        r = rows[e["i"]]
        dups = [f for f in r["factors"] if f.get("ftype") == "rel"
                and len(f.get("args", [])) == 2 and f["args"][0] == f["args"][1]]
        if dups:
            book_dup.append({"text": r["gen"]["dialect"],
                             "dv": dups[0]["args"][0], "op": dups[0]["op"],
                             "res": dups[0]["result"]})
print(f"[control] book-8 dup-carrying certified rows: {len(book_dup)}")

results = {}
for name, ckpt in (("g22_baseline", ".cache/g22.safetensors"),
                   ("arm_dry_d02", ".cache/g23_dry_d02.safetensors")):
    p = load_gate(ckpt)
    hm, ht = misbind_rate(p, held)
    bm, bt = misbind_rate(p, book_dup) if book_dup else (0, 0)
    results[name] = {"heldout_misbound": hm, "heldout_n": ht,
                     "book_misbound": bm, "book_n": bt}
    print(f"[{name}] HELD-OUT misbound {hm}/{ht} = {hm/max(ht,1):.0%}   "
          f"book-8 misbound {bm}/{bt} = {bm/max(bt,1):.0%}", flush=True)

# THE 2x2 (the rider): pool cells from the banked record — g22-on-pool
# 48/91 = 53% (bench_rung2b, the registered baseline; the artifact was
# overwritten per-arm during the fire, the number is ledger-banked),
# arm-on-pool 0/120 (dupfire_dry_d02_2b.json). Held-out cells fresh.
g22h = results["g22_baseline"]["heldout_misbound"] / max(results["g22_baseline"]["heldout_n"], 1)
armh = results["arm_dry_d02"]["heldout_misbound"] / max(results["arm_dry_d02"]["heldout_n"], 1)
print("\n[THE 2x2]  (misbinding rate)")
print("                 POOL template    HELD-OUT template")
print(f"  g22 baseline   53% (48/91)      {g22h:.0%} ({results['g22_baseline']['heldout_misbound']}/{results['g22_baseline']['heldout_n']})")
print(f"  arm dry_d02     0% (0/120)      {armh:.0%} ({results['arm_dry_d02']['heldout_misbound']}/{results['arm_dry_d02']['heldout_n']})")
if g22h < 0.10:
    verdict = ("CONTROL UNINFORMATIVE — g22 already binds the held-out template "
               f"({g22h:.0%}); the 53% baseline was the POOL template's own difficulty; "
               "the suspicious question ANSWERED DIFFERENTLY: the probe measured its "
               "template, and so did the cure — both scoped")
elif armh <= g22h / 2:
    verdict = f"CURE-REAL — held-out misbinding {g22h:.0%} -> {armh:.0%} (<= half); the species moved, not the fixture"
elif abs(armh - g22h) <= 0.10:
    verdict = f"TEMPLATE-MEMORIZED — held-out {g22h:.0%} -> {armh:.0%} (within 10 pts); the diet taught the fixture"
else:
    verdict = f"MIXED — held-out {g22h:.0%} -> {armh:.0%}; band unclaimed"
print(f"=== VERDICT (pinned): {verdict} ===")
json.dump({"results": results, "verdict": verdict},
          open(".cache/dup_cure_control.json", "w"), indent=1)
print("[saved] .cache/dup_cure_control.json")
