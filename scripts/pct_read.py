"""pct_read.py — THE PCT READ (2026-07-31, registered before firing).
The question: why does isolated pct fail (0.08) while embedded pct
half-works (0.32) — inverting the usual direction — and what separates
the 0.75 cells from the 0.00 ones?
Hypotheses pinned in the ledger: H1 RENDERING (mint surface form
mismatches the training register; signature = pct factor absent),
H2 BINDING (pct present, args/p wrong), H3 VALUE-SKEW (per-cell value
distributions, not consumer identity). Verdict rule: dominant failure
class decides the treatment (rendering law / binding docket / re-mint
value-matched)."""
import sys, os, json, re
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
os.environ.setdefault("ALG2", "1"); os.environ.setdefault("ALG_FTYPES", "8")
os.environ.setdefault("ALG_HW", "512"); os.environ.setdefault("ALG_DUP", "1")
import numpy as np
from collections import Counter, defaultdict
from phase1_algebra_head import T_ALG, build_params, forward, decode, sent_indices, TOKENIZER_JSON
from beacon_closing_arm import recompute_states
from tta_views import permuted_view
from tta_alg2_dials import solve2
from tokenizers import Tokenizer
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

# ---------- (a) CPU census: how does the TRAINING register phrase pct? ----------
mix = [json.loads(l) for l in open(".cache/gen22_mix.jsonl")]
pct_rows = [r for r in mix if any(f.get("ftype") == "pct" for f in r["factors"])]
print(f"[census] mix rows with a pct factor: {len(pct_rows)}/{len(mix)} = {len(pct_rows)/len(mix):.1%}")
forms = Counter()
for r in pct_rows:
    t = r["text"]
    m = re.search(r"[^.]*percent[^.]*\.", t)
    if not m: forms["(no 'percent' word)"] += 1; continue
    s = m.group(0).strip()
    # template-ize: letters -> V, numbers -> N
    tpl = re.sub(r"\b[a-j]\b", "V", re.sub(r"\d+", "N", s))
    forms[tpl] += 1
print("[census] pct sentence templates in the mix (top 8):")
for tpl, n in forms.most_common(8):
    print(f"  {n:5d}  {tpl[:110]}")
MINT_FORM = "V is N percent of V."
print(f"[census] the matrix mint's form: {MINT_FORM!r} — present in mix: "
      f"{forms.get(MINT_FORM, 0)} rows")

# ---------- (b) GPU: re-parse the pct cells, classify failures ----------
MAN = json.load(open(".cache/GENERATION.json"))
CKPT = MAN["parser_ckpt"]
tok = Tokenizer.from_file(TOKENIZER_JSON)
p = build_params(0)
sd = safe_load(CKPT)
for k in p: p[k].assign(sd[k].to(p[k].device).cast(p[k].dtype)).realize()
from composition_matrix import mk_producer, mk_consumer, mint_cell, KINDS  # same seeds = same rows

def parse_batch(texts):
    n = len(texts); N = ((n+7)//8)*8
    ids = np.zeros((N, T_ALG), np.int32); msk = np.zeros((N, T_ALG), np.float32); snt = np.zeros((N, T_ALG), np.int32)
    for i, t in enumerate(texts):
        e = tok.encode(t); Ln = min(len(e.ids), T_ALG)
        ids[i, :Ln] = e.ids[:Ln]; msk[i, :Ln] = 1.0
        snt[i] = sent_indices(t, list(e.offsets), msk[i])
    st = recompute_states(ids)
    out_r = []
    for s0 in range(0, N, 8):
        out = forward(p, Tensor(st[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(msk[s0:s0+8].astype(np.float32), dtype=dtypes.float),
                      Tensor(snt[s0:s0+8].astype(np.int32), dtype=dtypes.int))
        keys = ("pres","ftype","op","islit","dig","args","res","query") + (("sel",) if "sel" in out else ()) + (("dup",) if "dup" in out else ())
        o = {k: out[k].realize().numpy() for k in keys}
        for bi in range(8):
            if s0+bi < n: out_r.append(decode({k: o[k][bi] for k in o}))
    return out_r

# reconstruct exactly the matrix's pct-involving cells (same seed arithmetic)
cells = [(A, None) for A in KINDS] + [(A, B) for A in KINDS for B in KINDS]
targets = [(ci, A, B) for ci, (A, B) in enumerate(cells)
           if A == "pct" or B == "pct"]
CLS = ("pct_absent", "p_wrong", "args_wrong", "other_factor", "no_view_parse")
stats = defaultdict(Counter)
val_note = defaultdict(list)
for ci, A, B in targets:
    name = f"base:{A}" if B is None else f"{A}->{B}"
    rows = mint_cell(A, B, 12, 31000 + ci)
    for j, r in enumerate(rows):
        gold_pct = next(f for f in r["facs"] if f["ftype"] == "pct")
        vt = [r["text"]] + [permuted_view(r["text"], 99000 + 100*ci + 10*j + k) for k in range(1, 5)]
        parses = parse_batch(vt)
        answers = [solve2(f, q, {"n_vars": 24, "m": 300}) for f, q in parses]
        nn = [a for a in answers if a is not None]
        c = Counter(nn).most_common(1); plur, cnt = c[0] if c else (None, 0)
        if cnt >= 3 and plur == r["gold"]:
            stats[name]["correct"] += 1; continue
        # classify the FAILURE per view, take the modal class
        vc = Counter()
        for (f_, q_), a_ in zip(parses, answers):
            pcts = [fa for fa in f_ if fa.get("ftype") == "pct"]
            if not pcts: vc["pct_absent"] += 1
            elif all(int(fa.get("p", -1)) != int(gold_pct["p"]) for fa in pcts): vc["p_wrong"] += 1
            elif all(sorted(fa.get("args", [])) != sorted(gold_pct["args"]) for fa in pcts): vc["args_wrong"] += 1
            elif a_ is None: vc["other_factor"] += 1
            else: vc["other_factor"] += 1
        cls = vc.most_common(1)[0][0] if vc else "no_view_parse"
        stats[name][cls] += 1
        val_note[name].append((gold_pct["p"], r["gold"]))
print("\n[classification] per pct cell (12 rows each):")
for name in sorted(stats):
    s = stats[name]
    print(f"  {name:12s} correct {s['correct']:2d} | " +
          "  ".join(f"{c} {s[c]}" for c in CLS if s[c]))
tot = Counter()
for name, s in stats.items():
    if name.startswith("base:pct") or "pct" in name:
        for c in CLS: tot[c] += s[c]
n_fail = sum(tot.values())
print(f"\n=== DOMINANT FAILURE CLASS (all pct cells, {n_fail} failures): ===")
for c, n in tot.most_common():
    if n: print(f"  {c:14s} {n:3d}  ({n/max(n_fail,1):.0%})")
iso = stats.get("base:pct", Counter())
print(f"\n[isolated vs embedded] base:pct failures: " +
      "  ".join(f"{c} {iso[c]}" for c in CLS if iso[c]))
json.dump({"census_forms": dict(forms.most_common(12)),
           "mint_form_in_mix": forms.get(MINT_FORM, 0),
           "cells": {k: dict(v) for k, v in stats.items()},
           "dominant": tot.most_common()},
          open(".cache/pct_read.json", "w"), indent=1)
print("[saved] .cache/pct_read.json")
