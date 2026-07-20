"""macro_mint.py — THE MACRO MINT (2026-07-20, the fire's first build).
Mints synthetic OP_APPLY problems at volume, FLOOR-PAIRED by
construction: every mint emits the macro row (crown one-sentence) AND
its prime twin (expand_graph — deterministic, solution-preserving).
Gates: solution-first, values <=300, uniqueness by construction (all
crown inputs given; downstream chain forced), knot-dedup at LEVEL 0
(the floor-identity protocol — canon on the EXPANDED graph).
Charter dose: ~2,000 unique pairs (the bilingual-cure mass).
"""
import json, sys, os, random
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from mycelium.macros import expand_graph, MACRO_GRAMMAR_VERSION
from hash_audit_iso import canon
from tta_alg2_dials import solve2

N_TARGET = int(os.environ.get("MINT_N", "2000"))
SEED = int(os.environ.get("MINT_SEED", "1400"))
OUT = os.environ.get("MINT_OUT", ".cache/macro_mint_pairs.jsonl")
LET = "abcdefghijklmnopqrstuvwx"
rng = random.Random(SEED)


def render_prime(facs, q):
    used = sorted({v for f in facs for v in
                   (list(f.get("args", [])) + [f[k] for k in ("result", "var")
                    if k in f])})
    L = {v: LET[j] for j, v in enumerate(used)}
    s = ["Consider the numbers " + ", ".join(L[v] for v in used) + "."]
    for f in facs:
        if f["ftype"] == "given":
            s.append(f"{L[f['var']]} is {f['value']}.")
        elif f["ftype"] == "rel" and f["op"] == "add":
            s.append(f"{L[f['args'][0]]} plus {L[f['args'][1]]} equals {L[f['result']]}.")
        elif f["ftype"] == "rel" and f["op"] == "mul":
            s.append(f"{L[f['args'][0]]} times {L[f['args'][1]]} equals {L[f['result']]}.")
        elif f["ftype"] == "rel" and f["op"] == "sub":
            s.append(f"{L[f['args'][0]]} exceeds {L[f['result']]}"
                     f" by {L[f['args'][1]]}.")   # a - b = r  <=>  a exceeds r by b
    s.append(f"What is {L[q]}?")
    return " ".join(s)


def render_macro(facs, q):
    used = sorted({v for f in facs for v in
                   (list(f.get("args", [])) + [f[k] for k in ("result", "var", "x", "y")
                    if k in f])})
    L = {v: LET[j] for j, v in enumerate(used)}
    s = ["Consider the numbers " + ", ".join(L[v] for v in used) + "."]
    for f in facs:
        if f["ftype"] == "given":
            s.append(f"{L[f['var']]} is {f['value']}.")
        elif f["ftype"] == "macro":
            k1, k2 = f["k1"], f["k2"]
            xa = L[f["x"]] if k1 == 1 else f"{k1} times {L[f['x']]}"
            yb = L[f["y"]] if k2 == 1 else f"{k2} times {L[f['y']]}"
            w = "plus" if f["op"] == "add" else "minus"
            s.append(f"{xa} {w} {yb} equals {L[f['result']]}.")
        elif f["ftype"] == "rel":
            wd = "plus" if f["op"] == "add" else "times"
            s.append(f"{L[f['args'][0]]} {wd} {L[f['args'][1]]} equals {L[f['result']]}.")
    s.append(f"What is {L[q]}?")
    return " ".join(s)


seen = set()
out = []
attempts = 0
while len(out) < N_TARGET and attempts < N_TARGET * 30:
    attempts += 1
    op = rng.choice(["add", "sub"])
    k1 = rng.choice([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 20, 25, 40])
    k2 = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13])
    x = rng.randint(1, 60)
    y = rng.randint(1, 60)
    r = k1 * x + k2 * y if op == "add" else k1 * x - k2 * y
    if not (0 <= r <= 300 and k1 * x <= 300 and k2 * y <= 300):
        continue
    facs = [{"ftype": "given", "var": 0, "value": x, "spans": []},
            {"ftype": "given", "var": 1, "value": y, "spans": []},
            {"ftype": "macro", "name": "OP_APPLY", "op": op, "k1": k1, "x": 0,
             "k2": k2, "y": 1, "result": 2}]
    q = 2
    # optional downstream chain (40%): r feeds one more relation
    if rng.random() < 0.4:
        d = rng.randint(1, 40)
        if rng.random() < 0.5 and r + d <= 300:
            facs.append({"ftype": "given", "var": 3, "value": d, "spans": []})
            facs.append({"ftype": "rel", "op": "add", "args": [2, 3],
                         "result": 4, "spans": []})
            q = 4
        elif r * 2 <= 300:
            facs.append({"ftype": "given", "var": 3, "value": 2, "spans": []})
            facs.append({"ftype": "rel", "op": "mul", "args": [2, 3],
                         "result": 4, "spans": []})
            q = 4
    pfacs, nv = expand_graph(facs, 24)
    # compact renumbering: expansion temps land above the 24-slot bank —
    # relabel all vars consecutively (solution-preserving pure relabel) so
    # prime twins are TRAINABLE rows. (Caught by the 50-row gold smoke.)
    used = sorted({v for f in pfacs for v in
                   (list(f.get("args", [])) + [f[k] for k in ("result", "var")
                    if k in f])})
    remap = {v: i for i, v in enumerate(used)}
    for f in pfacs:
        if "args" in f:
            f["args"] = [remap[v] for v in f["args"]]
        for k in ("result", "var"):
            if k in f:
                f[k] = remap[f[k]]
    q_p = remap[q]
    a = solve2(pfacs, q_p, {"n_vars": 24, "m": 300})
    if a is None:
        continue
    dg, _ = canon({"factors": facs, "n_vars": 24, "query_var": q})
    if dg in seen:
        continue
    dg_p, _ = canon({"factors": pfacs, "n_vars": 24, "query_var": q_p})
    assert dg == dg_p, "floor-identity violated at mint"
    seen.add(dg)
    out.append({"answer": a, "query_var": q, "m": 300, "n_vars": 24,
                "knot": dg, "grammar": MACRO_GRAMMAR_VERSION,
                "macro": {"factors": facs, "text": render_macro(facs, q)},
                "prime": {"factors": pfacs, "text": render_prime(pfacs, q_p),
                          "query_var": q_p}})

with open(OUT, "w") as f:
    for r_ in out:
        f.write(json.dumps(r_) + "\n")
from collections import Counter
sig = Counter((r_["macro"]["factors"][2]["op"],
               r_["macro"]["factors"][2]["k1"] == 1) for r_ in out)
print(f"[mint] {len(out)} unique floor-paired crowns -> {OUT} "
      f"({attempts} attempts, {attempts-len(out)} rejected/dup)")
print(f"[mint] signatures (op, affine-leg): {dict(sig)}")
