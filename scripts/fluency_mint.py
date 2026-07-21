"""fluency_mint.py — the fluency run's key-diverse mint (2026-07-21).
FRAC_OF-centered pairs with a FIVE-KEY render bank (#42's rider): the
same crown voiced as quotient / of / per / split / scaled. Floor-paired,
knot-deduped at level 0, key recorded per row. Two outputs: the training
mint (seed 5100, n=2000) and the held-out bar set (seed 5900, n=200,
key-stratified, knot-disjoint from training).
"""
import json, sys, os, random
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from mycelium.macros import expand_graph, MACRO_GRAMMAR_VERSION
from hash_audit_iso import canon
from tta_alg2_dials import solve2

LET = "abcdefghijklmnopqrstuvwx"
KEYS = ["quotient", "of", "per", "split", "scaled"]


def frac_sentence(key, a, k, x, r, rng):
    if key == "quotient":
        return (f"When {x} is divided by {k}, the quotient is {r}." if a == 1 else
                f"{a} times {x}, divided by {k}, gives a quotient of {r}.")
    if key == "of":
        return (f"One {['','','half','third','quarter','fifth','sixth','seventh','eighth','ninth'][k] if 2 <= k <= 9 else f'{k}th'} of {x} is {r}." if a == 1 else
                f"{a} {['','','halves','thirds','quarters','fifths','sixths','sevenths','eighths','ninths'][k] if 2 <= k <= 9 else f'{k}ths'} of {x} is {r}.")
    if key == "per":
        return (f"Sharing {x} among {k} gives {r} each." if a == 1 else
                f"At {a} for every {k} of {x}, one gets {r}.")
    if key == "split":
        return (f"{x} split {k} ways yields {r} per part." if a == 1 else
                f"{x} scaled by {a} and split {k} ways yields {r} per part.")
    return (f"{x} reduced {k}-fold is {r}." if a == 1 else
            f"{x} grown {a}-fold then reduced {k}-fold is {r}.")


def render_macro(facs, q, key, rng):
    used = sorted({v for f in facs for v in
                   (list(f.get("args", [])) + [f[kk] for kk in
                    ("result", "var", "x", "y") if kk in f])})
    L = {v: LET[j] for j, v in enumerate(used)}
    s = ["Consider the numbers " + ", ".join(L[v] for v in used) + "."]
    for f in facs:
        if f["ftype"] == "given":
            s.append(f"{L[f['var']]} is {f['value']}.")
        elif f["ftype"] == "macro" and f["name"] == "FRAC_OF":
            s.append(frac_sentence(key, f["a"], f["k"], L[f["x"]],
                                   L[f["result"]], rng))
        elif f["ftype"] == "macro":
            k1, k2 = f["k1"], f["k2"]
            xa = L[f["x"]] if k1 == 1 else f"{k1} times {L[f['x']]}"
            yb = L[f["y"]] if k2 == 1 else f"{k2} times {L[f['y']]}"
            w = "plus" if f["op"] == "add" else "minus"
            s.append(f"{xa} {w} {yb} equals {L[f['result']]}.")
        elif f["ftype"] == "rel":
            wd = {"add": "plus", "mul": "times"}[f["op"]]
            s.append(f"{L[f['args'][0]]} {wd} {L[f['args'][1]]} equals {L[f['result']]}.")
        elif f["ftype"] == "fdiv":
            s.append(f"When {L[f['var']]} is divided by {f['k']}, the "
                     f"quotient is {L[f['result']]}.")
    s.append(f"What is {L[q]}?")
    return " ".join(s)


def render_prime(facs, q):
    used = sorted({v for f in facs for v in
                   (list(f.get("args", [])) + [f[kk] for kk in ("result", "var")
                    if kk in f])})
    L = {v: LET[j] for j, v in enumerate(used)}
    s = ["Consider the numbers " + ", ".join(L[v] for v in used) + "."]
    for f in facs:
        if f["ftype"] == "given":
            s.append(f"{L[f['var']]} is {f['value']}.")
        elif f["ftype"] == "rel":
            s.append(f"{L[f['args'][0]]} "
                     f"{'plus' if f['op']=='add' else 'times'} "
                     f"{L[f['args'][1]]} equals {L[f['result']]}.")
        elif f["ftype"] == "fdiv":
            s.append(f"When {L[f['var']]} is divided by {f['k']}, the "
                     f"quotient is {L[f['result']]}.")
    s.append(f"What is {L[q]}?")
    return " ".join(s)


def propose(rng):
    pat = rng.choice(["single", "chain2", "skeleton73", "crown_op", "crown_mul"])
    x = rng.randint(2, 120)
    a = rng.choice([1, 1, 2, 3, 3, 4, 5, 6, 7])
    k = rng.choice([2, 3, 4, 5, 6, 7, 8, 9])
    facs = [{"ftype": "given", "var": 0, "value": x, "spans": []},
            {"ftype": "macro", "name": "FRAC_OF", "a": a, "k": k, "x": 0,
             "result": 1, "spans": []}]
    q = 1
    if pat == "chain2":
        facs.append({"ftype": "macro", "name": "FRAC_OF", "a": 1,
                     "k": rng.choice([2, 3, 4, 5]), "x": 1, "result": 2,
                     "spans": []})
        q = 2
    elif pat == "skeleton73":
        facs.append({"ftype": "macro", "name": "FRAC_OF",
                     "a": rng.choice([1, 2, 3]), "k": rng.choice([3, 4, 5, 7]),
                     "x": 0, "result": 2, "spans": []})
        facs.append({"ftype": "rel", "op": "add", "args": [1, 2], "result": 3,
                     "spans": []})
        q = 3
        if rng.random() < 0.5:
            facs.append({"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 2,
                         "x": 3, "result": 4, "spans": []})
            q = 4
    elif pat == "crown_op":
        facs.append({"ftype": "given", "var": 2, "value": rng.randint(1, 40),
                     "spans": []})
        facs.append({"ftype": "macro", "name": "OP_APPLY",
                     "op": rng.choice(["add", "sub"]),
                     "k1": rng.choice([1, 2, 3]), "x": 1,
                     "k2": rng.choice([1, 2]), "y": 2, "result": 3, "spans": []})
        q = 3
    elif pat == "crown_mul":
        facs.append({"ftype": "given", "var": 2, "value": rng.randint(2, 9),
                     "spans": []})
        facs.append({"ftype": "rel", "op": "mul", "args": [1, 2], "result": 3,
                     "spans": []})
        q = 3
    return facs, q


def compact(pfacs, q):
    used = sorted({v for f in pfacs for v in
                   (list(f.get("args", [])) + [f[kk] for kk in ("result", "var")
                    if kk in f])})
    remap = {v: i for i, v in enumerate(used)}
    out = []
    for f in pfacs:
        f = dict(f)
        if "args" in f:
            f["args"] = [remap[v] for v in f["args"]]
        for kk in ("result", "var"):
            if kk in f:
                f[kk] = remap[f[kk]]
        out.append(f)
    return out, remap[q]


def mint(seed, n_target, exclude):
    rng = random.Random(seed)
    seen, out = set(exclude), []
    attempts = 0
    while len(out) < n_target and attempts < n_target * 60:
        attempts += 1
        facs, q = propose(rng)
        pfacs, _ = expand_graph([dict(f) for f in facs], 24)
        pfacs, q_p = compact(pfacs, q)
        if any(f.get("value", 0) > 300 for f in pfacs):
            continue
        dg, _ = canon({"factors": pfacs, "n_vars": 24, "query_var": q_p})
        if dg in seen:
            continue
        a_ = solve2(pfacs, q_p, {"n_vars": 24, "m": 300})
        if a_ is None or not (0 <= a_ <= 300):
            continue
        seen.add(dg)
        key = KEYS[len(out) % len(KEYS)]      # stratified
        out.append({"answer": a_, "query_var": q, "m": 300, "n_vars": 24,
                    "knot": dg, "grammar": MACRO_GRAMMAR_VERSION, "key": key,
                    "macro": {"factors": facs,
                              "text": render_macro(facs, q, key, rng)},
                    "prime": {"factors": pfacs, "text": render_prime(pfacs, q_p),
                              "query_var": q_p}})
    return out, seen


train, seen = mint(5100, 2000, set())
held, _ = mint(5900, 200, seen)               # knot-disjoint from training
for tag, rows_ in (("train", train), ("held", held)):
    with open(f".cache/fluency_mint_{tag}.jsonl", "w") as f:
        for r in rows_:
            f.write(json.dumps(r) + "\n")
from collections import Counter
print(f"[fluency-mint] train {len(train)} | held {len(held)} (knot-disjoint) | "
      f"keys {Counter(r['key'] for r in train)}")
