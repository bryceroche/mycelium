"""mint_search_greedy.py — GUT #41(b): THE MINT-SEARCH GREEDY FIRE
(2026-07-21) — the admitted crown's corpus engine. Two arms, 1000 banked
rows each, both knot-deduped at level 0 (the floor-identity protocol):
  BLIND  — random structure+values, dedup post-hoc (the standing mint).
  GREEDY — #41's spec: TRANSPOSITION peek before the expensive gate
           (skip already-banked isomorphs pre-solve) + COVERAGE steering
           via the value notebook (cover-multisets counted live; reject
           candidates whose composition is already >=3-populated — steer
           toward NEW and ABSENT-FROM-TRAIN compositions, the 44% sliver).
FRAME (#14 restored, translated one algebra down since dedup-by-knot
makes whole-classes trivial): GREEDY distinct-covers-per-1000 >= 2x
BLIND (the win); attempts-per-bank improvement MODEST (survival's
residual rejections are global-uniqueness failures lookahead can't see).
Output: .cache/crown_corpus_{blind,greedy}.jsonl — floor-paired
FRAC_OF-centered rows, the training era's substrate.
"""
import json, sys, os, random
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from collections import Counter
from mycelium.macros import expand_graph, MACRO_GRAMMAR_VERSION
from hash_audit_iso import canon
from tta_alg2_dials import solve2
from schema_miner import mine_graph

train_covers = set()
try:
    tc = json.load(open(".cache/knot_decomposition_census.json"))
    # rebuilt cover-set proxy: not banked as a set — recompute is heavy, so the
    # absent-read uses the banked train class-count table for membership at
    # sub-knot grain; whole-cover absence is reported vs the greedy's own log
    train_classes = set(json.load(open(".cache/train_class_counts.json")).keys())
except FileNotFoundError:
    train_classes = set()


def cover(factors):
    subs = sorted(mine_graph(factors), key=lambda t: (-t[1], t[0]))
    taken, out = set(), []
    for dg, k, idxs in subs:
        if not (set(idxs) & taken):
            taken.update(idxs)
            out.append(dg)
    return tuple(sorted(out))


LET = "abcdefghijklmnopqrstuvwx"


def render_macro(facs, q):
    used = sorted({v for f in facs for v in
                   (list(f.get("args", [])) + [f[k] for k in
                    ("result", "var", "x", "y") if k in f])})
    L = {v: LET[j] for j, v in enumerate(used)}
    s = ["Consider the numbers " + ", ".join(L[v] for v in used) + "."]
    for f in facs:
        if f["ftype"] == "given":
            s.append(f"{L[f['var']]} is {f['value']}.")
        elif f["ftype"] == "macro" and f["name"] == "FRAC_OF":
            if f["a"] == 1:
                s.append(f"When {L[f['x']]} is divided by {f['k']}, the "
                         f"quotient is {L[f['result']]}.")
            else:
                s.append(f"{f['a']} times {L[f['x']]}, divided by {f['k']}, "
                         f"gives a quotient of {L[f['result']]}.")
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
                   (list(f.get("args", [])) + [f[k] for k in ("result", "var")
                    if k in f])})
    L = {v: LET[j] for j, v in enumerate(used)}
    s = ["Consider the numbers " + ", ".join(L[v] for v in used) + "."]
    for f in facs:
        if f["ftype"] == "given":
            s.append(f"{L[f['var']]} is {f['value']}.")
        elif f["ftype"] == "rel":
            wd = {"add": "plus", "mul": "times", "sub": "exceeds"}[f["op"]]
            if f["op"] == "sub":
                s.append(f"{L[f['args'][0]]} exceeds {L[f['result']]} by {L[f['args'][1]]}.")
            else:
                s.append(f"{L[f['args'][0]]} {wd} {L[f['args'][1]]} equals {L[f['result']]}.")
        elif f["ftype"] == "fdiv":
            s.append(f"When {L[f['var']]} is divided by {f['k']}, the "
                     f"quotient is {L[f['result']]}.")
    s.append(f"What is {L[q]}?")
    return " ".join(s)


def propose(rng):
    """One candidate FRAC_OF-centered structure (the action space)."""
    pat = rng.choice(["single", "chain2", "skeleton73", "crown_op", "op_crown",
                      "crown_mul"])
    x = rng.randint(2, 120)
    a = rng.choice([1, 1, 2, 3, 3, 4, 5, 6, 7])
    k = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 12])
    facs = [{"ftype": "given", "var": 0, "value": x, "spans": []},
            {"ftype": "macro", "name": "FRAC_OF", "a": a, "k": k, "x": 0,
             "result": 1, "spans": []}]
    q = 1
    if pat == "chain2":
        k2 = rng.choice([2, 3, 4, 5])
        facs.append({"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": k2,
                     "x": 1, "result": 2, "spans": []})
        q = 2
    elif pat == "skeleton73":
        a2, k2 = rng.choice([1, 2, 3]), rng.choice([3, 4, 5, 7])
        facs.append({"ftype": "macro", "name": "FRAC_OF", "a": a2, "k": k2,
                     "x": 0, "result": 2, "spans": []})
        facs.append({"ftype": "rel", "op": "add", "args": [1, 2], "result": 3,
                     "spans": []})
        q = 3
        if rng.random() < 0.5:
            facs.append({"ftype": "macro", "name": "FRAC_OF", "a": 1, "k": 2,
                         "x": 3, "result": 4, "spans": []})
            q = 4
    elif pat == "crown_op":
        d = rng.randint(1, 40)
        facs.append({"ftype": "given", "var": 2, "value": d, "spans": []})
        facs.append({"ftype": "macro", "name": "OP_APPLY",
                     "op": rng.choice(["add", "sub"]),
                     "k1": rng.choice([1, 2, 3]), "x": 1,
                     "k2": rng.choice([1, 2]), "y": 2, "result": 3, "spans": []})
        q = 3
    elif pat == "op_crown":
        y = rng.randint(1, 60)
        facs = [{"ftype": "given", "var": 0, "value": x, "spans": []},
                {"ftype": "given", "var": 1, "value": y, "spans": []},
                {"ftype": "macro", "name": "OP_APPLY", "op": "add",
                 "k1": rng.choice([1, 2, 3]), "x": 0,
                 "k2": rng.choice([1, 2]), "y": 1, "result": 2, "spans": []},
                {"ftype": "macro", "name": "FRAC_OF", "a": 1,
                 "k": rng.choice([2, 3, 4, 5]), "x": 2, "result": 3, "spans": []}]
        q = 3
    elif pat == "crown_mul":
        d = rng.randint(2, 9)
        facs.append({"ftype": "given", "var": 2, "value": d, "spans": []})
        facs.append({"ftype": "rel", "op": "mul", "args": [1, 2], "result": 3,
                     "spans": []})
        q = 3
    return facs, q, pat


def compact(pfacs, q):
    used = sorted({v for f in pfacs for v in
                   (list(f.get("args", [])) + [f[k] for k in ("result", "var")
                    if k in f])})
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


def run_arm(greedy, seed, n_target=1000):
    rng = random.Random(seed)
    seen_knots, cover_counts = set(), Counter()
    banked, attempts, solves = [], 0, 0
    while len(banked) < n_target and attempts < n_target * 60:
        attempts += 1
        facs, q, pat = propose(rng)
        pfacs, _ = expand_graph([dict(f) for f in facs], 24)
        pfacs, q_p = compact(pfacs, q)
        if any(f.get("value", 0) > 300 for f in pfacs):
            continue
        dg, _ = canon({"factors": pfacs, "n_vars": 24, "query_var": q_p})
        if dg in seen_knots:
            continue                      # transposition peek (both arms dedup)
        cv = cover(pfacs)
        if greedy and cover_counts[cv] >= 3:
            continue                      # coverage steering: pre-solve reject
        solves += 1
        a_ = solve2(pfacs, q_p, {"n_vars": 24, "m": 300})
        if a_ is None or not (0 <= a_ <= 300):
            continue
        seen_knots.add(dg)
        cover_counts[cv] += 1
        banked.append({"answer": a_, "query_var": q, "m": 300, "n_vars": 24,
                       "knot": dg, "grammar": MACRO_GRAMMAR_VERSION, "pat": pat,
                       "macro": {"factors": facs, "text": render_macro(facs, q)},
                       "prime": {"factors": pfacs, "text": render_prime(pfacs, q_p),
                                 "query_var": q_p}})
    covers = set(cover_counts)
    absent = sum(1 for cv in covers
                 if any(dgc not in train_classes for dgc in cv))
    return banked, dict(attempts=attempts, solves=solves,
                        covers=len(covers), absent_covers=absent)


blind, mb = run_arm(False, 4100)
greedy, mg = run_arm(True, 4200)
for tag, rows_, m in (("BLIND", blind, mb), ("GREEDY", greedy, mg)):
    with open(f".cache/crown_corpus_{tag.lower()}.jsonl", "w") as f:
        for r in rows_:
            f.write(json.dumps(r) + "\n")
    print(f"[{tag}] banked {len(rows_)} | attempts {m['attempts']} "
          f"(solves {m['solves']}) | distinct covers {m['covers']} "
          f"| covers w/ train-absent primes {m['absent_covers']}")
ratio = mg["covers"] / max(mb["covers"], 1)
apb_b, apb_g = mb["attempts"] / max(len(blind), 1), mg["attempts"] / max(len(greedy), 1)
print(f"\n=== #14's RESTORED FRAME (translated to cover grain) ===")
print(f"  covers ratio greedy/blind: {ratio:.2f}  "
      f"(>=2x = THE WIN {'HOLDS' if ratio >= 2 else 'MISSES'})")
print(f"  attempts-per-bank: blind {apb_b:.1f} vs greedy {apb_g:.1f} "
      f"(survival axis — MODEST predicted)")
json.dump({"blind": mb, "greedy": mg, "cover_ratio": ratio},
          open(".cache/mint_search_greedy.json", "w"))
print("[mint-search] banked -> crown_corpus_{blind,greedy}.jsonl + metrics")
