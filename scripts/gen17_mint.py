"""gen17_mint.py — the gen-17 fire's audited diet lines (2026-07-23).

Three minted lines + held fixtures, every row solve-gated and
canon-deduped, letters consecutive by construction:
  H — hundreds coverage: given values spanning hundreds {3..9} (the
      diet-wall cure). 3,000 train + 200 held. m=1000.
  D — add-dup: additive self-pairs (the census hole's cure — zero
      mass in 54 corpora). 2,000 train + 200 held.
  C — crown synthetics under mg2 incl. a>1 FRAC_OF ([126]-validated)
      and OP_APPLY k-legs; floor-paired (macro row + expansion-rendered
      prime twin), one knot, dedup at level 0. 1,000 knots = 2,000 rows.
Sub voicing follows the trained convention: "X exceeds R by Y" ==
sub(args=[X,Y], result=R).
"""
import json, sys, random
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from hash_audit_iso import canon
from tta_alg2_dials import solve2
from mycelium.macros import expand_graph

LET = "abcdefghijklmnopqrstuvwx"
rng = random.Random(1700)


def compact(facs, q):
    used = sorted({v for f in facs for v in (list(f.get("args", [])) +
                   [f[k] for k in ("result", "var", "x", "y") if k in f])})
    rm = {v: i for i, v in enumerate(used)}
    out = []
    for f in facs:
        f = dict(f)
        if "args" in f:
            f["args"] = [rm[v] for v in f["args"]]
        for kk in ("result", "var", "x", "y"):
            if kk in f:
                f[kk] = rm[f[kk]]
        out.append(f)
    return out, rm[q]


def render(facs, q):
    """Desk-dialect text from gold factors (order = factor order)."""
    mx = max(v for f in facs for v in (list(f.get("args", [])) +
             [f[k] for k in ("result", "var", "x", "y") if k in f]))
    n_disp = mx + 1
    sents = []
    for f in facs:
        t = f["ftype"]
        if t == "given":
            sents.append(f"{LET[f['var']]} is {f['value']}.")
        elif t == "fdiv":
            rem = LET[n_disp]; n_disp += 1
            sents.append(f"When {LET[f['var']]} is divided by {f['k']}, "
                         f"the quotient is {LET[f['result']]} and the remainder is {rem}.")
        elif t == "macro" and f["name"] == "FRAC_OF":
            pre = f"{f['a']} times " if f["a"] > 1 else ""
            sents.append(f"When {pre}{LET[f['x']]} is divided by {f['k']}, "
                         f"the quotient is {LET[f['result']]}.")
        elif t == "macro" and f["name"] == "OP_APPLY":
            l1 = (f"{f['k1']} times " if f["k1"] > 1 else "") + LET[f["x"]]
            l2 = (f"{f['k2']} times " if f["k2"] > 1 else "") + LET[f["y"]]
            word = "plus" if f["op"] == "add" else "minus"
            sents.append(f"{l1} {word} {l2} equals {LET[f['result']]}.")
        elif t == "rel" and f["op"] == "mul":
            i, j = f["args"]
            sents.append(f"{LET[i]} times {LET[j]} equals {LET[f['result']]}.")
        elif t == "rel" and f["op"] == "add":
            i, j = f["args"]
            sents.append(f"{LET[i]} plus {LET[j]} equals {LET[f['result']]}.")
        elif t == "rel" and f["op"] == "sub":
            i, j = f["args"]
            sents.append(f"{LET[i]} exceeds {LET[f['result']]} by {LET[j]}.")
        else:
            raise AssertionError(f)
    pre = "Consider the numbers " + ", ".join(LET[:n_disp]) + ". "
    return pre + " ".join(sents) + f" What is {LET[q]}?"


def bank(rows, seen, facs, q, m):
    a_ = solve2(facs, q, {"n_vars": 24, "m": m})
    if a_ is None or not (0 <= a_ <= 999):
        return False
    dg, _ = canon({"factors": facs, "n_vars": 24, "query_var": q})
    if dg in seen:
        return False
    seen.add(dg)
    rows.append({"text": render(facs, q), "factors": facs, "mentions": {},
                 "n_vars": 24, "query_var": q, "decisions": 1, "m": m,
                 "solution": [0] * 24})
    return True


def mint_hundreds(n, seen):
    rows, tries = [], 0
    while len(rows) < n and tries < n * 40:
        tries += 1
        pat = rng.choice(["hadd", "hsuba", "hmulinv", "hfdiv", "hecho"])
        X = rng.randint(300, 899)
        if pat == "hadd":
            Y = rng.randint(10, 99)
            facs = [{"ftype": "given", "var": 0, "value": X, "spans": []},
                    {"ftype": "given", "var": 1, "value": Y, "spans": []},
                    {"ftype": "rel", "op": "add", "args": [0, 1], "result": 2, "spans": []}]
            q = 2
        elif pat == "hsuba":
            Y = rng.randint(10, min(99, X - 1))
            facs = [{"ftype": "given", "var": 0, "value": X, "spans": []},
                    {"ftype": "given", "var": 1, "value": Y, "spans": []},
                    {"ftype": "rel", "op": "add", "args": [1, 2], "result": 0, "spans": []}]
            q = 2
        elif pat == "hmulinv":
            K = rng.randint(2, 9)
            c = rng.randint(34, 999 // K)
            X = K * c
            if X < 300: continue
            facs = [{"ftype": "given", "var": 0, "value": X, "spans": []},
                    {"ftype": "given", "var": 1, "value": K, "spans": []},
                    {"ftype": "rel", "op": "mul", "args": [1, 2], "result": 0, "spans": []}]
            q = 2
        elif pat == "hfdiv":
            K = rng.randint(2, 9)
            facs = [{"ftype": "given", "var": 0, "value": X, "spans": []},
                    {"ftype": "fdiv", "var": 0, "k": K, "result": 1, "spans": []}]
            q = 1
        else:
            facs = [{"ftype": "given", "var": 0, "value": X, "spans": []},
                    {"ftype": "given", "var": 1, "value": 1, "spans": []},
                    {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2, "spans": []}]
            q = 2
        bank(rows, seen, facs, q, 1000)
    return rows


def mint_adup(n, seen):
    rows, tries = [], 0
    while len(rows) < n and tries < n * 40:
        tries += 1
        pat = rng.choice(["d1", "d1", "dchain", "dderived"])
        if pat == "d1":
            X = rng.randint(2, 480)
            facs = [{"ftype": "given", "var": 0, "value": X, "spans": []},
                    {"ftype": "rel", "op": "add", "args": [0, 0], "result": 1, "spans": []}]
            q = 1
        elif pat == "dchain":
            X = rng.randint(2, 140)
            D = rng.randint(1, 99)
            facs = [{"ftype": "given", "var": 0, "value": X, "spans": []},
                    {"ftype": "rel", "op": "add", "args": [0, 0], "result": 1, "spans": []},
                    {"ftype": "given", "var": 2, "value": D, "spans": []},
                    {"ftype": "rel", "op": "add", "args": [1, 2], "result": 3, "spans": []}]
            q = 3
        else:
            X = rng.randint(2, 40)
            Y = rng.randint(2, 40)
            facs = [{"ftype": "given", "var": 0, "value": X, "spans": []},
                    {"ftype": "given", "var": 1, "value": Y, "spans": []},
                    {"ftype": "rel", "op": "add", "args": [0, 1], "result": 2, "spans": []},
                    {"ftype": "rel", "op": "add", "args": [2, 2], "result": 3, "spans": []}]
            q = 3
        bank(rows, seen, facs, q, 1000)
    return rows


def mint_crowns(n_knots, seen):
    """Floor-paired: macro row + expansion-rendered prime twin, one knot."""
    rows, tries, knots = [], 0, 0
    while knots < n_knots and tries < n_knots * 60:
        tries += 1
        if rng.random() < 0.5:
            a = rng.choice([1, 1, 2, 3, 4])
            k = rng.randint(2, 9)
            x = rng.randint(2, 99)
            mfacs = [{"ftype": "given", "var": 0, "value": x, "spans": []},
                     {"ftype": "macro", "name": "FRAC_OF", "a": a, "k": k,
                      "x": 0, "result": 1}]
            q = 1
        else:
            op = rng.choice(["add", "sub"])
            k1, k2 = rng.randint(1, 9), rng.randint(1, 9)
            if op == "add" and k1 == 1 and k2 == 1: continue
            X, Y = rng.randint(2, 60), rng.randint(2, 60)
            if op == "sub" and k1 * X < k2 * Y: continue
            mfacs = [{"ftype": "given", "var": 0, "value": X, "spans": []},
                     {"ftype": "given", "var": 1, "value": Y, "spans": []},
                     {"ftype": "macro", "name": "OP_APPLY", "op": op, "k1": k1,
                      "x": 0, "k2": k2, "y": 1, "result": 2}]
            q = 2
        pfacs, _ = expand_graph([dict(f) for f in mfacs], 24)
        pf2, q_p = compact(pfacs, q)
        a_p = solve2(pf2, q_p, {"n_vars": 24, "m": 1000})
        if a_p is None or not (0 <= a_p <= 999): continue
        dg_p, _ = canon({"factors": pf2, "n_vars": 24, "query_var": q_p})
        if dg_p in seen: continue     # knot dedup at level 0
        dg_m, _ = canon({"factors": mfacs, "n_vars": 24, "query_var": q})
        assert dg_m == dg_p, (mfacs, dg_m, dg_p)
        seen.add(dg_p)
        for facs_, q_ in ((mfacs, q), (pf2, q_p)):
            rows.append({"text": render(facs_, q_), "factors": facs_,
                         "mentions": {}, "n_vars": 24, "query_var": q_,
                         "decisions": 1, "m": 1000, "solution": [0] * 24})
        knots += 1
    return rows


seen = set()
h_train = mint_hundreds(3000, seen)
h_held = mint_hundreds(200, seen)
d_train = mint_adup(2000, seen)
d_held = mint_adup(200, seen)
c_train = mint_crowns(1000, seen)

for name, rows in (("gen17_hundreds", h_train), ("gen17_hundreds_held", h_held),
                   ("gen17_adup", d_train), ("gen17_adup_held", d_held),
                   ("gen17_crowns", c_train)):
    with open(f".cache/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[g17-mint] {name}: {len(rows)} rows")

# THE MIX (dose law): base + lines; book5 organic at 10 reps/unique
mix = []
for ln in open(".cache/crown_v4.jsonl"):
    mix.append(ln.rstrip("\n"))
for rows in (h_train, d_train, c_train):
    for r in rows:
        mix.append(json.dumps(r))
b5 = [ln.rstrip("\n") for ln in open(".cache/book5_prose_pairs.jsonl")]
for _ in range(10):
    mix.extend(b5)
rng.shuffle(mix)
with open(".cache/gen17_mix.jsonl", "w") as f:
    for ln in mix:
        f.write(ln + "\n")
n = len(mix)
print(f"[g17-mint] MIX: {n} rows = base 70910 + hundreds {len(h_train)} "
      f"({len(h_train)/n:.1%}) + adup {len(d_train)} ({len(d_train)/n:.1%}) + "
      f"crowns {len(c_train)} ({len(c_train)/n:.1%}) + book5 {len(b5)}x10 "
      f"({10*len(b5)/n:.1%}, 10 reps/unique)")
