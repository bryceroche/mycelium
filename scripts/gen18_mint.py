"""gen18_mint.py — the gen-18 charter's mint (2026-07-24, on the word).

SPANS FROM BIRTH (charter input #8): every minted row emits gold factor
spans (each sentence's char range) AND letter mentions — the canvas
feeds from the whole diet. Maintenance doses (input #7): hundreds
1,000 fresh uniques + add-dup 500 + crowns 500 knots (both floors),
all deduped against every prior line. Mix = crown_v4 base + lines +
book-5 organic x10. RATION INDEX (inputs #4/#5): base rows with
n_vars >= 14 (the drift autopsy's hard band) marked for arm B's
hot-phase upweighting. One mix, one precompute; the arms differ only
in sampling.
"""
import json, random, re, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from hash_audit_iso import canon
from tta_alg2_dials import solve2
from mycelium.macros import expand_graph

LET = "abcdefghijklmnopqrstuvwx"
rng = random.Random(1800)


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


def render_spanned(facs, q):
    """Desk-dialect text with PER-FACTOR SPANS and letter mentions."""
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
    text = pre
    facs2 = []
    for f, s in zip(facs, sents):
        start = len(text)
        text += s + " "
        f = dict(f)
        f["spans"] = [[start, start + len(s)]]
        facs2.append(f)
    text += f"What is {LET[q]}?"
    mentions = {}
    for v in range(n_disp):
        sp = [[m.start(), m.end()]
              for m in re.finditer(rf"\b{LET[v]}\b", text)]
        if sp and v < 24:
            mentions[str(v)] = sp
    return text, facs2, mentions


def bank(rows, seen, facs, q, m):
    a_ = solve2(facs, q, {"n_vars": 24, "m": m})
    if a_ is None or not (0 <= a_ <= 999):
        return False
    dg, _ = canon({"factors": facs, "n_vars": 24, "query_var": q})
    if dg in seen:
        return False
    seen.add(dg)
    text, facs2, mentions = render_spanned(facs, q)
    rows.append({"text": text, "factors": facs2, "mentions": mentions,
                 "n_vars": 24, "query_var": q, "decisions": 1, "m": m,
                 "solution": [0] * 24})
    return True


# dedup against every prior minted line
seen = set()
for path in ("gen17_hundreds", "gen17_hundreds_held", "gen17c_hundreds",
             "gen17_adup", "gen17_adup_held", "gen17_crowns"):
    for ln in open(f".cache/{path}.jsonl"):
        r = json.loads(ln)
        dg, _ = canon({"factors": r["factors"], "n_vars": 24,
                       "query_var": r["query_var"]})
        seen.add(dg)
print(f"[g18] preloaded {len(seen)} prior knots", flush=True)

src = open("scripts/gen17_mint.py").read()
cut = src.index("seen = set()")
ns = {}
exec(src[:cut].replace("rng = random.Random(1700)",
                       "rng = random.Random(1800)"), ns)
hund, adup = [], []
raw_h = ns["mint_hundreds"](1400, seen)      # over-mint; re-render with spans
raw_d = ns["mint_adup"](700, seen)
for r in raw_h[:1000]:
    text, facs2, mentions = render_spanned(r["factors"], r["query_var"])
    hund.append({**r, "text": text, "factors": facs2, "mentions": mentions})
for r in raw_d[:500]:
    text, facs2, mentions = render_spanned(r["factors"], r["query_var"])
    adup.append({**r, "text": text, "factors": facs2, "mentions": mentions})
print(f"[g18] hundreds {len(hund)} | adup {len(adup)} (spans+mentions)", flush=True)

crowns = []
knots = 0
tries = 0
while knots < 500 and tries < 30000:
    tries += 1
    if rng.random() < 0.5:
        a = rng.choice([1, 1, 2, 3, 4]); k = rng.randint(2, 9); x = rng.randint(2, 99)
        mfacs = [{"ftype": "given", "var": 0, "value": x, "spans": []},
                 {"ftype": "macro", "name": "FRAC_OF", "a": a, "k": k, "x": 0, "result": 1}]
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
    if dg_p in seen: continue
    seen.add(dg_p)
    for facs_, q_ in ((mfacs, q), (pf2, q_p)):
        text, facs2, mentions = render_spanned(facs_, q_)
        crowns.append({"text": text, "factors": facs2, "mentions": mentions,
                       "n_vars": 24, "query_var": q_, "decisions": 1,
                       "m": 1000, "solution": [0] * 24})
    knots += 1
print(f"[g18] crowns {knots} knots ({len(crowns)} rows, spans+mentions)", flush=True)

for name, rows_ in (("gen18_hundreds", hund), ("gen18_adup", adup),
                    ("gen18_crowns", crowns)):
    with open(f".cache/{name}.jsonl", "w") as f:
        for r in rows_:
            f.write(json.dumps(r) + "\n")

mix = []
base_n = 0
for ln in open(".cache/crown_v4.jsonl"):
    mix.append(ln.rstrip("\n")); base_n += 1
for rows_ in (hund, adup, crowns):
    for r in rows_:
        mix.append(json.dumps(r))
b5 = []
for ln in open(".cache/book5_prose_pairs.jsonl"):
    r = json.loads(ln)
    if isinstance(r.get("decisions"), list): r["decisions"] = 1
    if isinstance(r.get("mentions"), list): r["mentions"] = {}
    b5.append(json.dumps(r))
for _ in range(10):
    mix.extend(b5)
random.Random(1818).shuffle(mix)
with open(".cache/gen18_mix.jsonl", "w") as f:
    for ln in mix:
        f.write(ln + "\n")
n = len(mix)

# RATION INDEX: hard band = base-family rows with n_vars >= 14
ration = [i for i, ln in enumerate(mix)
          if json.loads(ln).get("n_vars", 0) >= 14]
json.dump(ration, open(".cache/gen18_ration_idx.json", "w"))
new_mass = len(hund) + len(adup) + len(crowns) + 10 * len(b5)
print(f"[g18] MIX: {n} rows | new mass {new_mass} ({new_mass/n:.1%}) | "
      f"ration band (n_vars>=14): {len(ration)} rows ({len(ration)/n:.1%})", flush=True)
