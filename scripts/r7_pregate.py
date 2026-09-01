"""r7_pregate.py — round-7 mechanical pre-gate (CPU, zero-GPU).
Checks per draft: schema, bounds [0,300], var coverage (0..n-1 all used),
query-not-given, no degenerate/dup factors, anchor law (given values
literal in text, with the unit-1 and lexical whitelist), and SOLVE:
brute-force the factor system over [0,m] to confirm the query value is
UNIQUE and equals the answer. Emits a verdict per row.
"""
import json, re, sys, itertools

LEX_ANCHORS = {2: ["half", "couple", "twice", "double", "each other", "pair"],
               3: ["third", "triple", "thrice"],
               4: ["quarter", "fourth"],
               12: ["dozen"]}
WORDNUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
           "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12}

def anchored(val, text):
    t = text.lower()
    if val == 1:
        return True, "unit"
    if re.search(r'(?<![\d.])' + str(val) + r'(?!\d)', text):
        return True, "literal"
    for w, n in WORDNUM.items():
        if n == val and re.search(r'\b' + w + r'\b', t):
            return True, f"word:{w}"
    for w in LEX_ANCHORS.get(val, []):
        if w in t:
            return True, f"lex:{w}"
    return False, None

def solve(n_vars, factors, m):
    """Constraint-propagate; fall back to bounded search on stuck vars.
    Returns list of full assignments consistent with all factors."""
    val = {}
    for f in factors:
        if f["ftype"] == "given":
            if f["var"] in val and val[f["var"]] != f["value"]:
                return None, "conflicting givens"
            val[f["var"]] = f["value"]
    def propagate(val):
        val = dict(val); changed = True
        while changed:
            changed = False
            for f in factors:
                if f["ftype"] != "rel":
                    continue
                a, b, r = f["args"][0], f["args"][1], f["result"]
                ka, kb, kr = a in val, b in val, r in val
                if f["op"] == "add":
                    if ka and kb and not kr:
                        val[r] = val[a] + val[b]; changed = True
                    elif ka and kr and not kb:
                        val[b] = val[r] - val[a]; changed = True
                    elif kb and kr and not ka:
                        val[a] = val[r] - val[b]; changed = True
                    elif ka and kb and kr and val[a] + val[b] != val[r]:
                        return None
                else:
                    if ka and kb and not kr:
                        val[r] = val[a] * val[b]; changed = True
                    elif ka and kr and not kb and val[a] != 0:
                        if val[r] % val[a]:
                            return None
                        val[b] = val[r] // val[a]; changed = True
                    elif kb and kr and not ka and val[b] != 0:
                        if val[r] % val[b]:
                            return None
                        val[a] = val[r] // val[b]; changed = True
                    elif ka and kb and kr and val[a] * val[b] != val[r]:
                        return None
        return val
    v0 = propagate(val)
    if v0 is None:
        return None, "contradiction from givens"
    free = [i for i in range(n_vars) if i not in v0]
    if not free:
        return [v0], "forward"
    # bounded search: branch on one free var at a time (systems are tiny)
    sols = []
    def rec(v):
        miss = [i for i in range(n_vars) if i not in v]
        if not miss:
            sols.append(v); return
        x = miss[0]
        for c in range(0, m + 1):
            v2 = dict(v); v2[x] = c
            v3 = propagate(v2)
            if v3 is not None:
                rec(v3)
    rec(v0)
    return sols, "search"

def check(d):
    errs, warns = [], []
    n, m, q = d["n_vars"], d["m"], d["query_var"]
    fs = d["factors"]
    used = set()
    seen = set()
    for f in fs:
        key = json.dumps(f, sort_keys=True)
        if key in seen:
            errs.append(f"dup factor {key}")
        seen.add(key)
        if f["ftype"] == "given":
            used.add(f["var"])
            if not (0 <= f["value"] <= m):
                errs.append(f"given v{f['var']}={f['value']} out of bounds")
            ok, how = anchored(f["value"], d["original"])
            if not ok:
                errs.append(f"UNANCHORED given v{f['var']}={f['value']}")
            elif how and not how.startswith("literal") and how != "unit":
                warns.append(f"v{f['var']}={f['value']} anchored via {how}")
        elif f["ftype"] == "rel":
            used.update(f["args"]); used.add(f["result"])
            if f["op"] not in ("add", "mul"):
                errs.append(f"illegal op {f['op']}")
            if f["result"] in f["args"]:
                errs.append(f"self-referential rel {f}")
        else:
            errs.append(f"illegal ftype {f['ftype']}")
    if used != set(range(n)):
        errs.append(f"var coverage: used {sorted(used)} vs n_vars {n}")
    if any(f["ftype"] == "given" and f["var"] == q for f in fs):
        errs.append("query var is a given (answer-stuffing)")
    sols, mode = solve(n, fs, m)
    if not sols:
        errs.append(f"UNSOLVABLE ({mode})")
    else:
        qvals = sorted({s[q] for s in sols})
        allv = [v for s in sols for v in s.values()]
        if any(not (0 <= v <= m) for v in allv):
            # values outside bounds in *some* solution branch
            inb = [s for s in sols
                   if all(0 <= v <= m for v in s.values())]
            if inb != sols:
                warns.append(f"{len(sols)-len(inb)} sol branches leave bounds")
                sols = inb; qvals = sorted({s[q] for s in sols})
        if len(qvals) != 1:
            errs.append(f"NON-UNIQUE query values {qvals[:6]} ({mode})")
        elif qvals[0] != d["answer"]:
            errs.append(f"WRONG: solved {qvals[0]} != answer {d['answer']}")
        else:
            warns.append(f"solved={qvals[0]} mode={mode} branches={len(sols)}")
    return errs, warns

total_ok = 0
for path in sys.argv[1:]:
    data = json.load(open(path))
    print(f"=== {path.split('/')[-1]}: {len(data['drafts'])} drafts, "
          f"{len(data.get('skips', []))} skips ===")
    for d in data["drafts"]:
        errs, warns = check(d)
        tag = "PASS" if not errs else "FAIL"
        if not errs:
            total_ok += 1
        print(f"[{tag}] src={d['src_idx']} ans={d['answer']} "
              f"n={d['n_vars']} f={len(d['factors'])}")
        for e in errs:
            print(f"       ERR: {e}")
        for w in warns:
            print(f"       note: {w}")
print(f"\n[pregate] {total_ok} rows pass mechanical gate")
