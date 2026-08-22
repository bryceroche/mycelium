"""skel_mint.py — THE SKELETON MINT (2026-08-22, word given): authentic
narrative surfaces from the house trove (the MATH-train harvest) filled
with verified compositional graphs. Extraction: math-run slots only;
THE NEUTRALITY SCREEN — a skeleton is lawful only if its math content
lives entirely in the slots (no operative math words in the frame, no
digits outside math runs, exactly ONE interrogative slot). Refill: comp-
mint machinery (same double gate: anchor + solve_forced uniqueness cert
+ text oracle on the filled expression). Eval surfaces (fixed wild-val
20 + held-out draws) are EXCLUDED from skeleton harvesting.
"""
import sys, json, re, random, glob
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import numpy as np
from mycelium.anchor_law import unanchored_givens
from repair_replace_swap import solve_forced
from comp_mint import (sample_expr, ok_node, build_graph, render, oracle,
                       shape_sig, val, LETTERS)

OPERATIVE = re.compile(
    r"\b(sum|product|difference|quotient|twice|double[d]?|half|halve|"
    r"triple|times|total|average|mean|median|ratio|percent|consecutive|"
    r"remainder|divid\w*|divis\w*|multiple[s]?|factor\w*|digit\w*|"
    r"even|odd|prime|square[ds]?|cube[ds]?|root[s]?|sequence|arithmetic|"
    r"geometric|more|less|fewer|greater|smallest|largest|least|greatest|"
    r"maximum|minimum|positive|negative|integer[s]?|solution[s]?|"
    r"absolute|floor|ceiling|round\w*)\b", re.I)
INTERROG = re.compile(r"\b(what|find|compute|determine|evaluate|solve)\b",
                      re.I)
MATHRUN = re.compile(r"\$[^$]+\$")

def harvest_skeletons(excl_texts):
    h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
    skels = {}
    for r in h:
        t = " ".join(r["problem"].split())
        if t in excl_texts: continue
        runs = list(MATHRUN.finditer(t))
        if not (2 <= len(runs) <= 5): continue
        frame = MATHRUN.sub("\x00", t)
        if re.search(r"\d", frame): continue          # digits outside math
        if OPERATIVE.search(frame): continue          # operative words
        parts = frame.split("\x00")
        # exactly one interrogative clause containing a slot
        qslots = []
        for i in range(len(runs)):
            clause = parts[i][-60:] + " " + parts[i + 1][:20]
            if INTERROG.search(parts[i][-60:]):
                qslots.append(i)
        if len(qslots) != 1: continue
        if len(frame.split()) < 4: continue
        key = frame
        if key not in skels:
            skels[key] = {"parts": parts, "n": len(runs), "q": qslots[0],
                          "example": t}
    return list(skels.values())

def gen_fill(rng, ndecl):
    """eval-mode fill: ndecl declared letters + 1 query expr; returns
    (decl_units, expr_unit, facs, q, ans, bindings, sig) or None."""
    Ls = rng.sample(LETTERS, ndecl)
    lets = [(L, rng.randint(1, 30)) for L in Ls]
    n = sample_expr(rng, rng.randint(1, 3), lets)
    def used_lets(m, acc):
        if m[0] == "var": acc.add((m[1], m[2]))
        elif m[0] in ("add", "sub", "mul"):
            used_lets(m[1], acc); used_lets(m[2], acc)
        elif m[0] in ("sq", "fr"): used_lets(m[1], acc)
        elif m[0] == "opa": used_lets(m[2], acc); used_lets(m[4], acc)
        return acc
    ul = sorted(used_lets(n, set()))
    if len(ul) != ndecl: return None      # every declared letter must appear
    if not ok_node(n): return None
    if n[0] in ("lit", "var"): return None
    expr = render(n)
    facs = []; vmap = {}; nv = [0]
    q = build_graph(n, facs, vmap, nv)
    if nv[0] > 22: return None
    ans = val(n)
    try:
        o = oracle(expr, {L: v for L, v in ul})
    except Exception:
        return None
    if o != ans: return None
    decls = [f"${L} = {v}$" for L, v in ul]
    return decls, f"${expr}$", facs, q, ans, shape_sig(n)

def main():
    rng = random.Random(11111)
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    wv_texts = set(" ".join(json.loads(l)["original"].split())
                   for l in open('.cache/g55_wildval.jsonl'))
    # held-out draw surfaces (seeds 99/299) excluded too
    _all = [json.loads(l) for f in sorted(glob.glob('.cache/book*_t*_batch*.jsonl'))
            for l in open(f) if l.strip()]
    _byid = {r["src_idx"]: r for r in _all}
    for l in open('.cache/book12_anchor_batch1.jsonl'):
        r = json.loads(l); _byid[r["src_idx"]] = r
    _sk = set(json.load(open('.cache/book12_anchor_skips.json')))
    _dd = [json.loads(l) for l in open('.cache/base_t7self_deeds.jsonl')]
    h = [json.loads(l) for l in open('.cache/math_harvest_v0.jsonl')]
    drafted = set(_byid) | _sk | set(r["src_idx"] for r in _dd)
    for seed in (99, 299):
        rg = np.random.default_rng(seed)
        for i in [i for i in rg.permutation(len(h)) if i not in drafted][:10]:
            wv_texts.add(" ".join(h[i]["problem"].split()))
    skels = harvest_skeletons(wv_texts)
    print(f"[skel] {len(skels)} lawful skeletons harvested", flush=True)
    if not skels: return
    out = []; seen = set(); shape_ct = {}
    per = max(4, target // max(len(skels), 1) + 1)
    for sk in skels:
        made = 0; tries = 0
        while made < per and tries < per * 25:
            tries += 1
            fill = gen_fill(rng, sk["n"] - 1)
            if fill is None: continue
            decls, qunit, facs, q, ans, sig = fill
            if shape_ct.get(sig, 0) >= 12: continue
            units = []
            di = 0
            for i in range(sk["n"]):
                if i == sk["q"]: units.append(qunit)
                else: units.append(decls[di]); di += 1
            text = sk["parts"][0]
            for i in range(sk["n"]):
                text += units[i] + sk["parts"][i + 1]
            text = " ".join(text.split())
            if text in seen: continue
            row = {"src_idx": 400000 + len(out),
                   "construction": "skel-mint-v1", "factors": facs,
                   "query": q, "answer": ans, "lane": "L0-mint-skel",
                   "original": text, "shape": sig}
            if unanchored_givens(row): continue
            a = solve_forced(facs, q, {"n_vars": 24, "m": 300})
            if a != ans: continue
            seen.add(text); shape_ct[sig] = shape_ct.get(sig, 0) + 1
            out.append(row); made += 1
        if len(out) >= target: break
    with open('.cache/mint_skel_v1.jsonl', 'w') as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"[skel] MINTED {len(out)} rows from {len(skels)} skeletons, "
          f"{len(shape_ct)} shapes", flush=True)

if __name__ == "__main__":
    main()
