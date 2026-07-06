"""algebra_nl_gen.py — ALGEBRA-IN-WORDS: the unknowns corpus generator (math
expansion step 2, 2026-07-07).

THE TRANSPLANT (spec §10): the KenKen generator's deepest property carried whole —
the domain makes gold + equivalence FREE. Systems are generated SOLUTION-FIRST
(sample values, derive constants -> satisfiable by construction), gated by
ban-and-resolve UNIQUENESS and the ROUND TRIP (gold factors alone -> problem_from_
algebra -> solve_symbolic -> exact match with the generating solution). Every
emitted sample carries its measured DECISIONS count — the engine-band label — so
the corpus is band-stratifiable by construction (the generator DIAL, measured per
sample, not assumed from knobs).

GOLD FORMAT (shaped by the §10 neural-format finding BEFORE any head exists):
factors carry ROLE-TYPED membership — {op, args: [ids] (unordered pair; sub/div are
CANONICALIZED into add/mul form at generation so args are genuinely commutative),
result: id} — plus given-value factors {var, value}. Spans are SETS (contiguity
never assumed, the standing law). Char spans, 0-indexed vars, letters in NL.

CANONICALIZATION (kills the role problem at the source): "a minus b equals c" is
EMITTED as narrative sub/div text but its GOLD factor is the add/mul form
(add(b, c) = a / mul(b, c) = a) — one result pointer + an unordered arg pair
suffices, and the symbolic bridge consumes add/mul only.

RECORD OUT (jsonl):
  { "n_vars", "m", "text",
    "factors": [ {"ftype": "rel", "op": "add"|"mul", "args": [i, j],
                  "result": k, "spans": [[s,e],..]}
               | {"ftype": "given", "var": i, "value": v, "spans": [[s,e],..]} ],
    "query_var", "solution": [..], "decisions": int (the band label),
    "gen": {knobs}, }

USAGE:
  Selftest:  .venv/bin/python3 scripts/algebra_nl_gen.py --selftest
  Generate:  .venv/bin/python3 scripts/algebra_nl_gen.py --n 2000 --seed 0 \
                 --out .cache/algebra_nl_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Rendered op phrasings. KEY: the SURFACE op (narrative) may be sub/div; the GOLD
# factor is always the canonical add/mul form.
REL_TEMPLATES = {
    "add": ["{a} plus {b} equals {r}.", "The sum of {a} and {b} is {r}.",
            "{r} is the total of {a} and {b}."],
    "sub": ["{r} minus {b} equals {a}.", "{r} exceeds {b} by {a}.",
            "The difference between {r} and {b} is {a}."],
    "mul": ["{a} times {b} equals {r}.", "The product of {a} and {b} is {r}."],
}
GIVEN_TEMPLATES = ["{v} equals {val}.", "The value of {v} is {val}.",
                   "{v} is {val}."]
QUERY_TEMPLATES = ["What is {q}?", "Find {q}.", "Determine the value of {q}."]
PREAMBLES = ["Consider the numbers {vars}.", "Let {vars} be whole numbers.",
             "The following facts hold about {vars}."]


def gen_system(rng: random.Random, n_pairs: int, chain_len: int, m: int):
    """Solution-first generation. Returns (n_vars, factors, solution, query_var).
    factors: role-typed gold (canonical add/mul + givens). Coupled pairs supply the
    engine band; a triangular chain supplies calculator-band structure on top."""
    sol, factors = [], []

    def new_var(v):
        sol.append(int(v))
        return len(sol) - 1

    unknowns = []
    for _ in range(n_pairs):
        # coupled pair: x+y = s, x-y = d  (d rendered as sub; gold = add(y, d) = x)
        y = rng.randint(0, m // 3)
        x = y + rng.randint(1, m // 3)          # x > y keeps d positive
        xi, yi = new_var(x), new_var(y)
        si, di = new_var(x + y), new_var(x - y)
        factors.append({"ftype": "rel", "op": "add", "args": [xi, yi],
                        "result": si, "surface": "add"})
        factors.append({"ftype": "rel", "op": "add", "args": [yi, di],
                        "result": xi, "surface": "sub"})
        factors.append({"ftype": "given", "var": si, "value": x + y})
        factors.append({"ftype": "given", "var": di, "value": x - y})
        unknowns += [xi, yi]

    if unknowns:
        prev = unknowns[0]
    else:
        prev = new_var(rng.randint(1, m // 4))
        factors.append({"ftype": "given", "var": prev, "value": sol[prev]})
    for _ in range(chain_len):
        # chain: prev (op) k = nxt, k given — pure calculator band
        if rng.random() < 0.5 and sol[prev] * 2 <= m:
            k = rng.randint(1, max(1, m // max(sol[prev], 1)))
            if sol[prev] * k > m:
                k = 1
            nxt = new_var(sol[prev] * k)
            ki = new_var(k)
            factors.append({"ftype": "rel", "op": "mul", "args": [prev, ki],
                            "result": nxt, "surface": "mul"})
        else:
            k = rng.randint(0, m - sol[prev])
            nxt = new_var(sol[prev] + k)
            ki = new_var(k)
            factors.append({"ftype": "rel", "op": "add", "args": [prev, ki],
                            "result": nxt, "surface": "add"})
        factors.append({"ftype": "given", "var": ki, "value": k})
        prev = nxt

    query = rng.choice(unknowns) if unknowns else prev
    return len(sol), factors, sol, query


def render(rng: random.Random, n_vars: int, factors: list, query: int,
           shuffle: bool = True):
    """NL rendering with span-SET gold. Returns (text, factors-with-spans)."""
    names = {i: LETTERS[i] for i in range(n_vars)}
    units = []   # (sentence, factor_index or None)
    used = sorted({v for f in factors for v in
                   (f["args"] + [f["result"]] if f["ftype"] == "rel" else [f["var"]])})
    pre = rng.choice(PREAMBLES).format(vars=", ".join(names[v] for v in used))
    for fi, f in enumerate(factors):
        if f["ftype"] == "given":
            s = rng.choice(GIVEN_TEMPLATES).format(v=names[f["var"]], val=f["value"])
        else:
            a, b = f["args"]
            r = f["result"]
            if f["surface"] == "sub":
                # gold add(b, d)=x rendered as "x minus b equals d"-family
                s = rng.choice(REL_TEMPLATES["sub"]).format(
                    r=names[r], b=names[a], a=names[b])
            else:
                s = rng.choice(REL_TEMPLATES[f["surface"]]).format(
                    a=names[a], b=names[b], r=names[r])
        units.append((s, fi))
    if shuffle:
        rng.shuffle(units)
    units = [(pre, None)] + units + [(rng.choice(QUERY_TEMPLATES).format(q=names[query]), None)]

    text_parts, pos = [], 0
    out_factors = [dict(f, spans=[]) for f in factors]
    mentions = {v: [] for v in used}
    for i, (s, fi) in enumerate(units):
        if i > 0:
            text_parts.append(" ")
            pos += 1
        start = pos
        text_parts.append(s)
        pos += len(s)
        if fi is not None:
            out_factors[fi]["spans"].append([start, pos])
        # VARIABLE-MENTION ANNOTATIONS (registered pre-emption of the referential-
        # binding failure mode the text-NACK arm proved for shallow layers): the
        # generator KNOWS every mention at render time — emit them as structure so
        # the result pointer never has to LEARN name->slot binding from one hop.
        # Names are single letters; word-boundary search inside the sentence is exact.
        for v, nm in names.items():
            for mt in re.finditer(rf"\b{nm}\b", s):
                mentions.setdefault(v, []).append([start + mt.start(), start + mt.end()])
    for f in out_factors:
        f.pop("surface", None)
    return "".join(text_parts), out_factors, {int(k): v for k, v in mentions.items() if v}


def roundtrip(n_vars: int, factors: list, m: int, solution: list):
    """Gold factors alone -> bridge -> solve -> exact match + uniqueness + decisions.
    Returns (ok, decisions)."""
    from mycelium.csp_domains import problem_from_algebra
    from mycelium.csp_core import solve_symbolic

    relations = [( f["op"], f["args"][0], f["args"][1], f["result"])
                 for f in factors if f["ftype"] == "rel"]
    givens = {f["var"]: f["value"] for f in factors if f["ftype"] == "given"}
    prob = problem_from_algebra(n_vars, relations, givens, m)
    res = solve_symbolic(prob, budget=200_000, seed=0)
    if res["status"] != "solved":
        return False, -1
    if [int(res["assignment"][v]) for v in range(n_vars)] != list(solution):
        return False, -1
    # uniqueness: ban-and-resolve on the unknowns
    for v in range(n_vars):
        if v in givens:
            continue
        p2 = problem_from_algebra(n_vars, relations, givens, m)
        p2.domains0[v].discard(solution[v])
        if p2.domains0[v]:
            r2 = solve_symbolic(p2, budget=200_000, seed=0)
            if r2["status"] == "solved":
                return False, -1
    return True, int(res.get("decisions", -1))


def generate(n: int, seed: int, out: str, m: int = 60,
             max_pairs: int = 3, max_chain: int = 3) -> None:
    rng = random.Random(seed)
    n_ok = n_rej = 0
    bands = {}
    with open(out, "w") as fh:
        while n_ok < n:
            n_pairs = rng.randint(1, max_pairs)
            chain = rng.randint(0, max_chain)
            n_vars, factors, sol, query = gen_system(rng, n_pairs, chain, m)
            if n_vars > 24:
                n_rej += 1
                continue
            text, gfactors, mentions = render(rng, n_vars, factors, query)
            ok, decisions = roundtrip(n_vars, gfactors, m, sol)
            if not ok:
                n_rej += 1
                continue
            bands[decisions] = bands.get(decisions, 0) + 1
            fh.write(json.dumps({
                "n_vars": n_vars, "m": m, "text": text, "factors": gfactors,
                "mentions": mentions,
                "query_var": query, "solution": sol, "decisions": decisions,
                "gen": {"seed": seed, "n_pairs": n_pairs, "chain": chain},
            }) + "\n")
            n_ok += 1
    print(f"[gen] wrote {n_ok} to {out} ({n_rej} rejected by round-trip/uniqueness)")
    print(f"[gen] decisions histogram (the band label): "
          f"{dict(sorted(bands.items()))}")


def selftest() -> None:
    rng = random.Random(7)
    n_vars, factors, sol, q = gen_system(rng, n_pairs=2, chain_len=1, m=60)
    ok, dec = roundtrip(n_vars, factors, 60, sol)
    assert ok, "hand system fails round trip"
    assert dec > 0, f"coupled system should be engine band, got {dec}"
    text, gf, mentions = render(rng, n_vars, factors, q)
    assert all(f["spans"] for f in gf), "every factor needs spans"
    used = {v for f in gf for v in
            (f["args"] + [f["result"]] if f["ftype"] == "rel" else [f["var"]])}
    assert all(v in mentions and mentions[v] for v in used), "every used var mentioned"
    for v, spans in mentions.items():
        for (ms, me) in spans:
            assert text[ms:me] == LETTERS[v], (v, text[ms:me])
    for f in gf:
        for (s, e) in f["spans"]:
            assert 0 <= s < e <= len(text)
    # determinism
    r1, r2 = random.Random(3), random.Random(3)
    a = gen_system(r1, 1, 1, 60)
    b = gen_system(r2, 1, 1, 60)
    assert a == b
    # canonicalization: sub surface -> gold add with result role
    subs = [f for f in factors if f["ftype"] == "rel"]
    assert all(f["op"] in ("add", "mul") for f in subs), "gold must be canonical"
    # chain-only system is calculator band
    n2, f2, s2, q2 = gen_system(random.Random(11), n_pairs=0, chain_len=3, m=60)
    ok2, dec2 = roundtrip(n2, f2, 60, s2)
    assert ok2 and dec2 == 0, f"chain-only should be 0 decisions, got {dec2}"
    print("[selftest] PASS (round trip, engine/calculator bands, canonical gold, spans)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--out", default=".cache/algebra_nl.jsonl")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
    else:
        generate(args.n, args.seed, args.out, m=args.m)


if __name__ == "__main__":
    main()
