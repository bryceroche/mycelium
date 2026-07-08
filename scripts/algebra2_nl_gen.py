"""algebra2_nl_gen.py — the TRANCHE corpus: Vieta quadratics + selectors + MOD
(2026-07-09, per the ratified charter).

WHAT'S NEW over algebra_nl_gen:
  VIETA PAIRS — two unknown roots with given sum and product (the integer-
    domain quadratic: x^2 - Sx + N factored). Without a selector the pair is
    SYMMETRIC (answer not forced — proven in the soundness gates); the
    SELECTOR factor breaks it: x = sel(a, b), sel in {larger, smaller, even,
    odd} (closed vocabulary — the <=32-way regime). The parser reads WHICH
    comparison; the solver applies it. Even/odd selectors are only emitted
    when exactly one root qualifies (ill-defined selectors are VIOLATED by
    predicate and would fail round-trip anyway — double-gated).
  MOD FACTS — remainder relations, two roles: derived (a mod k = r with a
    known — calculator band) and CRT (an otherwise-ungiven var pinned by two
    coprime moduli with k1*k2 > m — engine band by construction).
  SELECTOR VIEW-INVARIANCE — selector templates are whole sentences, so
    sentence-permutation views move the sentence as a unit; the referent
    cannot change (the TTA requirement, by construction).
  SELECTOR TEETH — oblique selector phrasings drawn at teeth*0.3.
  NO-REAL-ROOTS — moot by construction (solution-first: roots picked before
    S, N are derived); discriminant is always a perfect square here.
  Deferred from v2 (kept in the old corpus): the distractor tooth (numeric
  noise sentences) — trimmed to keep render2 auditable; revisit if the teeth
  battery needs it.

GATES (transplanted whole): round-trip via problem_from_algebra2 + exact
solution match + ban-and-resolve uniqueness on every unknown + band label.

USAGE: .venv/bin/python3 scripts/algebra2_nl_gen.py --n 2000 --seed 11 \
           --out .cache/algebra2_nl_train.jsonl --teeth 0.8
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
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from algebra_nl_gen import (  # noqa: E402
    LETTERS, GIVEN_TEMPLATES, REL_TEMPLATES, PREAMBLES, QUERY_TEMPLATES,
    OBLIQUE_FORMS, gen_system,
)

MOD_TEMPLATES = [
    "The remainder when {a} is divided by {k} is {r}.",
    "Dividing {a} by {k} leaves a remainder of {r}.",
    "{a} leaves remainder {r} when divided by {k}.",
    "When {a} is divided by {k}, the remainder is {r}.",
]
SEL_TEMPLATES = {
    "larger": ["{x} is the larger of {a} and {b}.",
               "Of {a} and {b}, the larger one is {x}.",
               "{x} equals whichever of {a} and {b} is greater."],
    "smaller": ["{x} is the smaller of {a} and {b}.",
                "Of {a} and {b}, the smaller one is {x}.",
                "{x} equals whichever of {a} and {b} is less."],
    "even": ["{x} is whichever of {a} and {b} is even.",
             "Of {a} and {b}, the even one is {x}."],
    "odd": ["{x} is whichever of {a} and {b} is odd.",
            "Of {a} and {b}, the odd one is {x}."],
}
SEL_OBLIQUE = {
    "larger": ["{x} is the one of {a} and {b} that is not the smaller.",
               "{x} is {a} or {b}, whichever exceeds the other."],
    "smaller": ["{x} is the one of {a} and {b} that is not the larger.",
                "{x} is {a} or {b}, whichever the other exceeds."],
    "even": ["{x} is the one of {a} and {b} that is not odd."],
    "odd": ["{x} is the one of {a} and {b} that is not even."],
}
COPRIME = [(3, 4), (3, 5), (4, 5), (5, 6), (4, 7), (5, 7), (6, 7), (7, 8),
           (5, 8), (7, 9), (8, 9), (5, 9)]


def gen_system2(rng, n_pairs, chain_len, m, n_vieta=1, n_crt=0,
                irrelevant=False):
    """Old system + Vieta pairs with selectors + mod facts. Solution-first."""
    n_vars, factors, sol, query = gen_system(rng, n_pairs, chain_len, m,
                                             irrelevant=irrelevant)
    sol = list(sol)

    def new_var(v):
        sol.append(int(v))
        return len(sol) - 1

    query_pool = [query]
    sym_pairs = []   # Vieta root pairs: symmetric BY DESIGN — (p,q)/(q,p) both
    # satisfy sum+product; the text cannot bind letters to roots; only the
    # SELECTED var is forced. The roundtrip gate treats these specially.
    for _ in range(n_vieta):
        sel = rng.choice(list(SEL_TEMPLATES))
        while True:
            p = rng.randint(1, 7)
            q = rng.randint(1, 7)
            if p == q:
                continue
            if sel in ("even", "odd") and (p % 2) == (q % 2):
                continue
            break
        ai, bi = new_var(p), new_var(q)
        si, ni = new_var(p + q), new_var(p * q)
        from mycelium.csp_domains import _sel_apply, SEL_TO_ID
        xi = new_var(_sel_apply(SEL_TO_ID[sel], p, q))
        factors += [
            {"ftype": "rel", "op": "add", "args": [ai, bi], "result": si,
             "surface": "add"},
            {"ftype": "rel", "op": "mul", "args": [ai, bi], "result": ni,
             "surface": "mul"},
            {"ftype": "given", "var": si, "value": p + q, "role": "vieta_sum"},
            {"ftype": "given", "var": ni, "value": p * q, "role": "vieta_prod"},
            {"ftype": "sel", "sel": sel, "args": [ai, bi], "result": xi},
        ]
        sym_pairs.append((ai, bi))
        query_pool += [xi, xi]          # selector results carry the point
    for _ in range(n_crt):
        # uniqueness over {0..m} REQUIRES lcm(k1,k2) = k1*k2 > m — otherwise
        # w + lcm also satisfies both remainders and ban-and-resolve rejects.
        pairs = [(a_, b_) for (a_, b_) in COPRIME if a_ * b_ > m]
        if not pairs:
            break
        k1, k2 = rng.choice(pairs)
        w = rng.randint(0, m)
        wi = new_var(w)
        r1i, r2i = new_var(w % k1), new_var(w % k2)
        factors += [
            {"ftype": "mod", "var": wi, "k": k1, "result": r1i},
            {"ftype": "mod", "var": wi, "k": k2, "result": r2i},
            {"ftype": "given", "var": r1i, "value": w % k1, "role": "crt_r"},
            {"ftype": "given", "var": r2i, "value": w % k2, "role": "crt_r"},
        ]
        query_pool += [wi, wi]
    # derived mod fact on an existing var (calculator band) — Vieta roots
    # excluded (a mod given on a root would silently break the pair symmetry;
    # keep the teeth orthogonal)
    sym_vars = {v for pr in sym_pairs for v in pr}
    if rng.random() < 0.5:
        pool = [v for v in range(len(sol))
                if v not in sym_vars and not any(
                    f.get("var") == v and f["ftype"] == "given"
                    for f in factors)]
        base = rng.choice(pool or [query])
        k = rng.randint(2, 9)
        ri = new_var(sol[base] % k)
        factors.append({"ftype": "mod", "var": base, "k": k, "result": ri})
        query_pool.append(ri)
    return len(sol), factors, sol, rng.choice(query_pool), sym_pairs


def render2(rng, n_vars, factors, query, shuffle=True, shuffle_letters=False,
            oblique_prob=0.0, sel_oblique_prob=0.0):
    """algebra_nl_gen.render extended with MOD/SEL units. Same span-SET gold,
    same mention discipline; distractor tooth deferred (header)."""
    letters = list(LETTERS[:max(n_vars, 1)])
    if shuffle_letters:
        rng.shuffle(letters)
    names = {i: letters[i] for i in range(n_vars)}
    used = sorted({v for f in factors for v in
                   (f["args"] + [f["result"]] if f["ftype"] in ("rel", "sel")
                    else ([f["var"], f["result"]] if f["ftype"] == "mod"
                          else [f["var"]]))})
    pre = rng.choice(PREAMBLES).format(vars=", ".join(names[v] for v in used))
    units = []
    for fi, f in enumerate(factors):
        if f["ftype"] == "given":
            s = rng.choice(GIVEN_TEMPLATES).format(v=names[f["var"]],
                                                   val=f["value"])
        elif f["ftype"] == "rel":
            a, b = f["args"]
            r = f["result"]
            if f["surface"] == "sub":
                s = rng.choice(REL_TEMPLATES["sub"]).format(
                    r=names[r], b=names[a], a=names[b])
            else:
                s = rng.choice(REL_TEMPLATES[f["surface"]]).format(
                    a=names[a], b=names[b], r=names[r])
        elif f["ftype"] == "mod":
            s = rng.choice(MOD_TEMPLATES).format(
                a=names[f["var"]], k=f["k"], r=names[f["result"]])
        elif f["ftype"] == "sel":
            bank = (SEL_OBLIQUE if rng.random() < sel_oblique_prob
                    else SEL_TEMPLATES)[f["sel"]]
            s = rng.choice(bank).format(x=names[f["result"]],
                                        a=names[f["args"][0]],
                                        b=names[f["args"][1]])
        units.append((s, fi))
    # oblique variable references (the old tooth, single-letter -> ordinal)
    oblique_marks = []
    if oblique_prob > 0:
        for v in used:
            if v < len(OBLIQUE_FORMS) and rng.random() < oblique_prob:
                phrase = OBLIQUE_FORMS[v]
                nm = names[v]
                units = [(re.sub(rf"\b{nm}\b", phrase, s, count=1)
                          if rng.random() < 0.5 else s, fi)
                         for (s, fi) in units]
                oblique_marks.append((v, phrase))
    if shuffle:
        rng.shuffle(units)
    units = ([(pre, None)] + units +
             [(rng.choice(QUERY_TEMPLATES).format(q=names[query]), None)])
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
        for v, nm in names.items():
            for mt in re.finditer(rf"\b{nm}\b", s):
                mentions.setdefault(v, []).append(
                    [start + mt.start(), start + mt.end()])
        for v, phrase in oblique_marks:
            for mt in re.finditer(re.escape(phrase), s):
                mentions.setdefault(v, []).append(
                    [start + mt.start(), start + mt.end()])
    for f in out_factors:
        f.pop("surface", None)
    return ("".join(text_parts), out_factors,
            {int(k): v for k, v in mentions.items() if v})


def roundtrip2(n_vars, factors, m, solution, sym_pairs=()):
    """Round-trip gate, symmetry-aware: exact match + ban-and-resolve
    uniqueness on every unknown EXCEPT Vieta root pairs, which are symmetric
    by design — for those the gate requires MULTISET match (the unordered
    pair {p,q} is mathematically unique: roots of x^2 - Sx + N)."""
    from mycelium.csp_domains import problem_from_algebra2
    from mycelium.csp_core import solve_symbolic
    givens = {f["var"]: f["value"] for f in factors if f["ftype"] == "given"}
    sym_vars = {v for pr in sym_pairs for v in pr}
    prob = problem_from_algebra2(n_vars, factors, givens, m)
    res = solve_symbolic(prob, budget=200_000, seed=0)
    if res["status"] != "solved":
        return False, -1
    got = [int(res["assignment"][v]) for v in range(n_vars)]
    for v in range(n_vars):
        if v not in sym_vars and got[v] != solution[v]:
            return False, -1
    for (ai, bi) in sym_pairs:
        if {got[ai], got[bi]} != {solution[ai], solution[bi]}:
            return False, -1
    for v in range(n_vars):
        if v in givens or v in sym_vars:
            continue
        p2 = problem_from_algebra2(n_vars, factors, givens, m)
        p2.domains0[v].discard(solution[v])
        if p2.domains0[v]:
            r2 = solve_symbolic(p2, budget=200_000, seed=0)
            if r2["status"] == "solved":
                return False, -1
    return True, int(res.get("decisions", -1))


def generate2(n, seed, out, m=60, max_pairs=2, max_chain=2, teeth=0.0,
              token_budget=0):
    rng = random.Random(seed)
    tok = None
    if token_budget:
        from tokenizers import Tokenizer
        from phase1_algebra_head import TOKENIZER_JSON
        tok = Tokenizer.from_file(TOKENIZER_JSON)
    n_ok = n_rej = n_tok = 0
    bands, kinds = {}, {"sel": 0, "mod": 0, "crt": 0}
    with open(out, "w") as fh:
        while n_ok < n:
            n_pairs = rng.randint(1, max_pairs)
            chain = rng.randint(0, max_chain)
            n_vieta = rng.randint(0, 2) or 1     # bias toward >=1 Vieta
            n_crt = int(rng.random() < 0.4)
            irrel = rng.random() < teeth * 0.5
            n_vars, factors, sol, query, sym_pairs = gen_system2(
                rng, n_pairs, chain, m, n_vieta=n_vieta, n_crt=n_crt,
                irrelevant=irrel)
            # the head's geometry: K_VARS = L_FAC = 24 slots
            if n_vars > 24 or len(factors) > 24:
                n_rej += 1
                continue
            text, gfactors, mentions = render2(
                rng, n_vars, factors, query,
                shuffle_letters=(rng.random() < teeth * 0.5),
                oblique_prob=teeth * 0.35, sel_oblique_prob=teeth * 0.3)
            if tok is not None and len(tok.encode(text).ids) > token_budget:
                n_tok += 1
                continue
            ok, decisions = roundtrip2(n_vars, gfactors, m, sol, sym_pairs)
            if not ok:
                n_rej += 1
                continue
            bands[decisions] = bands.get(decisions, 0) + 1
            kinds["sel"] += sum(1 for f in gfactors if f["ftype"] == "sel")
            kinds["mod"] += sum(1 for f in gfactors if f["ftype"] == "mod")
            kinds["crt"] += n_crt
            fh.write(json.dumps({
                "n_vars": n_vars, "m": m, "text": text, "factors": gfactors,
                "mentions": mentions, "query_var": query, "solution": sol,
                "decisions": decisions,
                "gen": {"seed": seed, "n_pairs": n_pairs, "chain": chain,
                        "n_vieta": n_vieta, "n_crt": n_crt, "teeth": teeth,
                        "irrelevant": irrel,
                        "sym_pairs": [list(pr) for pr in sym_pairs]},
            }) + "\n")
            n_ok += 1
    print(f"[gen2] wrote {n_ok} to {out} ({n_rej} rejected by gates, "
          f"{n_tok} by token budget)")
    print(f"[gen2] bands: {dict(sorted(bands.items()))} | factor kinds: {kinds}")


def selftest2():
    rng = random.Random(7)
    n_vars, factors, sol, q, sym = gen_system2(rng, 1, 1, 60, n_vieta=1,
                                               n_crt=1)
    ok, dec = roundtrip2(n_vars, factors, 60, sol, sym)
    assert ok, "v2 hand system fails round trip"
    text, gf, mentions = render2(rng, n_vars, factors, q, sel_oblique_prob=1.0)
    assert all(f["spans"] for f in gf), "every factor needs spans"
    assert any(f["ftype"] == "sel" for f in gf)
    assert any(f["ftype"] == "mod" for f in gf)
    for f in gf:
        for (s, e) in f["spans"]:
            assert 0 <= s < e <= len(text)
    # selector must be FORCED (unique at the selected var) — the ratified core
    sel_f = next(f for f in gf if f["ftype"] == "sel")
    assert sol[sel_f["result"]] in (sol[sel_f["args"][0]],
                                    sol[sel_f["args"][1]])
    print("[selftest2] OK —", text[:120], "...")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=".cache/algebra2_nl_smoke.jsonl")
    ap.add_argument("--teeth", type=float, default=0.0)
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--token-budget", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest2()
        return
    generate2(a.n, a.seed, a.out, m=a.m, teeth=a.teeth,
              token_budget=a.token_budget)


if __name__ == "__main__":
    main()
