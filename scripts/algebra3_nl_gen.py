"""algebra3_nl_gen.py — TRANCHE 2 corpus (2026-07-10, integer-forced as
ratified): SEQUENCES (add/mul chains + ORDINAL mentions — "the fifth term" is
a mention phrase for an explicitly-enumerated term var; the solver walks the
chain, no pointer resolves constructed structure) + PCT (params ltype 10) +
FDIV/MOD compositions ("quotient and remainder" — two factors, one sentence;
tens-digit style). DEFERRED with formats pinned: ratio (twin-mul, hidden
product var) and abs (sel+add) — the hidden-variable gold species is
registered but unexercised until tranche 3 (closure table stays honest:
2 new ltypes / 3 categories shipped).
Gates: roundtrip via problem_from_algebra3, symmetry-aware (Vieta pairs from
the inherited v2 system), token budget, solution-first throughout.
USAGE: .venv/bin/python3 scripts/algebra3_nl_gen.py --selftest
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

from algebra_nl_gen import LETTERS  # noqa: E402
from algebra2_nl_gen import gen_system2, render2, SEL_TEMPLATES  # noqa: E402

ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
SEQ_STEP_T = ["The {oj} term is {d} more than the {oi} term.",
              "Each of these: the {oj} term exceeds the {oi} term by {d}."]
SEQ_STEP_MUL_T = ["The {oj} term is the {oi} term multiplied by {d}."]
SEQ_GIVEN_T = ["The {o} term is {val}.", "The {o} term equals {val}."]
PCT_T = ["{a} is {p} percent of {b}.", "{p} percent of {b} gives {a}.",
         "{a} equals {p} percent of {b}."]
FDIV_T = ["When {a} is divided by {k}, the quotient is {q} and the "
          "remainder is {r}.",
          "Dividing {a} by {k} gives quotient {q} and remainder {r}."]


def gen_system3(rng, m=60, teeth=0.0):
    n_vars, factors, sol, query, sym_pairs = gen_system2(
        rng, rng.randint(1, 2), rng.randint(0, 1), m,
        n_vieta=rng.randint(0, 1), n_crt=int(rng.random() < 0.3),
        irrelevant=rng.random() < teeth * 0.4)
    sol = list(sol)

    def new_var(v):
        sol.append(int(v))
        return len(sol) - 1

    extras = []          # (kind, payload) for render3
    query_pool = [query]
    # SEQUENCE (arith or geo), k terms, ordinal mentions
    if rng.random() < 0.8:
        k = rng.randint(3, 4)
        geo = rng.random() < 0.3
        if geo:
            t0, d = rng.randint(1, 4), rng.randint(2, 3)
            vals = [t0 * d ** i for i in range(k)]
        else:
            t0, d = rng.randint(0, 12), rng.randint(1, 9)
            vals = [t0 + d * i for i in range(k)]
        if max(vals) <= m:
            tv = [new_var(v) for v in vals]
            dv = new_var(d)
            factors.append({"ftype": "given", "var": dv, "value": d,
                            "role": "seq_d"})
            for i in range(k - 1):
                factors.append({"ftype": "rel",
                                "op": "mul" if geo else "add",
                                "args": [tv[i], dv], "result": tv[i + 1],
                                "surface": "mul" if geo else "add",
                                "seq": (i, i + 1)})
            gi = rng.randint(0, k - 1)
            factors.append({"ftype": "given", "var": tv[gi], "value": vals[gi],
                            "role": "seq_anchor", "seq_ord": gi})
            extras.append(("seq", {"terms": tv, "geo": geo}))
            query_pool += [tv[rng.randint(0, k - 1)]] * 2
    # PCT
    if rng.random() < 0.7:
        p = rng.choice([10, 20, 25, 50, 75, 150, 200])
        b = rng.choice([v for v in range(4, m + 1) if (p * v) % 100 == 0
                        and p * v <= 100 * m][:20] or [4])
        a = p * b // 100
        av, bv = new_var(a), new_var(b)
        factors.append({"ftype": "pct", "args": [av, bv], "p": p})
        factors.append({"ftype": "given", "var": bv, "value": b,
                        "role": "pct_base"})
        query_pool += [av, av]
    # FDIV+MOD composition
    if rng.random() < 0.7:
        kk = rng.randint(2, 9)
        a = rng.randint(kk, m)
        q, r = a // kk, a % kk
        avv, qv, rv = new_var(a), new_var(q), new_var(r)
        factors.append({"ftype": "fdiv", "var": avv, "k": kk, "result": qv,
                        "pair": True})
        factors.append({"ftype": "mod", "var": avv, "k": kk, "result": rv,
                        "pair": True})
        factors.append({"ftype": "given", "var": avv, "value": a,
                        "role": "fdiv_a"})
        query_pool += [qv, rv]
    return len(sol), factors, sol, rng.choice(query_pool), sym_pairs, extras


def render3(rng, n_vars, factors, query, extras, **teeth_kw):
    """render2 for inherited kinds; local unit synthesis for seq/pct/fdiv.
    Strategy: temporarily strip tranche-2 factors, render2 the rest, then
    append tranche-2 sentences with spans + ordinal mentions."""
    base = [f for f in factors if "seq" not in f and f["ftype"] not in
            ("pct", "fdiv") and not f.get("pair") and
            f.get("role") not in ("seq_d", "seq_anchor", "pct_base", "fdiv_a")]
    rest = [f for f in factors if f not in base]
    text, gf_base, mentions = render2(rng, n_vars, base, query, **teeth_kw)
    # strip the trailing query sentence; re-add after extras
    qs = text.rfind(". ") + 2
    body, qsent = text[:qs], text[qs:]
    letters = {}
    for v, spans in mentions.items():
        s, e = spans[0]
        letters[v] = text[s:e]

    def nm(v):
        return letters.get(v, LETTERS[v])
    seq_terms = {}
    for kind, pl in extras:
        if kind == "seq":
            for oi, tv in enumerate(pl["terms"]):
                seq_terms[tv] = oi
    units = []
    for f in rest:
        if "seq" in f:
            oi, oj = f["seq"]
            bank = SEQ_STEP_MUL_T if f["surface"] == "mul" else SEQ_STEP_T
            s = rng.choice(bank).format(oj=ORDINALS[oj], oi=ORDINALS[oi],
                                        d=nm(f["args"][1]))
        elif f.get("role") == "seq_anchor":
            s = rng.choice(SEQ_GIVEN_T).format(
                o=ORDINALS[f["seq_ord"]], val=f["value"])
        elif f["ftype"] == "pct":
            s = rng.choice(PCT_T).format(a=nm(f["args"][0]), p=f["p"],
                                         b=nm(f["args"][1]))
        elif f["ftype"] == "fdiv":
            mate = next(g for g in rest if g.get("pair")
                        and g["ftype"] == "mod" and g["var"] == f["var"])
            s = rng.choice(FDIV_T).format(a=nm(f["var"]), k=f["k"],
                                          q=nm(f["result"]),
                                          r=nm(mate["result"]))
            f["_mate"] = mate
        elif f["ftype"] == "mod" and f.get("pair"):
            continue                    # rendered with its fdiv mate
        else:                           # pct_base / fdiv_a givens
            from algebra_nl_gen import GIVEN_TEMPLATES
            s = rng.choice(GIVEN_TEMPLATES).format(v=nm(f["var"]),
                                                   val=f["value"])
        units.append((s, f))
    rng.shuffle(units)
    out_factors = list(gf_base)
    pos = len(body)
    parts = [body]
    for s, f in units:
        start = pos
        parts.append(s + " ")
        pos += len(s) + 1
        span = [[start, start + len(s)]]
        nf = dict(f, spans=span)
        nf.pop("_mate", None)
        nf.pop("pair", None)
        nf.pop("seq", None)
        out_factors.append(nf)
        if "_mate" in f:
            mm = dict(f["_mate"], spans=span)
            mm.pop("pair", None)
            out_factors.append(mm)
        for v, letter in letters.items():
            for mt in re.finditer(rf"\b{re.escape(letter)}\b", s):
                mentions.setdefault(v, []).append(
                    [start + mt.start(), start + mt.end()])
        for tv, oi in seq_terms.items():
            phrase = f"{ORDINALS[oi]} term"
            for mt in re.finditer(phrase, s):
                mentions.setdefault(tv, []).append(
                    [start + mt.start(), start + mt.end()])
    parts.append(qsent)
    full = "".join(parts)
    # query var may be a term var mentioned only by ordinal — rewrite qsent
    return full, out_factors, {int(k): v for k, v in mentions.items() if v}


def roundtrip3(n_vars, factors, m, solution, sym_pairs=()):
    from mycelium.csp_domains import problem_from_algebra3
    from mycelium.csp_core import solve_symbolic
    givens = {f["var"]: f["value"] for f in factors if f["ftype"] == "given"}
    sym_vars = {v for pr in sym_pairs for v in pr}
    prob = problem_from_algebra3(n_vars, factors, givens, m)
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
        p2 = problem_from_algebra3(n_vars, factors, givens, m)
        p2.domains0[v].discard(solution[v])
        if p2.domains0[v]:
            r2 = solve_symbolic(p2, budget=200_000, seed=0)
            if r2["status"] == "solved":
                return False, -1
    return True, int(res.get("decisions", -1))


def generate3(n, seed, out, m=60, teeth=0.0, token_budget=0):
    rng = random.Random(seed)
    tok = None
    if token_budget:
        from tokenizers import Tokenizer
        from phase1_algebra_head import TOKENIZER_JSON
        tok = Tokenizer.from_file(TOKENIZER_JSON)
    n_ok = n_rej = n_tok = 0
    bands, kinds = {}, {"seq": 0, "pct": 0, "fdiv": 0}
    with open(out, "w") as fh:
        while n_ok < n:
            n_vars, factors, sol, query, sym, extras = gen_system3(
                rng, m, teeth)
            if n_vars > 24 or len(factors) > 24:
                n_rej += 1
                continue
            try:
                text, gfactors, mentions = render3(
                    rng, n_vars, factors, query, extras,
                    shuffle_letters=(rng.random() < teeth * 0.5),
                    oblique_prob=teeth * 0.3, sel_oblique_prob=teeth * 0.3)
            except Exception:
                n_rej += 1
                continue
            if tok is not None and len(tok.encode(text).ids) > token_budget:
                n_tok += 1
                continue
            ok, decisions = roundtrip3(n_vars, gfactors, m, sol, sym)
            if not ok:
                n_rej += 1
                continue
            bands[decisions] = bands.get(decisions, 0) + 1
            kinds["seq"] += sum(1 for _, p in extras if _ == "seq")
            kinds["pct"] += sum(1 for f in gfactors if f["ftype"] == "pct")
            kinds["fdiv"] += sum(1 for f in gfactors if f["ftype"] == "fdiv")
            fh.write(json.dumps({
                "n_vars": n_vars, "m": m, "text": text, "factors": gfactors,
                "mentions": mentions, "query_var": query, "solution": sol,
                "decisions": decisions,
                "gen": {"seed": seed, "teeth": teeth,
                        "sym_pairs": [list(p) for p in sym]},
            }) + "\n")
            n_ok += 1
    print(f"[gen3] wrote {n_ok} to {out} ({n_rej} rejected, {n_tok} token)")
    print(f"[gen3] bands: {dict(sorted(bands.items()))} | kinds: {kinds}")


def selftest3():
    rng = random.Random(5)
    for _ in range(20):
        n_vars, factors, sol, q, sym, extras = gen_system3(rng, 60, 0.5)
        if n_vars > 24 or len(factors) > 24:
            continue
        text, gf, mentions = render3(rng, n_vars, factors, q, extras)
        ok, dec = roundtrip3(n_vars, gf, 60, sol, sym)
        assert ok, ("roundtrip3 failed", text[:200])
        assert all(f["spans"] for f in gf)
    print("[selftest3] OK —", text[:160], "...")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--out", default=".cache/algebra3_nl_smoke.jsonl")
    ap.add_argument("--teeth", type=float, default=0.0)
    ap.add_argument("--token-budget", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest3()
        return
    generate3(a.n, a.seed, a.out, teeth=a.teeth,
              token_budget=a.token_budget)


if __name__ == "__main__":
    main()
