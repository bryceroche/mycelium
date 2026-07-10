"""algebra_verbose_gen.py — the VERBOSE register (2026-07-10): same graphs,
two renders. The free lunch: paired (prose, dialect, graph) with gold at
every layer; size-confound pinned BY CONSTRUCTION (pairs share graphs).
Register = generator-verbose (narrative-mathematical), not yet real MATH prose.
"""
import json, random, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import algebra2_nl_gen as G2
from algebra2_nl_gen import gen_system2, roundtrip2

VERBOSE = {
 "PREAMBLES": ["In this problem we are working with the numbers {vars}.",
               "Suppose we have a collection of whole numbers called {vars}."],
 "GIVEN": ["If you look at {v}, you will find that its value is exactly {val}.",
           "It turns out that {v} works out to be {val} in the end.",
           "After checking carefully, {v} comes to a value of {val}."],
 "REL": {"add": ["When you put {a} together with {b}, the total you end up "
                 "with is {r}.",
                 "Combining what {a} holds with what {b} holds gives a "
                 "grand total of {r}.",
                 "If {a} and {b} are gathered into one pile, that pile "
                 "amounts to {r}."],
         "sub": ["If you start from {r} and take away everything {b} has, "
                 "what remains is {a}.",
                 "Removing the amount {b} from the amount {r} leaves you "
                 "holding {a}."],
         "mul": ["Taking {a} and scaling it up by a factor of {b} produces "
                 "{r}.",
                 "If {a} were repeated {b} separate times, all of it "
                 "together would come to {r}."]},
 "SEL": {"larger": ["Between {a} and {b}, whichever one turns out to be the "
                    "bigger of the two, that is what {x} equals."],
         "smaller": ["Between {a} and {b}, whichever one turns out to be "
                     "the smaller of the two, that is what {x} equals."],
         "even": ["One of {a} and {b} is an even number, and {x} is "
                  "whichever one that happens to be."],
         "odd": ["One of {a} and {b} is an odd number, and {x} is "
                 "whichever one that happens to be."]},
 "MOD": ["If you were to divide {a} into groups of {k}, the amount left "
         "over at the end would be {r}.",
         "Splitting {a} evenly by {k} leaves a remainder of {r} behind."],
 "QUERY": ["After all of this, what value does {q} end up holding?",
           "Given everything above, what must {q} be?"],
}

def render_register(rng, n_vars, factors, query, verbose):
    saved = {}
    if verbose:
        saved = {"PREAMBLES": G2.PREAMBLES, "GIVEN_TEMPLATES": G2.GIVEN_TEMPLATES,
                 "REL_TEMPLATES": G2.REL_TEMPLATES, "QUERY_TEMPLATES": G2.QUERY_TEMPLATES,
                 "SEL_TEMPLATES": dict(G2.SEL_TEMPLATES), "MOD_TEMPLATES": G2.MOD_TEMPLATES}
        G2.PREAMBLES = VERBOSE["PREAMBLES"]; G2.GIVEN_TEMPLATES = VERBOSE["GIVEN"]
        G2.REL_TEMPLATES = VERBOSE["REL"]; G2.QUERY_TEMPLATES = VERBOSE["QUERY"]
        G2.SEL_TEMPLATES.update(VERBOSE["SEL"]); G2.MOD_TEMPLATES = VERBOSE["MOD"]
    try:
        return G2.render2(rng, n_vars, factors, query)
    finally:
        for k, v in saved.items():
            setattr(G2, k, v)

def main(n, seed, out_prefix, budget=250):
    from tokenizers import Tokenizer
    from phase1_algebra_head import TOKENIZER_JSON
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    rng = random.Random(seed)
    fa = open(f"{out_prefix}_verbose.jsonl", "w")
    fb = open(f"{out_prefix}_terse.jsonl", "w")
    ok = rej = 0
    while ok < n:
        n_vars, factors, sol, q, sym = gen_system2(
            rng, rng.randint(1, 2), rng.randint(0, 1), 60,
            n_vieta=rng.randint(0, 1), n_crt=int(rng.random() < 0.3))
        if n_vars > 24 or len(factors) > 24:
            rej += 1; continue
        r1, r2 = random.Random(rng.random()), random.Random(rng.random())
        try:
            tv, gv_, mv, _ = render_register(r1, n_vars, [dict(f) for f in factors], q, True)
            tt, gt_, mt, _ = render_register(r2, n_vars, [dict(f) for f in factors], q, False)
        except Exception:
            rej += 1; continue
        if len(tok.encode(tv).ids) > budget or len(tok.encode(tt).ids) > budget:
            rej += 1; continue
        okg, dec = roundtrip2(n_vars, gv_, 60, sol, sym)
        if not okg:
            rej += 1; continue
        for fh, txt, gf, mn in ((fa, tv, gv_, mv), (fb, tt, gt_, mt)):
            fh.write(json.dumps({"n_vars": n_vars, "m": 60, "text": txt,
                "factors": gf, "mentions": mn, "query_var": q,
                "solution": sol, "decisions": dec,
                "gen": {"seed": seed, "paired": True}}) + "\n")
        ok += 1
    fa.close(); fb.close()
    print(f"[verbose-gen] {ok} pairs to {out_prefix}_{{verbose,terse}}.jsonl ({rej} rejected)")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
