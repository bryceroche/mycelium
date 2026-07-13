"""gen9b_booster.py — the [71]/[78] basin booster, minted at KNOT level
(2026-07-12; the knot-rehearsal matrix's first day on the job).

Shape: fdiv-tiny-chain — a given dividend, one fdiv/mod pair, 1-2 chain
relations on the quotient (the [71] 'a + a/k' family and neighbors).
SPEC: N distinct CANONICAL REDUNDANCY CLASSES (the canonical digest is
the mint-time ID; a booster of near-identical tiny chains would train
the DIAGRAM again — pigeonhole-dense regime). Every row checked
canonically DISJOINT from every test corpus — the generation-bump
disjointness gate's first live use.
"""
import json, random, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from algebra_dag7_gen import roundtrip7
from hash_audit_iso import canon, TESTS
import algebra3_nl_gen as G3

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 911
OUT = sys.argv[3] if len(sys.argv) > 3 else ".cache/gen9b_booster.jsonl"

test_digests = set()
for path in TESTS.values():
    for l in open(path):
        test_digests.add(canon(json.loads(l))[0])
print(f"[booster] test-corpus canonical digests: {len(test_digests)}")

from tokenizers import Tokenizer
from phase1_algebra_head import TOKENIZER_JSON
tok = Tokenizer.from_file(TOKENIZER_JSON)
rng = random.Random(SEED)

def gen_tiny(rng, m):
    sol, factors = [], []
    def nv(v):
        sol.append(int(v)); return len(sol) - 1
    k = rng.randint(2, 9)
    a = nv(rng.randint(k, m))
    factors.append({"ftype": "given", "var": a, "value": sol[a]})
    q, r = divmod(sol[a], k)
    qi, ri = nv(q), nv(r)
    factors.append({"ftype": "fdiv", "var": a, "k": k, "result": qi,
                    "pair": True})
    factors.append({"ftype": "mod", "var": a, "k": k, "result": ri,
                    "pair": True})
    for _ in range(rng.randint(1, 2)):     # chain on the quotient family
        u = rng.choice([a, qi, ri] + list(range(len(sol))))
        w = rng.choice([a, qi])
        if u == w:
            continue
        if rng.random() < 0.6 and sol[u] + sol[w] <= m:
            factors.append({"ftype": "rel", "op": "add", "args": [u, w],
                            "result": nv(sol[u] + sol[w]), "surface": "add"})
        elif 0 < sol[u] * sol[w] <= m:
            factors.append({"ftype": "rel", "op": "mul", "args": [u, w],
                            "result": nv(sol[u] * sol[w]), "surface": "mul"})
    return len(sol), factors, sol, rng.randrange(1, len(sol))

classes = set()
ok = rej = dup_knot = 0
with open(OUT, "w") as fh:
    while ok < N:
        m = 300 if rng.random() < 0.4 else 60
        n_vars, factors, sol, query = gen_tiny(rng, m)
        if n_vars > 24 or len(factors) > 24:
            rej += 1; continue
        # givens by gate
        givens = {f["var"] for f in factors if f["ftype"] == "given"}
        order = [v for v in range(n_vars) if v not in givens]
        rng.shuffle(order)
        okg = False
        for _ in range(n_vars + 1):
            facs = factors + [{"ftype": "given", "var": v, "value": sol[v]}
                              for v in givens
                              if not any(f.get("var") == v and f["ftype"] == "given"
                                         for f in factors)]
            okg, _ = roundtrip7(n_vars, facs, m, sol)
            if okg:
                break
            if not order:
                break
            givens.add(order.pop())
        if not okg:
            rej += 1; continue
        try:
            text, gf, mentions = G3.render3(
                rng, n_vars, facs, query, [],
                shuffle_letters=(rng.random() < 0.4), oblique_prob=0.25,
                sel_oblique_prob=0.25)
        except Exception:
            rej += 1; continue
        if len(tok.encode(text).ids) > 250:
            rej += 1; continue
        okg, dec = roundtrip7(n_vars, gf, m, sol)
        if not okg:
            rej += 1; continue
        row = {"n_vars": n_vars, "m": m, "text": text, "factors": gf,
               "mentions": mentions, "query_var": query, "solution": sol,
               "decisions": dec, "gen": {"seed": SEED, "shape": "fdiv-tiny",
                                         "generation": "9b"}}
        dg = canon(row)[0]
        if dg in classes:                 # KNOT-LEVEL distinctness
            dup_knot += 1; continue
        if dg in test_digests:            # the disjointness gate, live
            print(f"[booster] REJECTED test-isomorph at row {ok}")
            rej += 1; continue
        classes.add(dg)
        fh.write(json.dumps(row) + "\n")
        ok += 1
print(f"[booster] {ok} rows = {len(classes)} distinct knots "
      f"({dup_knot} knot-dups rejected, {rej} gate/render rejects, "
      f"0 test-isomorphs admitted)")
