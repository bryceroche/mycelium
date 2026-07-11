"""algebra_dag_gen.py — GEN-6: teach the moves, not more diagrams
(2026-07-11). Random DAG composition: wiring sampled, not architectures —
closure relations couple existing unknowns (unknown-first chains, the
garden's shape), givens chosen by the UNIQUENESS GATE (not by template),
query position sampled, forward references arise naturally. Value-range fix
rides along: givens minted to 300 on a slice (round-2's wall retired).
Kinds: add/mul/sel/mod (render2-compatible); the novelty is the WIRING.
"""
import json, random, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import algebra2_nl_gen as G2

def gen_dag(rng, m):
    sol, factors = [], []
    def nv(v):
        sol.append(int(v)); return len(sol) - 1
    for _ in range(rng.randint(2, 4)):          # seed pool
        nv(rng.randint(0, min(m // 4, 60)))
    for _ in range(rng.randint(3, 6)):          # sampled wiring
        r = rng.random()
        if r < 0.45 and len(sol) >= 2:          # forward rel -> new var
            a, b = rng.sample(range(len(sol)), 2)
            if rng.random() < 0.5 and sol[a] + sol[b] <= m:
                factors.append({"ftype": "rel", "op": "add", "args": [a, b],
                                "result": nv(sol[a] + sol[b]), "surface": "add"})
            elif sol[a] * sol[b] <= m and sol[a] > 0 and sol[b] > 0:
                factors.append({"ftype": "rel", "op": "mul", "args": [a, b],
                                "result": nv(sol[a] * sol[b]), "surface": "mul"})
        elif r < 0.75 and len(sol) >= 2:        # CLOSURE: couple existing vars
            x, y = rng.sample(range(len(sol)), 2)
            if sol[x] > sol[y]: x, y = y, x
            if sol[y] - sol[x] >= 0:            # add(x, k)=y, k NEW (unknown-first)
                k = nv(sol[y] - sol[x])
                factors.append({"ftype": "rel", "op": "add", "args": [x, k],
                                "result": y, "surface": rng.choice(["add", "sub"])})
        elif r < 0.9 and len(sol) >= 2:         # sel over an existing pair
            a, b = rng.sample(range(len(sol)), 2)
            if sol[a] != sol[b]:
                s = rng.choice(["larger", "smaller"])
                x = nv(max(sol[a], sol[b]) if s == "larger" else min(sol[a], sol[b]))
                factors.append({"ftype": "sel", "sel": s, "args": [a, b], "result": x})
        else:                                    # mod fact on an existing var
            a = rng.randrange(len(sol))
            k = rng.randint(2, 9)
            factors.append({"ftype": "mod", "var": a, "k": k,
                            "result": nv(sol[a] % k)})
    if not factors: return None
    # GIVENS BY GATE: start minimal, add until forced-unique everywhere
    order = list(range(len(sol))); rng.shuffle(order)
    givens = set()
    from mycelium.csp_domains import problem_from_algebra2
    from mycelium.csp_core import solve_symbolic
    def facs_with_givens():
        return factors + [{"ftype": "given", "var": v, "value": sol[v]}
                          for v in givens]
    for _ in range(len(sol) + 1):
        ok, _dec = G2.roundtrip2(len(sol), facs_with_givens(), m, sol, ())
        if ok: break
        for v in order:
            if v not in givens:
                givens.add(v); break
        else: return None
    else: return None
    return len(sol), facs_with_givens(), sol, rng.randrange(len(sol))

def main(n, seed, out, budget=250):
    from tokenizers import Tokenizer
    from phase1_algebra_head import TOKENIZER_JSON
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    rng = random.Random(seed)
    ok = rej = 0
    with open(out, "w") as fh:
        while ok < n:
            m = 300 if rng.random() < 0.4 else 60   # range fix on a slice
            g = gen_dag(rng, m)
            if g is None: rej += 1; continue
            n_vars, factors, sol, query = g
            if n_vars > 24 or len(factors) > 24: rej += 1; continue
            try:
                text, gf, mentions, _nm = G2.render2(
                    rng, n_vars, factors, query,
                    shuffle_letters=(rng.random() < 0.4), oblique_prob=0.25,
                    sel_oblique_prob=0.25)
            except Exception: rej += 1; continue
            if len(tok.encode(text).ids) > budget: rej += 1; continue
            okg, dec = G2.roundtrip2(n_vars, gf, m, sol, ())
            if not okg: rej += 1; continue
            fh.write(json.dumps({"n_vars": n_vars, "m": m, "text": text,
                "factors": gf, "mentions": mentions, "query_var": query,
                "solution": sol, "decisions": dec,
                "gen": {"seed": seed, "shape": "dag", "generation": 6}}) + "\n")
            ok += 1
    print(f"[dag-gen] {ok} to {out} ({rej} rejected)")

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
