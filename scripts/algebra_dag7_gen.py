"""algebra_dag7_gen.py — GEN-7: the receipts generation (2026-07-11).

The first generation chartered entirely by measured failures (book 1's
invoice, registered in docs/phase1_skeleton_spec.md):
  1. FDIV INTO THE DAG ROTATION — fdiv/mod mate pairs on existing vars
     (5 book items waited on it; alg4test's weakness caught in the wild,
     including stable-wrong votes).
  2. REPEATED-ARG MUL — a times a equals b (the [85] grammar gap:
     number+square problems).
  3. LADDER CHAINS — 8-12 givens + chained partial sums (the [56]
     19-var length wall; also [46]'s deep-chain surface coverage).
  4. COUPLED-LINEAR WIRING — k1*x+y=s1, x+k2*y=s2, x,y ungiven (the
     [72] burger system; det=k1*k2-1 != 0 by construction).
  5. SURFACE ROBUSTNESS rides on 1-4: render templates sampled uniformly
     over deeper/wider shapes than gen-6 reached.
Rendering via render3 (fdiv-aware, span/mention-correct); gate via
roundtrip3 (exact match + ban-and-resolve uniqueness). Kind tallies
logged — no silent caps.
"""
import json, random, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import algebra3_nl_gen as G3


def gen_dag7(rng, m, target=None):
    sol, factors = [], []

    def nv(v):
        sol.append(int(v)); return len(sol) - 1

    kinds = set()
    r0 = rng.random()
    if r0 < 0.18:                       # LADDER (receipt 3)
        k = rng.randint(8, 12)
        gs = [nv(rng.randint(2, min(25, m // 4))) for _ in range(k)]
        for v in gs:
            factors.append({"ftype": "given", "var": v, "value": sol[v]})
        acc = gs[0]
        for v in gs[1:]:
            s = sol[acc] + sol[v]
            if s > m:
                break
            ni = nv(s)
            factors.append({"ftype": "rel", "op": "add", "args": [acc, v],
                            "result": ni, "surface": "add"})
            acc = ni
        kinds.add("ladder")
    elif r0 < 0.33:                     # COUPLED-LINEAR (receipt 4)
        hi = min(40, m // 8)
        x, y = rng.randint(3, hi), rng.randint(3, hi)
        xi, yi = nv(x), nv(y)
        for (u, v, k) in ((xi, yi, rng.choice([2, 3])),
                          (yi, xi, rng.choice([2, 3]))):
            kv = nv(k)
            factors.append({"ftype": "given", "var": kv, "value": k})
            pi = nv(k * sol[u])
            factors.append({"ftype": "rel", "op": "mul", "args": [kv, u],
                            "result": pi, "surface": "mul"})
            si = nv(k * sol[u] + sol[v])
            factors.append({"ftype": "rel", "op": "add", "args": [pi, v],
                            "result": si, "surface": "add"})
            factors.append({"ftype": "given", "var": si, "value": sol[si]})
        kinds.add("coupled")
    else:
        for _ in range(rng.randint(2, 4)):      # seed pool
            nv(rng.randint(0, min(m // 4, 60)))

    for _ in range(rng.randint(3, 9)):          # sampled wiring (longer)
        r = rng.random()
        if r < 0.36 and len(sol) >= 2:          # forward rel -> new var
            if rng.random() < 0.15:             # REPEATED-ARG (receipt 2)
                cand = [i for i in range(len(sol))
                        if 2 <= sol[i] and sol[i] * sol[i] <= m]
                if cand:
                    a = rng.choice(cand)
                    factors.append({"ftype": "rel", "op": "mul",
                                    "args": [a, a],
                                    "result": nv(sol[a] * sol[a]),
                                    "surface": "mul"})
                    kinds.add("sq")
                continue
            a, b = rng.sample(range(len(sol)), 2)
            if rng.random() < 0.5 and sol[a] + sol[b] <= m:
                factors.append({"ftype": "rel", "op": "add", "args": [a, b],
                                "result": nv(sol[a] + sol[b]),
                                "surface": "add"})
            elif sol[a] * sol[b] <= m and sol[a] > 0 and sol[b] > 0:
                factors.append({"ftype": "rel", "op": "mul", "args": [a, b],
                                "result": nv(sol[a] * sol[b]),
                                "surface": "mul"})
        elif r < 0.58 and len(sol) >= 2:        # closure (gen-6's move)
            x, y = rng.sample(range(len(sol)), 2)
            if sol[x] > sol[y]:
                x, y = y, x
            k = nv(sol[y] - sol[x])
            factors.append({"ftype": "rel", "op": "add", "args": [x, k],
                            "result": y, "surface": rng.choice(["add", "sub"])})
        elif r < 0.70 and len(sol) >= 2:        # sel
            a, b = rng.sample(range(len(sol)), 2)
            if sol[a] != sol[b]:
                s = rng.choice(["larger", "smaller"])
                x = nv(max(sol[a], sol[b]) if s == "larger"
                       else min(sol[a], sol[b]))
                factors.append({"ftype": "sel", "sel": s, "args": [a, b],
                                "result": x})
        elif r < 0.92:                          # mod fact
            a = rng.randrange(len(sol))
            k = rng.randint(2, 9)
            factors.append({"ftype": "mod", "var": a, "k": k,
                            "result": nv(sol[a] % k)})
        else:                                   # FDIV pair (receipt 1)
            cand = [i for i in range(len(sol)) if sol[i] >= 2]
            if not cand:
                continue
            a = rng.choice(cand)
            k = rng.randint(2, 9)
            qi, ri = nv(sol[a] // k), nv(sol[a] % k)
            factors.append({"ftype": "fdiv", "var": a, "k": k,
                            "result": qi, "pair": True})
            factors.append({"ftype": "mod", "var": a, "k": k,
                            "result": ri, "pair": True})
            kinds.add("fdiv")
    if not any(f["ftype"] != "given" for f in factors):
        return None
    # quota early-reject BEFORE the expensive uniqueness gate
    if target is not None:
        have = kinds or {"plain"}
        if (target == "plain" and have != {"plain"}) or \
           (target != "plain" and target not in have):
            return None
    # GIVENS BY GATE: add until forced-unique everywhere
    order = list(range(len(sol))); rng.shuffle(order)
    givens = {f["var"] for f in factors if f["ftype"] == "given"}

    def facs():
        return factors + [{"ftype": "given", "var": v, "value": sol[v]}
                          for v in givens
                          if not any(f.get("var") == v and
                                     f["ftype"] == "given" for f in factors)]
    for _ in range(len(sol) + 1):
        ok, _dec = G3.roundtrip3(len(sol), facs(), m, sol, ())
        if ok:
            break
        for v in order:
            if v not in givens:
                givens.add(v); break
        else:
            return None
    else:
        return None
    return len(sol), facs(), sol, rng.randrange(len(sol)), kinds


def main(n, seed, out, budget=250):
    from tokenizers import Tokenizer
    from phase1_algebra_head import TOKENIZER_JSON
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    rng = random.Random(seed)
    ok = rej = 0
    tally = {"ladder": 0, "coupled": 0, "sq": 0, "fdiv": 0, "plain": 0}
    # DAG7_QUOTA: quota-balanced kinds — cycle target kinds, regenerate
    # until the item CONTAINS the target ('plain' = none of the four).
    # The v1 skew (fdiv in 60% of rows, ladder/coupled ~500 each) is the
    # registered defect this mode retires.
    quota = None
    if os.environ.get("DAG7_QUOTA"):
        quota = [t for spec in os.environ["DAG7_QUOTA"].split(",")
                 for (t, c) in [spec.split(":")]
                 for _ in range(int(c))]
        assert len(quota) == n, f"quota {len(quota)} != n {n}"
    with open(out, "w") as fh:
        while ok < n:
            m = 300 if rng.random() < 0.4 else 60
            g = gen_dag7(rng, m, target=(quota[ok] if quota else None))
            if g is None:
                rej += 1; continue
            n_vars, factors, sol, query, kinds = g
            if n_vars > 24 or len(factors) > 24:
                rej += 1; continue
            try:
                text, gf, mentions = G3.render3(
                    rng, n_vars, factors, query, [],
                    shuffle_letters=(rng.random() < 0.4), oblique_prob=0.25,
                    sel_oblique_prob=0.25)
            except Exception:
                rej += 1; continue
            if len(tok.encode(text).ids) > budget:
                rej += 1; continue
            okg, dec = G3.roundtrip3(n_vars, gf, m, sol, ())
            if not okg:
                rej += 1; continue
            for kk in (kinds or {"plain"}):
                tally[kk] = tally.get(kk, 0) + 1
            if not kinds:
                tally["plain"] += 1
            fh.write(json.dumps({"n_vars": n_vars, "m": m, "text": text,
                "factors": gf, "mentions": mentions, "query_var": query,
                "solution": sol, "decisions": dec,
                "gen": {"seed": seed, "shape": "dag7", "generation": 7}}) + "\n")
            ok += 1
    print(f"[dag7-gen] {ok} to {out} ({rej} rejected) | kinds {tally}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
