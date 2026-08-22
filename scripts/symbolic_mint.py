"""symbolic_mint.py — THE SYMBOLIC MINT (2026-08-22, word given): the
register gap's n-lever. Ten families mimicking the MEASURED frontier
surface (t7/t8's shapes: systems, compositions, evaluations, sqrt-as-
square, custom operators, Vieta, pct, sequences), solution-first,
values/intermediates <=300 by construction, gold graphs in the head's
grammar (OP_APPLY/rel/pct), one-fdiv-free. EVERY row passes the sitting
gate: anchor organ + expand -> solve_symbolic -> key. src_idx = 100000+i
(synthetic namespace, never collides with harvest).
"""
import sys, json, random
sys.path.insert(0, '.')
from mycelium.macros import expand_graph
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic
from mycelium.anchor_law import unanchored_givens

def G(v, x): return {"ftype": "given", "var": v, "value": x}
def R(op, a, b, r): return {"ftype": "rel", "op": op, "args": [a, b], "result": r}
def OPA(op, k1, x, k2, y, r): return {"ftype": "macro", "name": "OP_APPLY",
                                      "op": op, "k1": k1, "x": x, "k2": k2,
                                      "y": y, "result": r}
def PCT(a, b, p): return {"ftype": "pct", "args": [a, b], "p": p}

LETTERS = [("x", "y"), ("a", "b"), ("p", "q"), ("m", "n"), ("r", "s")]
Q = ["what is the value of ${v}$?", "find ${v}$.", "what is ${v}$?"]

def fam_sumdiff(rng):
    r1 = rng.randint(2, 90); r2 = rng.randint(1, r1 - 1)
    A, B = r1 - r2, r1 + r2
    u, v = rng.choice(LETTERS)
    qv, qi = rng.choice([(u, 2), (v, 3)])
    t = f"If ${u} - {v} = {A}$ and ${u} + {v} = {B}$, " + rng.choice(Q).format(v=qv)
    return t, [G(0, A), G(1, B), R("sub", 2, 3, 0), R("add", 2, 3, 1)], qi, (r1 if qi == 2 else r2)

def fam_coeff(rng):
    k1, k2 = rng.randint(2, 5), rng.randint(2, 5)
    X, Y = rng.randint(1, 20), rng.randint(1, 20)
    C, D = k1 * X + Y, X + k2 * Y
    if max(C, D) > 300: return None
    u, v = rng.choice(LETTERS)
    qv, qi = rng.choice([(u, 2), (v, 3)])
    t = (f"If ${k1}{u} + {v} = {C}$ and ${u} + {k2}{v} = {D}$, "
         + rng.choice(Q).format(v=qv))
    return t, [G(0, C), G(1, D), OPA("add", k1, 2, 1, 3, 0),
               OPA("add", 1, 2, k2, 3, 1)], qi, (X if qi == 2 else Y)

def fam_eval(rng):
    N, M = rng.randint(1, 30), rng.randint(1, 30)
    K, L = rng.randint(2, 6), rng.randint(2, 6)
    if K * N + L * M > 300: return None
    u, v = rng.choice(LETTERS)
    t = (f"If ${u} = {N}$ and ${v} = {M}$, what is the value of "
         f"${K}{u} + {L}{v}$?")
    return t, [G(0, N), G(1, M), OPA("add", K, 0, L, 1, 2)], 2, K * N + L * M

def fam_evalsq(rng):
    N, C = rng.randint(2, 16), rng.randint(1, 40)
    if N * N + C > 300: return None
    u, _ = rng.choice(LETTERS)
    t = f"If ${u} = {N}$, what is the value of ${u}^2 + {C}$?"
    return t, [G(0, N), G(1, C), R("mul", 0, 0, 2), R("add", 2, 1, 3)], 3, N * N + C

def fam_compose(rng):
    K, C, M, D = (rng.randint(2, 5), rng.randint(1, 9),
                  rng.randint(2, 5), rng.randint(1, 9))
    N = rng.randint(1, 9)
    inner = M * N + D; ans = K * inner + C
    if ans > 300: return None
    t = (f"Let $f(x) = {K}x + {C}$ and $g(x) = {M}x + {D}$. "
         f"What is the value of $f(g({N}))$?")
    return t, [G(0, N), G(1, M), G(2, D), G(3, K), G(4, C),
               R("mul", 1, 0, 5), R("add", 5, 2, 6), R("mul", 3, 6, 7),
               R("add", 7, 4, 8)], 8, ans

def fam_sqrt(rng):
    B = rng.randint(3, 17); A = rng.randint(1, min(40, B * B - 1))
    ansx = B * B - A
    u, _ = rng.choice(LETTERS)
    t = f"If $\\sqrt{{{u} + {A}}} = {B}$, what is the value of ${u}$?"
    return t, [G(0, A), G(1, B), R("add", 2, 0, 3), R("mul", 1, 1, 3)], 2, ansx

def fam_custom(rng):
    K, L = rng.randint(2, 6), rng.randint(2, 6)
    P, Qv = rng.randint(1, 20), rng.randint(1, 20)
    op = rng.choice(["add", "sub"])
    ans = K * P + L * Qv if op == "add" else K * P - L * Qv
    if not (0 <= ans <= 300): return None
    sym = rng.choice(["\\star", "\\oplus", "\\S", "Z"])
    sgn = "+" if op == "add" else "-"
    t = (f"Define $a {sym} b = {K}a {sgn} {L}b$. What is the value of "
         f"${P} {sym} {Qv}$?")
    return t, [G(0, P), G(1, Qv), OPA(op, K, 0, L, 1, 2)], 2, ans

def fam_vieta(rng):
    r1 = rng.randint(2, 12); r2 = rng.randint(1, r1 - 1)
    S, P = r1 + r2, r1 * r2
    ans = r1 * r1 + r2 * r2
    if ans > 300: return None
    u, v = rng.choice(LETTERS)
    t = (f"If ${u} + {v} = {S}$ and ${u}{v} = {P}$, what is the value of "
         f"${u}^2 + {v}^2$?")
    return t, [G(0, S), G(1, P), R("add", 2, 3, 0), R("mul", 2, 3, 1),
               R("mul", 2, 2, 4), R("mul", 3, 3, 5), R("add", 4, 5, 6)], 6, ans

def fam_pct(rng):
    p = rng.choice([10, 20, 25, 40, 50, 60, 75, 80])
    if rng.random() < 0.5:
        N = rng.choice([20, 40, 50, 60, 80, 100, 120, 150, 200, 240, 300])
        ans = p * N // 100
        if p * N % 100: return None
        t = f"What is ${p}\\%$ of ${N}$?"
        return t, [G(0, N), PCT(1, 0, p)], 1, ans
    V = rng.randint(2, 60)
    if (V * 100) % p: return None
    N = V * 100 // p
    if N > 300: return None
    t = f"If ${p}\\%$ of a number is ${V}$, what is the number?"
    return t, [G(0, V), PCT(0, 1, p)], 1, N

def fam_seq(rng):
    a = rng.randint(1, 40); d = rng.randint(2, 15)
    B, C = a + d, a + 2 * d
    ans = a + 3 * d
    if ans > 300: return None
    t = (f"The sequence ${a}, {B}, {C}, \\ldots$ is arithmetic. "
         f"What is the next term?")
    return t, [G(0, a), G(1, B), G(2, C), R("sub", 1, 0, 3),
               R("add", 2, 3, 4)], 4, ans

def fam_threeeval(rng):
    N, M = rng.randint(2, 12), rng.randint(1, 12)
    K = rng.randint(2, 4)
    ans = K * N * N - M
    if not (0 <= ans <= 300): return None
    u, v = rng.choice(LETTERS)
    t = (f"If ${u} = {N}$ and ${v} = {M}$, what is the value of "
         f"${K}{u}^2 - {v}$?")
    return t, [G(0, N), G(1, M), R("mul", 0, 0, 2),
               OPA("sub", K, 2, 1, 1, 3)], 3, ans

FAMS = [fam_sumdiff, fam_coeff, fam_eval, fam_evalsq, fam_compose,
        fam_sqrt, fam_custom, fam_vieta, fam_pct, fam_seq, fam_threeeval]

def main():
    rng = random.Random(4242)
    seen = set(); out = []; rej = {"dup": 0, "gate": 0, "none": 0}
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    per = target // len(FAMS) + 1
    for fam in FAMS:
        made = 0; tries = 0
        while made < per and tries < per * 30:
            tries += 1
            r = fam(rng)
            if r is None: rej["none"] += 1; continue
            t, facs, q, ans = r
            if t in seen: rej["dup"] += 1; continue
            row = {"src_idx": 100000 + len(out), "construction": "symbolic-mint-v1",
                   "factors": facs, "query": q, "answer": ans,
                   "lane": "L0-mint", "original": t}
            if unanchored_givens(row): rej["gate"] += 1; continue
            try:
                prim, nv = expand_graph(facs, 24)
                assert sum(1 for f in prim if f["ftype"] == "fdiv") <= 1
                gv = {f["var"]: f["value"] for f in prim if f["ftype"] == "given"}
                res = solve_symbolic(problem_from_algebra3(max(nv, 24), prim, gv, 300),
                                     budget=300_000, seed=0)
                assert res["status"] == "solved"
                assert int(res["assignment"][q]) == ans
            except Exception:
                rej["gate"] += 1; continue
            seen.add(t); out.append(row); made += 1
        print(f"[mint] {fam.__name__}: {made}", flush=True)
    with open('.cache/mint_symbolic_v1.jsonl', 'w') as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"[mint] MINTED {len(out)} rows (rejects {rej}) -> .cache/mint_symbolic_v1.jsonl")

if __name__ == "__main__":
    main()
