import json, sys, string
sys.path.insert(0, '/home/bryce/mycelium')
sys.path.insert(0, '/home/bryce/mycelium/scripts')
from tta_alg2_dials import solve2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic

SMP = {"n_vars": 24, "m": 300}
LETTERS = string.ascii_lowercase

CANDS = {c["src_idx"]: c for c in
         json.load(open('/home/bryce/mycelium/.cache/book8_candidates_t3.json'))["tranche3"]}


class G:
    """Small helper: issues consecutive var indices (0..n-1) as quantities are
    introduced, so 'letter position = var index' holds by construction."""
    def __init__(self):
        self.factors = []
        self.n = 0

    def given(self, value):
        v = self.n; self.n += 1
        self.factors.append({"ftype": "given", "var": v, "value": int(value)})
        return v

    def free(self):
        v = self.n; self.n += 1
        return v

    def rel(self, op, a, b, result=None):
        if result is None:
            result = self.n; self.n += 1
        self.factors.append({"ftype": "rel", "op": op, "args": [a, b], "result": result})
        return result

    def fdiv(self, a, k, result=None):
        if result is None:
            result = self.n; self.n += 1
        self.factors.append({"ftype": "fdiv", "var": a, "k": k, "result": result})
        return result


def L(v):
    return LETTERS[v]


def full_solution(factors, n_vars=24, m=300):
    gv = {f["var"]: f["value"] for f in factors if f["ftype"] == "given"}
    prob = problem_from_algebra3(n_vars, factors, gv, m)
    res = solve_symbolic(prob, budget=200_000, seed=0)
    if res["status"] != "solved":
        return None
    return [int(res["assignment"][v]) for v in range(n_vars)]


def check(name, factors, query_var, expect, n_vars=24, m=300):
    ans = solve2(factors, query_var, {"n_vars": n_vars, "m": m})
    ok = (ans == expect)
    print(f"[{name}] solve2={ans} expect={expect} {'OK' if ok else 'FAIL'}")
    return ok


def build_row(src_idx, factors, query_var, dialect, notes, watch=None, n_vars=24, m=300):
    sol = full_solution(factors, n_vars, m)
    assert sol is not None, f"src {src_idx}: solver failed to find full solution"
    gen = {
        "src_idx": src_idx, "book": 8, "tranche": 3, "floor": "prime", "fs": True,
        "dialect": dialect, "gate": "PENDING:5view-vote+key", "generation": "21",
        "notes": notes,
    }
    if watch:
        gen["watch"] = watch
    src_text = CANDS[src_idx]["problem"]
    return {
        "text": src_text, "factors": factors, "query_var": query_var,
        "n_vars": n_vars, "m": m, "decisions": [], "mentions": [],
        "solution": sol, "gen": gen,
    }


rows = []
fails = []
skips = []


def add(src_idx, factors, query_var, expect, dialect, notes, watch=None):
    ok = check(src_idx, factors, query_var, expect)
    if not ok:
        fails.append(src_idx)
        return
    rows.append(build_row(src_idx, factors, query_var, dialect, notes, watch))


def skip(src_idx, reason):
    skips.append((src_idx, reason))
    print(f"[{src_idx}] SKIP: {reason[:70]}...")


# ===========================================================================
# 489: 0.03n + 0.08(20+n) = 12.6  -> n=100
skip(489,
     "Decimal-coefficient linear equation. Clearing the two hundredths-place "
     "coefficients (0.03, 0.08) forces a x100 scale (gcd(3,8)=1, both are "
     "literally hundredths, so x100 is the unique integer-clearing factor); "
     "this drives the RHS to 1260 and the combined-term magnitudes to 1100+, "
     "both exceeding the 300 cap. Checked alternate partial scalings (x25, "
     "x4): neither clears both decimals to integer coefficients "
     "simultaneously. Multiplicative inversion (11*n=1100) also blocked: "
     "1100 exceeds the cap, and fdiv's k must be single-digit (11 isn't). "
     "No path renders this within the value cap; same magnitude-cap family "
     "as tranche2's [431]/[544]-class skips (see 544 below).")

# 491: 10 of 15 -> ratio; Berkeley 24 students -> A's = 16
g = G()
a = g.given(10)
b = g.given(15)
c = g.given(24)
d = g.free()
e = g.rel("mul", a, c)        # a*c = e  (10*24=240)
g.rel("mul", b, d, e)         # b*d = e  (15*d=240 -> d=16)
add(491, g.factors, d, 16,
    f"Consider the numbers {L(a)}, {L(b)}, {L(c)}, {L(d)}, {L(e)}. "
    f"{L(a)} is 10. {L(b)} is 15. {L(c)} is 24. "
    f"{L(a)} times {L(c)} equals {L(e)}. {L(b)} times {L(d)} equals {L(e)}. "
    f"What is {L(d)}?",
    "multiplicative inversion (Worked Example C): the ratio 10:15 cross-"
    "multiplied against the target 24 gives 10*24=15*d; the solver finds "
    "d=16 by search rather than dividing.")

# 493: sqrt(2/x + 2) = 3/2  -> x=8
# Square both sides -> 2/x+2 = 9/4. Multiply both sides by 4x (rearrangement,
# named): 8 + 8x = 9x  ->  x = 8.
g = G()
c8 = g.given(8)
c9 = g.given(9)
x = g.free()
f = g.rel("mul", c8, x)        # 8x
gg = g.rel("mul", c9, x)       # 9x
g.rel("add", f, c8, gg)        # 8x + 8 = 9x
add(493, g.factors, x, 8,
    f"Consider the numbers {L(c8)}, {L(c9)}, {L(x)}, {L(f)}, {L(gg)}. "
    f"{L(c8)} is 8. {L(c9)} is 9. {L(c8)} times {L(x)} equals {L(f)}. "
    f"{L(c9)} times {L(x)} equals {L(gg)}. "
    f"{L(f)} plus {L(c8)} equals {L(gg)}. What is {L(x)}?",
    "REARRANGEMENT (named): squaring both sides of sqrt(2/x+2)=3/2 gives "
    "2/x+2=9/4; multiplying by 4x clears both the fraction and the free "
    "variable's denominator, yielding 8+8x=9x -- a genuine restatement of "
    "the source equation (not a shortcut to the answer): the graph "
    "encodes both sides of the cleared equation as simultaneous "
    "multiplications of the free var x and lets the solver find the "
    "unique x satisfying both. Squaring-to-clear-a-radical is a standard "
    "necessary step to reach a polynomial form, degree stated here per "
    "the rearrangement law.")

# 495: f(x)=x^2-2x, f(-1) = 3
# x=-1 (mag 1); x^2=1; -2x at x=-1 is +2 (double-negative flips to add);
# f(-1) = x^2 + 2 = 3.
g = G()
ax = g.given(1)          # |x|
x2 = g.rel("mul", ax, ax)   # x^2 = 1
c2 = g.given(2)           # coeff magnitude
t2 = g.rel("mul", c2, ax)   # |{-2x}| = 2 (x negative, term positive)
res = g.rel("add", x2, t2)  # x^2 + 2 = 3
add(495, g.factors, res, 3,
    f"Consider the numbers {L(ax)}, {L(x2)}, {L(c2)}, {L(t2)}, {L(res)}. "
    f"{L(ax)} is 1. {L(ax)} times {L(ax)} equals {L(x2)}. {L(c2)} is 2. "
    f"{L(c2)} times {L(ax)} equals {L(t2)}. {L(x2)} plus {L(t2)} equals "
    f"{L(res)}. What is {L(res)}?",
    "magnitude-fold (Worked Example B / technique 3): x=-1 so |x|=1, "
    "x^2=1; the term -2x at x=-1 evaluates to +2 (double negative), so "
    "f(-1)=x^2-2x becomes x^2+2 directly -- rendered as addition of "
    "magnitudes since the sign flip makes both terms positive, sign "
    "bookkeeping done externally per precedent ([460]/[495]-style).")

# 496: line through (-1,6),(6,k),(20,3) -> k=5
# run1=6-(-1)=7, run2=20-(-1)=21, riseMag2=|6-3|=3.
# collinearity (equal slope ratio, theorem-application): run1*riseMag2 =
# run2*riseMag1  ->  7*3=21*m  ->  m=1.  k = 6-m = 5.
g = G()
x2v = g.given(6)     # x-coord of middle point (6)
ax1 = g.given(1)     # |x1| = |-1|
run1 = g.rel("add", x2v, ax1)   # 6+1=7
x3v = g.given(20)
run2 = g.rel("add", x3v, ax1)   # 20+1=21
y1 = g.given(6)      # y1 (same numeral as x2v, separate var)
y3 = g.given(3)
riseMag2 = g.rel("sub", y1, y3)  # 6-3=3
p = g.rel("mul", run1, riseMag2)  # 7*3=21
m = g.free()
g.rel("mul", run2, m, p)          # 21*m = 21 -> m=1
k = g.rel("sub", y1, m)           # 6-1=5
add(496, g.factors, k, 5,
    f"Consider the numbers {L(x2v)}, {L(ax1)}, {L(run1)}, {L(x3v)}, "
    f"{L(run2)}, {L(y1)}, {L(y3)}, {L(riseMag2)}, {L(p)}, {L(m)}, {L(k)}. "
    f"{L(x2v)} is 6. {L(ax1)} is 1. {L(x2v)} plus {L(ax1)} equals "
    f"{L(run1)}. {L(x3v)} is 20. {L(x3v)} plus {L(ax1)} equals {L(run2)}. "
    f"{L(y1)} is 6. {L(y3)} is 3. {L(y1)} exceeds {L(y3)} by "
    f"{L(riseMag2)}. {L(run1)} times {L(riseMag2)} equals {L(p)}. "
    f"{L(run2)} times {L(m)} equals {L(p)}. {L(y1)} exceeds {L(m)} by "
    f"{L(k)}. What is {L(k)}?",
    "THEOREM-APPLICATION (named: collinearity implies equal slope "
    "ratios / section-formula cross-multiplication), degree: transforms "
    "'three points on a line' into a solvable cross-multiplication "
    "system, the graph still performs that system's arithmetic (search "
    "for m via mult-inversion), not just the final answer. x-magnitudes "
    "folded (technique 3: |x1|=1 tracked externally since x1=-1); "
    "run1=x2-x1, run2=x3-x1 via add-of-opposite-signs; riseMag2=y1-y3; "
    "cross-multiply run1*riseMag2=run2*riseMag1 (Worked Example C mult-"
    "inversion) finds riseMag1=m=1; k=y1-m since the middle point lies "
    "between the endpoints in the same (positive) x-direction as the "
    "y1->y3 descent.")

# 497: map 12cm=72km, 17cm=? -> 102
g = G()
c12 = g.given(12)
c72 = g.given(72)
scale = g.free()
g.rel("mul", c12, scale, c72)     # 12*scale=72 -> scale=6
c17 = g.given(17)
ans = g.rel("mul", c17, scale)    # 17*6=102
add(497, g.factors, ans, 102,
    f"Consider the numbers {L(c12)}, {L(c72)}, {L(scale)}, {L(c17)}, "
    f"{L(ans)}. {L(c12)} is 12. {L(c72)} is 72. {L(c12)} times "
    f"{L(scale)} equals {L(c72)}. {L(c17)} is 17. {L(c17)} times "
    f"{L(scale)} equals {L(ans)}. What is {L(ans)}?",
    "multiplicative inversion (Worked Example C): km-per-cm scale found "
    "by search (12*scale=72 -> scale=6, since 6 isn't a single-digit-"
    "divisor-of-72-by-12 relation renderable as fdiv -- 12 isn't a valid "
    "fdiv k), then applied to 17cm.")

# 498: 16^(1/4) * 8^(1/3) * 4^(1/2) = 8
# search-based root extraction (technique 1) x3, then multiply results.
g = G()
c16 = g.given(16)
p = g.free()                          # 4th root of 16
p2 = g.rel("mul", p, p)               # p^2
g.rel("mul", p2, p2, c16)             # p^4 = 16 -> p=2
c8 = g.given(8)
q = g.free()                          # cube root of 8
q2 = g.rel("mul", q, q)               # q^2
g.rel("mul", q2, q, c8)               # q^3 = 8 -> q=2
c4 = g.given(4)
r = g.free()                          # sqrt of 4
g.rel("mul", r, r, c4)                # r^2 = 4 -> r=2
pq = g.rel("mul", p, q)
ans = g.rel("mul", pq, r)
add(498, g.factors, ans, 8,
    f"Consider the numbers {L(c16)}, {L(p)}, {L(p2)}, {L(c8)}, {L(q)}, "
    f"{L(q2)}, {L(c4)}, {L(r)}, {L(pq)}, {L(ans)}. {L(c16)} is 16. "
    f"{L(p)} times {L(p)} equals {L(p2)}. {L(p2)} times {L(p2)} equals "
    f"{L(c16)}. {L(c8)} is 8. {L(q)} times {L(q)} equals {L(q2)}. "
    f"{L(q2)} times {L(q)} equals {L(c8)}. {L(c4)} is 4. {L(r)} times "
    f"{L(r)} equals {L(c4)}. {L(p)} times {L(q)} equals {L(pq)}. "
    f"{L(pq)} times {L(r)} equals {L(ans)}. What is {L(ans)}?",
    "search-based root extraction (technique 1), applied three times: "
    "4th root of 16 via a chained square-of-square (p^2 then (p^2)^2=16), "
    "cube root of 8 via p^2*p, square root of 4 direct; all three "
    "extracted roots (2,2,2) multiplied together.")

# 499: floor(sqrt(80)) -- SKIP
skip(499,
     "Needs 'largest integer x with x^2<=N' bracket-search: 80 is not a "
     "perfect square, so no equality-based primitive finds it (only "
     "duplicate-arg EQUALITY search, e.g. x*x=N, is available, and that "
     "fails for non-perfect-squares). Same inequality/bracket-search "
     "family as tranche2's skipped [380] (log_10(17) between consecutive "
     "integers) and [412] (comparison-count over pairs) -- no primitive "
     "supports 'less-than' testing or bracketing an unknown against a "
     "range.")

# 504: closest integer to cube root of 100 -- SKIP
skip(504,
     "Needs a distance-minimization / bracket-search over candidate cubes "
     "(4^3=64, 5^3=125, compare |100-64| vs |100-125|): same inequality-"
     "reasoning family as [499] above and tranche2's [380]/[412]. No "
     "primitive supports comparing two magnitude-differences to pick the "
     "nearer one.")

# 508: a(op)b = (sqrt(2a+b))^3; 4(op)x=27 -> x=1
# y=sqrt(2a+b): y^2=2a+b (=target), y^3=27 -> y=3, target=9, x=target-2a.
g = G()
c2 = g.given(2)
a4 = g.given(4)
twoA = g.rel("mul", c2, a4)      # 2a = 8
y = g.free()
target = g.rel("mul", y, y)      # y^2 = target = 2a+b
c27 = g.given(27)
g.rel("mul", target, y, c27)     # y^3 = 27 -> y=3, target=9
x = g.rel("sub", target, twoA)   # target - 2a = 1
add(508, g.factors, x, 1,
    f"Consider the numbers {L(c2)}, {L(a4)}, {L(twoA)}, {L(y)}, "
    f"{L(target)}, {L(c27)}, {L(x)}. {L(c2)} is 2. {L(a4)} is 4. "
    f"{L(c2)} times {L(a4)} equals {L(twoA)}. {L(y)} times {L(y)} equals "
    f"{L(target)}. {L(c27)} is 27. {L(target)} times {L(y)} equals "
    f"{L(c27)}. {L(target)} exceeds {L(twoA)} by {L(x)}. What is "
    f"{L(x)}?",
    "REARRANGEMENT (named: the source's own definition a(op)b=(sqrt(2a+"
    "b))^3, degree: setting y=sqrt(2a+b) turns the cube-root chain into "
    "y^2=target (=2a+b) and y^3=27 simultaneously, both folded into the "
    "SAME chain so target is never asserted separately -- search-based "
    "cube extraction (technique 1) finds y=3, target=9; b=x is then "
    "target-2a via the source's own equation 2a+b=target.")

# 511: three numbers sum 67; larger two differ by 7; smaller two differ by 3
# -> largest = 28
g = G()
d1 = g.given(7)
d2 = g.given(3)
M = g.free()
Lg = g.rel("add", M, d1)     # L = M+7
S = g.rel("sub", M, d2)      # S = M-3
t1 = g.rel("add", Lg, M)     # L+M
c67 = g.given(67)
g.rel("add", t1, S, c67)     # L+M+S = 67
add(511, g.factors, Lg, 28,
    f"Consider the numbers {L(d1)}, {L(d2)}, {L(M)}, {L(Lg)}, {L(S)}, "
    f"{L(t1)}, {L(c67)}. {L(d1)} is 7. {L(d2)} is 3. {L(M)} plus "
    f"{L(d1)} equals {L(Lg)}. {L(M)} exceeds {L(d2)} by {L(S)}. "
    f"{L(Lg)} plus {L(M)} equals {L(t1)}. {L(c67)} is 67. {L(t1)} plus "
    f"{L(S)} equals {L(c67)}. What is {L(Lg)}?",
    "direct system encoding (technique 2): free var M (middle number) "
    "linked by both stated differences (L=M+7, S=M-3) and the sum "
    "constraint; the CSP search finds the unique M=21 (hence L=28, "
    "S=18) consistent with all three source facts simultaneously.")

# 515: degree of (3x^2+11)^12 = 24
g = G()
e2 = g.given(2)
e12 = g.given(12)
ans = g.rel("mul", e2, e12)
add(515, g.factors, ans, 24,
    f"Consider the numbers {L(e2)}, {L(e12)}, {L(ans)}. {L(e2)} is 2. "
    f"{L(e12)} is 12. {L(e2)} times {L(e12)} equals {L(ans)}. What is "
    f"{L(ans)}?",
    "degree-of-power identity: degree of (poly of degree d)^n = d*n.")

# 518: x-a+3 given x=a+7 -> 10 (independent of a; a cancels)
g = G()
c7 = g.given(7)
c3 = g.given(3)
ans = g.rel("add", c7, c3)
add(518, g.factors, ans, 10,
    f"Consider the numbers {L(c7)}, {L(c3)}, {L(ans)}. {L(c7)} is 7. "
    f"{L(c3)} is 3. {L(c7)} plus {L(c3)} equals {L(ans)}. What is "
    f"{L(ans)}?",
    "algebraic identity/cancellation (matches tranche2's [400] precedent: "
    "'perimeter equality cancels the shared side-length variable "
    "entirely'): substituting x=a+7 into x-a+3 gives (a+7)-a+3=7+3=10, "
    "independent of a -- the graph renders only the surviving 7+3 sum, "
    "since a never appears in the final value by the source's own "
    "algebra, not by any external shortcut.")

# 521: g(x)=2x^2+2x-1, g(g(2)) = 263
g = G()
x0 = g.given(2)          # x=2 (also = c1 = c2 numerically)
x2sq = g.rel("mul", x0, x0)     # x^2 = 4
term1a = g.rel("mul", x0, x2sq)  # c1*x^2 = 2*4=8
# term2a = c2*x = x0*x0 = x2sq (reuse, both equal 2*2=4)
suma = g.rel("add", term1a, x2sq)  # 8+4=12
c1const = g.given(1)
g2 = g.rel("sub", suma, c1const)   # g(2) = 11
x2sqb = g.rel("mul", g2, g2)       # 121
term1b = g.rel("mul", x0, x2sqb)   # 2*121=242
term2b = g.rel("mul", x0, g2)      # 2*11=22
sumb = g.rel("add", term1b, term2b)  # 264
ans = g.rel("sub", sumb, c1const)    # 263
add(521, g.factors, ans, 263,
    f"Consider the numbers {L(x0)}, {L(x2sq)}, {L(term1a)}, {L(suma)}, "
    f"{L(c1const)}, {L(g2)}, {L(x2sqb)}, {L(term1b)}, {L(term2b)}, "
    f"{L(sumb)}, {L(ans)}. {L(x0)} is 2. {L(x0)} times {L(x0)} equals "
    f"{L(x2sq)}. {L(x0)} times {L(x2sq)} equals {L(term1a)}. "
    f"{L(term1a)} plus {L(x2sq)} equals {L(suma)}. {L(c1const)} is 1. "
    f"{L(suma)} exceeds {L(c1const)} by {L(g2)}. {L(g2)} times {L(g2)} "
    f"equals {L(x2sqb)}. {L(x0)} times {L(x2sqb)} equals {L(term1b)}. "
    f"{L(x0)} times {L(g2)} equals {L(term2b)}. {L(term1b)} plus "
    f"{L(term2b)} equals {L(sumb)}. {L(sumb)} exceeds {L(c1const)} by "
    f"{L(ans)}. What is {L(ans)}?",
    "fixed small-exponent direct computation, applied twice (g(2) then "
    "g(g(2))): both coefficients of g happen to equal the input value 2 "
    "at the first call, so the given var a=2 is reused for x, c1(=2), "
    "and c2(=2) -- a genuine numeric coincidence in the source, not an "
    "invented shortcut (each reuse is checked against the actual source "
    "coefficients, which are literally both 2). term2 of the first call "
    "(c2*x=2*2) equals x^2 (var c) exactly, so it's reused rather than "
    "recomputed. Second call uses the derived g(2)=11 as input, computed "
    "fresh (no more coincidental reuse since 11 != 2).")

# 525: largest x s.t. (x+1)/(8x^2-65x+8) undefined -> denominator=0 -> x=8
# 8x^2-65x+8 factors as (8x-1)(x-8); rendered via the factored form (not
# the expanded quadratic) both to stay under the value cap (8*x^2=512 at
# x=8 blows the 300 cap in the expanded form -- caught by solve2 failing
# on the first attempt) and because it makes the integer-root argument
# explicit rather than implicit.
g = G()
c8 = g.given(8)      # reused: leading coeff of 8x-1, and the constant in x-8
c1 = g.given(1)
x = g.free()
t1 = g.rel("mul", c8, x)        # 8x
factor1 = g.rel("sub", t1, c1)  # 8x-1
factor2 = g.rel("sub", x, c8)   # x-8
c0 = g.given(0)
g.rel("mul", factor1, factor2, c0)   # (8x-1)(x-8) = 0
add(525, g.factors, x, 8,
    f"Consider the numbers {L(c8)}, {L(c1)}, {L(x)}, {L(t1)}, "
    f"{L(factor1)}, {L(factor2)}, {L(c0)}. {L(c8)} is 8. {L(c1)} is 1. "
    f"{L(c8)} times {L(x)} equals {L(t1)}. {L(t1)} exceeds {L(c1)} by "
    f"{L(factor1)}. {L(x)} exceeds {L(c8)} by {L(factor2)}. {L(c0)} is "
    f"0. {L(factor1)} times {L(factor2)} equals {L(c0)}. What is "
    f"{L(x)}?",
    "REARRANGEMENT (named: factoring 8x^2-65x+8=(8x-1)(x-8), a named "
    "identity of the source's own denominator -- verified by direct "
    "expansion), used INSTEAD of the expanded quadratic specifically "
    "because 8*x^2 at x=8 is 512, over the 300 cap (caught by solve2 "
    "failing on the first attempt); the factored form keeps every "
    "intermediate under the cap. LAW-6 TENSION (flagged for the wheel): "
    "denominator=0 forces (8x-1)(x-8)=0; the graph doesn't compute an "
    "ordering to pick the 'largest' root -- 'x exceeds 8 by factor2' "
    "(sub[x,c8]) is infeasible in the nonnegative domain for x<8, and "
    "8x-1=0 has no integer solution at all, so x=8 is the only integer "
    "reachable, same domain-restriction argument as the nested-radical "
    "items ([413]/[479] precedent), now made structurally visible "
    "through the factor x-8 itself rather than left implicit.")

# 528: circle x^2-6x+y^2+2y+6=0, radius -> 2
g = G()
d1 = g.given(6)     # coeff of x
sq1 = g.rel("mul", d1, d1)      # 36
d2 = g.given(2)      # coeff of y
sq2 = g.rel("mul", d2, d2)      # 4
ssum = g.rel("add", sq1, sq2)   # 40
quarter = g.fdiv(ssum, 4)       # 10   (ONE fdiv)
cconst = g.given(6)  # constant term
rad2 = g.rel("sub", quarter, cconst)   # 4
r = g.free()
g.rel("mul", r, r, rad2)        # r^2 = 4 -> r=2
add(528, g.factors, r, 2,
    f"Consider the numbers {L(d1)}, {L(sq1)}, {L(d2)}, {L(sq2)}, "
    f"{L(ssum)}, {L(quarter)}, {L(cconst)}, {L(rad2)}, {L(r)}. "
    f"{L(d1)} is 6. {L(d1)} times {L(d1)} equals {L(sq1)}. {L(d2)} is "
    f"2. {L(d2)} times {L(d2)} equals {L(sq2)}. {L(sq1)} plus {L(sq2)} "
    f"equals {L(ssum)}. When {L(ssum)} is divided by 4, the quotient is "
    f"{L(quarter)}. {L(cconst)} is 6. {L(quarter)} exceeds {L(cconst)} "
    f"by {L(rad2)}. {L(r)} times {L(r)} equals {L(rad2)}. What is "
    f"{L(r)}?",
    "THEOREM-APPLICATION (named: completing the square on both x and y "
    "to reach standard circle form (x-h)^2+(y-k)^2=r^2), degree: "
    "r^2=(coeff_x^2+coeff_y^2)/4 - const; both squared coefficients "
    "summed BEFORE the single division by 4 (rather than halved "
    "separately) specifically to stay within the one-fdiv budget (two "
    "separate halvings would need two fdivs); r extracted via search-"
    "based square-root (technique 1). Same identity family as "
    "tranche2's [463] completing-the-square precedent.")

# 529: A*B=(A+B)/3; (2*10)*5 -> 3
g = G()
a2 = g.given(2)
b10 = g.given(10)
s1 = g.rel("add", a2, b10)       # 12
r1 = g.fdiv(s1, 3)                # 4  (ONE fdiv)
c5 = g.given(5)
s2 = g.rel("add", r1, c5)         # 9
c3 = g.given(3)
x = g.free()
g.rel("mul", c3, x, s2)           # 3*x = 9 -> x=3
add(529, g.factors, x, 3,
    f"Consider the numbers {L(a2)}, {L(b10)}, {L(s1)}, {L(r1)}, "
    f"{L(c5)}, {L(s2)}, {L(c3)}, {L(x)}. {L(a2)} is 2. {L(b10)} is 10. "
    f"{L(a2)} plus {L(b10)} equals {L(s1)}. When {L(s1)} is divided by "
    f"3, the quotient is {L(r1)}. {L(c5)} is 5. {L(r1)} plus {L(c5)} "
    f"equals {L(s2)}. {L(c3)} is 3. {L(c3)} times {L(x)} equals "
    f"{L(s2)}. What is {L(x)}?",
    "the custom operator A*B=(A+B)/3 is applied twice ((2 star 10) star "
    "5); the 'at most one fdiv' rule bites on the repeated /3, so the "
    "FIRST application uses fdiv (k=3) and the SECOND uses "
    "multiplicative inversion (Worked Example C) instead of a second "
    "fdiv -- same pattern as tranche2's [401] (repeated halving).")

# 533: balance scale -> 16 blue balls
g = G()
gr3 = g.given(3)
bl6 = g.given(6)
perGreen = g.free()
g.rel("mul", gr3, perGreen, bl6)   # 3*perGreen=6 -> perGreen=2
gr4 = g.given(4)
greenBlue = g.rel("mul", gr4, perGreen)   # 4*2=8
yellowBlue = g.given(5)   # "2 yellow balls balance 5 blue balls" -- the
                           # source's OWN stated equivalence is exactly for
                           # 2 yellow, which is exactly the target quantity,
                           # so it's used directly (no ratio scaling needed)
whiteRatio = g.given(6)   # "6 blue balls balance 4 white balls"
whiteBlue = g.fdiv(whiteRatio, 2)   # half of (6 blue = 4 white) -> 2 white=3 blue
t1 = g.rel("add", greenBlue, yellowBlue)   # 8+5=13
ans = g.rel("add", t1, whiteBlue)          # 13+3=16
add(533, g.factors, ans, 16,
    f"Consider the numbers {L(gr3)}, {L(bl6)}, {L(perGreen)}, {L(gr4)}, "
    f"{L(greenBlue)}, {L(yellowBlue)}, {L(whiteRatio)}, {L(whiteBlue)}, "
    f"{L(t1)}, {L(ans)}. {L(gr3)} is 3. {L(bl6)} is 6. {L(gr3)} times "
    f"{L(perGreen)} equals {L(bl6)}. {L(gr4)} is 4. {L(gr4)} times "
    f"{L(perGreen)} equals {L(greenBlue)}. {L(yellowBlue)} is 5. "
    f"{L(whiteRatio)} is 6. When {L(whiteRatio)} is divided by 2, the "
    f"quotient is {L(whiteBlue)}. {L(greenBlue)} plus {L(yellowBlue)} "
    f"equals {L(t1)}. {L(t1)} plus {L(whiteBlue)} equals {L(ans)}. "
    f"What is {L(ans)}?",
    "LAW 1/2 TENSION (flagged for the wheel): the blue-equivalent of "
    "'2 yellow' is taken DIRECTLY from the source's stated '2 yellow "
    "balls balance 5 blue balls' (var yellowBlue=5) without an "
    "intervening ratio computation, since the target quantity (2 "
    "yellow) exactly matches the source's stated quantity -- a genuine "
    "reconstruction-co-test pass (the literal 5 is present and used for "
    "exactly the scenario it describes), but flagged because the "
    "source's yellow-count literal '2' never becomes its own graph "
    "node (no computation was needed to reach it). Green: 3green=6blue "
    "gives a per-green multiplier (2, mult-inversion) applied to the "
    "target 4green. White: since the target wants exactly HALF of the "
    "source's stated '4 white' (2white vs 4white), a single fdiv(k=2) "
    "on the blue side (6->3) directly gives the 2-white blue-"
    "equivalent, avoiding a second ratio derivation. One fdiv total.")

# 537: 1/5 red, 1/2 blue, 1/10 green, remaining 30 white -> green=15
g = G()
D = g.given(10)          # LCD, literally the denominator of the source's
                          # own '1/10' (green) fraction
redDenom = g.given(5)
redNum = g.fdiv(D, 5)     # 10/5=2   (ONE fdiv)
blueDenom = g.given(2)
blueNum = g.free()
g.rel("mul", blueDenom, blueNum, D)   # 2*blueNum=10 -> blueNum=5
greenNum = g.given(1)     # numerator of the source's own '1/10'
t1 = g.rel("add", redNum, blueNum)     # 2+5=7
t2 = g.rel("add", t1, greenNum)        # 7+1=8
whiteNum = g.rel("sub", D, t2)         # 10-8=2
whiteCount = g.given(30)
t3 = g.rel("mul", whiteCount, greenNum)   # 30*1=30
greenCount = g.free()
g.rel("mul", whiteNum, greenCount, t3)    # 2*greenCount=30 -> 15
add(537, g.factors, greenCount, 15,
    f"Consider the numbers {L(D)}, {L(redDenom)}, {L(redNum)}, "
    f"{L(blueDenom)}, {L(blueNum)}, {L(greenNum)}, {L(t1)}, {L(t2)}, "
    f"{L(whiteNum)}, {L(whiteCount)}, {L(t3)}, {L(greenCount)}. "
    f"{L(D)} is 10. {L(redDenom)} is 5. When {L(D)} is divided by 5, "
    f"the quotient is {L(redNum)}. {L(blueDenom)} is 2. "
    f"{L(blueDenom)} times {L(blueNum)} equals {L(D)}. {L(greenNum)} "
    f"is 1. {L(redNum)} plus {L(blueNum)} equals {L(t1)}. {L(t1)} plus "
    f"{L(greenNum)} equals {L(t2)}. {L(D)} exceeds {L(t2)} by "
    f"{L(whiteNum)}. {L(whiteCount)} is 30. {L(whiteCount)} times "
    f"{L(greenNum)} equals {L(t3)}. {L(whiteNum)} times {L(greenCount)} "
    f"equals {L(t3)}. What is {L(greenCount)}?",
    "direct fraction/LCD arithmetic, deliberately AVOIDING the pct "
    "primitive (which would need an externally-computed p=20 for the "
    "white share, hardcoded outside the graph -- itself a Law-3 residue "
    "risk, and pct is flagged in the brief as scattering at the book "
    "register besides): all three fractions (1/5 red, 1/2 blue, 1/10 "
    "green) rendered over the common denominator D=10 -- lawful since "
    "D is literally the source's own '1/10' denominator, not an "
    "invented LCD (Law 1/2). red/blue numerators over D found via fdiv "
    "and mult-inversion; green numerator is the source's own literal 1; "
    "white's numerator is derived in-graph as the remainder (D minus "
    "the other three), then used to invert white_count=30 into "
    "green_count via cross-multiplication (Worked Example C). One "
    "fdiv total.")

# 538: square vertices A(0,0) B(-5,-1) C(-4,-6) D(1,-5) -> area 26
g = G()
dx = g.given(5)
dx2 = g.rel("mul", dx, dx)
dy = g.given(1)
dy2 = g.rel("mul", dy, dy)
ans = g.rel("add", dx2, dy2)
add(538, g.factors, ans, 26,
    f"Consider the numbers {L(dx)}, {L(dx2)}, {L(dy)}, {L(dy2)}, "
    f"{L(ans)}. {L(dx)} is 5. {L(dx)} times {L(dx)} equals {L(dx2)}. "
    f"{L(dy)} is 1. {L(dy)} times {L(dy)} equals {L(dy2)}. {L(dx2)} "
    f"plus {L(dy2)} equals {L(ans)}. What is {L(ans)}?",
    "THEOREM-APPLICATION (named: area of a square = side-length squared "
    "= squared Euclidean distance between adjacent vertices, i.e. the "
    "distance formula composed with the area formula), degree: "
    "transforms 'find the area of this square given 4 vertices' into "
    "'compute (Bx-Ax)^2+(By-Ay)^2' directly -- the graph still performs "
    "that transformed problem's arithmetic (both squarings and the "
    "sum), it just never re-derives the general area/distance formulas "
    "themselves. Magnitude-fold for the negative coordinate deltas "
    "(Bx-Ax=-5, By-Ay=-1): both squared away, sign irrelevant.")

# 539: log_3(81) - log_3(1/9) -- SKIP
skip(539,
     "Both terms require solving 3^x=TARGET for an unknown exponent x "
     "(81 and 1/9 respectively) -- same 'no primitive for repeated "
     "multiplication a variable number of times' gap as tranche2's "
     "skipped [371]/[378]/[473]. The second term additionally needs a "
     "negative exponent (1/9=3^-2), which the nonnegative-integer domain "
     "can't represent at all. Double-blocked; same family as [371].")

# 540: x^2+15x+54=(x+a)(x+b); x^2-17x+72=(x-b)(x-c) -> a+b+c=23
g = G()
s1 = g.given(15)
p1 = g.given(54)
s2 = g.given(17)
p2 = g.given(72)
av = g.free()
bv = g.free()
cv = g.free()
g.rel("add", av, bv, s1)
g.rel("mul", av, bv, p1)
g.rel("add", bv, cv, s2)
g.rel("mul", bv, cv, p2)
ans = g.rel("add", s1, cv)   # (a+b)+c = 15+c
add(540, g.factors, ans, 23,
    f"Consider the numbers {L(s1)}, {L(p1)}, {L(s2)}, {L(p2)}, {L(av)}, "
    f"{L(bv)}, {L(cv)}, {L(ans)}. {L(s1)} is 15. {L(p1)} is 54. "
    f"{L(s2)} is 17. {L(p2)} is 72. {L(av)} plus {L(bv)} equals "
    f"{L(s1)}. {L(av)} times {L(bv)} equals {L(p1)}. {L(bv)} plus "
    f"{L(cv)} equals {L(s2)}. {L(bv)} times {L(cv)} equals {L(p2)}. "
    f"{L(s1)} plus {L(cv)} equals {L(ans)}. What is {L(ans)}?",
    "direct system encoding (technique 2): three free vars a,b,c jointly "
    "constrained by BOTH factorizations' sum/product pairs (Vieta's "
    "relations from the source's own two equations); the CSP search "
    "finds the unique (a,b,c)=(6,9,8) satisfying all four constraints "
    "simultaneously (b is forced to be the value shared by both "
    "quadratics' root pairs). a+b+c rendered as the already-known sum "
    "s1=a+b plus c, not recomputed from scratch.")

# 542: (3+x(3+x)-3^2)/(x-3+x^2) at x=-2 -> 8
g = G()
ax = g.given(2)     # |x|
c3 = g.given(3)
A = g.rel("sub", c3, ax)          # 3+x = 3-|x| = 1  (x negative)
Bmag = g.rel("mul", ax, A)        # |x*(3+x)| = 2*1=2 (negative)
nine = g.rel("mul", c3, c3)       # 3^2 = 9
d1 = g.rel("sub", nine, c3)       # 9-3=6
numMag = g.rel("add", d1, Bmag)   # 6+2=8  (numerator magnitude, negative)
xsq = g.rel("mul", ax, ax)        # x^2=4
d2 = g.rel("add", ax, c3)         # |x|+3=5 (magnitude of x-3, negative)
denomMag = g.rel("sub", d2, xsq)  # 5-4=1  (denominator magnitude, negative)
res = g.free()
g.rel("mul", denomMag, res, numMag)   # 1*res=8 -> res=8 (neg/neg=pos)
add(542, g.factors, res, 8,
    f"Consider the numbers {L(ax)}, {L(c3)}, {L(A)}, {L(Bmag)}, "
    f"{L(nine)}, {L(d1)}, {L(numMag)}, {L(xsq)}, {L(d2)}, {L(denomMag)}, "
    f"{L(res)}. {L(ax)} is 2. {L(c3)} is 3. {L(c3)} exceeds {L(ax)} by "
    f"{L(A)}. {L(ax)} times {L(A)} equals {L(Bmag)}. {L(c3)} times "
    f"{L(c3)} equals {L(nine)}. {L(nine)} exceeds {L(c3)} by {L(d1)}. "
    f"{L(d1)} plus {L(Bmag)} equals {L(numMag)}. {L(ax)} times {L(ax)} "
    f"equals {L(xsq)}. {L(ax)} plus {L(c3)} equals {L(d2)}. {L(d2)} "
    f"exceeds {L(xsq)} by {L(denomMag)}. {L(denomMag)} times {L(res)} "
    f"equals {L(numMag)}. What is {L(res)}?",
    "magnitude-fold (technique 3), the tranche's most sign-heavy item: "
    "at x=-2 every intermediate is tracked by magnitude with sign kept "
    "externally -- numerator magnitude 8 (negative), denominator "
    "magnitude 1 (negative), final ratio positive (neg/neg) found via "
    "multiplicative inversion since the denominator is a derived value "
    "(axis-1 rule: division by a derived quantity must be mult-with-"
    "derived-args, never fdiv). Zero fdivs.")

# 543: 4:x^2 = x:16 -> x=4
g = G()
c4 = g.given(4)
c16 = g.given(16)
prod = g.rel("mul", c4, c16)      # 64
x = g.free()
x2 = g.rel("mul", x, x)
g.rel("mul", x2, x, prod)         # x^3 = 64 -> x=4
add(543, g.factors, x, 4,
    f"Consider the numbers {L(c4)}, {L(c16)}, {L(prod)}, {L(x)}, "
    f"{L(x2)}. {L(c4)} is 4. {L(c16)} is 16. {L(c4)} times {L(c16)} "
    f"equals {L(prod)}. {L(x)} times {L(x)} equals {L(x2)}. {L(x2)} "
    f"times {L(x)} equals {L(prod)}. What is {L(x)}?",
    "REARRANGEMENT (named: cross-multiplying the proportion 4:x^2 = "
    "x:16 gives 4*16=x^3), search-based cube extraction (technique 1) "
    "finds the unique x=4.")

# 544: 9^4+9^4+9^4=3^x -- SKIP
skip(544,
     "Double-blocked, exactly like tranche2's [431]: (a) direct value-"
     "space computation of 9^4=6561 and the sum 19683 exceeds the 300 "
     "cap by two orders of magnitude, no rescaling avoids it since the "
     "base itself (9 or 3) must appear at real magnitude somewhere in "
     "any faithful chain; (b) the exponent-space shortcut (x = 2*4+1=9, "
     "using log rules on the exponents themselves) is the SAME banned "
     "'abstract exponent bookkeeping outside the value domain' family "
     "as tranche1's [233]/[234]/[324] and tranche2's [431]. Both routes "
     "blocked; skipped for consistency with that precedent.")

# 549: c*a^b - d, maximize over {0,1,2,3} -- SKIP
skip(549,
     "Requires searching over the 4!=24 assignments of {0,1,2,3} to "
     "(a,b,c,d) for the one that MAXIMIZES c*a^b-d -- no optimization/"
     "argmax primitive exists (the CSP core is equality-based, not an "
     "objective solver). Rendering the graph with the already-known-"
     "optimal assignment (a=3,b=2,c=1,d=0) hardcoded as givens would "
     "just compute 1*9-0=9 from constants with no genuine search "
     "performed -- textbook Law-3 residue (the convicted specimen: 'a "
     "graph whose only operation produces the queried value from "
     "constants is presumptively residue'), since the actual asked-for "
     "reasoning (which assignment is optimal) never happens in-graph. "
     "Honest skip rather than laundering the optimum.")

# 552: three positive integers sum 72, ratio 1:3:4 -> least = 9
g = G()
r1 = g.given(1)
r2 = g.given(3)
r3 = g.given(4)
s1 = g.rel("add", r1, r2)      # 4
s2 = g.rel("add", s1, r3)      # 8
c72 = g.given(72)
x = g.free()
g.rel("mul", s2, x, c72)       # 8x=72 -> x=9
add(552, g.factors, x, 9,
    f"Consider the numbers {L(r1)}, {L(r2)}, {L(r3)}, {L(s1)}, "
    f"{L(s2)}, {L(c72)}, {L(x)}. {L(r1)} is 1. {L(r2)} is 3. {L(r3)} "
    f"is 4. {L(r1)} plus {L(r2)} equals {L(s1)}. {L(s1)} plus {L(r3)} "
    f"equals {L(s2)}. {L(c72)} is 72. {L(s2)} times {L(x)} equals "
    f"{L(c72)}. What is {L(x)}?",
    "ratio-unit multiplicative inversion (Worked Example C): the ratio "
    "parts (1,3,4) summed to the total-parts count (8), then the unit "
    "size x found by search against the stated total (72); the least "
    "number is exactly 1*x=x since the smallest ratio part is 1, so "
    "the query var is x itself with no extra multiplication needed.")

# 553: log_5(x+4)=3, find log_11(x) -- SKIP
skip(553,
     "First step (x+4=5^3=125, x=121) is a fixed-known-exponent "
     "computation and would render fine (technique from [446]). But the "
     "QUERY is log_11(x)=log_11(121), which requires solving 11^y=121 "
     "for an unknown exponent y -- the same 'no primitive for variable-"
     "count repeated multiplication' gap as [371]/[378]/[473]/[539] "
     "above. A multiplicative (linear) inversion doesn't substitute: "
     "mul[11,y]=121 solves y=11, not the log's y=2 -- fundamentally the "
     "wrong operation, not just a magnitude problem. Since the entire "
     "query is the unsolvable step, the candidate is skipped whole "
     "rather than partially rendering the (unused) first step.")

# 558: 5th term=11, common diff=1 -> product of first two terms = 56
g = G()
t5 = g.given(11)
steps = g.given(4)     # 5th term is 4 steps from 1st
diff = g.given(1)
back = g.rel("mul", steps, diff)   # 4*1=4
first = g.rel("sub", t5, back)     # 11-4=7
second = g.rel("add", first, diff)  # 7+1=8
ans = g.rel("mul", first, second)   # 56
add(558, g.factors, ans, 56,
    f"Consider the numbers {L(t5)}, {L(steps)}, {L(diff)}, {L(back)}, "
    f"{L(first)}, {L(second)}, {L(ans)}. {L(t5)} is 11. {L(steps)} is "
    f"4. {L(diff)} is 1. {L(steps)} times {L(diff)} equals {L(back)}. "
    f"{L(t5)} exceeds {L(back)} by {L(first)}. {L(first)} plus "
    f"{L(diff)} equals {L(second)}. {L(first)} times {L(second)} "
    f"equals {L(ans)}. What is {L(ans)}?",
    "arithmetic-sequence identity: first term = 5th term - 4*diff "
    "(stated ordinal 'fifth' gives the step count 4); second term = "
    "first+diff; product of the two.")

# 560: (2^2)^3 = 64
g = G()
c2 = g.given(2)
sq = g.rel("mul", c2, c2)       # 4
sq2 = g.rel("mul", sq, sq)      # 16
ans = g.rel("mul", sq2, sq)     # 64
add(560, g.factors, ans, 64,
    f"Consider the numbers {L(c2)}, {L(sq)}, {L(sq2)}, {L(ans)}. "
    f"{L(c2)} is 2. {L(c2)} times {L(c2)} equals {L(sq)}. {L(sq)} "
    f"times {L(sq)} equals {L(sq2)}. {L(sq2)} times {L(sq)} equals "
    f"{L(ans)}. What is {L(ans)}?",
    "fixed small-exponent direct computation: 2^2=4, then 4^3 via two "
    "chained multiplications (4^2=16, 4^3=64).")

# 561: a inversely varies with b^2; a=9 at b=2; find a at b=3 -> 4
g = G()
a1 = g.given(9)
b1 = g.given(2)
b1sq = g.rel("mul", b1, b1)     # 4
k = g.rel("mul", a1, b1sq)      # 36
b2 = g.given(3)
b2sq = g.rel("mul", b2, b2)     # 9
a2 = g.free()
g.rel("mul", a2, b2sq, k)       # a2*9=36 -> a2=4
add(561, g.factors, a2, 4,
    f"Consider the numbers {L(a1)}, {L(b1)}, {L(b1sq)}, {L(k)}, "
    f"{L(b2)}, {L(b2sq)}, {L(a2)}. {L(a1)} is 9. {L(b1)} is 2. "
    f"{L(b1)} times {L(b1)} equals {L(b1sq)}. {L(a1)} times {L(b1sq)} "
    f"equals {L(k)}. {L(b2)} is 3. {L(b2)} times {L(b2)} equals "
    f"{L(b2sq)}. {L(a2)} times {L(b2sq)} equals {L(k)}. What is "
    f"{L(a2)}?",
    "inverse-variation identity a*b^2=k (constant); k found from the "
    "first pair, then multiplicative inversion (Worked Example C) finds "
    "a2 from the second pair's b^2.")

# 562: coffee inversely proportional to sleep; 9hrs->2gal; 6hrs->? -> 3
g = G()
s1 = g.given(9)
c1 = g.given(2)
k = g.rel("mul", s1, c1)        # 18
s2 = g.given(6)
c2 = g.fdiv(k, 6)                # 18/6=3   (ONE fdiv, k=6 matches the
                                  # source's own literal '6 hours')
add(562, g.factors, c2, 3,
    f"Consider the numbers {L(s1)}, {L(c1)}, {L(k)}, {L(s2)}, {L(c2)}. "
    f"{L(s1)} is 9. {L(c1)} is 2. {L(s1)} times {L(c1)} equals {L(k)}. "
    f"{L(s2)} is 6. When {L(k)} is divided by 6, the quotient is "
    f"{L(c2)}. What is {L(c2)}?",
    "inverse-proportion identity sleep*coffee=k; Tuesday's coffee found "
    "via fdiv(k=6) -- the divisor 6 coincides with the source's own "
    "stated 'six hours of sleep' literal, so the bare fdiv constant is "
    "not invented, it's the source's own value (var s2=6 is drafted "
    "explicitly for fidelity even though the fdiv factor itself takes "
    "the divisor as a literal parameter, not a var reference).")

# 564: James $66, $1 and $2 bills, 49 bills total -> 32 one-dollar bills
g = G()
c49 = g.given(49)
c66 = g.given(66)
y = g.rel("sub", c66, c49)      # 66-49=17 (# of $2 bills)
x = g.rel("sub", c49, y)        # 49-17=32 (# of $1 bills)
add(564, g.factors, x, 32,
    f"Consider the numbers {L(c49)}, {L(c66)}, {L(y)}, {L(x)}. "
    f"{L(c49)} is 49. {L(c66)} is 66. {L(c66)} exceeds {L(c49)} by "
    f"{L(y)}. {L(c49)} exceeds {L(y)} by {L(x)}. What is {L(x)}?",
    "REARRANGEMENT (named: source system x+y=49 (bill count), x+2y=66 "
    "(dollar total); subtracting the first from the second gives y=17 "
    "directly, then x=49-y=32) -- folds to two subtractions, both "
    "constants (49, 66) are the source's own totals.")

# 568: geometric sequence a,b,c,32,64 -> first term a = 4
g = G()
c32 = g.given(32)
c64 = g.given(64)
r = g.free()
g.rel("mul", c32, r, c64)       # 32r=64 -> r=2
cv = g.free()
g.rel("mul", cv, r, c32)        # c*r=32 -> c=16
bv = g.free()
g.rel("mul", bv, r, cv)         # b*r=c -> b=8
av = g.free()
g.rel("mul", av, r, bv)         # a*r=b -> a=4
add(568, g.factors, av, 4,
    f"Consider the numbers {L(c32)}, {L(c64)}, {L(r)}, {L(cv)}, "
    f"{L(bv)}, {L(av)}. {L(c32)} is 32. {L(c64)} is 64. {L(c32)} times "
    f"{L(r)} equals {L(c64)}. {L(cv)} times {L(r)} equals {L(c32)}. "
    f"{L(bv)} times {L(r)} equals {L(cv)}. {L(av)} times {L(r)} equals "
    f"{L(bv)}. What is {L(av)}?",
    "multiplicative inversion (Worked Example C), chained four times: "
    "common ratio r found from the two known consecutive terms (32,64), "
    "then walked backward one multiplicative-inversion step at a time "
    "(c, then b, then a) since each step divides by the derived ratio.")

# 570: J(a,b,c)=a/b+b/c+c/a, J(2,12,9) -> 6
g = G()
av = g.given(2)
bv = g.given(12)
cv = g.given(9)
aa = g.rel("mul", av, av)        # a^2 = 4
D = g.rel("mul", aa, cv)         # D = a^2*c = 36 (common denominator)
sb = g.free()
g.rel("mul", bv, sb, D)          # b*sb = D -> sb = 3  (scale for a/b term)
t1n = g.rel("mul", av, sb)       # a*sb = 6  (a/b term's numerator over D)
t2n = g.rel("mul", bv, aa)       # b*a^2 = 48 (b/c term's numerator over D,
                                  # since D/c = a^2 exactly)
ac = g.rel("mul", av, cv)        # a*c = 18 (= D/a exactly)
t3n = g.rel("mul", cv, ac)       # c*(a*c) = 162 (c/a term's numerator)
s1 = g.rel("add", t1n, t2n)      # 6+48=54
s2 = g.rel("add", s1, t3n)       # 54+162=216
res = g.free()
g.rel("mul", D, res, s2)         # D*res = 216 -> res=6
add(570, g.factors, res, 6,
    f"Consider the numbers {L(av)}, {L(bv)}, {L(cv)}, {L(aa)}, {L(D)}, "
    f"{L(sb)}, {L(t1n)}, {L(t2n)}, {L(ac)}, {L(t3n)}, {L(s1)}, "
    f"{L(s2)}, {L(res)}. {L(av)} is 2. {L(bv)} is 12. {L(cv)} is 9. "
    f"{L(av)} times {L(av)} equals {L(aa)}. {L(aa)} times {L(cv)} "
    f"equals {L(D)}. {L(bv)} times {L(sb)} equals {L(D)}. {L(av)} "
    f"times {L(sb)} equals {L(t1n)}. {L(bv)} times {L(aa)} equals "
    f"{L(t2n)}. {L(av)} times {L(cv)} equals {L(ac)}. {L(cv)} times "
    f"{L(ac)} equals {L(t3n)}. {L(t1n)} plus {L(t2n)} equals {L(s1)}. "
    f"{L(s1)} plus {L(t3n)} equals {L(s2)}. {L(D)} times {L(res)} "
    f"equals {L(s2)}. What is {L(res)}?",
    "LAW 3/4 TENSION (flagged for the wheel): fraction addition a/b+b/c"
    "+c/a rendered over a common denominator D=a^2*c=36, chosen instead "
    "of the standard LCD(12,9,2)=36 (same numeric value here, but "
    "reached via a DIFFERENT, source-literal-derived construction: "
    "D=a^2*c is directly computable from the given a,c, whereas 'LCD' "
    "as a general number-theoretic construction has no primitive). The "
    "true general common denominator a*b*c=216 was tried first and "
    "rejected: its per-term numerators (up to 288, 972) blow the 300 "
    "cap. D=a^2*c staying divisible by b=12 (36/12=3) is a numeric "
    "coincidence of THIS problem's specific values, not a general "
    "identity -- the graph doesn't assert this identity abstractly, it "
    "lets the solver's search for sb VERIFY the divisibility (if it "
    "didn't divide evenly, solve2 would return None and this row "
    "wouldn't exist), so the construction is checked rather than "
    "assumed. Flagging because the choice of D is non-obvious/clever "
    "and worth a second look under the deduction-vs-assembly test.")

# 572: t-6 hrs @ 2t-5/hr = 2t-8 hrs @ t-5/hr -> t=10
g = G()
c6 = g.given(6)
c5 = g.given(5)
c2 = g.given(2)
c8 = g.given(8)
t = g.free()
h1 = g.rel("sub", t, c6)          # t-6
twoT = g.rel("mul", c2, t)        # 2t
r1 = g.rel("sub", twoT, c5)       # 2t-5
h2 = g.rel("sub", twoT, c8)       # 2t-8
r2 = g.rel("sub", t, c5)          # t-5
pay1 = g.rel("mul", h1, r1)
g.rel("mul", h2, r2, pay1)        # pay2 = pay1
add(572, g.factors, t, 10,
    f"Consider the numbers {L(c6)}, {L(c5)}, {L(c2)}, {L(c8)}, {L(t)}, "
    f"{L(h1)}, {L(twoT)}, {L(r1)}, {L(h2)}, {L(r2)}, {L(pay1)}. "
    f"{L(c6)} is 6. {L(c5)} is 5. {L(c2)} is 2. {L(c8)} is 8. {L(t)} "
    f"exceeds {L(c6)} by {L(h1)}. {L(c2)} times {L(t)} equals "
    f"{L(twoT)}. {L(twoT)} exceeds {L(c5)} by {L(r1)}. {L(twoT)} "
    f"exceeds {L(c8)} by {L(h2)}. {L(t)} exceeds {L(c5)} by {L(r2)}. "
    f"{L(h1)} times {L(r1)} equals {L(pay1)}. {L(h2)} times {L(r2)} "
    f"equals {L(pay1)}. What is {L(t)}?",
    "direct system encoding (technique 2): both earners' pay (hours * "
    "rate, each a sub-expression of the free var t) forced equal via a "
    "shared result var; the graph never pre-cancels the quadratic terms "
    "algebraically (they DO cancel, reducing to a linear equation in t) "
    "-- the CSP just searches t directly against the two full products, "
    "so the cancellation is real solver work, not an annotator "
    "shortcut. The nonnegative domain also silently enforces t>=6 "
    "(h1=t-6 would be infeasible below that), which happens to match "
    "the unique real solution t=10 anyway.")

# 575: 12ft x 6ft room -> square yards of carpet -> 8
g = G()
c12 = g.given(12)
c6 = g.given(6)
area = g.rel("mul", c12, c6)     # 72 sqft
ft = g.given(3)                  # universal constant, feet per yard
sqft_per_sqyd = g.rel("mul", ft, ft)   # 9, derived (not raw given)
res = g.free()
g.rel("mul", sqft_per_sqyd, res, area)  # 9*res=72 -> res=8
add(575, g.factors, res, 8,
    f"Consider the numbers {L(c12)}, {L(c6)}, {L(area)}, {L(ft)}, "
    f"{L(sqft_per_sqyd)}, {L(res)}. {L(c12)} is 12. {L(c6)} is 6. "
    f"{L(c12)} times {L(c6)} equals {L(area)}. {L(ft)} is 3. {L(ft)} "
    f"times {L(ft)} equals {L(sqft_per_sqyd)}. {L(sqft_per_sqyd)} "
    f"times {L(res)} equals {L(area)}. What is {L(res)}?",
    "universal constant (3 feet/yard, matches tranche2's [390] "
    "precedent) squared in-graph to get square-feet-per-square-yard "
    "(9); since 9 is a DERIVED value (not a raw given), dividing the "
    "room's area by it uses multiplicative inversion, not fdiv, per "
    "the axis-1 rule (fdiv has no variable/derived form).")

# 577: 4^6=8^n -- SKIP
skip(577,
     "4^6=2^12, 8^n=2^(3n); solving 3n=12 for n requires either (a) "
     "abstract exponent-space arithmetic (2*6=12 for the LHS exponent, "
     "3 for RHS base) -- the same banned family as [431]/[544] above, "
     "or (b) direct value-space computation of 4^6=4096, which blows "
     "the 300 cap outright. Double-blocked exactly like [544]; skipped "
     "for consistency.")

# 583: 41^2=40^2+81; 39^2=40^2-X -> X=79
g = G()
c40 = g.given(40)
c2 = g.given(2)
t1 = g.rel("mul", c40, c2)       # 80
c1 = g.given(1)
ans = g.rel("sub", t1, c1)       # 79
add(583, g.factors, ans, 79,
    f"Consider the numbers {L(c40)}, {L(c2)}, {L(t1)}, {L(c1)}, "
    f"{L(ans)}. {L(c40)} is 40. {L(c2)} is 2. {L(c40)} times {L(c2)} "
    f"equals {L(t1)}. {L(c1)} is 1. {L(t1)} exceeds {L(c1)} by "
    f"{L(ans)}. What is {L(ans)}?",
    "difference-of-consecutive-squares identity n^2-(n-1)^2=2n-1, "
    "applied at n=40: 2*40-1=79.")

# 586: (m-4)^3 = (1/8)^(-1) -> m=6
g = G()
c8 = g.given(8)   # (1/8)^-1 = 8, a closed-form constant with no
                   # unknowns involved, evaluated externally like the
                   # 9/4 in tranche2's [493]-style radical simplification
r = g.free()
r2 = g.rel("mul", r, r)
g.rel("mul", r2, r, c8)          # r^3=8 -> r=2
c4 = g.given(4)
m = g.rel("add", r, c4)          # m = r+4 = 6
add(586, g.factors, m, 6,
    f"Consider the numbers {L(c8)}, {L(r)}, {L(r2)}, {L(c4)}, {L(m)}. "
    f"{L(c8)} is 8. {L(r)} times {L(r)} equals {L(r2)}. {L(r2)} times "
    f"{L(r)} equals {L(c8)}. {L(c4)} is 4. {L(r)} plus {L(c4)} equals "
    f"{L(m)}. What is {L(m)}?",
    "(1/8)^-1 is a closed-form numeric constant with no unknowns "
    "(evaluates to 8 by pure arithmetic on the problem's own literal), "
    "rendered as the given 8 directly -- same class of externally-"
    "simplified pure-constant expression as the (3/2)^2=9/4 step "
    "implicit in the RHS of [493] above. Search-based cube extraction "
    "(technique 1) finds m-4=2, then m=6.")

print()
print(f"TOTAL drafted (pre-checks passing): {len(rows)}")
print(f"FAILS: {fails}")
print(f"SKIPS: {[s[0] for s in skips]}")

with open('/home/bryce/mycelium/.cache/book8_t3_prose_pairs_draft.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

with open('/home/bryce/mycelium/.cache/book8_t3_skips.json', 'w') as f:
    json.dump(skips, f, indent=2)

print("done, wrote", len(rows), "rows")
