import json, sys, string
sys.path.insert(0, '/home/bryce/mycelium')
sys.path.insert(0, '/home/bryce/mycelium/scripts')
from tta_alg2_dials import solve2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic

SMP = {"n_vars": 24, "m": 300}
LETTERS = string.ascii_lowercase

CANDS = {c["src_idx"]: c for c in
         json.load(open('/home/bryce/mycelium/.cache/book8_candidates_t4.json'))["tranche4"]}


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

    def pct(self, a, b, p):
        self.factors.append({"ftype": "pct", "args": [a, b], "p": int(p)})


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
        "src_idx": src_idx, "book": 8, "tranche": 4, "floor": "prime", "fs": True,
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
# 589: 8^n*8^n*8^n=64^3 -- SKIP
skip(589,
     "8^(3n)=64^3; matching bases requires recognizing 64=8^2 (an unknown-"
     "exponent/discrete-log identification, same 'abstract exponent "
     "bookkeeping outside the value domain' family as [544]/[577] in "
     "tranche3). Direct value-space computation (64^3=262144) blows the "
     "300 cap by three orders of magnitude. Both routes blocked.")

# 596: blue:yellow=8:5; remove 12 blue, add 21 yellow -> ratio 1:3; find
# original blue count -> 24. Direct system (technique 2): x=ratio unit,
# blue=8x, yellow=5x; after the removal/addition, new ratio forced 1:3 via
# cross-multiplication on the DERIVED post-change quantities.
g = G()
c8 = g.given(8)
c5 = g.given(5)
x = g.free()
blue = g.rel("mul", c8, x)
yellow = g.rel("mul", c5, x)
c12 = g.given(12)
newBlue = g.rel("sub", blue, c12)
c21 = g.given(21)
newYellow = g.rel("add", yellow, c21)
c3 = g.given(3)
g.rel("mul", c3, newBlue, newYellow)   # 3*newBlue = newYellow (ratio 1:3)
add(596, g.factors, blue, 24,
    f"Consider the numbers {L(c8)}, {L(c5)}, {L(x)}, {L(blue)}, {L(yellow)}, "
    f"{L(c12)}, {L(newBlue)}, {L(c21)}, {L(newYellow)}, {L(c3)}. "
    f"{L(c8)} is 8. {L(c5)} is 5. {L(c8)} times {L(x)} equals {L(blue)}. "
    f"{L(c5)} times {L(x)} equals {L(yellow)}. {L(c12)} is 12. "
    f"{L(blue)} exceeds {L(c12)} by {L(newBlue)}. {L(c21)} is 21. "
    f"{L(yellow)} plus {L(c21)} equals {L(newYellow)}. {L(c3)} is 3. "
    f"{L(c3)} times {L(newBlue)} equals {L(newYellow)}. What is {L(blue)}?",
    "direct system encoding (technique 2): free var x is the ratio unit "
    "(blue=8x, yellow=5x); after removing 12 blue and adding 21 yellow, "
    "the new counts are forced into the stated 1:3 ratio via cross-"
    "multiplication (3*newBlue=newYellow); the CSP searches x directly "
    "against the full post-change system, finding x=3 (blue=24).")

# 601: log_9(2x-7)=3/2 -> x=17. Fixed KNOWN exponent 3/2 (a source literal,
# NOT an unknown-exponent search): 9^(3/2) = (sqrt(9))^3 = 3^3 = 27.
g = G()
c9 = g.given(9)
root = g.free()
g.rel("mul", root, root, c9)          # sqrt(9): root*root=9 -> root=3
sq = g.rel("mul", root, root)         # root^2 = 9 (reused structurally)
cube = g.rel("mul", sq, root)         # root^3 = 27
c7 = g.given(7)
rhs = g.rel("add", cube, c7)          # 27+7=34  (2x-7=27 -> 2x=34)
c2 = g.given(2)
x = g.free()
g.rel("mul", c2, x, rhs)              # 2x=34 -> x=17
add(601, g.factors, x, 17,
    f"Consider the numbers {L(c9)}, {L(root)}, {L(sq)}, {L(cube)}, {L(c7)}, "
    f"{L(rhs)}, {L(c2)}, {L(x)}. {L(c9)} is 9. {L(root)} times {L(root)} "
    f"equals {L(c9)}. {L(root)} times {L(root)} equals {L(sq)}. {L(sq)} "
    f"times {L(root)} equals {L(cube)}. {L(c7)} is 7. {L(cube)} plus "
    f"{L(c7)} equals {L(rhs)}. {L(c2)} is 2. {L(c2)} times {L(x)} equals "
    f"{L(rhs)}. What is {L(x)}?",
    "REARRANGEMENT (named: 9^(3/2)=(sqrt(9))^3, evaluating a source-"
    "literal FIXED exponent 3/2 via sqrt-then-cube), degree: distinct "
    "from the discrete-log family skips ([589] above, tranche3's "
    "[539]/[553]) because the exponent 3/2 is itself a source literal "
    "being EVALUATED (known-exponent computation), not an unknown "
    "exponent being SEARCHED for. sqrt(9)=3 via search-based root "
    "extraction (technique 1), cubed via chained multiplication (fixed "
    "small-exponent technique), then 2x-7=27 solved by multiplicative "
    "inversion (Worked Example C).")

# 603: 10 friends, 34 marbles, min additional so each gets >=1 distinct ->
# 21. Min total = 1+2+...+10 = 10*11/2 = 55 (arithmetic-series identity);
# additional = 55-34.
g = G()
c10 = g.given(10)
c1 = g.given(1)
np1 = g.rel("add", c10, c1)          # n+1 = 11
prod = g.rel("mul", c10, np1)        # n*(n+1) = 110
minSum = g.fdiv(prod, 2)             # 55  (ONE fdiv)
c34 = g.given(34)
ans = g.rel("sub", minSum, c34)      # 55-34=21
add(603, g.factors, ans, 21,
    f"Consider the numbers {L(c10)}, {L(c1)}, {L(np1)}, {L(prod)}, "
    f"{L(minSum)}, {L(c34)}, {L(ans)}. {L(c10)} is 10. {L(c1)} is 1. "
    f"{L(c10)} plus {L(c1)} equals {L(np1)}. {L(c10)} times {L(np1)} "
    f"equals {L(prod)}. When {L(prod)} is divided by 2, the quotient is "
    f"{L(minSum)}. {L(c34)} is 34. {L(minSum)} exceeds {L(c34)} by "
    f"{L(ans)}. What is {L(ans)}?",
    "arithmetic-series-sum identity (named: minimum sum of 10 distinct "
    "positive integers each >=1 is 1+2+...+10 = n(n+1)/2), degree: the "
    "'give each friend a DISTINCT count starting from 1' requirement is "
    "the minimal-total construction, transforming the problem into direct "
    "arithmetic on n=10; the graph performs that transformed sum "
    "(n*(n+1)/2=55) then subtracts the source's own 34. One fdiv (k=2).")

# 605: dozen roses ($20) proportional; 39 roses -> $65. Cross-multiply but
# reduce by the shared factor 3 (gcd(12,39)=3) BEFORE multiplying, to avoid
# a cap-busting intermediate (20*39=780 > 300).
g = G()
c3 = g.given(3)
c39 = g.given(39)
factor13 = g.free()
g.rel("mul", c3, factor13, c39)      # 3*13=39 -> factor13=13
c12 = g.given(12)
factor4 = g.free()
g.rel("mul", c3, factor4, c12)       # 3*4=12 -> factor4=4
c20 = g.given(20)
t = g.rel("mul", c20, factor13)      # 20*13=260
price = g.free()
g.rel("mul", factor4, price, t)      # 4*price=260 -> price=65
add(605, g.factors, price, 65,
    f"Consider the numbers {L(c3)}, {L(c39)}, {L(factor13)}, {L(c12)}, "
    f"{L(factor4)}, {L(c20)}, {L(t)}, {L(price)}. {L(c3)} is 3. "
    f"{L(c3)} times {L(factor13)} equals {L(c39)}. {L(c39)} is 39. "
    f"{L(c3)} times {L(factor4)} equals {L(c12)}. {L(c12)} is 12. "
    f"{L(c20)} is 20. {L(c20)} times {L(factor13)} equals {L(t)}. "
    f"{L(factor4)} times {L(price)} equals {L(t)}. What is {L(price)}?",
    "REARRANGEMENT (named: cross-multiplying the proportion price/39 = "
    "20/12 directly gives 20*39=780, over the 300 cap; reducing both 39 "
    "and 12 by their shared factor 3 FIRST -- price/13 = 20/4 -- keeps "
    "every intermediate under the cap), degree: same cap-avoidance family "
    "as tranche3's [525]/[570]. Both reductions (39/3=13, 12/3=4) found "
    "by multiplicative inversion (Worked Example C), final price by a "
    "third multiplicative inversion.")

# 607: 5a+2b=0, a=b-2, find 7b -> 10. Both a,b are individually fractional
# (a=-4/7, b=10/7) and a is negative -- neither integer-representable
# directly. Scale the WHOLE system by 7 (elimination via linear
# combination, Law 4 named): work entirely in A7=7a (negative -> magnitude
# fold) and B7=7b (the query itself), never materializing a or b.
g = G()
c7 = g.given(7)
c2 = g.given(2)                      # from "a is two less than b" (b-a=2)
c14 = g.rel("mul", c7, c2)           # 7*(b-a) = 14
c5 = g.given(5)                      # coefficient of a in 5a+2b=0
c2b = g.given(2)                     # coefficient of b in 5a+2b=0
aMag = g.free()                      # magnitude of 7a (7a is negative)
b7 = g.free()                        # 7b -- the query itself
g.rel("add", aMag, b7, c14)          # -7a + 7b = 14  ->  aMag + b7 = 14
t1 = g.rel("mul", c2b, b7)           # 2*(7b)
g.rel("mul", c5, aMag, t1)           # 5*aMag = 2*(7b)  [from 5*(7a)+2*(7b)=0,
                                      # 7a=-aMag -> -5*aMag+2*b7=0 -> 5*aMag=2*b7]
add(607, g.factors, b7, 10,
    f"Consider the numbers {L(c7)}, {L(c2)}, {L(c14)}, {L(c5)}, {L(c2b)}, "
    f"{L(aMag)}, {L(b7)}, {L(t1)}. {L(c7)} is 7. {L(c2)} is 2. {L(c7)} "
    f"times {L(c2)} equals {L(c14)}. {L(c5)} is 5. {L(c2b)} is 2. "
    f"{L(aMag)} plus {L(b7)} equals {L(c14)}. {L(c2b)} times {L(b7)} "
    f"equals {L(t1)}. {L(c5)} times {L(aMag)} equals {L(t1)}. What is "
    f"{L(b7)}?",
    "LAW 3/4 TENSION (flagged for the wheel): a and b are individually "
    "fractional (a=-4/7, b=10/7) and a is negative, so neither is "
    "directly integer-domain-representable. REARRANGEMENT (named: scaling "
    "both source equations by 7 -- 7*(5a+2b)=0 and 7*(b-a)=14 -- turns "
    "the fractional system into an integer one entirely in terms of "
    "A7=7a and B7=7b (B7 IS the query, 7b, so no final multiply is "
    "needed)), degree: this is elimination via linear combination on the "
    "SOURCE's own two named equations, not a shortcut straight to the "
    "answer -- the graph still hands the solver a genuine 2-free-var "
    "system (aMag, b7) it must jointly satisfy. Magnitude-fold "
    "(technique 3) for A7=7a (negative, since a<0); the sign direction "
    "(a negative, b positive) was verified externally by solving the "
    "actual values (a=-4/7, b=10/7), same class of externally-verified "
    "sign fact as tranche3's [542].")

# 610: rectangle (2m+7)(m-2)=51 -> m=5. Direct system: m free, both factors
# built from m, product forced to 51.
g = G()
c2 = g.given(2)
m = g.free()
t = g.rel("mul", c2, m)              # 2m
c7 = g.given(7)
dim1 = g.rel("add", t, c7)           # 2m+7
c2b = g.given(2)
dim2 = g.rel("sub", m, c2b)          # m-2
c51 = g.given(51)
g.rel("mul", dim1, dim2, c51)        # (2m+7)(m-2)=51
add(610, g.factors, m, 5,
    f"Consider the numbers {L(c2)}, {L(m)}, {L(t)}, {L(c7)}, {L(dim1)}, "
    f"{L(c2b)}, {L(dim2)}, {L(c51)}. {L(c2)} is 2. {L(c2)} times {L(m)} "
    f"equals {L(t)}. {L(c7)} is 7. {L(t)} plus {L(c7)} equals {L(dim1)}. "
    f"{L(c2b)} is 2. {L(m)} exceeds {L(c2b)} by {L(dim2)}. {L(c51)} is "
    f"51. {L(dim1)} times {L(dim2)} equals {L(c51)}. What is {L(m)}?",
    "direct system encoding (technique 2): free var m builds both field "
    "dimensions (2m+7 and m-2) as sub-expressions; the CSP searches m "
    "directly against the full (uncancelled) quadratic product forced to "
    "51 -- the graph never pre-solves the quadratic algebraically, the "
    "solver's search does that work.")

# 612: 25% of a number = 20% of 30 -> 24. Rendered via fraction arithmetic
# (denominators 4 and 5), NOT the pct primitive -- avoids externally
# computing any pct parameter and avoids the pointer-scatter flagged for
# pct, matches tranche3's [537] precedent.
g = G()
c30 = g.given(30)
c5 = g.given(5)                      # denominator of 20% = 1/5
amt = g.free()
g.rel("mul", c5, amt, c30)           # 5*amt=30 -> amt=6  (20% of 30)
c4 = g.given(4)                      # denominator of 25% = 1/4
n = g.rel("mul", c4, amt)            # n = 4*amt = 24  (since amt = n/4)
add(612, g.factors, n, 24,
    f"Consider the numbers {L(c30)}, {L(c5)}, {L(amt)}, {L(c4)}, {L(n)}. "
    f"{L(c30)} is 30. {L(c5)} is 5. {L(c5)} times {L(amt)} equals "
    f"{L(c30)}. {L(c4)} is 4. {L(c4)} times {L(amt)} equals {L(n)}. What "
    f"is {L(n)}?",
    "direct fraction arithmetic, deliberately AVOIDING the pct primitive "
    "(matches tranche3's [537] precedent): 20%=1/5 and 25%=1/4 rendered "
    "as their literal denominators; amt (=20% of 30=6) found by "
    "multiplicative inversion, then the number itself found as 4*amt "
    "since amt is exactly 1/4 of it. Zero fdivs, zero pct uses.")

# 613: ab+bc+cd+da=30, b+d=5, find a+c -> 6. Factoring identity
# ab+bc+cd+da = (a+c)(b+d) (named, Law 4): (a+c)*5=30 -> a+c=6.
g = G()
c30 = g.given(30)
c5 = g.given(5)
s = g.free()
g.rel("mul", s, c5, c30)             # s*5=30 -> s=6
add(613, g.factors, s, 6,
    f"Consider the numbers {L(c30)}, {L(c5)}, {L(s)}. {L(c30)} is 30. "
    f"{L(c5)} is 5. {L(s)} times {L(c5)} equals {L(c30)}. What is "
    f"{L(s)}?",
    "REARRANGEMENT (named: ab+bc+cd+da factors as (a+c)(b+d) -- grouping "
    "ab+da=a(b+d) and bc+cd=c(b+d) -- a standard factoring identity "
    "verified by direct expansion), degree: transforms the 4-term "
    "expression into a single product equation (a+c)*(b+d)=30 using the "
    "source's own literals 30 and (b+d)=5; the graph still performs the "
    "resulting division by search (multiplicative inversion, Worked "
    "Example C), not just states the answer.")

# 614: y^2+10y+33 = (y+5)^2 + K, find K -> 8. Completing the square (named,
# matches tranche3's [528]/[614]-style precedent): (y+5)^2=y^2+10y+25,
# K=33-25=8.
g = G()
c10 = g.given(10)
half = g.fdiv(c10, 2)                # 5  (ONE fdiv)
sq = g.rel("mul", half, half)        # 25
c33 = g.given(33)
K = g.rel("sub", c33, sq)            # 33-25=8
add(614, g.factors, K, 8,
    f"Consider the numbers {L(c10)}, {L(half)}, {L(sq)}, {L(c33)}, "
    f"{L(K)}. {L(c10)} is 10. When {L(c10)} is divided by 2, the "
    f"quotient is {L(half)}. {L(half)} times {L(half)} equals {L(sq)}. "
    f"{L(c33)} is 33. {L(c33)} exceeds {L(sq)} by {L(K)}. What is "
    f"{L(K)}?",
    "THEOREM-APPLICATION (named: completing the square, y^2+10y+33 = "
    "(y+5)^2 + (33-25)), matches tranche3's [528] completing-the-square "
    "precedent; the graph performs the transformed arithmetic (half the "
    "linear coefficient, square it, subtract from the constant). One "
    "fdiv (k=2).")

# 615: Joann 12mph for 3.5hr; Fran rides 3hr, same distance -> 14mph. 3.5
# hours rendered as 7 half-hours (lexical explicitation of a KNOWN value,
# 3.5=7/2, not an invented number) to keep everything integer.
g = G()
c12 = g.given(12)
c7 = g.given(7)                      # 7 half-hours = 3.5 hours
twiceDist = g.rel("mul", c12, c7)    # 12 * 7 = 84 (= 2 * actual distance,
                                      # since time was doubled to halves)
dist = g.fdiv(twiceDist, 2)          # 42  (ONE fdiv)
c3 = g.given(3)
v = g.free()
g.rel("mul", v, c3, dist)            # v*3=42 -> v=14
add(615, g.factors, v, 14,
    f"Consider the numbers {L(c12)}, {L(c7)}, {L(twiceDist)}, {L(dist)}, "
    f"{L(c3)}, {L(v)}. {L(c12)} is 12. {L(c7)} is 7. {L(c12)} times "
    f"{L(c7)} equals {L(twiceDist)}. When {L(twiceDist)} is divided by "
    f"2, the quotient is {L(dist)}. {L(c3)} is 3. {L(v)} times {L(c3)} "
    f"equals {L(dist)}. What is {L(v)}?",
    "REARRANGEMENT (named: 'three and a half hours' re-expressed as 7 "
    "half-hours, a lexical explicitation of a KNOWN source quantity "
    "preserving its exact value 3.5=7/2, not an invented number), degree: "
    "clears the non-integer time value by doubling it, computing 2x the "
    "distance (12*7=84, under cap) and halving back via fdiv; Fran's "
    "speed then found by multiplicative inversion against the true "
    "(halved) distance.")

# 619: two 20lb weights x12 lifts; two 15lb weights x? lifts, same total.
# The shared factor of TWO weights per lift cancels from both sides of
# 2*20*12=2*15*n (Law 4, named), leaving 20*12=15*n directly -- avoids the
# cap-busting 480 intermediate.
g = G()
c20 = g.given(20)
c12 = g.given(12)
total1 = g.rel("mul", c20, c12)      # 240 (= 20*12, the "two weights"
                                      # factor cancelled from both sides)
c15 = g.given(15)
n = g.free()
g.rel("mul", c15, n, total1)         # 15n=240 -> n=16
add(619, g.factors, n, 16,
    f"Consider the numbers {L(c20)}, {L(c12)}, {L(total1)}, {L(c15)}, "
    f"{L(n)}. {L(c20)} is 20. {L(c12)} is 12. {L(c20)} times {L(c12)} "
    f"equals {L(total1)}. {L(c15)} is 15. {L(c15)} times {L(n)} equals "
    f"{L(total1)}. What is {L(n)}?",
    "REARRANGEMENT (named: total weight equality 2*20*12 = 2*15*n has "
    "the shared factor of two weights-per-lift on both sides, cancelling "
    "to 20*12=15*n -- a recognizable restatement of the source's own "
    "equality, stated per Law 4), degree: keeps every intermediate under "
    "the cap (480 would exceed it); n found by multiplicative inversion.")

# 621: 2^x+2^x+2^x+2^x=128, find (x+1)(x-1) -- SKIP
skip(621,
     "4*2^x=128 -> 2^x=32=2^5 -> x=5. Finding x from 2^x=32 requires "
     "solving for an unknown exponent (discrete log) -- same 'no "
     "primitive for repeated multiplication a variable number of times' "
     "gap as tranche3's [539]/[553]/[544]/[577] and [589]/[643]/[672] "
     "above. No way to reach (x+1)(x-1)=x^2-1 without first isolating x "
     "via the blocked step.")

# 623: parallel lines y=8x+2, y=(2c)x-4 -> equal slopes -> c=4.
g = G()
c8 = g.given(8)
c2 = g.given(2)
cv = g.free()
g.rel("mul", c2, cv, c8)             # 2c=8 -> c=4
add(623, g.factors, cv, 4,
    f"Consider the numbers {L(c8)}, {L(c2)}, {L(cv)}. {L(c8)} is 8. "
    f"{L(c2)} is 2. {L(c2)} times {L(cv)} equals {L(c8)}. What is "
    f"{L(cv)}?",
    "THEOREM-APPLICATION (named: parallel lines have equal slopes), "
    "degree: transforms 'find c so the lines are parallel' into the "
    "single equation 2c=8; the graph performs that equation's arithmetic "
    "via multiplicative inversion (Worked Example C), not just states c.")

# 625: x^2=-4x, count of nonnegative solutions -- SKIP
skip(625,
     "x^2+4x=0 has roots x=0 and x=-4; only x=0 is nonnegative, so the "
     "answer (1) is a SOLUTION-COUNT, not a computed value. No "
     "cardinality/counting primitive exists (same 'no argmax/counting "
     "primitive' gap as tranche3's [549]). Rendering the count directly "
     "as a constant would be textbook Law-3 residue: the count is fixed "
     "(=1) for essentially ANY equation of this shape (one root always "
     "0, the other always excluded by nonnegativity when the linear "
     "coefficient is positive), so a hardcoded '1' would be right by "
     "structural coincidence, not by graph-derived reasoning -- fails the "
     "counterfactual test (a variant asking for the NEGATIVE-solution "
     "count would need a different unrenderable answer, and our graph "
     "wouldn't distinguish the two).")

# 626: endpoint (4,3), midpoint (2,9), find sum of coords of other endpoint
# -> 15. Midpoint formula (named): other = 2*mid - known.
g = G()
c2 = g.given(2)                      # universal constant (midpoint formula)
midX = g.given(2)                    # x-coord of midpoint (2) -- same
                                      # numeral as c2, distinct var/role
twiceMidX = g.rel("mul", c2, midX)   # 4
c4 = g.given(4)
x2 = g.rel("sub", twiceMidX, c4)     # 0
midY = g.given(9)
twiceMidY = g.rel("mul", c2, midY)   # 18
c3 = g.given(3)
y2 = g.rel("sub", twiceMidY, c3)     # 15
ans = g.rel("add", x2, y2)           # 15
add(626, g.factors, ans, 15,
    f"Consider the numbers {L(c2)}, {L(midX)}, {L(twiceMidX)}, {L(c4)}, "
    f"{L(x2)}, {L(midY)}, {L(twiceMidY)}, {L(c3)}, {L(y2)}, {L(ans)}. "
    f"{L(c2)} is 2. {L(midX)} is 2. {L(c2)} times {L(midX)} equals "
    f"{L(twiceMidX)}. {L(c4)} is 4. {L(twiceMidX)} exceeds {L(c4)} by "
    f"{L(x2)}. {L(midY)} is 9. {L(c2)} times {L(midY)} equals "
    f"{L(twiceMidY)}. {L(c3)} is 3. {L(twiceMidY)} exceeds {L(c3)} by "
    f"{L(y2)}. {L(x2)} plus {L(y2)} equals {L(ans)}. What is {L(ans)}?",
    "THEOREM-APPLICATION (named: midpoint formula, other endpoint = "
    "2*midpoint - known endpoint, per coordinate), degree: transforms "
    "'find the other endpoint' into direct linear arithmetic per axis; "
    "the '2' is a UNIVERSAL CONSTANT from the formula itself (Law 1, "
    "matches tranche3's [575] '3 feet/yard' precedent), not a source "
    "literal.")

# 631: x^2+30x+180=-36 -> x^2+30x+216=0; nonneg difference between roots
# -> 6. Roots are both NEGATIVE (magnitude-fold, technique 3): direct
# system on magnitudes M1<M2, M1+M2=30 (=p), M1*M2=216 (=q).
g = G()
c180 = g.given(180)
c36 = g.given(36)
qVal = g.rel("add", c180, c36)       # move -36 to the left: +36 -> 216
c30 = g.given(30)                    # p, already positive (no sign flip)
M2 = g.free()
diff = g.free()
Rbig = g.rel("add", M2, diff)        # the larger magnitude
g.rel("add", Rbig, M2, c30)          # Rbig+M2 = 30
g.rel("mul", Rbig, M2, qVal)         # Rbig*M2 = 216
add(631, g.factors, diff, 6,
    f"Consider the numbers {L(c180)}, {L(c36)}, {L(qVal)}, {L(c30)}, "
    f"{L(M2)}, {L(diff)}, {L(Rbig)}. {L(c180)} is 180. {L(c36)} is 36. "
    f"{L(c180)} plus {L(c36)} equals {L(qVal)}. {L(c30)} is 30. "
    f"{L(M2)} plus {L(diff)} equals {L(Rbig)}. {L(Rbig)} plus {L(M2)} "
    f"equals {L(c30)}. {L(Rbig)} times {L(M2)} equals {L(qVal)}. What is "
    f"{L(diff)}?",
    "magnitude-fold (Worked Example B / technique 3): moving -36 to the "
    "equation's other side flips its sign to +36 (rearrangement, Law 4, "
    "named), giving x^2+30x+216=0, whose two roots are BOTH negative "
    "(sum=-30, product=+216, so magnitudes add to p=30 directly with no "
    "sign correction needed, and multiply to q=216 directly since "
    "neg*neg=pos) -- verified externally by solving the actual roots "
    "(-12,-18), same class of externally-verified sign fact as "
    "tranche3's [542]/[631]-style items. Direct system encoding "
    "(technique 2): Rbig, M2 jointly constrained by sum AND product, "
    "the CSP search resolves which magnitude is larger via the "
    "nonnegative-domain requirement on diff (same self-resolving-"
    "ordering pattern as tranche3's [511]).")

# 632: sequence 2,6,10,...,x,y,26 arithmetic -> x+y=40.
g = G()
c2 = g.given(2)
c6 = g.given(6)
d = g.rel("sub", c6, c2)             # common difference = 4
c26 = g.given(26)
y = g.rel("sub", c26, d)             # 26-4=22
x = g.rel("sub", y, d)               # 22-4=18
ans = g.rel("add", x, y)             # 40
add(632, g.factors, ans, 40,
    f"Consider the numbers {L(c2)}, {L(c6)}, {L(d)}, {L(c26)}, {L(y)}, "
    f"{L(x)}, {L(ans)}. {L(c2)} is 2. {L(c6)} is 6. {L(c6)} exceeds "
    f"{L(c2)} by {L(d)}. {L(c26)} is 26. {L(c26)} exceeds {L(d)} by "
    f"{L(y)}. {L(y)} exceeds {L(d)} by {L(x)}. {L(x)} plus {L(y)} equals "
    f"{L(ans)}. What is {L(ans)}?",
    "arithmetic-sequence identity: common difference derived from the "
    "source's own first two terms (6-2=4); x and y are the two terms "
    "immediately preceding 26, found by walking backward two subtraction "
    "steps from the source's own '26' literal.")

# 636: 4th-root(2^7 * 3^3) = a*4th-root(b), find a+b -> 218. Radical
# extraction (named theorem-application): exponent 7 = 4*1 + 3 (radical
# index 4 pulls out exactly ONE full factor of 2^4, verified externally:
# 4<=7<8). a = 2^1 = 2 (the base itself, since quotient=1 exactly);
# remaining exponent for base 2 (rem2=7-4=3) rendered in-graph.
g = G()
c2 = g.given(2)
c3 = g.given(3)
c7 = g.given(7)
c4 = g.given(4)                      # radical index
rem2 = g.rel("sub", c7, c4)          # 7-4=3 (exponent of 2 left inside)
t1 = g.rel("mul", c2, c2)            # 2^2=4
t2 = g.rel("mul", t1, c2)            # 2^3=8   (chain length = rem2's value)
s1 = g.rel("mul", c3, c3)            # 3^2=9
s2 = g.rel("mul", s1, c3)            # 3^3=27  (chain length = source's own
                                      # literal exponent 3, unchanged since
                                      # it's < the radical index)
b = g.rel("mul", t2, s2)             # 8*27=216
ans = g.rel("add", c2, b)            # a(=c2 directly) + b = 218
add(636, g.factors, ans, 218,
    f"Consider the numbers {L(c2)}, {L(c3)}, {L(c7)}, {L(c4)}, {L(rem2)}, "
    f"{L(t1)}, {L(t2)}, {L(s1)}, {L(s2)}, {L(b)}, {L(ans)}. {L(c2)} is 2. "
    f"{L(c3)} is 3. {L(c7)} is 7. {L(c4)} is 4. {L(c7)} exceeds {L(c4)} "
    f"by {L(rem2)}. {L(c2)} times {L(c2)} equals {L(t1)}. {L(t1)} times "
    f"{L(c2)} equals {L(t2)}. {L(c3)} times {L(c3)} equals {L(s1)}. "
    f"{L(s1)} times {L(c3)} equals {L(s2)}. {L(t2)} times {L(s2)} equals "
    f"{L(b)}. {L(c2)} plus {L(b)} equals {L(ans)}. What is {L(ans)}?",
    "LAW 5/6/8 TENSION (flagged for the wheel, sharpest item this "
    "tranche): 4th-root(2^7*3^3) = a*4th-root(b) via the named radical-"
    "extraction identity nth-root(x^(qn+r)) = x^q * nth-root(x^r), "
    "0<=r<n. Verified EXTERNALLY (not in-graph, no inequality primitive "
    "exists, same domain-restriction-by-hand class as tranche3's [525]) "
    "that floor(7/4)=1 since 4<=7<8: this means exactly ONE full factor "
    "of 2^4 pulls out, so a=2^1=2 (the base itself, no exponentiation "
    "needed since the quotient is exactly 1) -- if the exponents had "
    "given a different quotient this specific rendering (a=c2 directly) "
    "would NOT generalize, a genuine counterfactual-test gap acknowledged "
    "here rather than papered over. The remaining exponent for base 2 "
    "(rem2=7-4=3) IS rendered in-graph (Law 8 bookkeeping: every "
    "expressible operation computing the transform's arguments is "
    "shown), and its VALUE determines the chain length for 2^3 (fixed-"
    "small-exponent technique, matches tranche3's [560]); 3^3 uses the "
    "source's own unchanged literal exponent 3 (since 3<4, all of 3^3 "
    "stays inside the radical, also externally verified). b=2^3*3^3=216 "
    "computed directly (avoiding ever forming the cap-busting 3456), "
    "final answer is the given base (a) plus the derived b.")

# 638: sum of first 10 odd positive integers -> 100. Named identity: sum of
# first n odd integers = n^2.
g = G()
c10 = g.given(10)
ans = g.rel("mul", c10, c10)
add(638, g.factors, ans, 100,
    f"Consider the numbers {L(c10)}, {L(ans)}. {L(c10)} is 10. {L(c10)} "
    f"times {L(c10)} equals {L(ans)}. What is {L(ans)}?",
    "sum-of-first-n-odd-integers identity (named: 1+3+...+(2n-1) = n^2), "
    "applied at n=10; matches the difference-of-squares/arithmetic-"
    "identity family used elsewhere (tranche3's [583]).")

# 643: (81)^(1/2)=3^m -- SKIP
skip(643,
     "sqrt(81)=9 (fine, search-based root extraction). But 3^m=9 requires "
     "solving for the UNKNOWN exponent m -- the base (3) matches the "
     "target (9=3^2) only via a hidden discrete-log recognition (9=3^2), "
     "the same 'abstract exponent bookkeeping outside the value domain' "
     "family as [589]/[621] above and tranche3's [544]/[577]. Distinct "
     "from [601]'s render: there the exponent was a KNOWN source literal "
     "being evaluated; here m is the UNKNOWN being searched for, which "
     "has no primitive.")

# 644: fraction, numerator=2, add 5 to both num/denom -> value 1/2; find
# original denominator -> 9.
g = G()
c2 = g.given(2)
c5 = g.given(5)
newNum = g.rel("add", c2, c5)        # 7
c2b = g.given(2)                     # denominator of the NEW fraction's
                                      # value 1/2
newDenom = g.rel("mul", c2b, newNum) # 2*7=14
origDenom = g.rel("sub", newDenom, c5)  # 14-5=9
add(644, g.factors, origDenom, 9,
    f"Consider the numbers {L(c2)}, {L(c5)}, {L(newNum)}, {L(c2b)}, "
    f"{L(newDenom)}, {L(origDenom)}. {L(c2)} is 2. {L(c5)} is 5. "
    f"{L(c2)} plus {L(c5)} equals {L(newNum)}. {L(c2b)} is 2. {L(c2b)} "
    f"times {L(newNum)} equals {L(newDenom)}. {L(newDenom)} exceeds "
    f"{L(c5)} by {L(origDenom)}. What is {L(origDenom)}?",
    "cross-multiplication (Worked Example C style): new_num/new_denom = "
    "1/2 cross-multiplies to new_denom = 2*new_num directly (a plain "
    "multiplication since new_denom is the 'result' side of the cross-"
    "product, no search needed); original denominator recovered by "
    "subtracting the same 5 that was added.")

# 645: distance between (-2,4) and (3,-8) -> 13. Distance formula (named),
# magnitude-fold for both coordinate deltas (signs irrelevant once
# squared).
g = G()
c5 = g.given(5)                      # |dx| = |3-(-2)|
c12 = g.given(12)                    # |dy| = |-8-4|
dx2 = g.rel("mul", c5, c5)           # 25
dy2 = g.rel("mul", c12, c12)         # 144
sumSq = g.rel("add", dx2, dy2)       # 169
dist = g.free()
g.rel("mul", dist, dist, sumSq)      # dist^2=169 -> dist=13
add(645, g.factors, dist, 13,
    f"Consider the numbers {L(c5)}, {L(c12)}, {L(dx2)}, {L(dy2)}, "
    f"{L(sumSq)}, {L(dist)}. {L(c5)} is 5. {L(c5)} times {L(c5)} equals "
    f"{L(dx2)}. {L(c12)} is 12. {L(c12)} times {L(c12)} equals {L(dy2)}. "
    f"{L(dx2)} plus {L(dy2)} equals {L(sumSq)}. {L(dist)} times "
    f"{L(dist)} equals {L(sumSq)}. What is {L(dist)}?",
    "THEOREM-APPLICATION (named: Euclidean distance formula), degree: "
    "transforms into 'square both coordinate deltas and sum, then take "
    "the root'; magnitude-fold (technique 3) for both deltas since signs "
    "vanish once squared, matches tranche3's [538] precedent closely. "
    "Distance found via search-based root extraction (technique 1).")

# 650: a*b = 3a+4b-ab custom operator, 5*2 -> 13. Direct computation on
# source literals.
g = G()
c3 = g.given(3)
c5 = g.given(5)
t1 = g.rel("mul", c3, c5)            # 15
c4 = g.given(4)
c2 = g.given(2)
t2 = g.rel("mul", c4, c2)            # 8
sum1 = g.rel("add", t1, t2)          # 23
prod = g.rel("mul", c5, c2)          # 10
ans = g.rel("sub", sum1, prod)       # 13
add(650, g.factors, ans, 13,
    f"Consider the numbers {L(c3)}, {L(c5)}, {L(t1)}, {L(c4)}, {L(c2)}, "
    f"{L(t2)}, {L(sum1)}, {L(prod)}, {L(ans)}. {L(c3)} is 3. {L(c5)} is "
    f"5. {L(c3)} times {L(c5)} equals {L(t1)}. {L(c4)} is 4. {L(c2)} is "
    f"2. {L(c4)} times {L(c2)} equals {L(t2)}. {L(t1)} plus {L(t2)} "
    f"equals {L(sum1)}. {L(c5)} times {L(c2)} equals {L(prod)}. "
    f"{L(sum1)} exceeds {L(prod)} by {L(ans)}. What is {L(ans)}?",
    "direct computation of the source's own custom-operator definition "
    "(3a+4b-ab at a=5,b=2), every coefficient and operand a source "
    "literal, no free variables needed.")

# 652: repeated floor-division by 2, count steps to reach 1 -- SKIP
skip(652,
     "Query asks for the COUNT of halving-with-floor steps (6), not a "
     "final value -- no counting/iteration-count primitive exists (same "
     "gap as [625] above). Even setting the counting problem aside, the "
     "computation itself needs floor division TWICE at non-exact steps "
     "(25->12 and 3->1, both odd numerators), but only ONE fdiv is "
     "allowed per item (rule 3); repeated floor-halving collapses to a "
     "single fdiv by 2^6=64 (floor(100/64)=1), but k=64 fails fdiv's "
     "single-digit (2-9) requirement. No path renders the count under "
     "the current primitive set.")

# 655: positive difference between (6^2+6^2)/6 and (6^2*6^2)/6 -> 204.
# Second term simplified algebraically (6^4/6=6^3=216) to avoid the
# cap-busting 1296 intermediate (rearrangement, Law 4, named, matches
# tranche3's [525] cap-avoidance family).
g = G()
c6 = g.given(6)
sq = g.rel("mul", c6, c6)            # 36
sum2 = g.rel("add", sq, sq)          # 72
term1 = g.fdiv(sum2, 6)              # 12  (ONE fdiv)
term2 = g.rel("mul", sq, c6)         # 6^3 = 216 (= 6^4/6, simplified
                                      # in-graph rather than dividing the
                                      # over-cap 1296)
ans = g.rel("sub", term2, term1)     # 216-12=204
add(655, g.factors, ans, 204,
    f"Consider the numbers {L(c6)}, {L(sq)}, {L(sum2)}, {L(term1)}, "
    f"{L(term2)}, {L(ans)}. {L(c6)} is 6. {L(c6)} times {L(c6)} equals "
    f"{L(sq)}. {L(sq)} plus {L(sq)} equals {L(sum2)}. When {L(sum2)} is "
    f"divided by 6, the quotient is {L(term1)}. {L(sq)} times {L(c6)} "
    f"equals {L(term2)}. {L(term2)} exceeds {L(term1)} by {L(ans)}. "
    f"What is {L(ans)}?",
    "REARRANGEMENT (named: (6^2*6^2)/6 = 6^4/6 = 6^3 algebraically, "
    "avoiding the cap-busting 1296=6^2*6^2 intermediate by never forming "
    "it -- same cap-avoidance family as tranche3's [525]), degree: the "
    "simplification is exact (not an approximation), the graph still "
    "performs the transformed problem's real multiplication (6^3 via "
    "chained mult). First term uses the one allowed fdiv (72/6=12).")

# 656: sqrt(1+sqrt(2y-3))=sqrt(6) -> y=14. Square both sides twice
# (rearrangement, named).
g = G()
c6 = g.given(6)
c1 = g.given(1)
s = g.rel("sub", c6, c1)             # sqrt(2y-3) = 6-1 = 5
s2 = g.rel("mul", s, s)              # 2y-3 = 25
c3 = g.given(3)
t = g.rel("add", s2, c3)             # 2y = 28
c2 = g.given(2)
y = g.free()
g.rel("mul", c2, y, t)               # 2y=28 -> y=14
add(656, g.factors, y, 14,
    f"Consider the numbers {L(c6)}, {L(c1)}, {L(s)}, {L(s2)}, {L(c3)}, "
    f"{L(t)}, {L(c2)}, {L(y)}. {L(c6)} is 6. {L(c1)} is 1. {L(c6)} "
    f"exceeds {L(c1)} by {L(s)}. {L(s)} times {L(s)} equals {L(s2)}. "
    f"{L(c3)} is 3. {L(s2)} plus {L(c3)} equals {L(t)}. {L(c2)} is 2. "
    f"{L(c2)} times {L(y)} equals {L(t)}. What is {L(y)}?",
    "REARRANGEMENT (named: squaring both sides twice to clear the nested "
    "radical -- sqrt(1+sqrt(2y-3))=sqrt(6) -> 1+sqrt(2y-3)=6 -> "
    "sqrt(2y-3)=5 -> 2y-3=25), degree: a standard necessary sequence of "
    "steps to reach a polynomial form; the graph performs each cleared "
    "equation's arithmetic, final step by multiplicative inversion.")

# 660: sum of integers >3 and <12 (i.e. 4..11) -> 60. Arithmetic-series
# identity, matches [603]'s technique.
g = G()
c4 = g.given(4)
c11 = g.given(11)
diffAB = g.rel("sub", c11, c4)       # 7
c1 = g.given(1)
n = g.rel("add", diffAB, c1)         # 8 (count of integers)
sumEnds = g.rel("add", c4, c11)      # 15
total = g.rel("mul", n, sumEnds)     # 120
ans = g.fdiv(total, 2)               # 60  (ONE fdiv)
add(660, g.factors, ans, 60,
    f"Consider the numbers {L(c4)}, {L(c11)}, {L(diffAB)}, {L(c1)}, "
    f"{L(n)}, {L(sumEnds)}, {L(total)}, {L(ans)}. {L(c4)} is 4. "
    f"{L(c11)} is 11. {L(c11)} exceeds {L(c4)} by {L(diffAB)}. {L(c1)} "
    f"is 1. {L(diffAB)} plus {L(c1)} equals {L(n)}. {L(c4)} plus "
    f"{L(c11)} equals {L(sumEnds)}. {L(n)} times {L(sumEnds)} equals "
    f"{L(total)}. When {L(total)} is divided by 2, the quotient is "
    f"{L(ans)}. What is {L(ans)}?",
    "arithmetic-series-sum identity (named: sum of consecutive integers "
    "a..b = count*(a+b)/2, count=b-a+1), matches [603]'s technique in "
    "this tranche; 'greater than 3 and less than 12' lexically "
    "explicated to the endpoints 4 and 11 (both KNOWN, derivable "
    "boundary integers of the stated open interval). One fdiv (k=2).")

# 661: 7 bowling balls = 3 canoes; 2 canoes = 56 lbs; find 1 ball's weight
# -> 12. Chained multiplicative inversion (Worked Example C), twice, ZERO
# fdivs (avoids the one-fdiv budget entirely since both divisions are via
# search).
g = G()
c56 = g.given(56)
c2 = g.given(2)
canoeW = g.free()
g.rel("mul", c2, canoeW, c56)        # 2*canoeW=56 -> canoeW=28
c3 = g.given(3)
threeCanoes = g.rel("mul", c3, canoeW)  # 3*28=84
c7 = g.given(7)
ball = g.free()
g.rel("mul", c7, ball, threeCanoes)  # 7*ball=84 -> ball=12
add(661, g.factors, ball, 12,
    f"Consider the numbers {L(c56)}, {L(c2)}, {L(canoeW)}, {L(c3)}, "
    f"{L(threeCanoes)}, {L(c7)}, {L(ball)}. {L(c56)} is 56. {L(c2)} is "
    f"2. {L(c2)} times {L(canoeW)} equals {L(c56)}. {L(c3)} is 3. "
    f"{L(c3)} times {L(canoeW)} equals {L(threeCanoes)}. {L(c7)} is 7. "
    f"{L(c7)} times {L(ball)} equals {L(threeCanoes)}. What is "
    f"{L(ball)}?",
    "multiplicative inversion (Worked Example C), chained twice: one "
    "canoe's weight found from the stated pair total, then the "
    "3-canoe/7-ball equivalence inverted for one ball's weight. Zero "
    "fdivs (both divisions land exactly on nice quotients via search, so "
    "neither needed the fdiv budget).")

# 663: a nabla b = 2+b^a. (1 nabla 2) nabla 3 -> 83. First application:
# exponent a=1 means b^1=b trivially (identity). Second application uses
# the FIRST result (=4, a genuinely small known value once computed) as
# the exponent for a fixed small-exponent chain (technique used in
# tranche3's [560]/[498]).
g = G()
c2const = g.given(2)
b1 = g.given(2)
firstRes = g.rel("add", c2const, b1)  # 2+2^1 = 2+2 = 4 (b^1=b, trivial)
b2 = g.given(3)
u1 = g.rel("mul", b2, b2)             # 3^2=9
u2 = g.rel("mul", u1, b2)             # 3^3=27
u3 = g.rel("mul", u2, b2)             # 3^4=81  (chain length matches
                                       # firstRes's value, 4, verified by
                                       # hand from the first application)
ans = g.rel("add", c2const, u3)       # 2+81=83
add(663, g.factors, ans, 83,
    f"Consider the numbers {L(c2const)}, {L(b1)}, {L(firstRes)}, "
    f"{L(b2)}, {L(u1)}, {L(u2)}, {L(u3)}, {L(ans)}. {L(c2const)} is 2. "
    f"{L(b1)} is 2. {L(c2const)} plus {L(b1)} equals {L(firstRes)}. "
    f"{L(b2)} is 3. {L(b2)} times {L(b2)} equals {L(u1)}. {L(u1)} times "
    f"{L(b2)} equals {L(u2)}. {L(u2)} times {L(b2)} equals {L(u3)}. "
    f"{L(c2const)} plus {L(u3)} equals {L(ans)}. What is {L(ans)}?",
    "fixed small-exponent direct computation (matches tranche3's "
    "[560]/[498] precedent), applied to the source's own custom operator "
    "twice: FIRST application (1 nabla 2) has exponent a=1, so b^1=b is "
    "the trivial identity (no chain needed), giving firstRes=2+2=4. "
    "SECOND application (4 nabla 3) needs 3^4=81; the chain length "
    "(three multiplications) is fixed by the annotator matching the "
    "FIRST result's known value (4), the same annotator-determined-"
    "chain-length technique already established for repeated/nested "
    "known-exponent items -- not re-derived from firstRes's graph value "
    "by any counting primitive (none exists).")

# 669: Betty flour f: f >= 6+sugar/2 AND f <= 2*sugar; least sugar s with a
# feasible f -- rendered via the TIGHT boundary equality (least s is where
# the two bounds coincide: 6+s/2=2s -> s=4).
g = G()
c6 = g.given(6)
c2 = g.given(2)
c12 = g.rel("mul", c6, c2)           # 6*2=12 (clearing the /2 by *2)
c4 = g.rel("mul", c2, c2)            # 2*2=4  (RHS multiplier after *2)
s = g.free()
t1 = g.rel("add", c12, s)            # 12+s
g.rel("mul", c4, s, t1)              # 4s = 12+s  (shared result var
                                      # forces both sides equal)
add(669, g.factors, s, 4,
    f"Consider the numbers {L(c6)}, {L(c2)}, {L(c12)}, {L(c4)}, {L(s)}, "
    f"{L(t1)}. {L(c6)} is 6. {L(c2)} is 2. {L(c6)} times {L(c2)} equals "
    f"{L(c12)}. {L(c2)} times {L(c2)} equals {L(c4)}. {L(c12)} plus "
    f"{L(s)} equals {L(t1)}. {L(c4)} times {L(s)} equals {L(t1)}. What "
    f"is {L(s)}?",
    "LAW 6 TENSION (flagged for the wheel): the source states an "
    "INEQUALITY system (f>=6+s/2 AND f<=2s); no inequality/feasibility "
    "primitive exists. Rendered via the TIGHT-BOUNDARY equality "
    "(6+s/2=2s), i.e. the least s for which the feasible range is "
    "non-empty is exactly where the two bounds coincide -- a real, "
    "standard technique for 'least value satisfying two opposing linear "
    "bounds' problems, but the graph does not itself verify infeasibility "
    "for s<4 (no primitive could); the equality-at-the-boundary "
    "assumption is imported and its correctness (that s=4 is genuinely "
    "the least INTEGER feasible s, not merely the algebraic crossing "
    "point) was checked externally by hand. Cleared the /2 by "
    "multiplying both sides by 2 (rearrangement, Law 4, named) to stay "
    "integer; the CSP still searches s against the two-sided system via "
    "a shared result var (direct system encoding, technique 2).")

# 670: spade a-b=|a-b|. 2 spade (4 spade 7) -> 1. Both sub-magnitudes
# derivable directly since both operands are always concrete known
# literals at each application (never an unknown), so the "larger minus
# smaller" ordering is a plain fact-check, not an unprovable assumption.
g = G()
c4 = g.given(4)
c7 = g.given(7)
inner = g.rel("sub", c7, c4)         # |4-7| = 3
c2 = g.given(2)
outer = g.rel("sub", inner, c2)      # |2-3| = 1
add(670, g.factors, outer, 1,
    f"Consider the numbers {L(c4)}, {L(c7)}, {L(inner)}, {L(c2)}, "
    f"{L(outer)}. {L(c4)} is 4. {L(c7)} is 7. {L(c7)} exceeds {L(c4)} by "
    f"{L(inner)}. {L(c2)} is 2. {L(inner)} exceeds {L(c2)} by {L(outer)}. "
    f"What is {L(outer)}?",
    "direct computation of the source's own custom absolute-difference "
    "operator, applied twice; the subtraction ORDER at each step (which "
    "operand is larger) is a plain fact-check on concrete literals "
    "(4 vs 7, then 3 vs 2), never an unprovable assumption about an "
    "unknown, so this is NOT an ordering-law (Law 6) concern -- a "
    "variant with the operands swapped inside the operator's definition "
    "would still be resolved the same way (compare the two known "
    "numbers), passing the counterfactual test.")

# 672: 100^3=10^x -- SKIP
skip(672,
     "100=10^2 (a hidden base-conversion, discrete-log-adjacent "
     "recognition) -> x=6, same 'abstract exponent bookkeeping' family as "
     "[589]/[621]/[643] above and tranche3's [544]/[577]. Direct "
     "value-space computation (100^3=1,000,000) blows the cap by four "
     "orders of magnitude. Both routes blocked.")

# 673: coefficient of x^2 in 4(x-x^3)-3(x^2-x^3+x^5)+2(4x^2-x^9) -> 5.
# Coefficient extraction: only the x^2-bearing terms are rendered (Law 3/4:
# every unrelated term genuinely contributes zero to the x^2 coefficient,
# verified by hand, not skipped as a shortcut to the final number).
g = G()
c3 = g.given(3)                      # from -3*(x^2-...)
c2 = g.given(2)                      # outer coefficient of the third group
c4 = g.given(4)                      # inner x^2 coefficient, "4x^2"
term2 = g.rel("mul", c2, c4)         # 2*4=8
ans = g.rel("sub", term2, c3)        # 8-3=5
add(673, g.factors, ans, 5,
    f"Consider the numbers {L(c3)}, {L(c2)}, {L(c4)}, {L(term2)}, "
    f"{L(ans)}. {L(c3)} is 3. {L(c2)} is 2. {L(c4)} is 4. {L(c2)} times "
    f"{L(c4)} equals {L(term2)}. {L(term2)} exceeds {L(c3)} by "
    f"{L(ans)}. What is {L(ans)}?",
    "LAW 3/4 TENSION (light, flagged for completeness): coefficient "
    "extraction from a multi-term polynomial expression -- only the "
    "x^2-bearing pieces are rendered (the -3 coefficient from the second "
    "bracket's x^2 term, and 2*4 from the third bracket's 4x^2 term "
    "times its outer 2); the x, x^3, x^5, x^9 terms genuinely contribute "
    "ZERO to the x^2 coefficient (verified by hand expansion) and are "
    "correctly absent, not shortcut around. The graph still performs "
    "real arithmetic (2*4=8, 8-3=5) on the source's own coefficients.")

# 674: half of |18^2-16^2| -> 34. Difference-of-squares identity (named,
# matches [583] precedent) to avoid the cap-busting 18^2=324.
g = G()
c18 = g.given(18)
c16 = g.given(16)
diff = g.rel("sub", c18, c16)        # 2
sum1 = g.rel("add", c18, c16)        # 34
prod = g.rel("mul", diff, sum1)      # 68
ans = g.fdiv(prod, 2)                # 34  (ONE fdiv)
add(674, g.factors, ans, 34,
    f"Consider the numbers {L(c18)}, {L(c16)}, {L(diff)}, {L(sum1)}, "
    f"{L(prod)}, {L(ans)}. {L(c18)} is 18. {L(c16)} is 16. {L(c18)} "
    f"exceeds {L(c16)} by {L(diff)}. {L(c18)} plus {L(c16)} equals "
    f"{L(sum1)}. {L(diff)} times {L(sum1)} equals {L(prod)}. When "
    f"{L(prod)} is divided by 2, the quotient is {L(ans)}. What is "
    f"{L(ans)}?",
    "difference-of-squares identity (named: a^2-b^2=(a-b)(a+b), matches "
    "tranche3's [583] precedent), used specifically to avoid 18^2=324 "
    "exceeding the cap; the graph performs the transformed problem's "
    "arithmetic in full. One fdiv (k=2, for the 'half of').")

# 678: x^2-3x+9=x+41 -> x^2-4x-32=0; positive difference between roots ->
# 12. Roots have OPPOSITE signs (product=-32<0): direct system on r1
# (positive) and M2 (magnitude of the negative root).
g = G()
c9 = g.given(9)
c41 = g.given(41)
qmag = g.rel("sub", c41, c9)         # 41-9=32 (magnitude of the constant
                                      # term after moving everything left)
c3 = g.given(3)
c1 = g.given(1)
bmag = g.rel("add", c3, c1)          # 3+1=4 (combined x-coefficient
                                      # magnitude)
M2 = g.free()
r1 = g.free()
g.rel("sub", r1, M2, bmag)           # r1 - M2 = 4  (r1 positive, M2 = the
                                      # negative root's magnitude)
g.rel("mul", r1, M2, qmag)           # r1*M2 = 32
diffAns = g.rel("add", r1, M2)       # positive difference r1-(-M2)=r1+M2
add(678, g.factors, diffAns, 12,
    f"Consider the numbers {L(c9)}, {L(c41)}, {L(qmag)}, {L(c3)}, "
    f"{L(c1)}, {L(bmag)}, {L(M2)}, {L(r1)}, {L(diffAns)}. {L(c9)} is 9. "
    f"{L(c41)} is 41. {L(c41)} exceeds {L(c9)} by {L(qmag)}. {L(c3)} is "
    f"3. {L(c1)} is 1. {L(c3)} plus {L(c1)} equals {L(bmag)}. {L(r1)} "
    f"exceeds {L(M2)} by {L(bmag)}. {L(r1)} times {L(M2)} equals "
    f"{L(qmag)}. {L(r1)} plus {L(M2)} equals {L(diffAns)}. What is "
    f"{L(diffAns)}?",
    "LAW 3/6 TENSION (flagged for the wheel): moving all terms to one "
    "side gives x^2-4x-32=0 (b-magnitude=3+1=4 from combining -3x and "
    "-x; constant magnitude=41-9=32 from moving 41 and -9 to the same "
    "side); the roots have OPPOSITE signs (product=-32, negative) -- "
    "verified externally by solving the actual roots (8, -4), same class "
    "of externally-verified sign-structure fact as tranche3's "
    "[542]/[631]-style items and [631] above. Direct system encoding "
    "(technique 2): r1 (the positive root) and M2 (the negative root's "
    "magnitude) jointly constrained by r1-M2=4 and r1*M2=32; the "
    "positive difference is r1+M2 by construction (r1-(-M2)).")

# 679: sum of solutions to |2n-7|=3 -> 7. Both cases rendered explicitly
# (genuine two-root computation, not a shortcut using the algebraic fact
# that the sum is independent of the RHS constant).
g = G()
c7 = g.given(7)
c3 = g.given(3)
c2 = g.given(2)
rhs1 = g.rel("add", c7, c3)          # 10  (2n-7=3 -> 2n=10)
n1 = g.free()
g.rel("mul", c2, n1, rhs1)           # 2*n1=10 -> n1=5
rhs2 = g.rel("sub", c7, c3)          # 4   (2n-7=-3 -> 2n=4)
n2 = g.free()
g.rel("mul", c2, n2, rhs2)           # 2*n2=4 -> n2=2
ans = g.rel("add", n1, n2)           # 7
add(679, g.factors, ans, 7,
    f"Consider the numbers {L(c7)}, {L(c3)}, {L(c2)}, {L(rhs1)}, "
    f"{L(n1)}, {L(rhs2)}, {L(n2)}, {L(ans)}. {L(c7)} is 7. {L(c3)} is 3. "
    f"{L(c2)} is 2. {L(c7)} plus {L(c3)} equals {L(rhs1)}. {L(c2)} times "
    f"{L(n1)} equals {L(rhs1)}. {L(c7)} exceeds {L(c3)} by {L(rhs2)}. "
    f"{L(c2)} times {L(n2)} equals {L(rhs2)}. {L(n1)} plus {L(n2)} "
    f"equals {L(ans)}. What is {L(ans)}?",
    "direct system encoding (technique 2), both absolute-value cases "
    "(2n-7=3 and 2n-7=-3) rendered explicitly and solved by "
    "multiplicative inversion, then genuinely summed -- passes the "
    "counterfactual test (a variant asking for the PRODUCT of solutions "
    "would still be correctly answerable by changing the final op, since "
    "both n1=5 and n2=2 are real derived values, not a shortcut using "
    "the algebraic fact that sum-of-roots is independent of the RHS "
    "constant).")

# 682: completing the square x^2-8x+8=0 -> (x+b)^2=c, find b+c -> 4.
# b is NEGATIVE (magnitude-fold, technique 3): b=-4, c=8, b+c=c-|b|.
g = G()
c8coef = g.given(8)                  # coefficient magnitude, "-8x"
half = g.fdiv(c8coef, 2)             # 4  (ONE fdiv) = |b|
halfSq = g.rel("mul", half, half)    # 16
cVal = g.rel("sub", halfSq, c8coef)  # 16-8=8 (=c)
ans = g.rel("sub", cVal, half)       # 8-4=4 (=c-|b|=c+b since b=-|b|)
add(682, g.factors, ans, 4,
    f"Consider the numbers {L(c8coef)}, {L(half)}, {L(halfSq)}, "
    f"{L(cVal)}, {L(ans)}. {L(c8coef)} is 8. When {L(c8coef)} is "
    f"divided by 2, the quotient is {L(half)}. {L(half)} times "
    f"{L(half)} equals {L(halfSq)}. {L(halfSq)} exceeds {L(c8coef)} by "
    f"{L(cVal)}. {L(cVal)} exceeds {L(half)} by {L(ans)}. What is "
    f"{L(ans)}?",
    "REARRANGEMENT (named: completing the square, matches tranche3's "
    "[528] and [614] above precedent) + magnitude-fold (technique 3): "
    "x^2-8x+16=(x-4)^2 means b=-4 in the (x+b)^2 form (negative), so the "
    "query b+c is rendered as c-|b| (=8-4=4) rather than a direct add, "
    "since |b| is what the graph actually holds. One fdiv (k=2).")

# 683: p,q inversely proportional; p=25 at q=6; find p at q=15 -> 10.
# Inverse-variation identity, matches tranche3's [561]/[562].
g = G()
c25 = g.given(25)
c6 = g.given(6)
k = g.rel("mul", c25, c6)            # 150
c15 = g.given(15)
newP = g.free()
g.rel("mul", c15, newP, k)           # 15*newP=150 -> newP=10
add(683, g.factors, newP, 10,
    f"Consider the numbers {L(c25)}, {L(c6)}, {L(k)}, {L(c15)}, "
    f"{L(newP)}. {L(c25)} is 25. {L(c6)} is 6. {L(c25)} times {L(c6)} "
    f"equals {L(k)}. {L(c15)} is 15. {L(c15)} times {L(newP)} equals "
    f"{L(k)}. What is {L(newP)}?",
    "inverse-variation identity p*q=k (constant), matches tranche3's "
    "[561]/[562] precedent; k found from the first pair, new p found by "
    "multiplicative inversion (Worked Example C) from the second q.")

# 684: line through (-2,0),(0,2), y=mx+b, find m+b -> 3. Slope formula
# (named); b taken DIRECTLY from the point (0,2) since x=0 there (light
# Law 1/2 flag, matches tranche3's [533] precedent).
g = G()
c2y = g.given(2)                     # rise: 2-0
c2x = g.given(2)                     # run: 0-(-2)
m = g.free()
g.rel("mul", c2x, m, c2y)            # 2m=2 -> m=1
bVal = g.given(2)                    # y-intercept: the point (0,2) IS the
                                      # y-intercept directly (x=0 there)
ans = g.rel("add", m, bVal)          # 1+2=3
add(684, g.factors, ans, 3,
    f"Consider the numbers {L(c2y)}, {L(c2x)}, {L(m)}, {L(bVal)}, "
    f"{L(ans)}. {L(c2y)} is 2. {L(c2x)} is 2. {L(c2x)} times {L(m)} "
    f"equals {L(c2y)}. {L(bVal)} is 2. {L(m)} plus {L(bVal)} equals "
    f"{L(ans)}. What is {L(ans)}?",
    "LAW 1/2 TENSION (light, flagged for completeness, matches "
    "tranche3's [533] precedent): THEOREM-APPLICATION (named: slope "
    "formula) finds m via multiplicative inversion from the two points' "
    "rise/run; b is taken DIRECTLY from the source's own point (0,2) "
    "since a point with x=0 IS the y-intercept by definition -- no "
    "computation needed, a genuine reconstruction-co-test pass (the "
    "literal 2 is present and used for exactly the role it plays), not a "
    "laundered constant.")

print()
print(f"TOTAL drafted (pre-checks passing): {len(rows)}")
print(f"FAILS: {fails}")
print(f"SKIPS: {[s[0] for s in skips]}")

with open('/home/bryce/mycelium/.cache/book8_t4_prose_pairs_draft.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

with open('/home/bryce/mycelium/.cache/book8_t4_skips.json', 'w') as f:
    json.dump(skips, f, indent=2)

print("done, wrote", len(rows), "rows")
