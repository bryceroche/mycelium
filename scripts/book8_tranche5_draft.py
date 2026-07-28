import json, sys, string
sys.path.insert(0, '/home/bryce/mycelium')
sys.path.insert(0, '/home/bryce/mycelium/scripts')
from tta_alg2_dials import solve2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic

SMP = {"n_vars": 24, "m": 300}
LETTERS = string.ascii_lowercase

CANDS = {c["src_idx"]: c for c in
         json.load(open('/home/bryce/mycelium/.cache/book8_candidates_t5.json'))["tranche5"]}


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


def build_row(src_idx, factors, query_var, dialect, notes, watch=None, accommodation=None,
              routing_fact=None, n_vars=24, m=300):
    sol = full_solution(factors, n_vars, m)
    assert sol is not None, f"src {src_idx}: solver failed to find full solution"
    gen = {
        "src_idx": src_idx, "book": 8, "tranche": 5, "floor": "prime", "fs": True,
        "dialect": dialect, "gate": "PENDING:5view-vote+key", "generation": "21",
        "notes": notes,
    }
    if watch:
        gen["watch"] = watch
    if accommodation:
        gen["accommodation"] = accommodation
    if routing_fact:
        gen["routing_fact"] = routing_fact
    src_text = CANDS[src_idx]["problem"]
    return {
        "text": src_text, "factors": factors, "query_var": query_var,
        "n_vars": n_vars, "m": m, "decisions": [], "mentions": [],
        "solution": sol, "gen": gen,
    }


rows = []
fails = []
skips = []


def add(src_idx, factors, query_var, expect, dialect, notes, watch=None, accommodation=None,
        routing_fact=None):
    ok = check(src_idx, factors, query_var, expect)
    if not ok:
        fails.append(src_idx)
        return
    rows.append(build_row(src_idx, factors, query_var, dialect, notes, watch, accommodation,
                           routing_fact))


def skip(src_idx, reason):
    skips.append((src_idx, reason))
    print(f"[{src_idx}] SKIP: {reason[:70]}...")


# ===========================================================================
# 685: arithmetic sequence a2=17, a5=19 -> a8=21. Step-count-matching
# identity (index gap 5-2=3 equals gap 8-5=3, so the SAME delta that carries
# term2->term5 carries term5->term8): delta=a5-a2, a8=a5+delta. Avoids ever
# needing the fractional common difference (2/3).
g = G()
c17 = g.given(17)
c19 = g.given(19)
delta = g.rel("sub", c19, c17)        # 19-17=2
a8 = g.rel("add", c19, delta)         # 19+2=21
add(685, g.factors, a8, 21,
    f"Consider the numbers {L(c17)}, {L(c19)}, {L(delta)}, {L(a8)}. "
    f"{L(c17)} is 17. {L(c19)} is 19. {L(c19)} exceeds {L(c17)} by "
    f"{L(delta)}. {L(c19)} plus {L(delta)} equals {L(a8)}. What is "
    f"{L(a8)}?",
    "step-count-matching identity (named: in an arithmetic sequence, equal "
    "index gaps carry equal deltas -- term5-term2 spans 3 indices, exactly "
    "matching term8-term5's span, so the SAME delta applies), degree: "
    "avoids ever computing the fractional common difference (2/3 per "
    "index); a8=a5+(a5-a2)=2*a5-a2 is an exact algebraic identity for this "
    "specific index alignment, not a shortcut around genuine computation "
    "-- matches the geometric-sequence analogue used at [762] this "
    "tranche.")

# 686: cylindrical tank 1/5 full; +3L -> 1/4 full. Find full capacity -> 60.
# C=20x scaling: 20=4*5 reconstructed in-graph (Law 2, LCM of the source's
# own two denominators), 1/5 full=4x, 1/4 full=5x, 4x+3=5x.
g = G()
c4 = g.given(4)
c5 = g.given(5)
c20 = g.rel("mul", c4, c5)            # 20 (LCM(4,5), reconstructed in-graph)
x = g.free()
amt1 = g.rel("mul", c4, x)            # 4x = C/5
amt2 = g.rel("mul", c5, x)            # 5x = C/4
c3 = g.given(3)
g.rel("add", amt1, c3, amt2)          # 4x+3=5x
C = g.rel("mul", c20, x)              # C=20x
add(686, g.factors, C, 60,
    f"Consider the numbers {L(c4)}, {L(c5)}, {L(c20)}, {L(x)}, {L(amt1)}, "
    f"{L(amt2)}, {L(c3)}, {L(C)}. {L(c4)} is 4. {L(c5)} is 5. {L(c4)} "
    f"times {L(c5)} equals {L(c20)}. {L(c4)} times {L(x)} equals "
    f"{L(amt1)}. {L(c5)} times {L(x)} equals {L(amt2)}. {L(c3)} is 3. "
    f"{L(amt1)} plus {L(c3)} equals {L(amt2)}. {L(c20)} times {L(x)} "
    f"equals {L(C)}. What is {L(C)}?",
    "REARRANGEMENT (named: clearing the fifths/fourths by a shared unit "
    "x=C/20, where 20=LCM(4,5) is RECONSTRUCTED in-graph as 4*5 since "
    "gcd(4,5)=1 -- Law 2 reconstruction-co-test pass, the literal 20 is "
    "DECOMPOSED from the source's own two denominators, not planted), "
    "degree: direct system encoding (technique 2) on x, forced by "
    "4x+3=5x; capacity found as 20x.")

# 687: N(x)=2sqrt(x), O(x)=x^2. N(O(N(O(N(O(3)))))) -> 24. Direct chain,
# alternating squaring and search-based root extraction (technique 1).
g = G()
c3 = g.given(3)
O1 = g.rel("mul", c3, c3)             # O(3)=9
r1 = g.free()
g.rel("mul", r1, r1, O1)              # sqrt(9): r1*r1=9 -> r1=3
c2 = g.given(2)                       # universal constant "2" in N(x)=2sqrt(x)
N1 = g.rel("mul", c2, r1)             # N(9)=2*3=6
O2 = g.rel("mul", N1, N1)             # O(6)=36
r2 = g.free()
g.rel("mul", r2, r2, O2)              # sqrt(36): r2*r2=36 -> r2=6
N2 = g.rel("mul", c2, r2)             # N(36)=2*6=12
O3 = g.rel("mul", N2, N2)             # O(12)=144
r3 = g.free()
g.rel("mul", r3, r3, O3)              # sqrt(144): r3*r3=144 -> r3=12
N3 = g.rel("mul", c2, r3)             # N(144)=2*12=24
add(687, g.factors, N3, 24,
    f"Consider the numbers {L(c3)}, {L(O1)}, {L(r1)}, {L(c2)}, {L(N1)}, "
    f"{L(O2)}, {L(r2)}, {L(N2)}, {L(O3)}, {L(r3)}, {L(N3)}. {L(c3)} is 3. "
    f"{L(c3)} times {L(c3)} equals {L(O1)}. {L(r1)} times {L(r1)} equals "
    f"{L(O1)}. {L(c2)} is 2. {L(c2)} times {L(r1)} equals {L(N1)}. "
    f"{L(N1)} times {L(N1)} equals {L(O2)}. {L(r2)} times {L(r2)} equals "
    f"{L(O2)}. {L(c2)} times {L(r2)} equals {L(N2)}. {L(N2)} times "
    f"{L(N2)} equals {L(O3)}. {L(r3)} times {L(r3)} equals {L(O3)}. "
    f"{L(c2)} times {L(r3)} equals {L(N3)}. What is {L(N3)}?",
    "direct computation of the source's own function composition, six "
    "nested applications alternating O(x)=x^2 (chained mult) and "
    "N(x)=2sqrt(x) (search-based root extraction, technique 1, times a "
    "universal constant 2 from N's own definition, Law 1); every "
    "intermediate value (9,3,6,36,6,12,144,12,24) stays well under the "
    "300 cap.",
    watch="pointer-collision: c2 (the universal constant '2' from N(x)=2sqrt(x)) "
          "serves as an argument in 3 separate mul factors (one per N application).")

# 690: a-clubsuit-b = (2a/b)*(b/a), always evaluates to 2 for any nonzero
# a,b (source's own operator identity: 2a*b/(a*b)=2). Rendered by ACTUALLY
# building ab and 2ab from the real numbers at each application and
# recovering 2 via multiplicative inversion -- not shortcut to a bare "2".
g = G()
c2const = g.given(2)                  # universal constant "2" in the operator
c3 = g.given(3)
c6 = g.given(6)
ab1 = g.rel("mul", c3, c6)            # 3*6=18
twoab1 = g.rel("mul", c2const, ab1)   # 36
mid1 = g.free()
g.rel("mul", mid1, ab1, twoab1)       # mid1*18=36 -> mid1=2  (=3 clubsuit 6)
c5 = g.given(5)
ab2 = g.rel("mul", c5, mid1)          # 5*2=10
twoab2 = g.rel("mul", c2const, ab2)   # 20
mid2 = g.free()
g.rel("mul", mid2, ab2, twoab2)       # mid2*10=20 -> mid2=2  (=5 clubsuit mid1)
c1 = g.given(1)
ab3 = g.rel("mul", mid2, c1)          # 2*1=2
twoab3 = g.rel("mul", c2const, ab3)   # 4
mid3 = g.free()
g.rel("mul", mid3, ab3, twoab3)       # mid3*2=4 -> mid3=2  (=mid2 clubsuit 1)
add(690, g.factors, mid3, 2,
    f"Consider the numbers {L(c2const)}, {L(c3)}, {L(c6)}, {L(ab1)}, "
    f"{L(twoab1)}, {L(mid1)}, {L(c5)}, {L(ab2)}, {L(twoab2)}, {L(mid2)}, "
    f"{L(c1)}, {L(ab3)}, {L(twoab3)}, {L(mid3)}. {L(c2const)} is 2. "
    f"{L(c3)} is 3. {L(c6)} is 6. {L(c3)} times {L(c6)} equals {L(ab1)}. "
    f"{L(c2const)} times {L(ab1)} equals {L(twoab1)}. {L(mid1)} times "
    f"{L(ab1)} equals {L(twoab1)}. {L(c5)} is 5. {L(c5)} times {L(mid1)} "
    f"equals {L(ab2)}. {L(c2const)} times {L(ab2)} equals {L(twoab2)}. "
    f"{L(mid2)} times {L(ab2)} equals {L(twoab2)}. {L(c1)} is 1. "
    f"{L(mid2)} times {L(c1)} equals {L(ab3)}. {L(c2const)} times "
    f"{L(ab3)} equals {L(twoab3)}. {L(mid3)} times {L(ab3)} equals "
    f"{L(twoab3)}. What is {L(mid3)}?",
    "LAW 3 TENSION (flagged for the wheel): the source's own operator "
    "algebraically simplifies to a constant (2a/b*b/a=2ab/ab=2) "
    "regardless of a,b -- rather than shortcutting to a bare '2' (which "
    "would be Law-3 residue, never touching 3,6,5,1), each application is "
    "rendered as REAL arithmetic on the actual operands (building a*b and "
    "2*a*b from the real numbers, recovering the operator's value via "
    "multiplicative inversion/search), chained three times through the "
    "nested nabla structure. The invariance to a,b is a genuine property "
    "of the SOURCE's own operator definition, not an annotator shortcut; "
    "flagged because the numeric path (18,10,2) does depend on the real "
    "inputs even though the outcome (2) doesn't.",
    watch="pointer-collision: c2const (universal constant '2' from the operator's "
          "own 2a/b term) serves as an argument in 3 separate mul factors (one per "
          "nabla application).")

# 691: distance from origin to (7,-24) -> 25. Direct sum-of-squares would
# need 24^2=576 (over the 300 cap). Instead: d^2-24^2=7^2=49, so
# (d-24)(d+24)=49 with (d+24)-(d-24)=48 -- a difference-and-product system
# (Vieta-family generalization) on lo=d-24, hi=d+24, both <=300.
g = G()
c24 = g.given(24)
c2 = g.given(2)                       # universal constant, doubling
twoC24 = g.rel("mul", c2, c24)        # 48 = (d+24)-(d-24)
c7 = g.given(7)
prod49 = g.rel("mul", c7, c7)         # 49 = (d-24)(d+24)
lo = g.free()
hi = g.free()
g.rel("add", lo, twoC24, hi)          # lo+48=hi
g.rel("mul", lo, hi, prod49)          # lo*hi=49
d = g.rel("sub", hi, c24)             # d = hi-24
add(691, g.factors, d, 25,
    f"Consider the numbers {L(c24)}, {L(c2)}, {L(twoC24)}, {L(c7)}, "
    f"{L(prod49)}, {L(lo)}, {L(hi)}, {L(d)}. {L(c24)} is 24. {L(c2)} is "
    f"2. {L(c2)} times {L(c24)} equals {L(twoC24)}. {L(c7)} is 7. "
    f"{L(c7)} times {L(c7)} equals {L(prod49)}. {L(lo)} plus {L(twoC24)} "
    f"equals {L(hi)}. {L(lo)} times {L(hi)} equals {L(prod49)}. {L(hi)} "
    f"exceeds {L(c24)} by {L(d)}. What is {L(d)}?",
    "REARRANGEMENT (named: cap-avoidance -- direct sum-of-squares needs "
    "24^2=576, exceeding 300, so d^2-24^2=49 is factored as "
    "(d-24)(d+24)=49 instead), NEW TECHNIQUE this tranche: a "
    "DIFFERENCE-and-product system (lo=d-24, hi=d+24, hi-lo=2*24=48 "
    "given, lo*hi=7^2=49 given) -- direct system encoding (technique 2), "
    "a generalization of the Vieta sum-and-product pattern (tranche4's "
    "[631]/[678]) to a KNOWN difference instead of a known sum. The "
    "solver's search finds the unique domain-valid factor pair (1,49) of "
    "49 satisfying the 48-gap, genuine deduction, not asserted.",
    accommodation="add-dup (X plus X self-addition misbinds): the '2*24' gap "
                   "(hi-lo=48) is rendered as c2 times c24, deliberately avoiding "
                   "c24 plus c24.")

# 694: #N=0.5N+1, three iterations on 58 -> 9. Each halving via
# multiplicative inversion (search), NOT fdiv (matches tranche4's [661]
# precedent), keeping the one-fdiv budget entirely unused.
g = G()
c58 = g.given(58)
c2 = g.given(2)
c1 = g.given(1)
h1 = g.free()
g.rel("mul", h1, c2, c58)             # 2*h1=58 -> h1=29
n1 = g.rel("add", h1, c1)             # 29+1=30
h2 = g.free()
g.rel("mul", h2, c2, n1)              # 2*h2=30 -> h2=15
n2 = g.rel("add", h2, c1)             # 15+1=16
h3 = g.free()
g.rel("mul", h3, c2, n2)              # 2*h3=16 -> h3=8
n3 = g.rel("add", h3, c1)             # 8+1=9
add(694, g.factors, n3, 9,
    f"Consider the numbers {L(c58)}, {L(c2)}, {L(c1)}, {L(h1)}, {L(n1)}, "
    f"{L(h2)}, {L(n2)}, {L(h3)}, {L(n3)}. {L(c58)} is 58. {L(c2)} is 2. "
    f"{L(c1)} is 1. {L(h1)} times {L(c2)} equals {L(c58)}. {L(h1)} plus "
    f"{L(c1)} equals {L(n1)}. {L(h2)} times {L(c2)} equals {L(n1)}. "
    f"{L(h2)} plus {L(c1)} equals {L(n2)}. {L(h3)} times {L(c2)} equals "
    f"{L(n2)}. {L(h3)} plus {L(c1)} equals {L(n3)}. What is {L(n3)}?",
    "direct computation of the source's own operator (#N=0.5N+1), "
    "chained three times; each halving found by multiplicative inversion "
    "(Worked Example C) rather than fdiv, matching tranche4's [661] "
    "precedent for avoiding the one-fdiv budget entirely when several "
    "exact divisions are needed.",
    watch="pointer-collision: both c2 (halving multiplier) and c1 (the '+1') "
          "serve as arguments in 3 separate factors each, one per iteration.")

# 696: triangle (2,2),(5,6),(6,2), longest side -> 5. Only the (2,2)-(5,6)
# side (legs 3,4) is rendered; identification that it's the longest is
# externally verified by comparing the three squared side lengths
# (25 vs 17 vs 16), a plain fact-check on concrete derived numbers.
g = G()
c3 = g.given(3)                       # dx = 5-2
c4 = g.given(4)                       # dy = 6-2
sqx = g.rel("mul", c3, c3)            # 9
sqy = g.rel("mul", c4, c4)            # 16
sumSq = g.rel("add", sqx, sqy)        # 25
s = g.free()
g.rel("mul", s, s, sumSq)             # s*s=25 -> s=5
add(696, g.factors, s, 5,
    f"Consider the numbers {L(c3)}, {L(c4)}, {L(sqx)}, {L(sqy)}, "
    f"{L(sumSq)}, {L(s)}. {L(c3)} is 3. {L(c4)} is 4. {L(c3)} times "
    f"{L(c3)} equals {L(sqx)}. {L(c4)} times {L(c4)} equals {L(sqy)}. "
    f"{L(sqx)} plus {L(sqy)} equals {L(sumSq)}. {L(s)} times {L(s)} "
    f"equals {L(sumSq)}. What is {L(s)}?",
    "ROUTING-FACT (Law 13, flagged): the three side lengths of the "
    "triangle are (2,2)-(5,6)=5 [legs 3,4], (5,6)-(6,2)=sqrt(17) "
    "[irrational], (2,2)-(6,2)=4; identifying that the FIRST side is "
    "longest is derivable from quantities the graph could hold (their "
    "squares 25, 17, 16 are all in-domain and directly comparable), but "
    "no MAX/comparison primitive exists to make that selection in-graph, "
    "so the identification was made externally by comparing the three "
    "concrete derived numbers (matches tranche4's [670] 'plain fact-check "
    "on concrete literals' exception to Law 6, extended to a 3-way "
    "comparison). THEOREM-APPLICATION (named: distance formula) for the "
    "chosen side, rendered in full.",
    routing_fact="which of the triangle's 3 sides to render (the longest) is "
                  "determined by comparing the three squared side lengths "
                  "(25 vs 17 vs 16), derivable from graph-held quantities but "
                  "requiring a MAX operation the graph doesn't have.")

# 697: x^2+13x+30=(x+a)(x+b) [a+b=13, ab=30] and x^2+5x-50=(x+b)(x-c)
# [b-c=5, bc=50]. a+b+c -> 18. Vieta system for a,b; c solved from b via
# the shared-var b, doubly constrained (b-c=5 AND bc=50) for genuine
# consistency.
g = G()
c13 = g.given(13)
c30 = g.given(30)
a = g.free()
b = g.free()
g.rel("add", a, b, c13)               # a+b=13
g.rel("mul", a, b, c30)               # a*b=30
c5 = g.given(5)
c50 = g.given(50)
cvar = g.free()
g.rel("add", cvar, c5, b)             # c+5=b  (b-c=5)
g.rel("mul", b, cvar, c50)            # b*c=50
ans = g.rel("add", c13, cvar)         # (a+b)+c = 13+c
add(697, g.factors, ans, 18,
    f"Consider the numbers {L(c13)}, {L(c30)}, {L(a)}, {L(b)}, {L(c5)}, "
    f"{L(c50)}, {L(cvar)}, {L(ans)}. {L(c13)} is 13. {L(c30)} is 30. "
    f"{L(a)} plus {L(b)} equals {L(c13)}. {L(a)} times {L(b)} equals "
    f"{L(c30)}. {L(c5)} is 5. {L(c50)} is 50. {L(cvar)} plus {L(c5)} "
    f"equals {L(b)}. {L(b)} times {L(cvar)} equals {L(c50)}. {L(c13)} "
    f"plus {L(cvar)} equals {L(ans)}. What is {L(ans)}?",
    "direct system encoding (technique 2), TWO systems chained through "
    "the shared variable b: Vieta sum/product pins {a,b}={3,10} (from "
    "x^2+13x+30's factorization), then b's value (10, whichever of "
    "{a,b} the solver assigns it) doubly constrains c via BOTH b-c=5 AND "
    "b*c=50 (over-determined but consistent -- Law 3, leaves the solver "
    "genuine work rather than a single shortcut equation); final sum "
    "a+b+c rendered as the already-constrained (a+b)=13 plus c, since "
    "a+b is FORCED equal to c13 by the first constraint.")

# 702: SKIP -- degree of g(x) is a pure polynomial-degree-cancellation
# argument, no numeric quantity to compute.
skip(702,
     "f(x)=-7x^4+3x^3+x-5; for deg(f+g)=1, g must cancel BOTH the x^4 "
     "and x^3 terms of f, forcing deg(g)=4 -- but this is pure reasoning "
     "about which terms cancel in a polynomial sum, not a numeric "
     "computation. The 'answer' (4) is literally f's own degree with no "
     "other quantity to combine it against -- rendering it would be "
     "Law-3 sole-operation residue (reading off f's degree with zero "
     "graph-derived work). No primitive represents 'degree of a "
     "polynomial' or 'polynomial addition cancels leading terms'.")

# 706: geometric sequence 20, a, 5/4 -> a=5. a^2=20*(5/4)=25 (cross-term
# product of a geometric sequence); a=sqrt(25).
g = G()
c20 = g.given(20)
c5 = g.given(5)
c4 = g.given(4)
num = g.rel("mul", c20, c5)           # 100
aSq = g.fdiv(num, 4)                  # 25  (ONE fdiv)
a = g.free()
g.rel("mul", a, a, aSq)               # a*a=25 -> a=5
add(706, g.factors, a, 5,
    f"Consider the numbers {L(c20)}, {L(c5)}, {L(c4)}, {L(num)}, "
    f"{L(aSq)}, {L(a)}. {L(c20)} is 20. {L(c5)} is 5. {L(c4)} is 4. "
    f"{L(c20)} times {L(c5)} equals {L(num)}. When {L(num)} is divided "
    f"by 4, the quotient is {L(aSq)}. {L(a)} times {L(a)} equals "
    f"{L(aSq)}. What is {L(a)}?",
    "THEOREM-APPLICATION (named: geometric-sequence middle-term "
    "identity, a^2 = first*third), degree: transforms the sequence "
    "condition into a^2=20*(5/4)=25, computed as 20*5=100 then divided "
    "by 4 (source's own denominator), avoiding ever forming the "
    "fraction 5/4 directly; a found by search-based root extraction. "
    "One fdiv (k=4).")

# 707: SKIP -- Vieta sum-of-roots reduces to a bare source literal (12)
# with no genuine graph-derived arithmetic (roots are irrational).
skip(707,
     "z^2=12z-7 -> z^2-12z+7=0; sum of roots by Vieta = 12, but the "
     "individual roots are irrational (z=(12+-sqrt(116))/2, not "
     "representable in the integer domain), so no genuine 2-var system "
     "can be posed (unlike tranche4's [631]/[678] where the roots "
     "THEMSELVES were representable). The sum-of-roots identity here "
     "reduces to echoing the source's own coefficient (12) verbatim -- "
     "Law-3 sole-operation residue, zero transformed arithmetic for the "
     "solver to perform.")

# 710: triangle+q=59, (triangle+q)+q=106 -> triangle=12.
g = G()
c59 = g.given(59)
c106 = g.given(106)
q = g.free()
g.rel("add", c59, q, c106)            # 59+q=106
t = g.rel("sub", c59, q)              # triangle = 59-q
add(710, g.factors, t, 12,
    f"Consider the numbers {L(c59)}, {L(c106)}, {L(q)}, {L(t)}. "
    f"{L(c59)} is 59. {L(c106)} is 106. {L(c59)} plus {L(q)} equals "
    f"{L(c106)}. {L(c59)} exceeds {L(q)} by {L(t)}. What is {L(t)}?",
    "direct system encoding (technique 2): q found by multiplicative-"
    "inversion-style search against the second stated equation "
    "(59+q=106), triangle then found as 59-q from the first.")

# 712: SKIP -- identifying the unique integer satisfying a strict
# inequality; no primitive supports inequality-satisfying-integer search.
skip(712,
     "n^2<2n holds exactly for 0<n<2, so the only integer is n=1 -- this "
     "is an IDENTIFICATION of the unique integer satisfying a strict "
     "inequality, not a computed value. No inequality-satisfying-integer "
     "search primitive exists (same 'no primitive for this class of "
     "question' family as [625] in tranche4); the answer (1) is "
     "structurally coincidental for this specific inequality's bounds "
     "and would not generalize to a rescaled variant, failing the "
     "counterfactual test.")

# 714: -b^2+7b-10>=0 -> b^2-7b+10<=0 -> roots 2,5, greatest b=5. Boundary
# clause (Law 12): binding equality is the quadratic's own roots; the
# GREATER root is pinned via a free nonneg gap (self-resolving ordering,
# matches tranche4's [631]).
g = G()
c7 = g.given(7)
c10 = g.given(10)
rSmall = g.free()
rDiff = g.free()
rBig = g.rel("add", rSmall, rDiff)
g.rel("add", rBig, rSmall, c7)        # rBig+rSmall=7
g.rel("mul", rBig, rSmall, c10)       # rBig*rSmall=10
add(714, g.factors, rBig, 5,
    f"Consider the numbers {L(c7)}, {L(c10)}, {L(rSmall)}, {L(rDiff)}, "
    f"{L(rBig)}. {L(c7)} is 7. {L(c10)} is 10. {L(rSmall)} plus "
    f"{L(rDiff)} equals {L(rBig)}. {L(rBig)} plus {L(rSmall)} equals "
    f"{L(c7)}. {L(rBig)} times {L(rSmall)} equals {L(c10)}. What is "
    f"{L(rBig)}?",
    "LAW 12 TENSION (boundary clause, flagged for the wheel): source "
    "states an INEQUALITY (-b^2+7b-10>=0); the binding constraint is the "
    "quadratic's own two roots (2 and 5), and the greatest FEASIBLE b is "
    "the larger root -- (a) binding constraint named: the downward "
    "parabola is nonneg exactly between its roots, a standard fact "
    "checked by hand; (b) no other constraints to verify; (c) "
    "counterfactual holds: 'least value of b' would query rSmall instead, "
    "genuinely different dialect. Vieta sum/product direct system "
    "(technique 2), larger root self-resolved via a free nonneg gap "
    "(matches tranche4's [631]/[678] self-resolving-ordering pattern).")

# 715: SKIP -- unknown exponent (day count) search, compounded with a
# no-comparison/threshold primitive gap.
skip(715,
     "colony doubles daily from 3; need the first day n with 3*2^n>100. "
     "This is BOTH an unknown-exponent search (n is the unknown exponent "
     "of 2, same discrete-log-adjacent family as [589]/[621]/[643]/[672] "
     "in tranche4 and this tranche's cluster) AND a threshold/comparison "
     "question (first day EXCEEDING 100, no inequality primitive). "
     "Doubly blocked, no path renders it.")

# 716: SKIP -- floor(sqrt(63)), no floor/inequality primitive.
skip(716,
     "floor(sqrt(63))=7 requires finding the integer s with s^2<=63< "
     "(s+1)^2 -- 63 is not a perfect square, and our rel primitives only "
     "express EXACT equalities (a*a=target), never inequalities/floor "
     "bounds. No primitive can express 'largest s with s^2<=63'.")

# 718: (x+4)/(x-2)=3 -> x=5. Direct system (technique 2): both sides of
# the cross-multiplied equation forced equal via a shared result var.
g = G()
c4 = g.given(4)
c2 = g.given(2)
c3 = g.given(3)
x = g.free()
lhs = g.rel("add", x, c4)             # x+4
rhsInner = g.rel("sub", x, c2)        # x-2
g.rel("mul", c3, rhsInner, lhs)       # 3*(x-2) = x+4
add(718, g.factors, x, 5,
    f"Consider the numbers {L(c4)}, {L(c2)}, {L(c3)}, {L(x)}, {L(lhs)}, "
    f"{L(rhsInner)}. {L(c4)} is 4. {L(c2)} is 2. {L(c3)} is 3. {L(x)} "
    f"plus {L(c4)} equals {L(lhs)}. {L(x)} exceeds {L(c2)} by "
    f"{L(rhsInner)}. {L(c3)} times {L(rhsInner)} equals {L(lhs)}. What "
    f"is {L(x)}?",
    "direct system encoding (technique 2): the CSP searches x directly "
    "against the cross-multiplied equation 3*(x-2)=x+4 (both sub-"
    "expressions forced equal via the shared result var lhs), never "
    "pre-solving the linear equation algebraically by hand.")

# 721: pairwise sums 29,46,53 -> total=64. Named identity: sum of pairwise
# sums = 2*(sum of all three).
g = G()
c29 = g.given(29)
c46 = g.given(46)
c53 = g.given(53)
t1 = g.rel("add", c29, c46)           # 75
t2 = g.rel("add", t1, c53)            # 128
ans = g.fdiv(t2, 2)                   # 64  (ONE fdiv)
add(721, g.factors, ans, 64,
    f"Consider the numbers {L(c29)}, {L(c46)}, {L(c53)}, {L(t1)}, "
    f"{L(t2)}, {L(ans)}. {L(c29)} is 29. {L(c46)} is 46. {L(c53)} is 53. "
    f"{L(c29)} plus {L(c46)} equals {L(t1)}. {L(t1)} plus {L(c53)} "
    f"equals {L(t2)}. When {L(t2)} is divided by 2, the quotient is "
    f"{L(ans)}. What is {L(ans)}?",
    "pairwise-sum identity (named: (a+b)+(b+c)+(a+c)=2(a+b+c), a "
    "standard identity verified by direct expansion), applied to the "
    "source's own three stated pairwise sums; the graph performs the "
    "real sum-then-halve arithmetic. One fdiv (k=2).")

# 722: f(x)=2x-3, g(x)=x+1. f(1+g(2)) -> 5. Direct composition.
g = G()
c2arg = g.given(2)
c1off = g.given(1)
g2 = g.rel("add", c2arg, c1off)       # g(2)=2+1=3
c1plus = g.given(1)
inner = g.rel("add", g2, c1plus)      # 1+g(2)=4
c2coef = g.given(2)
t = g.rel("mul", c2coef, inner)       # 2*4=8
c3 = g.given(3)
ans = g.rel("sub", t, c3)             # 8-3=5
add(722, g.factors, ans, 5,
    f"Consider the numbers {L(c2arg)}, {L(c1off)}, {L(g2)}, {L(c1plus)}, "
    f"{L(inner)}, {L(c2coef)}, {L(t)}, {L(c3)}, {L(ans)}. {L(c2arg)} is "
    f"2. {L(c1off)} is 1. {L(c2arg)} plus {L(c1off)} equals {L(g2)}. "
    f"{L(c1plus)} is 1. {L(g2)} plus {L(c1plus)} equals {L(inner)}. "
    f"{L(c2coef)} is 2. {L(c2coef)} times {L(inner)} equals {L(t)}. "
    f"{L(c3)} is 3. {L(t)} exceeds {L(c3)} by {L(ans)}. What is "
    f"{L(ans)}?",
    "direct computation of the source's own function composition "
    "(g(2), then 1+g(2), then f of that), every coefficient a source "
    "literal from f and g's definitions, no free variables needed.")

# 723: A=(0,0), B on y=4, slope AB=2/3 -> sum of B's coords=10.
# Cross-multiplication: 4/bx=2/3 -> bx=6.
g = G()
c4 = g.given(4)
c2 = g.given(2)
c3 = g.given(3)
bx = g.free()
t = g.rel("mul", c4, c3)              # 12
g.rel("mul", c2, bx, t)               # 2*bx=12 -> bx=6
ans = g.rel("add", bx, c4)            # 6+4=10
add(723, g.factors, ans, 10,
    f"Consider the numbers {L(c4)}, {L(c2)}, {L(c3)}, {L(bx)}, {L(t)}, "
    f"{L(ans)}. {L(c4)} is 4. {L(c2)} is 2. {L(c3)} is 3. {L(c4)} times "
    f"{L(c3)} equals {L(t)}. {L(c2)} times {L(bx)} equals {L(t)}. "
    f"{L(bx)} plus {L(c4)} equals {L(ans)}. What is {L(ans)}?",
    "cross-multiplication (Worked Example C style): slope=4/bx=2/3 "
    "clears to 2*bx=4*3=12, bx found by multiplicative inversion; sum of "
    "coordinates adds the source's own y=4.")

# 725: SKIP -- floor(sqrt(12))^2, same floor/inequality gap as [716].
skip(725,
     "floor(sqrt(12))=3 (9<=12<16), same 'no floor/inequality primitive' "
     "gap as [716] above -- 12 is not a perfect square, and our rel "
     "primitives only express exact equalities.")

# 726: (3+x)/(5+x)=(1+x)/(2+x) -> x=1. Direct system, cross-multiplied,
# both products forced equal via shared result var.
g = G()
c3 = g.given(3)
c5 = g.given(5)
c1 = g.given(1)
c2 = g.given(2)
x = g.free()
numA = g.rel("add", c3, x)            # 3+x
denA = g.rel("add", c5, x)            # 5+x
numB = g.rel("add", c1, x)            # 1+x
denB = g.rel("add", c2, x)            # 2+x
prodLeft = g.rel("mul", numA, denB)   # (3+x)(2+x)
g.rel("mul", numB, denA, prodLeft)    # (1+x)(5+x) = same var, forces equality
add(726, g.factors, x, 1,
    f"Consider the numbers {L(c3)}, {L(c5)}, {L(c1)}, {L(c2)}, {L(x)}, "
    f"{L(numA)}, {L(denA)}, {L(numB)}, {L(denB)}, {L(prodLeft)}. "
    f"{L(c3)} is 3. {L(c5)} is 5. {L(c1)} is 1. {L(c2)} is 2. {L(c3)} "
    f"plus {L(x)} equals {L(numA)}. {L(c5)} plus {L(x)} equals "
    f"{L(denA)}. {L(c1)} plus {L(x)} equals {L(numB)}. {L(c2)} plus "
    f"{L(x)} equals {L(denB)}. {L(numA)} times {L(denB)} equals "
    f"{L(prodLeft)}. {L(numB)} times {L(denA)} equals {L(prodLeft)}. "
    f"What is {L(x)}?",
    "direct system encoding (technique 2): the CSP searches x directly "
    "against the cross-multiplied equality (3+x)(2+x)=(1+x)(5+x), both "
    "products forced equal via a shared result var; never pre-expands "
    "the quadratics by hand.")

# 727: 81^(3/4) -> 27. Known fractional exponent evaluated (4th-root then
# cube), matches [601]/tranche4 precedent.
g = G()
c81 = g.given(81)
root = g.free()
t1 = g.rel("mul", root, root)
g.rel("mul", t1, t1, c81)             # root^4=81 -> root=3
cube = g.rel("mul", t1, root)         # root^3 = 9*3 = 27
add(727, g.factors, cube, 27,
    f"Consider the numbers {L(c81)}, {L(root)}, {L(t1)}, {L(cube)}. "
    f"{L(c81)} is 81. {L(root)} times {L(root)} equals {L(t1)}. "
    f"{L(t1)} times {L(t1)} equals {L(c81)}. {L(t1)} times {L(root)} "
    f"equals {L(cube)}. What is {L(cube)}?",
    "REARRANGEMENT (named: 81^(3/4)=(81^(1/4))^3, evaluating a source-"
    "literal FIXED exponent 3/4 via 4th-root-then-cube), degree: matches "
    "[601] and [769] this tranche exactly -- the exponent is a KNOWN "
    "source literal being evaluated, not an unknown being searched for. "
    "4th root found via search (root^4=81), cube via chained "
    "multiplication reusing the squared intermediate.")

# 736: kx^2-5x-12=0, roots 3 and -4/3 -> k=3. Vieta product identity
# (named), magnitude-fold for the negative fractional root (avoids ever
# forming -4/3 directly: uses its numerator/denominator).
g = G()
root1 = g.given(3)                    # x=3
negNum = g.given(4)                   # numerator of "-4/3"
negDen = g.given(3)                   # denominator of "-4/3" (distinct var
                                       # from root1 despite same numeral)
prodMagNum = g.rel("mul", root1, negNum)  # 3*4=12
prodMag = g.fdiv(prodMagNum, 3)       # 12/3=4  (ONE fdiv, |product|; k=3 is
                                       # the LITERAL divisor value, matching
                                       # negDen's own given value -- fdiv's k
                                       # must be a plain int, not a var index)
c12const = g.given(12)                # constant term magnitude, "-12"
k = g.free()
g.rel("mul", k, prodMag, c12const)    # k*4=12 -> k=3
add(736, g.factors, k, 3,
    f"Consider the numbers {L(root1)}, {L(negNum)}, {L(negDen)}, "
    f"{L(prodMagNum)}, {L(prodMag)}, {L(c12const)}, {L(k)}. {L(root1)} "
    f"is 3. {L(negNum)} is 4. {L(negDen)} is 3. {L(root1)} times "
    f"{L(negNum)} equals {L(prodMagNum)}. When {L(prodMagNum)} is "
    f"divided by 3, the quotient is {L(prodMag)}. {L(c12const)} is 12. "
    f"{L(k)} times {L(prodMag)} equals {L(c12const)}. What is {L(k)}?",
    "LAW 3/9 TENSION (flagged for the wheel): Vieta product identity "
    "(named: product of roots = c/a, i.e. 3*(-4/3) = -12/k), rendered "
    "via magnitude-fold (technique 3) for the negative fractional root "
    "-- |product|=3*4/3=4 computed from the root's own numerator (4) and "
    "denominator (3) WITHOUT ever forming the fraction -4/3 directly; k "
    "then found by multiplicative inversion from |product|*k=12. The "
    "linear coefficient (-5x) is unused (not needed for this "
    "derivation) -- consistent with prior precedent that graphs need not "
    "exhaust every source literal.")

# 737: 2x^2-5x-4=0, discriminant n=57. Quadratic-formula discriminant
# (named theorem-application), b and c magnitudes from the source, "4" in
# the formula is a universal constant (Law 1).
g = G()
bMag = g.given(5)                     # |b|, "-5x"
aCoef = g.given(2)
cMag = g.given(4)                     # |c|, "-4"
c4uni = g.given(4)                    # universal constant "4" in b^2-4ac
bsq = g.rel("mul", bMag, bMag)        # 25
ac = g.rel("mul", aCoef, cMag)        # 8
fourac = g.rel("mul", c4uni, ac)      # 32
n = g.rel("add", bsq, fourac)         # 25+32=57
add(737, g.factors, n, 57,
    f"Consider the numbers {L(bMag)}, {L(aCoef)}, {L(cMag)}, {L(c4uni)}, "
    f"{L(bsq)}, {L(ac)}, {L(fourac)}, {L(n)}. {L(bMag)} is 5. {L(aCoef)} "
    f"is 2. {L(cMag)} is 4. {L(c4uni)} is 4. {L(bMag)} times {L(bMag)} "
    f"equals {L(bsq)}. {L(aCoef)} times {L(cMag)} equals {L(ac)}. "
    f"{L(c4uni)} times {L(ac)} equals {L(fourac)}. {L(bsq)} plus "
    f"{L(fourac)} equals {L(n)}. What is {L(n)}?",
    "THEOREM-APPLICATION (named: quadratic formula discriminant, "
    "n=b^2-4ac), degree: since the source's b,c terms are both negative "
    "(-5x, -4), the discriminant's SIGN structure (b^2 positive, -4ac "
    "positive since a,c... wait a and c have opposite signs already "
    "folded into magnitudes -- computed directly as b^2 + 4*a*|c|, "
    "matching -4*a*c=-4*2*(-4)=+32) is a fixed, source-stated sign fact, "
    "not searched; the '4' in the formula itself is a universal "
    "constant (Law 1). All values stay far under the cap (max "
    "intermediate 57).")

# 739: SKIP -- log_sqrt6(216sqrt6), unknown-exponent + hidden-base-
# recognition (216=6^3).
skip(739,
     "log_sqrt(6)(216*sqrt(6))=7 since (sqrt6)^7=6^3*sqrt6=216sqrt6, but "
     "reaching this requires BOTH recognizing 216=6^3 (a hidden-base "
     "recognition, same family as tranche4's [672] '100=10^2') AND "
     "solving for the unknown exponent itself (the query IS the "
     "exponent) -- doubly in the discrete-exponent family flagged "
     "repeatedly this tranche ([589]/[621]/[643]/[672] in tranche4; "
     "[715] above). No primitive solves for an unknown exponent.")

# 742: y=kx^(1/4), y=3sqrt2 at x=81, find y at x=4 -> 2. Key cancellation:
# x1=81=3^4 EXACTLY matches y1's own coefficient base (3), so k^4=y1^4/x1
# reduces to just the radicand squared (2^2=4) WITHOUT ever forming
# y1^4=324 (over cap). The 81=3^4 identity is VERIFIED in-graph (Law 2),
# not asserted.
g = G()
c3 = g.given(3)                       # coefficient of sqrt2 in y1="3sqrt2"
t1 = g.rel("mul", c3, c3)             # 9
t2 = g.rel("mul", t1, c3)             # 27
c81 = g.given(81)                     # x1
g.rel("mul", t2, c3, c81)             # 27*3=81, VERIFIES x1=3^4 in-graph
c2 = g.given(2)                       # radicand inside sqrt2 for y1
k4 = g.rel("mul", c2, c2)             # k^4 = 2^2 = 4 (since 3^4 cancels x1)
c4x2 = g.given(4)                     # x2
y2pow4 = g.rel("mul", k4, c4x2)       # k^4 * x2 = 16
y2 = g.free()
u1 = g.rel("mul", y2, y2)
g.rel("mul", u1, u1, y2pow4)          # y2^4=16 -> y2=2
add(742, g.factors, y2, 2,
    f"Consider the numbers {L(c3)}, {L(t1)}, {L(t2)}, {L(c81)}, {L(c2)}, "
    f"{L(k4)}, {L(c4x2)}, {L(y2pow4)}, {L(y2)}, {L(u1)}. {L(c3)} is 3. "
    f"{L(c3)} times {L(c3)} equals {L(t1)}. {L(t1)} times {L(c3)} equals "
    f"{L(t2)}. {L(c81)} is 81. {L(t2)} times {L(c3)} equals {L(c81)}. "
    f"{L(c2)} is 2. {L(c2)} times {L(c2)} equals {L(k4)}. {L(c4x2)} is "
    f"4. {L(k4)} times {L(c4x2)} equals {L(y2pow4)}. {L(y2)} times "
    f"{L(y2)} equals {L(u1)}. {L(u1)} times {L(u1)} equals {L(y2pow4)}. "
    f"What is {L(y2)}?",
    "LAW 8 TENSION (flagged for the wheel, sharpest item this tranche): "
    "THEOREM-APPLICATION (named: y^4=k^4*x for y=kx^(1/4)); the key move "
    "is that x1=81 EXACTLY equals 3^4 (the coefficient of sqrt2 in y1's "
    "own value, 3sqrt2, raised to the 4th) -- so k^4=y1^4/x1 = "
    "(3^4*2^2)/3^4 = 2^2 = 4 with the 3^4 term cancelling algebraically, "
    "avoiding ever forming y1^4=(3sqrt2)^4=324 (over the 300 cap). "
    "Critically, unlike tranche4's [636], the 81=3^4 identity is "
    "VERIFIED IN-GRAPH (27*3=81, a real constraint the solver checks), "
    "not asserted externally -- Law 2 reconstruction-co-test pass. "
    "y2 found by 4th-root search from k^4*x2=16.",
    watch="pointer-collision: c3 (coefficient of sqrt2 in y1) serves as an "
          "argument in 3 separate mul factors building the 3^4=81 chain.")

# 744: (109^2-100^2)/9 -> 209. Difference-of-squares; the denominator (9)
# EXACTLY equals (109-100), verified in-graph, cancelling the division to
# a plain sum.
g = G()
c109 = g.given(109)
c100 = g.given(100)
c9 = g.given(9)                       # the fraction's denominator
g.rel("sub", c109, c100, c9)          # VERIFIES 109-100=9 in-graph
ans = g.rel("add", c109, c100)        # =(109-100)(109+100)/9 = 109+100
add(744, g.factors, ans, 209,
    f"Consider the numbers {L(c109)}, {L(c100)}, {L(c9)}, {L(ans)}. "
    f"{L(c109)} is 109. {L(c100)} is 100. {L(c9)} is 9. {L(c109)} "
    f"exceeds {L(c100)} by {L(c9)}. {L(c109)} plus {L(c100)} equals "
    f"{L(ans)}. What is {L(ans)}?",
    "REARRANGEMENT (named: difference-of-squares, (a^2-b^2)/(a-b)=a+b), "
    "degree: the source's own denominator (9) EXACTLY equals a-b "
    "(109-100), a fact VERIFIED IN-GRAPH via the shared result var (Law "
    "2 reconstruction-co-test pass, not asserted), so the division "
    "cancels exactly to a plain addition -- avoids ever forming "
    "109^2-100^2=1881 (over the 300 cap). Matches tranche4's cap-"
    "avoidance rearrangement family ([655],[674]).")

# 746: SKIP -- complex-number arithmetic (i as an algebraic symbol, i^2=-1
# substitution), no primitive represents non-real quantities.
skip(746,
     "2(3-i)+i(2+i) = 6-2i+2i+i^2 = 6-1 = 5 requires grouping real vs "
     "imaginary TERMS and substituting i^2=-1 -- genuine complex-number "
     "symbolic algebra, not arithmetic over concrete real magnitudes. "
     "Our primitives only ever hold real integer values in [0,300]; 'i' "
     "is not a number our graph can represent at all (not even via "
     "magnitude-fold, since it isn't real). Rendering only the real-part "
     "shortcut (6-1=5) would presuppose the imaginary-term cancellation "
     "externally -- Law-3 residue. NEW skip family this tranche "
     "(complex-number arithmetic), distinct from the exponent-family "
     "cluster.")

# 747: 1/x + 2/x-div-4/x = 0.75 -> x=4. Order of operations: 2/x div 4/x =
# 2/4, a constant independent of x (source-literal ratio, Law 1); explicated
# 0.75 as 3/4. 1/x = 3/4-2/4 -> x=4.
g = G()
c3 = g.given(3)                       # numerator of 0.75 as 3/4
c2 = g.given(2)                       # numerator of "2/x"
num = g.rel("sub", c3, c2)            # 3-2=1  (numerator of 1/x once solved)
c4 = g.given(4)                       # shared denominator (0.75's denom AND
                                       # "4/x"'s denominator, both literally 4)
xvar = g.free()
g.rel("mul", num, xvar, c4)           # num*x=4 -> x=4
add(747, g.factors, xvar, 4,
    f"Consider the numbers {L(c3)}, {L(c2)}, {L(num)}, {L(c4)}, "
    f"{L(xvar)}. {L(c3)} is 3. {L(c2)} is 2. {L(c3)} exceeds {L(c2)} by "
    f"{L(num)}. {L(c4)} is 4. {L(num)} times {L(xvar)} equals {L(c4)}. "
    f"What is {L(xvar)}?",
    "LAW 3 TENSION (light, flagged for completeness): order-of-"
    "operations resolves '2/x div 4/x' to the constant ratio 2/4 "
    "(independent of x, both source literals, matches [690]'s "
    "invariant-identity family), and 0.75 is lexically explicated as "
    "its exact fraction 3/4 (a known-value explicitation, not an "
    "invented number). Equation reduces to 1/x=(3-2)/4=1/4, cross-"
    "multiplied to x=4; flagged because the final coefficient (3-2=1) "
    "makes the last step a trivial multiply, though it IS a genuine "
    "derived quantity from two distinct source literals (the '3' from "
    "0.75, the '2' from '2/x'), not asserted.")

# 748: consecutive integers, first+third=118 -> second=59. Structural
# constant "2" (universal, the gap between first and third of three
# consecutive integers), matches tranche3/4 precedent.
g = G()
c118 = g.given(118)
c2 = g.given(2)                       # universal constant: third-first=2
n2 = g.rel("sub", c118, c2)           # 2*first = 118-2 = 116
c1 = g.given(1)
n = g.fdiv(n2, 2)                     # first=58  (ONE fdiv)
second = g.rel("add", n, c1)          # 59
add(748, g.factors, second, 59,
    f"Consider the numbers {L(c118)}, {L(c2)}, {L(c1)}, {L(n2)}, {L(n)}, "
    f"{L(second)}. {L(c118)} is 118. {L(c2)} is 2. {L(c118)} exceeds "
    f"{L(c2)} by {L(n2)}. {L(c1)} is 1. When {L(n2)} is divided by 2, "
    f"the quotient is {L(n)}. {L(n)} plus {L(c1)} equals {L(second)}. "
    f"What is {L(second)}?",
    "structural constant (Law 1, universal: for three consecutive "
    "integers first/second/third, third=first+2, matches tranche3/4's "
    "'universal constant from a counting structure' precedent); first "
    "found by clearing the constant gap and halving, second is first+1. "
    "One fdiv (k=2).")

# 753: Sarah's fence, area>=100, length=width+15, minimize perimeter ->
# width=5. Boundary clause (Law 12): perimeter increases monotonically
# with width (externally verified, standard fact), so minimizing it means
# the LEAST width satisfying the area bound, i.e. the binding equality
# w(w+15)=100.
g = G()
c15 = g.given(15)
c100 = g.given(100)
rPos = g.free()
rNegMag = g.free()
g.rel("sub", rNegMag, rPos, c15)      # rNegMag-rPos=15
g.rel("mul", rPos, rNegMag, c100)     # rPos*rNegMag=100
add(753, g.factors, rPos, 5,
    f"Consider the numbers {L(c15)}, {L(c100)}, {L(rPos)}, "
    f"{L(rNegMag)}. {L(c15)} is 15. {L(c100)} is 100. {L(rNegMag)} "
    f"exceeds {L(rPos)} by {L(c15)}. {L(rPos)} times {L(rNegMag)} "
    f"equals {L(c100)}. What is {L(rPos)}?",
    "LAW 12 TENSION (boundary clause, flagged for the wheel): source "
    "states area>=100 (an inequality) while minimizing perimeter=4w+30 "
    "(monotone increasing in w for w>0, verified externally as a "
    "standard fact) -- (a) binding constraint named: the least "
    "feasible w is exactly where w(w+15)=100; (b) no other constraint "
    "to verify; (c) counterfactual holds: maximizing material instead "
    "would be unbounded (no finite answer), confirming the render logic "
    "is genuinely tied to MINIMIZATION. Magnitude-fold (technique 3) "
    "for the negative root of w^2+15w-100=0 (w=5 or w=-20); direct "
    "system encoding on the positive root and the negative root's "
    "magnitude.")

# 755: cube-root(3^5+3^5+3^5) -> 9. 3*3^5=3^6=(3^2)^3; the EXPONENT
# arithmetic (5+1=6, 6/3=2) is rendered in the small-value exponent
# domain (Law 8 bookkeeping), avoiding ever materializing 3^5+3^5+3^5=729
# (over the 300 cap) in the value domain.
g = G()
c5exp = g.given(5)                    # the exponent 5 in 3^5
c1 = g.given(1)                       # from count=3=3^1 (structural: the
                                       # coefficient 3 IS the base 3^1)
combinedExp = g.rel("add", c5exp, c1)  # 5+1=6  (exponent-domain arithmetic)
resultExp = g.fdiv(combinedExp, 3)    # 6/3=2  (ONE fdiv, EXACT quotient --
                                       # cleaner than tranche4's [636],
                                       # no external remainder-bound needed)
c3base = g.given(3)                   # the base 3 (also the term count)
ans = g.rel("mul", c3base, c3base)    # 3^2=9 (chain length = resultExp's
                                       # derived value, fixed-small-exponent
                                       # technique, matches tranche4's [663])
add(755, g.factors, ans, 9,
    f"Consider the numbers {L(c5exp)}, {L(c1)}, {L(combinedExp)}, "
    f"{L(resultExp)}, {L(c3base)}, {L(ans)}. {L(c5exp)} is 5. {L(c1)} is "
    f"1. {L(c5exp)} plus {L(c1)} equals {L(combinedExp)}. When "
    f"{L(combinedExp)} is divided by 3, the quotient is {L(resultExp)}. "
    f"{L(c3base)} is 3. {L(c3base)} times {L(c3base)} equals {L(ans)}. "
    f"What is {L(ans)}?",
    "LAW 8 TENSION (bookkeeping clause, flagged for the wheel): "
    "THEOREM-APPLICATION (named: cube-root-of-power identity, "
    "cube_root(3*3^5)=cube_root(3^6)=3^2), since the term COUNT (3) "
    "equals the BASE (3), the sum collapses to 3^(5+1)=3^6 (a form "
    "identity, Law 10 structural literal). The transform's ARGUMENTS "
    "(exponent arithmetic 5+1=6, 6/3=2) render in-graph in the small "
    "exponent-value domain per Law 8 bookkeeping, while the actual "
    "power (3^6=729) is NEVER materialized in the value domain (would "
    "exceed the 300 cap by more than double) -- final answer computed "
    "as 3^2 via a chain whose length matches the derived exponent "
    "value (fixed-small-exponent technique, tranche4's [663] "
    "precedent). Distinguished from tranche4's [636]: the quotient here "
    "(6/3=2) is EXACT with no remainder, so no external remainder-bound "
    "verification is needed -- a cleaner instance of the same family. "
    "One fdiv (k=3).")

# 758: a*b=a^2+ab-b^2, 3*2 -> 11. Direct computation on source literals.
g = G()
c3 = g.given(3)
c2 = g.given(2)
sq1 = g.rel("mul", c3, c3)            # 9
cross = g.rel("mul", c3, c2)          # 6
sq2 = g.rel("mul", c2, c2)            # 4
t = g.rel("add", sq1, cross)          # 15
ans = g.rel("sub", t, sq2)            # 11
add(758, g.factors, ans, 11,
    f"Consider the numbers {L(c3)}, {L(c2)}, {L(sq1)}, {L(cross)}, "
    f"{L(sq2)}, {L(t)}, {L(ans)}. {L(c3)} is 3. {L(c2)} is 2. {L(c3)} "
    f"times {L(c3)} equals {L(sq1)}. {L(c3)} times {L(c2)} equals "
    f"{L(cross)}. {L(c2)} times {L(c2)} equals {L(sq2)}. {L(sq1)} plus "
    f"{L(cross)} equals {L(t)}. {L(t)} exceeds {L(sq2)} by {L(ans)}. "
    f"What is {L(ans)}?",
    "direct computation of the source's own custom-operator definition "
    "(a^2+ab-b^2 at a=3,b=2), every term a source literal, no free "
    "variables needed.")

# 762: geometric sequence, 7th=7, 10th=21 -> 13th=63. Step-count-matching
# (matches [685]'s analogue): ratio-cubed = 21/7=3 found by inversion
# (no need for the actual irrational per-step ratio), 13th = 10th * r^3.
g = G()
c7 = g.given(7)
c21 = g.given(21)
rcubed = g.free()
g.rel("mul", rcubed, c7, c21)         # rcubed*7=21 -> rcubed=3
ans = g.rel("mul", c21, rcubed)       # 21*3=63
add(762, g.factors, ans, 63,
    f"Consider the numbers {L(c7)}, {L(c21)}, {L(rcubed)}, {L(ans)}. "
    f"{L(c7)} is 7. {L(c21)} is 21. {L(rcubed)} times {L(c7)} equals "
    f"{L(c21)}. {L(c21)} times {L(rcubed)} equals {L(ans)}. What is "
    f"{L(ans)}?",
    "step-count-matching identity (named, matches [685]'s arithmetic-"
    "sequence analogue for geometric sequences): the index gap from "
    "term7 to term10 (3) equals the gap from term10 to term13 (3), so "
    "the SAME cubed-ratio (r^3=term10/term7) carries term10 to term13 -- "
    "avoids ever needing the irrational per-step ratio r=3^(1/3). "
    "r^3 found by multiplicative inversion (Worked Example C).")

# 763: two positive integers differ by 6, product=112 -> sum=22. Direct
# system (technique 2), no sign ambiguity (both positive, difference is a
# known literal so ordering is unambiguous).
g = G()
c6 = g.given(6)
c112 = g.given(112)
small = g.free()
big = g.rel("add", small, c6)         # big = small+6
g.rel("mul", small, big, c112)        # small*big=112
ans = g.rel("add", small, big)        # 22
add(763, g.factors, ans, 22,
    f"Consider the numbers {L(c6)}, {L(c112)}, {L(small)}, {L(big)}, "
    f"{L(ans)}. {L(c6)} is 6. {L(c112)} is 112. {L(small)} plus {L(c6)} "
    f"equals {L(big)}. {L(small)} times {L(big)} equals {L(c112)}. "
    f"{L(small)} plus {L(big)} equals {L(ans)}. What is {L(ans)}?",
    "direct system encoding (technique 2): big is defined directly as "
    "small+6 (unambiguous ordering since 6 is a known source literal, "
    "unlike Vieta-symmetric systems needing a self-resolving free gap); "
    "the CSP searches small against the product constraint, matches "
    "tranche4's [610] direct quadratic-product system precedent.")

# 764: a*b=2a-b^2, a*5=9 -> a=17.
g = G()
c5 = g.given(5)
c9 = g.given(9)
sq = g.rel("mul", c5, c5)             # 25
rhs = g.rel("add", c9, sq)            # 2a=9+25=34
c2 = g.given(2)
a = g.free()
g.rel("mul", c2, a, rhs)              # 2a=34 -> a=17
add(764, g.factors, a, 17,
    f"Consider the numbers {L(c5)}, {L(c9)}, {L(sq)}, {L(rhs)}, {L(c2)}, "
    f"{L(a)}. {L(c5)} is 5. {L(c9)} is 9. {L(c5)} times {L(c5)} equals "
    f"{L(sq)}. {L(c9)} plus {L(sq)} equals {L(rhs)}. {L(c2)} is 2. "
    f"{L(c2)} times {L(a)} equals {L(rhs)}. What is {L(a)}?",
    "direct computation of the source's own custom-operator definition "
    "(2a-b^2=9 at b=5), rearranged to 2a=9+25 (moving -b^2 across, "
    "rearrangement Law 4); a found by multiplicative inversion.")

# 769: (6th-root(4))^9 -> 8. Known fractional exponent (9/6=3/2) evaluated
# via sqrt-then-cube, matches [601]/[727] precedent.
g = G()
c4 = g.given(4)
root = g.free()
g.rel("mul", root, root, c4)          # sqrt(4): root*root=4 -> root=2
sq = g.rel("mul", root, root)         # root^2=4 (reused structurally)
cube = g.rel("mul", sq, root)         # root^3=8
add(769, g.factors, cube, 8,
    f"Consider the numbers {L(c4)}, {L(root)}, {L(sq)}, {L(cube)}. "
    f"{L(c4)} is 4. {L(root)} times {L(root)} equals {L(c4)}. {L(root)} "
    f"times {L(root)} equals {L(sq)}. {L(sq)} times {L(root)} equals "
    f"{L(cube)}. What is {L(cube)}?",
    "REARRANGEMENT (named: (4th... 6th-root(4))^9 = 4^(9/6) = 4^(3/2) = "
    "(sqrt4)^3, evaluating a source-literal FIXED exponent 3/2 via "
    "sqrt-then-cube), degree: matches [601] (tranche4) and [727] this "
    "tranche exactly -- the exponent is a KNOWN literal being evaluated, "
    "not searched. sqrt(4)=2 via search-based root extraction, cubed "
    "via chained multiplication.")

# 772: x*(x+y)=x^2+8 -> xy=8. Rendered as the EXACT source equation (x,y
# both free, x^2 built explicitly), never as a bare echo of the literal 8
# -- Invariance Clause (Law 7) applies cleanly: the CSP's own search
# picks SOME valid (x,y) (any divisor pair of 8), and xy is invariant to
# which one it finds, since the equation genuinely forces x^2 to cancel.
g = G()
c8 = g.given(8)
x = g.free()
y = g.free()
xpy = g.rel("add", x, y)              # x+y
xsq = g.rel("mul", x, x)              # x^2
rhs = g.rel("add", xsq, c8)           # x^2+8
g.rel("mul", x, xpy, rhs)             # x*(x+y) = rhs  (forces equality with
                                       # the SOURCE's own literal equation)
xyprod = g.rel("mul", x, y)           # query: xy
add(772, g.factors, xyprod, 8,
    f"Consider the numbers {L(c8)}, {L(x)}, {L(y)}, {L(xpy)}, {L(xsq)}, "
    f"{L(rhs)}, {L(xyprod)}. {L(c8)} is 8. {L(x)} plus {L(y)} equals "
    f"{L(xpy)}. {L(x)} times {L(x)} equals {L(xsq)}. {L(xsq)} plus "
    f"{L(c8)} equals {L(rhs)}. {L(x)} times {L(xpy)} equals {L(rhs)}. "
    f"{L(x)} times {L(y)} equals {L(xyprod)}. What is {L(xyprod)}?",
    "LAW 7 TENSION (invariance clause, flagged for the wheel): the "
    "SOURCE's own equation x(x+y)=x^2+8 is rendered EXACTLY as stated "
    "(x, y both free, x^2 built explicitly as x*x, both sides forced "
    "equal via a shared result var) -- the graph does NOT pre-cancel "
    "x^2 by hand or assert xy=8 directly (which would be Law-3 bare-"
    "echo residue, since 8 already IS the source's own literal). The "
    "CSP's SEARCH, not the annotator, picks which valid (x,y) pair "
    "(any divisor of 8) satisfies the equation; xy is queried as its "
    "own derived variable and is answer-invariant across every valid "
    "assignment BECAUSE the underlying equation genuinely forces x^2 to "
    "cancel -- this is the SHARPENED Law 7 test passing cleanly (the "
    "canonicalizing constraint, the full unreduced equation, is fully "
    "representable in-graph, unlike the convicted q=1 exponent case).")

# 774: SKIP -- a*b=a^b+ab=15, a,b>=2 integers, find a+b. Both base AND
# exponent are unknowns being jointly searched -- no pow-with-variable-
# exponent primitive, compounded case of the discrete-exponent family.
skip(774,
     "a star b = a^b+ab; a=3,b=2 gives 9+6=15 (a+b=5), found by "
     "enumerating small integer pairs -- but this requires an "
     "exponentiation operator with a VARIABLE exponent (b unknown) AND "
     "a variable base (a unknown) jointly searched, an even harder case "
     "than the single-unknown-exponent family flagged repeatedly this "
     "tranche ([715]/[739] above; tranche4's [589]/[621]/[643]/[672]). "
     "No primitive supports iterated multiplication a variable number "
     "of times with both operands unknown.")

print()
print(f"TOTAL drafted (pre-checks passing): {len(rows)}")
print(f"FAILS: {fails}")
print(f"SKIPS: {[s[0] for s in skips]}")

with open('/home/bryce/mycelium/.cache/book8_t5_prose_pairs_draft.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

with open('/home/bryce/mycelium/.cache/book8_t5_skips.json', 'w') as f:
    json.dump(skips, f, indent=2)

print("done, wrote", len(rows), "rows")
