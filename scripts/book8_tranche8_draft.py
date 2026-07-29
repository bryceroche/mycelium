import json, sys, string
sys.path.insert(0, '/home/bryce/mycelium')
sys.path.insert(0, '/home/bryce/mycelium/scripts')
from tta_alg2_dials import solve2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic

SMP = {"n_vars": 24, "m": 300}
LETTERS = string.ascii_lowercase

CAND_DOC = json.load(open('/home/bryce/mycelium/.cache/book8_candidates_t8.json'))
CANDS = {c["src_idx"]: c for c in CAND_DOC["tranche8"]}
DIET_SRCS = set(CAND_DOC["diet_srcs"])


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

    def pct(self, a, b, p, result_is_a=True):
        # a is p percent of b  =>  args=[a,b], p=p
        self.factors.append({"ftype": "pct", "args": [a, b], "p": int(p)})
        return a


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


def gen_sentence(f):
    if f["ftype"] == "given":
        return f"{L(f['var'])} is {f['value']}."
    if f["ftype"] == "rel":
        a, b = f["args"]; c = f["result"]
        if f["op"] == "add":
            return f"{L(a)} plus {L(b)} equals {L(c)}."
        if f["op"] == "sub":
            return f"{L(a)} exceeds {L(b)} by {L(c)}."
        if f["op"] == "mul":
            return f"{L(a)} times {L(b)} equals {L(c)}."
    if f["ftype"] == "fdiv":
        return f"When {L(f['var'])} is divided by {f['k']}, the quotient is {L(f['result'])}."
    if f["ftype"] == "pct":
        a, b = f["args"]
        return f"{L(a)} is {f['p']} percent of {L(b)}."
    raise ValueError(f)


def build_dialect(factors, query_var):
    n = max([f.get("var", -1) for f in factors] +
            [f.get("result", -1) for f in factors] +
            [a for f in factors if f["ftype"] in ("rel",) for a in f["args"]] +
            [a for f in factors if f["ftype"] == "pct" for a in f["args"]] +
            [query_var]) + 1
    header = "Consider the numbers " + ", ".join(L(i) for i in range(n)) + ". "
    body = " ".join(gen_sentence(f) for f in factors)
    return header + body + f" What is {L(query_var)}?"


def build_row(src_idx, factors, query_var, notes, watch=None, accommodation=None,
              routing_fact=None, diet=False, n_vars=24, m=300):
    sol = full_solution(factors, n_vars, m)
    assert sol is not None, f"src {src_idx}: solver failed to find full solution"
    dialect = build_dialect(factors, query_var)
    gen = {
        "src_idx": src_idx, "book": 8, "tranche": 8, "floor": "prime", "fs": True,
        "dialect": dialect, "gate": "PENDING:5view-vote+key", "generation": "21",
        "notes": notes,
    }
    if diet:
        gen["diet_customer"] = CANDS[src_idx]["diet_customer"]
        gen["watch"] = "diet-baseline"
    if watch:
        gen["watch"] = (gen.get("watch", "") + " | " + watch).strip(" |")
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


def add(src_idx, factors, query_var, expect, notes, watch=None, accommodation=None,
        routing_fact=None):
    ok = check(src_idx, factors, query_var, expect)
    if not ok:
        fails.append(src_idx)
        return
    diet = src_idx in DIET_SRCS
    rows.append(build_row(src_idx, factors, query_var, notes, watch, accommodation,
                           routing_fact, diet=diet))


def skip(src_idx, reason):
    skips.append((src_idx, reason))
    print(f"[{src_idx}] SKIP: {reason[:70]}...")

# ===========================================================================
# 949: line slope 3, line slope 5, intersect at (10,15). Distance between
# x-intercepts. Rearrangement (Law 4): x-intercept from point-slope form,
# solved via multiplicative inversion (technique C) twice, no fdiv.
g = G()
c10 = g.given(10)
c15 = g.given(15)
c3 = g.given(3)
t1 = g.rel("mul", c3, c10)            # 30
rhs1 = g.rel("sub", t1, c15)          # 15 (=3*x1)
x1 = g.free()
g.rel("mul", c3, x1, rhs1)            # 3*x1=15 -> x1=5
c5 = g.given(5)
t2 = g.rel("mul", c5, c10)            # 50
rhs2 = g.rel("sub", t2, c15)          # 35 (=5*x2)
x2 = g.free()
g.rel("mul", c5, x2, rhs2)            # 5*x2=35 -> x2=7
ans = g.rel("sub", x2, x1)            # 2
add(949, g.factors, ans, 2,
    "REARRANGEMENT (named: point-slope form y-15=m(x-10), solved for the "
    "x-intercept y=0), degree: two independent lines each solved via "
    "multiplicative inversion (technique C, avoiding fdiv), then the "
    "positive difference of the two x-intercepts queried directly.")


# 956: x^2+y^2=4x+12y-39, center (h,k), h+k. Completing the square: h=4/2,
# k=12/2. h via multiplicative inversion (avoids a 2nd fdiv), k via the one
# allowed fdiv. Matches [897]'s "two independently-derived quantities
# summed" shape, not a bare coefficient echo.
g = G()
c4 = g.given(4)
c2 = g.given(2)
h = g.free()
g.rel("mul", c2, h, c4)               # 2*h=4 -> h=2 (mult inversion)
c12 = g.given(12)
k = g.fdiv(c12, 2)                    # 6 (ONE fdiv, k=2)
ans = g.rel("add", h, k)              # 8
add(956, g.factors, ans, 8,
    "THEOREM-APPLICATION (named: completing the square, x^2-2hx+h^2=(x-h)^2, "
    "identifies h,k as half the linear coefficients), degree: h derived via "
    "multiplicative inversion (technique C), k via the one allowed fdiv; "
    "h+k queried as a fresh sum of two independently-derived quantities "
    "(matches [897]'s shape), not a bare echo of either source coefficient.")

# 957: rectangle (x-2) by (2x+5), area=8x-6, find x. Direct rendering of the
# geometric area equation (technique 2 style, single unknown) -- no
# algebraic simplification, the solver does the genuine quadratic search.
# The extraneous root (x=-1/2) is non-integer and structurally excluded by
# the CSP's own nonneg-integer domain, for free (unlike [880]'s skip).
g = G()
x = g.free()
c2a = g.given(2)
width = g.rel("sub", x, c2a)          # x-2
c2b = g.given(2)
twoX = g.rel("mul", c2b, x)
c5 = g.given(5)
length = g.rel("add", twoX, c5)       # 2x+5
c8 = g.given(8)
eightX = g.rel("mul", c8, x)
c6 = g.given(6)
rhs = g.rel("sub", eightX, c6)        # 8x-6
g.rel("mul", width, length, rhs)      # (x-2)(2x+5) = 8x-6, asserted directly
add(957, g.factors, x, 4,
    "direct rendering of the source's own geometric area equation "
    "(width*length=area, technique 2 style on a single unknown) -- no "
    "algebraic simplification performed by the pen, the solver does the "
    "genuine quadratic search. The extraneous root (x=-1/2) is non-integer "
    "and structurally excluded by the CSP's own nonneg-integer domain, for "
    "free (unlike [880]'s skip, where the extraneous root WAS in-domain).")

# 960: 3T+2C=21, 2T+3C=19 (triangles/circles), find 3C. Direct system
# encoding (technique 2): t,c free vars jointly satisfy BOTH of the
# source's own equations; the query (3 circles) is exactly one of the
# graph's own already-computed terms (threeC), reused directly -- not a
# residue, since t,c required genuine 2-var CSP search to pin.
g = G()
t = g.free()
c = g.free()
c3a = g.given(3)
threeT = g.rel("mul", c3a, t)
c2a = g.given(2)
twoC = g.rel("mul", c2a, c)
c21 = g.given(21)
g.rel("add", threeT, twoC, c21)       # 3t+2c=21
c2b = g.given(2)
twoT = g.rel("mul", c2b, t)
c3b = g.given(3)
threeC = g.rel("mul", c3b, c)
c19 = g.given(19)
g.rel("add", twoT, threeC, c19)       # 2t+3c=19
add(960, g.factors, threeC, 9,
    "direct system encoding (technique 2): t,c searched jointly against "
    "BOTH of the source's own equations (3t+2c=21, 2t+3c=19); the query "
    "(three circles) is exactly the graph's own 'threeC' term (3*c), "
    "reused directly rather than added as a redundant final step -- not a "
    "residue, since t,c required genuine simultaneous 2-var CSP search to "
    "pin (matches Worked Example E systems style).")

# 961: line: dx=3 -> dy=7. dx=9 -> dy=?. slope*3 via fdiv, then scale up.
g = G()
c3 = g.given(3)
c7 = g.given(7)
c9 = g.given(9)
mult = g.fdiv(c9, 3)                  # 3 (ONE fdiv, k=3)
ans = g.rel("mul", c7, mult)          # 21
add(961, g.factors, ans, 21,
    "THEOREM-APPLICATION (named: constant-slope proportional scaling), "
    "degree: the ratio of x-increments (9/3=3, the one allowed fdiv) "
    "scales the known y-increment directly; matches [892]'s two-point-"
    "slope family, simpler variant (no separate points needed).")

# 972: x^2-4x-14=3x+16 -> x^2-7x-30=0, roots 10,-3 (MIXED SIGN). LAW 3
# TENSION, novel sub-variant: unlike [873]/[905]/[943] (both roots
# positive), one root here is NEGATIVE, so sum-of-roots folds via
# SUBTRACTION of magnitudes, not addition -- the established technique's
# roles of sum/difference are swapped (sum-of-magnitudes pins the search,
# difference-of-magnitudes is the fresh query), extending the magnitude-
# fold technique ([897]/[902]) to the Vieta-root family for the first
# time. b(=7) is itself DERIVED (4+3, moving the 3x term across the
# equality), not a bare source literal, but the assembly-vs-deduction
# tension is about the FINAL COMPUTATION SHAPE, not b's literal/derived
# status, so the same rescue technique applies.
g = G()
c4 = g.given(4)
c3v = g.given(3)
b = g.rel("add", c4, c3v)             # 7 (derived: coefficients combined
                                       # across the equality, genuine work)
bsq = g.rel("mul", b, b)              # 49
c14 = g.given(14)
c16 = g.given(16)
cMag = g.rel("add", c14, c16)         # 30 (derived: constants combined
                                       # across the equality)
c4uni = g.given(4)                    # universal '4' in b^2-4ac
fourc = g.rel("mul", c4uni, cMag)     # 120
disc = g.rel("add", bsq, fourc)       # 169 (ADD not subtract -- c is
                                       # NEGATIVE here, so -4ac=+4|c|)
sqrtDisc = g.free()
g.rel("mul", sqrtDisc, sqrtDisc, disc)  # sqrtDisc=13 (=Rbig+Ssmall, sum
                                         # of MAGNITUDES, never fed as the
                                         # sum-of-roots constraint)
Ssmall = g.free()
gap = g.free()
Rbig = g.rel("add", Ssmall, gap)      # Rbig=Ssmall+gap (nonneg ordering,
                                       # matches [918]'s established
                                       # technique)
g.rel("add", Rbig, Ssmall, sqrtDisc)  # Rbig+Ssmall=13
g.rel("mul", Rbig, Ssmall, cMag)      # Rbig*Ssmall=30
ans = gap                             # = Rbig-Ssmall = 10-3 = 7 (fresh
                                       # query, never fed as a constraint)
add(972, g.factors, ans, 7,
    "LAW 3 TENSION (resolved, NEW sub-variant this tranche): x^2-7x-30=0 "
    "has roots 10 and -3 -- MIXED SIGN, unlike [873]/[905]/[943]/[796] "
    "(all same-sign root pairs). Sum-of-roots (7) folds via SUBTRACTION "
    "of magnitudes since one root is negative (10-3=7, not 10+3), so this "
    "row swaps the established technique's roles: the discriminant-sqrt "
    "(13=Rbig+Ssmall, sum of MAGNITUDES) is the search-pinning constraint "
    "(alongside the product, Rbig*Ssmall=|c|=30), and the DIFFERENCE "
    "(gap=Rbig-Ssmall) is the fresh, never-constrained query -- exactly "
    "mirroring [905]'s 'sum never fed in as a constraint' discipline, "
    "just with sum and difference swapped because the roots have opposite "
    "sign. b(=7) is derived (4+3, moving 3x across the equality) rather "
    "than a bare source literal, but is used only as an INPUT to bsq, "
    "never as the query itself.")

print(f"\nDrafted so far: {len(rows)}  Skipped: {len(skips)}  Fails: {len(fails)}")

# 958: SKIP (diet: subtrahend distractors) -- deg(f)=2, deg(g)<deg(f), find
# deg(f+g).
skip(958,
     "deg(f+g)=deg(f)=2 since deg(g)<deg(f) means g's terms can't reach "
     "f's leading degree -- a pure structural fact about polynomial "
     "addition, with ZERO numeric content in the source (no coefficients, "
     "no computable quantities at all). Bare Law-3 residue, matches "
     "[1009]/[1045] this tranche exactly. DIET NOTE: the target "
     "construction (subtrahend distractors, 'X exceeds Y by Z') has NO "
     "natural foothold here -- there is no subtraction anywhere in this "
     "problem's mathematics to render natively OR accommodate; this row "
     "would be skipped under normal production rules too (it fails the "
     "counterfactual test outright: no source quantity, present or "
     "absent, changes the render). Diet disposition: NOT exercised, "
     "source incompatible with ANY numeric construction.")

# 988: SKIP -- (x^2-9)/[(x^2+2x-3)(x-3)] undefined for how many x?
skip(988,
     "Denominator zero at x=-3,1,3 (three DISTINCT values, from two "
     "different factors: (x+3)(x-1) and (x-3)). Counting distinct roots "
     "across two polynomials, one of which (-3) is NEGATIVE and not "
     "representable in the nonneg domain, requires both a dedup-count "
     "primitive (none exists -- 'sel' only compares two ALREADY-COMPUTED "
     "candidates, not tallies a set) and negative-root handling beyond "
     "magnitude-fold's reach (fold rescues a SIGN, not a COUNT). "
     "Operation-shaped skip, new sub-family: counting distinct roots "
     "across multiple polynomial factors.")

# 990: SKIP -- for how many positive integers x is x^2+6x+9 between 20,40?
skip(990,
     "(x+3)^2 in (20,40) -> x+3 in {5,6} -> x in {2,3}, count=2. Counting "
     "integers satisfying a strict double inequality -- same 'no "
     "inequality-satisfying-integer-search primitive' family as "
     "tranche5's [712], tranche6's [814]/[838], this book's t7 [877].")

# 975: grandfather=12*Andrew; grandfather-Andrew=55 (age gap constant since
# birth). Direct system on a single unknown.
g = G()
c12 = g.given(12)
a = g.free()
gAge = g.rel("mul", c12, a)
c55 = g.given(55)
g.rel("sub", gAge, a, c55)            # 12a - a = 55
add(975, g.factors, a, 5,
    "direct system encoding: grandfather's age (12*Andrew) and the fixed "
    "birth-year age gap (55, constant across time) jointly pin Andrew's "
    "age via 11a=55; genuine single-unknown CSP search, not a residue.")

# 984: Jim+Bob=180. Bob-Jim = half of Bob (fdiv). Direct system, ONE fdiv.
g = G()
j = g.free()
b = g.free()
c180 = g.given(180)
g.rel("add", j, b, c180)              # J+B=180
half = g.fdiv(b, 2)                   # ONE fdiv, k=2
g.rel("sub", b, j, half)              # B-J = B/2
add(984, g.factors, b, 120,
    "direct system encoding (technique 2): J,B searched jointly against "
    "BOTH of the source's own facts (sum=180, and B-J equals half of B "
    "via the one allowed fdiv); genuine 2-var CSP search, B/2's floor "
    "constraint doesn't introduce ambiguity here since the unique integer "
    "solution (B=120) is comfortably even.")

# 985: Diana=Eduardo-3, Eduardo=Chad+4, Faye=Chad+3, Diana=14. Direct chain.
g = G()
c14 = g.given(14)                     # Diana
c3a = g.given(3)
ed = g.rel("add", c14, c3a)           # Eduardo = Diana+3 = 17
c4 = g.given(4)
chad = g.rel("sub", ed, c4)           # Chad = Eduardo-4 = 13
c3b = g.given(3)
faye = g.rel("add", chad, c3b)        # Faye = Chad+3 = 16
add(985, g.factors, faye, 16,
    "direct chain of the source's own stated age relations (Diana -> "
    "Eduardo -> Chad -> Faye), each a straight restatement of a source "
    "sentence; no algebraic rearrangement needed since Diana's value is "
    "given directly and the chain runs forward.")

# 987: x^2+y^2+21=4x+18y, circle radius. Completing the square (h=4/2 via
# mult inversion, k=18/2 via the one fdiv), then r=sqrt(h^2+k^2-21) via
# search-based root extraction (technique 1).
g = G()
c4 = g.given(4)
c2h = g.given(2)
h = g.free()
g.rel("mul", c2h, h, c4)              # 2h=4 -> h=2
c18 = g.given(18)
k = g.fdiv(c18, 2)                    # 9 (ONE fdiv, k=2)
hsq = g.rel("mul", h, h)              # 4
ksq = g.rel("mul", k, k)              # 81
s1 = g.rel("add", hsq, ksq)           # 85
c21 = g.given(21)
rsq = g.rel("sub", s1, c21)           # 64
r = g.free()
g.rel("mul", r, r, rsq)               # r^2=64 -> r=8
add(987, g.factors, r, 8,
    "THEOREM-APPLICATION (named: completing the square for a circle, "
    "r^2=h^2+k^2-F), degree: h via multiplicative inversion, k via the "
    "one allowed fdiv, h^2+k^2-21 assembled genuinely, then r via "
    "search-based square-root extraction (technique 1) -- multiple "
    "independent real steps, not a bare echo of any single coefficient.")

print(f"\nDrafted so far: {len(rows)}  Skipped so far: {len(skips)}  Fails: {len(fails)}")

# 989: piecewise f; f(-7)+f(0)+f(7). Routing (which branch applies) is
# derivable purely from comparing FIXED source literals (-7 vs -5; 0 vs
# [-5,5]; 7 vs 5), stated + flagged per Law 13, not an in-graph
# primitive. f(0)'s branch value is NEGATIVE (2*0-3=-3); magnitude-fold
# (technique 3) renders it as a subtraction at the final combining step.
g = G()
c3branch1 = g.given(3)                # f(-7)=3 (constant branch, x<-5;
                                       # source literal, branch's own
                                       # constant formula value)
c7 = g.given(7)
sq7 = g.rel("mul", c7, c7)            # 49
c1 = g.given(1)
f7 = g.rel("add", sq7, c1)            # f(7)=50
c2coef = g.given(2)                   # branch2's coefficient (2x-3)
c0 = g.given(0)                       # x=0, the input for branch2
twoX = g.rel("mul", c2coef, c0)       # 0
c3branch2 = g.given(3)                # branch2's constant '-3' magnitude
magF0 = g.rel("sub", c3branch2, twoX) # |2*0-3| = 3 (twoX < c3branch2)
partial = g.rel("add", c3branch1, f7) # 53
ans = g.rel("sub", partial, magF0)    # 53-3 = 50 (fold the negative f(0)
                                       # back in via subtraction)
add(989, g.factors, ans, 50,
    "MAGNITUDE-FOLD (technique 3, matches [897]/[902]): f(0)=2*0-3=-3 is "
    "negative, so rather than representing it directly, the graph tracks "
    "its MAGNITUDE (|2*0-3|=3, since twoX=0 < the branch's own constant "
    "3) and folds it back into the running sum via SUBTRACTION instead of "
    "addition at the final combining step. f(-7) and f(7) computed "
    "directly: the x<-5 branch is a bare constant (3) per the piecewise "
    "definition itself, and the x>5 branch (x^2+1) is a genuine two-step "
    "computation from the given x=7.",
    routing_fact="branch selection for all three evaluations (x=-7<-5 -> "
                 "constant branch; x=0 in [-5,5] -> 2x-3 branch; x=7>5 -> "
                 "x^2+1 branch) is derivable purely from comparing FIXED "
                 "source literals against the piecewise boundaries (no "
                 "unknowns involved anywhere in the routing decision); "
                 "stated and flagged per Law 13, not encoded as an "
                 "in-graph comparison primitive. Passes the counterfactual "
                 "test: a variant x-value crossing into a DIFFERENT "
                 "branch would change which sentences render (unlike a "
                 "residue), while a variant x-value STAYING within "
                 "branch1's region wouldn't change f(-7)'s render at all "
                 "-- correctly reflecting that branch1's formula is a "
                 "true constant, independent of the specific x chosen "
                 "within that region.")

# 991: (3/2)x^2+11x+c=0, roots=(-11+-sqrt7)/3. Find c. Quadratic-formula
# denominator (3=2a) and discriminant (7, read directly off the given
# root form) pin c via b^2-4ac=disc, i.e. 4ac=b^2-disc, divided by 4a=6
# (the one allowed fdiv, single-digit).
g = G()
c11 = g.given(11)
bsq = g.rel("mul", c11, c11)          # 121
c7 = g.given(7)
diff = g.rel("sub", bsq, c7)          # 114 (=4ac=b^2-disc)
ans = g.fdiv(diff, 6)                 # 19 (ONE fdiv, k=6=4a, single-digit)
add(991, g.factors, ans, 19,
    "THEOREM-APPLICATION (named: quadratic formula, roots=(-b+-sqrt(disc))"
    "/(2a)), degree: b=11 and disc=7 both read directly off the given "
    "root form (the denominator 3 IS 2a, confirming a=3/2 without a "
    "separate given), so 4ac=b^2-disc=114 is genuine arithmetic, then "
    "divided by 4a=6 (single-digit, the one allowed fdiv) to isolate c.")

# 1009: SKIP -- (7/8)^3 * (7/8)^-3 = 1.
skip(1009,
     "Exponent law a^m*a^-n=a^(m-n): 3+(-3)=0, and any nonzero base^0=1 "
     "-- the answer (1) is a bare structural identity, INVARIANT to the "
     "base entirely (7/8 could be replaced by any nonzero value with an "
     "identical answer). Explicit counterfactual-test failure: a variant "
     "with a different base would render IDENTICALLY (the base never "
     "needs to appear in the graph at all), confirming this is pure "
     "Law-3 residue, matching [914]/[1045] this tranche.")

# 1011: SKIP -- gcd(91,72).
skip(1011,
     "91=7*13, 72=2^3*3^2, no shared prime factors, gcd=1. No gcd/"
     "factorization primitive exists in the registry (add/sub/mul/fdiv/"
     "pct/sel/mod only); the Euclidean algorithm is an open, variable-"
     "length iterative search, not a fixed-shape graph. New family this "
     "tranche (see also [1073]/[1086]/[1103], a 4-item GCD/factorization "
     "cluster). Operation-shaped skip.")

print(f"\nDrafted so far: {len(rows)}  Skipped so far: {len(skips)}  Fails: {len(fails)}")

# 1013: mean of 7 noon temperatures. CAP-AVOIDANCE (caught by solve2
# itself returning None, matching t7's [934] lesson): the raw sum (588)
# is far over the 300 cap, so the chain is rebuilt in DEVIATION-FROM-
# BASELINE space (baseline=79, the coldest day) -- every intermediate
# stays under 35, well clear of the cap.
g = G()
c79 = g.given(79)                     # baseline (also day 2's own temp)
c80 = g.given(80)
d1 = g.rel("sub", c80, c79)           # 1
d2 = g.rel("sub", c79, c79)           # 0 (self-combination, Law 1)
c81 = g.given(81)
d3 = g.rel("sub", c81, c79)           # 2
c85 = g.given(85)
d4 = g.rel("sub", c85, c79)           # 6
c87a = g.given(87)
d5 = g.rel("sub", c87a, c79)          # 8
c89 = g.given(89)
d6 = g.rel("sub", c89, c79)           # 10
c87b = g.given(87)
d7 = g.rel("sub", c87b, c79)          # 8
s1 = g.rel("add", d1, d2)
s2 = g.rel("add", s1, d3)
s3 = g.rel("add", s2, d4)
s4 = g.rel("add", s3, d5)
s5 = g.rel("add", s4, d6)
sumDev = g.rel("add", s5, d7)         # 35
meanDev = g.fdiv(sumDev, 7)           # 5 (ONE fdiv, k=7)
ans = g.rel("add", c79, meanDev)      # 84
add(1013, g.factors, ans, 84,
    "CAP-AVOIDANCE (caught by solve2 returning None on the naive raw-sum "
    "render, matching t7's [934] lesson): the raw sum of all 7 "
    "temperatures is 588, far over the 300 cap, so the mean is computed "
    "in DEVIATION-FROM-BASELINE space instead (baseline=79, the week's "
    "coldest day) -- every intermediate stays under 35. The one allowed "
    "fdiv (k=7) divides the summed deviations, then the baseline is added "
    "back.",
    watch="pointer-collision: c79 (the baseline) serves as an argument "
          "in SEVEN separate sub factors (one per day, including a "
          "trivial self-cancellation for day 2 itself) -- a heavier "
          "instance than the established 2-3-use subtype, but the reuse "
          "here is CAP-AVOIDANCE-FORCED (a shared reference point is the "
          "only way to keep 7 large temperatures under 300), not "
          "convenience-driven; worth the bench distinguishing this "
          "origin from [786]/[815]/[925]'s milder reuse.")

# 1017: Bryce = Carter+6; Carter = half of Bryce. Rendered via 2*Carter=
# Bryce (direct multiplication, NOT fdiv) -- fdiv would introduce a
# spurious second solution (B=11 also satisfies floor(11/2)+6=11), a
# floor-division ambiguity pitfall avoided here.
g = G()
c2 = g.given(2)
carter = g.free()
bryce = g.rel("mul", c2, carter)      # 2*Carter = Bryce (exact, no floor)
c6 = g.given(6)
g.rel("add", carter, c6, bryce)       # Carter+6 = Bryce
add(1017, g.factors, bryce, 12,
    "direct system: 'Carter received half the raisins Bryce received' is "
    "rendered as 2*Carter=Bryce (exact multiplication), not fdiv(Bryce,2) "
    "-- fdiv's FLOOR semantics would admit a spurious second solution "
    "(Bryce=11: floor(11/2)=5, 5+6=11, also satisfies the floor-based "
    "constraint), breaking uniqueness. Choosing multiplication over "
    "division for an exact-halving relation is the general fix, flagged "
    "for the bench as a technique note.")

# 1018: 3 digits {2,4,7}, no repeat, count of 2-digit integers. Direct
# permutation formula nP2=n*(n-1) -- only the COUNT of digits (3) matters
# mathematically, never their specific identities (any 3 distinct digits
# give the same count), so the render correctly omits the digit VALUES
# entirely (passes the counterfactual test: changing the digit identities
# wouldn't change the render, changing the COUNT would).
g = G()
c3 = g.given(3)
c1 = g.given(1)
nMinus1 = g.rel("sub", c3, c1)        # 2
ans = g.rel("mul", c3, nMinus1)       # 6
add(1018, g.factors, ans, 6,
    "THEOREM-APPLICATION (named: permutation formula nPk=n*(n-1) for "
    "k=2), degree: direct arithmetic assembly from the digit COUNT (3), "
    "not a search over arrangements. Only the count matters (any 3 "
    "distinct digits yield 6 two-digit arrangements), so the render "
    "correctly never references the specific digit values 2,4,7 -- "
    "verified against the counterfactual test (varying the count changes "
    "the render, varying WHICH digits doesn't).")

# 1034: Kim 87,83,88 (sum=258, ONE fdiv by 3 -> oldAvg=86); 4th score 90;
# find the increase. CAP-AVOIDANCE (solve2 caught 348=new sum over the
# 300 cap, same lesson as [1013] above): rather than forming the new sum
# (348) at all, the increase is derived algebraically as (S-oldAvg)/(n+1)
# = (90-86)/4 = 1 -- a small, safe intermediate -- via a difference then
# multiplicative inversion (technique C), never touching 348.
g = G()
c87 = g.given(87)
c83 = g.given(83)
c88 = g.given(88)
s1 = g.rel("add", c87, c83)
sum3 = g.rel("add", s1, c88)          # 258
oldAvg = g.fdiv(sum3, 3)              # 86 (ONE fdiv, k=3)
c90 = g.given(90)
diff = g.rel("sub", c90, oldAvg)      # 4 (small: S-oldAvg, never forms
                                       # the 348 new-sum intermediate)
c4 = g.given(4)
x = g.free()
g.rel("mul", c4, x, diff)             # 4x=4 -> x=1 (mult inversion)
add(1034, g.factors, x, 1,
    "CAP-AVOIDANCE (caught by solve2 returning None on the naive "
    "new-sum render, matching [1013] this tranche and t7's [934] "
    "lesson): forming sum4=258+90=348 directly exceeds the 300 cap even "
    "though every given and the final answer are small. Instead the "
    "increase is derived via the identity increase=(S-oldAvg)/(n+1) "
    "(algebraically equivalent, standard average-update identity), "
    "computed as diff=90-86=4 (small, safe) then MULTIPLICATIVE "
    "INVERSION (technique C, avoiding a second fdiv and any "
    "floor-division ambiguity): 4*x=4 -> x=1.")

# 1041: SKIP -- minimum dimes for $32.75 using three $10s + eight quarters.
skip(1041,
     "$30 (bills) + $2 (quarters) = $32, need $0.75 more = 75 cents; "
     "minimum dimes = ceil(75/10) = 8 (7 dimes=70cents is insufficient). "
     "This is a MINIMUM-satisfying-a-threshold search (ceiling, not exact "
     "division) -- fdiv is floor-only and its natural k here (10) isn't "
     "single-digit besides, doubly blocking the natural render. Matches "
     "the 'no inequality/threshold-search primitive' family ([877]/[888]/"
     "[990] this book).")

# 1045: SKIP -- obtuse triangle, how many obtuse interior angles?
skip(1045,
     "Always exactly 1 (angle sum=180 forbids two angles both >90). Bare "
     "geometric fact with ZERO numeric content in the source at all (no "
     "numbers present) -- the most extreme Law-3 residue case this "
     "tranche, matches [1009]/[958] family, fails the counterfactual test "
     "trivially (there is nothing in the source to vary).")

print(f"\nDrafted so far: {len(rows)}  Skipped so far: {len(skips)}  Fails: {len(fails)}")

# 1058: SKIP -- 21 rooms, remove 12,13, median of remaining 19.
skip(1058,
     "Median (10th of 19 remaining) = room 10, unaffected since the two "
     "excluded rooms (12,13) both exceed the median position. COUNTER-"
     "FACTUAL TEST CAUGHT A FALSE RENDER during drafting: a naive graph "
     "using only the COUNT of excluded rooms (2) -- remaining=21-2=19, "
     "medianPos=fdiv(19+1,2)=10 -- gives the SAME answer (10) regardless "
     "of WHICH two rooms were excluded, but the TRUE answer genuinely "
     "depends on which rooms (excluding e.g. rooms 5,6 instead would "
     "shift the median to 12). This is exactly [880]/[935]'s witness-test "
     "family: the graph would secretly rely on the pen's off-graph "
     "verification that 12,13 > median position, without that dependency "
     "being represented -- Law 2 (reconstruction co-test) violation. No "
     "primitive exists for 'find the k-th order statistic of a set after "
     "removing named elements.' NEW FAMILY this tranche: order-statistic/"
     "median-with-exclusions.")

# 1060: irises:roses=2:5, 25 roses now, +20 more roses, find total irises
# after. Original irises via multiplicative inversion (no fdiv), increase
# via the one allowed fdiv on the roses-increase (20 is a clean multiple
# of the ratio's denominator).
g = G()
c25 = g.given(25)
c2 = g.given(2)
t0 = g.rel("mul", c25, c2)            # 50 (=roses*ratio-numerator)
c5r = g.given(5)                      # ratio's roses-side value
irisesOrig = g.free()
g.rel("mul", c5r, irisesOrig, t0)     # 5*irisesOrig=50 -> 10 (mult inv)
c20 = g.given(20)
unitInc = g.fdiv(c20, 5)              # 4 (ONE fdiv, k=5)
irisesInc = g.rel("mul", c2, unitInc) # 8 (reuses c2, the ratio's own
                                       # iris-side value)
ans = g.rel("add", irisesOrig, irisesInc)  # 18
add(1060, g.factors, ans, 18,
    "THEOREM-APPLICATION (named: ratio scaling, irises/roses=2/5 held "
    "constant), degree: original iris count via multiplicative inversion "
    "(technique C, avoiding fdiv), the NEW roses (20) scaled to new "
    "irises via the one allowed fdiv (k=5, single-digit); total is the "
    "genuine sum of two independently-derived quantities.",
    watch="pointer-collision: c2 (the ratio's iris-side value) serves as "
          "an argument in 2 separate mul factors (original-irises setup "
          "and increase-irises) -- matches [786]/[815]/[929]'s "
          "iterated-constant-reuse subtype (mild, 2 uses).")

# 1073: SKIP -- distinct prime factors of 210.
skip(1073,
     "210=2*3*5*7, 4 distinct primes. No factorization primitive exists; "
     "hardcoding the chain 2*3*5*7=210 would GIFT the very factorization "
     "the problem asks the solver to discover (Law 2 violation), and "
     "counting the number of factors used in such a hardcoded chain is "
     "circular -- the chain's shape IS the answer. Matches [1011]'s GCD/"
     "factorization family exactly.")

# 1074: 100 students, 65 math, 43 physics, 10 both, find neither. Direct
# inclusion-exclusion (Law 5, named), all specific literals used directly.
g = G()
c65 = g.given(65)
c43 = g.given(43)
c10 = g.given(10)
union = g.rel("add", c65, c43)
unionNet = g.rel("sub", union, c10)   # 65+43-10=98
c100 = g.given(100)
ans = g.rel("sub", c100, unionNet)    # 2
add(1074, g.factors, ans, 2,
    "THEOREM-APPLICATION (named: inclusion-exclusion, |A union B| = "
    "|A|+|B|-|A and B|), degree: union computed genuinely from all three "
    "given counts, then subtracted from the total; every specific source "
    "literal (100,65,43,10) enters the arithmetic directly.")

# 1086: SKIP -- largest package size s.t. 40 and 24 pencils both divide
# evenly (GCD).
skip(1086,
     "gcd(40,24)=8. Same no-gcd/factorization-primitive family as "
     "[1011]/[1073] this tranche (3rd instance).")

# 1103: SKIP -- GCF(30,90,75).
skip(1103,
     "gcd(30,90,75)=15 (3-way GCD). Same family, 4th instance this "
     "tranche ([1011]/[1073]/[1086]/[1103]) -- notable cluster, worth "
     "flagging for the bench as a recurring number-theory gap in the "
     "primitive registry.")

# 1106: SKIP -- largest possible median of {x,2x,3,2,5}, x any integer.
skip(1106,
     "Maximizing the median over an UNBOUNDED choice of x (x can be any "
     "integer, including negative) requires case analysis on x's "
     "ordering relative to 2,3,5 -- an argmax over branches with no "
     "search primitive (matches [941]'s tranche7 family), compounded by "
     "x's domain being explicitly unbounded/signed (outside the nonneg "
     "CSP domain by the source's own wording). Operation-shaped skip.")

# 1114: SKIP -- 63_ multiple of 3, greatest difference between two
# possible units digits.
skip(1114,
     "Valid units digits (0-9) with 6+3+d divisible by 3: d in "
     "{0,3,6,9}, greatest difference=9-0=9. Requires enumerating the "
     "full solution SET (all d satisfying a mod-3 condition) then taking "
     "its range (max-min) -- an enumerate-and-range operation with no "
     "primitive (matches [941]/[1106]'s argmax-over-solution-set family). "
     "The apparent shortcut (9-0=the digit-range boundary) is a "
     "coincidence for THIS modulus, not a general render -- fails the "
     "counterfactual test the same way [1058] did (a different divisor "
     "requirement would break the boundary-reuse shortcut).")

print(f"\nDrafted so far: {len(rows)}  Skipped so far: {len(skips)}  Fails: {len(fails)}")

# ===========================================================================
# DIET ROWS (10) -- interleaved baseline tranche, experimental clause
# governs accommodation guidance for these only.
# ===========================================================================

# 1085: SKIP (diet: subtrahend distractors) -- 5 consecutive 2-digit
# integers <30, none prime, find largest.
skip(1085,
     "Run is 24,25,26,27,28 (bracketed by primes 23,29), largest=28. This "
     "requires a PRIMALITY-TEST + consecutive-run SEARCH -- no primitive "
     "for either, and there is no way to derive the answer via ANY "
     "arithmetic shortcut without first knowing which numbers are prime "
     "(the search itself IS the problem). DIET NOTE: unlike [1133]/"
     "[1165]-family diet rows, this source offers NO natural foothold "
     "for the target construction at all -- it isn't that the subtraction "
     "FORM specifically is unavailable, the entire problem is opaque to "
     "every primitive in the registry (matches the GCD/factorization "
     "cluster's flavor: number-theoretic search, not algebra). Diet "
     "disposition: NOT exercised, source unrenderable regardless of "
     "construction choice.")

# 1173: Angela=a, Brian=twice Angela (ADD-DUP: a+a, per diet clause -- do
# NOT route to 2*a), Caden=3*Brian (NOT a doubling, stays multiplication),
# Daryl=5*Caden (NOT a doubling, stays multiplication). Total=78.
g = G()
a = g.free()
brian = g.rel("add", a, a)            # a+a (ADD-DUP form, args=[a,a],
                                       # for "twice as many" -- native)
c3 = g.given(3)
caden = g.rel("mul", c3, brian)       # "three times" -- NOT a doubling,
                                       # stays multiplication
c5 = g.given(5)
daryl = g.rel("mul", c5, caden)       # "five times" -- NOT a doubling
s1 = g.rel("add", a, brian)
s2 = g.rel("add", s1, caden)
c78 = g.given(78)
g.rel("add", s2, daryl, c78)          # total=78
add(1173, g.factors, a, 2,
    "ADD-DUP native (diet clause): Brian ('twice as many marbles as "
    "Angela') is rendered as a+a (args=[1,1] shape), NOT 2*a. Caden "
    "('three times as many as Brian') and Daryl ('five times as many as "
    "Caden') correctly stay as ordinary multiplication -- ONLY the "
    "doubling relation gets the add-dup treatment, a precise "
    "discrimination showing the diet construction isn't overapplied to "
    "every multiplication in the row.")

# 1229: 50% eliminated round1 (survivors1=50% of T, using the coincidence "
# that 50 is self-complementary: eliminated=remaining=50%). 1/3 of
# survivors1 remain after round2 = 24 (given). Native pct for the ONLY
# genuine percent literal (50); the "1/3" fraction is correctly rendered
# via lexical multiplication (law 9: p must be a source literal, and
# "33.33%" is never stated, only "1/3"), working FORWARD from the known
# 24 to avoid any floor-division ambiguity.
g = G()
survivors2 = g.given(24)
c3 = g.given(3)
survivors1 = g.rel("mul", c3, survivors2)   # 72 (exact: 3*24, "1/3"
                                             # lexically explicated)
c50 = g.given(50)
T = g.free()
g.pct(survivors1, T, 50)              # survivors1 is 50 percent of T
                                       # (NATIVE pct, p=50 a source
                                       # literal)
add(1229, g.factors, T, 144,
    "PCT NATIVE (diet clause): the ratified pct primitive used for the "
    "genuine percent literal (50, 'eliminated after round1' -- and 50 is "
    "self-complementary so 'survivors1 is 50 percent of T' correctly "
    "represents the REMAINING half too). The '1/3 of the remaining' "
    "fraction is NOT forced into pct (p=33 would not be a source "
    "literal, violating Law 9) -- it is lexically explicated via direct "
    "multiplication instead, working FORWARD from the known final count "
    "(24) to derive survivors1=72 exactly, avoiding any floor-division "
    "ambiguity a fdiv-based inverse would risk.")

# 1133: average of 23 and x is 27, find |23-x|. DIET (two-free-var
# systems): x and d (the difference) are handed to the solver JOINTLY as
# TWO free variables pinned by two simultaneous equations, rather than
# pre-solving x in closed form and subtracting after. Best-faith framing
# for a source with only one TRUE unknown (23 is a source literal, not a
# second independent unknown) -- see notes.
g = G()
c23 = g.given(23)
c2 = g.given(2)
c27 = g.given(27)
rhs = g.rel("mul", c2, c27)           # 54 (avg*2=sum)
x = g.free()
d = g.free()
g.rel("add", c23, d, x)               # x = 23+d (x exceeds 23 by d)
g.rel("add", c23, x, rhs)             # 23+x = 54 (joint constraint,
                                       # shares x with the line above)
add(1133, g.factors, d, 8,
    "TWO-FREE-VARIABLE SYSTEM (diet clause, [772]/[856] pattern): x and "
    "d handed to the solver JOINTLY (both free, pinned simultaneously by "
    "x=23+d and 23+x=54) rather than pre-solved (x found first, then d "
    "subtracted after). HONEST CAVEAT: this source has only ONE truly "
    "independent unknown in the algebra (23 is a source literal, not a "
    "second freely-varying quantity, unlike [920]'s x^2+y^2=193/xy=84 "
    "where BOTH x,y are genuinely unknown) -- this is the best-faith "
    "joint-system rendering available for a single-true-unknown source, "
    "not a claim that x and d vary independently. Flagged for the bench "
    "as a weaker instance of the pattern than [1165]'s pct rows are of "
    "theirs.")

print(f"\nDrafted so far: {len(rows)}  Skipped so far: {len(skips)}  Fails: {len(fails)}")

# 1165: snowboard $100, discounted 50% Friday, then Monday's price reduced
# 30%. DIET (pct pointers): BOTH discounts rendered via the ratified pct
# primitive, verbatim, with p as a source literal both times (50, 30) --
# the flagship diet demonstration, no accommodation needed at all.
g = G()
c100 = g.given(100)
c50 = g.given(50)
discount1 = g.free()
g.pct(discount1, c100, 50)            # discount1 is 50 percent of 100
friPrice = g.rel("sub", c100, discount1)  # 50
c30 = g.given(30)
discount2 = g.free()
g.pct(discount2, friPrice, 30)        # discount2 is 30 percent of
                                       # friPrice
monPrice = g.rel("sub", friPrice, discount2)  # 35
add(1165, g.factors, monPrice, 35,
    "PCT NATIVE, DOUBLE (diet clause flagship): both discounts use the "
    "ratified pct primitive verbatim, with p a source literal each time "
    "(50, then 30) -- 'discount1 is 50 percent of 100' then 'discount2 "
    "is 30 percent of friPrice'. No accommodation of any kind; this is "
    "the cleanest possible native exercise of the target construction, "
    "expected to produce the registered scattered gate vote per the "
    "diet's baseline hypothesis.")

# 1212: 60% of 200 passengers are women, 10% of those women are first
# class. DIET (pct pointers): double NATIVE pct chain, both p values (60,
# 10) source literals.
g = G()
c200 = g.given(200)
c60 = g.given(60)
women = g.free()
g.pct(women, c200, 60)                # women is 60 percent of 200
c10 = g.given(10)
fcWomen = g.free()
g.pct(fcWomen, women, 10)             # fcWomen is 10 percent of women
add(1212, g.factors, fcWomen, 12,
    "PCT NATIVE, DOUBLE (diet clause): women derived via pct(women,200,"
    "p=60), then first-class women via pct(fcWomen,women,p=10) -- both p "
    "values are source literals, a clean chained-pct exercise (women "
    "itself is an intermediate pct RESULT feeding a second pct's BASE, "
    "the exact pointer pattern the diet targets).")

# 1271: 27 increased by twice a number is 39. DIET (add-dup): "twice a
# number" rendered as n+n, not 2*n.
g = G()
n = g.free()
twoN = g.rel("add", n, n)             # n+n (ADD-DUP form)
c27 = g.given(27)
c39 = g.given(39)
g.rel("add", c27, twoN, c39)          # 27+2n=39
add(1271, g.factors, n, 6,
    "ADD-DUP native (diet clause): 'twice a number' rendered as n+n "
    "(args=[n,n]), not 2*n -- direct, single-relation exercise of the "
    "target construction, simplest instance this tranche.")

# 1286: SKIP (diet: two-free-variable systems) -- 5 positive integers,
# product even, max possible odd count.
skip(1286,
     "Product even requires >=1 even factor (a THEOREM constant, not a "
     "second independently-varying unknown), so maxOdd=5-1=4. DIET "
     "MISMATCH: unlike [1133] (which has a genuine, if weak, second "
     "derived quantity to pair with x), this source has exactly ONE "
     "true degree of freedom -- how many odds to choose -- with 'evens "
     "needed' FIXED at 1 by a logical theorem, not a second free "
     "variable that could take other values. Forcing a 2-free-var frame "
     "(oddCount, evenCount jointly free, sum=5, evenCount=1) would make "
     "the 'jointness' cosmetic, not real (Law 2 reconstruction "
     "violation -- evenCount isn't actually searched, it's asserted). "
     "COMPOUND-blocked regardless: the underlying shape is an argmax "
     "under a >= inequality (matches [941]/[1106]'s no-argmax-under-"
     "inequality family) even setting the diet mismatch aside. Honest "
     "skip; diet disposition: NOT exercised, source incompatible with "
     "the target construction (a genuine, reportable finding about "
     "diet_srcs selection, not a drafting failure).")

# 1382: square area=25, rectangle same width, length=2*width. DIET
# (add-dup): length rendered as width+width, not 2*width.
g = G()
c25 = g.given(25)
side = g.free()
g.rel("mul", side, side, c25)         # side^2=25 -> side=5 (search,
                                       # technique 1)
length = g.rel("add", side, side)     # side+side (ADD-DUP form, not
                                       # 2*side)
ans = g.rel("mul", side, length)      # width*length = 5*10 = 50
add(1382, g.factors, ans, 50,
    "ADD-DUP native (diet clause): rectangle length ('double its width') "
    "rendered as side+side (args=[side,side]), not 2*side. Square side "
    "found via search-based root extraction (technique 1, side^2=25), "
    "then area assembled as width*length -- a genuine multi-step render "
    "with add-dup at its core relation.")

print(f"\nFINAL -- Drafted: {len(rows)}  Skipped: {len(skips)}  Fails: {len(fails)}")
print(f"Total accounted: {len(rows) + len(skips) + len(fails)} / 40")
if fails:
    print("FAILS:", fails)

with open('/home/bryce/mycelium/.cache/book8_t8_prose_pairs_draft.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print("Wrote", len(rows), "rows to .cache/book8_t8_prose_pairs_draft.jsonl")

drafted_ids = {r["gen"]["src_idx"] for r in rows}
skipped_ids = {s[0] for s in skips}
all_ids = {c["src_idx"] for c in CANDS.values()}
missing = all_ids - drafted_ids - skipped_ids
overlap = drafted_ids & skipped_ids
print("Missing (neither drafted nor skipped):", sorted(missing))
print("Overlap (both drafted and skipped):", sorted(overlap))

diet_drafted = [r["gen"]["src_idx"] for r in rows if "diet_customer" in r["gen"]]
diet_skipped = [s[0] for s in skips if s[0] in DIET_SRCS]
print(f"\nDiet: {len(diet_drafted)} drafted {sorted(diet_drafted)}, "
      f"{len(diet_skipped)} skipped {sorted(diet_skipped)}")
normal_drafted = [r["gen"]["src_idx"] for r in rows if "diet_customer" not in r["gen"]]
normal_skipped = [s[0] for s in skips if s[0] not in DIET_SRCS]
print(f"Normal: {len(normal_drafted)} drafted, {len(normal_skipped)} skipped")
