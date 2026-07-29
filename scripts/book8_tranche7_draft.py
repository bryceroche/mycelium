import json, sys, string
sys.path.insert(0, '/home/bryce/mycelium')
sys.path.insert(0, '/home/bryce/mycelium/scripts')
from tta_alg2_dials import solve2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic

SMP = {"n_vars": 24, "m": 300}
LETTERS = string.ascii_lowercase

CANDS = {c["src_idx"]: c for c in
         json.load(open('/home/bryce/mycelium/.cache/book8_candidates_t7.json'))["tranche7"]}


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
        "src_idx": src_idx, "book": 8, "tranche": 7, "floor": "prime", "fs": True,
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
# 869: SKIP -- f(sqrt(29)); sqrt(29) not integer -> f=floor(sqrt(29))+5=5+5=10.
skip(869,
     "f(sqrt(29)) routes to the floor(x)+5 branch since sqrt(29) is not an "
     "integer; floor(sqrt(29))=5 (5^2=25<=29<36=6^2). No floor primitive "
     "exists (matches [716]/[725]'s tranche5 floor/sqrt family exactly -- "
     "the underlying integer IS representable, 5, but the irrational sqrt "
     "and the floor operation itself have no supporting primitive). "
     "Operation-shaped skip.")

# 870: sum=25, product=126, |difference|=11. Vieta system, matches
# [714]/[776]/[819] self-resolving-ordering pattern; query is the
# difference itself, not a bare coefficient echo.
g = G()
c25 = g.given(25)
c126 = g.given(126)
rSmall = g.free()
rDiff = g.free()
rBig = g.rel("add", rSmall, rDiff)
g.rel("add", rBig, rSmall, c25)
g.rel("mul", rBig, rSmall, c126)
add(870, g.factors, rDiff, 11,
    f"Consider the numbers {L(c25)}, {L(c126)}, {L(rSmall)}, {L(rDiff)}, "
    f"{L(rBig)}. {L(c25)} is 25. {L(c126)} is 126. {L(rSmall)} plus "
    f"{L(rDiff)} equals {L(rBig)}. {L(rBig)} plus {L(rSmall)} equals "
    f"{L(c25)}. {L(rBig)} times {L(rSmall)} equals {L(c126)}. What is "
    f"{L(rDiff)}?",
    "THEOREM-APPLICATION (named: Vieta sum/product for sum=25, "
    "product=126), degree: self-resolving-ordering system (matches "
    "[714]/[776]/[819] pattern) on rSmall/rDiff/rBig; query is the "
    "difference itself, a genuinely derived quantity requiring the full "
    "2-var system, not a bare coefficient echo.")

# 872: 2x-y=5, x+2y=5 -> x=3. Direct system encoding (technique 2) on
# both of the source's own stated equations; separate given vars for the
# coefficient '2' (distinct roles: coeff of x vs coeff of y) and for the
# RHS '5' (distinct source facts despite equal value).
g = G()
c2a = g.given(2)
x = g.free()
y = g.free()
twoX = g.rel("mul", c2a, x)
c5a = g.given(5)
g.rel("sub", twoX, y, c5a)          # 2x - y = 5
c2b = g.given(2)
twoY = g.rel("mul", c2b, y)
c5b = g.given(5)
g.rel("add", x, twoY, c5b)          # x + 2y = 5
add(872, g.factors, x, 3,
    f"Consider the numbers {L(c2a)}, {L(x)}, {L(y)}, {L(twoX)}, {L(c5a)}, "
    f"{L(c2b)}, {L(twoY)}, {L(c5b)}. {L(c2a)} is 2. {L(c2a)} times {L(x)} "
    f"equals {L(twoX)}. {L(c5a)} is 5. {L(twoX)} exceeds {L(y)} by "
    f"{L(c5a)}. {L(c2b)} is 2. {L(c2b)} times {L(y)} equals {L(twoY)}. "
    f"{L(c5b)} is 5. {L(x)} plus {L(twoY)} equals {L(c5b)}. What is "
    f"{L(x)}?",
    "direct system encoding (technique 2): x,y searched jointly against "
    "BOTH of the source's own equations (2x-y=5, x+2y=5); separate given "
    "vars for the coefficient '2' (distinct roles in each equation) and "
    "for the RHS '5' (distinct source facts), matching precedent for "
    "same-valued-distinct-role literals.")

# 873: domain excludes roots of 2x^2-6x+4=0 (A=1,B=2), A+B=3. A naive
# render would just divide the coefficient 6 by 2 directly (Law-3
# residue -- same coefficient-echo risk as [796]/[905]/[943]). Instead:
# work with SCALED roots S=a*r (S1+S2=-b=6 [not used directly], S1*S2=a*c=8,
# S2-S1=sqrt(disc)=2), pin S1,S2 via product+difference only (never via
# the sum), then unscale with the one allowed fdiv.
g = G()
bMag = g.given(6)
aCoef = g.given(2)
cConst = g.given(4)
bsq = g.rel("mul", bMag, bMag)        # 36
ac = g.rel("mul", aCoef, cConst)      # 8  (a*c, product of scaled roots)
c4uni = g.given(4)                    # universal '4' in b^2-4ac (distinct
                                       # role from cConst despite equal value)
ac4 = g.rel("mul", c4uni, ac)         # 32
disc = g.rel("sub", bsq, ac4)         # 4
sqrtDisc = g.free()
g.rel("mul", sqrtDisc, sqrtDisc, disc)  # sqrtDisc^2=4 -> 2 (=S2-S1)
s1 = g.free()
s2 = g.rel("add", s1, sqrtDisc)       # s1+2=s2
g.rel("mul", s1, s2, ac)              # s1*s2=8 -> s1=2,s2=4 (scaled roots)
sumScaled = g.rel("add", s1, s2)      # 6
ans = g.fdiv(sumScaled, 2)            # unscale by aCoef=2: 6/2=3 (ONE fdiv)
add(873, g.factors, ans, 3,
    f"Consider the numbers {L(bMag)}, {L(aCoef)}, {L(cConst)}, {L(bsq)}, "
    f"{L(ac)}, {L(c4uni)}, {L(ac4)}, {L(disc)}, {L(sqrtDisc)}, {L(s1)}, "
    f"{L(s2)}, {L(sumScaled)}, {L(ans)}. {L(bMag)} is 6. {L(aCoef)} is "
    f"2. {L(cConst)} is 4. {L(bMag)} times {L(bMag)} equals {L(bsq)}. "
    f"{L(aCoef)} times {L(cConst)} equals {L(ac)}. {L(c4uni)} is 4. "
    f"{L(c4uni)} times {L(ac)} equals {L(ac4)}. {L(bsq)} exceeds "
    f"{L(ac4)} by {L(disc)}. {L(sqrtDisc)} times {L(sqrtDisc)} equals "
    f"{L(disc)}. {L(s1)} plus {L(sqrtDisc)} equals {L(s2)}. {L(s1)} "
    f"times {L(s2)} equals {L(ac)}. {L(s1)} plus {L(s2)} equals "
    f"{L(sumScaled)}. When {L(sumScaled)} is divided by 2, the quotient "
    f"is {L(ans)}. What is {L(ans)}?",
    "LAW 3 TENSION (resolved, flagged for the wheel): a naive render "
    "would divide the coefficient 6 by 2 directly and echo the result -- "
    "same coefficient-echo risk as [796]/[905]/[943] this book. Instead "
    "this row tracks SCALED roots S=a*r (S1*S2=a*c=8, S2-S1=sqrt(disc)=2) "
    "pinned via product+difference alone (sum-of-roots '6' never fed in "
    "as a constraint, only the discriminant-derived difference and the "
    "product), then unscales with the one allowed fdiv (k=2=aCoef). "
    "THEOREM-APPLICATION (named: quadratic discriminant + Vieta on "
    "scaled roots) embedded; cap-avoidance note: the reduced-fraction "
    "form (x^2-3x+2=0) would need x^2 terms fractional, so the ORIGINAL "
    "integer coefficients (2,-6,4) are tracked throughout instead.")

# 875: g(x)=3x+7, f(x)=5x-9, f(g(8)) -> 146. Direct nested composition.
g = G()
c8 = g.given(8)
c3 = g.given(3)
t1 = g.rel("mul", c3, c8)             # 24
c7 = g.given(7)
g8 = g.rel("add", t1, c7)             # g(8)=31
c5 = g.given(5)
t2 = g.rel("mul", c5, g8)             # 155
c9 = g.given(9)
ans = g.rel("sub", t2, c9)            # f(31)=146
add(875, g.factors, ans, 146,
    f"Consider the numbers {L(c8)}, {L(c3)}, {L(t1)}, {L(c7)}, {L(g8)}, "
    f"{L(c5)}, {L(t2)}, {L(c9)}, {L(ans)}. {L(c8)} is 8. {L(c3)} is 3. "
    f"{L(c3)} times {L(c8)} equals {L(t1)}. {L(c7)} is 7. {L(t1)} plus "
    f"{L(c7)} equals {L(g8)}. {L(c5)} is 5. {L(c5)} times {L(g8)} "
    f"equals {L(t2)}. {L(c9)} is 9. {L(t2)} exceeds {L(c9)} by {L(ans)}. "
    f"What is {L(ans)}?",
    "direct computation of the source's own nested function composition "
    "(f(g(8))), every coefficient a source literal from f and g's "
    "definitions; all values well under the cap (max intermediate 155).")

# 877: SKIP -- 3 < sqrt(2x) < 4, count of integer x.
skip(877,
     "9<2x<16 -> 4.5<x<8 -> x in {5,6,7}, 3 integers. Counting integers "
     "satisfying a strict double inequality -- same 'no inequality-"
     "satisfying-integer-search primitive' family as tranche5's [712] "
     "and tranche6's [814]/[838].")

# 878: 2x+y=4, x+2y=5 -> x=1,y=2; find 5x^2+8xy+5y^2=41. Direct system
# on both equations, x,y both genuinely needed (unlike a residue, the
# query requires BOTH values individually, not a coefficient echo).
g = G()
c2a = g.given(2)
x = g.free()
y = g.free()
twoX = g.rel("mul", c2a, x)
c4 = g.given(4)
g.rel("add", twoX, y, c4)             # 2x+y=4
c2b = g.given(2)
twoY = g.rel("mul", c2b, y)
c5 = g.given(5)
g.rel("add", x, twoY, c5)             # x+2y=5
sqX = g.rel("mul", x, x)              # 1
c8 = g.given(8)
xy = g.rel("mul", x, y)               # 2
eightXY = g.rel("mul", c8, xy)        # 16
sqY = g.rel("mul", y, y)              # 4
c5c = g.given(5)
fiveSqX = g.rel("mul", c5c, sqX)      # 5
c5d = g.given(5)
fiveSqY = g.rel("mul", c5d, sqY)      # 20
t1 = g.rel("add", fiveSqX, eightXY)   # 21
ans = g.rel("add", t1, fiveSqY)       # 41
add(878, g.factors, ans, 41,
    f"Consider the numbers {L(c2a)}, {L(x)}, {L(y)}, {L(twoX)}, {L(c4)}, "
    f"{L(c2b)}, {L(twoY)}, {L(c5)}, {L(sqX)}, {L(c8)}, {L(xy)}, "
    f"{L(eightXY)}, {L(sqY)}, {L(c5c)}, {L(fiveSqX)}, {L(c5d)}, "
    f"{L(fiveSqY)}, {L(t1)}, {L(ans)}. {L(c2a)} is 2. {L(c2a)} times "
    f"{L(x)} equals {L(twoX)}. {L(c4)} is 4. {L(twoX)} plus {L(y)} "
    f"equals {L(c4)}. {L(c2b)} is 2. {L(c2b)} times {L(y)} equals "
    f"{L(twoY)}. {L(c5)} is 5. {L(x)} plus {L(twoY)} equals {L(c5)}. "
    f"{L(x)} times {L(x)} equals {L(sqX)}. {L(c8)} is 8. {L(x)} times "
    f"{L(y)} equals {L(xy)}. {L(c8)} times {L(xy)} equals {L(eightXY)}. "
    f"{L(y)} times {L(y)} equals {L(sqY)}. {L(c5c)} is 5. {L(c5c)} "
    f"times {L(sqX)} equals {L(fiveSqX)}. {L(c5d)} is 5. {L(c5d)} times "
    f"{L(sqY)} equals {L(fiveSqY)}. {L(fiveSqX)} plus {L(eightXY)} "
    f"equals {L(t1)}. {L(t1)} plus {L(fiveSqY)} equals {L(ans)}. What "
    f"is {L(ans)}?",
    "direct system encoding (technique 2): x,y searched jointly against "
    "BOTH of the source's own linear equations (2x+y=4, x+2y=5), then "
    "the target quadratic expression 5x^2+8xy+5y^2 is genuinely "
    "assembled from x AND y individually -- distinguishes this from the "
    "Law-3-residue family since neither x nor y alone (nor any bare "
    "source coefficient) equals the answer; both must be pinned and "
    "combined through real multiplication/addition.")

# 880: SKIP -- x^2-2x=0, x!=0 -> x=2.
skip(880,
     "x(x-2)=0 has two nonneg-domain solutions, x=0 and x=2; the source "
     "excludes x=0 by fiat ('x != 0'). Rendered as the naive equation "
     "x*x=2*x, the CSP's domain (0..300) contains BOTH roots, so "
     "solve2's uniqueness check on the query var correctly flags this "
     "as ambiguous (no unique answer) -- there is no primitive for "
     "'exclude this specific extraneous root while leaving the domain "
     "otherwise open' (sel selects between two COMPUTED candidates by "
     "comparison, not exclusion of a fixed known value). Rendering the "
     "pre-simplified x=2 directly would be a bare Law-3 residue (the "
     "division by x, licensed only by the external x!=0 fact, is not "
     "itself representable as graph work). Operation-shaped skip, new "
     "sub-family: extraneous-root exclusion when both roots remain "
     "in-domain.")

# 885: midpoint C=(3,5) of AB, A=(1,8) -> B=(5,2), xy=10. Direct
# midpoint-formula computation (theorem-application), matches [795]'s
# distance-formula-style direct rendering.
g = G()
c3 = g.given(3)
c2 = g.given(2)
twoCx = g.rel("mul", c2, c3)          # 6
c1 = g.given(1)
bx = g.rel("sub", twoCx, c1)          # Bx = 2*3-1=5
c5 = g.given(5)
twoCy = g.rel("mul", c2, c5)          # 10
c8 = g.given(8)
by = g.rel("sub", twoCy, c8)          # By = 2*5-8=2
ans = g.rel("mul", bx, by)            # 10
add(885, g.factors, ans, 10,
    f"Consider the numbers {L(c3)}, {L(c2)}, {L(twoCx)}, {L(c1)}, "
    f"{L(bx)}, {L(c5)}, {L(twoCy)}, {L(c8)}, {L(by)}, {L(ans)}. {L(c3)} "
    f"is 3. {L(c2)} is 2. {L(c2)} times {L(c3)} equals {L(twoCx)}. "
    f"{L(c1)} is 1. {L(twoCx)} exceeds {L(c1)} by {L(bx)}. {L(c5)} is "
    f"5. {L(c2)} times {L(c5)} equals {L(twoCy)}. {L(c8)} is 8. "
    f"{L(twoCy)} exceeds {L(c8)} by {L(by)}. {L(bx)} times {L(by)} "
    f"equals {L(ans)}. What is {L(ans)}?",
    "THEOREM-APPLICATION (named: midpoint formula, C=(A+B)/2 rearranged "
    "to B=2C-A), degree: both coordinates come out nonneg here (5,2), "
    "no magnitude-fold needed, direct computation.")

# 887: g(x)=x^2, f(x)=2x-1, f(g(2)) -> 7. Direct nested composition.
g = G()
c2a = g.given(2)
g2 = g.rel("mul", c2a, c2a)           # g(2)=4
c2b = g.given(2)
t1 = g.rel("mul", c2b, g2)            # 8
c1 = g.given(1)
ans = g.rel("sub", t1, c1)            # 7
add(887, g.factors, ans, 7,
    f"Consider the numbers {L(c2a)}, {L(g2)}, {L(c2b)}, {L(t1)}, {L(c1)}, "
    f"{L(ans)}. {L(c2a)} is 2. {L(c2a)} times {L(c2a)} equals {L(g2)}. "
    f"{L(c2b)} is 2. {L(c2b)} times {L(g2)} equals {L(t1)}. {L(c1)} is "
    f"1. {L(t1)} exceeds {L(c1)} by {L(ans)}. What is {L(ans)}?",
    "direct computation of the source's own nested function composition "
    "(f(g(2))); two separate given vars for the numeral '2' (the input "
    "x=2 squared for g, vs f's own leading coefficient), distinct source "
    "roles despite equal value, matching precedent.")

# 888: SKIP -- ball bounces to 2/3 height each time from 243cm, first
# bounce rising less than 30cm.
skip(888,
     "heights: 162, 108, 72, 48, 32, 21.33 -- first below 30 is bounce "
     "6, but the 6th height itself is NON-INTEGER (243*(2/3)^6=1024/48 "
     "is not an integer multiple staying in-domain past bounce 5). "
     "COMPOUND blocker, exact match to tranche6's [788]: (a) threshold/"
     "inequality question ('first ... less than 30'), no comparison "
     "primitive; (b) intermediate heights eventually go non-integer, a "
     "DOMAIN-BOUNDARY issue; (c) the bounce COUNT is an unknown being "
     "searched for (discrete-count family). Triply blocked, no path "
     "renders it.")

# 892: points (1,7),(3,13),(5,19) lie on a line; find y at x=28 -> 88.
# Slope derived from two points (one fdiv), then projected forward.
g = G()
c1 = g.given(1)
c7 = g.given(7)
c3 = g.given(3)
c13 = g.given(13)
dx = g.rel("sub", c3, c1)             # 2
dy = g.rel("sub", c13, c7)            # 6
slope = g.fdiv(dy, 2)                 # 3 (ONE fdiv, k=2=dx)
c28 = g.given(28)
deltaX = g.rel("sub", c28, c1)        # 27
scaledY = g.rel("mul", slope, deltaX) # 81
ans = g.rel("add", c7, scaledY)       # 88
add(892, g.factors, ans, 88,
    f"Consider the numbers {L(c1)}, {L(c7)}, {L(c3)}, {L(c13)}, {L(dx)}, "
    f"{L(dy)}, {L(slope)}, {L(c28)}, {L(deltaX)}, {L(scaledY)}, "
    f"{L(ans)}. {L(c1)} is 1. {L(c7)} is 7. {L(c3)} is 3. {L(c13)} is "
    f"13. {L(c3)} exceeds {L(c1)} by {L(dx)}. {L(c13)} exceeds {L(c7)} "
    f"by {L(dy)}. When {L(dy)} is divided by 2, the quotient is "
    f"{L(slope)}. {L(c28)} is 28. {L(c28)} exceeds {L(c1)} by "
    f"{L(deltaX)}. {L(slope)} times {L(deltaX)} equals {L(scaledY)}. "
    f"{L(c7)} plus {L(scaledY)} equals {L(ans)}. What is {L(ans)}?",
    "THEOREM-APPLICATION (named: two-point slope + linear extrapolation), "
    "degree: slope derived from the first pair of given points (one "
    "fdiv, k=2=run), then projected from the first point to x=28 via "
    "the derived slope -- the third table point (5,19) is redundant "
    "with the render's own path and is not separately consumed, "
    "matching precedent for not force-consuming every source datum when "
    "two facts already fully determine the line.")

# 893: (8-x)^2=x^2 -> 4. Theorem-application: expand and cancel the x^2
# terms, leaving 2*8*x=8^2 (real FOIL expansion, not a residue echo);
# solved via multiplicative inversion (technique C), avoiding both fdiv
# and any reuse of x as both argument and result in one factor.
g = G()
c8 = g.given(8)
sq8 = g.rel("mul", c8, c8)            # 64
c2 = g.given(2)
coef = g.rel("mul", c2, c8)           # 16 (=2*8, cross-term coefficient)
x = g.free()
g.rel("mul", coef, x, sq8)            # 16*x=64 -> x=4
add(893, g.factors, x, 4,
    f"Consider the numbers {L(c8)}, {L(sq8)}, {L(c2)}, {L(coef)}, "
    f"{L(x)}. {L(c8)} is 8. {L(c8)} times {L(c8)} equals {L(sq8)}. "
    f"{L(c2)} is 2. {L(c2)} times {L(c8)} equals {L(coef)}. {L(coef)} "
    f"times {L(x)} equals {L(sq8)}. What is {L(x)}?",
    "THEOREM-APPLICATION (named: binomial expansion (a-x)^2=a^2-2ax+x^2, "
    "Law 5 -- transforms the equation, doesn't just solve it), degree: "
    "expanding (8-x)^2=x^2 cancels the x^2 terms on both sides, leaving "
    "2*8*x=8^2, i.e. 16x=64 -- genuine arithmetic (16 and 64 both "
    "computed from the given 8), solved via multiplicative inversion "
    "(technique C), not a bare coefficient echo.")

# 894: f(x)=3x+1, f(3) -> 10. Direct.
g = G()
c3a = g.given(3)
c3b = g.given(3)
t1 = g.rel("mul", c3b, c3a)           # 9
c1 = g.given(1)
ans = g.rel("add", t1, c1)            # 10
add(894, g.factors, ans, 10,
    f"Consider the numbers {L(c3a)}, {L(c3b)}, {L(t1)}, {L(c1)}, "
    f"{L(ans)}. {L(c3a)} is 3. {L(c3b)} is 3. {L(c3b)} times {L(c3a)} "
    f"equals {L(t1)}. {L(c1)} is 1. {L(t1)} plus {L(c1)} equals "
    f"{L(ans)}. What is {L(ans)}?",
    "direct evaluation of f(3)=3*3+1; two separate given vars for the "
    "numeral '3' (the input value vs f's own coefficient), distinct "
    "source roles despite equal value, matching precedent.")

# 895: 2x+4=|-17+3| -> x=5. Magnitude of the abs-value term computed
# directly (17>3, so |-17+3|=17-3), then solved via multiplicative
# inversion.
g = G()
c17 = g.given(17)
c3 = g.given(3)
mag = g.rel("sub", c17, c3)           # 14 = |-17+3|
c4 = g.given(4)
t1 = g.rel("sub", mag, c4)            # 10 = 2x
c2 = g.given(2)
x = g.free()
g.rel("mul", c2, x, t1)               # 2x=10 -> x=5
add(895, g.factors, x, 5,
    f"Consider the numbers {L(c17)}, {L(c3)}, {L(mag)}, {L(c4)}, "
    f"{L(t1)}, {L(c2)}, {L(x)}. {L(c17)} is 17. {L(c3)} is 3. {L(c17)} "
    f"exceeds {L(c3)} by {L(mag)}. {L(c4)} is 4. {L(mag)} exceeds "
    f"{L(c4)} by {L(t1)}. {L(c2)} is 2. {L(c2)} times {L(x)} equals "
    f"{L(t1)}. What is {L(x)}?",
    "REARRANGEMENT (named: |-17+3|=17-3 since 17>3, a direct magnitude "
    "computation, Law 4), degree: 2x+4=14 rearranged to 2x=10, solved "
    "via multiplicative inversion (technique C), avoiding fdiv.")

# 896: SKIP -- y=(x^2+2x+8)/(x-4), vertical asymptote at x=4.
skip(896,
     "The denominator is the LINEAR factor (x-4), whose root (4) is "
     "already written verbatim in the source's own expression -- unlike "
     "[873]'s quadratic domain-exclusion (which needed genuine Vieta/"
     "discriminant work to pin TWO roots), a single linear factor's "
     "root requires ZERO transformation to read off; there is no "
     "algebra left for the solver to perform (Law 5's 'transforms, "
     "doesn't just solve' fails at degree zero). Rendering x=4 would be "
     "a bare Law-3 sole-operation residue, matching [702]/[782]/[914]'s "
     "structural-fact family exactly. (The fact that the numerator "
     "x^2+2x+8=32 at x=4, hence genuinely nonzero there, could be "
     "computed in-graph as a side confirmation, but it doesn't "
     "determine x -- x is already fully fixed by the denominator alone, "
     "so this wouldn't rescue the render from residue status.)")

# 897: midpoint M(3,7) of AB, A=(9,3), sum of B's coordinates -> 8.
# Bx=2*3-9=-3 (negative!), By=2*7-3=11. Magnitude-fold (technique 3,
# matches [784]/tranche6): track |Bx| via the OTHER subtraction order
# (Ax-2Mx, since Ax>2Mx here), then fold the negative back in at the
# final sum via a subtraction instead of an add.
g = G()
c3 = g.given(3)                       # Mx
c7 = g.given(7)                       # My
c9 = g.given(9)                       # Ax
c3b = g.given(3)                      # Ay, distinct role from Mx (same value)
c2 = g.given(2)
twoMx = g.rel("mul", c2, c3)          # 6
bxMag = g.rel("sub", c9, twoMx)       # |Bx| = 9-6=3 (Bx is actually -3)
twoMy = g.rel("mul", c2, c7)          # 14
by = g.rel("sub", twoMy, c3b)         # By = 14-3=11 (positive, direct)
ans = g.rel("sub", by, bxMag)         # By + Bx = By - |Bx| = 11-3=8
add(897, g.factors, ans, 8,
    f"Consider the numbers {L(c3)}, {L(c7)}, {L(c9)}, {L(c3b)}, {L(c2)}, "
    f"{L(twoMx)}, {L(bxMag)}, {L(twoMy)}, {L(by)}, {L(ans)}. {L(c3)} is "
    f"3. {L(c7)} is 7. {L(c9)} is 9. {L(c3b)} is 3. {L(c2)} is 2. "
    f"{L(c2)} times {L(c3)} equals {L(twoMx)}. {L(c9)} exceeds "
    f"{L(twoMx)} by {L(bxMag)}. {L(c2)} times {L(c7)} equals "
    f"{L(twoMy)}. {L(twoMy)} exceeds {L(c3b)} by {L(by)}. {L(by)} "
    f"exceeds {L(bxMag)} by {L(ans)}. What is {L(ans)}?",
    "MAGNITUDE-FOLD (technique 3, matches [784]/tranche6): the midpoint "
    "formula gives Bx=2*3-9=-3, negative -- rather than representing it "
    "directly (outside the nonneg domain), the graph computes |Bx| via "
    "the REVERSED subtraction order (Ax-2Mx=3, since Ax>2Mx here, unlike "
    "By's computation where 2My>Ay so By comes out positive directly). "
    "The final sum then folds the negative back in via SUBTRACTION "
    "(By-|Bx| = By+Bx since Bx<0) rather than addition -- the sign "
    "handling is the genuine work here, argued explicitly rather than "
    "assumed.",
    watch="pointer-collision: c2 (the midpoint formula's own '2', from "
          "B=2M-A) serves as an argument in 2 separate mul factors "
          "(computing 2*Mx and 2*My) -- matches [786]/[815]/[925]'s "
          "iterated-constant-reuse subtype (mild, 2 uses).")

# 899: SKIP -- 2*(3^x)=162 -> x=4.
skip(899,
     "3^x=81=3^4 -> x=4. Unknown exponent being solved for (find x with "
     "3^x=81) -- the discrete-exponent-search family flagged repeatedly "
     "(tranche4's [589]/[621]/[643]/[672]; tranche5's [715]/[739]; "
     "tranche6's [779]/[807]). No primitive solves for an unknown "
     "exponent; hardcoding the multiplication-chain length (4) would "
     "require having already solved the problem by hand.")

# 902: f(x)=3x+2, g(x)=(x-1)^2, f(g(-2)) -> 29. x=-2 is negative;
# magnitude-fold (technique 3, matches [784]/[897]) renders |x-1|
# directly as |x|+1 (since x<0<1), avoiding any negative intermediate.
g = G()
cXmag = g.given(2)                    # |x| = |-2| = 2
c1 = g.given(1)                       # the '-1' inside (x-1)^2
magMinus1 = g.rel("add", cXmag, c1)   # |x-1| = |x|+1 = 3 (x negative)
sq = g.rel("mul", magMinus1, magMinus1)  # g(-2)=9
c3f = g.given(3)                      # f's coefficient
t1 = g.rel("mul", c3f, sq)            # 27
c2f = g.given(2)                      # f's own '+2' constant, distinct role
ans = g.rel("add", t1, c2f)           # 29
add(902, g.factors, ans, 29,
    f"Consider the numbers {L(cXmag)}, {L(c1)}, {L(magMinus1)}, "
    f"{L(sq)}, {L(c3f)}, {L(t1)}, {L(c2f)}, {L(ans)}. {L(cXmag)} is 2. "
    f"{L(c1)} is 1. {L(cXmag)} plus {L(c1)} equals {L(magMinus1)}. "
    f"{L(magMinus1)} times {L(magMinus1)} equals {L(sq)}. {L(c3f)} is "
    f"3. {L(c3f)} times {L(sq)} equals {L(t1)}. {L(c2f)} is 2. {L(t1)} "
    f"plus {L(c2f)} equals {L(ans)}. What is {L(ans)}?",
    "MAGNITUDE-FOLD (technique 3, matches [784]/tranche6 and [897] this "
    "tranche): x=-2 is negative, so rather than representing x-1=-3 "
    "directly (outside the nonneg domain), the graph tracks |x-1| via "
    "|x|+1=3 (since x<0<1, the two magnitudes ADD rather than subtract) "
    "-- x is never squared directly, only its shifted magnitude, "
    "matching (-2-1)^2=(-3)^2=9. All downstream arithmetic (f's "
    "coefficient/constant) is genuine given-literal combination.")

# 905: x^2=7x-12, sum of solutions -> 7. LAW 3 tension (roots 3,4 are
# representable integers): a naive render would echo the source's own
# '7' coefficient directly (residue, same risk as [796]/[943]).
# Instead derive via product+difference (discriminant), matching
# [796]'s tranche6 technique exactly.
g = G()
bMag = g.given(7)
cConst = g.given(12)
bsq = g.rel("mul", bMag, bMag)        # 49
c4uni = g.given(4)
fourc = g.rel("mul", c4uni, cConst)   # 48
disc = g.rel("sub", bsq, fourc)       # 1
sqrtDisc = g.free()
g.rel("mul", sqrtDisc, sqrtDisc, disc)  # sqrtDisc=1
r1 = g.free()
r2 = g.rel("add", r1, sqrtDisc)
g.rel("mul", r1, r2, cConst)          # r1*r2=12 -> r1=3,r2=4
ans = g.rel("add", r1, r2)            # freshly-derived sum = 7
add(905, g.factors, ans, 7,
    f"Consider the numbers {L(bMag)}, {L(cConst)}, {L(bsq)}, {L(c4uni)}, "
    f"{L(fourc)}, {L(disc)}, {L(sqrtDisc)}, {L(r1)}, {L(r2)}, {L(ans)}. "
    f"{L(bMag)} is 7. {L(cConst)} is 12. {L(bMag)} times {L(bMag)} "
    f"equals {L(bsq)}. {L(c4uni)} is 4. {L(c4uni)} times {L(cConst)} "
    f"equals {L(fourc)}. {L(bsq)} exceeds {L(fourc)} by {L(disc)}. "
    f"{L(sqrtDisc)} times {L(sqrtDisc)} equals {L(disc)}. {L(r1)} plus "
    f"{L(sqrtDisc)} equals {L(r2)}. {L(r1)} times {L(r2)} equals "
    f"{L(cConst)}. {L(r1)} plus {L(r2)} equals {L(ans)}. What is "
    f"{L(ans)}?",
    "LAW 3 TENSION (resolved, flagged for the wheel, matches [796]'s "
    "tranche6 precedent exactly): a naive render would echo the "
    "source's own '7' coefficient as the sum directly -- pure Law-3 "
    "residue. Instead this row derives the sum from product (12) and "
    "discriminant-derived difference (1) alone, pinning both roots "
    "(3,4) WITHOUT ever inputting the sum, then queries their freshly-"
    "computed sum. THEOREM-APPLICATION (named: discriminant formula) "
    "embedded.")

# 911: (1/4) of 2^30 is 4^x, find x -> 14. Rendered ENTIRELY at the
# exponent level (2^30/4=2^28=4^x=2^(2x) -> 2x=28), never forming the
# actual magnitude of 2^30 -- a genuine exponent-law rearrangement.
g = G()
c30 = g.given(30)                     # exponent of 2^30
c2rm = g.given(2)                     # exponent removed by dividing by 4=2^2
reducedExp = g.rel("sub", c30, c2rm)  # 28 = exponent of 2^28
c2mult = g.given(2)                   # exponent multiplier from 4^x=2^(2x)
x = g.free()
g.rel("mul", c2mult, x, reducedExp)   # 2x=28 -> x=14
add(911, g.factors, x, 14,
    f"Consider the numbers {L(c30)}, {L(c2rm)}, {L(reducedExp)}, "
    f"{L(c2mult)}, {L(x)}. {L(c30)} is 30. {L(c2rm)} is 2. {L(c30)} "
    f"exceeds {L(c2rm)} by {L(reducedExp)}. {L(c2mult)} is 2. "
    f"{L(c2mult)} times {L(x)} equals {L(reducedExp)}. What is {L(x)}?",
    "THEOREM-APPLICATION (named: exponent laws a^m/a^n=a^(m-n) and "
    "(a^p)^k=a^(pk), Law 5 -- transforms the equation entirely into "
    "exponent arithmetic rather than solving with magnitudes), degree: "
    "2^30/4=2^28 (exponent 30-2=28, dividing by 4=2^2 subtracts 2 from "
    "the exponent) equated to 4^x=2^(2x) (exponent 2x); solved via "
    "multiplicative inversion. The actual magnitude of 2^30 (over a "
    "billion) is never formed -- the render stays entirely in exponent "
    "space, avoiding the cap by construction.")

# 914: SKIP -- y=-x^2+5, max value of y -> 5.
skip(914,
     "-x^2<=0 for all real x, so y<=5 with equality at x=0 -- the "
     "answer (5) is the bare constant term of the equation, a pure "
     "structural fact about the downward parabola's vertex value (no "
     "genuine numeric combination: max(y) = the given '5' itself, "
     "echoed without any arithmetic). Matches [782]'s (tranche6) and "
     "[702]'s (tranche5) Law-3 sole-operation-residue family exactly.")

# 915: x^3+x^2+x+1 at x=3 -> 40. Direct evaluation (Worked Example D
# style, faithful complexity).
g = G()
c3 = g.given(3)
sq = g.rel("mul", c3, c3)             # 9
cube = g.rel("mul", sq, c3)           # 27
s1 = g.rel("add", cube, sq)           # 36
s2 = g.rel("add", s1, c3)             # 39
c1 = g.given(1)
ans = g.rel("add", s2, c1)            # 40
add(915, g.factors, ans, 40,
    f"Consider the numbers {L(c3)}, {L(sq)}, {L(cube)}, {L(s1)}, "
    f"{L(s2)}, {L(c1)}, {L(ans)}. {L(c3)} is 3. {L(c3)} times {L(c3)} "
    f"equals {L(sq)}. {L(sq)} times {L(c3)} equals {L(cube)}. "
    f"{L(cube)} plus {L(sq)} equals {L(s1)}. {L(s1)} plus {L(c3)} "
    f"equals {L(s2)}. {L(c1)} is 1. {L(s2)} plus {L(c1)} equals "
    f"{L(ans)}. What is {L(ans)}?",
    "direct evaluation of the source's own polynomial at x=3, term by "
    "term (Worked Example D style, faithful complexity), all values "
    "well under the cap.")

# 916: Margo walks 10 min there, 20 min back, average rate 4mph for the
# whole trip -> total distance 2 miles. Multiplicative inversion
# (technique C) on d*60=rate*totalMin, with 60 (minutes/hour) as a
# universal constant (Law 1) -- avoids fdiv entirely (k=60 would not be
# single-digit).
g = G()
c10 = g.given(10)
c20 = g.given(20)
totalMin = g.rel("add", c10, c20)     # 30
c4 = g.given(4)
prod = g.rel("mul", c4, totalMin)     # 120
c60 = g.given(60)                     # universal constant: minutes/hour
d = g.free()
g.rel("mul", d, c60, prod)            # d*60=120 -> d=2
add(916, g.factors, d, 2,
    f"Consider the numbers {L(c10)}, {L(c20)}, {L(totalMin)}, {L(c4)}, "
    f"{L(prod)}, {L(c60)}, {L(d)}. {L(c10)} is 10. {L(c20)} is 20. "
    f"{L(c10)} plus {L(c20)} equals {L(totalMin)}. {L(c4)} is 4. "
    f"{L(c4)} times {L(totalMin)} equals {L(prod)}. {L(c60)} is 60. "
    f"{L(d)} times {L(c60)} equals {L(prod)}. What is {L(d)}?",
    "THEOREM-APPLICATION (named: distance = rate * time, with total "
    "time in minutes converted via the universal constant 60 "
    "minutes/hour, Law 1 -- universal constant lawful alone), degree: "
    "d*60 = rate*totalMinutes (120), solved via multiplicative "
    "inversion (technique C) rather than fdiv, since k=60 would not be "
    "single-digit.")

# 918: rectangle perimeter=42, area=108, shorter side -> 9. Vieta system
# (matches [776]/[819]/[870]'s self-resolving-ordering pattern), half
# of the perimeter via the one allowed fdiv.
g = G()
c42 = g.given(42)
half = g.fdiv(c42, 2)                 # 21 (ONE fdiv, k=2)
c108 = g.given(108)
rSmall = g.free()
rDiff = g.free()
rBig = g.rel("add", rSmall, rDiff)
g.rel("add", rBig, rSmall, half)      # rBig+rSmall=21
g.rel("mul", rBig, rSmall, c108)      # rBig*rSmall=108
add(918, g.factors, rSmall, 9,
    f"Consider the numbers {L(c42)}, {L(half)}, {L(c108)}, {L(rSmall)}, "
    f"{L(rDiff)}, {L(rBig)}. {L(c42)} is 42. When {L(c42)} is divided "
    f"by 2, the quotient is {L(half)}. {L(c108)} is 108. {L(rSmall)} "
    f"plus {L(rDiff)} equals {L(rBig)}. {L(rBig)} plus {L(rSmall)} "
    f"equals {L(half)}. {L(rBig)} times {L(rSmall)} equals {L(c108)}. "
    f"What is {L(rSmall)}?",
    "THEOREM-APPLICATION (named: perimeter=2(l+w), area=l*w -> Vieta "
    "sum/product system), degree: half-perimeter via the one allowed "
    "fdiv (k=2), then self-resolving-ordering Vieta system (matches "
    "[776]/[819]/[870]'s pattern) with rSmall structurally forced <= "
    "rBig via the nonneg gap variable rDiff; query is the smaller root "
    "directly (Law 6 derivable ordering).")

# 920: sum of squares of two positive integers=193, product=84, find
# their sum -> 19. Direct system encoding (technique 2, matches [856]):
# x,y free vars jointly satisfy BOTH constraints; avoids ever forming
# (x+y)^2=361, which would exceed the 300 cap.
g = G()
c193 = g.given(193)
c84 = g.given(84)
x = g.free()
y = g.free()
sqX = g.rel("mul", x, x)
sqY = g.rel("mul", y, y)
g.rel("add", sqX, sqY, c193)          # x^2+y^2=193
g.rel("mul", x, y, c84)               # x*y=84
ans = g.rel("add", x, y)              # x+y (query)
add(920, g.factors, ans, 19,
    f"Consider the numbers {L(c193)}, {L(c84)}, {L(x)}, {L(y)}, "
    f"{L(sqX)}, {L(sqY)}, {L(ans)}. {L(c193)} is 193. {L(c84)} is 84. "
    f"{L(x)} times {L(x)} equals {L(sqX)}. {L(y)} times {L(y)} equals "
    f"{L(sqY)}. {L(sqX)} plus {L(sqY)} equals {L(c193)}. {L(x)} times "
    f"{L(y)} equals {L(c84)}. {L(x)} plus {L(y)} equals {L(ans)}. What "
    f"is {L(ans)}?",
    "direct system encoding (technique 2, matches [856]): x,y are free "
    "vars searched jointly against BOTH of the source's own quantities "
    "(sum of squares, product) with no intermediate discriminant step -- "
    "avoids ever forming (x+y)^2=361, which would exceed the 300 cap. "
    "Query var (x+y) is symmetric under x<->y swap, so no ordering "
    "variable is needed for uniqueness (unlike [918]'s 'which root is "
    "smaller' case).")

# 921: sum of two numbers=19, difference=5, product -> 84. Difference is
# GIVEN directly by the source (not derived), so the smaller/larger
# split is immediate.
g = G()
c19 = g.given(19)
c5 = g.given(5)
rSmall = g.free()
rBig = g.rel("add", rSmall, c5)       # rBig = rSmall+5
g.rel("add", rBig, rSmall, c19)       # rBig+rSmall=19
ans = g.rel("mul", rBig, rSmall)      # 84
add(921, g.factors, ans, 84,
    f"Consider the numbers {L(c19)}, {L(c5)}, {L(rSmall)}, {L(rBig)}, "
    f"{L(ans)}. {L(c19)} is 19. {L(c5)} is 5. {L(rSmall)} plus {L(c5)} "
    f"equals {L(rBig)}. {L(rBig)} plus {L(rSmall)} equals {L(c19)}. "
    f"{L(rBig)} times {L(rSmall)} equals {L(ans)}. What is {L(ans)}?",
    "direct system: the source states the difference (5) directly, so "
    "rBig=rSmall+5 is a straight restatement (not a derived quantity), "
    "combined with the given sum (19) to pin both values; product "
    "queried as a freshly-computed derived variable.")

# 923: June rides 1 mile in 4 minutes; at the same rate, how long for
# 3.5 miles -> 14. 3.5=7/2 lexically explicated (matches [798]/[823]
# precedent), scaled by 7 then corrected with the one allowed fdiv.
g = G()
c7 = g.given(7)                       # numerator of 3.5 = 7/2
c4 = g.given(4)                       # minutes per mile
scaled = g.rel("mul", c7, c4)         # 28 (= 7 miles worth of minutes)
ans = g.fdiv(scaled, 2)               # 14 (ONE fdiv, k=2=denominator)
add(923, g.factors, ans, 14,
    f"Consider the numbers {L(c7)}, {L(c4)}, {L(scaled)}, {L(ans)}. "
    f"{L(c7)} is 7. {L(c4)} is 4. {L(c7)} times {L(c4)} equals "
    f"{L(scaled)}. When {L(scaled)} is divided by 2, the quotient is "
    f"{L(ans)}. What is {L(ans)}?",
    "REARRANGEMENT (named: 3.5 miles = 7/2 miles, a lexical "
    "explicitation matching [798]/[823] precedent), degree: scale by "
    "the numerator (7) first, then correct with the one allowed fdiv "
    "(k=2, the denominator) rather than ever forming a fractional "
    "intermediate.")

# 925: arithmetic sequence 2,7,12,a,b,27 -> a+b=39. Direct forward
# computation from the common difference; the redundant terminal check
# (b+diff=27) is included as a genuine, consistent constraint (matches
# [829]'s over-determined-but-consistent pattern) so the source's own
# terminal value isn't silently dropped.
g = G()
c2 = g.given(2)
c7 = g.given(7)
diff = g.rel("sub", c7, c2)           # 5
c12 = g.given(12)
a = g.rel("add", c12, diff)           # 17
b = g.rel("add", a, diff)             # 22
c27 = g.given(27)
g.rel("add", b, diff, c27)            # redundant consistency: 22+5=27
ans = g.rel("add", a, b)              # 39
add(925, g.factors, ans, 39,
    f"Consider the numbers {L(c2)}, {L(c7)}, {L(diff)}, {L(c12)}, "
    f"{L(a)}, {L(b)}, {L(c27)}, {L(ans)}. {L(c2)} is 2. {L(c7)} is 7. "
    f"{L(c7)} exceeds {L(c2)} by {L(diff)}. {L(c12)} is 12. {L(c12)} "
    f"plus {L(diff)} equals {L(a)}. {L(a)} plus {L(diff)} equals "
    f"{L(b)}. {L(c27)} is 27. {L(b)} plus {L(diff)} equals {L(c27)}. "
    f"{L(a)} plus {L(b)} equals {L(ans)}. What is {L(ans)}?",
    "direct computation of the source's own arithmetic-sequence common "
    "difference (7-2=5), applied forward twice from the given 12 to "
    "reach a,b; the terminal term (27) is rendered as a genuine "
    "redundant consistency check (b+diff=27) rather than silently "
    "unused, matching [829]'s (tranche6) over-determined-but-consistent "
    "pattern.",
    watch="pointer-collision: diff (the common difference) serves as "
          "an argument in 3 separate add factors (deriving a, deriving "
          "b, and the redundant terminal check) -- matches [786]/[866]'s "
          "iterated-constant-reuse subtype (mild, within the 2-3 "
          "certified range).")

# 926: f(x)=2x^3+4, find f^{-1}(58) -> 3. Search-based cube-root
# extraction (technique 1), matches Worked Example precedent.
g = G()
c58 = g.given(58)
c4 = g.given(4)
diff = g.rel("sub", c58, c4)          # 54
half = g.fdiv(diff, 2)                # 27 (ONE fdiv, k=2)
x = g.free()
sq = g.rel("mul", x, x)
g.rel("mul", sq, x, half)             # x^3=27 -> x=3
add(926, g.factors, x, 3,
    f"Consider the numbers {L(c58)}, {L(c4)}, {L(diff)}, {L(half)}, "
    f"{L(x)}, {L(sq)}. {L(c58)} is 58. {L(c4)} is 4. {L(c58)} exceeds "
    f"{L(c4)} by {L(diff)}. When {L(diff)} is divided by 2, the "
    f"quotient is {L(half)}. {L(x)} times {L(x)} equals {L(sq)}. "
    f"{L(sq)} times {L(x)} equals {L(half)}. What is {L(x)}?",
    "REARRANGEMENT (named: 2x^3+4=58 moves terms to x^3=27, Law 4 "
    "restatement of the source's own equation), degree: search-based "
    "cube-root extraction (technique 1), one fdiv (k=2).")

# 929: f(x)=2sqrt(x)+12/sqrt(x), g(x)=2x^2-2x-3, f(g(3)) -> 10. Both the
# sqrt AND the division-by-sqrt are rendered via search/multiplicative
# inversion (never fdiv, since fdiv's k must be a LITERAL constant, not
# a graph variable -- using a hardcoded k tied to the sqrt's value would
# be a witness-test violation, same failure mode as [818]).
g = G()
c3 = g.given(3)                       # x=3, input to g
sq3 = g.rel("mul", c3, c3)            # 9
c2a = g.given(2)                      # g's shared coefficient (2x^2 AND 2x)
twoSq = g.rel("mul", c2a, sq3)        # 18
twoX = g.rel("mul", c2a, c3)          # 6
t1 = g.rel("sub", twoSq, twoX)        # 12
c3b = g.given(3)                      # g's constant '-3', distinct role
gVal = g.rel("sub", t1, c3b)          # g(3)=9
root = g.free()
g.rel("mul", root, root, gVal)        # root^2=9 -> root=3 (sqrt search)
c2b = g.given(2)                      # f's leading coefficient, distinct role
twoRoot = g.rel("mul", c2b, root)     # 6
c12 = g.given(12)                     # f's numerator
invPart = g.free()
g.rel("mul", invPart, root, c12)      # invPart*3=12 -> 4 (mult. inversion)
ans = g.rel("add", twoRoot, invPart)  # 10
add(929, g.factors, ans, 10,
    f"Consider the numbers {L(c3)}, {L(sq3)}, {L(c2a)}, {L(twoSq)}, "
    f"{L(twoX)}, {L(t1)}, {L(c3b)}, {L(gVal)}, {L(root)}, {L(c2b)}, "
    f"{L(twoRoot)}, {L(c12)}, {L(invPart)}, {L(ans)}. {L(c3)} is 3. "
    f"{L(c3)} times {L(c3)} equals {L(sq3)}. {L(c2a)} is 2. {L(c2a)} "
    f"times {L(sq3)} equals {L(twoSq)}. {L(c2a)} times {L(c3)} equals "
    f"{L(twoX)}. {L(twoSq)} exceeds {L(twoX)} by {L(t1)}. {L(c3b)} is "
    f"3. {L(t1)} exceeds {L(c3b)} by {L(gVal)}. {L(root)} times "
    f"{L(root)} equals {L(gVal)}. {L(c2b)} is 2. {L(c2b)} times "
    f"{L(root)} equals {L(twoRoot)}. {L(c12)} is 12. {L(invPart)} "
    f"times {L(root)} equals {L(c12)}. {L(twoRoot)} plus {L(invPart)} "
    f"equals {L(ans)}. What is {L(ans)}?",
    "direct nested composition (f(g(3))); g(3) computed from the "
    "source's own coefficients (search-based root extraction, "
    "technique 1, for sqrt(g(3))=3), then f's '12/sqrt(x)' term solved "
    "via MULTIPLICATIVE INVERSION (technique C: invPart*root=12) rather "
    "than fdiv -- fdiv's k parameter must be a literal constant, and "
    "hardcoding k=3 (root's solved value) rather than linking it to the "
    "root variable would be a witness-test violation matching [818]'s "
    "failure mode. Multiplicative inversion keeps the division "
    "genuinely tied to the derived sqrt value.",
    watch="pointer-collision: c2a (g's own leading coefficient) serves "
          "as an argument in 2 separate mul factors (the 2x^2 and 2x "
          "terms of the SAME source function) -- matches [786]/[815]'s "
          "iterated-constant-reuse subtype (mild, 2 uses).")

# 933: a bowtie b = a + [infinite nested radical in b]. 7 bowtie g = 9,
# find g -> 2. The fixed-point identity x=sqrt(b+x) (x=the nested
# radical's value) means x is DIRECTLY derivable from the source's own
# equation (9-7=2), sidestepping any need to represent the infinite
# nesting itself.
g = G()
c7 = g.given(7)
c9 = g.given(9)
x = g.rel("sub", c9, c7)              # x = 9-7 = 2 (the radical's value)
sq = g.rel("mul", x, x)               # 4
ans = g.rel("sub", sq, x)             # g = x^2-x = 2
add(933, g.factors, ans, 2,
    f"Consider the numbers {L(c7)}, {L(c9)}, {L(x)}, {L(sq)}, {L(ans)}. "
    f"{L(c7)} is 7. {L(c9)} is 9. {L(c9)} exceeds {L(c7)} by {L(x)}. "
    f"{L(x)} times {L(x)} equals {L(sq)}. {L(sq)} exceeds {L(x)} by "
    f"{L(ans)}. What is {L(ans)}?",
    "THEOREM-APPLICATION (named: infinite nested radical fixed point -- "
    "if x=sqrt(b+x) then x^2=b+x, a standard identity), degree: the "
    "custom operator's OWN definition (a bowtie b = a + [the radical]) "
    "means the radical's value is directly 9-7=2 from the source's own "
    "stated equation, sidestepping any need to represent the infinite "
    "nesting; g is then genuinely derived via the fixed-point identity "
    "(x^2-x=b), not asserted.")

# 934: Abby+Bart=260, Bart+Cindy=245, Cindy+Damon=270, find Abby+Damon
# -> 285. Rearrangement (Law 4): (A+B)+(C+D)-(B+C) cancels B and C
# without solving for any individual weight. Operation ORDER matters
# for cap-avoidance: subtract first (260-245=15, safely under cap),
# THEN add 270 -- adding 260+270 first would form 530, over the 300 cap.
g = G()
c260 = g.given(260)
c245 = g.given(245)
c270 = g.given(270)
t1 = g.rel("sub", c260, c245)         # 15 (cap-avoidance: subtract first)
ans = g.rel("add", t1, c270)          # 285
add(934, g.factors, ans, 285,
    f"Consider the numbers {L(c260)}, {L(c245)}, {L(c270)}, {L(t1)}, "
    f"{L(ans)}. {L(c260)} is 260. {L(c245)} is 245. {L(c260)} exceeds "
    f"{L(c245)} by {L(t1)}. {L(c270)} is 270. {L(t1)} plus {L(c270)} "
    f"equals {L(ans)}. What is {L(ans)}?",
    "REARRANGEMENT (Law 4, named: (A+B)+(C+D)-(B+C)=A+D, the standard "
    "elimination identity for this three-pairwise-sum puzzle shape), "
    "degree: combines all THREE of the source's own given sums via two "
    "genuine operations to reach a quantity that equals none of them "
    "individually; never solves for any individual person's weight "
    "(which is in fact underdetermined). CAP-AVOIDANCE note: operation "
    "order is load-bearing -- subtracting first (260-245=15) then "
    "adding 270 stays under the 300 cap throughout, whereas adding "
    "260+270 first would form 530 (over cap) before the compensating "
    "subtraction.")

# 935: SKIP -- r=3^s-s, s=2^n+1, r at n=2 -> 238.
skip(935,
     "s=2^2+1=5 (n=2 is a literal substitution directly in the source, "
     "so 2^n's chain length of 2 is legitimately hardcoded). But r=3^s-s "
     "then needs 3^5, where s=5 is a DERIVED quantity (not itself a "
     "source literal) -- hardcoding a 5-step multiplication chain for "
     "3^s is disconnected from the graph's own derivation of s (if s "
     "came out different, the hardcoded chain wouldn't adapt). WITNESS "
     "TEST fails exactly as it did for tranche6's [818] ('derived "
     "non-source-literal exponent feeding a fixed-shape power chain'), "
     "confirming that sub-case as a recurring pattern worth the bench "
     "formally naming.")

# 939: 30oz of 30% acid solution, add pure water to reach 20% acid ->
# 15oz water. BOTH percentage steps rendered via lexical fraction
# explicitation + multiplicative technique, avoiding the pct primitive
# entirely (matches [798]/tranche6's accommodation) AND avoiding fdiv
# (30%'s denominator 10 is not single-digit).
g = G()
c30 = g.given(30)                     # oz of original solution
c3 = g.given(3)                       # numerator of 30% = 3/10
c10 = g.given(10)                     # denominator of 30% = 3/10
temp90 = g.rel("mul", c30, c3)        # 90
acidMass = g.free()
g.rel("mul", acidMass, c10, temp90)   # acidMass*10=90 -> 9 (mult. inversion)
c5 = g.given(5)                       # reciprocal denominator for 20%=1/5
totalOz = g.rel("mul", c5, acidMass)  # 45
ans = g.rel("sub", totalOz, c30)      # 15
add(939, g.factors, ans, 15,
    f"Consider the numbers {L(c30)}, {L(c3)}, {L(c10)}, {L(temp90)}, "
    f"{L(acidMass)}, {L(c5)}, {L(totalOz)}, {L(ans)}. {L(c30)} is 30. "
    f"{L(c3)} is 3. {L(c30)} times {L(c3)} equals {L(temp90)}. "
    f"{L(c10)} is 10. {L(acidMass)} times {L(c10)} equals {L(temp90)}. "
    f"{L(c5)} is 5. {L(c5)} times {L(acidMass)} equals {L(totalOz)}. "
    f"{L(totalOz)} exceeds {L(c30)} by {L(ans)}. What is {L(ans)}?",
    "REARRANGEMENT (named: 30%=3/10 and 20%=1/5, both lexically "
    "explicated fractions, matching [798]/[823] precedent), degree: "
    "acid mass (constant across dilution) derived via multiplicative "
    "inversion (30*3=90=acidMass*10), then total solution volume via "
    "direct multiplication (acidMass*5, since 1/20%=5), water added is "
    "the difference from the original volume.",
    accommodation="pct argument pointers (a known gate weakness) "
                   "avoided entirely -- BOTH percentages (30% and 20%) "
                   "are rendered via lexically-explicated fractions "
                   "(3/10, 1/5) combined with multiplicative inversion, "
                   "never touching the pct primitive, and also avoiding "
                   "fdiv (k=10 would not be single-digit).")

# 940: x-y=6, x+y=12, find y -> 3. Direct system.
g = G()
c6 = g.given(6)
c12 = g.given(12)
x = g.free()
y = g.free()
g.rel("sub", x, y, c6)                # x-y=6
g.rel("add", x, y, c12)               # x+y=12
add(940, g.factors, y, 3,
    f"Consider the numbers {L(c6)}, {L(c12)}, {L(x)}, {L(y)}. {L(c6)} "
    f"is 6. {L(c12)} is 12. {L(x)} exceeds {L(y)} by {L(c6)}. {L(x)} "
    f"plus {L(y)} equals {L(c12)}. What is {L(y)}?",
    "direct system encoding (technique 2): x,y searched jointly against "
    "BOTH of the source's own equations, matching Worked Example E.")

# 941: SKIP -- 12 ordered integer pairs (x,y) satisfy x^2+y^2=25,
# greatest possible x+y.
skip(941,
     "The 12 pairs come from (0,+-5),(+-5,0),(+-3,+-4),(+-4,+-3); the "
     "greatest sum (7) comes from (3,4)/(4,3). A naive nonneg render "
     "(x,y free, x^2+y^2=25) has TWO valid nonneg decompositions "
     "((0,5) sum=5, (3,4) sum=7) -- solve2's uniqueness check on the "
     "query var (x+y) correctly flags this as ambiguous. Forcing the "
     "MAXIMUM requires enumerating all nonneg decompositions of 25 as a "
     "sum of two squares and selecting the larger sum -- no primitive "
     "performs this kind of Diophantine-decomposition search/argmax "
     "(sel only selects between two ALREADY-COMPUTED candidates by "
     "direct comparison, not an open-ended search over solution "
     "branches), and hand-supplying the two candidate pairs (0,5) and "
     "(3,4) directly would be GIFTING the very decomposition the "
     "problem asks for. New family this tranche: maximization over an "
     "unenumerated integer-decomposition set. Operation-shaped skip.")

# 943: M*(M-6)=-5, sum of all possible M -> 6. LAW 3 tension (roots 1,5
# representable integers): naive sum-of-roots would echo the '6' in
# 'six less than M' directly (residue, matches [796]/[905]). Resolved
# via product+difference, same technique.
g = G()
bMag = g.given(6)
cConst = g.given(5)
bsq = g.rel("mul", bMag, bMag)        # 36
c4uni = g.given(4)
fourc = g.rel("mul", c4uni, cConst)   # 20
disc = g.rel("sub", bsq, fourc)       # 16
sqrtDisc = g.free()
g.rel("mul", sqrtDisc, sqrtDisc, disc)  # sqrtDisc=4
r1 = g.free()
r2 = g.rel("add", r1, sqrtDisc)
g.rel("mul", r1, r2, cConst)          # r1*r2=5 -> r1=1,r2=5
ans = g.rel("add", r1, r2)            # 6
add(943, g.factors, ans, 6,
    f"Consider the numbers {L(bMag)}, {L(cConst)}, {L(bsq)}, "
    f"{L(c4uni)}, {L(fourc)}, {L(disc)}, {L(sqrtDisc)}, {L(r1)}, "
    f"{L(r2)}, {L(ans)}. {L(bMag)} is 6. {L(cConst)} is 5. {L(bMag)} "
    f"times {L(bMag)} equals {L(bsq)}. {L(c4uni)} is 4. {L(c4uni)} "
    f"times {L(cConst)} equals {L(fourc)}. {L(bsq)} exceeds {L(fourc)} "
    f"by {L(disc)}. {L(sqrtDisc)} times {L(sqrtDisc)} equals {L(disc)}. "
    f"{L(r1)} plus {L(sqrtDisc)} equals {L(r2)}. {L(r1)} times {L(r2)} "
    f"equals {L(cConst)}. {L(r1)} plus {L(r2)} equals {L(ans)}. What is "
    f"{L(ans)}?",
    "LAW 3 TENSION (resolved, flagged for the wheel, matches [796]/"
    "[905] this book): a naive render would echo the source's own '6' "
    "(from 'six less than M') as the sum directly -- pure Law-3 "
    "residue. Instead this row derives the sum from product (5, the "
    "magnitude of M(M-6)=-5) and discriminant-derived difference (4) "
    "alone, pinning both roots (1,5) WITHOUT ever inputting the sum, "
    "then queries their freshly-computed sum.")

# 944: 10 volumes, paperback $15 or hardcover $25, total $220, how many
# hardcover -> 7. Direct system.
g = G()
c10 = g.given(10)
c25 = g.given(25)
c15 = g.given(15)
c220 = g.given(220)
h = g.free()
p = g.rel("sub", c10, h)              # p = 10-h
t1 = g.rel("mul", c25, h)
t2 = g.rel("mul", c15, p)
g.rel("add", t1, t2, c220)            # 25h+15p=220
add(944, g.factors, h, 7,
    f"Consider the numbers {L(c10)}, {L(c25)}, {L(c15)}, {L(c220)}, "
    f"{L(h)}, {L(p)}, {L(t1)}, {L(t2)}. {L(c10)} is 10. {L(c10)} "
    f"exceeds {L(h)} by {L(p)}. {L(c25)} is 25. {L(c25)} times {L(h)} "
    f"equals {L(t1)}. {L(c15)} is 15. {L(c15)} times {L(p)} equals "
    f"{L(t2)}. {L(c220)} is 220. {L(t1)} plus {L(t2)} equals {L(c220)}. "
    f"What is {L(h)}?",
    "direct system encoding (technique 2): h (hardcover count) is free, "
    "p (paperback count) derived as 10-h (h+p=10, the source's own "
    "total-volumes fact), then the total-cost equation (25h+15p=220) "
    "pins h uniquely.")

# 945: find a such that ax^2+12x+9 is the square of a binomial -> 4.
# Theorem-application (perfect-square trinomial), search-based
# extraction for the binomial's constant term, multiplicative inversion
# for its coefficient.
g = G()
c9 = g.given(9)
B = g.free()
g.rel("mul", B, B, c9)                # B^2=9 -> B=3
c2 = g.given(2)
twoB = g.rel("mul", c2, B)            # 6
c12 = g.given(12)
p = g.free()
g.rel("mul", p, twoB, c12)            # p*6=12 -> p=2
a = g.rel("mul", p, p)                # 4
add(945, g.factors, a, 4,
    f"Consider the numbers {L(c9)}, {L(B)}, {L(c2)}, {L(twoB)}, "
    f"{L(c12)}, {L(p)}, {L(a)}. {L(c9)} is 9. {L(B)} times {L(B)} "
    f"equals {L(c9)}. {L(c2)} is 2. {L(c2)} times {L(B)} equals "
    f"{L(twoB)}. {L(c12)} is 12. {L(p)} times {L(twoB)} equals "
    f"{L(c12)}. {L(p)} times {L(p)} equals {L(a)}. What is {L(a)}?",
    "THEOREM-APPLICATION (named: perfect-square trinomial (px+B)^2 = "
    "p^2x^2 + 2pBx + B^2, Law 5 -- transforms the source's own "
    "expression into the binomial-square identity), degree: B found by "
    "search-based root extraction (technique 1, B^2=9), the cross-term "
    "coefficient (2B) computed genuinely, then p found via "
    "multiplicative inversion (technique C, p*2B=12), a=p^2 assembled "
    "last -- no fdiv, everything search/inversion-based.")

# ===========================================================================
print(f"\nDrafted: {len(rows)}  Skipped: {len(skips)}  Fails: {len(fails)}")
print(f"Total accounted: {len(rows) + len(skips) + len(fails)} / 40")
if fails:
    print("FAILS:", fails)

with open('/home/bryce/mycelium/.cache/book8_t7_prose_pairs_draft.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print("Wrote", len(rows), "rows to .cache/book8_t7_prose_pairs_draft.jsonl")

# Sanity: all 40 src_idx accounted for, no gaps/dupes
drafted_ids = {r["gen"]["src_idx"] for r in rows}
skipped_ids = {s[0] for s in skips}
all_ids = {c["src_idx"] for c in CANDS.values()}
missing = all_ids - drafted_ids - skipped_ids
overlap = drafted_ids & skipped_ids
print("Missing (neither drafted nor skipped):", sorted(missing))
print("Overlap (both drafted and skipped):", sorted(overlap))
