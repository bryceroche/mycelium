import json, sys, string
sys.path.insert(0, '/home/bryce/mycelium')
sys.path.insert(0, '/home/bryce/mycelium/scripts')
from tta_alg2_dials import solve2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic

SMP = {"n_vars": 24, "m": 300}
LETTERS = string.ascii_lowercase

CANDS = {c["src_idx"]: c for c in
         json.load(open('/home/bryce/mycelium/.cache/book8_candidates_t6.json'))["tranche6"]}


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
        "src_idx": src_idx, "book": 8, "tranche": 6, "floor": "prime", "fs": True,
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
# 776: (x-3)(x+3)=21x-63 -> x^2-21x+54=0, roots p>q, find p-q -> 15.
# Vieta sum/product system (matches [714]/[819] self-resolving-ordering
# pattern), query IS the difference itself (rDiff), avoiding ever forming
# 21^2=441 (over cap) the way a naive discriminant approach would.
g = G()
c21 = g.given(21)
c54 = g.given(54)
rSmall = g.free()
rDiff = g.free()
rBig = g.rel("add", rSmall, rDiff)    # rSmall+rDiff=rBig
g.rel("add", rBig, rSmall, c21)       # rBig+rSmall=21
g.rel("mul", rBig, rSmall, c54)       # rBig*rSmall=54
add(776, g.factors, rDiff, 15,
    f"Consider the numbers {L(c21)}, {L(c54)}, {L(rSmall)}, {L(rDiff)}, "
    f"{L(rBig)}. {L(c21)} is 21. {L(c54)} is 54. {L(rSmall)} plus "
    f"{L(rDiff)} equals {L(rBig)}. {L(rBig)} plus {L(rSmall)} equals "
    f"{L(c21)}. {L(rBig)} times {L(rSmall)} equals {L(c54)}. What is "
    f"{L(rDiff)}?",
    "THEOREM-APPLICATION (named: Vieta sum/product for (x-3)(x+3)=21x-63 "
    "-> x^2-21x+54=0), degree: rendered as a direct system (technique 2) "
    "on rSmall/rDiff/rBig (self-resolving ordering, matches [714]'s "
    "pattern) rather than a discriminant approach, which would need "
    "21^2=441 (over the 300 cap). COUNTERFACTUAL note: the query is "
    "p-q itself (rDiff), a genuinely derived quantity requiring the full "
    "2-var system; had the source instead asked for p+q, that would "
    "reduce to the bare given literal 21 (Law-3 residue) -- the ACTUAL "
    "question asked is the one that requires genuine work.")

# 777: cube_root(2-x/2)=-3 -> x=58. -3 cubed = -27 (magnitude 27, computed
# via chained mult on the given magnitude 3); 2-x/2=-27 rearranges to
# x/2=2+27=29, x=58. The two "2"s (outer literal vs x's denominator) are
# distinct source roles despite the same numeral (matches [736] precedent).
g = G()
c3 = g.given(3)                       # magnitude of the cube root's value (-3)
sq = g.rel("mul", c3, c3)             # 9
cube = g.rel("mul", sq, c3)           # 27 = |(-3)^3|
c2a = g.given(2)                      # the outer literal "2" in "2 - x/2"
sum29 = g.rel("add", c2a, cube)       # 2+27=29 = x/2
c2b = g.given(2)                      # the denominator "2" in "x/2" (distinct
                                       # role from c2a despite equal value)
x = g.rel("mul", c2b, sum29)          # x/2=29 -> x=58
add(777, g.factors, x, 58,
    f"Consider the numbers {L(c3)}, {L(sq)}, {L(cube)}, {L(c2a)}, "
    f"{L(sum29)}, {L(c2b)}, {L(x)}. {L(c3)} is 3. {L(c3)} times {L(c3)} "
    f"equals {L(sq)}. {L(sq)} times {L(c3)} equals {L(cube)}. {L(c2a)} "
    f"is 2. {L(c2a)} plus {L(cube)} equals {L(sum29)}. {L(c2b)} is 2. "
    f"{L(c2b)} times {L(sum29)} equals {L(x)}. What is {L(x)}?",
    "REARRANGEMENT (named: cube_root(2-x/2)=-3 cubes to 2-x/2=-27, moving "
    "terms to x/2=2+27=29, x=58 -- Law 4 restatement of the source's own "
    "equation), degree: |-3|^3=27 built by chained multiplication "
    "(search-based root/power extraction, technique 1, inverted -- here "
    "the base is known and we cube it, not the reverse). Two separate "
    "given vars for the numeral '2' (outer subtrahend vs x's denominator) "
    "since they are distinct source roles, matching [736]'s precedent.")

# 778: log_16(r+16)=5/4 -> r=16. 16^(5/4) = 16 * 16^(1/4) = 16*2=32
# (16^(1/4) found by search: root^4=16 -> root=2); r=32-16=16. The base
# "16" is reused for both the log's base and the 16^(5/4) factor, since
# it is literally the same source quantity in both roles.
g = G()
c16 = g.given(16)
root4 = g.free()
sq = g.rel("mul", root4, root4)
g.rel("mul", sq, sq, c16)             # root4^4=16 -> root4=2
pow5 = g.rel("mul", c16, root4)       # 16^(5/4) = 16*root4 = 32
r = g.rel("sub", pow5, c16)           # r = 32-16=16
add(778, g.factors, r, 16,
    f"Consider the numbers {L(c16)}, {L(root4)}, {L(sq)}, {L(pow5)}, "
    f"{L(r)}. {L(c16)} is 16. {L(root4)} times {L(root4)} equals "
    f"{L(sq)}. {L(sq)} times {L(sq)} equals {L(c16)}. {L(c16)} times "
    f"{L(root4)} equals {L(pow5)}. {L(pow5)} exceeds {L(c16)} by "
    f"{L(r)}. What is {L(r)}?",
    "THEOREM-APPLICATION (named: log_16(r+16)=5/4 means r+16=16^(5/4)), "
    "degree: 16^(5/4) rearranged as 16^1 * 16^(1/4) (Law 4), with "
    "16^(1/4) found by search-based root extraction (technique 1, "
    "root4^4=16); r found by subtracting the source's own base literal "
    "(16) a second time. All values stay well under the cap.")

# 779: SKIP -- geometric series sum 63/128 for a=1/4,r=1/2, find n terms.
skip(779,
     "S_n=(1/2)(1-(1/2)^n)=63/128 -> (1/2)^n=1/64 -> n=6 -- n is an "
     "UNKNOWN EXPONENT being solved for (2^n=64), the same discrete-log-"
     "adjacent family flagged repeatedly (tranche4's [589]/[621]/[643]/"
     "[672]; tranche5's [715]/[739]). A tempting alternative is to render "
     "6 explicit halving steps summing to 63/128*128=63 (scaled by 128) "
     "and let n be the CHAIN LENGTH we chose -- but that hardcodes the "
     "very quantity being asked for (we would only know to write 6 steps "
     "because we already solved the problem by hand), which is gifting "
     "the unknown, not deriving it. No primitive solves for an unknown "
     "exponent/count.")

# 782: SKIP -- x-intercepts of x=-2y^2+y+1 (a sideways parabola, function
# of y). No arithmetic: any curve of the form x=f(y) crosses the x-axis
# (y=0) at exactly ONE point by definition (single-valued in y), so the
# answer "1" is a pure structural fact about the FORM of the equation,
# not a computed quantity -- matches [702]'s Law-3 sole-operation-residue
# family (no primitive computes "is this a function of y" or "how many
# times does a function cross an axis").
skip(782,
     "x=-2y^2+y+1 is a curve written as x=f(y); its x-intercepts are the "
     "points where y=0, and since it's single-valued in y there is "
     "EXACTLY ONE such point regardless of the coefficients -- the "
     "answer (1) is a structural fact about the equation's FORM (x "
     "expressed as a function of y), not a computed numeric quantity. "
     "Same family as [702] (tranche5): Law-3 sole-operation residue, no "
     "primitive represents 'count how many times a curve crosses an "
     "axis' or 'is this expression single-valued'.")

# 783: piecewise f: ax+3 (x>0), ab (x=0), bx+c (x<0). f(2)=5, f(0)=5,
# f(-2)=-10, a,b,c nonneg integers -> a+b+c=6. Direct system (technique 2)
# on the source's own three stated equations; the two "5"s (f(2) and f(0))
# are distinct source facts despite equal value, matching precedent.
g = G()
cx2 = g.given(2)                      # x=2 (argument for the x>0 branch)
c3 = g.given(3)                       # the "+3" in ax+3
c5a = g.given(5)                      # f(2)=5
twoA = g.rel("sub", c5a, c3)          # 5-3=2 = a*2
a = g.free()
g.rel("mul", a, cx2, twoA)            # a*2=2 -> a=1
c5b = g.given(5)                      # f(0)=5 (distinct fact from f(2)=5)
b = g.free()
g.rel("mul", a, b, c5b)               # a*b=5 -> b=5
cxNeg2 = g.given(2)                   # |x|=2 (argument for the x<0 branch)
c10 = g.given(10)                     # |f(-2)|=10
twoB = g.rel("mul", cxNeg2, b)        # 2*5=10 = |b*x|
c = g.rel("sub", twoB, c10)           # c = 10-10=0 (bx+c=-10 -> c=2b-10)
ans1 = g.rel("add", a, b)             # 1+5=6
ans = g.rel("add", ans1, c)           # 6+0=6
add(783, g.factors, ans, 6,
    f"Consider the numbers {L(cx2)}, {L(c3)}, {L(c5a)}, {L(twoA)}, "
    f"{L(a)}, {L(c5b)}, {L(b)}, {L(cxNeg2)}, {L(c10)}, {L(twoB)}, {L(c)}, "
    f"{L(ans1)}, {L(ans)}. {L(cx2)} is 2. {L(c3)} is 3. {L(c5a)} is 5. "
    f"{L(c5a)} exceeds {L(c3)} by {L(twoA)}. {L(a)} times {L(cx2)} "
    f"equals {L(twoA)}. {L(c5b)} is 5. {L(a)} times {L(b)} equals "
    f"{L(c5b)}. {L(cxNeg2)} is 2. {L(c10)} is 10. {L(cxNeg2)} times "
    f"{L(b)} equals {L(twoB)}. {L(twoB)} exceeds {L(c10)} by {L(c)}. "
    f"{L(a)} plus {L(b)} equals {L(ans1)}. {L(ans1)} plus {L(c)} equals "
    f"{L(ans)}. What is {L(ans)}?",
    "direct system encoding (technique 2), THREE of the source's own "
    "piecewise equations chained: a from 2a+3=5 (multiplicative "
    "inversion), b from a*b=5, c from |b*(-2)|+c=|-10| rearranged to "
    "c=2b-10 (Law 4, moving the magnitude term across); every "
    "coefficient a genuinely derived quantity, not asserted. Two "
    "separate given vars for the numeral '5' (f(2) and f(0), distinct "
    "source facts) and for '2' (x=2 vs |x|=-2's magnitude), matching "
    "precedent for same-valued-distinct-role literals.")

# 784: x^2+15x-54=0, greater root -> 3. Magnitude-fold (technique 3) for
# the opposite-sign root pair (3 and -18): sum=-15 (magnitude 15, larger
# root minus |smaller root| = -15, i.e. smallMag-rBig=15), product
# magnitude 54 (rBig*smallMag=54, since rBig>0, smallerRoot<0 makes the
# true product negative -54, magnitude 54). Matches tranche4's opposite-
# sign root precedent extended from [736]'s single-root fold to a full
# 2-var system.
g = G()
c15 = g.given(15)                     # magnitude of the "+15x" coefficient
c54 = g.given(54)                     # magnitude of the "-54" constant
rBig = g.free()
smallMag = g.free()
g.rel("add", rBig, c15, smallMag)     # rBig+15=smallMag (smallMag-rBig=15)
g.rel("mul", rBig, smallMag, c54)     # rBig*smallMag=54
add(784, g.factors, rBig, 3,
    f"Consider the numbers {L(c15)}, {L(c54)}, {L(rBig)}, {L(smallMag)}. "
    f"{L(c15)} is 15. {L(c54)} is 54. {L(rBig)} plus {L(c15)} equals "
    f"{L(smallMag)}. {L(rBig)} times {L(smallMag)} equals {L(c54)}. "
    f"What is {L(rBig)}?",
    "MAGNITUDE-FOLD (technique 3, opposite-sign root pair): the "
    "quadratic's two roots are 3 and -18 (opposite signs); rather than "
    "represent -18 directly (outside the nonneg domain), the graph "
    "tracks rBig=3 and smallMag=|-18|=18, related by rBig+15=smallMag "
    "(since sum-of-roots=-15 means smallMag-rBig=15) and "
    "rBig*smallMag=54 (product-of-roots=-54 means the MAGNITUDES "
    "multiply to 54). Extends [736]'s single-root fold (tranche5) to a "
    "full 2-var system, matching tranche4's opposite-sign-root-handling "
    "precedent.")

# 786: f(x)=3x+3, g(x)=4x+3. f(g(f(2))) -> 120. Direct nested composition;
# f's own coefficient/constant (c3fCoef, c3fConst) reused across BOTH
# applications of f (pointer-collision watch, matches [687]'s iterated-
# function-reuse subtype).
g = G()
c2 = g.given(2)                       # x=2
c3fCoef = g.given(3)                  # f's coefficient (3x)
t1 = g.rel("mul", c3fCoef, c2)        # 3*2=6
c3fConst = g.given(3)                 # f's own "+3" (distinct role from coef)
f2 = g.rel("add", t1, c3fConst)       # f(2)=9
c4gCoef = g.given(4)                  # g's coefficient (4x)
t2 = g.rel("mul", c4gCoef, f2)        # 4*9=36
c3gConst = g.given(3)                 # g's own "+3" (distinct from f's)
g9 = g.rel("add", t2, c3gConst)       # g(9)=39
t3 = g.rel("mul", c3fCoef, g9)        # 3*39=117 (f's coefficient, reused)
f39 = g.rel("add", t3, c3fConst)      # f(39)=120 (f's constant, reused)
add(786, g.factors, f39, 120,
    f"Consider the numbers {L(c2)}, {L(c3fCoef)}, {L(t1)}, {L(c3fConst)}, "
    f"{L(f2)}, {L(c4gCoef)}, {L(t2)}, {L(c3gConst)}, {L(g9)}, {L(t3)}, "
    f"{L(f39)}. {L(c2)} is 2. {L(c3fCoef)} is 3. {L(c3fCoef)} times "
    f"{L(c2)} equals {L(t1)}. {L(c3fConst)} is 3. {L(t1)} plus "
    f"{L(c3fConst)} equals {L(f2)}. {L(c4gCoef)} is 4. {L(c4gCoef)} "
    f"times {L(f2)} equals {L(t2)}. {L(c3gConst)} is 3. {L(t2)} plus "
    f"{L(c3gConst)} equals {L(g9)}. {L(c3fCoef)} times {L(g9)} equals "
    f"{L(t3)}. {L(t3)} plus {L(c3fConst)} equals {L(f39)}. What is "
    f"{L(f39)}?",
    "direct computation of the source's own nested function composition "
    "(f(g(f(2)))), every coefficient a source literal from f and g's "
    "definitions; f's coefficient and constant are each used TWICE "
    "(once per application of f), all values well under the cap "
    "(max intermediate 120).",
    watch="pointer-collision: c3fCoef (f's own coefficient) and c3fConst "
          "(f's own constant) each serve as an argument in 2 separate "
          "factors, one per application of f -- matches [687]/[694]'s "
          "iterated-function-reuse subtype.")

# 788: SKIP -- ball dropped 10ft, bounces back half distance each time,
# first bounce reaching max height <1ft.
skip(788,
     "heights after each bounce: 5, 2.5, 1.25, 0.625 -- first below 1ft "
     "is bounce 4. COMPOUND blocker: (a) threshold/inequality question "
     "('first ... less than 1'), no comparison primitive (same family "
     "as [712]/[814]/[838]); (b) the intermediate heights themselves "
     "(2.5, 1.25, 0.625) are non-integer, outside our integer-only "
     "[0,300] domain -- a DOMAIN-BOUNDARY issue layered on top of the "
     "operation-shaped inequality gap; (c) the bounce COUNT is an "
     "unknown being searched for (discrete-count family, matches "
     "[779]/[807]/[818] above). Triply blocked, no path renders it.")

# 789: sum of two numbers=30; double larger minus 3*smaller=5 -> positive
# difference=8. Direct system encoding (technique 2) on both of the
# source's own stated equations.
g = G()
c30 = g.given(30)
c2 = g.given(2)                       # doubling multiplier
c3 = g.given(3)                       # tripling multiplier
c5 = g.given(5)
S = g.free()                          # smaller
Lg = g.free()                         # larger
g.rel("add", Lg, S, c30)              # L+S=30
twoL = g.rel("mul", c2, Lg)           # 2L
threeS = g.rel("mul", c3, S)          # 3S
g.rel("sub", twoL, threeS, c5)        # 2L-3S=5
diff = g.rel("sub", Lg, S)            # L-S
add(789, g.factors, diff, 8,
    f"Consider the numbers {L(c30)}, {L(c2)}, {L(c3)}, {L(c5)}, {L(S)}, "
    f"{L(Lg)}, {L(twoL)}, {L(threeS)}, {L(diff)}. {L(c30)} is 30. "
    f"{L(c2)} is 2. {L(c3)} is 3. {L(c5)} is 5. {L(Lg)} plus {L(S)} "
    f"equals {L(c30)}. {L(c2)} times {L(Lg)} equals {L(twoL)}. {L(c3)} "
    f"times {L(S)} equals {L(threeS)}. {L(twoL)} exceeds {L(threeS)} by "
    f"{L(c5)}. {L(Lg)} exceeds {L(S)} by {L(diff)}. What is {L(diff)}?",
    "direct system encoding (technique 2): the CSP searches S (smaller) "
    "and L (larger) jointly against BOTH of the source's own equations "
    "(L+S=30, 2L-3S=5); the positive difference is queried as its own "
    "derived variable, never asserted.")

# 794: x+y=4, x^2+y^2=8, find x^3+y^3 -> 16. Sum-of-cubes identity
# (Law 5): x^3+y^3=(x+y)((x^2+y^2)-xy), with xy derived from
# (x+y)^2-(x^2+y^2)=2xy, all in-graph, x/y never individually needed.
g = G()
c4 = g.given(4)                       # x+y
c8 = g.given(8)                       # x^2+y^2
sumSq = g.rel("mul", c4, c4)          # (x+y)^2=16
twoXY = g.rel("sub", sumSq, c8)       # 16-8=8 = 2xy
xy = g.fdiv(twoXY, 2)                 # xy=4  (ONE fdiv, k=2)
diffSq = g.rel("sub", c8, xy)         # x^2+y^2-xy = 8-4=4
ans = g.rel("mul", c4, diffSq)        # (x+y)*(x^2+y^2-xy)=4*4=16
add(794, g.factors, ans, 16,
    f"Consider the numbers {L(c4)}, {L(c8)}, {L(sumSq)}, {L(twoXY)}, "
    f"{L(xy)}, {L(diffSq)}, {L(ans)}. {L(c4)} is 4. {L(c8)} is 8. "
    f"{L(c4)} times {L(c4)} equals {L(sumSq)}. {L(sumSq)} exceeds "
    f"{L(c8)} by {L(twoXY)}. When {L(twoXY)} is divided by 2, the "
    f"quotient is {L(xy)}. {L(c8)} exceeds {L(xy)} by {L(diffSq)}. "
    f"{L(c4)} times {L(diffSq)} equals {L(ans)}. What is {L(ans)}?",
    "THEOREM-APPLICATION (named: sum-of-cubes identity, "
    "x^3+y^3=(x+y)(x^2+y^2-xy)), degree: xy derived from "
    "(x+y)^2-(x^2+y^2)=2xy entirely from the source's two stated "
    "quantities (x+y=4, x^2+y^2=8), x and y never individually needed. "
    "One fdiv (k=2).")

# 795: distance between (0,4) and (3,0) -> 5. Direct distance formula,
# matches worked-example precedent exactly.
g = G()
c3 = g.given(3)
c4 = g.given(4)
sqx = g.rel("mul", c3, c3)            # 9
sqy = g.rel("mul", c4, c4)            # 16
sumSq = g.rel("add", sqx, sqy)        # 25
d = g.free()
g.rel("mul", d, d, sumSq)             # d*d=25 -> d=5
add(795, g.factors, d, 5,
    f"Consider the numbers {L(c3)}, {L(c4)}, {L(sqx)}, {L(sqy)}, "
    f"{L(sumSq)}, {L(d)}. {L(c3)} is 3. {L(c4)} is 4. {L(c3)} times "
    f"{L(c3)} equals {L(sqx)}. {L(c4)} times {L(c4)} equals {L(sqy)}. "
    f"{L(sqx)} plus {L(sqy)} equals {L(sumSq)}. {L(d)} times {L(d)} "
    f"equals {L(sumSq)}. What is {L(d)}?",
    "THEOREM-APPLICATION (named: distance formula), direct rendering of "
    "the coordinate differences (3,4) and search-based root extraction "
    "for the distance.")

# 796: (y+6)/(y^2-5y+4) undefined -> sum of such y = 5. A naive Vieta
# sum-of-roots render would just echo the source's own "-5y" coefficient
# (5) verbatim -- Law-3 residue, matching [707]'s skip precedent EXCEPT
# here the roots (1,4) are individually representable integers, so a
# genuinely non-circular derivation exists: use the coefficient's SQUARE
# and the constant term (discriminant) to derive the DIFFERENCE between
# roots, then combine with the PRODUCT (not the sum) to pin both roots,
# and query their freshly-derived sum. Avoids ever inputting "sum=5" as
# a given.
g = G()
bMag = g.given(5)                     # magnitude of the "-5y" coefficient
cConst = g.given(4)                   # constant term (also = product of roots)
bsq = g.rel("mul", bMag, bMag)        # 25
c4uni = g.given(4)                    # universal "4" in the discriminant
                                       # formula b^2-4ac (distinct role from
                                       # cConst despite equal value)
fourc = g.rel("mul", c4uni, cConst)   # 16
disc = g.rel("sub", bsq, fourc)       # 25-16=9
sqrtDisc = g.free()
g.rel("mul", sqrtDisc, sqrtDisc, disc)  # sqrtDisc^2=9 -> 3 (=root2-root1)
r1 = g.free()
r2 = g.free()
g.rel("add", r1, sqrtDisc, r2)        # r1+3=r2
g.rel("mul", r1, r2, cConst)          # r1*r2=4 (product of roots = c/a)
ans = g.rel("add", r1, r2)            # freshly-derived sum = 1+4=5
add(796, g.factors, ans, 5,
    f"Consider the numbers {L(bMag)}, {L(cConst)}, {L(bsq)}, "
    f"{L(c4uni)}, {L(fourc)}, {L(disc)}, {L(sqrtDisc)}, {L(r1)}, "
    f"{L(r2)}, {L(ans)}. {L(bMag)} is 5. {L(cConst)} is 4. {L(bMag)} "
    f"times {L(bMag)} equals {L(bsq)}. {L(c4uni)} is 4. {L(c4uni)} "
    f"times {L(cConst)} equals {L(fourc)}. {L(bsq)} exceeds {L(fourc)} "
    f"by {L(disc)}. {L(sqrtDisc)} times {L(sqrtDisc)} equals {L(disc)}. "
    f"{L(r1)} plus {L(sqrtDisc)} equals {L(r2)}. {L(r1)} times {L(r2)} "
    f"equals {L(cConst)}. {L(r1)} plus {L(r2)} equals {L(ans)}. What is "
    f"{L(ans)}?",
    "LAW 3 TENSION (resolved, flagged for the wheel): a naive render "
    "would input the source's own coefficient (5) as the sum directly "
    "and echo it back -- pure Law-3 residue, exactly [707]'s (tranche5) "
    "skip pattern. Instead this row derives the sum from the OTHER two "
    "quantities (product=4 via cConst, and the discriminant-derived "
    "root-difference=3 via b^2-4ac=9) -- r1,r2 are pinned uniquely by "
    "product AND difference alone (never given the sum directly), and "
    "the query (r1+r2) is a freshly-computed derived variable, not a "
    "re-echo. Distinguished from [707]: THERE the roots were irrational "
    "(no representable system existed at all); HERE the roots (1,4) are "
    "representable integers, making the non-circular derivation "
    "possible. THEOREM-APPLICATION (named: discriminant formula) "
    "embedded.")

# 798: 50 increased by 120% -> 110. Avoids the pct primitive (law-9-
# limited, pointer-scattered per brief) entirely: 120%=6/5 lexically
# explicated (matches [747]'s 0.75=3/4 precedent), computed via fdiv(k=5,
# single-digit) + mul.
g = G()
c50 = g.given(50)
c6 = g.given(6)                       # numerator of 120% as 6/5 (lexical
                                       # explicitation of 1.2, a known-value
                                       # restatement, not an invented number)
c5 = g.given(5)                       # denominator
tenthUnit = g.fdiv(c50, 5)            # 50/5=10  (ONE fdiv, k=5)
scaled = g.rel("mul", tenthUnit, c6)  # 10*6=60 = 120% of 50 (the increase)
final = g.rel("add", c50, scaled)     # 50+60=110
add(798, g.factors, final, 110,
    f"Consider the numbers {L(c50)}, {L(c6)}, {L(c5)}, {L(tenthUnit)}, "
    f"{L(scaled)}, {L(final)}. {L(c50)} is 50. {L(c6)} is 6. {L(c5)} is "
    f"5. When {L(c50)} is divided by 5, the quotient is {L(tenthUnit)}. "
    f"{L(tenthUnit)} times {L(c6)} equals {L(scaled)}. {L(c50)} plus "
    f"{L(scaled)} equals {L(final)}. What is {L(final)}?",
    "REARRANGEMENT (named: 'increased by X%' = original + X/100*original, "
    "Law 4), degree: 120% lexically explicated as the exact fraction 6/5 "
    "(matches [747]'s 0.75=3/4 precedent), computed as (50/5)*6=60 "
    "(the increase amount), avoiding the pct primitive altogether.",
    accommodation="pct argument pointers (a known gate weakness) avoided -- "
                   "the percentage increase is rendered via fdiv(k=5)+mul "
                   "on a lexically-explicated fraction (6/5) instead of "
                   "the pct primitive.")

# 802: SKIP -- floor(-2.54)+ceil(25.4) -> 23. No floor/ceil primitive AND
# the literals themselves (-2.54, 25.4) are non-integer decimals, a
# DOMAIN-BOUNDARY issue distinct from [716]/[725]'s floor(sqrt(int))
# family (there the input was an integer; here the input isn't even
# representable).
skip(802,
     "floor(-2.54)+ceil(25.4)=-3+26=23 requires (a) a floor/ceiling "
     "primitive, which doesn't exist (matches [716]/[725]'s tranche5 "
     "floor/sqrt family), AND (b) the source literals themselves "
     "(-2.54, 25.4) are non-integer decimals -- not representable at "
     "all in our integer-only [0,300] domain, a DOMAIN-BOUNDARY issue "
     "(distinct category from the operation-shaped floor/sqrt family, "
     "since there the underlying integer was representable and only the "
     "irrational sqrt blocked it; here the input itself is out of "
     "domain). Also the first term is negative, a second domain-boundary "
     "concern.")

# 805: discriminant of 3x^2-7x-12 -> 193. Direct theorem-application,
# matches [737]'s precedent exactly.
g = G()
bMag = g.given(7)
aCoef = g.given(3)
cMag = g.given(12)
c4uni = g.given(4)
bsq = g.rel("mul", bMag, bMag)        # 49
ac = g.rel("mul", aCoef, cMag)        # 36
fourac = g.rel("mul", c4uni, ac)      # 144
disc = g.rel("add", bsq, fourac)      # 193
add(805, g.factors, disc, 193,
    f"Consider the numbers {L(bMag)}, {L(aCoef)}, {L(cMag)}, "
    f"{L(c4uni)}, {L(bsq)}, {L(ac)}, {L(fourac)}, {L(disc)}. {L(bMag)} "
    f"is 7. {L(aCoef)} is 3. {L(cMag)} is 12. {L(c4uni)} is 4. "
    f"{L(bMag)} times {L(bMag)} equals {L(bsq)}. {L(aCoef)} times "
    f"{L(cMag)} equals {L(ac)}. {L(c4uni)} times {L(ac)} equals "
    f"{L(fourac)}. {L(bsq)} plus {L(fourac)} equals {L(disc)}. What is "
    f"{L(disc)}?",
    "THEOREM-APPLICATION (named: quadratic discriminant, "
    "disc=b^2-4ac), degree: since a,c have opposite signs (a=3, c=-12), "
    "-4ac is positive and adds directly; the '4' in the formula is a "
    "universal constant (Law 1). Matches [737] (tranche5) precedent "
    "exactly. All values under the cap (max 193).")

# 807: SKIP -- log_2(4^2). Discrete-exponent family (finding the exponent
# x with 2^x=16), matching precedent; ALSO the harvest's stated gold (2)
# appears inconsistent with the literal source math (log_2(16)=4), a
# separate data-quality flag for the bench.
skip(807,
     "log_2(4^2)=log_2(16)=4 by direct computation -- this is the "
     "discrete-exponent-search family (finding an unknown x with "
     "2^x=16) flagged repeatedly this tranche ([779] above; tranche5's "
     "[715]/[739]; tranche4's cluster), so it would be skipped on "
     "representability grounds regardless. SEPARATE DATA-QUALITY FLAG "
     "for the bench: the harvest's stated gold for this item is 2, but "
     "the literal source math evaluates to 4 (log_2(16)=4, not 2) -- "
     "verified independently via sympy. Rendering to match the STATED "
     "gold (2) rather than the literal source text would require "
     "reinterpreting the problem (e.g. as bare log_2(4)) in a way not "
     "supported by the given LaTeX, which would be LAUNDERING (Axis 2 "
     "violation, rendering a different problem than stated). Flagging "
     "for harvest-data review rather than attempting either render.")

# 808: x^2+6x+k=0, roots ratio 2:1 -> k=8. Direct system on magnitudes
# (both roots negative since sum=-6<0, product=k>0); THEOREM-APPLICATION
# (Vieta + given ratio).
g = G()
c2 = g.given(2)                       # the ratio 2:1
c6 = g.given(6)                       # magnitude of the "+6x" coefficient
smallMag = g.free()
bigMag = g.rel("mul", c2, smallMag)   # 2:1 ratio
g.rel("add", smallMag, bigMag, c6)    # smallMag+bigMag=6
k = g.rel("mul", smallMag, bigMag)    # k = smallMag*bigMag = 8
add(808, g.factors, k, 8,
    f"Consider the numbers {L(c2)}, {L(c6)}, {L(smallMag)}, "
    f"{L(bigMag)}, {L(k)}. {L(c2)} is 2. {L(c6)} is 6. {L(c2)} times "
    f"{L(smallMag)} equals {L(bigMag)}. {L(smallMag)} plus {L(bigMag)} "
    f"equals {L(c6)}. {L(smallMag)} times {L(bigMag)} equals {L(k)}. "
    f"What is {L(k)}?",
    "THEOREM-APPLICATION (named: Vieta sum/product with a given root "
    "ratio), degree: both roots are negative (sum=-6, product=k>0), so "
    "the graph tracks their MAGNITUDES throughout (smallMag, bigMag), "
    "sign consistently negative-negative -> positive product, argued "
    "once here rather than per-item.")

# 813: alternating sum of squares of consecutive odd numbers 19^2-17^2+
# 15^2-13^2+11^2-9^2+7^2-5^2+3^2-1^2 -> 200. NEW TECHNIQUE this tranche:
# N-pair telescoping difference-of-squares -- since EVERY consecutive
# pair has the same gap (2, a Law-10 structural literal, verified by
# inspection), the whole alternating sum collapses to
# 2*(19+17+15+13+11+9+7+5+3+1)=2*100=200, avoiding ever computing a
# single square (19^2=361 alone is over the 300 cap).
g = G()
c19 = g.given(19)
c17 = g.given(17)
c15 = g.given(15)
c13 = g.given(13)
c11 = g.given(11)
c9 = g.given(9)
c7 = g.given(7)
c5 = g.given(5)
c3 = g.given(3)
c1 = g.given(1)
s1 = g.rel("add", c19, c17)           # 36
s2 = g.rel("add", s1, c15)            # 51
s3 = g.rel("add", s2, c13)            # 64
s4 = g.rel("add", s3, c11)            # 75
s5 = g.rel("add", s4, c9)             # 84
s6 = g.rel("add", s5, c7)             # 91
s7 = g.rel("add", s6, c5)             # 96
s8 = g.rel("add", s7, c3)             # 99
s9 = g.rel("add", s8, c1)             # 100
c2 = g.given(2)                       # universal gap constant (Law 10
                                       # structural literal: consecutive odd
                                       # numbers always differ by 2)
total = g.rel("mul", c2, s9)          # 200
add(813, g.factors, total, 200,
    f"Consider the numbers {L(c19)}, {L(c17)}, {L(c15)}, {L(c13)}, "
    f"{L(c11)}, {L(c9)}, {L(c7)}, {L(c5)}, {L(c3)}, {L(c1)}, {L(s1)}, "
    f"{L(s2)}, {L(s3)}, {L(s4)}, {L(s5)}, {L(s6)}, {L(s7)}, {L(s8)}, "
    f"{L(s9)}, {L(c2)}, {L(total)}. {L(c19)} is 19. {L(c17)} is 17. "
    f"{L(c15)} is 15. {L(c13)} is 13. {L(c11)} is 11. {L(c9)} is 9. "
    f"{L(c7)} is 7. {L(c5)} is 5. {L(c3)} is 3. {L(c1)} is 1. {L(c19)} "
    f"plus {L(c17)} equals {L(s1)}. {L(s1)} plus {L(c15)} equals "
    f"{L(s2)}. {L(s2)} plus {L(c13)} equals {L(s3)}. {L(s3)} plus "
    f"{L(c11)} equals {L(s4)}. {L(s4)} plus {L(c9)} equals {L(s5)}. "
    f"{L(s5)} plus {L(c7)} equals {L(s6)}. {L(s6)} plus {L(c5)} equals "
    f"{L(s7)}. {L(s7)} plus {L(c3)} equals {L(s8)}. {L(s8)} plus "
    f"{L(c1)} equals {L(s9)}. {L(c2)} is 2. {L(c2)} times {L(s9)} "
    f"equals {L(total)}. What is {L(total)}?",
    "REARRANGEMENT (NEW TECHNIQUE this tranche, named: N-pair "
    "telescoping difference-of-squares -- a_k^2-b_k^2=(a_k-b_k)(a_k+b_k) "
    "for each of the 5 consecutive-odd pairs, and since EVERY pair has "
    "the identical gap 2 (Law 10 structural literal, verified by "
    "inspection of the source's own consecutive odd numbers), the "
    "alternating sum collapses to 2*(sum of ALL 10 given numbers)=200), "
    "degree: cap-avoidance (matches [744]'s tranche5 precedent extended "
    "from a single pair to 5) -- computing even one square directly "
    "(19^2=361) would exceed the 300 cap, so this rearrangement is not "
    "optional. The solver still performs real accumulation work (9 "
    "chained additions + 1 multiplication), not a shortcut echo.")

# 814: SKIP -- (x+3)^2<=1, count of integer solutions.
skip(814,
     "(x+3)^2<=1 holds for -4<=x<=-2, giving 3 integers -- counting "
     "integers satisfying an inequality has no supporting primitive "
     "(same 'no inequality-satisfying-integer-search' family as "
     "tranche5's [712]); the answer (3) is structurally coincidental to "
     "this specific inequality's bounds.")

# 815: canoes, Jan=7, each month doubles the PREVIOUS month's count (not
# cumulative), total Jan-May -> 217. Direct repeated-doubling chain
# (5 months, count derivable from the explicit Jan-May span, not
# searched); matches [694]/[851]'s multiplicative-chain pattern.
# Repeated-constant iteration chain: mild scatter risk per brief.
g = G()
c7 = g.given(7)
c2 = g.given(2)                       # doubling multiplier
feb = g.rel("mul", c2, c7)            # 14
mar = g.rel("mul", c2, feb)           # 28
apr = g.rel("mul", c2, mar)           # 56
may = g.rel("mul", c2, apr)           # 112
t1 = g.rel("add", c7, feb)            # 21
t2 = g.rel("add", t1, mar)            # 49
t3 = g.rel("add", t2, apr)            # 105
t4 = g.rel("add", t3, may)            # 217
add(815, g.factors, t4, 217,
    f"Consider the numbers {L(c7)}, {L(c2)}, {L(feb)}, {L(mar)}, "
    f"{L(apr)}, {L(may)}, {L(t1)}, {L(t2)}, {L(t3)}, {L(t4)}. {L(c7)} "
    f"is 7. {L(c2)} is 2. {L(c2)} times {L(c7)} equals {L(feb)}. "
    f"{L(c2)} times {L(feb)} equals {L(mar)}. {L(c2)} times {L(mar)} "
    f"equals {L(apr)}. {L(c2)} times {L(apr)} equals {L(may)}. {L(c7)} "
    f"plus {L(feb)} equals {L(t1)}. {L(t1)} plus {L(mar)} equals "
    f"{L(t2)}. {L(t2)} plus {L(apr)} equals {L(t3)}. {L(t3)} plus "
    f"{L(may)} equals {L(t4)}. What is {L(t4)}?",
    "direct computation of the source's own repeated-doubling rule "
    "(each month = 2x the previous month's count), chained 4 times from "
    "January's given 7, then summed across all 5 months (Jan-May is an "
    "explicit, source-stated span, not a searched/unknown count -- "
    "distinguishes this from the discrete-exponent-search family "
    "flagged elsewhere this tranche). Repeated-constant iteration chain "
    "(known mild-scatter gate-weakness pattern, drafted per brief).",
    watch="pointer-collision: c2 (the doubling multiplier) serves as an "
          "argument in 4 separate mul factors, one per month -- matches "
          "[694]/[687]'s iterated-constant-reuse subtype; ALSO matches "
          "the brief's named 'repeated-constant iteration chains' "
          "gate-weakness (mild scatter risk, 1-in-4).")

# 818: SKIP -- (10^0.5)(10^0.3)(10^0.2)(10^0.1)(10^0.9) -> 100.
skip(818,
     "10^(0.5+0.3+0.2+0.1+0.9)=10^2=100. The exponent sum (in tenths: "
     "5+3+2+1+9=20, /10=2) CAN be verified in-graph (e=free(), e*10=20 "
     "forces e=2), but the exponent (2) is NOT a source-stated fixed "
     "value the way [727]'s 3/4 or [742]'s 1/4 were -- it only emerges "
     "from summing five decimal exponents. Rendering the final "
     "10^2=10*10 requires HARDCODING the multiplication-chain LENGTH to "
     "2, which we only know because we solved the whole problem by "
     "hand first; the chain's SHAPE would need to change for a "
     "different sum, but a static graph can't make chain length depend "
     "on a derived variable. THE WITNESS TEST fails: if the exponent "
     "sum were wrong, the e=2 verification constraint could still be "
     "satisfied independently of the (separately hardcoded) 2-step "
     "final chain -- the two parts aren't actually linked. This is a "
     "NEW VARIANT of the discrete-exponent family (sum-of-exponents-"
     "then-apply, distinct from [779]/[807]'s direct unknown-exponent "
     "search) -- flagging for the bench as a possible rulebook "
     "clarification: 'derived (non-source-literal) exponents feeding a "
     "fixed-shape power chain' as its own named sub-case.")

# 819: a^2-10a+21<=0, greatest a -> 7. Boundary clause (Law 12), matches
# [714]'s precedent exactly.
g = G()
c10 = g.given(10)
c21 = g.given(21)
rSmall = g.free()
rDiff = g.free()
rBig = g.rel("add", rSmall, rDiff)
g.rel("add", rBig, rSmall, c10)       # rBig+rSmall=10
g.rel("mul", rBig, rSmall, c21)       # rBig*rSmall=21
add(819, g.factors, rBig, 7,
    f"Consider the numbers {L(c10)}, {L(c21)}, {L(rSmall)}, {L(rDiff)}, "
    f"{L(rBig)}. {L(c10)} is 10. {L(c21)} is 21. {L(rSmall)} plus "
    f"{L(rDiff)} equals {L(rBig)}. {L(rBig)} plus {L(rSmall)} equals "
    f"{L(c10)}. {L(rBig)} times {L(rSmall)} equals {L(c21)}. What is "
    f"{L(rBig)}?",
    "LAW 12 TENSION (boundary clause, flagged for the wheel, matches "
    "[714]'s tranche5 precedent exactly): source states an INEQUALITY "
    "(a^2-10a+21<=0); binding constraint is the quadratic's own two "
    "roots (3,7), and the greatest FEASIBLE a is the larger root -- "
    "(a) binding constraint named: upward parabola nonpositive exactly "
    "between its roots, standard fact; (b) no other constraint to "
    "verify; (c) counterfactual holds: 'least value of a' would query "
    "rSmall instead, genuinely different dialect. Vieta sum/product "
    "direct system, larger root self-resolved via a free nonneg gap.")

# 823: 30/50=sqrt(y/50) -> y=18. 30/50 reduced to 3/5 (lexical
# explicitation, matches [747]'s precedent); cross-multiply via
# reduction (50/25=2) rather than forming 30^2=900 (over cap).
g = G()
c3 = g.given(3)                       # reduced numerator of 30/50 (=3/5)
c5 = g.given(5)                       # reduced denominator
sq3 = g.rel("mul", c3, c3)            # 9 = 3^2
sq5 = g.rel("mul", c5, c5)            # 25 = 5^2
c50 = g.given(50)                     # y's denominator ("y/50")
twoConst = g.free()
g.rel("mul", twoConst, sq5, c50)      # twoConst*25=50 -> twoConst=2
y = g.rel("mul", twoConst, sq3)       # 2*9=18
add(823, g.factors, y, 18,
    f"Consider the numbers {L(c3)}, {L(c5)}, {L(sq3)}, {L(sq5)}, "
    f"{L(c50)}, {L(twoConst)}, {L(y)}. {L(c3)} is 3. {L(c5)} is 5. "
    f"{L(c3)} times {L(c3)} equals {L(sq3)}. {L(c5)} times {L(c5)} "
    f"equals {L(sq5)}. {L(c50)} is 50. {L(twoConst)} times {L(sq5)} "
    f"equals {L(c50)}. {L(twoConst)} times {L(sq3)} equals {L(y)}. "
    f"What is {L(y)}?",
    "REARRANGEMENT (named: 30/50 reduces exactly to 3/5, a lexical "
    "explicitation matching [747]'s 0.75=3/4 precedent), degree: "
    "squaring both sides gives y/50=9/25, and rather than cross-"
    "multiplying directly (25*y=9*50=450, over the 300 cap), the "
    "50/25=2 reduction is found by multiplicative inversion first, "
    "then y=2*9=18 -- cap-avoidance rearrangement.")

# 829: f(x)=ax+bx+2, f(1)=5, f(2)=8 -> f(3)=11. f depends only on (a+b)
# as a combined coefficient (both stated facts collapse to the SAME
# equation a+b=3); both facts are rendered as genuine, redundant
# constraints on the single combined variable (matches [697]'s
# over-determined-but-consistent precedent), so the source's second
# datum isn't silently dropped.
g = G()
c1 = g.given(1)                       # x=1
ab = g.free()                         # represents the combined (a+b)
c2 = g.given(2)                       # f's own "+2"
t1 = g.rel("mul", ab, c1)             # (a+b)*1
c5 = g.given(5)                       # f(1)=5
g.rel("add", t1, c2, c5)              # t1+2=5
c2x = g.given(2)                      # x=2 (distinct role from f's "+2")
t2 = g.rel("mul", ab, c2x)            # (a+b)*2
c8 = g.given(8)                       # f(2)=8
g.rel("add", t2, c2, c8)              # t2+2=8 (redundant, genuine 2nd check)
c3 = g.given(3)                       # x=3
t3 = g.rel("mul", ab, c3)             # (a+b)*3
ans = g.rel("add", t3, c2)            # f(3)=9+2=11
add(829, g.factors, ans, 11,
    f"Consider the numbers {L(c1)}, {L(ab)}, {L(c2)}, {L(t1)}, {L(c5)}, "
    f"{L(c2x)}, {L(t2)}, {L(c8)}, {L(c3)}, {L(t3)}, {L(ans)}. {L(c1)} "
    f"is 1. {L(ab)} times {L(c1)} equals {L(t1)}. {L(c2)} is 2. "
    f"{L(c5)} is 5. {L(t1)} plus {L(c2)} equals {L(c5)}. {L(c2x)} is 2. "
    f"{L(ab)} times {L(c2x)} equals {L(t2)}. {L(c8)} is 8. {L(t2)} "
    f"plus {L(c2)} equals {L(c8)}. {L(c3)} is 3. {L(ab)} times {L(c3)} "
    f"equals {L(t3)}. {L(t3)} plus {L(c2)} equals {L(ans)}. What is "
    f"{L(ans)}?",
    "LAW 3/9 note (light): f(x)=ax+bx+2 depends on a,b only through "
    "their SUM (a+b), which is individually underdetermined but the "
    "COMBINED quantity is exactly what f(3) needs -- rendered as a "
    "single free var 'ab' representing (a+b), matching the spirit of "
    "[772]'s querying-the-invariant-combination precedent. BOTH of the "
    "source's stated facts (f(1)=5, f(2)=8) are rendered as genuine "
    "constraints on 'ab' (they happen to be redundant/consistent, "
    "matching [697]'s over-determined-but-consistent pattern) rather "
    "than silently using only one and dropping the other.")

# 831: (x-4)/9 = 4/(x-9), positive real x -> 13. Direct system encoding
# (technique 2, cross-multiplied): the nonneg domain naturally excludes
# the extraneous root x=0 (would require x-4<0), no extra ordering
# constraint needed.
g = G()
c4 = g.given(4)
c9 = g.given(9)
x = g.free()
diffA = g.rel("sub", x, c4)           # x-4 (forces x>=4 by domain)
diffB = g.rel("sub", x, c9)           # x-9 (forces x>=9 by domain)
rhsProd = g.rel("mul", c4, c9)        # 36
g.rel("mul", diffA, diffB, rhsProd)   # (x-4)(x-9)=36
add(831, g.factors, x, 13,
    f"Consider the numbers {L(c4)}, {L(c9)}, {L(x)}, {L(diffA)}, "
    f"{L(diffB)}, {L(rhsProd)}. {L(c4)} is 4. {L(c9)} is 9. {L(x)} "
    f"exceeds {L(c4)} by {L(diffA)}. {L(x)} exceeds {L(c9)} by "
    f"{L(diffB)}. {L(c4)} times {L(c9)} equals {L(rhsProd)}. {L(diffA)} "
    f"times {L(diffB)} equals {L(rhsProd)}. What is {L(x)}?",
    "direct system encoding (technique 2): cross-multiplied "
    "(x-4)(x-9)=4*9=36, both sub-expressions forced equal via the "
    "shared result var rhsProd (matches [718]/[726] precedent). NOTE: "
    "the algebraic equation also admits x=0 (extraneous), but our "
    "nonneg-only domain naturally excludes it (x-4 would be negative), "
    "so the 'positive real x' qualifier is enforced for free by the "
    "domain rather than needing an explicit ordering primitive.")

# 832: a+4b=33, 6a+3b=51 -> a+b=12. Direct system (technique 2).
g = G()
c33 = g.given(33)
c4 = g.given(4)
c6 = g.given(6)
c3 = g.given(3)
c51 = g.given(51)
a = g.free()
b = g.free()
fourB = g.rel("mul", c4, b)
g.rel("add", a, fourB, c33)           # a+4b=33
sixA = g.rel("mul", c6, a)
threeB = g.rel("mul", c3, b)
g.rel("add", sixA, threeB, c51)       # 6a+3b=51
ans = g.rel("add", a, b)
add(832, g.factors, ans, 12,
    f"Consider the numbers {L(c33)}, {L(c4)}, {L(c6)}, {L(c3)}, "
    f"{L(c51)}, {L(a)}, {L(b)}, {L(fourB)}, {L(sixA)}, {L(threeB)}, "
    f"{L(ans)}. {L(c33)} is 33. {L(c4)} is 4. {L(c6)} is 6. {L(c3)} is "
    f"3. {L(c51)} is 51. {L(c4)} times {L(b)} equals {L(fourB)}. "
    f"{L(a)} plus {L(fourB)} equals {L(c33)}. {L(c6)} times {L(a)} "
    f"equals {L(sixA)}. {L(c3)} times {L(b)} equals {L(threeB)}. "
    f"{L(sixA)} plus {L(threeB)} equals {L(c51)}. {L(a)} plus {L(b)} "
    f"equals {L(ans)}. What is {L(ans)}?",
    "direct system encoding (technique 2): the CSP searches a,b "
    "jointly against BOTH of the source's own linear equations, "
    "never pre-solved algebraically by hand.")

# 838: SKIP -- greatest positive integer x with x^4/x^2<10.
skip(838,
     "x^4/x^2=x^2<10 for x>=1, so greatest x is 3 (3^2=9<10, 4^2=16 is "
     "not) -- counting/bounding integers by a strict inequality, same "
     "'no inequality-satisfying-integer-search primitive' family as "
     "[712] (tranche5) and [814] above. The answer is structurally tied "
     "to this specific bound and wouldn't generalize via the "
     "counterfactual test.")

# 841: James 6yr older than Louise; in 8 years James = 4x Louise's age
# 4 years ago -> sum of current ages=26. Direct system (technique 2);
# two separate "4"s (the "4 years before" offset vs the "4 times"
# multiplier) since they're distinct source roles.
g = G()
c6 = g.given(6)
Lval = g.free()
J = g.rel("add", Lval, c6)            # J=L+6
c4years = g.given(4)                  # "4 years before now"
Lminus4 = g.rel("sub", Lval, c4years) # L-4
c4times = g.given(4)                  # "4 times as old" (distinct role)
rhs = g.rel("mul", c4times, Lminus4)  # 4*(L-4)
c8 = g.given(8)
g.rel("add", J, c8, rhs)              # J+8 = 4*(L-4)
ans = g.rel("add", J, Lval)           # sum of current ages
add(841, g.factors, ans, 26,
    f"Consider the numbers {L(c6)}, {L(Lval)}, {L(J)}, {L(c4years)}, "
    f"{L(Lminus4)}, {L(c4times)}, {L(rhs)}, {L(c8)}, {L(ans)}. {L(c6)} "
    f"is 6. {L(Lval)} plus {L(c6)} equals {L(J)}. {L(c4years)} is 4. "
    f"{L(Lval)} exceeds {L(c4years)} by {L(Lminus4)}. {L(c4times)} is "
    f"4. {L(c4times)} times {L(Lminus4)} equals {L(rhs)}. {L(c8)} is "
    f"8. {L(J)} plus {L(c8)} equals {L(rhs)}. {L(J)} plus {L(Lval)} "
    f"equals {L(ans)}. What is {L(ans)}?",
    "direct system encoding (technique 2): James's current age (J) "
    "defined from Louise's (L) via the source's first fact (J=L+6), "
    "then the second fact (J+8=4*(L-4)) forces both sides equal via a "
    "shared result var (matches [718]/[726]/[831] precedent). Two "
    "separate given vars for the numeral '4' ('4 years before now' vs "
    "'4 times as old') since they are distinct source roles.")

# 845: distance between (2,3) and (7,15) -> 13. Direct distance formula.
g = G()
c5 = g.given(5)
c12 = g.given(12)
sqx = g.rel("mul", c5, c5)            # 25
sqy = g.rel("mul", c12, c12)          # 144
sumSq = g.rel("add", sqx, sqy)        # 169
d = g.free()
g.rel("mul", d, d, sumSq)             # d*d=169 -> d=13
add(845, g.factors, d, 13,
    f"Consider the numbers {L(c5)}, {L(c12)}, {L(sqx)}, {L(sqy)}, "
    f"{L(sumSq)}, {L(d)}. {L(c5)} is 5. {L(c12)} is 12. {L(c5)} times "
    f"{L(c5)} equals {L(sqx)}. {L(c12)} times {L(c12)} equals {L(sqy)}. "
    f"{L(sqx)} plus {L(sqy)} equals {L(sumSq)}. {L(d)} times {L(d)} "
    f"equals {L(sumSq)}. What is {L(d)}?",
    "THEOREM-APPLICATION (named: distance formula), direct rendering "
    "of the coordinate differences (5,12) and search-based root "
    "extraction. All values under cap (max 169).")

# 848: sqrt(2+sqrt(x))=3 -> x=49. Nested-radical rearrangement.
g = G()
c2 = g.given(2)
c3 = g.given(3)
nineVal = g.rel("mul", c3, c3)        # 9
sqrtX = g.rel("sub", nineVal, c2)     # 9-2=7
x = g.rel("mul", sqrtX, sqrtX)        # 7*7=49
add(848, g.factors, x, 49,
    f"Consider the numbers {L(c2)}, {L(c3)}, {L(nineVal)}, {L(sqrtX)}, "
    f"{L(x)}. {L(c2)} is 2. {L(c3)} is 3. {L(c3)} times {L(c3)} equals "
    f"{L(nineVal)}. {L(nineVal)} exceeds {L(c2)} by {L(sqrtX)}. "
    f"{L(sqrtX)} times {L(sqrtX)} equals {L(x)}. What is {L(x)}?",
    "REARRANGEMENT (named: sqrt(2+sqrt(x))=3 squares to 2+sqrt(x)=9, "
    "sqrt(x)=7, x=49 -- Law 4 restatement of the source's own nested "
    "equation), each step a genuine algebraic move, no primitive for "
    "'nested radical' needed beyond ordinary squaring.")

# 849: Billy=2*Joe, Billy+Joe=45 -> Billy=30. Direct system.
g = G()
c2 = g.given(2)
Joe = g.free()
Billy = g.rel("mul", c2, Joe)
c45 = g.given(45)
g.rel("add", Billy, Joe, c45)
add(849, g.factors, Billy, 30,
    f"Consider the numbers {L(c2)}, {L(Joe)}, {L(Billy)}, {L(c45)}. "
    f"{L(c2)} is 2. {L(c2)} times {L(Joe)} equals {L(Billy)}. "
    f"{L(c45)} is 45. {L(Billy)} plus {L(Joe)} equals {L(c45)}. What is "
    f"{L(Billy)}?",
    "direct system encoding (technique 2): Billy defined as 2*Joe from "
    "the source's first fact, Joe found by multiplicative-inversion-"
    "style search against the sum equation.")

# 851: bus with 48 students, half get off at each of 3 stops -> 6 remain.
# Three chained multiplicative inversions (matches [694]/[851]'s halving-
# chain precedent, avoids the one-fdiv budget entirely per row since
# each application uses inversion, though rule only caps at ONE per row
# anyway).
g = G()
c48 = g.given(48)
c2 = g.given(2)
h1 = g.free()
g.rel("mul", h1, c2, c48)             # 2*h1=48 -> h1=24
h2 = g.free()
g.rel("mul", h2, c2, h1)              # 2*h2=24 -> h2=12
h3 = g.free()
g.rel("mul", h3, c2, h2)              # 2*h3=12 -> h3=6
add(851, g.factors, h3, 6,
    f"Consider the numbers {L(c48)}, {L(c2)}, {L(h1)}, {L(h2)}, "
    f"{L(h3)}. {L(c48)} is 48. {L(c2)} is 2. {L(h1)} times {L(c2)} "
    f"equals {L(c48)}. {L(h2)} times {L(c2)} equals {L(h1)}. {L(h3)} "
    f"times {L(c2)} equals {L(h2)}. What is {L(h3)}?",
    "direct computation of the source's own repeated-halving rule, "
    "chained three times (one per stop), each halving found by "
    "multiplicative inversion (Worked Example C); matches [694]'s "
    "tranche5 precedent (3-step iteration chain).",
    watch="pointer-collision: c2 (halving divisor) serves as an "
          "argument in 3 separate mul factors, one per stop -- matches "
          "[694]/[687]'s iterated-constant-reuse subtype.")

# 855: (x-6)^2=25, sum of all solutions -> 12. Full two-root computation
# (not the Law-3-residue shortcut of doubling the vertex 6, though that
# identity DOES hold -- rendered as the genuine sum of both roots,
# leaving real solver work).
g = G()
c6 = g.given(6)
c25 = g.given(25)
r = g.free()
g.rel("mul", r, r, c25)               # r^2=25 -> r=5
sol1 = g.rel("add", c6, r)            # 6+5=11
sol2 = g.rel("sub", c6, r)            # 6-5=1
total = g.rel("add", sol1, sol2)      # 12
add(855, g.factors, total, 12,
    f"Consider the numbers {L(c6)}, {L(c25)}, {L(r)}, {L(sol1)}, "
    f"{L(sol2)}, {L(total)}. {L(c6)} is 6. {L(c25)} is 25. {L(r)} "
    f"times {L(r)} equals {L(c25)}. {L(c6)} plus {L(r)} equals "
    f"{L(sol1)}. {L(c6)} exceeds {L(r)} by {L(sol2)}. {L(sol1)} plus "
    f"{L(sol2)} equals {L(total)}. What is {L(total)}?",
    "direct computation of BOTH roots of (x-6)^2=25 (r found by "
    "search-based root extraction, roots as 6+r and 6-r) then summed -- "
    "a fuller/more-faithful render than the doubling-the-vertex "
    "shortcut (sum of roots of (x-h)^2=k is always 2h, a structural "
    "identity), deliberately choosing the render that leaves genuine "
    "solver work (finding r via search) rather than a bare Law-3-"
    "adjacent shortcut.")

# 856: xy=24, xz=48, yz=72, positive -> x+y+z=22. Direct system encoding
# (technique 2): three free vars, three product constraints, no
# pre-solving. Resembles the brief's named 'three parallel mul-
# inversions sharing an anchor' weakness pattern (each var shares two of
# the three constraints pairwise) -- flagged as a certification-risk
# watch, drafted anyway since it's the most faithful rendering (no
# simpler system exists).
g = G()
c24 = g.given(24)
c48 = g.given(48)
c72 = g.given(72)
x = g.free()
y = g.free()
z = g.free()
g.rel("mul", x, y, c24)
g.rel("mul", x, z, c48)
g.rel("mul", y, z, c72)
sum1 = g.rel("add", x, y)
total = g.rel("add", sum1, z)
add(856, g.factors, total, 22,
    f"Consider the numbers {L(c24)}, {L(c48)}, {L(c72)}, {L(x)}, "
    f"{L(y)}, {L(z)}, {L(sum1)}, {L(total)}. {L(c24)} is 24. {L(c48)} "
    f"is 48. {L(c72)} is 72. {L(x)} times {L(y)} equals {L(c24)}. "
    f"{L(x)} times {L(z)} equals {L(c48)}. {L(y)} times {L(z)} equals "
    f"{L(c72)}. {L(x)} plus {L(y)} equals {L(sum1)}. {L(sum1)} plus "
    f"{L(z)} equals {L(total)}. What is {L(total)}?",
    "direct system encoding (technique 2): the CSP searches x,y,z "
    "jointly against all THREE of the source's own product constraints "
    "-- no pre-solved intermediate needed, the solver finds the unique "
    "positive triple (4,6,12) directly.",
    watch="structurally resembles the brief's named 'three parallel "
          "mul-inversions sharing an anchor' gate weakness -- each of "
          "x, y, z is a shared argument across exactly 2 of the 3 mul "
          "factors (a genuine pairwise-sharing pattern, not a "
          "coincidence of this render); flagged for certification-risk "
          "tracking though this is the most faithful available system.")

# 858: two squares, sum of areas=65, difference=33 -> sum of perimeters
# -> 44. Direct system + theorem-application (perimeter=4*side, Law 1
# universal constant).
g = G()
c65 = g.given(65)
c33 = g.given(33)
aSq = g.free()
bSq = g.free()
g.rel("add", aSq, bSq, c65)           # aSq+bSq=65
g.rel("sub", aSq, bSq, c33)           # aSq exceeds bSq by 33
a = g.free()
g.rel("mul", a, a, aSq)               # a^2=49 -> a=7
b = g.free()
g.rel("mul", b, b, bSq)               # b^2=16 -> b=4
c4 = g.given(4)                       # perimeter formula's "4" (universal)
sumSides = g.rel("add", a, b)         # 11
perimSum = g.rel("mul", c4, sumSides) # 44
add(858, g.factors, perimSum, 44,
    f"Consider the numbers {L(c65)}, {L(c33)}, {L(aSq)}, {L(bSq)}, "
    f"{L(a)}, {L(b)}, {L(c4)}, {L(sumSides)}, {L(perimSum)}. {L(c65)} "
    f"is 65. {L(c33)} is 33. {L(aSq)} plus {L(bSq)} equals {L(c65)}. "
    f"{L(aSq)} exceeds {L(bSq)} by {L(c33)}. {L(a)} times {L(a)} "
    f"equals {L(aSq)}. {L(b)} times {L(b)} equals {L(bSq)}. {L(c4)} "
    f"is 4. {L(a)} plus {L(b)} equals {L(sumSides)}. {L(c4)} times "
    f"{L(sumSides)} equals {L(perimSum)}. What is {L(perimSum)}?",
    "direct system encoding (technique 2) for the two squares' areas "
    "(aSq, bSq found from their sum and difference), each side found "
    "by search-based root extraction; THEOREM-APPLICATION (named: "
    "square perimeter=4*side, Law 1 universal constant '4') for the "
    "final sum of perimeters.")

# 862: f(x)=x^4+x^2+5x, f(5)-f(-5) -> 50. LAW 5/8 TENSION: the even-power
# terms (x^4, x^2) are IDENTICAL at x=5 and x=-5 and cancel exactly in
# the difference, leaving f(5)-f(-5)=2*(5x at x=5)=2*5*5=50 -- avoids
# EVER forming x^4=625 (far over the 300 cap). Named theorem-application
# (even/odd function decomposition), not a literal f(5) and f(-5)
# computation.
g = G()
c5x = g.given(5)                      # the value being evaluated at
c5coef = g.given(5)                   # the coefficient of x in f(x) (distinct
                                       # role, though same numeral)
c2 = g.given(2)                       # from the identity f(a)-f(-a)=2*(odd
                                       # part at a); here the only odd-power
                                       # term is 5x
t1 = g.rel("mul", c5coef, c5x)        # 5*5=25 (value of the linear term)
ans = g.rel("mul", c2, t1)            # 2*25=50
add(862, g.factors, ans, 50,
    f"Consider the numbers {L(c5x)}, {L(c5coef)}, {L(c2)}, {L(t1)}, "
    f"{L(ans)}. {L(c5x)} is 5. {L(c5coef)} is 5. {L(c5coef)} times "
    f"{L(c5x)} equals {L(t1)}. {L(c2)} is 2. {L(c2)} times {L(t1)} "
    f"equals {L(ans)}. What is {L(ans)}?",
    "LAW 5/8 TENSION (flagged for the wheel, high certification-risk): "
    "THEOREM-APPLICATION (named: even/odd function decomposition -- for "
    "f(x)=x^4+x^2+5x, the even-power terms x^4 and x^2 are IDENTICAL "
    "at x=5 and x=-5 and cancel exactly in f(5)-f(-5), a standard "
    "algebraic fact verified by hand), leaving f(5)-f(-5)=2*(5*5)=50. "
    "The graph NEVER computes f(5) or f(-5) individually (both would "
    "require x^4=625, far over the 300 cap) -- this is a MORE "
    "AGGRESSIVE cap-avoidance rearrangement than [744]/[813] (tranche5/"
    "6), since it transforms away the even-power terms ENTIRELY rather "
    "than factoring a single difference-of-squares. COUNTERFACTUAL "
    "note: asking for f(5)+f(-5) instead (the EVEN part) would need the "
    "even-power terms directly and would be UNREPRESENTABLE (x^4=625 "
    "over cap) -- the render specifically depends on the DIFFERENCE "
    "being asked, similar in spirit to [772]'s sharpened-invariance "
    "test.",
    accommodation="add-dup (X plus X self-addition misbinds) avoided: "
                   "the doubling in f(5)-f(-5)=2*(5x) is rendered as "
                   "c2 times t1, not t1 plus t1.")

# 864: rectangular prism diagonal sqrt(l^2+w^2+h^2)=13, l=3,h=12 -> w=4.
# Direct theorem-application (3D distance/diagonal formula).
g = G()
c3 = g.given(3)
c12 = g.given(12)
c13 = g.given(13)
lsq = g.rel("mul", c3, c3)            # 9
hsq = g.rel("mul", c12, c12)          # 144
sumLH = g.rel("add", lsq, hsq)        # 153
diagSq = g.rel("mul", c13, c13)       # 169
wsq = g.rel("sub", diagSq, sumLH)     # 16
w = g.free()
g.rel("mul", w, w, wsq)               # w^2=16 -> w=4
add(864, g.factors, w, 4,
    f"Consider the numbers {L(c3)}, {L(c12)}, {L(c13)}, {L(lsq)}, "
    f"{L(hsq)}, {L(sumLH)}, {L(diagSq)}, {L(wsq)}, {L(w)}. {L(c3)} is "
    f"3. {L(c12)} is 12. {L(c13)} is 13. {L(c3)} times {L(c3)} equals "
    f"{L(lsq)}. {L(c12)} times {L(c12)} equals {L(hsq)}. {L(lsq)} plus "
    f"{L(hsq)} equals {L(sumLH)}. {L(c13)} times {L(c13)} equals "
    f"{L(diagSq)}. {L(diagSq)} exceeds {L(sumLH)} by {L(wsq)}. {L(w)} "
    f"times {L(w)} equals {L(wsq)}. What is {L(w)}?",
    "THEOREM-APPLICATION (named: 3D rectangular-prism diagonal "
    "formula, d^2=l^2+w^2+h^2), degree: rearranged (Law 4) to solve "
    "for w^2 first, then w found by search-based root extraction. All "
    "values under cap (max 169).")

# 866: nabla a nabla b=(a+b)/(1+ab). (1 nabla 2) nabla 3 -> 1. Both
# applications resolve to exact integer division (verified by hand);
# rendered via multiplicative inversion at each step. The operator's own
# "+1" constant reused across both applications (pointer-collision watch,
# matches [694]/[690]'s operator-constant-reuse subtype).
g = G()
ca = g.given(1)
cb = g.given(2)
sumAB = g.rel("add", ca, cb)          # 3
prodAB = g.rel("mul", ca, cb)         # 2
c1 = g.given(1)                       # the operator's own "+1" (distinct
                                       # role from ca despite equal value)
denom1 = g.rel("add", c1, prodAB)     # 1+2=3
r1 = g.free()
g.rel("mul", r1, denom1, sumAB)       # r1*3=3 -> r1=1
cc = g.given(3)
sumAB2 = g.rel("add", r1, cc)         # 1+3=4
prodAB2 = g.rel("mul", r1, cc)        # 1*3=3
denom2 = g.rel("add", c1, prodAB2)    # 1+3=4 (c1 reused)
r2 = g.free()
g.rel("mul", r2, denom2, sumAB2)      # r2*4=4 -> r2=1
add(866, g.factors, r2, 1,
    f"Consider the numbers {L(ca)}, {L(cb)}, {L(sumAB)}, {L(prodAB)}, "
    f"{L(c1)}, {L(denom1)}, {L(r1)}, {L(cc)}, {L(sumAB2)}, "
    f"{L(prodAB2)}, {L(denom2)}, {L(r2)}. {L(ca)} is 1. {L(cb)} is 2. "
    f"{L(ca)} plus {L(cb)} equals {L(sumAB)}. {L(ca)} times {L(cb)} "
    f"equals {L(prodAB)}. {L(c1)} is 1. {L(c1)} plus {L(prodAB)} "
    f"equals {L(denom1)}. {L(r1)} times {L(denom1)} equals {L(sumAB)}. "
    f"{L(cc)} is 3. {L(r1)} plus {L(cc)} equals {L(sumAB2)}. {L(r1)} "
    f"times {L(cc)} equals {L(prodAB2)}. {L(c1)} plus {L(prodAB2)} "
    f"equals {L(denom2)}. {L(r2)} times {L(denom2)} equals "
    f"{L(sumAB2)}. What is {L(r2)}?",
    "direct computation of the source's own custom-operator definition "
    "(a nabla b=(a+b)/(1+ab)), chained twice; both divisions happen to "
    "be exact integer results (verified by hand: 3/3=1, 4/4=1), found "
    "via multiplicative inversion (Worked Example C) rather than fdiv.",
    watch="pointer-collision: c1 (the operator's own '+1' denominator "
          "constant) serves as an argument in 2 separate add factors, "
          "one per nabla application -- matches [694]/[690]'s "
          "operator-constant-reuse subtype.")

# 867: piecewise f: sqrt(x) if x>4, x^2 if x<=4. f(f(f(2))) -> 4.
# f(2)=2^2=4 (2<=4), f(4)=4^2=16 (4<=4, boundary INCLUDED), f(16)=
# sqrt(16)=4 (16>4). Which branch applies at each step is determined
# externally by comparing 2, 4, 16 to the threshold 4 -- ROUTING-FACT
# (Law 13), matches [696]'s precedent.
g = G()
c2 = g.given(2)
sq1 = g.rel("mul", c2, c2)            # f(2)=4 (x<=4 branch)
sq2 = g.rel("mul", sq1, sq1)          # f(4)=16 (x<=4 branch, boundary
                                       # x=4 included)
r = g.free()
g.rel("mul", r, r, sq2)               # f(16)=sqrt(16): r^2=16 -> r=4
                                       # (x>4 branch)
add(867, g.factors, r, 4,
    f"Consider the numbers {L(c2)}, {L(sq1)}, {L(sq2)}, {L(r)}. "
    f"{L(c2)} is 2. {L(c2)} times {L(c2)} equals {L(sq1)}. {L(sq1)} "
    f"times {L(sq1)} equals {L(sq2)}. {L(r)} times {L(r)} equals "
    f"{L(sq2)}. What is {L(r)}?",
    "ROUTING-FACT (Law 13, flagged): which of the piecewise formula's "
    "two branches (x^2 vs sqrt(x)) applies at each of the 3 nested "
    "applications is determined by comparing the running value (2, "
    "then 4, then 16) to the threshold 4 -- derivable from the values "
    "themselves (2<=4, 4<=4 boundary-included, 16>4), but the graph "
    "lacks a comparison/branch primitive, so the routing was decided "
    "externally by hand. Matches [696]'s (tranche5) triangle-longest-"
    "side routing-fact precedent closely: same pattern (external "
    "comparison of concrete derived/given numbers), different domain "
    "(piecewise function branch vs geometric side selection).",
    routing_fact="which of f's 2 piecewise branches (x^2 for x<=4, "
                 "sqrt(x) for x>4) applies at each of the 3 nested "
                 "compositions is determined externally by comparing "
                 "2, 4, and 16 to the threshold 4; derivable from the "
                 "graph-held/given values but the graph has no "
                 "comparison/branch primitive.")

print()
print(f"TOTAL drafted (pre-checks passing): {len(rows)}")
print(f"FAILS: {fails}")
print(f"SKIPS: {[s[0] for s in skips]}")

with open('/home/bryce/mycelium/.cache/book8_t6_prose_pairs_draft.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

with open('/home/bryce/mycelium/.cache/book8_t6_skips.json', 'w') as f:
    json.dump(skips, f, indent=2)

print("done, wrote", len(rows), "rows")
