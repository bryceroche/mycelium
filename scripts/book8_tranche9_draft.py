import json, sys, string
sys.path.insert(0, '/home/bryce/mycelium')
sys.path.insert(0, '/home/bryce/mycelium/scripts')
from tta_alg2_dials import solve2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic

SMP = {"n_vars": 24, "m": 300}
LETTERS = string.ascii_lowercase

CAND_DOC = json.load(open('/home/bryce/mycelium/.cache/book8_candidates_t9.json'))
CANDS = {c["src_idx"]: c for c in CAND_DOC["tranche9"]}
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

    def pct(self, a, b, p):
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
              routing_fact=None, diet=False, engagement_count=None,
              n_vars=24, m=300):
    sol = full_solution(factors, n_vars, m)
    assert sol is not None, f"src {src_idx}: solver failed to find full solution"
    dialect = build_dialect(factors, query_var)
    gen = {
        "src_idx": src_idx, "book": 8, "tranche": 9, "floor": "prime", "fs": True,
        "dialect": dialect, "gate": "PENDING:5view-vote+key", "generation": "21",
        "notes": notes,
    }
    if diet:
        gen["diet_customer"] = CANDS[src_idx]["diet_customer"]
        gen["operand_engagement_count"] = engagement_count
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
        routing_fact=None, engagement_count=None):
    ok = check(src_idx, factors, query_var, expect)
    if not ok:
        fails.append(src_idx)
        return
    diet = src_idx in DIET_SRCS
    rows.append(build_row(src_idx, factors, query_var, notes, watch, accommodation,
                           routing_fact, diet=diet, engagement_count=engagement_count))


def skip(src_idx, reason):
    skips.append((src_idx, reason))
    print(f"[{src_idx}] SKIP: {reason[:70]}...")

# ===========================================================================
# NORMAL POPULATION (30)
# ===========================================================================

# 1117: SKIP -- how many 9s painted on houses 1-50.
skip(1117,
     "9s appear at units-digit positions 9,19,29,39,49 (5 total; no "
     "tens-digit 9s since range stops at 50). Counting DIGIT OCCURRENCES "
     "across a numeric range has no primitive (add/sub/mul/fdiv/pct/sel/"
     "mod only) -- an open, variable-length enumeration over 50 numbers' "
     "digit representations. Operation-shaped skip, new sub-family: "
     "digit-occurrence counting across a range.")

# 1124: SKIP -- primes between 30 and 40.
skip(1124,
     "31,37 are prime (32-39 all composite), count=2. Primality-search "
     "family, no primitive (matches t8's [1260]-flavor and the standing "
     "GCD/factorization cluster's number-theoretic-search relatives).")

# 1125: ratio apple:blueberry:cherry=1:4:3, total=24, find cherry.
# Bookkeeping (Law 8): the ratio-sum arithmetic (1+4+3=8) renders in-graph
# rather than being silently pre-computed; unit found via multiplicative
# inversion (technique C, no fdiv).
g = G()
c1 = g.given(1)
c4 = g.given(4)
c3 = g.given(3)
s1 = g.rel("add", c1, c4)             # 5
ratioSum = g.rel("add", s1, c3)       # 8
c24 = g.given(24)
unit = g.free()
g.rel("mul", ratioSum, unit, c24)     # ratioSum*unit=24 -> unit=3 (mult inv)
ans = g.rel("mul", c3, unit)          # cherry = 3*unit = 9
add(1125, g.factors, ans, 9,
    "BOOKKEEPING (Law 8): the ratio-sum (1+4+3=8) is assembled in-graph, "
    "not silently pre-computed off-graph, before being used as the "
    "multiplicative-inversion divisor (technique C, avoiding fdiv) to "
    "find the unit; cherry count is a fresh multiply of the unit by its "
    "own ratio share (3).")

# 1128: SKIP -- LCM(12,18,30).
skip(1128,
     "LCM(12,18,30)=180 (12=2^2*3, 18=2*3^2, 30=2*3*5 -> LCM=2^2*3^2*5="
     "180). No LCM/factorization primitive in the registry. Matches t8's "
     "GCD/factorization cluster ([1011]/[1073]/[1086]/[1103]), LCM being "
     "the multiplicative-closure twin of that same gap.")

# 1134: 7 bowling balls = 3 canoes (weight); one canoe=28 lbs, find one ball.
# Direct balance-equation rendering (theorem-application: equal-weight
# balance), the one allowed fdiv (k=7, single-digit).
g = G()
c28 = g.given(28)
c3 = g.given(3)
totalWeight = g.rel("mul", c3, c28)   # 3 canoes = 84 lbs = 7 balls' weight
c7 = g.given(7)
ball = g.fdiv(totalWeight, 7)         # ONE fdiv, k=7
add(1134, g.factors, ball, 12,
    "THEOREM-APPLICATION (named: equal-weight balance, 7 balls = 3 "
    "canoes), degree: total weight assembled genuinely (3*28), then "
    "divided by the ball count via the one allowed fdiv (k=7).")

# 1143: gift shop combos: 8 paper * 3 ribbon * 4 cards.
# Direct multiplication-principle chain (theorem-application, named).
g = G()
c8 = g.given(8)
c3 = g.given(3)
c4 = g.given(4)
t1 = g.rel("mul", c8, c3)             # 24
ans = g.rel("mul", t1, c4)            # 96
add(1143, g.factors, ans, 96,
    "THEOREM-APPLICATION (named: multiplication principle for "
    "independent choices), degree: direct three-way product chain, each "
    "source literal entering the arithmetic directly.")

# 1152: SKIP -- largest integer x s.t. x/3+4/5<5/3.
skip(1152,
     "Clearing denominators: 5x+12<25 -> 5x<13 -> x<=2 (largest integer "
     "x=2). Largest-integer-satisfying-a-strict-inequality search, "
     "compounded by fraction-clearing over three distinct denominators "
     "(3,5,3) needing an LCD (15) with no LCM primitive to derive it "
     "in-graph. Matches the standing 'no inequality-satisfying-integer-"
     "search primitive' family (t5 [712], t6 [814]/[838], t7 [877], t8 "
     "[1041]). A floor-based rescue (x=floor((C-1)/k)) is mathematically "
     "valid but UNPRECEDENTED in any worked example this campaign -- "
     "flagged for the bench as a possible future technique rather than "
     "unilaterally introduced here (see law-tension items).")

# 1155: SKIP -- largest integral x s.t. 1/3 < x/5 < 5/8.
skip(1155,
     "1/3<x/5<5/8 -> 5/3<x<25/8 -> 1.67<x<3.125 -> x=3. Same family as "
     "[1152] (strict double inequality, fraction-clearing over "
     "denominators 3,5,8 with no LCM primitive), doubled in complexity "
     "by having TWO bounds instead of one.")

# 1158: SKIP -- naturals in (150,300) divisible by 9.
skip(1158,
     "153,162,...,297: count=17. No 'count multiples of k in a range' "
     "primitive; even the clean identity floor(300/9)-floor(149/9) needs "
     "TWO fdivs, violating the one-fdiv-per-item rule. Matches the "
     "counting/threshold-search family ([1114]/[1106]/[990] t8).")

# 1161: 12-slice pizza, 6 pepperoni, 10 mushroom, every slice >=1 topping.
# Direct inclusion-exclusion (theorem-application, named), matches
# [1074]/[1263] t8 family.
g = G()
c6 = g.given(6)
c10 = g.given(10)
unionSum = g.rel("add", c6, c10)      # 16
c12 = g.given(12)
ans = g.rel("sub", unionSum, c12)     # both = 6+10-12 = 4
add(1161, g.factors, ans, 4,
    "THEOREM-APPLICATION (named: inclusion-exclusion, |A union B|=|A|+"
    "|B|-|A and B|, here total slices=|A union B| since every slice has "
    "at least one topping), degree: union computed genuinely from both "
    "topping counts, then the total subtracted.")

# 1164: isosceles right triangle, angle A=90, leg AC=6, find area.
# Theorem-application (named: area=leg^2/2), one fdiv (k=2).
g = G()
c6 = g.given(6)
legSq = g.rel("mul", c6, c6)          # 36
area = g.fdiv(legSq, 2)               # ONE fdiv, k=2
add(1164, g.factors, area, 18,
    "THEOREM-APPLICATION (named: area of an isosceles right triangle = "
    "leg^2/2, since both legs from the right angle are equal), degree: "
    "leg squared genuinely, halved via the one allowed fdiv.")

# 1170: 3x-5=10x+9, find 4(x+7). LAW-TENSION / NEW TECHNIQUE: solving for
# x directly gives x=-2, a NEGATIVE value not representable in the
# nonneg CSP domain. Rather than representing x at all, substitute
# y=x+7 (exactly the query's own expression) -- this happens to make
# the algebra stay nonneg throughout (y=5), avoiding the negative
# intermediate entirely. 7y=49-14=35 derives from combining the
# equation's own coefficients (10-3=7, the x-coefficient; 5+9=14, the
# constant sum) and the coincidence that the query's offset (7) equals
# the x-coefficient (7) -- specific to THIS problem's numbers, not a
# general claim.
g = G()
c3 = g.given(3)
c10 = g.given(10)
coefDiff = g.rel("sub", c10, c3)      # 7 (=10-3, the x-coefficient)
coefSq = g.rel("mul", coefDiff, coefDiff)  # 49
c5 = g.given(5)
c9 = g.given(9)
constSum = g.rel("add", c5, c9)       # 14 (=5+9, the constant sum)
rhs35 = g.rel("sub", coefSq, constSum)  # 35 (=49-14)
y = g.free()
g.rel("mul", coefDiff, y, rhs35)      # 7*y=35 -> y=5 (mult inversion)
c4 = g.given(4)
ans = g.rel("mul", c4, y)             # 4*(x+7) = 4*y = 20
add(1170, g.factors, ans, 20,
    "LAW-TENSION / NEW TECHNIQUE (negative-intermediate avoidance via "
    "query-substitution): solving 3x-5=10x+9 for x directly gives x=-2, "
    "NEGATIVE and unrepresentable in the nonneg CSP domain. Since the "
    "query itself is 4(x+7), substituting y=x+7 (the query's own "
    "expression) turns the algebra into 7y=35 -> y=5, entirely nonneg "
    "throughout -- never representing x. The coefficient-vs-offset "
    "coincidence (x-coefficient 7 = query offset 7) is specific to this "
    "problem's numbers, not a general claim; flagged for the bench as "
    "the first instance of this rescue technique.")

# 1176: Javier visits 4 attractions, orders (permutations of 4).
# Theorem-application (named: permutation formula 4! = 4*3*2*1).
g = G()
c4 = g.given(4)
c3 = g.given(3)
c2 = g.given(2)
t1 = g.rel("mul", c4, c3)             # 12
ans = g.rel("mul", t1, c2)            # 24 (the trailing *1 is Law 1
                                       # universal-constant, omitted)
add(1176, g.factors, ans, 24,
    "THEOREM-APPLICATION (named: permutation formula, 4!=4*3*2*1), "
    "degree: direct chained product; the trailing *1 factor is omitted "
    "per Law 1 (universal-constant self-combination, multiplying by 1 "
    "is a no-op).")

# 1193: 1 in 5 Americans has allergies, sample=250, expect how many.
# Direct proportion, one fdiv (k=5). NOT pct: 'one out of every five' is
# a ratio phrase, not a percent literal (Law 9 -- p must be a source
# literal matching the pct wording).
g = G()
c250 = g.given(250)
ans = g.fdiv(c250, 5)                 # ONE fdiv, k=5
add(1193, g.factors, ans, 50,
    "THEOREM-APPLICATION (named: proportion/expected-value scaling), "
    "degree: direct fdiv by the ratio's denominator (5). Deliberately "
    "NOT rendered via pct -- 'one out of every five' is a ratio phrase, "
    "not a literal percent, so Law 9 (p must be a source literal) "
    "correctly excludes pct here.")

# 1195: 7 oranges = 5 apples (weight). 28 oranges, how many apples.
# Theorem-application (ratio scaling), one fdiv (k=7).
g = G()
c28 = g.given(28)
c7 = g.given(7)
scale = g.fdiv(c28, 7)                # ONE fdiv, k=7 -> 4
c5 = g.given(5)
ans = g.rel("mul", scale, c5)         # 20
add(1195, g.factors, ans, 20,
    "THEOREM-APPLICATION (named: ratio scaling, weight-equivalence held "
    "constant), degree: scale factor via the one allowed fdiv (28/7=4), "
    "then applied to the apple side.")

# 1198: SKIP -- least integer greater than sqrt(300).
skip(1198,
     "sqrt(300)~17.32, least integer greater=18. Requires a smallest-"
     "integer-exceeding-a-threshold search (technique 1's exact-root "
     "extraction doesn't cover strict inequalities), AND 18^2=324 "
     "exceeds the 300 cap on its own terms -- doubly blocked (no "
     "primitive for the boundary search, and the natural render's own "
     "intermediate breaks the value cap even if a primitive existed).")

# 1199: Allen:Ben work ratio 3:5, total 240 sqft, find Ben's share.
# Bookkeeping (Law 8): ratio-sum rendered in-graph, one fdiv (k=8).
g = G()
c3 = g.given(3)
c5 = g.given(5)
ratioSum = g.rel("add", c3, c5)       # 8
c240 = g.given(240)
unit = g.fdiv(c240, 8)                # ONE fdiv, k=8 -> 30
ans = g.rel("mul", c5, unit)          # 150
add(1199, g.factors, ans, 150,
    "BOOKKEEPING (Law 8): ratio-sum (3+5=8) assembled in-graph before "
    "use as the fdiv divisor (the one allowed fdiv, k=8); Ben's share "
    "is a fresh multiply of the unit by his own ratio share (5).")

# 1203: 40-ft tree, 10-ft shadow; Andrea's 15-inch shadow, height in inches.
# Ratio scaling via multiplicative inversion (avoids fdiv's k=10, which
# is NOT single-digit) -- the height:shadow ratio is unitless (feet
# cancel against feet), so it applies directly to Andrea's inch-shadow
# without a unit-conversion step.
g = G()
c40 = g.given(40)
c10 = g.given(10)
scale = g.free()
g.rel("mul", c10, scale, c40)         # 10*scale=40 -> scale=4 (mult inv,
                                       # avoiding the two-digit fdiv k=10)
c15 = g.given(15)
ans = g.rel("mul", scale, c15)        # 4*15 = 60 inches
add(1203, g.factors, ans, 60,
    "THEOREM-APPLICATION (named: similar-triangles shadow ratio, "
    "height/shadow held constant), degree: scale factor via "
    "multiplicative inversion (technique C) -- deliberately NOT fdiv, "
    "since the natural divisor (10) is two-digit and violates the "
    "single-digit-k rule. The ratio is unitless (feet/feet), so it "
    "applies directly to Andrea's inch-measured shadow with no explicit "
    "unit conversion needed.")

# 1207: SKIP -- for how many digits C is 1C3 a multiple of 3.
skip(1207,
     "1+C+3=C+4 divisible by 3 -> C in {2,5,8}, count=3. Counting digits "
     "(of 10 candidates) satisfying a modular condition -- an aggregate "
     "count over a solution set, no primitive (mod tests ONE value, "
     "doesn't tally how many of a range satisfy it). Matches [1114]'s "
     "t8 family exactly.")

# 1209: Alice/Bob/Carol into 3 distinct officer roles, count orderings.
# Theorem-application (permutation, 3!=6), matches [1176]/[1223].
g = G()
c3 = g.given(3)
c2 = g.given(2)
ans = g.rel("mul", c3, c2)            # 6
add(1209, g.factors, ans, 6,
    "THEOREM-APPLICATION (named: permutation formula, 3!=3*2*1, trailing "
    "*1 omitted per Law 1), degree: direct product of the two decreasing "
    "choice-counts.")

# 1213: SKIP -- GCD(39,91).
skip(1213,
     "39=3*13, 91=7*13, gcd=13. GCD/factorization family, no primitive "
     "(matches t8's [1011]/[1073]/[1086]/[1103] cluster).")

# 1214: apples $4 per 5 lbs, cost for 15 lbs. One fdiv (k=5).
g = G()
c15 = g.given(15)
scale = g.fdiv(c15, 5)                # ONE fdiv, k=5 -> 3
c4 = g.given(4)
ans = g.rel("mul", c4, scale)         # 12
add(1214, g.factors, ans, 12,
    "THEOREM-APPLICATION (named: unit-rate scaling), degree: quantity "
    "ratio via the one allowed fdiv, then applied to the price side.")

# 1221: recipe 30 cookies / 2 cups flour. Eduardo wants 5 dozen (=60)
# cookies. Lexical explicitation of '5 dozen' as 5*12 (Law: knowns
# lexically explicable), scale via multiplicative inversion.
g = G()
c5 = g.given(5)
c12 = g.given(12)
sixty = g.rel("mul", c5, c12)         # 5 dozen = 60 (lexically explicated)
c30 = g.given(30)
scale = g.free()
g.rel("mul", c30, scale, sixty)       # 30*scale=60 -> scale=2 (mult inv)
c2flour = g.given(2)
ans = g.rel("mul", c2flour, scale)    # 4 cups
add(1221, g.factors, ans, 4,
    "THEOREM-APPLICATION (named: recipe scaling), degree: '5 dozen' "
    "lexically explicated as 5*12=60 (a known, not gifted -- the "
    "arithmetic itself renders per Law 8), scale factor via "
    "multiplicative inversion, applied to the flour amount.")

# 1223: three-digit area code, digits {9,8,7} in unknown order.
# Theorem-application (permutation, 3!=6), matches [1209].
g = G()
c3 = g.given(3)
c2 = g.given(2)
ans = g.rel("mul", c3, c2)            # 6
add(1223, g.factors, ans, 6,
    "THEOREM-APPLICATION (named: permutation formula, 3!=3*2*1), "
    "degree: direct product of decreasing choice-counts over the three "
    "digit positions; matches [1209]'s identical shape on different "
    "source content.")

# 1235: Jeff's 5 scores 89,92,88,95,91, arithmetic mean. CAP-AVOIDANCE
# (raw sum 455 exceeds the 300 cap): rebuilt in DEVIATION-FROM-BASELINE
# space (baseline=88, Jeff's own lowest score, matching [1013]'s t8
# technique), one fdiv (k=5).
g = G()
c88 = g.given(88)                     # baseline (also Jeff's own 3rd score)
d0 = g.rel("sub", c88, c88)           # 0 (self-combination, Law 1)
c89 = g.given(89)
d1 = g.rel("sub", c89, c88)           # 1
c92 = g.given(92)
d2 = g.rel("sub", c92, c88)           # 4
c95 = g.given(95)
d3 = g.rel("sub", c95, c88)           # 7
c91 = g.given(91)
d4 = g.rel("sub", c91, c88)           # 3
s1 = g.rel("add", d0, d1)
s2 = g.rel("add", s1, d2)
s3 = g.rel("add", s2, d3)
sumDev = g.rel("add", s3, d4)         # 15
meanDev = g.fdiv(sumDev, 5)           # ONE fdiv, k=5 -> 3
ans = g.rel("add", c88, meanDev)      # 91
add(1235, g.factors, ans, 91,
    "CAP-AVOIDANCE (raw sum of all 5 scores is 455, over the 300 cap; "
    "matches t8's [1013]/[1034] lesson): mean computed in DEVIATION-"
    "FROM-BASELINE space instead (baseline=88, Jeff's own lowest score), "
    "every intermediate well under 20. The one allowed fdiv (k=5) "
    "divides the summed deviations, baseline added back at the end.",
    watch="pointer-collision: c88 (the baseline) is an argument in FIVE "
          "separate sub factors (one per score, including a trivial "
          "self-cancellation for its own score) -- matches [1013]'s "
          "cap-avoidance-forced reuse pattern, milder (5 vs 7 uses).")

# 1246: O'Hara triple (36,25,x): sqrt(36)+sqrt(25)=x. Two independent
# root extractions via search (technique 1), then a direct sum.
g = G()
r1 = g.free()
c36 = g.given(36)
g.rel("mul", r1, r1, c36)             # r1^2=36 -> r1=6
r2 = g.free()
c25 = g.given(25)
g.rel("mul", r2, r2, c25)             # r2^2=25 -> r2=5
ans = g.rel("add", r1, r2)            # 11
add(1246, g.factors, ans, 11,
    "THEOREM-APPLICATION (named: O'Hara triple definition, sqrt(a)+"
    "sqrt(b)=x), degree: two independent search-based root extractions "
    "(technique 1, matches [987]/[1382]), then a genuine fresh sum.")

# 1251: SKIP -- n+10>11 and -4n>-12, integer n.
skip(1251,
     "n>1 and n<3 (dividing by -4 flips the inequality) -> n=2. Same "
     "inequality-satisfying-integer-search family as [1152]/[1155], "
     "DOUBLY blocked: the source's own literals (-4, -12) are genuinely "
     "negative and not representable in the nonneg CSP domain at all, "
     "on top of the missing search primitive.")

# 1260: SKIP -- two-digit primes with units digit 7.
skip(1260,
     "17,37,47,67,97 (27,57,77,87 all composite), count=5. Primality-"
     "search family, no primitive; matches [1124] this tranche.")

# 1263: 40 students, 18 apple pie, 15 chocolate cake, 12 neither, find both.
# Direct inclusion-exclusion (theorem-application), matches [1161]/[1074].
g = G()
c40 = g.given(40)
c12 = g.given(12)
atLeastOne = g.rel("sub", c40, c12)   # 28
c18 = g.given(18)
c15 = g.given(15)
unionSum = g.rel("add", c18, c15)     # 33
ans = g.rel("sub", unionSum, atLeastOne)  # 5
add(1263, g.factors, ans, 5,
    "THEOREM-APPLICATION (named: inclusion-exclusion), degree: "
    "at-least-one count derived from total minus neither, union-sum "
    "from the two liking-counts, both genuinely computed then combined.")

# 1267: 60% selected soda, 20% selected milk, 72 selected soda, find milk.
# NORMAL row with genuine percent literals -- PCT native (Law 9 is
# satisfied regardless of diet status; p=60 and p=20 are both source
# literals), a clean chained-pct exercise on a non-diet row.
g = G()
c72 = g.given(72)
total = g.free()
g.pct(c72, total, 60)                 # 72 is 60 percent of total -> 120
c20 = g.given(20)
milk = g.free()
g.pct(milk, total, 20)                # milk is 20 percent of total -> 24
add(1267, g.factors, milk, 24,
    "PCT native (Law 9 applies uniformly, not just to diet rows): both "
    "60 and 20 are genuine source-literal percents, so the ratified pct "
    "primitive is the lawful choice here even though this row is NOT "
    "diet-tagged -- total derived from the soda relation, milk from the "
    "same total via a second pct, a clean chained-pct instance on the "
    "normal population.")

print(f"\nNormal population so far -- Drafted: {len(rows)}  Skipped: {len(skips)}  Fails: {len(fails)}")

# ===========================================================================
# DIET POPULATION (10) -- interleaved delta tranche, experimental clause +
# engagement-count duty govern these.
# ===========================================================================

# 1327 (dup-add): marble ratio 1:5:3, green=27, total marbles?
# DIET/SOURCE MISMATCH: no doubling relation anywhere in this problem's
# math (ratio parts 1,5,3 and their sums 6,4,8,9 -- none is 2x another).
# Forcing an artificial X+X decomposition of e.g. the '5' or '3' part
# would be cosmetic (Law 2 risk), not a natural render. Drafted via the
# straightforward ratio technique instead (bookkeeping + mult inversion,
# matches [1125]/[1199] this tranche); NOT tagged as exercising add-dup.
g = G()
c1 = g.given(1)
c5 = g.given(5)
c3 = g.given(3)
s1 = g.rel("add", c1, c5)             # 6
ratioSum = g.rel("add", s1, c3)       # 9
c27 = g.given(27)
unit = g.free()
g.rel("mul", c3, unit, c27)           # 3*unit=27 -> unit=9 (mult inv)
ans = g.rel("mul", ratioSum, unit)    # 9*9 = 81
add(1327, g.factors, ans, 81,
    "DIET/SOURCE MISMATCH (dup-add): no doubling relation exists "
    "anywhere in this problem's math -- ratio parts 1,5,3 and every "
    "pairwise sum/difference among them (6,4,8,9) fail to show a 2x "
    "relationship, so there is no natural place for the X+X form "
    "without inventing a cosmetic decomposition (Law 2 risk, exactly "
    "the caution t8's [1286] raised). Rendered via the standard ratio "
    "technique instead (bookkeeping ratio-sum + multiplicative "
    "inversion, matches [1125]/[1199]); this row does NOT exercise the "
    "diet construction.",
    engagement_count=None)

# 1354 (two-free-var): x^2=y-3, x=-5, find y. DIET/SOURCE MISMATCH: x is
# a HARD SOURCE-STATED LITERAL (x=-5), not an algebraically-determined
# unknown the way [772]/[856]/t8's [1133] required -- there is no
# equation from which x's value emerges; the source simply hands it to
# us. Giving x directly makes it a singleton, not free; inventing a
# second 'equation' just to nominally free x (e.g. an add-zero identity)
# would be pure cosmetic jointness barred by Law 2, worse than [1133]'s
# already-weak precedent (there, x was genuinely undetermined by any
# single source statement). Rendered as a single-unknown (y) problem via
# MAGNITUDE-FOLD (technique 3): since only x^2 matters, x's magnitude
# (5) stands in for the unrepresentable negative value.
g = G()
xmag = g.given(5)                     # |x| = |-5| = 5 (magnitude-fold,
                                       # only x^2 is ever needed)
xsq = g.rel("mul", xmag, xmag)        # 25
c3 = g.given(3)
ans = g.rel("add", xsq, c3)           # y = 28
add(1354, g.factors, ans, 28,
    "DIET/SOURCE MISMATCH (two-free-var): x is a hard SOURCE-STATED "
    "literal (x=-5), not an algebraically-determined unknown -- unlike "
    "[772]/[856]/t8's [1133] (where the 'known' operand was reused "
    "across two equations but the unknown itself was never directly "
    "given), here there is no equation from which x's value emerges at "
    "all. Forcing a joint two-free-var frame would require either "
    "giving x directly (making it a singleton, not free) or inventing a "
    "cosmetic second equation (Law 2 violation). Rendered instead as a "
    "single-unknown (y) problem via MAGNITUDE-FOLD (technique 3, "
    "matches [897]/[902]/[989]): only x^2 is ever needed, so x's "
    "magnitude (5) stands in for the unrepresentable negative value. "
    "This row does NOT exercise the diet construction.",
    engagement_count=None)

# 1413: SKIP (diet: dup-add) -- least N divisible by 2,3,4,5,6 (LCM).
skip(1413,
     "LCM(2,3,4,5,6)=60. PARTIAL DIET FOOTHOLD FOUND: the "
     "divisible-by-2 fact DOES have a natural add-dup rendering (N = "
     "half+half, args=[half,half]), and divisibility by 3,4,5,6 could "
     "in principle use the mod primitive (a mod k = 0). But the overall "
     "problem remains BLOCKED regardless: the CSP finds ANY satisfying "
     "assignment, not the MINIMUM one, and the feasible set within the "
     "300 cap has FIVE members (60,120,180,240,300) -- without an "
     "explicit upper bound (not stated in the source, would violate Law "
     "9 to invent one), solve2's own uniqueness re-check would reject "
     "any render as non-unique. The gap is genuinely about minimality-"
     "search (no LCM primitive to derive 60 as the least value, and the "
     "boundary clause's slack-verification form doesn't apply to a "
     "multi-valued feasible SET rather than a single binding point). "
     "Diet disposition: NOT exercised (partial foothold noted, "
     "insufficient to rescue the row).")

# 1429: SKIP (diet: pct) -- Lucy $19.23, popsicles $1.60 each, max count.
skip(1429,
     "floor(1923/160)=12 (12*160=1920<=1923, 13*160=2080>1923). DOUBLY "
     "BLOCKED: (a) converting to a common cent-unit forms 1900+ as an "
     "intermediate (19*100), FAR over the 300 cap, with no smaller-unit "
     "path available (gcd(1923,160)=1, no reduction helps); (b) even "
     "setting the cap aside, the natural divisor (160) is nowhere near "
     "single-digit, violating the fdiv rule outright. DIET/SOURCE "
     "MISMATCH ALSO: no percent language anywhere in this source (it's "
     "a currency floor-division problem) -- the pct construction has no "
     "foothold here independent of the cap/fdiv blockers. Diet "
     "disposition: NOT exercised, source unrenderable AND construction-"
     "mismatched.")

# 1437 (sub-distract): four digits {2,4,6,7}, count 2-digit integers with
# no repeated digit. EXERCISED NATIVELY via COMPLEMENTARY COUNTING: total
# ordered pairs WITH repetition allowed (4*4=16) minus the repeated-digit
# pairs (one per digit, 4) gives the no-repeat count (12) -- this is a
# standard, legitimate combinatorics technique that happens to produce a
# genuine 'X exceeds Y by Z' relation (16 exceeds 4 by 12), a cleaner and
# more natural fit than the direct nPk=4*3 permutation render would have
# been for THIS diet clause (that render has no subtraction at all).
g = G()
c4 = g.given(4)                       # count of available digits
total = g.rel("mul", c4, c4)          # 16 = ordered pairs WITH repetition
                                       # (mul(x,x) form, matches [957]/
                                       # [1382] precedent)
ans = g.rel("sub", total, c4)         # SUB-DISTRACT NATIVE: 16 exceeds
                                       # (repeated-pair count, reusing c4
                                       # since each of the 4 digits
                                       # contributes exactly one repeated
                                       # pair) by 12
add(1437, g.factors, ans, 12,
    "SUB-DISTRACT NATIVE (diet clause, via COMPLEMENTARY COUNTING): "
    "rather than the direct permutation render (4*3=12, which has no "
    "subtraction at all and would have been a diet mismatch), this uses "
    "the standard complementary-counting technique -- total ordered "
    "pairs allowing repetition (4*4=16, mul(x,x) form) exceeds the "
    "repeated-digit-pair count (4, one per digit, reusing the SAME "
    "given c4 a third time) by the no-repeat answer (12). Genuine 'X "
    "exceeds Y by Z' native form, mathematically standard (not an "
    "invented shortcut) and more faithful to the diet clause's intent "
    "than the trivial permutation shape.",
    engagement_count=2)
    # engagement_count: key operand = c4 as SUBTRAHEND (the distractor
    # side of 'X exceeds Y by Z'). Prior factors referencing c4 before
    # the sub: its own given (1) + the mul(c4,c4,total) factor (1) = 2.

# 1469 (pct): 200 students, 70 band, 95 chorus, 150 band-and/or-chorus,
# find both. DIET/SOURCE MISMATCH: no percent language anywhere in this
# source (70, 95, 150 are all raw counts, not stated as percents of 200)
# -- Law 9 correctly excludes pct (p would not be a source literal; "
# back-computing 70/200=35% and forcing pct(70,200,35) would violate the
# same law [1229] used to correctly DECLINE a non-literal fraction).
# Rendered via direct inclusion-exclusion instead.
g = G()
c70 = g.given(70)
c95 = g.given(95)
unionSum = g.rel("add", c70, c95)     # 165
c150 = g.given(150)
ans = g.rel("sub", unionSum, c150)    # 15
add(1469, g.factors, ans, 15,
    "DIET/SOURCE MISMATCH (pct): no percent language anywhere in this "
    "source -- 70, 95, 150 are all raw student counts, never stated as "
    "percents of the 200 total. Forcing pct(70,total,35) (backward-"
    "computing 35% from 70/200) would violate Law 9 exactly the way "
    "[1229] correctly declined a non-literal 1/3 fraction. Rendered via "
    "direct inclusion-exclusion instead (matches [1161]/[1263]/[1074] "
    "this campaign); this row does NOT exercise the diet construction.",
    engagement_count=None)

# 1499: SKIP (diet: pct) -- 24 students, groups of AT MOST 10, least
# number of groups (ceiling division).
skip(1499,
     "ceil(24/10)=3 (2 groups of 10 leaves 4 uncovered; 3 groups "
     "suffice). Exact precedent match with t8's [1041] (minimum dimes): "
     "a CEILING (not floor) threshold search, with the natural divisor "
     "(10) two-digit and thus fdiv-ineligible regardless. Maintaining "
     "consistency with [1041]'s immediately-preceding-tranche ruling "
     "rather than unilaterally introducing a sub-based sandwich "
     "technique never used in any worked example this campaign (see "
     "law-tension items). DIET/SOURCE MISMATCH ALSO: no percent "
     "language in this source (a group-partition problem). Diet "
     "disposition: NOT exercised, both blocked and mismatched.")

# 1698 (sub-distract): sqrt(36+64) - sqrt(25-16). EXERCISED NATIVELY,
# clean fit: the final operation IS literally 'X exceeds Y by Z' (10
# exceeds 3 by 7), embedded in a genuinely distractor-rich context (two
# nested radicals, one built from an inner ADDITION and one from an
# inner SUBTRACTION, exactly matching the clause's 'distractor-rich
# contexts' wording).
g = G()
c36 = g.given(36)
c64 = g.given(64)
sum1 = g.rel("add", c36, c64)         # 100
r1 = g.free()
g.rel("mul", r1, r1, sum1)            # r1^2=100 -> r1=10
c25 = g.given(25)
c16 = g.given(16)
diff1 = g.rel("sub", c25, c16)        # 9 (inner distractor subtraction)
r2 = g.free()
g.rel("mul", r2, r2, diff1)           # r2^2=9 -> r2=3
ans = g.rel("sub", r1, r2)            # SUB-DISTRACT NATIVE: 10 exceeds 3
                                       # by 7
add(1698, g.factors, ans, 7,
    "SUB-DISTRACT NATIVE (diet clause, flagship instance): the final "
    "operation IS the source's own 'X exceeds Y by Z' form (10 exceeds "
    "3 by 7) with no accommodation needed. Genuinely distractor-rich "
    "context per the clause's own wording: two nested radicals, one "
    "resolved from an inner ADDITION (36+64) and one from an inner "
    "SUBTRACTION (25-16), both requiring search-based root extraction "
    "(technique 1, matches [987]/[1246]/[1382]) before the outer "
    "sub-distract relation ever fires.",
    engagement_count=1)
    # engagement_count: key operand = r2 (subtrahend, 'Y'). Prior
    # factors referencing r2: only its own defining mul (1).

# 1720 (dup-add): '20% of 10% of a number is 12, what is 10% of 20% of
# the same number?' EXERCISED NATIVELY via the algebraic identity
# 20%=2x10%: rather than a bare residue (Law 3, matches [1009]'s
# invariant-to-input family), the render genuinely derives u=10%-of-n
# (via pct, search-pinned by the given 12) then represents 20%-of-n as
# u+u (ADD-DUP NATIVE, since 20%=2x10% always, independent of n) before
# the final pct closes the loop. The numeric answer necessarily equals
# the given (12, by the commutativity the source is testing), but the
# GRAPH is not a bare echo -- u must be genuinely solved via a pct
# constraint, and twentyPctN is a genuine add-dup relation, not an
# invented shortcut.
g = G()
u = g.free()                          # u = 10% of n (unnamed n never
                                       # needs to appear at all)
c12 = g.given(12)
g.pct(c12, u, 20)                     # c12 is 20 percent of u -> u=60
                                       # (search-pinned, technique 1
                                       # style via pct)
twentyPctN = g.free()
g.rel("add", u, u, twentyPctN)        # ADD-DUP NATIVE: 20% of n = 2*u
                                       # (algebraic identity, always
                                       # true regardless of n)
ans = g.free()
g.pct(ans, twentyPctN, 10)            # ans is 10 percent of twentyPctN
add(1720, g.factors, ans, 12,
    "ADD-DUP NATIVE (diet clause), via the algebraic identity 20%=2x10%: "
    "rather than a bare residue (the naive reading -- order of percent "
    "multiplication doesn't matter, matches [1009]'s invariant-to-input "
    "family -- would make this a Law-3 skip), the render genuinely "
    "solves u=10%-of-n via a pct constraint (search-pinned by the given "
    "12, technique 1 style), represents 20%-of-n as u+u (native ADD-DUP, "
    "since 20%=2x10% is an algebraic identity true for ANY n, not a "
    "coincidence of this problem's specific numbers), then closes the "
    "loop with a final pct. The numeric answer necessarily equals the "
    "given (12, that IS the commutativity fact being tested), but the "
    "GRAPH requires genuine multi-step solving -- u is not directly "
    "given, it is search-pinned -- so this passes the counterfactual "
    "test the way [1009] failed it (an empty/trivial graph would NOT "
    "suffice here).",
    engagement_count=1)
    # engagement_count: key operand = u (the doubled value, 'X'). Prior
    # factors referencing u: only its own defining pct constraint (1).

# 1733 (two-free-var): 200% of x = 50% of y, x=16, find y. DIET/SOURCE
# MISMATCH: x is a HARD SOURCE-STATED LITERAL (x=16), exactly the same
# structural issue as [1354] -- there is no equation from which x's
# value emerges, so it cannot be rendered as a genuinely free variable
# without either giving it directly (not free) or inventing a cosmetic
# pinning equation (Law 2 risk). Rendered as a single-unknown (y)
# problem via a direct pct chain instead (both p values, 200 and 50, ARE
# genuine source literals, so pct usage itself is lawful here -- just
# not evidence of two-free-var jointness).
g = G()
c16 = g.given(16)
z = g.free()
g.pct(z, c16, 200)                    # z is 200 percent of 16 -> z=32
ans = g.free()
g.pct(z, ans, 50)                     # z is 50 percent of ans -> ans=64
add(1733, g.factors, ans, 64,
    "DIET/SOURCE MISMATCH (two-free-var): x is a hard SOURCE-STATED "
    "literal (x=16), the identical structural issue as [1354] this "
    "tranche -- no equation determines x, so a genuine joint frame is "
    "unavailable without either giving x directly (not free) or "
    "inventing a cosmetic pinning equation (Law 2 risk). Rendered as a "
    "single-unknown (y) problem via a direct pct chain instead (200% of "
    "16, then solving 50%-of-y=that value) -- both p values ARE genuine "
    "source literals so pct itself is lawful here, it just isn't "
    "evidence of the target construction. This row does NOT exercise "
    "the diet construction.",
    engagement_count=None)

print(f"\nFINAL -- Drafted: {len(rows)}  Skipped: {len(skips)}  Fails: {len(fails)}")
print(f"Total accounted: {len(rows) + len(skips) + len(fails)} / 40")
if fails:
    print("FAILS:", fails)

with open('/home/bryce/mycelium/.cache/book8_t9_prose_pairs_draft.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print("Wrote", len(rows), "rows to .cache/book8_t9_prose_pairs_draft.jsonl")

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
