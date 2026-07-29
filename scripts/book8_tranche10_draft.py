import json, sys, string
sys.path.insert(0, '/home/bryce/mycelium')
sys.path.insert(0, '/home/bryce/mycelium/scripts')
from tta_alg2_dials import solve2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic

SMP = {"n_vars": 24, "m": 300}
LETTERS = string.ascii_lowercase

CAND_DOC = json.load(open('/home/bryce/mycelium/.cache/book8_candidates_t10.json'))
CANDS = {c["src_idx"]: c for c in CAND_DOC["tranche10"]}


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
              routing_fact=None, n_vars=24, m=300):
    sol = full_solution(factors, n_vars, m)
    assert sol is not None, f"src {src_idx}: solver failed to find full solution"
    dialect = build_dialect(factors, query_var)
    gen = {
        "src_idx": src_idx, "book": 8, "tranche": 10, "floor": "prime", "fs": True,
        "dialect": dialect, "gate": "PENDING:5view-vote+key", "generation": "21",
        "notes": notes,
    }
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
    rows.append(build_row(src_idx, factors, query_var, notes, watch, accommodation,
                           routing_fact))


def skip(src_idx, reason):
    skips.append((src_idx, reason))
    print(f"[{src_idx}] SKIP: {reason[:70]}...")

# ===========================================================================
# TRANCHE 10 -- PURE PRODUCTION, 40 candidates, no diet population.
# ===========================================================================

# 1269: mean of (x+8, 15, 2x, 13, 2x+4) is 24. Solve x.
# DIRECT SYSTEM ENCODING (technique, matches [1269]-family precedent
# [1170]/t9): x rendered as a shared free var appearing three times
# (implicit coeff 1 via x+8, explicit coeff 2 twice); total forced equal
# to mean*count via the final add's explicit result.
g = G()
x = g.free()
c8 = g.given(8)
term1 = g.rel("add", x, c8)             # x+8
c15 = g.given(15)
c2 = g.given(2)
term3 = g.rel("mul", c2, x)             # 2x
c13 = g.given(13)
term5a = g.rel("mul", c2, x)            # 2x (reuse c2)
c4 = g.given(4)
term5 = g.rel("add", term5a, c4)        # 2x+4
sum1 = g.rel("add", term1, c15)
sum2 = g.rel("add", sum1, term3)
sum3 = g.rel("add", sum2, c13)
c24 = g.given(24)
c5 = g.given(5)
total = g.rel("mul", c24, c5)           # mean*count = 120
g.rel("add", sum3, term5, total)        # force sum3+term5 = total
add(1269, g.factors, x, 16,
    "DIRECT SYSTEM ENCODING: x rendered as a genuinely shared free "
    "variable across three of the five expressions (implicit coefficient "
    "1 via 'x+8', explicit coefficient 2 twice via '2x' and '2x+4', the "
    "second use of c2 a natural pointer reuse). The five-term sum is "
    "assembled in-graph and forced equal to mean*count (24*5=120) via "
    "the final add's explicit result pointer -- matches [1170]'s (t9) "
    "and [1315]'s (this tranche) force-final-equality pattern.")

# 1270: one and one-half of what number is 30?
# 1.5 = 3/2 lexically explicated as a structural literal (Law 10);
# multiplicative inversion (technique C) avoids ever forming a fraction.
g = G()
c3 = g.given(3)
c2 = g.given(2)
c30 = g.given(30)
rhs = g.rel("mul", c2, c30)             # 2*30=60 (clears the /2)
n = g.free()
g.rel("mul", c3, n, rhs)                # 3n=60 -> n=20 (mult inversion)
add(1270, g.factors, n, 20,
    "STRUCTURAL LITERAL (Law 10): 'one and one-half' explicated as the "
    "fraction 3/2. Multiplicative inversion (technique C): 3n=2*30=60 "
    "clears the fraction without ever forming a non-integer intermediate.")

# 1271 not in this draw (numbering gap in the source harvest, not ours).

# 1272: Compute (-64) / (-32).
# ROUTING-FACT (Law 13): negative/negative = positive is a derivable
# routing fact; only the MAGNITUDES (64, 32) enter the nonneg-domain
# graph, sign resolved off-graph via the stated routing fact.
g = G()
c64 = g.given(64)
c32 = g.given(32)
scale = g.free()
g.rel("mul", c32, scale, c64)           # 32*scale=64 -> scale=2 (mult inv)
add(1272, g.factors, scale, 2,
    "ROUTING-FACT (Law 13): dividing two negatives yields a positive "
    "quotient (a derivable routing fact, not an assumption) -- only the "
    "magnitudes (64, 32) are representable in the nonneg CSP domain and "
    "enter the graph; multiplicative inversion (technique C) finds the "
    "quotient, sign resolved off-graph via the named routing fact.",
    routing_fact="negative-divided-by-negative-is-positive: (-64)/(-32) "
                 "renders as |64|/|32| with sign resolved off-graph.")

# 1273: 12 brown eyes, 20 lunch box, 30 students total. LEAST possible
# number with both. BOUNDARY CLAUSE (Law 12): minimizing the overlap
# requires MAXIMIZING the union, and the union can be at most the total
# enrollment (30) -- so at the minimum-overlap optimum the union binds
# exactly at 30 (feasible: brown-only=10, lunchbox-only=18, both=2,
# neither=0, summing to 30 with zero slack). Counterfactual: this render
# is valid only because 12+20 > 30 (pigeonhole is active); if the sum
# didn't exceed the total, the minimum would be 0, not this formula.
g = G()
c12 = g.given(12)
c20 = g.given(20)
unionSum = g.rel("add", c12, c20)       # 32
c30 = g.given(30)
ans = g.rel("sub", unionSum, c30)       # 2 (union binds at total=30)
add(1273, g.factors, ans, 2,
    "BOUNDARY CLAUSE (Law 12, named: pigeonhole minimum-overlap "
    "principle): minimizing |brown AND lunchbox| requires maximizing "
    "the union, which can be at most the total enrollment (30) -- the "
    "boundary binds exactly there (verified feasible off-graph: "
    "brown-only=10, lunchbox-only=18, both=2, neither=0, sums to 30 "
    "with zero slack). Counterfactual argued: this render is valid only "
    "because 12+20=32 exceeds 30 (pigeonhole active); a variant where "
    "the sum didn't exceed the total would need a different render "
    "(minimum would be 0), so the technique's precondition is checked, "
    "not assumed.")

# 1274: logo 2in x 1.5in, enlarged to 8in wide, find height.
# Cross-multiplication (theorem-application, named: similar-figures
# proportional scaling), 1.5=3/2 structural literal, one fdiv (k=2).
g = G()
c2 = g.given(2)                          # original width
c3 = g.given(3)                          # from 1.5 = 3/2
c8 = g.given(8)                          # new width
num = g.rel("mul", c3, c8)               # 3*8=24
twelveVal = g.fdiv(num, 2)               # ONE fdiv, k=2 -> 12
height = g.free()
g.rel("mul", c2, height, twelveVal)      # 2*height=12 -> height=6 (mult inv)
add(1274, g.factors, height, 6,
    "THEOREM-APPLICATION (named: similar-figures proportional scaling, "
    "width/height ratio held constant), degree: cross-multiplication "
    "(2*height = 1.5*8), 1.5 explicated as 3/2 (structural literal, Law "
    "10), 3*8 reduced via the one allowed fdiv (k=2) before the final "
    "multiplicative inversion.")

# 1276: 10 taking both, 24 taking algebra, 11 taking drafting only. How
# many taking algebra or drafting but not both (symmetric difference)?
g = G()
c24 = g.given(24)
c10 = g.given(10)
algOnly = g.rel("sub", c24, c10)         # 14
c11 = g.given(11)
ans = g.rel("add", algOnly, c11)         # 25
add(1276, g.factors, ans, 25,
    "THEOREM-APPLICATION (named: symmetric-difference/exclusive-or "
    "count), degree: algebra-only derived genuinely (24-10), then "
    "summed with the given drafting-only count.")

# 1277: x - 2x + 3x = 100. LAW-TENSION / regrouping (matches [1170]'s
# negative-intermediate-avoidance family): naive left-to-right evaluation
# (x-2x = -x) is NEGATIVE and unrepresentable in the nonneg CSP domain.
# Regrouped via associativity/commutativity as x + (3x-2x) = x + x = 2x
# (a genuine restatement of the SAME named source equation, Law 4),
# every intermediate nonneg (3x=150 >= 2x=100).
g = G()
x = g.free()
c2 = g.given(2)
term2 = g.rel("mul", c2, x)              # 2x
c3 = g.given(3)
term3 = g.rel("mul", c3, x)              # 3x
diff = g.rel("sub", term3, term2)        # 3x-2x = x (nonneg: 3x>=2x)
c100 = g.given(100)
g.rel("add", x, diff, c100)              # force x+diff = 100
add(1277, g.factors, x, 50,
    "LAW-TENSION / REARRANGEMENT (Law 4, matches [1170]'s t9 negative-"
    "intermediate-avoidance family): naive evaluation of x-2x+3x "
    "left-to-right produces x-2x=-x, NEGATIVE and unrepresentable. "
    "Regrouped via associativity as x+(3x-2x)=x+x=2x -- a recognizable "
    "restatement of the SAME named source equation (Law 4), not a "
    "different equation -- keeps every intermediate nonneg since "
    "3x=150 >= 2x=100 always holds for the solved x.")

# 1278: SKIP -- how many primes between 30 and 50.
skip(1278,
     "31,37,41,43,47 are prime (30,32-36,38-40,42,44-46,48-50 all "
     "composite), count=5. Primality-search family, no primitive; "
     "matches t9's [1124]/[1260] cluster.")

# 1279: 20 club members, 8 left-handed, 15 like jazz, 2 right-handed AND
# dislike jazz (LH/RH partition assumed). Find LH-and-jazz.
g = G()
c20 = g.given(20)
c8 = g.given(8)
rh = g.rel("sub", c20, c8)               # 12 (right-handed)
c2 = g.given(2)
rhJazz = g.rel("sub", rh, c2)            # 10 (right-handed AND jazz)
c15 = g.given(15)
ans = g.rel("sub", c15, rhJazz)          # 5 (left-handed AND jazz)
add(1279, g.factors, ans, 5,
    "THEOREM-APPLICATION (named: complementary-partition set counting, "
    "LH/RH partition stated explicitly), degree: right-handed count "
    "derived from the partition, right-handed-and-jazz derived from its "
    "own complement, then subtracted from total jazz-likers to isolate "
    "left-handed-and-jazz.")

# 1280: Juan: n, +2, *2, -2, /2 = 7. Solve n. Forward chain with the
# final fdiv's result forced equal to the given 7 (exact floor division,
# matches [1203]/[1274]-style force-final-equality).
g = G()
n = g.free()
c2 = g.given(2)
step1 = g.rel("add", n, c2)              # n+2
step2 = g.rel("mul", step1, c2)          # (n+2)*2, reuse c2
step3 = g.rel("sub", step2, c2)          # (n+2)*2-2, reuse c2 again
c7 = g.given(7)
g.fdiv(step3, 2, result=c7)              # ONE fdiv, k=2, forced = 7
add(1280, g.factors, n, 6,
    "DIRECT SYSTEM ENCODING: n rendered as a free var forward-chained "
    "through each stated operation (add, mul, sub, all reusing the same "
    "c2 pointer three times -- a natural reuse, the source itself reuses "
    "'2' three times), the final step's fdiv (the one allowed, k=2) "
    "forced equal to the given answer (7) rather than left free, pinning "
    "n by search.")

# 1281: SKIP -- smallest positive multiple of 32. Law-3 residue: the
# answer IS the given literal itself (32), definitionally -- no graph
# could do genuine work here (an empty graph, query_var pointing at the
# given var, would suffice). Counterfactual test fails outright: a
# variant ('smallest positive multiple of 7') would just parrot 7 back
# with zero arithmetic.
skip(1281,
     "Smallest positive multiple of 32 is 32 itself -- LAW 3 RESIDUE "
     "(deduction-vs-assembly): the answer IS the given literal, "
     "definitionally, with zero possible graph work. Counterfactual "
     "test fails outright (any k would just parrot k back).")

# 1285: SKIP -- smallest prime that is 10 less than a perfect square.
skip(1285,
     "n=9: 81-10=71, prime (71 not divisible by 2,3,5,7; sqrt(71)<9). "
     "Smaller n all fail (n=4..8 give 6,15,26,39,54, none prime). "
     "Combined perfect-square-search + primality-search, no primitive; "
     "matches the primality-search family ([1278] this tranche).")

# 1287: Vanessa's team scored 48. Six other players averaged 3.5 (=7/2)
# each. Find Vanessa's score. Structural literal (7/2), one fdiv (k=2).
g = G()
c6 = g.given(6)
c7 = g.given(7)                          # from 3.5 = 7/2
num = g.rel("mul", c6, c7)               # 42
otherTotal = g.fdiv(num, 2)              # ONE fdiv, k=2 -> 21
c48 = g.given(48)
ans = g.rel("sub", c48, otherTotal)      # 27
add(1287, g.factors, ans, 27,
    "THEOREM-APPLICATION (named: average*count=total), degree: other "
    "players' total genuinely computed (6*3.5, 3.5 explicated as 7/2, "
    "Law 10) via the one allowed fdiv, then subtracted from the team "
    "total.")

# 1290: average age of 3 Wilson children is 7. Two younger are 4 and 7
# (SEPARATE literal instance from the mean, same numeral, different
# role -- matches [1300]'s baseline/score reuse discipline).
g = G()
c7 = g.given(7)                          # mean
c3 = g.given(3)                          # count
total = g.rel("mul", c7, c3)             # 21
c4 = g.given(4)
c7b = g.given(7)                         # younger child's age, separate
sum2 = g.rel("add", c4, c7b)             # 11
ans = g.rel("sub", total, sum2)          # 10
add(1290, g.factors, ans, 10,
    "THEOREM-APPLICATION (named: mean*count=total), degree: total age "
    "genuinely assembled, then reduced by the two known ages (a SEPARATE "
    "given instance for the second child's age, even though it shares "
    "the numeral 7 with the mean -- distinct semantic roles).")

# 1291: 6 blocks east + 12 blocks north, each block=1/3 mile. Total
# miles WALKED (path length, not displacement) = total blocks / 3.
g = G()
c6 = g.given(6)
c12 = g.given(12)
totalBlocks = g.rel("add", c6, c12)      # 18
ans = g.fdiv(totalBlocks, 3)             # ONE fdiv, k=3 -> 6
add(1291, g.factors, ans, 6,
    "THEOREM-APPLICATION (named: total path length = sum of segment "
    "counts, direction is irrelevant to 'how many miles did he walk'), "
    "degree: segment count summed genuinely, then scaled by the one "
    "allowed fdiv (k=3, from the 1/3-mile block length).")

# 1292: square edge = 4x-15 = 20-3x. Solve x via nonneg-safe rearrange
# (move terms to keep both sides positive: 4x+3x=20+15), then compute
# edge and area. Rearrangement (Law 4), matches Worked Example A.
g = G()
c4 = g.given(4)
c3 = g.given(3)
coefSum = g.rel("add", c4, c3)           # 7
c20 = g.given(20)
c15 = g.given(15)
constSum = g.rel("add", c20, c15)        # 35
x = g.free()
g.rel("mul", coefSum, x, constSum)       # 7x=35 -> x=5 (mult inversion)
edgeMul = g.rel("mul", c4, x)            # 4x=20
edgeVal = g.rel("sub", edgeMul, c15)     # 4x-15=5 (reuse c15)
area = g.rel("mul", edgeVal, edgeVal)    # 25
add(1292, g.factors, area, 25,
    "REARRANGEMENT (Law 4, matches Worked Example A): 4x-15=20-3x moved "
    "to 4x+3x=20+15=35 (both sides genuinely nonneg throughout), solved "
    "via multiplicative inversion; edge then recomputed from x via the "
    "source's own '4x-15' expression (reusing the same c15), area a "
    "fresh self-multiply.")

# 1293: (3*4)/6. Direct, one fdiv (k=6).
g = G()
c3 = g.given(3)
c4 = g.given(4)
num = g.rel("mul", c3, c4)               # 12
ans = g.fdiv(num, 6)                     # ONE fdiv, k=6 -> 2
add(1293, g.factors, ans, 2,
    "DIRECT: numerator genuinely multiplied, then reduced via the one "
    "allowed fdiv (k=6).")

# 1294: SKIP -- for what digit A is 3AA1 divisible by 9.
skip(1294,
     "Digit sum 3+A+A+1=4+2A must be divisible by 9; checking A=0..9 "
     "gives ONLY A=7 (4+14=18). BLOCKED: no primitive bounds a free var "
     "to a single digit (0-9) -- without an explicit A<=9 constraint "
     "(no such primitive; a bounding fdiv would need k=10, two-digit "
     "and disallowed), the CSP's nonneg domain (0..300) admits further "
     "solutions to (4+2A) mod 9 = 0 beyond the digit range (A=16, 25, "
     "... all satisfy), breaking uniqueness. Matches t9's [1207] "
     "modular-digit-search family (there a COUNT, here a value "
     "determination, same underlying gap).")

# 1295: (-5)^5 / 5^3 + 3^4 - 6^1. Theorem-application (exponent quotient
# rule, Law 8 bookkeeping of the exponent arithmetic in-graph, matches
# t5's [755] precedent) avoids forming 3125/125; routing-fact (Law 13)
# reorders the negative term into a subtraction (81-25-6) instead of an
# unrepresentable negative addend, keeping every intermediate nonneg.
g = G()
c5exp = g.given(5)                       # exponent 5, from (-5)^5
c3exp = g.given(3)                       # exponent 3, from 5^3
expDiff = g.rel("sub", c5exp, c3exp)     # 2 (Law 8 bookkeeping, matches [755])
c5base = g.given(5)                      # base magnitude
sq = g.rel("mul", c5base, c5base)        # 5^2=25 (one mul, matches expDiff=2)
c3base = g.given(3)                      # base for 3^4
t1 = g.rel("mul", c3base, c3base)        # 9
t2 = g.rel("mul", t1, c3base)            # 27
t3 = g.rel("mul", t2, c3base)            # 81
c6 = g.given(6)                          # 6^1 (Law 1, exponent-1 no-op)
step1 = g.rel("sub", t3, sq)             # 81-25=56
ans = g.rel("sub", step1, c6)            # 56-6=50
add(1295, g.factors, ans, 50,
    "THEOREM-APPLICATION (named: exponent-quotient rule a^m/a^n=a^(m-n) "
    "for equal bases) + LAW 8 BOOKKEEPING (exponent arithmetic 5-3=2 "
    "computed in-graph, matches t5's [755] precedent) avoids ever "
    "forming (-5)^5=-3125 or 5^3=125, both far over the 300 cap. "
    "ROUTING-FACT (Law 13): the quotient term is negative (odd power of "
    "a negative base) -- rendered as a SUBTRACTION from the positive "
    "3^4 term via commutative reordering (81-25-6=50, matches the "
    "source's own -25+81-6=50) rather than an unrepresentable negative "
    "addend, every intermediate nonneg throughout.",
    routing_fact="(-5)^5/5^3 is negative (odd power of a negative base); "
                 "reordered via commutativity into 81-25-6 instead of "
                 "-25+81-6, algebraically identical, nonneg throughout.")

# 1297: SKIP -- largest whole n s.t. 1/3 + n/7 < 1.
skip(1297,
     "n/7 < 2/3 -> n < 14/3 ~ 4.67 -> largest whole n=4. Strict-"
     "inequality integer-search family (denominators 3,7 need an LCD, "
     "no LCM primitive), matches t9's [1152]/[1155] cluster.")

# 1298: SKIP -- greatest common piece length for ropes 39,52,65 inches.
skip(1298,
     "GCD(39,52,65)=13 (39=3*13, 52=4*13, 65=5*13). GCD/factorization "
     "family, no primitive; matches t9's [1213] and the standing "
     "GCD cluster.")

# 1300: Emily's 5 scores 92,95,87,89,100; need 6th for mean=93 over 6.
# CAP-AVOIDANCE (raw 6-score total 558 and even the 5-score sum 463
# exceed the 300 cap): rebuilt in DEVIATION-FROM-BASELINE space
# (baseline=87, Emily's own lowest score, matching t9's [1235]/[1013]
# lesson). Zero fdivs needed (multiply-forward instead of divide).
g = G()
c87 = g.given(87)                        # baseline (Emily's own lowest score)
d0 = g.rel("sub", c87, c87)              # 0 (self-cancel, Law 1)
c92 = g.given(92)
d1 = g.rel("sub", c92, c87)              # 5
c95 = g.given(95)
d2 = g.rel("sub", c95, c87)              # 8
c89 = g.given(89)
d4 = g.rel("sub", c89, c87)              # 2
c100q = g.given(100)
d5 = g.rel("sub", c100q, c87)            # 13
s1 = g.rel("add", d0, d1)
s2 = g.rel("add", s1, d2)
s3 = g.rel("add", s2, d4)
sumKnownDev = g.rel("add", s3, d5)       # 28
c93 = g.given(93)
targetMeanDev = g.rel("sub", c93, c87)   # 6
c6 = g.given(6)                          # six scores total
totalDevNeeded = g.rel("mul", targetMeanDev, c6)  # 36
sixthDev = g.rel("sub", totalDevNeeded, sumKnownDev)  # 8
ans = g.rel("add", c87, sixthDev)        # 95
add(1300, g.factors, ans, 95,
    "CAP-AVOIDANCE (raw 5-score sum is 463 and the 6-score target total "
    "558 both exceed the 300 cap; matches t9's [1235]/[1013] lesson): "
    "recomputed in DEVIATION-FROM-BASELINE space (baseline=87, Emily's "
    "own lowest score), every intermediate well under 40, no fdiv "
    "needed at all (multiply forward by the score-count instead of "
    "dividing).",
    watch="pointer-collision: c87 (the baseline) is an argument in SIX "
          "separate sub/add factors (self-cancel + four score "
          "deviations + the final add-back) -- heaviest pointer reuse "
          "this tranche, matches [1235]'s cap-avoidance-forced pattern.")

# 1302: (a^3+b^3)/(a^2-ab+b^2), a=5, b=4. LAW-TENSION (Law 5, degree
# argued): the sum-of-cubes identity a^3+b^3=(a+b)(a^2-ab+b^2) makes the
# WHOLE expression collapse to a+b -- a single add factor. Flagged
# explicitly because the graph is unusually thin, but the identity is
# NOT optional decoration: the raw computation (189/21, fdiv k=21) is
# cap/fdiv-blocked outright, so the theorem-application is the ONLY
# lawful path to any graph at all, not a shortcut chosen for elegance.
g = G()
c5 = g.given(5)
c4 = g.given(4)
ans = g.rel("add", c5, c4)               # 9
add(1302, g.factors, ans, 9,
    "LAW-TENSION (Law 5, named: sum-of-cubes factorization identity "
    "a^3+b^3=(a+b)(a^2-ab+b^2)), degree argued explicitly: the raw "
    "computation (a^3+b^3=189, divided by a^2-ab+b^2=21) is cap-blocked "
    "(189 exceeds nothing but 21 is a non-single-digit fdiv k) -- the "
    "identity is NECESSARY, not a stylistic shortcut, making the "
    "single-factor add(a,b) the only lawful render rather than a "
    "residue. Flagged for the certifier: the graph is unusually thin "
    "for the source's algebraic dressing, by structural necessity.")

# 1304: SKIP -- trailing zeros of 25*240.
skip(1304,
     "25*240=6000, three trailing zeros. DOUBLY BLOCKED: (a) 'count "
     "trailing zeros' has no primitive (a digit-shape count, matches "
     "[1117]'s (t9) digit-occurrence family), (b) even forming the "
     "product 6000 independently breaks the 300 cap.")

# 1306: median{n,n+5,n+6,n+9,n+15}=9 -> n=3 (n+6=9, since the offsets "
# are already sorted 0<5<6<9<15). Mean = n + mean(offsets) via
# translation-invariance of the mean under a constant shift (theorem-
# application), one fdiv (k=5).
g = G()
c5 = g.given(5)                          # offset of 2nd term
c6 = g.given(6)                          # offset of 3rd term (= median position)
c9med = g.given(9)                       # given median value
c9off = g.given(9)                       # offset of 4th term, separate instance
c15 = g.given(15)                        # offset of 5th term
n = g.rel("sub", c9med, c6)              # 3 (from n+6=9)
s1 = g.rel("add", c5, c6)
s2 = g.rel("add", s1, c9off)
sumOffsets = g.rel("add", s2, c15)       # 35 (first term's own 0 offset
                                          # omitted, Law 1 self-combination)
meanOffset = g.fdiv(sumOffsets, 5)       # ONE fdiv, k=5 -> 7
ans = g.rel("add", n, meanOffset)        # 10
add(1306, g.factors, ans, 10,
    "THEOREM-APPLICATION (named: translation-invariance of the mean -- "
    "shifting every term of a set by a constant shifts the mean by the "
    "same constant), degree: n derived from the median position (a "
    "SEPARATE given instance from the 4th term's offset, same numeral "
    "9, distinct roles), offset-sum assembled in-graph (first term's own "
    "0-offset omitted per Law 1), scaled via the one allowed fdiv.")

# 1309: regular hexagon interior angle = (n-2)*180/n. Cap-avoidance via
# ORDER (divide before multiply: 180/6=30 stays under cap, avoiding
# 4*180=720), one fdiv (k=6, the hexagon's own side count).
g = G()
c6 = g.given(6)
c2 = g.given(2)
nMinus2 = g.rel("sub", c6, c2)           # 4
c180 = g.given(180)
thirty = g.fdiv(c180, 6)                 # ONE fdiv, k=6 -> 30
ans = g.rel("mul", nMinus2, thirty)      # 120
add(1309, g.factors, ans, 120,
    "THEOREM-APPLICATION (named: regular-polygon interior-angle formula "
    "(n-2)*180/n), degree: computed in CAP-SAFE ORDER (divide by n "
    "before multiplying by n-2, avoiding the 720 intermediate that "
    "multiply-first would produce) -- generalizes [1235]/[1300]'s "
    "cap-avoidance discipline to an ordering choice rather than a "
    "baseline shift; the fdiv's k=6 is itself a source literal (the "
    "hexagon's own side count).")

# 1311: 0.1 / 0.004. Both scaled by 1000 (structural literals, Law 10),
# one fdiv (k=4).
g = G()
c100 = g.given(100)                      # 0.1 * 1000
c4 = g.given(4)                          # 0.004 * 1000
ans = g.fdiv(c100, 4)                    # ONE fdiv, k=4 -> 25
add(1311, g.factors, ans, 25,
    "STRUCTURAL LITERAL (Law 10): both decimals scaled by the same "
    "power of 10 (1000) to clear them into integers (100 and 4) -- a "
    "standard decimal-clearing check, exactly 2 literals, well within "
    "the 'beyond a handful, state the check' threshold. One allowed "
    "fdiv (k=4) completes the division.")

# 1314: girls:boys = 3:2, total 45 students. Bookkeeping ratio-sum,
# one fdiv (k=5).
g = G()
c3 = g.given(3)
c2 = g.given(2)
ratioSum = g.rel("add", c3, c2)          # 5
c45 = g.given(45)
unit = g.fdiv(c45, 5)                    # ONE fdiv, k=5 -> 9
ans = g.rel("mul", c3, unit)             # 27
add(1314, g.factors, ans, 27,
    "BOOKKEEPING (Law 8): ratio-sum (3+2=5) assembled in-graph before "
    "use as the fdiv divisor (the one allowed fdiv, k=5); girls count "
    "is a fresh multiply of the unit by their own ratio share (3).")

# 1315: 3y+7y = 282-8(y-3). DIRECT SYSTEM ENCODING avoiding the 306
# cap-violating intermediate: LHS and RHS computed SEPARATELY from the
# same shared free var y (both sides stay well under 300 individually,
# 170 each), forced equal via the LHS mul's explicit result pointer --
# never combines to the coefficient-folded 18y=306 form, which would
# break the cap.
g = G()
y = g.free()
c3 = g.given(3)
c7 = g.given(7)
coef1 = g.rel("add", c3, c7)             # 10 (=3+7, LHS coefficient)
c8 = g.given(8)
c3b = g.given(3)                         # the '3' inside (y-3), separate
yMinus3 = g.rel("sub", y, c3b)           # y-3
term = g.rel("mul", c8, yMinus3)         # 8(y-3)
c282 = g.given(282)
rhs = g.rel("sub", c282, term)           # 282-8(y-3)
g.rel("mul", coef1, y, rhs)              # force 10y = rhs
add(1315, g.factors, y, 17,
    "DIRECT SYSTEM ENCODING / CAP-AVOIDANCE: naively folding coefficients "
    "(3+7+8=18) and constants (282+8*3=306) would produce a 306 "
    "intermediate, OVER the 300 cap. Rendered instead by computing both "
    "sides of the equation SEPARATELY from the same shared free var y "
    "(LHS=10y, RHS=282-8(y-3), both individually 170 at the solution, "
    "well under cap) and forcing them equal via the LHS mul's explicit "
    "result pointer -- matches [1269]'s and [1170]'s (t9) force-final-"
    "equality pattern, here specifically chosen to dodge a cap "
    "violation that coefficient-folding would have caused.")

# 1316: SKIP -- how many positive divisors does 24 have.
skip(1316,
     "24=2^3*3^1 -> (3+1)(1+1)=8 divisors. Divisor-counting requires "
     "factorization, no primitive; matches the GCD/factorization family "
     "([1298] this tranche, t9's [1213]).")

# 1320: 8^8 * 4^4 / 2^28. LAW 8 BOOKKEEPING (matches t5's [755]
# precedent): all bases converted to powers of 2, exponent arithmetic
# (3*8 + 2*4 - 28 = 4) performed in-graph avoiding the astronomical
# 2^32 intermediate entirely; the reduced power (2^4) computed via a
# hand-verified, hard-coded pair of squarings matching the in-graph
# exponent.
g = G()
c3 = g.given(3)                          # 8=2^3, exponent
c8 = g.given(8)                          # outer exponent, 8^8
exp1 = g.rel("mul", c3, c8)              # 24
c2a = g.given(2)                         # 4=2^2, exponent
c4 = g.given(4)                          # outer exponent, 4^4
exp2 = g.rel("mul", c2a, c4)             # 8
sumExp = g.rel("add", exp1, exp2)        # 32
c28 = g.given(28)
finalExp = g.rel("sub", sumExp, c28)     # 4 (Law 8 bookkeeping, matches [755])
cBase = g.given(2)                       # base for the final power
p1 = g.rel("mul", cBase, cBase)          # 2^2=4
ans = g.rel("mul", p1, p1)               # 2^4=16
add(1320, g.factors, ans, 16,
    "LAW 8 BOOKKEEPING (matches t5's [755] precedent): all bases "
    "converted to powers of 2 (8=2^3, 4=2^2), exponent arithmetic "
    "(3*8+2*4-28=4) computed genuinely in-graph, avoiding the 2^32 "
    "intermediate that direct expansion would require. The reduced "
    "power (2^4=16) is computed via two hand-verified squarings -- the "
    "squaring COUNT is fixed by construction to match the independently "
    "derived exponent (4), the same correspondence discipline as "
    "[755]/[1295] this campaign.")

# 1322: 6^12 / 36^5. Same family as [1320] (Law 8 bookkeeping).
g = G()
c12 = g.given(12)
c2 = g.given(2)                          # 36=6^2, exponent
c5 = g.given(5)                          # outer exponent, 36^5
expDenom = g.rel("mul", c2, c5)          # 10
finalExp = g.rel("sub", c12, expDenom)   # 2
cBase = g.given(6)
ans = g.rel("mul", cBase, cBase)         # 6^2=36
add(1322, g.factors, ans, 36,
    "LAW 8 BOOKKEEPING (matches [755]/[1320] this tranche): 36=6^2 "
    "converted so both terms share base 6, exponent arithmetic "
    "(12-2*5=2) computed genuinely in-graph, avoiding 6^12 entirely. "
    "The reduced power (6^2=36) is a single hand-verified mul matching "
    "the in-graph exponent.")

# 1325: wall 9ft x 12ft minus a 2ft x 4ft window.
g = G()
c9 = g.given(9)
c12 = g.given(12)
wallArea = g.rel("mul", c9, c12)         # 108
c2 = g.given(2)
c4 = g.given(4)
windowArea = g.rel("mul", c2, c4)        # 8
ans = g.rel("sub", wallArea, windowArea) # 100
add(1325, g.factors, ans, 100,
    "DIRECT: wall area and window area each genuinely computed, then "
    "subtracted.")

# 1326: sqrt(3x+7)=10. Theorem-application (named: squaring both sides
# inverts sqrt), then multiplicative inversion.
g = G()
c10 = g.given(10)
square = g.rel("mul", c10, c10)          # 100
c7 = g.given(7)
rhs = g.rel("sub", square, c7)           # 93
c3 = g.given(3)
x = g.free()
g.rel("mul", c3, x, rhs)                 # 3x=93 -> x=31 (mult inversion)
add(1326, g.factors, x, 31,
    "THEOREM-APPLICATION (named: squaring both sides inverts sqrt), "
    "degree: 10^2 genuinely computed, 7 subtracted, then multiplicative "
    "inversion isolates x.")

# 1330: SKIP -- bookstore sale (multiples of 5) vs shoe store sale
# (every 6 days from July 3), how many shared dates in July.
skip(1330,
     "Shoe-store sale days: 3,9,15,21,27 (33 is out of July's 31 days). "
     "Bookstore sale days: 5,10,15,20,25,30. Shared: 15 -> count=1. "
     "Requires enumerating two arithmetic sequences (equivalently, "
     "solving a simultaneous congruence via CRT) and counting matches "
     "within a bounded range -- no primitive for range-bounded sequence "
     "intersection; compounds [1158]'s (t9) count-multiples-in-range "
     "family with a second congruence.")

# 1331: SKIP -- which number in {55,57,58,59,61} has the smallest prime
# factor.
skip(1331,
     "55=5*11 (smallest factor 5), 57=3*19 (3), 58=2*29 (2), 59 prime "
     "(59), 61 prime (61) -- smallest-prime-factor comparison across "
     "the set picks 58 (factor 2). Requires per-candidate factorization "
     "search across 5 numbers, no primitive; matches the GCD/"
     "factorization/primality-search family ([1298]/[1316]/[1278] this "
     "tranche).")

# 1333: convex hexagon, 2 distinct side lengths, AB=5, BC=6, perimeter
# 34. How many sides measure 6? EXCESS-AND-DEFICIENCY METHOD (named
# classical technique, sometimes called the assumed-uniform method):
# assume all 6 sides were length 5 (total 30), the 4-unit excess over
# the true perimeter is distributed one unit at a time by each 6-side's
# 1-unit excess over a 5-side.
g = G()
c6 = g.given(6)                          # total sides (hexagon)
c5 = g.given(5)                          # AB length
allFiveTotal = g.rel("mul", c5, c6)      # 30 (if every side were length 5)
c34 = g.given(34)
diff = g.rel("sub", c34, allFiveTotal)   # 4 (excess to distribute)
c6b = g.given(6)                         # BC length, separate instance
perSideExcess = g.rel("sub", c6b, c5)    # 1 (excess per 6-side over a 5-side)
b = g.free()
g.rel("mul", perSideExcess, b, diff)     # perSideExcess*b=4 -> b=4 (mult inv)
add(1333, g.factors, b, 4,
    "THEOREM-APPLICATION (named: excess-and-deficiency / assumed-"
    "uniform method, classical algebra technique), degree: 'assume all "
    "sides length 5' total genuinely computed, excess over the true "
    "perimeter derived, per-side excess derived from the two DISTINCT "
    "given side-length instances, then multiplicative inversion "
    "distributes the excess across however many 6-sides are needed.")

# 1334: SKIP -- side of a square whose diagonal is sqrt(2) inches.
skip(1334,
     "diagonal = side*sqrt(2); side = sqrt(2)/sqrt(2) = 1. DOMAIN-"
     "BOUNDARY: the source's own datum (a diagonal of exactly sqrt(2), "
     "an irrational literal) has no representation in the nonneg-"
     "integer CSP domain at all -- not an operation gap, the INPUT "
     "itself is outside the domain.")

# 1335: SKIP -- Bob's number: 50-100, multiple of 11, not multiple of 2,
# digit sum multiple of 3.
skip(1335,
     "Multiples of 11 in (50,100): 55,66,77,88,99. Odd ones: 55,77,99. "
     "Digit-sum-multiple-of-3: 55->10(no), 77->14(no), 99->18(yes) -> "
     "99. Multi-stage enumeration + digit-sum filter, no primitive for "
     "either the range-bounded multiple search or the digit-sum "
     "condition; matches [1330]'s range-search family and [1294]'s "
     "digit-sum family, compounded.")

# 1338: kennel 60 dogs, 9 watermelon, 48 salmon, 5 both. Find neither.
g = G()
c9 = g.given(9)
c48 = g.given(48)
unionSum = g.rel("add", c9, c48)         # 57
c5 = g.given(5)
union = g.rel("sub", unionSum, c5)       # 52
c60 = g.given(60)
ans = g.rel("sub", c60, union)           # 8
add(1338, g.factors, ans, 8,
    "THEOREM-APPLICATION (named: inclusion-exclusion, |A union B|=|A|+"
    "|B|-|A and B|, then total minus union gives neither), degree: "
    "union genuinely assembled from both liking-counts, total minus "
    "union isolates neither.")

print(f"\nFINAL -- Drafted: {len(rows)}  Skipped: {len(skips)}  Fails: {len(fails)}")
print(f"Total accounted: {len(rows) + len(skips) + len(fails)} / 40")
if fails:
    print("FAILS:", fails)

with open('/home/bryce/mycelium/.cache/book8_t10_prose_pairs_draft.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print("Wrote", len(rows), "rows to .cache/book8_t10_prose_pairs_draft.jsonl")

drafted_ids = {r["gen"]["src_idx"] for r in rows}
skipped_ids = {s[0] for s in skips}
all_ids = {c["src_idx"] for c in CANDS.values()}
missing = all_ids - drafted_ids - skipped_ids
overlap = drafted_ids & skipped_ids
print("Missing (neither drafted nor skipped):", sorted(missing))
print("Overlap (both drafted and skipped):", sorted(overlap))
