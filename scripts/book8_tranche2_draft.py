import json, sys, os
sys.path.insert(0, '/home/bryce/mycelium')
sys.path.insert(0, '/home/bryce/mycelium/scripts')
from tta_alg2_dials import solve2
from mycelium.csp_domains import problem_from_algebra3
from mycelium.csp_core import solve_symbolic

SMP = {"n_vars": 24, "m": 300}

CANDS = {c["src_idx"]: c for c in
         json.load(open('/home/bryce/mycelium/.cache/book8_candidates_t2.json'))["tranche2"]}

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

def build_row(src_idx, text, factors, query_var, dialect, notes, watch=None, n_vars=24, m=300):
    sol = full_solution(factors, n_vars, m)
    assert sol is not None, f"src {src_idx}: solver failed to find full solution"
    gen = {
        "src_idx": src_idx, "book": 8, "tranche": 2, "floor": "prime", "fs": True,
        "dialect": dialect, "gate": "PENDING:5view-vote+key", "generation": "21",
        "notes": notes,
    }
    if watch:
        gen["watch"] = watch
    # text MUST be source-verbatim; pull directly from the candidates file rather
    # than trust a hand-transcribed literal (whitespace/newlines inside LaTeX
    # \begin{cases} blocks are easy to silently collapse when typing by hand).
    src_text = CANDS[src_idx]["problem"]
    return {
        "text": src_text, "factors": factors, "query_var": query_var,
        "n_vars": n_vars, "m": m, "decisions": [], "mentions": [],
        "solution": sol, "gen": gen,
    }

rows = []
fails = []

def add(src_idx, text, factors, query_var, expect, dialect, notes, watch=None):
    ok = check(src_idx, factors, query_var, expect)
    if not ok:
        fails.append(src_idx)
        return
    rows.append(build_row(src_idx, text, factors, query_var, dialect, notes, watch))

# ---------------------------------------------------------------------------
# 373: x+y=6, x^2-y^2=12 -> x-y
add(373,
    "If $x+y = 6$ and $x^2-y^2 = 12$, then what is $x-y$?",
    [
        {"ftype": "given", "var": 0, "value": 12},
        {"ftype": "fdiv", "var": 0, "k": 6, "result": 1},
    ], 1, 2,
    "Consider the numbers a, b. a is 12. When a is divided by 6, the quotient is b. What is b?",
    "algebraic identity x^2-y^2=(x+y)(x-y): a=x^2-y^2=12 (given), divided by the source's other "
    "given x+y=6 (rendered as bare fdiv k, not a separate var, matching Worked Example A's bare-"
    "coefficient precedent) yields x-y directly.")

# 375: three consecutive even integers, first+third=128 -> sum
add(375,
    "What is the sum of three consecutive even integers if the sum of the first and third "
    "integers is $128$?",
    [
        {"ftype": "given", "var": 0, "value": 128},
        {"ftype": "fdiv", "var": 0, "k": 2, "result": 1},
        {"ftype": "given", "var": 2, "value": 3},
        {"ftype": "rel", "op": "mul", "args": [1, 2], "result": 3},
    ], 3, 192,
    "Consider the numbers a, b, c, d. a is 128. When a is divided by 2, the quotient is b. "
    "c is 3. b times c equals d. What is d?",
    "arithmetic-sequence identity: middle term = (first+third)/2 = 64; sum of 3 terms = "
    "3*middle = 192. Count 3 is the stated cardinality (\"three consecutive even integers\").")

# 376: g(x)=2x^2-3, h(x)=4x^3+1, g(h(-1))
add(376,
    "If $g(x) = 2x^2 - 3$ and $h(x) = 4x^3 +1$, what is the value of $g(h(-1))$?",
    [
        {"ftype": "given", "var": 0, "value": 1},              # |x|=1 (x=-1)
        {"ftype": "rel", "op": "mul", "args": [0, 0], "result": 1},   # x^2 mag =1
        {"ftype": "rel", "op": "mul", "args": [1, 0], "result": 2},   # x^3 mag =1
        {"ftype": "given", "var": 3, "value": 4},               # coeff of x^3 in h
        {"ftype": "rel", "op": "mul", "args": [3, 2], "result": 4},   # |4x^3| = 4
        {"ftype": "given", "var": 5, "value": 1},               # constant +1 in h
        {"ftype": "rel", "op": "sub", "args": [4, 5], "result": 6},   # h(-1) magnitude = 3 (neg)
        {"ftype": "rel", "op": "mul", "args": [6, 6], "result": 7},   # y^2 = 9
        {"ftype": "given", "var": 8, "value": 2},               # coeff of y^2 in g
        {"ftype": "rel", "op": "mul", "args": [8, 7], "result": 9},   # 2*9=18
        {"ftype": "given", "var": 10, "value": 3},              # constant -3 in g
        {"ftype": "rel", "op": "sub", "args": [9, 10], "result": 11},  # 18-3=15
    ], 11, 15,
    "Consider the numbers a, b, c, d, e, f, g, h, i, j, k, l. a is 1. a times a equals b. "
    "b times a equals c. d is 4. d times c equals e. f is 1. e exceeds f by g. g times g "
    "equals h. i is 2. i times h equals j. k is 3. j exceeds k by l. What is l?",
    "magnitude-fold for the negative intermediate h(-1)=-3 (Worked Example B / technique 3): "
    "a=|x|=1 cubed via two chained muls (b=x^2 mag, c=x^3 mag, both 1 since |-1|=1); e=|4x^3|=4; "
    "'e exceeds f by g' captures 4x^3+1 = -4+1 = -3 (magnitude 3, negative, sign tracked here "
    "not in-graph); the negative sign is irrelevant from g onward since g(y) squares y first.")

# 382: quotient of two positive integers 5/2, product 160 -> larger
add(382,
    "The quotient of two positive integers is $\\frac{5}{2}$ and their product is 160. What "
    "is the value of the larger of the two integers?",
    [
        {"ftype": "given", "var": 0, "value": 5},
        {"ftype": "given", "var": 1, "value": 2},
        {"ftype": "given", "var": 2, "value": 160},
        {"ftype": "rel", "op": "mul", "args": [0, 3], "result": 4},   # a*t = p
        {"ftype": "rel", "op": "mul", "args": [1, 3], "result": 5},   # b*t = q
        {"ftype": "rel", "op": "mul", "args": [4, 5], "result": 2},   # p*q = 160
    ], 4, 20,
    "Consider the numbers a, b, c, d, e, f. a is 5. b is 2. c is 160. a times d equals e. "
    "b times d equals f. e times f equals c. What is e?",
    "direct system encoding (technique 2): free var d (t) shared between two multiplications "
    "carrying the 5:2 ratio; the CSP search finds the unique t consistent with e*f=160. Since "
    "a=5>b=2, e=p is always the larger factor. Note: dialect var d is a free (non-given) letter, "
    "consecutive with a-f; letter position still equals var index (d=var index 3).")

# 386: 1+3+...+17
add(386,
    "Calculate the sum $1 + 3 + 5 + \\cdots + 15 + 17$.",
    [
        {"ftype": "given", "var": 0, "value": 17},
        {"ftype": "given", "var": 1, "value": 1},
        {"ftype": "rel", "op": "sub", "args": [0, 1], "result": 2},
        {"ftype": "fdiv", "var": 2, "k": 2, "result": 3},
        {"ftype": "rel", "op": "add", "args": [3, 1], "result": 4},
        {"ftype": "rel", "op": "mul", "args": [4, 4], "result": 5},
    ], 5, 81,
    "Consider the numbers a, b, c, d, e, f. a is 17. b is 1. a exceeds b by c. When c is "
    "divided by 2, the quotient is d. d plus b equals e. e times e equals f. What is f?",
    "sum of first n odd numbers = n^2. n = (17-1)/2+1 = 9 (count of terms), then n^2=81 via "
    "duplicate-arg square.")

# 390: cubic feet in 3 cubic yards
add(390,
    "How many cubic feet are in three cubic yards?",
    [
        {"ftype": "given", "var": 0, "value": 3},   # feet per yard (universal constant)
        {"ftype": "rel", "op": "mul", "args": [0, 0], "result": 1},
        {"ftype": "rel", "op": "mul", "args": [1, 0], "result": 2},   # cubic feet per cubic yard
        {"ftype": "given", "var": 3, "value": 3},   # number of cubic yards
        {"ftype": "rel", "op": "mul", "args": [2, 3], "result": 4},
    ], 4, 81,
    "Consider the numbers a, b, c, d, e. a is 3. a times a equals b. b times a equals c. "
    "d is 3. c times d equals e. What is e?",
    "universal constant (3 feet/yard) cubed in-graph (a*a=b, b*a=c=27, cubic feet per cubic "
    "yard) then multiplied by the given 3 cubic yards.")

# 398: piecewise p(x,y); p(p(1,-1), p(-5,-2))
add(398,
    "Let \\[p(x,y) = \\begin{cases} x + y &\\quad \\text{if } x \\ge 0 \\text{ and } y \\ge 0, "
    "\\\\ x - 2y &\\quad \\text{if } x < 0 \\text{ and } y < 0, \\\\ 3x + y &\\quad "
    "\\text{otherwise}. \\end{cases} \\]What is $p(p(1,-1),p(-5,-2))$?",
    [
        {"ftype": "given", "var": 0, "value": 1},    # x=1 for p(1,-1)
        {"ftype": "given", "var": 1, "value": 1},    # |y|=1 for p(1,-1)
        {"ftype": "given", "var": 2, "value": 3},    # coeff "3" in otherwise-branch
        {"ftype": "rel", "op": "mul", "args": [2, 0], "result": 3},   # 3x = 3
        {"ftype": "rel", "op": "sub", "args": [3, 1], "result": 4},   # p(1,-1) = 2
        {"ftype": "given", "var": 5, "value": 5},    # |x|=5 for p(-5,-2)
        {"ftype": "given", "var": 6, "value": 2},    # |y|=2 for p(-5,-2)
        {"ftype": "given", "var": 7, "value": 2},    # coeff "2" in x-2y branch
        {"ftype": "rel", "op": "mul", "args": [7, 6], "result": 8},   # |2y| = 4
        {"ftype": "rel", "op": "sub", "args": [5, 8], "result": 9},   # p(-5,-2) mag = 1 (neg)
        {"ftype": "rel", "op": "mul", "args": [2, 4], "result": 10},  # 3 * p(1,-1) = 6 (reuse coeff 3)
        {"ftype": "rel", "op": "sub", "args": [10, 9], "result": 11},  # 6 - 1 = 5
    ], 11, 5,
    "Consider the numbers a, b, c, d, e, f, g, h, i, j, k, l. a is 1. b is 1. c is 3. "
    "c times a equals d. d exceeds b by e. f is 5. g is 2. h is 2. h times g equals i. "
    "f exceeds i by j. c times e equals k. k exceeds j by l. What is l?",
    "case selection by external inspection (matches book7's [339] precedent): p(1,-1) uses the "
    "otherwise-branch (x>=0,y<0); p(-5,-2) uses the x<0,y<0 branch; p(e,j) again otherwise-"
    "branch (e=2>=0, j represents -1<0). Negative y-values rendered as magnitudes with sign "
    "tracked externally (Worked Example B / technique 3); coefficient 3 (var c) reused across "
    "both applications of the otherwise-branch formula 3x+y, legitimate since same source "
    "formula constant.")

# 400: two regular polygons, same perimeter, 38 sides vs side length 2x -> other's sides
add(400,
    "Two regular polygons have the same perimeter. If the first has 38 sides and a side "
    "length twice as long as the second, how many sides does the second have?",
    [
        {"ftype": "given", "var": 0, "value": 38},
        {"ftype": "given", "var": 1, "value": 2},
        {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2},
    ], 2, 76,
    "Consider the numbers a, b, c. a is 38. b is 2. a times b equals c. What is c?",
    "perimeter equality 38*(2s) = n2*s cancels s entirely, leaving n2 = 38*2 = 76 regardless "
    "of side length s.")

# 401: #N = 0.5N+1, compute #(#(#50))
add(401,
    "Define $\\#N$ by the formula $\\#N = .5(N) + 1$. Calculate $\\#(\\#(\\#50))$.",
    [
        {"ftype": "given", "var": 0, "value": 50},
        {"ftype": "given", "var": 1, "value": 2},
        {"ftype": "given", "var": 2, "value": 1},
        {"ftype": "rel", "op": "mul", "args": [1, 3], "result": 0},   # 2*half1 = 50
        {"ftype": "rel", "op": "add", "args": [3, 2], "result": 4},   # half1 + 1 = #50
        {"ftype": "rel", "op": "mul", "args": [1, 5], "result": 4},   # 2*half2 = #50
        {"ftype": "rel", "op": "add", "args": [5, 2], "result": 6},   # half2 + 1 = #(#50)
        {"ftype": "rel", "op": "mul", "args": [1, 7], "result": 6},   # 2*half3 = #(#50)
        {"ftype": "rel", "op": "add", "args": [7, 2], "result": 8},   # half3 + 1 = #(#(#50))
    ], 8, 8,
    "Consider the numbers a, b, c, d, e, f, g, h, i. a is 50. b is 2. c is 1. b times d "
    "equals a. d plus c equals e. b times f equals e. f plus c equals g. b times h equals "
    "g. h plus c equals i. What is i?",
    "multiplicative inversion (Worked Example C) used three times to halve without fdiv: "
    "'b times d equals a' lets the solver find d=a/2 by search rather than dividing; reused "
    "constants b=2 and c=1 across all three applications of #N. Zero fdiv factors used.")

# 402: f(x) = (5x+1)/(x-1), f(7)
add(402,
    "If $f(x)=\\dfrac{5x+1}{x-1}$, find the value of $f(7)$.",
    [
        {"ftype": "given", "var": 0, "value": 5},
        {"ftype": "given", "var": 1, "value": 7},
        {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2},
        {"ftype": "given", "var": 3, "value": 1},
        {"ftype": "rel", "op": "add", "args": [2, 3], "result": 4},
        {"ftype": "rel", "op": "sub", "args": [1, 3], "result": 5},
        {"ftype": "rel", "op": "mul", "args": [5, 6], "result": 4},
    ], 6, 6,
    "Consider the numbers a, b, c, d, e, f, g. a is 5. b is 7. a times b equals c. d is 1. "
    "c plus d equals e. b exceeds d by f. f times g equals e. What is g?",
    "denominator (x-1) is a DERIVED value, so per axis-1 division-of-derived-value rule, "
    "rendered as multiplication with derived args (f*g=e) letting the solver invert, not fdiv "
    "(Worked Example C).")

# 403: 1/5+5/x=12/x+1/12 -> x
add(403,
    "In the equation, $\\frac15+\\frac{5}{x}=\\frac{12}{x}+\\frac{1}{12}$, what is the value "
    "of $x$?",
    [
        {"ftype": "given", "var": 0, "value": 5},
        {"ftype": "given", "var": 1, "value": 12},
        {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2},
    ], 2, 60,
    "Consider the numbers a, b, c. a is 5. b is 12. a times b equals c. What is c?",
    "REARRANGEMENT: 1/5+5/x=12/x+1/12 -> 5/x-12/x=1/12-1/5 -> (5-12)/x=(5-12)/60, and since "
    "the x-term numerators (5,12) exactly equal the constant-term denominators (5,12) in this "
    "instance, the (5-12) factor cancels identically on both sides, leaving x = 5*12 = 60. "
    "Verified directly: 1/5+5/60 = 17/60 = 12/60+1/12. Flagged per the rearrangement law as a "
    "nontrivial fold -- this is a genuine identity for THIS equation's specific coefficients, "
    "not a general shortcut.")

# 410: pebbles, day k -> k pebbles, sum 1..12
add(410,
    "Murtha has decided to start a pebble collection. She collects one pebble the first day "
    "and two pebbles on the second day. On each subsequent day she collects one more pebble "
    "than the previous day. How many pebbles will she have collected at the end of the "
    "twelfth day?",
    [
        {"ftype": "given", "var": 0, "value": 12},
        {"ftype": "given", "var": 1, "value": 1},
        {"ftype": "rel", "op": "add", "args": [0, 1], "result": 2},
        {"ftype": "rel", "op": "mul", "args": [2, 0], "result": 3},
        {"ftype": "fdiv", "var": 3, "k": 2, "result": 4},
    ], 4, 78,
    "Consider the numbers a, b, c, d, e. a is 12. b is 1. a plus b equals c. c times a "
    "equals d. When d is divided by 2, the quotient is e. What is e?",
    "day k collects k pebbles (stated: day1=1, day2=2, +1/day); sum 1..12 via n*(n+1)/2 "
    "using the stated ordinal (\"twelfth day\") as the count n.")

# 411: Sally shots percent (PCT primitive, first exposure)
add(411,
    "After Sally takes 20 shots, she has made $55\\%$ of her shots. After she takes 5 more "
    "shots, she raises her percentage to $56\\%$. How many of the last 5 shots did she make?",
    [
        {"ftype": "given", "var": 0, "value": 20, "role": "pct_base"},
        {"ftype": "pct", "args": [1, 0], "p": 55},
        {"ftype": "given", "var": 2, "value": 5},
        {"ftype": "rel", "op": "add", "args": [0, 2], "result": 3},
        {"ftype": "pct", "args": [4, 3], "p": 56},
        {"ftype": "rel", "op": "sub", "args": [4, 1], "result": 5},
    ], 5, 3,
    "Consider the numbers a, b, c, d, e, f. a is 20. b is 55 percent of a. c is 5. a plus c "
    "equals d. e is 56 percent of d. e exceeds b by f. What is f.",
    "RATIFIED pct primitive, first use this book: 'b is 55 percent of a' recovers shots made "
    "after 20 (b=11); total shots after +5 more is d=25; 'e is 56 percent of d' recovers total "
    "made (e=14); last-5-made = e-b = 3.",
    watch="pct")

# 413: nested radical sqrt(2-sqrt(2-...)) = x
def add413():
    factors = [
        {"ftype": "given", "var": 0, "value": 2},
        {"ftype": "rel", "op": "mul", "args": [1, 1], "result": 2},   # x*x = y
        {"ftype": "rel", "op": "add", "args": [2, 1], "result": 0},   # y + x = 2 (given)
    ]
    add(413,
        "Evaluate $\\sqrt{2 -\\!\\sqrt{2 - \\!\\sqrt{2 - \\!\\sqrt{2 - \\cdots}}}}$.",
        factors, 1, 1,
        "Consider the numbers a, b, c. a is 2. b times b equals c. c plus b equals a. What is b?",
        "direct system encoding (technique 2): x = sqrt(2-x) squares to x^2+x=2; free var b (x) "
        "is unconstrained except through these two relations; the CSP search finds the unique "
        "nonnegative solution b=1 (domains are nonnegative by construction, so the x=-2 root is "
        "excluded automatically).")
add413()

# 428: piecewise f(x)=9x+16 (x<2) or 2x-14 (x>=2); f(x)=-2, sum of all x
add(428,
    "Let \\[f(x) = \\begin{cases} 9x+16 &\\text{if }x<2, \\\\ 2x-14&\\text{if }x\\ge2. "
    "\\end{cases} \\]If $f(x)=-2$, find the sum of all possible values of $x$.",
    [
        {"ftype": "given", "var": 0, "value": 2},    # |RHS| = |-2|
        {"ftype": "given", "var": 1, "value": 16},   # const in case 1 (9x+16)
        {"ftype": "rel", "op": "add", "args": [0, 1], "result": 2},   # |9x1| = 18
        {"ftype": "given", "var": 3, "value": 9},    # coeff in case 1
        {"ftype": "rel", "op": "mul", "args": [3, 4], "result": 2},   # 9*|x1| = 18 -> |x1|=2 (neg)
        {"ftype": "given", "var": 5, "value": 14},   # const in case 2 (2x-14)
        {"ftype": "rel", "op": "sub", "args": [5, 0], "result": 6},   # |2x2| = 12
        {"ftype": "fdiv", "var": 6, "k": 2, "result": 7},             # x2 = 6 (pos)
        {"ftype": "rel", "op": "sub", "args": [7, 4], "result": 8},   # 6 - 2 = 4
    ], 8, 4,
    "Consider the numbers a, b, c, d, e, f, g, h, i. a is 2. b is 16. a plus b equals c. "
    "d is 9. d times e equals c. f is 14. f exceeds a by g. When g is divided by 2, the "
    "quotient is h. h exceeds e by i. What is i.",
    "case selection by external inspection (book7's [339] precedent: x<2 gives x1=-2, x>=2 "
    "gives x2=6, both verified valid for their branch). Magnitude-fold for the negative root "
    "x1 (technique 3): case1's 9x=-2-16=-18 rendered as add-then-mul-inversion (e is found by "
    "search from d*e=c); case2's 2x=-2+14=12 rendered via sub+fdiv (only fdiv in this item); "
    "final sum -2+6=4 via sub of magnitudes (opposite signs).")

# 433: a*b = a^2+2ab+b^2 = (a+b)^2, a=4,b=6
add(433,
    "Given $a \\star b = a^2 + 2ab + b^2$, what is the value of $a \\star b$ when $a = 4$ and "
    "$b = 6?$",
    [
        {"ftype": "given", "var": 0, "value": 4},
        {"ftype": "given", "var": 1, "value": 6},
        {"ftype": "rel", "op": "add", "args": [0, 1], "result": 2},
        {"ftype": "rel", "op": "mul", "args": [2, 2], "result": 3},
    ], 3, 100,
    "Consider the numbers a, b, c, d. a is 4. b is 6. a plus b equals c. c times c equals d. "
    "What is d.",
    "algebraic identity a^2+2ab+b^2=(a+b)^2, computed directly rather than expanded (matches "
    "tranche1's [333] precedent).")

# 436: telescoping fraction product
add(436,
    "Evaluate the expression \\[ \\frac{a+2}{a+1} \\cdot \\frac{b-1}{b-2} \\cdot \\frac{c + "
    "8}{c+6} , \\] given that $c = b-10$, $b = a+2$, $a = 4$, and none of the denominators "
    "are zero.",
    [
        {"ftype": "given", "var": 0, "value": 4},     # a
        {"ftype": "given", "var": 1, "value": 2},     # const "2"
        {"ftype": "rel", "op": "add", "args": [0, 1], "result": 2},   # b = a+2 = 6
        {"ftype": "given", "var": 3, "value": 10},    # const "10"
        {"ftype": "rel", "op": "sub", "args": [3, 2], "result": 4},   # |c| = 10-6 = 4 (c=-4)
        {"ftype": "given", "var": 5, "value": 1},     # const "1"
        {"ftype": "rel", "op": "add", "args": [0, 5], "result": 6},   # denom1 = a+1 = 5
        {"ftype": "rel", "op": "sub", "args": [2, 5], "result": 7},   # num2 = b-1 = 5
        {"ftype": "rel", "op": "sub", "args": [2, 1], "result": 8},   # denom2 = b-2 = 4
        {"ftype": "given", "var": 9, "value": 8},     # const "8"
        {"ftype": "rel", "op": "sub", "args": [9, 4], "result": 10},  # num3 = c+8 = 8-4 = 4
        {"ftype": "given", "var": 11, "value": 6},    # const "6" (c+6)
        {"ftype": "rel", "op": "sub", "args": [11, 4], "result": 12},  # denom3 = c+6 = 6-4 = 2
        {"ftype": "rel", "op": "mul", "args": [2, 7], "result": 13},   # num1*num2 = 6*5=30
        {"ftype": "rel", "op": "mul", "args": [13, 10], "result": 14},  # *num3(4) = 120
        {"ftype": "rel", "op": "mul", "args": [6, 8], "result": 15},   # denom1*denom2=5*4=20
        {"ftype": "rel", "op": "mul", "args": [15, 12], "result": 16},  # *denom3(2)=40
        {"ftype": "rel", "op": "mul", "args": [16, 17], "result": 14},  # 40 * x = 120 -> x=3
    ], 17, 3,
    "Consider the numbers a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r. a is 4. b is "
    "2. a plus b equals c. d is 10. d exceeds c by e. f is 1. a plus f equals g. c exceeds f "
    "by h. c exceeds b by i. j is 8. j exceeds e by k. l is 6. l exceeds e by m. c times h "
    "equals n. n times k equals o. g times i equals p. p times m equals q. q times r equals o. "
    "What is r.",
    "c = b-10 = -4 rendered via magnitude-fold (technique 3). All 6 product terms of the "
    "3-fraction product computed explicitly and multiplied out in full (numerator product 120, "
    "denominator product 40) rather than relying on the telescoping cancellation (denom1=num2=5, "
    "denom2=num3=4) that a human solver would use -- this avoids asserting an unproven-in-graph "
    "shortcut, at the cost of length (18 vars, near the upper end but under the 24 cap). Final "
    "division is a multiplicative inversion since the denominator product is derived. High "
    "certification-risk chain flagged per the cost guidance -- drafted anyway per the "
    "faithfulness mandate.")

# 439: 7 balls = 4 canoes; 3 canoes = 84 lbs -> 1 ball weight
add(439,
    "Seven identical bowling balls weigh the same as four identical canoes. If three of the "
    "canoes weigh a total of 84 pounds, how many pounds does one of the bowling balls weigh?",
    [
        {"ftype": "given", "var": 0, "value": 84},
        {"ftype": "fdiv", "var": 0, "k": 3, "result": 1},
        {"ftype": "given", "var": 2, "value": 4},
        {"ftype": "rel", "op": "mul", "args": [1, 2], "result": 3},
        {"ftype": "given", "var": 4, "value": 7},
        {"ftype": "rel", "op": "mul", "args": [5, 4], "result": 3},
    ], 5, 16,
    "Consider the numbers a, b, c, d, e, f. a is 84. When a is divided by 3, the quotient is "
    "b. c is 4. b times c equals d. e is 7. f times e equals d. What is f.",
    "one canoe's weight found via fdiv (k=3, single-digit); 4 canoes' weight = 7 balls' weight "
    "(the source's stated equivalence); ball weight found via multiplicative inversion since "
    "the target (4-canoe weight) is a derived value.")

# 446: 3x^y+4y^x, x=2,y=3
add(446,
    "Evaluate $3x^y + 4y^x$ when $x=2$ and $y=3$.",
    [
        {"ftype": "given", "var": 0, "value": 2},
        {"ftype": "rel", "op": "mul", "args": [0, 0], "result": 1},
        {"ftype": "rel", "op": "mul", "args": [1, 0], "result": 2},
        {"ftype": "given", "var": 3, "value": 3},
        {"ftype": "rel", "op": "mul", "args": [3, 3], "result": 4},
        {"ftype": "given", "var": 5, "value": 3},
        {"ftype": "rel", "op": "mul", "args": [5, 2], "result": 6},
        {"ftype": "given", "var": 7, "value": 4},
        {"ftype": "rel", "op": "mul", "args": [7, 4], "result": 8},
        {"ftype": "rel", "op": "add", "args": [6, 8], "result": 9},
    ], 9, 60,
    "Consider the numbers a, b, c, d, e, f, g, h, i, j. a is 2. a times a equals b. b times "
    "a equals c. d is 3. d times d equals e. f is 3. f times c equals g. h is 4. h times e "
    "equals i. g plus i equals j. What is j.",
    "fixed small exponents (x^3=8 via two chained muls, y^2=9 via one) computed directly, no "
    "search needed since exponents are known constants from the source.")

# 450: terms in (a+b+c)(d+e+f+g)
add(450,
    "How many terms are in the expansion of  \\[(a+b+c)(d+e+f+g)?\\]",
    [
        {"ftype": "given", "var": 0, "value": 3},
        {"ftype": "given", "var": 1, "value": 4},
        {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2},
    ], 2, 12,
    "Consider the numbers a, b, c. a is 3. b is 4. a times b equals c. What is c.",
    "distributive term-count identity: number of terms = (count in first factor) * (count in "
    "second factor).")

# 451: arithmetic seq 2,5,8,... 25th term
add(451,
    "What is the value of the 25th term of the arithmetic sequence $2, 5, 8, \\ldots$?",
    [
        {"ftype": "given", "var": 0, "value": 25},
        {"ftype": "given", "var": 1, "value": 1},
        {"ftype": "rel", "op": "sub", "args": [0, 1], "result": 2},
        {"ftype": "given", "var": 3, "value": 3},
        {"ftype": "rel", "op": "mul", "args": [2, 3], "result": 4},
        {"ftype": "given", "var": 5, "value": 2},
        {"ftype": "rel", "op": "add", "args": [4, 5], "result": 6},
    ], 6, 74,
    "Consider the numbers a, b, c, d, e, f, g. a is 25. b is 1. a exceeds b by c. d is 3. c "
    "times d equals e. f is 2. e plus f equals g. What is g.",
    "stated ordinal (\"25th term\") gives the count directly; term = first + (n-1)*diff.")

# 454: cbrt(30^3+40^3+50^3), scaled by common factor 10
add(454,
    "Simplify completely: $$\\sqrt[3]{30^3+40^3+50^3}$$.",
    [
        {"ftype": "given", "var": 0, "value": 10},
        {"ftype": "given", "var": 1, "value": 30},
        {"ftype": "rel", "op": "mul", "args": [0, 2], "result": 1},   # 10*a2 = 30 -> a2=3
        {"ftype": "given", "var": 3, "value": 40},
        {"ftype": "rel", "op": "mul", "args": [0, 4], "result": 3},   # 10*b2 = 40 -> b2=4
        {"ftype": "given", "var": 5, "value": 50},
        {"ftype": "rel", "op": "mul", "args": [0, 6], "result": 5},   # 10*c2 = 50 -> c2=5
        {"ftype": "rel", "op": "mul", "args": [2, 2], "result": 7},   # a2^2=9
        {"ftype": "rel", "op": "mul", "args": [7, 2], "result": 8},   # a2^3=27
        {"ftype": "rel", "op": "mul", "args": [4, 4], "result": 9},   # b2^2=16
        {"ftype": "rel", "op": "mul", "args": [9, 4], "result": 10},  # b2^3=64
        {"ftype": "rel", "op": "mul", "args": [6, 6], "result": 11},  # c2^2=25
        {"ftype": "rel", "op": "mul", "args": [11, 6], "result": 12},  # c2^3=125
        {"ftype": "rel", "op": "add", "args": [8, 10], "result": 13},  # 27+64=91
        {"ftype": "rel", "op": "add", "args": [13, 12], "result": 14},  # +125=216
        {"ftype": "rel", "op": "mul", "args": [15, 15], "result": 16},  # r*r
        {"ftype": "rel", "op": "mul", "args": [16, 15], "result": 14},  # r^3 = 216 -> r=6
        {"ftype": "rel", "op": "mul", "args": [0, 15], "result": 17},  # 10*r = 60
    ], 17, 60,
    "Consider the numbers a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r. a is 10. "
    "b is 30. a times c equals b. d is 40. a times e equals d. f is 50. a times g equals f. "
    "c times c equals h. h times c equals i. e times e equals j. j times e equals k. g times "
    "g equals l. l times g equals m. i plus k equals n. n plus m equals o. p times p equals "
    "q. q times p equals o. a times p equals r. What is r.",
    "canonical common-factor scaling (10 divides 30,40,50 exactly -- GCD-based, not arbitrary): "
    "30/40/50 reconstructed in-graph via multiplicative inversion from the source literals and "
    "the constant 10 (a*c=b etc., letting the solver find c=3,e=4,g=5); cubes of the reduced "
    "values (27,64,125) stay under the value cap where the true cubes (27000,64000,125000) "
    "would not; sum 216 = 6^3 found via search-based cube extraction (technique 1, chained "
    "duplicate-arg mul); final answer rescaled by the same factor of 10. Long chain (18 vars) "
    "flagged for certification risk but the magnitude cap leaves no shorter faithful path.")

# 455: Jimmy stairs, 5 flights, +5s each
def add455():
    factors = [
        {"ftype": "given", "var": 0, "value": 20},   # first
        {"ftype": "given", "var": 1, "value": 5},    # diff
        {"ftype": "given", "var": 2, "value": 5},    # n (flights)
        {"ftype": "given", "var": 3, "value": 1},
        {"ftype": "rel", "op": "sub", "args": [2, 3], "result": 4},   # n-1 = 4
        {"ftype": "rel", "op": "mul", "args": [4, 1], "result": 5},   # (n-1)*diff = 20
        {"ftype": "rel", "op": "add", "args": [5, 0], "result": 6},   # last = 40
        {"ftype": "rel", "op": "add", "args": [0, 6], "result": 7},   # first+last = 60
        {"ftype": "fdiv", "var": 7, "k": 2, "result": 8},             # avg = 30
        {"ftype": "rel", "op": "mul", "args": [8, 2], "result": 9},   # avg*n = 150
    ]
    add(455,
        "Climbing the first flight of stairs takes Jimmy 20 seconds, and each following "
        "flight takes 5 seconds more than the preceding one. How many total seconds does it "
        "take to climb the first five flights of stairs?",
        factors, 9, 150,
        "Consider the numbers a, b, c, d, e, f, g, h, i, j. a is 20. b is 5. c is 5. d is 1. "
        "c exceeds d by e. e times b equals f. f plus a equals g. a plus g equals h. When h "
        "is divided by 2, the quotient is i. i times c equals j. What is j.",
        "stated count (\"five flights\"); last term via (n-1)*diff+first, sum via n*avg, one "
        "fdiv for the average.")
add455()

# 457: y=4x-19, 2x+y=95 -> x
add(457,
    "The two lines $y = 4x - 19$ and $2x+y = 95$ intersect. What is the value of $x$ at the "
    "point of intersection?",
    [
        {"ftype": "given", "var": 0, "value": 4},
        {"ftype": "given", "var": 1, "value": 19},
        {"ftype": "rel", "op": "mul", "args": [0, 2], "result": 3},   # 4x = t
        {"ftype": "rel", "op": "sub", "args": [3, 1], "result": 4},   # t-19 = y
        {"ftype": "given", "var": 5, "value": 2},
        {"ftype": "rel", "op": "mul", "args": [5, 2], "result": 6},   # 2x = t2
        {"ftype": "given", "var": 7, "value": 95},
        {"ftype": "rel", "op": "add", "args": [6, 4], "result": 7},
    ], 2, 19,
    "Consider the numbers a, b, c, d, e, f, g, h. a is 4. b is 19. a times c equals d. d "
    "exceeds b by e. f is 2. f times c equals g. h is 95. g plus e equals h. What is c.",
    "direct system encoding (technique 2): free var c (x) and derived e (y) linked by both "
    "source equations; CSP search finds the unique consistent x=19.")

# 460: f(x)=3x^2-5, f(f(1))
add(460,
    "If $f(x) = 3x^2-5$, what is the value of $f(f(1))$?",
    [
        {"ftype": "given", "var": 0, "value": 1},
        {"ftype": "rel", "op": "mul", "args": [0, 0], "result": 1},
        {"ftype": "given", "var": 2, "value": 3},
        {"ftype": "rel", "op": "mul", "args": [2, 1], "result": 3},
        {"ftype": "given", "var": 4, "value": 5},
        {"ftype": "rel", "op": "sub", "args": [4, 3], "result": 5},
        {"ftype": "rel", "op": "mul", "args": [5, 5], "result": 6},
        {"ftype": "rel", "op": "mul", "args": [2, 6], "result": 7},
        {"ftype": "rel", "op": "sub", "args": [7, 4], "result": 8},
    ], 8, 7,
    "Consider the numbers a, b, c, d, e, f, g, h, i. a is 1. a times a equals b. c is 3. c "
    "times b equals d. e is 5. e exceeds d by f. f times f equals g. c times g equals h. h "
    "exceeds e by i. What is i.",
    "magnitude-fold: f(1)=3-5=-2 (mag 2, negative); sign washes out at the next squaring step "
    "(g=f(1)^2), so no further sign tracking is needed.")

# 461: sum 22, diff 4 -> greater
add(461,
    "The sum of two numbers is 22. Their difference is 4. What is the greater of the two "
    "numbers?",
    [
        {"ftype": "given", "var": 0, "value": 22},
        {"ftype": "given", "var": 1, "value": 4},
        {"ftype": "rel", "op": "add", "args": [0, 1], "result": 2},
        {"ftype": "fdiv", "var": 2, "k": 2, "result": 3},
    ], 3, 13,
    "Consider the numbers a, b, c, d. a is 22. b is 4. a plus b equals c. When c is divided "
    "by 2, the quotient is d. What is d.",
    "greater = (sum+diff)/2.")

# 462: a+b=6, a-b=2 -> a^2-b^2
add(462,
    "If $a+b = 6$ and $a - b = 2$, what is the value of $a^2 - b^2$?",
    [
        {"ftype": "given", "var": 0, "value": 6},
        {"ftype": "given", "var": 1, "value": 2},
        {"ftype": "rel", "op": "mul", "args": [0, 1], "result": 2},
    ], 2, 12,
    "Consider the numbers a, b, c. a is 6. b is 2. a times b equals c. What is c.",
    "algebraic identity a^2-b^2=(a+b)(a-b).")

# 463: x^2+bx+44 = (x+m)^2+8 -> b
add(463,
    "Lulu has a quadratic of the form $x^2+bx+44$, where $b$ is a specific positive number. "
    "Using her knowledge of how to complete the square, Lulu is able to rewrite this quadratic "
    "in the form $(x+m)^2+8$. What is $b$?",
    [
        {"ftype": "given", "var": 0, "value": 44},
        {"ftype": "given", "var": 1, "value": 8},
        {"ftype": "rel", "op": "sub", "args": [0, 1], "result": 2},
        {"ftype": "rel", "op": "mul", "args": [3, 3], "result": 2},
        {"ftype": "given", "var": 4, "value": 2},
        {"ftype": "rel", "op": "mul", "args": [4, 3], "result": 5},
    ], 5, 12,
    "Consider the numbers a, b, c, d, e, f. a is 44. b is 8. a exceeds b by c. d times d "
    "equals c. e is 2. e times d equals f. What is f.",
    "completing-the-square identity: m^2+8=44 -> m^2=36 -> m=6 via search-based root "
    "extraction (technique 1); b=2m.")

# 467: seq 88,85,82,... terms before -17
add(467,
    "How many terms of the arithmetic sequence 88, 85, 82, $\\dots$ appear before the number "
    "$-17$ appears?",
    [
        {"ftype": "given", "var": 0, "value": 88},
        {"ftype": "given", "var": 1, "value": 17},
        {"ftype": "rel", "op": "add", "args": [0, 1], "result": 2},
        {"ftype": "fdiv", "var": 2, "k": 3, "result": 3},
    ], 3, 35,
    "Consider the numbers a, b, c, d. a is 88. b is 17. a plus b equals c. When c is divided "
    "by 3, the quotient is d. What is d.",
    "target -17 rendered as its magnitude (given, negative sign external); 88-(-17)=105 via "
    "add of magnitudes (opposite signs); position offset (n-1) = 105/3 = 35, which is exactly "
    "the count of terms strictly before -17 appears (no further +1/-1 needed).")

# 479: nested radical sqrt(12+sqrt(12+...)) = x
def add479():
    factors = [
        {"ftype": "given", "var": 0, "value": 12},
        {"ftype": "rel", "op": "mul", "args": [1, 1], "result": 2},   # x*x = y
        {"ftype": "rel", "op": "sub", "args": [2, 1], "result": 0},   # y - x = 12
    ]
    add(479,
        "Evaluate $\\sqrt{12 +\\!\\sqrt{12 + \\!\\sqrt{12 + \\!\\sqrt{12 + \\cdots}}}}$.",
        factors, 1, 4,
        "Consider the numbers a, b, c. a is 12. b times b equals c. c exceeds b by a. What "
        "is b.",
        "direct system encoding (technique 2), same pattern as [413]: x=sqrt(12+x) squares "
        "to x^2-x=12; CSP search over nonnegative b finds the unique root b=4 (the x=-3 root "
        "excluded by the nonnegative domain).")
add479()

# 483: sum of 7 smallest distinct positive multiples of 9
add(483,
    "What is the sum of the seven smallest distinct positive integer multiples of 9?",
    [
        {"ftype": "given", "var": 0, "value": 9},
        {"ftype": "given", "var": 1, "value": 7},
        {"ftype": "given", "var": 2, "value": 1},
        {"ftype": "rel", "op": "add", "args": [1, 2], "result": 3},
        {"ftype": "rel", "op": "mul", "args": [1, 3], "result": 4},
        {"ftype": "fdiv", "var": 4, "k": 2, "result": 5},
        {"ftype": "rel", "op": "mul", "args": [0, 5], "result": 6},
    ], 6, 252,
    "Consider the numbers a, b, c, d, e, f, g. a is 9. b is 7. c is 1. b plus c equals d. b "
    "times d equals e. When e is divided by 2, the quotient is f. a times f equals g. What "
    "is g.",
    "sum = 9*(1+2+...+7) = 9*28 = 252, using n(n+1)/2 for the count-7 triangular sum.")

# 488: a Y b = a^2-2ab+b^2 = (a-b)^2, a=3,b=2
add(488,
    "If $a \\text{ Y } b$ is defined as $a \\text{ Y } b = a^2 - 2ab + b^2$, what is the "
    "value of $3 \\text{ Y } 2$?",
    [
        {"ftype": "given", "var": 0, "value": 3},
        {"ftype": "given", "var": 1, "value": 2},
        {"ftype": "rel", "op": "sub", "args": [0, 1], "result": 2},
        {"ftype": "rel", "op": "mul", "args": [2, 2], "result": 3},
    ], 3, 1,
    "Consider the numbers a, b, c, d. a is 3. b is 2. a exceeds b by c. c times c equals d. "
    "What is d.",
    "algebraic identity a^2-2ab+b^2=(a-b)^2.")

print()
print(f"TOTAL drafted (pre-checks passing): {len(rows)}")
print(f"FAILS: {fails}")

with open('/home/bryce/mycelium/.cache/book8_t2_prose_pairs_draft.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print("done, wrote", len(rows), "rows")
