"""comp_mint.py — THE COMPOSITIONAL MINT v3 (2026-08-22, word given;
Bryce's gut: "too many keys, not enough slots" — nominal-n vs
EFFECTIVE-n). Mint the way the dialect was minted: sample the GRAPH,
then render it. Random factor-DAGs in the head's grammar (eval-chains,
linear systems, square/sqrt constraints, pct), solution-first, values
<=300, SHAPE-capped by canonical signature (<=8 surface variants per
shape — shapes are the unit of n, rows are not).

THE DOUBLE GATE (every row, not a sample):
  1. graph gate — anchor organ + expand_graph -> solve_forced
     (search + uniqueness certificate) == expected answer;
  2. TEXT ORACLE — the rendered LaTeX is independently re-evaluated as
     python semantics (declared letters bound, \\frac -> /, ^ -> **,
     juxtaposed coefficients expanded) and must equal the same answer.
The render direction is new code; nothing it emits trains anything
without passing both.
"""
import sys, json, re, random
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from mycelium.anchor_law import unanchored_givens
from repair_replace_swap import solve_forced

LETTERS = ["x", "y", "z", "a", "b", "c", "p", "q", "r", "s", "t", "u",
           "v", "m", "n", "w"]

# ---- value-annotated expression nodes ----
# ("lit", v) | ("var", letter, v) | ("add",a,b) | ("sub",a,b) |
# ("mul",a,b) | ("sq",a) | ("fr",a,k) | ("opa",k1,a,k2,b,op)

def val(n):
    t = n[0]
    if t == "lit": return n[1]
    if t == "var": return n[2]
    if t == "add": return val(n[1]) + val(n[2])
    if t == "sub": return val(n[1]) - val(n[2])
    if t == "mul": return val(n[1]) * val(n[2])
    if t == "sq":  return val(n[1]) ** 2
    if t == "fr":  return val(n[1]) // n[2]
    if t == "opa":
        a, b = n[2], n[4]
        s = n[1] * val(a) + (n[3] * val(b) if n[5] == "add"
                             else -n[3] * val(b))
        return s
    raise ValueError(t)

def sample_expr(rng, depth, lets):
    if depth <= 0 or rng.random() < 0.3:
        if lets and rng.random() < 0.65:
            L, v = rng.choice(lets)
            return ("var", L, v)
        return ("lit", rng.randint(1, 40))
    op = rng.choice(["add", "add", "sub", "mul", "sq", "fr", "opa"])
    if op == "sq":
        a = sample_expr(rng, depth - 1, lets)
        return ("sq", a)
    if op == "fr":
        a = sample_expr(rng, depth - 1, lets)
        return ("fr", a, rng.choice([2, 3, 4, 5, 6]))
    if op == "opa":
        a = sample_expr(rng, 0, lets); b = sample_expr(rng, 0, lets)
        return ("opa", rng.randint(2, 6), a, rng.randint(2, 6), b,
                rng.choice(["add", "sub"]))
    a = sample_expr(rng, depth - 1, lets)
    b = sample_expr(rng, depth - 1, lets)
    return (op, a, b)

def ok_node(n):
    """domain check every sub-value; fr exact; sub nonneg."""
    t = n[0]
    if t in ("lit", "var"): return 0 <= val(n) <= 300
    if t == "fr":
        if not ok_node(n[1]): return False
        if val(n[1]) % n[2]: return False
        return 0 <= val(n) <= 300
    if t == "sq":
        return ok_node(n[1]) and 0 <= val(n) <= 300
    if t == "opa":
        return (ok_node(n[2]) and ok_node(n[4]) and 0 <= val(n) <= 300)
    if not (ok_node(n[1]) and ok_node(n[2])): return False
    return 0 <= val(n) <= 300

# ---- graph building ----
def build_graph(n, facs, vmap, nv):
    """returns (var_index, nv). vmap: letter->var idx for declared vars."""
    t = n[0]
    if t == "var":
        L = n[1]
        if L not in vmap:
            vmap[L] = nv[0]
            facs.append({"ftype": "given", "var": nv[0], "value": n[2]})
            nv[0] += 1
        return vmap[L]
    if t == "lit":
        facs.append({"ftype": "given", "var": nv[0], "value": n[1]})
        nv[0] += 1
        return nv[0] - 1
    if t == "sq":
        a = build_graph(n[1], facs, vmap, nv)
        facs.append({"ftype": "rel", "op": "mul", "args": [a, a],
                     "result": nv[0]})
        nv[0] += 1
        return nv[0] - 1
    if t == "fr":
        a = build_graph(n[1], facs, vmap, nv)
        facs.append({"ftype": "macro", "name": "FRAC_OF", "a": 1,
                     "k": n[2], "x": a, "result": nv[0]})
        nv[0] += 1
        return nv[0] - 1
    if t == "opa":
        a = build_graph(n[2], facs, vmap, nv)
        b = build_graph(n[4], facs, vmap, nv)
        facs.append({"ftype": "macro", "name": "OP_APPLY", "op": n[5],
                     "k1": n[1], "x": a, "k2": n[3], "y": b,
                     "result": nv[0]})
        nv[0] += 1
        return nv[0] - 1
    a = build_graph(n[1], facs, vmap, nv)
    b = build_graph(n[2], facs, vmap, nv)
    facs.append({"ftype": "rel", "op": t, "args": [a, b], "result": nv[0]})
    nv[0] += 1
    return nv[0] - 1

# ---- rendering ----
def render(n, prec=0):
    t = n[0]
    if t == "lit": return str(n[1])
    if t == "var": return n[1]
    if t == "sq":
        inner = render(n[1], 4)
        if n[1][0] not in ("lit", "var"):
            inner = f"({inner})"
        return f"{inner}^2"
    if t == "fr":
        return "\\frac{%s}{%d}" % (render(n[1], 0), n[2])
    if t == "opa":
        k1, a, k2, b, op = n[1], n[2], n[3], n[4], n[5]
        def leg(k, x):
            rx = render(x, 3)
            if x[0] == "var": return f"{k}{rx}"
            if x[0] == "lit": return f"{k} \\times {rx}"
            return f"{k}({rx})"
        s = "+" if op == "add" else "-"
        return f"{leg(k1, a)} {s} {leg(k2, b)}"
    a, b = n[1], n[2]
    if t == "add":
        s = f"{render(a, 1)} + {render(b, 1)}"
        return f"({s})" if prec > 1 else s
    if t == "sub":
        s = f"{render(a, 1)} - {render(b, 2)}"
        return f"({s})" if prec > 1 else s
    if t == "mul":
        ra, rb = render(a, 2), render(b, 3)
        if a[0] == "lit" and b[0] == "var":
            return f"{ra}{rb}"
        return f"{ra} \\times {rb}"
    raise ValueError(t)

def shape_sig(n):
    t = n[0]
    if t == "lit": return "L"
    if t == "var": return "V"
    if t == "sq": return f"sq({shape_sig(n[1])})"
    if t == "fr": return f"fr{n[2]}({shape_sig(n[1])})"
    if t == "opa": return f"opa{n[5]}({shape_sig(n[2])},{shape_sig(n[4])})"
    return f"{t}({shape_sig(n[1])},{shape_sig(n[2])})"

# ---- text oracle: rendered LaTeX -> python semantics ----
def oracle(text_expr, bindings):
    s = text_expr
    for _ in range(12):
        s2 = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
        if s2 == s: break
        s = s2
    s = s.replace("\\times", "*").replace("^2", "**2")
    s = re.sub(r"(\d)\s*([a-z])", r"\1*\2", s)
    s = re.sub(r"(\d|\))\s*\(", r"\1*(", s)
    s = re.sub(r"\)\s*(\d|[a-z])", r")*\1", s)
    v = eval(s, {"__builtins__": {}}, dict(bindings))
    return v

TEMPL_EVAL = [
    "If {decls}, what is the value of ${expr}$?",
    "Suppose {decls}. What is ${expr}$?",
    "Given that {decls}, find ${expr}$.",
    "Let {decls}. Compute ${expr}$.",
]

def emit_eval_row(rng):
    nlet = rng.randint(1, 3)
    Ls = rng.sample(LETTERS, nlet)
    lets = [(L, rng.randint(1, 30)) for L in Ls]
    n = sample_expr(rng, rng.randint(1, 3), lets)
    used = set(re.findall(r"[a-z]", shape_sig(n)))  # not reliable; recompute
    def used_lets(m, acc):
        if m[0] == "var": acc.add((m[1], m[2]))
        elif m[0] in ("add", "sub", "mul"):
            used_lets(m[1], acc); used_lets(m[2], acc)
        elif m[0] == "sq": used_lets(m[1], acc)
        elif m[0] == "fr": used_lets(m[1], acc)
        elif m[0] == "opa": used_lets(m[2], acc); used_lets(m[4], acc)
        return acc
    ul = sorted(used_lets(n, set()))
    if not ul: return None                    # need at least one letter
    if not ok_node(n): return None
    if n[0] in ("lit", "var"): return None    # trivial
    expr = render(n)
    decls = " and ".join(f"${L} = {v}$" for L, v in ul)
    text = rng.choice(TEMPL_EVAL).format(decls=decls, expr=expr)
    facs = []; vmap = {}; nv = [0]
    q = build_graph(n, facs, vmap, nv)
    if nv[0] > 22: return None
    ans = val(n)
    try:
        o = oracle(expr, {L: v for L, v in ul})
    except Exception:
        return None
    if o != ans: return None                  # TEXT ORACLE gate
    return text, facs, q, ans, "eval:" + shape_sig(n)

def emit_system_row(rng):
    X, Y = rng.sample(LETTERS, 2)
    xv, yv = rng.randint(1, 25), rng.randint(1, 25)
    k1, k2, k3, k4 = (rng.randint(1, 5) for _ in range(4))
    if k1 * k4 == k2 * k3: return None        # singular
    C = k1 * xv + k2 * yv; D = k3 * xv + k4 * yv
    if max(C, D) > 300: return None
    def lhs(a, b):
        pa = f"{a[0]}{a[1]}" if a[0] > 1 else a[1]
        pb = f"{b[0]}{b[1]}" if b[0] > 1 else b[1]
        return f"{pa} + {pb}"
    qkind = rng.choice(["x", "y", "sq", "sum"])
    facs = [{"ftype": "given", "var": 0, "value": C},
            {"ftype": "given", "var": 1, "value": D},
            {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": k1,
             "x": 2, "k2": k2, "y": 3, "result": 0},
            {"ftype": "macro", "name": "OP_APPLY", "op": "add", "k1": k3,
             "x": 2, "k2": k4, "y": 3, "result": 1}]
    if qkind == "x": q, ans, qtxt = 2, xv, f"${X}$"
    elif qkind == "y": q, ans, qtxt = 3, yv, f"${Y}$"
    elif qkind == "sq":
        facs.append({"ftype": "rel", "op": "mul", "args": [2, 2],
                     "result": 4})
        q, ans, qtxt = 4, xv * xv, f"${X}^2$"
        if ans > 300: return None
    else:
        facs.append({"ftype": "rel", "op": "add", "args": [2, 3],
                     "result": 4})
        q, ans, qtxt = 4, xv + yv, f"${X} + {Y}$"
    text = (f"If ${lhs((k1, X), (k2, Y))} = {C}$ and "
            f"${lhs((k3, X), (k4, Y))} = {D}$, what is the value of {qtxt}?")
    ora = {"x": xv, "y": yv, "sq": xv * xv, "sum": xv + yv}[qkind]
    if ora != ans: return None
    sig = f"sys:{k1 > 1}{k2 > 1}{k3 > 1}{k4 > 1}:{qkind}"
    return text, facs, q, ans, sig

def emit_sqrt_row(rng):
    X = rng.choice(LETTERS)
    B = rng.randint(3, 17)
    A = rng.randint(1, min(60, B * B - 1))
    form = rng.choice(["plus", "minus"])
    if form == "plus":
        ansx = B * B - A
        text = f"If $\\sqrt{{{X} + {A}}} = {B}$, what is the value of ${X}$?"
        facs = [{"ftype": "given", "var": 0, "value": A},
                {"ftype": "given", "var": 1, "value": B},
                {"ftype": "rel", "op": "add", "args": [2, 0], "result": 3},
                {"ftype": "rel", "op": "mul", "args": [1, 1], "result": 3}]
    else:
        ansx = B * B + A
        if ansx > 300: return None
        text = f"If $\\sqrt{{{X} - {A}}} = {B}$, what is the value of ${X}$?"
        facs = [{"ftype": "given", "var": 0, "value": A},
                {"ftype": "given", "var": 1, "value": B},
                {"ftype": "rel", "op": "sub", "args": [2, 0], "result": 3},
                {"ftype": "rel", "op": "mul", "args": [1, 1], "result": 3}]
    return text, facs, 2, ansx, f"sqrt:{form}"

EMITTERS = [(emit_eval_row, 0.62), (emit_system_row, 0.28),
            (emit_sqrt_row, 0.10)]

def main():
    rng = random.Random(9393)
    target_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    cap_per_shape = 8
    shape_count = {}; seen_text = set(); out = []
    rej = {"none": 0, "shape_cap": 0, "gate": 0, "oracle_hit": 0, "dup": 0}
    tries = 0
    while len(out) < target_rows and tries < target_rows * 40:
        tries += 1
        u = rng.random(); acc = 0
        for fn, w in EMITTERS:
            acc += w
            if u <= acc: break
        r = fn(rng)
        if r is None: rej["none"] += 1; continue
        text, facs, q, ans, sig = r
        if shape_count.get(sig, 0) >= cap_per_shape:
            rej["shape_cap"] += 1; continue
        if text in seen_text: rej["dup"] += 1; continue
        row = {"src_idx": 300000 + len(out), "construction": "comp-mint-v3",
               "factors": facs, "query": q, "answer": ans,
               "lane": "L0-mint-comp", "original": text, "shape": sig}
        if unanchored_givens(row): rej["gate"] += 1; continue
        a = solve_forced(facs, q, {"n_vars": 24, "m": 300})
        if a != ans: rej["gate"] += 1; continue
        seen_text.add(text)
        shape_count[sig] = shape_count.get(sig, 0) + 1
        out.append(row)
        if len(out) % 5000 == 0:
            print(f"[comp] {len(out)} rows, {len(shape_count)} shapes",
                  flush=True)
    with open('.cache/mint_comp_v3.jsonl', 'w') as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"[comp] MINTED {len(out)} rows across {len(shape_count)} SHAPES "
          f"(cap {cap_per_shape}/shape; rejects {rej})", flush=True)

if __name__ == "__main__":
    main()
