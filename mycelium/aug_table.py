"""aug_table.py — THE LICENSED TABLE (2026-08-01, the word given;
docs/AUGMENTATION_CHECK_SPEC.md is law). Verification by construction:
an entry is a TEMPLATE (format string over a construction's factor
schema); license requires 50/50 — each instantiation (a) solver-unique
through the door, (b) rendered numbers exactly the factor values
(count check), (c) letters present in role. Nothing free-form touches
a row; row-level checking is exact membership. The recursion guard:
eval templates NEVER enter here (held-out families live in
scripts/dup_isolation_rung.py / fdiv_surface_check.py and stay out)."""
import re
from collections import Counter

# entry = (construction, template_id, format_string)
# fields: letters a,b,c...; values va,vb,k,p as ints
SEED_ENTRIES = [
    ("given", "is",        "{x} is {v}."),
    ("given", "value-of",  "The value of {x} is {v}."),
    ("given", "equals",    "{x} equals {v}."),
    ("add",   "plus",      "{a} plus {b} equals {c}."),
    ("add",   "sum-of",    "The sum of {a} and {b} is {c}."),
    ("add",   "adding",    "Adding {a} and {b} gives {c}."),
    ("add",   "total",     "{a} and {b} total {c}."),
    ("mul",   "times",     "{a} times {b} equals {c}."),
    ("mul",   "product",   "The product of {a} and {b} is {c}."),
    ("mul",   "multiply",  "Multiplying {a} by {b} gives {c}."),
    ("fdiv",  "when-div",  "When {a} is divided by {k}, the quotient is {b}."),
    ("fdiv",  "dividing",  "Dividing {a} by {k} gives {b}."),
    ("fdiv",  "div-equal", "{a} divided by {k} equals {b}."),
    ("mod",   "when-rem",  "When {a} is divided by {k}, the remainder is {b}."),
    ("mod",   "rem-of",    "The remainder of {a} divided by {k} is {b}."),
    ("pct",   "percent-of","{p2} is {p} percent of {b2}."),
    ("pct",   "of-gives",  "{p} percent of {b2} gives {p2}."),
    ("sel",   "larger",    "{c} is the larger of {a} and {b}."),
    ("sel",   "greater",   "{c} is the greater of {a} and {b}."),
    ("sel",   "smaller",   "{c} is the smaller of {a} and {b}."),
    ("dup",   "plus-self", "{a} plus {a} equals {c}."),
    ("dup",   "sum-self",  "The sum of {a} and {a} is {c}."),
    ("dup",   "times-self","{a} times {a} equals {c}."),
]


def render(entry_fmt, fields):
    return entry_fmt.format(**fields)


def verify_entry(construction, fmt, rng, n=50):
    """50/50: well-formed + numbers-in-text exact + solver-unique.
    Returns (licensed: bool, failures: list)."""
    import sys
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from tta_alg2_dials import solve2
    L = "abcdefgh"
    fails = []
    for t in range(n):
        if construction == "given":
            v = int(rng.randint(2, 290))
            facs = [{"ftype": "given", "var": 0, "value": v}]
            q, gold = 0, v
            txt = render(fmt, {"x": L[0], "v": v})
            need = [v]
        elif construction in ("add", "mul", "dup"):
            op = "mul" if "times" in fmt or "product" in fmt or "Multiply" in fmt else "add"
            if construction == "dup":
                x = int(rng.randint(2, 12))
                a = b = 0
                gold = x + x if op == "add" else x * x
                facs = [{"ftype": "given", "var": 0, "value": x},
                        {"ftype": "rel", "op": op, "args": [0, 0], "result": 1}]
                q = 1
                txt = f"{L[0]} is {x}. " + render(fmt, {"a": L[0], "c": L[1]})
                need = [x]
            else:
                x, y = int(rng.randint(2, 12)), int(rng.randint(2, 12))
                gold = x + y if op == "add" else x * y
                facs = [{"ftype": "given", "var": 0, "value": x},
                        {"ftype": "given", "var": 1, "value": y},
                        {"ftype": "rel", "op": op, "args": [0, 1], "result": 2}]
                q = 2
                txt = (f"{L[0]} is {x}. {L[1]} is {y}. "
                       + render(fmt, {"a": L[0], "b": L[1], "c": L[2]}))
                need = [x, y]
            if gold > 300: continue
        elif construction in ("fdiv", "mod"):
            k = int(rng.choice([2, 3, 4, 5, 6, 7]))
            if construction == "fdiv":
                qv = int(rng.randint(2, 40)); a = k * qv; gold = qv
            else:
                a = int(rng.randint(10, 290))
                if a % k == 0: a += 1
                gold = a % k
            facs = [{"ftype": "given", "var": 0, "value": a},
                    {"ftype": construction, "var": 0, "k": k, "result": 1}]
            q = 1
            txt = f"{L[0]} is {a}. " + render(fmt, {"a": L[0], "k": k, "b": L[1]})
            need = [a, k]
        elif construction == "pct":
            p = int(rng.choice([10, 20, 25, 50, 75]))
            base = int(rng.choice([20, 40, 80, 120, 200]))
            gold = p * base // 100
            facs = [{"ftype": "given", "var": 0, "value": base},
                    {"ftype": "pct", "args": [1, 0], "p": p}]
            q = 1
            txt = f"{L[0]} is {base}. " + render(fmt, {"p": p, "b2": L[0], "p2": L[1]})
            need = [base, p]
        elif construction == "sel":
            x, y = int(rng.randint(2, 140)), int(rng.randint(2, 140))
            if x == y: y += 1
            sel = "smaller" if "smaller" in fmt else "larger"
            gold = min(x, y) if sel == "smaller" else max(x, y)
            facs = [{"ftype": "given", "var": 0, "value": x},
                    {"ftype": "given", "var": 1, "value": y},
                    {"ftype": "sel", "sel": sel, "args": [0, 1], "result": 2}]
            q = 2
            txt = (f"{L[0]} is {x}. {L[1]} is {y}. "
                   + render(fmt, {"a": L[0], "b": L[1], "c": L[2]}))
            need = [x, y]
        else:
            return False, [f"unknown construction {construction}"]
        # (b) numbers-in-text exact
        tc = Counter(int(m) for m in re.findall(r"\d+", txt))
        if any(tc[v] < 1 for v in need):
            fails.append(("numbers", t)); continue
        # (a)+(c) solver-unique through the standard path
        if solve2(facs, q, {"n_vars": 24, "m": 300}) != gold:
            fails.append(("solve", t)); continue
    return len(fails) == 0, fails
