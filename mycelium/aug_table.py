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
    # ---- tranche 2 (2026-08-01): THE SYMBOLIC CLASS (div first) ----
    ("fdiv",  "sym-slash", "{a} / {k} = {b}."),
    ("fdiv",  "sym-eq",    "{b} = {a} / {k}."),
    ("add",   "sym-plus",  "{a} + {b} = {c}."),
    ("add",   "sym-eq",    "{c} = {a} + {b}."),
    ("mul",   "sym-star",  "{a} * {b} = {c}."),
    ("mul",   "sym-eq",    "{c} = {a} * {b}."),
    ("given", "sym-eq",    "{x} = {v}."),
    ("pct",   "sym-pct",   "{p2} is {p}% of {b2}."),
    ("dup",   "sym-plus",  "{a} + {a} = {c}."),
    ("dup",   "sym-star",  "{a} * {a} = {c}."),
]

# tranche 3 (2026-08-01): (construction, id, fmt, constraint, latex)
TRANCHE3 = [
    ("fdiv",  "latex-frac", "$\\frac{{{a}}}{{{k}}} = {b}$.", None, True),
    ("mul",   "latex-cdot", "{a} $\\cdot$ {b} = {c}.", None, True),
    ("fdiv",  "half-of",    "Half of {a} is {b}.", {"k": 2}, False),
    ("fdiv",  "third-of",   "A third of {a} is {b}.", {"k": 3}, False),
    ("subadd","less-than",  "{c} is {b} less than {a}.", None, False),
    ("add",   "more-than",  "{c} is {a} more than {b}.", None, False),
    ("add",   "combined",   "{a} and {b} combined give {c}.", None, False),
    ("subadd","sym-minus",  "{c} = {a} - {b}.", None, False),
]


def render(entry_fmt, fields):
    return entry_fmt.format(**fields)


def tokens_present(txt, needed_ints, tok):
    """THE TOKENIZER PIN (2026-08-01): verify at the TOKENIZER, not the
    string — a \frac may read as text yet tokenize into pieces that
    never present the digits as the model sees them."""
    enc = tok.encode(txt)
    stream = "".join(tok.decode([t]) for t in enc.ids)
    return all(str(v) in stream for v in needed_ints)


def verify_entry(construction, fmt, rng, n=50, constraint=None, latex=False):
    """50/50: well-formed + numbers-in-text exact + solver-unique.
    Returns (licensed: bool, failures: list)."""
    import sys
    sys.path.insert(0, "."); sys.path.insert(0, "scripts")
    from tta_alg2_dials import solve2
    L = "abcdefgh"
    fails = []
    for t in range(n):
        constraint = constraint or {}
        if construction == "subadd":
            # sub's rearranged-add clause: "c is b less than a" => c=a-b,
            # trained as rel-add(args=[b,c], result=a) — the canonical fold
            X, Y = int(rng.randint(20, 200)), int(rng.randint(2, 19))
            gold = X - Y
            facs = [{"ftype": "given", "var": 0, "value": X},
                    {"ftype": "given", "var": 1, "value": Y},
                    {"ftype": "rel", "op": "add", "args": [1, 2], "result": 0}]
            q = 2
            txt = (f"{L[0]} is {X}. {L[1]} is {Y}. "
                   + render(fmt, {"a": L[0], "b": L[1], "c": L[2]}))
            need = [X, Y]
        elif construction == "given":
            v = int(rng.randint(2, 290))
            facs = [{"ftype": "given", "var": 0, "value": v}]
            q, gold = 0, v
            txt = render(fmt, {"x": L[0], "v": v})
            need = [v]
        elif construction in ("add", "mul", "dup"):
            op = "mul" if any(t in fmt for t in ("times", "product", "Multiply", "*")) else "add"
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
            k = int(constraint.get("k", rng.choice([2, 3, 4, 5, 6, 7])))
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
            need = [a] if "k" in constraint else [a, k]
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
        if latex:
            from phase1_algebra_head import TOKENIZER_JSON
            from tokenizers import Tokenizer
            global _TOK
            try: _TOK
            except NameError: _TOK = Tokenizer.from_file(TOKENIZER_JSON)
            if not tokens_present(txt, need, _TOK):
                fails.append(("tokenizer", t)); continue
        # (a)+(c) solver-unique through the standard path
        if solve2(facs, q, {"n_vars": 24, "m": 300}) != gold:
            fails.append(("solve", t)); continue
    return len(fails) == 0, fails

# tranche 4 (2026-08-06, the slope probe's pen tranche — word given;
# axis variety by design: passive, inverted, conditional, result-first,
# colloquial, formal, arithmetic-verb, comparative; the bench
# (verify_entry) licenses each; the pen only proposes):
TRANCHE4 = [
    # given — new axes: assignment, apposition, measurement, inverted
    ("given", "has-value",   "{x} has the value {v}.", None, False),
    ("given", "let-be",      "Let {x} be {v}.", None, False),
    ("given", "we-know",     "We know that {x} is {v}.", None, False),
    ("given", "takes",       "{x} takes the value {v}.", None, False),
    ("given", "suppose",     "Suppose {x} is {v}.", None, False),
    ("given", "given-that",  "Given that {x} equals {v}.", None, False),
    ("given", "v-first",     "{v} is the value of {x}.", None, False),
    ("given", "set-to",      "{x} is set to {v}.", None, False),
    ("given", "stands-at",   "{x} stands at {v}.", None, False),
    # add — result-first, conditional, verb-variety, passive
    ("add", "gives-when",    "Adding {a} to {b} yields {c}.", None, False),
    ("add", "c-first-sum",   "{c} is the sum of {a} and {b}.", None, False),
    ("add", "if-add",        "If you add {a} and {b}, you get {c}.", None, False),
    ("add", "together",      "Together, {a} and {b} make {c}.", None, False),
    ("add", "increased",     "{a} increased by {b} is {c}.", None, False),
    ("add", "result-add",    "The result of adding {a} and {b} is {c}.", None, False),
    ("add", "obtain-add",    "One obtains {c} by adding {a} and {b}.", None, False),
    ("add", "exceeds-by",    "{c} exceeds {b} by {a}.", None, False),
    ("add", "added-to",      "{a} added to {b} gives {c}.", None, False),
    ("add", "combine",       "Combining {a} with {b} produces {c}.", None, False),
    # mul — same axes
    ("mul", "c-first-prod",  "{c} is the product of {a} and {b}.", None, False),
    ("mul", "if-mul",        "If you multiply {a} by {b}, you get {c}.", None, False),
    ("mul", "result-mul",    "The result of multiplying {a} and {b} is {c}.", None, False),
    ("mul", "obtain-mul",    "One obtains {c} by multiplying {a} and {b}.", None, False),
    ("mul", "mult-by",       "{a} multiplied by {b} is {c}.", None, False),
    ("mul", "groups-of",     "{a} groups of {b} make {c}.", None, False),
    ("mul", "sym-x",         "{a} x {b} = {c}.", None, False),
    # fdiv — passive, result-first, conditional, unit-fraction
    ("fdiv", "c-first-quot", "The quotient of {a} and {k} is {b}.", None, False),
    ("fdiv", "if-div",       "If you divide {a} by {k}, you get {b}.", None, False),
    ("fdiv", "split-into",   "{a} split into {k} equal parts gives {b}.", None, False),
    ("fdiv", "shared",       "{a} shared equally among {k} is {b}.", None, False),
    ("fdiv", "goes-into",    "{k} goes into {a} exactly {b} times.", None, False),
    ("fdiv", "result-div",   "The result of dividing {a} by {k} is {b}.", None, False),
    ("fdiv", "quarter-of",   "A quarter of {a} is {b}.", {"k": 4}, False),
    ("fdiv", "fifth-of",     "A fifth of {a} is {b}.", {"k": 5}, False),
    ("fdiv", "b-first-div",  "{b} is {a} divided by {k}.", None, False),
    # mod — variety on remainder phrasing
    ("mod", "leaves-rem",    "{a} divided by {k} leaves a remainder of {b}.", None, False),
    ("mod", "rem-is",        "{a} modulo {k} is {b}.", None, False),
    ("mod", "rem-b-first",   "{b} is the remainder when {a} is divided by {k}.", None, False),
    ("mod", "leaves-over",   "Dividing {a} by {k} leaves {b} left over.", None, False),
    # pct — order and register variety
    ("pct", "pct-sym-of",    "{p}% of {b2} is {p2}.", None, False),
    ("pct", "equals-pct",    "{p2} equals {p} percent of {b2}.", None, False),
    ("pct", "taking-pct",    "Taking {p} percent of {b2} gives {p2}.", None, False),
    ("pct", "pct-comes-to",  "{p} percent of {b2} comes to {p2}.", None, False),
    # sel — superlative and comparative variety
    ("sel", "larger-of",     "The larger of {a} and {b} is {c}.", None, False),
    ("sel", "smaller-of",    "The smaller of {a} and {b} is {c}.", None, False),
    ("sel", "whichever",     "{c} is whichever of {a} and {b} is larger.", None, False),
    ("sel", "max-of",        "The maximum of {a} and {b} is {c}.", None, False),
    ("sel", "min-of",        "The minimum of {a} and {b} is {c}.", None, False),
    # dup — doubling/squaring register
    ("dup", "twice",         "Twice {a} is {c}.", None, False),
    ("dup", "double",        "Double {a} gives {c}.", None, False),
    ("dup", "square-of",     "The square of {a} is {c}.", None, False),
    ("dup", "self-sum",      "{a} added to itself gives {c}.", None, False),
    ("dup", "self-prod",     "{a} multiplied by itself is {c}.", None, False),
    ("dup", "c-first-twice", "{c} is twice {a}.", None, False),
    # subadd — difference register
    ("subadd", "difference", "The difference between {a} and {b} is {c}.", None, False),
    ("subadd", "minus",      "{a} minus {b} equals {c}.", None, False),
    ("subadd", "subtract",   "Subtracting {b} from {a} gives {c}.", None, False),
    ("subadd", "decreased",  "{a} decreased by {b} is {c}.", None, False),
    ("subadd", "fewer",      "{c} is {b} fewer than {a}.", None, False),
    ("subadd", "take-away",  "Taking {b} away from {a} leaves {c}.", None, False),
    ("subadd", "remains",    "When {b} is removed from {a}, {c} remains.", None, False),
]
