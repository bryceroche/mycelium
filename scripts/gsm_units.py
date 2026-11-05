"""gsm_units.py -- the unit heuristic (ledger 2026-11-04 OP-DIAG; 11-05 UNIT-BIND). A HEURISTIC, never gold.

A number token's unit = the first content token after it (stop list skipped; a comma / question word ends the scan; "$" is stripped by the
tokenizer, so a bare amount after a money word -> dollar; "<num> times" -> dimensionless; % / percent -> "%"; word multipliers half / twice /
double / ... -> dimensionless; "<unit> per|an|a|every <noun>" -> a rate). Moved verbatim from scripts/gsm_opdiag_read.py (the banked OP-DIAG
instrument keeps its own copy). For UNIT-BIND a unit is a product of base words with integer exponents: {} = dimensionless, {X: 1, Y: -1} =
X per Y, None = unknown; the phase of a unit = sum of exponent x theta(word), so x adds phases, / subtracts, and X/Y x Y = X exactly.
"""
STOPW = set("of a an the more less fewer each per extra additional other different new old big small large full whole total than as many much "
            "times and or to in on for with at by from his her their its my your our this that these those every only about around over under "
            "just all some few several is are was were be been has have had will would can could - 's \" ( ) : ; x".split())
MONEYC = {"cost", "costs", "pay", "paid", "pays", "earn", "earns", "earned", "save", "saves", "saved", "spend", "spends", "spent", "price",
          "charge", "charges", "charged", "sell", "sells", "sold", "buy", "buys", "bought", "worth", "for", "at", "makes", "made", "salary", "wage"}
MULT = {"half", "twice", "double", "doubled", "triple", "tripled", "thrice", "quarter", "quadruple"}
RATEW = {"per", "an", "a", "every"}
MONEY = {"dollar", "buck"}


def sing(w):
    if w.endswith("ies") and len(w) > 4: return w[:-3] + "y"
    if w.endswith("es") and w[:-2].endswith(("s", "x", "ch", "sh")): return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3: return w[:-1]
    return w


def content(toks, j, lim=3):
    """the first content token in toks[j : j + lim] (stops at a number or sentence punctuation) -> (index, word) or (None, None)"""
    for i in range(j, min(j + lim, len(toks))):
        w = toks[i]
        if w in ("<num>", ".", "?", "!", ",", "how", "what", "if", "then", "so", "when") or w in MULT: return None, None
        if w in STOPW or not any(ch.isalpha() for ch in w): continue
        return i, w
    return None, None


def norm(w):
    w = sing(w)
    if w in MONEY or w == "dollars": return "dollar"
    if w == "cents": return "cent"
    return w


def unit_of(it, i):
    """the unit string of the number token at i: "X", "X/Y", "1", "%" or None"""
    toks = it["toks"]
    if toks[i] in MULT: return "1"
    nx = toks[i + 1] if i + 1 < len(toks) else ""
    if nx in ("%", "percent"): return "%"
    if nx == "times": return "1"
    if (i > 0 and toks[i - 1] == "$") or nx in ("dollar", "dollars"):
        u, j = "dollar", i + (2 if nx in ("dollar", "dollars") else 1)
    else:
        j0, w = content(toks, i + 1)
        if w is None:                                    # "$" is stripped by the tokenizer: a bare amount after a money word is dollars
            return "dollar" if MONEYC & set(toks[max(0, i - 3):i]) else None
        u, j = norm(w), j0 + 1
    if j < len(toks) and toks[j] in RATEW:
        _, y = content(toks, j + 1, 2)
        if y is not None: u = f"{u}/{norm(y)}"
    return u


def exps(u):
    """a unit string -> {word: exponent} ({} dimensionless) or None"""
    if u is None: return None
    if u == "1": return {}
    if "/" in u:
        x, y = u.split("/", 1)
        return {} if x == y else {x: 1, y: -1}
    return {u: 1}


def combine(op, a, b):
    """the unit of a step from its operands' units (None unknown): x adds exponents, / subtracts, + / - keep A's (B's if A is unknown)"""
    if op in "+-": return a if a is not None else b
    if a is None or b is None: return None
    out = dict(a)
    for w, e in b.items(): out[w] = out.get(w, 0) + (e if op == "*" else -e)
    return {w: e for w, e in out.items() if e != 0}


def token_units(it):
    """per token: the exponent dict of each number token (None: not a number, or unknown) and its kind (0 none, 1 unknown, 2 dimless, 3 unit)"""
    if "_units" not in it:
        U, KD = [], []
        for i, v in enumerate(it["vals"]):
            if v is None: U.append(None); KD.append(0); continue
            e = exps(unit_of(it, i)); U.append(e); KD.append(1 if e is None else 2 if not e else 3)
        it["_units"] = (U, KD)
    return it["_units"]
