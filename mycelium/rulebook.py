"""rulebook.py — ONE RULEBOOK, TWO DOORS (2026-09-18, word given). The dialect's declarative rules, held
once and applied at the ADMISSION door (a row enters the diet only if it satisfies them) and at the DECODE
door (a decoded graph is constrained or censused by the same rules). Each rule names the register it is true
of: "all" rules hold on every register and may act as masks; "prose" rules hold on the pen convention only
and may only be READ, never applied to mint (the positional mask, 2026-09-17: +0.000 on prose, −0.70 on mint).

Doors:
  admission — admit_annotation.rulebook wraps `row_reasons` (the first failing rule's name);
  decode    — `legal_values(text)` + `choose_legal(dig_logits, values, nd)` = THE NUMERAL MASK (an "all" rule:
              a given's value is a numeral in the text, a lexicon constant, or 1); `violations(parse)` censuses
              the structural rules on a decoded graph without changing it.
"""
import re
import numpy as np

VALUE_CAP = 9999           # the dialect's range (the wide head decodes 7 digits; the diet's values sit under 10^4)
SLOT_CAP = 24              # the head's banks: K_VARS = L_FAC
FDIV_CAP = 3
OPS = ("add", "sub", "mul", "div")   # the annotation dialect; the gold grammar re-encodes sub/div as add/mul (canonical_positional.reencode_ops)

RULES = {   # name -> (register, description)
    "value_range":     ("all",   "a given's value is an integer in [0, VALUE_CAP]"),
    "value_legal":     ("all",   "a given's value is a numeral in the text, a lexicon constant, or 1 (the numeral mask)"),
    "capacity":        ("all",   "n_vars and the factor count are within the head's banks"),
    "pointers":        ("all",   "a relation's two args and its result are variables of the row"),
    "op":              ("all",   "a relation's operator is one of the dialect's"),
    # "single_intro" REFUTED 2026-09-18: the graph is a set of CONSTRAINTS — a variable may be both a relation's result and a
    # given ("40 = partner + 10": add(1,2)->0 with given 0 = 40); the repair read −0.02 wild / −0.32 mint on the control.
    "query_in_graph":  ("all",   "the query variable appears in some factor"),
    "fdiv":            ("all",   "at most FDIV_CAP constant divisions, in the diet's form"),
    "span_bounds":     ("all",   "every span lies inside the text"),
    "pointer_order":   ("prose", "a relation points only at variables introduced at or before its slot (READ ONLY — mint numbers by first mention)"),
}


def legal_values(text, nd=7):
    """the values a given may carry on this text: its numerals, the lexicon's constants, and 1"""
    from mycelium import lexicon as L
    vals = {1}
    for m in re.findall(r"\d[\d,]*", text):
        try: v = int(m.replace(",", ""))
        except ValueError: continue
        if 0 <= v < 10 ** nd: vals.add(v)
    for _, _, v in L.constants(text):
        if 0 <= int(v) < 10 ** nd: vals.add(int(v))
    return sorted(vals)


def digits_of(v, nd):
    return [(v // 10 ** (nd - 1 - d)) % 10 for d in range(nd)]


def _lsm(x):
    x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))


def choose_legal(dig_logits, values, nd=None):
    """THE NUMERAL MASK: the most probable legal value under the digit heads' joint log-probability"""
    nd = nd or dig_logits.shape[0]; ls = _lsm(dig_logits); best = None
    for v in values:
        sc = sum(ls[d, dd] for d, dd in enumerate(digits_of(v, nd)))
        if best is None or sc > best[0]: best = (sc, v)
    return best[1] if best else None


def legal_digit_logits(dig_logits, text, nd=None):
    """the digit logits rewritten so the argmax is the legal choice (what the read compares); None if no legal value"""
    nd = nd or dig_logits.shape[0]; v = choose_legal(dig_logits, legal_values(text, nd), nd)
    if v is None: return None
    fake = np.full_like(dig_logits, -1e9)
    for d, dd in enumerate(digits_of(v, nd)): fake[d, dd] = 0.0


def legal_arg_vars(pres, res, j, mode="intro"):
    """THE LEGAL-POINTER MASK (2026-09-21, the numeral mask's sibling for
    pointers): the "pointers" rule read at decode time — a relation's
    argument may only be a variable that some OTHER PRESENT slot
    introduced. `pres` and `res` are the model's OWN per-slot
    predictions (predicted presence, predicted res) for every slot of
    the row, in slot order — read-time, no gold. Legal variables for
    slot j = {res[k] : k present, k != j} ("intro"); mode "prefix"
    additionally requires k < j — THE POSITIONAL LAW: an argument points
    to an EARLIER slot (exact on the prose/positional convention;
    approximate on mint, which numbers variables by first mention
    instead of slot order — see RULES["pointer_order"], register
    "prose"). Mode "index" (2026-09-21) does not touch `res` at all —
    intro/prefix compound the res head's own error rate (0.80-0.90 on
    targets) into the mask, silently masking out the pointer's correct
    answer wherever res mislabels the introducing slot; "index" trusts
    THE POSITIONAL LAW directly (variable v is introduced by slot v)
    instead of the res head's guess: legal set = {v : v < j, pres[v]}
    — pure index + presence, exact on positional rows (94% of wild),
    approximate on mint (which numbers by first mention, not slot
    order). "index" still LOSES (2026-09-21 census): two conditions in
    it remove correct answers. (1) `pres[v] > 0` depends on the
    presence head (0.91 on targets) — the mask must not depend on
    ANOTHER head's error rate any more than intro/prefix depend on
    res's; drop it. (2) `v < j` strictly excludes v == j, but THE ONE-
    UNKNOWN-ARGUMENT pattern the canonicalizer licenses ("40 = partner
    + 10": the relation AT slot j introduces variable j as one of its
    OWN arguments, with an EARLIER variable as its result) makes the
    slot's own index a legal, correct pointer — ~15% of the diet's
    relation slots. Mode "index2" (2026-09-21) is the pure-index
    correction: legal set = {v : v <= j}, no presence at all — every
    variable index up to and including the slot's own. Returns a set
    of legal variable indices (possibly empty)."""
    if mode == "index":
        return {k for k in range(len(pres)) if k < j and pres[k]}
    if mode == "index2":
        return {v for v in range(j + 1)}
    out = set()
    for k in range(len(pres)):
        if k == j or not pres[k]:
            continue
        if mode == "prefix" and not (k < j):
            continue
        out.add(int(res[k]))
    return out


def legal_arg_logits(args_logits, pres, res, j, mode="intro"):
    """args_logits (K_VARS,) rewritten so illegal columns are -inf and
    legal columns keep their own score (a top-2/argmax over this ranks
    LEGAL candidates only) — the args-decode analogue of
    legal_digit_logits. None if no legal variable (the row has nothing
    to point at under this rule; the caller keeps the raw logits, same
    no-op convention as legal_digit_logits)."""
    legal = legal_arg_vars(pres, res, j, mode)
    if not legal:
        return None
    idx = [v for v in legal if 0 <= v < len(args_logits)]
    if not idx:
        return None
    fake = np.full_like(args_logits, -1e9)
    fake[idx] = args_logits[idx]
    return fake
    return fake


def row_reasons(row):
    """THE ADMISSION DOOR: the first failing "all" rule's name, or None when the row is well-formed"""
    n = row.get("n_vars"); facs = row.get("factors") or []; text = row.get("text", "")
    if not isinstance(n, int) or n <= 0: return "n_vars"
    if not facs: return "no_factors"
    if n > SLOT_CAP or len(facs) > SLOT_CAP: return f"capacity_{len(facs)}f_{n}v"
    if "query_var" not in row or not (0 <= row["query_var"] < n): return "query_var"
    used = set(); n_fdiv = 0
    for f in facs:
        ft = f.get("ftype")
        if ft == "given":
            v = f.get("value")
            if not isinstance(v, int) or not (0 <= v <= VALUE_CAP): return f"value_{v}"
            if not (0 <= f.get("var", -1) < n): return "given_var"
            used.add(f["var"])
        elif ft == "rel":
            if f.get("op") not in OPS: return f"op_{f.get('op')}"
            a = f.get("args", []); r = f.get("result")
            if len(a) != 2 or not all(0 <= x < n for x in a) or not (0 <= (r if r is not None else -1) < n): return "rel_pointers"
            used.update(a); used.add(r)
        elif ft == "fdiv":
            n_fdiv += 1
            if not (isinstance(f.get("k"), int) and f["k"] >= 2) or not (0 <= f.get("var", -1) < n) or not (0 <= f.get("result", -1) < n): return "fdiv_form"
            used.add(f["var"]); used.add(f["result"])
        else:
            return f"ftype_{ft}"
        for s, e in f.get("spans") or []:
            if not (0 <= s < e <= len(text)): return "span_bounds"
    if n_fdiv > FDIV_CAP: return "fdiv_count"
    if row["query_var"] not in used: return "query_not_in_graph"
    for v_str, spans in (row.get("mentions") or {}).items():
        if not (0 <= int(v_str) < n): return "mention_var"
        for s, e in spans:
            if not (0 <= s < e <= len(text)): return "mention_bounds"
    return None


def violations(parse, text=None, register="all"):
    """THE DECODE DOOR's census: which rules a decoded graph (a list of factor dicts) breaks — a dict of
    rule -> count. Never changes the parse. `register="prose"` adds the prose-only rules as reads."""
    out = {}
    def hit(k): out[k] = out.get(k, 0) + 1
    intro = {}; vals = set(legal_values(text)) if text is not None else None
    for k, f in enumerate(parse):
        if f.get("ftype") == "given":
            v = f.get("value")
            if not isinstance(v, int) or not (0 <= v <= VALUE_CAP): hit("value_range")
            if vals is not None and v not in vals: hit("value_legal")
            intro.setdefault(f.get("var"), []).append(k)
        elif f.get("ftype") == "rel":
            if f.get("op") not in ("add", "mul") + OPS: hit("op")
            a = list(f.get("args", [])); r = f.get("result")
            if len(a) != 2 or r is None: hit("pointers")
            if r is not None: intro.setdefault(r, []).append(k)
            if register == "prose":
                if any(x > k for x in a) or (r is not None and r > k): hit("pointer_order")
    for v, ks in intro.items():
        if len(ks) > 1: hit("single_intro")
    return out

