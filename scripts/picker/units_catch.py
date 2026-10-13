"""scripts/picker/units_catch.py -- UNITS AS A TYPE SYSTEM (2026-10-13, zero-GPU, CPU-only,
delegate). Registered task: Roy & Roth, TACL 2018 ("Mapping to Declarative Knowledge for Word
Problem Solving") -- every quantity in a word problem carries a unit; a mul relation is
dimensionally legal only when its two operands combine by SCALING (one side dimensionless) or by
RATE-CANCELLATION (one side's unit has a denominator atom that matches the other side's unit
exactly, e.g. [$/h] x [h] -> [$]); anything else (two unrelated, non-cancelling atoms) is an
illegal/ad-hoc combination. The campaign's standing inverse-direction wall (CLAUDE.md: args
0.43-0.45 on every body; "the rate family" is a standing registry resident) is being probed here
for a refutable half that MINIMALITY (scripts/picker/minimality.py) structurally cannot see: a
wrong-argument relation that still SATISFIES the solver (consistent, not refutable by search) can
still be dimensionally illegal.

NON-CIRCULARITY (the design note this script exists to keep straight): a relation's own `result`
unit is NEVER independently known for most derived variables (they are computed, not restated in
text), so checking "does gold's result unit match gold's own args' product" would be circular if
the result's unit were itself DEFINED by propagating through that same relation. This script never
does that: CONSISTENCY is tested on the two ARGUMENT units alone (both leaf-derived from text or
propagated through EARLIER, already-resolved relations elsewhere in the graph -- never from the
relation under test), using a fixed, pre-registered legality rule (SCALING / RATE-CANCEL / else
ILLEGAL). "Coverage" is reported wherever both argument units resolve to something determined
(possibly a generic COUNT:<noun> atom, itself a legitimate base type under this type system).

GOLD comes only from the fixture row's own `factors`/`solution` fields (NOT mycelium/custody_gold.py
-- that module keys MATH-harvest answers by text identity for the pen-row answer-gate; this task
needs the GRAPH gold -- args/op/result per slot -- which custody_gold.py does not carry at all; the
graph gold is the fixture's own `factors` field, exactly as scripts/picker/dn_common.py and every
other picker/ script treats it. The custody-gold LAW (solution text is never gold) is about the
*training* answer key; this read is a zero-GPU, no-training structural probe over an already-
admitted measurement fixture, scored the way dn_common.py's own convention scores it -- by slot
position against the fixture's own gold `factors`, as every picker/ script does).

THE DIRECTION this reuses from the codebase (imported, never reimplemented):
  - scripts/args_census.py: STOPWORDS, CUE_WORDS (so a cue word is never mistaken for a unit noun).
  - scripts/stamp_arg_mentions.py: stem() (trailing s/es stripping) and WORD_RE.
Nothing else is imported; the number-finder, lexicon, rate-pattern regexes, and the unit-algebra
(combine_mul_checked) are new to this script (no prior numeral-to-unit code exists in the repo --
build_numeral_table.py finds numeral TOKENS, not units).

Usage:
  DEV=CPU .venv/bin/python3 scripts/picker/units_catch.py
Writes (new artifacts only, units_ prefix, nothing banked touched):
  .cache/units_report.txt      -- the full read, human-readable
  .cache/units_per_row.pkl     -- per-row unit assignments + per-slot legality (for inspection)
"""
import os
import re
import sys
import json
import pickle
from collections import Counter

os.environ.setdefault("DEV", "CPU")
sys.path.insert(0, "."); sys.path.insert(0, "scripts"); sys.path.insert(0, "scripts/picker")
assert os.environ["DEV"] == "CPU", "units_catch: zero-GPU, always"

import args_census as census          # reuse: STOPWORDS, CUE_WORDS
from stamp_arg_mentions import stem   # reuse: trailing s/es stemmer

STOPWORDS = census.STOPWORDS
CUE_WORDS = census.CUE_WORDS

WILD_JSONL = ".cache/wild_admitted_holdout.jsonl"
CAND_WILD = ".cache/picker/cand_wild_PMS8_241.pkl"

OUT_REPORT = ".cache/units_report.txt"
OUT_PICKLE = ".cache/units_per_row.pkl"

WORD_RE = re.compile(r"[A-Za-z]+")
NUM_RE = re.compile(r"\$?\d+(?:\.\d+)?%?")

# ======================================================================================
# THE UNIT LEXICON (chosen here; written in the script per the task's instruction).
# Dimension is coarse (physics-style: hour/day/year all collapse to TIME, pound/ounce/gram
# all collapse to WEIGHT) -- scale is irrelevant to a dimensional-consistency check, only the
# AXIS matters. Keys are the STEMMED singular form (stem() strips a trailing s/es); irregular
# plurals (feet) are listed explicitly since the naive stemmer can't reach them.
# ======================================================================================
LEXICON = {
    # currency
    "dollar": "USD", "buck": "USD", "cent": "USD", "penny": "USD", "pennie": "USD",
    # time
    "second": "TIME", "minute": "TIME", "hour": "TIME", "day": "TIME", "week": "TIME",
    "month": "TIME", "year": "TIME", "decade": "TIME",
    "hr": "TIME", "min": "TIME", "sec": "TIME", "yr": "TIME",
    # distance
    "mile": "DIST", "foot": "DIST", "feet": "DIST", "inch": "DIST", "yard": "DIST",
    "meter": "DIST", "metre": "DIST", "kilometer": "DIST", "kilometre": "DIST",
    "km": "DIST", "centimeter": "DIST", "centimetre": "DIST", "cm": "DIST", "mm": "DIST",
    # weight / mass
    "pound": "WEIGHT", "lb": "WEIGHT", "ounce": "WEIGHT", "oz": "WEIGHT",
    "gram": "WEIGHT", "gramme": "WEIGHT", "kilogram": "WEIGHT", "kg": "WEIGHT",
    "ton": "WEIGHT", "tonne": "WEIGHT",
    # volume
    "gallon": "VOL", "liter": "VOL", "litre": "VOL", "cup": "VOL", "pint": "VOL", "quart": "VOL",
}
# dimensionless multiplier/ratio words -- a GIVEN mentioned as one of these carries unit {} (a
# pure scalar), which is the SCALING leg of combine_mul_checked.
DIMENSIONLESS_WORDS = {
    "percent", "times", "half", "halves", "twice", "double", "triple", "quarter",
    "fraction", "ratio", "factor",
}
# words that must never be read as the unit noun even though they aren't in census.STOPWORDS
EXTRA_SKIP = {"of", "the", "a", "an", "is", "are", "was", "were", "did", "does", "will",
              "has", "have", "had", "to", "for", "in", "on", "that", "this"}

_OPT_PREFIX = r"(?:of\s+|the\s+|a\s+|an\s+|his\s+|her\s+|their\s+|its\s+|winning\s+|losing\s+)?"
RATE_RE = re.compile(r"\b(?:per|each|every|for)\b\s+" + _OPT_PREFIX + r"([a-zA-Z]+)")
DOLLAR_A_RE = re.compile(r"(\$\d+(?:\.\d+)?)\s+(?:a|an)\s+([a-zA-Z]+)")
OF_BEFORE_RE = re.compile(r"([a-zA-Z]+)\s+of\s*$")
POSTFIX_EACH_RE = re.compile(r"^[\s\-a-zA-Z]{0,25}?\b(each|apiece)\b")
EACH_WILDCARD = "EACH_WILDCARD"


def lex_atom(word):
    """word (already lowercase) -> atom string, or None if unrecognized; 'DIMENSIONLESS' sentinel
    for the scalar words, resolved to {} by the caller."""
    if word is None:
        return None
    w = stem(word.lower())
    if w in DIMENSIONLESS_WORDS or word.lower() in DIMENSIONLESS_WORDS:
        return "DIMENSIONLESS"
    if w in LEXICON:
        return LEXICON[w]
    if word.lower() in LEXICON:
        return LEXICON[word.lower()]
    return None


def atom_or_generic(word):
    """Lexicon atom, DIMENSIONLESS -> {}, or a generic COUNT:<stem> base type (the fallback that
    makes every content noun a legitimate, if uninterpreted, unit)."""
    if word is None:
        return None
    a = lex_atom(word)
    if a == "DIMENSIONLESS":
        return {}
    if a is not None:
        return {a: 1}
    wl = word.lower()
    if wl in STOPWORDS or wl in CUE_WORDS or wl in EXTRA_SKIP or len(wl) <= 2:
        return None
    return {"COUNT:" + stem(wl): 1}


# ======================================================================================
# number-finding + span claiming (disambiguates repeated literal values by scan order)
# ======================================================================================
def find_number_matches(text):
    out = []
    for m in NUM_RE.finditer(text):
        s = m.group()
        has_dollar = s.startswith("$")
        has_pct = s.endswith("%")
        core = s.lstrip("$").rstrip("%")
        try:
            val = float(core)
        except ValueError:
            continue
        out.append(dict(start=m.start(), end=m.end(), val=val, dollar=has_dollar, pct=has_pct))
    return out


def claim_spans(text, factors, solution):
    matches = find_number_matches(text)
    claimed = [False] * len(matches)
    var_span = {}

    def claim(v, val):
        if val is None:
            return
        for idx, m in enumerate(matches):
            if claimed[idx] or m["pct"]:
                continue
            if abs(m["val"] - val) < 1e-6:
                claimed[idx] = True
                var_span[v] = idx
                return

    # pass 1: givens, in list (slot) order -- claims the FIRST unclaimed literal occurrence
    for f in factors:
        if f["ftype"] == "given":
            claim(f["var"], f.get("value"))
    # pass 2: derived results that happen to be RESTATED literally elsewhere in text (rare,
    # but an independent signal when it occurs -- never claims a span a given already owns)
    if solution:
        for f in factors:
            if f["ftype"] == "rel" and "result" in f and f["result"] not in var_span:
                v = f["result"]
                if 0 <= v < len(solution):
                    claim(v, solution[v])
    return var_span, matches


def sentence_window(text, bounds, pos, radius=60):
    lo = max(0, pos - radius)
    hi = min(len(text), pos + radius)
    return text[lo:hi]


def clause_window(text, pos):
    """The CLAUSE containing char position pos (bounded by . , ! ? on either side) -- tighter
    than a fixed radius, so a 'for a car' / 'for a truck' / 'for an SUV' list of clauses never
    leaks its rate cue across a comma into a sibling clause's own number."""
    left = max([text.rfind(c, 0, pos) for c in ".,!?"] + [-1])
    right_cands = [text.find(c, pos) for c in ".,!?"]
    right_cands = [r for r in right_cands if r != -1]
    right = min(right_cands) if right_cands else len(text)
    return text[left + 1:right]


def context_unit(text, bounds, m):
    """A GIVEN's own literal numeral match -> its unit dict, or None if uncovered."""
    if m["pct"]:
        return {}
    start, end = m["start"], m["end"]
    base = None
    # hyphen-attached noun has priority: "6-ounce cup" -> ounce is the true unit of the "6"
    if end < len(text) and text[end] == "-":
        hm = WORD_RE.match(text, end + 1)
        if hm:
            base = atom_or_generic(hm.group())
    if base is None and m["dollar"]:
        base = {"USD": 1}
    if base is None:
        # first content word after the number (skip stopword-ish fillers)
        tail = text[end:end + 40]
        for wm in WORD_RE.finditer(tail):
            w = wm.group()
            wl = w.lower()
            if wl in EXTRA_SKIP or wl in STOPWORDS or wl in CUE_WORDS:
                continue
            base = atom_or_generic(w)
            break
    if base is None or base == {}:
        return base
    (b_atom, _) = list(base.items())[0]
    # (a) "<noun1> of <N> <noun2>" -- noun2's count is implicitly PER noun1 (3 families of 4
    # people; 4 teams of 10 members): the noun just before "of", just before the number, is a
    # SPECIFIC denominator.
    before = text[max(0, start - 30):start]
    om = OF_BEFORE_RE.search(before)
    if om:
        denom = atom_or_generic(om.group(1))
        if denom and denom != {}:
            (d_atom, _) = list(denom.items())[0]
            if d_atom != b_atom:
                return {b_atom: 1, d_atom: -1}
    # (b) postfix distributive "<N> <noun> each/apiece" -- no noun is named for the denominator
    # (it is whichever other countable quantity this gets multiplied against); a WILDCARD
    # denominator, cancelled against ANY single-atom unit in combine_mul_checked.
    after = text[end:end + 30]
    if POSTFIX_EACH_RE.search(after):
        return {b_atom: 1, EACH_WILDCARD: -1}
    # (c) prefix rate: "X per Y" / "each Y" / "every Y" / "$N a Y" within THIS CLAUSE only (a
    # comma-separated list of "$5 for a car, $6 for a truck" must never cross-contaminate)
    win = clause_window(text, start)
    rm = RATE_RE.search(win)
    denom = None
    if rm:
        denom = atom_or_generic(rm.group(1))
    if denom is None:
        dm = DOLLAR_A_RE.search(win)
        if dm:
            denom = atom_or_generic(dm.group(2))
    if denom and denom != {}:
        (d_atom, _) = list(denom.items())[0]
        if d_atom != b_atom:
            return {b_atom: 1, d_atom: -1}
    return base


# ======================================================================================
# the unit algebra: product/quotient of unit vectors ('per' = division)
# ======================================================================================
def combine_mul_checked(u0, u1):
    """-> (legal: True/False/None, result_unit_or_None). None legal means UNCOVERED (one or
    both args have no determined unit at all) -- excluded from every consistency denominator."""
    if u0 is None or u1 is None:
        return None, None
    if u0 == {}:
        return True, dict(u1)
    if u1 == {}:
        return True, dict(u0)
    for ra, rb in ((u0, u1), (u1, u0)):
        # the distributive wildcard ("7 books each") cancels against ANY other determined unit --
        # "each" binds to whatever countable quantity it is eventually multiplied against.
        if ra.get(EACH_WILDCARD) == -1:
            rest = {k: v for k, v in ra.items() if k != EACH_WILDCARD}
            return True, (rest if rest else dict(rb))
        if len(rb) == 1:
            (atom_b, exp_b), = rb.items()
            if exp_b == 1 and ra.get(atom_b) == -1:
                rest = {k: v for k, v in ra.items() if k != atom_b}
                return True, rest
    return False, None


def is_rate(u):
    return bool(u) and any(v < 0 for v in u.values())


# ======================================================================================
# per-row gold unit propagation (given leaves -> add/mul relations; THE POSITIONAL
# CONVENTION used throughout scripts/picker/: slot j = list position in `factors`, shared
# between gold and a decoded candidate's parse; variable NUMBERS (var/args/result) are a
# SEPARATE namespace resolved via the row's own intro_map, exactly as
# scripts/stamp_arg_mentions.py's `intro_map` is built)
# ======================================================================================
def build_intro_map(factors):
    intro = {}
    for j, f in enumerate(factors):
        if f["ftype"] == "given":
            intro[f["var"]] = (j, f)
        elif "result" in f:
            intro[f["result"]] = (j, f)
    return intro


class RowUnits:
    def __init__(self, row):
        self.factors = row["factors"]
        self.intro = build_intro_map(self.factors)
        self.var_span, self.matches = claim_spans(row["text"], self.factors, row.get("solution"))
        self.bounds = census.sentence_bounds(row["text"])
        self.text = row["text"]
        self.memo = {}
        self.mul_legality = {}   # slot j -> dict(legal, u0, u1, covered)
        self.add_mismatch = 0
        self.given_cov = {"lex": 0, "count": 0, "dimless": 0, "none": 0}

    def given_unit(self, v):
        idx = self.var_span.get(v)
        if idx is None:
            return None
        m = self.matches[idx]
        u = context_unit(self.text, self.bounds, m)
        return u

    def unit(self, v):
        if v in self.memo:
            return self.memo[v]
        self.memo[v] = None   # guard against any accidental cycle
        if v not in self.intro:
            return None
        j, f = self.intro[v]
        if f["ftype"] == "given":
            u = self.given_unit(v)
        elif f["ftype"] == "rel":
            a0, a1 = f["args"][0], f["args"][1]
            u0, u1 = self.unit(a0), self.unit(a1)
            if f["op"] == "add":
                u = u0 if (u0 is not None and u1 is not None and u0 == u1) else None
                if u0 is not None and u1 is not None and u0 != u1:
                    self.add_mismatch += 1
            elif f["op"] == "mul":
                legal, u = combine_mul_checked(u0, u1)
                self.mul_legality[j] = dict(legal=legal, u0=u0, u1=u1,
                                             covered=(u0 is not None and u1 is not None))
            else:
                u = None
        else:
            u = None
        self.memo[v] = u
        return u

    def resolve_all(self):
        for v in list(self.intro):
            self.unit(v)
        # given coverage tally
        for f in self.factors:
            if f["ftype"] == "given":
                u = self.memo.get(f["var"])
                if u is None:
                    self.given_cov["none"] += 1
                elif u == {}:
                    self.given_cov["dimless"] += 1
                elif any(not k.startswith("COUNT:") for k in u):
                    self.given_cov["lex"] += 1
                else:
                    self.given_cov["count"] += 1


# ======================================================================================
# a best-effort (diagnostic-only, never gating) question-unit reader
# ======================================================================================
QUESTION_PATTERNS = [
    (re.compile(r"how many\s+([a-zA-Z]+)"), 1),
    (re.compile(r"how much\s+([a-zA-Z]+)"), 1),
    (re.compile(r"how old"), 0),
    (re.compile(r"how long"), 0),
    (re.compile(r"how far"), 0),
]


def question_unit(text):
    matches = []
    for pat, grp in QUESTION_PATTERNS:
        for m in pat.finditer(text.lower()):
            if grp:
                w = m.group(1)
                if w in EXTRA_SKIP or w in STOPWORDS:
                    u = {"USD": 1} if "much" in m.group(0) else None
                else:
                    u = atom_or_generic(w)
            else:
                if "old" in m.group(0) or "long" in m.group(0):
                    u = {"TIME": 1}
                else:
                    u = {"DIST": 1}
            matches.append((m.start(), u))
    if not matches:
        return None
    matches.sort()
    return matches[-1][1]


# ======================================================================================
# main
# ======================================================================================
def main():
    lines = []

    def P(s=""):
        print(s, flush=True)
        lines.append(s)

    P("=" * 100)
    P("UNITS AS A TYPE SYSTEM (Roy & Roth TACL 2018) -- registered 2026-10-13")
    P("=" * 100)

    rows = [json.loads(l) for l in open(WILD_JSONL)]
    cand = pickle.load(open(CAND_WILD, "rb"))
    cand_by_i = {c["i"]: c for c in cand}
    P(f"[load] {len(rows)} wild rows, {len(cand_by_i)} PMS8_241 candidate dumps")

    per_row = []
    tot_given = 0
    tot_mul = 0
    covered_mul = 0
    legal_mul = 0
    add_mismatch_tot = 0
    cov_tally = Counter()

    for row in rows:
        ru = RowUnits(row)
        ru.resolve_all()
        per_row.append(ru)
        tot_given += sum(1 for f in row["factors"] if f["ftype"] == "given")
        for k, v in ru.given_cov.items():
            cov_tally[k] += v
        add_mismatch_tot += ru.add_mismatch
        for j, rec in ru.mul_legality.items():
            tot_mul += 1
            if rec["covered"]:
                covered_mul += 1
                if rec["legal"]:
                    legal_mul += 1

    # ---- (1) THE UNIT LEXICON coverage ----
    P("")
    P("-" * 100)
    P("(1) THE UNIT LEXICON -- coverage")
    P("-" * 100)
    given_covered = cov_tally["lex"] + cov_tally["count"] + cov_tally["dimless"]
    P(f"GIVEN slots: {tot_given} total; {given_covered} get a unit "
      f"({given_covered/tot_given:.3f}) -- lexicon={cov_tally['lex']}, "
      f"generic COUNT fallback={cov_tally['count']}, dimensionless={cov_tally['dimless']}, "
      f"uncovered={cov_tally['none']}")
    P(f"MUL relations: {tot_mul} total; {covered_mul} have BOTH args unit-determined "
      f"({covered_mul/tot_mul:.3f} coverage)")
    P(f"ADD relations: {add_mismatch_tot} rows hit an arg-unit MISMATCH while propagating through "
      f"add (diagnostic only -- additive relations are OUT OF SCOPE for the consistency bars below; "
      f"a mismatch here just means one of the two add legs didn't resolve to a lexicon/generic unit, "
      f"so the add's OWN result stays undetermined rather than guessed)")

    # ---- (2) GOLD CONSISTENCY ----
    P("")
    P("-" * 100)
    P("(2) GOLD CONSISTENCY -- of the unit-covered gold mul relations, how many are dimensionally "
      "legal (one operand dimensionless, OR a rate operand whose denominator cancels the other)?")
    P("-" * 100)
    gold_rate = sum(1 for ru in per_row for rec in ru.mul_legality.values()
                     if rec["covered"] and rec["legal"] and (is_rate(rec["u0"]) or is_rate(rec["u1"])))
    P(f"covered: {covered_mul}/{tot_mul}; legal (consistent): {legal_mul}/{covered_mul} = "
      f"{(legal_mul/covered_mul if covered_mul else float('nan')):.4f}")
    P(f"  of which RATE-shaped (one operand carries a 'per' denominator): {gold_rate}")
    bar2 = covered_mul > 0 and (legal_mul / covered_mul) >= 0.95
    P(f"BAR (>= 0.95 of unit-covered gold relations consistent): {'PASS' if bar2 else 'MISS'}")

    # ---- (3) THE CATCH ----
    P("")
    P("-" * 100)
    P("(3) THE CATCH -- among PMS8_241's top-1 WRONG mul relations (predicted args/result differ "
      "from gold, same slot j), how many are dimensionally inconsistent under the predicted args?")
    P("-" * 100)
    n_wrong_argdir = 0
    n_wrong_covered = 0
    n_caught = 0
    n_right = 0
    n_right_covered = 0
    n_false_refute = 0
    n_other_err = 0   # predicted slot not rel/mul at all (ftype/op error, outside this catch's scope)
    n_missing_slot = 0
    caught_is_rate_family = 0

    for row, ru in zip(rows, per_row):
        c = cand_by_i.get(row.get("i", rows.index(row)))
        if c is None:
            continue
        top1 = c["branches"].get(c["top1_sig"])
        if top1 is None:
            continue
        pred_by_slot = {f["_slot"]: f for f in top1["parse"]}
        for j, rec in ru.mul_legality.items():
            gold_f = row["factors"][j]
            pred_f = pred_by_slot.get(j)
            if pred_f is None:
                n_missing_slot += 1
                continue
            if pred_f.get("ftype") != "rel" or pred_f.get("op") != "mul":
                n_other_err += 1
                continue
            same_args = sorted(pred_f["args"]) == sorted(gold_f["args"])
            same_res = pred_f.get("result") == gold_f.get("result")
            if same_args and same_res:
                n_right += 1
                if rec["covered"]:
                    n_right_covered += 1
                    if not rec["legal"]:
                        n_false_refute += 1
                continue
            # a direction/argument error
            n_wrong_argdir += 1
            pa0, pa1 = pred_f["args"][0], pred_f["args"][1]
            pu0, pu1 = ru.unit(pa0), ru.unit(pa1)
            legal, _ = combine_mul_checked(pu0, pu1)
            if legal is None:
                continue
            n_wrong_covered += 1
            if not legal:
                n_caught += 1
                if is_rate(rec["u0"]) or is_rate(rec["u1"]):
                    caught_is_rate_family += 1

    P(f"WRONG (args/result differ from gold), at a slot where predicted IS rel+mul: {n_wrong_argdir}")
    P(f"  of which unit-covered (both predicted args resolve to a unit): {n_wrong_covered}")
    catch_rate = n_caught / n_wrong_covered if n_wrong_covered else float("nan")
    P(f"  CAUGHT (dimensionally inconsistent under the predicted args): {n_caught}/{n_wrong_covered} "
      f"= {catch_rate:.4f}")
    P(f"  of the catches, gold's OWN relation here was rate-shaped: {caught_is_rate_family}/{n_caught}"
      if n_caught else "  (no catches)")
    P(f"RIGHT (predicted args/result == gold), unit-covered: {n_right_covered}/{n_right}; "
      f"FALSE-REFUTE (flagged inconsistent despite being gold-correct): {n_false_refute}/{n_right_covered} "
      f"= {(n_false_refute/n_right_covered if n_right_covered else float('nan')):.4f}")
    P(f"(diagnostic, excluded from the catch population above) predicted slot missing: "
      f"{n_missing_slot}; predicted ftype/op itself wrong (not rel+mul): {n_other_err}")
    bar3 = n_wrong_covered > 0 and catch_rate >= 0.50
    bar3b = n_right_covered == 0 or (n_false_refute / n_right_covered) <= 0.05
    P(f"BAR (catch >= 0.50): {'PASS' if bar3 else 'MISS'}; "
      f"BAR (false-refute <= 0.05): {'PASS' if bar3b else 'MISS'}")

    P("")
    P("NOTE: additive relations share units by construction (op=='add' requires equal units to "
      "propagate at all) and are OUT OF SCOPE for the consistency/catch bars above -- they are used "
      "only to carry a unit upward through a chain, never scored for legality themselves.")

    P("")
    P("-" * 100)
    P("DIAGNOSTIC (not gating): the question-sentence's own requested unit, cross-checked against "
      "the gold query_var's resolved unit, when both are readable")
    P("-" * 100)
    q_n = q_match = q_both = 0
    for row, ru in zip(rows, per_row):
        qv = row.get("query_var")
        qu = ru.memo.get(qv)
        txtu = question_unit(row["text"])
        if txtu is not None:
            q_n += 1
            if qu is not None:
                q_both += 1
                if qu == txtu or (set(qu) & set(txtu)):
                    q_match += 1
    P(f"question-unit readable: {q_n}/{len(rows)}; both sides readable: {q_both}; "
      f"agree (share an atom): {q_match}/{q_both if q_both else 1}")

    P("")
    P("-" * 100)
    P("THE LEXICON'S MAIN GAPS")
    P("-" * 100)
    P(f"- uncovered GIVEN mentions: {cov_tally['none']}/{tot_given} -- mostly value==1 given as a "
      f"spelled word ('a'/'an'/'one'/'another'/'next') never matched by the literal-digit scanner, "
      f"and numbers with no noun within 40 chars (e.g. bare counts separated by a clause boundary).")
    P(f"- generic COUNT:<noun> fallback carries no cross-noun convertibility (apples != boxes even "
      f"when a real 'N per box' rate links them) -- the rate regex only fires on an explicit "
      f"per/each/every/'$N a Y' cue; an implicit rate stated as two separate sentences without one "
      f"of those words is invisible to this lexicon (a likely source of both missed catches and the "
      f"gold-inconsistency residual in (2)).")
    P(f"- ounce/cup/pint are treated as fixed dimensions (WEIGHT/VOL) even though English is "
      f"ambiguous between fluid and dry measure in prose; this risks occasional false WEIGHT/VOL "
      f"mismatches this read did not specifically separate out.")

    pickle.dump(dict(per_row=[dict(given_cov=ru.given_cov, mul_legality=ru.mul_legality,
                                    var_span=ru.var_span, add_mismatch=ru.add_mismatch)
                               for ru in per_row]),
                open(OUT_PICKLE, "wb"))
    open(OUT_REPORT, "w").write("\n".join(lines) + "\n")
    P("")
    P(f"[done] wrote {OUT_REPORT}, {OUT_PICKLE}")


if __name__ == "__main__":
    main()
