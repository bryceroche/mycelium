"""lexicon.py — THE LEXICON (2026-09-15, word given; the operator lexicon of
the order of attack, built as THE SYMBOLIC CONVOLUTION): a sliding matcher
over the token sequence — the same template fires wherever the phrase sits
(weight sharing = translation invariance, zero parameters) — emitting a
token-born Certificate over the matched span. Entries carry a ROLE:
  "value"  a quantity phrase with a number   (a dozen -> 12, forty-five -> 45)
  "mul"    a multiplier on a relation        (twice -> 2, half -> 1/2, triple -> 3)
The bridge (mycelium/loop_bridge) projects a value certificate to the slot
that reads the span (its digit field); the multiplier certificates wait on
the dialect's encoding of constant factors (registered).
Entries are CURATED (hand / silver, fingerpost-gated, versioned apart) —
never mined from the model, never a loss target. The cardinal-number
grammar (composition of hundreds / tens / units) is the first, largest
class of entries and needs no table beyond the number words.
"""
import re

ONES = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"])}
TENS = {w: i * 10 for i, w in enumerate(["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]) if w}
# THE ENTRIES (v1): value phrases and multipliers. Version: lexicon_v1 (2026-09-15).
VALUE_ENTRIES = {"a dozen": 12, "dozen": 12, "a pair": 2, "a pair of": 2, "a hundred": 100, "a thousand": 1000, "a couple": 2, "a couple of": 2}
# THE MULTIPLIERS AS HIDDEN GIVENS (lexicon_v2, 2026-09-15): the pen dialect encodes "twice as
# many" as a GIVEN constant 2 + a mul relation, "half the price" as a given 2 with the relation
# reversed, "a quarter" as a given 4 — so a multiplier is a VALUE certificate for the constant's
# slot, and rides the same road. (The old MUL_ENTRIES with fractional values are retired.)
MUL_ENTRIES = {"twice": 2, "double": 2, "doubled": 2, "triple": 3, "tripled": 3, "thrice": 3, "half": 2, "half of": 2, "a quarter of": 4, "a quarter": 4, "quadruple": 4}
VERSION = "lexicon_v2"

_NUMWORD = re.compile(r"\b(?:(one|two|three|four|five|six|seven|eight|nine)\s+hundred(?:\s+(?:and\s+)?)?)?(?:(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[\s-](one|two|three|four|five|six|seven|eight|nine))?|(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen))?\b", re.I)


def cardinal(text):
    """Every cardinal-number phrase in `text`: [(start, end, value)]. Composition
    of hundreds / tens / units; 'one hundred twenty-three' -> 123; bare 'hundred'
    is not a number (that is 'a hundred', a value entry)."""
    out = []
    for m in _NUMWORD.finditer(text):
        h, t, u, o = (x.lower() if x else None for x in m.groups())
        if not (h or t or o):
            continue
        v = (ONES[h] * 100 if h else 0) + (TENS[t] if t else 0) + (ONES[u] if u else 0) + (ONES[o] if o else 0)
        out.append((m.start(), m.end(), v))
    return out


def entries(text):
    """Table entries in `text`: [(start, end, role, value)], longest match first
    at each position, no overlaps."""
    cands = []
    for tab, role in ((VALUE_ENTRIES, "value"), (MUL_ENTRIES, "value")):   # v2: multipliers are hidden-given VALUES
        for phrase, val in tab.items():
            for m in re.finditer(r"\b" + re.escape(phrase) + r"\b", text, re.I):
                cands.append((m.start(), -(m.end() - m.start()), m.end(), role, val))
    cands.sort(); out = []; last = -1
    for s, _, e, role, val in cands:
        if s >= last:
            out.append((s, e, role, val)); last = e
    return out


def match(text):
    """THE SYMBOLIC CONVOLUTION: all matches, [(start, end, role, value)], cardinals
    and entries merged (an entry wins where it overlaps a cardinal: 'a dozen')."""
    ents = entries(text); taken = [(s, e) for s, e, _, _ in ents]
    out = list(ents)
    for s, e, v in cardinal(text):
        if not any(s < te and e > ts for ts, te in taken):
            out.append((s, e, "value", v))
    return sorted(out)


def span_tokens(matches, offsets, T):
    """Character spans -> token index lists via the tokenizer's offsets
    [(start, end)] per token (the head's convention); tokens beyond T dropped."""
    out = []
    for s, e, role, val in matches:
        toks = [i for i, (a, b) in enumerate(offsets[:T]) if b > s and a < e and b > a]
        if toks:
            out.append((toks, role, val))
    return out


if __name__ == "__main__":
    assert cardinal("c is forty-five.") == [(5, 15, 45)]
    assert cardinal("g is one hundred twenty-three and h is twelve") == [(5, 29, 123), (39, 45, 12)]
    assert cardinal("two hundred and seven") == [(0, 21, 207)]
    assert cardinal("ten. zero") == [(0, 3, 10), (5, 9, 0)]
    assert cardinal("the second number") == []
    assert entries("She bought a dozen eggs and twice as many apples") == [(11, 18, "value", 12), (28, 33, "value", 2)]
    assert entries("half of a dozen") == [(0, 7, "value", 2), (8, 15, "value", 12)]
    m = match("a dozen eggs, one hundred cups, twice the rest, a pair of shoes")
    assert [(r, v) for _, _, r, v in m] == [("value", 12), ("value", 100), ("value", 2), ("value", 2)], m
    offs = [(0, 1), (1, 7), (8, 12), (13, 16), (17, 24), (25, 29)]      # "a dozen eggs and twelve cups"
    st = span_tokens(match("a dozen eggs and twelve cups"), offs, 6)
    assert st == [([0, 1], "value", 12), ([4], "value", 12)], st
    print("[lexicon] self-test PASS: cardinals compose (45, 123, 207), entries longest-first without overlap, "
          "match merges (entry beats cardinal on 'a dozen'), spans -> tokens")
