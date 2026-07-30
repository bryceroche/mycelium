"""trace_layer.py — THE ONE TRACE LAYER (2026-07-30, the deep clean's
second catch: two fences shared a name but not a body — book8_certify's
division marker lacked \\frac/\\div and required the literal ' divided
by ', while the wild fixture's copy was wider. Two fences with the same
name must be the same fence; both consumers now import THIS one).

The lexical-vs-parse consistency layer: the only wild-frontier defense
built on a different axis than the parse (the orthogonality law,
gut #101). Marker language in the text with no matching factor in the
winning parses -> trip. Parse-asserted numerics (given values, fdiv/mod
k) absent from the text -> trip (value fabrication, the wild-lie
template). Widened division semantics verified safe on all 300 book-8
dialects (zero 'divided'-without-'by', zero fraction notation) and on
the 644-row marker audit; value check at 0 false trips on 266 certified
rows (.cache/trace_layer_audit.json)."""
import re

WORDNUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORDNUM.update(thirty=30, forty=40, fifty=50, sixty=60, seventy=70,
               eighty=80, ninety=90, hundred=100)


def text_numbers(text):
    s = {int(n) for n in re.findall(r"\d+", text)}
    tl = text.lower()
    s |= {v for w, v in WORDNUM.items() if re.search(r"\b" + w + r"\b", tl)}
    return s


def trace_trips(text, win_parses):
    """All trips for a candidate: marker-vs-parse + value fabrication.
    win_parses: list of factor lists (the views that voted the winner)."""
    def wf(pred):
        return any(pred(fa) for f_ in win_parses for fa in f_)
    trips = []
    MARKERS = [
        ("division", ("divided" in text) or ("quotient" in text)
         or ("\\frac" in text) or ("\\div" in text),
         lambda fa: fa.get("ftype") in ("fdiv", "mod")),
        ("percent", " percent" in text, lambda fa: fa.get("ftype") == "pct"),
        ("selection", (" the larger of " in text) or (" the smaller of " in text),
         lambda fa: fa.get("ftype") == "sel"),
        ("times", " times " in text,
         lambda fa: fa.get("ftype") == "rel" and fa.get("op") == "mul"),
        ("plus", (" plus " in text) or ("The sum of" in text),
         lambda fa: fa.get("ftype") == "rel" and fa.get("op") == "add"),
        ("remainder", "remainder" in text, lambda fa: fa.get("ftype") == "mod"),
    ]
    trips += [name for name, lang, pred in MARKERS if lang and not wf(pred)]
    tn = text_numbers(text)
    # "p" joined 2026-07-30 (deep clean): pct's percentage field was the one
    # parse-asserted numeric NO fence checked — not TRIPS (presence only),
    # not this check (key omitted), not attestation (givens only). A parse
    # asserting p=40 against a "60 percent" text sailed through everything.
    fabs = sorted({(key, int(fa[key])) for f_ in win_parses for fa in f_
                   for key in ("value", "k", "p") if key in fa
                   and int(fa[key]) not in tn})
    trips += [f"value-fab:{k}={v}" for k, v in fabs]
    return trips
