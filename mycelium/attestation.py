"""attestation.py — the answer-provenance fence (2026-07-25, the re-ruling's
stage two, seated first).

The one door the quorum wall doesn't guard is correlated collapse reaching
quorum (item 228: gen16 sub-quorum on the truth, gen21 quorum 3/5 on an
answer of 0 attested nowhere — the first measured lie-regression). The
fence: an emitted answer must trace to the text. A view's answer is
ATTESTED iff re-solving its compiled graph with ONLY text-attested givens
(given value appears among the source text's numeric literals) still
forces the same answer for the bound query var. The quorum answer flags
ABSTAIN-UNGROUNDED iff no winning view attests it.

A legitimate derived answer always traces — its derivation chain bottoms
out in givens the parser read off the page. A hallucinated given (a value
the text never states) is the only way an unattested answer can reach the
solver, and stripping it either changes the answer or breaks uniqueness
(solve2's force returns None). Either way: flag.

Jurisdiction (stated per Ruling 7): vacuous-by-success on measured
fixtures (expected catch zero-or-one); its real jurisdiction is the wild
registers where walls haven't been measured. Literal attestation is
digit-based; wild registers with number-words need a literal extractor
upgrade BEFORE this fence is trusted there.

Bars (pinned blind in the ledger before any decode fired):
  (i) ZERO false flags on banked-correct rows — any flag on a correct row
      is itself a provenance-gap bug the check just found.
  (ii) catch on bigtest expected exactly 1 (item 228); >1 means the lie
      census and the check disagree and BOTH go under audit.
"""
import re


def text_literals(text):
    """The numeric literals the source text actually states."""
    return {int(x) for x in re.findall(r"\d+", text)}


def strip_unattested(facs, lits):
    """Remove givens whose value the text never states; keep all structure."""
    return [f for f in facs if f.get("ftype") != "given"
            or f.get("value") in lits]


def attest_view(facs, q, answer, text, m, solve_fn):
    """Does this view's answer survive on text-attested givens alone?"""
    kept = strip_unattested(facs, text_literals(text))
    if len(kept) == len(facs):
        return True  # nothing hallucinated; the emitted solve already traces
    return solve_fn(kept, q, {"n_vars": 24, "m": m}) == answer


def attest_quorum(view_parses, quorum_answer, text, m, solve_fn):
    """view_parses: [(facs, q, answer)] for the 5 views.
    Returns (attested, n_winning, n_attesting)."""
    winners = [(f, q) for f, q, a in view_parses if a == quorum_answer]
    n_att = sum(1 for f, q in winners
                if attest_view(f, q, quorum_answer, text, m, solve_fn))
    return n_att > 0, len(winners), n_att


# --- fence v2 (2026-07-25, the word after the 228 autopsy) -----------------

def multiplicity_ok(facs, text):
    """FORMAL REGISTER ONLY (the dialect's first register-conditional law;
    the register flag is provenance-carried, never inferred here — the
    caller owns the scoping). Givens carrying value X may not outnumber
    the text's statements of X."""
    from collections import Counter
    gv = Counter(f["value"] for f in facs if f.get("ftype") == "given")
    tv = Counter(int(x) for x in re.findall(r"\d+", text))
    return all(gv[v] <= tv.get(v, 0) for v in gv)


def attest_view_v2(facs, q, answer, text, m, solve_fn):
    """v2 = form law (E13 via the seated verifier) + multiplicity clause
    (formal register) + v1 strip-and-reforce. A view condemned by form
    cannot attest anything; a view wearing a literal more times than the
    text states it cannot attest; otherwise v1 applies."""
    from mycelium.arith3_verifier import verify
    if any(c == "A3.E13-selfloop" for c, _, _ in verify(facs)):
        return False
    if not multiplicity_ok(facs, text):
        return False
    return attest_view(facs, q, answer, text, m, solve_fn)


def attest_quorum_v2(view_parses, quorum_answer, text, m, solve_fn):
    winners = [(f, q) for f, q, a in view_parses if a == quorum_answer]
    n_att = sum(1 for f, q in winners
                if attest_view_v2(f, q, quorum_answer, text, m, solve_fn))
    return n_att > 0, len(winners), n_att


# --- fence v3 (2026-07-25, after the 92 autopsy): strip the wart, not the
# witness. ONE principle: the answer must be forced by the LAWFUL CORE.
# v2's error was condemning a view for carrying a wart (758/7500 wild
# graphs carry E13 self-loops; 81 of them answer RIGHT — a tautological
# unit-mul wart doesn't invalidate testimony that never rests on it).

def lawful_core(facs, text):
    """Strip (a) unattested givens, (b) E13 self-loop factors, (c) the
    entire value-class of multiplicity-excess givens (conservative: if
    givens carrying value X outnumber the text's statements of X, ALL
    X-givens go — the fence cannot know which copy is the misbinding)."""
    from collections import Counter
    from mycelium.arith3_verifier import verify
    lits = text_literals(text)
    loops = {j for c, j, _ in verify(facs) if c == "A3.E13-selfloop"}
    gv = Counter(f["value"] for f in facs if f.get("ftype") == "given")
    tv = Counter(int(x) for x in re.findall(r"\d+", text))
    excess = {v for v in gv if gv[v] > tv.get(v, 0)}
    return [f for j, f in enumerate(facs)
            if j not in loops
            and not (f.get("ftype") == "given"
                     and (f["value"] not in lits or f["value"] in excess))]


def attest_view_v3(facs, q, answer, text, m, solve_fn):
    core = lawful_core(facs, text)
    if len(core) == len(facs):
        return True  # already lawful; the emitted solve is the proof
    return solve_fn(core, q, {"n_vars": 24, "m": m}) == answer


def attest_quorum_v3(view_parses, quorum_answer, text, m, solve_fn):
    winners = [(f, q) for f, q, a in view_parses if a == quorum_answer]
    n_att = sum(1 for f, q in winners
                if attest_view_v3(f, q, quorum_answer, text, m, solve_fn))
    return n_att > 0, len(winners), n_att
