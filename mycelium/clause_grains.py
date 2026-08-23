"""clause_grains.py — THE CLAUSE-GRAIN ORGAN (2026-08-21; the G53-kill
correction). Pseudo-sentence indices for single-sentence wild text:
boundaries at ". ", ", ", " and ", "; ", "? ". Multi-sentence text falls
back to the period grains the dialect head was raised on.

THE METER-DIVERGENCE LESSON (the first G53 fire's grave): this logic
was previously EMBEDDED as an escaped string in two call sites; the
escaping differed between the organ and its positive-presence check, so
the check certified a code path the organ never ran — 25k dialect rows
trained on comma-grains and the ring collapsed to 1093. THE LAW: a
check must CALL the organ it certifies, never reimplement it. Import
from here; never inline.
"""
import re

import numpy as np

_SENT_SPLIT = re.compile(r"(?<=\.)\s+")
_CLAUSE = re.compile(r"\. |, | and |; |\? ")


def is_single_sentence(text):
    return len(_SENT_SPLIT.split(text.strip())) <= 1


def clause_indices(text, offs, mask_row, sent_indices_fn, t_alg, sent_max):
    """Same contract as phase1_algebra_head.sent_indices; activates only
    on single-sentence text (dialect untouched by construction)."""
    if not is_single_sentence(text):
        return sent_indices_fn(text, offs, mask_row)
    bounds = [m.end() - 1 for m in _CLAUSE.finditer(text)]
    out = np.zeros((t_alg,), np.int32)
    ntk = int(mask_row.sum())
    arr = np.asarray(offs[:min(ntk, t_alg)], dtype=np.int64)
    if len(arr) and bounds:
        idx = np.searchsorted(np.asarray(bounds, dtype=np.int64),
                              arr[:, 0], "right")
        out[:len(arr)] = np.minimum(idx, sent_max - 1)
    return out


# ---- MATH-MODE GRAINS (2026-08-22; the Karpathy-patch upgrade) ----
# Equations are the wild register's true clauses. Seams = $-math-run
# boundaries UNION prose punctuation. Same fallback contract: multi-
# sentence text keeps period grains (dialect untouched by construction).
_MATH_RUN = re.compile(r"\$[^$]+\$")


def math_cut_positions(text):
    cuts = set()
    for m in _MATH_RUN.finditer(text):
        cuts.add(m.start())
        cuts.add(m.end())
    for m in _CLAUSE.finditer(text):
        cuts.add(m.end() - 1)
    return sorted(cuts)


def math_clause_indices(text, offs, mask_row, sent_indices_fn, t_alg, sent_max):
    """Same contract as clause_indices; math runs become their own grains."""
    if not is_single_sentence(text):
        return sent_indices_fn(text, offs, mask_row)
    bounds = math_cut_positions(text)
    out = np.zeros((t_alg,), np.int32)
    ntk = int(mask_row.sum())
    arr = np.asarray(offs[:min(ntk, t_alg)], dtype=np.int64)
    if len(arr) and bounds:
        idx = np.searchsorted(np.asarray(bounds, dtype=np.int64),
                              arr[:, 0], "right")
        out[:len(arr)] = np.minimum(idx, sent_max - 1)
    return out


# ---- THE CANONICALIZER (2026-08-22; Bryce's gut: "make the workspace
# cleaner") ---- One canonical LaTeX surface — the MINT's surface — so
# real wild text meets the mint-trained model halfway. Input-space only
# (the len_asc family); fixes the digit-adjacency class at the root
# (\frac 56 -> \frac{5}{6}; \log_464 -> \log_{4}64).
_FRAC_BARE = re.compile(r"\\frac\s*(\d)(\d)(?!\d)")   # audit r1 #11: exactly two digits
_FRAC_ONE = re.compile(r"\\frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}")
_LOG_BARE = re.compile(r"\\log_(\d)(\d+)")
_DISPLAY = re.compile(r"\\\[|\\\]|\$\$")
_WS = re.compile(r"[ \t]+")


def canonicalize(text):
    t = text.replace("\n", " ")
    t = _DISPLAY.sub("$", t)
    t = _FRAC_BARE.sub(r"\\frac{\1}{\2}", t)
    t = _FRAC_ONE.sub(r"\\frac{\1}{\2}", t)
    # _LOG_BARE rewrite REMOVED (audit r1 #11): \log_10100 is inherently
    # ambiguous (base 10 vs base 1) — a canonicalizer must never guess
    t = t.replace("\\!", "").replace("\\quad", " ").replace("\\,", " ")
    t = t.replace("\\cdot", "\\times")
    t = _WS.sub(" ", t).strip()
    return t
