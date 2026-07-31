"""arith3_canonicalizer.py — SUPERSEDED (one-door sweep 2026-07-31):
zero external consumers; the knot-ID door is scripts/hash_audit_iso.py::canon
(39 importers, WL + exact verify, level-0). Kept for history; do not adopt.

Original header: — deterministic graph normalization (gut #69).

v0: canonical factor ordering only (no folds yet — folds arrive with
their provenance contract per the tightened law: fold MERGES loc sets).
Form-only; train == inference; the false-merge bar is constitutional
(zero, or the pass is corruption).
"""
import json


def _key(f):
    t = f.get("ftype", "?")
    return (t, f.get("name", ""), f.get("op", ""), f.get("sel", ""),
            tuple(f.get("args", [])), f.get("var", -1), f.get("result", -1),
            f.get("k", -1), f.get("a", -1), f.get("k1", -1), f.get("k2", -1),
            f.get("x", -1), f.get("y", -1), f.get("value", -1), f.get("p", -1))


def canonicalize(factors):
    """Deterministic order; content untouched. Idempotent by construction."""
    return sorted((dict(f) for f in factors), key=_key)


def digest(factors, query_var=None, n_vars=None):
    """Root-marked: the query is part of the graph's identity (the WL
    canon's own lesson — identical factor-sets asking different
    questions are different graphs)."""
    return json.dumps({"q": query_var, "n": n_vars,
                       "f": [_key(f) for f in canonicalize(factors)]},
                      sort_keys=True)
