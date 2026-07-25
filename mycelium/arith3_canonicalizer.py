"""arith3_canonicalizer.py — deterministic graph normalization (gut #69).

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
            f.get("k", -1), f.get("k1", -1), f.get("k2", -1),
            f.get("x", -1), f.get("y", -1), f.get("value", -1), f.get("p", -1))


def canonicalize(factors):
    """Deterministic order; content untouched. Idempotent by construction."""
    return sorted((dict(f) for f in factors), key=_key)


def digest(factors):
    return json.dumps([_key(f) for f in canonicalize(factors)], sort_keys=True)
