"""canonicalizer.py — THE CANONICALIZER (2026-08-03; #69's standing
build order, bars pinned blind 2026-07-25; CAMBIUM's ring-tidying).

v0 HISTORY, honest: the first build hand-rolled canonical variable
relabeling (P3) and THE PINNED BAR CAUGHT IT — 4 false merges +
303/1000 invariance failures on the mix gold fixture (order-dependent
labeling under symmetry is not a canonical form). P3 DEMOTED by its
own bar, same session. The one-door law then pointed at the authority
that already existed: **scripts/hash_audit_iso.py::canon** — the
knot-ID door (WL coloring + level-0 macro expansion + query
distinguished + values semantic: giv embeds the literal, mod/fdiv
carry k, pct carries p; 39 importers; exam 5/5). Identity questions
route through the door; this module is the ADAPTER plus the
readable-form passes.

Passes:
  identity  — canon digest via the door (the ONLY equivalence judge)
  P1 (form) — canonical factor ordering for human-readable output
  P2 (form) — commutative arg sort (add/mul rel only)
NO CONSTANT FOLDS (folds enter only with their own false-merge proof;
fold MERGES loc sets per the tightened law when they arrive).

Bar artifact (the orphaned-verdict guard's pointer):
.cache/canonicalizer_bar.json — fired by scripts/canonicalizer_bar.py.
Paraphrase-collapse read (5-view parses): PENDING-GPU.
"""
import sys
import json

_COMMUTATIVE_REL_OPS = {"add", "mul"}
_STRIP = {"spans", "span", "loc"}


def _clean(f):
    return {k: v for k, v in f.items() if k not in _STRIP}


def _norm_args(f):
    f = dict(f)
    if f.get("ftype") == "rel" and f.get("op") in _COMMUTATIVE_REL_OPS \
            and isinstance(f.get("args"), (list, tuple)):
        f["args"] = sorted(f["args"])
    return f


def canonical_digest(factors, query, n_vars=24):
    """THE identity judge: the knot-ID door's WL digest. Two parses are
    equivalent iff digests match (confirm with the door's verify_iso
    for exactness where a verdict rides on it)."""
    if 'scripts' not in str(sys.path[:3]):
        sys.path.insert(0, 'scripts')
    from hash_audit_iso import canon
    row = {"factors": [_clean(f) for f in factors],
           "query_var": query, "n_vars": n_vars}
    digest, _col = canon(row)
    return digest


def canonical_form(factors, query):
    """Readable canonical form (P1+P2) — for sheets and eyes, NOT for
    identity (identity is the digest's job)."""
    fs = sorted((_norm_args(_clean(f)) for f in factors),
                key=lambda f: json.dumps(f, sort_keys=True))
    return fs, query
