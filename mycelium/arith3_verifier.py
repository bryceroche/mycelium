"""arith3_verifier.py — the dialect's strict structural verifier (gut #68).

Deterministic passes over a compiled graph (list of factor dicts).
Every rejection carries CODE + SLOT. Constrains FORM only — no semantic
imports, by law. The spec is docs/ARITH3_DIALECT.md; changes travel in
the same transaction.
"""
K_VARS = 24
OPS = {"add", "mul", "sub"}
SELS = {"smaller", "larger", "even", "odd"}
FTYPES = {"given", "rel", "mod", "fdiv", "sel", "pct", "macro"}
MACROS = {"OP_APPLY", "FRAC_OF"}


def _var_ok(v):
    return isinstance(v, int) and 0 <= v < K_VARS


def verify(factors):
    """Returns list of (code, slot, detail). Empty list == well-formed."""
    errs = []
    if not factors:
        return [("A3.E09-empty", -1, "no factors")]
    for j, f in enumerate(factors):
        t = f.get("ftype")
        if t not in FTYPES:
            errs.append(("A3.E01-ftype", j, str(t))); continue
        if t == "given":
            if not _var_ok(f.get("var")):
                errs.append(("A3.E04-varrange", j, f"var={f.get('var')}"))
            v = f.get("value")
            if not (isinstance(v, int) and 0 <= v <= 999):
                errs.append(("A3.E05-value", j, f"value={v}"))
        elif t == "rel":
            if f.get("op") not in OPS:
                errs.append(("A3.E02-op", j, str(f.get("op"))))
            args = f.get("args", [])
            if len(args) != 2:
                errs.append(("A3.E03-arity", j, f"arity={len(args)}"))
            for a in list(args) + [f.get("result")]:
                if not _var_ok(a):
                    errs.append(("A3.E04-varrange", j, f"ref={a}")); break
            if f.get("result") in list(args):
                errs.append(("A3.E13-selfloop", j,
                             f"result={f.get('result')} in args"))
        elif t in ("mod", "fdiv"):
            k = f.get("k")
            if not (isinstance(k, int) and k >= 2):
                errs.append(("A3.E06-kdegenerate", j, f"k={k}"))
            for a in (f.get("var"), f.get("result")):
                if not _var_ok(a):
                    errs.append(("A3.E04-varrange", j, f"ref={a}")); break
        elif t == "sel":
            if f.get("sel") not in SELS:
                errs.append(("A3.E07-sel", j, str(f.get("sel"))))
            for a in list(f.get("args", [])) + [f.get("result")]:
                if not _var_ok(a):
                    errs.append(("A3.E04-varrange", j, f"ref={a}")); break
            if f.get("result") in list(f.get("args", [])):
                errs.append(("A3.E13-selfloop", j,
                             f"result={f.get('result')} in args"))
        elif t == "pct":
            if "args" not in f or "p" not in f:
                errs.append(("A3.E10-fields", j, "pct missing args/p"))
        elif t == "macro":
            nm = f.get("name")
            if nm not in MACROS:
                errs.append(("A3.E08-macro", j, str(nm)))
            elif nm == "FRAC_OF":
                if not (isinstance(f.get("a"), int) and f["a"] >= 1
                        and isinstance(f.get("k"), int) and f["k"] >= 2):
                    errs.append(("A3.E08-macro", j, f"a={f.get('a')},k={f.get('k')}"))
            elif nm == "OP_APPLY":
                if f.get("op") not in ("add", "sub") or \
                   not (isinstance(f.get("k1"), int) and f["k1"] >= 1 and
                        isinstance(f.get("k2"), int) and f["k2"] >= 1):
                    errs.append(("A3.E08-macro", j, "op/k1/k2"))
        if "loc" in f:
            lc = f["loc"]
            if isinstance(lc, dict) and lc.get("derived") is not None \
               and not lc.get("parents"):
                errs.append(("A3.E12-provenance", j, "derived with empty parents"))
    return errs
