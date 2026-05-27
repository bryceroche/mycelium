"""v77 Phase 1 — SymPy DAG executor.

Parses a DAG string like:

    x0 = 50 + 15 ; x1 = x0 * 2 ; answer = x1

and executes it via SymPy. Returns a float (the value of `answer`) or None
if the DAG is malformed / undefined / divides by zero / contains residual
symbolic variables.

DAG syntax:
    DAG       := STATEMENT (';' STATEMENT)* ';' 'answer' '=' EXPR
    STATEMENT := VAR '=' EXPR
    VAR       := 'x' DIGIT+                       # x0, x1, ...
    EXPR      := Python-ish arithmetic over numbers + previously-defined VARs

The first statement in the DAG cannot reference any VAR. The Nth statement
can reference any of the previous N-1 VARs. The final `answer =` is required.

Usage:
    from v77_sympy_eval import dag_to_answer
    assert dag_to_answer("x0 = 50 + 15 ; x1 = x0 * 2 ; answer = x1") == 130.0

The __main__ block runs 5 hardcoded tests and prints the verdict for each.
"""
from __future__ import annotations

import re
import sys
from typing import Optional

try:
    import sympy
    from sympy import sympify, Float, Integer, Rational, Symbol
except ImportError:
    print("ERROR: sympy not installed. Run: .venv/bin/pip install sympy", file=sys.stderr)
    sys.exit(1)


_VAR_PAT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _is_var_name(s: str) -> bool:
    return bool(_VAR_PAT.match(s))


def _split_statements(dag_str: str) -> list[str]:
    """Split on ';' and strip whitespace. Empty parts are dropped."""
    return [p.strip() for p in dag_str.split(";") if p.strip()]


def _parse_statement(stmt: str) -> Optional[tuple[str, str]]:
    """Return (lhs_name, rhs_expr_str) or None if malformed."""
    if "=" not in stmt:
        return None
    # Use a SINGLE split on the first '=' — RHS may contain '==' etc.
    lhs, _, rhs = stmt.partition("=")
    lhs = lhs.strip()
    rhs = rhs.strip()
    if not lhs or not rhs:
        return None
    if not _is_var_name(lhs):
        return None
    return (lhs, rhs)


def dag_to_answer(dag_str: str) -> Optional[float]:
    """Parse and execute a DAG string. Return the value of `answer`, or None.

    Steps:
      1. Strip; bail on empty.
      2. Split on ';' into statements.
      3. For each statement, split on '='; LHS is variable name, RHS is expr.
      4. Use SymPy to parse RHS, substituting all previously-defined variables.
      5. The value must be a finite Number (no residual symbols) — else bail.
      6. The final statement must be `answer = <expr>`; return its value.

    Returns None on any failure (malformed, undefined ref, div0, residual symbol).
    """
    if not dag_str or not dag_str.strip():
        return None
    statements = _split_statements(dag_str)
    if not statements:
        return None

    env: dict[str, "sympy.Expr"] = {}
    last_name = None

    for stmt in statements:
        parsed = _parse_statement(stmt)
        if parsed is None:
            return None
        name, rhs = parsed
        try:
            # `locals` arg lets sympify see x0, x1, ... as the bound values
            value = sympify(rhs, locals=env, evaluate=True)
        except (sympy.SympifyError, SyntaxError, TypeError, ZeroDivisionError):
            return None
        except Exception:
            # SymPy raises a wide zoo: catch anything that isn't a clean number
            return None

        # If any free symbols remain, the DAG references an undefined name
        try:
            free = value.free_symbols
        except AttributeError:
            return None
        if free:
            return None

        # Substitute all bound names just in case (defensive; sympify(locals=...)
        # already does this, but if RHS is a previously-bound variable name
        # alone, we need to dereference)
        try:
            value = value.subs(env)
        except Exception:
            return None
        try:
            free = value.free_symbols
        except AttributeError:
            return None
        if free:
            return None

        env[name] = value
        last_name = name

    # Require final variable to be 'answer'
    if last_name != "answer":
        # Forgiving: if last_name isn't 'answer' but 'answer' IS bound, return that
        if "answer" in env:
            value = env["answer"]
        else:
            return None
    else:
        value = env["answer"]

    # Cast to float; bail if NaN/inf or not coercible
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f == float("inf") or f == float("-inf"):
        return None
    return f


def _verdict(label: str, dag: str, expected: Optional[float]):
    actual = dag_to_answer(dag)
    if expected is None:
        passed = actual is None
        exp_str = "None (expected failure)"
    else:
        # 1e-6 tolerance for float comparison
        passed = actual is not None and abs(actual - expected) < 1e-6
        exp_str = f"{expected}"
    actual_str = f"{actual}" if actual is not None else "None"
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}: dag = {dag!r}")
    print(f"           expected={exp_str}  actual={actual_str}")
    return passed


def main():
    print("=== v77 SymPy DAG eval — self-test ===")
    print()
    tests = [
        ("simple add",       "x0 = 50 + 15 ; answer = x0",                                        65.0),
        ("two-step",         "x0 = 50 + 15 ; x1 = x0 * 2 ; answer = x1",                          130.0),
        ("five-step chain",  "x0 = 100 / 2 ; x1 = 15 * 2 ; x2 = x0 + 15 ; x3 = x2 + x1 ; "
                             "x4 = 100 - x3 ; answer = x4",                                       5.0),
        ("undefined var",    "x0 = 50 + 15 ; answer = x99",                                       None),
        ("div by zero",      "x0 = 10 / 0 ; answer = x0",                                         None),
    ]
    passed = 0
    for label, dag, expected in tests:
        if _verdict(label, dag, expected):
            passed += 1
    print()
    print(f"=== {passed}/{len(tests)} passed ===")
    if passed != len(tests):
        sys.exit(1)


if __name__ == "__main__":
    main()
