"""
SymPy Oracle for Mycelium v7.

The incorruptible layer. Parses telegram expressions and executes them.
Routes to the right SymPy parser based on expression format.
Not reasoning — plumbing.

Handles:
    - Implicit multiplication (2x → 2*x)
    - Caret notation (x^2 → x**2)
    - Equations (x^2+y^2=90 → Eq(lhs, rhs))
    - LaTeX (\frac{1}{2} → Rational(1,2))
    - Unicode (× → *, √ → sqrt)
    - _prev references (chain results between steps)
"""

import signal
from typing import Optional, Any, List, Dict

import sympy
from sympy import Eq, simplify, sympify, Symbol
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    implicit_application,
    convert_xor,
)

RELAXED_TRANSFORMATIONS = (
    standard_transformations +
    (implicit_multiplication_application,) +
    (implicit_application,) +
    (convert_xor,)
)

TIMEOUT_SECONDS = 5

VERBS = {"GIVEN", "EVAL", "SOLVE", "EXPAND", "SIMPLIFY", "SUBS", "APPLY", "ANSWER"}


# ─────────────────────────────────────────────────────────────
# Timeout handler
# ─────────────────────────────────────────────────────────────

class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("SymPy execution timed out")


# ─────────────────────────────────────────────────────────────
# Expression parser
# ─────────────────────────────────────────────────────────────

def normalize_unicode(expr: str) -> str:
    """Replace unicode math symbols with ASCII equivalents."""
    return (expr
            .replace('×', '*')
            .replace('·', '*')
            .replace('√', 'sqrt')
            .replace('−', '-')
            .replace('≤', '<=')
            .replace('≥', '>=')
            .replace('≠', '!=')
            .replace('π', 'pi'))


def parse_relaxed(expr: str) -> Any:
    """Parse with implicit multiplication and function application."""
    return parse_expr(expr, transformations=RELAXED_TRANSFORMATIONS, evaluate=False)


def parse_telegram_expr(expr: str) -> Any:
    """
    Route to the right SymPy parser. Not reasoning — plumbing.

    Handles equations, LaTeX, unicode, and relaxed notation.
    """
    if not expr or not expr.strip():
        return None

    expr = expr.strip()
    expr = normalize_unicode(expr)

    # LaTeX expressions
    if '\\' in expr:
        try:
            from sympy.parsing.latex import parse_latex
            return parse_latex(expr)
        except Exception:
            pass  # fall through to other parsers

    # Equations: split on = and wrap in Eq()
    if '=' in expr and expr.count('=') == 1 and not any(
        op in expr for op in ['==', '!=', '<=', '>=']
    ):
        lhs, rhs = expr.split('=', 1)
        try:
            left = parse_relaxed(lhs.strip())
            right = parse_relaxed(rhs.strip())
            return Eq(left, right)
        except Exception:
            pass

    # Standard relaxed parse
    try:
        return parse_relaxed(expr)
    except Exception:
        pass

    # Last resort: sympify
    try:
        return sympify(expr, evaluate=False)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Telegram executor
# ─────────────────────────────────────────────────────────────

def execute_telegram(telegram: str, previous_results: List[Any],
                     timeout: int = TIMEOUT_SECONDS) -> Dict:
    """
    Execute a single telegram instruction with SymPy.

    Returns dict with:
        success: bool
        result: SymPy expression or None
        error: error message or None
    """
    parts = telegram.strip().split(None, 1)
    if not parts:
        return {"success": False, "result": None, "error": "empty telegram"}

    verb = parts[0].upper()
    args_str = parts[1] if len(parts) > 1 else ""

    if verb not in VERBS:
        return {"success": False, "result": None, "error": f"invalid verb: {verb}"}

    # Resolve _prev references
    args_str = resolve_prev(args_str, previous_results)

    # Set timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        result = _execute_verb(verb, args_str, previous_results)
        signal.alarm(0)
        return {"success": True, "result": result, "error": None}
    except TimeoutError:
        return {"success": False, "result": None, "error": "timeout"}
    except Exception as e:
        signal.alarm(0)
        return {"success": False, "result": None, "error": str(e)[:200]}
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def resolve_prev(args_str: str, previous_results: List[Any]) -> str:
    """Replace _prev with the last successful result."""
    if "_prev" not in args_str:
        return args_str

    if not previous_results:
        return args_str

    # Find last non-None result
    prev_result = None
    for r in reversed(previous_results):
        if r is not None:
            prev_result = r
            break

    if prev_result is None:
        return args_str

    return args_str.replace("_prev", str(prev_result))


def _execute_verb(verb: str, args_str: str, previous_results: List[Any]) -> Any:
    """Execute a specific verb with parsed arguments."""

    if verb == "GIVEN":
        # Parse and return the expression/equation
        result = parse_telegram_expr(args_str)
        if result is None:
            raise ValueError(f"Cannot parse GIVEN: {args_str}")
        return result

    elif verb == "EVAL":
        # Parse and evaluate
        expr = parse_telegram_expr(args_str)
        if expr is None:
            raise ValueError(f"Cannot parse EVAL: {args_str}")
        result = sympy.simplify(expr)
        # Try to get a numeric value
        try:
            result = sympy.nsimplify(result)
        except Exception:
            pass
        return result

    elif verb == "SOLVE":
        # SOLVE formats:
        #   SOLVE Eq(lhs, rhs)     - complete equation, sympy infers variable
        #   SOLVE expr             - expression = 0, sympy infers variable
        #   SOLVE equation var     - explicit variable to solve for

        # First try parsing the entire args as an equation/expression
        eq_expr = parse_telegram_expr(args_str)
        if eq_expr is not None:
            # Successfully parsed the whole thing - let sympy infer the variable
            solutions = sympy.solve(eq_expr)
        else:
            # Fall back to "equation variable" format
            solve_parts = args_str.rsplit(None, 1)
            if len(solve_parts) == 2:
                eq_str, var_str = solve_parts
                var = Symbol(var_str)
                eq_expr = parse_telegram_expr(eq_str)
                if eq_expr is None:
                    raise ValueError(f"Cannot parse SOLVE equation: {eq_str}")
                solutions = sympy.solve(eq_expr, var)
            elif len(solve_parts) == 1:
                raise ValueError(f"Cannot parse SOLVE: {solve_parts[0]}")
            else:
                raise ValueError(f"Cannot parse SOLVE args: {args_str}")

        if isinstance(solutions, list):
            return solutions if len(solutions) > 1 else solutions[0] if solutions else None
        return solutions

    elif verb == "EXPAND":
        expr = parse_telegram_expr(args_str)
        if expr is None:
            raise ValueError(f"Cannot parse EXPAND: {args_str}")
        return sympy.expand(expr)

    elif verb == "SIMPLIFY":
        expr = parse_telegram_expr(args_str)
        if expr is None:
            raise ValueError(f"Cannot parse SIMPLIFY: {args_str}")
        return sympy.simplify(expr)

    elif verb == "SUBS":
        # SUBS expression old new
        subs_parts = args_str.split()
        if len(subs_parts) >= 3:
            expr_str = subs_parts[0]
            old_str = subs_parts[1]
            new_str = subs_parts[2]

            expr = parse_telegram_expr(expr_str)
            old = parse_telegram_expr(old_str)
            new = parse_telegram_expr(new_str)

            if expr is None or old is None or new is None:
                raise ValueError(f"Cannot parse SUBS args: {args_str}")

            return expr.subs(old, new)
        else:
            raise ValueError(f"SUBS needs 3+ args, got: {args_str}")

    elif verb == "APPLY":
        # APPLY theorem_name args...
        # For now, return the parsed expression
        expr = parse_telegram_expr(args_str)
        if expr is None:
            # Try parsing just the math part (skip theorem name)
            apply_parts = args_str.split(None, 1)
            if len(apply_parts) > 1:
                expr = parse_telegram_expr(apply_parts[1])
        return expr

    elif verb == "ANSWER":
        expr = parse_telegram_expr(args_str)
        if expr is not None:
            return sympy.simplify(expr)
        return None

    else:
        raise ValueError(f"Unknown verb: {verb}")


# ─────────────────────────────────────────────────────────────
# Execute full telegram sequence
# ─────────────────────────────────────────────────────────────

def execute_sequence(telegrams: List[str],
                     timeout: int = TIMEOUT_SECONDS) -> Dict:
    """
    Execute a full sequence of telegrams, chaining results via _prev.

    Returns:
        success: bool (did every step execute?)
        results: list of per-step results
        final_answer: the last result
        execution_rate: fraction of steps that executed
        errors: list of error messages
    """
    results = []
    errors = []
    previous_results = []

    for i, telegram in enumerate(telegrams):
        exec_result = execute_telegram(telegram, previous_results, timeout)

        results.append({
            "step": i,
            "telegram": telegram,
            "success": exec_result["success"],
            "result": str(exec_result["result"]) if exec_result["result"] is not None else None,
            "error": exec_result["error"],
        })

        if exec_result["success"]:
            previous_results.append(exec_result["result"])
        else:
            previous_results.append(None)
            errors.append(f"Step {i} ({telegram}): {exec_result['error']}")

    n_success = sum(1 for r in results if r["success"])
    final_answer = None
    for r in reversed(results):
        if r["success"] and r["result"] is not None:
            final_answer = r["result"]
            break

    return {
        "success": n_success == len(telegrams),
        "results": results,
        "final_answer": final_answer,
        "execution_rate": n_success / max(len(telegrams), 1),
        "n_success": n_success,
        "n_total": len(telegrams),
        "errors": errors,
    }


def compare_answers(predicted: Any, gold: str,
                    timeout: int = TIMEOUT_SECONDS) -> bool:
    """Compare predicted answer with gold using sympy.simplify(a - b) == 0."""
    if predicted is None:
        return False

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        gold_expr = parse_telegram_expr(gold)
        if gold_expr is None:
            signal.alarm(0)
            return str(predicted).strip() == gold.strip()

        diff = sympy.simplify(predicted - gold_expr)
        signal.alarm(0)
        return diff == 0
    except TimeoutError:
        return False
    except Exception:
        signal.alarm(0)
        return str(predicted).strip() == gold.strip()
    finally:
        signal.signal(signal.SIGALRM, old_handler)
