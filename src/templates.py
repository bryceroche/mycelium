"""
SymPy Templates for Mycelium v7

Deterministic execution of scaffold types that don't need a model:
- EXPAND: expand algebraic expressions
- SIMPLIFY: simplify/reduce expressions
- SOLVE: solve equations for unknowns
- COMPUTE: evaluate to numerical result
- ANSWER: passthrough (format final answer)

These templates take rough slot-filler outputs and execute deterministic SymPy operations.
"""

import signal
from typing import Optional, Any, Dict, Tuple
import sympy
from sympy import Symbol, symbols, expand, simplify, solve, Eq, N
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.parsing.latex import parse_latex


class TimeoutError(Exception):
    pass


def with_timeout(timeout_seconds: int = 5):
    """Decorator to add timeout to SymPy operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_seconds)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


def safe_parse(expr_str: str, local_dict: Optional[Dict] = None) -> Optional[sympy.Basic]:
    """
    Parse expression string to SymPy, trying multiple strategies.
    Returns None if all parsing fails.
    """
    if not expr_str or not expr_str.strip():
        return None

    expr_str = expr_str.strip()

    # Strategy 1: Try parse_latex for LaTeX expressions
    if '\\' in expr_str or '{' in expr_str:
        try:
            return parse_latex(expr_str)
        except Exception:
            pass

    # Strategy 2: Try standard parser with implicit multiplication
    transformations = standard_transformations + (implicit_multiplication_application,)
    try:
        return parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
    except Exception:
        pass

    # Strategy 3: Try sympify directly
    try:
        return sympy.sympify(expr_str, locals=local_dict)
    except Exception:
        pass

    return None


@with_timeout(5)
def template_expand(expr_str: str, local_dict: Optional[Dict] = None) -> Tuple[bool, Any, str]:
    """
    EXPAND template: Expand algebraic expression.

    Args:
        expr_str: Expression to expand (e.g., "(x+1)^2" or "(a+b)(a-b)")
        local_dict: Optional symbol definitions

    Returns:
        (success: bool, result: sympy expr or None, message: str)
    """
    expr = safe_parse(expr_str, local_dict)
    if expr is None:
        return False, None, f"Failed to parse: {expr_str}"

    try:
        result = expand(expr)
        return True, result, str(result)
    except Exception as e:
        return False, None, f"Expand failed: {e}"


@with_timeout(5)
def template_simplify(expr_str: str, local_dict: Optional[Dict] = None) -> Tuple[bool, Any, str]:
    """
    SIMPLIFY template: Simplify/reduce expression.

    Args:
        expr_str: Expression to simplify
        local_dict: Optional symbol definitions

    Returns:
        (success: bool, result: sympy expr or None, message: str)
    """
    expr = safe_parse(expr_str, local_dict)
    if expr is None:
        return False, None, f"Failed to parse: {expr_str}"

    try:
        result = simplify(expr)
        return True, result, str(result)
    except Exception as e:
        return False, None, f"Simplify failed: {e}"


@with_timeout(5)
def template_solve(equation_str: str, variable: str = None, local_dict: Optional[Dict] = None) -> Tuple[bool, Any, str]:
    """
    SOLVE template: Solve equation for variable.

    Args:
        equation_str: Equation to solve. Can be:
            - "x^2 - 4 = 0" (explicit equation)
            - "x^2 - 4" (implicit = 0)
        variable: Variable to solve for (auto-detected if None)
        local_dict: Optional symbol definitions

    Returns:
        (success: bool, result: list of solutions, message: str)
    """
    # Handle explicit equation
    if '=' in equation_str:
        parts = equation_str.split('=')
        if len(parts) == 2:
            lhs = safe_parse(parts[0].strip(), local_dict)
            rhs = safe_parse(parts[1].strip(), local_dict)
            if lhs is None or rhs is None:
                return False, None, f"Failed to parse equation: {equation_str}"
            expr = lhs - rhs
        else:
            return False, None, f"Invalid equation format: {equation_str}"
    else:
        expr = safe_parse(equation_str, local_dict)
        if expr is None:
            return False, None, f"Failed to parse: {equation_str}"

    # Determine variable to solve for
    if variable:
        var = Symbol(variable)
    else:
        free_syms = expr.free_symbols
        if len(free_syms) == 0:
            return False, None, "No variables in expression"
        elif len(free_syms) == 1:
            var = list(free_syms)[0]
        else:
            # Default to x if present, otherwise first symbol
            var = Symbol('x') if Symbol('x') in free_syms else list(free_syms)[0]

    try:
        solutions = solve(expr, var)
        if not solutions:
            return True, [], "No solutions found"
        return True, solutions, str(solutions)
    except Exception as e:
        return False, None, f"Solve failed: {e}"


@with_timeout(5)
def template_compute(expr_str: str, substitutions: Optional[Dict[str, float]] = None,
                     local_dict: Optional[Dict] = None) -> Tuple[bool, Any, str]:
    """
    COMPUTE template: Evaluate expression to numerical result.

    Args:
        expr_str: Expression to compute
        substitutions: Dict of variable -> value substitutions
        local_dict: Optional symbol definitions

    Returns:
        (success: bool, result: number or None, message: str)
    """
    expr = safe_parse(expr_str, local_dict)
    if expr is None:
        return False, None, f"Failed to parse: {expr_str}"

    try:
        if substitutions:
            sub_dict = {Symbol(k): v for k, v in substitutions.items()}
            expr = expr.subs(sub_dict)

        # Evaluate numerically
        result = N(expr)

        # Try to get exact rational if possible
        exact = sympy.nsimplify(result)
        if exact.is_Rational or exact.is_Integer:
            return True, exact, str(exact)

        return True, result, str(result)
    except Exception as e:
        return False, None, f"Compute failed: {e}"


def template_answer(value: Any) -> Tuple[bool, Any, str]:
    """
    ANSWER template: Passthrough - format final answer.

    Args:
        value: The computed answer (can be SymPy expr, number, or string)

    Returns:
        (success: bool, result: formatted answer, message: str)
    """
    if value is None:
        return False, None, "No value provided"

    # If it's a SymPy expression, try to simplify/format
    if isinstance(value, sympy.Basic):
        try:
            simplified = sympy.nsimplify(value)
            return True, simplified, str(simplified)
        except Exception:
            return True, value, str(value)

    # Otherwise just return as-is
    return True, value, str(value)


# Template registry for dispatch
TEMPLATES = {
    'EXPAND': template_expand,
    'SIMPLIFY': template_simplify,
    'SOLVE': template_solve,
    'COMPUTE': template_compute,
    'ANSWER': template_answer,
}


def execute_template(scaffold_type: str, expression: str,
                     variable: str = None,
                     substitutions: Optional[Dict[str, float]] = None,
                     local_dict: Optional[Dict] = None) -> Tuple[bool, Any, str]:
    """
    Execute a scaffold template.

    Args:
        scaffold_type: One of EXPAND, SIMPLIFY, SOLVE, COMPUTE, ANSWER
        expression: The expression/equation to process
        variable: For SOLVE - which variable to solve for
        substitutions: For COMPUTE - variable substitutions
        local_dict: Symbol definitions

    Returns:
        (success: bool, result, message: str)
    """
    scaffold_type = scaffold_type.upper()

    if scaffold_type not in TEMPLATES:
        return False, None, f"Unknown scaffold type: {scaffold_type}"

    template = TEMPLATES[scaffold_type]

    try:
        if scaffold_type == 'SOLVE':
            return template(expression, variable, local_dict)
        elif scaffold_type == 'COMPUTE':
            return template(expression, substitutions, local_dict)
        elif scaffold_type == 'ANSWER':
            return template(expression)
        else:
            return template(expression, local_dict)
    except TimeoutError as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, f"Template execution failed: {e}"


if __name__ == "__main__":
    # Quick tests
    print("Testing EXPAND:")
    print(template_expand("(x+1)**2"))

    print("\nTesting SIMPLIFY:")
    print(template_simplify("(x**2 - 1)/(x - 1)"))

    print("\nTesting SOLVE:")
    print(template_solve("x**2 - 4 = 0"))

    print("\nTesting COMPUTE:")
    print(template_compute("sqrt(9) + 2**3"))

    print("\nTesting ANSWER:")
    print(template_answer(sympy.Rational(11, 2)))
