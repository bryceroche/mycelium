"""Graph execution engine for templates."""
import math
import operator
from typing import Dict, Any, Callable
from sympy import symbols, solve, sqrt, simplify, Rational, log, expand, factor
from sympy.parsing.sympy_parser import parse_expr


def _sympy_solve(equation_str: str, variable: str = "x"):
    """Solve an equation using SymPy."""
    var = symbols(variable)

    # Handle "expr = value" format
    if "=" in equation_str:
        lhs, rhs = equation_str.split("=", 1)
        expr = parse_expr(lhs.strip()) - parse_expr(rhs.strip())
    else:
        expr = parse_expr(equation_str)

    solutions = solve(expr, var)

    if len(solutions) == 1:
        return float(solutions[0])
    elif len(solutions) > 1:
        return [float(s) for s in solutions]
    return 0


def _complete_square(a: float, b: float, c: float):
    """
    Complete the square for ax² + bx + c.
    Returns (h, k, r²) where (x-h)² + (y-k)² = r² for circles.
    For general: a(x - h)² + k where h = -b/(2a), k = c - b²/(4a)
    """
    h = -b / (2 * a)
    k = c - (b * b) / (4 * a)
    r_squared = -k  # For circles, the constant term becomes r²
    return {"h": h, "k": k, "r_squared": r_squared, "r": math.sqrt(abs(r_squared))}


# Operation registry - maps op names to functions
OPERATIONS: Dict[str, Callable] = {
    # Basic arithmetic
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "pow": operator.pow,
    "neg": operator.neg,
    "abs": abs,

    # Math functions
    "sqrt": lambda x: math.sqrt(x) if isinstance(x, (int, float)) else sqrt(x),
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,

    # Comparisons (return 1 or 0)
    "eq": lambda a, b: 1 if a == b else 0,
    "lt": lambda a, b: 1 if a < b else 0,
    "gt": lambda a, b: 1 if a > b else 0,

    # Special operations for common patterns
    "sum_list": sum,
    "product_list": lambda xs: math.prod(xs),
    "mean": lambda xs: sum(xs) / len(xs),

    # SymPy operations
    "sympy_solve": _sympy_solve,
    "sympy_simplify": lambda expr: float(simplify(parse_expr(str(expr)))),
    "complete_square": _complete_square,
    "vieta_sum": lambda a, b: -b / a,      # Sum of roots = -b/a
    "vieta_product": lambda a, c: c / a,   # Product of roots = c/a
}


def execute_graph(graph: Dict, slot_values: Dict[str, Any]) -> Any:
    """
    Execute a computation graph with the given slot values.

    Args:
        graph: Dict with "nodes" and "edges"
            - nodes: list of variable names
            - edges: list of {"op": str, "inputs": list, "output": str}
        slot_values: Dict mapping slot names to values

    Returns:
        The value of the last computed node (usually "answer")
    """
    # Initialize context with slot values
    context = dict(slot_values)

    # Get edges
    edges = graph.get("edges", [])

    # Execute edges in order
    for edge in edges:
        op_name = edge.get("op")
        inputs = edge.get("inputs", [])
        output = edge.get("output")
        params = edge.get("params", {})  # Optional parameters

        # Get operation function
        if op_name not in OPERATIONS:
            raise ValueError(f"Unknown operation: {op_name}")

        op_func = OPERATIONS[op_name]

        # Resolve input values
        input_values = []
        for inp in inputs:
            if inp in context:
                input_values.append(context[inp])
            else:
                # Try to parse as literal
                try:
                    input_values.append(float(inp))
                except ValueError:
                    input_values.append(inp)

        # Execute operation
        if params:
            result = op_func(*input_values, **params)
        else:
            result = op_func(*input_values) if len(input_values) > 1 else op_func(input_values[0])

        # Store result
        context[output] = result

    # Return final answer (last output or "answer" if exists)
    if "answer" in context:
        return context["answer"]
    elif edges:
        return context[edges[-1]["output"]]
    return None


def validate_graph(graph: Dict, slots: list) -> bool:
    """Validate that a graph is well-formed."""
    nodes = set(graph.get("nodes", []))
    edges = graph.get("edges", [])

    # All slots should be in nodes
    for slot in slots:
        if slot not in nodes:
            return False

    # All edge inputs should be defined before use
    defined = set(slots)
    for edge in edges:
        for inp in edge.get("inputs", []):
            if inp not in defined and not _is_literal(inp):
                return False
        defined.add(edge.get("output"))

    return True


def _is_literal(value: str) -> bool:
    """Check if a string is a literal value."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False
