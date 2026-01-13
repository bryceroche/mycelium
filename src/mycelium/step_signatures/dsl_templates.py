"""DSL Templates and Inference for Auto-Assignment.

When creating new signatures, we auto-assign DSL scripts based on
step_type and description patterns. This enables immediate execution
without waiting for learning.
"""

import json
import re
from typing import Optional

# =============================================================================
# DSL Templates by Step Type
# =============================================================================

DSL_TEMPLATES = {
    "identity": {"type": "math", "script": "x", "params": ["x"], "purpose": "Return single value unchanged"},
    "extract_value": {"type": "math", "script": "x", "params": ["x"], "purpose": "Extract and return a value"},
    "compute_sum": {"type": "math", "script": "sum_all()", "params": [], "purpose": "Sum all numeric inputs"},
    "compute_product": {"type": "math", "script": "a * b", "params": ["a", "b"], "purpose": "Multiply two numbers"},
    "compute_difference": {"type": "math", "script": "a - b", "params": ["a", "b"], "purpose": "Subtract b from a"},
    "compute_quotient": {"type": "math", "script": "a / b", "params": ["a", "b"], "purpose": "Divide a by b"},
    "compute_power": {"type": "math", "script": "base ** exponent", "params": ["base", "exponent"], "purpose": "Raise base to power"},
    "compute_factorial": {"type": "math", "script": "factorial(n)", "params": ["n"], "purpose": "Calculate n!"},
    "compute_sqrt": {"type": "math", "script": "sqrt(x)", "params": ["x"], "purpose": "Square root"},
    "compute_modulo": {"type": "math", "script": "a % b", "params": ["a", "b"], "purpose": "Remainder"},
    "compute_gcd": {"type": "math", "script": "gcd(a, b)", "params": ["a", "b"], "purpose": "Greatest common divisor"},
    "compute_lcm": {"type": "math", "script": "lcm(a, b)", "params": ["a", "b"], "purpose": "Least common multiple"},
    "compute_area": {"type": "math", "script": "length * width", "params": ["length", "width"], "purpose": "Calculate area"},
    "compute_average": {"type": "math", "script": "(a + b) / 2", "params": ["a", "b"], "purpose": "Calculate average"},
    "compute_probability": {"type": "decompose", "script": "compute_probability", "params": ["favorable", "total"], "purpose": "Calculate probability"},
    "simplify_expression": {"type": "sympy", "script": "simplify(expr)", "params": ["expr"], "purpose": "Simplify expression"},
    "solve_equation": {"type": "sympy", "script": "solve(equation, x)", "params": ["equation"], "purpose": "Solve equation"},
    "factor_expression": {"type": "sympy", "script": "factor(expr)", "params": ["expr"], "purpose": "Factor expression"},
    "evaluate_expression": {"type": "math", "script": "eval(expr)", "params": ["expr"], "purpose": "Evaluate expression"},
    "compute_angle": {"type": "math", "script": "degrees", "params": ["degrees"], "purpose": "Angle calculation"},
    "count_combinations": {"type": "math", "script": "factorial(n) / (factorial(r) * factorial(n - r))", "params": ["n", "r"], "purpose": "n choose r"},
    "count_permutations": {"type": "math", "script": "factorial(n) / factorial(n - r)", "params": ["n", "r"], "purpose": "P(n,r)"},
}

# =============================================================================
# Inference Patterns (regex → DSL)
# =============================================================================

DSL_INFERENCE_PATTERNS = [
    (r"combine.*result|final.*answer|synthesize", {"type": "decompose", "script": "synthesize_results", "params": ["results"], "purpose": "Combine results"}),
    (r"coordinate|point.*\(|define.*point", {"type": "sympy", "script": "Point(x, y)", "params": ["x", "y"], "purpose": "Coordinate point"}),
    (r"substitut|plug.*in|replace.*with", {"type": "sympy", "script": "expr.subs(var, value)", "params": ["expr", "var", "value"], "purpose": "Substitution"}),
    (r"find.*minimum|find.*maximum|minimize|maximize|min.*value|max.*value", {"type": "sympy", "script": "solve(diff(expr, x), x)", "params": ["expr"], "purpose": "Find min/max"}),
    (r"define.*constraint|constraint|given.*condition", {"type": "decompose", "script": "extract_constraints", "params": ["problem"], "purpose": "Extract constraints"}),
    (r"express.*in terms|write.*as|rewrite", {"type": "sympy", "script": "solve(eq, var)", "params": ["eq", "var"], "purpose": "Express in terms of"}),
    (r"find.*equation|equation of|derive.*equation", {"type": "sympy", "script": "Eq(lhs, rhs)", "params": ["lhs", "rhs"], "purpose": "Find equation"}),
    (r"identify|extract|determine.*value|find.*value", {"type": "math", "script": "x", "params": ["x"], "purpose": "Extract and return value"}),
    (r"magnitude|absolute|modulus", {"type": "sympy", "script": "Abs(z)", "params": ["z"], "purpose": "Magnitude"}),
    (r"argument|angle.*of|arg\(", {"type": "sympy", "script": "arg(z)", "params": ["z"], "purpose": "Argument/angle"}),
    (r"range|interval|bounds|between", {"type": "decompose", "script": "find_range", "params": ["expr", "var"], "purpose": "Find range"}),
    (r"solve for|find.*n\b|find.*x\b", {"type": "sympy", "script": "solve(eq, var)", "params": ["eq", "var"], "purpose": "Solve for variable"}),
    (r"critical point|derivative.*zero", {"type": "sympy", "script": "solve(diff(f, x), x)", "params": ["f"], "purpose": "Critical points"}),
    (r"relationship|connection|relate", {"type": "decompose", "script": "find_relationship", "params": ["a", "b"], "purpose": "Find relationship"}),
]


def infer_dsl_for_signature(step_type: str, description: str) -> tuple[Optional[str], str]:
    """Infer DSL script and type for a new signature.

    First checks DSL_TEMPLATES by step_type, then falls back to
    pattern matching on description.

    Args:
        step_type: The signature's step type (e.g., "compute_sum")
        description: The step description text

    Returns:
        Tuple of (dsl_script_json, dsl_type) or (None, "math") if no DSL
    """
    # First: check if step_type has a template
    if step_type in DSL_TEMPLATES:
        template = DSL_TEMPLATES[step_type]
        dsl_script = json.dumps({
            "type": template["type"],
            "script": template["script"],
            "params": template["params"],
        })
        return dsl_script, template["type"]

    # Second: try pattern matching on description
    desc_lower = description.lower()
    for pattern, template in DSL_INFERENCE_PATTERNS:
        if re.search(pattern, desc_lower):
            dsl_script = json.dumps({
                "type": template["type"],
                "script": template["script"],
                "params": template["params"],
            })
            return dsl_script, template["type"]

    # Default fallback: guidance DSL for decomposition
    fallback = {
        "type": "decompose",
        "script": "reason_step",
        "params": ["context"],
        "purpose": f"Execute: {description[:50]}",
    }
    return json.dumps(fallback), "decompose"
