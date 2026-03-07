"""
Mycelium Lambda Template Library

Each template is a callable that takes typed operands and returns a SymPy result.
The factor graph selects which template to apply and binds operands to slots.
"""

import sympy
from sympy import (Symbol, Rational, sqrt, solve, simplify, factor,
                   expand, Abs, floor, ceiling, binomial, gcd, lcm,
                   diff, integrate, Mod, oo, pi, E, I)


class TemplateLibrary:
    """
    Lambda templates for MATH reasoning steps.
    Each template is a callable that takes typed operands and returns a SymPy result.
    """

    # ================================================================
    # SETUP templates — read from problem, establish starting state
    # ================================================================

    @staticmethod
    def define_variable(name: str, assumptions: dict = None):
        """Create a symbolic variable. 'Let x be...'"""
        return Symbol(name, **(assumptions or {}))

    @staticmethod
    def parse_equation(lhs, rhs):
        """Establish an equation from problem text. 'Given that 2x + 3 = 7'"""
        return sympy.Eq(lhs, rhs)

    @staticmethod
    def parse_expression(expr_str: str):
        """Parse a mathematical expression from text."""
        return sympy.sympify(expr_str)

    @staticmethod
    def assign_value(value):
        """Direct value from problem text. 'There are 40 croissants'"""
        return value

    # ================================================================
    # EVALUATE templates — compute concrete results
    # ================================================================

    @staticmethod
    def arithmetic(a, b, op: str):
        """Basic arithmetic: a op b."""
        ops = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b,
            'pow': lambda a, b: a ** b,
            'mod': lambda a, b: Mod(a, b),
        }
        return ops[op](a, b)

    @staticmethod
    def evaluate_expression(expr, substitutions: dict = None):
        """Evaluate an expression, optionally with substitutions."""
        if substitutions:
            expr = expr.subs(substitutions)
        try:
            result = sympy.nsimplify(expr)
            if result.is_number:
                return result
            return expr
        except:
            return expr

    @staticmethod
    def evaluate_at_point(expr, var, value):
        """Evaluate expression at a specific point. 'f(3) = ...'"""
        return expr.subs(var, value)

    @staticmethod
    def compute_numeric(expr):
        """Force numeric evaluation."""
        return sympy.nsimplify(expr)

    # ================================================================
    # SOLVE templates — find unknowns
    # ================================================================

    @staticmethod
    def solve_for_variable(equation, variable):
        """Solve equation for variable. Returns list of solutions."""
        if isinstance(equation, sympy.Eq):
            return solve(equation, variable)
        else:
            return solve(equation, variable)

    @staticmethod
    def solve_system(equations: list, variables: list):
        """Solve system of equations."""
        return solve(equations, variables)

    @staticmethod
    def solve_inequality(inequality, variable):
        """Solve inequality."""
        return sympy.solve_univariate_inequality(inequality, variable)

    # ================================================================
    # SUBSTITUTE templates
    # ================================================================

    @staticmethod
    def substitute_value(expr, var, value):
        """Replace variable with value."""
        return expr.subs(var, value)

    @staticmethod
    def substitute_expression(expr, var, replacement_expr):
        """Replace variable with expression."""
        return expr.subs(var, replacement_expr)

    @staticmethod
    def substitute_multiple(expr, substitutions: dict):
        """Multiple substitutions."""
        return expr.subs(substitutions)

    # ================================================================
    # SIMPLIFY templates
    # ================================================================

    @staticmethod
    def simplify_expression(expr):
        """General simplification."""
        return simplify(expr)

    @staticmethod
    def cancel_terms(expr):
        """Cancel common factors."""
        return sympy.cancel(expr)

    @staticmethod
    def collect_terms(expr, var):
        """Collect like terms."""
        return sympy.collect(expr, var)

    @staticmethod
    def rationalize(expr):
        """Rationalize denominator or expression."""
        return sympy.radsimp(expr)

    # ================================================================
    # FACTOR templates
    # ================================================================

    @staticmethod
    def factor_expression(expr):
        """Factor polynomial."""
        return factor(expr)

    @staticmethod
    def factor_integer(n):
        """Prime factorization."""
        return sympy.factorint(n)

    # ================================================================
    # EXPAND templates
    # ================================================================

    @staticmethod
    def expand_expression(expr):
        """Expand/distribute."""
        return expand(expr)

    @staticmethod
    def expand_power(base, exponent):
        """Expand a power."""
        return expand(base ** exponent)

    # ================================================================
    # APPLY THEOREM templates
    # ================================================================

    @staticmethod
    def apply_vietas(coefficients: list, which: str = 'sum'):
        """Vieta's formulas for polynomial roots."""
        a, b = coefficients[0], coefficients[1]
        if which == 'sum':
            return -b / a
        elif which == 'product' and len(coefficients) > 2:
            c = coefficients[2]
            return c / a
        return None

    @staticmethod
    def apply_pythagorean(a=None, b=None, c=None):
        """Pythagorean theorem. a^2 + b^2 = c^2, solve for missing."""
        if c is None:
            return sqrt(a**2 + b**2)
        elif a is None:
            return sqrt(c**2 - b**2)
        elif b is None:
            return sqrt(c**2 - a**2)

    @staticmethod
    def apply_quadratic_formula(a, b, c):
        """Quadratic formula. ax^2 + bx + c = 0"""
        discriminant = b**2 - 4*a*c
        x1 = (-b + sqrt(discriminant)) / (2*a)
        x2 = (-b - sqrt(discriminant)) / (2*a)
        return [x1, x2]

    @staticmethod
    def apply_binomial_theorem(n, k):
        """Binomial coefficient C(n,k)."""
        return binomial(n, k)

    @staticmethod
    def apply_modular_arithmetic(a, b, m, op='add'):
        """Modular operations."""
        ops = {
            'add': lambda: Mod(a + b, m),
            'mul': lambda: Mod(a * b, m),
            'pow': lambda: Mod(a ** b, m),
        }
        return ops.get(op, lambda: Mod(a, m))()

    @staticmethod
    def apply_formula(formula_expr, substitutions: dict):
        """General formula application."""
        return formula_expr.subs(substitutions)

    # ================================================================
    # COUNT templates
    # ================================================================

    @staticmethod
    def count_combinations(n, k):
        """C(n,k) = n choose k."""
        return binomial(n, k)

    @staticmethod
    def count_permutations(n, k=None):
        """P(n,k) permutations."""
        if k is None:
            return sympy.factorial(n)
        return sympy.factorial(n) / sympy.factorial(n - k)

    @staticmethod
    def count_divisors(n):
        """Count number of divisors."""
        return sympy.divisor_count(n)

    @staticmethod
    def count_elements_in_range(start, end, condition=None):
        """Count integers in range satisfying condition."""
        if condition is None:
            return end - start + 1
        count = 0
        for i in range(int(start), int(end) + 1):
            if condition(i):
                count += 1
        return count

    # ================================================================
    # COMPARE templates
    # ================================================================

    @staticmethod
    def compare_values(a, b, comparison='eq'):
        """Compare two values."""
        comparisons = {
            'eq': lambda: sympy.Eq(a, b),
            'lt': lambda: a < b,
            'gt': lambda: a > b,
            'le': lambda: a <= b,
            'ge': lambda: a >= b,
        }
        return comparisons[comparison]()

    @staticmethod
    def find_minimum(expr, var, domain=None):
        """Find minimum of expression."""
        critical = solve(diff(expr, var), var)
        if domain:
            critical = [c for c in critical if domain[0] <= c <= domain[1]]
        return min(expr.subs(var, c) for c in critical) if critical else None

    @staticmethod
    def find_maximum(expr, var, domain=None):
        """Find maximum of expression."""
        critical = solve(diff(expr, var), var)
        if domain:
            critical = [c for c in critical if domain[0] <= c <= domain[1]]
        return max(expr.subs(var, c) for c in critical) if critical else None

    # ================================================================
    # CONVERT templates
    # ================================================================

    @staticmethod
    def to_decimal(expr):
        """Convert to decimal representation."""
        return float(expr.evalf())

    @staticmethod
    def to_fraction(decimal_val):
        """Convert decimal to fraction."""
        return Rational(decimal_val).limit_denominator()

    @staticmethod
    def floor_value(expr):
        """Floor function."""
        return floor(expr)

    @staticmethod
    def ceiling_value(expr):
        """Ceiling function."""
        return ceiling(expr)

    @staticmethod
    def absolute_value(expr):
        """Absolute value."""
        return Abs(expr)

    # ================================================================
    # UTILITY
    # ================================================================

    @staticmethod
    def compute_gcd(a, b):
        """Greatest common divisor."""
        return gcd(a, b)

    @staticmethod
    def compute_lcm(a, b):
        """Least common multiple."""
        return lcm(a, b)

    @staticmethod
    def compute_remainder(a, b):
        """Remainder / modulo."""
        return Mod(a, b)


# ================================================================
# CLUSTER → TEMPLATE MAPPING
# ================================================================

CLUSTER_TEMPLATE_MAP = {
    # SETUP clusters
    "setup_neutral": {
        "primary": ["assign_value", "parse_equation", "define_variable"],
        "description": "Initialize values and equations from problem text",
        "expected_operands": "values and expressions from problem text",
        "expected_output": "equation | expression | number"
    },
    "setup_increases": {
        "primary": ["parse_equation", "define_variable"],
        "description": "Set up complex equations or multiple variables",
        "expected_operands": "symbolic expressions",
        "expected_output": "equation | expression"
    },

    # EVALUATE clusters
    "local_evaluate_reduces": {
        "primary": ["evaluate_expression", "arithmetic", "compute_numeric"],
        "description": "Compute result using previous step's output",
        "expected_operands": "numbers or resolved expressions",
        "expected_output": "number"
    },
    "evaluate_reduces": {
        "primary": ["evaluate_expression", "evaluate_at_point", "compute_numeric"],
        "description": "Compute result from problem values directly",
        "expected_operands": "numbers or expressions from problem",
        "expected_output": "number"
    },
    "distant_evaluate_reduces": {
        "primary": ["evaluate_expression", "arithmetic"],
        "description": "Compute using results from much earlier steps",
        "expected_operands": "accumulated results",
        "expected_output": "number"
    },

    # SOLVE clusters
    "local_solve_equation_reduces": {
        "primary": ["solve_for_variable", "solve_system"],
        "description": "Solve equation built in previous step",
        "expected_operands": "equation + variable",
        "expected_output": "number | expression"
    },
    "solve_equation_reduces": {
        "primary": ["solve_for_variable", "solve_inequality"],
        "description": "Solve equation from problem directly",
        "expected_operands": "equation + variable",
        "expected_output": "number | set"
    },

    # SUBSTITUTE clusters
    "local_substitute_neutral": {
        "primary": ["substitute_value", "substitute_expression"],
        "description": "Plug in value from previous step",
        "expected_operands": "expression + variable + value",
        "expected_output": "expression"
    },
    "substitute_neutral": {
        "primary": ["substitute_value", "substitute_multiple"],
        "description": "Plug in values from problem text",
        "expected_operands": "expression + substitution pairs",
        "expected_output": "expression"
    },
    "medium_substitute_neutral": {
        "primary": ["substitute_expression", "substitute_multiple"],
        "description": "Substitute using results from 2-3 steps back",
        "expected_operands": "expression + multiple substitutions",
        "expected_output": "expression"
    },

    # SIMPLIFY clusters
    "local_simplify_reduces": {
        "primary": ["simplify_expression", "cancel_terms", "rationalize"],
        "description": "Simplify result from previous step",
        "expected_operands": "expression",
        "expected_output": "expression | number"
    },
    "simplify_reduces": {
        "primary": ["simplify_expression", "collect_terms"],
        "description": "Simplify expression from problem",
        "expected_operands": "expression",
        "expected_output": "expression | number"
    },

    # FACTOR clusters
    "local_factor_reduces": {
        "primary": ["factor_expression", "factor_integer"],
        "description": "Factor previous step's result",
        "expected_operands": "expression or integer",
        "expected_output": "expression | dict"
    },
    "factor_reduces": {
        "primary": ["factor_expression", "factor_integer"],
        "description": "Factor expression from problem",
        "expected_operands": "expression or integer",
        "expected_output": "expression | dict"
    },

    # EXPAND clusters
    "expand_increases": {
        "primary": ["expand_expression", "expand_power"],
        "description": "Expand/distribute expression",
        "expected_operands": "expression",
        "expected_output": "expression"
    },
    "local_expand_increases": {
        "primary": ["expand_expression", "expand_power"],
        "description": "Expand previous step's result",
        "expected_operands": "expression",
        "expected_output": "expression"
    },

    # APPLY THEOREM clusters
    "apply_theorem_neutral": {
        "primary": ["apply_formula", "apply_quadratic_formula",
                     "apply_pythagorean", "apply_binomial_theorem"],
        "description": "Apply theorem/formula from scratch",
        "expected_operands": "formula-specific operands",
        "expected_output": "number | expression"
    },
    "local_apply_theorem_neutral": {
        "primary": ["apply_formula", "apply_modular_arithmetic",
                     "apply_vietas"],
        "description": "Apply theorem using previous step's result",
        "expected_operands": "formula + previous results",
        "expected_output": "number | expression"
    },
    "apply_theorem_reduces": {
        "primary": ["apply_formula", "apply_quadratic_formula"],
        "description": "Apply theorem that simplifies to a result",
        "expected_operands": "formula-specific operands",
        "expected_output": "number"
    },

    # COUNT clusters
    "count_neutral": {
        "primary": ["count_combinations", "count_permutations",
                     "count_elements_in_range", "count_divisors"],
        "description": "Counting/combinatorial computation",
        "expected_operands": "integers, ranges, conditions",
        "expected_output": "number"
    },
    "local_count_reduces": {
        "primary": ["count_combinations", "count_elements_in_range"],
        "description": "Count using previous step's result",
        "expected_operands": "integers from previous steps",
        "expected_output": "number"
    },

    # COMPARE clusters
    "compare_neutral": {
        "primary": ["compare_values", "find_minimum", "find_maximum"],
        "description": "Compare or optimize values",
        "expected_operands": "two values or expression + domain",
        "expected_output": "boolean | number"
    },
    "local_compare_neutral": {
        "primary": ["compare_values", "find_minimum", "find_maximum"],
        "description": "Compare using previous results",
        "expected_operands": "values from previous steps",
        "expected_output": "boolean | number"
    },

    # CONVERT clusters
    "convert_neutral": {
        "primary": ["to_decimal", "to_fraction", "floor_value",
                     "ceiling_value", "absolute_value"],
        "description": "Change representation",
        "expected_operands": "number or expression",
        "expected_output": "number"
    },

    # OTHER/CATCH-ALL
    "local_other_neutral": {
        "primary": ["evaluate_expression", "simplify_expression",
                     "substitute_value"],
        "description": "Miscellaneous step with dependency",
        "expected_operands": "varies",
        "expected_output": "varies"
    },
}


# ================================================================
# SUPER-CATEGORY MAPPING (for factor graph priors)
# ================================================================

SUPER_CATEGORIES = {
    "SETUP": ["setup_neutral", "setup_increases"],
    "COMPUTE": ["local_evaluate_reduces", "evaluate_reduces", "distant_evaluate_reduces",
                "local_simplify_reduces", "simplify_reduces",
                "expand_increases", "local_expand_increases",
                "local_factor_reduces", "factor_reduces"],
    "SOLVE": ["local_solve_equation_reduces", "solve_equation_reduces",
              "local_substitute_neutral", "substitute_neutral", "medium_substitute_neutral"],
    "REASON": ["apply_theorem_neutral", "local_apply_theorem_neutral", "apply_theorem_reduces",
               "count_neutral", "local_count_reduces",
               "compare_neutral", "local_compare_neutral"],
    "CONVERT": ["convert_neutral"],
    "OTHER": ["local_other_neutral"],
}

# Reverse mapping: cluster -> super-category
CLUSTER_TO_SUPER = {}
for super_cat, clusters in SUPER_CATEGORIES.items():
    for cluster in clusters:
        CLUSTER_TO_SUPER[cluster] = super_cat


def get_template_function(template_name: str):
    """Get a template function by name."""
    return getattr(TemplateLibrary, template_name, None)


def get_cluster_templates(cluster_name: str):
    """Get the list of template names for a cluster."""
    if cluster_name not in CLUSTER_TEMPLATE_MAP:
        return ["evaluate_expression", "simplify_expression"]  # fallback
    return CLUSTER_TEMPLATE_MAP[cluster_name]["primary"]
