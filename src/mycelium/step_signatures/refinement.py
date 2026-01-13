"""Signature Refinement Loop: Self-improvement for negative-lift signatures.

This module implements the signature refinement loop from paper.md:

1. IDENTIFY: Query signatures with success_rate < threshold or negative lift
2. ANALYZE: LLM examines failure cases to understand why DSL fails
3. DECOMPOSE: Split into finer-grained sub-signatures (semantic umbrella)
4. REDIRECT: Add routing from parent to children
5. GENERATE DSL: Build new DSL for each sub-signature
6. VALIDATE: Test on held-out examples (lift tracking handles this automatically)

**Practical note for LLM-assisted refinement:**
The LLM may initially resist this task ("I can help you think through approaches...")
or produce overly cautious responses. Insist on concrete outputs: specific sub-signature
names, actual DSL code, explicit routing conditions. The model is capable; it just needs
clear direction that you want executable artifacts, not suggestions.

Usage:
    from mycelium.step_signatures.refinement import SignatureRefiner

    refiner = SignatureRefiner(db, client)
    results = await refiner.refine_negative_lift_signatures(min_lift=-0.20)
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from .db import StepSignatureDB
from .models import StepSignature

logger = logging.getLogger(__name__)


@dataclass
class RefinementResult:
    """Result of refining a single signature."""
    signature_id: int
    step_type: str
    original_lift: float
    action: str  # "decomposed", "dsl_fixed", "guidance_only", "skipped"
    children_created: int = 0
    new_dsl: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RefinementReport:
    """Summary of a refinement run."""
    signatures_analyzed: int
    decomposed: int
    dsl_fixed: int
    guidance_only: int
    skipped: int
    errors: int
    results: list[RefinementResult]


# Step types that should be decomposed into semantic umbrellas
# (too generic to have a single DSL)
UMBRELLA_CANDIDATES = {
    "general_step",
    "synthesize_results",
    "solve_equation",
    "simplify_expression",
    "compute_probability",
    "factor_expression",
    "substitute_value",
    "setup_equation",
    "solve_system",
    "compute_volume",
    "compute_sqrt",
    "count_items",
    "common_difference",
    # Added for remaining negative-lift signatures
    "compute_product",
    "compute_sum",
    "compute_difference",
    "vector_operation",
    "compute_logarithm",
    "compute_radius",
    "compute_length",
    "compute_angle",
    # Added for lift < 5% signatures
    "compute_power",
    "trig_function",
    "count_total_items",
    "express_relation",
    "compute_distance",
    "evaluate_function",
    "compute_remainder",
    "compute_gcd",
    "find_minimum",
    "apply_inequality",
    "factor_polynomial",
    "compute_perimeter",
    "compute_lcm",
    "area_circle",
    "compute_quotient",
    # Added for deeper decomposition
    "binomial_coefficient",
    "count_ways",
    "compute_area",
}

# Step types that should use guidance-only (no DSL appropriate)
GUIDANCE_ONLY_TYPES = {
    "synthesize_results",  # Requires LLM reasoning to combine
    "setup_equation",      # Requires understanding problem structure
    "define_variables",    # Conceptual, not computational
    "express_relation",    # Conceptual
    "apply_amgm",          # Inequality reasoning, needs LLM
}

# Step types with fixable DSLs (formula is just wrong/incomplete)
FIXABLE_DSL_TYPES = {
    "compute_sum",
    "compute_product",
    "compute_quotient",
    "compute_difference",
}


class SignatureRefiner:
    """Refines signatures with negative lift through decomposition or DSL fixes."""

    def __init__(self, db: StepSignatureDB, client=None):
        """Initialize the refiner.

        Args:
            db: Step signature database
            client: LLM client for generating decompositions/DSLs (optional)
        """
        self.db = db
        self.client = client

    async def refine_negative_lift_signatures(
        self,
        min_lift: float = -0.10,
        min_uses: int = 5,
        max_signatures: int = 20,
    ) -> RefinementReport:
        """Find and refine signatures with negative lift.

        Args:
            min_lift: Only refine signatures with lift below this threshold
            min_uses: Minimum uses in both arms to have reliable lift data
            max_signatures: Maximum signatures to process in one run

        Returns:
            RefinementReport with summary and per-signature results
        """
        # Get negative lift signatures
        candidates = self.db.get_signatures_for_dsl_improvement(
            min_uses=min_uses,
            lift_threshold=min_lift,
        )

        candidates = candidates[:max_signatures]

        logger.info(
            "[refinement] Found %d signatures with lift < %.0f%% to refine",
            len(candidates), min_lift * 100
        )

        results = []
        decomposed = 0
        dsl_fixed = 0
        guidance_only = 0
        skipped = 0
        errors = 0

        for sig in candidates:
            try:
                result = await self._refine_signature(sig)
                results.append(result)

                if result.action == "decomposed":
                    decomposed += 1
                elif result.action == "dsl_fixed":
                    dsl_fixed += 1
                elif result.action == "guidance_only":
                    guidance_only += 1
                elif result.action == "skipped":
                    skipped += 1

                if result.error:
                    errors += 1

            except Exception as e:
                logger.error("[refinement] Failed to refine sig=%d: %s", sig.id, e)
                results.append(RefinementResult(
                    signature_id=sig.id,
                    step_type=sig.step_type,
                    original_lift=self._compute_lift(sig),
                    action="error",
                    error=str(e),
                ))
                errors += 1

        return RefinementReport(
            signatures_analyzed=len(candidates),
            decomposed=decomposed,
            dsl_fixed=dsl_fixed,
            guidance_only=guidance_only,
            skipped=skipped,
            errors=errors,
            results=results,
        )

    async def _refine_signature(self, sig: StepSignature) -> RefinementResult:
        """Refine a single signature based on its type."""
        lift = self._compute_lift(sig)

        # Already a semantic umbrella? Skip
        if sig.is_semantic_umbrella:
            return RefinementResult(
                signature_id=sig.id,
                step_type=sig.step_type,
                original_lift=lift,
                action="skipped",
            )

        # Guidance-only types: remove DSL, use LLM
        if sig.step_type in GUIDANCE_ONLY_TYPES:
            self._convert_to_guidance_only(sig)
            return RefinementResult(
                signature_id=sig.id,
                step_type=sig.step_type,
                original_lift=lift,
                action="guidance_only",
            )

        # Umbrella candidates: decompose into children
        if sig.step_type in UMBRELLA_CANDIDATES:
            children = await self._decompose_to_umbrella(sig)
            return RefinementResult(
                signature_id=sig.id,
                step_type=sig.step_type,
                original_lift=lift,
                action="decomposed",
                children_created=len(children),
            )

        # Fixable DSL types: try to fix the DSL
        if sig.step_type in FIXABLE_DSL_TYPES:
            new_dsl = await self._fix_dsl(sig)
            if new_dsl:
                return RefinementResult(
                    signature_id=sig.id,
                    step_type=sig.step_type,
                    original_lift=lift,
                    action="dsl_fixed",
                    new_dsl=new_dsl,
                )

        # Unknown type - skip for now
        return RefinementResult(
            signature_id=sig.id,
            step_type=sig.step_type,
            original_lift=lift,
            action="skipped",
        )

    def _compute_lift(self, sig: StepSignature) -> float:
        """Compute lift for a signature."""
        if sig.injected_uses == 0 or sig.non_injected_uses == 0:
            return 0.0
        inj_rate = sig.injected_successes / sig.injected_uses
        base_rate = sig.non_injected_successes / sig.non_injected_uses
        return inj_rate - base_rate

    def _convert_to_guidance_only(self, sig: StepSignature) -> None:
        """Convert a signature to guidance-only (remove DSL)."""
        with self.db._connection() as conn:
            conn.execute(
                """UPDATE step_signatures
                   SET dsl_script = NULL,
                       injected_uses = 0,
                       injected_successes = 0
                   WHERE id = ?""",
                (sig.id,)
            )
        logger.info("[refinement] Converted sig=%d (%s) to guidance-only", sig.id, sig.step_type)

    async def _decompose_to_umbrella(self, sig: StepSignature) -> list[dict]:
        """Decompose a signature into a semantic umbrella with children.

        Returns list of created child specs.
        """
        # Get decomposition specs for this step type
        children_specs = self._get_decomposition_specs(sig.step_type)

        if not children_specs:
            logger.warning("[refinement] No decomposition specs for %s", sig.step_type)
            return []

        created = []
        for spec in children_specs:
            child = self.db.create_child_signature(
                parent_id=sig.id,
                step_type=spec["step_type"],
                description=spec["description"],
                dsl_script=json.dumps(spec["dsl"]),
                condition=spec["condition"],
            )
            if child:
                created.append(spec)
                logger.info(
                    "[refinement] Created child %s for umbrella %s",
                    spec["step_type"], sig.step_type
                )

        return created

    def _get_decomposition_specs(self, step_type: str) -> list[dict]:
        """Get child decomposition specs for a step type."""

        # Pre-defined decompositions for common problematic types
        DECOMPOSITIONS = {
            "factor_expression": [
                {
                    "step_type": "factor_gcf",
                    "description": "Factor out greatest common factor",
                    "condition": "expression has common factor",
                    "dsl": {"type": "sympy", "script": "factor(expr)", "params": ["expr"], "fallback": "decompose"}
                },
                {
                    "step_type": "factor_quadratic",
                    "description": "Factor quadratic ax^2 + bx + c",
                    "condition": "quadratic expression",
                    "dsl": {"type": "sympy", "script": "factor(a*x**2 + b*x + c)", "params": ["a", "b", "c"], "fallback": "decompose"}
                },
                {
                    "step_type": "factor_difference_squares",
                    "description": "Factor a^2 - b^2 = (a+b)(a-b)",
                    "condition": "difference of squares",
                    "dsl": {"type": "math", "script": "(a + b) * (a - b)", "params": ["a", "b"], "fallback": "decompose"}
                },
            ],
            "substitute_value": [
                {
                    "step_type": "substitute_numeric",
                    "description": "Substitute numeric value into expression",
                    "condition": "substituting a number",
                    "dsl": {"type": "sympy", "script": "expr.subs(var, value)", "params": ["expr", "var", "value"], "fallback": "decompose"}
                },
                {
                    "step_type": "substitute_expression",
                    "description": "Substitute one expression for another",
                    "condition": "substituting an expression",
                    "dsl": {"type": "sympy", "script": "expr.subs(old, new)", "params": ["expr", "old", "new"], "fallback": "decompose"}
                },
            ],
            "solve_system": [
                {
                    "step_type": "solve_system_2x2",
                    "description": "Solve 2x2 system of linear equations",
                    "condition": "two equations, two unknowns",
                    "dsl": {"type": "sympy", "script": "solve([eq1, eq2], [x, y])", "params": ["eq1", "eq2"], "fallback": "decompose"}
                },
                {
                    "step_type": "solve_system_substitution",
                    "description": "Solve by substitution method",
                    "condition": "one equation easily isolates a variable",
                    "dsl": {"type": "decompose", "script": "Isolate one variable, substitute into other equation", "params": [], "fallback": "decompose"}
                },
            ],
            "compute_volume": [
                {
                    "step_type": "volume_box",
                    "description": "Volume of rectangular box: l * w * h",
                    "condition": "rectangular box or prism",
                    "dsl": {"type": "math", "script": "length * width * height", "params": ["length", "width", "height"], "fallback": "decompose"}
                },
                {
                    "step_type": "volume_cylinder",
                    "description": "Volume of cylinder: pi * r^2 * h",
                    "condition": "cylinder",
                    "dsl": {"type": "math", "script": "pi * radius**2 * height", "params": ["radius", "height"], "fallback": "decompose"}
                },
                {
                    "step_type": "volume_sphere",
                    "description": "Volume of sphere: (4/3) * pi * r^3",
                    "condition": "sphere",
                    "dsl": {"type": "math", "script": "(4/3) * pi * radius**3", "params": ["radius"], "fallback": "decompose"}
                },
                {
                    "step_type": "volume_cone",
                    "description": "Volume of cone: (1/3) * pi * r^2 * h",
                    "condition": "cone",
                    "dsl": {"type": "math", "script": "(1/3) * pi * radius**2 * height", "params": ["radius", "height"], "fallback": "decompose"}
                },
            ],
            "compute_sqrt": [
                {
                    "step_type": "sqrt_perfect",
                    "description": "Square root of perfect square",
                    "condition": "number is a perfect square",
                    "dsl": {"type": "math", "script": "int(sqrt(n))", "params": ["n"], "fallback": "decompose"}
                },
                {
                    "step_type": "sqrt_simplify",
                    "description": "Simplify sqrt by factoring out perfect squares",
                    "condition": "simplify radical",
                    "dsl": {"type": "sympy", "script": "sqrt(n).simplify()", "params": ["n"], "fallback": "decompose"}
                },
                {
                    "step_type": "sqrt_decimal",
                    "description": "Compute decimal approximation of sqrt",
                    "condition": "decimal answer needed",
                    "dsl": {"type": "math", "script": "sqrt(n)", "params": ["n"], "fallback": "decompose"}
                },
            ],
            "general_step": [
                {
                    "step_type": "extract_value",
                    "description": "Extract a numeric value from problem text",
                    "condition": "extracting given values",
                    "dsl": {"type": "decompose", "script": "Identify and extract the numeric value", "params": [], "fallback": "decompose"}
                },
                {
                    "step_type": "apply_formula",
                    "description": "Apply a known formula with given values",
                    "condition": "applying a formula",
                    "dsl": {"type": "sympy", "script": "formula.subs(substitutions)", "params": ["formula", "substitutions"], "fallback": "decompose"}
                },
                {
                    "step_type": "compare_values",
                    "description": "Compare two values and determine relationship",
                    "condition": "comparing quantities",
                    "dsl": {"type": "math", "script": "a - b", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "unit_conversion",
                    "description": "Convert between units",
                    "condition": "unit conversion needed",
                    "dsl": {"type": "math", "script": "value * conversion_factor", "params": ["value", "conversion_factor"], "fallback": "decompose"}
                },
                {
                    "step_type": "compute_basic_arithmetic",
                    "description": "Perform basic arithmetic: +, -, *, /",
                    "condition": "arithmetic operation",
                    "dsl": {"type": "math", "script": "a + b", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "identify_pattern",
                    "description": "Identify a pattern or relationship",
                    "condition": "pattern recognition",
                    "dsl": {"type": "decompose", "script": "Identify the pattern in the sequence or relationship", "params": [], "fallback": "decompose"}
                },
                {
                    "step_type": "substitute_and_evaluate",
                    "description": "Substitute values and evaluate expression",
                    "condition": "substitution needed",
                    "dsl": {"type": "sympy", "script": "expr.subs(var, value)", "params": ["expr", "var", "value"], "fallback": "decompose"}
                },
                {
                    "step_type": "set_up_expression",
                    "description": "Set up mathematical expression from problem",
                    "condition": "building expression",
                    "dsl": {"type": "decompose", "script": "Translate problem into mathematical expression", "params": [], "fallback": "decompose"}
                },
            ],
            "solve_equation": [
                {
                    "step_type": "solve_linear_single",
                    "description": "Solve ax + b = c for x",
                    "condition": "linear equation with one variable",
                    "dsl": {"type": "math", "script": "(c - b) / a", "params": ["a", "b", "c"], "fallback": "decompose"}
                },
                {
                    "step_type": "solve_quadratic_formula",
                    "description": "Solve ax^2 + bx + c = 0 using quadratic formula",
                    "condition": "quadratic equation",
                    "dsl": {"type": "sympy", "script": "solve(a*x**2 + b*x + c, x)", "params": ["a", "b", "c"], "fallback": "decompose"}
                },
                {
                    "step_type": "solve_proportion",
                    "description": "Solve a/b = c/d for one variable",
                    "condition": "proportion or ratio equation",
                    "dsl": {"type": "math", "script": "(a * d) / b", "params": ["a", "b", "d"], "fallback": "decompose"}
                },
                {
                    "step_type": "isolate_variable",
                    "description": "Isolate a variable from a formula",
                    "condition": "rearranging formula for a variable",
                    "dsl": {"type": "sympy", "script": "solve(equation, target_var)", "params": ["equation", "target_var"], "fallback": "decompose"}
                },
            ],
            "simplify_expression": [
                {
                    "step_type": "simplify_fraction",
                    "description": "Reduce a fraction to lowest terms using GCD",
                    "condition": "fraction to reduce",
                    "dsl": {"type": "math", "script": "numerator // gcd(numerator, denominator)", "params": ["numerator", "denominator"], "fallback": "decompose"}
                },
                {
                    "step_type": "combine_like_terms",
                    "description": "Combine like terms in an algebraic expression",
                    "condition": "polynomial with like terms",
                    "dsl": {"type": "sympy", "script": "simplify(expand(expr))", "params": ["expr"], "fallback": "decompose"}
                },
                {
                    "step_type": "simplify_radical",
                    "description": "Simplify radical expressions (square roots, etc.)",
                    "condition": "expression with radicals",
                    "dsl": {"type": "sympy", "script": "simplify(sqrt(n))", "params": ["n"], "fallback": "decompose"}
                },
                {
                    "step_type": "factor_common",
                    "description": "Factor out common terms from expression",
                    "condition": "expression with common factors",
                    "dsl": {"type": "sympy", "script": "factor(expr)", "params": ["expr"], "fallback": "decompose"}
                },
            ],
            "count_items": [
                {
                    "step_type": "count_simple",
                    "description": "Count items in a set or list",
                    "condition": "simple counting",
                    "dsl": {"type": "math", "script": "n", "params": ["n"], "fallback": "decompose"}
                },
                {
                    "step_type": "count_with_condition",
                    "description": "Count items satisfying a condition",
                    "condition": "conditional counting",
                    "dsl": {"type": "decompose", "script": "Count items where condition holds", "params": [], "fallback": "decompose"}
                },
                {
                    "step_type": "count_complement",
                    "description": "Count by subtracting from total",
                    "condition": "easier to count what's excluded",
                    "dsl": {"type": "math", "script": "total - excluded", "params": ["total", "excluded"], "fallback": "decompose"}
                },
            ],
            "common_difference": [
                {
                    "step_type": "diff_arithmetic_seq",
                    "description": "Find common difference d = a2 - a1",
                    "condition": "arithmetic sequence given",
                    "dsl": {"type": "math", "script": "a2 - a1", "params": ["a1", "a2"], "fallback": "decompose"}
                },
                {
                    "step_type": "diff_from_formula",
                    "description": "Find difference from nth term formula",
                    "condition": "formula for nth term given",
                    "dsl": {"type": "math", "script": "(an - a1) / (n - 1)", "params": ["a1", "an", "n"], "fallback": "decompose"}
                },
            ],
            # Added for remaining negative-lift signatures
            "compute_product": [
                {
                    "step_type": "product_two_numbers",
                    "description": "Multiply two numbers: a * b",
                    "condition": "multiplying two values",
                    "dsl": {"type": "math", "script": "a * b", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "product_sequence",
                    "description": "Product of a sequence of numbers",
                    "condition": "multiplying multiple values",
                    "dsl": {"type": "math", "script": "prod(values)", "params": ["values"], "fallback": "decompose"}
                },
                {
                    "step_type": "product_factorial",
                    "description": "Factorial product: n!",
                    "condition": "factorial calculation",
                    "dsl": {"type": "math", "script": "factorial(n)", "params": ["n"], "fallback": "decompose"}
                },
            ],
            "compute_sum": [
                {
                    "step_type": "sum_two_numbers",
                    "description": "Add two numbers: a + b",
                    "condition": "adding two values",
                    "dsl": {"type": "math", "script": "a + b", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "sum_sequence",
                    "description": "Sum of a sequence of numbers",
                    "condition": "adding multiple values",
                    "dsl": {"type": "math", "script": "sum(values)", "params": ["values"], "fallback": "decompose"}
                },
                {
                    "step_type": "sum_arithmetic_series",
                    "description": "Sum of arithmetic series: n(a1 + an)/2",
                    "condition": "arithmetic series sum",
                    "dsl": {"type": "math", "script": "n * (a1 + an) / 2", "params": ["n", "a1", "an"], "fallback": "decompose"}
                },
                {
                    "step_type": "sum_geometric_series",
                    "description": "Sum of geometric series: a(1-r^n)/(1-r)",
                    "condition": "geometric series sum",
                    "dsl": {"type": "math", "script": "a * (1 - r**n) / (1 - r)", "params": ["a", "r", "n"], "fallback": "decompose"}
                },
            ],
            "compute_difference": [
                {
                    "step_type": "difference_two_numbers",
                    "description": "Subtract two numbers: a - b",
                    "condition": "subtracting two values",
                    "dsl": {"type": "math", "script": "a - b", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "difference_absolute",
                    "description": "Absolute difference: |a - b|",
                    "condition": "absolute difference needed",
                    "dsl": {"type": "math", "script": "abs(a - b)", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "difference_percentage",
                    "description": "Percentage difference: (new - old) / old * 100",
                    "condition": "percentage change",
                    "dsl": {"type": "math", "script": "(new_val - old_val) / old_val * 100", "params": ["new_val", "old_val"], "fallback": "decompose"}
                },
            ],
            "compute_probability": [
                {
                    "step_type": "probability_simple",
                    "description": "Simple probability: favorable / total",
                    "condition": "counting outcomes",
                    "dsl": {"type": "math", "script": "favorable / total", "params": ["favorable", "total"], "fallback": "decompose"}
                },
                {
                    "step_type": "probability_complement",
                    "description": "Complement probability: 1 - P(A)",
                    "condition": "complement event",
                    "dsl": {"type": "math", "script": "1 - p", "params": ["p"], "fallback": "decompose"}
                },
                {
                    "step_type": "probability_conditional",
                    "description": "Conditional probability: P(A|B) = P(A∩B) / P(B)",
                    "condition": "conditional or given event",
                    "dsl": {"type": "math", "script": "p_and / p_given", "params": ["p_and", "p_given"], "fallback": "decompose"}
                },
                {
                    "step_type": "probability_independent",
                    "description": "Independent events: P(A and B) = P(A) * P(B)",
                    "condition": "independent events",
                    "dsl": {"type": "math", "script": "p_a * p_b", "params": ["p_a", "p_b"], "fallback": "decompose"}
                },
            ],
            "vector_operation": [
                {
                    "step_type": "vector_add",
                    "description": "Add two vectors component-wise",
                    "condition": "vector addition",
                    "dsl": {"type": "math", "script": "[a[i] + b[i] for i in range(len(a))]", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "vector_dot",
                    "description": "Dot product: sum of component products",
                    "condition": "dot product",
                    "dsl": {"type": "math", "script": "sum(a[i] * b[i] for i in range(len(a)))", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "vector_magnitude",
                    "description": "Vector magnitude: sqrt(sum of squares)",
                    "condition": "magnitude or length",
                    "dsl": {"type": "math", "script": "sqrt(sum(x**2 for x in v))", "params": ["v"], "fallback": "decompose"}
                },
                {
                    "step_type": "vector_cross",
                    "description": "Cross product of 3D vectors",
                    "condition": "cross product",
                    "dsl": {"type": "decompose", "script": "Compute cross product component by component", "params": [], "fallback": "decompose"}
                },
            ],
            "compute_logarithm": [
                {
                    "step_type": "log_base_10",
                    "description": "Common logarithm (base 10)",
                    "condition": "log base 10",
                    "dsl": {"type": "math", "script": "log10(x)", "params": ["x"], "fallback": "decompose"}
                },
                {
                    "step_type": "log_natural",
                    "description": "Natural logarithm (base e)",
                    "condition": "natural log or ln",
                    "dsl": {"type": "math", "script": "log(x)", "params": ["x"], "fallback": "decompose"}
                },
                {
                    "step_type": "log_arbitrary_base",
                    "description": "Logarithm with arbitrary base: log_b(x)",
                    "condition": "log with specific base",
                    "dsl": {"type": "math", "script": "log(x) / log(base)", "params": ["x", "base"], "fallback": "decompose"}
                },
            ],
            "compute_radius": [
                {
                    "step_type": "radius_from_diameter",
                    "description": "Radius from diameter: r = d/2",
                    "condition": "diameter given",
                    "dsl": {"type": "math", "script": "diameter / 2", "params": ["diameter"], "fallback": "decompose"}
                },
                {
                    "step_type": "radius_from_circumference",
                    "description": "Radius from circumference: r = C/(2π)",
                    "condition": "circumference given",
                    "dsl": {"type": "math", "script": "circumference / (2 * pi)", "params": ["circumference"], "fallback": "decompose"}
                },
                {
                    "step_type": "radius_from_area",
                    "description": "Radius from circle area: r = sqrt(A/π)",
                    "condition": "area given",
                    "dsl": {"type": "math", "script": "sqrt(area / pi)", "params": ["area"], "fallback": "decompose"}
                },
            ],
            "compute_length": [
                {
                    "step_type": "length_pythagorean",
                    "description": "Length using Pythagorean theorem",
                    "condition": "right triangle or distance",
                    "dsl": {"type": "math", "script": "sqrt(a**2 + b**2)", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "length_segment",
                    "description": "Length of line segment between points",
                    "condition": "coordinates given",
                    "dsl": {"type": "math", "script": "sqrt((x2-x1)**2 + (y2-y1)**2)", "params": ["x1", "y1", "x2", "y2"], "fallback": "decompose"}
                },
                {
                    "step_type": "length_perimeter",
                    "description": "Perimeter as sum of sides",
                    "condition": "perimeter calculation",
                    "dsl": {"type": "math", "script": "sum(sides)", "params": ["sides"], "fallback": "decompose"}
                },
            ],
            "compute_angle": [
                {
                    "step_type": "angle_triangle_sum",
                    "description": "Third angle in triangle: 180 - a - b",
                    "condition": "triangle angles",
                    "dsl": {"type": "math", "script": "180 - angle1 - angle2", "params": ["angle1", "angle2"], "fallback": "decompose"}
                },
                {
                    "step_type": "angle_trig_inverse",
                    "description": "Angle from trig ratio using inverse function",
                    "condition": "inverse trig needed",
                    "dsl": {"type": "math", "script": "degrees(asin(ratio))", "params": ["ratio"], "fallback": "decompose"}
                },
                {
                    "step_type": "angle_supplementary",
                    "description": "Supplementary angle: 180 - θ",
                    "condition": "supplementary angles",
                    "dsl": {"type": "math", "script": "180 - angle", "params": ["angle"], "fallback": "decompose"}
                },
                {
                    "step_type": "angle_complementary",
                    "description": "Complementary angle: 90 - θ",
                    "condition": "complementary angles",
                    "dsl": {"type": "math", "script": "90 - angle", "params": ["angle"], "fallback": "decompose"}
                },
            ],
            # Added for lift < 5% signatures
            "compute_power": [
                {
                    "step_type": "power_integer",
                    "description": "Compute base^exponent for integer exponent",
                    "condition": "integer exponent",
                    "dsl": {"type": "math", "script": "base ** exponent", "params": ["base", "exponent"], "fallback": "decompose"}
                },
                {
                    "step_type": "power_square",
                    "description": "Square a number: x^2",
                    "condition": "squaring",
                    "dsl": {"type": "math", "script": "x ** 2", "params": ["x"], "fallback": "decompose"}
                },
                {
                    "step_type": "power_cube",
                    "description": "Cube a number: x^3",
                    "condition": "cubing",
                    "dsl": {"type": "math", "script": "x ** 3", "params": ["x"], "fallback": "decompose"}
                },
                {
                    "step_type": "power_fractional",
                    "description": "Fractional exponent (roots): x^(1/n)",
                    "condition": "fractional or root",
                    "dsl": {"type": "math", "script": "x ** (1/n)", "params": ["x", "n"], "fallback": "decompose"}
                },
            ],
            "trig_function": [
                {
                    "step_type": "trig_sin",
                    "description": "Compute sine of angle",
                    "condition": "sine calculation",
                    "dsl": {"type": "math", "script": "sin(radians(angle))", "params": ["angle"], "fallback": "decompose"}
                },
                {
                    "step_type": "trig_cos",
                    "description": "Compute cosine of angle",
                    "condition": "cosine calculation",
                    "dsl": {"type": "math", "script": "cos(radians(angle))", "params": ["angle"], "fallback": "decompose"}
                },
                {
                    "step_type": "trig_tan",
                    "description": "Compute tangent of angle",
                    "condition": "tangent calculation",
                    "dsl": {"type": "math", "script": "tan(radians(angle))", "params": ["angle"], "fallback": "decompose"}
                },
                {
                    "step_type": "trig_identity",
                    "description": "Apply trigonometric identity",
                    "condition": "identity application",
                    "dsl": {"type": "decompose", "script": "Apply appropriate trig identity", "params": [], "fallback": "decompose"}
                },
            ],
            "count_total_items": [
                {
                    "step_type": "count_direct",
                    "description": "Direct count of items in set",
                    "condition": "items listed",
                    "dsl": {"type": "math", "script": "len(items)", "params": ["items"], "fallback": "decompose"}
                },
                {
                    "step_type": "count_product_rule",
                    "description": "Count using multiplication principle",
                    "condition": "independent choices",
                    "dsl": {"type": "math", "script": "n1 * n2", "params": ["n1", "n2"], "fallback": "decompose"}
                },
                {
                    "step_type": "count_sum_rule",
                    "description": "Count using addition principle",
                    "condition": "mutually exclusive cases",
                    "dsl": {"type": "math", "script": "n1 + n2", "params": ["n1", "n2"], "fallback": "decompose"}
                },
            ],
            "express_relation": [
                {
                    "step_type": "relation_linear",
                    "description": "Express linear relationship: y = mx + b",
                    "condition": "linear relationship",
                    "dsl": {"type": "decompose", "script": "Express as y = mx + b", "params": [], "fallback": "decompose"}
                },
                {
                    "step_type": "relation_proportional",
                    "description": "Express proportional relationship: y = kx",
                    "condition": "direct proportion",
                    "dsl": {"type": "decompose", "script": "Express as y = kx", "params": [], "fallback": "decompose"}
                },
                {
                    "step_type": "relation_inverse",
                    "description": "Express inverse relationship: y = k/x",
                    "condition": "inverse proportion",
                    "dsl": {"type": "decompose", "script": "Express as y = k/x", "params": [], "fallback": "decompose"}
                },
            ],
            "compute_distance": [
                {
                    "step_type": "distance_2d",
                    "description": "Distance between two 2D points",
                    "condition": "2D coordinates",
                    "dsl": {"type": "math", "script": "sqrt((x2-x1)**2 + (y2-y1)**2)", "params": ["x1", "y1", "x2", "y2"], "fallback": "decompose"}
                },
                {
                    "step_type": "distance_3d",
                    "description": "Distance between two 3D points",
                    "condition": "3D coordinates",
                    "dsl": {"type": "math", "script": "sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)", "params": ["x1", "y1", "z1", "x2", "y2", "z2"], "fallback": "decompose"}
                },
                {
                    "step_type": "distance_rate_time",
                    "description": "Distance from rate and time: d = rt",
                    "condition": "rate and time given",
                    "dsl": {"type": "math", "script": "rate * time", "params": ["rate", "time"], "fallback": "decompose"}
                },
            ],
            "evaluate_function": [
                {
                    "step_type": "eval_polynomial",
                    "description": "Evaluate polynomial at a point",
                    "condition": "polynomial function",
                    "dsl": {"type": "sympy", "script": "poly.subs(x, value)", "params": ["poly", "value"], "fallback": "decompose"}
                },
                {
                    "step_type": "eval_composed",
                    "description": "Evaluate composed function f(g(x))",
                    "condition": "function composition",
                    "dsl": {"type": "decompose", "script": "Evaluate inner function first, then outer", "params": [], "fallback": "decompose"}
                },
                {
                    "step_type": "eval_piecewise",
                    "description": "Evaluate piecewise function",
                    "condition": "piecewise definition",
                    "dsl": {"type": "decompose", "script": "Determine which piece applies, then evaluate", "params": [], "fallback": "decompose"}
                },
            ],
            "compute_remainder": [
                {
                    "step_type": "remainder_division",
                    "description": "Remainder from integer division",
                    "condition": "integer division",
                    "dsl": {"type": "math", "script": "a % b", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "remainder_polynomial",
                    "description": "Remainder from polynomial division",
                    "condition": "polynomial division",
                    "dsl": {"type": "sympy", "script": "rem(poly, divisor)", "params": ["poly", "divisor"], "fallback": "decompose"}
                },
            ],
            "compute_gcd": [
                {
                    "step_type": "gcd_two_numbers",
                    "description": "GCD of two numbers",
                    "condition": "two numbers",
                    "dsl": {"type": "math", "script": "gcd(a, b)", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "gcd_multiple",
                    "description": "GCD of multiple numbers",
                    "condition": "more than two numbers",
                    "dsl": {"type": "math", "script": "reduce(gcd, numbers)", "params": ["numbers"], "fallback": "decompose"}
                },
                {
                    "step_type": "gcd_euclidean",
                    "description": "GCD using Euclidean algorithm steps",
                    "condition": "showing work",
                    "dsl": {"type": "decompose", "script": "Apply Euclidean algorithm iteratively", "params": [], "fallback": "decompose"}
                },
            ],
            "find_minimum": [
                {
                    "step_type": "min_list",
                    "description": "Minimum of a list of values",
                    "condition": "discrete values",
                    "dsl": {"type": "math", "script": "min(values)", "params": ["values"], "fallback": "decompose"}
                },
                {
                    "step_type": "min_calculus",
                    "description": "Minimum using calculus (derivative = 0)",
                    "condition": "continuous function",
                    "dsl": {"type": "sympy", "script": "solve(diff(f, x), x)", "params": ["f"], "fallback": "decompose"}
                },
                {
                    "step_type": "min_vertex",
                    "description": "Minimum of parabola at vertex: x = -b/(2a)",
                    "condition": "quadratic function",
                    "dsl": {"type": "math", "script": "-b / (2*a)", "params": ["a", "b"], "fallback": "decompose"}
                },
            ],
            "apply_inequality": [
                {
                    "step_type": "inequality_solve",
                    "description": "Solve linear inequality",
                    "condition": "linear inequality",
                    "dsl": {"type": "sympy", "script": "solve_univariate_inequality(ineq, x)", "params": ["ineq"], "fallback": "decompose"}
                },
                {
                    "step_type": "inequality_triangle",
                    "description": "Apply triangle inequality",
                    "condition": "triangle inequality",
                    "dsl": {"type": "decompose", "script": "Check |a-b| < c < a+b", "params": [], "fallback": "decompose"}
                },
                {
                    "step_type": "inequality_cauchy",
                    "description": "Apply Cauchy-Schwarz inequality",
                    "condition": "Cauchy-Schwarz",
                    "dsl": {"type": "decompose", "script": "Apply (sum a_i*b_i)^2 <= (sum a_i^2)(sum b_i^2)", "params": [], "fallback": "decompose"}
                },
            ],
            "factor_polynomial": [
                {
                    "step_type": "factor_poly_gcf",
                    "description": "Factor out GCF from polynomial",
                    "condition": "common factor exists",
                    "dsl": {"type": "sympy", "script": "factor(poly)", "params": ["poly"], "fallback": "decompose"}
                },
                {
                    "step_type": "factor_poly_quadratic",
                    "description": "Factor quadratic polynomial",
                    "condition": "degree 2",
                    "dsl": {"type": "sympy", "script": "factor(a*x**2 + b*x + c)", "params": ["a", "b", "c"], "fallback": "decompose"}
                },
                {
                    "step_type": "factor_poly_grouping",
                    "description": "Factor by grouping",
                    "condition": "four terms",
                    "dsl": {"type": "decompose", "script": "Group terms and factor each group", "params": [], "fallback": "decompose"}
                },
            ],
            "compute_perimeter": [
                {
                    "step_type": "perimeter_rectangle",
                    "description": "Perimeter of rectangle: 2(l + w)",
                    "condition": "rectangle",
                    "dsl": {"type": "math", "script": "2 * (length + width)", "params": ["length", "width"], "fallback": "decompose"}
                },
                {
                    "step_type": "perimeter_triangle",
                    "description": "Perimeter of triangle: a + b + c",
                    "condition": "triangle",
                    "dsl": {"type": "math", "script": "a + b + c", "params": ["a", "b", "c"], "fallback": "decompose"}
                },
                {
                    "step_type": "perimeter_circle",
                    "description": "Circumference of circle: 2πr",
                    "condition": "circle",
                    "dsl": {"type": "math", "script": "2 * pi * radius", "params": ["radius"], "fallback": "decompose"}
                },
            ],
            "compute_lcm": [
                {
                    "step_type": "lcm_two_numbers",
                    "description": "LCM of two numbers: (a*b)/gcd(a,b)",
                    "condition": "two numbers",
                    "dsl": {"type": "math", "script": "(a * b) // gcd(a, b)", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "lcm_multiple",
                    "description": "LCM of multiple numbers",
                    "condition": "more than two numbers",
                    "dsl": {"type": "math", "script": "reduce(lcm, numbers)", "params": ["numbers"], "fallback": "decompose"}
                },
            ],
            "area_circle": [
                {
                    "step_type": "area_circle_radius",
                    "description": "Area of circle from radius: πr²",
                    "condition": "radius given",
                    "dsl": {"type": "math", "script": "pi * radius**2", "params": ["radius"], "fallback": "decompose"}
                },
                {
                    "step_type": "area_circle_diameter",
                    "description": "Area of circle from diameter: π(d/2)²",
                    "condition": "diameter given",
                    "dsl": {"type": "math", "script": "pi * (diameter/2)**2", "params": ["diameter"], "fallback": "decompose"}
                },
                {
                    "step_type": "area_circle_circumference",
                    "description": "Area of circle from circumference",
                    "condition": "circumference given",
                    "dsl": {"type": "math", "script": "circumference**2 / (4*pi)", "params": ["circumference"], "fallback": "decompose"}
                },
            ],
            "compute_quotient": [
                {
                    "step_type": "quotient_integer",
                    "description": "Integer division quotient",
                    "condition": "integer division",
                    "dsl": {"type": "math", "script": "a // b", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "quotient_decimal",
                    "description": "Decimal division",
                    "condition": "exact division",
                    "dsl": {"type": "math", "script": "a / b", "params": ["a", "b"], "fallback": "decompose"}
                },
                {
                    "step_type": "quotient_polynomial",
                    "description": "Polynomial division",
                    "condition": "polynomial division",
                    "dsl": {"type": "sympy", "script": "quo(poly, divisor)", "params": ["poly", "divisor"], "fallback": "decompose"}
                },
            ],
            # Added for deeper decomposition
            "binomial_coefficient": [
                {
                    "step_type": "binomial_formula",
                    "description": "Binomial coefficient using n!/(k!(n-k)!)",
                    "condition": "general binomial",
                    "dsl": {"type": "math", "script": "factorial(n) // (factorial(k) * factorial(n - k))", "params": ["n", "k"], "fallback": "decompose"}
                },
                {
                    "step_type": "binomial_symmetric",
                    "description": "Use C(n,k) = C(n,n-k) when k > n/2",
                    "condition": "k > n/2",
                    "dsl": {"type": "math", "script": "factorial(n) // (factorial(n - k) * factorial(k))", "params": ["n", "k"], "fallback": "decompose"}
                },
                {
                    "step_type": "binomial_pascals",
                    "description": "Use Pascal's identity: C(n,k) = C(n-1,k-1) + C(n-1,k)",
                    "condition": "recursive calculation",
                    "dsl": {"type": "decompose", "script": "Apply Pascal's triangle identity", "params": [], "fallback": "decompose"}
                },
            ],
            "count_ways": [
                {
                    "step_type": "ways_multiplication",
                    "description": "Count using multiplication principle: n1 * n2",
                    "condition": "independent choices",
                    "dsl": {"type": "math", "script": "n1 * n2", "params": ["n1", "n2"], "fallback": "decompose"}
                },
                {
                    "step_type": "ways_permutation",
                    "description": "Count permutations: n!/(n-r)!",
                    "condition": "ordered selection",
                    "dsl": {"type": "math", "script": "factorial(n) // factorial(n - r)", "params": ["n", "r"], "fallback": "decompose"}
                },
                {
                    "step_type": "ways_combination",
                    "description": "Count combinations: n!/(r!(n-r)!)",
                    "condition": "unordered selection",
                    "dsl": {"type": "math", "script": "factorial(n) // (factorial(r) * factorial(n - r))", "params": ["n", "r"], "fallback": "decompose"}
                },
                {
                    "step_type": "ways_power",
                    "description": "Count with replacement: n^r",
                    "condition": "selection with replacement",
                    "dsl": {"type": "math", "script": "n ** r", "params": ["n", "r"], "fallback": "decompose"}
                },
            ],
            "compute_area": [
                {
                    "step_type": "area_rectangle",
                    "description": "Area of rectangle: length * width",
                    "condition": "rectangle",
                    "dsl": {"type": "math", "script": "length * width", "params": ["length", "width"], "fallback": "decompose"}
                },
                {
                    "step_type": "area_triangle",
                    "description": "Area of triangle: (1/2) * base * height",
                    "condition": "triangle",
                    "dsl": {"type": "math", "script": "0.5 * base * height", "params": ["base", "height"], "fallback": "decompose"}
                },
                {
                    "step_type": "area_circle",
                    "description": "Area of circle: pi * r^2",
                    "condition": "circle",
                    "dsl": {"type": "math", "script": "pi * radius ** 2", "params": ["radius"], "fallback": "decompose"}
                },
                {
                    "step_type": "area_trapezoid",
                    "description": "Area of trapezoid: (1/2)(b1 + b2) * h",
                    "condition": "trapezoid",
                    "dsl": {"type": "math", "script": "0.5 * (b1 + b2) * height", "params": ["b1", "b2", "height"], "fallback": "decompose"}
                },
                {
                    "step_type": "area_parallelogram",
                    "description": "Area of parallelogram: base * height",
                    "condition": "parallelogram",
                    "dsl": {"type": "math", "script": "base * height", "params": ["base", "height"], "fallback": "decompose"}
                },
            ],
            "count_combinations": [
                {
                    "step_type": "combination_formula",
                    "description": "Standard combination: C(n,r) = n!/(r!(n-r)!)",
                    "condition": "selecting r from n",
                    "dsl": {"type": "math", "script": "factorial(n) // (factorial(r) * factorial(n - r))", "params": ["n", "r"], "fallback": "decompose"}
                },
                {
                    "step_type": "combination_multiset",
                    "description": "Multiset combination with repetition",
                    "condition": "with replacement",
                    "dsl": {"type": "math", "script": "factorial(n + r - 1) // (factorial(r) * factorial(n - 1))", "params": ["n", "r"], "fallback": "decompose"}
                },
                {
                    "step_type": "combination_stars_bars",
                    "description": "Stars and bars for distributing items",
                    "condition": "distribution problem",
                    "dsl": {"type": "math", "script": "factorial(n + k - 1) // (factorial(k - 1) * factorial(n))", "params": ["n", "k"], "fallback": "decompose"}
                },
                {
                    "step_type": "combination_complement",
                    "description": "Count by complementary method",
                    "condition": "easier to count excluded",
                    "dsl": {"type": "math", "script": "total - excluded", "params": ["total", "excluded"], "fallback": "decompose"}
                },
            ],
            "synthesize_results": [
                {
                    "step_type": "sum_components",
                    "description": "Add up computed components",
                    "condition": "summing parts",
                    "dsl": {"type": "math", "script": "sum(values)", "params": ["values"], "fallback": "decompose"}
                },
                {
                    "step_type": "multiply_components",
                    "description": "Multiply computed components",
                    "condition": "product of parts",
                    "dsl": {"type": "math", "script": "prod(values)", "params": ["values"], "fallback": "decompose"}
                },
                {
                    "step_type": "combine_fractions",
                    "description": "Combine fractions into final answer",
                    "condition": "fraction result",
                    "dsl": {"type": "math", "script": "numerator / denominator", "params": ["numerator", "denominator"], "fallback": "decompose"}
                },
                {
                    "step_type": "apply_final_operation",
                    "description": "Apply final operation to intermediate results",
                    "condition": "final calculation",
                    "dsl": {"type": "sympy", "script": "simplify(expr)", "params": ["expr"], "fallback": "decompose"}
                },
                {
                    "step_type": "extract_answer",
                    "description": "Extract final answer from computed values",
                    "condition": "answer extraction",
                    "dsl": {"type": "decompose", "script": "Identify the answer from computed values", "params": [], "fallback": "decompose"}
                },
            ],
        }

        return DECOMPOSITIONS.get(step_type, [])

    async def _fix_dsl(self, sig: StepSignature) -> Optional[str]:
        """Try to fix a broken DSL.

        For now, uses predefined fixes. Could use LLM in the future.
        """
        # Predefined DSL fixes for common issues
        DSL_FIXES = {
            "compute_sum": {
                "type": "math",
                "script": "sum(values)",  # Generic sum
                "params": ["values"],
                "fallback": "decompose"
            },
            "compute_product": {
                "type": "math",
                "script": "a * b",
                "params": ["a", "b"],
                "fallback": "decompose"
            },
        }

        if sig.step_type in DSL_FIXES:
            new_dsl = json.dumps(DSL_FIXES[sig.step_type])

            # Reset lift stats with new DSL version
            self.db.reset_lift_stats_for_dsl_version(
                signature_id=sig.id,
                new_dsl_script=new_dsl,
                new_dsl_version=(sig.dsl_version or 1) + 1,
            )

            logger.info("[refinement] Fixed DSL for sig=%d (%s)", sig.id, sig.step_type)
            return new_dsl

        return None


async def run_refinement_loop(
    db: StepSignatureDB = None,
    client = None,
    min_lift: float = -0.10,
    min_uses: int = 5,
    max_signatures: int = 20,
) -> RefinementReport:
    """Run the signature refinement loop.

    Convenience function to run refinement with default settings.

    Args:
        db: Database (creates new if None)
        client: LLM client (optional, for future LLM-assisted refinement)
        min_lift: Threshold for negative lift
        min_uses: Minimum uses for reliable data
        max_signatures: Max to process per run

    Returns:
        RefinementReport with results
    """
    if db is None:
        db = StepSignatureDB()

    refiner = SignatureRefiner(db, client)
    return await refiner.refine_negative_lift_signatures(
        min_lift=min_lift,
        min_uses=min_uses,
        max_signatures=max_signatures,
    )
