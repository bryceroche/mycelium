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
                    "dsl": {"type": "sympy", "script": "factor(expr)", "params": ["expr"], "fallback": "guidance"}
                },
                {
                    "step_type": "factor_quadratic",
                    "description": "Factor quadratic ax^2 + bx + c",
                    "condition": "quadratic expression",
                    "dsl": {"type": "sympy", "script": "factor(a*x**2 + b*x + c)", "params": ["a", "b", "c"], "fallback": "guidance"}
                },
                {
                    "step_type": "factor_difference_squares",
                    "description": "Factor a^2 - b^2 = (a+b)(a-b)",
                    "condition": "difference of squares",
                    "dsl": {"type": "math", "script": "(a + b) * (a - b)", "params": ["a", "b"], "fallback": "guidance"}
                },
            ],
            "substitute_value": [
                {
                    "step_type": "substitute_numeric",
                    "description": "Substitute numeric value into expression",
                    "condition": "substituting a number",
                    "dsl": {"type": "sympy", "script": "expr.subs(var, value)", "params": ["expr", "var", "value"], "fallback": "guidance"}
                },
                {
                    "step_type": "substitute_expression",
                    "description": "Substitute one expression for another",
                    "condition": "substituting an expression",
                    "dsl": {"type": "sympy", "script": "expr.subs(old, new)", "params": ["expr", "old", "new"], "fallback": "guidance"}
                },
            ],
            "solve_system": [
                {
                    "step_type": "solve_system_2x2",
                    "description": "Solve 2x2 system of linear equations",
                    "condition": "two equations, two unknowns",
                    "dsl": {"type": "sympy", "script": "solve([eq1, eq2], [x, y])", "params": ["eq1", "eq2"], "fallback": "guidance"}
                },
                {
                    "step_type": "solve_system_substitution",
                    "description": "Solve by substitution method",
                    "condition": "one equation easily isolates a variable",
                    "dsl": {"type": "guidance", "script": "Isolate one variable, substitute into other equation", "params": [], "fallback": "guidance"}
                },
            ],
            "compute_volume": [
                {
                    "step_type": "volume_box",
                    "description": "Volume of rectangular box: l * w * h",
                    "condition": "rectangular box or prism",
                    "dsl": {"type": "math", "script": "length * width * height", "params": ["length", "width", "height"], "fallback": "guidance"}
                },
                {
                    "step_type": "volume_cylinder",
                    "description": "Volume of cylinder: pi * r^2 * h",
                    "condition": "cylinder",
                    "dsl": {"type": "math", "script": "pi * radius**2 * height", "params": ["radius", "height"], "fallback": "guidance"}
                },
                {
                    "step_type": "volume_sphere",
                    "description": "Volume of sphere: (4/3) * pi * r^3",
                    "condition": "sphere",
                    "dsl": {"type": "math", "script": "(4/3) * pi * radius**3", "params": ["radius"], "fallback": "guidance"}
                },
                {
                    "step_type": "volume_cone",
                    "description": "Volume of cone: (1/3) * pi * r^2 * h",
                    "condition": "cone",
                    "dsl": {"type": "math", "script": "(1/3) * pi * radius**2 * height", "params": ["radius", "height"], "fallback": "guidance"}
                },
            ],
            "compute_sqrt": [
                {
                    "step_type": "sqrt_perfect",
                    "description": "Square root of perfect square",
                    "condition": "number is a perfect square",
                    "dsl": {"type": "math", "script": "int(sqrt(n))", "params": ["n"], "fallback": "guidance"}
                },
                {
                    "step_type": "sqrt_simplify",
                    "description": "Simplify sqrt by factoring out perfect squares",
                    "condition": "simplify radical",
                    "dsl": {"type": "sympy", "script": "sqrt(n).simplify()", "params": ["n"], "fallback": "guidance"}
                },
                {
                    "step_type": "sqrt_decimal",
                    "description": "Compute decimal approximation of sqrt",
                    "condition": "decimal answer needed",
                    "dsl": {"type": "math", "script": "sqrt(n)", "params": ["n"], "fallback": "guidance"}
                },
            ],
            "general_step": [
                {
                    "step_type": "extract_value",
                    "description": "Extract a numeric value from problem text",
                    "condition": "extracting given values",
                    "dsl": {"type": "guidance", "script": "Identify and extract the numeric value", "params": [], "fallback": "guidance"}
                },
                {
                    "step_type": "apply_formula",
                    "description": "Apply a known formula with given values",
                    "condition": "applying a formula",
                    "dsl": {"type": "sympy", "script": "formula.subs(substitutions)", "params": ["formula", "substitutions"], "fallback": "guidance"}
                },
                {
                    "step_type": "compare_values",
                    "description": "Compare two values and determine relationship",
                    "condition": "comparing quantities",
                    "dsl": {"type": "math", "script": "a - b", "params": ["a", "b"], "fallback": "guidance"}
                },
                {
                    "step_type": "unit_conversion",
                    "description": "Convert between units",
                    "condition": "unit conversion needed",
                    "dsl": {"type": "math", "script": "value * conversion_factor", "params": ["value", "conversion_factor"], "fallback": "guidance"}
                },
            ],
            "solve_equation": [
                {
                    "step_type": "solve_linear_single",
                    "description": "Solve ax + b = c for x",
                    "condition": "linear equation with one variable",
                    "dsl": {"type": "math", "script": "(c - b) / a", "params": ["a", "b", "c"], "fallback": "guidance"}
                },
                {
                    "step_type": "solve_quadratic_formula",
                    "description": "Solve ax^2 + bx + c = 0 using quadratic formula",
                    "condition": "quadratic equation",
                    "dsl": {"type": "sympy", "script": "solve(a*x**2 + b*x + c, x)", "params": ["a", "b", "c"], "fallback": "guidance"}
                },
                {
                    "step_type": "solve_proportion",
                    "description": "Solve a/b = c/d for one variable",
                    "condition": "proportion or ratio equation",
                    "dsl": {"type": "math", "script": "(a * d) / b", "params": ["a", "b", "d"], "fallback": "guidance"}
                },
                {
                    "step_type": "isolate_variable",
                    "description": "Isolate a variable from a formula",
                    "condition": "rearranging formula for a variable",
                    "dsl": {"type": "sympy", "script": "solve(equation, target_var)", "params": ["equation", "target_var"], "fallback": "guidance"}
                },
            ],
            "simplify_expression": [
                {
                    "step_type": "simplify_fraction",
                    "description": "Reduce a fraction to lowest terms using GCD",
                    "condition": "fraction to reduce",
                    "dsl": {"type": "math", "script": "numerator // gcd(numerator, denominator)", "params": ["numerator", "denominator"], "fallback": "guidance"}
                },
                {
                    "step_type": "combine_like_terms",
                    "description": "Combine like terms in an algebraic expression",
                    "condition": "polynomial with like terms",
                    "dsl": {"type": "sympy", "script": "simplify(expand(expr))", "params": ["expr"], "fallback": "guidance"}
                },
                {
                    "step_type": "simplify_radical",
                    "description": "Simplify radical expressions (square roots, etc.)",
                    "condition": "expression with radicals",
                    "dsl": {"type": "sympy", "script": "simplify(sqrt(n))", "params": ["n"], "fallback": "guidance"}
                },
                {
                    "step_type": "factor_common",
                    "description": "Factor out common terms from expression",
                    "condition": "expression with common factors",
                    "dsl": {"type": "sympy", "script": "factor(expr)", "params": ["expr"], "fallback": "guidance"}
                },
            ],
            "count_items": [
                {
                    "step_type": "count_simple",
                    "description": "Count items in a set or list",
                    "condition": "simple counting",
                    "dsl": {"type": "math", "script": "n", "params": ["n"], "fallback": "guidance"}
                },
                {
                    "step_type": "count_with_condition",
                    "description": "Count items satisfying a condition",
                    "condition": "conditional counting",
                    "dsl": {"type": "guidance", "script": "Count items where condition holds", "params": [], "fallback": "guidance"}
                },
                {
                    "step_type": "count_complement",
                    "description": "Count by subtracting from total",
                    "condition": "easier to count what's excluded",
                    "dsl": {"type": "math", "script": "total - excluded", "params": ["total", "excluded"], "fallback": "guidance"}
                },
            ],
            "common_difference": [
                {
                    "step_type": "diff_arithmetic_seq",
                    "description": "Find common difference d = a2 - a1",
                    "condition": "arithmetic sequence given",
                    "dsl": {"type": "math", "script": "a2 - a1", "params": ["a1", "a2"], "fallback": "guidance"}
                },
                {
                    "step_type": "diff_from_formula",
                    "description": "Find difference from nth term formula",
                    "condition": "formula for nth term given",
                    "dsl": {"type": "math", "script": "(an - a1) / (n - 1)", "params": ["a1", "an", "n"], "fallback": "guidance"}
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
                "fallback": "guidance"
            },
            "compute_product": {
                "type": "math",
                "script": "a * b",
                "params": ["a", "b"],
                "fallback": "guidance"
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
