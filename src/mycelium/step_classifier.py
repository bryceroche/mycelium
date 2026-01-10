"""LLM-based step type classification.

Uses a fast LLM to classify mathematical problem steps into semantic types.
Falls back to heuristic classification if LLM is unavailable.
"""

import asyncio
import hashlib
import logging
import os
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Known step types with descriptions for the LLM
# Organized by category for clarity
STEP_TYPES = {
    # === Algebra ===
    "solve_equation": "Solving an equation or system of equations for unknown variables",
    "simplify_expression": "Simplifying, reducing, or combining algebraic expressions",
    "factor_expression": "Factoring polynomials or expressions",
    "expand_expression": "Expanding or distributing algebraic expressions",
    "substitute_value": "Substituting known values into expressions or formulas",

    # === Arithmetic ===
    "compute_sum": "Adding numbers or expressions together",
    "compute_product": "Multiplying numbers or expressions",
    "compute_division": "Dividing numbers or computing ratios/fractions",
    "compute_percentage": "Calculating percentages, ratios, or proportions",
    "compute_power": "Computing powers, exponents, or roots",

    # === Combinatorics & Counting ===
    "compute_factorial": "Computing factorials (n!)",
    "compute_combinations": "Computing combinations C(n,k), permutations P(n,k), or binomial coefficients",
    "count_arrangements": "Counting arrangements, orderings, or ways to select items",
    "compute_probability": "Calculating probabilities of events",

    # === Geometry ===
    "compute_geometry": "Calculating geometric properties (area, perimeter, volume)",
    "compute_angle": "Computing or finding angles in geometric figures",
    "apply_pythagorean": "Applying the Pythagorean theorem",

    # === Trigonometry ===
    "compute_trig": "Computing trigonometric values (sin, cos, tan, etc.)",
    "apply_trig_identity": "Applying trigonometric identities or formulas",
    "convert_angle": "Converting angles between degrees and radians",

    # === Functions & Calculus ===
    "evaluate_function": "Evaluating a function at specific values",
    "compute_derivative": "Computing derivatives or rates of change",
    "compute_integral": "Computing integrals or areas under curves",
    "find_extrema": "Finding maxima, minima, or critical points",

    # === Number Theory ===
    "find_divisibility": "Finding divisors, factors, GCD, or LCM",
    "compute_modular": "Computing modular arithmetic or remainders",
    "find_prime": "Finding or testing prime numbers",

    # === Linear Algebra & Vectors ===
    "compute_vector": "Computing vector operations (dot product, cross product, projection)",
    "compute_matrix": "Computing matrix operations or transformations",

    # === Problem Setup & Synthesis ===
    "setup_equation": "Translating a word problem into mathematical equations",
    "define_variable": "Defining or introducing variables for a problem",
    "apply_formula": "Applying a specific mathematical formula or theorem",
    "convert_units": "Converting between different units of measurement",
    "compare_values": "Comparing quantities or determining relationships",
    "synthesize_answer": "Combining intermediate results into a final answer",

    # === Fallback ===
    "general_step": "General mathematical operation not fitting other categories",
}

# Prompt for step classification
CLASSIFY_SYSTEM_PROMPT = """You are a mathematical step classifier. Given a step description from a math problem, classify it into exactly ONE of these types:

{type_list}

Respond with ONLY the type name (e.g., "solve_equation"), nothing else."""

CLASSIFY_USER_PROMPT = """Classify this math step:
"{step_text}"

Type:"""


def _build_type_list() -> str:
    """Build formatted type list for prompt."""
    return "\n".join(f"- {name}: {desc}" for name, desc in STEP_TYPES.items())


class StepClassifier:
    """LLM-based step type classifier with caching and fallback."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",  # Use same model as solver
        use_llm: bool = True,
        cache_size: int = 1024,
    ):
        """Initialize the classifier.

        Args:
            model: LLM model to use for classification
            use_llm: If False, only use heuristic classification
            cache_size: Max cached classifications (LRU)
        """
        self.model = model
        self.use_llm = use_llm and os.getenv("GROQ_API_KEY") is not None
        self._cache: dict[str, str] = {}
        self._cache_size = cache_size
        self._client = None

    def _get_client(self):
        """Lazy-load the Groq client."""
        if self._client is None:
            from .client import GroqClient
            self._client = GroqClient(model=self.model)
        return self._client

    def _cache_key(self, step_text: str) -> str:
        """Generate cache key from step text."""
        # Normalize whitespace and case for better cache hits
        normalized = " ".join(step_text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _evict_if_needed(self):
        """Evict oldest entries if cache is full."""
        while len(self._cache) >= self._cache_size:
            # Remove first (oldest) entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]

    async def classify(self, step_text: str) -> str:
        """Classify a step into a semantic type.

        Args:
            step_text: The step description to classify

        Returns:
            Step type string (e.g., "solve_equation")
        """
        # Check cache first
        cache_key = self._cache_key(step_text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try LLM classification
        if self.use_llm:
            try:
                step_type = await self._classify_with_llm(step_text)
                if step_type in STEP_TYPES:
                    self._evict_if_needed()
                    self._cache[cache_key] = step_type
                    return step_type
                else:
                    logger.warning(f"LLM returned unknown type '{step_type}', falling back")
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}, falling back to heuristic")

        # Fallback to heuristic
        step_type = self._classify_heuristic(step_text)
        self._evict_if_needed()
        self._cache[cache_key] = step_type
        return step_type

    def classify_sync(self, step_text: str) -> str:
        """Synchronous classification (uses heuristic only to avoid blocking).

        For sync contexts, we only use heuristics to avoid asyncio complications.
        Use classify() for full LLM-backed classification.

        Args:
            step_text: The step description to classify

        Returns:
            Step type string
        """
        cache_key = self._cache_key(step_text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        step_type = self._classify_heuristic(step_text)
        self._evict_if_needed()
        self._cache[cache_key] = step_type
        return step_type

    async def _classify_with_llm(self, step_text: str) -> str:
        """Classify using LLM."""
        client = self._get_client()

        system_prompt = CLASSIFY_SYSTEM_PROMPT.format(type_list=_build_type_list())
        user_prompt = CLASSIFY_USER_PROMPT.format(step_text=step_text[:500])  # Truncate long steps

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await client.generate(
            messages,
            temperature=0.0,  # Deterministic classification
            max_tokens=32,  # Only need the type name
        )

        # Extract type from response (handle potential formatting)
        step_type = response.strip().lower().replace(" ", "_")
        # Remove any quotes or extra punctuation
        step_type = re.sub(r'["\'\.\,]', '', step_type)

        return step_type

    def _classify_heuristic(self, step_text: str) -> str:
        """Classify using keyword heuristics (fallback method).

        Patterns are ordered by specificity - more specific patterns first
        to avoid overly general matches.
        """
        text_lower = step_text.lower()

        # Pattern matching for common step types (in priority order - more specific first)
        patterns = [
            # === Calculus (most specific first) ===
            (["antiderivative", "integral", "integrate", "∫"], "compute_integral"),
            (["derivative", "differentiate", "d/dx", "rate of change"], "compute_derivative"),
            (["maximum", "minimum", "extrema", "critical point", "optimize"], "find_extrema"),

            # === Combinatorics & Counting (before probability) ===
            (["factorial", "n!", "!"], "compute_factorial"),
            (["choose", "c(", "binomial", "\\binom", "dbinom", "ncr", "npr", "permutation", "combination"], "compute_combinations"),
            (["arrange", "arrangement", "ways to order", "ways to select", "how many ways"], "count_arrangements"),
            (["probability", "chance", "likely", "odds"], "compute_probability"),

            # === Trigonometry (before geometry - trig is more specific) ===
            (["sin", "cos", "tan", "cot", "sec", "csc", "trigonometric"], "compute_trig"),
            (["trig identity", "sum-to-product", "product-to-sum", "double angle", "half angle"], "apply_trig_identity"),
            (["radians", "degrees", "convert angle", "to radians", "to degrees", "°"], "convert_angle"),

            # === Linear Algebra & Vectors ===
            (["dot product", "cross product", "projection", "vector", "\\mathbf", "\\vec"], "compute_vector"),
            (["matrix", "matrices", "determinant", "eigenvalue", "transpose"], "compute_matrix"),

            # === Number Theory ===
            (["gcd", "lcm", "greatest common", "least common", "divisor", "divisible"], "find_divisibility"),
            (["mod ", "modulo", "remainder", "≡", "pmod", "congruent"], "compute_modular"),
            (["prime", "composite", "factor"], "find_prime"),

            # === Powers & Roots ===
            (["exponent", "power", "^", "sqrt", "square root", "cube root", "\\sqrt"], "compute_power"),

            # === Algebra ===
            (["substitute", "plug in", "replace with", "let x =", "given that"], "substitute_value"),
            (["expand", "distribute", "foil", "multiply out"], "expand_expression"),
            (["factor", "factorize", "factor out", "factored form"], "factor_expression"),
            (["simplify", "reduce", "combine like", "collect terms"], "simplify_expression"),

            # === Geometry ===
            (["pythagorean", "a² + b²", "a^2 + b^2", "hypotenuse"], "apply_pythagorean"),
            (["angle", "degree", "∠"], "compute_angle"),
            (["area", "perimeter", "volume", "circumference", "surface area"], "compute_geometry"),
            (["triangle", "circle", "rectangle", "square", "polygon", "radius", "diameter"], "compute_geometry"),

            # === Functions ===
            (["evaluate", "f(", "g(", "h(", "at x =", "when x =", "find f("], "evaluate_function"),

            # === Equations (after more specific types) ===
            (["solve for", "find x", "find y", "find the value", "= 0", "isolate"], "solve_equation"),
            (["set up equation", "write equation", "translate to equation", "express as equation"], "setup_equation"),
            (["define", "let", "denote", "introduce variable"], "define_variable"),
            (["equation", "solve"], "solve_equation"),  # Lower priority catchall

            # === Arithmetic (generic - lower priority) ===
            (["percent", "%", "proportion", "ratio of"], "compute_percentage"),
            (["add", "sum", "total", "plus", "+"], "compute_sum"),
            (["multiply", "product", "times", "×", "*"], "compute_product"),
            (["divide", "quotient", "fraction", "÷", "/"], "compute_division"),

            # === General patterns ===
            (["formula", "theorem", "apply", "use the", "by the"], "apply_formula"),
            (["convert", "unit", "meters to", "feet to", "celsius", "fahrenheit"], "convert_units"),
            (["compare", "greater", "less", "equal", "which is", "larger", "smaller"], "compare_values"),
            (["final answer", "combine results", "therefore", "thus", "answer is", "result is"], "synthesize_answer"),
        ]

        for keywords, step_type in patterns:
            if any(kw in text_lower for kw in keywords):
                return step_type

        return "general_step"


# Module-level singleton for convenience
_default_classifier: Optional[StepClassifier] = None


def get_classifier() -> StepClassifier:
    """Get the default step classifier instance."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = StepClassifier()
    return _default_classifier


async def classify_step(step_text: str) -> str:
    """Convenience function to classify a step using the default classifier.

    Args:
        step_text: The step description to classify

    Returns:
        Step type string
    """
    return await get_classifier().classify(step_text)


def classify_step_sync(step_text: str) -> str:
    """Synchronous convenience function (heuristic only).

    Args:
        step_text: The step description to classify

    Returns:
        Step type string
    """
    return get_classifier().classify_sync(step_text)
