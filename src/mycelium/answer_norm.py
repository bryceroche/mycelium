"""Answer normalization for mathematical expressions.

Handles LaTeX, fractions, decimals, and numeric equivalence checking.
Includes LLM judge for semantic equivalence.
"""

import asyncio
import logging
import re
from fractions import Fraction
from typing import Optional
import math

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Judge for answer equivalence
# =============================================================================

async def answers_equivalent_llm(
    predicted: str,
    expected: str,
    problem: Optional[str] = None,
) -> bool:
    """Use LLM to judge if two answers are mathematically equivalent.

    This is more robust than regex-based normalization and handles:
    - Different notations (1/2 vs 0.5 vs "half")
    - Units and formatting (800,000 vs 800000)
    - Semantic equivalence ("2 feet" vs "24 inches")

    Args:
        predicted: The model's predicted answer
        expected: The ground truth answer
        problem: Optional problem context for better judgment

    Returns:
        True if answers are equivalent
    """
    from mycelium.client import ask_llama

    # Empty prediction is always wrong (don't let LLM give false positives)
    if not predicted or not predicted.strip():
        return False

    # Quick string equality check first (avoid LLM call if obvious match)
    if predicted.strip().lower() == expected.strip().lower():
        return True

    context = f"\nProblem context: {problem}" if problem else ""

    prompt = f"""Are these two mathematical answers NUMERICALLY equivalent?

STRICT RULES:
- Numbers must evaluate to the SAME value (e.g., 0.5 = 1/2 = 50%)
- Different numbers are NOT equivalent (e.g., 0.83 != 7)
- Be strict: if in doubt, answer NO
{context}

Expected answer: {expected}
Predicted answer: {predicted}

Reply with ONLY 'YES' or 'NO'. Nothing else."""

    try:
        response = await ask_llama(prompt, temperature=0.0)
        result = "YES" in response.upper()
        logger.debug(f"LLM judge: '{predicted}' vs '{expected}' -> {result}")
        return result
    except Exception as e:
        logger.warning(f"LLM judge failed, falling back to regex: {e}")
        # Fallback to regex-based comparison
        return answers_equivalent(predicted, expected)


def answers_equivalent_llm_sync(
    predicted: str,
    expected: str,
    problem: Optional[str] = None,
) -> bool:
    """Synchronous wrapper for answers_equivalent_llm."""
    return asyncio.run(answers_equivalent_llm(predicted, expected, problem))

# =============================================================================
# Pre-compiled regex patterns (compiled once at module load for performance)
# =============================================================================

# LaTeX patterns
_RE_LATEX_FRAC = re.compile(
    r"\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
)
_RE_LATEX_SQRT_N = re.compile(r"\\sqrt\[(\d+)\]\{([^{}]+)\}")
_RE_LATEX_SQRT = re.compile(r"\\sqrt\{([^{}]+)\}")
_RE_LATEX_EXPONENT = re.compile(r"\^\{([^{}]+)\}")
_RE_LATEX_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_RE_LATEX_DISPLAY_MATH = re.compile(r"\$\$([^$]+)\$\$")
_RE_LATEX_INLINE_MATH = re.compile(r"\$([^$]+)\$")
_RE_LATEX_DELIMITERS = re.compile(r"\\(left|right|big|Big|bigg|Bigg)([|()[\]{}])?")
_RE_LATEX_TEXT_CMDS = re.compile(r"\\(text|mathrm|mathbf|mathit|textbf)\{([^{}]+)\}")
_RE_LATEX_CDOT_TIMES = re.compile(r"\\(cdot|times)")
_RE_LATEX_DIV = re.compile(r"\\div")
_RE_LATEX_PM = re.compile(r"\\pm")
_RE_LATEX_PI = re.compile(r"\\pi")
_RE_LATEX_INFTY = re.compile(r"\\infty")
_RE_LATEX_APPROX = re.compile(r"\\approx")
_RE_LATEX_NEQ = re.compile(r"\\neq")
_RE_LATEX_LEQ = re.compile(r"\\leq?")
_RE_LATEX_GEQ = re.compile(r"\\geq?")
_RE_LATEX_REMAINING = re.compile(r"\\[a-zA-Z]+")

# Numeric parsing patterns
_RE_FRACTION = re.compile(r"^(-?\d+)\s*/\s*(-?\d+)$")
# Matches (a)/(b), -(a)/(b), (-a)/(b), (a)/(-b)
_RE_PAREN_FRACTION = re.compile(r"^(-?)\((-?\d+)\)\s*/\s*\((-?\d+)\)$")
_RE_SQRT_NUMERIC = re.compile(r"^sqrt\((\d+(?:\.\d+)?)\)$")
_RE_NTH_ROOT = re.compile(r"^\(?(\d+(?:\.\d+)?)\)?\^\(1/(\d+)\)$")
_RE_SIMPLE_POWER = re.compile(r"^(\d+(?:\.\d+)?)\^(\d+)$")
_RE_DECIMAL = re.compile(r"^(-?\d+)\.(\d+)$")

# Normalization patterns
_RE_CURRENCY = re.compile(r"[\$€£]")
_RE_PERCENT_SYMBOL = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_RE_PERCENT_WORD = re.compile(r"(-?\d+(?:\.\d+)?)\s*percent\b", re.IGNORECASE)
_RE_UNITS = re.compile(
    r"\s*(cm|mm|m|km|kg|g|mg|lb|oz|dollars?|units?|feet|foot|ft|inches?|in)\s*$", re.IGNORECASE
)
_RE_COMMA_IN_NUMBER = re.compile(r"(\d),(\d)")
_RE_FRACTION_SPACING = re.compile(r"(\d+)\s*/\s*(\d+)")
_RE_DECIMAL_INTEGER = re.compile(r"(\d+)\.0+$")


def _parse_latex_frac(s: str) -> str:
    r"""Convert LaTeX \frac{a}{b}, \tfrac{a}{b}, \dfrac{a}{b} to a/b."""
    # Normalize all fraction variants to \frac first
    s = s.replace(r"\tfrac", r"\frac")
    s = s.replace(r"\dfrac", r"\frac")
    # Handle nested fracs by working from innermost outward
    while r"\frac{" in s:
        new_s = _RE_LATEX_FRAC.sub(r"(\1)/(\2)", s)
        if new_s == s:  # No change, avoid infinite loop
            break
        s = new_s
    return s


def _parse_latex_sqrt(s: str) -> str:
    r"""Convert LaTeX \sqrt{x} to sqrt(x) and \sqrt[n]{x} to x^(1/n)."""
    # \sqrt[n]{x} -> x^(1/n)
    s = _RE_LATEX_SQRT_N.sub(r"(\2)^(1/\1)", s)
    # \sqrt{x} -> sqrt(x)
    s = _RE_LATEX_SQRT.sub(r"sqrt(\1)", s)
    return s


def _parse_latex_exponents(s: str) -> str:
    """Convert LaTeX x^{y} to x^(y) and handle simple x^2."""
    # x^{expr} -> x^(expr)
    s = _RE_LATEX_EXPONENT.sub(r"^(\1)", s)
    return s


def _strip_latex(s: str) -> str:
    """Remove LaTeX markup while preserving mathematical content."""
    # Remove \boxed{...} - pattern handles nested braces
    while r"\boxed{" in s:
        new_s = _RE_LATEX_BOXED.sub(r"\1", s)
        if new_s == s:  # No change, avoid infinite loop
            break
        s = new_s

    # Remove $...$ and $$...$$
    s = _RE_LATEX_DISPLAY_MATH.sub(r"\1", s)
    s = _RE_LATEX_INLINE_MATH.sub(r"\1", s)

    # Convert LaTeX fractions
    s = _parse_latex_frac(s)

    # Convert sqrt
    s = _parse_latex_sqrt(s)

    # Convert exponents
    s = _parse_latex_exponents(s)

    # Remove common LaTeX commands
    s = _RE_LATEX_DELIMITERS.sub(r"\2", s)
    s = _RE_LATEX_TEXT_CMDS.sub(r"\2", s)
    s = _RE_LATEX_CDOT_TIMES.sub("*", s)
    s = _RE_LATEX_DIV.sub("/", s)
    s = _RE_LATEX_PM.sub("±", s)
    s = _RE_LATEX_PI.sub("pi", s)
    s = _RE_LATEX_INFTY.sub("inf", s)
    s = _RE_LATEX_APPROX.sub("≈", s)
    s = _RE_LATEX_NEQ.sub("≠", s)
    s = _RE_LATEX_LEQ.sub("<=", s)
    s = _RE_LATEX_GEQ.sub(">=", s)
    s = _RE_LATEX_REMAINING.sub("", s)  # Remove any remaining commands

    return s


def _try_parse_numeric(s: str) -> Optional[float]:
    """Try to parse a string as a numeric value."""
    s = s.strip()

    # Direct float
    try:
        return float(s)
    except ValueError:
        logger.debug("Failed to parse '%s' as float", s)

    # Fraction a/b
    frac_match = _RE_FRACTION.match(s)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            return num / den

    # Parenthesized fraction (a)/(b) or -(a)/(b)
    frac_match = _RE_PAREN_FRACTION.match(s)
    if frac_match:
        sign = -1 if frac_match.group(1) == '-' else 1
        num, den = int(frac_match.group(2)), int(frac_match.group(3))
        if den != 0:
            return sign * num / den

    # sqrt(x) for numeric x
    sqrt_match = _RE_SQRT_NUMERIC.match(s)
    if sqrt_match:
        return math.sqrt(float(sqrt_match.group(1)))

    # x^(1/n) for numeric x and n (nth root) - handles both 8^(1/3) and (8)^(1/3)
    root_match = _RE_NTH_ROOT.match(s)
    if root_match:
        base, n = float(root_match.group(1)), int(root_match.group(2))
        if n != 0:
            return base ** (1/n)

    # Simple power x^n
    pow_match = _RE_SIMPLE_POWER.match(s)
    if pow_match:
        base, exp = float(pow_match.group(1)), int(pow_match.group(2))
        return base ** exp

    # pi constant
    if s == "pi":
        return math.pi

    return None


def _try_as_fraction(s: str) -> Optional[Fraction]:
    """Try to parse string as an exact fraction."""
    s = s.strip()

    # Direct integer
    try:
        return Fraction(int(s))
    except ValueError:
        logger.debug("Failed to parse '%s' as integer fraction", s)

    # Fraction a/b
    frac_match = _RE_FRACTION.match(s)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            return Fraction(num, den)

    # Parenthesized fraction (a)/(b) or -(a)/(b)
    frac_match = _RE_PAREN_FRACTION.match(s)
    if frac_match:
        sign = -1 if frac_match.group(1) == '-' else 1
        num, den = int(frac_match.group(2)), int(frac_match.group(3))
        if den != 0:
            return Fraction(sign * num, den)

    # Decimal (try to convert to fraction for exact comparison)
    decimal_match = _RE_DECIMAL.match(s)
    if decimal_match:
        try:
            return Fraction(s).limit_denominator(10000)
        except (ValueError, ZeroDivisionError) as e:
            logger.debug("Failed to convert decimal '%s' to fraction: %s", s, e)

    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison.

    This performs string normalization:
    - Strips LaTeX markup
    - Normalizes whitespace
    - Removes currency symbols and units
    - Normalizes fraction formatting
    - Converts decimal integers (15.0 -> 15)
    """
    if not answer:
        return ""

    # Strip LaTeX
    answer = _strip_latex(answer)

    # Remove currency symbols
    answer = _RE_CURRENCY.sub("", answer)

    # Convert percentages to decimals BEFORE stripping units
    # Matches: "50%", "50 %", "50 percent", "50percent"
    # Converts: 50% -> 0.5, 100% -> 1, 25% -> 0.25
    def _percent_to_decimal(match: re.Match) -> str:
        value = float(match.group(1))
        decimal = value / 100
        # Format nicely: avoid trailing zeros, but keep full precision
        if decimal == int(decimal):
            return str(int(decimal))
        else:
            # Use repr for full precision, then clean up trailing zeros
            # This handles extreme values like 0.0001% correctly
            result = repr(decimal)
            # repr may use scientific notation for very small numbers - keep it
            if 'e' not in result and '.' in result:
                result = result.rstrip("0").rstrip(".")
            return result

    # Match "50%" or "50 %" (percent symbol doesn't need word boundary)
    answer = _RE_PERCENT_SYMBOL.sub(_percent_to_decimal, answer)
    # Match "50 percent" or "50percent" (word needs boundary)
    answer = _RE_PERCENT_WORD.sub(_percent_to_decimal, answer)

    # Remove trailing units (percent/% already handled above)
    answer = _RE_UNITS.sub("", answer)

    # Remove commas from numbers (800,000 -> 800000)
    answer = _RE_COMMA_IN_NUMBER.sub(r"\1\2", answer)

    # Normalize whitespace
    answer = " ".join(answer.split())

    # Normalize fraction spacing: a / b -> a/b
    answer = _RE_FRACTION_SPACING.sub(r"\1/\2", answer)

    # Normalize decimal integers: 15.0 -> 15, 15.00 -> 15
    answer = _RE_DECIMAL_INTEGER.sub(r"\1", answer)

    # Remove trailing periods/commas
    answer = answer.rstrip(".,")

    return answer.strip().lower()


def answers_equivalent(a: str, b: str, tolerance: float = 1e-6) -> bool:
    """Check if two answers are mathematically equivalent.

    Goes beyond string comparison to check numeric equivalence:
    - "1/2" == "0.5" == "\\frac{1}{2}"
    - "2" == "2.0" == "2.00"
    - "sqrt(4)" == "2"

    Args:
        a: First answer
        b: Second answer
        tolerance: Tolerance for floating point comparison

    Returns:
        True if answers are equivalent
    """
    # First try string equality after normalization
    norm_a = normalize_answer(a)
    norm_b = normalize_answer(b)

    if norm_a == norm_b:
        return True

    # Try exact fraction comparison (for cases like 1/2 == 2/4)
    frac_a = _try_as_fraction(norm_a)
    frac_b = _try_as_fraction(norm_b)

    if frac_a is not None and frac_b is not None:
        if frac_a == frac_b:
            return True

    # Try numeric comparison
    num_a = _try_parse_numeric(norm_a)
    num_b = _try_parse_numeric(norm_b)

    if num_a is not None and num_b is not None:
        # Use absolute tolerance for small numbers, relative for large
        max_abs = max(abs(num_a), abs(num_b))
        if max_abs < tolerance:
            # Both effectively zero
            return True
        elif max_abs > 1:
            # Relative tolerance for larger numbers
            rel_diff = abs(num_a - num_b) / max_abs
            if rel_diff < tolerance:
                return True
        else:
            # Absolute tolerance for small numbers
            if abs(num_a - num_b) < tolerance:
                return True

    return False
