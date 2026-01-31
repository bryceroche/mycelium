"""Answer normalization for mathematical expressions."""

import logging
import math
import re
from fractions import Fraction
from typing import Optional

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns
_RE_LATEX_FRAC = re.compile(r"\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
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
_RE_LATEX_REMAINING = re.compile(r"\\[a-zA-Z]+")

_RE_FRACTION = re.compile(r"^(-?\d+)\s*/\s*(-?\d+)$")
_RE_PAREN_FRACTION = re.compile(r"^(-?)\((-?\d+)\)\s*/\s*\((-?\d+)\)$")
_RE_SQRT_NUMERIC = re.compile(r"^sqrt\((\d+(?:\.\d+)?)\)$")
_RE_NTH_ROOT = re.compile(r"^\(?(\d+(?:\.\d+)?)\)?\^\(1/(\d+)\)$")
_RE_SIMPLE_POWER = re.compile(r"^(\d+(?:\.\d+)?)\^(\d+)$")
_RE_DECIMAL = re.compile(r"^(-?\d+)\.(\d+)$")

_RE_CURRENCY = re.compile(r"[\$€£]")
_RE_PERCENT_SYMBOL = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_RE_PERCENT_WORD = re.compile(r"(-?\d+(?:\.\d+)?)\s*percent\b", re.IGNORECASE)
_RE_UNITS = re.compile(r"\s*(cm|mm|m|km|kg|g|mg|lb|oz|dollars?|units?|feet|foot|ft|inches?|in)\s*$", re.IGNORECASE)
_RE_COMMA_IN_NUMBER = re.compile(r"(\d),(\d)")
_RE_FRACTION_SPACING = re.compile(r"(\d+)\s*/\s*(\d+)")
_RE_DECIMAL_INTEGER = re.compile(r"(\d+)\.0+$")


def _parse_latex_frac(s: str) -> str:
    s = s.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    while r"\frac{" in s:
        new_s = _RE_LATEX_FRAC.sub(r"(\1)/(\2)", s)
        if new_s == s:
            break
        s = new_s
    return s


def _parse_latex_sqrt(s: str) -> str:
    s = _RE_LATEX_SQRT_N.sub(r"(\2)^(1/\1)", s)
    s = _RE_LATEX_SQRT.sub(r"sqrt(\1)", s)
    return s


def _strip_latex(s: str) -> str:
    while r"\boxed{" in s:
        new_s = _RE_LATEX_BOXED.sub(r"\1", s)
        if new_s == s:
            break
        s = new_s
    s = _RE_LATEX_DISPLAY_MATH.sub(r"\1", s)
    s = _RE_LATEX_INLINE_MATH.sub(r"\1", s)
    s = _parse_latex_frac(s)
    s = _parse_latex_sqrt(s)
    s = _RE_LATEX_EXPONENT.sub(r"^(\1)", s)
    s = _RE_LATEX_DELIMITERS.sub(r"\2", s)
    s = _RE_LATEX_TEXT_CMDS.sub(r"\2", s)
    s = _RE_LATEX_CDOT_TIMES.sub("*", s)
    s = _RE_LATEX_DIV.sub("/", s)
    s = _RE_LATEX_PM.sub("±", s)
    s = _RE_LATEX_PI.sub("pi", s)
    s = _RE_LATEX_INFTY.sub("inf", s)
    s = _RE_LATEX_REMAINING.sub("", s)
    return s


def _try_parse_numeric(s: str) -> Optional[float]:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass

    frac_match = _RE_FRACTION.match(s)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            return num / den

    frac_match = _RE_PAREN_FRACTION.match(s)
    if frac_match:
        sign = -1 if frac_match.group(1) == '-' else 1
        num, den = int(frac_match.group(2)), int(frac_match.group(3))
        if den != 0:
            return sign * num / den

    sqrt_match = _RE_SQRT_NUMERIC.match(s)
    if sqrt_match:
        return math.sqrt(float(sqrt_match.group(1)))

    root_match = _RE_NTH_ROOT.match(s)
    if root_match:
        base, n = float(root_match.group(1)), int(root_match.group(2))
        if n != 0:
            return base ** (1/n)

    pow_match = _RE_SIMPLE_POWER.match(s)
    if pow_match:
        base, exp = float(pow_match.group(1)), int(pow_match.group(2))
        return base ** exp

    if s == "pi":
        return math.pi

    return None


def _try_as_fraction(s: str) -> Optional[Fraction]:
    s = s.strip()
    try:
        return Fraction(int(s))
    except ValueError:
        pass

    frac_match = _RE_FRACTION.match(s)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            return Fraction(num, den)

    frac_match = _RE_PAREN_FRACTION.match(s)
    if frac_match:
        sign = -1 if frac_match.group(1) == '-' else 1
        num, den = int(frac_match.group(2)), int(frac_match.group(3))
        if den != 0:
            return Fraction(sign * num, den)

    decimal_match = _RE_DECIMAL.match(s)
    if decimal_match:
        try:
            return Fraction(s).limit_denominator(10000)
        except (ValueError, ZeroDivisionError):
            pass

    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if not answer:
        return ""

    answer = _strip_latex(answer)
    answer = _RE_CURRENCY.sub("", answer)

    def _percent_to_decimal(match: re.Match) -> str:
        value = float(match.group(1))
        decimal = value / 100
        if decimal == int(decimal):
            return str(int(decimal))
        result = repr(decimal)
        if 'e' not in result and '.' in result:
            result = result.rstrip("0").rstrip(".")
        return result

    answer = _RE_PERCENT_SYMBOL.sub(_percent_to_decimal, answer)
    answer = _RE_PERCENT_WORD.sub(_percent_to_decimal, answer)
    answer = _RE_UNITS.sub("", answer)
    answer = _RE_COMMA_IN_NUMBER.sub(r"\1\2", answer)
    answer = " ".join(answer.split())
    answer = _RE_FRACTION_SPACING.sub(r"\1/\2", answer)
    answer = _RE_DECIMAL_INTEGER.sub(r"\1", answer)
    answer = answer.rstrip(".,")

    return answer.strip().lower()


def answers_equivalent(a: str, b: str, tolerance: float = 1e-6) -> bool:
    """Check if two answers are mathematically equivalent."""
    norm_a = normalize_answer(a)
    norm_b = normalize_answer(b)

    if norm_a == norm_b:
        return True

    frac_a = _try_as_fraction(norm_a)
    frac_b = _try_as_fraction(norm_b)

    if frac_a is not None and frac_b is not None:
        if frac_a == frac_b:
            return True

    num_a = _try_parse_numeric(norm_a)
    num_b = _try_parse_numeric(norm_b)

    if num_a is not None and num_b is not None:
        max_abs = max(abs(num_a), abs(num_b))
        if max_abs < tolerance:
            return True
        elif max_abs > 1:
            rel_diff = abs(num_a - num_b) / max_abs
            if rel_diff < tolerance:
                return True
        else:
            if abs(num_a - num_b) < tolerance:
                return True

    return False


def answers_equivalent_llm(a: str, b: str, model: str = "gpt-4o-mini") -> bool:
    """Check if two answers are semantically equivalent using an LLM judge.

    This is a fallback for when numeric comparison fails but answers
    might still be equivalent (e.g., "1/2" vs "0.5", "$10" vs "10 dollars").

    Args:
        a: First answer
        b: Second answer
        model: LiteLLM model name

    Returns:
        True if answers are semantically equivalent
    """
    # First try fast numeric comparison
    if answers_equivalent(a, b, tolerance=0.01):
        return True

    try:
        from litellm import completion

        prompt = f"""Are these two answers mathematically/semantically equivalent?

Answer A: {a}
Answer B: {b}

Consider:
- Different representations of same number (1/2 = 0.5 = 50%)
- Unit variations ($10 = 10 dollars)
- Rounding differences within 1%
- Equivalent expressions

Reply with ONLY "yes" or "no"."""

        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )

        answer = response.choices[0].message.content.strip().lower()
        return answer == "yes"

    except Exception as e:
        # Fallback to numeric comparison if LLM fails
        return answers_equivalent(a, b, tolerance=0.01)
