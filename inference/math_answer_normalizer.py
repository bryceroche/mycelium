#!/usr/bin/env python3
r"""
MATH Answer Normalizer

Handles all the various formats MATH answers come in:
- LaTeX fractions: \frac{3}{4}
- Square roots: \sqrt{2}, 3\sqrt{2}
- Sets: \{1, 2, 3\}
- Tuples: (2, -3)
- Pi: \pi, \frac{\pi}{6}
- Variable assignments: x = -2
- Dollar amounts: \$5.00
"""

import re
from typing import Any, Optional, Union
from fractions import Fraction


def normalize_math_answer(answer_str: str) -> Any:
    """
    Normalize MATH answer to comparable canonical form.

    Returns:
        - frozenset for sets
        - tuple for tuples/ordered pairs
        - Fraction for exact rationals
        - float for decimals/irrationals
        - str for expressions that can't be simplified
    """
    if answer_str is None:
        return None

    s = str(answer_str).strip()

    # Strip LaTeX wrappers
    s = re.sub(r'\\boxed\{(.+)\}', r'\1', s)
    s = re.sub(r'\\text\{(.+?)\}', r'\1', s)

    # Strip dollar signs (both LaTeX escaped and literal)
    # Handle \$ first (escaped dollar in LaTeX)
    s = re.sub(r'^\\\$', '', s)  # Leading \$
    s = re.sub(r'\\\$$', '', s)  # Trailing \$
    s = s.replace('$', '')       # Plain $ signs
    s = s.strip()

    # Handle "x = value" format
    if '=' in s and not s.startswith('='):
        parts = s.split('=')
        if len(parts) == 2:
            var_part = parts[0].strip()
            val_part = parts[1].strip()
            # If left side looks like a variable, extract right side
            if re.match(r'^[a-zA-Z]$', var_part):
                s = val_part

    # Handle sets: \{1, 2, 3\} or {1, 2, 3}
    # Match both \{...\} and {...}
    set_match = re.match(r'^\\?\{(.+?)\\?\}$', s)
    if set_match:
        inner = set_match.group(1)
        elements = [normalize_math_answer(e.strip()) for e in inner.split(',')]
        return frozenset(elements)

    # Handle tuples/ordered pairs: (2, -3)
    tuple_match = re.match(r'^\((.+)\)$', s)
    if tuple_match:
        inner = tuple_match.group(1)
        # Don't treat single values in parens as tuples
        if ',' in inner:
            elements = [normalize_math_answer(e.strip()) for e in inner.split(',')]
            return tuple(elements)

    # Handle pi
    if s == '\\pi' or s == 'pi':
        return ('pi', 1, 1)  # (pi, numerator, denominator)

    # Handle fractions with pi: \frac{\pi}{6}
    pi_frac_match = re.match(r'^\\frac\{\\pi\}\{(\d+)\}$', s)
    if pi_frac_match:
        denom = int(pi_frac_match.group(1))
        return ('pi', 1, denom)

    # Handle coefficient*pi: 2\pi
    coef_pi_match = re.match(r'^(\d+)\\pi$', s)
    if coef_pi_match:
        coef = int(coef_pi_match.group(1))
        return ('pi', coef, 1)

    # Handle compound sqrt: 3\sqrt{2}
    compound_sqrt_match = re.match(r'^(-?\d+)\\sqrt\{(\d+)\}$', s)
    if compound_sqrt_match:
        coef = int(compound_sqrt_match.group(1))
        radicand = int(compound_sqrt_match.group(2))
        return ('sqrt', coef, radicand)

    # Handle simple sqrt: \sqrt{2}
    sqrt_match = re.match(r'^\\sqrt\{(\d+)\}$', s)
    if sqrt_match:
        radicand = int(sqrt_match.group(1))
        return ('sqrt', 1, radicand)

    # Handle fractions: \frac{3}{4}
    frac_match = re.match(r'^\\frac\{(-?\d+)\}\{(\d+)\}$', s)
    if frac_match:
        num = int(frac_match.group(1))
        denom = int(frac_match.group(2))
        return Fraction(num, denom)

    # Handle negative fractions: -\frac{3}{4}
    neg_frac_match = re.match(r'^-\\frac\{(\d+)\}\{(\d+)\}$', s)
    if neg_frac_match:
        num = int(neg_frac_match.group(1))
        denom = int(neg_frac_match.group(2))
        return Fraction(-num, denom)

    # Handle mixed numbers: 2\frac{1}{3}
    mixed_match = re.match(r'^(-?\d+)\\frac\{(\d+)\}\{(\d+)\}$', s)
    if mixed_match:
        whole = int(mixed_match.group(1))
        num = int(mixed_match.group(2))
        denom = int(mixed_match.group(3))
        if whole < 0:
            return Fraction(whole * denom - num, denom)
        return Fraction(whole * denom + num, denom)

    # Handle simple fractions: 3/4
    simple_frac_match = re.match(r'^(-?\d+)/(\d+)$', s)
    if simple_frac_match:
        num = int(simple_frac_match.group(1))
        denom = int(simple_frac_match.group(2))
        return Fraction(num, denom)

    # Handle integers
    if re.match(r'^-?\d+$', s):
        return int(s)

    # Handle decimals
    if re.match(r'^-?\d+\.\d+$', s):
        return float(s)

    # Handle percentages: 75\% or 75%
    pct_match = re.match(r'^(\d+(?:\.\d+)?)\\?%$', s)
    if pct_match:
        return float(pct_match.group(1)) / 100

    # Return as string if we can't parse it
    return s


def answers_match(predicted: Any, gold: Any, tolerance: float = 1e-6) -> bool:
    """
    Compare two math answers for equivalence.

    Handles:
    - Exact matches
    - Set comparison (order-independent)
    - Tuple comparison
    - Numeric approximation
    - Fraction/float equivalence
    """
    # Normalize both
    pred_norm = normalize_math_answer(predicted)
    gold_norm = normalize_math_answer(gold)

    # Exact match
    if pred_norm == gold_norm:
        return True

    # Both are sets
    if isinstance(pred_norm, frozenset) and isinstance(gold_norm, frozenset):
        return pred_norm == gold_norm

    # Both are tuples
    if isinstance(pred_norm, tuple) and isinstance(gold_norm, tuple):
        if len(pred_norm) != len(gold_norm):
            return False
        # Check if they're special tuples (pi, sqrt) or regular tuples
        if (len(pred_norm) == 3 and isinstance(pred_norm[0], str) and
            len(gold_norm) == 3 and isinstance(gold_norm[0], str)):
            # Both are special tuples - compare directly
            return pred_norm == gold_norm
        # Regular tuples - compare elements
        return all(answers_match(p, g, tolerance) for p, g in zip(pred_norm, gold_norm))

    # Both are Fractions
    if isinstance(pred_norm, Fraction) and isinstance(gold_norm, Fraction):
        return pred_norm == gold_norm

    # Numeric comparison
    pred_num = _to_numeric(pred_norm)
    gold_num = _to_numeric(gold_norm)

    if pred_num is not None and gold_num is not None:
        if abs(gold_num) < tolerance:
            return abs(pred_num - gold_num) < tolerance
        return abs(pred_num - gold_num) / max(abs(gold_num), 1e-10) < tolerance

    # String comparison (case insensitive, whitespace normalized)
    if isinstance(pred_norm, str) and isinstance(gold_norm, str):
        return pred_norm.lower().replace(' ', '') == gold_norm.lower().replace(' ', '')

    return False


def _to_numeric(val: Any) -> Optional[float]:
    """Convert a normalized value to float if possible."""
    if val is None:
        return None

    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, Fraction):
        return float(val)

    # Handle sqrt tuples
    if isinstance(val, tuple) and len(val) == 3:
        if val[0] == 'sqrt':
            # (sqrt, coef, radicand) -> coef * sqrt(radicand)
            coef, radicand = val[1], val[2]
            return coef * (radicand ** 0.5)
        if val[0] == 'pi':
            # (pi, num, denom) -> num * pi / denom
            import math
            return val[1] * math.pi / val[2]

    # Try to parse string as number
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            pass

    return None


def test_normalizer():
    """Run tests on the normalizer."""
    test_cases = [
        # Basic integers
        ("5", "5", True),
        ("42", "42", True),
        ("-7", "-7", True),

        # Fractions
        ("\\frac{3}{4}", "0.75", True),
        ("\\frac{1}{2}", "\\frac{2}{4}", True),
        ("-\\frac{3}{4}", "-0.75", True),
        ("3/4", "\\frac{3}{4}", True),

        # Square roots
        ("\\sqrt{2}", "1.41421356237", True),
        ("3\\sqrt{2}", "4.24264068712", True),

        # Sets (order independent)
        ("\\{1, 2, 7\\}", "\\{7, 1, 2\\}", True),
        ("\\{1, 2, 3\\}", "\\{3, 2, 1\\}", True),
        ("{1, 2}", "{2, 1}", True),

        # Tuples (order dependent)
        ("(2, -3)", "(2, -3)", True),
        ("(2, -3)", "(-3, 2)", False),

        # Pi
        ("\\pi", "3.14159265359", True),
        ("\\frac{\\pi}{6}", "0.523598775598", True),

        # Variable assignments
        ("x = -2", "-2", True),
        ("y = 5", "5", True),

        # Dollar amounts
        ("\\$5.00", "5", True),
        ("$10", "10", True),

        # Boxed answers
        ("\\boxed{42}", "42", True),
        ("\\boxed{\\frac{1}{2}}", "0.5", True),

        # Percentages
        ("75\\%", "0.75", True),
        ("50%", "0.5", True),

        # Mixed numbers (if supported)
        # ("2\\frac{1}{3}", "\\frac{7}{3}", True),

        # Different but equivalent
        ("5", "5.0", True),
        ("\\frac{4}{2}", "2", True),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("MATH ANSWER NORMALIZER TESTS")
    print("=" * 60)

    for pred, gold, expected in test_cases:
        result = answers_match(pred, gold)
        status = "PASS" if result == expected else "FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"{status}: answers_match(\"{pred}\", \"{gold}\") = {result}, expected {expected}")
            print(f"       pred_norm = {normalize_math_answer(pred)}")
            print(f"       gold_norm = {normalize_math_answer(gold)}")

    print("-" * 60)
    print(f"Results: {passed}/{passed+failed} passed")

    if failed == 0:
        print("All tests passed!")

    return failed == 0


def test_real_math500():
    """Test against actual MATH500 answer formats."""
    # Sample of real MATH500 gold answers (from dataset)
    real_answers = [
        "2",
        "\\frac{1}{2}",
        "\\sqrt{3}",
        "\\{1, 2, 3\\}",
        "(0, 1)",
        "\\pi",
        "\\frac{\\pi}{4}",
        "12",
        "-3",
        "\\frac{7}{8}",
    ]

    print("\n" + "=" * 60)
    print("REAL MATH500 ANSWER NORMALIZATION")
    print("=" * 60)

    for ans in real_answers:
        norm = normalize_math_answer(ans)
        numeric = _to_numeric(norm)
        print(f"  {ans:20} -> {str(norm):20} (numeric: {numeric})")


if __name__ == "__main__":
    all_passed = test_normalizer()
    test_real_math500()

    if not all_passed:
        exit(1)
