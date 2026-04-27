"""
Pattern Classifier — Simple keyword-based pattern type classifier.

Auto-classifies a list of SymPy steps into a pattern type.
Simple keyword heuristic — not ML, just string matching.

Usage:
    from src.pattern_classifier import classify_pattern

    ptype = classify_pattern(["v1 = 48 / 2", "answer = v1"])
    # -> "half_of"
"""

import re


def _has_division_by(combined: str, divisor: int) -> bool:
    """Check if combined string has division by exactly this divisor (word boundary)."""
    # Match "/ N" where N is exactly the divisor (not "/ 25" matching "/ 2")
    pattern = rf'/ {divisor}(?![0-9])'
    return bool(re.search(pattern, combined))


def _has_multiply_by(combined: str, multiplier: int) -> bool:
    """Check if combined string has multiplication by exactly this multiplier."""
    pattern = rf'\* {multiplier}(?![0-9])'
    return bool(re.search(pattern, combined))


def classify_pattern(sympy_steps: list[str]) -> str:
    """
    Classify a list of SymPy step strings into a pattern type.

    Args:
        sympy_steps: List of SymPy step strings, e.g., ["v1 = 48 / 2", "answer = v1"]

    Returns:
        Pattern type string, one of:
        - "half_of"
        - "double_of"
        - "third_of"
        - "triple_of"
        - "percent_of"
        - "rate_times_time"
        - "compare"
        - "sum_of_parts"
        - "multiply_then_add"
        - "difference"
        - "multiply"
        - "divide"
        - "other"
    """
    combined = " ".join(sympy_steps).lower()

    # Half of: divide by 2 or multiply by 0.5
    if _has_division_by(combined, 2) or "* 0.5" in combined:
        return "half_of"

    # Double of: multiply by 2
    elif _has_multiply_by(combined, 2):
        return "double_of"

    # Third of: divide by 3
    elif _has_division_by(combined, 3):
        return "third_of"

    # Triple of: multiply by 3
    elif _has_multiply_by(combined, 3):
        return "triple_of"

    # Percent of: divide by 100, multiply by 0.01, or contains "percent"
    elif _has_division_by(combined, 100) or "percent" in combined or "* 0.01" in combined:
        return "percent_of"

    # Rate times time: contains time-related words
    elif "hour" in combined or "minute" in combined or "per" in combined:
        return "rate_times_time"

    # Compare: contains comparison operators or phrases
    elif ">" in combined or "<" in combined or "more than" in combined or "less than" in combined:
        return "compare"

    # Sum of parts: 2+ additions
    elif combined.count("+") >= 2:
        return "sum_of_parts"

    # Multiply then add: has both * and +
    elif "*" in combined and "+" in combined:
        return "multiply_then_add"

    # Difference: subtraction
    elif "-" in combined:
        return "difference"

    # Multiply: multiplication only
    elif "*" in combined:
        return "multiply"

    # Divide: division only
    elif "/" in combined:
        return "divide"

    # Other: no recognized pattern
    else:
        return "other"


# =============================================================================
# Tests
# =============================================================================

def _run_tests():
    """Comprehensive tests for classify_pattern."""

    print("Running pattern_classifier tests...\n")

    tests_passed = 0
    tests_failed = 0

    def test(name, steps, expected):
        nonlocal tests_passed, tests_failed
        result = classify_pattern(steps)
        if result == expected:
            print(f"  PASS: {name}")
            tests_passed += 1
        else:
            print(f"  FAIL: {name}")
            print(f"        Input: {steps}")
            print(f"        Expected: {expected}")
            print(f"        Got: {result}")
            tests_failed += 1

    # --- half_of ---
    print("Testing half_of:")
    test("divide by 2", ["v1 = 48 / 2", "answer = v1"], "half_of")
    test("multiply by 0.5", ["v1 = 100 * 0.5", "answer = v1"], "half_of")
    test("half in expression", ["v1 = (x + y) / 2"], "half_of")

    # --- double_of ---
    print("\nTesting double_of:")
    test("multiply by 2", ["v1 = 25 * 2", "answer = v1"], "double_of")
    test("double expression", ["total = count * 2"], "double_of")

    # --- third_of ---
    print("\nTesting third_of:")
    test("divide by 3", ["v1 = 90 / 3", "answer = v1"], "third_of")
    test("third in chain", ["v1 = x / 3", "v2 = v1 + 5"], "third_of")

    # --- triple_of ---
    print("\nTesting triple_of:")
    test("multiply by 3", ["v1 = 10 * 3", "answer = v1"], "triple_of")
    test("triple expression", ["total = base * 3"], "triple_of")

    # --- percent_of ---
    print("\nTesting percent_of:")
    test("divide by 100", ["v1 = rate / 100", "v2 = amount * v1"], "percent_of")
    test("multiply by 0.01", ["v1 = 50 * 0.01", "answer = v1"], "percent_of")
    test("percent keyword", ["v1 = percent_value * base"], "percent_of")

    # --- rate_times_time ---
    print("\nTesting rate_times_time:")
    test("hour keyword", ["v1 = speed * hour", "answer = v1"], "rate_times_time")
    test("minute keyword", ["v1 = rate_per_minute * 60"], "rate_times_time")
    test("per keyword", ["v1 = cost_per_item * count"], "rate_times_time")

    # --- compare ---
    print("\nTesting compare:")
    test("greater than operator", ["v1 = a > b"], "compare")
    test("less than operator", ["result = x < y"], "compare")
    test("more than phrase", ["v1 = x more than y"], "compare")
    test("less than phrase", ["v1 = a less than b"], "compare")

    # --- sum_of_parts ---
    print("\nTesting sum_of_parts:")
    test("three additions", ["v1 = a + b + c", "answer = v1"], "sum_of_parts")
    test("multiple additions", ["total = 10 + 20 + 30 + 40"], "sum_of_parts")
    test("two additions", ["v1 = x + y", "v2 = v1 + z"], "sum_of_parts")

    # --- multiply_then_add ---
    print("\nTesting multiply_then_add:")
    test("multiply and add", ["v1 = 10 * 5", "v2 = v1 + 3", "answer = v2"], "multiply_then_add")
    test("combined expression", ["total = base * rate + bonus"], "multiply_then_add")

    # --- difference ---
    print("\nTesting difference:")
    test("simple subtraction", ["v1 = 100 - 30", "answer = v1"], "difference")
    test("subtraction only", ["diff = a - b"], "difference")

    # --- multiply ---
    print("\nTesting multiply:")
    test("simple multiplication", ["v1 = 10 * 5", "answer = v1"], "multiply")
    test("multiplication only", ["product = x * y"], "multiply")

    # --- divide ---
    print("\nTesting divide:")
    test("simple division", ["v1 = 100 / 4", "answer = v1"], "divide")
    test("division chain", ["v1 = a / b", "v2 = v1 / c"], "divide")

    # --- other ---
    print("\nTesting other:")
    test("no operators", ["answer = 42"], "other")
    test("empty list", [], "other")
    test("assignment only", ["x = value"], "other")

    # --- Edge cases ---
    print("\nTesting edge cases:")
    test("case insensitive", ["V1 = X / 2", "ANSWER = V1"], "half_of")
    test("whitespace handling", ["  v1 = 10 * 2  "], "double_of")

    # --- Word boundary tests (bug fix) ---
    print("\nTesting word boundaries:")
    test("/ 25 is not half_of", ["v1 = 375 / 25", "answer = v1"], "divide")
    test("/ 20 is not half_of", ["v1 = 100 / 20", "answer = v1"], "divide")
    test("/ 30 is not third_of", ["v1 = 90 / 30", "answer = v1"], "divide")
    test("* 20 is not double_of", ["v1 = 5 * 20", "answer = v1"], "multiply")
    test("* 30 is not triple_of", ["v1 = 3 * 30", "answer = v1"], "multiply")

    # --- Priority tests (order matters in the if-elif chain) ---
    print("\nTesting priority (first match wins):")
    # half_of beats multiply (because / 2 is checked before *)
    test("/ 2 with * present", ["v1 = x * y / 2"], "half_of")
    # double_of beats multiply_then_add
    test("* 2 with + present", ["v1 = x * 2 + y"], "double_of")
    # percent_of beats divide
    test("/ 100 beats generic divide", ["v1 = x / 100"], "percent_of")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print(f"Total: {tests_passed + tests_failed}")

    if tests_failed == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n{tests_failed} test(s) failed.")
        exit(1)


if __name__ == "__main__":
    _run_tests()
