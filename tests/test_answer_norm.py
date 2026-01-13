"""Tests for answer normalization and equivalence checking."""

import pytest
from mycelium.answer_norm import (
    normalize_answer,
    answers_equivalent,
    _try_parse_numeric,
    _try_as_fraction,
    _strip_latex,
    _parse_latex_frac,
)


class TestNormalizeAnswer:
    """Tests for normalize_answer()."""

    def test_empty_input(self):
        assert normalize_answer("") == ""
        assert normalize_answer("   ") == ""

    def test_basic_normalization(self):
        assert normalize_answer("  42  ") == "42"
        assert normalize_answer("HELLO") == "hello"

    def test_latex_boxed(self):
        assert normalize_answer(r"\boxed{42}") == "42"
        assert normalize_answer(r"$\boxed{42}$") == "42"

    def test_latex_frac(self):
        assert normalize_answer(r"\frac{1}{2}") == "(1)/(2)"
        assert normalize_answer(r"$\frac{3}{4}$") == "(3)/(4)"

    def test_latex_sqrt(self):
        assert normalize_answer(r"\sqrt{4}") == "sqrt(4)"
        assert normalize_answer(r"\sqrt[3]{8}") == "(8)^(1/3)"

    def test_currency_removal(self):
        assert normalize_answer("$100") == "100"
        assert normalize_answer("€50") == "50"
        assert normalize_answer("£25") == "25"

    def test_percentage_conversion(self):
        assert normalize_answer("50%") == "0.5"
        assert normalize_answer("100%") == "1"
        assert normalize_answer("25 %") == "0.25"
        assert normalize_answer("50 percent") == "0.5"

    def test_unit_removal(self):
        assert normalize_answer("10 cm") == "10"
        assert normalize_answer("5 kg") == "5"
        assert normalize_answer("100 dollars") == "100"

    def test_comma_in_numbers(self):
        assert normalize_answer("1,000,000") == "1000000"
        assert normalize_answer("800,000") == "800000"

    def test_decimal_integer(self):
        assert normalize_answer("15.0") == "15"
        assert normalize_answer("42.00") == "42"
        assert normalize_answer("3.14") == "3.14"  # Not an integer

    def test_fraction_spacing(self):
        assert normalize_answer("1 / 2") == "1/2"
        assert normalize_answer("3  /  4") == "3/4"


class TestAnswersEquivalent:
    """Tests for answers_equivalent()."""

    def test_string_equality(self):
        assert answers_equivalent("42", "42")
        assert answers_equivalent("hello", "HELLO")

    def test_fraction_decimal_equivalence(self):
        assert answers_equivalent("1/2", "0.5")
        assert answers_equivalent("0.5", "1/2")
        assert answers_equivalent("3/4", "0.75")

    def test_latex_equivalence(self):
        assert answers_equivalent(r"\frac{1}{2}", "0.5")
        assert answers_equivalent(r"$\frac{1}{4}$", "0.25")
        assert answers_equivalent(r"\boxed{42}", "42")

    def test_fraction_simplification(self):
        assert answers_equivalent("1/2", "2/4")
        assert answers_equivalent("3/6", "1/2")

    def test_decimal_equivalence(self):
        assert answers_equivalent("2", "2.0")
        assert answers_equivalent("2.0", "2.00")

    def test_sqrt_equivalence(self):
        assert answers_equivalent("sqrt(4)", "2")
        assert answers_equivalent(r"\sqrt{9}", "3")

    def test_percentage_equivalence(self):
        assert answers_equivalent("50%", "0.5")
        assert answers_equivalent("100%", "1")

    def test_non_equivalent(self):
        assert not answers_equivalent("1", "2")
        assert not answers_equivalent("1/2", "1/3")
        assert not answers_equivalent("hello", "world")

    def test_near_zero(self):
        assert answers_equivalent("0", "0.0")
        assert answers_equivalent("0.0000001", "0", tolerance=1e-6)

    def test_relative_tolerance(self):
        # Large numbers use relative tolerance
        assert answers_equivalent("1000000", "1000001", tolerance=1e-5)
        assert not answers_equivalent("1000000", "1001000", tolerance=1e-5)


class TestTryParseNumeric:
    """Tests for _try_parse_numeric()."""

    def test_integer(self):
        assert _try_parse_numeric("42") == 42.0
        assert _try_parse_numeric("-10") == -10.0

    def test_float(self):
        assert _try_parse_numeric("3.14") == pytest.approx(3.14)
        assert _try_parse_numeric("-2.5") == pytest.approx(-2.5)

    def test_fraction(self):
        assert _try_parse_numeric("1/2") == pytest.approx(0.5)
        assert _try_parse_numeric("3/4") == pytest.approx(0.75)
        assert _try_parse_numeric("-1/4") == pytest.approx(-0.25)

    def test_parenthesized_fraction(self):
        assert _try_parse_numeric("(1)/(2)") == pytest.approx(0.5)
        assert _try_parse_numeric("-(3)/(4)") == pytest.approx(-0.75)

    def test_sqrt(self):
        assert _try_parse_numeric("sqrt(4)") == pytest.approx(2.0)
        assert _try_parse_numeric("sqrt(2)") == pytest.approx(1.4142135623730951)

    def test_nth_root(self):
        assert _try_parse_numeric("8^(1/3)") == pytest.approx(2.0)
        assert _try_parse_numeric("(27)^(1/3)") == pytest.approx(3.0)

    def test_simple_power(self):
        assert _try_parse_numeric("2^3") == pytest.approx(8.0)
        assert _try_parse_numeric("3^2") == pytest.approx(9.0)

    def test_pi(self):
        import math
        assert _try_parse_numeric("pi") == pytest.approx(math.pi)

    def test_invalid(self):
        assert _try_parse_numeric("hello") is None
        assert _try_parse_numeric("") is None


class TestTryAsFraction:
    """Tests for _try_as_fraction()."""

    def test_integer(self):
        from fractions import Fraction
        assert _try_as_fraction("42") == Fraction(42)
        assert _try_as_fraction("-10") == Fraction(-10)

    def test_fraction(self):
        from fractions import Fraction
        assert _try_as_fraction("1/2") == Fraction(1, 2)
        assert _try_as_fraction("3/4") == Fraction(3, 4)
        assert _try_as_fraction("-1/4") == Fraction(-1, 4)

    def test_parenthesized_fraction(self):
        from fractions import Fraction
        assert _try_as_fraction("(1)/(2)") == Fraction(1, 2)
        assert _try_as_fraction("-(3)/(4)") == Fraction(-3, 4)

    def test_decimal(self):
        from fractions import Fraction
        result = _try_as_fraction("0.5")
        assert result == Fraction(1, 2)

    def test_invalid(self):
        assert _try_as_fraction("hello") is None
        assert _try_as_fraction("") is None


class TestStripLatex:
    """Tests for _strip_latex()."""

    def test_boxed(self):
        assert _strip_latex(r"\boxed{42}") == "42"
        assert _strip_latex(r"\boxed{\frac{1}{2}}") == "(1)/(2)"

    def test_dollar_signs(self):
        assert _strip_latex("$42$") == "42"
        assert _strip_latex("$$42$$") == "42"

    def test_frac(self):
        assert _strip_latex(r"\frac{1}{2}") == "(1)/(2)"
        assert _strip_latex(r"\tfrac{1}{2}") == "(1)/(2)"
        assert _strip_latex(r"\dfrac{1}{2}") == "(1)/(2)"

    def test_sqrt(self):
        assert _strip_latex(r"\sqrt{4}") == "sqrt(4)"
        assert _strip_latex(r"\sqrt[3]{8}") == "(8)^(1/3)"

    def test_operators(self):
        assert _strip_latex(r"2 \cdot 3") == "2 * 3"
        assert _strip_latex(r"2 \times 3") == "2 * 3"
        assert _strip_latex(r"6 \div 2") == "6 / 2"

    def test_symbols(self):
        assert _strip_latex(r"\pi") == "pi"
        assert _strip_latex(r"\infty") == "inf"


class TestParseLatexFrac:
    """Tests for _parse_latex_frac()."""

    def test_simple_frac(self):
        assert _parse_latex_frac(r"\frac{1}{2}") == "(1)/(2)"
        assert _parse_latex_frac(r"\frac{a}{b}") == "(a)/(b)"

    def test_nested_frac(self):
        # \frac{\frac{1}{2}}{3} should become ((1)/(2))/(3)
        result = _parse_latex_frac(r"\frac{\frac{1}{2}}{3}")
        assert "1" in result and "2" in result and "3" in result

    def test_tfrac_dfrac(self):
        assert _parse_latex_frac(r"\tfrac{1}{2}") == "(1)/(2)"
        assert _parse_latex_frac(r"\dfrac{1}{2}") == "(1)/(2)"
