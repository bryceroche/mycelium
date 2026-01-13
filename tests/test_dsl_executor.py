"""Tests for DSL executor critical paths."""

import pytest
from mycelium.step_signatures.dsl_executor import (
    try_execute_dsl_math,
    try_execute_dsl_sympy,
    _extract_numeric_value,
    _prepare_math_inputs,
    _is_valid_dsl_result,
    DSLSpec,
    DSLLayer,
)


class TestTryExecuteDslMath:
    """Tests for math DSL execution."""

    def test_simple_addition(self):
        result = try_execute_dsl_math("a + b", {"a": 2, "b": 3})
        assert result == 5.0

    def test_multiplication(self):
        result = try_execute_dsl_math("a * b", {"a": 4, "b": 5})
        assert result == 20.0

    def test_division(self):
        result = try_execute_dsl_math("a / b", {"a": 10, "b": 2})
        assert result == 5.0

    def test_power(self):
        result = try_execute_dsl_math("a ** b", {"a": 2, "b": 3})
        assert result == 8.0

    def test_complex_expression(self):
        result = try_execute_dsl_math("(a + b) * c", {"a": 1, "b": 2, "c": 3})
        assert result == 9.0

    def test_sqrt(self):
        result = try_execute_dsl_math("sqrt(a)", {"a": 16})
        assert result == 4.0

    def test_missing_param_returns_none(self):
        result = try_execute_dsl_math("a + b", {"a": 2})
        assert result is None

    def test_division_by_zero_returns_none(self):
        result = try_execute_dsl_math("a / b", {"a": 1, "b": 0})
        assert result is None

    def test_negative_numbers(self):
        result = try_execute_dsl_math("a - b", {"a": 5, "b": 10})
        assert result == -5.0

    def test_float_inputs(self):
        result = try_execute_dsl_math("a * b", {"a": 2.5, "b": 4.0})
        assert result == 10.0


class TestTryExecuteDslSympy:
    """Tests for sympy DSL execution."""

    def test_simplify(self):
        result = try_execute_dsl_sympy("simplify(expr)", {"expr": "x**2 - x**2"})
        assert str(result) == "0"

    def test_expand(self):
        result = try_execute_dsl_sympy("expand(expr)", {"expr": "(x+1)**2"})
        assert "x**2" in str(result)

    def test_factor(self):
        result = try_execute_dsl_sympy("factor(expr)", {"expr": "x**2 - 1"})
        assert "x - 1" in str(result) or "x + 1" in str(result)

    def test_solve(self):
        result = try_execute_dsl_sympy("solve(expr, x)", {"expr": "x**2 - 4"})
        # solve returns [-2, 2]
        result_str = str(result)
        assert "2" in result_str

    def test_numeric_operations(self):
        result = try_execute_dsl_sympy("a + b", {"a": 2, "b": 3})
        assert result == 5


class TestExtractNumericValue:
    """Tests for numeric value extraction."""

    def test_integer(self):
        assert _extract_numeric_value(42) == 42.0

    def test_float(self):
        assert _extract_numeric_value(3.14) == pytest.approx(3.14)

    def test_string_integer(self):
        assert _extract_numeric_value("42") == 42.0

    def test_string_float(self):
        assert _extract_numeric_value("3.14") == pytest.approx(3.14)

    def test_string_with_commas(self):
        assert _extract_numeric_value("1,000,000") == 1000000.0

    def test_negative(self):
        assert _extract_numeric_value("-5") == -5.0

    def test_invalid_string(self):
        assert _extract_numeric_value("hello") is None

    def test_none(self):
        assert _extract_numeric_value(None) is None


class TestPrepareMathInputs:
    """Tests for math input preparation."""

    def test_numeric_values(self):
        result = _prepare_math_inputs({"a": 1, "b": 2.5})
        assert result == {"a": 1.0, "b": 2.5}

    def test_string_numbers(self):
        result = _prepare_math_inputs({"a": "42", "b": "3.14"})
        assert result == {"a": 42.0, "b": pytest.approx(3.14)}

    def test_filters_non_numeric(self):
        result = _prepare_math_inputs({"a": 1, "b": "hello", "c": 3})
        assert "a" in result
        assert "c" in result
        assert "b" not in result


class TestIsValidDslResult:
    """Tests for DSL result validation."""

    def test_valid_integer(self):
        assert _is_valid_dsl_result(42) is True

    def test_valid_float(self):
        assert _is_valid_dsl_result(3.14) is True

    def test_boolean_false_invalid(self):
        # sympy sometimes returns False for failed operations
        assert _is_valid_dsl_result(False) is False

    def test_boolean_true_valid(self):
        # True can be valid for predicate checks
        assert _is_valid_dsl_result(True) is True

    def test_none_invalid(self):
        assert _is_valid_dsl_result(None) is False

    def test_astronomically_large_invalid(self):
        # Results > 1e10 are likely param mapping errors
        assert _is_valid_dsl_result(1e15) is False

    def test_reasonable_large_valid(self):
        assert _is_valid_dsl_result(1e9) is True

    def test_string_result_valid(self):
        assert _is_valid_dsl_result("x**2 + 1") is True


class TestDSLSpec:
    """Tests for DSLSpec dataclass."""

    def test_from_json_math(self):
        json_str = '{"layer": "math", "script": "a + b", "params": ["a", "b"]}'
        spec = DSLSpec.from_json(json_str)
        assert spec is not None
        assert spec.layer == DSLLayer.MATH
        assert spec.script == "a + b"
        assert spec.params == ["a", "b"]

    def test_from_json_sympy(self):
        json_str = '{"layer": "sympy", "script": "simplify(expr)", "params": ["expr"]}'
        spec = DSLSpec.from_json(json_str)
        assert spec is not None
        assert spec.layer == DSLLayer.SYMPY

    def test_from_json_invalid(self):
        spec = DSLSpec.from_json("not valid json")
        assert spec is None

    def test_from_json_missing_fields(self):
        spec = DSLSpec.from_json('{"layer": "math"}')
        # Should handle missing script gracefully
        assert spec is None or spec.script == ""
