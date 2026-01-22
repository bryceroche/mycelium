"""Tests for computation graph extraction and routing."""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock

from mycelium.step_signatures.graph_extractor import (
    extract_computation_graph,
    graphs_equivalent,
    graph_to_natural_language,
    embed_computation_graph_sync,
    clear_graph_embedding_cache,
)


class TestExtractComputationGraph:
    """Tests for DSL to computation graph extraction."""

    def test_simple_multiplication(self):
        graph = extract_computation_graph("a * b")
        assert graph == "MUL(param_0, param_1)"

    def test_simple_addition(self):
        graph = extract_computation_graph("x + y")
        assert graph == "ADD(param_0, param_1)"

    def test_simple_subtraction(self):
        graph = extract_computation_graph("a - b")
        assert graph == "SUB(param_0, param_1)"

    def test_simple_division(self):
        graph = extract_computation_graph("x / y")
        assert graph == "DIV(param_0, param_1)"

    def test_power(self):
        graph = extract_computation_graph("base ** exp")
        assert graph == "POW(param_0, param_1)"

    def test_modulo(self):
        graph = extract_computation_graph("a % b")
        assert graph == "MOD(param_0, param_1)"

    def test_nested_expression(self):
        graph = extract_computation_graph("(a + b) * c")
        assert graph == "MUL(ADD(param_0, param_1), param_2)"

    def test_complex_expression(self):
        graph = extract_computation_graph("(price * quantity) + tax")
        assert graph == "ADD(MUL(param_0, param_1), param_2)"

    def test_with_constant(self):
        graph = extract_computation_graph("a / 100")
        assert graph == "DIV(param_0, CONST(100))"

    def test_sqrt_function(self):
        graph = extract_computation_graph("sqrt(x)")
        assert graph == "SQRT(param_0)"

    def test_gcd_function(self):
        graph = extract_computation_graph("gcd(a, b)")
        assert graph == "GCD(param_0, param_1)"

    def test_abs_function(self):
        graph = extract_computation_graph("abs(x)")
        assert graph == "ABS(param_0)"

    def test_json_dsl_format(self):
        """Test extraction from JSON DSL format."""
        json_dsl = '{"type": "math", "script": "a + b"}'
        graph = extract_computation_graph(json_dsl)
        assert graph == "ADD(param_0, param_1)"

    def test_decompose_type_returns_none(self):
        """Decompose DSLs have no meaningful graph."""
        json_dsl = '{"type": "decompose", "script": "reason_step"}'
        graph = extract_computation_graph(json_dsl)
        assert graph is None

    def test_empty_input(self):
        assert extract_computation_graph("") is None
        assert extract_computation_graph(None) is None

    def test_invalid_syntax(self):
        graph = extract_computation_graph("a +* b")
        assert graph is None


class TestGraphsEquivalent:
    """Tests for graph equivalence checking."""

    def test_identical_graphs(self):
        assert graphs_equivalent("MUL(param_0, param_1)", "MUL(param_0, param_1)")

    def test_different_graphs(self):
        assert not graphs_equivalent("MUL(param_0, param_1)", "ADD(param_0, param_1)")

    def test_none_graphs(self):
        assert graphs_equivalent(None, None)
        assert not graphs_equivalent("MUL(param_0, param_1)", None)
        assert not graphs_equivalent(None, "MUL(param_0, param_1)")


class TestGraphToNaturalLanguage:
    """Tests for graph to natural language expansion."""

    def test_simple_mul(self):
        nl = graph_to_natural_language("MUL(param_0, param_1)")
        assert "multiply" in nl.lower()
        assert "first" in nl.lower()
        assert "second" in nl.lower()

    def test_simple_add(self):
        nl = graph_to_natural_language("ADD(param_0, param_1)")
        assert "add" in nl.lower()

    def test_simple_sub(self):
        nl = graph_to_natural_language("SUB(param_0, param_1)")
        assert "subtract" in nl.lower()

    def test_simple_div(self):
        nl = graph_to_natural_language("DIV(param_0, param_1)")
        assert "divide" in nl.lower()

    def test_power(self):
        nl = graph_to_natural_language("POW(param_0, param_1)")
        assert "power" in nl.lower()

    def test_sqrt(self):
        nl = graph_to_natural_language("SQRT(param_0)")
        assert "square root" in nl.lower()

    def test_gcd(self):
        nl = graph_to_natural_language("GCD(param_0, param_1)")
        assert "greatest common divisor" in nl.lower()

    def test_nested_expression(self):
        nl = graph_to_natural_language("ADD(MUL(param_0, param_1), param_2)")
        assert "add" in nl.lower()
        assert "multiply" in nl.lower()

    def test_constant(self):
        nl = graph_to_natural_language("CONST(100)")
        assert "constant" in nl.lower()

    def test_empty_input(self):
        assert graph_to_natural_language("") == ""
        assert graph_to_natural_language(None) == ""


class TestEmbedComputationGraphSync:
    """Tests for sync graph embedding."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_graph_embedding_cache()

    def test_embed_returns_list(self):
        """Test that embedding returns a list."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.array([0.1, 0.2, 0.3])

        result = embed_computation_graph_sync(mock_embedder, "MUL(param_0, param_1)")

        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]

    def test_caching(self):
        """Test that repeated calls use cache."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.array([0.1, 0.2, 0.3])

        # First call
        result1 = embed_computation_graph_sync(mock_embedder, "MUL(param_0, param_1)")
        # Second call - should use cache
        result2 = embed_computation_graph_sync(mock_embedder, "MUL(param_0, param_1)")

        assert result1 == result2
        # Embedder should only be called once
        assert mock_embedder.embed.call_count == 1

    def test_empty_graph(self):
        """Empty graph returns None."""
        mock_embedder = MagicMock()
        result = embed_computation_graph_sync(mock_embedder, "")
        assert result is None
        mock_embedder.embed.assert_not_called()

    def test_embedder_failure(self):
        """Embedder failure returns None."""
        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = Exception("API error")

        result = embed_computation_graph_sync(mock_embedder, "MUL(param_0, param_1)")
        assert result is None

    def test_expands_to_natural_language(self):
        """Verify that graph is expanded to NL before embedding."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.array([0.1])

        embed_computation_graph_sync(mock_embedder, "MUL(param_0, param_1)")

        # Check that embed was called with natural language, not raw graph
        call_arg = mock_embedder.embed.call_args[0][0]
        assert "multiply" in call_arg.lower()
        assert "MUL" not in call_arg


class TestEndToEndGraphFlow:
    """Integration tests for the full graph extraction and NL flow."""

    def test_dsl_to_nl_pipeline(self):
        """Test DSL -> Graph -> NL pipeline."""
        test_cases = [
            ("a * b", "multiply"),
            ("base ** exp", "power"),
            ("gcd(x, y)", "greatest common divisor"),
            ("(a + b) / 2", "divide"),
        ]

        for dsl, expected_word in test_cases:
            graph = extract_computation_graph(dsl)
            assert graph is not None, f"Failed to extract graph from: {dsl}"

            nl = graph_to_natural_language(graph)
            assert expected_word in nl.lower(), f"Expected '{expected_word}' in NL for {dsl}, got: {nl}"

    def test_semantic_param_independence(self):
        """Verify that different param names produce same graph."""
        graph1 = extract_computation_graph("rate * time")
        graph2 = extract_computation_graph("price * quantity")
        graph3 = extract_computation_graph("length * width")

        assert graph1 == graph2 == graph3 == "MUL(param_0, param_1)"
