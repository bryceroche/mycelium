"""Tests for zero-LLM routing feature."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from mycelium.solver import Solver, SolverResult


# Helper to create 768-dimensional embeddings
EMBEDDING_DIM = 768


def make_embedding(seed: int) -> np.ndarray:
    """Create a deterministic 768-dim embedding from a seed."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(EMBEDDING_DIM)
    return emb / np.linalg.norm(emb)


class TestValueExtraction:
    """Tests for _extract_values_from_problem helper."""

    def test_extracts_integers(self):
        """Should extract integer values from problem text."""
        solver = Solver.__new__(Solver)  # Create without __init__

        values = solver._extract_values_from_problem(
            "Janet has 16 eggs. She eats 3 eggs for breakfast."
        )

        assert values["value_1"] == 16
        assert values["value_2"] == 3

    def test_extracts_decimals(self):
        """Should extract decimal values."""
        solver = Solver.__new__(Solver)

        values = solver._extract_values_from_problem(
            "The price is 12.50 dollars with 8.5% tax."
        )

        assert values["value_1"] == 12.50
        assert values["value_2"] == 8.5

    def test_extracts_negative_numbers(self):
        """Should extract negative numbers."""
        solver = Solver.__new__(Solver)

        values = solver._extract_values_from_problem(
            "Temperature dropped from 10 to -5 degrees."
        )

        assert values["value_1"] == 10
        assert values["value_2"] == -5

    def test_returns_empty_for_no_numbers(self):
        """Should return empty dict when no numbers found."""
        solver = Solver.__new__(Solver)

        values = solver._extract_values_from_problem(
            "What is the meaning of life?"
        )

        assert values == {}

    def test_adds_step_aliases(self):
        """Should add step_N aliases for compatibility."""
        solver = Solver.__new__(Solver)

        values = solver._extract_values_from_problem("5 plus 3")

        assert values["step_1"] == 5
        assert values["step_2"] == 3


class TestZeroLLMSolve:
    """Tests for _try_zero_llm_solve method."""

    def test_returns_none_when_disabled(self):
        """Should return None when ZERO_LLM_ROUTING_ENABLED is False."""
        solver = Solver.__new__(Solver)
        embedding = make_embedding(42)

        with patch('mycelium.solver.ZERO_LLM_ROUTING_ENABLED', False):
            result = solver._try_zero_llm_solve("5 + 3", embedding)

        assert result is None

    def test_returns_none_when_no_match(self):
        """Should return None when no signature matches."""
        solver = Solver.__new__(Solver)
        solver.step_db = Mock()
        solver.step_db.route_through_hierarchy.return_value = (None, [])

        embedding = make_embedding(42)

        with patch('mycelium.solver.ZERO_LLM_ROUTING_ENABLED', True):
            result = solver._try_zero_llm_solve("5 + 3", embedding)

        assert result is None

    def test_returns_none_for_low_uses(self):
        """Should return None when signature has too few uses."""
        solver = Solver.__new__(Solver)

        mock_sig = Mock()
        mock_sig.uses = 2  # Below ZERO_LLM_MIN_USES default of 5
        mock_sig.step_type = "test_sig"

        solver.step_db = Mock()
        solver.step_db.route_through_hierarchy.return_value = (mock_sig, [mock_sig])

        embedding = make_embedding(42)

        with patch('mycelium.solver.ZERO_LLM_ROUTING_ENABLED', True), \
             patch('mycelium.solver.ZERO_LLM_MIN_USES', 5):
            result = solver._try_zero_llm_solve("5 + 3", embedding)

        assert result is None

    def test_returns_none_for_low_success_rate(self):
        """Should return None when signature has low success rate."""
        solver = Solver.__new__(Solver)

        mock_sig = Mock()
        mock_sig.uses = 10
        mock_sig.successes = 5  # 50% success rate
        mock_sig.step_type = "test_sig"

        solver.step_db = Mock()
        solver.step_db.route_through_hierarchy.return_value = (mock_sig, [mock_sig])

        embedding = make_embedding(42)

        with patch('mycelium.solver.ZERO_LLM_ROUTING_ENABLED', True), \
             patch('mycelium.solver.ZERO_LLM_MIN_USES', 5), \
             patch('mycelium.solver.ZERO_LLM_MIN_SUCCESS_RATE', 0.70):
            result = solver._try_zero_llm_solve("5 + 3", embedding)

        assert result is None

    def test_returns_none_for_umbrella(self):
        """Should return None when matched signature is an umbrella."""
        solver = Solver.__new__(Solver)

        mock_sig = Mock()
        mock_sig.uses = 10
        mock_sig.successes = 9
        mock_sig.is_semantic_umbrella = True
        mock_sig.step_type = "test_sig"

        solver.step_db = Mock()
        solver.step_db.route_through_hierarchy.return_value = (mock_sig, [mock_sig])

        embedding = make_embedding(42)

        with patch('mycelium.solver.ZERO_LLM_ROUTING_ENABLED', True), \
             patch('mycelium.solver.ZERO_LLM_MIN_USES', 5), \
             patch('mycelium.solver.ZERO_LLM_MIN_SUCCESS_RATE', 0.70):
            result = solver._try_zero_llm_solve("5 + 3", embedding)

        assert result is None

    def test_returns_none_for_no_dsl(self):
        """Should return None when signature has no DSL script."""
        solver = Solver.__new__(Solver)

        mock_sig = Mock()
        mock_sig.uses = 10
        mock_sig.successes = 9
        mock_sig.is_semantic_umbrella = False
        mock_sig.dsl_script = None
        mock_sig.step_type = "test_sig"

        solver.step_db = Mock()
        solver.step_db.route_through_hierarchy.return_value = (mock_sig, [mock_sig])

        embedding = make_embedding(42)

        with patch('mycelium.solver.ZERO_LLM_ROUTING_ENABLED', True), \
             patch('mycelium.solver.ZERO_LLM_MIN_USES', 5), \
             patch('mycelium.solver.ZERO_LLM_MIN_SUCCESS_RATE', 0.70), \
             patch('mycelium.solver.ZERO_LLM_REQUIRE_DSL', True):
            result = solver._try_zero_llm_solve("5 + 3", embedding)

        assert result is None

    def test_successful_zero_llm_solve(self):
        """Should return SolverResult when all conditions met and DSL executes."""
        solver = Solver.__new__(Solver)

        mock_sig = Mock()
        mock_sig.id = 42
        mock_sig.uses = 10
        mock_sig.successes = 9
        mock_sig.is_semantic_umbrella = False
        mock_sig.dsl_script = "value_1 + value_2"
        mock_sig.dsl_type = "math"
        mock_sig.step_type = "addition"

        solver.step_db = Mock()
        solver.step_db.route_through_hierarchy.return_value = (mock_sig, [mock_sig])
        solver.step_db.record_usage = Mock()

        embedding = make_embedding(42)

        with patch('mycelium.solver.ZERO_LLM_ROUTING_ENABLED', True), \
             patch('mycelium.solver.ZERO_LLM_MIN_USES', 5), \
             patch('mycelium.solver.ZERO_LLM_MIN_SUCCESS_RATE', 0.70), \
             patch('mycelium.solver.ZERO_LLM_REQUIRE_DSL', True):
            result = solver._try_zero_llm_solve("What is 5 + 3?", embedding)

        assert result is not None
        assert result.success is True
        assert float(result.answer) == 8.0  # DSL may return 8 or 8.0
        assert result.steps[0].was_injected is True
        assert result.matched_and_reused == 1

        # Verify usage was recorded
        solver.step_db.record_usage.assert_called_once()
