"""Tests for Solver critical paths."""

import pytest
from mycelium.solver import Solver, SolverResult, StepResult
from mycelium.plan_models import Step, DAGPlan


class TestExtractJsonResult:
    """Tests for Solver._extract_json_result()."""

    @pytest.fixture
    def solver(self):
        """Create solver instance without full init."""
        s = Solver.__new__(Solver)
        return s

    def test_empty_response(self, solver):
        assert solver._extract_json_result("") == ""
        assert solver._extract_json_result(None) == ""

    def test_simple_result(self, solver):
        assert solver._extract_json_result('{"result": 42}') == "42"
        assert solver._extract_json_result('{"result": "hello"}') == "hello"

    def test_simple_answer(self, solver):
        assert solver._extract_json_result('{"answer": 42}') == "42"
        assert solver._extract_json_result('{"answer": "world"}') == "world"

    def test_nested_result(self, solver):
        # Critical bug fix: nested objects should parse correctly
        response = '{"result": {"value": 42, "unit": "m"}}'
        result = solver._extract_json_result(response)
        assert "value" in result
        assert "42" in result

    def test_deeply_nested(self, solver):
        response = '{"result": {"a": {"b": {"c": 5}}}}'
        result = solver._extract_json_result(response)
        assert "5" in result

    def test_json_in_text(self, solver):
        response = 'Here is the answer: {"result": 123} done'
        assert solver._extract_json_result(response) == "123"

    def test_float_result(self, solver):
        assert solver._extract_json_result('{"result": 3.14}') == "3.14"
        # Integer floats should be formatted as integers
        assert solver._extract_json_result('{"result": 42.0}') == "42"

    def test_result_takes_priority(self, solver):
        # When both "result" and "answer" exist, "result" should be used
        response = '{"result": 1, "answer": 2}'
        assert solver._extract_json_result(response) == "1"

    def test_malformed_json_fallback(self, solver):
        # Should fall back to regex extraction for malformed JSON
        response = "The answer is 42"
        result = solver._extract_json_result(response)
        assert result == "42"


class TestExtractResult:
    """Tests for Solver._extract_result() regex extraction."""

    @pytest.fixture
    def solver(self):
        s = Solver.__new__(Solver)
        return s

    def test_empty(self, solver):
        assert solver._extract_result("") == ""
        assert solver._extract_result(None) == ""

    def test_boxed_answer(self, solver):
        assert solver._extract_result(r"\boxed{42}") == "42"
        assert solver._extract_result(r"The answer is \boxed{100}") == "100"

    def test_equals_pattern(self, solver):
        assert solver._extract_result("x = 5") == "5"
        assert solver._extract_result("result = 42") == "42"

    def test_last_number(self, solver):
        assert solver._extract_result("The calculation gives us 42") == "42"
        assert solver._extract_result("1 + 2 + 3 = 6") == "6"

    def test_negative_numbers(self, solver):
        assert solver._extract_result("The answer is -5") == "-5"


class TestGetExecutionOrder:
    """Tests for Solver._get_execution_order()."""

    @pytest.fixture
    def solver(self):
        s = Solver.__new__(Solver)
        return s

    def test_empty_plan(self, solver):
        plan = DAGPlan(steps=[], problem="test")
        order = solver._get_execution_order(plan)
        assert order == []

    def test_single_step(self, solver):
        plan = DAGPlan(
            steps=[Step(id="s1", task="Do something")],
            problem="test"
        )
        order = solver._get_execution_order(plan)
        assert len(order) == 1
        assert order[0].id == "s1"

    def test_linear_chain(self, solver):
        plan = DAGPlan(
            steps=[
                Step(id="s1", task="First"),
                Step(id="s2", task="Second", depends_on=["s1"]),
                Step(id="s3", task="Third", depends_on=["s2"]),
            ],
            problem="test"
        )
        order = solver._get_execution_order(plan)
        ids = [s.id for s in order]
        assert ids == ["s1", "s2", "s3"]

    def test_parallel_steps(self, solver):
        plan = DAGPlan(
            steps=[
                Step(id="s1", task="First"),
                Step(id="s2a", task="Parallel A", depends_on=["s1"]),
                Step(id="s2b", task="Parallel B", depends_on=["s1"]),
                Step(id="s3", task="Join", depends_on=["s2a", "s2b"]),
            ],
            problem="test"
        )
        order = solver._get_execution_order(plan)
        ids = [s.id for s in order]

        # s1 must come first
        assert ids[0] == "s1"
        # s2a and s2b must come before s3
        assert ids.index("s2a") < ids.index("s3")
        assert ids.index("s2b") < ids.index("s3")
        # s3 must be last
        assert ids[-1] == "s3"

    def test_diamond_dependency(self, solver):
        # Classic diamond: A -> B, A -> C, B -> D, C -> D
        plan = DAGPlan(
            steps=[
                Step(id="A", task="A"),
                Step(id="B", task="B", depends_on=["A"]),
                Step(id="C", task="C", depends_on=["A"]),
                Step(id="D", task="D", depends_on=["B", "C"]),
            ],
            problem="test"
        )
        order = solver._get_execution_order(plan)
        ids = [s.id for s in order]

        assert ids[0] == "A"
        assert ids[-1] == "D"
        assert ids.index("B") < ids.index("D")
        assert ids.index("C") < ids.index("D")

    def test_handles_cycle_gracefully(self, solver):
        # Cycles should not cause infinite loop
        plan = DAGPlan(
            steps=[
                Step(id="A", task="A", depends_on=["B"]),
                Step(id="B", task="B", depends_on=["A"]),
            ],
            problem="test"
        )
        # Should return some order without hanging
        order = solver._get_execution_order(plan)
        assert len(order) == 2


class TestSolverResult:
    """Tests for SolverResult dataclass."""

    def test_success_result(self):
        result = SolverResult(
            problem="What is 2+2?",
            answer="4",
            success=True,
            total_steps=1,
        )
        assert result.success
        assert result.answer == "4"

    def test_failure_result(self):
        result = SolverResult(
            problem="Impossible problem",
            answer="",
            success=False,
            error="Could not solve",
        )
        assert not result.success
        assert result.error == "Could not solve"


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result(self):
        result = StepResult(
            step_id="step_1",
            task="Calculate something",
            result="42",
            success=True,
            signature_id=1,
            was_injected=True,
        )
        assert result.success
        assert result.was_injected
        assert result.result == "42"


# =============================================================================
# ERROR PATH TESTS
# =============================================================================

from unittest.mock import patch, MagicMock


class TestGetSignatureCount:
    """Tests for get_signature_count() error handling."""

    def test_db_error_returns_cached_value(self):
        """DB errors should return cached value without crashing."""
        from mycelium.solver import get_signature_count, _signature_count_cache

        # Reset cache to known state
        _signature_count_cache["count"] = 100
        _signature_count_cache["last_check"] = 0  # Force recheck

        # Patch get_step_db per "New Favorite Pattern" - uses data layer
        with patch("mycelium.step_signatures.db.get_step_db") as mock_get_step_db:
            mock_get_step_db.side_effect = Exception("DB connection failed")

            # Should return cached value without raising
            result = get_signature_count()
            assert result == 100

    def test_query_error_returns_cached_value(self):
        """Query execution errors should return cached value."""
        from mycelium.solver import get_signature_count, _signature_count_cache

        _signature_count_cache["count"] = 50
        _signature_count_cache["last_check"] = 0

        mock_step_db = MagicMock()
        mock_step_db.get_signature_count.side_effect = Exception("Query failed")

        with patch("mycelium.step_signatures.db.get_step_db", return_value=mock_step_db):
            result = get_signature_count()
            assert result == 50


class TestAdaptiveMatchThreshold:
    """Tests for get_adaptive_match_threshold() cold-start behavior."""

    def test_cold_start_higher_threshold(self):
        """Cold start (few signatures) should use higher threshold."""
        from mycelium.solver import get_adaptive_match_threshold
        from mycelium.config import MIN_MATCH_THRESHOLD_COLD_START

        with patch("mycelium.solver.get_signature_count", return_value=0):
            threshold = get_adaptive_match_threshold()
            # Cold start threshold should be at or near cold start value
            assert threshold >= MIN_MATCH_THRESHOLD_COLD_START - 0.01

    def test_mature_db_lower_threshold(self):
        """Mature DB (many signatures) should use lower threshold."""
        from mycelium.solver import get_adaptive_match_threshold
        from mycelium.config import MIN_MATCH_THRESHOLD, MIN_MATCH_THRESHOLD_COLD_START

        with patch("mycelium.solver.get_signature_count", return_value=1000):
            threshold = get_adaptive_match_threshold()
            # Mature threshold should be lower than cold start
            assert threshold < MIN_MATCH_THRESHOLD_COLD_START
            assert threshold <= MIN_MATCH_THRESHOLD + 0.01


class TestExtractValuesFromProblem:
    """Tests for Solver._extract_values_from_problem()."""

    @pytest.fixture
    def solver(self):
        s = Solver.__new__(Solver)
        return s

    def test_no_numbers(self, solver):
        """Problem with no numbers returns empty dict."""
        result = solver._extract_values_from_problem("What is the sum?")
        assert result == {}

    def test_single_number(self, solver):
        """Single number is extracted correctly."""
        result = solver._extract_values_from_problem("Calculate 42 squared")
        assert "value_1" in result
        assert result["value_1"] == 42

    def test_multiple_numbers(self, solver):
        """Multiple numbers are extracted in order."""
        result = solver._extract_values_from_problem("Add 10 and 5")
        assert result["value_1"] == 10
        assert result["value_2"] == 5
        # Also has step_N aliases
        assert result["step_1"] == 10
        assert result["step_2"] == 5

    def test_decimal_numbers(self, solver):
        """Decimal numbers are extracted as floats."""
        result = solver._extract_values_from_problem("Calculate 3.14 times 2")
        assert result["value_1"] == 3.14
        assert result["value_2"] == 2

    def test_negative_numbers(self, solver):
        """Negative numbers are handled."""
        result = solver._extract_values_from_problem("Sum -5 and 10")
        assert result["value_1"] == -5
        assert result["value_2"] == 10


class TestJsonExtractionEdgeCases:
    """Additional edge cases for JSON extraction."""

    @pytest.fixture
    def solver(self):
        s = Solver.__new__(Solver)
        return s

    def test_multiple_json_objects(self, solver):
        """First valid JSON with result/answer should be used."""
        response = '{"foo": 1} {"result": 42} {"bar": 2}'
        result = solver._extract_json_result(response)
        assert result == "42"

    def test_unbalanced_braces(self, solver):
        """Unbalanced braces should fall back to regex."""
        response = '{"result": 42'  # Missing closing brace
        result = solver._extract_json_result(response)
        # Falls back to regex, finds 42
        assert result == "42"

    def test_list_result(self, solver):
        """List results should be stringified."""
        response = '{"result": [1, 2, 3]}'
        result = solver._extract_json_result(response)
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_boolean_result(self, solver):
        """Boolean results should be stringified."""
        response = '{"result": true}'
        result = solver._extract_json_result(response)
        assert result.lower() == "true"

    def test_null_result(self, solver):
        """Null result should return empty or null string."""
        response = '{"result": null}'
        result = solver._extract_json_result(response)
        # Should handle gracefully
        assert result in ("", "null", "None")


# =============================================================================
# OPERATION EMBEDDING TESTS (Graph-based routing)
# =============================================================================


class TestPrewarmOperationEmbeddings:
    """Tests for Solver._prewarm_operation_embeddings()."""

    @pytest.fixture
    def solver(self):
        """Create solver instance with minimal init for testing."""
        s = Solver.__new__(Solver)
        s._operation_embeddings = {}
        s.embedder = None  # Will be mocked in tests
        return s

    def test_empty_steps(self, solver):
        """Empty steps should clear dict and return."""
        solver._operation_embeddings = {"old": [1.0, 2.0]}
        solver._prewarm_operation_embeddings([])
        assert solver._operation_embeddings == {}

    def test_steps_without_operations(self, solver, monkeypatch):
        """Steps without operations should be skipped."""
        steps = [
            Step(id="s1", task="Do something"),  # No operation
            Step(id="s2", task="Do another"),    # No operation
        ]
        # Track if cached_embed_batch was called
        embed_called = []
        def mock_embed(texts, embedder):
            embed_called.append(texts)
            return {}
        monkeypatch.setattr('mycelium.solver.cached_embed_batch', mock_embed)

        solver._prewarm_operation_embeddings(steps)

        # Should not call embed if no operations
        assert len(embed_called) == 0
        assert solver._operation_embeddings == {}

    def test_steps_with_operations(self, solver, monkeypatch):
        """Steps with operations should be batch embedded."""
        import numpy as np

        steps = [
            Step(id="s1", task="Convert units", operation="divide two numbers"),
            Step(id="s2", task="Calculate total", operation="multiply two numbers"),
        ]

        # Mock cached_embed_batch to return fake embeddings
        embed_calls = []
        def mock_embed(texts, embedder):
            embed_calls.append(texts)
            return {
                "divide two numbers": np.array([1.0, 2.0, 3.0]),
                "multiply two numbers": np.array([4.0, 5.0, 6.0]),
            }
        monkeypatch.setattr('mycelium.solver.cached_embed_batch', mock_embed)

        solver._prewarm_operation_embeddings(steps)

        # Should call embed with operations
        assert len(embed_calls) == 1
        assert "divide two numbers" in embed_calls[0]
        assert "multiply two numbers" in embed_calls[0]

        # Should store embeddings keyed by step.id
        assert "s1" in solver._operation_embeddings
        assert "s2" in solver._operation_embeddings
        assert solver._operation_embeddings["s1"] == [1.0, 2.0, 3.0]
        assert solver._operation_embeddings["s2"] == [4.0, 5.0, 6.0]

    def test_mixed_steps(self, solver, monkeypatch):
        """Mix of steps with and without operations."""
        import numpy as np

        steps = [
            Step(id="s1", task="Do something"),  # No operation
            Step(id="s2", task="Calculate", operation="add two numbers"),
            Step(id="s3", task="Another"),  # No operation
        ]

        def mock_embed(texts, embedder):
            return {"add two numbers": np.array([1.0, 2.0])}
        monkeypatch.setattr('mycelium.solver.cached_embed_batch', mock_embed)

        solver._prewarm_operation_embeddings(steps)

        # Only s2 should have embedding
        assert "s1" not in solver._operation_embeddings
        assert "s2" in solver._operation_embeddings
        assert "s3" not in solver._operation_embeddings

    def test_clears_previous_embeddings(self, solver, monkeypatch):
        """Should clear embeddings from previous problem."""
        # Pre-populate with old embeddings
        solver._operation_embeddings = {
            "old_step": [9.0, 9.0, 9.0]
        }

        steps = [Step(id="new_step", task="New", operation="new op")]
        def mock_embed(texts, embedder):
            return {"new op": [1.0, 2.0]}
        monkeypatch.setattr('mycelium.solver.cached_embed_batch', mock_embed)

        solver._prewarm_operation_embeddings(steps)

        # Old embedding should be gone
        assert "old_step" not in solver._operation_embeddings
        assert "new_step" in solver._operation_embeddings
