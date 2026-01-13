"""Tests for Solver critical paths."""

import pytest
from mycelium.solver import Solver, SolverResult, StepResult
from mycelium.planner import Step, DAGPlan


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
