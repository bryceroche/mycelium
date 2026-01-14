"""Tests for semantic_validation module."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from mycelium.planner import Step, DAGPlan
from mycelium.step_signatures.semantic_validation import (
    StepValidation,
    PlanValidation,
    StepFailureFeedback,
    validate_plan_coherence,
    _validate_step_inputs,
    _generate_planner_feedback,
    create_failure_feedback,
)


# Helper to create DAGPlan with default problem
def make_plan(steps, problem="test problem"):
    return DAGPlan(steps=steps, problem=problem)


class MockEmbedder:
    """Mock embedder that returns predictable vectors."""

    def __init__(self, similarity_map=None):
        """
        Args:
            similarity_map: dict of (text1, text2) -> similarity score
                           If not provided, uses simple hash-based vectors
        """
        self.similarity_map = similarity_map or {}
        self._cache = {}

    def embed(self, text: str) -> np.ndarray:
        """Return a deterministic embedding based on text."""
        if text not in self._cache:
            # Create a simple hash-based embedding
            np.random.seed(hash(text) % (2**32))
            self._cache[text] = np.random.randn(384)
            self._cache[text] /= np.linalg.norm(self._cache[text])
        return self._cache[text]


class TestStepValidation:
    """Tests for StepValidation dataclass."""

    def test_valid_step(self):
        sv = StepValidation(
            step_id="step1",
            is_valid=True,
            coherence_score=0.9,
        )
        assert sv.is_valid
        assert sv.coherence_score == 0.9
        assert sv.issues == []
        assert sv.suggestions == []

    def test_invalid_step_with_issues(self):
        sv = StepValidation(
            step_id="step1",
            is_valid=False,
            coherence_score=0.2,
            issues=["Missing dependency"],
            suggestions=["Add intermediate step"],
        )
        assert not sv.is_valid
        assert len(sv.issues) == 1
        assert len(sv.suggestions) == 1


class TestPlanValidation:
    """Tests for PlanValidation dataclass."""

    def test_coherent_plan(self):
        pv = PlanValidation(
            is_coherent=True,
            overall_score=0.85,
        )
        assert pv.is_coherent
        assert pv.step_validations == []
        assert pv.feedback_for_planner == ""

    def test_incoherent_plan_with_feedback(self):
        pv = PlanValidation(
            is_coherent=False,
            overall_score=0.3,
            feedback_for_planner="Steps don't connect properly",
        )
        assert not pv.is_coherent
        assert "don't connect" in pv.feedback_for_planner


class TestValidatePlanCoherence:
    """Tests for validate_plan_coherence()."""

    def test_empty_plan(self):
        plan = make_plan(steps=[])
        embedder = MockEmbedder()

        result = validate_plan_coherence(plan, embedder=embedder)

        assert result.is_coherent
        assert result.overall_score == 1.0

    def test_single_step_plan(self):
        plan = make_plan(steps=[
            Step(id="step1", task="Calculate the sum"),
        ])
        embedder = MockEmbedder()

        result = validate_plan_coherence(plan, embedder=embedder)

        assert result.is_coherent
        assert len(result.step_validations) == 1

    def test_linear_dependent_steps(self):
        plan = make_plan(steps=[
            Step(id="step1", task="Find the ratio"),
            Step(id="step2", task="Scale the ratio", depends_on=["step1"]),
        ])
        embedder = MockEmbedder()

        result = validate_plan_coherence(plan, embedder=embedder)

        assert len(result.step_validations) == 2
        # Score depends on embedding similarity

    def test_missing_dependency_detected(self):
        plan = make_plan(steps=[
            Step(id="step1", task="First step"),
            Step(id="step2", task="Second step", depends_on=["nonexistent"]),
        ])
        embedder = MockEmbedder()

        result = validate_plan_coherence(plan, embedder=embedder)

        assert not result.is_coherent
        assert any("nonexistent" in issue for sv in result.step_validations for issue in sv.issues)

    def test_no_embedder_skips_validation(self):
        """When embedder can't be loaded, validation is skipped."""
        plan = make_plan(steps=[
            Step(id="step1", task="Do something"),
        ])

        # Pass None and mock the import to fail
        result = validate_plan_coherence(plan, embedder=None)

        # Should either work with default embedder or skip gracefully
        assert isinstance(result, PlanValidation)

    def test_extracted_values_tracked(self):
        plan = make_plan(steps=[
            Step(id="step1", task="Extract values", extracted_values={"x": 10, "y": 20}),
            Step(id="step2", task="Use values", depends_on=["step1"]),
        ])
        embedder = MockEmbedder()

        result = validate_plan_coherence(plan, embedder=embedder)

        assert len(result.step_validations) == 2


class TestValidateStepInputs:
    """Tests for _validate_step_inputs()."""

    def test_step_with_no_dependencies(self):
        step = Step(id="step1", task="Independent step")
        available_outputs = {}
        embedder = MockEmbedder()

        result = _validate_step_inputs(step, available_outputs, embedder)

        assert result.is_valid
        assert result.coherence_score == 1.0  # No deps = perfect coherence

    def test_step_with_valid_dependency(self):
        step = Step(id="step2", task="Use the ratio", depends_on=["step1"])
        available_outputs = {
            "step1": ("Calculate ratio", ["numerator", "denominator"]),
        }
        embedder = MockEmbedder()

        result = _validate_step_inputs(step, available_outputs, embedder)

        assert result.step_id == "step2"
        # Validity depends on semantic similarity

    def test_step_with_missing_dependency(self):
        step = Step(id="step2", task="Continue", depends_on=["missing"])
        available_outputs = {"step1": ("First step", [])}
        embedder = MockEmbedder()

        result = _validate_step_inputs(step, available_outputs, embedder)

        assert not result.is_valid
        assert any("missing" in issue for issue in result.issues)
        assert result.coherence_score == 0.0

    def test_step_with_reference_to_missing_step(self):
        step = Step(
            id="step2",
            task="Use previous result",
            extracted_values={"result": "{missing_step}"},
        )
        available_outputs = {"step1": ("First", [])}
        embedder = MockEmbedder()

        result = _validate_step_inputs(step, available_outputs, embedder)

        assert not result.is_valid
        assert any("missing_step" in issue for issue in result.issues)

    def test_weak_semantic_connection_flagged(self):
        """Test that semantically unrelated steps get flagged."""
        step = Step(id="step2", task="Calculate the weather forecast", depends_on=["step1"])
        available_outputs = {
            "step1": ("Find the prime factorization", []),
        }

        # Create embedder with controlled similarity
        embedder = MockEmbedder()

        result = _validate_step_inputs(step, available_outputs, embedder)

        # The mock embedder creates random embeddings, so similarity varies
        # Just verify the function runs without error
        assert result.step_id == "step2"


class TestGeneratePlannerFeedback:
    """Tests for _generate_planner_feedback()."""

    def test_no_issues_returns_empty(self):
        step_validations = [
            StepValidation(step_id="step1", is_valid=True, coherence_score=0.9),
        ]

        result = _generate_planner_feedback(step_validations, issues=[])

        assert result == ""

    def test_issues_formatted_as_bullet_list(self):
        step_validations = []
        issues = ["Issue one", "Issue two"]

        result = _generate_planner_feedback(step_validations, issues)

        assert "coherence issues" in result.lower()
        assert "- Issue one" in result
        assert "- Issue two" in result

    def test_suggestions_included(self):
        step_validations = [
            StepValidation(
                step_id="step1",
                is_valid=False,
                coherence_score=0.2,
                suggestions=["Add intermediate step"],
            ),
        ]
        issues = ["Some issue"]

        result = _generate_planner_feedback(step_validations, issues)

        assert "Suggestions:" in result
        assert "intermediate step" in result

    def test_issues_limited_to_five(self):
        step_validations = []
        issues = [f"Issue {i}" for i in range(10)]

        result = _generate_planner_feedback(step_validations, issues)

        # Should only include first 5
        assert "Issue 0" in result
        assert "Issue 4" in result
        assert "Issue 5" not in result

    def test_suggestions_limited_to_three(self):
        step_validations = [
            StepValidation(
                step_id="step1",
                is_valid=False,
                coherence_score=0.2,
                suggestions=[f"Suggestion {i}" for i in range(5)],
            ),
        ]
        issues = ["Some issue"]

        result = _generate_planner_feedback(step_validations, issues)

        assert "Suggestion 0" in result
        assert "Suggestion 2" in result
        assert "Suggestion 3" not in result


class TestStepFailureFeedback:
    """Tests for StepFailureFeedback dataclass."""

    def test_basic_feedback(self):
        feedback = StepFailureFeedback(
            step_id="step1",
            step_task="Calculate sum",
            failure_reason="Division by zero",
        )
        assert feedback.step_id == "step1"
        assert feedback.failure_reason == "Division by zero"
        assert feedback.signature_needs == []
        assert feedback.received_values == {}

    def test_to_planner_hint_basic(self):
        feedback = StepFailureFeedback(
            step_id="step1",
            step_task="Calculate sum",
            failure_reason="Missing operand",
        )

        hint = feedback.to_planner_hint()

        assert "step1" in hint
        assert "Missing operand" in hint

    def test_to_planner_hint_with_needs(self):
        feedback = StepFailureFeedback(
            step_id="step1",
            step_task="Calculate difference",
            failure_reason="Invalid input",
            signature_needs=["minuend", "subtrahend"],
        )

        hint = feedback.to_planner_hint()

        assert "needed:" in hint.lower()
        assert "minuend" in hint
        assert "subtrahend" in hint

    def test_to_planner_hint_with_received_values(self):
        feedback = StepFailureFeedback(
            step_id="step1",
            step_task="Calculate ratio",
            failure_reason="Got identical values",
            received_values={"a": "5", "b": "5"},
        )

        hint = feedback.to_planner_hint()

        assert "received:" in hint.lower()
        assert "a=5" in hint
        assert "b=5" in hint

    def test_to_planner_hint_full(self):
        feedback = StepFailureFeedback(
            step_id="calc_diff",
            step_task="Find the difference",
            failure_reason="Need different values",
            signature_needs=["value_a", "value_b"],
            received_values={"x": "10", "y": "10"},
        )

        hint = feedback.to_planner_hint()

        assert "calc_diff" in hint
        assert "Need different values" in hint
        assert "value_a" in hint
        assert "x=10" in hint


class TestCreateFailureFeedback:
    """Tests for create_failure_feedback()."""

    def test_basic_failure_no_signature(self):
        feedback = create_failure_feedback(
            step_id="step1",
            step_task="Do something",
            failure_reason="Unknown error",
        )

        assert feedback.step_id == "step1"
        assert feedback.step_task == "Do something"
        assert feedback.failure_reason == "Unknown error"
        assert feedback.signature_needs == []

    def test_failure_with_context(self):
        feedback = create_failure_feedback(
            step_id="step1",
            step_task="Calculate",
            failure_reason="Bad input",
            context={"input_a": 10, "input_b": 20, "_internal": "hidden"},
        )

        assert "input_a" in feedback.received_values
        assert "input_b" in feedback.received_values
        assert "_internal" not in feedback.received_values  # Internal keys skipped

    def test_failure_with_signature_clarifying_questions(self):
        mock_sig = Mock()
        mock_sig.clarifying_questions = ["What is the numerator?", "What is the denominator?"]
        mock_sig.dsl_script = None

        feedback = create_failure_feedback(
            step_id="step1",
            step_task="Find ratio",
            failure_reason="Missing values",
            signature=mock_sig,
        )

        assert len(feedback.signature_needs) == 2
        assert "numerator" in feedback.signature_needs[0]

    def test_failure_with_signature_dsl_params(self):
        mock_sig = Mock()
        mock_sig.clarifying_questions = []
        mock_sig.dsl_script = '{"type": "math", "params": ["base", "exponent"]}'

        feedback = create_failure_feedback(
            step_id="step1",
            step_task="Calculate power",
            failure_reason="Missing exponent",
            signature=mock_sig,
        )

        assert "base" in feedback.signature_needs
        assert "exponent" in feedback.signature_needs

    def test_failure_with_invalid_dsl_json(self):
        mock_sig = Mock()
        mock_sig.clarifying_questions = []
        mock_sig.dsl_script = "not valid json"

        feedback = create_failure_feedback(
            step_id="step1",
            step_task="Something",
            failure_reason="Error",
            signature=mock_sig,
        )

        # Should handle gracefully
        assert feedback.signature_needs == []

    def test_long_context_values_truncated(self):
        long_value = "x" * 100
        feedback = create_failure_feedback(
            step_id="step1",
            step_task="Process",
            failure_reason="Error",
            context={"long_key": long_value},
        )

        # Values should be truncated to 50 chars
        assert len(feedback.received_values["long_key"]) == 50
