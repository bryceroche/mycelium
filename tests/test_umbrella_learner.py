"""Tests for UmbrellaLearner auto-decomposition system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from mycelium.step_signatures.umbrella_learner import (
    UmbrellaLearner,
    MIN_USES_FOR_EVALUATION,
    MAX_SUCCESS_RATE_FOR_DECOMPOSITION,
)
from mycelium.step_signatures.models import StepSignature
from mycelium.planner import Step, DAGPlan


class TestExtractJson:
    """Tests for UmbrellaLearner._extract_json()."""

    @pytest.fixture
    def learner(self):
        """Create learner without full init."""
        learner = UmbrellaLearner.__new__(UmbrellaLearner)
        return learner

    def test_empty_input(self, learner):
        assert learner._extract_json("") is None
        assert learner._extract_json("   ") is None

    def test_simple_json_object(self, learner):
        result = learner._extract_json('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_nested_json_object(self, learner):
        text = '{"outer": {"inner": 42}}'
        result = learner._extract_json(text)
        assert result == '{"outer": {"inner": 42}}'

    def test_json_with_array(self, learner):
        text = '{"items": [1, 2, 3]}'
        result = learner._extract_json(text)
        assert result == '{"items": [1, 2, 3]}'

    def test_json_in_markdown_block(self, learner):
        text = '''Here is the result:
```json
{"clarifying_questions": ["What is x?"]}
```
Done.'''
        result = learner._extract_json(text)
        assert result is not None
        assert "clarifying_questions" in result

    def test_json_with_surrounding_text(self, learner):
        text = 'The answer is {"result": 42} and that is final.'
        result = learner._extract_json(text)
        assert result == '{"result": 42}'

    def test_missing_opening_brace(self, learner):
        # Edge case: response starts with quote (missing {)
        text = '"clarifying_questions": ["What?"], "params": []}'
        result = learner._extract_json(text)
        assert result is not None
        # Should have added the opening brace
        assert result.startswith("{")

    def test_deeply_nested(self, learner):
        text = '{"a": {"b": {"c": {"d": 1}}}}'
        result = learner._extract_json(text)
        assert result == text

    def test_no_json_found(self, learner):
        assert learner._extract_json("just plain text") is None
        assert learner._extract_json("no json here at all") is None

    def test_unbalanced_braces(self, learner):
        # Should extract up to first balanced point
        text = '{"valid": 1} extra { stuff'
        result = learner._extract_json(text)
        assert result == '{"valid": 1}'


class TestGetDecompositionCandidates:
    """Tests for UmbrellaLearner.get_decomposition_candidates()."""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.fixture
    def learner(self, mock_db):
        learner = UmbrellaLearner.__new__(UmbrellaLearner)
        learner.db = mock_db
        return learner

    def _make_sig(self, id, dsl_type, uses, successes, is_umbrella=False):
        """Helper to create test signatures."""
        sig = StepSignature(
            id=id,
            step_type=f"step_{id}",
            description=f"Test step {id}",
            dsl_type=dsl_type,
            uses=uses,
            successes=successes,
            is_semantic_umbrella=is_umbrella,
        )
        return sig

    def test_empty_db(self, learner, mock_db):
        mock_db.get_all_signatures.return_value = []
        candidates = learner.get_decomposition_candidates()
        assert candidates == []

    def test_filters_non_decompose_type(self, learner, mock_db):
        # Signatures with dsl_type != "decompose" should be excluded
        mock_db.get_all_signatures.return_value = [
            self._make_sig(1, "math", uses=10, successes=2),
            self._make_sig(2, "sympy", uses=10, successes=2),
            self._make_sig(3, "python", uses=10, successes=2),
        ]
        candidates = learner.get_decomposition_candidates()
        assert candidates == []

    def test_filters_insufficient_uses(self, learner, mock_db):
        # Signatures with uses < MIN_USES_FOR_EVALUATION should be excluded
        mock_db.get_all_signatures.return_value = [
            self._make_sig(1, "decompose", uses=1, successes=0),
            self._make_sig(2, "decompose", uses=2, successes=0),
        ]
        candidates = learner.get_decomposition_candidates()
        assert candidates == []

    def test_filters_high_success_rate(self, learner, mock_db):
        # Signatures with success_rate > MAX_SUCCESS_RATE_FOR_DECOMPOSITION excluded
        uses = MIN_USES_FOR_EVALUATION
        high_successes = int(uses * (MAX_SUCCESS_RATE_FOR_DECOMPOSITION + 0.2)) + 1
        mock_db.get_all_signatures.return_value = [
            self._make_sig(1, "decompose", uses=uses, successes=high_successes),
        ]
        candidates = learner.get_decomposition_candidates()
        assert candidates == []

    def test_filters_already_umbrella(self, learner, mock_db):
        # Signatures already promoted to umbrella should be excluded
        mock_db.get_all_signatures.return_value = [
            self._make_sig(1, "decompose", uses=10, successes=2, is_umbrella=True),
        ]
        candidates = learner.get_decomposition_candidates()
        assert candidates == []

    def test_includes_valid_candidate(self, learner, mock_db):
        # Signature that meets all criteria: decompose type, enough uses, low success
        mock_db.get_all_signatures.return_value = [
            self._make_sig(1, "decompose", uses=5, successes=1),  # 20% success
        ]
        candidates = learner.get_decomposition_candidates()
        assert len(candidates) == 1
        assert candidates[0].id == 1

    def test_multiple_candidates(self, learner, mock_db):
        mock_db.get_all_signatures.return_value = [
            self._make_sig(1, "decompose", uses=10, successes=2),  # Valid
            self._make_sig(2, "math", uses=10, successes=2),  # Wrong type
            self._make_sig(3, "decompose", uses=1, successes=0),  # Not enough uses
            self._make_sig(4, "decompose", uses=10, successes=9),  # Too successful
            self._make_sig(5, "decompose", uses=5, successes=0),  # Valid
        ]
        candidates = learner.get_decomposition_candidates()
        assert len(candidates) == 2
        assert {c.id for c in candidates} == {1, 5}


class TestGenerateNlInterface:
    """Tests for UmbrellaLearner.generate_nl_interface()."""

    @pytest.fixture
    def mock_client(self):
        client = AsyncMock()
        return client

    @pytest.fixture
    def learner(self, mock_client):
        learner = UmbrellaLearner.__new__(UmbrellaLearner)
        learner._client = mock_client
        return learner

    @pytest.mark.asyncio
    async def test_successful_generation(self, learner, mock_client):
        mock_client.generate.return_value = '''{
            "clarifying_questions": ["What is the base?", "What is the exponent?"],
            "param_descriptions": {"base": "The number", "exponent": "The power"},
            "params": ["base", "exponent"]
        }'''

        result = await learner.generate_nl_interface("Compute base raised to power")

        assert len(result["clarifying_questions"]) == 2
        assert "base" in result["param_descriptions"]
        assert "exponent" in result["params"]

    @pytest.mark.asyncio
    async def test_empty_response(self, learner, mock_client):
        mock_client.generate.return_value = ""

        result = await learner.generate_nl_interface("Some step")

        assert result == {"clarifying_questions": [], "param_descriptions": {}, "params": []}

    @pytest.mark.asyncio
    async def test_invalid_json_response(self, learner, mock_client):
        mock_client.generate.return_value = "not valid json at all"

        result = await learner.generate_nl_interface("Some step")

        assert result == {"clarifying_questions": [], "param_descriptions": {}, "params": []}

    @pytest.mark.asyncio
    async def test_client_error(self, learner, mock_client):
        mock_client.generate.side_effect = Exception("API error")

        result = await learner.generate_nl_interface("Some step")

        assert result == {"clarifying_questions": [], "param_descriptions": {}, "params": []}

    @pytest.mark.asyncio
    async def test_partial_response(self, learner, mock_client):
        # Response missing some fields
        mock_client.generate.return_value = '{"clarifying_questions": ["What?"]}'

        result = await learner.generate_nl_interface("Some step")

        assert result["clarifying_questions"] == ["What?"]
        assert result["param_descriptions"] == {}
        assert result["params"] == []


class TestDecomposeSignature:
    """Tests for UmbrellaLearner.decompose_signature()."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.get_children.return_value = []
        return db

    @pytest.fixture
    def mock_planner(self):
        return AsyncMock()

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.embed.return_value = np.zeros(768)
        return embedder

    @pytest.fixture
    def learner(self, mock_db, mock_planner, mock_embedder):
        learner = UmbrellaLearner.__new__(UmbrellaLearner)
        learner.db = mock_db
        learner.planner = mock_planner
        learner.embedder = mock_embedder
        learner._client = AsyncMock()
        learner._client.generate.return_value = '{"clarifying_questions": [], "param_descriptions": {}, "params": []}'
        return learner

    def _make_sig(self, id, is_umbrella=False, depth=0):
        return StepSignature(
            id=id,
            step_type=f"step_{id}",
            description=f"Description for step {id}",
            dsl_type="decompose",
            is_semantic_umbrella=is_umbrella,
            depth=depth,
        )

    @pytest.mark.asyncio
    async def test_skip_existing_umbrella_with_children(self, learner, mock_db):
        sig = self._make_sig(1, is_umbrella=True)
        mock_db.get_children.return_value = [self._make_sig(2)]

        result = await learner.decompose_signature(sig)

        assert result == []
        mock_db.promote_to_umbrella.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_step_plan_marks_atomic(self, learner, mock_db, mock_planner):
        sig = self._make_sig(1)
        mock_planner.decompose.return_value = DAGPlan(
            steps=[Step(id="s1", task="Single step")],
            problem="test"
        )

        result = await learner.decompose_signature(sig)

        assert result == []
        mock_db.update_signature.assert_called_once()
        call_kwargs = mock_db.update_signature.call_args[1]
        assert call_kwargs["dsl_type"] == "atomic"

    @pytest.mark.asyncio
    async def test_creates_children_from_plan(self, learner, mock_db, mock_planner):
        sig = self._make_sig(1)
        mock_planner.decompose.return_value = DAGPlan(
            steps=[
                Step(id="s1", task="First sub-step"),
                Step(id="s2", task="Second sub-step"),
            ],
            problem="test"
        )

        # Mock find_deeper_signature to return None (no repoint)
        mock_db.find_deeper_signature.return_value = None

        # Mock find_or_create to create new signatures
        child1 = self._make_sig(10)
        child2 = self._make_sig(11)
        mock_db.find_or_create.side_effect = [(child1, True), (child2, True)]

        result = await learner.decompose_signature(sig)

        assert len(result) == 2
        assert mock_db.add_child.call_count == 2
        mock_db.promote_to_umbrella.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_skips_self_reference(self, learner, mock_db, mock_planner):
        sig = self._make_sig(1)
        mock_planner.decompose.return_value = DAGPlan(
            steps=[
                Step(id="s1", task="Same as parent"),
                Step(id="s2", task="Different step"),
            ],
            problem="test"
        )

        mock_db.find_deeper_signature.return_value = None

        # First find_or_create returns the parent itself (self-reference)
        # Second returns a different signature
        child2 = self._make_sig(2)
        mock_db.find_or_create.side_effect = [(sig, False), (child2, True)]

        result = await learner.decompose_signature(sig)

        # Should only add the non-self-referencing child
        assert len(result) == 1
        assert result[0] == 2

    @pytest.mark.asyncio
    async def test_skips_synthesis_steps(self, learner, mock_db, mock_planner):
        sig = self._make_sig(1)
        mock_planner.decompose.return_value = DAGPlan(
            steps=[
                Step(id="s1", task="Calculate value"),
                Step(id="s2", task="Combine final results"),  # Should skip
            ],
            problem="test"
        )

        mock_db.find_deeper_signature.return_value = None
        child1 = self._make_sig(10)
        mock_db.find_or_create.return_value = (child1, True)

        result = await learner.decompose_signature(sig)

        # Only one child (synthesis step skipped)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_repoints_to_existing_deeper_sig(self, learner, mock_db, mock_planner):
        sig = self._make_sig(1, depth=0)
        mock_planner.decompose.return_value = DAGPlan(
            steps=[Step(id="s1", task="Some step")],
            problem="test"
        )

        # Wait, this would be single step - let's use 2
        mock_planner.decompose.return_value = DAGPlan(
            steps=[
                Step(id="s1", task="Step one"),
                Step(id="s2", task="Step two"),
            ],
            problem="test"
        )

        # find_deeper_signature returns existing signature
        existing_child = self._make_sig(99, depth=2)
        mock_db.find_deeper_signature.return_value = existing_child

        result = await learner.decompose_signature(sig)

        # Should use existing sigs, not create new ones
        mock_db.find_or_create.assert_not_called()
        assert len(result) == 2


class TestLearnFromFailures:
    """Tests for UmbrellaLearner.learn_from_failures() end-to-end."""

    @pytest.fixture
    def mock_db(self):
        return MagicMock()

    @pytest.fixture
    def learner(self, mock_db):
        learner = UmbrellaLearner.__new__(UmbrellaLearner)
        learner.db = mock_db
        learner.planner = AsyncMock()
        learner.embedder = MagicMock()
        learner.embedder.embed.return_value = np.zeros(768)
        learner._client = AsyncMock()
        learner._client.generate.return_value = '{"clarifying_questions": [], "param_descriptions": {}, "params": []}'
        return learner

    @pytest.mark.asyncio
    async def test_no_candidates(self, learner, mock_db):
        mock_db.get_all_signatures.return_value = []

        result = await learner.learn_from_failures()

        assert result == {"candidates": 0, "decomposed": 0, "children_created": 0}

    @pytest.mark.asyncio
    async def test_counts_decompositions(self, learner, mock_db):
        # Create a failing signature
        sig = StepSignature(
            id=1,
            step_type="failing_step",
            description="A step that fails",
            dsl_type="decompose",
            uses=10,
            successes=2,
            is_semantic_umbrella=False,
        )
        mock_db.get_all_signatures.return_value = [sig]
        mock_db.get_children.return_value = []
        mock_db.find_deeper_signature.return_value = None

        # Plan with 2 steps
        learner.planner.decompose.return_value = DAGPlan(
            steps=[
                Step(id="s1", task="Child step 1"),
                Step(id="s2", task="Child step 2"),
            ],
            problem="test"
        )

        child1 = StepSignature(id=10, step_type="child_1")
        child2 = StepSignature(id=11, step_type="child_2")
        mock_db.find_or_create.side_effect = [(child1, True), (child2, True)]

        result = await learner.learn_from_failures()

        assert result["candidates"] == 1
        assert result["decomposed"] == 1
        assert result["children_created"] == 2

    @pytest.mark.asyncio
    async def test_handles_decomposition_error(self, learner, mock_db):
        sig = StepSignature(
            id=1,
            step_type="error_step",
            dsl_type="decompose",
            uses=10,
            successes=2,
            is_semantic_umbrella=False,
        )
        mock_db.get_all_signatures.return_value = [sig]
        mock_db.get_children.return_value = []

        # Planner throws error
        learner.planner.decompose.side_effect = Exception("Planner failed")

        result = await learner.learn_from_failures()

        # Should handle error gracefully
        assert result["candidates"] == 1
        assert result["decomposed"] == 0
        assert result["children_created"] == 0
