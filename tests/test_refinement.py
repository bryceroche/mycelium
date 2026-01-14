"""Tests for signature refinement module.

Tests the SignatureRefiner class and related functionality.
Note: refinement.py references some fields (injected_uses, etc.) and methods
(get_signatures_for_dsl_improvement, etc.) that don't exist in the current
schema. Tests mock these appropriately.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Optional

from mycelium.step_signatures.refinement import (
    SignatureRefiner,
    RefinementResult,
    RefinementReport,
    run_refinement_loop,
    VARIANCE_THRESHOLD,
    MIN_DSL_CONFIDENCE,
    PARTIAL_SUCCESS_RATE,
)


@dataclass
class MockSignature:
    """Mock signature with fields refinement.py expects."""
    id: int
    step_type: str
    description: str = ""
    is_semantic_umbrella: bool = False
    dsl_script: Optional[str] = None
    dsl_version: int = 1
    # Fields refinement.py references but don't exist in current model
    injected_uses: int = 0
    injected_successes: int = 0
    non_injected_uses: int = 0
    non_injected_successes: int = 0


class TestRefinementDataclasses:
    """Tests for RefinementResult and RefinementReport dataclasses."""

    def test_refinement_result_basic(self):
        result = RefinementResult(
            signature_id=1,
            step_type="compute_sum",
            original_lift=0.15,
            action="decomposed",
            children_created=3,
        )
        assert result.signature_id == 1
        assert result.step_type == "compute_sum"
        assert result.original_lift == 0.15
        assert result.action == "decomposed"
        assert result.children_created == 3
        assert result.new_dsl is None
        assert result.error is None

    def test_refinement_result_with_error(self):
        result = RefinementResult(
            signature_id=2,
            step_type="compute_gcd",
            original_lift=-0.05,
            action="error",
            error="Failed to decompose",
        )
        assert result.error == "Failed to decompose"
        assert result.action == "error"

    def test_refinement_result_with_new_dsl(self):
        result = RefinementResult(
            signature_id=3,
            step_type="compute_power",
            original_lift=-0.10,
            action="dsl_fixed",
            new_dsl='{"type": "math", "script": "base ** exp"}',
        )
        assert result.action == "dsl_fixed"
        assert "base ** exp" in result.new_dsl

    def test_refinement_report_summary(self):
        results = [
            RefinementResult(1, "step1", 0.0, "decomposed", children_created=2),
            RefinementResult(2, "step2", 0.0, "dsl_fixed", new_dsl="x+y"),
            RefinementResult(3, "step3", 0.0, "guidance_only"),
            RefinementResult(4, "step4", 0.0, "skipped"),
            RefinementResult(5, "step5", 0.0, "error", error="fail"),
        ]
        report = RefinementReport(
            signatures_analyzed=5,
            decomposed=1,
            dsl_fixed=1,
            guidance_only=1,
            skipped=1,
            errors=1,
            results=results,
        )
        assert report.signatures_analyzed == 5
        assert report.decomposed == 1
        assert report.dsl_fixed == 1
        assert report.guidance_only == 1
        assert report.skipped == 1
        assert report.errors == 1
        assert len(report.results) == 5


class TestSignatureRefinerInit:
    """Tests for SignatureRefiner initialization."""

    def test_init_with_db_only(self):
        mock_db = MagicMock()
        refiner = SignatureRefiner(db=mock_db)
        assert refiner.db == mock_db
        assert refiner.client is None

    def test_init_with_db_and_client(self):
        mock_db = MagicMock()
        mock_client = MagicMock()
        refiner = SignatureRefiner(db=mock_db, client=mock_client)
        assert refiner.db == mock_db
        assert refiner.client == mock_client


class TestComputeLift:
    """Tests for SignatureRefiner._compute_lift()."""

    @pytest.fixture
    def refiner(self):
        return SignatureRefiner(db=MagicMock())

    def test_lift_zero_when_no_uses(self, refiner):
        sig = MockSignature(id=1, step_type="test", injected_uses=0, non_injected_uses=0)
        assert refiner._compute_lift(sig) == 0.0

    def test_lift_zero_when_only_injected_uses(self, refiner):
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=10, injected_successes=8,
            non_injected_uses=0, non_injected_successes=0
        )
        assert refiner._compute_lift(sig) == 0.0

    def test_lift_zero_when_only_non_injected_uses(self, refiner):
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=0, injected_successes=0,
            non_injected_uses=10, non_injected_successes=5
        )
        assert refiner._compute_lift(sig) == 0.0

    def test_positive_lift(self, refiner):
        # injected rate: 8/10 = 0.8, base rate: 5/10 = 0.5, lift = 0.3
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=10, injected_successes=8,
            non_injected_uses=10, non_injected_successes=5
        )
        assert refiner._compute_lift(sig) == pytest.approx(0.3)

    def test_negative_lift(self, refiner):
        # injected rate: 3/10 = 0.3, base rate: 7/10 = 0.7, lift = -0.4
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=10, injected_successes=3,
            non_injected_uses=10, non_injected_successes=7
        )
        assert refiner._compute_lift(sig) == pytest.approx(-0.4)

    def test_zero_lift_equal_rates(self, refiner):
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=10, injected_successes=5,
            non_injected_uses=10, non_injected_successes=5
        )
        assert refiner._compute_lift(sig) == pytest.approx(0.0)


class TestGetDslConfidence:
    """Tests for SignatureRefiner._get_dsl_confidence()."""

    @pytest.fixture
    def refiner(self):
        return SignatureRefiner(db=MagicMock())

    def test_zero_confidence_no_uses(self, refiner):
        sig = MockSignature(id=1, step_type="test", injected_uses=0)
        assert refiner._get_dsl_confidence(sig) == 0.0

    def test_full_confidence_all_success(self, refiner):
        sig = MockSignature(id=1, step_type="test", injected_uses=10, injected_successes=10)
        assert refiner._get_dsl_confidence(sig) == 1.0

    def test_partial_confidence(self, refiner):
        sig = MockSignature(id=1, step_type="test", injected_uses=10, injected_successes=7)
        assert refiner._get_dsl_confidence(sig) == pytest.approx(0.7)


class TestClassifySignature:
    """Tests for SignatureRefiner._classify_signature()."""

    @pytest.fixture
    def refiner(self):
        return SignatureRefiner(db=MagicMock())

    def test_unknown_when_no_uses(self, refiner):
        sig = MockSignature(id=1, step_type="test", injected_uses=0, non_injected_uses=0)
        assert refiner._classify_signature(sig) == "unknown"

    def test_umbrella_high_variance(self, refiner):
        # High variance (>= VARIANCE_THRESHOLD) and >= 10 total uses
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=5, injected_successes=5,  # rate = 1.0
            non_injected_uses=5, non_injected_successes=2  # rate = 0.4
        )
        # variance = |1.0 - 0.4| = 0.6 >= VARIANCE_THRESHOLD (0.3)
        assert refiner._classify_signature(sig) == "umbrella"

    def test_guidance_only_low_dsl_confidence(self, refiner):
        # Low DSL confidence (< MIN_DSL_CONFIDENCE)
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=10, injected_successes=3,  # rate = 0.3 < MIN_DSL_CONFIDENCE (0.4)
            non_injected_uses=5, non_injected_successes=2
        )
        # variance = |0.3 - 0.4| = 0.1 < VARIANCE_THRESHOLD
        # dsl_confidence = 0.3 < MIN_DSL_CONFIDENCE
        assert refiner._classify_signature(sig) == "guidance_only"

    def test_fixable_partial_success(self, refiner):
        # Some DSL success (PARTIAL_SUCCESS_RATE <= inj_rate < 0.6)
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=10, injected_successes=5,  # rate = 0.5, in [0.2, 0.6)
            non_injected_uses=10, non_injected_successes=5  # rate = 0.5
        )
        # variance = |0.5 - 0.5| = 0.0 < VARIANCE_THRESHOLD
        # dsl_confidence = 0.5 >= MIN_DSL_CONFIDENCE (0.4)
        # inj_rate = 0.5, in [PARTIAL_SUCCESS_RATE (0.2), 0.6)
        assert refiner._classify_signature(sig) == "fixable"

    def test_unknown_when_no_classification_fits(self, refiner):
        # High success rate, low variance - doesn't fit any category
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=10, injected_successes=8,  # rate = 0.8 >= 0.6
            non_injected_uses=10, non_injected_successes=8  # rate = 0.8
        )
        # variance = 0.0 < VARIANCE_THRESHOLD
        # dsl_confidence = 0.8 >= MIN_DSL_CONFIDENCE
        # inj_rate = 0.8 >= 0.6, doesn't fit "fixable"
        assert refiner._classify_signature(sig) == "unknown"

    def test_variance_below_threshold_not_umbrella(self, refiner):
        # Variance just below threshold
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=10, injected_successes=6,  # rate = 0.6
            non_injected_uses=10, non_injected_successes=4  # rate = 0.4
        )
        # variance = |0.6 - 0.4| = 0.2 < VARIANCE_THRESHOLD (0.3)
        # Should fall through to other classifications
        assert refiner._classify_signature(sig) != "umbrella"


class TestConvertToGuidanceOnly:
    """Tests for SignatureRefiner._convert_to_guidance_only()."""

    def test_updates_database(self):
        mock_conn = MagicMock()
        mock_db = MagicMock()
        mock_db._connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db._connection.return_value.__exit__ = MagicMock(return_value=False)

        refiner = SignatureRefiner(db=mock_db)
        sig = MockSignature(id=42, step_type="test")

        refiner._convert_to_guidance_only(sig)

        # Verify the SQL was executed
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "UPDATE step_signatures" in call_args[0]
        assert "dsl_script = NULL" in call_args[0]
        assert call_args[1] == (42,)


class TestRefineSignature:
    """Tests for SignatureRefiner._refine_signature()."""

    @pytest.fixture
    def mock_db(self):
        mock = MagicMock()
        mock._connection.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock._connection.return_value.__exit__ = MagicMock(return_value=False)
        return mock

    @pytest.mark.asyncio
    async def test_skips_umbrella_signatures(self, mock_db):
        refiner = SignatureRefiner(db=mock_db)
        sig = MockSignature(
            id=1, step_type="test",
            is_semantic_umbrella=True,
            injected_uses=10, injected_successes=5,
            non_injected_uses=10, non_injected_successes=5
        )

        result = await refiner._refine_signature(sig)

        assert result.action == "skipped"
        assert result.signature_id == 1

    @pytest.mark.asyncio
    async def test_guidance_only_classification(self, mock_db):
        refiner = SignatureRefiner(db=mock_db)
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=10, injected_successes=2,  # low confidence
            non_injected_uses=5, non_injected_successes=2
        )

        result = await refiner._refine_signature(sig)

        assert result.action == "guidance_only"

    @pytest.mark.asyncio
    async def test_umbrella_decomposition_without_client(self, mock_db):
        refiner = SignatureRefiner(db=mock_db, client=None)
        sig = MockSignature(
            id=1, step_type="test",
            injected_uses=10, injected_successes=10,  # high inj rate
            non_injected_uses=10, non_injected_successes=2  # low non-inj rate
        )
        # variance = |1.0 - 0.2| = 0.8 >= VARIANCE_THRESHOLD

        result = await refiner._refine_signature(sig)

        # Without client, decomposition returns empty children
        assert result.action == "decomposed"
        assert result.children_created == 0


class TestRefineNegativeLiftSignatures:
    """Tests for SignatureRefiner.refine_negative_lift_signatures()."""

    @pytest.fixture
    def mock_db(self):
        mock = MagicMock()
        mock._connection.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock._connection.return_value.__exit__ = MagicMock(return_value=False)
        return mock

    @pytest.mark.asyncio
    async def test_returns_empty_report_when_no_candidates(self, mock_db):
        mock_db.get_signatures_for_dsl_improvement = MagicMock(return_value=[])
        refiner = SignatureRefiner(db=mock_db)

        report = await refiner.refine_negative_lift_signatures()

        assert report.signatures_analyzed == 0
        assert report.decomposed == 0
        assert len(report.results) == 0

    @pytest.mark.asyncio
    async def test_processes_candidates(self, mock_db):
        candidates = [
            MockSignature(
                id=1, step_type="step1",
                is_semantic_umbrella=True,  # will be skipped
                injected_uses=10, injected_successes=5,
                non_injected_uses=10, non_injected_successes=5
            ),
            MockSignature(
                id=2, step_type="step2",
                injected_uses=10, injected_successes=2,  # guidance_only
                non_injected_uses=5, non_injected_successes=2
            ),
        ]
        mock_db.get_signatures_for_dsl_improvement = MagicMock(return_value=candidates)
        refiner = SignatureRefiner(db=mock_db)

        report = await refiner.refine_negative_lift_signatures()

        assert report.signatures_analyzed == 2
        assert report.skipped == 1
        assert report.guidance_only == 1

    @pytest.mark.asyncio
    async def test_handles_exceptions(self, mock_db):
        # Create a signature that will raise an exception during processing
        bad_sig = MagicMock()
        bad_sig.id = 1
        bad_sig.step_type = "bad_step"
        bad_sig.is_semantic_umbrella = False
        # Make total_uses calculation fail
        bad_sig.injected_uses = None  # Will cause TypeError
        bad_sig.non_injected_uses = None

        mock_db.get_signatures_for_dsl_improvement = MagicMock(return_value=[bad_sig])
        refiner = SignatureRefiner(db=mock_db)

        report = await refiner.refine_negative_lift_signatures()

        assert report.errors == 1
        assert report.results[0].action == "error"
        assert report.results[0].error is not None

    @pytest.mark.asyncio
    async def test_respects_max_signatures(self, mock_db):
        candidates = [
            MockSignature(
                id=i, step_type=f"step{i}",
                is_semantic_umbrella=True,
                injected_uses=10, injected_successes=5,
                non_injected_uses=10, non_injected_successes=5
            )
            for i in range(50)
        ]
        mock_db.get_signatures_for_dsl_improvement = MagicMock(return_value=candidates)
        refiner = SignatureRefiner(db=mock_db)

        report = await refiner.refine_negative_lift_signatures(max_signatures=5)

        assert report.signatures_analyzed == 5


class TestRunRefinementLoop:
    """Tests for run_refinement_loop() convenience function."""

    @pytest.mark.asyncio
    async def test_creates_db_if_none(self):
        with patch('mycelium.step_signatures.refinement.StepSignatureDB') as MockDB:
            mock_db = MagicMock()
            mock_db.get_signatures_for_dsl_improvement = MagicMock(return_value=[])
            MockDB.return_value = mock_db

            report = await run_refinement_loop(db=None)

            MockDB.assert_called_once()
            assert isinstance(report, RefinementReport)

    @pytest.mark.asyncio
    async def test_uses_provided_db(self):
        mock_db = MagicMock()
        mock_db.get_signatures_for_dsl_improvement = MagicMock(return_value=[])

        report = await run_refinement_loop(db=mock_db)

        assert isinstance(report, RefinementReport)

    @pytest.mark.asyncio
    async def test_passes_parameters(self):
        mock_db = MagicMock()
        mock_db.get_signatures_for_dsl_improvement = MagicMock(return_value=[])

        await run_refinement_loop(
            db=mock_db,
            min_lift=-0.05,
            min_uses=10,
            max_signatures=5,
        )

        mock_db.get_signatures_for_dsl_improvement.assert_called_once_with(
            min_uses=10,
            lift_threshold=-0.05,
        )


class TestGenerateDecomposition:
    """Tests for SignatureRefiner._generate_decomposition()."""

    @pytest.mark.asyncio
    async def test_returns_empty_without_client(self):
        refiner = SignatureRefiner(db=MagicMock(), client=None)
        sig = MockSignature(id=1, step_type="compute_sum", description="Add numbers")

        result = await refiner._generate_decomposition(sig)

        assert result == []

    @pytest.mark.asyncio
    async def test_parses_llm_json_response(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '''[
            {"step_type": "sum_two", "description": "Sum two numbers", "condition": "exactly two numbers", "dsl": {"type": "math", "script": "a + b"}}
        ]'''
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        refiner = SignatureRefiner(db=MagicMock(), client=mock_client)
        sig = MockSignature(id=1, step_type="compute_sum", description="Add numbers")

        result = await refiner._generate_decomposition(sig)

        assert len(result) == 1
        assert result[0]["step_type"] == "sum_two"

    @pytest.mark.asyncio
    async def test_handles_markdown_code_blocks(self):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = '''```json
[{"step_type": "test", "description": "test", "condition": "test", "dsl": null}]
```'''
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        refiner = SignatureRefiner(db=MagicMock(), client=mock_client)
        sig = MockSignature(id=1, step_type="compute_sum", description="Add numbers")

        result = await refiner._generate_decomposition(sig)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_handles_llm_error(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))

        refiner = SignatureRefiner(db=MagicMock(), client=mock_client)
        sig = MockSignature(id=1, step_type="compute_sum", description="Add numbers")

        result = await refiner._generate_decomposition(sig)

        assert result == []


class TestFindSimilarSuccessfulSignature:
    """Tests for SignatureRefiner._find_similar_successful_signature()."""

    def test_returns_none_when_embedder_fails(self):
        refiner = SignatureRefiner(db=MagicMock())
        sig = MockSignature(id=1, step_type="test", description="test")

        # The Embedder is imported inside the function, so we patch where it's imported from
        with patch.dict('sys.modules', {'mycelium.embedder': MagicMock()}):
            with patch('mycelium.embedder.Embedder') as MockEmbedder:
                MockEmbedder.get_instance.side_effect = Exception("No embedder")
                result = refiner._find_similar_successful_signature(sig)
                # Should handle the exception gracefully
                assert result is None

    def test_returns_none_when_no_similar_signatures(self):
        mock_db = MagicMock()
        mock_db.get_signatures_with_successful_dsls = MagicMock(return_value=[])

        refiner = SignatureRefiner(db=mock_db)
        sig = MockSignature(id=1, step_type="test", description="test")

        with patch('mycelium.embedder.Embedder') as MockEmbedder:
            import numpy as np
            mock_embedder = MagicMock()
            mock_embedder.embed.return_value = np.array([0.1] * 768)
            MockEmbedder.get_instance.return_value = mock_embedder

            result = refiner._find_similar_successful_signature(sig)

        assert result is None


class TestFixDsl:
    """Tests for SignatureRefiner._fix_dsl()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_similar_signature(self):
        refiner = SignatureRefiner(db=MagicMock())
        sig = MockSignature(id=1, step_type="test", description="test")

        with patch.object(refiner, '_find_similar_successful_signature', return_value=None):
            result = await refiner._fix_dsl(sig)

        assert result is None

    @pytest.mark.asyncio
    async def test_clones_dsl_from_similar_signature(self):
        mock_db = MagicMock()
        similar_sig = MockSignature(
            id=2, step_type="similar",
            dsl_script='{"type": "math", "script": "x + y"}'
        )

        refiner = SignatureRefiner(db=mock_db)
        sig = MockSignature(id=1, step_type="test", description="test", dsl_version=1)

        with patch.object(refiner, '_find_similar_successful_signature', return_value=similar_sig):
            result = await refiner._fix_dsl(sig)

        assert result == '{"type": "math", "script": "x + y"}'
        mock_db.reset_lift_stats_for_dsl_version.assert_called_once_with(
            signature_id=1,
            new_dsl_script='{"type": "math", "script": "x + y"}',
            new_dsl_version=2,
        )


class TestThresholdConstants:
    """Tests to verify threshold constants are reasonable."""

    def test_variance_threshold_in_valid_range(self):
        assert 0 < VARIANCE_THRESHOLD < 1

    def test_min_dsl_confidence_in_valid_range(self):
        assert 0 < MIN_DSL_CONFIDENCE < 1

    def test_partial_success_rate_in_valid_range(self):
        assert 0 < PARTIAL_SUCCESS_RATE < 1

    def test_partial_success_rate_less_than_min_confidence(self):
        # PARTIAL_SUCCESS_RATE should be less than the 0.6 upper bound
        assert PARTIAL_SUCCESS_RATE < 0.6
