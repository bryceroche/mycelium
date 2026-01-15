"""Integration tests for DSL auto-rewriter."""

import json
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mycelium.step_signatures.db import StepSignatureDB
from mycelium.step_signatures.dsl_rewriter import (
    RewriteCandidate,
    RewriteResult,
    find_underperforming_signatures,
    generate_improved_dsl,
    rewrite_underperforming_dsls,
    _parse_dsl_response,
)


EMBEDDING_DIM = 768


def make_embedding(seed: int) -> np.ndarray:
    """Create a deterministic 768-dim embedding from a seed."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(EMBEDDING_DIM)
    return emb / np.linalg.norm(emb)


class TestFindUnderperformingSignatures:
    """Tests for finding signatures that need rewriting."""

    def test_finds_low_success_rate_signatures(self, clean_test_db):
        """Should find signatures with success rate below threshold."""
        db = StepSignatureDB()

        # Create a signature with low success rate
        emb = make_embedding(100)
        sig, _ = db.find_or_create(
            step_text="Failing signature test",
            embedding=emb,
            parent_problem="test",
        )

        # Set up stats: 20 uses, 5 successes = 25% success rate
        with db._connection() as conn:
            conn.execute(
                """UPDATE step_signatures
                   SET uses = 20, successes = 5, dsl_type = 'math'
                   WHERE id = ?""",
                (sig.id,)
            )

        # Find underperforming (threshold 40%)
        candidates = find_underperforming_signatures(
            db,
            min_uses=10,
            max_success_rate=0.40,
            min_traffic_share=0.0,  # Disable traffic filter for test
        )

        assert len(candidates) == 1
        assert candidates[0].signature_id == sig.id
        assert candidates[0].success_rate == 0.25

    def test_ignores_high_success_rate_signatures(self, clean_test_db):
        """Should not find signatures above success threshold."""
        db = StepSignatureDB()

        emb = make_embedding(200)
        sig, _ = db.find_or_create(
            step_text="Successful signature test",
            embedding=emb,
            parent_problem="test",
        )

        # Set up stats: 20 uses, 15 successes = 75% success rate
        with db._connection() as conn:
            conn.execute(
                """UPDATE step_signatures
                   SET uses = 20, successes = 15, dsl_type = 'math'
                   WHERE id = ?""",
                (sig.id,)
            )

        candidates = find_underperforming_signatures(
            db,
            min_uses=10,
            max_success_rate=0.40,
            min_traffic_share=0.0,
        )

        assert len(candidates) == 0

    def test_ignores_low_usage_signatures(self, clean_test_db):
        """Should not find signatures with too few uses."""
        db = StepSignatureDB()

        emb = make_embedding(300)
        sig, _ = db.find_or_create(
            step_text="Low usage signature test",
            embedding=emb,
            parent_problem="test",
        )

        # Set up stats: only 5 uses (below threshold)
        with db._connection() as conn:
            conn.execute(
                """UPDATE step_signatures
                   SET uses = 5, successes = 1, dsl_type = 'math'
                   WHERE id = ?""",
                (sig.id,)
            )

        candidates = find_underperforming_signatures(
            db,
            min_uses=10,
            max_success_rate=0.40,
            min_traffic_share=0.0,
        )

        assert len(candidates) == 0

    def test_ignores_decompose_signatures(self, clean_test_db):
        """Should not find decompose signatures (only math DSLs)."""
        db = StepSignatureDB()

        emb = make_embedding(400)
        sig, _ = db.find_or_create(
            step_text="Decompose signature test",
            embedding=emb,
            parent_problem="test",
        )

        # Set up as decompose type with low success rate
        with db._connection() as conn:
            conn.execute(
                """UPDATE step_signatures
                   SET uses = 20, successes = 2, dsl_type = 'decompose'
                   WHERE id = ?""",
                (sig.id,)
            )

        candidates = find_underperforming_signatures(
            db,
            min_uses=10,
            max_success_rate=0.40,
            min_traffic_share=0.0,
        )

        assert len(candidates) == 0


class TestParseDslResponse:
    """Tests for parsing LLM-generated DSL responses."""

    def test_parses_valid_json(self):
        """Should parse valid DSL JSON."""
        response = '{"script": "a + b", "params": ["a", "b"]}'
        result = _parse_dsl_response(response)

        assert result is not None
        assert result["script"] == "a + b"
        assert result["params"] == ["a", "b"]
        assert result["type"] == "math"  # Default

    def test_parses_markdown_code_block(self):
        """Should extract JSON from markdown code blocks."""
        response = '''```json
{"script": "x * y", "params": ["x", "y"]}
```'''
        result = _parse_dsl_response(response)

        assert result is not None
        assert result["script"] == "x * y"

    def test_rejects_invalid_json(self):
        """Should return None for invalid JSON."""
        result = _parse_dsl_response("not valid json")
        assert result is None

    def test_rejects_missing_script(self):
        """Should return None if script field missing."""
        result = _parse_dsl_response('{"params": ["a"]}')
        assert result is None

    def test_rejects_missing_params(self):
        """Should return None if params field missing."""
        result = _parse_dsl_response('{"script": "a + b"}')
        assert result is None


class TestGenerateImprovedDsl:
    """Tests for LLM-based DSL generation."""

    @pytest.mark.asyncio
    async def test_generates_dsl_from_llm(self):
        """Should generate DSL using LLM client."""
        candidate = RewriteCandidate(
            signature_id=1,
            step_type="compute_sum",
            description="Add two numbers together",
            current_dsl='{"script": "a - b", "params": ["a", "b"]}',
            uses=20,
            successes=5,
            success_rate=0.25,
        )

        # Mock LLM client
        mock_client = AsyncMock()
        mock_client.generate.return_value = '{"script": "a + b", "params": ["a", "b"], "type": "math"}'

        result = await generate_improved_dsl(candidate, mock_client)

        assert result is not None
        dsl = json.loads(result)
        assert dsl["script"] == "a + b"

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self):
        """Should return None if LLM fails."""
        candidate = RewriteCandidate(
            signature_id=1,
            step_type="compute_sum",
            description="Add numbers",
            current_dsl=None,
            uses=20,
            successes=5,
            success_rate=0.25,
        )

        mock_client = AsyncMock()
        mock_client.generate.side_effect = Exception("LLM error")

        result = await generate_improved_dsl(candidate, mock_client)
        assert result is None


class TestRewriteUnderperformingDsls:
    """Integration tests for the full rewrite flow."""

    @pytest.mark.asyncio
    async def test_rewrites_underperforming_signature(self, clean_test_db):
        """Should rewrite a failing signature's DSL."""
        db = StepSignatureDB()

        # Create underperforming signature
        emb = make_embedding(500)
        sig, _ = db.find_or_create(
            step_text="Bad DSL signature",
            embedding=emb,
            parent_problem="test",
        )

        old_dsl = '{"script": "a - b", "params": ["a", "b"], "type": "math"}'
        with db._connection() as conn:
            conn.execute(
                """UPDATE step_signatures
                   SET uses = 20, successes = 4, dsl_type = 'math', dsl_script = ?
                   WHERE id = ?""",
                (old_dsl, sig.id)
            )

        # Mock LLM client
        mock_client = AsyncMock()
        mock_client.generate.return_value = '{"script": "a + b", "params": ["a", "b"], "type": "math"}'

        # Run rewriter with config override
        with patch('mycelium.step_signatures.dsl_rewriter.DSL_REWRITER_ENABLED', True), \
             patch('mycelium.step_signatures.dsl_rewriter.DSL_REWRITER_MIN_USES', 10), \
             patch('mycelium.step_signatures.dsl_rewriter.DSL_REWRITER_MAX_SUCCESS_RATE', 0.40), \
             patch('mycelium.step_signatures.dsl_rewriter.DSL_REWRITER_MIN_TRAFFIC_SHARE', 0.0):

            results = await rewrite_underperforming_dsls(db, mock_client, max_rewrites=5)

        # Verify rewrite happened
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].signature_id == sig.id

        # Verify DB updated
        updated = db.get_signature(sig.id)
        new_dsl = json.loads(updated.dsl_script)
        assert new_dsl["script"] == "a + b"

        # Verify stats reset
        assert updated.uses == 0
        assert updated.successes == 0

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self, clean_test_db):
        """Should do nothing when rewriter is disabled."""
        db = StepSignatureDB()
        mock_client = AsyncMock()

        with patch('mycelium.step_signatures.dsl_rewriter.DSL_REWRITER_ENABLED', False):
            results = await rewrite_underperforming_dsls(db, mock_client)

        assert len(results) == 0
        mock_client.generate.assert_not_called()


class TestDbHelperMethods:
    """Tests for DB helper methods used by rewriter."""

    def test_get_total_signature_uses(self, clean_test_db):
        """Should sum all signature uses."""
        db = StepSignatureDB()

        # Create signatures with different uses
        for i, uses in enumerate([10, 20, 30]):
            emb = make_embedding(600 + i)
            sig, _ = db.find_or_create(
                step_text=f"Signature {i}",
                embedding=emb,
                parent_problem="test",
            )
            with db._connection() as conn:
                conn.execute(
                    "UPDATE step_signatures SET uses = ? WHERE id = ?",
                    (uses, sig.id)
                )

        total = db.get_total_signature_uses()
        assert total == 60

    def test_reset_signature_stats(self, clean_test_db):
        """Should reset uses and successes to zero."""
        db = StepSignatureDB()

        emb = make_embedding(700)
        sig, _ = db.find_or_create(
            step_text="Reset test",
            embedding=emb,
            parent_problem="test",
        )

        # Set some stats
        with db._connection() as conn:
            conn.execute(
                "UPDATE step_signatures SET uses = 50, successes = 25 WHERE id = ?",
                (sig.id,)
            )

        # Reset
        db.reset_signature_stats(sig.id)

        # Verify
        updated = db.get_signature(sig.id)
        assert updated.uses == 0
        assert updated.successes == 0
