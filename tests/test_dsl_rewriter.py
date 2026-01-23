"""Integration tests for DSL auto-rewriter."""

import json
import numpy as np
import pytest
from unittest.mock import AsyncMock

from mycelium.step_signatures.db import StepSignatureDB
from mycelium.step_signatures.dsl_rewriter import (
    RewriteCandidate,
    generate_improved_dsl,
    _parse_dsl_response,
)


EMBEDDING_DIM = 768


def make_embedding(seed: int) -> np.ndarray:
    """Create a deterministic 768-dim embedding from a seed."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(EMBEDDING_DIM)
    return emb / np.linalg.norm(emb)


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
