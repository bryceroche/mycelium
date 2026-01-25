"""Tests for centroid/embedding functionality.

NOTE: Text centroid running average updates have been removed - _update_centroid_atomic
is now a NO-OP that only updates last_used_at. Routing uses graph_embedding exclusively.

See commit 373a7e7 "Graph-first routing, remove text centroid learning".
See CLAUDE.md: "Route by what operations DO, not what they SOUND LIKE."
"""

import numpy as np
import pytest

from mycelium.step_signatures.db import StepSignatureDB


# Helper to create 768-dimensional embeddings for testing
EMBEDDING_DIM = 768


def make_embedding(seed: int, variation: float = 0.0) -> np.ndarray:
    """Create a deterministic 768-dim embedding from a seed.

    Args:
        seed: Random seed for reproducibility
        variation: Add small random variation (0.0-1.0) to create similar but different embeddings
    """
    rng = np.random.RandomState(seed)
    emb = rng.randn(EMBEDDING_DIM)
    if variation > 0:
        emb += np.random.RandomState(seed + 1000).randn(EMBEDDING_DIM) * variation
    return emb / np.linalg.norm(emb)


class TestCentroidAtomicHelper:
    """Tests for the internal _update_centroid_atomic helper.

    Note: Text centroid updates are now NO-OPs. The helper only updates last_used_at.
    Routing uses graph_embedding (computation graph embeddings).
    """

    def test_atomic_helper_returns_one_for_compatibility(self, clean_test_db):
        """The atomic helper returns 1 for API compatibility (NO-OP for centroid)."""
        db = StepSignatureDB()

        emb = make_embedding(7000)

        sig, _ = db.find_or_create(
            step_text="Atomic test",
            embedding=emb,
            parent_problem="test",
        )

        # Use the atomic helper directly - now a NO-OP that returns 1
        with db._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            result = db._update_centroid_atomic(conn, sig.id, emb)
            conn.commit()

        # Always returns 1 for API compatibility (text centroid updates removed)
        assert result == 1

    def test_atomic_helper_updates_last_used(self, clean_test_db):
        """The atomic helper should update last_used_at when flag is set."""
        db = StepSignatureDB()

        emb = make_embedding(9000)

        sig, _ = db.find_or_create(
            step_text="Last used test",
            embedding=emb,
            parent_problem="test",
        )
        original_last_used = sig.last_used_at

        # Update with last_used flag
        with db._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            db._update_centroid_atomic(conn, sig.id, emb, update_last_used=True)
            conn.commit()

        updated = db.get_signature(sig.id)
        assert updated.last_used_at is not None
        assert updated.last_used_at != original_last_used


class TestSignatureCreation:
    """Tests for signature creation with embeddings."""

    def test_signature_created_with_embedding(self, clean_test_db):
        """Signature should be created with initial embedding."""
        db = StepSignatureDB()

        emb = make_embedding(42)

        sig, is_new = db.find_or_create(
            step_text="Calculate the sum of two numbers",
            embedding=emb,
            parent_problem="test",
        )

        assert is_new is True
        assert sig.centroid is not None
        assert sig.embedding_count == 1
        # Initial centroid equals the initial embedding
        np.testing.assert_array_almost_equal(sig.centroid, emb, decimal=5)
