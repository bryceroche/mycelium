"""Tests for centroid averaging functionality."""

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


class TestCentroidAveraging:
    """Tests for running average centroid updates."""

    def test_centroid_updates_on_match(self, clean_test_db):
        """When a signature matches, its centroid should update with running average."""
        db = StepSignatureDB()

        # Create initial embedding
        emb1 = make_embedding(42)

        # Create signature
        sig, is_new = db.find_or_create(
            step_text="Calculate the sum of two numbers",
            embedding=emb1,
            parent_problem="test",
        )
        assert is_new is True
        assert sig.embedding_count == 1

        # Match with a similar embedding (small variation for high similarity)
        emb2 = make_embedding(42, variation=0.05)

        sig2, is_new2 = db.find_or_create(
            step_text="Calculate the sum of two numbers",
            embedding=emb2,
            parent_problem="test",
        )
        assert is_new2 is False
        assert sig2.signature_id == sig.signature_id

        # Verify embedding count increased
        updated_sig = db.get_signature(sig.id)
        assert updated_sig.embedding_count == 2

        # Verify centroid is the average
        expected_centroid = (emb1 + emb2) / 2
        np.testing.assert_array_almost_equal(
            updated_sig.centroid, expected_centroid, decimal=5
        )

    def test_update_centroid_method(self, clean_test_db):
        """Test the standalone update_centroid method."""
        db = StepSignatureDB()

        # Create signature
        emb1 = make_embedding(100)

        sig, _ = db.find_or_create(
            step_text="Test signature",
            embedding=emb1,
            parent_problem="test",
        )
        assert sig.embedding_count == 1

        # Manually update centroid with different embedding
        emb2 = make_embedding(200)

        db.update_centroid(sig.id, emb2, propagate_to_parents=False)

        # Verify
        updated = db.get_signature(sig.id)
        assert updated.embedding_count == 2

        expected = (emb1 + emb2) / 2
        np.testing.assert_array_almost_equal(updated.centroid, expected, decimal=5)

    def test_centroid_running_average_multiple(self, clean_test_db):
        """Test that multiple updates produce correct running average."""
        db = StepSignatureDB()

        # Create embeddings with different seeds
        embeddings = [make_embedding(seed) for seed in [10, 20, 30, 40]]

        # Create signature with first embedding
        sig, _ = db.find_or_create(
            step_text="Multi update test",
            embedding=embeddings[0],
            parent_problem="test",
        )

        # Add remaining embeddings
        for emb in embeddings[1:]:
            db.update_centroid(sig.id, emb, propagate_to_parents=False)

        # Verify final state
        updated = db.get_signature(sig.id)
        assert updated.embedding_count == 4

        expected = sum(embeddings) / 4
        np.testing.assert_array_almost_equal(updated.centroid, expected, decimal=5)


class TestParentCentroidPropagation:
    """Tests for propagating centroid changes to parent umbrellas."""

    def test_parent_centroid_updates_on_child_creation(self, clean_test_db):
        """When children are added, parent centroid should be average of children."""
        db = StepSignatureDB()

        # Create parent (umbrella) with distinct embedding
        parent_emb = make_embedding(1000)

        parent, _ = db.find_or_create(
            step_text="Parent umbrella for propagation test",
            embedding=parent_emb,
            parent_problem="test",
        )

        # Promote parent to umbrella
        db.promote_to_umbrella(parent.id)

        # Create child 1 under parent with distinct embedding
        child1_emb = make_embedding(2000)

        child1 = db.create_signature(
            step_text="Child one for propagation",
            embedding=child1_emb,
            parent_problem="test",
            parent_id=parent.id,
        )

        # Create child 2 under parent with distinct embedding
        child2_emb = make_embedding(3000)

        child2 = db.create_signature(
            step_text="Child two for propagation",
            embedding=child2_emb,
            parent_problem="test",
            parent_id=parent.id,
        )

        # Verify parent centroid is average of children (weighted by count)
        updated_parent = db.get_signature(parent.id)
        # Both children have count=1, so it's simple average
        expected = (child1_emb + child2_emb) / 2
        np.testing.assert_array_almost_equal(
            updated_parent.centroid, expected, decimal=4
        )

    def test_propagation_respects_embedding_count(self, clean_test_db):
        """Parent centroid should weight children by their embedding_count."""
        db = StepSignatureDB()

        # Create parent with distinct embedding
        parent_emb = make_embedding(4000)

        parent, _ = db.find_or_create(
            step_text="Weighted parent for propagation",
            embedding=parent_emb,
            parent_problem="test",
        )
        db.promote_to_umbrella(parent.id)

        # Create child 1 with distinct embedding
        child1_emb = make_embedding(5000)

        child1 = db.create_signature(
            step_text="Heavy child for propagation",
            embedding=child1_emb,
            parent_problem="test",
            parent_id=parent.id,
        )

        # Update child1 multiple times to increase its weight
        extra_emb = make_embedding(5001)
        db.update_centroid(child1.id, extra_emb)  # Now count=2
        db.update_centroid(child1.id, extra_emb)  # Now count=3

        # Create child 2 (count=1) with distinct embedding
        child2_emb = make_embedding(6000)

        child2 = db.create_signature(
            step_text="Light child for propagation",
            embedding=child2_emb,
            parent_problem="test",
            parent_id=parent.id,
        )

        # Get actual values
        c1 = db.get_signature(child1.id)
        c2 = db.get_signature(child2.id)
        updated_parent = db.get_signature(parent.id)

        # Parent centroid = (c1.centroid * c1.count + c2.centroid * c2.count) / total_count
        expected = (c1.centroid * c1.embedding_count + c2.centroid * c2.embedding_count) / (
            c1.embedding_count + c2.embedding_count
        )
        np.testing.assert_array_almost_equal(
            updated_parent.centroid, expected, decimal=4
        )


class TestCentroidAtomicHelper:
    """Tests for the internal _update_centroid_atomic helper."""

    def test_atomic_helper_returns_new_count(self, clean_test_db):
        """The atomic helper should return the new embedding count."""
        db = StepSignatureDB()

        emb = make_embedding(7000)

        sig, _ = db.find_or_create(
            step_text="Atomic test",
            embedding=emb,
            parent_problem="test",
        )

        # Use the atomic helper directly
        with db._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            new_count = db._update_centroid_atomic(conn, sig.id, emb)
            conn.commit()

        assert new_count == 2

    def test_atomic_helper_returns_none_for_missing(self, clean_test_db):
        """The atomic helper should return None if signature not found."""
        db = StepSignatureDB()

        emb = make_embedding(8000)

        with db._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            result = db._update_centroid_atomic(conn, 99999, emb)
            conn.commit()

        assert result is None

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
