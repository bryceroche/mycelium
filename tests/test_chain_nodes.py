"""Tests for Chain Node detection and creation.

Per CLAUDE.md Big 5 #5 (Primitive vs Chain Nodes):
- Track successful step sequences
- Create chain nodes when threshold is met
"""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone

import pytest

from mycelium.chain_nodes import (
    ChainNode,
    SequenceStats,
    record_sequence_outcome,
    check_for_chain_creation,
    get_sequence_stats,
    get_all_sequences,
    get_chain_candidates,
    _hash_sequence,
)
from mycelium.data_layer import create_connection_manager
from mycelium.data_layer.schema import init_db


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = create_connection_manager(db_path)
    with conn.connection() as raw_conn:
        init_db(raw_conn)

    yield conn

    # Cleanup
    conn.close()
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def test_db_with_signatures(test_db):
    """Create a test database with some leaf signatures."""
    with test_db.connection() as raw_conn:
        # Insert some test leaf signatures
        now = datetime.now(timezone.utc).isoformat()
        signatures = [
            ("sig_add", "add", "Add two numbers"),
            ("sig_sub", "subtract", "Subtract two numbers"),
            ("sig_mul", "multiply", "Multiply two numbers"),
            ("sig_div", "divide", "Divide two numbers"),
        ]
        for sig_id, step_type, description in signatures:
            raw_conn.execute(
                """
                INSERT INTO step_signatures (signature_id, step_type, description, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (sig_id, step_type, description, now)
            )

    return test_db


class TestSequenceHashing:
    """Tests for sequence hashing."""

    def test_same_sequence_same_hash(self):
        """Same sequence should produce same hash."""
        seq = ["sig_add", "sig_div"]
        hash1 = _hash_sequence(seq)
        hash2 = _hash_sequence(seq)
        assert hash1 == hash2

    def test_different_order_different_hash(self):
        """Order matters - different order should produce different hash."""
        seq1 = ["sig_add", "sig_div"]
        seq2 = ["sig_div", "sig_add"]
        hash1 = _hash_sequence(seq1)
        hash2 = _hash_sequence(seq2)
        assert hash1 != hash2

    def test_different_sequences_different_hash(self):
        """Different sequences should produce different hashes."""
        seq1 = ["sig_add", "sig_div"]
        seq2 = ["sig_add", "sig_mul"]
        hash1 = _hash_sequence(seq1)
        hash2 = _hash_sequence(seq2)
        assert hash1 != hash2

    def test_hash_length(self):
        """Hash should be 32 characters (truncated SHA256)."""
        seq = ["sig_add", "sig_div"]
        hash1 = _hash_sequence(seq)
        assert len(hash1) == 32


class TestSequenceTracking:
    """Tests for sequence outcome tracking."""

    def test_record_success(self, test_db):
        """Recording a success should increment success count."""
        sequence = ["sig_add", "sig_div"]
        stats = record_sequence_outcome(sequence, success=True, db_conn=test_db)

        assert stats is not None
        assert stats.success_count == 1
        assert stats.failure_count == 0
        assert stats.sequence == sequence

    def test_record_failure(self, test_db):
        """Recording a failure should increment failure count."""
        sequence = ["sig_add", "sig_div"]
        stats = record_sequence_outcome(sequence, success=False, db_conn=test_db)

        assert stats is not None
        assert stats.success_count == 0
        assert stats.failure_count == 1

    def test_record_multiple_outcomes(self, test_db):
        """Multiple outcomes should accumulate correctly."""
        sequence = ["sig_add", "sig_div"]

        # Record several successes and failures
        for _ in range(5):
            record_sequence_outcome(sequence, success=True, db_conn=test_db)
        for _ in range(2):
            record_sequence_outcome(sequence, success=False, db_conn=test_db)

        stats = get_sequence_stats(sequence, db_conn=test_db)
        assert stats.success_count == 5
        assert stats.failure_count == 2
        assert stats.total_attempts == 7
        assert stats.success_rate == pytest.approx(5 / 7)

    def test_single_step_ignored(self, test_db):
        """Single-step sequences should be ignored (not chains)."""
        sequence = ["sig_add"]
        stats = record_sequence_outcome(sequence, success=True, db_conn=test_db)
        assert stats is None

    def test_empty_sequence_ignored(self, test_db):
        """Empty sequences should be ignored."""
        stats = record_sequence_outcome([], success=True, db_conn=test_db)
        assert stats is None

    def test_different_sequences_tracked_separately(self, test_db):
        """Different sequences should have separate stats."""
        seq1 = ["sig_add", "sig_div"]
        seq2 = ["sig_mul", "sig_sub"]

        record_sequence_outcome(seq1, success=True, db_conn=test_db)
        record_sequence_outcome(seq1, success=True, db_conn=test_db)
        record_sequence_outcome(seq2, success=False, db_conn=test_db)

        stats1 = get_sequence_stats(seq1, db_conn=test_db)
        stats2 = get_sequence_stats(seq2, db_conn=test_db)

        assert stats1.success_count == 2
        assert stats2.success_count == 0
        assert stats2.failure_count == 1


class TestChainCreationThreshold:
    """Tests for chain creation threshold logic."""

    def test_chain_created_at_threshold(self, test_db_with_signatures):
        """Chain should be created when success count reaches threshold."""
        sequence = ["sig_add", "sig_div"]
        db = test_db_with_signatures

        # Record enough successes
        for _ in range(5):
            record_sequence_outcome(sequence, success=True, db_conn=db)

        # Check for chain creation with threshold=5
        chain = check_for_chain_creation(
            sequence, db_conn=db, threshold=5, min_success_rate=0.8
        )

        assert chain is not None
        assert chain.sequence == sequence
        assert "Chain:" in chain.description
        assert chain.db_id is not None

    def test_chain_not_created_below_threshold(self, test_db_with_signatures):
        """Chain should not be created before reaching threshold."""
        sequence = ["sig_add", "sig_div"]
        db = test_db_with_signatures

        # Record only 4 successes (below threshold of 5)
        for _ in range(4):
            record_sequence_outcome(sequence, success=True, db_conn=db)

        # Check for chain creation
        chain = check_for_chain_creation(
            sequence, db_conn=db, threshold=5, min_success_rate=0.8
        )

        assert chain is None

    def test_chain_not_created_low_success_rate(self, test_db_with_signatures):
        """Chain should not be created with low success rate."""
        sequence = ["sig_add", "sig_div"]
        db = test_db_with_signatures

        # Record 5 successes and 5 failures (50% success rate)
        for _ in range(5):
            record_sequence_outcome(sequence, success=True, db_conn=db)
        for _ in range(5):
            record_sequence_outcome(sequence, success=False, db_conn=db)

        # Check for chain creation (80% threshold not met)
        chain = check_for_chain_creation(
            sequence, db_conn=db, threshold=5, min_success_rate=0.8
        )

        assert chain is None

    def test_chain_created_with_high_success_rate(self, test_db_with_signatures):
        """Chain should be created with sufficient success rate."""
        sequence = ["sig_add", "sig_div"]
        db = test_db_with_signatures

        # Record 8 successes and 1 failure (88.9% success rate)
        for _ in range(8):
            record_sequence_outcome(sequence, success=True, db_conn=db)
        record_sequence_outcome(sequence, success=False, db_conn=db)

        # Check for chain creation (80% threshold met)
        chain = check_for_chain_creation(
            sequence, db_conn=db, threshold=5, min_success_rate=0.8
        )

        assert chain is not None

    def test_chain_not_recreated(self, test_db_with_signatures):
        """Once chain is created, it should not be recreated."""
        sequence = ["sig_add", "sig_div"]
        db = test_db_with_signatures

        # Record enough successes
        for _ in range(5):
            record_sequence_outcome(sequence, success=True, db_conn=db)

        # Create chain
        chain1 = check_for_chain_creation(
            sequence, db_conn=db, threshold=5, min_success_rate=0.8
        )
        assert chain1 is not None

        # Record more successes
        for _ in range(5):
            record_sequence_outcome(sequence, success=True, db_conn=db)

        # Check again - should return existing chain
        chain2 = check_for_chain_creation(
            sequence, db_conn=db, threshold=5, min_success_rate=0.8
        )

        assert chain2 is not None
        assert chain2.id == chain1.id
        assert chain2.db_id == chain1.db_id


class TestChainCandidates:
    """Tests for finding chain candidates."""

    def test_get_chain_candidates(self, test_db):
        """Should return sequences that meet thresholds but aren't chains yet."""
        seq1 = ["sig_add", "sig_div"]
        seq2 = ["sig_mul", "sig_sub"]
        seq3 = ["sig_add", "sig_mul"]

        # seq1: 6 successes, 1 failure (85.7% rate)
        for _ in range(6):
            record_sequence_outcome(seq1, success=True, db_conn=test_db)
        record_sequence_outcome(seq1, success=False, db_conn=test_db)

        # seq2: 3 successes (below threshold)
        for _ in range(3):
            record_sequence_outcome(seq2, success=True, db_conn=test_db)

        # seq3: 5 successes, 3 failures (62.5% rate, below min rate)
        for _ in range(5):
            record_sequence_outcome(seq3, success=True, db_conn=test_db)
        for _ in range(3):
            record_sequence_outcome(seq3, success=False, db_conn=test_db)

        candidates = get_chain_candidates(
            min_successes=5, min_success_rate=0.8, db_conn=test_db
        )

        # Only seq1 should be a candidate
        assert len(candidates) == 1
        assert candidates[0].sequence == seq1

    def test_get_all_sequences(self, test_db):
        """Should return all tracked sequences."""
        seq1 = ["sig_add", "sig_div"]
        seq2 = ["sig_mul", "sig_sub"]

        record_sequence_outcome(seq1, success=True, db_conn=test_db)
        record_sequence_outcome(seq2, success=False, db_conn=test_db)

        all_seqs = get_all_sequences(db_conn=test_db)

        assert len(all_seqs) == 2
        sequences = [s.sequence for s in all_seqs]
        assert seq1 in sequences
        assert seq2 in sequences


class TestSequenceStatsProperties:
    """Tests for SequenceStats dataclass properties."""

    def test_total_attempts(self):
        """total_attempts should sum success and failure counts."""
        stats = SequenceStats(
            id=1,
            sequence=["a", "b"],
            success_count=5,
            failure_count=3,
            chain_node_id=None,
            created_at="2024-01-01T00:00:00Z",
        )
        assert stats.total_attempts == 8

    def test_success_rate(self):
        """success_rate should be success_count / total_attempts."""
        stats = SequenceStats(
            id=1,
            sequence=["a", "b"],
            success_count=8,
            failure_count=2,
            chain_node_id=None,
            created_at="2024-01-01T00:00:00Z",
        )
        assert stats.success_rate == pytest.approx(0.8)

    def test_success_rate_zero_attempts(self):
        """success_rate should be 0 with no attempts."""
        stats = SequenceStats(
            id=1,
            sequence=["a", "b"],
            success_count=0,
            failure_count=0,
            chain_node_id=None,
            created_at="2024-01-01T00:00:00Z",
        )
        assert stats.success_rate == 0.0

    def test_has_chain(self):
        """has_chain should reflect chain_node_id presence."""
        stats_no_chain = SequenceStats(
            id=1,
            sequence=["a", "b"],
            success_count=5,
            failure_count=0,
            chain_node_id=None,
            created_at="2024-01-01T00:00:00Z",
        )
        assert stats_no_chain.has_chain is False

        stats_with_chain = SequenceStats(
            id=1,
            sequence=["a", "b"],
            success_count=5,
            failure_count=0,
            chain_node_id=42,
            created_at="2024-01-01T00:00:00Z",
        )
        assert stats_with_chain.has_chain is True


class TestChainNodeDataclass:
    """Tests for ChainNode dataclass."""

    def test_chain_node_creation(self):
        """ChainNode should store sequence and description."""
        chain = ChainNode(
            id="chain_abc123",
            sequence=["sig_add", "sig_div"],
            description="Chain: add -> divide",
        )
        assert chain.id == "chain_abc123"
        assert chain.sequence == ["sig_add", "sig_div"]
        assert chain.description == "Chain: add -> divide"
        assert chain.embedding is None
        assert chain.db_id is None

    def test_chain_node_with_optional_fields(self):
        """ChainNode should accept optional fields."""
        chain = ChainNode(
            id="chain_abc123",
            sequence=["sig_add", "sig_div"],
            description="Chain: add -> divide",
            embedding=[0.1, 0.2, 0.3],
            db_id=42,
        )
        assert chain.embedding == [0.1, 0.2, 0.3]
        assert chain.db_id == 42
