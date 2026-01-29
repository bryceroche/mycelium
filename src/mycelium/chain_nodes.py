"""Chain Node Detection and Creation.

Per CLAUDE.md Big 5 #5 (Primitive vs Chain Nodes):
1. Cold start bootstraps primitives (ADD, SUB, MUL, DIV)
2. Success patterns reveal which primitives chain together
3. Chains become new "atomic" units for L5 routing

This module tracks successful step sequences and creates chain nodes
when a sequence meets the success threshold.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List
import uuid

from mycelium.config import CHAIN_CREATION_THRESHOLD, CHAIN_MIN_SUCCESS_RATE
from mycelium.data_layer import get_db

logger = logging.getLogger(__name__)


@dataclass
class ChainNode:
    """Represents a chain of primitives that can be matched directly.

    Chain nodes emerge from successful sequences of leaf nodes.
    Once created, they can be routed to directly for similar problems.
    """
    id: str
    sequence: List[str]  # Ordered list of leaf_node IDs (signature_ids)
    description: str     # e.g., 'average two numbers' = [ADD, DIV]
    embedding: Optional[List[float]] = None
    db_id: Optional[int] = None  # ID in step_signatures table if chain created


@dataclass
class SequenceStats:
    """Statistics for a step sequence."""
    id: int
    sequence: List[str]
    success_count: int
    failure_count: int
    chain_node_id: Optional[int]
    created_at: str

    @property
    def total_attempts(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.success_count / self.total_attempts

    @property
    def has_chain(self) -> bool:
        return self.chain_node_id is not None


def _hash_sequence(sequence: List[str]) -> str:
    """Create a deterministic hash for a sequence of leaf node IDs."""
    # Sort-insensitive: order matters for sequences
    seq_str = json.dumps(sequence, sort_keys=False)
    return hashlib.sha256(seq_str.encode()).hexdigest()[:32]


def record_sequence_outcome(
    sequence: List[str],
    success: bool,
    db_conn=None,
) -> Optional[SequenceStats]:
    """Record success/failure for a step sequence.

    Per CLAUDE.md "New Favorite Pattern": All DB operations through data layer.

    Args:
        sequence: Ordered list of leaf_node signature_ids used in solving a problem
        success: Whether the problem was solved successfully
        db_conn: Optional database connection (uses get_db() if None)

    Returns:
        SequenceStats for the recorded sequence
    """
    if not sequence or len(sequence) < 2:
        # Single-step sequences aren't chains
        logger.debug("[chain_nodes] Skipping single-step sequence")
        return None

    conn = db_conn or get_db()
    seq_hash = _hash_sequence(sequence)
    seq_json = json.dumps(sequence)
    now = datetime.now(timezone.utc).isoformat()

    # Try to update existing sequence
    with conn.connection() as raw_conn:
        existing = raw_conn.execute(
            "SELECT id, success_count, failure_count, chain_node_id FROM step_sequences WHERE sequence_hash = ?",
            (seq_hash,)
        ).fetchone()

        if existing:
            # Update existing record
            if success:
                raw_conn.execute(
                    "UPDATE step_sequences SET success_count = success_count + 1 WHERE id = ?",
                    (existing["id"],)
                )
            else:
                raw_conn.execute(
                    "UPDATE step_sequences SET failure_count = failure_count + 1 WHERE id = ?",
                    (existing["id"],)
                )

            # Fetch updated stats
            updated = raw_conn.execute(
                "SELECT * FROM step_sequences WHERE id = ?",
                (existing["id"],)
            ).fetchone()
        else:
            # Insert new sequence
            cursor = raw_conn.execute(
                """
                INSERT INTO step_sequences (sequence, sequence_hash, success_count, failure_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (seq_json, seq_hash, 1 if success else 0, 0 if success else 1, now)
            )
            updated = raw_conn.execute(
                "SELECT * FROM step_sequences WHERE id = ?",
                (cursor.lastrowid,)
            ).fetchone()

    if updated:
        return SequenceStats(
            id=updated["id"],
            sequence=json.loads(updated["sequence"]),
            success_count=updated["success_count"],
            failure_count=updated["failure_count"],
            chain_node_id=updated["chain_node_id"],
            created_at=updated["created_at"],
        )
    return None


def check_for_chain_creation(
    sequence: List[str],
    db_conn=None,
    threshold: int = None,
    min_success_rate: float = None,
) -> Optional[ChainNode]:
    """Check if sequence meets threshold for chain creation.

    Per CLAUDE.md "The Flow": DB Stats -> Welford -> Tree Structure
    Chain creation is driven by accumulated success statistics.

    Args:
        sequence: Ordered list of leaf_node signature_ids
        db_conn: Optional database connection
        threshold: Override CHAIN_CREATION_THRESHOLD
        min_success_rate: Override CHAIN_MIN_SUCCESS_RATE

    Returns:
        ChainNode if created, None otherwise
    """
    if not sequence or len(sequence) < 2:
        return None

    conn = db_conn or get_db()
    seq_hash = _hash_sequence(sequence)
    threshold = threshold if threshold is not None else CHAIN_CREATION_THRESHOLD
    min_rate = min_success_rate if min_success_rate is not None else CHAIN_MIN_SUCCESS_RATE

    with conn.connection() as raw_conn:
        row = raw_conn.execute(
            "SELECT * FROM step_sequences WHERE sequence_hash = ?",
            (seq_hash,)
        ).fetchone()

        if not row:
            logger.debug("[chain_nodes] No sequence record found for hash %s", seq_hash[:8])
            return None

        # Check if chain already exists
        if row["chain_node_id"] is not None:
            logger.debug("[chain_nodes] Chain already exists for sequence")
            # Return existing chain info
            sig_row = raw_conn.execute(
                "SELECT signature_id, description FROM step_signatures WHERE id = ?",
                (row["chain_node_id"],)
            ).fetchone()
            if sig_row:
                return ChainNode(
                    id=sig_row["signature_id"],
                    sequence=json.loads(row["sequence"]),
                    description=sig_row["description"],
                    db_id=row["chain_node_id"],
                )
            return None

        # Check thresholds
        success_count = row["success_count"]
        failure_count = row["failure_count"]
        total = success_count + failure_count
        success_rate = success_count / total if total > 0 else 0.0

        if success_count < threshold:
            logger.debug(
                "[chain_nodes] Sequence has %d successes, need %d",
                success_count, threshold
            )
            return None

        if success_rate < min_rate:
            logger.debug(
                "[chain_nodes] Sequence success rate %.2f < threshold %.2f",
                success_rate, min_rate
            )
            return None

        # Create chain node!
        chain = _create_chain_node(sequence, raw_conn, row["id"])
        return chain


def _create_chain_node(
    sequence: List[str],
    raw_conn,
    sequence_db_id: int,
) -> Optional[ChainNode]:
    """Create a chain node in the signature database.

    Per CLAUDE.md "New Favorite Pattern": Consolidated signature creation.

    Args:
        sequence: Ordered list of leaf_node signature_ids
        raw_conn: Raw database connection (inside transaction)
        sequence_db_id: ID of the step_sequences record

    Returns:
        ChainNode if created successfully
    """
    # Look up the leaf nodes to build description
    placeholders = ",".join("?" * len(sequence))
    leaf_rows = raw_conn.execute(
        f"SELECT signature_id, step_type, description FROM step_signatures WHERE signature_id IN ({placeholders})",
        tuple(sequence)
    ).fetchall()

    # Build description from sequence
    # sqlite3.Row supports indexing but not .get(), so use dict()
    leaf_map = {row["signature_id"]: dict(row) for row in leaf_rows}
    step_types = [leaf_map.get(sig_id, {}).get("step_type", "?") for sig_id in sequence]
    description = f"Chain: {' -> '.join(step_types)}"

    # Generate new signature
    chain_sig_id = f"chain_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()

    # Insert into step_signatures as a chain node
    cursor = raw_conn.execute(
        """
        INSERT INTO step_signatures (
            signature_id, step_type, description, dsl_type,
            is_atomic, atomic_reason, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chain_sig_id,
            "chain",  # step_type for chains
            description,
            "chain",  # dsl_type
            1,  # is_atomic = True (chains are atomic units now)
            f"chain_from_sequence:{sequence_db_id}",
            now,
        )
    )
    chain_db_id = cursor.lastrowid

    # Update step_sequences to link to chain node
    raw_conn.execute(
        "UPDATE step_sequences SET chain_node_id = ? WHERE id = ?",
        (chain_db_id, sequence_db_id)
    )

    logger.info(
        "[chain_nodes] Created chain node %s from sequence %s",
        chain_sig_id, sequence
    )

    return ChainNode(
        id=chain_sig_id,
        sequence=sequence,
        description=description,
        db_id=chain_db_id,
    )


def get_sequence_stats(
    sequence: List[str],
    db_conn=None,
) -> Optional[SequenceStats]:
    """Get statistics for a step sequence without modifying them.

    Args:
        sequence: Ordered list of leaf_node signature_ids
        db_conn: Optional database connection

    Returns:
        SequenceStats if found, None otherwise
    """
    if not sequence or len(sequence) < 2:
        return None

    conn = db_conn or get_db()
    seq_hash = _hash_sequence(sequence)

    row = conn.fetchone(
        "SELECT * FROM step_sequences WHERE sequence_hash = ?",
        (seq_hash,)
    )

    if not row:
        return None

    return SequenceStats(
        id=row["id"],
        sequence=json.loads(row["sequence"]),
        success_count=row["success_count"],
        failure_count=row["failure_count"],
        chain_node_id=row["chain_node_id"],
        created_at=row["created_at"],
    )


def get_all_sequences(db_conn=None) -> List[SequenceStats]:
    """Get all tracked sequences.

    Args:
        db_conn: Optional database connection

    Returns:
        List of SequenceStats for all tracked sequences
    """
    conn = db_conn or get_db()
    rows = conn.fetchall("SELECT * FROM step_sequences ORDER BY success_count DESC")

    return [
        SequenceStats(
            id=row["id"],
            sequence=json.loads(row["sequence"]),
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            chain_node_id=row["chain_node_id"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def get_chain_candidates(
    min_successes: int = None,
    min_success_rate: float = None,
    db_conn=None,
) -> List[SequenceStats]:
    """Get sequences that are candidates for chain creation.

    Returns sequences that meet thresholds but haven't been converted to chains yet.

    Args:
        min_successes: Minimum success count (default: CHAIN_CREATION_THRESHOLD)
        min_success_rate: Minimum success rate (default: CHAIN_MIN_SUCCESS_RATE)
        db_conn: Optional database connection

    Returns:
        List of SequenceStats that could become chains
    """
    conn = db_conn or get_db()
    min_successes = min_successes if min_successes is not None else CHAIN_CREATION_THRESHOLD
    min_rate = min_success_rate if min_success_rate is not None else CHAIN_MIN_SUCCESS_RATE

    rows = conn.fetchall(
        """
        SELECT * FROM step_sequences
        WHERE chain_node_id IS NULL
          AND success_count >= ?
          AND (CAST(success_count AS REAL) / (success_count + failure_count)) >= ?
        ORDER BY success_count DESC
        """,
        (min_successes, min_rate)
    )

    return [
        SequenceStats(
            id=row["id"],
            sequence=json.loads(row["sequence"]),
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            chain_node_id=row["chain_node_id"],
            created_at=row["created_at"],
        )
        for row in rows
    ]
