"""StepSignatureDB - Minimal implementation for local decomposition architecture.

Core functionality only:
- Store/retrieve signatures
- Record success/failure stats
- Route steps to signatures via embedding similarity
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np

from mycelium.config import EMBEDDING_DIM, DB_PATH
from mycelium.data_layer import get_db
from mycelium.data_layer.schema import init_db
from mycelium.step_signatures.models import StepSignature
from mycelium.step_signatures.utils import (
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
)

logger = logging.getLogger(__name__)


def normalize_step_text(text: str) -> str:
    """Normalize step text by replacing numbers with N."""
    return re.sub(r'\b\d+\.?\d*\b', 'N', text)


@dataclass
class RoutingResult:
    """Result of routing a step to a signature."""
    signature: Optional[StepSignature]
    similarity: float
    path: List[int] = None

    def __post_init__(self):
        if self.path is None:
            self.path = []

    @property
    def is_match(self) -> bool:
        return self.signature is not None


class StepSignatureDB:
    """Minimal signature database for local decomposition."""

    def __init__(self, db_path: str = None, embedder=None):
        """Initialize database."""
        self.db_path = db_path or DB_PATH
        self._embedder = embedder
        with get_db().connection() as conn:
            init_db(conn)

    def _connection(self):
        """Get database connection manager."""
        return get_db()

    # =========================================================================
    # CORE QUERIES
    # =========================================================================

    def count_signatures(self) -> int:
        """Count total signatures."""
        conn = self._connection()
        row = conn.execute("SELECT COUNT(*) FROM step_signatures").fetchone()
        return row[0] if row else 0

    def get_signature_count(self) -> int:
        """Alias for count_signatures."""
        return self.count_signatures()

    def get_signature(self, signature_id: int) -> Optional[StepSignature]:
        """Get signature by ID."""
        conn = self._connection()
        row = conn.execute(
            "SELECT * FROM step_signatures WHERE id = ?",
            (signature_id,),
        ).fetchone()

        if row is None:
            return None

        return StepSignature.from_row(dict(row))

    def get_all_leaves(self) -> List[StepSignature]:
        """Get all leaf signatures (non-umbrellas)."""
        conn = self._connection()
        rows = conn.execute(
            """
            SELECT * FROM step_signatures
            WHERE is_semantic_umbrella = 0 OR is_semantic_umbrella IS NULL
            """
        ).fetchall()

        return [StepSignature.from_row_for_routing(dict(row)) for row in rows]

    def get_all_signatures(self) -> List[StepSignature]:
        """Get all signatures."""
        conn = self._connection()
        rows = conn.execute("SELECT * FROM step_signatures").fetchall()
        return [StepSignature.from_row(dict(row)) for row in rows]

    # =========================================================================
    # STATS RECORDING
    # =========================================================================

    def record_success(self, signature_id: int) -> None:
        """Record a successful execution."""
        conn = self._connection()
        conn.execute(
            """
            UPDATE step_signatures
            SET successes = COALESCE(successes, 0) + 1,
                uses = COALESCE(uses, 0) + 1
            WHERE id = ?
            """,
            (signature_id,),
        )

    def record_failure(self, signature_id: int) -> None:
        """Record a failed execution."""
        conn = self._connection()
        conn.execute(
            """
            UPDATE step_signatures
            SET operational_failures = COALESCE(operational_failures, 0) + 1,
                uses = COALESCE(uses, 0) + 1
            WHERE id = ?
            """,
            (signature_id,),
        )

    # =========================================================================
    # SIGNATURE CREATION
    # =========================================================================

    def find_or_create(
        self,
        step_text: str,
        embedding: List[float],
        dsl_hint: str = None,
        **kwargs,
    ) -> tuple[StepSignature, bool]:
        """Find existing signature or create new one.

        Args:
            step_text: Description of the step
            embedding: Step embedding vector
            dsl_hint: Optional DSL hint (+, -, *, /)

        Returns:
            (signature, created) tuple
        """
        # Find best match among leaves
        leaves = self.get_all_leaves()
        best_sig = None
        best_sim = 0.0

        for sig in leaves:
            if sig.centroid is None:
                continue
            sim = cosine_similarity(embedding, sig.centroid)
            if sim > best_sim:
                best_sim = sim
                best_sig = sig

        # If good match found, return it
        if best_sig and best_sim >= 0.85:
            return best_sig, False

        # Create new signature
        conn = self._connection()
        step_type = normalize_step_text(step_text)[:100]
        sig_id = str(uuid.uuid4())

        # Infer DSL from hint
        dsl_script = None
        if dsl_hint:
            dsl_map = {"+": "a + b", "-": "a - b", "*": "a * b", "/": "a / b"}
            dsl_script = dsl_map.get(dsl_hint, f"a {dsl_hint} b")

        now = datetime.now(timezone.utc).isoformat()

        with conn.connection() as raw_conn:
            cursor = raw_conn.execute(
                """
                INSERT INTO step_signatures (
                    signature_id, step_type, description, dsl_script,
                    centroid, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sig_id, step_type, step_text, dsl_script, pack_embedding(embedding), now),
            )
            last_id = cursor.lastrowid

        new_sig = self.get_signature(last_id)
        return new_sig, True

    # =========================================================================
    # DATA MANAGEMENT
    # =========================================================================

    def clear_all_data(self, force: bool = False) -> dict:
        """Clear all signature data."""
        if not force:
            return {"error": "Use force=True to confirm"}

        conn = self._connection()
        with conn.connection() as raw_conn:
            raw_conn.execute("DELETE FROM step_signatures")
            raw_conn.execute("DELETE FROM signature_relationships")

        return {"cleared": True}


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_step_db: Optional[StepSignatureDB] = None


def get_step_db() -> StepSignatureDB:
    """Get the singleton StepSignatureDB instance."""
    global _step_db
    if _step_db is None:
        _step_db = StepSignatureDB()
    return _step_db


def reset_step_db() -> None:
    """Reset the singleton instance."""
    global _step_db
    _step_db = None
