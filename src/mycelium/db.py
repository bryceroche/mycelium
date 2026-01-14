"""MyceliumDB: SQLite-backed database for step-level signatures."""

from datetime import datetime
from typing import Optional, List
import uuid
import os

import numpy as np

from mycelium.data_layer import get_db, init_db
from mycelium.step_signatures import StepSignature
from mycelium.step_signatures.utils import pack_embedding, unpack_embedding, cosine_similarity


class MyceliumDB:
    """SQLite-backed database for step-level signatures."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            os.environ["MYCELIUM_DB_PATH"] = db_path
        self._db = get_db()
        self._init_schema()

    def _init_schema(self):
        with self._db.connection() as conn:
            init_db(conn)

    def add_signature(
        self,
        step_type: str,
        method_name: str,
        method_template: str,
        initial_embedding: np.ndarray,
        description: str = "",
    ) -> StepSignature:
        signature_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO step_signatures (
                    signature_id, centroid, step_type, description,
                    method_name, method_template, example_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signature_id,
                    pack_embedding(initial_embedding),
                    step_type,
                    description,
                    method_name,
                    method_template,
                    1,
                    now,
                ),
            )
            row_id = cursor.lastrowid

        return StepSignature(
            id=row_id,
            signature_id=signature_id,
            centroid=initial_embedding,
            step_type=step_type,
            description=description,
            method_name=method_name,
            method_template=method_template,
            example_count=1,
            created_at=now,
        )

    def find_similar(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
        limit: int = 10,
    ) -> List[tuple]:
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM step_signatures")
            rows = cursor.fetchall()

        results = []
        for row in rows:
            centroid = unpack_embedding(row["centroid"])
            if centroid is not None:
                sim = cosine_similarity(embedding, centroid)
                if sim >= threshold:
                    sig = self._row_to_signature(row)
                    results.append((sig, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_signature(self, signature_id: str) -> Optional[StepSignature]:
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM step_signatures WHERE signature_id = ?",
                (signature_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None
        return self._row_to_signature(row)

    def update_stats(self, signature_id: str, success: bool) -> None:
        now = datetime.utcnow().isoformat()
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE step_signatures
                SET uses = uses + 1,
                    successes = successes + ?,
                    last_used_at = ?
                WHERE signature_id = ?
                """,
                (1 if success else 0, now, signature_id),
            )

    def get_stats(self) -> dict:
        with self._db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM step_signatures")
            sig_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM step_examples")
            example_count = cursor.fetchone()[0]

            cursor.execute("SELECT SUM(uses), SUM(successes) FROM step_signatures")
            row = cursor.fetchone()
            total_uses = row[0] or 0
            total_successes = row[1] or 0

        return {
            "signatures": sig_count,
            "examples": example_count,
            "total_uses": total_uses,
            "total_successes": total_successes,
            "success_rate": total_successes / total_uses if total_uses > 0 else 0.0,
        }

    def _row_to_signature(self, row) -> StepSignature:
        return StepSignature(
            id=row["id"],
            signature_id=row["signature_id"],
            centroid=unpack_embedding(row["centroid"]),
            step_type=row["step_type"],
            description=row["description"] or "",
            method_name=row["method_name"],
            method_template=row["method_template"],
            example_count=row["example_count"] or 0,
            uses=row["uses"] or 0,
            successes=row["successes"] or 0,
            cohesion=row["cohesion"] or 0.0,
            amplitude=row["amplitude"] or 0.1,
            phase=row["phase"] or 0.0,
            spread=row["spread"] or 0.3,
            is_canonical=bool(row["is_canonical"]),
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
        )


DB = MyceliumDB
