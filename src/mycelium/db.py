"""PostgreSQL database layer for Mycelium.

Centralizes all database operations following the consolidation pattern.
Stores Welford stats, embeddings cache, and labeled spans.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

# Use environment variable for connection, with fallback to RDS
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "REDACTED_DATABASE_URL"
)


@dataclass
class WelfordRow:
    """Welford stats from database."""
    stat_type: str
    count: int
    mean: float
    m2: float
    updated_at: datetime


@dataclass
class EmbeddingRow:
    """Cached embedding from database."""
    text_hash: str
    text: str
    embedding: np.ndarray
    model: str
    created_at: datetime


@dataclass
class LabeledSpanRow:
    """Labeled span from database."""
    id: int
    span_text: str
    operation: Optional[str]
    reference_entity: Optional[str]
    cross_similarity: Optional[float]
    source: str
    created_at: datetime


@dataclass
class OperationCentroidRow:
    """Operation centroid from database."""
    operation: str
    centroid: np.ndarray
    count: int
    updated_at: datetime


@dataclass
class DecisionBoundaryRow:
    """Decision boundary from database."""
    pair_key: str
    threshold: float
    count: int
    updated_at: datetime


def _get_connection():
    """Get database connection."""
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Initialize database schema."""
    conn = _get_connection()
    cur = conn.cursor()

    # Welford stats table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS welford_stats (
            stat_type VARCHAR(50) PRIMARY KEY,
            count INTEGER NOT NULL DEFAULT 0,
            mean FLOAT NOT NULL DEFAULT 0.0,
            m2 FLOAT NOT NULL DEFAULT 0.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Embeddings cache table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            text_hash VARCHAR(64) PRIMARY KEY,
            text TEXT NOT NULL,
            embedding BYTEA NOT NULL,
            model VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create index on model for filtering
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_model
        ON embeddings(model)
    """)

    # Labeled spans table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS labeled_spans (
            id SERIAL PRIMARY KEY,
            span_text TEXT NOT NULL,
            operation VARCHAR(20),
            reference_entity VARCHAR(100),
            cross_similarity FLOAT,
            source VARCHAR(50) NOT NULL DEFAULT 'unknown',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create index on operation for filtering
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_labeled_spans_operation
        ON labeled_spans(operation)
    """)

    # Problem results table (for tracking MCTS outcomes)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS problem_results (
            id SERIAL PRIMARY KEY,
            problem_id VARCHAR(100) NOT NULL,
            problem_text TEXT,
            correct BOOLEAN NOT NULL,
            predicted_answer VARCHAR(100),
            actual_answer VARCHAR(100),
            dag_steps JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Operation centroids table (for decision boundary computation)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS operation_centroids (
            operation VARCHAR(20) PRIMARY KEY,
            centroid BYTEA NOT NULL,
            count INTEGER NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Decision boundaries table (precomputed thresholds between confusable pairs)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS decision_boundaries (
            pair_key VARCHAR(50) PRIMARY KEY,
            threshold FLOAT NOT NULL,
            count INTEGER NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database schema initialized.")


# =============================================================================
# Welford Stats Operations
# =============================================================================

def get_welford_stats(stat_type: str) -> Optional[WelfordRow]:
    """Get Welford stats for a given type."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT stat_type, count, mean, m2, updated_at FROM welford_stats WHERE stat_type = %s",
        (stat_type,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        return WelfordRow(
            stat_type=row[0],
            count=row[1],
            mean=row[2],
            m2=row[3],
            updated_at=row[4]
        )
    return None


def update_welford_stats(stat_type: str, value: float) -> WelfordRow:
    """Update Welford stats with a new value using online algorithm."""
    conn = _get_connection()
    cur = conn.cursor()

    # Get current stats
    cur.execute(
        "SELECT count, mean, m2 FROM welford_stats WHERE stat_type = %s FOR UPDATE",
        (stat_type,)
    )
    row = cur.fetchone()

    if row:
        count, mean, m2 = row[0], row[1], row[2]
    else:
        count, mean, m2 = 0, 0.0, 0.0

    # Welford's online algorithm
    count += 1
    delta = value - mean
    mean += delta / count
    delta2 = value - mean
    m2 += delta * delta2

    # Upsert
    cur.execute("""
        INSERT INTO welford_stats (stat_type, count, mean, m2, updated_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (stat_type)
        DO UPDATE SET count = %s, mean = %s, m2 = %s, updated_at = CURRENT_TIMESTAMP
        RETURNING stat_type, count, mean, m2, updated_at
    """, (stat_type, count, mean, m2, count, mean, m2))

    result = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()

    return WelfordRow(
        stat_type=result[0],
        count=result[1],
        mean=result[2],
        m2=result[3],
        updated_at=result[4]
    )


def get_all_welford_stats() -> List[WelfordRow]:
    """Get all Welford stats."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT stat_type, count, mean, m2, updated_at FROM welford_stats ORDER BY stat_type")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        WelfordRow(stat_type=r[0], count=r[1], mean=r[2], m2=r[3], updated_at=r[4])
        for r in rows
    ]


# =============================================================================
# Embeddings Cache Operations
# =============================================================================

def _text_hash(text: str) -> str:
    """Hash text for cache key."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:64]


def get_embedding(text: str, model: str) -> Optional[np.ndarray]:
    """Get cached embedding for text."""
    conn = _get_connection()
    cur = conn.cursor()

    text_hash = _text_hash(text)
    cur.execute(
        "SELECT embedding FROM embeddings WHERE text_hash = %s AND model = %s",
        (text_hash, model)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        # Deserialize numpy array from bytes
        return np.frombuffer(row[0], dtype=np.float32)
    return None


def store_embedding(text: str, embedding: np.ndarray, model: str) -> None:
    """Store embedding in cache."""
    conn = _get_connection()
    cur = conn.cursor()

    text_hash = _text_hash(text)
    embedding_bytes = embedding.astype(np.float32).tobytes()

    cur.execute("""
        INSERT INTO embeddings (text_hash, text, embedding, model)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (text_hash) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            model = EXCLUDED.model
    """, (text_hash, text, embedding_bytes, model))

    conn.commit()
    cur.close()
    conn.close()


def get_embedding_count(model: Optional[str] = None) -> int:
    """Get count of cached embeddings."""
    conn = _get_connection()
    cur = conn.cursor()

    if model:
        cur.execute("SELECT COUNT(*) FROM embeddings WHERE model = %s", (model,))
    else:
        cur.execute("SELECT COUNT(*) FROM embeddings")

    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


# =============================================================================
# Labeled Spans Operations
# =============================================================================

def add_labeled_span(
    span_text: str,
    operation: Optional[str] = None,
    reference_entity: Optional[str] = None,
    cross_similarity: Optional[float] = None,
    source: str = "unknown"
) -> int:
    """Add a labeled span and return its ID."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO labeled_spans (span_text, operation, reference_entity, cross_similarity, source)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
    """, (span_text, operation, reference_entity, cross_similarity, source))

    span_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return span_id


def get_labeled_spans(
    operation: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100
) -> List[LabeledSpanRow]:
    """Get labeled spans with optional filtering."""
    conn = _get_connection()
    cur = conn.cursor()

    query = "SELECT id, span_text, operation, reference_entity, cross_similarity, source, created_at FROM labeled_spans"
    conditions = []
    params = []

    if operation:
        conditions.append("operation = %s")
        params.append(operation)
    if source:
        conditions.append("source = %s")
        params.append(source)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += f" ORDER BY created_at DESC LIMIT {limit}"

    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        LabeledSpanRow(
            id=r[0], span_text=r[1], operation=r[2], reference_entity=r[3],
            cross_similarity=r[4], source=r[5], created_at=r[6]
        )
        for r in rows
    ]


def get_span_counts_by_operation() -> Dict[str, int]:
    """Get count of spans by operation type."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT operation, COUNT(*)
        FROM labeled_spans
        WHERE operation IS NOT NULL
        GROUP BY operation
    """)

    counts = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    conn.close()
    return counts


# =============================================================================
# Problem Results Operations
# =============================================================================

def add_problem_result(
    problem_id: str,
    correct: bool,
    predicted_answer: str,
    actual_answer: str,
    problem_text: Optional[str] = None,
    dag_steps: Optional[List[Dict]] = None
) -> int:
    """Record a problem result."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO problem_results (problem_id, problem_text, correct, predicted_answer, actual_answer, dag_steps)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (problem_id, problem_text, correct, predicted_answer, actual_answer,
          json.dumps(dag_steps) if dag_steps else None))

    result_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return result_id


def get_accuracy_stats() -> Dict[str, Any]:
    """Get overall accuracy statistics."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN correct THEN 1 ELSE 0 END) as correct,
            COUNT(DISTINCT problem_id) as unique_problems
        FROM problem_results
    """)

    row = cur.fetchone()
    cur.close()
    conn.close()

    total = row[0] or 0
    correct = row[1] or 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "unique_problems": row[2] or 0
    }


# =============================================================================
# Operation Centroids Operations
# =============================================================================

def store_centroid(operation: str, centroid: np.ndarray, count: int) -> None:
    """Store or update a centroid for an operation type."""
    conn = _get_connection()
    cur = conn.cursor()

    centroid_bytes = centroid.astype(np.float32).tobytes()

    cur.execute("""
        INSERT INTO operation_centroids (operation, centroid, count, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (operation) DO UPDATE SET
            centroid = EXCLUDED.centroid,
            count = EXCLUDED.count,
            updated_at = CURRENT_TIMESTAMP
    """, (operation, centroid_bytes, count))

    conn.commit()
    cur.close()
    conn.close()


def get_centroid(operation: str) -> Optional[np.ndarray]:
    """Get centroid for an operation type."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT centroid FROM operation_centroids WHERE operation = %s",
        (operation,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        return np.frombuffer(row[0], dtype=np.float32)
    return None


def get_all_centroids() -> Dict[str, np.ndarray]:
    """Get all operation centroids."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("SELECT operation, centroid FROM operation_centroids")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return {
        row[0]: np.frombuffer(row[1], dtype=np.float32)
        for row in rows
    }


# =============================================================================
# Decision Boundaries Operations
# =============================================================================

def store_decision_boundary(pair_key: str, threshold: float, count: int) -> None:
    """Store or update a decision boundary threshold for an operation pair."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO decision_boundaries (pair_key, threshold, count, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (pair_key) DO UPDATE SET
            threshold = EXCLUDED.threshold,
            count = EXCLUDED.count,
            updated_at = CURRENT_TIMESTAMP
    """, (pair_key, threshold, count))

    conn.commit()
    cur.close()
    conn.close()


def get_decision_boundary(pair_key: str) -> Optional[float]:
    """Get decision boundary threshold for an operation pair."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT threshold FROM decision_boundaries WHERE pair_key = %s",
        (pair_key,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        return row[0]
    return None


# =============================================================================
# Migration helpers
# =============================================================================

def migrate_json_welford(json_path: str, stat_prefix: str = "") -> int:
    """Migrate Welford stats from JSON file to database."""
    with open(json_path) as f:
        data = json.load(f)

    count = 0
    conn = _get_connection()
    cur = conn.cursor()

    for key, stats in data.items():
        stat_type = f"{stat_prefix}{key}" if stat_prefix else key
        cur.execute("""
            INSERT INTO welford_stats (stat_type, count, mean, m2)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (stat_type) DO UPDATE SET
                count = EXCLUDED.count,
                mean = EXCLUDED.mean,
                m2 = EXCLUDED.m2,
                updated_at = CURRENT_TIMESTAMP
        """, (stat_type, stats.get("count", 0), stats.get("mean", 0.0), stats.get("m2", 0.0)))
        count += 1

    conn.commit()
    cur.close()
    conn.close()
    return count


if __name__ == "__main__":
    # Test connection and initialize schema
    print(f"Connecting to: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")
    init_db()
    print("Done!")
