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

# Import consolidated Welford implementation for in-memory stats
# NOTE: This module provides database persistence for Welford stats.
# For in-memory Welford calculations, use mycelium.welford.WelfordStats.
# The update_welford_stats() function below uses the same algorithm but
# persists directly to the database. Consider refactoring to use
# WelfordStats.from_db_row() and WelfordStats.to_dict() for consistency.
from mycelium.welford import WelfordStats

# Use environment variable for connection (required)
DATABASE_URL = os.environ.get("DATABASE_URL")


@dataclass
class WelfordRow:
    """Welford stats from database.

    For in-memory operations, use WelfordStats from mycelium.welford instead.
    This class represents the database row structure.
    """
    stat_type: str
    count: int
    mean: float
    m2: float
    updated_at: datetime

    def to_welford_stats(self) -> WelfordStats:
        """Convert to in-memory WelfordStats instance."""
        return WelfordStats.from_db_row(self.count, self.mean, self.m2)


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
class SpanTemplateRow:
    """Span template from database.

    Templates are clustered span patterns with centroids that serve as
    "gold standard" anchors for KNN classification. Each template represents
    a normalized pattern (e.g., "[NAME] sold [N] [ITEM]") with an operation.
    """
    template_id: str
    pattern: str
    centroid: np.ndarray  # 384-dim embedding
    operation: str  # SET, ADD, SUB, MUL, DIV
    dsl_type: str  # "simple" or "complex"
    examples: List[str]  # Original span texts in this cluster
    count: int  # Number of spans in cluster
    welford_count: int  # Match count for this template
    welford_mean: float  # Mean match similarity
    welford_m2: float  # Variance component
    created_at: datetime
    updated_at: datetime


def _get_connection():
    """Get database connection."""
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Initialize database schema."""
    conn = _get_connection()
    cur = conn.cursor()

    # Enable pgvector extension for vector similarity search
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()  # Commit extension separately to ensure it's available

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

    # Span templates table (clustered patterns with centroids)
    # Templates serve as "gold standard" anchors for two-tier KNN
    # Uses pgvector for DB-side similarity search (instant startup!)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS span_templates (
            template_id VARCHAR(100) PRIMARY KEY,
            pattern TEXT NOT NULL,
            centroid BYTEA NOT NULL,
            centroid_vec vector(384),
            operation VARCHAR(20) NOT NULL,
            dsl_type VARCHAR(20) DEFAULT 'simple',
            examples JSONB,
            count INTEGER NOT NULL DEFAULT 0,
            welford_count INTEGER DEFAULT 0,
            welford_mean FLOAT DEFAULT 0.0,
            welford_m2 FLOAT DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Index for filtering templates by operation
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_span_templates_operation
        ON span_templates(operation)
    """)

    # pgvector HNSW index for fast KNN search
    # Uses cosine distance (<=>); HNSW works well for any dataset size
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_span_templates_centroid_vec
        ON span_templates USING hnsw (centroid_vec vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database schema initialized.")


# =============================================================================
# Welford Stats Operations
# =============================================================================

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


# =============================================================================
# Labeled Spans Operations
# =============================================================================

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


# =============================================================================
# Span Templates Operations (Two-Tier KNN)
# =============================================================================

def _numpy_to_pgvector(arr: np.ndarray) -> str:
    """Convert numpy array to pgvector string format."""
    return '[' + ','.join(str(float(x)) for x in arr.astype(np.float32)) + ']'


def store_span_template(
    template_id: str,
    pattern: str,
    centroid: np.ndarray,
    operation: str,
    dsl_type: str = "simple",
    examples: Optional[List[str]] = None,
    count: int = 0
) -> None:
    """Store or update a span template.

    Templates are clustered span patterns that serve as "gold standard"
    anchors for KNN classification. They carry 2x weight vs individual spans.
    Stores both BYTEA (for backward compat) and pgvector (for fast search).
    """
    conn = _get_connection()
    cur = conn.cursor()

    centroid_bytes = centroid.astype(np.float32).tobytes()
    centroid_vec = _numpy_to_pgvector(centroid)
    examples_json = json.dumps(examples) if examples else None

    cur.execute("""
        INSERT INTO span_templates (
            template_id, pattern, centroid, centroid_vec, operation, dsl_type,
            examples, count, updated_at
        )
        VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (template_id) DO UPDATE SET
            pattern = EXCLUDED.pattern,
            centroid = EXCLUDED.centroid,
            centroid_vec = EXCLUDED.centroid_vec,
            operation = EXCLUDED.operation,
            dsl_type = EXCLUDED.dsl_type,
            examples = EXCLUDED.examples,
            count = EXCLUDED.count,
            updated_at = CURRENT_TIMESTAMP
    """, (template_id, pattern, centroid_bytes, centroid_vec, operation, dsl_type,
          examples_json, count))

    conn.commit()
    cur.close()
    conn.close()


def knn_query_templates(
    query_embedding: np.ndarray,
    k: int = 5,
    operation_filter: Optional[str] = None
) -> List[Tuple[str, str, float]]:
    """Query pgvector for k nearest template centroids.

    Uses cosine similarity via pgvector's <=> operator (cosine distance).
    Returns (template_id, operation, similarity) tuples sorted by similarity.

    This is the FAST path - no data loaded to RAM, DB does the search!
    """
    conn = _get_connection()
    cur = conn.cursor()

    query_vec = _numpy_to_pgvector(query_embedding)

    if operation_filter:
        cur.execute("""
            SELECT template_id, operation, 1 - (centroid_vec <=> %s::vector) as similarity
            FROM span_templates
            WHERE operation = %s AND centroid_vec IS NOT NULL
            ORDER BY centroid_vec <=> %s::vector
            LIMIT %s
        """, (query_vec, operation_filter, query_vec, k))
    else:
        cur.execute("""
            SELECT template_id, operation, 1 - (centroid_vec <=> %s::vector) as similarity
            FROM span_templates
            WHERE centroid_vec IS NOT NULL
            ORDER BY centroid_vec <=> %s::vector
            LIMIT %s
        """, (query_vec, query_vec, k))

    results = [(row[0], row[1], row[2]) for row in cur.fetchall()]
    cur.close()
    conn.close()

    return results


def get_template_count() -> int:
    """Get total number of templates."""
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM span_templates")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


if __name__ == "__main__":
    # Test connection and initialize schema
    print(f"Connecting to: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")
    init_db()
    print("Done!")
