"""Database schema - minimal for local decomposition architecture."""

import logging
from mycelium.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)

SQLITE_SCHEMA = """
-- =============================================================================
-- SIGNATURES: The vocabulary of reusable computation patterns
-- =============================================================================
CREATE TABLE IF NOT EXISTS step_signatures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_id TEXT UNIQUE NOT NULL,

    -- Embedding
    centroid TEXT,
    embedding_sum TEXT,
    embedding_count INTEGER DEFAULT 1,

    -- Computation Graph (per CLAUDE.md: route by what operations DO)
    computation_graph TEXT,
    graph_embedding TEXT,

    -- Identity
    step_type TEXT NOT NULL,
    description TEXT NOT NULL,
    clarifying_questions TEXT,
    param_descriptions TEXT,

    -- DSL
    dsl_script TEXT,
    dsl_type TEXT DEFAULT 'math',
    examples TEXT,

    -- Statistics
    uses INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    operational_failures INTEGER DEFAULT 0,
    rejection_count INTEGER DEFAULT 0,

    -- Welford variance tracking
    similarity_count INTEGER DEFAULT 0,
    similarity_mean REAL DEFAULT 0.0,
    similarity_m2 REAL DEFAULT 0.0,
    success_sim_count INTEGER DEFAULT 0,
    success_sim_mean REAL DEFAULT 0.0,
    success_sim_m2 REAL DEFAULT 0.0,

    -- Tree structure
    is_semantic_umbrella INTEGER DEFAULT 0,
    is_root INTEGER DEFAULT 0,
    depth INTEGER DEFAULT 0,
    is_atomic INTEGER DEFAULT 0,
    atomic_reason TEXT,
    is_archived INTEGER DEFAULT 0,

    -- Metadata
    created_at TEXT NOT NULL,
    last_used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sig_id ON step_signatures(signature_id);
CREATE INDEX IF NOT EXISTS idx_sig_type ON step_signatures(step_type);
CREATE INDEX IF NOT EXISTS idx_sig_umbrella ON step_signatures(is_semantic_umbrella);

-- =============================================================================
-- SIGNATURE RELATIONSHIPS: Tree structure for parent-child routing
-- =============================================================================
CREATE TABLE IF NOT EXISTS signature_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER NOT NULL REFERENCES step_signatures(id) ON DELETE CASCADE,
    child_id INTEGER NOT NULL REFERENCES step_signatures(id) ON DELETE CASCADE,
    condition TEXT NOT NULL,
    routing_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(child_id)
);

CREATE INDEX IF NOT EXISTS idx_sig_rel_parent ON signature_relationships(parent_id);
CREATE INDEX IF NOT EXISTS idx_sig_rel_child ON signature_relationships(child_id);

-- =============================================================================
-- MCTS DAGs: Problem tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS mcts_dags (
    id TEXT PRIMARY KEY,
    problem_text TEXT,
    created_at TEXT NOT NULL,
    success INTEGER DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS mcts_dag_steps (
    id TEXT PRIMARY KEY,
    dag_id TEXT NOT NULL REFERENCES mcts_dags(id) ON DELETE CASCADE,
    step_text TEXT NOT NULL,
    step_index INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dag_steps_dag ON mcts_dag_steps(dag_id);

-- =============================================================================
-- METADATA: Key-value store
-- =============================================================================
CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def get_schema() -> str:
    return SQLITE_SCHEMA


def init_db(conn) -> None:
    """Initialize database schema."""
    conn.executescript(SQLITE_SCHEMA)
    conn.commit()

    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception as e:
        logger.warning("[schema] WAL checkpoint failed: %s", e)


STEP_SCHEMA = SQLITE_SCHEMA
