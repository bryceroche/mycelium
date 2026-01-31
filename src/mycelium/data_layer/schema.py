"""Database schema - flat prototype store for 200-class classification.

Architecture: Signatures are prototypes that map to ~200 atomic functions.
Classification is via brute-force k-NN (fast at 5k scale).

The hierarchy fields (is_semantic_umbrella, is_root, depth) are kept for
backward compatibility but are no longer actively used in the flat architecture.
"""

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

    -- Function Pointer (replaces DSL)
    func_name TEXT,               -- Key into function_registry
    func_arity INTEGER DEFAULT 2, -- Expected number of arguments
    examples TEXT,

    -- Legacy DSL (deprecated, kept for migration)
    dsl_script TEXT,
    dsl_type TEXT DEFAULT 'math',

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
CREATE INDEX IF NOT EXISTS idx_sig_func ON step_signatures(func_name);
CREATE INDEX IF NOT EXISTS idx_sig_umbrella ON step_signatures(is_semantic_umbrella);

-- =============================================================================
-- SIGNATURE RELATIONSHIPS: DEPRECATED - kept for backward compatibility
-- The flat prototype architecture uses brute-force k-NN, not tree traversal.
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

-- =============================================================================
-- STEP SEQUENCES: Track primitive chains for chain node creation
-- Per CLAUDE.md Big 5 #5: Primitive vs Chain Nodes
-- =============================================================================
CREATE TABLE IF NOT EXISTS step_sequences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence TEXT NOT NULL,           -- JSON: ['leaf_1', 'leaf_2', 'leaf_3']
    sequence_hash TEXT UNIQUE NOT NULL,  -- Hash of sequence for fast lookup
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    chain_node_id INTEGER DEFAULT NULL,  -- FK to step_signatures if chain created
    created_at TEXT NOT NULL,
    FOREIGN KEY (chain_node_id) REFERENCES step_signatures(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_seq_hash ON step_sequences(sequence_hash);
CREATE INDEX IF NOT EXISTS idx_seq_chain ON step_sequences(chain_node_id);

-- =============================================================================
-- FAILURE LOG: Track failed executions for periodic review
-- =============================================================================
CREATE TABLE IF NOT EXISTS execution_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_description TEXT NOT NULL,
    func_name TEXT,
    signature_id INTEGER REFERENCES step_signatures(id),
    similarity REAL,
    error_type TEXT,           -- 'wrong_answer', 'execution_error', 'no_match'
    error_message TEXT,
    problem_id TEXT,           -- Optional reference to problem
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_failures_func ON execution_failures(func_name);
CREATE INDEX IF NOT EXISTS idx_failures_type ON execution_failures(error_type);
CREATE INDEX IF NOT EXISTS idx_failures_created ON execution_failures(created_at);
"""


def get_schema() -> str:
    return SQLITE_SCHEMA


def init_db(conn) -> None:
    """Initialize database schema."""
    conn.executescript(SQLITE_SCHEMA)
    conn.commit()

    # Run migrations for existing databases
    _run_migrations(conn)

    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception as e:
        logger.warning("[schema] WAL checkpoint failed: %s", e)


def _run_migrations(conn) -> None:
    """Run schema migrations for existing databases."""
    # Check if func_name column exists
    cursor = conn.execute("PRAGMA table_info(step_signatures)")
    columns = {row[1] for row in cursor.fetchall()}

    if "func_name" not in columns:
        logger.info("[schema] Adding func_name column")
        try:
            conn.execute("ALTER TABLE step_signatures ADD COLUMN func_name TEXT")
            conn.commit()
        except Exception as e:
            logger.warning("[schema] Failed to add func_name: %s", e)

    if "func_arity" not in columns:
        logger.info("[schema] Adding func_arity column")
        try:
            conn.execute("ALTER TABLE step_signatures ADD COLUMN func_arity INTEGER DEFAULT 2")
            conn.commit()
        except Exception as e:
            logger.warning("[schema] Failed to add func_arity: %s", e)

    # Signature merge columns - for tracking merged descriptions and Welford stats
    if "description_variants" not in columns:
        logger.info("[schema] Adding description_variants column")
        try:
            conn.execute("ALTER TABLE step_signatures ADD COLUMN description_variants TEXT")
            conn.commit()
        except Exception as e:
            logger.warning("[schema] Failed to add description_variants: %s", e)

    if "merge_dist_count" not in columns:
        logger.info("[schema] Adding merge_dist_count column")
        try:
            conn.execute("ALTER TABLE step_signatures ADD COLUMN merge_dist_count INTEGER DEFAULT 0")
            conn.commit()
        except Exception as e:
            logger.warning("[schema] Failed to add merge_dist_count: %s", e)

    if "merge_dist_mean" not in columns:
        logger.info("[schema] Adding merge_dist_mean column")
        try:
            conn.execute("ALTER TABLE step_signatures ADD COLUMN merge_dist_mean REAL DEFAULT 0.0")
            conn.commit()
        except Exception as e:
            logger.warning("[schema] Failed to add merge_dist_mean: %s", e)

    if "merge_dist_m2" not in columns:
        logger.info("[schema] Adding merge_dist_m2 column")
        try:
            conn.execute("ALTER TABLE step_signatures ADD COLUMN merge_dist_m2 REAL DEFAULT 0.0")
            conn.commit()
        except Exception as e:
            logger.warning("[schema] Failed to add merge_dist_m2: %s", e)

    # Coverage tracking columns - for tracking how well signatures cover step descriptions
    if "coverage_sim_count" not in columns:
        logger.info("[schema] Adding coverage tracking columns")
        try:
            conn.execute("ALTER TABLE step_signatures ADD COLUMN coverage_sim_count INTEGER DEFAULT 0")
            conn.execute("ALTER TABLE step_signatures ADD COLUMN coverage_sim_mean REAL DEFAULT 0.0")
            conn.execute("ALTER TABLE step_signatures ADD COLUMN coverage_sim_m2 REAL DEFAULT 0.0")
            conn.execute("ALTER TABLE step_signatures ADD COLUMN low_coverage_count INTEGER DEFAULT 0")
            conn.commit()
        except Exception as e:
            logger.warning("[schema] Failed to add coverage columns: %s", e)


STEP_SCHEMA = SQLITE_SCHEMA
