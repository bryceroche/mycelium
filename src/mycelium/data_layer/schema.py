"""Database schema - flat prototype store for signature-based function routing.

Architecture: Signatures are prototypes that map to ~200 atomic functions.
Classification is via brute-force k-NN (fast at this scale).
"""

import logging

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

    -- Computation Graph (for semantic routing)
    computation_graph TEXT,
    graph_embedding TEXT,

    -- Identity
    step_type TEXT NOT NULL,
    description TEXT NOT NULL,
    clarifying_questions TEXT,
    param_descriptions TEXT,

    -- Function Pointer
    func_name TEXT,               -- Key into function_registry
    func_arity INTEGER DEFAULT 2, -- Expected number of arguments
    examples TEXT,

    -- Statistics
    uses INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    operational_failures INTEGER DEFAULT 0,

    -- Welford variance tracking (success similarity)
    success_sim_count INTEGER DEFAULT 0,
    success_sim_mean REAL DEFAULT 0.0,
    success_sim_m2 REAL DEFAULT 0.0,

    -- Welford variance tracking (outcome: 1.0=success, 0.0=failure)
    -- High variance = inconsistent outcomes = decomposition candidate
    outcome_count INTEGER DEFAULT 0,
    outcome_mean REAL DEFAULT 0.0,
    outcome_m2 REAL DEFAULT 0.0,

    -- Coverage tracking
    coverage_sim_count INTEGER DEFAULT 0,
    coverage_sim_mean REAL DEFAULT 0.0,
    coverage_sim_m2 REAL DEFAULT 0.0,
    low_coverage_count INTEGER DEFAULT 0,

    -- Merge tracking
    description_variants TEXT,
    merge_dist_count INTEGER DEFAULT 0,
    merge_dist_mean REAL DEFAULT 0.0,
    merge_dist_m2 REAL DEFAULT 0.0,

    -- Metadata
    created_at TEXT NOT NULL,
    last_used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sig_id ON step_signatures(signature_id);
CREATE INDEX IF NOT EXISTS idx_sig_type ON step_signatures(step_type);
CREATE INDEX IF NOT EXISTS idx_sig_func ON step_signatures(func_name);

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

    # Run migrations for existing databases
    _run_migrations(conn)

    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception as e:
        logger.warning("[schema] WAL checkpoint failed: %s", e)


def _run_migrations(conn) -> None:
    """Run schema migrations for existing databases."""
    cursor = conn.execute("PRAGMA table_info(step_signatures)")
    columns = {row[1] for row in cursor.fetchall()}

    # Add columns if missing (for backward compatibility with older databases)
    migrations = [
        ("func_name", "TEXT"),
        ("func_arity", "INTEGER DEFAULT 2"),
        ("description_variants", "TEXT"),
        ("merge_dist_count", "INTEGER DEFAULT 0"),
        ("merge_dist_mean", "REAL DEFAULT 0.0"),
        ("merge_dist_m2", "REAL DEFAULT 0.0"),
        ("coverage_sim_count", "INTEGER DEFAULT 0"),
        ("coverage_sim_mean", "REAL DEFAULT 0.0"),
        ("coverage_sim_m2", "REAL DEFAULT 0.0"),
        ("low_coverage_count", "INTEGER DEFAULT 0"),
        ("outcome_count", "INTEGER DEFAULT 0"),
        ("outcome_mean", "REAL DEFAULT 0.0"),
        ("outcome_m2", "REAL DEFAULT 0.0"),
    ]

    for col_name, col_type in migrations:
        if col_name not in columns:
            logger.info(f"[schema] Adding {col_name} column")
            try:
                conn.execute(f"ALTER TABLE step_signatures ADD COLUMN {col_name} {col_type}")
                conn.commit()
            except Exception as e:
                logger.warning(f"[schema] Failed to add {col_name}: {e}")


STEP_SCHEMA = SQLITE_SCHEMA
