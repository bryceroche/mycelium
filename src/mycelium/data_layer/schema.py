"""Database schema definitions for SQLite.

V2 Schema: Natural Language Interface
=====================================
Signatures now speak natural language:
- description: What this signature does (for LLM understanding)
- clarifying_questions: Questions to ask to extract parameters
- param_descriptions: What each DSL parameter means in plain English
- examples: Few-shot examples of input → output

The planner and signatures can now "talk" to each other through text.
"""

import logging

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768  # MathBERT dimension

SQLITE_SCHEMA = """
-- =============================================================================
-- SIGNATURES: The vocabulary of reusable computation patterns
-- =============================================================================
CREATE TABLE IF NOT EXISTS step_signatures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_id TEXT UNIQUE NOT NULL,

    -- Embedding (768-dim MathBERT)
    -- centroid = embedding_sum / embedding_count (computed on read)
    centroid TEXT NOT NULL,           -- Current centroid (for index/queries)
    centroid_bucket TEXT,             -- Quantized hash for coarse-grained uniqueness
    embedding_sum TEXT,               -- Running sum of all matched embeddings
    embedding_count INTEGER DEFAULT 1, -- Number of embeddings in sum

    -- Identity
    step_type TEXT NOT NULL,  -- e.g., "compute_power", "find_gcd"

    -- Natural Language Interface (NEW)
    description TEXT NOT NULL,  -- "Raise a base number to an exponent power"
    clarifying_questions TEXT,  -- JSON: ["What is the base?", "What is the exponent?"]
    param_descriptions TEXT,    -- JSON: {"base": "The number being raised", "exponent": "The power"}

    -- DSL Execution
    dsl_script TEXT,           -- e.g., "base ** exponent"
    dsl_type TEXT DEFAULT 'math',  -- 'math', 'sympy', 'python'

    -- Few-shot Examples (JSON array)
    examples TEXT,  -- [{"input": "2^8", "params": {"base": 2, "exp": 8}, "result": "256"}]

    -- Statistics
    uses INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,

    -- Umbrella routing (DAG of DAGs)
    is_semantic_umbrella INTEGER DEFAULT 0,  -- 1 if routes to children
    is_root INTEGER DEFAULT 0,  -- 1 if this is THE root signature (single entry point)
    depth INTEGER DEFAULT 0,  -- Routing depth (0=root, increases with parent-child hops)

    -- Lifecycle
    is_archived INTEGER DEFAULT 0,  -- 1 if soft-deleted due to decay
    last_rewrite_at TEXT,  -- When DSL was last rewritten

    -- Metadata
    created_at TEXT NOT NULL,
    last_used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sig_id ON step_signatures(signature_id);
CREATE INDEX IF NOT EXISTS idx_sig_type ON step_signatures(step_type);
CREATE INDEX IF NOT EXISTS idx_sig_centroid ON step_signatures(centroid);  -- Non-unique, for queries
CREATE UNIQUE INDEX IF NOT EXISTS idx_sig_centroid_bucket ON step_signatures(centroid_bucket);  -- Coarse uniqueness
CREATE INDEX IF NOT EXISTS idx_sig_depth ON step_signatures(depth);
CREATE INDEX IF NOT EXISTS idx_sig_is_root ON step_signatures(is_root);
CREATE INDEX IF NOT EXISTS idx_sig_dsl_type ON step_signatures(dsl_type);
CREATE INDEX IF NOT EXISTS idx_sig_umbrella_archived ON step_signatures(is_semantic_umbrella, is_archived);
CREATE INDEX IF NOT EXISTS idx_sig_archived_created ON step_signatures(is_archived, created_at);

-- =============================================================================
-- SIGNATURE RELATIONSHIPS: Tree structure for parent-child routing
-- =============================================================================
-- Enables multi-layer umbrella routing (TREE structure - single parent per child):
--   A → B → C (A parent of B, B parent of C)
--   A → B, A → C (A parent of multiple children)
--   Each child has exactly ONE parent (enforced by UNIQUE(child_id))
CREATE TABLE IF NOT EXISTS signature_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER NOT NULL REFERENCES step_signatures(id) ON DELETE CASCADE,
    child_id INTEGER NOT NULL REFERENCES step_signatures(id) ON DELETE CASCADE,
    condition TEXT NOT NULL,           -- routing condition: "counting outcomes", "complement event"
    routing_order INTEGER DEFAULT 0,   -- priority for fallback (lower = higher priority)
    created_at TEXT NOT NULL,
    UNIQUE(child_id)  -- Tree structure: each child has exactly one parent
);

CREATE INDEX IF NOT EXISTS idx_sig_rel_parent ON signature_relationships(parent_id);
CREATE INDEX IF NOT EXISTS idx_sig_rel_child ON signature_relationships(child_id);

-- =============================================================================
-- EXAMPLES: Individual step instances that belong to signatures
-- =============================================================================
CREATE TABLE IF NOT EXISTS step_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_id INTEGER NOT NULL REFERENCES step_signatures(id),
    step_text TEXT NOT NULL,
    embedding TEXT,
    result TEXT,
    success INTEGER DEFAULT 0,
    parent_problem TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_examples_sig ON step_examples(signature_id);

-- =============================================================================
-- USAGE LOG: Track what happened when signatures were used
-- =============================================================================
CREATE TABLE IF NOT EXISTS step_usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_id INTEGER NOT NULL REFERENCES step_signatures(id),
    step_text TEXT,
    step_completed INTEGER NOT NULL,  -- Whether step returned result (NOT problem correctness)
    was_injected INTEGER DEFAULT 0,
    params_extracted TEXT,  -- JSON: what params were extracted
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_sig ON step_usage_log(signature_id);
CREATE INDEX IF NOT EXISTS idx_usage_sig_created ON step_usage_log(signature_id, created_at);

-- =============================================================================
-- STEP FAILURES: Track failure patterns for learning (per CLAUDE.md)
-- =============================================================================
-- "Failures Are Valuable Data Points" - Record every failure for refinement loop
-- Used to: identify signatures needing decomposition, feed planner hints
CREATE TABLE IF NOT EXISTS step_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_id INTEGER REFERENCES step_signatures(id),  -- Nullable if no sig matched
    step_text TEXT NOT NULL,                              -- The step that failed
    failure_type TEXT NOT NULL,                           -- dsl_error, no_match, llm_error, timeout, validation
    error_message TEXT,                                   -- Actual error text
    context TEXT,                                         -- JSON: {params, expected, problem, etc.}
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_failures_sig ON step_failures(signature_id);
CREATE INDEX IF NOT EXISTS idx_failures_type ON step_failures(failure_type);
CREATE INDEX IF NOT EXISTS idx_failures_created ON step_failures(created_at);
CREATE INDEX IF NOT EXISTS idx_failures_created_sig ON step_failures(created_at, signature_id);

-- =============================================================================
-- METADATA: Key-value store for DB-level settings
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
    """Initialize the V2 database schema (SQLite)."""
    conn.executescript(SQLITE_SCHEMA)
    conn.commit()

    # WAL checkpoint on startup - merge WAL back into main DB file
    # Prevents WAL file from growing unbounded between restarts
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception as e:
        logger.warning("[schema] WAL checkpoint failed: %s", e)

    # Run migrations for existing DBs
    migrate_db(conn)


def migrate_db(conn) -> None:
    """Run migrations to add new columns to existing databases.

    This is safe to run multiple times - it only adds columns that don't exist.
    """
    # Check which columns exist
    cursor = conn.execute("PRAGMA table_info(step_signatures)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    migrations = []

    # Add embedding_sum if missing
    if "embedding_sum" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN embedding_sum TEXT"
        )

    # Add embedding_count if missing
    if "embedding_count" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN embedding_count INTEGER DEFAULT 1"
        )

    # Add depth if missing
    if "depth" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN depth INTEGER DEFAULT 0"
        )

    # Add is_root if missing
    if "is_root" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN is_root INTEGER DEFAULT 0"
        )

    # Add is_archived if missing (decay lifecycle)
    if "is_archived" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN is_archived INTEGER DEFAULT 0"
        )

    # Add last_rewrite_at if missing (DSL rewriter)
    if "last_rewrite_at" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN last_rewrite_at TEXT"
        )

    # Add difficulty_stats if missing (universal tree)
    if "difficulty_stats" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN difficulty_stats TEXT DEFAULT '{}'"
        )

    # Add max_difficulty_solved if missing (universal tree)
    if "max_difficulty_solved" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN max_difficulty_solved REAL DEFAULT 0.0"
        )

    # Run migrations
    for sql in migrations:
        try:
            conn.execute(sql)
        except Exception as e:
            logger.warning("[schema] Migration failed for '%s': %s", sql[:50], e)

    if migrations:
        conn.commit()

    # Add new indexes (safe to run multiple times)
    index_migrations = [
        "CREATE INDEX IF NOT EXISTS idx_sig_is_root ON step_signatures(is_root)",
        "CREATE INDEX IF NOT EXISTS idx_sig_dsl_type ON step_signatures(dsl_type)",
        "CREATE INDEX IF NOT EXISTS idx_sig_umbrella_archived ON step_signatures(is_semantic_umbrella, is_archived)",
        # Performance indexes (added for query optimization)
        "CREATE INDEX IF NOT EXISTS idx_sig_archived_created ON step_signatures(is_archived, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_usage_sig_created ON step_usage_log(signature_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_failures_created_sig ON step_failures(created_at, signature_id)",
    ]
    for sql in index_migrations:
        try:
            conn.execute(sql)
        except Exception as e:
            logger.warning("[schema] Index migration failed for '%s': %s", sql[:50], e)
    conn.commit()

    # Update query planner statistics for better query plans
    try:
        conn.execute("ANALYZE")
        conn.commit()
    except Exception as e:
        logger.warning("[schema] ANALYZE failed: %s", e)

    # Fix multi-parent children (tree structure enforcement)
    # This cleans up any children that have multiple parents from old DAG schema
    _fix_multi_parent_children(conn)


def _fix_multi_parent_children(conn) -> None:
    """Remove duplicate parent relationships to enforce tree structure.

    Old schema allowed DAG (multiple parents per child). New schema enforces
    tree (single parent). This migration keeps only the first parent for each child.
    """
    # Find children with multiple parents
    cursor = conn.execute("""
        SELECT child_id, COUNT(*) as parent_count
        FROM signature_relationships
        GROUP BY child_id
        HAVING parent_count > 1
    """)
    multi_parent_children = cursor.fetchall()

    if not multi_parent_children:
        return

    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "[schema] Found %d children with multiple parents, fixing...",
        len(multi_parent_children)
    )

    # For each child with multiple parents, keep only the first (by id)
    for row in multi_parent_children:
        child_id = row[0]
        # Get all parent relationships for this child, ordered by id
        cursor = conn.execute("""
            SELECT id FROM signature_relationships
            WHERE child_id = ?
            ORDER BY id ASC
        """, (child_id,))
        rel_ids = [r[0] for r in cursor.fetchall()]

        # Keep first, delete rest
        if len(rel_ids) > 1:
            ids_to_delete = rel_ids[1:]
            conn.execute(
                f"DELETE FROM signature_relationships WHERE id IN ({','.join('?' * len(ids_to_delete))})",
                ids_to_delete
            )
            logger.info(
                "[schema] Fixed child %d: kept parent rel %d, removed %d duplicates",
                child_id, rel_ids[0], len(ids_to_delete)
            )

    conn.commit()


STEP_SCHEMA = SQLITE_SCHEMA
