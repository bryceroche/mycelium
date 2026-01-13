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

EMBEDDING_DIM = 768  # MathBERT dimension

SQLITE_SCHEMA = """
-- =============================================================================
-- SIGNATURES: The vocabulary of reusable computation patterns
-- =============================================================================
CREATE TABLE IF NOT EXISTS step_signatures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_id TEXT UNIQUE NOT NULL,

    -- Embedding (768-dim MathBERT)
    centroid TEXT NOT NULL,

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

    -- Metadata
    created_at TEXT NOT NULL,
    last_used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sig_id ON step_signatures(signature_id);
CREATE INDEX IF NOT EXISTS idx_sig_type ON step_signatures(step_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sig_centroid ON step_signatures(centroid);

-- =============================================================================
-- SIGNATURE RELATIONSHIPS: DAG of parent-child routing
-- =============================================================================
-- Enables multi-layer umbrella routing:
--   A → B → C (A parent of B, B parent of C)
--   A → B, A → C (A parent of multiple children)
--   B → D, C → D (D has multiple parents - true DAG)
CREATE TABLE IF NOT EXISTS signature_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id INTEGER NOT NULL REFERENCES step_signatures(id) ON DELETE CASCADE,
    child_id INTEGER NOT NULL REFERENCES step_signatures(id) ON DELETE CASCADE,
    condition TEXT NOT NULL,           -- routing condition: "counting outcomes", "complement event"
    routing_order INTEGER DEFAULT 0,   -- priority for fallback (lower = higher priority)
    created_at TEXT NOT NULL,
    UNIQUE(parent_id, child_id)
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
    success INTEGER NOT NULL,
    was_injected INTEGER DEFAULT 0,
    params_extracted TEXT,  -- JSON: what params were extracted
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_sig ON step_usage_log(signature_id);

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
    """Initialize the V2 database schema."""
    conn.executescript(SQLITE_SCHEMA)
    conn.commit()


STEP_SCHEMA = SQLITE_SCHEMA
