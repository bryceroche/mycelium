"""Database schema definitions for SQLite."""

EMBEDDING_DIM = 384

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS step_signatures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_id TEXT UNIQUE NOT NULL,
    centroid TEXT NOT NULL,
    step_type TEXT NOT NULL,
    description TEXT,
    method_name TEXT NOT NULL,
    method_template TEXT NOT NULL,
    example_count INTEGER DEFAULT 0,
    uses INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    injected_uses INTEGER DEFAULT 0,
    injected_successes INTEGER DEFAULT 0,
    non_injected_uses INTEGER DEFAULT 0,
    non_injected_successes INTEGER DEFAULT 0,
    cohesion REAL DEFAULT 0.0,
    is_canonical INTEGER DEFAULT 0,
    canonical_parent_id INTEGER REFERENCES step_signatures(id),
    variant_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    io_schema TEXT,
    amplitude REAL DEFAULT 0.1,
    phase REAL DEFAULT 0.0,
    spread REAL DEFAULT 0.3,
    plan_type TEXT,
    compressed_instruction TEXT,
    param_schema TEXT,
    output_format TEXT,
    plan_optimization_method TEXT,
    plan_tokens_before INTEGER,
    plan_tokens_after INTEGER,
    plan_validation_accuracy REAL,
    dsl_script TEXT,
    dsl_version INTEGER DEFAULT 1,
    dsl_version_uses INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_step_sig_id ON step_signatures(signature_id);
CREATE INDEX IF NOT EXISTS idx_step_sig_type ON step_signatures(step_type);

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

CREATE INDEX IF NOT EXISTS idx_step_examples_sig ON step_examples(signature_id);

CREATE TABLE IF NOT EXISTS step_usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_id INTEGER NOT NULL REFERENCES step_signatures(id),
    step_text TEXT,
    success INTEGER NOT NULL,
    amplitude_at_use REAL,
    was_injected INTEGER DEFAULT 0,
    match_mode TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_step_usage_sig ON step_usage_log(signature_id);

CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

def get_schema() -> str:
    return SQLITE_SCHEMA

def init_db(conn) -> None:
    conn.executescript(SQLITE_SCHEMA)

    # Migration: add new columns if they don't exist
    cursor = conn.execute("PRAGMA table_info(step_signatures)")
    columns = {row[1] for row in cursor.fetchall()}

    if "dsl_script" not in columns:
        conn.execute("ALTER TABLE step_signatures ADD COLUMN dsl_script TEXT")

    if "dsl_version" not in columns:
        conn.execute("ALTER TABLE step_signatures ADD COLUMN dsl_version INTEGER DEFAULT 1")

    if "dsl_version_uses" not in columns:
        conn.execute("ALTER TABLE step_signatures ADD COLUMN dsl_version_uses INTEGER DEFAULT 0")

    conn.commit()

STEP_SCHEMA = SQLITE_SCHEMA
