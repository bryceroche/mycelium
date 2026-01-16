-- Mycelium Cloud SQL PostgreSQL Schema
-- Run this after creating the database to initialize pgvector and tables
--
-- Usage: psql -h <host> -U mycelium -d mycelium -f init_schema.sql

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Step signatures table (main signature store)
CREATE TABLE IF NOT EXISTS step_signatures (
    id TEXT PRIMARY KEY,
    step_type TEXT NOT NULL,
    description TEXT NOT NULL,
    dsl_type TEXT,
    dsl_script TEXT,

    -- 768-dim embedding for semantic matching (text-embedding-004, pgvector compatible)
    centroid vector(768),
    embedding_sum vector(768),
    embedding_count INTEGER DEFAULT 1,

    -- Natural language interface
    clarifying_questions TEXT,
    param_descriptions TEXT,

    -- Usage statistics
    uses INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,

    -- Umbrella routing metadata
    is_semantic_umbrella BOOLEAN DEFAULT FALSE,
    is_root BOOLEAN DEFAULT FALSE,
    depth INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMP
);

-- Index for fast similarity search using pgvector
CREATE INDEX IF NOT EXISTS idx_step_signatures_centroid
ON step_signatures USING ivfflat (centroid vector_cosine_ops)
WITH (lists = 100);

-- Index for umbrella routing
CREATE INDEX IF NOT EXISTS idx_step_signatures_umbrella
ON step_signatures (is_semantic_umbrella, depth);

-- Signature relationships (parent-child tree structure)
CREATE TABLE IF NOT EXISTS signature_relationships (
    id SERIAL PRIMARY KEY,
    parent_id TEXT NOT NULL REFERENCES step_signatures(id) ON DELETE CASCADE,
    child_id TEXT NOT NULL REFERENCES step_signatures(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(parent_id, child_id)
);

CREATE INDEX IF NOT EXISTS idx_sig_rel_parent ON signature_relationships(parent_id);
CREATE INDEX IF NOT EXISTS idx_sig_rel_child ON signature_relationships(child_id);

-- Step examples (successful problem-solution pairs)
CREATE TABLE IF NOT EXISTS step_examples (
    id SERIAL PRIMARY KEY,
    signature_id TEXT NOT NULL REFERENCES step_signatures(id) ON DELETE CASCADE,
    problem TEXT NOT NULL,
    solution TEXT NOT NULL,
    is_negative BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_step_examples_sig ON step_examples(signature_id);

-- Step usage log (detailed execution history)
CREATE TABLE IF NOT EXISTS step_usage_log (
    id SERIAL PRIMARY KEY,
    signature_id TEXT NOT NULL,
    problem_hash TEXT,
    success BOOLEAN NOT NULL,
    execution_time_ms REAL,
    dsl_used BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_usage_log_sig ON step_usage_log(signature_id);
CREATE INDEX IF NOT EXISTS idx_usage_log_time ON step_usage_log(created_at);

-- Step failures (for learning from mistakes)
CREATE TABLE IF NOT EXISTS step_failures (
    id SERIAL PRIMARY KEY,
    signature_id TEXT NOT NULL,
    problem TEXT NOT NULL,
    expected TEXT,
    actual TEXT,
    error_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_failures_sig ON step_failures(signature_id);

-- Signature decay history
CREATE TABLE IF NOT EXISTS signature_decay (
    id SERIAL PRIMARY KEY,
    signature_id TEXT NOT NULL,
    action TEXT NOT NULL,  -- 'warning', 'demote', 'archive', 'recovered'
    traffic_share REAL,
    threshold REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Decay history for auditing
CREATE TABLE IF NOT EXISTS decay_history (
    id SERIAL PRIMARY KEY,
    signature_id TEXT NOT NULL,
    action TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Database metadata (key-value store for system settings)
CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for step_signatures
DROP TRIGGER IF EXISTS step_signatures_updated_at ON step_signatures;
CREATE TRIGGER step_signatures_updated_at
    BEFORE UPDATE ON step_signatures
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Trigger for db_metadata
DROP TRIGGER IF EXISTS db_metadata_updated_at ON db_metadata;
CREATE TRIGGER db_metadata_updated_at
    BEFORE UPDATE ON db_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
