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

from mycelium.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)

SQLITE_SCHEMA = """
-- =============================================================================
-- SIGNATURES: The vocabulary of reusable computation patterns
-- =============================================================================
CREATE TABLE IF NOT EXISTS step_signatures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_id TEXT UNIQUE NOT NULL,

    -- Embedding (3072-dim gemini-embedding-001)
    -- centroid = embedding_sum / embedding_count (computed on read)
    centroid TEXT,                    -- Current centroid (for index/queries), NULL for uninitialized scaffolds
    centroid_bucket TEXT,             -- Quantized hash for coarse-grained uniqueness
    embedding_sum TEXT,               -- Running sum of all matched embeddings
    embedding_count INTEGER DEFAULT 1, -- Number of embeddings in sum

    -- Computation Graph Embedding (per CLAUDE.md: route by what operations DO)
    -- Graph is structural representation: MUL(param_0, param_1) → result
    -- graph_embedding is what we route against (replaces text-based centroid for routing)
    computation_graph TEXT,           -- Structural graph: "MUL(param_0, param_1)", "ADD(MUL(p0, p1), p2)"
    graph_embedding TEXT,             -- Embedding of the computation graph (for routing)

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
    operational_failures INTEGER DEFAULT 0,  -- MCTS: times produced wrong answer vs ground truth
    rejection_count INTEGER DEFAULT 0,  -- Times this leaf rejected a dag_step (low similarity)

    -- Embedding Variance Tracking (Welford's online algorithm)
    -- Tracks how diverse the problems routed to this signature are
    -- High variance = too generic, should decompose into specialized children
    similarity_count INTEGER DEFAULT 0,     -- N in Welford's algorithm
    similarity_mean REAL DEFAULT 0.0,       -- Running mean of cosine similarities to centroid
    similarity_m2 REAL DEFAULT 0.0,         -- Sum of squared differences (variance = M2/N)

    -- Success Similarity Tracking (Welford's algorithm for adaptive rejection)
    -- Tracks similarity scores of SUCCESSFUL matches (for adaptive rejection threshold)
    -- Per CLAUDE.md: leaf nodes should reject dag_steps when similarity is below
    -- their historical success distribution: threshold = mean - k*std
    success_sim_count INTEGER DEFAULT 0,    -- N successful matches
    success_sim_mean REAL DEFAULT 0.0,      -- Running mean of similarity on success
    success_sim_m2 REAL DEFAULT 0.0,        -- M2 for variance (std = sqrt(M2/N))

    -- Umbrella routing (DAG of DAGs)
    is_semantic_umbrella INTEGER DEFAULT 0,  -- 1 if routes to children
    is_root INTEGER DEFAULT 0,  -- 1 if this is THE root signature (single entry point)
    depth INTEGER DEFAULT 0,  -- Routing depth (0=root, increases with parent-child hops)

    -- Atomic operations (math primes)
    -- Discovered via statistics: high success rate + enough uses = stop decomposing
    is_atomic INTEGER DEFAULT 0,  -- 1 if this is a "math prime" that should never decompose
    atomic_reason TEXT,  -- Why this was marked atomic: "high_success", "decomp_failed", etc.

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
CREATE INDEX IF NOT EXISTS idx_sig_graph_embedding ON step_signatures(graph_embedding);  -- For graph-based routing
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
    expression TEXT,            -- DSL script that was executed (e.g., "a * b")
    inputs TEXT,                -- JSON: parameter values used (e.g., {"a": 5, "b": 3})
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
-- STEP STATS: Per-step execution analytics (feature flagged)
-- =============================================================================
-- Per CLAUDE.md: "DB audit for signature step level stats"
-- Tracks latency, routing depth, cache hits for performance analysis.
CREATE TABLE IF NOT EXISTS step_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signature_id INTEGER REFERENCES step_signatures(id),  -- Nullable if no sig matched
    step_text TEXT NOT NULL,                              -- The step being executed

    -- Timing
    latency_ms REAL NOT NULL,                             -- Total step execution time
    embed_latency_ms REAL DEFAULT 0,                      -- Embedding lookup/compute time
    route_latency_ms REAL DEFAULT 0,                      -- Routing decision time
    exec_latency_ms REAL DEFAULT 0,                       -- DSL/LLM execution time

    -- Routing
    routing_depth INTEGER DEFAULT 0,                      -- How deep in umbrella tree
    was_routed INTEGER DEFAULT 0,                         -- 1 if routed through umbrella
    route_path TEXT,                                      -- JSON: signature IDs traversed

    -- Cache
    embed_cache_hit INTEGER DEFAULT 0,                    -- 1 if embedding was cached

    -- Outcome
    success INTEGER DEFAULT 0,                            -- 1 if step completed successfully
    used_dsl INTEGER DEFAULT 0,                           -- 1 if DSL executed (vs LLM fallback)

    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_step_stats_sig ON step_stats(signature_id);
CREATE INDEX IF NOT EXISTS idx_step_stats_created ON step_stats(created_at);
CREATE INDEX IF NOT EXISTS idx_step_stats_depth ON step_stats(routing_depth);

-- =============================================================================
-- METADATA: Key-value store for DB-level settings
-- =============================================================================
CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- =============================================================================
-- MCTS WAVE FUNCTION TABLES: Amplitude tracking for multi-path exploration
-- =============================================================================
-- Per ideas.md: "The combination of dag_step_id and node_id is what we're learning"
-- Wave function collapses at final step where we have ground truth.
-- Amplitude updates based on interference patterns across threads.

-- DAG: A problem and its decomposition plan
CREATE TABLE IF NOT EXISTS mcts_dags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dag_id TEXT UNIQUE NOT NULL,          -- Unique ID for this DAG
    problem_id TEXT NOT NULL,             -- Problem identifier (e.g., hash of text)
    problem_desc TEXT,                    -- Full problem text
    benchmark TEXT,                       -- gsm8k, math500_L1, math500_L5, etc.
    difficulty_level REAL,                -- Numeric difficulty score
    success INTEGER DEFAULT NULL,         -- NULL until graded, 0=wrong, 1=correct
    ground_truth TEXT,                    -- Correct answer for comparison
    created_at TEXT NOT NULL,
    graded_at TEXT                        -- When final answer was graded
);

CREATE INDEX IF NOT EXISTS idx_mcts_dags_problem ON mcts_dags(problem_id);
CREATE INDEX IF NOT EXISTS idx_mcts_dags_benchmark ON mcts_dags(benchmark);
CREATE INDEX IF NOT EXISTS idx_mcts_dags_success ON mcts_dags(success);
CREATE INDEX IF NOT EXISTS idx_mcts_dags_difficulty ON mcts_dags(difficulty_level);

-- DAG_Step: Individual steps in a decomposition plan
-- Per ideas.md: "Each step should not be capable of being broken down further"
CREATE TABLE IF NOT EXISTS mcts_dag_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dag_step_id TEXT UNIQUE NOT NULL,     -- Unique ID for this step
    dag_id TEXT NOT NULL REFERENCES mcts_dags(dag_id) ON DELETE CASCADE,
    step_desc TEXT NOT NULL,              -- What this step does (natural language)
    dsl_hint TEXT,                        -- Operation type hint (e.g., "compute_sum") for stats normalization
    step_num INTEGER NOT NULL,            -- Sequential order (1..n)
    branch_num INTEGER DEFAULT 1,         -- Parallel branch ID (1..n for independent steps)
    is_atomic INTEGER DEFAULT 0,          -- 1 if cannot be decomposed further
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mcts_dag_steps_dag ON mcts_dag_steps(dag_id);
CREATE INDEX IF NOT EXISTS idx_mcts_dag_steps_step_num ON mcts_dag_steps(dag_id, step_num);
CREATE INDEX IF NOT EXISTS idx_mcts_dag_steps_dsl_hint ON mcts_dag_steps(dsl_hint);

-- Thread: A single MCTS rollout path through the DAG
-- Per ideas.md: "Thread ID essential for backpropagation"
CREATE TABLE IF NOT EXISTS mcts_threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT UNIQUE NOT NULL,       -- Unique ID for this thread
    dag_id TEXT NOT NULL REFERENCES mcts_dags(dag_id) ON DELETE CASCADE,
    parent_thread_id TEXT REFERENCES mcts_threads(thread_id),  -- NULL for root thread, else forked from
    fork_at_step TEXT REFERENCES mcts_dag_steps(dag_step_id),  -- dag_step_id where this thread forked
    fork_reason TEXT,                     -- Why we branched: 'undecided', 'explore', 'top_k'
    final_answer TEXT,                    -- Answer produced by this thread
    success INTEGER DEFAULT NULL,         -- NULL until graded, 0=wrong, 1=correct
    created_at TEXT NOT NULL,
    graded_at TEXT                        -- When answer was compared to ground truth
);

CREATE INDEX IF NOT EXISTS idx_mcts_threads_dag ON mcts_threads(dag_id);
CREATE INDEX IF NOT EXISTS idx_mcts_threads_parent ON mcts_threads(parent_thread_id);
CREATE INDEX IF NOT EXISTS idx_mcts_threads_success ON mcts_threads(success);

-- Thread_Step: Fact table for MCTS rollouts (wave function amplitudes)
-- Per ideas.md: "The combination of dag_step_id and node_id is what we're learning"
-- This is the core table for post-mortem credit/blame analysis.
CREATE TABLE IF NOT EXISTS mcts_thread_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_step_id TEXT UNIQUE NOT NULL,  -- Unique ID for this step execution
    thread_id TEXT NOT NULL REFERENCES mcts_threads(thread_id) ON DELETE CASCADE,
    dag_id TEXT NOT NULL REFERENCES mcts_dags(dag_id) ON DELETE CASCADE,  -- Denormalized for query efficiency
    dag_step_id TEXT NOT NULL REFERENCES mcts_dag_steps(dag_step_id),
    node_id INTEGER NOT NULL REFERENCES step_signatures(id),  -- Which signature (leaf node) was used
    node_depth INTEGER,                   -- Depth of node in signature tree (for post-mortem analysis)

    -- Wave function amplitude tracking
    -- Per ideas.md: "High confidence + failure = strong negative signal"
    amplitude REAL DEFAULT 1.0,           -- Prior confidence in this choice (|α|²)
    amplitude_post REAL DEFAULT NULL,     -- Updated amplitude after grading

    -- Routing decision context
    similarity_score REAL,                -- Cosine similarity when routed
    was_undecided INTEGER DEFAULT 0,      -- 1 if we branched here (low confidence)
    ucb1_gap REAL,                        -- Gap between top-2 UCB1 scores
    alternatives_considered INTEGER DEFAULT 1,  -- How many options we evaluated

    -- Step execution outcome
    step_result TEXT,                     -- Result produced at this step
    step_success INTEGER DEFAULT NULL,    -- 1 if step executed without error

    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mcts_thread_steps_thread ON mcts_thread_steps(thread_id);
CREATE INDEX IF NOT EXISTS idx_mcts_thread_steps_dag ON mcts_thread_steps(dag_id);
CREATE INDEX IF NOT EXISTS idx_mcts_thread_steps_dag_step ON mcts_thread_steps(dag_step_id);
CREATE INDEX IF NOT EXISTS idx_mcts_thread_steps_node ON mcts_thread_steps(node_id);
CREATE INDEX IF NOT EXISTS idx_mcts_thread_steps_amplitude ON mcts_thread_steps(amplitude);
CREATE INDEX IF NOT EXISTS idx_mcts_thread_steps_undecided ON mcts_thread_steps(was_undecided);
-- Composite index for post-mortem analysis: (dag_step_id, node_id) is what we're learning
CREATE INDEX IF NOT EXISTS idx_mcts_thread_steps_learning ON mcts_thread_steps(dag_step_id, node_id);

-- =============================================================================
-- MCTS_STEP_SUMMARIES: Lightweight table for credit propagation
-- =============================================================================
-- Contains minimal data needed for credit propagation and aggregate stats.
-- mcts_thread_steps (detailed) only stores failures for debugging/analysis.
-- This separation reduces DB writes while keeping credit propagation working.
CREATE TABLE IF NOT EXISTS mcts_step_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL REFERENCES mcts_threads(thread_id) ON DELETE CASCADE,
    dag_id TEXT NOT NULL REFERENCES mcts_dags(dag_id) ON DELETE CASCADE,
    dag_step_id TEXT NOT NULL REFERENCES mcts_dag_steps(dag_step_id),
    node_id INTEGER NOT NULL REFERENCES step_signatures(id),

    -- Wave function amplitude (minimal data for credit propagation)
    amplitude REAL DEFAULT 1.0,           -- Prior confidence
    amplitude_post REAL DEFAULT NULL,     -- Updated after grading
    step_success INTEGER DEFAULT NULL,    -- 1=success, 0=failure, NULL=unknown
    similarity_score REAL DEFAULT NULL    -- Cosine similarity used for routing (for adaptive rejection)
);
CREATE INDEX IF NOT EXISTS idx_mcts_step_summaries_dag ON mcts_step_summaries(dag_id);
CREATE INDEX IF NOT EXISTS idx_mcts_step_summaries_thread ON mcts_step_summaries(thread_id);
CREATE INDEX IF NOT EXISTS idx_mcts_step_summaries_node ON mcts_step_summaries(node_id);
CREATE INDEX IF NOT EXISTS idx_mcts_step_summaries_dag_step ON mcts_step_summaries(dag_step_id);
CREATE INDEX IF NOT EXISTS idx_mcts_step_summaries_success ON mcts_step_summaries(step_success);

-- Materialized stats for (dag_step_type, node_id) pairs
-- This closes the feedback loop: post-mortem → stats → routing decisions
CREATE TABLE IF NOT EXISTS dag_step_node_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dag_step_type TEXT NOT NULL,              -- e.g., "compute_sum", "compute_product"
    node_id INTEGER NOT NULL,                 -- References step_signatures(id)
    uses INTEGER DEFAULT 0,                   -- Total times this pair was used
    wins INTEGER DEFAULT 0,                   -- Times the thread won
    losses INTEGER DEFAULT 0,                 -- Times the thread lost
    win_rate REAL DEFAULT 0.5,                -- Computed: wins / uses (with prior)
    avg_amplitude_post REAL DEFAULT 1.0,      -- Running avg of amplitude_post (legacy)
    amplitude_post_sum REAL DEFAULT 0.0,      -- Sum for incremental avg calculation (legacy)

    -- Welford's algorithm for amplitude_post variance tracking (OUTCOME VARIANCE)
    -- High variance = inconsistent performance = decomposition signal
    -- Per CLAUDE.md: "The combination of (dag_step_id, node_id) is what we're learning"
    amp_post_count INTEGER DEFAULT 0,         -- N in Welford's algorithm
    amp_post_mean REAL DEFAULT 0.0,           -- Running mean of amplitude_post
    amp_post_m2 REAL DEFAULT 0.0,             -- M2 for variance: var = M2/N

    -- Welford's algorithm for similarity variance tracking (EMBEDDING VARIANCE)
    -- High variance = dag_step_ids routed here have diverse embeddings = type too broad
    -- Per CLAUDE.md: leaf_node ≡ dag_step_type should be 1:1
    sim_count INTEGER DEFAULT 0,              -- N in Welford's algorithm
    sim_mean REAL DEFAULT 0.0,                -- Running mean of similarity scores
    sim_m2 REAL DEFAULT 0.0,                  -- M2 for variance: var = M2/N

    last_updated TEXT,                        -- ISO timestamp
    UNIQUE(dag_step_type, node_id),
    FOREIGN KEY (node_id) REFERENCES step_signatures(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_dsns_lookup ON dag_step_node_stats(dag_step_type, node_id);
CREATE INDEX IF NOT EXISTS idx_dsns_node ON dag_step_node_stats(node_id);

-- =============================================================================
-- DAG_STEP_EMBEDDINGS: Semantic similarity for decomposition decisions
-- =============================================================================
-- Stores embeddings for dag_steps to enable:
-- 1. "Find similar dag_steps that succeeded" queries
-- 2. Data-driven decomposition: decompose dag_step vs leaf_node based on stats
-- Per CLAUDE.md: "Semantic Embedding First"
CREATE TABLE IF NOT EXISTS dag_step_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dag_id TEXT NOT NULL REFERENCES mcts_dags(dag_id) ON DELETE CASCADE,
    dag_step_id TEXT NOT NULL REFERENCES mcts_dag_steps(dag_step_id) ON DELETE CASCADE,
    step_desc TEXT NOT NULL,              -- The step description (task)
    embedding BLOB,                       -- Packed embedding vector
    node_id INTEGER,                      -- Which leaf_node handled it (NULL if not yet executed)
    success INTEGER,                      -- 0=failed, 1=succeeded, NULL=not yet known
    created_at TEXT NOT NULL,             -- ISO timestamp
    FOREIGN KEY (node_id) REFERENCES step_signatures(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_dse_dag ON dag_step_embeddings(dag_id);
CREATE INDEX IF NOT EXISTS idx_dse_dag_step ON dag_step_embeddings(dag_step_id);
CREATE INDEX IF NOT EXISTS idx_dse_node ON dag_step_embeddings(node_id);
CREATE INDEX IF NOT EXISTS idx_dse_success ON dag_step_embeddings(success);

-- =============================================================================
-- DAG_PLAN_STATS: Track success rates of (DAG plan, Thread) pairs
-- =============================================================================
-- Per beads mycelium-ogo6: Track which decomposition strategies work.
-- plan_signature = hash of (step tasks + dependency structure)
-- A plan that consistently fails suggests the decomposition strategy is wrong.
-- A plan that consistently succeeds suggests good problem structure.
CREATE TABLE IF NOT EXISTS dag_plan_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_signature TEXT UNIQUE NOT NULL,  -- Hash of normalized plan structure
    step_count INTEGER NOT NULL,          -- Number of steps in plan
    plan_structure TEXT,                  -- JSON: normalized step descriptions + deps
    uses INTEGER DEFAULT 0,               -- Total times this plan was used
    successes INTEGER DEFAULT 0,          -- Times the plan led to correct answer
    success_rate REAL DEFAULT 0.5,        -- Computed: successes / uses (with prior)
    first_seen_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dps_signature ON dag_plan_stats(plan_signature);
CREATE INDEX IF NOT EXISTS idx_dps_success_rate ON dag_plan_stats(success_rate);
CREATE INDEX IF NOT EXISTS idx_dps_uses ON dag_plan_stats(uses);

-- =============================================================================
-- DECOMPOSITION_QUEUE: Batch complex steps for later decomposition
-- =============================================================================
-- Per beads mycelium-mm08: Instead of decomposing immediately (1 LLM call per step),
-- queue complex steps and batch decompose them periodically.
-- This is more efficient and allows the system to learn patterns.
CREATE TABLE IF NOT EXISTS decomposition_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_text TEXT NOT NULL,              -- The complex step to decompose
    embedding TEXT,                       -- Step embedding (for similarity after decomp)
    dag_step_id TEXT,                     -- Optional link to originating dag_step
    problem_context TEXT,                 -- Original problem (for LLM context)
    complexity_reason TEXT NOT NULL,      -- Why queued: multi_op, long_step, sequential, etc.
    queued_at TEXT NOT NULL,
    processed_at TEXT,                    -- NULL until processed
    result_signature_ids TEXT,            -- JSON array of created atomic signature IDs
    decomposition_steps TEXT              -- JSON: the atomic steps LLM produced
);
CREATE INDEX IF NOT EXISTS idx_decomp_queue_processed ON decomposition_queue(processed_at);
CREATE INDEX IF NOT EXISTS idx_decomp_queue_queued ON decomposition_queue(queued_at);

-- =============================================================================
-- WELFORD_STATS: Per-signature statistics for restructuring decisions
-- =============================================================================
-- Per mycelium-bjrf: Consolidated Welford stats for tree restructuring.
-- Cold start (first 20 problems): collect stats, flat structure under root.
-- After cold start: use these stats to guide sibling/child/merge decisions.
--
-- Three tracked distributions per signature:
-- 1. route_* - similarities when routing TO this node (how well does routing work?)
-- 2. child_* - similarities BETWEEN children (for umbrellas: how tight is cluster?)
-- 3. exec_* - execution success rate (is this node reliable?)
CREATE TABLE IF NOT EXISTS welford_stats (
    signature_id INTEGER PRIMARY KEY REFERENCES step_signatures(id) ON DELETE CASCADE,

    -- Routing similarity stats (how well does routing work to this node?)
    -- Updated every time a step is routed to this signature
    route_n INTEGER DEFAULT 0,            -- N in Welford's algorithm
    route_mean REAL DEFAULT 0.0,          -- Running mean of routing similarities
    route_m2 REAL DEFAULT 0.0,            -- Sum of squared differences (variance = M2/(N-1))

    -- Child cluster stats (for umbrellas: how tight is the cluster?)
    -- Updated when adding/removing children, tracks inter-child similarities
    child_n INTEGER DEFAULT 0,            -- N pairs compared
    child_mean REAL DEFAULT 0.0,          -- Mean similarity between children
    child_m2 REAL DEFAULT 0.0,            -- M2 for variance

    -- Execution success rate (simple ratio, not Welford)
    exec_n INTEGER DEFAULT 0,             -- Total executions
    exec_successes INTEGER DEFAULT 0,     -- Successful executions

    -- Decomposition success rate (guides whether to attempt decomposition)
    -- Per CLAUDE.md: "Failures Are Valuable Data Points" - learn which sigs are atomic
    decomp_attempts INTEGER DEFAULT 0,    -- Times decomposition was attempted
    decomp_successes INTEGER DEFAULT 0,   -- Times decomposition produced >1 children

    -- Metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_welford_route_n ON welford_stats(route_n);
CREATE INDEX IF NOT EXISTS idx_welford_exec_n ON welford_stats(exec_n);

-- =============================================================================
-- PLAN_STEP_STATS: Statistical blame accumulation for (plan, position, node)
-- =============================================================================
-- Per CLAUDE.md: "Failures Are Valuable Data Points" - accumulate blame statistically.
-- Tracks success rate at each step position within a plan structure.
-- This enables:
-- 1. Identifying which step positions consistently fail
-- 2. Detecting which nodes are problematic at certain positions
-- 3. Order-aware tracking (same node at step 1 vs step 5 may differ)
-- 4. Plan-aware routing (avoid nodes that fail in certain plan contexts)
CREATE TABLE IF NOT EXISTS plan_step_stats (
    plan_signature TEXT NOT NULL,         -- Hash of plan structure (from dag_plan_stats)
    step_position INTEGER NOT NULL,       -- 1, 2, 3... (order in plan)
    node_id INTEGER NOT NULL,             -- Which signature handled this step

    -- Welford's algorithm for success rate tracking
    n INTEGER DEFAULT 0,                  -- Number of observations
    mean_success REAL DEFAULT 0.5,        -- Running mean of success (0=fail, 1=success)
    m2 REAL DEFAULT 0.0,                  -- Sum of squared differences (variance = M2/(N-1))

    -- Metadata
    first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (plan_signature, step_position, node_id),
    FOREIGN KEY (node_id) REFERENCES step_signatures(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pss_plan ON plan_step_stats(plan_signature);
CREATE INDEX IF NOT EXISTS idx_pss_node ON plan_step_stats(node_id);
CREATE INDEX IF NOT EXISTS idx_pss_position ON plan_step_stats(step_position);
CREATE INDEX IF NOT EXISTS idx_pss_mean_success ON plan_step_stats(mean_success);

-- =============================================================================
-- PROPOSED_SIGNATURES: Staging table for new signature candidates
-- =============================================================================
-- Per mycelium-xv09: Signature proposals are staged before acceptance.
-- During cold start (first 20 problems): auto-accept as root children.
-- After cold start: use Welford stats to decide accept/reject/merge.
CREATE TABLE IF NOT EXISTS proposed_signatures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step_text TEXT NOT NULL,
    embedding BLOB,
    graph_embedding BLOB,
    computation_graph TEXT,
    proposed_parent_id INTEGER REFERENCES step_signatures(id),
    best_match_id INTEGER REFERENCES step_signatures(id),
    best_match_sim REAL,
    dsl_hint TEXT,
    extracted_values TEXT,
    status TEXT DEFAULT 'pending',  -- pending, accepted, rejected, merged
    decision_reason TEXT,
    origin_depth INTEGER DEFAULT 0,
    problem_context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    decided_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_proposed_status ON proposed_signatures(status);
CREATE INDEX IF NOT EXISTS idx_proposed_parent ON proposed_signatures(proposed_parent_id);
CREATE INDEX IF NOT EXISTS idx_proposed_best_match ON proposed_signatures(best_match_id);
CREATE INDEX IF NOT EXISTS idx_proposed_created ON proposed_signatures(created_at);
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

    # Add operational_failures if missing (MCTS operational equivalence learning)
    # Tracks how many times a signature produced a different answer than ground truth
    # Per CLAUDE.md: "Record every failure—it feeds the refinement loop"
    if "operational_failures" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN operational_failures INTEGER DEFAULT 0"
        )

    # Add computation_graph if missing (per mycelium-k509)
    # Structural graph representation: MUL(param_0, param_1), ADD(MUL(p0, p1), p2)
    # Per CLAUDE.md: route by what operations DO, not what they SOUND LIKE
    if "computation_graph" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN computation_graph TEXT"
        )

    # Add graph_embedding if missing (per mycelium-k509)
    # Embedding of computation graph for routing (replaces text-based centroid routing)
    if "graph_embedding" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN graph_embedding TEXT"
        )

    # Add embedding variance tracking columns (Welford's algorithm)
    # Per CLAUDE.md: High variance = too generic, should decompose
    if "similarity_count" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN similarity_count INTEGER DEFAULT 0"
        )
    if "similarity_mean" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN similarity_mean REAL DEFAULT 0.0"
        )
    if "similarity_m2" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN similarity_m2 REAL DEFAULT 0.0"
        )

    # Add atomic operations tracking (math primes discovery)
    # Per CLAUDE.md: system discovers which signatures are truly atomic
    if "is_atomic" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN is_atomic INTEGER DEFAULT 0"
        )
    if "atomic_reason" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN atomic_reason TEXT"
        )

    # Add rejection_count for leaf rejection tracking
    # Per CLAUDE.md: leaves reject low-similarity steps, high rejection rate triggers decomposition
    if "rejection_count" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN rejection_count INTEGER DEFAULT 0"
        )

    # Add success similarity tracking for adaptive rejection (Welford's algorithm)
    # Per mycelium-i601: leaves reject when similarity is below historical success distribution
    if "success_sim_count" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN success_sim_count INTEGER DEFAULT 0"
        )
    if "success_sim_mean" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN success_sim_mean REAL DEFAULT 0.0"
        )
    if "success_sim_m2" not in existing_cols:
        migrations.append(
            "ALTER TABLE step_signatures ADD COLUMN success_sim_m2 REAL DEFAULT 0.0"
        )

    # Run step_signatures migrations
    for sql in migrations:
        try:
            conn.execute(sql)
        except Exception as e:
            logger.warning("[schema] Migration failed for '%s': %s", sql[:50], e)

    if migrations:
        conn.commit()

    # Migrate mcts_dag_steps table (add dsl_hint column for step-node stats normalization)
    try:
        cursor = conn.execute("PRAGMA table_info(mcts_dag_steps)")
        dag_step_cols = {row[1] for row in cursor.fetchall()}
        if "dsl_hint" not in dag_step_cols:
            conn.execute("ALTER TABLE mcts_dag_steps ADD COLUMN dsl_hint TEXT")
            conn.commit()
            logger.info("[schema] Added dsl_hint column to mcts_dag_steps")
    except Exception as e:
        # Table might not exist yet (fresh DB)
        logger.debug("[schema] mcts_dag_steps migration skipped: %s", e)

    # Migrate mcts_step_summaries table (add similarity_score for adaptive rejection per mycelium-i601)
    try:
        cursor = conn.execute("PRAGMA table_info(mcts_step_summaries)")
        summary_cols = {row[1] for row in cursor.fetchall()}
        if "similarity_score" not in summary_cols:
            conn.execute("ALTER TABLE mcts_step_summaries ADD COLUMN similarity_score REAL DEFAULT NULL")
            conn.commit()
            logger.info("[schema] Added similarity_score column to mcts_step_summaries")
    except Exception as e:
        # Table might not exist yet (fresh DB)
        logger.debug("[schema] mcts_step_summaries migration skipped: %s", e)

    # Migrate dag_step_node_stats table (add Welford's columns for amplitude_post variance)
    # High variance = inconsistent performance = decomposition signal
    try:
        cursor = conn.execute("PRAGMA table_info(dag_step_node_stats)")
        dsns_cols = {row[1] for row in cursor.fetchall()}
        dsns_migrations = []
        if "amp_post_count" not in dsns_cols:
            dsns_migrations.append(
                "ALTER TABLE dag_step_node_stats ADD COLUMN amp_post_count INTEGER DEFAULT 0"
            )
        if "amp_post_mean" not in dsns_cols:
            dsns_migrations.append(
                "ALTER TABLE dag_step_node_stats ADD COLUMN amp_post_mean REAL DEFAULT 0.0"
            )
        if "amp_post_m2" not in dsns_cols:
            dsns_migrations.append(
                "ALTER TABLE dag_step_node_stats ADD COLUMN amp_post_m2 REAL DEFAULT 0.0"
            )
        for sql in dsns_migrations:
            conn.execute(sql)
        if dsns_migrations:
            conn.commit()
            logger.info("[schema] Added Welford's amplitude columns to dag_step_node_stats")
    except Exception as e:
        # Table might not exist yet (fresh DB)
        logger.debug("[schema] dag_step_node_stats amplitude migration skipped: %s", e)

    # Migrate step_examples table (add expression/inputs for DSL learning per mycelium-nvc9)
    # Per CLAUDE.md: DSL generation needs to see what expressions worked, not just results
    try:
        cursor = conn.execute("PRAGMA table_info(step_examples)")
        example_cols = {row[1] for row in cursor.fetchall()}
        example_migrations = []
        if "expression" not in example_cols:
            example_migrations.append(
                "ALTER TABLE step_examples ADD COLUMN expression TEXT"
            )
        if "inputs" not in example_cols:
            example_migrations.append(
                "ALTER TABLE step_examples ADD COLUMN inputs TEXT"
            )
        for sql in example_migrations:
            conn.execute(sql)
        if example_migrations:
            conn.commit()
            logger.info("[schema] Added expression/inputs columns to step_examples")
    except Exception as e:
        # Table might not exist yet (fresh DB)
        logger.debug("[schema] step_examples migration skipped: %s", e)

    # Migrate dag_step_node_stats table (add Welford's columns for similarity variance)
    # High similarity variance = dag_step_ids routed here have diverse embeddings = type too broad
    try:
        cursor = conn.execute("PRAGMA table_info(dag_step_node_stats)")
        dsns_cols = {row[1] for row in cursor.fetchall()}
        sim_migrations = []
        if "sim_count" not in dsns_cols:
            sim_migrations.append(
                "ALTER TABLE dag_step_node_stats ADD COLUMN sim_count INTEGER DEFAULT 0"
            )
        if "sim_mean" not in dsns_cols:
            sim_migrations.append(
                "ALTER TABLE dag_step_node_stats ADD COLUMN sim_mean REAL DEFAULT 0.0"
            )
        if "sim_m2" not in dsns_cols:
            sim_migrations.append(
                "ALTER TABLE dag_step_node_stats ADD COLUMN sim_m2 REAL DEFAULT 0.0"
            )
        for sql in sim_migrations:
            conn.execute(sql)
        if sim_migrations:
            conn.commit()
            logger.info("[schema] Added Welford's similarity columns to dag_step_node_stats")
    except Exception as e:
        # Table might not exist yet (fresh DB)
        logger.debug("[schema] dag_step_node_stats similarity migration skipped: %s", e)

    # Create welford_stats table if it doesn't exist (per mycelium-bjrf)
    # This table consolidates per-signature stats for tree restructuring decisions
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS welford_stats (
                signature_id INTEGER PRIMARY KEY REFERENCES step_signatures(id) ON DELETE CASCADE,
                route_n INTEGER DEFAULT 0,
                route_mean REAL DEFAULT 0.0,
                route_m2 REAL DEFAULT 0.0,
                child_n INTEGER DEFAULT 0,
                child_mean REAL DEFAULT 0.0,
                child_m2 REAL DEFAULT 0.0,
                exec_n INTEGER DEFAULT 0,
                exec_successes INTEGER DEFAULT 0,
                decomp_attempts INTEGER DEFAULT 0,
                decomp_successes INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_welford_route_n ON welford_stats(route_n)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_welford_exec_n ON welford_stats(exec_n)")
        conn.commit()
        logger.info("[schema] Created welford_stats table")
    except Exception as e:
        # Table already exists or other error
        logger.debug("[schema] welford_stats migration skipped: %s", e)

    # Add decomp columns to welford_stats if missing (data-driven atomic detection)
    try:
        cursor = conn.execute("PRAGMA table_info(welford_stats)")
        welford_cols = {row[1] for row in cursor.fetchall()}
        welford_migrations = []
        if "decomp_attempts" not in welford_cols:
            welford_migrations.append(
                "ALTER TABLE welford_stats ADD COLUMN decomp_attempts INTEGER DEFAULT 0"
            )
        if "decomp_successes" not in welford_cols:
            welford_migrations.append(
                "ALTER TABLE welford_stats ADD COLUMN decomp_successes INTEGER DEFAULT 0"
            )
        for sql in welford_migrations:
            conn.execute(sql)
        if welford_migrations:
            conn.commit()
            logger.info("[schema] Added decomp columns to welford_stats")
    except Exception as e:
        logger.debug("[schema] welford_stats decomp migration skipped: %s", e)

    # Create proposed_signatures table if it doesn't exist (per mycelium-xv09)
    # Staging table for signature proposals before acceptance
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS proposed_signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step_text TEXT NOT NULL,
                embedding BLOB,
                graph_embedding BLOB,
                computation_graph TEXT,
                proposed_parent_id INTEGER REFERENCES step_signatures(id),
                best_match_id INTEGER REFERENCES step_signatures(id),
                best_match_sim REAL,
                dsl_hint TEXT,
                extracted_values TEXT,
                status TEXT DEFAULT 'pending',
                decision_reason TEXT,
                origin_depth INTEGER DEFAULT 0,
                problem_context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                decided_at TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proposed_status ON proposed_signatures(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proposed_parent ON proposed_signatures(proposed_parent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proposed_created ON proposed_signatures(created_at)")
        conn.commit()
        logger.info("[schema] Created proposed_signatures table")
    except Exception as e:
        # Table already exists or other error
        logger.debug("[schema] proposed_signatures migration skipped: %s", e)

    # Create plan_step_stats table if it doesn't exist (statistical blame accumulation)
    # Per CLAUDE.md: "Failures Are Valuable Data Points" - accumulate blame statistically
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS plan_step_stats (
                plan_signature TEXT NOT NULL,
                step_position INTEGER NOT NULL,
                node_id INTEGER NOT NULL,
                n INTEGER DEFAULT 0,
                mean_success REAL DEFAULT 0.5,
                m2 REAL DEFAULT 0.0,
                first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (plan_signature, step_position, node_id),
                FOREIGN KEY (node_id) REFERENCES step_signatures(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pss_plan ON plan_step_stats(plan_signature)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pss_node ON plan_step_stats(node_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pss_position ON plan_step_stats(step_position)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pss_mean_success ON plan_step_stats(mean_success)")
        conn.commit()
        logger.info("[schema] Created plan_step_stats table")
    except Exception as e:
        # Table already exists or other error
        logger.debug("[schema] plan_step_stats migration skipped: %s", e)

    # Create ucb1_gap_stats table (per mycelium-02nn: Welford-guided exploration)
    # Tracks UCB1 gap values that led to correct vs incorrect routing decisions
    # Used to compute adaptive gap threshold: threshold = mean - k * std
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ucb1_gap_stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row

                -- Gaps from successful routing decisions
                success_n INTEGER DEFAULT 0,
                success_mean REAL DEFAULT 0.0,
                success_m2 REAL DEFAULT 0.0,

                -- Gaps from failed routing decisions
                failure_n INTEGER DEFAULT 0,
                failure_mean REAL DEFAULT 0.0,
                failure_m2 REAL DEFAULT 0.0,

                -- Combined stats (for overall threshold)
                total_n INTEGER DEFAULT 0,
                total_mean REAL DEFAULT 0.0,
                total_m2 REAL DEFAULT 0.0,

                -- Metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Initialize singleton row
        conn.execute("""
            INSERT OR IGNORE INTO ucb1_gap_stats (id) VALUES (1)
        """)
        conn.commit()
        logger.info("[schema] Created ucb1_gap_stats table")
    except Exception as e:
        logger.debug("[schema] ucb1_gap_stats migration skipped: %s", e)

    # Create reactive_exploration_stats table (per mycelium-02nn enhancement)
    # Tracks reactive exploration outcomes for Welford-adaptive multipliers
    # Per CLAUDE.md "The Flow": DB Statistics → Welford → Tree Structure
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reactive_exploration_stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row

                -- Welford stats for reactive exploration success rate
                -- Tracks whether reactive exploration finds winning paths
                n INTEGER DEFAULT 0,                  -- Total reactive explorations
                success_mean REAL DEFAULT 0.0,        -- Mean success rate (0-1)
                success_m2 REAL DEFAULT 0.0,          -- Welford M2 for variance

                -- Welford stats for gap multiplier effectiveness
                -- Tracks what gap_mult values led to success
                gap_mult_n INTEGER DEFAULT 0,
                gap_mult_mean REAL DEFAULT 2.0,       -- Mean effective gap multiplier
                gap_mult_m2 REAL DEFAULT 0.0,

                -- Welford stats for budget multiplier effectiveness
                budget_mult_n INTEGER DEFAULT 0,
                budget_mult_mean REAL DEFAULT 1.5,    -- Mean effective budget multiplier
                budget_mult_m2 REAL DEFAULT 0.0,

                -- Metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Initialize singleton row
        conn.execute("""
            INSERT OR IGNORE INTO reactive_exploration_stats (id) VALUES (1)
        """)
        conn.commit()
        logger.info("[schema] Created reactive_exploration_stats table")
    except Exception as e:
        logger.debug("[schema] reactive_exploration_stats migration skipped: %s", e)

    # Create pending_embedding_drifts table (per mycelium-ieq4)
    # Accumulates successful dag_step embeddings for batch drift updates
    # Per CLAUDE.md: "High-traffic signatures become semantic attractors"
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_embedding_drifts (
                signature_id INTEGER PRIMARY KEY,

                -- Accumulated embedding sum (JSON array of floats)
                -- We sum embeddings, then divide by count for average
                embedding_sum TEXT NOT NULL,

                -- Number of successful matches accumulated
                success_count INTEGER DEFAULT 0,

                -- Metadata
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (signature_id) REFERENCES step_signatures(id)
            )
        """)
        conn.commit()
        logger.info("[schema] Created pending_embedding_drifts table")
    except Exception as e:
        logger.debug("[schema] pending_embedding_drifts migration skipped: %s", e)

    # Add new indexes (safe to run multiple times)
    index_migrations = [
        "CREATE INDEX IF NOT EXISTS idx_sig_is_root ON step_signatures(is_root)",
        "CREATE INDEX IF NOT EXISTS idx_sig_dsl_type ON step_signatures(dsl_type)",
        "CREATE INDEX IF NOT EXISTS idx_sig_umbrella_archived ON step_signatures(is_semantic_umbrella, is_archived)",
        # Performance indexes (added for query optimization)
        "CREATE INDEX IF NOT EXISTS idx_sig_archived_created ON step_signatures(is_archived, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_usage_sig_created ON step_usage_log(signature_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_failures_created_sig ON step_failures(created_at, signature_id)",
        # Computation graph routing index (per mycelium-k509)
        "CREATE INDEX IF NOT EXISTS idx_sig_graph_embedding ON step_signatures(graph_embedding)",
        # Atomic operations index (math primes discovery)
        "CREATE INDEX IF NOT EXISTS idx_sig_is_atomic ON step_signatures(is_atomic)",
        # MCTS query optimization indexes
        "CREATE INDEX IF NOT EXISTS idx_mcts_dag_steps_dsl_hint ON mcts_dag_steps(dsl_hint)",
        "CREATE INDEX IF NOT EXISTS idx_mcts_step_summaries_success ON mcts_step_summaries(step_success)",
        "CREATE INDEX IF NOT EXISTS idx_proposed_best_match ON proposed_signatures(best_match_id)",
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
