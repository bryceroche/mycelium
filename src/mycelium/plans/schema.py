"""Database schema for recursive DAG of DAGs.

Supports unlimited nesting:
- Plans contain Steps
- Steps can contain sub-Plans (composite steps)
- sub-Plans contain more Steps, which can contain more sub-Plans...

The key relationship:
    Plan --has-many--> Step --has-one--> sub-Plan (optional)

A Step is:
- ATOMIC if no sub-plan exists for it
- COMPOSITE if a sub-plan exists with parent_step_id pointing to it
"""

PLANS_SCHEMA = """
-- =============================================================================
-- PLANS: Containers for step DAGs (can be nested inside steps)
-- =============================================================================
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    problem TEXT NOT NULL,

    -- Recursive nesting: which step contains this plan?
    -- NULL = root plan (top-level)
    -- non-NULL = sub-plan inside a composite step
    parent_step_id TEXT,

    -- Depth tracking for debugging/limits
    depth INTEGER DEFAULT 0,

    -- Execution state
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
    result TEXT,

    -- Timestamps
    created_at TEXT NOT NULL,
    completed_at TEXT,

    FOREIGN KEY (parent_step_id) REFERENCES steps(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_plans_parent ON plans(parent_step_id);
CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status);

-- =============================================================================
-- STEPS: Nodes in the DAG
-- =============================================================================
-- A step is either:
--   ATOMIC: Executed directly (no sub-plan)
--   COMPOSITE: Contains a sub-plan that must be executed first
--
-- To check if a step is composite:
--   SELECT EXISTS(SELECT 1 FROM plans WHERE parent_step_id = step.id)

CREATE TABLE IF NOT EXISTS steps (
    id TEXT PRIMARY KEY,

    -- Which plan contains this step
    plan_id TEXT NOT NULL,

    -- Local identifier within the plan (e.g., "step_1", "final")
    local_id TEXT NOT NULL,

    -- What this step does
    task TEXT NOT NULL,

    -- Execution state
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
    result TEXT,
    success INTEGER DEFAULT 0,

    -- Value extraction: JSON object {semantic_name: value}
    extracted_values TEXT,

    -- Optional DSL hint for signature matching
    dsl_hint TEXT,

    -- Timestamps
    created_at TEXT NOT NULL,
    completed_at TEXT,

    UNIQUE(plan_id, local_id),
    FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_steps_plan ON steps(plan_id);
CREATE INDEX IF NOT EXISTS idx_steps_status ON steps(status);

-- =============================================================================
-- STEP_DEPENDENCIES: Edges in the DAG (within a plan)
-- =============================================================================
-- Models: step depends_on another step
-- Both steps must be in the same plan

CREATE TABLE IF NOT EXISTS step_dependencies (
    step_id TEXT NOT NULL,
    depends_on_id TEXT NOT NULL,

    PRIMARY KEY (step_id, depends_on_id),
    FOREIGN KEY (step_id) REFERENCES steps(id) ON DELETE CASCADE,
    FOREIGN KEY (depends_on_id) REFERENCES steps(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_deps_step ON step_dependencies(step_id);
CREATE INDEX IF NOT EXISTS idx_deps_depends ON step_dependencies(depends_on_id);

-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

-- View: Steps with their sub-plan (if any)
CREATE VIEW IF NOT EXISTS steps_with_subplan AS
SELECT
    s.*,
    p.id AS sub_plan_id,
    p.status AS sub_plan_status,
    CASE WHEN p.id IS NOT NULL THEN 1 ELSE 0 END AS is_composite
FROM steps s
LEFT JOIN plans p ON p.parent_step_id = s.id;

-- View: Plan hierarchy (for debugging)
CREATE VIEW IF NOT EXISTS plan_hierarchy AS
WITH RECURSIVE hierarchy AS (
    -- Base case: root plans
    SELECT
        id,
        problem,
        parent_step_id,
        depth,
        status,
        id AS root_plan_id,
        problem AS path
    FROM plans
    WHERE parent_step_id IS NULL

    UNION ALL

    -- Recursive case: sub-plans
    SELECT
        p.id,
        p.problem,
        p.parent_step_id,
        p.depth,
        p.status,
        h.root_plan_id,
        h.path || ' -> ' || s.local_id || ' -> ' || p.problem
    FROM plans p
    JOIN steps s ON p.parent_step_id = s.id
    JOIN hierarchy h ON s.plan_id = h.id
)
SELECT * FROM hierarchy;
"""


def get_schema() -> str:
    return PLANS_SCHEMA


def init_db(conn) -> None:
    """Initialize the plans database schema."""
    conn.executescript(PLANS_SCHEMA)
    conn.commit()
