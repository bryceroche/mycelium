"""Mycelium configuration constants.

Single operating mode: explore signatures, collect lift data, learn from failures.
"""

# For backwards compatibility
# TRAINING_MODE = True means DSL avoidance is DISABLED (collect all data)
# TRAINING_MODE = False means DSL avoidance is ENABLED (skip bad DSLs)
TRAINING_MODE = True  # TRAINING: Disable DSL avoidance, collect all failure data

# =============================================================================
# SIGNATURE MATCHING
# =============================================================================

# Similarity Thresholds (adjusted for 768-dim MathBERT embeddings)
MIN_MATCH_THRESHOLD = 0.85  # Lowered to reduce signature fragmentation
MERGE_SIMILARITY_THRESHOLD = 0.75
VARIANT_THRESHOLD = 0.40
DEFAULT_INJECTION_THRESHOLD = 0.90  # Only inject on high-confidence matches
PIPELINE_MIN_SIMILARITY = 0.85  # Match threshold for pipeline
NEGATIVE_LIFT_SIMILARITY = 0.55

# =============================================================================
# DSL INJECTION
# =============================================================================

FORCE_INJECTION = True  # Always inject when signature matches (bypass lift checks)
MANDATE_DSL = True  # Mandate DSL execution - no LLM fallback allowed
DSL_PROBATION_ENABLED = False  # Inject on every signature hit
DSL_MIN_CONFIDENCE = 0.0  # Try DSL regardless of confidence
DSL_LLM_THRESHOLD = 1.0  # Always try DSL before LLM
DSL_TIMEOUT_SEC = 1.0
DSL_SEMANTIC_MIN_CONFIDENCE = 0.15  # Very permissive: let system try more DSLs
DSL_SEMANTIC_GATE_THRESHOLD = 0.25  # Lowered from 0.50: breathe, learn from failures
DSL_PARAM_SEMANTIC_THRESHOLD = 0.12  # Lowered from 0.25: allow looser param matches

# Per-DSL-type thresholds (very relaxed for exploration - failures teach us what needs decomposition)
DSL_THRESHOLDS_BY_TYPE = {
    # Power/exponent - relaxed for learning
    "power": {"gate": 0.30, "param": 0.15},
    # Geometry/distance - relaxed for learning
    "geometry": {"gate": 0.25, "param": 0.15},
    # Combinatorics - relaxed for learning
    "combinatorics": {"gate": 0.25, "param": 0.15},
    # Simple arithmetic - very lenient
    "arithmetic": {"gate": 0.20, "param": 0.12},
    # Division/modulo - relaxed for learning
    "division": {"gate": 0.25, "param": 0.12},
    # Default fallback - very relaxed
    "default": {"gate": 0.25, "param": 0.12},
}

# =============================================================================
# EXPLORATION
# =============================================================================

EXPLORATION_RATE = 1.0  # Always explore
EXPLORATION_UNPROVEN_RATE = 1.0  # Try unproven signatures
EXPLORATION_MIN_LIFT = 0.0
EXPLORATION_MIN_CONFIDENCE = 0.0
USAGE_CONFIDENCE_DECAY = 5.0
COLD_START_GUARANTEED_USES = 100  # Bootstrap new signatures
PROBATION_INJECTION_RATE = 0.3
AVOID_CACHE_TTL = 300.0  # Rebuild negative-lift cache every 5 min

# =============================================================================
# RELIABILITY THRESHOLDS
# =============================================================================

RELIABILITY_MIN_USES = 3
RELIABILITY_MIN_SUCCESS_RATE = 0.70

# =============================================================================
# DYNAMIC DEPTH ROUTING
# =============================================================================

RECURSIVE_DECOMPOSITION_ENABLED = True  # Enable decomposition for complex steps
RECURSIVE_MAX_DEPTH = 9  # Max routing depth: deep decomposition for complex problems
RECURSIVE_CONFIDENCE_THRESHOLD = 0.8  # Route deeper when DSL confidence < this
RECURSIVE_MAX_TOTAL_STEPS = 50

# =============================================================================
# DATABASE
# =============================================================================

DB_PATH = "mycelium.db"  # 768-dim MathBERT embeddings

# =============================================================================
# LLM CLIENT
# =============================================================================

# Provider: "groq" or "openai"
LLM_PROVIDER = "groq"

CLIENT_DEFAULT_TIMEOUT = 120.0
CLIENT_DEFAULT_TEMPERATURE = 0.0  # Zero for deterministic responses
CLIENT_CONNECT_TIMEOUT = 10.0
CLIENT_BASE_RETRY_DELAY = 1.0
CLIENT_MAX_RETRY_DELAY = 30.0
PLANNER_DEFAULT_TEMPERATURE = 0.0  # Zero for deterministic decomposition

# Model names per provider
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

# Active model (based on provider)
PLANNER_DEFAULT_MODEL = OPENAI_DEFAULT_MODEL if LLM_PROVIDER == "openai" else GROQ_DEFAULT_MODEL
SOLVER_DEFAULT_MODEL = OPENAI_DEFAULT_MODEL if LLM_PROVIDER == "openai" else GROQ_DEFAULT_MODEL

# =============================================================================
# SELF-CONSISTENCY (Reliability through sampling)
# =============================================================================

SELF_CONSISTENCY_ENABLED = False  # Disabled for V2 simplicity (enable later)
SELF_CONSISTENCY_SAMPLES = 3
SELF_CONSISTENCY_TEMPERATURE = 0.5
