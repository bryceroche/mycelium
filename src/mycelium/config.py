"""Mycelium configuration constants.

Single operating mode: explore signatures, collect lift data, learn from failures.

=============================================================================
CORE PRINCIPLE: FAILURES ARE DATA
=============================================================================
The system learns by failing. Every DSL failure is recorded and used to:
- Identify signatures that need decomposition (high variance = umbrella)
- Find DSLs that need fixing (partial success = fixable)
- Route future problems away from known-bad paths

DO NOT mask failures with LLM fallback. Let DSLs fail, record the outcome,
and let the refinement loop fix them. Short-term accuracy matters less than
long-term learning.

Config philosophy:
- TRAINING_MODE=True: Collect ALL data, even from known-bad DSLs
- All thresholds at 0.0: Let everything execute and fail naturally
- No gates, no guards: Pure data collection mode
=============================================================================
"""

import os

# TRAINING_MODE = True: DSL avoidance DISABLED, collect all failure data
# TRAINING_MODE = False: DSL avoidance ENABLED, skip known-bad DSLs
# Keep True during data collection phase
TRAINING_MODE = True

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
DSL_PROBATION_ENABLED = False  # Inject on every signature hit
DSL_MIN_CONFIDENCE = 0.0  # Try DSL regardless of confidence
DSL_TIMEOUT_SEC = 1.0
DSL_SEMANTIC_MIN_CONFIDENCE = 0.0  # Disabled: let DSLs execute and fail naturally
DSL_SEMANTIC_GATE_THRESHOLD = 0.0  # Disabled: semantic gate off (strict DAG mode)
DSL_PARAM_SEMANTIC_THRESHOLD = 0.0  # Disabled: let param mapping try and fail

# Per-DSL-type thresholds - ALL DISABLED (strict DAG mode - no LLM fallback)
DSL_THRESHOLDS_BY_TYPE = {
    "power": {"gate": 0.0, "param": 0.0},
    "geometry": {"gate": 0.0, "param": 0.0},
    "combinatorics": {"gate": 0.0, "param": 0.0},
    "arithmetic": {"gate": 0.0, "param": 0.0},
    "division": {"gate": 0.0, "param": 0.0},
    "default": {"gate": 0.0, "param": 0.0},
}

# DSL Executor thresholds
DSL_VALUE_TYPE_THRESHOLD = 0.15  # Threshold for value type matching
DSL_STEP_TYPE_ALIGNMENT_THRESHOLD = 0.20  # Threshold for step type alignment
DSL_PARAM_MATCH_THRESHOLD = 0.50  # Min score to accept param match
DSL_PARAM_EXACT_MATCH_SCORE = 0.95  # Score for exact param name match
DSL_GENERATOR_MIN_SUCCESS_RATE = 0.80  # Min success rate for DSL generation

# Negative example threshold
NEGATIVE_EXAMPLE_THRESHOLD = 0.85  # Similarity threshold for negative examples

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
# AUTO-DEMOTION (complex DSLs → umbrellas)
# =============================================================================
# When a DSL fails repeatedly, auto-promote to umbrella so it decomposes instead
# Uses adaptive thresholds: branch fast in cold start, slow down when mature

AUTO_DEMOTE_ENABLED = True  # Enable auto-demotion of failing DSLs
AUTO_DEMOTE_MAX_SUCCESS_RATE = 0.20  # Demote if success rate below this
AUTO_DEMOTE_EXCLUDED_TYPES = ["decompose"]  # Don't demote these DSL types

# Graduated MIN_USES: branch fast early, slow down as DB matures
# Formula: min_uses = 1 + (sig_count // RAMP_DIVISOR), capped at MAX
# Centroid averaging stabilizes good paths, so we can branch aggressively
AUTO_DEMOTE_RAMP_DIVISOR = 2000  # Every 2000 sigs, add 1 to MIN_USES (branch fast!)
AUTO_DEMOTE_MIN_USES_FLOOR = 1   # Start at 1 (branch on first failure)
AUTO_DEMOTE_MIN_USES_CAP = 5     # Never require more than 5 failures

# =============================================================================
# ROUTING SCORE FORMULA
# =============================================================================

# Score = ROUTING_SIM_WEIGHT * cosine_sim + ROUTING_SUCCESS_WEIGHT * effective_rate
ROUTING_SIM_WEIGHT = 0.85  # Weight for cosine similarity
ROUTING_SUCCESS_WEIGHT = 0.15  # Weight for success rate

# Bayesian prior for cold start (assume some successes before any data)
ROUTING_PRIOR_SUCCESSES = 2
ROUTING_PRIOR_USES = 4

# Parent credit propagation (reward successful routers)
PARENT_CREDIT_DECAY = 0.7  # Credit multiplier per depth (0.7^1=0.7, 0.7^2=0.49, 0.7^3=0.34...)
PARENT_CREDIT_MAX_DEPTH = 5  # Max depth to propagate (prevents infinite loops)
PARENT_CREDIT_MIN = 0.1  # Minimum credit to apply (filter noise)

# Staleness decay (deprioritize signatures that haven't been used recently)
# Penalty = min(days_since_use * STALENESS_DECAY_RATE, STALENESS_MAX_PENALTY)
STALENESS_DECAY_ENABLED = True  # Enable staleness penalty in routing
STALENESS_DECAY_RATE = 0.02  # Penalty per day of inactivity (0.02 = 2% per day)
STALENESS_MAX_PENALTY = 0.15  # Cap penalty at 15% (don't kill old sigs completely)
STALENESS_GRACE_DAYS = 1.0  # No penalty for first N days (let new sigs settle)

# Usage-based decay (deprioritize low-traffic signatures)
# traffic_share = sig_uses / total_problems_solved
# Penalty applied when traffic_share < TRAFFIC_MIN_SHARE
TRAFFIC_DECAY_ENABLED = True  # Enable traffic share penalty
TRAFFIC_MIN_SHARE = 0.01  # Min traffic share before penalty (1% of total runs)
TRAFFIC_DECAY_RATE = 0.10  # Penalty for very low traffic sigs
TRAFFIC_CACHE_TTL = 60.0  # Cache total_problems for N seconds (avoid DB hits)
TRAFFIC_GRACE_PROBLEMS = 50  # No penalty until system has run N problems

# =============================================================================
# DYNAMIC DEPTH ROUTING
# =============================================================================

RECURSIVE_DECOMPOSITION_ENABLED = True  # Enable decomposition for complex steps
RECURSIVE_MAX_DEPTH = 9  # Max routing depth: deep decomposition for complex problems
RECURSIVE_CONFIDENCE_THRESHOLD = 0.8  # Route deeper when DSL confidence < this
RECURSIVE_MAX_TOTAL_STEPS = 50
UMBRELLA_MAX_DEPTH = 10  # Max depth for umbrella routing chains
UMBRELLA_ROUTING_THRESHOLD = 0.5  # Min similarity for umbrella child routing (lower than global 0.85 since we're picking best among known children)

# =============================================================================
# DATABASE
# =============================================================================

DB_PATH = "mycelium.db"  # 768-dim MathBERT embeddings

# =============================================================================
# LLM CLIENT
# =============================================================================

# Provider: "groq" or "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "openai"

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
