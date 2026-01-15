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
# Cold-start aware: higher threshold early (more branching), lower later (consolidation)
MIN_MATCH_THRESHOLD = 0.85  # Mature threshold - reduce signature fragmentation
MIN_MATCH_THRESHOLD_COLD_START = 0.92  # Cold start threshold - create more signatures
MIN_MATCH_RAMP_SIGNATURES = 50  # Signatures needed to reach mature threshold
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

# DSL Operation Inference threshold (cold-start aware)
# Ramps from COLD_START to MATURE as signature count grows
# Cold start: try more DSLs to bootstrap learning
# Mature: be selective, use proven paths
DSL_OPERATION_INFERENCE_COLD_START = 0.35  # Low threshold when DB is empty
DSL_OPERATION_INFERENCE_MATURE = 0.60  # High threshold when DB is mature
DSL_OPERATION_INFERENCE_RAMP_SIGS = 100  # Signatures needed to reach mature threshold

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
# MCTS ROUTING (UCB1-based exploration/exploitation)
# =============================================================================
# Uses UCB1 formula: score = exploit + C * sqrt(ln(N) / n)
# Where exploit = similarity * success_rate, N = parent visits, n = child visits
# This balances trying known-good paths vs exploring under-visited ones

MCTS_EXPLORATION_C = 1.0  # Exploration constant (higher = more exploration)
                          # sqrt(2) ≈ 1.41 is theoretical optimal, 1.0 is more conservative
MCTS_SIMILARITY_WEIGHT = 0.7  # Weight for semantic similarity in exploitation term
MCTS_SUCCESS_WEIGHT = 0.3  # Weight for success rate in exploitation term
MCTS_MIN_VISITS_FOR_UCB = 1  # Min visits before UCB exploration bonus applies

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
# DECAY LIFECYCLE
# =============================================================================
# Slow decay system for signature lifecycle management.
# Signatures that don't pull their weight gradually fade out.
# Per CLAUDE.md: "slow decay: sig_uses / total_problems"

DECAY_ENABLED = True  # Enable decay lifecycle management
DECAY_CHECK_INTERVAL_SEC = 300  # Check decay every 5 minutes
DECAY_MIN_AGE_DAYS = 7  # Don't decay signatures younger than N days

# Thresholds (as fraction of TRAFFIC_MIN_SHARE)
DECAY_ARCHIVE_THRESHOLD = 0.05  # Archive if < 5% of min threshold
DECAY_DEMOTE_THRESHOLD = 0.20  # Demote umbrella if < 20% of min threshold
DECAY_WARNING_THRESHOLD = 0.50  # Warn if < 50% of min threshold
DECAY_RECOVERY_THRESHOLD = 0.80  # Recovered if back to 80% of min threshold

# Grace periods
DECAY_ARCHIVE_GRACE_DAYS = 30  # Wait 30 days before archiving

# Limits
DECAY_MAX_ACTIONS_PER_RUN = 10  # Max signatures to act on per cycle

# =============================================================================
# EMBEDDING CACHE
# =============================================================================
# Two-tier cache for MathBERT embeddings (expensive to compute ~50ms each).
# L1: In-memory LRU, L2: Persistent SQLite disk cache.

EMBEDDING_CACHE_ENABLED = True  # Enable embedding caching
EMBEDDING_CACHE_MEMORY_SIZE = 10000  # Max entries in memory LRU cache
EMBEDDING_CACHE_PERSIST = True  # Enable disk persistence (SQLite)
EMBEDDING_CACHE_WARM_ON_START = True  # Pre-load from signatures on startup
EMBEDDING_CACHE_TTL_DAYS = 30  # Prune disk entries older than N days

# =============================================================================
# DEPTH-AWARE DECOMPOSITION
# =============================================================================
# Force decomposition at shallow depths to build out the tree structure.
# Shallow = routing/categorization, Deep = execution
#
# Decompose probability decays exponentially with depth:
#   P(decompose) = 1.0 if depth <= FORCE_DECOMPOSE_DEPTH
#   P(decompose) = DECAY_BASE ^ (depth - FORCE_DECOMPOSE_DEPTH) for deeper
#
# Example with FORCE_DECOMPOSE_DEPTH=5, DECAY_BASE=0.5:
#   depth 0-5: 100% decompose (forced)
#   depth 6: 50% decompose
#   depth 7: 25% decompose
#   depth 8: 12.5% decompose
#   depth 9+: ~0% decompose (execute)

DEPTH_DECOMPOSE_ENABLED = True  # Enable depth-aware forced decomposition
DEPTH_FORCE_DECOMPOSE_DEPTH = 5  # Always decompose at depth 0-5
DEPTH_DECOMPOSE_DECAY_BASE = 0.5  # Decay rate per depth beyond force threshold
DEPTH_DECOMPOSE_MIN_PROB = 0.05  # Floor probability (never fully disable decompose option)

# =============================================================================
# DYNAMIC DEPTH ROUTING
# =============================================================================

# BIG BANG EXPANSION: Recursive decomposition during cold start
# When enabled: aggressively decompose signatures to rapidly build tree structure
# When disabled: only decompose on explicit failure, use existing tree
BIG_BANG_EXPANSION_ENABLED = False  # Toggle on for aggressive cold-start decomposition

RECURSIVE_DECOMPOSITION_ENABLED = True  # Enable decomposition for complex steps
RECURSIVE_MAX_DEPTH = 9  # Max routing depth: deep decomposition for complex problems
RECURSIVE_CONFIDENCE_THRESHOLD = 0.8  # Route deeper when DSL confidence < this
RECURSIVE_MAX_TOTAL_STEPS = 50

# Umbrella routing depth limits
_UMBRELLA_MAX_DEPTH_RAW = 10  # Configurable max depth for umbrella routing chains
_UMBRELLA_HARD_CAP = 100  # Absolute maximum to prevent unbounded recursion
# Validate and clamp: ensure positive integer, capped at hard limit
UMBRELLA_MAX_DEPTH = max(1, min(int(_UMBRELLA_MAX_DEPTH_RAW or 10), _UMBRELLA_HARD_CAP))
UMBRELLA_ROUTING_THRESHOLD = 0.5  # Min similarity for umbrella child routing (lower than global 0.85 since we're picking best among known children)

# =============================================================================
# ZERO-LLM ROUTING (Skip planner for mature signatures)
# =============================================================================
# When enabled, the solver will attempt to route problems directly through
# the signature tree without calling the planner. Only works for mature
# signatures with high success rates and working DSL scripts.

ZERO_LLM_ROUTING_ENABLED = True  # Master switch for zero-LLM routing
ZERO_LLM_MIN_SIMILARITY = 0.90  # High similarity required (stricter than normal 0.85)
ZERO_LLM_MIN_SUCCESS_RATE = 0.70  # Signature must have >= 70% success rate
ZERO_LLM_MIN_USES = 5  # Need enough data to trust the signature
ZERO_LLM_REQUIRE_DSL = True  # Signature must have a working DSL script

# =============================================================================
# DSL AUTO-REWRITER (Fix underperforming DSLs automatically)
# =============================================================================
# When a signature has low success rate but high traffic, the rewriter
# uses LLM to generate an improved DSL script.
# Per CLAUDE.md: "rewrite DSL if centroid avg outside confidence bounds"

DSL_REWRITER_ENABLED = True  # Master switch for auto-rewriting
DSL_REWRITER_MIN_USES = 10  # Need enough data to identify failure patterns
DSL_REWRITER_MAX_SUCCESS_RATE = 0.40  # Rewrite if success rate below this
DSL_REWRITER_MIN_TRAFFIC_SHARE = 0.005  # Only rewrite high-traffic sigs (0.5%)
DSL_REWRITER_COOLDOWN_HOURS = 24  # Don't rewrite same sig within this period

# =============================================================================
# DATABASE
# =============================================================================

DB_PATH = "mycelium.db"  # 768-dim MathBERT embeddings

# =============================================================================
# LLM CLIENT (OpenAI gpt-4.1-nano)
# =============================================================================

CLIENT_DEFAULT_TIMEOUT = 120.0
CLIENT_DEFAULT_TEMPERATURE = 0.0  # Zero for deterministic responses
CLIENT_CONNECT_TIMEOUT = 10.0
CLIENT_BASE_RETRY_DELAY = 1.0
CLIENT_MAX_RETRY_DELAY = 30.0
PLANNER_DEFAULT_TEMPERATURE = 0.0  # Zero for deterministic decomposition

# Model - OpenAI gpt-4.1-nano only
DEFAULT_MODEL = "gpt-4.1-nano"
PLANNER_DEFAULT_MODEL = DEFAULT_MODEL
SOLVER_DEFAULT_MODEL = DEFAULT_MODEL

# =============================================================================
# SELF-CONSISTENCY (Reliability through sampling)
# =============================================================================

SELF_CONSISTENCY_ENABLED = False  # Disabled for V2 simplicity (enable later)
SELF_CONSISTENCY_SAMPLES = 3
SELF_CONSISTENCY_TEMPERATURE = 0.5
