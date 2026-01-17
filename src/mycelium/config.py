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
# Set via environment variable or CLI: MYCELIUM_TRAINING_MODE=true/false
TRAINING_MODE = os.getenv("MYCELIUM_TRAINING_MODE", "true").lower() in ("true", "1", "yes")

# =============================================================================
# SIGNATURE MATCHING
# =============================================================================

# Similarity Thresholds (adjusted for 768-dim embeddings)
# Cold-start aware: higher threshold early (more branching), lower later (consolidation)
MIN_MATCH_THRESHOLD = 0.85  # Mature threshold - reduce signature fragmentation
MIN_MATCH_THRESHOLD_COLD_START = 0.92  # Cold start threshold - create more signatures
MIN_MATCH_RAMP_SIGNATURES = 50  # Signatures needed to reach mature threshold

# =============================================================================
# DSL EXECUTION
# =============================================================================

DSL_TIMEOUT_SEC = 1.0  # Timeout for DSL script execution

# DSL Operation Inference threshold (cold-start aware)
# Ramps from COLD_START to MATURE as signature count grows
# Cold start: try more DSLs to bootstrap learning
# Mature: be selective, use proven paths
DSL_OPERATION_INFERENCE_COLD_START = 0.35  # Low threshold when DB is empty
DSL_OPERATION_INFERENCE_MATURE = 0.60  # High threshold when DB is mature
DSL_OPERATION_INFERENCE_RAMP_SIGS = 100  # Signatures needed to reach mature threshold

# DSL exotic operation thresholds
# Exotic operations (**, factorial, perm, etc.) require higher confidence than basic (+, -, *, /)
DSL_EXOTIC_THRESHOLD_BONUS = 0.05  # Require 5% higher similarity for exotic ops
DSL_EXOTIC_THRESHOLD_MAX = 0.92  # Cap exotic threshold to ensure it's achievable

# =============================================================================
# UMBRELLA PROMOTION (failing signatures → decompose into children)
# =============================================================================
# Per CLAUDE.md: "Failing signatures get decomposed"
# Smart decomposition: give signatures a few chances, decompose if mostly failing

UMBRELLA_MIN_USES_FOR_EVALUATION = 3  # Need this many attempts before evaluating
UMBRELLA_MAX_SUCCESS_RATE_FOR_DECOMPOSITION = 0.5  # Decompose if failing more than succeeding
# Example: 2 failures out of 3 = 33% success → decompose
# Example: 20 successes + 2 failures = 91% success → keep

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

# =============================================================================
# ADAPTIVE MCTS (exploration weight and split threshold tied to accuracy)
# =============================================================================
# Per CLAUDE.md: "cold-start aware thresholds (adaptive branching more aggressive during cold start)"
# Low accuracy → high exploration, lenient splits
# High accuracy → low exploration, strict splits

ADAPTIVE_ACCURACY_WINDOW_SIZE = 100  # Rolling window for accuracy calculation

# Exploration weight (C) range: interpolates based on accuracy
ADAPTIVE_EXPLORATION_C_MAX = 2.0  # Cold start: explore aggressively
ADAPTIVE_EXPLORATION_C_MIN = 0.5  # Mature: mostly exploit

# Split threshold (failure rate) range: interpolates based on accuracy
ADAPTIVE_SPLIT_THRESHOLD_LENIENT = 0.7  # Cold start: tolerate 70% failure before split
ADAPTIVE_SPLIT_THRESHOLD_STRICT = 0.4   # Mature: split at 40% failure

# =============================================================================
# MCTS COMPUTE BUDGET (multi-path exploration)
# =============================================================================
# Budget controls how many paths to explore during routing.
# - 1.0 = single best path (default, backward compatible)
# - 2.0 = explore up to 2 paths at low-confidence nodes
# - 3.0+ = explore multiple paths (training mode)

COMPUTE_BUDGET_DEFAULT = 1.0  # Default: single-pass (backward compatible)
COMPUTE_BUDGET_TRAINING = 3.0  # Training: explore 3 paths for cluster splitting
COMPUTE_BUDGET_CONFIDENCE_THRESHOLD = 0.5  # Explore alternatives when confidence < this

# Bayesian prior for cold start (assume some successes before any data)
ROUTING_PRIOR_SUCCESSES = 2
ROUTING_PRIOR_USES = 4

# Parent credit propagation (reward successful routers)
# Per CLAUDE.md: "Parent umbrellas get decay^depth credit (default 0.5 per level)"
PARENT_CREDIT_DECAY = 0.5  # Credit multiplier per depth (0.5^1=0.5, 0.5^2=0.25, 0.5^3=0.125)
PARENT_CREDIT_MAX_DEPTH = 3  # Max depth to propagate (per CLAUDE.md: "default 3 levels")
PARENT_CREDIT_MIN = 0.1  # Minimum credit to apply (filter noise)

# Centroid propagation (batch update parent centroids)
CENTROID_PROPAGATION_MAX_DEPTH = 3  # Max levels to propagate centroid changes (perf optimization)
CENTROID_PROPAGATION_BATCH_SIZE = 5  # Accumulate N matches before propagating (perf vs freshness tradeoff)

# Centroid drift bounds (reject updates that would move centroid too far)
# This prevents a signature from drifting outside its semantic confidence bounds.
# Max drift decreases as embedding_count increases (more examples = more stable).
CENTROID_MAX_DRIFT = 0.15  # Max cosine distance allowed for centroid drift
CENTROID_DRIFT_DECAY = 0.9  # Drift threshold multiplier per log2(count) - tightens with more examples

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
# EMBEDDING MODEL
# =============================================================================
# Supports multiple embedding backends:
# - "gemini-embedding-001": Google Vertex AI flagship (3072-dim, state-of-the-art)
# - "text-embedding-3-large": OpenAI's best embeddings (up to 3072-dim)
# - "text-embedding-004": Vertex AI legacy (768-dim)
#
# gemini-embedding-001 is recommended - state-of-the-art, tops MTEB leaderboard

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 3072  # gemini-embedding-001 full dimension

# =============================================================================
# EMBEDDING CACHE
# =============================================================================
# Two-tier cache for embeddings (expensive to compute).
# L1: In-memory LRU, L2: Persistent SQLite disk cache.

EMBEDDING_CACHE_ENABLED = True  # Enable embedding caching
EMBEDDING_CACHE_MEMORY_SIZE = 10000  # Max entries in memory LRU cache
EMBEDDING_CACHE_PERSIST = True  # Enable disk persistence (SQLite)
EMBEDDING_CACHE_WARM_ON_START = True  # Pre-load from signatures on startup
EMBEDDING_CACHE_TTL_DAYS = 30  # Prune disk entries older than N days

# =============================================================================
# DSL EXPRESSION CACHE
# =============================================================================
# LRU cache for LLM-generated arithmetic expressions.
# Key: (operation, param_names), Value: (expression, used_params)
# Bounded to prevent memory growth over long runs.

DSL_EXPR_CACHE_MAX_SIZE = 1000  # Max entries in DSL expression cache

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

# Note: Smooth expansion is always enabled (no toggle) per CLAUDE.md
DEPTH_FORCE_DECOMPOSE_DEPTH = 5  # Always decompose at depth 0-5
DEPTH_DECOMPOSE_DECAY_BASE = 0.5  # Decay rate per depth beyond force threshold
DEPTH_DECOMPOSE_MIN_PROB = 0.05  # Floor probability (never fully disable decompose option)

# =============================================================================
# EXPANSION COLD-START BOOST
# =============================================================================
# Extra expansion multiplier when bootstrapping (few signatures).
# Decays as: cold_boost = 1 + exp(-sigs / COLD_START_HALFLIFE)
#   0 sigs: 2x expansion
#   3000 sigs: 1.37x (37% extra)
#   6000+ sigs: ~1x (no boost, system mature)
COLD_START_HALFLIFE = 3000  # Signatures at which cold boost decays to 37%

# =============================================================================
# SIGNATURE HINTS
# =============================================================================
HINT_LIMIT = 3           # Max hints to include in prompts
HINT_MIN_SIMILARITY = 0.5  # Min similarity for hints

RECURSIVE_DECOMPOSITION_ENABLED = True  # Enable decomposition for complex steps
RECURSIVE_MAX_DEPTH = 9  # Max routing depth: deep decomposition for complex problems
RECURSIVE_CONFIDENCE_THRESHOLD = 0.8  # Route deeper when DSL confidence < this

# Umbrella routing depth limits
_UMBRELLA_MAX_DEPTH_RAW = 10  # Configurable max depth for umbrella routing chains
_UMBRELLA_HARD_CAP = 100  # Absolute maximum to prevent unbounded recursion
# Validate and clamp: ensure positive integer, capped at hard limit
UMBRELLA_MAX_DEPTH = max(1, min(int(_UMBRELLA_MAX_DEPTH_RAW or 10), _UMBRELLA_HARD_CAP))
UMBRELLA_ROUTING_THRESHOLD = 0.5  # Min similarity for umbrella child routing (lower than global 0.85 since we're picking best among known children)

# =============================================================================
# SCAFFOLD STRUCTURE (Pre-allocated tree depth for domain emergence)
# =============================================================================
# The universal tree pre-allocates placeholder umbrella DEPTH at startup.
# This gives the tree vertical room to grow - domains emerge as traffic flows.
#
# NO HORIZONTAL PRE-ALLOCATION: We don't pre-create branches. Instead:
#   - Create a deep chain of placeholder umbrellas (1 per level)
#   - Branches fork DYNAMICALLY as different problem types arrive
#   - Each problem that doesn't match existing path creates new branch
#
# Structure (initial):
#   Level 0: ROOT
#   Level 1: [placeholder]        <- single chain, forks on demand
#   Level 2: [placeholder]
#   ...
#   Level N: [placeholder]
#   Level N+1+: LEAF SIGNATURES   <- GSM8K problems land here
#
# Structure (after training):
#   Level 0: ROOT
#   Level 1: [arithmetic] [algebra] [geometry]...   <- forked from traffic
#   Level 2: [addition] [subtraction]...
#   ...

SCAFFOLD_ENABLED = True  # Enable pre-allocated scaffold structure
SCAFFOLD_LEVELS = 8  # Deep scaffold (8 levels before leaves)
MIN_SIGNATURE_DEPTH = 8  # Minimum depth for leaf signatures (deep tree)
MIN_FORK_DEPTH = 4  # Don't fork until this depth (top levels stay abstract)
SCAFFOLD_FORK_THRESHOLD = 0.6  # Create new branch if best match below this (divergent problem)

# NO HORIZONTAL SCALING: Initial scaffold is a single chain.
# Branches fork DYNAMICALLY at runtime when problems diverge.
#
# Tree structure:
#   Level 0: ROOT
#   Level 1-3: Abstract routing (no forking, all problems flow through)
#   Level 4-7: Domain emergence (arithmetic, algebra fork dynamically here)
#   Level 8+: Leaf signatures (actual DSL executors)

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

# Database retry settings (for sqlite3.OperationalError handling)
DB_MAX_RETRIES = 5  # Max retry attempts for transient DB errors
DB_BASE_RETRY_DELAY = 0.05  # Base delay in seconds (exponential backoff with jitter)

# =============================================================================
# LLM CLIENT
# =============================================================================
# Training mode uses beefy models for better decomposition and learning.
# Inference mode uses lightweight models for fast, cost-effective execution.
#
# Training models (beefy): gpt-4o, claude-opus-4-20250514
# Inference models (light): gpt-4o-mini, gpt-4.1-nano

CLIENT_DEFAULT_TIMEOUT = 120.0
CLIENT_DEFAULT_TEMPERATURE = 0.0  # Zero for deterministic responses
CLIENT_CONNECT_TIMEOUT = 10.0
CLIENT_BASE_RETRY_DELAY = 1.0
CLIENT_MAX_RETRY_DELAY = 30.0
PLANNER_DEFAULT_TEMPERATURE = 0.0  # Zero for deterministic decomposition

# Model configuration - set via TRAINING_MODE env var
# TRAINING_MODE=true  -> use beefy models for learning
# TRAINING_MODE=false -> use lightweight models for inference
TRAINING_MODEL = os.getenv("TRAINING_MODEL", "gpt-4o")  # Beefy model for training
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "gpt-4o-mini")  # Light model for inference

# Active model selection based on mode
DEFAULT_MODEL = TRAINING_MODEL if TRAINING_MODE else INFERENCE_MODEL
PLANNER_DEFAULT_MODEL = DEFAULT_MODEL
SOLVER_DEFAULT_MODEL = DEFAULT_MODEL
