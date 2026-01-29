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
# DATABASE PROTECTION
# =============================================================================
# After cold start, the DB contains valuable learned data. Protect it!
# Set DB_PROTECTED=True to prevent accidental deletion via CLI or scripts.
# Override with MYCELIUM_DB_PROTECTED=false if you really need to clear it.
DB_PROTECTED = os.getenv("MYCELIUM_DB_PROTECTED", "true").lower() in ("true", "1", "yes")

# =============================================================================
# SIGNATURE MATCHING
# =============================================================================

# Similarity Thresholds (adjusted for 768-dim embeddings)
# Cold-start aware: higher threshold early (more branching), lower later (consolidation)
MIN_MATCH_THRESHOLD = 0.85  # Mature threshold - reduce signature fragmentation
MIN_MATCH_THRESHOLD_COLD_START = 0.92  # Cold start threshold - create more signatures
MIN_MATCH_RAMP_SIGNATURES = 50  # Signatures needed to reach mature threshold

# Atomic operation detection (per CLAUDE.md: route by what operations DO)
ATOMIC_SIMILARITY_THRESHOLD = 0.70  # Below this = unknown/complex operation
ATOMIC_GAP_THRESHOLD = 0.03  # Gap between best and 2nd best match; below this = multi-part

# Fork probability scaling (per CLAUDE.md "The Flow": thresholds from config)
FORK_GAP_SCALING_FACTOR = 2.0  # Multiplier for gap → probability conversion

# Signature routing thresholds (per CLAUDE.md "The Flow")
# These are the default min_similarity values for various routing functions
ROUTING_MIN_SIMILARITY = 0.85  # Default min_similarity for main routing (db.py, solver.py)
ROUTING_MIN_SIMILARITY_PERMISSIVE = 0.5  # Permissive threshold for alternative/candidate search
ROUTING_BEST_MATCH_MIN_SIMILARITY = 0.8  # find_best_match threshold

# Signature placement thresholds (db.py find_deeper_signature, decide_signature_placement)
PLACEMENT_MIN_SIMILARITY = 0.75  # Threshold for placement decisions and deeper routing

# Hint alternatives threshold (get_signature_hints)
HINT_ALTERNATIVES_MIN_SIMILARITY = 0.3  # Minimum for hint alternatives

# Welford-adaptive similarity thresholds (per CLAUDE.md "The Flow")
# Instead of static 0.85, adapt based on observed similarity distribution
ADAPTIVE_THRESHOLD_MIN_SAMPLES = 50  # Minimum Welford observations before using adaptive
ADAPTIVE_THRESHOLD_K = 1.5  # Standard deviations below mean (captures ~93% of good matches)
ADAPTIVE_THRESHOLD_MIN = 0.70  # Floor - never go below this
ADAPTIVE_THRESHOLD_MAX = 0.95  # Ceiling - never go above this

# =============================================================================
# DSL EXECUTION
# =============================================================================

DSL_TIMEOUT_SEC = 1.0  # Timeout for DSL script execution

# Decomposition queue settings
# Decomposition triggers when EITHER condition is met (whichever comes first):
DECOMP_MIN_BATCH_SIZE = 5  # Trigger when queue reaches this size
DECOMP_MAX_QUEUE_AGE_SEC = 15.0  # Or when oldest item is this old (seconds)

# =============================================================================
# PERIODIC TREE MAINTENANCE (all structural changes bundled together)
# =============================================================================
# Per CLAUDE.md: Batch expensive operations, don't block the hot path
# All structural tree changes happen together in periodic maintenance cycles:
#   - Auto-decomposition (LLM calls to create children)
#   - Merge/split from interference patterns
#   - Restructuring (clustering, orphan cleanup)
#   - Signature retirement
#
# This keeps the hot path fast and consolidates expensive operations.

TREE_MAINTENANCE_INTERVAL = 10  # Run maintenance every N problems (0 = disabled)
TREE_MAINTENANCE_ENABLED = True  # Master switch for periodic maintenance

# Auto-decomposition settings (part of tree maintenance)
AUTO_DECOMP_BATCH_ENABLED = True  # Queue decompositions instead of inline execution
AUTO_DECOMP_MAX_QUEUE_SIZE = 50  # Max queued decompositions before force-processing

# Pre-execution complexity detection
# When disabled, only post-mortem flags failing steps for decomposition
# (per CLAUDE.md: "failures are valuable data points")
PRE_EXECUTION_COMPLEXITY_DETECTION = False  # Disabled: embedding gap detection too aggressive

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

# DSL Template Matching (per CLAUDE.md "The Flow": thresholds from config, not magic numbers)
DSL_TEMPLATE_MIN_SIMILARITY = 0.5  # Minimum similarity for DSL template consideration
DSL_TEMPLATE_MATCH_THRESHOLD = 0.7  # Threshold for "good enough" DSL match
DSL_PARAM_WEIGHT_SEMANTIC = 0.7  # Weight when semantic params available
DSL_PARAM_WEIGHT_DEFAULT = 0.3  # Weight otherwise
DSL_MIN_SUCCESS_RATE = 0.6  # Minimum success rate for DSL selection

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
AUTO_DEMOTE_MIN_USES_FLOOR = 3   # Start at 3 (give signatures multiple chances before demotion)
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

# UCB1 adjustment from post-mortem hit/miss patterns
# Tracks: high_conf_wrong (confident but failed), low_conf_right (explored and succeeded)
# - High low_conf_right rate → exploration is finding good paths → increase C
# - High high_conf_wrong rate → confident picks are wrong → increase C
UCB1_ADJUSTMENT_ENABLED = True  # Enable post-mortem UCB1 adjustment
UCB1_ADJUSTMENT_WINDOW = 50  # Rolling window of problems for hit/miss tracking
UCB1_ADJUSTMENT_MAX_DELTA = 0.3  # Max adjustment to C (±0.3)
UCB1_ADJUSTMENT_SENSITIVITY = 0.5  # How quickly to respond to patterns (0-1)

# =============================================================================
# ADAPTIVE SIMILARITY THRESHOLDS (Welford-based, no magic numbers)
# =============================================================================
# Per CLAUDE.md: Route by what operations DO (graph_embedding), not what they SOUND LIKE.
# Instead of hardcoded thresholds, we learn what "same" and "similar" mean from data.
#
# Two thresholds:
# 1. dedup_threshold: Above this = same node (return existing signature)
# 2. cluster_threshold: Above this = same cluster (share parent)
#
# Computed as: mean - k * stddev from observed similarity distributions.

ADAPTIVE_THRESHOLD_K = 2.0  # Stddevs below mean (higher = more permissive)
ADAPTIVE_MIN_SAMPLES = 10   # Min observations before using learned thresholds

# Cold-start defaults (used until we have enough data)
COLD_START_DEDUP_THRESHOLD = 0.95    # Very high sim = same node
COLD_START_CLUSTER_THRESHOLD = 0.80  # Moderately high sim = same cluster

# =============================================================================
# ADAPTIVE REJECTION (per-leaf similarity thresholds from historical successes)
# =============================================================================
# Per mycelium-i601: Leaf nodes learn their own acceptance thresholds from historical
# success similarities. Instead of a global threshold, each leaf computes:
#   threshold = mean(success_similarities) - k * std(success_similarities)
# This lets specialized leaves become picky, while broad leaves stay permissive.

ADAPTIVE_REJECTION_ENABLED = True  # Enable per-leaf adaptive thresholds
ADAPTIVE_REJECTION_K = 1.5  # Number of std devs below mean (higher = more permissive)
ADAPTIVE_REJECTION_MIN_SAMPLES = 5  # Min successful matches before adaptive kicks in
ADAPTIVE_REJECTION_DEFAULT_THRESHOLD = 0.5  # Fallback threshold for cold-start leaves
ADAPTIVE_REJECTION_MIN_THRESHOLD = 0.3  # Floor: never reject below this similarity
ADAPTIVE_REJECTION_MAX_THRESHOLD = 0.95  # Ceiling: never require above this similarity

# Rejection decomposition thresholds (per CLAUDE.md "The Flow")
# When signatures accumulate rejections, flag for potential decomposition
# Cold-start ramping: be more aggressive early (flag quickly), patient later (wait for evidence)
REJECTION_COUNT_THRESHOLD_COLD = 3  # Cold start: flag after just 3 rejections (get signal fast)
REJECTION_COUNT_THRESHOLD_MATURE = 10  # Mature: require 10+ rejections before flagging
REJECTION_COUNT_RAMP_SIGNATURES = 500  # Signatures at which we transition from cold to mature

# Welford-guided rejection rate threshold (per CLAUDE.md "The Flow")
# Instead of hardcoded 30%, compute adaptive threshold from rejection rate distribution
# Threshold = mean + k*std (signatures with rates above this are flagged)
REJECTION_RATE_WELFORD_ENABLED = True  # Use Welford-guided threshold vs fixed
REJECTION_RATE_WELFORD_K = 1.5  # Std devs above mean to flag (higher = more selective)
REJECTION_RATE_MIN_SAMPLES = 5  # Min attempts (uses + rejections) before including in distribution
REJECTION_RATE_THRESHOLD = 0.30  # Fallback fixed threshold (used if Welford disabled or not enough data)

# Computed at runtime via get_rejection_count_threshold()
# For backward compatibility, expose the mature value as the default
REJECTION_COUNT_THRESHOLD = REJECTION_COUNT_THRESHOLD_MATURE

# =============================================================================
# MCTS COMPUTE BUDGET (multi-path exploration)
# =============================================================================
# Budget controls how many paths to explore during routing.
# Adaptive budget = BASE * (1 + difficulty), so:
#   - difficulty=0.0 (easy): 3.0 paths
#   - difficulty=0.5 (medium): 4.5 paths
#   - difficulty=1.0 (hard): 6.0 paths

COMPUTE_BUDGET_BASE = 3.0  # Base budget for adaptive scaling (both training & inference)
COMPUTE_BUDGET_CONFIDENCE_THRESHOLD = 0.5  # Explore alternatives when confidence < this

# Selective branching: only branch when undecided (per CLAUDE.md)
# UCB1 gap = difference between top-2 UCB1 scores at routing decision
# High gap = confident (don't branch), Low gap = undecided (branch)
UCB1_GAP_BRANCH_THRESHOLD = 0.15  # Branch when min_gap < this (undecided) - cold start fallback

# =============================================================================
# ADAPTIVE UCB1 GAP THRESHOLD (Welford-guided, per mycelium-02nn)
# =============================================================================
# Per CLAUDE.md "System Independence": Replace manual _force_exploration flag
# with Welford-guided adaptive thresholds based on historical gap outcomes.
#
# Tracks UCB1 gap values from routing decisions that led to success vs failure.
# Adaptive threshold: gap_mean - k * gap_std (from successful routings)
# This lets the system learn optimal branching thresholds from experience.

ADAPTIVE_GAP_ENABLED = True  # Use Welford-guided gap threshold instead of static
ADAPTIVE_GAP_K = 1.5  # k stddevs below mean for adaptive threshold
ADAPTIVE_GAP_MIN_SAMPLES = 10  # Min samples before using learned threshold
ADAPTIVE_GAP_MIN_THRESHOLD = 0.05  # Floor: never set threshold below this
ADAPTIVE_GAP_MAX_THRESHOLD = 0.5  # Ceiling: never set threshold above this

# Epsilon-greedy exploration: with probability EPSILON, pick random signature
# This ensures under-visited signatures get attempts even when UCB1 favors exploitation
EXPLORATION_EPSILON = 0.15  # 15% chance of random exploration (0 = pure UCB1)
EXPLORATION_EPSILON_DECAY = 0.995  # Decay factor per problem (0.995^100 ≈ 0.6)
EXPLORATION_EPSILON_MIN = 0.05  # Minimum epsilon floor

# =============================================================================
# REACTIVE EXPLORATION (retry on failure) - DISABLED BY DEFAULT
# =============================================================================
# Per mycelium-tnil + CLAUDE.md "System Independence":
# "Let signatures fail. This is how the system learns."
# "Record every failure—it feeds the post-mortem analysis"
# "Accumulated failure patterns (not individual failures) trigger decomposition"
#
# Reactive exploration (retrying with different seeds) MASKS failures instead of
# learning from them. The proper flow per "The Flow":
#   1. Problem fails → failure recorded in DB
#   2. Post-mortem runs on the failing DAG (already has multiple MCTS threads)
#   3. Welford stats accumulate from failures
#   4. Eventually triggers decomposition when variance is high
#
# Retrying just hopes to get lucky, doesn't learn. MCTS already explores
# multiple paths during the first solve - that's sufficient exploration.
#
# These configs are kept for optional use but DISABLED by default.
REACTIVE_EXPLORATION_ENABLED = False  # DISABLED: Let failures stand, system learns
REACTIVE_EXPLORATION_MAX_ALTERNATIVES = 3  # Max alternative nodes to try per step
REACTIVE_EXPLORATION_MAX_RETRIES = 5  # Max total retry attempts
REACTIVE_EXPLORATION_MIN_SIMILARITY = 0.5  # Min similarity for alternative candidates
REACTIVE_EXPLORATION_FULL_RESOLVE = False  # DISABLED: Don't re-solve, let failures stand
REACTIVE_EXPLORATION_NUM_THREADS = 3  # Number of parallel exploration threads to spawn
REACTIVE_EXPLORATION_TEMPERATURE = 0.3  # Higher temp for diversity (vs 0.0 default)
REACTIVE_EXPLORATION_EPSILON_BOOST = 0.3  # Boost epsilon during exploration (stacks with base)

# =============================================================================
# ADAPTIVE REACTIVE EXPLORATION MULTIPLIERS (Welford-guided, per mycelium-02nn)
# =============================================================================
# NOTE: Only used when REACTIVE_EXPLORATION_ENABLED = True (disabled by default)
#
# Per CLAUDE.md "The Flow": DB Statistics → Welford → Tree Structure
# Multipliers adjust based on whether reactive exploration is finding winning paths.
#
# Logic (when reactive exploration IS enabled):
# - Low success rate → need more exploration → increase multipliers
# - High success rate → current settings work → maintain or decrease
# - Welford tracks variance to detect when adjustments are needed

ADAPTIVE_REACTIVE_ENABLED = True  # Use Welford-guided multipliers (when reactive enabled)
ADAPTIVE_REACTIVE_MIN_SAMPLES = 5  # Min reactive explorations before adapting

# Fallback multipliers (used during cold start)
REACTIVE_EXPLORATION_GAP_MULT = 2.0  # Default gap threshold multiplier
REACTIVE_EXPLORATION_BUDGET_MULT = 1.5  # Default budget multiplier
REACTIVE_EXPLORATION_MIN_BUDGET = 3.0  # Minimum budget during reactive exploration

# Multiplier adjustment bounds
REACTIVE_EXPLORATION_GAP_MULT_MIN = 1.2  # Floor: at least 20% more lenient
REACTIVE_EXPLORATION_GAP_MULT_MAX = 4.0  # Ceiling: at most 4x more lenient
REACTIVE_EXPLORATION_BUDGET_MULT_MIN = 1.2  # Floor: at least 20% more budget
REACTIVE_EXPLORATION_BUDGET_MULT_MAX = 3.0  # Ceiling: at most 3x budget

# Adjustment sensitivity (how much success rate affects multipliers)
# At 0% success: multipliers increase toward MAX
# At 100% success: multipliers decrease toward MIN
REACTIVE_EXPLORATION_ADJUST_K = 1.5  # k stddevs for adjustment (higher = more conservative)

# Step decomposition fallback: when reactive exploration fails to find a winning path,
# decompose failing steps into smaller sub-steps and re-solve
STEP_DECOMPOSITION_FALLBACK_ENABLED = True  # Enable step decomposition on reactive failure
STEP_DECOMPOSITION_MAX_DEPTH = 2  # Max decomposition depth (prevent infinite recursion)
STEP_DECOMPOSITION_MIN_STEPS = 2  # Min steps in failed result to attempt decomposition

# =============================================================================
# ADAPTIVE DECOMPOSITION THRESHOLDS
# =============================================================================
# min_attempts for flagging nodes needing decomposition
# Lower during cold start (get signal faster), higher when mature (wait for evidence)
DECOMP_MIN_ATTEMPTS_COLD = 1  # Cold start: flag after just 1 failure
DECOMP_MIN_ATTEMPTS_MATURE = 3  # Mature: require 3+ attempts before flagging
DECOMP_MAX_WIN_RATE = 0.5  # Flag nodes with win rate below this
DECOMP_MAX_PER_CYCLE = 5  # Max signatures to decompose per learning cycle (gradual learning)

# Welford-based decomposition success rate (replaces is_atomic flag)
# If a signature's decomposition success rate falls below this threshold,
# skip future decomposition attempts (effectively atomic)
DECOMP_SUCCESS_RATE_THRESHOLD = 0.1  # Skip decomp if success rate < 10%
DECOMP_MIN_ATTEMPTS_FOR_RATE = 3  # Need at least 3 attempts to calculate meaningful rate

# Bayesian prior for cold start (assume some successes before any data)
ROUTING_PRIOR_SUCCESSES = 2
ROUTING_PRIOR_USES = 4

# Parent credit propagation (reward successful routers)
# Per CLAUDE.md: "Parent umbrellas get decay^depth credit (default 0.5 per level)"
PARENT_CREDIT_DECAY = 0.5  # Credit multiplier per depth (0.5^1=0.5, 0.5^2=0.25, 0.5^3=0.125)
PARENT_CREDIT_MAX_DEPTH = 3  # Max depth to propagate (per CLAUDE.md: "default 3 levels")
PARENT_CREDIT_MIN = 0.1  # Minimum credit to apply (filter noise)

# Graph centroid propagation (batch update parent graph_embeddings)
CENTROID_PROPAGATION_MAX_DEPTH = 3  # Max levels to propagate graph_centroid changes (perf optimization)

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
# SIGNATURE LOOKUP CACHE
# =============================================================================
# LRU cache with TTL for hot signature lookups (get_signature, get_children).
# Skips DB entirely for frequently accessed signatures during routing.
# Invalidated on centroid/signature updates.

SIGNATURE_CACHE_MAX_SIZE = 1000  # Max entries in signature lookup cache
SIGNATURE_CACHE_TTL_SECONDS = 60.0  # TTL for cached entries (seconds)
CHILDREN_CACHE_MAX_SIZE = 500  # Max entries for get_children cache
CENTROID_CACHE_MAX_SIZE = 10000  # Max entries for centroid embedding cache
GRAPH_EMBEDDING_CACHE_MAX_SIZE = 500  # Max entries for graph embedding cache

# =============================================================================
# EMBEDDING DRIFT (Semantic Attractors)
# =============================================================================
# Per CLAUDE.md: "High-traffic signatures become semantic attractors: their
# centroids stabilize around operational meaning rather than vocabulary."
#
# On successful (leaf_node, dag_step) matches, leaf node graph embeddings
# drift toward the successful dag_step embeddings using Welford-adaptive EMA:
#   α = 1 - (k / (k + variance))
#   new_embedding = α * old_embedding + (1-α) * success_embedding
#
# High variance nodes drift faster (exploring), low variance nodes are sticky.

EMBEDDING_DRIFT_ENABLED = True  # Enable embedding drift during tree review
EMBEDDING_DRIFT_INTERVAL = 50  # Problems between drift updates
EMBEDDING_DRIFT_VARIANCE_K = 1.0  # Welford α sensitivity (higher = slower drift)
EMBEDDING_DRIFT_MIN_SUCCESSES = 3  # Min successes before applying drift

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
# Example with FORCE_DECOMPOSE_DEPTH=13, DECAY_BASE=0.5:
#   depth 0-13: 100% decompose (forced through reserved + early fork zone)
#   depth 14: 50% decompose
#   depth 15: 25% decompose
#   depth 16: 12.5% decompose
#   depth 17: 6.25% decompose
#   depth 18+: ~3% decompose (mostly execute at leaf depth)

# Note: Smooth expansion is always enabled (no toggle) per CLAUDE.md
DEPTH_FORCE_DECOMPOSE_DEPTH = 13  # Always decompose at depth 0-13 (matches level 12 fork start + 1)
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
HINT_MAX_CHILDREN_PER_CLUSTER = 5  # Max child hints per umbrella cluster
HINT_MAX_DEPTH = 3       # How deep to traverse (1=level-1 only, 2=include grandchildren)
HINT_MAX_GRANDCHILDREN = 3  # Max grandchildren hints per level-2 umbrella

# Umbrella routing depth limits
_UMBRELLA_MAX_DEPTH_RAW = 20  # Configurable max depth for umbrella routing chains (2 beyond leaves)
_UMBRELLA_HARD_CAP = 100  # Absolute maximum to prevent unbounded recursion
# Validate and clamp: ensure positive integer, capped at hard limit
UMBRELLA_MAX_DEPTH = max(1, min(int(_UMBRELLA_MAX_DEPTH_RAW or 10), _UMBRELLA_HARD_CAP))
UMBRELLA_ROUTING_THRESHOLD = 0.5  # Min similarity for umbrella child routing (lower than global 0.85 since we're picking best among known children)
NEW_CHILD_SIMILARITY_THRESHOLD = 0.7  # Min similarity to reuse existing child instead of creating new
                                       # Higher than UMBRELLA_ROUTING_THRESHOLD: routing failed at 0.5,
                                       # but child at 0.7+ is "close enough" - use instead of duplicating
                                       # This prevents duplicate children for similar steps
UMBRELLA_REPOINT_SIMILARITY = 0.75  # Threshold for repointing to existing deeper signature
                                     # Slightly lower than MIN_MATCH_THRESHOLD for flexibility

# Tree growth settings (organic, no pre-allocation)
MIN_FORK_DEPTH = 0  # Allow forking at any depth (organic growth)
SCAFFOLD_ENABLED = False  # Legacy - scaffold removed, tree grows organically
MIN_SIGNATURE_DEPTH = 0  # Allow leaf signatures at any depth

# Big Bang forking thresholds (used by organic growth, not scaffold)
SCAFFOLD_FORK_THRESHOLD = 0.80  # Fork threshold at maturity
SCAFFOLD_FORK_THRESHOLD_COLD_START = 0.70  # Fork threshold during cold start (more aggressive)
SCAFFOLD_FORK_RAMP_SIGNATURES = 500  # Signatures needed to reach maturity

# =============================================================================
# BIG BANG - Smooth Fork Probability
# =============================================================================
# Controls when/where forking is allowed in the signature tree.
# Uses smooth, continuous functions - no hard thresholds.
#
# Key concepts:
#   - Maturity: 0→1 as signature count grows (sigmoid)
#   - Fork center: Starts at level 12, drifts toward root as maturity increases
#   - Hysteresis: Levels that have forked become easier to fork again
#   - Similarity gap: Larger gap from threshold = more likely to fork
#
# Per CLAUDE.md: "smooth and continuous learning process"
# Per CLAUDE.md: "aggressively branch out early, tapering off later"

BIG_BANG_TARGET_SIGNATURES = 10000  # Signature count at which system is "mature"
BIG_BANG_SIGMOID_STEEPNESS = 2.0  # Steepness of sigmoid transitions (higher = sharper)
BIG_BANG_HYSTERESIS_BONUS = 0.3  # 30% bonus fork probability for levels with existing forks
BIG_BANG_FORK_CENTER_DRIFT_RATE = 0.8  # How fast fork center drifts toward root (per maturity unit)
BIG_BANG_MIN_FORK_PROB = 0.05  # Floor probability (always some chance to fork)
BIG_BANG_MAX_FORK_PROB = 0.95  # Ceiling probability (never 100% certain to fork)

# Maturity exponential decay formula: maturity = 1 - exp(-sig_count / tau)
# where tau = BIG_BANG_TARGET_SIGNATURES / TAU_DIVISOR
# At sig_count = 3*tau, maturity reaches ~95% (since 1 - exp(-3) ≈ 0.95)
# TAU_DIVISOR = 3.0 means system reaches 95% maturity at TARGET_SIGNATURES
BIG_BANG_TAU_DIVISOR = 3.0

# =============================================================================
# WELFORD MINIMUM SAMPLES (per CLAUDE.md "The Flow")
# =============================================================================
# Different operations require different confidence levels before using Welford stats.
# Lower = more aggressive (act on less data), Higher = more conservative.

WELFORD_MIN_SAMPLES_BASIC = 5  # Basic operations (routing, threshold decisions)
WELFORD_MIN_SAMPLES_STRUCTURE = 5  # Tree structure changes (decomposition, merging)

# Synthesis step detection (for umbrella learner)
# These are anchors for steps that aggregate/combine results (should be skipped)
SYNTHESIS_STEP_ANCHORS = [
    "Combine or aggregate the results from previous calculations",
    "Add up or sum all the intermediate results to get the final answer",
    "Put together the computed values to find the total",
]
SYNTHESIS_STEP_THRESHOLD = 0.88  # Similarity threshold to detect synthesis steps (raised from 0.75 to avoid false positives)

# =============================================================================
# THREAD TRACKING (Multi-path credit/blame backpropagation)
# =============================================================================
# Tracks complete execution paths ("threads") through DAG for credit attribution.
# Per CLAUDE.md: "Positive credit to winning thread, negative to losing threads"
#
# Key insight: When multi-path exploration forks, each alternative is a separate
# thread. After grading against ground truth, we know which threads were correct
# vs incorrect, enabling per-signature win/loss tracking for cluster analysis.

THREAD_TRACKING_ENABLED = True  # Enable thread tracking (only active in TRAINING_MODE)
THREAD_MAX_FORKS_PER_STEP = 3  # Max alternative threads to create at any fork point
THREAD_CREDIT_DECAY_PER_FORK = 0.7  # Credit decay per fork depth (0.7^1=0.7, 0.7^2=0.49, etc.)
THREAD_MIN_CREDIT = 0.1  # Minimum credit to apply (filter noise from deep forks)

# =============================================================================
# MCTS POST-MORTEM (Amplitude updates after grading)
# =============================================================================
# Per CLAUDE.md: "High confidence + failure = strong negative signal"
# After grading, compute amplitude_post for each thread_step based on:
# - Thread outcome (won/lost)
# - Prior amplitude (confidence when routing decision was made)
#
# Formula:
# - Won + high conf: reinforce (× POSTMORTEM_REINFORCE_MULT)
# - Won + low conf: boost discovery (× POSTMORTEM_BOOST_MULT)
# - Lost + low conf: mild penalty (× POSTMORTEM_MILD_PENALTY_MULT)
# - Lost + high conf: strong penalty (× POSTMORTEM_STRONG_PENALTY_MULT)

POSTMORTEM_ENABLED = True  # Run postmortem analysis after grading
POSTMORTEM_HIGH_CONF_THRESHOLD = 0.7  # Amplitude >= this is "high confidence"
POSTMORTEM_REINFORCE_MULT = 1.4  # Won + high confidence: biggest boost (confident AND right)
POSTMORTEM_BOOST_MULT = 1.1  # Won + low confidence: small boost (lucky exploration)
POSTMORTEM_MILD_PENALTY_MULT = 0.85  # Lost + low confidence: expected uncertainty
POSTMORTEM_STRONG_PENALTY_MULT = 0.5  # Lost + high confidence: harsh penalty
POSTMORTEM_AMPLITUDE_MIN = 0.0  # Clamp amplitude_post minimum
POSTMORTEM_AMPLITUDE_MAX = 2.0  # Clamp amplitude_post maximum

# Credit propagation from amplitude_post to signature stats (per mycelium-itkn)
# Closes the loop: amplitude_post → signature.successes/operational_failures
# Works for BOTH single-path and multi-path problems (per mycelium-plm8)
CREDIT_PROPAGATION_ENABLED = True  # Enable amplitude_post → signature stats
CREDIT_PROPAGATION_THRESHOLD_CREDIT = 1.0  # amp_post >= this → credit (success)
CREDIT_PROPAGATION_THRESHOLD_BLAME = 0.7  # amp_post < this → blame (op_failure)

# Partial credit for correct steps in failed problems (per mycelium-7o8i)
# Steps with high confidence in losing threads get partial credit (benefit of doubt)
# Steps with low confidence in losing threads get blamed (likely caused failure)
PARTIAL_CREDIT_HIGH_CONF_THRESHOLD = 0.7  # amplitude >= this in lost thread → partial credit
PARTIAL_CREDIT_WEIGHT = 0.5  # Weight for partial credit (0.5 = half a success)

# =============================================================================
# DETAILED LOGGING OPTIMIZATION
# =============================================================================
# Only store detailed thread step records (mcts_thread_steps) for failures.
# Summaries (mcts_step_summaries) are always stored for credit propagation.
# This reduces DB writes while keeping learning signal intact.
LOG_DETAILED_STEPS_FAILURES_ONLY = True  # Only log detailed records for failures

# =============================================================================
# STEP-NODE STATS (Materialized (dag_step_type, node_id) performance)
# =============================================================================
# Per CLAUDE.md: "The combination of (dag_step_id, node_id) is what we're learning"
# This closes the feedback loop: post-mortem → dag_step_node_stats → routing UCB1

STEP_NODE_STATS_ENABLED = True  # Track and use (dag_step_type, node_id) stats
STEP_NODE_STATS_MIN_USES = 3  # Min uses before trusting step-level stats
STEP_NODE_STATS_WEIGHT = 0.6  # Blend weight (0.6 step + 0.4 signature-level)
STEP_NODE_STATS_PRIOR_WINS = 1  # Bayesian prior wins (prevents 0/0 division)
STEP_NODE_STATS_PRIOR_USES = 2  # Bayesian prior uses
AMPLITUDE_POST_PENALTY_THRESHOLD = 0.6  # avg_amplitude_post below this → penalize routing
AMPLITUDE_POST_PENALTY_MULT = 0.8  # Multiplicative penalty for low amplitude_post

# POSITION-AWARE ROUTING (plan_step_stats)
# Per CLAUDE.md: "Failures Are Valuable Data Points" - track success by step position
# Same node at step 1 vs step 5 may have very different success rates
POSITION_STATS_ENABLED = True  # Use plan_step_stats for position-aware routing
POSITION_STATS_MIN_OBS = 3  # Minimum observations to trust position stats
POSITION_STATS_WEIGHT = 0.5  # Penalty weight (0.5 = at 0% success, score *= 0.5)

# Variance-based decomposition (Welford's algorithm)
# Per CLAUDE.md: leaf_node ≡ dag_step_type (1:1 mapping)
# The learning unit is (dag_step_id, dag_step_type/node_id)
#
# Many dag_step_ids map to each node. Welford's tracks variance:
# - OUTCOME variance (amp_post): dag_step_ids have inconsistent results
# - EMBEDDING variance (sim): dag_step_ids are semantically diverse
#
# High variance in either → node is too broad → split into children
VARIANCE_DECOMPOSE_ENABLED = True  # Flag high-variance nodes for decomposition
VARIANCE_MIN_SAMPLES = 5  # Min observations before considering variance (cold start)
VARIANCE_THRESHOLD = 0.1  # Min variance to flag as "high" (needs decomposition)
VARIANCE_CHECK_LIMIT = 20  # Max nodes to check per postmortem batch

# =============================================================================
# MCTS INTERFERENCE PATTERNS (Constructive/Destructive)
# =============================================================================
# Per CLAUDE.md: When multiple threads visit the same (dag_step_id, node_id):
# - Constructive (all succeed): Reinforce, consider MERGE centroids
# - Destructive (mixed): Signal to SPLIT the cluster
#
# Interference reveals operational equivalence that embedding similarity cannot.

INTERFERENCE_ENABLED = True  # Detect and apply interference patterns
INTERFERENCE_MIN_CONSTRUCTIVE = 3  # Min occurrences for merge consideration
INTERFERENCE_MIN_DESTRUCTIVE = 2  # Min occurrences for split consideration
INTERFERENCE_CONSTRUCTIVE_BOOST = 0.1  # Strength for constructive effects (placeholder)
INTERFERENCE_DESTRUCTIVE_PENALTY = 0.15  # Strength for destructive effects (placeholder)

# =============================================================================
# MERGE/SPLIT SETTINGS (runs during periodic tree maintenance)
# =============================================================================
# Per CLAUDE.md: Constructive → merge, Destructive → split
# These operations run as part of TREE_MAINTENANCE_INTERVAL cycle.

MERGE_SPLIT_BATCH_SIZE = TREE_MAINTENANCE_INTERVAL  # Alias for backward compat
MERGE_MIN_SUCCESS_RATE = 0.75  # Min success rate for merge candidates
MERGE_MIN_USES = 10  # Min uses to trust signal for merge
MERGE_MIN_SIMILARITY = 0.90  # Min centroid similarity for merge
MERGE_MAX_PER_BATCH = 3  # Max merges per batch run

# =============================================================================
# SIGNATURE RETIREMENT (Prune consistently failing nodes)
# =============================================================================
# Per CLAUDE.md: Signatures that consistently fail should be flagged for retirement.
# Post-mortem identifies "dead weight" nodes that hurt routing.
#
# Accuracy tracking: For each leaf node, we track:
#   - How many times it was selected (uses)
#   - How many times the thread it was in won (successes)
# This gives us leaf_accuracy = successes / uses, inferred from MCTS post-mortem.
#
# Retirement options:
#   1. PRUNE: Delete signature entirely (reroute traffic to siblings)
#   2. DEMOTE: Add routing penalty (deprioritize but keep for learning)
#   3. MERGE_UP: Absorb back into parent umbrella
#
# Different from DECAY (traffic-based): retirement is accuracy-based.

RETIREMENT_ENABLED = True  # Master switch for retirement processing
RETIREMENT_MIN_USES = 10  # Need enough selections to trust accuracy
RETIREMENT_MAX_SUCCESS_RATE = 0.15  # Retire if accuracy below 15%
RETIREMENT_MIN_OPERATIONAL_FAILURES = 3  # Need multiple post-mortem flags
RETIREMENT_MAX_PER_BATCH = 5  # Max retirements per batch run

# Retirement action thresholds (escalating severity)
RETIREMENT_DEMOTE_PENALTY = 0.3  # Routing penalty for demoted signatures
RETIREMENT_PRUNE_SUCCESS_RATE = 0.05  # Prune entirely if below this (very bad)
RETIREMENT_PRUNE_MIN_USES = 20  # Only prune if we have high confidence

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
# GRAPH-BASED ROUTING (Route by operation, not vocabulary)
# =============================================================================
# Per CLAUDE.md: "Route by what operations DO, not what they SOUND LIKE"
#
# Flow: problem → extract operation → embed → compare to graph embeddings
# This addresses: "embedding clusters by vocab not operational semantics"
#
# Graph embeddings encode WHAT a DSL computes (structurally):
#   MUL(param_0, param_1) → "multiply two values"
# Operation extraction identifies WHAT is needed (semantically):
#   "Calculate 15% of 200" → "multiply percentage by base, divide by 100"
#
# Comparing operation embedding to graph embeddings routes by operation type,
# not by surface vocabulary similarity.

GRAPH_ROUTING_ENABLED = True  # Master switch for graph-based routing
GRAPH_ROUTING_MIN_SIMILARITY = 0.80  # Minimum similarity for graph match
GRAPH_ROUTING_HIGH_CONFIDENCE = 0.90  # High-confidence graph match (skip planner)
GRAPH_ROUTING_BOOST_FACTOR = 0.15  # Boost to UCB1 when graph matches
GRAPH_ROUTING_FALLBACK_TO_CENTROID = False  # No centroid fallback - graph-only routing

# =============================================================================
# INLINE DECOMPOSITION (when leaf rejects a step)
# =============================================================================
INLINE_DECOMP_MAX_DEPTH = 3  # Max recursion depth for inline decomposition (prevents infinite loops)
COLD_START_SIGNATURE_THRESHOLD = 100  # Below this, skip rejection and build vocabulary (raised for latency)

# =============================================================================
# WELFORD-BASED TREE RESTRUCTURING (per mycelium-bjrf)
# =============================================================================
# Cold start: first N problems create flat leaves under root, collecting Welford stats
# After cold start: use Welford stats to guide sibling/child/merge decisions
# Restructure runs periodically (every RESTRUCTURE_INTERVAL problems)
#
# Per CLAUDE.md "System Independence": fully automated, no manual intervention

COLD_START_PROBLEMS_THRESHOLD = 20  # Problems before restructuring begins (per mycelium-5cn0)
RESTRUCTURE_INTERVAL = 10  # Run restructure every N problems (per mycelium-heh3)
UMBRELLA_LEARNING_INTERVAL = 5  # Run umbrella learning every N problems after cold start

# Welford-based decision thresholds (z-scores)
# Per mycelium-br28: Principled thresholds based on observed data, not magic numbers
WELFORD_MERGE_THRESHOLD = 3.0     # z-score above which to merge (very similar to existing)
WELFORD_SIBLING_THRESHOLD = -2.0  # z-score above which to add as sibling (normal range)
WELFORD_CHILD_THRESHOLD = -3.0    # z-score above which to add as child (somewhat different)
# Below CHILD_THRESHOLD: new cluster under root (very different)

# Periodic tree review thresholds (per periodic tree review plan)
# These guide deduplication, outlier detection, and sub-clustering
RESTRUCTURE_VARIANCE_THRESHOLD = 0.15  # Subcluster if child_std > this (heterogeneous cluster)
RESTRUCTURE_MIN_CHILDREN_FOR_SPLIT = 4  # Need this many children to consider splitting
RESTRUCTURE_MERGE_FLOOR = 0.95  # Never merge signatures below this similarity (safety floor)
RESTRUCTURE_OUTLIER_IMPROVEMENT = 0.1  # Move outlier if new cluster is 10%+ better fit

# Welford-guided cluster boundaries (per CLAUDE.md "System Independence")
# These use relative measures (CV) rather than absolute thresholds
MAX_CHILDREN_PER_PARENT = 20  # Force subcluster consideration above this fan-out
RESTRUCTURE_CV_THRESHOLD = 0.3  # CV (std/mean) above which cluster is heterogeneous

# Divergence splitting (per mycelium-7khj: move magic numbers to config)
DIVERGENCE_CLOSE_DISTANCE = 0.20  # cosine distance < this = close (similarity > 0.80)

# =============================================================================
# MATURITY-BASED DECOMPOSE VS CREATE (Sigmoid transition)
# =============================================================================
# When routing fails (no matching signature), the system must decide:
#   - Create a new atomic signature, OR
#   - Decompose into sub-steps that match existing signatures
#
# Decision uses sigmoid based on system maturity:
#   P(decompose) = sigmoid(maturity_score)
#   maturity_score = (num_sigs - midpoint) / steepness + accuracy_weight * accuracy
#
# Early (few signatures): Low P(decompose) → create new signatures
# Mature (many signatures): High P(decompose) → reuse existing via decomposition
#
# Per CLAUDE.md: "With Mature DB" - smooth transition from expansion to consolidation

MATURITY_DECOMPOSE_ENABLED = True  # Master switch for maturity-based decompose/create
MATURITY_SIGMOID_MIDPOINT = 500  # Signature count where P(decompose) = 0.5
MATURITY_SIGMOID_STEEPNESS = 200  # Controls sigmoid sharpness (higher = smoother)
MATURITY_ACCURACY_WEIGHT = 2.0  # How much accuracy (0-1) shifts the decision
                                 # accuracy=0.8 adds 2.0*0.8=1.6 to maturity_score
MATURITY_MIN_DECOMPOSE_PROB = 0.1  # Floor: always some chance to decompose
MATURITY_MAX_DECOMPOSE_PROB = 0.9  # Ceiling: always some chance to create new

# Expansion rate sigmoid (accuracy-driven tree growth)
# Per mycelium-7khj: move hardcoded expansion parameters to config
# expansion = 1 / (1 + exp((accuracy - midpoint) / steepness))
EXPANSION_SIGMOID_MIDPOINT = 0.7  # Accuracy at which expansion = 0.5
EXPANSION_SIGMOID_STEEPNESS = 0.15  # How sharply expansion drops with accuracy

# Escape hatch: If decomposed sub-steps ALSO don't match, create new atomic
# This recognizes genuinely novel operations that can't be built from existing sigs
MATURITY_ESCAPE_ENABLED = True  # Enable escape hatch for novel operations
MATURITY_ESCAPE_MIN_SUBSTEPS = 2  # Need this many substeps to trigger escape
MATURITY_ESCAPE_MAX_MISSES = 1  # Max substeps allowed to miss before creating atomic

# =============================================================================
# DIAGNOSTIC POST-MORTEM (Accuracy-driven decomposition decisions)
# =============================================================================
# Per CLAUDE.md: "Failures are valuable data points" + "Failing signatures get decomposed"
#
# Post-mortem runs after problem completion to diagnose failures and decide:
#   1. Decompose the dag_step (step too complex for atomic execution)
#   2. Decompose the signature (approach/DSL is wrong for this problem class)
#   3. Reroute (wrong routing, similar steps succeeded elsewhere)
#   4. Wait (insufficient signal, collect more data)
#
# All decisions use smooth continuous functions based on:
#   - Accuracy = successes / uses (percent, not absolute counts)
#   - Confidence = smooth ramp with uses (more data → more trust)
#   - Maturity = sigmoid over signature count (cold start → mature)
#
# Key insight: A signature with 60 successes + 4 failures (93.75%) should NOT
# be decomposed. We use accuracy (percent), not failure count.
#
# CONFIG INTERACTION SUMMARY:
# ---------------------------
# 1. FAILURE THRESHOLD determines when to diagnose:
#    threshold = THRESHOLD_MIN + (THRESHOLD_MAX - THRESHOLD_MIN) * sigmoid(maturity)
#    Cold start: threshold → MIN (act fast, few failures)
#    Mature: threshold → MAX (be patient, more failures needed)
#    Uses MATURITY_SIGMOID_MIDPOINT/STEEPNESS for consistent system-wide maturity
#
# 2. ACCURACY + CONFIDENCE determine what to decompose:
#    confidence = 1 - exp(-uses / CONFIDENCE_HALFLIFE)
#    decompose_score = (1 - accuracy) * confidence * ACCURACY_WEIGHT
#                    + step_distance * DISTANCE_WEIGHT
#
# 3. VERDICT THRESHOLD determines if we act:
#    If max(decompose_step, decompose_sig, reroute) < ACTION_THRESHOLD → "wait"
#    Otherwise, highest score wins
#
# 4. STEP VS SIGNATURE decomposition heuristic:
#    Good sig (accuracy > GOOD_SIG_ACCURACY) + outlier step → decompose step
#    Bad sig (accuracy < BAD_SIG_ACCURACY) → decompose signature
#    Middle: blend between both scores

DIAGNOSTIC_POSTMORTEM_ENABLED = True  # Master switch for diagnostic post-mortem

# Failure threshold: How many failures before we act (smooth function of maturity)
# threshold = MIN + (MAX - MIN) * sigmoid(maturity)
# Cold start (low sigs): act fast (threshold → MIN)
# Mature (high sigs): be patient (threshold → MAX)
DIAGNOSTIC_THRESHOLD_MIN = 1.0  # Minimum failures before acting (cold start)
DIAGNOSTIC_THRESHOLD_MAX = 5.0  # Maximum failures before acting (mature)

# Maturity sigmoid parameters (shared with MATURITY_SIGMOID_* above)
# Uses same midpoint/steepness for consistency across system

# Accuracy confidence: How much we trust the accuracy signal
# confidence = 1 - exp(-uses / CONFIDENCE_HALFLIFE)
# Low uses → low confidence (don't trust accuracy yet)
# High uses → high confidence (accuracy is reliable)
DIAGNOSTIC_CONFIDENCE_HALFLIFE = 10.0  # Uses at which confidence reaches ~63%

# Decompose score weights (continuous blend, not thresholds)
# decompose_score = accuracy_component + distance_component
# Higher score = stronger signal to decompose
DIAGNOSTIC_ACCURACY_WEIGHT = 1.0  # How much low accuracy drives decomposition
DIAGNOSTIC_DISTANCE_WEIGHT = 0.5  # How much step-centroid distance matters

# Action threshold: Must exceed this to act (below = "wait")
DIAGNOSTIC_ACTION_THRESHOLD = 0.5  # Score threshold for taking action

# Reroute detection: When similar steps succeeded with different signatures
DIAGNOSTIC_REROUTE_SIMILARITY_MIN = 0.8  # Min similarity to consider reroute
DIAGNOSTIC_REROUTE_LOOKBACK_DAYS = 7  # How far back to search for similar successes

# Step vs Signature decomposition heuristic
# When signature is accurate overall but failed on THIS step:
#   → Step is unusual/complex → decompose step
# When signature has poor accuracy overall:
#   → Approach is wrong → decompose signature
DIAGNOSTIC_GOOD_SIG_ACCURACY = 0.7  # Accuracy above this = "good signature"
DIAGNOSTIC_BAD_SIG_ACCURACY = 0.3  # Accuracy below this = "bad signature"
# Between these values: blend between step and signature decomposition

# Post-mortem triggered DSL regeneration
# Per beads mycelium-flbq: When post-mortem detects high failure rate, trigger DSL regen
POSTMORTEM_DSL_REGEN_ENABLED = True  # Trigger DSL regen from post-mortem
POSTMORTEM_DSL_REGEN_MIN_HIGH_CONF_WRONG = 2  # Min high-conf-wrong to trigger regen
POSTMORTEM_DSL_REGEN_BATCH_SIZE = 10  # How many problems before running batch regen

# =============================================================================
# STEP-LEVEL ANALYTICS
# =============================================================================
# Per CLAUDE.md: "DB audit for signature step level stats"
# Tracks per-step execution metrics for performance analysis.

STEP_STATS_ENABLED = True  # Master switch for step-level analytics
STEP_STATS_SAMPLE_RATE = 1.0  # Sample rate (1.0 = log all, 0.1 = log 10%)

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

# =============================================================================
# TREE-GUIDED DECOMPOSITION (Segmentation LLM)
# =============================================================================
# Per CLAUDE.md: "Route by what operations DO, not what they SOUND LIKE"
# Two-phase decomposition guided by signature tree vocabulary.

# Enable tree-guided decomposition (set False to use legacy single-pass planner)
TREE_GUIDED_DECOMPOSITION_ENABLED = os.getenv("TREE_GUIDED_DECOMPOSITION", "false").lower() == "true"

# Number of vocabulary suggestions to show per step
TREE_GUIDED_TOP_K_SUGGESTIONS = 3

# Welford's k-factor for novelty threshold: threshold = mean - k * stddev
# Higher k = more permissive (fewer steps marked as novel)
# Lower k = stricter (more steps marked as novel, encourages vocabulary reuse)
TREE_GUIDED_NOVELTY_K = 1.5

# Minimum samples before using Welford's threshold (cold start protection)
TREE_GUIDED_NOVELTY_MIN_SAMPLES = 10

# Default novelty threshold during cold start (before enough samples)
TREE_GUIDED_NOVELTY_DEFAULT_THRESHOLD = 0.5

# Tree-Planner Negotiation (per CLAUDE.md line 16-17)
# When enabled, planner and tree negotiate dag_step decomposition iteratively
# Bias: prefer decomposing dag_steps (cheap) over leaf_nodes (permanent tree change)
TREE_PLANNER_NEGOTIATION_ENABLED = os.getenv("TREE_PLANNER_NEGOTIATION", "true").lower() == "true"
TREE_PLANNER_NEGOTIATION_MAX_ROUNDS = 2
TREE_PLANNER_NEGOTIATION_SIMILARITY_THRESHOLD = 0.7

# Model configuration - set via TRAINING_MODE env var
# TRAINING_MODE=true  -> use beefy models for learning
# TRAINING_MODE=false -> use lightweight models for inference
TRAINING_MODEL = os.getenv("TRAINING_MODEL", "gpt-4o")  # Beefy model for training
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "gpt-4o-mini")  # Light model for inference

# Active model selection based on mode
DEFAULT_MODEL = TRAINING_MODEL if TRAINING_MODE else INFERENCE_MODEL
PLANNER_DEFAULT_MODEL = DEFAULT_MODEL
SOLVER_DEFAULT_MODEL = DEFAULT_MODEL
