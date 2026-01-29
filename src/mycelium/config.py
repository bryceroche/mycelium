"""Mycelium configuration - minimal for local decomposition architecture.

Per CLAUDE.md "The Flow": DB Stats → Welford → Tree Structure
Thresholds come from config, not magic numbers.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
DB_PATH = os.getenv("MYCELIUM_DB_PATH", str(Path.home() / ".mycelium" / "signatures.db"))
DB_PROTECTED = os.getenv("MYCELIUM_DB_PROTECTED", "true").lower() in ("true", "1", "yes")
TRAINING_MODE = os.getenv("MYCELIUM_TRAINING_MODE", "true").lower() in ("true", "1", "yes")

# =============================================================================
# EMBEDDING
# =============================================================================
EMBEDDING_DIM = 768  # text-embedding-3-small
EMBEDDING_PROVIDER = os.getenv("MYCELIUM_EMBEDDING_PROVIDER", "openai")
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_CACHE_ENABLED = True
EMBEDDING_CACHE_MEMORY_SIZE = 10000
EMBEDDING_CACHE_PERSIST = True
EMBEDDING_CACHE_WARM_ON_START = False
EMBEDDING_CACHE_TTL_DAYS = 30

# =============================================================================
# SIGNATURE MATCHING (per CLAUDE.md "The Flow")
# =============================================================================
MIN_MATCH_THRESHOLD = 0.85
MIN_MATCH_THRESHOLD_COLD_START = 0.92
MIN_MATCH_RAMP_SIGNATURES = 50

ROUTING_MIN_SIMILARITY = 0.85
ROUTING_MIN_SIMILARITY_PERMISSIVE = 0.5

# =============================================================================
# PLACEMENT THRESHOLDS
# =============================================================================
PLACEMENT_DEDUP_SIMILARITY = 0.97
PLACEMENT_CHILD_SIMILARITY = 0.75
PLACEMENT_SIBLING_SIMILARITY = 0.50
PLACEMENT_MIN_SIMILARITY = PLACEMENT_CHILD_SIMILARITY

# =============================================================================
# CLUSTER THRESHOLDS (per CLAUDE.md "New Favorite Pattern")
# =============================================================================
CLUSTER_THRESHOLD_CV_CUTOFF = 0.1
CLUSTER_THRESHOLD_STD_MULTIPLIER = 2.0
CLUSTER_THRESHOLD_PERCENTILE = 0.03
CLUSTER_THRESHOLD_MIN = 0.85
CLUSTER_THRESHOLD_MAX = 0.95
CLUSTER_THRESHOLD_COLD_START = 0.90

# =============================================================================
# ADAPTIVE THRESHOLDS (Welford-based)
# =============================================================================
ADAPTIVE_THRESHOLD_MIN_SAMPLES = 50
ADAPTIVE_THRESHOLD_K = 1.5
ADAPTIVE_THRESHOLD_MIN = 0.70
ADAPTIVE_THRESHOLD_MAX = 0.95

# =============================================================================
# REJECTION TRACKING
# =============================================================================
REJECTION_COUNT_THRESHOLD = 5
REJECTION_COUNT_THRESHOLD_COLD = 3
REJECTION_COUNT_THRESHOLD_MATURE = 10
REJECTION_COUNT_RAMP_SIGNATURES = 100
REJECTION_RATE_THRESHOLD = 0.30
REJECTION_RATE_WELFORD_ENABLED = True
REJECTION_RATE_MIN_SAMPLES = 5
REJECTION_RATE_K = 1.0

# Adaptive rejection thresholds
ADAPTIVE_REJECTION_K = 2.0
ADAPTIVE_REJECTION_MIN_SAMPLES = 5
ADAPTIVE_REJECTION_DEFAULT_THRESHOLD = 0.75

# =============================================================================
# DSL EXECUTION
# =============================================================================
DSL_TIMEOUT_SEC = 1.0

# =============================================================================
# UMBRELLA ROUTING
# =============================================================================
UMBRELLA_MAX_DEPTH = 10
UMBRELLA_ROUTING_THRESHOLD = 0.75

# =============================================================================
# GRAPH ROUTING (computation graph embeddings)
# =============================================================================
GRAPH_ROUTING_ENABLED = True
GRAPH_ROUTING_MIN_SIMILARITY = 0.7
GRAPH_ROUTING_BOOST_FACTOR = 0.1

# =============================================================================
# CREDIT PROPAGATION
# =============================================================================
CREDIT_PROPAGATION_DECAY = 0.5
CREDIT_PROPAGATION_MAX_DEPTH = 3
PARENT_CREDIT_DECAY = 0.5
PARENT_CREDIT_MAX_DEPTH = 3
PARENT_CREDIT_MIN = 0.01

# =============================================================================
# COLD START
# =============================================================================
COLD_START_SIGNATURE_THRESHOLD = 20
COLD_START_PROBLEMS_THRESHOLD = 50

# =============================================================================
# CACHES
# =============================================================================
SIGNATURE_CACHE_MAX_SIZE = 1000
SIGNATURE_CACHE_TTL_SECONDS = 300
CHILDREN_CACHE_MAX_SIZE = 500
CENTROID_CACHE_MAX_SIZE = 500
GRAPH_EMBEDDING_CACHE_MAX_SIZE = 1000

# =============================================================================
# STEP STATS
# =============================================================================
STEP_STATS_ENABLED = True
STEP_STATS_SAMPLE_RATE = 1.0
POSITION_STATS_ENABLED = True
POSITION_STATS_WEIGHT = 0.1
POSITION_STATS_MIN_OBS = 5

# =============================================================================
# EXPLORATION
# =============================================================================
EXPLORATION_EPSILON = 0.1
MIN_FORK_DEPTH = 2

# =============================================================================
# RESTRUCTURE
# =============================================================================
RESTRUCTURE_INTERVAL = 10
RESTRUCTURE_MERGE_FLOOR = 0.85
RESTRUCTURE_OUTLIER_IMPROVEMENT = 0.05

# =============================================================================
# BIG BANG (signature creation)
# =============================================================================
BIG_BANG_TARGET_SIGNATURES = 100
BIG_BANG_TAU_DIVISOR = 50
BIG_BANG_FORK_CENTER_DRIFT_RATE = 0.1

# =============================================================================
# AUTO DEMOTE
# =============================================================================
AUTO_DEMOTE_ENABLED = True
AUTO_DEMOTE_MAX_SUCCESS_RATE = 0.20
AUTO_DEMOTE_EXCLUDED_TYPES = ["decompose"]
AUTO_DEMOTE_RAMP_DIVISOR = 2000
AUTO_DEMOTE_MIN_USES_FLOOR = 3
AUTO_DEMOTE_MIN_USES_CAP = 5

# =============================================================================
# CENTROID
# =============================================================================
CENTROID_PROPAGATION_MAX_DEPTH = 5
CENTROID_MAX_DRIFT = 0.3
CENTROID_DRIFT_DECAY = 0.9

# =============================================================================
# DATABASE
# =============================================================================
DB_MAX_RETRIES = 3
DB_BASE_RETRY_DELAY = 0.1

# =============================================================================
# ATOMIC OPERATION DETECTION
# =============================================================================
ATOMIC_SIMILARITY_THRESHOLD = 0.70
ATOMIC_GAP_THRESHOLD = 0.03
FORK_GAP_SCALING_FACTOR = 2.0
ROUTING_BEST_MATCH_MIN_SIMILARITY = 0.8
PLACEMENT_WELL_COVERED_ZSCORE = -0.5
PLACEMENT_WELL_COVERED_SIMILARITY = 0.85
HINT_ALTERNATIVES_MIN_SIMILARITY = 0.3
NEW_CHILD_SIMILARITY_THRESHOLD = 0.75

# =============================================================================
# MISC
# =============================================================================
EMBEDDING_MODEL = "text-embedding-3-small"
ADAPTIVE_REJECTION_ENABLED = True
