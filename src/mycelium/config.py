"""Mycelium configuration constants.

Two operating modes:
- TRAINING: Explore signatures, collect lift data, lower thresholds
- BENCHMARK: Conservative matching, proven patterns only, max accuracy
"""

from enum import Enum


class Mode(Enum):
    TRAINING = "training"
    BENCHMARK = "benchmark"


# =============================================================================
# ACTIVE MODE - Toggle this to switch between modes
# =============================================================================
ACTIVE_MODE = Mode.TRAINING

_is_training = ACTIVE_MODE == Mode.TRAINING
TRAINING_MODE = _is_training  # Exported for other modules


# =============================================================================
# MODE-DEPENDENT SETTINGS
# =============================================================================

# Signature Matching
MIN_MATCH_THRESHOLD = 0.92 if _is_training else 0.95

# DSL Injection
DSL_PROBATION_ENABLED = not _is_training
DSL_MIN_CONFIDENCE = 0.0 if _is_training else 0.3
DSL_LLM_THRESHOLD = 1.0 if _is_training else 0.5

# Exploration
EXPLORATION_RATE = 1.0 if _is_training else 0.5
EXPLORATION_UNPROVEN_RATE = 1.0 if _is_training else 0.3

# Recursive Decomposition
RECURSIVE_DECOMPOSITION_ENABLED = not _is_training


# =============================================================================
# MODE-INDEPENDENT SETTINGS
# =============================================================================

# Similarity Thresholds
MERGE_SIMILARITY_THRESHOLD = 0.95
VARIANT_THRESHOLD = 0.70
DEFAULT_INJECTION_THRESHOLD = 0.60
PIPELINE_MIN_SIMILARITY = 0.60
NEGATIVE_LIFT_SIMILARITY = 0.85  # Skip DSL if similar to known negative-lift sig

# Reliability Thresholds
RELIABILITY_MIN_USES = 3
RELIABILITY_MIN_SUCCESS_RATE = 0.70

# Exploration Parameters
EXPLORATION_MIN_LIFT = 0.0
EXPLORATION_MIN_CONFIDENCE = 0.5
USAGE_CONFIDENCE_DECAY = 5.0
COLD_START_GUARANTEED_USES = 10
PROBATION_INJECTION_RATE = 0.3
AVOID_CACHE_TTL = 300.0  # Rebuild negative-lift cache every 5 min

# Recursive Decomposition
RECURSIVE_MAX_DEPTH = 2
RECURSIVE_CONFIDENCE_THRESHOLD = 0.5
RECURSIVE_MAX_TOTAL_STEPS = 50

# DSL Execution
DSL_TIMEOUT_SEC = 1.0
DSL_SEMANTIC_MIN_CONFIDENCE = 0.5

# LLM Client
CLIENT_DEFAULT_TIMEOUT = 120.0
CLIENT_DEFAULT_TEMPERATURE = 0.3
CLIENT_CONNECT_TIMEOUT = 10.0
CLIENT_BASE_RETRY_DELAY = 1.0
CLIENT_MAX_RETRY_DELAY = 30.0
PLANNER_DEFAULT_TEMPERATURE = 0.2
PLANNER_DEFAULT_MODEL = "llama-3.3-70b-versatile"
SOLVER_DEFAULT_MODEL = "llama-3.3-70b-versatile"
