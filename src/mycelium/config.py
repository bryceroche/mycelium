"""Mycelium configuration constants."""

from enum import Enum


class Mode(Enum):
    TRAINING = "training"    # Explore signatures, collect data, lower thresholds
    BENCHMARK = "benchmark"  # Conservative matching, proven patterns only


# === ACTIVE MODE ===
# Toggle this to switch between training and benchmark modes
ACTIVE_MODE = Mode.TRAINING


# === Mode-dependent settings ===
_is_training = ACTIVE_MODE == Mode.TRAINING
TRAINING_MODE = _is_training  # Exported for other modules

# Similarity Thresholds
MIN_MATCH_THRESHOLD = 0.92 if _is_training else 0.95  # Higher in benchmark mode
VARIANT_THRESHOLD = 0.70
DEFAULT_INJECTION_THRESHOLD = 0.60
MERGE_SIMILARITY_THRESHOLD = 0.95

# Exploration Parameters
EXPLORATION_RATE = 1.0 if _is_training else 0.5
EXPLORATION_UNPROVEN_RATE = 1.0 if _is_training else 0.3
EXPLORATION_MIN_LIFT = 0.0
EXPLORATION_MIN_CONFIDENCE = 0.5
USAGE_CONFIDENCE_DECAY = 5.0
COLD_START_GUARANTEED_USES = 10

# DSL Injection
DSL_PROBATION_ENABLED = not _is_training  # Disabled in training to try everything
DSL_MIN_CONFIDENCE = 0.0 if _is_training else 0.3
DSL_LLM_THRESHOLD = 1.0 if _is_training else 0.5

# Recursive Decomposition
RECURSIVE_DECOMPOSITION_ENABLED = not _is_training  # Disabled in training mode
RECURSIVE_MAX_DEPTH = 2
RECURSIVE_CONFIDENCE_THRESHOLD = 0.5
RECURSIVE_MAX_TOTAL_STEPS = 50

# === Mode-independent settings ===

# Reliability Thresholds
RELIABILITY_MIN_USES = 3
RELIABILITY_MIN_SUCCESS_RATE = 0.70

# LLM Client Parameters
CLIENT_DEFAULT_TIMEOUT = 120.0
CLIENT_DEFAULT_TEMPERATURE = 0.3
PLANNER_DEFAULT_TEMPERATURE = 0.2
PLANNER_DEFAULT_MODEL = "llama-3.3-70b-versatile"
SOLVER_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Pipeline Parameters
PIPELINE_MIN_SIMILARITY = 0.60
