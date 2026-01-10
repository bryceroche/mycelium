"""Mycelium configuration constants."""

# Similarity Thresholds
EXACT_MATCH_THRESHOLD = 0.87
VARIANT_THRESHOLD = 0.70
DEFAULT_INJECTION_THRESHOLD = 0.60  # Raised from 0.50 to reduce false matches

# Reliability Thresholds
RELIABILITY_MIN_USES = 3
RELIABILITY_MIN_SUCCESS_RATE = 0.70

# Exploration Parameters
EXPLORATION_MIN_LIFT = 0.0
EXPLORATION_MIN_CONFIDENCE = 0.5
EXPLORATION_RATE = 0.5  # Reduced from 1.0 to allow baseline sampling
EXPLORATION_UNPROVEN_RATE = 0.3  # Conservative for unproven signatures
USAGE_CONFIDENCE_DECAY = 5.0
COLD_START_GUARANTEED_USES = 10  # Reduced from 15 to faster lift-gating

# DSL Probation Mode
# When True: randomly sample new/improved DSLs to gather lift data
# When False: skip all probation sampling (use for max benchmark scores)
DSL_PROBATION_ENABLED = True

# DSL Injection Mode (for data collection vs benchmarking)
# When True: Always try DSL injection on every signature hit (data collection mode)
# When False: Only inject when confidence is high (benchmark mode)
DSL_AGGRESSIVE_INJECTION = True
DSL_MIN_CONFIDENCE = 0.0 if DSL_AGGRESSIVE_INJECTION else 0.3  # Min confidence to execute DSL
DSL_LLM_THRESHOLD = 1.0 if DSL_AGGRESSIVE_INJECTION else 0.5   # Use LLM param matching below this

# LLM Client Parameters
CLIENT_DEFAULT_TIMEOUT = 120.0
CLIENT_DEFAULT_TEMPERATURE = 0.3
PLANNER_DEFAULT_TEMPERATURE = 0.2
PLANNER_DEFAULT_MODEL = "llama-3.3-70b-versatile"
SOLVER_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Pipeline Parameters
PIPELINE_MIN_SIMILARITY = 0.60  # Raised from 0.50 to reduce false matches

# Recursive Decomposition (decompose-until-confident)
# When a step has low confidence, decompose it further into sub-steps
# Disabled in aggressive injection mode to maximize DSL execution attempts
RECURSIVE_DECOMPOSITION_ENABLED = not DSL_AGGRESSIVE_INJECTION
RECURSIVE_MAX_DEPTH = 2  # Maximum levels of re-decomposition
RECURSIVE_CONFIDENCE_THRESHOLD = 0.5  # Re-decompose if signature confidence below this
RECURSIVE_MAX_TOTAL_STEPS = 50  # Hard limit on total sub-steps per problem to prevent runaway
