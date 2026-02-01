"""Mycelium configuration - minimal for flat prototype architecture.

Keep it simple: only config values that are actually used.
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
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072  # text-embedding-3-large dimensions
EMBEDDING_PROVIDER = os.getenv("MYCELIUM_EMBEDDING_PROVIDER", "openai")

# =============================================================================
# SIGNATURE MATCHING
# =============================================================================
MIN_MATCH_THRESHOLD = 0.85

# =============================================================================
# ADAPTIVE THRESHOLDS (Welford-based)
# =============================================================================
ADAPTIVE_THRESHOLD_MIN_SAMPLES = 50
ADAPTIVE_THRESHOLD_K = 1.5
ADAPTIVE_THRESHOLD_MIN = 0.70
ADAPTIVE_THRESHOLD_MAX = 0.95

# =============================================================================
# CACHES
# =============================================================================
CENTROID_CACHE_MAX_SIZE = 500
