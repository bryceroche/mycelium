"""Attention-based decomposition utilities.

NOTE: This module is NOT actively used by the main mycelium pipeline.
The active SpanDetector lives in dual_signal_templates.py.

This module contains experimental utilities for attention-based math problem
decomposition that may be useful for research or future development:

- extractor.py: Extract attention matrices from transformers (requires GPU + model weights)
- span_detector.py: Cluster tokens into semantic spans using attention patterns

The core insight these utilities explore: transformer attention can reveal
semantic units (e.g., "half the price of the cheese" as ONE span).

Usage:
    # These are optional utilities - import explicitly if needed
    from mycelium.attention.extractor import extract_attention
"""

# Minimal exports - Operation type is re-exported for convenience
# but the canonical definition is in mycelium.types
from mycelium.types import Operation

__all__ = [
    "Operation",
]
