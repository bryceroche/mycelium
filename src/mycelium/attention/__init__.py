"""Attention-based math problem decomposition.

Pipeline:
1. extractor.py   - Pull attention matrices from transformer (DeepSeek-Math)
2. span_detector.py - Cluster tokens into semantic spans
3. classifier.py   - Map spans to operations

The key insight: transformer attention reveals semantic units.
"half the price of the cheese" is ONE span, not four words.
"""

from .extractor import extract_attention
from .span_detector import detect_spans
from .classifier import classify_span

__all__ = ["extract_attention", "detect_spans", "classify_span"]
