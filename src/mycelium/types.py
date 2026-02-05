"""Unified type definitions for Mycelium.

This module provides canonical dataclass definitions used throughout the
codebase. All components should import from here rather than defining
their own duplicate types.

Following the CLAUDE.md principle: "consolidate methods - for example all
database connections should go through a data layer instead of having
multiple database connections."
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Span:
    """A semantic span in the text.

    Used by attention-based span detection to identify coherent
    semantic units in text (e.g., "half the price of the cheese").
    """
    start: int          # Start token index
    end: int            # End token index (exclusive)
    tokens: List[str]   # The tokens in this span
    text: str           # Joined text
    confidence: float   # How strongly tokens attend to each other


@dataclass
class Operation:
    """A mathematical operation extracted from text.

    This is the canonical Operation dataclass for all pipeline components.
    Fields that aren't universally needed are Optional.
    """
    subgraph: Optional[Dict[str, Any]]  # SubGraphDSL dict for execution
    value: Any                          # The numeric value or operands (float or tuple)
    confidence: float                   # Classification confidence [0, 1]

    # Text fields (at least one should be set)
    span_text: Optional[str] = None     # The text span this operation came from
    matched_text: Optional[str] = None  # The text that matched (may differ from span_text)

    # Entity tracking
    entity: Optional[str] = None        # Who this applies to (e.g., "Lisa")
    reference: Optional[str] = None     # Reference entity for comparisons (e.g., "than Lisa")

    # Source tracking
    source_span: Optional[Span] = None  # The Span object this came from (if attention-based)

    # Dual-signal fields (for dual_signal_solver)
    embedding_sim: Optional[float] = None   # Embedding similarity score
    attention_sim: Optional[float] = None   # Attention similarity score
    template_id: Optional[str] = None       # Matched template ID
