"""Classify spans into mathematical operations.

Maps detected spans to operations:
- "half the X" → multiply(X, 0.5)
- "X more than Y" → add(Y, X)
- "X percent of Y" → multiply(Y, X/100)
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .span_detector import Span

logger = logging.getLogger(__name__)


@dataclass
class Operation:
    """A mathematical operation extracted from a span."""
    op_type: str           # "add", "subtract", "multiply", "divide", "value"
    operands: List[Any]    # Can be numbers, variable refs, or nested Operations
    source_span: Span      # The span this came from
    confidence: float      # Classification confidence


# Pattern-based classification (bootstrap, will be learned later)
SPAN_PATTERNS = [
    # (regex, op_type, operand_extractor)
    (r"half (?:the |of )?(.+)", "multiply", lambda m: [m.group(1), 0.5]),
    (r"twice (?:as many as |the )?(.+)", "multiply", lambda m: [m.group(1), 2]),
    (r"(\d+(?:\.\d+)?)\s*(?:times|x)\s+(.+)", "multiply", lambda m: [m.group(2), float(m.group(1))]),
    (r"(\d+(?:\.\d+)?)\s*more than\s+(.+)", "add", lambda m: [m.group(2), float(m.group(1))]),
    (r"(\d+(?:\.\d+)?)\s*less than\s+(.+)", "subtract", lambda m: [m.group(2), float(m.group(1))]),
    (r"(\d+(?:\.\d+)?)\s*percent of\s+(.+)", "multiply", lambda m: [m.group(2), float(m.group(1)) / 100]),
    (r"(\d+(?:\.\d+)?)\s*%\s*of\s+(.+)", "multiply", lambda m: [m.group(2), float(m.group(1)) / 100]),
    (r"the sum of\s+(.+)\s+and\s+(.+)", "add", lambda m: [m.group(1), m.group(2)]),
    (r"the difference (?:between|of)\s+(.+)\s+and\s+(.+)", "subtract", lambda m: [m.group(1), m.group(2)]),
    (r"the product of\s+(.+)\s+and\s+(.+)", "multiply", lambda m: [m.group(1), m.group(2)]),
    (r"(.+)\s+divided by\s+(.+)", "divide", lambda m: [m.group(1), m.group(2)]),
    (r"(.+)\s+plus\s+(.+)", "add", lambda m: [m.group(1), m.group(2)]),
    (r"(.+)\s+minus\s+(.+)", "subtract", lambda m: [m.group(1), m.group(2)]),
]


def classify_span(span: Span) -> Optional[Operation]:
    """Classify a span into an operation.

    Currently uses pattern matching. Will be replaced/augmented
    with learned classifier trained on (span, operation) pairs.

    Args:
        span: The span to classify

    Returns:
        Operation if classified, None if not recognized
    """
    text = span.text.lower().strip()

    for pattern, op_type, extractor in SPAN_PATTERNS:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            try:
                operands = extractor(match)
                return Operation(
                    op_type=op_type,
                    operands=operands,
                    source_span=span,
                    confidence=span.confidence,
                )
            except (ValueError, IndexError) as e:
                logger.debug(f"Pattern matched but extraction failed: {e}")
                continue

    # Check if it's just a number
    try:
        value = float(text.replace(",", ""))
        return Operation(
            op_type="value",
            operands=[value],
            source_span=span,
            confidence=1.0,
        )
    except ValueError:
        pass

    # Unrecognized span - could be a variable reference
    return Operation(
        op_type="variable",
        operands=[text],
        source_span=span,
        confidence=0.5,  # Low confidence for unknowns
    )


def build_computation_graph(
    spans: List[Span],
    cross_span_attention: Optional[Dict[Tuple[int, int], float]] = None,
) -> List[Operation]:
    """Build computation graph from classified spans.

    Uses cross-span attention to connect operations.

    Args:
        spans: Detected spans
        cross_span_attention: Attention between spans (optional)

    Returns:
        List of Operations forming a computation graph
    """
    operations = []

    for span in spans:
        op = classify_span(span)
        if op:
            operations.append(op)

    # TODO: Use cross_span_attention to link operations
    # e.g., if span A attends to span B, A's operand might be B's result

    return operations
