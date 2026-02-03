"""Classify spans and text into mathematical operations.

NOTE: This is legacy/experimental code not actively used by the main pipeline.
The pattern matching here may still be useful for research or future development.

Maps detected spans/text to operations:
- "half the X" -> multiply(X, 0.5)
- "X more than Y" -> add(Y, X)
- "X percent of Y" -> multiply(Y, X/100)

Two approaches:
1. classify_span() - for attention-detected spans
2. find_all_patterns() - regex search on full text (60% detection on GSM8K)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from mycelium.types import Operation, Span

logger = logging.getLogger(__name__)


# Comprehensive math patterns (tested on GSM8K: 60% detection rate)
MATH_PATTERNS = [
    # Multiplication
    (r'half\s+(?:of\s+)?(?:that|the|this|as|what|her|his|its|a)', 'multiply', 0.5),
    (r'twice\s+(?:as|the|that|what|this)', 'multiply', 2),
    (r'(\d+(?:\.\d+)?)\s*times?\s*(?:as|the|that|more|what)?', 'multiply', None),
    (r'double[ds]?', 'multiply', 2),
    (r'triple[ds]?', 'multiply', 3),

    # Fractions
    (r'(?:one|a|1)\s*(?:/|-)?\s*third', 'fraction', 0.333),
    (r'(?:one|a|1)\s*(?:/|-)?\s*(?:quarter|fourth)', 'fraction', 0.25),
    (r'(?:one|a|1)\s*(?:/|-)?\s*half', 'fraction', 0.5),
    (r'(?:two|2)\s*(?:/|-)?\s*thirds', 'fraction', 0.667),
    (r'(?:three|3)\s*(?:/|-)?\s*(?:quarters|fourths)', 'fraction', 0.75),
    (r'(\d+)\s*/\s*(\d+)(?!\d)', 'fraction', None),

    # Percentages
    (r'(\d+(?:\.\d+)?)\s*%', 'percent', None),
    (r'(\d+(?:\.\d+)?)\s*percent', 'percent', None),

    # Addition
    (r'(\d+(?:\.\d+)?)\s*more\s+than', 'add', None),
    (r'plus\s*(\d+(?:\.\d+)?)', 'add', None),
    (r'add(?:ed|s|ing)?\s*(\d+(?:\.\d+)?)', 'add', None),
    (r'increase[ds]?\s*(?:by\s*)?(\d+(?:\.\d+)?)', 'add', None),
    (r'(\d+(?:\.\d+)?)\s*(?:more|additional|extra)', 'add', None),

    # Subtraction
    (r'(\d+(?:\.\d+)?)\s*(?:less|fewer)\s+than', 'subtract', None),
    (r'minus\s*(\d+(?:\.\d+)?)', 'subtract', None),
    (r'subtract(?:ed|s|ing)?\s*(\d+(?:\.\d+)?)', 'subtract', None),
    (r'decrease[ds]?\s*(?:by\s*)?(\d+(?:\.\d+)?)', 'subtract', None),
    (r'(?:spent|lost|gave away|used)\s*(\d+(?:\.\d+)?)', 'subtract', None),

    # Division / Rates
    (r'divide[ds]?\s*(?:by|into)\s*(\d+(?:\.\d+)?)', 'divide', None),
    (r'split\s*(?:into|between|among)\s*(\d+)', 'divide', None),
    (r'\$?(\d+(?:\.\d+)?)\s*(?:per|each|every|a)\s+\w+', 'rate', None),
    (r'(\d+(?:\.\d+)?)\s*(?:per|each|every)\s', 'rate', None),
]


def find_all_patterns(text: str) -> List[Operation]:
    """Find all math patterns in text using regex.

    This is the primary detection method - achieves 60% detection on GSM8K.

    Args:
        text: The problem text to search

    Returns:
        List of Operation objects found
    """
    text_lower = text.lower()
    found = []

    for pattern, op_type, default_value in MATH_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            # Extract value from match groups if no default
            if default_value is None and match.groups():
                try:
                    value = float(match.group(1))
                except (ValueError, IndexError):
                    value = match.groups()
            else:
                value = default_value

            # Skip very short matches (likely false positives)
            if len(match.group(0)) >= 3:
                found.append((
                    match.group(0),
                    op_type,
                    value,
                    match.start(),
                    match.end()
                ))

    # Remove overlapping matches (keep longer ones)
    found.sort(key=lambda x: (x[3], -(x[4] - x[3])))
    non_overlapping = []
    last_end = -1
    for matched, op_type, value, start, end in found:
        if start >= last_end:
            non_overlapping.append(Operation(
                op_type=op_type,
                value=value,
                matched_text=matched,
                confidence=0.8,
            ))
            last_end = end

    return non_overlapping


def classify_span(span: Span) -> Operation:
    """Classify a span into an operation.

    Uses the same patterns as find_all_patterns but on span text.

    Args:
        span: The span to classify

    Returns:
        Operation (may be 'variable' type if not recognized)
    """
    text = span.text.lower().strip()

    # Try each pattern
    for pattern, op_type, default_value in MATH_PATTERNS:
        match = re.search(pattern, text)
        if match:
            if default_value is None and match.groups():
                try:
                    value = float(match.group(1))
                except (ValueError, IndexError):
                    value = match.groups()
            else:
                value = default_value

            return Operation(
                op_type=op_type,
                value=value,
                matched_text=match.group(0),
                confidence=span.confidence,
                source_span=span,
            )

    # Check if it's just a number
    try:
        value = float(text.replace(",", ""))
        return Operation(
            op_type="value",
            value=value,
            matched_text=text,
            confidence=1.0,
            source_span=span,
        )
    except ValueError:
        pass

    # Unrecognized - treat as variable reference
    return Operation(
        op_type="variable",
        value=text,
        matched_text=text,
        confidence=0.3,
        source_span=span,
    )
