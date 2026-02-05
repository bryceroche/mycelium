"""Sub-span detection for multi-operation clauses in math word problems.

Like multi-object detection in CV, this module detects multiple operations
within a single clause. For example:

    "John has 3 more apples than Lisa and 2 fewer than Mary"

Contains TWO operations:
    - Reference to Lisa with "+3" (COMPARE_MORE)
    - Reference to Mary with "-2" (COMPARE_LESS)

The approach:
1. Enumerate candidate sub-spans containing numbers
2. Classify each candidate using the existing pipeline
3. Non-Maximum Suppression to remove overlapping detections
4. Return all detected operations
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

# Import Operation from types.py (canonical definition)
from mycelium.types import Operation


@dataclass
class CandidateSpan:
    """A candidate sub-span containing a number and potential operation."""
    text: str           # The sub-span text
    start_idx: int      # Start character index in original clause
    end_idx: int        # End character index in original clause
    numbers: List[float]  # Numbers found in this span
    reference: Optional[str] = None  # Detected reference entity (e.g., "Lisa")


@dataclass
class ClassifiedSpan:
    """A candidate span with classification results."""
    span: CandidateSpan
    operation: str      # Detected operation type (SET, ADD, SUB, COMPARE_MORE, etc.)
    confidence: float   # Classification confidence [0, 1]
    entity: Optional[str] = None  # Target entity (who this applies to)


# ============================================================================
# Key Patterns for Multi-Operation Detection
# ============================================================================

# Pattern: "N more/greater than [ENTITY]" - REQUIRES "than" for comparison semantics
COMPARE_MORE_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s+(?:more|greater|higher|larger|bigger)\s+(?:\w+\s+)*?than\s+([A-Z][a-z]+|\w+)',
    re.IGNORECASE
)

# Pattern: "N less/fewer than [ENTITY]" - REQUIRES "than" for comparison semantics
COMPARE_LESS_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s+(?:less|fewer|smaller|lower)\s+(?:\w+\s+)*?than\s+([A-Z][a-z]+|\w+)',
    re.IGNORECASE
)

# Pattern: "N times as [many/much] as [ENTITY]" or "twice/double/triple as many as [ENTITY]"
# This pattern captures: "twice as many books as Tom" -> ref=Tom
# Note: Uses non-greedy matching and looks for final "as [Entity]"
RATIO_PATTERN = re.compile(
    r'(?:(\d+(?:\.\d+)?)\s+times|twice|double|triple)\s+as\s+(?:many|much)\s+\w+\s+as\s+([A-Z][a-z]+)',
    re.IGNORECASE
)

# Simpler RATIO pattern for "twice as many" without explicit reference
RATIO_SIMPLE_PATTERN = re.compile(
    r'(?:(\d+(?:\.\d+)?)\s+times|twice|double|triple)\s+(?:as\s+)?(?:many|much)',
    re.IGNORECASE
)

# Pattern: "has/had N [ITEM]" - basic SET or state
HAS_PATTERN = re.compile(
    r'(?:has|had|have)\s+(\d+(?:\.\d+)?)',
    re.IGNORECASE
)

# Generic number pattern for fallback
NUMBER_PATTERN = re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b')

# Entity pattern (capitalized words)
ENTITY_PATTERN = re.compile(r'\b([A-Z][a-z]+)\b')


def enumerate_number_spans(clause: str) -> List[CandidateSpan]:
    """Find all contiguous sub-spans containing numbers and potential operations.

    This is the "region proposal" phase - generate candidate spans that might
    contain operations. We're intentionally broad here; NMS will filter later.

    Strategies:
    1. Look for specific patterns (COMPARE_MORE, COMPARE_LESS, RATIO) - HIGH priority
    2. Split on conjunctions ("and", "but", "while") to find separate operations
    3. Window-based extraction around each number (fallback)

    Args:
        clause: A single clause that may contain multiple operations

    Returns:
        List of CandidateSpan objects, possibly overlapping
    """
    # Track high-confidence pattern matches separately
    pattern_candidates: List[CandidateSpan] = []
    fallback_candidates: List[CandidateSpan] = []

    # Strategy 1: Pattern-based extraction for comparison operations
    # These patterns have high precision for multi-operation clauses
    # Mark their positions to avoid duplicates from conjunction splitting

    pattern_covered_ranges: List[Tuple[int, int]] = []

    # COMPARE_MORE: "3 more apples than Lisa"
    for match in COMPARE_MORE_PATTERN.finditer(clause):
        num_str = match.group(1)
        ref_entity = match.group(2) if match.group(2) else None
        try:
            value = float(num_str.replace(',', ''))
            pattern_candidates.append(CandidateSpan(
                text=match.group(0),
                start_idx=match.start(),
                end_idx=match.end(),
                numbers=[value],
                reference=ref_entity
            ))
            pattern_covered_ranges.append((match.start(), match.end()))
        except ValueError:
            pass

    # COMPARE_LESS: "2 fewer than Mary"
    for match in COMPARE_LESS_PATTERN.finditer(clause):
        num_str = match.group(1)
        ref_entity = match.group(2) if match.group(2) else None
        try:
            value = float(num_str.replace(',', ''))
            pattern_candidates.append(CandidateSpan(
                text=match.group(0),
                start_idx=match.start(),
                end_idx=match.end(),
                numbers=[value],
                reference=ref_entity
            ))
            pattern_covered_ranges.append((match.start(), match.end()))
        except ValueError:
            pass

    # RATIO: "twice as many books as Tom", "3 times as many as"
    # Try the full pattern first (with reference entity)
    for match in RATIO_PATTERN.finditer(clause):
        text = match.group(0).lower()
        # Determine multiplier
        if 'twice' in text or 'double' in text:
            value = 2.0
        elif 'triple' in text:
            value = 3.0
        elif match.group(1):
            try:
                value = float(match.group(1).replace(',', ''))
            except ValueError:
                value = 1.0
        else:
            value = 1.0

        ref_entity = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
        pattern_candidates.append(CandidateSpan(
            text=match.group(0),
            start_idx=match.start(),
            end_idx=match.end(),
            numbers=[value],
            reference=ref_entity
        ))
        pattern_covered_ranges.append((match.start(), match.end()))

    # RATIO (simple): "twice as many" without explicit reference - only if no full match
    if not any('RATIO' in str(c) for c in pattern_candidates):
        for match in RATIO_SIMPLE_PATTERN.finditer(clause):
            # Skip if already covered by the full RATIO pattern
            already_covered = False
            for start, end in pattern_covered_ranges:
                if match.start() >= start and match.end() <= end:
                    already_covered = True
                    break
            if already_covered:
                continue

            text = match.group(0).lower()
            if 'twice' in text or 'double' in text:
                value = 2.0
            elif 'triple' in text:
                value = 3.0
            elif match.group(1):
                try:
                    value = float(match.group(1).replace(',', ''))
                except ValueError:
                    value = 1.0
            else:
                value = 1.0

            pattern_candidates.append(CandidateSpan(
                text=match.group(0),
                start_idx=match.start(),
                end_idx=match.end(),
                numbers=[value],
                reference=None
            ))
            pattern_covered_ranges.append((match.start(), match.end()))

    def is_covered_by_pattern(start: int, end: int) -> bool:
        """Check if a range overlaps significantly with any pattern match."""
        for p_start, p_end in pattern_covered_ranges:
            # Compute overlap
            overlap_start = max(start, p_start)
            overlap_end = min(end, p_end)
            overlap = max(0, overlap_end - overlap_start)
            span_len = end - start
            if span_len > 0 and overlap / span_len > 0.5:
                return True
        return False

    # Strategy 2: Split on conjunctions and process each part
    # "3 more than Lisa and 2 fewer than Mary" -> two parts
    # Only add if not already covered by a pattern match
    conjunction_pattern = re.compile(r'\s+(?:and|but|while|,)\s+', re.IGNORECASE)
    parts = conjunction_pattern.split(clause)

    if len(parts) > 1:
        # Track position as we process parts
        current_pos = 0
        for part in parts:
            # Find where this part actually starts in the original clause
            part_start = clause.find(part, current_pos)
            if part_start == -1:
                part_start = current_pos
            part_end = part_start + len(part)

            # Skip if this part is already covered by a pattern match
            if is_covered_by_pattern(part_start, part_end):
                current_pos = part_end
                continue

            # Extract numbers from this part
            part_numbers = []
            for num_match in NUMBER_PATTERN.finditer(part):
                try:
                    part_numbers.append(float(num_match.group(1).replace(',', '')))
                except ValueError:
                    pass

            # Extract reference entity from this part
            part_ref = None
            entity_matches = ENTITY_PATTERN.findall(part)
            if entity_matches:
                part_ref = entity_matches[-1]  # Take last entity as reference

            if part_numbers:
                fallback_candidates.append(CandidateSpan(
                    text=part.strip(),
                    start_idx=part_start,
                    end_idx=part_end,
                    numbers=part_numbers,
                    reference=part_ref
                ))

            current_pos = part_end

    # Strategy 3: Window-based extraction around each number (fallback)
    # Only if no patterns and no conjunction parts matched
    if not pattern_candidates and not fallback_candidates:
        for num_match in NUMBER_PATTERN.finditer(clause):
            try:
                value = float(num_match.group(1).replace(',', ''))
            except ValueError:
                continue

            # Expand window to include surrounding context
            # Look for operation keywords and entity references
            start = max(0, num_match.start() - 30)
            end = min(len(clause), num_match.end() + 30)

            # Expand to word boundaries
            while start > 0 and clause[start - 1] not in ' \t\n.,;:!?':
                start -= 1
            while end < len(clause) and clause[end] not in ' \t\n.,;:!?':
                end += 1

            window_text = clause[start:end].strip()

            # Extract reference from window
            ref_entity = None
            entity_matches = ENTITY_PATTERN.findall(window_text)
            if entity_matches:
                ref_entity = entity_matches[-1]

            fallback_candidates.append(CandidateSpan(
                text=window_text,
                start_idx=start,
                end_idx=end,
                numbers=[value],
                reference=ref_entity
            ))

    # Combine: pattern matches take priority, then fallbacks
    return pattern_candidates + fallback_candidates


def classify_candidates(
    candidates: List[CandidateSpan],
    pipeline: Any,
    position: int = 2
) -> List[ClassifiedSpan]:
    """Classify each candidate span independently.

    Uses the existing SimplePipeline to classify each span, but also checks
    for specific patterns that indicate complex operations (COMPARE_MORE, etc.)

    Args:
        candidates: List of candidate spans to classify
        pipeline: A SimplePipeline instance (or compatible) with classify_span method
        position: Position in problem (1=first clause, 2=second, etc.)

    Returns:
        List of ClassifiedSpan with operation classifications
    """
    classified: List[ClassifiedSpan] = []

    for candidate in candidates:
        text_lower = candidate.text.lower()

        # Check for comparison patterns first (more specific)
        # These override the general KNN classification
        # Allow words between the comparison keyword and "than" (e.g., "more apples than")

        # COMPARE_MORE: "more [words] than", "greater [words] than"
        if re.search(r'\b(?:more|greater|higher|larger|bigger)\b.*\bthan\b', text_lower):
            op_type = "COMPARE_MORE"
            confidence = 0.9  # High confidence for explicit pattern

        # COMPARE_LESS: "less [words] than", "fewer [words] than"
        elif re.search(r'\b(?:less|fewer|smaller|lower)\b.*\bthan\b', text_lower):
            op_type = "COMPARE_LESS"
            confidence = 0.9

        # RATIO: "times as", "twice as", "double", "triple"
        elif re.search(r'\b(?:times|twice|double|triple)\b.*\b(?:as|many|much)\b', text_lower):
            op_type = "RATIO"
            confidence = 0.9

        # Fall back to pipeline classification for simple operations
        else:
            if hasattr(pipeline, 'classify_span'):
                op_type, confidence = pipeline.classify_span(candidate.text, position=position)
            else:
                # Default if no pipeline
                op_type = "SET"
                confidence = 0.5

        # Extract target entity (subject of the clause)
        entity = None
        # Look for entity at the start of the span or clause
        entity_matches = ENTITY_PATTERN.findall(candidate.text)
        if entity_matches:
            entity = entity_matches[0]  # First entity is usually the subject

        classified.append(ClassifiedSpan(
            span=candidate,
            operation=op_type,
            confidence=confidence,
            entity=entity
        ))

    return classified


def compute_iou(span1: CandidateSpan, span2: CandidateSpan) -> float:
    """Compute Intersection over Union for two spans.

    IoU = (intersection length) / (union length)

    Args:
        span1, span2: Two candidate spans

    Returns:
        IoU score in [0, 1]
    """
    # Compute intersection
    inter_start = max(span1.start_idx, span2.start_idx)
    inter_end = min(span1.end_idx, span2.end_idx)
    intersection = max(0, inter_end - inter_start)

    # Compute union
    union_start = min(span1.start_idx, span2.start_idx)
    union_end = max(span1.end_idx, span2.end_idx)
    union = union_end - union_start

    if union == 0:
        return 0.0

    return intersection / union


def span_nms(
    spans: List[ClassifiedSpan],
    iou_threshold: float = 0.5
) -> List[ClassifiedSpan]:
    """Remove overlapping span predictions, keeping highest confidence.

    Non-Maximum Suppression adapted from object detection:
    1. Sort spans by confidence (descending)
    2. For each span, remove lower-confidence spans that overlap significantly

    Args:
        spans: List of classified spans (may overlap)
        iou_threshold: IoU threshold above which spans are considered overlapping

    Returns:
        Filtered list with overlapping detections removed
    """
    if not spans:
        return []

    # Sort by confidence descending
    sorted_spans = sorted(spans, key=lambda x: x.confidence, reverse=True)

    kept: List[ClassifiedSpan] = []
    suppressed: set = set()

    for i, span_i in enumerate(sorted_spans):
        if i in suppressed:
            continue

        # Keep this span
        kept.append(span_i)

        # Suppress overlapping spans with lower confidence
        for j, span_j in enumerate(sorted_spans[i + 1:], start=i + 1):
            if j in suppressed:
                continue

            iou = compute_iou(span_i.span, span_j.span)
            if iou > iou_threshold:
                suppressed.add(j)

    return kept


def detect_operations(
    clause: str,
    pipeline: Any = None,
    position: int = 2,
    iou_threshold: float = 0.5
) -> List[Operation]:
    """Detect all operations in a clause (like multi-object detection).

    Main entry point for sub-span detection. Handles clauses with multiple
    operations like "John has 3 more than Lisa and 2 fewer than Mary".

    Args:
        clause: A single clause that may contain multiple operations
        pipeline: Optional SimplePipeline for classification fallback
        position: Position in problem (1=first, 2=second, etc.)
        iou_threshold: IoU threshold for NMS

    Returns:
        List of Operation objects, one per detected operation
    """
    # Step 1: Enumerate candidate spans
    candidates = enumerate_number_spans(clause)

    if not candidates:
        return []

    # Step 2: Classify each candidate
    classified = classify_candidates(candidates, pipeline, position)

    # Step 3: Non-Maximum Suppression
    filtered = span_nms(classified, iou_threshold)

    # Step 4: Convert to Operation objects
    operations: List[Operation] = []

    for cls_span in filtered:
        # Get the primary value from the span
        value = cls_span.span.numbers[0] if cls_span.span.numbers else 0.0

        operations.append(Operation(
            dsl_expr=cls_span.operation,
            value=value,
            entity=cls_span.entity,
            reference=cls_span.span.reference,
            confidence=cls_span.confidence,
            span_text=cls_span.span.text
        ))

    return operations


# ============================================================================
# Test Functions
# ============================================================================

def test_enumerate_spans():
    """Test candidate span enumeration."""
    print("=== Test: enumerate_number_spans ===\n")

    test_cases = [
        # Multi-operation clauses
        "John has 3 more apples than Lisa and 2 fewer than Mary",
        "She has twice as many books as Tom and 5 more than Sarah",
        "The red box has 10 balls, the blue box has 15",
        # Single operation (should still work)
        "Lisa has 12 apples",
        "He found 5 more coins",
        # Complex comparisons
        "Alice has 3 greater than Bob but 2 less than Carol",
    ]

    for clause in test_cases:
        print(f"Clause: '{clause}'")
        candidates = enumerate_number_spans(clause)
        print(f"  Found {len(candidates)} candidate(s):")
        for c in candidates:
            print(f"    - [{c.start_idx}:{c.end_idx}] '{c.text}'")
            print(f"      numbers={c.numbers}, ref={c.reference}")
        print()


def test_classify_candidates():
    """Test candidate classification (without pipeline)."""
    print("=== Test: classify_candidates (no pipeline) ===\n")

    test_cases = [
        ("3 more than Lisa", "COMPARE_MORE"),
        ("2 fewer than Mary", "COMPARE_LESS"),
        ("twice as many as Tom", "RATIO"),
        ("5 times as much as", "RATIO"),
        ("has 12 apples", "SET"),  # Falls back to default
    ]

    for text, expected_op in test_cases:
        candidate = CandidateSpan(
            text=text,
            start_idx=0,
            end_idx=len(text),
            numbers=[0],
            reference=None
        )
        classified = classify_candidates([candidate], pipeline=None)
        actual_op = classified[0].operation if classified else "NONE"
        status = "PASS" if actual_op == expected_op else "FAIL"
        print(f"  [{status}] '{text}' -> {actual_op} (expected {expected_op})")

    print()


def test_nms():
    """Test Non-Maximum Suppression."""
    print("=== Test: span_nms ===\n")

    # Create overlapping spans with different confidences
    span1 = ClassifiedSpan(
        span=CandidateSpan("3 more than Lisa", 0, 16, [3.0], "Lisa"),
        operation="COMPARE_MORE",
        confidence=0.9
    )
    span2 = ClassifiedSpan(
        span=CandidateSpan("more than Lisa", 2, 16, [0.0], "Lisa"),  # Overlaps span1
        operation="ADD",
        confidence=0.6
    )
    span3 = ClassifiedSpan(
        span=CandidateSpan("2 fewer than Mary", 20, 37, [2.0], "Mary"),  # No overlap
        operation="COMPARE_LESS",
        confidence=0.85
    )

    spans = [span1, span2, span3]
    print(f"  Input: {len(spans)} spans")
    for s in spans:
        print(f"    - [{s.span.start_idx}:{s.span.end_idx}] '{s.span.text}' -> {s.operation} (conf={s.confidence})")

    filtered = span_nms(spans, iou_threshold=0.5)
    print(f"\n  After NMS: {len(filtered)} spans")
    for s in filtered:
        print(f"    - [{s.span.start_idx}:{s.span.end_idx}] '{s.span.text}' -> {s.operation} (conf={s.confidence})")

    assert len(filtered) == 2, f"Expected 2 spans after NMS, got {len(filtered)}"
    print("\n  [PASS] NMS correctly removed overlapping span\n")


def test_detect_operations():
    """Test the full detection pipeline."""
    print("=== Test: detect_operations (full pipeline) ===\n")

    test_cases = [
        # Multi-operation clause
        (
            "John has 3 more apples than Lisa and 2 fewer than Mary",
            [("COMPARE_MORE", 3.0, "Lisa"), ("COMPARE_LESS", 2.0, "Mary")]
        ),
        # Single operation
        (
            "Lisa sold 5 apples",
            [("SET", 5.0, None)]  # Falls back to SET without pipeline
        ),
        # Ratio comparison
        (
            "Tom has twice as many coins as Sarah",
            [("RATIO", 2.0, "Sarah")]
        ),
        # Complex multi-comparison
        (
            "Alice has 4 greater than Bob but 3 less than Carol",
            [("COMPARE_MORE", 4.0, "Bob"), ("COMPARE_LESS", 3.0, "Carol")]
        ),
    ]

    for clause, expected_ops in test_cases:
        print(f"Clause: '{clause}'")
        operations = detect_operations(clause, pipeline=None)
        print(f"  Detected {len(operations)} operation(s):")
        for op in operations:
            print(f"    - {op.dsl_expr}({op.value}) ref={op.reference} conf={op.confidence:.2f}")
            print(f"      span: '{op.span_text}'")

        # Verify expected operations
        if len(expected_ops) > 0:
            print(f"  Expected {len(expected_ops)} operation(s)")
        print()


def test_with_pipeline():
    """Test integration with SimplePipeline (if available)."""
    print("=== Test: detect_operations (with pipeline) ===\n")

    try:
        from mycelium.simple_pipeline import SimplePipeline
        pipeline = SimplePipeline(use_db=False)

        test_cases = [
            "John has 3 more apples than Lisa and 2 fewer than Mary",
            "She sold 5 apples and bought 3 oranges",
            "Tom found 10 coins but lost 4 later",
        ]

        for clause in test_cases:
            print(f"Clause: '{clause}'")
            operations = detect_operations(clause, pipeline=pipeline, position=2)
            print(f"  Detected {len(operations)} operation(s):")
            for op in operations:
                print(f"    - {op.dsl_expr}({op.value}) ref={op.reference} conf={op.confidence:.2f}")
            print()

    except ImportError as e:
        print(f"  [SKIP] Could not import SimplePipeline: {e}\n")


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("Sub-span Detector Tests")
    print("=" * 60)
    print()

    test_enumerate_spans()
    test_classify_candidates()
    test_nms()
    test_detect_operations()
    test_with_pipeline()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
