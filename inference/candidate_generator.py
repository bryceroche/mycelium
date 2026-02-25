#!/usr/bin/env python3
"""
Mycelium v6: Candidate Grouping Generator

Generates plausible groupings of spans for candidate enumeration.
Each grouping partitions OP spans into groups, where each group
represents one operation.

Pruning heuristics:
- Non-adjacent spans rarely group (skip pairs separated by >2 spans)
- Question spans never group with operation spans
- Maximum group size of 3 (very few operations use 4+ clauses)
- Shares numbers heuristic (optional: spans should share related numbers)

For most GSM8K problems (2-4 spans), generates 2-15 candidates.
"""

import re
from itertools import combinations
from typing import List, Dict, Set, Tuple


def extract_numbers_from_text(text: str) -> Set[float]:
    """Extract all numbers from text."""
    numbers = set()
    for match in re.finditer(r'-?\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', text):
        num_str = match.group(1).replace(",", "")
        try:
            numbers.add(float(num_str))
        except ValueError:
            pass
    return numbers


def shares_numbers(span_a: Dict, span_b: Dict) -> bool:
    """Check if two spans reference related numbers."""
    nums_a = extract_numbers_from_text(span_a.get("text", ""))
    nums_b = extract_numbers_from_text(span_b.get("text", ""))

    # Direct overlap
    if nums_a & nums_b:
        return True

    # Check if one contains a multiplier of the other (e.g., 2 and 4)
    for a in nums_a:
        for b in nums_b:
            if a != 0 and b != 0:
                ratio = max(a, b) / min(a, b)
                if ratio == int(ratio) and 1 < ratio <= 10:
                    return True

    return False


def generate_candidate_groupings(
    spans: List[Dict],
    max_group_size: int = 3,
    max_candidates: int = 20,
    require_adjacent: bool = False,
    use_number_heuristic: bool = False,
) -> List[List[List[int]]]:
    """
    Generate plausible groupings of spans into operations.

    Each grouping is a partition of span indices into groups.

    Args:
        spans: List of span dicts with at least "type" and "text" fields
        max_group_size: Maximum number of spans per group
        max_candidates: Maximum number of candidates to return
        require_adjacent: Only allow adjacent spans to group
        use_number_heuristic: Require grouped spans to share numbers

    Returns:
        List of groupings, where each grouping is a list of groups,
        and each group is a list of span indices.
    """
    # Separate OP and Q spans
    op_indices = [i for i, s in enumerate(spans) if s.get("tag") == "OP"]
    q_indices = [i for i, s in enumerate(spans) if s.get("tag") == "Q"]

    n = len(op_indices)

    if n == 0:
        # No OP spans, return empty grouping
        return [[]]

    candidates = []

    # Candidate 1: All spans are separate operations (baseline)
    baseline = [[idx] for idx in op_indices]
    candidates.append(baseline)

    # For small n, enumerate more combinations
    if n == 2:
        # 2 spans: either separate or grouped
        candidates.append([[op_indices[0], op_indices[1]]])

    elif n == 3:
        # 3 spans: various combinations
        a, b, c = op_indices
        candidates.append([[a, b], [c]])  # first two together
        candidates.append([[a], [b, c]])  # last two together
        candidates.append([[a, c], [b]])  # first and last together (if adjacent rule allows)
        candidates.append([[a, b, c]])    # all three together

    elif n == 4:
        # 4 spans: key combinations
        a, b, c, d = op_indices
        # Adjacent pairs
        candidates.append([[a, b], [c], [d]])
        candidates.append([[a], [b, c], [d]])
        candidates.append([[a], [b], [c, d]])
        candidates.append([[a, b], [c, d]])  # two pairs
        # Triples
        candidates.append([[a, b, c], [d]])
        candidates.append([[a], [b, c, d]])

    else:
        # For larger n, generate systematically but limit
        # Generate adjacent pair merges
        for i in range(n - 1):
            groups = []
            j = 0
            while j < n:
                if j == i:
                    groups.append([op_indices[j], op_indices[j + 1]])
                    j += 2
                else:
                    groups.append([op_indices[j]])
                    j += 1
            candidates.append(groups)

        # Generate adjacent triple merges
        if max_group_size >= 3:
            for i in range(n - 2):
                groups = []
                j = 0
                while j < n:
                    if j == i:
                        groups.append([op_indices[j], op_indices[j + 1], op_indices[j + 2]])
                        j += 3
                    else:
                        groups.append([op_indices[j]])
                        j += 1
                candidates.append(groups)

        # Generate two adjacent pairs (if n >= 4)
        if n >= 4:
            for i in range(n - 3):
                for k in range(i + 2, n - 1):
                    groups = []
                    j = 0
                    while j < n:
                        if j == i:
                            groups.append([op_indices[j], op_indices[j + 1]])
                            j += 2
                        elif j == k:
                            groups.append([op_indices[j], op_indices[j + 1]])
                            j += 2
                        else:
                            groups.append([op_indices[j]])
                            j += 1
                    if len([g for g in groups if len(g) > 0]) > 0:
                        candidates.append(groups)

    # Apply filtering heuristics
    filtered = []
    for grouping in candidates:
        valid = True

        for group in grouping:
            if len(group) <= 1:
                continue

            # Check adjacency constraint
            if require_adjacent:
                sorted_group = sorted(group)
                for k in range(len(sorted_group) - 1):
                    if sorted_group[k + 1] - sorted_group[k] > 1:
                        valid = False
                        break

            # Check number sharing heuristic
            if use_number_heuristic and len(group) > 1:
                group_spans = [spans[idx] for idx in group]
                # At least some pair should share numbers
                any_share = False
                for i in range(len(group_spans)):
                    for j in range(i + 1, len(group_spans)):
                        if shares_numbers(group_spans[i], group_spans[j]):
                            any_share = True
                            break
                    if any_share:
                        break
                if not any_share:
                    valid = False

            if not valid:
                break

        if valid:
            filtered.append(grouping)

    # Deduplicate
    seen = set()
    unique = []
    for grouping in filtered:
        # Sort each group and the grouping for canonical form
        canonical = tuple(tuple(sorted(g)) for g in sorted(grouping, key=lambda x: min(x)))
        if canonical not in seen:
            seen.add(canonical)
            unique.append(grouping)

    # Limit candidates
    unique = unique[:max_candidates]

    return unique


def format_grouping(spans: List[Dict], grouping: List[List[int]]) -> str:
    """Format a grouping for display."""
    parts = []
    for group in grouping:
        group_texts = [spans[idx].get("text", f"span_{idx}")[:30] for idx in group]
        if len(group) == 1:
            parts.append(f"[{group_texts[0]}]")
        else:
            parts.append(f"[{', '.join(group_texts)}]")
    return " + ".join(parts)


def main():
    """Test candidate generation with example spans."""
    print("=" * 60)
    print("CANDIDATE GROUPING GENERATOR")
    print("=" * 60)

    # Example 1: Simple 2-span problem
    spans_2 = [
        {"tag": "OP", "text": "In the first box he counted 72 raisins"},
        {"tag": "OP", "text": "in a second box he counted 74 raisins"},
        {"tag": "Q", "text": "How many raisins total?"},
    ]

    print("\nExample 1: 2 OP spans")
    print("Spans:")
    for i, s in enumerate(spans_2):
        print(f"  [{i}] {s['tag']}: {s['text'][:50]}...")

    candidates = generate_candidate_groupings(spans_2)
    print(f"\nGenerated {len(candidates)} candidates:")
    for i, c in enumerate(candidates):
        print(f"  {i + 1}. {format_grouping(spans_2, c)}")

    # Example 2: 4-span problem
    spans_4 = [
        {"tag": "OP", "text": "sells apples for $2 each"},
        {"tag": "OP", "text": "oranges for $3 each"},
        {"tag": "OP", "text": "bought 5 apples"},
        {"tag": "OP", "text": "4 oranges"},
        {"tag": "OP", "text": "paid with a $50 bill"},
        {"tag": "Q", "text": "How much change?"},
    ]

    print("\n" + "-" * 60)
    print("\nExample 2: 5 OP spans (complex)")
    print("Spans:")
    for i, s in enumerate(spans_4):
        print(f"  [{i}] {s['tag']}: {s['text']}")

    candidates = generate_candidate_groupings(spans_4)
    print(f"\nGenerated {len(candidates)} candidates:")
    for i, c in enumerate(candidates[:10]):  # Show first 10
        print(f"  {i + 1}. {format_grouping(spans_4, c)}")
    if len(candidates) > 10:
        print(f"  ... and {len(candidates) - 10} more")

    # Example 3: Test with number heuristic
    print("\n" + "-" * 60)
    print("\nExample 3: With number-sharing heuristic")

    candidates_filtered = generate_candidate_groupings(
        spans_4,
        use_number_heuristic=True
    )
    print(f"Generated {len(candidates_filtered)} candidates (with number filter):")
    for i, c in enumerate(candidates_filtered[:10]):
        print(f"  {i + 1}. {format_grouping(spans_4, c)}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
