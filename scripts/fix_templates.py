#!/usr/bin/env python3
"""
Fix contaminated templates by re-classifying based on actual pattern semantics.

The original templates were classified by verb presence (e.g., "ate" anywhere → SUB),
but this led to contamination where patterns like "creates a movie" got labeled as SUB.

This script:
1. Analyzes each pattern's actual semantic meaning
2. Re-classifies templates based on pattern content
3. Removes/merges templates with invalid patterns
4. Outputs a cleaned template library
"""

import json
import re
from collections import defaultdict
from typing import List, Tuple, Optional

# Semantic indicators for each operation type
OPERATION_INDICATORS = {
    'SET': {
        'strong': ['has', 'have', 'had', 'costs', 'cost', 'is', 'are', 'was', 'were',
                   'contains', 'holds', 'needs', 'requires', 'wants', 'earns', 'makes',
                   'weighs', 'measures', 'takes', 'lasts'],
        'phrases': ['there are', 'there is', 'per hour', 'per day', 'per week',
                    'a day', 'an hour', 'each day', 'each hour', 'starting with',
                    'began with', 'initially']
    },
    'ADD': {
        'strong': ['received', 'found', 'bought', 'purchased', 'got', 'earned', 'won',
                   'collected', 'picked', 'gained', 'added', 'increased', 'grew'],
        'phrases': ['more than', 'additional', 'extra', 'another', 'plus', 'in addition']
    },
    'SUB': {
        'strong': ['gave', 'sold', 'spent', 'paid', 'lost', 'ate', 'drank', 'used up',
                   'removed', 'took away', 'donated', 'traded', 'thrown away', 'discarded'],
        'phrases': ['less than', 'fewer than', 'minus', 'reduced by', 'decreased by',
                    'gave away', 'took from']
    },
    'MUL': {
        'strong': ['times', 'multiplied', 'doubled', 'tripled'],
        'phrases': ['times as', 'twice as', 'three times', 'for each', 'per each',
                    'at rate of', 'per unit', 'each at', 'times that']
    },
    'DIV': {
        'strong': ['divided', 'split', 'shared', 'halved'],
        'phrases': ['divided by', 'split among', 'shared equally', 'half of',
                    'one third', 'one quarter', 'fraction of', 'portion of']
    }
}


def classify_pattern(pattern: str) -> Tuple[Optional[str], float]:
    """Classify a single pattern by its semantic content.

    Returns: (operation_type, confidence)
    """
    pattern_lower = pattern.lower()

    scores = defaultdict(float)

    for op, indicators in OPERATION_INDICATORS.items():
        # Check strong indicators (single words)
        for word in indicators['strong']:
            if re.search(rf'\b{word}\b', pattern_lower):
                scores[op] += 2.0

        # Check phrases
        for phrase in indicators['phrases']:
            if phrase in pattern_lower:
                scores[op] += 3.0  # Phrases are stronger signals

    if not scores:
        return None, 0.0

    best_op = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[best_op] / total if total > 0 else 0.0

    return best_op, confidence


def analyze_template(template: dict) -> dict:
    """Analyze a template and return classification info."""
    patterns = template.get('pattern_examples', [])
    original_op = template['operation']

    # Classify each pattern
    pattern_ops = []
    for p in patterns:
        op, conf = classify_pattern(p)
        pattern_ops.append((p, op, conf))

    # Count votes
    op_counts = defaultdict(int)
    for _, op, conf in pattern_ops:
        if op and conf > 0.3:
            op_counts[op] += 1

    # Determine if template is consistent
    if not op_counts:
        return {
            'template_id': template['template_id'],
            'original_op': original_op,
            'suggested_op': None,
            'status': 'UNCLEAR',
            'patterns_analyzed': pattern_ops
        }

    suggested_op = max(op_counts, key=op_counts.get)
    is_consistent = suggested_op == original_op

    return {
        'template_id': template['template_id'],
        'original_op': original_op,
        'suggested_op': suggested_op,
        'status': 'OK' if is_consistent else 'MISMATCH',
        'patterns_analyzed': pattern_ops,
        'op_votes': dict(op_counts)
    }


def fix_templates(input_path: str, output_path: str):
    """Fix templates and output cleaned version."""
    with open(input_path) as f:
        templates = json.load(f)

    print(f"Analyzing {len(templates)} templates...")

    fixed = []
    mismatches = []
    unclear = []

    for t in templates:
        analysis = analyze_template(t)

        if analysis['status'] == 'OK':
            # Keep as-is
            fixed.append(t)
        elif analysis['status'] == 'MISMATCH':
            mismatches.append(analysis)
            # Fix the template
            fixed_t = t.copy()
            fixed_t['operation'] = analysis['suggested_op']
            fixed_t['original_operation'] = analysis['original_op']
            fixed_t['auto_fixed'] = True

            # Update DSL based on new operation
            new_dsl = get_default_dsl(analysis['suggested_op'], t.get('base_dsl', ''))
            fixed_t['base_dsl'] = new_dsl

            fixed.append(fixed_t)
        else:
            unclear.append(analysis)
            # Keep but mark as unclear
            fixed_t = t.copy()
            fixed_t['classification_unclear'] = True
            fixed.append(fixed_t)

    print(f"\nResults:")
    print(f"  OK: {len(templates) - len(mismatches) - len(unclear)}")
    print(f"  Fixed mismatches: {len(mismatches)}")
    print(f"  Unclear: {len(unclear)}")

    if mismatches:
        print(f"\nMismatched templates that were fixed:")
        for m in mismatches[:20]:
            print(f"  {m['template_id']}: {m['original_op']} -> {m['suggested_op']}")
            print(f"    Votes: {m['op_votes']}")

    # Save fixed templates
    with open(output_path, 'w') as f:
        json.dump(fixed, f, indent=2)

    print(f"\nSaved {len(fixed)} fixed templates to {output_path}")

    return fixed, mismatches, unclear


def get_default_dsl(operation: str, current_dsl: str) -> str:
    """Get default DSL for an operation."""
    defaults = {
        'SET': 'value',
        'ADD': 'entity + value',
        'SUB': 'entity - value',
        'MUL': 'entity * value',
        'DIV': 'entity / value'
    }

    # Keep special DSLs like "ref * 2" or "entity / 2"
    if '* 2' in current_dsl or '/ 2' in current_dsl:
        return current_dsl
    if 'ref' in current_dsl:
        # Map ref DSL to new operation
        if operation == 'ADD':
            return 'ref + value'
        elif operation == 'SUB':
            return 'ref - value'
        elif operation == 'MUL':
            return 'ref * value'

    return defaults.get(operation, current_dsl)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='deduplicated_templates.json')
    parser.add_argument('--output', default='fixed_templates.json')
    args = parser.parse_args()

    fix_templates(args.input, args.output)
