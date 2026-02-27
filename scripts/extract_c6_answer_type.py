#!/usr/bin/env python3
"""
C6 Data Extractor: Answer Type Classification

Extract answer type labels from IAF data:
- Parse gold solution to determine answer format
- Label: INTEGER, FRACTION, DECIMAL, SET, EXPRESSION, etc.

No keyword heuristics - uses structural parsing only.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from fractions import Fraction
import sys


@dataclass
class AnswerTypeLabel:
    problem_idx: int
    problem_text: str
    answer_type: str
    raw_answer: str
    confidence: float


def parse_boxed_answer(solution: str) -> Optional[str]:
    """Extract content from \\boxed{...} - structural parsing only."""
    # Find \boxed{ and match braces
    start = solution.find('\\boxed{')
    if start == -1:
        return None

    start += len('\\boxed{')
    depth = 1
    end = start

    while end < len(solution) and depth > 0:
        if solution[end] == '{':
            depth += 1
        elif solution[end] == '}':
            depth -= 1
        end += 1

    if depth == 0:
        return solution[start:end-1].strip()
    return None


def classify_answer_type(answer: str) -> Tuple[str, float]:
    """
    Classify answer type from structural properties only.

    NO KEYWORDS - only structural checks:
    - Is it an integer? Check if all chars are digits/sign
    - Is it a fraction? Check for / or \frac
    - Is it a decimal? Check for .
    - Is it a set? Check for { } with commas
    - Otherwise: EXPRESSION
    """
    if not answer:
        return "UNKNOWN", 0.0

    answer = answer.strip()

    # Check for set notation: {a, b, c}
    if answer.startswith('{') and answer.endswith('}') and ',' in answer:
        return "SET", 0.95

    # Check for LaTeX fraction: \frac{a}{b}
    if '\\frac{' in answer:
        return "FRACTION", 0.95

    # Check for simple fraction: a/b
    if '/' in answer:
        parts = answer.split('/')
        if len(parts) == 2:
            left = parts[0].strip().lstrip('-')
            right = parts[1].strip().lstrip('-')
            if left.isdigit() and right.isdigit():
                return "FRACTION", 0.9

    # Check for decimal
    if '.' in answer:
        # Remove sign and check structure
        clean = answer.lstrip('-+')
        parts = clean.split('.')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return "DECIMAL", 0.95

    # Check for integer (with possible sign)
    clean = answer.lstrip('-+')
    if clean.isdigit():
        return "INTEGER", 0.98

    # Check for percentage (number followed by %)
    if answer.endswith('%'):
        num_part = answer[:-1].strip()
        clean = num_part.lstrip('-+')
        if clean.replace('.', '').isdigit():
            return "PERCENTAGE", 0.9

    # Check for pi expression
    if '\\pi' in answer or 'π' in answer:
        return "PI_EXPRESSION", 0.85

    # Check for sqrt expression
    if '\\sqrt' in answer:
        return "RADICAL", 0.85

    # Check for variable expression (contains letters)
    has_letter = any(c.isalpha() and c not in ['x', 'y', 'z', 'n', 'm'] for c in answer)
    if has_letter:
        return "EXPRESSION", 0.7

    # Check for simple algebraic (x, y, z with coefficients)
    if any(c in answer for c in 'xyz'):
        return "ALGEBRAIC", 0.8

    return "EXPRESSION", 0.5


def extract_c6_labels(iaf_data: List[Dict]) -> List[AnswerTypeLabel]:
    """Extract C6 answer type labels from IAF data."""
    labels = []

    for record in iaf_data:
        problem_idx = record.get('problem_idx', -1)
        problem_text = record.get('problem_text', '')
        solution = record.get('solution', '')

        # Try to extract boxed answer
        raw_answer = parse_boxed_answer(solution)

        if raw_answer is None:
            # Fallback: look for final = or "answer is"
            # Try to find last numeric value
            raw_answer = ""
            confidence = 0.3
            answer_type = "UNKNOWN"
        else:
            answer_type, confidence = classify_answer_type(raw_answer)

        labels.append(AnswerTypeLabel(
            problem_idx=problem_idx,
            problem_text=problem_text,
            answer_type=answer_type,
            raw_answer=raw_answer or "",
            confidence=confidence
        ))

    return labels


def main():
    parser = argparse.ArgumentParser(description='Extract C6 answer type labels from IAF data')
    parser.add_argument('input', help='Input IAF JSON file or S3 path')
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--stats', action='store_true', help='Print statistics')
    args = parser.parse_args()

    # Load data
    if args.input.startswith('s3://'):
        import subprocess
        result = subprocess.run(['aws', 's3', 'cp', args.input, '-'],
                              capture_output=True, text=True)
        data = json.loads(result.stdout)
    else:
        with open(args.input) as f:
            data = json.load(f)

    print(f"Loaded {len(data)} records", file=sys.stderr)

    # Extract labels
    labels = extract_c6_labels(data)

    # Convert to JSON-serializable format
    output = [
        {
            'problem_idx': l.problem_idx,
            'problem_text': l.problem_text,
            'answer_type': l.answer_type,
            'raw_answer': l.raw_answer,
            'confidence': l.confidence
        }
        for l in labels
    ]

    # Print stats
    if args.stats:
        from collections import Counter
        type_counts = Counter(l.answer_type for l in labels)
        print("\nAnswer type distribution:", file=sys.stderr)
        for t, c in type_counts.most_common():
            print(f"  {t}: {c} ({100*c/len(labels):.1f}%)", file=sys.stderr)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {len(output)} labels to {args.output}", file=sys.stderr)
    else:
        json.dump(output, sys.stdout, indent=2)


if __name__ == '__main__':
    main()
