#!/usr/bin/env python3
"""
Clean IAF targets to extract only operand-like tokens.

Filter out:
- Chat template tokens (<|im_*|>)
- Instruction words (Show, Solve, Express, etc.)
- Common English words

Keep:
- Numbers (integers, decimals)
- Variables (single letters like x, y, n)
- Math operators (+, -, *, /, ^, =)
- Parentheses, fractions
"""

import json
import re
import argparse
from collections import defaultdict

# Words to filter out
FILTER_WORDS = {
    # Chat tokens (will also regex filter)
    'im_start', 'im_end', 'assistant', 'user', 'system',
    # Instructions
    'show', 'solve', 'find', 'express', 'evaluate', 'calculate',
    'compute', 'determine', 'simplify', 'expand', 'prove',
    'work', 'step', 'answer', 'decimal', 'form', 'terms',
    # Common words
    'the', 'a', 'an', 'is', 'are', 'of', 'to', 'in', 'for',
    'and', 'or', 'if', 'that', 'which', 'with', 'as', 'by',
    'your', 'our', 'some', 'each', 'has', 'have', 'had',
    'what', 'how', 'when', 'where', 'why', 'who',
    # Other noise
    'display', 'style', 'log', 'inter', 'end', 'start',
    'equation', 'line', 'parallel', 'point', 'sequence',
    'arithmetic', 'geometric', 'first', 'second', 'third',
    'loses', 'pack', 'cherry', 'grape', 'gum', 'chew',
}

# Pattern for operand-like tokens
OPERAND_PATTERN = re.compile(r'''
    -?\d+\.?\d*         # numbers (int or decimal)
    | [a-z]             # single letter variables
    | [+\-*/^=(){}]     # operators and parens
    | \\?frac           # fractions
    | \\?sqrt           # sqrt
    | _prev             # reference
    | \|                # absolute value
''', re.VERBOSE | re.IGNORECASE)


def clean_target(raw_target: str) -> str:
    """Extract only operand-like tokens from target."""
    # Remove chat template tokens
    text = re.sub(r'<\|[^|]+\|>', '', raw_target)

    # Tokenize
    tokens = text.split()

    # Filter tokens
    clean_tokens = []
    for tok in tokens:
        tok_lower = tok.lower().strip('.,;:(){}[]')

        # Skip filter words
        if tok_lower in FILTER_WORDS:
            continue

        # Skip if too long to be an operand (unless it has digits)
        if len(tok) > 15 and not any(c.isdigit() for c in tok):
            continue

        # Keep if it matches operand pattern
        if OPERAND_PATTERN.search(tok):
            # Clean up the token
            clean_tok = tok.strip('.,;:')
            if clean_tok:
                clean_tokens.append(clean_tok)

    return ' '.join(clean_tokens)


def extract_numbers_and_vars(text: str) -> list:
    """Extract just numbers and variables from text."""
    # Find all numbers
    numbers = re.findall(r'-?\d+\.?\d*', text)

    # Find single-letter variables (excluding common words)
    text_no_nums = re.sub(r'-?\d+\.?\d*', '', text)
    potential_vars = re.findall(r'\b([a-zA-Z])\b', text_no_nums)
    vars = [v for v in potential_vars if v.lower() not in {'a', 'i', 's'}]

    return numbers + vars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/iaf_training.jsonl")
    parser.add_argument("--output", default="data/iaf_training_clean.jsonl")
    parser.add_argument("--min-operands", type=int, default=1)
    args = parser.parse_args()

    clean_examples = []
    stats = defaultdict(int)

    with open(args.input) as f:
        for line in f:
            ex = json.loads(line)
            raw_target = ex['target']

            # Clean the target
            clean_target_text = clean_target(raw_target)

            # Extract operands
            operands = extract_numbers_and_vars(clean_target_text)

            if len(operands) >= args.min_operands:
                # Use extracted operands as target (space-separated)
                operand_target = ' '.join(operands)
                ex['target'] = f" {operand_target}\n"
                ex['raw_target'] = raw_target
                ex['extracted_operands'] = operands
                clean_examples.append(ex)
                stats['kept'] += 1
            else:
                stats['filtered'] += 1

            stats['total'] += 1

    print(f"Total: {stats['total']}")
    print(f"Kept: {stats['kept']} ({100*stats['kept']/stats['total']:.1f}%)")
    print(f"Filtered: {stats['filtered']}")

    # Save
    with open(args.output, 'w') as f:
        for ex in clean_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved to {args.output}")

    # Show samples
    print("\n=== Sample Cleaned Targets ===")
    for ex in clean_examples[::len(clean_examples)//10][:10]:
        print(f"{ex['scaffold_type']:12} | {ex['target'].strip()[:50]}")
        print(f"             | Operands: {ex['extracted_operands']}")


if __name__ == "__main__":
    main()
