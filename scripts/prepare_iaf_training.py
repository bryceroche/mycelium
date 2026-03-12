#!/usr/bin/env python3
"""
Prepare IAF operand data for LoRA C training.

Cleans up tokenization artifacts and formats for training.
"""

import json
import os
import re
import argparse
from pathlib import Path


def clean_operand_text(text: str) -> str:
    """Clean up tokenization and LaTeX artifacts."""
    # Remove common LaTeX commands
    text = re.sub(r'\\(left|right|frac|sqrt|text|textbf|mathrm|mathbf)', ' ', text)
    text = re.sub(r'\\\\', ' ', text)  # double backslash

    # Remove formatting characters but keep math operators
    text = re.sub(r'[{}$&]', ' ', text)

    # Clean up caret notation - ^{2} -> ^2
    text = re.sub(r'\^\{?\s*(\d+)\s*\}?', r'^\1', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    # Remove leading/trailing punctuation
    text = text.strip(' .,;:')

    return text


def process_example(ex: dict) -> dict:
    """Process a single training example."""
    target = ex.get('target', '')
    clean_target = clean_operand_text(target)

    # Also clean the operand details
    clean_details = []
    for detail in ex.get('operand_details', []):
        clean_details.append({
            'text': clean_operand_text(detail.get('text', '')),
            'weight': detail.get('weight', 0),
        })

    return {
        'problem_id': ex.get('problem_id'),
        'step_idx': ex.get('step_idx'),
        'scaffold_type': ex.get('scaffold_type'),
        'prompt': ex.get('prompt'),
        'target': f" {clean_target}\n",
        'raw_target': target,
        'operand_details': clean_details,
        'source': 'iaf_attention',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/tmp/iaf_operands", help="Input directory")
    parser.add_argument("--output", default="data/iaf_training.jsonl", help="Output file")
    parser.add_argument("--min-operands", type=int, default=1, help="Min operands per example")
    parser.add_argument("--max-target-len", type=int, default=100, help="Max target length")
    args = parser.parse_args()

    all_examples = []

    for f in os.listdir(args.input_dir):
        if not f.endswith('.json'):
            continue

        filepath = os.path.join(args.input_dir, f)
        with open(filepath) as fp:
            for line in fp:
                if line.strip():
                    try:
                        ex = json.loads(line)
                        processed = process_example(ex)

                        # Filter
                        target = processed['target'].strip()
                        if len(target) < 3:
                            continue
                        if len(target) > args.max_target_len:
                            continue

                        all_examples.append(processed)
                    except Exception as e:
                        pass

    print(f"Loaded {len(all_examples)} examples")

    # Deduplicate by (problem_id, step_idx)
    seen = set()
    unique = []
    for ex in all_examples:
        key = (ex['problem_id'], ex['step_idx'])
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    print(f"After dedup: {len(unique)} examples")

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        for ex in unique:
            f.write(json.dumps(ex) + '\n')

    print(f"Saved to {args.output}")

    # Stats
    from collections import defaultdict
    by_type = defaultdict(int)
    for ex in unique:
        by_type[ex['scaffold_type']] += 1

    print(f"\nBy scaffold type:")
    for st, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {st}: {count}")

    # Show examples
    print(f"\n=== Sample Cleaned Examples ===")
    for i, ex in enumerate(unique[:5]):
        print(f"\n[{i+1}] {ex['scaffold_type']} step {ex['step_idx']}")
        print(f"    Raw:   '{ex['raw_target'].strip()}'")
        print(f"    Clean: '{ex['target'].strip()}'")


if __name__ == "__main__":
    main()
