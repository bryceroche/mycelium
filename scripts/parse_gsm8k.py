#!/usr/bin/env python3
"""Parse GSM8K's built-in step-by-step solutions into per-cycle targets.

GSM8K solutions already contain <<calc=result>> annotations and natural
language steps. We just parse them into our training format.

Usage:
    python scripts/parse_gsm8k.py
    python scripts/parse_gsm8k.py --max_problems 100  # test run
    python scripts/parse_gsm8k.py --min_steps 2 --max_steps 5  # filter by step count
"""
import argparse
import json
import os
import re
from datasets import load_dataset


def parse_gsm8k_steps(solution_text):
    """
    Parse GSM8K's built-in step-by-step solution into per-cycle targets.

    Input:  "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n
             Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n
             #### 72"

    Output: {
        "cycle_targets": [24, 72],
        "cycle_gen_targets": [
            "Natalia sold 48/2 = 24 clips in May.",
            "Natalia sold 48+24 = 72 clips altogether."
        ],
        "final_answer": 72,
        "num_steps": 2
    }
    """
    lines = solution_text.strip().split('\n')
    cycle_targets = []
    cycle_gen_targets = []
    final_answer = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('####'):
            # Extract final answer
            ans_str = line.replace('####', '').strip()
            # Handle commas in numbers
            ans_str = ans_str.replace(',', '')
            try:
                final_answer = int(float(ans_str))
            except ValueError:
                return None  # skip unparseable
            continue

        # Extract <<calculation=result>> annotations
        calc_matches = re.findall(r'<<[^>]*=([-]?[\d,]+\.?\d*)>>', line)
        if not calc_matches:
            # Line without annotation — might be context, skip
            continue

        # The intermediate result is the last annotation's result
        result_str = calc_matches[-1].replace(',', '')
        try:
            intermediate = int(float(result_str))
        except ValueError:
            continue

        # Clean annotations out for natural gen target
        gen_text = re.sub(r'<<[^>]*>>', '', line).strip()
        # Clean up double spaces
        gen_text = re.sub(r'\s+', ' ', gen_text).strip()

        cycle_targets.append(intermediate)
        cycle_gen_targets.append(gen_text)

    if final_answer is None or len(cycle_targets) == 0:
        return None

    # Validate: last cycle target should match final answer
    # (sometimes GSM8K has extra steps or rounding)
    if cycle_targets[-1] != final_answer:
        # Try to fix by appending final answer as last step
        cycle_targets.append(final_answer)
        cycle_gen_targets.append(f"The answer is {final_answer}.")

    return {
        'cycle_targets': cycle_targets,
        'cycle_gen_targets': cycle_gen_targets,
        'final_answer': final_answer,
        'num_steps': len(cycle_targets),
    }


def main():
    parser = argparse.ArgumentParser(description='Parse GSM8K into per-cycle targets')
    parser.add_argument('--max_problems', type=int, default=None,
                        help='Limit number of problems (for testing)')
    parser.add_argument('--min_steps', type=int, default=2,
                        help='Minimum number of computation steps (default: 2)')
    parser.add_argument('--max_steps', type=int, default=8,
                        help='Maximum number of computation steps (default: 8)')
    parser.add_argument('--output_dir', type=str, default='data/per_cycle',
                        help='Output directory')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'],
                        help='Which split to parse')
    args = parser.parse_args()

    print(f"Loading GSM8K {args.split} split...")
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    print(f"Loaded {len(ds)} problems")

    parsed = []
    skipped = 0
    step_counts = {}

    for i, example in enumerate(ds):
        if args.max_problems and i >= args.max_problems:
            break

        problem = example['question']
        solution = example['answer']

        result = parse_gsm8k_steps(solution)
        if result is None:
            skipped += 1
            continue

        n_steps = result['num_steps']
        if n_steps < args.min_steps or n_steps > args.max_steps:
            skipped += 1
            continue

        result['problem'] = problem
        parsed.append(result)
        step_counts[n_steps] = step_counts.get(n_steps, 0) + 1

    # Split into train/eval (90/10)
    split_idx = int(len(parsed) * 0.9)
    train_data = parsed[:split_idx]
    eval_data = parsed[split_idx:]

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, 'gsm8k_train.jsonl')
    eval_path = os.path.join(args.output_dir, 'gsm8k_eval.jsonl')

    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(eval_path, 'w') as f:
        for item in eval_data:
            f.write(json.dumps(item) + '\n')

    print(f"\nParsed {len(parsed)} problems ({skipped} skipped)")
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
    print(f"\nStep distribution:")
    for k in sorted(step_counts):
        print(f"  {k} steps: {step_counts[k]} problems")

    # Print a few examples
    print(f"\nExamples:")
    for item in parsed[:3]:
        print(f"\n  Problem: {item['problem'][:80]}...")
        print(f"  Steps: {item['num_steps']}")
        for j, (ct, gt) in enumerate(zip(item['cycle_targets'], item['cycle_gen_targets'])):
            print(f"    Cycle {j+1}: {gt[:70]}... → {ct}")
        print(f"  Final: {item['final_answer']}")


if __name__ == '__main__':
    main()
