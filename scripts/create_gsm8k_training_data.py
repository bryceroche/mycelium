#!/usr/bin/env python3
"""Create GTS training data from GSM8K dataset.

Converts GSM8K problems into (normalized_question, prefix_expression) pairs
for training a GTS model that can decompose math word problems.

Usage:
    uv run python scripts/create_gsm8k_training_data.py
"""

import sys
sys.path.insert(0, "src")

import json
import re
from pathlib import Path
from datasets import load_dataset
from typing import Optional


def normalize_number(n: str) -> str:
    """Normalize number string (remove trailing dots, standardize format)."""
    n = n.strip().rstrip('.')
    try:
        f = float(n)
        if f == int(f):
            return str(int(f))
        return str(f)
    except:
        return n


def tokenize_expr(expr: str) -> list[str]:
    """Tokenize infix expression into operators and operands."""
    expr = expr.strip()

    # Handle leading + (like +30+46...)
    if expr.startswith('+'):
        expr = '0' + expr

    tokens = []
    i = 0
    while i < len(expr):
        if expr[i] in '+-':
            # Check if unary (at start or after operator/open paren)
            if i == 0 or (tokens and tokens[-1] in '(+-*/'):
                j = i + 1
                while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                    j += 1
                if j > i + 1:  # Found digits after sign
                    tokens.append(normalize_number(expr[i:j]))
                    i = j
                    continue
            tokens.append(expr[i])
            i += 1
        elif expr[i] in '*/()':
            tokens.append(expr[i])
            i += 1
        elif expr[i] == 'x':  # Multiplication as 'x'
            tokens.append('*')
            i += 1
        elif expr[i].isdigit() or expr[i] == '.':
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(normalize_number(expr[i:j]))
            i = j
        elif expr[i].isspace():
            i += 1
        else:
            i += 1

    return tokens


def tokens_to_prefix(tokens: list[str]) -> Optional[list[str]]:
    """Convert infix tokens to prefix notation."""
    if not tokens:
        return None

    pos = [0]

    def parse_expr():
        result = parse_term()
        while pos[0] < len(tokens) and tokens[pos[0]] in '+-':
            op = tokens[pos[0]]
            pos[0] += 1
            right = parse_term()
            result = [op] + result + right
        return result

    def parse_term():
        result = parse_factor()
        while pos[0] < len(tokens) and tokens[pos[0]] in '*/':
            op = tokens[pos[0]]
            pos[0] += 1
            right = parse_factor()
            result = [op] + result + right
        return result

    def parse_factor():
        if pos[0] >= len(tokens):
            raise ValueError("Unexpected end of expression")
        tok = tokens[pos[0]]
        if tok == '(':
            pos[0] += 1
            result = parse_expr()
            if pos[0] < len(tokens) and tokens[pos[0]] == ')':
                pos[0] += 1
            return result
        else:
            pos[0] += 1
            return [tok]

    try:
        result = parse_expr()
        if pos[0] < len(tokens):
            return None  # Unconsumed tokens
        return result
    except:
        return None


def convert_gsm8k_problem(question: str, answer: str) -> Optional[dict]:
    """Convert a GSM8K problem to GTS training format.

    Returns:
        {
            'question': original question,
            'normalized_question': question with numbers replaced by NUM_X,
            'prefix': final prefix expression,
            'steps': list of step prefixes,
            'num_map': mapping of NUM_X to values,
            'answer': final answer
        }
    """
    # Clean question (remove $, commas)
    q_clean = question.replace('$', '').replace(',', '')

    # Extract numbers from question
    q_nums = [normalize_number(n) for n in re.findall(r'\d+\.?\d*', q_clean)]
    q_nums = list(dict.fromkeys(q_nums))  # Dedupe, preserve order

    # Build number map
    num_to_token = {n: f'NUM_{i}' for i, n in enumerate(q_nums)}

    # Extract equations from answer
    equations = re.findall(r'<<([^>]+)>>', answer)
    if not equations:
        return None

    # Extract final answer
    final_match = re.search(r'####\s*(-?\d+\.?\d*)', answer)
    final_answer = normalize_number(final_match.group(1)) if final_match else None

    # Process equations
    steps = []
    result_to_step = {}
    const_counter = [0]

    for i, eq in enumerate(equations):
        parts = eq.split('=')
        if len(parts) != 2:
            continue

        expr = parts[0].strip()
        result = normalize_number(parts[1].strip())

        tokens = tokenize_expr(expr)
        if not tokens:
            continue

        # Replace tokens with references
        new_tokens = []
        for tok in tokens:
            if tok in '+-*/()':
                new_tokens.append(tok)
            elif tok in result_to_step:
                new_tokens.append(result_to_step[tok])
            elif tok in num_to_token:
                new_tokens.append(num_to_token[tok])
            else:
                # Constant not in question
                const_tok = f'CONST_{const_counter[0]}'
                const_counter[0] += 1
                num_to_token[tok] = const_tok
                new_tokens.append(const_tok)

        prefix = tokens_to_prefix(new_tokens)
        if prefix is None:
            continue

        step_tok = f'step_{i+1}'
        steps.append({
            'eq': eq,
            'prefix': ' '.join(prefix),
            'result': result,
        })

        result_to_step[result] = step_tok

    if not steps:
        return None

    # Create normalized question (replace numbers with tokens)
    norm_q = q_clean
    for num, tok in sorted(num_to_token.items(), key=lambda x: -len(x[0])):
        # Replace whole numbers only (word boundaries)
        norm_q = re.sub(rf'\b{re.escape(num)}\b', tok, norm_q)

    return {
        'question': question,
        'normalized_question': norm_q,
        'prefix_steps': [s['prefix'] for s in steps],
        'final_prefix': steps[-1]['prefix'],
        'num_map': {v: k for k, v in num_to_token.items()},  # Reverse: token -> value
        'answer': final_answer,
        'num_steps': len(steps),
    }


def main():
    print("Loading GSM8K dataset...")
    train = load_dataset('gsm8k', 'main', split='train')
    test = load_dataset('gsm8k', 'main', split='test')

    print(f"Train: {len(train)}, Test: {len(test)}")

    # Convert datasets
    train_data = []
    test_data = []

    train_success = 0
    for item in train:
        result = convert_gsm8k_problem(item['question'], item['answer'])
        if result and all(result['prefix_steps']):
            train_data.append(result)
            train_success += 1

    test_success = 0
    for item in test:
        result = convert_gsm8k_problem(item['question'], item['answer'])
        if result and all(result['prefix_steps']):
            test_data.append(result)
            test_success += 1

    print(f"\nConversion results:")
    print(f"  Train: {train_success}/{len(train)} ({100*train_success/len(train):.1f}%)")
    print(f"  Test: {test_success}/{len(test)} ({100*test_success/len(test):.1f}%)")

    # Analyze step distribution
    from collections import Counter
    step_counts = Counter(d['num_steps'] for d in train_data)
    print(f"\nStep distribution (train):")
    for steps, count in sorted(step_counts.items()):
        print(f"  {steps} steps: {count} ({100*count/len(train_data):.1f}%)")

    # Save datasets
    output_dir = Path("data/gsm8k_gts")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(output_dir / "test.jsonl", 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

    print(f"\nSaved to {output_dir}/")
    print(f"  train.jsonl: {len(train_data)} examples")
    print(f"  test.jsonl: {len(test_data)} examples")

    # Show examples
    print("\n" + "=" * 70)
    print("Example conversions:")
    print("=" * 70)
    for i in range(3):
        d = train_data[i]
        print(f"\n[{i+1}] {d['question'][:70]}...")
        print(f"    Normalized: {d['normalized_question'][:70]}...")
        print(f"    Steps: {d['prefix_steps']}")
        print(f"    Final prefix: {d['final_prefix']}")
        print(f"    Answer: {d['answer']}")


if __name__ == "__main__":
    main()
