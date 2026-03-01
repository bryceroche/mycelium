#!/usr/bin/env python3
"""
E2E Test: C3 Beam + Combinatorial Execution + Majority Voting

For each operation, C3 generates k beam candidates.
For N operations, that's k^N paths.
Execute each path through sympy - most crash or produce nonsense.
Paths with correct expressions produce valid numeric answers.
Majority voting on valid results = answer.

This is the MCTS architecture: let candidates play out through
the full pipeline and see which produce coherent solutions.
"""

import json
import re
import torch
import sympy
import boto3
from collections import Counter
from itertools import product
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_numbers_from_text(text: str) -> set:
    """Extract all numbers from problem text."""
    nums = set()
    for match in re.findall(r'-?\d+(?:\.\d+)?', text):
        try:
            nums.add(float(match) if '.' in match else int(match))
        except:
            pass
    return nums


def parse_expression(expr_str: str):
    """Parse expression with sympy. Returns sympy expr or None."""
    try:
        return sympy.sympify(expr_str)
    except:
        return None


def expressions_equal(expr1, expr2) -> bool:
    """Check if two sympy expressions are equal (numeric or symbolic)."""
    try:
        if expr1 is None or expr2 is None:
            return False

        # If both are numbers, compare numerically
        if expr1.is_number and expr2.is_number:
            return abs(float(expr1) - float(expr2)) < 1e-6

        # Otherwise compare symbolically
        diff = sympy.simplify(expr1 - expr2)
        return diff == 0
    except:
        return False


def check_answer(predicted_expr, gold: str) -> bool:
    """Check if predicted expression equals gold."""
    try:
        gold_expr = sympy.sympify(gold)
        return expressions_equal(predicted_expr, gold_expr)
    except:
        return False


def load_c3_model(model_path: str):
    """Load causal LM C3 on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = model.to('cpu')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return tokenizer, model


def generate_beam_candidates(tokenizer, model, problem_text: str, template: str,
                              num_beams: int = 5):
    """Generate beam candidates for one operation."""
    prompt = f"[TEMPLATE: {template}] {problem_text}\nExpression:"

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    candidates = []
    for output in outputs:
        generated = tokenizer.decode(
            output[inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        generated = generated.split('\n')[0].strip()
        if generated:
            candidates.append(generated)

    return candidates


def execute_and_vote(candidates: list) -> tuple:
    """
    Parse all candidates through sympy.
    Group by symbolic equivalence.
    Return (most_common_expr, vote_count, total_valid).
    """
    parsed = []
    for expr_str in candidates:
        expr = parse_expression(expr_str)
        if expr is not None:
            parsed.append((expr_str, expr))

    if not parsed:
        return None, 0, 0

    # Group by equivalence
    groups = []  # List of (representative_expr, count, expr_str)
    for expr_str, expr in parsed:
        found = False
        for i, (rep_expr, count, rep_str) in enumerate(groups):
            if expressions_equal(expr, rep_expr):
                groups[i] = (rep_expr, count + 1, rep_str)
                found = True
                break
        if not found:
            groups.append((expr, 1, expr_str))

    # Find most common group
    groups.sort(key=lambda x: -x[1])
    best_expr, best_count, best_str = groups[0]
    return best_expr, best_count, len(parsed)


def run_mcts_test(c3_path: str, test_problems: list, num_beams: int = 5):
    """Run MCTS-style test with combinatorial execution + majority voting."""
    print(f"Loading C3 model from {c3_path}...")
    tokenizer, model = load_c3_model(c3_path)
    print("Using device: CPU")

    print(f"\nTesting on {len(test_problems)} problems with beam_k={num_beams}...")
    print("Strategy: Execute all candidates, majority vote on valid results\n")

    results = {
        'total': len(test_problems),
        'has_valid': 0,
        'correct_majority': 0,
        'correct_in_beam': 0,
    }

    for i, problem in enumerate(test_problems):
        problem_text = problem['problem_text']
        template = problem['template']
        gold_expr = problem['expression']

        # Generate beam candidates
        candidates = generate_beam_candidates(
            tokenizer, model, problem_text, template, num_beams
        )

        # Execute all and majority vote
        majority_expr, vote_count, total_valid = execute_and_vote(candidates)

        # Check if any candidate produces correct result
        correct_in_beam = False
        for expr_str in candidates:
            expr = parse_expression(expr_str)
            if expr is not None and check_answer(expr, gold_expr):
                correct_in_beam = True
                break

        if correct_in_beam:
            results['correct_in_beam'] += 1

        if total_valid > 0:
            results['has_valid'] += 1

        # Check majority vote result
        majority_correct = check_answer(majority_expr, gold_expr)
        if majority_correct:
            results['correct_majority'] += 1

        # Print progress
        status = '✓' if majority_correct else '✗'
        beam_status = '(in beam)' if correct_in_beam else ''
        majority_str = str(majority_expr)[:20] if majority_expr else 'NONE'
        print(f"[{i+1:3d}] {template:6s} | gold: {gold_expr:20s} | "
              f"majority: {majority_str:20s} (votes: {vote_count}/{total_valid}) {status} {beam_status}")

        if (i + 1) % 20 == 0:
            print(f"\n--- Progress: {results['correct_majority']}/{i+1} majority correct ---\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: MCTS-Style Combinatorial Execution + Majority Voting")
    print("=" * 70)
    print(f"Total problems:        {results['total']}")
    print(f"Has valid result:      {results['has_valid']} ({100*results['has_valid']/results['total']:.1f}%)")
    print(f"Correct in beam:       {results['correct_in_beam']} ({100*results['correct_in_beam']/results['total']:.1f}%)")
    print(f"Correct by majority:   {results['correct_majority']} ({100*results['correct_majority']/results['total']:.1f}%)")
    print()
    print("Note: 'Correct in beam' is the ceiling - perfect ranking would achieve this.")
    print("      'Correct by majority' is what we get with majority voting.")

    return results


def download_c3_model():
    """Download C3 model from S3."""
    import os
    c3_dir = '/tmp/c3_causal_lm'
    os.makedirs(c3_dir, exist_ok=True)

    s3 = boto3.client('s3')
    files = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
             'generation_config.json']

    for f in files:
        local_path = f'{c3_dir}/{f}'
        if not os.path.exists(local_path):
            print(f"  Downloading {f}...")
            try:
                s3.download_file('mycelium-data', f'models/c3_extractor_epoch1/{f}', local_path)
            except Exception as e:
                print(f"  Warning: {f} not found: {e}")

    return c3_dir


def main():
    print("=" * 70)
    print("MCTS-Style E2E Test: Combinatorial Execution + Majority Voting")
    print("=" * 70)

    # Download C3 model
    print("\nDownloading C3 causal LM (epoch 1 checkpoint)...")
    c3_path = download_c3_model()

    # Load test problems
    print("\nLoading test problems...")
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='mycelium-data', Key='c3_span_training/track_a_simple.jsonl')
    content = response['Body'].read().decode('utf-8')
    examples = [json.loads(line) for line in content.strip().split('\n') if line.strip()]

    # Take first 50 for testing
    test_problems = examples[:50]

    run_mcts_test(c3_path, test_problems, num_beams=5)


if __name__ == '__main__':
    main()
