#!/usr/bin/env python3
"""
E2E Test: C3 Beam + Weighted Voting + Numerical Consistency

Improvements over naive majority voting:
1. Weight candidates by C3 generation log-probability
2. Boost candidates containing numbers from the problem text
3. Penalize overly complex expressions
"""

import json
import re
import torch
import sympy
import boto3
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_numbers_from_text(text: str) -> set:
    """Extract all numbers from problem text."""
    nums = set()
    for match in re.findall(r'-?\d+(?:\.\d+)?', text):
        try:
            val = float(match) if '.' in match else int(match)
            nums.add(val)
            nums.add(abs(val))  # Also add absolute value
        except:
            pass
    return nums


def count_matching_numbers(expr_str: str, problem_numbers: set) -> int:
    """Count how many numbers in the expression appear in the problem."""
    expr_nums = set()
    for match in re.findall(r'-?\d+(?:\.\d+)?', expr_str):
        try:
            val = float(match) if '.' in match else int(match)
            expr_nums.add(abs(val))
        except:
            pass
    return len(expr_nums & problem_numbers)


def expression_complexity(expr_str: str) -> int:
    """Measure expression complexity (lower is simpler)."""
    # Count operators and depth indicators
    complexity = 0
    complexity += expr_str.count('(') * 2
    complexity += expr_str.count('*')
    complexity += expr_str.count('/')
    complexity += expr_str.count('+')
    complexity += expr_str.count('-')
    complexity += len(expr_str) // 10  # Length penalty
    return complexity


def parse_expression(expr_str: str):
    """Parse expression with sympy."""
    try:
        return sympy.sympify(expr_str)
    except:
        return None


def expressions_equal(expr1, expr2) -> bool:
    """Check if two sympy expressions are equal."""
    try:
        if expr1 is None or expr2 is None:
            return False
        if expr1.is_number and expr2.is_number:
            return abs(float(expr1) - float(expr2)) < 1e-6
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


def generate_beam_with_scores(tokenizer, model, problem_text: str, template: str,
                               num_beams: int = 10):
    """Generate beam candidates WITH their log-probabilities."""
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
            output_scores=True,
            return_dict_in_generate=True,
        )

    candidates = []
    sequences = outputs.sequences

    for i, seq in enumerate(sequences):
        generated = tokenizer.decode(
            seq[inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        generated = generated.split('\n')[0].strip()

        if generated:
            # Approximate score from beam rank (beam search returns in order)
            # Higher rank = lower probability
            score = 1.0 / (i + 1)  # Simple rank-based score
            candidates.append((generated, score))

    return candidates


def weighted_vote(candidates: list, problem_numbers: set) -> tuple:
    """
    Weighted voting with:
    1. C3 beam score (generation probability proxy)
    2. Numerical consistency boost
    3. Complexity penalty
    """
    if not candidates:
        return None, 0, 0

    # Parse and score each candidate
    scored = []
    for expr_str, beam_score in candidates:
        expr = parse_expression(expr_str)
        if expr is None:
            continue

        # Base score from beam rank
        score = beam_score

        # Boost for matching problem numbers
        num_matches = count_matching_numbers(expr_str, problem_numbers)
        score *= (1 + 0.5 * num_matches)  # 50% boost per matching number

        # Penalty for complexity
        complexity = expression_complexity(expr_str)
        score *= (1.0 / (1 + 0.1 * complexity))

        scored.append((expr, expr_str, score))

    if not scored:
        return None, 0, 0

    # Group by equivalence and sum scores
    groups = []  # List of (representative_expr, total_score, best_str)
    for expr, expr_str, score in scored:
        found = False
        for i, (rep_expr, total_score, rep_str) in enumerate(groups):
            if expressions_equal(expr, rep_expr):
                groups[i] = (rep_expr, total_score + score, rep_str)
                found = True
                break
        if not found:
            groups.append((expr, score, expr_str))

    # Sort by total score
    groups.sort(key=lambda x: -x[1])
    best_expr, best_score, best_str = groups[0]

    return best_expr, best_score, len(scored)


def run_weighted_test(c3_path: str, test_problems: list, num_beams: int = 10):
    """Run test with weighted voting."""
    print(f"Loading C3 model from {c3_path}...")
    tokenizer, model = load_c3_model(c3_path)
    print("Using device: CPU")

    print(f"\nTesting on {len(test_problems)} problems with beam_k={num_beams}...")
    print("Strategy: Weighted voting (beam rank + numerical consistency + simplicity)\n")

    results = {
        'total': len(test_problems),
        'has_valid': 0,
        'correct_weighted': 0,
        'correct_in_beam': 0,
    }

    for i, problem in enumerate(test_problems):
        problem_text = problem['problem_text']
        template = problem['template']
        gold_expr = problem['expression']

        # Extract numbers from problem
        problem_numbers = extract_numbers_from_text(problem_text)

        # Generate beam candidates with scores
        candidates = generate_beam_with_scores(
            tokenizer, model, problem_text, template, num_beams
        )

        # Weighted vote
        best_expr, best_score, total_valid = weighted_vote(candidates, problem_numbers)

        # Check if any candidate is correct
        correct_in_beam = False
        for expr_str, _ in candidates:
            expr = parse_expression(expr_str)
            if expr is not None and check_answer(expr, gold_expr):
                correct_in_beam = True
                break

        if correct_in_beam:
            results['correct_in_beam'] += 1

        if total_valid > 0:
            results['has_valid'] += 1

        # Check weighted vote result
        weighted_correct = check_answer(best_expr, gold_expr)
        if weighted_correct:
            results['correct_weighted'] += 1

        # Print progress
        status = '✓' if weighted_correct else '✗'
        beam_status = '(in beam)' if correct_in_beam else ''
        best_str = str(best_expr)[:20] if best_expr else 'NONE'
        print(f"[{i+1:3d}] {template:6s} | gold: {gold_expr:20s} | "
              f"best: {best_str:20s} (score: {best_score:.2f}) {status} {beam_status}")

        if (i + 1) % 20 == 0:
            print(f"\n--- Progress: {results['correct_weighted']}/{i+1} weighted correct ---\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Weighted Voting (beam rank + numbers + simplicity)")
    print("=" * 70)
    print(f"Total problems:        {results['total']}")
    print(f"Has valid result:      {results['has_valid']} ({100*results['has_valid']/results['total']:.1f}%)")
    print(f"Correct in beam:       {results['correct_in_beam']} ({100*results['correct_in_beam']/results['total']:.1f}%)")
    print(f"Correct weighted:      {results['correct_weighted']} ({100*results['correct_weighted']/results['total']:.1f}%)")
    print()
    print(f"Beam width: {num_beams}")
    print("Baseline to beat: 17.2%")

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
    print("Weighted Voting E2E Test")
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

    run_weighted_test(c3_path, test_problems, num_beams=10)


if __name__ == '__main__':
    main()
