#!/usr/bin/env python3
"""
E2E Test: Causal LM C3 with Beam Search + Numerical Constraints

Uses the epoch 1 checkpoint that showed 98% valid sympy.
Adds beam search (k=5) and filters candidates by:
1. Sympy validity
2. Numerical constraint (prefer expressions containing numbers from the problem)
"""

import json
import re
import torch
import sympy
import boto3
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_numbers_from_text(text: str) -> set:
    """Extract all numbers from problem text."""
    # Match integers, decimals, fractions
    nums = set()
    for match in re.findall(r'-?\d+(?:\.\d+)?', text):
        try:
            nums.add(float(match) if '.' in match else int(match))
        except:
            pass
    return nums


def count_matching_numbers(expression: str, problem_numbers: set) -> int:
    """Count how many numbers in the expression appear in the problem."""
    expr_nums = extract_numbers_from_text(expression)
    return len(expr_nums & problem_numbers)


def check_sympy_valid(expression: str) -> bool:
    """Check if expression is valid sympy."""
    try:
        sympy.sympify(expression)
        return True
    except:
        return False


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted expression equals gold using symbolic equivalence."""
    try:
        pred = sympy.sympify(predicted)
        gold_expr = sympy.sympify(gold)

        if pred.is_number and gold_expr.is_number:
            return abs(float(pred) - float(gold_expr)) < 1e-6

        return sympy.simplify(pred - gold_expr) == 0
    except:
        return False


def load_c3_model(model_path: str, use_cpu: bool = True):
    """Load causal LM C3."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Use CPU to avoid CUBLAS errors
    device = 'cpu' if use_cpu else 'cuda'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=device if not use_cpu else None,
    )

    if use_cpu:
        model = model.to('cpu')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return tokenizer, model


def generate_with_beam_search(tokenizer, model, problem_text: str, template: str,
                               num_beams: int = 5, num_return: int = 5):
    """Generate multiple candidate expressions using beam search."""
    prompt = f"[TEMPLATE: {template}] {problem_text}\nExpression:"

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=num_beams,
            num_return_sequences=num_return,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode all candidates
    candidates = []
    for output in outputs:
        # Get only the generated part (after the prompt)
        generated = tokenizer.decode(
            output[inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Clean up - take first line, strip whitespace
        generated = generated.split('\n')[0].strip()
        if generated:
            candidates.append(generated)

    return candidates


def filter_candidates(candidates: list, problem_text: str) -> list:
    """Filter and rank candidates by validity and numerical constraint."""
    problem_numbers = extract_numbers_from_text(problem_text)

    scored = []
    for candidate in candidates:
        # Check sympy validity
        if not check_sympy_valid(candidate):
            continue

        # Score by number matching
        num_matches = count_matching_numbers(candidate, problem_numbers)
        scored.append((candidate, num_matches))

    # Sort by number of matches (descending)
    scored.sort(key=lambda x: -x[1])

    return [c[0] for c in scored]


def run_e2e_test(c3_path: str, test_problems: list, num_beams: int = 5):
    """Run E2E test with beam search C3."""
    print(f"Loading C3 model from {c3_path}...")
    tokenizer, model = load_c3_model(c3_path, use_cpu=True)
    print("Using device: CPU (avoiding CUBLAS errors)")

    print(f"\nTesting on {len(test_problems)} problems with beam_k={num_beams}...\n")

    results = {
        'total': len(test_problems),
        'valid_sympy': 0,
        'has_valid_candidate': 0,
        'correct_in_beam': 0,  # Gold answer appears in beam candidates
        'correct_top1': 0,     # Best candidate matches gold
    }

    for i, problem in enumerate(test_problems):
        problem_text = problem['problem_text']
        template = problem['template']
        gold_expr = problem['expression']

        # Generate candidates with beam search
        candidates = generate_with_beam_search(
            tokenizer, model, problem_text, template,
            num_beams=num_beams, num_return=num_beams
        )

        # Filter by validity and numerical constraint
        valid_candidates = filter_candidates(candidates, problem_text)

        # Check metrics
        has_valid = len(valid_candidates) > 0
        if has_valid:
            results['has_valid_candidate'] += 1

        # Check if any candidate is correct
        correct_in_beam = any(check_answer(c, gold_expr) for c in valid_candidates)
        if correct_in_beam:
            results['correct_in_beam'] += 1

        # Check if top candidate is correct
        if valid_candidates and check_answer(valid_candidates[0], gold_expr):
            results['correct_top1'] += 1

        # Count valid sympy (any candidate)
        valid_count = sum(1 for c in candidates if check_sympy_valid(c))
        if valid_count > 0:
            results['valid_sympy'] += 1

        # Print progress
        status = '✓' if correct_in_beam else '✗'
        top1_status = '✓' if (valid_candidates and check_answer(valid_candidates[0], gold_expr)) else ''
        print(f"[{i+1:3d}] {template:6s} | gold: {gold_expr:20s} | top: {valid_candidates[0] if valid_candidates else 'NONE':20s} {status} {top1_status}")

        if (i + 1) % 20 == 0:
            print(f"\n--- Progress: {results['correct_in_beam']}/{i+1} correct in beam ---\n")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total problems:        {results['total']}")
    print(f"Has valid candidate:   {results['has_valid_candidate']} ({100*results['has_valid_candidate']/results['total']:.1f}%)")
    print(f"Valid sympy (any):     {results['valid_sympy']} ({100*results['valid_sympy']/results['total']:.1f}%)")
    print(f"Correct in beam:       {results['correct_in_beam']} ({100*results['correct_in_beam']/results['total']:.1f}%)")
    print(f"Correct top-1:         {results['correct_top1']} ({100*results['correct_top1']/results['total']:.1f}%)")

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
    # Download C3 model
    print("Downloading C3 causal LM (epoch 1 checkpoint)...")
    c3_path = download_c3_model()

    # Load test problems from Track A simple (clean binary ops)
    print("\nLoading test problems...")
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='mycelium-data', Key='c3_span_training/track_a_simple.jsonl')
    content = response['Body'].read().decode('utf-8')
    examples = [json.loads(line) for line in content.strip().split('\n') if line.strip()]

    # Take first 50 for testing
    test_problems = examples[:50]

    run_e2e_test(c3_path, test_problems, num_beams=5)


if __name__ == '__main__':
    main()
