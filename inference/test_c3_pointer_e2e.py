#!/usr/bin/env python3
"""
E2E Test: C3 Pointer + C4 Assembler Pipeline with Beam Search

Tests the new architecture where:
- C3 pointer SELECTS operands from a closed set (can't hallucinate)
- C4 assembler deterministically builds expressions
- Sympy evaluates and filters

Key: beam search over pointer candidates + sympy filtering
"""

import json
import re
import torch
import sympy
import boto3
from pathlib import Path
from itertools import product
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer
from models.c3_pointer import C3PointerModel, extract_text_operands, IMPLICIT_VALUES, TEMPLATE_ARITY, DOMAIN_CONSTANTS, MAX_OPERANDS
from inference.assembler import assemble_and_validate


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


def load_c3_pointer(model_path: str, tokenizer):
    """Load C3 pointer model."""
    model = C3PointerModel(backbone_name="Qwen/Qwen2-0.5B")

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def extract_operands_with_beam(model, tokenizer, problem_text: str, template: str, beam_k: int = 5):
    """
    Use C3 pointer to extract operands with beam search.

    Returns:
        - best_operands: greedy best operands
        - all_candidates: list of (operands, score) tuples for beam search
    """
    input_text = f"[TEMPLATE: {template}] {problem_text}"
    # Extract both numbers AND variables from text
    text_operands = extract_text_operands(problem_text, tokenizer, include_variables=True)
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=384)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])

    arity = TEMPLATE_ARITY.get(template, 2)
    implicit_keys = list(IMPLICIT_VALUES.keys())
    constant_keys = list(DOMAIN_CONSTANTS.keys())

    # For each operand slot, get top-k candidates
    operand_candidates = []  # List of lists: [[cand1, cand2, ...], [cand1, cand2, ...]]

    for i in range(min(arity, MAX_OPERANDS)):
        candidates = []

        # Get source type probabilities
        source_probs = torch.softmax(outputs['source_type_logits'][0, i], dim=0)
        source_types = ['TEXT', 'PRIOR', 'IMPLICIT', 'CONSTANT', 'NONE']

        # For TEXT source: get top-k pointer positions
        if text_operands:
            ptr_logits = outputs['text_pointer_logits'][0, i]
            ptr_probs = torch.softmax(ptr_logits, dim=0)

            # Get top-k positions
            top_k_positions = torch.topk(ptr_probs, min(beam_k * 2, len(ptr_probs))).indices.tolist()

            seen_values = set()  # Dedupe by value within this operand slot
            for pos in top_k_positions:
                # Find closest text number to this position
                closest = min(text_operands, key=lambda x: abs(x[1] - pos))
                value = closest[0]

                # Deduplicate - don't add same value multiple times
                if str(value) in seen_values:
                    continue
                seen_values.add(str(value))

                score = source_probs[0].item() * ptr_probs[pos].item()  # TEXT prob * position prob
                candidates.append((str(value), score, 'TEXT'))

                if len(candidates) >= beam_k:
                    break

        # Add IMPLICIT candidates
        impl_probs = torch.softmax(outputs['implicit_logits'][0, i], dim=0)
        top_impl = torch.topk(impl_probs, min(3, len(impl_probs)))
        for idx, prob in zip(top_impl.indices.tolist(), top_impl.values.tolist()):
            word = implicit_keys[idx]
            value = IMPLICIT_VALUES[word]
            score = source_probs[2].item() * prob  # IMPLICIT prob * word prob
            candidates.append((str(value), score, 'IMPLICIT'))

        # Add CONSTANT candidates
        const_probs = torch.softmax(outputs['constant_logits'][0, i], dim=0)
        top_const = torch.topk(const_probs, min(3, len(const_probs)))
        for idx, prob in zip(top_const.indices.tolist(), top_const.values.tolist()):
            value = constant_keys[idx]
            score = source_probs[3].item() * prob  # CONSTANT prob * value prob
            candidates.append((str(value), score, 'CONSTANT'))

        # Sort by score and take top beam_k
        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:beam_k]

        if not candidates:
            candidates = [('0', 0.0, 'NONE')]

        operand_candidates.append(candidates)

    # Generate all combinations
    all_combinations = []
    for combo in product(*operand_candidates):
        operands = [c[0] for c in combo]

        # Combined score = PRODUCT of individual confidences (joint probability)
        # One low-confidence operand tanks the whole path
        score = 1.0
        for c in combo:
            score *= c[1]

        # Penalize duplicate operands - "b + b" should score lower than "a + b"
        if len(operands) > 1 and len(set(operands)) < len(operands):
            score *= 0.1  # Heavy penalty for degenerate expressions

        all_combinations.append((operands, score))

    # Sort by score
    all_combinations.sort(key=lambda x: -x[1])

    # Best greedy result
    best_operands = all_combinations[0][0] if all_combinations else ['0', '0']

    return best_operands, all_combinations


def extract_operands_with_pointer(model, tokenizer, problem_text: str, template: str):
    """Original greedy extraction for comparison."""
    best_operands, _ = extract_operands_with_beam(model, tokenizer, problem_text, template, beam_k=1)
    return best_operands, []


def brute_force_all_operands(tokenizer, problem_text: str, template: str, gold_expr: str):
    """
    Try ALL combinations of text operands to find the true ceiling.
    Returns (found_correct, best_expr) where found_correct is True if any combo matches gold.
    """
    text_operands = extract_text_operands(problem_text, tokenizer, include_variables=True)
    arity = TEMPLATE_ARITY.get(template, 2)

    # Get all unique operand values
    operand_values = list(set(str(op[0]) for op in text_operands))

    # Also add implicit values
    for word, val in IMPLICIT_VALUES.items():
        operand_values.append(str(val))

    # Try all combinations
    for combo in product(operand_values, repeat=arity):
        operands = list(combo)
        assembly = assemble_and_validate(template, operands)
        if assembly['valid']:
            if check_answer(assembly['expression'], gold_expr):
                return True, assembly['expression']

    return False, None


def run_e2e_test(c3_path: str, test_problems: list, beam_k: int = 10, verbose: bool = False):
    """Run E2E test with pointer + assembler pipeline + beam search."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading C3 pointer model from {c3_path}...")
    model = load_c3_pointer(c3_path, tokenizer)
    print("Model loaded on CPU")

    print(f"\nTesting on {len(test_problems)} problems...")
    print(f"Pipeline: C3 pointer (beam={beam_k}) → C4 assembler → sympy filter → majority vote\n")

    results = {
        'total': len(test_problems),
        'greedy_correct': 0,
        'beam_correct': 0,
        'correct_in_beam': 0,  # Ceiling: was correct answer in any beam candidate?
        'true_ceiling': 0,  # True ceiling: is correct answer possible with ANY operand combo?
    }

    for i, problem in enumerate(test_problems):
        problem_text = problem['problem_text']
        template = problem['template']
        gold_expr = problem['expression']
        gold_operands = problem['operands']

        # Step 1: C3 pointer extracts operand candidates with beam search
        best_operands, all_candidates = extract_operands_with_beam(
            model, tokenizer, problem_text, template, beam_k=beam_k
        )

        # Step 2: Assemble and evaluate ALL candidates through sympy
        valid_answers = []  # (key, score, expression) - key can be numeric value OR expression string
        correct_in_any = False

        for operands, score in all_candidates[:beam_k * beam_k]:  # Limit combinations
            assembly = assemble_and_validate(template, operands)
            if assembly['valid']:
                # Use numeric value if available, otherwise use expression string as key
                if assembly['value'] is not None:
                    key = ('num', round(assembly['value'], 6))
                else:
                    # Symbolic expression - use normalized string as key
                    try:
                        key = ('sym', str(sympy.sympify(assembly['expression'])))
                    except:
                        key = ('sym', assembly['expression'])

                valid_answers.append((key, score, assembly['expression']))

                # Check if this candidate is correct
                if check_answer(assembly['expression'], gold_expr):
                    correct_in_any = True

        if correct_in_any:
            results['correct_in_beam'] += 1

        # True ceiling: try ALL operand combinations
        true_ceiling_found, _ = brute_force_all_operands(tokenizer, problem_text, template, gold_expr)
        if true_ceiling_found:
            results['true_ceiling'] += 1

        # Step 3: Pick the single best valid candidate by score (no voting)
        # Sort valid answers by score descending, take highest
        if valid_answers:
            # Sort by score (highest first)
            valid_answers.sort(key=lambda x: -x[1])
            final_expr = valid_answers[0][2]  # Take the top-scored expression
        else:
            # Fallback to greedy
            assembly = assemble_and_validate(template, best_operands)
            final_expr = assembly['expression'] if assembly['valid'] else f"{best_operands[0]}"

        # Check greedy result
        greedy_assembly = assemble_and_validate(template, best_operands)
        if greedy_assembly['valid'] and check_answer(greedy_assembly['expression'], gold_expr):
            results['greedy_correct'] += 1

        # Check beam result
        if check_answer(final_expr, gold_expr):
            results['beam_correct'] += 1
            status = '✓'
        else:
            status = '✗'

        # Compact output
        ceiling_mark = '★' if correct_in_any else ' '
        print(f"[{i+1:3d}] {template:6s} | beam: {final_expr[:25]:25s} | gold: {gold_expr:20s} {status} {ceiling_mark}")

        if (i + 1) % 20 == 0:
            print(f"\n--- Progress: greedy={results['greedy_correct']}, beam={results['beam_correct']}, ceiling={results['correct_in_beam']} / {i+1} ---\n")

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: C3 Pointer + C4 Assembler (beam_k={beam_k})")
    print("=" * 70)
    print(f"Total problems:        {results['total']}")
    print(f"Greedy correct:        {results['greedy_correct']} ({100*results['greedy_correct']/results['total']:.1f}%)")
    print(f"Beam correct:          {results['beam_correct']} ({100*results['beam_correct']/results['total']:.1f}%)")
    print(f"Ceiling (in beam):     {results['correct_in_beam']} ({100*results['correct_in_beam']/results['total']:.1f}%)")
    print(f"True ceiling (brute):  {results['true_ceiling']} ({100*results['true_ceiling']/results['total']:.1f}%)")
    print()
    print("Comparison: Old causal LM C3 achieved 18% with beam search.")
    print("If beam > 18%, the C3/C4 split is working.")

    return results


def download_model():
    """Download C3 pointer model from S3."""
    import os
    local_path = '/tmp/c3_pointer_model.pt'
    if not os.path.exists(local_path):
        print("Downloading C3 pointer model from S3...")
        s3 = boto3.client('s3')
        s3.download_file('mycelium-data', 'models/c3_pointer/model.pt', local_path)
    return local_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam-k", type=int, default=10, help="Beam size for pointer candidates")
    parser.add_argument("--n-problems", type=int, default=50, help="Number of problems to test")
    parser.add_argument("--verbose", action="store_true", help="Print detailed beam info")
    args = parser.parse_args()

    print("=" * 70)
    print("E2E Test: C3 Pointer + C4 Assembler + Beam Search")
    print("=" * 70)

    # Download model
    c3_path = download_model()

    # Load test problems
    print("\nLoading test problems...")
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='mycelium-data', Key='c3_pointer_training/track_a_pointer.jsonl')
    content = response['Body'].read().decode('utf-8')
    examples = [json.loads(line) for line in content.strip().split('\n') if line.strip()]

    # Take n problems for testing
    test_problems = examples[:args.n_problems]

    run_e2e_test(c3_path, test_problems, beam_k=args.beam_k)


if __name__ == '__main__':
    main()
