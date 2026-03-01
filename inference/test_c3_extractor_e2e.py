#!/usr/bin/env python3
"""
E2E Test: SQuAD-style C3 Span Extractor

Quick sanity check to see if 58% span exact match translates to E2E accuracy.
Uses beam search over span candidates + sympy validation.
"""

import json
import re
import torch
import sympy
from sympy import sqrt, sin, cos, tan, log, factorial, Mod
import boto3
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer

# Try importing from models, fallback to local definitions
try:
    from models.c3_extractor import (
        C3SpanExtractor,
        format_input_with_priors,
        MAX_OPERANDS,
        MAX_SEQ_LEN,
        MODEL_NAME,
    )
except ImportError:
    # Local definitions when model module unavailable
    MAX_OPERANDS = 4
    MAX_SEQ_LEN = 512
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"

    # Placeholder - will be loaded from checkpoint
    C3SpanExtractor = None

    def format_input_with_priors(problem_text: str, template: str, prior_results: List = None) -> str:
        """Build input with PRIOR values injected as extractable spans.

        MUST match training's build_input_with_priors() exactly:
        - 0-indexed PRIOR tags: [PRIOR_0: val], [PRIOR_1: val], ...
        - Single space between parts (no double spaces)
        """
        parts = [f"[TEMPLATE: {template}]"]
        if prior_results:
            for i, val in enumerate(prior_results):
                # Format value like training does
                if isinstance(val, float) and val == int(val):
                    val_str = str(int(val))
                else:
                    val_str = str(val)
                parts.append(f"[PRIOR_{i}: {val_str}]")
        parts.append(problem_text)
        return " ".join(parts)

# Expanded word map for operand resolution
WORD_MAP = {
    "half": 0.5, "quarter": 0.25, "third": 1/3,
    "twice": 2, "double": 2, "triple": 3,
    "dozen": 12, "hundred": 100, "thousand": 1000,
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
    "pi": 3.14159265359, "e": 2.71828182846,
}

# Template arities
TEMPLATE_ARITY = {
    'ADD': 2, 'SUB': 2, 'MUL': 2, 'DIV': 2,
    'ADD3': 3, 'MUL3': 3, 'ADD4': 4,
    'SQUARE': 1, 'SQRT': 1, 'CUBE': 1,
    'HIGH_POW': 2,
    'SIN': 1, 'COS': 1, 'TAN': 1,
    'LOG': 1, 'FACTORIAL': 1, 'MOD': 2,
}


def resolve_span(span_text: str, prior_results: List[float] = None) -> Any:
    """Convert extracted span to value, resolving PRIOR references."""
    text = span_text.strip()
    text_lower = text.lower()

    # Check for PRIOR reference first (e.g., "PRIOR_0", "[PRIOR_0: 42]")
    # Training uses 0-indexed PRIORs
    prior_match = re.search(r'PRIOR_(\d+)', text, re.IGNORECASE)
    if prior_match and prior_results:
        prior_idx = int(prior_match.group(1))  # Already 0-indexed
        if 0 <= prior_idx < len(prior_results):
            return prior_results[prior_idx]

    # Try as number
    try:
        return float(text)
    except ValueError:
        pass

    # Try word mapping
    if text_lower in WORD_MAP:
        return WORD_MAP[text_lower]

    # Single letter variable
    if len(text) == 1 and text.isalpha():
        return sympy.Symbol(text)

    # Try to extract number from span
    nums = re.findall(r'-?\d+\.?\d*', text)
    if nums:
        try:
            return float(nums[0])
        except:
            pass

    return None


def extract_spans_with_beam(
    model,
    tokenizer,
    problem_text: str,
    template: str,
    prior_results: List[float] = None,
    beam_k: int = 5,
) -> List[Tuple[List[str], float]]:
    """
    Extract operand spans with beam search.

    Returns list of (operand_spans, score) tuples.
    """
    input_text = format_input_with_priors(problem_text, template, prior_results)

    encodings = tokenizer(
        input_text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding='max_length',
        return_tensors='pt',
    )

    device = next(model.parameters()).device
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask)

    # start_logits: (1, seq, max_operands)
    start_probs = torch.softmax(start_logits[0], dim=0)  # (seq, max_operands)
    end_probs = torch.softmax(end_logits[0], dim=0)

    arity = TEMPLATE_ARITY.get(template, 2)

    # For each operand slot, get top-k start/end candidates
    all_candidates = []

    for slot in range(min(arity, MAX_OPERANDS)):
        slot_start_probs = start_probs[:, slot]
        slot_end_probs = end_probs[:, slot]

        # Get top-k starts
        topk_starts = torch.topk(slot_start_probs, min(beam_k, len(slot_start_probs)))
        topk_ends = torch.topk(slot_end_probs, min(beam_k, len(slot_end_probs)))

        slot_candidates = []
        for si, (start_idx, start_p) in enumerate(zip(topk_starts.indices, topk_starts.values)):
            for ei, (end_idx, end_p) in enumerate(zip(topk_ends.indices, topk_ends.values)):
                start_i = start_idx.item()
                end_i = end_idx.item()

                # Skip invalid spans
                if end_i < start_i:
                    continue
                if end_i - start_i > 10:  # Skip very long spans
                    continue

                # Extract span text
                tokens = input_ids[0, start_i:end_i+1]
                span_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()

                score = (start_p * end_p).item()
                slot_candidates.append((span_text, score))

        # Deduplicate and sort
        seen = {}
        for text, score in slot_candidates:
            if text not in seen or score > seen[text]:
                seen[text] = score

        slot_candidates = sorted(seen.items(), key=lambda x: -x[1])[:beam_k]
        all_candidates.append(slot_candidates)

    # Generate beam combinations
    from itertools import product

    if not all_candidates:
        return []

    beam_results = []
    for combo in product(*all_candidates):
        spans = [c[0] for c in combo]
        score = sum(c[1] for c in combo) / len(combo)
        beam_results.append((spans, score))

    beam_results.sort(key=lambda x: -x[1])
    return beam_results[:beam_k * 2]


def assemble_expression(template: str, operands: List[Any]) -> str:
    """Assemble expression from template and operands."""
    if template == 'ADD':
        return f"({operands[0]}) + ({operands[1]})"
    elif template == 'SUB':
        return f"({operands[0]}) - ({operands[1]})"
    elif template == 'MUL':
        return f"({operands[0]}) * ({operands[1]})"
    elif template == 'DIV':
        return f"({operands[0]}) / ({operands[1]})"
    elif template == 'ADD3':
        return f"({operands[0]}) + ({operands[1]}) + ({operands[2]})"
    elif template == 'MUL3':
        return f"({operands[0]}) * ({operands[1]}) * ({operands[2]})"
    elif template == 'ADD4':
        return f"({operands[0]}) + ({operands[1]}) + ({operands[2]}) + ({operands[3]})"
    elif template == 'SQUARE':
        return f"({operands[0]})**2"
    elif template == 'SQRT':
        return f"sqrt({operands[0]})"
    elif template == 'CUBE':
        return f"({operands[0]})**3"
    elif template == 'HIGH_POW':
        return f"({operands[0]})**({operands[1]})"
    elif template == 'SIN':
        return f"sin({operands[0]})"
    elif template == 'COS':
        return f"cos({operands[0]})"
    elif template == 'TAN':
        return f"tan({operands[0]})"
    elif template == 'LOG':
        return f"log({operands[0]})"
    elif template == 'FACTORIAL':
        return f"factorial({operands[0]})"
    elif template == 'MOD':
        return f"Mod({operands[0]}, {operands[1]})"
    else:
        raise ValueError(f"Unknown template: {template}")


def evaluate_expression(expr_str: str) -> float | None:
    """Evaluate expression with sympy."""
    try:
        result = sympy.sympify(expr_str)
        if result.is_number:
            return float(result)
        return None
    except:
        return None


def check_answer(predicted: float | None, gold: str) -> bool:
    """Check if predicted equals gold answer."""
    if predicted is None:
        return False
    try:
        gold_val = float(sympy.sympify(gold))
        return abs(predicted - gold_val) < 1e-6
    except:
        return False


def execute_single_step(
    model,
    tokenizer,
    problem_text: str,
    template: str,
    prior_results: List[float],
    beam_k: int = 5,
) -> Tuple[Optional[float], List[Tuple[List[str], float]]]:
    """
    Execute a single step: extract spans, resolve, assemble, evaluate.
    Returns (best_result, beam_results).
    """
    beam_results = extract_spans_with_beam(
        model, tokenizer, problem_text, template,
        prior_results=prior_results,
        beam_k=beam_k
    )

    if not beam_results:
        return None, []

    # Try greedy (first beam result)
    spans, _ = beam_results[0]
    operands = [resolve_span(s, prior_results) for s in spans]

    if None in operands:
        return None, beam_results

    try:
        expr = assemble_expression(template, operands)
        result = evaluate_expression(expr)
        return result, beam_results
    except ValueError:
        return None, beam_results


def extract_gold_priors(provenance: List[Dict]) -> List[float]:
    """Extract gold PRIOR values from provenance to inject into input.

    Training data has source_type='PRIOR' with prior_index and value.
    We need these to reconstruct the input the model was trained on.
    """
    prior_values = {}
    for prov in provenance:
        if prov.get('source_type') == 'PRIOR':
            idx = prov.get('prior_index', 0)
            val = prov.get('value')
            if val is not None:
                try:
                    prior_values[idx] = float(val)
                except (ValueError, TypeError):
                    # Skip non-numeric values (variables, expressions, etc.)
                    continue

    # Return as ordered list
    if not prior_values:
        return []
    max_idx = max(prior_values.keys())
    return [prior_values.get(i, 0.0) for i in range(max_idx + 1)]


def run_e2e_test(
    model,
    tokenizer,
    test_examples: List[Dict],
    beam_k: int = 5,
    max_examples: int = 100,
) -> Dict[str, Any]:
    """Run E2E test on examples with gold PRIOR injection."""

    correct = 0
    correct_in_beam = 0
    total = 0

    # Per-template tracking
    template_stats = {}

    for ex in test_examples[:max_examples]:
        problem_text = ex['problem_text']
        template = ex['template']
        gold_expr = ex.get('expression', '')
        provenance = ex.get('provenance', [])

        # Extract gold PRIOR values from provenance to match training format
        gold_priors = extract_gold_priors(provenance)

        # Track template stats
        if template not in template_stats:
            template_stats[template] = {'total': 0, 'correct': 0}
        template_stats[template]['total'] += 1

        # Execute with gold priors injected
        result, beam_results = execute_single_step(
            model, tokenizer, problem_text, template,
            prior_results=gold_priors,
            beam_k=beam_k
        )
        final_result = result

        # Check if final result matches gold
        greedy_correct = check_answer(final_result, gold_expr)
        if greedy_correct:
            correct += 1
            template_stats[template]['correct'] += 1

        # Check if answer exists in any beam combination
        found_in_beam = greedy_correct
        if not found_in_beam and beam_results:
            for spans, score in beam_results:
                operands = [resolve_span(s, gold_priors) for s in spans]
                if None in operands:
                    continue
                try:
                    expr = assemble_expression(template, operands)
                    result = evaluate_expression(expr)
                    if check_answer(result, gold_expr):
                        found_in_beam = True
                        break
                except ValueError:
                    continue

        if found_in_beam:
            correct_in_beam += 1

        total += 1

        if total % 20 == 0:
            print(f"  Progress: {total}/{max_examples}, "
                  f"Greedy: {correct}/{total} ({100*correct/total:.1f}%), "
                  f"In-beam: {correct_in_beam}/{total} ({100*correct_in_beam/total:.1f}%)")

    return {
        'total': total,
        'greedy_correct': correct,
        'greedy_acc': correct / total if total > 0 else 0,
        'beam_correct': correct_in_beam,
        'beam_acc': correct_in_beam / total if total > 0 else 0,
        'template_stats': template_stats,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/c3_extractor/best/model.pt")
    parser.add_argument("--data-path", default="s3://mycelium-data/c3_pointer_training/track_a_b_pointer.jsonl")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--beam-k", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("C3 SQuAD-style Span Extractor E2E Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = C3SpanExtractor()
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded. Checkpoint epoch: {checkpoint.get('epoch', '?')}")
    print(f"Checkpoint metrics: {checkpoint.get('metrics', {})}")

    # Load test data
    print(f"\nLoading test data from {args.data_path}...")
    if args.data_path.startswith("s3://"):
        s3 = boto3.client('s3')
        bucket = args.data_path.split('/')[2]
        key = '/'.join(args.data_path.split('/')[3:])
        response = s3.get_object(Bucket=bucket, Key=key)
        examples = [json.loads(line) for line in response['Body'].iter_lines() if line]
    else:
        with open(args.data_path) as f:
            examples = [json.loads(line) for line in f if line.strip()]

    # Use validation split (last 10%)
    split_idx = int(len(examples) * 0.9)
    test_examples = examples[split_idx:]
    print(f"Using {len(test_examples)} test examples (validation split)")

    # Run E2E test
    print(f"\nRunning E2E test (beam_k={args.beam_k})...")
    results = run_e2e_test(
        model, tokenizer, test_examples,
        beam_k=args.beam_k,
        max_examples=args.max_examples
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total examples:     {results['total']}")
    print(f"Greedy accuracy:    {results['greedy_correct']}/{results['total']} = {100*results['greedy_acc']:.1f}%")
    print(f"In-beam accuracy:   {results['beam_correct']}/{results['total']} = {100*results['beam_acc']:.1f}%")

    # Per-template breakdown
    template_stats = results.get('template_stats', {})
    if template_stats:
        print("\n" + "-" * 40)
        print("PER-TEMPLATE BREAKDOWN")
        print("-" * 40)
        for template, stats in sorted(template_stats.items()):
            total = stats['total']
            correct = stats['correct']
            acc = 100 * correct / total if total > 0 else 0
            print(f"  {template:12s}: {correct:3d}/{total:3d} = {acc:5.1f}%")
        print("-" * 40)

    print("=" * 60)


if __name__ == "__main__":
    main()
