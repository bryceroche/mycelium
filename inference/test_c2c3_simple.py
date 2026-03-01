#!/usr/bin/env python3
"""
Simple E2E Test: C2 Classifier + C3-Extract Span Extraction + Assembler

Tests the pipeline on simple single-operation MATH problems.
"""

import json
import torch
import sympy
import boto3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.c3_extract import C3ExtractModel
from inference.assembler import assemble_and_validate


def load_c2_model(model_path: str):
    """Load C2 classifier."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def load_c3_model(model_path: str):
    """Load C3-Extract span extraction model."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = C3ExtractModel(backbone_name="Qwen/Qwen2-0.5B")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return tokenizer, model


def classify_operation(tokenizer, model, problem_text: str, threshold: float = 0.3):
    """Classify operation type with multi-label thresholding."""
    inputs = tokenizer(problem_text, return_tensors='pt', truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0]

    # Get labels above threshold
    labels = []
    for idx, prob in enumerate(probs):
        if prob > threshold:
            label = model.config.id2label[idx]
            labels.append((label, prob.item()))

    # Sort by probability
    labels.sort(key=lambda x: -x[1])
    return labels


def extract_operands(tokenizer, model, problem_text: str, template: str):
    """Extract operand spans from problem text."""
    input_text = f"[TEMPLATE: {template}] {problem_text}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True,
                       max_length=384, padding='max_length')

    operands = model.extract_spans(
        inputs['input_ids'],
        inputs['attention_mask'],
        tokenizer,
        template=template,
        confidence_threshold=0.3
    )

    return operands


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted expression equals gold using symbolic equivalence."""
    try:
        pred = sympy.sympify(predicted)
        gold_expr = sympy.sympify(gold)

        # Numeric comparison
        if pred.is_number and gold_expr.is_number:
            return abs(float(pred) - float(gold_expr)) < 1e-6

        # Symbolic comparison
        return sympy.simplify(pred - gold_expr) == 0
    except:
        return False


def run_e2e_test(c2_path: str, c3_path: str, test_problems: list):
    """Run E2E test on simple problems."""
    print("Loading C2 classifier...")
    c2_tokenizer, c2_model = load_c2_model(c2_path)

    print("Loading C3-Extract model...")
    c3_tokenizer, c3_model = load_c3_model(c3_path)

    print(f"\nTesting on {len(test_problems)} problems...\n")

    results = {
        'total': len(test_problems),
        'c2_correct': 0,
        'c3_spans_found': 0,
        'assembly_valid': 0,
        'answer_correct': 0,
    }

    for i, problem in enumerate(test_problems):
        problem_text = problem['problem_text']
        gold_template = problem['template']
        gold_expr = problem['expression']
        gold_operands = problem['operands']

        print(f"[{i+1}] Template: {gold_template}, Expression: {gold_expr}")

        # Step 1: C2 classify
        classifications = classify_operation(c2_tokenizer, c2_model, problem_text)
        pred_templates = [c[0] for c in classifications]
        c2_correct = gold_template in pred_templates
        if c2_correct:
            results['c2_correct'] += 1
        print(f"  C2: {pred_templates[:3]} {'✓' if c2_correct else '✗'}")

        # Step 2: C3-Extract operands (using gold template for now)
        operands = extract_operands(c3_tokenizer, c3_model, problem_text, gold_template)
        operand_texts = [op['text'] for op in operands]

        # Check if operands match (order-independent)
        c3_correct = set(operand_texts) == set(gold_operands)
        if c3_correct:
            results['c3_spans_found'] += 1
        print(f"  C3: {operand_texts} (gold: {gold_operands}) {'✓' if c3_correct else '✗'}")

        # Step 3: Assemble
        assembly = assemble_and_validate(gold_template, operand_texts)
        if assembly['valid']:
            results['assembly_valid'] += 1
            print(f"  Asm: {assembly['expression']} (valid: {assembly['valid']})")

            # Step 4: Check answer
            if check_answer(assembly['expression'], gold_expr):
                results['answer_correct'] += 1
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ Wrong (expected: {gold_expr})")
        else:
            print(f"  Asm: FAILED - {assembly['error']}")

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total problems:    {results['total']}")
    print(f"C2 correct:        {results['c2_correct']} ({100*results['c2_correct']/results['total']:.1f}%)")
    print(f"C3 spans found:    {results['c3_spans_found']} ({100*results['c3_spans_found']/results['total']:.1f}%)")
    print(f"Assembly valid:    {results['assembly_valid']} ({100*results['assembly_valid']/results['total']:.1f}%)")
    print(f"Answer correct:    {results['answer_correct']} ({100*results['answer_correct']/results['total']:.1f}%)")

    return results


def download_c2_model():
    """Download C2 model from S3."""
    import os
    c2_dir = '/tmp/c2_model'
    os.makedirs(c2_dir, exist_ok=True)

    s3 = boto3.client('s3')
    files = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']

    for f in files:
        local_path = f'{c2_dir}/{f}'
        if not os.path.exists(local_path):
            print(f"  Downloading {f}...")
            s3.download_file('mycelium-data', f'models/c2_ib_templates_frozen_v1/{f}', local_path)

    return c2_dir


def main():
    import os

    # Download C2 model
    print("Downloading C2 model from S3...")
    c2_path = download_c2_model()

    # Download C3 model from S3
    c3_local_path = '/tmp/c3_extract_model.pt'
    if not os.path.exists(c3_local_path):
        print("Downloading C3 model from S3...")
        s3 = boto3.client('s3')
        s3.download_file('mycelium-data', 'models/c3_extract/model.pt', c3_local_path)

    # Load test problems from Track A simple data
    print("Loading test problems...")
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='mycelium-data', Key='c3_span_training/track_a_simple.jsonl')
    content = response['Body'].read().decode('utf-8')
    examples = [json.loads(line) for line in content.strip().split('\n') if line.strip()]

    # Take first 50 for testing
    test_problems = examples[:50]

    run_e2e_test(c2_path, c3_local_path, test_problems)


if __name__ == '__main__':
    main()
