#!/usr/bin/env python3
"""
Add attention signatures to dual-signal templates and run GSM8K benchmark.

This script:
1. Loads dual_signal_templates.json (has embeddings, needs attention)
2. Uses fine-tuned MiniLM to extract attention signatures for each template
3. Saves updated templates
4. Runs GSM8K benchmark

Run on VM with GPU for best results.
"""

import json
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def add_attention_signatures(templates_path: str, model_path: str, output_path: str):
    """Add attention signatures to templates using fine-tuned model."""

    from mycelium.dual_signal_templates import SpanDetector

    print(f"Loading templates from {templates_path}...")
    with open(templates_path, 'r') as f:
        templates = json.load(f)
    print(f"Loaded {len(templates)} templates")

    print(f"\nLoading fine-tuned model from {model_path}...")
    if os.path.exists(model_path):
        detector = SpanDetector(model_path=model_path)
        print("Fine-tuned model loaded successfully")
    else:
        print(f"Warning: Model not found at {model_path}")
        print("Using base MiniLM weights")
        detector = SpanDetector(model_path=None)

    print("\nExtracting attention signatures...")
    updated = 0
    for tid, tdata in tqdm(templates.items()):
        examples = tdata.get('span_examples', [])
        if not examples:
            continue

        # Use first example to extract attention
        rep_example = examples[0]

        try:
            embedding, attention, tokens = detector.extract_features(rep_example)

            # Create fixed-size attention signature (128 dim)
            if attention.ndim == 2:
                seq_len = attention.shape[0]
                diag = np.diag(attention)
                row_means = attention.mean(axis=1)
                col_means = attention.mean(axis=0)
                attention_sig = np.concatenate([diag, row_means, col_means])
                if len(attention_sig) > 128:
                    attention_sig = attention_sig[:128]
                else:
                    attention_sig = np.pad(attention_sig, (0, 128 - len(attention_sig)))
            else:
                attention_sig = attention.flatten()[:128]
                if len(attention_sig) < 128:
                    attention_sig = np.pad(attention_sig, (0, 128 - len(attention_sig)))

            # Normalize
            att_norm = np.linalg.norm(attention_sig)
            if att_norm > 0:
                attention_sig = attention_sig / att_norm

            # Also update embedding to match model output
            emb_norm = np.linalg.norm(embedding)
            if emb_norm > 0:
                embedding = embedding / emb_norm

            tdata['attention_signature'] = attention_sig.tolist()
            tdata['embedding_centroid'] = embedding.tolist()
            updated += 1

        except Exception as e:
            print(f"  Warning: Failed for {tid}: {e}")
            continue

    print(f"\nUpdated {updated}/{len(templates)} templates with attention signatures")

    # Save output
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(templates, f, indent=2)

    return templates


def run_benchmark(templates_path: str, model_path: str, n_problems: int = 1000):
    """Run GSM8K benchmark with dual-signal matching."""

    from datasets import load_dataset
    from mycelium.dual_signal_solver import DualSignalSolver
    import re

    print(f"\n{'='*60}")
    print("Running GSM8K Benchmark")
    print(f"{'='*60}")

    # Load GSM8K dataset
    print("\nLoading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    print(f"Loaded {len(dataset)} problems")

    # Initialize solver with dual-signal templates
    print(f"\nInitializing solver with templates from {templates_path}...")
    solver = DualSignalSolver(
        templates_path=templates_path,
        model_path=model_path,
        mock_model=False,  # Use real model
    )
    solver.print_stats()

    # Run benchmark
    print(f"\nEvaluating on {n_problems} problems...")
    correct = 0
    total = 0
    errors = []

    for i, example in enumerate(tqdm(dataset)):
        if i >= n_problems:
            break

        problem = example['question']
        answer_text = example['answer']

        # Extract expected answer from GSM8K format
        # Answer is after "#### " at the end
        match = re.search(r'####\s*([\d,]+)', answer_text)
        if not match:
            continue

        expected = float(match.group(1).replace(',', ''))

        try:
            result = solver.solve(problem)
            predicted = result.answer

            if abs(predicted - expected) < 0.01:
                correct += 1
            else:
                errors.append({
                    'problem': problem[:100],
                    'expected': expected,
                    'predicted': predicted,
                    'operations': [(op.dsl_expr, op.value) for op in result.operations]
                })

            total += 1

        except Exception as e:
            print(f"  Error on problem {i}: {e}")
            continue

    # Print results
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*60}")

    # Show some errors
    if errors[:5]:
        print("\nSample errors:")
        for err in errors[:5]:
            print(f"  Problem: {err['problem']}...")
            print(f"  Expected: {err['expected']}, Got: {err['predicted']}")
            print(f"  Operations: {err['operations']}")
            print()

    return accuracy, correct, total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add attention signatures and run benchmark")
    parser.add_argument(
        "--templates",
        default="dual_signal_templates.json",
        help="Path to templates file"
    )
    parser.add_argument(
        "--model",
        default="models/minilm_attention_finetuned.pt",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--output",
        default="dual_signal_templates.json",
        help="Output path for updated templates"
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=1000,
        help="Number of problems to benchmark"
    )
    parser.add_argument(
        "--skip-attention",
        action="store_true",
        help="Skip attention signature extraction"
    )

    args = parser.parse_args()

    # Add attention signatures
    if not args.skip_attention:
        add_attention_signatures(
            templates_path=args.templates,
            model_path=args.model,
            output_path=args.output,
        )

    # Run benchmark
    run_benchmark(
        templates_path=args.output,
        model_path=args.model,
        n_problems=args.n_problems,
    )
