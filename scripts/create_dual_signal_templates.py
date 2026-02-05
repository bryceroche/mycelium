#!/usr/bin/env python3
"""
Create dual-signal templates from span examples.

This script:
1. Loads all span examples from the export file
2. Runs them through the fine-tuned MiniLM to extract embeddings AND attention
3. Uses verb classifier to assign correct operation types
4. Creates specialized templates with dual-signal data

Output: dual_signal_templates.json with embeddings and attention signatures
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.dual_signal_templates import SpanDetector


def load_span_examples(path: str) -> dict:
    """Load span examples from export file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def infer_operation(examples: list, pattern: str) -> tuple:
    """Infer DSL expression from pattern keywords.

    Returns: (dsl_expr, confidence)
    """
    pattern_lower = pattern.lower()

    # Check for division patterns
    if any(kw in pattern_lower for kw in ['split', 'divided', 'shared equally', 'distributed']):
        return "entity / value", 0.6

    # Check for multiplication patterns
    if any(kw in pattern_lower for kw in ['times', 'each', 'per', 'doubled', 'tripled']):
        return "entity * value", 0.6

    # Check for subtraction patterns
    if any(kw in pattern_lower for kw in ['gave', 'sold', 'spent', 'lost', 'ate', 'used']):
        return "entity - value", 0.6

    # Check for addition patterns
    if any(kw in pattern_lower for kw in ['found', 'earned', 'received', 'bought', 'got']):
        return "entity + value", 0.6

    # Default to value assignment
    return "value", 0.5


def create_dual_signal_templates(
    input_path: str,
    output_path: str,
    model_path: str,
    batch_size: int = 32,
):
    """Create dual-signal templates with embeddings and attention signatures."""

    print(f"Loading span examples from {input_path}...")
    raw_templates = load_span_examples(input_path)
    print(f"Loaded {len(raw_templates)} templates")

    # Initialize span detector with fine-tuned model
    print(f"\nLoading fine-tuned model from {model_path}...")
    if os.path.exists(model_path):
        detector = SpanDetector(model_path=model_path)
        print("Fine-tuned model loaded successfully")
    else:
        print(f"Warning: Model not found at {model_path}, using base MiniLM")
        detector = SpanDetector(model_path=None)

    # Process each template
    dual_signal_templates = {}
    stats = defaultdict(int)

    print(f"\nProcessing {len(raw_templates)} templates...")

    for i, (tid, tdata) in enumerate(raw_templates.items()):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(raw_templates)}")

        examples = tdata.get('examples', [])
        pattern = tdata.get('pattern', '')

        if not examples:
            stats['skipped_no_examples'] += 1
            continue

        # Infer DSL expression from pattern
        dsl_expr, op_confidence = infer_operation(examples, pattern)
        stats[f'dsl_{dsl_expr}'] += 1

        # Get representative example (first one)
        rep_example = examples[0]

        # Extract dual-signal features
        try:
            embedding, attention, tokens = detector.extract_features(rep_example)

            # Flatten attention to 1D signature
            # Use mean across sequence positions to get fixed-size signature
            if attention.ndim == 2:
                # attention is [seq_len, seq_len]
                # Take diagonal + off-diagonal means as signature
                seq_len = attention.shape[0]
                # Create signature: diagonal attention + row means + col means
                diag = np.diag(attention)
                row_means = attention.mean(axis=1)
                col_means = attention.mean(axis=0)
                # Concatenate and pad/truncate to fixed size (128)
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
            emb_norm = np.linalg.norm(embedding)
            if emb_norm > 0:
                embedding = embedding / emb_norm

            att_norm = np.linalg.norm(attention_sig)
            if att_norm > 0:
                attention_sig = attention_sig / att_norm

        except Exception as e:
            print(f"  Warning: Failed to extract features for {tid}: {e}")
            stats['extraction_failed'] += 1
            continue

        # Create dual-signal template
        dual_signal_templates[tid] = {
            "template_id": tid,
            "dsl_expr": dsl_expr,
            "pattern": pattern,
            "dsl_expr": dsl_expr,
            "embedding_centroid": embedding.tolist(),
            "attention_signature": attention_sig.tolist(),
            "span_examples": examples[:10],  # Keep up to 10 examples
            "count": tdata.get('count', len(examples)),
            "welford_count": 0,
            "welford_mean": 0.0,
            "welford_m2": 0.0,
        }
        stats['created'] += 1

    # Save output
    print(f"\nSaving {len(dual_signal_templates)} dual-signal templates to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(dual_signal_templates, f, indent=2)

    # Print stats
    print("\n=== Statistics ===")
    print(f"Templates created: {stats['created']}")
    print(f"Skipped (no examples): {stats['skipped_no_examples']}")
    print(f"Extraction failed: {stats['extraction_failed']}")
    print("\nDSL distribution:")
    for dsl in ['value', 'entity + value', 'entity - value', 'entity * value', 'entity / value']:
        print(f"  {dsl}: {stats.get(f'dsl_{dsl}', 0)}")

    return dual_signal_templates


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create dual-signal templates")
    parser.add_argument(
        "--input",
        default="span_templates_export.json",
        help="Input span templates file"
    )
    parser.add_argument(
        "--output",
        default="dual_signal_templates.json",
        help="Output dual-signal templates file"
    )
    parser.add_argument(
        "--model",
        default="models/minilm_attention_finetuned.pt",
        help="Path to fine-tuned MiniLM model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )

    args = parser.parse_args()

    create_dual_signal_templates(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        batch_size=args.batch_size,
    )

    print("\nDone!")
