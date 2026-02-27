#!/usr/bin/env python3
"""
Mycelium v6: Generate Multi-Span Classifier Training Data

Key change from single-span: find ALL spans whose operands appear in each
CoT step, mark them all with <SPAN> tags. The classifier learns to classify
groups of 1, 2, or 3 spans as a single operation.

Training data derived from Qwen-7B CoT:
- For each CoT step, find ALL spans that contribute operands
- Mark all matching spans with <SPAN>...</SPAN>
- Label = operation type (ADD, SUB, MUL, DIV)
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import random
from transformers import AutoTokenizer


# BIO label mapping
BIO_LABELS = ["O", "B-OP", "I-OP", "B-Q", "I-Q"]

# Operation labels
OP_LABELS = ["ADD", "SUB", "MUL", "DIV"]


def extract_spans_from_bio(input_ids, bio_labels, tokenizer, problem_text):
    """
    Extract text spans from BIO labels over tokenized input.
    Returns list of {"text": str, "tag": "OP" or "Q", "start": int, "end": int}
    """
    spans = []
    current_span = None

    for i, (tok_id, label_id) in enumerate(zip(input_ids, bio_labels)):
        if label_id < 0 or label_id >= len(BIO_LABELS):
            continue

        label = BIO_LABELS[label_id]

        if label.startswith("B-"):
            if current_span is not None:
                spans.append(current_span)
            tag = label[2:]
            current_span = {"token_ids": [tok_id], "tag": tag, "start_idx": i}
        elif label.startswith("I-") and current_span is not None:
            current_span["token_ids"].append(tok_id)
        else:
            if current_span is not None:
                spans.append(current_span)
                current_span = None

    if current_span is not None:
        spans.append(current_span)

    # Convert token spans to text
    text_spans = []
    for idx, sp in enumerate(spans):
        text = tokenizer.decode(sp["token_ids"], skip_special_tokens=True).strip()
        if len(text) > 3:
            start = problem_text.find(text)
            if start == -1:
                start = problem_text.lower().find(text.lower())
            if start == -1:
                partial = text[:20]
                start = problem_text.find(partial)

            text_spans.append({
                "text": text,
                "tag": sp["tag"],
                "start": start if start != -1 else None,
                "idx": idx,
            })

    return text_spans


def extract_numbers_from_text(text):
    """Extract all numbers from text."""
    numbers = set()
    for match in re.finditer(r'-?\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', text):
        num_str = match.group(1).replace(",", "")
        try:
            numbers.add(float(num_str))
        except ValueError:
            pass
    return numbers


def parse_operation_type(step_text):
    """
    DEPRECATED: Keyword heuristics removed.

    Operation type should come from IB template labels in the training data,
    not from keyword matching. This function now returns None.
    """
    # DO NOT ADD KEYWORD HEURISTICS HERE
    # Use IB template labels from data/ib_templates/ instead
    return None


def mark_multiple_spans(problem_text, spans):
    """
    Mark multiple spans in problem text with <SPAN>...</SPAN> tags.
    Returns marked text or None if marking fails.
    """
    if not spans:
        return None

    # Get positions for all spans
    positions = []
    for span in spans:
        text = span["text"]
        start = span.get("start")

        if start is None:
            start = problem_text.find(text)
            if start == -1:
                start = problem_text.lower().find(text.lower())

        if start == -1:
            return None

        end = start + len(text)
        positions.append((start, end, span))

    # Check for overlaps
    positions.sort(key=lambda x: x[0])
    for i in range(len(positions) - 1):
        if positions[i][1] > positions[i + 1][0]:
            return None  # Overlapping spans

    # Insert markers from end to start
    result = problem_text
    for start, end, span in reversed(positions):
        result = result[:end] + " </SPAN>" + result[end:]
        result = result[:start] + "<SPAN> " + result[start:]

    return result


def find_all_matching_spans(op_spans, step_operands):
    """
    Find ALL spans that contribute operands to a CoT step.
    Returns list of span indices.
    """
    matching = []

    for idx, span in enumerate(op_spans):
        span_numbers = extract_numbers_from_text(span["text"])
        if span_numbers & step_operands:
            matching.append(idx)

    return matching


def main():
    print("=" * 60)
    print("MYCELIUM V6: MULTI-SPAN CLASSIFIER DATA GENERATION")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)

    # Load BIO segmentation data
    bio_path = Path("data/bio_segmentation_clean.json")
    print(f"\nLoading BIO segmentation data from {bio_path}...")
    with open(bio_path) as f:
        bio_data = json.load(f)
    print(f"Loaded {len(bio_data)} problems with BIO labels")

    # Load JSD data (has parsed operations/CoT steps)
    data_dir = Path("data/v6_integrated")
    local_files = sorted(data_dir.glob("results_worker*.json"))
    print(f"\nLoading JSD data from {len(local_files)} files...")
    jsd_data = {}
    for f in local_files:
        with open(f) as fp:
            for item in json.load(fp):
                jsd_data[item['question']] = item
    print(f"Indexed {len(jsd_data)} JSD entries")

    # Index BIO data by problem text
    bio_by_problem = {}
    for item in bio_data:
        bio_by_problem[item["problem_text"]] = item

    # Generate classifier training pairs
    print("\nGenerating multi-span classifier training pairs...")
    training_pairs = []
    stats = Counter()

    for problem_text, bio_item in tqdm(bio_by_problem.items()):
        # Get JSD entry
        jsd_entry = jsd_data.get(problem_text)
        if not jsd_entry:
            stats["no_jsd_match"] += 1
            continue

        operations = jsd_entry.get("operations", [])
        if not operations:
            stats["no_operations"] += 1
            continue

        # Extract spans from BIO labels
        input_ids = bio_item.get("input_ids", [])
        bio_labels = bio_item.get("bio_labels", [])

        if not input_ids or not bio_labels:
            stats["no_bio_data"] += 1
            continue

        spans = extract_spans_from_bio(input_ids, bio_labels, tokenizer, problem_text)

        # Filter to OP spans only
        op_spans = [s for s in spans if s.get("tag") == "OP"]
        if not op_spans:
            stats["no_op_spans"] += 1
            continue

        # For each CoT step, find ALL matching spans
        for step_idx, step in enumerate(operations):
            # Get operands from CoT step
            step_operands = set()
            if step.get("arg1") is not None:
                try:
                    step_operands.add(float(step["arg1"]))
                except (ValueError, TypeError):
                    pass
            if step.get("arg2") is not None:
                try:
                    step_operands.add(float(step["arg2"]))
                except (ValueError, TypeError):
                    pass

            if not step_operands:
                stats["no_step_operands"] += 1
                continue

            # Find ALL spans that match
            matching_indices = find_all_matching_spans(op_spans, step_operands)

            if not matching_indices:
                stats["no_matching_spans"] += 1
                continue

            # Parse operation type (field is op_type, values are lowercase)
            op_type = step.get("op_type", "").upper()
            if op_type not in OP_LABELS:
                # Try parsing from segment text
                raw = step.get("segment_text", "")
                op_type = parse_operation_type(raw)

            if op_type not in OP_LABELS:
                stats["unknown_op_type"] += 1
                continue

            # Get matching spans
            matching_spans = [op_spans[i] for i in matching_indices]

            # Mark all matching spans
            marked = mark_multiple_spans(problem_text, matching_spans)
            if marked is None:
                stats["marking_failed"] += 1
                continue

            training_pairs.append({
                "marked_problem": marked,
                "operation": op_type,
                "n_spans": len(matching_spans),
                "span_texts": [s["text"] for s in matching_spans],
                "step_idx": step_idx,
                "problem_text": problem_text,
            })

            stats[f"spans_{len(matching_spans)}"] += 1
            stats[f"op_{op_type}"] += 1

    print(f"\nGenerated {len(training_pairs)} training pairs")
    print("\nStatistics:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # Span count distribution
    print("\nSpan count distribution:")
    for i in range(1, 5):
        count = stats.get(f"spans_{i}", 0)
        pct = 100 * count / len(training_pairs) if training_pairs else 0
        print(f"  {i} span(s): {count} ({pct:.1f}%)")

    # Operation distribution
    print("\nOperation distribution:")
    for op in OP_LABELS:
        count = stats.get(f"op_{op}", 0)
        pct = 100 * count / len(training_pairs) if training_pairs else 0
        print(f"  {op}: {count} ({pct:.1f}%)")

    # Save training data
    output_path = Path("data/classification_multispan.json")
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(training_pairs, f)

    # Show examples
    print("\n" + "=" * 60)
    print("SAMPLE TRAINING PAIRS")
    print("=" * 60)

    random.seed(42)

    # Show examples by span count
    for n_spans in [1, 2, 3]:
        examples = [p for p in training_pairs if p["n_spans"] == n_spans]
        if examples:
            print(f"\n--- {n_spans}-SPAN EXAMPLES ---")
            for sample in random.sample(examples, min(2, len(examples))):
                print(f"\nOperation: {sample['operation']}")
                print(f"Spans: {sample['span_texts']}")
                print(f"Marked (first 200 chars):\n{sample['marked_problem'][:200]}...")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
