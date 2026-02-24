#!/usr/bin/env python3
"""
Mycelium v6: Generate Multi-Span Extractor Training Data

Key change: arguments are extracted from GROUPS of spans, not single spans.
When multiple spans contribute to an operation, all are marked and the
extractor learns to pull the right arguments from the group.

Input format: [OP_LABEL]\n<problem with multiple marked spans>
Output format: arg0|source\narg1|source

Sources: PROB, IMP:word, PREV, PREV:N, UNK
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

# Implicit value mappings
IMPLICIT_VALUES = {
    "half": 2, "double": 2, "twice": 2, "triple": 3, "third": 3,
    "quarter": 4, "fourth": 4, "dozen": 12, "pair": 2, "score": 20,
    "hundred": 100, "thousand": 1000, "million": 1000000,
}


def extract_spans_from_bio(input_ids, bio_labels, tokenizer, problem_text):
    """Extract text spans from BIO labels."""
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


def mark_multiple_spans(problem_text, spans):
    """Mark multiple spans with <SPAN>...</SPAN> tags."""
    if not spans:
        return None

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

    positions.sort(key=lambda x: x[0])
    for i in range(len(positions) - 1):
        if positions[i][1] > positions[i + 1][0]:
            return None

    result = problem_text
    for start, end, span in reversed(positions):
        result = result[:end] + " </SPAN>" + result[end:]
        result = result[:start] + "<SPAN> " + result[start:]

    return result


def find_all_matching_spans(op_spans, step_operands):
    """Find ALL spans that contribute operands to a CoT step."""
    matching = []
    for idx, span in enumerate(op_spans):
        span_numbers = extract_numbers_from_text(span["text"])
        if span_numbers & step_operands:
            matching.append(idx)
    return matching


def classify_arg_source(value, span_texts, problem_text, prev_results, step_idx):
    """
    Classify the source of an argument value.
    Returns: PROB, IMP:word, PREV, PREV:N, or UNK
    """
    # Combine all span texts for checking
    all_span_text = " ".join(span_texts).lower()

    # Check if value appears explicitly in ANY span
    value_str = str(int(value)) if value == int(value) else str(value)
    if value_str in all_span_text:
        return "PROB"

    # Also check in problem text
    if value_str in problem_text:
        return "PROB"

    # Check for implicit values
    for word, impl_value in IMPLICIT_VALUES.items():
        if word in all_span_text and abs(value - impl_value) < 0.01:
            return f"IMP:{word}"

    # Check if it's a previous result
    for prev_idx, prev_result in enumerate(prev_results):
        if prev_result is not None and abs(value - prev_result) < 0.01:
            steps_back = step_idx - prev_idx
            if steps_back == 1:
                return "PREV"
            else:
                return f"PREV:{steps_back}"

    return "UNK"


def format_output(arg1, arg1_source, arg2, arg2_source):
    """Format extractor output."""
    lines = []
    if arg1 is not None:
        val = int(arg1) if arg1 == int(arg1) else arg1
        lines.append(f"{val}|{arg1_source}")
    if arg2 is not None:
        val = int(arg2) if arg2 == int(arg2) else arg2
        lines.append(f"{val}|{arg2_source}")
    return "\n".join(lines) if lines else "NONE"


def main():
    print("=" * 60)
    print("MYCELIUM V6: MULTI-SPAN EXTRACTOR DATA GENERATION")
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

    # Load JSD data
    data_dir = Path("data/v6_integrated")
    local_files = sorted(data_dir.glob("results_worker*.json"))
    print(f"\nLoading JSD data from {len(local_files)} files...")
    jsd_data = {}
    for f in local_files:
        with open(f) as fp:
            for item in json.load(fp):
                jsd_data[item['question']] = item
    print(f"Indexed {len(jsd_data)} JSD entries")

    # Index BIO data
    bio_by_problem = {}
    for item in bio_data:
        bio_by_problem[item["problem_text"]] = item

    # Generate extractor training pairs
    print("\nGenerating multi-span extractor training pairs...")
    training_pairs = []
    stats = Counter()

    for problem_text, bio_item in tqdm(bio_by_problem.items()):
        jsd_entry = jsd_data.get(problem_text)
        if not jsd_entry:
            stats["no_jsd_match"] += 1
            continue

        operations = jsd_entry.get("operations", [])
        if not operations:
            stats["no_operations"] += 1
            continue

        input_ids = bio_item.get("input_ids", [])
        bio_labels = bio_item.get("bio_labels", [])

        if not input_ids or not bio_labels:
            stats["no_bio_data"] += 1
            continue

        spans = extract_spans_from_bio(input_ids, bio_labels, tokenizer, problem_text)
        op_spans = [s for s in spans if s.get("tag") == "OP"]

        if not op_spans:
            stats["no_op_spans"] += 1
            continue

        # Track previous results for PREV classification
        prev_results = []

        for step_idx, step in enumerate(operations):
            # Get operands
            step_operands = set()
            arg1 = step.get("arg1")
            arg2 = step.get("arg2")
            result = step.get("result")

            if arg1 is not None:
                try:
                    step_operands.add(float(arg1))
                except (ValueError, TypeError):
                    arg1 = None
            if arg2 is not None:
                try:
                    step_operands.add(float(arg2))
                except (ValueError, TypeError):
                    arg2 = None

            if not step_operands:
                prev_results.append(result)
                stats["no_step_operands"] += 1
                continue

            # Find matching spans
            matching_indices = find_all_matching_spans(op_spans, step_operands)

            if not matching_indices:
                prev_results.append(result)
                stats["no_matching_spans"] += 1
                continue

            # Get operation type (field is op_type, values are lowercase)
            op_type = step.get("op_type", "").upper()
            if op_type not in OP_LABELS:
                prev_results.append(result)
                stats["unknown_op_type"] += 1
                continue

            # Get matching spans
            matching_spans = [op_spans[i] for i in matching_indices]
            span_texts = [s["text"] for s in matching_spans]

            # Mark spans
            marked = mark_multiple_spans(problem_text, matching_spans)
            if marked is None:
                prev_results.append(result)
                stats["marking_failed"] += 1
                continue

            # Classify argument sources
            arg1_source = "UNK"
            arg2_source = "UNK"

            if arg1 is not None:
                arg1 = float(arg1)
                arg1_source = classify_arg_source(
                    arg1, span_texts, problem_text, prev_results, step_idx
                )

            if arg2 is not None:
                arg2 = float(arg2)
                arg2_source = classify_arg_source(
                    arg2, span_texts, problem_text, prev_results, step_idx
                )

            # Build input/output
            input_text = f"[{op_type}]\n{marked}"
            output_text = format_output(arg1, arg1_source, arg2, arg2_source)

            training_pairs.append({
                "input": input_text,
                "output": output_text,
                "operation": op_type,
                "n_spans": len(matching_spans),
                "span_texts": span_texts,
                "arg1": arg1,
                "arg1_source": arg1_source,
                "arg2": arg2,
                "arg2_source": arg2_source,
                "problem_text": problem_text,
            })

            stats[f"spans_{len(matching_spans)}"] += 1
            stats[f"src1_{arg1_source.split(':')[0]}"] += 1
            if arg2 is not None:
                stats[f"src2_{arg2_source.split(':')[0]}"] += 1

            prev_results.append(result)

    print(f"\nGenerated {len(training_pairs)} training pairs")
    print("\nStatistics:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    # Source distribution
    print("\nArgument source distribution:")
    for src in ["PROB", "IMP", "PREV", "UNK"]:
        count1 = stats.get(f"src1_{src}", 0)
        count2 = stats.get(f"src2_{src}", 0)
        total = count1 + count2
        pct = 100 * total / (2 * len(training_pairs)) if training_pairs else 0
        print(f"  {src}: {total} ({pct:.1f}%)")

    # Save training data
    output_path = Path("data/extraction_multispan.json")
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
                print(f"Output: {sample['output']}")
                print(f"Input (first 150 chars):\n{sample['input'][:150]}...")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
