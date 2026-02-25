#!/usr/bin/env python3
"""
Prepare training data from INPUT attention JSD data.

This is the fixed version that uses boundaries in QUESTION TEXT (not CoT).

Input: Directory of problem_*.json files with:
  - question: the problem text
  - boundaries_char: character positions of boundaries IN the question
  - generated_cot: for parsing operations
  - gold_answer: for validation

Output:
  - data/bio_segmentation_v2.json (for segmenter)
  - data/classification_v2.json (for classifier)
  - data/extraction_v2.json (for extractor)
"""

import json
import re
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
import argparse


# Map parsed operation types to labels
OP_MAP = {
    "+": "ADD",
    "-": "SUB",
    "*": "MUL",
    "×": "MUL",
    "/": "DIV",
    "÷": "DIV",
}


def parse_operations_from_cot(cot_text: str):
    """
    Parse arithmetic operations from CoT text.

    Returns list of dicts with:
      - op_type: ADD, SUB, MUL, DIV
      - arg1, arg2: numeric operands
      - result: result value
      - text_match: the matched equation text
    """
    operations = []

    # Pattern: number op number = number
    # Handles: 48 + 24 = 72, 16 - 3 = 13, 5 × 2 = 10, 100 / 4 = 25
    pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'

    for match in re.finditer(pattern, cot_text):
        arg1 = float(match.group(1))
        op_symbol = match.group(2)
        arg2 = float(match.group(3))
        result = float(match.group(4))

        op_type = OP_MAP.get(op_symbol)
        if op_type:
            operations.append({
                "op_type": op_type,
                "arg1": arg1,
                "arg2": arg2,
                "result": result,
                "text_match": match.group(0),
            })

    return operations


def find_numbers_in_text(text: str):
    """Find all numbers and their positions in text."""
    numbers = []
    for match in re.finditer(r'\d+(?:\.\d+)?', text):
        numbers.append({
            "value": float(match.group()),
            "start": match.start(),
            "end": match.end(),
            "text": match.group(),
        })
    return numbers


def create_spans_from_boundaries(question: str, boundaries_char: list, min_span_len: int = 5):
    """
    Create spans from boundary positions.

    Boundaries split the question into segments. Each segment between
    consecutive boundaries (or start/end) is a potential span.
    """
    # Sort and deduplicate boundaries
    boundaries = sorted(set([0] + boundaries_char + [len(question)]))

    spans = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        text = question[start:end].strip()
        if len(text) >= min_span_len:
            spans.append({
                "start": start,
                "end": end,
                "text": text,
            })

    return spans


def assign_operations_to_spans(spans: list, operations: list, question: str):
    """
    Assign operations to spans based on which numbers appear in each span.

    Returns spans with operation labels.
    """
    labeled_spans = []

    # Find numbers in question
    question_numbers = find_numbers_in_text(question)

    for span in spans:
        # Find which numbers are in this span
        span_numbers = []
        for num in question_numbers:
            if span["start"] <= num["start"] and num["end"] <= span["end"]:
                span_numbers.append(num["value"])

        if len(span_numbers) < 2:
            # Need at least 2 numbers for a binary operation
            continue

        # Find matching operation
        best_op = None
        best_score = 0

        for op in operations:
            # Check if this span's numbers match the operation's operands
            if op["arg1"] in span_numbers and op["arg2"] in span_numbers:
                # This span likely corresponds to this operation
                score = 2  # Both operands present
                if op["result"] in span_numbers:
                    score += 1  # Result also present
                if score > best_score:
                    best_score = score
                    best_op = op

        if best_op:
            labeled_spans.append({
                **span,
                "operation": best_op["op_type"],
                "arg1": best_op["arg1"],
                "arg2": best_op["arg2"],
                "result": best_op["result"],
            })

    return labeled_spans


def prepare_bio_segmentation(data: list, tokenizer):
    """
    Prepare BIO segmentation data.

    Uses boundaries in QUESTION TEXT to create labels.
    """
    LABELS = ["O", "B-OP", "I-OP"]
    label2id = {l: i for i, l in enumerate(LABELS)}

    examples = []

    for item in data:
        question = item["question"]
        boundaries_char = item.get("boundaries_char", [])

        if not boundaries_char:
            continue

        # Create spans from boundaries
        spans = create_spans_from_boundaries(question, boundaries_char)

        if not spans:
            continue

        # Tokenize question (not CoT!)
        encoding = tokenizer(
            question,
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"]
        offset_mapping = encoding.get("offset_mapping", [])

        # Initialize all labels as O
        bio_labels = [label2id["O"]] * len(input_ids)

        # Map spans to token positions
        for span in spans:
            span_start = span["start"]
            span_end = span["end"]

            # Find token range for this span
            for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
                if char_start >= span_start and char_end <= span_end:
                    # This token is within the span
                    if bio_labels[tok_idx] == label2id["O"]:
                        # First token of span
                        if tok_idx == 0 or bio_labels[tok_idx - 1] == label2id["O"]:
                            bio_labels[tok_idx] = label2id["B-OP"]
                        else:
                            bio_labels[tok_idx] = label2id["I-OP"]

        # Count OP tokens
        n_op_tokens = sum(1 for l in bio_labels if l != label2id["O"])

        if n_op_tokens > 0:
            examples.append({
                "input_ids": input_ids,
                "bio_labels": bio_labels,
                "n_spans": len(spans),
                "text": question[:200],
            })

    return examples


def prepare_classification(data: list):
    """
    Prepare classification data with spans marked in QUESTION TEXT.
    """
    examples = []

    for item in data:
        question = item["question"]
        boundaries_char = item.get("boundaries_char", [])
        cot = item.get("generated_cot", "")

        if not boundaries_char:
            continue

        # Parse operations from CoT
        operations = parse_operations_from_cot(cot)

        if not operations:
            continue

        # Create spans from boundaries
        spans = create_spans_from_boundaries(question, boundaries_char)

        # Assign operations to spans
        labeled_spans = assign_operations_to_spans(spans, operations, question)

        for span in labeled_spans:
            # Create marked question
            marked = question[:span["start"]] + "<SPAN>" + span["text"] + "</SPAN>" + question[span["end"]:]

            examples.append({
                "marked_problem": marked,
                "operation": span["operation"],
                "span_text": span["text"],
                "arg1": span["arg1"],
                "arg2": span["arg2"],
            })

    return examples


def prepare_extraction(data: list):
    """
    Prepare extraction data with spans marked in QUESTION TEXT.
    """
    examples = []

    for item in data:
        question = item["question"]
        boundaries_char = item.get("boundaries_char", [])
        cot = item.get("generated_cot", "")

        if not boundaries_char:
            continue

        # Parse operations from CoT
        operations = parse_operations_from_cot(cot)

        if not operations:
            continue

        # Create spans from boundaries
        spans = create_spans_from_boundaries(question, boundaries_char)

        # Assign operations to spans
        labeled_spans = assign_operations_to_spans(spans, operations, question)

        for span in labeled_spans:
            op_type = span["operation"]
            arg1 = span["arg1"]
            arg2 = span["arg2"]

            # Determine sources
            arg1_source = "SPAN" if str(int(arg1) if arg1 == int(arg1) else arg1) in span["text"] else "PROB"
            arg2_source = "SPAN" if str(int(arg2) if arg2 == int(arg2) else arg2) in span["text"] else "PROB"

            # Build input/output
            marked = question[:span["start"]] + "<SPAN>" + span["text"] + "</SPAN>" + question[span["end"]:]
            input_text = f"[{op_type}]\n{marked}"

            output_parts = []
            val1 = int(arg1) if arg1 == int(arg1) else arg1
            val2 = int(arg2) if arg2 == int(arg2) else arg2
            output_parts.append(f"{val1}|{arg1_source}")
            output_parts.append(f"{val2}|{arg2_source}")
            output_text = "\n".join(output_parts)

            examples.append({
                "input": input_text,
                "output": output_text,
                "operation": op_type,
                "arg1": arg1,
                "arg2": arg2,
                "arg1_source": arg1_source,
                "arg2_source": arg2_source,
            })

    return examples


def load_jsd_data(input_dir: str):
    """Load all problem_*.json files from input directory."""
    input_path = Path(input_dir)
    data = []

    for f in sorted(input_path.glob("problem_*.json")):
        try:
            with open(f) as fp:
                item = json.load(fp)
                # Only include correct predictions with boundaries
                if item.get("is_correct") and item.get("boundaries_char"):
                    data.append(item)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with problem_*.json files")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    print("=" * 60)
    print("PREPARE TRAINING DATA FROM INPUT ATTENTION JSD")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {args.input_dir}...")
    data = load_jsd_data(args.input_dir)
    print(f"Loaded {len(data)} correct problems with boundaries")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare segmentation data
    print("\n" + "-" * 40)
    print("Preparing BIO segmentation data...")
    bio_data = prepare_bio_segmentation(data, tokenizer)
    print(f"Generated {len(bio_data)} segmentation examples")

    bio_path = output_dir / "bio_segmentation_v2.json"
    with open(bio_path, "w") as f:
        json.dump(bio_data, f)
    print(f"Saved to {bio_path}")

    # Prepare classification data
    print("\n" + "-" * 40)
    print("Preparing classification data...")
    class_data = prepare_classification(data)
    print(f"Generated {len(class_data)} classification examples")

    if class_data:
        op_counts = Counter(d["operation"] for d in class_data)
        print("Operation distribution:")
        for op, count in sorted(op_counts.items()):
            print(f"  {op}: {count} ({100*count/len(class_data):.1f}%)")

    class_path = output_dir / "classification_v2.json"
    with open(class_path, "w") as f:
        json.dump(class_data, f)
    print(f"Saved to {class_path}")

    # Prepare extraction data
    print("\n" + "-" * 40)
    print("Preparing extraction data...")
    extract_data = prepare_extraction(data)
    print(f"Generated {len(extract_data)} extraction examples")

    extract_path = output_dir / "extraction_v2.json"
    with open(extract_path, "w") as f:
        json.dump(extract_data, f)
    print(f"Saved to {extract_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input problems: {len(data)}")
    print(f"Segmentation: {len(bio_data)} examples -> {bio_path}")
    print(f"Classification: {len(class_data)} examples -> {class_path}")
    print(f"Extraction: {len(extract_data)} examples -> {extract_path}")


if __name__ == "__main__":
    main()
