#!/usr/bin/env python3
"""
Convert v6 span data to training formats for segmenter, classifier, extractor.

Input: gsm8k_v6_spans_complete.json (from S3/GitHub release)
Output:
  - data/bio_segmentation_clean.json (for segmenter)
  - data/classification_multispan.json (for classifier)
  - data/extraction_multispan.json (for extractor)
"""

import json
import re
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

# Map v6 op_types to our labels
OP_MAP = {
    "add": "ADD",
    "subtract": "SUB",
    "multiply": "MUL",
    "divide": "DIV",
    "percentage": "MUL",  # percentage is often multiply by fraction
    "set": None,  # skip assignment operations
    "noop": None,  # skip no-ops
}


def load_v6_data(path: str):
    """Load v6 span data."""
    print(f"Loading {path}...")
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} problems")
    return data


def prepare_bio_segmentation(data: list, tokenizer):
    """
    Prepare BIO segmentation data.

    Format: input_ids, bio_labels where:
    - O = outside
    - B-OP = begin operation span
    - I-OP = inside operation span
    - B-Q = begin question span (unused for now)
    - I-Q = inside question span (unused for now)
    """
    LABELS = ["O", "B-OP", "I-OP", "B-Q", "I-Q"]
    label2id = {l: i for i, l in enumerate(LABELS)}

    examples = []

    for item in data:
        text = item["generated_text"]
        segment_ranges = item.get("segment_ranges", [])
        operations = item.get("operations", [])

        # Skip if no operations
        if not operations:
            continue

        # Get operation segments (non-noop)
        op_segments = []
        for seg_range, op in zip(segment_ranges, operations):
            if op.get("op_type") not in ["noop", "set", None]:
                op_segments.append(seg_range)

        if not op_segments:
            continue

        # Tokenize text
        encoding = tokenizer(text, truncation=True, max_length=256)
        input_ids = encoding["input_ids"]

        # Initialize all labels as O
        bio_labels = [label2id["O"]] * len(input_ids)

        # Mark operation segments
        for seg_start, seg_end in op_segments:
            # Convert char positions to token positions (approximate)
            # This is simplified - ideally use offset mapping
            char_to_token = {}
            current_char = 0
            for tok_idx, tok_id in enumerate(input_ids):
                tok_text = tokenizer.decode([tok_id])
                for _ in tok_text:
                    if current_char < len(text):
                        char_to_token[current_char] = tok_idx
                        current_char += 1

            # Find token range
            tok_start = char_to_token.get(seg_start, 0)
            tok_end = char_to_token.get(min(seg_end, len(text)-1), len(input_ids)-1)

            # Mark BIO labels
            if tok_start < len(bio_labels):
                bio_labels[tok_start] = label2id["B-OP"]
                for i in range(tok_start + 1, min(tok_end + 1, len(bio_labels))):
                    bio_labels[i] = label2id["I-OP"]

        n_op_spans = len(op_segments)
        if n_op_spans > 0:
            examples.append({
                "input_ids": input_ids,
                "bio_labels": bio_labels,
                "n_op_spans": n_op_spans,
                "text": text[:200],
            })

    return examples


def prepare_classification(data: list):
    """
    Prepare classification data with <SPAN> markers.

    Format: marked_problem (text with <SPAN>...</SPAN>), operation label
    """
    examples = []

    for item in data:
        text = item["generated_text"]
        question = item["question"]
        segment_ranges = item.get("segment_ranges", [])
        operations = item.get("operations", [])

        for seg_range, op in zip(segment_ranges, operations):
            op_type = op.get("op_type")
            mapped_op = OP_MAP.get(op_type)

            if mapped_op is None:
                continue

            seg_start, seg_end = seg_range
            seg_text = text[seg_start:seg_end]

            # Create marked problem: question + marked segment
            marked = f"{question}\n\n<SPAN>{seg_text}</SPAN>"

            examples.append({
                "marked_problem": marked,
                "operation": mapped_op,
                "n_spans": 1,
                "segment_text": seg_text,
            })

    return examples


def prepare_extraction(data: list):
    """
    Prepare extraction data.

    Format:
      input: [OP_LABEL]\n<problem with marked spans>
      output: arg1|source\narg2|source
    """
    examples = []

    for item in data:
        text = item["generated_text"]
        question = item["question"]
        segment_ranges = item.get("segment_ranges", [])
        operations = item.get("operations", [])

        for seg_range, op in zip(segment_ranges, operations):
            op_type = op.get("op_type")
            mapped_op = OP_MAP.get(op_type)

            if mapped_op is None:
                continue

            seg_start, seg_end = seg_range
            seg_text = text[seg_start:seg_end]

            # Extract arguments from operation
            arg1 = op.get("arg1")
            arg2 = op.get("arg2")
            result = op.get("result")

            # Determine sources
            arg1_source = "PROB" if arg1 is not None else None
            arg2_source = "PROB" if arg2 is not None else None

            # Check if arguments appear in segment text
            if arg1 is not None and str(int(arg1) if arg1 == int(arg1) else arg1) in seg_text:
                arg1_source = "SPAN"
            if arg2 is not None and str(int(arg2) if arg2 == int(arg2) else arg2) in seg_text:
                arg2_source = "SPAN"

            # Build output
            output_parts = []
            if arg1 is not None:
                val = int(arg1) if arg1 == int(arg1) else arg1
                output_parts.append(f"{val}|{arg1_source}")
            if arg2 is not None:
                val = int(arg2) if arg2 == int(arg2) else arg2
                output_parts.append(f"{val}|{arg2_source}")

            if not output_parts:
                continue

            input_text = f"[{mapped_op}]\n{question}\n\n<SPAN>{seg_text}</SPAN>"
            output_text = "\n".join(output_parts)

            examples.append({
                "input": input_text,
                "output": output_text,
                "n_spans": 1,
                "operation": mapped_op,
                "arg1": arg1,
                "arg2": arg2,
                "arg1_source": arg1_source,
                "arg2_source": arg2_source,
            })

    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/tmp/v6_data/gsm8k_v6_spans_complete.json")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    print("=" * 60)
    print("PREPARE TRAINING DATA FROM V6 SPANS")
    print("=" * 60)

    # Load v6 data
    data = load_v6_data(args.input)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for segmenter
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare segmentation data
    print("\n" + "-" * 40)
    print("Preparing BIO segmentation data...")
    bio_data = prepare_bio_segmentation(data, tokenizer)
    print(f"Generated {len(bio_data)} segmentation examples")

    bio_path = output_dir / "bio_segmentation_clean.json"
    with open(bio_path, "w") as f:
        json.dump(bio_data, f)
    print(f"Saved to {bio_path}")

    # Prepare classification data
    print("\n" + "-" * 40)
    print("Preparing classification data...")
    class_data = prepare_classification(data)
    print(f"Generated {len(class_data)} classification examples")

    # Show distribution
    op_counts = Counter(d["operation"] for d in class_data)
    print("Operation distribution:")
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count} ({100*count/len(class_data):.1f}%)")

    class_path = output_dir / "classification_multispan.json"
    with open(class_path, "w") as f:
        json.dump(class_data, f)
    print(f"Saved to {class_path}")

    # Prepare extraction data
    print("\n" + "-" * 40)
    print("Preparing extraction data...")
    extract_data = prepare_extraction(data)
    print(f"Generated {len(extract_data)} extraction examples")

    extract_path = output_dir / "extraction_multispan.json"
    with open(extract_path, "w") as f:
        json.dump(extract_data, f)
    print(f"Saved to {extract_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Segmentation: {len(bio_data)} examples -> {bio_path}")
    print(f"Classification: {len(class_data)} examples -> {class_path}")
    print(f"Extraction: {len(extract_data)} examples -> {extract_path}")


if __name__ == "__main__":
    main()
