"""
Lambda MapReduce: Extract operand training data from IAF attention patterns.

The teacher (Qwen-7B) attended to specific input tokens when generating each
CoT step. Those attended tokens ARE the operands. The IAF data captures this
mapping — we just need to extract it.

Actual IAF data format (from s3://mycelium-data/iaf_extraction/chunked/):
    - problem_text: str
    - input_tokens: list of str (tokenized input, ~100-150 tokens)
    - input_len: int
    - generated_cot: str (full CoT text)
    - num_tokens: int (generated tokens, ~200-900)
    - top_positions: list of dicts, ONE PER GENERATED TOKEN
      Each dict maps layer/head (e.g., "L22H4") to list of {pos, weight}
    - problem_idx: int

Architecture:
    Map Lambda (per chunk of ~50 problems):
        1. For each problem, segment generated_cot into steps
        2. For each step, aggregate attention from top_positions
        3. Extract most-attended input tokens → operands
        4. Build training examples

S3 paths:
    Input:  s3://mycelium-data/iaf_extraction/chunked/ (117 chunks, ~28GB)
    Output: s3://mycelium-data-v7/iaf_operands/

Lambda config:
    Memory: 3GB (NOT 1GB — chunks are ~200MB)
    Timeout: 300s

Usage (local testing):
    python iaf_operand_extraction.py --input-dir /tmp/iaf_sample/ --output /tmp/iaf_operands.jsonl
"""

import argparse
import json
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# CoT Step Segmentation
# ─────────────────────────────────────────────────────────────

def segment_cot_into_steps(generated_cot: str) -> List[Dict]:
    """
    Segment the generated CoT into logical steps.

    Looks for patterns like:
    - "Step 1:", "First,", "Next,"
    - Equations with "="
    - "Therefore", "Thus", "So"
    - Numbered lists "1.", "2."

    Returns list of {start_char, end_char, step_type, text}
    """
    steps = []

    # Split by common step markers
    patterns = [
        (r'(?:Step\s*\d+[:.]\s*)', 'COMPUTE'),
        (r'(?:First,?\s+)', 'SETUP'),
        (r'(?:Next,?\s+)', 'COMPUTE'),
        (r'(?:Then,?\s+)', 'COMPUTE'),
        (r'(?:Now,?\s+)', 'COMPUTE'),
        (r'(?:Finally,?\s+)', 'COMPUTE'),
        (r'(?:Therefore,?\s+)', 'SIMPLIFY'),
        (r'(?:Thus,?\s+)', 'SIMPLIFY'),
        (r'(?:So,?\s+)', 'SIMPLIFY'),
        (r'(?:Hence,?\s+)', 'SIMPLIFY'),
        (r'(?:We\s+(?:have|get|find|know|can|need)\s+)', 'COMPUTE'),
        (r'(?:Solving\s+)', 'SOLVE'),
        (r'(?:Substituting\s+)', 'SUBSTITUTE'),
        (r'(?:Expanding\s+)', 'EXPAND'),
        (r'(?:Simplifying\s+)', 'SIMPLIFY'),
    ]

    # Find all step boundaries
    boundaries = [(0, 'SETUP')]

    for pattern, step_type in patterns:
        for match in re.finditer(pattern, generated_cot, re.IGNORECASE):
            boundaries.append((match.start(), step_type))

    # Also split on double newlines
    for match in re.finditer(r'\n\n+', generated_cot):
        boundaries.append((match.start(), 'COMPUTE'))

    # Sort and deduplicate
    boundaries = sorted(set(boundaries), key=lambda x: x[0])

    # Build steps
    for i, (start, step_type) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(generated_cot)
        text = generated_cot[start:end].strip()

        if len(text) > 10:  # Skip very short segments
            steps.append({
                'start_char': start,
                'end_char': end,
                'step_type': step_type,
                'text': text[:200],  # Truncate for storage
            })

    # If no steps found, treat entire CoT as one step
    if not steps:
        steps = [{
            'start_char': 0,
            'end_char': len(generated_cot),
            'step_type': 'COMPUTE',
            'text': generated_cot[:200],
        }]

    return steps


def estimate_token_positions(generated_cot: str, num_tokens: int,
                             char_start: int, char_end: int) -> Tuple[int, int]:
    """
    Estimate which token positions correspond to a character range.

    Assumes roughly uniform characters per token (~4 chars/token for English).
    """
    if num_tokens == 0 or len(generated_cot) == 0:
        return 0, 0

    chars_per_token = len(generated_cot) / num_tokens

    token_start = int(char_start / chars_per_token)
    token_end = int(char_end / chars_per_token)

    return max(0, token_start), min(num_tokens, token_end)


# ─────────────────────────────────────────────────────────────
# Attention Aggregation
# ─────────────────────────────────────────────────────────────

def aggregate_attention_for_step(top_positions_list: List[Dict],
                                  token_start: int, token_end: int,
                                  input_len: int, top_k: int = 10) -> List[Tuple[int, float]]:
    """
    Aggregate attention from generation positions [token_start, token_end).

    top_positions_list[i] is the attention at generation step i.
    Each entry maps layer/head → list of {pos, weight}.

    Returns list of (input_position, aggregated_weight) for top-k positions.
    """
    # Aggregate attention across all generation steps in this range
    position_weights = defaultdict(float)

    for gen_idx in range(token_start, min(token_end, len(top_positions_list))):
        step_attention = top_positions_list[gen_idx]

        if not isinstance(step_attention, dict):
            continue

        # Average across layer/heads (use a few key heads known to be important)
        # L22H4 and L22H3 tend to have good attention patterns
        for layer_head in ['L22H4', 'L22H3', 'L14H0', 'L24H6']:
            if layer_head not in step_attention:
                continue

            for entry in step_attention[layer_head]:
                if isinstance(entry, dict):
                    pos = entry.get('pos', -1)
                    weight = entry.get('weight', 0)

                    # Only count input positions (not generated tokens attending to themselves)
                    if 0 <= pos < input_len:
                        position_weights[pos] += weight

    # Get top-k positions by aggregated weight
    sorted_positions = sorted(position_weights.items(), key=lambda x: -x[1])
    return sorted_positions[:top_k]


# ─────────────────────────────────────────────────────────────
# Operand Extraction
# ─────────────────────────────────────────────────────────────

def extract_operands_from_iaf(problem_data: dict, top_k: int = 10) -> List[dict]:
    """
    Extract operand mappings from a single problem's IAF data.
    """
    problem_text = problem_data.get("problem_text", "")
    input_tokens = problem_data.get("input_tokens", [])
    input_len = problem_data.get("input_len", len(input_tokens))
    generated_cot = problem_data.get("generated_cot", "")
    num_tokens = problem_data.get("num_tokens", 0)
    top_positions_list = problem_data.get("top_positions", [])
    problem_id = problem_data.get("problem_idx", "unknown")

    if not input_tokens or not top_positions_list:
        return []

    # Segment CoT into steps
    steps = segment_cot_into_steps(generated_cot)

    results = []

    for step_idx, step in enumerate(steps):
        # Estimate which generation tokens correspond to this step
        token_start, token_end = estimate_token_positions(
            generated_cot, num_tokens,
            step['start_char'], step['end_char']
        )

        if token_end <= token_start:
            continue

        # Aggregate attention for this step
        attended = aggregate_attention_for_step(
            top_positions_list, token_start, token_end,
            input_len, top_k
        )

        if not attended:
            continue

        # Group contiguous attended positions into spans
        spans = group_into_spans(attended, max_gap=2)

        # Extract text at each span
        operand_texts = []
        operand_details = []

        for span_start, span_end, span_weight in spans:
            # Get tokens in this span
            span_tokens = input_tokens[span_start:span_end + 1]
            span_text = clean_token_text(" ".join(span_tokens))

            if span_text.strip():
                operand_texts.append(span_text.strip())
                operand_details.append({
                    "text": span_text.strip(),
                    "start": span_start,
                    "end": span_end,
                    "weight": span_weight,
                })

        if not operand_texts:
            continue

        operand_string = " ".join(operand_texts)

        results.append({
            "problem_id": problem_id,
            "step_idx": step_idx,
            "step_type": step['step_type'],
            "problem_text": problem_text,
            "operand_string": operand_string,
            "operand_details": operand_details,
            "n_operands": len(operand_texts),
            "total_attention": sum(w for _, w in attended),
            "cot_text": step['text'],
            "source": "iaf_attention",
        })

    return results


def group_into_spans(positions: List[Tuple[int, float]],
                     max_gap: int = 2) -> List[Tuple[int, int, float]]:
    """Group contiguous attended positions into spans."""
    if not positions:
        return []

    positions = sorted(positions, key=lambda x: x[0])

    spans = []
    span_start = positions[0][0]
    span_end = positions[0][0]
    span_weights = [positions[0][1]]

    for pos, weight in positions[1:]:
        if pos <= span_end + max_gap:
            span_end = pos
            span_weights.append(weight)
        else:
            avg_weight = sum(span_weights) / len(span_weights)
            spans.append((span_start, span_end, avg_weight))
            span_start = pos
            span_end = pos
            span_weights = [weight]

    avg_weight = sum(span_weights) / len(span_weights)
    spans.append((span_start, span_end, avg_weight))

    return spans


def clean_token_text(text: str) -> str:
    """Clean up tokenizer artifacts."""
    text = text.replace("Ġ", " ")   # GPT-2 style
    text = text.replace("▁", " ")   # sentencepiece
    text = text.replace("##", "")   # BERT wordpiece
    text = text.replace("\u010a", "\n")  # newline token
    text = text.replace("\u0120", " ")   # space token
    return " ".join(text.split())


# ─────────────────────────────────────────────────────────────
# Build Training Data
# ─────────────────────────────────────────────────────────────

def build_training_data(operands: List[dict]) -> List[dict]:
    """Convert IAF-extracted operands into LoRA C training format."""
    by_problem = defaultdict(list)
    for o in operands:
        by_problem[o["problem_id"]].append(o)

    training = []

    for pid, problem_operands in by_problem.items():
        problem_operands.sort(key=lambda x: x["step_idx"])

        previous_results = []

        for op in problem_operands:
            step_idx = op["step_idx"]
            step_type = op["step_type"]
            total_steps = len(problem_operands)
            problem_text = op["problem_text"]

            prev_str = ""
            if previous_results:
                prev_str = f"Previous results: {', '.join(previous_results[-3:])}\n"

            prompt = (
                f"Problem: {problem_text}\n"
                f"Step: {step_type} (step {step_idx + 1} of {total_steps})\n"
                f"{prev_str}"
                f"What values does the teacher attend to for this step?\n"
                f"Operands:"
            )

            training.append({
                "problem_id": pid,
                "step_idx": step_idx,
                "scaffold_type": step_type,
                "prompt": prompt,
                "target": f" {op['operand_string']}\n",
                "operand_details": op["operand_details"],
                "source": "iaf_attention",
            })

            if op.get("operand_string"):
                previous_results.append(op["operand_string"][:50])

    return training


# ─────────────────────────────────────────────────────────────
# Map/Reduce Functions
# ─────────────────────────────────────────────────────────────

def map_chunk(chunk_data: List[dict], top_k: int = 10) -> List[dict]:
    """Process a chunk of problems, extract operands."""
    all_operands = []

    for problem_data in chunk_data:
        try:
            operands = extract_operands_from_iaf(problem_data, top_k=top_k)
            all_operands.extend(operands)
        except Exception as e:
            problem_id = problem_data.get("problem_idx", "unknown")
            print(f"  Error processing {problem_id}: {e}")

    return all_operands


def reduce_results(all_operands: List[dict]) -> dict:
    """Aggregate statistics."""
    stats = {
        "total_examples": len(all_operands),
        "unique_problems": len(set(o["problem_id"] for o in all_operands)),
        "by_step_type": defaultdict(int),
        "avg_operands_per_step": 0,
    }

    for o in all_operands:
        stats["by_step_type"][o["step_type"]] += 1

    if all_operands:
        stats["avg_operands_per_step"] = (
            sum(o["n_operands"] for o in all_operands) / len(all_operands)
        )

    stats["by_step_type"] = dict(stats["by_step_type"])
    return stats


# ─────────────────────────────────────────────────────────────
# Lambda Handler
# ─────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """AWS Lambda handler for MapReduce processing."""
    import boto3

    s3 = boto3.client('s3')

    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    print(f"Processing s3://{bucket}/{key}")

    response = s3.get_object(Bucket=bucket, Key=key)
    body = response['Body'].read().decode('utf-8')

    chunk = json.loads(body)
    if not isinstance(chunk, list):
        chunk = [chunk]

    operands = map_chunk(chunk, top_k=10)
    training = build_training_data(operands)

    output_key = key.replace("iaf_extraction/chunked/", "iaf_operands/")
    output_body = "\n".join(json.dumps(item) for item in training)

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=output_body.encode('utf-8'),
    )

    print(f"Wrote {len(training)} examples to s3://{bucket}/{output_key}")

    return {
        "statusCode": 200,
        "processed": len(chunk),
        "operands_extracted": len(operands),
        "training_examples": len(training),
    }


# ─────────────────────────────────────────────────────────────
# Local Processing
# ─────────────────────────────────────────────────────────────

def process_local(input_path: str, output_path: str, top_k: int = 10):
    """Process IAF files locally for testing."""
    import glob
    import os

    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.json*"))
    else:
        files = [input_path]

    print(f"Processing {len(files)} files")

    all_operands = []

    for filepath in files:
        print(f"  Processing {os.path.basename(filepath)}...")

        with open(filepath) as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    chunk = data
                else:
                    chunk = [data]
            except json.JSONDecodeError:
                f.seek(0)
                chunk = [json.loads(line) for line in f if line.strip()]

        operands = map_chunk(chunk, top_k)
        all_operands.extend(operands)
        print(f"    Extracted {len(operands)} operand examples")

    stats = reduce_results(all_operands)
    print(f"\n=== Results ===")
    print(f"Total operand examples: {stats['total_examples']}")
    print(f"Unique problems: {stats['unique_problems']}")
    print(f"Avg operands per step: {stats['avg_operands_per_step']:.1f}")
    print(f"\nBy step type:")
    for stype, count in sorted(stats["by_step_type"].items(), key=lambda x: -x[1]):
        print(f"  {stype}: {count}")

    training = build_training_data(all_operands)
    print(f"\nTraining examples: {len(training)}")

    with open(output_path, "w") as f:
        for item in all_operands:
            f.write(json.dumps(item) + "\n")

    training_path = output_path.replace(".jsonl", "_training.jsonl")
    with open(training_path, "w") as f:
        for item in training:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved operands: {output_path}")
    print(f"Saved training: {training_path}")

    # Show examples
    print(f"\n=== Examples ===")
    for ex in all_operands[:3]:
        print(f"  Problem: {ex['problem_text'][:60]}...")
        print(f"  Step {ex['step_idx']} ({ex['step_type']}): {ex['operand_string']}")
        print()

    return all_operands, training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="iaf_operands.jsonl")
    parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()
    process_local(args.input_dir, args.output, args.top_k)


if __name__ == "__main__":
    main()
