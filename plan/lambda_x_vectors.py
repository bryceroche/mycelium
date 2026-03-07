"""
Lambda function: Extract Per-Step X Vectors for IB Annealing

Aligns Sonnet-parsed reasoning steps with IAF attention data to create
(X, Y) training pairs for IB annealing.

X = 20-dim attention features (4 features × 5 heads)
Y = 7-dim structural labels from Sonnet

Input:
  - IAF chunk from s3://mycelium-data/iaf_extraction/chunked/
  - Sonnet steps from s3://mycelium-data/c2c3_training_data_v2/parsed_steps.jsonl

Output:
  - s3://mycelium-data/ib_ready/chunks/{chunk_name}.jsonl

Memory: 3GB
Timeout: 15 minutes
"""

import json
import math
import re
import boto3
from collections import defaultdict

s3 = boto3.client("s3")
BUCKET = "mycelium-data"

# Key attention heads for feature extraction
KEY_HEADS = ['L22H3', 'L22H4', 'L23H11', 'L23H23', 'L24H6']


def load_sonnet_steps_index(bucket: str) -> dict:
    """Load Sonnet parsed steps indexed by problem_id."""
    resp = s3.get_object(Bucket=bucket, Key="c2c3_training_data_v2/parsed_steps.jsonl")
    content = resp["Body"].read().decode("utf-8")

    index = defaultdict(list)
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        step = json.loads(line)
        problem_id = step.get("problem_id", "")
        index[problem_id].append(step)

    # Sort steps by step_idx within each problem
    for pid in index:
        index[pid].sort(key=lambda s: s.get("step_idx", 0))

    return dict(index)


def tokenize_simple(text: str) -> list:
    """Simple word-based tokenization for alignment.

    We don't need exact tokenizer match - we just need to find where
    raw_cot_text appears in generated_cot and map to approximate decode positions.
    """
    # Split on whitespace and punctuation, keeping tokens
    tokens = re.findall(r'\S+', text)
    return tokens


def normalize_latex_whitespace(text: str) -> str:
    """Normalize whitespace inside and around LaTeX delimiters for matching.

    Handles:
    - CoT uses \\[\\n...\\n\\] but Sonnet flattens to \\[ ... \\]
    - Whitespace around delimiters (: \\[ vs :\\n\\[)
    - $...$ vs \\(...\\) notation differences
    """
    import re

    # Convert $...$ to \(...\) for consistency (single $ only, not $$)
    # Be careful not to match $$ (display math)
    text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)',
                  lambda m: '\\(' + m.group(1) + '\\)',
                  text, flags=re.DOTALL)

    # Normalize \[...\] blocks: collapse internal whitespace
    def norm_display(m):
        inner = m.group(1)
        inner = ' '.join(inner.split())
        return '\\[' + inner + '\\]'

    text = re.sub(r'\\\[(.*?)\\\]', norm_display, text, flags=re.DOTALL)

    # Normalize \(...\) blocks
    def norm_inline(m):
        inner = m.group(1)
        inner = ' '.join(inner.split())
        return '\\(' + inner + '\\)'

    text = re.sub(r'\\\((.*?)\\\)', norm_inline, text, flags=re.DOTALL)

    # Collapse whitespace around LaTeX delimiters
    # ": \[" vs ":\n\[" -> ":\["
    text = re.sub(r'\s*\\\[', '\\[', text)
    text = re.sub(r'\\\]\s*', '\\]', text)
    text = re.sub(r'\s*\\\(', '\\(', text)
    text = re.sub(r'\\\)\s*', '\\)', text)

    # Collapse multiple spaces to single space
    text = re.sub(r' +', ' ', text)

    return text


def find_text_range(raw_cot_text: str, generated_cot: str) -> tuple:
    """Find character range where raw_cot_text appears in generated_cot.

    Returns (char_start, char_end) or (None, None) if not found.
    """
    clean_cot = generated_cot.strip()
    clean_step = raw_cot_text.strip()

    # Try exact match first
    char_start = clean_cot.find(clean_step)
    if char_start != -1:
        return char_start, char_start + len(clean_step), "exact"

    # Try with normalized LaTeX whitespace
    norm_cot = normalize_latex_whitespace(clean_cot)
    norm_step = normalize_latex_whitespace(clean_step)

    char_start = norm_cot.find(norm_step)
    if char_start != -1:
        # Map back to original position approximately
        return char_start, char_start + len(norm_step), "latex_norm"

    # Try fuzzy match with progressively shorter prefixes
    for trim in range(1, min(50, len(norm_step) // 2)):
        prefix = norm_step[:len(norm_step) - trim]
        if len(prefix) < 20:
            break
        char_start = norm_cot.find(prefix)
        if char_start != -1:
            return char_start, char_start + len(norm_step), "fuzzy"

    # Try matching first sentence only
    first_sentence = norm_step.split('.')[0] + '.'
    if len(first_sentence) > 20:
        char_start = norm_cot.find(first_sentence)
        if char_start != -1:
            return char_start, char_start + len(norm_step), "sentence"

    return None, None, "failed"


def char_to_token_range(char_start: int, char_end: int, generated_cot: str,
                        input_len: int, num_tokens: int) -> tuple:
    """Map character range to approximate decode token range.

    We estimate token positions based on character position ratio.
    """
    total_chars = len(generated_cot)
    decode_tokens = num_tokens - input_len

    if total_chars == 0 or decode_tokens <= 0:
        return None, None

    # Approximate: chars_per_token ≈ total_chars / decode_tokens
    chars_per_token = total_chars / decode_tokens

    tok_start = int(char_start / chars_per_token)
    tok_end = int(char_end / chars_per_token) + 1

    # Clamp to valid range
    tok_start = max(0, min(tok_start, decode_tokens - 1))
    tok_end = max(tok_start + 1, min(tok_end, decode_tokens))

    return tok_start, tok_end


def compute_entropy(weights: list) -> float:
    """Compute entropy from attention weights."""
    if not weights:
        return 0.0

    total = sum(w for w in weights if w > 0)
    if total < 1e-10:
        return 0.0

    entropy = 0.0
    for w in weights:
        if w > 0:
            p = w / total
            entropy -= p * math.log(p + 1e-10)

    return entropy


def extract_x_vector(iaf_traces: list, top_positions: list,
                     input_len: int, tok_start: int, tok_end: int) -> list:
    """Extract 20-dim X vector from attention data for one reasoning step.

    Features per head (4 features × 5 heads = 20):
    1. mean_reading: average reading signal across step
    2. reading_slope: change in reading signal from start to end
    3. mean_entropy: average attention entropy
    4. focus: total attention mass in top-k positions
    """
    features = []

    for head in KEY_HEADS:
        reading_signals = []
        entropies = []
        focus_values = []

        for decode_idx in range(tok_start, tok_end):
            trace_idx = input_len + decode_idx

            if trace_idx >= len(iaf_traces) or trace_idx >= len(top_positions):
                continue

            # Reading signal from iaf_traces
            trace = iaf_traces[trace_idx]
            if head in trace:
                reading_signals.append(float(trace[head]))

            # Entropy and focus from top_positions
            tp = top_positions[trace_idx]
            if head in tp:
                weights = [p.get("weight", 0.0) for p in tp[head]]
                entropies.append(compute_entropy(weights))
                focus_values.append(sum(weights))  # total mass in top-k

        # Compute features
        if reading_signals:
            mean_reading = sum(reading_signals) / len(reading_signals)
            if len(reading_signals) > 1:
                reading_slope = (reading_signals[-1] - reading_signals[0]) / len(reading_signals)
            else:
                reading_slope = 0.0
        else:
            mean_reading = 0.0
            reading_slope = 0.0

        mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        mean_focus = sum(focus_values) / len(focus_values) if focus_values else 0.0

        features.extend([mean_reading, reading_slope, mean_entropy, mean_focus])

    return features


def encode_y_vector(step: dict) -> list:
    """Encode 7-dim Y vector as integers for IB."""
    step_type_map = {
        "evaluate": 0, "simplify": 1, "substitute": 2, "solve_equation": 3,
        "factor": 4, "expand": 5, "apply_theorem": 6, "count": 7,
        "compare": 8, "convert": 9, "setup": 10, "other": 11
    }
    complexity_map = {"reduces": 0, "neutral": 1, "increases": 2}
    output_map = {"number": 0, "expression": 1, "equation": 2, "boolean": 3, "set": 4}
    position_map = {"early": 0, "middle": 1, "late": 2}
    distance_map = {"none": 0, "local": 1, "medium": 2, "distant": 3}

    y = [
        step_type_map.get(step.get("step_type", "other"), 11),
        complexity_map.get(step.get("complexity_change", "neutral"), 1),
        min(3, max(0, int(step.get("n_operands", 1)) - 1)),  # 1-4 -> 0-3
        1 if step.get("has_dependency", False) else 0,
        output_map.get(step.get("output_type", "expression"), 1),
        position_map.get(step.get("step_position", "middle"), 1),
        distance_map.get(step.get("reference_distance", "none"), 0),
    ]

    return y


def process_problem(problem: dict, sonnet_steps: list, chunk_name: str) -> list:
    """Process one problem, aligning Sonnet steps with IAF attention."""
    results = []

    problem_idx = problem.get("problem_idx")
    generated_cot = problem.get("generated_cot", "")
    iaf_traces = problem.get("iaf_traces", [])
    top_positions = problem.get("top_positions", [])
    input_len = problem.get("input_len", 0)
    num_tokens = problem.get("num_tokens", 0)

    if not generated_cot or not iaf_traces or not sonnet_steps:
        return results

    for step in sonnet_steps:
        raw_cot_text = step.get("raw_cot_text", "")
        if not raw_cot_text:
            continue

        # Find text range in generated CoT
        char_start, char_end, match_type = find_text_range(raw_cot_text, generated_cot)

        if char_start is None:
            results.append({
                "problem_id": step.get("problem_id"),
                "step_idx": step.get("step_idx"),
                "alignment_status": "failed",
                "match_type": match_type,
            })
            continue

        # Map to token range
        tok_start, tok_end = char_to_token_range(
            char_start, char_end, generated_cot, input_len, num_tokens
        )

        if tok_start is None:
            results.append({
                "problem_id": step.get("problem_id"),
                "step_idx": step.get("step_idx"),
                "alignment_status": "token_range_failed",
            })
            continue

        # Extract X vector
        x_vector = extract_x_vector(
            iaf_traces, top_positions, input_len, tok_start, tok_end
        )

        # Encode Y vector
        y_vector = encode_y_vector(step)

        results.append({
            "problem_id": step.get("problem_id"),
            "step_idx": step.get("step_idx"),
            "x_vector": x_vector,
            "y_vector": y_vector,
            "y_labels": {
                "step_type": step.get("step_type"),
                "complexity_change": step.get("complexity_change"),
                "n_operands": step.get("n_operands"),
                "has_dependency": step.get("has_dependency"),
                "output_type": step.get("output_type"),
                "step_position": step.get("step_position"),
                "reference_distance": step.get("reference_distance"),
            },
            "alignment": {
                "tok_start": tok_start,
                "tok_end": tok_end,
                "n_tokens": tok_end - tok_start,
                "match_type": match_type,
            },
            "alignment_status": "success",
        })

    return results


def lambda_handler(event, context):
    """Process one IAF chunk, extracting aligned X,Y vectors."""

    chunk_key = event["chunk_key"]
    chunk_index = event.get("chunk_index", 0)  # Position in sorted chunk order
    output_prefix = event.get("output_prefix", "ib_ready/chunks/")

    # Extract chunk name for output
    chunk_filename = chunk_key.split("/")[-1]

    # Load Sonnet steps index
    print("Loading Sonnet steps index...")
    sonnet_index = load_sonnet_steps_index(BUCKET)
    print(f"Loaded {len(sonnet_index)} problem IDs from Sonnet steps")

    # Load IAF chunk
    print(f"Loading IAF chunk: {chunk_key} (index={chunk_index})")
    resp = s3.get_object(Bucket=BUCKET, Key=chunk_key)
    chunk_data = json.loads(resp["Body"].read().decode("utf-8"))
    problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

    print(f"Processing {len(problems)} problems")

    # Process each problem
    all_results = []
    stats = {
        "total_problems": len(problems),
        "problems_with_steps": 0,
        "total_steps": 0,
        "aligned_steps": 0,
        "failed_steps": 0,
        "match_types": {"exact": 0, "fuzzy": 0, "sentence": 0, "failed": 0},
    }

    for problem in problems:
        problem_idx = problem.get("problem_idx")

        # problem_id format: chunk{index}_{problem_idx}
        # where index is the position in sorted chunk order
        pid = f"chunk{chunk_index}_{problem_idx}"
        sonnet_steps = sonnet_index.get(pid)

        if not sonnet_steps:
            continue

        stats["problems_with_steps"] += 1
        stats["total_steps"] += len(sonnet_steps)

        results = process_problem(problem, sonnet_steps, chunk_filename)

        for r in results:
            if r.get("alignment_status") == "success":
                stats["aligned_steps"] += 1
                match_type = r.get("alignment", {}).get("match_type", "unknown")
                stats["match_types"][match_type] = stats["match_types"].get(match_type, 0) + 1
            else:
                stats["failed_steps"] += 1
                stats["match_types"]["failed"] += 1

        all_results.extend(results)

    # Upload results
    output_key = f"{output_prefix}{chunk_filename.replace('.json', '.jsonl')}"

    jsonl_content = "\n".join(json.dumps(r) for r in all_results)
    s3.put_object(
        Bucket=BUCKET,
        Key=output_key,
        Body=jsonl_content.encode("utf-8"),
        ContentType="application/jsonl",
    )

    # Upload stats
    stats_key = f"{output_prefix}stats/{chunk_filename.replace('.json', '_stats.json')}"
    s3.put_object(
        Bucket=BUCKET,
        Key=stats_key,
        Body=json.dumps(stats, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    print(f"Results: {stats['aligned_steps']}/{stats['total_steps']} steps aligned")
    print(f"Match types: {stats['match_types']}")

    return {
        "statusCode": 200,
        "chunk_key": chunk_key,
        "output_key": output_key,
        "stats": stats,
    }
