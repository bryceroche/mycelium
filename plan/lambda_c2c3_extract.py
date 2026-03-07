"""
Lambda function: C2/C3 Per-Segment Training Label Extraction

KEY INSIGHT: C2 labels come from PROBLEM TEXT analysis, not CoT parsing.
The 14 coarse labels detect what operations are NEEDED to solve the problem,
not what the teacher explicitly wrote in reasoning.

For MATH dataset (algebraic reasoning), we detect patterns like:
- FACTORIAL: n! notation
- SQRT: \sqrt{}, square root
- TRIG: sin, cos, tan
- etc.

Input:  s3://mycelium-data/iaf_extraction/chunked/*.json
Output: s3://mycelium-data/c2c3_training_data/{chunk_id}.jsonl

Memory: 3GB Lambda
"""

import json
import re
import boto3
from typing import Optional

s3 = boto3.client("s3")

BUCKET = "mycelium-data"

# Window parameters matching C1-A
WINDOW_SIZE = 16
STRIDE = 8

# The 14 coarse C2 labels (from existing training data)
C2_LABELS = [
    "FACTORIAL", "LOG", "TRIG", "MOD", "SQRT", "CUBE", "FRAC_POW",
    "HIGH_POW", "SQUARE", "EQUATION", "DIV", "MUL", "ADD", "OTHER"
]


# ---------------------------------------------------------------------------
# C2 Label Detection from Problem Text
# ---------------------------------------------------------------------------

def detect_c2_labels(problem_text: str) -> list[str]:
    """
    Detect which of the 14 coarse C2 labels apply to this problem.
    Analyzes LaTeX patterns in problem text.
    Returns list of labels (multi-label per problem).
    """
    labels = set()

    # Normalize text
    text = problem_text.lower()

    # FACTORIAL: n!, factorial notation
    if re.search(r'\d+\s*!', problem_text) or 'factorial' in text:
        labels.add("FACTORIAL")
    if re.search(r'\\binom|\\choose|C\s*\(\s*\d+\s*,\s*\d+\s*\)|permutation|combination', problem_text, re.I):
        labels.add("FACTORIAL")  # Combinatorics uses factorials

    # LOG: logarithm
    if re.search(r'\\log|\\ln|log\s*\(|log_|logarithm', problem_text, re.I):
        labels.add("LOG")

    # TRIG: trigonometric functions
    if re.search(r'\\sin|\\cos|\\tan|\\cot|\\sec|\\csc|\\arcsin|\\arccos|\\arctan', problem_text):
        labels.add("TRIG")
    if re.search(r'\bsin\b|\bcos\b|\btan\b|trigonometr|degree|radian', text):
        labels.add("TRIG")

    # MOD: modular arithmetic
    if re.search(r'\\mod|\\pmod|\bmod\b|modulo|remainder|congruent', problem_text, re.I):
        labels.add("MOD")

    # SQRT: square roots
    if re.search(r'\\sqrt\{[^}]*\}|\\sqrt\s|square\s*root|√', problem_text, re.I):
        labels.add("SQRT")

    # CUBE: cubes and cube roots
    if re.search(r'\^\s*\{?\s*3\s*\}?|\^3\b|\\sqrt\[3\]|cube', problem_text, re.I):
        labels.add("CUBE")

    # FRAC_POW: fractional exponents
    if re.search(r'\^\s*\{?\s*\d+/\d+\s*\}?|\^\s*\{?\s*0\.\d+\s*\}?|\\sqrt\[\d+\]', problem_text):
        labels.add("FRAC_POW")

    # HIGH_POW: exponents >= 4 (excluding common squares/cubes)
    pow_matches = re.findall(r'\^\s*\{?\s*(\d+)\s*\}?', problem_text)
    for exp in pow_matches:
        try:
            if int(exp) >= 4:
                labels.add("HIGH_POW")
                break
        except ValueError:
            pass
    if re.search(r'polynomial.*degree\s*[4-9]|degree\s*[4-9].*polynomial', text):
        labels.add("HIGH_POW")

    # SQUARE: squares
    if re.search(r'\^\s*\{?\s*2\s*\}?|\^2\b|squared|square(?!\s*root)', problem_text, re.I):
        labels.add("SQUARE")

    # EQUATION: equations to solve
    if re.search(r'solve|find.*value|what.*is.*x|equation|=\s*0\b', text):
        labels.add("EQUATION")
    if re.search(r'[a-z]\s*=\s*[^,\s]', problem_text):  # Variable assignment
        labels.add("EQUATION")

    # DIV: division operations
    if re.search(r'\\frac\{|\bdivide\b|\bdivision\b|quotient|ratio', problem_text, re.I):
        labels.add("DIV")
    if re.search(r'\d+\s*/\s*\d+|÷', problem_text):
        labels.add("DIV")

    # MUL: multiplication
    if re.search(r'\\times|\\cdot|\bmultiply\b|\bproduct\b|\btimes\b', problem_text, re.I):
        labels.add("MUL")
    if re.search(r'\d+\s*[×*]\s*\d+', problem_text):
        labels.add("MUL")
    if re.search(r'\d+\s*\\\s*cdot\s*\d+', problem_text):
        labels.add("MUL")

    # ADD: addition (very common, only add if evidence)
    if re.search(r'\bsum\b|\btotal\b|\badd\b|\bplus\b|\bcombined\b', text):
        labels.add("ADD")
    if re.search(r'\d+\s*\+\s*\d+', problem_text):
        labels.add("ADD")
    # Most algebra involves ADD implicitly
    if re.search(r'\\[a-z]?\+', problem_text):
        labels.add("ADD")

    # Default: if no strong signals, check for basic arithmetic in equations
    if not labels:
        if '=' in problem_text:
            labels.add("EQUATION")
        if '+' in problem_text or '-' in problem_text:
            labels.add("ADD")
        if any(c in problem_text for c in ['×', '*', '\\cdot', '\\times']):
            labels.add("MUL")
        if '/' in problem_text or '\\frac' in problem_text:
            labels.add("DIV")

    # Still nothing? Use OTHER
    if not labels:
        labels.add("OTHER")

    return sorted(list(labels))


# ---------------------------------------------------------------------------
# LaTeX Preprocessing
# ---------------------------------------------------------------------------

def preprocess_latex(text: str) -> str:
    """Clean LaTeX for tokenization consistency."""
    # Remove display math delimiters
    text = re.sub(r'\\\[|\\\]', ' ', text)
    text = re.sub(r'\$\$', ' ', text)
    # Inline math: keep content
    text = re.sub(r'\$([^$]+)\$', r' \1 ', text)
    # Common LaTeX commands
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', text)
    text = re.sub(r'\\left|\\right', '', text)
    text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Operand Extraction from Problem Text (for C3)
# ---------------------------------------------------------------------------

def extract_operands_from_text(problem_text: str) -> list[dict]:
    """
    Extract numeric operands and their positions from problem text.
    For C3 training: what values appear where.
    """
    operands = []

    # Find all numbers (integers, decimals, fractions)
    # Match: 123, 3.14, 1/2, etc.
    patterns = [
        (r'(-?\d+\.\d+)', 'decimal'),  # Decimals
        (r'(-?\d+)', 'integer'),        # Integers
        (r'(\d+/\d+)', 'fraction'),     # Fractions like 1/2
    ]

    seen_positions = set()

    for pattern, num_type in patterns:
        for match in re.finditer(pattern, problem_text):
            # Avoid overlapping matches
            pos = (match.start(), match.end())
            overlaps = any(
                (pos[0] <= s and pos[1] >= e) or (pos[0] >= s and pos[0] < e)
                for s, e in seen_positions
            )
            if not overlaps:
                seen_positions.add(pos)
                operands.append({
                    "value": match.group(1),
                    "type": num_type,
                    "start_char": match.start(),
                    "end_char": match.end(),
                })

    # Also extract special constants
    special = [
        (r'\\pi\b', 'pi'),
        (r'\\sqrt\{(\d+)\}', 'sqrt_arg'),
    ]

    for pattern, const_type in special:
        for match in re.finditer(pattern, problem_text):
            operands.append({
                "value": match.group(0),
                "type": const_type,
                "start_char": match.start(),
                "end_char": match.end(),
            })

    return sorted(operands, key=lambda x: x["start_char"])


# ---------------------------------------------------------------------------
# Per-Segment Label Building
# ---------------------------------------------------------------------------

def build_per_segment_labels(
    problem_text: str,
    c2_labels: list[str],
    operands: list[dict],
    n_tokens: int,
    W: int = WINDOW_SIZE,
    S: int = STRIDE,
) -> list[dict]:
    """
    Build per-window segment labels.

    C2 approach: Labels apply at PROBLEM level, then we distribute
    operands to windows for C3 extraction.

    Windows without operands get NO_OP for C3 (no extraction target).
    """
    # Build window grid
    windows = []
    for w_start in range(0, n_tokens, S):
        w_end = min(w_start + W, n_tokens)
        if w_end - w_start < W // 2:
            break
        windows.append({"start": w_start, "end": w_end})

    if not windows:
        return []

    # Estimate char-to-token ratio
    text_len = max(1, len(problem_text))
    char_to_token = n_tokens / text_len

    # Map operands to windows
    operand_to_window = {}
    for op_idx, op in enumerate(operands):
        # Estimate token position of this operand
        start_tok = int(op["start_char"] * char_to_token)
        end_tok = int(op["end_char"] * char_to_token) + 1

        # Find best window
        best_window = None
        best_overlap = 0
        for w_idx, w in enumerate(windows):
            overlap = max(0, min(w["end"], end_tok) - max(w["start"], start_tok))
            if overlap > best_overlap:
                best_overlap = overlap
                best_window = w_idx

        if best_window is not None:
            if best_window not in operand_to_window:
                operand_to_window[best_window] = []
            operand_to_window[best_window].append({
                "value": op["value"],
                "type": op["type"],
                "start": start_tok - windows[best_window]["start"],  # Relative
                "end": end_tok - windows[best_window]["start"],
            })

    # Build segments
    segments = []
    for w_idx, w in enumerate(windows):
        ops_in_window = operand_to_window.get(w_idx, [])

        # C2 label: apply problem-level labels to windows with operands
        # Windows without operands are "NO_OP" from C3 perspective
        if len(ops_in_window) == 0:
            c2_label = "NO_OP"
        else:
            # Use first applicable label from problem
            c2_label = c2_labels[0] if c2_labels else "OTHER"

        segment = {
            "window_idx": w_idx,
            "window_start": w["start"],
            "window_end": w["end"],
            "c2_label": c2_label,
            "c2_problem_labels": c2_labels,  # Full set at problem level
            "c3_operands": ops_in_window,
            "n_operands": len(ops_in_window),
        }

        segments.append(segment)

    return segments


# ---------------------------------------------------------------------------
# Generate Labels for One Problem
# ---------------------------------------------------------------------------

def generate_c2c3_labels(problem: dict) -> Optional[dict]:
    """
    Generate C2/C3 training labels for one problem.
    Returns per-segment labels.
    """
    problem_text = problem.get("problem_text", problem.get("question", ""))
    cot_text = problem.get("generated_cot", "")
    problem_id = str(problem.get("problem_idx", problem.get("idx", "unknown")))
    n_tokens = problem.get("input_len", problem.get("num_tokens", 0))
    level = problem.get("level", "unknown")
    prob_type = problem.get("type", "unknown")

    if not problem_text:
        return None

    # Estimate token count if not provided
    if n_tokens == 0:
        n_tokens = len(problem_text) // 4 + 1

    # Detect C2 labels from problem text
    c2_labels = detect_c2_labels(problem_text)

    # Extract operands for C3
    operands = extract_operands_from_text(problem_text)

    # Build per-segment labels
    segments = build_per_segment_labels(
        problem_text, c2_labels, operands, n_tokens
    )

    # Stats
    n_no_op = sum(1 for seg in segments if seg["c2_label"] == "NO_OP")
    n_with_op = len(segments) - n_no_op

    return {
        "problem_id": problem_id,
        "problem_text": problem_text,
        "problem_text_preprocessed": preprocess_latex(problem_text),
        "n_input_tokens": n_tokens,
        "level": level,
        "type": prob_type,
        "c2_labels": c2_labels,
        "n_operands": len(operands),
        "n_windows": len(segments),
        "segments": segments,
        "stats": {
            "n_windows_with_ops": n_with_op,
            "n_windows_no_op": n_no_op,
            "n_operands": len(operands),
        },
        # Keep CoT for reference but don't parse it
        "cot_text": cot_text[:2000] if cot_text else "",
    }


# ---------------------------------------------------------------------------
# Lambda Handler
# ---------------------------------------------------------------------------

def lambda_handler(event, context):
    """
    Process one IAF chunk, extract C2/C3 training labels.
    """
    bucket = event.get("bucket", BUCKET)
    chunk_key = event["chunk_key"]
    output_prefix = event.get("output_prefix", "c2c3_training_data/")

    # Download IAF chunk
    print(f"Downloading s3://{bucket}/{chunk_key}")
    response = s3.get_object(Bucket=bucket, Key=chunk_key)
    chunk_data = json.loads(response["Body"].read().decode("utf-8"))
    problems = chunk_data if isinstance(chunk_data, list) else chunk_data.get("problems", [])

    # Process each problem
    records = []
    stats = {
        "total": 0,
        "generated": 0,
        "skipped": 0,
        "total_segments": 0,
        "no_op_segments": 0,
        "op_segments": 0,
        "label_dist": {},
        "total_operands": 0,
    }

    for idx, problem in enumerate(problems):
        stats["total"] += 1

        result = generate_c2c3_labels(problem)

        if result is None:
            stats["skipped"] += 1
            continue

        records.append(result)
        stats["generated"] += 1

        # Accumulate stats
        stats["total_segments"] += result["n_windows"]
        stats["no_op_segments"] += result["stats"]["n_windows_no_op"]
        stats["op_segments"] += result["stats"]["n_windows_with_ops"]
        stats["total_operands"] += result["n_operands"]

        # Label distribution (problem-level)
        for lbl in result["c2_labels"]:
            stats["label_dist"][lbl] = stats["label_dist"].get(lbl, 0) + 1

    # Compute rates
    total_segs = max(1, stats["total_segments"])
    stats["no_op_fraction"] = stats["no_op_segments"] / total_segs
    stats["op_fraction"] = stats["op_segments"] / total_segs

    # Upload records as JSONL
    chunk_name = chunk_key.split("/")[-1].replace(".json", ".jsonl")
    output_key = f"{output_prefix}{chunk_name}"

    jsonl_content = "\n".join(json.dumps(r, default=str) for r in records)

    print(
        f"Uploading s3://{bucket}/{output_key} | "
        f"{stats['generated']} problems | "
        f"{stats['total_segments']} segs | "
        f"NO_OP: {stats['no_op_fraction']:.1%}"
    )

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=jsonl_content.encode("utf-8"),
        ContentType="application/jsonl",
    )

    # Stats for reduce phase
    stats_key = f"{output_prefix}stats/{chunk_name.replace('.jsonl', '.json')}"
    s3.put_object(
        Bucket=bucket,
        Key=stats_key,
        Body=json.dumps(stats, default=str).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "chunk_key": chunk_key,
        "output_key": output_key,
        "stats": {
            "total": stats["total"],
            "generated": stats["generated"],
            "segments": stats["total_segments"],
            "no_op_fraction": stats["no_op_fraction"],
            "label_dist": stats["label_dist"],
        },
    }
