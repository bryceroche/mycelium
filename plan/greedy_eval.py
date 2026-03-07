"""
Greedy End-to-End Pipeline Evaluation

Runs C1-A → C3 → heuristic op inference → assembly → SymPy execution
WITHOUT the factor graph, to establish baseline and identify failure modes.

Usage:
    python greedy_eval.py  # Run on GPU (g5.xlarge)
"""

import json
import re
import io
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
import torch
import torch.nn as nn
import boto3
from botocore.config import Config
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

config = Config(read_timeout=120, connect_timeout=30, retries={'max_attempts': 3})
s3 = boto3.client("s3", config=config)
BUCKET = "mycelium-data"

# Window parameters
W = 16
S = 8

# C3 model parameters
HIDDEN_DIM = 896
MAX_OPERANDS = 4


def preprocess_latex(text: str) -> str:
    """Apply LaTeX preprocessing."""
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\begin{array}", "").replace("\\end{array}", "")
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =============================================================================
# Model Loading
# =============================================================================

def load_c1a_model(device):
    """Load C1-A model with LoRA adapter."""
    print("Loading C1-A model...")

    adapter_local = Path("/tmp/c1a_adapter")
    adapter_local.mkdir(exist_ok=True)

    # Download adapter
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="models/c1a_coarse_v6_aux_telegraph/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]
            if filename:
                s3.download_file(BUCKET, key, str(adapter_local / filename))

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        "Qwen/Qwen2-0.5B",
        trust_remote_code=True,
        torch_dtype=torch.float32
    )

    model = PeftModel.from_pretrained(base_model, str(adapter_local))
    model = model.to(device)
    model.eval()

    # Load classification head
    head_path = adapter_local / "head_weights.pt"
    head_weights = torch.load(head_path, map_location=device)

    return model, tokenizer, head_weights


N_C2_CLASSES = 26  # For C3's c2_embedding


class C3Model(nn.Module):
    """C3 operand extractor (matches train_c2c3.py architecture)."""
    def __init__(self, hidden_dim=HIDDEN_DIM, c2_embed_dim=32,
                 n_c2_classes=N_C2_CLASSES, max_operands=MAX_OPERANDS, window_size=W):
        super().__init__()
        self.c2_embedding = nn.Embedding(n_c2_classes, c2_embed_dim)
        self.max_operands = max_operands
        self.window_size = window_size

        # Per-operand position scorers (W+1 positions: W real + 1 BACKREF)
        self.operand_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + c2_embed_dim, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, window_size + 1)
            )
            for _ in range(max_operands)
        ])

        # Operand count head
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim + c2_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, max_operands + 1)
        )

    def forward(self, window_features, c2_label):
        """
        window_features: (batch, hidden_dim) - pooled
        c2_label: (batch,) int - C2 predicted label (use dummy if unavailable)
        """
        c2_embed = self.c2_embedding(c2_label)  # (batch, c2_embed_dim)
        pooled_with_c2 = torch.cat([window_features, c2_embed], dim=-1)

        count_logits = self.count_head(pooled_with_c2)

        operand_logits = []
        for scorer in self.operand_scorers:
            logits = scorer(pooled_with_c2)
            operand_logits.append(logits)

        return count_logits, operand_logits


def load_c3_model(device):
    """Load C3 model from S3."""
    print("Loading C3 model...")

    c3_local = Path("/tmp/c3_model")
    c3_local.mkdir(exist_ok=True)

    s3.download_file(BUCKET, "models/c3_v1/best_checkpoint.pt", str(c3_local / "checkpoint.pt"))
    s3.download_file(BUCKET, "models/c3_v1/config.json", str(c3_local / "config.json"))

    with open(c3_local / "config.json") as f:
        cfg = json.load(f)

    model = C3Model(
        hidden_dim=cfg.get("hidden_dim", HIDDEN_DIM),
        max_operands=cfg.get("max_operands", MAX_OPERANDS),
        window_size=cfg.get("window_size", W)
    )

    # Checkpoint is raw state_dict (not wrapped)
    checkpoint = torch.load(c3_local / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    return model


# =============================================================================
# Pipeline Components
# =============================================================================

def run_c1a_segmentation(model, tokenizer, head_weights, problem_text, device):
    """Run C1-A to get boundary predictions per window."""
    text = preprocess_latex(problem_text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # (1, seq_len, 896)

    seq_len = hidden_states.shape[1]

    # Extract per-window features and predict
    windows = []
    window_features = []

    for w_start in range(0, seq_len, S):
        w_end = min(w_start + W, seq_len)
        if w_end - w_start < W // 2:
            break

        window_feats = hidden_states[0, w_start:w_end, :].mean(dim=0)  # (896,)
        window_features.append(window_feats)

        # Get tokens for this window
        token_ids = inputs["input_ids"][0, w_start:w_end].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        window_text = tokenizer.decode(token_ids, skip_special_tokens=True)

        windows.append({
            "start": w_start,
            "end": w_end,
            "tokens": tokens,
            "text": window_text
        })

    if not window_features:
        return [], []

    # Apply classification head
    features = torch.stack(window_features)  # (n_windows, 896)

    # Simple linear head: features @ W + b
    if "classifier.weight" in head_weights:
        W_cls = head_weights["classifier.weight"].to(device)
        b_cls = head_weights["classifier.bias"].to(device)
        logits = features @ W_cls.T + b_cls
    else:
        # Fallback: use first available weight
        for k, v in head_weights.items():
            if "weight" in k and v.shape[1] == 896:
                logits = features @ v.T
                break
        else:
            # No head found, use threshold on feature norm
            logits = features.norm(dim=1, keepdim=True)

    probs = torch.sigmoid(logits[:, 0] if logits.dim() > 1 else logits)

    # Positive windows
    positive_mask = probs > 0.5
    positive_indices = positive_mask.nonzero().squeeze(-1).tolist()
    if isinstance(positive_indices, int):
        positive_indices = [positive_indices]

    positive_windows = [windows[i] for i in positive_indices]
    positive_features = [window_features[i] for i in positive_indices]

    return positive_windows, positive_features


def run_c3_extraction(c3_model, window_features, device):
    """Run C3 to extract operands for each window."""
    if not window_features:
        return []

    features = torch.stack(window_features).to(device)  # (n_windows, 896)

    # Use dummy C2 labels (0 = NO_OP placeholder since we don't have C2)
    # This is fine for greedy eval - C3 should still predict operand counts
    dummy_c2_labels = torch.zeros(len(window_features), dtype=torch.long, device=device)

    with torch.no_grad():
        count_logits, position_logits = c3_model(features, dummy_c2_labels)

    # Predict counts
    counts = count_logits.argmax(dim=1).tolist()

    # Extract operand positions
    extractions = []
    for i, count in enumerate(counts):
        operands = []
        for op_idx in range(min(count, MAX_OPERANDS)):
            pos_logits = position_logits[op_idx][i]
            pos = pos_logits.argmax().item()

            is_backref = (pos == W)  # Position W is BACKREF token
            operands.append({
                "position": pos if pos < W else -1,
                "is_backref": is_backref
            })

        extractions.append({
            "count": count,
            "operands": operands
        })

    return extractions


def infer_operation_greedy(window_text):
    """
    Simple keyword heuristic for operation inference.
    This is ONLY for baseline evaluation - factor graph replaces this.
    """
    text = window_text.lower()

    # Order matters - check more specific patterns first
    if any(w in text for w in ['square root', 'sqrt', '√', '\\sqrt']):
        return 'SQRT'
    elif any(w in text for w in ['squared', 'square', '^2', '**2']):
        return 'POW'
    elif any(w in text for w in ['power', 'exponent', '^', '**']):
        return 'POW'
    elif any(w in text for w in ['times', 'multiply', 'product', '×', '*', 'multiplied']):
        return 'MUL'
    elif any(w in text for w in ['divide', 'divided', 'ratio', '÷', '/', 'quotient']):
        return 'DIV'
    elif any(w in text for w in ['add', 'plus', 'sum', 'total', 'combined', '+']):
        return 'ADD'
    elif any(w in text for w in ['subtract', 'minus', 'difference', 'less', 'remain', '-', 'fewer']):
        return 'SUB'
    elif any(w in text for w in ['solve', 'find', 'calculate', 'compute', 'evaluate']):
        return 'EVAL'
    elif any(w in text for w in ['factor']):
        return 'FACTOR'
    elif any(w in text for w in ['equal', '=']):
        return 'EQ'
    else:
        return 'UNKNOWN'


def extract_numbers_from_text(text):
    """Extract numeric values from text."""
    # Match integers, decimals, fractions
    patterns = [
        r'-?\d+\.\d+',  # decimals
        r'-?\d+/\d+',   # fractions
        r'-?\d+',       # integers
    ]

    numbers = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                val = match.group()
                if '/' in val:
                    num, den = val.split('/')
                    numbers.append(sympy.Rational(int(num), int(den)))
                elif '.' in val:
                    numbers.append(sympy.Float(val))
                else:
                    numbers.append(sympy.Integer(val))
            except:
                pass

    return numbers


def assemble_greedy(segments):
    """
    Assemble a SymPy expression from segments.
    Process in order, use back-references where flagged.
    """
    if not segments:
        return None, "no_segments"

    results = []

    for seg_idx, seg in enumerate(segments):
        op = seg.get('operation', 'UNKNOWN')
        c3_out = seg.get('c3_output', {})
        window_text = seg.get('text', '')

        # Get operands
        operands = []
        c3_operands = c3_out.get('operands', [])

        # Extract numbers from window text as fallback
        text_numbers = extract_numbers_from_text(window_text)

        for i, op_info in enumerate(c3_operands):
            if op_info.get('is_backref'):
                # Use most recent result
                if results:
                    operands.append(results[-1])
                else:
                    operands.append(sympy.Symbol('prev'))
            else:
                # Try to get from text numbers
                pos = op_info.get('position', -1)
                if i < len(text_numbers):
                    operands.append(text_numbers[i])
                elif text_numbers:
                    operands.append(text_numbers[0])
                else:
                    operands.append(sympy.Symbol(f'x{i}'))

        # If no operands from C3, extract from text
        if not operands and text_numbers:
            operands = text_numbers[:2]

        # Assemble based on operation
        result = None
        try:
            if op == 'ADD' and len(operands) >= 2:
                result = operands[0] + operands[1]
            elif op == 'SUB' and len(operands) >= 2:
                result = operands[0] - operands[1]
            elif op == 'MUL' and len(operands) >= 2:
                result = operands[0] * operands[1]
            elif op == 'DIV' and len(operands) >= 2:
                result = operands[0] / operands[1]
            elif op == 'POW' and len(operands) >= 2:
                result = operands[0] ** operands[1]
            elif op == 'SQRT' and len(operands) >= 1:
                result = sympy.sqrt(operands[0])
            elif op == 'EVAL' and len(operands) >= 1:
                result = operands[0]
            elif op in ['EQ', 'UNKNOWN'] and len(operands) >= 1:
                result = operands[-1]
            elif operands:
                result = operands[-1]
        except Exception as e:
            pass

        if result is not None:
            results.append(result)

    if results:
        return results[-1], "success"
    else:
        return None, "no_results"


def evaluate_answer(pipeline_answer, ground_truth_str):
    """Compare pipeline output to expected answer."""
    if pipeline_answer is None:
        return {'status': 'assembly_failed', 'correct': False}

    try:
        # Try to parse ground truth
        gt_clean = ground_truth_str.strip()

        # Handle common formats
        gt_clean = re.sub(r'\$([^$]+)\$', r'\1', gt_clean)  # Remove $ delimiters
        gt_clean = gt_clean.replace('\\frac{', '(').replace('}{', ')/(').replace('}', ')')
        gt_clean = gt_clean.replace('\\cdot', '*').replace('\\times', '*')
        gt_clean = gt_clean.replace('\\div', '/').replace('\\pm', '+')

        # Try sympy parsing
        try:
            expected = parse_expr(gt_clean)
        except:
            # Try as number
            try:
                expected = sympy.sympify(gt_clean)
            except:
                return {'status': 'gt_parse_error', 'correct': False, 'error': f'Cannot parse: {gt_clean}'}

        # Compare
        result = sympy.simplify(pipeline_answer)
        diff = sympy.simplify(result - expected)

        if diff == 0:
            return {'status': 'correct', 'correct': True, 'result': str(result)}
        else:
            # Check if numerically close
            try:
                result_float = float(result.evalf())
                expected_float = float(expected.evalf())
                if abs(result_float - expected_float) < 1e-6:
                    return {'status': 'correct', 'correct': True, 'result': str(result)}
            except:
                pass

            return {'status': 'incorrect', 'correct': False,
                    'result': str(result), 'expected': str(expected)}

    except Exception as e:
        return {'status': 'execution_error', 'correct': False, 'error': str(e)}


def attribute_failure(trace):
    """Determine which component caused the failure."""
    if trace['execution']['correct']:
        return 'correct', 'Problem solved correctly'

    # Check C1-A
    c1a = trace.get('c1a_output', {})
    if c1a.get('n_windows_predicted', 0) == 0:
        return 'c1a_error', 'C1-A predicted no segments'

    # Check operation inference
    segments = trace.get('segments', [])
    unknown_ops = sum(1 for s in segments if s.get('operation') == 'UNKNOWN')
    if unknown_ops > len(segments) / 2:
        return 'operation_error', f'{unknown_ops}/{len(segments)} operations unknown'

    # Check C3
    c3_issues = 0
    for s in segments:
        c3_out = s.get('c3_output', {})
        if c3_out.get('count', 0) == 0:
            c3_issues += 1
    if c3_issues > len(segments) / 2:
        return 'c3_error', f'{c3_issues}/{len(segments)} segments with no operands'

    # Check assembly
    if trace['assembly'].get('status') != 'success':
        return 'assembly_error', trace['assembly'].get('status', 'unknown')

    # Execution error
    exec_status = trace['execution'].get('status', '')
    if 'error' in exec_status:
        return 'execution_error', trace['execution'].get('error', 'unknown')

    # Wrong answer - could be any component
    return 'wrong_answer', 'Assembled expression evaluates to wrong result'


# =============================================================================
# Main Evaluation
# =============================================================================

def load_test_problems():
    """Load validation split of problems with answers."""
    print("Loading test problems...")

    # Load C1 training data
    resp = s3.get_object(Bucket=BUCKET, Key="c1_training_v6/merged_training.jsonl")

    all_problems = []
    for line in resp["Body"].iter_lines():
        p = json.loads(line.decode("utf-8"))
        all_problems.append(p)

    # Use last 10% as validation
    n_val = len(all_problems) // 10
    val_problems = all_problems[-n_val:]

    # Try to load answers from MATH dataset
    # For now, we'll evaluate structural correctness

    print(f"  Loaded {len(val_problems)} validation problems")
    return val_problems


def run_greedy_pipeline(problem, c1a_model, c1a_tokenizer, c1a_head, c3_model, device):
    """Run the full greedy pipeline on a single problem."""
    problem_text = problem.get("original_text", problem.get("problem_text", ""))

    trace = {
        "problem_idx": problem.get("problem_idx"),
        "problem_text": problem_text[:500],  # Truncate for logging
    }

    # Step 1: C1-A segmentation
    positive_windows, window_features = run_c1a_segmentation(
        c1a_model, c1a_tokenizer, c1a_head, problem_text, device
    )

    trace["c1a_output"] = {
        "n_windows_predicted": len(positive_windows),
        "windows": [w["text"][:100] for w in positive_windows[:5]]  # Sample
    }

    if not positive_windows:
        trace["segments"] = []
        trace["assembly"] = {"status": "no_segments", "expression": None}
        trace["execution"] = {"status": "skipped", "correct": False}
        trace["failure_attribution"] = attribute_failure(trace)
        return trace

    # Step 2: C3 operand extraction
    c3_outputs = run_c3_extraction(c3_model, window_features, device)

    # Step 3: Operation inference (heuristic)
    segments = []
    for i, (window, c3_out) in enumerate(zip(positive_windows, c3_outputs)):
        op = infer_operation_greedy(window["text"])
        segments.append({
            "window_idx": i,
            "text": window["text"],
            "operation": op,
            "c3_output": c3_out
        })

    trace["segments"] = segments

    # Step 4: Assembly
    expression, assembly_status = assemble_greedy(segments)
    trace["assembly"] = {
        "status": assembly_status,
        "expression": str(expression) if expression else None
    }

    # Step 5: Evaluation
    # For now, we can't evaluate against ground truth without answers
    # Just check if we got a valid expression
    if expression is not None:
        try:
            simplified = sympy.simplify(expression)
            trace["execution"] = {
                "status": "executed",
                "correct": False,  # Can't verify without ground truth
                "result": str(simplified)
            }
        except Exception as e:
            trace["execution"] = {
                "status": "execution_error",
                "correct": False,
                "error": str(e)
            }
    else:
        trace["execution"] = {
            "status": "assembly_failed",
            "correct": False
        }

    trace["failure_attribution"] = attribute_failure(trace)
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-problems", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load models
    c1a_model, c1a_tokenizer, c1a_head = load_c1a_model(device)
    c3_model = load_c3_model(device)

    # Load problems
    problems = load_test_problems()
    problems = problems[:args.max_problems]

    print(f"\nRunning greedy pipeline on {len(problems)} problems...")

    # Run pipeline
    traces = []
    failure_counts = defaultdict(int)

    for i, problem in enumerate(problems):
        trace = run_greedy_pipeline(
            problem, c1a_model, c1a_tokenizer, c1a_head, c3_model, device
        )
        traces.append(trace)

        failure_type, _ = trace["failure_attribution"]
        failure_counts[failure_type] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(problems)}] Failures: {dict(failure_counts)}")

    # Compute summary
    total = len(traces)
    summary = {
        "total_problems": total,
        "failure_breakdown": dict(failure_counts),
        "rates": {k: v/total for k, v in failure_counts.items()}
    }

    # Segment statistics
    all_ops = []
    all_counts = []
    for t in traces:
        for s in t.get("segments", []):
            all_ops.append(s.get("operation", "UNKNOWN"))
            all_counts.append(s.get("c3_output", {}).get("count", 0))

    from collections import Counter
    summary["operation_distribution"] = dict(Counter(all_ops))
    summary["operand_count_distribution"] = dict(Counter(all_counts))

    # Average segments per problem
    segs_per_problem = [len(t.get("segments", [])) for t in traces]
    summary["avg_segments_per_problem"] = sum(segs_per_problem) / len(segs_per_problem)

    print("\n" + "="*60)
    print("GREEDY PIPELINE EVALUATION SUMMARY")
    print("="*60)
    print(f"Total problems: {total}")
    print(f"\nFailure breakdown:")
    for k, v in sorted(failure_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({v/total*100:.1f}%)")
    print(f"\nOperation distribution:")
    for k, v in sorted(summary["operation_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"\nAvg segments per problem: {summary['avg_segments_per_problem']:.2f}")

    # Save results
    print("\nSaving results to S3...")

    # Per-problem traces
    traces_jsonl = "\n".join(json.dumps(t) for t in traces)
    s3.put_object(
        Bucket=BUCKET,
        Key="greedy_eval/per_problem_traces.jsonl",
        Body=traces_jsonl.encode("utf-8")
    )

    # Summary
    s3.put_object(
        Bucket=BUCKET,
        Key="greedy_eval/summary.json",
        Body=json.dumps(summary, indent=2).encode("utf-8")
    )

    # Examples by failure type
    examples = defaultdict(list)
    for t in traces:
        failure_type, _ = t["failure_attribution"]
        if len(examples[failure_type]) < 20:
            examples[failure_type].append(t)

    for failure_type, ex_list in examples.items():
        s3.put_object(
            Bucket=BUCKET,
            Key=f"greedy_eval/examples/{failure_type}.json",
            Body=json.dumps(ex_list, indent=2).encode("utf-8")
        )

    print(f"Results saved to s3://{BUCKET}/greedy_eval/")
    print("\nDone!")


if __name__ == "__main__":
    main()
