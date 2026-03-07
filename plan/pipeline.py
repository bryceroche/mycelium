"""
Mycelium Full Pipeline

C1-A (boundaries) → C3 (operands) → Factor Graph (inference) → Result

Usage:
    python pipeline.py --max-problems 100
"""

import json
import io
import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import sympy
import boto3
from botocore.config import Config
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# Import our modules
from template_library import TemplateLibrary, CLUSTER_TEMPLATE_MAP
from factor_graph import (
    run_factor_graph, evaluate_result, print_inference_trace,
    Segment, Operand, extract_numbers_from_text
)

config = Config(read_timeout=120, connect_timeout=30, retries={'max_attempts': 3})
s3 = boto3.client("s3", config=config)
BUCKET = "mycelium-data"

# Parameters
W = 16
S = 8
HIDDEN_DIM = 896
MAX_OPERANDS = 4
N_C2_CLASSES = 26


def preprocess_latex(text: str) -> str:
    """Apply LaTeX preprocessing."""
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\begin{array}", "").replace("\\end{array}", "")
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ================================================================
# MODEL LOADING
# ================================================================

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

    print("  C1-A loaded")
    return model, tokenizer, head_weights


class C3Model(nn.Module):
    """C3 operand extractor."""
    def __init__(self, hidden_dim=HIDDEN_DIM, c2_embed_dim=32,
                 n_c2_classes=N_C2_CLASSES, max_operands=MAX_OPERANDS, window_size=W):
        super().__init__()
        self.c2_embedding = nn.Embedding(n_c2_classes, c2_embed_dim)
        self.max_operands = max_operands
        self.window_size = window_size

        self.operand_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + c2_embed_dim, 128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, window_size + 1)
            )
            for _ in range(max_operands)
        ])

        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim + c2_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, max_operands + 1)
        )

    def forward(self, window_features, c2_label):
        c2_embed = self.c2_embedding(c2_label)
        pooled_with_c2 = torch.cat([window_features, c2_embed], dim=-1)
        count_logits = self.count_head(pooled_with_c2)
        operand_logits = [scorer(pooled_with_c2) for scorer in self.operand_scorers]
        return count_logits, operand_logits


def load_c3_model(device):
    """Load C3 model from S3."""
    print("Loading C3 model...")

    c3_local = Path("/tmp/c3_model")
    c3_local.mkdir(exist_ok=True)

    s3.download_file(BUCKET, "models/c3_v1/best_checkpoint.pt", str(c3_local / "checkpoint.pt"))
    s3.download_file(BUCKET, "models/c3_v1/config.json", str(c3_local / "config.json"))

    model = C3Model()
    checkpoint = torch.load(c3_local / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print("  C3 loaded")
    return model


# ================================================================
# INFERENCE COMPONENTS
# ================================================================

def run_c1a(model, tokenizer, head_weights, problem_text, device):
    """Run C1-A segmentation."""
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
        hidden_states = outputs.last_hidden_state

    seq_len = hidden_states.shape[1]

    # Extract windows
    windows = []
    window_features = []

    for w_start in range(0, seq_len, S):
        w_end = min(w_start + W, seq_len)
        if w_end - w_start < W // 2:
            break

        window_feats = hidden_states[0, w_start:w_end, :].mean(dim=0)
        window_features.append(window_feats)

        token_ids = inputs["input_ids"][0, w_start:w_end].tolist()
        window_text = tokenizer.decode(token_ids, skip_special_tokens=True)

        windows.append({
            "start": w_start,
            "end": w_end,
            "text": window_text
        })

    if not window_features:
        return [], []

    # Apply head for boundary prediction
    features = torch.stack(window_features)

    if "classifier.weight" in head_weights:
        W_cls = head_weights["classifier.weight"].to(device)
        b_cls = head_weights["classifier.bias"].to(device)
        logits = features @ W_cls.T + b_cls
    else:
        # Fallback
        logits = features.norm(dim=1, keepdim=True)

    probs = torch.sigmoid(logits[:, 0] if logits.dim() > 1 else logits)

    # Get positive windows
    positive_mask = probs > 0.5
    positive_indices = positive_mask.nonzero().squeeze(-1).tolist()
    if isinstance(positive_indices, int):
        positive_indices = [positive_indices]

    positive_windows = [windows[i] for i in positive_indices]
    positive_features = [window_features[i] for i in positive_indices]

    return positive_windows, positive_features


def run_c3(model, window_features, device):
    """Run C3 operand extraction."""
    if not window_features:
        return []

    features = torch.stack(window_features).to(device)
    dummy_c2 = torch.zeros(len(window_features), dtype=torch.long, device=device)

    with torch.no_grad():
        count_logits, position_logits = model(features, dummy_c2)

    counts = count_logits.argmax(dim=1).tolist()

    outputs = []
    for i, count in enumerate(counts):
        operands = []
        for op_idx in range(min(count, MAX_OPERANDS)):
            pos = position_logits[op_idx][i].argmax().item()
            operands.append({
                "position": pos if pos < W else -1,
                "is_backref": pos == W
            })
        outputs.append({"count": count, "operands": operands})

    return outputs


def run_full_pipeline(problem_text, c1a_model, c1a_tokenizer, c1a_head,
                      c3_model, device, verbose=False):
    """Run full pipeline: C1-A → C3 → Factor Graph."""

    # Step 1: C1-A segmentation
    windows, features = run_c1a(c1a_model, c1a_tokenizer, c1a_head, problem_text, device)

    if not windows:
        return None, {"status": "no_segments", "n_windows": 0}

    # Step 2: C3 operand extraction
    c3_outputs = run_c3(c3_model, features, device)

    # Step 3: Factor graph inference
    result, state = run_factor_graph(windows, c3_outputs, n_steps=20)

    if verbose:
        print_inference_trace(state)

    trace = {
        "n_windows": len(windows),
        "n_segments_with_result": sum(1 for s in state.segments if s.result is not None),
        "converged": state.converged,
        "windows": [w["text"][:50] for w in windows[:3]],
        "selected_templates": [s.selected_template for s in state.segments if s.selected_template],
    }

    return result, trace


# ================================================================
# EVALUATION
# ================================================================

def load_test_problems():
    """Load validation split of problems."""
    print("Loading test problems...")

    resp = s3.get_object(Bucket=BUCKET, Key="c1_training_v6/merged_training.jsonl")

    all_problems = []
    for line in resp["Body"].iter_lines():
        p = json.loads(line.decode("utf-8"))
        all_problems.append(p)

    # Use last 10% as validation
    n_val = len(all_problems) // 10
    val_problems = all_problems[-n_val:]

    print(f"  Loaded {len(val_problems)} validation problems")
    return val_problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-problems", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load models
    c1a_model, c1a_tokenizer, c1a_head = load_c1a_model(device)
    c3_model = load_c3_model(device)

    # Load problems
    problems = load_test_problems()
    problems = problems[:args.max_problems]

    print(f"\nRunning pipeline on {len(problems)} problems...")

    # Run pipeline
    results = []
    stats = defaultdict(int)

    for i, problem in enumerate(problems):
        problem_text = problem.get("original_text", problem.get("problem_text", ""))

        result, trace = run_full_pipeline(
            problem_text, c1a_model, c1a_tokenizer, c1a_head,
            c3_model, device, verbose=args.verbose
        )

        # Record result
        record = {
            "problem_idx": problem.get("problem_idx"),
            "problem_text": problem_text[:200],
            "result": str(result) if result is not None else None,
            "trace": trace,
        }

        if result is not None:
            stats["has_result"] += 1
        else:
            stats["no_result"] += 1

        if trace.get("converged"):
            stats["converged"] += 1

        results.append(record)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(problems)}] Results: {stats['has_result']}, No result: {stats['no_result']}")

    # Summary
    print("\n" + "="*60)
    print("PIPELINE EVALUATION SUMMARY")
    print("="*60)
    print(f"Total problems: {len(problems)}")
    print(f"Has result: {stats['has_result']} ({100*stats['has_result']/len(problems):.1f}%)")
    print(f"No result: {stats['no_result']} ({100*stats['no_result']/len(problems):.1f}%)")
    print(f"Converged: {stats['converged']} ({100*stats['converged']/len(problems):.1f}%)")

    # Template usage
    template_counts = defaultdict(int)
    for r in results:
        for t in r["trace"].get("selected_templates", []):
            if t:
                template_counts[t] += 1

    print("\nTemplate usage:")
    for t, c in sorted(template_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t}: {c}")

    # Save results
    print("\nSaving results to S3...")
    s3.put_object(
        Bucket=BUCKET,
        Key="factor_graph_eval/pipeline_results.jsonl",
        Body="\n".join(json.dumps(r) for r in results).encode("utf-8")
    )

    summary = {
        "total": len(problems),
        "has_result": stats["has_result"],
        "no_result": stats["no_result"],
        "converged": stats["converged"],
        "template_usage": dict(template_counts),
    }
    s3.put_object(
        Bucket=BUCKET,
        Key="factor_graph_eval/summary.json",
        Body=json.dumps(summary, indent=2).encode("utf-8")
    )

    print(f"Results saved to s3://{BUCKET}/factor_graph_eval/")
    print("\nDone!")


if __name__ == "__main__":
    main()
