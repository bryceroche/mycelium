#!/usr/bin/env python3
"""
C2 + C3 + Sympy Integration Test on 50 MATH Problems

Tests the pipeline on real C3 training data (which has gold expressions):
  Problem text → C2 (classify ops) → C3 (extract expression) → Sympy (evaluate)

Metrics:
  - C2: Does the predicted label match the gold template?
  - C3: Does the extracted expression match the gold expression (symbolic equivalence)?
  - Sympy: Does the expression evaluate to a valid number?
"""

import json
import torch
import torch.nn as nn
import sympy
import argparse
import random
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import boto3
import os

# ============================================================
# C2 Model (Multi-label classifier)
# ============================================================

class C2Model(nn.Module):
    def __init__(self, backbone_name, num_labels, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )

        self.heartbeat_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        heartbeat_pred = self.heartbeat_head(cls_output).squeeze(-1)
        return logits, heartbeat_pred


class C2Classifier:
    def __init__(self, model_path: str):
        with open(f"{model_path}/config.json") as f:
            config = json.load(f)

        self.label2id = config["label2id"]
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        self.num_labels = config["num_labels"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        backbone_name = config["model_name"]
        self.model = C2Model(backbone_name, self.num_labels)
        checkpoint = torch.load(f"{model_path}/model.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"C2 loaded: {self.num_labels} labels, device={self.device}")

    def classify(self, problem_text: str, threshold: float = 0.3):
        encoding = self.tokenizer(
            problem_text, truncation=True, max_length=256,
            padding="max_length", return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits, heartbeat = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

        # Get all labels with prob > threshold, sorted by confidence
        label_probs = [(self.id2label[i], float(p)) for i, p in enumerate(probs[0].cpu().numpy())]
        label_probs.sort(key=lambda x: -x[1])
        labels = [(lbl, conf) for lbl, conf in label_probs if conf > threshold]

        return {
            "labels": labels,
            "heartbeat": heartbeat.item(),
            "all_probs": label_probs
        }


# ============================================================
# C3 Model (Expression extractor)
# ============================================================

class C3Extractor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"C3 loaded, device={self.device}")

    def extract(self, problem_text: str, template_tag: str) -> str:
        input_text = f"Problem: {problem_text}\n\nTemplate: {template_tag}"
        prompt = f"{input_text}\nExpression: "

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        expression = generated.strip().split("\n")[0].strip()
        return expression


# ============================================================
# Sympy Utilities
# ============================================================

def eval_expr(expr_str: str):
    """Evaluate expression, return (result, is_valid)."""
    try:
        expr = sympy.sympify(expr_str)
        result = float(expr.evalf())
        if not (result != result):  # NaN check
            return result, True
    except:
        pass
    return None, False


def symbolic_match(pred_expr: str, gold_expr: str) -> bool:
    """Check if two expressions are symbolically equivalent."""
    try:
        pred = sympy.sympify(pred_expr)
        gold = sympy.sympify(gold_expr)
        # Check if difference simplifies to 0
        diff = sympy.simplify(pred - gold)
        return diff == 0
    except:
        return False


def numeric_match(pred_expr: str, gold_expr: str, tol: float = 1e-6) -> bool:
    """Check if two expressions evaluate to the same number."""
    pred_val, pred_ok = eval_expr(pred_expr)
    gold_val, gold_ok = eval_expr(gold_expr)
    if pred_ok and gold_ok:
        return abs(pred_val - gold_val) < tol
    return False


# ============================================================
# Data Loading
# ============================================================

def download_from_s3(s3_path: str, local_path: str):
    s3 = boto3.client('s3')
    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:])
    s3.download_file(bucket, key, local_path)


def download_model_from_s3(s3_path: str, local_dir: str):
    s3 = boto3.client('s3')
    bucket = s3_path.split('/')[2]
    prefix = '/'.join(s3_path.split('/')[3:])
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = key.split('/')[-1]
            if filename:
                s3.download_file(bucket, key, f"{local_dir}/{filename}")


def load_test_problems(data_path: str, n: int = 50, seed: int = 42):
    """Load n random problems from C3 training data."""
    problems = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            # Parse input to extract problem text and template
            input_text = ex.get("input", "")
            if "Problem:" in input_text and "Template:" in input_text:
                parts = input_text.split("Template:")
                problem_text = parts[0].replace("Problem:", "").strip()
                template = parts[1].strip() if len(parts) > 1 else ""
            else:
                problem_text = input_text
                template = ""

            problems.append({
                "problem_text": problem_text,
                "template": template,
                "gold_expression": ex.get("output", "").strip()
            })

    # Sample n problems
    random.seed(seed)
    if len(problems) > n:
        problems = random.sample(problems, n)

    return problems


# ============================================================
# Main Test
# ============================================================

def run_test(c2_path: str, c3_path: str, data_path: str, n_problems: int = 50):
    print("=" * 70)
    print("C2 + C3 + SYMPY INTEGRATION TEST (50 MATH PROBLEMS)")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    c2 = C2Classifier(c2_path)
    c3 = C3Extractor(c3_path)

    # Load test problems
    print(f"\nLoading {n_problems} test problems...")
    problems = load_test_problems(data_path, n_problems)
    print(f"Loaded {len(problems)} problems")

    # Run tests
    results = {
        "c2_any_correct": 0,  # C2 predicted the gold template in top-k
        "c2_top1_correct": 0,  # C2 predicted gold template as top-1
        "c3_symbolic_match": 0,  # C3 expression matches gold symbolically
        "c3_numeric_match": 0,  # C3 expression matches gold numerically
        "sympy_valid": 0,  # C3 expression is valid sympy
        "total": len(problems)
    }

    print("\n" + "-" * 70)
    for i, prob in enumerate(problems):
        problem_text = prob["problem_text"]
        gold_template = prob["template"]
        gold_expr = prob["gold_expression"]

        # C2: Classify
        c2_result = c2.classify(problem_text)
        predicted_labels = [lbl for lbl, _ in c2_result["labels"]]

        c2_any = gold_template in predicted_labels
        c2_top1 = len(predicted_labels) > 0 and predicted_labels[0] == gold_template

        if c2_any:
            results["c2_any_correct"] += 1
        if c2_top1:
            results["c2_top1_correct"] += 1

        # C3: Extract expression using gold template
        pred_expr = c3.extract(problem_text, gold_template)

        # Evaluate
        pred_val, sympy_valid = eval_expr(pred_expr)
        if sympy_valid:
            results["sympy_valid"] += 1

        if symbolic_match(pred_expr, gold_expr):
            results["c3_symbolic_match"] += 1
        if numeric_match(pred_expr, gold_expr):
            results["c3_numeric_match"] += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(problems)}...")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total problems: {results['total']}")
    print()
    print("C2 Classifier:")
    print(f"  Any-correct (gold in top-k): {results['c2_any_correct']}/{results['total']} ({100*results['c2_any_correct']/results['total']:.1f}%)")
    print(f"  Top-1 correct:               {results['c2_top1_correct']}/{results['total']} ({100*results['c2_top1_correct']/results['total']:.1f}%)")
    print()
    print("C3 Extractor:")
    print(f"  Sympy valid:     {results['sympy_valid']}/{results['total']} ({100*results['sympy_valid']/results['total']:.1f}%)")
    print(f"  Symbolic match:  {results['c3_symbolic_match']}/{results['total']} ({100*results['c3_symbolic_match']/results['total']:.1f}%)")
    print(f"  Numeric match:   {results['c3_numeric_match']}/{results['total']} ({100*results['c3_numeric_match']/results['total']:.1f}%)")
    print()
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2-path", default="s3://mycelium-data/models/c2_heartbeat/")
    parser.add_argument("--c3-path", default="s3://mycelium-data/models/c3_extractor/")
    parser.add_argument("--data-path", default="s3://mycelium-data/c3_training/c3_train_fulltext.jsonl")
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--local-dir", default="/tmp/mycelium_test")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    # Download models
    c2_local = args.c2_path
    c3_local = args.c3_path
    data_local = args.data_path

    if args.c2_path.startswith("s3://"):
        c2_local = f"{args.local_dir}/c2"
        print(f"Downloading C2 from {args.c2_path}...")
        download_model_from_s3(args.c2_path, c2_local)

    if args.c3_path.startswith("s3://"):
        c3_local = f"{args.local_dir}/c3"
        print(f"Downloading C3 from {args.c3_path}...")
        download_model_from_s3(args.c3_path, c3_local)

    if args.data_path.startswith("s3://"):
        data_local = f"{args.local_dir}/c3_test.jsonl"
        print(f"Downloading test data from {args.data_path}...")
        download_from_s3(args.data_path, data_local)

    # Run test
    results = run_test(c2_local, c3_local, data_local, args.n_problems)

    # Save
    with open(f"{args.local_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
