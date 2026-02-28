#!/usr/bin/env python3
"""
Quick C2 + C3 + Sympy Integration Test

Tests the minimal pipeline:
  Problem text → C2 (classify ops) → C3 (extract expressions) → Sympy (evaluate)

Usage:
  python test_c2c3_integration.py --c2-path models/c2 --c3-path models/c3
"""

import json
import torch
import torch.nn as nn
import sympy
import argparse
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import boto3
import tempfile
import os

# ============================================================
# C2 Model (Multi-label classifier with heartbeat)
# ============================================================

class C2Model(nn.Module):
    """C2 classifier: problem text → operation labels."""

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
    """Wrapper for C2 inference."""

    def __init__(self, model_path: str):
        # Load config
        with open(f"{model_path}/config.json") as f:
            config = json.load(f)

        self.label2id = config["label2id"]
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        self.num_labels = config["num_labels"]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model
        backbone_name = config["model_name"]
        self.model = C2Model(backbone_name, self.num_labels)

        checkpoint = torch.load(f"{model_path}/model.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        print(f"C2 loaded: {self.num_labels} labels")

    def classify(self, problem_text: str, threshold: float = 0.5):
        """Classify problem text → list of operation labels."""
        encoding = self.tokenizer(
            problem_text,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits, heartbeat = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

        # Get labels above threshold
        labels = []
        confidences = []
        for i, p in enumerate(probs[0].cpu().numpy()):
            if p > threshold:
                labels.append(self.id2label[i])
                confidences.append(float(p))

        return {
            "labels": labels,
            "confidences": confidences,
            "heartbeat_count": heartbeat.item()
        }


# ============================================================
# C3 Model (Expression extractor)
# ============================================================

class C3Extractor:
    """Wrapper for C3 inference."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"C3 loaded: {self.model.config._name_or_path}")

    def extract(self, problem_text: str, template_tag: str) -> str:
        """Extract sympy expression for a given operation template."""
        input_text = f"Problem: {problem_text}\n\nTemplate: {template_tag}"
        prompt = f"{input_text}\nExpression: "

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Clean up - take first line, strip whitespace
        expression = generated.strip().split("\n")[0].strip()
        return expression


# ============================================================
# Sympy Evaluator
# ============================================================

def evaluate_expression(expr_str: str) -> dict:
    """Evaluate a sympy expression string."""
    try:
        expr = sympy.sympify(expr_str)
        result = float(expr.evalf())
        return {
            "expression": expr_str,
            "result": result,
            "valid": True,
            "error": None
        }
    except Exception as e:
        return {
            "expression": expr_str,
            "result": None,
            "valid": False,
            "error": str(e)
        }


# ============================================================
# Integration Test
# ============================================================

def download_model_from_s3(s3_path: str, local_dir: str):
    """Download model files from S3."""
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
                local_path = f"{local_dir}/{filename}"
                print(f"  Downloading {filename}...")
                s3.download_file(bucket, key, local_path)


def run_test(c2_path: str, c3_path: str, test_problems: list = None):
    """Run integration test."""

    print("=" * 60)
    print("C2 + C3 + SYMPY INTEGRATION TEST")
    print("=" * 60)

    # Default test problems
    if test_problems is None:
        test_problems = [
            {
                "text": "John has 5 apples. He buys 3 more apples. How many apples does John have?",
                "expected": 8
            },
            {
                "text": "A store has 24 oranges. If 6 oranges are sold, how many oranges remain?",
                "expected": 18
            },
            {
                "text": "Each box contains 12 pencils. If there are 4 boxes, how many pencils are there in total?",
                "expected": 48
            },
            {
                "text": "Sarah has 36 cookies to share equally among 9 friends. How many cookies does each friend get?",
                "expected": 4
            },
        ]

    # Load models
    print("\nLoading C2 classifier...")
    c2 = C2Classifier(c2_path)

    print("\nLoading C3 extractor...")
    c3 = C3Extractor(c3_path)

    # Run tests
    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)

    results = []
    correct = 0

    for i, problem in enumerate(test_problems):
        print(f"\n--- Test {i+1} ---")
        print(f"Problem: {problem['text'][:80]}...")

        # Step 1: C2 classification
        c2_result = c2.classify(problem["text"])
        labels = c2_result["labels"]
        print(f"C2 labels: {labels}")
        print(f"C2 heartbeat: {c2_result['heartbeat_count']:.2f}")

        # Step 2: C3 extraction for each label
        expressions = []
        for label in labels[:3]:  # Limit to top 3 labels
            expr = c3.extract(problem["text"], label)
            expressions.append((label, expr))
            print(f"C3 [{label}]: {expr}")

        # Step 3: Sympy evaluation
        best_result = None
        for label, expr in expressions:
            eval_result = evaluate_expression(expr)
            if eval_result["valid"]:
                print(f"Sympy [{label}]: {eval_result['result']}")
                if best_result is None:
                    best_result = eval_result["result"]

        # Check answer
        expected = problem.get("expected")
        is_correct = False
        if best_result is not None and expected is not None:
            is_correct = abs(best_result - expected) < 0.01
            if is_correct:
                correct += 1
                print(f"✓ CORRECT (expected {expected})")
            else:
                print(f"✗ WRONG (got {best_result}, expected {expected})")
        else:
            print(f"? NO VALID RESULT (expected {expected})")

        results.append({
            "problem": problem["text"],
            "expected": expected,
            "c2_labels": labels,
            "expressions": expressions,
            "result": best_result,
            "correct": is_correct
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(test_problems)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {100*correct/len(test_problems):.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2-path", default="s3://mycelium-data/models/c2_heartbeat/")
    parser.add_argument("--c3-path", default="s3://mycelium-data/models/c3_extractor/")
    parser.add_argument("--local-dir", default="/tmp/mycelium_models")
    args = parser.parse_args()

    # Download models if S3 paths
    c2_local = args.c2_path
    c3_local = args.c3_path

    if args.c2_path.startswith("s3://"):
        c2_local = f"{args.local_dir}/c2"
        print(f"\nDownloading C2 from {args.c2_path}...")
        download_model_from_s3(args.c2_path, c2_local)

    if args.c3_path.startswith("s3://"):
        c3_local = f"{args.local_dir}/c3"
        print(f"\nDownloading C3 from {args.c3_path}...")
        download_model_from_s3(args.c3_path, c3_local)

    # Run test
    results = run_test(c2_local, c3_local)

    # Save results
    output_path = f"{args.local_dir}/integration_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
