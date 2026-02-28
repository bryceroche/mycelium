#!/usr/bin/env python3
"""
C2 + C3 + Sympy Batch Integration Test

Batch inference for efficiency:
  Problem text → C2 (batch classify) → C3 (batch extract) → Sympy (evaluate)
"""

import json
import torch
import torch.nn as nn
import sympy
import argparse
import random
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import boto3
import os
import re

# ============================================================
# C2 Model (Multi-label classifier) - BATCH INFERENCE
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
        heartbeat = self.heartbeat_head(cls_output).squeeze(-1)
        return logits, heartbeat


class C2Classifier:
    def __init__(self, model_path: str):
        with open(f"{model_path}/config.json") as f:
            config = json.load(f)
        self.label2id = config["label2id"]
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        self.num_labels = config["num_labels"]
        self.labels_list = config["labels"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        backbone_name = config["model_name"]
        self.model = C2Model(backbone_name, self.num_labels)
        checkpoint = torch.load(f"{model_path}/model.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"C2 loaded: {self.num_labels} labels on {self.device}")

    def classify_batch(self, texts: list, threshold: float = 0.3):
        """Batch classify multiple texts."""
        encodings = self.tokenizer(
            texts, truncation=True, max_length=256,
            padding=True, return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            logits, heartbeats = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)

        results = []
        for i in range(len(texts)):
            labels = []
            for j, p in enumerate(probs[i].cpu().numpy()):
                if p > threshold:
                    labels.append((self.id2label[j], float(p)))
            labels.sort(key=lambda x: -x[1])
            results.append({
                "labels": labels,
                "heartbeat": heartbeats[i].item()
            })
        return results


# ============================================================
# C3 Model - BATCH INFERENCE
# ============================================================

class C3Extractor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"  # For batch generation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"C3 loaded on {self.device}")

    def extract_batch(self, problems: list, templates: list, batch_size: int = 8) -> list:
        """Batch extract expressions."""
        all_results = []

        for i in range(0, len(problems), batch_size):
            batch_problems = problems[i:i+batch_size]
            batch_templates = templates[i:i+batch_size]

            # Build prompts - MUST match training format exactly
            # Training format: "[TEMPLATE: X] problem text...\nExpression: output"
            prompts = []
            for prob, tmpl in zip(batch_problems, batch_templates):
                prompt = f"[TEMPLATE: {tmpl}] {prob}\nExpression: "
                prompts.append(prompt)

            # Tokenize
            inputs = self.tokenizer(
                prompts, return_tensors="pt", truncation=True,
                max_length=384, padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=64, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode
            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                generated = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)
                expr = generated.strip().split("\n")[0].strip()
                all_results.append(expr)

        return all_results


# ============================================================
# Sympy Utilities
# ============================================================

def eval_expr(expr_str: str):
    try:
        expr = sympy.sympify(expr_str)
        result = float(expr.evalf())
        if result == result:  # Not NaN
            return result, True
    except:
        pass
    return None, False


def numeric_match(pred_expr: str, gold_expr: str, tol: float = 1e-4) -> bool:
    pred_val, pred_ok = eval_expr(pred_expr)
    gold_val, gold_ok = eval_expr(gold_expr)
    if pred_ok and gold_ok:
        if gold_val == 0:
            return abs(pred_val) < tol
        return abs(pred_val - gold_val) / max(abs(gold_val), 1e-10) < tol
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


def load_test_data(data_path: str, n: int = 50, seed: int = 42):
    """Load and parse C3 training data correctly."""
    problems = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            inp = ex.get("input", "")

            # Parse [TEMPLATE: X] format
            template = ex.get("template", "")

            # Extract problem text (remove template tag)
            problem_text = re.sub(r'\[TEMPLATE:\s*\w+\]\s*', '', inp).strip()

            gold_expr = ex.get("output", "").strip()

            if template and problem_text:
                problems.append({
                    "problem_text": problem_text,
                    "template": template,
                    "gold_expression": gold_expr
                })

    # Sample
    random.seed(seed)
    if len(problems) > n:
        problems = random.sample(problems, n)

    return problems


# ============================================================
# Main Test
# ============================================================

def run_test(c2_path: str, c3_path: str, data_path: str, n_problems: int = 50):
    print("=" * 70)
    print(f"C2 + C3 + SYMPY BATCH TEST ({n_problems} MATH PROBLEMS)")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    c2 = C2Classifier(c2_path)
    c3 = C3Extractor(c3_path)

    # Load data
    print(f"\nLoading {n_problems} test problems...")
    problems = load_test_data(data_path, n_problems)
    print(f"Loaded {len(problems)} problems")

    # Show sample
    print(f"\nSample problem:")
    print(f"  Text: {problems[0]['problem_text'][:100]}...")
    print(f"  Template: {problems[0]['template']}")
    print(f"  Gold expr: {problems[0]['gold_expression']}")

    # C2: Batch classify
    print("\nRunning C2 batch classification...")
    texts = [p["problem_text"] for p in problems]
    c2_results = c2.classify_batch(texts)

    # C2 metrics
    c2_any_correct = 0
    c2_top1_correct = 0
    for i, (prob, c2r) in enumerate(zip(problems, c2_results)):
        gold = prob["template"]
        pred_labels = [lbl for lbl, _ in c2r["labels"]]
        if gold in pred_labels:
            c2_any_correct += 1
        if pred_labels and pred_labels[0] == gold:
            c2_top1_correct += 1

    print(f"C2 any-correct: {c2_any_correct}/{len(problems)} ({100*c2_any_correct/len(problems):.1f}%)")
    print(f"C2 top-1 correct: {c2_top1_correct}/{len(problems)} ({100*c2_top1_correct/len(problems):.1f}%)")

    # C3: Batch extract (using gold templates for evaluation)
    print("\nRunning C3 batch extraction...")
    templates = [p["template"] for p in problems]
    c3_exprs = c3.extract_batch(texts, templates, batch_size=8)

    # C3 metrics
    sympy_valid = 0
    numeric_correct = 0
    exact_match = 0

    for i, (prob, pred_expr) in enumerate(zip(problems, c3_exprs)):
        gold_expr = prob["gold_expression"]

        # Valid sympy?
        _, is_valid = eval_expr(pred_expr)
        if is_valid:
            sympy_valid += 1

        # Exact match?
        if pred_expr.strip() == gold_expr.strip():
            exact_match += 1

        # Numeric match?
        if numeric_match(pred_expr, gold_expr):
            numeric_correct += 1

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total: {len(problems)}")
    print()
    print("C2 Classifier:")
    print(f"  Any-correct:  {c2_any_correct}/{len(problems)} ({100*c2_any_correct/len(problems):.1f}%)")
    print(f"  Top-1:        {c2_top1_correct}/{len(problems)} ({100*c2_top1_correct/len(problems):.1f}%)")
    print()
    print("C3 Extractor (using gold templates):")
    print(f"  Sympy valid:  {sympy_valid}/{len(problems)} ({100*sympy_valid/len(problems):.1f}%)")
    print(f"  Exact match:  {exact_match}/{len(problems)} ({100*exact_match/len(problems):.1f}%)")
    print(f"  Numeric match: {numeric_correct}/{len(problems)} ({100*numeric_correct/len(problems):.1f}%)")
    print("=" * 70)

    # Show some examples
    print("\nSample predictions:")
    for i in range(min(5, len(problems))):
        prob = problems[i]
        pred = c3_exprs[i]
        gold = prob["gold_expression"]
        c2_labels = [l for l,_ in c2_results[i]["labels"][:3]]
        match = "✓" if numeric_match(pred, gold) else "✗"
        print(f"\n{i+1}. Template: {prob['template']}")
        print(f"   C2 predicted: {c2_labels}")
        print(f"   Gold: {gold}")
        print(f"   Pred: {pred} {match}")

    return {
        "c2_any": c2_any_correct,
        "c2_top1": c2_top1_correct,
        "c3_valid": sympy_valid,
        "c3_exact": exact_match,
        "c3_numeric": numeric_correct,
        "total": len(problems)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2-path", default="s3://mycelium-data/models/c2_heartbeat/")
    parser.add_argument("--c3-path", default="s3://mycelium-data/models/c3_extractor/")
    parser.add_argument("--data-path", default="s3://mycelium-data/c3_training/c3_train_fulltext.jsonl")
    parser.add_argument("--n-problems", type=int, default=50)
    parser.add_argument("--local-dir", default="/tmp/mycelium_test")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    c2_local = args.c2_path
    c3_local = args.c3_path
    data_local = args.data_path

    if args.c2_path.startswith("s3://"):
        c2_local = f"{args.local_dir}/c2"
        if not os.path.exists(f"{c2_local}/model.pt"):
            print(f"Downloading C2...")
            download_model_from_s3(args.c2_path, c2_local)

    if args.c3_path.startswith("s3://"):
        c3_local = f"{args.local_dir}/c3"
        if not os.path.exists(f"{c3_local}/config.json"):
            print(f"Downloading C3...")
            download_model_from_s3(args.c3_path, c3_local)

    if args.data_path.startswith("s3://"):
        data_local = f"{args.local_dir}/c3_test.jsonl"
        if not os.path.exists(data_local):
            print(f"Downloading test data...")
            download_from_s3(args.data_path, data_local)

    run_test(c2_local, c3_local, data_local, args.n_problems)


if __name__ == "__main__":
    main()
