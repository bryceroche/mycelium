#!/usr/bin/env python3
"""
Minimal E2E test: C1 → C2 → C3 → sympy executor

Tests the trained models without the deprecated heuristics.
"""

import re
import math
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)


# ============================================================
# Simple Number Extraction (no word-to-num, just regex)
# ============================================================

def extract_numbers_simple(text: str) -> list[float]:
    """Extract numeric values from text using regex."""
    numbers = []
    # Match integers and decimals, handle commas
    for match in re.finditer(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b', text):
        num_str = match.group(1).replace(',', '')
        try:
            numbers.append(float(num_str))
        except ValueError:
            pass
    return numbers


# ============================================================
# Symbolic Executor
# ============================================================

OPERATIONS = {
    "ADD": lambda a, b: a + b,
    "SUB": lambda a, b: a - b,
    "MUL": lambda a, b: a * b,
    "DIV": lambda a, b: a / b if b != 0 else None,
}

def execute_operation(op: str, args: list[float]) -> float | None:
    """Execute a single operation."""
    op = op.upper()
    if op not in OPERATIONS or len(args) < 2:
        return None
    try:
        result = OPERATIONS[op](args[0], args[1])
        if result is not None and math.isfinite(result):
            return result
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return None


# ============================================================
# Model Loaders
# ============================================================

class C1Segmenter:
    """BIO token tagger for span extraction."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label
        print(f"C1 loaded: {len(self.id2label)} labels")

    def predict(self, text: str) -> list[dict]:
        """Returns list of spans with text and char offsets."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, return_offsets_mapping=True
        )
        offset_mapping = inputs.pop("offset_mapping")[0]

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)[0]

        spans = []
        current_start = None
        current_end = None

        for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
            label = self.id2label.get(pred.item(), "O")

            if label.startswith("B"):
                if current_start is not None:
                    span_text = text[current_start:current_end]
                    spans.append({
                        "text": span_text,
                        "start": current_start,
                        "end": current_end,
                        "numbers": extract_numbers_simple(span_text)
                    })
                current_start = offset[0].item()
                current_end = offset[1].item()
            elif label.startswith("I") and current_start is not None:
                current_end = offset[1].item()
            else:
                if current_start is not None:
                    span_text = text[current_start:current_end]
                    spans.append({
                        "text": span_text,
                        "start": current_start,
                        "end": current_end,
                        "numbers": extract_numbers_simple(span_text)
                    })
                    current_start = None

        # Final span
        if current_start is not None:
            span_text = text[current_start:current_end]
            spans.append({
                "text": span_text,
                "start": current_start,
                "end": current_end,
                "numbers": extract_numbers_simple(span_text)
            })

        return spans


class C2Classifier:
    """Span group → operation classifier."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label
        print(f"C2 loaded: {len(self.id2label)} classes")

    def classify(self, text: str) -> tuple[str, float]:
        """Classify text, return (label, confidence)."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_id].item()

        return self.id2label.get(pred_id, "UNKNOWN"), confidence


class C3Extractor:
    """Generative argument extractor."""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"C3 loaded")

    def extract(self, marked_text: str, operation: str) -> list[tuple[float, str]]:
        """Extract arguments from marked text."""
        prompt = f"[{operation}]\n{marked_text}\nArguments:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Parse output: "value|source" format
        args = []
        for line in generated.strip().split('\n')[:3]:
            match = re.match(r'([\d.]+)\|(\w+)', line.strip())
            if match:
                try:
                    val = float(match.group(1))
                    source = match.group(2)
                    args.append((val, source))
                except ValueError:
                    pass

        return args


# ============================================================
# E2E Test
# ============================================================

def mark_spans(text: str, spans: list[dict]) -> str:
    """Insert <SPAN></SPAN> markers around spans."""
    marked = text
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        marked = marked[:span["start"]] + "<SPAN>" + marked[span["start"]:span["end"]] + "</SPAN>" + marked[span["end"]:]
    return marked


def run_e2e_test(c1, c2, c3, problem: str, gold_answer: float = None):
    """Run full pipeline on one problem."""
    print(f"\n{'='*60}")
    print(f"Problem: {problem[:80]}...")
    if gold_answer:
        print(f"Gold answer: {gold_answer}")
    print('='*60)

    # Step 1: C1 Segmentation
    spans = c1.predict(problem)
    print(f"\n[C1] Segmentation: {len(spans)} spans")
    for s in spans:
        print(f"  • '{s['text']}' → numbers: {s['numbers']}")

    if not spans:
        print("  No spans found!")
        return None

    # Step 2: Create marked text and classify with C2
    marked = mark_spans(problem, spans)
    label, conf = c2.classify(marked)
    print(f"\n[C2] Classification: {label} (confidence: {conf:.3f})")

    # Step 3: Extract arguments with C3
    args = c3.extract(marked, label)
    print(f"\n[C3] Extraction:")
    for val, source in args:
        print(f"  • {val} ({source})")

    # Step 4: Execute operation
    if len(args) >= 2:
        arg_values = [a[0] for a in args[:2]]
        result = execute_operation(label, arg_values)
        print(f"\n[Executor] {label}({arg_values[0]}, {arg_values[1]}) = {result}")

        if gold_answer and result:
            correct = abs(result - gold_answer) < 0.01
            print(f"Correct: {'YES' if correct else 'NO'}")

        return result
    else:
        print(f"\n[Executor] Not enough arguments for {label}")
        return None


def main():
    print("\n" + "="*60)
    print("  MYCELIUM E2E MINIMAL TEST")
    print("  C1 (Segmenter) → C2 (Classifier) → C3 (Extractor) → Sympy")
    print("="*60)

    # Model paths
    c1_path = "models/c1_relevance_v2"
    c2_path = "models/c2_ib_templates_frozen_v1"
    c3_path = "models/c3_extractor_partial_freeze_v1"

    # Check models exist
    for path in [c1_path, c2_path, c3_path]:
        if not Path(path).exists():
            print(f"ERROR: Model not found at {path}")
            print("Download from S3 first:")
            print(f"  aws s3 cp --recursive s3://mycelium-data/{path}/ {path}/")
            return

    # Load models
    print("\nLoading models...")
    c1 = C1Segmenter(c1_path)
    c2 = C2Classifier(c2_path)
    c3 = C3Extractor(c3_path)

    # Test problems
    test_problems = [
        {
            "problem": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "gold": 72
        },
        {
            "problem": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            "gold": 10
        },
        {
            "problem": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
            "gold": 5
        },
        {
            "problem": "A baker sold 5 cakes for $20 each. How much did the baker earn?",
            "gold": 100
        },
    ]

    results = []
    for test in test_problems:
        result = run_e2e_test(c1, c2, c3, test["problem"], test.get("gold"))
        results.append({
            "problem": test["problem"][:50] + "...",
            "gold": test.get("gold"),
            "predicted": result,
            "correct": result and test.get("gold") and abs(result - test["gold"]) < 0.01
        })

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    correct = sum(1 for r in results if r["correct"])
    print(f"Correct: {correct}/{len(results)}")
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} gold={r['gold']}, pred={r['predicted']}")


if __name__ == "__main__":
    main()
