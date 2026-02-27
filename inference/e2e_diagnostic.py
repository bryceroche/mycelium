#!/usr/bin/env python3
"""
E2E Diagnostic: Error Attribution for C1→C2→C3→Sympy Pipeline

Logs every intermediate to identify where failures occur:
- C1: How many regions detected per problem?
- C2: What operations predicted? (confusion tracking)
- C3: Correct/wrong/missing operands?
- Sympy: Crash, wrong answer, correct?

Run: python e2e_diagnostic.py --problems-path data/gsm8k_test.jsonl --max-problems 100
"""

import json
import re
import math
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Data Structures for Diagnostics
# ============================================================

@dataclass
class ProblemDiagnostic:
    """Full diagnostic for one problem."""
    problem_id: int
    problem_text: str
    gold_answer: float

    # C1 diagnostics
    c1_num_regions: int = 0
    c1_peak_values: List[float] = field(default_factory=list)
    c1_region_texts: List[str] = field(default_factory=list)

    # C2 diagnostics
    c2_predictions: List[str] = field(default_factory=list)
    c2_confidences: List[float] = field(default_factory=list)

    # C3 diagnostics
    c3_expressions: List[Dict] = field(default_factory=list)
    c3_num_valid: int = 0
    c3_num_invalid: int = 0

    # Sympy diagnostics
    sympy_results: List[float] = field(default_factory=list)
    sympy_crashes: int = 0

    # Final result
    predicted_answer: Optional[float] = None
    correct: bool = False

    # Error attribution
    failure_stage: str = "none"  # "c1_no_regions", "c2_wrong_op", "c3_wrong_args", "sympy_wrong", "correct"


# ============================================================
# Simple Operations
# ============================================================

OPERATIONS = {
    "ADD": lambda a, b: a + b,
    "SUB": lambda a, b: a - b,
    "MUL": lambda a, b: a * b,
    "DIV": lambda a, b: a / b if b != 0 else None,
}

def execute_op(op: str, args: List[float]) -> Optional[float]:
    op = op.upper()
    if op not in OPERATIONS or len(args) < 2:
        return None
    try:
        result = OPERATIONS[op](args[0], args[1])
        if result is not None and math.isfinite(result):
            return result
    except:
        pass
    return None

def extract_numbers(text: str) -> List[float]:
    numbers = []
    for match in re.finditer(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b', text):
        try:
            numbers.append(float(match.group(1).replace(',', '')))
        except:
            pass
    return numbers

def answer_matches(pred: float, gold: float, tol: float = 1e-6) -> bool:
    if abs(pred - gold) < tol:
        return True
    if abs(round(pred) - gold) < tol:
        return True
    return False


# ============================================================
# Clustering
# ============================================================

def cluster_relevance(relevance: np.ndarray, tokens: List[str],
                      min_height: float = 0.2, min_distance: int = 3) -> List[Dict]:
    """Cluster relevance field into regions."""
    if not HAS_SCIPY:
        # Simple threshold clustering
        regions = []
        in_region = False
        start = peak_idx = 0
        peak_val = 0

        for i, score in enumerate(relevance):
            if score > min_height:
                if not in_region:
                    in_region = True
                    start = i
                    peak_idx = i
                    peak_val = score
                elif score > peak_val:
                    peak_idx = i
                    peak_val = score
            else:
                if in_region:
                    text = " ".join(tokens[start:i])
                    regions.append({"start": start, "end": i, "peak": peak_val, "text": text})
                    in_region = False

        if in_region:
            text = " ".join(tokens[start:])
            regions.append({"start": start, "end": len(relevance), "peak": peak_val, "text": text})

        return regions

    # Scipy peak detection
    smoothed = gaussian_filter1d(relevance, sigma=1.5)
    peaks, props = find_peaks(smoothed, height=min_height, distance=min_distance)

    if len(peaks) == 0:
        # Try lower threshold
        peaks, props = find_peaks(smoothed, height=min_height * 0.5, distance=min_distance)

    regions = []
    threshold = min_height * 0.3

    for peak_idx in peaks:
        start = peak_idx
        while start > 0 and smoothed[start - 1] > threshold:
            start -= 1

        end = peak_idx
        while end < len(smoothed) - 1 and smoothed[end + 1] > threshold:
            end += 1

        text = " ".join(tokens[start:end + 1])
        regions.append({
            "start": start,
            "end": end,
            "peak": float(smoothed[peak_idx]),
            "text": text
        })

    return regions


# ============================================================
# Model Wrappers
# ============================================================

class Qwen2ForTokenRegression(torch.nn.Module):
    def __init__(self, config, base_model):
        super().__init__()
        self.qwen = base_model
        self.dropout = torch.nn.Dropout(0.1)
        self.regressor = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        return self.regressor(sequence_output).squeeze(-1)


class C1Model:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)

        # Load weights
        from safetensors.torch import load_file
        safetensors_path = Path(model_path) / "model.safetensors"
        state_dict = load_file(str(safetensors_path))

        # Build model
        base_model = AutoModel.from_config(config)
        mapped = {k.replace("qwen.", ""): v for k, v in state_dict.items() if k.startswith("qwen.")}
        base_model.load_state_dict(mapped, strict=False)

        self.model = Qwen2ForTokenRegression(config, base_model)
        if "regressor.weight" in state_dict:
            self.model.regressor.weight.data = state_dict["regressor.weight"]
            self.model.regressor.bias.data = state_dict["regressor.bias"]
        self.model.eval()

    def predict(self, text: str) -> Tuple[np.ndarray, List[str]]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            preds = self.model(**inputs)
        relevance = torch.clamp(preds[0], 0, 1).numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return relevance, tokens


class C2Model:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def classify(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        top_indices = torch.argsort(probs, descending=True)[:top_k]
        return [(self.id2label.get(i.item(), "UNK"), probs[i].item()) for i in top_indices]


class C3Model:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract(self, marked_text: str, operation: str) -> List[Tuple[float, str]]:
        prompt = f"[{operation}]\n{marked_text}\nArguments:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        args = []
        for line in generated.strip().split('\n')[:2]:
            match = re.match(r'([\d.]+)\|(\w+)', line.strip())
            if match:
                try:
                    args.append((float(match.group(1)), match.group(2)))
                except:
                    pass
        return args


# ============================================================
# Diagnostic Pipeline
# ============================================================

class DiagnosticPipeline:
    def __init__(self, c1_path: str, c2_path: str, c3_path: str):
        print("Loading models...")
        self.c1 = C1Model(c1_path)
        print("  C1 loaded")
        self.c2 = C2Model(c2_path)
        print(f"  C2 loaded ({len(self.c2.id2label)} classes)")
        self.c3 = C3Model(c3_path)
        print("  C3 loaded")

    def diagnose(self, problem_id: int, problem_text: str, gold_answer: float) -> ProblemDiagnostic:
        diag = ProblemDiagnostic(
            problem_id=problem_id,
            problem_text=problem_text[:200],
            gold_answer=gold_answer,
        )

        # === C1: Relevance ===
        relevance, tokens = self.c1.predict(problem_text)
        regions = cluster_relevance(relevance, tokens)

        diag.c1_num_regions = len(regions)
        diag.c1_peak_values = [r["peak"] for r in regions]
        diag.c1_region_texts = [r["text"][:50] for r in regions]

        if len(regions) == 0:
            diag.failure_stage = "c1_no_regions"
            return diag

        # === C2: Classification ===
        all_expressions = []

        for region in regions:
            # Mark region in text
            region_text = region["text"].replace("Ġ", " ").strip()
            if region_text in problem_text:
                marked = problem_text.replace(region_text, f"<SPAN>{region_text}</SPAN>", 1)
            else:
                marked = f"<SPAN>{region_text}</SPAN> {problem_text}"

            # Classify
            preds = self.c2.classify(marked, top_k=3)
            diag.c2_predictions.extend([p[0] for p in preds])
            diag.c2_confidences.extend([p[1] for p in preds])

            # === C3: Extraction ===
            for op, conf in preds[:2]:  # Top 2 templates
                args = self.c3.extract(marked, op)

                if len(args) >= 2:
                    result = execute_op(op, [args[0][0], args[1][0]])
                    expr = {
                        "op": op,
                        "args": [args[0][0], args[1][0]],
                        "result": result,
                    }
                    diag.c3_expressions.append(expr)

                    if result is not None:
                        diag.c3_num_valid += 1
                        diag.sympy_results.append(result)
                    else:
                        diag.c3_num_invalid += 1
                        diag.sympy_crashes += 1
                else:
                    diag.c3_num_invalid += 1

        # === Find best answer ===
        if diag.sympy_results:
            # Try each result
            for result in diag.sympy_results:
                if answer_matches(result, gold_answer):
                    diag.predicted_answer = result
                    diag.correct = True
                    diag.failure_stage = "correct"
                    return diag

            # No match - pick most common or first
            diag.predicted_answer = diag.sympy_results[0]

        # === Attribute failure ===
        if not diag.correct:
            # Check if gold answer could be computed from problem numbers
            problem_nums = extract_numbers(problem_text)

            # Try all binary ops on problem numbers
            could_compute = False
            for op in ["ADD", "SUB", "MUL", "DIV"]:
                for a in problem_nums:
                    for b in problem_nums:
                        if a != b:
                            r = execute_op(op, [a, b])
                            if r and answer_matches(r, gold_answer):
                                could_compute = True
                                break

            if diag.c1_num_regions == 1 and len(problem_nums) > 2:
                # Single blob when multiple ops needed
                diag.failure_stage = "c1_single_region"
            elif diag.c3_num_valid == 0:
                diag.failure_stage = "c3_no_valid"
            elif could_compute and not diag.correct:
                # Numbers are there but wrong ops
                diag.failure_stage = "c2_wrong_op"
            else:
                diag.failure_stage = "c3_wrong_args"

        return diag


# ============================================================
# Main
# ============================================================

def load_gsm8k(path: str, max_n: int = 100) -> List[Dict]:
    problems = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            match = re.search(r'####\s*([\d,.-]+)', data.get("answer", ""))
            gold = float(match.group(1).replace(",", "")) if match else None
            if gold is not None:
                problems.append({
                    "question": data.get("question", data.get("problem", "")),
                    "gold": gold,
                })
            if len(problems) >= max_n:
                break
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems-path", default="data/gsm8k_test.jsonl")
    parser.add_argument("--max-problems", type=int, default=50)
    parser.add_argument("--c1-path", default="/opt/dlami/nvme/models/c1")
    parser.add_argument("--c2-path", default="/opt/dlami/nvme/models/c2")
    parser.add_argument("--c3-path", default="/opt/dlami/nvme/models/qwen05b_extractor_multispan/final")
    parser.add_argument("--output", default="diagnostic_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("E2E DIAGNOSTIC: Error Attribution")
    print("=" * 60)

    # Load pipeline
    pipeline = DiagnosticPipeline(args.c1_path, args.c2_path, args.c3_path)

    # Load problems
    print(f"\nLoading problems from {args.problems_path}...")
    problems = load_gsm8k(args.problems_path, args.max_problems)
    print(f"Loaded {len(problems)} problems")

    # Run diagnostics
    print(f"\nRunning diagnostics...")
    diagnostics = []

    for i, prob in enumerate(problems):
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(problems)}...")

        diag = pipeline.diagnose(i, prob["question"], prob["gold"])
        diagnostics.append(diag)

    # === Aggregate Stats ===
    print("\n" + "=" * 60)
    print("ERROR ATTRIBUTION SUMMARY")
    print("=" * 60)

    total = len(diagnostics)
    correct = sum(1 for d in diagnostics if d.correct)

    print(f"\nOverall: {correct}/{total} correct ({100*correct/total:.1f}%)")

    # Failure stage breakdown
    stage_counts = Counter(d.failure_stage for d in diagnostics)
    print(f"\nFailure Attribution:")
    for stage, count in sorted(stage_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"  {stage:20s}: {count:3d} ({pct:5.1f}%) {bar}")

    # C1 stats
    c1_regions = [d.c1_num_regions for d in diagnostics]
    print(f"\nC1 (Clustering):")
    print(f"  Mean regions/problem: {np.mean(c1_regions):.2f}")
    print(f"  Single-region problems: {sum(1 for r in c1_regions if r == 1)}")
    print(f"  Zero-region problems: {sum(1 for r in c1_regions if r == 0)}")
    print(f"  Multi-region (2+): {sum(1 for r in c1_regions if r >= 2)}")

    # C2 stats
    c2_ops = []
    for d in diagnostics:
        c2_ops.extend(d.c2_predictions)
    print(f"\nC2 (Classification):")
    print(f"  Total predictions: {len(c2_ops)}")
    op_counts = Counter(c2_ops)
    print(f"  Top 5 operations:")
    for op, count in op_counts.most_common(5):
        print(f"    {op}: {count}")

    # C3 stats
    c3_valid = sum(d.c3_num_valid for d in diagnostics)
    c3_invalid = sum(d.c3_num_invalid for d in diagnostics)
    print(f"\nC3 (Extraction):")
    print(f"  Valid expressions: {c3_valid}")
    print(f"  Invalid expressions: {c3_invalid}")
    print(f"  Success rate: {100*c3_valid/(c3_valid+c3_invalid+0.001):.1f}%")

    # Sympy stats
    sympy_crashes = sum(d.sympy_crashes for d in diagnostics)
    print(f"\nSympy (Execution):")
    print(f"  Crashes: {sympy_crashes}")

    # Save full diagnostics
    print(f"\nSaving full diagnostics to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump([asdict(d) for d in diagnostics], f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
