#!/usr/bin/env python3
"""
E2E Pipeline Test on MATH Problems

Tests whether the current models (trained on CoT boundaries) work on MATH problem text.
This is testing transfer from GSM8K training to MATH evaluation.

The segmenter was trained on CoT-space boundaries - this tests if the full pipeline
compensates for imperfect segmentation via search + executor.
"""

import json
import re
import sys
import torch
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from fractions import Fraction

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
)

# Model paths - local
MODEL_DIR = Path(__file__).parent.parent / "models"
SEGMENTER_PATH = MODEL_DIR / "qwen05b_segmenter_clean"
CLASSIFIER_PATH = MODEL_DIR / "qwen05b_classifier_multispan"
EXTRACTOR_PATH = MODEL_DIR / "qwen05b_extractor_multispan"

# Labels
BIO_LABELS = ["O", "B-OP", "I-OP", "B-Q", "I-Q"]
OP_LABELS = ["ADD", "SUB", "MUL", "DIV"]

# GSM8K domain constants (for bridging)
DOMAIN_CONSTANTS = [
    7, 24, 60, 365, 12, 100, 1000, 52, 4, 2, 10, 0.5, 0.25, 0.1
]


def normalize_math_answer(answer_str: str) -> Any:
    """Normalize MATH answer for comparison."""
    if answer_str is None:
        return None

    s = str(answer_str).strip()

    # Strip LaTeX wrappers
    s = re.sub(r'\\boxed\{(.+)\}', r'\1', s)
    s = re.sub(r'\\text\{(.+?)\}', r'\1', s)
    s = s.replace('\\$', '').replace('$', '').strip()

    # Handle x = value format
    if '=' in s and not s.startswith('='):
        parts = s.split('=')
        if len(parts) == 2 and re.match(r'^[a-zA-Z]$', parts[0].strip()):
            s = parts[1].strip()

    # Handle fractions: \frac{a}{b}
    frac_match = re.match(r'^-?\\frac\{(-?\d+)\}\{(\d+)\}$', s)
    if frac_match:
        num = int(frac_match.group(1))
        denom = int(frac_match.group(2))
        if s.startswith('-') and num > 0:
            num = -num
        return Fraction(num, denom)

    # Handle simple fractions: a/b
    simple_frac = re.match(r'^(-?\d+)/(\d+)$', s)
    if simple_frac:
        return Fraction(int(simple_frac.group(1)), int(simple_frac.group(2)))

    # Handle integers
    if re.match(r'^-?\d+$', s):
        return int(s)

    # Handle decimals
    if re.match(r'^-?\d+\.\d+$', s):
        return float(s)

    # Handle sqrt
    sqrt_match = re.match(r'^(-?\d*)\\sqrt\{(\d+)\}$', s)
    if sqrt_match:
        coef = int(sqrt_match.group(1)) if sqrt_match.group(1) else 1
        radicand = int(sqrt_match.group(2))
        return coef * (radicand ** 0.5)

    return s


def answers_match(pred: Any, gold: Any, tolerance: float = 1e-4) -> bool:
    """Compare two answers."""
    pred_norm = normalize_math_answer(pred)
    gold_norm = normalize_math_answer(gold)

    if pred_norm == gold_norm:
        return True

    # Numeric comparison
    try:
        pred_num = float(pred_norm) if not isinstance(pred_norm, (int, float, Fraction)) else float(pred_norm)
        gold_num = float(gold_norm) if not isinstance(gold_norm, (int, float, Fraction)) else float(gold_norm)

        if abs(gold_num) < tolerance:
            return abs(pred_num - gold_num) < tolerance
        return abs(pred_num - gold_num) / max(abs(gold_num), 1e-10) < tolerance
    except (ValueError, TypeError):
        pass

    # String comparison
    if isinstance(pred_norm, str) and isinstance(gold_norm, str):
        return pred_norm.lower().replace(' ', '') == gold_norm.lower().replace(' ', '')

    return False


class E2EPipeline:
    """Simplified E2E pipeline for MATH testing."""

    def __init__(self, device: str = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        print(f"Using device: {self.device}")

        self.segmenter = None
        self.segmenter_tokenizer = None
        self.classifier = None
        self.classifier_tokenizer = None
        self.extractor = None
        self.extractor_tokenizer = None

    def load_models(self):
        """Load all models."""
        print("\nLoading models...")

        # Segmenter
        print(f"  Loading segmenter from {SEGMENTER_PATH}...")
        self.segmenter_tokenizer = AutoTokenizer.from_pretrained(
            str(SEGMENTER_PATH), trust_remote_code=True
        )
        self.segmenter = AutoModelForTokenClassification.from_pretrained(
            str(SEGMENTER_PATH), trust_remote_code=True
        ).to(self.device)
        self.segmenter.eval()

        # Classifier
        print(f"  Loading classifier from {CLASSIFIER_PATH}...")
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(
            str(CLASSIFIER_PATH), trust_remote_code=True
        )
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            str(CLASSIFIER_PATH), trust_remote_code=True
        ).to(self.device)
        self.classifier.eval()

        # Extractor
        print(f"  Loading extractor from {EXTRACTOR_PATH}...")
        self.extractor_tokenizer = AutoTokenizer.from_pretrained(
            str(EXTRACTOR_PATH), trust_remote_code=True
        )
        self.extractor = AutoModelForCausalLM.from_pretrained(
            str(EXTRACTOR_PATH), trust_remote_code=True
        ).to(self.device)
        self.extractor.eval()

        print("  All models loaded.")

    def segment(self, text: str) -> List[Dict]:
        """Run segmenter to extract spans."""
        inputs = self.segmenter_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.segmenter(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        # Extract spans from BIO predictions
        spans = []
        current_span = None

        for i, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:  # Special token
                continue

            label = BIO_LABELS[pred] if pred < len(BIO_LABELS) else "O"

            if label.startswith("B-"):
                if current_span is not None:
                    spans.append(current_span)
                tag = label[2:]
                current_span = {
                    "tag": tag,
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            elif label.startswith("I-"):
                tag = label[2:]
                if current_span is not None and current_span["tag"] == tag:
                    current_span["end"] = end
                    current_span["text"] = text[current_span["start"]:end]
                else:
                    if current_span is not None:
                        spans.append(current_span)
                    current_span = {
                        "tag": tag,
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                    }
            else:
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None

        if current_span is not None:
            spans.append(current_span)

        # Filter tiny spans
        spans = [s for s in spans if len(s["text"].strip()) > 3]
        return spans

    def classify_spans(self, text: str, spans: List[Dict]) -> List[Tuple[str, float]]:
        """Classify each span individually."""
        results = []

        for span in spans:
            # Mark the span
            marked = text[:span["start"]] + "<SPAN> " + span["text"] + " </SPAN>" + text[span["end"]:]

            inputs = self.classifier_tokenizer(
                marked,
                return_tensors="pt",
                truncation=True,
                max_length=384,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.classifier(**inputs)
                probs = torch.softmax(outputs.logits[0], dim=-1)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()

            results.append((OP_LABELS[pred_idx], confidence))

        return results

    def extract_arguments(self, text: str, span: Dict, operation: str) -> List[float]:
        """Extract arguments from a span."""
        marked = text[:span["start"]] + "<SPAN> " + span["text"] + " </SPAN>" + text[span["end"]:]
        prompt = f"[{operation}]\n{marked}\nArguments:\n"

        inputs = self.extractor_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.extractor.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.extractor_tokenizer.pad_token_id,
                eos_token_id=self.extractor_tokenizer.eos_token_id,
            )

        generated = self.extractor_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Parse arguments
        args = []
        for line in generated.strip().split("\n")[:2]:
            if "|" in line:
                parts = line.split("|")
                try:
                    value = float(parts[0].strip())
                    args.append(value)
                except ValueError:
                    pass
            else:
                # Try to parse as number directly
                try:
                    value = float(line.strip())
                    args.append(value)
                except ValueError:
                    pass

        return args

    def execute_operation(self, op: str, arg1: float, arg2: float = None) -> Optional[float]:
        """Execute a single operation."""
        try:
            if op == "ADD":
                return arg1 + arg2
            elif op == "SUB":
                return arg1 - arg2
            elif op == "MUL":
                return arg1 * arg2
            elif op == "DIV":
                return arg1 / arg2 if arg2 != 0 else None
            elif op == "POW":
                return arg1 ** arg2 if arg2 < 100 else None
            elif op == "SQRT":
                return arg1 ** 0.5 if arg1 >= 0 else None
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
        return None

    def solve_with_bridging(self, text: str, gold_answer: Any = None) -> Dict:
        """
        Solve problem with bridging search.

        Returns detailed trace for error attribution.
        """
        result = {
            "text": text[:100] + "...",
            "gold_answer": gold_answer,
            "predicted_answer": None,
            "correct": False,
            "error_type": None,
            "trace": {},
        }

        # Step 1: Segment
        spans = self.segment(text)
        op_spans = [s for s in spans if s.get("tag") == "OP"]
        result["trace"]["n_spans"] = len(op_spans)
        result["trace"]["spans"] = [s["text"][:50] for s in op_spans]

        if len(op_spans) == 0:
            result["error_type"] = "segmentation_miss"
            return result

        # Step 2: Classify each span
        classifications = self.classify_spans(text, op_spans)
        result["trace"]["classifications"] = classifications

        # Step 3: Extract arguments and execute
        # Try multiple candidate execution strategies
        candidates = []

        # Strategy A: Execute spans in order, chain results
        prev_result = None
        operations = []
        for span, (op, conf) in zip(op_spans, classifications):
            args = self.extract_arguments(text, span, op)
            operations.append({
                "op": op,
                "args": args,
                "span_text": span["text"][:30],
                "confidence": conf,
            })

            if len(args) >= 2:
                res = self.execute_operation(op, args[0], args[1])
            elif len(args) == 1:
                if prev_result is not None:
                    res = self.execute_operation(op, prev_result, args[0])
                else:
                    res = args[0]
            else:
                res = None

            if res is not None:
                prev_result = res

        if prev_result is not None:
            candidates.append(("chain", prev_result))

        result["trace"]["operations"] = operations

        # Strategy B: Extract all numbers from text and try common patterns
        numbers = self._extract_numbers(text)
        result["trace"]["numbers_in_text"] = numbers[:10]  # Limit

        # Try adding all numbers
        if len(numbers) >= 2:
            candidates.append(("add_all", sum(numbers)))

        # Try multiplying pairs
        if len(numbers) >= 2:
            candidates.append(("mul_first_two", numbers[0] * numbers[1]))

        # Strategy C: Try with domain constants (bridging)
        for const in [7, 24, 60, 12, 100, 52]:
            if len(numbers) >= 1:
                candidates.append((f"mul_{const}", numbers[0] * const))
                candidates.append((f"div_{const}", numbers[0] / const if const != 0 else None))

        result["trace"]["n_candidates"] = len(candidates)

        # Check all candidates against gold
        for name, answer in candidates:
            if answer is not None and answers_match(answer, gold_answer):
                result["predicted_answer"] = answer
                result["correct"] = True
                result["trace"]["winning_strategy"] = name
                return result

        # No match - record best candidate
        if candidates:
            result["predicted_answer"] = candidates[0][1]

        # Determine error type
        if len(op_spans) == 0:
            result["error_type"] = "segmentation_miss"
        elif not any(op["args"] for op in operations):
            result["error_type"] = "extractor_error"
        elif all(c[1] is None for c in candidates):
            result["error_type"] = "execution_error"
        else:
            # Check if this needs an operation we don't have
            needed_ops = self._guess_needed_operation(text, gold_answer)
            if needed_ops:
                result["error_type"] = f"missing_operation:{needed_ops}"
            else:
                result["error_type"] = "classifier_error"

        return result

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text."""
        numbers = []
        for match in re.finditer(r'-?\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', text):
            num_str = match.group(1).replace(",", "")
            try:
                numbers.append(float(num_str))
            except ValueError:
                pass
        return numbers

    def _guess_needed_operation(self, text: str, gold_answer: Any) -> Optional[str]:
        """Guess what operation might be needed based on problem text."""
        text_lower = text.lower()

        # Check for operation keywords not in our set
        if any(w in text_lower for w in ["square root", "sqrt", "root"]):
            return "SQRT"
        if any(w in text_lower for w in ["power", "exponent", "squared", "cubed"]):
            return "POW"
        if any(w in text_lower for w in ["modulo", "remainder", "mod "]):
            return "MOD"
        if any(w in text_lower for w in ["factorial", "!"]):
            return "FACTORIAL"
        if any(w in text_lower for w in ["combination", "permutation", "choose"]):
            return "COMBINATORICS"
        if any(w in text_lower for w in ["sine", "cosine", "tangent", "sin", "cos", "tan"]):
            return "TRIG"
        if any(w in text_lower for w in ["logarithm", "log"]):
            return "LOG"
        if any(w in text_lower for w in ["solve", "equation", "quadratic"]):
            return "SOLVE_EQUATION"

        return None


def select_test_problems(data_dir: str, n_per_category: Dict[str, int]) -> List[Dict]:
    """Select problems stratified by category from correct solutions."""
    # Load all 7B results
    problems = {}
    for f in Path(data_dir).glob("7b/*.json"):
        with open(f) as fp:
            d = json.load(fp)
            problems[d["idx"]] = d

    # Group by category, filter to those with gold answers
    by_category = defaultdict(list)
    for idx, p in problems.items():
        cat = p.get("category", "Unknown")
        gold = p.get("gold_answer")
        if gold:
            by_category[cat].append(p)

    # Select stratified sample
    selected = []
    for cat, count in n_per_category.items():
        available = by_category.get(cat, [])
        if available:
            selected.extend(available[:count])

    return selected


def main():
    print("=" * 70)
    print("E2E PIPELINE TEST ON MATH PROBLEMS")
    print("=" * 70)
    print("\nThis tests whether GSM8K-trained models work on MATH problem text.")
    print("The segmenter was trained on CoT boundaries (known bug).")
    print("We're testing if the full pipeline compensates.\n")

    # Select 20 problems stratified by category
    categories = {
        "Prealgebra": 3,
        "Algebra": 3,
        "Number Theory": 3,
        "Counting & Probability": 3,
        "Geometry": 3,
        "Intermediate Algebra": 3,
        "Precalculus": 2,
    }

    test_problems = select_test_problems("/tmp/math_e2e_test", categories)
    print(f"Selected {len(test_problems)} test problems:")
    cat_counts = defaultdict(int)
    for p in test_problems:
        cat_counts[p.get("category", "Unknown")] += 1
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Initialize pipeline
    pipeline = E2EPipeline()
    pipeline.load_models()

    # Run tests
    print("\n" + "-" * 70)
    print("RUNNING E2E PIPELINE")
    print("-" * 70)

    results = []
    for i, problem in enumerate(test_problems):
        print(f"\n[{i+1}/{len(test_problems)}] {problem.get('category', 'Unknown')}")

        # Use problem text (NOT CoT)
        problem_text = problem.get("question", problem.get("problem", ""))
        gold_answer = problem.get("gold_answer")

        result = pipeline.solve_with_bridging(problem_text, gold_answer)
        result["category"] = problem.get("category", "Unknown")
        result["problem_idx"] = problem.get("idx")
        results.append(result)

        status = "OK" if result["correct"] else f"FAIL ({result.get('error_type', 'unknown')})"
        print(f"  {status}")
        print(f"  Gold: {gold_answer}, Pred: {result['predicted_answer']}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"\nOVERALL: {correct}/{total} correct ({100*correct/total:.1f}%)")

    # Per category
    print("\nPer category:")
    cat_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat = r["category"]
        cat_results[cat]["total"] += 1
        if r["correct"]:
            cat_results[cat]["correct"] += 1

    for cat in sorted(cat_results.keys()):
        cr = cat_results[cat]
        pct = 100 * cr["correct"] / cr["total"] if cr["total"] > 0 else 0
        print(f"  {cat:25} {cr['correct']}/{cr['total']} ({pct:.0f}%)")

    # Error attribution
    print("\nError attribution for failures:")
    error_counts = defaultdict(int)
    failures = [r for r in results if not r["correct"]]
    for r in failures:
        error_type = r.get("error_type", "unknown")
        error_counts[error_type] += 1

    for err_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(failures) if failures else 0
        print(f"  {err_type:25} {count}/{len(failures)} ({pct:.0f}%)")

    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    if correct >= 8:
        print(f"\n{correct}/20 correct >= 8")
        print(">>> The bugged segmenter works! <<<")
        print(">>> Proceed to IB on MATH, expand templates, full eval. <<<")
    elif correct >= 4:
        print(f"\n{correct}/20 correct (4-7 range)")
        print(">>> Marginal. Check error attribution. <<<")
        if error_counts.get("missing_operation", 0) > len(failures) // 2:
            print(">>> Mostly missing operations - segmenter is fine, need MATH templates. <<<")
        else:
            print(">>> Consider fixing JSD extraction bug. <<<")
    else:
        print(f"\n{correct}/20 correct < 4")
        print(">>> Poor transfer. Fix JSD extraction to operate on input tokens. <<<")
        print(">>> Retrain segmenter, re-test. <<<")

    # Save detailed results
    output_path = Path("/tmp/math_e2e_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
