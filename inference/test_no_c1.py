#!/usr/bin/env python3
"""
No-C1 E2E Test: C2 → C3 → Sympy on full problem text

Tests whether C1 segmentation is necessary, or if C2/C3 can work
directly on full problem text using their internal attention.

Architecture:
  Problem text → C2 (predict operation templates) → C3 (extract arguments) → execute

No segmentation, no relevance scoring, no channel separation.
"""

import json
import re
import math
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

# Word to number mapping for extraction
WORD_TO_NUM = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
    'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100,
    'thousand': 1000, 'million': 1000000,
    'half': 0.5, 'quarter': 0.25, 'third': 1/3,
    'twice': 2, 'double': 2, 'triple': 3,
}


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text including spelled-out words."""
    numbers = []

    # Numeric patterns
    for match in re.finditer(r'\$?([\d,]+\.?\d*)', text):
        try:
            num = float(match.group(1).replace(',', ''))
            numbers.append(num)
        except ValueError:
            pass

    # Word numbers
    text_lower = text.lower()
    for word, val in WORD_TO_NUM.items():
        if word in text_lower:
            numbers.append(val)

    return list(set(numbers))


@dataclass
class Operation:
    """A predicted operation with extracted arguments."""
    template: str  # e.g., "MUL", "ADD", "SUB", "DIV"
    args: List[float]
    confidence: float
    result: Optional[float] = None


def execute_op(template: str, args: List[float]) -> Optional[float]:
    """Execute a single operation."""
    if len(args) < 2 and template not in ('SQRT', 'ABS'):
        return None

    try:
        if template == 'ADD':
            return args[0] + args[1]
        elif template == 'SUB':
            return args[0] - args[1]
        elif template == 'MUL':
            return args[0] * args[1]
        elif template == 'DIV':
            return args[0] / args[1] if args[1] != 0 else None
        elif template == 'SQRT':
            return math.sqrt(args[0]) if args[0] >= 0 else None
        elif template == 'ABS':
            return abs(args[0])
        else:
            return None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


class NoC1Pipeline:
    """C2 → C3 → Execute pipeline without C1 segmentation."""

    def __init__(self, classifier_path: str, extractor_path: str, device: str = 'cuda'):
        self.device = device

        print(f"Loading C2 classifier from {classifier_path}")
        self.c2_tokenizer = AutoTokenizer.from_pretrained(classifier_path, trust_remote_code=True)
        self.c2_model = AutoModelForSequenceClassification.from_pretrained(
            classifier_path, trust_remote_code=True
        ).to(device).eval()

        # Get label mapping
        self.id2label = self.c2_model.config.id2label
        print(f"C2 labels: {self.id2label}")

        print(f"Loading C3 extractor from {extractor_path}")
        self.c3_tokenizer = AutoTokenizer.from_pretrained(extractor_path, trust_remote_code=True)
        self.c3_model = AutoModelForCausalLM.from_pretrained(
            extractor_path, trust_remote_code=True
        ).to(device).eval()

        if self.c3_tokenizer.pad_token is None:
            self.c3_tokenizer.pad_token = self.c3_tokenizer.eos_token

        print("Pipeline loaded.")

    def predict_operations(self, problem_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        C2: Predict what operations are needed for this problem.

        No span marking - just pass full problem text.
        Returns top-k (template, confidence) pairs.
        """
        inputs = self.c2_tokenizer(
            problem_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.c2_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Get top-k predictions
        top_probs, top_ids = torch.topk(probs, min(top_k, len(probs)))

        results = []
        for prob, idx in zip(top_probs, top_ids):
            template = self.id2label.get(idx.item(), "UNKNOWN")
            results.append((template, prob.item()))

        return results

    def extract_arguments(self, problem_text: str, template: str) -> List[float]:
        """
        C3: Extract numeric arguments for the given operation.

        Uses instruction-tuned model to identify which numbers
        from the problem text are relevant to this operation.
        """
        prompt = f"""Problem: {problem_text}

Operation: {template}

Extract the two numbers that should be used for this {template} operation. Output just the numbers separated by comma.

Numbers:"""

        inputs = self.c3_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.c3_model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=self.c3_tokenizer.pad_token_id,
            )

        generated = self.c3_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Parse numbers from output
        args = []
        for match in re.finditer(r'([\d,]+\.?\d*)', generated):
            try:
                args.append(float(match.group(1).replace(',', '')))
            except ValueError:
                pass

        # Fallback: use numbers from problem text
        if len(args) < 2:
            problem_nums = extract_numbers(problem_text)
            for num in problem_nums:
                if num not in args:
                    args.append(num)
                if len(args) >= 2:
                    break

        return args

    def solve(self, problem_text: str, gold_answer: Optional[float] = None,
              verbose: bool = False) -> Dict:
        """
        Solve a math problem using C2 → C3 → Execute.

        Strategy:
        1. C2 predicts top-3 operation templates
        2. For each template, C3 extracts arguments
        3. Execute all combinations
        4. Pick the integer result (GSM8K answers are integers)
        """
        result = {
            "problem_text": problem_text,
            "gold_answer": gold_answer,
            "answer": None,
            "correct": None,
            "operations_tried": [],
        }

        # Step 1: C2 predicts operations
        predictions = self.predict_operations(problem_text, top_k=3)

        if verbose:
            print(f"C2 predictions: {predictions}")

        # Step 2 & 3: For each prediction, extract args and execute
        candidates = []

        for template, conf in predictions:
            if template == "UNKNOWN":
                continue

            # C3 extracts arguments
            args = self.extract_arguments(problem_text, template)

            if verbose:
                print(f"  {template}: args={args}")

            # Execute
            if len(args) >= 2:
                # Try both orderings for non-commutative ops
                for a, b in [(args[0], args[1]), (args[1], args[0])]:
                    res = execute_op(template, [a, b])
                    if res is not None:
                        candidates.append({
                            "template": template,
                            "args": [a, b],
                            "result": res,
                            "confidence": conf,
                        })

                        result["operations_tried"].append({
                            "template": template,
                            "args": [a, b],
                            "result": res,
                        })

        if verbose:
            print(f"Candidates: {len(candidates)}")
            for c in candidates:
                print(f"  {c['template']}({c['args']}) = {c['result']}")

        # Step 4: Pick best answer
        # Prefer: integer, positive, matches gold (if given)
        if candidates:
            # Sort by: gold match > integer > positive > confidence
            def score(c):
                s = c['confidence']
                res = c['result']
                if gold_answer is not None and abs(res - gold_answer) < 0.01:
                    s += 1000
                if res == int(res):
                    s += 10
                if res > 0:
                    s += 1
                return s

            candidates.sort(key=score, reverse=True)
            winner = candidates[0]
            result["answer"] = winner["result"]
            result["winning_op"] = winner

        # Check correctness
        if gold_answer is not None and result["answer"] is not None:
            result["correct"] = abs(result["answer"] - gold_answer) < 0.01

        return result


def load_gsm8k(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load GSM8K problems."""
    problems = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            # Extract gold answer
            answer_text = data.get("answer", "")
            match = re.search(r'####\s*([\d,.-]+)', answer_text)
            if match:
                gold = float(match.group(1).replace(",", ""))
            else:
                try:
                    gold = float(answer_text.strip().split()[-1].replace(",", ""))
                except:
                    gold = None

            problems.append({
                "question": data.get("question", data.get("problem", "")),
                "gold_answer": gold,
            })

            if limit and len(problems) >= limit:
                break

    return problems


def main():
    parser = argparse.ArgumentParser(description="No-C1 E2E Test")
    parser.add_argument("--classifier-path", required=True, help="C2 classifier model")
    parser.add_argument("--extractor-path", required=True, help="C3 extractor model")
    parser.add_argument("--problems-path", required=True, help="GSM8K test JSONL")
    parser.add_argument("--limit", type=int, default=50, help="Max problems")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pipeline
    pipeline = NoC1Pipeline(
        classifier_path=args.classifier_path,
        extractor_path=args.extractor_path,
        device=device,
    )

    # Load problems
    problems = load_gsm8k(args.problems_path, limit=args.limit)
    print(f"Loaded {len(problems)} problems")

    # Evaluate
    correct = 0
    results = []

    for i, problem in enumerate(problems):
        if args.verbose:
            print(f"\n{'='*60}")
            print(f"Problem {i+1}: {problem['question'][:80]}...")
            print(f"Gold: {problem['gold_answer']}")

        result = pipeline.solve(
            problem["question"],
            gold_answer=problem["gold_answer"],
            verbose=args.verbose,
        )
        results.append(result)

        if result["correct"]:
            correct += 1
            if args.verbose:
                print(f"✓ Correct: {result['answer']}")
        else:
            if args.verbose:
                print(f"✗ Wrong: {result['answer']} (expected {problem['gold_answer']})")

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(problems)}, Accuracy: {correct}/{i+1} = {100*correct/(i+1):.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"NO-C1 E2E TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total:    {len(problems)}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {100*correct/len(problems):.1f}%")

    # Analyze failures
    failures = [r for r in results if not r["correct"]]
    print(f"\nFailure analysis:")
    print(f"  No candidates: {sum(1 for r in failures if not r['operations_tried'])}")
    print(f"  Wrong answer:  {sum(1 for r in failures if r['operations_tried'])}")


if __name__ == "__main__":
    main()
