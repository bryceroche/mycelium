#!/usr/bin/env python3
"""
Mycelium v6: End-to-End Pipeline

Wires all components together:
1. SEGMENTER: BIO tagging → spans
2. CANDIDATE GENERATOR: enumerate groupings
3. CLASSIFIER: classify each group → operation
4. EXTRACTOR: extract arguments
5. EXECUTOR: execute and score
6. Pick best candidate → answer

Usage:
    python v6_e2e_pipeline.py --problem "Janet's ducks lay 16 eggs per day..."
    python v6_e2e_pipeline.py --eval data/gsm8k_test.json --limit 100
"""

import argparse
import json
import re
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForTokenClassification
from tqdm import tqdm

# Import local modules
from v6_candidate_generator import generate_candidate_groupings, extract_numbers_from_text
from v6_executor import DSLExecutor, score_candidate, evaluate_candidates


# Model paths
SEGMENTER_PATH = "/opt/dlami/nvme/models/qwen05b_segmenter_clean/final"
CLASSIFIER_PATH = "/opt/dlami/nvme/models/qwen05b_classifier_multispan/final"
EXTRACTOR_PATH = "/opt/dlami/nvme/models/qwen05b_extractor_multispan/final"

# Local fallbacks
if not Path("/opt/dlami/nvme").exists():
    SEGMENTER_PATH = "models/qwen05b_segmenter_clean/final"
    CLASSIFIER_PATH = "models/qwen05b_classifier_multispan/final"
    EXTRACTOR_PATH = "models/qwen05b_extractor_multispan/final"

# Labels
BIO_LABELS = ["O", "B-OP", "I-OP", "B-Q", "I-Q"]
OP_LABELS = ["ADD", "SUB", "MUL", "DIV"]


class MyceliumPipeline:
    """End-to-end math word problem solver."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.segmenter = None
        self.segmenter_tokenizer = None
        self.classifier = None
        self.classifier_tokenizer = None
        self.extractor = None
        self.extractor_tokenizer = None
        self.executor = DSLExecutor()

    def load_models(self):
        """Load all models."""
        print("\nLoading models...")

        # Segmenter
        print("  Loading segmenter...")
        self.segmenter_tokenizer = AutoTokenizer.from_pretrained(
            SEGMENTER_PATH, trust_remote_code=True
        )
        self.segmenter = AutoModelForTokenClassification.from_pretrained(
            SEGMENTER_PATH, trust_remote_code=True
        ).to(self.device)
        self.segmenter.eval()

        # Classifier
        print("  Loading classifier...")
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(
            CLASSIFIER_PATH, trust_remote_code=True
        )
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            CLASSIFIER_PATH, trust_remote_code=True
        ).to(self.device)
        self.classifier.eval()

        # Extractor
        print("  Loading extractor...")
        self.extractor_tokenizer = AutoTokenizer.from_pretrained(
            EXTRACTOR_PATH, trust_remote_code=True
        )
        self.extractor = AutoModelForCausalLM.from_pretrained(
            EXTRACTOR_PATH, trust_remote_code=True
        ).to(self.device)
        self.extractor.eval()

        print("  All models loaded.")

    def segment(self, problem_text: str) -> List[Dict]:
        """Run segmenter to extract spans."""
        # Tokenize
        inputs = self.segmenter_tokenizer(
            problem_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.segmenter(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        # Extract spans from BIO predictions
        # Handle case where model only predicts I-OP without B-OP
        spans = []
        current_span = None
        prev_label = "O"

        for i, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:  # Special token
                continue

            label = BIO_LABELS[pred] if pred < len(BIO_LABELS) else "O"

            if label.startswith("B-"):
                # Explicit B- tag
                if current_span is not None:
                    spans.append(current_span)
                tag = label[2:]
                current_span = {
                    "tag": tag,
                    "start": start,
                    "end": end,
                    "text": problem_text[start:end],
                }
            elif label.startswith("I-"):
                tag = label[2:]
                if current_span is not None and current_span["tag"] == tag:
                    # Continue current span
                    current_span["end"] = end
                    current_span["text"] = problem_text[current_span["start"]:end]
                else:
                    # I- without matching B- or different tag: treat as new span
                    if current_span is not None:
                        spans.append(current_span)
                    current_span = {
                        "tag": tag,
                        "start": start,
                        "end": end,
                        "text": problem_text[start:end],
                    }
            else:  # "O" label
                if current_span is not None:
                    spans.append(current_span)
                    current_span = None

            prev_label = label

        if current_span is not None:
            spans.append(current_span)

        # Filter tiny spans
        spans = [s for s in spans if len(s["text"].strip()) > 3]

        return spans

    def classify_group(self, problem_text: str, spans: List[Dict], group_indices: List[int]) -> Tuple[str, float]:
        """Classify a group of spans."""
        # Mark spans
        marked = self._mark_spans(problem_text, [spans[i] for i in group_indices])
        if marked is None:
            return "ADD", 0.0  # Fallback

        # Tokenize
        inputs = self.classifier_tokenizer(
            marked,
            return_tensors="pt",
            truncation=True,
            max_length=384,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

        return OP_LABELS[pred_idx], confidence

    def batch_classify_groups(
        self, problem_text: str, spans: List[Dict], all_groups: List[List[int]]
    ) -> List[Tuple[str, float]]:
        """Batch classify multiple groups in one forward pass."""
        if not all_groups:
            return []

        # Prepare all marked texts
        marked_texts = []
        valid_indices = []
        for i, group in enumerate(all_groups):
            marked = self._mark_spans(problem_text, [spans[idx] for idx in group])
            if marked is not None:
                marked_texts.append(marked)
                valid_indices.append(i)

        if not marked_texts:
            return [("ADD", 0.0)] * len(all_groups)

        # Batch tokenize
        inputs = self.classifier_tokenizer(
            marked_texts,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Batch predict
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_indices = torch.argmax(probs, dim=-1).cpu().tolist()
            confidences = probs.max(dim=-1).values.cpu().tolist()

        # Build results for all groups (including invalid ones)
        results = [("ADD", 0.0)] * len(all_groups)
        for batch_idx, orig_idx in enumerate(valid_indices):
            results[orig_idx] = (OP_LABELS[pred_indices[batch_idx]], confidences[batch_idx])

        return results

    def extract_arguments(self, problem_text: str, spans: List[Dict], group_indices: List[int], operation: str) -> List[Dict]:
        """Extract arguments from a group of spans."""
        # Mark spans
        marked = self._mark_spans(problem_text, [spans[i] for i in group_indices])
        if marked is None:
            return []

        # Build prompt
        prompt = f"[{operation}]\n{marked}\nArguments:\n"

        inputs = self.extractor_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.extractor.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.extractor_tokenizer.pad_token_id,
                eos_token_id=self.extractor_tokenizer.eos_token_id,
            )

        # Decode
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
                    source = parts[1].strip() if len(parts) > 1 else "UNK"
                    args.append({"value": value, "source": source})
                except ValueError:
                    pass

        return args

    def _mark_spans(self, problem_text: str, spans: List[Dict]) -> Optional[str]:
        """Mark spans with <SPAN>...</SPAN> tags."""
        if not spans:
            return None

        # Get positions
        positions = []
        for span in spans:
            start = span.get("start")
            text = span["text"]

            if start is None:
                start = problem_text.find(text)
                if start == -1:
                    start = problem_text.lower().find(text.lower())

            if start == -1:
                continue

            end = start + len(text)
            positions.append((start, end))

        if not positions:
            return None

        # Sort and check overlaps
        positions.sort()
        for i in range(len(positions) - 1):
            if positions[i][1] > positions[i + 1][0]:
                return None  # Overlapping

        # Insert markers from end to start
        result = problem_text
        for start, end in reversed(positions):
            result = result[:end] + " </SPAN>" + result[end:]
            result = result[:start] + "<SPAN> " + result[start:]

        return result

    def solve(self, problem_text: str, verbose: bool = False, prune_threshold: float = 0.4, max_candidates: int = 5) -> Dict:
        """Solve a math word problem with batched classification + early pruning."""
        result = {
            "problem": problem_text,
            "answer": None,
            "confidence": 0.0,
            "trace": {},
        }

        # Step 1: Segment
        spans = self.segment(problem_text)
        op_spans = [s for s in spans if s.get("tag") == "OP"]
        result["trace"]["spans"] = spans
        result["trace"]["n_spans"] = len(op_spans)

        if verbose:
            print(f"\nSegmented {len(op_spans)} OP spans:")
            for i, s in enumerate(op_spans):
                print(f"  [{i}] {s['text'][:50]}...")

        if len(op_spans) == 0:
            result["trace"]["error"] = "No OP spans found"
            return result

        # Step 2: Generate candidates
        candidates = generate_candidate_groupings(spans, max_candidates=15)
        result["trace"]["n_candidates"] = len(candidates)

        if verbose:
            print(f"\nGenerated {len(candidates)} candidate groupings")

        # Step 3: Collect all unique groups and batch classify
        # Build mapping: candidate_idx -> list of (group, group_key)
        candidate_groups = []
        all_groups = []
        group_to_idx = {}

        for cand_idx, grouping in enumerate(candidates):
            op_groups = [g for g in grouping if all(spans[i].get("tag") == "OP" for i in g)]
            groups_for_cand = []
            for group in op_groups:
                group_key = tuple(sorted(group))
                if group_key not in group_to_idx:
                    group_to_idx[group_key] = len(all_groups)
                    all_groups.append(group)
                groups_for_cand.append((group, group_key))
            candidate_groups.append(groups_for_cand)

        # Batch classify all unique groups
        if verbose:
            print(f"\nBatch classifying {len(all_groups)} unique groups...")

        classification_results = self.batch_classify_groups(problem_text, spans, all_groups)

        # Build lookup: group_key -> (op_type, confidence)
        group_classifications = {}
        for i, group in enumerate(all_groups):
            group_key = tuple(sorted(group))
            group_classifications[group_key] = classification_results[i]

        # Step 4: Prune candidates - keep top N by average confidence
        candidate_scores = []
        for cand_idx, groups_for_cand in enumerate(candidate_groups):
            if not groups_for_cand:
                continue
            avg_conf = sum(group_classifications[gk][1] for _, gk in groups_for_cand) / len(groups_for_cand)
            candidate_scores.append((cand_idx, groups_for_cand, avg_conf))

        # Sort by average confidence and keep top N
        candidate_scores.sort(key=lambda x: x[2], reverse=True)
        surviving_candidates = [(c[0], c[1]) for c in candidate_scores[:max_candidates]]

        result["trace"]["candidates_after_pruning"] = len(surviving_candidates)

        if verbose:
            print(f"Pruned to {len(surviving_candidates)}/{len(candidate_scores)} candidates (top-{max_candidates})")

        # Step 5: Run extractor only on surviving candidates
        evaluated = []

        for cand_idx, groups_for_cand in surviving_candidates:
            operations = []
            total_confidence = 0.0

            for group, group_key in groups_for_cand:
                op_type, conf = group_classifications[group_key]
                total_confidence += conf

                # Extract arguments (expensive - only on survivors)
                args = self.extract_arguments(problem_text, spans, group, op_type)

                # Build operation dict
                op_dict = {
                    "operation": op_type,
                    "group": group,
                    "confidence": conf,
                }

                if len(args) >= 1:
                    op_dict["arg1"] = args[0]["value"]
                if len(args) >= 2:
                    op_dict["arg2"] = args[1]["value"]

                operations.append(op_dict)

            # Execute
            if operations:
                try:
                    answer, trace = self.executor.execute_graph(operations)
                    score = score_candidate(answer, None, trace)

                    # Add confidence bonus
                    avg_conf = total_confidence / len(operations) if operations else 0
                    score += avg_conf * 0.5

                    evaluated.append({
                        "grouping": candidates[cand_idx],
                        "operations": operations,
                        "answer": answer,
                        "score": score,
                        "trace": trace,
                    })
                except Exception as e:
                    if verbose:
                        print(f"  Candidate {cand_idx} failed: {e}")

        # Sort by score
        evaluated.sort(key=lambda x: x.get("score", -999), reverse=True)
        result["trace"]["evaluated_candidates"] = len(evaluated)

        if verbose:
            print(f"\nEvaluated {len(evaluated)} candidates:")
            for i, e in enumerate(evaluated[:3]):
                print(f"  {i+1}. answer={e['answer']}, score={e['score']:.2f}")

        # Pick best
        if evaluated:
            best = evaluated[0]
            result["answer"] = best["answer"]
            result["confidence"] = best["score"]
            result["trace"]["best_candidate"] = best

        return result


def extract_gold_answer(answer_str: str) -> Optional[float]:
    """Extract numeric answer from GSM8K answer string."""
    # GSM8K format: "#### 42"
    if "####" in answer_str:
        parts = answer_str.split("####")
        if len(parts) > 1:
            num_str = parts[1].strip().replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                pass

    # Try to find last number
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_str)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Mycelium v6 E2E Pipeline")
    parser.add_argument("--problem", type=str, help="Single problem to solve")
    parser.add_argument("--eval", type=str, help="Evaluation dataset path")
    parser.add_argument("--limit", type=int, default=100, help="Max problems to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = MyceliumPipeline()
    pipeline.load_models()

    if args.problem:
        # Solve single problem
        print("\n" + "=" * 60)
        print("SOLVING PROBLEM")
        print("=" * 60)
        print(f"\nProblem: {args.problem[:200]}...")

        result = pipeline.solve(args.problem, verbose=True)

        print(f"\n{'=' * 40}")
        print(f"ANSWER: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"{'=' * 40}")

    elif args.eval:
        # Evaluate on dataset
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)

        # Load dataset
        with open(args.eval) as f:
            if args.eval.endswith(".jsonl"):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)

        data = data[:args.limit]
        print(f"\nEvaluating on {len(data)} problems...")

        correct = 0
        total = 0
        errors = []

        for item in tqdm(data):
            question = item.get("question", item.get("problem", ""))
            answer_str = item.get("answer", "")
            gold = extract_gold_answer(answer_str)

            if gold is None:
                continue

            result = pipeline.solve(question, verbose=False)
            pred = result.get("answer")

            total += 1

            if pred is not None and abs(pred - gold) < 0.01:
                correct += 1
            else:
                errors.append({
                    "question": question[:100],
                    "gold": gold,
                    "pred": pred,
                    "n_spans": result["trace"].get("n_spans", 0),
                })

        accuracy = correct / total if total > 0 else 0
        print(f"\n{'=' * 40}")
        print(f"Results: {correct}/{total} = {100*accuracy:.1f}%")
        print(f"{'=' * 40}")

        # Show some errors
        if errors and args.verbose:
            print("\nSample errors:")
            for e in errors[:5]:
                print(f"  Q: {e['question']}...")
                print(f"  Gold: {e['gold']}, Pred: {e['pred']}, Spans: {e['n_spans']}")
                print()

    else:
        # Demo mode
        print("\n" + "=" * 60)
        print("DEMO MODE")
        print("=" * 60)

        demo_problems = [
            "In the first box Marcus counted 72 raisins. In a second box he counted 74 raisins. How many raisins total?",
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. How many eggs does she have left?",
            "A store sells apples for $2 each. Maria bought 5 apples. How much did she spend?",
        ]

        for problem in demo_problems:
            print(f"\nProblem: {problem}")
            result = pipeline.solve(problem, verbose=args.verbose)
            print(f"Answer: {result['answer']} (confidence: {result['confidence']:.2f})")


if __name__ == "__main__":
    main()
