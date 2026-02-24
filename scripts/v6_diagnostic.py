#!/usr/bin/env python3
"""
Mycelium v6: Diagnostic Logging for E2E Pipeline

Traces exactly where failures occur to enable error attribution.
"""

import json
import argparse
from typing import List, Dict, Optional, Tuple
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForTokenClassification

# Import local modules
from v6_candidate_generator import generate_candidate_groupings
from v6_executor import DSLExecutor

# Model paths
SEGMENTER_PATH = "/opt/dlami/nvme/models/qwen05b_segmenter_clean/final"
CLASSIFIER_PATH = "/opt/dlami/nvme/models/qwen05b_classifier_multispan/final"
EXTRACTOR_PATH = "/opt/dlami/nvme/models/qwen05b_extractor_multispan/final"

BIO_LABELS = ["O", "B-OP", "I-OP", "B-Q", "I-Q"]
OP_LABELS = ["ADD", "SUB", "MUL", "DIV"]


class DiagnosticPipeline:
    """E2E pipeline with detailed diagnostic output."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.executor = DSLExecutor()
        self.models_loaded = False

    def load_models(self):
        print("Loading models...")

        # Segmenter
        self.segmenter_tokenizer = AutoTokenizer.from_pretrained(
            SEGMENTER_PATH, trust_remote_code=True
        )
        self.segmenter = AutoModelForTokenClassification.from_pretrained(
            SEGMENTER_PATH, trust_remote_code=True
        ).to(self.device)
        self.segmenter.eval()

        # Classifier
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(
            CLASSIFIER_PATH, trust_remote_code=True
        )
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            CLASSIFIER_PATH, trust_remote_code=True
        ).to(self.device)
        self.classifier.eval()

        # Extractor
        self.extractor_tokenizer = AutoTokenizer.from_pretrained(
            EXTRACTOR_PATH, trust_remote_code=True
        )
        self.extractor = AutoModelForCausalLM.from_pretrained(
            EXTRACTOR_PATH, trust_remote_code=True
        ).to(self.device)
        self.extractor.eval()

        self.models_loaded = True
        print("Models loaded.\n")

    def segment(self, problem_text: str) -> List[Dict]:
        """Run segmenter to extract spans."""
        inputs = self.segmenter_tokenizer(
            problem_text,
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

        # Extract spans (handle I-OP without B-OP)
        spans = []
        current_span = None

        for i, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            if start == end:
                continue

            label = BIO_LABELS[pred] if pred < len(BIO_LABELS) else "O"

            if label.startswith("B-"):
                if current_span is not None:
                    spans.append(current_span)
                tag = label[2:]
                current_span = {
                    "tag": tag, "start": start, "end": end,
                    "text": problem_text[start:end],
                }
            elif label.startswith("I-"):
                tag = label[2:]
                if current_span is not None and current_span["tag"] == tag:
                    current_span["end"] = end
                    current_span["text"] = problem_text[current_span["start"]:end]
                else:
                    if current_span is not None:
                        spans.append(current_span)
                    current_span = {
                        "tag": tag, "start": start, "end": end,
                        "text": problem_text[start:end],
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

    def _mark_spans(self, problem_text: str, spans: List[Dict]) -> Optional[str]:
        """Mark spans with <SPAN>...</SPAN> tags."""
        if not spans:
            return None

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

        positions.sort()
        for i in range(len(positions) - 1):
            if positions[i][1] > positions[i + 1][0]:
                return None

        result = problem_text
        for start, end in reversed(positions):
            result = result[:end] + " </SPAN>" + result[end:]
            result = result[:start] + "<SPAN> " + result[start:]
        return result

    def classify_group(self, problem_text: str, spans: List[Dict], group_indices: List[int]) -> Tuple[str, float, List[float]]:
        """Classify a group of spans. Returns (op, confidence, all_probs)."""
        marked = self._mark_spans(problem_text, [spans[i] for i in group_indices])
        if marked is None:
            return "ADD", 0.0, [0.25, 0.25, 0.25, 0.25]

        inputs = self.classifier_tokenizer(
            marked, return_tensors="pt", truncation=True, max_length=384,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.classifier(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            pred_idx = torch.argmax(torch.tensor(probs)).item()
            confidence = probs[pred_idx]

        return OP_LABELS[pred_idx], confidence, probs

    def extract_arguments(self, problem_text: str, spans: List[Dict], group_indices: List[int], operation: str) -> Tuple[List[Dict], str]:
        """Extract arguments. Returns (args, raw_output)."""
        marked = self._mark_spans(problem_text, [spans[i] for i in group_indices])
        if marked is None:
            return [], ""

        prompt = f"[{operation}]\n{marked}\nArguments:\n"
        inputs = self.extractor_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256,
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

        return args, generated.strip()

    def diagnose_problem(self, problem_text: str, gold_answer: float, problem_idx: int) -> Dict:
        """Run full diagnostic on a single problem."""
        print(f"\n{'='*70}")
        print(f"PROBLEM {problem_idx}: \"{problem_text[:60]}...\"")
        print(f"GOLD ANSWER: {gold_answer}")
        print("=" * 70)

        result = {
            "problem_idx": problem_idx,
            "gold": gold_answer,
            "pred": None,
            "error_type": None,
        }

        # Step 1: Segment
        spans = self.segment(problem_text)
        op_spans = [s for s in spans if s.get("tag") == "OP"]

        print(f"\nSEGMENTER: {len(op_spans)} OP spans found")
        for i, s in enumerate(op_spans):
            print(f"  [{i}] \"{s['text'][:50]}{'...' if len(s['text']) > 50 else ''}\"")

        if len(op_spans) == 0:
            print("\nERROR: No spans found!")
            result["error_type"] = "SEGMENTATION_MISS"
            return result

        # Step 2: Generate candidates
        candidates = generate_candidate_groupings(spans, max_candidates=10)
        print(f"\nCANDIDATES: {len(candidates)} generated")

        # Step 3: Evaluate each candidate
        evaluated = []
        for cand_idx, grouping in enumerate(candidates):
            op_groups = [g for g in grouping if all(spans[i].get("tag") == "OP" for i in g)]

            # Format candidate for display
            group_str = ", ".join(["{" + ",".join(str(i) for i in g) + "}" for g in op_groups])
            print(f"\n  Candidate {cand_idx + 1}: {group_str}")

            operations = []
            classifier_info = []
            extractor_info = []
            total_confidence = 0.0

            for group in op_groups:
                # Classify
                op_type, conf, all_probs = self.classify_group(problem_text, spans, group)
                total_confidence += conf

                # Format probs
                prob_str = " ".join([f"{OP_LABELS[i]}:{p:.2f}" for i, p in enumerate(all_probs)])
                classifier_info.append(f"{{{','.join(str(i) for i in group)}}}→{op_type}({conf:.2f}) [{prob_str}]")

                # Extract
                args, raw_output = self.extract_arguments(problem_text, spans, group, op_type)
                arg_str = ", ".join([f"{a['value']}|{a['source']}" for a in args]) if args else "NONE"
                extractor_info.append(f"{{{','.join(str(i) for i in group)}}}→[{arg_str}]")

                # Build operation
                op_dict = {"operation": op_type, "group": group, "confidence": conf}
                if len(args) >= 1:
                    op_dict["arg1"] = args[0]["value"]
                if len(args) >= 2:
                    op_dict["arg2"] = args[1]["value"]
                operations.append(op_dict)

            # Print classifier output
            print(f"    Classifier: {' '.join(classifier_info[:3])}")
            if len(classifier_info) > 3:
                print(f"                {' '.join(classifier_info[3:])}")

            # Print extractor output
            print(f"    Extractor:  {' '.join(extractor_info[:3])}")
            if len(extractor_info) > 3:
                print(f"                {' '.join(extractor_info[3:])}")

            # Execute
            if operations:
                try:
                    answer, trace = self.executor.execute_graph(operations)

                    # Build execution trace
                    exec_parts = []
                    for step in trace.get("steps", []):
                        op = step.get("op", "?")
                        a1 = step.get("arg1", "?")
                        a2 = step.get("arg2", "")
                        res = step.get("result", "ERR")
                        if a2:
                            exec_parts.append(f"{op}({a1},{a2})={res}")
                        else:
                            exec_parts.append(f"{op}({a1})={res}")

                    print(f"    Execute:    {' → '.join(exec_parts[:4])}")
                    if len(exec_parts) > 4:
                        print(f"                {' → '.join(exec_parts[4:])}")

                    # Score
                    from v6_executor import score_candidate
                    score = score_candidate(answer, None, trace)
                    avg_conf = total_confidence / len(operations) if operations else 0
                    score += avg_conf * 0.5
                    print(f"    Score:      {score:.2f} (answer={answer})")

                    evaluated.append({
                        "cand_idx": cand_idx,
                        "grouping": grouping,
                        "operations": operations,
                        "answer": answer,
                        "score": score,
                    })
                except Exception as e:
                    print(f"    Execute:    ERROR - {e}")
                    print(f"    Score:      -1.0")

        # Pick winner
        if evaluated:
            evaluated.sort(key=lambda x: x.get("score", -999), reverse=True)
            winner = evaluated[0]
            result["pred"] = winner["answer"]

            print(f"\nWINNER: candidate {winner['cand_idx'] + 1}, answer={winner['answer']}")
            print(f"GOLD: {gold_answer}")

            # Determine error type
            if winner["answer"] is not None and abs(winner["answer"] - gold_answer) < 0.01:
                print("RESULT: CORRECT!")
                result["error_type"] = "CORRECT"
            else:
                # Check if gold answer appears in any candidate
                gold_in_candidates = any(
                    e["answer"] is not None and abs(e["answer"] - gold_answer) < 0.01
                    for e in evaluated
                )

                if gold_in_candidates:
                    result["error_type"] = "SCORING_WRONG"
                    print("ERROR TYPE: SCORING_WRONG (gold answer was in candidates but didn't win)")
                else:
                    # Check if this looks like missing implicit ops
                    # Heuristic: if we have multiple operations but answer doesn't match
                    n_ops = len(winner["operations"])
                    if n_ops >= 2:
                        result["error_type"] = "MISSING_IMPLICIT_OPS"
                        print("ERROR TYPE: MISSING_IMPLICIT_OPS (need bridging operations)")
                    else:
                        # Could be classifier or extractor
                        result["error_type"] = "CLASSIFIER_OR_EXTRACTOR"
                        print("ERROR TYPE: CLASSIFIER_OR_EXTRACTOR (wrong op or args)")
        else:
            result["error_type"] = "EXECUTION_FAILED"
            print("ERROR TYPE: EXECUTION_FAILED (no candidates executed)")

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/ubuntu/mycelium/data/dsl_test_pairs_clean.json")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    pipeline = DiagnosticPipeline()
    pipeline.load_models()

    # Load data
    with open(args.data) as f:
        data = json.load(f)
    data = data[:args.limit]

    print(f"\nRunning diagnostic on {len(data)} problems...\n")

    results = []
    for i, item in enumerate(data):
        problem = item.get("problem_text", item.get("question", ""))
        gold = item.get("gold_answer")

        if gold is None:
            continue
        if isinstance(gold, str):
            try:
                gold = float(gold.replace(",", ""))
            except:
                continue

        result = pipeline.diagnose_problem(problem, gold, i + 1)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("FAILURE BREAKDOWN")
    print("=" * 70)

    error_counts = Counter(r["error_type"] for r in results)
    total = len(results)

    print(f"\n  Correct:                         {error_counts.get('CORRECT', 0)}/{total}")
    print(f"  Segmentation miss:               {error_counts.get('SEGMENTATION_MISS', 0)}/{total}")
    print(f"  Classifier or extractor error:   {error_counts.get('CLASSIFIER_OR_EXTRACTOR', 0)}/{total}")
    print(f"  Missing implicit ops:            {error_counts.get('MISSING_IMPLICIT_OPS', 0)}/{total}")
    print(f"  Scoring picked wrong candidate:  {error_counts.get('SCORING_WRONG', 0)}/{total}")
    print(f"  Execution failed:                {error_counts.get('EXECUTION_FAILED', 0)}/{total}")

    # Additional stats
    print("\n" + "-" * 40)
    print("Additional observations:")

    correct_count = error_counts.get('CORRECT', 0)
    accuracy = 100 * correct_count / total if total > 0 else 0
    print(f"  Overall accuracy: {correct_count}/{total} = {accuracy:.1f}%")


if __name__ == "__main__":
    main()
