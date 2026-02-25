#!/usr/bin/env python3
"""
Batch-optimized GSM8K evaluation.

Key optimizations:
1. Batch segmentation: process N problems at once
2. Batch classification: collect groups from N problems, classify together
3. Batch extraction: batch the generative calls

This should be 5-10x faster than sequential evaluation.
"""

import json
import re
import sys
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForTokenClassification

# Model paths
SEGMENTER_PATH = "/opt/dlami/nvme/models/qwen05b_segmenter_clean/final"
CLASSIFIER_PATH = "/opt/dlami/nvme/models/qwen05b_classifier_multispan/final"
EXTRACTOR_PATH = "/opt/dlami/nvme/models/qwen05b_extractor_multispan/final"

BIO_LABELS = ["O", "B-OP", "I-OP", "B-Q", "I-Q"]
OP_LABELS = ["ADD", "SUB", "MUL", "DIV"]


def extract_gold(answer_text: str):
    if "####" in answer_text:
        parts = answer_text.split("####")
        if len(parts) > 1:
            num_str = parts[1].strip().replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                pass
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except:
            pass
    return None


class BatchPipeline:
    """Batch-optimized pipeline."""

    def __init__(self, device="cuda", batch_size=32):
        self.device = device
        self.batch_size = batch_size

    def load_models(self):
        print("Loading models...")

        # Segmenter
        print("  Loading segmenter...")
        self.seg_tokenizer = AutoTokenizer.from_pretrained(SEGMENTER_PATH, trust_remote_code=True)
        self.segmenter = AutoModelForTokenClassification.from_pretrained(SEGMENTER_PATH, trust_remote_code=True).to(self.device)
        self.segmenter.eval()

        # Classifier
        print("  Loading classifier...")
        self.cls_tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH, trust_remote_code=True)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH, trust_remote_code=True).to(self.device)
        self.classifier.eval()

        # Extractor
        print("  Loading extractor...")
        self.ext_tokenizer = AutoTokenizer.from_pretrained(EXTRACTOR_PATH, trust_remote_code=True)
        self.extractor = AutoModelForCausalLM.from_pretrained(EXTRACTOR_PATH, trust_remote_code=True).to(self.device)
        self.extractor.eval()

        print("  All models loaded.")

    def batch_segment(self, problems: list) -> list:
        """Segment multiple problems at once."""
        # Tokenize all
        encodings = self.seg_tokenizer(
            problems,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            return_offsets_mapping=True,
        )

        offset_mappings = encodings.pop("offset_mapping")
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        # Predict
        with torch.no_grad():
            outputs = self.segmenter(**encodings)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu()

        # Extract spans for each problem
        all_spans = []
        for idx, (problem, pred, offset_map) in enumerate(zip(problems, predictions, offset_mappings)):
            spans = []
            current_span = None

            for i, (label_id, (start, end)) in enumerate(zip(pred.tolist(), offset_map.tolist())):
                if start == end:  # Special token
                    continue

                label = BIO_LABELS[label_id] if label_id < len(BIO_LABELS) else "O"

                if label.startswith("B-"):
                    if current_span is not None:
                        spans.append(current_span)
                    tag = label[2:]
                    current_span = {"tag": tag, "start": start, "end": end, "text": problem[start:end]}
                elif label.startswith("I-"):
                    tag = label[2:]
                    if current_span is not None and current_span["tag"] == tag:
                        current_span["end"] = end
                        current_span["text"] = problem[current_span["start"]:end]
                    else:
                        if current_span is not None:
                            spans.append(current_span)
                        current_span = {"tag": tag, "start": start, "end": end, "text": problem[start:end]}
                else:
                    if current_span is not None:
                        spans.append(current_span)
                        current_span = None

            if current_span is not None:
                spans.append(current_span)

            # Filter tiny spans
            spans = [s for s in spans if len(s["text"].strip()) > 3]
            all_spans.append(spans)

        return all_spans

    def batch_classify(self, marked_texts: list) -> list:
        """Classify multiple marked texts at once."""
        if not marked_texts:
            return []

        encodings = self.cls_tokenizer(
            marked_texts,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            padding=True,
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.classifier(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_indices = torch.argmax(probs, dim=-1).cpu().tolist()
            confidences = probs.max(dim=-1).values.cpu().tolist()

        return [(OP_LABELS[idx], conf) for idx, conf in zip(pred_indices, confidences)]

    def batch_extract(self, prompts: list) -> list:
        """Extract arguments from multiple prompts at once."""
        if not prompts:
            return []

        encodings = self.ext_tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        input_lens = [encodings["attention_mask"][i].sum().item() for i in range(len(prompts))]
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.extractor.generate(
                **encodings,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.ext_tokenizer.pad_token_id,
                eos_token_id=self.ext_tokenizer.eos_token_id,
            )

        results = []
        for i, (output, input_len) in enumerate(zip(outputs, input_lens)):
            generated = self.ext_tokenizer.decode(output[input_len:], skip_special_tokens=True)

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
            results.append(args)

        return results

    def mark_spans(self, problem: str, spans: list) -> str:
        """Mark spans with <SPAN> tags."""
        if not spans:
            return None

        positions = []
        for span in spans:
            start = span.get("start")
            text = span["text"]
            if start is None:
                start = problem.find(text)
            if start == -1:
                continue
            positions.append((start, start + len(text)))

        if not positions:
            return None

        positions.sort()
        result = problem
        for start, end in reversed(positions):
            result = result[:end] + " </SPAN>" + result[end:]
            result = result[:start] + "<SPAN> " + result[start:]

        return result

    def execute_operations(self, operations: list) -> float:
        """Execute operations and return result."""
        OPS = {
            "ADD": lambda a, b: a + b,
            "SUB": lambda a, b: a - b,
            "MUL": lambda a, b: a * b,
            "DIV": lambda a, b: a / b if b != 0 else None,
        }

        results = []
        for op in operations:
            op_type = op.get("operation", "ADD")
            arg1 = op.get("arg1")
            arg2 = op.get("arg2")

            if arg1 is None or arg2 is None:
                continue

            if op_type in OPS:
                try:
                    result = OPS[op_type](arg1, arg2)
                    if result is not None:
                        results.append(result)
                except:
                    pass

        return results[-1] if results else None

    def solve_batch(self, problems: list) -> list:
        """Solve a batch of problems."""
        n = len(problems)

        # Step 1: Batch segment
        all_spans = self.batch_segment(problems)

        # Step 2: Generate candidate groupings and collect all marked texts
        all_marked = []  # (problem_idx, group_indices, marked_text)

        for pidx, (problem, spans) in enumerate(zip(problems, all_spans)):
            op_spans = [s for s in spans if s.get("tag") == "OP"]

            if not op_spans:
                continue

            # Simple groupings: each span alone, all spans together
            groupings = [[i] for i in range(len(op_spans))]
            if len(op_spans) > 1:
                groupings.append(list(range(len(op_spans))))

            for group in groupings:
                marked = self.mark_spans(problem, [op_spans[i] for i in group])
                if marked:
                    all_marked.append((pidx, group, marked))

        # Step 3: Batch classify all marked texts
        if all_marked:
            marked_texts = [m[2] for m in all_marked]
            classifications = self.batch_classify(marked_texts)
        else:
            classifications = []

        # Step 4: Prepare extraction prompts
        ext_prompts = []
        ext_info = []  # (problem_idx, group_indices, op_type)

        for (pidx, group, marked), (op_type, conf) in zip(all_marked, classifications):
            prompt = f"[{op_type}]\n{marked}\nArguments:\n"
            ext_prompts.append(prompt)
            ext_info.append((pidx, group, op_type, conf))

        # Step 5: Batch extract
        if ext_prompts:
            all_args = self.batch_extract(ext_prompts)
        else:
            all_args = []

        # Step 6: Build operations and execute for each problem
        results = [{"answer": None, "n_spans": len([s for s in sp if s.get("tag") == "OP"])} for sp in all_spans]

        # Group by problem
        problem_ops = {i: [] for i in range(n)}
        for (pidx, group, op_type, conf), args in zip(ext_info, all_args):
            op = {"operation": op_type, "confidence": conf}
            if len(args) >= 1:
                op["arg1"] = args[0]["value"]
            if len(args) >= 2:
                op["arg2"] = args[1]["value"]
            problem_ops[pidx].append(op)

        # Execute each problem's best candidate
        for pidx in range(n):
            ops = problem_ops[pidx]
            if ops:
                # Try each operation as a candidate
                for op in ops:
                    answer = self.execute_operations([op])
                    if answer is not None:
                        results[pidx]["answer"] = answer
                        break

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("GSM8K BATCH EVALUATION")
    print("=" * 60)

    # Load pipeline
    pipeline = BatchPipeline(batch_size=args.batch_size)
    pipeline.load_models()

    # Load GSM8K
    print("\nLoading GSM8K train set...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    if args.limit:
        ds = ds.select(range(args.limit))

    print(f"Total: {len(ds)} problems")

    correct = 0
    processed = 0
    results_list = []

    # Process in batches
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(ds), batch_size), desc="Batches"):
        batch_end = min(batch_start + batch_size, len(ds))
        batch = ds.select(range(batch_start, batch_end))

        problems = [item["question"] for item in batch]
        golds = [extract_gold(item["answer"]) for item in batch]

        # Solve batch
        batch_results = pipeline.solve_batch(problems)

        # Score
        for i, (result, gold) in enumerate(zip(batch_results, golds)):
            if gold is None:
                continue

            pred = result.get("answer")
            processed += 1
            is_correct = pred is not None and abs(pred - gold) < 0.01

            if is_correct:
                correct += 1

            results_list.append({
                "idx": batch_start + i,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "n_spans": result.get("n_spans", 0),
            })

    accuracy = correct / processed if processed > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"FINAL: {correct}/{processed} = {100*accuracy:.2f}%")
    print("=" * 60)

    # Save
    output = {
        "dataset": "gsm8k_train",
        "total": processed,
        "correct": correct,
        "accuracy": accuracy,
        "results": results_list,
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open("/opt/dlami/nvme/gsm8k_batch_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Saved to /opt/dlami/nvme/gsm8k_batch_results.json")


if __name__ == "__main__":
    main()
