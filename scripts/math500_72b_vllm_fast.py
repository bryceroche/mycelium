#!/usr/bin/env python3
"""
MATH500 72B Fast - vLLM without attention capture.

Runs remaining problems from checkpoint using vLLM for 3-5x speedup.
"""

import json
import subprocess
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse


def extract_boxed(text):
    """Extract answer from \\boxed{...} handling nested braces."""
    results = []
    idx = 0
    while True:
        start = text.find(r'\boxed{', idx)
        if start == -1:
            break
        brace_count = 1
        content_start = start + 7
        i = content_start
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            results.append(text[content_start:i-1])
        idx = i
    return results[-1] if results else None


def normalize_answer(s):
    """Basic answer normalization."""
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r'\\boxed\{(.+)\}', r'\1', s)
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace(' ', '').lower()
    return s


def answers_match(pred, gold):
    """Check if answers match."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except:
        return False


def sync_to_s3(output_dir: Path, s3_bucket: str):
    """Sync results to S3."""
    try:
        cmd = f"aws s3 sync {output_dir} s3://{s3_bucket}/math500_72b/"
        subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
        print(f"  [S3] Synced to s3://{s3_bucket}/math500_72b/")
    except Exception as e:
        print(f"  [S3] Sync error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/home/ubuntu/math500_72b")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-72B-Instruct")
    parser.add_argument("--failure-indices", default="/home/ubuntu/math500_7b/failure_indices.json")
    parser.add_argument("--results-7b", default="/home/ubuntu/math500_7b")
    parser.add_argument("--s3-bucket", default="mycelium-data")
    parser.add_argument("--sync-every", type=int, default=20)
    parser.add_argument("--precalc-attempts", type=int, default=3)
    parser.add_argument("--precalc-temp", type=float, default=0.7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MATH500 72B Fast (vLLM)")
    print("=" * 70)

    # Load failure indices
    with open(args.failure_indices) as f:
        failure_indices = json.load(f)

    # Load problems from 7B results
    problems = {}
    results_path = Path(args.results_7b)
    for f in sorted(results_path.glob("problem_*.json")):
        if "error" in f.name:
            continue
        with open(f) as fp:
            r = json.load(fp)
            problems[r["idx"]] = {
                "problem": r["question"],
                "answer": r["gold_answer"],
                "type": r.get("category", "Unknown"),
                "level": r.get("level", "Unknown"),
            }

    # Check existing progress
    completed = set()
    for f in output_dir.glob("problem_*.json"):
        if "error" not in f.name:
            try:
                idx = int(f.stem.split("_")[1])
                completed.add(idx)
            except:
                pass

    remaining = [i for i in failure_indices if i not in completed and i in problems]
    print(f"Failures to process: {len(failure_indices)}")
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("\nAll failures already processed!")
        return

    # Initialize vLLM
    print(f"\nLoading {args.model} with vLLM...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=8,  # Use all 8 GPUs
        trust_remote_code=True,
        max_model_len=2048,
    )
    print("Model loaded!")

    # Sampling params
    greedy_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )

    sampling_params = SamplingParams(
        temperature=args.precalc_temp,
        top_p=0.95,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )

    # Process in batches for efficiency
    stats = {"success": 0, "correct": 0, "errors": 0}

    for i, idx in enumerate(tqdm(remaining, desc="Processing")):
        problem = problems[idx]
        question = problem["problem"]
        gold = problem["answer"]
        category = problem.get("type", "unknown")
        level = problem.get("level", "unknown")

        try:
            is_precalc = category == "Precalculus"
            prompt = f"<|im_start|>user\nSolve this math problem step by step:\n{question}<|im_end|>\n<|im_start|>assistant\n"

            if is_precalc:
                # Multiple attempts for Precalculus
                best_result = None
                for attempt in range(args.precalc_attempts):
                    outputs = llm.generate([prompt], sampling_params)
                    cot = outputs[0].outputs[0].text
                    pred = extract_boxed(cot)
                    is_correct = answers_match(pred, gold)

                    if is_correct:
                        best_result = (cot, pred, attempt + 1)
                        break
                    elif best_result is None:
                        best_result = (cot, pred, attempt + 1)

                cot, pred, attempts_used = best_result
                is_correct = answers_match(pred, gold)
            else:
                # Single greedy attempt
                outputs = llm.generate([prompt], greedy_params)
                cot = outputs[0].outputs[0].text
                pred = extract_boxed(cot)
                is_correct = answers_match(pred, gold)
                attempts_used = 1

            result = {
                "idx": idx,
                "question": question,
                "gold_answer": gold,
                "category": category,
                "level": level,
                "generated_cot": cot,
                "predicted_answer": pred,
                "is_correct": is_correct,
                "n_tokens_generated": len(outputs[0].outputs[0].token_ids),
                "attempts_used": attempts_used,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Save immediately
            output_path = output_dir / f"problem_{idx:04d}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            stats["success"] += 1
            if is_correct:
                stats["correct"] += 1

            # Periodic S3 sync
            if args.s3_bucket and (i + 1) % args.sync_every == 0:
                sync_to_s3(output_dir, args.s3_bucket)
                rate = stats["correct"] / stats["success"] * 100 if stats["success"] > 0 else 0
                print(f"\n  Progress: {stats['success']}/{len(remaining)}, Accuracy: {rate:.1f}%")

        except Exception as e:
            print(f"\nError on problem {idx}: {e}")
            stats["errors"] += 1

    # Final sync
    if args.s3_bucket:
        sync_to_s3(output_dir, args.s3_bucket)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {stats['success']}")
    print(f"Correct: {stats['correct']} ({stats['correct']/max(stats['success'],1)*100:.1f}%)")
    print(f"Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
