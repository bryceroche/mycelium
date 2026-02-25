#!/usr/bin/env python3
"""
Full MATH 72B with vLLM - Fast inference without attention capture.

Usage:
    # Run on all problems:
    python math_full_72b_vllm.py --output-dir /data/math_72b

    # Run only on 7B failures (hybrid approach):
    python math_full_72b_vllm.py --failure-indices ./failure_indices.json --output-dir /data/math_72b
"""

import json
import subprocess
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse


def load_full_math():
    """Load full MATH dataset (~12.5K problems)."""
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split="test")

    problems = []
    for i, item in enumerate(ds):
        problems.append({
            "idx": i,
            "problem": item["problem"],
            "solution": item["solution"],
            "type": item["type"],
            "level": item["level"],
        })
    return problems


def extract_gold_answer(solution_text: str) -> str:
    """Extract the boxed answer from MATH solution text."""
    match = re.search(r'\\boxed\{([^}]+)\}', solution_text)
    if match:
        return match.group(1)
    # Handle nested braces
    start = solution_text.find(r'\boxed{')
    if start != -1:
        brace_count = 1
        content_start = start + 7
        i = content_start
        while i < len(solution_text) and brace_count > 0:
            if solution_text[i] == '{':
                brace_count += 1
            elif solution_text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            return solution_text[content_start:i-1]
    return ""


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
    s = s.replace('\\frac', '').replace('\\sqrt', '')
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = s.replace('{', '').replace('}', '')
    s = s.replace(' ', '').lower()
    return s


def answers_match(pred, gold, tolerance=1e-6):
    """Check if answers match."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)

    if p == g:
        return True

    try:
        p_num = float(p.replace(',', ''))
        g_num = float(g.replace(',', ''))
        if abs(g_num) < tolerance:
            return abs(p_num - g_num) < tolerance
        return abs(p_num - g_num) / max(abs(g_num), 1e-10) < tolerance
    except (ValueError, TypeError, AttributeError):
        pass

    return False


def sync_to_s3(output_dir: Path, s3_bucket: str, prefix: str = "math_full_72b"):
    """Sync results to S3."""
    try:
        cmd = f"aws s3 sync {output_dir} s3://{s3_bucket}/{prefix}/"
        subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
        print(f"  [S3] Synced to s3://{s3_bucket}/{prefix}/")
    except Exception as e:
        print(f"  [S3] Sync error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/math_72b")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-72B-Instruct")
    parser.add_argument("--failure-indices", default=None,
                        help="JSON file with indices to process (for hybrid approach)")
    parser.add_argument("--s3-bucket", default="mycelium-data")
    parser.add_argument("--sync-every", type=int, default=100)
    parser.add_argument("--precalc-attempts", type=int, default=3)
    parser.add_argument("--precalc-temp", type=float, default=0.7)
    parser.add_argument("--tensor-parallel", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Limit problems (for testing)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Full MATH 72B (vLLM)")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")

    # Load dataset
    print("\nLoading MATH dataset...")
    all_problems = load_full_math()
    print(f"Total problems: {len(all_problems)}")

    # Filter to failure indices if provided (hybrid approach)
    if args.failure_indices:
        with open(args.failure_indices) as f:
            failure_indices = set(json.load(f))
        problems = [p for p in all_problems if p["idx"] in failure_indices]
        print(f"Filtering to {len(problems)} failures from 7B")
    else:
        problems = all_problems

    if args.limit:
        problems = problems[:args.limit]
        print(f"Limited to {len(problems)} problems")

    # Check existing progress
    completed = set()
    for f in output_dir.glob("problem_*.json"):
        if "error" not in f.name:
            try:
                idx = int(f.stem.split("_")[1])
                completed.add(idx)
            except:
                pass

    remaining = [p for p in problems if p["idx"] not in completed]
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("\nAll problems already processed!")
        return

    # Initialize vLLM
    print(f"\nLoading {args.model} with vLLM (TP={args.tensor_parallel})...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
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

    # Process problems
    stats = {"success": 0, "correct": 0, "errors": 0}
    category_stats = {}

    for i, problem in enumerate(tqdm(remaining, desc="Processing")):
        idx = problem["idx"]
        question = problem["problem"]
        gold = extract_gold_answer(problem["solution"])
        category = problem.get("type", "Unknown")
        level = problem.get("level", "Unknown")

        try:
            is_precalc = category == "Precalculus"
            prompt = f"<|im_start|>user\nSolve this math problem step by step:\n{question}<|im_end|>\n<|im_start|>assistant\n"

            if is_precalc:
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
                outputs = llm.generate([prompt], greedy_params)
                cot = outputs[0].outputs[0].text
                pred = extract_boxed(cot)
                is_correct = answers_match(pred, gold)
                attempts_used = 1

            result = {
                "idx": idx,
                "problem_text": question,
                "gold_answer": gold,
                "category": category,
                "level": level,
                "cot_text": cot,
                "predicted_answer": pred,
                "is_correct": is_correct,
                "n_tokens_generated": len(outputs[0].outputs[0].token_ids),
                "attempts_used": attempts_used,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Save immediately
            output_path = output_dir / f"problem_{idx:05d}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            stats["success"] += 1
            if is_correct:
                stats["correct"] += 1

            # Track per-category
            if category not in category_stats:
                category_stats[category] = {"total": 0, "correct": 0}
            category_stats[category]["total"] += 1
            if is_correct:
                category_stats[category]["correct"] += 1

            # Periodic S3 sync
            if args.s3_bucket and (i + 1) % args.sync_every == 0:
                sync_to_s3(output_dir, args.s3_bucket)
                rate = stats["correct"] / stats["success"] * 100 if stats["success"] > 0 else 0
                print(f"\n  Progress: {stats['success']}/{len(remaining)}, Accuracy: {rate:.1f}%")

        except Exception as e:
            print(f"\nError on problem {idx}: {e}")
            stats["errors"] += 1
            error_path = output_dir / f"problem_{idx:05d}_error.json"
            with open(error_path, "w") as f:
                json.dump({"idx": idx, "error": str(e)}, f)

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

    print("\nPer-category breakdown:")
    print("-" * 50)
    for cat in sorted(category_stats.keys()):
        cs = category_stats[cat]
        rate = cs["correct"] / cs["total"] * 100 if cs["total"] > 0 else 0
        print(f"  {cat:<30} {cs['correct']:>4}/{cs['total']:<4} ({rate:>5.1f}%)")

    # Save summary
    summary = {
        "total_processed": stats["success"],
        "correct": stats["correct"],
        "accuracy": stats["correct"] / max(stats["success"], 1) * 100,
        "errors": stats["errors"],
        "category_stats": category_stats,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
