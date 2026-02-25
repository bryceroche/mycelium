#!/usr/bin/env python3
"""
Full MATH 7B with vLLM - Fast inference for hybrid approach.

Runs 7B on full MATH (~12.5K), saves failures for 72B pass.
With vLLM on 8x A100, expect ~5s/problem = ~17 hours for full dataset.
Or use fewer GPUs for 7B since it's small.

Usage:
    python math_full_7b_vllm.py --output-dir /data/math_7b --tensor-parallel 2
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

    # MATH dataset has multiple configs
    configs = ['algebra', 'counting_and_probability', 'geometry',
               'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

    problems = []
    idx = 0
    for config in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        for item in ds:
            problems.append({
                "idx": idx,
                "problem": item["problem"],
                "solution": item["solution"],
                "type": config.replace("_", " ").title(),
                "level": item.get("level", "Unknown"),
            })
            idx += 1

    return problems


def extract_gold_answer(solution_text: str) -> str:
    """Extract the boxed answer from MATH solution text."""
    # Handle nested braces properly
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


def sync_to_s3(output_dir: Path, s3_bucket: str, prefix: str = "math_full_7b"):
    """Sync results to S3."""
    try:
        cmd = f"aws s3 sync {output_dir} s3://{s3_bucket}/{prefix}/"
        subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
        print(f"  [S3] Synced to s3://{s3_bucket}/{prefix}/")
    except Exception as e:
        print(f"  [S3] Sync error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/math_7b")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--s3-bucket", default="mycelium-data")
    parser.add_argument("--sync-every", type=int, default=200)
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="TP size (1-2 for 7B is enough)")
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Full MATH 7B (vLLM)")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")

    # Load dataset
    print("\nLoading MATH dataset...")
    all_problems = load_full_math()
    print(f"Total problems: {len(all_problems)}")

    # Shard the problems if specified
    if args.shard is not None:
        shard_size = len(all_problems) // args.num_shards
        start_idx = args.shard * shard_size
        end_idx = start_idx + shard_size if args.shard < args.num_shards - 1 else len(all_problems)
        problems = all_problems[start_idx:end_idx]
        print(f"Shard {args.shard + 1}/{args.num_shards}: problems {start_idx}-{end_idx} ({len(problems)} total)")
    else:
        problems = all_problems

    if args.limit:
        problems = problems[:args.limit]
        print(f"Limited to {len(problems)}")

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

    # Greedy sampling for consistency
    params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )

    # Process problems
    stats = {"success": 0, "correct": 0, "errors": 0}
    category_stats = {}
    failure_indices = []

    for i, problem in enumerate(tqdm(remaining, desc="Processing")):
        idx = problem["idx"]
        question = problem["problem"]
        gold = extract_gold_answer(problem["solution"])
        category = problem.get("type", "Unknown")
        level = problem.get("level", "Unknown")

        try:
            prompt = f"<|im_start|>user\nSolve this math problem step by step:\n{question}<|im_end|>\n<|im_start|>assistant\n"

            outputs = llm.generate([prompt], params)
            cot = outputs[0].outputs[0].text
            pred = extract_boxed(cot)
            is_correct = answers_match(pred, gold)

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
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Save immediately
            output_path = output_dir / f"problem_{idx:05d}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            stats["success"] += 1
            if is_correct:
                stats["correct"] += 1
            else:
                failure_indices.append(idx)

            # Track per-category
            if category not in category_stats:
                category_stats[category] = {"total": 0, "correct": 0}
            category_stats[category]["total"] += 1
            if is_correct:
                category_stats[category]["correct"] += 1

            # Periodic S3 sync and save failure indices
            if args.s3_bucket and (i + 1) % args.sync_every == 0:
                # Save failure indices for 72B
                with open(output_dir / "failure_indices.json", "w") as f:
                    json.dump(failure_indices, f)
                sync_to_s3(output_dir, args.s3_bucket)
                rate = stats["correct"] / stats["success"] * 100 if stats["success"] > 0 else 0
                print(f"\n  Progress: {stats['success']}/{len(remaining)}, Accuracy: {rate:.1f}%, Failures: {len(failure_indices)}")

        except Exception as e:
            print(f"\nError on problem {idx}: {e}")
            stats["errors"] += 1
            failure_indices.append(idx)  # Count errors as failures

    # Save final failure indices
    with open(output_dir / "failure_indices.json", "w") as f:
        json.dump(failure_indices, f)

    # Final sync
    if args.s3_bucket:
        sync_to_s3(output_dir, args.s3_bucket)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {stats['success']}")
    print(f"Correct: {stats['correct']} ({stats['correct']/max(stats['success'],1)*100:.1f}%)")
    print(f"Failures: {len(failure_indices)} (for 72B pass)")
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
        "failures": len(failure_indices),
        "errors": stats["errors"],
        "category_stats": category_stats,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"Failure indices saved for 72B pass: {output_dir}/failure_indices.json")

    # Estimate 72B cost
    print("\n" + "=" * 70)
    print("72B Pass Estimate")
    print("=" * 70)
    print(f"Failures to process: {len(failure_indices)}")
    print(f"Estimated time @ 15s/problem: {len(failure_indices) * 15 / 3600:.1f} hours")
    print(f"Estimated cost @ $14.32/hr: ${len(failure_indices) * 15 / 3600 * 14.32:.0f}")


if __name__ == "__main__":
    main()
