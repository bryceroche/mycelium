#!/usr/bin/env python3
"""
Full MATH 7B CoT Generation - Hybrid Approach

Runs Qwen-7B on the full MATH dataset (12.5K problems) with checkpointing.
Designed for parallel execution across multiple instances.

Usage:
    # Single instance (all problems):
    python math_full_7b_hybrid.py --output-dir /data/math_7b

    # Parallel across 4 instances:
    python math_full_7b_hybrid.py --shard 0 --num-shards 4 --output-dir /data/math_7b
    python math_full_7b_hybrid.py --shard 1 --num-shards 4 --output-dir /data/math_7b
    python math_full_7b_hybrid.py --shard 2 --num-shards 4 --output-dir /data/math_7b
    python math_full_7b_hybrid.py --shard 3 --num-shards 4 --output-dir /data/math_7b
"""

import json
import os
import subprocess
import torch
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse


def load_full_math():
    """Load full MATH dataset (~12.5K problems)."""
    from datasets import load_dataset

    # Full competition_math test set
    ds = load_dataset("hendrycks/competition_math", split="test")

    problems = []
    for i, item in enumerate(ds):
        problems.append({
            "idx": i,
            "problem": item["problem"],
            "answer": item["solution"],  # Full solution text
            "type": item["type"],  # Category
            "level": item["level"],  # Difficulty 1-5
        })

    return problems


def extract_gold_answer(solution_text: str) -> str:
    """Extract the boxed answer from MATH solution text."""
    # MATH solutions end with \boxed{answer}
    match = re.search(r'\\boxed\{([^}]+)\}', solution_text)
    if match:
        return match.group(1)

    # Fallback: try to find last number or expression
    lines = solution_text.strip().split('\n')
    for line in reversed(lines):
        if '=' in line:
            parts = line.split('=')
            if len(parts) >= 2:
                return parts[-1].strip()

    return ""


def compute_jsd(attn_row_i, attn_row_j):
    """Compute Jensen-Shannon divergence."""
    p = attn_row_i / (attn_row_i.sum() + 1e-10)
    q = attn_row_j / (attn_row_j.sum() + 1e-10)
    m = 0.5 * (p + q)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    m = np.clip(m, 1e-10, 1.0)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


def extract_boxed(text):
    """Extract answer from \\boxed{...} in generated text."""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    return matches[-1] if matches else None


def normalize_answer(s):
    """Basic answer normalization for comparison."""
    if s is None:
        return None
    s = str(s).strip()
    s = re.sub(r'\\boxed\{(.+)\}', r'\1', s)
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\frac', '').replace('\\sqrt', '')
    s = s.replace(' ', '').lower()
    # Remove LaTeX formatting
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = s.replace('{', '').replace('}', '')
    return s


def answers_match(pred, gold, tolerance=1e-6):
    """Check if answers match."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)

    if p == g:
        return True

    # Try numeric comparison
    try:
        p_num = float(p.replace(',', ''))
        g_num = float(g.replace(',', ''))
        if abs(g_num) < tolerance:
            return abs(p_num - g_num) < tolerance
        return abs(p_num - g_num) / max(abs(g_num), 1e-10) < tolerance
    except (ValueError, TypeError, AttributeError):
        pass

    return False


def get_completed_indices(output_dir: Path) -> set:
    """Get set of already completed problem indices."""
    completed = set()
    if not output_dir.exists():
        return completed

    for f in output_dir.glob("problem_*.json"):
        if "error" not in f.name:
            try:
                idx = int(f.stem.split("_")[1])
                completed.add(idx)
            except (ValueError, IndexError):
                continue

    return completed


def sync_to_s3(output_dir: Path, s3_bucket: str, prefix: str = "math_full_7b"):
    """Sync results to S3."""
    try:
        cmd = f"aws s3 sync {output_dir} s3://{s3_bucket}/{prefix}/"
        subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
        print(f"  [S3] Synced to s3://{s3_bucket}/{prefix}/")
    except Exception as e:
        print(f"  [S3] Sync error: {e}")


def process_problem(model, tokenizer, problem, device="cuda"):
    """Generate CoT solution with JSD capture."""
    question = problem["problem"]
    gold_solution = problem["answer"]
    gold_answer = extract_gold_answer(gold_solution)

    # Build prompt
    prompt = f"<|im_start|>user\nSolve this math problem step by step:\n{question}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    # Generate with attention output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute JSD trace
    jsd_trace = []
    if hasattr(outputs, 'attentions') and outputs.attentions:
        prev_attn = None
        for step_attn in outputs.attentions:
            if step_attn is None:
                continue
            curr_layer_attn = step_attn[-1]  # Last layer
            if prev_attn is not None:
                prev_row = prev_attn[0, :, -1, :].mean(dim=0).float().cpu().numpy()
                curr_row = curr_layer_attn[0, :, -1, :].mean(dim=0).float().cpu().numpy()
                max_len = max(len(prev_row), len(curr_row))
                prev_row = np.pad(prev_row, (0, max_len - len(prev_row)))
                curr_row = np.pad(curr_row, (0, max_len - len(curr_row)))
                jsd = compute_jsd(prev_row, curr_row)
                jsd_trace.append(float(jsd))
            prev_attn = curr_layer_attn

    # Extract predicted answer and check correctness
    predicted_answer = extract_boxed(generated_text)
    is_correct = answers_match(predicted_answer, gold_answer)

    # Extract problem numbers for later analysis
    problem_numbers = re.findall(r'[\d,]+\.?\d*', question)
    problem_numbers = [n.replace(',', '') for n in problem_numbers]

    result = {
        "idx": problem["idx"],
        "problem_text": question,
        "problem_numbers": problem_numbers,
        "gold_answer": gold_answer,
        "category": problem["type"],
        "level": problem["level"],
        "cot_text": generated_text,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "jsd_scores": jsd_trace,
        "n_tokens_generated": len(generated_ids),
        "timestamp": datetime.utcnow().isoformat(),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Full MATH 7B CoT Generation")
    parser.add_argument("--output-dir", default="/data/math_7b",
                        help="Output directory for results")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help="Model to use")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index for parallel execution (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards")
    parser.add_argument("--s3-bucket", default="mycelium-data",
                        help="S3 bucket for sync")
    parser.add_argument("--sync-every", type=int, default=50,
                        help="Sync to S3 every N problems")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of problems (for testing)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL MATH 7B CoT Generation")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    if args.shard is not None:
        print(f"Shard: {args.shard + 1}/{args.num_shards}")

    # Load dataset
    print("\nLoading full MATH dataset...")
    all_problems = load_full_math()
    print(f"Total problems in dataset: {len(all_problems)}")

    # Apply limit if specified
    if args.limit:
        all_problems = all_problems[:args.limit]
        print(f"Limited to: {len(all_problems)}")

    # Shard the problems
    if args.shard is not None:
        shard_size = len(all_problems) // args.num_shards
        start_idx = args.shard * shard_size
        end_idx = start_idx + shard_size if args.shard < args.num_shards - 1 else len(all_problems)
        problems = all_problems[start_idx:end_idx]
        print(f"This shard: problems {start_idx} to {end_idx} ({len(problems)} total)")
    else:
        problems = all_problems

    # Check existing progress
    completed = get_completed_indices(output_dir)
    remaining = [p for p in problems if p["idx"] not in completed]

    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("\nAll problems in this shard completed!")
        return

    # Load model
    print(f"\nLoading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded!")

    # Process problems
    stats = {"success": 0, "correct": 0, "errors": 0}
    category_stats = {}

    for i, problem in enumerate(tqdm(remaining, desc="Generating")):
        try:
            result = process_problem(model, tokenizer, problem, args.device)

            # Save immediately
            output_path = output_dir / f"problem_{problem['idx']:05d}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            stats["success"] += 1
            if result["is_correct"]:
                stats["correct"] += 1

            # Track per-category stats
            cat = result["category"]
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "correct": 0}
            category_stats[cat]["total"] += 1
            if result["is_correct"]:
                category_stats[cat]["correct"] += 1

            # Periodic S3 sync
            if args.s3_bucket and (i + 1) % args.sync_every == 0:
                sync_to_s3(output_dir, args.s3_bucket)

                # Print interim stats
                rate = stats["correct"] / stats["success"] * 100 if stats["success"] > 0 else 0
                print(f"\n  Progress: {stats['success']}/{len(remaining)}, Accuracy: {rate:.1f}%")

        except Exception as e:
            print(f"\nError on problem {problem['idx']}: {e}")
            stats["errors"] += 1

            error_path = output_dir / f"problem_{problem['idx']:05d}_error.json"
            with open(error_path, "w") as f:
                json.dump({
                    "idx": problem["idx"],
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }, f)

    # Final S3 sync
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
        "shard": args.shard,
        "num_shards": args.num_shards,
        "total_processed": stats["success"],
        "correct": stats["correct"],
        "accuracy": stats["correct"] / max(stats["success"], 1) * 100,
        "errors": stats["errors"],
        "category_stats": category_stats,
        "timestamp": datetime.utcnow().isoformat(),
    }

    shard_suffix = f"_shard{args.shard}" if args.shard is not None else ""
    with open(output_dir / f"summary{shard_suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
