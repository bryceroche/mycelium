#!/usr/bin/env python3
"""
MATH500 72B: Process 7B Failures

Runs Qwen-72B on the 207 problems that 7B failed.
Special handling for Precalculus: 3 attempts at temp 0.7

Usage:
    python scripts/math500_72b_failures.py --output-dir /opt/dlami/nvme/math500_72b
"""

import json
import os
import subprocess
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import re


def load_math500_from_7b(results_dir="/opt/dlami/nvme/math500_7b"):
    """Load MATH500 problems from 7B results."""
    problems = {}
    results_path = Path(results_dir)
    for f in sorted(results_path.glob("problem_*.json")):
        if "error" in f.name:
            continue
        with open(f) as fp:
            r = json.load(fp)
            # Convert to standard format
            problems[r["idx"]] = {
                "problem": r["question"],
                "answer": r["gold_answer"],
                "type": r.get("category", "Unknown"),
                "level": r.get("level", "Unknown"),
            }
    return problems


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
    """Extract answer from \\boxed{...} handling nested braces."""
    # Find all \boxed{ occurrences and extract balanced content
    results = []
    idx = 0
    while True:
        start = text.find(r'\boxed{', idx)
        if start == -1:
            break
        # Find matching closing brace
        brace_count = 1
        content_start = start + 7  # len(r'\boxed{')
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


def generate_cot(model, tokenizer, question, device="cuda",
                 temperature=0.0, do_sample=False, max_tokens=1024):
    """Generate CoT with attention capture."""
    prompt = f"<|im_start|>user\nSolve this math problem step by step:\n{question}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "output_attentions": True,
        "return_dict_in_generate": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }

    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute JSD trace
    jsd_trace = []
    if hasattr(outputs, 'attentions') and outputs.attentions:
        prev_attn = None
        for step_attn in outputs.attentions:
            if step_attn is None:
                continue
            curr_layer_attn = step_attn[-1]
            if prev_attn is not None:
                prev_row = prev_attn[0, :, -1, :].mean(dim=0).float().cpu().numpy()
                curr_row = curr_layer_attn[0, :, -1, :].mean(dim=0).float().cpu().numpy()
                max_len = max(len(prev_row), len(curr_row))
                prev_row = np.pad(prev_row, (0, max_len - len(prev_row)))
                curr_row = np.pad(curr_row, (0, max_len - len(curr_row)))
                jsd = compute_jsd(prev_row, curr_row)
                jsd_trace.append(float(jsd))
            prev_attn = curr_layer_attn

    return generated_text, jsd_trace, len(generated_ids)


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
    parser.add_argument("--output-dir", default="/opt/dlami/nvme/math500_72b")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-72B-Instruct")
    parser.add_argument("--failure-indices", default="/opt/dlami/nvme/math500_72b/failure_indices.json")
    parser.add_argument("--results-7b", default="/opt/dlami/nvme/math500_7b",
                        help="Directory with 7B results")
    parser.add_argument("--s3-bucket", default="mycelium-data")
    parser.add_argument("--sync-every", type=int, default=10)
    parser.add_argument("--precalc-attempts", type=int, default=3,
                        help="Number of attempts for Precalculus problems")
    parser.add_argument("--precalc-temp", type=float, default=0.7,
                        help="Temperature for Precalculus sampling")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MATH500 72B: Processing 7B Failures")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print(f"Precalculus: {args.precalc_attempts} attempts @ temp {args.precalc_temp}")

    # Load failure indices
    with open(args.failure_indices) as f:
        failure_indices = json.load(f)
    print(f"Failures to process: {len(failure_indices)}")

    # Load problems from 7B results
    print(f"\nLoading problems from 7B results ({args.results_7b})...")
    problems = load_math500_from_7b(args.results_7b)
    print(f"Loaded {len(problems)} problems")

    # Check existing progress
    completed = set()
    for f in output_dir.glob("problem_*.json"):
        if "error" not in f.name:
            try:
                idx = int(f.stem.split("_")[1])
                completed.add(idx)
            except:
                pass

    remaining = [i for i in failure_indices if i not in completed]
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("\nAll failures already processed!")
        return

    # Load model
    print(f"\nLoading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("Model loaded!")

    # Process failures
    stats = {"success": 0, "correct": 0, "errors": 0}
    precalc_stats = {"total": 0, "correct": 0, "attempts_used": []}

    for i, idx in enumerate(tqdm(remaining, desc="Processing")):
        if idx not in problems:
            print(f"\nWarning: Problem {idx} not found in 7B results, skipping")
            continue
        problem = problems[idx]
        question = problem.get("problem", "")
        gold = problem.get("answer", "")
        category = problem.get("type", "unknown")
        level = problem.get("level", "unknown")

        try:
            is_precalc = category == "Precalculus"

            if is_precalc:
                # Multiple attempts for Precalculus
                precalc_stats["total"] += 1
                best_result = None

                for attempt in range(args.precalc_attempts):
                    cot, jsd_trace, n_tokens = generate_cot(
                        model, tokenizer, question,
                        temperature=args.precalc_temp,
                        do_sample=True
                    )
                    pred = extract_boxed(cot)
                    is_correct = answers_match(pred, gold)

                    if is_correct:
                        best_result = (cot, jsd_trace, n_tokens, pred, attempt + 1)
                        break
                    elif best_result is None:
                        best_result = (cot, jsd_trace, n_tokens, pred, attempt + 1)

                cot, jsd_trace, n_tokens, pred, attempts_used = best_result
                is_correct = answers_match(pred, gold)

                if is_correct:
                    precalc_stats["correct"] += 1
                precalc_stats["attempts_used"].append(attempts_used)

            else:
                # Single greedy attempt for other categories
                cot, jsd_trace, n_tokens = generate_cot(
                    model, tokenizer, question,
                    temperature=0.0, do_sample=False
                )
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
                "jsd_trace": jsd_trace,
                "n_tokens_generated": n_tokens,
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

        except Exception as e:
            print(f"\nError on problem {idx}: {e}")
            stats["errors"] += 1
            error_path = output_dir / f"problem_{idx:04d}_error.json"
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

    if precalc_stats["total"] > 0:
        print(f"\nPrecalculus:")
        print(f"  Total: {precalc_stats['total']}")
        print(f"  Correct: {precalc_stats['correct']} ({precalc_stats['correct']/precalc_stats['total']*100:.1f}%)")
        avg_attempts = np.mean(precalc_stats["attempts_used"])
        print(f"  Avg attempts used: {avg_attempts:.1f}")

    # Save summary
    summary = {
        "total_processed": stats["success"],
        "correct": stats["correct"],
        "errors": stats["errors"],
        "precalculus": precalc_stats,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
