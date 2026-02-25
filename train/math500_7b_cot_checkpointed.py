#!/usr/bin/env python3
"""
MATH500 CoT Generation with Checkpointing

Generates chain-of-thought solutions using Qwen-7B with attention capture.
Saves after EVERY problem to prevent data loss on instance termination.

Usage:
    python scripts/math500_7b_cot_checkpointed.py --output-dir /opt/dlami/nvme/math500_7b

Features:
- Resumes from existing checkpoint (checks completed problem indices)
- Saves each result immediately after generation
- Optional S3 sync every N problems
- Captures full attention for JSD segmentation
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


def load_math500():
    """Load MATH500 benchmark dataset."""
    from datasets import load_dataset

    # MATH500 is a curated subset - check common locations
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return list(ds)
    except Exception:
        pass

    # Fallback to full MATH and take first 500
    try:
        ds = load_dataset("hendrycks/competition_math", split="test")
        return list(ds)[:500]
    except Exception as e:
        print(f"Failed to load MATH dataset: {e}")
        raise


def get_completed_indices(output_dir: Path) -> set:
    """Get set of already completed problem indices."""
    completed = set()

    if not output_dir.exists():
        return completed

    for f in output_dir.glob("problem_*.json"):
        try:
            idx = int(f.stem.split("_")[1])
            completed.add(idx)
        except (ValueError, IndexError):
            continue

    return completed


def compute_jsd(attn_row_i, attn_row_j):
    """Compute Jensen-Shannon divergence between two attention distributions."""
    p = attn_row_i / (attn_row_i.sum() + 1e-10)
    q = attn_row_j / (attn_row_j.sum() + 1e-10)

    m = 0.5 * (p + q)

    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    m = np.clip(m, 1e-10, 1.0)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return 0.5 * (kl_pm + kl_qm)


def extract_answer(solution_text: str) -> str:
    """Extract the boxed answer from solution text."""
    import re

    # Try \boxed{...}
    match = re.search(r'\\boxed\{([^}]+)\}', solution_text)
    if match:
        return match.group(1)

    # Try final number/expression
    lines = solution_text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('#'):
            # Look for = followed by answer
            eq_match = re.search(r'=\s*([^\s=]+)$', line)
            if eq_match:
                return eq_match.group(1)
            # Just take last word/number
            parts = line.split()
            if parts:
                return parts[-1]

    return ""


def sync_to_s3(output_dir: Path, s3_bucket: str):
    """Sync results to S3."""
    try:
        cmd = f"aws s3 sync {output_dir} s3://{s3_bucket}/math500_7b/"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"  [S3] Synced to s3://{s3_bucket}/math500_7b/")
        else:
            print(f"  [S3] Sync failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("  [S3] Sync timed out")
    except Exception as e:
        print(f"  [S3] Sync error: {e}")


def process_problem(model, tokenizer, problem, idx, device="cuda"):
    """Generate CoT solution with attention capture for a single problem."""
    question = problem.get("problem", problem.get("question", ""))
    gold_answer = problem.get("answer", problem.get("solution", ""))
    category = problem.get("type", problem.get("subject", "unknown"))
    level = problem.get("level", "unknown")

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

    # Decode generated text
    generated_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute JSD trace from attention
    jsd_trace = []

    if hasattr(outputs, 'attentions') and outputs.attentions:
        prev_attn = None
        for step_idx, step_attn in enumerate(outputs.attentions):
            if step_attn is None:
                continue

            # Get last layer attention
            curr_layer_attn = step_attn[-1]  # Last layer

            if prev_attn is not None:
                # Average over heads, get last token's attention
                prev_row = prev_attn[0, :, -1, :].mean(dim=0).float().cpu().numpy()
                curr_row = curr_layer_attn[0, :, -1, :].mean(dim=0).float().cpu().numpy()

                # Pad to same length
                max_len = max(len(prev_row), len(curr_row))
                prev_row = np.pad(prev_row, (0, max_len - len(prev_row)))
                curr_row = np.pad(curr_row, (0, max_len - len(curr_row)))

                jsd = compute_jsd(prev_row, curr_row)
                jsd_trace.append(float(jsd))

            prev_attn = curr_layer_attn

    # Extract predicted answer
    predicted_answer = extract_answer(generated_text)

    result = {
        "idx": idx,
        "question": question,
        "gold_answer": gold_answer,
        "category": category,
        "level": level,
        "generated_cot": generated_text,
        "predicted_answer": predicted_answer,
        "jsd_trace": jsd_trace,
        "n_tokens_generated": len(generated_ids),
        "timestamp": datetime.utcnow().isoformat(),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="MATH500 7B CoT Generation with Checkpointing")
    parser.add_argument("--output-dir", default="/opt/dlami/nvme/math500_7b",
                        help="Output directory for results")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help="Model to use")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems")
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket for periodic sync")
    parser.add_argument("--sync-every", type=int, default=10,
                        help="Sync to S3 every N problems")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MATH500 7B CoT Generation (Checkpointed)")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    print(f"S3 bucket: {args.s3_bucket or 'disabled'}")

    # Load dataset
    print("\nLoading MATH500 dataset...")
    problems = load_math500()
    if args.limit:
        problems = problems[:args.limit]
    print(f"Total problems: {len(problems)}")

    # Check for existing progress
    completed = get_completed_indices(output_dir)
    remaining_indices = [i for i in range(len(problems)) if i not in completed]

    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining_indices)}")

    if not remaining_indices:
        print("\nAll problems already completed!")
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
        attn_implementation="eager",  # Required for attention output
    )
    model.eval()
    print("Model loaded!")

    # Process remaining problems
    print(f"\nStarting generation...")

    stats = {"success": 0, "errors": 0}

    for i, idx in enumerate(tqdm(remaining_indices, desc="Generating")):
        problem = problems[idx]

        try:
            result = process_problem(model, tokenizer, problem, idx, args.device)

            # Save immediately
            output_path = output_dir / f"problem_{idx:04d}.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            stats["success"] += 1

            # Periodic S3 sync
            if args.s3_bucket and (i + 1) % args.sync_every == 0:
                sync_to_s3(output_dir, args.s3_bucket)

        except Exception as e:
            print(f"\nError on problem {idx}: {e}")
            stats["errors"] += 1

            # Save error record
            error_path = output_dir / f"problem_{idx:04d}_error.json"
            with open(error_path, "w") as f:
                json.dump({
                    "idx": idx,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }, f, indent=2)

    # Final S3 sync
    if args.s3_bucket:
        print("\nFinal S3 sync...")
        sync_to_s3(output_dir, args.s3_bucket)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful: {stats['success']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total completed: {len(completed) + stats['success']}/{len(problems)}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_problems": len(problems),
            "completed": len(completed) + stats["success"],
            "errors": stats["errors"],
            "timestamp": datetime.utcnow().isoformat(),
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
