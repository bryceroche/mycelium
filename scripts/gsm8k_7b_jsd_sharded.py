#!/usr/bin/env python3
"""
GSM8K 7B with JSD Attention Capture - Sharded for parallel execution.

Generates CoT solutions with full attention capture for segmentation training.
Uses HuggingFace (not vLLM) to get attention matrices.

Usage:
    # Single shard (all problems):
    python gsm8k_7b_jsd_sharded.py --output-dir /data/gsm8k_7b

    # Parallel across 4 shards:
    CUDA_VISIBLE_DEVICES=0,1 python gsm8k_7b_jsd_sharded.py --shard 0 --num-shards 4 --output-dir /data/gsm8k_7b/shard_0
"""

import json
import subprocess
import torch
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse


def load_gsm8k():
    """Load GSM8K dataset."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")

    problems = []
    for i, item in enumerate(ds):
        # Extract numeric answer from "#### answer" format
        answer_text = item["answer"]
        match = re.search(r'####\s*(.+)$', answer_text)
        gold_answer = match.group(1).strip() if match else ""

        problems.append({
            "idx": i,
            "question": item["question"],
            "answer": answer_text,
            "gold_answer": gold_answer,
        })

    return problems


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


def extract_answer(text):
    """Extract numeric answer from generated text."""
    # Look for #### pattern first
    match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()

    # Look for "answer is" pattern
    match = re.search(r'answer\s+is\s+[\$]?([\d,\.]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')

    # Look for boxed
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1)

    # Last number in text
    numbers = re.findall(r'[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def normalize_answer(s):
    """Normalize answer for comparison."""
    if s is None:
        return None
    s = str(s).strip()
    s = s.replace(',', '').replace('$', '').replace('%', '')
    try:
        return str(float(s))
    except:
        return s.lower()


def answers_match(pred, gold):
    """Check if answers match."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-4
    except:
        return False


def process_problem(model, tokenizer, problem, device="cuda"):
    """Generate CoT with full JSD attention capture."""
    question = problem["question"]
    gold = problem["gold_answer"]

    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    # Generate with attention output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute JSD trace from attention
    jsd_trace = []
    if hasattr(outputs, 'attentions') and outputs.attentions:
        prev_attn = None
        for step_attn in outputs.attentions:
            if step_attn is None:
                continue
            # Last layer attention
            curr_layer_attn = step_attn[-1]
            if prev_attn is not None:
                # Average across heads, take last token's attention row
                prev_row = prev_attn[0, :, -1, :].mean(dim=0).float().cpu().numpy()
                curr_row = curr_layer_attn[0, :, -1, :].mean(dim=0).float().cpu().numpy()
                # Pad to same length
                max_len = max(len(prev_row), len(curr_row))
                prev_row = np.pad(prev_row, (0, max_len - len(prev_row)))
                curr_row = np.pad(curr_row, (0, max_len - len(curr_row)))
                jsd = compute_jsd(prev_row, curr_row)
                jsd_trace.append(float(jsd))
            prev_attn = curr_layer_attn

    # Extract and check answer
    pred = extract_answer(generated_text)
    is_correct = answers_match(pred, gold)

    # Get token strings for alignment
    tokens = [tokenizer.decode([t]) for t in generated_ids.tolist()]

    result = {
        "idx": problem["idx"],
        "question": question,
        "gold_answer": gold,
        "generated_cot": generated_text,
        "predicted_answer": pred,
        "is_correct": is_correct,
        "jsd_trace": jsd_trace,
        "tokens": tokens,
        "n_tokens": len(generated_ids),
        "timestamp": datetime.utcnow().isoformat(),
    }

    return result


def sync_to_s3(output_dir: Path, s3_bucket: str, prefix: str):
    """Sync results to S3."""
    try:
        cmd = f"aws s3 sync {output_dir} s3://{s3_bucket}/{prefix}/"
        subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
        print(f"  [S3] Synced to s3://{s3_bucket}/{prefix}/")
    except Exception as e:
        print(f"  [S3] Sync error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data/gsm8k_7b")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--s3-bucket", default="mycelium-data")
    parser.add_argument("--s3-prefix", default="gsm8k_7b_jsd")
    parser.add_argument("--sync-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GSM8K 7B with JSD Attention Capture")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")
    if args.shard is not None:
        print(f"Shard: {args.shard + 1}/{args.num_shards}")

    # Load dataset
    print("\nLoading GSM8K...")
    all_problems = load_gsm8k()
    print(f"Total problems: {len(all_problems)}")

    # Shard if specified
    if args.shard is not None:
        shard_size = len(all_problems) // args.num_shards
        start_idx = args.shard * shard_size
        end_idx = start_idx + shard_size if args.shard < args.num_shards - 1 else len(all_problems)
        problems = all_problems[start_idx:end_idx]
        print(f"This shard: problems {start_idx}-{end_idx} ({len(problems)} total)")
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
        print("\nAll problems completed!")
        return

    # Load model with HuggingFace (for attention capture)
    print(f"\nLoading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Required for attention output
    )
    model.eval()
    print("Model loaded!")

    # Process problems
    stats = {"success": 0, "correct": 0, "errors": 0}

    for i, problem in enumerate(tqdm(remaining, desc="Processing")):
        try:
            result = process_problem(model, tokenizer, problem)

            # Save immediately
            output_path = output_dir / f"problem_{problem['idx']:05d}.json"
            with open(output_path, "w") as f:
                json.dump(result, f)

            stats["success"] += 1
            if result["is_correct"]:
                stats["correct"] += 1

            # Periodic sync
            if args.s3_bucket and (i + 1) % args.sync_every == 0:
                s3_prefix = f"{args.s3_prefix}/shard_{args.shard}" if args.shard is not None else args.s3_prefix
                sync_to_s3(output_dir, args.s3_bucket, s3_prefix)
                rate = stats["correct"] / stats["success"] * 100 if stats["success"] > 0 else 0
                print(f"\n  Progress: {stats['success']}/{len(remaining)}, Accuracy: {rate:.1f}%")

        except Exception as e:
            print(f"\nError on problem {problem['idx']}: {e}")
            stats["errors"] += 1

    # Final sync
    if args.s3_bucket:
        s3_prefix = f"{args.s3_prefix}/shard_{args.shard}" if args.shard is not None else args.s3_prefix
        sync_to_s3(output_dir, args.s3_bucket, s3_prefix)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed: {stats['success']}")
    print(f"Correct: {stats['correct']} ({stats['correct']/max(stats['success'],1)*100:.1f}%)")
    print(f"Errors: {stats['errors']}")

    # Save summary
    summary = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "total_processed": stats["success"],
        "correct": stats["correct"],
        "accuracy": stats["correct"] / max(stats["success"], 1) * 100,
        "errors": stats["errors"],
        "timestamp": datetime.utcnow().isoformat(),
    }
    shard_suffix = f"_shard{args.shard}" if args.shard is not None else ""
    with open(output_dir / f"summary{shard_suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
