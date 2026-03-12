#!/usr/bin/env python3
"""
Mycelium v7 Evaluation - Default Pipeline

Batched stage inference:
  Stage 1: C1-A → scaffolds for all problems
  Stage 2: Canonicalizer → telegrams for all problems
  Stage 3: Oracle → execute and verify

Usage:
    # Default: 50 problems, downloads models from S3
    python scripts/eval.py

    # Custom:
    python scripts/eval.py --problems data/math_50_test.jsonl --n 20 --output results.json
"""

import os
import sys
import json
import argparse
import subprocess
import time
import gc
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "plan"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from oracle import execute_sequence, compare_answers, parse_telegram_expr

SCAFFOLD_TO_VERB = {
    "SETUP": "GIVEN", "COMPUTE": "EVAL", "SOLVE": "SOLVE",
    "SUBSTITUTE": "SUBS", "SIMPLIFY": "SIMPLIFY", "EXPAND": "EXPAND",
    "THEOREM": "APPLY", "OTHER": "EVAL", "ANSWER": "ANSWER",
}


def download_model(s3_path: str, local_path: str):
    """Download model from S3 if not exists."""
    if not os.path.exists(local_path):
        print(f"  Downloading from {s3_path}...")
        os.makedirs(local_path, exist_ok=True)
        subprocess.run(['aws', 's3', 'sync', s3_path, local_path], capture_output=True)
    return local_path


def load_problems(path: str, n: int = None):
    """Load problems from JSON or JSONL."""
    if path.endswith('.json'):
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, dict):
                problems = list(data.values())
            else:
                problems = data
    else:
        problems = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))

    # Normalize field names
    for p in problems:
        if 'problem' in p and 'text' not in p:
            p['text'] = p['problem']
        if 'answer' in p and 'gold_answer' not in p:
            p['gold_answer'] = str(p['answer'])

    if n:
        problems = problems[:n]
    return problems


# ─────────────────────────────────────────────────────────────
# Stage 1: C1-A Scaffolds
# ─────────────────────────────────────────────────────────────

def stage1_scaffolds(problems, model_path=None, device="cuda"):
    """Generate scaffolds for all problems."""
    print("\n=== Stage 1: C1-A Scaffolds ===")
    print(f"  Device: {device}")

    base = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, trust_remote_code=True)
    if model_path:
        model = PeftModel.from_pretrained(model, model_path)
    model = model.to(device)
    model.eval()

    SCAFFOLD_TYPES = ["SETUP", "SUBSTITUTE", "SIMPLIFY", "SOLVE", "COMPUTE", "THEOREM", "OTHER"]

    for i, p in enumerate(problems):
        prompt = f"Problem: {p['text'][:300]}\nPredict the solution scaffold:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)

        pred = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip().upper()

        types = []
        for token in pred.replace(",", " ").replace("[", "").replace("]", "").split():
            if token in SCAFFOLD_TYPES:
                types.append(token)
                if len(types) >= 10:
                    break

        if not types:
            types = ["SETUP", "COMPUTE", "COMPUTE"]

        p['scaffold'] = types

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(problems)}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Done: {len(problems)} problems")


# ─────────────────────────────────────────────────────────────
# Stage 2: Canonicalizer Telegrams
# ─────────────────────────────────────────────────────────────

def stage2_canonicalizer(problems, model_path, device="cuda"):
    """Generate telegrams for all problems (step-at-a-time)."""
    print("\n=== Stage 2: Canonicalizer ===")
    print(f"  Device: {device}")

    base = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, model_path)
    model = model.to(device)
    model.eval()

    total_steps = 0
    for i, p in enumerate(problems):
        telegrams = []

        for step_i, scaffold_type in enumerate(p['scaffold']):
            # Build prompt with context
            prev_str = "\n".join(telegrams) if telegrams else "(none)"
            prompt = f"""Problem: {p['text'][:200]}
Structure: {scaffold_type} (step {step_i+1} of {len(p['scaffold'])})
Previous steps:
{prev_str}
Next instruction:"""

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=newline_token
                )

            response = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
            telegram = response.split('\n')[0].strip()

            # Ensure telegram has verb prefix
            verb = SCAFFOLD_TO_VERB.get(scaffold_type, 'EVAL')
            if not any(telegram.startswith(v) for v in SCAFFOLD_TO_VERB.values()):
                telegram = f"{verb} {telegram}"

            telegrams.append(telegram)
            total_steps += 1

        # Add ANSWER at end if not present
        if telegrams and not telegrams[-1].startswith('ANSWER'):
            telegrams.append("ANSWER _prev")

        p['telegrams'] = telegrams

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(problems)} ({total_steps} steps)")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Done: {total_steps} total steps")


# ─────────────────────────────────────────────────────────────
# Stage 3: Oracle Execution
# ─────────────────────────────────────────────────────────────

def stage3_oracle(problems):
    """Execute telegrams and compare to gold."""
    print("\n=== Stage 3: Oracle Execution ===")

    for p in problems:
        result = execute_sequence(p['telegrams'], timeout=5)

        p['n_executed'] = result['n_success']
        p['n_total'] = result['n_total']
        p['execution_rate'] = result['execution_rate']
        p['chain_success'] = result['success']
        p['predicted_answer'] = str(result['final_answer']) if result['final_answer'] else None
        p['step_results'] = result['results']
        p['errors'] = result['errors']

        # Compare to gold
        pred = result['final_answer']
        gold = p.get('gold_answer', '')

        if pred is not None and gold:
            try:
                pred_expr = parse_telegram_expr(str(pred))
                p['correct'] = compare_answers(pred_expr, gold, timeout=5)
            except:
                p['correct'] = str(pred).strip() == str(gold).strip()
        else:
            p['correct'] = False

    print(f"  Done: {len(problems)} problems")


# ─────────────────────────────────────────────────────────────
# Error Attribution
# ─────────────────────────────────────────────────────────────

def error_attribution(problems):
    """Analyze error patterns."""
    print("\n=== Error Attribution ===")

    categories = defaultdict(int)
    scaffold_errors = defaultdict(lambda: defaultdict(int))

    for p in problems:
        if p['correct']:
            categories['correct'] += 1
            continue

        # Check execution errors
        has_parse_error = False
        has_timeout = False
        failed_scaffold = None

        for step in p.get('step_results', []):
            if not step['success']:
                if 'timeout' in str(step.get('error', '')).lower():
                    has_timeout = True
                else:
                    has_parse_error = True
                # Extract scaffold type from telegram
                telegram = step.get('telegram', '')
                for scaffold, verb in SCAFFOLD_TO_VERB.items():
                    if telegram.startswith(verb):
                        failed_scaffold = scaffold
                        scaffold_errors[scaffold]['failed'] += 1
                        break

        if has_timeout:
            categories['timeout'] += 1
        elif has_parse_error:
            categories['parse_error'] += 1
        elif not p['chain_success']:
            categories['chain_failed'] += 1
        elif p['predicted_answer'] is None:
            categories['no_answer'] += 1
        else:
            categories['wrong_answer'] += 1

    total = len(problems)
    print(f"\n  Categories (n={total}):")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count} ({100*count/total:.1f}%)")

    if scaffold_errors:
        print(f"\n  Scaffold failures:")
        for scaffold, errors in sorted(scaffold_errors.items()):
            print(f"    {scaffold}: {errors['failed']} failures")

    return dict(categories)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mycelium v7 Evaluation")
    parser.add_argument("--problems", default="data/math_50_test.jsonl",
                       help="Path to test problems (JSON or JSONL)")
    parser.add_argument("--n", type=int, default=50,
                       help="Number of problems to evaluate")
    parser.add_argument("--output", default="eval_results.json",
                       help="Output file path")
    parser.add_argument("--canonicalizer", default=None,
                       help="Path to canonicalizer model (local or S3)")
    parser.add_argument("--c1a", default=None,
                       help="Path to C1-A model (local or S3)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-s3-upload", action="store_true",
                       help="Skip uploading results to S3")
    args = parser.parse_args()

    print("=" * 60)
    print("        MYCELIUM v7 EVALUATION - STAGED INFERENCE")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Problems: {args.problems}")
    print(f"N: {args.n}")

    start_time = time.time()

    # Load problems
    print("\nLoading problems...")
    problems = load_problems(args.problems, args.n)
    print(f"  Loaded {len(problems)} problems")

    # Download/resolve canonicalizer path
    if args.canonicalizer:
        if args.canonicalizer.startswith('s3://'):
            canon_path = download_model(args.canonicalizer, '/tmp/canonicalizer')
        else:
            canon_path = args.canonicalizer
    else:
        # Default: use latest verified canonicalizer from S3
        canon_path = download_model(
            's3://mycelium-data/models/canonicalizer_v3_verified/',
            '/tmp/canonicalizer_v3_verified'
        )

    # C1-A path (optional)
    c1a_path = None
    if args.c1a:
        if args.c1a.startswith('s3://'):
            c1a_path = download_model(args.c1a, '/tmp/c1a')
        else:
            c1a_path = args.c1a

    # Run staged inference
    stage1_scaffolds(problems, model_path=c1a_path, device=args.device)
    stage2_canonicalizer(problems, model_path=canon_path, device=args.device)
    stage3_oracle(problems)

    # Compute metrics
    elapsed = time.time() - start_time
    n = len(problems)

    exec_rates = [p['execution_rate'] for p in problems]
    chain_success = sum(1 for p in problems if p['chain_success'])
    correct = sum(1 for p in problems if p['correct'])

    metrics = {
        'n_problems': n,
        'per_step_execution_rate': sum(exec_rates) / n,
        'full_chain_success_rate': chain_success / n,
        'answer_accuracy': correct / n,
        'elapsed_seconds': elapsed,
    }

    # Error attribution
    error_cats = error_attribution(problems)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Per-step execution rate:  {metrics['per_step_execution_rate']:.1%}")
    print(f"Full-chain success:       {metrics['full_chain_success_rate']:.1%}")
    print(f"Answer correctness:       {metrics['answer_accuracy']:.1%}")
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/n:.1f}s per problem)")
    print("=" * 60)

    # Save results
    output = {
        'metrics': metrics,
        'error_attribution': error_cats,
        'config': {
            'canonicalizer': canon_path,
            'c1a': c1a_path,
            'device': args.device,
        },
        'results': problems
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")

    # Upload to S3
    if not args.skip_s3_upload:
        try:
            s3_path = f's3://mycelium-data-v7/eval_results/eval_{int(time.time())}.json'
            subprocess.run(['aws', 's3', 'cp', args.output, s3_path], capture_output=True)
            print(f"Uploaded to {s3_path}")
        except Exception as e:
            print(f"Warning: S3 upload failed: {e}")


if __name__ == "__main__":
    main()
