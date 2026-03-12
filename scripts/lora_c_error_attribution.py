#!/usr/bin/env python3
"""
LoRA C Error Attribution

Diagnose WHY operand generation fails:
1. Wrong selection - numbers ARE in problem text (sees them, picks wrong ones)
2. Hallucination - numbers AREN'T in problem text
3. Incomplete - got some operands right, missed others
4. Transcoding error - right numbers, wrong format

Usage:
    python scripts/lora_c_error_attribution.py --model models/lora_c_v2 --n 20
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "plan"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from oracle import execute_telegram, parse_telegram_expr


def extract_numbers_from_text(text: str) -> set:
    """Extract all numbers from text (integers, decimals, fractions)."""
    numbers = set()

    # Integers and decimals
    for match in re.findall(r'-?\d+\.?\d*', text):
        try:
            if '.' in match:
                numbers.add(float(match))
            else:
                numbers.add(int(match))
        except:
            pass

    # Fractions like 1/2, 3/4
    for match in re.findall(r'(\d+)/(\d+)', text):
        try:
            numbers.add(f"{match[0]}/{match[1]}")
        except:
            pass

    return numbers


def extract_variables_from_text(text: str) -> set:
    """Extract variable names from text."""
    # Common math variables
    vars_found = set()
    for var in re.findall(r'\b([a-zA-Z])\b', text):
        if var.lower() not in {'a', 'the', 'is', 'if', 'of', 'to', 'in', 'and', 'or'}:
            vars_found.add(var)
    return vars_found


def load_model(model_path: str, device: str):
    """Load LoRA C model."""
    base = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, model_path)
    model = model.to(device)
    model.eval()

    return model, tokenizer


def generate_operands(model, tokenizer, problem_text: str, step_type: str,
                      step_idx: int, total_steps: int, prev_results: list, device: str) -> str:
    """Generate operands using LoRA C."""
    prompt = f"Problem: {problem_text}\n"
    prompt += f"Step: {step_type} (step {step_idx + 1} of {total_steps})\n"

    if prev_results:
        prev_str = ", ".join(prev_results[-3:])
        prompt += f"Previous results: {prev_str}\n"

    prompt += "What values are needed for this step?\nOperands:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=newline_token
        )

    output = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
    return output.split("\n")[0].strip()


def categorize_error(problem_text: str, generated_operands: str, gold_answer: str) -> dict:
    """Categorize the error type."""
    problem_numbers = extract_numbers_from_text(problem_text)
    problem_vars = extract_variables_from_text(problem_text)

    generated_numbers = extract_numbers_from_text(generated_operands)
    generated_vars = extract_variables_from_text(generated_operands)

    gold_numbers = extract_numbers_from_text(str(gold_answer))

    result = {
        "problem_numbers": list(problem_numbers),
        "generated_numbers": list(generated_numbers),
        "generated_vars": list(generated_vars),
        "gold_numbers": list(gold_numbers),
        "category": None,
        "details": {}
    }

    # Check what was generated
    if not generated_numbers and not generated_vars:
        result["category"] = "empty"
        result["details"]["reason"] = "No operands generated"
        return result

    # Check if generated numbers are in problem
    numbers_in_problem = generated_numbers & problem_numbers
    numbers_hallucinated = generated_numbers - problem_numbers

    # Check if we got the gold answer numbers
    gold_in_generated = gold_numbers & generated_numbers

    result["details"]["numbers_in_problem"] = list(numbers_in_problem)
    result["details"]["numbers_hallucinated"] = list(numbers_hallucinated)
    result["details"]["gold_in_generated"] = list(gold_in_generated)

    # Categorize
    if len(numbers_hallucinated) > 0 and len(numbers_in_problem) == 0:
        result["category"] = "hallucination"
    elif len(numbers_hallucinated) > len(numbers_in_problem):
        result["category"] = "mostly_hallucination"
    elif len(numbers_in_problem) > 0 and len(gold_in_generated) == 0:
        result["category"] = "wrong_selection"
    elif len(gold_in_generated) > 0 and gold_in_generated != gold_numbers:
        result["category"] = "incomplete"
    elif generated_operands and not generated_numbers:
        result["category"] = "format_only"  # Only variables/symbols, no numbers
    else:
        result["category"] = "other"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/lora_c_v2", help="Path to LoRA C model")
    parser.add_argument("--problems", default="data/math_50_test.jsonl", help="Test problems")
    parser.add_argument("--n", type=int, default=20, help="Number of problems")
    parser.add_argument("--output", default="lora_c_attribution.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.model}...")
    model, tokenizer = load_model(args.model, device)

    # Load problems
    print(f"Loading problems from {args.problems}...")
    problems = []
    with open(args.problems) as f:
        content = f.read().strip()
        if content.startswith('[') or content.startswith('{'):
            # JSON format
            data = json.loads(content)
            if isinstance(data, dict):
                problems = list(data.values())
            else:
                problems = data
        else:
            # JSONL format
            for line in content.split('\n'):
                if line.strip():
                    problems.append(json.loads(line))

    # Normalize field names
    for p in problems:
        if 'problem' in p and 'text' not in p:
            p['text'] = p['problem']
        if 'answer' in p and 'gold_answer' not in p:
            p['gold_answer'] = str(p['answer'])

    problems = problems[:args.n]
    print(f"Evaluating {len(problems)} problems")

    # Run attribution
    results = []
    category_counts = defaultdict(int)

    step_types = ["SETUP", "COMPUTE", "SOLVE"]  # Same as eval

    for i, p in enumerate(problems):
        problem_text = p['text']
        gold_answer = p.get('gold_answer', '')

        problem_result = {
            "problem_id": i,
            "problem_text": problem_text[:200],
            "gold_answer": gold_answer,
            "steps": []
        }

        prev_results = []

        for step_idx, step_type in enumerate(step_types):
            # Generate operands
            operands = generate_operands(
                model, tokenizer, problem_text, step_type,
                step_idx, len(step_types), prev_results, device
            )

            # Categorize
            cat_result = categorize_error(problem_text, operands, gold_answer)
            cat_result["step_type"] = step_type
            cat_result["generated_operands"] = operands

            problem_result["steps"].append(cat_result)
            category_counts[cat_result["category"]] += 1

            prev_results.append(operands)

        results.append(problem_result)

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(problems)}")

    # Print summary
    print("\n" + "=" * 60)
    print("ERROR ATTRIBUTION SUMMARY")
    print("=" * 60)

    total = sum(category_counts.values())
    print(f"\nCategory breakdown (n={total} steps):")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {cat:20} {count:4} ({pct:5.1f}%)")

    # Show examples of each category
    print("\n" + "-" * 60)
    print("EXAMPLES BY CATEGORY")
    print("-" * 60)

    examples_shown = defaultdict(int)
    for r in results:
        for step in r["steps"]:
            cat = step["category"]
            if examples_shown[cat] < 2:
                print(f"\n[{cat.upper()}]")
                print(f"  Problem: {r['problem_text'][:100]}...")
                print(f"  Step: {step['step_type']}")
                print(f"  Generated: {step['generated_operands']}")
                print(f"  Problem numbers: {step['problem_numbers'][:5]}")
                print(f"  Generated numbers: {step['generated_numbers']}")
                if step['details'].get('numbers_hallucinated'):
                    print(f"  Hallucinated: {step['details']['numbers_hallucinated']}")
                examples_shown[cat] += 1

    # Save detailed results
    output = {
        "summary": dict(category_counts),
        "total_steps": total,
        "results": results
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
