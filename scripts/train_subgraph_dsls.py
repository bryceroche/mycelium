#!/usr/bin/env python3
"""
Train SubGraphDSLs for templates using LLM generation + GSM8K verification.

This script:
1. Loads templates sorted by priority (usage count)
2. Generates SubGraphDSLs using LLM (GPT-4o or Llama-3.3-70b)
3. Verifies against GSM8K ground truth answers
4. Regenerates failed templates with failure context

Usage:
    python scripts/train_subgraph_dsls.py \
        --templates deduplicated_templates.json \
        --output templates_trained.json \
        --verify-samples 5

Requires: OPENAI_API_KEY or GROQ_API_KEY in environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.subgraph_dsl import SubGraphDSL, SubGraphStep


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VerificationResult:
    """Result of verifying a template against a GSM8K problem."""
    template_id: str
    problem_idx: int
    problem_text: str
    expected: float
    predicted: Optional[float]
    correct: bool
    error: Optional[str] = None


@dataclass
class TemplateStats:
    """Stats for a template after verification."""
    template_id: str
    total_verified: int = 0
    correct: int = 0
    failures: List[VerificationResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total_verified == 0:
            return 0.0
        return self.correct / self.total_verified


# ============================================================================
# Prompt Templates
# ============================================================================

SUBGRAPH_PROMPT = """You are generating composable sub-graph DSLs for math word problem templates.

Each template represents a recurring pattern in math problems. Your job is to define the EXACT computation as a sub-graph that can compose with other sub-graphs in a DAG.

## DSL Spec

```json
{
  "params": {"var_name": "description"},
  "inputs": {"var_name": "description"},
  "steps": [{"var": "name", "op": "OP", "args": ["var_or_literal"]}],
  "output": "var_name"
}
```

**params**: Values extracted from the span text (numbers, quantities mentioned in THIS span).
**inputs**: Values that must come from a PREVIOUS span's output (references to earlier computations).
**steps**: Ordered computation. Each step produces a named variable.
**output**: The single variable exposed to downstream sub-graphs.

## Operators
SET(a) → a (pass-through)
ADD(a, b) → a + b
SUB(a, b) → a - b
MUL(a, b) → a * b
DIV(a, b) → a / b
MOD(a, b) → a % b
NEG(a) → -a

## Rules
1. Every template gets its OWN sub-graph — even if similar to others.
2. "params" are values IN the span text. "inputs" are values from PREVIOUS spans.
3. Steps execute top-to-bottom. Each step can reference params, inputs, or earlier step vars.
4. Literal numbers are allowed in args (e.g., 100 for percentage conversion).
5. Single output per sub-graph.
6. If the span is purely informational (no computation), use SET with the main value.

## Examples

Template: "[PERSON1] has [N] [ITEM1]"
```json
{"params": {"value": "the quantity"}, "inputs": {}, "steps": [{"var": "result", "op": "SET", "args": ["value"]}], "output": "result"}
```

Template: "[PERSON1] earns [N] per [TIME1] for [N] [TIME1]"
```json
{"params": {"rate": "earning rate", "periods": "number of periods"}, "inputs": {}, "steps": [{"var": "total", "op": "MUL", "args": ["rate", "periods"]}], "output": "total"}
```

Template: "insurance covers [N] percent of the [ITEM1]"
```json
{"params": {"percent": "percentage covered"}, "inputs": {"cost": "the cost being covered"}, "steps": [{"var": "rate", "op": "DIV", "args": ["percent", 100]}, {"var": "covered", "op": "MUL", "args": ["cost", "rate"]}], "output": "covered"}
```

Now generate sub-graph DSLs for each template below. Return a JSON array of objects, one per template, with fields: template_id, params, inputs, steps, output.

TEMPLATES:
"""

REGENERATION_PROMPT = """The previous SubGraphDSL for this template produced incorrect results.

Template: {pattern}
Template ID: {template_id}

## Previous Failures:
{failures}

Generate a CORRECTED SubGraphDSL. Consider:
- Are the params extracted correctly from the span?
- Are inputs from previous spans needed?
- Is the computation order correct?
- Are percentage/ratio conversions needed?

Return JSON with: params, inputs, steps, output
"""


# ============================================================================
# LLM Calls
# ============================================================================

def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """Call OpenAI API."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Call Groq API."""
    from groq import Groq
    client = Groq()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def call_llm(prompt: str, provider: str = "openai", model: str = None) -> str:
    """Call LLM with specified provider."""
    if provider == "openai":
        return call_openai(prompt, model or "gpt-4o")
    else:
        return call_groq(prompt, model or "llama-3.3-70b-versatile")


# ============================================================================
# Template Processing
# ============================================================================

def load_templates(path: str) -> List[Dict]:
    """Load and sort templates by priority."""
    with open(path) as f:
        templates = json.load(f)

    if isinstance(templates, dict):
        templates = list(templates.values())

    # Sort by priority: count * examples
    def priority(t):
        count = t.get("count", t.get("member_count", 1))
        examples = len(t.get("pattern_examples", t.get("span_examples", [])))
        return count * max(1, examples)

    templates.sort(key=priority, reverse=True)
    return templates


def build_batch_prompt(templates: List[Dict]) -> str:
    """Build generation prompt for a batch of templates."""
    template_block = ""
    for t in templates:
        tid = t.get("template_id", "unknown")
        pattern = t.get("pattern", "")
        examples = t.get("pattern_examples", t.get("span_examples", []))[:5]
        template_block += f"\nTemplate ID: {tid}\n"
        template_block += f"Pattern: {pattern}\n"
        template_block += f"Examples: {json.dumps(examples)}\n"

    return SUBGRAPH_PROMPT + template_block


def parse_response(response_text: str) -> List[Dict]:
    """Parse LLM response into sub-graph DSL dicts."""
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"```json?\s*(.*?)```", response_text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            return []

    # Handle various response formats
    if isinstance(data, dict):
        if "templates" in data:
            data = data["templates"]
        elif "results" in data:
            data = data["results"]
        else:
            data = [data]

    return data


def validate_and_build(template: Dict, dsl_data: Dict) -> Optional[SubGraphDSL]:
    """Build and validate a SubGraphDSL from LLM output."""
    try:
        dsl = SubGraphDSL(
            template_id=template.get("template_id", dsl_data.get("template_id", "")),
            pattern=template.get("pattern", ""),
            params=dsl_data.get("params", {}),
            inputs=dsl_data.get("inputs", {}),
            steps=[SubGraphStep.from_dict(s) for s in dsl_data.get("steps", [])],
            output=dsl_data.get("output", "result"),
        )
        errors = dsl.validate()
        if errors:
            print(f"    Validation errors: {errors}")
            return None
        return dsl
    except Exception as e:
        print(f"    Build error: {e}")
        return None


# ============================================================================
# GSM8K Verification
# ============================================================================

def extract_answer_from_solution(solution: str) -> float:
    """Extract numeric answer from GSM8K solution string."""
    # GSM8K answers are formatted as "#### <number>"
    match = re.search(r'####\s*([\d,]+\.?\d*)', solution)
    if match:
        return float(match.group(1).replace(',', ''))

    # Fallback: find last number
    numbers = re.findall(r'[\d,]+\.?\d*', solution)
    if numbers:
        return float(numbers[-1].replace(',', ''))
    return 0.0


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text."""
    # Match numbers with optional commas and decimals
    pattern = r'\$?([\d,]+\.?\d*)'
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m.replace(',', '')))
        except ValueError:
            pass
    return numbers


def verify_dsl_simple(dsl: SubGraphDSL, problem_text: str, expected: float) -> Tuple[Optional[float], bool, str]:
    """Simple verification: execute DSL with extracted numbers.

    Returns: (predicted, correct, error_msg)
    """
    try:
        # Extract numbers from problem text
        numbers = extract_numbers(problem_text)
        if not numbers:
            return None, False, "No numbers found"

        # Map numbers to params in order
        param_values = {}
        param_names = list(dsl.params.keys())
        for i, name in enumerate(param_names):
            if i < len(numbers):
                param_values[name] = numbers[i]
            else:
                param_values[name] = 0.0

        # Execute with no inputs (simple verification)
        predicted = dsl.execute(param_values, {})

        # Check with tolerance
        if abs(expected) < 0.01:
            correct = abs(predicted) < 0.01
        else:
            correct = abs(predicted - expected) / abs(expected) < 0.05

        return predicted, correct, ""

    except Exception as e:
        return None, False, str(e)


def verify_template(
    template: Dict,
    dsl: SubGraphDSL,
    gsm8k_problems: List[Dict],
    max_samples: int = 5
) -> TemplateStats:
    """Verify a template's DSL against GSM8K problems."""
    stats = TemplateStats(template_id=template.get("template_id", ""))
    pattern = template.get("pattern", "").lower()

    # Find problems that might match this pattern (simple keyword matching)
    pattern_words = set(re.findall(r'\w+', pattern))
    pattern_words -= {'n', 'person1', 'person2', 'item1', 'item2', 'time1'}

    verified = 0
    for i, problem in enumerate(gsm8k_problems):
        if verified >= max_samples:
            break

        question = problem.get("question", "")
        question_words = set(re.findall(r'\w+', question.lower()))

        # Check for word overlap (rough matching)
        overlap = len(pattern_words & question_words)
        if overlap < 2:
            continue

        expected = extract_answer_from_solution(problem.get("answer", ""))
        predicted, correct, error = verify_dsl_simple(dsl, question, expected)

        result = VerificationResult(
            template_id=stats.template_id,
            problem_idx=i,
            problem_text=question[:100] + "..." if len(question) > 100 else question,
            expected=expected,
            predicted=predicted,
            correct=correct,
            error=error if not correct else None,
        )

        stats.total_verified += 1
        if correct:
            stats.correct += 1
        else:
            stats.failures.append(result)

        verified += 1

    return stats


# ============================================================================
# Regeneration
# ============================================================================

def regenerate_dsl(
    template: Dict,
    failures: List[VerificationResult],
    provider: str = "openai",
    model: str = None
) -> Optional[SubGraphDSL]:
    """Regenerate a DSL with failure context."""
    pattern = template.get("pattern", "")
    template_id = template.get("template_id", "")

    # Format failures for prompt
    failure_text = ""
    for f in failures[:3]:
        failure_text += f"- Problem: \"{f.problem_text}\"\n"
        failure_text += f"  Expected: {f.expected}, Got: {f.predicted}\n"
        if f.error:
            failure_text += f"  Error: {f.error}\n"

    prompt = REGENERATION_PROMPT.format(
        pattern=pattern,
        template_id=template_id,
        failures=failure_text
    )

    try:
        response = call_llm(prompt, provider, model)
        dsl_data = parse_response(response)
        if dsl_data:
            return validate_and_build(template, dsl_data[0] if isinstance(dsl_data, list) else dsl_data)
    except Exception as e:
        print(f"    Regeneration failed: {e}")

    return None


# ============================================================================
# Parallel Batch Processing
# ============================================================================

def process_single_batch(
    batch: List[Dict],
    batch_idx: int,
    provider: str,
    model: str,
    gsm8k_problems: List[Dict],
    verify_samples: int,
    max_regenerations: int
) -> Tuple[List[Dict], List[str]]:
    """Process a single batch of templates. Returns (results, failed_ids)."""
    results = []
    failed = []

    prompt = build_batch_prompt(batch)
    try:
        response = call_llm(prompt, provider, model)
        dsl_datas = parse_response(response)
    except Exception as e:
        # Batch failed
        return [], [t.get("template_id", "unknown") for t in batch]

    for j, template in enumerate(batch):
        tid = template.get("template_id", f"idx_{batch_idx + j}")

        if j >= len(dsl_datas):
            failed.append(tid)
            continue

        dsl = validate_and_build(template, dsl_datas[j])
        if not dsl:
            failed.append(tid)
            continue

        # Verify (skip if no GSM8K problems)
        if gsm8k_problems:
            stats = verify_template(template, dsl, gsm8k_problems, verify_samples)

            # Regenerate if needed
            regenerations = 0
            while stats.accuracy < 0.5 and regenerations < max_regenerations and stats.failures:
                new_dsl = regenerate_dsl(template, stats.failures, provider, model)
                if new_dsl:
                    dsl = new_dsl
                    stats = verify_template(template, dsl, gsm8k_problems, verify_samples)
                regenerations += 1
                time.sleep(0.3)
        else:
            stats = TemplateStats(template_id=tid)
            regenerations = 0

        # Save result
        merged = dict(template)
        merged["subgraph"] = dsl.to_dict()
        merged["train_stats"] = {
            "verified": stats.total_verified,
            "correct": stats.correct,
            "accuracy": stats.accuracy,
            "regenerations": regenerations
        }
        results.append(merged)

    return results, failed


# ============================================================================
# Main Training Loop
# ============================================================================

def train_subgraph_dsls(
    templates: List[Dict],
    gsm8k_problems: List[Dict],
    provider: str = "openai",
    model: str = None,
    batch_size: int = 20,
    verify_samples: int = 5,
    max_regenerations: int = 2,
    output_path: str = "templates_trained.json"
) -> List[Dict]:
    """Main training loop."""

    results = []
    failed_templates = []

    total_batches = (len(templates) + batch_size - 1) // batch_size
    print(f"\nTraining {len(templates)} templates in {total_batches} batches")
    print(f"Verification: {verify_samples} samples per template")
    print(f"Max regenerations: {max_regenerations}")
    print("=" * 60)

    for batch_idx in range(0, len(templates), batch_size):
        batch = templates[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        print(f"\n[Batch {batch_num}/{total_batches}]")

        # Generate DSLs for batch
        prompt = build_batch_prompt(batch)
        try:
            response = call_llm(prompt, provider, model)
            dsl_datas = parse_response(response)
        except Exception as e:
            print(f"  Batch generation failed: {e}")
            for t in batch:
                failed_templates.append(t.get("template_id", "unknown"))
            continue

        # Process each template in batch
        for j, template in enumerate(batch):
            tid = template.get("template_id", f"idx_{batch_idx + j}")

            if j >= len(dsl_datas):
                print(f"  {tid}: No response")
                failed_templates.append(tid)
                continue

            dsl = validate_and_build(template, dsl_datas[j])
            if not dsl:
                print(f"  {tid}: Invalid DSL")
                failed_templates.append(tid)
                continue

            # Verify against GSM8K
            stats = verify_template(template, dsl, gsm8k_problems, verify_samples)

            # Regenerate if accuracy too low
            regenerations = 0
            while stats.accuracy < 0.5 and regenerations < max_regenerations and stats.failures:
                print(f"  {tid}: {stats.accuracy:.0%} accuracy, regenerating...")
                new_dsl = regenerate_dsl(template, stats.failures, provider, model)
                if new_dsl:
                    dsl = new_dsl
                    stats = verify_template(template, dsl, gsm8k_problems, verify_samples)
                regenerations += 1
                time.sleep(0.5)  # Rate limit

            # Report final status
            status = "OK" if stats.accuracy >= 0.5 else "LOW"
            print(f"  {tid}: {stats.correct}/{stats.total_verified} ({stats.accuracy:.0%}) [{status}]")

            # Save result
            merged = dict(template)
            merged["subgraph"] = dsl.to_dict()
            merged["train_stats"] = {
                "verified": stats.total_verified,
                "correct": stats.correct,
                "accuracy": stats.accuracy,
                "regenerations": regenerations
            }
            results.append(merged)

        # Rate limit between batches
        time.sleep(1)

        # Save progress
        if batch_num % 3 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [Saved {len(results)} templates]")

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete: {len(results)} templates saved")
    print(f"Failed templates: {len(failed_templates)}")

    # Summary stats
    if results:
        avg_accuracy = sum(r.get("train_stats", {}).get("accuracy", 0) for r in results) / len(results)
        print(f"Average accuracy: {avg_accuracy:.1%}")

    return results


def train_parallel(
    templates: List[Dict],
    gsm8k_problems: List[Dict],
    provider: str = "openai",
    model: str = None,
    batch_size: int = 20,
    verify_samples: int = 5,
    max_regenerations: int = 2,
    output_path: str = "templates_trained.json",
    num_workers: int = 4
) -> List[Dict]:
    """Parallel training using ThreadPoolExecutor."""

    # Split into batches
    batches = []
    for i in range(0, len(templates), batch_size):
        batches.append((i, templates[i:i + batch_size]))

    total_batches = len(batches)
    print(f"\nParallel training: {len(templates)} templates in {total_batches} batches")
    print(f"Workers: {num_workers}, Batch size: {batch_size}")
    print("=" * 60)

    all_results = []
    all_failed = []
    completed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for batch_idx, batch in batches:
            future = executor.submit(
                process_single_batch,
                batch, batch_idx, provider, model,
                gsm8k_problems, verify_samples, max_regenerations
            )
            futures[future] = batch_idx

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                results, failed = future.result()
                all_results.extend(results)
                all_failed.extend(failed)
                completed += 1
                print(f"  [{completed}/{total_batches}] Batch {batch_idx // batch_size + 1}: {len(results)} OK, {len(failed)} failed")

                # Save periodically
                if completed % 5 == 0:
                    with open(output_path, "w") as f:
                        json.dump(all_results, f, indent=2)
            except Exception as e:
                print(f"  Batch {batch_idx} error: {e}")
                completed += 1

    # Final save
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete: {len(all_results)} templates saved")
    print(f"Failed: {len(all_failed)}")
    if all_results:
        avg_acc = sum(r.get("train_stats", {}).get("accuracy", 0) for r in all_results) / len(all_results)
        print(f"Average accuracy: {avg_acc:.1%}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train SubGraphDSLs with GSM8K verification")
    parser.add_argument("--templates", required=True, help="Input templates JSON")
    parser.add_argument("--output", default="templates_trained.json", help="Output file")
    parser.add_argument("--batch-size", type=int, default=20, help="Templates per LLM call")
    parser.add_argument("--verify-samples", type=int, default=5, help="GSM8K samples per template")
    parser.add_argument("--max-regenerations", type=int, default=2, help="Max regen attempts")
    parser.add_argument("--provider", choices=["openai", "groq"], default="openai")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--limit", type=int, default=None, help="Limit templates to process")
    parser.add_argument("--gsm8k-split", default="test", help="GSM8K split to use")
    parser.add_argument("--parallel", type=int, default=0, help="Number of parallel workers (0=sequential)")
    args = parser.parse_args()

    # Load templates
    print("Loading templates...")
    templates = load_templates(args.templates)
    if args.limit:
        templates = templates[:args.limit]
    print(f"  {len(templates)} templates (sorted by priority)")

    # Load GSM8K
    print("\nLoading GSM8K dataset...")
    try:
        from datasets import load_dataset
        gsm8k = load_dataset("openai/gsm8k", "main", split=args.gsm8k_split)
        gsm8k_problems = [{"question": ex["question"], "answer": ex["answer"]} for ex in gsm8k]
        print(f"  {len(gsm8k_problems)} problems loaded")
    except Exception as e:
        print(f"  Warning: Could not load GSM8K: {e}")
        print("  Running without verification")
        gsm8k_problems = []

    # Run training
    if args.parallel > 0:
        train_parallel(
            templates=templates,
            gsm8k_problems=gsm8k_problems,
            provider=args.provider,
            model=args.model,
            batch_size=args.batch_size,
            verify_samples=args.verify_samples,
            max_regenerations=args.max_regenerations,
            output_path=args.output,
            num_workers=args.parallel,
        )
    else:
        train_subgraph_dsls(
            templates=templates,
            gsm8k_problems=gsm8k_problems,
            provider=args.provider,
            model=args.model,
            batch_size=args.batch_size,
            verify_samples=args.verify_samples,
            max_regenerations=args.max_regenerations,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
