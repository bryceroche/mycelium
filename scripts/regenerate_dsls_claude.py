#!/usr/bin/env python3
"""
Regenerate SubGraphDSLs using Claude for proper computation graphs.

The previous DSLs (from GPT-4o/Llama) had 71% SET operations - just value
assignment with no actual computation. This script uses Claude to analyze
each template's patterns and generate DSLs with correct operations.

Key insight: The DSL should capture WHAT COMPUTATION the span represents,
not just "there's a number here".

Example:
  Pattern: "3 sprints 3 times a week"
  Bad DSL:  SET(value) -> just assigns 3
  Good DSL: MUL(sprints, times_per_week) -> computes 3 * 3 = 9

Usage:
    export ANTHROPIC_API_KEY=...
    python scripts/regenerate_dsls_claude.py \
        --templates templates_with_graph_emb.json \
        --output templates_claude_dsls.json \
        --batch-size 20

"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mycelium.subgraph_dsl import SubGraphDSL, SubGraphStep

# ============================================================================
# Prompt Template
# ============================================================================

DSL_PROMPT = """You are generating a SubGraphDSL for a math word problem template.

## Your Task
Analyze the pattern examples and determine what COMPUTATION this template represents.

## DSL Specification
```python
@dataclass
class SubGraphDSL:
    template_id: str
    pattern: str  # Generic description
    params: Dict[str, str]  # Values extracted from span text
    inputs: Dict[str, str]  # Values from upstream spans (for chaining)
    steps: List[SubGraphStep]  # Computation steps
    output: str  # Variable exposed to downstream

class SubGraphStep:
    var: str  # Variable this step produces
    op: str   # Operator: SET, ADD, SUB, MUL, DIV, MOD, NEG
    args: List[str | float]  # References to params/inputs/step vars
```

## Key Rules
1. **Identify the COMPUTATION, not just values**
   - "3 apples at $2 each" → MUL(quantity, price_each), NOT SET
   - "She has 10 and gives away 3" → SUB(initial, given_away), NOT SET
   - "He earns $5 plus $3 tip" → ADD(base, tip), NOT SET

2. **params = values IN the span text** (numbers, quantities mentioned)
3. **inputs = values FROM previous spans** (references like "that", "the result", pronouns)
4. **Use inputs for chaining** when the span references something computed earlier

5. **Operators:**
   - SET: Just assign a value (ONLY when truly just stating a fact)
   - ADD: Combining quantities
   - SUB: Difference, remaining, taking away
   - MUL: Rate × time, quantity × price, repetition
   - DIV: Splitting, per-unit, fractions
   - NEG: Negation

## Template to Analyze
Template ID: {template_id}

Pattern Examples (what this template matches):
{examples}

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
```json
{{
  "template_id": "{template_id}",
  "pattern": "generic description of what this computes",
  "params": {{"var_name": "description", ...}},
  "inputs": {{"var_name": "description", ...}},
  "steps": [
    {{"var": "result_var", "op": "OPERATOR", "args": ["param1", "param2"]}}
  ],
  "output": "result_var"
}}
```

Think carefully: What mathematical operation does this pattern represent?
"""


def call_claude(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Claude API."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def generate_dsl_for_template(
    template: Dict[str, Any],
    model: str = "claude-sonnet-4-20250514"
) -> Optional[Dict[str, Any]]:
    """Generate a proper DSL for a template using Claude."""
    template_id = template.get("template_id", "unknown")

    # Get pattern examples
    examples = template.get("pattern_examples", []) or template.get("span_examples", [])
    if not examples:
        # Use the pattern field if no examples
        pattern = template.get("pattern", "")
        if pattern:
            examples = [pattern]

    if not examples:
        print(f"  {template_id}: No examples, skipping")
        return None

    # Format examples for prompt
    examples_text = "\n".join(f"- {ex}" for ex in examples[:5])

    prompt = DSL_PROMPT.format(
        template_id=template_id,
        examples=examples_text
    )

    try:
        response = call_claude(prompt, model)

        # Extract JSON from response (handle markdown code blocks)
        json_text = response.strip()
        if json_text.startswith("```"):
            # Remove markdown code block
            lines = json_text.split("\n")
            json_text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        dsl_dict = json.loads(json_text)

        # Validate basic structure
        required = ["template_id", "params", "steps", "output"]
        for key in required:
            if key not in dsl_dict:
                print(f"  {template_id}: Missing key '{key}'")
                return None

        # Convert steps format if needed
        steps = []
        for step in dsl_dict.get("steps", []):
            if isinstance(step, dict):
                steps.append({
                    "var": step.get("var", "result"),
                    "op": step.get("op", "SET"),
                    "args": step.get("args", [])
                })
        dsl_dict["steps"] = steps

        # Validate using SubGraphDSL
        try:
            dsl = SubGraphDSL(
                template_id=dsl_dict["template_id"],
                pattern=dsl_dict.get("pattern", ""),
                params=dsl_dict.get("params", {}),
                inputs=dsl_dict.get("inputs", {}),
                steps=[SubGraphStep.from_dict(s) for s in dsl_dict["steps"]],
                output=dsl_dict["output"]
            )
            errors = dsl.validate()
            if errors:
                print(f"  {template_id}: Validation errors: {errors[:2]}")
                return None
        except Exception as e:
            print(f"  {template_id}: DSL parse error: {e}")
            return None

        return dsl_dict

    except json.JSONDecodeError as e:
        print(f"  {template_id}: JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"  {template_id}: Error: {e}")
        return None


def process_batch(
    templates: List[Dict[str, Any]],
    batch_idx: int,
    model: str,
    delay: float = 0.5
) -> List[Dict[str, Any]]:
    """Process a batch of templates."""
    results = []

    for i, template in enumerate(templates):
        template_id = template.get("template_id", f"unknown_{i}")

        dsl = generate_dsl_for_template(template, model)

        if dsl:
            # Update template with new DSL
            template["subgraph"] = {
                "params": dsl.get("params", {}),
                "inputs": dsl.get("inputs", {}),
                "steps": dsl.get("steps", []),
                "output": dsl.get("output", "result")
            }
            template["pattern"] = dsl.get("pattern", template.get("pattern", ""))
            ops = tuple(s["op"] for s in dsl.get("steps", []))
            print(f"  ✓ {template_id}: {ops}")
        else:
            # Keep existing DSL
            print(f"  ✗ {template_id}: kept existing")

        results.append(template)

        # Rate limiting
        if delay > 0:
            time.sleep(delay)

    return results


def main():
    parser = argparse.ArgumentParser(description="Regenerate DSLs with Claude")
    parser.add_argument("--templates", default="templates_with_graph_emb.json",
                        help="Input templates file")
    parser.add_argument("--output", default="templates_claude_dsls.json",
                        help="Output file")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Claude model to use")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Templates per batch")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit total templates to process")
    parser.add_argument("--skip-set", action="store_true",
                        help="Only process templates that are currently SET-only")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between API calls (rate limiting)")
    args = parser.parse_args()

    print("=" * 60)
    print("Regenerating DSLs with Claude")
    print("=" * 60)

    # Load templates
    print(f"\n1. Loading templates from {args.templates}...")
    with open(args.templates) as f:
        templates = json.load(f)
    print(f"   Loaded {len(templates)} templates")

    # Filter if requested
    if args.skip_set:
        # Only process SET-only templates
        to_process = []
        to_keep = []
        for t in templates:
            sg = t.get("subgraph", {})
            steps = sg.get("steps", [])
            ops = tuple(s.get("op") for s in steps)
            if ops == ("SET",) or not ops:
                to_process.append(t)
            else:
                to_keep.append(t)
        print(f"   Filtering: {len(to_process)} SET-only, {len(to_keep)} to keep")
        templates_to_process = to_process
    else:
        templates_to_process = templates
        to_keep = []

    # Apply limit
    if args.limit:
        templates_to_process = templates_to_process[:args.limit]
        print(f"   Limited to {len(templates_to_process)} templates")

    # Process in batches
    print(f"\n2. Processing {len(templates_to_process)} templates...")
    print(f"   Model: {args.model}")
    print(f"   Batch size: {args.batch_size}")
    print("-" * 60)

    processed = []
    for batch_idx in range(0, len(templates_to_process), args.batch_size):
        batch = templates_to_process[batch_idx:batch_idx + args.batch_size]
        print(f"\nBatch {batch_idx // args.batch_size + 1}:")

        batch_results = process_batch(batch, batch_idx, args.model, args.delay)
        processed.extend(batch_results)

    # Combine with kept templates
    all_templates = processed + to_keep

    # Statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    from collections import Counter
    op_counts = Counter()
    for t in all_templates:
        sg = t.get("subgraph", {})
        steps = sg.get("steps", [])
        ops = tuple(s.get("op") for s in steps)
        op_counts[ops] += 1

    print("\nOperation distribution:")
    for ops, count in op_counts.most_common(20):
        print(f"  {ops}: {count}")

    # Save
    print(f"\n3. Saving to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(all_templates, f, indent=2)
    print(f"   Saved {len(all_templates)} templates")

    print("\nDone!")


if __name__ == "__main__":
    main()
