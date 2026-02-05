"""
Generate sub-graph DSLs for each deduplicated template using a Frontier LLM.

Reads templates (with patterns + raw examples) and asks the LLM to produce
a composable sub-graph DSL for each one.

Usage:
    python scripts/generate_subgraph_dsls.py \
        --input templates_1k_with_dsl.json \
        --output templates_with_subgraph_dsl.json \
        --batch-size 20

Requires OPENAI_API_KEY or GROQ_API_KEY in environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mycelium.subgraph_dsl import SubGraphDSL, SubGraphStep

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
7. For question spans ("how many..."), inputs should reference what's being asked about.

## Examples

Template: "[PERSON1] has [N] [ITEM1]"
Examples: ["Sally has 12 apples", "Tom has 5 oranges"]
```json
{"params": {"value": "the quantity"}, "inputs": {}, "steps": [{"var": "result", "op": "SET", "args": ["value"]}], "output": "result"}
```

Template: "insurance covers [N] percent of the [ITEM1]"
Examples: ["Insurance covers 80% of the cost"]
```json
{"params": {"percent": "percentage covered"}, "inputs": {"cost": "the cost being covered"}, "steps": [{"var": "rate", "op": "DIV", "args": ["percent", 100]}, {"var": "covered", "op": "MUL", "args": ["cost", "rate"]}], "output": "covered"}
```

Template: "[PERSON1] earns [N] per [TIME1] for [N] [TIME1]"
Examples: ["She earns $15 per hour for 8 hours"]
```json
{"params": {"rate": "earning rate", "periods": "number of periods"}, "inputs": {}, "steps": [{"var": "total", "op": "MUL", "args": ["rate", "periods"]}], "output": "total"}
```

Template: "[PERSON1]'s pay is docked [N] each time"
Examples: ["Her pay is docked $5 each time she's late"]
```json
{"params": {"dock_amount": "amount docked"}, "inputs": {"current_pay": "pay before docking"}, "steps": [{"var": "result", "op": "SUB", "args": ["current_pay", "dock_amount"]}], "output": "result"}
```

Template: "how many [ITEM1] do [PERSON1] and [PERSON2] have together"
Examples: ["How many apples do Tom and Sally have together?"]
```json
{"params": {}, "inputs": {"count_a": "first person's count", "count_b": "second person's count"}, "steps": [{"var": "total", "op": "ADD", "args": ["count_a", "count_b"]}], "output": "total"}
```

Now generate sub-graph DSLs for each template below. Return a JSON array of objects, one per template, with fields: template_id, params, inputs, steps, output.

TEMPLATES:
"""


def build_batch_prompt(templates: list[dict]) -> str:
    """Build the prompt for a batch of templates."""
    template_block = ""
    for t in templates:
        tid = t.get("template_id", "unknown")
        pattern = t.get("pattern", "")
        examples = t.get("raw_examples", t.get("span_examples", []))[:5]
        template_block += f"\nTemplate ID: {tid}\n"
        template_block += f"Pattern: {pattern}\n"
        template_block += f"Examples: {json.dumps(examples)}\n"

    return SUBGRAPH_PROMPT + template_block


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


def parse_response(response_text: str, template_ids: list[str]) -> list[dict]:
    """Parse LLM response into sub-graph DSL dicts."""
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re

        match = re.search(r"```json?\s*(.*?)```", response_text, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
        else:
            print(f"  Failed to parse response")
            return []

    # Handle both {"templates": [...]} and [...] formats
    if isinstance(data, dict):
        if "templates" in data:
            data = data["templates"]
        elif "results" in data:
            data = data["results"]
        else:
            # Single result wrapped in dict
            data = [data]

    return data


def validate_and_build(template: dict, dsl_data: dict) -> SubGraphDSL | None:
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
            print(f"  Validation errors for {dsl.template_id}: {errors}")
            return None
        return dsl
    except Exception as e:
        print(f"  Build error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate sub-graph DSLs for templates")
    parser.add_argument("--input", required=True, help="Input templates JSON file")
    parser.add_argument("--output", required=True, help="Output file with sub-graph DSLs")
    parser.add_argument("--batch-size", type=int, default=20, help="Templates per LLM call")
    parser.add_argument(
        "--provider",
        choices=["openai", "groq"],
        default="openai",
        help="LLM provider",
    )
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--start", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--limit", type=int, default=None, help="Max templates to process")
    args = parser.parse_args()

    # Load templates
    with open(args.input) as f:
        templates = json.load(f)

    if isinstance(templates, dict):
        templates = list(templates.values())

    print(f"Loaded {len(templates)} templates from {args.input}")

    # Slice
    templates = templates[args.start:]
    if args.limit:
        templates = templates[: args.limit]
    print(f"Processing {len(templates)} templates (start={args.start})")

    # Select provider
    if args.provider == "openai":
        call_fn = lambda p: call_openai(p, model=args.model or "gpt-4o")
    else:
        call_fn = lambda p: call_groq(p, model=args.model or "llama-3.3-70b-versatile")

    # Process in batches
    results = []
    failed = []

    for i in range(0, len(templates), args.batch_size):
        batch = templates[i : i + args.batch_size]
        batch_ids = [t.get("template_id", f"idx_{i + j}") for j, t in enumerate(batch)]

        print(f"\nBatch {i // args.batch_size + 1}: {batch_ids[0]} ... {batch_ids[-1]}")

        prompt = build_batch_prompt(batch)

        try:
            response = call_fn(prompt)
            dsl_datas = parse_response(response, batch_ids)

            # Match responses to templates
            for j, template in enumerate(batch):
                if j < len(dsl_datas):
                    dsl = validate_and_build(template, dsl_datas[j])
                    if dsl:
                        # Merge original template data with new DSL
                        merged = dict(template)
                        merged["subgraph"] = dsl.to_dict()
                        results.append(merged)
                        print(f"  {dsl.template_id}: {len(dsl.steps)} steps")
                    else:
                        failed.append(template.get("template_id", f"idx_{i + j}"))
                else:
                    print(f"  Missing response for {template.get('template_id')}")
                    failed.append(template.get("template_id", f"idx_{i + j}"))

        except Exception as e:
            print(f"  Batch failed: {e}")
            for t in batch:
                failed.append(t.get("template_id", "unknown"))

        # Rate limit
        time.sleep(1)

        # Save progress incrementally
        if (i // args.batch_size + 1) % 5 == 0:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved {len(results)} templates to {args.output}")

    # Final save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone: {len(results)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed template IDs: {failed[:20]}{'...' if len(failed) > 20 else ''}")


if __name__ == "__main__":
    main()
