#!/usr/bin/env python3
"""
Fix DSLs inline using pattern matching.

Detects patterns that indicate computation:
- "each" → MUL (quantity × per_item)
- "per hour/minute/day" + time → MUL (rate × time)
- "twice/triple/half" → MUL with multiplier
- "more than" → ADD
- "less than" / "fewer" → SUB
- References to previous values → needs inputs

This doesn't require an API - uses heuristics to upgrade SET to proper ops.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mycelium.subgraph_dsl import SubGraphDSL, SubGraphStep


def analyze_pattern(text: str) -> Tuple[str, Dict[str, str], Dict[str, str], List[Dict]]:
    """Analyze text pattern and return (op, params, inputs, steps).

    Returns the best operation type based on linguistic patterns.
    """
    text_lower = text.lower()

    # Extract numbers for param naming
    numbers = re.findall(r'\d+(?:\.\d+)?', text)

    # Default params
    params = {"value": "quantity from text"}
    inputs = {}

    # Pattern detection - from most specific to least

    # MUL patterns
    mul_patterns = [
        (r'\b(\d+)\s+(?:of\s+)?(?:the\s+)?(\w+)\s+(?:each|per)\b', 'quantity × per_item'),
        (r'\beach\b.*?\b(\d+)\b', 'each implies multiplication'),
        (r'\b(\d+)\s+(?:per|a)\s+(?:hour|minute|day|week|month|year)\b', 'rate pattern'),
        (r'\btwice\s+(?:as\s+)?(?:many|much|frequent)', 'double'),
        (r'\btriple\b|\bthree\s+times\b', 'triple'),
        (r'\bhalf\b(?:\s+(?:as\s+)?(?:many|much))?', 'half (DIV by 2)'),
        (r'\b(\d+)\s+times\b', 'multiplier'),
        (r'\b(\d+)\s*x\s*(\d+)\b', 'explicit multiplication'),
        (r'\b(\d+)\s+(?:rows?|columns?|groups?)\s+(?:of|with)\s+(\d+)', 'grid pattern'),
    ]

    for pattern, reason in mul_patterns:
        if re.search(pattern, text_lower):
            if 'half' in text_lower:
                params = {"value": "the quantity to halve"}
                inputs = {}
                steps = [{"var": "result", "op": "DIV", "args": ["value", 2]}]
                return "DIV", params, inputs, steps

            if 'twice' in text_lower or 'double' in text_lower:
                # Check if it references something upstream
                if re.search(r'(?:as|than)\s+(?:the\s+)?(?:first|other|previous)', text_lower):
                    params = {}
                    inputs = {"upstream": "value from previous span"}
                    steps = [{"var": "result", "op": "MUL", "args": ["upstream", 2]}]
                    return "MUL", params, inputs, steps
                else:
                    params = {"value": "the quantity to double"}
                    steps = [{"var": "result", "op": "MUL", "args": ["value", 2]}]
                    return "MUL", params, inputs, steps

            if 'triple' in text_lower or 'three times' in text_lower:
                params = {"value": "the quantity to triple"}
                steps = [{"var": "result", "op": "MUL", "args": ["value", 3]}]
                return "MUL", params, inputs, steps

            # Generic MUL with two quantities
            if len(numbers) >= 2:
                params = {"quantity": "first quantity", "multiplier": "second quantity"}
                steps = [{"var": "result", "op": "MUL", "args": ["quantity", "multiplier"]}]
                return "MUL", params, inputs, steps
            elif 'each' in text_lower or 'per' in text_lower:
                params = {"count": "number of items", "per_item": "amount per item"}
                steps = [{"var": "result", "op": "MUL", "args": ["count", "per_item"]}]
                return "MUL", params, inputs, steps

    # SUB patterns
    sub_patterns = [
        (r'\b(?:less|fewer)\s+than\b', 'less than'),
        (r'\bgave\s+away\b|\bgives\s+away\b', 'gave away'),
        (r'\b(?:ate|eats|eaten)\b', 'consumption'),
        (r'\b(?:sold|sells)\b', 'sold'),
        (r'\b(?:spent|spends)\b', 'spending'),
        (r'\b(?:lost|loses)\b', 'lost'),
        (r'\b(?:used|uses)\b', 'used'),
        (r'\brotten\b|\bspoiled\b|\bbad\b', 'waste'),
        (r'\bleft\b|\bremaining\b', 'remainder (needs SUB)'),
        (r'\btook\b|\btakes\b', 'took away'),
    ]

    for pattern, reason in sub_patterns:
        if re.search(pattern, text_lower):
            # Check if it references upstream
            if re.search(r'(?:of\s+)?(?:the|her|his|their|its)\s+\w+', text_lower):
                params = {"amount": "amount to subtract"}
                inputs = {"total": "total from previous span"}
                steps = [{"var": "result", "op": "SUB", "args": ["total", "amount"]}]
                return "SUB", params, inputs, steps
            elif len(numbers) >= 2:
                params = {"total": "starting amount", "amount": "amount to subtract"}
                steps = [{"var": "result", "op": "SUB", "args": ["total", "amount"]}]
                return "SUB", params, inputs, steps

    # ADD patterns
    add_patterns = [
        (r'\bmore\s+than\b', 'more than'),
        (r'\bplus\b|\band\b.*\btotal\b', 'addition'),
        (r'\b(?:added|adds)\b', 'added'),
        (r'\b(?:combined|altogether|total)\b', 'combination'),
        (r'\b(?:received|receives|got|gets)\b', 'received'),
        (r'\b(?:earned|earns)\b', 'earned'),
        (r'\b(?:bought|buys)\b.*\bmore\b', 'bought more'),
    ]

    for pattern, reason in add_patterns:
        if re.search(pattern, text_lower):
            if len(numbers) >= 2:
                params = {"amount1": "first amount", "amount2": "second amount"}
                steps = [{"var": "result", "op": "ADD", "args": ["amount1", "amount2"]}]
                return "ADD", params, inputs, steps
            elif re.search(r'(?:more|additional)\s+(?:than)?', text_lower):
                params = {"additional": "amount to add"}
                inputs = {"base": "base amount from upstream"}
                steps = [{"var": "result", "op": "ADD", "args": ["base", "additional"]}]
                return "ADD", params, inputs, steps

    # DIV patterns
    div_patterns = [
        (r'\bdivided?\b', 'division'),
        (r'\bsplit\b', 'split'),
        (r'\bshared?\s+(?:equally|evenly)\b', 'shared equally'),
        (r'\bper\s+(?:person|item|piece|unit)\b', 'per unit'),
        (r'\b(?:one|1)\s*(?:/|÷)\s*(\d+)', 'fraction'),
        (r'\b(?:quarter|fourth|third|fifth)\b', 'fraction'),
    ]

    for pattern, reason in div_patterns:
        if re.search(pattern, text_lower):
            if len(numbers) >= 2:
                params = {"total": "total to divide", "divisor": "number to divide by"}
                steps = [{"var": "result", "op": "DIV", "args": ["total", "divisor"]}]
                return "DIV", params, inputs, steps
            elif 'half' in text_lower:
                params = {"value": "quantity to halve"}
                steps = [{"var": "result", "op": "DIV", "args": ["value", 2]}]
                return "DIV", params, inputs, steps

    # Default to SET if no computation pattern detected
    params = {"value": "the quantity"}
    steps = [{"var": "result", "op": "SET", "args": ["value"]}]
    return "SET", params, inputs, steps


def fix_template(template: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Fix a template's DSL based on pattern analysis.

    Only modifies SET-only templates. Preserves existing multi-step DSLs.

    Returns (updated_template, change_description)
    """
    # Check current operation - only fix SET-only templates
    sg = template.get("subgraph", {})
    current_steps = sg.get("steps", [])
    current_ops = tuple(s.get("op") for s in current_steps)

    # Skip if not SET-only (preserve good multi-step DSLs)
    if current_ops and current_ops != ("SET",):
        return template, "preserved (multi-step)"

    examples = template.get("pattern_examples", []) or template.get("span_examples", [])

    if not examples:
        return template, "no examples"

    # Analyze all examples and vote on the best operation
    op_votes = Counter()
    all_analyses = []

    for ex in examples[:5]:
        op, params, inputs, steps = analyze_pattern(ex)
        op_votes[op] += 1
        all_analyses.append((op, params, inputs, steps, ex))

    # Use the most common operation (but prefer non-SET if close)
    best_op = op_votes.most_common(1)[0][0]

    # If it's still SET but there were some non-SET votes, investigate further
    if best_op == "SET" and len(op_votes) > 1:
        non_set_votes = sum(c for op, c in op_votes.items() if op != "SET")
        if non_set_votes >= len(examples) * 0.3:  # 30% threshold
            # Use the most common non-SET operation
            for op, _ in op_votes.most_common():
                if op != "SET":
                    best_op = op
                    break

    # Get the analysis for the best operation
    for op, params, inputs, steps, ex in all_analyses:
        if op == best_op:
            best_params = params
            best_inputs = inputs
            best_steps = steps
            break
    else:
        # Fallback
        best_params = {"value": "the quantity"}
        best_inputs = {}
        best_steps = [{"var": "result", "op": "SET", "args": ["value"]}]

    # Update template
    template["subgraph"] = {
        "params": best_params,
        "inputs": best_inputs,
        "steps": best_steps,
        "output": "result"
    }

    old_ops = None
    if "subgraph" in template:
        old_steps = template.get("subgraph", {}).get("steps", [])
        if old_steps:
            old_ops = tuple(s.get("op") for s in old_steps)

    new_ops = tuple(s["op"] for s in best_steps)

    if old_ops != new_ops:
        return template, f"{old_ops} -> {new_ops}"
    return template, "unchanged"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fix DSLs using pattern matching")
    parser.add_argument("--templates", default="templates_with_graph_emb.json")
    parser.add_argument("--output", default="templates_fixed_dsls.json")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just analyze")
    args = parser.parse_args()

    print("=" * 60)
    print("Fixing DSLs with Pattern Matching")
    print("=" * 60)

    # Load templates
    print(f"\n1. Loading templates from {args.templates}...")
    with open(args.templates) as f:
        templates = json.load(f)
    print(f"   Loaded {len(templates)} templates")

    # Count before
    before_ops = Counter()
    for t in templates:
        sg = t.get("subgraph", {})
        steps = sg.get("steps", [])
        ops = tuple(s.get("op") for s in steps)
        before_ops[ops] += 1

    print("\nBefore fixing:")
    for ops, count in before_ops.most_common(10):
        print(f"  {ops}: {count}")

    # Fix templates
    print(f"\n2. Analyzing and fixing templates...")

    changes = []
    for i, t in enumerate(templates):
        old_ops = tuple(s.get("op") for s in t.get("subgraph", {}).get("steps", []))
        template, change = fix_template(t)
        templates[i] = template  # Update in place
        new_ops = tuple(s.get("op") for s in template.get("subgraph", {}).get("steps", []))
        if old_ops != new_ops:
            changes.append((t.get("template_id"), f"{old_ops} -> {new_ops}"))

    print(f"   Changed {len(changes)} templates")

    if args.dry_run:
        print("\n   (Dry run - showing first 20 changes)")
        for tid, change in changes[:20]:
            print(f"     {tid}: {change}")

    # Count after
    after_ops = Counter()
    for t in templates:
        sg = t.get("subgraph", {})
        steps = sg.get("steps", [])
        ops = tuple(s.get("op") for s in steps)
        after_ops[ops] += 1

    print("\nAfter fixing:")
    for ops, count in after_ops.most_common(10):
        print(f"  {ops}: {count}")

    # Show delta
    print("\nChanges:")
    all_ops = set(before_ops.keys()) | set(after_ops.keys())
    for ops in sorted(all_ops, key=lambda x: -after_ops.get(x, 0)):
        before = before_ops.get(ops, 0)
        after = after_ops.get(ops, 0)
        if before != after:
            delta = after - before
            sign = "+" if delta > 0 else ""
            print(f"  {ops}: {before} -> {after} ({sign}{delta})")

    # Save
    if not args.dry_run:
        print(f"\n3. Saving to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(templates, f, indent=2)
        print(f"   Saved {len(templates)} templates")

    print("\nDone!")


if __name__ == "__main__":
    main()
