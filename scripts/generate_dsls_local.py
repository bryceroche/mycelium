#!/usr/bin/env python3
"""Generate SubGraphDSLs for atomic templates using Claude Code.

This script processes templates in batches, generating DSL JSON
for each template pattern based on its span examples.

Since Claude Code IS the Claude generating these, this runs locally
and writes DSLs directly based on pattern analysis.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATA_DIR = Path(__file__).parent.parent / "data"


# Keyword → operation mapping for deterministic DSL generation
OPERATION_KEYWORDS = {
    # SET operations (initial values)
    "has": "SET",
    "have": "SET",
    "had": "SET",
    "starts with": "SET",
    "begins with": "SET",
    "there are": "SET",
    "costs": "SET",
    "weighs": "SET",
    "is": "SET",

    # ADD operations
    "more than": "ADD",
    "added": "ADD",
    "adds": "ADD",
    "additional": "ADD",
    "extra": "ADD",
    "gets": "ADD",
    "gets more": "ADD",
    "receives": "ADD",
    "gained": "ADD",
    "plus": "ADD",
    "increased": "ADD",
    "together": "ADD",
    "combined": "ADD",
    "total": "ADD",

    # SUB operations
    "less than": "SUB",
    "gave": "SUB",
    "gives": "SUB",
    "lost": "SUB",
    "loses": "SUB",
    "ate": "SUB",
    "eats": "SUB",
    "spent": "SUB",
    "spends": "SUB",
    "sold": "SUB",
    "sells": "SUB",
    "used": "SUB",
    "removed": "SUB",
    "took": "SUB",
    "takes": "SUB",
    "left": "SUB",
    "remaining": "SUB",
    "fewer": "SUB",
    "minus": "SUB",
    "decreased": "SUB",

    # MUL operations
    "times": "MUL",
    "each": "MUL",
    "per": "MUL",
    "twice": "MUL",
    "double": "MUL",
    "triple": "MUL",
    "groups of": "MUL",

    # DIV operations
    "half": "DIV",
    "half of": "DIV",
    "divided": "DIV",
    "split": "DIV",
    "shared equally": "DIV",
    "quarter": "DIV",
    "third": "DIV",
    "percent": "DIV",  # requires special handling

    # Query/pass-through
    "how many": "SET",
    "how much": "SET",
    "what": "SET",
}


def infer_dsl(pattern: str, examples: list) -> dict:
    """Infer a SubGraphDSL from pattern text and examples.

    Uses keyword matching and pattern analysis to determine
    the correct arithmetic operation.
    """
    p = pattern.lower().strip()

    # Detect operation from keywords (longest match first)
    sorted_keywords = sorted(OPERATION_KEYWORDS.keys(), key=len, reverse=True)
    detected_op = None
    for kw in sorted_keywords:
        if kw in p:
            detected_op = OPERATION_KEYWORDS[kw]
            break

    # Count 'n' occurrences to determine param count
    n_count = len(re.findall(r'\bn\b', p))

    # Special patterns
    has_upstream = any(kw in p for kw in [
        "more than", "less than", "gave", "gives", "lost", "loses",
        "ate", "eats", "spent", "spends", "sold", "sells",
        "took", "takes", "left", "remaining", "how many", "how much",
        "what", "gets", "receives", "added", "adds",
    ])

    is_rate = any(kw in p for kw in ["per", "each", "every"])
    is_fraction = any(kw in p for kw in ["half", "third", "quarter", "percent"])
    is_comparison = any(kw in p for kw in ["more than", "less than", "twice", "double", "triple"])
    is_query = any(kw in p for kw in ["how many", "how much", "what"])

    # Build DSL based on analysis
    if is_query:
        # Query: pass through upstream value
        return {
            "params": {},
            "inputs": {"upstream": "current total"},
            "steps": [{"var": "out", "op": "SET", "args": ["upstream"]}],
            "output": "out",
        }

    if is_fraction and "percent" in p:
        # N percent of something
        if has_upstream:
            return {
                "params": {"n1": "percentage"},
                "inputs": {"upstream": "base amount"},
                "steps": [
                    {"var": "pct", "op": "DIV", "args": ["n1", 100]},
                    {"var": "out", "op": "MUL", "args": ["upstream", "pct"]},
                ],
                "output": "out",
            }
        else:
            return {
                "params": {"n1": "percentage", "n2": "base"},
                "inputs": {},
                "steps": [
                    {"var": "pct", "op": "DIV", "args": ["n1", 100]},
                    {"var": "out", "op": "MUL", "args": ["n2", "pct"]},
                ],
                "output": "out",
            }

    if is_fraction and "half" in p:
        if has_upstream or n_count == 0:
            return {
                "params": {},
                "inputs": {"upstream": "value"},
                "steps": [{"var": "out", "op": "DIV", "args": ["upstream", 2]}],
                "output": "out",
            }
        else:
            return {
                "params": {"n1": "value"},
                "inputs": {},
                "steps": [{"var": "out", "op": "DIV", "args": ["n1", 2]}],
                "output": "out",
            }

    if is_fraction and "third" in p:
        if has_upstream or n_count == 0:
            return {
                "params": {},
                "inputs": {"upstream": "value"},
                "steps": [{"var": "out", "op": "DIV", "args": ["upstream", 3]}],
                "output": "out",
            }
        else:
            return {
                "params": {"n1": "value"},
                "inputs": {},
                "steps": [{"var": "out", "op": "DIV", "args": ["n1", 3]}],
                "output": "out",
            }

    if is_fraction and "quarter" in p:
        if has_upstream or n_count == 0:
            return {
                "params": {},
                "inputs": {"upstream": "value"},
                "steps": [{"var": "out", "op": "DIV", "args": ["upstream", 4]}],
                "output": "out",
            }
        else:
            return {
                "params": {"n1": "value"},
                "inputs": {},
                "steps": [{"var": "out", "op": "DIV", "args": ["n1", 4]}],
                "output": "out",
            }

    if is_comparison and "twice" in p:
        if has_upstream or n_count == 0:
            return {
                "params": {},
                "inputs": {"upstream": "base value"},
                "steps": [{"var": "out", "op": "MUL", "args": ["upstream", 2]}],
                "output": "out",
            }
        else:
            return {
                "params": {"n1": "value"},
                "inputs": {},
                "steps": [{"var": "out", "op": "MUL", "args": ["n1", 2]}],
                "output": "out",
            }

    if is_comparison and "double" in p:
        if has_upstream or n_count == 0:
            return {
                "params": {},
                "inputs": {"upstream": "base value"},
                "steps": [{"var": "out", "op": "MUL", "args": ["upstream", 2]}],
                "output": "out",
            }
        else:
            return {
                "params": {"n1": "value"},
                "inputs": {},
                "steps": [{"var": "out", "op": "MUL", "args": ["n1", 2]}],
                "output": "out",
            }

    if is_comparison and "triple" in p:
        if has_upstream or n_count == 0:
            return {
                "params": {},
                "inputs": {"upstream": "base value"},
                "steps": [{"var": "out", "op": "MUL", "args": ["upstream", 3]}],
                "output": "out",
            }
        else:
            return {
                "params": {"n1": "value"},
                "inputs": {},
                "steps": [{"var": "out", "op": "MUL", "args": ["n1", 3]}],
                "output": "out",
            }

    if is_rate and n_count >= 2:
        # "N items at N each" → MUL
        return {
            "params": {"n1": "quantity", "n2": "rate"},
            "inputs": {},
            "steps": [{"var": "out", "op": "MUL", "args": ["n1", "n2"]}],
            "output": "out",
        }

    if is_rate and n_count == 1:
        # "per day" with a single N → rate context
        return {
            "params": {"n1": "rate"},
            "inputs": {},
            "steps": [{"var": "out", "op": "SET", "args": ["n1"]}],
            "output": "out",
        }

    if is_rate and n_count == 0:
        # "per day" without numbers → rate modifier
        return {
            "params": {},
            "inputs": {"upstream": "value"},
            "steps": [{"var": "out", "op": "SET", "args": ["upstream"]}],
            "output": "out",
        }

    # Standard operations with detected op
    if detected_op and detected_op in ("ADD", "SUB"):
        if n_count >= 1 and has_upstream:
            return {
                "params": {"n1": "amount"},
                "inputs": {"upstream": "current value"},
                "steps": [{"var": "out", "op": detected_op, "args": ["upstream", "n1"]}],
                "output": "out",
            }
        elif n_count >= 2:
            return {
                "params": {"n1": "value1", "n2": "value2"},
                "inputs": {},
                "steps": [{"var": "out", "op": detected_op, "args": ["n1", "n2"]}],
                "output": "out",
            }
        elif n_count == 1:
            return {
                "params": {"n1": "amount"},
                "inputs": {"upstream": "current value"},
                "steps": [{"var": "out", "op": detected_op, "args": ["upstream", "n1"]}],
                "output": "out",
            }
        else:
            return {
                "params": {},
                "inputs": {"upstream": "current value"},
                "steps": [{"var": "out", "op": "SET", "args": ["upstream"]}],
                "output": "out",
            }

    if detected_op == "MUL":
        if n_count >= 2:
            return {
                "params": {"n1": "factor1", "n2": "factor2"},
                "inputs": {},
                "steps": [{"var": "out", "op": "MUL", "args": ["n1", "n2"]}],
                "output": "out",
            }
        elif n_count == 1:
            return {
                "params": {"n1": "factor"},
                "inputs": {"upstream": "base value"},
                "steps": [{"var": "out", "op": "MUL", "args": ["upstream", "n1"]}],
                "output": "out",
            }
        else:
            return {
                "params": {},
                "inputs": {"upstream": "base value"},
                "steps": [{"var": "out", "op": "SET", "args": ["upstream"]}],
                "output": "out",
            }

    if detected_op == "DIV":
        if n_count >= 2:
            return {
                "params": {"n1": "dividend", "n2": "divisor"},
                "inputs": {},
                "steps": [{"var": "out", "op": "DIV", "args": ["n1", "n2"]}],
                "output": "out",
            }
        elif n_count == 1:
            return {
                "params": {"n1": "divisor"},
                "inputs": {"upstream": "value"},
                "steps": [{"var": "out", "op": "DIV", "args": ["upstream", "n1"]}],
                "output": "out",
            }

    # Default: SET with available params
    if n_count >= 1:
        return {
            "params": {"n1": "value"},
            "inputs": {},
            "steps": [{"var": "out", "op": "SET", "args": ["n1"]}],
            "output": "out",
        }
    else:
        # No numbers, pass through
        return {
            "params": {},
            "inputs": {"upstream": "value"},
            "steps": [{"var": "out", "op": "SET", "args": ["upstream"]}],
            "output": "out",
        }


def validate_dsl(dsl: dict) -> list:
    """Validate a DSL dict. Returns list of errors."""
    errors = []

    allowed_ops = {"SET", "ADD", "SUB", "MUL", "DIV", "MOD", "NEG"}

    for step in dsl.get("steps", []):
        if step.get("op") not in allowed_ops:
            errors.append(f"Unknown op: {step.get('op')}")

    # Check variable resolution
    available = set(dsl.get("params", {}).keys()) | set(dsl.get("inputs", {}).keys())
    for step in dsl.get("steps", []):
        for arg in step.get("args", []):
            if isinstance(arg, str) and arg not in available:
                errors.append(f"Unresolved var: {arg}")
        available.add(step.get("var", ""))

    output = dsl.get("output", "")
    if output not in available:
        errors.append(f"Output var '{output}' not defined")

    return errors


def main():
    # Load templates
    input_path = DATA_DIR / "atomic_templates_v2.json"
    print(f"Loading from {input_path}...")
    with open(input_path) as f:
        templates = json.load(f)
    print(f"Loaded {len(templates)} templates")

    # Generate DSLs
    op_counts = Counter()
    valid_count = 0
    fallback_count = 0

    for t in templates:
        pattern = t["pattern"]
        examples = t.get("span_examples", [])

        dsl = infer_dsl(pattern, examples)
        dsl["template_id"] = t["template_id"]
        dsl["pattern"] = pattern

        errors = validate_dsl(dsl)
        if errors:
            # Fallback to SET
            dsl = {
                "template_id": t["template_id"],
                "pattern": pattern,
                "params": {"n1": "value"},
                "inputs": {},
                "steps": [{"var": "out", "op": "SET", "args": ["n1"]}],
                "output": "out",
            }
            fallback_count += 1
        else:
            valid_count += 1

        t["subgraph"] = dsl

        # Track ops
        ops = tuple(s["op"] for s in dsl["steps"])
        op_counts[ops] += 1

    # Stats
    print(f"\nDSL generation results:")
    print(f"  Valid: {valid_count}")
    print(f"  Fallback (SET): {fallback_count}")
    print(f"  Fallback rate: {fallback_count / len(templates) * 100:.1f}%")
    print(f"\n  Operation distribution:")
    for ops, count in op_counts.most_common(20):
        print(f"    {' → '.join(ops)}: {count}")

    # Save
    output_path = DATA_DIR / "atomic_templates_v2_with_dsl.json"
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Show some examples
    print(f"\nSample DSLs:")
    for t in templates[:10]:
        sg = t["subgraph"]
        ops_str = " → ".join(s["op"] for s in sg["steps"])
        print(f"  [{t['member_count']:5d}] {t['pattern'][:40]:40s} → {ops_str}")


if __name__ == "__main__":
    main()
