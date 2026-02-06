#!/usr/bin/env python3
"""Debug: test DSL parsing and validation on sample responses."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mycelium.subgraph_dsl import SubGraphDSL


def parse_subgraph_json(response):
    cleaned = response.strip()
    brace_start = cleaned.find("{")
    if brace_start < 0:
        return None, "no opening brace"
    brace_end = cleaned.rfind("}")
    if brace_end <= brace_start:
        return None, "no closing brace"
    candidate = cleaned[brace_start:brace_end + 1]
    try:
        return json.loads(candidate), "strategy1_ok"
    except json.JSONDecodeError as e:
        pass
    fixed = re.sub(r'\}\}(\s*,\s*")', r'}\1', candidate)
    try:
        return json.loads(fixed), "strategy2_ok"
    except json.JSONDecodeError as e:
        pass
    # Strategy 3: balanced braces
    depth = 0
    in_string = False
    escape_next = False
    for i in range(brace_start, len(cleaned)):
        c = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\':
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[brace_start:i + 1]), "strategy3_ok"
                except json.JSONDecodeError:
                    return None, "strategy3_json_error"
    return None, "no_balanced_close"


# Test responses from actual Qwen output
tests = [
    ("person bought n items",
     '{"params": {"n1": "quantity bought"}, "inputs": {}, "steps": [{"var": "out", "op": "SET", "args": ["n1"]}], "output": "out"}'),
    ("person gave n items",
     '{"params": {"n1": "quantity given"}, "inputs": {"upstream": "giver total"}, "steps": [{"var": "out", "op": "SUB", "args": ["upstream", "n1"]}], "output": "out"}'),
    ("n times as many",
     '{"params": {"n1": "multiplier"}}, "inputs": {}, "steps": [{"var": "out", "op": "SET", "args": ["n1"]}], "output": "out"}'),
    ("how many items",
     'how many items are there\n{"params": {}, "inputs": {}, "steps": [{"var": "out", "op": "SET", "args": ["upstream"]}], "output": "out"}'),
]

for pat, raw in tests:
    print(f"\n{'='*60}")
    print(f"Pattern: {pat}")
    print(f"  Raw: {repr(raw[:300])}")
    parsed, strategy = parse_subgraph_json(raw)
    print(f"  Strategy: {strategy}")
    print(f"  Parsed keys: {list(parsed.keys()) if parsed else 'None'}")
    if parsed:
        print(f"  Parsed: {json.dumps(parsed, indent=2)[:300]}")
        parsed["template_id"] = "test"
        parsed["pattern"] = pat
        if "params" not in parsed:
            parsed["params"] = {}
        if "inputs" not in parsed:
            parsed["inputs"] = {}
        try:
            dsl = SubGraphDSL.from_dict(parsed)
            errors = dsl.validate()
            print(f"  Validation errors: {errors}")
            if not errors:
                print(f"  VALID DSL: {dsl}")
        except Exception as e:
            print(f"  from_dict error: {e}")
