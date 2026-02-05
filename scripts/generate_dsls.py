#!/usr/bin/env python3
"""
Generate SubGraphDSL for each atomic template using Qwen.

Takes atomic_templates.json (1K clustered templates) and generates a
SubGraphDSL computation graph for each one. The DSL defines what
arithmetic operation the template performs (SET, ADD, SUB, MUL, DIV).

Pipeline: vocab_reduce → atomic_split → cluster → fine-tune MiniLM → **generate DSLs**

USAGE:
    python scripts/generate_dsls.py --tp-size 4
    python scripts/generate_dsls.py --tp-size 4 --retry-fallbacks  # re-generate fallbacks
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_DIR = Path(__file__).parent.parent

DSL_PROMPT = """You are writing a SubGraphDSL for a math word problem pattern.

A SubGraphDSL defines the computation a span performs as JSON:
- "params": values extracted from the span text (numbers: n1, n2, ...)
- "inputs": values from upstream spans (use "upstream" for running entity value)
- "steps": ordered computation steps, each with "var", "op", "args"
- "output": which variable is exposed downstream

Allowed operators: SET (1 arg), ADD (2 args), SUB (2 args), MUL (2 args), DIV (2 args), NEG (1 arg)
Args can be variable names (strings) or literal numbers (floats).

Output ONLY valid JSON on a single line. No explanation.

Examples:

Pattern: "person has n items"
{{"params": {{"n1": "quantity"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "SET", "args": ["n1"]}}], "output": "out"}}

Pattern: "person gave n items to person"
{{"params": {{"n1": "quantity given"}}, "inputs": {{"upstream": "giver total"}}, "steps": [{{"var": "out", "op": "SUB", "args": ["upstream", "n1"]}}], "output": "out"}}

Pattern: "person buys n items at n each"
{{"params": {{"n1": "quantity", "n2": "price each"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "MUL", "args": ["n1", "n2"]}}], "output": "out"}}

Pattern: "person earns n percent more"
{{"params": {{"n1": "percentage"}}, "inputs": {{"upstream": "base amount"}}, "steps": [{{"var": "pct", "op": "DIV", "args": ["n1", 100]}}, {{"var": "bonus", "op": "MUL", "args": ["upstream", "pct"]}}, {{"var": "out", "op": "ADD", "args": ["upstream", "bonus"]}}], "output": "out"}}

Pattern: "n items split equally n"
{{"params": {{"n1": "total", "n2": "groups"}}, "inputs": {{}}, "steps": [{{"var": "out", "op": "DIV", "args": ["n1", "n2"]}}], "output": "out"}}

Pattern: "how many items does person have left"
{{"params": {{}}, "inputs": {{"upstream": "current total"}}, "steps": [{{"var": "out", "op": "SET", "args": ["upstream"]}}], "output": "out"}}

Pattern: "n more than person"
{{"params": {{"n1": "difference"}}, "inputs": {{"upstream": "base amount"}}, "steps": [{{"var": "out", "op": "ADD", "args": ["upstream", "n1"]}}], "output": "out"}}

Pattern: "n less than person"
{{"params": {{"n1": "difference"}}, "inputs": {{"upstream": "base amount"}}, "steps": [{{"var": "out", "op": "SUB", "args": ["upstream", "n1"]}}], "output": "out"}}

Pattern: "twice as many"
{{"params": {{}}, "inputs": {{"upstream": "base amount"}}, "steps": [{{"var": "out", "op": "MUL", "args": ["upstream", 2]}}], "output": "out"}}

Pattern: "per day"
{{"params": {{}}, "inputs": {{"upstream": "daily rate"}}, "steps": [{{"var": "out", "op": "SET", "args": ["upstream"]}}], "output": "out"}}

Pattern: "{pattern}"
Example spans from this cluster:
{examples}

Now write the SubGraphDSL JSON for the pattern "{pattern}":
"""


def load_qwen(tp_size: int = 4):
    """Load Qwen via vLLM."""
    from vllm import LLM
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    return llm


def batch_qwen(llm, prompts: List[str], max_tokens: int = 300) -> List[str]:
    """Run batch inference with JSON-friendly stop tokens."""
    from vllm import SamplingParams
    params = SamplingParams(
        temperature=0.1,
        max_tokens=max_tokens,
        # Allow multi-line JSON but stop at double newline or code block
        stop=["\n\n", "```", "Pattern:", "Example"],
    )
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text.strip() for o in outputs]


def parse_subgraph_json(response: str) -> Optional[Dict]:
    """Parse Qwen's JSON response with robust error recovery.

    Handles common Qwen 7B failure modes:
    - Preamble text before JSON
    - Extra closing braces (}}, after nested objects)
    - Missing closing braces
    """
    cleaned = response.strip()

    # Strategy 1: Find first { to last } and try json.loads
    brace_start = cleaned.find('{')
    if brace_start < 0:
        return None

    brace_end = cleaned.rfind('}')
    if brace_end <= brace_start:
        return None

    candidate = cleaned[brace_start:brace_end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Fix common "extra }}" issue
    # Qwen often writes {"params": {"n1": "val"}}, "inputs": ...
    # The }}, should be }, — remove double-close after nested objects
    fixed = re.sub(r'\}\}(\s*,\s*")', r'}\1', candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Balanced brace extraction (find the outermost matched pair)
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
                    return json.loads(cleaned[brace_start:i + 1])
                except json.JSONDecodeError:
                    break
    return None


def auto_repair_dsl(parsed: Dict) -> Dict:
    """Auto-repair common validation issues in parsed DSL.

    Fixes:
    - Steps referencing 'upstream' when inputs is empty
    - Steps referencing undefined params (auto-add as params)
    - Missing 'output' field
    - params/inputs as wrong types (string instead of dict)
    """
    params = parsed.get('params', {})
    inputs = parsed.get('inputs', {})
    steps = parsed.get('steps', [])

    # Fix non-dict params/inputs
    if not isinstance(params, dict):
        params = {}
    if not isinstance(inputs, dict):
        inputs = {}
    if not isinstance(steps, list):
        return parsed

    # Collect all variable references from step args
    defined = set(params.keys()) | set(inputs.keys())
    for step in steps:
        for arg in step.get('args', []):
            if isinstance(arg, str) and arg not in defined:
                if arg == 'upstream':
                    inputs['upstream'] = 'value from previous step'
                elif arg.startswith('n') and arg[1:].isdigit():
                    params[arg] = 'extracted value'
                else:
                    inputs[arg] = 'upstream value'
        defined.add(step.get('var', ''))

    parsed['params'] = params
    parsed['inputs'] = inputs

    # Ensure output is set
    if 'output' not in parsed and steps:
        parsed['output'] = steps[-1].get('var', 'out')

    return parsed


def validate_dsl(parsed: Dict, template_id: str, pattern: str) -> Optional[Dict]:
    """Validate parsed DSL against SubGraphDSL dataclass with auto-repair."""
    from mycelium.subgraph_dsl import SubGraphDSL

    # Inject template metadata
    parsed['template_id'] = template_id
    parsed['pattern'] = pattern

    # Ensure required fields exist
    if 'params' not in parsed:
        parsed['params'] = {}
    if 'inputs' not in parsed:
        parsed['inputs'] = {}
    if 'steps' not in parsed or not parsed['steps']:
        return None
    if 'output' not in parsed:
        return None

    # First try as-is
    try:
        dsl = SubGraphDSL.from_dict(parsed)
        errors = dsl.validate()
        if not errors:
            return dsl.to_dict()
    except Exception:
        pass

    # Auto-repair and retry
    parsed = auto_repair_dsl(parsed)
    try:
        dsl = SubGraphDSL.from_dict(parsed)
        errors = dsl.validate()
        if not errors:
            return dsl.to_dict()
    except Exception:
        pass

    return None


def classify_dsl(template_id: str, pattern: str) -> Dict:
    """Rule-based DSL classification for atomic templates.

    GSM8K has ~10 fundamental operation types. Classify by keyword patterns
    rather than relying on Qwen's unreliable JSON generation.
    """
    p = pattern.lower().strip()
    words = p.split()

    # Helper to build DSL dict
    def dsl(params, inputs, steps, output="out"):
        return {
            "template_id": template_id,
            "pattern": pattern,
            "params": params,
            "inputs": inputs,
            "steps": steps,
            "output": output,
        }

    # === QUERY patterns (pass through upstream) ===
    if any(q in p for q in ['how many', 'how much', 'what total', 'what is',
                             'what was', 'what does', 'how long', 'how far']):
        return dsl({}, {"upstream": "current total"},
                   [{"var": "out", "op": "SET", "args": ["upstream"]}])

    # === PERCENTAGE patterns ===
    if 'percent' in p:
        if any(w in p for w in ['more', 'increase', 'raise', 'markup']):
            # N percent more: upstream + upstream * n1/100
            return dsl({"n1": "percentage"}, {"upstream": "base amount"},
                       [{"var": "pct", "op": "DIV", "args": ["n1", 100]},
                        {"var": "bonus", "op": "MUL", "args": ["upstream", "pct"]},
                        {"var": "out", "op": "ADD", "args": ["upstream", "bonus"]}])
        if any(w in p for w in ['less', 'decrease', 'discount', 'off']):
            # N percent less: upstream - upstream * n1/100
            return dsl({"n1": "percentage"}, {"upstream": "base amount"},
                       [{"var": "pct", "op": "DIV", "args": ["n1", 100]},
                        {"var": "reduction", "op": "MUL", "args": ["upstream", "pct"]},
                        {"var": "out", "op": "SUB", "args": ["upstream", "reduction"]}])
        # N percent of: upstream * n1/100
        return dsl({"n1": "percentage"}, {"upstream": "base amount"},
                   [{"var": "pct", "op": "DIV", "args": ["n1", 100]},
                    {"var": "out", "op": "MUL", "args": ["upstream", "pct"]}])

    # === FRACTION patterns ===
    if 'half' in p or '1/2' in p:
        return dsl({}, {"upstream": "base amount"},
                   [{"var": "out", "op": "DIV", "args": ["upstream", 2]}])
    if 'third' in p or '1/3' in p:
        return dsl({}, {"upstream": "base amount"},
                   [{"var": "out", "op": "DIV", "args": ["upstream", 3]}])
    if 'quarter' in p or '1/4' in p:
        return dsl({}, {"upstream": "base amount"},
                   [{"var": "out", "op": "DIV", "args": ["upstream", 4]}])
    if 'double' in p or 'twice' in p:
        return dsl({}, {"upstream": "base amount"},
                   [{"var": "out", "op": "MUL", "args": ["upstream", 2]}])
    if 'triple' in p:
        return dsl({}, {"upstream": "base amount"},
                   [{"var": "out", "op": "MUL", "args": ["upstream", 3]}])

    # === MULTIPLICATION patterns ===
    if any(w in p for w in ['times as', 'n times', 'times more', 'times as many',
                             'times as much']):
        return dsl({"n1": "multiplier"}, {"upstream": "base amount"},
                   [{"var": "out", "op": "MUL", "args": ["upstream", "n1"]}])
    if 'at n each' in p or 'n each' in p or 'at n per' in p:
        return dsl({"n1": "quantity", "n2": "unit price"}, {},
                   [{"var": "out", "op": "MUL", "args": ["n1", "n2"]}])

    # === DIVISION patterns ===
    if any(w in p for w in ['split equally', 'divide equally', 'shared equally',
                             'divided by', 'split between', 'divided among',
                             'split among']):
        return dsl({"n1": "total", "n2": "groups"}, {},
                   [{"var": "out", "op": "DIV", "args": ["n1", "n2"]}])

    # === COMPARISON: MORE THAN ===
    if any(w in p for w in ['more than', 'extra', 'additional', 'added',
                             'adds n', 'gained', 'gains', 'received',
                             'receives', 'collects', 'collected']):
        return dsl({"n1": "amount added"}, {"upstream": "base amount"},
                   [{"var": "out", "op": "ADD", "args": ["upstream", "n1"]}])

    # === COMPARISON: LESS THAN ===
    if any(w in p for w in ['less than', 'fewer than', 'lost', 'loses',
                             'gave away', 'removed', 'took away', 'spent',
                             'used']):
        return dsl({"n1": "amount removed"}, {"upstream": "base amount"},
                   [{"var": "out", "op": "SUB", "args": ["upstream", "n1"]}])

    # === TRANSFER: GIVE/SELL (subtraction from giver) ===
    if any(w in p for w in ['gave n', 'gives n', 'sold n', 'sells n',
                             'donated n', 'sent n', 'lent n']):
        return dsl({"n1": "quantity transferred"}, {"upstream": "giver total"},
                   [{"var": "out", "op": "SUB", "args": ["upstream", "n1"]}])

    # === TRANSFER: BUY/GET (addition or multiplication) ===
    if any(w in p for w in ['buys n', 'bought n', 'gets n', 'got n',
                             'received n', 'found n', 'picked n', 'earned n']):
        if 'at n' in p or 'for n' in p:
            # buys N at N each → multiply
            return dsl({"n1": "quantity", "n2": "unit price"}, {},
                       [{"var": "out", "op": "MUL", "args": ["n1", "n2"]}])
        return dsl({"n1": "quantity"}, {},
                   [{"var": "out", "op": "SET", "args": ["n1"]}])

    # === RATE/UNIT markers (pass through) ===
    if any(w in p for w in ['per day', 'per hour', 'per week', 'per month',
                             'per year', 'per minute', 'each day', 'each hour',
                             'each week', 'every day', 'every hour', 'every week',
                             'per item', 'per unit']):
        return dsl({}, {"upstream": "rate value"},
                   [{"var": "out", "op": "SET", "args": ["upstream"]}])

    # === REMAINING/LEFT (pass through) ===
    if any(w in p for w in ['have left', 'has left', 'remaining', 'left over',
                             'rest of', 'the rest']):
        return dsl({}, {"upstream": "remaining amount"},
                   [{"var": "out", "op": "SET", "args": ["upstream"]}])

    # === LOCATION/CONTEXT markers (no computation, pass through) ===
    if any(w in p for w in ['at place', 'to place', 'from place', 'in place',
                             'at the', 'to the', 'from the', 'in the',
                             'on the', 'for the']):
        return dsl({}, {"upstream": "value"},
                   [{"var": "out", "op": "SET", "args": ["upstream"]}])

    # === CONDITION markers (pass through) ===
    if any(w in p for w in ['to person', 'for person', 'from person',
                             'to buy', 'to make', 'to get']):
        return dsl({}, {"upstream": "value"},
                   [{"var": "out", "op": "SET", "args": ["upstream"]}])

    # === HAS/STARTS WITH (initial assignment) ===
    if any(w in p for w in ['has n', 'had n', 'have n', 'starts with n',
                             'began with n', 'owns n', 'contains n']):
        return dsl({"n1": "quantity"}, {},
                   [{"var": "out", "op": "SET", "args": ["n1"]}])

    # === EARNS/MAKES (assignment with potential rate) ===
    if any(w in p for w in ['earns n', 'makes n', 'pays n', 'costs n',
                             'charges n', 'weighs n', 'measures n']):
        return dsl({"n1": "amount"}, {},
                   [{"var": "out", "op": "SET", "args": ["n1"]}])

    # === N items (just a quantity) ===
    if p.startswith('n ') or ' n ' in p:
        return dsl({"n1": "quantity"}, {},
                   [{"var": "out", "op": "SET", "args": ["n1"]}])

    # === Default: SET with n1 (assignment) ===
    return dsl({"n1": "value"}, {},
               [{"var": "out", "op": "SET", "args": ["n1"]}])


def classify_all_dsls(templates: List[Dict]) -> List[Dict]:
    """Assign DSLs to all templates using rule-based classification.

    No Qwen needed — keyword patterns reliably identify the ~10 operation
    types in GSM8K arithmetic (SET, ADD, SUB, MUL, DIV, and multi-step
    combos like percentage).
    """
    print(f"\n{'='*60}")
    print(f"DSL CLASSIFICATION (Rule-Based): {len(templates)} templates")
    print(f"{'='*60}")

    from mycelium.subgraph_dsl import SubGraphDSL

    valid_count = 0
    for i, tpl in enumerate(templates):
        dsl_dict = classify_dsl(tpl['template_id'], tpl['pattern'])

        # Validate against SubGraphDSL dataclass
        try:
            dsl = SubGraphDSL.from_dict(dsl_dict)
            errors = dsl.validate()
            if not errors:
                templates[i]['subgraph'] = dsl.to_dict()
                valid_count += 1
            else:
                print(f"  WARN: {tpl['pattern']} → {errors}")
                templates[i]['subgraph'] = dsl_dict
        except Exception as e:
            print(f"  ERROR: {tpl['pattern']} → {e}")
            templates[i]['subgraph'] = dsl_dict

    print(f"  Validated: {valid_count}/{len(templates)}")
    return templates


def print_stats(templates: List[Dict]):
    """Print DSL generation statistics."""
    print(f"\n{'='*60}")
    print("DSL GENERATION RESULTS")
    print(f"{'='*60}")

    total = len(templates)
    has_dsl = sum(1 for t in templates if t.get('subgraph'))

    # Count by structure
    def structure_key(tpl):
        sg = tpl.get('subgraph', {})
        steps = sg.get('steps', [])
        return tuple((s['op'], len(s['args'])) for s in steps)

    structure_counter = Counter(structure_key(t) for t in templates if t.get('subgraph'))

    # Count by primary operation
    op_counter = Counter()
    for t in templates:
        sg = t.get('subgraph', {})
        steps = sg.get('steps', [])
        if steps:
            # Primary op is the last step's op (the one producing output)
            op_counter[steps[-1]['op']] += 1

    # Count fallbacks (single SET with n1 param)
    fallback_count = 0
    for t in templates:
        sg = t.get('subgraph', {})
        steps = sg.get('steps', [])
        if (len(steps) == 1 and steps[0].get('op') == 'SET'
                and 'n1' in sg.get('params', {})):
            fallback_count += 1

    # Count multi-step DSLs
    multi_step = sum(1 for t in templates
                     if len(t.get('subgraph', {}).get('steps', [])) > 1)

    print(f"  Total templates: {total}")
    print(f"  With DSL: {has_dsl}")
    print(f"  Multi-step DSLs: {multi_step}")
    print(f"  Likely fallbacks: {fallback_count}")
    print(f"  Unique structures: {len(structure_counter)}")

    print(f"\n  Primary operation distribution:")
    for op, count in op_counter.most_common():
        print(f"    {op}: {count} ({count/total*100:.1f}%)")

    print(f"\n  Top 15 DSL structures:")
    for struct, count in structure_counter.most_common(15):
        ops_str = " -> ".join(f"{op}({n})" for op, n in struct)
        print(f"    [{count:4d}] {ops_str}")

    # Sample: show 10 templates with their DSLs
    print(f"\n  Sample templates with DSLs:")
    sorted_tpls = sorted(templates, key=lambda t: t.get('member_count', 0), reverse=True)
    for t in sorted_tpls[:10]:
        sg = t.get('subgraph', {})
        steps = sg.get('steps', [])
        steps_str = " -> ".join(f"{s['op']}({','.join(str(a) for a in s['args'])})" for s in steps)
        print(f"    [{t.get('member_count', 0):5d}] {t['pattern']}")
        print(f"           params={sg.get('params', {})} inputs={sg.get('inputs', {})}")
        print(f"           {steps_str} -> {sg.get('output', '?')}")


def main():
    parser = argparse.ArgumentParser(description="Generate SubGraphDSLs")
    parser.add_argument("--templates", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("SubGraphDSL Generation for Atomic Templates")
    print("=" * 60)

    # Load templates
    templates_path = Path(args.templates) if args.templates else OUTPUT_DIR / "atomic_templates.json"
    print(f"\nLoading templates from: {templates_path}")
    with open(templates_path) as f:
        templates = json.load(f)
    print(f"  Templates: {len(templates)}")

    # Classify DSLs (rule-based, no GPU needed)
    templates = classify_all_dsls(templates)

    # Stats
    print_stats(templates)

    # Save templates with DSLs
    output_path = OUTPUT_DIR / "atomic_templates_with_dsl.json"
    with open(output_path, "w") as f:
        json.dump(templates, f, indent=2)
    print(f"\n  Saved templates with DSLs to {output_path}")

    # Save standalone DSLs
    dsls_path = OUTPUT_DIR / "subgraph_dsls.json"
    dsls = [t['subgraph'] for t in templates if t.get('subgraph')]
    with open(dsls_path, "w") as f:
        json.dump(dsls, f, indent=2)
    print(f"  Standalone DSLs saved to {dsls_path} ({len(dsls)} DSLs)")


if __name__ == "__main__":
    main()
