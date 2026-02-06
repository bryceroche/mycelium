#!/usr/bin/env python3
"""Debug: test DSL generation on actual atomic templates."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mycelium.subgraph_dsl import SubGraphDSL
from vllm import LLM, SamplingParams

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

Pattern: "{pattern}"
Example spans from this cluster:
{examples}
"""


def parse_subgraph_json(response):
    cleaned = response.strip()
    brace_start = cleaned.find("{")
    if brace_start < 0:
        return None, "no_brace"
    brace_end = cleaned.rfind("}")
    if brace_end <= brace_start:
        return None, "no_close"
    candidate = cleaned[brace_start:brace_end + 1]
    try:
        return json.loads(candidate), "s1"
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r'\}\}(\s*,\s*")', r'}\1', candidate)
    try:
        return json.loads(fixed), "s2"
    except json.JSONDecodeError:
        pass
    return None, "all_failed"


# Load templates
with open(OUTPUT_DIR / "atomic_templates.json") as f:
    templates = json.load(f)

# Load Qwen
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=4,
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.85,
)

params = SamplingParams(
    temperature=0.1,
    max_tokens=300,
    stop=["\n\n", "```", "Pattern:", "Example"],
)

# Test first 20 templates
test_tpls = templates[:20]
prompts = []
for tpl in test_tpls:
    examples = tpl.get('span_examples', [])[:8]
    pats = tpl.get('all_patterns', [])[:5]
    examples_str = "\n".join(f'  "{ex}"' for ex in examples)
    if pats:
        examples_str += "\nSimilar patterns in this cluster:\n"
        examples_str += "\n".join(f'  "{p}"' for p in pats)
    prompts.append(DSL_PROMPT.format(
        pattern=tpl['pattern'],
        examples=examples_str or '  (no examples)',
    ))

outputs = llm.generate(prompts, params)

for tpl, out in zip(test_tpls, outputs):
    raw = out.outputs[0].text.strip()
    parsed, strategy = parse_subgraph_json(raw)

    print(f"\n{'='*60}")
    print(f"Pattern: {tpl['pattern']} (count={tpl['member_count']})")
    print(f"  Raw: {repr(raw[:400])}")
    print(f"  Parse: {strategy}")

    if parsed:
        parsed["template_id"] = tpl["template_id"]
        parsed["pattern"] = tpl["pattern"]
        if "params" not in parsed:
            parsed["params"] = {}
        if "inputs" not in parsed:
            parsed["inputs"] = {}
        print(f"  Keys: {list(parsed.keys())}")
        print(f"  Steps: {parsed.get('steps', 'MISSING')}")
        try:
            dsl = SubGraphDSL.from_dict(parsed)
            errors = dsl.validate()
            print(f"  Valid: {not errors}")
            if errors:
                print(f"  Errors: {errors}")
        except Exception as e:
            print(f"  from_dict error: {e}")
    else:
        print(f"  PARSE FAILED")
