#!/usr/bin/env python3
"""Debug: test the generalization prompt on a few spans and print raw Qwen output."""

import json
import re
from vllm import LLM, SamplingParams

PROMPT = """Replace specific details with generic tokens. Rules:
- Names/pronouns → [PERSON1],[PERSON2],etc.
- Objects/nouns → [ITEM1],[ITEM2],etc.
- Numbers/amounts → [N]
- Locations/places → [LOC1],[LOC2],etc.
- Time references → [TIME1],[TIME2],etc.
Keep verbs and sentence structure. Output ONLY the generalized pattern, nothing else.

Examples:
"John has 5 apples" → [PERSON1] has [N] [ITEM1]
"She gave 2 cookies to Mary at the store" → [PERSON1] gave [N] [ITEM1] to [PERSON2] at [LOC1]
"He walks 3 miles to school every morning" → [PERSON1] walks [N] [ITEM1] to [LOC1] every [TIME1]

Input: "{span}"
Output: """

TEST_SPANS = [
    "Natalia sold clips to 48 of her friends in April",
    "Weng earns $12 an hour for babysitting",
    "Betty has 3 red and 2 blue marbles",
    "She eats three for breakfast every morning",
    "Josh decides to try flipping a house",
    "The farmers market sells oranges for $2 each",
    "How many apples does John have left?",
]

print("Loading Qwen...")
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=4,
    trust_remote_code=True,
    max_model_len=512,
    gpu_memory_utilization=0.85,
)

params = SamplingParams(temperature=0.1, max_tokens=150, stop=["\n\n", "```"])

prompts = [PROMPT.format(span=s) for s in TEST_SPANS]
outputs = llm.generate(prompts, params)

print("\n" + "="*60)
for span, out in zip(TEST_SPANS, outputs):
    raw = out.outputs[0].text.strip()
    print(f"\nInput:  \"{span}\"")
    print(f"Raw:    \"{raw}\"")

    # Check if it has generic tokens
    has_tokens = bool(re.search(r'\[(?:PERSON|ITEM|N|LOC|TIME|UNIT|ROLE)\d*\]', raw))
    print(f"Valid:  {has_tokens}")
print("="*60)
