"""
LLM API integration for mathdecomp.

Supports OpenAI and Anthropic APIs for reliable structured output.

Usage:
    export OPENAI_API_KEY=sk-...
    # or
    export ANTHROPIC_API_KEY=sk-ant-...

    from mycelium.mathdecomp.llm_api import decompose_with_api
    result = decompose_with_api("Tim has 10 dollars...")
"""

import os
import json
from typing import Optional, Literal

from .schema import Decomposition, Ref, RefType
from .executor import verify_decomposition


# Structured prompt for API calls
# Note: Double braces {{ }} escape the braces for .format()
DECOMPOSE_PROMPT = '''You are a math problem decomposer. Break down math problems into atomic computation steps.

RULES:
1. Extract all numbers from the problem with semantic names
2. Each step uses a function with flexible arity (inputs as a list)
3. Each input is a reference with "type" and "id" fields
4. Available functions: add, sub, mul, truediv, sqrt, abs, floor, ceil
5. Steps must be in dependency order

OUTPUT FORMAT (valid JSON only, no markdown):
{{
  "extractions": [
    {{"id": "<name>", "value": <number>, "span": "<text>", "offset": [<start>, <end>]}}
  ],
  "steps": [
    {{
      "id": "s1",
      "func": "<function_name>",
      "inputs": [{{"type": "extraction", "id": "<id>"}}, ...],
      "result": <number>,
      "semantic": "<meaning>"
    }}
  ],
  "answer_ref": {{"type": "step", "id": "<id>"}},
  "answer_value": <number>
}}

EXAMPLE:
Problem: Tim buys 3 toys at $2 each. How much does he spend?
{{
  "extractions": [
    {{"id": "num_toys", "value": 3, "span": "3 toys", "offset": [10, 16]}},
    {{"id": "toy_price", "value": 2, "span": "$2", "offset": [20, 22]}}
  ],
  "steps": [
    {{
      "id": "s1",
      "func": "mul",
      "inputs": [{{"type": "extraction", "id": "num_toys"}}, {{"type": "extraction", "id": "toy_price"}}],
      "result": 6,
      "semantic": "total_cost"
    }}
  ],
  "answer_ref": {{"type": "step", "id": "s1"}},
  "answer_value": 6
}}

Problem: {problem}'''


def call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """Call OpenAI API with JSON mode."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI()  # Uses OPENAI_API_KEY env var

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output valid JSON only. No markdown, no explanation."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    return response.choices[0].message.content


def call_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Anthropic API."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = Anthropic()  # Uses ANTHROPIC_API_KEY env var

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt + "\n\nRespond with valid JSON only, no markdown code blocks."}
        ],
        temperature=0.1,
    )

    return response.content[0].text


def parse_response(response: str, problem: str) -> Optional[Decomposition]:
    """Parse API response into Decomposition."""
    import re

    # Strip markdown code blocks if present
    response = re.sub(r'^```json\s*', '', response.strip())
    response = re.sub(r'\s*```$', '', response)

    try:
        data = json.loads(response)

        return Decomposition(
            problem=problem,
            extractions=data.get("extractions", []),
            steps=data.get("steps", []),
            answer_ref=data.get("answer_ref", {"type": "step", "id": "s1"}),
            answer_value=data.get("answer_value", 0),
            verified=False,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Parse error: {e}")
        print(f"Response: {response[:500]}")
        return None


def decompose_with_api(
    problem: str,
    provider: Literal["openai", "anthropic", "auto"] = "auto",
    model: Optional[str] = None,
    expected_answer: Optional[float] = None,
    max_retries: int = 2,
) -> Decomposition:
    """
    Decompose a math problem using LLM API.

    Args:
        problem: The math problem text
        provider: "openai", "anthropic", or "auto" (tries both)
        model: Override model name
        expected_answer: If provided, verify final answer matches
        max_retries: Number of retries on failure

    Returns:
        Decomposition object
    """
    # Auto-detect provider
    if provider == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            raise ValueError(
                "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
            )

    # Select call function and default model
    if provider == "openai":
        call_fn = call_openai
        model = model or "gpt-4o"
    else:
        call_fn = call_anthropic
        model = model or "claude-sonnet-4-20250514"

    prompt = DECOMPOSE_PROMPT.format(problem=problem)
    decomp = None

    for attempt in range(max_retries):
        try:
            response = call_fn(prompt, model)
            decomp = parse_response(response, problem)

            if decomp is None:
                continue

            # Verify
            decomp = verify_decomposition(decomp, expected_answer)

            if decomp.verified:
                return decomp

            # Retry with error context
            if attempt < max_retries - 1:
                prompt = (
                    f"{DECOMPOSE_PROMPT.format(problem=problem)}\n\n"
                    f"Previous attempt had error: {decomp.error}\n"
                    f"Fix the issue and output valid JSON."
                )

        except Exception as e:
            print(f"API error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return Decomposition(
                    problem=problem,
                    extractions=[],
                    steps=[],
                    answer_ref=Ref(type=RefType.STEP, id="s1"),
                    answer_value=0,
                    verified=False,
                    error=str(e),
                )

    return decomp or Decomposition(
        problem=problem,
        extractions=[],
        steps=[],
        answer_ref=Ref(type=RefType.STEP, id="s1"),
        answer_value=0,
        verified=False,
        error="Failed after all retries",
    )


def decompose_with_cascade(
    problem: str,
    expected_answer: Optional[float] = None,
    cheap_model: str = "gpt-4o-mini",
    expensive_model: str = "gpt-4o",
) -> Decomposition:
    """
    Cascade approach: try cheap model first, escalate to expensive on failure.

    This optimizes cost while maintaining accuracy.
    """
    # Try cheap model first
    decomp = decompose_with_api(
        problem,
        model=cheap_model,
        expected_answer=expected_answer,
        max_retries=1,
    )

    if decomp.verified:
        return decomp

    # Escalate to expensive model
    return decompose_with_api(
        problem,
        model=expensive_model,
        expected_answer=expected_answer,
        max_retries=2,
    )


def test_api():
    """Quick test of API integration."""
    from .executor import trace_execution

    problems = [
        ("Tim has 10 dollars. He buys 3 toys at 2 dollars each. How much left?", 4),
        ("A baker made 24 cookies. She sold 8 in the morning and 6 in the afternoon. How many left?", 10),
        ("There are 5 boxes with 4 apples each. Tom takes 3 apples. How many apples are left?", 17),
    ]

    print("=" * 60)
    print("mathdecomp API Integration Test")
    print("=" * 60)

    correct = 0
    for problem, expected in problems:
        print(f"\nProblem: {problem}")
        print(f"Expected: {expected}")

        decomp = decompose_with_api(problem, expected_answer=expected, model="gpt-4o")

        print(f"Verified: {decomp.verified}")
        if decomp.verified:
            print(trace_execution(decomp))
            correct += 1
        else:
            print(f"Error: {decomp.error}")

    print(f"\n{'=' * 60}")
    print(f"Results: {correct}/{len(problems)} correct")
    print("=" * 60)


if __name__ == "__main__":
    test_api()
