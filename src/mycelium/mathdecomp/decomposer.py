"""
Decomposer - LLM-powered recursive decomposition of math problems.

This is the main entry point. It uses an LLM to:
1. Extract variables from the problem text
2. Break down the solution into atomic steps
3. Verify the decomposition is correct
4. Recursively refine if steps aren't truly atomic
"""

import json
import re
from typing import Optional, Callable, Any
from .schema import Decomposition, Extraction, Step, Ref, RefType


# Default prompt template for decomposition
DECOMPOSE_PROMPT = '''Decompose this math problem into atomic computation steps.

RULES:
1. First extract all numbers from the problem with semantic names
2. Each step has EXACTLY two inputs (either extracted values or prior step results)
3. Each input must be a reference: {"type": "extraction", "id": "..."} or {"type": "step", "id": "..."}
4. Use only these operators: +, -, *, /
5. Steps must be in dependency order (can only reference prior steps)

OUTPUT FORMAT (JSON):
{
  "extractions": [
    {"id": "<semantic_name>", "value": <number>, "span": "<text from problem>", "offset": [<start>, <end>]}
  ],
  "steps": [
    {
      "id": "s1",
      "op": "<+|-|*|/>",
      "left": {"type": "extraction|step", "id": "<ref_id>"},
      "right": {"type": "extraction|step", "id": "<ref_id>"},
      "result": <computed_number>,
      "semantic": "<what_this_represents>"
    }
  ],
  "answer_ref": {"type": "step", "id": "<final_step_id>"},
  "answer_value": <final_number>
}

EXAMPLE:
Problem: "Tim has 10 dollars. He buys 3 toys at 2 dollars each. How much does he have left?"

{
  "extractions": [
    {"id": "tim_money", "value": 10, "span": "10 dollars", "offset": [8, 18]},
    {"id": "num_toys", "value": 3, "span": "3 toys", "offset": [28, 34]},
    {"id": "toy_price", "value": 2, "span": "2 dollars each", "offset": [38, 52]}
  ],
  "steps": [
    {
      "id": "s1",
      "op": "*",
      "left": {"type": "extraction", "id": "num_toys"},
      "right": {"type": "extraction", "id": "toy_price"},
      "result": 6,
      "semantic": "total_cost"
    },
    {
      "id": "s2",
      "op": "-",
      "left": {"type": "extraction", "id": "tim_money"},
      "right": {"type": "step", "id": "s1"},
      "result": 4,
      "semantic": "remaining_money"
    }
  ],
  "answer_ref": {"type": "step", "id": "s2"},
  "answer_value": 4
}

Now decompose this problem (output JSON only):
Problem: "{problem}"
'''


def parse_llm_response(response: str, problem: str) -> Optional[Decomposition]:
    """Parse LLM response into Decomposition object."""
    # Try to extract JSON from response
    # Look for { ... } pattern
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())

        # Build Decomposition from parsed data
        return Decomposition(
            problem=problem,
            extractions=data.get("extractions", []),
            steps=data.get("steps", []),
            answer_ref=data.get("answer_ref", {"type": "step", "id": "s1"}),
            answer_value=data.get("answer_value", 0),
            verified=False,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return None


def decompose(
    problem: str,
    llm_call: Callable[[str], str],
    expected_answer: Optional[float] = None,
    max_retries: int = 3,
    verify: bool = True,
) -> Decomposition:
    """
    Decompose a math problem into atomic steps.

    Args:
        problem: The math problem text
        llm_call: Function that takes a prompt and returns LLM response
        expected_answer: If provided, verify final answer matches
        max_retries: Number of retries if decomposition fails
        verify: Whether to verify the decomposition

    Returns:
        Decomposition object (check .verified and .error for status)
    """
    from .executor import verify_decomposition

    prompt = DECOMPOSE_PROMPT.format(problem=problem)

    for attempt in range(max_retries):
        # Call LLM
        response = llm_call(prompt)

        # Parse response
        decomp = parse_llm_response(response, problem)

        if decomp is None:
            continue  # Retry on parse failure

        if not verify:
            return decomp

        # Verify decomposition
        decomp = verify_decomposition(decomp, expected_answer)

        if decomp.verified:
            return decomp

        # Add error context to prompt for retry
        if attempt < max_retries - 1:
            prompt = (
                f"{DECOMPOSE_PROMPT.format(problem=problem)}\n\n"
                f"Previous attempt had error: {decomp.error}\n"
                f"Please fix and try again."
            )

    # Return last attempt even if failed
    if decomp is None:
        decomp = Decomposition(
            problem=problem,
            extractions=[],
            steps=[],
            answer_ref=Ref(type=RefType.STEP, id="s1"),
            answer_value=0,
            verified=False,
            error="Failed to parse LLM response after all retries",
        )

    return decomp


# Convenience function for testing with a simple mock LLM
def mock_decompose(problem: str) -> Decomposition:
    """
    Simple rule-based decomposition for testing.
    Only handles very simple single-operation problems.
    """
    import re

    # Extract all numbers
    numbers = re.findall(r'\d+(?:\.\d+)?', problem)
    if len(numbers) < 2:
        return Decomposition(
            problem=problem,
            extractions=[],
            steps=[],
            answer_ref=Ref.step("s1"),
            answer_value=0,
            verified=False,
            error="Not enough numbers found",
        )

    # Create extractions
    extractions = [
        Extraction(
            id=f"n{i}",
            value=float(n),
            span=n,
            offset_start=problem.find(n),
            offset_end=problem.find(n) + len(n),
        )
        for i, n in enumerate(numbers)
    ]

    # Guess operation from keywords
    problem_lower = problem.lower()
    if "total" in problem_lower or "sum" in problem_lower or "add" in problem_lower:
        op = "+"
    elif "left" in problem_lower or "remain" in problem_lower or "subtract" in problem_lower:
        op = "-"
    elif "each" in problem_lower or "times" in problem_lower or "multiply" in problem_lower:
        op = "*"
    elif "split" in problem_lower or "divide" in problem_lower or "per" in problem_lower:
        op = "/"
    else:
        op = "+"  # Default

    # Simple two-number operation
    a, b = float(numbers[0]), float(numbers[1])
    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        result = a / b if b != 0 else 0

    step = Step(
        id="s1",
        op=op,
        left=Ref.extraction("n0"),
        right=Ref.extraction("n1"),
        result=result,
        semantic="result",
    )

    return Decomposition(
        problem=problem,
        extractions=extractions,
        steps=[step],
        answer_ref=Ref.step("s1"),
        answer_value=result,
        verified=True,  # Mock always "works"
    )
