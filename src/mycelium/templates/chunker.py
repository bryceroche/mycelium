"""Problem chunker module.

Breaks complex problems into sub-problems that each match a template.
Uses [RESULT] placeholder to reference previous step's answer.
"""

import json
import logging
import os
from typing import List

from mycelium.mathdecomp.llm_api import call_openai, call_anthropic

logger = logging.getLogger(__name__)

CHUNKER_PROMPT = """You are a math problem decomposer. Break complex problems into simple sub-problems.

Each chunk MUST be a complete, self-contained word problem that:
- Can be solved in one arithmetic step
- Preserves full context (what are we counting? money? items? people?)
- Makes the operation CLEAR (adding, subtracting, multiplying, dividing)
- Uses [RESULT] to reference the answer from the previous chunk

EXAMPLES:

Problem: "Tim has 10 apples, gives 3 to Mary, then buys 5 more. How many does he have?"
Chunks:
1. "Tim has 10 apples and gives 3 apples to Mary. How many apples does he have left?"
2. "Tim has [RESULT] apples. He buys 5 more apples. How many apples does he have now?"

Problem: "Maria has $50. She spends $20 on lunch and then finds $10. How much does she have?"
Chunks:
1. "Maria has $50. She spends $20 on lunch. How much money does she have left?"
2. "Maria has $[RESULT]. She finds $10 more. How much money does she have now?"

Problem: "A store has 50 shirts. They sell 15 in the morning and 20 in the afternoon. How many are left?"
Chunks:
1. "A store has 50 shirts. They sell 15 shirts in the morning. How many shirts remain?"
2. "A store has [RESULT] shirts. They sell 20 shirts in the afternoon. How many shirts are left?"

Problem: "Maria earns $12 per hour and works 8 hours. How much does she earn?"
Chunks:
1. "Maria earns $12 per hour and works 8 hours. How much money does she earn?"
(Single-step problems stay as one chunk)

RULES:
- Each chunk = one arithmetic operation
- PRESERVE CONTEXT: Include units (dollars, apples, items) in every chunk
- MAKE OPERATION CLEAR: "gives away" = subtract, "finds/gets/adds" = add, "each" = multiply
- Use [RESULT] to chain chunks together
- If the problem is already simple (one step), return it as a single chunk

Now decompose this problem:
{problem}

Respond with JSON only:
{{"chunks": ["chunk1", "chunk2", ...]}}"""


def _call_llm(prompt: str) -> str:
    """Call LLM API, auto-detecting which one to use based on env vars.

    Priority: ANTHROPIC_API_KEY > OPENAI_API_KEY

    Args:
        prompt: The prompt to send

    Returns:
        Response text from LLM

    Raises:
        ValueError: If no API key is found
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.debug("Using Anthropic API")
        return call_anthropic(prompt)
    elif os.environ.get("OPENAI_API_KEY"):
        logger.debug("Using OpenAI API")
        return call_openai(prompt)
    else:
        raise ValueError("No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")


def chunk_problem(problem: str) -> List[str]:
    """Break a problem into sub-problems that each match a template.

    Each sub-problem is written as a complete problem statement.
    Uses [RESULT] placeholder to reference previous step's answer.

    Args:
        problem: The original problem text

    Returns:
        List of sub-problem strings
    """
    logger.info(f"Chunking problem: {problem[:100]}...")

    prompt = CHUNKER_PROMPT.format(problem=problem)

    try:
        response = _call_llm(prompt)
        logger.debug(f"LLM response: {response}")

        data = json.loads(response)
        chunks = data.get("chunks", [])

        if not chunks:
            logger.warning("LLM returned empty chunks, using original problem")
            return [problem]

        logger.info(f"Problem decomposed into {len(chunks)} chunks")
        return chunks

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return [problem]
    except Exception as e:
        logger.error(f"Error chunking problem: {e}")
        return [problem]
