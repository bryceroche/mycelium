"""DSL Auto-Rewriter: Generate improved DSL scripts using LLM.

Per CLAUDE.md: "rewrite DSL if centroid avg outside confidence bounds"
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RewriteCandidate:
    """A signature that needs DSL rewriting."""
    signature_id: int
    step_type: str
    description: str
    current_dsl: Optional[str]
    uses: int
    successes: int
    success_rate: float
    last_rewrite_at: Optional[str] = None


async def generate_improved_dsl(
    candidate: RewriteCandidate,
    client,
    failure_examples: list[dict] = None,
) -> Optional[str]:
    """Use LLM to generate an improved DSL script.

    Args:
        candidate: The signature needing improvement
        client: LLM client for generation
        failure_examples: Optional examples of what went wrong

    Returns:
        New DSL script JSON, or None if generation failed
    """
    # Build prompt with context about the signature and its failures
    system_prompt = """You are a DSL script optimizer. Given a math operation signature
that is failing too often, generate an improved DSL script.

DSL scripts are JSON with this format:
{
    "type": "math",
    "script": "a + b",  // The math expression using param names
    "params": ["a", "b"],  // Parameter names to extract
    "aliases": {"a": ["first", "value1"], "b": ["second", "value2"]},  // Optional aliases
    "purpose": "Add two numbers together"  // Description
}

Rules:
1. The script must be a valid Python math expression
2. Use descriptive param names that match the operation semantics
3. Include aliases for common alternative names
4. Keep it simple - prefer basic operations

Respond with ONLY the JSON, no explanation."""

    # Build user prompt with signature context
    user_prompt = f"""Signature: {candidate.step_type}
Description: {candidate.description}

Current DSL (success rate {candidate.success_rate:.1%}):
{candidate.current_dsl or "None"}

"""

    if failure_examples:
        user_prompt += "Recent failures:\n"
        for ex in failure_examples[:3]:
            user_prompt += f"- Input: {ex.get('input', 'N/A')}, Error: {ex.get('error', 'N/A')}\n"
        user_prompt += "\n"

    user_prompt += "Generate an improved DSL script that will work better for this operation."

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await client.generate(messages, temperature=0.2)

        # Parse and validate the response
        dsl = _parse_dsl_response(response)
        if dsl:
            logger.info(
                "[rewriter] Generated new DSL for '%s': %s",
                candidate.step_type, dsl.get("script", "N/A")
            )
            return json.dumps(dsl)

    except Exception as e:
        logger.error("[rewriter] Failed to generate DSL for '%s': %s", candidate.step_type, e)

    return None


def _parse_dsl_response(response: str) -> Optional[dict]:
    """Parse and validate LLM-generated DSL."""
    try:
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        dsl = json.loads(response)

        # Validate required fields
        if not isinstance(dsl, dict):
            return None
        if "script" not in dsl or "params" not in dsl:
            return None
        if not isinstance(dsl["params"], list):
            return None

        # Ensure type is set
        dsl.setdefault("type", "math")

        return dsl

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("[rewriter] Failed to parse DSL response: %s", e)
        return None


