"""DSL Generator: LLM-based generation of DSL scripts for reliable signatures.

When a signature becomes reliable (enough uses + high success rate), we use
an LLM to analyze its examples and generate a custom DSL script.

This is the "smart work once, execute forever" approach:
1. Signature accumulates successful examples
2. Once reliable, LLM analyzes the pattern
3. LLM generates appropriate DSL (math, sympy, or custom)
4. DSL stored on signature, used for all future matches
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Prompt for DSL generation
DSL_GENERATION_PROMPT = '''You are a DSL generator for a math problem solver.

Given examples of a step type that has been successfully solved multiple times,
generate a DSL script that can solve this type of step deterministically.

## DSL Format

```json
{
  "type": "math|sympy|custom",
  "script": "executable expression",
  "params": ["param1", "param2"],
  "aliases": {"param1": ["alias1", "alias2"]},
  "fallback": "guidance"
}
```

## DSL Types

1. **math**: Simple arithmetic (no external deps)
   - Operators: +, -, *, /, **, //, %
   - Functions: sqrt, abs, min, max, round, sin, cos, tan, log, exp
   - Example: `{"type": "math", "script": "(a + b) / 2", "params": ["a", "b"]}`

2. **sympy**: Symbolic algebra (for equations, simplification)
   - Functions: solve, simplify, expand, factor, diff, integrate
   - Example: `{"type": "sympy", "script": "solve(Eq(a*x + b, 0), x)", "params": ["a", "b"]}`

3. **custom**: Predefined operators
   - apply_quadratic_formula(a, b, c), solve_linear(a, b), complete_square(a, b, c)
   - Example: `{"type": "custom", "script": "apply_quadratic_formula(a, b, c)", "params": ["a", "b", "c"]}`

## Rules

1. ONLY generate DSL if the pattern is truly deterministic
2. If the step requires reasoning/judgment, respond with: {"type": "none"}
3. Params should be generic (a, b, c) not specific to examples
4. Include aliases for common variations of param names
5. Use the simplest type that works (math > sympy > custom)

## Examples

Step: "Calculate 25% of 200"
DSL: {"type": "math", "script": "(percentage / 100) * base", "params": ["percentage", "base"], "aliases": {"percentage": ["pct", "percent"], "base": ["value", "amount", "total"]}}

Step: "Solve 2x + 5 = 15 for x"
DSL: {"type": "sympy", "script": "solve(Eq(a*x + b, c), x)", "params": ["a", "b", "c"], "aliases": {"a": ["coefficient"], "b": ["constant"], "c": ["result"]}}

Step: "Determine which approach is most efficient"
DSL: {"type": "none"}

Now analyze these examples and generate the DSL:

## Signature Info
Step Type: {step_type}
Description: {description}

## Successful Examples
{examples}

Generate the DSL JSON (or {"type": "none"} if not suitable for DSL):
'''


async def generate_dsl_for_signature(
    client,  # GroqClient or similar
    step_type: str,
    description: str,
    examples: list[dict],
) -> Optional[str]:
    """Generate a DSL script for a signature using LLM.

    Args:
        client: LLM client for generation
        step_type: The signature's step type
        description: The signature's description
        examples: List of example dicts with step_text, result, success

    Returns:
        JSON string of DSL spec, or None if DSL not suitable
    """
    if not examples:
        return None

    # Format examples for prompt
    examples_str = "\n".join(
        f"- Step: {ex.get('step_text', '')}\n  Result: {ex.get('result', '')}"
        for ex in examples[:5]  # Limit to 5 examples
        if ex.get('success', False)
    )

    if not examples_str:
        return None

    prompt = DSL_GENERATION_PROMPT.format(
        step_type=step_type,
        description=description,
        examples=examples_str,
    )

    try:
        messages = [
            {"role": "system", "content": "You generate DSL scripts. Respond with only valid JSON."},
            {"role": "user", "content": prompt},
        ]

        response = await client.generate(messages, temperature=0.0)

        # Extract JSON from response
        dsl_json = _extract_json(response)
        if not dsl_json:
            logger.warning("Failed to extract JSON from DSL generation response")
            return None

        # Parse and validate
        try:
            dsl = json.loads(dsl_json)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in DSL generation response: %s", dsl_json[:100])
            return None

        # Check if DSL is suitable
        if dsl.get("type") == "none":
            logger.info("LLM determined DSL not suitable for step_type=%s", step_type)
            return None

        # Validate required fields
        if not all(k in dsl for k in ["type", "script", "params"]):
            logger.warning("DSL missing required fields: %s", dsl)
            return None

        if dsl["type"] not in ("math", "sympy", "custom"):
            logger.warning("Invalid DSL type: %s", dsl["type"])
            return None

        # Add fallback if not present
        if "fallback" not in dsl:
            dsl["fallback"] = "guidance"

        logger.info(
            "Generated DSL for step_type=%s: type=%s script=%s",
            step_type, dsl["type"], dsl["script"][:50]
        )

        return json.dumps(dsl)

    except Exception as e:
        logger.error("DSL generation failed: %s", e)
        return None


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON object from text response."""
    # Try to find JSON in the response
    text = text.strip()

    # If it starts with {, try to parse directly
    if text.startswith("{"):
        # Find matching closing brace
        depth = 0
        for i, c in enumerate(text):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[:i+1]

    # Try to find ```json blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    # Try to find any { } block
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

    return None


async def maybe_generate_dsl(
    db,  # StepSignatureDB
    client,  # GroqClient
    signature_id: int,
    min_uses: int = 3,
    min_success_rate: float = 0.8,
) -> bool:
    """Check if signature is ready for DSL generation, and generate if so.

    Args:
        db: StepSignatureDB instance
        client: LLM client for generation
        signature_id: ID of signature to check
        min_uses: Minimum uses before generating DSL (default: 3)
        min_success_rate: Minimum success rate before generating DSL

    Returns:
        True if DSL was generated, False otherwise
    """
    sig = db.get_signature(signature_id)
    if not sig:
        return False

    # Already has DSL
    if sig.dsl_script:
        return False

    # Not enough data
    if sig.uses < min_uses:
        return False

    # Not reliable enough
    if sig.success_rate < min_success_rate:
        return False

    # Get examples for context
    examples = db.get_signature_examples(signature_id, limit=10)
    if not examples:
        return False

    # Generate DSL
    dsl_script = await generate_dsl_for_signature(
        client=client,
        step_type=sig.step_type,
        description=sig.description,
        examples=examples,
    )

    if dsl_script:
        db.update_dsl_script(signature_id, dsl_script)
        logger.info("Generated and saved DSL for signature %d", signature_id)
        return True

    return False
