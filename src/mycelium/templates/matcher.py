"""Template matching and slot mapping."""
import json
import logging
from typing import Dict, Any, Tuple, Optional

from .models import Template, Example
from .db import find_nearest_example, get_template, get_all_templates
from mycelium.embedding_cache import cached_embed
from mycelium.mathdecomp.llm_api import call_openai, call_anthropic

logger = logging.getLogger(__name__)


def _call_llm(prompt: str) -> str:
    """Call LLM using available provider (OpenAI or Anthropic).

    Auto-detects based on environment variables.
    """
    import os

    if os.environ.get("OPENAI_API_KEY"):
        return call_openai(prompt)
    elif os.environ.get("ANTHROPIC_API_KEY"):
        return call_anthropic(prompt)
    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
        )


def match_template(problem: str) -> Tuple[Optional[Template], float]:
    """
    Find the best matching template for a problem.

    Returns:
        Tuple of (Template, similarity_score) or (None, 0.0) if no match
    """
    # Embed the problem
    embedding = cached_embed(problem)
    if embedding is None:
        logger.warning("[matcher] Failed to embed problem")
        return None, 0.0

    # Find nearest example
    results = find_nearest_example(embedding, top_k=1)

    if not results:
        # No examples yet - fall back to first template or None
        templates = get_all_templates()
        if templates:
            logger.info("[matcher] No examples found, using first template")
            return templates[0], 0.0
        return None, 0.0

    example, similarity = results[0]
    template = get_template(example.template_id)

    logger.info(f"[matcher] Matched template '{template.name}' with similarity {similarity:.3f}")

    return template, similarity


def map_slots(problem: str, template: Template) -> Dict[str, Any]:
    """
    Use LLM to map problem text to template slots.

    This is the KEY insight - we ask the LLM to do pure token alignment,
    which is what transformers are best at.
    """
    prompt = f'''Map this problem to the template slots.

TEMPLATE: {template.pattern}
SLOTS: {template.slots}

PROBLEM: {problem}

Extract the value for each slot from the problem.
For numeric slots, extract just the number.
For text slots, extract the relevant text.

Output JSON mapping slot names to values:
{{"SLOT1": value1, "SLOT2": value2, ...}}

Example:
Template: "[AGENT] has [X] [OBJECT]. Gives [Y] to [RECIPIENT]."
Slots: ["AGENT", "X", "OBJECT", "Y", "RECIPIENT"]
Problem: "Tim has 10 apples. He gives 3 to Mary."
Output: {{"AGENT": "Tim", "X": 10, "OBJECT": "apples", "Y": 3, "RECIPIENT": "Mary"}}

Now map the problem to slots. Output ONLY valid JSON:'''

    try:
        response = _call_llm(prompt)
        # Strip markdown code blocks if present
        import re
        response = re.sub(r'^```json\s*', '', response.strip())
        response = re.sub(r'\s*```$', '', response)

        slots = json.loads(response)
        logger.info(f"[matcher] Mapped slots: {slots}")
        return slots
    except Exception as e:
        logger.error(f"[matcher] Failed to map slots: {e}")
        return {}


def map_slots_simple(problem: str, template: Template) -> Dict[str, Any]:
    """
    Simpler slot mapping for templates with obvious numeric slots.
    Uses a more constrained prompt.
    """
    # Build slot descriptions
    slot_list = ", ".join(template.slots)

    prompt = f'''Extract values from this math problem.

Problem: {problem}

Extract these values: {slot_list}

Output JSON with numeric values:
{{"{template.slots[0]}": <number>, ...}}

ONLY output the JSON, nothing else:'''

    try:
        response = _call_llm(prompt)
        # Strip markdown code blocks if present
        import re
        response = re.sub(r'^```json\s*', '', response.strip())
        response = re.sub(r'\s*```$', '', response)

        slots = json.loads(response)

        # Ensure all values are numeric where possible
        for key, value in slots.items():
            if isinstance(value, str):
                try:
                    slots[key] = float(value)
                except ValueError:
                    pass

        logger.info(f"[matcher] Mapped slots (simple): {slots}")
        return slots
    except Exception as e:
        logger.error(f"[matcher] Failed to map slots (simple): {e}")
        return {}
