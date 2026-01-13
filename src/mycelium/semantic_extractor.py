"""Semantic Extractor: Extract meaning from step results for DSL parameter mapping.

The Problem:
    DSL params have semantic names: area_ABC, base, height
    Context has generic names: step_1, step_2, problem_num_0
    LLM guessing at mappings produces garbage results

The Solution:
    Extract semantic meaning when step is solved:
    - "area of triangle ABC" → can match DSL param "area_ABC"
    - "length of base" → can match DSL param "base"

    This enables accurate parameter mapping without LLM guessing.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SemanticInfo:
    """Semantic information extracted from a step result."""
    value: Optional[float]  # Numeric value if applicable
    meaning: str  # What the result represents
    semantic_type: str  # Category: area, length, count, ratio, etc.
    confidence: float = 1.0  # How confident we are in extraction


# Common semantic types and their indicators
SEMANTIC_TYPES = {
    "area": ["area", "square", "region", "surface"],
    "length": ["length", "distance", "side", "radius", "diameter", "height", "width", "base"],
    "angle": ["angle", "degrees", "radians", "theta"],
    "count": ["count", "number of", "how many", "total"],
    "ratio": ["ratio", "proportion", "fraction", "percentage", "percent"],
    "sum": ["sum", "total", "add", "plus"],
    "difference": ["difference", "subtract", "minus", "less"],
    "product": ["product", "multiply", "times"],
    "quotient": ["quotient", "divide", "divided by"],
    "coordinate": ["coordinate", "point", "position", "x", "y", "z"],
    "expression": ["expression", "equation", "formula", "polynomial"],
    "probability": ["probability", "chance", "likelihood"],
    "volume": ["volume", "capacity", "cubic"],
    "rate": ["rate", "speed", "velocity", "per"],
}


def extract_numeric_value(text: str) -> Optional[float]:
    """Extract numeric value from result text.

    Handles formats like:
    - "24"
    - "24.5"
    - "The area is 24 square units"
    - "Answer: 3/4" → 0.75
    - "\\frac{3}{4}" → 0.75
    """
    if not text:
        return None

    text = text.strip()

    # Try direct float conversion first
    try:
        return float(text)
    except ValueError:
        pass

    # Try to find fraction patterns
    # LaTeX fraction: \frac{num}{den}
    frac_match = re.search(r'\\frac\{([^}]+)\}\{([^}]+)\}', text)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                return num / den
        except ValueError:
            pass

    # Simple fraction: num/den
    simple_frac = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', text)
    if simple_frac:
        try:
            num = float(simple_frac.group(1))
            den = float(simple_frac.group(2))
            if den != 0:
                return num / den
        except ValueError:
            pass

    # Extract last number in text (often the answer)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def infer_semantic_type(task: str, result: str) -> str:
    """Infer semantic type from task description and result.

    Args:
        task: The step's task description
        result: The step's result

    Returns:
        Semantic type string (e.g., "area", "length", "count")
    """
    combined = f"{task} {result}".lower()

    # Check each semantic type
    for sem_type, indicators in SEMANTIC_TYPES.items():
        for indicator in indicators:
            if indicator in combined:
                return sem_type

    return "value"  # Default


def extract_meaning_from_task(task: str) -> str:
    """Extract semantic meaning from task description.

    The task often contains what we're computing:
    - "Calculate the area of triangle ABC" → "area of triangle ABC"
    - "Find the length of side BC" → "length of side BC"
    - "Compute the sum of x and y" → "sum of x and y"

    This is often sufficient without LLM extraction.
    """
    task_lower = task.lower()

    # Pattern: "Calculate/Find/Compute the X of Y"
    patterns = [
        r"(?:calculate|find|compute|determine|get)\s+(?:the\s+)?(.+)",
        r"(?:what is|what's)\s+(?:the\s+)?(.+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, task_lower)
        if match:
            meaning = match.group(1).strip()
            # Clean up trailing punctuation
            meaning = re.sub(r'[?.!]+$', '', meaning)
            return meaning

    # Fallback: use the whole task as meaning
    return task


def extract_semantic_info(
    task: str,
    result: str,
    use_llm: bool = False,
    client = None,
) -> SemanticInfo:
    """Extract semantic information from a step's task and result.

    Args:
        task: The step's task description
        result: The step's result text
        use_llm: Whether to use LLM for more accurate extraction
        client: GroqClient for LLM extraction (required if use_llm=True)

    Returns:
        SemanticInfo with value, meaning, and type
    """
    # Extract numeric value
    value = extract_numeric_value(result)

    # Extract meaning from task (rule-based, fast)
    meaning = extract_meaning_from_task(task)

    # Infer semantic type
    sem_type = infer_semantic_type(task, result)

    logger.debug(
        "[semantic] Extracted: value=%s meaning='%s' type='%s'",
        value, meaning[:50] if meaning else "", sem_type
    )

    return SemanticInfo(
        value=value,
        meaning=meaning,
        semantic_type=sem_type,
        confidence=0.8 if value is not None else 0.5,
    )


def build_rich_context(step_results: list) -> dict:
    """Build rich context dictionary from step results.

    Instead of: {"step_1": "24"}
    Returns: {
        "step_1": {
            "value": 24.0,
            "meaning": "area of triangle ABC",
            "type": "area",
            "raw": "24"
        }
    }

    Args:
        step_results: List of StepResult objects with semantic fields

    Returns:
        Rich context dictionary for DSL parameter mapping
    """
    context = {}

    for sr in step_results:
        context[sr.step_id] = {
            "value": sr.numeric_value,
            "meaning": sr.semantic_meaning,
            "type": sr.semantic_type,
            "raw": sr.result,
        }

    return context


def match_param_to_context(
    param_name: str,
    rich_context: dict,
    param_aliases: list[str] = None,
) -> Optional[tuple[str, float]]:
    """Match a DSL parameter name to a context entry by semantic similarity.

    Args:
        param_name: DSL parameter name (e.g., "area_ABC", "base")
        rich_context: Rich context from build_rich_context()
        param_aliases: Optional list of aliases for the param

    Returns:
        Tuple of (context_key, confidence) or None if no match
    """
    param_lower = param_name.lower()
    aliases = [a.lower() for a in (param_aliases or [])]

    best_match = None
    best_score = 0.0

    for ctx_key, ctx_info in rich_context.items():
        meaning = ctx_info.get("meaning", "").lower()
        sem_type = ctx_info.get("type", "").lower()

        score = 0.0

        # Exact match in meaning
        if param_lower in meaning or meaning in param_lower:
            score = 0.9

        # Check aliases
        for alias in aliases:
            if alias in meaning or meaning in alias:
                score = max(score, 0.85)

        # Type match (e.g., param "area_ABC" matches type "area")
        if sem_type and sem_type in param_lower:
            score = max(score, 0.7)

        # Partial token match
        param_tokens = set(re.split(r'[_\s]+', param_lower))
        meaning_tokens = set(re.split(r'[_\s]+', meaning))
        overlap = param_tokens & meaning_tokens
        if overlap:
            token_score = len(overlap) / max(len(param_tokens), len(meaning_tokens))
            score = max(score, token_score * 0.8)

        if score > best_score:
            best_score = score
            best_match = ctx_key

    if best_match and best_score >= 0.5:
        return best_match, best_score

    return None
