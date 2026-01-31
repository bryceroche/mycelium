"""LLM Decomposer - Breaks problems into atomic steps using signature menu."""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecomposedStep:
    """A single decomposed step from the LLM."""
    description: str
    params: List = None  # Extracted numeric parameters
    func_hint: Optional[str] = None  # Optional function name hint

    def __post_init__(self):
        if self.params is None:
            self.params = []


class LLMDecomposer:
    """Decomposes math problems into atomic steps using LLM + signature menu."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize decomposer.

        Args:
            model: LiteLLM model name (e.g., "gpt-4o-mini", "claude-3-haiku-20240307")
        """
        self.model = model

    def decompose(self, problem: str, signature_menu: str) -> List[DecomposedStep]:
        """Decompose a problem into atomic steps.

        Args:
            problem: The math problem text
            signature_menu: Formatted menu of proven patterns (from db.format_signature_menu())

        Returns:
            List of DecomposedStep objects
        """
        prompt = self._build_prompt(problem, signature_menu)
        response = self._call_llm(prompt)
        steps = self._parse_response(response)
        return steps

    def _build_prompt(self, problem: str, signature_menu: str) -> str:
        """Build the decomposition prompt."""
        return f'''You are decomposing math word problems into atomic computational steps.

Here are atomic operations with proven phrasings. Use these patterns:

{signature_menu}

INSTRUCTIONS:
1. Break the problem into simple computational steps
2. Each step should match one of the patterns above
3. Extract numeric parameters from the problem
4. Output valid JSON

PROBLEM:
{problem}

OUTPUT FORMAT (JSON array):
[
  {{"description": "step description matching a pattern", "params": [num1, num2]}},
  {{"description": "another step", "params": [num]}}
]

Respond ONLY with the JSON array, no other text.'''

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return response text."""
        try:
            from litellm import completion

            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            logger.warning("litellm not installed, using mock response")
            return self._mock_response(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _mock_response(self, prompt: str) -> str:
        """Mock response for testing without LLM."""
        return '[{"description": "mock step", "params": [1, 2]}]'

    def _parse_response(self, response: str) -> List[DecomposedStep]:
        """Parse LLM response into DecomposedStep objects."""
        # Clean up response (remove markdown code blocks if present)
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        try:
            data = json.loads(response)
            if not isinstance(data, list):
                logger.warning(f"Expected list, got {type(data)}")
                return []

            steps = []
            for item in data:
                if isinstance(item, dict):
                    step = DecomposedStep(
                        description=item.get("description", ""),
                        params=item.get("params", []),
                        func_hint=item.get("func_hint"),
                    )
                    steps.append(step)

            return steps
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response[:500]}")
            return []


def decompose_problem(problem: str, db=None, model: str = "gpt-4o-mini") -> List[DecomposedStep]:
    """Convenience function to decompose a problem.

    Args:
        problem: The math problem text
        db: Optional StepSignatureDB instance (will create if not provided)
        model: LLM model to use

    Returns:
        List of DecomposedStep objects
    """
    if db is None:
        from mycelium.step_signatures.db import get_step_db
        db = get_step_db()

    menu = db.format_signature_menu(max_examples_per_func=10)
    decomposer = LLMDecomposer(model=model)
    return decomposer.decompose(problem, menu)
