"""LLM Decomposer - Breaks problems into atomic steps using signature menu.

Supports step chaining: steps can reference previous step results or extracted values.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class StepInput:
    """An input to a step - either a literal value or a reference."""
    value: Optional[float] = None  # Literal numeric value
    ref: Optional[str] = None  # Reference to extraction or previous step (e.g., "s1", "price")

    @property
    def is_ref(self) -> bool:
        return self.ref is not None

    def resolve(self, context: Dict[str, float]) -> float:
        """Resolve this input to a value using the context of computed results."""
        if self.value is not None:
            return self.value
        if self.ref and self.ref in context:
            return context[self.ref]
        raise ValueError(f"Cannot resolve input: ref={self.ref} not in context")


@dataclass
class Extraction:
    """An extracted value from the problem text."""
    id: str
    value: float
    span: str = ""  # The text span from the problem


@dataclass
class DecomposedStep:
    """A single decomposed step from the LLM."""
    id: str  # Step identifier (e.g., "s1", "s2")
    description: str
    inputs: List[StepInput] = field(default_factory=list)  # Inputs (refs or literals)
    params: List = None  # Legacy: raw numeric parameters (for backwards compat)
    func_hint: Optional[str] = None  # Optional function name hint
    result: Optional[float] = None  # Computed result (filled in during execution)

    def __post_init__(self):
        if self.params is None:
            self.params = []
        # Convert legacy params to inputs if no inputs provided
        if not self.inputs and self.params:
            self.inputs = [StepInput(value=p) for p in self.params]


@dataclass
class Decomposition:
    """Complete decomposition of a problem."""
    extractions: List[Extraction] = field(default_factory=list)
    steps: List[DecomposedStep] = field(default_factory=list)
    answer_step: str = ""  # ID of the step that produces the final answer


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
            List of DecomposedStep objects with step chaining support
        """
        prompt = self._build_prompt(problem, signature_menu)
        response = self._call_llm(prompt)
        decomposition = self._parse_response(response)
        return decomposition.steps

    def decompose_full(self, problem: str, signature_menu: str) -> Decomposition:
        """Decompose a problem and return the full Decomposition object.

        Returns:
            Decomposition with extractions, steps, and answer_step
        """
        prompt = self._build_prompt(problem, signature_menu)
        response = self._call_llm(prompt)
        return self._parse_response(response)

    def _build_prompt(self, problem: str, signature_menu: str) -> str:
        """Build the decomposition prompt with step chaining support."""
        return f'''You are decomposing math word problems into atomic computational steps.

Here are atomic operations you can use:

{signature_menu}

PROBLEM:
{problem}

INSTRUCTIONS:
1. First, think through the solution step by step
2. Extract all relevant numbers with semantic names
3. Create computation steps that reference extractions or previous steps
4. Each step computes ONE arithmetic operation

IMPORTANT PATTERNS:
- "twice as many X" or "2 times X" → multiply X by 2
- "X per Y for N" → X * N
- "remaining after removing X from Y" → Y - X (order matters!)
- When finding what's left: subtract what was taken from the original

OUTPUT FORMAT (JSON only, no other text):
{{
  "reasoning": "Brief explanation of solution approach",
  "extractions": [
    {{"id": "name", "value": <number>, "span": "text"}}
  ],
  "steps": [
    {{"id": "s1", "description": "operation description", "inputs": [{{"ref": "id"}}]}}
  ],
  "answer_step": "final_step_id"
}}

EXAMPLE - "Tim has 10 dollars. He buys 3 toys at 2 dollars each. How much left?":
{{
  "reasoning": "Calculate cost (3*2=6), then subtract from initial (10-6=4)",
  "extractions": [
    {{"id": "money", "value": 10, "span": "10 dollars"}},
    {{"id": "toys", "value": 3, "span": "3 toys"}},
    {{"id": "price", "value": 2, "span": "2 dollars each"}}
  ],
  "steps": [
    {{"id": "s1", "description": "multiply the numbers", "inputs": [{{"ref": "toys"}}, {{"ref": "price"}}]}},
    {{"id": "s2", "description": "subtract from the total", "inputs": [{{"ref": "money"}}, {{"ref": "s1"}}]}}
  ],
  "answer_step": "s2"
}}'''

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return response text."""
        try:
            from litellm import completion

            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000,
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
        return '''{"extractions": [{"id": "a", "value": 1}, {"id": "b", "value": 2}],
                   "steps": [{"id": "s1", "description": "add the values", "inputs": [{"ref": "a"}, {"ref": "b"}]}],
                   "answer_step": "s1"}'''

    def _parse_response(self, response: str) -> Decomposition:
        """Parse LLM response into Decomposition object."""
        # Clean up response (remove markdown code blocks if present)
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            # Find the end of code block
            end_idx = len(lines) - 1
            for i, line in enumerate(lines[1:], 1):
                if line.strip().startswith("```"):
                    end_idx = i
                    break
            response = "\n".join(lines[1:end_idx])

        try:
            data = json.loads(response)

            # Parse extractions
            extractions = []
            for ext in data.get("extractions", []):
                extractions.append(Extraction(
                    id=ext.get("id", ""),
                    value=float(ext.get("value", 0)),
                    span=ext.get("span", "")
                ))

            # Parse steps
            steps = []
            for step_data in data.get("steps", []):
                inputs = []
                for inp in step_data.get("inputs", []):
                    if isinstance(inp, dict):
                        if "ref" in inp:
                            inputs.append(StepInput(ref=inp["ref"]))
                        elif "value" in inp:
                            inputs.append(StepInput(value=float(inp["value"])))
                    elif isinstance(inp, (int, float)):
                        inputs.append(StepInput(value=float(inp)))

                step = DecomposedStep(
                    id=step_data.get("id", f"s{len(steps)+1}"),
                    description=step_data.get("description", ""),
                    inputs=inputs,
                    func_hint=step_data.get("func_hint"),
                )
                steps.append(step)

            return Decomposition(
                extractions=extractions,
                steps=steps,
                answer_step=data.get("answer_step", steps[-1].id if steps else "")
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response[:500]}")
            return Decomposition()
        except Exception as e:
            logger.error(f"Error parsing decomposition: {e}")
            return Decomposition()


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


def decompose_problem_full(problem: str, db=None, model: str = "gpt-4o-mini") -> Decomposition:
    """Decompose a problem and return the full Decomposition with extractions.

    Args:
        problem: The math problem text
        db: Optional StepSignatureDB instance (will create if not provided)
        model: LLM model to use

    Returns:
        Decomposition object with extractions, steps, and answer_step
    """
    if db is None:
        from mycelium.step_signatures.db import get_step_db
        db = get_step_db()

    menu = db.format_signature_menu(max_examples_per_func=10)
    decomposer = LLMDecomposer(model=model)
    return decomposer.decompose_full(problem, menu)
