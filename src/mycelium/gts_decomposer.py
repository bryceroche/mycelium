"""GTSDecomposer - Wrapper for GTS model to decompose math problems to atomic steps.

Per CLAUDE.md New Favorite Pattern: consolidate decomposition to single entry point.
Per CLAUDE.md Big 5 #4 (True Atomic Decomposition): GTS decomposes problems into atomic steps
during cold start for routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from mycelium.expression_tree import ExprNode, parse_prefix, decompose_to_atomic
from mycelium.gts_model import load_gts_model, GTSModel


@dataclass
class DecomposedStep:
    """Represents a single atomic step from problem decomposition.

    Attributes:
        step_number: Sequential step number (1, 2, 3...)
        operation: Human-readable operation description for graph embedding
                   (e.g., 'add two numbers', 'subtract two numbers')
        expr_tree: The atomic ExprNode tree for this step
        extracted_values: Mapping of NUM_X variables to their float values
        depends_on: List of step numbers this step depends on
    """
    step_number: int
    operation: str
    expr_tree: ExprNode
    extracted_values: dict[str, float]
    depends_on: list[int]


class GTSDecomposer:
    """Wrapper for GTS model to decompose math problems to atomic steps.

    Per CLAUDE.md New Favorite Pattern: consolidate decomposition to single entry point.

    This class provides two main entry points:
    1. decompose() - Full pipeline from problem text (requires beam search, not yet implemented)
    2. decompose_from_prefix() - From prefix expression (for testing without full inference)

    Example:
        decomposer = GTSDecomposer()

        # Using decompose_from_prefix for testing
        steps = decomposer.decompose_from_prefix(
            "- + NUM_0 NUM_1 NUM_2",
            {"NUM_0": 5.0, "NUM_1": 3.0, "NUM_2": 2.0}
        )

        for step in steps:
            print(f"Step {step.step_number}: {step.operation}")
            print(f"  Values: {step.extracted_values}")
            print(f"  Depends on: {step.depends_on}")
    """

    def __init__(self, model_path: str = "trained_model/GTS-mawps"):
        """Load GTS model and vocabularies.

        Args:
            model_path: Path to the trained GTS model directory.
        """
        self.model: Optional[GTSModel] = None
        self._model_path = model_path

    def _ensure_model_loaded(self) -> GTSModel:
        """Lazy load the model when first needed."""
        if self.model is None:
            self.model = load_gts_model(self._model_path)
        return self.model

    def decompose(self, problem_text: str) -> list[DecomposedStep]:
        """Main entry point - decompose problem to atomic steps.

        Args:
            problem_text: The math word problem text.

        Returns:
            List of DecomposedStep objects representing atomic operations.

        Raises:
            NotImplementedError: GTS beam search decoding not yet implemented.
        """
        # 1. Extract numbers and normalize
        numbers, normalized = self._extract_numbers(problem_text)

        # 2. Get prefix expression (NOTE: full inference not yet implemented)
        prefix = self._run_inference(normalized)

        # 3. Parse to expression tree
        tree = parse_prefix(prefix)

        # 4. Decompose to atomic steps
        atomic_steps = decompose_to_atomic(tree)

        # 5. Convert to DecomposedStep objects
        return self._build_decomposed_steps(atomic_steps, numbers)

    def decompose_from_prefix(
        self,
        prefix_str: str,
        number_mapping: dict[str, float]
    ) -> list[DecomposedStep]:
        """Decompose directly from a prefix expression (for testing without full inference).

        This method bypasses the GTS model inference and works directly from
        a known prefix expression, making it useful for testing the decomposition
        logic without requiring the full beam search implementation.

        Args:
            prefix_str: Prefix notation string (e.g., "- + NUM_0 NUM_1 NUM_2")
            number_mapping: Mapping of NUM_X variables to float values
                           (e.g., {"NUM_0": 5.0, "NUM_1": 3.0, "NUM_2": 2.0})

        Returns:
            List of DecomposedStep objects representing atomic operations.

        Example:
            steps = decomposer.decompose_from_prefix(
                "* + NUM_0 NUM_1 NUM_2",
                {"NUM_0": 10.0, "NUM_1": 5.0, "NUM_2": 3.0}
            )
            # Returns 2 steps: add(NUM_0, NUM_1) then multiply(step_1, NUM_2)
        """
        tree = parse_prefix(prefix_str)
        atomic_steps = decompose_to_atomic(tree)
        return self._build_decomposed_steps(atomic_steps, number_mapping)

    def _extract_numbers(self, text: str) -> tuple[dict[str, float], str]:
        """Extract numbers and replace with NUM_0, NUM_1, etc.

        Args:
            text: Problem text containing numbers.

        Returns:
            Tuple of (number_mapping, normalized_text) where:
            - number_mapping: Dict mapping NUM_X to float values
            - normalized_text: Text with numbers replaced by "NUM"
        """
        numbers: dict[str, float] = {}
        counter = [0]  # List for mutability in nested function

        def replace(match: re.Match) -> str:
            num_str = match.group()
            # Handle both integers and floats
            num = float(num_str)
            key = f"NUM_{counter[0]}"
            numbers[key] = num
            counter[0] += 1
            return "NUM"

        # Match integers and floats (including negative numbers)
        normalized = re.sub(r"-?\d+\.?\d*", replace, text)
        return numbers, normalized

    def _run_inference(self, normalized_text: str) -> str:
        """Run GTS model inference.

        Args:
            normalized_text: Text with numbers replaced by NUM tokens.

        Returns:
            Prefix notation expression string.

        Raises:
            ValueError: If model returns empty or invalid output.
        """
        model = self._ensure_model_loaded()

        # Run beam search decoding
        prefix = model.generate(normalized_text)

        if not prefix or prefix == "<decoding_not_implemented>":
            raise ValueError(
                f"GTS model returned invalid output: {prefix!r}"
            )

        return prefix

    def _build_decomposed_steps(
        self,
        atomic_steps: list[tuple[str, ExprNode, dict]],
        number_mapping: dict[str, float]
    ) -> list[DecomposedStep]:
        """Convert atomic steps to DecomposedStep objects.

        Args:
            atomic_steps: List of (step_name, tree, metadata) from decompose_to_atomic()
            number_mapping: Mapping of NUM_X to float values

        Returns:
            List of DecomposedStep objects with dependencies and extracted values.
        """
        result: list[DecomposedStep] = []

        for step_name, tree, metadata in atomic_steps:
            # Extract step number from "step_N" format
            step_num = int(step_name.split("_")[1])

            # Find dependencies (references to previous steps)
            depends: list[int] = []
            left_operand = metadata.get("left_operand")
            right_operand = metadata.get("right_operand")

            if left_operand and left_operand.startswith("step_"):
                dep_num = int(left_operand.split("_")[1])
                depends.append(dep_num)
            if right_operand and right_operand.startswith("step_"):
                dep_num = int(right_operand.split("_")[1])
                depends.append(dep_num)

            # Extract values used in this step (only NUM_X variables, not step refs)
            extracted: dict[str, float] = {}
            for operand in [left_operand, right_operand]:
                if operand and operand in number_mapping:
                    extracted[operand] = number_mapping[operand]

            # Build operation string for graph embedding
            operation = metadata.get("operation")
            if operation:
                operation_str = f"{operation} two numbers"
            else:
                # Handle leaf nodes (shouldn't normally happen after decomposition)
                operation_str = "unknown"

            result.append(DecomposedStep(
                step_number=step_num,
                operation=operation_str,
                expr_tree=tree,
                extracted_values=extracted,
                depends_on=sorted(depends),
            ))

        return result
