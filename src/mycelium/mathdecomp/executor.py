"""
Executor for decomposition verification.

Runs the computation steps and verifies they produce the claimed results.
"""

from typing import Dict, Optional, Tuple
from .schema import Decomposition, Step, Ref, RefType, Operator


def execute_step(
    step: Step,
    extractions: Dict[str, float],
    step_results: Dict[str, float],
) -> Tuple[Optional[float], Optional[str]]:
    """
    Execute a single step.

    Returns: (result, error_message)
    """
    # Resolve left operand
    left_val = step.left.resolve(extractions, step_results)
    if left_val is None:
        return None, f"Cannot resolve left operand: {step.left}"

    # Resolve right operand
    right_val = step.right.resolve(extractions, step_results)
    if right_val is None:
        return None, f"Cannot resolve right operand: {step.right}"

    # Execute operation
    op = step.op
    if op == "+":
        return left_val + right_val, None
    elif op == "-":
        return left_val - right_val, None
    elif op == "*":
        return left_val * right_val, None
    elif op == "/":
        if right_val == 0:
            return None, "Division by zero"
        return left_val / right_val, None
    else:
        return None, f"Unknown operator: {op}"


def execute_decomposition(
    decomp: Decomposition,
    tolerance: float = 0.001,
) -> Tuple[Dict[str, float], Optional[str]]:
    """
    Execute all steps in a decomposition.

    Returns: (step_results, first_error)
    """
    # Build extraction lookup
    extractions = {e.id: e.value for e in decomp.extractions}

    # Get execution order
    order = decomp.dependency_order()

    # Execute steps in order
    step_results = {}
    for step_id in order:
        step = decomp.get_step(step_id)
        if step is None:
            return step_results, f"Step not found: {step_id}"

        result, error = execute_step(step, extractions, step_results)
        if error:
            return step_results, f"Step {step_id}: {error}"

        # Check against claimed result
        if abs(result - step.result) > tolerance:
            return step_results, (
                f"Step {step_id}: computed {result}, claimed {step.result}"
            )

        step_results[step_id] = result

    return step_results, None


def verify_decomposition(
    decomp: Decomposition,
    expected_answer: Optional[float] = None,
    tolerance: float = 0.001,
) -> Decomposition:
    """
    Verify a decomposition by executing it.

    Sets decomp.verified and decomp.error based on results.

    If expected_answer is provided, also checks final answer matches.
    """
    step_results, error = execute_decomposition(decomp, tolerance)

    if error:
        decomp.verified = False
        decomp.error = error
        return decomp

    # Check answer reference resolves correctly
    extractions = {e.id: e.value for e in decomp.extractions}
    final_value = decomp.answer_ref.resolve(extractions, step_results)

    if final_value is None:
        decomp.verified = False
        decomp.error = f"Cannot resolve answer ref: {decomp.answer_ref}"
        return decomp

    if abs(final_value - decomp.answer_value) > tolerance:
        decomp.verified = False
        decomp.error = (
            f"Answer mismatch: computed {final_value}, claimed {decomp.answer_value}"
        )
        return decomp

    # Check against expected answer if provided
    if expected_answer is not None:
        if abs(final_value - expected_answer) > tolerance:
            decomp.verified = False
            decomp.error = (
                f"Wrong answer: got {final_value}, expected {expected_answer}"
            )
            return decomp

    decomp.verified = True
    decomp.error = None
    return decomp


def trace_execution(decomp: Decomposition) -> str:
    """
    Generate a human-readable trace of execution.
    """
    lines = []
    lines.append(f"Problem: {decomp.problem}")
    lines.append("")
    lines.append("Extractions:")
    for e in decomp.extractions:
        lines.append(f"  {e.id} = {e.value}  (from: \"{e.span}\")")

    lines.append("")
    lines.append("Steps:")

    extractions = {e.id: e.value for e in decomp.extractions}
    step_results = {}

    for step in decomp.steps:
        left_val = step.left.resolve(extractions, step_results)
        right_val = step.right.resolve(extractions, step_results)

        left_str = f"{step.left.id}={left_val}" if left_val else step.left.id
        right_str = f"{step.right.id}={right_val}" if right_val else step.right.id

        lines.append(
            f"  {step.id}: {left_str} {step.op} {right_str} = {step.result}"
            f"  [{step.semantic}]"
        )

        if step.result is not None:
            step_results[step.id] = step.result

    lines.append("")
    lines.append(f"Answer: {decomp.answer_ref.id} = {decomp.answer_value}")
    lines.append(f"Verified: {decomp.verified}")
    if decomp.error:
        lines.append(f"Error: {decomp.error}")

    return "\n".join(lines)
