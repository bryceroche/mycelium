"""
Executor for decomposition verification.

Runs the computation steps and verifies they produce the claimed results.
"""

from typing import Dict, Optional, Tuple
from .schema import Decomposition, Step, Ref, RefType
from ..function_registry import call_function, get_function_info, FUNCTION_REGISTRY


def execute_step(
    step: Step,
    extractions: Dict[str, float],
    step_results: Dict[str, float],
) -> Tuple[Optional[float], Optional[str]]:
    """
    Execute a single step using the function registry.

    Returns: (result, error_message)
    """
    # Resolve all input operands
    resolved_inputs = []
    for i, inp in enumerate(step.inputs):
        val = inp.resolve(extractions, step_results)
        if val is None:
            return None, f"Cannot resolve input {i}: {inp}"
        resolved_inputs.append(val)

    # Check function exists in registry
    if step.func not in FUNCTION_REGISTRY:
        return None, f"Unknown function: {step.func}"

    # Execute operation via function registry
    try:
        result = call_function(step.func, *resolved_inputs)
        return float(result), None
    except ZeroDivisionError:
        return None, "Division by zero"
    except Exception as e:
        return None, f"Execution error: {e}"


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
        # Resolve all inputs
        input_strs = []
        for inp in step.inputs:
            val = inp.resolve(extractions, step_results)
            input_strs.append(f"{inp.id}={val}" if val is not None else inp.id)

        inputs_display = ", ".join(input_strs)
        lines.append(
            f"  {step.id}: {step.func}({inputs_display}) = {step.result}"
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
