"""Phase constraints for DAG execution ordering.

This module provides phase assignment and scoring for step execution.
Phases help ensure steps are executed in a coherent order that respects
the DAG structure.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhaseAssignment:
    """Phase assignments for steps in a DAG.

    Each step gets a phase value (0.0 to 1.0) based on its position
    in the DAG execution order.
    """
    step_phases: dict[str, float] = field(default_factory=dict)


@dataclass
class PhaseScore:
    """Score for how well execution respected phase ordering."""
    base_score: float = 1.0
    coherence: float = 1.0  # How coherent the execution was (0-1)
    penalties: list[str] = field(default_factory=list)


def assign_phases(plan) -> PhaseAssignment:
    """Assign phases to steps based on DAG structure.

    Args:
        plan: DAGPlan with steps and execution order

    Returns:
        PhaseAssignment mapping step IDs to phase values
    """
    assignment = PhaseAssignment()

    levels = plan.get_execution_order()
    if not levels:
        return assignment

    # Assign phases based on execution level
    total_levels = len(levels)
    for level_idx, level in enumerate(levels):
        phase = level_idx / max(total_levels - 1, 1)
        for step in level:
            assignment.step_phases[step.id] = phase

    return assignment


def compute_execution_score(
    base_score: float,
    assignment: PhaseAssignment,
    execution_phases: dict[str, float],
) -> PhaseScore:
    """Compute how well execution respected the phase ordering.

    Args:
        base_score: Starting score (typically 1.0)
        assignment: The phase assignments from planning
        execution_phases: The actual phases at execution time

    Returns:
        PhaseScore with coherence and any penalties
    """
    if not assignment.step_phases or not execution_phases:
        return PhaseScore(base_score=base_score, coherence=1.0)

    # Calculate coherence as average alignment between planned and actual
    deviations = []
    for step_id, planned_phase in assignment.step_phases.items():
        if step_id in execution_phases:
            actual_phase = execution_phases[step_id]
            deviation = abs(planned_phase - actual_phase)
            deviations.append(deviation)

    if not deviations:
        return PhaseScore(base_score=base_score, coherence=1.0)

    avg_deviation = sum(deviations) / len(deviations)
    coherence = max(0.0, 1.0 - avg_deviation)

    return PhaseScore(
        base_score=base_score,
        coherence=coherence,
    )


def infer_execution_phase(
    step_id: str,
    assignment: PhaseAssignment,
    completed_steps: set[str],
) -> float:
    """Infer the execution phase for a step based on progress.

    Args:
        step_id: ID of the step to execute
        assignment: Phase assignments from planning
        completed_steps: Set of already completed step IDs

    Returns:
        Inferred phase value (0.0 to 1.0)
    """
    if step_id in assignment.step_phases:
        return assignment.step_phases[step_id]

    # Fallback: estimate based on completion progress
    if not assignment.step_phases:
        return 0.5

    total_steps = len(assignment.step_phases)
    completed = len(completed_steps)

    return completed / max(total_steps, 1)
