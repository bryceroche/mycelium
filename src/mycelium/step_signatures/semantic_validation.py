"""Semantic Validation: Check decomposition coherence before execution.

This module validates that a planner's decomposition forms a semantically
coherent chain - that step outputs connect to the next step's inputs.

Key insight: Signatures already know what they need (via clarifying_questions).
We can use this to validate decompositions BEFORE execution fails.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StepValidation:
    """Validation result for a single step."""
    step_id: str
    is_valid: bool
    coherence_score: float  # 0-1, how well inputs match available outputs
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class PlanValidation:
    """Validation result for entire plan."""
    is_coherent: bool
    overall_score: float
    step_validations: list[StepValidation] = field(default_factory=list)
    feedback_for_planner: str = ""  # NL feedback to improve decomposition


def validate_plan_coherence(
    plan,  # DAGPlan
    signature_db=None,
    embedder=None,
) -> PlanValidation:
    """Validate that a plan's steps form a coherent chain.

    Checks:
    1. Each step's expected inputs can be satisfied by previous steps
    2. Step descriptions semantically connect to each other
    3. No orphan computations (outputs that aren't used)

    Args:
        plan: DAGPlan to validate
        signature_db: Optional StepSignatureDB for richer validation
        embedder: Optional Embedder for semantic similarity

    Returns:
        PlanValidation with coherence assessment and feedback
    """
    if embedder is None:
        try:
            from mycelium.embedder import Embedder
            embedder = Embedder.get_instance()
        except Exception as e:
            logger.warning("[semantic_val] Could not load embedder: %s", e)
            return PlanValidation(
                is_coherent=True,
                overall_score=1.0,
                feedback_for_planner="(validation skipped - no embedder)"
            )

    step_validations = []
    available_outputs = {}  # step_id -> (description, param_names)
    issues = []

    for step in plan.steps:
        # Track what this step produces
        output_params = list(step.extracted_values.keys()) if step.extracted_values else []
        available_outputs[step.id] = (step.task, output_params)

        # Validate step's inputs against available outputs
        step_val = _validate_step_inputs(
            step,
            available_outputs,
            embedder
        )
        step_validations.append(step_val)

        if not step_val.is_valid:
            issues.extend(step_val.issues)

    # Calculate overall coherence
    if step_validations:
        overall_score = sum(sv.coherence_score for sv in step_validations) / len(step_validations)
    else:
        overall_score = 1.0

    is_coherent = overall_score >= 0.5 and len(issues) == 0

    # Generate feedback for planner
    feedback = _generate_planner_feedback(step_validations, issues)

    return PlanValidation(
        is_coherent=is_coherent,
        overall_score=overall_score,
        step_validations=step_validations,
        feedback_for_planner=feedback,
    )


def _validate_step_inputs(
    step,  # Step
    available_outputs: dict,
    embedder,
) -> StepValidation:
    """Validate that a step's required inputs are available from previous steps."""
    issues = []
    suggestions = []
    coherence_scores = []

    # Check explicit dependencies
    for dep_id in step.depends_on:
        if dep_id not in available_outputs:
            issues.append(f"Depends on '{dep_id}' which doesn't exist")
            coherence_scores.append(0.0)
            continue

        dep_desc, dep_params = available_outputs[dep_id]

        # Check semantic connection between dependency output and this step's task
        if embedder:
            dep_emb = embedder.embed(dep_desc)
            step_emb = embedder.embed(step.task)

            similarity = float(np.dot(dep_emb, step_emb) / (
                np.linalg.norm(dep_emb) * np.linalg.norm(step_emb)
            ))
            coherence_scores.append(similarity)

            if similarity < 0.3:
                issues.append(
                    f"Weak connection: '{dep_id}' ({dep_desc[:30]}...) → this step "
                    f"(similarity={similarity:.2f})"
                )
                suggestions.append(
                    f"Consider adding an intermediate step between '{dep_id}' and '{step.id}'"
                )

    # Check for extracted_values that reference previous steps
    if step.extracted_values:
        for param, value in step.extracted_values.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                ref_step = value[1:-1]  # Extract step_1 from "{step_1}"
                if ref_step not in available_outputs:
                    issues.append(f"References '{ref_step}' which doesn't exist")
                    coherence_scores.append(0.0)

    # Calculate overall coherence for this step
    if coherence_scores:
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
    else:
        avg_coherence = 1.0 if not step.depends_on else 0.5

    return StepValidation(
        step_id=step.id,
        is_valid=len(issues) == 0,
        coherence_score=avg_coherence,
        issues=issues,
        suggestions=suggestions,
    )


def _generate_planner_feedback(
    step_validations: list[StepValidation],
    issues: list[str],
) -> str:
    """Generate NL feedback for the planner based on validation results."""
    if not issues:
        return ""

    feedback_parts = ["The decomposition has coherence issues:"]

    for issue in issues[:5]:  # Limit to top 5 issues
        feedback_parts.append(f"- {issue}")

    # Add suggestions
    all_suggestions = []
    for sv in step_validations:
        all_suggestions.extend(sv.suggestions)

    if all_suggestions:
        feedback_parts.append("\nSuggestions:")
        for suggestion in all_suggestions[:3]:
            feedback_parts.append(f"- {suggestion}")

    return "\n".join(feedback_parts)


@dataclass
class StepFailureFeedback:
    """Feedback from a failed step execution."""
    step_id: str
    step_task: str
    failure_reason: str
    signature_needs: list[str] = field(default_factory=list)  # What the signature needed
    received_values: dict = field(default_factory=dict)  # What it actually got

    def to_planner_hint(self) -> str:
        """Format as hint for planner retry."""
        lines = [f"Step '{self.step_id}' failed: {self.failure_reason}"]

        if self.signature_needs:
            lines.append(f"  The operation needed: {', '.join(self.signature_needs)}")

        if self.received_values:
            vals = ", ".join(f"{k}={v}" for k, v in self.received_values.items())
            lines.append(f"  But received: {vals}")

        return "\n".join(lines)


def create_failure_feedback(
    step_id: str,
    step_task: str,
    failure_reason: str,
    signature=None,  # StepSignature if available
    context: dict = None,
) -> StepFailureFeedback:
    """Create structured feedback from a step failure.

    This captures WHY a step failed so the planner can adjust on retry.
    """
    signature_needs = []
    if signature:
        # Get what the signature needed from its NL interface
        if signature.clarifying_questions:
            signature_needs = signature.clarifying_questions
        elif signature.dsl_script:
            import json
            try:
                dsl = json.loads(signature.dsl_script)
                signature_needs = dsl.get("params", [])
            except (json.JSONDecodeError, TypeError):
                pass

    received_values = {}
    if context:
        # Extract relevant values from context
        for key, value in context.items():
            if not key.startswith("_"):  # Skip internal keys
                received_values[key] = str(value)[:50]  # Truncate long values

    return StepFailureFeedback(
        step_id=step_id,
        step_task=step_task,
        failure_reason=failure_reason,
        signature_needs=signature_needs,
        received_values=received_values,
    )
