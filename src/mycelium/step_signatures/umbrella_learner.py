"""Umbrella Learner: Automatically decompose failing guidance signatures.

Key insight: guidance DSL type means "I don't know how to compute this"
When guidance signatures fail, decompose them into specialized children.

Flow:
1. Detect failing guidance signatures (low success rate)
2. Decompose the step description into sub-steps
3. Create child signatures with actual DSLs
4. Promote parent to umbrella, link children
"""

import logging
from typing import Optional

from mycelium.planner import Planner
from mycelium.step_signatures.db import StepSignatureDB
from mycelium.step_signatures.models import StepSignature
from mycelium.embedder import Embedder

logger = logging.getLogger(__name__)

# Thresholds for umbrella promotion
# Smart decomposition: give signatures 3 chances, decompose if mostly failing
MIN_USES_FOR_EVALUATION = 3  # Need 3 attempts before evaluating
MAX_SUCCESS_RATE_FOR_DECOMPOSITION = 0.5  # Decompose if failing more than succeeding
# Example: 2 failures out of 3 = 33% success → decompose
# Example: 20 successes + 2 failures = 91% success → keep


class UmbrellaLearner:
    """Learn umbrella structure from failing guidance signatures."""

    def __init__(self, db: StepSignatureDB = None):
        self.db = db or StepSignatureDB()
        self.planner = Planner()
        self.embedder = Embedder.get_instance()

    def get_decomposition_candidates(self) -> list[StepSignature]:
        """Find guidance signatures that should be decomposed.

        Criteria:
        - dsl_type = "decompose" (no actual computation)
        - uses >= MIN_USES_FOR_EVALUATION (enough data)
        - success_rate < MAX_SUCCESS_RATE_FOR_DECOMPOSITION (failing)
        - is_semantic_umbrella = False (not already decomposed)
        """
        all_sigs = self.db.get_all_signatures()

        candidates = []
        for sig in all_sigs:
            if (
                sig.dsl_type == "decompose"
                and sig.uses >= MIN_USES_FOR_EVALUATION
                and sig.success_rate <= MAX_SUCCESS_RATE_FOR_DECOMPOSITION
                and not sig.is_semantic_umbrella
            ):
                candidates.append(sig)
                logger.debug(
                    "[umbrella] Candidate: '%s' (uses=%d, success=%.1f%%)",
                    sig.step_type, sig.uses, sig.success_rate * 100
                )

        return candidates

    async def decompose_signature(self, signature: StepSignature) -> list[int]:
        """Decompose a guidance signature into child signatures.

        Args:
            signature: The failing guidance signature to decompose

        Returns:
            List of child signature IDs created
        """
        if signature.is_semantic_umbrella:
            logger.warning("[umbrella] Signature %d is already an umbrella", signature.id)
            return []

        # Use planner to decompose the step description
        problem = f"Break down this step into smaller sub-steps: {signature.description}"
        plan = await self.planner.decompose(problem)

        if len(plan.steps) <= 1:
            # Mark as atomic to prevent repeated decomposition attempts
            self.db.update_signature(
                signature.id,
                dsl_type="atomic",  # No longer "decompose" - won't be picked up again
            )
            logger.info(
                "[umbrella] Marked '%s' as atomic (cannot decompose further, got %d steps)",
                signature.step_type, len(plan.steps)
            )
            return []

        # Create child signatures from decomposition
        child_ids = []
        for i, step in enumerate(plan.steps):
            # Skip synthesis/final steps
            if "final" in step.task.lower() or "combine" in step.task.lower():
                continue

            # Embed and create child signature
            embedding = self.embedder.embed(step.task)
            child_sig, is_new = self.db.find_or_create(
                step_text=step.task,
                embedding=embedding,
                min_similarity=0.85,
                parent_problem=signature.description,
            )

            if is_new:
                logger.info(
                    "[umbrella] Created child: '%s' (type=%s, dsl=%s)",
                    step.task[:40], child_sig.step_type, child_sig.dsl_type
                )

            # Add relationship
            condition = step.task[:100]  # Use task as routing condition
            self.db.add_child(
                parent_id=signature.id,
                child_id=child_sig.id,
                condition=condition,
                routing_order=i,
            )
            child_ids.append(child_sig.id)

        if child_ids:
            # Promote to umbrella (already done by add_child, but be explicit)
            self.db.promote_to_umbrella(signature.id)
            logger.info(
                "[umbrella] Promoted '%s' to umbrella with %d children",
                signature.step_type, len(child_ids)
            )

        return child_ids

    async def learn_from_failures(self) -> dict:
        """Main entry point: find failing guidance sigs and decompose them.

        Returns:
            Dict with learning statistics
        """
        candidates = self.get_decomposition_candidates()

        if not candidates:
            logger.debug("[umbrella] No decomposition candidates found")
            return {"candidates": 0, "decomposed": 0, "children_created": 0}

        decomposed = 0
        total_children = 0

        for sig in candidates:
            try:
                child_ids = await self.decompose_signature(sig)
                if child_ids:
                    decomposed += 1
                    total_children += len(child_ids)
            except Exception as e:
                logger.warning(
                    "[umbrella] Failed to decompose '%s': %s",
                    sig.step_type, e
                )

        result = {
            "candidates": len(candidates),
            "decomposed": decomposed,
            "children_created": total_children,
        }
        logger.info("[umbrella] Learning complete: %s", result)
        return result


async def learn_umbrellas(db: StepSignatureDB = None) -> dict:
    """Convenience function to run umbrella learning."""
    learner = UmbrellaLearner(db)
    return await learner.learn_from_failures()
