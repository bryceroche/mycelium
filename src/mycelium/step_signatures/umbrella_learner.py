"""Umbrella Learner: Automatically decompose failing guidance signatures.

Key insight: guidance DSL type means "I don't know how to compute this"
When guidance signatures fail, decompose them into specialized children.

Flow:
1. Detect failing guidance signatures (low success rate)
2. Decompose the step description into sub-steps
3. Create child signatures with actual DSLs + NL interface
4. Promote parent to umbrella, link children

NL Interface: Each signature has clarifying_questions and param_descriptions
so it can communicate with the decomposer about what parameters it needs.
"""

import json
import logging
from typing import Optional

from mycelium.planner import Planner
from mycelium.step_signatures.db import StepSignatureDB
from mycelium.step_signatures.models import StepSignature
from mycelium.embedder import Embedder
from mycelium.client import get_client

logger = logging.getLogger(__name__)


# Prompt for generating NL interface for a signature
NL_INTERFACE_PROMPT = '''Given a mathematical step, generate the natural language interface that helps extract parameters.

Step: {step_task}

Generate a JSON response with:
1. clarifying_questions: Questions to ask to extract each parameter needed
2. param_descriptions: What each parameter means in plain English
3. params: List of parameter names (short, snake_case)

Example for "Calculate 25% of a total":
```json
{{
  "clarifying_questions": ["What is the percentage?", "What is the total amount?"],
  "param_descriptions": {{"percentage": "The percentage to calculate", "total": "The base amount to take percentage of"}},
  "params": ["percentage", "total"]
}}
```

Example for "Add two quantities together":
```json
{{
  "clarifying_questions": ["What is the first quantity?", "What is the second quantity?"],
  "param_descriptions": {{"quantity_a": "The first number to add", "quantity_b": "The second number to add"}},
  "params": ["quantity_a", "quantity_b"]
}}
```

Now generate for the step above. Respond with ONLY valid JSON:'''

# Thresholds for umbrella promotion
# Smart decomposition: give signatures 3 chances, decompose if mostly failing
MIN_USES_FOR_EVALUATION = 3  # Need 3 attempts before evaluating
MAX_SUCCESS_RATE_FOR_DECOMPOSITION = 0.5  # Decompose if failing more than succeeding
# Example: 2 failures out of 3 = 33% success → decompose
# Example: 20 successes + 2 failures = 91% success → keep


class UmbrellaLearner:
    """Learn umbrella structure from failing guidance signatures."""

    def __init__(self, db: StepSignatureDB = None, client=None):
        self.db = db or StepSignatureDB()
        self.planner = Planner()
        self.embedder = Embedder.get_instance()
        self._client = client  # Lazy load

    @property
    def client(self):
        """Lazy-load LLM client for NL interface generation."""
        if self._client is None:
            self._client = get_client()
        return self._client

    async def generate_nl_interface(self, step_task: str) -> dict:
        """Generate NL interface (clarifying_questions, param_descriptions) for a step.

        Args:
            step_task: The step description/task

        Returns:
            Dict with clarifying_questions, param_descriptions, params
        """
        prompt = NL_INTERFACE_PROMPT.format(step_task=step_task)

        try:
            messages = [
                {"role": "system", "content": "You are a JSON generator. You must respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ]

            # Use JSON response format for Groq (requires "JSON" in prompt)
            response = await self.client.generate(
                messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            # Extract JSON from response (should be clean JSON with response_format)
            nl_data = self._extract_json(response)
            if not nl_data:
                logger.warning("[umbrella] Failed to extract NL interface JSON for: %s", step_task[:50])
                return {"clarifying_questions": [], "param_descriptions": {}, "params": []}

            # Parse and validate
            try:
                data = json.loads(nl_data)
            except json.JSONDecodeError:
                logger.warning("[umbrella] Invalid JSON in NL interface response")
                return {"clarifying_questions": [], "param_descriptions": {}, "params": []}

            result = {
                "clarifying_questions": data.get("clarifying_questions", []),
                "param_descriptions": data.get("param_descriptions", {}),
                "params": data.get("params", []),
            }

            logger.debug(
                "[umbrella] Generated NL interface for '%s': %d questions, %d params",
                step_task[:30], len(result["clarifying_questions"]), len(result["params"])
            )

            return result

        except Exception as e:
            logger.error("[umbrella] NL interface generation failed: %s", e)
            return {"clarifying_questions": [], "param_descriptions": {}, "params": []}

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text response."""
        text = text.strip()

        # Fix: if response starts with quote (missing opening brace), add it
        if text.startswith('"') or text.startswith("'"):
            text = "{" + text
            # Find the matching closing brace or add one
            if not text.rstrip().endswith("}"):
                text = text.rstrip() + "}"

        # If it starts with {, try to parse directly
        if text.startswith("{"):
            depth = 0
            for i, c in enumerate(text):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[:i+1]

        # Try to find ```json blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try to find any { } block
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]

        return None

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
        # Check if already an umbrella WITH children - skip decomposition
        if signature.is_semantic_umbrella:
            existing_children = self.db.get_children(signature.id)
            if existing_children:
                logger.warning(
                    "[umbrella] Signature %d is already an umbrella with %d children",
                    signature.id, len(existing_children)
                )
                return []
            # Umbrella with no children (auto-demoted) - proceed with decomposition
            logger.info(
                "[umbrella] Signature %d is umbrella with NO children - decomposing",
                signature.id
            )

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
        # Strategy: prefer repointing to existing deeper sigs over creating new ones
        child_ids = []
        parent_depth = signature.depth if signature.depth else 0
        min_child_depth = parent_depth + 1

        for i, step in enumerate(plan.steps):
            # Skip synthesis/final steps
            if "final" in step.task.lower() or "combine" in step.task.lower():
                continue

            # Embed the step
            embedding = self.embedder.embed(step.task)

            # First: try to repoint to existing deeper signature
            child_sig = self.db.find_deeper_signature(
                embedding=embedding,
                min_depth=min_child_depth,
                min_similarity=0.75,  # Slightly lower threshold for repointing
                exclude_ids={signature.id},  # Don't repoint to self
            )
            is_new = False
            is_repoint = False

            if child_sig:
                is_repoint = True
                logger.info(
                    "[umbrella] Repointing to existing sig: '%s' (id=%d, depth=%d)",
                    child_sig.description[:40], child_sig.id, child_sig.depth
                )
            else:
                # Fall back: find or create new signature
                # Pass extracted_values from planner to enable DSL generation from structure
                child_sig, is_new = self.db.find_or_create(
                    step_text=step.task,
                    embedding=embedding,
                    min_similarity=0.85,
                    parent_problem=signature.description,
                    origin_depth=min_child_depth,  # Set proper depth for new sigs
                    extracted_values=getattr(step, 'extracted_values', None),
                )

            if is_new:
                logger.info(
                    "[umbrella] Created child: '%s' (type=%s, dsl=%s, depth=%d)",
                    step.task[:40], child_sig.step_type, child_sig.dsl_type, min_child_depth
                )

                # Generate NL interface for new child so it can communicate params
                try:
                    nl_interface = await self.generate_nl_interface(step.task)
                    if nl_interface["clarifying_questions"] or nl_interface["param_descriptions"]:
                        self.db.update_nl_interface(
                            signature_id=child_sig.id,
                            clarifying_questions=nl_interface["clarifying_questions"],
                            param_descriptions=nl_interface["param_descriptions"],
                        )
                        logger.info(
                            "[umbrella] Added NL interface to child %d: %d questions",
                            child_sig.id, len(nl_interface["clarifying_questions"])
                        )
                except Exception as e:
                    logger.warning("[umbrella] NL interface generation failed for child %d: %s", child_sig.id, e)

            # Add relationship (skip if child matches parent - prevents self-references)
            if child_sig.id == signature.id:
                logger.warning(
                    "[umbrella] Skipping self-reference: child '%s' matched parent signature %d",
                    step.task[:40], signature.id
                )
                continue

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


async def learn_umbrellas(db: StepSignatureDB = None, client=None) -> dict:
    """Convenience function to run umbrella learning."""
    learner = UmbrellaLearner(db, client=client)
    return await learner.learn_from_failures()
