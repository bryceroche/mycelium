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

import numpy as np

from mycelium.config import (
    UMBRELLA_MIN_USES_FOR_EVALUATION,
    UMBRELLA_MAX_SUCCESS_RATE_FOR_DECOMPOSITION,
    SYNTHESIS_STEP_ANCHORS,
    SYNTHESIS_STEP_THRESHOLD,
)
from mycelium.planner import Planner
from mycelium.step_signatures.db import StepSignatureDB
from mycelium.step_signatures.models import StepSignature
from mycelium.step_signatures.utils import cosine_similarity
from mycelium.embedder import Embedder
from mycelium.embedding_cache import cached_embed, cached_embed_batch
from mycelium.client import get_client

logger = logging.getLogger(__name__)

# Cache for synthesis step anchor embeddings (lazy-loaded)
_synthesis_anchor_embeddings: Optional[list[np.ndarray]] = None


def _get_synthesis_anchor_embeddings(embedder) -> list[np.ndarray]:
    """Get or compute cached embeddings for synthesis step anchors."""
    global _synthesis_anchor_embeddings
    if _synthesis_anchor_embeddings is None:
        # Batch embed all anchors (returns dict: text -> embedding)
        embeddings_dict = cached_embed_batch(SYNTHESIS_STEP_ANCHORS, embedder)
        # Extract embeddings in same order as anchors
        _synthesis_anchor_embeddings = [
            embeddings_dict[anchor] for anchor in SYNTHESIS_STEP_ANCHORS
            if anchor in embeddings_dict
        ]
        logger.debug(
            "[umbrella] Computed %d synthesis anchor embeddings",
            len(_synthesis_anchor_embeddings)
        )
    return _synthesis_anchor_embeddings


def is_synthesis_step(step_embedding: np.ndarray, embedder, step_text: str = "") -> bool:
    """Check if a step is a synthesis/aggregation step.

    Uses embedding similarity (preferred) with keyword fallback for robustness.
    Synthesis steps combine results from previous steps and should be skipped
    when creating umbrella children (they don't add computational value).

    Args:
        step_embedding: The step's embedding vector
        embedder: Embedder instance for getting anchor embeddings
        step_text: The step text (for keyword fallback)

    Returns:
        True if step appears to be a synthesis step
    """
    # First try embedding-based detection (preferred)
    if isinstance(step_embedding, np.ndarray):
        try:
            anchor_embeddings = _get_synthesis_anchor_embeddings(embedder)

            # Validate anchor embeddings are real list of arrays
            if isinstance(anchor_embeddings, list) and len(anchor_embeddings) > 0:
                if isinstance(anchor_embeddings[0], np.ndarray):
                    for anchor_emb in anchor_embeddings:
                        similarity = cosine_similarity(step_embedding, anchor_emb)
                        if similarity >= SYNTHESIS_STEP_THRESHOLD:
                            return True

        except (TypeError, ValueError, KeyError, IndexError):
            # Fall through to keyword fallback
            pass

    # Keyword fallback (for tests/mocked embedder/robustness)
    # These are synthesis indicators that should be rare in actual computation steps
    if step_text:
        text_lower = step_text.lower()
        synthesis_keywords = ["final", "combine", "aggregate", "sum up", "put together"]
        for keyword in synthesis_keywords:
            if keyword in text_lower:
                return True

    return False


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

# Thresholds imported from config:
# - UMBRELLA_MIN_USES_FOR_EVALUATION: Need N attempts before evaluating
# - UMBRELLA_MAX_SUCCESS_RATE_FOR_DECOMPOSITION: Decompose if failing more than succeeding


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

    async def batch_generate_nl_interfaces(self, step_tasks: list[str]) -> dict[str, dict]:
        """Generate NL interfaces for multiple steps in ONE LLM call.

        This replaces N individual generate_nl_interface() calls with a single
        batch call using JSON mode.

        Args:
            step_tasks: List of step descriptions/tasks

        Returns:
            Dict mapping step_task -> {clarifying_questions, param_descriptions, params}
        """
        if not step_tasks:
            return {}

        # Build the batch prompt
        steps_text = []
        for i, task in enumerate(step_tasks, 1):
            steps_text.append(f"{i}. {task}")

        prompt = f"""Generate natural language interfaces for these mathematical steps.
For each step, provide:
- clarifying_questions: Questions to extract each parameter needed
- param_descriptions: What each parameter means in plain English
- params: List of parameter names (short, snake_case)

Steps:
{chr(10).join(steps_text)}

Output valid JSON with this exact structure:
{{
  "interfaces": [
    {{
      "step_index": 1,
      "clarifying_questions": ["What is X?", "What is Y?"],
      "param_descriptions": {{"x": "description", "y": "description"}},
      "params": ["x", "y"]
    }},
    ...
  ]
}}

Rules:
- Generate one interface entry for each step (in order)
- Keep parameter names short and snake_case
- Questions should be clear and specific to the step
- Output ONLY the JSON, no explanation"""

        try:
            messages = [
                {"role": "system", "content": "You are a JSON generator. You must respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ]

            response = await self.client.generate(
                messages,
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            data = json.loads(response.strip())
            interfaces = data.get("interfaces", [])

            # Map back to step tasks
            result = {}
            for i, task in enumerate(step_tasks):
                if i < len(interfaces):
                    iface = interfaces[i]
                    result[task] = {
                        "clarifying_questions": iface.get("clarifying_questions", []),
                        "param_descriptions": iface.get("param_descriptions", {}),
                        "params": iface.get("params", []),
                    }
                else:
                    result[task] = {"clarifying_questions": [], "param_descriptions": {}, "params": []}

            logger.info(
                "[umbrella] Batch generated %d/%d NL interfaces in single LLM call",
                len([r for r in result.values() if r["clarifying_questions"]]), len(step_tasks)
            )
            return result

        except Exception as e:
            logger.warning("[umbrella] Batch NL interface generation failed: %s", e)
            # Return empty interfaces for all tasks
            return {task: {"clarifying_questions": [], "param_descriptions": {}, "params": []} for task in step_tasks}

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

        # Log the failure with a preview of the text
        logger.warning(
            "[umbrella] Failed to extract JSON from response (len=%d): %s",
            len(text), text[:200] if text else "(empty)"
        )
        return None

    def get_decomposition_candidates(self) -> list[StepSignature]:
        """Find signatures that should be decomposed.

        Two categories:
        1. Failing guidance signatures (dsl_type="decompose", not yet umbrella)
        2. Auto-demoted router umbrellas with NO children (need actual decomposition)

        Criteria for both:
        - uses >= UMBRELLA_MIN_USES_FOR_EVALUATION (enough data)
        - operational_failures > 0 (flagged by MCTS post-mortem)
        - MCTS win_rate <= max_success_rate (actually failing per ground truth)

        Per CLAUDE.md: "Do not decompose a leaf node until instructed by the
        MCTS rollout post-mortem analysis." Decomposition is triggered by
        destructive interference patterns, not by low success rate alone.

        The split threshold is adaptive based on global accuracy:
        - Low accuracy (cold start): lenient threshold (tolerate more failures)
        - High accuracy (mature): strict threshold (split confidently)

        NOTE: Uses MCTS win rates (ground truth) instead of signature.success_rate
        (which includes partial credit). This ensures we only decompose signatures
        that are actually failing operationally.
        """
        from mycelium.mcts.adaptive import AdaptiveExploration
        from mycelium.data_layer.mcts import get_mcts_win_rates

        all_sigs = self.db.get_all_signatures()

        # Get adaptive split threshold (failure rate)
        adaptive = AdaptiveExploration.get_instance()
        split_threshold = adaptive.split_threshold
        # Convert failure threshold to max success rate
        max_success_rate = 1.0 - split_threshold

        # Adaptive min_uses: lower during cold start, higher when mature
        # This ensures we evaluate failing signatures quickly during early learning
        from mycelium.config import DECOMP_MIN_ATTEMPTS_COLD, DECOMP_MIN_ATTEMPTS_MATURE
        from mycelium.mcts.adaptive import AdaptiveExploration
        adaptive = AdaptiveExploration.get_instance()
        accuracy = adaptive.global_accuracy
        adaptive_min_uses = int(
            DECOMP_MIN_ATTEMPTS_COLD + accuracy * (DECOMP_MIN_ATTEMPTS_MATURE - DECOMP_MIN_ATTEMPTS_COLD)
        )
        adaptive_min_uses = max(1, adaptive_min_uses)

        # Get MCTS win rates for ground truth filtering
        # This is the actual step-level win rate, not partial credit
        mcts_stats = get_mcts_win_rates(min_attempts=adaptive_min_uses)

        candidates = []
        for sig in all_sigs:
            # Skip if not enough uses (adaptive threshold)
            if sig.uses < adaptive_min_uses:
                continue
            # CRITICAL: Per CLAUDE.md, only decompose when flagged by MCTS post-mortem
            # operational_failures > 0 means destructive interference was detected
            if sig.operational_failures <= 0:
                continue

            # Use MCTS win rate (ground truth) instead of sig.success_rate (partial credit)
            # Fall back to sig.success_rate if no MCTS data (cold start)
            mcts_data = mcts_stats.get(sig.id)
            if mcts_data:
                actual_win_rate = mcts_data["win_rate"]
            else:
                # No MCTS data - use signature success rate as fallback
                actual_win_rate = sig.success_rate

            # Skip if win rate is acceptable
            if actual_win_rate > max_success_rate:
                continue

            # Category 1: decompose type not yet promoted to umbrella
            is_decompose_candidate = (
                sig.dsl_type == "decompose"
                and not sig.is_semantic_umbrella
            )

            # Category 2: auto-demoted router umbrellas without children
            # NOTE: We DON'T include orphan umbrellas for abstract decomposition here.
            # Abstract decomposition (decomposing generic description like "compute_product")
            # creates children with placeholder variables (X, Y) that can't execute.
            # Orphan umbrellas get children through CONCRETE problem solving, when actual
            # values route through them and create new leaf signatures.
            # is_orphan_umbrella logic removed - let them stay as orphans until concrete use.

            # Category 3: high-variance leaves flagged for decomposition
            # NOTE: Also not included - abstract decomposition creates broken umbrellas.
            # High-variance leaves should be decomposed during CONCRETE problem solving.

            if is_decompose_candidate:
                candidates.append(sig)
                logger.info(
                    "[umbrella] Candidate: '%s' (id=%d, reason=decompose_type, mcts_win=%.1f%%, op_fail=%d)",
                    sig.step_type, sig.id, actual_win_rate * 100, sig.operational_failures
                )

        return candidates

    def _get_failing_step_descriptions(self, node_id: int, limit: int = 5) -> list[str]:
        """Get specific step descriptions that failed with this node.

        Queries mcts_thread_steps to find actual step descriptions that
        this node failed on. These are more specific than the generic
        signature step_type and may be decomposable.

        Args:
            node_id: The signature ID that's failing
            limit: Maximum number of step descriptions to return

        Returns:
            List of unique step descriptions that failed with this node
        """
        from mycelium.data_layer import get_db

        try:
            db = get_db()
            with db.connection() as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT s.step_desc
                    FROM mcts_thread_steps t
                    JOIN mcts_dag_steps s ON t.dag_step_id = s.dag_step_id
                    WHERE t.node_id = ?
                      AND t.step_success = 0
                    ORDER BY t.created_at DESC
                    LIMIT ?
                """, (node_id, limit))

                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            # Table may not exist in test environment
            logger.debug("[umbrella] Could not query failing steps: %s", e)
            return []

    async def decompose_signature(self, signature: StepSignature) -> list[int]:
        """Decompose a guidance signature into child signatures.

        Args:
            signature: The failing guidance signature to decompose

        Returns:
            List of child signature IDs created
        """
        # Check if already an umbrella WITH children - skip decomposition
        # Skip cache to ensure fresh data in multiprocess environments
        if signature.is_semantic_umbrella:
            existing_children = self.db.get_children(signature.id, for_routing=True, skip_cache=True)
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
        # skip_validation=True because we're creating templates, not concrete plans
        # The umbrella children are generic operation patterns, not specific calculations
        problem = f"Break down this step into smaller sub-steps: {signature.description}"
        plan = await self.planner.decompose(problem, skip_validation=True)

        if len(plan.steps) <= 1:
            # Signature description is already atomic - try decomposing actual failing steps
            # Get specific step descriptions that failed with this node
            failing_steps = self._get_failing_step_descriptions(signature.id)
            if failing_steps:
                logger.info(
                    "[umbrella] Trying to decompose %d failing step descriptions for sig %d",
                    len(failing_steps), signature.id
                )
                # Try decomposing each failing step
                for step_desc in failing_steps[:3]:  # Limit to 3 to avoid explosion
                    step_problem = f"Break down this math step into simpler sub-steps: {step_desc}"
                    step_plan = await self.planner.decompose(step_problem, skip_validation=True)
                    if len(step_plan.steps) > 1:
                        logger.info(
                            "[umbrella] Decomposed failing step '%s' into %d sub-steps",
                            step_desc[:40], len(step_plan.steps)
                        )
                        plan = step_plan  # Use this successful decomposition
                        break
                else:
                    logger.info(
                        "[umbrella] Cannot decompose '%s' or its failing steps - keeping as decompose",
                        signature.step_type
                    )
                    return []
            else:
                logger.info(
                    "[umbrella] Cannot decompose '%s' further (got %d steps) - keeping as decompose",
                    signature.step_type, len(plan.steps)
                )
                return []

        # Create child signatures from decomposition
        # Strategy: prefer repointing to existing deeper sigs over creating new ones
        child_ids = []
        parent_depth = signature.depth if signature.depth else 0
        min_child_depth = parent_depth + 1

        for i, step in enumerate(plan.steps):
            # Embed the step first (needed for synthesis check and routing)
            embedding = cached_embed(step.task, self.embedder)

            # Skip synthesis/aggregation steps (embedding-based with keyword fallback)
            if is_synthesis_step(embedding, self.embedder, step.task):
                logger.debug("[umbrella] Skipping synthesis step: '%s'", step.task[:40])
                continue

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
                # Pass extracted_values and dsl_hint from planner for bidirectional LLM-signature communication
                # CRITICAL: Pass parent_id so new signatures are created under THIS signature, not root!
                # CRITICAL: Pass exclude_ids to prevent child matching back to parent (circular routing)
                child_sig, is_new = await self.db.find_or_create_async(
                    step_text=step.task,
                    embedding=embedding,
                    min_similarity=0.85,
                    parent_problem=signature.description,
                    origin_depth=min_child_depth,  # Set proper depth for new sigs
                    extracted_values=getattr(step, 'extracted_values', None),
                    dsl_hint=getattr(step, 'dsl_hint', None),  # LLM → signature communication
                    parent_id=signature.id,  # Ensure children are created under decomposing signature
                    exclude_ids={signature.id},  # Prevent child from routing back to parent
                )

            # Skip if child_sig is None (step was queued for decomposition, not created)
            if child_sig is None:
                logger.info(
                    "[umbrella] Skipping child (queued for decomposition): '%s'",
                    step.task[:40]
                )
                continue

            # If signature already has a DIFFERENT parent (matched OR repointed), create a NEW one instead
            # This ensures tree structure can grow deeper
            # NOTE: This check is OUTSIDE the else block so it also handles repointed signatures!
            # NOTE: If already a child of THIS signature, reuse it (don't create duplicate)
            if not is_new:
                existing_parent = self.db.get_parent(child_sig.id)
                if existing_parent is not None and existing_parent.id != signature.id:
                    logger.info(
                        "[umbrella] Sig %d already has parent %d, creating new for parent %d (repoint=%s)",
                        child_sig.id, existing_parent.id, signature.id, is_repoint
                    )
                    # Force create a new signature with THIS signature as parent
                    child_sig = self.db.create_signature(
                        step_text=step.task,
                        embedding=embedding,
                        parent_problem=signature.description,
                        origin_depth=min_child_depth,
                        extracted_values=getattr(step, 'extracted_values', None),
                        dsl_hint=getattr(step, 'dsl_hint', None),
                        parent_id=signature.id,  # Set parent to decomposing signature
                    )
                    is_new = True

            if is_new:
                logger.info(
                    "[umbrella] Created child: '%s' (type=%s, dsl=%s, depth=%d)",
                    step.task[:40], child_sig.step_type, child_sig.dsl_type, min_child_depth
                )

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
            child_ids.append((child_sig.id, step.task, is_new, child_sig.dsl_type))

        # BATCH NL Interface Generation: Generate all NL interfaces in ONE LLM call
        # Collect step tasks for new non-router children
        nl_tasks = [(sig_id, task) for sig_id, task, is_new, dsl_type in child_ids
                    if is_new and dsl_type != "router"]

        if nl_tasks:
            step_tasks = [task for _, task in nl_tasks]
            try:
                nl_interfaces = await self.batch_generate_nl_interfaces(step_tasks)

                # Update each signature with its NL interface
                for sig_id, task in nl_tasks:
                    nl_interface = nl_interfaces.get(task, {})
                    if nl_interface.get("clarifying_questions") or nl_interface.get("param_descriptions"):
                        self.db.update_nl_interface(
                            signature_id=sig_id,
                            clarifying_questions=nl_interface.get("clarifying_questions", []),
                            param_descriptions=nl_interface.get("param_descriptions", {}),
                        )
                        logger.debug(
                            "[umbrella] Added NL interface to child %d: %d questions",
                            sig_id, len(nl_interface.get("clarifying_questions", []))
                        )
            except Exception as e:
                logger.warning("[umbrella] Batch NL interface generation failed: %s", e)

        # Extract just the signature IDs for return value
        child_ids = [sig_id for sig_id, _, _, _ in child_ids]

        if len(child_ids) == 1:
            # Single child = pointless router, mark parent as atomic instead
            # Per CLAUDE.md: "healthy tree would have ~5:1 ratio where each router has roughly five children"
            # A router with 1 child is just a chain, not meaningful branching
            from mycelium.data_layer.mcts import mark_signature_atomic
            logger.info(
                "[umbrella] Decomposition produced only 1 child for '%s' - marking as atomic (no chain)",
                signature.step_type
            )
            # Remove the single child relationship we just added
            self.db.remove_child(parent_id=signature.id, child_id=child_ids[0])
            mark_signature_atomic(signature.id, "single_child_decomp")
            return []  # No meaningful decomposition occurred
        elif len(child_ids) >= 2:
            # Multiple children = meaningful branching, promote to umbrella
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
