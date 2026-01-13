"""Solver V2: Simplified step-level execution with Natural Language signatures.

Flow:
    Problem → Planner → DAG steps → For each step:
        1. Embed step text
        2. Find matching signature (or create new)
        3. Execute: DSL if available, else LLM
        4. Record success/failure

Key difference from V1: Signatures speak natural language.
- clarifying_questions help extract parameters
- param_descriptions explain what each DSL param means
- Lazy NL: new signatures start empty, get filled in as we learn
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mycelium.config import (
    MIN_MATCH_THRESHOLD,
    RECURSIVE_DECOMPOSITION_ENABLED,
    RECURSIVE_MAX_DEPTH,
    RECURSIVE_CONFIDENCE_THRESHOLD,
    UMBRELLA_MAX_DEPTH,
    UMBRELLA_ROUTING_THRESHOLD,
)
from mycelium.planner import Planner, Step, DAGPlan
from mycelium.step_signatures import StepSignatureDB, StepSignature
from mycelium.step_signatures.db import normalize_step_text
from mycelium.step_signatures.dsl_executor import DSLSpec, try_execute_dsl, llm_rewrite_script, try_execute_dsl_math
from mycelium.step_signatures.dsl_generator import regenerate_dsl
from mycelium.embedder import Embedder

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class StepResult:
    """Result of executing a single step."""
    step_id: str
    task: str
    result: str
    success: bool
    signature_id: Optional[int] = None
    signature_type: Optional[str] = None
    is_new_signature: bool = False
    was_injected: bool = False  # True if DSL was used
    elapsed_ms: float = 0.0


@dataclass
class SolverResult:
    """Result of solving a complete problem."""
    problem: str
    answer: str
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    elapsed_ms: float = 0.0
    total_steps: int = 0
    signatures_matched: int = 0
    signatures_new: int = 0
    steps_with_injection: int = 0
    matched_and_injected: int = 0  # Matched existing sig AND DSL succeeded
    error: Optional[str] = None


# =============================================================================
# Solver
# =============================================================================

class Solver:
    """V2 Solver with Natural Language signatures.

    Simple flow:
    1. Plan: Decompose problem into steps
    2. Execute: For each step, find signature and execute
    3. Synthesize: Combine results into final answer
    """

    def __init__(
        self,
        solver_client=None,
        db_path: str = None,
        min_similarity: float = MIN_MATCH_THRESHOLD,
    ):
        """Initialize the solver.

        Args:
            solver_client: LLM client for step execution (optional, creates default if None)
            db_path: Path to signature database
            min_similarity: Minimum cosine similarity for signature matching
        """
        from mycelium.client import get_client

        self.planner = Planner()  # Uses its own client internally
        self.solver_client = solver_client or get_client()
        self.step_db = StepSignatureDB(db_path=db_path)
        self.embedder = Embedder.get_instance()
        self.min_similarity = min_similarity
        self._background_tasks: set[asyncio.Task] = set()  # Track background tasks

    def _create_background_task(self, coro) -> asyncio.Task:
        """Create a background task with proper lifecycle management.

        Tasks are tracked to prevent garbage collection and enable clean shutdown.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def wait_for_background_tasks(self, timeout: float = 5.0) -> int:
        """Wait for pending background tasks to complete.

        Call this for clean shutdown. Returns count of tasks that were pending.
        """
        if not self._background_tasks:
            return 0
        pending = len(self._background_tasks)
        logger.debug("[solver] Waiting for %d background tasks", pending)
        done, not_done = await asyncio.wait(
            self._background_tasks,
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED,
        )
        if not_done:
            logger.warning("[solver] %d background tasks did not complete in time", len(not_done))
        return pending

    async def solve(self, problem: str) -> SolverResult:
        """Solve a problem end-to-end.

        Args:
            problem: The problem text

        Returns:
            SolverResult with answer and step details
        """
        import time
        start_time = time.time()

        try:
            # 1. Plan: Decompose into steps (with signature hints for NL interface)
            # Embed problem to filter hints by semantic similarity
            problem_embedding = self.embedder.embed(problem)
            signature_hints = self.step_db.get_signature_hints(
                limit=15,
                problem_embedding=problem_embedding,
                min_similarity=0.3,  # Only hints somewhat related to this problem
            )
            plan = await self.planner.decompose(problem, signature_hints=signature_hints)

            # Validate DAG structure before execution
            is_valid, errors = plan.validate()
            if not is_valid:
                return SolverResult(
                    problem=problem,
                    answer="",
                    success=False,
                    error=f"Invalid DAG: {'; '.join(errors)}",
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

            if not plan.steps:
                return SolverResult(
                    problem=problem,
                    answer="",
                    success=False,
                    error="Planning failed: no steps generated",
                    elapsed_ms=(time.time() - start_time) * 1000,
                )

            # 2. Execute steps in dependency order
            step_results = []
            context = {}  # step_id → result
            step_descriptions = {}  # step_id → task description (for NL param matching)
            signatures_new = 0
            signatures_matched = 0
            steps_with_injection = 0
            matched_and_injected = 0

            execution_order = self._get_execution_order(plan)

            for step in execution_order:
                # Build context from dependencies
                step_context = {
                    dep: context[dep]
                    for dep in step.depends_on
                    if dep in context
                }
                # Build step descriptions for semantic param matching
                step_desc_context = {
                    dep: step_descriptions[dep]
                    for dep in step.depends_on
                    if dep in step_descriptions
                }

                # Execute step
                result = await self._execute_step(step, problem, step_context, step_desc_context)
                step_results.append(result)

                # Abort DAG on step failure (prevent cascading empty strings)
                if not result.success:
                    logger.warning(
                        "[solver] Step failed, aborting DAG: step=%s task='%s'",
                        step.id, step.task[:50]
                    )
                    elapsed_ms = (time.time() - start_time) * 1000
                    return SolverResult(
                        problem=problem,
                        answer="",
                        success=False,
                        error=f"Step {step.id} failed: {step.task[:100]}",
                        steps=step_results,
                        elapsed_ms=elapsed_ms,
                        total_steps=len(step_results),
                        signatures_matched=signatures_matched,
                        signatures_new=signatures_new,
                        steps_with_injection=steps_with_injection,
                        matched_and_injected=matched_and_injected,
                    )

                # Track stats
                if result.is_new_signature:
                    signatures_new += 1
                else:
                    signatures_matched += 1
                    if result.was_injected:
                        matched_and_injected += 1
                if result.was_injected:
                    steps_with_injection += 1

                # Store result and description for dependent steps
                context[step.id] = result.result
                step_descriptions[step.id] = step.task

            # 3. Synthesize final answer
            final_answer = await self._synthesize(problem, step_results, context)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "[solver] Solved in %.0fms: steps=%d new=%d matched=%d matched+injected=%d",
                elapsed_ms, len(step_results), signatures_new, signatures_matched, matched_and_injected
            )

            return SolverResult(
                problem=problem,
                answer=final_answer,
                success=True,
                steps=step_results,
                elapsed_ms=elapsed_ms,
                total_steps=len(step_results),
                signatures_matched=signatures_matched,
                signatures_new=signatures_new,
                steps_with_injection=steps_with_injection,
                matched_and_injected=matched_and_injected,
            )

        except Exception as e:
            logger.exception("[solver] Error solving problem")
            return SolverResult(
                problem=problem,
                answer="",
                success=False,
                error=str(e),
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    async def _execute_step(
        self,
        step: Step,
        problem: str,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
        depth: int = 0,
    ) -> StepResult:
        """Execute a single step.

        Flow:
        0. If composite (has sub_plan), recursively execute sub-plan first
        1. Embed step text
        2. Find or create signature
        3. If umbrella, route to child signature
        4. Try DSL if available (using NL interface for param matching)
        5. Fall back to LLM if needed
        6. Record usage

        Args:
            step: The step to execute
            problem: Original problem text
            context: step_id → result from previous steps
            step_descriptions: step_id → task description (for NL param matching)
            depth: Recursion depth for composite steps
        """
        step_descriptions = step_descriptions or {}
        import time
        start_time = time.time()

        # 0. Handle composite steps (recursive DAG of DAGs)
        if step.is_composite:
            return await self._execute_composite_step(step, problem, context, step_descriptions, depth)

        # 1. Normalize and embed step (strip numbers for better matching)
        normalized_task = normalize_step_text(step.task)
        embedding = self.embedder.embed(normalized_task)

        # 2. Find or create signature (use original text for description)
        signature, is_new = self.step_db.find_or_create(
            step_text=step.task,  # Keep original for description
            embedding=embedding,   # Use normalized embedding for matching
            min_similarity=self.min_similarity,
            parent_problem=problem,
            origin_depth=depth,  # Track decomposition depth
        )

        logger.debug(
            "[solver] Step '%s' → signature '%s' (new=%s, umbrella=%s, dsl_type=%s)",
            step.task[:40], signature.step_type, is_new, signature.is_semantic_umbrella,
            signature.dsl_type
        )

        # 2.5. Auto-decompose if signature needs children
        # Case 1: decompose-type that isn't umbrella yet
        # Case 2: umbrella (possibly auto-demoted) with no children
        needs_decompose = False
        if signature.dsl_type == "decompose" and not signature.is_semantic_umbrella:
            needs_decompose = True
            reason = "decompose type needs children"
        elif signature.is_semantic_umbrella:
            children = self.step_db.get_children(signature.id)
            if not children:
                needs_decompose = True
                reason = "umbrella has no children (auto-demoted?)"

        if needs_decompose:
            logger.info(
                "[solver] Auto-decomposing '%s' (%s)",
                signature.step_type, reason
            )
            await self._auto_decompose_signature(signature)
            # Refresh signature after decomposition
            signature = self.step_db.get_signature(signature.id)

        # 3. If umbrella, try routing to child signature
        result = None
        was_injected = False
        routed_signature = signature

        if signature.is_semantic_umbrella:
            child_result = await self._try_umbrella_routing(signature, step, problem, context, step_descriptions, embedding=embedding)
            if child_result is not None:
                result, routed_signature, was_injected = child_result
                logger.info(
                    "[solver] Umbrella routed: '%s' → '%s'",
                    signature.step_type, routed_signature.step_type
                )

        # 4. Try DSL execution if not already routed
        if result is None and routed_signature.dsl_script:
            dsl_result = await self._try_dsl(routed_signature, step, context, step_descriptions)
            if dsl_result is not None:
                result = dsl_result
                was_injected = True
                logger.debug("[solver] DSL executed: %s", result[:50] if result else "")

        # 5. No LLM fallback - strict DAG execution
        # Three outcomes: route to child, create child, or fail
        if result is None:
            logger.warning(
                "[solver] DSL failed, step failed (no LLM fallback): %s",
                step.task[:50]
            )
            result = ""  # Empty result = failure

        # 6. Record usage (step_completed = returned result, not problem correctness)
        # Problem correctness is tracked separately via update_problem_outcome()
        step_completed = bool(result)
        uses = self.step_db.record_usage(
            signature_id=routed_signature.id,
            step_text=step.task,
            step_completed=step_completed,
            was_injected=was_injected,
        )

        # 7. Regenerate DSL on mod 10 uses (continuous learning)
        # Fire-and-forget: don't block the hot path
        if uses > 0 and uses % 10 == 0:
            asyncio.create_task(
                self._regenerate_dsl_background(routed_signature.id, uses)
            )

        elapsed_ms = (time.time() - start_time) * 1000

        return StepResult(
            step_id=step.id,
            task=step.task,
            result=result or "",
            success=step_completed,
            signature_id=routed_signature.id,
            signature_type=routed_signature.step_type,
            is_new_signature=is_new,
            was_injected=was_injected,
            elapsed_ms=elapsed_ms,
        )

    async def _execute_composite_step(
        self,
        step: Step,
        problem: str,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
        depth: int = 0,
    ) -> StepResult:
        """Execute a composite step by recursively executing its sub-plan.

        A composite step contains a sub-DAG. We:
        1. Execute all steps in the sub-plan (respecting dependencies)
        2. Use the sub-plan's final result as this step's result

        This enables unlimited recursive nesting: DAG of DAGs of DAGs...

        Args:
            step: The composite step to execute
            problem: Original problem text
            context: step_id → result from parent/sibling steps
            step_descriptions: step_id → task description (for NL param matching)
            depth: Recursion depth for nested composites
        """
        step_descriptions = step_descriptions or {}
        import time
        start_time = time.time()

        sub_plan = step.sub_plan
        logger.info(
            "[solver] Executing composite step '%s' with %d sub-steps (depth=%d)",
            step.id, len(sub_plan.steps), depth
        )

        # Execute sub-plan steps in dependency order
        sub_context = dict(context)  # Inherit parent context
        sub_step_descriptions = dict(step_descriptions)  # Inherit parent descriptions
        sub_results = []

        sub_execution_order = self._get_execution_order(sub_plan)

        for sub_step in sub_execution_order:
            # Build context from sub-plan dependencies
            step_context = {
                dep: sub_context[dep]
                for dep in sub_step.depends_on
                if dep in sub_context
            }
            # Also include parent context
            step_context.update({
                k: v for k, v in context.items()
                if k not in step_context
            })

            # Build step descriptions for semantic param matching
            step_desc_context = {
                dep: sub_step_descriptions[dep]
                for dep in sub_step.depends_on
                if dep in sub_step_descriptions
            }
            # Also include parent descriptions
            step_desc_context.update({
                k: v for k, v in step_descriptions.items()
                if k not in step_desc_context
            })

            # Recursively execute (handles nested composites)
            sub_result = await self._execute_step(
                sub_step, problem, step_context, step_desc_context, depth=depth + 1
            )
            sub_results.append(sub_result)

            # Store result and description for dependent sub-steps
            sub_context[sub_step.id] = sub_result.result
            sub_step_descriptions[sub_step.id] = sub_step.task

        # Aggregate sub-results into composite result
        # The final sub-step's result becomes this step's result
        final_result = sub_results[-1].result if sub_results else ""
        # Empty sub_results should be considered failure (not vacuous truth)
        all_success = bool(sub_results) and all(r.success for r in sub_results)

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            "[solver] Composite step '%s' completed: sub_steps=%d success=%s (%.0fms)",
            step.id, len(sub_results), all_success, elapsed_ms
        )

        return StepResult(
            step_id=step.id,
            task=step.task,
            result=final_result,
            success=all_success,
            signature_id=None,  # Composite steps don't have their own signature
            signature_type=f"composite[{len(sub_results)}]",
            is_new_signature=False,
            was_injected=False,
            elapsed_ms=elapsed_ms,
        )

    async def _try_umbrella_routing(
        self,
        umbrella: StepSignature,
        step: Step,
        problem: str,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
        visited: Optional[set[int]] = None,
        embedding: Optional[np.ndarray] = None,
        depth: int = 0,
    ) -> Optional[tuple[str, StepSignature, bool]]:
        """Try to route through umbrella to a child signature.

        Uses embedding similarity for fast routing (~0ms) instead of LLM calls.

        Returns (result, child_signature, was_injected) or None if no match.

        Args:
            umbrella: The umbrella signature to route through
            step: The step being executed
            problem: The original problem text
            context: Results from previous steps
            visited: Set of already-visited umbrella IDs (cycle detection)
            embedding: Step embedding for similarity-based routing
            depth: Current recursion depth (for limiting chain length)
        """
        from mycelium.step_signatures.utils import cosine_similarity

        # Depth limit: prevent unbounded recursion through long umbrella chains
        if depth >= UMBRELLA_MAX_DEPTH:
            logger.warning(
                "[solver] Umbrella routing depth limit reached: depth=%d, umbrella=%s",
                depth, umbrella.step_type
            )
            return None

        # Cycle detection: prevent infinite recursion on malformed DAG
        if visited is None:
            visited = set()
        if umbrella.id in visited:
            logger.warning(
                "[solver] Cycle detected in umbrella routing: %d already visited",
                umbrella.id
            )
            return None
        visited.add(umbrella.id)

        children = self.step_db.get_children(umbrella.id)
        if not children:
            return None

        # Embedding-based routing: compare step embedding to child centroids
        # This is ~0ms vs ~500ms for LLM routing
        if embedding is not None:
            best_child = None
            best_sim = 0.0
            best_condition = ""

            for child_sig, condition in children:
                if child_sig.centroid is not None:
                    sim = cosine_similarity(embedding, child_sig.centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_child = child_sig
                        best_condition = condition

            # Use embedding match if similarity is reasonable
            if best_child and best_sim > UMBRELLA_ROUTING_THRESHOLD:
                logger.debug(
                    "[solver] Umbrella routing (embedding): '%s' → '%s' (sim=%.3f)",
                    umbrella.step_type, best_child.step_type, best_sim
                )
                child_sig = best_child
            else:
                logger.debug(
                    "[solver] Umbrella routing: no good embedding match (best=%.3f)",
                    best_sim
                )
                return None
        else:
            # Fallback to LLM routing if no embedding available
            conditions = [f"{i+1}. {cond}" for i, (_, cond) in enumerate(children)]
            prompt = f"""Given this step: "{step.task}"

Which of these sub-categories best matches?
{chr(10).join(conditions)}
0. None of the above

Respond with ONLY the number (0-{len(children)})."""

            messages = [{"role": "user", "content": prompt}]
            response = await self.solver_client.generate(messages, temperature=0.0)

            # Parse response - robust regex extraction
            choice = 0
            match = re.search(r'\b([0-9]+)\b', response)
            if match:
                try:
                    choice = int(match.group(1))
                except ValueError:
                    pass

            if choice <= 0 or choice > len(children):
                logger.debug(
                    "[solver] Umbrella routing: no valid choice from response '%s'",
                    response[:50]
                )
                return None

            child_sig, _ = children[choice - 1]
            logger.debug(
                "[solver] Umbrella routing (LLM) selected child %d: %s",
                choice, child_sig.step_type
            )

        # Recurse if child is also an umbrella (pass visited set, embedding, and depth)
        if child_sig.is_semantic_umbrella:
            return await self._try_umbrella_routing(
                child_sig, step, problem, context, step_descriptions, visited, embedding,
                depth=depth + 1
            )

        # Try child's DSL
        if child_sig.dsl_script:
            dsl_result = await self._try_dsl(child_sig, step, context, step_descriptions)
            if dsl_result is not None:
                return (dsl_result, child_sig, True)

        # Return child for DSL execution (no LLM fallback)
        return (None, child_sig, False)

    async def _try_dsl(
        self,
        signature: StepSignature,
        step: Step,
        context: dict[str, str],
        step_descriptions: dict[str, str] = None,
    ) -> Optional[str]:
        """Try to execute a DSL script.

        Uses step.extracted_values (from planner) and context for params.
        Uses signature's param_descriptions + step_descriptions for semantic matching.
        """
        step_descriptions = step_descriptions or {}
        if not signature.dsl_script:
            return None

        try:
            # Parse DSL spec from JSON
            dsl_spec = DSLSpec.from_json(signature.dsl_script)
            if not dsl_spec:
                logger.debug("[solver] Failed to parse DSL spec")
                return None

            # Build params from multiple sources:
            # 1. Step's extracted_values (from planner) - these have semantic names
            # 2. Context from previous steps
            # 3. Numbers extracted from step text
            params = {}

            # Add step's extracted values (highest priority - planner knows the semantics)
            if hasattr(step, 'extracted_values') and step.extracted_values:
                for key, val in step.extracted_values.items():
                    # Resolve references like "{step_1}" from context
                    if isinstance(val, str) and val.startswith('{') and val.endswith('}'):
                        ref_key = val[1:-1]  # Remove braces
                        if ref_key in context:
                            try:
                                params[key] = float(context[ref_key])
                            except (ValueError, TypeError):
                                params[key] = context[ref_key]
                    else:
                        params[key] = val

            # Add context values
            for key, value in context.items():
                if key not in params:  # Don't override extracted values
                    try:
                        params[key] = float(value)
                    except (ValueError, TypeError):
                        params[key] = value

            # Use NL interface for semantic param mapping
            # Match signature's param_descriptions against step_descriptions
            if signature.param_descriptions and step_descriptions and dsl_spec.params:
                nl_mapped = self._semantic_map_params(
                    dsl_spec.params,
                    signature.param_descriptions,
                    context,
                    step_descriptions,
                )
                if nl_mapped:
                    logger.debug("[solver] NL interface mapped params: %s", nl_mapped)
                    # Add NL-mapped params (don't override existing)
                    for param, value in nl_mapped.items():
                        if param not in params:
                            params[param] = value

            # Fall back to extracting from step text if no params yet
            if not params:
                params = self._extract_params(signature, step.task, context, dsl_spec)

            if not params:
                logger.debug("[solver] No params extracted for DSL")
                return None

            logger.debug("[solver] DSL params: %s", params)

            # Execute DSL
            result, success = try_execute_dsl(
                dsl_spec,
                params,
                step_task=step.task,
            )

            if success and result is not None:
                logger.info("[solver] DSL injection success: %s → %s", step.task[:30], result)
                return str(result)

            # NOTE: LLM script rewriting is available but disabled by default
            # It can help with param name mismatches but may produce wrong results
            # when the matched signature's operation doesn't match the step's intent
            # Uncomment to enable:
            # if not success and params and dsl_spec.layer.value == "math":
            #     rewritten_script = await llm_rewrite_script(dsl_spec, params, self.solver_client, current_step_task=step.task)
            #     if rewritten_script and rewritten_script != dsl_spec.script:
            #         rewritten_result = try_execute_dsl_math(rewritten_script, params)
            #         if rewritten_result is not None:
            #             return str(rewritten_result)

        except Exception as e:
            logger.debug("[solver] DSL execution failed: %s", e)

        return None

    def _semantic_map_params(
        self,
        dsl_params: list[str],
        param_descriptions: dict[str, str],
        context: dict[str, str],
        step_descriptions: dict[str, str],
    ) -> dict:
        """Map DSL params to context values using NL interface semantic matching.

        Uses param_descriptions (from signature's NL interface) to find which
        context step's description best matches each param's meaning.

        Example:
            param_descriptions = {"base": "The number being raised to a power"}
            step_descriptions = {"step_1": "Calculate the base value"}
            → "base" matches "step_1" because descriptions are similar

        Args:
            dsl_params: List of param names from DSL spec
            param_descriptions: Param name → description from signature
            context: step_id → result value
            step_descriptions: step_id → task description

        Returns:
            Dict mapping param_name → value from context
        """
        mapped = {}
        used_steps = set()

        for param in dsl_params:
            # Get param's semantic description
            param_desc = param_descriptions.get(param, "").lower()
            if not param_desc:
                continue

            best_match = None
            best_score = 0.0
            param_lower = param.lower().replace("_", " ")

            for step_id, step_desc in step_descriptions.items():
                if step_id in used_steps:
                    continue
                if step_id not in context:
                    continue

                step_desc_lower = step_desc.lower()
                score = 0.0

                # Score 1: Param description words appear in step description
                param_words = set(param_desc.split())
                step_words = set(step_desc_lower.split())
                overlap = param_words & step_words
                if overlap:
                    # More overlap = better match
                    score = len(overlap) / max(len(param_words), 1)

                # Score 2: Param name appears in step description
                if param_lower in step_desc_lower:
                    score = max(score, 0.8)

                # Score 3: Key semantic words match
                semantic_keywords = {"calculate", "compute", "find", "determine", "get"}
                if any(kw in step_desc_lower for kw in semantic_keywords):
                    # Step is a computation - boost if param desc also mentions computation
                    if any(kw in param_desc for kw in semantic_keywords):
                        score += 0.1

                if score > best_score:
                    best_score = score
                    best_match = step_id

            if best_match and best_score >= 0.3:
                # Get value from context
                try:
                    mapped[param] = float(context[best_match])
                except (ValueError, TypeError):
                    mapped[param] = context[best_match]
                used_steps.add(best_match)
                logger.debug(
                    "[solver] NL mapped param '%s' (%s) → %s (score=%.2f)",
                    param, param_desc[:30], best_match, best_score
                )

        return mapped

    def _extract_params(
        self,
        signature: StepSignature,
        step_text: str,
        context: dict[str, str],
        dsl_spec: Optional[DSLSpec] = None,
    ) -> dict:
        """Extract DSL parameters from step text and context.

        Uses DSL spec's param names if available, else param_descriptions.
        """
        params = {}

        # Extract numbers from step text
        numbers = re.findall(r'(?<![a-zA-Z])(\d+\.?\d*)(?![a-zA-Z])', step_text)

        # Get param names from DSL spec, param_descriptions, or use generic
        if dsl_spec and dsl_spec.params:
            param_names = dsl_spec.params
        elif signature.param_descriptions:
            param_names = list(signature.param_descriptions.keys())
        else:
            param_names = [f"x{i}" for i in range(len(numbers))]

        # Assign numbers in order to param names
        for i, num in enumerate(numbers):
            if i < len(param_names):
                try:
                    params[param_names[i]] = float(num)
                except ValueError:
                    params[param_names[i]] = num

        # Add context values (may contain results from previous steps)
        for key, value in context.items():
            # Try to map context keys to param names
            try:
                numeric_val = float(value)
                # Add with original key and prefixed
                params[key] = numeric_val
                params[f"ctx_{key}"] = numeric_val
            except (ValueError, TypeError):
                params[key] = str(value)
                params[f"ctx_{key}"] = str(value)

        return params

    def _extract_json_result(self, response: str) -> str:
        """Extract result from JSON response (may be embedded in text)."""
        if not response:
            return ""

        import json

        def format_result(value):
            """Format a result value for output."""
            if isinstance(value, (int, float)):
                if isinstance(value, float) and value == int(value):
                    return str(int(value))
                return str(value)
            return str(value).strip()

        # Try parsing entire response as JSON first
        try:
            data = json.loads(response.strip())
            if isinstance(data, dict):
                if "result" in data:
                    return format_result(data["result"])
                if "answer" in data:
                    return format_result(data["answer"])
        except json.JSONDecodeError:
            pass

        # Find JSON objects with balanced braces (handles nested objects)
        for key in ("result", "answer"):
            pattern = f'"{key}"'
            if pattern not in response:
                continue

            # Find all { positions and try parsing from each
            for i, char in enumerate(response):
                if char != '{':
                    continue
                # Try to find balanced closing brace
                depth = 0
                for j in range(i, len(response)):
                    if response[j] == '{':
                        depth += 1
                    elif response[j] == '}':
                        depth -= 1
                        if depth == 0:
                            # Found balanced braces, try parsing
                            candidate = response[i:j+1]
                            try:
                                data = json.loads(candidate)
                                if isinstance(data, dict) and key in data:
                                    return format_result(data[key])
                            except json.JSONDecodeError:
                                pass
                            break

        # Fallback to regex extraction
        logger.debug("[solver] JSON extraction failed, using regex")
        return self._extract_result(response)

    def _extract_result(self, response: str) -> str:
        """Extract numeric/symbolic result from LLM response."""
        if not response:
            return ""

        # Clean up response
        text = response.strip()

        # Try to find boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Try to find "= X" pattern
        equals_match = re.search(r'=\s*([^\n,;]+)$', text, re.MULTILINE)
        if equals_match:
            return equals_match.group(1).strip()

        # Try to find last number
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]

        # Return cleaned text
        return text.split('\n')[0].strip()

    async def _synthesize(
        self,
        problem: str,
        step_results: list[StepResult],
        context: dict[str, str],
    ) -> str:
        """Synthesize final answer from step results using JSON mode."""
        # If only one step, return its result
        if len(step_results) == 1:
            return step_results[0].result

        # Build synthesis prompt
        prompt_parts = [
            "Based on these step results, provide the FINAL answer to the problem.",
            "",
            f"Problem: {problem}",
            "",
            "Step results:",
        ]

        for result in step_results:
            prompt_parts.append(f"  {result.step_id}: {result.task}")
            prompt_parts.append(f"    Result: {result.result}")

        # Request JSON output (include "JSON" for Groq compatibility)
        prompt_parts.append("")
        prompt_parts.append('Respond with valid JSON: {"result": <final_answer>}')

        prompt = "\n".join(prompt_parts)
        messages = [{"role": "user", "content": prompt}]

        response = await self.solver_client.generate(
            messages,
            response_format={"type": "json_object"},
        )
        return self._extract_json_result(response)

    def _get_execution_order(self, plan: DAGPlan) -> list[Step]:
        """Get steps in dependency-respecting execution order."""
        # Topological sort
        completed = set()
        order = []

        while len(order) < len(plan.steps):
            for step in plan.steps:
                if step.id in completed:
                    continue
                if all(dep in completed for dep in step.depends_on):
                    order.append(step)
                    completed.add(step.id)
                    break
            else:
                # No progress - cycle or missing dep
                remaining = [s for s in plan.steps if s.id not in completed]
                order.extend(remaining)
                break

        return order

    def record_problem_outcome(
        self,
        result: SolverResult,
        correct: bool,
    ) -> list[int]:
        """Propagate problem correctness to all signatures used.

        Call this after grading a problem to track real success rates.
        This enables negative lift detection for umbrella learning.

        Args:
            result: The SolverResult from solve()
            correct: Whether the final answer was correct

        Returns:
            List of signature IDs that may need decomposition (low confidence)
        """
        signature_ids = [
            step.signature_id
            for step in result.steps
            if step.signature_id is not None
        ]
        self.step_db.update_problem_outcome(signature_ids, correct)

        # Log step-level details on failure for debugging
        if not correct:
            logger.warning(
                "[solver] Problem failed - steps involved: %s",
                [(s.step_id, s.signature_type, s.result[:30] if s.result else "None")
                 for s in result.steps]
            )

        # Check which signatures might need decomposition
        # Use same thresholds as umbrella_learner for consistency
        from mycelium.step_signatures.umbrella_learner import (
            MIN_USES_FOR_EVALUATION,
            MAX_SUCCESS_RATE_FOR_DECOMPOSITION,
        )
        candidates = []
        for sig_id in signature_ids:
            sig = self.step_db.get_signature(sig_id)
            if sig and sig.dsl_type == "decompose" and sig.uses >= MIN_USES_FOR_EVALUATION:
                if sig.success_rate <= MAX_SUCCESS_RATE_FOR_DECOMPOSITION and not sig.is_semantic_umbrella:
                    candidates.append(sig_id)
                    logger.info(
                        "[solver] Signature '%s' (id=%d) needs decomposition: "
                        "uses=%d, success_rate=%.1f%%",
                        sig.step_type, sig_id, sig.uses, sig.success_rate * 100
                    )

        return candidates

    async def _auto_decompose_signature(self, signature) -> bool:
        """Auto-decompose a decompose-type signature into computable children.

        Called when we encounter a decompose-type signature that needs children.
        Creates children with actual DSLs and promotes parent to umbrella.

        Args:
            signature: The decompose-type signature to decompose

        Returns:
            True if decomposition succeeded, False otherwise
        """
        from mycelium.step_signatures.umbrella_learner import UmbrellaLearner

        learner = UmbrellaLearner(self.step_db)
        try:
            child_ids = await learner.decompose_signature(signature)
            if child_ids:
                logger.info(
                    "[solver] Auto-decomposed '%s' into %d children",
                    signature.step_type, len(child_ids)
                )
                return True
            else:
                logger.warning(
                    "[solver] Could not auto-decompose '%s' (no children created)",
                    signature.step_type
                )
                return False
        except Exception as e:
            logger.error("[solver] Auto-decomposition failed: %s", e)
            return False

    async def maybe_learn_umbrellas(self, candidates: list[int]) -> dict:
        """Trigger umbrella learning if there are candidates.

        Call this after record_problem_outcome() with its return value.

        Args:
            candidates: Signature IDs that may need decomposition

        Returns:
            Dict with learning statistics (empty if no candidates)
        """
        if not candidates:
            return {"candidates": 0, "decomposed": 0, "children_created": 0}

        logger.info("[solver] Auto-triggering umbrella learning for %d candidates", len(candidates))
        return await self.learn_umbrellas()

    async def _regenerate_dsl_background(self, signature_id: int, uses: int) -> None:
        """Background task to regenerate DSL without blocking hot path."""
        try:
            regenerated = await regenerate_dsl(
                db=self.step_db,
                client=self.solver_client,
                signature_id=signature_id,
            )
            if regenerated:
                logger.info(
                    "[solver] Regenerated DSL for signature %d at %d uses",
                    signature_id, uses
                )
        except Exception as e:
            logger.warning("[solver] DSL regeneration failed: %s", e)

    async def learn_umbrellas(self) -> dict:
        """Learn umbrella structure from failing guidance signatures.

        Call this periodically or after a batch of solves to:
        1. Find guidance signatures that are failing
        2. Decompose them into specialized children
        3. Future solves will route through umbrellas to children

        Returns:
            Dict with learning statistics
        """
        from mycelium.step_signatures.umbrella_learner import UmbrellaLearner

        learner = UmbrellaLearner(self.step_db)
        return await learner.learn_from_failures()
