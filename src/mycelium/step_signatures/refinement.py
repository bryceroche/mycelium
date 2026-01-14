"""Signature Refinement Loop: Self-improvement for negative-lift signatures.

DESIGN PHILOSOPHY: No hardcoded classifications. The system learns which
signatures need decomposition, guidance-only, or DSL fixes from execution
data. Classifications emerge from success/failure patterns, not manual lists.

This module implements the signature refinement loop:

1. IDENTIFY: Query signatures with success_rate < threshold or negative lift
2. CLASSIFY: Determine refinement strategy from execution metrics (not hardcoded)
3. REFINE: Apply appropriate strategy (decompose, fix DSL, or guidance-only)
4. VALIDATE: Track lift to see if refinement helped

Classification is learned:
- Umbrella candidates: step_types with high variance (sometimes works, sometimes fails)
- Guidance-only: step_types where DSL confidence is consistently low
- Fixable DSL: step_types where DSL sometimes succeeds (formula might be incomplete)
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from .db import StepSignatureDB
from .models import StepSignature

logger = logging.getLogger(__name__)


@dataclass
class RefinementResult:
    """Result of refining a single signature."""
    signature_id: int
    step_type: str
    original_lift: float
    action: str  # "decomposed", "dsl_fixed", "guidance_only", "skipped"
    children_created: int = 0
    new_dsl: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RefinementReport:
    """Summary of a refinement run."""
    signatures_analyzed: int
    decomposed: int
    dsl_fixed: int
    guidance_only: int
    skipped: int
    errors: int
    results: list[RefinementResult]


# Thresholds for dynamic classification (learned from data patterns)
VARIANCE_THRESHOLD = 0.3  # High variance = umbrella candidate
MIN_DSL_CONFIDENCE = 0.4  # Below this = guidance only
PARTIAL_SUCCESS_RATE = 0.2  # Some success = fixable DSL


class SignatureRefiner:
    """Refines signatures with negative lift through decomposition or DSL fixes.

    All classifications are learned from execution data, not hardcoded lists.
    """

    def __init__(self, db: StepSignatureDB, client=None):
        """Initialize the refiner.

        Args:
            db: Step signature database
            client: LLM client for generating decompositions/DSLs (optional)
        """
        self.db = db
        self.client = client

    async def refine_negative_lift_signatures(
        self,
        min_lift: float = -0.10,
        min_uses: int = 5,
        max_signatures: int = 20,
    ) -> RefinementReport:
        """Find and refine signatures with negative lift.

        Args:
            min_lift: Only refine signatures with lift below this threshold
            min_uses: Minimum uses in both arms to have reliable lift data
            max_signatures: Maximum signatures to process in one run

        Returns:
            RefinementReport with summary and per-signature results
        """
        # Get negative lift signatures
        candidates = self.db.get_signatures_for_dsl_improvement(
            min_uses=min_uses,
            lift_threshold=min_lift,
        )

        candidates = candidates[:max_signatures]

        logger.info(
            "[refinement] Found %d signatures with lift < %.0f%% to refine",
            len(candidates), min_lift * 100
        )

        results = []
        decomposed = 0
        dsl_fixed = 0
        guidance_only = 0
        skipped = 0
        errors = 0

        for sig in candidates:
            try:
                result = await self._refine_signature(sig)
                results.append(result)

                if result.action == "decomposed":
                    decomposed += 1
                elif result.action == "dsl_fixed":
                    dsl_fixed += 1
                elif result.action == "guidance_only":
                    guidance_only += 1
                elif result.action == "skipped":
                    skipped += 1

                if result.error:
                    errors += 1

            except Exception as e:
                logger.error("[refinement] Failed to refine sig=%d: %s", sig.id, e)
                results.append(RefinementResult(
                    signature_id=sig.id,
                    step_type=sig.step_type,
                    original_lift=self._compute_lift(sig),
                    action="error",
                    error=str(e),
                ))
                errors += 1

        return RefinementReport(
            signatures_analyzed=len(candidates),
            decomposed=decomposed,
            dsl_fixed=dsl_fixed,
            guidance_only=guidance_only,
            skipped=skipped,
            errors=errors,
            results=results,
        )

    async def _refine_signature(self, sig: StepSignature) -> RefinementResult:
        """Refine a single signature based on learned metrics."""
        lift = self._compute_lift(sig)

        # Already a semantic umbrella? Skip
        if sig.is_semantic_umbrella:
            return RefinementResult(
                signature_id=sig.id,
                step_type=sig.step_type,
                original_lift=lift,
                action="skipped",
            )

        # Classify based on execution metrics (not hardcoded lists)
        classification = self._classify_signature(sig)

        if classification == "guidance_only":
            self._convert_to_guidance_only(sig)
            return RefinementResult(
                signature_id=sig.id,
                step_type=sig.step_type,
                original_lift=lift,
                action="guidance_only",
            )

        if classification == "umbrella":
            children = await self._decompose_to_umbrella(sig)
            return RefinementResult(
                signature_id=sig.id,
                step_type=sig.step_type,
                original_lift=lift,
                action="decomposed",
                children_created=len(children),
            )

        if classification == "fixable":
            new_dsl = await self._fix_dsl(sig)
            if new_dsl:
                return RefinementResult(
                    signature_id=sig.id,
                    step_type=sig.step_type,
                    original_lift=lift,
                    action="dsl_fixed",
                    new_dsl=new_dsl,
                )

        # Can't determine refinement strategy - skip
        return RefinementResult(
            signature_id=sig.id,
            step_type=sig.step_type,
            original_lift=lift,
            action="skipped",
        )

    def _classify_signature(self, sig: StepSignature) -> str:
        """Classify signature based on execution metrics.

        Returns: "umbrella", "guidance_only", "fixable", or "unknown"
        """
        # Calculate metrics
        total_uses = sig.injected_uses + sig.non_injected_uses
        if total_uses == 0:
            return "unknown"

        injected_rate = sig.injected_successes / sig.injected_uses if sig.injected_uses > 0 else 0
        non_injected_rate = sig.non_injected_successes / sig.non_injected_uses if sig.non_injected_uses > 0 else 0

        # Variance between injected and non-injected (high = inconsistent DSL)
        variance = abs(injected_rate - non_injected_rate)

        # Get DSL confidence from execution history
        dsl_confidence = self._get_dsl_confidence(sig)

        # Classification logic based on learned patterns:

        # 1. High variance = umbrella candidate (DSL works for some contexts but not others)
        if variance >= VARIANCE_THRESHOLD and total_uses >= 10:
            logger.debug(
                "[classify] sig=%d (%s): umbrella (variance=%.2f)",
                sig.id, sig.step_type, variance
            )
            return "umbrella"

        # 2. Consistently low DSL confidence = guidance only (DSL not appropriate)
        if dsl_confidence < MIN_DSL_CONFIDENCE:
            logger.debug(
                "[classify] sig=%d (%s): guidance_only (dsl_conf=%.2f)",
                sig.id, sig.step_type, dsl_confidence
            )
            return "guidance_only"

        # 3. Some DSL success but not great = fixable (DSL might need tweaking)
        if PARTIAL_SUCCESS_RATE <= injected_rate < 0.6:
            logger.debug(
                "[classify] sig=%d (%s): fixable (inj_rate=%.2f)",
                sig.id, sig.step_type, injected_rate
            )
            return "fixable"

        return "unknown"

    def _get_dsl_confidence(self, sig: StepSignature) -> float:
        """Get average DSL execution confidence for this signature.

        Queries execution history to see how confident DSL matches were.
        Returns 0.0 if no data, 1.0 if always high confidence.
        """
        # For now, use injected success rate as proxy for confidence
        # In production, this would query actual confidence scores from execution logs
        if sig.injected_uses == 0:
            return 0.0
        return sig.injected_successes / sig.injected_uses

    def _compute_lift(self, sig: StepSignature) -> float:
        """Compute lift for a signature."""
        if sig.injected_uses == 0 or sig.non_injected_uses == 0:
            return 0.0
        inj_rate = sig.injected_successes / sig.injected_uses
        base_rate = sig.non_injected_successes / sig.non_injected_uses
        return inj_rate - base_rate

    def _convert_to_guidance_only(self, sig: StepSignature) -> None:
        """Convert a signature to guidance-only (remove DSL)."""
        with self.db._connection() as conn:
            conn.execute(
                """UPDATE step_signatures
                   SET dsl_script = NULL,
                       injected_uses = 0,
                       injected_successes = 0
                   WHERE id = ?""",
                (sig.id,)
            )
        logger.info("[refinement] Converted sig=%d (%s) to guidance-only", sig.id, sig.step_type)

    async def _decompose_to_umbrella(self, sig: StepSignature) -> list[dict]:
        """Decompose a signature into a semantic umbrella with children.

        Uses LLM to generate appropriate sub-signatures based on the step_type
        and failure patterns. No hardcoded decomposition templates.

        Returns list of created child specs.
        """
        if not self.client:
            logger.warning("[refinement] No LLM client for decomposition of %s", sig.step_type)
            return []

        # Generate decomposition using LLM
        children_specs = await self._generate_decomposition(sig)

        if not children_specs:
            logger.warning("[refinement] LLM returned no decomposition for %s", sig.step_type)
            return []

        created = []
        for spec in children_specs:
            child = self.db.create_child_signature(
                parent_id=sig.id,
                step_type=spec["step_type"],
                description=spec["description"],
                dsl_script=json.dumps(spec["dsl"]) if spec.get("dsl") else None,
                condition=spec.get("condition", ""),
            )
            if child:
                created.append(spec)
                logger.info(
                    "[refinement] Created child %s for umbrella %s",
                    spec["step_type"], sig.step_type
                )

        return created

    async def _generate_decomposition(self, sig: StepSignature) -> list[dict]:
        """Use LLM to generate decomposition for a generic step_type.

        Returns list of child specs: [{"step_type": str, "description": str, "dsl": dict, "condition": str}]
        """
        prompt = f"""You are decomposing a generic math operation into specific sub-types.

Step type: {sig.step_type}
Description: {sig.description}

This step has high variance - it works for some problems but not others.
Break it into 2-4 specific sub-types that cover different cases.

For each sub-type, provide:
1. step_type: specific snake_case name (e.g., "sum_two_numbers", "sum_arithmetic_series")
2. description: what this specific case handles
3. condition: when to use this sub-type
4. dsl: the DSL script if deterministic, or null if needs LLM

Return JSON array:
[
  {{"step_type": "...", "description": "...", "condition": "...", "dsl": {{"type": "math", "script": "...", "params": [...]}}}}
]

Only return the JSON array, no other text."""

        try:
            response = await self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            text = response.content[0].text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            return json.loads(text)

        except Exception as e:
            logger.error("[refinement] LLM decomposition failed: %s", e)
            return []

    async def _fix_dsl(self, sig: StepSignature) -> Optional[str]:
        """Try to fix a broken DSL using similar successful signatures.

        Finds signatures with similar step_type that have working DSLs,
        and adapts their DSL for this signature.
        """
        # Find similar signatures with successful DSLs
        similar = self._find_similar_successful_signature(sig)
        if similar and similar.dsl_script:
            # Clone the working DSL
            new_dsl = similar.dsl_script

            # Reset lift stats with new DSL version
            self.db.reset_lift_stats_for_dsl_version(
                signature_id=sig.id,
                new_dsl_script=new_dsl,
                new_dsl_version=(sig.dsl_version or 1) + 1,
            )

            logger.info(
                "[refinement] Fixed DSL for sig=%d (%s) by cloning from sig=%d",
                sig.id, sig.step_type, similar.id
            )
            return new_dsl

        return None

    def _find_similar_successful_signature(self, sig: StepSignature) -> Optional[StepSignature]:
        """Find a similar signature with a working DSL."""
        try:
            from mycelium.embedder import Embedder
            import numpy as np

            embedder = Embedder.get_instance()

            # Get embedding for this signature
            query_embedding = embedder.embed(f"{sig.step_type}: {sig.description}")

            # Query successful signatures
            candidates = self.db.get_signatures_with_successful_dsls(
                min_success_rate=0.7,
                min_uses=5,
                limit=20,
            )

            best_match = None
            best_similarity = 0.0

            for candidate in candidates:
                if candidate.id == sig.id:
                    continue

                sig_text = f"{candidate.step_type}: {candidate.description}"
                sig_embedding = embedder.embed(sig_text)

                similarity = float(np.dot(query_embedding, sig_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(sig_embedding)
                ))

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = candidate

            # Require high similarity for DSL cloning
            if best_match and best_similarity >= 0.8:
                return best_match

            return None

        except Exception as e:
            logger.debug("[refinement] Similar signature lookup failed: %s", e)
            return None


async def run_refinement_loop(
    db: StepSignatureDB = None,
    client = None,
    min_lift: float = -0.10,
    min_uses: int = 5,
    max_signatures: int = 20,
) -> RefinementReport:
    """Run the signature refinement loop.

    Convenience function to run refinement with default settings.

    Args:
        db: Database (creates new if None)
        client: LLM client (optional, for LLM-assisted decomposition)
        min_lift: Threshold for negative lift
        min_uses: Minimum uses for reliable data
        max_signatures: Max to process per run

    Returns:
        RefinementReport with results
    """
    if db is None:
        db = StepSignatureDB()

    refiner = SignatureRefiner(db, client)
    return await refiner.refine_negative_lift_signatures(
        min_lift=min_lift,
        min_uses=min_uses,
        max_signatures=max_signatures,
    )
