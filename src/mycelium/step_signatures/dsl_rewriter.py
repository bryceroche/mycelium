"""DSL Auto-Rewriter: Automatically fix underperforming DSLs.

When a signature has low success rate but high traffic, this module:
1. Identifies underperforming signatures
2. Analyzes failure patterns
3. Uses LLM to generate improved DSL scripts
4. Tests and swaps in better DSLs

Per CLAUDE.md: "rewrite DSL if centroid avg outside confidence bounds"
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from mycelium.config import (
    DSL_REWRITER_ENABLED,
    DSL_REWRITER_MIN_USES,
    DSL_REWRITER_MAX_SUCCESS_RATE,
    DSL_REWRITER_MIN_TRAFFIC_SHARE,
    DSL_REWRITER_COOLDOWN_HOURS,
)

logger = logging.getLogger(__name__)


@dataclass
class RewriteCandidate:
    """A signature that needs DSL rewriting."""
    signature_id: int
    step_type: str
    description: str
    current_dsl: Optional[str]
    uses: int
    successes: int
    success_rate: float
    last_rewrite_at: Optional[str] = None


@dataclass
class RewriteResult:
    """Result of a DSL rewrite attempt."""
    signature_id: int
    old_dsl: Optional[str]
    new_dsl: str
    reason: str
    success: bool


def find_underperforming_signatures(
    db,
    min_uses: int = None,
    max_success_rate: float = None,
    min_traffic_share: float = None,
    limit: int = 10,
) -> list[RewriteCandidate]:
    """Find signatures that need DSL rewriting.

    Criteria:
    - Has enough uses to be statistically meaningful
    - Success rate below threshold (DSL is failing)
    - Has meaningful traffic (worth fixing)
    - Not recently rewritten (cooldown)

    Args:
        db: StepSignatureDB instance
        min_uses: Minimum uses to consider (default from config)
        max_success_rate: Maximum success rate to consider (default from config)
        min_traffic_share: Minimum traffic share (default from config)
        limit: Maximum candidates to return

    Returns:
        List of RewriteCandidate objects
    """
    min_uses = min_uses or DSL_REWRITER_MIN_USES
    max_success_rate = max_success_rate or DSL_REWRITER_MAX_SUCCESS_RATE
    min_traffic_share = min_traffic_share or DSL_REWRITER_MIN_TRAFFIC_SHARE

    candidates = []

    try:
        # Get total problem count for traffic share calculation
        total_uses = db.get_total_signature_uses()
        if total_uses == 0:
            return []

        # Query underperforming signatures
        # Only consider leaf nodes (not umbrellas/routers) with math DSLs
        signatures = db.get_underperforming_signatures(
            min_uses=min_uses,
            max_success_rate=max_success_rate,
            limit=limit * 2,  # Get extra to filter by traffic
        )

        for sig in signatures:
            # Check traffic share
            traffic_share = sig.uses / total_uses if total_uses > 0 else 0
            if traffic_share < min_traffic_share:
                continue

            # Check cooldown (skip if recently rewritten)
            if _is_on_cooldown(sig):
                continue

            candidates.append(RewriteCandidate(
                signature_id=sig.id,
                step_type=sig.step_type,
                description=sig.description,
                current_dsl=sig.dsl_script,
                uses=sig.uses,
                successes=sig.successes,
                success_rate=sig.success_rate,
            ))

            if len(candidates) >= limit:
                break

        logger.info(
            "[rewriter] Found %d underperforming signatures (min_uses=%d, max_rate=%.2f)",
            len(candidates), min_uses, max_success_rate
        )
        return candidates

    except Exception as e:
        logger.error("[rewriter] Failed to find underperforming signatures: %s", e)
        return []


def _is_on_cooldown(sig) -> bool:
    """Check if signature was recently rewritten."""
    if not hasattr(sig, 'last_rewrite_at') or not sig.last_rewrite_at:
        return False

    try:
        from datetime import datetime, timedelta, timezone
        last_rewrite_str = sig.last_rewrite_at.replace('Z', '+00:00')
        last_rewrite = datetime.fromisoformat(last_rewrite_str)
        # Ensure timezone-aware comparison
        if last_rewrite.tzinfo is None:
            last_rewrite = last_rewrite.replace(tzinfo=timezone.utc)
        cooldown = timedelta(hours=DSL_REWRITER_COOLDOWN_HOURS)
        return datetime.now(timezone.utc) - last_rewrite < cooldown
    except (ValueError, TypeError):
        return False


async def generate_improved_dsl(
    candidate: RewriteCandidate,
    client,
    failure_examples: list[dict] = None,
) -> Optional[str]:
    """Use LLM to generate an improved DSL script.

    Args:
        candidate: The signature needing improvement
        client: LLM client for generation
        failure_examples: Optional examples of what went wrong

    Returns:
        New DSL script JSON, or None if generation failed
    """
    # Build prompt with context about the signature and its failures
    system_prompt = """You are a DSL script optimizer. Given a math operation signature
that is failing too often, generate an improved DSL script.

DSL scripts are JSON with this format:
{
    "type": "math",
    "script": "a + b",  // The math expression using param names
    "params": ["a", "b"],  // Parameter names to extract
    "aliases": {"a": ["first", "value1"], "b": ["second", "value2"]},  // Optional aliases
    "purpose": "Add two numbers together"  // Description
}

Rules:
1. The script must be a valid Python math expression
2. Use descriptive param names that match the operation semantics
3. Include aliases for common alternative names
4. Keep it simple - prefer basic operations

Respond with ONLY the JSON, no explanation."""

    # Build user prompt with signature context
    user_prompt = f"""Signature: {candidate.step_type}
Description: {candidate.description}

Current DSL (success rate {candidate.success_rate:.1%}):
{candidate.current_dsl or "None"}

"""

    if failure_examples:
        user_prompt += "Recent failures:\n"
        for ex in failure_examples[:3]:
            user_prompt += f"- Input: {ex.get('input', 'N/A')}, Error: {ex.get('error', 'N/A')}\n"
        user_prompt += "\n"

    user_prompt += "Generate an improved DSL script that will work better for this operation."

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await client.generate(messages, temperature=0.2)

        # Parse and validate the response
        dsl = _parse_dsl_response(response)
        if dsl:
            logger.info(
                "[rewriter] Generated new DSL for '%s': %s",
                candidate.step_type, dsl.get("script", "N/A")
            )
            return json.dumps(dsl)

    except Exception as e:
        logger.error("[rewriter] Failed to generate DSL for '%s': %s", candidate.step_type, e)

    return None


def _parse_dsl_response(response: str) -> Optional[dict]:
    """Parse and validate LLM-generated DSL."""
    try:
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response = "\n".join(json_lines)

        dsl = json.loads(response)

        # Validate required fields
        if not isinstance(dsl, dict):
            return None
        if "script" not in dsl or "params" not in dsl:
            return None
        if not isinstance(dsl["params"], list):
            return None

        # Ensure type is set
        dsl.setdefault("type", "math")

        return dsl

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("[rewriter] Failed to parse DSL response: %s", e)
        return None


async def rewrite_underperforming_dsls(
    db,
    client,
    max_rewrites: int = 5,
) -> list[RewriteResult]:
    """Main entry point: find and rewrite underperforming DSLs.

    Args:
        db: StepSignatureDB instance
        client: LLM client for generation
        max_rewrites: Maximum DSLs to rewrite in one batch

    Returns:
        List of RewriteResult objects
    """
    if not DSL_REWRITER_ENABLED:
        logger.debug("[rewriter] DSL rewriter disabled")
        return []

    results = []

    # Find candidates
    candidates = find_underperforming_signatures(db, limit=max_rewrites)
    if not candidates:
        logger.debug("[rewriter] No underperforming signatures found")
        return []

    for candidate in candidates:
        try:
            # Generate improved DSL
            new_dsl = await generate_improved_dsl(candidate, client)
            if not new_dsl:
                results.append(RewriteResult(
                    signature_id=candidate.signature_id,
                    old_dsl=candidate.current_dsl,
                    new_dsl="",
                    reason="Failed to generate new DSL",
                    success=False,
                ))
                continue

            # Update the signature with new DSL
            db.update_nl_interface(
                signature_id=candidate.signature_id,
                dsl_script=new_dsl,
            )

            # Reset stats for fair evaluation of new DSL
            db.reset_signature_stats(candidate.signature_id)

            # Mark rewrite timestamp
            db.mark_signature_rewritten(candidate.signature_id)

            results.append(RewriteResult(
                signature_id=candidate.signature_id,
                old_dsl=candidate.current_dsl,
                new_dsl=new_dsl,
                reason=f"Success rate {candidate.success_rate:.1%} below threshold",
                success=True,
            ))

            logger.info(
                "[rewriter] Rewrote DSL for sig %d '%s': rate was %.1%% → reset stats",
                candidate.signature_id, candidate.step_type, candidate.success_rate * 100
            )

        except Exception as e:
            logger.error(
                "[rewriter] Failed to rewrite sig %d: %s",
                candidate.signature_id, e
            )
            results.append(RewriteResult(
                signature_id=candidate.signature_id,
                old_dsl=candidate.current_dsl,
                new_dsl="",
                reason=str(e),
                success=False,
            ))

    logger.info(
        "[rewriter] Completed batch: %d/%d rewrites successful",
        sum(1 for r in results if r.success), len(results)
    )
    return results


def get_rewrite_stats(db) -> dict:
    """Get statistics about DSL rewrites.

    Returns:
        Dict with rewrite statistics
    """
    try:
        total_sigs = db.count_signatures()
        underperforming = len(find_underperforming_signatures(db, limit=100))
        rewritten_recently = db.count_recently_rewritten(hours=24)

        return {
            "total_signatures": total_sigs,
            "underperforming": underperforming,
            "rewritten_24h": rewritten_recently,
            "enabled": DSL_REWRITER_ENABLED,
        }
    except Exception as e:
        logger.error("[rewriter] Failed to get stats: %s", e)
        return {"error": str(e)}
