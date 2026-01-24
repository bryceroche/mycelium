"""Divergence-based signature splitting.

Inspired by nature:
- Nautilus: depth from accumulated growth
- Trees: branching follows resource/traffic gradients
- Lungs: depth from optimization under constraint
- Visual cortex: depth from abstraction levels

Key principles:
1. Binary split is the atomic operation (like cell division)
2. Split on DIVERGENCE (success vs failure clusters)
3. WIDTH vs DEPTH based on semantic distance:
   - Close embeddings but divergent outcomes -> WIDTH (variants)
   - Distant embeddings with divergent outcomes -> DEPTH (abstraction)
4. The tree structure EMERGES, not designed
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from mycelium.step_signatures.models import StepSignature

logger = logging.getLogger(__name__)

# Thresholds for width vs depth decision
# Close = same abstraction level, different variants
# Distant = different abstraction levels
CLOSE_DISTANCE_THRESHOLD = 0.20  # cosine distance < 0.20 = close (similarity > 0.80)
MIN_SAMPLES_FOR_SPLIT = 2  # Need at least 2 samples to detect divergence


@dataclass
class DivergenceResult:
    """Result of divergence detection."""
    has_divergence: bool
    success_centroid: Optional[np.ndarray] = None
    failure_centroid: Optional[np.ndarray] = None
    distance: float = 0.0  # Cosine distance between clusters
    n_successes: int = 0
    n_failures: int = 0
    split_type: str = ""  # "width", "depth", or ""


@dataclass
class SplitResult:
    """Result of a binary split operation."""
    success: bool
    parent_id: int
    child_a_id: Optional[int] = None  # Success cluster child
    child_b_id: Optional[int] = None  # Failure cluster child
    split_type: str = ""  # "width" or "depth"
    reason: str = ""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity)."""
    return 1.0 - cosine_similarity(a, b)


def compute_centroid(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    """Compute centroid of a list of embeddings."""
    if not embeddings:
        return None
    stacked = np.stack(embeddings)
    centroid = np.mean(stacked, axis=0)
    # Normalize for cosine similarity
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


def detect_divergence(
    success_embeddings: List[np.ndarray],
    failure_embeddings: List[np.ndarray],
) -> DivergenceResult:
    """Detect if there's divergence between success and failure cases.

    Divergence = the signature is being used for semantically different things
    that have different outcomes. This is the signal to split.

    Args:
        success_embeddings: Embeddings of problems that succeeded with this signature
        failure_embeddings: Embeddings of problems that failed with this signature

    Returns:
        DivergenceResult with divergence info and recommended split type
    """
    n_successes = len(success_embeddings)
    n_failures = len(failure_embeddings)

    # Need both successes and failures to detect divergence
    if n_successes == 0 or n_failures == 0:
        return DivergenceResult(
            has_divergence=False,
            n_successes=n_successes,
            n_failures=n_failures,
        )

    # Need minimum samples for reliable divergence detection
    if n_successes + n_failures < MIN_SAMPLES_FOR_SPLIT:
        return DivergenceResult(
            has_divergence=False,
            n_successes=n_successes,
            n_failures=n_failures,
        )

    # Compute cluster centroids
    success_centroid = compute_centroid(success_embeddings)
    failure_centroid = compute_centroid(failure_embeddings)

    if success_centroid is None or failure_centroid is None:
        return DivergenceResult(
            has_divergence=False,
            n_successes=n_successes,
            n_failures=n_failures,
        )

    # Measure semantic distance between clusters
    distance = cosine_distance(success_centroid, failure_centroid)

    # Determine split type based on distance
    # Close but divergent = variants at same abstraction level (WIDTH)
    # Distant and divergent = different abstraction levels (DEPTH)
    if distance < CLOSE_DISTANCE_THRESHOLD:
        split_type = "width"
    else:
        split_type = "depth"

    logger.info(
        "[divergence] Detected divergence: distance=%.3f, type=%s, "
        "successes=%d, failures=%d",
        distance, split_type, n_successes, n_failures
    )

    return DivergenceResult(
        has_divergence=True,
        success_centroid=success_centroid,
        failure_centroid=failure_centroid,
        distance=distance,
        n_successes=n_successes,
        n_failures=n_failures,
        split_type=split_type,
    )


def binary_split_wide(
    db,  # StepSignatureDB
    parent_sig: StepSignature,
    success_centroid: np.ndarray,
    failure_centroid: np.ndarray,
) -> SplitResult:
    """Create siblings at same level (width split).

    Both children are at parent.depth + 1.
    This is for when success and failure cases are semantically close
    but have different outcomes - variants of the same operation type.

    Args:
        db: StepSignatureDB instance
        parent_sig: The signature to split
        success_centroid: Centroid of successful cases
        failure_centroid: Centroid of failed cases

    Returns:
        SplitResult with child IDs
    """
    try:
        # Create child for success cluster
        child_a = db.create_signature(
            step_text=f"{parent_sig.description} (variant A)",
            embedding=success_centroid,
            parent_id=parent_sig.id,
            origin_depth=parent_sig.depth + 1,
        )

        # Create child for failure cluster
        child_b = db.create_signature(
            step_text=f"{parent_sig.description} (variant B)",
            embedding=failure_centroid,
            parent_id=parent_sig.id,
            origin_depth=parent_sig.depth + 1,
        )

        # Promote parent to umbrella (router)
        db.promote_to_umbrella(parent_sig.id)

        logger.info(
            "[divergence] Width split: parent %d -> children %d, %d (depth %d)",
            parent_sig.id, child_a.id, child_b.id, parent_sig.depth + 1
        )

        return SplitResult(
            success=True,
            parent_id=parent_sig.id,
            child_a_id=child_a.id,
            child_b_id=child_b.id,
            split_type="width",
            reason=f"Close divergence at depth {parent_sig.depth}",
        )

    except Exception as e:
        logger.error("[divergence] Width split failed: %s", e)
        return SplitResult(
            success=False,
            parent_id=parent_sig.id,
            reason=str(e),
        )


def binary_split_deep(
    db,  # StepSignatureDB
    parent_sig: StepSignature,
    success_centroid: np.ndarray,
    failure_centroid: np.ndarray,
) -> SplitResult:
    """Create depth by inserting intermediate umbrella.

    The failure cluster is so semantically different that it needs
    its own branch of the tree. We create an umbrella above the parent
    to capture the new abstraction level.

    Structure:
        Before: parent (leaf)
        After:  umbrella -> parent (for successes)
                        -> new_child (for failures)

    Args:
        db: StepSignatureDB instance
        parent_sig: The signature that needs a new branch
        success_centroid: Centroid of successful cases (stays with parent)
        failure_centroid: Centroid of failed cases (new branch)

    Returns:
        SplitResult with child IDs
    """
    try:
        # Create new umbrella above the parent
        # Umbrella centroid is average of both clusters
        umbrella_centroid = (success_centroid + failure_centroid) / 2
        umbrella_centroid = umbrella_centroid / np.linalg.norm(umbrella_centroid)

        umbrella = db.create_upward_umbrella(
            child_signature=parent_sig,
            problem_embedding=umbrella_centroid,
            difficulty=0.5,  # Neutral difficulty
            description=f"Umbrella for {parent_sig.description[:30]}",
        )

        if umbrella is None:
            return SplitResult(
                success=False,
                parent_id=parent_sig.id,
                reason="Failed to create upward umbrella",
            )

        # Create new child for failure cluster
        failure_child = db.create_signature(
            step_text=f"{parent_sig.description} (divergent)",
            embedding=failure_centroid,
            parent_id=umbrella.id,
            origin_depth=umbrella.depth + 1,
        )

        logger.info(
            "[divergence] Depth split: umbrella %d (d%d) -> parent %d, failure_child %d",
            umbrella.id, umbrella.depth, parent_sig.id, failure_child.id
        )

        return SplitResult(
            success=True,
            parent_id=umbrella.id,
            child_a_id=parent_sig.id,
            child_b_id=failure_child.id,
            split_type="depth",
            reason=f"Distant divergence, created umbrella at depth {umbrella.depth}",
        )

    except Exception as e:
        logger.error("[divergence] Depth split failed: %s", e)
        return SplitResult(
            success=False,
            parent_id=parent_sig.id,
            reason=str(e),
        )


def maybe_split_on_divergence(
    db,  # StepSignatureDB
    signature: StepSignature,
    success_embeddings: List[np.ndarray],
    failure_embeddings: List[np.ndarray],
) -> Optional[SplitResult]:
    """Check for divergence and split if detected.

    This is the main entry point for natural splitting.
    Call this after a signature has accumulated some success/failure data.

    Args:
        db: StepSignatureDB instance
        signature: The signature to potentially split
        success_embeddings: Embeddings of successful problems
        failure_embeddings: Embeddings of failed problems

    Returns:
        SplitResult if split occurred, None otherwise
    """
    # Skip if already an umbrella (routers don't execute)
    if signature.is_semantic_umbrella:
        return None

    # Detect divergence
    divergence = detect_divergence(success_embeddings, failure_embeddings)

    if not divergence.has_divergence:
        logger.debug(
            "[divergence] No divergence for sig %d: successes=%d, failures=%d",
            signature.id, divergence.n_successes, divergence.n_failures
        )
        return None

    # Split based on divergence type
    if divergence.split_type == "width":
        return binary_split_wide(
            db, signature,
            divergence.success_centroid,
            divergence.failure_centroid,
        )
    else:  # depth
        return binary_split_deep(
            db, signature,
            divergence.success_centroid,
            divergence.failure_centroid,
        )


def get_signature_outcome_embeddings(
    signature_id: int,
    limit: int = 100,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Get success and failure embeddings for a signature.

    Queries the database for problems that routed to this signature
    and their outcomes.

    Args:
        signature_id: The signature to get embeddings for
        limit: Maximum number of embeddings to retrieve per outcome

    Returns:
        Tuple of (success_embeddings, failure_embeddings)
    """
    from mycelium.data_layer import get_db
    from mycelium.step_signatures.utils import unpack_embedding

    conn = get_db()

    # Get embeddings from dag_step_embeddings table grouped by success/failure
    # node_id links to the signature that handled the step
    cursor = conn.execute(
        """
        SELECT e.embedding, e.success
        FROM dag_step_embeddings e
        WHERE e.node_id = ?
          AND e.success IS NOT NULL
          AND e.embedding IS NOT NULL
        ORDER BY e.id DESC
        LIMIT ?
        """,
        (signature_id, limit * 2),  # Get enough for both success and failure
    )

    success_embeddings = []
    failure_embeddings = []

    for row in cursor:
        embedding = unpack_embedding(row["embedding"])
        if embedding is None:
            continue

        if row["success"] == 1 and len(success_embeddings) < limit:
            success_embeddings.append(embedding)
        elif row["success"] == 0 and len(failure_embeddings) < limit:
            failure_embeddings.append(embedding)

    return success_embeddings, failure_embeddings
