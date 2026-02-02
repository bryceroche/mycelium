"""Example coverage management for pattern matching.

Ensures examples span the embedding space without redundant clustering.
Uses Welford-based adaptive thresholds (per CLAUDE.md "The Flow").
"""
import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from mycelium.embedding_cache import cached_embed
from mycelium.config import MIN_MATCH_THRESHOLD

logger = logging.getLogger(__name__)

# Default minimum cosine distance for a new example to be added
# (1 - similarity), so 0.15 means similarity must be < 0.85
# This is overridden by Welford-based adaptive thresholds when available
MIN_DISTANCE_THRESHOLD = 1.0 - MIN_MATCH_THRESHOLD  # 0.15


@dataclass
class ExampleProposal:
    """A proposed new example for a pattern."""
    problem_text: str
    pattern_name: str
    embedding: np.ndarray
    similarity_to_nearest: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "pending"  # pending, approved, rejected


# In-memory storage (could be persisted to DB)
_proposals: List[ExampleProposal] = []


def check_coverage(
    problem_embedding: np.ndarray,
    existing_embeddings: List[np.ndarray],
    threshold: float = MIN_DISTANCE_THRESHOLD
) -> Tuple[bool, float]:
    """
    Check if a new example provides coverage (is far enough from existing).

    Args:
        problem_embedding: Embedding of the candidate example
        existing_embeddings: List of existing example embeddings
        threshold: Minimum distance (1 - similarity) required

    Returns:
        Tuple of (provides_coverage, nearest_similarity)
    """
    if not existing_embeddings:
        return True, 0.0

    # Normalize
    problem_norm = problem_embedding / (np.linalg.norm(problem_embedding) + 1e-9)

    max_similarity = 0.0
    for existing in existing_embeddings:
        existing_norm = existing / (np.linalg.norm(existing) + 1e-9)
        similarity = float(np.dot(problem_norm, existing_norm))
        max_similarity = max(max_similarity, similarity)

    distance = 1 - max_similarity
    provides_coverage = distance >= threshold

    return provides_coverage, max_similarity


def propose_example(
    problem_text: str,
    pattern_name: str,
    similarity: float,
    threshold: Optional[float] = None,
    was_correct: bool = True,
    example_id: Optional[str] = None
) -> Optional[ExampleProposal]:
    """
    Propose a new example if it provides coverage.

    Uses Welford-based adaptive thresholds when available.

    Args:
        problem_text: The problem that was solved correctly
        pattern_name: The pattern that was used
        similarity: Similarity to nearest existing example
        threshold: Override threshold (uses adaptive if None)
        was_correct: Whether the answer was correct
        example_id: Optional identifier for the matched example/signature
                    (for per-example Welford tracking)

    Returns:
        ExampleProposal if proposed, None if too similar to existing
    """
    # Import here to avoid circular dependency
    from mycelium.patterns.welford import record_similarity, get_adaptive_threshold, record_example_match

    # Record observation for Welford stats
    problem_hash = str(hash(problem_text))
    record_similarity(pattern_name, similarity, was_correct, problem_hash)

    # Record per-example stats if example_id provided (two-signal variance tracking)
    if example_id is not None:
        record_example_match(example_id, pattern_name, similarity, was_correct)

    # Get threshold (adaptive or override)
    if threshold is None:
        threshold = get_adaptive_threshold(pattern_name)

    if similarity >= threshold:
        logger.debug(f"[coverage] Not proposing - similarity {similarity:.3f} >= {threshold:.3f}")
        return None

    embedding = cached_embed(problem_text)
    if embedding is None:
        return None

    proposal = ExampleProposal(
        problem_text=problem_text,
        pattern_name=pattern_name,
        embedding=embedding,
        similarity_to_nearest=similarity,
    )

    _proposals.append(proposal)
    logger.info(f"[coverage] Proposed new example for '{pattern_name}' (sim={similarity:.3f}, thresh={threshold:.3f})")

    return proposal


def get_proposals(status: str = "pending") -> List[ExampleProposal]:
    """Get proposals by status."""
    return [p for p in _proposals if p.status == status]


def approve_proposal(proposal: ExampleProposal) -> None:
    """Approve a proposal (would add to pattern's examples)."""
    proposal.status = "approved"
    logger.info(f"[coverage] Approved example for '{proposal.pattern_name}'")


def reject_proposal(proposal: ExampleProposal) -> None:
    """Reject a proposal."""
    proposal.status = "rejected"


def clear_proposals() -> None:
    """Clear all proposals."""
    global _proposals
    _proposals = []


def compute_coverage_stats(pattern_embeddings: dict) -> dict:
    """
    Compute coverage statistics for all patterns.

    Returns dict with:
    - total_examples: Total number of examples
    - avg_intra_pattern_distance: Average distance within patterns
    - min_inter_example_distance: Minimum distance between any two examples
    """
    all_embeddings = []
    pattern_distances = []

    for pattern_name, examples in pattern_embeddings.items():
        embeddings = [e[1] for e in examples]  # (text, embedding) tuples
        all_embeddings.extend(embeddings)

        # Compute intra-pattern distances
        if len(embeddings) > 1:
            for i, e1 in enumerate(embeddings):
                for e2 in embeddings[i+1:]:
                    e1_norm = e1 / (np.linalg.norm(e1) + 1e-9)
                    e2_norm = e2 / (np.linalg.norm(e2) + 1e-9)
                    sim = float(np.dot(e1_norm, e2_norm))
                    pattern_distances.append(1 - sim)

    # Compute minimum distance between any two examples
    min_distance = float('inf')
    for i, e1 in enumerate(all_embeddings):
        for e2 in all_embeddings[i+1:]:
            e1_norm = e1 / (np.linalg.norm(e1) + 1e-9)
            e2_norm = e2 / (np.linalg.norm(e2) + 1e-9)
            sim = float(np.dot(e1_norm, e2_norm))
            min_distance = min(min_distance, 1 - sim)

    return {
        "total_examples": len(all_embeddings),
        "avg_intra_pattern_distance": np.mean(pattern_distances) if pattern_distances else 0,
        "min_inter_example_distance": min_distance if min_distance != float('inf') else 0,
    }
