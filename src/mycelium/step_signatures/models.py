"""Step Signature Models: Core dataclasses for step-level signature matching.

This module defines the primary data structures used throughout the step
signature system:

- StepSignature: A reusable pattern for solving a type of subproblem
- StepExample: An example step that belongs to a signature cluster
- PendingStepExample: A step in superposition (not yet assigned to a cluster)
- SignatureRelationship: A relationship between two signatures
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from mycelium.config import (
    RELIABILITY_MIN_USES,
    RELIABILITY_MIN_SUCCESS_RATE,
    EXPLORATION_MIN_LIFT,
    EXPLORATION_MIN_CONFIDENCE,
    EXPLORATION_RATE,
    EXPLORATION_UNPROVEN_RATE,
    USAGE_CONFIDENCE_DECAY,
    COLD_START_GUARANTEED_USES,
)

# Import schema types
from mycelium.step_signatures.schema import (
    StepIOSchema,
    get_default_schema,
)


# Plan types determine how the instruction should be applied
PlanType = Literal["formula", "procedure", "pattern", "composite"]


@dataclass
class StepSignature:
    """A signature for a type of step/subproblem.

    Each signature represents a cluster of similar steps that can be
    solved with the same method. Identity is determined by embedding
    similarity (centroid matching), not string hashing.

    Wave Properties:
        Signatures have wave-like properties enabling interference-based matching:
        - amplitude: Signal strength derived from success rate and usage
        - phase: Wave phase for interference patterns (learnable)
        - spread: Controls decay in embedding space (Gaussian envelope)

        The interference score replaces binary similarity thresholding:
        interference = cosine_sim * amplitude * exp(-d^2/spread^2) * cos(phase_diff)

    Canonical Status:
        A signature is "canonical" if it represents a well-established pattern:
        - High example count (cluster is mature)
        - High cohesion (examples cluster tightly)
        - Proven reliability (high success rate)

    Variants:
        Signatures can be linked as variants of a canonical pattern.
        This helps track related solution patterns and enables knowledge
        transfer between similar problem types.
    """
    id: Optional[int] = None
    signature_id: str = ""  # UUID for stable reference (replaces brittle hash fingerprint)

    # Cluster centroid (average embedding of all examples in cluster)
    centroid: Optional[np.ndarray] = None

    # What kind of step this is
    step_type: str = ""  # e.g., "solve_linear_equation", "compute_percentage"
    description: str = ""

    # Solution method
    method_name: str = ""
    method_template: str = ""  # Template for solving this type of step

    # Example problems in this cluster
    example_count: int = 0

    # Statistics
    uses: int = 0
    successes: int = 0

    # Lift tracking: compare injected vs non-injected outcomes
    # This enables lift-gated injection - only inject when it demonstrably helps
    injected_uses: int = 0
    injected_successes: int = 0
    non_injected_uses: int = 0
    non_injected_successes: int = 0

    # Cluster quality metrics
    cohesion: float = 0.0  # Average similarity of examples to centroid (0-1, higher = tighter)

    # Wave properties for interference-based matching
    amplitude: float = 0.5  # Signal strength (derived from success_rate * usage_factor)
    phase: float = 0.0  # Wave phase in [0, 2pi), initialized randomly, later learnable
    spread: float = 0.3  # Gaussian spread in embedding space (controls locality)

    # Canonical pattern tracking
    is_canonical: bool = False  # True if this is an established canonical pattern
    canonical_parent_id: Optional[int] = None  # If variant, points to canonical parent
    variant_count: int = 0  # Number of variants derived from this signature (if canonical)

    # Decomposition tracking
    origin_depth: int = 0  # Decomposition depth at which this signature was created
    is_atomic: bool = False  # True if this signature represents an atomic operation

    # Metadata
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None

    # I/O Schema for standardized input/output
    io_schema: Optional[StepIOSchema] = None

    # DSL script for deterministic execution (JSON-encoded DSLSpec)
    # Format: {"type": "math|sympy|custom", "script": "...", "params": [...], "fallback": "guidance"}
    dsl_script: Optional[str] = None

    # DSL versioning: track when DSL is improved so lift stats can be reset
    # New DSL version = fresh lift stats (don't penalize new DSL with old failures)
    dsl_version: int = 1
    dsl_version_uses: int = 0  # Uses since last DSL version update

    # Execution plan: optimized injection instruction (computed once, used forever)
    # If compressed_instruction is None, falls back to method_template
    plan_type: Optional[PlanType] = None  # 'formula', 'procedure', 'pattern', 'composite'
    compressed_instruction: Optional[str] = None  # Minimal, LLM-optimized instruction
    param_schema: Optional[dict] = None  # Parameter extraction hints
    output_format: Optional[str] = None  # Expected output format hint
    plan_optimization_method: Optional[str] = None  # How the plan was optimized
    plan_tokens_before: Optional[int] = None  # Original method_template token count
    plan_tokens_after: Optional[int] = None  # Compressed instruction token count
    plan_validation_accuracy: Optional[float] = None  # Accuracy during optimization

    @property
    def has_optimized_plan(self) -> bool:
        """Check if this signature has an optimized execution plan."""
        return self.compressed_instruction is not None

    @property
    def effective_instruction(self) -> str:
        """Get the best available instruction (compressed if available, else template)."""
        return self.compressed_instruction or self.method_template

    @property
    def token_savings_pct(self) -> Optional[float]:
        """Calculate token savings percentage from optimization."""
        if self.plan_tokens_before and self.plan_tokens_after:
            return 100.0 * (self.plan_tokens_before - self.plan_tokens_after) / self.plan_tokens_before
        return None

    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses > 0 else 0.0

    @property
    def is_reliable(self) -> bool:
        """Reliable if meets minimum uses and success rate thresholds."""
        return (self.uses >= RELIABILITY_MIN_USES and
                self.success_rate >= RELIABILITY_MIN_SUCCESS_RATE)

    def compute_amplitude(self) -> float:
        """Compute amplitude from success rate and usage.

        Amplitude increases with:
        - Higher success rate (more reliable)
        - More uses (more confident)

        Formula: amplitude = success_rate * (1 - exp(-uses/5))
        The usage factor saturates around 10-15 uses.
        """
        if self.uses == 0:
            return 0.1  # Small base amplitude for new signatures
        usage_factor = 1.0 - np.exp(-self.uses / USAGE_CONFIDENCE_DECAY)
        return float(self.success_rate * usage_factor + 0.1)  # Min 0.1

    def update_amplitude(self) -> None:
        """Update amplitude based on current statistics."""
        self.amplitude = self.compute_amplitude()

    def get_io_schema(self) -> StepIOSchema:
        """Get the I/O schema, falling back to default for step_type."""
        if self.io_schema:
            return self.io_schema
        return get_default_schema(self.step_type) or StepIOSchema()

    def get_effective_method_template(self) -> str:
        """Get the best method template for injection.

        Priority order:
        1. compressed_instruction (if optimized plan exists)
        2. method_template_v2 from I/O schema
        3. method_template (original)
        """
        # Prefer optimized compressed instruction if available
        if self.compressed_instruction:
            return self.compressed_instruction

        # Fall back to v2 schema template if available
        schema = self.get_io_schema()
        if schema.method_template_v2:
            return schema.method_template_v2

        return self.method_template

    @property
    def canonical_score(self) -> float:
        """Score indicating how canonical this pattern is (0-1).

        Higher scores mean the pattern is more established and trustworthy.
        Based on: example count, cohesion, and reliability.
        """
        if self.example_count == 0:
            return 0.0

        # Normalize example count (caps at 10 for full score)
        example_score = min(self.example_count / 10.0, 1.0)

        # Cohesion score (already 0-1)
        cohesion_score = self.cohesion

        # Reliability score
        reliability_score = self.success_rate if self.uses >= 3 else 0.0

        # Weighted combination (cohesion most important for canonical status)
        return 0.3 * example_score + 0.4 * cohesion_score + 0.3 * reliability_score

    # =========================================================================
    # Lift-Gated Injection: Only inject when it demonstrably helps
    # =========================================================================

    @property
    def injected_success_rate(self) -> float:
        """Success rate when this signature was injected."""
        return self.injected_successes / self.injected_uses if self.injected_uses > 0 else 0.0

    @property
    def non_injected_success_rate(self) -> float:
        """Success rate when this signature was NOT injected (baseline)."""
        return self.non_injected_successes / self.non_injected_uses if self.non_injected_uses > 0 else 0.0

    def compute_lift(self) -> tuple[float, float]:
        """Compute lift from injection vs non-injection.

        Lift = (injected_success_rate - non_injected_success_rate)

        Returns:
            (lift, confidence) where:
            - lift: Success rate improvement from injection (-1 to +1)
            - confidence: Statistical confidence in the lift estimate (0 to 1)
              Based on sample size using Wilson score interval width
        """
        if self.injected_uses == 0 or self.non_injected_uses == 0:
            return 0.0, 0.0

        lift = self.injected_success_rate - self.non_injected_success_rate

        # Confidence based on sample sizes (harmonic mean of sample sizes)
        # More samples = higher confidence
        min_samples = min(self.injected_uses, self.non_injected_uses)
        # Confidence saturates around 10+ samples per arm
        confidence = 1.0 - np.exp(-min_samples / USAGE_CONFIDENCE_DECAY)

        return float(lift), float(confidence)

    def should_inject(
        self,
        min_lift: float = EXPLORATION_MIN_LIFT,
        min_confidence: float = EXPLORATION_MIN_CONFIDENCE,
        min_samples_per_arm: int = 3,
        exploration_rate: float = EXPLORATION_RATE,
        unproven_exploration_rate: float = EXPLORATION_UNPROVEN_RATE,
        cold_start_guaranteed_uses: int = COLD_START_GUARANTEED_USES,
    ) -> bool:
        """Determine if this signature should be injected based on lift.

        Injection is gated on:
        1. Cold-start bootstrap: ALWAYS inject for first N uses (default 10, configurable)
        2. After bootstrap: positive lift with sufficient confidence

        The cold-start bootstrap solves the catch-22 where new signatures can't
        prove themselves if we never inject them. By guaranteeing injection for
        the first N uses, every signature gets a fair chance to sample success rate.

        Args:
            min_lift: Minimum lift required (default 0.0 = any positive lift)
            min_confidence: Minimum confidence required (default 0.5)
            min_samples_per_arm: Minimum samples needed in each arm (default 3)
            exploration_rate: Probability of injecting during cold-start for reliable sigs (default 0.5)
            unproven_exploration_rate: Probability of injecting for unproven sigs in cold-start (default 0.3)
            cold_start_guaranteed_uses: Always inject for first N uses (default 15)

        Returns:
            True if signature should be injected, False otherwise
        """
        # Cold-start bootstrap: ALWAYS inject for first N uses to solve catch-22
        # New signatures can't prove themselves if we never use them
        if self.uses < cold_start_guaranteed_uses:
            return True

        # After bootstrap: need enough data in both arms to compute lift
        is_cold_start = (self.injected_uses < min_samples_per_arm or
                         self.non_injected_uses < min_samples_per_arm)

        if is_cold_start:
            # Use exploration to bootstrap lift data
            # Reliable signatures get higher exploration rate since we trust the method
            # Unproven signatures get lower rate to be more conservative
            rate = exploration_rate if self.is_reliable else unproven_exploration_rate
            return np.random.random() < rate

        # After cold-start: must be reliable AND have positive lift
        if not self.is_reliable:
            return False

        lift, confidence = self.compute_lift()

        # Gate on positive lift with sufficient confidence
        return lift >= min_lift and confidence >= min_confidence

    @property
    def lift_status(self) -> str:
        """Human-readable lift status for debugging."""
        if self.injected_uses == 0 and self.non_injected_uses == 0:
            return "no_data"
        if self.injected_uses == 0:
            return f"baseline_only({self.non_injected_uses})"
        if self.non_injected_uses == 0:
            return f"injected_only({self.injected_uses})"

        lift, confidence = self.compute_lift()
        inj_rate = self.injected_success_rate
        base_rate = self.non_injected_success_rate

        return (
            f"lift={lift:+.1%} conf={confidence:.0%} "
            f"(inj:{inj_rate:.0%}@{self.injected_uses} vs base:{base_rate:.0%}@{self.non_injected_uses})"
        )


@dataclass
class StepExample:
    """An example step that belongs to a signature cluster."""
    id: Optional[int] = None
    signature_id: int = 0

    step_text: str = ""  # The step description
    embedding: Optional[np.ndarray] = None

    # Solution trace
    result: str = ""
    success: bool = False

    # Context
    parent_problem: str = ""  # Original problem this step came from

    created_at: Optional[str] = None


@dataclass
class PendingStepExample:
    """A step in superposition - not yet assigned to a cluster.

    Wave function metaphor:
    - Before observation: step has probability distribution over possible clusters
    - Observation (execution with known success/failure) partially collapses the wave
    - Full collapse: assignment to a signature (existing or newly created)

    A pending step remains in superposition until:
    1. It matches an existing cluster with high similarity (>= COLLAPSE_THRESHOLD) AND succeeds
    2. Enough similar pending steps accumulate to form a new cluster
    """
    id: Optional[int] = None

    step_text: str = ""
    embedding: Optional[np.ndarray] = None

    # Observation state
    executed: bool = False
    success: Optional[bool] = None  # None = not observed yet
    result: str = ""

    # Probability distribution over clusters
    # List of (signature_id, similarity) tuples, sorted by similarity desc
    cluster_probabilities: list[tuple[int, float]] = field(default_factory=list)

    # Best match for quick access
    best_match_signature_id: Optional[int] = None
    best_match_similarity: float = 0.0

    # Context
    parent_problem: str = ""

    # Lifecycle timestamps
    created_at: Optional[str] = None
    observed_at: Optional[str] = None  # When success/failure was recorded
    collapsed_at: Optional[str] = None  # When assigned to signature
    collapsed_to_signature_id: Optional[int] = None

    @property
    def is_observed(self) -> bool:
        """Has this step been executed and outcome recorded?"""
        return self.success is not None

    @property
    def is_collapsed(self) -> bool:
        """Has this step been assigned to a signature?"""
        return self.collapsed_to_signature_id is not None

    @property
    def in_superposition(self) -> bool:
        """Is this step still in superposition (not collapsed)?"""
        return not self.is_collapsed


@dataclass
class SignatureRelationship:
    """A relationship between two signatures in the mycelium network."""

    id: Optional[int]
    source_signature_id: str  # UUID of source signature
    target_signature_id: str  # UUID of target signature
    relationship_type: str  # e.g., "similar_to", "prerequisite", "conflicts"
    strength: float  # Relationship strength (0-1)
    bidirectional: bool  # True if A->B implies B->A
    uses: int  # Times this relationship was traversed
    successes: int  # Successful traversals
    metadata: Optional[dict]  # Additional context
    created_at: Optional[str]
    last_used_at: Optional[str]

    @property
    def success_rate(self) -> Optional[float]:
        """Success rate for this relationship."""
        if self.uses == 0:
            return None
        return self.successes / self.uses
