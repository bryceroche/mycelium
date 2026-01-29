"""Local Decomposer stub - to be implemented with embedding-based decomposition.

This is the core of the new architecture:
- Embed problem text
- Compare with tree nodes (text + graph embeddings)
- Use residual signal to determine atomicity
- Split locally if not atomic
- Track failures for periodic review

TODO: Implement per beads mycelium-1b8w.4
"""

from dataclasses import dataclass
from typing import Optional

from mycelium.plan_models import Step, DAGPlan

@dataclass
class DecompositionResult:
    """Result of local decomposition."""
    steps: list[Step]
    coverage_scores: list[float]
    is_atomic: bool
    failure_reason: Optional[str] = None

def decompose_locally(problem: str, tree_vocabulary: list[str] = None) -> DecompositionResult:
    """Local decomposition using embeddings.

    TODO: Implement with:
    1. Embed problem
    2. Find best leaf matches
    3. Compute coverage/residual signal
    4. Split if coverage < threshold
    5. Recurse

    For now, raises NotImplementedError.
    """
    raise NotImplementedError("Local decomposition not yet implemented - see beads mycelium-1b8w.4")

async def decompose_problem(problem: str, client=None, tree_vocabulary: list[str] = None) -> Optional[DAGPlan]:
    """Stub for old interface - raises NotImplementedError."""
    raise NotImplementedError("Use decompose_locally() instead")
