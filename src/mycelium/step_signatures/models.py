"""Step Signature Models - prototypes for 200-class function classification.

Each signature is a prototype that maps a semantic description to a function.
Multiple signatures can map to the same function (semantic variants).

Key fields:
- centroid: Learned embedding (prototype vector)
- func_name: Key into function_registry (~200 functions)
- description: Natural language description (few-shot example for LLM)
- successes/uses: Track success rate for ranking

Hierarchy fields (is_semantic_umbrella, is_root, depth) are kept for
backward compatibility but are not used in the flat prototype architecture.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _parse_centroid_data(data: Union[str, bytes, None]) -> Optional[np.ndarray]:
    """Parse centroid data (JSON string or binary bytes)."""
    if data is None:
        return None
    if isinstance(data, str):
        return np.array(json.loads(data), dtype=np.float32)
    return np.frombuffer(data, dtype=np.float32)


@dataclass
class StepSignature:
    """A step signature for routing and function execution."""
    id: Optional[int] = None
    signature_id: str = ""
    centroid: Optional[np.ndarray] = None
    embedding_sum: Optional[np.ndarray] = None
    embedding_count: int = 1
    step_type: str = ""
    description: str = ""
    clarifying_questions: list[str] = field(default_factory=list)
    param_descriptions: dict[str, str] = field(default_factory=dict)
    func_name: Optional[str] = None  # Key into function_registry
    func_arity: int = 2              # Expected number of arguments
    computation_graph: Optional[str] = None
    graph_embedding: Optional[np.ndarray] = None
    examples: list[dict] = field(default_factory=list)
    uses: int = 0
    successes: int = 0
    operational_failures: int = 0
    similarity_count: int = 0
    similarity_mean: float = 0.0
    similarity_m2: float = 0.0
    success_sim_count: int = 0
    success_sim_mean: float = 0.0
    success_sim_m2: float = 0.0
    # Merge tracking
    description_variants: list[str] = field(default_factory=list)
    merge_dist_count: int = 0
    merge_dist_mean: float = 0.0
    merge_dist_m2: float = 0.0
    # Coverage tracking - Welford stats for similarity scores
    coverage_sim_count: int = 0
    coverage_sim_mean: float = 0.0
    coverage_sim_m2: float = 0.0
    low_coverage_count: int = 0
    is_semantic_umbrella: bool = False
    is_root: bool = False
    depth: int = 0
    is_atomic: bool = False
    atomic_reason: Optional[str] = None
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None

    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses > 0 else 0.0

    @property
    def has_func(self) -> bool:
        return self.func_name is not None

    @classmethod
    def from_row(cls, row: dict) -> "StepSignature":
        """Create from database row."""
        sig_id = row.get("id", "?")

        # Parse JSON fields
        clarifying_questions = []
        if row.get("clarifying_questions"):
            try:
                clarifying_questions = json.loads(row["clarifying_questions"])
            except json.JSONDecodeError:
                pass

        param_descriptions = {}
        if row.get("param_descriptions"):
            try:
                param_descriptions = json.loads(row["param_descriptions"])
            except json.JSONDecodeError:
                pass

        examples = []
        if row.get("examples"):
            try:
                examples = json.loads(row["examples"])
            except json.JSONDecodeError:
                pass

        description_variants = []
        if row.get("description_variants"):
            try:
                description_variants = json.loads(row["description_variants"])
            except json.JSONDecodeError:
                pass

        centroid = None
        if row.get("centroid"):
            try:
                centroid = _parse_centroid_data(row["centroid"])
            except Exception:
                pass

        embedding_sum = None
        if row.get("embedding_sum"):
            try:
                embedding_sum = _parse_centroid_data(row["embedding_sum"])
            except Exception:
                pass

        graph_embedding = None
        if row.get("graph_embedding"):
            try:
                graph_embedding = np.array(json.loads(row["graph_embedding"]))
            except (json.JSONDecodeError, TypeError):
                pass

        # Handle func_name: use new column if present, fallback to None
        func_name = row.get("func_name")

        # Handle func_arity: use new column if present, default to 2
        func_arity = row.get("func_arity", 2) or 2

        return cls(
            id=row.get("id"),
            signature_id=row.get("signature_id", ""),
            centroid=centroid,
            embedding_sum=embedding_sum,
            embedding_count=row.get("embedding_count", 1) or 1,
            step_type=row.get("step_type", ""),
            description=row.get("description", ""),
            clarifying_questions=clarifying_questions,
            param_descriptions=param_descriptions,
            func_name=func_name,
            func_arity=func_arity,
            computation_graph=row.get("computation_graph"),
            graph_embedding=graph_embedding,
            examples=examples,
            uses=row.get("uses", 0),
            successes=row.get("successes", 0),
            operational_failures=row.get("operational_failures", 0) or 0,
            similarity_count=row.get("similarity_count", 0) or 0,
            similarity_mean=row.get("similarity_mean", 0.0) or 0.0,
            similarity_m2=row.get("similarity_m2", 0.0) or 0.0,
            success_sim_count=row.get("success_sim_count", 0) or 0,
            success_sim_mean=row.get("success_sim_mean", 0.0) or 0.0,
            success_sim_m2=row.get("success_sim_m2", 0.0) or 0.0,
            description_variants=description_variants,
            merge_dist_count=row.get("merge_dist_count", 0) or 0,
            merge_dist_mean=row.get("merge_dist_mean", 0.0) or 0.0,
            merge_dist_m2=row.get("merge_dist_m2", 0.0) or 0.0,
            coverage_sim_count=row.get("coverage_sim_count", 0) or 0,
            coverage_sim_mean=row.get("coverage_sim_mean", 0.0) or 0.0,
            coverage_sim_m2=row.get("coverage_sim_m2", 0.0) or 0.0,
            low_coverage_count=row.get("low_coverage_count", 0) or 0,
            is_semantic_umbrella=bool(row.get("is_semantic_umbrella", 0)),
            is_root=bool(row.get("is_root", 0)),
            depth=row.get("depth", 0) or 0,
            is_atomic=bool(row.get("is_atomic", 0)),
            atomic_reason=row.get("atomic_reason"),
            created_at=row.get("created_at"),
            last_used_at=row.get("last_used_at"),
        )

    @classmethod
    def from_row_for_routing(cls, row: dict) -> "StepSignature":
        """Create from database row optimized for routing (parse centroid only)."""
        from mycelium.step_signatures.utils import get_cached_centroid

        centroid = get_cached_centroid(row.get("id"), row.get("centroid"))

        graph_embedding = None
        if row.get("graph_embedding"):
            try:
                graph_embedding = np.array(json.loads(row["graph_embedding"]))
            except (json.JSONDecodeError, TypeError):
                pass

        # Handle func_name: use new column if present, fallback to None
        func_name = row.get("func_name")

        # Handle func_arity: use new column if present, default to 2
        func_arity = row.get("func_arity", 2) or 2

        return cls(
            id=row.get("id"),
            signature_id=row.get("signature_id", ""),
            centroid=centroid,
            embedding_count=row.get("embedding_count", 1) or 1,
            step_type=row.get("step_type", ""),
            description=row.get("description", ""),
            func_name=func_name,
            func_arity=func_arity,
            computation_graph=row.get("computation_graph"),
            graph_embedding=graph_embedding,
            uses=row.get("uses", 0),
            successes=row.get("successes", 0),
            operational_failures=row.get("operational_failures", 0) or 0,
            similarity_count=row.get("similarity_count", 0) or 0,
            similarity_mean=row.get("similarity_mean", 0.0) or 0.0,
            similarity_m2=row.get("similarity_m2", 0.0) or 0.0,
            success_sim_count=row.get("success_sim_count", 0) or 0,
            success_sim_mean=row.get("success_sim_mean", 0.0) or 0.0,
            success_sim_m2=row.get("success_sim_m2", 0.0) or 0.0,
            coverage_sim_count=row.get("coverage_sim_count", 0) or 0,
            coverage_sim_mean=row.get("coverage_sim_mean", 0.0) or 0.0,
            coverage_sim_m2=row.get("coverage_sim_m2", 0.0) or 0.0,
            low_coverage_count=row.get("low_coverage_count", 0) or 0,
            is_semantic_umbrella=bool(row.get("is_semantic_umbrella", 0)),
            is_root=bool(row.get("is_root", 0)),
            depth=row.get("depth", 0) or 0,
            is_atomic=bool(row.get("is_atomic", 0)),
            atomic_reason=row.get("atomic_reason"),
            created_at=row.get("created_at"),
            last_used_at=row.get("last_used_at"),
        )
