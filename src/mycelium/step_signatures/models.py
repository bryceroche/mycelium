"""Step Signature Models V2: Natural Language Interface.

Signatures now speak natural language:
- description: What this signature does (for LLM understanding)
- clarifying_questions: Questions to ask to extract parameters
- param_descriptions: What each DSL parameter means in plain English
- examples: Few-shot examples of input → output

The planner and signatures can now "talk" to each other through text.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import logging

import numpy as np
from typing import Union

logger = logging.getLogger(__name__)


def _parse_centroid_data(data: Union[str, bytes, None]) -> Optional[np.ndarray]:
    """Parse centroid data which may be JSON string or binary bytes.

    SQLite stores centroids as JSON strings, but legacy code stored binary.
    This helper handles both formats.
    """
    if data is None:
        return None
    if isinstance(data, str):
        return np.array(json.loads(data), dtype=np.float32)
    # Binary numpy data
    return np.frombuffer(data, dtype=np.float32)


@dataclass
class StepSignature:
    """A signature that speaks natural language.

    The key insight: signatures need to communicate with LLMs, not just
    match embeddings. This means:

    1. description: Tells the LLM what this signature does
    2. clarifying_questions: Asks the LLM to extract specific parameters
    3. param_descriptions: Explains what each parameter means
    4. examples: Shows the LLM how to use this signature

    Example:
        signature = StepSignature(
            step_type="compute_power",
            description="Raise a base number to an exponent power",
            clarifying_questions=["What is the base number?", "What is the exponent?"],
            param_descriptions={"base": "The number being raised", "exponent": "The power"},
            dsl_script="base ** exponent",
            examples=[{"input": "2^8", "params": {"base": 2, "exponent": 8}, "result": "256"}]
        )
    """
    id: Optional[int] = None
    signature_id: str = ""

    # Embedding (3072-dim gemini-embedding-001 centroid)
    # centroid = embedding_sum / embedding_count (running average)
    centroid: Optional[np.ndarray] = None
    embedding_sum: Optional[np.ndarray] = None  # Running sum of all matched embeddings
    embedding_count: int = 1  # Number of embeddings in sum

    # Identity
    step_type: str = ""  # e.g., "compute_power", "find_gcd"

    # Natural Language Interface
    description: str = ""  # "Raise a base number to an exponent power"
    clarifying_questions: list[str] = field(default_factory=list)  # ["What is the base?", ...]
    param_descriptions: dict[str, str] = field(default_factory=dict)  # {"base": "The number..."}

    # DSL Execution
    dsl_script: Optional[str] = None  # "base ** exponent"
    dsl_type: str = "math"  # "math", "sympy", "python"

    # Computation Graph (per CLAUDE.md: route by what operations DO)
    # Graph is structural representation: MUL(param_0, param_1), ADD(MUL(p0, p1), p2)
    computation_graph: Optional[str] = None  # e.g., "MUL(param_0, param_1)"
    graph_embedding: Optional[np.ndarray] = None  # Embedding of graph for routing

    # Few-shot Examples
    examples: list[dict] = field(default_factory=list)  # [{"input": "2^8", "params": {...}, "result": "256"}]

    # Statistics
    uses: int = 0
    successes: int = 0
    operational_failures: int = 0  # MCTS post-mortem: destructive interference flags

    # Embedding Variance Tracking (Welford's online algorithm)
    # Tracks how diverse the problems routed to this signature are
    # High variance = too generic, should decompose into specialized children
    similarity_count: int = 0      # N in Welford's algorithm
    similarity_mean: float = 0.0   # Running mean of cosine similarities
    similarity_m2: float = 0.0     # Sum of squared differences (variance = M2/N)

    # Difficulty tracking (for universal tree)
    # Format: {"0.2": {"uses": 10, "successes": 8}, "0.8": {"uses": 2, "successes": 0}}
    difficulty_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    max_difficulty_solved: float = 0.0  # Highest difficulty this sig has succeeded on

    # Umbrella routing (DAG of DAGs)
    is_semantic_umbrella: bool = False  # True if routes to children
    is_root: bool = False  # True if this is THE root signature (single entry point)
    depth: int = 0  # Routing depth (0=root, increases with parent-child hops)

    # Atomic operations (math primes - per CLAUDE.md: system discovers atomic ops)
    is_atomic: bool = False  # True if this is a "math prime" that should never decompose
    atomic_reason: Optional[str] = None  # Why it's atomic: "high_success", "decomp_failed", etc.

    # Metadata
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None

    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses > 0 else 0.0

    @property
    def has_dsl(self) -> bool:
        return bool(self.dsl_script)

    @property
    def similarity_variance(self) -> float:
        """Compute variance of embedding similarities (Welford's algorithm).

        High variance indicates diverse problems routing to this signature,
        suggesting it should decompose into specialized children.

        Returns:
            Variance of similarities, or 0.0 if insufficient data.
        """
        if self.similarity_count < 2:
            return 0.0
        return self.similarity_m2 / self.similarity_count

    @property
    def similarity_stddev(self) -> float:
        """Standard deviation of embedding similarities."""
        import math
        return math.sqrt(self.similarity_variance)

    def get_difficulty_success_rate(self, difficulty: float, tolerance: float = 0.15) -> float:
        """Get success rate for problems at similar difficulty level.

        Args:
            difficulty: Target difficulty (0.0 to 1.0)
            tolerance: How close difficulties must be to count

        Returns:
            Success rate for this difficulty range, or -1 if no data
        """
        total_uses = 0
        total_successes = 0

        for diff_str, stats in self.difficulty_stats.items():
            try:
                diff = float(diff_str)
                if abs(diff - difficulty) <= tolerance:
                    total_uses += stats.get("uses", 0)
                    total_successes += stats.get("successes", 0)
            except ValueError:
                continue

        if total_uses == 0:
            return -1.0  # No data for this difficulty
        return total_successes / total_uses

    def difficulty_stats_json(self) -> str:
        """Serialize difficulty_stats to JSON."""
        return json.dumps(self.difficulty_stats)

    def get_clarifying_prompt(self) -> str:
        """Generate a prompt asking the LLM to extract parameters."""
        if not self.clarifying_questions:
            return ""

        lines = [f"To execute '{self.step_type}', I need to know:"]
        for i, q in enumerate(self.clarifying_questions, 1):
            lines.append(f"  {i}. {q}")

        if self.param_descriptions:
            lines.append("\nParameter meanings:")
            for param, desc in self.param_descriptions.items():
                lines.append(f"  - {param}: {desc}")

        return "\n".join(lines)

    def get_few_shot_prompt(self, max_examples: int = 3) -> str:
        """Generate few-shot examples for the LLM."""
        if not self.examples:
            return ""

        lines = ["Examples:"]
        for ex in self.examples[:max_examples]:
            lines.append(f"  Input: {ex.get('input', '')}")
            if 'params' in ex:
                lines.append(f"  Params: {ex['params']}")
            lines.append(f"  Result: {ex.get('result', '')}")
            lines.append("")

        return "\n".join(lines)

    def to_prompt(self) -> str:
        """Generate full prompt for LLM interaction."""
        parts = [
            f"Signature: {self.step_type}",
            f"Description: {self.description}",
        ]

        if self.clarifying_questions:
            parts.append(self.get_clarifying_prompt())

        if self.examples:
            parts.append(self.get_few_shot_prompt())

        if self.dsl_script:
            parts.append(f"DSL: {self.dsl_script}")

        return "\n\n".join(parts)

    # Serialization helpers for JSON columns
    def clarifying_questions_json(self) -> str:
        return json.dumps(self.clarifying_questions)

    def param_descriptions_json(self) -> str:
        return json.dumps(self.param_descriptions)

    def examples_json(self) -> str:
        return json.dumps(self.examples)

    @classmethod
    def from_row(cls, row: dict) -> "StepSignature":
        """Create from database row (full parsing)."""
        # Parse JSON fields
        sig_id = row.get("id", "?")
        clarifying_questions = []
        if row.get("clarifying_questions"):
            try:
                clarifying_questions = json.loads(row["clarifying_questions"])
            except json.JSONDecodeError as e:
                logger.warning("[models] Invalid clarifying_questions JSON for sig %s: %s", sig_id, e)

        param_descriptions = {}
        if row.get("param_descriptions"):
            try:
                param_descriptions = json.loads(row["param_descriptions"])
            except json.JSONDecodeError as e:
                logger.warning("[models] Invalid param_descriptions JSON for sig %s: %s", sig_id, e)

        examples = []
        if row.get("examples"):
            try:
                examples = json.loads(row["examples"])
            except json.JSONDecodeError as e:
                logger.warning("[models] Invalid examples JSON for sig %s: %s", sig_id, e)

        difficulty_stats = {}
        if row.get("difficulty_stats"):
            try:
                difficulty_stats = json.loads(row["difficulty_stats"])
            except json.JSONDecodeError as e:
                logger.warning("[models] Invalid difficulty_stats JSON for sig %s: %s", sig_id, e)

        # Parse centroid and embedding_sum (handles both JSON and legacy binary)
        centroid = None
        if row.get("centroid"):
            try:
                centroid = _parse_centroid_data(row["centroid"])
            except Exception as e:
                logger.warning("[models] Invalid centroid for sig %s: %s", sig_id, e)

        embedding_sum = None
        if row.get("embedding_sum"):
            try:
                embedding_sum = _parse_centroid_data(row["embedding_sum"])
            except Exception as e:
                logger.warning("[models] Invalid embedding_sum for sig %s: %s", sig_id, e)

        embedding_count = row.get("embedding_count", 1) or 1

        # Parse graph_embedding (JSON-serialized numpy array)
        graph_embedding = None
        if row.get("graph_embedding"):
            try:
                graph_embedding = np.array(json.loads(row["graph_embedding"]))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("[models] Invalid graph_embedding for sig %s: %s", sig_id, e)

        return cls(
            id=row.get("id"),
            signature_id=row.get("signature_id", ""),
            centroid=centroid,
            embedding_sum=embedding_sum,
            embedding_count=embedding_count,
            step_type=row.get("step_type", ""),
            description=row.get("description", ""),
            clarifying_questions=clarifying_questions,
            param_descriptions=param_descriptions,
            dsl_script=row.get("dsl_script"),
            dsl_type=row.get("dsl_type", "math"),
            computation_graph=row.get("computation_graph"),
            graph_embedding=graph_embedding,
            examples=examples,
            uses=row.get("uses", 0),
            successes=row.get("successes", 0),
            operational_failures=row.get("operational_failures", 0) or 0,
            similarity_count=row.get("similarity_count", 0) or 0,
            similarity_mean=row.get("similarity_mean", 0.0) or 0.0,
            similarity_m2=row.get("similarity_m2", 0.0) or 0.0,
            difficulty_stats=difficulty_stats,
            max_difficulty_solved=row.get("max_difficulty_solved", 0.0) or 0.0,
            is_semantic_umbrella=bool(row.get("is_semantic_umbrella", 0)),
            is_root=bool(row.get("is_root", 0)),
            depth=row.get("depth", 0) or 0,
            is_atomic=bool(row.get("is_atomic", 0)),
            atomic_reason=row.get("atomic_reason"),
            created_at=row.get("created_at"),
            last_used_at=row.get("last_used_at"),
        )

    @classmethod
    def from_row_fast(cls, row: dict) -> "StepSignature":
        """Create from database row (skip expensive JSON parsing).

        ~2x faster than from_row. Use when you only need basic fields.
        Skips: centroid, embedding_sum, clarifying_questions, examples.
        Parses: param_descriptions (needed for extraction validation).
        """
        # Parse param_descriptions (small, needed for extraction validation)
        param_descriptions = {}
        if row.get("param_descriptions"):
            try:
                param_descriptions = json.loads(row["param_descriptions"])
            except json.JSONDecodeError as e:
                logger.warning("[models] Invalid param_descriptions JSON for sig %s: %s", row.get("id", "?"), e)

        return cls(
            id=row.get("id"),
            signature_id=row.get("signature_id", ""),
            centroid=None,  # Skip expensive parsing
            embedding_sum=None,
            embedding_count=row.get("embedding_count", 1) or 1,
            step_type=row.get("step_type", ""),
            description=row.get("description", ""),
            clarifying_questions=[],  # Skip JSON parsing
            param_descriptions=param_descriptions,  # Parse for extraction validation
            dsl_script=row.get("dsl_script"),
            dsl_type=row.get("dsl_type", "math"),
            computation_graph=row.get("computation_graph"),
            examples=[],  # Skip JSON parsing
            uses=row.get("uses", 0),
            successes=row.get("successes", 0),
            operational_failures=row.get("operational_failures", 0) or 0,
            difficulty_stats={},  # Skip JSON parsing
            max_difficulty_solved=row.get("max_difficulty_solved", 0.0) or 0.0,
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
        """Create from database row optimized for routing (parse centroid only).

        ~4x faster than from_row. Use for umbrella routing where we only need
        centroid for similarity, uses/successes for UCB1, and is_semantic_umbrella
        for recursion. Per CLAUDE.md: "Umbrella signature routing should not
        require an LLM call" - routing is purely embedding-based.

        Parses: centroid (REQUIRED for similarity), graph_embedding (for graph routing)
        Skips: clarifying_questions, param_descriptions, examples, embedding_sum
        """
        from mycelium.step_signatures.utils import get_cached_centroid

        # Use cached centroid to avoid repeated JSON parsing
        centroid = get_cached_centroid(row.get("id"), row.get("centroid"))

        # Parse graph_embedding for graph-based routing
        graph_embedding = None
        if row.get("graph_embedding"):
            try:
                graph_embedding = np.array(json.loads(row["graph_embedding"]))
            except (json.JSONDecodeError, TypeError):
                pass  # Silent fail for routing - not critical

        return cls(
            id=row.get("id"),
            signature_id=row.get("signature_id", ""),
            centroid=centroid,
            embedding_sum=None,  # Skip - not needed for routing
            embedding_count=row.get("embedding_count", 1) or 1,
            step_type=row.get("step_type", ""),
            description=row.get("description", ""),
            clarifying_questions=[],  # Skip JSON parsing
            param_descriptions={},  # Skip JSON parsing
            dsl_script=row.get("dsl_script"),
            dsl_type=row.get("dsl_type", "math"),
            computation_graph=row.get("computation_graph"),
            graph_embedding=graph_embedding,
            examples=[],  # Skip JSON parsing
            uses=row.get("uses", 0),
            successes=row.get("successes", 0),
            operational_failures=row.get("operational_failures", 0) or 0,
            difficulty_stats={},  # Skip JSON parsing
            max_difficulty_solved=row.get("max_difficulty_solved", 0.0) or 0.0,
            is_semantic_umbrella=bool(row.get("is_semantic_umbrella", 0)),
            is_root=bool(row.get("is_root", 0)),
            depth=row.get("depth", 0) or 0,
            is_atomic=bool(row.get("is_atomic", 0)),
            atomic_reason=row.get("atomic_reason"),
            created_at=row.get("created_at"),
            last_used_at=row.get("last_used_at"),
        )


@dataclass
class StepExample:
    """An example step that belongs to a signature cluster."""
    id: Optional[int] = None
    signature_id: int = 0
    step_text: str = ""
    embedding: Optional[np.ndarray] = None
    result: str = ""
    success: bool = False
    parent_problem: str = ""
    created_at: Optional[str] = None
