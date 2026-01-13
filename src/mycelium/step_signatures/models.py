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

import numpy as np


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

    # Embedding (768-dim MathBERT centroid)
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

    # Few-shot Examples
    examples: list[dict] = field(default_factory=list)  # [{"input": "2^8", "params": {...}, "result": "256"}]

    # Statistics
    uses: int = 0
    successes: int = 0

    # Umbrella routing (DAG of DAGs)
    is_semantic_umbrella: bool = False  # True if routes to children
    depth: int = 0  # Routing depth (0=root, increases with parent-child hops)

    # Metadata
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None

    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses > 0 else 0.0

    @property
    def has_dsl(self) -> bool:
        return bool(self.dsl_script)

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
        """Create from database row."""
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

        # Parse centroid and embedding_sum
        centroid = None
        if row.get("centroid"):
            try:
                centroid = np.array(json.loads(row["centroid"]))
            except (json.JSONDecodeError, ValueError):
                pass

        embedding_sum = None
        if row.get("embedding_sum"):
            try:
                embedding_sum = np.array(json.loads(row["embedding_sum"]))
            except (json.JSONDecodeError, ValueError):
                pass

        embedding_count = row.get("embedding_count", 1) or 1

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
            examples=examples,
            uses=row.get("uses", 0),
            successes=row.get("successes", 0),
            is_semantic_umbrella=bool(row.get("is_semantic_umbrella", 0)),
            depth=row.get("depth", 0) or 0,
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
