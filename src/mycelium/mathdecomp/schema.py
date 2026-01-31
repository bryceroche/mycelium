"""
Schema definitions for math problem decomposition.

Core principle: Everything is explicit. No inference from naming conventions.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import json


class RefType(str, Enum):
    """Type of reference - explicit, no guessing."""
    EXTRACTION = "extraction"  # Points to extracted variable from problem text
    STEP = "step"              # Points to result of a prior computation step
    CONSTANT = "constant"      # Inline constant value (e.g., percentages like 0.5)


@dataclass
class Ref:
    """
    Explicit pointer to a value source.

    Examples:
        Ref(type=RefType.EXTRACTION, id="num_toys")     # → extracted variable
        Ref(type=RefType.STEP, id="s1")                 # → prior step result
        Ref(type=RefType.CONSTANT, id="0.5")            # → inline constant
    """
    type: RefType
    id: str

    def __post_init__(self):
        # Normalize type if passed as string
        if isinstance(self.type, str):
            self.type = RefType(self.type)

    def to_dict(self) -> dict:
        return {"type": self.type.value, "id": self.id}

    @classmethod
    def from_dict(cls, d: dict) -> "Ref":
        return cls(type=RefType(d["type"]), id=d["id"])

    @classmethod
    def extraction(cls, id: str) -> "Ref":
        """Shorthand for extraction reference."""
        return cls(type=RefType.EXTRACTION, id=id)

    @classmethod
    def step(cls, id: str) -> "Ref":
        """Shorthand for step reference."""
        return cls(type=RefType.STEP, id=id)

    @classmethod
    def constant(cls, value: float) -> "Ref":
        """Shorthand for constant reference."""
        return cls(type=RefType.CONSTANT, id=str(value))

    def resolve(self, extractions: dict, step_results: dict) -> Optional[float]:
        """Resolve this reference to a numeric value."""
        if self.type == RefType.EXTRACTION:
            return extractions.get(self.id)
        elif self.type == RefType.STEP:
            return step_results.get(self.id)
        elif self.type == RefType.CONSTANT:
            try:
                return float(self.id)
            except ValueError:
                return None
        return None


@dataclass
class Extraction:
    """
    A variable extracted from the problem text.

    Tracks provenance: where in the text did this number come from?
    """
    id: str              # Semantic name: "num_toys", "tim_money"
    value: float         # Numeric value: 3.0, 10.0
    span: str            # Text span: "3 toys", "10 dollars"
    offset_start: int    # Character offset start in problem text
    offset_end: int      # Character offset end in problem text
    unit: Optional[str] = None  # Optional unit: "dollars", "toys", "hours"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "value": self.value,
            "span": self.span,
            "offset": [self.offset_start, self.offset_end],
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Extraction":
        offset = d.get("offset", [0, 0])
        return cls(
            id=d["id"],
            value=d["value"],
            span=d["span"],
            offset_start=offset[0],
            offset_end=offset[1],
            unit=d.get("unit"),
        )


@dataclass
class Step:
    """
    A single atomic computation step.

    Uses function registry keys for flexible operations with variable arity.
    """
    id: str                          # Step identifier: "s1", "s2"
    func: str                        # Key into function_registry (e.g., "add", "mul", "sqrt")
    inputs: List[Ref]                # Flexible arity - List of input references
    result: Optional[float] = None   # Computed result (optional, set after execution)
    semantic: str = ""               # What this represents: "total_cost", "remaining_money"

    def __post_init__(self):
        # Convert dict refs to Ref objects if needed
        self.inputs = [
            Ref.from_dict(inp) if isinstance(inp, dict) else inp
            for inp in self.inputs
        ]

    def dependencies(self) -> List[str]:
        """Return IDs of steps this step depends on."""
        return [inp.id for inp in self.inputs if inp.type == RefType.STEP]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "func": self.func,
            "inputs": [inp.to_dict() for inp in self.inputs],
            "result": self.result,
            "semantic": self.semantic,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(
            id=d["id"],
            func=d["func"],
            inputs=[Ref.from_dict(inp) for inp in d["inputs"]],
            result=d.get("result"),
            semantic=d.get("semantic", ""),
        )


@dataclass
class Decomposition:
    """
    Complete decomposition of a math problem.

    Contains:
    - The original problem text
    - Extracted variables with provenance
    - Ordered computation steps (topologically sorted)
    - Reference to the answer step
    - Verification status
    """
    problem: str                    # Original problem text
    extractions: List[Extraction]   # Variables extracted from text
    steps: List[Step]               # Computation steps (topologically sorted)
    answer_ref: Ref                 # Pointer to step that produces the answer
    answer_value: float             # The final answer
    verified: bool = False          # Did execution match claimed results?
    error: Optional[str] = None     # Error message if verification failed

    def __post_init__(self):
        # Convert dict answer_ref to Ref if needed
        if isinstance(self.answer_ref, dict):
            self.answer_ref = Ref.from_dict(self.answer_ref)
        # Convert dict extractions/steps if needed
        self.extractions = [
            Extraction.from_dict(e) if isinstance(e, dict) else e
            for e in self.extractions
        ]
        self.steps = [
            Step.from_dict(s) if isinstance(s, dict) else s
            for s in self.steps
        ]

    def get_extraction(self, id: str) -> Optional[Extraction]:
        """Get extraction by ID."""
        for e in self.extractions:
            if e.id == id:
                return e
        return None

    def get_step(self, id: str) -> Optional[Step]:
        """Get step by ID."""
        for s in self.steps:
            if s.id == id:
                return s
        return None

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "extractions": [e.to_dict() for e in self.extractions],
            "steps": [s.to_dict() for s in self.steps],
            "answer_ref": self.answer_ref.to_dict(),
            "answer_value": self.answer_value,
            "verified": self.verified,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "Decomposition":
        return cls(
            problem=d["problem"],
            extractions=[Extraction.from_dict(e) for e in d["extractions"]],
            steps=[Step.from_dict(s) for s in d["steps"]],
            answer_ref=Ref.from_dict(d["answer_ref"]),
            answer_value=d["answer_value"],
            verified=d.get("verified", False),
            error=d.get("error"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "Decomposition":
        return cls.from_dict(json.loads(json_str))

    def dependency_order(self) -> List[str]:
        """Return step IDs in dependency order (topological sort)."""
        # Build adjacency list
        deps = {s.id: s.dependencies() for s in self.steps}

        # Kahn's algorithm
        in_degree = {s.id: 0 for s in self.steps}
        for s in self.steps:
            for dep in deps[s.id]:
                if dep in in_degree:
                    in_degree[s.id] += 1

        queue = [s_id for s_id, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            s_id = queue.pop(0)
            order.append(s_id)

            for other_id, other_deps in deps.items():
                if s_id in other_deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0 and other_id not in order:
                        queue.append(other_id)

        return order
