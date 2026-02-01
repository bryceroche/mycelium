"""Template and Example models for the template matching system."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class ComputeGraph:
    """A computation graph with nodes and edges."""
    nodes: List[str]  # Variable names: ["X", "Y", "answer"]
    edges: List[Dict[str, Any]]  # Operations: [{"op": "sub", "inputs": ["X", "Y"], "output": "answer"}]

    def to_dict(self) -> Dict:
        return {"nodes": self.nodes, "edges": self.edges}

    @classmethod
    def from_dict(cls, d: Dict) -> "ComputeGraph":
        return cls(nodes=d["nodes"], edges=d["edges"])


@dataclass
class Template:
    """A curated template with slots and computation graph."""
    id: Optional[int] = None
    name: str = ""                    # "system_of_equations", "circle_radius"
    description: str = ""             # Human-readable description
    pattern: str = ""                 # Display pattern with [SLOTS]
    slots: List[str] = field(default_factory=list)  # ["X", "Y", "Z"]
    graph: ComputeGraph = field(default_factory=lambda: ComputeGraph([], []))
    created_at: Optional[str] = None


@dataclass
class Example:
    """A problem example that maps to a template."""
    id: Optional[int] = None
    problem_text: str = ""
    embedding: Optional[np.ndarray] = None
    template_id: int = 0
    slots_mapped: Dict[str, Any] = field(default_factory=dict)  # Cached slot values
    similarity_to_nearest: float = 0.0
    created_at: Optional[str] = None


@dataclass
class ExampleProposal:
    """A proposed example awaiting human review."""
    id: Optional[int] = None
    problem_text: str = ""
    embedding: Optional[np.ndarray] = None
    template_id: int = 0
    similarity_to_nearest: float = 0.0
    slots_mapped: Dict[str, Any] = field(default_factory=dict)
    computed_answer: Any = None
    expected_answer: Any = None
    status: str = "pending"  # "pending" | "approved" | "rejected"
    created_at: Optional[str] = None
