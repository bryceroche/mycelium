"""Template and Example models for the template matching system."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class Template:
    """A coarse-grained reasoning template that guides LLM expression generation.

    Templates represent problem-solving patterns (e.g., sequential operations,
    complement calculations, algebraic equations). The LLM uses the guidance
    and prompt_template to understand the pattern and write a Python expression.
    """
    id: Optional[int] = None
    name: str = ""                    # Pattern name: "sequential", "complement", "algebra"
    description: str = ""             # What this pattern is for
    guidance: str = ""                # Reasoning guidance for LLM (e.g., "Do operations in sequence")
    prompt_template: str = ""         # Prompt with {problem} placeholder
    examples: List[str] = field(default_factory=list)  # Example problems for context
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
