"""
mathdecomp - Recursive decomposition of math problems into atomic steps.

One job, done right: take a text math problem and output structured JSON
with all variables cleanly mapped and all steps neatly ordered.
"""

from .schema import (
    Ref,
    RefType,
    Extraction,
    Step,
    Decomposition,
)
from .executor import execute_decomposition, verify_decomposition
from .decomposer import decompose
from .llm_api import decompose_with_api, decompose_with_cascade

__all__ = [
    # Schema
    "Ref",
    "RefType",
    "Extraction",
    "Step",
    "Decomposition",
    # Execution
    "execute_decomposition",
    "verify_decomposition",
    # Main entry points
    "decompose",
    "decompose_with_api",
    "decompose_with_cascade",
]
