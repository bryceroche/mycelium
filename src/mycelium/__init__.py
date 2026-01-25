"""Mycelium: Recursive problem decomposition with signature-based solution networks."""

__version__ = "1.8.14"

from .step_signatures import StepSignature, StepExample
from .prompt_templates import (
    PromptTemplate,
    PromptRegistry,
    get_registry,
    format_prompt,
)

__all__ = [
    "StepSignature",
    "StepExample",
    "PromptTemplate",
    "PromptRegistry",
    "get_registry",
    "format_prompt",
]
