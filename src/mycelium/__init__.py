"""Mycelium: Template-based math problem solver."""

__version__ = "3.0.0"

from .engine import TemplateEngine, solve
from .templates import Template, Example, seed_database

__all__ = [
    "TemplateEngine",
    "Template",
    "Example",
    "solve",
    "seed_database",
]
