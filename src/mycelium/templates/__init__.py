"""Templates module for computation graph execution."""
from .graphs import execute_graph, validate_graph, OPERATIONS
from .models import Template, ComputeGraph, Example
from .db import save_template, save_example, get_template_by_name, get_all_templates
from .seed import SEED_TEMPLATES, SEED_EXAMPLES, seed_database
from .chunker import chunk_problem

__all__ = [
    "execute_graph",
    "validate_graph",
    "OPERATIONS",
    "Template",
    "ComputeGraph",
    "Example",
    "save_template",
    "save_example",
    "get_template_by_name",
    "get_all_templates",
    "SEED_TEMPLATES",
    "SEED_EXAMPLES",
    "seed_database",
    "chunk_problem",
]
