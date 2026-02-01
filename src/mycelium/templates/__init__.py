"""Templates module for coarse-grained reasoning patterns."""
from .models import Template, Example
from .db import save_template, save_example, get_template_by_name, get_all_templates
from .seed import SEED_TEMPLATES, SEED_EXAMPLES, seed_database
from .matcher import match_template

__all__ = [
    "Template",
    "Example",
    "save_template",
    "save_example",
    "get_template_by_name",
    "get_all_templates",
    "SEED_TEMPLATES",
    "SEED_EXAMPLES",
    "seed_database",
    "match_template",
]
