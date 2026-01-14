"""Step Signatures Package V2: Natural Language Interface.

Signatures now speak natural language:
- description: What this signature does
- clarifying_questions: Questions to extract parameters
- param_descriptions: What each parameter means
- examples: Few-shot examples
"""

from mycelium.step_signatures.models import (
    StepSignature,
    StepExample,
)

from mycelium.step_signatures.utils import (
    pack_embedding,
    unpack_embedding,
    cosine_similarity,
)

from mycelium.step_signatures.db import StepSignatureDB

from mycelium.step_signatures.dsl_executor import (
    DSLSpec,
    try_execute_dsl,
    execute_dsl_with_confidence,
)

from mycelium.data_layer.schema import STEP_SCHEMA as STEP_SCHEMA_SQL

__all__ = [
    "StepSignature",
    "StepExample",
    "StepSignatureDB",
    "DSLSpec",
    "try_execute_dsl",
    "execute_dsl_with_confidence",
    "pack_embedding",
    "unpack_embedding",
    "cosine_similarity",
    "STEP_SCHEMA_SQL",
]
