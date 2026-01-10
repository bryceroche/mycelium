"""Step Signatures Package: Reusable solution patterns for subproblems."""

from mycelium.step_signatures.schema import (
    VALUE_TYPES,
    OUTPUT_FORMATS,
    EXECUTION_MODES,
    InputSpec,
    OutputSpec,
    StepIOSchema,
    try_execute_formula,
    get_default_schema,
    DEFAULT_IO_SCHEMAS,
)

from mycelium.step_signatures.models import (
    PlanType,
    StepSignature,
    StepExample,
    PendingStepExample,
    SignatureRelationship,
)

from mycelium.step_signatures.utils import (
    pack_embedding,
    unpack_embedding,
    cosine_similarity,
)

from mycelium.step_signatures.db import StepSignatureDB

from mycelium.step_signatures.dsl_executor import (
    DSLLayer,
    DSLSpec,
    try_execute_dsl,
    execute_dsl_from_json,
    execute_dsl_with_confidence,
    execute_dsl_with_llm_matching,
    llm_rewrite_script,
    register_operator,
)

from mycelium.step_signatures.dsl_generator import (
    generate_dsl_for_signature,
    maybe_generate_dsl,
)

from mycelium.data_layer.schema import STEP_SCHEMA as STEP_SCHEMA_SQL

from mycelium.config import (
    EXACT_MATCH_THRESHOLD,
    VARIANT_THRESHOLD,
    RELIABILITY_MIN_USES,
    RELIABILITY_MIN_SUCCESS_RATE,
    DEFAULT_INJECTION_THRESHOLD,
)

__all__ = [
    "VALUE_TYPES",
    "OUTPUT_FORMATS",
    "EXECUTION_MODES",
    "InputSpec",
    "OutputSpec",
    "StepIOSchema",
    "try_execute_formula",
    "get_default_schema",
    "DEFAULT_IO_SCHEMAS",
    "PlanType",
    "StepSignature",
    "StepExample",
    "PendingStepExample",
    "SignatureRelationship",
    "StepSignatureDB",
    "DSLLayer",
    "DSLSpec",
    "try_execute_dsl",
    "execute_dsl_from_json",
    "execute_dsl_with_confidence",
    "execute_dsl_with_llm_matching",
    "llm_rewrite_script",
    "register_operator",
    "generate_dsl_for_signature",
    "maybe_generate_dsl",
    "pack_embedding",
    "unpack_embedding",
    "cosine_similarity",
    "STEP_SCHEMA_SQL",
    "EXACT_MATCH_THRESHOLD",
    "VARIANT_THRESHOLD",
    "RELIABILITY_MIN_USES",
    "RELIABILITY_MIN_SUCCESS_RATE",
    "DEFAULT_INJECTION_THRESHOLD",
]
