"""Step Signatures Package: Core tree structure for routing.

Signatures match (leaf_node, dag_step) pairs via embeddings.
Welford stats guide all structural decisions.
"""

from mycelium.step_signatures.models import (
    StepSignature,
    StepExample,
    ProposedSignature,
)

from mycelium.step_signatures.utils import (
    pack_embedding,
    unpack_embedding,
    cosine_similarity,
    get_cached_centroid,
    invalidate_centroid_cache,
    get_centroid_cache_stats,
    get_cached_signature,
    cache_signature,
    get_cached_children,
    cache_children,
    invalidate_signature_cache,
    invalidate_children_cache,
    get_signature_cache_stats,
)

from mycelium.step_signatures.db import StepSignatureDB, RoutingResult, get_step_db, reset_step_db

from mycelium.step_signatures.dsl_executor import (
    DSLSpec,
    try_execute_dsl,
)

from mycelium.step_signatures.stats import record_step_stats

from mycelium.step_signatures.graph_extractor import (
    extract_computation_graph,
    graphs_equivalent,
    graph_to_natural_language,
    embed_computation_graph_sync,
    clear_graph_embedding_cache,
)

from mycelium.data_layer.schema import STEP_SCHEMA as STEP_SCHEMA_SQL

__all__ = [
    "StepSignature",
    "StepExample",
    "ProposedSignature",
    "StepSignatureDB",
    "RoutingResult",
    "DSLSpec",
    "try_execute_dsl",
    "pack_embedding",
    "unpack_embedding",
    "cosine_similarity",
    "get_cached_centroid",
    "invalidate_centroid_cache",
    "get_centroid_cache_stats",
    "get_cached_signature",
    "cache_signature",
    "get_cached_children",
    "cache_children",
    "invalidate_signature_cache",
    "invalidate_children_cache",
    "get_signature_cache_stats",
    "STEP_SCHEMA_SQL",
    "record_step_stats",
    "extract_computation_graph",
    "graphs_equivalent",
    "graph_to_natural_language",
    "embed_computation_graph_sync",
    "clear_graph_embedding_cache",
]
