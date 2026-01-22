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
    get_cached_centroid,
    invalidate_centroid_cache,
    get_centroid_cache_stats,
    # Signature lookup caches
    get_cached_signature,
    cache_signature,
    get_cached_children,
    cache_children,
    invalidate_signature_cache,
    invalidate_children_cache,
    get_signature_cache_stats,
)

from mycelium.step_signatures.db import StepSignatureDB, RoutingResult

from mycelium.step_signatures.dsl_executor import (
    DSLSpec,
    try_execute_dsl,
    execute_dsl_with_confidence,
)

from mycelium.step_signatures.stats import (
    StepExecution,
    SignatureStats,
    RoutingContext,
    StepStatsCollector,
    compute_routing_bonus,
    should_prefer_decomposition,
    get_signature_health,
)

from mycelium.step_signatures.dsl_rewriter import (
    RewriteCandidate,
    RewriteResult,
    find_underperforming_signatures,
    rewrite_underperforming_dsls,
    get_rewrite_stats,
)

from mycelium.step_signatures.decay import (
    DecayStatus,
    DecayState,
    DecayAction,
    DecayReport,
    DecayManager,
    run_decay_cycle,
    get_signature_decay_state,
    get_decay_summary,
)

from mycelium.step_signatures.operational_alignment import (
    OperationalAlignmentScore,
    CentroidDriftMetrics,
    RoutingOutcome,
    OperationalAlignmentTracker,
    CentroidDriftAnalyzer,
    compute_operational_alignment_score,
    get_alignment_trend,
    record_routing_outcome,
    print_alignment_report,
)

from mycelium.step_signatures.graph_extractor import (
    extract_computation_graph,
    graphs_equivalent,
    graph_to_natural_language,
    embed_computation_graph,
    embed_computation_graph_sync,
    clear_graph_embedding_cache,
    populate_graph_embeddings,
)

from mycelium.step_signatures.operation_extractor import (
    extract_operation_needed,
    get_operation_embedding,
    extract_and_embed_operation,
    clear_operation_cache,
)

from mycelium.data_layer.schema import STEP_SCHEMA as STEP_SCHEMA_SQL

__all__ = [
    "StepSignature",
    "StepExample",
    "StepSignatureDB",
    "RoutingResult",
    "DSLSpec",
    "try_execute_dsl",
    "execute_dsl_with_confidence",
    "pack_embedding",
    "unpack_embedding",
    "cosine_similarity",
    "get_cached_centroid",
    "invalidate_centroid_cache",
    "get_centroid_cache_stats",
    # Signature lookup caches
    "get_cached_signature",
    "cache_signature",
    "get_cached_children",
    "cache_children",
    "invalidate_signature_cache",
    "invalidate_children_cache",
    "get_signature_cache_stats",
    "STEP_SCHEMA_SQL",
    # Stats module
    "StepExecution",
    "SignatureStats",
    "RoutingContext",
    "StepStatsCollector",
    "compute_routing_bonus",
    "should_prefer_decomposition",
    "get_signature_health",
    # DSL Rewriter
    "RewriteCandidate",
    "RewriteResult",
    "find_underperforming_signatures",
    "rewrite_underperforming_dsls",
    "get_rewrite_stats",
    # Decay Lifecycle
    "DecayStatus",
    "DecayState",
    "DecayAction",
    "DecayReport",
    "DecayManager",
    "run_decay_cycle",
    "get_signature_decay_state",
    "get_decay_summary",
    # Operational Alignment Validation
    "OperationalAlignmentScore",
    "CentroidDriftMetrics",
    "RoutingOutcome",
    "OperationalAlignmentTracker",
    "CentroidDriftAnalyzer",
    "compute_operational_alignment_score",
    "get_alignment_trend",
    "record_routing_outcome",
    "print_alignment_report",
    # Computation Graph Extraction & Embedding
    "extract_computation_graph",
    "graphs_equivalent",
    "graph_to_natural_language",
    "embed_computation_graph",
    "embed_computation_graph_sync",
    "clear_graph_embedding_cache",
    "populate_graph_embeddings",
    # Operation Extraction (for routing)
    "extract_operation_needed",
    "get_operation_embedding",
    "extract_and_embed_operation",
    "clear_operation_cache",
]
