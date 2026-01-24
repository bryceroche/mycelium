"""Mycelium Data Layer - SQLite database access."""

from mycelium.data_layer.connection import (
    ConnectionManager,
    configure_connection,
    get_db,
    reset_db,
    EMBEDDING_DIM,
)
from mycelium.data_layer.schema import (
    SQLITE_SCHEMA,
    STEP_SCHEMA,
    get_schema,
    init_db,
)

# MCTS Wave Function data access
from mycelium.data_layer.mcts import (
    MCTSDag,
    MCTSDagStep,
    MCTSThread,
    MCTSThreadStep,
    create_dag,
    grade_dag,
    create_dag_steps,
    create_thread,
    complete_thread,
    grade_thread,
    log_thread_step,
    update_amplitude_post,
    update_summary_amplitude_post,
    batch_update_amplitudes,
    batch_update_summary_amplitudes,
    get_thread_steps_for_dag,
    get_node_step_stats,
    get_dag_step_node_performance,
    run_postmortem,
    # Step-node stats (closes feedback loop)
    update_dag_step_node_stats,
    get_dag_step_node_stats_batch,
    get_dag_step_node_stats_single,
    propagate_step_node_stats,
    get_problem_nodes_needing_attention,
    # DSL regeneration from post-mortem (per beads mycelium-flbq)
    trigger_dsl_regeneration_for_nodes,
    get_accumulated_failing_nodes,
    clear_accumulated_failing_nodes,
    should_trigger_dsl_regen,
    # Divergence-point analysis (per beads mycelium-2rss)
    ThreadPath,
    DivergencePoint,
    get_thread_paths,
    find_divergence_points,
    assign_divergence_blame,
    # Diagnostic post-mortem
    DiagnosticResult,
    # Dag-step embeddings
    store_dag_step_embedding,
    update_dag_step_embedding_outcome,
    find_similar_dag_steps,
    # Atomic discovery (math primes)
    AtomicCandidate,
    discover_atomic_signatures,
    mark_signature_atomic,
    unmark_signature_atomic,
    is_signature_atomic,
    run_atomic_discovery,
    # DAG plan stats (per beads mycelium-ogo6)
    compute_plan_signature,
    record_plan_outcome,
    get_plan_stats_summary,
    get_top_plans,
    get_worst_plans,
    # Decomposition queue (per beads mycelium-mm08)
    check_substeps_match_existing,
    queue_for_decomposition,
    get_pending_decompositions,
    get_decomposition_queue_size,
    get_oldest_pending_age_seconds,
    mark_decomposition_processed,
    get_decomposition_queue_stats,
    # Blocking decomposition coordination
    get_decomposition_results,
    are_decompositions_ready,
    get_pending_queue_ids,
    # Leaf rejection tracking
    REJECTION_SIM_THRESHOLD,
    REJECTION_COUNT_THRESHOLD,
    REJECTION_RATE_THRESHOLD,
    record_leaf_rejection,
    get_leaf_rejection_stats,
    get_leaves_needing_decomposition,
    check_and_reject_if_low_similarity,
    flag_high_rejection_leaves_for_decomposition,
    # DB maturity (for general use)
    compute_db_maturity,
)

db = get_db()

__all__ = [
    "db",
    "get_db",
    "reset_db",
    "ConnectionManager",
    "configure_connection",
    "EMBEDDING_DIM",
    "SQLITE_SCHEMA",
    "STEP_SCHEMA",
    "get_schema",
    "init_db",
    # MCTS Wave Function
    "MCTSDag",
    "MCTSDagStep",
    "MCTSThread",
    "MCTSThreadStep",
    "create_dag",
    "grade_dag",
    "create_dag_steps",
    "create_thread",
    "complete_thread",
    "grade_thread",
    "log_thread_step",
    "update_amplitude_post",
    "update_summary_amplitude_post",
    "batch_update_amplitudes",
    "batch_update_summary_amplitudes",
    "get_thread_steps_for_dag",
    "get_node_step_stats",
    "get_dag_step_node_performance",
    "run_postmortem",
    # Step-node stats
    "update_dag_step_node_stats",
    "get_dag_step_node_stats_batch",
    "get_dag_step_node_stats_single",
    "propagate_step_node_stats",
    "get_problem_nodes_needing_attention",
    # DSL regeneration
    "trigger_dsl_regeneration_for_nodes",
    "get_accumulated_failing_nodes",
    "clear_accumulated_failing_nodes",
    "should_trigger_dsl_regen",
    # Divergence-point analysis
    "ThreadPath",
    "DivergencePoint",
    "get_thread_paths",
    "find_divergence_points",
    "assign_divergence_blame",
    # Diagnostic post-mortem
    "DiagnosticResult",
    # Dag-step embeddings
    "store_dag_step_embedding",
    "update_dag_step_embedding_outcome",
    "find_similar_dag_steps",
    # Atomic discovery (math primes)
    "AtomicCandidate",
    "discover_atomic_signatures",
    "mark_signature_atomic",
    "unmark_signature_atomic",
    "is_signature_atomic",
    "run_atomic_discovery",
    # DAG plan stats (per beads mycelium-ogo6)
    "compute_plan_signature",
    "record_plan_outcome",
    "get_plan_stats_summary",
    "get_top_plans",
    "get_worst_plans",
    # Decomposition queue
    "check_substeps_match_existing",
    "queue_for_decomposition",
    "get_pending_decompositions",
    "get_decomposition_queue_size",
    "get_oldest_pending_age_seconds",
    "mark_decomposition_processed",
    "get_decomposition_queue_stats",
    # Blocking decomposition coordination
    "get_decomposition_results",
    "are_decompositions_ready",
    "get_pending_queue_ids",
    # Leaf rejection tracking
    "REJECTION_SIM_THRESHOLD",
    "REJECTION_COUNT_THRESHOLD",
    "REJECTION_RATE_THRESHOLD",
    "record_leaf_rejection",
    "get_leaf_rejection_stats",
    "get_leaves_needing_decomposition",
    "check_and_reject_if_low_similarity",
    "flag_high_rejection_leaves_for_decomposition",
    # DB maturity
    "compute_db_maturity",
]
