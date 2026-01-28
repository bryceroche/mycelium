"""Mycelium Data Layer - SQLite database access."""

from mycelium.data_layer.connection import (
    ConnectionManager,
    configure_connection,
    create_connection_manager,
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
# StateManager for db_metadata access (per CLAUDE.md New Favorite Pattern)
from mycelium.data_layer.state_manager import (
    StateManager,
    WelfordStats,
    get_state_manager,
)

# Thresholds re-exported from config per CLAUDE.md "The Flow"
from mycelium.config import (
    REJECTION_COUNT_THRESHOLD,
    REJECTION_RATE_THRESHOLD,
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
    # Leaf rejection tracking (thresholds now from config.py per "The Flow")
    get_rejection_count_threshold,  # Adaptive threshold based on system maturity
    record_leaf_rejection,
    get_leaf_rejection_stats,
    get_leaves_needing_decomposition,
    flag_high_rejection_leaves_for_decomposition,
    # DB maturity (for general use)
    compute_db_maturity,
    # Segmentation novelty stats (for TreeGuidedPlanner)
    get_segmentation_novelty_stats,
    save_segmentation_novelty_stats,
)

db = get_db()

__all__ = [
    "db",
    "get_db",
    "reset_db",
    "create_connection_manager",
    "ConnectionManager",
    "configure_connection",
    "EMBEDDING_DIM",
    "SQLITE_SCHEMA",
    "STEP_SCHEMA",
    "get_schema",
    "init_db",
    # StateManager (per CLAUDE.md New Favorite Pattern)
    "StateManager",
    "WelfordStats",
    "get_state_manager",
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
    # Leaf rejection tracking (thresholds from config.py per "The Flow")
    # Note: REJECTION_*_THRESHOLD constants should be imported from config.py directly
    "get_rejection_count_threshold",  # Adaptive threshold function
    "record_leaf_rejection",
    "get_leaf_rejection_stats",
    "get_leaves_needing_decomposition",
    "flag_high_rejection_leaves_for_decomposition",
    # DB maturity
    "compute_db_maturity",
    # Segmentation novelty stats
    "get_segmentation_novelty_stats",
    "save_segmentation_novelty_stats",
]
