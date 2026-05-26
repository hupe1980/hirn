//! Prometheus-compatible metrics for the hirn engine.
//!
//! Uses the `metrics` crate facade so any `Recorder` backend works — when no
//! recorder is installed, all calls are no-op with zero overhead.
//!
//! Metric naming follows Prometheus conventions:
//! - Counters: `_total` suffix
//! - Histograms: `_seconds` or `_bytes` unit suffix
//! - Gauges: descriptive name, no suffix convention

// ─── Counter names ──────────────────────────────────────────────

/// Total remember operations (labels: realm, status).
pub const REMEMBER_TOTAL: &str = "hirn_remember_total";

/// Total recall operations (labels: realm, status).
pub const RECALL_TOTAL: &str = "hirn_recall_total";

/// Total consolidation operations.
pub const CONSOLIDATION_TOTAL: &str = "hirn_consolidation_total";

/// Total admission rejections (labels: realm).
pub const ADMISSION_REJECTED_TOTAL: &str = "hirn_admission_rejected_total";

/// Total authorization decisions (labels: decision=allow|deny).
pub const AUTHZ_DECISIONS_TOTAL: &str = "hirn_authz_decisions_total";

/// Total RPE routing decisions (labels: realm, namespace, model_id, route, threshold_band).
pub const RPE_PARTITION_ROUTING_TOTAL: &str = "hirn_rpe_partition_routing_total";

/// Total interference-driven consolidation requests (labels: cause).
pub const INTERFERENCE_CONSOLIDATION_TRIGGER_TOTAL: &str =
    "hirn_interference_consolidation_trigger_total";

/// Total suppressed interference-driven consolidation requests (labels: reason).
pub const INTERFERENCE_CONSOLIDATION_SUPPRESSED_TOTAL: &str =
    "hirn_interference_consolidation_suppressed_total";

/// Total consolidation feedback events applied to the interference backlog (labels: outcome).
pub const INTERFERENCE_CONSOLIDATION_FEEDBACK_TOTAL: &str =
    "hirn_interference_consolidation_feedback_total";

/// Total preview-package path decisions (labels: surface=recall|think, path=seeded_reuse|hydrated_refetch).
pub const PREVIEW_PACKAGE_PATH_TOTAL: &str = "hirn_preview_package_path_total";

/// Total offline jobs submitted to the scheduler.
pub const OFFLINE_JOB_SUBMITTED_TOTAL: &str = "hirn_offline_job_submitted_total";

/// Total offline jobs completed successfully.
pub const OFFLINE_JOB_COMPLETED_TOTAL: &str = "hirn_offline_job_completed_total";

/// Total offline jobs that failed.
pub const OFFLINE_JOB_FAILED_TOTAL: &str = "hirn_offline_job_failed_total";

/// Total offline jobs that were skipped.
pub const OFFLINE_JOB_SKIPPED_TOTAL: &str = "hirn_offline_job_skipped_total";

/// Total HirnQL executions by runtime path (labels: statement, path).
pub const QL_EXECUTION_PATH_TOTAL: &str = "hirn_ql_execution_path_total";

// ─── Histogram names ────────────────────────────────────────────

/// Recall duration in seconds (labels: realm).
pub const RECALL_DURATION_SECONDS: &str = "hirn_recall_duration_seconds";

/// Store (remember) duration in seconds (labels: realm).
pub const STORE_DURATION_SECONDS: &str = "hirn_store_duration_seconds";

/// Batch remember stage duration in seconds (labels: realm, stage).
pub const BATCH_REMEMBER_STAGE_DURATION_SECONDS: &str =
    "hirn_batch_remember_stage_duration_seconds";

/// Consolidation duration in seconds.
pub const CONSOLIDATION_DURATION_SECONDS: &str = "hirn_consolidation_duration_seconds";

/// Authorization latency in seconds.
pub const AUTHZ_LATENCY_SECONDS: &str = "hirn_authz_latency_seconds";

/// Embedding latency in seconds.
pub const EMBEDDING_LATENCY_SECONDS: &str = "hirn_embedding_latency_seconds";

/// Preview-package resolution latency in seconds (labels: surface=recall|think, path=seeded_reuse|hydrated_refetch).
pub const PREVIEW_PACKAGE_RESOLUTION_SECONDS: &str = "hirn_preview_package_resolution_seconds";

/// Offline job runtime duration in seconds.
pub const OFFLINE_JOB_DURATION_SECONDS: &str = "hirn_offline_job_duration_seconds";

// ─── Gauge names ────────────────────────────────────────────────

/// Total memory count across all layers.
pub const MEMORY_COUNT: &str = "hirn_memory_count";

/// Total graph node count.
pub const GRAPH_NODE_COUNT: &str = "hirn_graph_node_count";

/// Number of recall candidates returned.
pub const RECALL_CANDIDATES: &str = "hirn_recall_candidates";

/// Storage bytes (labels: realm).
pub const STORAGE_BYTES: &str = "hirn_storage_bytes";

/// Total graph edges (labels: realm).
pub const GRAPH_EDGES_TOTAL: &str = "hirn_graph_edges_total";

/// Total provider fallback events (labels: realm, provider_type=embed|llm).
pub const PROVIDER_FALLBACK_TOTAL: &str = "hirn_provider_fallback_total";

/// Event log sequence number (labels: realm).
pub const EVENT_LOG_SEQ: &str = "hirn_event_log_seq";

/// Number of loaded Cedar policies.
pub const POLICY_COUNT: &str = "hirn_policy_count";

/// Current unresolved interference backlog score.
pub const INTERFERENCE_CONSOLIDATION_BACKLOG_SCORE: &str =
    "hirn_interference_consolidation_backlog_score";

/// Number of namespaces contributing to the unresolved interference backlog.
pub const INTERFERENCE_CONSOLIDATION_BACKLOG_NAMESPACES: &str =
    "hirn_interference_consolidation_backlog_namespaces";

/// Backlog score reduced by the last successful consolidation feedback.
pub const INTERFERENCE_CONSOLIDATION_LAST_SUCCESS_REDUCED_SCORE: &str =
    "hirn_interference_consolidation_last_success_reduced_score";

/// Backlog score still outstanding after the last successful consolidation feedback.
pub const INTERFERENCE_CONSOLIDATION_LAST_SUCCESS_REMAINING_SCORE: &str =
    "hirn_interference_consolidation_last_success_remaining_score";

/// Number of queued offline jobs.
pub const OFFLINE_JOB_QUEUE_DEPTH: &str = "hirn_offline_job_queue_depth";

/// Number of offline jobs currently running.
pub const OFFLINE_JOB_RUNNING: &str = "hirn_offline_job_running";

/// Number of offline jobs completed since process start.
pub const OFFLINE_JOB_COMPLETED: &str = "hirn_offline_job_completed";

/// Number of offline jobs failed since process start.
pub const OFFLINE_JOB_FAILED: &str = "hirn_offline_job_failed";

/// Number of offline jobs skipped since process start.
pub const OFFLINE_JOB_SKIPPED: &str = "hirn_offline_job_skipped";

// ─── Compaction metrics ─────────────────────────────────────────

/// Compaction duration in seconds.
pub const COMPACTION_DURATION_SECONDS: &str = "hirn_compaction_duration_seconds";

/// Total compaction operations.
pub const COMPACTION_TOTAL: &str = "hirn_compaction_total";

/// Fragments removed during compaction.
pub const COMPACTION_FRAGMENTS_REMOVED: &str = "hirn_compaction_fragments_removed";

/// Fragments added during compaction.
pub const COMPACTION_FRAGMENTS_ADDED: &str = "hirn_compaction_fragments_added";

/// Number of datasets compacted per pass.
pub const COMPACTION_DATASETS: &str = "hirn_compaction_datasets";

/// Memories archived during compaction.
pub const COMPACTION_MEMORIES_ARCHIVED: &str = "hirn_compaction_memories_archived";
