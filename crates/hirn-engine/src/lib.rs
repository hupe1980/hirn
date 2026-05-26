//! `hirn-engine` — orchestrator crate for the Hirn cognitive memory database.
//!
//! Wires together storage, graph, execution, query compilation, and policy
//! enforcement into the [`HirnDB`] entry-point. Sub-modules cover
//! recall/think pipelines, consolidation, admission control, observability,
//! and agent tooling.

pub use hirn_graph::activation;
pub use hirn_storage::HydratedResource;
pub mod adaptive;
pub mod admission;
pub mod agent_context;
pub mod backup;
pub mod consolidation;
mod db;
mod error;
pub mod export;
pub mod graph;
pub mod index_advisor;
pub mod integrity;
pub mod observability;
pub mod operators;
pub mod policy;
pub mod provider_registry;
pub mod ql;
mod resource_presentation;
mod result_json;
pub mod retrieval;
pub use hirn_graph::hebbian;
pub mod scoring;
pub mod security;
pub mod tools;
pub mod watch;

// ── Backward-compatible re-exports (sub-modules moved into directories) ──
pub use graph::cached_graph_store;
pub use graph::causal;
pub use graph::graph_store;
pub use graph::persistent_activation;
pub use graph::persistent_graph;
pub use graph::persistent_hebbian;
pub use observability::diagnostics;
pub use observability::event;
pub use observability::event_log;
pub use observability::inspect;
pub use observability::metrics;
pub use observability::trace;
pub use retrieval::global_retrieval;
pub use retrieval::recall;
pub use retrieval::think;

pub use activation::ActivationMode;
pub use admission::{
    AdmissionController, AdmissionDecision, AdmissionPipeline, ContradictionGate,
    ControllerVerdict, DuplicateAction, DuplicateDetector, MemoryCandidate, PipelineResult,
    RateLimiter, SurpriseGate, TokenBudgetGate,
};
pub use agent_context::AgentContext;
pub use backup::{RollbackReport, Snapshot, SnapshotReport};
pub use causal::{
    CausalChain, CausalChainResult, CausalLink, ContradictionDetection, Counterfactual,
    CounterfactualConstraint, TraceReport, causal_relevance,
};
pub use consolidation::{
    Community, CommunityConfig, CommunityResult, CommunitySummaryResult, ConsolidateBuilder,
    ConsolidationConfig, ConsolidationResult, ConsolidationSchedule, ConsolidationScheduler,
    ConsolidationStatus, DetectedPatterns, DreamCycleConfig, DreamCycleResult, DreamHypothesis,
    DreamPhase, EpisodeSegment, ForgettingResult, NarrativeThread, Pattern, PhaseResult,
    ReconsolidationTracker, ReconsolidationUpdate, execute_dream_cycle,
    generate_community_summaries, retention_score,
};
pub use db::{
    AdminView, CausalView, CrossAgentConsolidationResult, DbStats, EpisodicFilter, EpisodicView,
    GraphView, HirnDB, LayerCounts, MutationWriteContract, MutationWriteGuarantee, NamespaceView,
    PolicyView, PrefetchStats, ProceduralView, PurgeReport, QueryView, RecallView, SemanticFilter,
    SemanticMerge, SemanticMergeOutcome, SemanticOverride, SemanticRetraction,
    SemanticSupersession, SemanticUpdate, SemanticView, WorkingView, mutation_write_contracts,
};
pub use diagnostics::{QueryDiagnostics, QueryId};
pub use error::StoreError;
pub use event::{EventEnvelope, MemoryEvent};
pub use event_log::{CompactionResult, EventFilter, EventLog, RetentionPolicy, SnapshotMeta};
pub use export::{ExportData, ExportReport, ImportReport};
pub use global_retrieval::{
    CommunityMatch, GlobalRetrievalConfig, GlobalRetrievalResult, global_recall,
};
pub use graph::{EdgeId, GraphEdge, GraphNodeData, MAX_EDGES_PER_NODE};
pub use graph_store::GraphStore;
pub use hebbian::{HebbianBuffer, HebbianConfig, HebbianUpdateResult};
pub use index_advisor::{DatasetQueryStats, IndexAdvisor, IndexRecommendation, QueryKind};
pub use inspect::{InspectBuilder, InspectResult, NeighborInfo};
pub use integrity::{
    IntegrityIssue, IntegrityReport, IssueKind, RepairReport, SemanticRevisionIntegrityIssue,
    SemanticRevisionIntegrityReport, SemanticRevisionIssueKind, SemanticRevisionRepairReport,
};
pub use observability::write_path::{
    AdmissionExplanation, EmbeddingDisposition, InterferenceDisposition, InterferenceExplanation,
    RememberExplanation, RememberFailure, RememberStatus, RpeExplanation,
    WritePathOperationExplanation, WritePathOperationStatus,
};
pub use persistent_graph::{BfsResult, CausalBfsRow, PersistentGraph};
pub use policy::{
    Action, AuthzDecision, AuthzRequest, EntityKind, PolicyEngine, PolicyError,
    PolicyNamespaceResolver,
};
pub use provider_registry::{
    ApiKeySource, DefaultsConfig, EmbedderConfig, LlmConfig, ProviderConfig, ProviderDefaults,
    ProviderRegistry, ProvidersSection, RerankerConfig, TokenizerConfig,
};
pub use ql::{ParseError, QueryPlan, QueryResult, Statement};
pub use recall::{
    LayerFilter, RecallBuilder, RecallPresentation, RecallPresentationItem, RecallResult,
    RecallViewMode, ResourceEvidenceSummary,
};
pub use resource_presentation::ResourceScoreAttribution;
pub use result_json::{inspected_result_to_json, trace_result_to_json, traced_result_to_json};
pub use retrieval::explanation::{
    RetrievalExplanation, RetrievalPolicyScope, RetrievalPolicySummary,
    RetrievalSuppressionSummary, RetrievedRecordExplanation, ThinkExplanation,
};
pub use scoring::{ScoreBreakdown, ScoringWeights};
pub use security::{
    CorruptionDefense, CorruptionDefenseConfig, QuarantineApprovalOutcome, QuarantineEntry,
    QuarantineStatus,
};
pub use think::ThinkBuilder;
pub use tools::{
    IntrospectionResult, LinkRequest, MemoryAgent, MemoryToolkit, RecallOptions, RecallRecord,
    StoreRequest, UpdateRequest,
};
pub use trace::{TraceBuilder, TraceResult};
pub use watch::{WatchFilter, WatchSubscription};
