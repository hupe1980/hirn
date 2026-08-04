pub mod agent;
pub mod audit;
pub mod causal;
pub mod circuit_breaker;
pub mod config;
pub mod content;
pub mod embed;
pub mod embedding_dimension;
pub mod episodic;
pub mod error;
pub mod id;
pub mod interner;
pub mod metadata;
pub mod namespace;
pub mod nlu;
pub mod offline;
pub mod persist;
pub mod preference;
pub mod procedural;
pub mod prospective;
pub mod provenance;
pub mod quarantine;
pub mod record;
pub mod resource;
pub mod revision;
pub mod sanitize;
pub mod scoring;
pub mod semantic;
pub mod stats;
pub mod svo_event;
pub mod temporal;
pub mod temporal_ledger;
pub mod text_util;
pub mod timestamp;
pub mod tokenizer;
pub mod types;
pub mod working;

// Re-exports for convenience.
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use config::{
    AdmissionPoisoningAction, ConflictResolutionPolicy, ConflictResolutionPolicyOverrides,
    DistanceMetric, EmbedderCircuitBreakerRuntimeConfig, EmbedderPersistentCacheRuntimeConfig,
    EmbedderRetryConfig, EmbedderRuntimeConfig, EvolutionMode, HirnConfig, NluConfig,
    TextRetention, TierPolicy,
};
pub use content::{ExternalFetchPolicy, MemoryContent};
pub use embed::{
    CharEstimateCounter, ChatMessage, Embedder, Embedding, EntityExtractor, ExtractedEntity,
    ExtractedRelation, LlmChunk, LlmOptions, LlmProvider, LlmResponse, LlmStream, NoopReranker,
    RerankResult, Reranker, ResponseFormat, TokenCounter, TokenUsage,
};
pub use embedding_dimension::EmbeddingDimension;
pub use error::{EmbeddingFailureDetail, HirnError, HirnResult, PartialEmbeddingBatch};
pub use id::MemoryId;
pub use nlu::{
    Calibration, CalibrationBin, CalibrationReport, CalibrationSample, Classification,
    ClassificationTask, DecisionSource, EventExtractor, LabelSpec, MIN_CALIBRATION_SAMPLES,
    NliJudgment, NliLabel, NliModel, NluBudget, StructuredEvent, TextClassifier,
};
pub use offline::{
    BudgetExceededPolicy, CognitiveJob, CognitiveJobKind, ConflictResolutionPolicySnapshot,
    GeneratedCognitionDecision, GeneratedCognitionKind, GeneratedCognitionReview,
    GeneratedCognitionRollbackReceipt, GeneratedReviewRequirement, OfflineJobId,
    OfflineJobInspection, OfflineJobOutcome, OfflineJobPriority, OfflineJobRecord,
    OfflineJobStatus, OfflineJobTarget, OfflineRecoveryPolicy, OfflineRetryPolicy,
    OfflineSchedulerConfig, OfflineSchedulerMetrics, OperatorBudget, PlanningAgenda,
    PlanningMemoryRef, PlanningSubgoal, PlanningSupportKind, ReconcileArbitrationStatus,
    ReconcileProposal, ReconcileProposalAction, ReconcileProposalMember, TemporalWindow,
};
pub use preference::{
    PREFERENCE_EVIDENCE_METADATA_KEY, PreferenceEvidence, PreferencePolarity,
    annotate_inferred_preference, infer_preference_evidence,
};
pub use quarantine::QuarantinedRecordKind;
pub use resource::{
    DerivedArtifact, DerivedArtifactId, DerivedArtifactIndexPolicy, DerivedArtifactIndexRule,
    DerivedArtifactKind, EvidenceLink, EvidenceRole, HydrationMode, LogicalResourceId,
    ModalityProfile, ResourceGovernanceState, ResourceId, ResourceIndexPolicy, ResourceIndexRule,
    ResourceLocation, ResourceObject, ResourceQuotaPolicy, ResourceQuotaRule, ResourceQuotaScope,
    ResourceRetentionAction, ResourceRetentionPolicy, ResourceRetentionRule, ResourceRevisionId,
    SecondaryIndexType,
};
pub use revision::{
    LogicalMemoryId, RecallSnapshot, RevisionId, RevisionOperation, RevisionRef, RevisionState,
};
pub use stats::WelfordStats;
pub use temporal::{TemporalEnvelope, TemporalState, TimePrecision};
pub use timestamp::Timestamp;
pub use tokenizer::{EstimatingTokenizer, Tokenizer};
