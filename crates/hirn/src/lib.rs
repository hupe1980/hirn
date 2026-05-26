//! # hirn — A Brain for Large Language Models
//!
//! > *hirn* /hɪʁn/ (German: brain) — a cognitive memory database engine for
//! > LLM systems, written in Rust.
//!
//! hirn is a **purpose-built database engine** for cognitive memory. Not a
//! wrapper around a vector database. Not an agent framework. A production-grade
//! memory engine implementing neuroscience-grounded layered memory — working,
//! episodic, and semantic — with graph-based associations, spreading activation,
//! Hebbian self-organization, and narrative consolidation.
//!
//! Ship it how you need it:
//! - **`hirn`** — embed as a library. LanceDB-backed, zero network
//!   overhead. Like SQLite for memory.
//! - **`hirnd`** — run as a standalone daemon with gRPC + HTTP + MCP.
//!
//! # Quick Start
//!
//! The easiest way to get started is `HirnMemory` — zero-config, auto-detects
//! embedding providers from environment variables:
//!
//! ```rust,no_run
//! use hirn::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> HirnResult<()> {
//!     let memory = HirnMemory::open("./brain").await?;
//!     memory.remember("User prefers dark mode").await?;
//!     let ctx = memory.think("What are the user's preferences?", 2048).await?;
//!     println!("{}", ctx.context);
//!     Ok(())
//! }
//! ```
//!
//! # Tokenizers
//!
//! Tokenizer implementations are provider-owned and resolved through
//! `hirn_engine::ProviderRegistry`. The `tiktoken` feature is enabled by
//! default; enable `hf-tokenizer` for local HuggingFace tokenizers. When no
//! model-backed tokenizer is available, hirn falls back to `EstimatingTokenizer`.
//!
//! For fine-grained control, use `Hirn` directly with a `PhysicalStore`:
//!
//! ```rust,no_run
//! use hirn::prelude::*;
//! use hirn_storage::{HirnDb, HirnDbConfig};
//!
//! #[tokio::main]
//! async fn main() -> HirnResult<()> {
//!     // Open LanceDB storage
//!     let config = HirnDbConfig::local("./brain/lance");
//!     let storage = HirnDb::open(config).await.unwrap().store_arc();
//!
//!     // Open the database
//!     let brain = Hirn::open("./brain", storage).await.unwrap();
//!
//!     // Register an agent
//!     let agent = AgentId::new("my_agent").unwrap();
//!     brain.register_agent(&agent, "My Agent").await.unwrap();
//!
//!     // Get an agent-scoped context
//!     let ctx = brain.as_agent(&agent).await.unwrap();
//!
//!     // Remember an experience (episodic memory)
//!     let episode = EpisodicRecord::builder()
//!         .content("Benchmark: HNSW with PQ outperforms brute-force by 40x")
//!         .event_type(EventType::Experiment)
//!         .agent_id(agent.clone())
//!         .importance(0.85)
//!         .embedding(vec![0.1; 768])
//!         .build()
//!         .unwrap();
//!     let id = ctx.remember(episode).await.unwrap();
//!
//!     // Recall with spreading activation
//!     let results = brain
//!         .recall_view()
//!         .query(vec![0.1; 768])
//!         .activation(ActivationMode::Spreading)
//!         .limit(10)
//!         .execute()
//!         .await
//!         .unwrap();
//!
//!     // Think — assemble optimal LLM context under token budget
//!     let context = brain
//!         .recall_view()
//!         .think(vec![0.1; 768])
//!         .budget(4096)
//!         .execute()
//!         .await
//!         .unwrap();
//!
//!     // Or use HirnQL directly
//!     let result = brain.ql().execute(
//!         r#"RECALL episodic ABOUT "vector database" LIMIT 5"#
//!     ).await.unwrap();
//!     Ok(())
//! }
//! ```
//!
//! # Architecture
//!
//! hirn implements three memory layers inspired by neuroscience:
//!
//! | Layer | Brain Analog | Purpose |
//! |-------|-------------|---------|
//! | **Working** | Prefrontal cortex | Token-bounded scratchpad for active reasoning |
//! | **Episodic** | Hippocampus | Time-anchored experiences and events |
//! | **Semantic** | Neocortex | Consolidated knowledge and stable facts |
//!
//! All memories exist as nodes in a typed property graph with spreading
//! activation, Hebbian co-retrieval learning, causal reasoning, and
//! namespace-based multi-agent isolation.
//!
//! # Module Organization
//!
//! - [`prelude`] — Common imports for most use cases
//! - [`agent`] — Multi-agent context and namespace isolation
//! - [`episodic`] — Episodic memory records and builders
//! - [`semantic`] — Semantic knowledge records and builders
//! - [`working`] — Working memory entries and builders
//! - [`graph`] — Property graph with typed edges
//! - [`activation`] — Spreading activation with lateral inhibition
//! - [`causal`] — Causal chain extraction and counterfactual reasoning
//! - [`consolidation`] — Episodic → semantic consolidation engine
//! - [`scoring`] — Composite relevance scoring
//! - [`hebbian`] — Hebbian edge weight learning
//! - [`ql`] — HirnQL query language parser and executor
//! - [`provenance`] — Provenance tracking and audit trail
//! - [`security`] — Anomaly detection and quarantine
//! - `vector` — HNSW vector index (advanced)

// ── Re-export: core types ───────────────────────────────────────────────

pub mod memory;
pub use memory::{HirnMemory, MemoryRecallBuilder, MemoryThinkBuilder};

/// The database handle. Open a `Hirn` to start working with cognitive memory.
///
/// This is the primary entry point for fine-grained control. A `Hirn` instance
/// owns a LanceDB-backed database and provides all memory operations.
///
/// For zero-config usage, prefer [`HirnMemory`].
///
/// ```rust,no_run
/// use hirn::Hirn;
/// use hirn_storage::{HirnDb, HirnDbConfig};
///
/// # async fn demo() {
/// let config = HirnDbConfig::local("./brain/lance");
/// let storage = HirnDb::open(config).await.unwrap().store_arc();
/// let brain = Hirn::open("./brain", storage).await.unwrap();
/// # }
/// ```
pub type Hirn = hirn_engine::HirnDB;

pub use hirn_core::ConflictResolutionPolicy;
pub use hirn_core::ConflictResolutionPolicyOverrides;
pub use hirn_core::EmbedderCircuitBreakerRuntimeConfig;
pub use hirn_core::EmbedderPersistentCacheRuntimeConfig;
pub use hirn_core::EmbedderRetryConfig;
pub use hirn_core::EmbedderRuntimeConfig;
pub use hirn_core::EstimatingTokenizer;
pub use hirn_core::HirnConfig;
pub use hirn_core::HirnError;
pub use hirn_core::HirnResult;
pub use hirn_core::MemoryId;
pub use hirn_core::RecallSnapshot;
pub use hirn_core::RevisionId;
pub use hirn_core::Timestamp;
pub use hirn_core::Tokenizer;

// F-39/F-41: Trait abstractions for pluggable embedding, LLM, and extraction.
pub use hirn_core::embed::{
    CharEstimateCounter, ChatMessage, Embedder, Embedding, EntityExtractor, ExtractedEntity,
    ExtractedRelation, LlmChunk, LlmOptions, LlmProvider, LlmResponse, LlmStream, NoopReranker,
    RerankResult, Reranker, ResponseFormat, TokenCounter, TokenUsage,
};

pub use hirn_engine::AgentContext;
pub use hirn_engine::DbStats;
pub use hirn_engine::HirnDB;
pub use hirn_engine::IntegrityIssue;
pub use hirn_engine::IntegrityReport;
pub use hirn_engine::IssueKind;
pub use hirn_engine::LayerCounts;
pub use hirn_engine::MemoryEvent;
pub use hirn_engine::RepairReport;
pub use hirn_engine::SemanticRevisionIntegrityIssue;
pub use hirn_engine::SemanticRevisionIntegrityReport;
pub use hirn_engine::SemanticRevisionIssueKind;
pub use hirn_engine::SemanticRevisionRepairReport;
pub use hirn_engine::StoreError;
pub use hirn_engine::{
    ApiKeySource, DefaultsConfig, EmbedderConfig, LlmConfig, ProviderConfig, ProvidersSection,
    RerankerConfig, TokenizerConfig,
};
pub use hirn_engine::{ProviderDefaults, ProviderRegistry};
pub use hirn_engine::{inspected_result_to_json, trace_result_to_json, traced_result_to_json};

// ── Submodules ──────────────────────────────────────────────────────────

/// Multi-agent context, namespace isolation, and team management.
///
/// Use [`Hirn::as_agent`] to get an [`AgentContext`] that enforces namespace
/// boundaries on all memory operations.
pub mod agent {
    pub use hirn_core::agent::AgentRecord;
    pub use hirn_core::namespace::NamespaceRecord;
    pub use hirn_core::types::{AgentId, Namespace, NamespaceKind};
    pub use hirn_engine::AgentContext;
    pub use hirn_engine::CrossAgentConsolidationResult;
}

/// Episodic memory — the hippocampus.
///
/// Time-anchored experiences and events with full provenance.
pub mod episodic {
    pub use hirn_core::episodic::{EpisodicRecord, EpisodicRecordBuilder};
    pub use hirn_core::types::EventType;
    pub use hirn_engine::EpisodicFilter;
}

/// Semantic memory — the neocortex.
///
/// Consolidated knowledge, stable facts, and concept hierarchies.
pub mod semantic {
    pub use hirn_core::semantic::{ConceptEdge, SemanticRecord, SemanticRecordBuilder};
    pub use hirn_core::types::KnowledgeType;
    pub use hirn_engine::{
        SemanticFilter, SemanticMerge, SemanticMergeOutcome, SemanticOverride, SemanticRetraction,
        SemanticSupersession, SemanticUpdate,
    };
}

/// Working memory — the prefrontal cortex.
///
/// Token-bounded scratchpad for active reasoning context.
pub mod working {
    pub use hirn_core::types::Priority;
    pub use hirn_core::working::{WorkingMemoryEntry, WorkingMemoryEntryBuilder};
}

/// Procedural memory — the basal ganglia / cerebellum.
///
/// Learned action sequences, tool-use procedures, and automated workflows
/// with success-rate tracking and reinforcement.
pub mod procedural {
    pub use hirn_core::procedural::{ActionStep, ProceduralRecord, ProceduralRecordBuilder};
}

/// Memory record types spanning all layers.
pub mod record {
    pub use hirn_core::record::MemoryRecord;
    pub use hirn_core::types::{Layer, MemoryRef};
}

/// Property graph with typed, weighted edges.
///
/// All memories exist as nodes in a typed property graph. Edges carry
/// relationship semantics (causal, temporal, similarity, contradiction)
/// with weights that evolve through Hebbian co-retrieval learning.
pub mod graph {
    pub use hirn_core::types::EdgeRelation;
    pub use hirn_engine::{EdgeId, GraphEdge, GraphNodeData};
}

/// Spreading activation with lateral inhibition.
///
/// Implements cognitive spreading activation theory (Collins & Loftus, 1975)
/// for graph-based memory retrieval. Activation propagates from seed nodes
/// through weighted edges, with lateral inhibition suppressing competing
/// subgraphs.
pub mod activation {
    pub use hirn_engine::activation::{
        ActivationConfig, ActivationMode, ActivationResult, ActivationTrace,
    };
}

/// Causal chain extraction and counterfactual reasoning.
///
/// Traverse `causes`/`caused_by` edges to reconstruct causal narratives
/// and detect implicit constraints.
pub mod causal {
    pub use hirn_engine::{
        CausalChain, CausalChainResult, CausalLink, ContradictionDetection, Counterfactual,
        CounterfactualConstraint, TraceReport,
    };
}

/// Episodic → semantic consolidation engine.
///
/// Mimics hippocampal replay during sleep: episode segmentation, pattern
/// detection, narrative thread formation, and concept extraction.
pub mod consolidation {
    pub use hirn_engine::{
        ConsolidateBuilder, ConsolidationConfig, ConsolidationResult, ConsolidationScheduler,
        ConsolidationStatus, DetectedPatterns, EpisodeSegment, ForgettingResult, NarrativeThread,
        Pattern, ReconsolidationTracker, ReconsolidationUpdate,
    };
}

/// Composite relevance scoring.
///
/// Combines similarity, importance, recency, activation, and causal
/// relevance into a single composite score with configurable weights.
pub mod scoring {
    pub use hirn_engine::ScoringWeights;
}

/// Hebbian edge weight learning.
///
/// Co-retrieved memories strengthen their connections. The graph
/// self-organizes to reflect actual usage patterns.
pub mod hebbian {
    pub use hirn_engine::{HebbianConfig, HebbianUpdateResult};
}

/// HirnQL — the cognitive memory query language.
///
/// A declarative query language purpose-built for cognitive memory operations.
/// HirnQL is to hirn what SQL is to PostgreSQL.
///
/// ```text
/// RECALL episodic
///   ABOUT "vector database optimization"
///   EXPAND GRAPH DEPTH 2 ACTIVATION spreading
///   WHERE importance > 0.4
///   LIMIT 20
/// ```
pub mod ql {
    pub use hirn_engine::ql::ast;
    pub use hirn_engine::ql::context::{
        ConflictArbitrationStatus, ConflictGroup, ConflictMember, ConflictMemberStatus,
        ConflictPair, ContextConfig, ContextFormat, ThinkResult,
    };
    pub use hirn_engine::ql::revision_query_result_to_json;
    pub use hirn_engine::{ParseError, QueryPlan, QueryResult, Statement};
}

/// Provenance tracking and audit trail.
///
/// Every memory carries a full lineage trace: origin, mutation history,
/// evidence chain, and contributing agents.
pub mod provenance {
    pub use hirn_core::audit::{AuditAction, AuditEntry};
    pub use hirn_core::provenance::{EvidenceRef, Mutation, Provenance};
    pub use hirn_core::types::{MutationTrigger, Origin};
}

/// Memory security: anomaly detection and quarantine.
///
/// Adversarial memory injection defense (OWASP ASI06). Anomalous memories
/// are quarantined pending review. Bayesian trust scoring for cross-agent
/// memory integrity.
pub mod security {
    pub use hirn_core::QuarantinedRecordKind;
    pub use hirn_engine::{QuarantineEntry, QuarantineStatus};
}

/// Recall and think builder APIs.
pub mod query {
    pub use hirn_engine::recall::{LayerFilter, RecallResult};
    pub use hirn_engine::{RecallBuilder, ThinkBuilder, TraceBuilder, TraceResult};
}

/// Metadata key-value storage.
pub mod metadata {
    pub use hirn_core::metadata::{Metadata, MetadataValue};
}

/// Multi-modal content payloads and composite embedding helpers.
pub mod content {
    pub use hirn_core::content::{
        CompositeEmbeddingPolicy, CompositeModalityWeights, ExternalFetchPolicy, MemoryContent,
    };
}

/// First-class resource memory types: resources, artifacts, hydration, and governance.
pub mod resource {
    pub use hirn_core::resource::{
        DerivedArtifact, DerivedArtifactBuilder, DerivedArtifactId, DerivedArtifactIndexPolicy,
        DerivedArtifactIndexRule, DerivedArtifactKind, EvidenceLink, EvidenceProvenance,
        EvidenceRole, HydrationMode, LogicalResourceId, ModalityProfile, ResourceGovernanceState,
        ResourceId, ResourceIndexPolicy, ResourceIndexRule, ResourceLocation, ResourceObject,
        ResourceObjectBuilder, ResourceQuotaPolicy, ResourceQuotaRule, ResourceQuotaScope,
        ResourceRetentionAction, ResourceRetentionPolicy, ResourceRetentionRule,
        ResourceRevisionId, SecondaryIndexType,
    };
}

// ── Prelude ─────────────────────────────────────────────────────────────

/// Common imports for most hirn use cases.
///
/// ```rust
/// use hirn::prelude::*;
/// ```
///
/// This imports the types needed for 90% of interactions with hirn:
/// the database handle, memory builders, agent context, core enums,
/// and the most-used query types.
pub mod prelude {
    // Database
    pub use crate::Hirn;
    pub use crate::{HirnConfig, HirnError, HirnResult};
    pub use crate::{HirnMemory, MemoryRecallBuilder, MemoryThinkBuilder};
    pub use crate::{MemoryId, RecallSnapshot, RevisionId, Timestamp};

    // Memory records & builders
    pub use hirn_core::episodic::EpisodicRecord;
    pub use hirn_core::procedural::ProceduralRecord;
    pub use hirn_core::record::MemoryRecord;
    pub use hirn_core::semantic::SemanticRecord;
    pub use hirn_core::working::WorkingMemoryEntry;

    // Core enums
    pub use hirn_core::types::{
        AgentId, EdgeRelation, EventType, KnowledgeType, Layer, Namespace, Origin, Priority,
    };

    // Agent context
    pub use hirn_engine::AgentContext;

    // Events
    pub use crate::MemoryEvent;

    // Query
    pub use hirn_engine::ActivationMode;
    pub use hirn_engine::QueryResult;
    pub use hirn_engine::ql::context::ThinkResult;
    pub use hirn_engine::recall::RecallResult;

    // Metadata
    pub use hirn_core::metadata::Metadata;

    // Multi-modal/resource memory
    pub use crate::content::MemoryContent;
    pub use crate::resource::{DerivedArtifactKind, EvidenceRole, HydrationMode, ModalityProfile};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use hirn_storage::memory_store::MemoryStore;
    use std::sync::Arc;

    fn null_storage() -> Arc<dyn hirn_storage::PhysicalStore> {
        Arc::new(MemoryStore::new())
    }

    async fn test_storage(base: &std::path::Path) -> Arc<dyn hirn_storage::PhysicalStore> {
        let lance_path = base.join("lance");
        hirn_storage::HirnDb::open(hirn_storage::HirnDbConfig::local(
            lance_path.to_str().unwrap(),
        ))
        .await
        .unwrap()
        .store_arc()
    }

    fn write_dev_hirn_toml(dir: &std::path::Path, extra: &str) {
        std::fs::write(
            dir.join("hirn.toml"),
            format!("allow_pseudo_embedder_fallback = true\n{extra}"),
        )
        .unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_and_close() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let config = HirnConfig::builder().db_path(&path).build().unwrap();
        let brain = Hirn::open_with_config(config, null_storage())
            .await
            .unwrap();
        let stats = brain.admin().stats().await.unwrap();
        assert_eq!(stats.total_count, 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn agent_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(3)
            .build()
            .unwrap();
        let brain = Hirn::open_with_config(config, test_storage(dir.path()).await)
            .await
            .unwrap();

        let agent = AgentId::new("test_agent").unwrap();
        brain.register_agent(&agent, "Test Agent").await.unwrap();

        let ctx = brain.as_agent(&agent).await.unwrap();

        let episode = EpisodicRecord::builder()
            .content("hirn is a cognitive memory engine")
            .event_type(EventType::Observation)
            .agent_id(agent.clone())
            .build()
            .unwrap();

        let id = ctx.remember(episode).await.unwrap();
        let inspected = ctx.inspect(id).await;
        assert!(inspected.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn multi_agent_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(3)
            .build()
            .unwrap();
        let brain = Hirn::open_with_config(config, test_storage(dir.path()).await)
            .await
            .unwrap();

        let a = AgentId::new("agent_a").unwrap();
        let b = AgentId::new("agent_b").unwrap();
        brain.register_agent(&a, "A").await.unwrap();
        brain.register_agent(&b, "B").await.unwrap();

        let ctx_a = brain.as_agent(&a).await.unwrap();
        let ctx_b = brain.as_agent(&b).await.unwrap();

        let ep = EpisodicRecord::builder()
            .content("secret from agent A")
            .event_type(EventType::Observation)
            .agent_id(a.clone())
            .build()
            .unwrap();
        let id = ctx_a.remember(ep).await.unwrap();

        // Agent B cannot see Agent A's private memory
        assert!(ctx_b.inspect(id).await.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn shared_namespace_visible_to_all() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(3)
            .build()
            .unwrap();
        let brain = Hirn::open_with_config(config, test_storage(dir.path()).await)
            .await
            .unwrap();

        let a = AgentId::new("agent_a").unwrap();
        let b = AgentId::new("agent_b").unwrap();
        brain.register_agent(&a, "A").await.unwrap();
        brain.register_agent(&b, "B").await.unwrap();

        let ctx_a = brain.as_agent(&a).await.unwrap();
        let ctx_b = brain.as_agent(&b).await.unwrap();

        let mut ep = EpisodicRecord::builder()
            .content("shared knowledge")
            .event_type(EventType::Observation)
            .agent_id(a.clone())
            .build()
            .unwrap();
        ep.namespace = Namespace::shared();
        let id = ctx_a.remember(ep).await.unwrap();

        // Both agents can see shared memory
        assert!(ctx_a.inspect(id).await.is_ok());
        assert!(ctx_b.inspect(id).await.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn semantic_store_and_retrieve() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(3)
            .build()
            .unwrap();
        let brain = Hirn::open_with_config(config, test_storage(dir.path()).await)
            .await
            .unwrap();

        let agent = AgentId::new("learner").unwrap();
        brain.register_agent(&agent, "Learner").await.unwrap();

        let sem = SemanticRecord::builder()
            .concept("rust_ownership")
            .description("Rust uses ownership and borrowing for memory safety")
            .knowledge_type(KnowledgeType::Propositional)
            .confidence(0.95)
            .agent_id(agent)
            .build()
            .unwrap();

        let id = brain.semantic().store(sem).await.unwrap();
        let retrieved = brain.semantic().get(id).await.unwrap();
        assert_eq!(retrieved.concept, "rust_ownership");
        assert!((retrieved.confidence - 0.95).abs() < f32::EPSILON);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hirnql_execution() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let config = HirnConfig::builder().db_path(&path).build().unwrap();
        let brain = Hirn::open_with_config(config, null_storage())
            .await
            .unwrap();

        let result = brain
            .ql()
            .execute(r#"RECALL episodic ABOUT "test" LIMIT 5"#)
            .await;
        assert!(result.is_ok());
    }

    // Verify that the type alias works seamlessly
    #[tokio::test(flavor = "multi_thread")]
    async fn hirn_type_alias_is_hirn_storage() {
        fn accepts_hirn(_brain: &Hirn) {}
        fn accepts_hirn_storage(_brain: &crate::HirnDB) {}

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");
        let config = HirnConfig::builder().db_path(&path).build().unwrap();
        let brain = Hirn::open_with_config(config, null_storage())
            .await
            .unwrap();

        // Both work — Hirn IS HirnDB
        accepts_hirn(&brain);
        accepts_hirn_storage(&brain);
    }

    // ── Zero-Config API ─────────────────────────────────────────────

    mod zero_config {
        use super::*;

        /// Open → remember → think → relevant context returned (end-to-end)
        #[tokio::test(flavor = "multi_thread")]
        async fn open_remember_think_e2e() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            std::fs::write(
                dir.path().join("hirn.toml"),
                "allow_pseudo_embedder_fallback = true\n",
            )
            .unwrap();
            let memory = HirnMemory::open(&path).await.unwrap();

            memory.remember("User prefers dark mode").await.unwrap();
            memory.remember("User likes Vim keybindings").await.unwrap();
            memory.remember("User's timezone is UTC+1").await.unwrap();

            let ctx = memory
                .think("What are the user's preferences?", 2048)
                .await
                .unwrap();
            // Context should be non-empty.
            assert!(!ctx.context.is_empty(), "think context should be non-empty");
        }

        /// Explicit pseudo fallback keeps the facade usable in dev/test mode.
        #[tokio::test(flavor = "multi_thread")]
        async fn explicit_pseudo_embedder_fallback_works() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            let mut config = HirnConfig::builder()
                .db_path(&path)
                .allow_pseudo_embedder_fallback(true)
                .build()
                .unwrap();
            config.admission_enabled = true;
            let memory = HirnMemory::open_with_config(config).await.unwrap();
            let id = memory
                .remember("Testing with pseudo embedder")
                .await
                .unwrap();
            // Verify a valid MemoryId was returned.
            let record = memory.db().admin().get_memory(id).await.unwrap();
            match record {
                hirn_core::record::MemoryRecord::Episodic(ep) => {
                    assert_eq!(ep.content, "Testing with pseudo embedder");
                }
                _ => panic!("expected episodic record"),
            }
        }

        #[tokio::test(flavor = "multi_thread")]
        async fn open_fails_closed_without_provider_or_explicit_pseudo_opt_in() {
            if crate::ProviderRegistry::from_env_strict()
                .embedder()
                .is_some()
            {
                return;
            }

            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            let error = HirnMemory::open(&path).await.err().unwrap();
            assert!(matches!(
                error,
                HirnError::InvalidConfig { ref field, .. }
                    if field == "allow_pseudo_embedder_fallback"
            ));
        }

        /// 5-line example from RFC compiles and runs.
        #[tokio::test(flavor = "multi_thread")]
        async fn five_line_example() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            std::fs::write(
                dir.path().join("hirn.toml"),
                "allow_pseudo_embedder_fallback = true\n",
            )
            .unwrap();
            let memory = HirnMemory::open(&path).await.unwrap();
            memory.remember("User prefers dark mode").await.unwrap();
            let ctx = memory
                .think("What are the user's UI preferences?", 2048)
                .await
                .unwrap();
            assert!(!ctx.context.is_empty());
        }

        /// Auto entity extraction: entities are extracted from remember text.
        #[tokio::test(flavor = "multi_thread")]
        async fn auto_entity_extraction() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            std::fs::write(
                dir.path().join("hirn.toml"),
                "allow_pseudo_embedder_fallback = true\n",
            )
            .unwrap();
            let memory = HirnMemory::open(&path).await.unwrap();
            let id = memory
                .remember("User prefers Dark Mode in Visual Studio Code")
                .await
                .unwrap();

            // Inspect the stored record to verify entities were extracted.
            let record = memory.db().admin().get_memory(id).await.unwrap();
            match record {
                hirn_core::record::MemoryRecord::Episodic(ep) => {
                    assert!(
                        !ep.entities.is_empty(),
                        "entities should be auto-extracted; got none"
                    );
                    let names: Vec<&str> = ep.entities.iter().map(|e| e.name.as_str()).collect();
                    // RegexEntityExtractor finds capitalized multi-word sequences.
                    assert!(
                        names
                            .iter()
                            .any(|n| n.contains("Dark Mode") || n.contains("Visual Studio")),
                        "expected entity like 'Dark Mode' or 'Visual Studio Code', got {names:?}"
                    );
                }
                _ => panic!("expected episodic record"),
            }
        }

        /// prelude re-exports HirnMemory.
        #[tokio::test(flavor = "multi_thread")]
        async fn prelude_exports_hirn_memory() {
            // This test just verifies compilation — HirnMemory is available via prelude.
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            std::fs::write(
                dir.path().join("hirn.toml"),
                "allow_pseudo_embedder_fallback = true\n",
            )
            .unwrap();
            let _memory: HirnMemory = HirnMemory::open(&path).await.unwrap();
        }
    }

    // ── HirnQL API ──────────────────────────────────────────────────

    mod hirnql_api {
        use super::*;

        /// RECALL query → results match expected.
        #[tokio::test(flavor = "multi_thread")]
        async fn recall_query_returns_results() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            write_dev_hirn_toml(dir.path(), "");
            let memory = HirnMemory::open(&path).await.unwrap();

            memory
                .remember("JWT token should expire after 15 minutes")
                .await
                .unwrap();
            memory
                .remember("Auth uses OAuth2 with PKCE flow")
                .await
                .unwrap();

            let result = memory
                .query(r#"RECALL episodic ABOUT "auth" LIMIT 10"#)
                .await
                .unwrap();

            match result {
                hirn_engine::ql::QueryResult::Records(rr) => {
                    assert!(rr.records_returned > 0, "expected some records");
                    assert!(rr.query_time_ms >= 0.0);
                }
                other => panic!("expected Records, got {other:?}"),
            }
        }

        /// REMEMBER is intentionally outside embedded HirnQL.
        #[tokio::test(flavor = "multi_thread")]
        async fn remember_via_hirnql_is_rejected() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            write_dev_hirn_toml(dir.path(), "");
            let memory = HirnMemory::open(&path).await.unwrap();

            let error = memory
                .query(r#"REMEMBER episode CONTENT "Database uses connection pooling""#)
                .await
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("REMEMBER is not supported via embedded HirnQL anymore")
            );
        }

        /// THINK via HirnQL → context assembled.
        #[tokio::test(flavor = "multi_thread")]
        async fn think_via_hirnql() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            write_dev_hirn_toml(dir.path(), "");
            let memory = HirnMemory::open(&path).await.unwrap();

            memory
                .remember("User uses Neovim with LSP support")
                .await
                .unwrap();

            let result = memory
                .query(r#"THINK ABOUT "editor setup" BUDGET 1024"#)
                .await
                .unwrap();

            match result {
                hirn_engine::ql::QueryResult::Records(rr) => {
                    // THINK produces a context string.
                    assert!(rr.context.is_some());
                }
                other => panic!("expected Records with context, got {other:?}"),
            }
        }

        /// FORGET is intentionally outside embedded HirnQL.
        #[tokio::test(flavor = "multi_thread")]
        async fn forget_via_hirnql_is_rejected() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            write_dev_hirn_toml(dir.path(), "");
            let memory = HirnMemory::open(&path).await.unwrap();

            let id = memory.remember("Temporary note to delete").await.unwrap();

            let error = memory.query(&format!("FORGET \"{id}\"")).await.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("FORGET is not supported via embedded HirnQL anymore")
            );
        }

        /// Invalid HirnQL → error with position info.
        #[tokio::test(flavor = "multi_thread")]
        async fn invalid_hirnql_error_position() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            write_dev_hirn_toml(dir.path(), "");
            let memory = HirnMemory::open(&path).await.unwrap();

            let err = memory.query("INVALID SYNTAX HERE").await.unwrap_err();

            let msg = err.to_string();
            // Error should contain position info (line:column).
            assert!(
                msg.contains(':'),
                "error should contain position info, got: {msg}"
            );
        }
    }

    // ── Builder API ─────────────────────────────────────────────────

    mod builder_api {
        use super::*;

        /// Builder produces results — recall builder with chained options.
        #[tokio::test(flavor = "multi_thread")]
        async fn recall_builder_works() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            write_dev_hirn_toml(dir.path(), "");
            let memory = HirnMemory::open(&path).await.unwrap();

            memory
                .remember("JWT tokens expire after 15 minutes")
                .await
                .unwrap();
            memory
                .remember("OAuth2 PKCE flow for authentication")
                .await
                .unwrap();

            let results = memory
                .recall_builder("auth tokens")
                .limit(5)
                .episodic_only()
                .execute()
                .await
                .unwrap();

            assert!(!results.is_empty(), "builder recall should return results");
        }

        /// Think builder with budget produces context.
        #[tokio::test(flavor = "multi_thread")]
        async fn think_builder_with_budget() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            write_dev_hirn_toml(dir.path(), "");
            let memory = HirnMemory::open(&path).await.unwrap();

            memory
                .remember("User prefers dark mode in all editors")
                .await
                .unwrap();
            memory.remember("UI theme is Gruvbox Dark").await.unwrap();

            let ctx = memory
                .think_builder("editor theme preferences")
                .budget(2048)
                .execute()
                .await
                .unwrap();

            assert!(
                !ctx.context.is_empty(),
                "think builder should produce context"
            );
        }

        /// Builder produces same results as equivalent HirnQL string.
        #[tokio::test(flavor = "multi_thread")]
        async fn builder_matches_hirnql() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            write_dev_hirn_toml(dir.path(), "");
            let memory = HirnMemory::open(&path).await.unwrap();

            memory
                .remember("Kubernetes pods use resource limits")
                .await
                .unwrap();

            // Both should return records.
            let builder_results = memory
                .recall_builder("kubernetes")
                .limit(10)
                .execute()
                .await
                .unwrap();

            let ql_result = memory
                .query(r#"RECALL episodic ABOUT "kubernetes" LIMIT 10"#)
                .await
                .unwrap();

            match ql_result {
                hirn_engine::ql::QueryResult::Records(rr) => {
                    // Both should find at least 1 record.
                    assert!(!builder_results.is_empty());
                    assert!(rr.records_returned > 0);
                }
                other => panic!("expected Records, got {other:?}"),
            }
        }

        /// Chain all options → complex query executes correctly.
        #[tokio::test(flavor = "multi_thread")]
        async fn chain_all_options() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            write_dev_hirn_toml(dir.path(), "");
            let memory = HirnMemory::open(&path).await.unwrap();

            memory
                .remember("System uses Redis for caching")
                .await
                .unwrap();

            let results = memory
                .recall_builder("caching")
                .limit(5)
                .episodic_only()
                .activation(hirn_engine::ActivationMode::Spreading)
                .depth(2)
                .execute()
                .await
                .unwrap();

            // Should not error, results may or may not be empty.
            let _ = results;
        }
    }

    // ── Auto-Configuration & Defaults ───────────────────────────────

    mod auto_config {
        use super::*;

        /// Open with no config → defaults applied, admission pipeline active.
        #[tokio::test(flavor = "multi_thread")]
        async fn open_no_config_defaults_applied() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");
            write_dev_hirn_toml(dir.path(), "");
            let memory = HirnMemory::open(&path).await.unwrap();

            // Admission pipeline should be set up by default.
            assert!(
                memory.db().admission_pipeline().is_some(),
                "default admission pipeline should be active"
            );

            // Default admission thresholds should match expected values.
            let cfg = memory.db().config();
            assert!(
                cfg.admission_enabled,
                "admission should be enabled by default"
            );
            assert!((cfg.admission_surprise_threshold - 0.3).abs() < f32::EPSILON);
            assert!((cfg.admission_duplicate_threshold - 0.95).abs() < f32::EPSILON);
            assert_eq!(cfg.admission_token_budget_limit, 500_000);

            // Consolidation schedule should have a sensible default.
            assert_eq!(cfg.consolidation_interval_secs, 3600);

            // System should work end-to-end.
            memory.remember("test memory").await.unwrap();
            let ctx = memory.think("test", 2048).await.unwrap();
            assert!(!ctx.context.is_empty());
        }

        /// Open with hirn.toml → config overrides defaults.
        #[tokio::test(flavor = "multi_thread")]
        async fn open_hirn_toml_overrides_defaults() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("brain");

            // Write a partial hirn.toml in the brain directory (parent of db file).
            let toml_content = r#"
token_budget = 8192
consolidation_interval_secs = 1800
admission_surprise_threshold = 0.5
"#;
            write_dev_hirn_toml(dir.path(), toml_content);

            let memory = HirnMemory::open(&db_path).await.unwrap();
            let cfg = memory.db().config();

            // Overridden values from hirn.toml.
            assert_eq!(cfg.token_budget, 8192);
            assert_eq!(cfg.consolidation_interval_secs, 1800);
            assert!((cfg.admission_surprise_threshold - 0.5).abs() < f32::EPSILON);

            // Defaults for fields NOT in the TOML.
            assert!((cfg.admission_duplicate_threshold - 0.95).abs() < f32::EPSILON);
        }

        /// Open with HirnConfig → programmatic config works.
        #[tokio::test(flavor = "multi_thread")]
        async fn open_with_hirnconfig_programmatic() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("brain");

            let config = HirnConfig::builder()
                .db_path(&path)
                .token_budget(16384)
                .consolidation_interval_secs(7200)
                .allow_pseudo_embedder_fallback(true)
                .build()
                .unwrap();

            let memory = HirnMemory::open_with_config(config).await.unwrap();
            let cfg = memory.db().config();

            assert_eq!(cfg.token_budget, 16384);
            assert_eq!(cfg.consolidation_interval_secs, 7200);

            // open_with_config uses config as-is — admission_enabled is
            // whatever the caller set (default builder = false).
            assert!(!cfg.admission_enabled);
        }

        /// Config precedence: HirnConfig > hirn.toml > defaults.
        #[tokio::test(flavor = "multi_thread")]
        async fn config_precedence() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("brain");

            // hirn.toml sets token_budget=8192 and consolidation=1800.
            let toml_content = r#"
token_budget = 8192
consolidation_interval_secs = 1800
"#;
            write_dev_hirn_toml(dir.path(), toml_content);

            // open() loads hirn.toml → overrides defaults.
            let memory = HirnMemory::open(&db_path).await.unwrap();
            let cfg = memory.db().config();
            assert_eq!(cfg.token_budget, 8192, "hirn.toml > defaults");
            assert_eq!(
                cfg.consolidation_interval_secs, 1800,
                "hirn.toml > defaults"
            );

            // open_with_config() ignores hirn.toml → programmatic wins.
            // Use a different db path to avoid file lock.
            let dir2 = tempfile::tempdir().unwrap();
            let db_path2 = dir2.path().join("brain");
            // Write the same hirn.toml in dir2 too.
            write_dev_hirn_toml(dir2.path(), toml_content);

            let explicit = HirnConfig::builder()
                .db_path(&db_path2)
                .token_budget(32768)
                .allow_pseudo_embedder_fallback(true)
                .build()
                .unwrap();
            let memory2 = HirnMemory::open_with_config(explicit).await.unwrap();
            let cfg2 = memory2.db().config();
            assert_eq!(cfg2.token_budget, 32768, "HirnConfig > hirn.toml");
            // consolidation_interval_secs was NOT set programmatically →
            // defaults (3600), NOT hirn.toml's 1800.
            assert_eq!(
                cfg2.consolidation_interval_secs, 3600,
                "HirnConfig defaults, not hirn.toml"
            );
        }

        /// Invalid config → clear error at open time.
        #[tokio::test(flavor = "multi_thread")]
        async fn invalid_config_error_at_open() {
            let dir = tempfile::tempdir().unwrap();
            let db_path = dir.path().join("brain");

            // hnsw_m=0 is invalid (must be >= 2).
            let toml_content = "hnsw_m = 0\n";
            std::fs::write(dir.path().join("hirn.toml"), toml_content).unwrap();

            let result = HirnMemory::open(&db_path).await;
            assert!(
                result.is_err(),
                "invalid hirn.toml should cause open to fail"
            );
            let err = match result {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected error for invalid hirn.toml"),
            };
            assert!(
                err.contains("hnsw_m"),
                "error should mention the invalid field: {err}"
            );
        }
    }
}
