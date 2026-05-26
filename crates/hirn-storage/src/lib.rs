pub mod cache;
pub mod compaction;
pub mod datasets;
pub mod embed_cache_ops;
pub mod embedding_registry;
pub mod engine;
pub mod error;
pub mod fragment_cache;
pub mod index;
pub mod lance_store;
pub mod memory_store;
pub mod multimodal;
pub mod multivector;
pub mod mutation_envelope_ops;
pub mod namespace;
pub mod policy_store;
pub mod reranker;
pub mod resource_ops;
pub mod scan;
pub mod session;
pub mod store;
pub mod with_embeddings;

pub use compaction::{
    LifecycleCompactOptions, LifecycleCompactResult, Summarizer, lifecycle_compact,
};
pub use embedding_registry::EmbeddingRegistry;
pub use engine::{HirnDb, HirnDbConfig};
pub use error::HirnDbError;
pub use fragment_cache::{FragmentCache, FragmentCacheConfig};
pub use mutation_envelope_ops::{
    MutationEnvelopeRecord, MutationEnvelopeState, append_mutation_envelope,
    append_mutation_envelopes, get_mutation_envelope, list_mutation_envelopes,
    list_pending_mutation_envelopes, replace_mutation_envelope, replace_mutation_envelopes,
    update_mutation_envelope_state,
};
pub use policy_store::{CURRENT_PRINCIPAL, NamespacePolicy, PolicyEnforcedStore};
pub use reranker::{
    ColBERTReranker, LinearCombinationReranker, RELEVANCE_SCORE_COLUMN, RRFReranker, Reranker,
    RerankerPipeline,
};
pub use resource_ops::{
    DerivedArtifactInput, HydratedResource, RESOURCE_HEAD_TRANSITION_KIND,
    ResourceGovernanceUpdate, ResourceRetentionApplyResult, ResourceSupersession,
    apply_resource_retention_policy, build_configured_blob_resource,
    configure_audio_resource_builder, derived_artifact_evidence_role,
    evidence_links_for_derived_artifacts, fetch_resource, get_resource, get_resource_head,
    list_derived_artifacts, list_resource_revisions, load_resource_blob,
    persist_default_derived_artifacts, persist_derived_artifact, persist_resource,
    persist_resource_with_quota_policy, purge_resource, reconcile_pending_resource_blob_staging,
    reconcile_resource_head_mutations, redact_resource, supersede_resource,
    supersede_resource_with_quota_policy, text_backed_resource_checksum,
};
pub use store::{DistanceMetric, NormalizeMethod, PhysicalStore, RecordBatchStream};
pub use with_embeddings::{EmbeddingMapping, WithEmbeddings};
