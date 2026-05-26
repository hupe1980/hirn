//! Token-aware context assembly pipeline.
//!
//! Implements the "last mile" from memory retrieval to LLM-ready context:
//! layer-priority budget allocation, progressive compression, contradiction
//! surfacing, and structured formatting.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::Write;

use arrow_array::{Array, Float32Array, RecordBatch, StringArray, UInt32Array};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use hirn_core::error::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::record::MemoryRecord;
use hirn_core::resource::ResourceGovernanceState;
use hirn_core::revision::{LogicalMemoryId, RecallSnapshot, RevisionId, RevisionState};
use hirn_core::semantic::SemanticRecord;
use hirn_core::tokenizer::Tokenizer;
use hirn_core::types::{AgentId, EdgeRelation, Layer, Namespace};
use hirn_core::working::WorkingMemoryEntry;
use hirn_core::{ConflictResolutionPolicy, HirnConfig};

use crate::GraphEdge;
use crate::db::HirnDB;
use crate::graph_store::GraphStore;
use crate::recall::ResourceEvidenceSummary;
use crate::resource_presentation::{
    PreviewPackageCache, PreviewPackageSurface, ResourcePreviewPackage, ResourceScoreAttribution,
    package_resource_preview_packages_for_evidence, resource_preview_packages_to_json,
    resource_score_attribution_to_json,
};
use crate::result_json::{resource_evidence_to_json, resource_hydration_to_json};

use super::results::ScoredMemory;

#[async_trait]
pub(crate) trait ConflictReadRuntime: Send + Sync {
    fn config(&self) -> &HirnConfig;

    fn graph_store(&self) -> &dyn GraphStore;

    async fn get_memory(&self, id: MemoryId) -> HirnResult<MemoryRecord>;

    async fn semantic_head_for_logical_id(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<SemanticRecord>;

    async fn semantic_revision_for_logical_id_at_snapshot(
        &self,
        logical_memory_id: LogicalMemoryId,
        snapshot: RecallSnapshot,
    ) -> HirnResult<Option<SemanticRecord>>;
}

#[async_trait]
impl ConflictReadRuntime for HirnDB {
    fn config(&self) -> &HirnConfig {
        HirnDB::config(self)
    }

    fn graph_store(&self) -> &dyn GraphStore {
        HirnDB::graph_store(self)
    }

    async fn get_memory(&self, id: MemoryId) -> HirnResult<MemoryRecord> {
        HirnDB::get_memory(self, id).await
    }

    async fn semantic_head_for_logical_id(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> HirnResult<SemanticRecord> {
        HirnDB::semantic_head_for_logical_id(self, logical_memory_id).await
    }

    async fn semantic_revision_for_logical_id_at_snapshot(
        &self,
        logical_memory_id: LogicalMemoryId,
        snapshot: RecallSnapshot,
    ) -> HirnResult<Option<SemanticRecord>> {
        HirnDB::semantic_revision_for_logical_id_at_snapshot(self, logical_memory_id, snapshot)
            .await
    }
}

// ── Configuration ──────────────────────────────────────────────────────

/// Configuration for context assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextFeatures(u8);

impl ContextFeatures {
    const GRAPH_CONTEXT: u8 = 0b0001;
    const CAUSAL_CHAINS: u8 = 0b0010;
    const CONTRADICTIONS: u8 = 0b0100;
    const RESOURCE_PREVIEWS: u8 = 0b1000;

    #[must_use]
    pub const fn all() -> Self {
        Self(
            Self::GRAPH_CONTEXT
                | Self::CAUSAL_CHAINS
                | Self::CONTRADICTIONS
                | Self::RESOURCE_PREVIEWS,
        )
    }

    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[must_use]
    pub const fn include_graph_context(self) -> bool {
        self.0 & Self::GRAPH_CONTEXT != 0
    }

    #[must_use]
    pub const fn include_causal_chains(self) -> bool {
        self.0 & Self::CAUSAL_CHAINS != 0
    }

    #[must_use]
    pub const fn surface_contradictions(self) -> bool {
        self.0 & Self::CONTRADICTIONS != 0
    }

    #[must_use]
    pub const fn package_resource_previews(self) -> bool {
        self.0 & Self::RESOURCE_PREVIEWS != 0
    }

    #[must_use]
    pub const fn with_graph_context(self, enabled: bool) -> Self {
        if enabled {
            Self(self.0 | Self::GRAPH_CONTEXT)
        } else {
            Self(self.0 & !Self::GRAPH_CONTEXT)
        }
    }

    #[must_use]
    pub const fn with_causal_chains(self, enabled: bool) -> Self {
        if enabled {
            Self(self.0 | Self::CAUSAL_CHAINS)
        } else {
            Self(self.0 & !Self::CAUSAL_CHAINS)
        }
    }

    #[must_use]
    pub const fn with_surface_contradictions(self, enabled: bool) -> Self {
        if enabled {
            Self(self.0 | Self::CONTRADICTIONS)
        } else {
            Self(self.0 & !Self::CONTRADICTIONS)
        }
    }

    #[must_use]
    pub const fn with_resource_previews(self, enabled: bool) -> Self {
        if enabled {
            Self(self.0 | Self::RESOURCE_PREVIEWS)
        } else {
            Self(self.0 & !Self::RESOURCE_PREVIEWS)
        }
    }
}

impl Default for ContextFeatures {
    fn default() -> Self {
        Self::all()
    }
}

#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Total token budget for the assembled context.
    pub token_budget: usize,
    /// Fraction of budget reserved for working memory (0.0–1.0).
    pub working_memory_reserve: f32,
    /// Relative weight for semantic vs episodic allocation.
    pub semantic_weight: f32,
    /// Score below which summaries replace full content.
    pub compression_threshold: f32,
    /// Maximum number of episodic entries to include.
    pub max_episodic_entries: usize,
    /// Optional context assembly features.
    pub features: ContextFeatures,
    /// Maximum preview packages to attach per included record.
    pub max_resource_previews_per_entry: usize,
    /// Maximum characters retained from a packaged preview artifact.
    pub max_resource_preview_chars: usize,
    /// Output format for the assembled context.
    pub output_format: ContextFormat,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            token_budget: 4096,
            working_memory_reserve: 0.2,
            semantic_weight: 0.6,
            compression_threshold: 0.4,
            max_episodic_entries: 50,
            features: ContextFeatures::all(),
            max_resource_previews_per_entry: 1,
            max_resource_preview_chars: 160,
            output_format: ContextFormat::Structured,
        }
    }
}

impl ContextConfig {
    #[must_use]
    pub fn from_hirn_config(cfg: &HirnConfig) -> Self {
        Self {
            token_budget: cfg.token_budget as usize,
            working_memory_reserve: cfg.working_memory_reserve,
            max_resource_previews_per_entry: cfg.think_preview_package_max_previews,
            max_resource_preview_chars: cfg.think_preview_package_max_chars,
            ..Self::default()
        }
    }
}

/// Output format for assembled context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextFormat {
    /// Sectioned output with clear headers.
    Structured,
    /// Flowing narrative text.
    Narrative,
    /// Machine-readable JSON.
    Json,
}

// ── Result types ───────────────────────────────────────────────────────

/// Result of a THINK context assembly.
#[derive(Debug, Clone)]
pub struct ThinkResult {
    /// The assembled context string, ready for LLM consumption.
    pub context: String,
    /// Token count of the assembled context.
    pub token_count: usize,
    /// IDs of records included in the context.
    pub records_included: Vec<MemoryId>,
    /// Number of candidate records that were excluded.
    pub records_excluded_count: usize,
    /// Detected contradictions.
    pub contradictions: Vec<ConflictPair>,
    /// Grouped contradiction context derived from contradiction edges.
    pub conflict_groups: Vec<ConflictGroup>,
    /// Query execution time in milliseconds.
    pub query_time_ms: f64,
    /// Score distribution of included records.
    pub score_distribution: ScoreDistribution,
}

/// Wire-format for `ContextAssemblyExec` output.
///
/// JSON-serialised by the engine's `ScopedContextAssemblyRuntime`, and
/// deserialised by `decode_compiled_think_assembly_from_batches` in
/// `query_exec.rs`.  Carries both the assembled context text and the
/// hydrated `ScoredMemory` records so the engine can reconstruct a full
/// `QueryResult::Records` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkAssemblyOutput {
    /// The assembled context string, ready for LLM consumption.
    pub context: String,
    /// Token count of the assembled context.
    pub token_count: usize,
    /// Hydrated, scored records (all candidates after post-load filters).
    pub records: Vec<super::results::ScoredMemory>,
    /// IDs of records included in the assembled context (subset of `records`).
    pub records_included: Vec<MemoryId>,
    /// Number of candidates excluded by token budget.
    pub records_excluded_count: usize,
    /// Detected contradictions.
    pub contradictions: Vec<ConflictPair>,
    /// Grouped contradiction context.
    pub conflict_groups: Vec<ConflictGroup>,
    /// Score distribution.
    pub score_distribution: ScoreDistribution,
}

/// A pair of contradicting memories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictPair {
    pub memory_a: MemoryId,
    pub memory_b: MemoryId,
    pub content_a: String,
    pub content_b: String,
    pub confidence: f32,
    pub source_reliability_a: f32,
    pub source_reliability_b: f32,
}

/// The per-member state exposed for a grouped belief conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictMemberStatus {
    Active,
    Superseded,
    Retracted,
    Quarantined,
    Merged,
}

/// The current arbitration state for a grouped belief conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictArbitrationStatus {
    Unresolved,
    Resolved,
    Quarantined,
    Superseded,
}

/// A visible member of a grouped belief conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictMember {
    pub memory_id: MemoryId,
    pub logical_memory_id: Option<LogicalMemoryId>,
    pub revision_id: Option<RevisionId>,
    pub status: ConflictMemberStatus,
    pub layer: Layer,
    pub content: String,
    pub in_result_set: bool,
    pub source_reliability: f32,
    #[serde(skip)]
    recency_basis_ms: i64,
}

/// A grouped belief conflict derived from contradiction connected components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictGroup {
    pub conflict_id: String,
    pub members: Vec<ConflictMember>,
    pub omitted_member_count: usize,
    pub pair_count: usize,
    pub confidence: f32,
    pub evidence_count: usize,
    pub source_reliability: f32,
    pub arbitration_status: ConflictArbitrationStatus,
    pub authoritative_memory_id: Option<MemoryId>,
    pub preferred_memory_id: Option<MemoryId>,
}

/// Statistics about the scores of included records.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScoreDistribution {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
}

impl Default for ScoreDistribution {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
        }
    }
}

// ── Internal types ─────────────────────────────────────────────────────

/// A candidate memory classified for assembly.
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    id: MemoryId,
    layer: Layer,
    full_content: String,
    summary: String,
    score: f32,
    trust_score: f32,
    /// Test-only instrumentation for classify_candidates().
    #[cfg_attr(not(test), allow(dead_code))]
    token_count_full: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    token_count_summary: usize,
    /// Pre-computed token costs for each compression level's composed text.
    /// Populated by `finalize_candidate_render_tokens` after trust scoring.
    /// Zero means not yet computed; `select_candidate_render` falls back to
    /// the tokenizer in that case.
    tokens_full: usize,
    tokens_summary: usize,
    tokens_entity: usize,
    is_contradiction: bool,
    entities: Vec<String>,
    resource_evidence: Vec<ResourceEvidenceSummary>,
    resource_preview_packages: Vec<ResourcePreviewPackage>,
    resource_score_attribution: Vec<ResourceScoreAttribution>,
}

#[derive(Debug, Clone)]
struct ContextEntry {
    id: MemoryId,
    content: String,
    /// Token cost of `content` in isolation (does not include per-entry format
    /// overhead such as bullet prefixes or JSON wrappers).  Used by
    /// `fit_context_to_budget` to estimate total tokens without re-serializing
    /// the entire context on every binary-search or greedy-trim iteration.
    token_cost: usize,
    resource_evidence: Vec<ResourceEvidenceSummary>,
    resource_preview_packages: Vec<ResourcePreviewPackage>,
    resource_score_attribution: Vec<ResourceScoreAttribution>,
}

#[derive(Debug, Clone, Default)]
struct ContextSections {
    working_memory: Vec<ContextEntry>,
    contradictions: Vec<String>,
    semantic: Vec<ContextEntry>,
    episodic: Vec<ContextEntry>,
    procedural: Vec<ContextEntry>,
    graph_connected: Vec<String>,
    causal_upstream: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct ContextSectionLengths {
    working_memory: usize,
    contradictions: usize,
    semantic: usize,
    episodic: usize,
    procedural: usize,
    graph_connected: usize,
    causal_upstream: usize,
}

impl ContextSections {
    fn included_ids(&self) -> Vec<MemoryId> {
        let mut included_ids = Vec::with_capacity(
            self.working_memory.len()
                + self.semantic.len()
                + self.episodic.len()
                + self.procedural.len(),
        );
        for entry in &self.working_memory {
            included_ids.push(entry.id);
        }
        for entry in &self.semantic {
            included_ids.push(entry.id);
        }
        for entry in &self.episodic {
            included_ids.push(entry.id);
        }
        for entry in &self.procedural {
            included_ids.push(entry.id);
        }
        included_ids
    }

    fn trimmable_count(&self) -> usize {
        self.working_memory.len()
            + self.contradictions.len()
            + self.semantic.len()
            + self.episodic.len()
            + self.procedural.len()
            + self.graph_connected.len()
            + self.causal_upstream.len()
    }

    fn section_lengths(&self) -> ContextSectionLengths {
        ContextSectionLengths {
            working_memory: self.working_memory.len(),
            contradictions: self.contradictions.len(),
            semantic: self.semantic.len(),
            episodic: self.episodic.len(),
            procedural: self.procedural.len(),
            graph_connected: self.graph_connected.len(),
            causal_upstream: self.causal_upstream.len(),
        }
    }

    fn keep_lengths_after_trim(&self, trim_count: usize) -> ContextSectionLengths {
        let mut remaining = trim_count;
        let mut lengths = self.section_lengths();

        trim_section_length(&mut lengths.causal_upstream, &mut remaining);
        trim_section_length(&mut lengths.graph_connected, &mut remaining);
        trim_section_length(&mut lengths.procedural, &mut remaining);
        trim_section_length(&mut lengths.episodic, &mut remaining);
        trim_section_length(&mut lengths.semantic, &mut remaining);
        trim_section_length(&mut lengths.contradictions, &mut remaining);
        trim_section_length(&mut lengths.working_memory, &mut remaining);

        lengths
    }

    fn truncate_to_lengths(&mut self, lengths: ContextSectionLengths) {
        self.working_memory.truncate(lengths.working_memory);
        self.contradictions.truncate(lengths.contradictions);
        self.semantic.truncate(lengths.semantic);
        self.episodic.truncate(lengths.episodic);
        self.procedural.truncate(lengths.procedural);
        self.graph_connected.truncate(lengths.graph_connected);
        self.causal_upstream.truncate(lengths.causal_upstream);
    }

    /// Build the per-entry token cost list in trim priority order (lowest
    /// priority first).  Within each section the last entry is listed first
    /// because `keep_lengths_after_trim` removes from the tail.
    fn compute_formatted_entry_costs(
        &self,
        format: ContextFormat,
        tokenizer: &dyn Tokenizer,
    ) -> Vec<usize> {
        let overhead = per_entry_format_overhead(format);
        let mut costs = Vec::with_capacity(self.trimmable_count());

        match format {
            ContextFormat::Structured => {
                // Tokenise the fully-rendered bullet line so the prefix-sum is
                // exact, removing any BPE-boundary approximation error.
                for s in self.causal_upstream.iter().rev() {
                    costs.push(tokenizer.count_tokens(&format!("• {s}\n")));
                }
                for s in self.graph_connected.iter().rev() {
                    costs.push(tokenizer.count_tokens(&format!("• {s}\n")));
                }
                for e in self.procedural.iter().rev() {
                    costs.push(tokenizer.count_tokens(&format!("• {}\n", e.content)));
                }
                for e in self.episodic.iter().rev() {
                    costs.push(tokenizer.count_tokens(&format!("• {}\n", e.content)));
                }
                for e in self.semantic.iter().rev() {
                    costs.push(tokenizer.count_tokens(&format!("• {}\n", e.content)));
                }
                for s in self.contradictions.iter().rev() {
                    costs.push(tokenizer.count_tokens(&format!("{s}\n")));
                }
                for e in self.working_memory.iter().rev() {
                    costs.push(tokenizer.count_tokens(&format!("• {}\n", e.content)));
                }
            }
            ContextFormat::Narrative | ContextFormat::Json => {
                // Per-entry cost depends on surrounding context for these
                // formats; use the same overhead constant as the greedy phase.
                for s in self.causal_upstream.iter().rev() {
                    costs.push(tokenizer.count_tokens(s) + overhead);
                }
                for s in self.graph_connected.iter().rev() {
                    costs.push(tokenizer.count_tokens(s) + overhead);
                }
                for e in self.procedural.iter().rev() {
                    costs.push(e.token_cost + overhead);
                }
                for e in self.episodic.iter().rev() {
                    costs.push(e.token_cost + overhead);
                }
                for e in self.semantic.iter().rev() {
                    costs.push(e.token_cost + overhead);
                }
                for s in self.contradictions.iter().rev() {
                    costs.push(tokenizer.count_tokens(s) + overhead);
                }
                for e in self.working_memory.iter().rev() {
                    costs.push(e.token_cost + overhead);
                }
            }
        }

        costs
    }
}

fn trim_section_length(length: &mut usize, remaining: &mut usize) {
    let trimmed = (*length).min(*remaining);
    *length -= trimmed;
    *remaining -= trimmed;
}

/// Compression level applied to a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompressionLevel {
    Full,
    Summary,
    EntityOnly,
}

/// F-43: Budget allocation per section.
///
/// Implements CONCEPT §10.2 tiered budget allocation:
/// working memory (mandatory) → contradictions → direct results (semantic +
/// episodic + procedural) → graph-connected neighbors → causal-upstream.
#[derive(Debug, Clone)]
struct BudgetAllocation {
    working_memory: usize,
    contradictions: usize,
    semantic: usize,
    episodic: usize,
    procedural: usize,
    graph_connected: usize,
    causal_upstream: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ConflictSummary {
    pub pairs: Vec<ConflictPair>,
    pub groups: Vec<ConflictGroup>,
}

#[derive(Debug, Clone)]
struct ConflictEdgeMeta {
    a: MemoryId,
    b: MemoryId,
    confidence: f32,
    evidence_count: usize,
    resolved: bool,
}

const FALLBACK_CONTRADICTION_CONFIDENCE: f32 = 0.5;

// ── Assembly pipeline ──────────────────────────────────────────────────

/// Assemble context from scored memories with full pipeline:
/// retrieve working memory → detect contradictions → allocate budget →
/// compress → format.
pub async fn assemble_think_context(
    db: &HirnDB,
    actor_id: &AgentId,
    candidates: &[ScoredMemory],
    config: &ContextConfig,
    visible_namespaces: Option<&[Namespace]>,
    // Optional wider pool of already-loaded records. When provided, graph
    // and causal neighbor content is served from this pool before any Lance
    // I/O, typically eliminating the second batch hydration round-trip.
    content_pool: Option<&[ScoredMemory]>,
    // Arrow-native fast path: raw RecordBatches from the DataFusion pipeline
    // (ContextBudgetExec output). When provided, Candidates are built directly
    // from Arrow columns, skipping the secondary Lance hydration round-trip and
    // trust-score computation (~5.5 ms saved). `candidates` is unused when this
    // is Some.
    raw_batches: Option<&[RecordBatch]>,
) -> HirnResult<ThinkResult> {
    let tokenizer = db.tokenizer();

    // 1 + 2. Retrieve working memory in parallel with candidate classification,
    // trust scoring, and token-cost pre-computation. These are independent: working
    // memory is a separate Lance scan; classification + trust are CPU + hot-graph.
    let (working_entries, mut classified) = tokio::join!(
        async { db.working_memory().await.unwrap_or_default() },
        async {
            if let Some(batches) = raw_batches {
                // Arrow fast path: build Candidates from Arrow columns directly.
                // token_count is pre-populated from ContextBudgetExec output.
                // Trust score defaults to 1.0; entities are empty.
                let mut classified = candidates_from_batches(batches, config.token_budget);

                // Preserve resource-side evidence captured during recall hydration.
                // The Arrow payload does not currently carry these fields, but
                // JSON assembly (preview packages / score attribution) depends on them.
                let evidence_by_id: HashMap<
                    MemoryId,
                    (
                        Vec<ResourceEvidenceSummary>,
                        Vec<ResourcePreviewPackage>,
                        Vec<ResourceScoreAttribution>,
                    ),
                > = candidates
                    .iter()
                    .map(|scored| {
                        (
                            scored.record.id(),
                            (
                                scored.resource_evidence.clone(),
                                scored.resource_preview_packages.clone(),
                                scored.resource_score_attribution.clone(),
                            ),
                        )
                    })
                    .collect();
                for candidate in &mut classified {
                    if let Some((
                        resource_evidence,
                        resource_preview_packages,
                        resource_score_attribution,
                    )) = evidence_by_id.get(&candidate.id)
                    {
                        candidate.resource_evidence.clone_from(resource_evidence);
                        candidate
                            .resource_preview_packages
                            .clone_from(resource_preview_packages);
                        candidate
                            .resource_score_attribution
                            .clone_from(resource_score_attribution);
                    }
                }

                // Still run finalize so tokens_entity and any zero tokens_full/
                // tokens_summary are computed (skips non-zero pre-populated values).
                finalize_candidate_render_tokens(&mut classified, tokenizer.as_ref());
                classified
            } else {
                // Classic path: decode ScoredMemory records, compute trust scores.
                let mut classified = classify_candidates(candidates, tokenizer.as_ref());
                compute_trust_scores(db, candidates, &mut classified).await;
                finalize_candidate_render_tokens(&mut classified, tokenizer.as_ref());
                classified
            }
        }
    );
    let sorted_direct_candidates = prepare_sorted_direct_candidates(&classified);

    // 3. Allocate a preliminary budget without contradiction reserve and use it
    // to bound the direct-result slice that can possibly survive the first fit.
    let preliminary_allocation = allocate_budget(
        config,
        &working_entries,
        &[],
        &classified,
        tokenizer.as_ref(),
    );
    let (preliminary_semantic, preliminary_episodic, preliminary_procedural) =
        build_direct_sections(
            &sorted_direct_candidates,
            &preliminary_allocation,
            config,
            tokenizer.as_ref(),
        );

    // 4. Concurrently: detect contradictions among preliminary-section candidates
    //    AND speculatively build graph/causal sections from those same seeds.
    //
    //    Both are async I/O operations with no shared mutable state:
    //    - `collect_conflict_summary` queries Cedar + graph contradiction edges
    //    - `build_graph_and_causal_sections` hydrates graph/causal neighbor content
    //
    //    In the common case (no contradictions) the speculative result is used
    //    directly, overlapping ~5-10 ms of graph hydration with ~5-10 ms of
    //    contradiction detection. When contradictions DO change the final seed,
    //    only the graph/causal sections are rebuilt (rare path).
    let preliminary_seed_ids = collect_direct_section_ids(
        &preliminary_semantic,
        &preliminary_episodic,
        &preliminary_procedural,
    );
    let preliminary_seed_candidates: Vec<ScoredMemory> = candidates
        .iter()
        .filter(|c| preliminary_seed_ids.contains(&c.record.id()))
        .cloned()
        .collect();

    let effective_pool = content_pool.unwrap_or(candidates);
    let pre_needs_graph = config.features.include_graph_context()
        && preliminary_allocation.graph_connected > 0
        && !preliminary_seed_candidates.is_empty();
    let pre_needs_causal = config.features.include_causal_chains()
        && preliminary_allocation.causal_upstream > 0
        && !preliminary_seed_candidates.is_empty();

    let (conflict_summary, speculative_graph_causal) = tokio::join!(
        async {
            if config.features.surface_contradictions() && !preliminary_seed_ids.is_empty() {
                let scoped_candidates = candidates
                    .iter()
                    .filter(|c| preliminary_seed_ids.contains(&c.record.id()))
                    .cloned()
                    .collect::<Vec<_>>();
                collect_conflict_summary(db, &scoped_candidates, visible_namespaces, None).await
            } else {
                ConflictSummary::default()
            }
        },
        async {
            if pre_needs_graph || pre_needs_causal {
                build_graph_and_causal_sections(
                    db,
                    &preliminary_seed_candidates,
                    effective_pool,
                    if pre_needs_graph {
                        preliminary_allocation.graph_connected
                    } else {
                        0
                    },
                    if pre_needs_causal {
                        preliminary_allocation.causal_upstream
                    } else {
                        0
                    },
                    tokenizer.as_ref(),
                )
                .await
            } else {
                (Vec::new(), Vec::new())
            }
        }
    );

    // Apply is_contradiction marks synchronously (no I/O — O(N) scan).
    // `sorted_direct_candidates` borrows `classified`; drop it so we can mutate.
    drop(sorted_direct_candidates);
    let contradiction_ids: HashSet<MemoryId> = conflict_summary
        .groups
        .iter()
        .flat_map(|g| g.members.iter().map(|m| m.memory_id))
        .collect();
    for candidate in &mut classified {
        if contradiction_ids.contains(&candidate.id) {
            candidate.is_contradiction = true;
        }
    }
    // Recreate the sorted view now that `is_contradiction` marks are applied.
    let sorted_direct_candidates = prepare_sorted_direct_candidates(&classified);

    // 5. Allocate the final budget with contradiction overhead included.
    let allocation = allocate_budget(
        config,
        &working_entries,
        &conflict_summary.groups,
        &classified,
        tokenizer.as_ref(),
    );

    // 6. Select and compress candidates within budget.
    let (working_section, _wm_tokens) = build_working_memory_section(
        &working_entries,
        allocation.working_memory,
        tokenizer.as_ref(),
    );

    let (contradiction_section, _contra_tokens) = if config.features.surface_contradictions() {
        build_contradiction_section(
            &conflict_summary.groups,
            allocation.contradictions,
            tokenizer.as_ref(),
        )
    } else {
        (Vec::new(), 0)
    };

    let (semantic_section, episodic_section, procedural_section) =
        if config.features.surface_contradictions() && !conflict_summary.groups.is_empty() {
            // Contradictions were found; the final allocation includes contradiction
            // overhead so the direct-section budgets may differ from the preliminary.
            build_direct_sections(
                &sorted_direct_candidates,
                &allocation,
                config,
                tokenizer.as_ref(),
            )
        } else {
            // No contradictions found, or feature disabled: reuse the preliminary
            // sections already computed above.  Avoids a second section-building pass.
            (
                preliminary_semantic,
                preliminary_episodic,
                preliminary_procedural,
            )
        };

    // Use the speculative graph/causal sections when the final direct-section seed
    // matches the preliminary seed (common case: no contradictions or contradictions
    // did not alter the survivor set). When the seed changed (rare: contradictions
    // shifted the allocation), rebuild from the corrected seed.
    let helper_seed_candidate_ids =
        collect_direct_section_ids(&semantic_section, &episodic_section, &procedural_section);
    let (graph_section, causal_section) = if helper_seed_candidate_ids == preliminary_seed_ids {
        // Speculative sections are valid — use without extra I/O.
        speculative_graph_causal
    } else {
        // Rare path: rebuild from the corrected final seed.
        let helper_seed_candidates = candidates
            .iter()
            .filter(|c| helper_seed_candidate_ids.contains(&c.record.id()))
            .cloned()
            .collect::<Vec<_>>();
        let needs_graph = config.features.include_graph_context()
            && allocation.graph_connected > 0
            && !helper_seed_candidates.is_empty();
        let needs_causal = config.features.include_causal_chains()
            && allocation.causal_upstream > 0
            && !helper_seed_candidates.is_empty();
        if needs_graph || needs_causal {
            build_graph_and_causal_sections(
                db,
                &helper_seed_candidates,
                effective_pool,
                if needs_graph {
                    allocation.graph_connected
                } else {
                    0
                },
                if needs_causal {
                    allocation.causal_upstream
                } else {
                    0
                },
                tokenizer.as_ref(),
            )
            .await
        } else {
            (Vec::new(), Vec::new())
        }
    };

    // 7. Fit formatted output to budget by trimming the section model rather than
    // truncating the rendered string, so structured formats remain syntactically valid.
    let mut sections = ContextSections {
        working_memory: working_section,
        contradictions: contradiction_section,
        semantic: semantic_section,
        episodic: episodic_section,
        procedural: procedural_section,
        graph_connected: graph_section,
        causal_upstream: causal_section,
    };

    if should_package_resource_previews(config) {
        // Trim once before hydrating previews so we do not fetch/package preview
        // artifacts for entries that are definitely outside the initial budget.
        let _ = fit_context_to_budget(
            config.output_format,
            &mut sections,
            config.token_budget,
            tokenizer.as_ref(),
        );
        hydrate_selected_resource_previews(db, actor_id, &mut sections, config).await?;
    }
    let final_context = fit_context_to_budget(
        config.output_format,
        &mut sections,
        config.token_budget,
        tokenizer.as_ref(),
    );
    let final_tokens = tokenizer.count_tokens(&final_context);

    // 8. Collect included/excluded IDs after budget fitting.
    let included_ids = sections.included_ids();
    let total_candidates = candidates.len();
    let records_excluded_count = total_candidates.saturating_sub(included_ids.len());

    // 9. Compute score distribution for the records that survived budget fitting.
    let score_distribution = compute_score_distribution(candidates, &included_ids);

    Ok(ThinkResult {
        context: final_context,
        token_count: final_tokens,
        records_included: included_ids,
        records_excluded_count,
        contradictions: conflict_summary.pairs,
        conflict_groups: conflict_summary.groups,
        query_time_ms: 0.0, // caller sets this
        score_distribution,
    })
}

// ── Step 2: Classify candidates ────────────────────────────────────────

#[cfg(test)]
fn classify_token_counts(
    full_content: &str,
    summary: &str,
    tokenizer: &dyn Tokenizer,
) -> (usize, usize) {
    (
        tokenizer.count_tokens(full_content),
        tokenizer.count_tokens(summary),
    )
}

#[cfg(not(test))]
fn classify_token_counts(
    _full_content: &str,
    _summary: &str,
    _tokenizer: &dyn Tokenizer,
) -> (usize, usize) {
    (0, 0)
}

/// Build `Candidate` structs directly from Arrow `RecordBatch`es produced by
/// the DataFusion THINK pipeline (`ContextBudgetExec` output).
///
/// This is the Arrow-native fast path that eliminates the secondary Lance
/// round-trip. The batch schema must include at minimum: `id`, `content`,
/// `full_content`, `layer`, `score`, `importance`.  Optional columns read
/// when present: `token_count` (pre-computed by `ContextBudgetExec`),
/// `assembly_mode`.
///
/// Trade-offs vs the ScoredMemory path:
/// - `trust_score` defaults to 1.0 (no provenance graph lookup).
/// - `entities = []` (entity-only compression emits empty string; acceptable
///   because entity-only entries are extremely low-score and rarely shown).
/// - Resource evidence fields are empty (not available from Arrow).
pub(crate) fn candidates_from_batches(batches: &[RecordBatch], limit: usize) -> Vec<Candidate> {
    let mut result = Vec::new();

    'outer: for batch in batches {
        if batch.num_rows() == 0 {
            continue;
        }

        let Some(ids) = batch
            .column_by_name("id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        else {
            continue;
        };
        let Some(contents) = batch
            .column_by_name("content")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        else {
            continue;
        };
        let full_contents = batch
            .column_by_name("full_content")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let Some(layers) = batch
            .column_by_name("layer")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        else {
            continue;
        };
        let Some(scores) = batch
            .column_by_name("score")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        else {
            continue;
        };
        let importances = batch
            .column_by_name("importance")
            .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
        // `token_count` emitted by `ContextBudgetExec` — represents tokens for
        // the `content` (display/summary) column.
        let token_counts = batch
            .column_by_name("token_count")
            .and_then(|c| c.as_any().downcast_ref::<UInt32Array>());

        for row in 0..batch.num_rows() {
            if result.len() >= limit {
                break 'outer;
            }

            let id_str = ids.value(row);
            let Ok(id) = MemoryId::parse(id_str) else {
                continue;
            };

            let content = contents.value(row).to_string();
            let full_content = full_contents
                .map(|fc| fc.value(row).to_string())
                .unwrap_or_else(|| content.clone());

            let layer = match layers.value(row) {
                "episodic" => Layer::Episodic,
                "semantic" => Layer::Semantic,
                "procedural" => Layer::Procedural,
                "working" => Layer::Working,
                _ => Layer::Semantic,
            };

            let raw_score = if scores.is_null(row) {
                0.0_f32
            } else {
                scores.value(row)
            };
            // When the composite score is absent or zero (e.g. MemoryStore test
            // environments where no real ANN similarity is computed), fall back to
            // `importance` so that `determine_compression` can still distinguish
            // high-importance records from low-importance ones.  In production
            // (Lance with real vector search) raw_score is always > 0.
            let score = if raw_score == 0.0 {
                importances
                    .and_then(|imp| {
                        if imp.is_null(row) {
                            None
                        } else {
                            Some(imp.value(row))
                        }
                    })
                    .unwrap_or(0.0)
            } else {
                raw_score
            };

            // Pre-populated token count from ContextBudgetExec (tokens for
            // the display/summary text). Use as tokens_summary for all layers.
            // Use as tokens_full only when content == full_content (i.e. no
            // truncation happened — semantic, procedural, short episodics).
            let pre_tokens = match token_counts {
                Some(tc) if !tc.is_null(row) => tc.value(row) as usize,
                _ => 0,
            };
            let same_content = content == full_content;
            let tokens_full = if same_content && pre_tokens > 0 {
                pre_tokens
            } else {
                0 // will be filled in by finalize_candidate_render_tokens
            };
            let tokens_summary = pre_tokens;

            // summary = content (display text); full_content = raw text
            let summary = content;

            result.push(Candidate {
                id,
                layer,
                full_content,
                summary,
                score,
                trust_score: 1.0,
                token_count_full: tokens_full,
                token_count_summary: tokens_summary,
                tokens_full,
                tokens_summary,
                tokens_entity: 0, // always recomputed (fast: entities = [])
                is_contradiction: false,
                entities: vec![],
                resource_evidence: vec![],
                resource_preview_packages: vec![],
                resource_score_attribution: vec![],
            });
        }
    }

    result
}

fn classify_candidates(candidates: &[ScoredMemory], tokenizer: &dyn Tokenizer) -> Vec<Candidate> {
    candidates
        .iter()
        .map(|sm| {
            let (full_content, summary, entities) = match &sm.record {
                MemoryRecord::Episodic(e) => {
                    let entities: Vec<String> =
                        e.entities.iter().map(|er| er.name.clone()).collect();
                    (e.content.clone(), e.summary.clone(), entities)
                }
                MemoryRecord::Semantic(s) => (s.description.clone(), s.concept.clone(), vec![]),
                MemoryRecord::Working(w) => (w.content.clone(), w.content.clone(), vec![]),
                MemoryRecord::Procedural(p) => (p.description.clone(), p.name.clone(), vec![]),
            };

            let (token_count_full, token_count_summary) =
                classify_token_counts(&full_content, &summary, tokenizer);

            Candidate {
                id: sm.record.id(),
                layer: sm.record.layer(),
                full_content,
                summary,
                score: sm.score,
                trust_score: 1.0,
                token_count_full,
                token_count_summary,
                // Populated later by finalize_candidate_render_tokens().
                tokens_full: 0,
                tokens_summary: 0,
                tokens_entity: 0,
                is_contradiction: false,
                entities,
                resource_evidence: sm.resource_evidence.clone(),
                resource_preview_packages: sm.resource_preview_packages.clone(),
                resource_score_attribution: sm.resource_score_attribution.clone(),
            }
        })
        .collect()
}

/// Pre-compute token costs for all three compression levels of each candidate.
///
/// Must be called after `compute_trust_scores` because `compose_candidate_text`
/// includes a low-trust prefix when `trust_score < 0.5`.  The cached counts
/// are used by `select_candidate_render` to avoid repeated tokenizer calls.
///
/// Skips `tokens_full` and `tokens_summary` when already pre-populated (i.e.
/// non-zero) from the Arrow fast path — `candidates_from_batches` fills these
/// in from the `token_count` column emitted by `ContextBudgetExec`.
fn finalize_candidate_render_tokens(classified: &mut [Candidate], tokenizer: &dyn Tokenizer) {
    for candidate in classified.iter_mut() {
        if candidate.tokens_full == 0 {
            candidate.tokens_full =
                tokenizer.count_tokens(&compose_candidate_text(candidate, CompressionLevel::Full));
        }
        if candidate.tokens_summary == 0 {
            candidate.tokens_summary = tokenizer.count_tokens(&compose_candidate_text(
                candidate,
                CompressionLevel::Summary,
            ));
        }
        // tokens_entity is always recomputed: entities is typically empty in the
        // Arrow fast path, making this call trivially fast.
        candidate.tokens_entity = tokenizer.count_tokens(&compose_candidate_text(
            candidate,
            CompressionLevel::EntityOnly,
        ));
    }
}

/// Compute trust scores for classified candidates using provenance and graph data.
async fn compute_trust_scores(
    db: &HirnDB,
    candidates: &[ScoredMemory],
    classified: &mut [Candidate],
) {
    let graph = db.graph_store();
    let candidate_ids = candidates
        .iter()
        .map(|sm| sm.record.id())
        .collect::<Vec<_>>();
    let contradiction_edges =
        get_relation_edges_best_effort(graph, &candidate_ids, EdgeRelation::Contradicts).await;

    for (i, sm) in candidates.iter().enumerate() {
        let provenance = match &sm.record {
            MemoryRecord::Episodic(e) => Some(&e.provenance),
            MemoryRecord::Semantic(s) => Some(&s.provenance),
            MemoryRecord::Working(_) => None,
            MemoryRecord::Procedural(p) => Some(&p.provenance),
        };
        if let Some(prov) = provenance {
            let contra_count = contradiction_edges.get(&sm.record.id()).map_or(0, Vec::len);
            classified[i].trust_score = crate::causal::compute_trust_score(prov, contra_count);
        }
    }
}

async fn get_relation_edges_best_effort(
    graph: &dyn GraphStore,
    node_ids: &[MemoryId],
    relation: EdgeRelation,
) -> HashMap<MemoryId, Vec<GraphEdge>> {
    match graph.get_edges_of_type_many(node_ids, relation).await {
        Ok(edges) => edges,
        Err(_) => {
            let mut edges_by_node = HashMap::with_capacity(node_ids.len());
            for &node_id in node_ids {
                let edges = graph
                    .get_edges_of_type(node_id, relation)
                    .await
                    .unwrap_or_default();
                if !edges.is_empty() {
                    edges_by_node.insert(node_id, edges);
                }
            }
            edges_by_node
        }
    }
}

// ── Step 3: Detect contradictions ──────────────────────────────────────

fn extract_content_str(record: &MemoryRecord) -> &str {
    match record {
        MemoryRecord::Episodic(e) => &e.content,
        MemoryRecord::Semantic(s) => &s.description,
        MemoryRecord::Working(w) => &w.content,
        MemoryRecord::Procedural(p) => &p.description,
    }
}

// ── WITH CONFLICTS for RECALL ──────────────────────────────────────────

/// Detect contradictions among recall results by querying `Contradicts` edges.
///
/// Returns grouped conflict context plus compatibility `ConflictPair`s.
pub(crate) async fn detect_conflicts_for_recall(
    db: &HirnDB,
    candidates: &[ScoredMemory],
    visible_namespaces: Option<&[Namespace]>,
    snapshot: Option<RecallSnapshot>,
) -> ConflictSummary {
    collect_conflict_summary(db, candidates, visible_namespaces, snapshot).await
}

pub(crate) async fn detect_conflicts_for_record(
    db: &HirnDB,
    record: &MemoryRecord,
    visible_namespaces: Option<&[Namespace]>,
) -> ConflictSummary {
    detect_conflicts_for_record_with_runtime(db, record, visible_namespaces).await
}

pub(crate) async fn detect_conflicts_for_record_with_runtime<R>(
    runtime: &R,
    record: &MemoryRecord,
    visible_namespaces: Option<&[Namespace]>,
) -> ConflictSummary
where
    R: ConflictReadRuntime + ?Sized,
{
    collect_conflict_summary_for_record(
        runtime,
        record,
        visible_namespaces,
        RecordConflictResolution::Live,
    )
    .await
}

pub(crate) async fn detect_conflicts_for_exact_record(
    db: &HirnDB,
    record: &MemoryRecord,
    visible_namespaces: Option<&[Namespace]>,
) -> ConflictSummary {
    detect_conflicts_for_exact_record_with_runtime(db, record, visible_namespaces).await
}

pub(crate) async fn detect_conflicts_for_exact_record_with_runtime<R>(
    runtime: &R,
    record: &MemoryRecord,
    visible_namespaces: Option<&[Namespace]>,
) -> ConflictSummary
where
    R: ConflictReadRuntime + ?Sized,
{
    collect_conflict_summary_for_record(
        runtime,
        record,
        visible_namespaces,
        RecordConflictResolution::Exact,
    )
    .await
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordConflictResolution {
    Live,
    Exact,
}

async fn collect_conflict_summary<R>(
    db: &R,
    candidates: &[ScoredMemory],
    visible_namespaces: Option<&[Namespace]>,
    snapshot: Option<RecallSnapshot>,
) -> ConflictSummary
where
    R: ConflictReadRuntime + ?Sized,
{
    let visible_members: BTreeMap<MemoryId, ConflictMember> = candidates
        .iter()
        .map(|candidate| {
            let member = conflict_member_from_scored(candidate, true);
            (member.memory_id, member)
        })
        .collect();

    if visible_members.is_empty() {
        return ConflictSummary::default();
    }

    let mut human_override_members = HashSet::new();
    let mut semantic_contradictions = BTreeMap::new();
    for candidate in candidates {
        if let MemoryRecord::Semantic(record) = &candidate.record {
            if semantic_record_has_human_override(record) {
                human_override_members.insert(record.id);
            }
            if !record.contradiction_ids.is_empty() {
                semantic_contradictions.insert(record.id, record.contradiction_ids.clone());
            }
        }
    }

    let mut seen_pairs: HashSet<(MemoryId, MemoryId)> = HashSet::new();
    let mut adjacency: BTreeMap<MemoryId, Vec<MemoryId>> = BTreeMap::new();
    let mut namespace_cache: BTreeMap<MemoryId, Option<Namespace>> = candidates
        .iter()
        .map(|candidate| {
            (
                candidate.record.id(),
                Some(candidate.record.effective_namespace()),
            )
        })
        .collect();
    let mut pair_edges = Vec::new();
    let mut visible_pairs = Vec::new();

    let graph = db.graph_store();
    let visible_ids = visible_members.keys().copied().collect::<Vec<_>>();
    let contradiction_edges =
        get_relation_edges_best_effort(graph, &visible_ids, EdgeRelation::Contradicts).await;

    for id in visible_ids {
        let Some(edges) = contradiction_edges.get(&id) else {
            continue;
        };

        for edge in edges {
            let other_id = if edge.source == id {
                edge.target
            } else {
                edge.source
            };
            if !conflict_node_is_visible(graph, other_id, visible_namespaces, &mut namespace_cache)
                .await
            {
                continue;
            }

            let pair = normalize_conflict_pair(id, other_id);
            if !seen_pairs.insert(pair) {
                continue;
            }

            let confidence = edge.confidence().unwrap_or(edge.weight).clamp(0.0, 1.0);
            let evidence_count = edge
                .evidence_count()
                .unwrap_or(1)
                .max(1)
                .try_into()
                .unwrap_or(1);

            push_conflict_pair_edge(
                pair,
                confidence,
                evidence_count,
                edge.resolved,
                &mut adjacency,
                &mut pair_edges,
            );

            if let (Some(member_a), Some(member_b)) =
                (visible_members.get(&pair.0), visible_members.get(&pair.1))
            {
                visible_pairs.push(ConflictPair {
                    memory_a: pair.0,
                    memory_b: pair.1,
                    content_a: member_a.content.clone(),
                    content_b: member_b.content.clone(),
                    confidence,
                    source_reliability_a: member_a.source_reliability,
                    source_reliability_b: member_b.source_reliability,
                });
            }
        }
    }

    for (source_id, contradiction_ids) in semantic_contradictions {
        for contradiction_id in contradiction_ids {
            let Some(target_record) = resolve_conflict_target_record(
                db,
                contradiction_id,
                visible_namespaces,
                snapshot,
                false,
            )
            .await
            else {
                continue;
            };
            let target_id = target_record.id();
            if !visible_members.contains_key(&target_id) {
                continue;
            }
            if let MemoryRecord::Semantic(record) = &target_record {
                if semantic_record_has_human_override(record) {
                    human_override_members.insert(record.id);
                }
            }

            let pair = normalize_conflict_pair(source_id, target_id);
            if !seen_pairs.insert(pair) {
                continue;
            }

            push_conflict_pair_edge(
                pair,
                FALLBACK_CONTRADICTION_CONFIDENCE,
                1,
                false,
                &mut adjacency,
                &mut pair_edges,
            );

            if let (Some(member_a), Some(member_b)) =
                (visible_members.get(&pair.0), visible_members.get(&pair.1))
            {
                visible_pairs.push(ConflictPair {
                    memory_a: pair.0,
                    memory_b: pair.1,
                    content_a: member_a.content.clone(),
                    content_b: member_b.content.clone(),
                    confidence: FALLBACK_CONTRADICTION_CONFIDENCE,
                    source_reliability_a: member_a.source_reliability,
                    source_reliability_b: member_b.source_reliability,
                });
            }
        }
    }

    visible_pairs.sort_by_key(|pair| (pair.memory_a, pair.memory_b));
    let policy = resolve_conflict_resolution_policy(db, visible_namespaces);
    let groups = build_conflict_groups(
        &visible_members,
        &adjacency,
        &pair_edges,
        &human_override_members,
        policy,
    );

    ConflictSummary {
        pairs: visible_pairs,
        groups,
    }
}

async fn collect_conflict_summary_for_record<R>(
    db: &R,
    record: &MemoryRecord,
    visible_namespaces: Option<&[Namespace]>,
    resolution: RecordConflictResolution,
) -> ConflictSummary
where
    R: ConflictReadRuntime + ?Sized,
{
    if let Some(visible_namespaces) = visible_namespaces {
        if !visible_namespaces.contains(&record.effective_namespace()) {
            return ConflictSummary::default();
        }
    }

    let seed_id = record.id();
    let graph = db.graph_store();
    let mut visible_members = BTreeMap::new();
    visible_members.insert(seed_id, conflict_member_from_record(record, true));

    let mut loaded_records = BTreeMap::from([(seed_id, record.clone())]);
    let mut human_override_members = HashSet::new();
    if let MemoryRecord::Semantic(record) = record {
        if semantic_record_has_human_override(record) {
            human_override_members.insert(record.id);
        }
    }

    let mut seen_nodes: HashSet<MemoryId> = HashSet::from([seed_id]);
    let mut seen_pairs: HashSet<(MemoryId, MemoryId)> = HashSet::new();
    let mut adjacency: BTreeMap<MemoryId, Vec<MemoryId>> = BTreeMap::new();
    let mut pair_edges = Vec::new();
    let mut queue = VecDeque::from([seed_id]);

    while let Some(current_id) = queue.pop_front() {
        let Some(current_record) = loaded_records.get(&current_id).cloned() else {
            continue;
        };
        let resolution_snapshot = match resolution {
            RecordConflictResolution::Live => {
                conflict_resolution_snapshot_for_record(db, &current_record).await
            }
            RecordConflictResolution::Exact => None,
        };

        if !matches!(resolution, RecordConflictResolution::Exact) {
            let edges = graph
                .get_edges_of_type(current_id, EdgeRelation::Contradicts)
                .await
                .unwrap_or_default();

            for edge in edges {
                let raw_other_id = if edge.source == current_id {
                    edge.target
                } else {
                    edge.source
                };

                let Some(other_record) = resolve_conflict_target_record(
                    db,
                    raw_other_id,
                    visible_namespaces,
                    resolution_snapshot,
                    matches!(resolution, RecordConflictResolution::Exact),
                )
                .await
                else {
                    continue;
                };

                let other_id = other_record.id();
                if other_id == current_id {
                    continue;
                }

                let pair = normalize_conflict_pair(current_id, other_id);
                if seen_pairs.insert(pair) {
                    let confidence = edge.confidence().unwrap_or(edge.weight).clamp(0.0, 1.0);
                    let evidence_count = edge
                        .evidence_count()
                        .unwrap_or(1)
                        .max(1)
                        .try_into()
                        .unwrap_or(1);
                    push_conflict_pair_edge(
                        pair,
                        confidence,
                        evidence_count,
                        edge.resolved,
                        &mut adjacency,
                        &mut pair_edges,
                    );
                }

                if seen_nodes.insert(other_id) {
                    if let MemoryRecord::Semantic(record) = &other_record {
                        if semantic_record_has_human_override(record) {
                            human_override_members.insert(record.id);
                        }
                    }
                    visible_members
                        .entry(other_id)
                        .or_insert_with(|| conflict_member_from_record(&other_record, false));
                    loaded_records.insert(other_id, other_record);
                    queue.push_back(other_id);
                }
            }
        }

        if let MemoryRecord::Semantic(semantic) = &current_record {
            if matches!(resolution, RecordConflictResolution::Exact) && current_id != seed_id {
                continue;
            }

            for contradiction_id in &semantic.contradiction_ids {
                let Some(target_record) = resolve_conflict_target_record(
                    db,
                    *contradiction_id,
                    visible_namespaces,
                    resolution_snapshot,
                    matches!(resolution, RecordConflictResolution::Exact),
                )
                .await
                else {
                    continue;
                };
                let target_id = target_record.id();
                if target_id == current_id {
                    continue;
                }

                let pair = normalize_conflict_pair(current_id, target_id);
                if seen_pairs.insert(pair) {
                    push_conflict_pair_edge(
                        pair,
                        FALLBACK_CONTRADICTION_CONFIDENCE,
                        1,
                        false,
                        &mut adjacency,
                        &mut pair_edges,
                    );
                }

                if seen_nodes.insert(target_id) {
                    if let MemoryRecord::Semantic(record) = &target_record {
                        if semantic_record_has_human_override(record) {
                            human_override_members.insert(record.id);
                        }
                    }
                    visible_members
                        .entry(target_id)
                        .or_insert_with(|| conflict_member_from_record(&target_record, false));
                    loaded_records.insert(target_id, target_record);
                    queue.push_back(target_id);
                }
            }
        }
    }

    if pair_edges.is_empty() {
        return ConflictSummary::default();
    }

    let mut visible_pairs = Vec::new();
    for edge in &pair_edges {
        if let (Some(member_a), Some(member_b)) =
            (visible_members.get(&edge.a), visible_members.get(&edge.b))
        {
            visible_pairs.push(ConflictPair {
                memory_a: edge.a,
                memory_b: edge.b,
                content_a: member_a.content.clone(),
                content_b: member_b.content.clone(),
                confidence: edge.confidence,
                source_reliability_a: member_a.source_reliability,
                source_reliability_b: member_b.source_reliability,
            });
        }
    }
    visible_pairs.sort_by_key(|pair| (pair.memory_a, pair.memory_b));
    let policy = resolve_conflict_resolution_policy(db, visible_namespaces);

    ConflictSummary {
        pairs: visible_pairs,
        groups: build_conflict_groups(
            &visible_members,
            &adjacency,
            &pair_edges,
            &human_override_members,
            policy,
        ),
    }
}

fn push_conflict_pair_edge(
    pair: (MemoryId, MemoryId),
    confidence: f32,
    evidence_count: usize,
    resolved: bool,
    adjacency: &mut BTreeMap<MemoryId, Vec<MemoryId>>,
    pair_edges: &mut Vec<ConflictEdgeMeta>,
) {
    adjacency.entry(pair.0).or_default().push(pair.1);
    adjacency.entry(pair.1).or_default().push(pair.0);
    pair_edges.push(ConflictEdgeMeta {
        a: pair.0,
        b: pair.1,
        confidence,
        evidence_count,
        resolved,
    });
}

pub(crate) fn build_semantic_conflict_groups(
    records: &[hirn_core::semantic::SemanticRecord],
    policy: ConflictResolutionPolicy,
) -> Vec<ConflictGroup> {
    let mut visible_members = BTreeMap::new();
    let mut adjacency = BTreeMap::new();
    let mut pair_edges = Vec::new();
    let mut human_override_members = HashSet::new();
    let mut seen_pairs = HashSet::new();
    let records_by_id: BTreeMap<MemoryId, &hirn_core::semantic::SemanticRecord> =
        records.iter().map(|record| (record.id, record)).collect();

    for record in records {
        let memory_record = MemoryRecord::Semantic(record.clone());
        visible_members.insert(record.id, conflict_member_from_record(&memory_record, true));
        if semantic_record_has_human_override(record) {
            human_override_members.insert(record.id);
        }
    }

    for record in records {
        for contradiction_id in &record.contradiction_ids {
            let Some(other) = records_by_id.get(contradiction_id) else {
                continue;
            };

            let pair = normalize_conflict_pair(record.id, *contradiction_id);
            if !seen_pairs.insert(pair) {
                continue;
            }

            let resolved = !record.is_live() || !other.is_live();
            push_conflict_pair_edge(
                pair,
                FALLBACK_CONTRADICTION_CONFIDENCE,
                1,
                resolved,
                &mut adjacency,
                &mut pair_edges,
            );
        }
    }

    build_conflict_groups(
        &visible_members,
        &adjacency,
        &pair_edges,
        &human_override_members,
        policy,
    )
}

async fn resolve_conflict_target_record<R>(
    db: &R,
    target_id: MemoryId,
    visible_namespaces: Option<&[Namespace]>,
    snapshot: Option<RecallSnapshot>,
    preserve_exact_semantic_targets: bool,
) -> Option<MemoryRecord>
where
    R: ConflictReadRuntime + ?Sized,
{
    let record = db.get_memory(target_id).await.ok()?;
    let resolved = match record {
        MemoryRecord::Semantic(record) if preserve_exact_semantic_targets => {
            MemoryRecord::Semantic(record)
        }
        MemoryRecord::Semantic(record) => match snapshot {
            Some(snapshot) => match db
                .semantic_revision_for_logical_id_at_snapshot(record.logical_memory_id, snapshot)
                .await
            {
                Ok(Some(revision)) => MemoryRecord::Semantic(revision),
                Ok(None) => return None,
                Err(_) => MemoryRecord::Semantic(record),
            },
            None => match db
                .semantic_head_for_logical_id(record.logical_memory_id)
                .await
            {
                Ok(head) => MemoryRecord::Semantic(head),
                Err(_) => MemoryRecord::Semantic(record),
            },
        },
        other => other,
    };

    if let Some(visible_namespaces) = visible_namespaces {
        if !visible_namespaces.contains(&resolved.effective_namespace()) {
            return None;
        }
    }

    Some(resolved)
}

async fn conflict_resolution_snapshot_for_record<R>(
    db: &R,
    record: &MemoryRecord,
) -> Option<RecallSnapshot>
where
    R: ConflictReadRuntime + ?Sized,
{
    let MemoryRecord::Semantic(record) = record else {
        return None;
    };

    match db
        .semantic_head_for_logical_id(record.logical_memory_id)
        .await
    {
        Ok(head) if head.revision_id != record.revision_id => {
            Some(RecallSnapshot::revision(record.revision_id))
        }
        _ => None,
    }
}

fn semantic_record_has_human_override(record: &hirn_core::semantic::SemanticRecord) -> bool {
    matches!(
        record.revision_operation,
        hirn_core::RevisionOperation::Override
    )
}

fn resolve_conflict_resolution_policy<R>(
    db: &R,
    visible_namespaces: Option<&[Namespace]>,
) -> ConflictResolutionPolicy
where
    R: ConflictReadRuntime + ?Sized,
{
    let config = db.config();
    if let Some(namespaces) = visible_namespaces {
        if namespaces.len() == 1 {
            if let Some(policy) = config
                .conflict_resolution_overrides
                .by_namespace
                .get(namespaces[0].as_str())
            {
                return *policy;
            }
        }
    }

    config
        .conflict_resolution_overrides
        .by_realm
        .get(&config.default_realm)
        .copied()
        .unwrap_or(config.conflict_resolution_policy)
}

fn normalize_conflict_pair(a: MemoryId, b: MemoryId) -> (MemoryId, MemoryId) {
    if a < b { (a, b) } else { (b, a) }
}

async fn conflict_node_is_visible(
    graph: &dyn crate::graph_store::GraphStore,
    node_id: MemoryId,
    visible_namespaces: Option<&[Namespace]>,
    namespace_cache: &mut BTreeMap<MemoryId, Option<Namespace>>,
) -> bool {
    let Some(visible_namespaces) = visible_namespaces else {
        return true;
    };

    let namespace = match namespace_cache.get(&node_id) {
        Some(namespace) => *namespace,
        None => {
            let namespace = graph.node_namespace(node_id).await.ok().flatten();
            namespace_cache.insert(node_id, namespace);
            namespace
        }
    };

    namespace.is_some_and(|namespace| visible_namespaces.contains(&namespace))
}

fn conflict_member_from_scored(scored: &ScoredMemory, in_result_set: bool) -> ConflictMember {
    let (logical_memory_id, revision_id, status) = conflict_member_identity(scored);
    ConflictMember {
        memory_id: scored.record.id(),
        logical_memory_id,
        revision_id,
        status,
        layer: scored.record.layer(),
        content: extract_content_str(&scored.record).to_string(),
        in_result_set,
        source_reliability: crate::scoring::source_reliability_for_record(&scored.record),
        recency_basis_ms: conflict_member_recency_basis_ms(&scored.record),
    }
}

fn conflict_member_from_record(record: &MemoryRecord, in_result_set: bool) -> ConflictMember {
    let (logical_memory_id, revision_id, status) = conflict_member_identity_from_record(record);
    ConflictMember {
        memory_id: record.id(),
        logical_memory_id,
        revision_id,
        status,
        layer: record.layer(),
        content: extract_content_str(record).to_string(),
        in_result_set,
        source_reliability: crate::scoring::source_reliability_for_record(record),
        recency_basis_ms: conflict_member_recency_basis_ms(record),
    }
}

fn conflict_member_recency_basis_ms(record: &MemoryRecord) -> i64 {
    match record {
        MemoryRecord::Episodic(record) => record.timestamp.timestamp_ms(),
        MemoryRecord::Semantic(record) => record.valid_from.timestamp_ms(),
        MemoryRecord::Working(record) => record.observed_at.timestamp_ms(),
        MemoryRecord::Procedural(record) => record.observed_at.timestamp_ms(),
    }
}

fn conflict_member_identity(
    scored: &ScoredMemory,
) -> (
    Option<LogicalMemoryId>,
    Option<RevisionId>,
    ConflictMemberStatus,
) {
    if let Some(revision) = scored.revision {
        return (
            Some(revision.logical_memory_id),
            Some(revision.revision_id),
            conflict_member_status_from_revision_state(revision.state),
        );
    }

    conflict_member_identity_from_record(&scored.record)
}

fn conflict_member_identity_from_record(
    record: &MemoryRecord,
) -> (
    Option<LogicalMemoryId>,
    Option<RevisionId>,
    ConflictMemberStatus,
) {
    match record {
        MemoryRecord::Semantic(record) => {
            let status = if record.is_retracted() {
                ConflictMemberStatus::Retracted
            } else if record.is_merged() {
                ConflictMemberStatus::Merged
            } else if record.superseded_by.is_some() {
                ConflictMemberStatus::Superseded
            } else {
                ConflictMemberStatus::Active
            };
            (
                Some(record.logical_memory_id),
                Some(record.revision_id),
                status,
            )
        }
        MemoryRecord::Episodic(record) => (
            None,
            None,
            if record.archived {
                ConflictMemberStatus::Superseded
            } else {
                ConflictMemberStatus::Active
            },
        ),
        MemoryRecord::Procedural(record) => (
            None,
            None,
            if record.archived {
                ConflictMemberStatus::Superseded
            } else {
                ConflictMemberStatus::Active
            },
        ),
        MemoryRecord::Working(_) => (None, None, ConflictMemberStatus::Active),
    }
}

fn conflict_member_status_from_revision_state(state: RevisionState) -> ConflictMemberStatus {
    match state {
        RevisionState::Active => ConflictMemberStatus::Active,
        RevisionState::Superseded => ConflictMemberStatus::Superseded,
        RevisionState::Retracted => ConflictMemberStatus::Retracted,
        RevisionState::Quarantined => ConflictMemberStatus::Quarantined,
        RevisionState::Merged => ConflictMemberStatus::Merged,
    }
}

fn build_conflict_groups(
    visible_members: &BTreeMap<MemoryId, ConflictMember>,
    adjacency: &BTreeMap<MemoryId, Vec<MemoryId>>,
    pair_edges: &[ConflictEdgeMeta],
    human_override_members: &HashSet<MemoryId>,
    policy: ConflictResolutionPolicy,
) -> Vec<ConflictGroup> {
    let mut visited: HashSet<MemoryId> = HashSet::new();
    let mut groups = Vec::new();

    for start in visible_members.keys().copied() {
        if visited.contains(&start) || !adjacency.contains_key(&start) {
            continue;
        }

        let mut queue = VecDeque::from([start]);
        let mut component = HashSet::new();
        component.insert(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&current) {
                for &neighbor in neighbors {
                    if component.insert(neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        let mut members: Vec<ConflictMember> = component
            .iter()
            .filter_map(|id| visible_members.get(id).cloned())
            .collect();
        if members.is_empty() {
            continue;
        }
        members.sort_by_key(|member| member.memory_id);

        let omitted_member_count = component.len().saturating_sub(members.len());
        let component_edges: Vec<&ConflictEdgeMeta> = pair_edges
            .iter()
            .filter(|edge| component.contains(&edge.a) && component.contains(&edge.b))
            .collect();
        let pair_count = component_edges.len();
        let confidence = if pair_count > 0 {
            component_edges
                .iter()
                .map(|edge| edge.confidence)
                .sum::<f32>()
                / pair_count as f32
        } else {
            0.0
        };
        let evidence_count = component_edges.iter().map(|edge| edge.evidence_count).sum();
        let source_reliability = members
            .iter()
            .map(|member| member.source_reliability)
            .sum::<f32>()
            / members.len() as f32;
        let arbitration_status =
            derive_conflict_arbitration_status(&members, &component_edges, omitted_member_count);
        let authoritative_memory_id =
            authoritative_conflict_memory_id(&members, omitted_member_count);
        let preferred_memory_id = if authoritative_memory_id.is_none() {
            select_conflict_preferred_memory_id(
                &members,
                &component_edges,
                omitted_member_count,
                human_override_members,
                policy,
            )
        } else {
            None
        };

        groups.push(ConflictGroup {
            conflict_id: members
                .iter()
                .map(|member| member.memory_id.to_string())
                .collect::<Vec<_>>()
                .join(":"),
            members,
            omitted_member_count,
            pair_count,
            confidence,
            evidence_count,
            source_reliability,
            arbitration_status,
            authoritative_memory_id,
            preferred_memory_id,
        });
    }

    groups.sort_by(|a, b| a.conflict_id.cmp(&b.conflict_id));
    groups
}

fn derive_conflict_arbitration_status(
    members: &[ConflictMember],
    component_edges: &[&ConflictEdgeMeta],
    omitted_member_count: usize,
) -> ConflictArbitrationStatus {
    let active_member_count = members
        .iter()
        .filter(|member| member.status == ConflictMemberStatus::Active)
        .count();
    let has_resolved_loser = members.iter().any(|member| {
        matches!(
            member.status,
            ConflictMemberStatus::Retracted | ConflictMemberStatus::Merged
        )
    });

    if !component_edges.is_empty() && component_edges.iter().all(|edge| edge.resolved) {
        ConflictArbitrationStatus::Resolved
    } else if members
        .iter()
        .any(|member| member.status == ConflictMemberStatus::Quarantined)
    {
        ConflictArbitrationStatus::Quarantined
    } else if omitted_member_count == 0
        && active_member_count == 1
        && members
            .iter()
            .any(|member| member.status != ConflictMemberStatus::Active)
    {
        if has_resolved_loser {
            ConflictArbitrationStatus::Resolved
        } else {
            ConflictArbitrationStatus::Superseded
        }
    } else if members
        .iter()
        .all(|member| member.status != ConflictMemberStatus::Active)
    {
        ConflictArbitrationStatus::Resolved
    } else {
        ConflictArbitrationStatus::Unresolved
    }
}

fn authoritative_conflict_memory_id(
    members: &[ConflictMember],
    omitted_member_count: usize,
) -> Option<MemoryId> {
    if omitted_member_count > 0 {
        return None;
    }

    let active_members: Vec<MemoryId> = members
        .iter()
        .filter(|member| member.status == ConflictMemberStatus::Active)
        .map(|member| member.memory_id)
        .collect();

    (active_members.len() == 1).then_some(active_members[0])
}

fn select_conflict_preferred_memory_id(
    members: &[ConflictMember],
    component_edges: &[&ConflictEdgeMeta],
    omitted_member_count: usize,
    human_override_members: &HashSet<MemoryId>,
    policy: ConflictResolutionPolicy,
) -> Option<MemoryId> {
    if omitted_member_count > 0 {
        return None;
    }

    let mut active_members: Vec<&ConflictMember> = members
        .iter()
        .filter(|member| member.status == ConflictMemberStatus::Active)
        .collect();
    if active_members.is_empty() {
        return None;
    }

    if policy.prefer_human_override
        && active_members
            .iter()
            .any(|member| human_override_members.contains(&member.memory_id))
    {
        active_members.retain(|member| human_override_members.contains(&member.memory_id));
    }

    let supports: BTreeMap<MemoryId, (usize, f32)> = active_members
        .iter()
        .map(|member| {
            (
                member.memory_id,
                conflict_member_support(member.memory_id, component_edges),
            )
        })
        .collect();
    let max_evidence = supports
        .values()
        .map(|(evidence, _)| *evidence)
        .max()
        .unwrap_or(0);
    let max_confidence = supports
        .values()
        .map(|(_, confidence)| *confidence)
        .fold(0.0, f32::max);

    let mut recency_order = active_members.clone();
    recency_order.sort_by_key(|member| {
        (
            member.recency_basis_ms,
            member.revision_id,
            member.memory_id,
        )
    });
    let recency_rank: BTreeMap<MemoryId, usize> = recency_order
        .iter()
        .enumerate()
        .map(|(index, member)| (member.memory_id, index))
        .collect();

    active_members
        .into_iter()
        .max_by(|left, right| {
            compare_conflict_member_preference(
                left,
                right,
                &supports,
                &recency_rank,
                recency_order.len(),
                max_evidence,
                max_confidence,
                human_override_members,
                policy,
            )
        })
        .map(|member| member.memory_id)
}

fn compare_conflict_member_preference(
    left: &ConflictMember,
    right: &ConflictMember,
    supports: &BTreeMap<MemoryId, (usize, f32)>,
    recency_rank: &BTreeMap<MemoryId, usize>,
    active_member_count: usize,
    max_evidence: usize,
    max_confidence: f32,
    human_override_members: &HashSet<MemoryId>,
    policy: ConflictResolutionPolicy,
) -> std::cmp::Ordering {
    let left_score = conflict_member_preference_score(
        left,
        supports,
        recency_rank,
        active_member_count,
        max_evidence,
        max_confidence,
        human_override_members,
        policy,
    );
    let right_score = conflict_member_preference_score(
        right,
        supports,
        recency_rank,
        active_member_count,
        max_evidence,
        max_confidence,
        human_override_members,
        policy,
    );

    left_score
        .total_cmp(&right_score)
        .then_with(|| left.revision_id.cmp(&right.revision_id))
        .then_with(|| left.memory_id.cmp(&right.memory_id))
}

fn conflict_member_preference_score(
    member: &ConflictMember,
    supports: &BTreeMap<MemoryId, (usize, f32)>,
    recency_rank: &BTreeMap<MemoryId, usize>,
    active_member_count: usize,
    max_evidence: usize,
    max_confidence: f32,
    human_override_members: &HashSet<MemoryId>,
    policy: ConflictResolutionPolicy,
) -> f32 {
    let (evidence_count, confidence_sum) =
        supports.get(&member.memory_id).copied().unwrap_or((0, 0.0));
    let recency_score = if active_member_count <= 1 {
        1.0
    } else {
        recency_rank.get(&member.memory_id).copied().unwrap_or(0) as f32
            / (active_member_count - 1) as f32
    };
    let evidence_score = if max_evidence == 0 {
        0.0
    } else {
        evidence_count as f32 / max_evidence as f32
    };
    let confidence_score = if max_confidence <= 0.0 {
        0.0
    } else {
        confidence_sum / max_confidence
    };
    let support_score = if evidence_count == 0 && confidence_sum <= 0.0 {
        0.0
    } else {
        f32::midpoint(evidence_score, confidence_score)
    };
    let human_override_score = if human_override_members.contains(&member.memory_id) {
        1.0
    } else {
        0.0
    };

    recency_score * policy.recency_weight
        + member.source_reliability.clamp(0.0, 1.0) * policy.source_reliability_weight
        + support_score * policy.supporting_evidence_weight
        + human_override_score * policy.human_override_weight
}

fn conflict_member_support(
    memory_id: MemoryId,
    component_edges: &[&ConflictEdgeMeta],
) -> (usize, f32) {
    component_edges
        .iter()
        .filter(|edge| edge.a == memory_id || edge.b == memory_id)
        .fold((0, 0.0), |(evidence_count, confidence), edge| {
            (
                evidence_count + edge.evidence_count,
                confidence + edge.confidence,
            )
        })
}

fn format_conflict_group_line(group: &ConflictGroup) -> String {
    let mut line = format!(
        "⚠ CONFLICT {:?} (conf={:.2}, evidence={}): ",
        group.arbitration_status, group.confidence, group.evidence_count
    );

    if let Some(authoritative_memory_id) = group.authoritative_memory_id {
        let _ = write!(line, "active=[{authoritative_memory_id}] ");
    } else if let Some(preferred_memory_id) = group.preferred_memory_id {
        let _ = write!(line, "preferred_visible=[{preferred_memory_id}] ");
    }

    let member_summary = group
        .members
        .iter()
        .map(|member| {
            format!(
                "[{} {:?}] {}",
                member.memory_id, member.status, member.content
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    line.push_str(&member_summary);

    if group.omitted_member_count > 0 {
        let _ = write!(
            line,
            " | {} contradictory claim(s) omitted from this result set",
            group.omitted_member_count
        );
    }

    line
}

// ── Step 4: Budget allocation (F-43: tiered) ──────────────────────────

/// F-43: Tiered budget allocation per CONCEPT §10.2.
///
/// Budget tiers (after mandatory WM + contradiction reserves):
/// 1. **Direct results** (50%): semantic + episodic + procedural
/// 2. **Graph-connected neighbors** (25%): memories linked by graph edges
/// 3. **Causally-upstream** (15%): memories reachable via CAUSED_BY / LED_TO
/// 4. **Filler / expansion** (10%): redistributed to direct if unused
fn allocate_budget(
    config: &ContextConfig,
    working_entries: &[WorkingMemoryEntry],
    contradictions: &[ConflictGroup],
    classified: &[Candidate],
    tokenizer: &dyn Tokenizer,
) -> BudgetAllocation {
    let total = config
        .token_budget
        .saturating_sub(estimate_context_format_overhead(
            config,
            working_entries,
            contradictions,
            classified,
            tokenizer,
        ));

    // Working memory: mandatory reserve.
    let wm_needed: usize = working_entries
        .iter()
        .map(|w| tokenizer.count_tokens(&w.content) + 5)
        .sum();
    let wm_reserve = (total as f32 * config.working_memory_reserve) as usize;
    let wm_budget = if wm_needed == 0 { 0 } else { wm_reserve.max(1) };

    // Contradiction overhead.
    let contra_budget = if contradictions.is_empty() {
        0
    } else {
        let actual: usize = contradictions
            .iter()
            .map(|group| tokenizer.count_tokens(&format_conflict_group_line(group)) + 2)
            .sum();
        actual.min(total / 4)
    };

    // Remaining budget after WM + contradictions.
    let remaining = total.saturating_sub(wm_budget + contra_budget);

    // F-43: Tiered allocation of the remaining budget.
    // Tier 1: Direct results (50%) — split among semantic/episodic/procedural
    // Tier 2: Graph-connected (25%) — only if graph context is enabled.
    // Tier 3: Causal-upstream (15%) — only if causal chains are enabled.
    // Tier 4: Filler (10%) — redistributed to Tier 1
    let graph_fraction = if config.features.include_graph_context() {
        0.25
    } else {
        0.0
    };
    let causal_fraction = if config.features.include_causal_chains() {
        0.15
    } else {
        0.0
    };
    let filler_fraction = 0.10_f32;
    let direct_fraction = 1.0 - graph_fraction - causal_fraction - filler_fraction;

    let direct_budget = (remaining as f32 * (direct_fraction + filler_fraction)) as usize;
    let graph_budget = (remaining as f32 * graph_fraction) as usize;
    let causal_budget = (remaining as f32 * causal_fraction) as usize;

    // Split direct budget among semantic, episodic, and procedural.
    let has_semantic = classified.iter().any(|c| c.layer == Layer::Semantic);
    let has_episodic = classified.iter().any(|c| c.layer == Layer::Episodic);
    let has_procedural = classified.iter().any(|c| c.layer == Layer::Procedural);

    let active_layers = has_semantic as usize + has_episodic as usize + has_procedural as usize;
    let (sem_budget, ep_budget, proc_budget) = if active_layers == 0 {
        (0, 0, 0)
    } else {
        // Weighted split: semantic gets `semantic_weight`, episodic and procedural share the rest.
        let sw = config.semantic_weight;
        let sem = if has_semantic {
            (direct_budget as f32 * sw) as usize
        } else {
            0
        };
        let rest = direct_budget.saturating_sub(sem);
        let ep = if has_episodic && has_procedural {
            // 70/30 split between episodic and procedural
            (rest as f32 * 0.7) as usize
        } else if has_episodic {
            rest
        } else {
            0
        };
        let proc = rest.saturating_sub(ep);
        (sem, ep, proc)
    };

    BudgetAllocation {
        working_memory: wm_budget,
        contradictions: contra_budget,
        semantic: sem_budget,
        episodic: ep_budget,
        procedural: proc_budget,
        graph_connected: graph_budget,
        causal_upstream: causal_budget,
    }
}

fn estimate_context_format_overhead(
    config: &ContextConfig,
    working_entries: &[WorkingMemoryEntry],
    contradictions: &[ConflictGroup],
    classified: &[Candidate],
    tokenizer: &dyn Tokenizer,
) -> usize {
    let include_working = !working_entries.is_empty();
    let include_conflicts = !contradictions.is_empty();
    let include_semantic = classified.iter().any(|c| c.layer == Layer::Semantic);
    let include_episodic = classified.iter().any(|c| c.layer == Layer::Episodic);
    let include_procedural = classified.iter().any(|c| c.layer == Layer::Procedural);
    let include_graph = config.features.include_graph_context() && !classified.is_empty();
    let include_causal = config.features.include_causal_chains() && !classified.is_empty();

    let placeholder_entry = ContextEntry {
        id: MemoryId::new(),
        content: "x".to_string(),
        token_cost: 0,
        resource_evidence: Vec::new(),
        resource_preview_packages: Vec::new(),
        resource_score_attribution: Vec::new(),
    };
    let placeholder_line = String::from("x");

    let working = if include_working {
        vec![placeholder_entry.clone()]
    } else {
        vec![]
    };
    let conflicts = if include_conflicts {
        vec![placeholder_line.clone()]
    } else {
        vec![]
    };
    let semantic = if include_semantic {
        vec![placeholder_entry.clone()]
    } else {
        vec![]
    };
    let episodic = if include_episodic {
        vec![placeholder_entry.clone()]
    } else {
        vec![]
    };
    let procedural = if include_procedural {
        vec![placeholder_entry]
    } else {
        vec![]
    };
    let graph = if include_graph {
        vec![placeholder_line.clone()]
    } else {
        vec![]
    };
    let causal = if include_causal {
        vec![placeholder_line]
    } else {
        vec![]
    };

    let placeholder_tokens = [
        include_working,
        include_conflicts,
        include_semantic,
        include_episodic,
        include_procedural,
        include_graph,
        include_causal,
    ]
    .into_iter()
    .filter(|included| *included)
    .count()
        * tokenizer.count_tokens("x");

    let formatted = format_context(
        config.output_format,
        &working,
        &conflicts,
        &semantic,
        &episodic,
        &procedural,
        &graph,
        &causal,
    );

    tokenizer
        .count_tokens(&formatted)
        .saturating_sub(placeholder_tokens)
}

// ── Step 5: Build sections ─────────────────────────────────────────────

fn build_working_memory_section(
    entries: &[WorkingMemoryEntry],
    budget_tokens: usize,
    tokenizer: &dyn Tokenizer,
) -> (Vec<ContextEntry>, usize) {
    let mut lines = Vec::new();
    let mut used = 0;

    for entry in entries {
        let line = format!("• {}", entry.content);
        let tokens = tokenizer.count_tokens(&line);
        if used + tokens > budget_tokens {
            // Truncate last entry if partially fits.
            let remaining = budget_tokens.saturating_sub(used);
            if remaining > 5 {
                let truncated = truncate_to_budget(&line, remaining, tokenizer);
                used += tokenizer.count_tokens(&truncated);
                let content = truncated
                    .strip_prefix("• ")
                    .unwrap_or(truncated.as_str())
                    .to_string();
                let content_tokens = tokenizer.count_tokens(&content);
                lines.push(ContextEntry {
                    id: entry.id,
                    content,
                    token_cost: content_tokens,
                    resource_evidence: Vec::new(),
                    resource_preview_packages: Vec::new(),
                    resource_score_attribution: Vec::new(),
                });
            }
            break;
        }
        used += tokens;
        let content_tokens = tokenizer.count_tokens(&entry.content);
        lines.push(ContextEntry {
            id: entry.id,
            content: entry.content.clone(),
            token_cost: content_tokens,
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
        });
    }

    (lines, used)
}

fn build_contradiction_section(
    conflicts: &[ConflictGroup],
    budget_tokens: usize,
    tokenizer: &dyn Tokenizer,
) -> (Vec<String>, usize) {
    let mut lines = Vec::new();
    let mut used = 0;

    for conflict in conflicts {
        let line = format_conflict_group_line(conflict);
        let tokens = tokenizer.count_tokens(&line);
        if used + tokens > budget_tokens {
            break;
        }
        used += tokens;
        lines.push(line);
    }

    (lines, used)
}

fn build_direct_sections(
    sorted: &SortedDirectCandidates<'_>,
    allocation: &BudgetAllocation,
    config: &ContextConfig,
    tokenizer: &dyn Tokenizer,
) -> (Vec<ContextEntry>, Vec<ContextEntry>, Vec<ContextEntry>) {
    let (semantic_section, _sem_tokens) = build_layer_section_from_sorted(
        &sorted.semantic,
        allocation.semantic,
        config.compression_threshold,
        None,
        tokenizer,
    );

    let (episodic_section, _ep_tokens) = build_layer_section_from_sorted(
        &sorted.episodic,
        allocation.episodic,
        config.compression_threshold,
        Some(config.max_episodic_entries),
        tokenizer,
    );

    let (procedural_section, _proc_tokens) = build_layer_section_from_sorted(
        &sorted.procedural,
        allocation.procedural,
        config.compression_threshold,
        None,
        tokenizer,
    );

    (semantic_section, episodic_section, procedural_section)
}

fn collect_direct_section_ids(
    semantic: &[ContextEntry],
    episodic: &[ContextEntry],
    procedural: &[ContextEntry],
) -> HashSet<MemoryId> {
    semantic
        .iter()
        .chain(episodic.iter())
        .chain(procedural.iter())
        .map(|entry| entry.id)
        .collect()
}

/// Build a section for a specific layer, applying progressive compression.
/// Returns (entries as (id, text) pairs, total tokens used).
const MAX_CONTEXT_EVIDENCE_ITEMS: usize = 2;

struct SortedDirectCandidates<'a> {
    semantic: Vec<&'a Candidate>,
    episodic: Vec<&'a Candidate>,
    procedural: Vec<&'a Candidate>,
}

fn sort_candidates_by_weighted_score(candidates: &mut Vec<&Candidate>) {
    candidates.sort_by(|a, b| {
        let weighted_a = a.score * a.trust_score;
        let weighted_b = b.score * b.trust_score;
        weighted_b
            .partial_cmp(&weighted_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn prepare_sorted_direct_candidates(classified: &[Candidate]) -> SortedDirectCandidates<'_> {
    let mut semantic = classified
        .iter()
        .filter(|candidate| candidate.layer == Layer::Semantic)
        .collect::<Vec<_>>();
    sort_candidates_by_weighted_score(&mut semantic);

    let mut episodic = classified
        .iter()
        .filter(|candidate| candidate.layer == Layer::Episodic)
        .collect::<Vec<_>>();
    sort_candidates_by_weighted_score(&mut episodic);

    let mut procedural = classified
        .iter()
        .filter(|candidate| candidate.layer == Layer::Procedural)
        .collect::<Vec<_>>();
    sort_candidates_by_weighted_score(&mut procedural);

    SortedDirectCandidates {
        semantic,
        episodic,
        procedural,
    }
}

fn access_mode_label(summary: &ResourceEvidenceSummary) -> &'static str {
    if summary.can_hydrate_full {
        "full"
    } else if summary.can_hydrate_preview {
        "preview"
    } else {
        "metadata"
    }
}

fn evidence_identity(summary: &ResourceEvidenceSummary) -> String {
    summary
        .display_name
        .clone()
        .or_else(|| summary.mime_type.clone())
        .unwrap_or_else(|| summary.resource_id.to_string())
}

fn summarize_resource_evidence(summary: &ResourceEvidenceSummary) -> String {
    let mut qualifiers = Vec::new();
    qualifiers.push(summary.provenance.as_str().to_string());
    if let Some(modality) = summary.modality {
        qualifiers.push(modality.as_str().to_string());
    }
    if let Some(artifact_kind) = summary.artifact_kind {
        qualifiers.push(format!("artifact={}", artifact_kind.as_str()));
    }
    if summary.lifecycle_state != ResourceGovernanceState::Active {
        qualifiers.push(summary.lifecycle_state.as_str().to_string());
    }
    qualifiers.push(access_mode_label(summary).to_string());
    if summary.has_preview {
        qualifiers.push("preview".to_string());
    }
    if !summary.available_artifacts.is_empty() {
        let artifacts = summary
            .available_artifacts
            .iter()
            .map(|kind| kind.as_str())
            .collect::<Vec<_>>()
            .join("|");
        qualifiers.push(format!("artifacts={artifacts}"));
    }

    format!(
        "{} {} [{}]",
        summary.role.as_str(),
        evidence_identity(summary),
        qualifiers.join(", ")
    )
}

fn evidence_suffix(resource_evidence: &[ResourceEvidenceSummary]) -> String {
    if resource_evidence.is_empty() {
        return String::new();
    }

    let mut parts = resource_evidence
        .iter()
        .take(MAX_CONTEXT_EVIDENCE_ITEMS)
        .map(summarize_resource_evidence)
        .collect::<Vec<_>>();
    if resource_evidence.len() > MAX_CONTEXT_EVIDENCE_ITEMS {
        parts.push(format!(
            "+{} more",
            resource_evidence.len() - MAX_CONTEXT_EVIDENCE_ITEMS
        ));
    }

    format!(" Evidence: {}.", parts.join("; "))
}

async fn hydrate_selected_resource_previews(
    db: &HirnDB,
    actor_id: &AgentId,
    sections: &mut ContextSections,
    config: &ContextConfig,
) -> HirnResult<()> {
    if !should_package_resource_previews(config) {
        return Ok(());
    }

    let mut preview_cache = PreviewPackageCache::default();
    hydrate_resource_preview_packages_for_entries(
        db,
        actor_id,
        &mut sections.semantic,
        config,
        &mut preview_cache,
    )
    .await?;
    hydrate_resource_preview_packages_for_entries(
        db,
        actor_id,
        &mut sections.episodic,
        config,
        &mut preview_cache,
    )
    .await?;
    hydrate_resource_preview_packages_for_entries(
        db,
        actor_id,
        &mut sections.procedural,
        config,
        &mut preview_cache,
    )
    .await?;

    Ok(())
}

fn should_package_resource_previews(config: &ContextConfig) -> bool {
    config.output_format == ContextFormat::Json
        && config.features.package_resource_previews()
        && config.max_resource_previews_per_entry > 0
        && config.max_resource_preview_chars > 0
}

async fn hydrate_resource_preview_packages_for_entries(
    db: &HirnDB,
    actor_id: &AgentId,
    entries: &mut [ContextEntry],
    config: &ContextConfig,
    preview_cache: &mut PreviewPackageCache,
) -> HirnResult<()> {
    for entry in entries {
        entry.resource_preview_packages = package_resource_preview_packages_for_evidence(
            db,
            actor_id,
            &entry.resource_evidence,
            &entry.resource_preview_packages,
            config.max_resource_previews_per_entry,
            config.max_resource_preview_chars,
            preview_cache,
            PreviewPackageSurface::Think,
        )
        .await;
    }

    Ok(())
}

fn compose_candidate_text(candidate: &Candidate, compression: CompressionLevel) -> String {
    let raw_text = match compression {
        CompressionLevel::Full => candidate.full_content.clone(),
        CompressionLevel::Summary => {
            if candidate.summary.is_empty() {
                candidate.full_content.clone()
            } else {
                candidate.summary.clone()
            }
        }
        CompressionLevel::EntityOnly => {
            if candidate.entities.is_empty() {
                format!("(record {}, score: {:.2})", candidate.id, candidate.score)
            } else {
                format!(
                    "Re: {} (score: {:.2})",
                    candidate.entities.join(", "),
                    candidate.score
                )
            }
        }
    };

    let mut text = if candidate.trust_score < 0.5 {
        format!("[low trust: {:.2}] {}", candidate.trust_score, raw_text)
    } else {
        raw_text
    };
    text.push_str(&evidence_suffix(&candidate.resource_evidence));
    text
}

#[cfg(test)]
fn build_layer_section(
    classified: &[Candidate],
    layer: Layer,
    budget_tokens: usize,
    compression_threshold: f32,
    max_entries: Option<usize>,
    tokenizer: &dyn Tokenizer,
) -> (Vec<ContextEntry>, usize) {
    let mut layer_candidates: Vec<&Candidate> =
        classified.iter().filter(|c| c.layer == layer).collect();

    sort_candidates_by_weighted_score(&mut layer_candidates);

    build_layer_section_from_sorted(
        &layer_candidates,
        budget_tokens,
        compression_threshold,
        max_entries,
        tokenizer,
    )
}

fn build_layer_section_from_sorted(
    layer_candidates: &[&Candidate],
    budget_tokens: usize,
    compression_threshold: f32,
    max_entries: Option<usize>,
    tokenizer: &dyn Tokenizer,
) -> (Vec<ContextEntry>, usize) {
    let mut entries = Vec::new();
    let mut used = 0;

    for candidate in layer_candidates {
        if max_entries.is_some_and(|limit| entries.len() >= limit) {
            break;
        }

        let preferred = determine_compression(candidate, compression_threshold);
        let selected = match preferred {
            CompressionLevel::Full => select_candidate_render(
                candidate,
                &[
                    CompressionLevel::Full,
                    CompressionLevel::Summary,
                    CompressionLevel::EntityOnly,
                ],
                used,
                budget_tokens,
                tokenizer,
            ),
            CompressionLevel::Summary => select_candidate_render(
                candidate,
                &[CompressionLevel::Summary, CompressionLevel::EntityOnly],
                used,
                budget_tokens,
                tokenizer,
            ),
            CompressionLevel::EntityOnly => select_candidate_render(
                candidate,
                &[CompressionLevel::EntityOnly],
                used,
                budget_tokens,
                tokenizer,
            ),
        };

        let Some((content, tokens)) = selected else {
            break;
        };

        used += tokens;
        entries.push(ContextEntry {
            id: candidate.id,
            content,
            token_cost: tokens,
            resource_evidence: candidate.resource_evidence.clone(),
            resource_preview_packages: candidate.resource_preview_packages.clone(),
            resource_score_attribution: candidate.resource_score_attribution.clone(),
        });
    }

    (entries, used)
}

fn select_candidate_render(
    candidate: &Candidate,
    levels: &[CompressionLevel],
    used_tokens: usize,
    budget_tokens: usize,
    tokenizer: &dyn Tokenizer,
) -> Option<(String, usize)> {
    for &level in levels {
        // Use the pre-computed token cost when available (non-zero).  A zero
        // cost means finalize_candidate_render_tokens has not run yet (e.g.
        // in isolated tests), so we fall back to live counting in that branch.
        let precomputed = match level {
            CompressionLevel::Full => candidate.tokens_full,
            CompressionLevel::Summary => candidate.tokens_summary,
            CompressionLevel::EntityOnly => candidate.tokens_entity,
        };
        if precomputed > 0 {
            if used_tokens + precomputed <= budget_tokens {
                return Some((compose_candidate_text(candidate, level), precomputed));
            }
            // Pre-computed cost confirms this level doesn't fit; skip text alloc.
            continue;
        }
        // Fallback: compute text and measure on the fly.
        let text = compose_candidate_text(candidate, level);
        let tokens = tokenizer.count_tokens(&text);
        if used_tokens + tokens <= budget_tokens {
            return Some((text, tokens));
        }
    }

    None
}

/// Build graph-connected and causal-upstream sections in a **single** async
/// pass, sharing one batch hydration call for all neighbour IDs.
///
/// Collecting IDs from the hot graph is synchronous (zero I/O).  Hydrating
/// record content requires Lance reads; combining both sections into one
/// `hydrate_context_contents_batch` call reduces the number of Lance
/// round-trips from 2–4 sequential per-section chunks to a single batched
/// request, cutting the wall-clock contribution of this step roughly in half.
async fn build_graph_and_causal_sections(
    db: &HirnDB,
    candidates: &[ScoredMemory],
    content_pool: &[ScoredMemory],
    graph_budget: usize,
    causal_budget: usize,
    tokenizer: &dyn Tokenizer,
) -> (Vec<String>, Vec<String>) {
    let candidate_ids: HashSet<MemoryId> = candidates.iter().map(|c| c.record.id()).collect();

    // ── Collect all needed IDs from the hot graph (sync, no I/O) ────────
    let (neighbor_ids, causal_ids) = {
        let graph = db.cached_graph().hot_graph();

        let mut neighbor_ids: Vec<(MemoryId, String)> = Vec::new();
        if graph_budget > 0 {
            let mut seen: HashSet<MemoryId> = HashSet::new();
            for sm in candidates {
                for edge in graph.get_edges(sm.record.id()) {
                    let neighbor = if edge.source == sm.record.id() {
                        edge.target
                    } else {
                        edge.source
                    };
                    if !candidate_ids.contains(&neighbor) && seen.insert(neighbor) {
                        let rel_label = format!("{:?}", edge.relation);
                        neighbor_ids.push((neighbor, rel_label));
                    }
                }
            }
        }

        let mut causal_ids: Vec<MemoryId> = Vec::new();
        if causal_budget > 0 {
            let mut visited = candidate_ids.clone();
            let mut frontier: Vec<MemoryId> = candidates.iter().map(|c| c.record.id()).collect();
            for _depth in 0..3 {
                let mut next_frontier = Vec::new();
                for id in &frontier {
                    for edge in graph.get_edges(*id) {
                        if edge.relation != EdgeRelation::CausedBy
                            && edge.relation != EdgeRelation::Causes
                        {
                            continue;
                        }
                        let upstream = if edge.source == *id {
                            edge.target
                        } else {
                            edge.source
                        };
                        if visited.insert(upstream) {
                            causal_ids.push(upstream);
                            next_frontier.push(upstream);
                        }
                    }
                }
                frontier = next_frontier;
                if frontier.is_empty() {
                    break;
                }
            }
        }

        (neighbor_ids, causal_ids)
    };

    // ── Single batch hydration for ALL IDs across both sections ─────────
    let all_ids: Vec<MemoryId> = {
        let mut seen = HashSet::new();
        neighbor_ids
            .iter()
            .map(|(id, _)| *id)
            .chain(causal_ids.iter().copied())
            .filter(|id| seen.insert(*id))
            .collect()
    };

    if all_ids.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // ── Build preliminary cache from content_pool (zero extra I/O) ──────
    // Any neighbor that was already loaded in the initial DataFusion scan is
    // served from this borrow-only map, avoiding a second Lance round-trip.
    let pool_cache: HashMap<MemoryId, &str> = content_pool
        .iter()
        .map(|sm| (sm.record.id(), extract_content_str(&sm.record)))
        .collect();

    let hydrated = hydrate_context_contents_batch(db, all_ids, &pool_cache).await;

    // ── Build graph section from cached results ──────────────────────────
    let mut graph_lines: Vec<String> = Vec::new();
    let mut graph_used = 0usize;
    if graph_budget > 0 {
        for (nid, rel) in &neighbor_ids {
            let content = hydrated
                .get(nid)
                .cloned()
                .unwrap_or_else(|| format!("(record {})", nid));
            let line = format!("[via {}] {}", rel, content);
            let tokens = tokenizer.count_tokens(&line);
            if graph_used + tokens > graph_budget {
                break;
            }
            graph_used += tokens;
            graph_lines.push(line);
        }
    }

    // ── Build causal section from cached results ─────────────────────────
    let mut causal_lines: Vec<String> = Vec::new();
    let mut causal_used = 0usize;
    if causal_budget > 0 {
        for cid in &causal_ids {
            let content = hydrated
                .get(cid)
                .cloned()
                .unwrap_or_else(|| format!("(record {})", cid));
            let line = format!("[causal] {}", content);
            let tokens = tokenizer.count_tokens(&line);
            if causal_used + tokens > causal_budget {
                break;
            }
            causal_used += tokens;
            causal_lines.push(line);
        }
    }

    (graph_lines, causal_lines)
}

async fn hydrate_context_contents_batch(
    db: &HirnDB,
    ids: impl IntoIterator<Item = MemoryId>,
    preliminary_cache: &HashMap<MemoryId, &str>,
) -> HashMap<MemoryId, String> {
    let unique_ids: Vec<MemoryId> = ids
        .into_iter()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    if unique_ids.is_empty() {
        return HashMap::new();
    }

    // Serve any IDs already present in the preliminary cache (zero Lance I/O).
    let mut contents = HashMap::with_capacity(unique_ids.len());
    let mut cache_miss_ids: Vec<MemoryId> = Vec::new();
    for id in &unique_ids {
        if let Some(&content) = preliminary_cache.get(id) {
            contents.insert(*id, content.to_string());
        } else {
            cache_miss_ids.push(*id);
        }
    }

    if cache_miss_ids.is_empty() {
        return contents;
    }

    // Single cross-layer batch fetch. Eliminates the previous two-phase
    // (episodic-first then non-episodic) serial approach — one fewer Lance
    // round-trip on the hot path. `get_memories_batch` scans all relevant
    // datasets in a single DataFusion plan.
    let records = db
        .get_memories_batch(&cache_miss_ids)
        .await
        .unwrap_or_default();
    for (id, record) in records {
        contents.insert(id, extract_content_str(&record).to_string());
    }

    contents
}

fn determine_compression(candidate: &Candidate, threshold: f32) -> CompressionLevel {
    if candidate.score >= threshold {
        CompressionLevel::Full
    } else if candidate.score >= threshold * 0.5 {
        CompressionLevel::Summary
    } else {
        CompressionLevel::EntityOnly
    }
}

// ── Step 6: Score distribution ─────────────────────────────────────────

fn compute_score_distribution(
    candidates: &[ScoredMemory],
    included_ids: &[MemoryId],
) -> ScoreDistribution {
    let included_set: HashSet<MemoryId> = included_ids.iter().copied().collect();
    let scores: Vec<f32> = candidates
        .iter()
        .filter(|c| included_set.contains(&c.record.id()))
        .map(|c| c.score)
        .collect();

    if scores.is_empty() {
        return ScoreDistribution::default();
    }

    let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;

    ScoreDistribution { min, max, mean }
}

// ── Step 8: Format output ──────────────────────────────────────────────

/// Per-entry token overhead added by the formatter beyond the raw content
/// token cost stored in `ContextEntry::token_cost`.  Used only for the
/// greedy trim estimate; the exact binary search corrects any discrepancy.
fn per_entry_format_overhead(format: ContextFormat) -> usize {
    match format {
        // "• " prefix (1 token) + trailing newline token (1 token)
        ContextFormat::Structured => 2,
        // Separators like ". Also, " or ". Then, " average ~2 tokens
        ContextFormat::Narrative => 2,
        // JSON object wrapper: `{"id":"…","content":"…"}` adds ~5 tokens
        ContextFormat::Json => 5,
    }
}

fn fit_context_to_budget(
    format: ContextFormat,
    sections: &mut ContextSections,
    token_budget: usize,
    tokenizer: &dyn Tokenizer,
) -> String {
    let context = format_context_for_lengths(format, sections, sections.section_lengths());
    let tokens = tokenizer.count_tokens(&context);

    if tokens <= token_budget {
        return context;
    }

    let max_trims = sections.trimmable_count();
    if max_trims == 0 {
        return match format {
            ContextFormat::Json => minimal_json_context(token_budget, tokenizer),
            ContextFormat::Structured | ContextFormat::Narrative => String::new(),
        };
    }

    // ── Prefix-sum phase ────────────────────────────────────────────────
    // Pre-compute the exact formatted cost of every trimmable entry (O(N)
    // small-string tokeniser calls) and find the minimum trim count K via a
    // linear scan over the cumulative savings.
    //
    // For `Structured` format the per-entry cost is the exact rendered bullet
    // line, so the prefix sum is accurate and K is found on the first attempt.
    // For `Narrative`/`Json` a fixed overhead constant is used, which may
    // under-estimate (especially for JSON where serde_json pretty-print adds
    // ~50 tokens of field names and indentation per entry vs the 5-token
    // constant).  When the prefix-sum exhausts all entries without covering
    // the deficit (k == max_trims), the estimate is unreliable and we skip
    // directly to binary search.
    let entry_costs = sections.compute_formatted_entry_costs(format, tokenizer);
    let deficit = tokens.saturating_sub(token_budget);
    let mut cum_cost = 0usize;
    let mut k = 0usize;
    let mut prefix_sum_covered = false;
    for &cost in &entry_costs {
        cum_cost += cost;
        k += 1;
        if cum_cost >= deficit {
            prefix_sum_covered = true;
            break;
        }
    }
    let k = k.min(max_trims);

    // ── Single-shot format at estimated K ───────────────────────────────
    // Only attempt the single-shot when the prefix-sum actually covered the
    // deficit (prefix_sum_covered == true).  When k == max_trims with
    // prefix_sum_covered == false the overhead was grossly underestimated;
    // trimming everything trivially fits but produces an empty context — skip
    // straight to binary search which finds the true minimum trim count.
    if prefix_sum_covered {
        let keep = sections.keep_lengths_after_trim(k);
        let trial = format_context_for_lengths(format, sections, keep);
        let trial_tokens = tokenizer.count_tokens(&trial);
        if trial_tokens <= token_budget {
            sections.truncate_to_lengths(keep);
            return trial;
        }
    }

    // ── Binary-search fallback ───────────────────────────────────────────
    // Covers two cases:
    // 1. prefix_sum_covered && single-shot failed  → search [k+1, max_trims]
    //    (small residual from section-header token changes on boundary trims)
    // 2. !prefix_sum_covered (k == max_trims)       → search [1, max_trims]
    //    (large overhead underestimation, e.g. JSON format with many entries)
    // Binary search guarantees O(log max_trims) full-context renders.
    let mut best_fit: Option<(ContextSectionLengths, String)> = None;
    let mut low = if prefix_sum_covered { k + 1 } else { 1 };
    let mut high = max_trims;

    while low <= high {
        let mid = low + (high - low) / 2;
        let keep = sections.keep_lengths_after_trim(mid);
        let trial = format_context_for_lengths(format, sections, keep);
        let trial_tokens = tokenizer.count_tokens(&trial);

        if trial_tokens <= token_budget {
            best_fit = Some((keep, trial));
            if mid == low {
                break;
            }
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    if let Some((best_lengths, best_context)) = best_fit {
        sections.truncate_to_lengths(best_lengths);
        return best_context;
    }

    // Fully-trimmed fallback (only reachable when every entry alone exceeds
    // the token budget — essentially empty output).
    let fully_trimmed = sections.keep_lengths_after_trim(max_trims);
    sections.truncate_to_lengths(fully_trimmed);

    match format {
        ContextFormat::Json => minimal_json_context(token_budget, tokenizer),
        ContextFormat::Structured | ContextFormat::Narrative => String::new(),
    }
}

fn format_context_for_lengths(
    format: ContextFormat,
    sections: &ContextSections,
    lengths: ContextSectionLengths,
) -> String {
    format_context(
        format,
        &sections.working_memory[..lengths.working_memory],
        &sections.contradictions[..lengths.contradictions],
        &sections.semantic[..lengths.semantic],
        &sections.episodic[..lengths.episodic],
        &sections.procedural[..lengths.procedural],
        &sections.graph_connected[..lengths.graph_connected],
        &sections.causal_upstream[..lengths.causal_upstream],
    )
}

fn minimal_json_context(token_budget: usize, tokenizer: &dyn Tokenizer) -> String {
    const FULL_SCHEMA: &str = concat!(
        "{",
        "\"working_memory\":[],",
        "\"conflicts\":[],",
        "\"semantic\":[],",
        "\"episodic\":[],",
        "\"procedural\":[],",
        "\"graph_connected\":[],",
        "\"causal_upstream\":[]",
        "}"
    );

    for candidate in [FULL_SCHEMA, r#"{"truncated":true}"#, "{}"] {
        if tokenizer.count_tokens(candidate) <= token_budget {
            return candidate.to_string();
        }
    }

    "{}".to_string()
}

fn format_context(
    format: ContextFormat,
    working_section: &[ContextEntry],
    contradiction_section: &[String],
    semantic_section: &[ContextEntry],
    episodic_section: &[ContextEntry],
    procedural_section: &[ContextEntry],
    graph_section: &[String],
    causal_section: &[String],
) -> String {
    match format {
        ContextFormat::Structured => format_structured(
            working_section,
            contradiction_section,
            semantic_section,
            episodic_section,
            procedural_section,
            graph_section,
            causal_section,
        ),
        ContextFormat::Narrative => format_narrative(
            working_section,
            contradiction_section,
            semantic_section,
            episodic_section,
            procedural_section,
            graph_section,
            causal_section,
        ),
        ContextFormat::Json => format_json(
            working_section,
            contradiction_section,
            semantic_section,
            episodic_section,
            procedural_section,
            graph_section,
            causal_section,
        ),
    }
}

fn format_structured(
    working_section: &[ContextEntry],
    contradiction_section: &[String],
    semantic_section: &[ContextEntry],
    episodic_section: &[ContextEntry],
    procedural_section: &[ContextEntry],
    graph_section: &[String],
    causal_section: &[String],
) -> String {
    // Pre-size the buffer: header bytes (~30 per section) + per-entry content.
    // "• " prefix + content + "\n" = 3 extra bytes per entry.
    let capacity = working_section
        .iter()
        .map(|e| e.content.len() + 3)
        .sum::<usize>()
        + contradiction_section
            .iter()
            .map(|s| s.len() + 1)
            .sum::<usize>()
        + semantic_section
            .iter()
            .map(|e| e.content.len() + 3)
            .sum::<usize>()
        + episodic_section
            .iter()
            .map(|e| e.content.len() + 3)
            .sum::<usize>()
        + procedural_section
            .iter()
            .map(|e| e.content.len() + 3)
            .sum::<usize>()
        + graph_section.iter().map(|s| s.len() + 3).sum::<usize>()
        + causal_section.iter().map(|s| s.len() + 3).sum::<usize>()
        + 240; // section headers: 7 × ~30 bytes + padding
    let mut out = String::with_capacity(capacity);

    if !working_section.is_empty() {
        out.push_str("## Working Memory\n");
        for entry in working_section {
            out.push_str("• ");
            out.push_str(&entry.content);
            out.push('\n');
        }
        out.push('\n');
    }

    if !contradiction_section.is_empty() {
        out.push_str("## Conflicts\n");
        for line in contradiction_section {
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n');
    }

    if !semantic_section.is_empty() {
        out.push_str("## Semantic Knowledge\n");
        for entry in semantic_section {
            out.push_str("• ");
            out.push_str(&entry.content);
            out.push('\n');
        }
        out.push('\n');
    }

    if !episodic_section.is_empty() {
        out.push_str("## Episodic Records\n");
        for entry in episodic_section {
            out.push_str("• ");
            out.push_str(&entry.content);
            out.push('\n');
        }
        out.push('\n');
    }

    if !procedural_section.is_empty() {
        out.push_str("## Procedural Knowledge\n");
        for entry in procedural_section {
            out.push_str("• ");
            out.push_str(&entry.content);
            out.push('\n');
        }
        out.push('\n');
    }

    if !graph_section.is_empty() {
        out.push_str("## Related (Graph-Connected)\n");
        for line in graph_section {
            out.push_str("• ");
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n');
    }

    if !causal_section.is_empty() {
        out.push_str("## Causal Context\n");
        for line in causal_section {
            out.push_str("• ");
            out.push_str(line);
            out.push('\n');
        }
        out.push('\n');
    }

    // Truncate trailing whitespace in-place — avoids a second full-string allocation.
    let trimmed_len = out.trim_end().len();
    out.truncate(trimmed_len);
    out
}

fn format_narrative(
    working_section: &[ContextEntry],
    contradiction_section: &[String],
    semantic_section: &[ContextEntry],
    episodic_section: &[ContextEntry],
    procedural_section: &[ContextEntry],
    graph_section: &[String],
    causal_section: &[String],
) -> String {
    // Narrative format has connective phrases (~30 bytes per section) + content.
    let capacity = working_section
        .iter()
        .map(|e| e.content.len() + 8)
        .sum::<usize>()
        + contradiction_section
            .iter()
            .map(|s| s.len() + 2)
            .sum::<usize>()
        + semantic_section
            .iter()
            .map(|e| e.content.len() + 4)
            .sum::<usize>()
        + episodic_section
            .iter()
            .map(|e| e.content.len() + 8)
            .sum::<usize>()
        + procedural_section
            .iter()
            .map(|e| e.content.len() + 8)
            .sum::<usize>()
        + graph_section.iter().map(|s| s.len() + 4).sum::<usize>()
        + causal_section.iter().map(|s| s.len() + 4).sum::<usize>()
        + 200; // per-section lead phrases
    let mut out = String::with_capacity(capacity);

    if !working_section.is_empty() {
        out.push_str("Currently in focus: ");
        for (i, entry) in working_section.iter().enumerate() {
            if i > 0 {
                out.push_str(". Also, ");
            }
            out.push_str(&entry.content);
        }
        out.push_str(".\n\n");
    }

    if !contradiction_section.is_empty() {
        out.push_str("Note: There are conflicting memories. ");
        for line in contradiction_section {
            out.push_str(line);
            out.push_str(". ");
        }
        out.push('\n');
        out.push('\n');
    }

    if !semantic_section.is_empty() {
        out.push_str("Known facts: ");
        for (i, entry) in semantic_section.iter().enumerate() {
            if i > 0 {
                out.push_str(". ");
            }
            out.push_str(&entry.content);
        }
        out.push_str(".\n\n");
    }

    if !episodic_section.is_empty() {
        out.push_str("From recent experience: ");
        for (i, entry) in episodic_section.iter().enumerate() {
            if i > 0 {
                out.push_str(". Then, ");
            }
            out.push_str(&entry.content);
        }
        out.push_str(".\n\n");
    }

    if !procedural_section.is_empty() {
        out.push_str("Known procedures: ");
        for (i, entry) in procedural_section.iter().enumerate() {
            if i > 0 {
                out.push_str(". Also, ");
            }
            out.push_str(&entry.content);
        }
        out.push_str(".\n\n");
    }

    if !graph_section.is_empty() {
        out.push_str("Related context: ");
        for (i, line) in graph_section.iter().enumerate() {
            if i > 0 {
                out.push_str(". ");
            }
            out.push_str(line);
        }
        out.push_str(".\n\n");
    }

    if !causal_section.is_empty() {
        out.push_str("Causal background: ");
        for (i, line) in causal_section.iter().enumerate() {
            if i > 0 {
                out.push_str(". ");
            }
            out.push_str(line);
        }
        out.push('.');
    }

    // Truncate trailing whitespace in-place — avoids a second full-string allocation.
    let trimmed_len = out.trim_end().len();
    out.truncate(trimmed_len);
    out
}

// ── JSON serialisation helpers ──────────────────────────────────────────────
// Typed structs replace per-entry `serde_json::json!` macro calls, eliminating
// one BTreeMap allocation per section entry (O(N) savings over the old path).

#[derive(Serialize)]
struct JsonWmEntry<'a> {
    id: String,
    content: &'a str,
}

#[derive(Serialize)]
struct JsonRichEntry<'a> {
    id: String,
    content: &'a str,
    resource_evidence: serde_json::Value,
    resource_hydration_available: serde_json::Value,
    resource_preview_packages: serde_json::Value,
    resource_score_attribution: serde_json::Value,
}

#[derive(Serialize)]
struct JsonContextRoot<'a> {
    working_memory: Vec<JsonWmEntry<'a>>,
    conflicts: &'a [String],
    semantic: Vec<JsonRichEntry<'a>>,
    episodic: Vec<JsonRichEntry<'a>>,
    procedural: Vec<JsonRichEntry<'a>>,
    graph_connected: &'a [String],
    causal_upstream: &'a [String],
}

fn format_json(
    working_section: &[ContextEntry],
    contradiction_section: &[String],
    semantic_section: &[ContextEntry],
    episodic_section: &[ContextEntry],
    procedural_section: &[ContextEntry],
    graph_section: &[String],
    causal_section: &[String],
) -> String {
    fn rich_entry(entry: &ContextEntry) -> JsonRichEntry<'_> {
        JsonRichEntry {
            id: entry.id.to_string(),
            content: &entry.content,
            resource_evidence: resource_evidence_to_json(&entry.resource_evidence),
            resource_hydration_available: resource_hydration_to_json(&entry.resource_evidence),
            resource_preview_packages: resource_preview_packages_to_json(
                &entry.resource_preview_packages,
            ),
            resource_score_attribution: resource_score_attribution_to_json(
                &entry.resource_score_attribution,
            ),
        }
    }

    let root = JsonContextRoot {
        working_memory: working_section
            .iter()
            .map(|e| JsonWmEntry {
                id: e.id.to_string(),
                content: &e.content,
            })
            .collect(),
        conflicts: contradiction_section,
        semantic: semantic_section.iter().map(rich_entry).collect(),
        episodic: episodic_section.iter().map(rich_entry).collect(),
        procedural: procedural_section.iter().map(rich_entry).collect(),
        graph_connected: graph_section,
        causal_upstream: causal_section,
    };

    serde_json::to_string_pretty(&root).unwrap_or_else(|_| "{}".to_string())
}

// ── Truncation ─────────────────────────────────────────────────────────

fn truncate_to_budget(text: &str, max_tokens: usize, tokenizer: &dyn Tokenizer) -> String {
    if tokenizer.count_tokens(text) <= max_tokens {
        return text.to_string();
    }

    // Binary search for the right character position.
    let chars: Vec<char> = text.chars().collect();
    let mut lo = 0;
    let mut hi = chars.len();

    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        let slice: String = chars[..mid].iter().collect();
        if tokenizer.count_tokens(&slice) <= max_tokens {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    let result: String = chars[..lo].iter().collect();
    result
}

// ── AS clause formatters for RECALL ────────────────────────────────────

/// Format RECALL results as a narrative.
pub async fn format_as_narrative(db: &HirnDB, records: &[ScoredMemory]) -> String {
    if records.is_empty() {
        return String::new();
    }

    let mut out = String::new();

    // Pre-collect contradiction edges for all records so we can show evolution.
    let mut contradictions: Vec<Vec<MemoryId>> = Vec::new();
    for sm in records {
        let edges = db
            .cached_graph()
            .get_edges_of_type(sm.record.id(), EdgeRelation::Contradicts)
            .await
            .unwrap_or_default();
        let ids = edges
            .iter()
            .map(|e| {
                if e.source == sm.record.id() {
                    e.target
                } else {
                    e.source
                }
            })
            .collect();
        contradictions.push(ids);
    }

    // Track which record IDs are in the set for contradiction annotation.
    let record_ids: HashSet<MemoryId> = records.iter().map(|sm| sm.record.id()).collect();

    for (i, sm) in records.iter().enumerate() {
        let content = extract_content_str(&sm.record);
        let transition = if i == 0 {
            String::new()
        } else {
            // Check if there's a causal link from the previous record.
            let prev_id = records[i - 1].record.id();
            let curr_id = sm.record.id();
            let causal = db
                .cached_graph()
                .get_edges_of_type(prev_id, EdgeRelation::Causes)
                .await
                .unwrap_or_default();
            let is_causal = causal
                .iter()
                .any(|e| e.target == curr_id || e.source == curr_id);

            // Check if this record contradicts an earlier one in the sequence.
            let contradicts_earlier = contradictions[i]
                .iter()
                .any(|cid| records[..i].iter().any(|r| r.record.id() == *cid));

            if contradicts_earlier {
                "However, this was later revised: ".to_string()
            } else if is_causal {
                "As a result of this, ".to_string()
            } else {
                // Check for temporal gap.
                let prev_ts = extract_timestamp(&records[i - 1].record);
                let curr_ts = extract_timestamp(&sm.record);
                let gap_hours = (curr_ts.timestamp_ms() as f64 - prev_ts.timestamp_ms() as f64)
                    / (3600.0 * 1000.0);
                if gap_hours > 24.0 {
                    format!("After a gap of {gap_hours:.0} hours, ")
                } else {
                    "Then, ".to_string()
                }
            }
        };

        // Add timestamp context.
        let ts = extract_timestamp(&sm.record);
        let dt = ts.as_datetime();
        let ts_str = dt.format("%Y-%m-%d %H:%M").to_string();

        if !transition.is_empty() {
            out.push_str(&transition);
        }

        // Annotate superseded records (contradicted by a later record in sequence).
        let is_superseded = contradictions[i].iter().any(|cid| {
            records[i + 1..]
                .iter()
                .any(|r| r.record.id() == *cid && record_ids.contains(cid))
        });

        if is_superseded {
            write!(out, "[{ts_str}] [superseded] {content}").ok();
        } else {
            write!(out, "[{ts_str}] {content}").ok();
        }
        if i < records.len() - 1 {
            out.push_str(". ");
        }
    }

    out
}

/// Format RECALL results as a causal chain.
pub async fn format_as_causal_chain(
    db: &HirnDB,
    records: &[ScoredMemory],
    depth: Option<usize>,
) -> String {
    if records.is_empty() {
        return String::new();
    }

    let max_depth = depth.unwrap_or(usize::MAX);
    let record_ids: HashSet<MemoryId> = records.iter().map(|sm| sm.record.id()).collect();

    // Build causal chains starting from each record.
    let mut chains: Vec<Vec<(MemoryId, String, f32)>> = Vec::new();
    let mut visited: HashSet<MemoryId> = HashSet::new();

    for sm in records {
        let id = sm.record.id();
        if visited.contains(&id) {
            continue;
        }

        let mut chain = vec![(id, extract_content_str(&sm.record).to_string(), 0.0f32)];
        visited.insert(id);
        let mut current = id;
        let mut hops = 0;

        // Follow causes edges forward.
        while hops < max_depth {
            let causal_edges = db
                .cached_graph()
                .get_edges_of_type(current, EdgeRelation::Causes)
                .await
                .unwrap_or_default();
            let next_info = causal_edges
                .iter()
                .find(|e| {
                    let target = if e.source == current {
                        e.target
                    } else {
                        e.source
                    };
                    record_ids.contains(&target) && !visited.contains(&target)
                })
                .map(|e| {
                    let target = if e.source == current {
                        e.target
                    } else {
                        e.source
                    };
                    (target, e.weight)
                });

            if let Some((target, weight)) = next_info {
                if let Some(sm) = records.iter().find(|r| r.record.id() == target) {
                    chain.push((target, extract_content_str(&sm.record).to_string(), weight));
                    visited.insert(target);
                    current = target;
                    hops += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if chain.len() > 1 || chains.is_empty() {
            chains.push(chain);
        }
    }

    // Format chains.
    let mut out = String::new();
    for (ci, chain) in chains.iter().enumerate() {
        if ci > 0 {
            out.push_str("\n\n");
        }
        for (i, (_, content, weight)) in chain.iter().enumerate() {
            if i > 0 {
                write!(out, " → [caused, w={weight:.2}] → ").ok();
            }
            out.push_str(content);
        }
    }

    out
}

/// Format RECALL results as a graph structure.
pub async fn format_as_graph(db: &HirnDB, records: &[ScoredMemory]) -> String {
    if records.is_empty() {
        return "{}".to_string();
    }

    let record_ids: HashSet<MemoryId> = records.iter().map(|sm| sm.record.id()).collect();

    // Build nodes.
    let nodes: Vec<serde_json::Value> = records
        .iter()
        .map(|sm| {
            let (importance, _) = extract_record_stats_context(&sm.record);
            serde_json::json!({
                "id": sm.record.id().to_string(),
                "content": extract_content_str(&sm.record),
                "layer": format!("{:?}", sm.record.layer()),
                "importance": importance,
                "activation": sm.score_breakdown.activation,
                "score": sm.score,
            })
        })
        .collect();

    // Build edges (only between result nodes).
    let mut edges_out: Vec<serde_json::Value> = Vec::new();
    let mut seen_edges: HashSet<MemoryId> = HashSet::new();

    for sm in records {
        let id = sm.record.id();
        let all_edges = db.cached_graph().get_edges(id).await.unwrap_or_default();
        for edge in all_edges {
            if seen_edges.contains(&edge.id) {
                continue;
            }
            let other = if edge.source == id {
                edge.target
            } else {
                edge.source
            };
            if record_ids.contains(&other) {
                seen_edges.insert(edge.id);
                edges_out.push(serde_json::json!({
                    "source": edge.source.to_string(),
                    "target": edge.target.to_string(),
                    "relation": format!("{:?}", edge.relation),
                    "weight": edge.weight,
                }));
            }
        }
    }

    let graph_json = serde_json::json!({
        "nodes": nodes,
        "edges": edges_out,
    });

    serde_json::to_string_pretty(&graph_json).unwrap_or_else(|_| "{}".to_string())
}

// ── Helpers ────────────────────────────────────────────────────────────

const fn extract_timestamp(record: &MemoryRecord) -> hirn_core::Timestamp {
    match record {
        MemoryRecord::Episodic(e) => e.timestamp,
        MemoryRecord::Semantic(s) => s.created_at,
        MemoryRecord::Working(w) => w.created_at,
        MemoryRecord::Procedural(p) => p.created_at,
    }
}

const fn extract_record_stats_context(record: &MemoryRecord) -> (f32, hirn_core::Timestamp) {
    match record {
        MemoryRecord::Episodic(e) => (e.importance, e.timestamp),
        MemoryRecord::Semantic(s) => (s.confidence, s.created_at),
        MemoryRecord::Working(w) => (w.relevance_score, w.created_at),
        MemoryRecord::Procedural(p) => (p.success_rate, p.created_at),
    }
}

// ── ScopedContextAssemblyRuntime ──────────────────────────────────────────

/// Engine-side implementation of [`hirn_exec::extensions::ContextAssemblyRuntime`].
///
/// Registered once per THINK query execution, capturing the `HirnDB` reference,
/// actor identity, scored candidates, and context config. `ContextAssemblyExec`
/// (pipeline mode) calls `assemble_from_batches`, passing the raw Arrow output
/// from `ContextBudgetExec`.  This makes THINK context assembly a true
/// `SendableRecordBatchStream` operator rather than an imperative post-step.
///
/// # Design Notes
/// - The `candidates` and `content_pool` are pre-computed before plan execution;
///   they are used as the hydration pool for graph/causal neighbor resolution.
/// - Raw Arrow batches from `ContextBudgetExec` bypass the secondary Lance scan
///   for the top-k candidate set (the Arrow fast path in `assemble_think_context`).
/// - The runtime is RAII-managed via `RegisteredContextAssemblyRuntime`; the
///   caller must hold the guard until the plan output is decoded.
pub struct ScopedContextAssemblyRuntime {
    pub db: std::sync::Arc<HirnDB>,
    pub actor_id: AgentId,
    pub candidates: Vec<super::results::ScoredMemory>,
    pub content_pool: Vec<super::results::ScoredMemory>,
    pub config: ContextConfig,
    pub allowed_namespaces: Option<Vec<Namespace>>,
}

#[async_trait]
impl hirn_exec::extensions::ContextAssemblyRuntime for ScopedContextAssemblyRuntime {
    async fn assemble_from_batches(
        &self,
        raw_batches: Vec<arrow_array::RecordBatch>,
    ) -> HirnResult<Vec<u8>> {
        let result = assemble_think_context(
            &self.db,
            &self.actor_id,
            &self.candidates,
            &self.config,
            self.allowed_namespaces.as_deref(),
            Some(&self.content_pool),
            Some(&raw_batches),
        )
        .await?;

        let output = ThinkAssemblyOutput {
            context: result.context,
            token_count: result.token_count,
            records: self.candidates.clone(),
            records_included: result.records_included,
            records_excluded_count: result.records_excluded_count,
            contradictions: result.contradictions,
            conflict_groups: result.conflict_groups,
            score_distribution: result.score_distribution,
        };

        serde_json::to_vec(&output).map_err(|e| {
            hirn_core::HirnError::InvalidInput(format!(
                "ScopedContextAssemblyRuntime: ThinkAssemblyOutput serialization failed: {e}"
            ))
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::Timestamp;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::metadata::Metadata;
    use hirn_core::resource::{
        DerivedArtifactKind, EvidenceProvenance, EvidenceRole, ModalityProfile, ResourceId,
    };
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::types::{AgentId, EdgeRelation, EventType, KnowledgeType};
    use hirn_storage::memory_store::MemoryStore;

    use super::super::results::ScoreBreakdown;

    fn make_episodic(content: &str, summary: &str, importance: f32) -> ScoredMemory {
        ScoredMemory {
            record: MemoryRecord::Episodic(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content(content)
                    .summary(summary)
                    .importance(importance)
                    .agent_id(AgentId::new("test").unwrap())
                    .build()
                    .unwrap(),
            ),
            revision: None,
            score: importance,
            score_breakdown: ScoreBreakdown {
                similarity: importance,
                importance,
                recency: 0.5,
                activation: 0.0,
                causal_relevance: 0.0,
                surprise: 0.0,
                source_reliability: 0.0,
            },
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
        }
    }

    fn make_semantic(concept: &str, description: &str, confidence: f32) -> ScoredMemory {
        ScoredMemory {
            record: MemoryRecord::Semantic(
                SemanticRecord::builder()
                    .concept(concept)
                    .knowledge_type(KnowledgeType::Propositional)
                    .description(description)
                    .confidence(confidence)
                    .agent_id(AgentId::new("test").unwrap())
                    .build()
                    .unwrap(),
            ),
            revision: None,
            score: confidence,
            score_breakdown: ScoreBreakdown {
                similarity: confidence,
                importance: confidence,
                recency: 0.5,
                activation: 0.0,
                causal_relevance: 0.0,
                surprise: 0.0,
                source_reliability: 0.0,
            },
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
        }
    }

    fn make_context_entry(content: &str) -> ContextEntry {
        ContextEntry {
            id: MemoryId::new(),
            content: content.to_string(),
            token_cost: 0,
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
        }
    }

    fn make_resource_evidence_summary() -> ResourceEvidenceSummary {
        ResourceEvidenceSummary {
            resource_id: ResourceId::new(),
            role: EvidenceRole::Source,
            provenance: EvidenceProvenance::ObservedResource,
            artifact_id: None,
            artifact_kind: None,
            lifecycle_state: ResourceGovernanceState::Active,
            modality: Some(ModalityProfile::Document),
            mime_type: Some("application/pdf".into()),
            display_name: Some("architecture.pdf".into()),
            available_artifacts: vec![DerivedArtifactKind::Preview],
            has_preview: true,
            can_hydrate_preview: true,
            can_hydrate_full: false,
        }
    }

    fn test_tokenizer() -> std::sync::Arc<dyn Tokenizer> {
        hirn_provider::default_tokenizer()
    }

    struct CountingTokenizer {
        count_calls: AtomicUsize,
    }

    impl hirn_core::embed::TokenCounter for CountingTokenizer {
        fn count_tokens(&self, text: &str) -> usize {
            self.count_calls.fetch_add(1, Ordering::SeqCst);
            text.split_whitespace().count()
        }
    }

    impl Tokenizer for CountingTokenizer {
        fn truncate(&self, text: &str, max_tokens: usize) -> String {
            text.split_whitespace()
                .take(max_tokens)
                .collect::<Vec<_>>()
                .join(" ")
        }

        fn encode(&self, text: &str) -> Vec<usize> {
            (0..text.split_whitespace().count()).collect()
        }

        fn decode(&self, tokens: &[usize]) -> HirnResult<String> {
            Ok(tokens
                .iter()
                .map(|token| token.to_string())
                .collect::<Vec<_>>()
                .join(" "))
        }

        fn model_id(&self) -> &str {
            "counting"
        }

        fn max_tokens(&self) -> usize {
            4096
        }
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ql-context-tests");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap();
        (db, dir)
    }

    fn scored_record(record: MemoryRecord, score: f32) -> ScoredMemory {
        ScoredMemory {
            record,
            revision: None,
            score,
            score_breakdown: ScoreBreakdown {
                similarity: score,
                importance: score,
                recency: 0.5,
                activation: 0.0,
                causal_relevance: 0.0,
                surprise: 0.0,
                source_reliability: 0.0,
            },
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
        }
    }

    fn make_conflict_member(memory_id: MemoryId, status: ConflictMemberStatus) -> ConflictMember {
        ConflictMember {
            memory_id,
            logical_memory_id: None,
            revision_id: None,
            status,
            layer: Layer::Semantic,
            content: format!("claim {memory_id}"),
            in_result_set: true,
            source_reliability: 1.0,
            recency_basis_ms: memory_id.timestamp_ms() as i64,
        }
    }

    fn default_conflict_policy() -> ConflictResolutionPolicy {
        ConflictResolutionPolicy::default()
    }

    #[test]
    fn classify_candidates_layers() {
        let tok = test_tokenizer();
        let candidates = vec![
            make_episodic("episode content", "ep summary", 0.9),
            make_semantic("concept_a", "semantic description", 0.8),
        ];

        let classified = classify_candidates(&candidates, tok.as_ref());

        assert_eq!(classified.len(), 2);
        assert_eq!(classified[0].layer, Layer::Episodic);
        assert_eq!(classified[1].layer, Layer::Semantic);
        assert!(classified[0].token_count_full > 0);
        assert!(classified[1].token_count_full > 0);
    }

    #[test]
    fn classify_candidates_token_counts() {
        let tok = test_tokenizer();
        let candidates = vec![make_episodic(
            "This is a longer piece of content for testing token counting",
            "short summary",
            0.7,
        )];

        let classified = classify_candidates(&candidates, tok.as_ref());

        assert!(classified[0].token_count_full > classified[0].token_count_summary);
    }

    #[test]
    fn classify_and_build_layer_preserve_seeded_preview_packages() {
        let tok = test_tokenizer();
        let mut candidate = make_episodic("episode content", "ep summary", 0.9);
        candidate
            .resource_preview_packages
            .push(ResourcePreviewPackage {
                resource_id: ResourceId::new(),
                role: hirn_core::EvidenceRole::Source,
                display_name: Some("diagram.png".into()),
                modality: Some(ModalityProfile::Image),
                artifact_kind: DerivedArtifactKind::Preview,
                artifact_modality: ModalityProfile::Text,
                text_content: "seeded topology preview".into(),
                truncated: false,
            });

        let classified = classify_candidates(&[candidate], tok.as_ref());
        assert_eq!(classified[0].resource_preview_packages.len(), 1);
        assert_eq!(
            classified[0].resource_preview_packages[0].text_content,
            "seeded topology preview"
        );

        let (entries, _used) =
            build_layer_section(&classified, Layer::Episodic, 256, 0.4, None, tok.as_ref());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].resource_preview_packages.len(), 1);
        assert_eq!(
            entries[0].resource_preview_packages[0].text_content,
            "seeded topology preview"
        );
    }

    #[test]
    fn budget_allocation_with_no_working_memory() {
        let config = ContextConfig {
            token_budget: 1000,
            working_memory_reserve: 0.2,
            ..Default::default()
        };
        let tok = test_tokenizer();
        let wm: Vec<WorkingMemoryEntry> = vec![];
        let conflicts: Vec<ConflictGroup> = vec![];
        let classified = vec![
            Candidate {
                id: MemoryId::new(),
                layer: Layer::Semantic,
                full_content: "sem".into(),
                summary: "s".into(),
                score: 0.9,
                trust_score: 1.0,
                token_count_full: 10,
                token_count_summary: 2,
                tokens_full: 0,
                tokens_summary: 0,
                tokens_entity: 0,
                is_contradiction: false,
                entities: vec![],
                resource_evidence: Vec::new(),
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
            },
            Candidate {
                id: MemoryId::new(),
                layer: Layer::Episodic,
                full_content: "ep".into(),
                summary: "e".into(),
                score: 0.8,
                trust_score: 1.0,
                token_count_full: 10,
                token_count_summary: 2,
                tokens_full: 0,
                tokens_summary: 0,
                tokens_entity: 0,
                is_contradiction: false,
                entities: vec![],
                resource_evidence: Vec::new(),
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
            },
        ];

        let alloc = allocate_budget(&config, &wm, &conflicts, &classified, tok.as_ref());

        // No WM → no WM budget.
        assert_eq!(alloc.working_memory, 0);
        // Full budget distributed across tiers (direct + graph + causal).
        let total_alloc = alloc.semantic
            + alloc.episodic
            + alloc.procedural
            + alloc.graph_connected
            + alloc.causal_upstream;
        assert!(total_alloc > 0);
        assert!(total_alloc <= 1000);
    }

    #[test]
    fn budget_allocation_only_semantic() {
        let config = ContextConfig {
            token_budget: 1000,
            ..Default::default()
        };
        let tok = test_tokenizer();
        let classified = vec![Candidate {
            id: MemoryId::new(),
            layer: Layer::Semantic,
            full_content: "sem".into(),
            summary: "s".into(),
            score: 0.9,
            trust_score: 1.0,
            token_count_full: 10,
            token_count_summary: 2,
            tokens_full: 0,
            tokens_summary: 0,
            tokens_entity: 0,
            is_contradiction: false,
            entities: vec![],
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
        }];

        let alloc = allocate_budget(&config, &[], &[], &classified, tok.as_ref());

        // Only semantic → gets all remaining budget.
        assert_eq!(alloc.episodic, 0);
        assert!(alloc.semantic > 0);
    }

    #[test]
    fn budget_allocation_reserves_output_format_overhead() {
        let tok = std::sync::Arc::new(CountingTokenizer {
            count_calls: AtomicUsize::new(0),
        });
        let classified = vec![
            Candidate {
                id: MemoryId::new(),
                layer: Layer::Semantic,
                full_content: "semantic".into(),
                summary: "semantic".into(),
                score: 0.9,
                trust_score: 1.0,
                token_count_full: 1,
                token_count_summary: 1,
                tokens_full: 0,
                tokens_summary: 0,
                tokens_entity: 0,
                is_contradiction: false,
                entities: vec![],
                resource_evidence: Vec::new(),
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
            },
            Candidate {
                id: MemoryId::new(),
                layer: Layer::Episodic,
                full_content: "episodic".into(),
                summary: "episodic".into(),
                score: 0.8,
                trust_score: 1.0,
                token_count_full: 1,
                token_count_summary: 1,
                tokens_full: 0,
                tokens_summary: 0,
                tokens_entity: 0,
                is_contradiction: false,
                entities: vec![],
                resource_evidence: Vec::new(),
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
            },
            Candidate {
                id: MemoryId::new(),
                layer: Layer::Procedural,
                full_content: "procedural".into(),
                summary: "procedural".into(),
                score: 0.7,
                trust_score: 1.0,
                token_count_full: 1,
                token_count_summary: 1,
                tokens_full: 0,
                tokens_summary: 0,
                tokens_entity: 0,
                is_contradiction: false,
                entities: vec![],
                resource_evidence: Vec::new(),
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
            },
        ];
        let structured = ContextConfig {
            token_budget: 120,
            output_format: ContextFormat::Structured,
            features: ContextFeatures::empty(),
            ..Default::default()
        };
        let json = ContextConfig {
            token_budget: 120,
            output_format: ContextFormat::Json,
            features: ContextFeatures::empty(),
            ..Default::default()
        };

        let structured_alloc = allocate_budget(&structured, &[], &[], &classified, tok.as_ref());
        let json_alloc = allocate_budget(&json, &[], &[], &classified, tok.as_ref());
        let structured_total = structured_alloc.working_memory
            + structured_alloc.contradictions
            + structured_alloc.semantic
            + structured_alloc.episodic
            + structured_alloc.procedural
            + structured_alloc.graph_connected
            + structured_alloc.causal_upstream;
        let json_total = json_alloc.working_memory
            + json_alloc.contradictions
            + json_alloc.semantic
            + json_alloc.episodic
            + json_alloc.procedural
            + json_alloc.graph_connected
            + json_alloc.causal_upstream;

        assert!(structured_total < structured.token_budget);
        assert!(json_total < json.token_budget);
        assert!(json_total < structured_total);
    }

    #[test]
    fn progressive_compression_full_content() {
        let c = Candidate {
            id: MemoryId::new(),
            layer: Layer::Episodic,
            full_content: "full".into(),
            summary: "sum".into(),
            score: 0.9,
            trust_score: 1.0,
            token_count_full: 5,
            token_count_summary: 2,
            tokens_full: 0,
            tokens_summary: 0,
            tokens_entity: 0,
            is_contradiction: false,
            entities: vec![],
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
        };

        assert_eq!(determine_compression(&c, 0.4), CompressionLevel::Full);
    }

    #[test]
    fn progressive_compression_summary() {
        let mut c = Candidate {
            id: MemoryId::new(),
            layer: Layer::Episodic,
            full_content: "full".into(),
            summary: "sum".into(),
            score: 0.3,
            trust_score: 1.0,
            token_count_full: 5,
            token_count_summary: 2,
            tokens_full: 0,
            tokens_summary: 0,
            tokens_entity: 0,
            is_contradiction: false,
            entities: vec![],
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
        };

        assert_eq!(determine_compression(&c, 0.4), CompressionLevel::Summary);

        c.score = 0.1;
        assert_eq!(determine_compression(&c, 0.4), CompressionLevel::EntityOnly);
    }

    #[test]
    fn truncate_to_budget_exact() {
        let tok = test_tokenizer();
        let text = "hello world this is a test of truncation to budget limits";
        let truncated = truncate_to_budget(text, 5, tok.as_ref());
        let count = tok.count_tokens(&truncated);
        assert!(count <= 5);
    }

    #[test]
    fn format_structured_sections() {
        let wm = vec![make_context_entry("active task")];
        let conflicts = vec![];
        let sem = vec![make_context_entry("semantic fact")];
        let ep = vec![make_context_entry("episode event")];

        let output = format_structured(&wm, &conflicts, &sem, &ep, &[], &[], &[]);

        assert!(output.contains("## Working Memory"));
        assert!(output.contains("## Semantic Knowledge"));
        assert!(output.contains("## Episodic Records"));
        assert!(!output.contains("## Conflicts"));
    }

    #[test]
    fn format_structured_empty_sections_omitted() {
        let output = format_structured(&[], &[], &[], &[], &[], &[], &[]);
        assert!(output.is_empty());
    }

    #[test]
    fn format_json_valid() {
        let wm = vec![make_context_entry("context item")];
        let sem = vec![make_context_entry("fact")];
        let ep = vec![make_context_entry("event")];

        let output = format_json(&wm, &[], &sem, &ep, &[], &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert!(parsed.get("working_memory").is_some());
        assert_eq!(parsed["working_memory"][0]["content"], "context item");
        assert!(parsed["working_memory"][0].get("id").is_some());
        assert!(parsed.get("semantic").is_some());
        assert!(parsed.get("episodic").is_some());
        assert!(parsed.get("conflicts").is_some());
        assert!(parsed.get("procedural").is_some());
        assert!(parsed.get("graph_connected").is_some());
        assert!(parsed.get("causal_upstream").is_some());
    }

    #[test]
    fn format_json_includes_resource_evidence() {
        let entry = ContextEntry {
            id: MemoryId::new(),
            content: "fact Evidence: source architecture.pdf [document, preview, preview, artifacts=preview].".to_string(),
            token_cost: 0,
            resource_evidence: vec![make_resource_evidence_summary()],
            resource_preview_packages: vec![ResourcePreviewPackage {
                resource_id: ResourceId::new(),
                role: hirn_core::EvidenceRole::Source,
                display_name: Some("architecture.pdf".into()),
                modality: Some(ModalityProfile::Document),
                artifact_kind: DerivedArtifactKind::Preview,
                artifact_modality: ModalityProfile::Text,
                text_content: "deployment preview".into(),
                truncated: false,
            }],
            resource_score_attribution: Vec::new(),
        };

        let output = format_json(&[], &[], &[entry], &[], &[], &[], &[]);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert_eq!(
            parsed["semantic"][0]["resource_evidence"][0]["display_name"],
            "architecture.pdf"
        );
        assert_eq!(
            parsed["semantic"][0]["resource_evidence"][0]["role"],
            "source"
        );
        assert_eq!(
            parsed["semantic"][0]["resource_evidence"][0]["has_preview"],
            true
        );
        assert_eq!(
            parsed["semantic"][0]["resource_hydration_available"]["preview"][0]["display_name"],
            "architecture.pdf"
        );
        assert!(
            parsed["semantic"][0]["resource_hydration_available"]["full"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            parsed["semantic"][0]["resource_preview_packages"][0]["text_content"],
            "deployment preview"
        );
        assert_eq!(
            parsed["semantic"][0]["resource_preview_packages"][0]["artifact_kind"],
            "preview"
        );
    }

    #[test]
    fn format_narrative_flowing() {
        let wm = vec![make_context_entry("current task")];
        let sem = vec![make_context_entry("known fact about caching")];
        let ep = vec![make_context_entry("observed deployment")];

        let output = format_narrative(&wm, &[], &sem, &ep, &[], &[], &[]);

        assert!(output.contains("Currently in focus:"));
        assert!(output.contains("Known facts:"));
        assert!(output.contains("From recent experience:"));
    }

    #[test]
    fn fit_context_to_budget_keeps_json_valid() {
        let tok = test_tokenizer();
        let mut sections = ContextSections {
            semantic: vec![ContextEntry {
                id: MemoryId::new(),
                content: "fact ".repeat(64),
                token_cost: 0,
                resource_evidence: vec![make_resource_evidence_summary()],
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
            }],
            episodic: vec![make_context_entry(&"event ".repeat(64))],
            ..Default::default()
        };

        let output = fit_context_to_budget(ContextFormat::Json, &mut sections, 64, tok.as_ref());
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert!(parsed.is_object());
        assert!(tok.count_tokens(&output) <= 64);
    }

    #[test]
    fn fit_context_to_budget_avoids_linear_full_rerenders() {
        let tok = std::sync::Arc::new(CountingTokenizer {
            count_calls: AtomicUsize::new(0),
        });
        let n = 16usize;
        let mut sections = ContextSections {
            semantic: (0..n)
                .map(|index| make_context_entry(&format!("fact {index} payload")))
                .collect(),
            ..Default::default()
        };

        let output =
            fit_context_to_budget(ContextFormat::Structured, &mut sections, 10, tok.as_ref());

        assert!(output.split_whitespace().count() <= 10);
        assert!(sections.semantic.len() < n);
        // New algorithm: O(N) individual entry tokenisations plus O(1) full-context
        // calls, instead of the previous O(log N) full-context binary search.
        // Upper bound: N (compute_formatted_entry_costs) + a small constant for the
        // initial and final full-context calls.
        assert!(tok.count_calls.load(Ordering::SeqCst) <= n + 10);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn assemble_think_context_json_preserves_preview_packages() {
        let (db, _dir) = temp_db().await;
        let actor_id = AgentId::new("test").unwrap();
        let mut candidate = make_semantic("architecture", "deployment architecture", 0.9);
        candidate.resource_evidence = vec![make_resource_evidence_summary()];
        candidate
            .resource_preview_packages
            .push(ResourcePreviewPackage {
                resource_id: ResourceId::new(),
                role: hirn_core::EvidenceRole::Source,
                display_name: Some("architecture.pdf".into()),
                modality: Some(ModalityProfile::Document),
                artifact_kind: DerivedArtifactKind::Preview,
                artifact_modality: ModalityProfile::Text,
                text_content: "deployment preview".into(),
                truncated: false,
            });
        let config = ContextConfig {
            token_budget: 1024,
            output_format: ContextFormat::Json,
            features: ContextFeatures::empty()
                .with_resource_previews(true)
                .with_graph_context(false)
                .with_causal_chains(false)
                .with_surface_contradictions(false),
            ..Default::default()
        };

        let result =
            assemble_think_context(&db, &actor_id, &[candidate], &config, None, None, None)
                .await
                .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result.context).unwrap();

        assert!(result.token_count <= config.token_budget);
        assert_eq!(result.records_included.len(), 1);
        assert!(
            parsed["semantic"][0]["content"]
                .as_str()
                .unwrap()
                .contains("deployment architecture")
        );
        assert_eq!(
            parsed["semantic"][0]["resource_preview_packages"][0]["text_content"],
            "deployment preview"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn assemble_think_context_surfaces_conflicts_after_preselection() {
        let (db, _dir) = temp_db().await;
        let actor_id = AgentId::new("test").unwrap();

        let older_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("database-postgres")
                    .knowledge_type(KnowledgeType::Propositional)
                    .description("service uses postgres")
                    .confidence(0.92)
                    .agent_id(actor_id)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let newer_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("database-mysql")
                    .knowledge_type(KnowledgeType::Propositional)
                    .description("service uses mysql")
                    .confidence(0.91)
                    .contradiction(older_id)
                    .agent_id(actor_id)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let filler_id = db
            .store_semantic(
                SemanticRecord::builder()
                    .concept("cache")
                    .knowledge_type(KnowledgeType::Propositional)
                    .description("service uses redis")
                    .confidence(0.20)
                    .agent_id(actor_id)
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let candidates = vec![
            scored_record(db.get_memory(older_id).await.unwrap(), 0.92),
            scored_record(db.get_memory(newer_id).await.unwrap(), 0.91),
            scored_record(db.get_memory(filler_id).await.unwrap(), 0.20),
        ];
        let config = ContextConfig {
            token_budget: 256,
            features: ContextFeatures::empty()
                .with_surface_contradictions(true)
                .with_graph_context(false)
                .with_causal_chains(false)
                .with_resource_previews(false),
            ..Default::default()
        };

        let result = assemble_think_context(&db, &actor_id, &candidates, &config, None, None, None)
            .await
            .unwrap();

        assert_eq!(result.conflict_groups.len(), 1);
        let member_contents = result.conflict_groups[0]
            .members
            .iter()
            .map(|member| member.content.as_str())
            .collect::<Vec<_>>();
        assert!(
            member_contents
                .iter()
                .any(|content| content.contains("postgres"))
        );
        assert!(
            member_contents
                .iter()
                .any(|content| content.contains("mysql"))
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn assemble_think_context_graph_context_ignores_excluded_tail_candidates() {
        let (db, _dir) = temp_db().await;
        let tokenizer = test_tokenizer();
        let source_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("included seed")
                    .summary("included seed")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(AgentId::new("test").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let excluded_seed_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("excluded tail seed")
                    .summary("excluded tail seed")
                    .embedding(vec![0.0, 1.0, 0.0, 0.0])
                    .importance(0.1)
                    .agent_id(AgentId::new("test").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let hidden_neighbor_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("hidden graph neighbor")
                    .summary("hidden graph neighbor")
                    .embedding(vec![0.0, 0.0, 1.0, 0.0])
                    .importance(0.8)
                    .agent_id(AgentId::new("test").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        {
            let mut hot_graph = db.cached_graph().hot_graph_mut();
            hot_graph
                .add_edge(
                    excluded_seed_id,
                    hidden_neighbor_id,
                    EdgeRelation::RelatedTo,
                    0.7,
                    Metadata::new(),
                )
                .unwrap();
        }

        let config = ContextConfig {
            token_budget: 256,
            max_episodic_entries: 1,
            features: ContextFeatures::empty()
                .with_graph_context(true)
                .with_causal_chains(true)
                .with_surface_contradictions(false)
                .with_resource_previews(false),
            ..Default::default()
        };
        let candidates = vec![
            scored_record(db.get_memory(source_id).await.unwrap(), 0.9),
            scored_record(db.get_memory(excluded_seed_id).await.unwrap(), 0.1),
        ];

        let result = assemble_think_context(
            &db,
            &AgentId::new("test").unwrap(),
            &candidates,
            &config,
            None,
            None,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.records_included, vec![source_id]);
        assert!(!result.context.contains("hidden graph neighbor"));

        let direct_ids = collect_direct_section_ids(&[make_context_entry("included")], &[], &[]);
        assert_eq!(direct_ids.len(), 1);
        assert!(tokenizer.count_tokens(&result.context) <= config.token_budget);
    }

    #[test]
    fn build_layer_section_includes_resource_evidence_summary() {
        let tok = test_tokenizer();
        let candidate = Candidate {
            id: MemoryId::new(),
            layer: Layer::Semantic,
            full_content: "deployment architecture".into(),
            summary: "architecture summary".into(),
            score: 0.9,
            trust_score: 1.0,
            token_count_full: 5,
            token_count_summary: 2,
            tokens_full: 0,
            tokens_summary: 0,
            tokens_entity: 0,
            is_contradiction: false,
            entities: vec![],
            resource_evidence: vec![make_resource_evidence_summary()],
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
        };

        let (entries, _used) =
            build_layer_section(&[candidate], Layer::Semantic, 256, 0.4, None, tok.as_ref());

        assert_eq!(entries.len(), 1);
        assert!(entries[0].content.contains("Evidence:"));
        assert!(entries[0].content.contains("architecture.pdf"));
        assert_eq!(entries[0].resource_evidence.len(), 1);
    }

    #[test]
    fn build_layer_section_stops_rendering_after_budget_exhausted() {
        let tok = std::sync::Arc::new(CountingTokenizer {
            count_calls: AtomicUsize::new(0),
        });
        let candidates = (0..3)
            .map(|index| Candidate {
                id: MemoryId::new(),
                layer: Layer::Semantic,
                full_content: format!("full {index} uses five tokens"),
                summary: format!("summary {index}"),
                score: 0.9 - (index as f32 * 0.1),
                trust_score: 1.0,
                token_count_full: 5,
                token_count_summary: 2,
                tokens_full: 0,
                tokens_summary: 0,
                tokens_entity: 0,
                is_contradiction: false,
                entities: vec![format!("entity-{index}")],
                resource_evidence: Vec::new(),
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
            })
            .collect::<Vec<_>>();

        let (entries, _used) =
            build_layer_section(&candidates, Layer::Semantic, 5, 0.4, None, tok.as_ref());

        assert_eq!(entries.len(), 1);
        assert_eq!(tok.count_calls.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn build_layer_section_stops_after_max_entries() {
        let tok = std::sync::Arc::new(CountingTokenizer {
            count_calls: AtomicUsize::new(0),
        });
        let candidates = (0..4)
            .map(|index| Candidate {
                id: MemoryId::new(),
                layer: Layer::Episodic,
                full_content: format!("full {index} uses five tokens"),
                summary: format!("summary {index}"),
                score: 0.9 - (index as f32 * 0.1),
                trust_score: 1.0,
                token_count_full: 5,
                token_count_summary: 2,
                tokens_full: 0,
                tokens_summary: 0,
                tokens_entity: 0,
                is_contradiction: false,
                entities: vec![format!("entity-{index}")],
                resource_evidence: Vec::new(),
                resource_preview_packages: Vec::new(),
                resource_score_attribution: Vec::new(),
            })
            .collect::<Vec<_>>();

        let (entries, _used) = build_layer_section(
            &candidates,
            Layer::Episodic,
            100,
            0.4,
            Some(2),
            tok.as_ref(),
        );

        assert_eq!(entries.len(), 2);
        assert_eq!(tok.count_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn build_conflict_groups_tracks_omitted_visible_members() {
        let visible_a = MemoryId::new();
        let visible_b = MemoryId::new();
        let hidden = MemoryId::new();

        let visible_members = BTreeMap::from([
            (
                visible_a,
                make_conflict_member(visible_a, ConflictMemberStatus::Active),
            ),
            (
                visible_b,
                make_conflict_member(visible_b, ConflictMemberStatus::Active),
            ),
        ]);
        let adjacency = BTreeMap::from([
            (visible_a, vec![hidden]),
            (visible_b, vec![hidden]),
            (hidden, vec![visible_a, visible_b]),
        ]);
        let pair_edges = vec![
            ConflictEdgeMeta {
                a: normalize_conflict_pair(visible_a, hidden).0,
                b: normalize_conflict_pair(visible_a, hidden).1,
                confidence: 0.9,
                evidence_count: 1,
                resolved: false,
            },
            ConflictEdgeMeta {
                a: normalize_conflict_pair(visible_b, hidden).0,
                b: normalize_conflict_pair(visible_b, hidden).1,
                confidence: 0.8,
                evidence_count: 2,
                resolved: false,
            },
        ];

        let groups = build_conflict_groups(
            &visible_members,
            &adjacency,
            &pair_edges,
            &HashSet::new(),
            default_conflict_policy(),
        );

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members.len(), 2);
        assert_eq!(groups[0].omitted_member_count, 1);
        assert_eq!(groups[0].pair_count, 2);
        assert_eq!(
            groups[0].arbitration_status,
            ConflictArbitrationStatus::Unresolved
        );
        assert!(groups[0].authoritative_memory_id.is_none());
        assert!(groups[0].preferred_memory_id.is_none());
    }

    #[test]
    fn derive_conflict_arbitration_status_marks_superseded_components() {
        let active = make_conflict_member(MemoryId::new(), ConflictMemberStatus::Active);
        let superseded = make_conflict_member(MemoryId::new(), ConflictMemberStatus::Superseded);
        let edges = [ConflictEdgeMeta {
            a: normalize_conflict_pair(active.memory_id, superseded.memory_id).0,
            b: normalize_conflict_pair(active.memory_id, superseded.memory_id).1,
            confidence: 0.95,
            evidence_count: 1,
            resolved: false,
        }];

        let status = derive_conflict_arbitration_status(&[active, superseded], &[&edges[0]], 0);

        assert_eq!(status, ConflictArbitrationStatus::Superseded);
    }

    #[test]
    fn derive_conflict_arbitration_status_marks_retracted_components_resolved() {
        let active = make_conflict_member(MemoryId::new(), ConflictMemberStatus::Active);
        let retracted = make_conflict_member(MemoryId::new(), ConflictMemberStatus::Retracted);

        let status = derive_conflict_arbitration_status(&[active, retracted], &[], 0);

        assert_eq!(status, ConflictArbitrationStatus::Resolved);
    }

    #[test]
    fn authoritative_conflict_memory_id_requires_single_visible_active_member() {
        let active = make_conflict_member(MemoryId::new(), ConflictMemberStatus::Active);
        let expected = active.memory_id;
        let superseded = make_conflict_member(MemoryId::new(), ConflictMemberStatus::Superseded);

        let selected = authoritative_conflict_memory_id(&[active, superseded], 0);

        assert_eq!(selected, Some(expected));
    }

    #[test]
    fn select_conflict_preferred_memory_id_prefers_reliable_supported_claims() {
        let mut less_reliable = make_conflict_member(MemoryId::new(), ConflictMemberStatus::Active);
        less_reliable.source_reliability = 0.4;

        let mut more_reliable = make_conflict_member(MemoryId::new(), ConflictMemberStatus::Active);
        more_reliable.source_reliability = 0.9;
        let expected = more_reliable.memory_id;

        let supporting = make_conflict_member(MemoryId::new(), ConflictMemberStatus::Superseded);

        let edges = [
            ConflictEdgeMeta {
                a: normalize_conflict_pair(less_reliable.memory_id, supporting.memory_id).0,
                b: normalize_conflict_pair(less_reliable.memory_id, supporting.memory_id).1,
                confidence: 0.6,
                evidence_count: 1,
                resolved: false,
            },
            ConflictEdgeMeta {
                a: normalize_conflict_pair(more_reliable.memory_id, supporting.memory_id).0,
                b: normalize_conflict_pair(more_reliable.memory_id, supporting.memory_id).1,
                confidence: 0.95,
                evidence_count: 3,
                resolved: false,
            },
        ];

        let selected = select_conflict_preferred_memory_id(
            &[less_reliable, more_reliable, supporting],
            &[&edges[0], &edges[1]],
            0,
            &HashSet::new(),
            default_conflict_policy(),
        );

        assert_eq!(selected, Some(expected));
    }

    #[test]
    fn select_conflict_preferred_memory_id_refuses_partial_visibility() {
        let active = make_conflict_member(MemoryId::new(), ConflictMemberStatus::Active);

        let selected = select_conflict_preferred_memory_id(
            &[active],
            &[],
            1,
            &HashSet::new(),
            default_conflict_policy(),
        );

        assert_eq!(selected, None);
    }

    #[test]
    fn select_conflict_preferred_memory_id_can_prioritize_recency() {
        let older_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap();
        let newer_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FB0").unwrap();

        let mut older = make_conflict_member(older_id, ConflictMemberStatus::Active);
        older.source_reliability = 0.95;
        older.revision_id = Some(RevisionId::from_memory_id(older_id));

        let mut newer = make_conflict_member(newer_id, ConflictMemberStatus::Active);
        newer.source_reliability = 0.55;
        newer.revision_id = Some(RevisionId::from_memory_id(newer_id));

        let policy = ConflictResolutionPolicy {
            recency_weight: 0.80,
            source_reliability_weight: 0.10,
            supporting_evidence_weight: 0.10,
            human_override_weight: 0.0,
            prefer_human_override: true,
        };

        let selected =
            select_conflict_preferred_memory_id(&[older, newer], &[], 0, &HashSet::new(), policy);

        assert_eq!(selected, Some(newer_id));
    }

    #[test]
    fn select_conflict_preferred_memory_id_can_prioritize_reliability_over_recency() {
        let older_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FB1").unwrap();
        let newer_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FB2").unwrap();

        let mut older = make_conflict_member(older_id, ConflictMemberStatus::Active);
        older.source_reliability = 0.95;
        older.revision_id = Some(RevisionId::from_memory_id(older_id));

        let mut newer = make_conflict_member(newer_id, ConflictMemberStatus::Active);
        newer.source_reliability = 0.35;
        newer.revision_id = Some(RevisionId::from_memory_id(newer_id));

        let policy = ConflictResolutionPolicy {
            recency_weight: 0.10,
            source_reliability_weight: 0.80,
            supporting_evidence_weight: 0.10,
            human_override_weight: 0.0,
            prefer_human_override: true,
        };

        let selected =
            select_conflict_preferred_memory_id(&[older, newer], &[], 0, &HashSet::new(), policy);

        assert_eq!(selected, Some(older_id));
    }

    #[test]
    fn select_conflict_preferred_memory_id_is_stable_across_input_order() {
        let older_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FB3").unwrap();
        let newer_id = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FB4").unwrap();

        let mut older = make_conflict_member(older_id, ConflictMemberStatus::Active);
        older.source_reliability = 0.95;
        older.revision_id = Some(RevisionId::from_memory_id(older_id));

        let mut newer = make_conflict_member(newer_id, ConflictMemberStatus::Active);
        newer.source_reliability = 0.55;
        newer.revision_id = Some(RevisionId::from_memory_id(newer_id));

        let policy = ConflictResolutionPolicy {
            recency_weight: 0.15,
            source_reliability_weight: 0.75,
            supporting_evidence_weight: 0.10,
            human_override_weight: 0.0,
            prefer_human_override: true,
        };

        let forward = select_conflict_preferred_memory_id(
            &[older.clone(), newer.clone()],
            &[],
            0,
            &HashSet::new(),
            policy,
        );
        let reverse =
            select_conflict_preferred_memory_id(&[newer, older], &[], 0, &HashSet::new(), policy);

        assert_eq!(forward, Some(older_id));
        assert_eq!(reverse, Some(older_id));
    }

    #[test]
    fn select_conflict_preferred_memory_id_prefers_explicit_override() {
        let regular_id = MemoryId::new();
        let override_id = MemoryId::new();

        let mut regular = make_conflict_member(regular_id, ConflictMemberStatus::Active);
        regular.source_reliability = 0.95;

        let mut overridden = make_conflict_member(override_id, ConflictMemberStatus::Active);
        overridden.source_reliability = 0.30;

        let override_members = HashSet::from([override_id]);
        let selected = select_conflict_preferred_memory_id(
            &[regular, overridden],
            &[],
            0,
            &override_members,
            default_conflict_policy(),
        );

        assert_eq!(selected, Some(override_id));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn context_graph_helpers_use_authoritative_cached_graph_edges() {
        let (db, _dir) = temp_db().await;
        let _tokenizer = test_tokenizer();
        // Both records must share the same namespace — the hot-graph enforces
        // that edges are intra-namespace only.
        let ns = Namespace::new("graph_helper_ns").unwrap();

        let source_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("source event")
                    .summary("source event")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.9)
                    .namespace(ns)
                    .agent_id(AgentId::new("test").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let target_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("hot only neighbor")
                    .summary("hot only neighbor")
                    .embedding(vec![0.0, 1.0, 0.0, 0.0])
                    .importance(0.8)
                    .namespace(ns)
                    .agent_id(AgentId::new("test").unwrap())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        {
            let mut hot_graph = db.cached_graph().hot_graph_mut();
            hot_graph
                .add_edge(
                    source_id,
                    target_id,
                    EdgeRelation::Causes,
                    0.8,
                    Metadata::new(),
                )
                .unwrap();
        }

        let source_record = scored_record(
            MemoryRecord::Episodic(db.get_episode(source_id).await.unwrap()),
            0.9,
        );
        let target_record = scored_record(
            MemoryRecord::Episodic(db.get_episode(target_id).await.unwrap()),
            0.8,
        );

        let narrative =
            format_as_narrative(&db, &[source_record.clone(), target_record.clone()]).await;
        assert!(narrative.contains("As a result of this"));

        let causal_chain = format_as_causal_chain(
            &db,
            &[source_record.clone(), target_record.clone()],
            Some(1),
        )
        .await;
        assert!(causal_chain.contains("source event"));
        assert!(causal_chain.contains("hot only neighbor"));
        assert!(causal_chain.contains("[caused, w=0.80]"));

        let graph_json = format_as_graph(&db, &[source_record, target_record]).await;
        let parsed: serde_json::Value = serde_json::from_str(&graph_json).unwrap();
        let edges = parsed["edges"].as_array().unwrap();
        assert!(edges.iter().any(|edge| {
            edge["source"] == source_id.to_string()
                && edge["target"] == target_id.to_string()
                && edge["relation"] == "Causes"
        }));
    }

    #[test]
    fn build_semantic_conflict_groups_surfaces_preferred_and_authoritative_members() {
        let mut older = SemanticRecord::builder()
            .concept("policy")
            .knowledge_type(KnowledgeType::Propositional)
            .description("Policy remains disabled")
            .confidence(0.55)
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();
        let mut newer = SemanticRecord::builder()
            .concept("policy")
            .knowledge_type(KnowledgeType::Propositional)
            .description("Policy is enabled")
            .confidence(0.55)
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();

        older.valid_from = Timestamp::from_millis(10);
        newer.valid_from = Timestamp::from_millis(20);
        older.revision_id = RevisionId::from_memory_id(older.id);
        newer.revision_id = RevisionId::from_memory_id(newer.id);
        older.contradiction_ids.push(newer.id);
        newer.contradiction_ids.push(older.id);

        let groups = build_semantic_conflict_groups(
            &[older.clone(), newer.clone()],
            default_conflict_policy(),
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].preferred_memory_id, Some(newer.id));
        assert_eq!(groups[0].authoritative_memory_id, None);
        assert_eq!(
            groups[0].arbitration_status,
            ConflictArbitrationStatus::Unresolved
        );

        older.superseded_by = Some(MemoryId::new());
        let groups =
            build_semantic_conflict_groups(&[older, newer.clone()], default_conflict_policy());
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].authoritative_memory_id, Some(newer.id));
        assert_eq!(
            groups[0].arbitration_status,
            ConflictArbitrationStatus::Superseded
        );
    }

    #[test]
    fn score_distribution_computed() {
        let candidates = vec![
            make_episodic("a", "s", 0.9),
            make_episodic("b", "s", 0.5),
            make_episodic("c", "s", 0.3),
        ];
        let ids: Vec<MemoryId> = candidates.iter().map(|c| c.record.id()).collect();

        let dist = compute_score_distribution(&candidates, &ids);

        assert!((dist.min - 0.3).abs() < 0.01);
        assert!((dist.max - 0.9).abs() < 0.01);
    }
}
