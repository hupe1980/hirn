//! `HirnMemory` — zero-config, high-level memory API.
//!
//! Wraps [`HirnDB`] with automatic embedding, entity extraction, and
//! environment-based provider discovery.  Five lines of code is all you need:
//!
//! ```rust,no_run
//! use hirn::prelude::*;
//!
//! # async fn demo() -> HirnResult<()> {
//! let memory = HirnMemory::open("./brain").await?;
//! memory.remember("User prefers dark mode").await?;
//! let ctx = memory.think("What are the user's UI preferences?", 2048).await?;
//! println!("{}", ctx.context);
//! # Ok(())
//! # }
//! ```

use std::path::Path;
use std::sync::Arc;

use hirn_core::embed::{Embedder, EntityExtractor, ExtractedEntity};
use hirn_core::episodic::{EntityRef, EpisodicRecord};
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, EventType, Namespace};
use hirn_core::{HirnConfig, HirnError, HirnResult};
use hirn_engine::HirnDB;
use hirn_engine::ProviderRegistry;
use hirn_engine::activation::ActivationMode;
use hirn_engine::ql::QueryResult;
use hirn_engine::ql::context::{ContextConfig, ContextFormat, ThinkResult};
use hirn_engine::recall::{LayerFilter, RecallResult};
use hirn_engine::scoring::ScoringWeights;
use hirn_storage::{HirnDb, HirnDbConfig};

/// Maximum token budget accepted by [`HirnMemory::think()`].
///
/// Corresponds to the largest commercially available LLM context window as of
/// 2026. Larger values are rejected with [`HirnError::InvalidInput`] to prevent
/// accidental OOM from unbounded context assembly.
pub const MAX_TOKEN_BUDGET: usize = 128_000;

/// Zero-config memory API.
///
/// Discovers providers from environment variables automatically:
///
/// | Variable | Effect |
/// |----------|--------|
/// | `OPENAI_API_KEY` | Uses OpenAI embeddings + LLM |
/// | `OLLAMA_HOST` | Uses Ollama embeddings + LLM |
/// | *(none)* | Fails unless `allow_pseudo_embedder_fallback = true`, then uses `PseudoEmbedder` for explicit dev/test mode |
pub struct HirnMemory {
    db: HirnDB,
    entity_extractor: Arc<dyn EntityExtractor>,
    agent_id: AgentId,
}

impl HirnMemory {
    /// Open (or create) a brain at the given path.
    ///
    /// Provider discovery runs automatically — set `OPENAI_API_KEY` or
    /// `OLLAMA_HOST` for real embeddings. When no real embedder is configured,
    /// this open path fails unless `hirn.toml` or [`HirnConfig`] explicitly sets
    /// `allow_pseudo_embedder_fallback = true` for development or testing.
    ///
    /// If a `hirn.toml` file exists in the brain directory, its values
    /// override the defaults. Use [`open_with_config`](Self::open_with_config)
    /// for full programmatic control (HirnConfig > hirn.toml > defaults).
    pub async fn open(path: impl AsRef<Path>) -> HirnResult<Self> {
        let path = path.as_ref();

        // Start with sensible defaults — admission enabled for HirnMemory.
        let mut config = HirnConfig::builder().db_path(path).build()?;
        config.admission_enabled = true;

        // Auto-load hirn.toml from brain directory (parent of db_path) if present.
        let brain_dir = path.parent().unwrap_or(path);
        let toml_path = brain_dir.join("hirn.toml");
        if toml_path.is_file() {
            config = load_toml_over_defaults(&toml_path, config)?;
        }

        Self::open_with_config(config).await
    }

    /// Open with an explicit [`HirnConfig`].
    ///
    /// The provided config takes full precedence — `hirn.toml` is **not**
    /// auto-loaded. Use [`open`](Self::open) for auto-configuration.
    pub async fn open_with_config(mut config: HirnConfig) -> HirnResult<Self> {
        let registry = if config.allow_pseudo_embedder_fallback {
            ProviderRegistry::from_env()
        } else {
            ProviderRegistry::from_env_strict()
        };

        // Resolve embedder and align embedding dimensions.
        let embedder: Arc<dyn Embedder> = registry
            .embedder()
            .ok_or_else(|| HirnError::InvalidConfig {
                field: "allow_pseudo_embedder_fallback".to_string(),
                value: config.allow_pseudo_embedder_fallback.to_string(),
                reason: "HirnMemory requires a configured embedder from environment or explicit allow_pseudo_embedder_fallback = true for dev/test mode".to_string(),
            })?;
        config.embedding_dimensions = hirn_core::EmbeddingDimension::new(
            u32::try_from(embedder.dimensions()).map_err(|_| HirnError::InvalidConfig {
                field: "embedding_dimensions".to_string(),
                value: embedder.dimensions().to_string(),
                reason: "embedder reported dimension exceeds u32::MAX".to_string(),
            })?,
        )?;

        // Open LanceDB storage at the same path.
        let lance_path = config
            .db_path
            .parent()
            .unwrap_or(config.db_path.as_path())
            .join("lance");
        let storage_config = HirnDbConfig::local(lance_path.to_string_lossy());
        let hirn_storage = HirnDb::open(storage_config)
            .await
            .map_err(|e| HirnError::storage(e.to_string()))?;
        let storage: Arc<dyn hirn_storage::PhysicalStore> = hirn_storage.store_arc();

        let mut db = HirnDB::open_with_config(config, storage).await?;
        db.set_embedder(embedder);
        if let Some(tokenizer) = registry.tokenizer() {
            db.set_tokenizer(tokenizer);
        }
        db.setup_default_admission_pipeline();

        let agent_id = AgentId::new("hirn_memory")?;
        db.register_agent(&agent_id, "HirnMemory default agent")
            .await?;

        let entity_extractor: Arc<dyn EntityExtractor> =
            Arc::new(hirn_provider::RegexEntityExtractor::new());

        Ok(Self {
            db,
            entity_extractor,
            agent_id,
        })
    }

    /// Store a text memory with automatic embedding and entity extraction.
    ///
    /// Returns the assigned `MemoryId`.
    pub async fn remember(&self, text: &str) -> HirnResult<crate::MemoryId> {
        let embedding = self.db.embed_text(text).await?;

        let entities = self
            .entity_extractor
            .extract_entities(text, &[])
            .await
            .unwrap_or_default();

        let entity_refs = extracted_to_refs(&entities);

        let record = EpisodicRecord::builder()
            .content(text)
            .embedding(embedding)
            .event_type(EventType::Observation)
            .agent_id(self.agent_id.clone())
            .entities(entity_refs)
            .build()?;

        self.db.episodic().remember(record).await
    }

    /// Assemble optimal LLM context for a query under `budget` tokens.
    ///
    /// `budget` is capped at [`MAX_TOKEN_BUDGET`] (128 000 tokens — the largest
    /// commercially available context window as of 2026). Passing a larger value
    /// returns [`HirnError::InvalidInput`].
    pub async fn think(&self, query: &str, budget: usize) -> HirnResult<ThinkResult> {
        if budget > MAX_TOKEN_BUDGET {
            return Err(HirnError::InvalidInput(format!(
                "token budget {budget} exceeds the maximum allowed value of {MAX_TOKEN_BUDGET}"
            )));
        }
        let embedding = self.db.embed_text(query).await?;
        self.db
            .recall_view()
            .think(embedding)
            .budget(budget)
            .execute()
            .await
    }

    /// Recall memories relevant to a query.
    pub async fn recall(&self, query: &str, limit: usize) -> HirnResult<Vec<RecallResult>> {
        let embedding = self.db.embed_text(query).await?;
        self.db
            .recall_view()
            .query(embedding)
            .limit(limit)
            .execute()
            .await
    }

    /// Execute a HirnQL query string.
    ///
    /// Returns [`QueryResult`] for the embedded HirnQL statement classes that
    /// remain supported on the authoritative engine bridge.
    ///
    /// Graph mutation, policy mutation, watch, and other direct-only surfaces
    /// are intentionally not available through `query()` and should use the
    /// corresponding view or daemon APIs instead.
    ///
    /// Parse errors include line/column position information.
    pub async fn query(&self, hirnql: &str) -> HirnResult<QueryResult> {
        self.db.ql().execute(hirnql).await
    }

    /// Get a reference to the underlying [`HirnDB`] for advanced operations.
    #[must_use]
    pub fn db(&self) -> &HirnDB {
        &self.db
    }

    /// Get a mutable reference to the underlying [`HirnDB`].
    pub fn db_mut(&mut self) -> &mut HirnDB {
        &mut self.db
    }

    // ── Level 3: Builder API ──────────────────────────────────────

    /// Fluent recall builder that accepts text and auto-embeds on execute.
    ///
    /// ```ignore
    /// let results = memory.recall_builder("auth")
    ///     .episodic_only()
    ///     .limit(20)
    ///     .activation(ActivationMode::Spreading)
    ///     .depth(2)
    ///     .execute()
    ///     .await?;
    /// ```
    pub fn recall_builder(&self, about: &str) -> MemoryRecallBuilder<'_> {
        MemoryRecallBuilder::new(self, about.to_owned())
    }

    /// Fluent think builder that accepts text and auto-embeds on execute.
    ///
    /// ```ignore
    /// let ctx = memory.think_builder("deployment strategies")
    ///     .budget(4096)
    ///     .episodic_only()
    ///     .execute()
    ///     .await?;
    /// ```
    pub fn think_builder(&self, about: &str) -> MemoryThinkBuilder<'_> {
        MemoryThinkBuilder::new(self, about.to_owned())
    }
}

// ── MemoryRecallBuilder ─────────────────────────────────────────────────

/// Text-based fluent recall builder. Auto-embeds on [`execute`](Self::execute).
#[must_use]
pub struct MemoryRecallBuilder<'a> {
    memory: &'a HirnMemory,
    about: String,
    limit: usize,
    threshold: Option<f32>,
    layer_filter: LayerFilter,
    namespace: Option<Namespace>,
    after: Option<Timestamp>,
    before: Option<Timestamp>,
    weights: Option<ScoringWeights>,
    activation_mode: ActivationMode,
    activation_depth: Option<usize>,
    hybrid: bool,
    agent_id: Option<String>,
}

impl<'a> MemoryRecallBuilder<'a> {
    fn new(memory: &'a HirnMemory, about: String) -> Self {
        Self {
            memory,
            about,
            limit: 10,
            threshold: None,
            layer_filter: LayerFilter::default(),
            namespace: None,
            after: None,
            before: None,
            weights: None,
            activation_mode: ActivationMode::None,
            activation_depth: None,
            hybrid: false,
            agent_id: None,
        }
    }

    /// Maximum number of results.
    pub fn limit(mut self, k: usize) -> Self {
        self.limit = k;
        self
    }

    /// Minimum similarity threshold.
    pub fn threshold(mut self, min: f32) -> Self {
        self.threshold = Some(min);
        self
    }

    /// Only episodic records.
    pub fn episodic_only(mut self) -> Self {
        self.layer_filter = LayerFilter::EpisodicOnly;
        self
    }

    /// Only semantic records.
    pub fn semantic_only(mut self) -> Self {
        self.layer_filter = LayerFilter::SemanticOnly;
        self
    }

    /// Only procedural records.
    pub fn procedural_only(mut self) -> Self {
        self.layer_filter = LayerFilter::ProceduralOnly;
        self
    }

    /// Filter by namespace.
    pub fn namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Only records after this timestamp.
    pub fn after(mut self, ts: Timestamp) -> Self {
        self.after = Some(ts);
        self
    }

    /// Only records before this timestamp.
    pub fn before(mut self, ts: Timestamp) -> Self {
        self.before = Some(ts);
        self
    }

    /// Custom scoring weights.
    pub fn weights(mut self, w: ScoringWeights) -> Self {
        self.weights = Some(w);
        self
    }

    /// Graph activation mode.
    pub fn activation(mut self, mode: ActivationMode) -> Self {
        self.activation_mode = mode;
        self
    }

    /// Spreading activation depth.
    pub fn depth(mut self, d: usize) -> Self {
        self.activation_depth = Some(d);
        self
    }

    /// Enable hybrid BM25 + vector search with Reciprocal Rank Fusion.
    pub fn hybrid(mut self, enable: bool) -> Self {
        self.hybrid = enable;
        self
    }

    /// Filter by agent ID for Cedar policy enforcement.
    pub fn agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    /// Only records within a time range (inclusive).
    pub fn between(mut self, start: Timestamp, end: Timestamp) -> Self {
        self.after = Some(start);
        self.before = Some(end);
        self
    }

    /// Execute: embed text → recall → return results.
    pub async fn execute(self) -> HirnResult<Vec<RecallResult>> {
        let embedding = self.memory.db.embed_text(&self.about).await?;
        let mut builder = self
            .memory
            .db
            .recall_view()
            .query(embedding)
            .limit(self.limit)
            .activation(self.activation_mode)
            .query_text(self.about);

        if let Some(t) = self.threshold {
            builder = builder.threshold(t);
        }
        if let Some(ns) = self.namespace {
            builder = builder.namespace(ns);
        }
        if let Some(ts) = self.after {
            builder = builder.after(ts);
        }
        if let Some(ts) = self.before {
            builder = builder.before(ts);
        }
        if let Some(w) = self.weights {
            builder = builder.weights(w);
        }
        if let Some(d) = self.activation_depth {
            builder = builder.depth(d);
        }
        if self.hybrid {
            builder = builder.hybrid(true);
        }
        if let Some(id) = self.agent_id {
            builder = builder.agent_id(id);
        }
        match self.layer_filter {
            LayerFilter::EpisodicOnly => builder = builder.episodic_only(),
            LayerFilter::SemanticOnly => builder = builder.semantic_only(),
            LayerFilter::ProceduralOnly => builder = builder.procedural_only(),
            LayerFilter::All => {}
        }

        builder.execute().await
    }
}

// ── MemoryThinkBuilder ──────────────────────────────────────────────────

/// Text-based fluent think builder. Auto-embeds on [`execute`](Self::execute).
#[must_use]
pub struct MemoryThinkBuilder<'a> {
    memory: &'a HirnMemory,
    about: String,
    budget: Option<usize>,
    limit: usize,
    layer_filter: LayerFilter,
    namespace: Option<Namespace>,
    after: Option<Timestamp>,
    before: Option<Timestamp>,
    weights: Option<ScoringWeights>,
    activation_mode: ActivationMode,
    activation_depth: Option<usize>,
    format: Option<ContextFormat>,
    context_config: Option<ContextConfig>,
}

impl<'a> MemoryThinkBuilder<'a> {
    fn new(memory: &'a HirnMemory, about: String) -> Self {
        Self {
            memory,
            about,
            budget: None,
            limit: 50,
            layer_filter: LayerFilter::default(),
            namespace: None,
            after: None,
            before: None,
            weights: None,
            activation_mode: ActivationMode::None,
            activation_depth: None,
            format: None,
            context_config: None,
        }
    }

    /// Token budget for context assembly.
    pub fn budget(mut self, tokens: usize) -> Self {
        self.budget = Some(tokens);
        self
    }

    /// Maximum candidates to consider.
    pub fn limit(mut self, k: usize) -> Self {
        self.limit = k;
        self
    }

    /// Only episodic records.
    pub fn episodic_only(mut self) -> Self {
        self.layer_filter = LayerFilter::EpisodicOnly;
        self
    }

    /// Only semantic records.
    pub fn semantic_only(mut self) -> Self {
        self.layer_filter = LayerFilter::SemanticOnly;
        self
    }

    /// Filter by namespace.
    pub fn namespace(mut self, ns: Namespace) -> Self {
        self.namespace = Some(ns);
        self
    }

    /// Only records after this timestamp.
    pub fn after(mut self, ts: Timestamp) -> Self {
        self.after = Some(ts);
        self
    }

    /// Only records before this timestamp.
    pub fn before(mut self, ts: Timestamp) -> Self {
        self.before = Some(ts);
        self
    }

    /// Graph activation mode.
    pub fn activation(mut self, mode: ActivationMode) -> Self {
        self.activation_mode = mode;
        self
    }

    /// Spreading activation depth.
    pub fn depth(mut self, d: usize) -> Self {
        self.activation_depth = Some(d);
        self
    }

    /// Custom scoring weights.
    pub fn weights(mut self, w: ScoringWeights) -> Self {
        self.weights = Some(w);
        self
    }

    /// Output format for the assembled context.
    pub fn format(mut self, fmt: ContextFormat) -> Self {
        self.format = Some(fmt);
        self
    }

    /// Full context configuration (format, include-metadata flags, etc.).
    pub fn context_config(mut self, config: ContextConfig) -> Self {
        self.context_config = Some(config);
        self
    }

    /// Only records within a time range (inclusive).
    pub fn between(mut self, start: Timestamp, end: Timestamp) -> Self {
        self.after = Some(start);
        self.before = Some(end);
        self
    }

    /// Execute: embed text → think → return context.
    pub async fn execute(self) -> HirnResult<ThinkResult> {
        let embedding = self.memory.db.embed_text(&self.about).await?;
        let mut builder = self
            .memory
            .db
            .recall_view()
            .think(embedding)
            .limit(self.limit)
            .activation(self.activation_mode);

        if let Some(b) = self.budget {
            builder = builder.budget(b);
        }
        if let Some(ns) = self.namespace {
            builder = builder.namespace(ns);
        }
        if let Some(ts) = self.after {
            builder = builder.after(ts);
        }
        if let Some(ts) = self.before {
            builder = builder.before(ts);
        }
        if let Some(d) = self.activation_depth {
            builder = builder.depth(d);
        }
        if let Some(w) = self.weights {
            builder = builder.weights(w);
        }
        if let Some(fmt) = self.format {
            builder = builder.format(fmt);
        }
        if let Some(cfg) = self.context_config {
            builder = builder.context_config(cfg);
        }
        match self.layer_filter {
            LayerFilter::EpisodicOnly => builder = builder.episodic_only(),
            LayerFilter::SemanticOnly => builder = builder.semantic_only(),
            // ThinkBuilder does not support procedural-only filtering.
            LayerFilter::ProceduralOnly | LayerFilter::All => {}
        }

        builder.execute().await
    }
}

/// Convert extracted entities into episodic entity refs.
fn extracted_to_refs(entities: &[ExtractedEntity]) -> Vec<EntityRef> {
    entities
        .iter()
        .map(|e| EntityRef {
            name: e.name.clone(),
            role: e.entity_type.clone(),
            entity_id: None,
        })
        .collect()
}

// ── hirn.toml loading ───────────────────────────────────────────────────

/// Merge a `hirn.toml` file on top of a base config.
///
/// Fields present in the TOML file override the corresponding default values;
/// fields absent in the file keep their defaults.  The merged result is
/// validated before returning.
fn load_toml_over_defaults(toml_path: &Path, base: HirnConfig) -> HirnResult<HirnConfig> {
    let content = std::fs::read_to_string(toml_path).map_err(|e| {
        HirnError::InvalidInput(format!("failed to read {}: {e}", toml_path.display()))
    })?;

    // Serialize baseline config into a TOML table.
    let mut base_table: toml::Table = toml::from_str(
        &toml::to_string(&base)
            .map_err(|e| HirnError::InvalidInput(format!("config serialization error: {e}")))?,
    )
    .map_err(|e| HirnError::InvalidInput(format!("config round-trip error: {e}")))?;

    // Parse the user's file into a table.
    let file_table: toml::Table = toml::from_str(&content).map_err(|e| {
        HirnError::InvalidInput(format!("invalid hirn.toml at {}: {e}", toml_path.display()))
    })?;

    // Merge: file values override base values.
    for (key, value) in file_table {
        base_table.insert(key, value);
    }

    // Deserialize back — this runs HirnConfig::validate() via TryFrom.
    let merged: HirnConfig = base_table.try_into().map_err(|e: toml::de::Error| {
        HirnError::InvalidInput(format!("invalid hirn.toml config: {e}"))
    })?;

    Ok(merged)
}
