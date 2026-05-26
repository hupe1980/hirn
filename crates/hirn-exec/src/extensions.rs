//! Runtime state for DataFusion operators.
//!
//! [`HirnSessionExt`] carries shared references — graph store, config, and
//! an embedder — that operators retrieve at execution time via DataFusion's
//! `SessionContext` extension mechanism.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::RecordBatch;
use async_trait::async_trait;
use datafusion::prelude::SessionContext;
use datafusion_common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use hirn_core::HirnResult;
use hirn_core::config::HirnConfig;
use hirn_core::embed::Embedder;
use hirn_core::id::MemoryId;
use hirn_core::tokenizer::Tokenizer;
use hirn_core::types::{EdgeRelation, Namespace};
use hirn_graph::PprConfig;
use hirn_query::compiler::plan_compiler::SemanticTargetKindRepr;
use hirn_storage::PhysicalStore;
use hirn_storage::store::DistanceMetric;
use parking_lot::RwLock;

use crate::operators::ActivationMode;
use crate::operators::SearchNumericFilter;
use crate::operators::nli_contradiction::NliClassifier;

#[derive(Debug, Clone, PartialEq)]
pub struct GraphActivationOutput {
    pub ids: Vec<String>,
    pub scores: Vec<f32>,
    pub depths: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphCausalChainRow {
    pub chain_id: String,
    pub source_id: String,
    pub target_id: String,
    pub strength: f32,
    pub confidence: f32,
    pub evidence_count: u32,
    pub mechanism: Option<String>,
    pub depth: u32,
    pub chain_score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphTraverseRow {
    pub node_id: String,
    pub depth: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecallSearchBinding {
    pub query_vector: Vec<f32>,
    pub filter: Option<String>,
    pub limit: usize,
    pub metric: DistanceMetric,
    pub numeric_filters: Vec<SearchNumericFilter>,
    pub temporal_start_ms: Option<i64>,
    pub temporal_end_ms: Option<i64>,
    /// Enable dual-pass temporal expansion in `LanceHybridSearchExec`.
    pub temporal_expansion: bool,
}

#[async_trait]
pub trait GraphReadRuntime: Send + Sync {
    async fn activate_graph(
        &self,
        seeds: &[MemoryId],
        mode: ActivationMode,
        ppr_config: Option<&PprConfig>,
        max_depth: u32,
        epsilon: f32,
        inhibition_mu: f32,
        delegation_threshold: usize,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<GraphActivationOutput>;

    async fn causal_chain(
        &self,
        start_ids: &[MemoryId],
        max_depth: u32,
        confidence_threshold: f32,
        delegation_threshold: usize,
        relation: EdgeRelation,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphCausalChainRow>>;

    async fn traverse_graph(
        &self,
        start_ids: &[MemoryId],
        max_depth: u32,
        delegation_threshold: usize,
        relation_filter: Option<&[EdgeRelation]>,
        allowed_namespaces: Option<&[Namespace]>,
    ) -> HirnResult<Vec<GraphTraverseRow>>;
}

#[async_trait]
pub trait QueryReadRuntime: Send + Sync {
    async fn inspect_json(
        &self,
        target: &str,
        target_kind: SemanticTargetKindRepr,
        agent_id: &str,
        allowed_namespaces: Option<&[String]>,
    ) -> HirnResult<Vec<u8>>;

    async fn trace_json(
        &self,
        target: &str,
        target_kind: SemanticTargetKindRepr,
        agent_id: &str,
        allowed_namespaces: Option<&[String]>,
    ) -> HirnResult<Vec<u8>>;

    async fn explain_causes_json(
        &self,
        query: &str,
        depth: u32,
        namespace: Option<&str>,
        allowed_namespaces: Option<&[String]>,
    ) -> HirnResult<Vec<u8>>;

    async fn what_if_json(
        &self,
        intervention: &str,
        outcome: &str,
        namespace: Option<&str>,
        allowed_namespaces: Option<&[String]>,
    ) -> HirnResult<Vec<u8>>;

    async fn counterfactual_json(
        &self,
        antecedent: &str,
        consequent: &str,
        namespace: Option<&str>,
        allowed_namespaces: Option<&[String]>,
    ) -> HirnResult<Vec<u8>>;

    async fn show_policies_json(
        &self,
        principal_kind: Option<&str>,
        principal_name: Option<&str>,
    ) -> HirnResult<Vec<u8>>;

    async fn explain_policy_json(
        &self,
        principal_kind: &str,
        principal_name: &str,
        resource_type: &str,
        resource_name: &str,
        action: &str,
    ) -> HirnResult<Vec<u8>>;
}

static QUERY_READ_RUNTIME_IDS: AtomicU64 = AtomicU64::new(1);

fn query_read_runtime_registry() -> &'static RwLock<HashMap<u64, Arc<dyn QueryReadRuntime>>> {
    static REGISTRY: OnceLock<RwLock<HashMap<u64, Arc<dyn QueryReadRuntime>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

#[derive(Debug)]
pub struct RegisteredQueryReadRuntime {
    id: u64,
}

impl RegisteredQueryReadRuntime {
    pub fn key(&self) -> String {
        self.id.to_string()
    }
}

impl Drop for RegisteredQueryReadRuntime {
    fn drop(&mut self) {
        query_read_runtime_registry().write().remove(&self.id);
    }
}

pub fn register_query_read_runtime(
    runtime: Arc<dyn QueryReadRuntime>,
) -> RegisteredQueryReadRuntime {
    let id = QUERY_READ_RUNTIME_IDS.fetch_add(1, Ordering::Relaxed);
    query_read_runtime_registry().write().insert(id, runtime);
    RegisteredQueryReadRuntime { id }
}

fn lookup_query_read_runtime(key: &str) -> Option<Arc<dyn QueryReadRuntime>> {
    let id = key.parse::<u64>().ok()?;
    query_read_runtime_registry().read().get(&id).cloned()
}

// ── ContextAssemblyRuntime ─────────────────────────────────────────────

/// Per-query runtime bridge for the THINK context assembly operator.
///
/// Registered once per THINK query execution (with actor identity, config,
/// and recall context captured at registration time), then looked up by
/// key inside `ContextAssemblyExec::execute()`.
///
/// The implementation in `hirn-engine` calls `assemble_think_context` and
/// JSON-serialises the full `ThinkAssemblyOutput` (including decoded
/// `ScoredMemory` records) so the operator can return a single opaque row.
#[async_trait]
pub trait ContextAssemblyRuntime: Send + Sync {
    /// Assemble context from scored candidate batches.
    ///
    /// Receives the raw Arrow output from `ContextBudgetExec` (or `McfaDefenseExec`
    /// if MCFA defense is enabled).  Returns opaque JSON bytes that the engine
    /// decodes into a fully-hydrated `ThinkAssemblyOutput`.
    async fn assemble_from_batches(
        &self,
        candidate_batches: Vec<RecordBatch>,
    ) -> HirnResult<Vec<u8>>;
}

static CONTEXT_ASSEMBLY_RUNTIME_IDS: AtomicU64 = AtomicU64::new(1);

fn context_assembly_runtime_registry()
-> &'static RwLock<HashMap<u64, Arc<dyn ContextAssemblyRuntime>>> {
    static REGISTRY: OnceLock<RwLock<HashMap<u64, Arc<dyn ContextAssemblyRuntime>>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// RAII handle for a registered [`ContextAssemblyRuntime`].
///
/// Removes the runtime from the global registry on drop so resources are freed
/// as soon as the query scope exits.
#[derive(Debug)]
pub struct RegisteredContextAssemblyRuntime {
    id: u64,
}

impl RegisteredContextAssemblyRuntime {
    /// Opaque string key for injecting into [`HirnSessionExt`].
    pub fn key(&self) -> String {
        self.id.to_string()
    }
}

impl Drop for RegisteredContextAssemblyRuntime {
    fn drop(&mut self) {
        context_assembly_runtime_registry().write().remove(&self.id);
    }
}

/// Register a `ContextAssemblyRuntime` for the current query scope.
///
/// Returns a RAII guard; drop it after plan execution to clean up the registry.
pub fn register_context_assembly_runtime(
    runtime: Arc<dyn ContextAssemblyRuntime>,
) -> RegisteredContextAssemblyRuntime {
    let id = CONTEXT_ASSEMBLY_RUNTIME_IDS.fetch_add(1, Ordering::Relaxed);
    context_assembly_runtime_registry()
        .write()
        .insert(id, runtime);
    RegisteredContextAssemblyRuntime { id }
}

pub(crate) fn lookup_context_assembly_runtime(
    key: &str,
) -> Option<Arc<dyn ContextAssemblyRuntime>> {
    let id = key.parse::<u64>().ok()?;
    context_assembly_runtime_registry().read().get(&id).cloned()
}

/// Shared runtime state accessible by all hirn DataFusion operators.
///
/// Registered in [`SessionContext`] at database open time. Operators retrieve
/// it via [`HirnSessionExt::get`] — never through constructor injection.
#[derive(Clone)]
pub struct HirnSessionExt {
    /// Hot+cold two-tier graph (CachedGraphStore lives in hirn-engine;
    /// we store as `Arc<dyn Any + Send + Sync>` to avoid depending on
    /// hirn-engine from hirn-exec).
    graph: Arc<dyn Any + Send + Sync>,

    /// Authoritative graph read runtime.
    ///
    /// When present, graph-aware operators should prefer this contract over
    /// downcasting the raw hot graph handle so the engine can enforce a single
    /// hot-vs-cold delegation rule.
    graph_read_runtime: Option<Arc<dyn GraphReadRuntime>>,

    /// Scoring weights and database configuration.
    pub config: Arc<HirnConfig>,

    /// Embedding provider (optional — not all operators need it).
    embedder: Option<Arc<dyn Embedder>>,

    /// Storage backend (optional — operators needing vector search use it).
    storage: Option<Arc<dyn PhysicalStore>>,

    /// Authoritative tokenizer for token-aware budgeted operators.
    tokenizer: Option<Arc<dyn Tokenizer>>,

    /// Authenticated agent identity for the current session.
    /// Used by `PolicyPushdownRule` to identify the requesting agent.
    agent_id: Option<String>,

    /// Pre-resolved namespace access list from the policy engine.
    ///
    /// - `None` — open mode: no namespace filtering applied.
    /// - `Some(vec)` — restrict scans to the listed namespaces.
    ///   An empty vec means deny all access.
    ///
    /// Resolved once at session setup by evaluating Cedar policies.
    allowed_namespaces: Option<Vec<String>>,

    /// Query-scoped runtime handle for compiled terminal read commands.
    query_read_runtime_key: Option<String>,

    /// Query-scoped runtime handle for THINK context assembly.
    ///
    /// Set once per THINK query, looked up by `ContextAssemblyExec` at
    /// execution time to retrieve the per-query `ContextAssemblyRuntime`.
    context_assembly_runtime_key: Option<String>,

    /// Query-scoped recall/search bindings used by compiled search operators.
    recall_search_binding: Option<RecallSearchBinding>,

    /// Shared historical RPE population statistics (Welford's online algorithm).
    ///
    /// Seeded from `WriteRuntime` at session setup; updated by `RpeScoreExec`
    /// after each batch so that z-scores compare against the full historical
    /// distribution, not just the current write batch (N-H08).
    pub rpe_population_stats: Arc<RwLock<hirn_core::WelfordStats>>,

    /// NLI classifier for `InterferenceDetectorExec` Check 3.
    ///
    /// `None` — operator uses its own default (`HeuristicNliClassifier`).
    /// `Some(clf)` — operator uses the injected classifier, enabling ONNX upgrade
    /// without recompiling or changing `InterferenceConfig`.
    nli_classifier: Option<Arc<dyn NliClassifier>>,
}

// SAFETY: Arc fields are Send + Sync.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HirnSessionExt>();
};

#[allow(clippy::missing_fields_in_debug)] // rpe_population_stats and recall_search_binding intentionally omitted (lock + query-scoped)
impl std::fmt::Debug for HirnSessionExt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HirnSessionExt")
            .field("graph", &"<type-erased>")
            .field("has_graph_read_runtime", &self.graph_read_runtime.is_some())
            .field("config", &self.config)
            .field("has_embedder", &self.embedder.is_some())
            .field("has_storage", &self.storage.is_some())
            .field("has_tokenizer", &self.tokenizer.is_some())
            .field("agent_id", &self.agent_id)
            .field("allowed_namespaces", &self.allowed_namespaces)
            .field(
                "has_query_read_runtime",
                &self.query_read_runtime_key.is_some(),
            )
            .field(
                "has_context_assembly_runtime",
                &self.context_assembly_runtime_key.is_some(),
            )
            .field(
                "nli_classifier_backend",
                &self
                    .nli_classifier
                    .as_ref()
                    .map(|c| c.backend_name())
                    .unwrap_or("default"),
            )
            // rpe_population_stats and recall_search_binding omitted from Debug
            // (locking during format is undesirable; binding is query-scoped).
            .finish_non_exhaustive()
    }
}

impl HirnSessionExt {
    /// Create a new extension bundle.
    /// Create a new extension bundle.
    pub fn new(
        graph: Arc<dyn Any + Send + Sync>,
        config: Arc<HirnConfig>,
        embedder: Option<Arc<dyn Embedder>>,
    ) -> Self {
        Self {
            graph,
            graph_read_runtime: None,
            config,
            embedder,
            storage: None,
            tokenizer: None,
            agent_id: None,
            allowed_namespaces: None,
            query_read_runtime_key: None,
            context_assembly_runtime_key: None,
            recall_search_binding: None,
            rpe_population_stats: Arc::new(RwLock::new(hirn_core::WelfordStats::new())),
            nli_classifier: None,
        }
    }

    /// Seed historical RPE population statistics (from `WriteRuntime`).
    ///
    /// Allows `RpeScoreExec` to z-score against the full historical
    /// distribution instead of only the current write batch (N-H08).
    pub fn with_rpe_population_stats(
        mut self,
        stats: Arc<RwLock<hirn_core::WelfordStats>>,
    ) -> Self {
        self.rpe_population_stats = stats;
        self
    }

    /// Set the authenticated agent identity.
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Inject an NLI classifier for `InterferenceDetectorExec` Check 3.
    ///
    /// Use this to upgrade from the default `HeuristicNliClassifier` to a
    /// DeBERTa-MNLI ONNX model at database open time, without changing any
    /// `InterferenceConfig` fields.
    pub fn with_nli_classifier(mut self, clf: Arc<dyn NliClassifier>) -> Self {
        self.nli_classifier = Some(clf);
        self
    }

    /// Returns the injected NLI classifier, if any.
    ///
    /// `None` — `InterferenceDetectorExec` will use its own default.
    pub fn nli_classifier(&self) -> Option<Arc<dyn NliClassifier>> {
        self.nli_classifier.clone()
    }

    /// Set the storage backend for operators needing vector search.
    pub fn with_storage(mut self, storage: Arc<dyn PhysicalStore>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Set the authoritative graph read runtime.
    pub fn with_graph_read_runtime(
        mut self,
        graph_read_runtime: Arc<dyn GraphReadRuntime>,
    ) -> Self {
        self.graph_read_runtime = Some(graph_read_runtime);
        self
    }

    /// Set the tokenizer for operators needing authoritative token counts.
    pub fn with_tokenizer(mut self, tokenizer: Arc<dyn Tokenizer>) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Set pre-resolved allowed namespaces.
    ///
    /// `None` means open mode (no filtering). `Some(vec)` restricts to those namespaces.
    pub fn with_allowed_namespaces(mut self, namespaces: Option<Vec<String>>) -> Self {
        self.allowed_namespaces = namespaces;
        self
    }

    /// Set a query-scoped runtime handle for compiled terminal read operators.
    pub fn with_query_read_runtime_key(mut self, key: Option<String>) -> Self {
        self.query_read_runtime_key = key;
        self
    }

    /// Set a query-scoped runtime handle for THINK context assembly.
    pub fn with_context_assembly_runtime_key(mut self, key: Option<String>) -> Self {
        self.context_assembly_runtime_key = key;
        self
    }

    /// Set query-scoped compiled recall/search bindings.
    pub fn with_recall_search_binding(mut self, binding: Option<RecallSearchBinding>) -> Self {
        self.recall_search_binding = binding;
        self
    }

    /// Returns the agent ID, if set.
    pub fn agent_id(&self) -> Option<&str> {
        self.agent_id.as_deref()
    }

    /// Returns the pre-resolved allowed namespaces.
    ///
    /// `None` = open mode (no filtering), `Some(&[])` = deny all.
    pub fn allowed_namespaces(&self) -> Option<&[String]> {
        self.allowed_namespaces.as_deref()
    }

    /// Retrieve `HirnSessionExt` from a [`SessionContext`].
    ///
    /// Returns a clone since `SessionContext::state()` returns by value.
    ///
    /// # Errors
    /// Returns an error if the extension was never registered.
    pub fn get(ctx: &SessionContext) -> datafusion_common::Result<Self> {
        let state = ctx.state();
        let ext = state
            .config()
            .options()
            .extensions
            .get::<Self>()
            .ok_or_else(|| {
                datafusion_common::DataFusionError::Configuration(
                    "HirnSessionExt not registered in SessionContext — \
                     was the database opened correctly?"
                        .into(),
                )
            })?;
        Ok(ext.clone())
    }

    /// Register this extension in a [`SessionContext`].
    ///
    /// # Errors
    /// Returns an error if the `SessionState` has already been dropped.
    pub fn register(self, ctx: &SessionContext) -> datafusion_common::Result<()> {
        let state = ctx.state_weak_ref().upgrade().ok_or_else(|| {
            datafusion_common::DataFusionError::Internal(
                "Cannot register HirnSessionExt: SessionState already dropped".into(),
            )
        })?;
        state
            .write()
            .config_mut()
            .options_mut()
            .extensions
            .insert(self);
        Ok(())
    }

    /// Downcast the type-erased graph handle to the concrete type `T`.
    ///
    /// Returns `None` if the stored graph is not of type `T`.
    pub fn graph_as<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.graph.downcast_ref::<T>()
    }

    /// Clone the graph `Arc` and downcast to `Arc<T>`.
    ///
    /// Returns `None` if the stored graph is not of type `T`.
    pub fn graph_arc<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.graph.clone().downcast::<T>().ok()
    }

    /// Raw `Arc<dyn Any>` graph handle.
    pub fn graph_any(&self) -> &Arc<dyn Any + Send + Sync> {
        &self.graph
    }

    /// Optional graph read runtime.
    pub fn graph_read_runtime(&self) -> Option<Arc<dyn GraphReadRuntime>> {
        self.graph_read_runtime.clone()
    }

    /// Optional terminal-read runtime resolved from the query-scoped registry.
    pub fn query_read_runtime(&self) -> Option<Arc<dyn QueryReadRuntime>> {
        self.query_read_runtime_key
            .as_deref()
            .and_then(lookup_query_read_runtime)
    }

    /// Optional context-assembly runtime resolved from the query-scoped registry.
    pub fn context_assembly_runtime(&self) -> Option<Arc<dyn ContextAssemblyRuntime>> {
        self.context_assembly_runtime_key
            .as_deref()
            .and_then(lookup_context_assembly_runtime)
    }

    /// Optional query-scoped compiled recall/search binding.
    pub fn recall_search_binding(&self) -> Option<&RecallSearchBinding> {
        self.recall_search_binding.as_ref()
    }

    /// Optional embedder reference.
    pub fn embedder(&self) -> Option<&dyn Embedder> {
        self.embedder.as_deref()
    }

    /// Optional embedder Arc clone.
    pub fn embedder_arc(&self) -> Option<Arc<dyn Embedder>> {
        self.embedder.clone()
    }

    /// Optional storage reference.
    pub fn storage(&self) -> Option<&dyn PhysicalStore> {
        self.storage.as_deref()
    }

    /// Optional storage Arc clone.
    pub fn storage_arc(&self) -> Option<Arc<dyn PhysicalStore>> {
        self.storage.clone()
    }

    /// Optional tokenizer reference.
    pub fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        self.tokenizer.as_deref()
    }

    /// Optional tokenizer Arc clone.
    pub fn tokenizer_arc(&self) -> Option<Arc<dyn Tokenizer>> {
        self.tokenizer.clone()
    }
}

impl ExtensionOptions for HirnSessionExt {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, _key: &str, _value: &str) -> datafusion_common::Result<()> {
        Ok(())
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        vec![]
    }
}

impl ConfigExtension for HirnSessionExt {
    const PREFIX: &'static str = "hirn";
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time assertion: `HirnSessionExt` is `Send + Sync`.
    const _: fn() = || {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HirnSessionExt>();
    };

    #[test]
    fn register_and_retrieve() {
        let ctx = SessionContext::new();
        let config = Arc::new(HirnConfig::default());
        let ext = HirnSessionExt::new(Arc::new(42_u32), config.clone(), None);
        ext.register(&ctx).expect("register should succeed");

        let retrieved = HirnSessionExt::get(&ctx).expect("extension should be present");
        // Same Arc — pointer equality proves we got the same config back.
        assert!(Arc::ptr_eq(&retrieved.config, &config));
        assert!(retrieved.embedder().is_none());
        assert!(retrieved.tokenizer().is_none());
    }

    #[test]
    fn missing_extension_gives_clear_error() {
        let ctx = SessionContext::new();
        let err = HirnSessionExt::get(&ctx).unwrap_err();
        assert!(
            err.to_string().contains("HirnSessionExt not registered"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn graph_downcast() {
        let ctx = SessionContext::new();
        let ext = HirnSessionExt::new(
            Arc::new(String::from("test_graph")),
            Arc::new(HirnConfig::default()),
            None,
        );
        ext.register(&ctx).expect("register should succeed");

        let retrieved = HirnSessionExt::get(&ctx).unwrap();
        let graph = retrieved.graph_as::<String>().unwrap();
        assert_eq!(graph, "test_graph");
    }
}
