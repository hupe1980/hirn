//! Cognitive benchmark runner — ingests sessions, queries via think/recall, scores results.

use async_trait::async_trait;
use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use hirn::activation::ActivationMode;
use hirn::agent::{AgentId, Namespace};
use hirn::episodic::{EpisodicRecord, EventType};
use hirn::graph::EdgeRelation;
use hirn::metadata::{Metadata, MetadataValue};
use hirn::ql::{QueryResult as QlQueryResult, ast as ql_ast};
use hirn::record::Layer;
use hirn::{HirnConfig, HirnDB, MemoryId, Timestamp};
use hirn_core::embed::{Embedder, Embedding};
use hirn_core::error::{HirnError, HirnResult};
use hirn_core::types::Origin;
use hirn_engine::ProviderRegistry;
use hirn_engine::retrieval::recall::RecallBuilder;
use hirn_engine::{QueryDiagnostics, ThinkBuilder};
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

use crate::dataset::pseudo_embedding;
use crate::metrics::latency_percentiles;

/// Run an async future to completion on a shared tokio runtime.
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    static RT: std::sync::LazyLock<tokio::runtime::Runtime> = std::sync::LazyLock::new(|| {
        tokio::runtime::Runtime::new().expect("tokio runtime for cognitive bench")
    });
    RT.block_on(f)
}

#[track_caller]
fn require_query_stage<T, E>(query_id: &str, stage: &str, result: Result<T, E>) -> T
where
    E: std::fmt::Display,
{
    match result {
        Ok(value) => value,
        Err(error) => panic!("benchmark {stage} failed for query `{query_id}`: {error}"),
    }
}

#[track_caller]
fn require_query_clean_diagnostics(query_id: &str, stage: &str, diagnostics: &QueryDiagnostics) {
    if let Some(summary) = diagnostics.advanced_retrieval_fallback_summary() {
        panic!("benchmark {stage} used retrieval fallback for query `{query_id}`: {summary}");
    }
}

fn benchmark_turn_importance(turn: &super::Turn) -> f32 {
    let mut importance = 0.55_f32;

    if turn.speaker == "SessionSummary" {
        importance = 0.85;
    } else if turn.speaker.starts_with("Observation/") {
        importance = 0.75;
    }

    if turn.timestamp.is_some() || turn.timestamp_text.as_deref().is_some() {
        importance += 0.05;
    }
    if turn.source_id.is_some() {
        importance += 0.03;
    }

    importance.clamp(0.0, 1.0)
}

fn benchmark_turn_origin(turn: &super::Turn) -> Origin {
    if turn.speaker == "SessionSummary" || turn.speaker.starts_with("Observation/") {
        Origin::LlmExtraction
    } else {
        Origin::DirectObservation
    }
}

use super::eval;
use super::openai::EmbeddingCache;
use super::{
    ActiveRetrievalSurfaces, BaselineStrategy, Benchmark, BenchmarkExecutionSurface,
    BenchmarkRetrievalProfile, CategoryScore, CognitiveConfig, CognitiveDataset, CognitiveResult,
    CompiledPhaseTimingSummary, EmbedderPolicy, MetricDrift, QueryEmbeddingSource, QueryScore,
    ReproducibilitySummary, TokenCostEstimate, render_turn_content,
};

const HIRN_STRATEGY: &str = "hirn";
const ITERATIVE_BASELINE_HOPS: usize = 3;
const INGEST_BATCH_SIZE_SMALL: usize = 100;
const INGEST_BATCH_SIZE_LARGE: usize = 2_000;
const INGEST_BATCH_SIZE_HUGE: usize = 5_000;
const COMPACTION_TARGET_ROWS_PER_FRAGMENT: usize = 4_096;
const QUERY_PROGRESS_INTERVAL: usize = 100;
const BENCH_SOURCE_ID_META_KEY: &str = "hirn_bench_source_id";
const BENCH_THINK_CANDIDATE_LIMIT: usize = 50;
const BENCHMARK_CACHE_EMBEDDER_MODEL_ID: &str = "benchmark-cache";
const BENCHMARK_PROVIDER_EMBED_BATCH_SIZE: usize = 100;

fn ingest_batch_size(total_records: usize) -> usize {
    if total_records >= 250_000 {
        INGEST_BATCH_SIZE_HUGE
    } else if total_records >= 50_000 {
        INGEST_BATCH_SIZE_LARGE
    } else {
        INGEST_BATCH_SIZE_SMALL
    }
}

fn ingest_progress_every_batches(total_records: usize) -> usize {
    if total_records >= 250_000 {
        25
    } else if total_records >= 50_000 {
        10
    } else {
        1
    }
}

fn should_log_ingest_progress(
    flushed_batches: usize,
    total_flushed_records: usize,
    total_records: usize,
    progress_interval: usize,
) -> bool {
    total_flushed_records == total_records
        || flushed_batches <= 3
        || flushed_batches.is_multiple_of(progress_interval)
}

#[derive(Debug)]
struct BenchmarkCacheEmbedder {
    cache: Arc<EmbeddingCache>,
    dims: usize,
}

impl BenchmarkCacheEmbedder {
    fn new(cache: Arc<EmbeddingCache>, dims: usize) -> Self {
        Self { cache, dims }
    }

    fn embedding_for_text(&self, text: &str) -> HirnResult<Vec<f32>> {
        let embedding = self.cache.get(text).ok_or_else(|| {
            HirnError::InvalidInput(format!(
                "benchmark embedding cache missing text: {}",
                truncate_benchmark_text_for_error(text)
            ))
        })?;

        if embedding.len() != self.dims {
            return Err(HirnError::InvalidInput(format!(
                "benchmark embedding cache dimension mismatch for text {}: expected {}, got {}",
                truncate_benchmark_text_for_error(text),
                self.dims,
                embedding.len()
            )));
        }

        Ok(embedding.clone())
    }
}

#[async_trait]
impl Embedder for BenchmarkCacheEmbedder {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        texts
            .iter()
            .map(|text| {
                self.embedding_for_text(text).map(|vector| Embedding {
                    vector,
                    model_id: BENCHMARK_CACHE_EMBEDDER_MODEL_ID.to_string(),
                })
            })
            .collect()
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model_id(&self) -> &str {
        BENCHMARK_CACHE_EMBEDDER_MODEL_ID
    }

    fn max_input_tokens(&self) -> usize {
        usize::MAX
    }
}

fn truncate_benchmark_text_for_error(text: &str) -> String {
    const MAX_CHARS: usize = 80;

    let mut truncated = text.chars().take(MAX_CHARS).collect::<String>();
    if text.chars().count() > MAX_CHARS {
        truncated.push_str("...");
    }
    truncated
}

#[derive(Default)]
struct PendingIngestBatch {
    keys: Vec<(String, usize)>,
    records: Vec<EpisodicRecord>,
}

fn flush_ingest_batch(
    db: &HirnDB,
    batch: &mut PendingIngestBatch,
    memory_ids: &mut HashMap<(String, usize), MemoryId>,
) -> usize {
    if batch.records.is_empty() {
        return 0;
    }

    let keys = std::mem::take(&mut batch.keys);
    let records = std::mem::take(&mut batch.records);
    let flushed = keys.len();
    let results = block_on(db.episodic().batch_remember(records));

    for (key, result) in keys.into_iter().zip(results) {
        let mid = result.expect("remember episode");
        memory_ids.insert(key, mid);
    }

    flushed
}

fn compact_benchmark_storage(db: &HirnDB) {
    let compaction_start = Instant::now();
    let compaction = block_on(
        db.admin()
            .lifecycle_compact()
            .skip_consolidation()
            .skip_archival()
            .target_rows_per_fragment(COMPACTION_TARGET_ROWS_PER_FRAGMENT)
            .slow_threshold_secs(5)
            .agent_id("system")
            .execute(),
    )
    .expect("compact benchmark datasets");
    let compaction_elapsed = compaction_start.elapsed();

    eprintln!(
        "  compaction: {} datasets, -{} +{} fragments in {:.2}s",
        compaction.datasets_compacted,
        compaction.fragments_removed,
        compaction.fragments_added,
        compaction_elapsed.as_secs_f64(),
    );
}

#[derive(Debug, Default)]
struct StrategyRunData {
    query_scores: Vec<QueryScore>,
    execution_latencies: Vec<Duration>,
    evaluation_latencies: Vec<Duration>,
    end_to_end_latencies: Vec<Duration>,
    compiled_optimize_latencies: Vec<Duration>,
    compiled_physical_plan_latencies: Vec<Duration>,
    compiled_execute_plan_latencies: Vec<Duration>,
    compiled_embed_latencies: Vec<Duration>,
    /// Secondary record-hydration latencies (THINK decode phase, Lance I/O).
    compiled_decode_latencies: Vec<Duration>,
    compiled_assemble_latencies: Vec<Duration>,
    compiled_total_latencies: Vec<Duration>,
    context_tokens: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
struct CompiledPhaseSample {
    optimize: Duration,
    physical_plan: Duration,
    execute_plan: Duration,
    embed: Duration,
    /// Secondary record hydration (Lance I/O) — only set by THINK queries.
    decode: Duration,
    assemble: Duration,
    total: Duration,
}

#[derive(Debug, Clone)]
struct QueryExecution {
    context: String,
    ranked_results: Vec<RetrievedCandidate>,
    context_tokens: usize,
    compiled_phase_sample: Option<CompiledPhaseSample>,
}

#[derive(Debug, Clone)]
pub struct CognitiveRunReport {
    pub result: CognitiveResult,
    pub active_retrieval_surfaces: ActiveRetrievalSurfaces,
    pub query_embedding_source: QueryEmbeddingSource,
    pub query_embedding_model_label: Option<String>,
}

#[derive(Debug, Clone)]
struct BenchmarkRetrievalSetup {
    active_retrieval_surfaces: ActiveRetrievalSurfaces,
    query_embedding_source: QueryEmbeddingSource,
    query_embedding_model_label: Option<String>,
}

#[derive(Clone)]
pub struct BenchmarkEmbeddingRuntime {
    lookup: Option<Arc<EmbeddingCache>>,
    runtime_embedder: Option<Arc<dyn Embedder>>,
    source: QueryEmbeddingSource,
    model_label: Option<String>,
    strict_lookup: bool,
    /// Counts how many times resolve_embedding fell back to pseudo_embedding (N-L11).
    pub pseudo_miss_count: Arc<AtomicU64>,
}

impl BenchmarkEmbeddingRuntime {
    fn cache_backed(cache: Arc<EmbeddingCache>, dims: usize) -> Self {
        let runtime_embedder =
            Arc::new(BenchmarkCacheEmbedder::new(Arc::clone(&cache), dims)) as Arc<dyn Embedder>;

        Self {
            lookup: Some(cache),
            runtime_embedder: Some(runtime_embedder),
            source: QueryEmbeddingSource::Cache,
            model_label: None,
            strict_lookup: false,
            pseudo_miss_count: Arc::new(AtomicU64::new(0)),
        }
    }

    fn provider_backed(cache: Arc<EmbeddingCache>, dims: usize, model_label: String) -> Self {
        let runtime_embedder =
            Arc::new(BenchmarkCacheEmbedder::new(Arc::clone(&cache), dims)) as Arc<dyn Embedder>;

        Self {
            lookup: Some(cache),
            runtime_embedder: Some(runtime_embedder),
            source: QueryEmbeddingSource::Provider,
            model_label: Some(model_label),
            strict_lookup: true,
            pseudo_miss_count: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn pseudo() -> Self {
        Self {
            lookup: None,
            runtime_embedder: None,
            source: QueryEmbeddingSource::Pseudo,
            model_label: None,
            strict_lookup: false,
            pseudo_miss_count: Arc::new(AtomicU64::new(0)),
        }
    }

    fn runtime_embedder(&self) -> Option<Arc<dyn Embedder>> {
        self.runtime_embedder.clone()
    }

    fn query_embedding_source(&self) -> QueryEmbeddingSource {
        self.source
    }

    fn query_embedding_model_label(&self) -> Option<String> {
        self.model_label.clone()
    }

    fn resolve_embedding(&self, text: &str, dims: usize) -> Vec<f32> {
        if let Some(lookup) = &self.lookup {
            if let Some(embedding) = lookup.get(text) {
                return embedding.clone();
            }

            assert!(
                !self.strict_lookup,
                "benchmark provider embedding runtime missing text: {}",
                truncate_benchmark_text_for_error(text)
            );
        }

        // N-L11: Count pseudo-embedding fallbacks so callers can detect degraded runs.
        self.pseudo_miss_count.fetch_add(1, Ordering::Relaxed);
        pseudo_embedding(text, dims)
    }
}

fn provider_backed_benchmark_texts(dataset: &CognitiveDataset) -> BTreeSet<String> {
    let mut texts = super::dataset_embedding_texts(dataset);

    for query in &dataset.queries {
        texts.extend(query.expected_answers.iter().cloned());
    }

    texts
}

fn build_provider_backed_embedding_cache(
    dataset: &CognitiveDataset,
    dims: usize,
    embedder: Arc<dyn Embedder>,
) -> HirnResult<EmbeddingCache> {
    if embedder.dimensions() != dims {
        return Err(HirnError::InvalidInput(format!(
            "benchmark provider embedder dimension mismatch: expected {}, got {}",
            dims,
            embedder.dimensions()
        )));
    }

    let texts = provider_backed_benchmark_texts(dataset);
    let text_list: Vec<String> = texts.into_iter().collect();
    let mut cache = EmbeddingCache::new();

    for chunk in text_list.chunks(BENCHMARK_PROVIDER_EMBED_BATCH_SIZE) {
        let text_refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        let embeddings = block_on(embedder.embed(&text_refs))?;

        if embeddings.len() != text_refs.len() {
            return Err(HirnError::InvalidInput(format!(
                "benchmark provider embedder returned {} embeddings for {} texts",
                embeddings.len(),
                text_refs.len()
            )));
        }

        for (text, embedding) in chunk.iter().zip(embeddings) {
            if embedding.vector.len() != dims {
                return Err(HirnError::InvalidInput(format!(
                    "benchmark provider embedding dimension mismatch for text {}: expected {}, got {}",
                    truncate_benchmark_text_for_error(text),
                    dims,
                    embedding.vector.len()
                )));
            }
            cache.insert(text.clone(), embedding.vector);
        }
    }

    Ok(cache)
}

fn provider_backed_benchmark_runtime(
    dataset: &CognitiveDataset,
    dims: usize,
    embedder: Arc<dyn Embedder>,
) -> HirnResult<BenchmarkEmbeddingRuntime> {
    let model_label = embedder.model_id().to_string();
    let cache = Arc::new(build_provider_backed_embedding_cache(
        dataset, dims, embedder,
    )?);

    Ok(BenchmarkEmbeddingRuntime::provider_backed(
        cache,
        dims,
        model_label,
    ))
}

/// Prepare the embedding runtime for a benchmark run.
///
/// Returns `Ok(Some(runtime))` when a real or pseudo embedder is ready.
/// Returns `Ok(None)` when `config.embedder_policy` is `SkipIfAbsent` and no real
/// embedder or cache is available — callers should treat this as a skipped run.
/// Returns `Err(_)` when `config.embedder_policy` is `RealRequired` and no real
/// embedder or cache is available.
pub fn prepare_benchmark_embedding_runtime(
    dataset: &CognitiveDataset,
    config: &CognitiveConfig,
    embeddings: Option<&EmbeddingCache>,
) -> HirnResult<Option<BenchmarkEmbeddingRuntime>> {
    // A precomputed cache always takes priority regardless of policy.
    if let Some(cache) = embeddings {
        return Ok(Some(BenchmarkEmbeddingRuntime::cache_backed(
            Arc::new(cache.clone()),
            config.embedding_dims,
        )));
    }

    // No cache — check if a live provider is reachable.
    let registry = ProviderRegistry::from_env_strict();
    if let Some(embedder) = registry.embedder() {
        return provider_backed_benchmark_runtime(dataset, config.embedding_dims, embedder)
            .map(Some);
    }

    // No real embedder available — apply the configured policy.
    match config.embedder_policy {
        EmbedderPolicy::PseudoFallback => Ok(Some(BenchmarkEmbeddingRuntime::pseudo())),
        EmbedderPolicy::SkipIfAbsent => {
            tracing::warn!(
                "benchmark skipped: no real embedder or cache available \
                 and embedder_policy=skip-if-absent"
            );
            Ok(None)
        }
        EmbedderPolicy::RealRequired => Err(hirn_core::HirnError::config(
            "no real embedder or cache available but embedder_policy=real-required; \
             set HIRN_OPENAI_API_KEY, HIRN_VOYAGE_API_KEY, or similar",
        )),
    }
}

#[derive(Debug, Clone)]
struct ContextDocument {
    content: String,
    token_count: usize,
    lexical_terms: BTreeSet<String>,
    source_id: Option<String>,
}

#[derive(Debug, Clone)]
struct RetrievedCandidate {
    content: String,
    source_id: Option<String>,
}

/// Run a single cognitive benchmark and return scored results.
///
/// When `embeddings` is `Some`, uses precomputed OpenAI embeddings for real
/// semantic similarity. Falls back to `pseudo_embedding` when `None`.
pub fn run(
    dataset: &CognitiveDataset,
    config: &CognitiveConfig,
    db_path: &Path,
    run_id: &str,
) -> CognitiveResult {
    run_with_embeddings(dataset, config, db_path, run_id, None)
        .expect("run() called with no embeddings must not skip (policy should be PseudoFallback)")
        .result
}

fn benchmark_hirn_config(config: &CognitiveConfig, db_path: &Path) -> HirnConfig {
    let mut builder = HirnConfig::builder()
        .db_path(db_path)
        .embedding_dimensions(config.embedding_dims as u32)
        .token_budget(config.token_budget as u32);

    if matches!(config.retrieval_profile, BenchmarkRetrievalProfile::Minimal) {
        builder = builder.quality_gate_threshold(0.0);
    } else {
        builder = builder.multivector_enabled(true).multivector_weight(0.3);
    }

    builder.build().expect("valid HirnConfig")
}

fn configure_benchmark_retrieval(
    db: &mut HirnDB,
    config: &CognitiveConfig,
    embedding_runtime: &BenchmarkEmbeddingRuntime,
) -> BenchmarkRetrievalSetup {
    let mut surfaces = ActiveRetrievalSurfaces {
        query_text_hybrid: config.effective_query_text_hybrid(),
        graph_routing: true,
        compiled_hirnql: matches!(
            config.execution_surface,
            BenchmarkExecutionSurface::CompiledHirnql
        ),
        quality_gate: matches!(
            (config.execution_surface, config.retrieval_profile),
            (
                BenchmarkExecutionSurface::CompiledHirnql,
                BenchmarkRetrievalProfile::NormalFullStack | BenchmarkRetrievalProfile::Ablation
            )
        ),
        iterative_retrieval: false,
        ..ActiveRetrievalSurfaces::default()
    };
    let mut query_embedding_source = embedding_runtime.query_embedding_source();
    let mut query_embedding_model_label = embedding_runtime.query_embedding_model_label();

    match config.execution_surface {
        BenchmarkExecutionSurface::DirectBuilders => surfaces.notes.push(
            "compiled_hirnql=false, quality_gate=false, iterative_retrieval=false because the benchmark still executes direct THINK/RECALL builders"
                .to_string(),
        ),
        BenchmarkExecutionSurface::CompiledHirnql => {
            let note = if matches!(config.retrieval_profile, BenchmarkRetrievalProfile::Minimal) {
                "compiled_hirnql=true via plain THINK/RECALL execution with diagnostics; quality_gate=false for benchmark minimal-profile parity, while iterative retrieval remains off because benchmark queries use local THINK mode"
            } else {
                "compiled_hirnql=true via plain THINK/RECALL execution with diagnostics; quality_gate reflects the compiled read path, while iterative retrieval remains off because benchmark queries use local THINK mode"
            };
            surfaces.notes.push(note.to_string());
        }
    }

    let runtime_embedder = embedding_runtime.runtime_embedder();

    if let Some(embedder) = runtime_embedder.clone() {
        db.set_embedder(embedder);
        match embedding_runtime.query_embedding_source() {
            QueryEmbeddingSource::Cache => surfaces.notes.push(
                "cache-backed benchmark embedder installed for query-time parity with ingest"
                    .to_string(),
            ),
            QueryEmbeddingSource::Provider => surfaces.notes.push(
                "provider-backed benchmark embedder installed for ingest/query parity without a precomputed cache"
                    .to_string(),
            ),
            QueryEmbeddingSource::Pseudo => {}
        }
    } else if matches!(
        embedding_runtime.query_embedding_source(),
        QueryEmbeddingSource::Pseudo
    ) {
        // N-L11: Pseudo embeddings degrade benchmark scores. Make the fallback
        // explicit in the notes so it is visible in both stderr and the JSON output.
        let warn = "WARNING: using hash-based pseudo-embeddings (no real embedding provider or cache found) \
            — scores will be LOWER than with real embeddings; do not publish these results as-is";
        eprintln!("[hirn-bench] {warn}");
        surfaces.notes.push(warn.to_string());
    }

    if matches!(config.retrieval_profile, BenchmarkRetrievalProfile::Minimal) {
        surfaces
            .notes
            .push("minimal profile keeps provider-backed retrieval extras disabled".to_string());
        return BenchmarkRetrievalSetup {
            active_retrieval_surfaces: surfaces,
            query_embedding_source,
            query_embedding_model_label,
        };
    }

    let registry = ProviderRegistry::from_env_strict();

    if let Some(tokenizer) = registry.tokenizer() {
        db.set_tokenizer(tokenizer);
        surfaces.tokenizer = true;
    }

    if let Some(embedder) = registry.embedder() {
        if runtime_embedder.is_none() {
            if matches!(
                config.execution_surface,
                BenchmarkExecutionSurface::CompiledHirnql
            ) {
                query_embedding_source = QueryEmbeddingSource::Provider;
                query_embedding_model_label = Some(embedder.model_id().to_string());
            }
            db.set_embedder(embedder.clone());
        }
        if embedder.supports_multivec() {
            db.set_multivec_embedder(embedder);
            surfaces.multivector = true;
        } else {
            surfaces
                .notes
                .push("default embedder does not support multivector late interaction".to_string());
        }
    } else {
        surfaces
            .notes
            .push("no provider embedder discovered from environment".to_string());
    }

    if let Some(reranker) = registry.reranker() {
        db.set_reranker(reranker);
        surfaces.reranker = true;
    } else {
        surfaces
            .notes
            .push("no provider reranker discovered from environment".to_string());
    }

    if matches!(
        config.retrieval_profile,
        BenchmarkRetrievalProfile::NormalFullStack
    ) {
        require_full_stack_surfaces(&surfaces);
    }

    BenchmarkRetrievalSetup {
        active_retrieval_surfaces: surfaces,
        query_embedding_source,
        query_embedding_model_label,
    }
}

#[track_caller]
fn require_full_stack_surfaces(surfaces: &ActiveRetrievalSurfaces) {
    let mut missing = Vec::new();

    if !surfaces.query_text_hybrid {
        missing.push("hybrid".to_string());
    }
    if !surfaces.graph_routing {
        missing.push("graph".to_string());
    }
    if !surfaces.multivector {
        missing.push("multivector".to_string());
    }
    if !surfaces.reranker {
        missing.push("reranker".to_string());
    }

    if !missing.is_empty() {
        let mut detail = format!(
            "normal-full-stack benchmark profile requires provider-backed retrieval surfaces: {}",
            missing.join(", ")
        );
        if !surfaces.notes.is_empty() {
            detail.push_str("; ");
            detail.push_str(&surfaces.notes.join("; "));
        }
        panic!("{detail}");
    }
}

/// Run with optional precomputed embeddings.
///
/// Returns `None` when `config.embedder_policy` is `SkipIfAbsent` and no real
/// embedder or cache is available.
pub fn run_with_embeddings(
    dataset: &CognitiveDataset,
    config: &CognitiveConfig,
    db_path: &Path,
    run_id: &str,
    embeddings: Option<&EmbeddingCache>,
) -> Option<CognitiveRunReport> {
    let embedding_runtime =
        prepare_benchmark_embedding_runtime(dataset, config, embeddings)
            .unwrap_or_else(|error| panic!("prepare benchmark embeddings: {error}"))?;

    Some(run_with_prepared_embeddings(dataset, config, db_path, run_id, &embedding_runtime))
}

pub fn run_with_prepared_embeddings(
    dataset: &CognitiveDataset,
    config: &CognitiveConfig,
    db_path: &Path,
    run_id: &str,
    embedding_runtime: &BenchmarkEmbeddingRuntime,
) -> CognitiveRunReport {
    let total_start = Instant::now();

    // Open database with LanceDB storage backend.
    let lance_path = db_path.parent().unwrap_or(db_path).join("lance_brain");
    let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
    let backend: Arc<dyn PhysicalStore> = block_on(HirnDb::open(storage_config))
        .expect("open HirnDb")
        .store_arc();

    let hirn_config = benchmark_hirn_config(config, db_path);
    let mut db = block_on(HirnDB::open_with_config(hirn_config, backend)).expect("open HirnDB");
    let retrieval_setup = configure_benchmark_retrieval(&mut db, config, embedding_runtime);

    // Phase 1: Ingest all sessions as episodic records.
    let ingest_start = Instant::now();
    let memory_ids = ingest_sessions(&db, dataset, config.embedding_dims, embedding_runtime);
    let ingest_time = ingest_start.elapsed();

    // Phase 1b: Create graph edges for suites that need them.
    if dataset.benchmark == Benchmark::H3Graph {
        create_h3_causal_edges(&db, dataset, &memory_ids);
    }
    if dataset.benchmark == Benchmark::H5Action {
        create_h5_action_edges(&db, &memory_ids);
    }
    if dataset.benchmark == Benchmark::H6Safety {
        create_h6_conflict_edges(&db, &memory_ids);
    }

    compact_benchmark_storage(&db);

    // Phase 2: Run queries and score.
    let query_start = Instant::now();
    let query_results = evaluate_queries(&db, dataset, config, embedding_runtime);
    let query_time = query_start.elapsed();

    let total_time = total_start.elapsed();

    CognitiveRunReport {
        result: finalize_result(
            dataset,
            HIRN_STRATEGY,
            run_id,
            query_results,
            ingest_time,
            query_time,
            total_time,
            embedding_runtime.pseudo_miss_count.load(Ordering::Relaxed),
        ),
        active_retrieval_surfaces: retrieval_setup.active_retrieval_surfaces,
        query_embedding_source: retrieval_setup.query_embedding_source,
        query_embedding_model_label: retrieval_setup.query_embedding_model_label,
    }
}

/// Run an executable reference baseline with the same scoring surface as hirn.
///
/// Returns `None` when `config.embedder_policy` is `SkipIfAbsent` and no real
/// embedder or cache is available.
pub fn run_baseline_with_embeddings(
    dataset: &CognitiveDataset,
    config: &CognitiveConfig,
    run_id: &str,
    strategy: BaselineStrategy,
    embeddings: Option<&EmbeddingCache>,
) -> Option<CognitiveResult> {
    let embedding_runtime =
        prepare_benchmark_embedding_runtime(dataset, config, embeddings)
            .unwrap_or_else(|error| panic!("prepare benchmark embeddings: {error}"))?;

    Some(run_baseline_with_prepared_embeddings(dataset, config, run_id, strategy, &embedding_runtime))
}

pub fn run_baseline_with_prepared_embeddings(
    dataset: &CognitiveDataset,
    config: &CognitiveConfig,
    run_id: &str,
    strategy: BaselineStrategy,
    embedding_runtime: &BenchmarkEmbeddingRuntime,
) -> CognitiveResult {
    let total_start = Instant::now();
    let documents = build_context_documents(dataset);

    let query_start = Instant::now();
    let query_results =
        evaluate_baseline_queries(&documents, dataset, config, strategy, embedding_runtime);
    let query_time = query_start.elapsed();
    let total_time = total_start.elapsed();

    finalize_result(
        dataset,
        strategy.name(),
        run_id,
        query_results,
        Duration::ZERO,
        query_time,
        total_time,
        embedding_runtime.pseudo_miss_count.load(Ordering::Relaxed),
    )
}

fn finalize_result(
    dataset: &CognitiveDataset,
    strategy: &str,
    run_id: &str,
    query_results: StrategyRunData,
    ingest_time: Duration,
    query_time: Duration,
    total_time: Duration,
    embedding_cache_miss_count: u64,
) -> CognitiveResult {
    let StrategyRunData {
        query_scores,
        mut execution_latencies,
        mut evaluation_latencies,
        mut end_to_end_latencies,
        mut compiled_optimize_latencies,
        mut compiled_physical_plan_latencies,
        mut compiled_execute_plan_latencies,
        mut compiled_embed_latencies,
        mut compiled_decode_latencies,
        mut compiled_assemble_latencies,
        mut compiled_total_latencies,
        context_tokens,
        prompt_tokens,
        completion_tokens,
    } = query_results;

    // Aggregate by category.
    let categories = aggregate_categories(&query_scores);

    // Overall metrics (computed over positive queries only for containment/f1/recall).
    let positive_scores: Vec<&QueryScore> = query_scores.iter().filter(|s| !s.negative).collect();
    let negative_scores: Vec<&QueryScore> = query_scores.iter().filter(|s| s.negative).collect();
    let total = query_scores.len();
    let pos_total = positive_scores.len();

    let overall_containment = if pos_total > 0 {
        positive_scores.iter().map(|s| s.containment).sum::<f64>() / pos_total as f64
    } else {
        0.0
    };
    let overall_token_f1 = if pos_total > 0 {
        positive_scores.iter().map(|s| s.token_f1).sum::<f64>() / pos_total as f64
    } else {
        0.0
    };
    let overall_recall_accuracy = if pos_total > 0 {
        positive_scores
            .iter()
            .map(|score| score.recall_accuracy)
            .sum::<f64>()
            / pos_total as f64
    } else {
        0.0
    };
    let overall_mrr = if pos_total > 0 {
        positive_scores.iter().map(|s| s.mrr).sum::<f64>() / pos_total as f64
    } else {
        0.0
    };
    let overall_ndcg = if pos_total > 0 {
        positive_scores.iter().map(|s| s.ndcg).sum::<f64>() / pos_total as f64
    } else {
        0.0
    };
    let false_positive_rate = if !negative_scores.is_empty() {
        let fps = negative_scores.iter().filter(|s| s.false_positive).count();
        fps as f64 / negative_scores.len() as f64
    } else {
        0.0
    };

    let overall_semantic_similarity = if pos_total > 0 {
        positive_scores
            .iter()
            .map(|s| s.semantic_similarity)
            .sum::<f64>()
            / pos_total as f64
    } else {
        0.0
    };

    execution_latencies.sort_unstable();
    evaluation_latencies.sort_unstable();
    end_to_end_latencies.sort_unstable();
    compiled_optimize_latencies.sort_unstable();
    compiled_physical_plan_latencies.sort_unstable();
    compiled_execute_plan_latencies.sort_unstable();
    compiled_embed_latencies.sort_unstable();
    compiled_decode_latencies.sort_unstable();
    compiled_assemble_latencies.sort_unstable();
    compiled_total_latencies.sort_unstable();
    let execution_latency = latency_percentiles(&execution_latencies);
    let evaluation_latency = latency_percentiles(&evaluation_latencies);
    let end_to_end_latency = latency_percentiles(&end_to_end_latencies);
    let token_cost =
        TokenCostEstimate::from_totals(context_tokens, prompt_tokens, completion_tokens, total);
    let compiled_phase_timings = if compiled_optimize_latencies.is_empty() {
        None
    } else {
        Some(CompiledPhaseTimingSummary {
            optimize: latency_percentiles(&compiled_optimize_latencies),
            physical_plan: latency_percentiles(&compiled_physical_plan_latencies),
            execute_plan: latency_percentiles(&compiled_execute_plan_latencies),
            embed: latency_percentiles(&compiled_embed_latencies),
            decode: latency_percentiles(&compiled_decode_latencies),
            assemble: latency_percentiles(&compiled_assemble_latencies),
            total: latency_percentiles(&compiled_total_latencies),
        })
    };

    CognitiveResult {
        benchmark: dataset.name.clone(),
        strategy: strategy.to_string(),
        run_id: run_id.to_string(),
        categories,
        overall_containment,
        overall_token_f1,
        overall_recall_accuracy,
        overall_mrr,
        overall_ndcg,
        overall_semantic_similarity,
        false_positive_rate,
        execution_latency,
        evaluation_latency,
        end_to_end_latency,
        token_cost,
        total_queries: total,
        ingest_time_secs: ingest_time.as_secs_f64(),
        query_time_secs: query_time.as_secs_f64(),
        total_time_secs: total_time.as_secs_f64(),
        compiled_phase_timings,
        baselines: Vec::new(),
        reproducibility: None,
        embedding_cache_miss_count,
    }
}

fn duration_from_diag_ms(value: Option<f64>) -> Option<Duration> {
    value.and_then(|ms| {
        if ms.is_finite() && ms >= 0.0 {
            Some(Duration::from_secs_f64(ms / 1_000.0))
        } else {
            None
        }
    })
}

fn compiled_phase_sample(
    think_diagnostics: &QueryDiagnostics,
    recall_diagnostics: &QueryDiagnostics,
) -> Option<CompiledPhaseSample> {
    Some(CompiledPhaseSample {
        optimize: duration_from_diag_ms(think_diagnostics.optimize_ms)?
            + duration_from_diag_ms(recall_diagnostics.optimize_ms)?,
        physical_plan: duration_from_diag_ms(think_diagnostics.physical_plan_ms)?
            + duration_from_diag_ms(recall_diagnostics.physical_plan_ms)?,
        execute_plan: duration_from_diag_ms(think_diagnostics.execute_plan_ms)?
            + duration_from_diag_ms(recall_diagnostics.execute_plan_ms)?,
        embed: duration_from_diag_ms(think_diagnostics.embed_ms)?
            + duration_from_diag_ms(recall_diagnostics.embed_ms)?,
        // decode_ms is only set on THINK; treat absence (RECALL leg) as 0
        decode: duration_from_diag_ms(think_diagnostics.decode_ms).unwrap_or(Duration::ZERO)
            + duration_from_diag_ms(recall_diagnostics.decode_ms).unwrap_or(Duration::ZERO),
        assemble: duration_from_diag_ms(think_diagnostics.assemble_ms)?
            + duration_from_diag_ms(recall_diagnostics.assemble_ms)?,
        total: duration_from_diag_ms(think_diagnostics.total_ms)?
            + duration_from_diag_ms(recall_diagnostics.total_ms)?,
    })
}

fn require_compiled_diagnostics(
    query_id: &str,
    stage: &str,
    diagnostics: Option<QueryDiagnostics>,
) -> QueryDiagnostics {
    let diagnostics = diagnostics
        .unwrap_or_else(|| panic!("benchmark {stage} omitted diagnostics for query `{query_id}`"));
    require_query_clean_diagnostics(query_id, stage, &diagnostics);
    diagnostics
}

fn agent_id(name: &str) -> AgentId {
    AgentId::new(name).unwrap()
}

/// Convert epoch-millisecond timestamp from dataset Turn to a hirn Timestamp.
fn epoch_ms_to_timestamp(ms: u64) -> Timestamp {
    let dt = DateTime::<Utc>::from_timestamp_millis(ms as i64)
        .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap());
    Timestamp::from_datetime(dt)
}

/// Derive an agent name from a session ID (e.g., "h4-agent-alpha" -> "agent-alpha").
fn session_agent_name(session_id: &str) -> &str {
    session_id.strip_prefix("h4-").unwrap_or(session_id)
}

/// Resolve an embedding through the benchmark runtime.
fn resolve_embedding(
    text: &str,
    dims: usize,
    embedding_runtime: &BenchmarkEmbeddingRuntime,
) -> Vec<f32> {
    embedding_runtime.resolve_embedding(text, dims)
}

fn estimate_tokens(text: &str) -> usize {
    static TOKENIZER: std::sync::LazyLock<Option<tiktoken_rs::CoreBPE>> =
        std::sync::LazyLock::new(|| tiktoken_rs::cl100k_base().ok());

    if let Some(tokenizer) = &*TOKENIZER {
        tokenizer.encode_ordinary(text).len()
    } else {
        text.split_whitespace().count()
    }
}

fn lexical_terms(text: &str) -> BTreeSet<String> {
    const STOP_WORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is",
        "it", "of", "on", "or", "that", "the", "to", "was", "what", "when", "where", "which",
        "who", "why", "with",
    ];

    text.split_whitespace()
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|token| token.len() > 2 && !STOP_WORDS.contains(&token.as_str()))
        .collect()
}

fn lexical_overlap_score(query_terms: &BTreeSet<String>, doc_terms: &BTreeSet<String>) -> usize {
    query_terms.intersection(doc_terms).count()
}

#[derive(Debug, Clone)]
struct QueryRoutingProfile {
    namespace: Option<Namespace>,
    after: Option<Timestamp>,
    activation: Option<ActivationProfile>,
}

#[derive(Debug, Clone, Copy)]
struct ActivationProfile {
    depth: usize,
}

impl QueryRoutingProfile {
    fn for_query(dataset: &CognitiveDataset, query: &super::QAQuery) -> Self {
        let mut profile = Self {
            namespace: None,
            after: None,
            activation: None,
        };

        if query_uses_graph_reasoning(dataset, query) {
            profile.activation = Some(ActivationProfile { depth: 3 });
        }

        if dataset.benchmark == Benchmark::H5Action {
            profile.activation = Some(ActivationProfile { depth: 2 });
        }

        if dataset.benchmark == Benchmark::H4Agent {
            if let Some(session_id) = query.relevant_session_ids.first() {
                let name = session_agent_name(session_id);
                let agent = agent_id(name);
                profile.namespace = Some(Namespace::private_for(&agent));
                profile.activation = Some(ActivationProfile { depth: 2 });
            }
        }

        if dataset.benchmark == Benchmark::H6Safety {
            if query.category != "adversarial-robustness" {
                profile.namespace = Some(if query.category == "pii-handling" {
                    let hr_agent = agent_id("hr");
                    Namespace::private_for(&hr_agent)
                } else {
                    Namespace::default()
                });
            }

            if query.category == "conflict-resolution" {
                profile.activation = Some(ActivationProfile { depth: 2 });
            }
        }

        if dataset.benchmark == Benchmark::H2Temporal
            && matches!(
                query.category.as_str(),
                "knowledge-update" | "temporal-contradiction" | "recency"
            )
        {
            let base_ts = 1_700_000_000_000_u64;
            let week = 7 * 86_400_000_u64;
            let cutoff = if query.category == "temporal-contradiction" {
                base_ts + 9 * week
            } else {
                base_ts + 5 * week
            };
            profile.after = Some(epoch_ms_to_timestamp(cutoff));
        }

        profile
    }
}

fn apply_routing_to_think<'a>(
    mut builder: ThinkBuilder<'a>,
    profile: &QueryRoutingProfile,
) -> ThinkBuilder<'a> {
    if let Some(namespace) = profile.namespace {
        builder = builder.namespace(namespace);
    }
    if let Some(after) = profile.after {
        builder = builder.after(after);
    }
    if let Some(activation) = profile.activation {
        builder = builder
            .activation(ActivationMode::Spreading)
            .depth(activation.depth);
    }
    builder
}

fn apply_routing_to_recall<'a>(
    mut builder: RecallBuilder<'a>,
    profile: &QueryRoutingProfile,
) -> RecallBuilder<'a> {
    if let Some(namespace) = profile.namespace {
        builder = builder.namespace(namespace);
    }
    if let Some(after) = profile.after {
        builder = builder.after(after);
    }
    if let Some(activation) = profile.activation {
        builder = builder
            .activation(ActivationMode::Spreading)
            .depth(activation.depth);
    }
    builder
}

fn retrieved_candidate_from_record(record: &hirn::record::MemoryRecord) -> RetrievedCandidate {
    match record {
        hirn::record::MemoryRecord::Episodic(record) => RetrievedCandidate {
            content: record.content.clone(),
            source_id: metadata_source_id(&record.metadata),
        },
        hirn::record::MemoryRecord::Semantic(record) => RetrievedCandidate {
            content: record.description.clone(),
            source_id: None,
        },
        hirn::record::MemoryRecord::Working(record) => RetrievedCandidate {
            content: record.content.clone(),
            source_id: None,
        },
        hirn::record::MemoryRecord::Procedural(record) => RetrievedCandidate {
            content: record.description.clone(),
            source_id: None,
        },
    }
}

fn compiled_temporal_clause(profile: &QueryRoutingProfile) -> Option<ql_ast::TemporalClause> {
    profile
        .after
        .map(|timestamp| ql_ast::TemporalClause::After(timestamp.to_string()))
}

fn compiled_expand_clause(profile: &QueryRoutingProfile) -> Option<ql_ast::ExpandClause> {
    profile.activation.map(|activation| ql_ast::ExpandClause {
        depth: activation.depth,
        min_weight: None,
        activation: Some(ql_ast::ActivationModeAst::Spreading),
    })
}

fn compiled_namespace_clause(profile: &QueryRoutingProfile) -> Option<String> {
    profile
        .namespace
        .map(|namespace| namespace.as_str().to_string())
}

fn build_compiled_think_query(
    query: &super::QAQuery,
    profile: &QueryRoutingProfile,
    config: &CognitiveConfig,
) -> String {
    ql_ast::Statement::Think(Box::new(ql_ast::ThinkStmt {
        about: query.question.clone(),
        involving: None,
        temporal: compiled_temporal_clause(profile),
        expand: compiled_expand_clause(profile),
        follow_causes: None,
        where_clauses: Vec::new(),
        output_format: None,
        budget: Some(config.token_budget),
        namespace: compiled_namespace_clause(profile),
        consistency: None,
        limit: Some(BENCH_THINK_CANDIDATE_LIMIT),
        hybrid: config.effective_query_text_hybrid(),
        mode: ql_ast::RetrievalMode::Local,
        depth_mode: None,
        with_prospective: None,
        with_mcfa: None,
        provenance_depth: None,
        max_hops: None,
        community_depth: None,
    }))
    .to_string()
}

fn build_compiled_recall_query(
    query: &super::QAQuery,
    profile: &QueryRoutingProfile,
    config: &CognitiveConfig,
) -> String {
    ql_ast::Statement::Recall(Box::new(ql_ast::RecallStmt {
        layers: vec![Layer::Episodic, Layer::Semantic],
        about: query.question.clone(),
        involving: None,
        temporal: compiled_temporal_clause(profile),
        as_of: None,
        expand: compiled_expand_clause(profile),
        follow_causes: None,
        where_clauses: Vec::new(),
        subquery_filters: Vec::new(),
        modality: None,
        resource_roles: None,
        hydration_modes: None,
        artifact_kinds: None,
        depth_mode: None,
        with_prospective: None,
        with_mcfa: None,
        with_conflicts: false,
        provenance_depth: None,
        topic: None,
        group_by: None,
        projection: None,
        output_format: None,
        result_format: None,
        budget: None,
        namespace: compiled_namespace_clause(profile),
        from_realms: None,
        consistency: None,
        limit: Some(config.k),
        hybrid: config.effective_query_text_hybrid(),
    }))
    .to_string()
}

fn unpack_compiled_records(
    query_id: &str,
    stage: &str,
    result: QlQueryResult,
) -> (Option<String>, Vec<RetrievedCandidate>) {
    match result {
        QlQueryResult::Records(records) => (
            records.context,
            records
                .records
                .iter()
                .map(|record| retrieved_candidate_from_record(&record.record))
                .collect(),
        ),
        other => panic!(
            "benchmark {stage} expected record results for query `{query_id}`, got {other:?}"
        ),
    }
}

fn execute_direct_query(
    db: &HirnDB,
    config: &CognitiveConfig,
    query: &super::QAQuery,
    query_embedding: Vec<f32>,
    routing_profile: &QueryRoutingProfile,
) -> QueryExecution {
    let mut think = db
        .recall_view()
        .think(query_embedding.clone())
        .budget(config.token_budget);
    if config.effective_query_text_hybrid() {
        think = think.query_text(query.question.clone());
    }
    think = apply_routing_to_think(think, routing_profile);

    let (think_result, think_explanation) = require_query_stage(
        &query.id,
        "THINK",
        block_on(think.execute_with_explanation()),
    );
    require_query_clean_diagnostics(&query.id, "THINK", &think_explanation.retrieval.diagnostics);

    let mut recall = db.recall_view().query(query_embedding).limit(config.k);
    if config.effective_query_text_hybrid() {
        recall = recall.query_text(query.question.clone());
    }
    recall = apply_routing_to_recall(recall, routing_profile);

    let (recall_results, recall_explanation) = require_query_stage(
        &query.id,
        "RECALL",
        block_on(recall.execute_with_explanation()),
    );
    require_query_clean_diagnostics(&query.id, "RECALL", &recall_explanation.diagnostics);

    let ranked_results = recall_results
        .iter()
        .map(|result| retrieved_candidate_from_record(&result.record))
        .collect::<Vec<_>>();
    let context = think_result.context;
    let context_tokens = estimate_tokens(&context);

    QueryExecution {
        context,
        ranked_results,
        context_tokens,
        compiled_phase_sample: None,
    }
}

fn execute_compiled_query(
    db: &HirnDB,
    query: &super::QAQuery,
    profile: &QueryRoutingProfile,
    config: &CognitiveConfig,
) -> QueryExecution {
    let think_query = build_compiled_think_query(query, profile, config);
    let (think_result, think_diagnostics) = require_query_stage(
        &query.id,
        "THINK",
        block_on(db.ql().execute_with_diagnostics(&think_query)),
    );
    let think_diagnostics = require_compiled_diagnostics(&query.id, "THINK", think_diagnostics);
    let (context, _) = unpack_compiled_records(&query.id, "THINK", think_result);
    let context = context.unwrap_or_else(|| {
        panic!(
            "benchmark THINK omitted assembled context for query `{}`",
            query.id
        )
    });

    let recall_query = build_compiled_recall_query(query, profile, config);
    let (recall_result, recall_diagnostics) = require_query_stage(
        &query.id,
        "RECALL",
        block_on(db.ql().execute_with_diagnostics(&recall_query)),
    );
    let recall_diagnostics = require_compiled_diagnostics(&query.id, "RECALL", recall_diagnostics);
    let (_, ranked_results) = unpack_compiled_records(&query.id, "RECALL", recall_result);

    QueryExecution {
        context_tokens: estimate_tokens(&context),
        context,
        ranked_results,
        compiled_phase_sample: compiled_phase_sample(&think_diagnostics, &recall_diagnostics),
    }
}

fn execute_benchmark_query(
    db: &HirnDB,
    config: &CognitiveConfig,
    query: &super::QAQuery,
    query_embedding: Vec<f32>,
    routing_profile: &QueryRoutingProfile,
) -> QueryExecution {
    match config.execution_surface {
        BenchmarkExecutionSurface::DirectBuilders => {
            execute_direct_query(db, config, query, query_embedding, routing_profile)
        }
        BenchmarkExecutionSurface::CompiledHirnql => {
            execute_compiled_query(db, query, routing_profile, config)
        }
    }
}

fn query_uses_graph_reasoning(dataset: &CognitiveDataset, query: &super::QAQuery) -> bool {
    dataset.benchmark == Benchmark::H3Graph
        || matches!(query.category.as_str(), "multi-hop" | "world-knowledge")
}

fn build_context_documents(dataset: &CognitiveDataset) -> Vec<ContextDocument> {
    dataset
        .sessions
        .iter()
        .flat_map(|session| {
            session.turns.iter().map(|turn| {
                let content = render_turn_content(&session.id, turn);
                ContextDocument {
                    lexical_terms: lexical_terms(&turn.content),
                    token_count: estimate_tokens(&content),
                    content,
                    source_id: turn.source_id.clone(),
                }
            })
        })
        .collect()
}

fn query_uses_explicit_evidence(query: &super::QAQuery) -> bool {
    !query.evidence_ids.is_empty() || !query.evidence_snippets.is_empty()
}

fn relevant_target_count(query: &super::QAQuery) -> usize {
    if !query.evidence_ids.is_empty() {
        return query
            .evidence_ids
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            .max(1);
    }
    if !query.evidence_snippets.is_empty() {
        return query
            .evidence_snippets
            .iter()
            .map(|snippet| snippet.to_lowercase())
            .collect::<BTreeSet<_>>()
            .len()
            .max(1);
    }

    usize::from(!query.expected_answers.is_empty())
}

fn matched_evidence_snippet_key(
    candidate: &RetrievedCandidate,
    snippets: &[String],
) -> Option<String> {
    if snippets.is_empty() {
        return None;
    }

    let lower = candidate.content.to_lowercase();
    snippets
        .iter()
        .map(|snippet| snippet.trim())
        .filter(|snippet| !snippet.is_empty())
        .find_map(|snippet| {
            let snippet_key = snippet.to_lowercase();
            lower
                .contains(&snippet_key)
                .then(|| format!("snippet:{snippet_key}"))
        })
}

fn matched_target_key(candidate: &RetrievedCandidate, query: &super::QAQuery) -> Option<String> {
    if !query.evidence_ids.is_empty() {
        if let Some(source_id) = candidate.source_id.as_deref() {
            if query
                .evidence_ids
                .iter()
                .any(|expected| expected == source_id)
            {
                return Some(source_id.to_string());
            }
        }

        if let Some(snippet_key) = matched_evidence_snippet_key(candidate, &query.evidence_snippets)
        {
            return Some(snippet_key);
        }

        return None;
    }

    if !query.evidence_snippets.is_empty() {
        return matched_evidence_snippet_key(candidate, &query.evidence_snippets);
    }

    let lower = candidate.content.to_lowercase();
    if query
        .expected_answers
        .iter()
        .any(|answer| lower.contains(&answer.to_lowercase()))
    {
        return Some("answer".to_string());
    }

    None
}

fn recall_accuracy_for_query(results: &[RetrievedCandidate], query: &super::QAQuery) -> f64 {
    let target_count = relevant_target_count(query);
    if target_count == 0 {
        return 0.0;
    }

    let hits = results
        .iter()
        .filter_map(|candidate| matched_target_key(candidate, query))
        .collect::<BTreeSet<_>>()
        .len();

    hits.min(target_count) as f64 / target_count as f64
}

fn mrr_for_query(results: &[RetrievedCandidate], query: &super::QAQuery) -> f64 {
    for (index, candidate) in results.iter().enumerate() {
        if matched_target_key(candidate, query).is_some() {
            return 1.0 / (index as f64 + 1.0);
        }
    }

    0.0
}

fn ndcg_at_k_for_query(results: &[RetrievedCandidate], query: &super::QAQuery, k: usize) -> f64 {
    let top_k: Vec<&RetrievedCandidate> = results.iter().take(k).collect();
    if top_k.is_empty() {
        return 0.0;
    }

    let dcg: f64 = top_k
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let relevance = if matched_target_key(candidate, query).is_some() {
                1.0
            } else {
                0.0
            };
            relevance / (index as f64 + 2.0_f64).log2()
        })
        .sum();

    let ideal_k = k.min(relevant_target_count(query));
    if ideal_k == 0 {
        return 0.0;
    }

    let ideal_dcg: f64 = (0..ideal_k)
        .map(|index| 1.0 / (index as f64 + 2.0_f64).log2())
        .sum();
    if ideal_dcg == 0.0 {
        0.0
    } else {
        dcg / ideal_dcg
    }
}

fn metadata_source_id(metadata: &Metadata) -> Option<String> {
    match metadata.get(BENCH_SOURCE_ID_META_KEY) {
        Some(MetadataValue::String(source_id)) => Some(source_id.clone()),
        _ => None,
    }
}

fn score_query(
    query: &super::QAQuery,
    query_emb: &[f32],
    config: &CognitiveConfig,
    context: &str,
    ranked_results: &[RetrievedCandidate],
    embedding_runtime: &BenchmarkEmbeddingRuntime,
) -> QueryScore {
    if query.negative {
        let fp_context = eval::has_false_positive(context, &query.expected_answers);
        let fp_recall = ranked_results.iter().any(|candidate| {
            let lower = candidate.content.to_lowercase();
            query
                .expected_answers
                .iter()
                .any(|answer| lower.contains(&answer.to_lowercase()))
        });

        QueryScore {
            query_id: query.id.clone(),
            category: query.category.clone(),
            containment: 0.0,
            token_f1: 0.0,
            recall_accuracy: 0.0,
            recall_hit: false,
            mrr: 0.0,
            ndcg: 0.0,
            semantic_similarity: 0.0,
            negative: true,
            false_positive: fp_context || fp_recall,
        }
    } else {
        let ranked_contents: Vec<String> = ranked_results
            .iter()
            .map(|candidate| candidate.content.clone())
            .collect();
        let containment = eval::containment(context, &query.expected_answers);
        let token_f1 = eval::token_f1(context, &query.expected_answers);
        let recall_accuracy = recall_accuracy_for_query(ranked_results, query);
        let recall_hit = recall_accuracy > 0.0;
        let mrr = if query_uses_explicit_evidence(query) {
            mrr_for_query(ranked_results, query)
        } else {
            eval::mrr(&ranked_contents, &query.expected_answers)
        };
        let ndcg = if query_uses_explicit_evidence(query) {
            ndcg_at_k_for_query(ranked_results, query, config.k)
        } else {
            eval::ndcg_at_k(&ranked_contents, &query.expected_answers, config.k)
        };
        let expected_embs: Vec<Vec<f32>> = query
            .expected_answers
            .iter()
            .map(|answer| resolve_embedding(answer, config.embedding_dims, embedding_runtime))
            .collect();
        let semantic_similarity = eval::semantic_similarity(Some(query_emb), &expected_embs);

        QueryScore {
            query_id: query.id.clone(),
            category: query.category.clone(),
            containment,
            token_f1,
            recall_accuracy,
            recall_hit,
            mrr,
            ndcg,
            semantic_similarity,
            negative: false,
            false_positive: false,
        }
    }
}

fn execute_full_context_baseline(
    documents: &[ContextDocument],
    question: &str,
    token_budget: usize,
) -> QueryExecution {
    let available_budget = token_budget.saturating_sub(estimate_tokens(question));
    let mut selected = Vec::new();
    let mut used_tokens = 0;

    for doc in documents {
        if used_tokens >= available_budget {
            break;
        }
        if !selected.is_empty() && used_tokens + doc.token_count > available_budget {
            break;
        }
        selected.push(RetrievedCandidate {
            content: doc.content.clone(),
            source_id: doc.source_id.clone(),
        });
        used_tokens += doc.token_count;
    }

    QueryExecution {
        context: selected
            .iter()
            .map(|candidate| candidate.content.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        ranked_results: selected,
        context_tokens: used_tokens,
        compiled_phase_sample: None,
    }
}

fn choose_expansion_terms(
    doc_terms: &BTreeSet<String>,
    query_terms: &BTreeSet<String>,
    limit: usize,
) -> Vec<String> {
    let mut candidates: Vec<String> = doc_terms.difference(query_terms).cloned().collect();
    candidates.sort_by_key(|term| (Reverse(term.len()), term.clone()));
    candidates.into_iter().take(limit).collect()
}

fn execute_iterative_baseline(
    documents: &[ContextDocument],
    question: &str,
    config: &CognitiveConfig,
) -> QueryExecution {
    let question_tokens = estimate_tokens(question);
    let available_budget = config.token_budget.saturating_sub(question_tokens);
    let mut query_terms = lexical_terms(question);
    let mut selected_indices = Vec::new();
    let mut used_tokens = 0;
    let mut remaining: Vec<usize> = (0..documents.len()).collect();

    for _ in 0..config.k.clamp(1, ITERATIVE_BASELINE_HOPS) {
        let mut ranked: Vec<(usize, usize)> = remaining
            .iter()
            .map(|&idx| {
                (
                    idx,
                    lexical_overlap_score(&query_terms, &documents[idx].lexical_terms),
                )
            })
            .filter(|(_, score)| *score > 0)
            .collect();
        ranked.sort_by(|(left_idx, left_score), (right_idx, right_score)| {
            right_score
                .cmp(left_score)
                .then_with(|| left_idx.cmp(right_idx))
        });

        let Some((best_idx, _)) = ranked.first().copied() else {
            break;
        };
        let doc = &documents[best_idx];
        if !selected_indices.is_empty() && used_tokens + doc.token_count > available_budget {
            break;
        }

        used_tokens += doc.token_count;
        selected_indices.push(best_idx);
        remaining.retain(|&idx| idx != best_idx);

        for term in choose_expansion_terms(&doc.lexical_terms, &query_terms, 4) {
            query_terms.insert(term);
        }

        if used_tokens >= available_budget {
            break;
        }
    }

    if selected_indices.is_empty() {
        return execute_full_context_baseline(documents, question, config.token_budget);
    }

    let mut ranked: Vec<(usize, usize)> = documents
        .iter()
        .enumerate()
        .map(|(idx, doc)| (idx, lexical_overlap_score(&query_terms, &doc.lexical_terms)))
        .filter(|(_, score)| *score > 0)
        .collect();
    ranked.sort_by(|(left_idx, left_score), (right_idx, right_score)| {
        right_score
            .cmp(left_score)
            .then_with(|| left_idx.cmp(right_idx))
    });

    let ranked_contents = ranked
        .into_iter()
        .take(config.k.max(selected_indices.len()))
        .map(|(idx, _)| RetrievedCandidate {
            content: documents[idx].content.clone(),
            source_id: documents[idx].source_id.clone(),
        })
        .collect();
    let context = selected_indices
        .iter()
        .map(|&idx| documents[idx].content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    QueryExecution {
        context,
        ranked_results: ranked_contents,
        context_tokens: used_tokens,
        compiled_phase_sample: None,
    }
}

fn evaluate_baseline_queries(
    documents: &[ContextDocument],
    dataset: &CognitiveDataset,
    config: &CognitiveConfig,
    strategy: BaselineStrategy,
    embedding_runtime: &BenchmarkEmbeddingRuntime,
) -> StrategyRunData {
    let mut results = StrategyRunData::default();

    for query in &dataset.queries {
        let query_emb =
            resolve_embedding(&query.question, config.embedding_dims, embedding_runtime);
        let question_tokens = estimate_tokens(&query.question);
        let execution_start = Instant::now();
        let execution = match strategy {
            BaselineStrategy::FullContext => {
                execute_full_context_baseline(documents, &query.question, config.token_budget)
            }
            BaselineStrategy::IterativeRetrieval => {
                execute_iterative_baseline(documents, &query.question, config)
            }
        };
        let execution_latency = execution_start.elapsed();

        let evaluation_start = Instant::now();
        let query_score = score_query(
            query,
            &query_emb,
            config,
            &execution.context,
            &execution.ranked_results,
            embedding_runtime,
        );
        let evaluation_latency = evaluation_start.elapsed();

        results.query_scores.push(query_score);
        results.execution_latencies.push(execution_latency);
        results.evaluation_latencies.push(evaluation_latency);
        results
            .end_to_end_latencies
            .push(execution_latency + evaluation_latency);
        results.context_tokens += execution.context_tokens;
        results.prompt_tokens += question_tokens + execution.context_tokens;
    }

    results
}

/// Ingest all sessions as episodic records, returning a map of (session_id, turn_index) → MemoryId.
///
/// For H4 (multi-agent), each session is ingested under a separate agent with
/// a private namespace. Source datasets that provide timestamps keep them.
/// For H6 (safety), PII sessions are isolated under a private HR namespace.
/// All other benchmarks use a shared "bench" agent.
fn ingest_sessions(
    db: &HirnDB,
    dataset: &CognitiveDataset,
    dims: usize,
    embedding_runtime: &BenchmarkEmbeddingRuntime,
) -> HashMap<(String, usize), MemoryId> {
    let is_h4 = dataset.benchmark == Benchmark::H4Agent;
    let is_h6 = dataset.benchmark == Benchmark::H6Safety;
    let ingest_start = Instant::now();
    let total_records: usize = dataset
        .sessions
        .iter()
        .map(|session| session.turns.len())
        .sum();
    let ingest_batch_size = ingest_batch_size(total_records);
    let progress_interval = ingest_progress_every_batches(total_records);
    let mut memory_ids = HashMap::new();
    let mut pending_batches: HashMap<AgentId, PendingIngestBatch> = HashMap::new();
    let mut flushed_batches = 0usize;
    let mut ingested_records = 0usize;

    for session in &dataset.sessions {
        // H4: per-session agent identity + private namespace.
        // H6: PII sessions go under a private HR namespace;
        //     injection sessions go under a quarantine namespace.
        let aid = if is_h4 {
            agent_id(session_agent_name(&session.id))
        } else if is_h6 && session.id == "h6-pii" {
            agent_id("hr")
        } else {
            agent_id("bench")
        };
        let ns = if is_h4 {
            Some(Namespace::private_for(&aid))
        } else if is_h6 && session.id == "h6-pii" {
            Some(Namespace::private_for(&aid))
        } else if is_h6 && session.id == "h6-injection" {
            Some(Namespace::new("quarantine").unwrap())
        } else {
            None
        };

        for (i, turn) in session.turns.iter().enumerate() {
            let content = render_turn_content(&session.id, turn);
            let emb = resolve_embedding(&content, dims, embedding_runtime);
            let importance = benchmark_turn_importance(turn);
            let origin = benchmark_turn_origin(turn);

            let mut builder = EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(&content)
                .summary(&turn.content)
                .importance(importance)
                .agent_id(aid.clone())
                .origin(origin)
                .embedding(emb);

            if let Some(source_id) = turn.source_id.as_deref() {
                builder = builder.metadata_entry(BENCH_SOURCE_ID_META_KEY, source_id);
            }

            // H4/H6-PII: attach private namespace.
            if let Some(ref ns) = ns {
                builder = builder.namespace(ns.clone());
            }

            if let Some(ts_ms) = turn.timestamp {
                builder = builder.timestamp(epoch_ms_to_timestamp(ts_ms));
            }

            let record = builder.build().expect("build episodic record");
            let batch = pending_batches.entry(aid.clone()).or_default();
            batch.keys.push((session.id.clone(), i));
            batch.records.push(record);

            if batch.records.len() >= ingest_batch_size {
                let staged_records = ingested_records + batch.records.len();
                ingested_records += flush_ingest_batch(db, batch, &mut memory_ids);
                flushed_batches += 1;
                if should_log_ingest_progress(
                    flushed_batches,
                    staged_records,
                    total_records,
                    progress_interval,
                ) {
                    eprintln!(
                        "  ingest: {}/{} records ({:.1}%) across {} batch(es) in {:.2}s",
                        ingested_records,
                        total_records,
                        ingested_records as f64 * 100.0 / total_records as f64,
                        flushed_batches,
                        ingest_start.elapsed().as_secs_f64(),
                    );
                }
            }
        }
    }

    for batch in pending_batches.values_mut() {
        if !batch.records.is_empty() {
            let staged_records = ingested_records + batch.records.len();
            ingested_records += flush_ingest_batch(db, batch, &mut memory_ids);
            flushed_batches += 1;
            if should_log_ingest_progress(
                flushed_batches,
                staged_records,
                total_records,
                progress_interval,
            ) {
                eprintln!(
                    "  ingest: {}/{} records ({:.1}%) across {} batch(es) in {:.2}s",
                    ingested_records,
                    total_records,
                    ingested_records as f64 * 100.0 / total_records as f64,
                    flushed_batches,
                    ingest_start.elapsed().as_secs_f64(),
                );
            }
        }
    }

    eprintln!(
        "  batched ingest: {} sessions, {} episodic records across {} batch(es) in {:.2}s",
        dataset.sessions.len(),
        memory_ids.len(),
        flushed_batches,
        ingest_start.elapsed().as_secs_f64(),
    );

    memory_ids
}

/// Create causal and dependency edges for H3 (Graph & Causal Reasoning).
///
/// This builds explicit graph structure so spreading activation can traverse
/// causal chains and dependency relationships during multi-hop queries.
fn create_h3_causal_edges(
    db: &HirnDB,
    _dataset: &CognitiveDataset,
    memory_ids: &std::collections::HashMap<(String, usize), MemoryId>,
) {
    // Helper to get a MemoryId by session + turn index.
    let mid = |session: &str, idx: usize| -> Option<MemoryId> {
        memory_ids.get(&(session.to_string(), idx)).copied()
    };

    // System dependency chain: A → B, A → C, C → D, D → E.
    // h3-system-state turns: 0=A, 1=B, 2=C, 3=D, 4=E
    let deps: &[(&str, usize, &str, usize, EdgeRelation)] = &[
        // Service A routes to B and C
        (
            "h3-system-state",
            0,
            "h3-system-state",
            1,
            EdgeRelation::RelatedTo,
        ),
        (
            "h3-system-state",
            0,
            "h3-system-state",
            2,
            EdgeRelation::RelatedTo,
        ),
        // Service C depends on B tokens and writes to D
        (
            "h3-system-state",
            2,
            "h3-system-state",
            1,
            EdgeRelation::RelatedTo,
        ),
        (
            "h3-system-state",
            2,
            "h3-system-state",
            3,
            EdgeRelation::RelatedTo,
        ),
        // Service E reads from D via CDC
        (
            "h3-system-state",
            3,
            "h3-system-state",
            4,
            EdgeRelation::RelatedTo,
        ),
    ];

    for &(src_s, src_i, tgt_s, tgt_i, ref rel) in deps {
        if let (Some(src), Some(tgt)) = (mid(src_s, src_i), mid(tgt_s, tgt_i)) {
            let _ =
                block_on(
                    db.graph_view()
                        .connect_with(src, tgt, rel.clone(), 0.8, Metadata::new()),
                );
        }
    }

    // Incident causal chain: disk full → write fail → C fail → A 500s → E stall.
    // h3-incident-chain turns: 0=disk full, 1=write fail, 2=C→A 500s, 3=B healthy, 4=E stall, 5=customer impact
    let causals: &[(&str, usize, &str, usize)] = &[
        ("h3-incident-chain", 0, "h3-incident-chain", 1), // disk full → write fail
        ("h3-incident-chain", 1, "h3-incident-chain", 2), // write fail → C→A 500s
        ("h3-incident-chain", 1, "h3-incident-chain", 4), // write fail → E stall
        ("h3-incident-chain", 2, "h3-incident-chain", 5), // 500s → customer impact
    ];

    for &(src_s, src_i, tgt_s, tgt_i) in causals {
        if let (Some(src), Some(tgt)) = (mid(src_s, src_i), mid(tgt_s, tgt_i)) {
            let _ = block_on(db.graph_view().connect_with(
                src,
                tgt,
                EdgeRelation::Causes,
                0.9,
                Metadata::new(),
            ));
        }
    }

    // Link system state to incident: D (system-state:3) → disk full (incident:0)
    if let (Some(d_sys), Some(d_inc)) = (mid("h3-system-state", 3), mid("h3-incident-chain", 0)) {
        let _ = block_on(db.graph_view().connect_with(
            d_sys,
            d_inc,
            EdgeRelation::RelatedTo,
            0.7,
            Metadata::new(),
        ));
    }

    // Contradiction chain: Dev-1 claim → DBA confirms actual cause → Dev-1 corrects
    // h3-contradiction turns: 0=Dev-1 claim, 1=Dev-2 claim, 2=DBA confirms, 3=Dev-1 corrects
    if let (Some(claim), Some(confirm)) = (mid("h3-contradiction", 0), mid("h3-contradiction", 2)) {
        let _ = block_on(db.graph_view().connect_with(
            claim,
            confirm,
            EdgeRelation::Contradicts,
            0.9,
            Metadata::new(),
        ));
    }
    if let (Some(confirm), Some(correct)) = (mid("h3-contradiction", 2), mid("h3-contradiction", 3))
    {
        let _ = block_on(db.graph_view().connect_with(
            confirm,
            correct,
            EdgeRelation::Supports,
            0.8,
            Metadata::new(),
        ));
    }

    // Resolution chain: freed disk → D resumes → C recovers → E drains
    // h3-resolution turns: 0=freed disk, 1=C recovered, 2=E draining, 3=root cause fix
    if let (Some(freed), Some(c_recov)) = (mid("h3-resolution", 0), mid("h3-resolution", 1)) {
        let _ = block_on(db.graph_view().connect_with(
            freed,
            c_recov,
            EdgeRelation::Causes,
            0.9,
            Metadata::new(),
        ));
    }
    if let (Some(c_recov), Some(e_drain)) = (mid("h3-resolution", 1), mid("h3-resolution", 2)) {
        let _ = block_on(db.graph_view().connect_with(
            c_recov,
            e_drain,
            EdgeRelation::Causes,
            0.8,
            Metadata::new(),
        ));
    }
    // Link incident root cause to resolution
    if let (Some(disk_full), Some(freed)) = (mid("h3-incident-chain", 0), mid("h3-resolution", 0)) {
        let _ = block_on(db.graph_view().connect_with(
            disk_full,
            freed,
            EdgeRelation::RelatedTo,
            0.7,
            Metadata::new(),
        ));
    }

    // Cross-session: DBA confirmation (contradiction:2) → incident cause (incident:0)
    if let (Some(dba), Some(disk)) = (mid("h3-contradiction", 2), mid("h3-incident-chain", 0)) {
        let _ = block_on(db.graph_view().connect_with(
            dba,
            disk,
            EdgeRelation::Supports,
            0.9,
            Metadata::new(),
        ));
    }
    // Cross-session: DBA confirmation → root cause fix (resolution:3)
    if let (Some(dba), Some(fix)) = (mid("h3-contradiction", 2), mid("h3-resolution", 3)) {
        let _ = block_on(db.graph_view().connect_with(
            dba,
            fix,
            EdgeRelation::Causes,
            0.8,
            Metadata::new(),
        ));
    }
    // Cross-session: write fail → freed disk (connects incident to resolution)
    if let (Some(wfail), Some(freed)) = (mid("h3-incident-chain", 1), mid("h3-resolution", 0)) {
        let _ = block_on(db.graph_view().connect_with(
            wfail,
            freed,
            EdgeRelation::RelatedTo,
            0.7,
            Metadata::new(),
        ));
    }
    // Cross-session: customer impact → resolution (connecting impact to fix)
    if let (Some(impact), Some(c_recov)) = (mid("h3-incident-chain", 5), mid("h3-resolution", 1)) {
        let _ = block_on(db.graph_view().connect_with(
            impact,
            c_recov,
            EdgeRelation::RelatedTo,
            0.7,
            Metadata::new(),
        ));
    }
    // Cross-session: incident chain → resolution root cause fix
    if let (Some(incident_disk), Some(fix)) = (mid("h3-incident-chain", 0), mid("h3-resolution", 3))
    {
        let _ = block_on(db.graph_view().connect_with(
            incident_disk,
            fix,
            EdgeRelation::RelatedTo,
            0.8,
            Metadata::new(),
        ));
    }
    // Cross-session: E stall (incident:4) → E draining (resolution:2)
    if let (Some(e_stall), Some(e_drain)) = (mid("h3-incident-chain", 4), mid("h3-resolution", 2)) {
        let _ = block_on(db.graph_view().connect_with(
            e_stall,
            e_drain,
            EdgeRelation::RelatedTo,
            0.8,
            Metadata::new(),
        ));
    }
    // Cross-session: C→A 500s (incident:2) → C recovered (resolution:1)
    if let (Some(c_fail), Some(c_recov)) = (mid("h3-incident-chain", 2), mid("h3-resolution", 1)) {
        let _ = block_on(db.graph_view().connect_with(
            c_fail,
            c_recov,
            EdgeRelation::RelatedTo,
            0.8,
            Metadata::new(),
        ));
    }
    // Cross-session: customer impact → freed disk (resolution:0)
    if let (Some(impact), Some(freed)) = (mid("h3-incident-chain", 5), mid("h3-resolution", 0)) {
        let _ = block_on(db.graph_view().connect_with(
            impact,
            freed,
            EdgeRelation::Causes,
            0.8,
            Metadata::new(),
        ));
    }
}

/// Create action-grounding edges for H5 (Memory → Action).
///
/// Links the current-plan session to relevant past-decisions and failure-log
/// entries so spreading activation can traverse from planning context to the
/// supporting evidence (past decisions, past failures).
fn create_h5_action_edges(
    db: &HirnDB,
    memory_ids: &std::collections::HashMap<(String, usize), MemoryId>,
) {
    let mid = |session: &str, idx: usize| -> Option<MemoryId> {
        memory_ids.get(&(session.to_string(), idx)).copied()
    };

    // h5-current-plan turns:
    //   0 = "Deploy a new real-time analytics service..."
    //   1 = "The analytics service needs a message queue, a time-series database, and an API gateway with gRPC."
    //   2 = "Step 1: Set up the message queue. Step 2: Deploy time-series DB. Step 3: Configure API gateway..."
    //   3 = "Use existing infrastructure tools and avoid repeating past failures."
    //
    // h5-past-decisions turns:
    //   0 = Kafka chosen over RabbitMQ
    //   1 = PostgreSQL over MySQL
    //   2 = blue-green deployments
    //   3 = AWS over GCP
    //
    // h5-failure-log turns:
    //   0 = MySQL failed → TimescaleDB
    //   1 = NGINX failed → Envoy
    //   2 = Redis Cluster failed → Redis Sentinel
    //   3 = Jenkins failed → GitHub Actions
    //
    // h5-tool-configs turns:
    //   0 = Terraform state
    //   1 = Docker base image
    //   2 = CI pipeline: GitHub Actions
    //   3 = Secret management

    let edges: &[(&str, usize, &str, usize, EdgeRelation, f64)] = &[
        // "needs a message queue" → Kafka decision
        (
            "h5-current-plan",
            1,
            "h5-past-decisions",
            0,
            EdgeRelation::RelatedTo,
            0.9,
        ),
        // "needs a time-series database" → MySQL failure → TimescaleDB
        (
            "h5-current-plan",
            1,
            "h5-failure-log",
            0,
            EdgeRelation::RelatedTo,
            0.9,
        ),
        // "needs an API gateway with gRPC" → NGINX failure → Envoy
        (
            "h5-current-plan",
            1,
            "h5-failure-log",
            1,
            EdgeRelation::RelatedTo,
            0.9,
        ),
        // "avoid repeating past failures" → each failure log entry
        (
            "h5-current-plan",
            3,
            "h5-failure-log",
            0,
            EdgeRelation::RelatedTo,
            0.8,
        ),
        (
            "h5-current-plan",
            3,
            "h5-failure-log",
            1,
            EdgeRelation::RelatedTo,
            0.8,
        ),
        (
            "h5-current-plan",
            3,
            "h5-failure-log",
            2,
            EdgeRelation::RelatedTo,
            0.8,
        ),
        (
            "h5-current-plan",
            3,
            "h5-failure-log",
            3,
            EdgeRelation::RelatedTo,
            0.8,
        ),
        // Step list → tool configs (CI pipeline)
        (
            "h5-current-plan",
            2,
            "h5-tool-configs",
            2,
            EdgeRelation::RelatedTo,
            0.7,
        ),
        // Kafka decision → failure log (Jenkins → GitHub Actions for CI)
        (
            "h5-past-decisions",
            3,
            "h5-tool-configs",
            0,
            EdgeRelation::RelatedTo,
            0.6,
        ),
    ];

    for &(src_s, src_i, tgt_s, tgt_i, ref rel, weight) in edges {
        if let (Some(src), Some(tgt)) = (mid(src_s, src_i), mid(tgt_s, tgt_i)) {
            let _ = block_on(db.graph_view().connect_with(
                src,
                tgt,
                rel.clone(),
                weight as f32,
                Metadata::new(),
            ));
        }
    }
}

/// Create Contradicts/Supports edges for H6 conflict-resolution sessions.
///
/// Links the conflicting deadline statements so spreading activation can
/// traverse from one to the authoritative correction.
fn create_h6_conflict_edges(
    db: &HirnDB,
    memory_ids: &std::collections::HashMap<(String, usize), MemoryId>,
) {
    let mid = |session: &str, idx: usize| -> Option<MemoryId> {
        memory_ids.get(&(session.to_string(), idx)).copied()
    };

    // h6-conflict turns:
    //   0 = Manager-A: "March 30th"
    //   1 = Manager-B: "extended to April 15th"
    //   2 = Director: "AUTHORITATIVE: April 15th, March 30th superseded"
    //   3 = Manager-A: "Acknowledged. Updating to April 15th"
    let edges: &[(&str, usize, &str, usize, EdgeRelation, f32)] = &[
        // Director contradicts Manager-A's original date
        (
            "h6-conflict",
            2,
            "h6-conflict",
            0,
            EdgeRelation::Contradicts,
            0.9,
        ),
        // Director supports Manager-B's extension
        (
            "h6-conflict",
            2,
            "h6-conflict",
            1,
            EdgeRelation::Supports,
            0.9,
        ),
        // Manager-A acknowledges the new date
        (
            "h6-conflict",
            3,
            "h6-conflict",
            2,
            EdgeRelation::Supports,
            0.8,
        ),
        // Manager-B's extension relates to Manager-A's original
        (
            "h6-conflict",
            1,
            "h6-conflict",
            0,
            EdgeRelation::Contradicts,
            0.8,
        ),
    ];

    for &(src_s, src_i, tgt_s, tgt_i, ref rel, weight) in edges {
        if let (Some(src), Some(tgt)) = (mid(src_s, src_i), mid(tgt_s, tgt_i)) {
            let _ = block_on(db.graph_view().connect_with(
                src,
                tgt,
                rel.clone(),
                weight,
                Metadata::new(),
            ));
        }
    }
}

/// Evaluate all queries via think() and recall().
///
/// Benchmark-aware query strategy:
/// - H3 (graph): enables spreading activation for multi-hop reasoning.
/// - H4 (agent): scopes queries to the appropriate agent namespace.
/// - H2 (temporal): applies `.after()` temporal filters for recency queries.
fn evaluate_queries(
    db: &HirnDB,
    dataset: &CognitiveDataset,
    config: &CognitiveConfig,
    embedding_runtime: &BenchmarkEmbeddingRuntime,
) -> StrategyRunData {
    let mut results = StrategyRunData::default();
    let benchmark_start = Instant::now();
    let total_queries = dataset.queries.len();

    for (query_index, q) in dataset.queries.iter().enumerate() {
        let query_emb = resolve_embedding(&q.question, config.embedding_dims, embedding_runtime);
        let routing_profile = QueryRoutingProfile::for_query(dataset, q);
        let execution_start = Instant::now();
        let execution = execute_benchmark_query(db, config, q, query_emb.clone(), &routing_profile);
        let execution_latency = execution_start.elapsed();

        let evaluation_start = Instant::now();
        let query_score = score_query(
            q,
            &query_emb,
            config,
            &execution.context,
            &execution.ranked_results,
            embedding_runtime,
        );
        let evaluation_latency = evaluation_start.elapsed();

        results.query_scores.push(query_score);
        results.execution_latencies.push(execution_latency);
        results.evaluation_latencies.push(evaluation_latency);
        results
            .end_to_end_latencies
            .push(execution_latency + evaluation_latency);
        if let Some(compiled_phase_sample) = execution.compiled_phase_sample {
            results
                .compiled_optimize_latencies
                .push(compiled_phase_sample.optimize);
            results
                .compiled_physical_plan_latencies
                .push(compiled_phase_sample.physical_plan);
            results
                .compiled_execute_plan_latencies
                .push(compiled_phase_sample.execute_plan);
            results
                .compiled_embed_latencies
                .push(compiled_phase_sample.embed);
            results
                .compiled_decode_latencies
                .push(compiled_phase_sample.decode);
            results
                .compiled_assemble_latencies
                .push(compiled_phase_sample.assemble);
            results
                .compiled_total_latencies
                .push(compiled_phase_sample.total);
        }
        let question_tokens = estimate_tokens(&q.question);
        results.context_tokens += execution.context_tokens;
        results.prompt_tokens += question_tokens + execution.context_tokens;

        let completed = query_index + 1;
        if completed == 1 || completed == total_queries || completed % QUERY_PROGRESS_INTERVAL == 0
        {
            let elapsed = benchmark_start.elapsed();
            let average_ms = elapsed.as_secs_f64() * 1_000.0 / completed as f64;
            eprintln!(
                "  queries: {completed}/{total_queries} ({:.1}%) elapsed {:.2}s avg {:.2}ms/query",
                completed as f64 * 100.0 / total_queries as f64,
                elapsed.as_secs_f64(),
                average_ms,
            );
        }
    }

    results
}

pub fn compute_reproducibility(
    runs: &[CognitiveResult],
    threshold: f64,
) -> Option<ReproducibilitySummary> {
    if runs.len() < 2 {
        return None;
    }

    let reference = &runs[0];
    let metric_series = [
        ("containment", reference.overall_containment),
        ("token_f1", reference.overall_token_f1),
        ("recall_accuracy", reference.overall_recall_accuracy),
        ("mrr", reference.overall_mrr),
        ("ndcg", reference.overall_ndcg),
        ("false_positive_rate", reference.false_positive_rate),
        (
            "execution_p50_us",
            reference.execution_latency.p50.as_secs_f64() * 1_000_000.0,
        ),
        (
            "execution_p95_us",
            reference.execution_latency.p95.as_secs_f64() * 1_000_000.0,
        ),
        (
            "execution_p99_us",
            reference.execution_latency.p99.as_secs_f64() * 1_000_000.0,
        ),
        (
            "evaluation_p95_us",
            reference.evaluation_latency.p95.as_secs_f64() * 1_000_000.0,
        ),
        (
            "end_to_end_p95_us",
            reference.end_to_end_latency.p95.as_secs_f64() * 1_000_000.0,
        ),
        ("total_tokens", reference.token_cost.total_tokens as f64),
    ];

    let mut drifts = Vec::with_capacity(metric_series.len());
    let mut all_relative_deltas = Vec::new();

    for (metric, baseline) in metric_series {
        let deltas: Vec<f64> = runs[1..]
            .iter()
            .map(|run| {
                let current = match metric {
                    "containment" => run.overall_containment,
                    "token_f1" => run.overall_token_f1,
                    "recall_accuracy" => run.overall_recall_accuracy,
                    "mrr" => run.overall_mrr,
                    "ndcg" => run.overall_ndcg,
                    "false_positive_rate" => run.false_positive_rate,
                    "execution_p50_us" => run.execution_latency.p50.as_secs_f64() * 1_000_000.0,
                    "execution_p95_us" => run.execution_latency.p95.as_secs_f64() * 1_000_000.0,
                    "execution_p99_us" => run.execution_latency.p99.as_secs_f64() * 1_000_000.0,
                    "evaluation_p95_us" => run.evaluation_latency.p95.as_secs_f64() * 1_000_000.0,
                    "end_to_end_p95_us" => run.end_to_end_latency.p95.as_secs_f64() * 1_000_000.0,
                    "total_tokens" => run.token_cost.total_tokens as f64,
                    _ => 0.0,
                };
                relative_delta(current, baseline).abs()
            })
            .collect();
        let max_relative_delta = deltas.iter().copied().fold(0.0_f64, f64::max);
        let mean_relative_delta = if deltas.is_empty() {
            0.0
        } else {
            deltas.iter().sum::<f64>() / deltas.len() as f64
        };
        all_relative_deltas.extend(deltas.iter().copied());
        drifts.push(MetricDrift {
            metric: metric.to_string(),
            max_relative_delta,
            mean_relative_delta,
        });
    }

    let max_relative_delta = drifts
        .iter()
        .map(|drift| drift.max_relative_delta)
        .fold(0.0_f64, f64::max);
    let mean_relative_delta = if all_relative_deltas.is_empty() {
        0.0
    } else {
        all_relative_deltas.iter().sum::<f64>() / all_relative_deltas.len() as f64
    };

    Some(ReproducibilitySummary {
        runs: runs.len(),
        threshold,
        materially_similar: max_relative_delta <= threshold,
        max_relative_delta,
        mean_relative_delta,
        metrics: drifts,
    })
}

fn relative_delta(current: f64, baseline: f64) -> f64 {
    if baseline.abs() <= f64::EPSILON {
        if current.abs() <= f64::EPSILON {
            0.0
        } else {
            1.0
        }
    } else {
        (current - baseline) / baseline
    }
}

/// Aggregate per-query scores into per-category summaries.
fn aggregate_categories(scores: &[QueryScore]) -> Vec<CategoryScore> {
    let mut categories: std::collections::BTreeMap<&str, Vec<&QueryScore>> =
        std::collections::BTreeMap::new();

    for s in scores {
        categories.entry(s.category.as_str()).or_default().push(s);
    }

    categories
        .into_iter()
        .map(|(name, items)| {
            let n = items.len() as f64;
            let positive: Vec<&&QueryScore> = items.iter().filter(|s| !s.negative).collect();
            let negative: Vec<&&QueryScore> = items.iter().filter(|s| s.negative).collect();
            let pos_n = positive.len() as f64;

            CategoryScore {
                name: name.to_string(),
                containment: if pos_n > 0.0 {
                    positive.iter().map(|s| s.containment).sum::<f64>() / pos_n
                } else {
                    0.0
                },
                token_f1: if pos_n > 0.0 {
                    positive.iter().map(|s| s.token_f1).sum::<f64>() / pos_n
                } else {
                    0.0
                },
                recall_accuracy: if pos_n > 0.0 {
                    positive.iter().map(|s| s.recall_accuracy).sum::<f64>() / pos_n
                } else {
                    0.0
                },
                mrr: if pos_n > 0.0 {
                    positive.iter().map(|s| s.mrr).sum::<f64>() / pos_n
                } else {
                    0.0
                },
                ndcg: if pos_n > 0.0 {
                    positive.iter().map(|s| s.ndcg).sum::<f64>() / pos_n
                } else {
                    0.0
                },
                semantic_similarity: if pos_n > 0.0 {
                    positive.iter().map(|s| s.semantic_similarity).sum::<f64>() / pos_n
                } else {
                    0.0
                },
                false_positive_rate: if !negative.is_empty() {
                    negative.iter().filter(|s| s.false_positive).count() as f64
                        / negative.len() as f64
                } else {
                    0.0
                },
                total: n as usize,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::Benchmark;
    use tempfile::TempDir;

    fn sample_turn(speaker: &str) -> crate::cognitive::Turn {
        crate::cognitive::Turn {
            speaker: speaker.to_string(),
            content: "content".to_string(),
            timestamp: None,
            timestamp_text: None,
            source_id: None,
        }
    }

    fn benchmark_config_with_surface(
        retrieval_profile: BenchmarkRetrievalProfile,
        execution_surface: BenchmarkExecutionSurface,
        query_text_hybrid: bool,
    ) -> CognitiveConfig {
        CognitiveConfig {
            embedding_dims: 64,
            token_budget: 1024,
            k: 5,
            retrieval_profile,
            execution_surface,
            query_text_hybrid,
            embedder_policy: Default::default(),
        }
    }

    fn benchmark_config(
        retrieval_profile: BenchmarkRetrievalProfile,
        query_text_hybrid: bool,
    ) -> CognitiveConfig {
        benchmark_config_with_surface(
            retrieval_profile,
            BenchmarkExecutionSurface::CompiledHirnql,
            query_text_hybrid,
        )
    }

    fn run_benchmark_with_config(
        benchmark: Benchmark,
        retrieval_profile: BenchmarkRetrievalProfile,
        query_text_hybrid: bool,
    ) -> CognitiveResult {
        let ds = crate::cognitive::synthetic::generate(benchmark);
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("test");
        // Lightweight config: just validates that benchmarks produce valid
        // numeric scores. Full-scale runs use larger k and budget.
        let config = benchmark_config(retrieval_profile, query_text_hybrid);
        run(&ds, &config, &db_path, "test-run")
    }

    fn run_benchmark(benchmark: Benchmark) -> CognitiveResult {
        run_benchmark_with_config(benchmark, BenchmarkRetrievalProfile::Minimal, false)
    }

    #[test]
    fn h1_produces_scores() {
        let result = run_benchmark(Benchmark::H1Retrieval);
        assert!(result.total_queries > 0);
        assert!(result.overall_containment >= 0.0);
        assert!(result.overall_token_f1 >= 0.0);
        assert!(!result.categories.is_empty());
    }

    #[test]
    fn h1_produces_scores_with_query_text_hybrid() {
        let result = run_benchmark_with_config(
            Benchmark::H1Retrieval,
            BenchmarkRetrievalProfile::Ablation,
            true,
        );
        assert!(result.total_queries > 0);
        assert!(result.overall_containment >= 0.0);
        assert!(result.overall_token_f1 >= 0.0);
        assert!(!result.categories.is_empty());
    }

    fn assert_benchmark_valid(bench: Benchmark) {
        let result = run_benchmark(bench);
        assert!(
            result.total_queries > 0,
            "{}: should have queries",
            bench.name()
        );
        assert!(
            result.overall_containment.is_finite(),
            "{}: containment not finite",
            bench.name()
        );
        assert!(
            result.overall_token_f1.is_finite(),
            "{}: token_f1 not finite",
            bench.name()
        );
        assert!(
            result.overall_recall_accuracy.is_finite(),
            "{}: recall_accuracy not finite",
            bench.name()
        );
    }

    #[test]
    fn h2_produces_numeric_scores() {
        assert_benchmark_valid(Benchmark::H2Temporal);
    }

    #[test]
    fn h3_produces_numeric_scores() {
        assert_benchmark_valid(Benchmark::H3Graph);
    }

    #[test]
    fn h4_produces_numeric_scores() {
        assert_benchmark_valid(Benchmark::H4Agent);
    }

    #[test]
    fn h5_produces_numeric_scores() {
        assert_benchmark_valid(Benchmark::H5Action);
    }

    #[test]
    fn h6_produces_numeric_scores() {
        assert_benchmark_valid(Benchmark::H6Safety);
    }

    #[test]
    fn baseline_strategies_emit_latency_and_token_cost() {
        let dataset = crate::cognitive::synthetic::generate(Benchmark::H1Retrieval);
        let config = benchmark_config(BenchmarkRetrievalProfile::Minimal, false);

        let full_context = run_baseline_with_embeddings(
            &dataset,
            &config,
            "baseline-test",
            BaselineStrategy::FullContext,
            None,
        )
        .expect("baseline run should not be skipped with PseudoFallback policy");
        let iterative = run_baseline_with_embeddings(
            &dataset,
            &config,
            "baseline-test",
            BaselineStrategy::IterativeRetrieval,
            None,
        )
        .expect("baseline run should not be skipped with PseudoFallback policy");

        assert_eq!(full_context.strategy, "full-context");
        assert_eq!(iterative.strategy, "iterative-retrieval");
        assert!(full_context.execution_latency.p95 >= full_context.execution_latency.p50);
        assert!(iterative.execution_latency.p95 >= iterative.execution_latency.p50);
        assert!(full_context.token_cost.total_tokens > 0);
        assert!(iterative.token_cost.total_tokens > 0);
    }

    #[test]
    fn reproducibility_marks_identical_runs_as_similar() {
        let mut result = run_benchmark(Benchmark::H2Temporal);
        result.run_id = "run-a".to_string();
        let mut duplicate = result.clone();
        duplicate.run_id = "run-b".to_string();

        let reproducibility = compute_reproducibility(&[result, duplicate], 0.05).unwrap();
        assert!(reproducibility.materially_similar);
        assert_eq!(reproducibility.max_relative_delta, 0.0);
    }

    #[test]
    fn benchmark_turn_profile_prioritizes_summaries_and_observations() {
        let mut direct = sample_turn("Alice");
        direct.timestamp_text = Some("1:56 pm on 8 May, 2023".to_string());
        direct.source_id = Some("D1:1".to_string());

        let observation = sample_turn("Observation/Alice");
        let summary = sample_turn("SessionSummary");

        assert_eq!(benchmark_turn_origin(&direct), Origin::DirectObservation);
        assert_eq!(benchmark_turn_origin(&observation), Origin::LlmExtraction);
        assert_eq!(benchmark_turn_origin(&summary), Origin::LlmExtraction);

        assert!(benchmark_turn_importance(&summary) > benchmark_turn_importance(&observation));
        assert!(benchmark_turn_importance(&observation) > benchmark_turn_importance(&direct));
        assert!(benchmark_turn_importance(&summary) <= 1.0);
    }

    #[test]
    fn matched_target_key_falls_back_to_evidence_snippet_when_source_id_is_missing() {
        let query = crate::cognitive::QAQuery {
            id: "locomo-q1".to_string(),
            question: "Where did Alice move?".to_string(),
            expected_answers: vec!["Seattle".to_string()],
            category: "world-knowledge".to_string(),
            relevant_session_ids: vec!["sample-1::session_1".to_string()],
            evidence_ids: vec!["D1:1".to_string()],
            evidence_snippets: vec!["I moved to Seattle in 2022.".to_string()],
            negative: false,
        };
        let candidate = RetrievedCandidate {
            content: "[sample-1::session_1] Alice: I moved to Seattle in 2022.".to_string(),
            source_id: None,
        };

        assert_eq!(
            matched_target_key(&candidate, &query),
            Some("snippet:i moved to seattle in 2022.".to_string())
        );
    }

    #[test]
    fn world_knowledge_queries_use_graph_reasoning_generically() {
        let dataset = crate::cognitive::CognitiveDataset {
            name: "External fixture".to_string(),
            benchmark: Benchmark::H1Retrieval,
            sessions: Vec::new(),
            queries: Vec::new(),
        };
        let query = crate::cognitive::QAQuery {
            id: "locomo-q2".to_string(),
            question: "What fields would Caroline be likely to pursue in her education?"
                .to_string(),
            expected_answers: vec!["Psychology".to_string()],
            category: "world-knowledge".to_string(),
            relevant_session_ids: vec!["sample-1::session_1".to_string()],
            evidence_ids: vec!["D1:9".to_string(), "D1:11".to_string()],
            evidence_snippets: vec![
                "Gonna continue my edu and check out career options".to_string(),
                "I'm keen on counseling or working in mental health".to_string(),
            ],
            negative: false,
        };

        assert!(query_uses_graph_reasoning(&dataset, &query));
    }

    #[test]
    fn query_routing_profile_matches_generic_world_knowledge_path() {
        let dataset = crate::cognitive::CognitiveDataset {
            name: "External fixture".to_string(),
            benchmark: Benchmark::H1Retrieval,
            sessions: Vec::new(),
            queries: Vec::new(),
        };
        let query = crate::cognitive::QAQuery {
            id: "q-routing".to_string(),
            question: "What fields would Caroline be likely to pursue in her education?"
                .to_string(),
            expected_answers: vec!["Psychology".to_string()],
            category: "world-knowledge".to_string(),
            relevant_session_ids: vec![],
            evidence_ids: vec![],
            evidence_snippets: vec![],
            negative: false,
        };

        let profile = QueryRoutingProfile::for_query(&dataset, &query);
        let activation = profile.activation.expect("activation profile");

        assert!(profile.namespace.is_none());
        assert!(profile.after.is_none());
        assert_eq!(activation.depth, 3);
    }

    #[test]
    fn benchmark_config_defaults_to_compiled_surface() {
        assert_eq!(
            benchmark_config(BenchmarkRetrievalProfile::Minimal, false).execution_surface,
            BenchmarkExecutionSurface::CompiledHirnql
        );
    }

    #[test]
    fn benchmark_hirn_config_keeps_default_auto_edges_enabled() {
        let config = benchmark_config(BenchmarkRetrievalProfile::Minimal, false);
        let db_path = std::path::Path::new("/tmp/hirn-bench-config-test");
        let benchmark_config = benchmark_hirn_config(&config, db_path);

        assert_eq!(
            benchmark_config.max_auto_edges_per_record,
            HirnConfig::default().max_auto_edges_per_record
        );
    }

    #[test]
    fn benchmark_hirn_config_disables_quality_gate_for_minimal() {
        let config = benchmark_config(BenchmarkRetrievalProfile::Minimal, false);
        let db_path = std::path::Path::new("/tmp/hirn-bench-quality-gate-minimal");
        let benchmark_config = benchmark_hirn_config(&config, db_path);

        assert_eq!(benchmark_config.quality_gate_threshold, 0.0);
    }

    #[test]
    fn benchmark_hirn_config_keeps_quality_gate_for_full_stack_profiles() {
        let config = benchmark_config(BenchmarkRetrievalProfile::NormalFullStack, false);
        let db_path = std::path::Path::new("/tmp/hirn-bench-quality-gate-full-stack");
        let benchmark_config = benchmark_hirn_config(&config, db_path);

        assert_eq!(
            benchmark_config.quality_gate_threshold,
            HirnConfig::default().quality_gate_threshold
        );
    }

    #[test]
    fn require_query_stage_returns_success_value() {
        let value = require_query_stage("q-1", "THINK", Result::<usize, &str>::Ok(7));
        assert_eq!(value, 7);
    }

    #[test]
    fn require_query_stage_panics_on_failure() {
        let panic = std::panic::catch_unwind(|| {
            let _: () = require_query_stage("q-2", "RECALL", Result::<(), &str>::Err("boom"));
        })
        .expect_err("expected query-stage failure to panic");

        let message = if let Some(message) = panic.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = panic.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            String::new()
        };

        assert!(message.contains("benchmark RECALL failed for query `q-2`: boom"));
    }

    #[test]
    fn require_query_clean_diagnostics_accepts_clean_queries() {
        require_query_clean_diagnostics("q-clean", "THINK", &QueryDiagnostics::default());
    }

    #[test]
    fn require_query_clean_diagnostics_panics_on_advanced_retrieval_fallback() {
        let panic = std::panic::catch_unwind(|| {
            require_query_clean_diagnostics(
                "q-3",
                "RECALL",
                &QueryDiagnostics {
                    multivector_fallback_count: Some(1),
                    ..QueryDiagnostics::default()
                },
            );
        })
        .expect_err("expected fallback diagnostics to panic");

        let message = if let Some(message) = panic.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = panic.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            String::new()
        };

        assert!(message.contains("benchmark RECALL used retrieval fallback for query `q-3`"));
        assert!(message.contains("multivector_fallback_count=1"));
    }

    #[test]
    fn cognitive_config_profile_controls_effective_hybrid() {
        let minimal = benchmark_config(BenchmarkRetrievalProfile::Minimal, true);
        let full_stack = benchmark_config(BenchmarkRetrievalProfile::NormalFullStack, false);
        let ablation = benchmark_config(BenchmarkRetrievalProfile::Ablation, true);

        assert!(!minimal.effective_query_text_hybrid());
        assert!(full_stack.effective_query_text_hybrid());
        assert!(ablation.effective_query_text_hybrid());
    }

    #[test]
    fn full_stack_surface_guard_panics_when_provider_backed_features_are_missing() {
        let panic = std::panic::catch_unwind(|| {
            require_full_stack_surfaces(&ActiveRetrievalSurfaces {
                query_text_hybrid: true,
                graph_routing: true,
                notes: vec!["no provider reranker discovered from environment".to_string()],
                ..ActiveRetrievalSurfaces::default()
            });
        })
        .expect_err("expected incomplete full-stack surfaces to panic");

        let message = if let Some(message) = panic.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = panic.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            String::new()
        };

        assert!(message.contains(
            "normal-full-stack benchmark profile requires provider-backed retrieval surfaces"
        ));
        assert!(message.contains("multivector"));
        assert!(message.contains("reranker"));
    }

    #[test]
    fn benchmark_cache_embedder_fails_closed_on_missing_text() {
        let embedder = BenchmarkCacheEmbedder::new(
            Arc::new(crate::cognitive::openai::EmbeddingCache::new()),
            3,
        );

        let error = block_on(embedder.embed(&["missing query"]))
            .expect_err("missing cache entry should fail");

        assert!(
            error
                .to_string()
                .contains("benchmark embedding cache missing text")
        );
    }

    #[test]
    fn minimal_compiled_profile_installs_cache_backed_embedder() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("cache-backed-embedder");
        let lance_path = dir.path().join("lance");
        let storage: Arc<dyn PhysicalStore> = block_on(HirnDb::open(HirnDbConfig::local(
            lance_path.to_str().unwrap(),
        )))
        .unwrap()
        .store_arc();
        let config = benchmark_config_with_surface(
            BenchmarkRetrievalProfile::Minimal,
            BenchmarkExecutionSurface::CompiledHirnql,
            false,
        );
        let mut db = block_on(HirnDB::open_with_config(
            benchmark_hirn_config(&config, &db_path),
            storage,
        ))
        .unwrap();
        let cache = crate::cognitive::openai::EmbeddingCache::from([(
            "query".to_string(),
            vec![0.25; config.embedding_dims],
        )]);
        let embedding_runtime =
            BenchmarkEmbeddingRuntime::cache_backed(Arc::new(cache), config.embedding_dims);

        let setup = configure_benchmark_retrieval(&mut db, &config, &embedding_runtime);

        assert!(db.embedder().is_some());
        assert_eq!(setup.query_embedding_source, QueryEmbeddingSource::Cache);
        assert!(setup.query_embedding_model_label.is_none());
        assert!(
            setup
                .active_retrieval_surfaces
                .notes
                .iter()
                .any(|note| { note.contains("cache-backed benchmark embedder installed") })
        );
    }

    #[test]
    fn compiled_minimal_without_cache_tracks_pseudo_query_embeddings() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("compiled-pseudo-embedder");
        let lance_path = dir.path().join("lance");
        let storage: Arc<dyn PhysicalStore> = block_on(HirnDb::open(HirnDbConfig::local(
            lance_path.to_str().unwrap(),
        )))
        .unwrap()
        .store_arc();
        let config = benchmark_config_with_surface(
            BenchmarkRetrievalProfile::Minimal,
            BenchmarkExecutionSurface::CompiledHirnql,
            false,
        );
        let mut db = block_on(HirnDB::open_with_config(
            benchmark_hirn_config(&config, &db_path),
            storage,
        ))
        .unwrap();
        let embedding_runtime = BenchmarkEmbeddingRuntime::pseudo();

        let setup = configure_benchmark_retrieval(&mut db, &config, &embedding_runtime);

        assert_eq!(setup.query_embedding_source, QueryEmbeddingSource::Pseudo);
        assert!(setup.query_embedding_model_label.is_none());
    }

    #[derive(Debug)]
    struct TestProviderEmbedder {
        dims: usize,
        model_id: &'static str,
    }

    impl TestProviderEmbedder {
        fn new(dims: usize, model_id: &'static str) -> Self {
            Self { dims, model_id }
        }
    }

    fn test_provider_vector(text: &str, dims: usize) -> Vec<f32> {
        let mut vector = vec![0.0; dims];
        for (index, byte) in text.bytes().enumerate() {
            vector[index % dims] += byte as f32 / 255.0;
        }
        vector
    }

    #[async_trait]
    impl Embedder for TestProviderEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|text| Embedding {
                    vector: test_provider_vector(text, self.dims),
                    model_id: self.model_id.to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            self.dims
        }

        fn model_id(&self) -> &str {
            self.model_id
        }

        fn max_input_tokens(&self) -> usize {
            usize::MAX
        }
    }

    #[test]
    fn compiled_minimal_without_cache_can_use_provider_backed_benchmark_embeddings() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("compiled-provider-embedder");
        let lance_path = dir.path().join("lance");
        let storage: Arc<dyn PhysicalStore> = block_on(HirnDb::open(HirnDbConfig::local(
            lance_path.to_str().unwrap(),
        )))
        .unwrap()
        .store_arc();
        let config = benchmark_config_with_surface(
            BenchmarkRetrievalProfile::Minimal,
            BenchmarkExecutionSurface::CompiledHirnql,
            false,
        );
        let mut db = block_on(HirnDB::open_with_config(
            benchmark_hirn_config(&config, &db_path),
            storage,
        ))
        .unwrap();
        let dataset = crate::cognitive::CognitiveDataset {
            name: "provider-runtime".to_string(),
            benchmark: Benchmark::H1Retrieval,
            sessions: vec![crate::cognitive::Session {
                id: "session-a".to_string(),
                turns: vec![sample_turn("user")],
            }],
            queries: vec![crate::cognitive::QAQuery {
                id: "query-a".to_string(),
                question: "query".to_string(),
                expected_answers: vec!["answer".to_string()],
                category: "single-hop".to_string(),
                relevant_session_ids: vec!["session-a".to_string()],
                evidence_ids: Vec::new(),
                evidence_snippets: Vec::new(),
                negative: false,
            }],
        };
        let embedding_runtime = provider_backed_benchmark_runtime(
            &dataset,
            config.embedding_dims,
            Arc::new(TestProviderEmbedder::new(
                config.embedding_dims,
                "test-provider",
            )),
        )
        .unwrap();

        let setup = configure_benchmark_retrieval(&mut db, &config, &embedding_runtime);

        assert!(db.embedder().is_some());
        assert_eq!(setup.query_embedding_source, QueryEmbeddingSource::Provider);
        assert_eq!(
            setup.query_embedding_model_label.as_deref(),
            Some("test-provider")
        );
        assert_eq!(
            embedding_runtime.resolve_embedding("answer", config.embedding_dims),
            test_provider_vector("answer", config.embedding_dims)
        );
        assert!(
            setup
                .active_retrieval_surfaces
                .notes
                .iter()
                .any(|note| { note.contains("provider-backed benchmark embedder installed") })
        );
        assert!(setup.active_retrieval_surfaces.notes.iter().any(|note| {
            note.contains("minimal profile keeps provider-backed retrieval extras disabled")
        }));
    }

    #[test]
    fn compiled_surface_executes_without_forced_hybrid_minimal_profile() {
        let dataset = crate::cognitive::synthetic::generate(Benchmark::H1Retrieval);
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("compiled-surface");
        let config = benchmark_config_with_surface(
            BenchmarkRetrievalProfile::Minimal,
            BenchmarkExecutionSurface::CompiledHirnql,
            false,
        );

        let report = run_with_embeddings(&dataset, &config, &db_path, "compiled-surface", None)
            .expect("run should not be skipped with PseudoFallback policy");

        assert!(report.result.overall_containment.is_finite());
        assert!(report.active_retrieval_surfaces.compiled_hirnql);
        assert!(!report.active_retrieval_surfaces.query_text_hybrid);
        assert!(!report.active_retrieval_surfaces.quality_gate);
        assert!(!report.active_retrieval_surfaces.notes.iter().any(|note| {
            note.contains("approximation-only") || note.contains("hybrid recall statement")
        }));
    }

    #[test]
    fn compiled_think_query_emits_hybrid_clause_when_profile_enables_it() {
        let query = crate::cognitive::QAQuery {
            id: "hybrid-think".to_string(),
            question: "release readiness".to_string(),
            expected_answers: vec!["go".to_string()],
            category: "world-knowledge".to_string(),
            relevant_session_ids: Vec::new(),
            evidence_ids: Vec::new(),
            evidence_snippets: Vec::new(),
            negative: false,
        };
        let config = benchmark_config_with_surface(
            BenchmarkRetrievalProfile::Ablation,
            BenchmarkExecutionSurface::CompiledHirnql,
            true,
        );
        let profile = QueryRoutingProfile::for_query(
            &crate::cognitive::synthetic::generate(Benchmark::H1Retrieval),
            &query,
        );

        let compiled = build_compiled_think_query(&query, &profile, &config);

        assert!(compiled.contains("THINK ABOUT \"release readiness\""));
        assert!(compiled.contains(" HYBRID"), "compiled query: {compiled}");
    }
}
