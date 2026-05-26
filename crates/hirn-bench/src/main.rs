//! hirn-bench: benchmark framework for the hirn cognitive memory database.

use std::io::Write;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use hirn_bench::{
    advanced, cognitive, compare, load, metrics, output, provenance, runner, storage,
};
use std::collections::HashSet;

const H2_TEMPORAL_CONTRADICTION_SLICE: &str = "h2-temporal-contradiction";
const H2_TEMPORAL_CONTRADICTION_CATEGORY: &str = "temporal-contradiction";
const DEFAULT_EXTERNAL_MAX_SESSIONS: usize = 500;
const DEFAULT_EXTERNAL_MAX_RECORDS: usize = 10_000;
const DEFAULT_EXTERNAL_MAX_QUERIES: usize = 200;

fn truncate_for_log(text: &str, max_chars: usize) -> String {
    let mut truncated = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index >= max_chars {
            truncated.push_str("...");
            return truncated;
        }
        truncated.push(ch);
    }
    truncated
}

fn validate_embedding_cache_coverage(
    dataset: &cognitive::CognitiveDataset,
    cache: &cognitive::openai::EmbeddingCache,
) {
    let mut covered = 0usize;
    let mut missing_samples = Vec::new();
    let required_texts = cognitive::dataset_embedding_texts(dataset);
    let total = required_texts.len();

    for text in required_texts {
        if cache.contains_key(&text) {
            covered += 1;
        } else if missing_samples.len() < 3 {
            missing_samples.push(truncate_for_log(&text, 120));
        }
    }

    if covered != total {
        let missing = total.saturating_sub(covered);
        eprintln!(
            "Error: embedding cache coverage mismatch for '{}': covered {covered}/{total} required texts; missing {missing}.",
            dataset.name
        );
        eprintln!(
            "Regenerate the cache with `cargo run -p hirn-bench -- precompute-external --format-name ...` or the matching `precompute` command before rerunning this benchmark."
        );
        for sample in missing_samples {
            eprintln!("  missing key sample: {sample}");
        }
        std::process::exit(1);
    }

    eprintln!("  cache coverage: {covered}/{total} required texts");
}

fn resolve_query_embedding_model_label_for_artifact(
    corpus_embedding_source: &str,
    corpus_embedding_model_label: &str,
    query_embedding_source: cognitive::QueryEmbeddingSource,
    query_embedding_model_label: Option<&str>,
) -> Result<String, String> {
    match query_embedding_source {
        cognitive::QueryEmbeddingSource::Cache => {
            if !corpus_embedding_source.starts_with("cache:") {
                return Err(format!(
                    "benchmark corpus embeddings came from `{corpus_embedding_source}` but query embeddings came from `{query_embedding_source}`; corpus/query vectors must be produced by the same embedding source"
                ));
            }
            Ok(corpus_embedding_model_label.to_string())
        }
        cognitive::QueryEmbeddingSource::Pseudo => {
            if corpus_embedding_source != "pseudo-embedding" {
                return Err(format!(
                    "benchmark corpus embeddings came from `{corpus_embedding_source}` but query embeddings came from `{query_embedding_source}`; corpus/query vectors must be produced by the same embedding source"
                ));
            }
            Ok("pseudo-embedding".to_string())
        }
        cognitive::QueryEmbeddingSource::Provider => {
            if corpus_embedding_source != "provider" {
                return Err(format!(
                    "benchmark corpus embeddings came from `{corpus_embedding_source}` but query embeddings came from `{query_embedding_source}`; corpus/query vectors must be produced by the same embedding source"
                ));
            }
            query_embedding_model_label
                .filter(|label| !label.is_empty())
                .map(ToString::to_string)
                .ok_or_else(|| {
                    "benchmark query embedding source `provider` did not record a model id"
                        .to_string()
                })
        }
    }
}

fn reconcile_corpus_embedding_runtime(
    embeddings_path: Option<&str>,
    corpus_embedding_source: &mut String,
    corpus_embedding_model_label: &mut String,
    query_embedding_source: cognitive::QueryEmbeddingSource,
    query_embedding_model_label: Option<&str>,
) -> Result<(), String> {
    if embeddings_path.is_some()
        || !matches!(
            query_embedding_source,
            cognitive::QueryEmbeddingSource::Provider
        )
    {
        return Ok(());
    }

    let provider_model_label = query_embedding_model_label
        .filter(|label| !label.is_empty())
        .ok_or_else(|| {
            "benchmark query embedding source `provider` did not record a model id".to_string()
        })?;

    *corpus_embedding_source = "provider".to_string();
    *corpus_embedding_model_label = provider_model_label.to_string();
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct CognitiveBenchmarkTarget {
    benchmark: cognitive::Benchmark,
    category_filter: Option<&'static str>,
}

#[derive(Debug, Clone, Copy)]
struct ExternalSafetyLimits {
    max_sessions: usize,
    max_records: usize,
    max_queries: usize,
}

#[derive(Debug, Clone, Copy)]
struct ExternalDatasetSize {
    sessions: usize,
    records: usize,
    queries: usize,
}

fn external_dataset_size(dataset: &cognitive::CognitiveDataset) -> ExternalDatasetSize {
    let records = dataset
        .sessions
        .iter()
        .map(|session| session.turns.len())
        .sum();
    ExternalDatasetSize {
        sessions: dataset.sessions.len(),
        records,
        queries: dataset.queries.len(),
    }
}

fn apply_external_safety_limits(
    dataset: &mut cognitive::CognitiveDataset,
    limits: ExternalSafetyLimits,
) {
    if dataset.sessions.len() > limits.max_sessions {
        dataset.sessions.truncate(limits.max_sessions);
    }

    let mut kept_records = 0usize;
    for session in &mut dataset.sessions {
        if kept_records >= limits.max_records {
            session.turns.clear();
            continue;
        }

        let remaining = limits.max_records - kept_records;
        if session.turns.len() > remaining {
            session.turns.truncate(remaining);
        }
        kept_records += session.turns.len();
    }

    dataset.sessions.retain(|session| !session.turns.is_empty());

    let kept_session_ids: HashSet<&str> = dataset
        .sessions
        .iter()
        .map(|session| session.id.as_str())
        .collect();

    dataset.queries.retain(|query| {
        query.relevant_session_ids.is_empty()
            || query
                .relevant_session_ids
                .iter()
                .any(|id| kept_session_ids.contains(id.as_str()))
    });

    if dataset.queries.len() > limits.max_queries {
        dataset.queries.truncate(limits.max_queries);
    }
}

impl CognitiveBenchmarkTarget {
    fn new(benchmark: cognitive::Benchmark) -> Self {
        Self {
            benchmark,
            category_filter: None,
        }
    }

    fn h2_temporal_contradiction() -> Self {
        Self {
            benchmark: cognitive::Benchmark::H2Temporal,
            category_filter: Some(H2_TEMPORAL_CONTRADICTION_CATEGORY),
        }
    }

    fn display_name(self) -> String {
        match self.category_filter {
            Some(category) => format!("{}:{category}", self.benchmark.name()),
            None => self.benchmark.name().to_string(),
        }
    }
}

fn parse_cognitive_benchmark_targets(
    benchmark: &str,
) -> Result<Vec<CognitiveBenchmarkTarget>, String> {
    if benchmark == "all" {
        return Ok(cognitive::Benchmark::all()
            .iter()
            .copied()
            .map(CognitiveBenchmarkTarget::new)
            .collect());
    }

    if benchmark.eq_ignore_ascii_case(H2_TEMPORAL_CONTRADICTION_SLICE) {
        return Ok(vec![CognitiveBenchmarkTarget::h2_temporal_contradiction()]);
    }

    let parsed = benchmark.parse::<cognitive::Benchmark>()?;
    Ok(vec![CognitiveBenchmarkTarget::new(parsed)])
}

/// Benchmark framework for the hirn cognitive memory database.
#[derive(Parser)]
#[command(name = "hirn-bench", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone, Debug)]
struct ArtifactOutputArgs {
    /// Primary output format: json, csv, markdown.
    #[arg(long, default_value = "json")]
    format: String,

    /// Primary output file (default: stdout).
    #[arg(short, long)]
    output: Option<String>,

    /// Additional JSON artifact path written from the same in-memory result.
    #[arg(long)]
    json_output: Option<String>,

    /// Additional CSV artifact path written from the same in-memory result.
    #[arg(long)]
    csv_output: Option<String>,

    /// Additional Markdown artifact path written from the same in-memory result.
    #[arg(long)]
    markdown_output: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Synthetic IR benchmark (remember / recall / think).
    Synthetic {
        /// Number of records to insert.
        #[arg(long, default_value_t = 1000)]
        records: usize,

        /// Embedding dimensions.
        #[arg(long, default_value_t = 64)]
        dims: usize,

        /// Number of benchmark queries.
        #[arg(long, default_value_t = 10)]
        queries: usize,

        /// Top-K for retrieval metrics.
        #[arg(long, default_value_t = 10)]
        k: usize,

        /// Token budget for think operations.
        #[arg(long, default_value_t = 4096)]
        token_budget: usize,

        /// Number of warmup runs (discarded).
        #[arg(long, default_value_t = 1)]
        warmup: usize,

        /// Number of measured runs.
        #[arg(long, default_value_t = 3)]
        runs: usize,

        #[command(flatten)]
        outputs: ArtifactOutputArgs,
    },

    /// HIRN-Bench cognitive memory suites (H1-Retrieval, H2-Temporal,
    /// H3-Graph, H4-Agent, H5-Action, H6-Safety).
    Cognitive {
        /// Suite to run: h1, h2, h3, h4, h5, h6, h2-temporal-contradiction,
        /// or "all" to run every suite.
        #[arg(long, default_value = "all")]
        benchmark: String,

        /// Path to dataset directory. Uses synthetic data when omitted.
        #[arg(long)]
        data_dir: Option<String>,

        /// Path to precomputed embedding cache (JSON). When provided, uses
        /// real OpenAI embeddings instead of pseudo_embedding.
        #[arg(long)]
        embeddings: Option<String>,

        /// Human-readable embedding model label for published benchmark artifacts.
        #[arg(long)]
        embedding_model_label: Option<String>,

        /// Embedding dimensions (ignored when --embeddings is set; auto-detected from cache).
        #[arg(long, default_value_t = 64)]
        dims: usize,

        /// Token budget for think operations.
        #[arg(long, default_value_t = 4096)]
        token_budget: usize,

        /// Top-K for retrieval.
        #[arg(long, default_value_t = 10)]
        k: usize,

        /// Retrieval benchmark profile: minimal, normal-full-stack, or ablation.
        #[arg(long, default_value = "minimal")]
        retrieval_profile: String,

        /// Benchmark execution surface: direct-builders or compiled-hirnql.
        #[arg(long, default_value = "compiled-hirnql")]
        execution_surface: String,

        /// For the ablation profile, opt into passing raw question text into hybrid BM25+vector retrieval.
        #[arg(long)]
        query_text_hybrid: bool,

        #[command(flatten)]
        outputs: ArtifactOutputArgs,

        /// Score tracker file for regression detection.
        #[arg(long)]
        tracker: Option<String>,

        /// Number of independent runs for confidence intervals (F-27).
        #[arg(long, default_value_t = 1)]
        runs: usize,

        /// Synthetic data scale multiplier (F-37). 1 = base, N = (N-1)*base noise sessions added.
        #[arg(long, default_value_t = 1)]
        synthetic_scale: usize,

        /// Relative drift threshold for reproducibility checks, expressed as a percentage.
        #[arg(long, default_value_t = 15.0)]
        repro_threshold_percent: f64,

        /// Skip executable full-context and iterative-retrieval baseline runs.
        #[arg(long)]
        no_baselines: bool,

        /// Optional environment label for published artifacts.
        #[arg(long)]
        environment_label: Option<String>,
    },

    /// Advanced offline cognition and explanation benchmark suite.
    Advanced {
        /// Surface to run: explanation, dream, reconcile, plan, or "all".
        #[arg(long, default_value = "all")]
        benchmark: String,

        /// Number of independent runs for reproducibility summaries.
        #[arg(long, default_value_t = 1)]
        runs: usize,

        /// Maximum time to wait for a single offline operator run.
        #[arg(long, default_value_t = 5_000)]
        offline_wait_ms: u64,

        /// Relative drift threshold for reproducibility checks, expressed as a percentage.
        #[arg(long, default_value_t = 15.0)]
        repro_threshold_percent: f64,

        #[command(flatten)]
        outputs: ArtifactOutputArgs,

        /// Score tracker file for regression detection.
        #[arg(long)]
        tracker: Option<String>,

        /// Optional environment label for published artifacts.
        #[arg(long)]
        environment_label: Option<String>,
    },

    /// Run benchmarks against an external dataset (LoCoMo, DMR) (F-38).
    External {
        /// External format: locomo, dmr, longmemeval.
        #[arg(long)]
        format_name: String,

        /// Path to the dataset directory (required unless --auto-download is set).
        #[arg(long)]
        data_dir: Option<String>,

        /// Automatically download the dataset when a verified source is configured.
        /// The dataset will be stored in --cache-dir (default: ~/.cache/hirn-bench/).
        #[arg(long)]
        auto_download: bool,

        /// Cache directory for auto-downloaded datasets.
        #[arg(long, default_value = "~/.cache/hirn-bench")]
        cache_dir: String,

        /// Path to precomputed embedding cache (JSON).
        #[arg(long)]
        embeddings: Option<String>,

        /// Human-readable embedding model label for published benchmark artifacts.
        #[arg(long)]
        embedding_model_label: Option<String>,

        /// Embedding dimensions.
        #[arg(long, default_value_t = 64)]
        dims: usize,

        /// Token budget for think operations.
        #[arg(long, default_value_t = 4096)]
        token_budget: usize,

        /// Top-K for retrieval.
        #[arg(long, default_value_t = 10)]
        k: usize,

        /// Retrieval benchmark profile: minimal, normal-full-stack, or ablation.
        #[arg(long, default_value = "minimal")]
        retrieval_profile: String,

        /// Benchmark execution surface: direct-builders or compiled-hirnql.
        #[arg(long, default_value = "compiled-hirnql")]
        execution_surface: String,

        /// For the ablation profile, opt into passing raw question text into hybrid BM25+vector retrieval.
        #[arg(long)]
        query_text_hybrid: bool,

        /// Number of independent runs for variance analysis.
        #[arg(long, default_value_t = 1)]
        runs: usize,

        /// Relative drift threshold for reproducibility checks, expressed as a percentage.
        #[arg(long, default_value_t = 15.0)]
        repro_threshold_percent: f64,

        /// Skip executable full-context and iterative-retrieval baseline runs.
        #[arg(long)]
        no_baselines: bool,

        /// Optional environment label for published artifacts.
        #[arg(long)]
        environment_label: Option<String>,

        /// Maximum sessions to ingest for memory-safe local runs.
        #[arg(long, default_value_t = DEFAULT_EXTERNAL_MAX_SESSIONS)]
        max_sessions: usize,

        /// Maximum episodic records to ingest for memory-safe local runs.
        #[arg(long, default_value_t = DEFAULT_EXTERNAL_MAX_RECORDS)]
        max_records: usize,

        /// Maximum benchmark queries to execute for memory-safe local runs.
        #[arg(long, default_value_t = DEFAULT_EXTERNAL_MAX_QUERIES)]
        max_queries: usize,

        /// Disable safety limits and run the full external corpus.
        #[arg(long)]
        full_corpus: bool,

        #[command(flatten)]
        outputs: ArtifactOutputArgs,
    },

    /// Concurrent mixed remember/recall load benchmark.
    Load {
        /// Number of concurrent writer tasks.
        #[arg(long, default_value_t = 4)]
        writers: usize,

        /// Number of concurrent reader tasks.
        #[arg(long, default_value_t = 8)]
        readers: usize,

        /// Number of remember operations per writer.
        #[arg(long, default_value_t = 50)]
        writes_per_writer: usize,

        /// Number of records each writer submits per batch.
        #[arg(long, default_value_t = 16)]
        writer_batch_size: usize,

        /// Max auto edges each remembered record may create. Set to 0 to isolate storage/recall throughput.
        #[arg(long, default_value_t = 0)]
        max_auto_edges_per_record: usize,

        /// Number of recall operations per reader.
        #[arg(long, default_value_t = 100)]
        reads_per_reader: usize,

        /// Number of records to pre-seed before the timed load window.
        #[arg(long, default_value_t = 128)]
        preseed_records: usize,

        /// Embedding dimensions.
        #[arg(long, default_value_t = 64)]
        dims: usize,

        /// Top-K for recall during the load run.
        #[arg(long, default_value_t = 10)]
        k: usize,

        #[command(flatten)]
        outputs: ArtifactOutputArgs,
    },

    /// Precompute OpenAI embeddings for all synthetic datasets.
    ///
    /// Reads OPENAI_API_KEY from environment or .env file.
    /// Saves embedding cache to the specified directory.
    Precompute {
        /// Suite to precompute: h1, h2, ..., or "all".
        #[arg(long, default_value = "all")]
        benchmark: String,

        /// Output directory for embedding cache files.
        #[arg(long, default_value = "embeddings")]
        output_dir: String,

        /// Embedding model name (F-39).
        #[arg(long, default_value = cognitive::openai::DEFAULT_EMBEDDING_MODEL)]
        embedding_model: String,

        /// Embedding dimensions (F-39).
        #[arg(long, default_value_t = cognitive::openai::DEFAULT_EMBEDDING_DIMS)]
        embedding_dims: usize,

        /// Maximum number of texts to embed via API (F-29 spend guard).
        #[arg(long, default_value_t = 5000)]
        max_api_texts: usize,
    },

    /// Precompute OpenAI embeddings for an external dataset (LoCoMo, DMR, LongMemEval).
    ///
    /// Reads OPENAI_API_KEY from environment or .env file.
    /// Incrementally caches: re-running only embeds new/missing texts.
    PrecomputeExternal {
        /// External format: locomo, dmr, longmemeval.
        #[arg(long)]
        format_name: String,

        /// Path to the dataset directory (required unless --auto-download is set).
        #[arg(long)]
        data_dir: Option<String>,

        /// Automatically download the dataset from HuggingFace if not cached.
        #[arg(long)]
        auto_download: bool,

        /// Cache directory for auto-downloaded datasets.
        #[arg(long, default_value = "~/.cache/hirn-bench")]
        cache_dir: String,

        /// Output file for the embedding cache.
        #[arg(long, default_value = "embeddings/external_embeddings.json")]
        output: String,

        /// Embedding model name.
        #[arg(long, default_value = cognitive::openai::DEFAULT_EMBEDDING_MODEL)]
        embedding_model: String,

        /// Embedding dimensions.
        #[arg(long, default_value_t = cognitive::openai::DEFAULT_EMBEDDING_DIMS)]
        embedding_dims: usize,

        /// Maximum number of texts to embed via API (spend guard).
        #[arg(long, default_value_t = 10000)]
        max_api_texts: usize,
    },

    /// Compare two benchmark results and detect regressions.
    ///
    /// Exit code 1 if regressions exceed the threshold.
    BenchCompare {
        /// Path to baseline results JSON file.
        #[arg(long)]
        baseline: String,

        /// Path to current results JSON file.
        #[arg(long)]
        current: String,

        /// Regression threshold as percentage (e.g. 5.0 = 5%).
        #[arg(long, default_value_t = 5.0)]
        threshold: f64,

        /// Output format: text, github (GitHub Actions annotations).
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Storage-level benchmarks (hirn-storage): vector/hybrid/multivector search,
    /// resource persist/fetch, batch BFS.
    Storage {
        /// Number of episodic records to insert.
        #[arg(long, default_value_t = 1000)]
        records: usize,

        /// Embedding dimensions.
        #[arg(long, default_value_t = 64)]
        dims: usize,

        /// Number of graph edges to create.
        #[arg(long, default_value_t = 10_000)]
        edges: usize,

        /// BFS traversal depth.
        #[arg(long, default_value_t = 2)]
        bfs_depth: usize,

        /// BFS frontier size (number of start nodes).
        #[arg(long, default_value_t = 100)]
        bfs_frontier: usize,

        /// Warmup iterations per benchmark.
        #[arg(long, default_value_t = 1)]
        warmup: usize,

        /// Measured iterations per benchmark.
        #[arg(long, default_value_t = 5)]
        measured: usize,

        /// Top-K limit for search operations.
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Synthetic {
            records,
            dims,
            queries,
            k,
            token_budget,
            warmup,
            runs,
            outputs,
        } => run_synthetic(
            records,
            dims,
            queries,
            k,
            token_budget,
            warmup,
            runs,
            outputs,
        ),
        Command::Cognitive {
            benchmark,
            data_dir,
            embeddings,
            embedding_model_label,
            dims,
            token_budget,
            k,
            retrieval_profile,
            execution_surface,
            query_text_hybrid,
            outputs,
            tracker,
            runs,
            synthetic_scale,
            repro_threshold_percent,
            no_baselines,
            environment_label,
        } => run_cognitive(
            benchmark,
            data_dir,
            embeddings,
            embedding_model_label,
            dims,
            token_budget,
            k,
            retrieval_profile,
            execution_surface,
            query_text_hybrid,
            outputs,
            tracker,
            runs,
            synthetic_scale,
            repro_threshold_percent,
            no_baselines,
            environment_label,
        ),
        Command::Advanced {
            benchmark,
            runs,
            offline_wait_ms,
            repro_threshold_percent,
            outputs,
            tracker,
            environment_label,
        } => run_advanced(
            benchmark,
            runs,
            offline_wait_ms,
            repro_threshold_percent,
            outputs,
            tracker,
            environment_label,
        ),
        Command::Precompute {
            benchmark,
            output_dir,
            embedding_model,
            embedding_dims,
            max_api_texts,
        } => run_precompute(
            benchmark,
            output_dir,
            embedding_model,
            embedding_dims,
            max_api_texts,
        ),
        Command::PrecomputeExternal {
            format_name,
            data_dir,
            auto_download,
            cache_dir,
            output,
            embedding_model,
            embedding_dims,
            max_api_texts,
        } => run_precompute_external(
            format_name,
            data_dir,
            auto_download,
            cache_dir,
            output,
            embedding_model,
            embedding_dims,
            max_api_texts,
        ),
        Command::External {
            format_name,
            data_dir,
            auto_download,
            cache_dir,
            embeddings,
            embedding_model_label,
            dims,
            token_budget,
            k,
            retrieval_profile,
            execution_surface,
            query_text_hybrid,
            runs,
            repro_threshold_percent,
            no_baselines,
            environment_label,
            max_sessions,
            max_records,
            max_queries,
            full_corpus,
            outputs,
        } => run_external(
            format_name,
            data_dir,
            auto_download,
            cache_dir,
            embeddings,
            embedding_model_label,
            dims,
            token_budget,
            k,
            retrieval_profile,
            execution_surface,
            query_text_hybrid,
            runs,
            repro_threshold_percent,
            no_baselines,
            environment_label,
            max_sessions,
            max_records,
            max_queries,
            full_corpus,
            outputs,
        ),
        Command::BenchCompare {
            baseline,
            current,
            threshold,
            format,
        } => run_bench_compare(baseline, current, threshold, format),
        Command::Load {
            writers,
            readers,
            writes_per_writer,
            writer_batch_size,
            max_auto_edges_per_record,
            reads_per_reader,
            preseed_records,
            dims,
            k,
            outputs,
        } => run_load(
            writers,
            readers,
            writes_per_writer,
            writer_batch_size,
            max_auto_edges_per_record,
            reads_per_reader,
            preseed_records,
            dims,
            k,
            outputs,
        ),
        Command::Storage {
            records,
            dims,
            edges,
            bfs_depth,
            bfs_frontier,
            warmup,
            measured,
            limit,
        } => run_storage(
            records,
            dims,
            edges,
            bfs_depth,
            bfs_frontier,
            warmup,
            measured,
            limit,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_load(
    writers: usize,
    readers: usize,
    writes_per_writer: usize,
    writer_batch_size: usize,
    max_auto_edges_per_record: usize,
    reads_per_reader: usize,
    preseed_records: usize,
    dims: usize,
    k: usize,
    outputs: ArtifactOutputArgs,
) {
    let outputs = OutputTargets::from_args(outputs).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    let config = load::LoadConfig {
        writers,
        readers,
        writes_per_writer,
        writer_batch_size,
        max_auto_edges_per_record,
        reads_per_reader,
        preseed_records,
        embedding_dims: dims,
        k,
    };

    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("load");
    let run_id = ulid::Ulid::new().to_string();

    eprintln!("Running concurrent load benchmark: {run_id}");
    eprintln!(
        "  writers={} readers={} writes/worker={} writer_batch_size={} max_auto_edges_per_record={} reads/worker={} preseed={} dims={} k={}",
        writers,
        readers,
        writes_per_writer,
        writer_batch_size,
        max_auto_edges_per_record,
        reads_per_reader,
        preseed_records,
        dims,
        k,
    );

    let result = load::run(&config, &db_path, &run_id).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    eprintln!("Done in {:.2}s", result.total_time.as_secs_f64());
    eprintln!(
        "  remember p95={:.2}ms recall p95={:.2}ms total_ops/s={:.1}",
        result.remember_latency.p95.as_secs_f64() * 1_000.0,
        result.recall_latency.p95.as_secs_f64() * 1_000.0,
        result.throughput.total_ops_per_sec,
    );

    emit_outputs(&outputs, &result, load::write_result);
}

#[allow(clippy::too_many_arguments)]
fn run_synthetic(
    records: usize,
    dims: usize,
    queries: usize,
    k: usize,
    token_budget: usize,
    warmup: usize,
    runs: usize,
    outputs: ArtifactOutputArgs,
) {
    let config = metrics::BenchmarkConfig {
        num_records: records,
        embedding_dims: dims,
        num_queries: queries,
        k,
        token_budget,
        warmup_runs: warmup,
        measured_runs: runs,
    };

    let outputs = OutputTargets::from_args(outputs).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join("bench");
    let run_id = ulid::Ulid::new().to_string();

    eprintln!("Running synthetic benchmark: {run_id}");
    eprintln!(
        "  records={} dims={} queries={} k={} budget={} warmup={} runs={}",
        config.num_records,
        config.embedding_dims,
        config.num_queries,
        config.k,
        config.token_budget,
        config.warmup_runs,
        config.measured_runs,
    );

    let result = runner::run(&config, &db_path, &run_id);

    eprintln!("Done in {:.2}s", result.total_time.as_secs_f64());
    eprintln!(
        "  precision={:.4} recall={:.4} f1={:.4} mrr={:.4} ndcg={:.4}",
        result.aggregate.mean_precision,
        result.aggregate.mean_recall,
        result.aggregate.mean_f1,
        result.aggregate.mean_mrr,
        result.aggregate.mean_ndcg,
    );

    emit_outputs(&outputs, &result, output::write_result);
}

#[allow(clippy::too_many_arguments)]
fn run_advanced(
    benchmark: String,
    runs: usize,
    offline_wait_ms: u64,
    repro_threshold_percent: f64,
    outputs: ArtifactOutputArgs,
    tracker_path: Option<String>,
    environment_label: Option<String>,
) {
    let outputs = OutputTargets::from_args(outputs).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    let benchmarks: Vec<advanced::AdvancedBenchmark> = if benchmark == "all" {
        advanced::AdvancedBenchmark::all().to_vec()
    } else {
        vec![benchmark.parse().unwrap_or_else(|e: String| {
            eprintln!("Error: {e}");
            std::process::exit(1);
        })]
    };

    let run_id = ulid::Ulid::new().to_string();
    let config = advanced::AdvancedConfig {
        runs,
        offline_wait_ms,
        repro_threshold: repro_threshold_percent / 100.0,
        environment_label,
    };

    eprintln!("Running advanced offline cognition benchmarks: {run_id}");
    eprintln!(
        "  benchmarks={} runs={} offline_wait_ms={} repro_threshold={:.1}%",
        benchmarks
            .iter()
            .map(|benchmark| benchmark.name())
            .collect::<Vec<_>>()
            .join(", "),
        runs,
        offline_wait_ms,
        repro_threshold_percent,
    );

    let suite = advanced::run_suite(&benchmarks, &config, &run_id).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    for result in &suite.results {
        eprintln!(
            "  {}: primary={:.4} precision={:.4} recall={:.4} accuracy={:.4} usefulness={:.4} p95={:.2}ms tokens={} spend=${:.4}",
            result.benchmark,
            result.quality.primary_score,
            result.quality.precision,
            result.quality.recall,
            result.quality.accuracy,
            result.quality.usefulness,
            result.latency.p95.as_secs_f64() * 1_000.0,
            result.cost.total_tokens,
            result.cost.estimated_spend_usd,
        );
        if let Some(summary) = &result.reproducibility {
            eprintln!(
                "    reproducibility: {} (max drift {:.2}%, threshold {:.2}%)",
                if summary.materially_similar {
                    "materially similar"
                } else {
                    "drift exceeds threshold"
                },
                summary.max_relative_delta * 100.0,
                summary.threshold * 100.0,
            );
        }

        if let Some(ref tracker) = tracker_path {
            let tracker = std::path::Path::new(tracker);
            if let Ok(history) = advanced::tracker::load(tracker) {
                let regressions = advanced::tracker::check_regressions(result, &history);
                if regressions.is_empty() {
                    eprintln!("    ✓ No regressions detected");
                } else {
                    for regression in regressions {
                        eprintln!("    ⚠ REGRESSION: {regression}");
                    }
                }
            }
            advanced::tracker::save(tracker, result).unwrap_or_else(|error| {
                eprintln!("Warning: failed to save tracker: {error}");
            });
        }
    }

    eprintln!("\nDone in {:.2}s", suite.total_time_secs);
    eprintln!(
        "  overall primary score={:.1}%",
        suite.overall_primary_score * 100.0,
    );

    emit_outputs(&outputs, &suite, output::write_advanced_result);
}

#[allow(clippy::too_many_arguments)]
fn run_cognitive(
    benchmark: String,
    data_dir: Option<String>,
    embeddings_path: Option<String>,
    embedding_model_label: Option<String>,
    dims: usize,
    token_budget: usize,
    k: usize,
    retrieval_profile: String,
    execution_surface: String,
    query_text_hybrid: bool,
    outputs: ArtifactOutputArgs,
    tracker_path: Option<String>,
    runs: usize,
    synthetic_scale: usize,
    repro_threshold_percent: f64,
    no_baselines: bool,
    environment_label: Option<String>,
) {
    use cognitive::{
        BaselineStrategy, BenchmarkRetrievalProfile, CognitiveConfig, CognitiveSuiteResult,
    };
    use std::time::Instant;

    let outputs = OutputTargets::from_args(outputs).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    // Load precomputed embeddings if provided.
    let embedding_cache = embeddings_path.as_ref().map(|path| {
        eprintln!("Loading precomputed embeddings from {path}");
        let cache = cognitive::openai::load_cache(std::path::Path::new(path)).unwrap_or_else(|e| {
            eprintln!("Error loading embeddings: {e}");
            std::process::exit(1);
        });
        eprintln!("  loaded {} embeddings", cache.len());
        cache
    });

    // When using precomputed embeddings, use their dimensions (1536 for text-embedding-3-small).
    let effective_dims = if embedding_cache.is_some() {
        let sample_dim = embedding_cache
            .as_ref()
            .unwrap()
            .values()
            .next()
            .map(|v| v.len())
            .unwrap_or(dims);
        eprintln!("  using embedding dims={sample_dim}");
        sample_dim
    } else {
        dims
    };

    let retrieval_profile = retrieval_profile
        .parse::<BenchmarkRetrievalProfile>()
        .unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(1);
        });
    let execution_surface = execution_surface
        .parse::<cognitive::BenchmarkExecutionSurface>()
        .unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(1);
        });

    let config = CognitiveConfig {
        embedding_dims: effective_dims,
        token_budget,
        k,
        retrieval_profile,
        execution_surface,
        query_text_hybrid,
        embedder_policy: Default::default(),
    };
    let repro_threshold = repro_threshold_percent / 100.0;
    let mut corpus_embedding_source = embeddings_path.as_ref().map_or_else(
        || "pseudo-embedding".to_string(),
        |path| format!("cache:{path}"),
    );
    let mut embedding_model_label = embedding_model_label.unwrap_or_else(|| {
        if embeddings_path.is_some() {
            if effective_dims == cognitive::openai::DEFAULT_EMBEDDING_DIMS {
                cognitive::openai::model_name().to_string()
            } else {
                format!("precomputed-cache/{}d", effective_dims)
            }
        } else {
            "pseudo-embedding".to_string()
        }
    });
    let environment = provenance::current_environment_info(environment_label);

    let benchmark_targets = parse_cognitive_benchmark_targets(&benchmark).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        eprintln!("Expected: h1, h2, h3, h4, h5, h6, {H2_TEMPORAL_CONTRADICTION_SLICE}, or all");
        std::process::exit(1);
    });

    let run_id = ulid::Ulid::new().to_string();
    let suite_start = Instant::now();
    let mut results = Vec::new();
    let mut active_retrieval_surfaces = cognitive::ActiveRetrievalSurfaces::default();
    let mut query_embedding_source = None;
    let mut query_embedding_model_label = None;

    for target in benchmark_targets {
        let bench = target.benchmark;
        eprintln!(
            "Running cognitive benchmark: {} — {}",
            target.display_name(),
            bench.description()
        );

        // Load or generate dataset.
        let mut dataset = if let Some(ref dir) = data_dir {
            cognitive::loader::load(bench, std::path::Path::new(dir)).unwrap_or_else(|e| {
                eprintln!("Error loading dataset: {e}");
                std::process::exit(1);
            })
        } else {
            cognitive::synthetic::generate_scaled(bench, synthetic_scale)
        };

        if let Some(category) = target.category_filter {
            dataset.queries.retain(|query| query.category == category);
            if dataset.queries.is_empty() {
                eprintln!(
                    "Error: benchmark '{}' has no queries in category '{}'.",
                    bench.name(),
                    category
                );
                std::process::exit(1);
            }
            eprintln!(
                "  query filter: category='{}' ({} queries retained)",
                category,
                dataset.queries.len(),
            );
        }

        eprintln!(
            "  sessions={} queries={} runs={}",
            dataset.sessions.len(),
            dataset.queries.len(),
            runs,
        );

        if let Some(ref cache) = embedding_cache {
            validate_embedding_cache_coverage(&dataset, cache);
        }

        // Multi-run: run benchmark `runs` times and average (F-27).
        let mut run_results = Vec::with_capacity(runs);
        for run_i in 0..runs {
            let result = execute_cognitive_bundle(
                &dataset,
                &config,
                &run_id,
                embedding_cache.as_ref(),
                !no_baselines,
                "cognitive",
            );

            if runs > 1 {
                eprintln!(
                    "  run {}/{}: containment={:.4} f1={:.4} recall={:.4}",
                    run_i + 1,
                    runs,
                    result.primary.overall_containment,
                    result.primary.overall_token_f1,
                    result.primary.overall_recall_accuracy,
                );
            }

            active_retrieval_surfaces = result.active_retrieval_surfaces.clone();
            track_query_embedding_runtime(
                &mut query_embedding_source,
                &mut query_embedding_model_label,
                &result,
            );
            run_results.push(result);
        }

        let hirn_runs: Vec<cognitive::CognitiveResult> = run_results
            .iter()
            .map(|bundle| bundle.primary.clone())
            .collect();
        let mut result = if hirn_runs.len() == 1 {
            hirn_runs[0].clone()
        } else {
            average_cognitive_results(&hirn_runs)
        };
        result.reproducibility =
            cognitive::runner::compute_reproducibility(&hirn_runs, repro_threshold);
        if !no_baselines {
            result.baselines = average_baseline_results(&run_results, repro_threshold);
        }

        eprintln!(
            "  containment={:.4} token_f1={:.4} recall_accuracy={:.4} mrr={:.4} ndcg={:.4} fpr={:.4} p95={:.2}ms tokens={} ({:.2}s)",
            result.overall_containment,
            result.overall_token_f1,
            result.overall_recall_accuracy,
            result.overall_mrr,
            result.overall_ndcg,
            result.false_positive_rate,
            result.execution_latency.p95.as_secs_f64() * 1_000.0,
            result.token_cost.total_tokens,
            result.total_time_secs,
        );
        for baseline in &result.baselines {
            eprintln!(
                "    baseline {}: containment={:.4} recall={:.4} exec_p95={:.2}ms tokens={} delta_containment={:+.4}",
                baseline.strategy,
                baseline.overall_containment,
                baseline.overall_recall_accuracy,
                baseline.execution_latency.p95.as_secs_f64() * 1_000.0,
                baseline.token_cost.total_tokens,
                result.overall_containment - baseline.overall_containment,
            );
        }
        if let Some(repro) = &result.reproducibility {
            eprintln!(
                "  reproducibility: {} (max drift {:.2}%, threshold {:.2}%)",
                if repro.materially_similar {
                    "materially similar"
                } else {
                    "drift exceeds threshold"
                },
                repro.max_relative_delta * 100.0,
                repro.threshold * 100.0,
            );
        }

        // Per-category breakdown.
        for cat in &result.categories {
            eprintln!(
                "    {}: containment={:.4} f1={:.4} recall={:.4} mrr={:.4} ndcg={:.4} fpr={:.4} (n={})",
                cat.name,
                cat.containment,
                cat.token_f1,
                cat.recall_accuracy,
                cat.mrr,
                cat.ndcg,
                cat.false_positive_rate,
                cat.total,
            );
        }

        // Show SOTA comparison.
        let target = cognitive::baselines::target(bench);
        let competitive = cognitive::baselines::is_competitive(bench, result.overall_containment);
        let status = if competitive {
            "✓ COMPETITIVE"
        } else {
            "✗ below target"
        };
        eprintln!(
            "  vs SOTA: {status} (hirn={:.1}% target={} threshold={:.1}%)",
            result.overall_containment * 100.0,
            target.description,
            target.competitive_threshold * 100.0,
        );

        // Regression tracking.
        if let Some(ref tp) = tracker_path {
            let tp = std::path::Path::new(tp);
            if let Ok(history) = cognitive::tracker::load(tp) {
                let regressions = cognitive::tracker::check_regressions(&result, &history);
                if regressions.is_empty() {
                    eprintln!("  ✓ No regressions detected");
                } else {
                    for r in &regressions {
                        eprintln!("  ⚠ REGRESSION: {r}");
                    }
                }
            }
            cognitive::tracker::save(tp, &result).unwrap_or_else(|e| {
                eprintln!("Warning: failed to save tracker: {e}");
            });
        }

        results.push(result);
    }

    let final_score = cognitive::compute_final_score(&results);
    let geometric_mean = cognitive::compute_geometric_mean(&results);
    let min_suite_score = cognitive::compute_min_suite_score(&results);
    let all_competitive = cognitive::all_suites_competitive(&results);
    let resolved_query_embedding_source = query_embedding_source
        .expect("at least one benchmark run should record a query embedding source");
    reconcile_corpus_embedding_runtime(
        embeddings_path.as_deref(),
        &mut corpus_embedding_source,
        &mut embedding_model_label,
        resolved_query_embedding_source,
        query_embedding_model_label.as_deref(),
    )
    .unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });
    let resolved_query_embedding_model_label = resolve_query_embedding_model_label_for_artifact(
        &corpus_embedding_source,
        &embedding_model_label,
        resolved_query_embedding_source,
        query_embedding_model_label.as_deref(),
    )
    .unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });
    let suite_result = CognitiveSuiteResult {
        run_id,
        metadata: cognitive::SuiteMetadata {
            generated_at_rfc3339: provenance::generated_at_rfc3339(),
            dataset_source: data_dir.as_ref().map_or_else(
                || "synthetic".to_string(),
                |dir| format!("dataset-dir:{dir}"),
            ),
            corpus_embedding_source,
            embedding_model_label,
            query_embedding_source: resolved_query_embedding_source,
            query_embedding_model_label: resolved_query_embedding_model_label,
            embedding_dims: effective_dims,
            token_budget,
            k,
            retrieval_profile,
            execution_surface: config.execution_surface,
            query_text_hybrid: active_retrieval_surfaces.query_text_hybrid,
            active_retrieval_surfaces: active_retrieval_surfaces.clone(),
            runs,
            synthetic_scale: if data_dir.is_none() {
                Some(synthetic_scale)
            } else {
                None
            },
            baseline_strategies: if no_baselines {
                Vec::new()
            } else {
                BaselineStrategy::all()
                    .iter()
                    .map(|strategy| strategy.name().to_string())
                    .collect()
            },
            environment,
        },
        results,
        total_time_secs: suite_start.elapsed().as_secs_f64(),
        final_score,
        geometric_mean,
        min_suite_score,
        all_competitive,
    };

    eprintln!("\nDone in {:.2}s", suite_result.total_time_secs);

    // Per-suite validation summary.
    let validations = cognitive::baselines::validate_all(&suite_result.results);
    if !validations.is_empty() {
        eprintln!("\nPer-suite validation:");
        for v in &validations {
            let floor_icon = if v.meets_floor { "✓" } else { "✗" };
            let comp_icon = if v.meets_competitive { "✓" } else { "·" };
            let tgt_icon = if v.meets_target { "✓" } else { "·" };
            eprintln!(
                "  {}: floor={floor_icon} competitive={comp_icon} target={tgt_icon} \
                 (containment={:.1}% recall={:.1}% fpr={:.1}%)",
                v.benchmark.name(),
                v.containment * 100.0,
                v.recall_accuracy * 100.0,
                v.fpr * 100.0,
            );
        }
        let all_floor = validations.iter().all(|v| v.meets_floor);
        let all_competitive = validations.iter().all(|v| v.meets_competitive);
        eprintln!(
            "  Overall: floor={} competitive={}",
            if all_floor { "ALL PASS" } else { "SOME FAIL" },
            if all_competitive {
                "ALL PASS"
            } else {
                "SOME BELOW"
            },
        );
    }

    emit_outputs(&outputs, &suite_result, output::write_cognitive_result);
}

#[allow(clippy::too_many_arguments)]
fn run_external(
    format_name: String,
    data_dir: Option<String>,
    auto_download: bool,
    cache_dir: String,
    embeddings_path: Option<String>,
    embedding_model_label: Option<String>,
    dims: usize,
    token_budget: usize,
    k: usize,
    retrieval_profile: String,
    execution_surface: String,
    query_text_hybrid: bool,
    runs: usize,
    repro_threshold_percent: f64,
    no_baselines: bool,
    environment_label: Option<String>,
    max_sessions: usize,
    max_records: usize,
    max_queries: usize,
    full_corpus: bool,
    outputs: ArtifactOutputArgs,
) {
    use cognitive::{BenchmarkRetrievalProfile, CognitiveConfig};

    let outputs = OutputTargets::from_args(outputs).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    // Resolve cache directory (expand ~ to home).
    let resolved_cache_dir = if cache_dir.starts_with("~/") {
        if let Some(home) = dirs_home() {
            home.join(&cache_dir[2..])
        } else {
            std::path::PathBuf::from(&cache_dir)
        }
    } else {
        std::path::PathBuf::from(&cache_dir)
    };

    // Determine data directory, potentially auto-downloading.
    let data_path = if let Some(ref dir) = data_dir {
        std::path::PathBuf::from(dir)
    } else if auto_download {
        let format_dir = resolved_cache_dir.join(&format_name);
        match format_name.as_str() {
            "locomo" => cognitive::external::download_locomo(&format_dir).unwrap_or_else(|e| {
                eprintln!("Error downloading LoCoMo: {e}");
                std::process::exit(1);
            }),
            "dmr" => cognitive::external::download_dmr(&format_dir).unwrap_or_else(|e| {
                eprintln!("Error downloading DMR: {e}");
                std::process::exit(1);
            }),
            "longmemeval" | "lme" => cognitive::external::download_longmemeval(&format_dir)
                .unwrap_or_else(|e| {
                    eprintln!("Error downloading LongMemEval: {e}");
                    std::process::exit(1);
                }),
            _ => {
                eprintln!(
                    "Error: --auto-download is only supported for 'locomo', 'dmr', and 'longmemeval' formats"
                );
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Error: either --data-dir or --auto-download is required");
        std::process::exit(1);
    };

    let limit_hint = if full_corpus {
        None
    } else {
        Some(cognitive::external::ExternalLoadLimits {
            max_sessions,
            max_records,
            max_queries,
        })
    };

    let mut dataset = match format_name.as_str() {
        "locomo" => cognitive::external::load_locomo(&data_path),
        "dmr" => cognitive::external::load_dmr(&data_path),
        "longmemeval" | "lme" => {
            cognitive::external::load_longmemeval_with_limits(&data_path, limit_hint)
        }
        other => Err(format!(
            "unsupported format: {other} (expected: locomo, dmr, longmemeval)"
        )),
    }
    .unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    let original_size = external_dataset_size(&dataset);
    if full_corpus {
        eprintln!(
            "Warning: --full-corpus enabled; safety limits disabled for external dataset '{}' (sessions={} records={} queries={}).",
            dataset.name, original_size.sessions, original_size.records, original_size.queries,
        );
    } else {
        let limits = ExternalSafetyLimits {
            max_sessions,
            max_records,
            max_queries,
        };
        apply_external_safety_limits(&mut dataset, limits);
        let limited_size = external_dataset_size(&dataset);
        eprintln!(
            "Applied external safety limits: sessions {}→{}, records {}→{}, queries {}→{} (limits: sessions <= {}, records <= {}, queries <= {}).",
            original_size.sessions,
            limited_size.sessions,
            original_size.records,
            limited_size.records,
            original_size.queries,
            limited_size.queries,
            max_sessions,
            max_records,
            max_queries,
        );
    }

    if dataset.sessions.is_empty() || dataset.queries.is_empty() {
        eprintln!(
            "Error: external safety limits filtered the dataset to zero sessions or zero queries. Increase --max-sessions/--max-records/--max-queries, or use --full-corpus if you intentionally want the full dataset."
        );
        std::process::exit(1);
    }

    eprintln!(
        "External dataset '{}': sessions={} queries={}",
        dataset.name,
        dataset.sessions.len(),
        dataset.queries.len(),
    );

    let embedding_cache = embeddings_path.as_ref().map(|path| {
        eprintln!("Loading precomputed embeddings from {path}");
        let cache = cognitive::openai::load_cache(std::path::Path::new(path)).unwrap_or_else(|e| {
            eprintln!("Error loading embeddings: {e}");
            std::process::exit(1);
        });
        eprintln!("  loaded {} embeddings", cache.len());
        cache
    });

    let effective_dims = if let Some(ref cache) = embedding_cache {
        cache.values().next().map(|v| v.len()).unwrap_or(dims)
    } else {
        dims
    };

    let retrieval_profile = retrieval_profile
        .parse::<BenchmarkRetrievalProfile>()
        .unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(1);
        });
    let execution_surface = execution_surface
        .parse::<cognitive::BenchmarkExecutionSurface>()
        .unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(1);
        });

    if let Some(ref cache) = embedding_cache {
        validate_embedding_cache_coverage(&dataset, cache);
    }

    let config = CognitiveConfig {
        embedding_dims: effective_dims,
        token_budget,
        k,
        retrieval_profile,
        execution_surface,
        query_text_hybrid,
        embedder_policy: Default::default(),
    };
    let repro_threshold = repro_threshold_percent / 100.0;
    let mut corpus_embedding_source = embeddings_path.as_ref().map_or_else(
        || "pseudo-embedding".to_string(),
        |path| format!("cache:{path}"),
    );
    let mut embedding_model_label = embedding_model_label.unwrap_or_else(|| {
        if embeddings_path.is_some() {
            if effective_dims == cognitive::openai::DEFAULT_EMBEDDING_DIMS {
                cognitive::openai::model_name().to_string()
            } else {
                format!("precomputed-cache/{}d", effective_dims)
            }
        } else {
            "pseudo-embedding".to_string()
        }
    });
    let environment = provenance::current_environment_info(environment_label);

    let run_id = ulid::Ulid::new().to_string();

    // Multi-run support for variance analysis.
    let mut run_results = Vec::with_capacity(runs);
    let mut active_retrieval_surfaces = cognitive::ActiveRetrievalSurfaces::default();
    let mut query_embedding_source = None;
    let mut query_embedding_model_label = None;
    for run_i in 0..runs {
        let result = execute_cognitive_bundle(
            &dataset,
            &config,
            &run_id,
            embedding_cache.as_ref(),
            !no_baselines,
            "external",
        );

        if runs > 1 {
            eprintln!(
                "  run {}/{}: containment={:.4} f1={:.4} recall={:.4}",
                run_i + 1,
                runs,
                result.primary.overall_containment,
                result.primary.overall_token_f1,
                result.primary.overall_recall_accuracy,
            );
        }

        active_retrieval_surfaces = result.active_retrieval_surfaces.clone();
        track_query_embedding_runtime(
            &mut query_embedding_source,
            &mut query_embedding_model_label,
            &result,
        );
        run_results.push(result);
    }

    let resolved_query_embedding_source = query_embedding_source
        .expect("at least one benchmark run should record a query embedding source");
    reconcile_corpus_embedding_runtime(
        embeddings_path.as_deref(),
        &mut corpus_embedding_source,
        &mut embedding_model_label,
        resolved_query_embedding_source,
        query_embedding_model_label.as_deref(),
    )
    .unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });
    let resolved_query_embedding_model_label = resolve_query_embedding_model_label_for_artifact(
        &corpus_embedding_source,
        &embedding_model_label,
        resolved_query_embedding_source,
        query_embedding_model_label.as_deref(),
    )
    .unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    let primary_runs: Vec<cognitive::CognitiveResult> = run_results
        .iter()
        .map(|bundle| bundle.primary.clone())
        .collect();
    let mut result = if primary_runs.len() == 1 {
        primary_runs[0].clone()
    } else {
        average_cognitive_results(&primary_runs)
    };
    result.reproducibility =
        cognitive::runner::compute_reproducibility(&primary_runs, repro_threshold);
    if !no_baselines {
        result.baselines = average_baseline_results(&run_results, repro_threshold);
    }

    eprintln!(
        "  containment={:.4} token_f1={:.4} recall={:.4} mrr={:.4} ndcg={:.4} fpr={:.4} p95={:.2}ms tokens={} ({:.2}s)",
        result.overall_containment,
        result.overall_token_f1,
        result.overall_recall_accuracy,
        result.overall_mrr,
        result.overall_ndcg,
        result.false_positive_rate,
        result.execution_latency.p95.as_secs_f64() * 1_000.0,
        result.token_cost.total_tokens,
        result.total_time_secs,
    );
    for baseline in &result.baselines {
        eprintln!(
            "    baseline {}: containment={:.4} recall={:.4} exec_p95={:.2}ms tokens={} delta_containment={:+.4}",
            baseline.strategy,
            baseline.overall_containment,
            baseline.overall_recall_accuracy,
            baseline.execution_latency.p95.as_secs_f64() * 1_000.0,
            baseline.token_cost.total_tokens,
            result.overall_containment - baseline.overall_containment,
        );
    }
    if let Some(repro) = &result.reproducibility {
        eprintln!(
            "  reproducibility: {} (max drift {:.2}%, threshold {:.2}%)",
            if repro.materially_similar {
                "materially similar"
            } else {
                "drift exceeds threshold"
            },
            repro.max_relative_delta * 100.0,
            repro.threshold * 100.0,
        );
    }

    // Per-category breakdown.
    for cat in &result.categories {
        eprintln!(
            "    {}: containment={:.4} f1={:.4} recall={:.4} mrr={:.4} ndcg={:.4} (n={})",
            cat.name,
            cat.containment,
            cat.token_f1,
            cat.recall_accuracy,
            cat.mrr,
            cat.ndcg,
            cat.total,
        );
    }

    let single_result = std::slice::from_ref(&result);
    let final_score = cognitive::compute_final_score(single_result);
    let geometric_mean = cognitive::compute_geometric_mean(single_result);
    let min_suite_score = cognitive::compute_min_suite_score(single_result);
    let all_competitive = cognitive::all_suites_competitive(single_result);
    let total_time_secs = result.total_time_secs;
    let suite_result = cognitive::CognitiveSuiteResult {
        run_id: result.run_id.clone(),
        metadata: cognitive::SuiteMetadata {
            generated_at_rfc3339: provenance::generated_at_rfc3339(),
            dataset_source: if auto_download {
                format!("external:{format_name}:auto-download")
            } else {
                format!("external:{format_name}")
            },
            corpus_embedding_source,
            embedding_model_label,
            query_embedding_source: resolved_query_embedding_source,
            query_embedding_model_label: resolved_query_embedding_model_label,
            embedding_dims: effective_dims,
            token_budget,
            k,
            retrieval_profile,
            execution_surface: config.execution_surface,
            query_text_hybrid: active_retrieval_surfaces.query_text_hybrid,
            active_retrieval_surfaces,
            runs,
            synthetic_scale: None,
            baseline_strategies: if no_baselines {
                Vec::new()
            } else {
                cognitive::BaselineStrategy::all()
                    .iter()
                    .map(|strategy| strategy.name().to_string())
                    .collect()
            },
            environment,
        },
        results: vec![result],
        final_score,
        geometric_mean,
        min_suite_score,
        all_competitive,
        total_time_secs,
    };

    emit_outputs(&outputs, &suite_result, output::write_cognitive_result);
}

fn run_precompute(
    benchmark: String,
    output_dir: String,
    embedding_model: String,
    embedding_dims: usize,
    max_api_texts: usize,
) {
    use cognitive::Benchmark;

    // Load .env for OPENAI_API_KEY.
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: OPENAI_API_KEY not set. Create a .env file or export the variable.");
        std::process::exit(1);
    });

    let model_config = cognitive::openai::EmbeddingModelConfig {
        model: embedding_model,
        dims: embedding_dims,
    };

    let benchmarks: Vec<Benchmark> = if benchmark == "all" {
        Benchmark::all().to_vec()
    } else {
        vec![benchmark.parse().unwrap_or_else(|e: String| {
            eprintln!("Error: {e}");
            std::process::exit(1);
        })]
    };

    eprintln!(
        "Precomputing embeddings for {} benchmark(s) using {}",
        benchmarks.len(),
        model_config.model,
    );

    let report = cognitive::precompute::precompute(
        &benchmarks,
        &api_key,
        std::path::Path::new(&output_dir),
        &model_config,
        max_api_texts,
    )
    .unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    eprintln!("\nPrecompute complete:");
    eprintln!("  model: {}", report.model);
    eprintln!("  dims: {}", report.dims);
    eprintln!("  benchmarks: {}", report.benchmarks);
    eprintln!("  total texts: {}", report.total_texts);
    eprintln!("  est. tokens: ~{}", report.total_tokens_estimate);
    eprintln!("\nRun benchmarks with:");
    eprintln!(
        "  cargo run -p hirn-bench -- cognitive --embeddings {output_dir}/all_embeddings.json"
    );
}

#[allow(clippy::too_many_arguments)]
fn run_precompute_external(
    format_name: String,
    data_dir: Option<String>,
    auto_download: bool,
    cache_dir: String,
    output: String,
    embedding_model: String,
    embedding_dims: usize,
    max_api_texts: usize,
) {
    // Load .env for OPENAI_API_KEY.
    let _ = dotenvy::dotenv();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: OPENAI_API_KEY not set. Create a .env file or export the variable.");
        std::process::exit(1);
    });

    let model_config = cognitive::openai::EmbeddingModelConfig {
        model: embedding_model,
        dims: embedding_dims,
    };

    // Resolve cache directory (expand ~ to home).
    let resolved_cache_dir = if cache_dir.starts_with("~/") {
        if let Some(home) = dirs_home() {
            home.join(&cache_dir[2..])
        } else {
            std::path::PathBuf::from(&cache_dir)
        }
    } else {
        std::path::PathBuf::from(&cache_dir)
    };

    // Determine data directory, potentially auto-downloading.
    let data_path = if let Some(ref dir) = data_dir {
        std::path::PathBuf::from(dir)
    } else if auto_download {
        let format_dir = resolved_cache_dir.join(&format_name);
        match format_name.as_str() {
            "locomo" => cognitive::external::download_locomo(&format_dir).unwrap_or_else(|e| {
                eprintln!("Error downloading LoCoMo: {e}");
                std::process::exit(1);
            }),
            "dmr" => cognitive::external::download_dmr(&format_dir).unwrap_or_else(|e| {
                eprintln!("Error downloading DMR: {e}");
                std::process::exit(1);
            }),
            "longmemeval" | "lme" => cognitive::external::download_longmemeval(&format_dir)
                .unwrap_or_else(|e| {
                    eprintln!("Error downloading LongMemEval: {e}");
                    std::process::exit(1);
                }),
            _ => {
                eprintln!(
                    "Error: --auto-download is only supported for 'locomo', 'dmr', and 'longmemeval' formats"
                );
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Error: either --data-dir or --auto-download is required");
        std::process::exit(1);
    };

    let dataset = match format_name.as_str() {
        "locomo" => cognitive::external::load_locomo(&data_path),
        "dmr" => cognitive::external::load_dmr(&data_path),
        "longmemeval" | "lme" => cognitive::external::load_longmemeval(&data_path),
        other => Err(format!("unsupported format: {other}")),
    }
    .unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "Precomputing embeddings for {} ({} sessions, {} queries) using {}",
        dataset.name,
        dataset.sessions.len(),
        dataset.queries.len(),
        model_config.model,
    );

    let output_path = std::path::Path::new(&output);
    let report = cognitive::precompute::precompute_external(
        &dataset,
        &api_key,
        output_path,
        &model_config,
        max_api_texts,
    )
    .unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });

    eprintln!("\nPrecompute complete:");
    eprintln!("  model: {}", report.model);
    eprintln!("  dims: {}", report.dims);
    eprintln!("  total texts: {}", report.total_texts);
    eprintln!("  est. tokens: ~{}", report.total_tokens_estimate);
    eprintln!("\nRun benchmarks with:");
    eprintln!(
        "  cargo run -p hirn-bench -- external --format-name {format_name} --data-dir {} --embeddings {output}",
        data_path.display()
    );
}

#[derive(Debug, Clone)]
struct OutputTargets {
    primary_format: output::OutputFormat,
    primary_output: Option<String>,
    extra_file_outputs: Vec<(output::OutputFormat, String)>,
}

impl OutputTargets {
    fn from_args(args: ArtifactOutputArgs) -> Result<Self, String> {
        let primary_format: output::OutputFormat = args.format.parse()?;
        let extra_file_outputs = [
            (output::OutputFormat::Json, args.json_output),
            (output::OutputFormat::Csv, args.csv_output),
            (output::OutputFormat::Markdown, args.markdown_output),
        ]
        .into_iter()
        .filter_map(|(format, path)| path.map(|path| (format, path)))
        .collect::<Vec<_>>();

        validate_unique_output_paths(args.output.as_deref(), &extra_file_outputs)?;

        Ok(Self {
            primary_format,
            primary_output: args.output,
            extra_file_outputs,
        })
    }
}

fn validate_unique_output_paths(
    primary_output: Option<&str>,
    extra_file_outputs: &[(output::OutputFormat, String)],
) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();

    if let Some(path) = primary_output {
        seen.insert(path.to_string());
    }

    for (_, path) in extra_file_outputs {
        if !seen.insert(path.clone()) {
            return Err(format!("duplicate output path: {path}"));
        }
    }

    Ok(())
}

fn emit_outputs<T>(
    outputs: &OutputTargets,
    result: &T,
    mut write_result: impl FnMut(&T, output::OutputFormat, &mut dyn Write) -> std::io::Result<()>,
) {
    let mut primary_writer: Box<dyn Write> = open_output(&outputs.primary_output);
    write_result(result, outputs.primary_format, &mut primary_writer).expect("write output");

    for (format, path) in &outputs.extra_file_outputs {
        let mut writer = open_output_path(path);
        write_result(result, *format, &mut writer).expect("write output");
    }
}

fn open_output_path(path: &str) -> Box<dyn Write> {
    Box::new(std::fs::File::create(path).unwrap_or_else(|e| {
        eprintln!("Error creating {path}: {e}");
        std::process::exit(1);
    }))
}

fn open_output(path: &Option<String>) -> Box<dyn Write> {
    match path {
        Some(path) => open_output_path(path),
        None => Box::new(std::io::stdout().lock()),
    }
}

/// Resolve the user's home directory.
fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Average multiple `CognitiveResult`s from repeated runs (F-27).
fn average_cognitive_results(runs: &[cognitive::CognitiveResult]) -> cognitive::CognitiveResult {
    let n = runs.len() as f64;
    let first = &runs[0];
    let mut category_map: std::collections::BTreeMap<String, Vec<&cognitive::CategoryScore>> =
        std::collections::BTreeMap::new();
    for run in runs {
        for category in &run.categories {
            category_map
                .entry(category.name.clone())
                .or_default()
                .push(category);
        }
    }

    cognitive::CognitiveResult {
        benchmark: first.benchmark.clone(),
        strategy: first.strategy.clone(),
        run_id: first.run_id.clone(),
        categories: category_map
            .into_iter()
            .map(|(name, categories)| {
                let count = categories.len() as f64;
                cognitive::CategoryScore {
                    name,
                    containment: categories
                        .iter()
                        .map(|category| category.containment)
                        .sum::<f64>()
                        / count,
                    token_f1: categories
                        .iter()
                        .map(|category| category.token_f1)
                        .sum::<f64>()
                        / count,
                    recall_accuracy: categories
                        .iter()
                        .map(|category| category.recall_accuracy)
                        .sum::<f64>()
                        / count,
                    mrr: categories.iter().map(|category| category.mrr).sum::<f64>() / count,
                    ndcg: categories.iter().map(|category| category.ndcg).sum::<f64>() / count,
                    semantic_similarity: categories
                        .iter()
                        .map(|category| category.semantic_similarity)
                        .sum::<f64>()
                        / count,
                    false_positive_rate: categories
                        .iter()
                        .map(|category| category.false_positive_rate)
                        .sum::<f64>()
                        / count,
                    total: categories[0].total,
                }
            })
            .collect(),
        overall_containment: runs.iter().map(|r| r.overall_containment).sum::<f64>() / n,
        overall_token_f1: runs.iter().map(|r| r.overall_token_f1).sum::<f64>() / n,
        overall_recall_accuracy: runs.iter().map(|r| r.overall_recall_accuracy).sum::<f64>() / n,
        overall_mrr: runs.iter().map(|r| r.overall_mrr).sum::<f64>() / n,
        overall_ndcg: runs.iter().map(|r| r.overall_ndcg).sum::<f64>() / n,
        overall_semantic_similarity: runs
            .iter()
            .map(|r| r.overall_semantic_similarity)
            .sum::<f64>()
            / n,
        false_positive_rate: runs.iter().map(|r| r.false_positive_rate).sum::<f64>() / n,
        execution_latency: average_latency_stats(runs.iter().map(|run| &run.execution_latency)),
        evaluation_latency: average_latency_stats(runs.iter().map(|run| &run.evaluation_latency)),
        end_to_end_latency: average_latency_stats(runs.iter().map(|run| &run.end_to_end_latency)),
        token_cost: average_token_cost(runs),
        total_queries: first.total_queries,
        ingest_time_secs: runs.iter().map(|r| r.ingest_time_secs).sum::<f64>() / n,
        query_time_secs: runs.iter().map(|r| r.query_time_secs).sum::<f64>() / n,
        total_time_secs: runs.iter().map(|r| r.total_time_secs).sum::<f64>() / n,
        compiled_phase_timings: average_compiled_phase_timings(runs),
        baselines: Vec::new(),
        reproducibility: None,
        embedding_cache_miss_count: runs.iter().map(|r| r.embedding_cache_miss_count).sum(),
    }
}

fn average_compiled_phase_timings(
    runs: &[cognitive::CognitiveResult],
) -> Option<cognitive::CompiledPhaseTimingSummary> {
    let timings: Vec<&cognitive::CompiledPhaseTimingSummary> = runs
        .iter()
        .filter_map(|run| run.compiled_phase_timings.as_ref())
        .collect();
    if timings.is_empty() {
        return None;
    }

    Some(cognitive::CompiledPhaseTimingSummary {
        optimize: average_latency_stats(timings.iter().map(|timing| &timing.optimize)),
        physical_plan: average_latency_stats(timings.iter().map(|timing| &timing.physical_plan)),
        execute_plan: average_latency_stats(timings.iter().map(|timing| &timing.execute_plan)),
        embed: average_latency_stats(timings.iter().map(|timing| &timing.embed)),
        decode: average_latency_stats(timings.iter().map(|timing| &timing.decode)),
        assemble: average_latency_stats(timings.iter().map(|timing| &timing.assemble)),
        total: average_latency_stats(timings.iter().map(|timing| &timing.total)),
    })
}

#[derive(Clone)]
struct CognitiveRunBundle {
    primary: cognitive::CognitiveResult,
    baselines: Vec<cognitive::CognitiveResult>,
    active_retrieval_surfaces: cognitive::ActiveRetrievalSurfaces,
    query_embedding_source: cognitive::QueryEmbeddingSource,
    query_embedding_model_label: Option<String>,
}

fn track_query_embedding_runtime(
    expected_source: &mut Option<cognitive::QueryEmbeddingSource>,
    expected_model_label: &mut Option<String>,
    bundle: &CognitiveRunBundle,
) {
    match expected_source {
        None => {
            *expected_source = Some(bundle.query_embedding_source);
            expected_model_label.clone_from(&bundle.query_embedding_model_label);
        }
        Some(source) => {
            assert_eq!(
                *source, bundle.query_embedding_source,
                "benchmark runs used inconsistent query embedding sources"
            );
            assert_eq!(
                expected_model_label.as_deref(),
                bundle.query_embedding_model_label.as_deref(),
                "benchmark runs used inconsistent query embedding model labels"
            );
        }
    }
}

fn execute_cognitive_bundle(
    dataset: &cognitive::CognitiveDataset,
    config: &cognitive::CognitiveConfig,
    run_id: &str,
    embedding_cache: Option<&cognitive::openai::EmbeddingCache>,
    include_baselines: bool,
    subdir: &str,
) -> CognitiveRunBundle {
    let dir = tempfile::tempdir().expect("create temp dir");
    let db_path = dir.path().join(subdir);
    let embedding_runtime =
        cognitive::runner::prepare_benchmark_embedding_runtime(dataset, config, embedding_cache)
            .unwrap_or_else(|error| panic!("prepare benchmark embeddings: {error}"))
            .unwrap_or_else(|| {
                eprintln!(
                    "[hirn-bench] benchmark skipped: no real embedder available and \
             embedder_policy={}; using pseudo embeddings as fallback for bundle execution",
                    config.embedder_policy
                );
                cognitive::runner::BenchmarkEmbeddingRuntime::pseudo()
            });
    let report = cognitive::runner::run_with_prepared_embeddings(
        dataset,
        config,
        &db_path,
        run_id,
        &embedding_runtime,
    );
    let baselines = if include_baselines {
        cognitive::BaselineStrategy::all()
            .iter()
            .map(|&strategy| {
                cognitive::runner::run_baseline_with_prepared_embeddings(
                    dataset,
                    config,
                    run_id,
                    strategy,
                    &embedding_runtime,
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    CognitiveRunBundle {
        primary: report.result,
        baselines,
        active_retrieval_surfaces: report.active_retrieval_surfaces,
        query_embedding_source: report.query_embedding_source,
        query_embedding_model_label: report.query_embedding_model_label,
    }
}

fn average_baseline_results(
    runs: &[CognitiveRunBundle],
    repro_threshold: f64,
) -> Vec<cognitive::CognitiveResult> {
    cognitive::BaselineStrategy::all()
        .iter()
        .filter_map(|strategy| {
            let strategy_runs: Vec<cognitive::CognitiveResult> = runs
                .iter()
                .filter_map(|bundle| {
                    bundle
                        .baselines
                        .iter()
                        .find(|result| result.strategy == strategy.name())
                        .cloned()
                })
                .collect();

            if strategy_runs.is_empty() {
                None
            } else {
                let mut averaged = if strategy_runs.len() == 1 {
                    strategy_runs[0].clone()
                } else {
                    average_cognitive_results(&strategy_runs)
                };
                averaged.reproducibility =
                    cognitive::runner::compute_reproducibility(&strategy_runs, repro_threshold);
                Some(averaged)
            }
        })
        .collect()
}

fn average_latency_stats<'a>(
    stats: impl Iterator<Item = &'a metrics::LatencyStats>,
) -> metrics::LatencyStats {
    let collected: Vec<&metrics::LatencyStats> = stats.collect();
    let n = collected.len().max(1) as f64;
    let average_duration = |selector: fn(&metrics::LatencyStats) -> Duration| {
        Duration::from_secs_f64(
            collected
                .iter()
                .map(|stat| selector(stat).as_secs_f64())
                .sum::<f64>()
                / n,
        )
    };

    metrics::LatencyStats {
        p50: average_duration(|stat| stat.p50),
        p95: average_duration(|stat| stat.p95),
        p99: average_duration(|stat| stat.p99),
        min: average_duration(|stat| stat.min),
        max: average_duration(|stat| stat.max),
        mean: average_duration(|stat| stat.mean),
    }
}

fn average_token_cost(runs: &[cognitive::CognitiveResult]) -> cognitive::TokenCostEstimate {
    let n = runs.len().max(1) as f64;
    cognitive::TokenCostEstimate {
        context_tokens: (runs
            .iter()
            .map(|run| run.token_cost.context_tokens as f64)
            .sum::<f64>()
            / n)
            .round() as usize,
        prompt_tokens: (runs
            .iter()
            .map(|run| run.token_cost.prompt_tokens as f64)
            .sum::<f64>()
            / n)
            .round() as usize,
        completion_tokens: (runs
            .iter()
            .map(|run| run.token_cost.completion_tokens as f64)
            .sum::<f64>()
            / n)
            .round() as usize,
        total_tokens: (runs
            .iter()
            .map(|run| run.token_cost.total_tokens as f64)
            .sum::<f64>()
            / n)
            .round() as usize,
        avg_context_tokens_per_query: runs
            .iter()
            .map(|run| run.token_cost.avg_context_tokens_per_query)
            .sum::<f64>()
            / n,
        avg_prompt_tokens_per_query: runs
            .iter()
            .map(|run| run.token_cost.avg_prompt_tokens_per_query)
            .sum::<f64>()
            / n,
        avg_total_tokens_per_query: runs
            .iter()
            .map(|run| run.token_cost.avg_total_tokens_per_query)
            .sum::<f64>()
            / n,
    }
}

fn run_bench_compare(baseline: String, current: String, threshold: f64, format: String) {
    let baseline_path = std::path::Path::new(&baseline);
    let current_path = std::path::Path::new(&current);

    let baseline_results = compare::load_result_set(baseline_path)
        .unwrap_or_else(|e| panic!("cannot load baseline: {e}"));
    let current_results = compare::load_result_set(current_path)
        .unwrap_or_else(|e| panic!("cannot load current: {e}"));

    let report = compare::compare_result_sets(&baseline_results, &current_results, threshold)
        .unwrap_or_else(|e| panic!("cannot compare artifacts: {e}"));

    match format.as_str() {
        "github" => {
            print!("{}", compare::format_github(&report));
        }
        _ => {
            println!("{report}");
        }
    }

    if report.has_regressions {
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_storage(
    records: usize,
    dims: usize,
    edges: usize,
    bfs_depth: usize,
    bfs_frontier: usize,
    warmup: usize,
    measured: usize,
    limit: usize,
) {
    let config = storage::StorageBenchConfig {
        num_records: records,
        dims,
        warmup,
        measured,
        limit,
        num_edges: edges,
        bfs_depth,
        bfs_frontier,
    };

    eprintln!("Running storage benchmarks:");
    eprintln!(
        "  records={records} dims={dims} edges={edges} bfs_depth={bfs_depth} frontier={bfs_frontier} warmup={warmup} measured={measured} limit={limit}"
    );

    let result = storage::run(&config);
    println!("{}", storage::format_markdown(&result));
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn output_targets_reject_duplicate_paths() {
        let temp = tempfile::tempdir().unwrap();
        let shared_path = temp.path().join("artifact.json");
        let shared_path = shared_path.to_string_lossy().into_owned();

        let error = OutputTargets::from_args(ArtifactOutputArgs {
            format: "json".to_string(),
            output: Some(shared_path.clone()),
            json_output: Some(shared_path),
            csv_output: None,
            markdown_output: None,
        })
        .unwrap_err();

        assert!(error.contains("duplicate output path"));
    }

    #[test]
    fn emit_outputs_writes_primary_and_companion_files() {
        let temp = tempfile::tempdir().unwrap();
        let markdown_path = temp.path().join("artifact.md");
        let json_path = temp.path().join("artifact.json");

        let outputs = OutputTargets::from_args(ArtifactOutputArgs {
            format: "markdown".to_string(),
            output: Some(markdown_path.to_string_lossy().into_owned()),
            json_output: Some(json_path.to_string_lossy().into_owned()),
            csv_output: None,
            markdown_output: None,
        })
        .unwrap();

        emit_outputs(&outputs, &"payload", |result, format, writer| {
            writeln!(writer, "{format:?}:{result}")
        });

        assert_eq!(
            fs::read_to_string(markdown_path).unwrap(),
            "Markdown:payload\n"
        );
        assert_eq!(fs::read_to_string(json_path).unwrap(), "Json:payload\n");
    }

    #[test]
    fn query_embedding_artifact_label_uses_corpus_model_for_cache_runtime() {
        let label = resolve_query_embedding_model_label_for_artifact(
            "cache:embeddings/locomo_embeddings.json",
            "text-embedding-3-small",
            cognitive::QueryEmbeddingSource::Cache,
            None,
        )
        .unwrap();

        assert_eq!(label, "text-embedding-3-small");
    }

    #[test]
    fn query_embedding_artifact_label_rejects_mixed_cache_and_pseudo_runtime() {
        let error = resolve_query_embedding_model_label_for_artifact(
            "cache:embeddings/locomo_embeddings.json",
            "text-embedding-3-small",
            cognitive::QueryEmbeddingSource::Pseudo,
            None,
        )
        .unwrap_err();

        assert!(error.contains("corpus embeddings came from"));
        assert!(error.contains("query embeddings came from `pseudo`"));
    }

    #[test]
    fn query_embedding_artifact_label_accepts_provider_runtime_when_corpus_is_provider() {
        let label = resolve_query_embedding_model_label_for_artifact(
            "provider",
            "text-embedding-3-large",
            cognitive::QueryEmbeddingSource::Provider,
            Some("text-embedding-3-large"),
        )
        .unwrap();

        assert_eq!(label, "text-embedding-3-large");
    }
}
