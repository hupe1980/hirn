use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::{Duration, Instant};

use hirn::{HirnConfig, HirnDB};
use hirn_core::episodic::EpisodicRecord;
use hirn_core::types::{AgentId, EventType};
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use metrics_util::{CompositeKey, MetricKind};
use serde::Serialize;
use tokio::sync::Barrier;

use crate::metrics::{LatencyStats, latency_percentiles};
use crate::output::OutputFormat;

type DebugSnapshotEntry = (
    CompositeKey,
    Option<metrics::Unit>,
    Option<metrics::SharedString>,
    DebugValue,
);

static BENCH_SNAPSHOTTER: OnceLock<Option<Snapshotter>> = OnceLock::new();

const BATCH_REMEMBER_STAGE_ORDER: &[&str] = &[
    "authorize",
    "admission",
    "embedding",
    "prepare",
    "auto_edge_prefetch",
    "graph_prepare",
    "append",
    "temporal_next",
    "events",
    "slow_path",
];

fn benchmark_snapshotter() -> Option<&'static Snapshotter> {
    BENCH_SNAPSHOTTER
        .get_or_init(|| {
            let recorder = DebuggingRecorder::new();
            let snapshotter = recorder.snapshotter();
            metrics::set_global_recorder(recorder).ok()?;
            Some(snapshotter)
        })
        .as_ref()
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    static RT: LazyLock<tokio::runtime::Runtime> =
        LazyLock::new(|| tokio::runtime::Runtime::new().expect("tokio runtime for load bench"));
    RT.block_on(future)
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadConfig {
    pub writers: usize,
    pub readers: usize,
    pub writes_per_writer: usize,
    pub writer_batch_size: usize,
    pub max_auto_edges_per_record: usize,
    pub reads_per_reader: usize,
    pub preseed_records: usize,
    pub embedding_dims: usize,
    pub k: usize,
}

impl LoadConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.writers == 0 {
            return Err("writers must be greater than zero".into());
        }
        if self.readers == 0 {
            return Err("readers must be greater than zero".into());
        }
        if self.writes_per_writer == 0 {
            return Err("writes_per_writer must be greater than zero".into());
        }
        if self.writer_batch_size == 0 {
            return Err("writer_batch_size must be greater than zero".into());
        }
        if self.reads_per_reader == 0 {
            return Err("reads_per_reader must be greater than zero".into());
        }
        if self.preseed_records == 0 {
            return Err("preseed_records must be greater than zero".into());
        }
        if self.embedding_dims == 0 {
            return Err("embedding_dims must be greater than zero".into());
        }
        if self.k == 0 {
            return Err("k must be greater than zero".into());
        }
        Ok(())
    }
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            writers: 4,
            readers: 8,
            writes_per_writer: 50,
            writer_batch_size: 16,
            max_auto_edges_per_record: 0,
            reads_per_reader: 100,
            preseed_records: 128,
            embedding_dims: 64,
            k: 10,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LoadThroughput {
    pub remember_ops_per_sec: f64,
    pub recall_ops_per_sec: f64,
    pub total_ops_per_sec: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadResult {
    pub run_id: String,
    pub config: LoadConfig,
    pub remember_latency: LatencyStats,
    pub recall_latency: LatencyStats,
    pub batch_remember_stage_latency: Vec<LoadStageLatency>,
    pub throughput: LoadThroughput,
    pub writer_failures: usize,
    pub reader_failures: usize,
    pub peak_memory_bytes: u64,
    pub db_file_size_bytes: u64,
    #[serde(serialize_with = "serialize_duration_us")]
    pub preseed_time: Duration,
    #[serde(serialize_with = "serialize_duration_us")]
    pub load_time: Duration,
    #[serde(serialize_with = "serialize_duration_us")]
    pub total_time: Duration,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadStageLatency {
    pub stage: String,
    pub observations: usize,
    pub latency: LatencyStats,
}

enum WorkerResult {
    Writer {
        latencies: Vec<Duration>,
        failures: usize,
    },
    Reader {
        latencies: Vec<Duration>,
        failures: usize,
    },
}

pub fn run(config: &LoadConfig, db_path: &Path, run_id: &str) -> Result<LoadResult, String> {
    config.validate()?;

    let total_start = Instant::now();
    let snapshotter = benchmark_snapshotter();
    let lance_path = db_path.parent().unwrap_or(db_path).join("lance_load");
    let backend: Arc<dyn PhysicalStore> = block_on(HirnDb::open(HirnDbConfig::local(
        lance_path
            .to_str()
            .ok_or_else(|| "invalid db path".to_string())?,
    )))
    .map_err(|error| error.to_string())?
    .store_arc();

    let hirn_config = HirnConfig::builder()
        .db_path(db_path)
        .embedding_dimensions(config.embedding_dims as u32)
        .token_budget(4096)
        .max_auto_edges_per_record(config.max_auto_edges_per_record)
        .build()
        .map_err(|error| error.to_string())?;
    let db = Arc::new(
        block_on(HirnDB::open_with_config(hirn_config, backend))
            .map_err(|error| error.to_string())?,
    );

    let preseed_start = Instant::now();
    let preseed_results = block_on(db.episodic().batch_remember(build_records(
        "preseed",
        0,
        config.preseed_records,
        config.embedding_dims,
    )));
    let preseed_failures = preseed_results
        .iter()
        .filter(|result| result.is_err())
        .count();
    if preseed_failures > 0 {
        return Err(format!(
            "preseed failed for {preseed_failures} records during load benchmark"
        ));
    }
    let preseed_time = preseed_start.elapsed();
    let post_preseed_metrics = snapshotter.map(|s| s.snapshot().into_vec());

    let query_embeddings = Arc::new(build_query_embeddings(
        config.preseed_records,
        config.embedding_dims,
    ));
    let concurrent_workers = config.writers + config.readers;

    let load_start = Instant::now();
    let (remember_latencies, recall_latencies, writer_failures, reader_failures) = block_on({
        let db = Arc::clone(&db);
        let query_embeddings = Arc::clone(&query_embeddings);
        async move {
            let barrier = Arc::new(Barrier::new(concurrent_workers));
            let mut handles = Vec::with_capacity(concurrent_workers);

            for writer_idx in 0..config.writers {
                let db = Arc::clone(&db);
                let barrier = Arc::clone(&barrier);
                let dims = config.embedding_dims;
                let writes_per_worker = config.writes_per_writer;
                let writer_batch_size = config.writer_batch_size;
                let preseed_records = config.preseed_records;
                handles.push(tokio::spawn(async move {
                    barrier.wait().await;

                    let mut latencies = Vec::with_capacity(writes_per_worker);
                    let mut failures = 0usize;
                    let effective_batch_size = writer_batch_size.min(writes_per_worker.max(1));
                    for offset in (0..writes_per_worker).step_by(effective_batch_size) {
                        let batch_len = (writes_per_worker - offset).min(effective_batch_size);
                        let records = build_records(
                            "writer",
                            preseed_records + writer_idx * writes_per_worker + offset,
                            batch_len,
                            dims,
                        );
                        let start = Instant::now();
                        let results = db.episodic().batch_remember(records).await;
                        let batch_elapsed = start.elapsed();
                        let success_count = results.iter().filter(|result| result.is_ok()).count();
                        failures += results.len().saturating_sub(success_count);
                        if success_count > 0 {
                            let amortized = batch_elapsed / success_count as u32;
                            latencies.extend(std::iter::repeat_n(amortized, success_count));
                        }
                    }

                    WorkerResult::Writer {
                        latencies,
                        failures,
                    }
                }));
            }

            for reader_idx in 0..config.readers {
                let db = Arc::clone(&db);
                let barrier = Arc::clone(&barrier);
                let query_embeddings = Arc::clone(&query_embeddings);
                let reads_per_reader = config.reads_per_reader;
                let k = config.k;
                handles.push(tokio::spawn(async move {
                    barrier.wait().await;

                    let mut latencies = Vec::with_capacity(reads_per_reader);
                    let mut failures = 0usize;
                    for offset in 0..reads_per_reader {
                        let query = query_embeddings
                            [(reader_idx + offset) % query_embeddings.len()]
                        .clone();
                        let start = Instant::now();
                        match db.recall_view().query(query).limit(k).execute().await {
                            Ok(_) => latencies.push(start.elapsed()),
                            Err(e) => {
                                tracing::warn!(error = %e, "load-bench reader failure");
                                failures += 1;
                            }
                        }
                    }

                    WorkerResult::Reader {
                        latencies,
                        failures,
                    }
                }));
            }

            let mut remember_latencies = Vec::new();
            let mut recall_latencies = Vec::new();
            let mut writer_failures = 0usize;
            let mut reader_failures = 0usize;

            for handle in handles {
                match handle.await.expect("load worker should not panic") {
                    WorkerResult::Writer {
                        latencies,
                        failures,
                    } => {
                        remember_latencies.extend(latencies);
                        writer_failures += failures;
                    }
                    WorkerResult::Reader {
                        latencies,
                        failures,
                    } => {
                        recall_latencies.extend(latencies);
                        reader_failures += failures;
                    }
                }
            }

            (
                remember_latencies,
                recall_latencies,
                writer_failures,
                reader_failures,
            )
        }
    });
    let load_time = load_start.elapsed();
    let post_load_metrics = snapshotter.map(|s| s.snapshot().into_vec());

    let peak_memory_bytes = rss_bytes();
    let db_file_size_bytes = block_on(db.admin().stats())
        .map(|stats| stats.file_size_bytes)
        .unwrap_or(0);

    let remember_ops = config.writers * config.writes_per_writer;
    let recall_ops = config.readers * config.reads_per_reader;
    let load_secs = load_time.as_secs_f64();

    Ok(LoadResult {
        run_id: run_id.to_string(),
        config: config.clone(),
        remember_latency: compute_latency_stats(remember_latencies),
        recall_latency: compute_latency_stats(recall_latencies),
        batch_remember_stage_latency: compute_batch_remember_stage_latency(
            post_preseed_metrics.as_deref(),
            post_load_metrics.as_deref(),
        ),
        throughput: LoadThroughput {
            remember_ops_per_sec: if load_secs > 0.0 {
                remember_ops as f64 / load_secs
            } else {
                0.0
            },
            recall_ops_per_sec: if load_secs > 0.0 {
                recall_ops as f64 / load_secs
            } else {
                0.0
            },
            total_ops_per_sec: if load_secs > 0.0 {
                (remember_ops + recall_ops) as f64 / load_secs
            } else {
                0.0
            },
        },
        writer_failures,
        reader_failures,
        peak_memory_bytes,
        db_file_size_bytes,
        preseed_time,
        load_time,
        total_time: total_start.elapsed(),
    })
}

pub fn write_result(
    result: &LoadResult,
    format: OutputFormat,
    writer: &mut dyn Write,
) -> io::Result<()> {
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(result)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
            writeln!(writer, "{json}")
        }
        OutputFormat::Csv => write_csv(result, writer),
        OutputFormat::Markdown => write_markdown(result, writer),
    }
}

fn write_csv(result: &LoadResult, writer: &mut dyn Write) -> io::Result<()> {
    writeln!(
        writer,
        "run_id,writers,readers,writes_per_writer,writer_batch_size,max_auto_edges_per_record,reads_per_reader,preseed_records,dims,k,remember_p50_us,remember_p95_us,remember_p99_us,recall_p50_us,recall_p95_us,recall_p99_us,remember_ops_sec,recall_ops_sec,total_ops_sec,writer_failures,reader_failures,peak_memory_bytes,db_file_size_bytes,preseed_time_us,load_time_us,total_time_us"
    )?;
    writeln!(
        writer,
        "{},{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{:.1},{},{},{},{},{:.0},{:.0},{:.0}",
        result.run_id,
        result.config.writers,
        result.config.readers,
        result.config.writes_per_writer,
        result.config.writer_batch_size,
        result.config.max_auto_edges_per_record,
        result.config.reads_per_reader,
        result.config.preseed_records,
        result.config.embedding_dims,
        result.config.k,
        result.remember_latency.p50.as_secs_f64() * 1e6,
        result.remember_latency.p95.as_secs_f64() * 1e6,
        result.remember_latency.p99.as_secs_f64() * 1e6,
        result.recall_latency.p50.as_secs_f64() * 1e6,
        result.recall_latency.p95.as_secs_f64() * 1e6,
        result.recall_latency.p99.as_secs_f64() * 1e6,
        result.throughput.remember_ops_per_sec,
        result.throughput.recall_ops_per_sec,
        result.throughput.total_ops_per_sec,
        result.writer_failures,
        result.reader_failures,
        result.peak_memory_bytes,
        result.db_file_size_bytes,
        result.preseed_time.as_secs_f64() * 1e6,
        result.load_time.as_secs_f64() * 1e6,
        result.total_time.as_secs_f64() * 1e6,
    )
}

fn write_markdown(result: &LoadResult, writer: &mut dyn Write) -> io::Result<()> {
    writeln!(writer, "# Concurrent Load Report")?;
    writeln!(writer)?;
    writeln!(writer, "**Run ID:** {}", result.run_id)?;
    writeln!(writer, "**Writers:** {}", result.config.writers)?;
    writeln!(writer, "**Readers:** {}", result.config.readers)?;
    writeln!(
        writer,
        "**Writes/Writer:** {}",
        result.config.writes_per_writer
    )?;
    writeln!(
        writer,
        "**Writer Batch Size:** {}",
        result.config.writer_batch_size
    )?;
    writeln!(
        writer,
        "**Max Auto Edges/Record:** {}",
        result.config.max_auto_edges_per_record
    )?;
    writeln!(
        writer,
        "**Reads/Reader:** {}",
        result.config.reads_per_reader
    )?;
    writeln!(
        writer,
        "**Preseed Records:** {}",
        result.config.preseed_records
    )?;
    writeln!(
        writer,
        "**Embedding Dims:** {}",
        result.config.embedding_dims
    )?;
    writeln!(writer, "**Top-K:** {}", result.config.k)?;
    writeln!(writer)?;

    writeln!(writer, "## Latency (µs)")?;
    writeln!(writer)?;
    writeln!(writer, "| Operation | p50 | p95 | p99 | min | max | mean |")?;
    writeln!(writer, "|-----------|----:|----:|----:|----:|----:|-----:|")?;
    for (name, stats) in [
        ("remember", &result.remember_latency),
        ("recall", &result.recall_latency),
    ] {
        writeln!(
            writer,
            "| {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |",
            name,
            stats.p50.as_secs_f64() * 1e6,
            stats.p95.as_secs_f64() * 1e6,
            stats.p99.as_secs_f64() * 1e6,
            stats.min.as_secs_f64() * 1e6,
            stats.max.as_secs_f64() * 1e6,
            stats.mean.as_secs_f64() * 1e6,
        )?;
    }
    writeln!(writer)?;

    if !result.batch_remember_stage_latency.is_empty() {
        writeln!(writer, "## Batch Remember Stages (µs)")?;
        writeln!(writer)?;
        writeln!(writer, "| Stage | observations | p50 | p95 | p99 | mean |")?;
        writeln!(writer, "|-------|-------------:|----:|----:|----:|-----:|")?;
        for stage in &result.batch_remember_stage_latency {
            writeln!(
                writer,
                "| {} | {} | {:.1} | {:.1} | {:.1} | {:.1} |",
                stage.stage,
                stage.observations,
                stage.latency.p50.as_secs_f64() * 1e6,
                stage.latency.p95.as_secs_f64() * 1e6,
                stage.latency.p99.as_secs_f64() * 1e6,
                stage.latency.mean.as_secs_f64() * 1e6,
            )?;
        }
        writeln!(writer)?;
    }

    writeln!(writer, "## Throughput")?;
    writeln!(writer)?;
    writeln!(writer, "| Metric | Value |")?;
    writeln!(writer, "|--------|------:|")?;
    writeln!(
        writer,
        "| remember ops/sec | {:.1} |",
        result.throughput.remember_ops_per_sec
    )?;
    writeln!(
        writer,
        "| recall ops/sec | {:.1} |",
        result.throughput.recall_ops_per_sec
    )?;
    writeln!(
        writer,
        "| total ops/sec | {:.1} |",
        result.throughput.total_ops_per_sec
    )?;
    writeln!(writer)?;

    writeln!(writer, "## Failures")?;
    writeln!(writer)?;
    writeln!(writer, "| Worker Type | Count |")?;
    writeln!(writer, "|-------------|------:|")?;
    writeln!(writer, "| writers | {} |", result.writer_failures)?;
    writeln!(writer, "| readers | {} |", result.reader_failures)?;
    writeln!(writer)?;

    writeln!(writer, "## Resource Usage")?;
    writeln!(writer)?;
    writeln!(writer, "| Metric | Value |")?;
    writeln!(writer, "|--------|------:|")?;
    writeln!(
        writer,
        "| Peak RSS | {:.2} MB |",
        result.peak_memory_bytes as f64 / 1_048_576.0
    )?;
    writeln!(
        writer,
        "| DB file size | {:.2} MB |",
        result.db_file_size_bytes as f64 / 1_048_576.0
    )?;
    writeln!(
        writer,
        "| Preseed time | {:.2} s |",
        result.preseed_time.as_secs_f64()
    )?;
    writeln!(
        writer,
        "| Load window | {:.2} s |",
        result.load_time.as_secs_f64()
    )?;
    writeln!(
        writer,
        "| Total time | {:.2} s |",
        result.total_time.as_secs_f64()
    )?;
    writeln!(writer)
}

fn build_records(prefix: &str, start: usize, count: usize, dims: usize) -> Vec<EpisodicRecord> {
    (0..count)
        .map(|offset| build_record(prefix, start + offset, dims).expect("valid load record"))
        .collect()
}

fn build_record(prefix: &str, index: usize, dims: usize) -> Result<EpisodicRecord, String> {
    EpisodicRecord::builder()
        .content(format!("{prefix}-memory-{index}"))
        .event_type(EventType::Observation)
        .importance(0.5)
        .agent_id(load_agent())
        .embedding(embedding_for(index, dims))
        .build()
        .map_err(|error| error.to_string())
}

fn build_query_embeddings(query_count: usize, dims: usize) -> Vec<Vec<f32>> {
    (0..query_count.max(1))
        .map(|index| embedding_for(index, dims))
        .collect()
}

fn embedding_for(index: usize, dims: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; dims];
    let primary = index % dims;
    let secondary = (index / 7 + 1) % dims;
    embedding[primary] = 1.0;
    if secondary != primary {
        embedding[secondary] = 0.25;
    }
    embedding
}

fn load_agent() -> AgentId {
    AgentId::new("load-bench").expect("load benchmark agent id must be valid")
}

fn compute_latency_stats(mut latencies: Vec<Duration>) -> LatencyStats {
    latencies.sort();
    latency_percentiles(&latencies)
}

fn compute_batch_remember_stage_latency(
    baseline: Option<&[DebugSnapshotEntry]>,
    current: Option<&[DebugSnapshotEntry]>,
) -> Vec<LoadStageLatency> {
    let Some(current) = current else {
        return Vec::new();
    };

    let delta = batch_stage_latency_from_snapshot_diff(baseline, current);
    if !delta.is_empty() {
        return delta;
    }

    batch_stage_latency_from_snapshot(current)
}

fn batch_stage_latency_from_snapshot_diff(
    baseline: Option<&[DebugSnapshotEntry]>,
    current: &[DebugSnapshotEntry],
) -> Vec<LoadStageLatency> {
    BATCH_REMEMBER_STAGE_ORDER
        .iter()
        .filter_map(|stage| {
            let baseline_count = baseline
                .map(|baseline| batch_stage_observations(baseline, stage).len())
                .unwrap_or(0);
            let observations = batch_stage_observations(current, stage);
            let delta = observations
                .into_iter()
                .skip(baseline_count)
                .collect::<Vec<_>>();
            if delta.is_empty() {
                return None;
            }

            Some(LoadStageLatency {
                stage: (*stage).to_string(),
                observations: delta.len(),
                latency: compute_latency_stats(delta),
            })
        })
        .collect()
}

fn batch_stage_latency_from_snapshot(current: &[DebugSnapshotEntry]) -> Vec<LoadStageLatency> {
    BATCH_REMEMBER_STAGE_ORDER
        .iter()
        .filter_map(|stage| {
            let observations = batch_stage_observations(current, stage);
            if observations.is_empty() {
                return None;
            }

            Some(LoadStageLatency {
                stage: (*stage).to_string(),
                observations: observations.len(),
                latency: compute_latency_stats(observations),
            })
        })
        .collect()
}

fn batch_stage_observations(snapshot: &[DebugSnapshotEntry], stage: &str) -> Vec<Duration> {
    snapshot
        .iter()
        .filter(|(key, _, _, _)| {
            key.kind() == MetricKind::Histogram
                && key.key().name() == hirn_engine::metrics::BATCH_REMEMBER_STAGE_DURATION_SECONDS
                && key
                    .key()
                    .labels()
                    .any(|label| label.key() == "stage" && label.value() == stage)
        })
        .flat_map(|(_, _, _, value)| match value {
            DebugValue::Histogram(values) => values
                .iter()
                .filter_map(|sample| sample.to_string().parse::<f64>().ok())
                .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
                .map(Duration::from_secs_f64)
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

fn rss_bytes() -> u64 {
    let pid = std::process::id();
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|stdout| stdout.trim().parse::<u64>().ok())
            .map(|kilobytes| kilobytes * 1024)
            .unwrap_or(0)
    }
    #[cfg(target_os = "linux")]
    {
        let _ = pid;
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|contents| contents.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map(|pages| pages * 4096)
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = pid;
        0
    }
}

fn serialize_duration_us<S: serde::Serializer>(
    duration: &Duration,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_f64(duration.as_secs_f64() * 1_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_zero_concurrency() {
        let config = LoadConfig {
            writers: 0,
            ..LoadConfig::default()
        };

        assert!(config.validate().is_err());
    }

    #[test]
    fn markdown_output_includes_load_sections() {
        let result = LoadResult {
            run_id: "load-001".into(),
            config: LoadConfig::default(),
            remember_latency: LatencyStats::default(),
            recall_latency: LatencyStats::default(),
            batch_remember_stage_latency: vec![LoadStageLatency {
                stage: "append".into(),
                observations: 2,
                latency: LatencyStats::default(),
            }],
            throughput: LoadThroughput::default(),
            writer_failures: 0,
            reader_failures: 0,
            peak_memory_bytes: 1024,
            db_file_size_bytes: 2048,
            preseed_time: Duration::from_millis(10),
            load_time: Duration::from_millis(20),
            total_time: Duration::from_millis(30),
        };

        let mut buffer = Vec::new();
        write_result(&result, OutputFormat::Markdown, &mut buffer).unwrap();
        let markdown = String::from_utf8(buffer).unwrap();
        assert!(markdown.contains("# Concurrent Load Report"));
        assert!(markdown.contains("## Latency"));
        assert!(markdown.contains("## Batch Remember Stages"));
        assert!(markdown.contains("## Throughput"));
        assert!(markdown.contains("## Failures"));
        assert!(markdown.contains("Writer Batch Size"));
    }
}
