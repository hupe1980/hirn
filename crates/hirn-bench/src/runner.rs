//! Benchmark runner — ingests dataset, executes recall/think, collects metrics.
//!
//! Ingestion uses `batch_remember()` and `batch_store_semantic()` to avoid
//! O(n) Lance fragments. After ingestion, the datasets are compacted so that
//! recall/think benchmarks run against properly merged data.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hirn::{HirnConfig, HirnDB};
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

use crate::dataset::{self, SyntheticDataset};

/// Run an async future to completion on a shared tokio runtime.
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    static RT: std::sync::LazyLock<tokio::runtime::Runtime> = std::sync::LazyLock::new(|| {
        tokio::runtime::Runtime::new().expect("tokio runtime for bench")
    });
    RT.block_on(f)
}
use crate::metrics::{
    AggregateQuality, BenchmarkConfig, BenchmarkResult, QueryMetrics, ThroughputStats, f1_score,
    latency_percentiles, mrr, ndcg_at_k, precision_at_k, recall_at_k,
};

/// Get current process RSS in bytes (safe, no libc).
fn rss_bytes() -> u64 {
    let pid = std::process::id();
    #[cfg(target_os = "macos")]
    {
        // Use `ps` to read RSS (in KB) for our PID.
        std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|kb| kb * 1024)
            .unwrap_or(0)
    }
    #[cfg(target_os = "linux")]
    {
        let _ = pid;
        std::fs::read_to_string("/proc/self/statm")
            .ok()
            .and_then(|s| s.split_whitespace().nth(1)?.parse::<u64>().ok())
            .map(|pages| pages * 4096)
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = pid;
        0
    }
}

/// Execute a full benchmark suite.
pub fn run(config: &BenchmarkConfig, db_path: &Path, run_id: &str) -> BenchmarkResult {
    let total_start = Instant::now();

    // Generate dataset.
    let ds = dataset::generate(config);

    // Open DB with LanceDB storage backend.
    let lance_path = db_path.parent().unwrap_or(db_path).join("lance_brain");
    let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
    let backend: Arc<dyn PhysicalStore> = block_on(HirnDb::open(storage_config))
        .expect("open HirnDb")
        .store_arc();

    let hirn_config = HirnConfig::builder()
        .db_path(db_path)
        .embedding_dimensions(config.embedding_dims as u32)
        .token_budget(config.token_budget as u32)
        .build()
        .expect("valid config");
    let db = block_on(HirnDB::open_with_config(hirn_config, backend)).expect("open db");

    // ── Phase 1: Ingest (batched) ──────────────────────────────
    eprintln!(
        "  Phase 1/3: Ingesting {} episodic + {} semantic records (batched)...",
        ds.episodic_records.len(),
        ds.semantic_records.len()
    );
    let remember_latencies = ingest(&db, &ds);
    eprintln!("  Phase 1/3: Done.");

    // ── Phase 2: Recall ──────────────────────────────────────
    eprintln!(
        "  Phase 2/3: Running recall ({} queries × {} runs)...",
        ds.queries.len(),
        config.measured_runs
    );
    let (recall_latencies, query_metrics) = bench_recall(&db, &ds, config);
    eprintln!("  Phase 2/3: Done.");

    // ── Phase 3: Think ───────────────────────────────────────
    eprintln!(
        "  Phase 3/3: Running think ({} queries × {} runs)...",
        ds.queries.len(),
        config.measured_runs
    );
    let think_latencies = bench_think(&db, &ds, config);
    eprintln!("  Phase 3/3: Done.");

    // ── Collect stats ────────────────────────────────────────
    let peak_mem = rss_bytes();
    let db_file_size = block_on(db.admin().stats())
        .map(|s| s.file_size_bytes)
        .unwrap_or(0);
    let total_time = total_start.elapsed();

    let aggregate = AggregateQuality::from_queries(&query_metrics);

    let total_inserts = ds.episodic_records.len() + ds.semantic_records.len();
    let ingest_secs = remember_latencies
        .iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>();
    let recall_secs = recall_latencies
        .iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>();
    let think_secs = think_latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>();

    BenchmarkResult {
        suite_name: "synthetic".to_string(),
        run_id: run_id.to_string(),
        config: config.clone(),
        query_metrics,
        aggregate,
        remember_latency: latency_percentiles(&sorted(remember_latencies)),
        recall_latency: latency_percentiles(&sorted(recall_latencies)),
        think_latency: latency_percentiles(&sorted(think_latencies)),
        throughput: ThroughputStats {
            remember_ops_per_sec: if ingest_secs > 0.0 {
                total_inserts as f64 / ingest_secs
            } else {
                0.0
            },
            recall_ops_per_sec: if recall_secs > 0.0 {
                ds.queries.len() as f64 / recall_secs
            } else {
                0.0
            },
            think_ops_per_sec: if think_secs > 0.0 {
                ds.queries.len() as f64 / think_secs
            } else {
                0.0
            },
        },
        peak_memory_bytes: peak_mem,
        db_file_size_bytes: db_file_size,
        total_time,
    }
}

fn ingest(db: &HirnDB, ds: &SyntheticDataset) -> Vec<Duration> {
    let ep_count = ds.episodic_records.len();
    let sem_count = ds.semantic_records.len();
    let mut latencies = Vec::with_capacity(2);

    // Batch episodic records — single Lance fragment instead of O(n).
    let start = Instant::now();
    let results = block_on(db.episodic().batch_remember(ds.episodic_records.clone()));
    let ep_dur = start.elapsed();
    let ep_failures: Vec<_> = results.iter().filter(|r| r.is_err()).collect();
    if !ep_failures.is_empty() {
        eprintln!(
            "    WARNING: {}/{} episodic records failed: {:?}",
            ep_failures.len(),
            ep_count,
            ep_failures.first().unwrap().as_ref().unwrap_err()
        );
    }
    latencies.push(ep_dur);

    // Batch semantic records — single Lance fragment.
    let start = Instant::now();
    let results = block_on(db.semantic().batch_store(ds.semantic_records.clone()));
    let sem_dur = start.elapsed();
    let sem_failures: Vec<_> = results.iter().filter(|r| r.is_err()).collect();
    if !sem_failures.is_empty() {
        eprintln!(
            "    WARNING: {}/{} semantic records failed: {:?}",
            sem_failures.len(),
            sem_count,
            sem_failures.first().unwrap().as_ref().unwrap_err()
        );
    }
    latencies.push(sem_dur);

    eprintln!(
        "    Batched ingest: {} episodic in {:.2}ms, {} semantic in {:.2}ms",
        ep_count,
        ep_dur.as_secs_f64() * 1000.0,
        sem_count,
        sem_dur.as_secs_f64() * 1000.0
    );

    latencies
}

fn bench_recall(
    db: &HirnDB,
    ds: &SyntheticDataset,
    config: &BenchmarkConfig,
) -> (Vec<Duration>, Vec<QueryMetrics>) {
    let mut latencies = Vec::new();
    let mut metrics = Vec::new();

    // Warmup.
    for q in ds.queries.iter().take(config.warmup_runs) {
        let _ = block_on(
            db.recall_view()
                .query(q.embedding.clone())
                .limit(config.k)
                .execute(),
        );
    }

    // Measured runs.
    for _run in 0..config.measured_runs {
        for q in &ds.queries {
            let start = Instant::now();
            let results = db
                .recall_view()
                .query(q.embedding.clone())
                .limit(config.k)
                .execute();
            let results = block_on(results).expect("recall");
            latencies.push(start.elapsed());

            // Compute retrieval quality metrics.
            let retrieved_ids: Vec<String> =
                results.iter().map(|r| r.record.id().to_string()).collect();
            let relevant_ids: Vec<String> =
                q.relevant_ids.iter().map(|id| id.to_string()).collect();

            let retrieved_refs: Vec<&str> = retrieved_ids.iter().map(|s| s.as_str()).collect();
            let relevant_refs: Vec<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();

            let p = precision_at_k(&retrieved_refs, &relevant_refs, config.k);
            let r = recall_at_k(&retrieved_refs, &relevant_refs, config.k);
            let f1 = f1_score(p, r);
            let m = mrr(&retrieved_refs, &relevant_refs);
            let n = ndcg_at_k(&retrieved_refs, &relevant_refs, config.k);

            metrics.push(QueryMetrics {
                precision_at_k: p,
                recall_at_k: r,
                f1,
                mrr: m,
                ndcg_at_k: n,
            });
        }
    }

    (latencies, metrics)
}

fn bench_think(db: &HirnDB, ds: &SyntheticDataset, config: &BenchmarkConfig) -> Vec<Duration> {
    let mut latencies = Vec::new();

    // Warmup.
    for q in ds.queries.iter().take(config.warmup_runs) {
        let _ = block_on(
            db.recall_view()
                .think(q.embedding.clone())
                .budget(config.token_budget)
                .execute(),
        );
    }

    // Measured runs.
    for _run in 0..config.measured_runs {
        for q in &ds.queries {
            let start = Instant::now();
            let _ = block_on(
                db.recall_view()
                    .think(q.embedding.clone())
                    .budget(config.token_budget)
                    .execute(),
            )
            .expect("think");
            latencies.push(start.elapsed());
        }
    }

    latencies
}

fn sorted(mut v: Vec<Duration>) -> Vec<Duration> {
    v.sort();
    v
}
