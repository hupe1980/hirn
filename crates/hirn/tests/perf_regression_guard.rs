//! Performance Regression Guard (BACKLOG10 Story 5.3)
//!
//! Lightweight synthetic benchmark that measures store and recall latency.
//! Asserts that operations complete within reasonable bounds to catch
//! major regressions.
//!
//! These tests are `#[ignore]`d by default — they run on explicit opt-in:
//! ```sh
//! cargo test -p hirn --test perf_regression_guard -- --ignored
//! ```
//!
//! For full benchmarks use hirn-bench.

use std::time::Instant;

use hirn::prelude::*;

const NUM_STORE: usize = 50;
const NUM_QUERIES: usize = 20;

/// Diverse memories for benchmarking.
fn bench_memories() -> Vec<String> {
    (0..NUM_STORE)
        .map(|i| {
            format!(
                "Benchmark memory {} about {} covering {} with unique content variation {}.",
                i,
                ["biology", "physics", "history", "computing", "economics"][i % 5],
                [
                    "cellular processes and mitochondrial function",
                    "quantum field theory and particle interactions",
                    "medieval European political developments",
                    "distributed consensus algorithms and replication",
                    "macroeconomic market equilibrium theory"
                ][i % 5],
                i
            )
        })
        .collect()
}

async fn open_memory() -> (HirnMemory, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("perf-guard");
    let mut config = HirnConfig::builder()
        .db_path(&path)
        .allow_pseudo_embedder_fallback(true)
        .build()
        .unwrap();
    config.admission_enabled = false;
    let mem = HirnMemory::open_with_config(config).await.unwrap();
    (mem, dir)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "perf regression guard — run with --ignored"]
async fn perf_guard_store_throughput() {
    let (mem, _dir) = open_memory().await;
    let memories = bench_memories();

    let start = Instant::now();
    for text in &memories {
        mem.remember(text).await.unwrap();
    }
    let elapsed = start.elapsed();

    let ms_per_op = elapsed.as_millis() as f64 / NUM_STORE as f64;
    eprintln!(
        "Store: {} records in {:.1}s ({:.1} ms/op)",
        NUM_STORE,
        elapsed.as_secs_f64(),
        ms_per_op,
    );

    // Each store should take < 2s on average in debug mode.
    assert!(
        ms_per_op < 2000.0,
        "Store latency regression: {ms_per_op:.1}ms/op exceeds 2000ms threshold"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "perf regression guard — run with --ignored"]
async fn perf_guard_recall_latency() {
    let (mem, _dir) = open_memory().await;

    // Seed data
    for text in &bench_memories() {
        mem.remember(text).await.unwrap();
    }

    let queries = [
        "biological cell energy production",
        "quantum mechanics wave duality",
        "medieval historical events",
        "distributed computing consensus",
        "economic market theory",
    ];

    let mut latencies = Vec::with_capacity(NUM_QUERIES);
    for i in 0..NUM_QUERIES {
        let query = queries[i % queries.len()];
        let start = Instant::now();
        let results = mem.recall(query, 10).await.unwrap();
        latencies.push(start.elapsed().as_millis() as f64);
        assert!(!results.is_empty(), "recall should return results");
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];

    eprintln!("Recall ({NUM_QUERIES} queries): p50={p50:.0}ms, p95={p95:.0}ms");

    assert!(
        p50 < 500.0,
        "Recall p50 regression: {p50:.0}ms exceeds 500ms threshold"
    );
    assert!(
        p95 < 2000.0,
        "Recall p95 regression: {p95:.0}ms exceeds 2000ms threshold"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "perf regression guard — run with --ignored"]
async fn perf_guard_think_latency() {
    let (mem, _dir) = open_memory().await;

    for text in bench_memories().iter().take(30) {
        mem.remember(text).await.unwrap();
    }

    let queries = [
        "What are the fundamental biological processes in cells?",
        "How does quantum mechanics explain particle behavior?",
        "What were the key historical events of the medieval period?",
    ];

    let mut latencies = Vec::with_capacity(queries.len());
    for query in &queries {
        let start = Instant::now();
        let ctx = mem.think(query, 2048).await.unwrap();
        latencies.push(start.elapsed().as_millis() as f64);
        assert!(!ctx.context.is_empty(), "think should produce context");
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    eprintln!("Think ({} queries): p50={p50:.0}ms", queries.len());

    assert!(
        p50 < 3000.0,
        "Think p50 regression: {p50:.0}ms exceeds 3000ms threshold"
    );
}
