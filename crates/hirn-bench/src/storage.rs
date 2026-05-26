//! Storage-level benchmarks for hirn-storage capabilities.
//!
//! Benchmarks: hybrid search (RRF), batch BFS, resource persistence, lifecycle
//! compaction, and multivector search. Uses `MemoryStore` for fast,
//! deterministic runs suitable for CI.

use std::sync::Arc;
use std::time::Instant;

use hirn_core::{HydrationMode, MemoryId, ModalityProfile, ResourceLocation, ResourceObject};
use hirn_storage::resource_ops::{fetch_resource, persist_resource};
use hirn_storage::store::{
    HybridSearchOptions, MultivectorQuery, MultivectorSearchOptions, NormalizeMethod,
    VectorSearchOptions,
};
use hirn_storage::{HirnDb, PhysicalStore};

use crate::metrics::{LatencyStats, latency_percentiles};

/// Configuration for storage benchmarks.
#[derive(Debug, Clone)]
pub struct StorageBenchConfig {
    /// Number of records to insert.
    pub num_records: usize,
    /// Embedding dimensions.
    pub dims: usize,
    /// Warmup iterations.
    pub warmup: usize,
    /// Measured iterations.
    pub measured: usize,
    /// Limit (top-K) for searches.
    pub limit: usize,
    /// Number of graph edges to create.
    pub num_edges: usize,
    /// BFS max depth.
    pub bfs_depth: usize,
    /// BFS frontier size.
    pub bfs_frontier: usize,
}

impl Default for StorageBenchConfig {
    fn default() -> Self {
        Self {
            num_records: 1000,
            dims: 64,
            warmup: 1,
            measured: 5,
            limit: 10,
            num_edges: 10_000,
            bfs_depth: 2,
            bfs_frontier: 100,
        }
    }
}

/// Results from storage benchmarks.
#[derive(Debug, Clone)]
pub struct StorageBenchResult {
    pub vector_search: LatencyStats,
    pub hybrid_search: LatencyStats,
    pub multivector_search: LatencyStats,
    pub resource_persist: LatencyStats,
    pub resource_fetch: LatencyStats,
    pub batch_bfs: LatencyStats,
}

/// Run all storage benchmarks.
pub fn run(config: &StorageBenchConfig) -> StorageBenchResult {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");

    let store = HirnDb::open_memory();
    let phy = store.store_arc();

    // Pre-generate node MemoryIds for graph edges and BFS.
    let num_nodes = (config.num_edges as f64).sqrt().max(1.0) as usize;
    let node_ids: Vec<MemoryId> = (0..num_nodes).map(|_| MemoryId::new()).collect();

    // Ingest data.
    rt.block_on(ingest_episodic(&phy, config));
    rt.block_on(ingest_graph_edges(&phy, config, &node_ids));

    // Run individual benchmarks.
    let vector_search = rt.block_on(bench_vector_search(&phy, config));
    let hybrid_search = rt.block_on(bench_hybrid_search(&phy, config));
    let multivector_search = rt.block_on(bench_multivector_search(&phy, config));
    let (resource_persist, resource_fetch) = rt.block_on(bench_resource_ops(&phy, config));

    // Batch BFS uses PersistentGraph from hirn-engine.
    let batch_bfs = rt.block_on(bench_batch_bfs(&phy, config, &node_ids));

    StorageBenchResult {
        vector_search,
        hybrid_search,
        multivector_search,
        resource_persist,
        resource_fetch,
        batch_bfs,
    }
}

fn benchmark_resource(index: usize, size_bytes: usize) -> ResourceObject {
    ResourceObject::builder()
        .modality(ModalityProfile::Document)
        .mime_type("application/octet-stream")
        .display_name(format!("bench-resource-{index}.bin"))
        .checksum(format!("checksum:bench-resource-{index}"))
        .size_bytes(size_bytes as u64)
        .location(ResourceLocation::Blob { blob_index: 0 })
        .build()
        .expect("benchmark resource should be valid")
}

fn benchmark_payload(index: usize) -> Vec<u8> {
    (0..1024)
        .map(|offset| ((offset + index) % 256) as u8)
        .collect()
}

/// Generate a pseudo-random embedding vector.
fn pseudo_embedding(seed: usize, dims: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dims);
    let mut x = seed as f64 * 0.618_033_988_749_895;
    for _ in 0..dims {
        x = (x * 6.283_185_307).sin() * 10_000.0;
        v.push((x - x.floor()) as f32);
    }
    // Normalize.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

async fn ingest_episodic(store: &Arc<dyn PhysicalStore>, config: &StorageBenchConfig) {
    use arrow_array::builder::{Float32Builder, Int64Builder, StringBuilder};
    use arrow_array::{FixedSizeListArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    let dims = config.dims as i32;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dims),
            false,
        ),
        Field::new("created_at_ms", DataType::Int64, false),
    ]));

    let batch_size = 500;
    for chunk_start in (0..config.num_records).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(config.num_records);
        let n = chunk_end - chunk_start;

        let mut ids = StringBuilder::new();
        let mut content = StringBuilder::new();
        let mut ns = StringBuilder::new();
        let mut emb_values = Float32Builder::new();
        let mut ts = Int64Builder::new();

        for i in chunk_start..chunk_end {
            ids.append_value(format!("mem_{i}"));
            content.append_value(format!(
                "Memory content number {i} with some text for search"
            ));
            ns.append_value(format!("ns_{}", i % 5));
            let vec = pseudo_embedding(i, config.dims);
            for v in &vec {
                emb_values.append_value(*v);
            }
            ts.append_value(1_700_000_000_000 + i as i64 * 1000);
        }

        let embedding_array = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dims,
            Arc::new(emb_values.finish()),
            None,
        )
        .expect("valid fsl");

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids.finish()),
                Arc::new(content.finish()),
                Arc::new(ns.finish()),
                Arc::new(embedding_array),
                Arc::new(ts.finish()),
            ],
        )
        .expect("valid batch");

        store
            .append("episodic", batch)
            .await
            .unwrap_or_else(|e| panic!("append episodic chunk [{chunk_start}..{chunk_end}]: {e}"));
        eprintln!("    Ingested episodic [{chunk_start}..{chunk_end}) ({n} rows)");
    }
}

async fn ingest_graph_edges(
    store: &Arc<dyn PhysicalStore>,
    config: &StorageBenchConfig,
    node_ids: &[MemoryId],
) {
    use arrow_array::RecordBatch;
    use arrow_array::builder::{Float32Builder, Int32Builder, StringBuilder, UInt64Builder};
    use arrow_schema::{DataType, Field, Schema};

    // Use the graph_edges schema matching the current hirn-storage schema.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new("source", DataType::Utf8, false),
        Field::new("target", DataType::Utf8, false),
        Field::new("relation", DataType::Utf8, false),
        Field::new("weight", DataType::Float32, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("metadata", DataType::Utf8, true),
        Field::new("label", DataType::Utf8, true),
        Field::new("strength", DataType::Float32, true),
        Field::new("confidence", DataType::Float32, true),
        Field::new("evidence_count", DataType::Int32, true),
        Field::new(
            "confounders",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
            true,
        ),
        Field::new("provenance", DataType::Utf8, true),
        Field::new("mechanism", DataType::Utf8, true),
        Field::new("direction", DataType::Utf8, true),
    ]));

    let batch_size = 2000;
    for chunk_start in (0..config.num_edges).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(config.num_edges);

        let mut ids = UInt64Builder::new();
        let mut sources = StringBuilder::new();
        let mut targets = StringBuilder::new();
        let mut rels = StringBuilder::new();
        let mut weights = Float32Builder::new();
        let mut nss = StringBuilder::new();
        let mut ts = arrow_array::builder::Int64Builder::new();
        let mut metadata = StringBuilder::new();
        let mut labels = StringBuilder::new();
        let mut strengths = Float32Builder::new();
        let mut confidences = Float32Builder::new();
        let mut evidence_counts = Int32Builder::new();
        // For confounders (List<Utf8>), use ListBuilder.
        let mut confounders_builder = arrow_array::builder::ListBuilder::new(StringBuilder::new());
        let mut provenances = StringBuilder::new();
        let mut mechanisms = StringBuilder::new();
        let mut directions = StringBuilder::new();

        for i in chunk_start..chunk_end {
            ids.append_value(i as u64);
            let src = i % node_ids.len();
            let tgt = (i + 1) % node_ids.len();
            sources.append_value(node_ids[src].to_string());
            targets.append_value(node_ids[tgt].to_string());
            rels.append_value(if i % 3 == 0 {
                "Causes"
            } else if i % 3 == 1 {
                "RelatedTo"
            } else {
                "SimilarTo"
            });
            weights.append_value(0.5 + (i % 10) as f32 * 0.05);
            nss.append_value(format!("ns_{}", i % 5));
            ts.append_value(1_700_000_000_000 + i as i64 * 100);
            metadata.append_null();
            labels.append_null();
            strengths.append_null();
            confidences.append_null();
            evidence_counts.append_null();
            confounders_builder.append_null();
            provenances.append_null();
            mechanisms.append_null();
            directions.append_null();
        }

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids.finish()),
                Arc::new(sources.finish()),
                Arc::new(targets.finish()),
                Arc::new(rels.finish()),
                Arc::new(weights.finish()),
                Arc::new(nss.finish()),
                Arc::new(ts.finish()),
                Arc::new(metadata.finish()),
                Arc::new(labels.finish()),
                Arc::new(strengths.finish()),
                Arc::new(confidences.finish()),
                Arc::new(evidence_counts.finish()),
                Arc::new(confounders_builder.finish()),
                Arc::new(provenances.finish()),
                Arc::new(mechanisms.finish()),
                Arc::new(directions.finish()),
            ],
        )
        .expect("valid batch");

        store
            .append("graph_edges", batch)
            .await
            .unwrap_or_else(|e| panic!("append edges chunk [{chunk_start}..{chunk_end}]: {e}"));
    }
    eprintln!("    Ingested {} graph edges", config.num_edges);
}

async fn bench_vector_search(
    store: &Arc<dyn PhysicalStore>,
    config: &StorageBenchConfig,
) -> LatencyStats {
    let query = pseudo_embedding(42, config.dims);
    let opts = VectorSearchOptions {
        column: "embedding".to_string(),
        query: query.clone(),
        limit: config.limit,
        ..Default::default()
    };

    // Warmup.
    for _ in 0..config.warmup {
        let _ = store.vector_search("episodic", opts.clone()).await;
    }

    let mut latencies = Vec::with_capacity(config.measured);
    for _ in 0..config.measured {
        let start = Instant::now();
        let _ = store.vector_search("episodic", opts.clone()).await;
        latencies.push(start.elapsed());
    }
    latencies.sort();
    latency_percentiles(&latencies)
}

async fn bench_hybrid_search(
    store: &Arc<dyn PhysicalStore>,
    config: &StorageBenchConfig,
) -> LatencyStats {
    let query_vec = pseudo_embedding(42, config.dims);
    let opts = HybridSearchOptions {
        vector_column: "embedding".to_string(),
        query_vector: query_vec,
        fts_columns: vec!["content".to_string()],
        fts_query: "memory content search".to_string(),
        normalize: NormalizeMethod::Score,
        metric: hirn_storage::store::DistanceMetric::default(),
        limit: config.limit,
        filter: None,
        reranker: None, // Uses default RRF reranker.
    };

    for _ in 0..config.warmup {
        let _ = store.hybrid_search("episodic", opts.clone()).await;
    }

    let mut latencies = Vec::with_capacity(config.measured);
    for _ in 0..config.measured {
        let start = Instant::now();
        let _ = store.hybrid_search("episodic", opts.clone()).await;
        latencies.push(start.elapsed());
    }
    latencies.sort();
    latency_percentiles(&latencies)
}

async fn bench_multivector_search(
    store: &Arc<dyn PhysicalStore>,
    config: &StorageBenchConfig,
) -> LatencyStats {
    // MultivectorSearch uses a single query vector against the embedding column.
    let query = pseudo_embedding(42, config.dims);
    let opts = MultivectorSearchOptions {
        column: "embedding".to_string(),
        query: MultivectorQuery::Single(query),
        metric: hirn_storage::store::DistanceMetric::default(),
        limit: config.limit,
        filter: None,
        dense_column: Some("embedding".to_string()),
        first_stage_limit: None,
    };

    for _ in 0..config.warmup {
        let _ = store.multivector_search("episodic", opts.clone()).await;
    }

    let mut latencies = Vec::with_capacity(config.measured);
    for _ in 0..config.measured {
        let start = Instant::now();
        let _ = store.multivector_search("episodic", opts.clone()).await;
        latencies.push(start.elapsed());
    }
    latencies.sort();
    latency_percentiles(&latencies)
}

async fn bench_resource_ops(
    store: &Arc<dyn PhysicalStore>,
    config: &StorageBenchConfig,
) -> (LatencyStats, LatencyStats) {
    let warmup_base = 10_000;
    let measured_base = 20_000;

    for index in 0..config.warmup {
        let payload = benchmark_payload(warmup_base + index);
        let resource = benchmark_resource(warmup_base + index, payload.len());
        let _ = persist_resource(store.as_ref(), resource, Some(payload)).await;
    }

    let mut persist_latencies = Vec::with_capacity(config.measured);
    let mut resource_ids = Vec::with_capacity(config.measured);
    for i in 0..config.measured {
        let index = measured_base + i;
        let payload = benchmark_payload(index);
        let resource = benchmark_resource(index, payload.len());
        let start = Instant::now();
        let persisted = persist_resource(store.as_ref(), resource, Some(payload)).await;
        persist_latencies.push(start.elapsed());
        if let Ok(resource) = persisted {
            resource_ids.push(resource.id);
        }
    }
    persist_latencies.sort();

    let mut fetch_latencies = Vec::with_capacity(config.measured);
    if !resource_ids.is_empty() {
        for _ in 0..config.warmup {
            let _ = fetch_resource(store.as_ref(), resource_ids[0], HydrationMode::Full).await;
        }
        for i in 0..config.measured {
            let start = Instant::now();
            let _ = fetch_resource(
                store.as_ref(),
                resource_ids[i % resource_ids.len()],
                HydrationMode::Full,
            )
            .await;
            fetch_latencies.push(start.elapsed());
        }
        fetch_latencies.sort();
    }

    (
        latency_percentiles(&persist_latencies),
        latency_percentiles(&fetch_latencies),
    )
}

async fn bench_batch_bfs(
    store: &Arc<dyn PhysicalStore>,
    config: &StorageBenchConfig,
    node_ids: &[MemoryId],
) -> LatencyStats {
    use hirn_engine::PersistentGraph;

    let graph = PersistentGraph::new(store.clone());

    // Build frontier from the first N node IDs.
    let frontier_size = config.bfs_frontier.min(node_ids.len());
    let frontier: Vec<MemoryId> = node_ids[..frontier_size].to_vec();

    for _ in 0..config.warmup {
        let _ = graph.batch_bfs(&frontier, config.bfs_depth).await;
    }

    let mut latencies = Vec::with_capacity(config.measured);
    for _ in 0..config.measured {
        let start = Instant::now();
        let _ = graph.batch_bfs(&frontier, config.bfs_depth).await;
        latencies.push(start.elapsed());
    }
    latencies.sort();
    latency_percentiles(&latencies)
}

/// Format storage bench results as a markdown table.
pub fn format_markdown(result: &StorageBenchResult) -> String {
    let mut out = String::new();
    out.push_str("| Operation | P50 (µs) | P95 (µs) | P99 (µs) | Mean (µs) |\n");
    out.push_str("|-----------|----------|----------|----------|----------|\n");
    for (name, stats) in [
        ("Vector Search", &result.vector_search),
        ("Hybrid Search (RRF)", &result.hybrid_search),
        ("Multivector Search", &result.multivector_search),
        ("Resource Persist", &result.resource_persist),
        ("Resource Fetch", &result.resource_fetch),
        ("Batch BFS (depth-2)", &result.batch_bfs),
    ] {
        out.push_str(&format!(
            "| {} | {:.0} | {:.0} | {:.0} | {:.0} |\n",
            name,
            stats.p50.as_secs_f64() * 1_000_000.0,
            stats.p95.as_secs_f64() * 1_000_000.0,
            stats.p99.as_secs_f64() * 1_000_000.0,
            stats.mean.as_secs_f64() * 1_000_000.0,
        ));
    }
    out
}
