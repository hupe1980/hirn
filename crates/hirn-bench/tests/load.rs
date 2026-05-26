use std::time::Duration;

use hirn_bench::load::{self, LoadConfig};

#[test]
fn concurrent_load_benchmark_produces_latency_envelope() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("load");
    let config = LoadConfig {
        writers: 2,
        readers: 3,
        writes_per_writer: 8,
        writer_batch_size: 4,
        max_auto_edges_per_record: 4,
        reads_per_reader: 12,
        preseed_records: 24,
        embedding_dims: 16,
        k: 5,
    };

    let result = load::run(&config, &db_path, "load-integration").unwrap();

    assert_eq!(result.writer_failures, 0);
    assert_eq!(result.reader_failures, 0);
    assert!(result.remember_latency.p50 > Duration::ZERO);
    assert!(result.remember_latency.p95 >= result.remember_latency.p50);
    assert!(result.recall_latency.p50 > Duration::ZERO);
    assert!(result.recall_latency.p95 >= result.recall_latency.p50);
    assert!(result.throughput.remember_ops_per_sec > 0.0);
    assert!(result.throughput.recall_ops_per_sec > 0.0);
    assert!(result.throughput.total_ops_per_sec > 0.0);
    assert!(!result.batch_remember_stage_latency.is_empty());
    assert!(result.load_time > Duration::ZERO);
    assert!(result.total_time >= result.load_time);
}
