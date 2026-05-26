use std::path::PathBuf;
use std::time::Duration;

use hirn_bench::advanced::{self, AdvancedBenchmark, AdvancedConfig};

#[test]
fn advanced_suite_smoke_executes_story32_surfaces() {
    let config = AdvancedConfig {
        runs: 1,
        offline_wait_ms: 5_000,
        repro_threshold: 0.15,
        environment_label: Some("test-runner".to_string()),
    };

    let suite = advanced::run_suite(AdvancedBenchmark::all(), &config, "advanced-smoke-run")
        .expect("advanced suite should run");

    assert_eq!(suite.results.len(), AdvancedBenchmark::all().len());
    assert!(suite.overall_primary_score >= 0.75);
    assert!(suite.total_time_secs > 0.0);

    for result in &suite.results {
        assert!(
            result.quality.primary_score >= 0.75,
            "{} should remain above smoke floor",
            result.benchmark
        );
        assert!(result.total_cases > 0);
        assert!(result.latency.p95 > Duration::ZERO);
    }
}

#[test]
fn documentation_smoke_covers_advanced_offline_workflow() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let readme =
        std::fs::read_to_string(crate_root.join("README.md")).expect("read hirn-bench README");
    let benchmarks_doc = std::fs::read_to_string(crate_root.join("../../docs/benchmarks.md"))
        .expect("read benchmarks doc");

    for document in [&readme, &benchmarks_doc] {
        assert!(document.contains("Advanced Offline Cognition Workflow"));
        assert!(document.contains("cargo run -p hirn-bench -- advanced --benchmark all"));
        assert!(document.contains("--tracker bench-results/advanced-history.json"));
        assert!(document.contains("cargo run -p hirn-bench -- bench-compare"));
    }
}
