//! Integration tests for HIRN-Bench cognitive memory suites.

use hirn_bench::cognitive::{
    Benchmark, BenchmarkExecutionSurface, BenchmarkRetrievalProfile, CognitiveConfig,
};

fn run_benchmark(benchmark: Benchmark) -> hirn_bench::cognitive::CognitiveResult {
    let ds = hirn_bench::cognitive::synthetic::generate(benchmark);
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("test");
    // Lightweight config — sufficient for floor/regression checks.
    // Full-scale benchmarks use larger k and budget via CLI.
    let config = CognitiveConfig {
        embedding_dims: 64,
        token_budget: 2048,
        k: 10,
        retrieval_profile: BenchmarkRetrievalProfile::Minimal,
        execution_surface: BenchmarkExecutionSurface::DirectBuilders,
        query_text_hybrid: false,
        embedder_policy: Default::default(),
    };
    hirn_bench::cognitive::runner::run(&ds, &config, &db_path, "integration-test")
}

// ─── Each suite dataset loadable and parseable ───────────────

#[test]
fn all_synthetic_datasets_are_valid() {
    for &bench in Benchmark::all() {
        let ds = hirn_bench::cognitive::synthetic::generate(bench);
        assert!(!ds.sessions.is_empty(), "{bench}: empty sessions");
        assert!(!ds.queries.is_empty(), "{bench}: empty queries");

        for q in &ds.queries {
            assert!(
                !q.category.is_empty(),
                "{bench}: query {} has empty category",
                q.id
            );
            assert!(
                !q.expected_answers.is_empty(),
                "{bench}: query {} has no expected answers",
                q.id
            );
        }
    }
}

#[test]
fn dataset_loader_roundtrip() {
    for &bench in Benchmark::all() {
        let ds = hirn_bench::cognitive::synthetic::generate(bench);
        let dir = tempfile::TempDir::new().unwrap();

        let sessions: String = ds
            .sessions
            .iter()
            .map(|s| serde_json::to_string(s).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let queries: String = ds
            .queries
            .iter()
            .map(|q| serde_json::to_string(q).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("sessions.jsonl"), &sessions).unwrap();
        std::fs::write(dir.path().join("queries.jsonl"), &queries).unwrap();

        let loaded = hirn_bench::cognitive::loader::load(bench, dir.path()).unwrap();
        assert_eq!(
            loaded.sessions.len(),
            ds.sessions.len(),
            "{bench}: session count mismatch"
        );
        assert_eq!(
            loaded.queries.len(),
            ds.queries.len(),
            "{bench}: query count mismatch"
        );
    }
}

// NOTE: per-suite numeric score tests (h1–h6_produces_numeric_scores) live in
// the unit test module `cognitive::runner::tests` where they run with a lighter
// config. Integration tests here focus on floor validation and cross-suite checks.

// ── Shared assertion helpers ────────────────────────────────

fn assert_result_sane(result: &hirn_bench::cognitive::CognitiveResult) {
    assert!(result.total_queries > 0, "no queries");
    assert_eq!(result.strategy, "hirn");
    assert!(
        result.overall_containment.is_finite() && result.overall_containment >= 0.0,
        "containment invalid: {}",
        result.overall_containment,
    );
    assert!(
        result.overall_token_f1.is_finite() && result.overall_token_f1 >= 0.0,
        "token_f1 invalid: {}",
        result.overall_token_f1,
    );
    assert!(
        result.overall_recall_accuracy.is_finite() && result.overall_recall_accuracy >= 0.0,
        "recall_accuracy invalid: {}",
        result.overall_recall_accuracy,
    );
    assert!(
        result.overall_mrr.is_finite() && result.overall_mrr >= 0.0,
        "mrr invalid: {}",
        result.overall_mrr,
    );
    assert!(
        result.overall_ndcg.is_finite() && result.overall_ndcg >= 0.0,
        "ndcg invalid: {}",
        result.overall_ndcg,
    );
    assert!(
        result.false_positive_rate.is_finite() && result.false_positive_rate >= 0.0,
        "fpr invalid: {}",
        result.false_positive_rate,
    );
    assert!(result.total_time_secs > 0.0, "total_time should be > 0");
    assert!(result.execution_latency.p95 >= result.execution_latency.p50);
    assert!(result.end_to_end_latency.p95 >= result.end_to_end_latency.p50);
    assert!(result.token_cost.total_tokens > 0);
    assert!(!result.categories.is_empty(), "should have categories");
}

fn assert_meets_floor(benchmark: Benchmark, result: &hirn_bench::cognitive::CognitiveResult) {
    let v = hirn_bench::cognitive::baselines::validate(benchmark, result);
    assert!(
        v.meets_floor,
        "{}: floor failed \u{2014} containment={:.4} (min={:.2}), \
         recall={:.4} (min={:.2}), fpr={:.4} (max={:.2})",
        benchmark.name(),
        v.containment,
        v.floor.min_containment,
        v.recall_accuracy,
        v.floor.min_recall_accuracy,
        v.fpr,
        v.floor.max_fpr,
    );
    eprintln!(
        "{}: containment={:.4} recall={:.4} fpr={:.4} | floor=\u{2713} competitive={} target={}",
        benchmark.name(),
        v.containment,
        v.recall_accuracy,
        v.fpr,
        if v.meets_competitive {
            "\u{2713}"
        } else {
            "\u{2717}"
        },
        if v.meets_target {
            "\u{2713}"
        } else {
            "\u{2717}"
        },
    );
}

// ── Per-suite integration tests (one benchmark run each) ────
//
// Each test runs its benchmark exactly once and checks:
// - scores are finite and non-negative
// - pseudo-embedding floor thresholds are met
// - suite-specific invariants (negative queries, causal traps, etc.)

#[test]
fn h1_retrieval() {
    let result = run_benchmark(Benchmark::H1Retrieval);
    assert_result_sane(&result);
    assert_meets_floor(Benchmark::H1Retrieval, &result);

    // Negative-retrieval queries should not drag overall containment to 0.
    assert!(
        result.overall_containment > 0.0,
        "overall containment should not be dragged to 0 by negative queries"
    );

    // Tracker save/load roundtrip using this real result.
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("scores.json");
    hirn_bench::cognitive::tracker::save(&path, &result).unwrap();
    let history = hirn_bench::cognitive::tracker::load(&path).unwrap();
    assert_eq!(history.entries.len(), 1);
    assert_eq!(history.entries[0].benchmark, result.benchmark);
}

#[test]
fn h2_temporal() {
    let result = run_benchmark(Benchmark::H2Temporal);
    assert_result_sane(&result);
    assert_meets_floor(Benchmark::H2Temporal, &result);
}

#[test]
fn h3_graph() {
    let result = run_benchmark(Benchmark::H3Graph);
    assert_result_sane(&result);
    assert_meets_floor(Benchmark::H3Graph, &result);

    // H3 should be competitive with pseudo-embeddings.
    let v = hirn_bench::cognitive::baselines::validate(Benchmark::H3Graph, &result);
    assert!(
        v.meets_competitive,
        "H3 should be competitive: containment={:.4}",
        v.containment,
    );

    // Causal-trap (negative) queries should have 0 containment.
    if let Some(cat) = result.categories.iter().find(|c| c.name == "causal-trap") {
        assert_eq!(
            cat.containment, 0.0,
            "causal-trap should have 0 containment"
        );
    }
}

#[test]
fn h4_agent() {
    let result = run_benchmark(Benchmark::H4Agent);
    assert_result_sane(&result);
    assert_meets_floor(Benchmark::H4Agent, &result);
}

#[test]
fn h5_action() {
    let result = run_benchmark(Benchmark::H5Action);
    assert_result_sane(&result);
    assert_meets_floor(Benchmark::H5Action, &result);
}

#[test]
fn h6_safety() {
    let result = run_benchmark(Benchmark::H6Safety);
    assert_result_sane(&result);
    assert_meets_floor(Benchmark::H6Safety, &result);
}
