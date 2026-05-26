//! Integration tests for the LongMemEval benchmark.
//!
//! Tests use a synthetic LongMemEval-format dataset to avoid network dependencies.
//! The tests verify: conversion, execution, scoring with multiple task types,
//! abstention (false-positive detection), reproducibility, and caching logic.

use std::path::Path;

use hirn_bench::cognitive::{
    BenchmarkRetrievalProfile, CognitiveConfig, CognitiveResult,
    external::{
        LongMemEvalCase, LongMemEvalQuestion, LongMemEvalSession, LongMemEvalTurn, load_longmemeval,
    },
};

// ─── Helpers ────────────────────────────────────────────────

/// Create a synthetic LongMemEval-format dataset in a temp directory.
///
/// Contains 3 cases exercising:
///   - information extraction
///   - temporal reasoning
///   - knowledge update
///   - abstention (should_abstain = true)
fn create_synthetic_longmemeval(dir: &Path) {
    let cases = vec![
        // Case 1: Travel planning – info extraction + temporal reasoning
        LongMemEvalCase {
            id: "case-1".to_string(),
            sessions: vec![
                LongMemEvalSession {
                    session_id: Some("case-1-s0".to_string()),
                    turns: vec![
                        LongMemEvalTurn {
                            role: "user".to_string(),
                            content: "I'm planning a trip to Tokyo next April. I want to visit during cherry blossom season.".to_string(),
                        },
                        LongMemEvalTurn {
                            role: "assistant".to_string(),
                            content: "Cherry blossom season in Tokyo usually peaks late March to mid April. Great timing!".to_string(),
                        },
                        LongMemEvalTurn {
                            role: "user".to_string(),
                            content: "I booked a flight with ANA departing March 28 and returning April 10. The hotel is Park Hyatt in Shinjuku.".to_string(),
                        },
                    ],
                },
                LongMemEvalSession {
                    session_id: Some("case-1-s1".to_string()),
                    turns: vec![
                        LongMemEvalTurn {
                            role: "user".to_string(),
                            content: "I also want to take the bullet train to Kyoto on April 3. My friend Yuki recommended visiting Fushimi Inari shrine.".to_string(),
                        },
                        LongMemEvalTurn {
                            role: "assistant".to_string(),
                            content: "The Shinkansen from Tokyo to Kyoto takes about 2 hours 15 minutes. Fushimi Inari is beautiful.".to_string(),
                        },
                    ],
                },
            ],
            questions: vec![
                LongMemEvalQuestion {
                    question: "What airline did the user book for the Tokyo trip?".to_string(),
                    answer: "ANA".to_string(),
                    task_type: Some("information_extraction".to_string()),
                    should_abstain: false,
                },
                LongMemEvalQuestion {
                    question: "When is the user departing for Tokyo?".to_string(),
                    answer: "March 28".to_string(),
                    task_type: Some("temporal_reasoning".to_string()),
                    should_abstain: false,
                },
                LongMemEvalQuestion {
                    question: "What hotel is the user staying at?".to_string(),
                    answer: "Park Hyatt".to_string(),
                    task_type: Some("information_extraction".to_string()),
                    should_abstain: false,
                },
                LongMemEvalQuestion {
                    question: "Who recommended visiting Fushimi Inari?".to_string(),
                    answer: "Yuki".to_string(),
                    task_type: Some("information_extraction".to_string()),
                    should_abstain: false,
                },
                LongMemEvalQuestion {
                    question: "When does the user plan to travel to Kyoto by bullet train?".to_string(),
                    answer: "April 3".to_string(),
                    task_type: Some("temporal_reasoning".to_string()),
                    should_abstain: false,
                },
            ],
        },
        // Case 2: Career changes – knowledge update + abstention
        LongMemEvalCase {
            id: "case-2".to_string(),
            sessions: vec![
                LongMemEvalSession {
                    session_id: Some("case-2-s0".to_string()),
                    turns: vec![
                        LongMemEvalTurn {
                            role: "user".to_string(),
                            content: "I just started a new job at Google as a senior software engineer. My manager is Lisa Park.".to_string(),
                        },
                        LongMemEvalTurn {
                            role: "assistant".to_string(),
                            content: "Congratulations on joining Google! That's exciting.".to_string(),
                        },
                    ],
                },
                LongMemEvalSession {
                    session_id: Some("case-2-s1".to_string()),
                    turns: vec![
                        LongMemEvalTurn {
                            role: "user".to_string(),
                            content: "Actually I got promoted to staff engineer last week. My new manager is David Kim.".to_string(),
                        },
                        LongMemEvalTurn {
                            role: "assistant".to_string(),
                            content: "Amazing, congrats on the promotion to staff engineer!".to_string(),
                        },
                    ],
                },
            ],
            questions: vec![
                LongMemEvalQuestion {
                    question: "Where does the user work?".to_string(),
                    answer: "Google".to_string(),
                    task_type: Some("information_extraction".to_string()),
                    should_abstain: false,
                },
                LongMemEvalQuestion {
                    question: "What is the user's current role?".to_string(),
                    answer: "staff engineer".to_string(),
                    task_type: Some("knowledge_update".to_string()),
                    should_abstain: false,
                },
                LongMemEvalQuestion {
                    question: "Who is the user's current manager?".to_string(),
                    answer: "David Kim".to_string(),
                    task_type: Some("knowledge_update".to_string()),
                    should_abstain: false,
                },
                // Abstention: question about something never discussed.
                LongMemEvalQuestion {
                    question: "What is the user's salary at Google?".to_string(),
                    answer: "".to_string(),
                    task_type: Some("abstention".to_string()),
                    should_abstain: true,
                },
            ],
        },
        // Case 3: Health & fitness – multi-session reasoning + abstention
        LongMemEvalCase {
            id: "case-3".to_string(),
            sessions: vec![
                LongMemEvalSession {
                    session_id: Some("case-3-s0".to_string()),
                    turns: vec![
                        LongMemEvalTurn {
                            role: "user".to_string(),
                            content: "I started a new workout plan. Running 5k three times a week, plus yoga on Saturdays at Zen Studio.".to_string(),
                        },
                        LongMemEvalTurn {
                            role: "assistant".to_string(),
                            content: "That sounds like a balanced routine! How long have you been running?".to_string(),
                        },
                        LongMemEvalTurn {
                            role: "user".to_string(),
                            content: "About 6 months. My personal best for 5k is 23 minutes. My trainer is Coach Martinez.".to_string(),
                        },
                    ],
                },
                LongMemEvalSession {
                    session_id: Some("case-3-s1".to_string()),
                    turns: vec![
                        LongMemEvalTurn {
                            role: "user".to_string(),
                            content: "I just ran a half marathon in 1 hour 52 minutes! Also switched to a new gym called FitCore.".to_string(),
                        },
                        LongMemEvalTurn {
                            role: "assistant".to_string(),
                            content: "1:52 for a half marathon is impressive! How's the new gym?".to_string(),
                        },
                    ],
                },
            ],
            questions: vec![
                LongMemEvalQuestion {
                    question: "What is the user's 5k personal best?".to_string(),
                    answer: "23 minutes".to_string(),
                    task_type: Some("information_extraction".to_string()),
                    should_abstain: false,
                },
                LongMemEvalQuestion {
                    question: "What was the user's half marathon time?".to_string(),
                    answer: "1 hour 52 minutes".to_string(),
                    task_type: Some("information_extraction".to_string()),
                    should_abstain: false,
                },
                LongMemEvalQuestion {
                    question: "Where does the user do yoga?".to_string(),
                    answer: "Zen Studio".to_string(),
                    task_type: Some("information_extraction".to_string()),
                    should_abstain: false,
                },
                LongMemEvalQuestion {
                    question: "Who is the user's trainer?".to_string(),
                    answer: "Coach Martinez".to_string(),
                    task_type: Some("information_extraction".to_string()),
                    should_abstain: false,
                },
                // Abstention: never discussed diet.
                LongMemEvalQuestion {
                    question: "What diet plan is the user following?".to_string(),
                    answer: "".to_string(),
                    task_type: Some("abstention".to_string()),
                    should_abstain: true,
                },
            ],
        },
    ];

    let json = serde_json::to_string_pretty(&cases).unwrap();
    std::fs::write(dir.join("cases.json"), json).unwrap();
}

fn run_lme_benchmark(data_dir: &Path) -> CognitiveResult {
    let dataset = load_longmemeval(data_dir).unwrap();
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("lme");
    let config = CognitiveConfig {
        embedding_dims: 64,
        token_budget: 2048,
        k: 5,
        retrieval_profile: BenchmarkRetrievalProfile::Minimal,
        execution_surface: hirn_bench::cognitive::BenchmarkExecutionSurface::DirectBuilders,
        query_text_hybrid: false,
        embedder_policy: Default::default(),
    };
    hirn_bench::cognitive::runner::run(&dataset, &config, &db_path, "lme-test")
}

// ── Adapter (fast, no benchmark run) ─────────────────────────

#[test]
fn longmemeval_adapter_converts_correctly() {
    let dir = tempfile::TempDir::new().unwrap();
    create_synthetic_longmemeval(dir.path());

    let dataset = load_longmemeval(dir.path()).unwrap();

    // 3 cases x 2 sessions each = 6 sessions.
    assert_eq!(dataset.sessions.len(), 6, "should have 6 sessions");

    // Sessions should have correct turn counts.
    assert_eq!(dataset.sessions[0].turns.len(), 3, "case-1-s0 has 3 turns");
    assert_eq!(dataset.sessions[1].turns.len(), 2, "case-1-s1 has 2 turns");
    assert_eq!(dataset.sessions[2].turns.len(), 2, "case-2-s0 has 2 turns");
    assert_eq!(dataset.sessions[3].turns.len(), 2, "case-2-s1 has 2 turns");
    assert_eq!(dataset.sessions[4].turns.len(), 3, "case-3-s0 has 3 turns");
    assert_eq!(dataset.sessions[5].turns.len(), 2, "case-3-s1 has 2 turns");

    // 14 queries total (5 + 4 + 5).
    assert_eq!(dataset.queries.len(), 14, "should have 14 queries");

    // 2 negative (abstention) queries.
    let neg_count = dataset.queries.iter().filter(|q| q.negative).count();
    assert_eq!(neg_count, 2, "should have 2 negative (abstention) queries");

    // Verify turn content.
    assert_eq!(dataset.sessions[0].turns[0].speaker, "user");
    assert!(dataset.sessions[0].turns[0].content.contains("Tokyo"));

    // Verify session IDs.
    assert_eq!(dataset.sessions[0].id, "case-1-s0");
    assert_eq!(dataset.sessions[3].id, "case-2-s1");
}

#[test]
fn longmemeval_cache_marker_prevents_redownload() {
    let cache_dir = tempfile::TempDir::new().unwrap();
    let cache_path = cache_dir.path();

    // Simulate a cached dataset by writing the marker and data file.
    create_synthetic_longmemeval(cache_path);
    std::fs::write(cache_path.join(".longmemeval_downloaded"), "3 cases").unwrap();

    // download_longmemeval should detect the cache marker and return immediately.
    let result = hirn_bench::cognitive::external::download_longmemeval(cache_path);
    assert!(
        result.is_ok(),
        "cached download should succeed: {:?}",
        result
    );
    assert_eq!(result.unwrap(), cache_path.to_path_buf());

    // Verify it loaded the cached data, not re-downloaded.
    let dataset = load_longmemeval(cache_path).unwrap();
    assert_eq!(dataset.sessions.len(), 6, "should have 6 sessions (2+2+2)");
}

// ── Benchmark execution (single run) ────────────────────────

#[test]
fn longmemeval_benchmark_produces_valid_scores() {
    let dir = tempfile::TempDir::new().unwrap();
    create_synthetic_longmemeval(dir.path());

    let result = run_lme_benchmark(dir.path());

    // 3 cases: 5 + 4 + 5 = 14 queries.
    assert_eq!(result.total_queries, 14);
    assert!(result.total_time_secs > 0.0);
    assert!(result.ingest_time_secs > 0.0);
    assert!(result.query_time_secs > 0.0);

    // Should have categories for each task type.
    let cat_names: Vec<&str> = result.categories.iter().map(|c| c.name.as_str()).collect();
    assert!(
        cat_names.contains(&"information_extraction"),
        "missing: {cat_names:?}"
    );
    assert!(
        cat_names.contains(&"temporal_reasoning"),
        "missing: {cat_names:?}"
    );
    assert!(
        cat_names.contains(&"knowledge_update"),
        "missing: {cat_names:?}"
    );
    assert!(cat_names.contains(&"abstention"), "missing: {cat_names:?}");

    // Info extraction: 8 queries.
    let ie = result
        .categories
        .iter()
        .find(|c| c.name == "information_extraction")
        .unwrap();
    assert_eq!(ie.total, 8);

    // 2 abstention (negative) queries -> FPR is a valid fraction.
    assert!(result.false_positive_rate >= 0.0 && result.false_positive_rate <= 1.0);

    // All metrics finite and non-negative.
    assert!(result.overall_containment >= 0.0 && result.overall_containment.is_finite());
    assert!(result.overall_token_f1 >= 0.0 && result.overall_token_f1.is_finite());
    assert!(result.overall_recall_accuracy >= 0.0 && result.overall_recall_accuracy.is_finite());
}
