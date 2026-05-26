//! Integration tests for the DMR (Deep Memory Retrieval) benchmark.
//!
//! Tests use a synthetic DMR-format dataset to avoid network dependencies.
//! The tests verify: conversion, execution, scoring, reproducibility,
//! and caching logic.

use std::path::Path;

use hirn_bench::cognitive::{
    BenchmarkRetrievalProfile, CognitiveConfig, CognitiveResult,
    external::{DmrDialog, DmrQuery, DmrTurn, load_dmr},
};

// ─── Helpers ────────────────────────────────────────────────

/// Create a synthetic DMR-format dataset in a temp directory.
///
/// Contains 3 dialogs with multi-turn conversations and retrieval queries.
/// Designed to exercise the DMR evaluation harness with realistic fact-retrieval
/// patterns matching Zep's DMR benchmark protocol.
fn create_synthetic_dmr(dir: &Path) {
    let dialogs = vec![
        DmrDialog {
            dialog_id: "dialog-1".to_string(),
            turns: vec![
                DmrTurn {
                    speaker: "User".to_string(),
                    utterance: "I just adopted a golden retriever puppy named Max. He's 3 months old.".to_string(),
                    turn_id: Some(0),
                },
                DmrTurn {
                    speaker: "Assistant".to_string(),
                    utterance: "Congratulations on adopting Max! Golden retrievers are wonderful dogs. At 3 months, he'll need puppy vaccinations soon.".to_string(),
                    turn_id: Some(1),
                },
                DmrTurn {
                    speaker: "User".to_string(),
                    utterance: "Yes, our vet Dr. Sarah Chen at PetCare Clinic scheduled his shots for next Tuesday.".to_string(),
                    turn_id: Some(2),
                },
                DmrTurn {
                    speaker: "Assistant".to_string(),
                    utterance: "Dr. Chen is great! Make sure Max doesn't eat for 2 hours before the appointment.".to_string(),
                    turn_id: Some(3),
                },
                DmrTurn {
                    speaker: "User".to_string(),
                    utterance: "I also signed up for puppy training classes at Bark Academy starting next month.".to_string(),
                    turn_id: Some(4),
                },
            ],
            queries: vec![
                DmrQuery {
                    query: "What is the name of the user's dog?".to_string(),
                    answer: "Max".to_string(),
                    relevant_turn_ids: vec![0],
                },
                DmrQuery {
                    query: "What breed is the puppy?".to_string(),
                    answer: "golden retriever".to_string(),
                    relevant_turn_ids: vec![0],
                },
                DmrQuery {
                    query: "Who is the user's veterinarian?".to_string(),
                    answer: "Dr. Sarah Chen".to_string(),
                    relevant_turn_ids: vec![2],
                },
                DmrQuery {
                    query: "Where are the puppy training classes?".to_string(),
                    answer: "Bark Academy".to_string(),
                    relevant_turn_ids: vec![4],
                },
            ],
        },
        DmrDialog {
            dialog_id: "dialog-2".to_string(),
            turns: vec![
                DmrTurn {
                    speaker: "User".to_string(),
                    utterance: "I'm planning to renovate my kitchen. The budget is $25,000.".to_string(),
                    turn_id: Some(0),
                },
                DmrTurn {
                    speaker: "Assistant".to_string(),
                    utterance: "That's a reasonable budget for a kitchen renovation. What changes are you considering?".to_string(),
                    turn_id: Some(1),
                },
                DmrTurn {
                    speaker: "User".to_string(),
                    utterance: "I want to replace the countertops with quartz, install new cabinets from IKEA, and get a SubZero refrigerator.".to_string(),
                    turn_id: Some(2),
                },
                DmrTurn {
                    speaker: "Assistant".to_string(),
                    utterance: "A SubZero refrigerator alone can be $8,000-12,000. You may need to adjust other items to stay in budget.".to_string(),
                    turn_id: Some(3),
                },
                DmrTurn {
                    speaker: "User".to_string(),
                    utterance: "Good point. I'll go with a Samsung refrigerator instead and use the savings for a better backsplash.".to_string(),
                    turn_id: Some(4),
                },
                DmrTurn {
                    speaker: "User".to_string(),
                    utterance: "My contractor Mike Johnson said the work will take about 6 weeks starting in March.".to_string(),
                    turn_id: Some(5),
                },
            ],
            queries: vec![
                DmrQuery {
                    query: "What is the budget for the kitchen renovation?".to_string(),
                    answer: "$25,000".to_string(),
                    relevant_turn_ids: vec![0],
                },
                DmrQuery {
                    query: "What countertop material did the user choose?".to_string(),
                    answer: "quartz".to_string(),
                    relevant_turn_ids: vec![2],
                },
                DmrQuery {
                    query: "What refrigerator brand did the user finally decide on?".to_string(),
                    answer: "Samsung".to_string(),
                    relevant_turn_ids: vec![4],
                },
                DmrQuery {
                    query: "Who is the user's contractor?".to_string(),
                    answer: "Mike Johnson".to_string(),
                    relevant_turn_ids: vec![5],
                },
                DmrQuery {
                    query: "How long will the renovation take?".to_string(),
                    answer: "6 weeks".to_string(),
                    relevant_turn_ids: vec![5],
                },
            ],
        },
        DmrDialog {
            dialog_id: "dialog-3".to_string(),
            turns: vec![
                DmrTurn {
                    speaker: "User".to_string(),
                    utterance: "I started learning Python programming last week. I'm using the book 'Automate the Boring Stuff'.".to_string(),
                    turn_id: Some(0),
                },
                DmrTurn {
                    speaker: "Assistant".to_string(),
                    utterance: "That's an excellent beginner book by Al Sweigart. Have you set up your development environment?".to_string(),
                    turn_id: Some(1),
                },
                DmrTurn {
                    speaker: "User".to_string(),
                    utterance: "Yes, I installed VS Code and Python 3.12. I also joined the r/learnpython subreddit.".to_string(),
                    turn_id: Some(2),
                },
                DmrTurn {
                    speaker: "User".to_string(),
                    utterance: "My goal is to build a web scraper for my work at DataFlow Analytics to automate report generation.".to_string(),
                    turn_id: Some(3),
                },
            ],
            queries: vec![
                DmrQuery {
                    query: "What programming language is the user learning?".to_string(),
                    answer: "Python".to_string(),
                    relevant_turn_ids: vec![0],
                },
                DmrQuery {
                    query: "What book is the user studying from?".to_string(),
                    answer: "Automate the Boring Stuff".to_string(),
                    relevant_turn_ids: vec![0],
                },
                DmrQuery {
                    query: "What code editor does the user use?".to_string(),
                    answer: "VS Code".to_string(),
                    relevant_turn_ids: vec![2],
                },
                DmrQuery {
                    query: "Where does the user work?".to_string(),
                    answer: "DataFlow Analytics".to_string(),
                    relevant_turn_ids: vec![3],
                },
            ],
        },
    ];

    let json = serde_json::to_string_pretty(&dialogs).unwrap();
    std::fs::write(dir.join("dialogs.json"), json).unwrap();
}

fn run_dmr_benchmark(data_dir: &Path) -> CognitiveResult {
    let dataset = load_dmr(data_dir).unwrap();
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("dmr");
    let config = CognitiveConfig {
        embedding_dims: 64,
        token_budget: 2048,
        k: 5,
        retrieval_profile: BenchmarkRetrievalProfile::Minimal,
        execution_surface: hirn_bench::cognitive::BenchmarkExecutionSurface::DirectBuilders,
        query_text_hybrid: false,
        embedder_policy: Default::default(),
    };
    hirn_bench::cognitive::runner::run(&dataset, &config, &db_path, "dmr-test")
}

// ── Adapter (fast, no benchmark run) ─────────────────────────

#[test]
fn dmr_adapter_converts_correctly() {
    let dir = tempfile::TempDir::new().unwrap();
    create_synthetic_dmr(dir.path());

    let dataset = load_dmr(dir.path()).unwrap();

    // Should have 3 sessions (one per dialog).
    assert_eq!(dataset.sessions.len(), 3, "should have 3 sessions");

    // Dialog 1 has 5 turns, Dialog 2 has 6 turns, Dialog 3 has 4 turns.
    assert_eq!(dataset.sessions[0].turns.len(), 5);
    assert_eq!(dataset.sessions[1].turns.len(), 6);
    assert_eq!(dataset.sessions[2].turns.len(), 4);

    // Verify turn content.
    assert_eq!(dataset.sessions[0].turns[0].speaker, "User");
    assert!(dataset.sessions[0].turns[0].content.contains("Max"));

    // Should have 13 queries total (4 + 5 + 4).
    assert_eq!(dataset.queries.len(), 13, "should have 13 queries");

    // All DMR queries are in the "retrieval" category.
    assert!(
        dataset.queries.iter().all(|q| q.category == "retrieval"),
        "all DMR queries should be 'retrieval' category"
    );

    // Verify query IDs follow pattern.
    assert!(dataset.queries[0].id.starts_with("dmr-dialog-1-"));
    assert!(dataset.queries[4].id.starts_with("dmr-dialog-2-"));

    // Verify expected answers.
    assert_eq!(dataset.queries[0].expected_answers, vec!["Max"]);
    assert_eq!(dataset.queries[6].expected_answers, vec!["Samsung"]);
}

#[test]
fn dmr_cache_marker_prevents_redownload() {
    let cache_dir = tempfile::TempDir::new().unwrap();
    let cache_path = cache_dir.path();

    // Simulate a cached dataset by writing the marker and data file.
    create_synthetic_dmr(cache_path);
    std::fs::write(cache_path.join(".dmr_downloaded"), "3 dialogs").unwrap();

    // download_dmr should detect the cache marker and return immediately.
    let result = hirn_bench::cognitive::external::download_dmr(cache_path);
    assert!(
        result.is_ok(),
        "cached download should succeed: {:?}",
        result
    );
    assert_eq!(result.unwrap(), cache_path.to_path_buf());

    // Verify it loaded the cached data, not re-downloaded.
    let dataset = load_dmr(cache_path).unwrap();
    assert_eq!(dataset.sessions.len(), 3);
}

// ── Benchmark execution (single run) ────────────────────────

#[test]
fn dmr_benchmark_produces_valid_scores() {
    let dir = tempfile::TempDir::new().unwrap();
    create_synthetic_dmr(dir.path());

    let result = run_dmr_benchmark(dir.path());

    // 3 dialogs: 4 + 5 + 4 = 13 queries.
    assert_eq!(result.total_queries, 13);
    assert!(result.total_time_secs > 0.0);
    assert!(result.ingest_time_secs > 0.0);
    assert!(result.query_time_secs > 0.0);

    // DMR has a single "retrieval" category.
    assert_eq!(result.categories.len(), 1);
    assert_eq!(result.categories[0].name, "retrieval");
    assert_eq!(result.categories[0].total, 13);

    // All metrics finite and non-negative.
    assert!(result.overall_containment >= 0.0 && result.overall_containment.is_finite());
    assert!(result.overall_token_f1 >= 0.0 && result.overall_token_f1.is_finite());
    assert!(result.overall_recall_accuracy >= 0.0 && result.overall_recall_accuracy.is_finite());
    assert!(result.overall_mrr >= 0.0 && result.overall_mrr.is_finite());
    assert!(result.overall_ndcg >= 0.0 && result.overall_ndcg.is_finite());

    // No negative queries in DMR -> FPR should be 0.
    assert_eq!(result.false_positive_rate, 0.0);
}
