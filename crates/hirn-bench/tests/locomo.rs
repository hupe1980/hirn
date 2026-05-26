//! Integration tests for the LoCoMo benchmark.
//!
//! Tests use a synthetic LoCoMo-format dataset to avoid network dependencies.
//! The tests verify: conversion, execution, per-category scoring, reproducibility,
//! and caching logic.

use std::collections::HashMap;
use std::path::Path;

use hirn_bench::cognitive::{
    BenchmarkRetrievalProfile, CognitiveConfig, CognitiveResult,
    external::{LoCoMoConversation, LoCoMoQuestion, LoCoMoTurn, load_locomo},
};

// ─── Helpers ────────────────────────────────────────────────

/// Create a minimal LoCoMo-format dataset in a temp directory.
///
/// Contains 2 conversations with 5 question categories matching the real
/// LoCoMo benchmark: single-hop, multi-hop, temporal, world-knowledge, adversarial.
fn create_synthetic_locomo(dir: &Path) {
    let conversations = vec![
        LoCoMoConversation {
            id: "conv-1".to_string(),
            conversation: vec![
                LoCoMoTurn {
                    speaker: "Alice".to_string(),
                    text: "I started a new job at Acme Corp last Monday as a software engineer.".to_string(),
                    timestamp: Some("2024-01-15 09:00:00".to_string()),
                    source_id: Some("conv-1:1".to_string()),
                    session_id: Some("session_1".to_string()),
                },
                LoCoMoTurn {
                    speaker: "Bob".to_string(),
                    text: "That's great! I heard Acme Corp is working on a quantum computing project.".to_string(),
                    timestamp: Some("2024-01-15 09:05:00".to_string()),
                    source_id: Some("conv-1:2".to_string()),
                    session_id: Some("session_1".to_string()),
                },
                LoCoMoTurn {
                    speaker: "Alice".to_string(),
                    text: "Yes, I'm joining the quantum team. We're using Qiskit for circuit simulation.".to_string(),
                    timestamp: Some("2024-01-15 09:10:00".to_string()),
                    source_id: Some("conv-1:3".to_string()),
                    session_id: Some("session_1".to_string()),
                },
                LoCoMoTurn {
                    speaker: "Bob".to_string(),
                    text: "My brother works at Google's quantum lab. They use Cirq instead of Qiskit.".to_string(),
                    timestamp: Some("2024-01-15 09:15:00".to_string()),
                    source_id: Some("conv-1:4".to_string()),
                    session_id: Some("session_1".to_string()),
                },
                LoCoMoTurn {
                    speaker: "Alice".to_string(),
                    text: "Actually I just got reassigned to the AI safety team instead of quantum. The project was cancelled.".to_string(),
                    timestamp: Some("2024-02-01 10:00:00".to_string()),
                    source_id: Some("conv-1:5".to_string()),
                    session_id: Some("session_2".to_string()),
                },
            ],
            questions: HashMap::from([
                ("single-hop".to_string(), vec![
                    LoCoMoQuestion {
                        question: "Where does Alice work?".to_string(),
                        answer: "Acme Corp".to_string(),
                        evidence: vec!["conv-1:1".to_string()],
                    },
                ]),
                ("multi-hop".to_string(), vec![
                    LoCoMoQuestion {
                        question: "What framework does the company where Bob's brother works use for quantum computing?".to_string(),
                        answer: "Cirq".to_string(),
                        evidence: vec![
                            "conv-1:4".to_string(),
                        ],
                    },
                ]),
                ("temporal".to_string(), vec![
                    LoCoMoQuestion {
                        question: "When did Alice start her job?".to_string(),
                        answer: "last Monday".to_string(),
                        evidence: vec!["conv-1:1".to_string()],
                    },
                ]),
                ("world-knowledge".to_string(), vec![
                    LoCoMoQuestion {
                        question: "What is Qiskit used for?".to_string(),
                        answer: "circuit simulation".to_string(),
                        evidence: vec!["conv-1:3".to_string()],
                    },
                ]),
                ("adversarial".to_string(), vec![
                    LoCoMoQuestion {
                        question: "What team is Alice currently on?".to_string(),
                        answer: "AI safety".to_string(),
                        evidence: vec!["conv-1:5".to_string()],
                    },
                ]),
            ]),
        },
        LoCoMoConversation {
            id: "conv-2".to_string(),
            conversation: vec![
                LoCoMoTurn {
                    speaker: "Charlie".to_string(),
                    text: "I'm planning a trip to Tokyo next month. I've always wanted to visit the Tsukiji fish market.".to_string(),
                    timestamp: Some("2024-03-01 14:00:00".to_string()),
                    source_id: Some("conv-2:1".to_string()),
                    session_id: Some("session_1".to_string()),
                },
                LoCoMoTurn {
                    speaker: "Diana".to_string(),
                    text: "Tsukiji's inner market actually moved to Toyosu in 2018. But the outer market is still there.".to_string(),
                    timestamp: Some("2024-03-01 14:05:00".to_string()),
                    source_id: Some("conv-2:2".to_string()),
                    session_id: Some("session_1".to_string()),
                },
                LoCoMoTurn {
                    speaker: "Charlie".to_string(),
                    text: "Oh I didn't know that. I'll visit Toyosu then. Diana, have you been to Japan?".to_string(),
                    timestamp: Some("2024-03-01 14:10:00".to_string()),
                    source_id: Some("conv-2:3".to_string()),
                    session_id: Some("session_1".to_string()),
                },
                LoCoMoTurn {
                    speaker: "Diana".to_string(),
                    text: "Yes, I lived in Osaka for two years. The street food in Dotonbori is amazing.".to_string(),
                    timestamp: Some("2024-03-01 14:15:00".to_string()),
                    source_id: Some("conv-2:4".to_string()),
                    session_id: Some("session_1".to_string()),
                },
            ],
            questions: HashMap::from([
                ("single-hop".to_string(), vec![
                    LoCoMoQuestion {
                        question: "Where is Charlie planning to travel?".to_string(),
                        answer: "Tokyo".to_string(),
                        evidence: vec!["conv-2:1".to_string()],
                    },
                ]),
                ("multi-hop".to_string(), vec![
                    LoCoMoQuestion {
                        question: "Where did the person who corrected Charlie about Tsukiji live in Japan?".to_string(),
                        answer: "Osaka".to_string(),
                        evidence: vec!["conv-2:2".to_string(), "conv-2:4".to_string()],
                    },
                ]),
                ("temporal".to_string(), vec![
                    LoCoMoQuestion {
                        question: "When did the Tsukiji inner market move?".to_string(),
                        answer: "2018".to_string(),
                        evidence: vec!["conv-2:2".to_string()],
                    },
                ]),
                ("adversarial".to_string(), vec![
                    LoCoMoQuestion {
                        question: "Where does Charlie plan to see the fish market after learning about the move?".to_string(),
                        answer: "Toyosu".to_string(),
                        evidence: vec!["conv-2:3".to_string()],
                    },
                ]),
            ]),
        },
    ];

    let json = serde_json::to_string_pretty(&conversations).unwrap();
    std::fs::write(dir.join("conversations.json"), json).unwrap();
}

fn run_locomo_benchmark(data_dir: &Path) -> CognitiveResult {
    let dataset = load_locomo(data_dir).unwrap();
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("locomo");
    let config = CognitiveConfig {
        embedding_dims: 64,
        token_budget: 2048,
        k: 5,
        retrieval_profile: BenchmarkRetrievalProfile::Minimal,
        execution_surface: hirn_bench::cognitive::BenchmarkExecutionSurface::DirectBuilders,
        query_text_hybrid: false,
        embedder_policy: Default::default(),
    };
    hirn_bench::cognitive::runner::run(&dataset, &config, &db_path, "locomo-test")
}

// ── Adapter (fast, no benchmark run) ─────────────────────────

#[test]
fn locomo_adapter_converts_correctly() {
    let dir = tempfile::TempDir::new().unwrap();
    create_synthetic_locomo(dir.path());

    let dataset = load_locomo(dir.path()).unwrap();

    // Session ids should split on the source session structure.
    assert_eq!(dataset.sessions.len(), 3, "should have 3 logical sessions");

    // First logical session should preserve turn ids.
    assert_eq!(dataset.sessions[0].turns.len(), 4);

    // Verify turn content includes speaker and text.
    assert_eq!(dataset.sessions[0].turns[0].speaker, "Alice");
    assert!(dataset.sessions[0].turns[0].content.contains("Acme Corp"));
    assert_eq!(
        dataset.sessions[0].turns[0].source_id.as_deref(),
        Some("conv-1:1")
    );

    // Should have queries across multiple categories.
    let total_queries = dataset.queries.len();
    assert!(
        total_queries >= 9,
        "should have at least 9 queries (5 from conv-1 + 4 from conv-2), got {total_queries}"
    );

    // Verify category diversity.
    let categories: std::collections::HashSet<&str> = dataset
        .queries
        .iter()
        .map(|q| q.category.as_str())
        .collect();
    assert!(categories.contains("single-hop"), "should have single-hop");
    assert!(categories.contains("multi-hop"), "should have multi-hop");
    assert!(categories.contains("temporal"), "should have temporal");
    assert!(
        categories.contains("world-knowledge"),
        "should have world-knowledge"
    );
    assert!(
        categories.contains("adversarial"),
        "should have adversarial"
    );

    let work_query = dataset
        .queries
        .iter()
        .find(|query| query.question == "Where does Alice work?")
        .unwrap();
    assert_eq!(work_query.evidence_ids, vec!["conv-1:1"]);
    assert_eq!(
        work_query.evidence_snippets,
        vec!["I started a new job at Acme Corp last Monday as a software engineer."]
    );
    assert_eq!(work_query.relevant_session_ids, vec!["conv-1::session_1"]);
}

#[test]
fn locomo_cache_marker_prevents_redownload() {
    let cache_dir = tempfile::TempDir::new().unwrap();
    let cache_path = cache_dir.path();

    // Simulate a cached dataset by writing the marker and data file.
    create_synthetic_locomo(cache_path);
    std::fs::write(cache_path.join(".locomo_downloaded"), "2 conversations").unwrap();

    // download_locomo should detect the cache marker and return immediately.
    let result = hirn_bench::cognitive::external::download_locomo(cache_path);
    assert!(
        result.is_ok(),
        "cached download should succeed: {:?}",
        result
    );
    assert_eq!(result.unwrap(), cache_path.to_path_buf());

    // Verify it loaded the cached data, not re-downloaded.
    let dataset = load_locomo(cache_path).unwrap();
    assert_eq!(dataset.sessions.len(), 3);
}

// ── Benchmark execution (single run) ────────────────────────

#[test]
fn locomo_benchmark_produces_valid_scores() {
    let dir = tempfile::TempDir::new().unwrap();
    create_synthetic_locomo(dir.path());

    let result = run_locomo_benchmark(dir.path());

    assert!(result.total_queries > 0);
    assert!(result.total_time_secs > 0.0);
    assert!(result.ingest_time_secs > 0.0);
    assert!(result.query_time_secs > 0.0);

    // LoCoMo should produce at least 4 of 5 standard categories.
    assert!(
        result.categories.len() >= 4,
        "expected >= 4 categories, got {}",
        result.categories.len(),
    );

    for cat in &result.categories {
        assert!(
            cat.containment.is_finite(),
            "{}: containment not finite",
            cat.name
        );
        assert!(
            cat.token_f1.is_finite(),
            "{}: token_f1 not finite",
            cat.name
        );
        assert!(cat.total > 0, "{}: should have queries", cat.name);
    }

    // All metrics finite and non-negative.
    assert!(result.overall_containment >= 0.0 && result.overall_containment.is_finite());
    assert!(result.overall_token_f1 >= 0.0 && result.overall_token_f1.is_finite());
    assert!(result.overall_recall_accuracy >= 0.0 && result.overall_recall_accuracy.is_finite());
    assert!(result.overall_mrr >= 0.0 && result.overall_mrr.is_finite());
    assert!(result.overall_ndcg >= 0.0 && result.overall_ndcg.is_finite());
}
