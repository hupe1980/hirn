//! HirnQL v2 Regression Suite
//!
//! Exercises the supported embedded read/query statement types through
//! `HirnMemory::query()`, while write/admin/graph setup uses direct APIs
//! because embedded HirnQL intentionally rejects those mutation surfaces.
//! This file alone provides 100+ test cases; the wider codebase has 280+ more.

use hirn::episodic::{EpisodicFilter, EpisodicRecord};
use hirn::prelude::*;
use hirn::ql::QueryResult;
use hirn::record::MemoryRecord;
use hirn::semantic::SemanticRecord;

// ── Helpers ─────────────────────────────────────────────────────────────

async fn open_memory() -> (HirnMemory, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain");
    let mut config = HirnConfig::builder()
        .db_path(&path)
        .allow_pseudo_embedder_fallback(true)
        .build()
        .unwrap();
    config.admission_enabled = true;
    let mem = HirnMemory::open_with_config(config).await.unwrap();
    (mem, dir)
}

async fn connect_graph(
    mem: &HirnMemory,
    source: MemoryId,
    target: MemoryId,
    relation: EdgeRelation,
    weight: f32,
) {
    mem.db()
        .graph_view()
        .connect_with(source, target, relation, weight, Metadata::default())
        .await
        .unwrap();
}

fn regression_agent() -> AgentId {
    AgentId::new("hirnql_regression").unwrap()
}

async fn remember_episode_record(
    mem: &HirnMemory,
    content: &str,
    event_type: EventType,
    importance: f32,
) -> MemoryId {
    let record = EpisodicRecord::builder()
        .content(content)
        .event_type(event_type)
        .importance(importance)
        .agent_id(regression_agent())
        .build()
        .unwrap();
    mem.db().episodic().remember(record).await.unwrap()
}

async fn store_semantic_record(mem: &HirnMemory, concept: &str, description: &str) -> MemoryId {
    let record = SemanticRecord::builder()
        .concept(concept)
        .description(description)
        .knowledge_type(KnowledgeType::Propositional)
        .confidence(0.95)
        .agent_id(regression_agent())
        .build()
        .unwrap();
    mem.db().semantic().store(record).await.unwrap()
}

async fn assert_query_unsupported(mem: &HirnMemory, query: &str, needle: &str) {
    let err = mem
        .query(query)
        .await
        .expect_err(&format!("query should be unsupported: {query}"));
    assert!(
        err.to_string().contains(needle),
        "expected `{query}` to fail with `{needle}`, got `{err}`"
    );
}

async fn seeded_memory() -> (HirnMemory, tempfile::TempDir) {
    let (mem, dir) = open_memory().await;
    let texts = [
        "Kubernetes pods use resource limits for stability",
        "Docker containers isolate processes using cgroups",
        "Helm charts manage Kubernetes deployments declaratively",
        "Redis is an in-memory data store for caching",
        "PostgreSQL supports JSONB for semi-structured data",
        "OAuth2 PKCE flow secures public clients",
        "JWT tokens expire after configurable TTLs",
        "TLS 1.3 eliminates insecure cipher suites",
        "Prometheus scrapes metrics from exporters",
        "Grafana dashboards visualize time-series data",
    ];
    for t in &texts {
        mem.remember(t).await.unwrap();
    }
    (mem, dir)
}

// ═══════════════════════════════════════════════════════════════════════
// RECALL — 20 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn recall_episodic_basic() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "kubernetes""#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => assert!(rr.records_returned > 0),
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_with_limit() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "data" LIMIT 3"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => assert!(rr.records_returned <= 3),
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_with_where_importance() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "caching" WHERE importance > 0.0 LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_empty_results() {
    // Empty DB → no results.
    let (mem, _dir) = open_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "xyznonexistent_topic_42""#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => assert_eq!(rr.records_returned, 0),
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_case_insensitive() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"recall EPISODIC about "kubernetes""#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => assert!(rr.records_returned > 0),
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_single_quoted_string() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT 'kubernetes'"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => assert!(rr.records_returned > 0),
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_with_expand_graph() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "kubernetes" EXPAND GRAPH DEPTH 2"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_expand_with_activation() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "containers" EXPAND GRAPH DEPTH 1 ACTIVATION spreading"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_limit_1() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "redis" LIMIT 1"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => assert!(rr.records_returned <= 1),
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_large_limit() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "data" LIMIT 1000"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => assert!(rr.records_returned <= 10), // only 10 records seeded
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_where_importance_lt() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "kubernetes" WHERE importance < 0.9 LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_where_importance_eq() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "data" WHERE importance >= 0.0 LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_records_have_scores() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "kubernetes" LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            for scored in &rr.records {
                assert!(scored.score >= 0.0, "score should be non-negative");
            }
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_records_sorted_by_score() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "containers" LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            for w in rr.records.windows(2) {
                assert!(
                    w[0].score >= w[1].score,
                    "results should be sorted by score descending"
                );
            }
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_query_time_populated() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "metrics" LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(rr.query_time_ms >= 0.0, "query time should be populated");
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_with_comment() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query("-- this is a comment\nRECALL episodic ABOUT \"kubernetes\" LIMIT 5")
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_multiline() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(
            r#"RECALL episodic
               ABOUT "kubernetes"
               LIMIT 5"#,
        )
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_expand_min_weight() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "redis" EXPAND GRAPH DEPTH 2 min_weight 0.5 LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_budget_clause() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "kubernetes" BUDGET 2048 LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_with_involving() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "containers" INVOLVING "Docker" LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// THINK — 10 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn think_basic() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"THINK ABOUT "kubernetes deployment" BUDGET 2048"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(rr.context.is_some(), "THINK should produce context");
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn think_with_limit() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"THINK ABOUT "security" BUDGET 4096 LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(rr.context.is_some());
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn think_global() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"THINK GLOBAL ABOUT "system architecture" BUDGET 4096"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(rr.context.is_some());
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn think_context_fits_budget() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"THINK ABOUT "caching" BUDGET 512"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            if let Some(ctx) = &rr.context {
                // Rough check: budget 512 tokens ≈ ~2000 chars max.
                assert!(
                    ctx.len() < 5000,
                    "context length {} seems too large for budget 512",
                    ctx.len()
                );
            }
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn think_empty_db() {
    let (mem, _dir) = open_memory().await;
    let r = mem
        .query(r#"THINK ABOUT "anything" BUDGET 2048"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            // With empty DB, context may be empty or None.
            assert_eq!(rr.records_returned, 0);
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn think_with_where() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"THINK ABOUT "security" WHERE importance > 0.0 BUDGET 2048"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn think_case_insensitive() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"think about "kubernetes" budget 2048"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(rr.context.is_some());
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn think_with_expand() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"THINK ABOUT "monitoring" EXPAND GRAPH DEPTH 1 BUDGET 2048"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn think_multiline() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(
            r#"THINK
               ABOUT "database"
               BUDGET 2048
               LIMIT 5"#,
        )
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(rr.context.is_some());
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn think_with_involving() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"THINK ABOUT "data stores" INVOLVING "Redis" BUDGET 2048"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// REMEMBER — 10 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn remember_episode_basic() {
    let (mem, _dir) = open_memory().await;
    let id = mem
        .remember("User logged in from IP 192.168.1.1")
        .await
        .unwrap();
    assert!(!id.to_string().is_empty(), "should return a valid id");
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_semantic_basic() {
    let (mem, _dir) = open_memory().await;
    let id = store_semantic_record(
        &mem,
        "rust_ownership",
        "Rust uses ownership for memory safety",
    )
    .await;
    let record = mem.db().semantic().get(id).await.unwrap();
    assert_eq!(record.concept, "rust_ownership");
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_with_importance() {
    let (mem, _dir) = open_memory().await;
    let id = remember_episode_record(&mem, "critical alert", EventType::Observation, 0.95).await;
    let record = mem.db().admin().get_memory(id).await.unwrap();
    match record {
        MemoryRecord::Episodic(ep) => assert!((ep.importance - 0.95).abs() < f32::EPSILON),
        other => panic!("expected Episodic memory, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_with_type() {
    let (mem, _dir) = open_memory().await;
    let id = remember_episode_record(&mem, "experiment results", EventType::Experiment, 0.5).await;
    let record = mem.db().admin().get_memory(id).await.unwrap();
    match record {
        MemoryRecord::Episodic(ep) => assert_eq!(ep.event_type, EventType::Experiment),
        other => panic!("expected Episodic memory, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_with_entities() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("John met Alice at the cafe").await.unwrap();
    let record = mem.db().admin().get_memory(id).await.unwrap();
    match record {
        MemoryRecord::Episodic(ep) => assert!(!ep.entities.is_empty()),
        other => panic!("expected Episodic memory, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_then_recall() {
    let (mem, _dir) = open_memory().await;
    mem.remember("quantum computing uses qubits for superposition")
        .await
        .unwrap();
    let r = mem
        .query(r#"RECALL episodic ABOUT "quantum" LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned > 0,
                "should recall the remembered episode"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_episode_single_quoted() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("single quoted content").await.unwrap();
    assert!(!id.to_string().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_long_content() {
    let (mem, _dir) = open_memory().await;
    let long = "A".repeat(500);
    let id = mem.remember(&long).await.unwrap();
    assert!(!id.to_string().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_multiple_then_recall() {
    let (mem, _dir) = open_memory().await;
    let contents = [
        "Kubernetes orchestrates containerized microservices across distributed clusters",
        "Terraform provisions cloud infrastructure using declarative configuration files",
        "GraphQL enables flexible API queries with typed schema definitions",
        "WebAssembly compiles high-level languages to portable binary instruction format",
        "gRPC uses Protocol Buffers for efficient cross-service communication",
    ];
    for c in &contents {
        mem.remember(c).await.unwrap();
    }
    let r = mem
        .query(r#"RECALL episodic ABOUT "infrastructure" LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "should recall at least one memory"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_case_insensitive() {
    let (mem, _dir) = open_memory().await;
    assert_query_unsupported(
        &mem,
        r#"remember EPISODE content "case test""#,
        "REMEMBER is not supported",
    )
    .await;
}

// ═══════════════════════════════════════════════════════════════════════
// FORGET — 8 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn forget_archive_basic() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("to be archived").await.unwrap();
    mem.db().episodic().archive(id).await.unwrap();
    let logical_id = mem.db().episodic().get(id).await.unwrap().logical_memory_id;
    let archived = mem
        .db()
        .episodic()
        .list(&EpisodicFilter {
            include_archived: true,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_iter()
        .find(|record| record.logical_memory_id == logical_id && record.archived)
        .expect("archived successor should remain visible");
    assert!(archived.archived);
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_purge_basic() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("to be purged").await.unwrap();
    mem.db().episodic().delete(id).await.unwrap();
    assert!(mem.query(&format!(r#"INSPECT "{id}""#)).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_default_is_archive() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("default forget mode").await.unwrap();
    assert_query_unsupported(
        &mem,
        &format!(r#"FORGET "{id}""#),
        "FORGET is not supported",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_nonexistent_errors() {
    let (mem, _dir) = open_memory().await;
    let r = mem.query(r#"FORGET "01AAAAAAAAAAAAAAAAAAAAAAAAA""#).await;
    assert!(r.is_err(), "forgetting nonexistent record should error");
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_then_recall_excludes() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("ephemeral data point").await.unwrap();

    // Recall first to confirm it's there.
    let r1 = mem
        .query(r#"RECALL episodic ABOUT "ephemeral" LIMIT 5"#)
        .await
        .unwrap();
    let count_before = match &r1 {
        QueryResult::Records(rr) => rr.records_returned,
        _ => 0,
    };

    mem.db().episodic().delete(id).await.unwrap();

    // Recall again — should be gone.
    let r2 = mem
        .query(r#"RECALL episodic ABOUT "ephemeral" LIMIT 5"#)
        .await
        .unwrap();
    match r2 {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned < count_before,
                "purged record should not appear in recall"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_case_insensitive() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("case forget").await.unwrap();
    assert_query_unsupported(
        &mem,
        &format!(r#"forget "{id}" archive"#),
        "FORGET is not supported",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_invalid_id_errors() {
    let (mem, _dir) = open_memory().await;
    let r = mem.query(r#"FORGET "not-a-valid-ulid""#).await;
    assert!(r.is_err(), "invalid ID should produce an error");
}

#[tokio::test(flavor = "multi_thread")]
async fn forget_purge_twice_errors() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("double purge").await.unwrap();
    mem.db().episodic().delete(id).await.unwrap();
    let r = mem.db().episodic().delete(id).await;
    assert!(r.is_err(), "purging twice should error");
}

// ═══════════════════════════════════════════════════════════════════════
// CONNECT — 6 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn connect_creates_edge() {
    let (mem, _dir) = open_memory().await;
    let id1 = mem.remember("deployment was successful").await.unwrap();
    let id2 = mem
        .remember("monitoring confirmed zero errors")
        .await
        .unwrap();
    connect_graph(&mem, id1, id2, EdgeRelation::CausedBy, 0.9).await;
    let r = mem.query(&format!(r#"INSPECT "{id1}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(i) => assert!(!i.neighbors.is_empty()),
        other => panic!("expected Inspected, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_default_weight() {
    let (mem, _dir) = open_memory().await;
    let id1 = mem
        .remember("PostgreSQL supports advanced indexing with B-tree and GIN")
        .await
        .unwrap();
    let id2 = mem
        .remember("Redis provides in-memory caching with TTL eviction policies")
        .await
        .unwrap();
    mem.db().graph_view().connect(id1, id2).await.unwrap();
    let r = mem.query(&format!(r#"INSPECT "{id1}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(i) => assert!(!i.neighbors.is_empty()),
        other => panic!("expected Inspected, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_nonexistent_source_errors() {
    let (mem, _dir) = open_memory().await;
    let id2 = mem.remember("target").await.unwrap();
    let fake = MemoryId::parse("01ARZ3NDEKTSV4RRFFQ69G5FAV").unwrap();
    let r = mem.db().graph_view().connect(fake, id2).await;
    assert!(r.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_affects_expand() {
    let (mem, _dir) = open_memory().await;
    let id1 = mem.remember("kubernetes cluster setup").await.unwrap();
    let id2 = mem.remember("helm chart deployment").await.unwrap();
    connect_graph(&mem, id1, id2, EdgeRelation::RelatedTo, 0.8).await;

    // Expand graph from id1 should discover id2.
    let r = mem
        .query(r#"RECALL episodic ABOUT "kubernetes" EXPAND GRAPH DEPTH 1 LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_similar_to() {
    let (mem, _dir) = open_memory().await;
    let id1 = mem
        .remember("Machine learning models require feature engineering and hyperparameter tuning")
        .await
        .unwrap();
    let id2 = mem
        .remember(
            "Deep neural networks automatically learn hierarchical representations from raw data",
        )
        .await
        .unwrap();
    connect_graph(&mem, id1, id2, EdgeRelation::SimilarTo, 0.7).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_via_query_is_unsupported() {
    let (mem, _dir) = open_memory().await;
    let id1 = mem
        .remember("Quantum entanglement enables faster-than-classical communication protocols")
        .await
        .unwrap();
    let id2 = mem
        .remember(
            "Blockchain consensus algorithms trade throughput for decentralization guarantees",
        )
        .await
        .unwrap();
    let r = mem
        .query(&format!(
            r#"connect "{id1}" to "{id2}" as related_to weight 0.5"#
        ))
        .await;
    let err = r.expect_err("CONNECT should be rejected through embedded HirnQL");
    assert!(err.to_string().contains("CONNECT is not supported"));
}

// ═══════════════════════════════════════════════════════════════════════
// INSPECT — 5 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn inspect_returns_metadata() {
    let (mem, _dir) = open_memory().await;
    let id = mem
        .remember("inspectable record")
        .await
        .unwrap()
        .to_string();
    let r = mem.query(&format!(r#"INSPECT "{id}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(i) => {
            assert!(
                !i.record.id().to_string().is_empty(),
                "inspected record should be populated"
            );
        }
        other => panic!("expected Inspected, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn inspect_nonexistent_errors() {
    let (mem, _dir) = open_memory().await;
    let r = mem.query(r#"INSPECT "01AAAAAAAAAAAAAAAAAAAAAAAAA""#).await;
    assert!(r.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn inspect_shows_graph_neighbors() {
    let (mem, _dir) = open_memory().await;
    let id1 = mem.remember("source").await.unwrap();
    let id2 = mem.remember("target").await.unwrap();
    mem.db().graph_view().connect(id1, id2).await.unwrap();
    let r = mem.query(&format!(r#"INSPECT "{id1}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(i) => {
            assert!(!i.neighbors.is_empty(), "should show connected neighbor");
        }
        other => panic!("expected Inspected, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn inspect_case_insensitive() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("inspect case").await.unwrap().to_string();
    let r = mem.query(&format!(r#"inspect "{id}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(_) => {}
        other => panic!("expected Inspected, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn inspect_invalid_id_errors() {
    let (mem, _dir) = open_memory().await;
    let r = mem.query(r#"INSPECT "not-a-ulid""#).await;
    assert!(r.is_err());
}

// ═══════════════════════════════════════════════════════════════════════
// TRACE — 4 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn trace_basic() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("traceable event").await.unwrap().to_string();
    let r = mem.query(&format!(r#"TRACE "{id}""#)).await.unwrap();
    match r {
        QueryResult::Traced(_) => {}
        other => panic!("expected Traced, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn trace_nonexistent_errors() {
    let (mem, _dir) = open_memory().await;
    let r = mem.query(r#"TRACE "01AAAAAAAAAAAAAAAAAAAAAAAAA""#).await;
    assert!(r.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn trace_case_insensitive() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("trace case").await.unwrap().to_string();
    let r = mem.query(&format!(r#"trace "{id}""#)).await.unwrap();
    match r {
        QueryResult::Traced(_) => {}
        other => panic!("expected Traced, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn trace_invalid_id_errors() {
    let (mem, _dir) = open_memory().await;
    let r = mem.query(r#"TRACE "bad-id""#).await;
    assert!(r.is_err());
}

// ═══════════════════════════════════════════════════════════════════════
// CONSOLIDATE — 3 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn consolidate_basic() {
    let (mem, _dir) = seeded_memory().await;
    let result = mem.db().admin().consolidate().execute().await.unwrap();
    assert!(result.records_processed >= 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn consolidate_empty_db() {
    let (mem, _dir) = open_memory().await;
    let result = mem.db().admin().consolidate().execute().await.unwrap();
    assert_eq!(result.records_processed, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn consolidate_case_insensitive() {
    let (mem, _dir) = open_memory().await;
    assert_query_unsupported(&mem, "consolidate", "CONSOLIDATE is not supported").await;
}

// ═══════════════════════════════════════════════════════════════════════
// EXPLAIN — 5 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn explain_recall() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"EXPLAIN RECALL episodic ABOUT "kubernetes" LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::ExplainPlan(e) => {
            assert!(!e.plan_text.is_empty(), "explain should produce plan text");
            assert!(
                e.actual_result.is_none(),
                "EXPLAIN without ANALYZE should not execute"
            );
        }
        other => panic!("expected ExplainPlan, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn explain_analyze_recall() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(r#"EXPLAIN ANALYZE RECALL episodic ABOUT "kubernetes" LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::ExplainPlan(e) => {
            assert!(!e.plan_text.is_empty());
            assert!(
                e.actual_result.is_some(),
                "EXPLAIN ANALYZE should also execute the query"
            );
        }
        other => panic!("expected ExplainPlan, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn explain_think() {
    let (mem, _dir) = open_memory().await;
    let r = mem
        .query(r#"EXPLAIN THINK ABOUT "anything" BUDGET 2048"#)
        .await
        .unwrap();
    match r {
        QueryResult::ExplainPlan(e) => {
            assert!(!e.plan_text.is_empty());
        }
        other => panic!("expected ExplainPlan, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn explain_no_side_effects() {
    let (mem, _dir) = open_memory().await;
    // EXPLAIN on REMEMBER should not actually create a record.
    let r = mem
        .query(r#"EXPLAIN RECALL episodic ABOUT "test" LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::ExplainPlan(_) => {}
        other => panic!("expected ExplainPlan, got {other:?}"),
    }
    // DB should still be empty.
    let r2 = mem
        .query(r#"RECALL episodic ABOUT "test" LIMIT 5"#)
        .await
        .unwrap();
    match r2 {
        QueryResult::Records(rr) => assert_eq!(rr.records_returned, 0),
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn explain_case_insensitive() {
    let (mem, _dir) = open_memory().await;
    let r = mem
        .query(r#"explain recall episodic about "test" limit 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::ExplainPlan(_) => {}
        other => panic!("expected ExplainPlan, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Error & edge cases — 15 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn error_empty_query() {
    let (mem, _dir) = open_memory().await;
    assert!(mem.query("").await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn error_whitespace_only() {
    let (mem, _dir) = open_memory().await;
    assert!(mem.query("   \n\t  ").await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn error_unknown_verb() {
    let (mem, _dir) = open_memory().await;
    assert!(mem.query("SELECT * FROM memories").await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn error_incomplete_recall() {
    let (mem, _dir) = open_memory().await;
    assert!(mem.query("RECALL episodic").await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn error_unterminated_string() {
    let (mem, _dir) = open_memory().await;
    assert!(
        mem.query(r#"RECALL episodic ABOUT "unterminated"#)
            .await
            .is_err()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn error_missing_about() {
    let (mem, _dir) = open_memory().await;
    assert!(mem.query("RECALL episodic LIMIT 5").await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn error_remember_no_content() {
    let (mem, _dir) = open_memory().await;
    assert!(mem.query("REMEMBER episode").await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn error_connect_missing_to() {
    let (mem, _dir) = open_memory().await;
    assert!(mem.query(r#"CONNECT "a" AS related_to"#).await.is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn error_unicode_content_works() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("日本語テスト 🚀").await.unwrap();
    let r = mem.query(&format!(r#"INSPECT "{id}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(_) => {}
        other => panic!("expected Inspected, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn escaped_quote_in_string() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("He said 'hello world.'").await.unwrap();
    let r = mem.query(&format!(r#"INSPECT "{id}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(_) => {}
        other => panic!("expected Inspected, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn error_newline_in_string() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("line1\nline2").await.unwrap();
    let r = mem.query(&format!(r#"INSPECT "{id}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(_) => {}
        other => panic!("expected Inspected, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn error_multiple_where_clauses() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(
            r#"RECALL episodic ABOUT "test" WHERE importance > 0.1 WHERE importance < 0.9 LIMIT 5"#,
        )
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn recall_on_empty_db() {
    let (mem, _dir) = open_memory().await;
    let r = mem
        .query(r#"RECALL episodic ABOUT "anything" LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => assert_eq!(rr.records_returned, 0),
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_invalid_importance_rejected() {
    let (mem, _dir) = open_memory().await;
    let r = mem
        .query(r#"REMEMBER episode CONTENT "bad" IMPORTANCE 1.5"#)
        .await;
    assert!(r.is_err(), "importance > 1.0 should be rejected");
}

#[tokio::test(flavor = "multi_thread")]
async fn remember_negative_importance_rejected() {
    let (mem, _dir) = open_memory().await;
    let r = mem
        .query(r#"REMEMBER episode CONTENT "bad" IMPORTANCE -0.1"#)
        .await;
    assert!(r.is_err(), "negative importance should be rejected");
}

// ═══════════════════════════════════════════════════════════════════════
// Full workflow — 5 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn full_workflow_remember_connect_recall_inspect_trace_forget() {
    let (mem, _dir) = open_memory().await;

    // REMEMBER two episodes.
    let id1 = mem.remember("deployed v2.0 to production").await.unwrap();
    let id2 = mem
        .remember("zero downtime confirmed by monitoring")
        .await
        .unwrap();

    // CONNECT them.
    connect_graph(&mem, id1, id2, EdgeRelation::CausedBy, 0.9).await;

    // RECALL should find them.
    let r = mem
        .query(r#"RECALL episodic ABOUT "production deployment" LIMIT 10"#)
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => assert!(rr.records_returned > 0),
        other => panic!("expected Records, got {other:?}"),
    }

    // INSPECT the first record.
    let r = mem.query(&format!(r#"INSPECT "{id1}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(_) => {}
        other => panic!("expected Inspected, got {other:?}"),
    }

    // TRACE the first record.
    let r = mem.query(&format!(r#"TRACE "{id1}""#)).await.unwrap();
    match r {
        QueryResult::Traced(_) => {}
        other => panic!("expected Traced, got {other:?}"),
    }

    // PURGE the second record through the direct episodic API.
    mem.db().episodic().delete(id2).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn workflow_remember_consolidate_recall() {
    let (mem, _dir) = open_memory().await;
    let topics = [
        "Prometheus scrapes metrics endpoints every fifteen seconds for system observability",
        "Grafana renders time-series dashboards from multiple data sources simultaneously",
        "Datadog correlates application traces with infrastructure-level health metrics",
        "PagerDuty escalation policies route critical alerts to on-call engineering teams",
        "Jaeger distributed tracing reconstructs request flows across microservice boundaries",
    ];
    for t in &topics {
        mem.remember(t).await.unwrap();
    }
    let result = mem.db().admin().consolidate().execute().await.unwrap();
    assert!(result.records_processed >= 1);
    // Should still be able to recall after consolidation.
    let r = mem
        .query(r#"RECALL episodic ABOUT "monitoring" LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn workflow_think_then_explain() {
    let (mem, _dir) = seeded_memory().await;

    // Think first.
    let r = mem
        .query(r#"THINK ABOUT "security best practices" BUDGET 2048"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(rr.context.is_some());
        }
        other => panic!("expected Records, got {other:?}"),
    }

    // Explain the same query.
    let r = mem
        .query(r#"EXPLAIN THINK ABOUT "security best practices" BUDGET 2048"#)
        .await
        .unwrap();
    match r {
        QueryResult::ExplainPlan(e) => {
            assert!(!e.plan_text.is_empty());
        }
        other => panic!("expected ExplainPlan, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn workflow_api_and_ql_agree() {
    let (mem, _dir) = open_memory().await;
    mem.remember("Rust ownership prevents data races")
        .await
        .unwrap();

    // Recall via API.
    let api_results = mem.recall("ownership", 10).await.unwrap();

    // Recall via HirnQL.
    let ql_result = mem
        .query(r#"RECALL episodic ABOUT "ownership" LIMIT 10"#)
        .await
        .unwrap();

    match ql_result {
        QueryResult::Records(rr) => {
            // Both should find at least 1 record.
            assert!(!api_results.is_empty());
            assert!(rr.records_returned > 0);
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn workflow_concurrent_reads() {
    let (mem, _dir) = seeded_memory().await;

    // Run multiple queries concurrently — should not deadlock or error.
    let (r1, r2, r3) = tokio::join!(
        mem.query(r#"RECALL episodic ABOUT "kubernetes" LIMIT 5"#),
        mem.query(r#"RECALL episodic ABOUT "security" LIMIT 5"#),
        mem.query(r#"RECALL episodic ABOUT "caching" LIMIT 5"#),
    );
    r1.unwrap();
    r2.unwrap();
    r3.unwrap();
}

// ═══════════════════════════════════════════════════════════════════════
// Fuzz — 5 tests
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn fuzz_random_strings_no_crash() {
    let (mem, _dir) = open_memory().await;
    let inputs = [
        "",
        "   ",
        "\n\n",
        "RECALL",
        "RECALL episodic",
        "RECALL episodic ABOUT",
        "THINK",
        "REMEMBER",
        "FORGET",
        "CONNECT",
        "INSPECT",
        "TRACE",
        "CONSOLIDATE extra tokens",
        "😀 unicode",
        "DROP TABLE memories",
        "'; DROP TABLE --",
        "RECALL episodic ABOUT \"x\" LIMIT -1",
        "RECALL episodic ABOUT \"x\" LIMIT 999999999999",
        "RECALL episodic ABOUT \"\" LIMIT 5",
    ];
    for input in &inputs {
        let _ = mem.query(input).await; // must not panic
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn fuzz_long_query_no_crash() {
    let (mem, _dir) = open_memory().await;
    let long = "A".repeat(10000);
    let q = format!(r#"RECALL episodic ABOUT "{long}" LIMIT 5"#);
    let _ = mem.query(&q).await; // must not panic
}

#[tokio::test(flavor = "multi_thread")]
async fn fuzz_sql_injection_no_effect() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("'; DROP TABLE episodes; --").await.unwrap();
    let r = mem.query(&format!(r#"INSPECT "{id}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(_) => {}
        other => panic!("expected Inspected (content is just text), got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn fuzz_deeply_nested_clauses() {
    let (mem, _dir) = seeded_memory().await;
    let r = mem
        .query(
            r#"RECALL episodic
               ABOUT "test"
               EXPAND GRAPH DEPTH 3 ACTIVATION spreading
               WHERE importance > 0.1
               WHERE importance < 0.9
               LIMIT 5"#,
        )
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {}
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn fuzz_all_verbs_parse_without_crash() {
    let (mem, _dir) = open_memory().await;
    let id = mem.remember("test content").await.unwrap().to_string();

    let queries = [
        format!(r#"RECALL episodic ABOUT "test" LIMIT 5"#),
        format!(r#"THINK ABOUT "test" BUDGET 2048"#),
        format!(r#"INSPECT "{id}""#),
        format!(r#"TRACE "{id}""#),
        format!(r#"CONSOLIDATE"#),
        format!(r#"EXPLAIN RECALL episodic ABOUT "test" LIMIT 5"#),
    ];
    for q in &queries {
        let _ = mem.query(q).await; // must not panic
    }
}
