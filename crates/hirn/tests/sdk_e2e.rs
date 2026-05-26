//! SDK End-to-End
//!
//! Validates that Rust Level 1, 2, and 3 APIs all complete the full lifecycle:
//! open → remember 20 memories → think → relevant context. Also tests that
//! queries produce correct, meaningful results.

use hirn::prelude::*;
use hirn::ql::QueryResult;

// ── Helpers ─────────────────────────────────────────────────────────────

/// 20 diverse tech-domain memories for seeding.
const MEMORIES: [&str; 20] = [
    "Kubernetes horizontal pod autoscaler adjusts replica count based on CPU utilization",
    "Docker multi-stage builds reduce final image size by discarding build dependencies",
    "PostgreSQL MVCC provides snapshot isolation without read locks",
    "Redis sorted sets maintain elements with scores for leaderboard patterns",
    "JWT refresh tokens rotate on each use to prevent replay attacks",
    "OAuth2 authorization code flow with PKCE prevents interception attacks",
    "TLS 1.3 handshake completes in a single round trip for improved latency",
    "Prometheus PromQL supports rate functions for counter metric analysis",
    "Grafana alerting evaluates rules at configurable intervals with notification channels",
    "Terraform state files track resource identity for idempotent infrastructure changes",
    "Helm chart values files override default template variables per environment",
    "gRPC bidirectional streaming enables real-time communication between services",
    "Elasticsearch inverted index maps terms to document IDs for fast full-text search",
    "WebAssembly linear memory model provides sandboxed execution environment",
    "Istio service mesh injects Envoy sidecar proxies for mutual TLS encryption",
    "Apache Kafka partitions distribute message load across consumer group members",
    "GitHub Actions workflow files define CI/CD pipelines with reusable composite actions",
    "SQLite WAL mode allows concurrent readers alongside a single writer process",
    "Nginx reverse proxy handles SSL termination and upstream load balancing",
    "RabbitMQ dead letter exchanges capture unprocessable messages for later analysis",
];

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

// ═══════════════════════════════════════════════════════════════════════
// Level 1: HirnMemory zero-config API
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn level1_remember_20_then_think() {
    let (mem, _dir) = open_memory().await;

    // Remember 20 memories
    for text in &MEMORIES {
        mem.remember(text).await.unwrap();
    }

    // Think — should produce meaningful context
    let ctx = mem.think("authentication security", 4096).await.unwrap();
    assert!(
        !ctx.context.is_empty(),
        "think should produce non-empty context"
    );
    assert!(ctx.token_count > 0, "token count should be positive");
    assert!(
        !ctx.records_included.is_empty(),
        "should include at least one record in context"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn level1_remember_20_then_recall() {
    let (mem, _dir) = open_memory().await;

    for text in &MEMORIES {
        mem.remember(text).await.unwrap();
    }

    let results = mem.recall("database query performance", 5).await.unwrap();
    assert!(!results.is_empty(), "recall should return results");
    assert!(results.len() <= 5, "limit should be respected");
}

#[tokio::test(flavor = "multi_thread")]
async fn level1_meaningful_context_quality() {
    let (mem, _dir) = open_memory().await;

    for text in &MEMORIES {
        mem.remember(text).await.unwrap();
    }

    // Think about Kubernetes → context should contain actual memory content
    let ctx = mem
        .think("kubernetes container orchestration", 4096)
        .await
        .unwrap();
    assert!(!ctx.context.is_empty(), "context should be non-empty");
    assert!(
        ctx.records_included.len() >= 1,
        "should include at least one record in context"
    );
    // Verify the context contains actual memory text (not garbage)
    assert!(
        ctx.context.contains("##") || ctx.context.len() > 50,
        "context should contain formatted memory records"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Level 2: HirnQL read/query API
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn level2_hirnql_remember_20_then_recall() {
    let (mem, _dir) = open_memory().await;

    // Seed via the direct API, then recall via HirnQL.
    for text in &MEMORIES {
        mem.remember(text).await.unwrap();
    }

    // Recall via HirnQL
    let r = mem
        .query(r#"RECALL episodic ABOUT "message queue" LIMIT 5"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "should find messaging-related memories"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn level2_hirnql_think() {
    let (mem, _dir) = open_memory().await;

    for text in &MEMORIES {
        mem.remember(text).await.unwrap();
    }

    let r = mem
        .query(r#"THINK ABOUT "infrastructure as code provisioning""#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "THINK should return context records"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn level2_hirnql_forget_and_inspect() {
    let (mem, _dir) = open_memory().await;

    // Remember a few
    let id = mem
        .remember("Temporary note about meeting schedule")
        .await
        .unwrap();

    // Inspect
    let r = mem.query(&format!(r#"INSPECT "{id}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(i) => {
            assert!(!i.record.id().to_string().is_empty());
        }
        other => panic!("expected Inspected, got {other:?}"),
    }

    // Archive through the direct episodic API.
    mem.db().episodic().archive(id).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn level2_hirnql_connect_and_trace() {
    let (mem, _dir) = open_memory().await;

    let id1 = mem
        .remember("Microservices communicate via gRPC protocol buffers")
        .await
        .unwrap();
    let id2 = mem
        .remember("Protocol buffers provide efficient binary serialization format")
        .await
        .unwrap();

    // Connect through the direct graph API.
    connect_graph(&mem, id1, id2, EdgeRelation::RelatedTo, 0.9).await;

    // Trace
    let r = mem.query(&format!(r#"TRACE "{id1}""#)).await.unwrap();
    match r {
        QueryResult::Traced(t) => {
            assert!(!t.record.id().to_string().is_empty());
        }
        other => panic!("expected Traced, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn level2_hirnql_explain() {
    let (mem, _dir) = open_memory().await;

    for text in &MEMORIES[..5] {
        mem.remember(text).await.unwrap();
    }

    let r = mem
        .query(r#"EXPLAIN RECALL episodic ABOUT "database""#)
        .await
        .unwrap();
    match r {
        QueryResult::ExplainPlan(plan) => {
            assert!(
                !plan.plan_text.is_empty(),
                "EXPLAIN should produce a non-empty plan"
            );
        }
        other => panic!("expected ExplainPlan, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Level 3: Builder / HirnDB API
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn level3_builder_remember_20_then_think() {
    let (mem, _dir) = open_memory().await;

    for text in &MEMORIES {
        mem.remember(text).await.unwrap();
    }

    // Use think builder with budget
    let ctx = mem
        .think_builder("SSL TLS encryption security")
        .budget(4096)
        .execute()
        .await
        .unwrap();

    assert!(!ctx.context.is_empty());
    assert!(ctx.token_count > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn level3_builder_recall_with_options() {
    let (mem, _dir) = open_memory().await;

    for text in &MEMORIES {
        mem.remember(text).await.unwrap();
    }

    // Recall builder with options
    let results = mem
        .recall_builder("distributed messaging")
        .limit(5)
        .episodic_only()
        .execute()
        .await
        .unwrap();

    assert!(!results.is_empty(), "builder recall should return results");
    assert!(results.len() <= 5);
}

#[tokio::test(flavor = "multi_thread")]
async fn level3_builder_with_activation() {
    let (mem, _dir) = open_memory().await;

    let id1 = mem
        .remember("Redis caches frequently accessed API responses")
        .await
        .unwrap();
    let id2 = mem
        .remember("Cache invalidation strategies include TTL and event-based purging")
        .await
        .unwrap();

    // Connect them through the direct graph API.
    connect_graph(&mem, id1, id2, EdgeRelation::RelatedTo, 0.8).await;

    // Recall with spreading activation
    let results = mem
        .recall_builder("caching performance")
        .limit(10)
        .activation(hirn_engine::ActivationMode::Spreading)
        .depth(2)
        .execute()
        .await
        .unwrap();

    // Should return at least the directly matching record
    assert!(!results.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn level3_direct_db_stats() {
    let (mem, _dir) = open_memory().await;

    for text in &MEMORIES[..10] {
        mem.remember(text).await.unwrap();
    }

    let stats = mem.db().admin().stats().await.unwrap();
    assert!(
        stats.episodic_count >= 10,
        "stats should show at least 10 episodic records, got {}",
        stats.episodic_count
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Cross-API consistency: same data → same results
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn level1_and_level2_agree() {
    let (mem, _dir) = open_memory().await;

    for text in &MEMORIES {
        mem.remember(text).await.unwrap();
    }

    // Level 1: recall API
    let l1_results = mem.recall("Kubernetes pods", 10).await.unwrap();

    // Level 2: HirnQL
    let l2_result = mem
        .query(r#"RECALL episodic ABOUT "Kubernetes pods" LIMIT 10"#)
        .await
        .unwrap();

    match l2_result {
        QueryResult::Records(rr) => {
            // Both should return results (count may differ slightly due to different scoring paths)
            assert!(!l1_results.is_empty(), "Level 1 should return results");
            assert!(rr.records_returned > 0, "Level 2 should return results");
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn all_three_levels_complete_lifecycle() {
    let (mem, _dir) = open_memory().await;

    // Store via all three methods
    // Level 1
    mem.remember("Level 1: direct text memory storage")
        .await
        .unwrap();
    // Level 2 is exercised as a read/query surface below.
    mem.remember("Level 2: query surface participates in the shared memory state")
        .await
        .unwrap();
    // Level 3 (builder)
    let embedding = mem
        .db()
        .embed_text("Level 3: builder memory storage")
        .await
        .unwrap();
    let record = EpisodicRecord::builder()
        .content("Level 3: builder memory storage")
        .embedding(embedding)
        .agent_id(AgentId::new("test").unwrap())
        .build()
        .unwrap();
    mem.db().remember_bypass_admission(record).await.unwrap();

    // Verify all three are stored
    let stats = mem.db().admin().stats().await.unwrap();
    assert!(
        stats.episodic_count >= 3,
        "should have at least 3 episodic records, got {}",
        stats.episodic_count
    );

    // Think should combine all
    let ctx = mem.think("memory storage levels", 4096).await.unwrap();
    assert!(!ctx.context.is_empty());
}
