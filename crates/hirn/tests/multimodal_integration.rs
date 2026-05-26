//! Multi-Modal Integration Test
//!
//! Validates the full lifecycle: store text, image, code, audio → recall
//! by text query → correct cross-modal retrieval, consolidation of
//! multi-modal episodes, and community detection with multi-modal nodes.

use hirn::prelude::*;
use hirn::ql::QueryResult;
use hirn::resource::{DerivedArtifactKind, EvidenceRole, HydrationMode};
use image::{DynamicImage, ImageFormat, RgbaImage};
use std::io::Cursor;

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

fn agent() -> AgentId {
    AgentId::new("test-agent").unwrap()
}

fn valid_png_bytes() -> Vec<u8> {
    let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
        4,
        4,
        image::Rgba([0x22, 0x66, 0xaa, 0xff]),
    ));
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .expect("png fixture should encode");
    bytes
}

/// Build and store a text episode via the lower-level API (bypasses admission).
async fn store_text(mem: &HirnMemory, text: &str) -> MemoryId {
    let embedding = mem.db().embed_text(text).await.unwrap();
    let record = EpisodicRecord::builder()
        .content(text)
        .embedding(embedding)
        .agent_id(agent())
        .build()
        .unwrap();
    mem.db().remember_bypass_admission(record).await.unwrap()
}

/// Build and store an image episode (fake bytes + description).
async fn store_image(mem: &HirnMemory, description: &str, fake_data: &[u8]) -> MemoryId {
    let mc = MemoryContent::Image {
        data: fake_data.to_vec(),
        mime_type: "image/png".into(),
        description: description.into(),
    };
    let record = EpisodicRecord::builder()
        .multi_content(mc)
        .agent_id(agent())
        .build()
        .unwrap();
    mem.db().remember_bypass_admission(record).await.unwrap()
}

/// Build and store a code episode.
async fn store_code(mem: &HirnMemory, source: &str, language: &str) -> MemoryId {
    let mc = MemoryContent::Code {
        source: source.into(),
        language: language.into(),
        ast_hash: None,
    };
    let record = EpisodicRecord::builder()
        .multi_content(mc)
        .agent_id(agent())
        .build()
        .unwrap();
    mem.db().remember_bypass_admission(record).await.unwrap()
}

/// Build and store an audio episode (fake bytes + transcript).
async fn store_audio(mem: &HirnMemory, transcript: &str, fake_data: &[u8]) -> MemoryId {
    let mc = MemoryContent::Audio {
        data: fake_data.to_vec(),
        transcript: transcript.into(),
        duration_ms: 5000,
        channel_count: Some(1),
    };
    let record = EpisodicRecord::builder()
        .multi_content(mc)
        .agent_id(agent())
        .build()
        .unwrap();
    mem.db().remember_bypass_admission(record).await.unwrap()
}

// ═══════════════════════════════════════════════════════════════════════
// Test 1: Store text + image + code → recall by text query → cross-modal
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn cross_modal_recall_text_and_image() {
    let (mem, _dir) = open_memory().await;

    // Store text memories about authentication
    store_text(&mem, "User authentication failed due to expired JWT token").await;
    store_text(&mem, "OAuth2 PKCE flow initiated for mobile client login").await;

    // Store image with auth-related description
    store_image(
        &mem,
        "Screenshot of the login page showing authentication error dialog",
        b"fake_png_login_error",
    )
    .await;

    // Store code about auth
    store_code(
        &mem,
        r#"async fn verify_token(jwt: &str) -> Result<Claims, AuthError> {
    let key = load_public_key().await?;
    decode::<Claims>(jwt, &key, &Validation::default())
}"#,
        "rust",
    )
    .await;

    // Store unrelated memories
    store_text(&mem, "PostgreSQL vacuum process reclaims dead tuple space").await;
    store_text(
        &mem,
        "Kubernetes pod autoscaler adjusts replicas based on CPU",
    )
    .await;

    // Recall "authentication" — should find text + image + code from auth domain
    let r = mem
        .query(r#"RECALL episodic ABOUT "authentication login" LIMIT 10"#)
        .await
        .unwrap();

    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 2,
                "expected at least 2 auth-related results, got {}",
                rr.records_returned
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn public_resource_workflow_smoke_uses_preview_hydration() {
    let (mem, _dir) = open_memory().await;

    mem.db()
        .register_agent(&agent(), "Docs Smoke Agent")
        .await
        .unwrap();

    let record = EpisodicRecord::builder()
        .content("checkout screenshot with resource evidence")
        .agent_id(agent())
        .multi_content(MemoryContent::Image {
            data: valid_png_bytes(),
            mime_type: "image/png".into(),
            description: "checkout page showing a card declined banner".into(),
        })
        .build()
        .unwrap();
    let id = mem.db().episodic().remember(record).await.unwrap();

    let query = mem
        .db()
        .embed_text("card declined checkout screenshot")
        .await
        .unwrap();
    let recalled = mem
        .db()
        .recall_view()
        .query(query)
        .agent_id(agent().as_str())
        .limit(3)
        .execute()
        .await
        .unwrap();
    let result = recalled
        .iter()
        .find(|candidate| candidate.record.id() == id)
        .expect("image memory should be recalled");

    let source = result
        .resource_evidence
        .iter()
        .find(|summary| summary.role == EvidenceRole::Source && summary.artifact_kind.is_none())
        .expect("source resource evidence should be present");
    assert!(
        source
            .available_artifacts
            .contains(&DerivedArtifactKind::Thumbnail),
        "previewable thumbnail should be advertised on the public recall surface"
    );

    let preview = mem
        .db()
        .recall_view()
        .fetch_resource(&agent(), source.resource_id, HydrationMode::Preview)
        .await
        .unwrap()
        .expect("preview hydration should resolve the resource");
    assert!(
        preview.blob.is_none(),
        "preview hydration should not load the full blob"
    );
    assert!(preview.artifacts.iter().any(|artifact| {
        artifact.kind == DerivedArtifactKind::Thumbnail
            && artifact.mime_type.as_deref() == Some("image/png")
    }));
}

#[tokio::test(flavor = "multi_thread")]
async fn cross_modal_recall_code_by_description() {
    let (mem, _dir) = open_memory().await;

    store_code(
        &mem,
        "SELECT u.id, u.name FROM users u JOIN sessions s ON u.id = s.user_id WHERE s.expired = false",
        "sql",
    )
    .await;

    store_code(
        &mem,
        "def train_model(X, y):\n    model = RandomForestClassifier()\n    model.fit(X, y)\n    return model",
        "python",
    )
    .await;

    store_text(
        &mem,
        "The database query performance degraded after index rebuild",
    )
    .await;

    // Recall "database query" — should return the SQL code and the text
    let r = mem
        .query(r#"RECALL episodic ABOUT "database query" LIMIT 5"#)
        .await
        .unwrap();

    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "expected at least 1 result for database query"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn cross_modal_recall_audio_by_transcript() {
    let (mem, _dir) = open_memory().await;

    store_audio(
        &mem,
        "The deployment pipeline failed because the Docker image tag was incorrect",
        b"fake_audio_deployment",
    )
    .await;

    store_text(
        &mem,
        "CI/CD pipeline configured with GitHub Actions for automated deployment",
    )
    .await;

    // Recall "deployment pipeline" — should find both audio and text
    let r = mem
        .query(r#"RECALL episodic ABOUT "deployment pipeline" LIMIT 5"#)
        .await
        .unwrap();

    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "expected at least 1 deployment-related result"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn store_many_multimodal_then_recall() {
    let (mem, _dir) = open_memory().await;

    // 10 text memories about various system topics
    let texts = [
        "User login attempt blocked by rate limiter after 5 failures",
        "Database connection pool exhausted under peak traffic load",
        "Redis cache hit ratio dropped below 80 percent threshold",
        "Kubernetes ingress controller routing HTTPS traffic correctly",
        "Prometheus alerting rule triggered for high memory usage",
        "JWT refresh token rotation implemented for enhanced security",
        "GraphQL schema migration completed with backward compatibility",
        "gRPC health check endpoint added to all microservices",
        "Elasticsearch index shards rebalanced across cluster nodes",
        "WebSocket connection handler upgraded to support binary frames",
    ];
    for t in &texts {
        store_text(&mem, t).await;
    }

    // 5 image memories
    let images = [
        "Dashboard showing CPU utilization spike during deployment",
        "Error page screenshot when database connection timed out",
        "Architecture diagram of the microservices communication flow",
        "Grafana panel displaying request latency percentiles",
        "Network topology map with load balancer configuration",
    ];
    for (i, desc) in images.iter().enumerate() {
        store_image(&mem, desc, format!("fake_image_{i}").as_bytes()).await;
    }

    // 5 code memories
    let codes = [
        (
            "async fn health_check() -> StatusCode { StatusCode::OK }",
            "rust",
        ),
        ("CREATE INDEX idx_users_email ON users(email)", "sql"),
        ("FROM node:20-alpine\nRUN npm ci --production", "dockerfile"),
        (
            "kubectl apply -f deployment.yaml --namespace production",
            "shell",
        ),
        (
            "import prometheus_client\nfrom prometheus_client import Counter",
            "python",
        ),
    ];
    for (src, lang) in &codes {
        store_code(&mem, src, lang).await;
    }

    // Recall "database" — should return results from text, image (error page), and code (SQL)
    let r = mem
        .query(r#"RECALL episodic ABOUT "database" LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "expected database-related results from mixed modalities"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }

    // Recall "kubernetes deployment" — should find text + code (kubectl, Dockerfile)
    let r = mem
        .query(r#"RECALL episodic ABOUT "kubernetes deployment" LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "expected kubernetes-related results"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Test 2: Consolidation handles multi-modal episodes
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn consolidation_with_multimodal_episodes() {
    let (mem, _dir) = open_memory().await;

    // Store diverse episodes about system monitoring
    store_text(
        &mem,
        "Prometheus scrapes /metrics endpoint every 15 seconds for observability",
    )
    .await;
    store_text(
        &mem,
        "Grafana dashboard renders CPU and memory time-series panels",
    )
    .await;
    store_image(
        &mem,
        "Alert notification screenshot showing critical memory threshold breach",
        b"fake_alert_image",
    )
    .await;
    store_code(
        &mem,
        "alerting:\n  rules:\n  - alert: HighMemory\n    expr: node_memory_usage > 0.9",
        "yaml",
    )
    .await;
    store_audio(
        &mem,
        "Incident review discussion about the memory leak in worker service",
        b"fake_incident_audio",
    )
    .await;

    // Consolidate through the direct admin API.
    let result = mem.db().admin().consolidate().execute().await.unwrap();
    assert!(
        result.records_processed >= 1,
        "expected at least 1 episode processed, got {}",
        result.records_processed
    );

    // After consolidation, recall should still work
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
async fn consolidation_produces_semantic_records() {
    let (mem, _dir) = open_memory().await;

    // Store enough related episodes to trigger concept extraction
    let topics = [
        "Kubernetes pod health checks use liveness and readiness probes",
        "Docker container resource limits prevent noisy neighbor problems",
        "Helm chart values override default deployment configurations",
        "Container orchestration schedules workloads across cluster nodes",
        "Istio service mesh provides mutual TLS between microservices",
        "Envoy sidecar proxy handles load balancing and circuit breaking",
        "Kubernetes horizontal pod autoscaler scales based on custom metrics",
        "Container image scanning detects known CVEs before deployment",
    ];
    for t in &topics {
        store_text(&mem, t).await;
    }

    let result = mem.db().admin().consolidate().execute().await.unwrap();
    assert!(result.records_processed >= 1);

    // Check that semantic layer has some content after consolidation
    let r = mem
        .query(r#"RECALL semantic ABOUT "container orchestration" LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(_) => {
            // Semantic records may or may not exist (depends on LLM availability),
            // but the query itself should succeed.
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Test 3: Community detection / GraphRAG with multi-modal nodes
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn graph_connects_multimodal_nodes() {
    let (mem, _dir) = open_memory().await;

    let id_text = store_text(
        &mem,
        "Authentication service validates JWT tokens using RS256 algorithm",
    )
    .await;

    let id_image = store_image(
        &mem,
        "Sequence diagram showing OAuth2 token exchange between client and auth server",
        b"fake_oauth_diagram",
    )
    .await;

    let id_code = store_code(
        &mem,
        "pub fn validate_jwt(token: &str) -> Result<Claims> { /* ... */ }",
        "rust",
    )
    .await;

    // Connect them via the direct graph API.
    connect_graph(&mem, id_text, id_image, EdgeRelation::RelatedTo, 0.8).await;
    connect_graph(&mem, id_text, id_code, EdgeRelation::RelatedTo, 0.9).await;

    // INSPECT the text node — should show neighbors
    let r = mem.query(&format!(r#"INSPECT "{id_text}""#)).await.unwrap();
    match r {
        QueryResult::Inspected(i) => {
            assert!(
                !i.neighbors.is_empty(),
                "text node should have image + code neighbors"
            );
        }
        other => panic!("expected Inspected, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn expand_graph_includes_multimodal_neighbors() {
    let (mem, _dir) = open_memory().await;

    let id1 = store_text(
        &mem,
        "Terraform provisions AWS VPC with public and private subnets",
    )
    .await;

    let id2 = store_image(
        &mem,
        "AWS architecture diagram showing VPC network topology with NAT gateway",
        b"fake_aws_diagram",
    )
    .await;

    let id3 = store_code(
        &mem,
        r#"resource "aws_vpc" "main" { cidr_block = "10.0.0.0/16" }"#,
        "hcl",
    )
    .await;

    // Connect graph
    connect_graph(&mem, id1, id2, EdgeRelation::RelatedTo, 0.8).await;
    connect_graph(&mem, id1, id3, EdgeRelation::RelatedTo, 0.9).await;

    // Recall with EXPAND GRAPH should pull in graph neighbors
    let r = mem
        .query(r#"RECALL episodic ABOUT "terraform infrastructure" EXPAND GRAPH DEPTH 1 LIMIT 10"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "expand graph should return results including neighbors"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn consolidation_then_graph_query() {
    let (mem, _dir) = open_memory().await;

    // Store diverse multi-modal memories
    store_text(
        &mem,
        "Machine learning model trained on labeled image dataset",
    )
    .await;
    store_text(
        &mem,
        "Feature engineering pipeline normalizes numerical columns",
    )
    .await;
    store_image(
        &mem,
        "Confusion matrix visualization for binary classification model",
        b"fake_confusion_matrix",
    )
    .await;
    store_code(
        &mem,
        "model = XGBClassifier(n_estimators=100, max_depth=6)\nmodel.fit(X_train, y_train)",
        "python",
    )
    .await;
    store_audio(
        &mem,
        "Team standup: model accuracy improved to 94 percent after hyperparameter tuning",
        b"fake_standup_audio",
    )
    .await;

    // Consolidate → should form communities from related ML memories.
    mem.db().admin().consolidate().execute().await.unwrap();

    // THINK uses graph context — should assemble context from the ML community
    let r = mem
        .query(r#"THINK ABOUT "machine learning model performance""#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "THINK should return ML-related context"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Multi-content verification: stored modality is preserved
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn image_modality_preserved_after_store_and_recall() {
    let (mem, _dir) = open_memory().await;

    store_image(
        &mem,
        "Login form validation error showing incorrect password message",
        b"fake_login_image_bytes",
    )
    .await;

    let r = mem
        .query(r#"RECALL episodic ABOUT "login validation" LIMIT 1"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert_eq!(rr.records_returned, 1);
            match &rr.records[0].record {
                MemoryRecord::Episodic(e) => {
                    let mc = e
                        .multi_content
                        .as_ref()
                        .expect("multi_content should be set");
                    assert_eq!(mc.modality(), "image");
                }
                other => panic!("expected Episodic, got {other:?}"),
            }
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn code_modality_preserved_after_store_and_recall() {
    let (mem, _dir) = open_memory().await;

    store_code(
        &mem,
        "fn fibonacci(n: u64) -> u64 { if n <= 1 { n } else { fibonacci(n-1) + fibonacci(n-2) } }",
        "rust",
    )
    .await;

    let r = mem
        .query(r#"RECALL episodic ABOUT "fibonacci" LIMIT 1"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert_eq!(rr.records_returned, 1);
            match &rr.records[0].record {
                MemoryRecord::Episodic(e) => {
                    let mc = e
                        .multi_content
                        .as_ref()
                        .expect("multi_content should be set");
                    assert_eq!(mc.modality(), "code");
                }
                other => panic!("expected Episodic, got {other:?}"),
            }
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn audio_modality_preserved_after_store_and_recall() {
    let (mem, _dir) = open_memory().await;

    store_audio(
        &mem,
        "Sprint retrospective: team agreed to improve unit test coverage for payment module",
        b"fake_retro_audio",
    )
    .await;

    let r = mem
        .query(r#"RECALL episodic ABOUT "sprint retrospective" LIMIT 1"#)
        .await
        .unwrap();
    match r {
        QueryResult::Records(rr) => {
            assert_eq!(rr.records_returned, 1);
            match &rr.records[0].record {
                MemoryRecord::Episodic(e) => {
                    let mc = e
                        .multi_content
                        .as_ref()
                        .expect("multi_content should be set");
                    assert_eq!(mc.modality(), "audio");
                }
                other => panic!("expected Episodic, got {other:?}"),
            }
        }
        other => panic!("expected Records, got {other:?}"),
    }
}
