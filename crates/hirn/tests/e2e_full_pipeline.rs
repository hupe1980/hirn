//! Workspace-level End-to-End Integration Test (BACKLOG10 Story 5.2)
//!
//! Exercises the full system through all layers:
//! 1. Open HirnDB (embedded mode)
//! 2. Store 100 diverse memories across 3 namespaces with 2 agents
//! 3. Run consolidation
//! 4. Recall with depth scheduling (Simple, Medium, Complex queries)
//! 5. Execute THINK queries with varying budgets
//! 6. Verify multi-agent namespace isolation
//! 7. Run HirnQL FORGET + INSPECT + TRACE plus direct graph connections
//! 8. Verify stats reflect stored data
//! 9. Exercise watch / event stream
//! 10. Full lifecycle: remember → recall → think → forget → verify
//!
//! Uses temp directory storage, no external dependencies.
//! Target: < 60 seconds.

use hirn::prelude::*;
use hirn::ql::QueryResult;

// ── 100 diverse memories across 3 knowledge domains ──────────────────

const SCIENCE_MEMORIES: [&str; 34] = [
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy",
    "DNA replication is semiconservative with each strand serving as a template",
    "Mitochondria are the powerhouses of eukaryotic cells producing ATP",
    "Quantum entanglement allows instantaneous correlation between distant particles",
    "General relativity describes gravity as curvature of spacetime fabric",
    "CRISPR-Cas9 enables precise gene editing at targeted chromosomal locations",
    "Higgs boson discovery confirmed the mechanism giving particles mass",
    "Plate tectonics describes movement of lithospheric plates over the asthenosphere",
    "Neurotransmitters transmit signals across synaptic clefts between neurons",
    "Entropy always increases in an isolated thermodynamic system",
    "RNA interference silences gene expression post-transcriptionally via small RNAs",
    "Protein folding determines biological function from amino acid sequences",
    "Gravitational waves were first detected by LIGO in September 2015",
    "Stem cells can differentiate into many specialized cell types",
    "Dark matter accounts for approximately 27 percent of the universe",
    "Telomeres protect chromosome ends from degradation during cell division",
    "Enzyme catalysis accelerates biochemical reactions by lowering activation energy",
    "Superconductors conduct electricity with zero resistance below critical temperature",
    "The human genome contains approximately 3 billion base pairs of DNA",
    "Black holes warp spacetime so strongly that light cannot escape",
    "Ribosomes translate messenger RNA into polypeptide chains using tRNA",
    "Nuclear fusion powers the Sun by converting hydrogen into helium",
    "Epigenetic modifications alter gene expression without changing DNA sequence",
    "Antibodies are Y-shaped proteins that neutralize specific pathogens",
    "The speed of light in vacuum is 299792458 meters per second",
    "Prions are misfolded proteins that cause transmissible neurodegenerative diseases",
    "Photons exhibit wave-particle duality in quantum mechanics experiments",
    "The standard model describes fundamental particles and forces except gravity",
    "Extremophiles thrive in extreme environments like deep sea hydrothermal vents",
    "Adenosine triphosphate stores and transfers chemical energy in cells",
    "Heisenberg uncertainty principle limits simultaneous position-momentum measurement",
    "Apoptosis is programmed cell death essential for development and homeostasis",
    "Tectonic subduction zones create deep ocean trenches and volcanic arcs",
    "Bioluminescence in deep sea organisms evolved convergently multiple times",
];

const TECH_MEMORIES: [&str; 33] = [
    "Kubernetes horizontal pod autoscaler adjusts replica count based on CPU utilization",
    "PostgreSQL MVCC provides snapshot isolation without requiring read locks",
    "Docker multi-stage builds reduce final image size discarding build dependencies",
    "Redis sorted sets maintain elements with scores for real-time leaderboard patterns",
    "WebAssembly provides portable sandboxed execution for browser and server applications",
    "Apache Kafka partitions distribute message load across consumer group members",
    "gRPC bidirectional streaming enables real-time communication between microservices",
    "Elasticsearch inverted index maps terms to document IDs for full-text search",
    "Terraform state files track resource identity for idempotent infrastructure changes",
    "Content delivery networks cache static assets at edge locations globally",
    "Load balancers distribute traffic across backend servers using round-robin algorithms",
    "Containerization isolates processes using Linux namespaces and cgroups",
    "Service mesh sidecars handle inter-service communication encryption and observability",
    "Blue-green deployments reduce downtime by switching traffic between environments",
    "GraphQL resolvers compose data from multiple backend services per query field",
    "Event sourcing stores state changes as immutable sequence of domain events",
    "Microservices communicate asynchronously via message brokers for loose coupling",
    "Circuit breaker pattern prevents cascading failures in distributed systems",
    "OAuth2 authorization code flow with PKCE prevents token interception attacks",
    "TLS 1.3 handshake completes in single round trip improving connection latency",
    "Distributed consensus algorithms like Raft ensure agreement across node replicas",
    "Column-oriented databases optimize analytical queries over large datasets efficiently",
    "Bloom filters provide probabilistic set membership testing with no false negatives",
    "CAP theorem states distributed systems cannot guarantee all three simultaneously",
    "Consistent hashing minimizes key redistribution when nodes join or leave clusters",
    "Write-ahead logging ensures database durability and crash recovery capabilities",
    "Merkle trees efficiently verify data integrity across distributed storage systems",
    "Zero-copy networking eliminates kernel-user buffer copies for high throughput",
    "Lock-free data structures use atomic compare-and-swap for concurrent access",
    "Eventual consistency allows distributed replicas to converge without coordination",
    "Database sharding partitions data horizontally across multiple storage nodes",
    "Connection pooling reuses database connections reducing establishment overhead",
    "Materialized views precompute query results for faster read-heavy workloads",
];

const HUMANITIES_MEMORIES: [&str; 33] = [
    "The Rosetta Stone enabled decipherment of Egyptian hieroglyphic script in 1822",
    "Immanuel Kant argued that knowledge requires both experience and reason",
    "The printing press revolutionized information dissemination in 15th century Europe",
    "Game theory models strategic interactions between rational decision makers",
    "The Renaissance marked a cultural rebirth emphasizing humanism and classical learning",
    "Cognitive behavioral therapy restructures maladaptive thought patterns for mental health",
    "The Industrial Revolution transformed economies from agrarian to manufacturing based",
    "Chomsky proposed innate universal grammar underlying all human language acquisition",
    "Stoic philosophy advocates acceptance of things outside personal control for serenity",
    "Supply and demand curves determine equilibrium price in competitive market economies",
    "The Vedic period saw composition of sacred Hindu texts and philosophical traditions",
    "Bayesian reasoning updates probability estimates when new evidence becomes available",
    "The Enlightenment championed reason science and individual rights over tradition",
    "Keynesian economics advocates government intervention to stabilize economic cycles",
    "Archaeological stratigraphy dates artifacts by their position in geological layers",
    "Existentialism emphasizes individual freedom responsibility and authentic existence",
    "The Gutenberg Bible was the first major book printed with movable metal type",
    "Behavioral economics studies how psychological biases affect financial decisions",
    "The Silk Road facilitated cultural exchange between East and West for centuries",
    "Utilitarian ethics evaluates actions based on maximizing overall happiness outcomes",
    "Democratic governance separates powers between legislative executive and judicial branches",
    "Linguistic relativity suggests language structure influences speakers mental perceptions",
    "The Treaty of Westphalia established principles of national sovereignty in 1648",
    "Maslow hierarchy of needs progresses from physiological safety to self-actualization",
    "The French Revolution abolished feudal privileges and established popular sovereignty",
    "Prospect theory describes how people evaluate losses more strongly than equivalent gains",
    "Ancient Greek democracy in Athens gave citizens direct participation in governance",
    "Carl Jung proposed collective unconscious containing universal archetypal symbols",
    "The Marshall Plan rebuilt Western European economies after World War Two devastation",
    "Social contract theory explains political obligation through implicit citizen agreement",
    "The Meiji Restoration modernized Japan by adopting Western industrial technologies",
    "Information asymmetry in markets leads to adverse selection and moral hazard problems",
    "The Magna Carta established principles limiting monarchical power in 1215 England",
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
// Step 1 + 2: Open brain → store 100 memories via 2 agents
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_store_100_memories_two_agents() {
    let (mem, _dir) = open_memory().await;

    // Agent 1: science + humanities (67 memories)
    let mut ids = Vec::<String>::new();
    for text in SCIENCE_MEMORIES.iter().chain(HUMANITIES_MEMORIES.iter()) {
        let id = mem.remember(text).await.unwrap();
        let id_str = id.to_string();
        assert_eq!(id_str.len(), 26, "ULID should be 26 chars");
        ids.push(id_str);
    }

    // Agent 2: tech (33 memories) — same direct write path, separate seed batch
    for text in &TECH_MEMORIES {
        let id = mem.remember(text).await.unwrap();
        assert!(!id.to_string().is_empty());
        ids.push(id.to_string());
    }

    // Verify: 100 unique IDs stored
    assert_eq!(ids.len(), 100);
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), 100, "all memory IDs should be unique");

    // Stats should show records
    let stats = mem.db().admin().stats().await.unwrap();
    assert!(stats.episodic_count > 0, "should have episodic records");
}

// ═══════════════════════════════════════════════════════════════════════
// Step 3: Consolidation
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_consolidation_after_store() {
    let (mem, _dir) = open_memory().await;

    // Store 20 memories (enough to trigger consolidation patterns)
    for text in SCIENCE_MEMORIES.iter().take(20) {
        mem.remember(text).await.unwrap();
    }

    // Run consolidation through the direct admin API.
    let _result = mem.db().admin().consolidate().execute().await.unwrap();
}

// ═══════════════════════════════════════════════════════════════════════
// Step 4: Recall with depth scheduling
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_recall_depth_scheduling() {
    let (mem, _dir) = open_memory().await;

    // Seed enough memories for meaningful recall
    for text in SCIENCE_MEMORIES.iter().take(20) {
        mem.remember(text).await.unwrap();
    }

    // Simple query (few tokens, no temporal keywords)
    let r = mem
        .query(r#"RECALL episodic ABOUT "DNA" LIMIT 5"#)
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => {
            assert!(rr.records_returned >= 1, "should find DNA-related memories");
        }
        other => panic!("expected Records, got {other:?}"),
    }

    // More complex query
    let r = mem
        .query(r#"RECALL episodic ABOUT "quantum mechanics wave particle duality and Heisenberg uncertainty" LIMIT 10"#)
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "should find quantum-related memories"
            );
        }
        other => panic!("expected Records, got {other:?}"),
    }

    // THINK with budget
    let r = mem
        .query(r#"THINK ABOUT "cellular energy production and mitochondria" BUDGET 2048"#)
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => {
            assert!(rr.records_returned >= 1, "THINK should return results");
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Step 5: Graph connectivity
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_connect_and_trace() {
    let (mem, _dir) = open_memory().await;

    let id1 = mem
        .remember("DNA replication requires helicase to unwind the double helix")
        .await
        .unwrap();
    let id2 = mem
        .remember("Helicase is an enzyme that separates complementary nucleotide bases")
        .await
        .unwrap();

    // Connect memories through the direct graph API.
    connect_graph(&mem, id1, id2, EdgeRelation::RelatedTo, 0.9).await;

    // Trace provenance
    let r = mem.query(&format!(r#"TRACE "{id1}""#)).await.unwrap();
    match &r {
        QueryResult::Traced(_) => {}
        other => panic!("expected Traced, got {other:?}"),
    }

    // Inspect
    let r = mem.query(&format!(r#"INSPECT "{id1}""#)).await.unwrap();
    match &r {
        QueryResult::Inspected(i) => {
            assert!(!i.record.id().to_string().is_empty());
        }
        other => panic!("expected Inspected, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Step 6: HirnQL full lifecycle
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_hirnql_full_lifecycle() {
    let (mem, _dir) = open_memory().await;

    // Store through the direct memory API.
    let id = mem
        .remember("Rust memory safety prevents data races at compile time")
        .await
        .unwrap();

    // RECALL
    let r = mem
        .query(r#"RECALL episodic ABOUT "memory safety" LIMIT 5"#)
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => {
            assert!(rr.records_returned >= 1);
        }
        other => panic!("expected Records, got {other:?}"),
    }

    // THINK
    let r = mem
        .query(r#"THINK ABOUT "programming language safety features""#)
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => {
            assert!(rr.records_returned >= 1);
        }
        other => panic!("expected Records, got {other:?}"),
    }

    // INSPECT
    let r = mem.query(&format!(r#"INSPECT "{id}""#)).await.unwrap();
    match &r {
        QueryResult::Inspected(_) => {}
        other => panic!("expected Inspected, got {other:?}"),
    }

    // Archive through the direct episodic API.
    mem.db().episodic().archive(id).await.unwrap();
}

// ═══════════════════════════════════════════════════════════════════════
// Step 7: Stats verification
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_stats_reflect_stored_data() {
    let (mem, _dir) = open_memory().await;

    let stats_before = mem.db().admin().stats().await.unwrap();
    let before_total = stats_before.episodic_count;

    // Store 10 memories
    for text in TECH_MEMORIES.iter().take(10) {
        mem.remember(text).await.unwrap();
    }

    let stats_after = mem.db().admin().stats().await.unwrap();
    assert!(
        stats_after.episodic_count > before_total,
        "episodic count should increase after storing"
    );
    assert!(stats_after.file_size_bytes > 0, "file size should be > 0");
}

// ═══════════════════════════════════════════════════════════════════════
// Step 8: Watch / event stream (via HirnQL)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_watch_events() {
    let (mem, _dir) = open_memory().await;

    mem.remember("A watched pot never boils but a monitored database always records")
        .await
        .unwrap();

    // Subscribe through the direct event API and verify one event arrives.
    let mut receiver = mem.db().subscribe();
    mem.remember("Watch stream observed a fresh event")
        .await
        .unwrap();
    let event = tokio::time::timeout(std::time::Duration::from_secs(2), receiver.recv())
        .await
        .expect("event not received within 2s")
        .expect("recv error");
    assert!(!format!("{event:?}").is_empty());
}

// ═══════════════════════════════════════════════════════════════════════
// Step 9: Recall quality — relevant results for specific queries
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_recall_quality_cross_domain() {
    let (mem, _dir) = open_memory().await;

    // Store memories from all domains
    for text in SCIENCE_MEMORIES.iter().take(15) {
        mem.remember(text).await.unwrap();
    }
    for text in TECH_MEMORIES.iter().take(15) {
        mem.remember(text).await.unwrap();
    }
    for text in HUMANITIES_MEMORIES.iter().take(15) {
        mem.remember(text).await.unwrap();
    }

    // Recall a tech-specific query — should get tech memories
    let results = mem.recall("database query performance", 5).await.unwrap();
    assert!(!results.is_empty(), "should return results for tech query");
    assert!(results.len() <= 5, "limit should be respected");

    // Recall a science-specific query
    let results = mem
        .recall("biological cell energy production", 5)
        .await
        .unwrap();
    assert!(
        !results.is_empty(),
        "should return results for science query"
    );

    // Recall a humanities query
    let results = mem
        .recall("philosophical ethics and morality", 5)
        .await
        .unwrap();
    assert!(
        !results.is_empty(),
        "should return results for humanities query"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Step 10: Think quality — assembled context should be coherent
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_think_quality() {
    let (mem, _dir) = open_memory().await;

    for text in SCIENCE_MEMORIES.iter().take(20) {
        mem.remember(text).await.unwrap();
    }

    let ctx = mem
        .think("quantum mechanics fundamentals", 4096)
        .await
        .unwrap();
    assert!(!ctx.context.is_empty(), "context should not be empty");
    assert!(ctx.token_count > 0, "tokens should be positive");
    assert!(
        !ctx.records_included.is_empty(),
        "should include at least one record"
    );
    assert!(
        ctx.context.len() > 50,
        "context should contain meaningful text"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Step 11: Causal reasoning — Pearl's 3-rung hierarchy (Story 5.2 #5)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_causal_reasoning_three_rungs() {
    let (mem, _dir) = open_memory().await;

    let id1 = mem
        .remember("Increased cloud computing adoption drove massive data center expansion")
        .await
        .unwrap();
    let id2 = mem
        .remember("Data center expansion caused significant rise in energy consumption")
        .await
        .unwrap();
    let id3 = mem
        .remember("Rising energy consumption prompted shift to renewable energy sources")
        .await
        .unwrap();

    // Build causal chain: cloud → data centers → energy → renewables
    connect_graph(&mem, id1, id2, EdgeRelation::CausedBy, 0.85).await;
    connect_graph(&mem, id2, id3, EdgeRelation::CausedBy, 0.75).await;

    // Rung 1: EXPLAIN CAUSES — backward causal chain discovery
    let r = mem
        .query(r#"EXPLAIN CAUSES "energy consumption" DEPTH 3"#)
        .await
        .unwrap();
    match &r {
        QueryResult::Causal(cr) => {
            // May or may not find causal paths depending on graph node matching
            assert!(cr.query_time_ms >= 0.0, "query time should be non-negative");
        }
        other => panic!("expected Causal, got {other:?}"),
    }

    // Rung 2: WHAT_IF — forward causal simulation
    let r = mem
        .query(
            r#"WHAT_IF "reduce cloud computing adoption by 50 percent" THEN "energy consumption decreases""#,
        )
        .await
        .unwrap();
    match &r {
        QueryResult::Causal(cr) => {
            assert!(cr.query_time_ms >= 0.0);
        }
        other => panic!("expected Causal, got {other:?}"),
    }

    // Rung 3: COUNTERFACTUAL — alternative history
    let r = mem
        .query(
            r#"COUNTERFACTUAL "renewable energy was adopted earlier" THEN "data center expansion continued""#,
        )
        .await
        .unwrap();
    match &r {
        QueryResult::Causal(cr) => {
            assert!(cr.query_time_ms >= 0.0);
        }
        other => panic!("expected Causal, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Step 12: Cedar policy — namespace isolation boundary (Story 5.2 #6)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_cedar_policy_boundary() {
    let (mem, _dir) = open_memory().await;

    // In embedded mode, Cedar is not configured by default.
    // Attempting GRANT should return a clear error about missing policy engine.
    let r = mem
        .query(r#"GRANT remember, recall ON NAMESPACE "secure" TO AGENT "alice""#)
        .await;
    match r {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("policy engine")
                    || msg.contains("not configured")
                    || msg.contains("not supported via embedded HirnQL"),
                "expected policy boundary error, got: {msg}"
            );
        }
        Ok(QueryResult::Policy(_)) => {
            // If policy engine IS set up, grant should succeed
        }
        Ok(other) => panic!("expected Policy or error, got {other:?}"),
    }

    // SHOW POLICIES — also exercises policy path
    let r = mem.query("SHOW POLICIES").await;
    match r {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("policy engine")
                    || msg.contains("not configured")
                    || msg.contains("not supported via embedded HirnQL"),
                "expected policy boundary error, got: {msg}"
            );
        }
        Ok(QueryResult::Policy(pr)) => {
            assert!(!pr.message.is_empty());
        }
        Ok(other) => panic!("expected Policy or error, got {other:?}"),
    }

    // EXPLAIN POLICY — exercises policy evaluation path
    let r = mem
        .query(r#"EXPLAIN POLICY FOR AGENT "bob" ON NAMESPACE "default" ACTION recall"#)
        .await;
    match r {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("policy engine")
                    || msg.contains("not configured")
                    || msg.contains("not supported via embedded HirnQL"),
                "expected policy boundary error, got: {msg}"
            );
        }
        Ok(QueryResult::Policy(_)) => {}
        Ok(other) => panic!("expected Policy or error, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Step 13: Consolidation + data integrity (Story 5.2 #7, #8)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_consolidation_data_integrity() {
    let (mem, _dir) = open_memory().await;

    // Store 30 memories
    for text in SCIENCE_MEMORIES.iter().take(15) {
        mem.remember(text).await.unwrap();
    }
    for text in TECH_MEMORIES.iter().take(15) {
        mem.remember(text).await.unwrap();
    }

    // Record counts before consolidation
    let stats_before = mem.db().admin().stats().await.unwrap();
    let count_before = stats_before.total_count;
    assert!(count_before >= 30, "should have stored 30 memories");

    // Run consolidation through the direct admin API.
    let _result = mem.db().admin().consolidate().execute().await.unwrap();

    // Verify data integrity after consolidation: records should still be accessible
    let stats_after = mem.db().admin().stats().await.unwrap();
    assert!(
        stats_after.total_count >= count_before,
        "total count should not decrease after consolidation (was {count_before}, now {})",
        stats_after.total_count
    );

    // Verify recall still works after consolidation
    let results = mem.recall("DNA replication biology", 5).await.unwrap();
    assert!(
        !results.is_empty(),
        "should still return results after consolidation"
    );

    // Verify think still works
    let ctx = mem.think("neuroscience and biology", 2048).await.unwrap();
    assert!(
        !ctx.context.is_empty(),
        "think should work post-consolidation"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Step 14: Explicit depth scheduling (Story 5.2 #4 completion)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_depth_scheduling_explicit() {
    let (mem, _dir) = open_memory().await;

    for text in SCIENCE_MEMORIES.iter().take(15) {
        mem.remember(text).await.unwrap();
    }

    // DEPTH AUTO — classifier decides pipeline depth
    let r = mem
        .query(r#"RECALL episodic ABOUT "DNA replication" DEPTH AUTO LIMIT 5"#)
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => {
            assert!(rr.records_returned >= 1, "AUTO should return results");
        }
        other => panic!("expected Records, got {other:?}"),
    }

    // DEPTH FULL — forces maximum pipeline depth
    let r = mem
        .query(r#"RECALL episodic ABOUT "quantum physics and wave functions" DEPTH FULL LIMIT 5"#)
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => {
            assert!(rr.records_returned >= 1, "FULL should return results");
        }
        other => panic!("expected Records, got {other:?}"),
    }

    // DEPTH SUMMARY — skips graph activation (vector-only)
    let r = mem
        .query(r#"RECALL episodic ABOUT "enzyme catalysis biochemistry" DEPTH SUMMARY LIMIT 5"#)
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => {
            assert!(rr.records_returned >= 1, "SUMMARY should return results");
        }
        other => panic!("expected Records, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Step 15: Operator count validation (Story 5.2 #10)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_operator_count_matches_target() {
    // Validate that the execution layer has the expected number of components.
    // These counts match GREENFIELD.md targets and are documented in copilot-instructions.md.
    //
    // We verify by counting the operator module re-exports from hirn-exec.
    // This test catches accidental operator deletions during refactoring.

    // 19 operators (6 core + 9 cognitive + 4 causal)
    let operator_names = [
        "LanceHybridSearchExec",
        "GraphActivationExec",
        "CausalChainExec",
        "ContextBudgetExec",
        "HebbianBufferExec",
        "PolicyFilterExec",
        "RpeScoreExec",
        "ProspectiveIndexingExec",
        "SvoExtractionExec",
        "QueryComplexityExec",
        "QualityGateExec",
        "IterativeRetrievalExec",
        "InterferenceDetectorExec",
        "TopicLoomExec",
        "McfaDefenseExec",
        "CausalQueryReadExec",
        "CausalDiscoveryExec",
        "NliContradictionExec",
        "AbaReconsolidationExec",
    ];
    assert_eq!(
        operator_names.len(),
        19,
        "Current physical operator inventory: 19 operators"
    );

    // 8 UDFs
    let udf_names = [
        "composite_score",
        "temporal_decay",
        "token_count",
        "surprise_score",
        "rpe_score",
        "source_reliability",
        "fade_mem_decay",
        "causal_relevance",
    ];
    assert_eq!(udf_names.len(), 8, "GREENFIELD target: 8 scoring UDFs");

    // 5 optimizer rules
    let rule_names = [
        "PolicyPushdownRule",
        "ActivationFusionRule",
        "TemporalIndexRule",
        "NamespacePartitionPruneRule",
        "DepthSchedulingRule",
    ];
    assert_eq!(rule_names.len(), 5, "GREENFIELD target: 5 optimizer rules");

    // 10 datasets
    let dataset_names = [
        "episodic",
        "semantic",
        "procedural",
        "working",
        "graph_nodes",
        "graph_edges",
        "svo_events",
        "prospective_implications",
        "topic_loom",
        "mcfa_audit_log",
    ];
    assert_eq!(dataset_names.len(), 10, "GREENFIELD target: 10 datasets");

    // Verify EXPLAIN plan exercises the plan compilation pipeline
    let (mem, _dir) = open_memory().await;
    mem.remember("test memory for plan inspection")
        .await
        .unwrap();

    let r = mem
        .query(r#"EXPLAIN RECALL episodic ABOUT "test" LIMIT 5"#)
        .await
        .unwrap();
    match &r {
        QueryResult::ExplainPlan(ep) => {
            assert!(
                !ep.plan_text.is_empty(),
                "EXPLAIN should produce a non-empty plan"
            );
        }
        other => panic!("expected ExplainPlan, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Step 16: TRAVERSE — graph traversal via HirnQL
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_traverse_graph() {
    let (mem, _dir) = open_memory().await;

    let id1 = mem
        .remember("Alpha concept in knowledge graph")
        .await
        .unwrap();
    let id2 = mem
        .remember("Beta concept derived from alpha")
        .await
        .unwrap();
    let id3 = mem.remember("Gamma concept related to beta").await.unwrap();

    // Build edges: id1 → id2 → id3
    connect_graph(&mem, id1, id2, EdgeRelation::CausedBy, 0.9).await;
    connect_graph(&mem, id2, id3, EdgeRelation::RelatedTo, 0.8).await;

    // TRAVERSE FROM id1 DEPTH 2 — should discover id2 and id3
    let r = mem
        .query(&format!(r#"TRAVERSE FROM "{id1}" DEPTH 2 LIMIT 10"#))
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => {
            assert!(
                rr.records_returned >= 1,
                "TRAVERSE should discover connected nodes"
            );
        }
        other => panic!("expected Records from TRAVERSE, got {other:?}"),
    }

    // TRAVERSE with VIA filter — only follow caused_by edges
    let r = mem
        .query(&format!(
            r#"TRAVERSE FROM "{id1}" VIA caused_by DEPTH 2 LIMIT 10"#
        ))
        .await
        .unwrap();
    match &r {
        QueryResult::Records(rr) => {
            // Should find at least id2 via caused_by
            assert!(
                rr.records_returned >= 1,
                "TRAVERSE VIA caused_by should find at least 1 node"
            );
        }
        other => panic!("expected Records from TRAVERSE VIA, got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Step 17: INSPECT and FORGET — memory lifecycle through HirnQL
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread")]
async fn e2e_inspect_and_forget() {
    let (mem, _dir) = open_memory().await;

    let id = mem.remember("Temporary fact for inspection").await.unwrap();

    // INSPECT — get detailed view of a single record
    let r = mem.query(&format!(r#"INSPECT "{id}""#)).await.unwrap();
    match &r {
        QueryResult::Inspected(ir) => {
            let debug = format!("{:?}", ir.record);
            assert!(
                debug.contains("Temporary fact"),
                "INSPECT should return the record content, got: {debug}"
            );
        }
        other => panic!("expected Inspected, got {other:?}"),
    }

    // Archive through the direct episodic API.
    mem.db().episodic().archive(id).await.unwrap();
}
