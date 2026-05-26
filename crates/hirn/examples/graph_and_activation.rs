//! # Graph, Activation & Causal Reasoning
//!
//! This example demonstrates hirn's graph-native features:
//! 1. Build a property graph with typed edges
//! 2. Spreading activation vs static retrieval
//! 3. Hebbian co-retrieval learning (edges strengthen with use)
//! 4. Causal chain traversal
//! 5. Graph inspection and neighborhood exploration
//!
//! Run with: `cargo run --example graph_and_activation -p hirn`

use hirn::prelude::*;
use hirn_storage::{HirnDb, HirnDbConfig};

#[tokio::main]
async fn main() -> HirnResult<()> {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().join("graph");

    let config = HirnConfig::builder()
        .db_path(&path)
        .embedding_dimensions(64)
        .build()?;
    let storage_config = HirnDbConfig::local(path.join("lance").to_str().unwrap());
    let storage = HirnDb::open(storage_config).await?.store_arc();
    let brain = Hirn::open_with_config(config, storage).await?;

    let agent = AgentId::new("detective").expect("non-empty agent id");
    brain.register_agent(&agent, "Detective Agent").await?;

    // ── Build a causal chain of events ──────────────────────────────────
    // Simulate a production incident investigation
    let events = [
        (
            "Developer pushed config change removing cache TTL",
            EventType::Observation,
        ),
        (
            "Cache hit rate dropped from 95% to 12%",
            EventType::Observation,
        ),
        (
            "Database load increased by 8x due to cache misses",
            EventType::Observation,
        ),
        ("Database connection pool exhausted", EventType::Error),
        (
            "API response times spiked to 30 seconds",
            EventType::Observation,
        ),
        (
            "Production outage: 503 errors for 45 minutes",
            EventType::Error,
        ),
        (
            "Rollback: restored cache TTL, service recovered",
            EventType::Decision,
        ),
    ];

    let mut ids = Vec::new();
    for (content, event_type) in &events {
        let ep = EpisodicRecord::builder()
            .content(*content)
            .event_type(*event_type)
            .agent_id(agent.clone())
            .importance(0.8)
            .embedding(simple_embedding(content, 64))
            .entity("cache", "system")
            .entity("database", "system")
            .entity("production", "environment")
            .build()?;
        ids.push(brain.episodic().remember(ep).await?);
    }
    println!("✓ Stored {} events", ids.len());

    // ── 1. Build causal edges ───────────────────────────────────────────
    // config change → cache drop → db load → pool exhaustion → api spike → outage
    let causal_chain = [
        (0, 1, "Config change eliminated caching"),
        (1, 2, "Cache misses overwhelmed database"),
        (2, 3, "High load exhausted connection pool"),
        (3, 4, "No db connections caused slow responses"),
        (4, 5, "Sustained slowness triggered outage"),
        (5, 6, "Outage prompted emergency rollback"),
    ];

    for (src, tgt, reason) in &causal_chain {
        brain
            .graph_view()
            .connect_with(
                ids[*src],
                ids[*tgt],
                EdgeRelation::Causes,
                0.95,
                [(
                    "reason".to_string(),
                    hirn::metadata::MetadataValue::from(reason.to_string()),
                )]
                .into_iter()
                .collect(),
            )
            .await?;
    }
    println!("✓ Built causal chain ({} edges)", causal_chain.len());

    // Add a contradiction edge — two conflicting hypotheses
    let hypothesis_a = EpisodicRecord::builder()
        .content("Root cause: cache TTL was accidentally set to 0")
        .event_type(EventType::Observation)
        .agent_id(agent.clone())
        .importance(0.7)
        .embedding(simple_embedding("cache ttl zero root cause", 64))
        .build()?;
    let h_a = brain.episodic().remember(hypothesis_a).await?;

    let hypothesis_b = EpisodicRecord::builder()
        .content("Root cause: cache service crashed due to memory leak")
        .event_type(EventType::Observation)
        .agent_id(agent.clone())
        .importance(0.6)
        .embedding(simple_embedding("cache crash memory leak root cause", 64))
        .build()?;
    let h_b = brain.episodic().remember(hypothesis_b).await?;

    brain
        .graph_view()
        .connect_with(
            h_a,
            h_b,
            EdgeRelation::Contradicts,
            0.85,
            Default::default(),
        )
        .await?;
    println!("✓ Added contradicting hypotheses");

    // ── 2. Compare retrieval modes ──────────────────────────────────────
    let query = simple_embedding("production outage root cause", 64);

    // Static retrieval (no graph traversal)
    let static_results = brain
        .recall_view()
        .query(query.clone())
        .activation(ActivationMode::None)
        .limit(5)
        .execute()
        .await?;

    println!("\n── Static Retrieval (similarity only) ──");
    for (i, r) in static_results.iter().enumerate() {
        println!(
            "  #{}: score={:.3} | {}",
            i + 1,
            r.composite_score,
            record_summary(&r.record)
        );
    }

    // Spreading activation (graph-aware)
    let spreading_results = brain
        .recall_view()
        .query(query.clone())
        .activation(ActivationMode::Spreading)
        .depth(3)
        .limit(5)
        .execute()
        .await?;

    println!("\n── Spreading Activation (graph-aware) ──");
    for (i, r) in spreading_results.iter().enumerate() {
        println!(
            "  #{}: score={:.3} sim={:.3} act={:.3} | {}",
            i + 1,
            r.composite_score,
            r.similarity,
            r.score_breakdown.activation,
            record_summary(&r.record)
        );
    }

    // ── 3. Graph inspection ─────────────────────────────────────────────
    {
        let pg = brain.persistent_graph();
        println!("\n── Graph Stats ──");
        println!("  Nodes: {}", pg.node_count().await?);
        println!("  Edges: {}", pg.edge_count().await?);

        // Explore neighborhood of the outage event
        let outage_id = ids[5];
        let neighbors = pg.get_neighbors(outage_id, 2, 0.0).await?;
        println!(
            "\n  Outage event neighborhood (depth=2): {} connected memories",
            neighbors.len()
        );

        let outage_edges = pg.get_edges(outage_id).await?;
        println!("  Direct edges on outage event:");
        for edge in &outage_edges {
            println!(
                "    {} --[{:?} w={:.2}]--> {}",
                edge.source, edge.relation, edge.weight, edge.target
            );
        }

        // Shortest path from config change to outage
        if let Some(path) = pg.shortest_path(ids[0], ids[5]).await? {
            println!(
                "\n  Shortest path (config change → outage): {} hops",
                path.len() - 1
            );
            for (i, id) in path.iter().enumerate() {
                let mem = brain.admin().get_memory(*id).await?;
                println!("    {i}. {}", record_summary(&mem));
            }
        }
    }

    // ── 4. Think with causal context ────────────────────────────────────
    println!("\n── Assembling context via think()... ──");
    let context = brain
        .recall_view()
        .think(query)
        .budget(2048)
        .execute()
        .await?;

    println!("\n── Think Result (with graph context) ──");
    println!("  Tokens: {}", context.token_count);
    println!("  Records included: {}", context.records_included.len());
    println!("  Contradictions found: {}", context.contradictions.len());
    if !context.contradictions.is_empty() {
        println!("  ⚠ Conflict detected — both sides included for LLM reasoning");
    }

    // ── 5. Trace provenance ─────────────────────────────────────────────
    let trace = brain.recall_view().trace(ids[5]).execute().await?;
    println!("\n── Provenance Trace (outage event) ──");
    println!("  Trust score: {:.2}", trace.trust_score);
    println!("  Mutations:   {}", trace.mutation_count);
    println!("  Lineage:\n{}", indent(&trace.lineage_tree, 4));

    println!("\n✓ Graph & activation demo complete!");
    Ok(())
}

fn record_summary(rec: &MemoryRecord) -> String {
    match rec {
        MemoryRecord::Episodic(ep) => {
            let s: String = ep.content.chars().take(65).collect();
            format!("[ep] {s}")
        }
        MemoryRecord::Semantic(sem) => format!("[sem] {}", sem.concept),
        MemoryRecord::Working(wm) => {
            let s: String = wm.content.chars().take(65).collect();
            format!("[wm] {s}")
        }
        MemoryRecord::Procedural(p) => {
            let s: String = p.description.chars().take(65).collect();
            format!("[proc] {s}")
        }
    }
}

fn simple_embedding(text: &str, dims: usize) -> Vec<f32> {
    let mut emb = vec![0.0f32; dims];
    for (i, byte) in text.bytes().enumerate() {
        emb[i % dims] += f32::from(byte) / 255.0;
    }
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut emb {
            *x /= norm;
        }
    }
    emb
}

fn indent(s: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    s.lines()
        .map(|line| format!("{pad}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}
