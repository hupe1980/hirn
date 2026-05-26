//! # HirnQL — The Cognitive Memory Query Language
//!
//! This example demonstrates HirnQL, hirn's declarative query language.
//! HirnQL is to hirn what SQL is to PostgreSQL — a domain-specific language
//! for cognitive memory operations.
//!
//! Demonstrates:
//! 1. Direct memory views — store memories through the supported write API
//! 2. RECALL — retrieve with HirnQL semantic search, filters, and activation
//! 3. GraphView connect — create graph edges via the direct API
//! 4. THINK — assemble LLM context under token budget
//! 5. INSPECT — view memory metadata
//! 6. TRACE — show provenance chain
//! 7. CONSOLIDATE — episodic → semantic consolidation
//! 8. EXPLAIN — view query execution plans
//! 9. Direct archive — mutate memories through the supported write API
//!
//! Run with: `cargo run --example hirnql -p hirn`

use hirn::prelude::*;
use hirn::ql::QueryResult;
use hirn_storage::{HirnDb, HirnDbConfig};

#[tokio::main]
async fn main() -> HirnResult<()> {
    Box::pin(run()).await
}

async fn run() -> HirnResult<()> {
    let (brain, _dir, path) = open_demo_db().await?;
    println!("✓ Opened database at {}", path.display());

    let agent = AgentId::new("engineer").expect("non-empty agent id");
    brain.register_agent(&agent, "Software Engineer").await?;

    let episode_ids = seed_episodic_memories(&brain, agent).await?;
    seed_semantic_memories(&brain, agent, &episode_ids).await?;

    Box::pin(run_recall_examples(&brain)).await?;

    let hnsw_deploy_id = episode_ids[0];
    let timeout_id = episode_ids[2];
    connect_graph_examples(&brain, hnsw_deploy_id, timeout_id).await?;
    run_think_example(&brain).await?;
    run_inspect_example(&brain, hnsw_deploy_id).await?;
    run_trace_example(&brain, hnsw_deploy_id).await?;
    run_explain_example(&brain)?;
    run_consolidate_example(&brain).await?;
    run_archive_example(&brain, timeout_id).await?;
    print_final_stats(&brain).await?;

    println!("\n✓ HirnQL demo complete!");
    Ok(())
}

async fn open_demo_db() -> HirnResult<(Hirn, tempfile::TempDir, std::path::PathBuf)> {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().join("hirnql");

    let config = HirnConfig::builder()
        .db_path(&path)
        .embedding_dimensions(64)
        .allow_pseudo_embedder_fallback(true)
        .build()?;
    let storage_config = HirnDbConfig::local(path.join("lance").to_str().unwrap());
    let storage = HirnDb::open(storage_config).await?.store_arc();
    let brain = Hirn::open_with_config(config, storage).await?;

    Ok((brain, dir, path))
}

async fn seed_episodic_memories(brain: &Hirn, agent: AgentId) -> HirnResult<Vec<MemoryId>> {
    println!("── Store Episodic Memories ──");

    let memories = [
        (
            "Deployed HNSW index to production, search latency dropped from 45ms to 2ms",
            EventType::Experiment,
            0.9,
        ),
        (
            "Product quantization reduced memory usage by 75% but recall dropped to 92%",
            EventType::Experiment,
            0.8,
        ),
        (
            "User reported intermittent timeouts on vector search endpoint under high load",
            EventType::Error,
            0.7,
        ),
        (
            "Added connection pooling, timeout issues resolved",
            EventType::Decision,
            0.75,
        ),
        (
            "A/B test: HNSW M=16 vs M=32 shows M=16 uses 40% less memory with only 1% recall difference",
            EventType::Experiment,
            0.85,
        ),
    ];

    let mut ids = Vec::with_capacity(memories.len());
    for (content, event_type, importance) in memories {
        let record = EpisodicRecord::builder()
            .content(content)
            .event_type(event_type)
            .agent_id(agent)
            .importance(importance)
            .entity("HNSW", "technology")
            .entity("vector_search", "domain")
            .build()?;

        let id = brain.episodic().remember(record).await?;
        ids.push(id);
        println!("  ✓ Stored episode: {id}");
    }

    Ok(ids)
}

async fn seed_semantic_memories(
    brain: &Hirn,
    agent: AgentId,
    episode_ids: &[MemoryId],
) -> HirnResult<Vec<MemoryId>> {
    println!("\n── Store Semantic Memories ──");

    let hnsw = SemanticRecord::builder()
        .concept("hnsw_indexing")
        .description(
            "HNSW achieves sub-linear ANN search via multi-layer navigable small world graphs",
        )
        .knowledge_type(KnowledgeType::Propositional)
        .confidence(0.95)
        .agent_id(agent)
        .source_episode(episode_ids[0])
        .build()?;

    let quantization = SemanticRecord::builder()
        .concept("product_quantization")
        .description("PQ and OPQ compress vectors by splitting into subvectors and quantizing each to a codebook")
        .knowledge_type(KnowledgeType::Propositional)
        .confidence(0.88)
        .agent_id(agent)
        .source_episode(episode_ids[1])
        .build()?;

    let ids = vec![
        brain.semantic().store(hnsw).await?,
        brain.semantic().store(quantization).await?,
    ];

    for id in &ids {
        println!("  ✓ Stored concept: {id}");
    }

    Ok(ids)
}

async fn run_recall_examples(brain: &Hirn) -> HirnResult<()> {
    println!("\n── RECALL (basic) ──");
    let result = brain
        .ql()
        .execute(r#"RECALL episodic ABOUT "search performance optimization" LIMIT 3"#)
        .await?;
    print_result(&result);

    println!("\n── RECALL (with importance filter) ──");
    let result = brain
        .ql()
        .execute(r#"RECALL episodic ABOUT "vector search" WHERE importance > 0.75 LIMIT 5"#)
        .await?;
    print_result(&result);

    println!("\n── RECALL (with spreading activation) ──");
    let result = brain.ql().execute(
        r#"RECALL episodic, semantic ABOUT "nearest neighbor search" EXPAND GRAPH DEPTH 2 ACTIVATION spreading LIMIT 5"#,
    ).await?;
    print_result(&result);

    Ok(())
}

async fn connect_graph_examples(
    brain: &Hirn,
    hnsw_deploy_id: MemoryId,
    timeout_id: MemoryId,
) -> HirnResult<()> {
    println!("\n── GraphView connect ──");
    let edge_id = brain
        .graph_view()
        .connect_with(
            hnsw_deploy_id,
            timeout_id,
            EdgeRelation::Causes,
            0.7,
            Default::default(),
        )
        .await?;
    println!("  ✓ Connected: {hnsw_deploy_id} --[causes]--> {timeout_id} ({edge_id})");

    Ok(())
}

async fn run_think_example(brain: &Hirn) -> HirnResult<()> {
    println!("\n── THINK ──");
    let result = brain
        .ql()
        .execute(r#"THINK ABOUT "How should I optimize vector search for production?" BUDGET 1024"#)
        .await?;
    match &result {
        QueryResult::Records(r) => {
            println!("  Records matched: {}", r.records_returned);
            println!("  Query time: {:.2}ms", r.query_time_ms);
            if let Some(ctx) = &r.context {
                println!(
                    "  Context preview:\n{}",
                    indent(&ctx.chars().take(300).collect::<String>(), 4)
                );
            }
        }
        _ => println!("  ? Unexpected result"),
    }

    Ok(())
}

async fn run_inspect_example(brain: &Hirn, id: MemoryId) -> HirnResult<()> {
    println!("\n── INSPECT ──");
    let result = brain.ql().execute(&format!("INSPECT \"{id}\"")).await?;
    match &result {
        QueryResult::Inspected(insp) => {
            println!("  Importance: {:.2}", insp.importance);
            println!("  Access count: {}", insp.access_count);
            println!("  Trust score: {:.2}", insp.trust_score);
            println!("  Neighbors: {}", insp.neighbors.len());
            for n in &insp.neighbors {
                println!(
                    "    {} --[{:?} w={:.2}]--> {}",
                    n.edge.source, n.edge.relation, n.edge.weight, n.edge.target
                );
            }
        }
        _ => println!("  ? Unexpected result"),
    }

    Ok(())
}

async fn run_trace_example(brain: &Hirn, id: MemoryId) -> HirnResult<()> {
    println!("\n── TRACE ──");
    let result = brain.ql().execute(&format!("TRACE \"{id}\"")).await?;
    match &result {
        QueryResult::Traced(t) => {
            println!("  Trust score: {:.2}", t.trust_score);
            println!("  Mutations: {}", t.mutation_count);
            println!("  Source episodes: {}", t.source_episodes.len());
            println!("  Lineage:\n{}", indent(&t.lineage_tree, 4));
        }
        _ => println!("  ? Unexpected result"),
    }

    Ok(())
}

fn run_explain_example(brain: &Hirn) -> HirnResult<()> {
    println!("\n── EXPLAIN ──");
    let plan = brain.ql().explain(
        r#"RECALL episodic, semantic ABOUT "vector optimization" EXPAND GRAPH DEPTH 2 ACTIVATION spreading WHERE importance > 0.5 LIMIT 10"#,
    )?;
    println!("{plan}");

    Ok(())
}

async fn run_consolidate_example(brain: &Hirn) -> HirnResult<()> {
    println!("\n── CONSOLIDATE ──");
    let result = brain.admin().consolidate().execute().await?;
    println!("  Records processed: {}", result.records_processed);

    Ok(())
}

async fn run_archive_example(brain: &Hirn, id: MemoryId) -> HirnResult<()> {
    println!("\n── Archive ──");
    brain.episodic().archive(id).await?;
    println!("  ✓ Archived: {id}");

    Ok(())
}

async fn print_final_stats(brain: &Hirn) -> HirnResult<()> {
    let stats = brain.admin().stats().await?;
    println!("\n── Final Stats ──");
    println!("  Total: {}", stats.total_count);
    println!(
        "  Episodic: {} | Semantic: {} | Working: {}",
        stats.episodic_count, stats.semantic_count, stats.working_count
    );

    Ok(())
}

fn print_result(result: &QueryResult) {
    match result {
        QueryResult::Records(r) => {
            println!(
                "  Found {} results (scanned {}, {:.2}ms):",
                r.records_returned, r.records_scanned, r.query_time_ms
            );
            for (i, scored) in r.records.iter().enumerate() {
                let summary = match &scored.record {
                    MemoryRecord::Episodic(ep) => ep.content.chars().take(65).collect::<String>(),
                    MemoryRecord::Semantic(s) => {
                        format!("[{}] {}", s.concept, s.description)
                    }
                    MemoryRecord::Working(w) => w.content.chars().take(65).collect::<String>(),
                    MemoryRecord::Procedural(p) => {
                        p.description.chars().take(65).collect::<String>()
                    }
                };
                println!("    #{}: score={:.3} | {}", i + 1, scored.score, summary);
            }
            if let Some(ctx) = &r.context {
                println!("  Context:\n{}", indent(ctx, 4));
            }
        }
        _ => println!("  (non-recall result)"),
    }
}

fn indent(s: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    s.lines()
        .map(|line| format!("{pad}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}
