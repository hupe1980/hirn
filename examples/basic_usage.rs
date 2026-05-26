//! # Basic Usage — Getting Started with hirn
//!
//! This example demonstrates the core workflow:
//! 1. Open a database
//! 2. Register an agent
//! 3. Store episodic memories (experiences)
//! 4. Store semantic memories (knowledge)
//! 5. Use working memory (scratchpad)
//! 6. Recall memories by similarity
//! 7. Assemble LLM context with `think()`
//!
//! Run with: `cargo run --example basic_usage -p hirn`

use hirn::prelude::*;
use hirn_storage::{HirnDb, HirnDbConfig};

#[tokio::main]
async fn main() -> HirnResult<()> {
    // ── 1. Open a database ──────────────────────────────────────────────
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().join("brain");

    let config = HirnConfig::builder()
        .db_path(&path)
        .embedding_dimensions(64) // small dims for this example
        .build()?;

    let storage_config = HirnDbConfig::local(path.join("lance").to_str().unwrap());
    let storage = HirnDb::open(storage_config).await?.store_arc();
    let brain = Hirn::open_with_config(config, storage).await?;
    println!("✓ Opened database at {}", path.display());

    // ── 2. Register an agent ────────────────────────────────────────────
    let agent = AgentId::new("researcher").expect("non-empty agent id");
    brain.register_agent(&agent, "Research Assistant").await?;
    println!("✓ Registered agent: {agent}");

    // ── 3. Store episodic memories ──────────────────────────────────────
    // Episodic memory = "what happened" — time-anchored experiences
    let experiments = [
        (
            "HNSW index with M=16 achieved 95% recall@10 on 1M vectors in 2ms",
            EventType::Experiment,
            0.85,
        ),
        (
            "Product quantization reduced memory usage by 4x but recall dropped to 88%",
            EventType::Experiment,
            0.75,
        ),
        (
            "User reported slow search times with brute-force on >100k vectors",
            EventType::Observation,
            0.60,
        ),
        (
            "Decision: switch from brute-force to HNSW for production deployment",
            EventType::Decision,
            0.90,
        ),
        (
            "Error: OOM when building HNSW index with M=64 on 10M vectors",
            EventType::Error,
            0.70,
        ),
    ];

    let mut episode_ids = Vec::new();
    for (content, event_type, importance) in &experiments {
        // Generate a simple deterministic embedding for this example
        let embedding = simple_embedding(content, 64);

        let episode = EpisodicRecord::builder()
            .content(*content)
            .event_type(*event_type)
            .agent_id(agent.clone())
            .importance(*importance)
            .embedding(embedding)
            .entity("HNSW", "technology")
            .entity("vector_search", "domain")
            .build()?;

        let id = brain.episodic().remember(episode).await?;
        episode_ids.push(id);
        println!("  → Stored episode: {content:.60}...");
    }
    println!("✓ Stored {} episodic memories", episode_ids.len());

    // ── 4. Store semantic memories ──────────────────────────────────────
    // Semantic memory = "what I know" — consolidated knowledge
    let hnsw_knowledge = hirn::semantic::SemanticRecord::builder()
        .concept("hnsw_indexing")
        .description(
            "HNSW (Hierarchical Navigable Small World) is a graph-based \
             approximate nearest neighbor algorithm. It achieves sub-linear \
             search time with high recall by building a multi-layer navigation \
             graph. Key parameters: M (max connections per node), ef \
             (search beam width).",
        )
        .knowledge_type(KnowledgeType::Propositional)
        .confidence(0.92)
        .agent_id(agent.clone())
        .source_episode(episode_ids[0])
        .build()?;

    let sem_id = brain.semantic().store(hnsw_knowledge).await?;
    println!("✓ Stored semantic knowledge: hnsw_indexing (id={sem_id})");

    // ── 5. Working memory ───────────────────────────────────────────────
    // Working memory = "what I'm thinking about right now"
    let focus = WorkingMemoryEntry::builder()
        .content("Current task: optimize vector search for production deployment")
        .priority(Priority::High)
        .agent_id(agent.clone())
        .build()?;

    let wm_id = brain.working().focus(focus).await?;
    println!("✓ Focused working memory: {wm_id}");

    let wm = brain.working().entries().await?;
    println!("  Working memory entries: {}", wm.len());

    // ── 6. Recall memories by similarity ────────────────────────────────
    let query = simple_embedding("vector search performance optimization", 64);

    let results = brain
        .recall_view()
        .query(query.clone())
        .activation(ActivationMode::Spreading)
        .limit(5)
        .execute()
        .await?;

    println!("\n── Recall Results (spreading activation) ──");
    for (i, r) in results.iter().enumerate() {
        let summary = match &r.record {
            MemoryRecord::Episodic(ep) => ep.content.chars().take(60).collect::<String>(),
            MemoryRecord::Semantic(sem) => format!(
                "[{}] {}",
                sem.concept,
                sem.description.chars().take(40).collect::<String>()
            ),
            MemoryRecord::Working(wm) => wm.content.chars().take(60).collect::<String>(),
            MemoryRecord::Procedural(p) => p.description.chars().take(60).collect::<String>(),
        };
        println!(
            "  #{}: score={:.3} sim={:.3} act={:.3} | {}",
            i + 1,
            r.composite_score,
            r.similarity,
            r.score_breakdown.activation,
            summary
        );
    }

    // ── 7. Think — assemble LLM context ─────────────────────────────────
    let context = brain
        .recall_view()
        .think(query)
        .budget(2048)
        .activation(ActivationMode::Spreading)
        .execute()
        .await?;

    println!("\n── Think Result (token-budget-aware context assembly) ──");
    println!("  Tokens used:      {}", context.token_count);
    println!("  Records included: {}", context.records_included.len());
    println!("  Records excluded: {}", context.records_excluded_count);
    println!("  Contradictions:   {}", context.contradictions.len());
    println!("  Query time:       {:.2}ms", context.query_time_ms);
    println!("\n  Assembled context:\n{}", indent(&context.context, 4));

    // ── 8. Database stats ───────────────────────────────────────────────
    let stats = brain.admin().stats().await?;
    println!("\n── Database Stats ──");
    println!("  Total memories: {}", stats.total_count);
    println!(
        "  Episodic: {} | Semantic: {} | Working: {}",
        stats.episodic_count, stats.semantic_count, stats.working_count
    );

    // ── Cleanup ─────────────────────────────────────────────────────────
    brain.working().defocus(wm_id).await?;
    println!("\n✓ Done! Database at: {}", path.display());

    Ok(())
}

/// Generate a simple deterministic embedding from text content.
/// In production, use a real embedding model (see `openai_embeddings` example).
fn simple_embedding(text: &str, dims: usize) -> Vec<f32> {
    let mut emb = vec![0.0f32; dims];
    for (i, byte) in text.bytes().enumerate() {
        emb[i % dims] += f32::from(byte) / 255.0;
    }
    // Normalize to unit length
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
