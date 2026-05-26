//! # Multi-Agent Memory — Isolation and Sharing
//!
//! This example demonstrates hirn's multi-agent capabilities:
//! 1. Agent-private memory (isolated by default)
//! 2. Shared namespace (visible to all agents)
//! 3. Team namespaces (visible to specific agents)
//! 4. Memory sharing and promotion
//! 5. Cross-agent consolidation
//!
//! Run with: `cargo run --example multi_agent -p hirn`

#![allow(clippy::large_stack_frames)]

use hirn::prelude::*;
use hirn_storage::{HirnDb, HirnDbConfig};

#[tokio::main]
async fn main() -> HirnResult<()> {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().join("team_brain");

    let config = HirnConfig::builder()
        .db_path(&path)
        .embedding_dimensions(64)
        .build()?;
    let storage_config = HirnDbConfig::local(path.join("lance").to_str().unwrap());
    let storage = HirnDb::open(storage_config).await?.store_arc();
    let brain = Hirn::open_with_config(config, storage).await?;

    // ── Register agents ─────────────────────────────────────────────────
    let alice = AgentId::new("alice").expect("non-empty agent id");
    let bob = AgentId::new("bob").expect("non-empty agent id");

    brain
        .register_agent(&alice, "Alice — Frontend Engineer")
        .await?;
    brain.register_agent(&bob, "Bob — Backend Engineer").await?;
    println!("✓ Registered agents: Alice, Bob");

    // ── Agent-scoped contexts ───────────────────────────────────────────
    let ctx_alice = brain.as_agent(&alice).await?;
    let ctx_bob = brain.as_agent(&bob).await?;

    // ── 1. Private memories ─────────────────────────────────────────────
    // Each agent's memories are private by default
    let alice_mem = EpisodicRecord::builder()
        .content("React 19 Server Components reduce client bundle by 30%")
        .event_type(EventType::Experiment)
        .agent_id(alice.clone())
        .importance(0.80)
        .embedding(simple_embedding("react server components bundle", 64))
        .build()?;
    let alice_id = ctx_alice.remember(alice_mem).await?;

    let bob_mem = EpisodicRecord::builder()
        .content("PostgreSQL jsonb queries are 10x faster with GIN index")
        .event_type(EventType::Experiment)
        .agent_id(bob.clone())
        .importance(0.85)
        .embedding(simple_embedding("postgres jsonb gin index performance", 64))
        .build()?;
    let bob_id = ctx_bob.remember(bob_mem).await?;

    println!("✓ Stored private memories");

    // Alice CAN see her own memory
    assert!(ctx_alice.inspect(alice_id).await.is_ok());
    println!("  ✓ Alice can see her own memory");

    // Alice CANNOT see Bob's private memory
    assert!(ctx_alice.inspect(bob_id).await.is_err());
    println!("  ✓ Alice cannot see Bob's private memory");

    // Bob CANNOT see Alice's private memory
    assert!(ctx_bob.inspect(alice_id).await.is_err());
    println!("  ✓ Bob cannot see Alice's private memory");

    // ── 2. Shared namespace ─────────────────────────────────────────────
    // Memories in the shared namespace are visible to all agents
    let mut shared_mem = EpisodicRecord::builder()
        .content("API response times improved 40% after CDN rollout")
        .event_type(EventType::Observation)
        .agent_id(alice.clone())
        .importance(0.90)
        .embedding(simple_embedding("api response time cdn improvement", 64))
        .build()?;
    shared_mem.namespace = Namespace::shared();

    let shared_id = ctx_alice.remember(shared_mem).await?;

    // Both agents can see shared memories
    assert!(ctx_alice.inspect(shared_id).await.is_ok());
    assert!(ctx_bob.inspect(shared_id).await.is_ok());
    println!("✓ Shared memories visible to all agents");

    // ── 3. Team namespace ───────────────────────────────────────────────
    brain
        .create_team_namespace("platform_team", vec![alice.clone(), bob.clone()])
        .await?;
    println!("✓ Created team namespace: platform_team");

    // Re-create agent contexts so they pick up the new namespace
    let ctx_alice = brain.as_agent(&alice).await?;
    let ctx_bob = brain.as_agent(&bob).await?;

    let team_ns = Namespace::new("platform_team").expect("valid namespace");
    let team_mem = EpisodicRecord::builder()
        .content("Team decision: adopt GraphQL for the new API gateway")
        .event_type(EventType::Decision)
        .agent_id(bob.clone())
        .importance(0.95)
        .namespace(team_ns.clone())
        .embedding(simple_embedding("graphql api gateway team decision", 64))
        .build()?;
    let team_id = ctx_bob.remember_in(team_mem, team_ns).await?;

    // Both team members can see it
    assert!(ctx_alice.inspect(team_id).await.is_ok());
    assert!(ctx_bob.inspect(team_id).await.is_ok());
    println!("  ✓ Team memories visible to team members");

    // ── 4. Share private → shared ─────────────────────────────────────
    // Alice discovers something important and shares it with everyone
    let promoted_id = ctx_alice
        .share_memory(alice_id, &Namespace::shared())
        .await?;
    println!("✓ Alice shared her discovery to shared namespace");

    // Now Bob can see it too
    assert!(ctx_bob.inspect(promoted_id).await.is_ok());
    println!("  ✓ Bob can now see Alice's promoted memory");

    // ── 5. Agent-scoped recall ──────────────────────────────────────────
    let query = simple_embedding("performance optimization", 64);

    println!("\n── Alice's recall ──");
    let alice_results = ctx_alice.recall(query.clone()).limit(5).execute().await?;
    for r in &alice_results {
        let content = match &r.record {
            MemoryRecord::Episodic(ep) => &ep.content,
            MemoryRecord::Semantic(s) => &s.description,
            MemoryRecord::Working(w) => &w.content,
            MemoryRecord::Procedural(p) => &p.description,
        };
        println!("  score={:.3} | {:.70}", r.composite_score, content);
    }

    println!("\n── Bob's recall ──");
    let bob_results = ctx_bob.recall(query).limit(5).execute().await?;
    for r in &bob_results {
        let content = match &r.record {
            MemoryRecord::Episodic(ep) => &ep.content,
            MemoryRecord::Semantic(s) => &s.description,
            MemoryRecord::Working(w) => &w.content,
            MemoryRecord::Procedural(p) => &p.description,
        };
        println!("  score={:.3} | {:.70}", r.composite_score, content);
    }

    // ── 6. Cross-agent consolidation ────────────────────────────────────
    // Both agents learn about the same concept independently — store semantic knowledge
    let alice_knowledge = hirn::semantic::SemanticRecord::builder()
        .concept("caching_strategy")
        .description("CDN caching with 5-minute TTL is optimal for our API responses")
        .knowledge_type(KnowledgeType::Prescriptive)
        .confidence(0.85)
        .agent_id(alice.clone())
        .build()?;
    brain.semantic().store(alice_knowledge).await?;

    let bob_knowledge = hirn::semantic::SemanticRecord::builder()
        .concept("caching_strategy")
        .description("Redis caching with 3-minute TTL provides best latency for API responses")
        .knowledge_type(KnowledgeType::Prescriptive)
        .confidence(0.78)
        .agent_id(bob.clone())
        .build()?;
    brain.semantic().store(bob_knowledge).await?;

    let result = brain
        .admin()
        .cross_agent_consolidate(&Namespace::shared(), 0.9)
        .await?;
    println!("\n── Cross-Agent Consolidation ──");
    println!("  Concepts merged: {}", result.merged_count);
    println!("  Merged IDs: {:?}", result.merged_ids);
    println!("  Contradictions: {}", result.contradiction_count);
    println!("  Contradiction pairs: {:?}", result.contradiction_pairs);

    // ── Summary ─────────────────────────────────────────────────────────
    let stats = brain.admin().stats().await?;
    println!("\n── Final Stats ──");
    println!("  Total memories: {}", stats.total_count);
    println!(
        "  Episodic: {} | Semantic: {}",
        stats.episodic_count, stats.semantic_count
    );

    println!("\n✓ Multi-agent demo complete!");
    Ok(())
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
