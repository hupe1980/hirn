//! # Consolidation — From Episodes to Knowledge
//!
//! This example demonstrates hirn's memory consolidation pipeline,
//! inspired by hippocampal replay during sleep:
//!
//! 1. Store many related episodic memories
//! 2. Run consolidation (episode segmentation → pattern detection → concept extraction)
//! 3. Observe new semantic knowledge auto-created from patterns
//! 4. Verify provenance links from semantic back to episodic sources
//! 5. Run forgetting to prune low-importance memories
//!
//! Run with: `cargo run --example consolidation -p hirn`

use hirn::prelude::*;
use hirn_storage::{HirnDb, HirnDbConfig};

#[tokio::main]
async fn main() -> HirnResult<()> {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().join("consolidation");

    let config = HirnConfig::builder()
        .db_path(&path)
        .embedding_dimensions(64)
        .build()?;
    let storage_config = HirnDbConfig::local(path.join("lance").to_str().unwrap());
    let storage = HirnDb::open(storage_config).await?.store_arc();
    let brain = Hirn::open_with_config(config, storage).await?;

    let agent = AgentId::new("learner").expect("non-empty agent id");
    brain.register_agent(&agent, "Learning Agent").await?;

    // ── Phase 1: Generate many related experiences ──────────────────────
    // Simulate an agent learning about caching over several days.
    // Related episodes should consolidate into stable semantic knowledge.
    let episodes = [
        // Cluster 1: Redis caching experiences
        (
            "Experiment: Redis cache with 5min TTL reduced API latency from 200ms to 15ms",
            0.85,
        ),
        (
            "Observation: Redis cache hit rate stable at 94% over 48 hours",
            0.70,
        ),
        (
            "Experiment: Increasing Redis memory from 4GB to 8GB improved hit rate to 97%",
            0.80,
        ),
        (
            "Decision: Adopt Redis as primary caching layer for all API endpoints",
            0.90,
        ),
        (
            "Observation: Redis cluster failover completed in 2.3 seconds with no data loss",
            0.75,
        ),
        // Cluster 2: CDN caching experiences
        (
            "Experiment: CloudFront CDN reduced static asset load time by 65%",
            0.80,
        ),
        (
            "Observation: CDN cache invalidation takes up to 15 minutes globally",
            0.65,
        ),
        (
            "Experiment: Using versioned URLs eliminates need for cache invalidation",
            0.85,
        ),
        (
            "Decision: Move all static assets to CDN with versioned URL strategy",
            0.88,
        ),
        // Cluster 3: Database query caching
        (
            "Error: Query cache caused stale data bug — user saw outdated product prices",
            0.70,
        ),
        (
            "Experiment: Write-through cache pattern eliminated stale data issues",
            0.82,
        ),
        (
            "Observation: Write-through adds 3ms to write path but eliminates all staleness",
            0.78,
        ),
        (
            "Decision: Use write-through for transactional data, TTL-based for analytics",
            0.85,
        ),
        // Low-importance noise (should be filtered/forgotten)
        ("Observation: Office coffee machine broken again", 0.05),
        ("Observation: Team standup moved to 10am", 0.10),
    ];

    let mut episode_ids = Vec::new();
    for (content, importance) in &episodes {
        let ep = EpisodicRecord::builder()
            .content(*content)
            .event_type(EventType::Observation)
            .agent_id(agent.clone())
            .importance(*importance)
            .embedding(simple_embedding(content, 64))
            .entity("cache", "technology")
            .build()?;
        episode_ids.push(brain.episodic().remember(ep).await?);
    }
    println!("✓ Stored {} episodic memories\n", episodes.len());

    let stats_before = brain.admin().stats().await?;
    println!("── Before Consolidation ──");
    println!(
        "  Episodic: {} | Semantic: {}",
        stats_before.episodic_count, stats_before.semantic_count
    );

    // ── Phase 2: Run consolidation ──────────────────────────────────────
    println!("\n── Running Consolidation Pipeline ──");

    let result = brain
        .admin()
        .consolidate()
        .topic_threshold(0.15) // lower threshold to detect patterns in simple embeddings
        .thread_threshold(0.15)
        .execute()
        .await?;

    println!("  Records processed:       {}", result.records_processed);
    println!("  Segments created:        {}", result.segments_created);
    println!("  Patterns detected:       {}", result.patterns_detected);
    println!("  Narrative threads formed: {}", result.threads_formed);
    println!("  Concepts extracted:      {}", result.concepts_extracted);
    println!(
        "  Provenance edges:        {}",
        result.provenance_edges_created
    );
    println!("  Episodes archived:       {}", result.episodes_archived);
    println!(
        "  Execution time:          {:.1}ms",
        result.execution_time_ms
    );

    // ── Phase 3: Examine new semantic knowledge ─────────────────────────
    let stats_after = brain.admin().stats().await?;
    println!("\n── After Consolidation ──");
    println!(
        "  Episodic: {} | Semantic: {}",
        stats_after.episodic_count, stats_after.semantic_count
    );

    // List all semantic records
    let semantics = brain
        .semantic()
        .list(&hirn::semantic::SemanticFilter {
            knowledge_type: None,
            min_confidence: None,
            namespace: None,
            limit: None,
        })
        .await?;

    if !semantics.is_empty() {
        println!("\n  Consolidated knowledge:");
        for sem in &semantics {
            println!(
                "    • [{}] {} (confidence={:.2}, sources={})",
                sem.concept,
                sem.description.chars().take(60).collect::<String>(),
                sem.confidence,
                sem.source_episodes.len()
            );

            // ── Phase 4: Verify provenance ──────────────────────────────
            let trace = brain.recall_view().trace(sem.id).execute().await?;
            println!(
                "      Provenance: trust={:.2}, {} source episodes",
                trace.trust_score,
                trace.source_episodes.len()
            );
        }
    }

    // ── Phase 5: Adaptive forgetting ────────────────────────────────────
    println!("\n── Adaptive Forgetting ──");

    // Archive low-importance episodes
    let mut archived = 0;
    for (i, &id) in episode_ids.iter().enumerate() {
        if episodes[i].1 < 0.15 {
            brain.episodic().archive(id).await?;
            archived += 1;
            println!("  → Archived: {:.50}...", episodes[i].0);
        }
    }
    println!("  Archived {} low-importance memories", archived);

    // ── Final: Recall after consolidation ───────────────────────────────
    println!("\n── Recall After Consolidation ──");

    let query = simple_embedding("caching strategy best practices", 64);
    let results = brain
        .recall_view()
        .query(query)
        .activation(ActivationMode::Spreading)
        .limit(5)
        .execute()
        .await?;

    for (i, r) in results.iter().enumerate() {
        let label = match &r.record {
            MemoryRecord::Episodic(ep) => {
                format!("[ep] {}", ep.content.chars().take(55).collect::<String>())
            }
            MemoryRecord::Semantic(s) => {
                format!(
                    "[sem:{}] {}",
                    s.concept,
                    s.description.chars().take(45).collect::<String>()
                )
            }
            MemoryRecord::Working(w) => {
                format!("[wm] {}", w.content.chars().take(55).collect::<String>())
            }
            MemoryRecord::Procedural(p) => {
                format!(
                    "[proc] {}",
                    p.description.chars().take(55).collect::<String>()
                )
            }
        };
        println!("  #{}: score={:.3} | {}", i + 1, r.composite_score, label);
    }

    println!("\n✓ Consolidation demo complete!");
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
