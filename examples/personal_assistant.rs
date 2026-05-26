//! # Personal Assistant — Zero-Config Memory
//!
//! This example demonstrates using `HirnMemory` for a personal assistant
//! that learns from conversations and recalls relevant context.
//!
//! Key concepts:
//! 1. Zero-config with `HirnMemory::open()` (auto-detects providers)
//! 2. Natural language remember/recall/think
//! 3. Token-budgeted context assembly for LLM prompts
//! 4. HirnQL for advanced queries
//!
//! Run with: `cargo run --example personal_assistant -p hirn`

use hirn::prelude::*;

fn record_text(r: &RecallResult) -> &str {
    match &r.record {
        MemoryRecord::Episodic(ep) => &ep.content,
        MemoryRecord::Semantic(sem) => &sem.concept,
        MemoryRecord::Working(wm) => &wm.content,
        MemoryRecord::Procedural(p) => &p.description,
    }
}

#[tokio::main]
async fn main() -> HirnResult<()> {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let brain_path = dir.path().join("assistant_brain");

    // ── 1. Open with zero config ────────────────────────────────────────
    // HirnMemory auto-detects embedding providers from environment:
    //   OPENAI_API_KEY → OpenAI embeddings
    //   OLLAMA_HOST    → Ollama embeddings
    //   (none)         → PseudoEmbedder (testing/dev)
    let memory = HirnMemory::open(&brain_path).await?;
    println!(
        "✓ Opened personal assistant brain at {}",
        brain_path.display()
    );

    // ── 2. Learn from conversations ─────────────────────────────────────
    // The assistant remembers facts from interactions with the user.
    // Embedding + entity extraction happen automatically.
    let conversations = [
        "User prefers dark mode and uses vim keybindings in all editors",
        "User's favorite programming language is Rust, followed by TypeScript",
        "User works at a fintech startup building payment infrastructure",
        "Meeting with the team: decided to migrate from PostgreSQL to CockroachDB",
        "User is allergic to peanuts and prefers vegetarian restaurants",
        "Deployed the new payment gateway, latency dropped from 200ms to 45ms",
        "User mentioned their partner's birthday is March 15th",
        "Sprint retrospective: team velocity increased 30% after switching to Kanban",
        "User wants to learn more about distributed systems and Raft consensus",
        "Bug report: intermittent timeout on payment verification endpoint",
    ];

    for msg in &conversations {
        memory.remember(msg).await?;
        println!("  📝 Remembered: {:.60}...", msg);
    }
    println!("✓ Stored {} memories\n", conversations.len());

    // ── 3. Recall relevant memories ─────────────────────────────────────
    // Semantic search finds the most relevant memories for a query.
    println!("── Recall: \"What are the user's preferences?\" ──");
    let results = memory.recall("What are the user's preferences?", 5).await?;
    for (i, r) in results.iter().enumerate() {
        println!("  #{}: [{:.2}] {}", i + 1, r.similarity, record_text(r));
    }

    println!("\n── Recall: \"What happened with the deployment?\" ──");
    let results = memory
        .recall("What happened with the deployment?", 3)
        .await?;
    for (i, r) in results.iter().enumerate() {
        println!("  #{}: [{:.2}] {}", i + 1, r.similarity, record_text(r));
    }

    // ── 4. Think — assemble context for an LLM ─────────────────────────
    // think() combines relevant memories into optimal context for an LLM prompt.
    // It respects a token budget and includes graph-connected memories.
    println!("\n── Think: \"Help me prepare for the upcoming architecture review\" ──");
    let ctx = memory
        .think("Help me prepare for the upcoming architecture review", 1024)
        .await?;
    println!("  Tokens used: {}", ctx.token_count);
    println!("  Records included: {}", ctx.records_included.len());
    println!("  Context preview:\n{}\n", indent(&ctx.context, 4));

    // In a real assistant, you'd pass ctx.context to an LLM:
    // let response = llm.chat(&format!(
    //     "You are a personal assistant. Here is relevant context:\n{}\n\nUser: {}",
    //     ctx.context,
    //     "Help me prepare for the upcoming architecture review"
    // )).await?;

    // ── 5. Advanced queries with HirnQL ─────────────────────────────────
    println!("── HirnQL: recall with importance filter ──");
    let result = memory
        .query(r#"RECALL episodic ABOUT "work decisions" WHERE importance > 0.5 LIMIT 5"#)
        .await?;
    match &result {
        QueryResult::Records(r) => {
            println!(
                "  Found {} records in {:.2}ms",
                r.records_returned, r.query_time_ms
            );
        }
        _ => {}
    }

    // ── 6. Builder API for fine-tuned recall ───────────────────────────
    println!("\n── Builder API: recall with spreading activation ──");
    let results = memory
        .recall_builder("system performance issues")
        .limit(5)
        .execute()
        .await?;
    for (i, r) in results.iter().enumerate() {
        println!("  #{}: [{:.2}] {}", i + 1, r.similarity, record_text(r));
    }

    println!("\n✓ Personal assistant demo complete!");
    Ok(())
}

fn indent(s: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    s.lines()
        .map(|line| format!("{pad}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}
