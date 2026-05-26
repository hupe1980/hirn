//! # Multi-Agent Coding with ACL — Cedar Policy Enforcement
//!
//! This example demonstrates hirn's Cedar-based authorization for
//! a multi-agent coding team scenario:
//!
//! 1. Set up a PolicyEngine with team-based policies
//! 2. Register agents in different teams (writers, readers, admins)
//! 3. Verify authorization decisions (allow/deny) for operations
//! 4. Agent reputation-based access control (ABAC)
//! 5. Namespace classification restrictions
//!
//! Run with: `cargo run --example multi_agent_acl -p hirn`

use hirn::prelude::*;
use hirn_engine::policy::{Action, AuthzRequest, PolicyEngine};
use hirn_storage::{HirnDb, HirnDbConfig};

#[tokio::main]
async fn main() -> HirnResult<()> {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().join("team");

    // ── 1. Create Cedar policies ────────────────────────────────────────
    // Define team-based access control policies.
    let policy_text = r#"
// Senior engineers can read and write in all realms
permit(
    principal in Hirn::Team::"senior-engineers",
    action in [Hirn::Action::"remember", Hirn::Action::"recall", Hirn::Action::"think",
               Hirn::Action::"connect"],
    resource in Hirn::Realm::"development"
);

// Junior engineers can only read
permit(
    principal in Hirn::Team::"junior-engineers",
    action in [Hirn::Action::"recall", Hirn::Action::"think"],
    resource in Hirn::Realm::"development"
);

// Only admins can consolidate, forget, or do admin operations
permit(
    principal in Hirn::Team::"admins",
    action,
    resource
);

// Block agents with low reputation from writing
forbid(
    principal,
    action in [Hirn::Action::"remember", Hirn::Action::"connect"],
    resource
) when { principal.reputation < 30 };

// Restricted namespaces require admin access
forbid(
    principal,
    action,
    resource
) when { resource.classification == "restricted" }
unless { principal in Hirn::Team::"admins" };
"#;

    let engine = PolicyEngine::new(
        hirn_engine::policy::DEFAULT_SCHEMA,
        &[("team-policies", policy_text)],
    )
    .expect("valid Cedar policies");

    println!("✓ Cedar policy engine initialized\n");

    // ── 2. Register entities ────────────────────────────────────────────
    // Register teams and agents in the policy engine's entity store.
    engine
        .register_organization("acme-corp", "ACME Corp")
        .expect("register org");

    engine
        .register_team(
            "senior-engineers",
            "Senior engineering team",
            Some("acme-corp"),
        )
        .expect("register team");
    engine
        .register_team(
            "junior-engineers",
            "Junior engineering team",
            Some("acme-corp"),
        )
        .expect("register team");
    engine
        .register_team("admins", "Admin team", Some("acme-corp"))
        .expect("register team");

    // Register agents with reputations
    engine
        .register_agent("alice", 85, "2024-01-15", &["senior-engineers"])
        .expect("register alice");
    engine
        .register_agent("bob", 45, "2024-06-01", &["junior-engineers"])
        .expect("register bob");
    engine
        .register_agent("charlie", 15, "2025-01-01", &["junior-engineers"])
        .expect("register charlie"); // Low reputation
    engine
        .register_agent("admin", 100, "2023-01-01", &["admins"])
        .expect("register admin");

    // Register realm and namespaces
    engine
        .register_realm("development", "Development environment")
        .expect("register realm");
    engine
        .register_namespace("shared", "public", "development")
        .expect("register namespace");
    engine
        .register_namespace("secrets", "restricted", "development")
        .expect("register namespace");

    println!("✓ Registered agents, teams, realm, and namespaces\n");

    // ── 3. Test RBAC — team-based access ────────────────────────────────
    println!("── RBAC Tests ──");

    // Alice (senior) can remember
    let decision = engine.authorize(&AuthzRequest {
        agent_id: "alice".to_string(),
        action: Action::Remember,
        realm: "development".to_string(),
        namespace: "shared".to_string(),
    });
    println!(
        "  Alice remember in shared: {} ✓",
        if decision.allowed { "ALLOW" } else { "DENY" }
    );
    assert!(
        decision.allowed,
        "Senior engineer should be able to remember"
    );

    // Bob (junior) can recall but NOT remember
    let decision = engine.authorize(&AuthzRequest {
        agent_id: "bob".to_string(),
        action: Action::Recall,
        realm: "development".to_string(),
        namespace: "shared".to_string(),
    });
    println!(
        "  Bob recall in shared:     {} ✓",
        if decision.allowed { "ALLOW" } else { "DENY" }
    );
    assert!(decision.allowed, "Junior engineer should be able to recall");

    let decision = engine.authorize(&AuthzRequest {
        agent_id: "bob".to_string(),
        action: Action::Remember,
        realm: "development".to_string(),
        namespace: "shared".to_string(),
    });
    println!(
        "  Bob remember in shared:   {} ✓",
        if decision.allowed { "ALLOW" } else { " DENY" }
    );
    assert!(
        !decision.allowed,
        "Junior engineer should NOT be able to remember"
    );

    // Admin can do everything
    let decision = engine.authorize(&AuthzRequest {
        agent_id: "admin".to_string(),
        action: Action::Admin,
        realm: "development".to_string(),
        namespace: "shared".to_string(),
    });
    println!(
        "  Admin admin in shared:    {} ✓",
        if decision.allowed { "ALLOW" } else { "DENY" }
    );
    assert!(decision.allowed, "Admin should have full access");

    // ── 4. Test ABAC — reputation-based ─────────────────────────────────
    println!("\n── ABAC Tests (reputation) ──");

    // Charlie (rep=15) is blocked from writing even though he's a junior engineer
    let decision = engine.authorize(&AuthzRequest {
        agent_id: "charlie".to_string(),
        action: Action::Remember,
        realm: "development".to_string(),
        namespace: "shared".to_string(),
    });
    println!(
        "  Charlie (rep=15) remember: {} ✓",
        if decision.allowed { "ALLOW" } else { "DENY" }
    );
    assert!(
        !decision.allowed,
        "Low-reputation agent should be blocked from writing"
    );

    // Charlie can still read (no reputation restriction on recall)
    let decision = engine.authorize(&AuthzRequest {
        agent_id: "charlie".to_string(),
        action: Action::Recall,
        realm: "development".to_string(),
        namespace: "shared".to_string(),
    });
    println!(
        "  Charlie (rep=15) recall:   {} ✓",
        if decision.allowed { "ALLOW" } else { "DENY" }
    );

    // ── 5. Test namespace classification ─────────────────────────────────
    println!("\n── Namespace Classification Tests ──");

    // Alice cannot access restricted namespace
    let decision = engine.authorize(&AuthzRequest {
        agent_id: "alice".to_string(),
        action: Action::Recall,
        realm: "development".to_string(),
        namespace: "secrets".to_string(),
    });
    println!(
        "  Alice recall in secrets:  {} ✓",
        if decision.allowed { "ALLOW" } else { "DENY" }
    );
    assert!(
        !decision.allowed,
        "Non-admin should be blocked from restricted namespace"
    );

    // Admin CAN access restricted namespace
    let decision = engine.authorize(&AuthzRequest {
        agent_id: "admin".to_string(),
        action: Action::Recall,
        realm: "development".to_string(),
        namespace: "secrets".to_string(),
    });
    println!(
        "  Admin recall in secrets:  {} ✓",
        if decision.allowed { "ALLOW" } else { "DENY" }
    );
    assert!(decision.allowed, "Admin should access restricted namespace");

    // ── 6. Policy diagnostics ───────────────────────────────────────────
    println!("\n── Policy Diagnostics ──");
    let decision = engine.authorize(&AuthzRequest {
        agent_id: "bob".to_string(),
        action: Action::Consolidate,
        realm: "development".to_string(),
        namespace: "shared".to_string(),
    });
    println!(
        "  Bob consolidate: {}",
        if decision.allowed { "ALLOW" } else { "DENY" }
    );
    if !decision.reasons.is_empty() {
        println!("  Reasons: {:?}", decision.reasons);
    }
    if !decision.policy_ids.is_empty() {
        println!("  Matching policies: {:?}", decision.policy_ids);
    }

    // ── 7. Use with HirnDB ─────────────────────────────────────────────
    println!("\n── Integration with HirnDB ──");
    let config = HirnConfig::builder()
        .db_path(&path)
        .embedding_dimensions(64)
        .build()?;
    let storage_config = HirnDbConfig::local(path.join("lance").to_str().unwrap());
    let storage = HirnDb::open(storage_config).await?.store_arc();
    let mut brain = Hirn::open_with_config(config, storage).await?;

    // Attach the policy engine to the database
    brain.set_policy_engine(engine);

    // Register agents
    let alice_id = AgentId::new("alice").expect("valid");
    let bob_id = AgentId::new("bob").expect("valid");
    brain
        .register_agent(&alice_id, "Alice — Senior Engineer")
        .await?;
    brain
        .register_agent(&bob_id, "Bob — Junior Engineer")
        .await?;

    // Alice can remember (senior engineer)
    let record = EpisodicRecord::builder()
        .content("Implemented new caching layer with 95% hit rate")
        .event_type(EventType::Experiment)
        .agent_id(alice_id.clone())
        .importance(0.85)
        .embedding(simple_embedding("caching layer implementation", 64))
        .build()?;

    match brain.episodic().remember(record).await {
        Ok(id) => println!("  Alice stored memory: {id}"),
        Err(e) => println!("  Alice blocked: {e}"),
    }

    println!("\n✓ Multi-agent ACL demo complete!");
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
