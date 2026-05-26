//! # Enterprise RAG with Cedar Policies
//!
//! This example demonstrates hirn as an enterprise RAG (Retrieval-Augmented
//! Generation) backend with multi-tenant Cedar policy enforcement:
//!
//! 1. Multi-realm isolation (tenant A vs tenant B)
//! 2. Cedar policies enforce realm boundaries
//! 3. Namespace classification (public, confidential, restricted)
//! 4. HirnQL for policy management (GRANT/REVOKE/SHOW POLICIES)
//! 5. Cross-tenant access denied by Cedar
//!
//! Run with: `cargo run --example enterprise_rag_cedar -p hirn`

use hirn::prelude::*;
use hirn_engine::policy::{Action, AuthzRequest, PolicyEngine};
use hirn_storage::{HirnDb, HirnDbConfig};

#[tokio::main]
async fn main() -> HirnResult<()> {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let path = dir.path().join("enterprise");

    // ── 1. Define enterprise Cedar policies ─────────────────────────────
    let enterprise_policies = r#"
// ── Tenant isolation ─────────────────────────────────────────
// Each tenant's agents can only access their own realm.

// Tenant A: full access to their realm
permit(
    principal in Hirn::Organization::"tenant-a",
    action in [Hirn::Action::"remember", Hirn::Action::"recall",
               Hirn::Action::"think", Hirn::Action::"connect",
               Hirn::Action::"watch"],
    resource in Hirn::Realm::"tenant-a"
);

// Tenant B: full access to their realm
permit(
    principal in Hirn::Organization::"tenant-b",
    action in [Hirn::Action::"remember", Hirn::Action::"recall",
               Hirn::Action::"think", Hirn::Action::"connect",
               Hirn::Action::"watch"],
    resource in Hirn::Realm::"tenant-b"
);

// ── Classification-based access ──────────────────────────────
// Confidential data requires team-level access
forbid(
    principal,
    action,
    resource
) when { resource.classification == "confidential" }
unless { principal in Hirn::Team::"data-team-a" || principal in Hirn::Team::"data-team-b" };

// Restricted data requires admin
forbid(
    principal,
    action,
    resource
) when { resource.classification == "restricted" }
unless { principal in Hirn::Team::"platform-admins" };

// ── Admin access ─────────────────────────────────────────────
// Platform admins have full cross-tenant access (for ops)
permit(
    principal in Hirn::Team::"platform-admins",
    action,
    resource
);

// ── Consolidation restricted to admins ──────────────────────
// Only admins can run consolidation and admin operations
forbid(
    principal,
    action in [Hirn::Action::"consolidate", Hirn::Action::"admin",
               Hirn::Action::"forget"],
    resource
) unless { principal in Hirn::Team::"platform-admins" };
"#;

    let engine = PolicyEngine::new(
        hirn_engine::policy::DEFAULT_SCHEMA,
        &[("enterprise", enterprise_policies)],
    )
    .expect("valid enterprise policies");

    println!("✓ Enterprise Cedar policies loaded\n");

    // ── 2. Set up entity hierarchy ──────────────────────────────────────
    // Organizations
    engine
        .register_organization("tenant-a", "Tenant A — FinTech Corp")
        .unwrap();
    engine
        .register_organization("tenant-b", "Tenant B — HealthCare Inc")
        .unwrap();

    // Teams
    engine
        .register_team("data-team-a", "Tenant A data team", Some("tenant-a"))
        .unwrap();
    engine
        .register_team("data-team-b", "Tenant B data team", Some("tenant-b"))
        .unwrap();
    engine
        .register_team("platform-admins", "Platform administrators", None)
        .unwrap();

    // Agents
    engine
        .register_agent("alice-a", 80, "2024-01-01", &["data-team-a"])
        .unwrap();
    engine
        .register_agent("bob-b", 75, "2024-02-01", &["data-team-b"])
        .unwrap();
    engine
        .register_agent("ops-admin", 100, "2023-01-01", &["platform-admins"])
        .unwrap();

    // Realms
    engine
        .register_realm("tenant-a", "Tenant A isolated realm")
        .unwrap();
    engine
        .register_realm("tenant-b", "Tenant B isolated realm")
        .unwrap();

    // Namespaces with classifications
    engine
        .register_namespace("public-docs-a", "public", "tenant-a")
        .unwrap();
    engine
        .register_namespace("financial-data-a", "confidential", "tenant-a")
        .unwrap();
    engine
        .register_namespace("pii-a", "restricted", "tenant-a")
        .unwrap();
    engine
        .register_namespace("patient-records-b", "confidential", "tenant-b")
        .unwrap();
    println!("✓ Entity hierarchy registered\n");

    // ── 3. Test tenant isolation ────────────────────────────────────────
    println!("── Tenant Isolation ──");

    // Alice (tenant A) can access tenant A realm
    let d = engine.authorize(&AuthzRequest {
        agent_id: "alice-a".to_string(),
        action: Action::Recall,
        realm: "tenant-a".to_string(),
        namespace: "public-docs-a".to_string(),
    });
    println!("  Alice-A recall in tenant-a/public:  {}", decision_str(&d));
    assert!(d.allowed);

    // Alice (tenant A) CANNOT access tenant B realm
    let d = engine.authorize(&AuthzRequest {
        agent_id: "alice-a".to_string(),
        action: Action::Recall,
        realm: "tenant-b".to_string(),
        namespace: "patient-records-b".to_string(),
    });
    println!("  Alice-A recall in tenant-b:         {}", decision_str(&d));
    assert!(!d.allowed, "Cross-tenant access must be denied");

    // Bob (tenant B) CANNOT access tenant A realm
    let d = engine.authorize(&AuthzRequest {
        agent_id: "bob-b".to_string(),
        action: Action::Recall,
        realm: "tenant-a".to_string(),
        namespace: "public-docs-a".to_string(),
    });
    println!("  Bob-B recall in tenant-a:           {}", decision_str(&d));
    assert!(!d.allowed, "Cross-tenant access must be denied");

    // Platform admin CAN access both realms
    let d = engine.authorize(&AuthzRequest {
        agent_id: "ops-admin".to_string(),
        action: Action::Recall,
        realm: "tenant-a".to_string(),
        namespace: "public-docs-a".to_string(),
    });
    println!("  Admin recall in tenant-a:           {}", decision_str(&d));
    assert!(d.allowed);

    let d = engine.authorize(&AuthzRequest {
        agent_id: "ops-admin".to_string(),
        action: Action::Recall,
        realm: "tenant-b".to_string(),
        namespace: "patient-records-b".to_string(),
    });
    println!("  Admin recall in tenant-b:           {}", decision_str(&d));
    assert!(d.allowed);

    // ── 4. Test classification-based access ─────────────────────────────
    println!("\n── Classification-Based Access ──");

    // Alice can access confidential data (she's in data-team-a)
    let d = engine.authorize(&AuthzRequest {
        agent_id: "alice-a".to_string(),
        action: Action::Recall,
        realm: "tenant-a".to_string(),
        namespace: "financial-data-a".to_string(),
    });
    println!("  Alice-A recall confidential:        {}", decision_str(&d));
    assert!(d.allowed);

    // Alice CANNOT access restricted (PII) data
    let d = engine.authorize(&AuthzRequest {
        agent_id: "alice-a".to_string(),
        action: Action::Recall,
        realm: "tenant-a".to_string(),
        namespace: "pii-a".to_string(),
    });
    println!("  Alice-A recall restricted (PII):    {}", decision_str(&d));
    assert!(!d.allowed, "Only admins can access restricted data");

    // Admin CAN access restricted data
    let d = engine.authorize(&AuthzRequest {
        agent_id: "ops-admin".to_string(),
        action: Action::Recall,
        realm: "tenant-a".to_string(),
        namespace: "pii-a".to_string(),
    });
    println!("  Admin recall restricted (PII):      {}", decision_str(&d));
    assert!(d.allowed);

    // ── 5. Test admin-only operations ───────────────────────────────────
    println!("\n── Admin-Only Operations ──");

    let d = engine.authorize(&AuthzRequest {
        agent_id: "alice-a".to_string(),
        action: Action::Consolidate,
        realm: "tenant-a".to_string(),
        namespace: "public-docs-a".to_string(),
    });
    println!("  Alice-A consolidate:                {}", decision_str(&d));
    assert!(!d.allowed, "Non-admin cannot consolidate");

    let d = engine.authorize(&AuthzRequest {
        agent_id: "ops-admin".to_string(),
        action: Action::Consolidate,
        realm: "tenant-a".to_string(),
        namespace: "public-docs-a".to_string(),
    });
    println!("  Admin consolidate:                  {}", decision_str(&d));
    assert!(d.allowed);

    // ── 6. RAG workflow with policy-enforced HirnDB ─────────────────────
    println!("\n── RAG Workflow ──");

    let config = HirnConfig::builder()
        .db_path(&path)
        .embedding_dimensions(64)
        .build()?;
    let storage_config = HirnDbConfig::local(path.join("lance").to_str().unwrap());
    let storage = HirnDb::open(storage_config).await?.store_arc();
    let mut brain = Hirn::open_with_config(config, storage).await?;
    brain.set_policy_engine(engine);

    let alice = AgentId::new("alice-a").expect("valid");
    brain
        .register_agent(&alice, "Alice — FinTech Data Analyst")
        .await?;

    // Alice ingests documents into tenant A's knowledge base
    let docs = [
        "Q3 revenue grew 15% YoY driven by payment processing volume",
        "New fraud detection model reduces false positives by 40%",
        "Customer churn rate decreased to 2.1% after loyalty program launch",
        "Payment gateway latency P99 improved from 200ms to 45ms after CDN deployment",
    ];

    for doc in &docs {
        let record = EpisodicRecord::builder()
            .content(*doc)
            .event_type(EventType::Observation)
            .agent_id(alice.clone())
            .importance(0.8)
            .embedding(simple_embedding(doc, 64))
            .build()?;
        brain.episodic().remember(record).await?;
    }
    println!("  Ingested {} documents into knowledge base", docs.len());

    // RAG query: Alice recalls relevant context for an LLM
    let query_emb = simple_embedding("What improvements did we make to payment processing?", 64);
    let results = brain
        .recall_view()
        .query(query_emb.clone())
        .limit(3)
        .execute()
        .await?;
    println!("  RAG recall returned {} results", results.len());
    for (i, r) in results.iter().enumerate() {
        let content = match &r.record {
            MemoryRecord::Episodic(ep) => &ep.content,
            _ => "(non-episodic)",
        };
        println!("    #{}: [{:.2}] {:.70}", i + 1, r.composite_score, content);
    }

    // Think for LLM context assembly
    let ctx = brain
        .recall_view()
        .think(query_emb)
        .budget(1024)
        .execute()
        .await?;
    println!(
        "  Context assembled: {} tokens, {} records",
        ctx.token_count,
        ctx.records_included.len()
    );

    // HirnQL policy management
    println!("\n── HirnQL Policy Management ──");
    println!("  Example HirnQL commands for runtime policy management:");
    println!("  > GRANT remember, recall ON REALM \"tenant-a\" TO AGENT \"new-analyst\"");
    println!("  > REVOKE remember ON REALM \"tenant-a\" FROM AGENT \"departing-employee\"");
    println!("  > SHOW POLICIES FOR AGENT \"alice-a\"");
    println!("  > EXPLAIN POLICY FOR AGENT \"alice-a\" ON REALM \"tenant-a\" ACTION recall");

    println!("\n✓ Enterprise RAG with Cedar demo complete!");
    Ok(())
}

fn decision_str(d: &hirn_engine::policy::AuthzDecision) -> &'static str {
    if d.allowed { "ALLOW ✓" } else { "DENY  ✗" }
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
