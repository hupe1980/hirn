use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use hirn_core::HirnConfig;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::semantic::SemanticRecord;
use hirn_core::types::{AgentId, EventType, KnowledgeType};

use hirn_engine::HirnDB;
use hirn_engine::ql;
use hirn_storage::memory_store::MemoryStore;

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn agent() -> AgentId {
    AgentId::new("bench").unwrap()
}

fn pseudo_embedding(text: &str, dims: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; dims];
    let bytes = text.as_bytes();
    for (i, window) in bytes.windows(3).enumerate() {
        let hash = u32::from(window[0])
            .wrapping_mul(31)
            .wrapping_add(u32::from(window[1]))
            .wrapping_mul(31)
            .wrapping_add(u32::from(window[2]));
        let idx = (hash as usize).wrapping_add(i) % dims;
        embedding[idx] += 1.0;
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut embedding {
            *v /= norm;
        }
    } else {
        embedding[0] = 1.0;
    }
    embedding
}

fn build_db(n: usize, rt: &tokio::runtime::Runtime) -> (HirnDB, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench");
    let config = HirnConfig::builder()
        .db_path(&path)
        .token_budget(4096)
        .build()
        .unwrap();
    let db = rt
        .block_on(HirnDB::open_with_config(
            config,
            Arc::new(MemoryStore::new()),
        ))
        .unwrap();
    let dims = db.embedding_dims();

    for i in 0..n {
        let content = format!(
            "Episodic record {i}: deployment strategy for service-{}",
            i % 50
        );
        let emb = pseudo_embedding(&content, dims);
        let rec = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content(&content)
            .summary(format!("ep {i}"))
            .importance((i as f32 % 7.0).mul_add(0.1, 0.3))
            .agent_id(agent())
            .embedding(emb)
            .build()
            .unwrap();
        rt.block_on(db.episodic().remember(rec)).unwrap();
    }

    let sem = n / 5;
    for i in 0..sem {
        let desc = format!("Semantic knowledge about topic {i}: caching strategies");
        let emb = pseudo_embedding(&desc, dims);
        let rec = SemanticRecord::builder()
            .concept(format!("topic_{i}"))
            .knowledge_type(KnowledgeType::Propositional)
            .description(&desc)
            .confidence(0.8)
            .embedding(emb)
            .agent_id(agent())
            .build()
            .unwrap();
        rt.block_on(db.semantic().store(rec)).unwrap();
    }

    (db, dir)
}

// ─── Parse Benchmarks ────────────────────────────────────────

fn bench_ql_parse(c: &mut Criterion) {
    c.bench_function("ql_parse_recall", |b| {
        b.iter(|| {
            let stmt =
                ql::parse(r#"RECALL episodic ABOUT "deployment strategies" LIMIT 10"#).unwrap();
            black_box(&stmt);
        });
    });

    c.bench_function("ql_parse_think", |b| {
        b.iter(|| {
            let stmt = ql::parse(r#"THINK ABOUT "deployment strategies" BUDGET 4096"#).unwrap();
            black_box(&stmt);
        });
    });

    c.bench_function("ql_parse_recall_complex", |b| {
        b.iter(|| {
            let stmt = ql::parse(
                r#"RECALL episodic, semantic ABOUT "vector search" EXPAND GRAPH DEPTH 2 ACTIVATION spreading WHERE importance > 0.75 LIMIT 5"#,
            )
            .unwrap();
            black_box(&stmt);
        });
    });
}

// ─── Plan Benchmarks ─────────────────────────────────────────

fn bench_ql_plan(c: &mut Criterion) {
    let recall_stmt =
        ql::parse(r#"RECALL episodic ABOUT "deployment strategies" LIMIT 10"#).unwrap();
    let think_stmt = ql::parse(r#"THINK ABOUT "deployment strategies" BUDGET 4096"#).unwrap();

    c.bench_function("ql_plan_recall", |b| {
        b.iter(|| {
            let plan = ql::plan(&recall_stmt, None);
            black_box(&plan);
        });
    });

    c.bench_function("ql_plan_think", |b| {
        b.iter(|| {
            let plan = ql::plan(&think_stmt, None);
            black_box(&plan);
        });
    });
}

// ─── Execute Benchmarks ──────────────────────────────────────

fn bench_ql_execute(c: &mut Criterion) {
    let rt = rt();
    let (db, _dir) = build_db(500, &rt);

    c.bench_function("ql_execute_recall_500", |b| {
        b.iter(|| {
            let result = rt
                .block_on(
                    db.ql()
                        .execute(r#"RECALL episodic ABOUT "deployment strategy" LIMIT 10"#),
                )
                .unwrap();
            black_box(&result);
        });
    });

    c.bench_function("ql_execute_think_500", |b| {
        b.iter(|| {
            let result = rt
                .block_on(
                    db.ql()
                        .execute(r#"THINK ABOUT "deployment strategy" BUDGET 2048"#),
                )
                .unwrap();
            black_box(&result);
        });
    });

    c.bench_function("ql_explain_recall", |b| {
        b.iter(|| {
            let plan = db
                .ql()
                .explain(r#"RECALL episodic ABOUT "deployment strategy" LIMIT 10"#)
                .unwrap();
            black_box(&plan);
        });
    });
}

criterion_group!(benches, bench_ql_parse, bench_ql_plan, bench_ql_execute);
criterion_main!(benches);
