use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use hirn_core::HirnConfig;
use hirn_core::embed::TokenCounter as _;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::semantic::SemanticRecord;
use hirn_core::types::{AgentId, EventType, KnowledgeType};

use hirn_engine::HirnDB;
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

/// Build a DB with `n` episodic + `n/5` semantic records.
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

    let sem_count = n / 5;
    for i in 0..sem_count {
        let desc = format!(
            "Semantic knowledge about topic {i}: caching strategies in distributed systems"
        );
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

fn bench_think_query(c: &mut Criterion) {
    let rt = rt();
    let (db, _dir) = build_db(10_000, &rt);

    c.bench_function("think_10k_records_budget_4096", |b| {
        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for services", dims);
        b.iter(|| {
            let result = rt
                .block_on(db.recall_view().think(emb.clone()).budget(4096).execute())
                .unwrap();
            black_box(&result.context);
        });
    });

    c.bench_function("think_10k_records_budget_512", |b| {
        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for services", dims);
        b.iter(|| {
            let result = rt
                .block_on(db.recall_view().think(emb.clone()).budget(512).execute())
                .unwrap();
            black_box(&result.context);
        });
    });
}

fn bench_think_budget_enforcement(c: &mut Criterion) {
    let rt = rt();
    let (db, _dir) = build_db(500, &rt);
    let tokenizer =
        hirn_provider::TiktokenTokenizer::new(hirn_provider::TokenizerModel::Cl100kBase).unwrap();

    c.bench_function("think_budget_enforcement_sweep", |b| {
        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment", dims);
        b.iter(|| {
            for budget in [64, 128, 256, 512, 1024, 2048, 4096] {
                let result = rt
                    .block_on(db.recall_view().think(emb.clone()).budget(budget).execute())
                    .unwrap();
                let tokens = tokenizer.count_tokens(&result.context);
                assert!(tokens <= budget);
                black_box(&result.context);
            }
        });
    });
}

fn bench_compression_ratio(c: &mut Criterion) {
    let rt = rt();
    let (db, _dir) = build_db(1000, &rt);
    let tokenizer =
        hirn_provider::TiktokenTokenizer::new(hirn_provider::TokenizerModel::Cl100kBase).unwrap();

    c.bench_function("think_compression_ratio_measurement", |b| {
        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for services", dims);
        b.iter(|| {
            // Get all candidates at full size (large budget)
            let full = rt
                .block_on(
                    db.recall_view()
                        .think(emb.clone())
                        .budget(100_000)
                        .execute(),
                )
                .unwrap();
            let full_tokens = tokenizer.count_tokens(&full.context);

            // Get compressed version
            let compressed = rt
                .block_on(db.recall_view().think(emb.clone()).budget(1024).execute())
                .unwrap();
            let compressed_tokens = tokenizer.count_tokens(&compressed.context);

            // Verify compression actually compresses
            assert!(
                compressed_tokens < full_tokens,
                "compressed ({compressed_tokens}) should be less than full ({full_tokens})"
            );

            black_box((full_tokens, compressed_tokens));
        });
    });
}

criterion_group!(
    benches,
    bench_think_query,
    bench_think_budget_enforcement,
    bench_compression_ratio
);
criterion_main!(benches);
