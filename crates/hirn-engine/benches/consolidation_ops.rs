use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use hirn_core::HirnConfig;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::types::{AgentId, EventType};

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
            "Episode {i}: meeting about project-{} with team-{}, discussed topic-{}",
            i % 10,
            i % 5,
            i % 20
        );
        let emb = pseudo_embedding(&content, dims);
        let rec = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content(&content)
            .summary(format!("meeting {i}"))
            .importance((i as f32 % 7.0).mul_add(0.1, 0.3))
            .agent_id(agent())
            .embedding(emb)
            .build()
            .unwrap();
        rt.block_on(db.episodic().remember(rec)).unwrap();
    }

    (db, dir)
}

fn bench_consolidation(c: &mut Criterion) {
    let rt = rt();
    let (db, _dir) = build_db(200, &rt);

    c.bench_function("consolidation_200_records", |b| {
        b.iter(|| {
            let result = rt.block_on(db.admin().consolidate().execute()).unwrap();
            black_box(&result);
        });
    });
}

fn bench_consolidation_with_config(c: &mut Criterion) {
    let rt = rt();
    let (db, _dir) = build_db(500, &rt);

    c.bench_function("consolidation_500_topic_0.5", |b| {
        b.iter(|| {
            let result = rt
                .block_on(
                    db.admin()
                        .consolidate()
                        .topic_threshold(0.5)
                        .surprise_threshold(0.3)
                        .execute(),
                )
                .unwrap();
            black_box(&result);
        });
    });

    c.bench_function("consolidation_500_with_archive", |b| {
        b.iter(|| {
            let result = rt
                .block_on(
                    db.admin()
                        .consolidate()
                        .topic_threshold(0.5)
                        .archive(true)
                        .execute(),
                )
                .unwrap();
            black_box(&result);
        });
    });
}

criterion_group!(
    benches,
    bench_consolidation,
    bench_consolidation_with_config
);
criterion_main!(benches);
