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

    (db, dir)
}

// ─── Scaling: Recall (vector, top-10) latency ────────────────

fn bench_scaling_recall(c: &mut Criterion) {
    let rt = rt();
    let mut group = c.benchmark_group("scaling_recall");
    group.sample_size(10);

    for &n in &[1_000, 10_000] {
        let (db, _dir) = build_db(n, &rt);
        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for services", dims);

        group.bench_function(format!("recall_top10_{n}"), |b| {
            b.iter(|| {
                let results = rt
                    .block_on(db.recall_view().query(emb.clone()).limit(10).execute())
                    .unwrap();
                black_box(&results);
            });
        });
    }

    group.finish();
}

// ─── Scaling: Recall (hybrid vector+BM25, top-10) latency ───

fn bench_scaling_recall_hybrid(c: &mut Criterion) {
    let rt = rt();
    let mut group = c.benchmark_group("scaling_recall_hybrid");
    group.sample_size(10);

    for &n in &[1_000, 10_000] {
        let (db, _dir) = build_db(n, &rt);
        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for services", dims);

        group.bench_function(format!("recall_hybrid_top10_{n}"), |b| {
            b.iter(|| {
                let results = rt
                    .block_on(
                        db.recall_view()
                            .query(emb.clone())
                            .query_text("deployment strategy")
                            .limit(10)
                            .execute(),
                    )
                    .unwrap();
                black_box(&results);
            });
        });
    }

    group.finish();
}

// ─── Scaling: Think latency at different DB sizes ────────────

fn bench_scaling_think(c: &mut Criterion) {
    let rt = rt();
    let mut group = c.benchmark_group("scaling_think");
    group.sample_size(10);

    for &n in &[1_000, 10_000] {
        let (db, _dir) = build_db(n, &rt);
        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies for services", dims);

        group.bench_function(format!("think_4096_{n}"), |b| {
            b.iter(|| {
                let result = rt
                    .block_on(db.recall_view().think(emb.clone()).budget(4096).execute())
                    .unwrap();
                black_box(&result.context);
            });
        });
    }

    group.finish();
}

// ─── Scaling: Remember (single insert) throughput ────────────

fn bench_scaling_remember(c: &mut Criterion) {
    let rt = rt();
    let mut group = c.benchmark_group("scaling_remember");
    group.sample_size(10);

    for &n in &[1_000, 10_000] {
        let (db, _dir) = build_db(n, &rt);
        let dims = db.embedding_dims();
        let mut counter = n as u64;

        group.bench_function(format!("remember_single_at_{n}"), |b| {
            b.iter(|| {
                let content = format!("New record after {n}: item-{counter}");
                let emb = pseudo_embedding(&content, dims);
                let rec = EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content(&content)
                    .summary(format!("new {counter}"))
                    .importance(0.5)
                    .agent_id(agent())
                    .embedding(emb)
                    .build()
                    .unwrap();
                black_box(rt.block_on(db.episodic().remember(rec)).unwrap());
                counter += 1;
            });
        });
    }

    group.finish();
}

// ─── Batch remember: insert N records in a tight loop ────────

fn bench_batch_remember(c: &mut Criterion) {
    let rt = rt();
    let mut group = c.benchmark_group("batch_remember");
    group.sample_size(10);

    for &batch in &[100, 1_000] {
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

        group.bench_function(format!("remember_batch_{batch}"), |b| {
            b.iter(|| {
                for i in 0..batch {
                    let content = format!("batch record {i}: info about topic-{}", i % 20);
                    let emb = pseudo_embedding(&content, dims);
                    let rec = EpisodicRecord::builder()
                        .event_type(EventType::Observation)
                        .content(&content)
                        .summary(format!("batch {i}"))
                        .importance(0.5)
                        .agent_id(agent())
                        .embedding(emb)
                        .build()
                        .unwrap();
                    rt.block_on(db.episodic().remember(rec)).unwrap();
                }
            });
        });
    }

    group.finish();
}

// ─── Consolidation benchmark ─────────────────────────────────

fn bench_scaling_consolidate(c: &mut Criterion) {
    let rt = rt();
    let mut group = c.benchmark_group("scaling_consolidate");
    group.sample_size(10);

    for &n in &[200, 1_000] {
        let (db, _dir) = build_db(n, &rt);

        group.bench_function(format!("consolidate_{n}"), |b| {
            b.iter(|| {
                let result = rt.block_on(db.admin().consolidate().execute()).unwrap();
                black_box(&result);
            });
        });
    }

    group.finish();
}

// ─── Memory usage benchmark (measures DB file size as proxy) ─

fn bench_memory_usage(c: &mut Criterion) {
    let rt = rt();
    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(10);

    for &n in &[1_000, 10_000] {
        let (db, _dir) = build_db(n, &rt);

        group.bench_function(format!("stats_at_{n}"), |b| {
            b.iter(|| {
                let stats = rt.block_on(db.admin().stats()).unwrap();
                assert_eq!(stats.episodic_count as usize, n);
                black_box(stats.file_size_bytes);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_scaling_recall,
    bench_scaling_recall_hybrid,
    bench_scaling_think,
    bench_scaling_remember,
    bench_batch_remember,
    bench_scaling_consolidate,
    bench_memory_usage,
);
criterion_main!(benches);
