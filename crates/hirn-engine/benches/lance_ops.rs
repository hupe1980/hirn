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
        let content = format!("Record {i}: deployment strategy for service-{}", i % 50);
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

fn bench_lance_write(c: &mut Criterion) {
    let rt = rt();
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

    let mut i = 0u64;
    c.bench_function("lance_write_episodic", |b| {
        b.iter(|| {
            let content = format!("Write bench record {i}");
            let emb = pseudo_embedding(&content, dims);
            let rec = EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(&content)
                .summary(format!("w{i}"))
                .importance(0.5)
                .agent_id(agent())
                .embedding(emb)
                .build()
                .unwrap();
            black_box(rt.block_on(db.episodic().remember(rec)).unwrap());
            i += 1;
        });
    });
}

fn bench_lance_read(c: &mut Criterion) {
    let rt = rt();
    let (db, _dir) = build_db(1000, &rt);

    c.bench_function("lance_read_recall_top10", |b| {
        let dims = db.embedding_dims();
        let emb = pseudo_embedding("deployment strategies", dims);
        b.iter(|| {
            let results = rt
                .block_on(db.recall_view().query(emb.clone()).limit(10).execute())
                .unwrap();
            black_box(&results);
        });
    });

    c.bench_function("lance_read_stats", |b| {
        b.iter(|| {
            let stats = rt.block_on(db.admin().stats()).unwrap();
            black_box(&stats);
        });
    });
}

/// Benchmark LanceHybridSearchExec against an in-memory storage backend.
fn bench_hybrid_search_exec(c: &mut Criterion) {
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_plan::ExecutionPlan;
    use datafusion::prelude::SessionContext;
    use futures::StreamExt;
    use hirn_exec::HirnSessionExt;
    use hirn_exec::operators::LanceHybridSearchExec;
    use hirn_exec::operators::lance_hybrid_search::HybridSearchParams;
    use hirn_storage::PhysicalStore;
    use hirn_storage::datasets::episodic;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("full_content", DataType::Utf8, false),
        Field::new("layer", DataType::Utf8, false),
        Field::new("namespace", DataType::Utf8, false),
        Field::new("score", DataType::Float32, true),
        Field::new("temporal_ms", DataType::Int64, false),
        Field::new("created_at_ms", DataType::Int64, false),
        Field::new("importance", DataType::Float32, true),
        Field::new("access_count", DataType::UInt32, true),
        Field::new("surprise", DataType::Float32, true),
        Field::new("evidence_count", DataType::UInt32, true),
        Field::new("invocation_count", DataType::UInt64, true),
    ]));

    let params = HybridSearchParams {
        datasets: vec!["episodic".into()],
        vector_column: "embedding".into(),
        query_vector: vec![0.1; 32],
        hybrid_mode: false,
        fts_columns: vec!["content".into()],
        fts_query: "deployment".into(),
        limit: 100,
        metric: hirn_storage::store::DistanceMetric::L2,
        filter: None,
        numeric_filters: Vec::new(),
        temporal_start_ms: None,
        temporal_end_ms: None,
        temporal_expansion: false,
        temporal_boost: 1.25,
    };

    let n = 10_000;
    let rt = rt();
    let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
    let records = (0..n)
        .map(|i| {
            let content = format!("Record {i}: deployment strategy for service-{}", i % 50);
            EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(&content)
                .summary(format!("ep {i}"))
                .importance((i as f32 % 7.0).mul_add(0.1, 0.3))
                .agent_id(agent())
                .embedding(pseudo_embedding(&content, 32))
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>();
    rt.block_on(async {
        storage
            .append(
                episodic::DATASET_NAME,
                episodic::to_batch(&records, 32).unwrap(),
            )
            .await
            .unwrap();
    });

    let ctx = SessionContext::new();
    HirnSessionExt::new(Arc::new(0_u8), Arc::new(HirnConfig::default()), None)
        .with_storage(Arc::clone(&storage))
        .register(&ctx)
        .unwrap();

    c.bench_function("hybrid_search_exec_10k_rows", |b| {
        b.iter(|| {
            let exec = LanceHybridSearchExec::new(schema.clone(), params.clone());
            let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
            rt.block_on(async {
                while let Some(result) = stream.next().await {
                    let batch = result.unwrap();
                    black_box(&batch);
                }
            });
        });
    });
}

criterion_group!(
    benches,
    bench_lance_write,
    bench_lance_read,
    bench_hybrid_search_exec
);
criterion_main!(benches);
