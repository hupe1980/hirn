use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use hirn_core::HirnConfig;
use hirn_core::MemoryId;
use hirn_core::semantic::SemanticRecord;
use hirn_core::types::{AgentId, KnowledgeType, Namespace};
use hirn_engine::{HirnDB, SemanticUpdate};
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

const DEFAULT_CHAIN_COUNT: usize = 128;
const DEFAULT_REVISION_COUNT: usize = 4;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn chain_count() -> usize {
    env_usize("HIRN_BENCH_CHAIN_COUNT", DEFAULT_CHAIN_COUNT)
}

fn revision_count() -> usize {
    env_usize("HIRN_BENCH_REVISION_COUNT", DEFAULT_REVISION_COUNT)
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn agent() -> AgentId {
    AgentId::new("bench").unwrap()
}

fn semantic_record(concept: &str, description: &str) -> SemanticRecord {
    SemanticRecord::builder()
        .concept(concept)
        .knowledge_type(KnowledgeType::Propositional)
        .description(description)
        .agent_id(agent())
        .build()
        .unwrap()
}

fn build_db(
    chain_count: usize,
    revision_count: usize,
    rt: &tokio::runtime::Runtime,
) -> (HirnDB, tempfile::TempDir, Vec<String>, Vec<MemoryId>) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("bench");
    let lance_path = dir.path().join("lance_brain");
    let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
    let backend: Arc<dyn PhysicalStore> = rt
        .block_on(HirnDb::open(storage_config))
        .unwrap()
        .store_arc();

    let config = HirnConfig::builder().db_path(&db_path).build().unwrap();
    let db = rt
        .block_on(HirnDB::open_with_config(config, backend))
        .unwrap();

    let mut concepts = Vec::with_capacity(chain_count);
    let mut head_ids = Vec::with_capacity(chain_count);

    for idx in 0..chain_count {
        let concept = format!("revision-topic-{idx}");
        let mut head_id = rt
            .block_on(db.semantic().store(semantic_record(&concept, "revision 1")))
            .unwrap();

        for revision in 2..=revision_count {
            let mut update = SemanticUpdate::with_metadata(agent(), MemoryId::new());
            update.description = Some(format!("revision {revision}"));
            update.reason = Some("benchmark revision chain".into());
            head_id = rt
                .block_on(db.semantic().correct(head_id, update))
                .unwrap()
                .id;
        }

        concepts.push(concept);
        head_ids.push(head_id);
    }

    (db, dir, concepts, head_ids)
}

fn bench_semantic_revision_lookups(c: &mut Criterion) {
    let rt = rt();
    let chain_count = chain_count();
    let revision_count = revision_count();
    let (db, _dir, concepts, head_ids) = build_db(chain_count, revision_count, &rt);
    let namespace = Namespace::default();
    let concept = concepts[chain_count / 2].clone();
    let head_id = head_ids[chain_count / 2];

    c.bench_function("semantic_revision_current_state_lookup", |b| {
        b.iter(|| {
            let record = rt
                .block_on(db.semantic().get_by_concept_ns(&concept, &namespace))
                .unwrap();
            black_box(record);
        });
    });

    c.bench_function("semantic_revision_history_lookup", |b| {
        b.iter(|| {
            let history = rt.block_on(db.semantic().history(head_id)).unwrap();
            black_box(history);
        });
    });
}

fn bench_semantic_revision_storage_overhead(c: &mut Criterion) {
    let rt = rt();
    let chain_count = chain_count();
    let revision_count = revision_count();
    let (base_db, _base_dir, _, _) = build_db(chain_count, 1, &rt);
    let base_stats = rt.block_on(base_db.admin().stats()).unwrap();

    let (revision_db, _revision_dir, _, _) = build_db(chain_count, revision_count, &rt);
    let revision_stats = rt.block_on(revision_db.admin().stats()).unwrap();
    let overhead_bytes = revision_stats
        .file_size_bytes
        .saturating_sub(base_stats.file_size_bytes);
    let overhead_factor = if base_stats.file_size_bytes == 0 {
        0.0
    } else {
        revision_stats.file_size_bytes as f64 / base_stats.file_size_bytes as f64
    };

    eprintln!(
        "semantic_revision_storage_overhead chain_count={} revision_count={} baseline_bytes={} revision_bytes={} overhead_bytes={} factor={:.2}",
        chain_count,
        revision_count,
        base_stats.file_size_bytes,
        revision_stats.file_size_bytes,
        overhead_bytes,
        overhead_factor,
    );

    c.bench_function("semantic_revision_storage_overhead_snapshot", |b| {
        b.iter(|| black_box((overhead_bytes, overhead_factor)));
    });
}

criterion_group!(
    benches,
    bench_semantic_revision_lookups,
    bench_semantic_revision_storage_overhead
);
criterion_main!(benches);
