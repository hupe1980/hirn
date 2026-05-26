use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use hirn_core::id::MemoryId;
use hirn_engine::{EventLog, MemoryEvent};
use hirn_storage::memory_store::MemoryStore;

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn make_event(i: usize) -> MemoryEvent {
    MemoryEvent::EpisodeCreated {
        id: MemoryId::new(),
        content_preview: format!("event-{i}"),
    }
}

/// Benchmark: publish 100K events via append_batch (seq assignment + storage + broadcast).
fn bench_event_bus_100k(c: &mut Criterion) {
    let runtime = rt();

    c.bench_function("event_bus_100k_append_batch", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let store = Arc::new(MemoryStore::new());
                let log = EventLog::open(store).await.unwrap();

                let events: Vec<MemoryEvent> = (0..100_000).map(|i| make_event(i)).collect();

                let result = log
                    .append_batch("bench", "default", "bench-agent", events)
                    .await
                    .unwrap();

                black_box(result.len())
            })
        });
    });
}

/// Benchmark: publish 100K events one-by-one (worst case — per-event storage write).
fn bench_event_bus_100k_individual(c: &mut Criterion) {
    let runtime = rt();

    let mut group = c.benchmark_group("event_bus_individual");
    group.sample_size(10); // Single-event append is slow (100K writes).

    group.bench_function("event_bus_100k_individual", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let store = Arc::new(MemoryStore::new());
                let log = EventLog::open(store).await.unwrap();

                for i in 0..100_000u64 {
                    log.append("bench", "default", "bench-agent", make_event(i as usize))
                        .await
                        .unwrap();
                }

                black_box(log.next_seq())
            })
        });
    });

    group.finish();
}

/// Benchmark: broadcast throughput with active subscriber.
fn bench_event_bus_100k_with_subscriber(c: &mut Criterion) {
    let runtime = rt();

    c.bench_function("event_bus_100k_batch_with_subscriber", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let store = Arc::new(MemoryStore::new());
                let log = EventLog::open(store).await.unwrap();

                // Subscribe to create backpressure.
                let _rx = log.subscribe();

                let events: Vec<MemoryEvent> = (0..100_000).map(|i| make_event(i)).collect();

                let result = log
                    .append_batch("bench", "default", "bench-agent", events)
                    .await
                    .unwrap();

                black_box(result.len())
            })
        });
    });
}

criterion_group!(
    benches,
    bench_event_bus_100k,
    bench_event_bus_100k_individual,
    bench_event_bus_100k_with_subscriber,
);
criterion_main!(benches);
