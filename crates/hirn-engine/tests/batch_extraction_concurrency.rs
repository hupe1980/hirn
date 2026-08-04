//! Batch ingest must run typed extraction concurrently, in bounded fashion,
//! without reordering records.
//!
//! Sequential extraction is correct but unusable at corpus scale: one provider
//! round-trip per record means a 10k-record ingest spends over an hour in
//! provider latency. Concurrency introduces two hazards this pins down —
//! records coming back out of order (downstream stages index positionally),
//! and unbounded fan-out against the provider.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use hirn_core::HirnConfig;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::nlu::NluBudget;
use hirn_core::temporal::{TemporalEnvelope, TemporalState, TimePrecision};
use hirn_core::timestamp::Timestamp;
use hirn_core::types::{AgentId, EventType};
use hirn_engine::{EpisodicFilter, HirnDB};
use hirn_provider::TemporalExtractor;
use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

/// Records observed concurrency and encodes the input text into the returned
/// envelope, so ordering can be checked at the far end.
struct TracingExtractor {
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    calls: AtomicUsize,
}

#[async_trait]
impl TemporalExtractor for TracingExtractor {
    async fn extract_temporal(
        &self,
        text: &str,
        _reference: Timestamp,
        _budget: &NluBudget,
    ) -> hirn_core::HirnResult<Option<TemporalEnvelope>> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        self.calls.fetch_add(1, Ordering::SeqCst);

        // Long enough that a sequential implementation cannot overlap calls.
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Encode the record's index into event_time so ordering is verifiable.
        let index: u64 = text
            .rsplit('-')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(Some(TemporalEnvelope {
            event_time: Some(Timestamp::from_millis(1_700_000_000_000 + index)),
            precision: TimePrecision::Day,
            state: TemporalState::Completed,
        }))
    }

    fn model_id(&self) -> &str {
        "tracing-temporal"
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn batch_extraction_is_concurrent_bounded_and_order_preserving() {
    let dir = tempfile::tempdir().unwrap();
    let lance = dir.path().join("lance");
    let backend: Arc<dyn PhysicalStore> =
        HirnDb::open(HirnDbConfig::local(lance.to_str().unwrap()))
            .await
            .unwrap()
            .store_arc();

    let concurrency = 4;
    let mut config = HirnConfig::builder()
        .db_path(dir.path().join("db"))
        .build()
        .unwrap();
    config.nlu.typed_temporal_extraction = true;
    config.nlu.extraction_concurrency = concurrency;

    let db = HirnDB::open_with_config(config, backend).await.unwrap();
    let agent = AgentId::new("batch-agent").unwrap();
    db.register_agent(&agent, "Batch Agent").await.unwrap();

    let extractor = Arc::new(TracingExtractor {
        in_flight: AtomicUsize::new(0),
        peak: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
    });
    db.set_temporal_extractor(extractor.clone());

    let count = 12;
    let records: Vec<EpisodicRecord> = (0..count)
        .map(|i| {
            EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(format!("batch record-{i}"))
                .summary(format!("summary-{i}"))
                .agent_id(agent)
                .build()
                .unwrap()
        })
        .collect();

    let started = std::time::Instant::now();
    let results = db.episodic().batch_remember(records).await;
    let elapsed = started.elapsed();

    let ids: Vec<_> = results.into_iter().map(|r| r.unwrap()).collect();
    assert_eq!(ids.len(), count);
    assert_eq!(extractor.calls.load(Ordering::SeqCst), count);

    // Concurrency actually happened, and stayed within the configured bound.
    let peak = extractor.peak.load(Ordering::SeqCst);
    assert!(
        peak > 1,
        "extraction ran sequentially (peak in-flight = {peak})"
    );
    assert!(
        peak <= concurrency,
        "extraction exceeded the configured bound: {peak} > {concurrency}"
    );

    // Sequential would be >= 12 * 60ms = 720ms; concurrent should be far less.
    assert!(
        elapsed < Duration::from_millis(600),
        "batch took {elapsed:?}, consistent with sequential extraction"
    );

    // Order preserved: record i must carry the envelope encoding i. Downstream
    // stages index `records` positionally, so a reorder here would silently
    // attach every envelope to the wrong memory.
    let stored = db
        .episodic()
        .list(&EpisodicFilter {
            include_archived: true,
            ..Default::default()
        })
        .await
        .unwrap();
    for (i, id) in ids.iter().enumerate() {
        let record = stored.iter().find(|r| r.id == *id).expect("stored record");
        assert!(
            record.content.ends_with(&format!("record-{i}")),
            "returned ids must follow input order"
        );
        assert_eq!(
            record.temporal.event_time,
            Some(Timestamp::from_millis(1_700_000_000_000 + i as u64)),
            "record {i} carries another record's envelope"
        );
    }
}
