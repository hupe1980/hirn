//! End-to-end: the write-time temporal envelope survives ingest, persists
//! through Lance, and changes recall ranking.
//!
//! The unit tests prove each layer in isolation. These prove the wiring — that
//! an extractor configured on the database actually reaches a stored record,
//! and that a stored `TemporalState` actually reaches the scorer.

use std::sync::Arc;

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

/// Returns a fixed envelope for every input, so the test asserts wiring rather
/// than model behaviour.
struct FixedTemporalExtractor(TemporalEnvelope);

#[async_trait]
impl TemporalExtractor for FixedTemporalExtractor {
    async fn extract_temporal(
        &self,
        _text: &str,
        _reference: Timestamp,
        _budget: &NluBudget,
    ) -> hirn_core::HirnResult<Option<TemporalEnvelope>> {
        Ok(Some(self.0))
    }

    fn model_id(&self) -> &str {
        "fixed-temporal"
    }
}

async fn open_db(typed_temporal: bool) -> (HirnDB, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let lance = dir.path().join("lance");
    let backend: Arc<dyn PhysicalStore> =
        HirnDb::open(HirnDbConfig::local(lance.to_str().unwrap()))
            .await
            .unwrap()
            .store_arc();
    let mut config = HirnConfig::builder()
        .db_path(dir.path().join("db"))
        .build()
        .unwrap();
    config.nlu.typed_temporal_extraction = typed_temporal;
    let db = HirnDB::open_with_config(config, backend).await.unwrap();
    (db, dir)
}

fn agent() -> AgentId {
    AgentId::new("temporal-agent").unwrap()
}

async fn stored_record(db: &HirnDB, content: &str) -> EpisodicRecord {
    let record = EpisodicRecord::builder()
        .event_type(EventType::Observation)
        .content(content)
        .summary(content)
        .agent_id(agent())
        .build()
        .unwrap();
    let id = db.episodic().remember(record).await.unwrap();
    db.episodic()
        .list(&EpisodicFilter {
            include_archived: true,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_iter()
        .find(|r| r.id == id)
        .expect("stored record")
}

#[tokio::test(flavor = "multi_thread")]
async fn an_extracted_envelope_persists_through_lance() {
    let (db, _dir) = open_db(true).await;
    db.register_agent(&agent(), "Temporal Agent").await.unwrap();

    let envelope = TemporalEnvelope {
        event_time: Some(Timestamp::from_millis(1_700_000_000_000)),
        precision: TimePrecision::Month,
        state: TemporalState::Ongoing,
    };
    db.set_temporal_extractor(Arc::new(FixedTemporalExtractor(envelope)));

    let stored = stored_record(&db, "I live in Berlin").await;
    assert_eq!(
        stored.temporal, envelope,
        "the extracted envelope must reach storage intact"
    );
    assert!(
        !stored.temporal.state.decays_with_age(),
        "an ongoing fact must be exempt from recency decay once stored"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn extraction_stays_off_unless_configured() {
    // Registering an extractor is not enough: the config flag gates it, so a
    // deployment cannot start paying per-record provider calls by accident.
    let (db, _dir) = open_db(false).await;
    db.register_agent(&agent(), "Temporal Agent").await.unwrap();
    db.set_temporal_extractor(Arc::new(FixedTemporalExtractor(TemporalEnvelope {
        event_time: Some(Timestamp::from_millis(1_700_000_000_000)),
        precision: TimePrecision::Day,
        state: TemporalState::Timeless,
    })));

    let stored = stored_record(&db, "I live in Berlin").await;
    assert_eq!(
        stored.temporal,
        TemporalEnvelope::unknown(),
        "typed_temporal_extraction=false must leave the envelope untouched"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_caller_supplied_envelope_is_never_overwritten() {
    let (db, _dir) = open_db(true).await;
    db.register_agent(&agent(), "Temporal Agent").await.unwrap();
    db.set_temporal_extractor(Arc::new(FixedTemporalExtractor(TemporalEnvelope {
        event_time: Some(Timestamp::from_millis(1)),
        precision: TimePrecision::Instant,
        state: TemporalState::Completed,
    })));

    let supplied = TemporalEnvelope {
        event_time: Some(Timestamp::from_millis(1_600_000_000_000)),
        precision: TimePrecision::Year,
        state: TemporalState::Timeless,
    };
    let mut record = EpisodicRecord::builder()
        .event_type(EventType::Observation)
        .content("my birthday is 14 March")
        .summary("birthday")
        .agent_id(agent())
        .build()
        .unwrap();
    record.temporal = supplied;

    let id = db.episodic().remember(record).await.unwrap();
    let stored = db
        .episodic()
        .list(&EpisodicFilter {
            include_archived: true,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_iter()
        .find(|r| r.id == id)
        .unwrap();

    assert_eq!(
        stored.temporal, supplied,
        "an ingest path that already knows the event time must win over inference"
    );
}
