use std::sync::Arc;

use parking_lot::RwLock;
use tokio::sync::broadcast;

use crate::event::MemoryEvent;
use crate::event_log::EventLog;

/// In-memory broadcast channel capacity.
///
/// At the 30 K+ writes/sec target a 4096-slot ring buffer provides ~137 ms of
/// headroom before slow subscribers start lagging. Increase via
/// `HIRN_EVENT_BROADCAST_CAPACITY` environment variable if WATCH consumers are
/// slow.  Lagging subscribers receive a
/// [`tokio::sync::broadcast::error::RecvError::Lagged`] error and skip missed
/// events rather than blocking the write path.
const BROADCAST_CAPACITY: usize = 4096;

pub(crate) struct EventRuntime {
    /// Broadcast sender shared by all active subscribers.  Creating a new
    /// subscriber is O(1) and lock-free (`broadcast::Sender::subscribe()`).
    tx: broadcast::Sender<MemoryEvent>,
    event_log: RwLock<Option<Arc<EventLog>>>,
}

impl EventRuntime {
    pub(crate) fn new() -> Self {
        let (tx, _) = broadcast::channel(BROADCAST_CAPACITY);
        Self {
            tx,
            event_log: RwLock::new(None),
        }
    }

    pub(crate) fn set_event_log(&self, log: Arc<EventLog>) {
        *self.event_log.write() = Some(log);
    }

    pub(crate) fn event_log(&self) -> Option<Arc<EventLog>> {
        self.event_log.read().clone()
    }

    /// Subscribe to in-memory real-time event delivery.
    ///
    /// Each call returns an independent `Receiver<MemoryEvent>`.  The
    /// broadcast ring buffer automatically drops lagging receivers rather
    /// than blocking the write path — consume events promptly or increase
    /// `BROADCAST_CAPACITY`.
    ///
    /// # No `async` required
    ///
    /// Unlike the old `std::sync::mpsc`-based API, subscribing is now
    /// synchronous and never blocks a Tokio worker thread.
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<MemoryEvent> {
        self.tx.subscribe()
    }

    /// Broadcast `event` to all live subscribers (zero allocation, zero locks).
    ///
    /// Returns silently when no subscribers are active (normal during startup).
    fn notify_subscribers(&self, event: &MemoryEvent) {
        // broadcast::Sender::send returns Err only when there are zero
        // receivers; that is expected and safe to ignore.
        let _ = self.tx.send(event.clone());
    }

    /// Append to the durable event log then notify subscribers.
    ///
    /// Subscribers are only notified **after** a successful durable append so
    /// that observers never see phantom events that did not survive a crash.
    /// Non-persisted events (e.g. `MemoryRecalled`) skip the log append and
    /// are broadcast directly.
    pub(crate) async fn emit_checked(
        &self,
        realm: &str,
        namespace: &str,
        agent_id: &str,
        event: MemoryEvent,
    ) -> hirn_core::HirnResult<()> {
        if event.should_persist() {
            if let Some(log) = self.event_log() {
                // Durable append FIRST; only notify subscribers on success.
                // This prevents phantom events under storage degradation.
                log.append(realm, namespace, agent_id, event.clone())
                    .await?;
            }
        }

        self.notify_subscribers(&event);
        Ok(())
    }

    /// Batch variant of [`emit_checked`].  All durable events are appended in
    /// a single batch write before any subscriber notification fires.
    pub(crate) async fn emit_checked_batch(
        &self,
        realm: &str,
        namespace: &str,
        agent_id: &str,
        events: Vec<MemoryEvent>,
    ) -> hirn_core::HirnResult<()> {
        if events.is_empty() {
            return Ok(());
        }

        if let Some(log) = self.event_log() {
            let durable: Vec<MemoryEvent> = events
                .iter()
                .filter(|e| e.should_persist())
                .cloned()
                .collect();
            if !durable.is_empty() {
                // Single batch append — only broadcast after the write succeeds.
                log.append_batch(realm, namespace, agent_id, durable)
                    .await?;
            }
        }

        for event in &events {
            self.notify_subscribers(event);
        }
        Ok(())
    }

    /// Infallible emit — logs a warning on durable-append failure but does
    /// **not** notify subscribers when the append fails, preserving the
    /// atomicity guarantee: observers only see events backed by durable storage.
    pub(crate) async fn emit(
        &self,
        realm: &str,
        namespace: &str,
        agent_id: &str,
        event: MemoryEvent,
    ) {
        if let Err(error) = self.emit_checked(realm, namespace, agent_id, event).await {
            tracing::warn!(error = %error, "event log append failed — event NOT broadcast to subscribers");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::RecordBatch;
    use async_trait::async_trait;
    use datafusion::catalog::TableProvider;
    use hirn_core::id::MemoryId;
    use hirn_storage::HirnDbError;
    use hirn_storage::PhysicalStore;
    use hirn_storage::datasets::events::DATASET_NAME as EVENTS_DATASET_NAME;
    use hirn_storage::memory_store::MemoryStore;
    use hirn_storage::store::{
        ColumnTransform, CompactOptions, CompactResult, DatasetInfo, FtsSearchOptions,
        HybridSearchOptions, IndexConfig, MultivectorSearchOptions, ScanOptions,
        VectorSearchOptions, VersionTag,
    };

    struct RejectEventAppendStore {
        inner: MemoryStore,
    }

    #[async_trait]
    impl PhysicalStore for RejectEventAppendStore {
        async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
            if dataset == EVENTS_DATASET_NAME {
                return Err(HirnDbError::Unsupported(
                    "simulated event log append failure".to_string(),
                ));
            }
            self.inner.append(dataset, batch).await
        }

        async fn append_batches(
            &self,
            dataset: &str,
            batches: Vec<RecordBatch>,
        ) -> Result<(), HirnDbError> {
            for batch in batches {
                self.append(dataset, batch).await?;
            }
            Ok(())
        }

        async fn scan(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.scan(dataset, opts).await
        }

        async fn scan_stream(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<hirn_storage::store::RecordBatchStream, HirnDbError> {
            self.inner.scan_stream(dataset, opts).await
        }

        async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError> {
            self.inner.delete(dataset, predicate).await
        }

        async fn update_where(
            &self,
            dataset: &str,
            filter: &str,
            updates: &[(&str, &str)],
        ) -> Result<u64, HirnDbError> {
            self.inner.update_where(dataset, filter, updates).await
        }

        async fn merge_insert(
            &self,
            dataset: &str,
            on: &[&str],
            batch: RecordBatch,
        ) -> Result<(), HirnDbError> {
            self.inner.merge_insert(dataset, on, batch).await
        }

        async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError> {
            self.inner.count(dataset, filter).await
        }

        async fn vector_search(
            &self,
            dataset: &str,
            opts: VectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.vector_search(dataset, opts).await
        }

        async fn vector_search_many(
            &self,
            dataset: &str,
            queries: Vec<VectorSearchOptions>,
        ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
            self.inner.vector_search_many(dataset, queries).await
        }

        async fn fts_search(
            &self,
            dataset: &str,
            opts: FtsSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.fts_search(dataset, opts).await
        }

        async fn hybrid_search(
            &self,
            dataset: &str,
            opts: HybridSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.hybrid_search(dataset, opts).await
        }

        async fn multivector_search(
            &self,
            dataset: &str,
            opts: MultivectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.multivector_search(dataset, opts).await
        }

        async fn create_index(
            &self,
            dataset: &str,
            config: IndexConfig,
        ) -> Result<(), HirnDbError> {
            self.inner.create_index(dataset, config).await
        }

        async fn optimize_indices(&self, dataset: &str) -> Result<(), HirnDbError> {
            self.inner.optimize_indices(dataset).await
        }

        async fn compact(
            &self,
            dataset: &str,
            opts: CompactOptions,
        ) -> Result<CompactResult, HirnDbError> {
            self.inner.compact(dataset, opts).await
        }

        async fn version(&self, dataset: &str) -> Result<u64, HirnDbError> {
            self.inner.version(dataset).await
        }

        async fn tag(&self, dataset: &str, tag: &str) -> Result<(), HirnDbError> {
            self.inner.tag(dataset, tag).await
        }

        async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
            self.inner.checkout(dataset, version).await
        }

        async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError> {
            self.inner.list_tags(dataset).await
        }

        async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError> {
            self.inner.list_datasets().await
        }

        async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError> {
            self.inner.exists(dataset).await
        }

        async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError> {
            self.inner.list_namespaces().await
        }

        async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.create_namespace(name).await
        }

        async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.drop_namespace(name).await
        }

        async fn add_columns(
            &self,
            dataset: &str,
            transforms: Vec<ColumnTransform>,
        ) -> Result<(), HirnDbError> {
            self.inner.add_columns(dataset, transforms).await
        }

        async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError> {
            self.inner.drop_columns(dataset, columns).await
        }

        async fn table_provider(&self, dataset: &str) -> Option<Arc<dyn TableProvider>> {
            self.inner.table_provider(dataset).await
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn emit_reaches_live_subscribers() {
        let runtime = EventRuntime::new();
        let mut receiver = runtime.subscribe();
        let id = MemoryId::new();

        runtime
            .emit("default", "shared", "", MemoryEvent::Forgotten { id })
            .await;

        let event = receiver
            .recv()
            .await
            .expect("subscriber should receive event");
        assert!(matches!(event, MemoryEvent::Forgotten { id: eid } if eid == id));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn emit_with_event_log_reaches_live_subscribers_after_append() {
        let runtime = EventRuntime::new();
        let log = Arc::new(EventLog::open(Arc::new(MemoryStore::new())).await.unwrap());
        runtime.set_event_log(Arc::clone(&log));

        let mut receiver = runtime.subscribe();
        let id = MemoryId::new();

        runtime
            .emit("default", "shared", "", MemoryEvent::Forgotten { id })
            .await;

        let event = receiver
            .recv()
            .await
            .expect("subscriber should receive event");
        assert!(matches!(event, MemoryEvent::Forgotten { id: eid } if eid == id));
        assert_eq!(log.next_seq(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn emit_memory_recalled_notifies_subscribers_without_persisting() {
        let runtime = EventRuntime::new();
        let log = Arc::new(EventLog::open(Arc::new(MemoryStore::new())).await.unwrap());
        runtime.set_event_log(Arc::clone(&log));

        let mut receiver = runtime.subscribe();

        runtime
            .emit(
                "default",
                "shared",
                "",
                MemoryEvent::MemoryRecalled {
                    query_preview: "where is aurora".to_string(),
                    results_count: 3,
                },
            )
            .await;

        let event = receiver
            .recv()
            .await
            .expect("subscriber should receive recall event");
        assert!(matches!(
            event,
            MemoryEvent::MemoryRecalled {
                query_preview,
                results_count: 3,
            } if query_preview == "where is aurora"
        ));
        assert_eq!(log.next_seq(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn emit_does_not_notify_subscribers_when_event_log_append_fails() {
        let runtime = EventRuntime::new();
        let failing_store = Arc::new(RejectEventAppendStore {
            inner: MemoryStore::new(),
        });
        let log = Arc::new(EventLog::open(failing_store).await.unwrap());
        runtime.set_event_log(log);

        let mut receiver = runtime.subscribe();
        let id = MemoryId::new();

        runtime
            .emit("default", "shared", "", MemoryEvent::Forgotten { id })
            .await;

        // Verify no event was broadcast within a short deadline.
        let result =
            tokio::time::timeout(std::time::Duration::from_millis(50), receiver.recv()).await;
        assert!(
            result.is_err(),
            "subscriber should not receive event after failed append"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn emit_checked_batch_with_event_log_reaches_live_subscribers_after_append() {
        let runtime = EventRuntime::new();
        let log = Arc::new(EventLog::open(Arc::new(MemoryStore::new())).await.unwrap());
        runtime.set_event_log(Arc::clone(&log));

        let mut receiver = runtime.subscribe();
        let first = MemoryId::new();
        let second = MemoryId::new();

        runtime
            .emit_checked_batch(
                "default",
                "shared",
                "",
                vec![
                    MemoryEvent::Forgotten { id: first },
                    MemoryEvent::Forgotten { id: second },
                ],
            )
            .await
            .unwrap();

        let first_event = receiver
            .recv()
            .await
            .expect("subscriber should receive first event");
        let second_event = receiver
            .recv()
            .await
            .expect("subscriber should receive second event");
        assert!(matches!(first_event, MemoryEvent::Forgotten { id } if id == first));
        assert!(matches!(second_event, MemoryEvent::Forgotten { id } if id == second));
        assert_eq!(log.next_seq(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn emit_checked_batch_does_not_notify_subscribers_when_event_log_append_fails() {
        let runtime = EventRuntime::new();
        let failing_store = Arc::new(RejectEventAppendStore {
            inner: MemoryStore::new(),
        });
        let log = Arc::new(EventLog::open(failing_store).await.unwrap());
        runtime.set_event_log(log);

        let mut receiver = runtime.subscribe();

        let error = runtime
            .emit_checked_batch(
                "default",
                "shared",
                "",
                vec![MemoryEvent::Forgotten {
                    id: MemoryId::new(),
                }],
            )
            .await;
        assert!(error.is_err());

        let result =
            tokio::time::timeout(std::time::Duration::from_millis(50), receiver.recv()).await;
        assert!(
            result.is_err(),
            "subscriber should not receive event after failed append"
        );
    }
}
