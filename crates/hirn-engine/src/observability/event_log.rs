//! Append-only event log backed by LanceDB.
//!
//! The [`EventLog`] is the foundation for event sourcing in hirn
//!. Every mutation is appended to the `events` dataset
//! before being materialized, enabling replay, streaming, audit, and
//! time-travel queries.
//!
//! # Architecture
//!
//! Three tiers:
//! 1. `events.lance` — durable, queryable event history (this module)
//! 2. `tokio::sync::broadcast` — real-time in-memory WATCH subscriptions
//! 3. LanceDB table versions/tags — coarse checkpoints (snapshots)

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use futures::TryStreamExt;

use hirn_core::HirnResult;

use hirn_storage::PhysicalStore;
use hirn_storage::datasets::events::{self, DATASET_NAME, EventRow};
use hirn_storage::store::{ScanOptions, ScanOrdering};

use crate::event::{EventEnvelope, MemoryEvent};

/// Filter for reading events from the log.
#[derive(Debug, Default, Clone)]
pub struct EventFilter {
    /// Filter by realm.
    pub realm: Option<String>,
    /// Filter by namespace.
    pub namespace: Option<String>,
    /// Filter by event type string.
    pub event_type: Option<String>,
    /// Filter by agent ID.
    pub agent_id: Option<String>,
    /// Filter events after this timestamp (microseconds, inclusive).
    pub after_us: Option<i64>,
    /// Filter events before this timestamp (microseconds, inclusive).
    pub before_us: Option<i64>,
}

/// Snapshot metadata stored alongside LanceDB tags.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotMeta {
    /// Sequence number at which the snapshot was taken.
    pub seq: u64,
    /// Wall-clock time of snapshot (microseconds).
    pub timestamp_us: i64,
    /// Number of events in the log at snapshot time.
    pub event_count: u64,
}

/// Retention policy for event log compaction.
#[derive(Debug, Clone)]
pub enum RetentionPolicy {
    /// Keep events newer than the last snapshot.
    SnapshotBased,
    /// Keep at most N events; compact the oldest.
    MaxEvents(u64),
    /// Keep events from the last N seconds.
    TimeBased(u64),
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// Number of events removed.
    pub events_removed: u64,
    /// Sequence number up to which events were removed.
    pub compacted_before_seq: u64,
}

/// Append-only event log backed by a LanceDB dataset.
///
/// Thread-safe: the atomic seq counter ensures gap-free sequence numbers
/// from a single writer. For multi-writer (Raft), the leader assigns seqs.
pub struct EventLog {
    storage: Arc<dyn PhysicalStore>,
    /// Next sequence number to assign.
    next_seq: AtomicU64,
    /// Broadcast channel for real-time push to WATCH subscribers.
    tx: tokio::sync::broadcast::Sender<EventEnvelope>,
    /// HMAC secret. When `Some`, the production append paths sign every event
    /// and chain it to the previous one, producing a tamper-evident log.
    hmac_secret: Option<Vec<u8>>,
    /// Tag of the most recently appended event — the head of the hash chain.
    /// Held behind an async mutex so signed appends serialize their
    /// sign-and-chain critical section, keeping the chain gap-free and ordered.
    chain_head: tokio::sync::Mutex<Option<String>>,
}

impl EventLog {
    /// Create a new event log on the given storage backend.
    ///
    /// Scans the existing `events` dataset (if any) to recover the next
    /// sequence number, ensuring gap-free continuation after restart.
    pub async fn open(storage: Arc<dyn PhysicalStore>) -> HirnResult<Self> {
        Self::open_inner(storage, None).await
    }

    /// Open an event log that signs and hash-chains every appended event with
    /// `secret`, making the log tamper-evident. Recovers the chain head from the
    /// existing dataset so the chain continues unbroken across restarts.
    pub async fn open_signed(storage: Arc<dyn PhysicalStore>, secret: Vec<u8>) -> HirnResult<Self> {
        Self::open_inner(storage, Some(secret)).await
    }

    async fn open_inner(
        storage: Arc<dyn PhysicalStore>,
        hmac_secret: Option<Vec<u8>>,
    ) -> HirnResult<Self> {
        let (tx, _) = tokio::sync::broadcast::channel(4096);

        // Recover next seq from existing events.
        let next_seq = Self::recover_next_seq(&*storage).await?;
        // Recover the chain head (hmac of the max-seq event) so signed appends
        // continue the existing chain rather than forking a new one.
        let chain_head = if hmac_secret.is_some() {
            Self::recover_chain_head(&*storage).await?
        } else {
            None
        };

        Ok(Self {
            storage,
            next_seq: AtomicU64::new(next_seq),
            tx,
            hmac_secret,
            chain_head: tokio::sync::Mutex::new(chain_head),
        })
    }

    /// Recover the hmac of the highest-seq event (the current chain head).
    async fn recover_chain_head(storage: &dyn PhysicalStore) -> HirnResult<Option<String>> {
        use arrow_array::Array;
        if !storage.exists(DATASET_NAME).await? {
            return Ok(None);
        }
        if storage.count(DATASET_NAME, None).await? == 0 {
            return Ok(None);
        }
        let mut batches = storage
            .scan_stream(
                DATASET_NAME,
                ScanOptions {
                    columns: Some(vec!["seq".into(), "hmac".into()]),
                    filter: None,
                    exact_filter: None,
                    order_by: Some(vec![ScanOrdering::desc("seq")]),
                    limit: Some(1),
                    offset: None,
                },
            )
            .await?;
        if let Some(batch) = batches.try_next().await? {
            if let Some(col) = batch.column_by_name("hmac") {
                if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                    if arr.len() > 0 && !arr.is_null(0) {
                        return Ok(Some(arr.value(0).to_string()));
                    }
                }
            }
        }
        Ok(None)
    }

    /// Recover the next sequence number by finding the max seq in the dataset.
    async fn recover_next_seq(storage: &dyn PhysicalStore) -> HirnResult<u64> {
        let exists = storage.exists(DATASET_NAME).await?;
        if !exists {
            return Ok(0);
        }

        let count = storage.count(DATASET_NAME, None).await?;
        if count == 0 {
            return Ok(0);
        }

        // Scan for the maximum seq value. We scan just the seq column,
        // sorted by seq descending, limit 1.
        let mut batches = storage
            .scan_stream(
                DATASET_NAME,
                ScanOptions {
                    columns: Some(vec!["seq".into()]),
                    filter: None,
                    exact_filter: None,
                    order_by: Some(vec![ScanOrdering::desc("seq")]),
                    limit: Some(1),
                    offset: None,
                },
            )
            .await?;

        let mut max_seq: u64 = 0;
        while let Some(batch) = batches.try_next().await? {
            if let Some(col) = batch.column_by_name("seq") {
                let arr = col
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>()
                    .ok_or_else(|| {
                        hirn_core::HirnError::storage("event_log seq column is not UInt64")
                    })?;
                for i in 0..arr.len() {
                    if arr.value(i) > max_seq {
                        max_seq = arr.value(i);
                    }
                }
            }
        }

        Ok(max_seq + 1)
    }

    /// Get a broadcast receiver for real-time event subscriptions.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<EventEnvelope> {
        self.tx.subscribe()
    }

    /// Get a filtered receiver that only delivers events matching the filter.
    ///
    /// Spawns a background task that reads from the broadcast channel and
    /// forwards matching events to the returned `mpsc::Receiver`. The task
    /// terminates when the receiver is dropped or the broadcast sender is
    /// closed.
    pub fn subscribe_filtered(
        &self,
        filter: EventFilter,
    ) -> tokio::sync::mpsc::Receiver<EventEnvelope> {
        let mut rx = self.tx.subscribe();
        let (tx, filtered_rx) = tokio::sync::mpsc::channel(256);

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(env) => {
                        if filter_matches(&filter, &env) {
                            if tx.send(env).await.is_err() {
                                break; // receiver dropped
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(skipped = n, "event subscriber lagged, lost events");
                        metrics::counter!("hirn_event_subscriber_lagged_total").increment(n);
                        continue;
                    }
                }
            }
        });

        filtered_rx
    }

    /// Current next sequence number (the number of events appended so far,
    /// if no compaction has occurred).
    pub fn next_seq(&self) -> u64 {
        self.next_seq.load(Ordering::Acquire)
    }

    // ── Event Log Writer ─────────────────────────────────────

    /// Append a single event to the log.
    ///
    /// Assigns a monotonic seq number, writes to LanceDB, and broadcasts
    /// to real-time subscribers.
    pub async fn append(
        &self,
        realm: impl Into<String>,
        namespace: impl Into<String>,
        agent_id: impl Into<String>,
        event: MemoryEvent,
    ) -> HirnResult<EventEnvelope> {
        let envelope = if let Some(secret) = self.hmac_secret.as_deref() {
            // Allocate the seq *inside* the chain-head critical section so that
            // seq order == chain-link order == persist order. Allocating the seq
            // before taking the lock lets two concurrent appends acquire the lock
            // in the opposite order to their seq numbers, which makes each
            // event's `prev_hmac` link to the wrong predecessor and causes
            // `verify_chain` (which walks seq-ascending) to report a false
            // tamper on a log that was never touched. The head advances only
            // after the durable write so a failed write never orphans the chain.
            let mut head = self.chain_head.lock().await;
            let seq = self.next_seq.fetch_add(1, Ordering::AcqRel);
            let mut envelope = EventEnvelope::new(seq, realm, namespace, agent_id, event);
            envelope.sign_chained(secret, head.clone());
            self.persist_one(&envelope).await?;
            head.clone_from(&envelope.hmac);
            envelope
        } else {
            let seq = self.next_seq.fetch_add(1, Ordering::AcqRel);
            let envelope = EventEnvelope::new(seq, realm, namespace, agent_id, event);
            self.persist_one(&envelope).await?;
            envelope
        };

        // Best-effort broadcast (receivers may be lagging — that's OK).
        let _ = self.tx.send(envelope.clone());

        Ok(envelope)
    }

    /// Serialize one envelope to a single-row batch and append it.
    async fn persist_one(&self, envelope: &EventEnvelope) -> HirnResult<()> {
        let row = Self::row_for(envelope)?;
        let batch = events::to_batch(std::slice::from_ref(&row))?;
        self.storage.append(DATASET_NAME, batch).await?;
        Ok(())
    }

    /// Build the storage row for an envelope.
    ///
    /// The payload column is version-prefixed (see `hirn_core::persist`) so a
    /// future change to `MemoryEvent`'s shape becomes an explicit migration
    /// instead of making old rows undecodable. The HMAC canonicalization in
    /// `event.rs` serializes the in-memory event independently, so the stored
    /// encoding is free to evolve without affecting chain verification.
    fn row_for(envelope: &EventEnvelope) -> HirnResult<EventRow> {
        let payload = hirn_core::persist::to_versioned_bytes(&envelope.event)?;
        Ok(EventRow {
            seq: envelope.seq,
            timestamp_us: envelope.timestamp_us,
            realm: envelope.realm.clone(),
            namespace: envelope.namespace.clone(),
            agent_id: envelope.agent_id.clone(),
            event_type: envelope.event_type().to_string(),
            payload,
            hmac: envelope.hmac.clone(),
            prev_hmac: envelope.prev_hmac.clone(),
        })
    }

    /// Append a single event with HMAC signing.
    ///
    /// Same as [`Self::append`] but signs the event envelope with the provided secret
    /// before persisting it. Auditors can later call [`Self::verify_integrity`] to
    /// confirm no events have been tampered with.
    pub async fn append_signed(
        &self,
        event: MemoryEvent,
        realm: impl Into<String>,
        namespace: impl Into<String>,
        agent_id: impl Into<String>,
        secret: &[u8],
    ) -> HirnResult<EventEnvelope> {
        // Chain under the head lock so this explicit-secret path also forms a
        // continuous chain with the surrounding signed appends. The seq is
        // allocated inside the critical section so seq order matches chain order
        // even under concurrent appends (see `append`).
        let mut head = self.chain_head.lock().await;
        let seq = self.next_seq.fetch_add(1, Ordering::AcqRel);
        let mut envelope = EventEnvelope::new(seq, realm, namespace, agent_id, event);
        envelope.sign_chained(secret, head.clone());
        self.persist_one(&envelope).await?;
        head.clone_from(&envelope.hmac);
        drop(head);

        let _ = self.tx.send(envelope.clone());

        Ok(envelope)
    }

    /// Append a batch of events atomically.
    ///
    /// All events get consecutive seq numbers.
    pub async fn append_batch(
        &self,
        realm: &str,
        namespace: &str,
        agent_id: &str,
        events_in: Vec<MemoryEvent>,
    ) -> HirnResult<Vec<EventEnvelope>> {
        if events_in.is_empty() {
            return Ok(vec![]);
        }

        // When signing, hold the chain-head lock for the whole batch so every
        // event chains to its predecessor (across and within the batch). The seq
        // block is allocated *inside* the critical section so seq order matches
        // chain-link order even when other appends race this one (see `append`).
        let mut head_guard = if self.hmac_secret.is_some() {
            Some(self.chain_head.lock().await)
        } else {
            None
        };
        let base_seq = self
            .next_seq
            .fetch_add(events_in.len() as u64, Ordering::AcqRel);

        let mut envelopes = Vec::with_capacity(events_in.len());
        let mut rows = Vec::with_capacity(events_in.len());
        let mut running_prev: Option<String> = head_guard.as_ref().and_then(|g| (**g).clone());

        for (i, event) in events_in.into_iter().enumerate() {
            let seq = base_seq + i as u64;
            let mut envelope = EventEnvelope::new(seq, realm, namespace, agent_id, event);

            if let Some(secret) = self.hmac_secret.as_deref() {
                envelope.sign_chained(secret, running_prev.clone());
                running_prev = envelope.hmac.clone();
            }

            rows.push(Self::row_for(&envelope)?);
            envelopes.push(envelope);
        }

        let batch = events::to_batch(&rows)?;
        self.storage.append(DATASET_NAME, batch).await?;

        // Advance the chain head to the last event's tag only after a durable
        // write, then release the lock.
        if let Some(ref mut head) = head_guard {
            **head = running_prev;
        }
        drop(head_guard);

        // Broadcast all envelopes.
        for env in &envelopes {
            let _ = self.tx.send(env.clone());
        }

        Ok(envelopes)
    }

    // ── Event Log Reader & Replay ────────────────────────────

    /// Read events in a sequence range [from_seq, to_seq] inclusive.
    pub async fn read(&self, from_seq: u64, to_seq: u64) -> HirnResult<Vec<EventEnvelope>> {
        let filter = format!("seq >= {from_seq} AND seq <= {to_seq}");
        self.read_filtered(Some(&filter)).await
    }

    /// Read all events from a sequence number onward.
    pub async fn tail(&self, from_seq: u64) -> HirnResult<Vec<EventEnvelope>> {
        let filter = format!("seq >= {from_seq}");
        self.read_filtered(Some(&filter)).await
    }

    /// Read all events matching an optional filter.
    pub async fn read_all(&self) -> HirnResult<Vec<EventEnvelope>> {
        self.read_filtered(None).await
    }

    /// Read events with an advanced filter.
    pub async fn read_with_filter(&self, filter: &EventFilter) -> HirnResult<Vec<EventEnvelope>> {
        let mut predicates = Vec::new();

        if let Some(ref realm) = filter.realm {
            let escaped = realm.replace('\'', "''");
            predicates.push(format!("realm = '{escaped}'"));
        }
        if let Some(ref ns) = filter.namespace {
            let escaped = ns.replace('\'', "''");
            predicates.push(format!("namespace = '{escaped}'"));
        }
        if let Some(ref et) = filter.event_type {
            let escaped = et.replace('\'', "''");
            predicates.push(format!("event_type = '{escaped}'"));
        }
        if let Some(ref aid) = filter.agent_id {
            let escaped = aid.replace('\'', "''");
            predicates.push(format!("agent_id = '{escaped}'"));
        }
        if let Some(after) = filter.after_us {
            predicates.push(format!("timestamp_us >= {after}"));
        }
        if let Some(before) = filter.before_us {
            predicates.push(format!("timestamp_us <= {before}"));
        }

        let combined = if predicates.is_empty() {
            None
        } else {
            Some(predicates.join(" AND "))
        };

        self.read_filtered(combined.as_deref()).await
    }

    /// Replay all events through a handler function to reconstruct state.
    ///
    /// Events are read in seq order and passed one-by-one to `handler`.
    pub async fn replay<F>(&self, mut handler: F) -> HirnResult<u64>
    where
        F: FnMut(&EventEnvelope) -> HirnResult<()>,
    {
        let envelopes = self.read_all().await?;
        let count = envelopes.len() as u64;
        for env in &envelopes {
            handler(env)?;
        }
        Ok(count)
    }

    /// Verify HMAC integrity of all events in the log.
    ///
    /// Returns the sequence numbers of events whose HMAC validation failed
    /// (missing HMAC or tampered data). An empty vec means all events are valid.
    /// Intended for use by external auditors.
    pub async fn verify_integrity(&self, secret: &[u8]) -> HirnResult<Vec<u64>> {
        let events = self.read_all().await?;
        let failures: Vec<u64> = events
            .iter()
            .filter(|env| !env.verify_hmac(secret))
            .map(|env| env.seq)
            .collect();
        Ok(failures)
    }

    /// Verify the full tamper-evident chain: every event's own HMAC, the
    /// `prev_hmac` linkage between consecutive events, and gap-free `seq`
    /// contiguity. Unlike [`Self::verify_integrity`], this also detects
    /// *deleted* or *truncated* events (a removed event breaks its successor's
    /// linkage or leaves a seq gap), which per-event tags alone cannot catch.
    ///
    /// Returns `Ok(())` if the chain is intact, or an error describing the first
    /// break (bad tag, broken linkage, or seq gap).
    pub async fn verify_chain(&self, secret: &[u8]) -> HirnResult<()> {
        let events = self.read_all().await?;
        let mut prev_seq: Option<u64> = None;
        let mut prev_tag: Option<String> = None;
        for env in &events {
            if let Some(ps) = prev_seq {
                if env.seq != ps + 1 {
                    return Err(hirn_core::HirnError::storage(format!(
                        "audit chain seq gap: {ps} → {} (missing events)",
                        env.seq
                    )));
                }
            }
            if !env.verify_hmac(secret) {
                return Err(hirn_core::HirnError::storage(format!(
                    "audit chain: event seq {} has an invalid or missing HMAC",
                    env.seq
                )));
            }
            if env.prev_hmac != prev_tag {
                return Err(hirn_core::HirnError::storage(format!(
                    "audit chain: event seq {} does not link to its predecessor \
                     (expected prev_hmac {:?}, found {:?})",
                    env.seq, prev_tag, env.prev_hmac
                )));
            }
            prev_seq = Some(env.seq);
            prev_tag.clone_from(&env.hmac);
        }
        Ok(())
    }

    /// Replay events from a specific seq onward.
    pub async fn replay_from<F>(&self, from_seq: u64, mut handler: F) -> HirnResult<u64>
    where
        F: FnMut(&EventEnvelope) -> HirnResult<()>,
    {
        let envelopes = self.tail(from_seq).await?;
        let count = envelopes.len() as u64;
        for env in &envelopes {
            handler(env)?;
        }
        Ok(count)
    }

    /// Internal: read events with an optional SQL filter predicate.
    async fn read_filtered(&self, filter: Option<&str>) -> HirnResult<Vec<EventEnvelope>> {
        self.read_filtered_limited(filter, None).await
    }

    /// Internal: read events with optional filter and limit.
    async fn read_filtered_limited(
        &self,
        filter: Option<&str>,
        limit: Option<usize>,
    ) -> HirnResult<Vec<EventEnvelope>> {
        self.read_filtered_limited_ordered(filter, limit, vec![ScanOrdering::asc("seq")])
            .await
    }

    async fn read_filtered_limited_ordered(
        &self,
        filter: Option<&str>,
        limit: Option<usize>,
        order_by: Vec<ScanOrdering>,
    ) -> HirnResult<Vec<EventEnvelope>> {
        let exists = self.storage.exists(DATASET_NAME).await?;
        if !exists {
            return Ok(vec![]);
        }

        let mut batches = self
            .storage
            .scan_stream(
                DATASET_NAME,
                ScanOptions {
                    columns: None,
                    filter: filter.map(String::from),
                    exact_filter: None,
                    order_by: Some(order_by),
                    limit,
                    offset: None,
                },
            )
            .await?;

        let mut envelopes = Vec::new();
        while let Some(batch) = batches.try_next().await? {
            let rows = events::from_batch(&batch)?;
            for row in rows {
                let event: MemoryEvent = hirn_core::persist::from_versioned_bytes(&row.payload)
                    .map_err(|e| {
                        hirn_core::HirnError::storage(format!(
                            "event deserialize at seq {}: {e}",
                            row.seq
                        ))
                    })?;

                envelopes.push(EventEnvelope {
                    seq: row.seq,
                    timestamp_us: row.timestamp_us,
                    realm: row.realm,
                    namespace: row.namespace,
                    agent_id: row.agent_id,
                    event: event,
                    hmac: row.hmac,
                    prev_hmac: row.prev_hmac,
                });
            }
        }
        Ok(envelopes)
    }

    // ── Snapshots & Compaction ───────────────────────────────

    /// Take a snapshot at the current seq, creating LanceDB tags on
    /// materialized tables.
    ///
    /// Returns the snapshot metadata including the seq at which it was taken.
    pub async fn snapshot(&self, materialized_tables: &[&str]) -> HirnResult<SnapshotMeta> {
        let current_seq = self.next_seq.load(Ordering::Acquire).saturating_sub(1);
        let tag = format!("snapshot-{current_seq}");

        // Tag each materialized table at its current version.
        for table_name in materialized_tables {
            if self.storage.exists(table_name).await? {
                self.storage.tag(table_name, &tag).await?;
            }
        }

        // Log the snapshot event.
        let _ = self
            .append(
                "system",
                "system",
                "system",
                MemoryEvent::SnapshotTaken {
                    seq: current_seq,
                    tag: tag.clone(),
                },
            )
            .await?;

        let event_count = self.storage.count(DATASET_NAME, None).await.unwrap_or(0);

        let meta = SnapshotMeta {
            seq: current_seq,
            timestamp_us: chrono::Utc::now().timestamp_micros(),
            event_count,
        };

        Ok(meta)
    }

    /// Compact (prune) events before the given sequence number.
    ///
    /// Events with `seq < before_seq` are deleted from the events dataset.
    /// Call `optimize` afterward to reclaim storage.
    pub async fn compact(&self, before_seq: u64) -> HirnResult<CompactionResult> {
        let exists = self.storage.exists(DATASET_NAME).await?;
        if !exists {
            return Ok(CompactionResult {
                events_removed: 0,
                compacted_before_seq: before_seq,
            });
        }

        let predicate = format!(
            "seq < {before_seq} AND event_type NOT IN ('access_granted', 'access_denied', 'policy_changed')"
        );
        let deleted = self.storage.delete(DATASET_NAME, &predicate).await?;

        // Optimize: compact + optimize indices.
        self.storage
            .compact(DATASET_NAME, Default::default())
            .await?;
        self.storage.optimize_indices(DATASET_NAME).await?;

        // Log the compaction event.
        let _ = self
            .append(
                "system",
                "system",
                "system",
                MemoryEvent::CompactionCompleted {
                    before_seq,
                    events_removed: deleted,
                },
            )
            .await?;

        Ok(CompactionResult {
            events_removed: deleted,
            compacted_before_seq: before_seq,
        })
    }

    /// Apply a retention policy to compact old events.
    pub async fn apply_retention(&self, policy: &RetentionPolicy) -> HirnResult<CompactionResult> {
        match policy {
            RetentionPolicy::SnapshotBased => {
                let snapshots = self
                    .read_filtered_limited_ordered(
                        Some("event_type = 'snapshot_taken'"),
                        Some(1),
                        vec![ScanOrdering::desc("seq")],
                    )
                    .await?;
                let last_snapshot_seq = snapshots.iter().find_map(|e| {
                    if let MemoryEvent::SnapshotTaken { seq, .. } = &e.event {
                        Some(*seq)
                    } else {
                        None
                    }
                });

                match last_snapshot_seq {
                    Some(seq) => self.compact(seq).await,
                    None => Ok(CompactionResult {
                        events_removed: 0,
                        compacted_before_seq: 0,
                    }),
                }
            }
            RetentionPolicy::MaxEvents(max) => {
                let count = self.storage.count(DATASET_NAME, None).await.unwrap_or(0);
                if count <= *max {
                    return Ok(CompactionResult {
                        events_removed: 0,
                        compacted_before_seq: 0,
                    });
                }
                let to_remove = count - max;
                // Read only the oldest events up to the cutoff point + 1 to find
                // the seq boundary, instead of loading the entire log.
                let cutoff_events = self
                    .read_filtered_limited(None, Some((to_remove + 1) as usize))
                    .await?;
                if let Some(env) = cutoff_events.get(to_remove as usize) {
                    self.compact(env.seq).await
                } else {
                    Ok(CompactionResult {
                        events_removed: 0,
                        compacted_before_seq: 0,
                    })
                }
            }
            RetentionPolicy::TimeBased(max_age_secs) => {
                let cutoff_us =
                    chrono::Utc::now().timestamp_micros() - (*max_age_secs as i64 * 1_000_000);
                // Scan only events at/after the cutoff to find the compact boundary,
                // instead of loading all events into memory.
                let filter = format!("timestamp_us >= {cutoff_us}");
                let after_cutoff = self.read_filtered_limited(Some(&filter), Some(1)).await?;
                let compact_seq = after_cutoff.first().map(|e| e.seq);
                match compact_seq {
                    Some(seq) => self.compact(seq).await,
                    None => Ok(CompactionResult {
                        events_removed: 0,
                        compacted_before_seq: 0,
                    }),
                }
            }
        }
    }
}

/// Check whether an event envelope matches the given filter criteria.
fn filter_matches(filter: &EventFilter, env: &EventEnvelope) -> bool {
    if let Some(ref realm) = filter.realm {
        if env.realm != *realm {
            return false;
        }
    }
    if let Some(ref ns) = filter.namespace {
        if env.namespace != *ns {
            return false;
        }
    }
    if let Some(ref et) = filter.event_type {
        if env.event_type() != et.as_str() {
            return false;
        }
    }
    if let Some(ref aid) = filter.agent_id {
        if env.agent_id != *aid {
            return false;
        }
    }
    if let Some(after) = filter.after_us {
        if env.timestamp_us < after {
            return false;
        }
    }
    if let Some(before) = filter.before_us {
        if env.timestamp_us > before {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_storage::memory_store::MemoryStore;

    fn null_storage() -> Arc<dyn PhysicalStore> {
        Arc::new(MemoryStore::new())
    }

    #[tokio::test]
    async fn open_on_empty_storage() {
        let log = EventLog::open(null_storage()).await.unwrap();
        assert_eq!(log.next_seq(), 0);
    }

    #[tokio::test]
    async fn append_assigns_sequential_seqs() {
        let log = EventLog::open(null_storage()).await.unwrap();

        let e1 = log
            .append(
                "r",
                "ns",
                "a",
                MemoryEvent::WorkingPushed {
                    id: hirn_core::id::MemoryId::new(),
                },
            )
            .await
            .unwrap();
        assert_eq!(e1.seq, 0);

        let e2 = log
            .append(
                "r",
                "ns",
                "a",
                MemoryEvent::Archived {
                    id: hirn_core::id::MemoryId::new(),
                },
            )
            .await
            .unwrap();
        assert_eq!(e2.seq, 1);

        assert_eq!(log.next_seq(), 2);
    }

    #[tokio::test]
    async fn signed_log_produces_valid_chain() {
        let secret = b"a-32-byte-secret-key-for-testing".to_vec();
        let log = EventLog::open_signed(null_storage(), secret.clone())
            .await
            .unwrap();
        for _ in 0..5 {
            log.append(
                "r",
                "ns",
                "a",
                MemoryEvent::WorkingPushed {
                    id: hirn_core::id::MemoryId::new(),
                },
            )
            .await
            .unwrap();
        }
        // Every event signed, chained, and gap-free.
        log.verify_chain(&secret).await.unwrap();
        assert!(log.verify_integrity(&secret).await.unwrap().is_empty());

        // Each event (after the first) links to its predecessor's tag.
        let events = log.read_all().await.unwrap();
        assert_eq!(events.len(), 5);
        assert!(events[0].prev_hmac.is_none());
        for w in events.windows(2) {
            assert_eq!(w[1].prev_hmac, w[0].hmac);
        }
    }

    #[tokio::test]
    async fn deleting_an_event_breaks_the_chain() {
        let secret = b"a-32-byte-secret-key-for-testing".to_vec();
        let storage = null_storage();
        let log = EventLog::open_signed(storage.clone(), secret.clone())
            .await
            .unwrap();
        for _ in 0..4 {
            log.append(
                "r",
                "ns",
                "a",
                MemoryEvent::WorkingPushed {
                    id: hirn_core::id::MemoryId::new(),
                },
            )
            .await
            .unwrap();
        }
        log.verify_chain(&secret).await.unwrap();

        // Excise a middle event directly from storage (an attacker with store
        // access). Per-event tags still verify, but the chain must not.
        storage.delete(DATASET_NAME, "seq = 2").await.unwrap();
        assert!(
            log.verify_chain(&secret).await.is_err(),
            "removing an event must break the hash chain"
        );
    }

    #[tokio::test]
    async fn append_batch_consecutive_seqs() {
        let log = EventLog::open(null_storage()).await.unwrap();

        let events = vec![
            MemoryEvent::WorkingPushed {
                id: hirn_core::id::MemoryId::new(),
            },
            MemoryEvent::Archived {
                id: hirn_core::id::MemoryId::new(),
            },
            MemoryEvent::Consolidated {
                records_processed: 5,
            },
        ];

        let envs = log.append_batch("r", "ns", "a", events).await.unwrap();
        assert_eq!(envs.len(), 3);
        assert_eq!(envs[0].seq, 0);
        assert_eq!(envs[1].seq, 1);
        assert_eq!(envs[2].seq, 2);
        assert_eq!(log.next_seq(), 3);
    }

    #[tokio::test]
    async fn broadcast_subscriber_receives_events() {
        let log = EventLog::open(null_storage()).await.unwrap();
        let mut rx = log.subscribe();

        let id = hirn_core::id::MemoryId::new();
        log.append("r", "ns", "a", MemoryEvent::WorkingPushed { id })
            .await
            .unwrap();

        let received = rx.try_recv().unwrap();
        assert_eq!(received.seq, 0);
        assert_eq!(received.event_type(), "working_pushed");
    }
}
