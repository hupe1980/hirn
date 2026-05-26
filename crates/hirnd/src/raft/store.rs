//! In-memory Raft log storage — **development only**.
//!
//! `DevMemLogStore` is intentionally not durable.  On any process restart all
//! votes, log entries, and the committed index are lost.  This violates the Raft
//! safety property that a vote once cast must survive restarts (§5.2 Ongaro
//! 2014): a node can restart and vote for a different candidate in the same term,
//! which can elect two leaders simultaneously.
//!
//! **Never use `DevMemLogStore` in multi-node production deployments.**
//! Use [`DurableLogStore`](super::durable_store::DurableLogStore) instead.
//! `hirnd` will log an `error!` at startup if peers are configured and no
//! `raft.data_dir` is set.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::RangeBounds;
use std::sync::Arc;

use openraft::storage::RaftLogStorage;
use openraft::{Entry, LogId, LogState, RaftLogReader, StorageError, Vote};
use parking_lot::RwLock;

use super::types::*;

/// In-memory Raft log store — **development / single-node only**.
///
/// Not durable.  Votes, log entries, and the committed index are lost on every
/// process restart.  See module-level docs for the safety implication.
///
/// For multi-node production clusters use
/// [`DurableLogStore`](super::durable_store::DurableLogStore) instead.
#[derive(Default)]
pub struct DevMemLogStore {
    inner: Arc<RwLock<DevMemLogStoreInner>>,
}

#[derive(Default)]
struct DevMemLogStoreInner {
    vote: Option<Vote<NodeId>>,
    log: BTreeMap<u64, Entry<TypeConfig>>,
    committed: Option<LogId<NodeId>>,
    last_purged: Option<LogId<NodeId>>,
}

impl DevMemLogStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Clone for DevMemLogStore {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl RaftLogReader<TypeConfig> for DevMemLogStore {
    async fn try_get_log_entries<RB: RangeBounds<u64> + Clone + Debug + Send>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<TypeConfig>>, StorageError<NodeId>> {
        let inner = self.inner.read();
        Ok(inner.log.range(range).map(|(_, e)| e.clone()).collect())
    }
}

impl RaftLogStorage<TypeConfig> for DevMemLogStore {
    type LogReader = Self;

    async fn get_log_state(&mut self) -> Result<LogState<TypeConfig>, StorageError<NodeId>> {
        let inner = self.inner.read();
        let last = inner.log.values().last().map(|e| e.log_id);
        Ok(LogState {
            last_purged_log_id: inner.last_purged,
            last_log_id: last,
        })
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    async fn save_vote(&mut self, vote: &Vote<NodeId>) -> Result<(), StorageError<NodeId>> {
        self.inner.write().vote = Some(*vote);
        Ok(())
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<NodeId>>, StorageError<NodeId>> {
        Ok(self.inner.read().vote)
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogId<NodeId>>,
    ) -> Result<(), StorageError<NodeId>> {
        self.inner.write().committed = committed;
        Ok(())
    }

    async fn read_committed(&mut self) -> Result<Option<LogId<NodeId>>, StorageError<NodeId>> {
        Ok(self.inner.read().committed)
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: openraft::storage::LogFlushed<TypeConfig>,
    ) -> Result<(), StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>> + Send,
    {
        let mut inner = self.inner.write();
        for entry in entries {
            inner.log.insert(entry.log_id.index, entry);
        }
        callback.log_io_completed(Ok(()));
        Ok(())
    }

    async fn truncate(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.write();
        // O(log n) truncation using BTreeMap::split_off.
        inner.log.split_off(&log_id.index);
        Ok(())
    }

    async fn purge(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let mut inner = self.inner.write();
        // O(log n) purge: keep entries after log_id.index, discard the rest.
        let remaining = inner.log.split_off(&(log_id.index + 1));
        inner.log = remaining;
        inner.last_purged = Some(log_id);
        Ok(())
    }
}
