//! Durable Raft log storage backed by [`redb`].
//!
//! [`DurableLogStore`] persists votes, log entries, the committed index, and the
//! last-purged marker to a single `redb` embedded database file.  Every write is
//! committed inside a `redb` write transaction before the call returns, so the
//! Raft safety property "a vote once cast survives restarts" (§5.2 Ongaro 2014)
//! is upheld.
//!
//! # Usage
//!
//! ```toml
//! # hirnd.toml
//! [raft]
//! node_id = 1
//! advertise_addr = "https://node-1.example:3000"
//! data_dir = "/var/lib/hirnd/raft"
//! ```
//!
//! When `raft.data_dir` is set, `main.rs` opens a `DurableLogStore` rooted at
//! `<data_dir>/raft-log.redb`.  When it is absent, the daemon falls back to
//! [`DevMemLogStore`](super::store::DevMemLogStore) and emits an `error!` log if
//! peers are configured (in-memory store is unsafe for multi-node production).

use std::fmt::Debug;
use std::ops::RangeBounds;
use std::path::Path;
use std::sync::Arc;

use openraft::storage::RaftLogStorage;
use openraft::{Entry, LogId, LogState, RaftLogReader, StorageError, StorageIOError, Vote};
use parking_lot::Mutex;
use redb::{Database, ReadableTable, TableDefinition};

use super::types::*;

// ── redb table definitions ────────────────────────────────────────────────────

/// Stores the current vote (single row, key = VOTE_KEY).
const VOTE_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("vote");
/// Stores log entries keyed by their Raft log index.
const LOG_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("log");
/// Stores small metadata values (committed index, last-purged id).
const META_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("meta");

const VOTE_KEY: &[u8] = b"vote";
const COMMITTED_KEY: &[u8] = b"committed";
const LAST_PURGED_KEY: &[u8] = b"last_purged";

// ── serde helpers ─────────────────────────────────────────────────────────────

fn encode<T: serde::Serialize>(v: &T) -> Vec<u8> {
    // bincode is infallible for all well-formed openraft types.
    bincode::serialize(v).expect("bincode serialize is infallible for openraft types")
}

fn decode<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    ctx: &'static str,
) -> Result<T, StorageError<NodeId>> {
    bincode::deserialize(bytes).map_err(|e| {
        let io_err = std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("bincode decode ({ctx}): {e}"),
        );
        StorageIOError::read_logs(openraft::AnyError::new(&io_err)).into()
    })
}

fn db_io_err<E: std::error::Error>(e: E, ctx: &'static str) -> StorageError<NodeId> {
    let io_err = std::io::Error::new(std::io::ErrorKind::Other, format!("{ctx}: {e}"));
    StorageIOError::read_logs(openraft::AnyError::new(&io_err)).into()
}

// ── DurableLogStore ──────────────────────────────────────────────────────────

/// Production Raft log store backed by a `redb` embedded database.
///
/// Cloning shares the underlying database handle — all clones see the same
/// persisted state, protected by an internal write mutex (redb allows only one
/// concurrent write transaction).
#[derive(Clone)]
pub struct DurableLogStore {
    db: Arc<Database>,
    /// Serializes write transactions: redb permits only one writer at a time.
    write_lock: Arc<Mutex<()>>,
}

impl DurableLogStore {
    /// Open (or create) the Raft log database at `path`.
    ///
    /// Creates all required tables on first open.
    pub fn open(path: &Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create raft data dir {}: {e}", parent.display()))?;
        }
        let db =
            Database::create(path).map_err(|e| format!("redb open {}: {e}", path.display()))?;

        // Ensure all tables exist (idempotent on subsequent opens).
        {
            let wtxn = db
                .begin_write()
                .map_err(|e| format!("redb begin_write: {e}"))?;
            wtxn.open_table(VOTE_TABLE)
                .map_err(|e| format!("redb vote table: {e}"))?;
            wtxn.open_table(LOG_TABLE)
                .map_err(|e| format!("redb log table: {e}"))?;
            wtxn.open_table(META_TABLE)
                .map_err(|e| format!("redb meta table: {e}"))?;
            wtxn.commit()
                .map_err(|e| format!("redb init commit: {e}"))?;
        }

        Ok(Self {
            db: Arc::new(db),
            write_lock: Arc::new(Mutex::new(())),
        })
    }
}

impl RaftLogReader<TypeConfig> for DurableLogStore {
    async fn try_get_log_entries<RB>(
        &mut self,
        range: RB,
    ) -> Result<Vec<Entry<TypeConfig>>, StorageError<NodeId>>
    where
        RB: RangeBounds<u64> + Clone + Debug + Send,
    {
        let rtxn = self
            .db
            .begin_read()
            .map_err(|e| db_io_err(e, "begin_read"))?;
        let table = rtxn
            .open_table(LOG_TABLE)
            .map_err(|e| db_io_err(e, "open log table"))?;
        let mut entries = Vec::new();
        for result in table.range(range).map_err(|e| db_io_err(e, "range"))? {
            let (_, v) = result.map_err(|e| db_io_err(e, "range iter"))?;
            entries.push(decode::<Entry<TypeConfig>>(v.value(), "log entry")?);
        }
        Ok(entries)
    }
}

impl RaftLogStorage<TypeConfig> for DurableLogStore {
    type LogReader = Self;

    async fn get_log_state(&mut self) -> Result<LogState<TypeConfig>, StorageError<NodeId>> {
        let rtxn = self
            .db
            .begin_read()
            .map_err(|e| db_io_err(e, "begin_read"))?;
        let log_table = rtxn
            .open_table(LOG_TABLE)
            .map_err(|e| db_io_err(e, "open log"))?;
        let meta_table = rtxn
            .open_table(META_TABLE)
            .map_err(|e| db_io_err(e, "open meta"))?;

        let last = log_table
            .last()
            .map_err(|e| db_io_err(e, "log last"))?
            .map(|(_, v)| decode::<Entry<TypeConfig>>(v.value(), "last entry"))
            .transpose()?
            .map(|e| e.log_id);

        let last_purged = meta_table
            .get(LAST_PURGED_KEY)
            .map_err(|e| db_io_err(e, "read last_purged"))?
            .map(|v| decode::<LogId<NodeId>>(v.value(), "last_purged"))
            .transpose()?;

        Ok(LogState {
            last_purged_log_id: last_purged,
            last_log_id: last,
        })
    }

    async fn get_log_reader(&mut self) -> Self::LogReader {
        self.clone()
    }

    async fn save_vote(&mut self, vote: &Vote<NodeId>) -> Result<(), StorageError<NodeId>> {
        let _guard = self.write_lock.lock();
        let wtxn = self
            .db
            .begin_write()
            .map_err(|e| db_io_err(e, "begin_write vote"))?;
        {
            let mut table = wtxn
                .open_table(VOTE_TABLE)
                .map_err(|e| db_io_err(e, "open vote"))?;
            let bytes = encode(vote);
            table
                .insert(VOTE_KEY, bytes.as_slice())
                .map_err(|e| db_io_err(e, "insert vote"))?;
        }
        wtxn.commit().map_err(|e| db_io_err(e, "commit vote"))?;
        Ok(())
    }

    async fn read_vote(&mut self) -> Result<Option<Vote<NodeId>>, StorageError<NodeId>> {
        let rtxn = self
            .db
            .begin_read()
            .map_err(|e| db_io_err(e, "begin_read vote"))?;
        let table = rtxn
            .open_table(VOTE_TABLE)
            .map_err(|e| db_io_err(e, "open vote"))?;
        match table.get(VOTE_KEY).map_err(|e| db_io_err(e, "read vote"))? {
            Some(v) => Ok(Some(decode::<Vote<NodeId>>(v.value(), "vote")?)),
            None => Ok(None),
        }
    }

    async fn save_committed(
        &mut self,
        committed: Option<LogId<NodeId>>,
    ) -> Result<(), StorageError<NodeId>> {
        let _guard = self.write_lock.lock();
        let wtxn = self
            .db
            .begin_write()
            .map_err(|e| db_io_err(e, "begin_write committed"))?;
        {
            let mut table = wtxn
                .open_table(META_TABLE)
                .map_err(|e| db_io_err(e, "open meta"))?;
            let bytes = encode(&committed);
            table
                .insert(COMMITTED_KEY, bytes.as_slice())
                .map_err(|e| db_io_err(e, "insert committed"))?;
        }
        wtxn.commit()
            .map_err(|e| db_io_err(e, "commit committed"))?;
        Ok(())
    }

    async fn read_committed(&mut self) -> Result<Option<LogId<NodeId>>, StorageError<NodeId>> {
        let rtxn = self
            .db
            .begin_read()
            .map_err(|e| db_io_err(e, "begin_read committed"))?;
        let table = rtxn
            .open_table(META_TABLE)
            .map_err(|e| db_io_err(e, "open meta"))?;
        match table
            .get(COMMITTED_KEY)
            .map_err(|e| db_io_err(e, "read committed"))?
        {
            Some(v) => Ok(decode::<Option<LogId<NodeId>>>(v.value(), "committed")?),
            None => Ok(None),
        }
    }

    async fn append<I>(
        &mut self,
        entries: I,
        callback: openraft::storage::LogFlushed<TypeConfig>,
    ) -> Result<(), StorageError<NodeId>>
    where
        I: IntoIterator<Item = Entry<TypeConfig>> + Send,
    {
        let _guard = self.write_lock.lock();
        let wtxn = self
            .db
            .begin_write()
            .map_err(|e| db_io_err(e, "begin_write append"))?;
        {
            let mut table = wtxn
                .open_table(LOG_TABLE)
                .map_err(|e| db_io_err(e, "open log"))?;
            for entry in entries {
                let bytes = encode(&entry);
                table
                    .insert(entry.log_id.index, bytes.as_slice())
                    .map_err(|e| db_io_err(e, "insert log entry"))?;
            }
        }
        wtxn.commit().map_err(|e| db_io_err(e, "commit append"))?;
        callback.log_io_completed(Ok(()));
        Ok(())
    }

    async fn truncate(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let _guard = self.write_lock.lock();
        let wtxn = self
            .db
            .begin_write()
            .map_err(|e| db_io_err(e, "begin_write truncate"))?;
        {
            let mut table = wtxn
                .open_table(LOG_TABLE)
                .map_err(|e| db_io_err(e, "open log truncate"))?;
            // Collect indices to remove (range [log_id.index, +∞)).
            let to_remove: Vec<u64> = table
                .range(log_id.index..)
                .map_err(|e| db_io_err(e, "range truncate"))?
                .map(|r| r.map(|(k, _)| k.value()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| db_io_err(e, "collect truncate"))?;
            for idx in to_remove {
                table
                    .remove(idx)
                    .map_err(|e| db_io_err(e, "remove truncate"))?;
            }
        }
        wtxn.commit().map_err(|e| db_io_err(e, "commit truncate"))?;
        Ok(())
    }

    async fn purge(&mut self, log_id: LogId<NodeId>) -> Result<(), StorageError<NodeId>> {
        let _guard = self.write_lock.lock();
        let wtxn = self
            .db
            .begin_write()
            .map_err(|e| db_io_err(e, "begin_write purge"))?;
        {
            let mut log_table = wtxn
                .open_table(LOG_TABLE)
                .map_err(|e| db_io_err(e, "open log purge"))?;
            // Remove all entries up to and including log_id.index.
            let to_remove: Vec<u64> = log_table
                .range(..=log_id.index)
                .map_err(|e| db_io_err(e, "range purge"))?
                .map(|r| r.map(|(k, _)| k.value()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| db_io_err(e, "collect purge"))?;
            for idx in to_remove {
                log_table
                    .remove(idx)
                    .map_err(|e| db_io_err(e, "remove purge"))?;
            }
            // Persist last_purged marker.
            let mut meta_table = wtxn
                .open_table(META_TABLE)
                .map_err(|e| db_io_err(e, "open meta purge"))?;
            let bytes = encode(&log_id);
            meta_table
                .insert(LAST_PURGED_KEY, bytes.as_slice())
                .map_err(|e| db_io_err(e, "insert last_purged"))?;
        }
        wtxn.commit().map_err(|e| db_io_err(e, "commit purge"))?;
        Ok(())
    }
}
