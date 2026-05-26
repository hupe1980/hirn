use std::fmt;

use serde::{Deserialize, Serialize};

use crate::id::{MemoryId, next_monotonic_ulid};
use crate::timestamp::Timestamp;

/// Stable identity for a memory across all of its revisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LogicalMemoryId(ulid::Ulid);

impl LogicalMemoryId {
    /// Create a new logical memory identifier.
    #[must_use]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(next_monotonic_ulid())
    }

    /// Derive a stable logical identifier from a memory record ID.
    #[must_use]
    pub const fn from_memory_id(id: MemoryId) -> Self {
        Self(id.as_ulid())
    }

    /// Parse a `LogicalMemoryId` from a ULID string.
    pub fn parse(s: &str) -> Result<Self, crate::HirnError> {
        ulid::Ulid::from_string(s).map(Self).map_err(|e| {
            crate::HirnError::InvalidInput(format!("invalid logical memory id '{s}': {e}"))
        })
    }
}

impl fmt::Display for LogicalMemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Immutable identifier for a specific revision of a logical memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RevisionId(ulid::Ulid);

impl RevisionId {
    /// Create a new revision identifier.
    #[must_use]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(next_monotonic_ulid())
    }

    /// Derive a revision identifier from a memory record ID.
    #[must_use]
    pub const fn from_memory_id(id: MemoryId) -> Self {
        Self(id.as_ulid())
    }

    /// Parse a `RevisionId` from a ULID string.
    pub fn parse(s: &str) -> Result<Self, crate::HirnError> {
        ulid::Ulid::from_string(s)
            .map(Self)
            .map_err(|e| crate::HirnError::InvalidInput(format!("invalid revision id '{s}': {e}")))
    }

    /// Convert a revision identifier back into its underlying memory ID.
    #[must_use]
    pub const fn as_memory_id(&self) -> MemoryId {
        MemoryId::from_ulid(self.0)
    }
}

impl fmt::Display for RevisionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Computed state of a revision within its logical memory chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum RevisionState {
    #[default]
    Active,
    Superseded,
    Retracted,
    Quarantined,
    Merged,
}

/// Operation that produced a revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum RevisionOperation {
    #[default]
    Create,
    Correct,
    Override,
    Retract,
    Supersede,
    Merge,
}

/// Revision identifiers attached to recall and query results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RevisionRef {
    pub logical_memory_id: LogicalMemoryId,
    pub revision_id: RevisionId,
    pub state: RevisionState,
}

/// Snapshot target for revision-aware recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecallSnapshot {
    /// Resolve revisions by observed/effective time.
    Observed(Timestamp),
    /// Resolve revisions by recorded/transaction time.
    Recorded(Timestamp),
    /// Resolve a snapshot at the transaction boundary of a specific revision.
    Revision(RevisionId),
}

impl RecallSnapshot {
    /// Build an observed-time snapshot target.
    #[must_use]
    pub const fn observed(ts: Timestamp) -> Self {
        Self::Observed(ts)
    }

    /// Build a recorded-time snapshot target.
    #[must_use]
    pub const fn recorded(ts: Timestamp) -> Self {
        Self::Recorded(ts)
    }

    /// Build a revision-boundary snapshot target.
    #[must_use]
    pub const fn revision(revision_id: RevisionId) -> Self {
        Self::Revision(revision_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_id_round_trip() {
        let id = LogicalMemoryId::new();
        let parsed = LogicalMemoryId::parse(&id.to_string()).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn revision_id_round_trip() {
        let id = RevisionId::new();
        let parsed = RevisionId::parse(&id.to_string()).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn revision_id_maps_back_to_memory_id() {
        let memory_id = MemoryId::new();
        let revision_id = RevisionId::from_memory_id(memory_id);
        assert_eq!(revision_id.as_memory_id(), memory_id);
    }

    #[test]
    fn memory_id_maps_stably() {
        let memory_id = MemoryId::new();
        assert_eq!(
            LogicalMemoryId::from_memory_id(memory_id).to_string(),
            memory_id.to_string()
        );
        assert_eq!(
            RevisionId::from_memory_id(memory_id).to_string(),
            memory_id.to_string()
        );
    }
}
