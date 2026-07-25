//! Consolidation lease protocol.
//!
//! Ensures only one node runs consolidation/compaction for a given realm at a time.
//! Leases are stored in the Raft state machine for consistency across the cluster.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::types::NodeId;

/// A time-limited lease granting exclusive consolidation rights for a realm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationLease {
    /// Node holding this lease.
    pub holder: NodeId,
    /// When the lease was acquired (Unix epoch seconds).
    pub acquired_at: u64,
    /// When the lease expires (Unix epoch seconds).
    pub expires_at: u64,
    /// The realm this lease covers.
    pub realm: String,
    /// Monotonic fencing token, issued by consensus and strictly increasing
    /// across every acquisition cluster-wide (Kleppmann fencing token). A
    /// stalled ex-holder that resumes after a GC/VM pause carries a stale
    /// fence, so downstream storage mutations can reject it. Renewal preserves
    /// the fence (same acquisition session); a fresh acquisition bumps it.
    #[serde(default)]
    pub fence: u64,
}

impl ConsolidationLease {
    /// Default lease duration: 5 minutes.
    pub const DEFAULT_DURATION_SECS: u64 = 300;

    /// Create a new lease starting at the given epoch timestamp.
    ///
    /// The timestamp must be the one stamped into the Raft log entry at
    /// proposal time — never the local clock — so that every replica applying
    /// the entry computes the identical `expires_at`. `fence` is the
    /// consensus-issued monotonic fencing token for this acquisition.
    pub fn new(
        realm: String,
        holder: NodeId,
        duration_secs: u64,
        acquired_at_epoch_secs: u64,
        fence: u64,
    ) -> Self {
        Self {
            holder,
            acquired_at: acquired_at_epoch_secs,
            expires_at: acquired_at_epoch_secs.saturating_add(duration_secs),
            realm,
            fence,
        }
    }

    /// Check if the lease has expired relative to an explicit epoch timestamp.
    ///
    /// Used during `apply()` with the proposal timestamp carried in the log
    /// entry so the expiry decision is identical on every replica.
    pub fn is_expired_at(&self, now_epoch_secs: u64) -> bool {
        now_epoch_secs >= self.expires_at
    }

    /// Check if the lease has expired against the local clock.
    ///
    /// Query-time only — must not be used inside the Raft state machine's
    /// `apply()` path, where all inputs have to come from the log entry.
    pub fn is_expired(&self) -> bool {
        self.is_expired_at(now_epoch_secs())
    }

    /// Check if a specific node holds this lease (and it hasn't expired).
    pub fn is_held_by(&self, node: NodeId) -> bool {
        self.holder == node && !self.is_expired()
    }

    /// Renew the lease from an explicit epoch timestamp (only by the holder).
    ///
    /// Like [`ConsolidationLease::new`], the timestamp must come from the log
    /// entry so renewal produces the same `expires_at` on every replica.
    pub fn renew_at(&mut self, duration_secs: u64, now_epoch_secs: u64) {
        self.expires_at = now_epoch_secs.saturating_add(duration_secs);
    }

    /// Remaining seconds before expiry.
    pub fn remaining_secs(&self) -> u64 {
        let now = now_epoch_secs();
        self.expires_at.saturating_sub(now)
    }
}

/// Current Unix time in seconds.
///
/// Only for proposal sites and query-time checks; `apply()` must use the
/// timestamp carried in the log entry instead.
pub fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before UNIX epoch")
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_creation_and_expiry() {
        let now = now_epoch_secs();
        let lease = ConsolidationLease::new("test-realm".into(), 1, 300, now, 7);
        assert_eq!(lease.holder, 1);
        assert_eq!(lease.realm, "test-realm");
        assert_eq!(lease.acquired_at, now);
        assert_eq!(lease.expires_at, now + 300);
        assert_eq!(lease.fence, 7);
        assert!(!lease.is_expired());
        assert!(!lease.is_expired_at(now));
        assert!(lease.is_expired_at(now + 300));
        assert!(lease.is_held_by(1));
        assert!(!lease.is_held_by(2));
        assert!(lease.remaining_secs() > 0);
    }

    #[test]
    fn expired_lease() {
        let lease = ConsolidationLease {
            holder: 1,
            acquired_at: 0,
            expires_at: 1, // expired long ago
            realm: "test".into(),
            fence: 1,
        };
        assert!(lease.is_expired());
        assert!(!lease.is_held_by(1));
        assert_eq!(lease.remaining_secs(), 0);
    }

    #[test]
    fn renewal() {
        let now = now_epoch_secs();
        let mut lease = ConsolidationLease::new("r".into(), 1, 10, now, 3);
        lease.renew_at(600, now + 5);
        assert_eq!(lease.expires_at, now + 605);
        // Renewal preserves the fencing token (same acquisition session).
        assert_eq!(lease.fence, 3);
        assert!(lease.remaining_secs() > 500);
    }

    #[test]
    fn expiry_is_deterministic_for_a_fixed_timestamp() {
        // Two leases built from the same entry data must agree on expiry
        // regardless of when (or where) the check runs.
        let a = ConsolidationLease::new("r".into(), 1, 300, 1_000, 1);
        let b = ConsolidationLease::new("r".into(), 1, 300, 1_000, 1);
        assert_eq!(a.expires_at, b.expires_at);
        assert!(!a.is_expired_at(1_299));
        assert!(a.is_expired_at(1_300));
        assert!(b.is_expired_at(1_300));
    }
}
