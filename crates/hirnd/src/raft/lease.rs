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
}

impl ConsolidationLease {
    /// Default lease duration: 5 minutes.
    pub const DEFAULT_DURATION_SECS: u64 = 300;

    /// Create a new lease starting now.
    pub fn new(realm: String, holder: NodeId, duration_secs: u64) -> Self {
        let now = now_epoch_secs();
        Self {
            holder,
            acquired_at: now,
            expires_at: now.saturating_add(duration_secs),
            realm,
        }
    }

    /// Check if the lease has expired.
    pub fn is_expired(&self) -> bool {
        now_epoch_secs() >= self.expires_at
    }

    /// Check if a specific node holds this lease (and it hasn't expired).
    pub fn is_held_by(&self, node: NodeId) -> bool {
        self.holder == node && !self.is_expired()
    }

    /// Renew the lease for an additional duration (only by the holder).
    pub fn renew(&mut self, duration_secs: u64) {
        let now = now_epoch_secs();
        self.expires_at = now.saturating_add(duration_secs);
    }

    /// Remaining seconds before expiry.
    pub fn remaining_secs(&self) -> u64 {
        let now = now_epoch_secs();
        self.expires_at.saturating_sub(now)
    }
}

fn now_epoch_secs() -> u64 {
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
        let lease = ConsolidationLease::new("test-realm".into(), 1, 300);
        assert_eq!(lease.holder, 1);
        assert_eq!(lease.realm, "test-realm");
        assert!(!lease.is_expired());
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
        };
        assert!(lease.is_expired());
        assert!(!lease.is_held_by(1));
        assert_eq!(lease.remaining_secs(), 0);
    }

    #[test]
    fn renewal() {
        let mut lease = ConsolidationLease::new("r".into(), 1, 10);
        lease.renew(600);
        assert!(lease.remaining_secs() > 500);
    }
}
