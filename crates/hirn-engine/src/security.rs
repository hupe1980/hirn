//! Memory security: anomaly detection, quarantine management, Bayesian trust,
//! and collective corruption defense.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use hirn_core::id::MemoryId;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::AgentId;
use hirn_core::{GeneratedCognitionReview, QuarantinedRecordKind};

/// Status of a quarantined record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuarantineStatus {
    /// Awaiting review.
    Pending,
    /// Approved and released to main store.
    Approved,
    /// Rejected and retained as a durable review artifact.
    Rejected,
    /// Previously approved output was rolled back.
    RolledBack,
}

/// A quarantined memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineEntry {
    /// The original memory ID.
    pub memory_id: MemoryId,
    /// Logical type of the quarantined record.
    pub record_kind: QuarantinedRecordKind,
    /// Bincode-serialized original record.
    pub record: Vec<u8>,
    /// Anomaly score that triggered quarantine.
    pub anomaly_score: f32,
    /// Human-readable reason for quarantine.
    pub reason: String,
    /// Current status.
    pub status: QuarantineStatus,
    /// When the record was quarantined.
    pub created_at: Timestamp,
    /// Who reviewed it (if reviewed).
    pub reviewed_by: Option<AgentId>,
    /// When it was reviewed (if reviewed).
    pub reviewed_at: Option<Timestamp>,
    /// Durable generated-cognition quality, approval, and rollback metadata.
    pub generated_review: Option<GeneratedCognitionReview>,
}

/// Result of approving a quarantined record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineApprovalOutcome {
    /// The quarantine entry that was approved.
    pub approved_entry_id: MemoryId,
    /// Semantic or episodic revisions/materializations created by approval.
    pub applied_memory_ids: Vec<MemoryId>,
    /// Human-readable description of the applied change.
    pub change_summary: String,
    /// Updated review metadata after approval.
    pub generated_review: Option<GeneratedCognitionReview>,
}

/// Result of rolling back a previously approved generated output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuarantineRollbackOutcome {
    pub rolled_back_entry_id: MemoryId,
    pub removed_memory_ids: Vec<MemoryId>,
    pub restored_memory_ids: Vec<MemoryId>,
    pub reason: String,
    pub generated_review: Option<GeneratedCognitionReview>,
}

// ── Collective Corruption Defense ───────────────────────────────────────

/// Configuration for the collective corruption detector.
#[derive(Debug, Clone)]
pub struct CorruptionDefenseConfig {
    /// Max quarantine events from a single agent within `window_seconds`
    /// before that agent is rate-limited.
    pub max_quarantines_per_window: usize,
    /// Sliding window size in seconds.
    pub window_seconds: u64,
}

impl Default for CorruptionDefenseConfig {
    fn default() -> Self {
        Self {
            max_quarantines_per_window: 5,
            window_seconds: 300, // 5 minutes
        }
    }
}

/// Tracks per-agent quarantine timestamps for burst detection.
#[derive(Debug, Default)]
pub struct CorruptionDefense {
    /// agent_id → list of quarantine timestamps (sorted ascending).
    history: HashMap<AgentId, Vec<Timestamp>>,
    config: CorruptionDefenseConfig,
}

impl CorruptionDefense {
    /// Create a new corruption defense tracker with the given configuration.
    pub fn new(config: CorruptionDefenseConfig) -> Self {
        Self {
            history: HashMap::new(),
            config,
        }
    }

    /// Record a quarantine event for an agent. Returns `true` if the agent
    /// has exceeded the threshold and should be rate-limited.
    pub fn record_quarantine(&mut self, agent_id: &AgentId) -> bool {
        let now = Timestamp::now();
        let cutoff = now
            .as_datetime()
            .checked_sub_signed(chrono::Duration::seconds(self.config.window_seconds as i64));

        let timestamps = self.history.entry(agent_id.clone()).or_default();

        // Evict events outside the window.
        if let Some(cutoff_dt) = cutoff {
            timestamps.retain(|ts| ts.as_datetime() >= cutoff_dt);
        }

        timestamps.push(now);

        timestamps.len() > self.config.max_quarantines_per_window
    }

    /// Check whether an agent is currently rate-limited (burst in progress).
    pub fn is_rate_limited(&self, agent_id: &AgentId) -> bool {
        let Some(timestamps) = self.history.get(agent_id) else {
            return false;
        };

        let now = Timestamp::now();
        let cutoff = now
            .as_datetime()
            .checked_sub_signed(chrono::Duration::seconds(self.config.window_seconds as i64));

        let recent_count = match cutoff {
            Some(cutoff_dt) => timestamps
                .iter()
                .filter(|ts| ts.as_datetime() >= cutoff_dt)
                .count(),
            None => timestamps.len(),
        };

        recent_count > self.config.max_quarantines_per_window
    }

    /// Clear rate-limit state for an agent (e.g., after manual review).
    pub fn clear_agent(&mut self, agent_id: &AgentId) {
        self.history.remove(agent_id);
    }

    /// The current config.
    pub fn config(&self) -> &CorruptionDefenseConfig {
        &self.config
    }

    /// F-14: Snapshot the per-agent quarantine burst history for persistence.
    pub fn snapshot(&self) -> Vec<(String, Vec<u64>)> {
        self.history
            .iter()
            .map(|(agent_id, timestamps)| {
                let ms: Vec<u64> = timestamps.iter().map(|ts| ts.millis()).collect();
                (agent_id.to_string(), ms)
            })
            .collect()
    }

    /// F-14: Restore per-agent quarantine burst history from a persisted snapshot.
    pub fn restore(&mut self, entries: &[(String, Vec<u64>)]) {
        for (agent_str, timestamps_ms) in entries {
            if let Ok(agent_id) = AgentId::new(agent_str) {
                let timestamps: Vec<Timestamp> = timestamps_ms
                    .iter()
                    .map(|&ms| Timestamp::from_millis(ms))
                    .collect();
                self.history.insert(agent_id, timestamps);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarantine_entry_serde_round_trip() {
        let entry = QuarantineEntry {
            memory_id: MemoryId::new(),
            record_kind: QuarantinedRecordKind::Episodic,
            record: vec![1, 2, 3],
            anomaly_score: 0.85,
            reason: "outlier embedding".to_string(),
            status: QuarantineStatus::Pending,
            created_at: Timestamp::now(),
            reviewed_by: None,
            reviewed_at: None,
            generated_review: None,
        };
        let bytes = bincode::serialize(&entry).unwrap();
        let back: QuarantineEntry = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.memory_id, entry.memory_id);
        assert_eq!(back.status, QuarantineStatus::Pending);
    }

    #[test]
    fn corruption_defense_rate_limits_after_burst() {
        let config = CorruptionDefenseConfig {
            max_quarantines_per_window: 3,
            window_seconds: 300,
        };
        let mut defense = CorruptionDefense::new(config);
        let agent = AgentId::new("bad-agent").unwrap();

        assert!(!defense.record_quarantine(&agent));
        assert!(!defense.record_quarantine(&agent));
        assert!(!defense.record_quarantine(&agent));
        // 4th quarantine triggers rate limiting
        assert!(defense.record_quarantine(&agent));
        assert!(defense.is_rate_limited(&agent));

        // Other agent is unaffected
        let good_agent = AgentId::new("good-agent").unwrap();
        assert!(!defense.is_rate_limited(&good_agent));
    }

    #[test]
    fn corruption_defense_clear_resets() {
        let config = CorruptionDefenseConfig {
            max_quarantines_per_window: 1,
            window_seconds: 300,
        };
        let mut defense = CorruptionDefense::new(config);
        let agent = AgentId::new("agent").unwrap();

        defense.record_quarantine(&agent);
        defense.record_quarantine(&agent);
        assert!(defense.is_rate_limited(&agent));

        defense.clear_agent(&agent);
        assert!(!defense.is_rate_limited(&agent));
    }
}
