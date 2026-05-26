use serde::{Deserialize, Serialize};

use crate::content::MemoryContent;
use crate::error::HirnError;
use crate::id::MemoryId;
use crate::revision::{LogicalMemoryId, RevisionId, RevisionOperation, RevisionState};
use crate::timestamp::Timestamp;
use crate::types::{AgentId, MemoryRef, Priority};

/// A working memory entry — short-lived, high-priority, always in context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkingMemoryEntry {
    pub id: MemoryId,
    pub logical_memory_id: LogicalMemoryId,
    pub revision_id: RevisionId,
    pub content: String,
    /// Event/observed time for this revision.
    pub observed_at: Timestamp,
    pub created_at: Timestamp,
    pub expires_at: Timestamp,
    /// Monotonic revision number within a logical memory chain.
    pub version: u32,
    /// Operation that produced this immutable revision.
    pub revision_operation: RevisionOperation,
    /// Optional human-readable reason for the revision.
    pub revision_reason: Option<String>,
    /// Optional revision or memory that caused this revision to be written.
    pub revision_causation_id: Option<MemoryId>,
    /// ID of the revision that superseded this one, if any.
    pub superseded_by: Option<MemoryId>,
    pub relevance_score: f32,
    pub token_count: u32,
    pub source: Option<MemoryRef>,
    pub priority: Priority,
    pub agent_id: AgentId,
    /// Optional thread/conversation ID for grouping related entries.
    #[serde(default)]
    pub thread_id: Option<String>,
    /// Optional multi-modal content. When present, takes precedence over `content`.
    #[serde(default)]
    pub multi_content: Option<MemoryContent>,
}

impl WorkingMemoryEntry {
    /// Create a new builder for this entry type.
    #[must_use]
    pub fn builder() -> WorkingMemoryEntryBuilder {
        WorkingMemoryEntryBuilder::default()
    }

    /// Check whether this entry has expired relative to the given timestamp.
    #[must_use]
    pub fn is_expired(&self, now: Timestamp) -> bool {
        self.expires_at <= now
    }

    /// Whether this revision is a retraction/tombstone.
    #[must_use]
    pub const fn is_retracted(&self) -> bool {
        matches!(self.revision_operation, RevisionOperation::Retract)
    }

    /// Whether this revision should participate in current-state recall.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        !self.is_retracted()
    }

    /// Computed state for this revision within the context of a logical chain head.
    #[must_use]
    pub fn revision_state_against(&self, head: &Self) -> RevisionState {
        if self.revision_id == head.revision_id {
            if head.is_live() {
                RevisionState::Active
            } else {
                RevisionState::Retracted
            }
        } else {
            RevisionState::Superseded
        }
    }
}

/// Builder for [`WorkingMemoryEntry`].
#[derive(Debug, Default)]
pub struct WorkingMemoryEntryBuilder {
    content: Option<String>,
    expires_at: Option<Timestamp>,
    relevance_score: Option<f32>,
    token_count: Option<u32>,
    source: Option<MemoryRef>,
    priority: Option<Priority>,
    agent_id: Option<AgentId>,
    thread_id: Option<String>,
}

impl WorkingMemoryEntryBuilder {
    /// Set the textual content.
    #[must_use]
    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    #[must_use]
    pub const fn expires_at(mut self, ts: Timestamp) -> Self {
        self.expires_at = Some(ts);
        self
    }

    #[must_use]
    pub const fn relevance_score(mut self, score: f32) -> Self {
        self.relevance_score = Some(score);
        self
    }

    #[must_use]
    pub const fn token_count(mut self, count: u32) -> Self {
        self.token_count = Some(count);
        self
    }

    #[must_use]
    pub const fn source(mut self, source: MemoryRef) -> Self {
        self.source = Some(source);
        self
    }

    #[must_use]
    pub const fn priority(mut self, priority: Priority) -> Self {
        self.priority = Some(priority);
        self
    }

    /// Set the agent that owns this entry.
    #[must_use]
    pub fn agent_id(mut self, agent_id: AgentId) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    /// Set the conversation thread ID.
    #[must_use]
    pub fn thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// Build the entry. Returns an error if required fields are missing
    /// or values are invalid.
    pub fn build(self) -> Result<WorkingMemoryEntry, HirnError> {
        self.build_with_counter(&crate::embed::CharEstimateCounter)
    }

    /// Build the entry using an explicit token counter.
    pub fn build_with_counter(
        self,
        token_counter: &dyn crate::embed::TokenCounter,
    ) -> Result<WorkingMemoryEntry, HirnError> {
        let content = self
            .content
            .ok_or_else(|| HirnError::InvalidInput("content is required".into()))?;
        if content.is_empty() {
            return Err(HirnError::InvalidInput("content must be non-empty".into()));
        }

        let agent_id = self
            .agent_id
            .ok_or_else(|| HirnError::InvalidInput("agent_id is required".into()))?;

        let now = Timestamp::now();
        let expires_at = self.expires_at.unwrap_or_else(|| {
            // Default: 1 hour from now.
            let dt = now.as_datetime() + chrono::Duration::hours(1);
            Timestamp::from_datetime(dt)
        });

        if expires_at <= now {
            return Err(HirnError::InvalidInput(
                "expires_at must be after current time".into(),
            ));
        }

        let relevance_score = self.relevance_score.unwrap_or(1.0);
        if !(0.0..=1.0).contains(&relevance_score) || relevance_score.is_nan() {
            return Err(HirnError::InvalidInput(
                "relevance_score must be in [0.0, 1.0]".into(),
            ));
        }

        let token_count = self
            .token_count
            .unwrap_or_else(|| token_counter.count_tokens(&content) as u32);
        let id = MemoryId::new();

        Ok(WorkingMemoryEntry {
            id,
            logical_memory_id: LogicalMemoryId::from_memory_id(id),
            revision_id: RevisionId::from_memory_id(id),
            content,
            observed_at: now,
            created_at: now,
            expires_at,
            version: 1,
            revision_operation: RevisionOperation::Create,
            revision_reason: None,
            revision_causation_id: None,
            superseded_by: None,
            relevance_score,
            token_count,
            source: self.source,
            priority: self.priority.unwrap_or(Priority::Normal),
            agent_id,
            thread_id: self.thread_id,
            multi_content: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent() -> AgentId {
        AgentId::new("test").unwrap()
    }

    fn future_ts() -> Timestamp {
        let dt = chrono::Utc::now() + chrono::Duration::hours(2);
        Timestamp::from_datetime(dt)
    }

    #[test]
    fn builder_produces_valid_entry() {
        let entry = WorkingMemoryEntry::builder()
            .content("hello world")
            .agent_id(agent())
            .expires_at(future_ts())
            .priority(Priority::High)
            .relevance_score(0.9)
            .token_count(5)
            .build()
            .unwrap();

        assert_eq!(entry.content, "hello world");
        assert_eq!(entry.priority, Priority::High);
        assert!((entry.relevance_score - 0.9).abs() < f32::EPSILON);
        assert_eq!(entry.token_count, 5);
    }

    #[test]
    fn build_with_counter_uses_injected_counter() {
        let entry = WorkingMemoryEntry::builder()
            .content("12345678")
            .agent_id(agent())
            .expires_at(future_ts())
            .build_with_counter(&crate::embed::CharEstimateCounter)
            .unwrap();

        assert_eq!(entry.token_count, 2);
    }

    #[test]
    fn builder_missing_content_fails() {
        let result = WorkingMemoryEntry::builder().agent_id(agent()).build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_empty_content_fails() {
        let result = WorkingMemoryEntry::builder()
            .content("")
            .agent_id(agent())
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_missing_agent_id_fails() {
        let result = WorkingMemoryEntry::builder().content("test").build();
        assert!(result.is_err());
    }

    #[test]
    fn relevance_out_of_range_fails() {
        let result = WorkingMemoryEntry::builder()
            .content("test")
            .agent_id(agent())
            .relevance_score(1.5)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn relevance_nan_fails() {
        let result = WorkingMemoryEntry::builder()
            .content("test")
            .agent_id(agent())
            .relevance_score(f32::NAN)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn expired_entry_detected() {
        let entry = WorkingMemoryEntry::builder()
            .content("test")
            .agent_id(agent())
            .expires_at(future_ts())
            .build()
            .unwrap();

        assert!(!entry.is_expired(Timestamp::now()));
        // Far future should make it expired
        let far_future = Timestamp::from_datetime(chrono::Utc::now() + chrono::Duration::hours(24));
        assert!(entry.is_expired(far_future));
    }

    #[test]
    fn serde_round_trip() {
        let entry = WorkingMemoryEntry::builder()
            .content("test")
            .agent_id(agent())
            .expires_at(future_ts())
            .build()
            .unwrap();
        let bytes = bincode::serialize(&entry).unwrap();
        let back: WorkingMemoryEntry = bincode::deserialize(&bytes).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn default_priority_is_normal() {
        let entry = WorkingMemoryEntry::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        assert_eq!(entry.priority, Priority::Normal);
    }

    #[test]
    fn source_ref_preserved() {
        let source = MemoryRef::new(crate::types::Layer::Episodic, MemoryId::new());
        let entry = WorkingMemoryEntry::builder()
            .content("test")
            .agent_id(agent())
            .source(source)
            .build()
            .unwrap();
        assert!(entry.source.is_some());
    }
}
