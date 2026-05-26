use serde::{Deserialize, Serialize};

use crate::content::MemoryContent;
use crate::error::HirnError;
use crate::id::MemoryId;
use crate::metadata::{Metadata, MetadataValue};
use crate::provenance::Provenance;
use crate::resource::EvidenceLink;
use crate::revision::{LogicalMemoryId, RevisionId, RevisionOperation, RevisionState};
use crate::timestamp::Timestamp;
use crate::types::{AgentId, EventType, Namespace, Origin};

const DEFAULT_EPISODIC_STABILITY_HOURS: f32 = 24.0;
const EPISODIC_STABILITY_GROWTH_FACTOR: f32 = 1.1;
const MAX_EPISODIC_STABILITY_HOURS: f32 = 24.0 * 365.0;

/// An extracted entity reference within an episodic record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityRef {
    pub name: String,
    pub role: String,
    pub entity_id: Option<MemoryId>,
}

/// An episodic memory record — a time-stamped event with full context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodicRecord {
    pub id: MemoryId,
    pub logical_memory_id: LogicalMemoryId,
    pub revision_id: RevisionId,
    /// Monotonic revision number within a logical memory chain.
    pub version: u32,
    /// Operation that produced this immutable revision.
    pub revision_operation: RevisionOperation,
    /// Optional human-readable reason for the revision.
    pub revision_reason: Option<String>,
    /// Optional revision or memory that caused this revision to be written.
    pub revision_causation_id: Option<MemoryId>,
    pub timestamp: Timestamp,
    /// Transaction time: when this revision was durably recorded.
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    /// ID of the revision that superseded this one, if any.
    pub superseded_by: Option<MemoryId>,
    pub event_type: EventType,
    pub content: String,
    pub summary: String,
    pub entities: Vec<EntityRef>,
    pub embedding: Option<Vec<f32>>,
    pub importance: f32,
    pub surprise: f32,
    pub access_count: u64,
    pub last_accessed: Timestamp,
    /// Back-links to semantic records derived via consolidation.
    pub consolidation_ids: Vec<MemoryId>,
    /// Optional episode boundary identifier (event segmentation, EM-LLM).
    pub episode_id: Option<String>,
    /// Memory stability for Ebbinghaus forgetting curve (hours).
    /// Higher = slower decay. Rehearsal grows stability by 10% per access,
    /// saturating at `MAX_EPISODIC_STABILITY_HOURS` to keep retrieval math finite.
    pub stability: f32,
    pub provenance: Provenance,
    pub metadata: Metadata,
    pub namespace: Namespace,
    /// Whether this record has been soft-deleted (archived).
    pub archived: bool,
    /// Optional TTL expiration timestamp. When present, the record is considered
    /// expired after this time and will be hard-deleted by `purge_expired()`.
    #[serde(default)]
    pub expires_at: Option<Timestamp>,
    /// Temporal validity end — when this event was superseded or ceased to be
    /// current. `None` = still valid. Distinct from `expires_at` (TTL-driven
    /// hard deletion) and `superseded_by` (logical chain pointer).
    ///
    /// Together with `timestamp` as the validity start, this gives EpisodicRecord
    /// a full bi-temporal model: transaction time (`created_at`) + valid time
    /// (`timestamp` … `valid_until`).
    #[serde(default)]
    pub valid_until: Option<Timestamp>,
    /// Optional multi-modal content. When present, takes precedence over `content`
    /// for embedding and storage. For backward compatibility, `content` is always
    /// populated with the text representation.
    #[serde(default)]
    pub multi_content: Option<MemoryContent>,
    /// Emotional valence: -1.0 (strongly negative) to 1.0 (strongly positive).
    /// Modulates memory encoding, consolidation, and retrieval strength.
    #[serde(default)]
    pub valence: Option<f32>,
}

impl EpisodicRecord {
    /// Create a new builder for this record type.
    #[must_use]
    pub fn builder() -> EpisodicRecordBuilder {
        EpisodicRecordBuilder::default()
    }

    /// Record an access: bump count, update timestamp, and grow stability.
    ///
    /// Formula: `stability = min(stability × 1.1, MAX_EPISODIC_STABILITY_HOURS)`.
    /// This preserves the existing spaced-repetition ordering for normal access
    /// counts while preventing runaway growth under extreme rehearsal patterns.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        let now = Timestamp::now();
        self.last_accessed = now;
        self.updated_at = now;
        // Spaced repetition: each retrieval strengthens the memory, but the
        // bounded cap prevents extreme rehearsal counts from overflowing f32.
        self.stability =
            (self.stability * EPISODIC_STABILITY_GROWTH_FACTOR).min(MAX_EPISODIC_STABILITY_HOURS);
    }

    /// Whether this revision is a retraction/tombstone.
    #[must_use]
    pub const fn is_retracted(&self) -> bool {
        matches!(self.revision_operation, RevisionOperation::Retract)
    }

    /// Whether this revision should participate in current-state recall.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        !self.archived && !self.is_retracted()
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

    /// Check whether this record has expired relative to the given timestamp.
    #[must_use]
    pub fn is_expired(&self, now: Timestamp) -> bool {
        self.expires_at.is_some_and(|exp| exp <= now)
    }

    /// Whether the fact described by this record was valid at the given point in time.
    ///
    /// A record is valid at `t` when:
    /// - Its event occurred at or before `t` (`timestamp <= t`), AND
    /// - It had not yet been superseded at `t` (`valid_until` is `None` or `valid_until > t`).
    ///
    /// This implements the *valid-time* dimension of the bi-temporal model.
    /// For the *transaction-time* dimension see `created_at`/`updated_at`.
    #[must_use]
    pub fn is_valid_at(&self, t: Timestamp) -> bool {
        self.timestamp <= t && self.valid_until.is_none_or(|valid_until| valid_until > t)
    }
}

/// Builder for [`EpisodicRecord`].
#[derive(Debug, Default)]
pub struct EpisodicRecordBuilder {
    event_type: Option<EventType>,
    content: Option<String>,
    summary: Option<String>,
    entities: Vec<EntityRef>,
    embedding: Option<Vec<f32>>,
    importance: Option<f32>,
    surprise: Option<f32>,
    consolidation_ids: Vec<MemoryId>,
    episode_id: Option<String>,
    metadata: Metadata,
    namespace: Option<Namespace>,
    agent_id: Option<AgentId>,
    timestamp: Option<Timestamp>,
    ttl: Option<std::time::Duration>,
    expires_at: Option<Timestamp>,
    valid_until: Option<Timestamp>,
    multi_content: Option<MemoryContent>,
    valence: Option<f32>,
    evidence_links: Vec<EvidenceLink>,
    origin: Option<Origin>,
}

impl EpisodicRecordBuilder {
    #[must_use]
    pub const fn event_type(mut self, et: EventType) -> Self {
        self.event_type = Some(et);
        self
    }

    /// Set the textual content of the episode.
    #[must_use]
    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Set an optional summary of the episode.
    #[must_use]
    pub fn summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Add a named entity reference to this episode.
    #[must_use]
    pub fn entity(mut self, name: impl Into<String>, role: impl Into<String>) -> Self {
        self.entities.push(EntityRef {
            name: name.into(),
            role: role.into(),
            entity_id: None,
        });
        self
    }

    /// Replace the entity list with the given entities.
    #[must_use]
    pub fn entities(mut self, entities: Vec<EntityRef>) -> Self {
        self.entities = entities;
        self
    }

    /// Set a pre-computed embedding vector.
    #[must_use]
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    #[must_use]
    pub const fn importance(mut self, importance: f32) -> Self {
        self.importance = Some(importance);
        self
    }

    #[must_use]
    pub const fn surprise(mut self, surprise: f32) -> Self {
        self.surprise = Some(surprise);
        self
    }

    /// Insert a key-value metadata entry.
    #[must_use]
    pub fn metadata_entry(
        mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the namespace for this record.
    #[must_use]
    pub fn namespace(mut self, namespace: Namespace) -> Self {
        self.namespace = Some(namespace);
        self
    }

    /// Set the agent that created this record.
    #[must_use]
    pub fn agent_id(mut self, agent_id: AgentId) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    /// Set the IDs of episodes consolidated into this record.
    #[must_use]
    pub fn consolidation_ids(mut self, ids: Vec<MemoryId>) -> Self {
        self.consolidation_ids = ids;
        self
    }

    /// Set an external episode identifier.
    #[must_use]
    pub fn episode_id(mut self, id: impl Into<String>) -> Self {
        self.episode_id = Some(id.into());
        self
    }

    /// Override the record timestamp (defaults to `Timestamp::now()`).
    #[must_use]
    pub const fn timestamp(mut self, ts: Timestamp) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Set a time-to-live duration. The `expires_at` timestamp will be
    /// computed from the record timestamp plus this duration at build time.
    #[must_use]
    pub const fn ttl(mut self, ttl: std::time::Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Set an explicit expiration timestamp.
    #[must_use]
    pub const fn expires_at(mut self, ts: Timestamp) -> Self {
        self.expires_at = Some(ts);
        self
    }

    /// Set the temporal validity end — when this observation was superseded or
    /// became no longer current. Used for retroactive fact correction and
    /// bi-temporal range queries (`AS OF`).
    #[must_use]
    pub const fn valid_until(mut self, ts: Timestamp) -> Self {
        self.valid_until = Some(ts);
        self
    }

    /// Set multi-modal content. The `content` field will be auto-populated
    /// from the text representation if not explicitly set.
    #[must_use]
    pub fn multi_content(mut self, mc: MemoryContent) -> Self {
        if self.content.is_none() {
            self.content = Some(mc.text_for_embedding().into_owned());
        }
        self.multi_content = Some(mc);
        self
    }

    /// Set the emotional valence (-1.0 to 1.0).
    #[must_use]
    pub const fn valence(mut self, valence: f32) -> Self {
        self.valence = Some(valence);
        self
    }

    /// Add a typed resource evidence link to this record's provenance.
    #[must_use]
    pub fn evidence_link(mut self, evidence_link: EvidenceLink) -> Self {
        self.evidence_links.push(evidence_link);
        self
    }

    /// Set the provenance origin (defaults to `DirectObservation`).
    #[must_use]
    pub const fn origin(mut self, origin: Origin) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Build the episodic record.
    pub fn build(self) -> Result<EpisodicRecord, HirnError> {
        let content = self
            .content
            .ok_or_else(|| HirnError::InvalidInput("content is required".into()))?;
        if content.is_empty() {
            return Err(HirnError::InvalidInput("content must be non-empty".into()));
        }

        let agent_id = self
            .agent_id
            .ok_or_else(|| HirnError::InvalidInput("agent_id is required".into()))?;

        let importance = self.importance.unwrap_or(0.5).clamp(0.0, 1.0);
        let surprise = self.surprise.unwrap_or(0.0).clamp(0.0, 1.0);

        let now = Timestamp::now();
        let ts = self.timestamp.unwrap_or(now);
        let id = MemoryId::new();
        let mut provenance =
            Provenance::with_origin(self.origin.unwrap_or(Origin::DirectObservation), agent_id);
        provenance.evidence_links = self.evidence_links;

        // Compute expires_at: explicit > TTL-derived > None.
        let expires_at = self.expires_at.or_else(|| {
            self.ttl.map(|d| {
                let dt = ts.as_datetime()
                    + chrono::Duration::from_std(d).unwrap_or(chrono::Duration::zero());
                Timestamp::from_datetime(dt)
            })
        });

        Ok(EpisodicRecord {
            id,
            logical_memory_id: LogicalMemoryId::from_memory_id(id),
            revision_id: RevisionId::from_memory_id(id),
            version: 1,
            revision_operation: RevisionOperation::Create,
            revision_reason: None,
            revision_causation_id: None,
            timestamp: ts,
            created_at: now,
            updated_at: now,
            superseded_by: None,
            event_type: self.event_type.unwrap_or(EventType::Observation),
            content,
            summary: self.summary.unwrap_or_default(),
            entities: self.entities,
            embedding: self.embedding,
            importance,
            surprise,
            access_count: 0,
            last_accessed: ts,
            stability: DEFAULT_EPISODIC_STABILITY_HOURS,
            consolidation_ids: self.consolidation_ids,
            episode_id: self.episode_id,
            provenance,
            metadata: self.metadata,
            namespace: self.namespace.unwrap_or_default(),
            archived: false,
            expires_at,
            valid_until: self.valid_until,
            multi_content: self.multi_content,
            valence: self.valence.map(|v| v.clamp(-1.0, 1.0)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent() -> AgentId {
        AgentId::new("test").unwrap()
    }

    #[test]
    fn builder_with_all_fields() {
        let rec = EpisodicRecord::builder()
            .content("deployment failed")
            .summary("deploy fail")
            .event_type(EventType::Error)
            .entity("production", "environment")
            .importance(0.9)
            .surprise(0.7)
            .metadata_entry("key", "value")
            .agent_id(agent())
            .build()
            .unwrap();

        assert_eq!(rec.content, "deployment failed");
        assert_eq!(rec.summary, "deploy fail");
        assert_eq!(rec.event_type, EventType::Error);
        assert_eq!(rec.entities.len(), 1);
        assert!((rec.importance - 0.9).abs() < f32::EPSILON);
        assert!((rec.surprise - 0.7).abs() < f32::EPSILON);
        assert_eq!(
            rec.metadata.get("key").unwrap(),
            &MetadataValue::String("value".into())
        );
    }

    #[test]
    fn default_importance_is_half() {
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        assert!((rec.importance - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn importance_clamped() {
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .importance(5.0)
            .build()
            .unwrap();
        assert!((rec.importance - 1.0).abs() < f32::EPSILON);

        let rec2 = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .importance(-2.0)
            .build()
            .unwrap();
        assert!(rec2.importance.abs() < f32::EPSILON);
    }

    #[test]
    fn access_count_starts_at_zero() {
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        assert_eq!(rec.access_count, 0);
    }

    #[test]
    fn last_accessed_equals_timestamp_on_creation() {
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        assert_eq!(rec.last_accessed, rec.timestamp);
    }

    #[test]
    fn mutation_log_starts_empty() {
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        assert!(rec.provenance.mutation_log.is_empty());
    }

    #[test]
    fn builder_attaches_evidence_links_to_provenance() {
        let link = EvidenceLink::new(
            crate::resource::ResourceId::new(),
            crate::resource::EvidenceRole::Attachment,
        );
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .evidence_link(link.clone())
            .build()
            .unwrap();

        assert_eq!(rec.provenance.evidence_links, vec![link]);
    }

    #[test]
    fn empty_content_fails() {
        let result = EpisodicRecord::builder()
            .content("")
            .agent_id(agent())
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn missing_content_fails() {
        let result = EpisodicRecord::builder().agent_id(agent()).build();
        assert!(result.is_err());
    }

    #[test]
    fn record_access_increments() {
        let mut rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        let old_ts = rec.last_accessed;
        let old_stability = rec.stability;
        std::thread::sleep(std::time::Duration::from_millis(2));
        rec.record_access();
        assert_eq!(rec.access_count, 1);
        assert!(rec.last_accessed > old_ts);
        assert!(rec.stability > old_stability);
    }

    #[test]
    fn record_access_stability_is_monotonic_and_bounded() {
        let mut rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();

        for _ in 0..1_000 {
            let prev = rec.stability;
            rec.record_access();
            assert!(rec.stability >= prev);
            assert!(rec.stability <= MAX_EPISODIC_STABILITY_HOURS);
        }
    }

    #[test]
    fn record_access_million_times_stays_finite() {
        let mut rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();

        for _ in 0..1_000_000 {
            rec.record_access();
        }

        assert!(rec.stability.is_finite());
        assert_eq!(rec.stability, MAX_EPISODIC_STABILITY_HOURS);
    }

    #[test]
    fn record_access_preserves_normal_count_ordering() {
        let mut rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();

        for _ in 0..10 {
            rec.record_access();
        }

        let expected = DEFAULT_EPISODIC_STABILITY_HOURS * EPISODIC_STABILITY_GROWTH_FACTOR.powi(10);
        assert!((rec.stability - expected).abs() < f32::EPSILON);
        assert!(rec.stability < MAX_EPISODIC_STABILITY_HOURS);
    }

    #[test]
    fn metadata_supports_various_types() {
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .metadata_entry("str", "hello")
            .metadata_entry("num", 42i64)
            .metadata_entry("bool", true)
            .metadata_entry("null", MetadataValue::Null)
            .build()
            .unwrap();
        assert_eq!(rec.metadata.len(), 4);
    }

    #[test]
    fn serde_round_trip() {
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .entity("prod", "env")
            .metadata_entry("k", 1i64)
            .build()
            .unwrap();
        let bytes = bincode::serialize(&rec).unwrap();
        let back: EpisodicRecord = bincode::deserialize(&bytes).unwrap();
        assert_eq!(rec, back);
    }

    // --- is_valid_at tests ---

    #[test]
    fn is_valid_at_no_valid_until_true_at_and_after_timestamp() {
        let t0 = Timestamp::from_millis(1_000);
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        // Manually construct a record whose timestamp == t0 and valid_until == None
        let rec = EpisodicRecord {
            timestamp: t0,
            valid_until: None,
            ..rec
        };
        assert!(rec.is_valid_at(t0), "valid at its own start time");
        assert!(
            rec.is_valid_at(Timestamp::from_millis(2_000)),
            "valid indefinitely into future"
        );
    }

    #[test]
    fn is_valid_at_false_before_timestamp() {
        let t0 = Timestamp::from_millis(1_000);
        let before = Timestamp::from_millis(500);
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        let rec = EpisodicRecord {
            timestamp: t0,
            valid_until: None,
            ..rec
        };
        assert!(!rec.is_valid_at(before), "not valid before its start time");
    }

    #[test]
    fn is_valid_at_within_validity_window() {
        let t0 = Timestamp::from_millis(1_000);
        let t_end = Timestamp::from_millis(3_000);
        let t_mid = Timestamp::from_millis(2_000);
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        let rec = EpisodicRecord {
            timestamp: t0,
            valid_until: Some(t_end),
            ..rec
        };
        assert!(rec.is_valid_at(t0), "valid at start");
        assert!(rec.is_valid_at(t_mid), "valid in middle of window");
    }

    #[test]
    fn is_valid_at_false_at_and_after_valid_until() {
        let t0 = Timestamp::from_millis(1_000);
        let t_end = Timestamp::from_millis(3_000);
        let after = Timestamp::from_millis(4_000);
        let rec = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        let rec = EpisodicRecord {
            timestamp: t0,
            valid_until: Some(t_end),
            ..rec
        };
        // valid_until is exclusive (valid_until > t is the condition)
        assert!(
            !rec.is_valid_at(t_end),
            "not valid at exclusive valid_until boundary"
        );
        assert!(!rec.is_valid_at(after), "not valid after valid_until");
    }
}
