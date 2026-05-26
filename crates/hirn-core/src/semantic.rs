use serde::{Deserialize, Serialize};

use crate::error::HirnError;
use crate::id::MemoryId;
use crate::provenance::Provenance;
use crate::resource::EvidenceLink;
use crate::revision::{LogicalMemoryId, RevisionId, RevisionOperation, RevisionState};
use crate::timestamp::Timestamp;
use crate::types::{AgentId, EdgeRelation, KnowledgeType, Namespace, Origin};

/// A typed relationship to another semantic concept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConceptEdge {
    pub target_id: MemoryId,
    pub relation: EdgeRelation,
    pub weight: f32,
}

/// A semantic memory record — long-term knowledge distilled from episodes.
///
/// `created_at` tracks transaction time while `valid_from` tracks observed/event
/// time, which allows revision-aware reads to distinguish when a fact was
/// recorded from when it was considered true.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticRecord {
    pub id: MemoryId,
    pub logical_memory_id: LogicalMemoryId,
    pub revision_id: RevisionId,
    pub concept: String,
    pub knowledge_type: KnowledgeType,
    pub description: String,
    pub embedding: Option<Vec<f32>>,
    pub related_concepts: Vec<ConceptEdge>,
    pub confidence: f32,
    pub source_episodes: Vec<MemoryId>,
    pub evidence_count: u32,
    pub contradiction_ids: Vec<MemoryId>,
    /// Last time this record was accessed/retrieved (for temporal decay).
    pub last_accessed: Timestamp,
    pub access_count: u64,
    /// Monotonic revision number within a logical memory chain.
    pub version: u32,
    /// Operation that produced this immutable revision.
    pub revision_operation: RevisionOperation,
    /// Optional human-readable reason for the revision.
    pub revision_reason: Option<String>,
    /// Optional revision or memory that caused this revision to be written.
    pub revision_causation_id: Option<MemoryId>,
    pub provenance: Provenance,
    pub namespace: Namespace,
    /// Transaction time: when this revision was durably recorded.
    /// This remains distinct from `valid_from`, which models event/observed time.
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    /// Temporal validity start (when this fact became true / was observed).
    /// Defaults to `created_at` for directly recorded facts.
    pub valid_from: Timestamp,
    /// Temporal validity end (when this fact was superseded). None = still current.
    pub valid_until: Option<Timestamp>,
    /// ID of the record that supersedes this one (version chain).
    pub superseded_by: Option<MemoryId>,
    /// Logical memory this revision was merged into, if this chain was retired.
    pub merged_into: Option<LogicalMemoryId>,
    /// Whether this record has been archived (importance decayed below threshold).
    pub archived: bool,
}

impl SemanticRecord {
    /// Create a new builder for this record type.
    #[must_use]
    pub fn builder() -> SemanticRecordBuilder {
        SemanticRecordBuilder::default()
    }

    /// Record an access: bump count and update timestamps.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        let now = Timestamp::now();
        self.last_accessed = now;
        self.updated_at = now;
    }

    /// Whether this revision is a retraction/tombstone.
    #[must_use]
    pub const fn is_retracted(&self) -> bool {
        matches!(self.revision_operation, RevisionOperation::Retract)
    }

    /// Whether this logical chain has been retired into another logical memory.
    #[must_use]
    pub const fn is_merged(&self) -> bool {
        self.merged_into.is_some()
    }

    /// Whether this revision should participate in current-state recall.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        !self.is_retracted() && !self.is_merged()
    }

    /// Computed state for this revision when treated as the active head.
    #[must_use]
    pub const fn logical_state(&self) -> RevisionState {
        if self.is_retracted() {
            RevisionState::Retracted
        } else if self.is_merged() {
            RevisionState::Merged
        } else {
            RevisionState::Active
        }
    }

    /// Computed state for this revision within the context of a logical chain head.
    #[must_use]
    pub fn revision_state_against(&self, head: &Self) -> RevisionState {
        if self.revision_id == head.revision_id {
            head.logical_state()
        } else {
            RevisionState::Superseded
        }
    }
}

/// Builder for [`SemanticRecord`].
#[derive(Debug, Default)]
pub struct SemanticRecordBuilder {
    concept: Option<String>,
    knowledge_type: Option<KnowledgeType>,
    description: Option<String>,
    embedding: Option<Vec<f32>>,
    related_concepts: Vec<ConceptEdge>,
    confidence: Option<f32>,
    source_episodes: Vec<MemoryId>,
    contradiction_ids: Vec<MemoryId>,
    namespace: Option<Namespace>,
    agent_id: Option<AgentId>,
    evidence_links: Vec<EvidenceLink>,
    origin: Option<Origin>,
}

impl SemanticRecordBuilder {
    /// Set the concept name.
    #[must_use]
    pub fn concept(mut self, concept: impl Into<String>) -> Self {
        self.concept = Some(concept.into());
        self
    }

    #[must_use]
    pub const fn knowledge_type(mut self, kt: KnowledgeType) -> Self {
        self.knowledge_type = Some(kt);
        self
    }

    /// Set the concept description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set a pre-computed embedding vector.
    #[must_use]
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Add a related concept edge.
    #[must_use]
    pub fn related_concept(mut self, edge: ConceptEdge) -> Self {
        self.related_concepts.push(edge);
        self
    }

    #[must_use]
    pub const fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Add a source episode that contributed to this concept.
    #[must_use]
    pub fn source_episode(mut self, id: MemoryId) -> Self {
        self.source_episodes.push(id);
        self
    }

    /// Add a contradicting memory ID.
    #[must_use]
    pub fn contradiction(mut self, id: MemoryId) -> Self {
        self.contradiction_ids.push(id);
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

    /// Add a typed resource evidence link to this record's provenance.
    #[must_use]
    pub fn evidence_link(mut self, evidence_link: EvidenceLink) -> Self {
        self.evidence_links.push(evidence_link);
        self
    }

    #[must_use]
    pub const fn origin(mut self, origin: Origin) -> Self {
        self.origin = Some(origin);
        self
    }

    /// Build the semantic record.
    pub fn build(self) -> Result<SemanticRecord, HirnError> {
        let concept = self
            .concept
            .ok_or_else(|| HirnError::InvalidInput("concept is required".into()))?;
        if concept.is_empty() {
            return Err(HirnError::InvalidInput("concept must be non-empty".into()));
        }

        let description = self
            .description
            .ok_or_else(|| HirnError::InvalidInput("description is required".into()))?;

        let agent_id = self
            .agent_id
            .ok_or_else(|| HirnError::InvalidInput("agent_id is required".into()))?;

        let confidence = self.confidence.unwrap_or(0.5).clamp(0.0, 1.0);

        let now = Timestamp::now();

        let id = MemoryId::new();
        let mut provenance =
            Provenance::with_origin(self.origin.unwrap_or(Origin::DirectObservation), agent_id);
        provenance.evidence_links = self.evidence_links;

        Ok(SemanticRecord {
            id,
            logical_memory_id: LogicalMemoryId::from_memory_id(id),
            revision_id: RevisionId::from_memory_id(id),
            concept,
            knowledge_type: self.knowledge_type.unwrap_or(KnowledgeType::Propositional),
            description,
            embedding: self.embedding,
            related_concepts: self.related_concepts,
            confidence,
            source_episodes: self.source_episodes,
            evidence_count: 0,
            contradiction_ids: self.contradiction_ids,
            last_accessed: now,
            access_count: 0,
            version: 1,
            revision_operation: RevisionOperation::Create,
            revision_reason: None,
            revision_causation_id: None,
            provenance,
            namespace: self.namespace.unwrap_or_default(),
            created_at: now,
            updated_at: now,
            valid_from: now,
            valid_until: None,
            superseded_by: None,
            merged_into: None,
            archived: false,
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
    fn builder_produces_valid_record() {
        let rec = SemanticRecord::builder()
            .concept("caching")
            .knowledge_type(KnowledgeType::Propositional)
            .description("Caching improves performance")
            .confidence(0.9)
            .agent_id(agent())
            .build()
            .unwrap();

        assert_eq!(rec.concept, "caching");
        assert_eq!(rec.knowledge_type, KnowledgeType::Propositional);
        assert!((rec.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn version_starts_at_one() {
        let rec = SemanticRecord::builder()
            .concept("test")
            .description("desc")
            .agent_id(agent())
            .build()
            .unwrap();
        assert_eq!(rec.version, 1);
        assert_eq!(rec.logical_memory_id.to_string(), rec.id.to_string());
        assert_eq!(rec.revision_id.to_string(), rec.id.to_string());
    }

    #[test]
    fn evidence_count_starts_at_zero() {
        let rec = SemanticRecord::builder()
            .concept("test")
            .description("desc")
            .agent_id(agent())
            .build()
            .unwrap();
        assert_eq!(rec.evidence_count, 0);
    }

    #[test]
    fn confidence_clamped() {
        let rec = SemanticRecord::builder()
            .concept("test")
            .description("desc")
            .agent_id(agent())
            .confidence(5.0)
            .build()
            .unwrap();
        assert!((rec.confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn empty_concept_fails() {
        let result = SemanticRecord::builder()
            .concept("")
            .description("desc")
            .agent_id(agent())
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn missing_concept_fails() {
        let result = SemanticRecord::builder()
            .description("desc")
            .agent_id(agent())
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn record_access_increments() {
        let mut rec = SemanticRecord::builder()
            .concept("test")
            .description("desc")
            .agent_id(agent())
            .build()
            .unwrap();
        let old_ts = rec.updated_at;
        std::thread::sleep(std::time::Duration::from_millis(2));
        rec.record_access();
        assert_eq!(rec.access_count, 1);
        assert!(rec.updated_at > old_ts);
    }

    #[test]
    fn serde_round_trip() {
        let rec = SemanticRecord::builder()
            .concept("test")
            .description("desc")
            .agent_id(agent())
            .source_episode(MemoryId::new())
            .build()
            .unwrap();
        let bytes = bincode::serialize(&rec).unwrap();
        let back: SemanticRecord = bincode::deserialize(&bytes).unwrap();
        assert_eq!(rec, back);
    }

    #[test]
    fn logical_state_reports_merged_and_retracted_heads() {
        let mut merged = SemanticRecord::builder()
            .concept("test")
            .description("desc")
            .agent_id(agent())
            .build()
            .unwrap();
        merged.merged_into = Some(LogicalMemoryId::new());
        assert_eq!(merged.logical_state(), RevisionState::Merged);

        let mut retracted = SemanticRecord::builder()
            .concept("test")
            .description("desc")
            .agent_id(agent())
            .build()
            .unwrap();
        retracted.revision_operation = RevisionOperation::Retract;
        assert_eq!(retracted.logical_state(), RevisionState::Retracted);
    }

    #[test]
    fn revision_state_against_marks_older_revisions_superseded() {
        let original = SemanticRecord::builder()
            .concept("test")
            .description("desc")
            .agent_id(agent())
            .build()
            .unwrap();

        let mut head = original.clone();
        head.id = MemoryId::new();
        head.revision_id = RevisionId::from_memory_id(head.id);
        head.version = 2;

        assert_eq!(
            original.revision_state_against(&head),
            RevisionState::Superseded
        );
        assert_eq!(head.revision_state_against(&head), RevisionState::Active);
    }

    #[test]
    fn builder_attaches_evidence_links_to_provenance() {
        let link = EvidenceLink::new(
            crate::resource::ResourceId::new(),
            crate::resource::EvidenceRole::Proof,
        );
        let record = SemanticRecord::builder()
            .concept("test")
            .description("desc")
            .agent_id(agent())
            .evidence_link(link.clone())
            .build()
            .unwrap();

        assert_eq!(record.provenance.evidence_links, vec![link]);
    }
}
