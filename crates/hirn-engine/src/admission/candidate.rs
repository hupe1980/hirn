//! Memory candidate — the input to the admission pipeline.

use hirn_core::episodic::{EntityRef, EpisodicRecord};
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::types::{AgentId, Namespace};

/// A candidate memory to be evaluated by the admission pipeline.
///
/// Extracted from an [`EpisodicRecord`] before it enters storage.
#[derive(Debug, Clone)]
pub struct MemoryCandidate {
    /// Unique ID for this candidate (same as the record's ID).
    pub id: MemoryId,
    /// Text content of the memory.
    pub content: String,
    /// Extracted entities.
    pub entities: Vec<EntityRef>,
    /// Pre-computed embedding vector (if available).
    pub embedding: Option<Vec<f32>>,
    /// Source agent.
    pub agent_id: AgentId,
    /// Namespace scope.
    pub namespace: Namespace,
    /// Importance score assigned by the caller.
    pub importance: f32,
    /// Surprise score assigned by the caller.
    pub surprise: f32,
    /// Arbitrary metadata.
    pub metadata: Metadata,
}

impl MemoryCandidate {
    /// Build a candidate from an [`EpisodicRecord`].
    pub fn from_record(record: &EpisodicRecord) -> Self {
        Self {
            id: record.id,
            content: record.content.clone(),
            entities: record.entities.clone(),
            embedding: record.embedding.clone(),
            agent_id: record.provenance.created_by.clone(),
            namespace: record.namespace.clone(),
            importance: record.importance,
            surprise: record.surprise,
            metadata: record.metadata.clone(),
        }
    }
}
