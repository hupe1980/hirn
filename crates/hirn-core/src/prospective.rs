//! Prospective implication record for forward-looking inference.
//!
//! Prospective implications capture what future consequences a memory might
//! have. They enable proactive retrieval: when new context matches an
//! implication's embedding, the source memory is surfaced before it becomes
//! directly relevant.

use serde::{Deserialize, Serialize};

use crate::id::MemoryId;
use crate::timestamp::Timestamp;

/// A forward-looking implication derived from a source memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProspectiveImplication {
    /// Unique identifier for this implication.
    pub id: MemoryId,
    /// The memory that generated this implication.
    pub source_memory_id: MemoryId,
    /// Natural-language description of the anticipated consequence.
    pub implication_text: String,
    /// When this implication was created (epoch ms).
    pub created_at: Timestamp,
}

impl ProspectiveImplication {
    /// Create a new prospective implication with auto-generated ID.
    #[must_use]
    pub fn new(source_memory_id: MemoryId, implication_text: impl Into<String>) -> Self {
        Self {
            id: MemoryId::new(),
            source_memory_id,
            implication_text: implication_text.into(),
            created_at: Timestamp::now(),
        }
    }
}
