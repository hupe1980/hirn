use serde::{Deserialize, Serialize};

use crate::id::MemoryId;
use crate::resource::EvidenceLink;
use crate::timestamp::Timestamp;
use crate::types::{AgentId, MutationTrigger, Origin};

/// Evidence reference for confidence basis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub source_id: MemoryId,
    pub description: String,
}

/// A single mutation in a memory record's history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mutation {
    pub timestamp: Timestamp,
    pub trigger: MutationTrigger,
    pub field: String,
    pub old_value: String,
    pub new_value: String,
    pub reason: String,
}

/// Full provenance chain for a memory record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    origin: Origin,
    pub source_event: Option<MemoryId>,
    pub extraction_model: Option<String>,
    pub confidence_basis: Vec<EvidenceRef>,
    #[serde(default)]
    pub evidence_links: Vec<EvidenceLink>,
    pub mutation_log: Vec<Mutation>,
    pub created_by: AgentId,
}

impl Provenance {
    /// Returns the immutable origin of this provenance record.
    #[must_use]
    pub const fn origin(&self) -> &Origin {
        &self.origin
    }

    /// Create a new provenance record for a direct observation.
    #[must_use]
    pub const fn direct(agent_id: AgentId) -> Self {
        Self {
            origin: Origin::DirectObservation,
            source_event: None,
            extraction_model: None,
            confidence_basis: Vec::new(),
            evidence_links: Vec::new(),
            mutation_log: Vec::new(),
            created_by: agent_id,
        }
    }

    /// Create a new provenance record with a specific origin.
    #[must_use]
    pub const fn with_origin(origin: Origin, agent_id: AgentId) -> Self {
        Self {
            origin,
            source_event: None,
            extraction_model: None,
            confidence_basis: Vec::new(),
            evidence_links: Vec::new(),
            mutation_log: Vec::new(),
            created_by: agent_id,
        }
    }

    /// Maximum number of mutations retained in the log.
    const MAX_MUTATION_LOG: usize = 100;

    /// Record a mutation in the provenance log.
    /// Oldest entries are evicted when the log exceeds the cap.
    pub fn record_mutation(&mut self, mutation: Mutation) {
        if self.mutation_log.len() >= Self::MAX_MUTATION_LOG {
            self.mutation_log.remove(0);
        }
        self.mutation_log.push(mutation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    #[test]
    fn direct_provenance_defaults() {
        let p = Provenance::direct(test_agent());
        assert_eq!(p.origin, Origin::DirectObservation);
        assert!(p.source_event.is_none());
        assert!(p.extraction_model.is_none());
        assert!(p.confidence_basis.is_empty());
        assert!(p.evidence_links.is_empty());
        assert!(p.mutation_log.is_empty());
    }

    #[test]
    fn record_mutation_appends() {
        let mut p = Provenance::direct(test_agent());
        let m = Mutation {
            timestamp: Timestamp::now(),
            trigger: MutationTrigger::Manual,
            field: "importance".to_string(),
            old_value: "0.5".to_string(),
            new_value: "0.8".to_string(),
            reason: "tested".to_string(),
        };
        p.record_mutation(m);
        assert_eq!(p.mutation_log.len(), 1);
    }

    #[test]
    fn serde_round_trip() {
        let p = Provenance::direct(test_agent());
        let bytes = bincode::serialize(&p).unwrap();
        let back: Provenance = bincode::deserialize(&bytes).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn mutation_log_eviction_cap() {
        let mut p = Provenance::direct(test_agent());
        // Fill to exactly MAX_MUTATION_LOG.
        for i in 0..Provenance::MAX_MUTATION_LOG {
            p.record_mutation(Mutation {
                timestamp: Timestamp::now(),
                trigger: MutationTrigger::Manual,
                field: "importance".to_string(),
                old_value: format!("{i}"),
                new_value: format!("{}", i + 1),
                reason: "fill".to_string(),
            });
        }
        assert_eq!(p.mutation_log.len(), Provenance::MAX_MUTATION_LOG);
        // One more should evict the oldest.
        p.record_mutation(Mutation {
            timestamp: Timestamp::now(),
            trigger: MutationTrigger::Decay,
            field: "importance".to_string(),
            old_value: "100".to_string(),
            new_value: "101".to_string(),
            reason: "overflow".to_string(),
        });
        assert_eq!(p.mutation_log.len(), Provenance::MAX_MUTATION_LOG);
        // Oldest entry (old_value "0") was evicted; first entry now has old_value "1".
        assert_eq!(p.mutation_log[0].old_value, "1");
        // Newest entry is the overflow.
        assert_eq!(p.mutation_log.last().unwrap().reason, "overflow");
    }
}
