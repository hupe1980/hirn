//! Agent registration types for multi-agent memory.

use serde::{Deserialize, Serialize};

use crate::timestamp::Timestamp;
use crate::types::AgentId;

/// A registered agent in the multi-agent memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRecord {
    /// Unique agent identifier.
    pub id: AgentId,
    /// Human-readable display name.
    pub display_name: String,
    /// When the agent was registered.
    pub created_at: Timestamp,
    /// Bayesian trust score [0.0, 1.0]. Starts at `initial_trust` (default 0.5).
    pub trust_score: f32,
    /// Number of memories confirmed by other agents.
    pub confirmed_count: u32,
    /// Number of memories contradicted by other agents.
    pub contradicted_count: u32,
}

impl AgentRecord {
    /// Create a new agent record with default trust.
    pub fn new(id: AgentId, display_name: impl Into<String>) -> Self {
        Self {
            id,
            display_name: display_name.into(),
            created_at: Timestamp::now(),
            trust_score: 0.5,
            confirmed_count: 0,
            contradicted_count: 0,
        }
    }

    /// Update trust score based on confirmation/contradiction counts using
    /// Bayesian updating: `trust = (confirmed + α) / (confirmed + contradicted + α + β)`
    /// where α=1.0 (prior successes) and β=1.0 (prior failures) for a uniform prior.
    pub fn update_trust(&mut self) {
        let alpha = 1.0_f32;
        let beta = 1.0_f32;
        self.trust_score = (self.confirmed_count as f32 + alpha)
            / (self.confirmed_count as f32 + self.contradicted_count as f32 + alpha + beta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_agent_has_neutral_trust() {
        let agent = AgentRecord::new(AgentId::new("test").unwrap(), "Test Agent");
        assert!((agent.trust_score - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn trust_increases_with_confirmations() {
        let mut agent = AgentRecord::new(AgentId::new("test").unwrap(), "Test Agent");
        agent.confirmed_count = 9;
        agent.contradicted_count = 1;
        agent.update_trust();
        assert!(agent.trust_score > 0.8);
    }

    #[test]
    fn trust_decreases_with_contradictions() {
        let mut agent = AgentRecord::new(AgentId::new("test").unwrap(), "Test Agent");
        agent.confirmed_count = 2;
        agent.contradicted_count = 8;
        agent.update_trust();
        assert!(agent.trust_score < 0.35);
    }

    #[test]
    fn serde_round_trip() {
        let agent = AgentRecord::new(AgentId::new("agent_a").unwrap(), "Agent A");
        let bytes = bincode::serialize(&agent).unwrap();
        let back: AgentRecord = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.id, agent.id);
        assert_eq!(back.display_name, agent.display_name);
    }
}
