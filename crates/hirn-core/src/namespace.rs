//! Namespace record types for multi-agent namespace management.

use serde::{Deserialize, Serialize};

use crate::timestamp::Timestamp;
use crate::types::{AgentId, Namespace, NamespaceKind};

/// A persisted namespace record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceRecord {
    /// The namespace identifier.
    pub namespace: Namespace,
    /// The kind of namespace (Private, Shared, Team).
    pub kind: NamespaceKind,
    /// When the namespace was created.
    pub created_at: Timestamp,
    /// Agent IDs with access to this namespace (for Team namespaces).
    /// For Private: contains only the owning agent's ID.
    /// For Shared: empty (all agents have access).
    pub member_agents: Vec<AgentId>,
}

impl NamespaceRecord {
    /// Create a new shared namespace record.
    pub fn shared() -> Self {
        Self {
            namespace: Namespace::shared(),
            kind: NamespaceKind::Shared,
            created_at: Timestamp::now(),
            member_agents: Vec::new(),
        }
    }

    /// Create a private namespace record for an agent.
    pub fn private_for(agent_id: &AgentId) -> Self {
        Self {
            namespace: Namespace::private_for(agent_id),
            kind: NamespaceKind::Private,
            created_at: Timestamp::now(),
            member_agents: vec![agent_id.clone()],
        }
    }

    /// Create a team namespace with the given members.
    pub fn team(namespace: Namespace, members: Vec<AgentId>) -> Self {
        Self {
            namespace,
            kind: NamespaceKind::Team,
            created_at: Timestamp::now(),
            member_agents: members,
        }
    }

    /// Check whether a given agent has access to this namespace.
    pub fn agent_has_access(&self, agent_id: &AgentId) -> bool {
        match self.kind {
            NamespaceKind::Shared | NamespaceKind::Default => true,
            // AgentId is Copy (interned u32) — equality is O(1); no HashSet needed for
            // the typical small team size, but any() short-circuits on the first match.
            NamespaceKind::Private | NamespaceKind::Team => {
                self.member_agents.iter().any(|m| m == agent_id)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_gives_access_to_all() {
        let ns = NamespaceRecord::shared();
        let agent = AgentId::new("any_agent").unwrap();
        assert!(ns.agent_has_access(&agent));
    }

    #[test]
    fn private_only_allows_owner() {
        let owner = AgentId::new("owner").unwrap();
        let other = AgentId::new("other").unwrap();
        let ns = NamespaceRecord::private_for(&owner);
        assert!(ns.agent_has_access(&owner));
        assert!(!ns.agent_has_access(&other));
    }

    #[test]
    fn team_allows_members_only() {
        let a = AgentId::new("alice").unwrap();
        let b = AgentId::new("bob").unwrap();
        let c = AgentId::new("charlie").unwrap();
        let ns = NamespaceRecord::team(
            Namespace::new("team_dev").unwrap(),
            vec![a.clone(), b.clone()],
        );
        assert!(ns.agent_has_access(&a));
        assert!(ns.agent_has_access(&b));
        assert!(!ns.agent_has_access(&c));
    }

    #[test]
    fn serde_round_trip() {
        let agent = AgentId::new("agent_a").unwrap();
        let rec = NamespaceRecord::private_for(&agent);
        let bytes = bincode::serialize(&rec).unwrap();
        let back: NamespaceRecord = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.namespace, rec.namespace);
        assert_eq!(back.kind, rec.kind);
    }
}
