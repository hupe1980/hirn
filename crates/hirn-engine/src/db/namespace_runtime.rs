use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::RwLock;

use hirn_core::agent::AgentRecord;
use hirn_core::namespace::NamespaceRecord;
use hirn_core::types::{AgentId, Namespace};

#[derive(Default)]
pub(crate) struct NamespaceRuntime {
    agents: DashMap<AgentId, AgentRecord>,
    namespaces: RwLock<Option<Arc<Vec<NamespaceRecord>>>>,
    accessible_namespaces: DashMap<AgentId, Vec<Namespace>>,
}

impl NamespaceRuntime {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn cached_agent(&self, agent_id: &AgentId) -> Option<AgentRecord> {
        self.agents.get(agent_id).map(|entry| entry.clone())
    }

    pub(crate) fn cache_agent(&self, agent: AgentRecord) {
        self.agents.insert(agent.id, agent);
    }

    pub(crate) fn evict_agent(&self, agent_id: &AgentId) {
        self.agents.remove(agent_id);
        self.accessible_namespaces.remove(agent_id);
    }

    pub(crate) fn cached_namespaces(&self) -> Option<Arc<Vec<NamespaceRecord>>> {
        self.namespaces.read().clone()
    }

    pub(crate) fn cache_namespaces(&self, namespaces: Vec<NamespaceRecord>) {
        *self.namespaces.write() = Some(Arc::new(namespaces));
        self.accessible_namespaces.clear();
    }

    pub(crate) fn invalidate_namespaces(&self) {
        *self.namespaces.write() = None;
        self.accessible_namespaces.clear();
    }

    pub(crate) fn cached_accessible_namespaces(
        &self,
        agent_id: &AgentId,
    ) -> Option<Vec<Namespace>> {
        self.accessible_namespaces
            .get(agent_id)
            .map(|entry| entry.clone())
    }

    pub(crate) fn cache_accessible_namespaces(
        &self,
        agent_id: AgentId,
        namespaces: Vec<Namespace>,
    ) {
        self.accessible_namespaces.insert(agent_id, namespaces);
    }
}
