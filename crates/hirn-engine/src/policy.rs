//! Cedar-based authorization for hirn.
//!
//! The core [`PolicyEngine`], types, and error definitions live in the
//! [`hirn_policy`] crate. This module re-exports them and provides the
//! [`PolicyNamespaceResolver`] bridge between `hirn-policy` and
//! `hirn-storage::NamespacePolicy`.

// ── Re-exports from hirn-policy ─────────────────────────────────────────

pub use hirn_policy::{
    Action, AuthzDecision, AuthzRequest, DEFAULT_OPEN_POLICY, DEFAULT_SCHEMA, EntityKind,
    PolicyEngine, PolicyError,
};

// ── NamespacePolicy adapter ─────────────────────────────────────────────

use std::sync::Arc;

/// Bridges [`PolicyEngine`] to [`hirn_storage::NamespacePolicy`] for
/// storage-level scan filtering.
///
/// For each registered namespace, checks whether the principal is authorized to
/// perform the configured action. Only namespaces that pass the Cedar check are
/// returned as allowed.
///
/// When the policy engine is in open mode, returns `None` (permit all).
pub struct PolicyNamespaceResolver {
    engine: Arc<PolicyEngine>,
    /// The Cedar action used to test namespace access (typically `Recall`).
    action: Action,
}

impl PolicyNamespaceResolver {
    /// Create a resolver that checks the given action for each namespace.
    pub fn new(engine: Arc<PolicyEngine>, action: Action) -> Self {
        Self { engine, action }
    }

    /// Create a resolver that checks `Recall` access for each namespace.
    pub fn for_recall(engine: Arc<PolicyEngine>) -> Self {
        Self::new(engine, Action::Recall)
    }
}

#[async_trait::async_trait]
impl hirn_storage::NamespacePolicy for PolicyNamespaceResolver {
    async fn allowed_namespaces(&self, principal: &str) -> Option<Vec<String>> {
        self.engine.allowed_namespaces_for(principal, self.action)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_storage::NamespacePolicy;

    #[tokio::test(flavor = "multi_thread")]
    async fn policy_namespace_resolver_open_mode() {
        let engine = Arc::new(PolicyEngine::open_mode());
        let resolver = PolicyNamespaceResolver::for_recall(engine);
        let result = resolver.allowed_namespaces("anyone").await;
        // Open mode -> None (permit all).
        assert!(result.is_none());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn policy_namespace_resolver_filters_namespaces() {
        let engine = Arc::new(
            PolicyEngine::new(DEFAULT_SCHEMA, &[("default.cedar", DEFAULT_OPEN_POLICY)]).unwrap(),
        );
        engine.register_realm("production", "prod realm").unwrap();
        engine
            .register_namespace("ns_a", "public", "production")
            .unwrap();
        engine
            .register_namespace("ns_b", "public", "production")
            .unwrap();
        engine
            .register_agent("agent-1", 50, "2024-01-01", &[])
            .unwrap();

        let resolver = PolicyNamespaceResolver::for_recall(engine);
        let result = resolver.allowed_namespaces("agent-1").await;

        // With default open policy, all namespaces should be allowed.
        assert!(result.is_some());
        let mut allowed = result.unwrap();
        allowed.sort();
        assert_eq!(allowed, vec!["ns_a", "ns_b"]);
    }
}
