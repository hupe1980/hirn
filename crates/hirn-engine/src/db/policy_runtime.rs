use std::sync::Arc;

use hirn_core::timestamp::Timestamp;
use hirn_core::types::AgentId;
use hirn_core::{HirnError, HirnResult};
use hirn_storage::PhysicalStore;

use crate::event::MemoryEvent;
use crate::policy::{Action, AuthzRequest, PolicyEngine};

pub(crate) struct PolicyAuthorization {
    pub(crate) audit_event: MemoryEvent,
    pub(crate) denial_error: Option<HirnError>,
}

pub(crate) struct PolicyRuntime {
    storage: Arc<dyn PhysicalStore>,
    policy_engine: Option<PolicyEngine>,
}

impl PolicyRuntime {
    pub(crate) fn new(storage: Arc<dyn PhysicalStore>) -> Self {
        Self {
            storage,
            policy_engine: None,
        }
    }

    pub(crate) fn set_engine(&mut self, engine: PolicyEngine) {
        self.policy_engine = Some(engine);
    }

    pub(crate) fn engine(&self) -> Option<&PolicyEngine> {
        self.policy_engine.as_ref()
    }

    pub(crate) fn authorize(
        &self,
        agent_id: &str,
        action: Action,
        realm: &str,
        namespace: &str,
    ) -> Option<PolicyAuthorization> {
        let Some(engine) = &self.policy_engine else {
            return None;
        };

        let span = tracing::info_span!(
            "recall.authorize",
            agent_id = %agent_id,
            action = %action,
            decision = tracing::field::Empty,
            policy_ids = tracing::field::Empty,
            latency_us = tracing::field::Empty,
        );

        let _guard = span.enter();

        let request = AuthzRequest {
            agent_id: agent_id.to_string(),
            action,
            realm: realm.to_string(),
            namespace: namespace.to_string(),
        };

        let authz_start = std::time::Instant::now();
        let decision = engine.authorize(&request);
        let authz_elapsed = authz_start.elapsed();

        let decision_label = if decision.allowed { "allow" } else { "deny" };
        let latency_us = authz_elapsed.as_micros() as u64;
        span.record("decision", decision_label);
        span.record("latency_us", latency_us);
        span.record("policy_ids", &format!("{:?}", decision.policy_ids));

        metrics::counter!(crate::metrics::AUTHZ_DECISIONS_TOTAL, "decision" => decision_label)
            .increment(1);
        metrics::histogram!(crate::metrics::AUTHZ_LATENCY_SECONDS)
            .record(authz_elapsed.as_secs_f64());

        let audit_event = if decision.allowed {
            MemoryEvent::AccessGranted {
                action: action.to_string(),
                realm: realm.to_string(),
                namespace: namespace.to_string(),
                policy_ids: decision.policy_ids.clone(),
            }
        } else {
            MemoryEvent::AccessDenied {
                action: action.to_string(),
                realm: realm.to_string(),
                namespace: namespace.to_string(),
                reasons: decision.reasons.clone(),
                policy_ids: decision.policy_ids.clone(),
            }
        };

        let denial_error = if decision.allowed {
            None
        } else {
            let reasons = if decision.reasons.is_empty() {
                "no matching permit policy".to_string()
            } else {
                decision.reasons.join("; ")
            };
            Some(HirnError::AccessDenied(format!(
                "{} cannot {} on {}{}: {}",
                agent_id,
                action,
                realm,
                if namespace.is_empty() {
                    String::new()
                } else {
                    format!("/{namespace}")
                },
                reasons,
            )))
        };

        Some(PolicyAuthorization {
            audit_event,
            denial_error,
        })
    }

    pub(crate) fn is_action_allowed(
        &self,
        agent_id: &str,
        action: Action,
        realm: &str,
        namespace: &str,
    ) -> bool {
        let Some(engine) = &self.policy_engine else {
            return true;
        };

        let request = AuthzRequest {
            agent_id: agent_id.to_string(),
            action,
            realm: realm.to_string(),
            namespace: namespace.to_string(),
        };

        engine.authorize(&request).allowed
    }

    pub(crate) async fn append_audit(
        &self,
        actor: Option<AgentId>,
        action: hirn_core::audit::AuditAction,
    ) -> HirnResult<()> {
        let entry = hirn_core::audit::AuditEntry::new(actor, action);
        let batch = hirn_storage::datasets::audit::to_batch(std::slice::from_ref(&entry))
            .map_err(|e| HirnError::storage(e))?;
        self.storage
            .append(hirn_storage::datasets::audit::DATASET_NAME, batch)
            .await
            .map_err(|e| HirnError::storage(e))?;
        Ok(())
    }

    pub(crate) async fn audit_log(
        &self,
        after: Option<&Timestamp>,
        before: Option<&Timestamp>,
    ) -> HirnResult<Vec<hirn_core::audit::AuditEntry>> {
        let mut parts = Vec::new();
        if let Some(a) = after {
            parts.push(format!("timestamp_ms > {}", a.timestamp_ms()));
        }
        if let Some(b) = before {
            parts.push(format!("timestamp_ms < {}", b.timestamp_ms()));
        }
        let filter = if parts.is_empty() {
            None
        } else {
            Some(parts.join(" AND "))
        };

        let opts = hirn_storage::store::ScanOptions {
            filter,
            ..Default::default()
        };
        let batches = self
            .storage
            .scan(hirn_storage::datasets::audit::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let mut result = Vec::new();
        for batch in &batches {
            let entries = hirn_storage::datasets::audit::from_batch(batch)
                .map_err(|e| HirnError::storage(e))?;
            result.extend(entries);
        }
        result.sort_by_key(|entry| entry.timestamp);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hirn_storage::{HirnDb, HirnDbConfig};

    #[tokio::test(flavor = "multi_thread")]
    async fn no_engine_soft_check_allows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let lance_path = dir.path().join("lance");
        let storage = HirnDb::open(HirnDbConfig::local(lance_path.to_str().expect("path")))
            .await
            .expect("open storage")
            .store_arc();
        let runtime = PolicyRuntime::new(storage);

        assert!(runtime.is_action_allowed("agent", Action::Recall, "realm", "namespace"));
        assert!(
            runtime
                .authorize("agent", Action::Recall, "realm", "namespace")
                .is_none()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_mode_engine_authorize_produces_audit_event() {
        let dir = tempfile::tempdir().expect("tempdir");
        let lance_path = dir.path().join("lance");
        let storage = HirnDb::open(HirnDbConfig::local(lance_path.to_str().expect("path")))
            .await
            .expect("open storage")
            .store_arc();
        let mut runtime = PolicyRuntime::new(storage);
        runtime.set_engine(PolicyEngine::open_mode());

        let decision = runtime
            .authorize("agent", Action::Recall, "realm", "namespace")
            .expect("configured engine should authorize");

        assert!(decision.denial_error.is_none());
        assert!(matches!(
            decision.audit_event,
            MemoryEvent::AccessGranted { .. }
        ));
    }
}
