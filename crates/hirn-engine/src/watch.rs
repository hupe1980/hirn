//! Watch subscriptions for real-time reactive memory.
//!
//! Builds on the existing `EventLog` broadcast channel by adding
//! filter-based subscriptions. Subscribers receive only events
//! matching their `WatchFilter`.

use hirn_core::error::{HirnError, HirnResult};
use hirn_core::types::{Layer, Namespace};
use tokio::sync::broadcast;

use crate::event::{EventEnvelope, MemoryEvent};

// ═══════════════════════════════════════════════════════════════════════════
// Watch Filter
// ═══════════════════════════════════════════════════════════════════════════

/// Filter criteria for watch subscriptions.
#[derive(Debug, Clone, PartialEq)]
pub enum WatchFilter {
    /// Receive all events.
    All,
    /// Events from a specific realm.
    Realm(String),
    /// Events affecting any of the provided memory layers.
    Layers(Vec<Layer>),
    /// Events from a specific namespace.
    Namespace(String),
    /// Events from any of the provided namespaces.
    Namespaces(Vec<String>),
    /// Events triggered by a specific agent.
    AgentId(String),
    /// Events mentioning any of these entities (checked in content previews).
    Entities(Vec<String>),
    /// Only events with importance updates above this threshold.
    ImportanceAbove(f32),
    /// Only contradiction-related events.
    Contradictions,
    /// Only specific event types.
    EventTypes(Vec<String>),
    /// All child filters must match.
    AllOf(Vec<WatchFilter>),
}

impl WatchFilter {
    /// Combine filters conjunctively, flattening nested `AllOf` nodes.
    #[must_use]
    pub fn all_of(filters: Vec<Self>) -> Self {
        let mut flattened = Vec::new();
        for filter in filters {
            match filter {
                Self::All => {}
                Self::AllOf(children) => flattened.extend(children),
                other => flattened.push(other),
            }
        }

        match flattened.len() {
            0 => Self::All,
            1 => flattened.into_iter().next().unwrap_or(Self::All),
            _ => Self::AllOf(flattened),
        }
    }

    /// Restrict this filter to a namespace set.
    #[must_use]
    pub fn scoped_to_namespaces(self, allowed_namespaces: &[Namespace]) -> Self {
        let namespaces = allowed_namespaces
            .iter()
            .map(|namespace| namespace.as_str().to_string())
            .collect();
        Self::all_of(vec![Self::Namespaces(namespaces), self])
    }

    /// Reject filters that explicitly name namespaces outside the allowed set.
    pub fn validate_allowed_namespaces(&self, allowed_namespaces: &[Namespace]) -> HirnResult<()> {
        let mut referenced_namespaces = Vec::new();
        self.collect_referenced_namespaces(&mut referenced_namespaces);

        for namespace in referenced_namespaces {
            let allowed = allowed_namespaces
                .iter()
                .any(|allowed_namespace| allowed_namespace.as_str() == namespace);
            if !allowed {
                return Err(HirnError::AccessDenied(format!(
                    "watch cannot access namespace '{}'",
                    namespace
                )));
            }
        }

        Ok(())
    }

    fn collect_referenced_namespaces(&self, namespaces: &mut Vec<String>) {
        match self {
            Self::Namespace(namespace) => namespaces.push(namespace.clone()),
            Self::Namespaces(items) => namespaces.extend(items.iter().cloned()),
            Self::AllOf(filters) => {
                for filter in filters {
                    filter.collect_referenced_namespaces(namespaces);
                }
            }
            Self::All
            | Self::Realm(_)
            | Self::Layers(_)
            | Self::AgentId(_)
            | Self::Entities(_)
            | Self::ImportanceAbove(_)
            | Self::Contradictions
            | Self::EventTypes(_) => {}
        }
    }

    /// Check whether an event envelope matches this filter.
    pub fn matches(&self, envelope: &EventEnvelope) -> bool {
        match self {
            WatchFilter::All => true,
            WatchFilter::Realm(realm) => envelope.realm == *realm,
            WatchFilter::Layers(layers) => envelope
                .event
                .layer()
                .is_some_and(|layer| layers.contains(&layer)),
            WatchFilter::Namespace(ns) => envelope.namespace == *ns,
            WatchFilter::Namespaces(namespaces) => namespaces.contains(&envelope.namespace),
            WatchFilter::AgentId(agent_id) => envelope.agent_id == *agent_id,
            WatchFilter::Entities(entities) => {
                let text = match &envelope.event {
                    MemoryEvent::EpisodeCreated {
                        content_preview, ..
                    } => content_preview.as_str(),
                    MemoryEvent::SemanticCreated { concept_name, .. } => concept_name.as_str(),
                    MemoryEvent::ProceduralCreated { procedure_name, .. } => {
                        procedure_name.as_str()
                    }
                    MemoryEvent::Reconsolidated { reason, .. } => reason.as_str(),
                    _ => "",
                };
                let lower = text.to_lowercase();
                entities.iter().any(|e| lower.contains(&e.to_lowercase()))
            }
            WatchFilter::ImportanceAbove(threshold) => {
                matches!(
                    &envelope.event,
                    MemoryEvent::ImportanceUpdated { new_value, .. }
                        if *new_value > *threshold
                )
            }
            WatchFilter::Contradictions => match &envelope.event {
                MemoryEvent::ContradictionDetected { .. } => true,
                MemoryEvent::Reconsolidated { reason, .. } => reason.contains("contradict"),
                _ => false,
            },
            WatchFilter::EventTypes(types) => {
                let event_type = envelope.event.event_type();
                types.iter().any(|t| t == event_type)
            }
            WatchFilter::AllOf(filters) => filters.iter().all(|filter| filter.matches(envelope)),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Watch Subscription
// ═══════════════════════════════════════════════════════════════════════════

/// A filtered watch subscription over the event stream.
pub struct WatchSubscription {
    filter: WatchFilter,
    rx: broadcast::Receiver<EventEnvelope>,
}

impl WatchSubscription {
    /// Create a new subscription from a broadcast receiver and filter.
    pub fn new(rx: broadcast::Receiver<EventEnvelope>, filter: WatchFilter) -> Self {
        Self { filter, rx }
    }

    /// Receive the next matching event, blocking until one arrives.
    ///
    /// Returns `Err` if the channel is closed or the subscriber lagged
    /// (missed events due to slow consumption).
    pub async fn next(&mut self) -> HirnResult<EventEnvelope> {
        loop {
            match self.rx.recv().await {
                Ok(envelope) => {
                    if self.filter.matches(&envelope) {
                        return Ok(envelope);
                    }
                    // Skip non-matching events.
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    return Err(HirnError::LimitExceeded(format!(
                        "watch subscriber lagged, missed {n} events"
                    )));
                }
                Err(broadcast::error::RecvError::Closed) => {
                    return Err(HirnError::InvalidInput("event channel closed".to_string()));
                }
            }
        }
    }

    /// Try to receive the next matching event without blocking indefinitely.
    ///
    /// Returns `Ok(None)` if the channel is closed.
    /// Returns `Err` if the subscriber lagged (missed events).
    pub fn try_next(&mut self) -> HirnResult<Option<EventEnvelope>> {
        loop {
            match self.rx.try_recv() {
                Ok(envelope) => {
                    if self.filter.matches(&envelope) {
                        return Ok(Some(envelope));
                    }
                }
                Err(broadcast::error::TryRecvError::Empty) => return Ok(None),
                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                    return Err(HirnError::LimitExceeded(format!(
                        "watch subscriber lagged, missed {n} events"
                    )));
                }
                Err(broadcast::error::TryRecvError::Closed) => return Ok(None),
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HirnDB::watch integration
// ═══════════════════════════════════════════════════════════════════════════

use crate::db::HirnDB;

impl HirnDB {
    /// Create a watch subscription with the given filter.
    ///
    /// Requires an active `EventLog` (returns error otherwise).
    pub fn watch(&self, filter: WatchFilter) -> HirnResult<WatchSubscription> {
        let event_log = self
            .event_log()
            .ok_or_else(|| HirnError::InvalidInput("event log not configured".to_string()))?;
        let rx = event_log.subscribe();
        Ok(WatchSubscription::new(rx, filter))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::id::MemoryId;
    use hirn_core::types::Layer;

    fn make_envelope(event: MemoryEvent, namespace: &str) -> EventEnvelope {
        EventEnvelope::new(1, "default", namespace, "test-agent", event)
    }

    #[test]
    fn filter_all_matches_everything() {
        let filter = WatchFilter::All;
        let env = make_envelope(
            MemoryEvent::Forgotten {
                id: MemoryId::new(),
            },
            "ns1",
        );
        assert!(filter.matches(&env));
    }

    #[test]
    fn filter_namespace_matches_correct_ns() {
        let filter = WatchFilter::Namespace("shared".to_string());

        let matching = make_envelope(
            MemoryEvent::Forgotten {
                id: MemoryId::new(),
            },
            "shared",
        );
        let non_matching = make_envelope(
            MemoryEvent::Forgotten {
                id: MemoryId::new(),
            },
            "private",
        );

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn filter_namespaces_matches_any_allowed_ns() {
        let filter = WatchFilter::Namespaces(vec!["shared".to_string(), "team".to_string()]);

        let matching = make_envelope(
            MemoryEvent::Forgotten {
                id: MemoryId::new(),
            },
            "team",
        );
        let non_matching = make_envelope(
            MemoryEvent::Forgotten {
                id: MemoryId::new(),
            },
            "private",
        );

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn filter_entities_case_insensitive() {
        let filter = WatchFilter::Entities(vec!["auth".to_string()]);

        let matching = make_envelope(
            MemoryEvent::EpisodeCreated {
                id: MemoryId::new(),
                content_preview: "Discussed Auth flow with OAuth2".to_string(),
            },
            "ns",
        );
        let non_matching = make_envelope(
            MemoryEvent::EpisodeCreated {
                id: MemoryId::new(),
                content_preview: "Talked about recipes".to_string(),
            },
            "ns",
        );

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn filter_importance_above_threshold() {
        let filter = WatchFilter::ImportanceAbove(0.8);

        let above = make_envelope(
            MemoryEvent::ImportanceUpdated {
                id: MemoryId::new(),
                old_value: 0.5,
                new_value: 0.9,
            },
            "ns",
        );
        let below = make_envelope(
            MemoryEvent::ImportanceUpdated {
                id: MemoryId::new(),
                old_value: 0.5,
                new_value: 0.7,
            },
            "ns",
        );
        let other = make_envelope(
            MemoryEvent::Forgotten {
                id: MemoryId::new(),
            },
            "ns",
        );

        assert!(filter.matches(&above));
        assert!(!filter.matches(&below));
        assert!(!filter.matches(&other));
    }

    #[test]
    fn filter_layers_match_actual_event_layer() {
        let filter = WatchFilter::Layers(vec![Layer::Procedural]);

        let matching = make_envelope(
            MemoryEvent::ProceduralCreated {
                id: MemoryId::new(),
                procedure_name: "deploy-to-staging".to_string(),
            },
            "ns",
        );
        let non_matching = make_envelope(
            MemoryEvent::EpisodeCreated {
                id: MemoryId::new(),
                content_preview: "deploy-to-staging".to_string(),
            },
            "ns",
        );

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&non_matching));
    }

    #[test]
    fn filter_contradictions_matches_detected_events() {
        let filter = WatchFilter::Contradictions;

        let contradiction = make_envelope(
            MemoryEvent::ContradictionDetected {
                memory_a: MemoryId::new(),
                memory_b: MemoryId::new(),
                confidence: 0.92,
            },
            "ns",
        );
        let other = make_envelope(
            MemoryEvent::Forgotten {
                id: MemoryId::new(),
            },
            "ns",
        );

        assert!(filter.matches(&contradiction));
        assert!(!filter.matches(&other));
    }

    #[test]
    fn filter_event_types() {
        let filter = WatchFilter::EventTypes(vec![
            "episode_created".to_string(),
            "semantic_created".to_string(),
        ]);

        let ep = make_envelope(
            MemoryEvent::EpisodeCreated {
                id: MemoryId::new(),
                content_preview: "test".to_string(),
            },
            "ns",
        );
        let sem = make_envelope(
            MemoryEvent::SemanticCreated {
                id: MemoryId::new(),
                concept_name: "test".to_string(),
            },
            "ns",
        );
        let other = make_envelope(
            MemoryEvent::Forgotten {
                id: MemoryId::new(),
            },
            "ns",
        );

        assert!(filter.matches(&ep));
        assert!(filter.matches(&sem));
        assert!(!filter.matches(&other));
    }

    #[test]
    fn filter_all_of_requires_every_child_to_match() {
        let filter = WatchFilter::all_of(vec![
            WatchFilter::Namespace("shared".to_string()),
            WatchFilter::Entities(vec!["auth".to_string()]),
        ]);

        let matching = make_envelope(
            MemoryEvent::EpisodeCreated {
                id: MemoryId::new(),
                content_preview: "auth rollout completed".to_string(),
            },
            "shared",
        );
        let wrong_namespace = make_envelope(
            MemoryEvent::EpisodeCreated {
                id: MemoryId::new(),
                content_preview: "auth rollout completed".to_string(),
            },
            "private",
        );
        let wrong_entity = make_envelope(
            MemoryEvent::EpisodeCreated {
                id: MemoryId::new(),
                content_preview: "recipe rollout completed".to_string(),
            },
            "shared",
        );

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&wrong_namespace));
        assert!(!filter.matches(&wrong_entity));
    }

    #[test]
    fn filter_validate_allowed_namespaces_rejects_unauthorized_reference() {
        let filter = WatchFilter::Namespace("private:agent_a".to_string());
        let agent_b = hirn_core::types::AgentId::new("agent_b").unwrap();
        let allowed_namespaces = [Namespace::shared(), Namespace::private_for(&agent_b)];

        let result = filter.validate_allowed_namespaces(&allowed_namespaces);
        assert!(result.is_err());
    }

    #[test]
    fn multiple_subscribers_independent() {
        let (tx, _) = broadcast::channel::<EventEnvelope>(16);

        let sub1 = WatchSubscription::new(tx.subscribe(), WatchFilter::All);
        let sub2 =
            WatchSubscription::new(tx.subscribe(), WatchFilter::Namespace("shared".to_string()));

        // Both created — dropping one doesn't affect the other.
        drop(sub1);
        assert!(matches!(sub2.filter, WatchFilter::Namespace(_)));
    }

    #[tokio::test]
    async fn subscription_receives_filtered_events() {
        let (tx, _) = broadcast::channel::<EventEnvelope>(16);

        let mut sub =
            WatchSubscription::new(tx.subscribe(), WatchFilter::Namespace("target".to_string()));

        // Send matching and non-matching events.
        let matching = make_envelope(
            MemoryEvent::EpisodeCreated {
                id: MemoryId::new(),
                content_preview: "test".to_string(),
            },
            "target",
        );
        let non_matching = make_envelope(
            MemoryEvent::Forgotten {
                id: MemoryId::new(),
            },
            "other",
        );

        tx.send(non_matching).unwrap();
        tx.send(matching.clone()).unwrap();

        let received = sub.next().await.unwrap();
        assert_eq!(received.namespace, "target");
    }

    #[tokio::test]
    async fn subscriber_drop_no_error_on_others() {
        let (tx, _rx) = broadcast::channel::<EventEnvelope>(16);

        let sub1 = WatchSubscription::new(tx.subscribe(), WatchFilter::All);
        let mut sub2 = WatchSubscription::new(tx.subscribe(), WatchFilter::All);

        drop(sub1);

        // sub2 should still work fine.
        let env = make_envelope(
            MemoryEvent::Forgotten {
                id: MemoryId::new(),
            },
            "ns",
        );
        tx.send(env).unwrap();

        let received = sub2.next().await.unwrap();
        assert_eq!(received.event.event_type(), "forgotten");
    }
}
