use hirn::prelude::*;

use crate::convert;
use crate::proto;

#[derive(Clone, Debug)]
pub struct WatchNamespaceScope {
    requested_namespace: Option<String>,
    token_allowed_namespaces: Option<Vec<String>>,
    private_namespace: Option<String>,
    shared_namespace: Option<String>,
}

impl WatchNamespaceScope {
    pub fn unrestricted(requested_namespace: Option<String>) -> Self {
        Self {
            requested_namespace,
            token_allowed_namespaces: None,
            private_namespace: None,
            shared_namespace: None,
        }
    }

    pub fn token_scoped(
        agent_id: &AgentId,
        requested_namespace: Option<String>,
        token_allowed_namespaces: Vec<String>,
    ) -> Self {
        Self {
            requested_namespace,
            token_allowed_namespaces: Some(token_allowed_namespaces),
            private_namespace: Some(Namespace::private_for(agent_id).as_str().to_owned()),
            shared_namespace: Some(Namespace::shared().as_str().to_owned()),
        }
    }

    fn allows_namespace(&self, namespace: Option<&Namespace>) -> bool {
        let Some(namespace) = namespace else {
            return self.requested_namespace.is_none() && self.token_allowed_namespaces.is_none();
        };

        if let Some(requested_namespace) = self.requested_namespace.as_deref() {
            if namespace.as_str() != requested_namespace {
                return false;
            }
        }

        let Some(token_allowed_namespaces) = &self.token_allowed_namespaces else {
            return true;
        };

        let private_namespace = self
            .private_namespace
            .as_deref()
            .expect("token-scoped watch filter requires a private namespace");
        let shared_namespace = self
            .shared_namespace
            .as_deref()
            .expect("token-scoped watch filter requires a shared namespace");
        let event_namespace = namespace.as_str();

        if token_allowed_namespaces.is_empty() {
            return event_namespace == private_namespace || event_namespace == shared_namespace;
        }

        token_allowed_namespaces
            .iter()
            .any(|allowed_namespace| match allowed_namespace.as_str() {
                "private" | "default" => event_namespace == private_namespace,
                "shared" => event_namespace == shared_namespace,
                other => event_namespace == other,
            })
    }
}

/// Internal watch event that is broadcast to all subscribers.
#[derive(Clone, Debug)]
pub enum WatchEvent {
    Created {
        id: MemoryId,
        layer: Layer,
        entities: Vec<String>,
        importance: f32,
        namespace: Namespace,
    },
    Updated {
        id: MemoryId,
        layer: Layer,
        entities: Vec<String>,
        importance: f32,
        namespace: Namespace,
    },
    Consolidated {
        records_processed: usize,
    },
    Conflict {
        memory_a: MemoryId,
        memory_b: MemoryId,
    },
}

impl WatchEvent {
    /// Convert to proto `WatchEvent` if it matches the subscriber's filter.
    /// Returns `None` if the event should be filtered out.
    pub fn to_proto(
        &self,
        layer_filter: &Option<Layer>,
        entities: &[String],
        min_importance: Option<f32>,
        namespace_scope: &WatchNamespaceScope,
    ) -> Option<proto::WatchEvent> {
        match self {
            WatchEvent::Created {
                id,
                layer,
                entities: event_entities,
                importance,
                namespace: event_ns,
            } => {
                if !matches_filters(
                    layer,
                    event_entities,
                    *importance,
                    event_ns,
                    layer_filter,
                    entities,
                    min_importance,
                    namespace_scope,
                ) {
                    return None;
                }
                Some(proto::WatchEvent {
                    event_type: proto::WatchEventType::Created as i32,
                    record: None,
                    timestamp: Some(convert::timestamp_to_proto(&Timestamp::now())),
                    description: Some(format!("Memory created: {id} ({layer:?})")),
                })
            }
            WatchEvent::Updated {
                id,
                layer,
                entities: event_entities,
                importance,
                namespace: event_ns,
            } => {
                if !matches_filters(
                    layer,
                    event_entities,
                    *importance,
                    event_ns,
                    layer_filter,
                    entities,
                    min_importance,
                    namespace_scope,
                ) {
                    return None;
                }
                Some(proto::WatchEvent {
                    event_type: proto::WatchEventType::Updated as i32,
                    record: None,
                    timestamp: Some(convert::timestamp_to_proto(&Timestamp::now())),
                    description: Some(format!("Memory updated: {id} ({layer:?})")),
                })
            }
            WatchEvent::Consolidated { records_processed } => {
                if !namespace_scope.allows_namespace(None) {
                    return None;
                }
                Some(proto::WatchEvent {
                    event_type: proto::WatchEventType::Consolidated as i32,
                    record: None,
                    timestamp: Some(convert::timestamp_to_proto(&Timestamp::now())),
                    description: Some(format!(
                        "Consolidation completed: {records_processed} records processed"
                    )),
                })
            }
            WatchEvent::Conflict { memory_a, memory_b } => {
                if !namespace_scope.allows_namespace(None) {
                    return None;
                }
                Some(proto::WatchEvent {
                    event_type: proto::WatchEventType::Conflict as i32,
                    record: None,
                    timestamp: Some(convert::timestamp_to_proto(&Timestamp::now())),
                    description: Some(format!("Conflict detected: {memory_a} vs {memory_b}")),
                })
            }
        }
    }
}

/// Check whether a memory event passes all subscriber filters.
fn matches_filters(
    layer: &Layer,
    event_entities: &[String],
    importance: f32,
    event_ns: &Namespace,
    layer_filter: &Option<Layer>,
    filter_entities: &[String],
    min_importance: Option<f32>,
    namespace_scope: &WatchNamespaceScope,
) -> bool {
    // Layer filter
    if let Some(filter) = layer_filter {
        if filter != layer {
            return false;
        }
    }
    // Entity filter — at least one requested entity must be present
    if !filter_entities.is_empty()
        && !filter_entities
            .iter()
            .any(|e| event_entities.iter().any(|ee| ee.eq_ignore_ascii_case(e)))
    {
        return false;
    }
    // Importance threshold
    if let Some(min) = min_importance {
        if importance < min {
            return false;
        }
    }
    namespace_scope.allows_namespace(Some(event_ns))
}
