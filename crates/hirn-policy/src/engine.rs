//! Cedar-based authorization engine for hirn.
//!
//! Provides fine-grained RBAC + ABAC authorization using the Cedar policy
//! language. The [`PolicyEngine`] wraps Cedar's `Authorizer`, `PolicySet`,
//! `Schema`, and entity store to evaluate authorization requests against
//! loaded policies.
//!
//! # Feature gating
//!
//! This module requires the `cedar` feature flag. When the feature is
//! disabled, [`PolicyEngine::open_mode`] returns an engine that permits
//! all requests (useful for development and testing).
//!
//! # Entity model
//!
//! ```text
//! Agent ∈ Team ∈ Organization
//! Namespace ∈ Realm
//! ```
//!
//! Eighteen actions: `remember`, `correct`, `supersede`, `merge`,
//! `retract`, `purge`, `recall`, `think`, `forget`, `consolidate`,
//! `watch`, `connect`, `execute`, `admin`, `recall_raw_text`, `read`,
//! `write`, `delete`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

#[cfg(feature = "cedar")]
use cedar_policy::{
    Authorizer, Context, Decision as CedarDecision, Entities, Entity, EntityId, EntityTypeName,
    EntityUid, PolicySet, Request, Schema, ValidationMode,
};

use crate::error::PolicyError;

/// The default Cedar schema shipped with hirn.
pub const DEFAULT_SCHEMA: &str = include_str!("cedar/hirn.cedarschema");

/// The default open-mode policy (permit all).
pub const DEFAULT_OPEN_POLICY: &str = include_str!("cedar/default.cedar");

/// Hirn authorization actions mapped to Cedar action names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Remember,
    Correct,
    Supersede,
    Merge,
    Retract,
    Purge,
    Recall,
    Think,
    Forget,
    Consolidate,
    Watch,
    Connect,
    Execute,
    Admin,
    RecallRawText,
    Read,
    Write,
    Delete,
}

impl Action {
    /// Cedar action entity ID string (matches the schema action names).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Remember => "remember",
            Self::Correct => "correct",
            Self::Supersede => "supersede",
            Self::Merge => "merge",
            Self::Retract => "retract",
            Self::Purge => "purge",
            Self::Recall => "recall",
            Self::Think => "think",
            Self::Forget => "forget",
            Self::Consolidate => "consolidate",
            Self::Watch => "watch",
            Self::Connect => "connect",
            Self::Execute => "execute",
            Self::Admin => "admin",
            Self::RecallRawText => "recall_raw_text",
            Self::Read => "read",
            Self::Write => "write",
            Self::Delete => "delete",
        }
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Action {
    type Err = PolicyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "remember" => Ok(Self::Remember),
            "correct" => Ok(Self::Correct),
            "supersede" => Ok(Self::Supersede),
            "merge" => Ok(Self::Merge),
            "retract" => Ok(Self::Retract),
            "purge" => Ok(Self::Purge),
            "recall" => Ok(Self::Recall),
            "think" => Ok(Self::Think),
            "forget" => Ok(Self::Forget),
            "consolidate" => Ok(Self::Consolidate),
            "watch" => Ok(Self::Watch),
            "connect" => Ok(Self::Connect),
            "execute" => Ok(Self::Execute),
            "admin" => Ok(Self::Admin),
            "recall_raw_text" => Ok(Self::RecallRawText),
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "delete" => Ok(Self::Delete),
            _ => Err(PolicyError::InvalidAction(s.to_string())),
        }
    }
}

/// Result of an authorization check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthzDecision {
    /// Whether the request is allowed.
    pub allowed: bool,
    /// IDs of policies that contributed to the decision.
    pub policy_ids: Vec<String>,
    /// Human-readable reasons (from Cedar diagnostics).
    pub reasons: Vec<String>,
    /// Error messages from Cedar evaluation (if any).
    pub errors: Vec<String>,
}

impl AuthzDecision {
    /// Create an Allow decision with no diagnostics.
    pub fn allow() -> Self {
        Self {
            allowed: true,
            policy_ids: Vec::new(),
            reasons: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// Create a Deny decision with a reason.
    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            policy_ids: Vec::new(),
            reasons: vec![reason.into()],
            errors: Vec::new(),
        }
    }
}

/// An authorization request to evaluate.
#[derive(Debug, Clone)]
pub struct AuthzRequest {
    /// The agent performing the action.
    pub agent_id: String,
    /// The action being performed.
    pub action: Action,
    /// Target realm.
    pub realm: String,
    /// Target namespace (empty string means realm-level).
    pub namespace: String,
}

/// An entity registered in the policy engine's entity store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityKind {
    Agent {
        reputation: i64,
        created_at: String,
        teams: Vec<String>,
    },
    Team {
        description: String,
        organization: Option<String>,
    },
    Organization {
        description: String,
    },
    Realm {
        description: String,
    },
    Namespace {
        classification: String,
        realm: String,
    },
    MemoryLayer {
        description: String,
    },
    Operation {
        description: String,
    },
    Tool {
        description: String,
    },
}

/// Thread-safe Cedar policy engine.
///
/// Wraps the Cedar `Authorizer` + `PolicySet` + `Schema` + entity store.
/// All state is behind an `RwLock` so policies and entities can be updated
/// at runtime without restart.
pub struct PolicyEngine {
    inner: Arc<RwLock<PolicyEngineInner>>,
}

struct PolicyEngineInner {
    /// Whether Cedar is actually evaluating policies (false = open mode).
    enabled: bool,
    /// Registered entities keyed by `"Type::\"id\""` form.
    entities: HashMap<String, EntityKind>,
    /// Raw policy text per source (filename → text).
    policy_sources: HashMap<String, String>,
    /// The parsed Cedar schema text.
    schema_text: String,
    /// Cached Cedar objects (rebuilt on policy/entity changes).
    #[cfg(feature = "cedar")]
    cedar: Option<CedarState>,
}

#[derive(Clone)]
struct PolicyEngineDraft {
    enabled: bool,
    entities: HashMap<String, EntityKind>,
    policy_sources: HashMap<String, String>,
    schema_text: String,
}

impl From<&PolicyEngineInner> for PolicyEngineDraft {
    fn from(inner: &PolicyEngineInner) -> Self {
        Self {
            enabled: inner.enabled,
            entities: inner.entities.clone(),
            policy_sources: inner.policy_sources.clone(),
            schema_text: inner.schema_text.clone(),
        }
    }
}

#[cfg(feature = "cedar")]
struct CedarState {
    schema: Schema,
    policy_set: PolicySet,
    entities: Entities,
}

impl PolicyEngine {
    /// Create a new policy engine from schema and policy files.
    ///
    /// Validates the schema and all policies at construction time.
    /// Returns an error if any schema or policy is invalid.
    pub fn new(schema_text: &str, policies: &[(&str, &str)]) -> Result<Self, PolicyError> {
        let mut policy_sources = HashMap::new();
        for &(name, text) in policies {
            policy_sources.insert(name.to_string(), text.to_string());
        }

        let inner = PolicyEngineInner {
            enabled: true,
            entities: HashMap::new(),
            policy_sources,
            schema_text: schema_text.to_string(),
            #[cfg(feature = "cedar")]
            cedar: None,
        };

        let engine = Self {
            inner: Arc::new(RwLock::new(inner)),
        };

        #[cfg(feature = "cedar")]
        engine.rebuild_cedar()?;

        Ok(engine)
    }

    /// Create a policy engine in open mode: all requests are allowed.
    ///
    /// Used when the `cedar` feature is disabled or for development/testing.
    ///
    /// # Warning
    ///
    /// **Open mode is unsafe for production.** Every authorization request is
    /// permitted without any policy evaluation.  A `tracing::error!` is emitted
    /// at construction time so that production monitoring surfaces the
    /// misconfiguration.  Configure `policies_dir` to enable Cedar evaluation.
    #[must_use]
    pub fn open_mode() -> Self {
        tracing::error!(
            "PolicyEngine::open_mode — ALL authorization requests PERMITTED without evaluation. \
             This is NOT safe for production. Set a `policies_dir` to enable Cedar policy enforcement."
        );
        let inner = PolicyEngineInner {
            enabled: false,
            entities: HashMap::new(),
            policy_sources: HashMap::new(),
            schema_text: String::new(),
            #[cfg(feature = "cedar")]
            cedar: None,
        };

        Self {
            inner: Arc::new(RwLock::new(inner)),
        }
    }

    /// Load policies from a brain directory.
    ///
    /// Reads `{brain_dir}/policies/hirn.cedarschema` (or uses default schema)
    /// and all `*.cedar` files in `{brain_dir}/policies/`.
    ///
    /// Fails closed when no policy files are present. Use
    /// [`Self::load_from_brain_insecure_dev_mode`] to explicitly opt into the
    /// built-in permit-all development policy.
    pub fn load_from_brain(brain_dir: &Path) -> Result<Self, PolicyError> {
        Self::load_from_brain_inner(brain_dir, false)
    }

    /// Load policies from a brain directory, permitting the built-in default
    /// open policy when no `*.cedar` files are present.
    ///
    /// This is intended only for explicit development/test posture. A
    /// `tracing::error!` is emitted to surface this misconfiguration.
    pub fn load_from_brain_insecure_dev_mode(brain_dir: &Path) -> Result<Self, PolicyError> {
        tracing::error!(
            brain_dir = %brain_dir.display(),
            "PolicyEngine::load_from_brain_insecure_dev_mode — falling back to OPEN mode when \
             no Cedar policies are found. This is NOT safe for production."
        );
        Self::load_from_brain_inner(brain_dir, true)
    }

    fn load_from_brain_inner(
        brain_dir: &Path,
        allow_default_open_policy: bool,
    ) -> Result<Self, PolicyError> {
        let policies_dir = brain_dir.join("policies");

        let schema_path = policies_dir.join("hirn.cedarschema");
        let schema_text = if schema_path.exists() {
            std::fs::read_to_string(&schema_path).map_err(|e| PolicyError::Io {
                path: schema_path.display().to_string(),
                reason: e.to_string(),
            })?
        } else {
            DEFAULT_SCHEMA.to_string()
        };

        let mut policy_files: Vec<(String, String)> = Vec::new();
        if policies_dir.exists() {
            let entries = std::fs::read_dir(&policies_dir).map_err(|e| PolicyError::Io {
                path: policies_dir.display().to_string(),
                reason: e.to_string(),
            })?;
            for entry in entries {
                let entry = entry.map_err(|e| PolicyError::Io {
                    path: policies_dir.display().to_string(),
                    reason: e.to_string(),
                })?;
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "cedar") {
                    let name = path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();
                    let text = std::fs::read_to_string(&path).map_err(|e| PolicyError::Io {
                        path: path.display().to_string(),
                        reason: e.to_string(),
                    })?;
                    policy_files.push((name, text));
                }
            }
        }

        if policy_files.is_empty() {
            if !allow_default_open_policy {
                return Err(PolicyError::MissingPolicies {
                    path: policies_dir.display().to_string(),
                });
            }
            policy_files.push(("default.cedar".to_string(), DEFAULT_OPEN_POLICY.to_string()));
        }

        let refs: Vec<(&str, &str)> = policy_files
            .iter()
            .map(|(n, t)| (n.as_str(), t.as_str()))
            .collect();

        Self::new(&schema_text, &refs)
    }

    /// Check whether this engine is in open mode (all requests allowed).
    #[must_use]
    pub fn is_open_mode(&self) -> bool {
        !self.inner.read().enabled
    }

    /// Whether Cedar policy evaluation is active (not open mode).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.inner.read().enabled
    }

    /// Returns the number of loaded policies.
    #[must_use]
    pub fn policy_count(&self) -> usize {
        #[cfg(feature = "cedar")]
        {
            let guard = self.inner.read();
            guard
                .cedar
                .as_ref()
                .map_or(0, |c| c.policy_set.policies().count())
        }
        #[cfg(not(feature = "cedar"))]
        {
            0
        }
    }

    /// Returns the number of registered entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.inner.read().entities.len()
    }

    /// List all registered namespace IDs and their associated realms.
    #[must_use]
    pub fn registered_namespaces(&self) -> Vec<(String, String)> {
        let guard = self.inner.read();
        guard
            .entities
            .iter()
            .filter_map(|(key, kind)| {
                if let EntityKind::Namespace { realm, .. } = kind {
                    let id = key
                        .strip_prefix("Hirn::Namespace::\"")
                        .and_then(|s| s.strip_suffix('"'))
                        .unwrap_or(key);
                    Some((id.to_string(), realm.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    fn update_state<R>(
        &self,
        mutate: impl FnOnce(&mut PolicyEngineDraft) -> R,
    ) -> Result<R, PolicyError> {
        let mut guard = self.inner.write();
        let mut draft = PolicyEngineDraft::from(&*guard);
        let result = mutate(&mut draft);

        #[cfg(feature = "cedar")]
        let cedar = if draft.enabled {
            Some(Self::build_cedar_state(
                &draft.schema_text,
                &draft.policy_sources,
                &draft.entities,
            )?)
        } else {
            None
        };

        guard.enabled = draft.enabled;
        guard.entities = draft.entities;
        guard.policy_sources = draft.policy_sources;
        guard.schema_text = draft.schema_text;

        #[cfg(feature = "cedar")]
        {
            guard.cedar = cedar;
        }

        Ok(result)
    }

    /// List all policy source names and their raw Cedar text.
    #[must_use]
    pub fn list_policies(&self) -> Vec<(String, String)> {
        let guard = self.inner.read();
        let mut policies: Vec<(String, String)> = guard
            .policy_sources
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        policies.sort_by(|a, b| a.0.cmp(&b.0));
        policies
    }

    /// Authorize a request against loaded policies.
    ///
    /// Returns an [`AuthzDecision`] indicating whether the request is allowed
    /// or denied, along with diagnostic information.
    pub fn authorize(&self, request: &AuthzRequest) -> AuthzDecision {
        let guard = self.inner.read();

        if !guard.enabled {
            return AuthzDecision::allow();
        }

        #[cfg(feature = "cedar")]
        {
            self.authorize_cedar(&guard, request)
        }
        #[cfg(not(feature = "cedar"))]
        {
            let _ = &guard;
            let _ = request;
            AuthzDecision::allow()
        }
    }

    /// Resolve which namespaces an agent can access for a given action.
    ///
    /// Returns `None` if engine is in open mode (permit all).
    /// Returns `Some(vec![...])` with allowed namespace IDs otherwise.
    pub fn allowed_namespaces_for(&self, agent_id: &str, action: Action) -> Option<Vec<String>> {
        if !self.is_enabled() {
            return None;
        }

        let namespaces = self.registered_namespaces();
        let mut allowed = Vec::new();
        for (ns_id, realm) in &namespaces {
            let decision = self.authorize(&AuthzRequest {
                agent_id: agent_id.to_string(),
                action,
                realm: realm.clone(),
                namespace: ns_id.clone(),
            });
            if decision.allowed {
                allowed.push(ns_id.clone());
            }
        }
        Some(allowed)
    }

    // ── Entity management ────────────────────────────────────────────

    /// Register an agent entity.
    pub fn register_agent(
        &self,
        agent_id: &str,
        reputation: i64,
        created_at: &str,
        teams: &[&str],
    ) -> Result<(), PolicyError> {
        let key = format!("Hirn::Agent::\"{}\"", agent_id);
        let entity = EntityKind::Agent {
            reputation,
            created_at: created_at.to_string(),
            teams: teams.iter().map(|s| (*s).to_string()).collect(),
        };
        self.update_state(move |draft| {
            draft.entities.insert(key, entity);
        })
    }

    /// Register a team entity.
    pub fn register_team(
        &self,
        team_id: &str,
        description: &str,
        organization: Option<&str>,
    ) -> Result<(), PolicyError> {
        let key = format!("Hirn::Team::\"{}\"", team_id);
        let entity = EntityKind::Team {
            description: description.to_string(),
            organization: organization.map(String::from),
        };
        self.update_state(move |draft| {
            draft.entities.insert(key, entity);
        })
    }

    /// Register an organization entity.
    pub fn register_organization(
        &self,
        org_id: &str,
        description: &str,
    ) -> Result<(), PolicyError> {
        let key = format!("Hirn::Organization::\"{}\"", org_id);
        let entity = EntityKind::Organization {
            description: description.to_string(),
        };
        self.update_state(move |draft| {
            draft.entities.insert(key, entity);
        })
    }

    /// Register a realm entity.
    pub fn register_realm(&self, realm_id: &str, description: &str) -> Result<(), PolicyError> {
        let key = format!("Hirn::Realm::\"{}\"", realm_id);
        let entity = EntityKind::Realm {
            description: description.to_string(),
        };
        self.update_state(move |draft| {
            draft.entities.insert(key, entity);
        })
    }

    /// Register a namespace entity.
    pub fn register_namespace(
        &self,
        namespace_id: &str,
        classification: &str,
        realm: &str,
    ) -> Result<(), PolicyError> {
        let key = format!("Hirn::Namespace::\"{}\"", namespace_id);
        let entity = EntityKind::Namespace {
            classification: classification.to_string(),
            realm: realm.to_string(),
        };
        self.update_state(move |draft| {
            draft.entities.insert(key, entity);
        })
    }

    /// Register a memory layer entity (Working, Episodic, Semantic, Procedural).
    pub fn register_memory_layer(
        &self,
        layer_id: &str,
        description: &str,
    ) -> Result<(), PolicyError> {
        let key = format!("Hirn::MemoryLayer::\"{}\"", layer_id);
        let entity = EntityKind::MemoryLayer {
            description: description.to_string(),
        };
        self.update_state(move |draft| {
            draft.entities.insert(key, entity);
        })
    }

    /// Register an operation entity (Recall, Think, Remember, etc.).
    pub fn register_operation(
        &self,
        operation_id: &str,
        description: &str,
    ) -> Result<(), PolicyError> {
        let key = format!("Hirn::Operation::\"{}\"", operation_id);
        let entity = EntityKind::Operation {
            description: description.to_string(),
        };
        self.update_state(move |draft| {
            draft.entities.insert(key, entity);
        })
    }

    /// Register a tool entity for MCP tool-level access control.
    pub fn register_tool(&self, tool_id: &str, description: &str) -> Result<(), PolicyError> {
        let key = format!("Hirn::Tool::\"{}\"", tool_id);
        let entity = EntityKind::Tool {
            description: description.to_string(),
        };
        self.update_state(move |draft| {
            draft.entities.insert(key, entity);
        })
    }

    /// Remove an entity by its Cedar key (e.g. `Hirn::Agent::"agent-007"`).
    pub fn remove_entity(&self, key: &str) -> Result<bool, PolicyError> {
        self.update_state(|draft| draft.entities.remove(key).is_some())
    }

    // ── Policy management ────────────────────────────────────────────

    /// Add or replace a policy source.
    pub fn add_policy(&self, name: &str, policy_text: &str) -> Result<(), PolicyError> {
        self.add_policies(&[(name, policy_text)])
    }

    /// Atomically add or replace multiple policy sources.
    pub fn add_policies(&self, policies: &[(&str, &str)]) -> Result<(), PolicyError> {
        self.update_state(|draft| {
            for &(name, text) in policies {
                draft
                    .policy_sources
                    .insert(name.to_string(), text.to_string());
            }
        })
    }

    /// Remove a policy source by name.
    pub fn remove_policy(&self, name: &str) -> Result<bool, PolicyError> {
        self.update_state(|draft| draft.policy_sources.remove(name).is_some())
    }

    /// Validate the current schema against all loaded policies.
    pub fn validate(&self) -> Vec<String> {
        #[cfg(feature = "cedar")]
        {
            let guard = self.inner.read();
            if let Some(cedar) = &guard.cedar {
                let validator = cedar_policy::Validator::new(cedar.schema.clone());
                let result = validator.validate(&cedar.policy_set, ValidationMode::default());
                let mut messages = Vec::new();
                for note in result.validation_errors() {
                    messages.push(format!("error: {note}"));
                }
                for note in result.validation_warnings() {
                    messages.push(format!("warning: {note}"));
                }
                messages
            } else {
                Vec::new()
            }
        }
        #[cfg(not(feature = "cedar"))]
        {
            Vec::new()
        }
    }

    /// Save the current policies and schema to a brain directory.
    pub fn save_to_brain(&self, brain_dir: &Path) -> Result<(), PolicyError> {
        let policies_dir = brain_dir.join("policies");
        std::fs::create_dir_all(&policies_dir).map_err(|e| PolicyError::Io {
            path: policies_dir.display().to_string(),
            reason: e.to_string(),
        })?;

        let guard = self.inner.read();

        let schema_path = policies_dir.join("hirn.cedarschema");
        std::fs::write(&schema_path, &guard.schema_text).map_err(|e| PolicyError::Io {
            path: schema_path.display().to_string(),
            reason: e.to_string(),
        })?;

        for (name, text) in &guard.policy_sources {
            let policy_path = policies_dir.join(name);
            std::fs::write(&policy_path, text).map_err(|e| PolicyError::Io {
                path: policy_path.display().to_string(),
                reason: e.to_string(),
            })?;
        }

        Ok(())
    }

    // ── Cedar internals ──────────────────────────────────────────────

    #[cfg(feature = "cedar")]
    fn rebuild_cedar(&self) -> Result<(), PolicyError> {
        let mut guard = self.inner.write();

        guard.cedar = if guard.enabled {
            Some(Self::build_cedar_state(
                &guard.schema_text,
                &guard.policy_sources,
                &guard.entities,
            )?)
        } else {
            None
        };

        Ok(())
    }

    #[cfg(feature = "cedar")]
    fn build_cedar_state(
        schema_text: &str,
        policy_sources: &HashMap<String, String>,
        entities: &HashMap<String, EntityKind>,
    ) -> Result<CedarState, PolicyError> {
        let schema = schema_text
            .parse::<Schema>()
            .map_err(|e| PolicyError::SchemaInvalid(format!("{e}")))?;

        let combined_text: String = policy_sources
            .iter()
            .map(|(name, text)| format!("// source: {name}\n{text}\n"))
            .collect();

        let policy_set =
            combined_text
                .parse::<PolicySet>()
                .map_err(|e| PolicyError::PolicyInvalid {
                    name: "combined".to_string(),
                    detail: format!("{e}"),
                })?;

        let entities = Self::build_entities(entities, &schema)?;

        Ok(CedarState {
            schema,
            policy_set,
            entities,
        })
    }

    #[cfg(feature = "cedar")]
    fn authorize_cedar(&self, guard: &PolicyEngineInner, request: &AuthzRequest) -> AuthzDecision {
        let cedar = match &guard.cedar {
            Some(c) => c,
            None => return AuthzDecision::deny("policy engine not initialized"),
        };

        let principal = EntityUid::from_type_name_and_id(
            Self::parse_type_name("Hirn::Agent"),
            EntityId::new(request.agent_id.clone()),
        );

        let action = EntityUid::from_type_name_and_id(
            Self::parse_type_name("Hirn::Action"),
            EntityId::new(request.action.as_str()),
        );

        let resource = if request.namespace.is_empty() {
            EntityUid::from_type_name_and_id(
                Self::parse_type_name("Hirn::Realm"),
                EntityId::new(request.realm.clone()),
            )
        } else {
            EntityUid::from_type_name_and_id(
                Self::parse_type_name("Hirn::Namespace"),
                EntityId::new(request.namespace.clone()),
            )
        };

        let context = Context::empty();

        let cedar_request =
            match Request::new(principal, action, resource, context, Some(&cedar.schema)) {
                Ok(r) => r,
                Err(e) => return AuthzDecision::deny(format!("invalid request: {e}")),
            };

        let authorizer = Authorizer::new();
        let response = authorizer.is_authorized(&cedar_request, &cedar.policy_set, &cedar.entities);

        let mut decision = AuthzDecision {
            allowed: response.decision() == CedarDecision::Allow,
            policy_ids: response
                .diagnostics()
                .reason()
                .map(|id| id.to_string())
                .collect(),
            reasons: Vec::new(),
            errors: response
                .diagnostics()
                .errors()
                .map(|e| e.to_string())
                .collect(),
        };

        if !decision.allowed {
            decision.reasons.push(format!(
                "denied: {} cannot {} on {}",
                request.agent_id,
                request.action,
                if request.namespace.is_empty() {
                    &request.realm
                } else {
                    &request.namespace
                }
            ));
        }

        decision
    }

    #[cfg(feature = "cedar")]
    fn parse_type_name(name: &str) -> EntityTypeName {
        name.parse().expect("valid Cedar entity type name")
    }

    #[cfg(feature = "cedar")]
    fn build_entities(
        entity_map: &HashMap<String, EntityKind>,
        schema: &Schema,
    ) -> Result<Entities, PolicyError> {
        let mut entities_vec: Vec<Entity> = Vec::new();

        for (key, kind) in entity_map {
            let entity = Self::build_entity(key, kind)?;
            entities_vec.push(entity);
        }

        Entities::from_entities(entities_vec, Some(schema))
            .map_err(|e| PolicyError::EntityInvalid(format!("{e}")))
    }

    #[cfg(feature = "cedar")]
    fn build_entity(key: &str, kind: &EntityKind) -> Result<Entity, PolicyError> {
        use cedar_policy::RestrictedExpression;

        let (type_str, id_str) = Self::parse_entity_key(key)?;
        let uid = EntityUid::from_type_name_and_id(
            Self::parse_type_name(type_str),
            EntityId::new(id_str),
        );

        match kind {
            EntityKind::Agent {
                reputation,
                created_at,
                teams,
            } => {
                let parents: Vec<EntityUid> = teams
                    .iter()
                    .map(|t| {
                        EntityUid::from_type_name_and_id(
                            Self::parse_type_name("Hirn::Team"),
                            EntityId::new(t.as_str()),
                        )
                    })
                    .collect();

                let attrs = HashMap::from([
                    (
                        "reputation".to_string(),
                        RestrictedExpression::new_long(*reputation),
                    ),
                    (
                        "created_at".to_string(),
                        RestrictedExpression::new_string(created_at.clone()),
                    ),
                ]);

                Ok(Entity::new(uid, attrs, parents.into_iter().collect())
                    .map_err(|e| PolicyError::EntityInvalid(format!("{e}")))?)
            }
            EntityKind::Team {
                description,
                organization,
            } => {
                let parents: Vec<EntityUid> = organization
                    .iter()
                    .map(|o| {
                        EntityUid::from_type_name_and_id(
                            Self::parse_type_name("Hirn::Organization"),
                            EntityId::new(o.as_str()),
                        )
                    })
                    .collect();

                let attrs = HashMap::from([(
                    "description".to_string(),
                    RestrictedExpression::new_string(description.clone()),
                )]);

                Ok(Entity::new(uid, attrs, parents.into_iter().collect())
                    .map_err(|e| PolicyError::EntityInvalid(format!("{e}")))?)
            }
            EntityKind::Organization { description } => {
                let attrs = HashMap::from([(
                    "description".to_string(),
                    RestrictedExpression::new_string(description.clone()),
                )]);

                Ok(Entity::new(uid, attrs, [].into_iter().collect())
                    .map_err(|e| PolicyError::EntityInvalid(format!("{e}")))?)
            }
            EntityKind::Realm { description } => {
                let attrs = HashMap::from([(
                    "description".to_string(),
                    RestrictedExpression::new_string(description.clone()),
                )]);

                Ok(Entity::new(uid, attrs, [].into_iter().collect())
                    .map_err(|e| PolicyError::EntityInvalid(format!("{e}")))?)
            }
            EntityKind::Namespace {
                classification,
                realm,
            } => {
                let parents = vec![EntityUid::from_type_name_and_id(
                    Self::parse_type_name("Hirn::Realm"),
                    EntityId::new(realm.as_str()),
                )];

                let attrs = HashMap::from([(
                    "classification".to_string(),
                    RestrictedExpression::new_string(classification.clone()),
                )]);

                Ok(Entity::new(uid, attrs, parents.into_iter().collect())
                    .map_err(|e| PolicyError::EntityInvalid(format!("{e}")))?)
            }
            EntityKind::MemoryLayer { description }
            | EntityKind::Operation { description }
            | EntityKind::Tool { description } => {
                let attrs = HashMap::from([(
                    "description".to_string(),
                    RestrictedExpression::new_string(description.clone()),
                )]);

                Ok(Entity::new(uid, attrs, [].into_iter().collect())
                    .map_err(|e| PolicyError::EntityInvalid(format!("{e}")))?)
            }
        }
    }

    /// Parse `Hirn::Agent::"agent-007"` → `("Hirn::Agent", "agent-007")`
    #[cfg(feature = "cedar")]
    fn parse_entity_key(key: &str) -> Result<(&str, &str), PolicyError> {
        if let Some(idx) = key.rfind("::\"") {
            let type_str = &key[..idx];
            let id_raw = &key[idx + 2..];
            let id_str = id_raw.trim_matches('"');
            Ok((type_str, id_str))
        } else {
            Err(PolicyError::EntityInvalid(format!(
                "invalid entity key format: {key}"
            )))
        }
    }
}

impl Clone for PolicyEngine {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl std::fmt::Debug for PolicyEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.inner.read();
        f.debug_struct("PolicyEngine")
            .field("enabled", &guard.enabled)
            .field("entities", &guard.entities.len())
            .field("policy_sources", &guard.policy_sources.len())
            .finish()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_schema_parses() {
        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("default.cedar", DEFAULT_OPEN_POLICY)]);
        assert!(engine.is_ok(), "default schema should parse: {engine:?}");
    }

    #[test]
    fn invalid_schema_returns_error() {
        let bad_schema = "this is not a valid cedar schema!!!";
        let result = PolicyEngine::new(bad_schema, &[("default.cedar", DEFAULT_OPEN_POLICY)]);
        assert!(result.is_err());
        match result.unwrap_err() {
            PolicyError::SchemaInvalid(msg) => {
                assert!(!msg.is_empty());
            }
            other => panic!("expected SchemaInvalid, got: {other:?}"),
        }
    }

    #[test]
    fn schema_covers_all_actions() {
        for action in [
            "remember",
            "correct",
            "supersede",
            "merge",
            "retract",
            "purge",
            "recall",
            "think",
            "forget",
            "consolidate",
            "watch",
            "connect",
            "execute",
            "admin",
            "recall_raw_text",
            "read",
            "write",
            "delete",
        ] {
            assert!(
                DEFAULT_SCHEMA.contains(action),
                "schema should include action '{action}'"
            );
        }
    }

    #[test]
    fn action_strings_round_trip() {
        for action in [
            "remember",
            "correct",
            "supersede",
            "merge",
            "retract",
            "purge",
            "recall",
            "think",
            "forget",
            "consolidate",
            "watch",
            "connect",
            "execute",
            "admin",
            "recall_raw_text",
            "read",
            "write",
            "delete",
        ] {
            let parsed: Action = action.parse().unwrap();
            assert_eq!(parsed.as_str(), action);
        }
    }

    #[test]
    fn open_mode_allows_everything() {
        let engine = PolicyEngine::open_mode();
        assert!(engine.is_open_mode());

        let decision = engine.authorize(&AuthzRequest {
            agent_id: "any-agent".to_string(),
            action: Action::Remember,
            realm: "any-realm".to_string(),
            namespace: String::new(),
        });
        assert!(decision.allowed);
    }

    #[test]
    fn default_open_policy_allows_everything() {
        let engine =
            PolicyEngine::new(DEFAULT_SCHEMA, &[("default.cedar", DEFAULT_OPEN_POLICY)]).unwrap();

        engine
            .register_agent("test-agent", 100, "2025-01-01T00:00:00Z", &[])
            .unwrap();
        engine.register_realm("test-realm", "Test").unwrap();

        let decision = engine.authorize(&AuthzRequest {
            agent_id: "test-agent".to_string(),
            action: Action::Remember,
            realm: "test-realm".to_string(),
            namespace: String::new(),
        });
        assert!(decision.allowed, "open policy should allow: {decision:?}");
    }

    #[test]
    fn team_policy_allows_members_denies_others() {
        let policy = r#"
            permit(
                principal in Hirn::Team::"writers",
                action == Hirn::Action::"remember",
                resource == Hirn::Realm::"production"
            );
        "#;

        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("acl.cedar", policy)]).unwrap();

        engine
            .register_team("writers", "Writer team", None)
            .unwrap();
        engine.register_realm("production", "Prod").unwrap();

        engine
            .register_agent("alice", 100, "2025-01-01T00:00:00Z", &["writers"])
            .unwrap();
        let decision = engine.authorize(&AuthzRequest {
            agent_id: "alice".to_string(),
            action: Action::Remember,
            realm: "production".to_string(),
            namespace: String::new(),
        });
        assert!(decision.allowed, "alice should be allowed: {decision:?}");

        engine
            .register_agent("bob", 100, "2025-01-01T00:00:00Z", &[])
            .unwrap();
        let decision = engine.authorize(&AuthzRequest {
            agent_id: "bob".to_string(),
            action: Action::Remember,
            realm: "production".to_string(),
            namespace: String::new(),
        });
        assert!(!decision.allowed, "bob should be denied: {decision:?}");
    }

    #[test]
    fn abac_reputation_constraint() {
        let policy = r#"
            permit(
                principal,
                action == Hirn::Action::"remember",
                resource
            ) when { principal.reputation >= 50 };
        "#;

        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("acl.cedar", policy)]).unwrap();
        engine.register_realm("test", "Test").unwrap();

        engine
            .register_agent("good-agent", 100, "2025-01-01T00:00:00Z", &[])
            .unwrap();
        let decision = engine.authorize(&AuthzRequest {
            agent_id: "good-agent".to_string(),
            action: Action::Remember,
            realm: "test".to_string(),
            namespace: String::new(),
        });
        assert!(decision.allowed, "high rep allowed: {decision:?}");

        engine
            .register_agent("bad-agent", 10, "2025-01-01T00:00:00Z", &[])
            .unwrap();
        let decision = engine.authorize(&AuthzRequest {
            agent_id: "bad-agent".to_string(),
            action: Action::Remember,
            realm: "test".to_string(),
            namespace: String::new(),
        });
        assert!(!decision.allowed, "low rep denied: {decision:?}");
    }

    #[test]
    fn save_and_load_from_brain() {
        let temp = tempfile::tempdir().unwrap();
        let brain_dir = temp.path();

        let custom_policy = r#"
            permit(
                principal in Hirn::Team::"writers",
                action == Hirn::Action::"remember",
                resource
            );
        "#;
        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("custom.cedar", custom_policy)]).unwrap();
        engine.save_to_brain(brain_dir).unwrap();

        assert!(brain_dir.join("policies/hirn.cedarschema").exists());
        assert!(brain_dir.join("policies/custom.cedar").exists());

        let loaded = PolicyEngine::load_from_brain(brain_dir).unwrap();
        assert!(loaded.policy_count() >= 1);
    }

    #[test]
    fn invalid_policy_add_rolls_back_policy_sources() {
        let engine =
            PolicyEngine::new(DEFAULT_SCHEMA, &[("default.cedar", DEFAULT_OPEN_POLICY)]).unwrap();
        let before = engine.list_policies();

        let err = engine
            .add_policy("broken.cedar", "this is not valid cedar")
            .unwrap_err();
        assert!(matches!(err, PolicyError::PolicyInvalid { .. }));
        assert_eq!(engine.list_policies(), before);
    }

    #[test]
    fn invalid_policy_remove_rolls_back_policy_sources() {
        let engine = PolicyEngine::new(
            DEFAULT_SCHEMA,
            &[
                ("default.cedar", DEFAULT_OPEN_POLICY),
                ("extra.cedar", DEFAULT_OPEN_POLICY),
            ],
        )
        .unwrap();

        {
            let mut guard = engine.inner.write();
            guard.schema_text = "this is not a valid cedar schema!!!".to_string();
        }

        let err = engine.remove_policy("extra.cedar").unwrap_err();
        assert!(matches!(err, PolicyError::SchemaInvalid(_)));
        assert!(
            engine
                .list_policies()
                .iter()
                .any(|(name, _)| name == "extra.cedar")
        );
    }

    #[test]
    fn invalid_entity_registration_rolls_back_entities() {
        let engine =
            PolicyEngine::new(DEFAULT_SCHEMA, &[("default.cedar", DEFAULT_OPEN_POLICY)]).unwrap();
        let before = engine.entity_count();

        {
            let mut guard = engine.inner.write();
            guard.schema_text = "this is not a valid cedar schema!!!".to_string();
        }

        let err = engine
            .register_namespace("candidate-ns", "public", "candidate-realm")
            .unwrap_err();
        assert!(matches!(err, PolicyError::SchemaInvalid(_)));
        assert_eq!(engine.entity_count(), before);
        assert!(engine.registered_namespaces().is_empty());
    }

    #[test]
    fn load_from_brain_without_policies_fails_closed() {
        let temp = tempfile::tempdir().unwrap();
        let err = PolicyEngine::load_from_brain(temp.path()).unwrap_err();
        assert!(matches!(err, PolicyError::MissingPolicies { .. }));
    }

    #[test]
    fn load_from_brain_insecure_dev_mode_uses_default_open_policy() {
        let temp = tempfile::tempdir().unwrap();
        let loaded = PolicyEngine::load_from_brain_insecure_dev_mode(temp.path()).unwrap();
        assert!(loaded.policy_count() >= 1);
        assert!(!loaded.is_open_mode());
    }

    #[test]
    fn allowed_namespaces_open_mode() {
        let engine = PolicyEngine::open_mode();
        let result = engine.allowed_namespaces_for("anyone", Action::Recall);
        assert!(result.is_none());
    }

    #[test]
    fn allowed_namespaces_filters() {
        let engine =
            PolicyEngine::new(DEFAULT_SCHEMA, &[("default.cedar", DEFAULT_OPEN_POLICY)]).unwrap();
        engine.register_realm("production", "prod").unwrap();
        engine
            .register_namespace("ns_a", "public", "production")
            .unwrap();
        engine
            .register_namespace("ns_b", "public", "production")
            .unwrap();
        engine
            .register_agent("agent-1", 50, "2024-01-01", &[])
            .unwrap();

        let result = engine.allowed_namespaces_for("agent-1", Action::Recall);
        assert!(result.is_some());
        let mut allowed = result.unwrap();
        allowed.sort();
        assert_eq!(allowed, vec!["ns_a", "ns_b"]);
    }

    #[test]
    fn concurrent_authorization() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(
            PolicyEngine::new(DEFAULT_SCHEMA, &[("default.cedar", DEFAULT_OPEN_POLICY)]).unwrap(),
        );

        engine
            .register_agent("concurrent-agent", 100, "2025-01-01T00:00:00Z", &[])
            .unwrap();
        engine.register_realm("test", "Test").unwrap();

        let handles: Vec<_> = (0..100)
            .map(|_| {
                let eng = Arc::clone(&engine);
                thread::spawn(move || {
                    let d = eng.authorize(&AuthzRequest {
                        agent_id: "concurrent-agent".to_string(),
                        action: Action::Recall,
                        realm: "test".to_string(),
                        namespace: String::new(),
                    });
                    assert!(d.allowed);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn schema_includes_entity_types() {
        for entity in ["MemoryLayer", "Operation", "Tool"] {
            assert!(
                DEFAULT_SCHEMA.contains(entity),
                "schema should include entity '{entity}'"
            );
        }
    }

    #[test]
    fn register_memory_layer_entity() {
        let engine =
            PolicyEngine::new(DEFAULT_SCHEMA, &[("default.cedar", DEFAULT_OPEN_POLICY)]).unwrap();
        engine
            .register_memory_layer("Episodic", "Episodic memory layer")
            .unwrap();

        let entities = engine.entity_count();
        assert!(entities >= 1, "should have at least 1 entity");
    }

    #[test]
    fn register_operation_entity() {
        let engine =
            PolicyEngine::new(DEFAULT_SCHEMA, &[("default.cedar", DEFAULT_OPEN_POLICY)]).unwrap();
        engine
            .register_operation("Recall", "Recall operation")
            .unwrap();

        let entities = engine.entity_count();
        assert!(entities >= 1, "should have at least 1 entity");
    }

    #[test]
    fn register_tool_entity() {
        let engine =
            PolicyEngine::new(DEFAULT_SCHEMA, &[("default.cedar", DEFAULT_OPEN_POLICY)]).unwrap();
        engine
            .register_tool("remember_tool", "Memory toolkit: remember")
            .unwrap();

        let entities = engine.entity_count();
        assert!(entities >= 1, "should have at least 1 entity");
    }

    #[test]
    fn namespace_scoped_permit_policy() {
        let policy = r#"
            permit(
                principal == Hirn::Agent::"agent-a",
                action == Hirn::Action::"recall",
                resource == Hirn::Namespace::"team_x"
            );
        "#;

        let engine = PolicyEngine::new(DEFAULT_SCHEMA, &[("ns.cedar", policy)]).unwrap();
        engine.register_realm("prod", "Production").unwrap();
        engine
            .register_namespace("team_x", "public", "prod")
            .unwrap();
        engine
            .register_namespace("team_y", "classified", "prod")
            .unwrap();
        engine
            .register_agent("agent-a", 50, "2025-01-01", &[])
            .unwrap();
        engine
            .register_agent("agent-b", 50, "2025-01-01", &[])
            .unwrap();

        // agent-a can recall on team_x
        let d = engine.authorize(&AuthzRequest {
            agent_id: "agent-a".to_string(),
            action: Action::Recall,
            realm: "prod".to_string(),
            namespace: "team_x".to_string(),
        });
        assert!(d.allowed, "agent-a should access team_x: {d:?}");

        // agent-a cannot recall on team_y (no matching permit)
        let d = engine.authorize(&AuthzRequest {
            agent_id: "agent-a".to_string(),
            action: Action::Recall,
            realm: "prod".to_string(),
            namespace: "team_y".to_string(),
        });
        assert!(!d.allowed, "agent-a should be denied team_y: {d:?}");

        // agent-b cannot recall on team_x (no matching permit)
        let d = engine.authorize(&AuthzRequest {
            agent_id: "agent-b".to_string(),
            action: Action::Recall,
            realm: "prod".to_string(),
            namespace: "team_x".to_string(),
        });
        assert!(!d.allowed, "agent-b should be denied team_x: {d:?}");
    }
}
