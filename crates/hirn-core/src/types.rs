use serde::{Deserialize, Serialize};

use crate::error::HirnError;
use crate::id::MemoryId;
use crate::interner::{agent_id_interner, namespace_interner};

/// The four memory layers in the cognitive architecture (CLS theory + procedural).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layer {
    Working,
    Episodic,
    Semantic,
    /// Procedural memory — learned skills, tool-use patterns, multi-step workflows.
    /// Inspired by `CoALA` (arXiv:2309.02427) and AWM (arXiv:2409.07429).
    Procedural,
}

/// Event type for episodic records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    Conversation,
    ToolCall,
    Observation,
    Experiment,
    Error,
    Decision,
}

/// Priority level for working memory entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Lowest priority — evicted first.
    Normal = 0,
    /// Medium priority.
    High = 1,
    /// Highest priority — evicted last.
    Critical = 2,
}

/// Knowledge types for semantic records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KnowledgeType {
    /// Facts — "X is Y".
    Propositional,
    /// Rules/heuristics — "When X, do Y".
    Prescriptive,
    /// Type hierarchies — "X is-a Y".
    Taxonomic,
    /// Hypothetical inferences from dream replay.
    Inferred,
    /// Community summary from Leiden clustering.
    Community,
    /// RAPTOR hierarchical summary (Sarthi et al., 2024).
    /// Multi-level tree of clustered summaries for top-down retrieval.
    RaptorSummary,
}

/// Origin of a memory record for provenance tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Origin {
    DirectObservation,
    LlmExtraction,
    Consolidation,
    UserProvided,
    CrossAgent,
    DreamReplay,
}

/// What triggered a mutation in a memory record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MutationTrigger {
    Reconsolidation,
    Consolidation,
    Manual,
    Decay,
}

/// Cross-layer reference to a memory record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryRef {
    pub layer: Layer,
    pub id: MemoryId,
}

impl MemoryRef {
    #[must_use]
    pub const fn new(layer: Layer, id: MemoryId) -> Self {
        Self { layer, id }
    }
}

/// Edge relation types for the property graph (CONCEPT.md §6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeRelation {
    /// General association — bidirectional.
    RelatedTo,
    /// Causal relationship — directed.
    Causes,
    /// Inverse causal — directed.
    CausedBy,
    /// Derivation — directed.
    DerivedFrom,
    /// Contradiction — bidirectional.
    Contradicts,
    /// Evidential support — directed.
    Supports,
    /// Arrival-order sequence within a namespace — directed.
    TemporalNext,
    /// Composition — directed.
    PartOf,
    /// Instance of a class — directed.
    InstanceOf,
    /// Embedding similarity — bidirectional.
    SimilarTo,
    /// Suppression/inhibition — directed.
    Inhibits,
    /// F-056 FIX: Participation in an N-ary fact — directed (entity → fact node).
    ParticipatesIn,
}

impl EdgeRelation {
    /// Whether this relation type is inherently bidirectional.
    #[must_use]
    pub const fn is_bidirectional(self) -> bool {
        matches!(self, Self::RelatedTo | Self::Contradicts | Self::SimilarTo)
    }
}

/// Multi-agent namespace scoping.
///
/// Backed by an interned `u32` handle for `Copy` semantics and O(1) equality.
/// Serializes as a UTF-8 string for backward compatibility with Arrow/Lance/JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Namespace(u32);

/// The kind of namespace — determines default access rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NamespaceKind {
    /// Agent-private namespace (`private:{agent_id}`). Only the owning agent can access.
    Private,
    /// Shared namespace accessible to all agents.
    Shared,
    /// Team namespace accessible to a defined set of agents.
    Team,
    /// The default single-agent namespace.
    Default,
}

impl Namespace {
    /// Create a new namespace. Returns an error if the name is empty, contains
    /// invalid characters (only alphanumeric, underscore, colon, hyphen allowed;
    /// no leading/trailing whitespace), or is `"private:"` with no agent suffix.
    pub fn new(name: impl Into<String>) -> Result<Self, HirnError> {
        let name = name.into();
        let trimmed = name.trim();
        if trimmed.is_empty() || trimmed != name {
            return Err(HirnError::InvalidInput(format!(
                "invalid namespace: '{name}' (empty or has leading/trailing whitespace)"
            )));
        }
        // Allow alphanumeric, underscore, colon (for private:agent), hyphen.
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == ':' || c == '-')
        {
            return Err(HirnError::InvalidInput(format!(
                "invalid namespace: '{name}' (only alphanumeric, underscore, colon, hyphen allowed)"
            )));
        }
        // Reject "private:" with no agent suffix.
        if name == "private:" {
            return Err(HirnError::InvalidInput(
                "invalid namespace: 'private:' requires an agent suffix".into(),
            ));
        }
        Ok(Self(namespace_interner().try_intern(&name)?))
    }

    /// The default namespace for single-agent use.
    #[must_use]
    pub fn default_ns() -> Self {
        Self(namespace_interner().intern("default"))
    }

    /// The shared namespace accessible to all agents.
    #[must_use]
    pub fn shared() -> Self {
        Self(namespace_interner().intern("shared"))
    }

    /// Create a private namespace for a specific agent.
    #[must_use]
    pub fn private_for(agent_id: &AgentId) -> Self {
        let name = format!("private:{}", agent_id.as_str());
        // AgentId is already validated; intern() is safe here.
        Self(namespace_interner().intern(&name))
    }

    /// The kind of this namespace based on its naming convention.
    /// F-77 FIX: `default_ns()` now returns `NamespaceKind::Default` (not `Team`).
    #[must_use]
    pub fn kind(&self) -> NamespaceKind {
        let s = self.as_str();
        if s == "default" {
            NamespaceKind::Default
        } else if s.starts_with("private:") {
            NamespaceKind::Private
        } else if s == "shared" {
            NamespaceKind::Shared
        } else {
            // Team or custom namespaces
            NamespaceKind::Team
        }
    }

    /// The agent ID that owns this namespace, if it is a private namespace.
    #[must_use]
    pub fn owning_agent(&self) -> Option<AgentId> {
        self.as_str()
            .strip_prefix("private:")
            .and_then(|rest| AgentId::new(rest).ok())
    }

    /// Return the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        namespace_interner().resolve(self.0)
    }

    /// Return the raw interned integer ID.
    ///
    /// Stable within a process lifetime.  Useful for fast cache-key mixing
    /// where string hashing would be too expensive.
    #[must_use]
    #[inline]
    pub fn as_interned_id(&self) -> u32 {
        self.0
    }
}

impl Default for Namespace {
    fn default() -> Self {
        Self::default_ns()
    }
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Serialize for Namespace {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Namespace {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Namespace::new(s).map_err(serde::de::Error::custom)
    }
}

/// Agent identifier.
///
/// Backed by an interned `u32` handle for `Copy` semantics and O(1) equality.
/// Serializes as a UTF-8 string for backward compatibility.
///
/// Must match `[a-zA-Z0-9_.-]{1,128}` — safe for use in namespaces, file paths,
/// HTTP headers, and metric labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(u32);

impl AgentId {
    /// Create a well-known agent ID from a string literal.
    ///
    /// Use this for hard-coded system agent IDs (e.g., `"system"`, `"hirnql"`)
    /// where the input is known-valid at compile time.
    ///
    /// # Panics
    /// F-80 FIX: Panic is intentional — only use with known-valid string literals.
    /// Invalid input is a programming error caught at startup.
    #[must_use]
    pub fn well_known(id: &str) -> Self {
        Self::new(id).unwrap_or_else(|_| {
            panic!(
                "well-known AgentId '{id}' must be a non-empty ASCII alphanumeric string ≤128 chars"
            )
        })
    }

    /// Create a new agent ID. Returns an error if the id is empty, exceeds 128
    /// characters, or contains characters outside `[a-zA-Z0-9_.-]`.
    pub fn new(id: impl Into<String>) -> Result<Self, HirnError> {
        let id = id.into();
        if id.is_empty() || id.len() > 128 {
            return Err(HirnError::InvalidInput(format!(
                "invalid agent_id: '{id}' (must be 1-128 chars)"
            )));
        }
        if !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
        {
            return Err(HirnError::InvalidInput(format!(
                "invalid agent_id: '{id}' (only ASCII alphanumeric, underscore, dot, hyphen allowed)"
            )));
        }
        Ok(Self(agent_id_interner().try_intern(&id)?))
    }

    /// Return the underlying string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        agent_id_interner().resolve(self.0)
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Serialize for AgentId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AgentId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        AgentId::new(s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
    }

    #[test]
    fn namespace_rejects_empty() {
        assert!(Namespace::new("").is_err());
        assert!(Namespace::new("test").is_ok());
    }

    #[test]
    fn namespace_rejects_invalid_chars() {
        assert!(Namespace::new("hello world").is_err()); // space
        assert!(Namespace::new(" leading").is_err()); // leading space
        assert!(Namespace::new("trailing ").is_err()); // trailing space
        assert!(Namespace::new("a/b").is_err()); // slash
        assert!(Namespace::new("a@b").is_err()); // at sign
    }

    #[test]
    fn namespace_accepts_valid_names() {
        assert!(Namespace::new("test_ns").is_ok());
        assert!(Namespace::new("private:agent_a").is_ok());
        assert!(Namespace::new("team-backend").is_ok());
        assert!(Namespace::new("shared").is_ok());
    }

    #[test]
    fn namespace_kind_detection() {
        assert_eq!(Namespace::shared().kind(), NamespaceKind::Shared);
        let agent = AgentId::new("agent_a").unwrap();
        let private = Namespace::private_for(&agent);
        assert_eq!(private.kind(), NamespaceKind::Private);
        assert_eq!(private.owning_agent(), Some(agent));
        assert_eq!(
            Namespace::new("team_backend").unwrap().kind(),
            NamespaceKind::Team
        );
    }

    #[test]
    fn agent_id_rejects_empty() {
        assert!(AgentId::new("").is_err());
        assert!(AgentId::new("agent_a").is_ok());
    }

    #[test]
    fn agent_id_rejects_invalid_chars() {
        assert!(AgentId::new("hello world").is_err()); // space
        assert!(AgentId::new("agent/a").is_err()); // slash
        assert!(AgentId::new("agent@a").is_err()); // at sign
        assert!(AgentId::new("a:b").is_err()); // colon
        assert!(AgentId::new("\n").is_err()); // control char
    }

    #[test]
    fn agent_id_rejects_too_long() {
        let long = "a".repeat(129);
        assert!(AgentId::new(long).is_err());
        let ok = "a".repeat(128);
        assert!(AgentId::new(ok).is_ok());
    }

    #[test]
    fn agent_id_accepts_valid_names() {
        assert!(AgentId::new("agent_a").is_ok());
        assert!(AgentId::new("agent-1").is_ok());
        assert!(AgentId::new("agent.v2").is_ok());
        assert!(AgentId::new("A").is_ok());
    }

    #[test]
    fn memory_ref_serde_round_trip() {
        let r = MemoryRef::new(Layer::Episodic, MemoryId::new());
        let bytes = bincode::serialize(&r).unwrap();
        let back: MemoryRef = bincode::deserialize(&bytes).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn all_enums_serde_round_trip() {
        // Layer
        for layer in [
            Layer::Working,
            Layer::Episodic,
            Layer::Semantic,
            Layer::Procedural,
        ] {
            let bytes = bincode::serialize(&layer).unwrap();
            let back: Layer = bincode::deserialize(&bytes).unwrap();
            assert_eq!(layer, back);
        }
        // EventType
        for et in [
            EventType::Conversation,
            EventType::ToolCall,
            EventType::Observation,
            EventType::Experiment,
            EventType::Error,
            EventType::Decision,
        ] {
            let bytes = bincode::serialize(&et).unwrap();
            let back: EventType = bincode::deserialize(&bytes).unwrap();
            assert_eq!(et, back);
        }
        // KnowledgeType
        for kt in [
            KnowledgeType::Propositional,
            KnowledgeType::Prescriptive,
            KnowledgeType::Taxonomic,
            KnowledgeType::Inferred,
            KnowledgeType::Community,
            KnowledgeType::RaptorSummary,
        ] {
            let bytes = bincode::serialize(&kt).unwrap();
            let back: KnowledgeType = bincode::deserialize(&bytes).unwrap();
            assert_eq!(kt, back);
        }
        // Origin
        for o in [
            Origin::DirectObservation,
            Origin::LlmExtraction,
            Origin::Consolidation,
            Origin::UserProvided,
            Origin::CrossAgent,
            Origin::DreamReplay,
        ] {
            let bytes = bincode::serialize(&o).unwrap();
            let back: Origin = bincode::deserialize(&bytes).unwrap();
            assert_eq!(o, back);
        }
        // MutationTrigger
        for mt in [
            MutationTrigger::Reconsolidation,
            MutationTrigger::Consolidation,
            MutationTrigger::Manual,
            MutationTrigger::Decay,
        ] {
            let bytes = bincode::serialize(&mt).unwrap();
            let back: MutationTrigger = bincode::deserialize(&bytes).unwrap();
            assert_eq!(mt, back);
        }
        // EdgeRelation
        for er in [
            EdgeRelation::RelatedTo,
            EdgeRelation::Causes,
            EdgeRelation::CausedBy,
            EdgeRelation::DerivedFrom,
            EdgeRelation::Contradicts,
            EdgeRelation::Supports,
            EdgeRelation::TemporalNext,
            EdgeRelation::PartOf,
            EdgeRelation::InstanceOf,
            EdgeRelation::SimilarTo,
            EdgeRelation::Inhibits,
            EdgeRelation::ParticipatesIn,
        ] {
            let bytes = bincode::serialize(&er).unwrap();
            let back: EdgeRelation = bincode::deserialize(&bytes).unwrap();
            assert_eq!(er, back);
        }
        // Priority
        for p in [Priority::Normal, Priority::High, Priority::Critical] {
            let bytes = bincode::serialize(&p).unwrap();
            let back: Priority = bincode::deserialize(&bytes).unwrap();
            assert_eq!(p, back);
        }
        // NamespaceKind
        for nk in [
            NamespaceKind::Private,
            NamespaceKind::Shared,
            NamespaceKind::Team,
            NamespaceKind::Default,
        ] {
            let bytes = bincode::serialize(&nk).unwrap();
            let back: NamespaceKind = bincode::deserialize(&bytes).unwrap();
            assert_eq!(nk, back);
        }
    }

    #[test]
    fn edge_relation_bidirectional() {
        assert!(EdgeRelation::RelatedTo.is_bidirectional());
        assert!(EdgeRelation::Contradicts.is_bidirectional());
        assert!(EdgeRelation::SimilarTo.is_bidirectional());
        assert!(!EdgeRelation::Causes.is_bidirectional());
        assert!(!EdgeRelation::DerivedFrom.is_bidirectional());
    }

    #[test]
    fn namespace_bincode_round_trip() {
        let ns = Namespace::new("test_ns").unwrap();
        let bytes = bincode::serialize(&ns).unwrap();
        eprintln!("Namespace bytes: {bytes:?} (len {})", bytes.len());
        let back: Namespace = bincode::deserialize(&bytes).unwrap();
        assert_eq!(ns, back);
    }

    #[test]
    fn agent_id_bincode_round_trip() {
        let aid = AgentId::new("test_agent").unwrap();
        let bytes = bincode::serialize(&aid).unwrap();
        eprintln!("AgentId bytes: {bytes:?} (len {})", bytes.len());
        let back: AgentId = bincode::deserialize(&bytes).unwrap();
        assert_eq!(aid, back);
    }

    #[test]
    fn interned_namespace_is_copy_and_4_bytes() {
        assert_eq!(std::mem::size_of::<Namespace>(), 4);
        let ns = Namespace::new("copy_test").unwrap();
        let copy = ns; // Copy
        assert_eq!(ns, copy);
    }

    #[test]
    fn interned_agent_id_is_copy_and_4_bytes() {
        assert_eq!(std::mem::size_of::<AgentId>(), 4);
        let aid = AgentId::new("copy_test").unwrap();
        let copy = aid; // Copy
        assert_eq!(aid, copy);
    }

    #[test]
    fn namespace_json_serializes_as_string() {
        let ns = Namespace::new("my_namespace").unwrap();
        let json = serde_json::to_string(&ns).unwrap();
        assert_eq!(
            json, "\"my_namespace\"",
            "should serialize as string, not integer"
        );
        let back: Namespace = serde_json::from_str(&json).unwrap();
        assert_eq!(ns, back, "JSON round-trip should preserve identity");
    }

    #[test]
    fn agent_id_json_serializes_as_string() {
        let aid = AgentId::new("agent_007").unwrap();
        let json = serde_json::to_string(&aid).unwrap();
        assert_eq!(
            json, "\"agent_007\"",
            "should serialize as string, not integer"
        );
        let back: AgentId = serde_json::from_str(&json).unwrap();
        assert_eq!(aid, back, "JSON round-trip should preserve identity");
    }
}
