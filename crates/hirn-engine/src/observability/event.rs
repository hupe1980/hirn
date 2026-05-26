//! Event system for memory operations.
//!
//! Two layers:
//! - [`MemoryEvent`] — the structured event enum covering all mutation types.
//! - [`EventEnvelope`] — wraps a `MemoryEvent` with monotonic seq, wall-clock
//!   timestamp, realm/namespace/agent metadata. Serializable via bincode + JSON.
//!
//! Subscribers receive [`MemoryEvent`] values through `mpsc` channels for
//! real-time in-process push. The [`EventLog`](super::event_log::EventLog)
//! persists [`EventEnvelope`] to the `events` LanceDB dataset for durable
//! history, replay, and audit.

use hirn_core::id::MemoryId;
use hirn_core::revision::{LogicalMemoryId, RevisionId};
use hirn_core::types::{EdgeRelation, Layer};

/// An event emitted when the database state changes.
///
/// Covers all mutation types for event sourcing.
/// New variants can be added without breaking old readers thanks to
/// `#[serde(other)]` on the `Unknown` fallback.
///
/// Externally tagged (default) for bincode compatibility. JSON uses
/// `{"EpisodeCreated": {...}}` style.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum MemoryEvent {
    /// A new episodic memory was created.
    EpisodeCreated {
        id: MemoryId,
        content_preview: String,
    },
    /// A new semantic record was created.
    SemanticCreated { id: MemoryId, concept_name: String },
    /// A new procedural record was created.
    ProceduralCreated {
        id: MemoryId,
        procedure_name: String,
    },
    /// A semantic memory was corrected with a new head revision.
    MemoryCorrected {
        logical_memory_id: LogicalMemoryId,
        old_revision_id: RevisionId,
        new_revision_id: RevisionId,
        #[serde(default)]
        reason: Option<String>,
    },
    /// A semantic memory was explicitly superseded by a new head revision.
    MemorySuperseded {
        logical_memory_id: LogicalMemoryId,
        prior_revision_id: RevisionId,
        new_revision_id: RevisionId,
        #[serde(default)]
        reason: Option<String>,
    },
    /// A semantic memory head was explicitly overridden by a human/admin revision.
    MemoryOverridden {
        logical_memory_id: LogicalMemoryId,
        prior_revision_id: RevisionId,
        override_revision_id: RevisionId,
        #[serde(default)]
        reason: Option<String>,
    },
    /// One or more semantic logical memories were merged into an active target chain.
    MemoryMerged {
        target_logical_memory_id: LogicalMemoryId,
        prior_target_revision_id: RevisionId,
        new_target_revision_id: RevisionId,
        source_logical_memory_ids: Vec<LogicalMemoryId>,
        source_revision_ids: Vec<RevisionId>,
        #[serde(default)]
        reason: Option<String>,
    },
    /// A semantic memory was retracted via a tombstone revision.
    MemoryRetracted {
        logical_memory_id: LogicalMemoryId,
        prior_revision_id: RevisionId,
        tombstone_revision_id: RevisionId,
        #[serde(default)]
        reason: Option<String>,
    },
    /// A working memory entry was pushed.
    WorkingPushed { id: MemoryId },
    /// Importance score was updated.
    ImportanceUpdated {
        id: MemoryId,
        old_value: f32,
        new_value: f32,
    },
    /// A memory was reconsolidated (modified during labile window).
    Reconsolidated { id: MemoryId, reason: String },
    /// A graph edge was created.
    EdgeCreated {
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        weight: f32,
    },
    /// A graph edge weight was updated.
    EdgeWeightUpdated {
        source: MemoryId,
        target: MemoryId,
        relation: EdgeRelation,
        old_weight: f32,
        new_weight: f32,
    },
    /// A memory was archived (soft-deleted).
    Archived { id: MemoryId },
    /// A memory was permanently forgotten (hard-deleted).
    Forgotten { id: MemoryId },
    /// Consolidation completed.
    Consolidated { records_processed: usize },
    /// A snapshot was taken.
    SnapshotTaken { seq: u64, tag: String },
    /// Compaction completed.
    CompactionCompleted {
        before_seq: u64,
        events_removed: u64,
    },
    /// An admission control decision was made.
    AdmissionEvaluated {
        candidate_id: MemoryId,
        decision: String,
        controllers_consulted: Vec<String>,
    },
    /// A dream hypothesis was generated.
    HypothesisGenerated {
        id: MemoryId,
        source_a: MemoryId,
        source_b: MemoryId,
        batch_id: String,
    },
    /// A dream hypothesis was validated and promoted.
    HypothesisValidated {
        id: MemoryId,
        new_confidence: f32,
        evidence_count: u32,
        batch_id: String,
    },
    /// A dream hypothesis was discarded after validation.
    HypothesisDiscarded {
        id: MemoryId,
        reason: String,
        batch_id: String,
    },
    /// An authorization request was granted.
    AccessGranted {
        action: String,
        realm: String,
        namespace: String,
        policy_ids: Vec<String>,
    },
    /// An authorization request was denied.
    AccessDenied {
        action: String,
        realm: String,
        namespace: String,
        reasons: Vec<String>,
        policy_ids: Vec<String>,
    },
    /// A Cedar policy was added, removed, or modified.
    PolicyChanged {
        policy_name: String,
        change_type: String,
        #[serde(default)]
        policy_content: String,
    },
    /// A memory was recalled (query executed).
    MemoryRecalled {
        query_preview: String,
        results_count: usize,
    },
    /// A contradiction was detected between two memories.
    ContradictionDetected {
        memory_a: MemoryId,
        memory_b: MemoryId,
        confidence: f32,
    },
    /// A causal edge was discovered during consolidation.
    CausalEdgeDiscovered {
        cause: MemoryId,
        effect: MemoryId,
        strength: f32,
    },
    /// An error occurred during a database operation.
    Error { operation: String, message: String },
    /// Unknown event variant for forward compatibility.
    #[serde(other)]
    Unknown,
}

impl MemoryEvent {
    /// Event type as a short string.
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::EpisodeCreated { .. } => "episode_created",
            Self::SemanticCreated { .. } => "semantic_created",
            Self::ProceduralCreated { .. } => "procedural_created",
            Self::MemoryCorrected { .. } => "memory_corrected",
            Self::MemorySuperseded { .. } => "memory_superseded",
            Self::MemoryOverridden { .. } => "memory_overridden",
            Self::MemoryMerged { .. } => "memory_merged",
            Self::MemoryRetracted { .. } => "memory_retracted",
            Self::WorkingPushed { .. } => "working_pushed",
            Self::ImportanceUpdated { .. } => "importance_updated",
            Self::Reconsolidated { .. } => "reconsolidated",
            Self::EdgeCreated { .. } => "edge_created",
            Self::EdgeWeightUpdated { .. } => "edge_weight_updated",
            Self::Archived { .. } => "archived",
            Self::Forgotten { .. } => "forgotten",
            Self::Consolidated { .. } => "consolidated",
            Self::SnapshotTaken { .. } => "snapshot_taken",
            Self::CompactionCompleted { .. } => "compaction_completed",
            Self::AdmissionEvaluated { .. } => "admission_evaluated",
            Self::HypothesisGenerated { .. } => "hypothesis_generated",
            Self::HypothesisValidated { .. } => "hypothesis_validated",
            Self::HypothesisDiscarded { .. } => "hypothesis_discarded",
            Self::AccessGranted { .. } => "access_granted",
            Self::AccessDenied { .. } => "access_denied",
            Self::PolicyChanged { .. } => "policy_changed",
            Self::MemoryRecalled { .. } => "memory_recalled",
            Self::ContradictionDetected { .. } => "contradiction_detected",
            Self::CausalEdgeDiscovered { .. } => "causal_edge_discovered",
            Self::Error { .. } => "error",
            Self::Unknown => "unknown",
        }
    }

    /// Whether this event should be appended to the durable event log.
    pub fn should_persist(&self) -> bool {
        !matches!(self, Self::MemoryRecalled { .. })
    }

    /// The layer this event affects, if applicable.
    pub fn layer(&self) -> Option<Layer> {
        match self {
            Self::EpisodeCreated { .. } => Some(Layer::Episodic),
            Self::SemanticCreated { .. } => Some(Layer::Semantic),
            Self::ProceduralCreated { .. } => Some(Layer::Procedural),
            Self::MemoryCorrected { .. } => Some(Layer::Semantic),
            Self::MemorySuperseded { .. } => Some(Layer::Semantic),
            Self::MemoryOverridden { .. } => Some(Layer::Semantic),
            Self::MemoryMerged { .. } => Some(Layer::Semantic),
            Self::MemoryRetracted { .. } => Some(Layer::Semantic),
            Self::WorkingPushed { .. } => Some(Layer::Working),
            _ => None,
        }
    }
}

// ── Event Envelope ───────────────────────────────────────────────────────

/// A durable event wrapper with monotonic sequence number and metadata.
///
/// Envelopes are stored in the `events` LanceDB dataset for replay, audit,
/// and streaming.
///
/// When a realm secret is configured, each envelope carries an HMAC-SHA256
/// tag computed over the bincode-serialized payload. This enables tamper
/// detection on audit events.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EventEnvelope {
    /// Monotonically increasing sequence number (gap-free per writer).
    pub seq: u64,
    /// Wall-clock microsecond timestamp.
    pub timestamp_us: i64,
    /// Realm (tenant) isolation.
    pub realm: String,
    /// Namespace within the realm.
    pub namespace: String,
    /// Agent that triggered the mutation.
    pub agent_id: String,
    /// The actual event payload.
    pub event: MemoryEvent,
    /// HMAC-SHA256 tag for tamper detection (hex-encoded).
    /// `None` when no realm secret is configured.
    #[serde(default)]
    pub hmac: Option<String>,
}

impl EventEnvelope {
    /// Create a new envelope with the given metadata.
    pub fn new(
        seq: u64,
        realm: impl Into<String>,
        namespace: impl Into<String>,
        agent_id: impl Into<String>,
        event: MemoryEvent,
    ) -> Self {
        let now = chrono::Utc::now();
        Self {
            seq,
            timestamp_us: now.timestamp_micros(),
            realm: realm.into(),
            namespace: namespace.into(),
            agent_id: agent_id.into(),
            event,
            hmac: None,
        }
    }

    /// Compute and attach an HMAC-SHA256 tag using the given secret.
    ///
    /// The HMAC is computed over the bincode-serialized event payload
    /// concatenated with the envelope metadata (seq, timestamp, realm,
    /// namespace, agent_id). This ensures any tampering with the envelope
    /// is detectable.
    pub fn sign(&mut self, secret: &[u8]) {
        let bytes = self.signable_bytes();
        let tag = Self::compute_hmac(secret, &bytes);
        self.hmac = Some(tag);
    }

    /// Verify the HMAC tag against the given secret.
    ///
    /// Returns `true` if the HMAC matches, `false` if tampered or missing.
    pub fn verify_hmac(&self, secret: &[u8]) -> bool {
        let Some(ref stored_hmac) = self.hmac else {
            return false;
        };
        let bytes = self.signable_bytes();
        let expected = Self::compute_hmac(secret, &bytes);
        // Constant-time comparison to prevent timing attacks.
        constant_time_eq(stored_hmac.as_bytes(), expected.as_bytes())
    }

    /// The bytes that are signed/verified by HMAC.
    fn signable_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(&self.seq.to_le_bytes());
        buf.extend_from_slice(&self.timestamp_us.to_le_bytes());
        buf.extend_from_slice(self.realm.as_bytes());
        buf.push(0); // separator
        buf.extend_from_slice(self.namespace.as_bytes());
        buf.push(0);
        buf.extend_from_slice(self.agent_id.as_bytes());
        buf.push(0);
        if let Ok(payload) = bincode::serialize(&self.event) {
            buf.extend_from_slice(&payload);
        }
        buf
    }

    /// Compute HMAC-SHA256 using blake3 keyed hash (256-bit, faster than
    /// HMAC-SHA256, same security guarantees) and return as hex string.
    fn compute_hmac(secret: &[u8], data: &[u8]) -> String {
        // Derive a 32-byte key from the secret using blake3.
        let key = blake3::derive_key("hirn event hmac v1", secret);
        let hash = blake3::keyed_hash(&key, data);
        hash.to_hex().to_string()
    }

    /// Event type string (delegates to inner event).
    pub fn event_type(&self) -> &'static str {
        self.event.event_type()
    }

    /// Byte size of the bincode-serialized envelope (for size budgeting).
    pub fn bincode_size(&self) -> usize {
        bincode::serialized_size(self).unwrap_or(0) as usize
    }
}

/// Constant-time byte comparison to prevent timing side-channels on HMAC verification.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_id() -> MemoryId {
        MemoryId::new()
    }

    // ── Serialization round-trips ──

    #[test]
    fn bincode_round_trip_all_variants() {
        let variants = vec![
            MemoryEvent::EpisodeCreated {
                id: sample_id(),
                content_preview: "hello world".into(),
            },
            MemoryEvent::SemanticCreated {
                id: sample_id(),
                concept_name: "Rust".into(),
            },
            MemoryEvent::ProceduralCreated {
                id: sample_id(),
                procedure_name: "deploy-to-staging".into(),
            },
            MemoryEvent::WorkingPushed { id: sample_id() },
            MemoryEvent::ImportanceUpdated {
                id: sample_id(),
                old_value: 0.3,
                new_value: 0.7,
            },
            MemoryEvent::Reconsolidated {
                id: sample_id(),
                reason: "new evidence".into(),
            },
            MemoryEvent::EdgeCreated {
                source: sample_id(),
                target: sample_id(),
                relation: EdgeRelation::Causes,
                weight: 0.8,
            },
            MemoryEvent::EdgeWeightUpdated {
                source: sample_id(),
                target: sample_id(),
                relation: EdgeRelation::SimilarTo,
                old_weight: 0.5,
                new_weight: 0.9,
            },
            MemoryEvent::Archived { id: sample_id() },
            MemoryEvent::Forgotten { id: sample_id() },
            MemoryEvent::Consolidated {
                records_processed: 42,
            },
            MemoryEvent::SnapshotTaken {
                seq: 100,
                tag: "snapshot-100".into(),
            },
            MemoryEvent::CompactionCompleted {
                before_seq: 50,
                events_removed: 50,
            },
            MemoryEvent::MemoryRecalled {
                query_preview: "test query".into(),
                results_count: 5,
            },
            MemoryEvent::ContradictionDetected {
                memory_a: sample_id(),
                memory_b: sample_id(),
                confidence: 0.92,
            },
            MemoryEvent::CausalEdgeDiscovered {
                cause: sample_id(),
                effect: sample_id(),
                strength: 0.75,
            },
            MemoryEvent::Error {
                operation: "remember".into(),
                message: "embedding failed".into(),
            },
        ];

        for event in &variants {
            let bytes = bincode::serialize(event).expect("serialize");
            let decoded: MemoryEvent = bincode::deserialize(&bytes).expect("deserialize");
            assert_eq!(event.event_type(), decoded.event_type());
        }
    }

    #[test]
    fn json_round_trip_all_variants() {
        let variants = vec![
            MemoryEvent::EpisodeCreated {
                id: sample_id(),
                content_preview: "test".into(),
            },
            MemoryEvent::SemanticCreated {
                id: sample_id(),
                concept_name: "concept".into(),
            },
            MemoryEvent::ProceduralCreated {
                id: sample_id(),
                procedure_name: "deploy-to-staging".into(),
            },
            MemoryEvent::WorkingPushed { id: sample_id() },
            MemoryEvent::ImportanceUpdated {
                id: sample_id(),
                old_value: 0.1,
                new_value: 0.9,
            },
            MemoryEvent::Reconsolidated {
                id: sample_id(),
                reason: "updated".into(),
            },
            MemoryEvent::EdgeCreated {
                source: sample_id(),
                target: sample_id(),
                relation: EdgeRelation::DerivedFrom,
                weight: 0.5,
            },
            MemoryEvent::EdgeWeightUpdated {
                source: sample_id(),
                target: sample_id(),
                relation: EdgeRelation::Contradicts,
                old_weight: 0.2,
                new_weight: 0.8,
            },
            MemoryEvent::Archived { id: sample_id() },
            MemoryEvent::Forgotten { id: sample_id() },
            MemoryEvent::Consolidated {
                records_processed: 10,
            },
            MemoryEvent::SnapshotTaken {
                seq: 200,
                tag: "snap-200".into(),
            },
            MemoryEvent::CompactionCompleted {
                before_seq: 100,
                events_removed: 100,
            },
            MemoryEvent::MemoryRecalled {
                query_preview: "recall test".into(),
                results_count: 3,
            },
            MemoryEvent::ContradictionDetected {
                memory_a: sample_id(),
                memory_b: sample_id(),
                confidence: 0.85,
            },
            MemoryEvent::CausalEdgeDiscovered {
                cause: sample_id(),
                effect: sample_id(),
                strength: 0.6,
            },
            MemoryEvent::Error {
                operation: "consolidation".into(),
                message: "timeout".into(),
            },
        ];

        for event in &variants {
            let json = serde_json::to_string(event).expect("to json");
            let decoded: MemoryEvent = serde_json::from_str(&json).expect("from json");
            assert_eq!(event.event_type(), decoded.event_type());
        }
    }

    #[test]
    fn envelope_seq_monotonic() {
        let envelopes: Vec<EventEnvelope> = (0..100)
            .map(|seq| {
                EventEnvelope::new(
                    seq,
                    "default",
                    "shared",
                    "agent-1",
                    MemoryEvent::WorkingPushed { id: sample_id() },
                )
            })
            .collect();

        for pair in envelopes.windows(2) {
            assert!(
                pair[1].seq > pair[0].seq,
                "seq must be monotonically increasing"
            );
        }
    }

    #[test]
    fn unknown_variant_forward_compatibility() {
        // Externally-tagged enums with #[serde(other)] capture unknown
        // unit variants. For JSON, we test via bincode with an Unknown value.
        let event = MemoryEvent::Unknown;
        let bytes = bincode::serialize(&event).expect("serialize");
        let decoded: MemoryEvent = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(decoded.event_type(), "unknown");

        // Also verify JSON round-trip of the Unknown variant itself works.
        let json = serde_json::to_string(&event).expect("to json");
        let decoded: MemoryEvent = serde_json::from_str(&json).expect("from json");
        assert_eq!(decoded.event_type(), "unknown");
    }

    #[test]
    fn envelope_bincode_round_trip() {
        let envelope = EventEnvelope::new(
            42,
            "prod",
            "default",
            "agent-x",
            MemoryEvent::EpisodeCreated {
                id: sample_id(),
                content_preview: "test episode".into(),
            },
        );

        let bytes = bincode::serialize(&envelope).expect("serialize");
        let decoded: EventEnvelope = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(decoded.seq, 42);
        assert_eq!(decoded.realm, "prod");
        assert_eq!(decoded.namespace, "default");
        assert_eq!(decoded.agent_id, "agent-x");
        assert_eq!(decoded.event.event_type(), "episode_created");
    }

    #[test]
    fn envelope_json_round_trip() {
        let envelope = EventEnvelope::new(
            7,
            "staging",
            "team-a",
            "agent-y",
            MemoryEvent::Consolidated {
                records_processed: 99,
            },
        );

        let json = serde_json::to_string(&envelope).expect("to json");
        let decoded: EventEnvelope = serde_json::from_str(&json).expect("from json");
        assert_eq!(decoded.seq, 7);
        assert_eq!(decoded.realm, "staging");
    }

    #[test]
    fn typical_episode_created_envelope_under_2kb() {
        let envelope = EventEnvelope::new(
            1,
            "default",
            "shared",
            "test-agent",
            MemoryEvent::EpisodeCreated {
                id: sample_id(),
                content_preview: "A moderately long preview of an episodic memory entry that contains enough text to be representative of real-world usage".into(),
            },
        );

        let size = envelope.bincode_size();
        assert!(
            size < 2048,
            "EpisodeCreated envelope should be < 2KB, got {size}"
        );
    }

    // ── Authorization event variants ──

    #[test]
    fn access_granted_event_serde() {
        let event = MemoryEvent::AccessGranted {
            action: "remember".into(),
            realm: "production".into(),
            namespace: "shared".into(),
            policy_ids: vec!["policy0".into()],
        };
        assert_eq!(event.event_type(), "access_granted");

        let bytes = bincode::serialize(&event).unwrap();
        let decoded: MemoryEvent = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.event_type(), "access_granted");

        let json = serde_json::to_string(&event).unwrap();
        let decoded: MemoryEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.event_type(), "access_granted");
    }

    #[test]
    fn access_denied_event_serde() {
        let event = MemoryEvent::AccessDenied {
            action: "consolidate".into(),
            realm: "production".into(),
            namespace: "restricted".into(),
            reasons: vec!["denied: agent cannot consolidate".into()],
            policy_ids: vec!["forbid0".into()],
        };
        assert_eq!(event.event_type(), "access_denied");

        let json = serde_json::to_string(&event).unwrap();
        let decoded: MemoryEvent = serde_json::from_str(&json).unwrap();
        if let MemoryEvent::AccessDenied { reasons, .. } = decoded {
            assert_eq!(reasons.len(), 1);
            assert!(reasons[0].contains("cannot consolidate"));
        } else {
            panic!("expected AccessDenied");
        }
    }

    #[test]
    fn policy_changed_event_serde() {
        let event = MemoryEvent::PolicyChanged {
            policy_name: "acl.cedar".into(),
            change_type: "added".into(),
            policy_content: "permit(principal, action, resource);".into(),
        };
        assert_eq!(event.event_type(), "policy_changed");

        let bytes = bincode::serialize(&event).unwrap();
        let decoded: MemoryEvent = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.event_type(), "policy_changed");
    }

    // ── HMAC tamper detection ──

    #[test]
    fn hmac_sign_and_verify() {
        let secret = b"realm-secret-key-for-testing";
        let mut envelope = EventEnvelope::new(
            1,
            "production",
            "shared",
            "agent-007",
            MemoryEvent::EpisodeCreated {
                id: sample_id(),
                content_preview: "classified intel".into(),
            },
        );

        assert!(envelope.hmac.is_none());

        envelope.sign(secret);
        assert!(envelope.hmac.is_some());
        assert!(envelope.verify_hmac(secret));
    }

    #[test]
    fn hmac_detects_tampered_payload() {
        let secret = b"realm-secret";
        let mut envelope = EventEnvelope::new(
            1,
            "prod",
            "shared",
            "agent-007",
            MemoryEvent::EpisodeCreated {
                id: sample_id(),
                content_preview: "original content".into(),
            },
        );

        envelope.sign(secret);
        assert!(envelope.verify_hmac(secret));

        // Tamper with the event payload.
        envelope.event = MemoryEvent::EpisodeCreated {
            id: sample_id(),
            content_preview: "TAMPERED content".into(),
        };

        assert!(
            !envelope.verify_hmac(secret),
            "tampered payload should fail HMAC"
        );
    }

    #[test]
    fn hmac_detects_tampered_metadata() {
        let secret = b"realm-secret";
        let mut envelope = EventEnvelope::new(
            1,
            "production",
            "shared",
            "agent-007",
            MemoryEvent::Consolidated {
                records_processed: 10,
            },
        );

        envelope.sign(secret);
        assert!(envelope.verify_hmac(secret));

        // Tamper with agent_id.
        envelope.agent_id = "impersonator".into();
        assert!(
            !envelope.verify_hmac(secret),
            "tampered agent_id should fail HMAC"
        );
    }

    #[test]
    fn hmac_wrong_secret_fails() {
        let secret = b"correct-secret";
        let wrong = b"wrong-secret";
        let mut envelope = EventEnvelope::new(
            1,
            "prod",
            "shared",
            "agent",
            MemoryEvent::Forgotten { id: sample_id() },
        );

        envelope.sign(secret);
        assert!(envelope.verify_hmac(secret));
        assert!(!envelope.verify_hmac(wrong), "wrong secret should fail");
    }

    #[test]
    fn hmac_missing_returns_false() {
        let envelope = EventEnvelope::new(
            1,
            "prod",
            "shared",
            "agent",
            MemoryEvent::WorkingPushed { id: sample_id() },
        );

        assert!(
            !envelope.verify_hmac(b"any-secret"),
            "missing HMAC should return false"
        );
    }

    #[test]
    fn hmac_on_authorization_events() {
        let secret = b"audit-secret";

        let mut granted = EventEnvelope::new(
            10,
            "production",
            "shared",
            "agent-007",
            MemoryEvent::AccessGranted {
                action: "recall".into(),
                realm: "production".into(),
                namespace: "shared".into(),
                policy_ids: vec!["policy0".into()],
            },
        );
        granted.sign(secret);
        assert!(granted.verify_hmac(secret));

        let mut denied = EventEnvelope::new(
            11,
            "production",
            "restricted",
            "intern-bot",
            MemoryEvent::AccessDenied {
                action: "remember".into(),
                realm: "production".into(),
                namespace: "restricted".into(),
                reasons: vec!["denied by policy".into()],
                policy_ids: vec!["forbid0".into()],
            },
        );
        denied.sign(secret);
        assert!(denied.verify_hmac(secret));
    }

    #[test]
    fn hmac_envelope_serde_preserves_tag() {
        let secret = b"serde-test-secret";
        let mut envelope = EventEnvelope::new(
            42,
            "realm-a",
            "ns-1",
            "agent-x",
            MemoryEvent::Consolidated {
                records_processed: 5,
            },
        );
        envelope.sign(secret);

        // JSON round-trip preserves HMAC.
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: EventEnvelope = serde_json::from_str(&json).unwrap();
        assert!(decoded.verify_hmac(secret));

        // Bincode round-trip preserves HMAC.
        let bytes = bincode::serialize(&envelope).unwrap();
        let decoded: EventEnvelope = bincode::deserialize(&bytes).unwrap();
        assert!(decoded.verify_hmac(secret));
    }
}
