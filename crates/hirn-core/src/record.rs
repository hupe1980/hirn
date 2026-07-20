use serde::{Deserialize, Serialize};

use crate::episodic::EpisodicRecord;
use crate::procedural::ProceduralRecord;
use crate::semantic::SemanticRecord;
use crate::working::WorkingMemoryEntry;

/// A memory record from any layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryRecord {
    Working(WorkingMemoryEntry),
    Episodic(EpisodicRecord),
    Semantic(SemanticRecord),
    Procedural(ProceduralRecord),
}

impl MemoryRecord {
    /// Get the layer this record belongs to.
    #[must_use]
    pub const fn layer(&self) -> crate::types::Layer {
        match self {
            Self::Working(_) => crate::types::Layer::Working,
            Self::Episodic(_) => crate::types::Layer::Episodic,
            Self::Semantic(_) => crate::types::Layer::Semantic,
            Self::Procedural(_) => crate::types::Layer::Procedural,
        }
    }

    /// Get the memory ID of this record.
    #[must_use]
    pub const fn id(&self) -> crate::id::MemoryId {
        match self {
            Self::Working(w) => w.id,
            Self::Episodic(e) => e.id,
            Self::Semantic(s) => s.id,
            Self::Procedural(p) => p.id,
        }
    }

    /// Get the namespace this record belongs to.
    /// Working memory entries do not have namespaces and return `None`.
    #[must_use]
    pub const fn namespace(&self) -> Option<&crate::types::Namespace> {
        match self {
            Self::Working(_) => None,
            Self::Episodic(e) => Some(&e.namespace),
            Self::Semantic(s) => Some(&s.namespace),
            Self::Procedural(p) => Some(&p.namespace),
        }
    }

    /// Get the primary human-readable text of this record.
    ///
    /// Returns the layer's main text field: episodic/working `content`, or the
    /// semantic/procedural `description`. Convenient for displaying recall
    /// results without matching on the variant.
    #[must_use]
    pub fn content(&self) -> &str {
        match self {
            Self::Working(w) => &w.content,
            Self::Episodic(e) => &e.content,
            Self::Semantic(s) => &s.description,
            Self::Procedural(p) => &p.description,
        }
    }

    /// Get the effective namespace used for access control and visibility.
    /// Working memory derives a private namespace from the owning agent.
    #[must_use]
    pub fn effective_namespace(&self) -> crate::types::Namespace {
        match self {
            Self::Working(w) => crate::types::Namespace::private_for(&w.agent_id),
            Self::Episodic(e) => e.namespace,
            Self::Semantic(s) => s.namespace,
            Self::Procedural(p) => p.namespace,
        }
    }

    /// Strip raw text and payload-bearing content from this record, for privacy
    /// and Cedar `recall_raw_text` enforcement.
    ///
    /// This must clear **every** field that can carry user-authored text or raw
    /// bytes — not just the primary `content`/`description`. In particular it
    /// clears multi-modal payloads (raw image/audio/document bytes and their
    /// transcripts), extracted entities, and free-form metadata, which earlier
    /// versions leaked. Structural identity fields (ids, timestamps, revision
    /// chain, namespace, importance/confidence scores) are preserved so a
    /// redacted record is still routable and rankable.
    ///
    /// Invariant: after this call the record must contain no user payload bytes.
    /// The `redacted_record_carries_no_payload` test enforces it per layer.
    pub fn strip_raw_text(&mut self) {
        match self {
            Self::Working(w) => {
                w.content = String::new();
                w.multi_content = None;
                w.revision_reason = None;
                w.thread_id = None;
            }
            Self::Episodic(e) => {
                e.content = String::new();
                e.summary = String::new();
                e.entities.clear();
                e.metadata.clear();
                e.multi_content = None;
                e.revision_reason = None;
                // `episode_id` is a free-form segment label and can echo content.
                e.episode_id = None;
            }
            Self::Semantic(s) => {
                // `concept` is a short derived label used as the record's key and
                // is preserved; the free-form body and any contradiction context
                // are cleared.
                s.description = String::new();
                s.revision_reason = None;
            }
            Self::Procedural(p) => {
                p.description = String::new();
                // `steps` carry tool parameters (potentially secrets/PII) and
                // `preconditions` are free text — both must go.
                p.steps.clear();
                p.preconditions.clear();
                p.metadata.clear();
                p.revision_reason = None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{AgentId, Layer, Namespace};

    use super::*;

    fn agent() -> AgentId {
        AgentId::new("test").unwrap()
    }

    #[test]
    fn working_record_layer() {
        let entry = WorkingMemoryEntry::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        let record = MemoryRecord::Working(entry);
        assert_eq!(record.layer(), Layer::Working);
    }

    #[test]
    fn working_record_effective_namespace_is_private_for_owner() {
        let owner = agent();
        let entry = WorkingMemoryEntry::builder()
            .content("test")
            .agent_id(owner.clone())
            .build()
            .unwrap();
        let record = MemoryRecord::Working(entry);
        assert_eq!(record.effective_namespace(), Namespace::private_for(&owner));
    }

    #[test]
    fn episodic_record_layer() {
        let entry = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        let record = MemoryRecord::Episodic(entry);
        assert_eq!(record.layer(), Layer::Episodic);
    }

    #[test]
    fn semantic_record_layer() {
        let entry = SemanticRecord::builder()
            .concept("test")
            .description("desc")
            .agent_id(agent())
            .build()
            .unwrap();
        let record = MemoryRecord::Semantic(entry);
        assert_eq!(record.layer(), Layer::Semantic);
    }

    #[test]
    fn redacted_episodic_record_carries_no_payload() {
        use crate::content::MemoryContent;
        let mut entry = EpisodicRecord::builder()
            .content("secret raw content")
            .summary("secret summary")
            .agent_id(agent())
            .entity("alice", "actor")
            .metadata_entry("pii", "ssn-123-45-6789")
            .build()
            .unwrap();
        entry.multi_content = Some(MemoryContent::Text("raw multimodal payload".into()));
        entry.revision_reason = Some("superseded because the address changed".into());
        entry.episode_id = Some("trip to the secret facility".into());
        let mut record = MemoryRecord::Episodic(entry);
        record.strip_raw_text();
        let MemoryRecord::Episodic(e) = &record else {
            unreachable!()
        };
        assert!(e.content.is_empty());
        assert!(e.summary.is_empty());
        assert!(e.entities.is_empty(), "entities must be stripped");
        assert!(e.metadata.is_empty(), "metadata must be stripped");
        assert!(e.multi_content.is_none(), "multi_content must be stripped");
        assert!(
            e.revision_reason.is_none(),
            "revision_reason is free text and must be stripped"
        );
        assert!(
            e.episode_id.is_none(),
            "episode_id is a free-form label and must be stripped"
        );
    }

    #[test]
    fn redacted_working_record_carries_no_payload() {
        use crate::content::MemoryContent;
        let mut entry = WorkingMemoryEntry::builder()
            .content("secret working thought")
            .agent_id(agent())
            .build()
            .unwrap();
        entry.multi_content = Some(MemoryContent::Text("raw payload".into()));
        entry.revision_reason = Some("changed my mind about the plan".into());
        entry.thread_id = Some("conversation-with-secrets".into());
        let mut record = MemoryRecord::Working(entry);
        record.strip_raw_text();
        let MemoryRecord::Working(w) = &record else {
            unreachable!()
        };
        assert!(w.content.is_empty());
        assert!(w.multi_content.is_none(), "multi_content must be stripped");
        assert!(
            w.revision_reason.is_none(),
            "revision_reason is free text and must be stripped"
        );
        assert!(
            w.thread_id.is_none(),
            "thread_id is a free-form label and must be stripped"
        );
    }

    #[test]
    fn redacted_semantic_record_clears_revision_reason() {
        let mut rec = SemanticRecord::builder()
            .concept("keep_this_concept")
            .description("secret description")
            .agent_id(agent())
            .build()
            .unwrap();
        rec.revision_reason = Some("contradicted by a newer observation".into());
        let mut record = MemoryRecord::Semantic(rec);
        record.strip_raw_text();
        let MemoryRecord::Semantic(s) = &record else {
            unreachable!()
        };
        assert!(s.description.is_empty());
        assert_eq!(s.concept, "keep_this_concept", "concept key is preserved");
        assert!(
            s.revision_reason.is_none(),
            "revision_reason is free text and must be stripped"
        );
    }

    #[test]
    fn redacted_procedural_record_carries_no_steps() {
        let mut rec = ProceduralRecord::builder()
            .name("deploy")
            .description("run the deploy")
            .preconditions(vec!["has creds".to_string()])
            .agent_id(agent())
            .build()
            .unwrap();
        rec.metadata.insert(
            "token".into(),
            crate::metadata::MetadataValue::String("secret".into()),
        );
        let mut record = MemoryRecord::Procedural(rec);
        record.strip_raw_text();
        let MemoryRecord::Procedural(p) = &record else {
            unreachable!()
        };
        assert!(p.description.is_empty());
        assert!(p.steps.is_empty());
        assert!(p.preconditions.is_empty());
        assert!(p.metadata.is_empty());
    }

    #[test]
    fn serde_round_trip() {
        let entry = EpisodicRecord::builder()
            .content("test")
            .agent_id(agent())
            .build()
            .unwrap();
        let record = MemoryRecord::Episodic(entry);
        let bytes = bincode::serialize(&record).unwrap();
        let back: MemoryRecord = bincode::deserialize(&bytes).unwrap();
        assert_eq!(record, back);
    }

    #[test]
    fn bincode_size_reasonable() {
        // Working memory entry with minimal content.
        let wm = WorkingMemoryEntry::builder()
            .content("short")
            .agent_id(agent())
            .build()
            .unwrap();
        let wm_bytes = bincode::serialize(&wm).unwrap();
        assert!(
            wm_bytes.len() < 512,
            "WorkingMemoryEntry serialized to {} bytes, expected < 512",
            wm_bytes.len()
        );

        // Episodic record with some content.
        let ep = EpisodicRecord::builder()
            .content("a typical episodic event description")
            .agent_id(agent())
            .entity("user", "actor")
            .metadata_entry("key", "value")
            .build()
            .unwrap();
        let ep_bytes = bincode::serialize(&ep).unwrap();
        assert!(
            ep_bytes.len() < 1024,
            "EpisodicRecord serialized to {} bytes, expected < 1024",
            ep_bytes.len()
        );

        // Semantic record.
        let sem = SemanticRecord::builder()
            .concept("database_indexing")
            .description("B-tree indexes accelerate range queries on sorted data")
            .agent_id(agent())
            .build()
            .unwrap();
        let sem_bytes = bincode::serialize(&sem).unwrap();
        assert!(
            sem_bytes.len() < 1024,
            "SemanticRecord serialized to {} bytes, expected < 1024",
            sem_bytes.len()
        );
    }
}

#[cfg(test)]
mod proptest_tests {
    use proptest::prelude::*;

    use crate::episodic::EpisodicRecord;
    use crate::semantic::SemanticRecord;
    use crate::types::AgentId;
    use crate::working::WorkingMemoryEntry;

    use super::MemoryRecord;

    fn agent() -> AgentId {
        AgentId::new("prop_agent").unwrap()
    }

    prop_compose! {
        fn arb_working_entry()(
            content in "[a-zA-Z0-9 ]{1,100}",
            token_count in 0u32..10000,
            relevance in 0.0f32..=1.0,
        ) -> WorkingMemoryEntry {
            WorkingMemoryEntry::builder()
                .content(content)
                .agent_id(agent())
                .token_count(token_count)
                .relevance_score(relevance)
                .build()
                .unwrap()
        }
    }

    prop_compose! {
        fn arb_episodic_record()(
            content in "[a-zA-Z0-9 ]{1,200}",
            importance in 0.0f32..=1.0,
            surprise in 0.0f32..=1.0,
        ) -> EpisodicRecord {
            EpisodicRecord::builder()
                .content(content)
                .importance(importance)
                .surprise(surprise)
                .agent_id(agent())
                .build()
                .unwrap()
        }
    }

    prop_compose! {
        fn arb_semantic_record()(
            concept in "[a-zA-Z_]{1,50}",
            description in "[a-zA-Z0-9 ]{1,200}",
            confidence in 0.0f32..=1.0,
        ) -> SemanticRecord {
            SemanticRecord::builder()
                .concept(concept)
                .description(description)
                .confidence(confidence)
                .agent_id(agent())
                .build()
                .unwrap()
        }
    }

    fn arb_memory_record() -> impl Strategy<Value = MemoryRecord> {
        prop_oneof![
            arb_working_entry().prop_map(MemoryRecord::Working),
            arb_episodic_record().prop_map(MemoryRecord::Episodic),
            arb_semantic_record().prop_map(MemoryRecord::Semantic),
        ]
    }

    proptest! {
        #[test]
        fn memory_record_bincode_round_trip(record in arb_memory_record()) {
            let bytes = bincode::serialize(&record).unwrap();
            let back: MemoryRecord = bincode::deserialize(&bytes).unwrap();
            prop_assert_eq!(record, back);
        }
    }
}
