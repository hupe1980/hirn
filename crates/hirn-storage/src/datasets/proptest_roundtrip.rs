//! F-028 FIX: Property-based round-trip tests for Arrow batch
//! serialization/deserialization across all dataset modules.

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use proptest::prelude::*;

    use hirn_core::episodic::{EntityRef, EpisodicRecord};
    use hirn_core::id::MemoryId;
    use hirn_core::procedural::{ActionStep, ProceduralRecord};
    use hirn_core::provenance::Provenance;
    use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation};
    use hirn_core::semantic::SemanticRecord;
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::{
        AgentId, EdgeRelation, EventType, KnowledgeType, Layer, Namespace, Priority,
    };
    use hirn_core::working::WorkingMemoryEntry;
    use hirn_graph::graph::{GraphEdge, GraphNodeData};

    use crate::datasets::{episodic, graph, procedural, semantic, working};

    // ── Strategies ──────────────────────────────────────────────

    fn agent() -> AgentId {
        AgentId::well_known("proptest-agent")
    }

    fn arb_event_type() -> impl Strategy<Value = EventType> {
        prop_oneof![
            Just(EventType::Conversation),
            Just(EventType::ToolCall),
            Just(EventType::Observation),
            Just(EventType::Experiment),
            Just(EventType::Error),
            Just(EventType::Decision),
        ]
    }

    fn arb_priority() -> impl Strategy<Value = Priority> {
        prop_oneof![
            Just(Priority::Normal),
            Just(Priority::High),
            Just(Priority::Critical),
        ]
    }

    fn arb_knowledge_type() -> impl Strategy<Value = KnowledgeType> {
        prop_oneof![
            Just(KnowledgeType::Propositional),
            Just(KnowledgeType::Prescriptive),
            Just(KnowledgeType::Taxonomic),
            Just(KnowledgeType::Inferred),
            Just(KnowledgeType::Community),
            Just(KnowledgeType::RaptorSummary),
        ]
    }

    fn arb_layer() -> impl Strategy<Value = Layer> {
        prop_oneof![
            Just(Layer::Working),
            Just(Layer::Episodic),
            Just(Layer::Semantic),
            Just(Layer::Procedural),
        ]
    }

    fn arb_edge_relation() -> impl Strategy<Value = EdgeRelation> {
        prop_oneof![
            Just(EdgeRelation::RelatedTo),
            Just(EdgeRelation::Causes),
            Just(EdgeRelation::CausedBy),
            Just(EdgeRelation::DerivedFrom),
            Just(EdgeRelation::Contradicts),
            Just(EdgeRelation::Supports),
            Just(EdgeRelation::TemporalNext),
            Just(EdgeRelation::PartOf),
            Just(EdgeRelation::InstanceOf),
            Just(EdgeRelation::SimilarTo),
            Just(EdgeRelation::Inhibits),
            Just(EdgeRelation::ParticipatesIn),
        ]
    }

    const EMBED_DIMS: usize = 4;

    fn arb_embedding() -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-1.0f32..1.0, EMBED_DIMS..=EMBED_DIMS)
    }

    fn arb_entity() -> impl Strategy<Value = EntityRef> {
        ("[a-zA-Z]{1,20}", "[a-zA-Z]{1,10}").prop_map(|(name, role)| EntityRef {
            name,
            role,
            entity_id: None,
        })
    }

    // ── Episodic ────────────────────────────────────────────────

    prop_compose! {
        fn arb_episodic()(
            content in "[a-zA-Z0-9 ]{1,100}",
            summary in "[a-zA-Z0-9 ]{0,80}",
            event_type in arb_event_type(),
            importance in 0.0f32..=1.0,
            surprise in 0.0f32..=1.0,
            stability in 0.1f32..100.0,
            access_count in 0u64..1000,
            entities in prop::collection::vec(arb_entity(), 0..3),
            embedding in prop::option::of(arb_embedding()),
            archived in proptest::bool::ANY,
            valence in prop::option::of(-1.0f32..=1.0f32),
        ) -> EpisodicRecord {
            let id = MemoryId::new();
            let now = Timestamp::now();
            EpisodicRecord {
                id,
                logical_memory_id: LogicalMemoryId::from_memory_id(id),
                revision_id: RevisionId::from_memory_id(id),
                version: 1,
                revision_operation: RevisionOperation::Create,
                revision_reason: None,
                revision_causation_id: None,
                timestamp: now,
                created_at: now,
                updated_at: now,
                superseded_by: None,
                event_type,
                content,
                summary,
                entities,
                embedding,
                importance,
                surprise,
                access_count,
                last_accessed: now,
                stability,
                consolidation_ids: vec![],
                episode_id: None,
                provenance: Provenance::direct(agent()),
                metadata: BTreeMap::default(),
                namespace: Namespace::default_ns(),
                archived,
                expires_at: None,
                valid_until: None,
                multi_content: None,
                valence,
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn episodic_arrow_round_trip(record in arb_episodic()) {
            let batch = episodic::to_batch(std::slice::from_ref(&record), EMBED_DIMS).unwrap();
            let decoded = episodic::from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), 1);
            let d = &decoded[0];
            prop_assert_eq!(&d.content, &record.content);
            prop_assert_eq!(&d.summary, &record.summary);
            prop_assert_eq!(d.event_type, record.event_type);
            prop_assert!((d.importance - record.importance).abs() < f32::EPSILON);
            prop_assert!((d.surprise - record.surprise).abs() < f32::EPSILON);
            prop_assert!((d.stability - record.stability).abs() < f32::EPSILON);
            prop_assert_eq!(d.access_count, record.access_count);
            prop_assert_eq!(d.entities.len(), record.entities.len());
            prop_assert_eq!(d.archived, record.archived);
            prop_assert_eq!(&d.embedding, &record.embedding);
            prop_assert_eq!(d.valence, record.valence);
        }

        #[test]
        fn episodic_batch_size(records in prop::collection::vec(arb_episodic(), 1..8)) {
            let batch = episodic::to_batch(&records, EMBED_DIMS).unwrap();
            prop_assert_eq!(batch.num_rows(), records.len());
            let decoded = episodic::from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), records.len());
            for (orig, dec) in records.iter().zip(decoded.iter()) {
                prop_assert_eq!(&orig.content, &dec.content);
            }
        }
    }

    // ── Semantic ────────────────────────────────────────────────

    prop_compose! {
        fn arb_semantic()(
            concept in "[a-zA-Z_]{1,30}",
            description in "[a-zA-Z0-9 ]{1,100}",
            knowledge_type in arb_knowledge_type(),
            confidence in 0.0f32..=1.0,
            embedding in prop::option::of(arb_embedding()),
        ) -> SemanticRecord {
            SemanticRecord::builder()
                .concept(concept)
                .description(description)
                .knowledge_type(knowledge_type)
                .confidence(confidence)
                .agent_id(agent())
                .build()
                .unwrap()
                .tap_mut(|r| r.embedding = embedding)
        }
    }

    // Helper: SemanticRecord doesn't expose embedding via builder easily,
    // so we mutate after build.
    trait TapMut: Sized {
        fn tap_mut(mut self, f: impl FnOnce(&mut Self)) -> Self {
            f(&mut self);
            self
        }
    }
    impl<T> TapMut for T {}

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn semantic_arrow_round_trip(record in arb_semantic()) {
            let batch = semantic::to_batch(std::slice::from_ref(&record), EMBED_DIMS).unwrap();
            let decoded = semantic::from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), 1);
            let d = &decoded[0];
            prop_assert_eq!(&d.concept, &record.concept);
            prop_assert_eq!(&d.description, &record.description);
            prop_assert_eq!(d.knowledge_type, record.knowledge_type);
            prop_assert!((d.confidence - record.confidence).abs() < f32::EPSILON);
            prop_assert_eq!(&d.embedding, &record.embedding);
        }

        #[test]
        fn semantic_batch_size(records in prop::collection::vec(arb_semantic(), 1..8)) {
            let batch = semantic::to_batch(&records, EMBED_DIMS).unwrap();
            prop_assert_eq!(batch.num_rows(), records.len());
            let decoded = semantic::from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), records.len());
        }
    }

    // ── Working ─────────────────────────────────────────────────

    prop_compose! {
        fn arb_working()(
            content in "[a-zA-Z0-9 ]{1,100}",
            token_count in 0u32..10000,
            relevance in 0.0f32..=1.0,
            priority in arb_priority(),
        ) -> WorkingMemoryEntry {
            WorkingMemoryEntry::builder()
                .content(content)
                .agent_id(agent())
                .token_count(token_count)
                .relevance_score(relevance)
                .priority(priority)
                .build()
                .unwrap()
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn working_arrow_round_trip(entry in arb_working()) {
            let batch = working::to_batch(std::slice::from_ref(&entry)).unwrap();
            let decoded = working::from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), 1);
            let d = &decoded[0];
            prop_assert_eq!(&d.content, &entry.content);
            prop_assert_eq!(d.token_count, entry.token_count);
            prop_assert!((d.relevance_score - entry.relevance_score).abs() < f32::EPSILON);
            prop_assert_eq!(d.priority, entry.priority);
        }

        #[test]
        fn working_batch_size(entries in prop::collection::vec(arb_working(), 1..8)) {
            let batch = working::to_batch(&entries).unwrap();
            prop_assert_eq!(batch.num_rows(), entries.len());
            let decoded = working::from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), entries.len());
        }
    }

    // ── Procedural ──────────────────────────────────────────────

    prop_compose! {
        fn arb_procedural()(
            name in "[a-zA-Z_]{1,30}",
            description in "[a-zA-Z][a-zA-Z0-9 ]{0,99}",
            embedding in prop::option::of(arb_embedding()),
            n_steps in 0usize..3,
        ) -> ProceduralRecord {
            let steps: Vec<ActionStep> = (0..n_steps)
                .map(|i| ActionStep {
                    description: format!("step-{i}"),
                    tool: None,
                    parameters: BTreeMap::default(),
                })
                .collect();
            ProceduralRecord::builder()
                .name(name)
                .description(description)
                .steps(steps)
                .agent_id(agent())
                .build()
                .unwrap()
                .tap_mut(|r| r.embedding = embedding)
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn procedural_arrow_round_trip(record in arb_procedural()) {
            let batch = procedural::to_batch(std::slice::from_ref(&record), EMBED_DIMS).unwrap();
            let decoded = procedural::from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), 1);
            let d = &decoded[0];
            prop_assert_eq!(&d.name, &record.name);
            prop_assert_eq!(&d.description, &record.description);
            prop_assert_eq!(d.steps.len(), record.steps.len());
            prop_assert_eq!(&d.embedding, &record.embedding);
        }
    }

    // ── Graph Nodes ─────────────────────────────────────────────

    prop_compose! {
        fn arb_graph_node()(
            layer in arb_layer(),
            importance in 0.0f32..=1.0,
            access_count in 0u64..1000,
        ) -> GraphNodeData {
            GraphNodeData {
                id: MemoryId::new(),
                layer,
                importance,
                created_at: Timestamp::now(),
                namespace: Namespace::default_ns(),
                access_count,
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn graph_node_round_trip(node in arb_graph_node()) {
            let batch = graph::nodes_to_batch(std::slice::from_ref(&node)).unwrap();
            let decoded = graph::nodes_from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), 1);
            let d = &decoded[0];
            prop_assert_eq!(d.id, node.id);
            prop_assert_eq!(d.layer, node.layer);
            prop_assert!((d.importance - node.importance).abs() < f32::EPSILON);
        }

        #[test]
        fn graph_nodes_batch_size(nodes in prop::collection::vec(arb_graph_node(), 1..8)) {
            let batch = graph::nodes_to_batch(&nodes).unwrap();
            prop_assert_eq!(batch.num_rows(), nodes.len());
            let decoded = graph::nodes_from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), nodes.len());
        }
    }

    // ── Graph Edges ─────────────────────────────────────────────

    prop_compose! {
        fn arb_graph_edge()(
            relation in arb_edge_relation(),
            weight in 0.0f32..=1.0,
            co_retrieval_count in 0u64..1000,
        ) -> GraphEdge {
            GraphEdge {
                id: MemoryId::new(),
                source: MemoryId::new(),
                target: MemoryId::new(),
                relation,
                weight,
                co_retrieval_count,
                created_at: Timestamp::now(),
                updated_at: Timestamp::now(),
                valid_from: None,
                valid_until: None,
                metadata: BTreeMap::default(),
                resolved: false,
                namespace: Namespace::default(),
                causal: None,
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn graph_edge_round_trip(edge in arb_graph_edge()) {
            let batch = graph::edges_to_batch(std::slice::from_ref(&edge)).unwrap();
            let decoded = graph::edges_from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), 1);
            let d = &decoded[0];
            prop_assert_eq!(d.id, edge.id);
            prop_assert_eq!(d.source, edge.source);
            prop_assert_eq!(d.target, edge.target);
            prop_assert_eq!(d.relation, edge.relation);
            prop_assert!((d.weight - edge.weight).abs() < f32::EPSILON);
            prop_assert_eq!(d.co_retrieval_count, edge.co_retrieval_count);
        }

        #[test]
        fn graph_edges_batch_size(edges in prop::collection::vec(arb_graph_edge(), 1..8)) {
            let batch = graph::edges_to_batch(&edges).unwrap();
            prop_assert_eq!(batch.num_rows(), edges.len());
            let decoded = graph::edges_from_batch(&batch).unwrap();
            prop_assert_eq!(decoded.len(), edges.len());
        }
    }
}
