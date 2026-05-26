//! Proto serialization round-trip tests.
//!
//! Every message type defined in hirn.proto is constructed with all fields
//! populated, encoded to bytes, decoded back, and compared for equality.

use std::collections::HashMap;

use hirnd::proto::{self, *};
use prost::Message;

/// Encode a proto message to bytes, decode it back, and assert it equals the original.
fn roundtrip<M: Message + Default + PartialEq + std::fmt::Debug>(msg: &M) {
    let mut buf = Vec::new();
    msg.encode(&mut buf).expect("encode should not fail");
    let decoded = M::decode(buf.as_slice()).expect("decode should not fail");
    assert_eq!(*msg, decoded);
}

// ─── Fixture Helpers ─────────────────────────────────────────

fn sample_id(s: &str) -> MemoryId {
    MemoryId {
        value: s.to_string(),
    }
}

fn sample_ts(secs: i64, nanos: i32) -> Timestamp {
    Timestamp {
        seconds: secs,
        nanos,
    }
}

fn sample_metadata() -> HashMap<String, MetadataValue> {
    let mut m = HashMap::new();
    m.insert(
        "key_bool".into(),
        MetadataValue {
            value: Some(metadata_value::Value::BoolValue(true)),
        },
    );
    m.insert(
        "key_int".into(),
        MetadataValue {
            value: Some(metadata_value::Value::IntValue(42)),
        },
    );
    m.insert(
        "key_float".into(),
        MetadataValue {
            value: Some(metadata_value::Value::FloatValue(3.14)),
        },
    );
    m.insert(
        "key_string".into(),
        MetadataValue {
            value: Some(metadata_value::Value::StringValue("hello".into())),
        },
    );
    m
}

fn sample_entity_ref() -> EntityRef {
    EntityRef {
        name: "server-01".into(),
        role: "target".into(),
        entity_id: Some(sample_id("entity-001")),
    }
}

fn sample_concept_edge() -> ConceptEdge {
    ConceptEdge {
        target_id: Some(sample_id("concept-002")),
        relation: "PartOf".into(),
        weight: 0.85,
    }
}

fn sample_episodic() -> EpisodicRecord {
    EpisodicRecord {
        id: Some(sample_id("ep-001")),
        timestamp: Some(sample_ts(1700000000, 123456)),
        event_type: EventType::ToolCall.into(),
        content: "Deployed version 2.1".into(),
        summary: "v2.1 deployment".into(),
        entities: vec![sample_entity_ref()],
        embedding: vec![0.1, 0.2, 0.3],
        importance: 0.8,
        surprise: 0.4,
        access_count: 5,
        last_accessed: Some(sample_ts(1700001000, 0)),
        namespace: "private".into(),
        archived: false,
        metadata: sample_metadata(),
    }
}

fn sample_semantic() -> SemanticRecord {
    SemanticRecord {
        id: Some(sample_id("sem-001")),
        concept: "deployment pipeline".into(),
        knowledge_type: KnowledgeType::Prescriptive.into(),
        description: "Best practices for CI/CD pipelines".into(),
        embedding: vec![0.4, 0.5, 0.6],
        related_concepts: vec![sample_concept_edge()],
        confidence: 0.95,
        source_episodes: vec![sample_id("ep-001"), sample_id("ep-002")],
        evidence_count: 3,
        contradiction_ids: vec![sample_id("sem-099")],
        created_at: Some(sample_ts(1699000000, 0)),
        updated_at: Some(sample_ts(1700000000, 0)),
        access_count: 12,
        version: 2,
        namespace: "shared".into(),
    }
}

fn sample_working() -> WorkingMemoryEntry {
    WorkingMemoryEntry {
        id: Some(sample_id("wm-001")),
        content: "Current task: fix CI".into(),
        created_at: Some(sample_ts(1700000000, 0)),
        expires_at: Some(sample_ts(1700003600, 0)),
        relevance_score: 0.9,
        token_count: 15,
        source: Some(MemoryRef {
            layer: Layer::Episodic.into(),
            id: Some(sample_id("ep-001")),
        }),
        priority: Priority::High.into(),
        agent_id: "agent-alpha".into(),
    }
}

fn sample_scoring_weights() -> ScoringWeights {
    ScoringWeights {
        similarity: 1.0,
        importance: 0.8,
        recency: 0.6,
        activation: 0.4,
        causal_relevance: 0.2,
        source_reliability: 0.0,
    }
}

fn sample_score_breakdown() -> ScoreBreakdown {
    ScoreBreakdown {
        similarity: 0.92,
        importance: 0.75,
        recency: 0.50,
        activation: 0.30,
        causal_relevance: 0.10,
        surprise: 0.15,
        source_reliability: 0.0,
    }
}

fn sample_conflict_pair() -> ConflictPair {
    ConflictPair {
        memory_a: Some(sample_id("sem-001")),
        memory_b: Some(sample_id("sem-002")),
        content_a: "A is true".into(),
        content_b: "A is false".into(),
        confidence: 0.95,
        source_reliability_a: 1.0,
        source_reliability_b: 0.8,
    }
}

fn sample_conflict_group() -> ConflictGroup {
    ConflictGroup {
        conflict_id: "sem-001:sem-002".into(),
        members: vec![
            ConflictMember {
                memory_id: Some(sample_id("sem-001")),
                logical_memory_id: Some("logical-001".into()),
                revision_id: Some("revision-001".into()),
                status: ConflictMemberStatus::Active.into(),
                layer: Layer::Semantic.into(),
                content: "A is true".into(),
                in_result_set: true,
                source_reliability: 1.0,
            },
            ConflictMember {
                memory_id: Some(sample_id("sem-002")),
                logical_memory_id: Some("logical-002".into()),
                revision_id: Some("revision-002".into()),
                status: ConflictMemberStatus::Superseded.into(),
                layer: Layer::Semantic.into(),
                content: "A is false".into(),
                in_result_set: true,
                source_reliability: 0.8,
            },
        ],
        omitted_member_count: 1,
        pair_count: 2,
        confidence: 0.9,
        evidence_count: 3,
        source_reliability: 0.9,
        arbitration_status: ConflictArbitrationStatus::Superseded.into(),
        authoritative_memory_id: Some(sample_id("sem-001")),
        preferred_memory_id: None,
    }
}

fn sample_graph_edge() -> GraphEdge {
    GraphEdge {
        id: Some(sample_id("edge-001")),
        source: Some(sample_id("ep-001")),
        target: Some(sample_id("sem-001")),
        relation: EdgeRelation::DerivedFrom.into(),
        weight: 0.9,
        co_retrieval_count: 7,
        created_at: Some(sample_ts(1700000000, 0)),
        updated_at: Some(sample_ts(1700001000, 0)),
        valid_from: None,
        valid_until: None,
        metadata: sample_metadata(),
    }
}

fn sample_semantic_revision_summary() -> SemanticRevisionSummary {
    SemanticRevisionSummary {
        logical_memory_id: "logical-001".into(),
        current_revision_id: "revision-001".into(),
        head_revision_id: "revision-002".into(),
        current_state: "Superseded".into(),
        logical_state: "Active".into(),
        revision_count: 2,
        revisions: vec![
            SemanticRevisionEntry {
                memory_id: Some(sample_id("sem-001")),
                revision_id: "revision-001".into(),
                version: 1,
                operation: "Create".into(),
                state: "Superseded".into(),
                reason: None,
                created_at: Some(sample_ts(1699000000, 0)),
                superseded_by: Some(sample_id("sem-002")),
            },
            SemanticRevisionEntry {
                memory_id: Some(sample_id("sem-002")),
                revision_id: "revision-002".into(),
                version: 2,
                operation: "Correct".into(),
                state: "Active".into(),
                reason: Some("updated threshold".into()),
                created_at: Some(sample_ts(1700000000, 0)),
                superseded_by: None,
            },
        ],
    }
}

fn sample_revision_ref() -> RevisionRef {
    RevisionRef {
        logical_memory_id: "logical-001".into(),
        revision_id: "revision-002".into(),
        state: "Active".into(),
    }
}

fn sample_recall_snapshot() -> RecallSnapshot {
    RecallSnapshot {
        target: Some(recall_snapshot::Target::RecordedAt(sample_ts(
            1700000500, 0,
        ))),
    }
}

// ─── Core Identity ───────────────────────────────────────────

#[test]
fn roundtrip_memory_id() {
    roundtrip(&sample_id("test-id-001"));
}

#[test]
fn roundtrip_timestamp() {
    roundtrip(&sample_ts(1700000000, 999_999_999));
}

// ─── Metadata & References ───────────────────────────────────

#[test]
fn roundtrip_metadata_value_bool() {
    roundtrip(&MetadataValue {
        value: Some(metadata_value::Value::BoolValue(true)),
    });
}

#[test]
fn roundtrip_metadata_value_int() {
    roundtrip(&MetadataValue {
        value: Some(metadata_value::Value::IntValue(i64::MAX)),
    });
}

#[test]
fn roundtrip_metadata_value_float() {
    roundtrip(&MetadataValue {
        value: Some(metadata_value::Value::FloatValue(std::f64::consts::PI)),
    });
}

#[test]
fn roundtrip_metadata_value_string() {
    roundtrip(&MetadataValue {
        value: Some(metadata_value::Value::StringValue("hello world".into())),
    });
}

#[test]
fn roundtrip_metadata_value_none() {
    roundtrip(&MetadataValue { value: None });
}

#[test]
fn roundtrip_entity_ref() {
    roundtrip(&sample_entity_ref());
}

#[test]
fn roundtrip_entity_ref_no_id() {
    roundtrip(&EntityRef {
        name: "anon".into(),
        role: "observer".into(),
        entity_id: None,
    });
}

#[test]
fn roundtrip_concept_edge() {
    roundtrip(&sample_concept_edge());
}

#[test]
fn roundtrip_memory_ref() {
    roundtrip(&MemoryRef {
        layer: Layer::Semantic.into(),
        id: Some(sample_id("ref-001")),
    });
}

// ─── Memory Records ─────────────────────────────────────────

#[test]
fn roundtrip_episodic_record() {
    roundtrip(&sample_episodic());
}

#[test]
fn roundtrip_semantic_record() {
    roundtrip(&sample_semantic());
}

#[test]
fn roundtrip_working_memory_entry() {
    roundtrip(&sample_working());
}

#[test]
fn roundtrip_memory_record_working() {
    roundtrip(&MemoryRecord {
        record: Some(memory_record::Record::Working(sample_working())),
    });
}

#[test]
fn roundtrip_memory_record_episodic() {
    roundtrip(&MemoryRecord {
        record: Some(memory_record::Record::Episodic(sample_episodic())),
    });
}

#[test]
fn roundtrip_memory_record_semantic() {
    roundtrip(&MemoryRecord {
        record: Some(memory_record::Record::Semantic(sample_semantic())),
    });
}

#[test]
fn roundtrip_memory_record_none() {
    roundtrip(&MemoryRecord { record: None });
}

// ─── Scoring ─────────────────────────────────────────────────

#[test]
fn roundtrip_scoring_weights() {
    roundtrip(&sample_scoring_weights());
}

#[test]
fn roundtrip_score_breakdown() {
    roundtrip(&sample_score_breakdown());
}

#[test]
fn roundtrip_scored_memory() {
    roundtrip(&ScoredMemory {
        record: Some(MemoryRecord {
            record: Some(memory_record::Record::Episodic(sample_episodic())),
        }),
        score: 0.87,
        score_breakdown: Some(sample_score_breakdown()),
        revision: Some(sample_revision_ref()),
    });
}

// ─── Graph ───────────────────────────────────────────────────

#[test]
fn roundtrip_graph_edge() {
    roundtrip(&sample_graph_edge());
}

#[test]
fn roundtrip_neighbor_info() {
    roundtrip(&NeighborInfo {
        edge: Some(sample_graph_edge()),
        neighbor_id: Some(sample_id("neighbor-001")),
    });
}

// ─── Remember ────────────────────────────────────────────────

#[test]
fn roundtrip_remember_request_episodic() {
    roundtrip(&RememberRequest {
        record: Some(remember_request::Record::Episodic(sample_episodic())),
    });
}

#[test]
fn roundtrip_remember_request_semantic() {
    roundtrip(&RememberRequest {
        record: Some(remember_request::Record::Semantic(sample_semantic())),
    });
}

#[test]
fn roundtrip_remember_request_working() {
    roundtrip(&RememberRequest {
        record: Some(remember_request::Record::Working(sample_working())),
    });
}

#[test]
fn roundtrip_remember_response() {
    roundtrip(&RememberResponse {
        id: Some(sample_id("new-001")),
        layer: Layer::Episodic.into(),
    });
}

// ─── Recall ──────────────────────────────────────────────────

#[test]
fn roundtrip_recall_request_full() {
    roundtrip(&RecallRequest {
        query_embedding: vec![0.1, 0.2, 0.3],
        limit: 10,
        threshold: 0.5,
        layer_filter: Some(Layer::Episodic.into()),
        namespace: Some("private".into()),
        after: Some(sample_ts(1699000000, 0)),
        before: Some(sample_ts(1700000000, 0)),
        weights: Some(sample_scoring_weights()),
        activation_mode: ActivationMode::Spreading.into(),
        activation_depth: 3,
        snapshot: Some(sample_recall_snapshot()),
    });
}

#[test]
fn roundtrip_recall_request_minimal() {
    roundtrip(&RecallRequest {
        query_embedding: vec![0.5; 768],
        limit: 5,
        ..Default::default()
    });
}

#[test]
fn roundtrip_recall_result() {
    roundtrip(&RecallResult {
        record: Some(MemoryRecord {
            record: Some(memory_record::Record::Episodic(sample_episodic())),
        }),
        similarity: 0.92,
        composite_score: 0.85,
        revision: Some(sample_revision_ref()),
        score_breakdown: Some(sample_score_breakdown()),
    });
}

#[test]
fn roundtrip_recall_response() {
    roundtrip(&RecallResponse {
        results: vec![RecallResult {
            record: Some(MemoryRecord {
                record: Some(memory_record::Record::Semantic(sample_semantic())),
            }),
            similarity: 0.88,
            composite_score: 0.80,
            revision: Some(sample_revision_ref()),
            score_breakdown: Some(sample_score_breakdown()),
        }],
    });
}

// ─── Think ───────────────────────────────────────────────────

#[test]
fn roundtrip_think_request_full() {
    roundtrip(&ThinkRequest {
        query_embedding: vec![0.1, 0.2, 0.3],
        limit: 20,
        threshold: 0.3,
        layer_filter: Some(Layer::Semantic.into()),
        namespace: Some("shared".into()),
        after: Some(sample_ts(1699000000, 0)),
        before: Some(sample_ts(1700000000, 0)),
        weights: Some(sample_scoring_weights()),
        activation_mode: ActivationMode::Static.into(),
        activation_depth: 2,
        budget: 4096,
        format: ContextFormat::Json.into(),
    });
}

#[test]
fn roundtrip_think_response() {
    roundtrip(&ThinkResponse {
        context: "## Working Memory\n- Fix CI\n## Episodic\n- Deployed v2.1".into(),
        token_count: 42,
        records_included: vec![sample_id("ep-001"), sample_id("sem-001")],
        records_excluded_count: 3,
        contradictions: vec![sample_conflict_pair()],
        query_time_ms: 12.5,
        score_distribution: Some(ScoreDistribution {
            min: 0.3,
            max: 0.95,
            mean: 0.72,
        }),
        conflict_groups: vec![sample_conflict_group()],
    });
}

#[test]
fn roundtrip_conflict_pair() {
    roundtrip(&sample_conflict_pair());
}

#[test]
fn roundtrip_conflict_group() {
    roundtrip(&sample_conflict_group());
}

#[test]
fn roundtrip_score_distribution() {
    roundtrip(&ScoreDistribution {
        min: 0.1,
        max: 0.99,
        mean: 0.55,
    });
}

// ─── Forget ──────────────────────────────────────────────────

#[test]
fn roundtrip_forget_request_archive() {
    roundtrip(&ForgetRequest {
        id: Some(sample_id("ep-999")),
        mode: ForgetMode::Archive.into(),
    });
}

#[test]
fn roundtrip_forget_request_purge() {
    roundtrip(&ForgetRequest {
        id: Some(sample_id("ep-999")),
        mode: ForgetMode::Purge.into(),
    });
}

#[test]
fn roundtrip_forget_response() {
    roundtrip(&ForgetResponse {});
}

// ─── Focus / Defocus ─────────────────────────────────────────

#[test]
fn roundtrip_focus_request() {
    roundtrip(&FocusRequest {
        entry: Some(sample_working()),
    });
}

#[test]
fn roundtrip_focus_response() {
    roundtrip(&FocusResponse {
        id: Some(sample_id("wm-001")),
    });
}

#[test]
fn roundtrip_defocus_request() {
    roundtrip(&DefocusRequest {
        id: Some(sample_id("wm-001")),
    });
}

#[test]
fn roundtrip_defocus_response() {
    roundtrip(&DefocusResponse {});
}

// ─── Connect ─────────────────────────────────────────────────

#[test]
fn roundtrip_connect_request() {
    roundtrip(&ConnectRequest {
        source: Some(sample_id("ep-001")),
        target: Some(sample_id("sem-001")),
        relation: EdgeRelation::Supports.into(),
        weight: 0.75,
        metadata: sample_metadata(),
    });
}

#[test]
fn roundtrip_connect_response() {
    roundtrip(&ConnectResponse {
        edge_id: Some(sample_id("edge-new")),
    });
}

// ─── Inspect ─────────────────────────────────────────────────

#[test]
fn roundtrip_inspect_request() {
    roundtrip(&InspectRequest {
        id: Some(sample_id("ep-001")),
    });
}

#[test]
fn roundtrip_inspect_response() {
    roundtrip(&InspectResponse {
        record: Some(MemoryRecord {
            record: Some(memory_record::Record::Episodic(sample_episodic())),
        }),
        importance: 0.8,
        access_count: 5,
        last_accessed: Some(sample_ts(1700001000, 0)),
        neighbors: vec![NeighborInfo {
            edge: Some(sample_graph_edge()),
            neighbor_id: Some(sample_id("sem-001")),
        }],
        trust_score: 0.95,
        semantic_revision: Some(sample_semantic_revision_summary()),
        conflict_groups: vec![sample_conflict_group()],
    });
}

// ─── Trace ───────────────────────────────────────────────────

#[test]
fn roundtrip_trace_request() {
    roundtrip(&TraceRequest {
        id: Some(sample_id("sem-001")),
    });
}

#[test]
fn roundtrip_trace_response() {
    roundtrip(&TraceResponse {
        record: Some(MemoryRecord {
            record: Some(memory_record::Record::Semantic(sample_semantic())),
        }),
        source_episodes: vec![sample_id("ep-001"), sample_id("ep-002")],
        derived_records: vec![sample_id("sem-010")],
        mutation_count: 3,
        trust_score: 0.9,
        lineage_tree: "sem-001\n  ← ep-001\n  ← ep-002".into(),
        semantic_revision: Some(sample_semantic_revision_summary()),
        conflict_groups: vec![sample_conflict_group()],
    });
}

// ─── Execute ─────────────────────────────────────────────────

#[test]
fn roundtrip_execute_request() {
    roundtrip(&ExecuteRequest {
        query: "RECALL WHERE importance > 0.5 LIMIT 10".into(),
        allowed_namespaces: vec!["private".into(), "shared".into()],
    });
}

#[test]
fn roundtrip_execute_response_records() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Records(RecordResults {
            records: vec![ScoredMemory {
                record: Some(MemoryRecord {
                    record: Some(memory_record::Record::Episodic(sample_episodic())),
                }),
                score: 0.9,
                score_breakdown: Some(sample_score_breakdown()),
                revision: Some(sample_revision_ref()),
            }],
            query_time_ms: 5.2,
            records_scanned: 100,
            records_returned: 1,
            context: Some("assembled context".into()),
            conflicts: vec![sample_conflict_pair()],
            conflict_groups: vec![sample_conflict_group()],
        })),
    });
}

#[test]
fn roundtrip_execute_response_created() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Created(CreatedResult {
            id: Some(sample_id("new-001")),
            layer: Layer::Episodic.into(),
        })),
    });
}

#[test]
fn roundtrip_execute_response_forgotten() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Forgotten(ForgottenResult {
            target: "ep-999".into(),
            mode: "archive".into(),
        })),
    });
}

#[test]
fn roundtrip_execute_response_corrected() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Corrected(CorrectedResult {
            logical_memory_id: "logical-001".into(),
            prior_revision_id: "revision-001".into(),
            new_revision_id: "revision-002".into(),
        })),
    });
}

#[test]
fn roundtrip_execute_response_superseded() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Superseded(SupersededResult {
            logical_memory_id: "logical-001".into(),
            prior_revision_id: "revision-002".into(),
            new_revision_id: "revision-003".into(),
            reason: Some("new authority".into()),
        })),
    });
}

#[test]
fn roundtrip_execute_response_merged() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Merged(MergedResult {
            target_logical_memory_id: "logical-001".into(),
            prior_target_revision_id: "revision-002".into(),
            new_target_revision_id: "revision-003".into(),
            source_logical_memory_ids: vec!["logical-010".into(), "logical-011".into()],
            source_revision_ids: vec!["revision-010".into(), "revision-011".into()],
            reason: Some("deduplicate".into()),
        })),
    });
}

#[test]
fn roundtrip_execute_response_retracted() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Retracted(RetractedResult {
            logical_memory_id: "logical-001".into(),
            prior_revision_id: "revision-002".into(),
            tombstone_revision_id: "revision-003".into(),
            reason: Some("obsolete".into()),
        })),
    });
}

#[test]
fn roundtrip_execute_response_history() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::History(HistoryResult {
            semantic_revision: Some(sample_semantic_revision_summary()),
            items: vec![SemanticHistoryItem {
                record: Some(sample_semantic()),
                revision: Some(sample_semantic_revision_summary().revisions[0].clone()),
            }],
        })),
    });
}

#[test]
fn roundtrip_execute_response_inspected() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Inspected(InspectedResult {
            record: Some(MemoryRecord {
                record: Some(memory_record::Record::Episodic(sample_episodic())),
            }),
            importance: 0.8,
            access_count: 5,
            last_accessed: Some(sample_ts(1700001000, 0)),
            neighbors: vec![],
            trust_score: 0.95,
            semantic_revision: Some(sample_semantic_revision_summary()),
            conflict_groups: vec![sample_conflict_group()],
        })),
    });
}

#[test]
fn roundtrip_execute_response_traced() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Traced(TracedResult {
            record: Some(MemoryRecord {
                record: Some(memory_record::Record::Semantic(sample_semantic())),
            }),
            source_episodes: vec![sample_id("ep-001")],
            derived_records: vec![],
            mutation_count: 1,
            trust_score: 0.85,
            lineage_tree: "tree".into(),
            semantic_revision: Some(sample_semantic_revision_summary()),
            conflict_groups: vec![sample_conflict_group()],
        })),
    });
}

#[test]
fn roundtrip_execute_response_explain_plan() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::ExplainPlan(Box::new(
            ExplainPlanResult {
                plan_text: "HirnDirectInspect".into(),
                actual_result: Some(Box::new(ExecuteResponse {
                    result: Some(execute_response::Result::Inspected(InspectedResult {
                        record: Some(MemoryRecord {
                            record: Some(memory_record::Record::Episodic(sample_episodic())),
                        }),
                        importance: 0.8,
                        access_count: 5,
                        last_accessed: Some(sample_ts(1700001000, 0)),
                        neighbors: vec![],
                        trust_score: 0.95,
                        semantic_revision: Some(sample_semantic_revision_summary()),
                        conflict_groups: vec![sample_conflict_group()],
                    })),
                })),
                diagnostics: Some(QueryDiagnostics {
                    query_id: Some("01JEXPLAIN1234567890ABCDEF".into()),
                    authorize_us: Some(42),
                    embed_ms: Some(0.5),
                    optimize_ms: Some(0.15),
                    physical_plan_ms: Some(0.2),
                    execute_plan_ms: Some(0.65),
                    vector_search_ms: Some(1.0),
                    graph_expand_ms: Some(0.25),
                    rerank_ms: Some(0.1),
                    neural_rerank_ms: Some(0.0),
                    assemble_ms: Some(0.2),
                    total_ms: Some(2.1),
                    records_scanned: Some(3),
                    records_returned: Some(1),
                    threshold_filtered_count: Some(0),
                    competitive_inhibition_count: Some(0),
                    truncated_by_limit_count: Some(0),
                    raw_text_redacted_results: Some(0),
                    multivector_fallback_count: Some(0),
                    neural_rerank_fallback_count: Some(0),
                }),
            },
        ))),
    });
}

#[test]
fn roundtrip_execute_response_aggregated() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Aggregated(AggregatedResult {
            group_field: "importance".into(),
            function: "COUNT".into(),
            groups: vec![AggregatedGroup {
                key: "0.9".into(),
                value: 1.0,
            }],
            query_time_ms: 12.5,
            formatted: Some("[{\"importance\":\"0.9\",\"COUNT\":1.0}]".into()),
        })),
    });
}

#[test]
fn roundtrip_execute_response_policy() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Policy(PolicyResult {
            message: "1 policy".into(),
            policies: vec![PolicyEntry {
                name: "allow_default".into(),
                text: "permit(...)".into(),
            }],
        })),
    });
}

#[test]
fn roundtrip_execute_response_svo_events() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::SvoEvents(SvoEventQueryResult {
            events: vec![SvoEventEntry {
                source_memory_id: "mem-1".into(),
                subject: "Alice".into(),
                verb: "diagnosed".into(),
                object: "outage".into(),
                time_start: Some("2026-01-01T00:00:00Z".into()),
                time_end: None,
                confidence: 0.91,
            }],
            events_returned: 1,
        })),
    });
}

#[test]
fn roundtrip_execute_response_causal() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Causal(CausalResult {
            kind: "EXPLAIN CAUSES".into(),
            query_time_ms: 4.2,
            rows: vec![CausalRow {
                columns: vec![
                    CausalColumn {
                        key: "cause_content".into(),
                        value: "deploy timeout".into(),
                    },
                    CausalColumn {
                        key: "depth".into(),
                        value: "1".into(),
                    },
                ],
            }],
        })),
    });
}

#[test]
fn roundtrip_execute_response_consolidated() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::Consolidated(ConsolidatedResult {
            records_processed: 42,
        })),
    });
}

#[test]
fn roundtrip_execute_response_watch_ack() {
    roundtrip(&ExecuteResponse {
        result: Some(execute_response::Result::WatchAck(WatchAckResult {
            message: "subscribed".into(),
        })),
    });
}

#[test]
fn roundtrip_execute_response_none() {
    roundtrip(&ExecuteResponse { result: None });
}

// ─── Standalone Execute result types ─────────────────────────

#[test]
fn roundtrip_record_results() {
    roundtrip(&RecordResults {
        records: vec![],
        query_time_ms: 0.5,
        records_scanned: 50,
        records_returned: 0,
        context: None,
        conflicts: vec![],
        conflict_groups: vec![],
    });
}

#[test]
fn roundtrip_query_diagnostics() {
    roundtrip(&QueryDiagnostics {
        query_id: Some("01JEXPLAIN1234567890ABCDEF".into()),
        authorize_us: Some(42),
        embed_ms: Some(0.5),
        optimize_ms: Some(0.15),
        physical_plan_ms: Some(0.2),
        execute_plan_ms: Some(0.65),
        vector_search_ms: Some(1.0),
        graph_expand_ms: Some(0.25),
        rerank_ms: Some(0.1),
        neural_rerank_ms: Some(0.0),
        assemble_ms: Some(0.2),
        total_ms: Some(2.1),
        records_scanned: Some(3),
        records_returned: Some(1),
        threshold_filtered_count: Some(0),
        competitive_inhibition_count: Some(0),
        truncated_by_limit_count: Some(0),
        raw_text_redacted_results: Some(0),
        multivector_fallback_count: Some(0),
        neural_rerank_fallback_count: Some(0),
    });
}

#[test]
fn roundtrip_explain_plan_result() {
    roundtrip(&ExplainPlanResult {
        plan_text: "HirnDirectInspect".into(),
        actual_result: Some(Box::new(ExecuteResponse {
            result: Some(execute_response::Result::Created(CreatedResult {
                id: Some(sample_id("new-001")),
                layer: Layer::Episodic.into(),
            })),
        })),
        diagnostics: Some(QueryDiagnostics {
            query_id: Some("01JEXPLAIN1234567890ABCDEF".into()),
            authorize_us: Some(42),
            embed_ms: None,
            optimize_ms: None,
            physical_plan_ms: None,
            execute_plan_ms: None,
            vector_search_ms: None,
            graph_expand_ms: None,
            rerank_ms: None,
            neural_rerank_ms: None,
            assemble_ms: None,
            total_ms: Some(1.5),
            records_scanned: None,
            records_returned: Some(1),
            threshold_filtered_count: None,
            competitive_inhibition_count: None,
            truncated_by_limit_count: None,
            raw_text_redacted_results: None,
            multivector_fallback_count: None,
            neural_rerank_fallback_count: None,
        }),
    });
}

#[test]
fn roundtrip_aggregated_result() {
    roundtrip(&AggregatedResult {
        group_field: "importance".into(),
        function: "COUNT".into(),
        groups: vec![AggregatedGroup {
            key: "0.4".into(),
            value: 2.0,
        }],
        query_time_ms: 7.5,
        formatted: Some("[{\"importance\":\"0.4\",\"COUNT\":2.0}]".into()),
    });
}

#[test]
fn roundtrip_policy_result() {
    roundtrip(&PolicyResult {
        message: "Decision: ALLOW".into(),
        policies: vec![PolicyEntry {
            name: "allow_ops".into(),
            text: "permit(...)".into(),
        }],
    });
}

#[test]
fn roundtrip_svo_event_query_result() {
    roundtrip(&SvoEventQueryResult {
        events: vec![SvoEventEntry {
            source_memory_id: "mem-42".into(),
            subject: "Alice".into(),
            verb: "diagnosed".into(),
            object: "outage".into(),
            time_start: Some("2026-03-01T00:00:00Z".into()),
            time_end: Some("2026-03-01T01:00:00Z".into()),
            confidence: 0.88,
        }],
        events_returned: 1,
    });
}

#[test]
fn roundtrip_causal_result() {
    roundtrip(&CausalResult {
        kind: "WHAT_IF".into(),
        query_time_ms: 8.4,
        rows: vec![CausalRow {
            columns: vec![
                CausalColumn {
                    key: "outcome".into(),
                    value: "fewer errors".into(),
                },
                CausalColumn {
                    key: "probability".into(),
                    value: "0.72".into(),
                },
            ],
        }],
    });
}

#[test]
fn roundtrip_created_result() {
    roundtrip(&CreatedResult {
        id: Some(sample_id("x")),
        layer: Layer::Working.into(),
    });
}

#[test]
fn roundtrip_forgotten_result() {
    roundtrip(&ForgottenResult {
        target: "id".into(),
        mode: "purge".into(),
    });
}

#[test]
fn roundtrip_corrected_result() {
    roundtrip(&CorrectedResult {
        logical_memory_id: "logical-001".into(),
        prior_revision_id: "revision-001".into(),
        new_revision_id: "revision-002".into(),
    });
}

#[test]
fn roundtrip_superseded_result() {
    roundtrip(&SupersededResult {
        logical_memory_id: "logical-001".into(),
        prior_revision_id: "revision-002".into(),
        new_revision_id: "revision-003".into(),
        reason: Some("new authority".into()),
    });
}

#[test]
fn roundtrip_retracted_result() {
    roundtrip(&RetractedResult {
        logical_memory_id: "logical-001".into(),
        prior_revision_id: "revision-002".into(),
        tombstone_revision_id: "revision-003".into(),
        reason: Some("obsolete".into()),
    });
}

#[test]
fn roundtrip_inspected_result() {
    roundtrip(&InspectedResult {
        record: None,
        importance: 0.0,
        access_count: 0,
        last_accessed: None,
        neighbors: vec![],
        trust_score: 0.0,
        semantic_revision: Some(sample_semantic_revision_summary()),
        conflict_groups: vec![sample_conflict_group()],
    });
}

#[test]
fn roundtrip_traced_result() {
    roundtrip(&TracedResult {
        record: None,
        source_episodes: vec![],
        derived_records: vec![],
        mutation_count: 0,
        trust_score: 0.0,
        lineage_tree: String::new(),
        semantic_revision: Some(sample_semantic_revision_summary()),
        conflict_groups: vec![sample_conflict_group()],
    });
}

#[test]
fn roundtrip_history_result() {
    roundtrip(&HistoryResult {
        semantic_revision: Some(sample_semantic_revision_summary()),
        items: vec![SemanticHistoryItem {
            record: Some(sample_semantic()),
            revision: Some(sample_semantic_revision_summary().revisions[0].clone()),
        }],
    });
}

#[test]
fn roundtrip_semantic_history_item() {
    roundtrip(&SemanticHistoryItem {
        record: Some(sample_semantic()),
        revision: Some(sample_semantic_revision_summary().revisions[0].clone()),
    });
}

#[test]
fn roundtrip_semantic_revision_entry() {
    roundtrip(&sample_semantic_revision_summary().revisions[0]);
}

#[test]
fn roundtrip_semantic_revision_summary() {
    roundtrip(&sample_semantic_revision_summary());
}

#[test]
fn roundtrip_consolidated_result() {
    roundtrip(&ConsolidatedResult {
        records_processed: 0,
    });
}

#[test]
fn roundtrip_watch_ack_result() {
    roundtrip(&WatchAckResult {
        message: String::new(),
    });
}

// ─── Consolidate ─────────────────────────────────────────────

#[test]
fn roundtrip_consolidate_request_full() {
    roundtrip(&ConsolidateRequest {
        topic_threshold: Some(0.7),
        surprise_threshold: Some(0.5),
        temporal_gap_secs: Some(3600),
        archive: true,
    });
}

#[test]
fn roundtrip_consolidate_request_minimal() {
    roundtrip(&ConsolidateRequest {
        archive: false,
        ..Default::default()
    });
}

#[test]
fn roundtrip_consolidate_response() {
    roundtrip(&ConsolidateResponse {
        records_processed: 100,
        segments_created: 5,
        patterns_detected: 3,
        threads_formed: 2,
        concepts_extracted: 7,
        provenance_edges_created: 14,
        episodes_archived: 80,
        execution_time_ms: 350.0,
    });
}

// ─── Stats ───────────────────────────────────────────────────

#[test]
fn roundtrip_stats_request() {
    roundtrip(&StatsRequest {});
}

#[test]
fn roundtrip_stats_response() {
    roundtrip(&StatsResponse {
        working_count: 10,
        episodic_count: 500,
        semantic_count: 50,
        total_count: 560,
        file_size_bytes: 1_048_576,
    });
}

// ─── Namespace ───────────────────────────────────────────────

#[test]
fn roundtrip_create_namespace_request() {
    roundtrip(&CreateNamespaceRequest {
        name: "team-backend".into(),
        kind: "team".into(),
        member_agent_ids: vec!["agent-a".into(), "agent-b".into()],
    });
}

#[test]
fn roundtrip_create_namespace_response() {
    roundtrip(&CreateNamespaceResponse {});
}

#[test]
fn roundtrip_share_memory_request() {
    roundtrip(&ShareMemoryRequest {
        id: Some(sample_id("ep-001")),
        target_namespace: "shared".into(),
        agent_id: "agent-alpha".into(),
    });
}

#[test]
fn roundtrip_share_memory_response() {
    roundtrip(&ShareMemoryResponse {
        new_id: Some(sample_id("ep-copy-001")),
    });
}

// ─── Watch ───────────────────────────────────────────────────

#[test]
fn roundtrip_watch_request_full() {
    roundtrip(&WatchRequest {
        layer_filter: Some(Layer::Episodic.into()),
        entities: vec!["production".into(), "server-01".into()],
        min_importance: Some(0.7),
        namespace: Some("private".into()),
    });
}

#[test]
fn roundtrip_watch_request_minimal() {
    roundtrip(&WatchRequest::default());
}

#[test]
fn roundtrip_watch_event_created() {
    roundtrip(&proto::WatchEvent {
        event_type: WatchEventType::Created.into(),
        record: Some(MemoryRecord {
            record: Some(memory_record::Record::Episodic(sample_episodic())),
        }),
        timestamp: Some(sample_ts(1700000000, 0)),
        description: Some("New episodic record created".into()),
    });
}

#[test]
fn roundtrip_watch_event_updated() {
    roundtrip(&proto::WatchEvent {
        event_type: WatchEventType::Updated.into(),
        record: Some(MemoryRecord {
            record: Some(memory_record::Record::Semantic(sample_semantic())),
        }),
        timestamp: Some(sample_ts(1700000000, 0)),
        description: None,
    });
}

#[test]
fn roundtrip_watch_event_consolidated() {
    roundtrip(&proto::WatchEvent {
        event_type: WatchEventType::Consolidated.into(),
        record: None,
        timestamp: Some(sample_ts(1700000000, 0)),
        description: Some("Consolidation completed".into()),
    });
}

#[test]
fn roundtrip_watch_event_conflict() {
    roundtrip(&proto::WatchEvent {
        event_type: WatchEventType::Conflict.into(),
        record: Some(MemoryRecord {
            record: Some(memory_record::Record::Semantic(sample_semantic())),
        }),
        timestamp: Some(sample_ts(1700000000, 0)),
        description: Some("Contradiction found".into()),
    });
}

// ─── Enum coverage ───────────────────────────────────────────
// Verify all enum values survive round-trip through a carrier message.

#[test]
fn roundtrip_all_layers() {
    for layer in [
        Layer::Unspecified,
        Layer::Working,
        Layer::Episodic,
        Layer::Semantic,
    ] {
        let msg = RememberResponse {
            id: Some(sample_id("x")),
            layer: layer.into(),
        };
        roundtrip(&msg);
    }
}

#[test]
fn roundtrip_all_event_types() {
    for et in [
        EventType::Unspecified,
        EventType::Conversation,
        EventType::ToolCall,
        EventType::Observation,
        EventType::Experiment,
        EventType::Error,
        EventType::Decision,
    ] {
        let msg = EpisodicRecord {
            event_type: et.into(),
            content: "test".into(),
            ..Default::default()
        };
        roundtrip(&msg);
    }
}

#[test]
fn roundtrip_all_priorities() {
    for p in [
        Priority::Unspecified,
        Priority::Normal,
        Priority::High,
        Priority::Critical,
    ] {
        let msg = WorkingMemoryEntry {
            priority: p.into(),
            content: "test".into(),
            ..Default::default()
        };
        roundtrip(&msg);
    }
}

#[test]
fn roundtrip_all_knowledge_types() {
    for kt in [
        KnowledgeType::Unspecified,
        KnowledgeType::Propositional,
        KnowledgeType::Prescriptive,
        KnowledgeType::Taxonomic,
    ] {
        let msg = SemanticRecord {
            knowledge_type: kt.into(),
            concept: "test".into(),
            ..Default::default()
        };
        roundtrip(&msg);
    }
}

#[test]
fn roundtrip_all_edge_relations() {
    for r in [
        EdgeRelation::Unspecified,
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
        let msg = ConnectRequest {
            source: Some(sample_id("a")),
            target: Some(sample_id("b")),
            relation: r.into(),
            weight: 1.0,
            metadata: Default::default(),
        };
        roundtrip(&msg);
    }
}

#[test]
fn roundtrip_all_activation_modes() {
    for m in [
        ActivationMode::Unspecified,
        ActivationMode::None,
        ActivationMode::Static,
        ActivationMode::Spreading,
    ] {
        let msg = RecallRequest {
            activation_mode: m.into(),
            query_embedding: vec![0.1],
            ..Default::default()
        };
        roundtrip(&msg);
    }
}

#[test]
fn roundtrip_all_context_formats() {
    for f in [
        ContextFormat::Unspecified,
        ContextFormat::Structured,
        ContextFormat::Narrative,
        ContextFormat::Json,
    ] {
        let msg = ThinkRequest {
            format: f.into(),
            query_embedding: vec![0.1],
            ..Default::default()
        };
        roundtrip(&msg);
    }
}

#[test]
fn roundtrip_all_forget_modes() {
    for m in [
        ForgetMode::Unspecified,
        ForgetMode::Archive,
        ForgetMode::Purge,
    ] {
        let msg = ForgetRequest {
            id: Some(sample_id("x")),
            mode: m.into(),
        };
        roundtrip(&msg);
    }
}

#[test]
fn roundtrip_all_watch_event_types() {
    for t in [
        WatchEventType::Unspecified,
        WatchEventType::Created,
        WatchEventType::Updated,
        WatchEventType::Consolidated,
        WatchEventType::Conflict,
    ] {
        let msg = proto::WatchEvent {
            event_type: t.into(),
            ..Default::default()
        };
        roundtrip(&msg);
    }
}

// ─── Edge cases ──────────────────────────────────────────────

#[test]
fn roundtrip_empty_embedding() {
    let ep = EpisodicRecord {
        content: "no embedding".into(),
        embedding: vec![],
        ..Default::default()
    };
    roundtrip(&ep);
}

#[test]
fn roundtrip_large_embedding() {
    let ep = EpisodicRecord {
        content: "large embedding".into(),
        embedding: vec![0.001; 1536],
        ..Default::default()
    };
    roundtrip(&ep);
}

#[test]
fn roundtrip_unicode_content() {
    let ep = EpisodicRecord {
        content: "日本語テスト 🧠 émojis и кириллица".into(),
        summary: "Ünïcödé".into(),
        ..Default::default()
    };
    roundtrip(&ep);
}

#[test]
fn roundtrip_max_numeric_values() {
    let ts = Timestamp {
        seconds: i64::MAX,
        nanos: 999_999_999,
    };
    roundtrip(&ts);

    let stats = StatsResponse {
        working_count: u64::MAX,
        episodic_count: u64::MAX,
        semantic_count: u64::MAX,
        total_count: u64::MAX,
        file_size_bytes: u64::MAX,
    };
    roundtrip(&stats);
}

#[test]
fn roundtrip_deeply_nested() {
    // ExecuteResponse → RecordResults → ScoredMemory → MemoryRecord → EpisodicRecord
    // with all sub-messages populated, including metadata maps and repeated fields.
    let msg = ExecuteResponse {
        result: Some(execute_response::Result::Records(RecordResults {
            records: vec![
                ScoredMemory {
                    record: Some(MemoryRecord {
                        record: Some(memory_record::Record::Episodic(sample_episodic())),
                    }),
                    score: 0.95,
                    score_breakdown: Some(sample_score_breakdown()),
                    revision: Some(sample_revision_ref()),
                },
                ScoredMemory {
                    record: Some(MemoryRecord {
                        record: Some(memory_record::Record::Semantic(sample_semantic())),
                    }),
                    score: 0.80,
                    score_breakdown: Some(sample_score_breakdown()),
                    revision: Some(sample_revision_ref()),
                },
                ScoredMemory {
                    record: Some(MemoryRecord {
                        record: Some(memory_record::Record::Working(sample_working())),
                    }),
                    score: 0.70,
                    score_breakdown: Some(sample_score_breakdown()),
                    revision: Some(sample_revision_ref()),
                },
            ],
            query_time_ms: 25.3,
            records_scanned: 1000,
            records_returned: 3,
            context: Some("Full context with all three layers".into()),
            conflicts: vec![sample_conflict_pair()],
            conflict_groups: vec![sample_conflict_group()],
        })),
    };
    roundtrip(&msg);
}

#[test]
fn roundtrip_all_defaults() {
    // Every message with default values should also round-trip cleanly.
    roundtrip(&MemoryId::default());
    roundtrip(&Timestamp::default());
    roundtrip(&MetadataValue::default());
    roundtrip(&EntityRef::default());
    roundtrip(&ConceptEdge::default());
    roundtrip(&MemoryRef::default());
    roundtrip(&EpisodicRecord::default());
    roundtrip(&SemanticRecord::default());
    roundtrip(&WorkingMemoryEntry::default());
    roundtrip(&MemoryRecord::default());
    roundtrip(&ScoringWeights::default());
    roundtrip(&ScoreBreakdown::default());
    roundtrip(&RevisionRef::default());
    roundtrip(&ScoredMemory::default());
    roundtrip(&ConflictPair::default());
    roundtrip(&ConflictMember::default());
    roundtrip(&ConflictGroup::default());
    roundtrip(&GraphEdge::default());
    roundtrip(&NeighborInfo::default());
    roundtrip(&RememberRequest::default());
    roundtrip(&RememberResponse::default());
    roundtrip(&RecallSnapshot::default());
    roundtrip(&RecallRequest::default());
    roundtrip(&RecallResult::default());
    roundtrip(&RecallResponse::default());
    roundtrip(&ThinkRequest::default());
    roundtrip(&ThinkResponse::default());
    roundtrip(&ConflictPair::default());
    roundtrip(&ScoreDistribution::default());
    roundtrip(&ForgetRequest::default());
    roundtrip(&ForgetResponse::default());
    roundtrip(&FocusRequest::default());
    roundtrip(&FocusResponse::default());
    roundtrip(&DefocusRequest::default());
    roundtrip(&DefocusResponse::default());
    roundtrip(&ConnectRequest::default());
    roundtrip(&ConnectResponse::default());
    roundtrip(&InspectRequest::default());
    roundtrip(&InspectResponse::default());
    roundtrip(&TraceRequest::default());
    roundtrip(&TraceResponse::default());
    roundtrip(&ExecuteRequest::default());
    roundtrip(&ExecuteResponse::default());
    roundtrip(&RecordResults::default());
    roundtrip(&CreatedResult::default());
    roundtrip(&ForgottenResult::default());
    roundtrip(&InspectedResult::default());
    roundtrip(&HistoryResult::default());
    roundtrip(&SemanticHistoryItem::default());
    roundtrip(&TracedResult::default());
    roundtrip(&ConsolidatedResult::default());
    roundtrip(&WatchAckResult::default());
    roundtrip(&ConsolidateRequest::default());
    roundtrip(&ConsolidateResponse::default());
    roundtrip(&StatsRequest::default());
    roundtrip(&StatsResponse::default());
    roundtrip(&CreateNamespaceRequest::default());
    roundtrip(&CreateNamespaceResponse::default());
    roundtrip(&ShareMemoryRequest::default());
    roundtrip(&ShareMemoryResponse::default());
    roundtrip(&WatchRequest::default());
    roundtrip(&proto::WatchEvent::default());
}
