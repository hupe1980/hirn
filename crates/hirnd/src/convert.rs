/// Conversions between proto-generated types and hirn engine types.
use std::collections::BTreeMap;

use hirn::prelude::*;
use hirn_core::metadata::MetadataValue;
use hirn_engine::ql::context::{
    ConflictArbitrationStatus, ConflictGroup, ConflictMember, ConflictMemberStatus, ContextFormat,
};
use hirn_engine::ql::{
    ExplainResult, QueryResult, ScoreBreakdown, ScoredMemory, SemanticRevisionEntry,
    SemanticRevisionSummary,
};

use crate::proto;

// ─── Timestamp ───────────────────────────────────────────────

/// Convert a [`Timestamp`] to its protobuf representation.
pub fn timestamp_to_proto(ts: &Timestamp) -> proto::Timestamp {
    let dt = ts.as_datetime();
    proto::Timestamp {
        seconds: dt.timestamp(),
        nanos: dt.timestamp_subsec_nanos() as i32,
    }
}

/// Convert a protobuf timestamp to a [`Timestamp`].
pub fn timestamp_from_proto(ts: &proto::Timestamp) -> Timestamp {
    use chrono::{DateTime, Utc};
    let dt = DateTime::<Utc>::from_timestamp(ts.seconds, ts.nanos as u32).unwrap_or_else(Utc::now);
    Timestamp::from(dt)
}

pub fn recall_snapshot_from_proto(snapshot: &proto::RecallSnapshot) -> HirnResult<RecallSnapshot> {
    match snapshot.target.as_ref() {
        Some(proto::recall_snapshot::Target::ObservedAt(ts)) => {
            Ok(RecallSnapshot::observed(timestamp_from_proto(ts)))
        }
        Some(proto::recall_snapshot::Target::RecordedAt(ts)) => {
            Ok(RecallSnapshot::recorded(timestamp_from_proto(ts)))
        }
        Some(proto::recall_snapshot::Target::RevisionId(revision_id)) => {
            let revision_id = RevisionId::parse(revision_id).map_err(|e| {
                HirnError::InvalidInput(format!("invalid revision_id in recall snapshot: {e}"))
            })?;
            Ok(RecallSnapshot::revision(revision_id))
        }
        None => Err(HirnError::InvalidInput(
            "recall snapshot target is required".to_string(),
        )),
    }
}

// ─── MemoryId ────────────────────────────────────────────────

/// Convert a [`MemoryId`] to its protobuf representation.
pub fn memory_id_to_proto(id: &MemoryId) -> proto::MemoryId {
    proto::MemoryId {
        value: id.to_string(),
    }
}

/// Convert a protobuf memory ID to a [`MemoryId`].
pub fn memory_id_from_proto(id: &proto::MemoryId) -> HirnResult<MemoryId> {
    parse_memory_id(&id.value)
}

/// Parse a ULID string into a `MemoryId`.
pub fn parse_memory_id(s: &str) -> HirnResult<MemoryId> {
    let ulid = ulid::Ulid::from_string(s)
        .map_err(|e| HirnError::InvalidInput(format!("invalid memory ID: {e}")))?;
    Ok(MemoryId::from_ulid(ulid))
}

pub fn semantic_revision_entry_to_proto(
    entry: &SemanticRevisionEntry,
) -> proto::SemanticRevisionEntry {
    proto::SemanticRevisionEntry {
        memory_id: Some(memory_id_to_proto(&entry.memory_id)),
        revision_id: entry.revision_id.to_string(),
        version: entry.version,
        operation: format!("{:?}", entry.operation),
        state: format!("{:?}", entry.state),
        reason: entry.reason.clone(),
        created_at: Some(timestamp_to_proto(&entry.created_at)),
        superseded_by: entry.superseded_by.as_ref().map(memory_id_to_proto),
    }
}

pub fn semantic_revision_summary_to_proto(
    summary: &SemanticRevisionSummary,
) -> proto::SemanticRevisionSummary {
    proto::SemanticRevisionSummary {
        logical_memory_id: summary.logical_memory_id.to_string(),
        current_revision_id: summary.current_revision_id.to_string(),
        head_revision_id: summary.head_revision_id.to_string(),
        current_state: format!("{:?}", summary.current_state),
        logical_state: format!("{:?}", summary.logical_state),
        revision_count: u32::try_from(summary.revision_count).unwrap_or(u32::MAX),
        revisions: summary
            .revisions
            .iter()
            .map(semantic_revision_entry_to_proto)
            .collect(),
    }
}

pub fn revision_ref_to_proto(revision: &hirn_core::revision::RevisionRef) -> proto::RevisionRef {
    proto::RevisionRef {
        logical_memory_id: revision.logical_memory_id.to_string(),
        revision_id: revision.revision_id.to_string(),
        state: format!("{:?}", revision.state),
    }
}

// ─── Enums ───────────────────────────────────────────────────

/// Convert a [`Layer`] to its protobuf `i32` discriminant.
pub fn layer_to_proto(layer: &Layer) -> i32 {
    match layer {
        Layer::Working => proto::Layer::Working as i32,
        Layer::Episodic => proto::Layer::Episodic as i32,
        Layer::Semantic => proto::Layer::Semantic as i32,
        Layer::Procedural => proto::Layer::Procedural as i32,
    }
}

/// Convert a protobuf discriminant back to a [`Layer`].
pub fn layer_from_proto(v: i32) -> Option<Layer> {
    match proto::Layer::try_from(v) {
        Ok(proto::Layer::Working) => Some(Layer::Working),
        Ok(proto::Layer::Episodic) => Some(Layer::Episodic),
        Ok(proto::Layer::Semantic) => Some(Layer::Semantic),
        Ok(proto::Layer::Procedural) => Some(Layer::Procedural),
        _ => None,
    }
}

/// Convert an [`EventType`] to its protobuf discriminant.
pub fn event_type_to_proto(et: &EventType) -> i32 {
    match et {
        EventType::Conversation => proto::EventType::Conversation as i32,
        EventType::ToolCall => proto::EventType::ToolCall as i32,
        EventType::Observation => proto::EventType::Observation as i32,
        EventType::Experiment => proto::EventType::Experiment as i32,
        EventType::Error => proto::EventType::Error as i32,
        EventType::Decision => proto::EventType::Decision as i32,
    }
}

/// Convert a protobuf discriminant back to an [`EventType`].
pub fn event_type_from_proto(v: i32) -> EventType {
    match proto::EventType::try_from(v) {
        Ok(proto::EventType::Conversation) => EventType::Conversation,
        Ok(proto::EventType::ToolCall) => EventType::ToolCall,
        Ok(proto::EventType::Observation) => EventType::Observation,
        Ok(proto::EventType::Experiment) => EventType::Experiment,
        Ok(proto::EventType::Error) => EventType::Error,
        Ok(proto::EventType::Decision) => EventType::Decision,
        _ => EventType::Observation,
    }
}

/// Convert a [`Priority`] to its protobuf discriminant.
pub fn priority_to_proto(p: &Priority) -> i32 {
    match p {
        Priority::Normal => proto::Priority::Normal as i32,
        Priority::High => proto::Priority::High as i32,
        Priority::Critical => proto::Priority::Critical as i32,
    }
}

/// Convert a protobuf discriminant back to a [`Priority`].
pub fn priority_from_proto(v: i32) -> Priority {
    match proto::Priority::try_from(v) {
        Ok(proto::Priority::High) => Priority::High,
        Ok(proto::Priority::Critical) => Priority::Critical,
        _ => Priority::Normal,
    }
}

/// Convert a [`KnowledgeType`] to its protobuf discriminant.
pub fn knowledge_type_to_proto(kt: &KnowledgeType) -> i32 {
    match kt {
        KnowledgeType::Propositional => proto::KnowledgeType::Propositional as i32,
        KnowledgeType::Prescriptive => proto::KnowledgeType::Prescriptive as i32,
        KnowledgeType::Taxonomic => proto::KnowledgeType::Taxonomic as i32,
        // Inferred, Community, and RaptorSummary don't have proto equivalents yet; map to Propositional.
        KnowledgeType::Inferred | KnowledgeType::Community | KnowledgeType::RaptorSummary => {
            proto::KnowledgeType::Propositional as i32
        }
    }
}

/// Convert a protobuf discriminant back to a [`KnowledgeType`].
pub fn knowledge_type_from_proto(v: i32) -> KnowledgeType {
    match proto::KnowledgeType::try_from(v) {
        Ok(proto::KnowledgeType::Prescriptive) => KnowledgeType::Prescriptive,
        Ok(proto::KnowledgeType::Taxonomic) => KnowledgeType::Taxonomic,
        _ => KnowledgeType::Propositional,
    }
}

/// Convert an [`EdgeRelation`] to its protobuf discriminant.
pub fn edge_relation_to_proto(er: &EdgeRelation) -> i32 {
    match er {
        EdgeRelation::RelatedTo => proto::EdgeRelation::RelatedTo as i32,
        EdgeRelation::Causes => proto::EdgeRelation::Causes as i32,
        EdgeRelation::CausedBy => proto::EdgeRelation::CausedBy as i32,
        EdgeRelation::DerivedFrom => proto::EdgeRelation::DerivedFrom as i32,
        EdgeRelation::Contradicts => proto::EdgeRelation::Contradicts as i32,
        EdgeRelation::Supports => proto::EdgeRelation::Supports as i32,
        EdgeRelation::TemporalNext => proto::EdgeRelation::TemporalNext as i32,
        EdgeRelation::PartOf => proto::EdgeRelation::PartOf as i32,
        EdgeRelation::InstanceOf => proto::EdgeRelation::InstanceOf as i32,
        EdgeRelation::SimilarTo => proto::EdgeRelation::SimilarTo as i32,
        EdgeRelation::Inhibits => proto::EdgeRelation::Inhibits as i32,
        EdgeRelation::ParticipatesIn => proto::EdgeRelation::ParticipatesIn as i32,
    }
}

/// Convert a protobuf discriminant back to an [`EdgeRelation`].
pub fn edge_relation_from_proto(v: i32) -> EdgeRelation {
    match proto::EdgeRelation::try_from(v) {
        Ok(proto::EdgeRelation::Causes) => EdgeRelation::Causes,
        Ok(proto::EdgeRelation::CausedBy) => EdgeRelation::CausedBy,
        Ok(proto::EdgeRelation::DerivedFrom) => EdgeRelation::DerivedFrom,
        Ok(proto::EdgeRelation::Contradicts) => EdgeRelation::Contradicts,
        Ok(proto::EdgeRelation::Supports) => EdgeRelation::Supports,
        Ok(proto::EdgeRelation::TemporalNext) => EdgeRelation::TemporalNext,
        Ok(proto::EdgeRelation::PartOf) => EdgeRelation::PartOf,
        Ok(proto::EdgeRelation::InstanceOf) => EdgeRelation::InstanceOf,
        Ok(proto::EdgeRelation::SimilarTo) => EdgeRelation::SimilarTo,
        Ok(proto::EdgeRelation::Inhibits) => EdgeRelation::Inhibits,
        Ok(proto::EdgeRelation::ParticipatesIn) => EdgeRelation::ParticipatesIn,
        _ => EdgeRelation::RelatedTo,
    }
}

/// Convert a protobuf discriminant back to an [`ActivationMode`].
pub fn activation_mode_from_proto(v: i32) -> ActivationMode {
    match proto::ActivationMode::try_from(v) {
        Ok(proto::ActivationMode::Static) => ActivationMode::Static,
        Ok(proto::ActivationMode::Spreading) => ActivationMode::Spreading,
        Ok(proto::ActivationMode::Ppr) => ActivationMode::PersonalizedPageRank(Default::default()),
        _ => ActivationMode::None,
    }
}

/// Convert a protobuf discriminant back to a [`ContextFormat`].
pub fn context_format_from_proto(v: i32) -> ContextFormat {
    match proto::ContextFormat::try_from(v) {
        Ok(proto::ContextFormat::Narrative) => ContextFormat::Narrative,
        Ok(proto::ContextFormat::Json) => ContextFormat::Json,
        _ => ContextFormat::Structured,
    }
}

pub fn conflict_member_status_to_proto(status: ConflictMemberStatus) -> i32 {
    match status {
        ConflictMemberStatus::Active => proto::ConflictMemberStatus::Active as i32,
        ConflictMemberStatus::Superseded => proto::ConflictMemberStatus::Superseded as i32,
        ConflictMemberStatus::Retracted => proto::ConflictMemberStatus::Retracted as i32,
        ConflictMemberStatus::Quarantined => proto::ConflictMemberStatus::Quarantined as i32,
        ConflictMemberStatus::Merged => proto::ConflictMemberStatus::Merged as i32,
    }
}

pub fn conflict_arbitration_status_to_proto(status: ConflictArbitrationStatus) -> i32 {
    match status {
        ConflictArbitrationStatus::Unresolved => {
            proto::ConflictArbitrationStatus::Unresolved as i32
        }
        ConflictArbitrationStatus::Resolved => proto::ConflictArbitrationStatus::Resolved as i32,
        ConflictArbitrationStatus::Quarantined => {
            proto::ConflictArbitrationStatus::Quarantined as i32
        }
        ConflictArbitrationStatus::Superseded => {
            proto::ConflictArbitrationStatus::Superseded as i32
        }
    }
}

pub fn conflict_pair_to_proto(
    pair: &hirn_engine::ql::context::ConflictPair,
) -> proto::ConflictPair {
    proto::ConflictPair {
        memory_a: Some(memory_id_to_proto(&pair.memory_a)),
        memory_b: Some(memory_id_to_proto(&pair.memory_b)),
        content_a: pair.content_a.clone(),
        content_b: pair.content_b.clone(),
        confidence: pair.confidence,
        source_reliability_a: pair.source_reliability_a,
        source_reliability_b: pair.source_reliability_b,
    }
}

pub fn conflict_member_to_proto(member: &ConflictMember) -> proto::ConflictMember {
    proto::ConflictMember {
        memory_id: Some(memory_id_to_proto(&member.memory_id)),
        logical_memory_id: member.logical_memory_id.map(|id| id.to_string()),
        revision_id: member.revision_id.map(|id| id.to_string()),
        status: conflict_member_status_to_proto(member.status),
        layer: layer_to_proto(&member.layer),
        content: member.content.clone(),
        in_result_set: member.in_result_set,
        source_reliability: member.source_reliability,
    }
}

pub fn conflict_group_to_proto(group: &ConflictGroup) -> proto::ConflictGroup {
    proto::ConflictGroup {
        conflict_id: group.conflict_id.clone(),
        members: group.members.iter().map(conflict_member_to_proto).collect(),
        omitted_member_count: u32::try_from(group.omitted_member_count).unwrap_or(u32::MAX),
        pair_count: u32::try_from(group.pair_count).unwrap_or(u32::MAX),
        confidence: group.confidence,
        evidence_count: u32::try_from(group.evidence_count).unwrap_or(u32::MAX),
        source_reliability: group.source_reliability,
        arbitration_status: conflict_arbitration_status_to_proto(group.arbitration_status),
        authoritative_memory_id: group
            .authoritative_memory_id
            .as_ref()
            .map(memory_id_to_proto),
        preferred_memory_id: group.preferred_memory_id.as_ref().map(memory_id_to_proto),
    }
}

// ─── Metadata ────────────────────────────────────────────────

/// Convert a [`Metadata`] map to its protobuf representation.
pub fn metadata_to_proto(
    m: &BTreeMap<String, MetadataValue>,
) -> std::collections::HashMap<String, proto::MetadataValue> {
    m.iter()
        .map(|(k, v)| {
            let pv = match v {
                MetadataValue::Bool(b) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::BoolValue(*b)),
                },
                MetadataValue::Int(i) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::IntValue(*i)),
                },
                MetadataValue::Float(f) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::FloatValue(*f)),
                },
                MetadataValue::String(s) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::StringValue(s.clone())),
                },
                MetadataValue::Null => proto::MetadataValue { value: None },
                // F-79: List and Map values are serialized as JSON strings for proto transport.
                MetadataValue::List(l) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::StringValue(
                        serde_json::to_string(l).unwrap_or_default(),
                    )),
                },
                MetadataValue::Map(m) => proto::MetadataValue {
                    value: Some(proto::metadata_value::Value::StringValue(
                        serde_json::to_string(m).unwrap_or_default(),
                    )),
                },
            };
            (k.clone(), pv)
        })
        .collect()
}

/// Convert protobuf metadata back to a [`Metadata`] map.
pub fn metadata_from_proto(
    m: &std::collections::HashMap<String, proto::MetadataValue>,
) -> BTreeMap<String, MetadataValue> {
    m.iter()
        .map(|(k, v)| {
            let mv = match &v.value {
                Some(proto::metadata_value::Value::BoolValue(b)) => MetadataValue::Bool(*b),
                Some(proto::metadata_value::Value::IntValue(i)) => MetadataValue::Int(*i),
                Some(proto::metadata_value::Value::FloatValue(f)) => MetadataValue::Float(*f),
                Some(proto::metadata_value::Value::StringValue(s)) => {
                    MetadataValue::String(s.clone())
                }
                None => MetadataValue::Null,
            };
            (k.clone(), mv)
        })
        .collect()
}

// ─── Records ─────────────────────────────────────────────────

/// Convert an [`EpisodicRecord`] to its protobuf representation.
pub fn episodic_record_to_proto(r: &hirn::episodic::EpisodicRecord) -> proto::EpisodicRecord {
    proto::EpisodicRecord {
        id: Some(memory_id_to_proto(&r.id)),
        timestamp: Some(timestamp_to_proto(&r.timestamp)),
        event_type: event_type_to_proto(&r.event_type),
        content: r.content.clone(),
        summary: r.summary.clone(),
        entities: r
            .entities
            .iter()
            .map(|e| proto::EntityRef {
                name: e.name.clone(),
                role: e.role.clone(),
                entity_id: e.entity_id.as_ref().map(memory_id_to_proto),
            })
            .collect(),
        embedding: r.embedding.clone().unwrap_or_default(),
        importance: r.importance,
        surprise: r.surprise,
        access_count: r.access_count,
        last_accessed: Some(timestamp_to_proto(&r.last_accessed)),
        namespace: r.namespace.as_str().to_owned(),
        archived: r.archived,
        metadata: metadata_to_proto(&r.metadata),
    }
}

/// Convert a [`SemanticRecord`] to its protobuf representation.
pub fn semantic_record_to_proto(r: &hirn::semantic::SemanticRecord) -> proto::SemanticRecord {
    proto::SemanticRecord {
        id: Some(memory_id_to_proto(&r.id)),
        concept: r.concept.clone(),
        knowledge_type: knowledge_type_to_proto(&r.knowledge_type),
        description: r.description.clone(),
        embedding: r.embedding.clone().unwrap_or_default(),
        related_concepts: r
            .related_concepts
            .iter()
            .map(|c| proto::ConceptEdge {
                target_id: Some(memory_id_to_proto(&c.target_id)),
                relation: format!("{:?}", c.relation),
                weight: c.weight,
            })
            .collect(),
        confidence: r.confidence,
        source_episodes: r.source_episodes.iter().map(memory_id_to_proto).collect(),
        evidence_count: r.evidence_count,
        contradiction_ids: r.contradiction_ids.iter().map(memory_id_to_proto).collect(),
        created_at: Some(timestamp_to_proto(&r.created_at)),
        updated_at: Some(timestamp_to_proto(&r.updated_at)),
        access_count: r.access_count,
        version: r.version,
        namespace: r.namespace.as_str().to_owned(),
    }
}

/// Convert a [`WorkingMemoryEntry`] to its protobuf representation.
pub fn working_entry_to_proto(r: &hirn::working::WorkingMemoryEntry) -> proto::WorkingMemoryEntry {
    proto::WorkingMemoryEntry {
        id: Some(memory_id_to_proto(&r.id)),
        content: r.content.clone(),
        created_at: Some(timestamp_to_proto(&r.created_at)),
        expires_at: Some(timestamp_to_proto(&r.expires_at)),
        relevance_score: r.relevance_score,
        token_count: r.token_count,
        source: r.source.as_ref().map(|s| proto::MemoryRef {
            layer: layer_to_proto(&s.layer),
            id: Some(memory_id_to_proto(&s.id)),
        }),
        priority: priority_to_proto(&r.priority),
        agent_id: r.agent_id.as_str().to_owned(),
    }
}

/// Convert a [`ProceduralRecord`] to its protobuf representation.
pub fn procedural_record_to_proto(
    r: &hirn::procedural::ProceduralRecord,
) -> proto::ProceduralRecord {
    proto::ProceduralRecord {
        id: Some(memory_id_to_proto(&r.id)),
        name: r.name.clone(),
        description: r.description.clone(),
        steps: r
            .steps
            .iter()
            .map(|s| proto::ActionStep {
                description: s.description.clone(),
                tool: s.tool.clone().unwrap_or_default(),
                parameters: s
                    .parameters
                    .iter()
                    .map(|(k, v)| (k.clone(), format!("{v:?}")))
                    .collect(),
            })
            .collect(),
        preconditions: r.preconditions.clone(),
        embedding: r.embedding.clone().unwrap_or_default(),
        success_count: r.success_count,
        invocation_count: r.invocation_count,
        success_rate: f64::from(r.success_rate),
        source_episodes: r.source_episodes.iter().map(memory_id_to_proto).collect(),
        created_at: Some(timestamp_to_proto(&r.created_at)),
        updated_at: Some(timestamp_to_proto(&r.updated_at)),
        last_accessed: Some(timestamp_to_proto(&r.last_accessed)),
        access_count: r.access_count,
        namespace: r.namespace.as_str().to_owned(),
        archived: r.archived,
        metadata: metadata_to_proto(&r.metadata),
    }
}

/// Convert a [`MemoryRecord`] to its protobuf representation.
pub fn memory_record_to_proto(r: &MemoryRecord) -> proto::MemoryRecord {
    match r {
        MemoryRecord::Working(w) => proto::MemoryRecord {
            record: Some(proto::memory_record::Record::Working(
                working_entry_to_proto(w),
            )),
        },
        MemoryRecord::Episodic(e) => proto::MemoryRecord {
            record: Some(proto::memory_record::Record::Episodic(
                episodic_record_to_proto(e),
            )),
        },
        MemoryRecord::Semantic(s) => proto::MemoryRecord {
            record: Some(proto::memory_record::Record::Semantic(
                semantic_record_to_proto(s),
            )),
        },
        MemoryRecord::Procedural(p) => proto::MemoryRecord {
            record: Some(proto::memory_record::Record::Procedural(
                procedural_record_to_proto(p),
            )),
        },
    }
}

pub fn recall_result_to_proto(result: &hirn_engine::RecallResult) -> proto::RecallResult {
    proto::RecallResult {
        record: Some(memory_record_to_proto(&result.record)),
        similarity: result.similarity,
        composite_score: result.composite_score,
        revision: result.revision.as_ref().map(revision_ref_to_proto),
        score_breakdown: Some(score_breakdown_to_proto(&result.score_breakdown)),
    }
}

// ─── Score ───────────────────────────────────────────────────

/// Convert a [`ScoreBreakdown`] to its protobuf representation.
pub fn score_breakdown_to_proto(sb: &ScoreBreakdown) -> proto::ScoreBreakdown {
    proto::ScoreBreakdown {
        similarity: sb.similarity,
        importance: sb.importance,
        recency: sb.recency,
        activation: sb.activation,
        causal_relevance: sb.causal_relevance,
        surprise: sb.surprise,
        source_reliability: sb.source_reliability,
    }
}

pub fn scored_memory_to_proto(memory: &ScoredMemory) -> proto::ScoredMemory {
    proto::ScoredMemory {
        record: Some(memory_record_to_proto(&memory.record)),
        score: memory.score,
        score_breakdown: Some(score_breakdown_to_proto(&memory.score_breakdown)),
        revision: memory.revision.as_ref().map(revision_ref_to_proto),
    }
}

pub fn query_result_to_json(result: &QueryResult) -> serde_json::Value {
    if let Some(json) = hirn_engine::ql::revision_query_result_to_json(result) {
        return json;
    }

    match result {
        QueryResult::Records(r) => serde_json::json!({
            "type": "records",
            "records_returned": r.records_returned,
            "records_scanned": r.records_scanned,
            "query_time_ms": r.query_time_ms,
            "context": r.context,
            "conflicts": serde_json::to_value(&r.conflicts).unwrap_or(serde_json::Value::Null),
            "conflict_groups": serde_json::to_value(&r.conflict_groups).unwrap_or(serde_json::Value::Null),
        }),
        QueryResult::Created(c) => serde_json::json!({
            "type": "created",
            "id": c.id.to_string(),
            "layer": format!("{:?}", c.layer),
        }),
        QueryResult::Forgotten(f) => serde_json::json!({
            "type": "forgotten",
            "target": f.target,
        }),
        QueryResult::Inspected(i) => hirn_engine::inspected_result_to_json(i),
        QueryResult::Traced(t) => hirn_engine::traced_result_to_json(t),
        QueryResult::Consolidated(c) => serde_json::json!({
            "type": "consolidated",
            "records_processed": c.records_processed,
        }),
        QueryResult::WatchAck(w) => serde_json::json!({
            "type": "watch_ack",
            "message": w.message,
        }),
        QueryResult::Aggregated(a) => serde_json::json!({
            "type": "aggregated",
            "group_field": a.group_field,
            "function": format!("{}", a.function),
            "groups": a.groups.iter().map(|g| serde_json::json!({
                "key": g.key,
                "value": g.value,
            })).collect::<Vec<_>>(),
            "query_time_ms": a.query_time_ms,
            "formatted": a.formatted,
        }),
        QueryResult::ExplainPlan(e) => {
            let mut explain = serde_json::json!({
                "type": "explain",
                "plan_text": e.plan_text,
                "has_actual_results": e.actual_result.is_some(),
            });
            if let Some(actual_result) = e.actual_result.as_deref() {
                explain["actual_result"] = query_result_to_json(actual_result);
            }
            if let Some(ref diag) = e.diagnostics {
                explain["diagnostics"] = query_diagnostics_to_json(diag);
            }
            explain
        }
        QueryResult::Policy(p) => serde_json::json!({
            "type": "policy",
            "message": p.message,
            "policies": p.policies.iter().map(|(name, text)| serde_json::json!({
                "name": name,
                "text": text,
            })).collect::<Vec<_>>(),
        }),
        QueryResult::SvoEvents(e) => serde_json::json!({
            "type": "svo_events",
            "events_returned": e.events_returned,
            "events": e.events.iter().map(|ev| serde_json::json!({
                "source_memory_id": ev.source_memory_id,
                "subject": ev.subject,
                "verb": ev.verb,
                "object": ev.object,
                "time_start": ev.time_start,
                "time_end": ev.time_end,
                "confidence": ev.confidence,
            })).collect::<Vec<_>>(),
        }),
        QueryResult::Causal(c) => serde_json::json!({
            "type": "causal",
            "kind": c.kind.to_string(),
            "query_time_ms": c.query_time_ms,
            "rows": c.rows.iter().map(|row| {
                row.columns
                    .iter()
                    .map(|(key, value)| (key.clone(), serde_json::Value::String(value.clone())))
                    .collect::<serde_json::Map<String, serde_json::Value>>()
            }).collect::<Vec<_>>(),
        }),
        QueryResult::Corrected(_)
        | QueryResult::Superseded(_)
        | QueryResult::Merged(_)
        | QueryResult::Retracted(_)
        | QueryResult::History(_) => unreachable!("handled by revision_query_result_to_json"),
    }
}

fn query_diagnostics_to_json(diag: &hirn_engine::QueryDiagnostics) -> serde_json::Value {
    serde_json::json!({
        "query_id": diag.query_id.as_ref().map(|id| id.to_string()),
        "authorize_us": diag.authorize_us,
        "embed_ms": diag.embed_ms,
        "optimize_ms": diag.optimize_ms,
        "physical_plan_ms": diag.physical_plan_ms,
        "execute_plan_ms": diag.execute_plan_ms,
        "vector_search_ms": diag.vector_search_ms,
        "graph_expand_ms": diag.graph_expand_ms,
        "rerank_ms": diag.rerank_ms,
        "neural_rerank_ms": diag.neural_rerank_ms,
        "assemble_ms": diag.assemble_ms,
        "total_ms": diag.total_ms,
        "records_scanned": diag.records_scanned,
        "records_returned": diag.records_returned,
        "threshold_filtered_count": diag.threshold_filtered_count,
        "competitive_inhibition_count": diag.competitive_inhibition_count,
        "truncated_by_limit_count": diag.truncated_by_limit_count,
        "raw_text_redacted_results": diag.raw_text_redacted_results,
        "multivector_fallback_count": diag.multivector_fallback_count,
        "neural_rerank_fallback_count": diag.neural_rerank_fallback_count,
    })
}

pub fn query_diagnostics_to_proto(diag: &hirn_engine::QueryDiagnostics) -> proto::QueryDiagnostics {
    proto::QueryDiagnostics {
        query_id: diag.query_id.as_ref().map(ToString::to_string),
        authorize_us: diag.authorize_us,
        embed_ms: diag.embed_ms,
        optimize_ms: diag.optimize_ms,
        physical_plan_ms: diag.physical_plan_ms,
        execute_plan_ms: diag.execute_plan_ms,
        vector_search_ms: diag.vector_search_ms,
        graph_expand_ms: diag.graph_expand_ms,
        rerank_ms: diag.rerank_ms,
        neural_rerank_ms: diag.neural_rerank_ms,
        assemble_ms: diag.assemble_ms,
        total_ms: diag.total_ms,
        records_scanned: diag
            .records_scanned
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        records_returned: diag
            .records_returned
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        threshold_filtered_count: diag
            .threshold_filtered_count
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        competitive_inhibition_count: diag
            .competitive_inhibition_count
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        truncated_by_limit_count: diag
            .truncated_by_limit_count
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        raw_text_redacted_results: diag
            .raw_text_redacted_results
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        multivector_fallback_count: diag
            .multivector_fallback_count
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
        neural_rerank_fallback_count: diag
            .neural_rerank_fallback_count
            .map(|value| u32::try_from(value).unwrap_or(u32::MAX)),
    }
}

pub fn explain_result_to_proto(
    result: &ExplainResult,
) -> Result<proto::ExplainPlanResult, &'static str> {
    Ok(proto::ExplainPlanResult {
        plan_text: result.plan_text.clone(),
        actual_result: result
            .actual_result
            .as_deref()
            .map(query_result_to_execute_response)
            .transpose()?
            .map(Box::new),
        diagnostics: result.diagnostics.as_ref().map(query_diagnostics_to_proto),
    })
}

pub fn query_result_to_execute_response(
    result: &QueryResult,
) -> Result<proto::ExecuteResponse, &'static str> {
    let response = match result {
        QueryResult::Records(r) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Records(
                proto::RecordResults {
                    records: r.records.iter().map(scored_memory_to_proto).collect(),
                    query_time_ms: r.query_time_ms,
                    records_scanned: u32::try_from(r.records_scanned).unwrap_or(u32::MAX),
                    records_returned: u32::try_from(r.records_returned).unwrap_or(u32::MAX),
                    context: r.context.clone(),
                    conflicts: r
                        .conflicts
                        .as_deref()
                        .unwrap_or_default()
                        .iter()
                        .map(conflict_pair_to_proto)
                        .collect(),
                    conflict_groups: r
                        .conflict_groups
                        .as_deref()
                        .unwrap_or_default()
                        .iter()
                        .map(conflict_group_to_proto)
                        .collect(),
                },
            )),
        },
        QueryResult::Created(c) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Created(
                proto::CreatedResult {
                    id: Some(memory_id_to_proto(&c.id)),
                    layer: layer_to_proto(&c.layer),
                },
            )),
        },
        QueryResult::Forgotten(f) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Forgotten(
                proto::ForgottenResult {
                    target: f.target.clone(),
                    mode: format!("{:?}", f.mode),
                },
            )),
        },
        QueryResult::Corrected(c) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Corrected(
                proto::CorrectedResult {
                    logical_memory_id: c.logical_memory_id.to_string(),
                    prior_revision_id: c.prior_revision_id.to_string(),
                    new_revision_id: c.new_revision_id.to_string(),
                },
            )),
        },
        QueryResult::Superseded(s) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Superseded(
                proto::SupersededResult {
                    logical_memory_id: s.logical_memory_id.to_string(),
                    prior_revision_id: s.prior_revision_id.to_string(),
                    new_revision_id: s.new_revision_id.to_string(),
                    reason: s.reason.clone(),
                },
            )),
        },
        QueryResult::Merged(m) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Merged(
                proto::MergedResult {
                    target_logical_memory_id: m.target_logical_memory_id.to_string(),
                    prior_target_revision_id: m.prior_target_revision_id.to_string(),
                    new_target_revision_id: m.new_target_revision_id.to_string(),
                    source_logical_memory_ids: m
                        .source_logical_memory_ids
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                    source_revision_ids: m
                        .source_revision_ids
                        .iter()
                        .map(ToString::to_string)
                        .collect(),
                    reason: m.reason.clone(),
                },
            )),
        },
        QueryResult::Retracted(r) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Retracted(
                proto::RetractedResult {
                    logical_memory_id: r.logical_memory_id.to_string(),
                    prior_revision_id: r.prior_revision_id.to_string(),
                    tombstone_revision_id: r.tombstone_revision_id.to_string(),
                    reason: r.reason.clone(),
                },
            )),
        },
        QueryResult::Inspected(i) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Inspected(
                proto::InspectedResult {
                    record: Some(memory_record_to_proto(&i.record)),
                    importance: i.importance,
                    access_count: i.access_count,
                    last_accessed: Some(timestamp_to_proto(&i.last_accessed)),
                    neighbors: i
                        .neighbors
                        .iter()
                        .map(|n| proto::NeighborInfo {
                            edge: Some(proto::GraphEdge {
                                id: Some(memory_id_to_proto(&n.edge.id)),
                                source: Some(memory_id_to_proto(&n.edge.source)),
                                target: Some(memory_id_to_proto(&n.edge.target)),
                                relation: 0,
                                weight: n.edge.weight,
                                co_retrieval_count: n.edge.co_retrieval_count,
                                created_at: Some(timestamp_to_proto(&n.edge.created_at)),
                                updated_at: Some(timestamp_to_proto(&n.edge.updated_at)),
                                valid_from: n.edge.valid_from.map(|ts| ts.to_string()),
                                valid_until: n.edge.valid_until.map(|ts| ts.to_string()),
                                metadata: metadata_to_proto(&n.edge.metadata),
                            }),
                            neighbor_id: Some(memory_id_to_proto(&n.neighbor_id)),
                        })
                        .collect(),
                    trust_score: i.trust_score,
                    semantic_revision: i
                        .semantic_revision
                        .as_ref()
                        .map(semantic_revision_summary_to_proto),
                    conflict_groups: i
                        .conflict_groups
                        .iter()
                        .map(conflict_group_to_proto)
                        .collect(),
                },
            )),
        },
        QueryResult::History(h) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::History(
                proto::HistoryResult {
                    semantic_revision: Some(semantic_revision_summary_to_proto(
                        &h.semantic_revision,
                    )),
                    items: h
                        .items
                        .iter()
                        .map(|item| proto::SemanticHistoryItem {
                            record: Some(semantic_record_to_proto(&item.record)),
                            revision: Some(semantic_revision_entry_to_proto(&item.revision)),
                        })
                        .collect(),
                },
            )),
        },
        QueryResult::Traced(t) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Traced(
                proto::TracedResult {
                    record: Some(memory_record_to_proto(&t.record)),
                    source_episodes: t.source_episodes.iter().map(memory_id_to_proto).collect(),
                    derived_records: t.derived_records.iter().map(memory_id_to_proto).collect(),
                    mutation_count: u32::try_from(t.mutation_count).unwrap_or(u32::MAX),
                    trust_score: t.trust_score,
                    lineage_tree: t.lineage_tree.clone(),
                    semantic_revision: t
                        .semantic_revision
                        .as_ref()
                        .map(semantic_revision_summary_to_proto),
                    conflict_groups: t
                        .conflict_groups
                        .iter()
                        .map(conflict_group_to_proto)
                        .collect(),
                },
            )),
        },
        QueryResult::Consolidated(c) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Consolidated(
                proto::ConsolidatedResult {
                    records_processed: u32::try_from(c.records_processed).unwrap_or(u32::MAX),
                },
            )),
        },
        QueryResult::WatchAck(w) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::WatchAck(
                proto::WatchAckResult {
                    message: w.message.clone(),
                },
            )),
        },
        QueryResult::ExplainPlan(e) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::ExplainPlan(Box::new(
                explain_result_to_proto(e)?,
            ))),
        },
        QueryResult::Aggregated(a) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Aggregated(
                proto::AggregatedResult {
                    group_field: a.group_field.clone(),
                    function: a.function.to_string(),
                    groups: a
                        .groups
                        .iter()
                        .map(|group| proto::AggregatedGroup {
                            key: group.key.clone(),
                            value: group.value,
                        })
                        .collect(),
                    query_time_ms: a.query_time_ms,
                    formatted: a.formatted.clone(),
                },
            )),
        },
        QueryResult::Policy(p) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Policy(
                proto::PolicyResult {
                    message: p.message.clone(),
                    policies: p
                        .policies
                        .iter()
                        .map(|(name, text)| proto::PolicyEntry {
                            name: name.clone(),
                            text: text.clone(),
                        })
                        .collect(),
                },
            )),
        },
        QueryResult::SvoEvents(e) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::SvoEvents(
                proto::SvoEventQueryResult {
                    events: e
                        .events
                        .iter()
                        .map(|event| proto::SvoEventEntry {
                            source_memory_id: event.source_memory_id.clone(),
                            subject: event.subject.clone(),
                            verb: event.verb.clone(),
                            object: event.object.clone(),
                            time_start: event.time_start.clone(),
                            time_end: event.time_end.clone(),
                            confidence: event.confidence,
                        })
                        .collect(),
                    events_returned: u32::try_from(e.events_returned).unwrap_or(u32::MAX),
                },
            )),
        },
        QueryResult::Causal(c) => proto::ExecuteResponse {
            result: Some(proto::execute_response::Result::Causal(
                proto::CausalResult {
                    kind: c.kind.to_string(),
                    query_time_ms: c.query_time_ms,
                    rows: c
                        .rows
                        .iter()
                        .map(|row| proto::CausalRow {
                            columns: row
                                .columns
                                .iter()
                                .map(|(key, value)| proto::CausalColumn {
                                    key: key.clone(),
                                    value: value.clone(),
                                })
                                .collect(),
                        })
                        .collect(),
                },
            )),
        },
    };

    Ok(response)
}
