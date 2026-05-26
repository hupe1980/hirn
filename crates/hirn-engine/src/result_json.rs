use hirn_core::id::MemoryId;
use hirn_core::record::MemoryRecord;

use crate::inspect::InspectResult;
use crate::ql::context::ConflictGroup;
use crate::ql::results::SemanticRevisionSummary;
use crate::retrieval::recall::ResourceEvidenceSummary;
use crate::trace::TraceResult;

#[must_use]
pub fn inspected_result_to_json(result: &InspectResult) -> serde_json::Value {
    serde_json::json!({
        "type": "inspected",
        "id": result.record.id().to_string(),
        "layer": format!("{:?}", result.record.layer()),
        "importance": result.importance,
        "access_count": result.access_count,
        "trust_score": result.trust_score,
        "neighbor_count": result.neighbors.len(),
        "conflict_groups": serde_json::to_value(&result.conflict_groups).unwrap_or(serde_json::Value::Null),
        "semantic_revision": serde_json::to_value(&result.semantic_revision).unwrap_or(serde_json::Value::Null),
        "resource_evidence": resource_evidence_to_json(&result.resource_evidence),
        "resource_hydration_available": resource_hydration_to_json(&result.resource_evidence),
    })
}

#[must_use]
pub fn traced_result_to_json(result: &TraceResult) -> serde_json::Value {
    trace_result_to_json(result)
}

#[must_use]
pub fn trace_result_to_json(result: &TraceResult) -> serde_json::Value {
    trace_json(
        &result.record,
        result.trust_score,
        result.mutation_count,
        &result.lineage_tree,
        &result.source_episodes,
        &result.derived_records,
        &result.conflict_groups,
        &result.semantic_revision,
        &result.resource_evidence,
    )
}

fn trace_json(
    record: &MemoryRecord,
    trust_score: f32,
    mutation_count: usize,
    lineage_tree: &str,
    source_episodes: &[MemoryId],
    derived_records: &[MemoryId],
    conflict_groups: &[ConflictGroup],
    semantic_revision: &Option<SemanticRevisionSummary>,
    resource_evidence: &[ResourceEvidenceSummary],
) -> serde_json::Value {
    serde_json::json!({
        "type": "traced",
        "id": record.id().to_string(),
        "layer": format!("{:?}", record.layer()),
        "source_episodes": source_episodes.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
        "derived_records": derived_records.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
        "mutation_count": mutation_count,
        "trust_score": trust_score,
        "lineage_tree": lineage_tree,
        "conflict_groups": serde_json::to_value(conflict_groups).unwrap_or(serde_json::Value::Null),
        "semantic_revision": serde_json::to_value(semantic_revision).unwrap_or(serde_json::Value::Null),
        "resource_evidence": resource_evidence_to_json(resource_evidence),
        "resource_hydration_available": resource_hydration_to_json(resource_evidence),
    })
}

pub(crate) fn resource_evidence_to_json(
    resource_evidence: &[ResourceEvidenceSummary],
) -> serde_json::Value {
    serde_json::Value::Array(
        resource_evidence
            .iter()
            .map(|summary| {
                serde_json::json!({
                    "resource_id": summary.resource_id.to_string(),
                    "role": summary.role.as_str(),
                    "provenance": summary.provenance.as_str(),
                    "artifact_id": summary.artifact_id.map(|artifact_id| artifact_id.to_string()),
                    "artifact_kind": summary.artifact_kind.map(|kind| kind.as_str()),
                    "resource_state": summary.lifecycle_state.as_str(),
                    "modality": summary.modality.map(|modality| modality.as_str()),
                    "mime_type": summary.mime_type,
                    "display_name": summary.display_name,
                    "available_artifacts": summary.available_artifacts.iter().map(|kind| kind.as_str()).collect::<Vec<_>>(),
                    "has_preview": summary.has_preview,
                    "can_hydrate_preview": summary.can_hydrate_preview,
                    "can_hydrate_full": summary.can_hydrate_full,
                })
            })
            .collect(),
    )
}

pub(crate) fn resource_hydration_to_json(
    resource_evidence: &[ResourceEvidenceSummary],
) -> serde_json::Value {
    let preview = resource_evidence
        .iter()
        .filter(|summary| summary.has_preview && summary.can_hydrate_preview)
        .map(resource_hydration_summary_to_json)
        .collect::<Vec<_>>();
    let full = resource_evidence
        .iter()
        .filter(|summary| summary.can_hydrate_full)
        .map(resource_hydration_summary_to_json)
        .collect::<Vec<_>>();

    serde_json::json!({
        "preview": preview,
        "full": full,
    })
}

fn resource_hydration_summary_to_json(summary: &ResourceEvidenceSummary) -> serde_json::Value {
    serde_json::json!({
        "resource_id": summary.resource_id.to_string(),
        "role": summary.role.as_str(),
        "provenance": summary.provenance.as_str(),
        "artifact_id": summary.artifact_id.map(|artifact_id| artifact_id.to_string()),
        "artifact_kind": summary.artifact_kind.map(|kind| kind.as_str()),
        "resource_state": summary.lifecycle_state.as_str(),
        "modality": summary.modality.map(|modality| modality.as_str()),
        "mime_type": summary.mime_type,
        "display_name": summary.display_name,
        "available_artifacts": summary.available_artifacts.iter().map(|kind| kind.as_str()).collect::<Vec<_>>(),
    })
}
