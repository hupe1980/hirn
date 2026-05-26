use std::fmt;

use serde::{Deserialize, Serialize};

use hirn_core::id::MemoryId;
use hirn_core::record::MemoryRecord;
use hirn_core::revision::{LogicalMemoryId, RevisionId, RevisionOperation, RevisionState};
use hirn_core::semantic::SemanticRecord;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::Layer;
use hirn_core::{HirnError, HirnResult};

use crate::db::HirnDB;
use crate::inspect::InspectResult;
use crate::resource_presentation::{ResourcePreviewPackage, ResourceScoreAttribution};
pub use crate::scoring::ScoreBreakdown;
use crate::trace::TraceResult;

use super::ast::{AggFunction, ForgetMode};

#[derive(Debug, Clone)]
pub enum QueryResult {
    Records(RecordResults),
    Aggregated(AggregatedResults),
    Created(CreatedResult),
    Forgotten(ForgottenResult),
    Corrected(CorrectedResult),
    Superseded(SupersededResult),
    Merged(MergedResult),
    Retracted(RetractedResult),
    Inspected(InspectResult),
    History(HistoryResult),
    Traced(TraceResult),
    Consolidated(ConsolidatedResult),
    WatchAck(WatchAckResult),
    ExplainPlan(ExplainResult),
    Policy(PolicyResult),
    SvoEvents(SvoEventResults),
    Causal(CausalQueryResult),
}

#[derive(Debug, Clone)]
pub struct ExplainResult {
    pub plan_text: String,
    pub actual_result: Option<Box<QueryResult>>,
    pub diagnostics: Option<crate::diagnostics::QueryDiagnostics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub message: String,
    pub policies: Vec<(String, String)>,
}

#[derive(Debug, Clone)]
pub struct SvoEventResults {
    pub events: Vec<SvoEventResult>,
    pub events_returned: usize,
}

#[derive(Debug, Clone)]
pub struct SvoEventResult {
    pub source_memory_id: String,
    pub subject: String,
    pub verb: String,
    pub object: String,
    pub time_start: Option<String>,
    pub time_end: Option<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalQueryResult {
    pub kind: CausalQueryKind,
    pub rows: Vec<CausalRow>,
    pub query_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalQueryKind {
    ExplainCauses,
    WhatIf,
    Counterfactual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalRow {
    pub columns: Vec<(String, String)>,
}

impl fmt::Display for CausalQueryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExplainCauses => write!(f, "EXPLAIN CAUSES"),
            Self::WhatIf => write!(f, "WHAT_IF"),
            Self::Counterfactual => write!(f, "COUNTERFACTUAL"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RecordResults {
    pub records: Vec<ScoredMemory>,
    pub query_time_ms: f64,
    pub records_scanned: usize,
    pub records_returned: usize,
    pub context: Option<String>,
    pub conflicts: Option<Vec<super::context::ConflictPair>>,
    pub conflict_groups: Option<Vec<super::context::ConflictGroup>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMemory {
    pub record: MemoryRecord,
    pub revision: Option<hirn_core::revision::RevisionRef>,
    pub score: f32,
    pub score_breakdown: ScoreBreakdown,
    pub resource_evidence: Vec<crate::retrieval::recall::ResourceEvidenceSummary>,
    pub(crate) resource_preview_packages: Vec<ResourcePreviewPackage>,
    pub resource_score_attribution: Vec<ResourceScoreAttribution>,
}

#[derive(Debug, Clone)]
pub struct CreatedResult {
    pub id: MemoryId,
    pub layer: Layer,
}

#[derive(Debug, Clone)]
pub struct ForgottenResult {
    pub target: String,
    pub mode: ForgetMode,
}

#[derive(Debug, Clone)]
pub struct CorrectedResult {
    pub logical_memory_id: LogicalMemoryId,
    pub prior_revision_id: RevisionId,
    pub new_revision_id: RevisionId,
}

#[derive(Debug, Clone)]
pub struct SupersededResult {
    pub logical_memory_id: LogicalMemoryId,
    pub prior_revision_id: RevisionId,
    pub new_revision_id: RevisionId,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MergedResult {
    pub target_logical_memory_id: LogicalMemoryId,
    pub prior_target_revision_id: RevisionId,
    pub new_target_revision_id: RevisionId,
    pub source_logical_memory_ids: Vec<LogicalMemoryId>,
    pub source_revision_ids: Vec<RevisionId>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RetractedResult {
    pub logical_memory_id: LogicalMemoryId,
    pub prior_revision_id: RevisionId,
    pub tombstone_revision_id: RevisionId,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticRevisionEntry {
    pub memory_id: MemoryId,
    pub revision_id: RevisionId,
    pub version: u32,
    pub operation: RevisionOperation,
    pub state: RevisionState,
    pub reason: Option<String>,
    pub created_at: Timestamp,
    pub superseded_by: Option<MemoryId>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticRevisionSummary {
    pub logical_memory_id: LogicalMemoryId,
    pub current_revision_id: RevisionId,
    pub head_revision_id: RevisionId,
    pub current_state: RevisionState,
    pub logical_state: RevisionState,
    pub revision_count: usize,
    pub revisions: Vec<SemanticRevisionEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticHistoryItem {
    pub record: SemanticRecord,
    pub revision: SemanticRevisionEntry,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoryResult {
    pub semantic_revision: SemanticRevisionSummary,
    pub items: Vec<SemanticHistoryItem>,
}

#[must_use]
pub fn revision_query_result_to_json(result: &QueryResult) -> Option<serde_json::Value> {
    match result {
        QueryResult::Corrected(c) => Some(serde_json::json!({
            "type": "corrected",
            "logical_memory_id": c.logical_memory_id.to_string(),
            "prior_revision_id": c.prior_revision_id.to_string(),
            "new_revision_id": c.new_revision_id.to_string(),
        })),
        QueryResult::Superseded(s) => Some(serde_json::json!({
            "type": "superseded",
            "logical_memory_id": s.logical_memory_id.to_string(),
            "prior_revision_id": s.prior_revision_id.to_string(),
            "new_revision_id": s.new_revision_id.to_string(),
            "reason": s.reason,
        })),
        QueryResult::Merged(m) => Some(serde_json::json!({
            "type": "merged",
            "target_logical_memory_id": m.target_logical_memory_id.to_string(),
            "prior_target_revision_id": m.prior_target_revision_id.to_string(),
            "new_target_revision_id": m.new_target_revision_id.to_string(),
            "source_logical_memory_ids": m.source_logical_memory_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "source_revision_ids": m.source_revision_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "reason": m.reason,
        })),
        QueryResult::Retracted(r) => Some(serde_json::json!({
            "type": "retracted",
            "logical_memory_id": r.logical_memory_id.to_string(),
            "prior_revision_id": r.prior_revision_id.to_string(),
            "tombstone_revision_id": r.tombstone_revision_id.to_string(),
            "reason": r.reason,
        })),
        QueryResult::History(h) => Some(serde_json::json!({
            "type": "history",
            "semantic_revision": serde_json::to_value(&h.semantic_revision).unwrap_or(serde_json::Value::Null),
            "items": serde_json::to_value(&h.items).unwrap_or(serde_json::Value::Null),
        })),
        _ => None,
    }
}

pub(crate) async fn load_semantic_revision_summary(
    db: &HirnDB,
    record: &SemanticRecord,
) -> HirnResult<SemanticRevisionSummary> {
    let history = db.semantic().history(record.id).await?;
    summarize_semantic_revision_chain(record, &history)
}

pub(crate) fn summarize_semantic_revision_chain(
    current: &SemanticRecord,
    history: &[SemanticRecord],
) -> HirnResult<SemanticRevisionSummary> {
    let head = history.last().ok_or_else(|| {
        HirnError::NotFound(format!(
            "semantic revision history missing for {}",
            current.logical_memory_id
        ))
    })?;

    if head.logical_memory_id != current.logical_memory_id {
        return Err(HirnError::InvalidInput(format!(
            "semantic revision history mismatch for {}",
            current.logical_memory_id
        )));
    }

    Ok(SemanticRevisionSummary {
        logical_memory_id: current.logical_memory_id,
        current_revision_id: current.revision_id,
        head_revision_id: head.revision_id,
        current_state: current.revision_state_against(head),
        logical_state: head.logical_state(),
        revision_count: history.len(),
        revisions: history
            .iter()
            .enumerate()
            .map(|(index, revision)| SemanticRevisionEntry {
                memory_id: revision.id,
                revision_id: revision.revision_id,
                version: revision.version,
                operation: revision.revision_operation,
                state: revision.revision_state_against(head),
                reason: revision.revision_reason.clone(),
                created_at: revision.created_at,
                superseded_by: semantic_revision_superseded_by(history, index),
            })
            .collect(),
    })
}

fn semantic_revision_superseded_by(history: &[SemanticRecord], index: usize) -> Option<MemoryId> {
    let revision = &history[index];
    revision
        .superseded_by
        .or_else(|| history.get(index + 1).map(|next| next.id))
}

pub(crate) fn build_semantic_history_items(
    history: &[SemanticRecord],
    summary: &SemanticRevisionSummary,
) -> HirnResult<Vec<SemanticHistoryItem>> {
    if history.len() != summary.revisions.len() {
        return Err(HirnError::InvalidInput(format!(
            "semantic history item mismatch for {}",
            summary.logical_memory_id
        )));
    }

    history
        .iter()
        .zip(summary.revisions.iter())
        .map(|(record, revision)| {
            if (record.id, record.revision_id) != (revision.memory_id, revision.revision_id) {
                return Err(HirnError::InvalidInput(format!(
                    "semantic history revision mismatch for {}",
                    summary.logical_memory_id
                )));
            }

            Ok(SemanticHistoryItem {
                record: record.clone(),
                revision: revision.clone(),
            })
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct ConsolidatedResult {
    pub records_processed: usize,
}

#[derive(Debug, Clone)]
pub struct WatchAckResult {
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct AggregatedResults {
    pub group_field: String,
    pub function: AggFunction,
    pub groups: Vec<AggregatedGroup>,
    pub query_time_ms: f64,
    pub formatted: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AggregatedGroup {
    pub key: String,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct ProjectedRecord {
    pub fields: std::collections::BTreeMap<String, serde_json::Value>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn revision_query_result_json_serializes_merge_shape() {
        let result = QueryResult::Merged(MergedResult {
            target_logical_memory_id: LogicalMemoryId::new(),
            prior_target_revision_id: RevisionId::new(),
            new_target_revision_id: RevisionId::new(),
            source_logical_memory_ids: vec![LogicalMemoryId::new(), LogicalMemoryId::new()],
            source_revision_ids: vec![RevisionId::new(), RevisionId::new()],
            reason: Some("dedupe".to_owned()),
        });

        let json = revision_query_result_to_json(&result).expect("expected revision JSON");

        assert_eq!(json["type"], "merged");
        assert_eq!(json["reason"], "dedupe");
        assert_eq!(
            json["source_logical_memory_ids"].as_array().unwrap().len(),
            2
        );
        assert_eq!(json["source_revision_ids"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn revision_query_result_json_skips_non_revision_results() {
        let result = QueryResult::WatchAck(WatchAckResult {
            message: "ok".to_owned(),
        });

        assert!(revision_query_result_to_json(&result).is_none());
    }
}
