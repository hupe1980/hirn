use hirn_core::MemoryId;
use hirn_core::revision::RevisionRef;
use hirn_core::types::{Layer, Namespace};

use crate::db::HirnDB;
use crate::diagnostics::QueryDiagnostics;
use crate::ql::context::ThinkResult;
use crate::recall::RecallResult;
use crate::scoring::{ScoreBreakdown, ScoringWeights};

#[derive(Debug, Clone)]
pub struct RetrievalSuppressionSummary {
    pub candidate_count: usize,
    pub threshold_filtered_count: usize,
    pub competitive_inhibition_count: usize,
    pub truncated_by_limit_count: usize,
}

#[derive(Debug, Clone)]
pub struct RetrievedRecordExplanation {
    pub memory_id: MemoryId,
    pub layer: Layer,
    pub revision: Option<RevisionRef>,
    pub composite_score: Option<f32>,
    pub score_breakdown: Option<ScoreBreakdown>,
    pub raw_text_redacted: bool,
    pub ranking_details_redacted: bool,
    pub resource_evidence_count: usize,
}

#[derive(Debug, Clone)]
pub struct RetrievalExplanation {
    pub diagnostics: QueryDiagnostics,
    pub scoring_weights: ScoringWeights,
    pub policy: RetrievalPolicySummary,
    pub suppression: RetrievalSuppressionSummary,
    pub raw_text_redacted_results: usize,
    pub results: Vec<RetrievedRecordExplanation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalPolicyScope {
    Unrestricted,
    RequestedNamespace,
    AllowedNamespaces,
    RequestedNamespaceDenied,
}

#[derive(Debug, Clone)]
pub struct RetrievalPolicySummary {
    pub scope: RetrievalPolicyScope,
    pub requested_namespace: Option<Namespace>,
    pub allowed_namespaces: Option<Vec<Namespace>>,
    pub namespace_restriction_applied: bool,
    pub raw_text_redaction_applied: bool,
}

#[derive(Debug, Clone)]
pub struct ThinkExplanation {
    pub retrieval: RetrievalExplanation,
    pub token_budget: usize,
    pub token_count: usize,
    pub records_included_count: usize,
    pub records_excluded_count: usize,
    pub conflict_group_count: usize,
    pub query_time_ms: f64,
}

pub(crate) fn build_retrieval_explanation(
    db: &HirnDB,
    actor_id: &str,
    results: &[RecallResult],
    diagnostics: QueryDiagnostics,
    scoring_weights: ScoringWeights,
    requested_namespace: Option<Namespace>,
    allowed_namespaces: Option<Vec<Namespace>>,
) -> RetrievalExplanation {
    let result_explanations: Vec<_> = results
        .iter()
        .map(|result| {
            let raw_text_redacted = !db.can_read_raw_content(actor_id, &result.record);
            let ranking_details_redacted = raw_text_redacted;

            RetrievedRecordExplanation {
                memory_id: result.record.id(),
                layer: result.record.layer(),
                revision: result.revision,
                composite_score: (!ranking_details_redacted).then_some(result.composite_score),
                score_breakdown: (!ranking_details_redacted).then_some(result.score_breakdown),
                raw_text_redacted,
                ranking_details_redacted,
                resource_evidence_count: result.resource_evidence.len(),
            }
        })
        .collect();

    let raw_text_redacted_results = result_explanations
        .iter()
        .filter(|result| result.raw_text_redacted)
        .count();

    let policy = build_policy_summary(
        requested_namespace,
        allowed_namespaces,
        raw_text_redacted_results,
    );

    RetrievalExplanation {
        suppression: RetrievalSuppressionSummary {
            candidate_count: diagnostics.records_scanned.unwrap_or(results.len()),
            threshold_filtered_count: diagnostics.threshold_filtered_count.unwrap_or(0),
            competitive_inhibition_count: diagnostics.competitive_inhibition_count.unwrap_or(0),
            truncated_by_limit_count: diagnostics.truncated_by_limit_count.unwrap_or(0),
        },
        diagnostics,
        scoring_weights,
        policy,
        raw_text_redacted_results,
        results: result_explanations,
    }
}

fn build_policy_summary(
    requested_namespace: Option<Namespace>,
    allowed_namespaces: Option<Vec<Namespace>>,
    raw_text_redacted_results: usize,
) -> RetrievalPolicySummary {
    let scope = match (requested_namespace, allowed_namespaces.as_ref()) {
        (Some(requested), Some(allowed)) if !allowed.contains(&requested) => {
            RetrievalPolicyScope::RequestedNamespaceDenied
        }
        (Some(_), _) => RetrievalPolicyScope::RequestedNamespace,
        (None, Some(_)) => RetrievalPolicyScope::AllowedNamespaces,
        (None, None) => RetrievalPolicyScope::Unrestricted,
    };

    RetrievalPolicySummary {
        scope,
        requested_namespace,
        allowed_namespaces,
        namespace_restriction_applied: !matches!(scope, RetrievalPolicyScope::Unrestricted),
        raw_text_redaction_applied: raw_text_redacted_results > 0,
    }
}

pub(crate) fn build_think_explanation(
    retrieval: RetrievalExplanation,
    think_result: &ThinkResult,
    token_budget: usize,
) -> ThinkExplanation {
    ThinkExplanation {
        retrieval,
        token_budget,
        token_count: think_result.token_count,
        records_included_count: think_result.records_included.len(),
        records_excluded_count: think_result.records_excluded_count,
        conflict_group_count: think_result.conflict_groups.len(),
        query_time_ms: think_result.query_time_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_summary_reports_allowed_namespace_scope() {
        let policy = build_policy_summary(
            None,
            Some(vec![Namespace::default_ns(), Namespace::shared()]),
            0,
        );

        assert_eq!(policy.scope, RetrievalPolicyScope::AllowedNamespaces);
        assert!(policy.namespace_restriction_applied);
        assert!(!policy.raw_text_redaction_applied);
    }

    #[test]
    fn policy_summary_reports_denied_requested_namespace() {
        let requested = Namespace::new("private-a").unwrap();
        let allowed = Namespace::new("private-b").unwrap();

        let policy = build_policy_summary(Some(requested), Some(vec![allowed]), 1);

        assert_eq!(policy.scope, RetrievalPolicyScope::RequestedNamespaceDenied);
        assert!(policy.namespace_restriction_applied);
        assert!(policy.raw_text_redaction_applied);
        assert_eq!(policy.requested_namespace, Some(requested));
    }
}
