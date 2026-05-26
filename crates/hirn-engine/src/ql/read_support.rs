use std::collections::HashSet;

use futures::TryStreamExt;

use hirn_core::id::MemoryId;
use hirn_core::record::MemoryRecord;
use hirn_core::types::{AgentId, Layer, Namespace};
use hirn_core::{
    DerivedArtifactKind, EvidenceRole, HirnError, HirnResult, HydrationMode, RecallSnapshot,
};

use crate::db::HirnDB;
use crate::graph_store::GraphStore;
use crate::ql::ast::{
    AggFunction, ComparisonOp, ConditionValue, OutputFormat, RecallStmt, Subquery, ThinkStmt,
    WhereCondition,
};
use crate::ql::results::{
    AggregatedGroup, AggregatedResults, ProjectedRecord, QueryResult, RecordResults,
    ScoreBreakdown, ScoredMemory,
};
use crate::resource_presentation::{
    ResourcePreviewPackage, hydrate_resource_preview_packages_for_scored_records,
    resource_preview_packages_to_json, resource_score_attribution_to_json,
};

pub(crate) async fn postprocess_scored_recall_results(
    db: &HirnDB,
    stmt: &RecallStmt,
    mut scored: Vec<ScoredMemory>,
    allowed_query_namespaces: Option<&[Namespace]>,
) -> HirnResult<Vec<ScoredMemory>> {
    apply_mcfa_filter_to_scored(&mut scored, stmt.with_mcfa);

    if let Some(ref modalities) = stmt.modality {
        scored.retain(|record| scored_memory_matches_modalities(record, modalities));
    }

    apply_resource_evidence_filters_to_scored(&mut scored, stmt)?;

    if let Some(depth) = stmt.provenance_depth {
        if depth > 0 {
            scored = expand_provenance(db, scored, depth, allowed_query_namespaces).await;
        }
    }

    if let Some(ref topic_label) = stmt.topic {
        scored = filter_by_topic(db, scored, topic_label).await;
    }

    Ok(scored)
}

pub(crate) async fn apply_scored_recall_postload_filters(
    db: &HirnDB,
    stmt: &RecallStmt,
    mut scored: Vec<ScoredMemory>,
    actor_id: AgentId,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<Vec<ScoredMemory>> {
    apply_recall_postload_filters(
        db,
        stmt,
        &stmt.where_clauses,
        &mut scored,
        actor_id,
        allowed_namespaces,
        |record| &record.record,
    )
    .await?;

    Ok(scored)
}

pub(crate) async fn finalize_scored_recall_results(
    db: &HirnDB,
    stmt: &RecallStmt,
    mut scored: Vec<ScoredMemory>,
    actor_id: AgentId,
    allowed_query_namespaces: Option<&[Namespace]>,
    recall_snapshot: Option<RecallSnapshot>,
    records_scanned: usize,
    query_time_ms: f64,
    recall_preview_policy: crate::retrieval::recall::RecallPreviewPolicy,
) -> HirnResult<(QueryResult, usize)> {
    if let Some(limit) = stmt.limit {
        if scored.len() > limit {
            scored.truncate(limit);
        }
    }
    let records_returned = scored.len();

    let conflict_summary = if stmt.with_conflicts {
        Some(
            super::context::detect_conflicts_for_recall(
                db,
                &scored,
                allowed_query_namespaces,
                recall_snapshot,
            )
            .await,
        )
    } else {
        None
    };
    let conflicts = conflict_summary
        .as_ref()
        .map(|summary| summary.pairs.clone());
    let conflict_groups = conflict_summary
        .as_ref()
        .map(|summary| summary.groups.clone());

    if let Some(ref group_by) = stmt.group_by {
        let groups = apply_aggregation(&scored, &group_by.field, group_by.function);
        let formatted = format_aggregated(
            &groups,
            &group_by.field,
            group_by.function,
            stmt.result_format,
        );
        return Ok((
            QueryResult::Aggregated(AggregatedResults {
                group_field: group_by.field.clone(),
                function: group_by.function,
                groups,
                query_time_ms,
                formatted,
            }),
            records_returned,
        ));
    }

    if let Some(ref fields) = stmt.projection {
        let preview_packages = if fields
            .iter()
            .any(|field| field == "resource_preview_packages")
        {
            Some(
                hydrate_resource_preview_packages_for_scored_records(
                    db,
                    &actor_id,
                    &scored,
                    recall_preview_policy.package.max_previews,
                    recall_preview_policy.package.max_chars,
                )
                .await?,
            )
        } else {
            None
        };
        let projected: Vec<ProjectedRecord> = scored
            .iter()
            .map(|record| {
                project_record_with_preview_packages(record, fields, preview_packages.as_ref())
            })
            .collect();
        let formatted = format_projected(&projected, stmt.result_format);
        return Ok((
            QueryResult::Records(RecordResults {
                records: scored,
                query_time_ms,
                records_scanned,
                records_returned,
                context: Some(formatted),
                conflicts: conflicts.clone(),
                conflict_groups: conflict_groups.clone(),
            }),
            records_returned,
        ));
    }

    if let Some(ref fmt) = stmt.result_format {
        let preview_packages = hydrate_resource_preview_packages_for_scored_records(
            db,
            &actor_id,
            &scored,
            recall_preview_policy.package.max_previews,
            recall_preview_policy.package.max_chars,
        )
        .await?;
        let formatted =
            format_records_with_preview_packages(&scored, *fmt, Some(&preview_packages));
        return Ok((
            QueryResult::Records(RecordResults {
                records: scored,
                query_time_ms,
                records_scanned,
                records_returned,
                context: Some(formatted),
                conflicts: conflicts.clone(),
                conflict_groups: conflict_groups.clone(),
            }),
            records_returned,
        ));
    }

    let context = match stmt.output_format {
        Some(OutputFormat::Narrative) => {
            Some(super::context::format_as_narrative(db, &scored).await)
        }
        Some(OutputFormat::CausalChain) => {
            Some(super::context::format_as_causal_chain(db, &scored, stmt.follow_causes).await)
        }
        Some(OutputFormat::Graph) => Some(super::context::format_as_graph(db, &scored).await),
        _ => None,
    };

    Ok((
        QueryResult::Records(RecordResults {
            records: scored,
            query_time_ms,
            records_scanned,
            records_returned,
            context,
            conflicts,
            conflict_groups,
        }),
        records_returned,
    ))
}

pub(crate) fn recall_stmt_from_think(stmt: &ThinkStmt) -> RecallStmt {
    RecallStmt {
        layers: vec![Layer::Episodic, Layer::Semantic],
        about: stmt.about.clone(),
        involving: stmt.involving.clone(),
        temporal: stmt.temporal.clone(),
        expand: stmt.expand.clone(),
        follow_causes: stmt.follow_causes,
        where_clauses: stmt.where_clauses.clone(),
        modality: None,
        resource_roles: None,
        hydration_modes: None,
        artifact_kinds: None,
        group_by: None,
        projection: None,
        output_format: stmt.output_format,
        result_format: None,
        as_of: None,
        subquery_filters: vec![],
        budget: stmt.budget,
        namespace: stmt.namespace.clone(),
        consistency: stmt.consistency,
        limit: stmt.limit,
        hybrid: stmt.hybrid,
        depth_mode: stmt.depth_mode,
        with_prospective: stmt.with_prospective,
        with_mcfa: stmt.with_mcfa,
        with_conflicts: false,
        provenance_depth: stmt.provenance_depth,
        topic: None,
        from_realms: None,
    }
}

pub(crate) fn classify_recall_depth(stmt: &RecallStmt) -> hirn_exec::operators::Complexity {
    use hirn_exec::operators::{ComplexityConfig, QueryFeatures};

    let features = QueryFeatures {
        token_count: stmt.about.split_whitespace().count(),
        has_temporal: stmt.temporal.is_some(),
        entity_count: stmt.involving.as_ref().map_or(0, |value| value.len()),
        graph_depth: stmt.expand.as_ref().map_or(0, |expand| expand.depth as u32),
        has_causal: stmt.follow_causes.is_some(),
        is_iterative: false,
    };

    let config = ComplexityConfig::default();
    let (complexity, points) = features.classify(&config);
    tracing::trace!(
        ?complexity,
        points,
        token_count = features.token_count,
        has_temporal = features.has_temporal,
        entity_count = features.entity_count,
        graph_depth = features.graph_depth,
        "query complexity classification"
    );
    complexity
}

pub(crate) fn recall_quality_should_escalate(results: &[ScoredMemory], threshold: f32) -> bool {
    compute_quality_score(results, threshold).should_escalate
}

/// Detect common English temporal keywords in free-text query strings.
///
/// Used to enable temporal expansion even when no explicit `BETWEEN`/`AS OF`
/// clause is present in the HirnQL statement.
pub(crate) fn detect_temporal_in_query_text(query: &str) -> bool {
    // Case-insensitive scan for common temporal anchors.
    let lower = query.to_ascii_lowercase();
    let patterns = [
        "yesterday",
        "last week",
        "last month",
        "last year",
        "last night",
        "this week",
        "this month",
        "this year",
        "today",
        "recently",
        "earlier today",
        "a few days ago",
        "a week ago",
        "a month ago",
        "a year ago",
        "days ago",
        "weeks ago",
        "months ago",
        "years ago",
    ];
    patterns.iter().any(|p| lower.contains(p))
}

/// Derive a `[start_ms, end_ms]` window from common temporal phrases in a
/// free-text query relative to `now_ms` (Unix milliseconds).
///
/// Returns `None` when no recognisable pattern is found, leaving the caller to
/// treat the query as temporally unbounded.
pub(crate) fn derive_temporal_bounds_from_query_text(
    query: &str,
    now_ms: i64,
) -> Option<(i64, i64)> {
    const MS_PER_DAY: i64 = 86_400_000;
    let lower = query.to_ascii_lowercase();

    // Helper: window ending at now, spanning `days` days back.
    let window = |days: i64| -> (i64, i64) {
        let start = now_ms.saturating_sub(days * MS_PER_DAY);
        (start, now_ms)
    };

    if lower.contains("yesterday") {
        let start = now_ms.saturating_sub(2 * MS_PER_DAY);
        let end = now_ms.saturating_sub(MS_PER_DAY);
        return Some((start, end));
    }
    if lower.contains("earlier today") || lower.contains("today") {
        let start = now_ms.saturating_sub(MS_PER_DAY);
        return Some((start, now_ms));
    }
    if lower.contains("last night") {
        let start = now_ms.saturating_sub(2 * MS_PER_DAY);
        let end = now_ms.saturating_sub(MS_PER_DAY);
        return Some((start, end));
    }
    if lower.contains("last week") || lower.contains("this week") {
        return Some(window(7));
    }
    if lower.contains("a week ago") || lower.contains("weeks ago") {
        return Some(window(14));
    }
    if lower.contains("last month") || lower.contains("this month") || lower.contains("a month ago") {
        return Some(window(30));
    }
    if lower.contains("months ago") {
        return Some(window(90));
    }
    if lower.contains("last year") || lower.contains("this year") || lower.contains("a year ago") {
        return Some(window(365));
    }
    if lower.contains("years ago") {
        return Some(window(730));
    }
    if lower.contains("recently") || lower.contains("a few days ago") || lower.contains("days ago") {
        return Some(window(7));
    }
    None
}

pub(crate) fn recall_candidate_limit(
    limit: usize,
    stmt: &RecallStmt,
    has_storage_backed_filter: bool,
) -> usize {
    const RECALL_OVERFETCH_FACTOR: usize = 3;
    const RECALL_OVERFETCH_CAP: usize = 64;
    const NARROWING_OVERFETCH_FACTOR: usize = 4;
    const NARROWING_OVERFETCH_CAP: usize = 128;

    let base = limit.max(1);
    let overfetched = base.saturating_mul(RECALL_OVERFETCH_FACTOR);
    let capped = base.saturating_add(RECALL_OVERFETCH_CAP);
    let candidate_limit = overfetched.min(capped).max(base);

    if !(has_storage_backed_filter || recall_has_narrowing_postload(stmt)) {
        return candidate_limit;
    }

    let boosted = candidate_limit.saturating_mul(NARROWING_OVERFETCH_FACTOR);
    let boosted_cap = candidate_limit.saturating_add(NARROWING_OVERFETCH_CAP);
    boosted.min(boosted_cap).max(candidate_limit)
}

fn recall_has_narrowing_postload(stmt: &RecallStmt) -> bool {
    stmt.temporal.is_some()
        || stmt.as_of.is_some()
        || stmt
            .involving
            .as_ref()
            .is_some_and(|entities| !entities.is_empty())
        || !stmt.where_clauses.is_empty()
        || !stmt.subquery_filters.is_empty()
        || stmt.modality.is_some()
        || stmt.resource_roles.is_some()
        || stmt.hydration_modes.is_some()
        || stmt.artifact_kinds.is_some()
        || stmt.topic.is_some()
}

fn apply_mcfa_filter_to_scored(scored: &mut Vec<ScoredMemory>, enabled: Option<bool>) {
    if !enabled.unwrap_or(false) {
        return;
    }

    let config = hirn_exec::operators::McfaConfig::default();
    scored.retain(|record| {
        hirn_exec::operators::detect_threat(scored_memory_content(&record.record), &config)
            .is_none()
    });
}

fn scored_memory_content(record: &MemoryRecord) -> &str {
    match record {
        MemoryRecord::Episodic(record) => &record.content,
        MemoryRecord::Semantic(record) => &record.description,
        MemoryRecord::Working(record) => &record.content,
        MemoryRecord::Procedural(record) => &record.description,
    }
}

async fn apply_recall_postload_filters<T, F>(
    db: &HirnDB,
    stmt: &RecallStmt,
    where_clauses: &[WhereCondition],
    items: &mut Vec<T>,
    actor_id: AgentId,
    allowed_namespaces: Option<&[Namespace]>,
    record_of: F,
) -> HirnResult<()>
where
    T: Send + Sync,
    F: Fn(&T) -> &MemoryRecord + Send + Sync,
{
    if let Some(involving) = &stmt.involving {
        items.retain(|item| record_matches_involving(record_of(item), involving));
    }

    let (trust_clauses, regular_clauses): (Vec<_>, Vec<_>) = where_clauses
        .iter()
        .partition(|clause| clause.field == "trust");

    for clause in regular_clauses {
        items.retain(|item| record_matches_condition(record_of(item), clause));
    }

    if !trust_clauses.is_empty() {
        let store = db.graph_store();
        let mut remove_indices = HashSet::new();
        for (index, item) in items.iter().enumerate() {
            let id = record_of(item).id();
            let contra_count = store
                .get_edges_of_type(id, hirn_core::types::EdgeRelation::Contradicts)
                .await
                .unwrap_or_default()
                .len();
            let trust = if let Some(provenance) = record_provenance(record_of(item)) {
                crate::causal::compute_trust_score(provenance, contra_count) as f64
            } else {
                1.0
            };

            if !trust_clauses.iter().all(|clause| {
                let threshold = match &clause.value {
                    ConditionValue::Float(value) => *value,
                    ConditionValue::Int(value) => *value as f64,
                    ConditionValue::String(_) | ConditionValue::Param(_) => return true,
                };
                matches_comparison(trust, &clause.op, threshold)
            }) {
                remove_indices.insert(index);
            }
        }

        let mut index = 0;
        items.retain(|_| {
            let keep = !remove_indices.contains(&index);
            index += 1;
            keep
        });
    }

    if stmt.subquery_filters.is_empty() {
        return Ok(());
    }

    for subquery_filter in &stmt.subquery_filters {
        let inner_values =
            resolve_subquery(db, &subquery_filter.subquery, actor_id, allowed_namespaces).await?;
        items.retain(|item| {
            record_field_in_set(record_of(item), &subquery_filter.field, &inner_values)
        });
    }

    Ok(())
}

fn record_provenance(record: &MemoryRecord) -> Option<&hirn_core::provenance::Provenance> {
    match record {
        MemoryRecord::Episodic(record) => Some(&record.provenance),
        MemoryRecord::Semantic(record) => Some(&record.provenance),
        MemoryRecord::Working(_) => None,
        MemoryRecord::Procedural(record) => Some(&record.provenance),
    }
}

fn record_matches_involving(record: &MemoryRecord, entities: &[String]) -> bool {
    match record {
        MemoryRecord::Episodic(record) => record
            .entities
            .iter()
            .any(|entity| entities.contains(&entity.name)),
        MemoryRecord::Semantic(record) => entities.contains(&record.concept),
        MemoryRecord::Working(_) | MemoryRecord::Procedural(_) => false,
    }
}

fn matches_comparison(value: f64, op: &ComparisonOp, threshold: f64) -> bool {
    match op {
        ComparisonOp::Gt => value > threshold,
        ComparisonOp::Lt => value < threshold,
        ComparisonOp::Gte => value >= threshold,
        ComparisonOp::Lte => value <= threshold,
        ComparisonOp::Eq => (value - threshold).abs() < f64::EPSILON,
        ComparisonOp::Neq => (value - threshold).abs() >= f64::EPSILON,
    }
}

fn resolve_subquery<'a>(
    db: &'a HirnDB,
    subquery: &'a Subquery,
    actor_id: AgentId,
    allowed_namespaces: Option<&'a [Namespace]>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = HirnResult<HashSet<String>>> + Send + 'a>> {
    Box::pin(async move {
        let recall_stmt = RecallStmt {
            layers: subquery.layers.clone(),
            about: subquery.about.clone(),
            involving: subquery.involving.clone(),
            temporal: subquery.temporal.clone(),
            expand: None,
            follow_causes: None,
            where_clauses: vec![],
            modality: None,
            resource_roles: None,
            hydration_modes: None,
            artifact_kinds: None,
            group_by: None,
            projection: None,
            output_format: None,
            result_format: None,
            as_of: None,
            subquery_filters: vec![],
            budget: None,
            namespace: None,
            consistency: None,
            limit: subquery.limit,
            hybrid: false,
            depth_mode: None,
            with_prospective: None,
            with_mcfa: None,
            with_conflicts: false,
            provenance_depth: None,
            topic: None,
            from_realms: None,
        };
        let result = db
            .try_execute_compiled_recall_statement_untracked(
                &recall_stmt,
                actor_id,
                allowed_namespaces,
            )
            .await?
            .ok_or_else(|| {
                HirnError::Unsupported(
                    "subquery RECALL must execute on the compiled DataFusion bridge".to_string(),
                )
            })?;

        let mut values = HashSet::new();
        if let QueryResult::Records(records) = result {
            for scored in &records.records {
                match &scored.record {
                    MemoryRecord::Episodic(record) => {
                        for entity in &record.entities {
                            values.insert(entity.name.clone());
                        }
                    }
                    MemoryRecord::Semantic(record) => {
                        values.insert(record.concept.clone());
                    }
                    MemoryRecord::Working(_) | MemoryRecord::Procedural(_) => {}
                }
                values.insert(scored.record.id().to_string());
            }
        }

        Ok(values)
    })
}

fn extract_field_value_from_record(record: &MemoryRecord, field: &str) -> Option<String> {
    match field {
        "id" => return Some(record.id().to_string()),
        "layer" => return Some(format!("{:?}", record.layer()).to_lowercase()),
        _ => {}
    }
    match record {
        MemoryRecord::Episodic(record) => match field {
            "entity" | "entities" => record.entities.first().map(|entity| entity.name.clone()),
            "summary" => Some(record.summary.clone()),
            "content" => Some(record.content.clone()),
            "event_type" => Some(format!("{:?}", record.event_type)),
            "namespace" => Some(record.namespace.to_string()),
            _ => None,
        },
        MemoryRecord::Semantic(record) => match field {
            "entity" | "entities" => Some(record.concept.clone()),
            "concept" | "summary" => Some(record.concept.clone()),
            "description" | "content" => Some(record.description.clone()),
            "knowledge_type" => Some(format!("{:?}", record.knowledge_type)),
            "namespace" => Some(record.namespace.to_string()),
            _ => None,
        },
        MemoryRecord::Working(record) => match field {
            "content" | "summary" => Some(record.content.clone()),
            _ => None,
        },
        MemoryRecord::Procedural(record) => match field {
            "name" | "summary" => Some(record.name.clone()),
            "description" | "content" => Some(record.description.clone()),
            "namespace" => Some(record.namespace.to_string()),
            _ => None,
        },
    }
}

fn record_field_in_set(record: &MemoryRecord, field: &str, values: &HashSet<String>) -> bool {
    match field {
        "entity" | "entities" => match record {
            MemoryRecord::Episodic(record) => record
                .entities
                .iter()
                .any(|entity| values.contains(&entity.name)),
            MemoryRecord::Semantic(record) => values.contains(&record.concept),
            MemoryRecord::Working(_) | MemoryRecord::Procedural(_) => false,
        },
        _ => extract_field_value_from_record(record, field)
            .is_some_and(|value| values.contains(&value)),
    }
}

fn scored_memory_matches_modalities(result: &ScoredMemory, modalities: &[String]) -> bool {
    let record_modality = match &result.record {
        MemoryRecord::Episodic(record) => record
            .multi_content
            .as_ref()
            .map(|content| content.modality()),
        MemoryRecord::Semantic(_) | MemoryRecord::Working(_) | MemoryRecord::Procedural(_) => None,
    };

    if let Some(record_modality) = record_modality {
        return modalities.iter().any(|wanted| wanted == record_modality);
    }

    if result.resource_evidence.iter().any(|summary| {
        summary
            .modality
            .is_some_and(|modality| modalities.iter().any(|wanted| wanted == modality.as_str()))
    }) {
        return true;
    }

    result.resource_evidence.is_empty() && modalities.iter().any(|wanted| wanted == "text")
}

fn apply_resource_evidence_filters_to_scored(
    results: &mut Vec<ScoredMemory>,
    stmt: &RecallStmt,
) -> HirnResult<()> {
    let resource_roles = parse_evidence_roles(stmt.resource_roles.as_ref())?;
    let hydration_modes = parse_hydration_modes(stmt.hydration_modes.as_ref())?;
    let artifact_kinds = parse_artifact_kinds(stmt.artifact_kinds.as_ref())?;

    if resource_roles.is_none() && hydration_modes.is_none() && artifact_kinds.is_none() {
        return Ok(());
    }

    results.retain_mut(|result| {
        let filtered_evidence: Vec<_> = result
            .resource_evidence
            .iter()
            .cloned()
            .filter_map(|summary| {
                let summary = filter_artifact_selection(summary, artifact_kinds.as_deref());
                matches_resource_filters(
                    &summary,
                    resource_roles.as_deref(),
                    hydration_modes.as_deref(),
                    artifact_kinds.as_deref(),
                )
                .then_some(summary)
            })
            .collect();

        result.resource_evidence = filtered_evidence;
        !result.resource_evidence.is_empty()
    });

    Ok(())
}

fn filter_artifact_selection(
    mut summary: crate::recall::ResourceEvidenceSummary,
    artifact_kinds: Option<&[DerivedArtifactKind]>,
) -> crate::recall::ResourceEvidenceSummary {
    if let Some(artifact_kinds) = artifact_kinds {
        summary
            .available_artifacts
            .retain(|kind| artifact_kinds.contains(kind));
        if summary
            .artifact_kind
            .is_some_and(|kind| !artifact_kinds.contains(&kind))
        {
            summary.artifact_kind = None;
            summary.artifact_id = None;
        }
        summary.has_preview = summary
            .available_artifacts
            .iter()
            .any(|kind| kind.is_previewable());
        summary.can_hydrate_preview &= summary.has_preview;
    }

    summary
}

fn matches_resource_filters(
    summary: &crate::recall::ResourceEvidenceSummary,
    resource_roles: Option<&[EvidenceRole]>,
    hydration_modes: Option<&[HydrationMode]>,
    artifact_kinds: Option<&[DerivedArtifactKind]>,
) -> bool {
    let matches_role = resource_roles.is_none_or(|roles| roles.contains(&summary.role));
    let matches_hydration = hydration_modes.is_none_or(|modes| {
        modes.iter().any(|mode| match mode {
            HydrationMode::MetadataOnly => true,
            HydrationMode::Preview => summary.has_preview && summary.can_hydrate_preview,
            HydrationMode::Full => summary.can_hydrate_full,
        })
    });
    let matches_artifact = artifact_kinds.is_none_or(|_| !summary.available_artifacts.is_empty());

    matches_role && matches_hydration && matches_artifact
}

fn parse_evidence_roles(values: Option<&Vec<String>>) -> HirnResult<Option<Vec<EvidenceRole>>> {
    values
        .map(|entries| {
            entries
                .iter()
                .map(|value| EvidenceRole::parse(value))
                .collect::<HirnResult<Vec<_>>>()
                .map(Some)
        })
        .unwrap_or(Ok(None))
}

fn parse_hydration_modes(values: Option<&Vec<String>>) -> HirnResult<Option<Vec<HydrationMode>>> {
    values
        .map(|entries| {
            entries
                .iter()
                .map(|value| HydrationMode::parse(value))
                .collect::<HirnResult<Vec<_>>>()
                .map(Some)
        })
        .unwrap_or(Ok(None))
}

fn parse_artifact_kinds(
    values: Option<&Vec<String>>,
) -> HirnResult<Option<Vec<DerivedArtifactKind>>> {
    values
        .map(|entries| {
            entries
                .iter()
                .map(|value| DerivedArtifactKind::parse(value))
                .collect::<HirnResult<Vec<_>>>()
                .map(Some)
        })
        .unwrap_or(Ok(None))
}

async fn expand_provenance(
    db: &HirnDB,
    primary: Vec<ScoredMemory>,
    depth: usize,
    allowed_namespaces: Option<&[Namespace]>,
) -> Vec<ScoredMemory> {
    use hirn_core::types::EdgeRelation;

    if primary.is_empty() || depth == 0 {
        return primary;
    }

    let graph = db.cached_graph();
    let min_primary_score = primary
        .iter()
        .map(|record| record.score)
        .min_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0);
    let expansion_score = (min_primary_score * 0.5).max(0.01);

    let primary_ids: HashSet<MemoryId> = primary.iter().map(|record| record.record.id()).collect();
    let mut seen = primary_ids.clone();
    let mut frontier: Vec<MemoryId> = primary.iter().map(|record| record.record.id()).collect();
    let mut expanded = Vec::new();

    for level in 0..depth {
        let mut next_frontier = Vec::new();
        for id in &frontier {
            let derived = graph
                .get_edges_of_type(*id, EdgeRelation::DerivedFrom)
                .await
                .unwrap_or_default();
            let parts = graph
                .get_edges_of_type(*id, EdgeRelation::PartOf)
                .await
                .unwrap_or_default();

            for edge in derived.into_iter().chain(parts) {
                let target_id = edge.target;
                if seen.contains(&target_id) {
                    continue;
                }
                seen.insert(target_id);
                next_frontier.push(target_id);

                if let Ok(record) = db.get_memory(target_id).await {
                    if let Some(namespaces) = allowed_namespaces {
                        let record_ns = record.effective_namespace();
                        if !namespaces.contains(&record_ns) {
                            tracing::debug!(
                                record_id = %target_id,
                                record_ns = %record_ns,
                                allowed_namespaces = ?namespaces,
                                "provenance expansion: skipping record from different namespace"
                            );
                            continue;
                        }
                    }
                    let depth_penalty = 1.0 / (level as f32 + 2.0);
                    expanded.push(ScoredMemory {
                        revision: None,
                        score: expansion_score * depth_penalty,
                        score_breakdown: ScoreBreakdown {
                            similarity: 0.0,
                            importance: 0.0,
                            recency: 0.0,
                            activation: 0.0,
                            causal_relevance: 0.0,
                            surprise: 0.0,
                            source_reliability: 0.0,
                        },
                        record,
                        resource_evidence: Vec::new(),
                        resource_preview_packages: Vec::new(),
                        resource_score_attribution: Vec::new(),
                    });
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    let mut result = primary;
    result.extend(expanded);
    result
}

async fn filter_by_topic(
    db: &HirnDB,
    scored: Vec<ScoredMemory>,
    topic_label: &str,
) -> Vec<ScoredMemory> {
    use arrow_array::Array;
    use hirn_storage::store::ScanOptions;

    let filter = format!("topic_label = '{}'", topic_label.replace('\'', "''"));
    let opts = ScanOptions {
        columns: Some(vec!["memory_id".to_string()]),
        filter: Some(filter),
        exact_filter: None,
        order_by: None,
        limit: None,
        offset: None,
    };
    let mut batches = match db.storage_backend().scan_stream("topic_loom", opts).await {
        Ok(batches) => batches,
        Err(_) => return scored,
    };

    let mut topic_ids = HashSet::new();
    loop {
        let batch = match batches.try_next().await {
            Ok(Some(batch)) => batch,
            Ok(None) => break,
            Err(_) => return scored,
        };
        let col: &dyn Array = match batch.column_by_name("memory_id") {
            Some(column) => column.as_ref(),
            None => continue,
        };
        if let Some(array) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
            for index in 0..array.len() {
                if !array.is_null(index) {
                    if let Ok(id) = MemoryId::parse(array.value(index)) {
                        topic_ids.insert(id);
                    }
                }
            }
        }
    }

    if topic_ids.is_empty() {
        return scored;
    }

    scored
        .into_iter()
        .filter(|record| topic_ids.contains(&record.record.id()))
        .collect()
}

struct QualityAssessment {
    should_escalate: bool,
}

fn compute_quality_score(results: &[ScoredMemory], threshold: f32) -> QualityAssessment {
    let count = results.len();
    let coverage = (count as f32 / 5.0).min(1.0);
    let confidence = if count > 0 {
        results.iter().map(|record| record.score).sum::<f32>() / count as f32
    } else {
        0.0
    };
    let coherence = compute_coherence(results);
    let sufficiency = (count as f32 / 10.0).min(1.0);

    let score = coverage * 0.3 + confidence * 0.3 + coherence * 0.2 + sufficiency * 0.2;
    QualityAssessment {
        should_escalate: score < threshold,
    }
}

fn compute_coherence(results: &[ScoredMemory]) -> f32 {
    let embeddings: Vec<&[f32]> = results
        .iter()
        .filter_map(|record| match &record.record {
            MemoryRecord::Episodic(episodic) => episodic.embedding.as_deref(),
            MemoryRecord::Semantic(semantic) => semantic.embedding.as_deref(),
            MemoryRecord::Procedural(procedural) => procedural.embedding.as_deref(),
            MemoryRecord::Working(_) => None,
        })
        .collect();

    if embeddings.len() < 2 {
        return 0.6;
    }

    let mut sum = 0.0_f32;
    let mut count = 0_u32;
    for left in 0..embeddings.len() {
        for right in (left + 1)..embeddings.len() {
            sum += cosine_similarity(embeddings[left], embeddings[right]);
            count += 1;
        }
    }

    if count > 0 {
        (sum / count as f32).clamp(0.0, 1.0)
    } else {
        0.6
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_left = 0.0_f32;
    let mut norm_right = 0.0_f32;
    for (x, y) in left.iter().zip(right.iter()) {
        dot += x * y;
        norm_left += x * x;
        norm_right += y * y;
    }
    let denom = norm_left.sqrt() * norm_right.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

fn record_matches_condition(record: &MemoryRecord, condition: &WhereCondition) -> bool {
    let field = condition.field.as_str();
    let value = match record {
        MemoryRecord::Episodic(record) => match field {
            "importance" => Some(record.importance as f64),
            "surprise" => Some(record.surprise as f64),
            "access_count" | "episodic.access_count" => Some(record.access_count as f64),
            "confidence" => Some(record.importance as f64),
            _ => None,
        },
        MemoryRecord::Semantic(record) => match field {
            "confidence" => Some(record.confidence as f64),
            "evidence_count" => Some(record.evidence_count as f64),
            "access_count" => Some(record.access_count as f64),
            "importance" => Some(record.confidence as f64),
            _ => None,
        },
        MemoryRecord::Working(record) => match field {
            "relevance_score" | "importance" => Some(record.relevance_score as f64),
            _ => None,
        },
        MemoryRecord::Procedural(record) => match field {
            "success_rate" | "importance" => Some(record.success_rate as f64),
            "invocation_count" => Some(record.invocation_count as f64),
            "access_count" => Some(record.access_count as f64),
            _ => None,
        },
    };

    let Some(record_value) = value else {
        return false;
    };

    let threshold = match &condition.value {
        ConditionValue::Float(value) => *value,
        ConditionValue::Int(value) => *value as f64,
        ConditionValue::String(_) | ConditionValue::Param(_) => return true,
    };

    match condition.op {
        ComparisonOp::Gt => record_value > threshold,
        ComparisonOp::Lt => record_value < threshold,
        ComparisonOp::Gte => record_value >= threshold,
        ComparisonOp::Lte => record_value <= threshold,
        ComparisonOp::Eq => (record_value - threshold).abs() < f64::EPSILON,
        ComparisonOp::Neq => (record_value - threshold).abs() >= f64::EPSILON,
    }
}

fn extract_group_key(record: &MemoryRecord, field: &str) -> String {
    match record {
        MemoryRecord::Episodic(record) => match field {
            "event_type" | "entity_type" => format!("{:?}", record.event_type),
            "layer" => "episodic".to_string(),
            "importance" => format!("{:.1}", record.importance),
            "namespace" => record.namespace.to_string(),
            _ => "unknown".to_string(),
        },
        MemoryRecord::Semantic(record) => match field {
            "knowledge_type" | "entity_type" => format!("{:?}", record.knowledge_type),
            "layer" => "semantic".to_string(),
            "confidence" | "importance" => format!("{:.1}", record.confidence),
            "namespace" => record.namespace.to_string(),
            _ => "unknown".to_string(),
        },
        MemoryRecord::Working(record) => match field {
            "layer" => "working".to_string(),
            "importance" => format!("{:.1}", record.relevance_score),
            _ => "unknown".to_string(),
        },
        MemoryRecord::Procedural(record) => match field {
            "layer" => "procedural".to_string(),
            "importance" => format!("{:.1}", record.success_rate),
            "namespace" => record.namespace.to_string(),
            _ => "unknown".to_string(),
        },
    }
}

fn extract_numeric_value(record: &MemoryRecord, field: &str) -> f64 {
    match record {
        MemoryRecord::Episodic(record) => match field {
            "importance" => record.importance as f64,
            "surprise" => record.surprise as f64,
            "access_count" => record.access_count as f64,
            _ => 0.0,
        },
        MemoryRecord::Semantic(record) => match field {
            "confidence" | "importance" => record.confidence as f64,
            "evidence_count" => record.evidence_count as f64,
            "access_count" => record.access_count as f64,
            _ => 0.0,
        },
        MemoryRecord::Working(record) => match field {
            "importance" | "relevance_score" => record.relevance_score as f64,
            _ => 0.0,
        },
        MemoryRecord::Procedural(record) => match field {
            "importance" | "success_rate" => record.success_rate as f64,
            "invocation_count" => record.invocation_count as f64,
            "access_count" => record.access_count as f64,
            _ => 0.0,
        },
    }
}

fn apply_aggregation(
    scored: &[ScoredMemory],
    field: &str,
    function: AggFunction,
) -> Vec<AggregatedGroup> {
    use std::collections::BTreeMap;

    let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for record in scored {
        let key = extract_group_key(&record.record, field);
        let value = match function {
            AggFunction::Count => 1.0,
            _ => extract_numeric_value(&record.record, "importance"),
        };
        groups.entry(key).or_default().push(value);
    }

    groups
        .into_iter()
        .map(|(key, values)| {
            let value = match function {
                AggFunction::Count => values.len() as f64,
                AggFunction::Sum => values.iter().sum(),
                AggFunction::Avg => {
                    if values.is_empty() {
                        0.0
                    } else {
                        values.iter().sum::<f64>() / values.len() as f64
                    }
                }
                AggFunction::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
                AggFunction::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            };
            AggregatedGroup { key, value }
        })
        .collect()
}

fn format_aggregated(
    groups: &[AggregatedGroup],
    field: &str,
    function: AggFunction,
    result_format: Option<OutputFormat>,
) -> Option<String> {
    let format = result_format?;
    match format {
        OutputFormat::Json => {
            let entries: Vec<String> = groups
                .iter()
                .map(|group| {
                    format!(
                        "  {{\"{}\": \"{}\", \"{}\": {}}}",
                        field, group.key, function, group.value
                    )
                })
                .collect();
            Some(format!("[\n{}\n]", entries.join(",\n")))
        }
        OutputFormat::Csv => {
            let mut lines = vec![format!("{},{}", csv_escape(field), function)];
            for group in groups {
                lines.push(format!("{},{}", csv_escape(&group.key), group.value));
            }
            Some(lines.join("\n"))
        }
        _ => None,
    }
}

fn extract_field_value(
    scored: &ScoredMemory,
    field: &str,
    preview_packages: Option<&std::collections::BTreeMap<MemoryId, Vec<ResourcePreviewPackage>>>,
) -> Option<serde_json::Value> {
    if field == "resource_evidence" {
        return Some(crate::result_json::resource_evidence_to_json(
            &scored.resource_evidence,
        ));
    }
    if field == "resource_hydration_available" {
        return Some(crate::result_json::resource_hydration_to_json(
            &scored.resource_evidence,
        ));
    }
    if field == "resource_preview_packages" {
        let packages = preview_packages
            .and_then(|packages| packages.get(&scored.record.id()))
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        return Some(resource_preview_packages_to_json(packages));
    }
    if field == "resource_score_attribution" {
        return Some(resource_score_attribution_to_json(
            &scored.resource_score_attribution,
        ));
    }

    match field {
        "id" => return Some(serde_json::Value::String(scored.record.id().to_string())),
        "score" => return Some(serde_json::Value::String(format!("{:.4}", scored.score))),
        "layer" => {
            return Some(serde_json::Value::String(
                format!("{:?}", scored.record.layer()).to_lowercase(),
            ));
        }
        _ => {}
    }

    match &scored.record {
        MemoryRecord::Episodic(record) => match field {
            "summary" => Some(serde_json::Value::String(record.summary.clone())),
            "content" => Some(serde_json::Value::String(record.content.clone())),
            "importance" => Some(serde_json::Value::String(format!(
                "{:.3}",
                record.importance
            ))),
            "event_type" => Some(serde_json::Value::String(format!(
                "{:?}",
                record.event_type
            ))),
            "surprise" => Some(serde_json::Value::String(format!("{:.3}", record.surprise))),
            "access_count" => Some(serde_json::Value::String(record.access_count.to_string())),
            "timestamp" => Some(serde_json::Value::String(
                record.timestamp.as_datetime().to_rfc3339(),
            )),
            "namespace" => Some(serde_json::Value::String(record.namespace.to_string())),
            _ => None,
        },
        MemoryRecord::Semantic(record) => match field {
            "summary" | "concept" => Some(serde_json::Value::String(record.concept.clone())),
            "content" | "description" => {
                Some(serde_json::Value::String(record.description.clone()))
            }
            "importance" | "confidence" => Some(serde_json::Value::String(format!(
                "{:.3}",
                record.confidence
            ))),
            "knowledge_type" => Some(serde_json::Value::String(format!(
                "{:?}",
                record.knowledge_type
            ))),
            "evidence_count" => Some(serde_json::Value::String(record.evidence_count.to_string())),
            "access_count" => Some(serde_json::Value::String(record.access_count.to_string())),
            "namespace" => Some(serde_json::Value::String(record.namespace.to_string())),
            _ => None,
        },
        MemoryRecord::Working(record) => match field {
            "summary" | "content" => Some(serde_json::Value::String(record.content.clone())),
            "importance" | "relevance_score" => Some(serde_json::Value::String(format!(
                "{:.3}",
                record.relevance_score
            ))),
            _ => None,
        },
        MemoryRecord::Procedural(record) => match field {
            "summary" | "name" => Some(serde_json::Value::String(record.name.clone())),
            "content" | "description" => {
                Some(serde_json::Value::String(record.description.clone()))
            }
            "importance" | "success_rate" => Some(serde_json::Value::String(format!(
                "{:.3}",
                record.success_rate
            ))),
            "invocation_count" => Some(serde_json::Value::String(
                record.invocation_count.to_string(),
            )),
            "access_count" => Some(serde_json::Value::String(record.access_count.to_string())),
            "namespace" => Some(serde_json::Value::String(record.namespace.to_string())),
            _ => None,
        },
    }
}

fn project_record_with_preview_packages(
    scored: &ScoredMemory,
    fields: &[String],
    preview_packages: Option<&std::collections::BTreeMap<MemoryId, Vec<ResourcePreviewPackage>>>,
) -> ProjectedRecord {
    let mut map = std::collections::BTreeMap::new();
    for field in fields {
        if let Some(value) = extract_field_value(scored, field, preview_packages) {
            map.insert(field.clone(), value);
        }
    }
    ProjectedRecord {
        fields: map,
        score: scored.score,
    }
}

fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_owned()
    }
}

fn projected_value_to_csv_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::String(value) => value.clone(),
        _ => value.to_string(),
    }
}

fn format_projected(projected: &[ProjectedRecord], result_format: Option<OutputFormat>) -> String {
    let format = result_format.unwrap_or(OutputFormat::Json);
    match format {
        OutputFormat::Csv => {
            if projected.is_empty() {
                return String::new();
            }
            let headers: Vec<&String> = projected[0].fields.keys().collect();
            let mut lines = vec![
                headers
                    .iter()
                    .map(|header| csv_escape(header))
                    .collect::<Vec<_>>()
                    .join(","),
            ];
            for record in projected {
                let values: Vec<String> = headers
                    .iter()
                    .map(|header| {
                        let value = record
                            .fields
                            .get(*header)
                            .map(projected_value_to_csv_string)
                            .unwrap_or_default();
                        csv_escape(&value)
                    })
                    .collect();
                lines.push(values.join(","));
            }
            lines.join("\n")
        }
        _ => {
            let entries: Vec<serde_json::Value> = projected
                .iter()
                .map(|record| {
                    let object = record
                        .fields
                        .iter()
                        .map(|(key, value)| (key.clone(), value.clone()))
                        .collect::<serde_json::Map<String, serde_json::Value>>();
                    serde_json::Value::Object(object)
                })
                .collect();
            serde_json::to_string_pretty(&serde_json::Value::Array(entries))
                .unwrap_or_else(|_| "[]".to_string())
        }
    }
}

fn format_records_with_preview_packages(
    scored: &[ScoredMemory],
    format: OutputFormat,
    preview_packages: Option<&std::collections::BTreeMap<MemoryId, Vec<ResourcePreviewPackage>>>,
) -> String {
    let all_fields = &[
        "id".to_string(),
        "layer".to_string(),
        "score".to_string(),
        "summary".to_string(),
        "importance".to_string(),
        "resource_evidence".to_string(),
        "resource_hydration_available".to_string(),
        "resource_preview_packages".to_string(),
        "resource_score_attribution".to_string(),
    ];
    let projected: Vec<ProjectedRecord> = scored
        .iter()
        .map(|record| project_record_with_preview_packages(record, all_fields, preview_packages))
        .collect();
    format_projected(&projected, Some(format))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ql::ast::{DepthModeAst, RetrievalMode};

    fn minimal_recall_stmt() -> RecallStmt {
        RecallStmt {
            layers: vec![Layer::Episodic],
            about: "incident analysis".to_owned(),
            involving: None,
            temporal: None,
            as_of: None,
            expand: None,
            follow_causes: None,
            where_clauses: vec![],
            subquery_filters: vec![],
            modality: None,
            resource_roles: None,
            hydration_modes: None,
            artifact_kinds: None,
            depth_mode: None,
            with_prospective: None,
            with_mcfa: None,
            with_conflicts: false,
            provenance_depth: None,
            topic: None,
            group_by: None,
            projection: None,
            output_format: None,
            result_format: None,
            budget: None,
            namespace: None,
            from_realms: None,
            consistency: None,
            limit: Some(10),
            hybrid: true,
        }
    }

    #[test]
    fn recall_stmt_from_think_preserves_query_shape() {
        let think = ThinkStmt {
            about: "incident analysis".to_owned(),
            involving: Some(vec!["auth".to_owned()]),
            temporal: None,
            expand: None,
            follow_causes: Some(2),
            where_clauses: vec![],
            output_format: Some(OutputFormat::Narrative),
            budget: Some(512),
            namespace: Some("shared".to_owned()),
            consistency: None,
            limit: Some(5),
            hybrid: false,
            mode: RetrievalMode::Global,
            depth_mode: Some(DepthModeAst::Full),
            with_prospective: Some(false),
            with_mcfa: Some(true),
            provenance_depth: Some(2),
            max_hops: Some(4),
            community_depth: Some(3),
        };

        let recall = recall_stmt_from_think(&think);

        assert_eq!(recall.about, think.about);
        assert_eq!(recall.involving, think.involving);
        assert_eq!(recall.follow_causes, think.follow_causes);
        assert_eq!(recall.output_format, think.output_format);
        assert_eq!(recall.budget, think.budget);
        assert_eq!(recall.namespace, think.namespace);
        assert_eq!(recall.limit, think.limit);
        assert_eq!(recall.depth_mode, think.depth_mode);
        assert_eq!(recall.with_prospective, think.with_prospective);
        assert_eq!(recall.with_mcfa, think.with_mcfa);
        assert_eq!(recall.provenance_depth, think.provenance_depth);
        assert_eq!(recall.layers, vec![Layer::Episodic, Layer::Semantic]);
        assert!(!recall.hybrid);
    }

    #[test]
    fn recall_stmt_from_think_preserves_hybrid_flag() {
        let mut think = ThinkStmt {
            about: "incident analysis".to_owned(),
            involving: None,
            temporal: None,
            expand: None,
            follow_causes: None,
            where_clauses: vec![],
            output_format: None,
            budget: Some(512),
            namespace: Some("shared".to_owned()),
            consistency: None,
            limit: Some(5),
            hybrid: false,
            mode: RetrievalMode::Local,
            depth_mode: None,
            with_prospective: Some(false),
            with_mcfa: Some(false),
            provenance_depth: Some(0),
            max_hops: None,
            community_depth: None,
        };

        think.hybrid = true;
        let recall = recall_stmt_from_think(&think);
        assert!(recall.hybrid);
    }

    #[test]
    fn recall_candidate_limit_uses_base_overfetch_without_narrowing() {
        let stmt = minimal_recall_stmt();
        assert_eq!(recall_candidate_limit(10, &stmt, false), 30);
    }

    #[test]
    fn recall_candidate_limit_boosts_for_narrowing_postload_filters() {
        let mut stmt = minimal_recall_stmt();
        stmt.where_clauses.push(WhereCondition {
            field: "importance".to_owned(),
            op: ComparisonOp::Gt,
            value: ConditionValue::Float(0.5),
        });

        assert_eq!(recall_candidate_limit(10, &stmt, false), 120);
    }

    #[test]
    fn detect_temporal_positive_cases() {
        assert!(detect_temporal_in_query_text("What happened yesterday?"));
        assert!(detect_temporal_in_query_text("Tell me about last week's meeting"));
        assert!(detect_temporal_in_query_text("What did I do today?"));
        assert!(detect_temporal_in_query_text("Show me events from last month"));
        assert!(detect_temporal_in_query_text("What happened recently?"));
        assert!(detect_temporal_in_query_text("Tell me about things from days ago"));
        assert!(detect_temporal_in_query_text("YESTERDAY upper case")); // case-insensitive
    }

    #[test]
    fn detect_temporal_negative_cases() {
        assert!(!detect_temporal_in_query_text("What is quantum computing?"));
        assert!(!detect_temporal_in_query_text("List all my memories"));
        assert!(!detect_temporal_in_query_text(""));
    }

    #[test]
    fn derive_temporal_bounds_yesterday() {
        let now_ms: i64 = 1_700_000_000_000; // arbitrary epoch
        let (start, end) = derive_temporal_bounds_from_query_text("What happened yesterday?", now_ms).unwrap();
        assert!(start < end);
        assert!(end <= now_ms);
        // should span roughly 1 day ending before now
        let span = end - start;
        assert!(span >= 86_400_000 - 1000 && span <= 86_400_000 + 1000);
    }

    #[test]
    fn derive_temporal_bounds_last_week() {
        let now_ms: i64 = 1_700_000_000_000;
        let (start, end) = derive_temporal_bounds_from_query_text("last week's results", now_ms).unwrap();
        assert_eq!(end, now_ms);
        assert_eq!(now_ms - start, 7 * 86_400_000);
    }

    #[test]
    fn derive_temporal_bounds_no_match_returns_none() {
        let now_ms: i64 = 1_700_000_000_000;
        assert!(derive_temporal_bounds_from_query_text("Who is Ada Lovelace?", now_ms).is_none());
    }
}
