use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use hirn_core::id::MemoryId;
use hirn_core::record::MemoryRecord;
use hirn_core::types::{EdgeRelation, Namespace};
use hirn_core::{HirnError, HirnResult};
use hirn_exec::GraphCausalChainRow;
use hirn_exec::GraphReadRuntime;

use crate::db::HirnDB;
use crate::ql::results::{
    CausalQueryKind, CausalQueryResult, CausalRow, PolicyResult, QueryResult,
};

use super::ast::*;

#[async_trait]
pub(crate) trait CausalReadRuntime: Send + Sync {
    fn config(&self) -> &hirn_core::HirnConfig;

    fn graph_store(&self) -> &dyn crate::graph_store::GraphStore;

    fn graph_read_runtime(&self) -> &dyn GraphReadRuntime;

    async fn get_memories_batch(
        &self,
        ids: &[MemoryId],
    ) -> HirnResult<HashMap<MemoryId, MemoryRecord>>;
}

#[async_trait]
impl CausalReadRuntime for HirnDB {
    fn config(&self) -> &hirn_core::HirnConfig {
        self.config()
    }

    fn graph_store(&self) -> &dyn crate::graph_store::GraphStore {
        self.cached_graph()
    }

    fn graph_read_runtime(&self) -> &dyn GraphReadRuntime {
        self.cached_graph()
    }

    async fn get_memories_batch(
        &self,
        ids: &[MemoryId],
    ) -> HirnResult<HashMap<MemoryId, MemoryRecord>> {
        self.get_memories_batch(ids).await
    }
}

pub async fn execute_think_with_config(
    db: &HirnDB,
    stmt: &ThinkStmt,
    override_config: Option<super::context::ContextConfig>,
) -> HirnResult<QueryResult> {
    let query = Statement::Think(Box::new(stmt.clone())).to_string();
    db.execute_ql_with_think_context(&query, override_config.as_ref())
        .await
}

fn namespace_of_record(record: &MemoryRecord) -> Namespace {
    record.effective_namespace()
}

fn record_is_visible(record: &MemoryRecord, allowed_namespaces: Option<&[Namespace]>) -> bool {
    allowed_namespaces
        .is_none_or(|allowed_namespaces| allowed_namespaces.contains(&namespace_of_record(record)))
}

pub(crate) fn resolve_query_namespaces(
    requested_namespace: Option<Namespace>,
    allowed_namespaces: Option<&[Namespace]>,
) -> Option<Vec<Namespace>> {
    requested_namespace
        .map(|namespace| vec![namespace])
        .or_else(|| allowed_namespaces.map(|namespaces| namespaces.to_vec()))
}

fn namespace_is_visible(
    namespace: Option<&Namespace>,
    allowed_namespaces: Option<&[Namespace]>,
) -> bool {
    match allowed_namespaces {
        Some(allowed_namespaces) => {
            namespace.is_some_and(|namespace| allowed_namespaces.contains(namespace))
        }
        None => true,
    }
}

pub(crate) trait PolicyReadRuntime {
    fn policy_engine(&self) -> Option<&crate::policy::PolicyEngine>;
}

impl PolicyReadRuntime for HirnDB {
    fn policy_engine(&self) -> Option<&crate::policy::PolicyEngine> {
        self.policy_engine()
    }
}

pub(crate) fn execute_show_policies_with_runtime<R>(
    runtime: &R,
    stmt: &ShowPoliciesStmt,
) -> HirnResult<QueryResult>
where
    R: PolicyReadRuntime + ?Sized,
{
    let engine = runtime
        .policy_engine()
        .ok_or_else(|| HirnError::InvalidInput("no policy engine configured".into()))?;

    let all_policies = engine.list_policies();
    let policies: Vec<(String, String)> = if let Some(ref principal) = stmt.principal {
        let needle = match principal {
            PrincipalRef::Agent(a) => format!("Hirn::Agent::\"{}\"", a),
            PrincipalRef::Team(t) => format!("Hirn::Team::\"{}\"", t),
        };
        all_policies
            .into_iter()
            .filter(|(_, text)| text.contains(&needle))
            .collect()
    } else {
        all_policies
    };

    let count = policies.len();
    Ok(QueryResult::Policy(PolicyResult {
        message: format!("{count} polic{}", if count == 1 { "y" } else { "ies" }),
        policies,
    }))
}

pub(crate) fn execute_explain_policy_with_runtime<R>(
    runtime: &R,
    stmt: &ExplainPolicyStmt,
) -> HirnResult<QueryResult>
where
    R: PolicyReadRuntime + ?Sized,
{
    use crate::policy::{Action, AuthzRequest};

    let engine = runtime
        .policy_engine()
        .ok_or_else(|| HirnError::InvalidInput("no policy engine configured".into()))?;

    let action: Action = stmt.action.parse::<Action>().map_err(HirnError::from)?;

    let agent_id = match &stmt.principal {
        PrincipalRef::Agent(a) => a.clone(),
        PrincipalRef::Team(t) => t.clone(),
    };

    let (realm, namespace) = match stmt.resource_type.to_ascii_lowercase().as_str() {
        "realm" => (stmt.resource_name.clone(), String::new()),
        "namespace" => ("default".to_string(), stmt.resource_name.clone()),
        _ => {
            return Err(HirnError::InvalidInput(format!(
                "unsupported resource type '{}' (expected REALM or NAMESPACE)",
                stmt.resource_type
            )));
        }
    };

    let request = AuthzRequest {
        agent_id,
        action,
        realm,
        namespace,
    };

    let decision = engine.authorize(&request);

    let mut explanation = String::new();
    explanation.push_str(&format!(
        "Decision: {}\n",
        if decision.allowed { "ALLOW" } else { "DENY" }
    ));
    if !decision.policy_ids.is_empty() {
        explanation.push_str(&format!(
            "Determining policies: {}\n",
            decision.policy_ids.join(", ")
        ));
    }
    if !decision.reasons.is_empty() {
        explanation.push_str(&format!("Reasons: {}\n", decision.reasons.join("; ")));
    }
    if !decision.errors.is_empty() {
        explanation.push_str(&format!("Errors: {}\n", decision.errors.join("; ")));
    }

    Ok(QueryResult::Policy(PolicyResult {
        message: explanation,
        policies: Vec::new(),
    }))
}

pub(crate) fn record_matches_condition(record: &MemoryRecord, wc: &WhereCondition) -> bool {
    let field = wc.field.as_str();
    let val = match record {
        MemoryRecord::Episodic(e) => match field {
            "importance" => Some(e.importance as f64),
            "surprise" => Some(e.surprise as f64),
            "access_count" | "episodic.access_count" => Some(e.access_count as f64),
            "confidence" => Some(e.importance as f64),
            _ => None,
        },
        MemoryRecord::Semantic(s) => match field {
            "confidence" => Some(s.confidence as f64),
            "evidence_count" => Some(s.evidence_count as f64),
            "access_count" => Some(s.access_count as f64),
            "importance" => Some(s.confidence as f64),
            _ => None,
        },
        MemoryRecord::Working(w) => match field {
            "relevance_score" | "importance" => Some(w.relevance_score as f64),
            _ => None,
        },
        MemoryRecord::Procedural(p) => match field {
            "success_rate" | "importance" => Some(p.success_rate as f64),
            "invocation_count" => Some(p.invocation_count as f64),
            "access_count" => Some(p.access_count as f64),
            _ => None,
        },
    };

    let Some(record_val) = val else {
        return false;
    };

    let threshold = match &wc.value {
        ConditionValue::Float(v) => *v,
        ConditionValue::Int(v) => *v as f64,
        ConditionValue::String(_) | ConditionValue::Param(_) => return true,
    };

    match wc.op {
        ComparisonOp::Gt => record_val > threshold,
        ComparisonOp::Lt => record_val < threshold,
        ComparisonOp::Gte => record_val >= threshold,
        ComparisonOp::Lte => record_val <= threshold,
        ComparisonOp::Eq => (record_val - threshold).abs() < f64::EPSILON,
        ComparisonOp::Neq => (record_val - threshold).abs() >= f64::EPSILON,
    }
}

async fn load_causal_chain_rows(
    runtime: &dyn CausalReadRuntime,
    start_ids: &[MemoryId],
    max_depth: u32,
    relation: EdgeRelation,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<Vec<GraphCausalChainRow>> {
    runtime
        .graph_read_runtime()
        .causal_chain(
            start_ids,
            max_depth,
            0.0,
            runtime.config().graph_depth_delegation_threshold,
            relation,
            allowed_namespaces,
        )
        .await
}

fn group_causal_chain_rows(rows: Vec<GraphCausalChainRow>) -> Vec<Vec<GraphCausalChainRow>> {
    let mut grouped: HashMap<String, Vec<GraphCausalChainRow>> = HashMap::new();
    for row in rows {
        grouped.entry(row.chain_id.clone()).or_default().push(row);
    }

    let mut chains = grouped.into_values().collect::<Vec<_>>();
    for chain in &mut chains {
        chain.sort_by_key(|row| row.depth);
    }
    chains
}

fn parse_causal_row_id(id: &str, label: &str) -> HirnResult<MemoryId> {
    MemoryId::parse(id).map_err(|error| {
        HirnError::InvalidInput(format!(
            "graph runtime returned invalid causal {label} id `{id}`: {error}"
        ))
    })
}

fn causal_row_source_id(row: &GraphCausalChainRow) -> HirnResult<MemoryId> {
    parse_causal_row_id(&row.source_id, "source")
}

fn causal_row_target_id(row: &GraphCausalChainRow) -> HirnResult<MemoryId> {
    parse_causal_row_id(&row.target_id, "target")
}

async fn find_graph_nodes_by_content(
    runtime: &dyn CausalReadRuntime,
    query: &str,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<Vec<(MemoryId, String)>> {
    let all_ids = runtime.graph_store().node_ids().await?;
    if all_ids.is_empty() {
        return Ok(Vec::new());
    }

    let records = runtime.get_memories_batch(&all_ids).await?;
    let query_lower = query.to_lowercase();
    let mut matches = Vec::new();
    for (id, rec) in &records {
        if !record_is_visible(rec, allowed_namespaces) {
            continue;
        }
        let content = crate::causal::record_content_str(rec);
        if content.to_lowercase().contains(&query_lower) {
            matches.push((*id, content.to_string()));
        }
    }
    Ok(matches)
}

fn chain_probability(chain: &[GraphCausalChainRow]) -> f32 {
    chain
        .iter()
        .map(|row| row.strength * row.confidence)
        .product()
}

fn mechanism_path(chain: &[GraphCausalChainRow]) -> String {
    chain
        .iter()
        .filter_map(|row| row.mechanism.as_deref())
        .filter(|m| !m.is_empty())
        .collect::<Vec<_>>()
        .join(" → ")
}

fn supporting_evidence(chain: &[GraphCausalChainRow]) -> u32 {
    chain.iter().map(|row| row.evidence_count.max(1)).sum()
}

fn chain_reaches_target(
    chain: &[GraphCausalChainRow],
    target_ids: &HashSet<MemoryId>,
) -> HirnResult<bool> {
    for row in chain {
        if target_ids.contains(&causal_row_target_id(row)?) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn chain_mentions_any(chain: &[GraphCausalChainRow], ids: &HashSet<MemoryId>) -> HirnResult<bool> {
    for row in chain {
        if ids.contains(&causal_row_source_id(row)?) || ids.contains(&causal_row_target_id(row)?) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) async fn execute_explain_causes_with_runtime(
    runtime: &dyn CausalReadRuntime,
    stmt: &ExplainCausesStmt,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<CausalQueryResult> {
    let start = std::time::Instant::now();
    const MAX_CAUSAL_DEPTH: usize = 20;
    let depth = (stmt.depth.unwrap_or(3) as usize).min(MAX_CAUSAL_DEPTH);
    let requested_namespace = stmt
        .namespace
        .as_ref()
        .and_then(|value| Namespace::new(value).ok());
    let allowed_query_namespaces =
        resolve_query_namespaces(requested_namespace, allowed_namespaces);

    let target_nodes =
        find_graph_nodes_by_content(runtime, &stmt.target, allowed_query_namespaces.as_deref())
            .await?;
    let target_ids: Vec<MemoryId> = target_nodes.iter().map(|(id, _)| *id).collect();
    let chain_rows = runtime
        .graph_read_runtime()
        .causal_chain(
            &target_ids,
            depth as u32,
            0.0,
            runtime.config().graph_depth_delegation_threshold,
            EdgeRelation::CausedBy,
            allowed_query_namespaces.as_deref(),
        )
        .await?;

    let parsed_rows = chain_rows
        .into_iter()
        .map(|row| {
            let target_id = row.target_id.clone();
            MemoryId::parse(&target_id)
                .map(|cause_id| (cause_id, row))
                .map_err(|error| {
                    HirnError::InvalidInput(format!(
                        "graph runtime returned invalid causal target id `{}`: {error}",
                        target_id
                    ))
                })
        })
        .collect::<HirnResult<Vec<_>>>()?;

    let cause_ids: Vec<MemoryId> = parsed_rows.iter().map(|(cause_id, _)| *cause_id).collect();
    let cause_records = if cause_ids.is_empty() {
        std::collections::HashMap::new()
    } else {
        runtime.get_memories_batch(&cause_ids).await?
    };

    let mut rows = Vec::new();
    let mut seen_causes = HashSet::new();
    for (cause_id, row) in parsed_rows {
        if !seen_causes.insert(cause_id) {
            continue;
        }
        if !namespace_is_visible(
            cause_records
                .get(&cause_id)
                .map(|record| record.effective_namespace())
                .as_ref(),
            allowed_query_namespaces.as_deref(),
        ) {
            continue;
        }
        let cause_content = cause_records
            .get(&cause_id)
            .map(|record| crate::causal::record_content_str(record).to_string())
            .unwrap_or_default();
        rows.push(CausalRow {
            columns: vec![
                ("cause_id".into(), cause_id.to_string()),
                ("cause_content".into(), cause_content),
                ("depth".into(), (row.depth + 1).to_string()),
                ("edge_strength".into(), format!("{:.3}", row.strength)),
                ("edge_confidence".into(), format!("{:.3}", row.confidence)),
                (
                    "mechanism".into(),
                    row.mechanism.clone().unwrap_or_default(),
                ),
                ("chain_score".into(), format!("{:.4}", row.chain_score)),
            ],
        });
    }

    rows.sort_by(|a, b| {
        let score_a = a
            .columns
            .iter()
            .find(|(k, _)| k == "chain_score")
            .map(|(_, v)| v.parse::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);
        let score_b = b
            .columns
            .iter()
            .find(|(k, _)| k == "chain_score")
            .map(|(_, v)| v.parse::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);
        score_b.total_cmp(&score_a)
    });

    Ok(CausalQueryResult {
        kind: CausalQueryKind::ExplainCauses,
        rows,
        query_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

pub(crate) async fn execute_what_if_with_runtime(
    runtime: &dyn CausalReadRuntime,
    stmt: &WhatIfStmt,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<CausalQueryResult> {
    let start = std::time::Instant::now();
    let requested_namespace = stmt
        .namespace
        .as_ref()
        .and_then(|value| Namespace::new(value).ok());
    let allowed_query_namespaces =
        resolve_query_namespaces(requested_namespace, allowed_namespaces);

    let intervention_nodes = find_graph_nodes_by_content(
        runtime,
        &stmt.intervention,
        allowed_query_namespaces.as_deref(),
    )
    .await?;
    let outcome_nodes =
        find_graph_nodes_by_content(runtime, &stmt.outcome, allowed_query_namespaces.as_deref())
            .await?;

    let outcome_ids: HashSet<MemoryId> = outcome_nodes.iter().map(|(id, _)| *id).collect();
    let mut rows = Vec::new();

    for (int_id, _) in &intervention_nodes {
        let chains = group_causal_chain_rows(
            load_causal_chain_rows(
                runtime,
                &[*int_id],
                5,
                EdgeRelation::Causes,
                allowed_query_namespaces.as_deref(),
            )
            .await?,
        );

        let mut best_prob = 0.0_f32;
        let mut best_path = String::new();
        let mut best_evidence = 0_u32;

        for chain in &chains {
            if !chain_reaches_target(chain, &outcome_ids)? {
                continue;
            }

            let prob = chain_probability(chain);
            if prob > best_prob {
                best_prob = prob;
                best_path = mechanism_path(chain);
                best_evidence = supporting_evidence(chain);
            }
        }

        rows.push(CausalRow {
            columns: vec![
                ("outcome".into(), stmt.outcome.clone()),
                ("probability".into(), format!("{:.4}", best_prob)),
                ("path".into(), best_path),
                ("supporting_evidence".into(), best_evidence.to_string()),
            ],
        });
    }

    if rows.is_empty() {
        rows.push(CausalRow {
            columns: vec![
                ("outcome".into(), stmt.outcome.clone()),
                ("probability".into(), "0.0000".into()),
                ("path".into(), String::new()),
                ("supporting_evidence".into(), "0".into()),
            ],
        });
    }

    rows.sort_by(|a, b| {
        let prob_a = a
            .columns
            .iter()
            .find(|(k, _)| k == "probability")
            .map(|(_, v)| v.parse::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);
        let prob_b = b
            .columns
            .iter()
            .find(|(k, _)| k == "probability")
            .map(|(_, v)| v.parse::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0);
        prob_b.total_cmp(&prob_a)
    });

    Ok(CausalQueryResult {
        kind: CausalQueryKind::WhatIf,
        rows,
        query_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

pub(crate) async fn execute_counterfactual_with_runtime(
    runtime: &dyn CausalReadRuntime,
    stmt: &CounterfactualStmt,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<CausalQueryResult> {
    let start = std::time::Instant::now();
    let requested_namespace = stmt
        .namespace
        .as_ref()
        .and_then(|value| Namespace::new(value).ok());
    let allowed_query_namespaces =
        resolve_query_namespaces(requested_namespace, allowed_namespaces);

    let antecedent_nodes = find_graph_nodes_by_content(
        runtime,
        &stmt.antecedent,
        allowed_query_namespaces.as_deref(),
    )
    .await?;
    let consequent_nodes = find_graph_nodes_by_content(
        runtime,
        &stmt.consequent,
        allowed_query_namespaces.as_deref(),
    )
    .await?;

    let antecedent_ids: HashSet<MemoryId> = antecedent_nodes.iter().map(|(id, _)| *id).collect();
    let mut rows = Vec::new();

    for (con_id, _) in &consequent_nodes {
        let factual_chains = group_causal_chain_rows(
            load_causal_chain_rows(
                runtime,
                &[*con_id],
                5,
                EdgeRelation::CausedBy,
                allowed_query_namespaces.as_deref(),
            )
            .await?,
        );

        let factual_prob: f32 = factual_chains
            .iter()
            .map(|c| chain_probability(c))
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);

        let cf_chains: Vec<_> = factual_chains
            .iter()
            .filter(|chain| !chain_mentions_any(chain, &antecedent_ids).unwrap_or(false))
            .collect();

        let cf_prob: f32 = cf_chains
            .iter()
            .map(|c| chain_probability(c))
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0);

        let necessity = if factual_prob > 0.0 {
            (1.0 - (cf_prob / factual_prob)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        rows.push(CausalRow {
            columns: vec![
                ("consequent".into(), stmt.consequent.clone()),
                ("factual_probability".into(), format!("{:.4}", factual_prob)),
                (
                    "counterfactual_probability".into(),
                    format!("{:.4}", cf_prob),
                ),
                ("necessity_score".into(), format!("{:.4}", necessity)),
                ("alternative_paths".into(), cf_chains.len().to_string()),
            ],
        });
    }

    if rows.is_empty() {
        rows.push(CausalRow {
            columns: vec![
                ("consequent".into(), stmt.consequent.clone()),
                ("factual_probability".into(), "0.0000".into()),
                ("counterfactual_probability".into(), "0.0000".into()),
                ("necessity_score".into(), "0.0000".into()),
                ("alternative_paths".into(), "0".into()),
            ],
        });
    }

    Ok(CausalQueryResult {
        kind: CausalQueryKind::Counterfactual,
        rows,
        query_time_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use hirn_core::HirnConfig;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::{AgentId, EdgeRelation, EventType};
    use hirn_storage::memory_store::MemoryStore;

    use crate::graph_store::GraphStore;

    fn agent() -> AgentId {
        AgentId::new("ql-direct-support-tests").unwrap()
    }

    async fn temp_db() -> (HirnDB, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ql-direct-support-tests");
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(4)
            .graph_depth_delegation_threshold(1)
            .working_memory_token_limit(1000)
            .build()
            .unwrap();
        let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
            .await
            .unwrap();
        (db, dir)
    }

    fn causal_value<'a>(row: &'a CausalRow, key: &str) -> Option<&'a str> {
        row.columns
            .iter()
            .find(|(column, _)| column == key)
            .map(|(_, value)| value.as_str())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn explain_causes_keeps_cause_orientation_across_delegation() {
        let (db, _dir) = temp_db().await;

        let cause_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("root cause event")
                    .summary("root cause event")
                    .embedding(vec![0.0, 1.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let effect_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("effect event")
                    .summary("effect event")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.cached_graph()
            .add_edge(
                effect_id,
                cause_id,
                EdgeRelation::CausedBy,
                0.9,
                Metadata::new(),
            )
            .await
            .unwrap();

        let hot = execute_explain_causes_with_runtime(
            &db,
            &ExplainCausesStmt {
                target: "effect event".to_owned(),
                namespace: None,
                depth: Some(1),
            },
            None,
        )
        .await
        .unwrap();
        let cold = execute_explain_causes_with_runtime(
            &db,
            &ExplainCausesStmt {
                target: "effect event".to_owned(),
                namespace: None,
                depth: Some(6),
            },
            None,
        )
        .await
        .unwrap();

        let hot_rows = hot.rows;
        let cold_rows = cold.rows;

        let expected = cause_id.to_string();
        assert_eq!(hot_rows.len(), 1);
        assert_eq!(cold_rows.len(), 1);
        assert_eq!(
            causal_value(&hot_rows[0], "cause_id"),
            Some(expected.as_str())
        );
        assert_eq!(
            causal_value(&cold_rows[0], "cause_id"),
            Some(expected.as_str())
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn traverse_uses_all_via_relations_across_delegation() {
        let (db, _dir) = temp_db().await;

        let start_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("start event")
                    .summary("start event")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let causes_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("causal neighbor")
                    .summary("causal neighbor")
                    .embedding(vec![0.0, 1.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let related_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("related neighbor")
                    .summary("related neighbor")
                    .embedding(vec![0.0, 0.0, 1.0, 0.0])
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.persistent_graph()
            .add_edge(
                start_id,
                causes_id,
                EdgeRelation::Causes,
                0.8,
                Metadata::new(),
            )
            .await
            .unwrap();
        db.persistent_graph()
            .add_edge(
                start_id,
                related_id,
                EdgeRelation::RelatedTo,
                0.8,
                Metadata::new(),
            )
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(&format!(
                r#"TRAVERSE FROM "{start_id}" VIA causes,relatedto DEPTH 2"#
            ))
            .await
            .unwrap();

        let records = match result {
            QueryResult::Records(result) => result.records,
            other => panic!("expected records result, got {other:?}"),
        };

        let ids = records
            .iter()
            .map(|scored| scored.record.id().to_string())
            .collect::<HashSet<_>>();

        assert!(ids.contains(&causes_id.to_string()));
        assert!(ids.contains(&related_id.to_string()));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn what_if_uses_delegated_causal_runtime() {
        let (db, _dir) = temp_db().await;

        let cause_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("root cause event")
                    .summary("root cause event")
                    .embedding(vec![0.0, 1.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let effect_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("effect event")
                    .summary("effect event")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.persistent_graph()
            .add_edge(
                cause_id,
                effect_id,
                EdgeRelation::Causes,
                0.8,
                Metadata::new(),
            )
            .await
            .unwrap();

        let result = execute_what_if_with_runtime(
            &db,
            &WhatIfStmt {
                intervention: "root cause event".to_owned(),
                outcome: "effect event".to_owned(),
                namespace: None,
            },
            None,
        )
        .await
        .unwrap();

        let rows = result.rows;

        assert_eq!(rows.len(), 1);
        assert_eq!(causal_value(&rows[0], "probability"), Some("0.4000"));
        assert_eq!(causal_value(&rows[0], "supporting_evidence"), Some("1"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn counterfactual_uses_delegated_causal_runtime() {
        let (db, _dir) = temp_db().await;

        let antecedent_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("root cause event")
                    .summary("root cause event")
                    .embedding(vec![0.0, 1.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let consequent_id = db
            .remember(
                EpisodicRecord::builder()
                    .event_type(EventType::Observation)
                    .content("effect event")
                    .summary("effect event")
                    .embedding(vec![1.0, 0.0, 0.0, 0.0])
                    .importance(0.9)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.persistent_graph()
            .add_edge(
                consequent_id,
                antecedent_id,
                EdgeRelation::CausedBy,
                0.8,
                Metadata::new(),
            )
            .await
            .unwrap();

        let result = execute_counterfactual_with_runtime(
            &db,
            &CounterfactualStmt {
                antecedent: "root cause event".to_owned(),
                consequent: "effect event".to_owned(),
                namespace: None,
            },
            None,
        )
        .await
        .unwrap();

        let rows = result.rows;

        assert_eq!(rows.len(), 1);
        assert_eq!(
            causal_value(&rows[0], "factual_probability"),
            Some("0.4000")
        );
        assert_eq!(
            causal_value(&rows[0], "counterfactual_probability"),
            Some("0.0000")
        );
        assert_eq!(causal_value(&rows[0], "necessity_score"), Some("1.0000"));
    }
}
