//! HirnQL parser — transforms query text into AST.

use pest::Parser;
use pest_derive::Parser;

use hirn_core::types::Layer;

use super::ast::*;

// ── Pest parser definition ─────────────────────────────────────────────

#[derive(Parser)]
#[grammar = "parser/hirnql.pest"]
struct HirnQlParser;

// ── Public API ─────────────────────────────────────────────────────────

/// Parse error with location and context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "parse error at {}:{}: {}",
            self.line, self.column, self.message
        )
    }
}

impl std::error::Error for ParseError {}

impl ParseError {
    /// Shorthand for errors without precise location (line/column default to 1).
    fn simple(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            line: 1,
            column: 1,
        }
    }
}

/// Configurable limits for query parsing.
#[derive(Debug, Clone)]
pub struct QueryLimits {
    /// Maximum query string length in bytes (default: 1 MB).
    pub max_query_length: usize,
    /// Maximum EXPAND GRAPH DEPTH value (default: 10).
    pub max_expand_depth: usize,
    /// Maximum LIMIT value (default: 10,000).
    pub max_limit: usize,
    /// Maximum BUDGET (context token budget) value (default: 1,000,000).
    ///
    /// Prevents runaway context budgets that could exhaust memory.
    pub max_context_budget: usize,
    /// Maximum MODE ITERATIVE MAX_HOPS value (default: 5).
    ///
    /// Prevents clients from escalating beyond the validated iterative-retrieval
    /// hop ceiling — enforces the same cap regardless of operator-level default.
    pub max_iterative_hops: usize,
}

impl Default for QueryLimits {
    fn default() -> Self {
        Self {
            max_query_length: 1_048_576, // 1 MB
            max_expand_depth: 10,
            max_limit: 10_000,
            max_context_budget: 1_000_000,
            max_iterative_hops: 5,
        }
    }
}

/// Parse a HirnQL query string into a `Statement`.
pub fn parse(input: &str) -> Result<Statement, ParseError> {
    parse_with_limits(input, &QueryLimits::default())
}

/// Parse a HirnQL query string with configurable limits.
pub fn parse_with_limits(input: &str, limits: &QueryLimits) -> Result<Statement, ParseError> {
    if input.len() > limits.max_query_length {
        return Err(ParseError::simple(format!(
            "query too large: {} bytes exceeds maximum of {} bytes",
            input.len(),
            limits.max_query_length
        )));
    }

    let pairs = HirnQlParser::parse(Rule::statement, input).map_err(|e| {
        let (line, col) = match e.line_col {
            pest::error::LineColLocation::Pos((l, c)) => (l, c),
            pest::error::LineColLocation::Span((l, c), _) => (l, c),
        };

        let msg = format_pest_error(&e, input);
        ParseError {
            message: msg,
            line,
            column: col,
        }
    })?;

    let statement_pair = pairs
        .into_iter()
        .next()
        .ok_or_else(|| ParseError::simple("empty input"))?;

    let stmt = build_statement(statement_pair)?;
    validate_limits(&stmt, limits)?;
    Ok(stmt)
}

/// Validate parsed statement against configured limits.
fn validate_limits(stmt: &Statement, limits: &QueryLimits) -> Result<(), ParseError> {
    match stmt {
        Statement::Recall(r) => {
            if let Some(limit) = r.limit {
                check_limit(limit, limits.max_limit)?;
            }
            if let Some(budget) = r.budget {
                check_budget(budget, limits.max_context_budget)?;
            }
            if let Some(ref expand) = r.expand {
                check_depth(expand.depth, limits.max_expand_depth)?;
            }
        }
        Statement::Think(t) => {
            if let Some(limit) = t.limit {
                check_limit(limit, limits.max_limit)?;
            }
            if let Some(budget) = t.budget {
                check_budget(budget, limits.max_context_budget)?;
            }
            if let Some(hops) = t.max_hops {
                check_max_hops(hops, limits.max_iterative_hops)?;
            }
            if let Some(ref expand) = t.expand {
                check_depth(expand.depth, limits.max_expand_depth)?;
            }
        }
        Statement::RecallEvents(r) => {
            if let Some(limit) = r.limit {
                check_limit(limit, limits.max_limit)?;
            }
        }
        Statement::Traverse(t) => {
            check_depth(t.depth, limits.max_expand_depth)?;
            if let Some(limit) = t.limit {
                check_limit(limit, limits.max_limit)?;
            }
        }
        Statement::Explain(e) => validate_limits(&e.inner, limits)?,
        _ => {}
    }
    Ok(())
}

fn check_limit(value: usize, max: usize) -> Result<(), ParseError> {
    if value > max {
        return Err(ParseError::simple(format!(
            "LIMIT {value} exceeds maximum allowed value of {max}"
        )));
    }
    Ok(())
}

fn check_depth(value: usize, max: usize) -> Result<(), ParseError> {
    if value > max {
        return Err(ParseError::simple(format!(
            "DEPTH {value} exceeds maximum allowed value of {max}"
        )));
    }
    Ok(())
}

fn check_budget(value: usize, max: usize) -> Result<(), ParseError> {
    if value > max {
        return Err(ParseError::simple(format!(
            "BUDGET {value} exceeds maximum allowed value of {max}"
        )));
    }
    Ok(())
}

fn check_max_hops(value: usize, max: usize) -> Result<(), ParseError> {
    if value > max {
        return Err(ParseError::simple(format!(
            "MAX_HOPS {value} exceeds maximum allowed value of {max}"
        )));
    }
    Ok(())
}

/// Format pest errors into user-friendly messages.
fn format_pest_error(e: &pest::error::Error<Rule>, input: &str) -> String {
    // Detect common error patterns and provide helpful suggestions.
    let base = e.variant.message().to_string();

    let trimmed = input.trim();
    if let Some(first_word) = trimmed.split_whitespace().next() {
        let upper = first_word.to_uppercase();
        let known = [
            "RECALL",
            "THINK",
            "REMEMBER",
            "FORGET",
            "CORRECT",
            "SUPERSEDE",
            "RETRACT",
            "CONNECT",
            "INSPECT",
            "HISTORY",
            "TRACE",
            "CONSOLIDATE",
            "WATCH",
            "TRAVERSE",
            "EXPLAIN",
            "CREATE",
            "DROP",
            "GRANT",
            "REVOKE",
            "SHOW",
        ];
        if !known.contains(&upper.as_str()) {
            return format!("unknown verb '{first_word}', did you mean 'RECALL'?");
        }
    }

    base
}

// ── AST construction ───────────────────────────────────────────────────

fn unsupported_embedded_statement(rule: Rule) -> Result<Statement, ParseError> {
    Err(ParseError::simple(match rule {
        Rule::remember_stmt => {
            "REMEMBER is not supported via embedded HirnQL anymore; use the direct memory view APIs instead"
        }
        Rule::forget_stmt => {
            "FORGET is not supported via embedded HirnQL anymore; use the direct memory view APIs instead"
        }
        Rule::connect_stmt => {
            "CONNECT is not supported via embedded HirnQL anymore; use the graph view APIs instead"
        }
        Rule::consolidate_stmt => {
            "CONSOLIDATE is not supported via HirnQL anymore; use db.admin().consolidate().execute() instead"
        }
        Rule::watch_stmt => {
            "WATCH is not supported via embedded HirnQL anymore; use the event or daemon APIs instead"
        }
        _ => "statement is not supported via embedded HirnQL anymore",
    }))
}

fn build_statement(pair: pest::iterators::Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::simple("empty statement"))?;

    match inner.as_rule() {
        Rule::recall_events_stmt => Ok(Statement::RecallEvents(build_recall_events(inner)?)),
        Rule::recall_stmt => build_recall(inner).map(|stmt| Statement::Recall(Box::new(stmt))),
        Rule::think_stmt => build_think(inner).map(|stmt| Statement::Think(Box::new(stmt))),
        Rule::remember_stmt => unsupported_embedded_statement(Rule::remember_stmt),
        Rule::forget_stmt => unsupported_embedded_statement(Rule::forget_stmt),
        Rule::correct_stmt => build_correct(inner).map(Statement::Correct),
        Rule::supersede_stmt => build_supersede(inner).map(Statement::Supersede),
        Rule::merge_memory_stmt => build_merge_memory(inner).map(Statement::MergeMemory),
        Rule::retract_stmt => build_retract(inner).map(Statement::Retract),
        Rule::connect_stmt => unsupported_embedded_statement(Rule::connect_stmt),
        Rule::inspect_stmt => Ok(Statement::Inspect(build_inspect(inner)?)),
        Rule::history_stmt => Ok(Statement::History(build_history(inner)?)),
        Rule::trace_stmt => Ok(Statement::Trace(build_trace(inner)?)),
        Rule::consolidate_stmt => unsupported_embedded_statement(Rule::consolidate_stmt),
        Rule::watch_stmt => unsupported_embedded_statement(Rule::watch_stmt),
        Rule::traverse_stmt => build_traverse(inner).map(Statement::Traverse),
        Rule::explain_stmt => build_explain(inner),
        Rule::explain_causes_stmt => build_explain_causes(inner).map(Statement::ExplainCauses),
        Rule::what_if_stmt => build_what_if(inner).map(Statement::WhatIf),
        Rule::counterfactual_stmt => build_counterfactual(inner).map(Statement::Counterfactual),
        Rule::create_realm_stmt => Ok(Statement::CreateRealm(build_create_realm(inner)?)),
        Rule::drop_realm_stmt => Ok(Statement::DropRealm(build_drop_realm(inner)?)),
        Rule::grant_stmt => build_grant(inner).map(Statement::Grant),
        Rule::revoke_stmt => build_revoke(inner).map(Statement::Revoke),
        Rule::show_policies_stmt => Ok(Statement::ShowPolicies(build_show_policies(inner)?)),
        Rule::explain_policy_stmt => build_explain_policy(inner).map(Statement::ExplainPolicy),
        Rule::show_cluster_stmt => Ok(Statement::ShowCluster),
        Rule::set_tier_policy_stmt => Ok(Statement::SetTierPolicy(build_set_tier_policy(inner)?)),
        _ => Err(ParseError::simple(format!(
            "unexpected rule: {:?}",
            inner.as_rule()
        ))),
    }
}

fn build_recall(pair: pest::iterators::Pair<'_, Rule>) -> Result<RecallStmt, ParseError> {
    let mut stmt = RecallStmt {
        layers: vec![],
        about: String::new(),
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
        limit: None,
        hybrid: false,
    };

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::layer_filter => stmt.layers = build_layer_filter(inner),
            Rule::about_clause => stmt.about = extract_about(inner)?,
            Rule::involving_clause => stmt.involving = Some(extract_string_list(inner)?),
            Rule::temporal_clause => stmt.temporal = Some(build_temporal(inner)?),
            Rule::as_of_clause => stmt.as_of = Some(build_as_of(inner)?),
            Rule::expand_clause => stmt.expand = Some(build_expand(inner)?),
            Rule::follow_causes_clause => stmt.follow_causes = Some(extract_follow_causes(inner)?),
            Rule::where_clause => {
                // where_clause can contain either a condition or in_subquery_condition
                let child = inner
                    .into_inner()
                    .next()
                    .ok_or_else(|| ParseError::simple("empty WHERE clause"))?;
                match child.as_rule() {
                    Rule::in_subquery_condition => {
                        stmt.subquery_filters.push(build_in_subquery(child)?);
                    }
                    Rule::condition => {
                        stmt.where_clauses.push(build_condition(child)?);
                    }
                    _ => {}
                }
            }
            Rule::group_by_clause => stmt.group_by = Some(build_group_by(inner)),
            Rule::select_clause => stmt.projection = Some(build_field_list(inner)),
            Rule::as_clause => stmt.output_format = Some(build_output_format(inner)),
            Rule::format_clause => stmt.result_format = Some(build_format_clause(inner)),
            Rule::budget_clause => stmt.budget = Some(extract_budget(inner)?),
            Rule::namespace_clause => stmt.namespace = Some(extract_namespace(inner)),
            Rule::from_realm_clause => stmt.from_realms = Some(extract_realm_list(inner)),
            Rule::consistency_clause => stmt.consistency = Some(build_consistency(inner)),
            Rule::limit_clause => stmt.limit = Some(extract_limit(inner)?),
            Rule::modality_clause => stmt.modality = Some(build_modality_list(inner)?),
            Rule::resource_role_clause => {
                stmt.resource_roles = Some(build_evidence_role_list(inner)?);
            }
            Rule::hydration_clause => {
                stmt.hydration_modes = Some(build_hydration_mode_list(inner)?);
            }
            Rule::artifact_clause => {
                stmt.artifact_kinds = Some(build_artifact_kind_list(inner)?);
            }
            Rule::depth_clause => stmt.depth_mode = Some(build_depth_mode(inner)?),
            Rule::topic_clause => stmt.topic = Some(extract_string_from_clause(inner)?),
            Rule::with_prospective_clause => stmt.with_prospective = Some(build_on_off(inner)?),
            Rule::with_mcfa_clause => stmt.with_mcfa = Some(build_on_off(inner)?),
            Rule::with_conflicts_clause => stmt.with_conflicts = true,
            Rule::with_provenance_clause => {
                stmt.provenance_depth = Some(extract_integer_from_clause(inner)?);
            }
            Rule::hybrid_clause => stmt.hybrid = true,
            _ => {}
        }
    }

    if stmt.about.is_empty() {
        return Err(ParseError::simple("RECALL requires ABOUT clause"));
    }

    Ok(stmt)
}

fn build_recall_events(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<RecallEventsStmt, ParseError> {
    let mut stmt = RecallEventsStmt {
        entity_filter: None,
        where_clauses: vec![],
        temporal: None,
        namespace: None,
        limit: None,
    };

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::events_for_clause => {
                stmt.entity_filter = Some(extract_string_from_clause(inner)?);
            }
            Rule::where_clause => stmt.where_clauses.push(build_where(inner)?),
            Rule::temporal_clause => stmt.temporal = Some(build_temporal(inner)?),
            Rule::namespace_clause => stmt.namespace = Some(extract_namespace(inner)),
            Rule::limit_clause => stmt.limit = Some(extract_limit(inner)?),
            _ => {}
        }
    }

    Ok(stmt)
}

fn build_think(pair: pest::iterators::Pair<'_, Rule>) -> Result<ThinkStmt, ParseError> {
    let mut stmt = ThinkStmt {
        about: String::new(),
        involving: None,
        temporal: None,
        expand: None,
        follow_causes: None,
        where_clauses: vec![],
        output_format: None,
        budget: None,
        namespace: None,
        consistency: None,
        limit: None,
        hybrid: false,
        mode: RetrievalMode::Local,
        depth_mode: None,
        with_prospective: None,
        with_mcfa: None,
        provenance_depth: None,
        max_hops: None,
        community_depth: None,
    };

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::about_clause => stmt.about = extract_about(inner)?,
            Rule::involving_clause => stmt.involving = Some(extract_string_list(inner)?),
            Rule::temporal_clause => stmt.temporal = Some(build_temporal(inner)?),
            Rule::expand_clause => stmt.expand = Some(build_expand(inner)?),
            Rule::follow_causes_clause => stmt.follow_causes = Some(extract_follow_causes(inner)?),
            Rule::where_clause => stmt.where_clauses.push(build_where(inner)?),
            Rule::as_clause => stmt.output_format = Some(build_output_format(inner)),
            Rule::budget_clause => stmt.budget = Some(extract_budget(inner)?),
            Rule::namespace_clause => stmt.namespace = Some(extract_namespace(inner)),
            Rule::consistency_clause => stmt.consistency = Some(build_consistency(inner)),
            Rule::limit_clause => stmt.limit = Some(extract_limit(inner)?),
            Rule::global_clause => stmt.mode = RetrievalMode::Global,
            Rule::mode_clause => {
                let (mode, max_hops) = build_retrieval_mode_with_hops(inner)?;
                stmt.mode = mode;
                if max_hops.is_some() {
                    stmt.max_hops = max_hops;
                }
            }
            Rule::hybrid_clause => stmt.hybrid = true,
            Rule::depth_clause => stmt.depth_mode = Some(build_depth_mode(inner)?),
            Rule::with_prospective_clause => stmt.with_prospective = Some(build_on_off(inner)?),
            Rule::with_mcfa_clause => stmt.with_mcfa = Some(build_on_off(inner)?),
            Rule::with_provenance_clause => {
                stmt.provenance_depth = Some(extract_integer_from_clause(inner)?);
            }
            Rule::community_depth_clause => {
                stmt.community_depth = Some(extract_integer_from_clause(inner)?);
            }
            _ => {}
        }
    }

    Ok(stmt)
}

fn build_correct(pair: pest::iterators::Pair<'_, Rule>) -> Result<CorrectStmt, ParseError> {
    let mut target = None;
    let mut updates = Vec::new();
    let mut reason = None;
    let mut observed_at = None;
    let mut caused_by = None;
    let mut namespace = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::semantic_target_ref if target.is_none() => {
                target = Some(build_semantic_target_ref(inner)?);
            }
            Rule::set_assignment_list => updates = build_set_assignment_list(inner)?,
            Rule::reason_clause => reason = Some(extract_string_from_clause(inner)?),
            Rule::observed_at_clause => observed_at = Some(extract_string_from_clause(inner)?),
            Rule::caused_by_clause => caused_by = Some(extract_string_from_clause(inner)?),
            Rule::namespace_clause => namespace = Some(extract_namespace(inner)),
            _ => {}
        }
    }

    Ok(CorrectStmt {
        target: target.unwrap_or_else(|| SemanticTargetRef::Memory(String::new())),
        updates,
        reason,
        observed_at,
        caused_by,
        namespace,
    })
}

fn build_supersede(pair: pest::iterators::Pair<'_, Rule>) -> Result<SupersedeStmt, ParseError> {
    let mut target = None;
    let mut updates = Vec::new();
    let mut reason = None;
    let mut observed_at = None;
    let mut caused_by = None;
    let mut namespace = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::semantic_target_ref if target.is_none() => {
                target = Some(build_semantic_target_ref(inner)?);
            }
            Rule::set_assignment_list => updates = build_set_assignment_list(inner)?,
            Rule::reason_clause => reason = Some(extract_string_from_clause(inner)?),
            Rule::observed_at_clause => observed_at = Some(extract_string_from_clause(inner)?),
            Rule::caused_by_clause => caused_by = Some(extract_string_from_clause(inner)?),
            Rule::namespace_clause => namespace = Some(extract_namespace(inner)),
            _ => {}
        }
    }

    Ok(SupersedeStmt {
        target: target.unwrap_or_else(|| SemanticTargetRef::Memory(String::new())),
        updates,
        reason,
        observed_at,
        caused_by,
        namespace,
    })
}

fn build_merge_memory(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<MergeMemoryStmt, ParseError> {
    let mut sources = Vec::new();
    let mut target = None;
    let mut updates = Vec::new();
    let mut reason = None;
    let mut observed_at = None;
    let mut caused_by = None;
    let mut namespace = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::semantic_target_list if sources.is_empty() => {
                sources = build_semantic_target_list(inner)?;
            }
            Rule::semantic_target_ref if target.is_none() => {
                target = Some(build_semantic_target_ref(inner)?);
            }
            Rule::merge_set_clause => {
                for child in inner.into_inner() {
                    if child.as_rule() == Rule::set_assignment_list {
                        updates = build_set_assignment_list(child)?;
                    }
                }
            }
            Rule::reason_clause => reason = Some(extract_string_from_clause(inner)?),
            Rule::observed_at_clause => observed_at = Some(extract_string_from_clause(inner)?),
            Rule::caused_by_clause => caused_by = Some(extract_string_from_clause(inner)?),
            Rule::namespace_clause => namespace = Some(extract_namespace(inner)),
            _ => {}
        }
    }

    Ok(MergeMemoryStmt {
        sources,
        target: target.unwrap_or_else(|| SemanticTargetRef::Memory(String::new())),
        updates,
        reason,
        observed_at,
        caused_by,
        namespace,
    })
}

fn build_retract(pair: pest::iterators::Pair<'_, Rule>) -> Result<RetractStmt, ParseError> {
    let mut target = None;
    let mut reason = None;
    let mut observed_at = None;
    let mut caused_by = None;
    let mut namespace = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::semantic_target_ref if target.is_none() => {
                target = Some(build_semantic_target_ref(inner)?);
            }
            Rule::reason_clause => reason = Some(extract_string_from_clause(inner)?),
            Rule::observed_at_clause => observed_at = Some(extract_string_from_clause(inner)?),
            Rule::caused_by_clause => caused_by = Some(extract_string_from_clause(inner)?),
            Rule::namespace_clause => namespace = Some(extract_namespace(inner)),
            _ => {}
        }
    }

    Ok(RetractStmt {
        target: target.unwrap_or_else(|| SemanticTargetRef::Memory(String::new())),
        reason,
        observed_at,
        caused_by,
        namespace,
    })
}

fn build_inspect(pair: pest::iterators::Pair<'_, Rule>) -> Result<InspectStmt, ParseError> {
    let target = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::semantic_target_ref)
        .map(build_semantic_target_ref)
        .transpose()?
        .unwrap_or_else(|| SemanticTargetRef::Memory(String::new()));
    Ok(InspectStmt { target })
}

fn build_history(pair: pest::iterators::Pair<'_, Rule>) -> Result<HistoryStmt, ParseError> {
    let mut target = None;
    let mut namespace = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::semantic_target_ref if target.is_none() => {
                target = Some(build_semantic_target_ref(inner)?);
            }
            Rule::namespace_clause => namespace = Some(extract_namespace(inner)),
            _ => {}
        }
    }

    Ok(HistoryStmt {
        target: target.unwrap_or_else(|| SemanticTargetRef::Memory(String::new())),
        namespace,
    })
}

fn build_trace(pair: pest::iterators::Pair<'_, Rule>) -> Result<TraceStmt, ParseError> {
    let target = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::semantic_target_ref)
        .map(build_semantic_target_ref)
        .transpose()?
        .unwrap_or_else(|| SemanticTargetRef::Memory(String::new()));
    Ok(TraceStmt { target })
}

fn build_semantic_target_list(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<Vec<SemanticTargetRef>, ParseError> {
    pair.into_inner()
        .filter(|child| child.as_rule() == Rule::semantic_target_ref)
        .map(build_semantic_target_ref)
        .collect()
}

fn build_semantic_target_ref(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<SemanticTargetRef, ParseError> {
    let Some(inner) = pair.into_inner().next() else {
        return Ok(SemanticTargetRef::Memory(String::new()));
    };

    match inner.as_rule() {
        Rule::logical_target_ref => {
            let value = inner
                .into_inner()
                .find(|child| child.as_rule() == Rule::string_literal)
                .map(extract_string_value)
                .transpose()?
                .unwrap_or_default();
            Ok(SemanticTargetRef::Logical(value))
        }
        Rule::revision_target_ref => {
            let value = inner
                .into_inner()
                .find(|child| child.as_rule() == Rule::string_literal)
                .map(extract_string_value)
                .transpose()?
                .unwrap_or_default();
            Ok(SemanticTargetRef::Revision(value))
        }
        Rule::string_literal => Ok(SemanticTargetRef::Memory(extract_string_value(inner)?)),
        _ => Err(ParseError::simple(format!(
            "unexpected semantic target rule: {:?}",
            inner.as_rule()
        ))),
    }
}

// ── Clause builders ────────────────────────────────────────────────────

fn build_layer_filter(pair: pest::iterators::Pair<'_, Rule>) -> Vec<Layer> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::layer_name)
        .map(|p| {
            let s = p.as_str();
            if s.eq_ignore_ascii_case("episodic") {
                Layer::Episodic
            } else if s.eq_ignore_ascii_case("semantic") {
                Layer::Semantic
            } else if s.eq_ignore_ascii_case("working") {
                Layer::Working
            } else if s.eq_ignore_ascii_case("procedural") {
                Layer::Procedural
            } else {
                Layer::Episodic
            }
        })
        .collect()
}

fn extract_about(pair: pest::iterators::Pair<'_, Rule>) -> Result<String, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::simple("empty ABOUT clause"))?;
    Ok(match inner.as_rule() {
        Rule::parameter => inner.as_str().to_string(),
        Rule::string_literal => extract_string_value(inner)?,
        _ => String::new(),
    })
}

fn extract_string_list(pair: pest::iterators::Pair<'_, Rule>) -> Result<Vec<String>, ParseError> {
    fn inner_list(pair: pest::iterators::Pair<'_, Rule>) -> Result<Vec<String>, ParseError> {
        let mut result = Vec::new();
        for p in pair.into_inner() {
            if p.as_rule() == Rule::string_list {
                result.extend(inner_list(p)?);
            } else if p.as_rule() == Rule::string_literal {
                result.push(extract_string_value(p)?);
            }
        }
        Ok(result)
    }
    inner_list(pair)
}

fn extract_string_value(pair: pest::iterators::Pair<'_, Rule>) -> Result<String, ParseError> {
    // string_literal → double_inner | single_inner (containing plain chars + escape_seq)
    let raw = pair
        .into_inner()
        .next()
        .map(|p| p.as_str().to_string())
        .unwrap_or_default();
    unescape_string(&raw)
}

/// Process escape sequences (\\, \", \', \n, \t, \r) in a parsed string.
/// Returns an error for unrecognised escape sequences.
fn unescape_string(s: &str) -> Result<String, ParseError> {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('\'') => out.push('\''),
                Some(other) => {
                    return Err(ParseError::simple(format!(
                        "invalid escape sequence: '\\{other}'"
                    )));
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

fn extract_string_from_clause(pair: pest::iterators::Pair<'_, Rule>) -> Result<String, ParseError> {
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::parameter => return Ok(p.as_str().to_string()),
            Rule::string_literal => return extract_string_value(p),
            _ => {}
        }
    }
    Ok(String::new())
}

fn extract_float(pair: pest::iterators::Pair<'_, Rule>) -> Result<f32, ParseError> {
    let p = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::float_literal)
        .ok_or_else(|| ParseError::simple("expected float literal"))?;
    let text = p.as_str();
    text.parse::<f32>()
        .map_err(|_| ParseError::simple(format!("invalid float literal: '{text}'")))
}

fn extract_int(pair: pest::iterators::Pair<'_, Rule>) -> Result<usize, ParseError> {
    let p = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::integer_literal || p.as_rule() == Rule::parameter)
        .ok_or_else(|| ParseError::simple("expected integer literal"))?;
    // Parameters use a placeholder value; the real value is substituted after bind().
    if p.as_rule() == Rule::parameter {
        return Ok(0);
    }
    let text = p.as_str();
    text.parse::<usize>()
        .map_err(|_| ParseError::simple(format!("invalid integer literal: '{text}'")))
}

fn build_temporal(pair: pest::iterators::Pair<'_, Rule>) -> Result<TemporalClause, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::simple("empty temporal clause"))?;
    Ok(match inner.as_rule() {
        Rule::after_clause => {
            let s = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::string_literal)
                .map(extract_string_value)
                .transpose()?
                .unwrap_or_default();
            TemporalClause::After(s)
        }
        Rule::before_clause => {
            let s = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::string_literal)
                .map(extract_string_value)
                .transpose()?
                .unwrap_or_default();
            TemporalClause::Before(s)
        }
        Rule::between_clause => {
            let strings: Vec<String> = inner
                .into_inner()
                .filter(|p| p.as_rule() == Rule::string_literal)
                .map(extract_string_value)
                .collect::<Result<Vec<_>, _>>()?;
            TemporalClause::Between {
                start: strings.first().cloned().unwrap_or_default(),
                end: strings.get(1).cloned().unwrap_or_default(),
            }
        }
        _ => TemporalClause::After(String::new()),
    })
}

fn build_expand(pair: pest::iterators::Pair<'_, Rule>) -> Result<ExpandClause, ParseError> {
    let mut depth = 1;
    let mut min_weight = None;
    let mut activation = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::integer_literal => {
                depth = inner.as_str().parse::<usize>().map_err(|_| {
                    ParseError::simple(format!("invalid DEPTH value: '{}'", inner.as_str()))
                })?;
            }
            Rule::min_weight_clause => min_weight = Some(extract_float(inner)?),
            Rule::activation_clause => {
                let mode_str = inner
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::activation_mode)
                    .map(|p| p.as_str())
                    .unwrap_or_default();
                activation = Some(if mode_str.eq_ignore_ascii_case("spreading") {
                    ActivationModeAst::Spreading
                } else if mode_str.eq_ignore_ascii_case("static") {
                    ActivationModeAst::Static
                } else if mode_str.eq_ignore_ascii_case("ppr")
                    || mode_str.eq_ignore_ascii_case("pagerank")
                {
                    ActivationModeAst::Ppr
                } else {
                    ActivationModeAst::None
                });
            }
            _ => {}
        }
    }

    Ok(ExpandClause {
        depth,
        min_weight,
        activation,
    })
}

fn extract_follow_causes(pair: pest::iterators::Pair<'_, Rule>) -> Result<usize, ParseError> {
    let p = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::integer_literal)
        .ok_or_else(|| ParseError::simple("FOLLOW CAUSES requires an integer depth"))?;
    let text = p.as_str();
    text.parse::<usize>()
        .map_err(|_| ParseError::simple(format!("invalid FOLLOW CAUSES depth: '{text}'")))
}

fn build_where(pair: pest::iterators::Pair<'_, Rule>) -> Result<WhereCondition, ParseError> {
    let condition = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::condition)
        .ok_or_else(|| ParseError::simple("WHERE clause missing condition"))?;
    build_condition(condition)
}

/// Parse a single condition (field op value).
fn build_condition(
    condition: pest::iterators::Pair<'_, Rule>,
) -> Result<WhereCondition, ParseError> {
    let mut field = String::new();
    let mut op = ComparisonOp::Gt;
    let mut value = ConditionValue::Float(0.0);

    for inner in condition.into_inner() {
        match inner.as_rule() {
            Rule::identifier => field = inner.as_str().to_string(),
            Rule::comparison_op => {
                op = match inner.as_str() {
                    ">=" => ComparisonOp::Gte,
                    "<=" => ComparisonOp::Lte,
                    "!=" => ComparisonOp::Neq,
                    ">" => ComparisonOp::Gt,
                    "<" => ComparisonOp::Lt,
                    "=" => ComparisonOp::Eq,
                    _ => ComparisonOp::Eq,
                };
            }
            Rule::float_literal => {
                let text = inner.as_str();
                value = ConditionValue::Float(text.parse().map_err(|_| {
                    ParseError::simple(format!("invalid float in WHERE: '{text}'"))
                })?);
            }
            Rule::integer_literal => {
                let text = inner.as_str();
                value = ConditionValue::Int(text.parse().map_err(|_| {
                    ParseError::simple(format!("invalid integer in WHERE: '{text}'"))
                })?);
            }
            Rule::string_literal => {
                value = ConditionValue::String(extract_string_value(inner)?);
            }
            Rule::parameter => {
                value = ConditionValue::Param(inner.as_str().to_string());
            }
            _ => {}
        }
    }

    Ok(WhereCondition { field, op, value })
}

/// Parse an IN subquery condition: `field IN (RECALL ...)`.
fn build_in_subquery(pair: pest::iterators::Pair<'_, Rule>) -> Result<SubqueryFilter, ParseError> {
    let mut field = String::new();
    let mut subquery = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => field = inner.as_str().to_string(),
            Rule::subquery => subquery = Some(build_subquery(inner)?),
            _ => {}
        }
    }

    Ok(SubqueryFilter {
        field,
        subquery: subquery.unwrap_or(Subquery {
            layers: vec![],
            about: String::new(),
            involving: None,
            temporal: None,
            limit: None,
        }),
    })
}

/// Parse the inner subquery (RECALL layer ABOUT "..." ...).
fn build_subquery(pair: pest::iterators::Pair<'_, Rule>) -> Result<Subquery, ParseError> {
    let mut layers = vec![];
    let mut about = String::new();
    let mut involving = None;
    let mut temporal = None;
    let mut limit = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::layer_filter => layers = build_layer_filter(inner),
            Rule::about_clause => about = extract_about(inner)?,
            Rule::involving_clause => involving = Some(extract_string_list(inner)?),
            Rule::temporal_clause => temporal = Some(build_temporal(inner)?),
            Rule::limit_clause => limit = Some(extract_limit(inner)?),
            _ => {}
        }
    }

    Ok(Subquery {
        layers,
        about,
        involving,
        temporal,
        limit,
    })
}

/// Parse an AS OF clause for time-travel queries.
fn build_as_of(pair: pest::iterators::Pair<'_, Rule>) -> Result<RecallSnapshotAst, ParseError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| ParseError::simple("AS OF clause requires a snapshot target"))?;

    match inner.as_rule() {
        Rule::string_literal => Ok(RecallSnapshotAst::Unqualified(extract_string_value(inner)?)),
        Rule::as_of_observed => Ok(RecallSnapshotAst::Observed(extract_single_string_literal(
            inner,
            "AS OF OBSERVED",
        )?)),
        Rule::as_of_recorded => Ok(RecallSnapshotAst::Recorded(extract_single_string_literal(
            inner,
            "AS OF RECORDED",
        )?)),
        Rule::as_of_revision => Ok(RecallSnapshotAst::Revision(extract_single_string_literal(
            inner,
            "AS OF REVISION",
        )?)),
        other => Err(ParseError::simple(format!(
            "unexpected AS OF target: {other:?}"
        ))),
    }
}

fn extract_single_string_literal(
    pair: pest::iterators::Pair<'_, Rule>,
    clause: &str,
) -> Result<String, ParseError> {
    pair.into_inner()
        .find(|p| p.as_rule() == Rule::string_literal)
        .map(extract_string_value)
        .transpose()?
        .ok_or_else(|| ParseError::simple(format!("{clause} requires a string literal")))
}

fn build_output_format(pair: pest::iterators::Pair<'_, Rule>) -> OutputFormat {
    let fmt_str = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::output_format)
        .map(|p| p.as_str())
        .unwrap_or_default();
    parse_output_format(fmt_str)
}

fn build_format_clause(pair: pest::iterators::Pair<'_, Rule>) -> OutputFormat {
    let fmt_str = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::output_format)
        .map(|p| p.as_str())
        .unwrap_or_default();
    parse_output_format(fmt_str)
}

fn parse_output_format(s: &str) -> OutputFormat {
    if s.eq_ignore_ascii_case("narrative") {
        OutputFormat::Narrative
    } else if s.eq_ignore_ascii_case("context") {
        OutputFormat::Context
    } else if s.eq_ignore_ascii_case("graph") {
        OutputFormat::Graph
    } else if s.eq_ignore_ascii_case("causal_chain") {
        OutputFormat::CausalChain
    } else if s.eq_ignore_ascii_case("json") {
        OutputFormat::Json
    } else if s.eq_ignore_ascii_case("csv") {
        OutputFormat::Csv
    } else if s.eq_ignore_ascii_case("structured") {
        OutputFormat::Structured
    } else {
        OutputFormat::Context
    }
}

fn build_group_by(pair: pest::iterators::Pair<'_, Rule>) -> GroupByClause {
    let mut field = String::new();
    let mut function = AggFunction::Count;
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => field = inner.as_str().to_string(),
            Rule::agg_function => {
                let s = inner.as_str();
                function = if s.eq_ignore_ascii_case("count") {
                    AggFunction::Count
                } else if s.eq_ignore_ascii_case("avg") {
                    AggFunction::Avg
                } else if s.eq_ignore_ascii_case("sum") {
                    AggFunction::Sum
                } else if s.eq_ignore_ascii_case("min") {
                    AggFunction::Min
                } else if s.eq_ignore_ascii_case("max") {
                    AggFunction::Max
                } else {
                    AggFunction::Count
                };
            }
            _ => {}
        }
    }
    GroupByClause { field, function }
}

fn build_field_list(pair: pest::iterators::Pair<'_, Rule>) -> Vec<String> {
    let mut fields = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::field_list {
            for field in inner.into_inner() {
                if field.as_rule() == Rule::identifier {
                    fields.push(field.as_str().to_string());
                }
            }
        }
    }
    fields
}

fn extract_budget(pair: pest::iterators::Pair<'_, Rule>) -> Result<usize, ParseError> {
    extract_int(pair)
}

fn extract_namespace(pair: pest::iterators::Pair<'_, Rule>) -> String {
    pair.into_inner()
        .find_map(|p| match p.as_rule() {
            Rule::namespace_identifier => Some(p.as_str().to_string()),
            Rule::string_literal => extract_string_value(p).ok(),
            _ => None,
        })
        .unwrap_or_default()
}

/// Extract realm IDs from `from_realm_clause`:
/// `FROM REALM "a", "b"` → `["a", "b"]`
fn extract_realm_list(pair: pest::iterators::Pair<'_, Rule>) -> Vec<String> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::string_literal)
        .filter_map(|p| extract_string_value(p).ok())
        .collect()
}

fn build_consistency(pair: pest::iterators::Pair<'_, Rule>) -> ConsistencyLevel {
    let level_str = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::consistency_level)
        .map(|p| p.as_str())
        .unwrap_or_default();
    if level_str.eq_ignore_ascii_case("linearizable") {
        ConsistencyLevel::Linearizable
    } else if level_str.eq_ignore_ascii_case("eventual") {
        ConsistencyLevel::Eventual
    } else {
        ConsistencyLevel::Session
    }
}

fn extract_limit(pair: pest::iterators::Pair<'_, Rule>) -> Result<usize, ParseError> {
    extract_int(pair)
}

fn build_retrieval_mode_with_hops(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<(RetrievalMode, Option<usize>), ParseError> {
    let mut mode = RetrievalMode::Local;
    let mut max_hops = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::retrieval_mode => {
                let mode_str = inner.as_str();
                mode = if mode_str.eq_ignore_ascii_case("global") {
                    RetrievalMode::Global
                } else if mode_str.eq_ignore_ascii_case("hybrid") {
                    RetrievalMode::Hybrid
                } else if mode_str.eq_ignore_ascii_case("raptor") {
                    RetrievalMode::Raptor
                } else if mode_str.eq_ignore_ascii_case("adaptive") {
                    RetrievalMode::Adaptive
                } else if mode_str.eq_ignore_ascii_case("iterative") {
                    RetrievalMode::Iterative
                } else {
                    RetrievalMode::Local
                };
            }
            Rule::max_hops_clause => {
                let hops = extract_integer_from_clause(inner)?;
                if hops == 0 || hops > 5 {
                    return Err(ParseError::simple(format!(
                        "MAX_HOPS must be between 1 and 5, got {hops}"
                    )));
                }
                max_hops = Some(hops);
            }
            _ => {}
        }
    }

    // MAX_HOPS is only valid with ITERATIVE mode.
    if max_hops.is_some() && mode != RetrievalMode::Iterative {
        return Err(ParseError::simple(
            "MAX_HOPS can only be used with MODE ITERATIVE",
        ));
    }

    Ok((mode, max_hops))
}

fn build_depth_mode(pair: pest::iterators::Pair<'_, Rule>) -> Result<DepthModeAst, ParseError> {
    let mode_str = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::depth_mode)
        .map(|p| p.as_str())
        .unwrap_or_default();
    if mode_str.eq_ignore_ascii_case("full") {
        Ok(DepthModeAst::Full)
    } else if mode_str.eq_ignore_ascii_case("summary") {
        Ok(DepthModeAst::Summary)
    } else if mode_str.eq_ignore_ascii_case("auto") {
        Ok(DepthModeAst::Auto)
    } else {
        Err(ParseError::simple(format!(
            "unknown DEPTH mode '{mode_str}', expected AUTO, FULL, or SUMMARY"
        )))
    }
}

fn build_on_off(pair: pest::iterators::Pair<'_, Rule>) -> Result<bool, ParseError> {
    let val = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::on_off)
        .map(|p| p.as_str().to_string())
        .unwrap_or_default();
    if val.eq_ignore_ascii_case("on") {
        Ok(true)
    } else if val.eq_ignore_ascii_case("off") {
        Ok(false)
    } else {
        Err(ParseError::simple(format!(
            "expected ON or OFF, got '{val}'"
        )))
    }
}

fn extract_integer_from_clause(pair: pest::iterators::Pair<'_, Rule>) -> Result<usize, ParseError> {
    extract_int(pair)
}

fn build_traverse(pair: pest::iterators::Pair<'_, Rule>) -> Result<TraverseStmt, ParseError> {
    let mut from = String::new();
    let mut via = None;
    let mut depth = 1;
    let mut where_clauses = vec![];
    let mut limit = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_literal => from = extract_string_value(inner)?,
            Rule::via_clause => {
                let mut rels = vec![];
                for child in inner.into_inner() {
                    if child.as_rule() == Rule::relation_list {
                        for id in child.into_inner() {
                            if id.as_rule() == Rule::identifier {
                                rels.push(id.as_str().to_string());
                            }
                        }
                    }
                }
                via = Some(rels);
            }
            Rule::integer_literal => {
                depth = inner.as_str().parse::<usize>().map_err(|_| {
                    ParseError::simple(format!("invalid DEPTH value: '{}'", inner.as_str()))
                })?;
            }
            Rule::where_clause => where_clauses.push(build_where(inner)?),
            Rule::limit_clause => limit = Some(extract_limit(inner)?),
            _ => {}
        }
    }

    Ok(TraverseStmt {
        from,
        via,
        depth,
        where_clauses,
        limit,
        namespace: None,
    })
}

fn build_explain(pair: pest::iterators::Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let mut analyze = false;
    let mut inner_stmt = None;

    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::analyze_flag => analyze = true,
            Rule::inner_stmt => {
                let actual = child
                    .into_inner()
                    .next()
                    .ok_or_else(|| ParseError::simple("EXPLAIN requires a statement"))?;
                inner_stmt = Some(match actual.as_rule() {
                    Rule::recall_events_stmt => {
                        build_recall_events(actual).map(Statement::RecallEvents)?
                    }
                    Rule::recall_stmt => {
                        build_recall(actual).map(|stmt| Statement::Recall(Box::new(stmt)))?
                    }
                    Rule::think_stmt => {
                        build_think(actual).map(|stmt| Statement::Think(Box::new(stmt)))?
                    }
                    Rule::forget_stmt => return unsupported_embedded_statement(Rule::forget_stmt),
                    Rule::correct_stmt => build_correct(actual).map(Statement::Correct)?,
                    Rule::supersede_stmt => build_supersede(actual).map(Statement::Supersede)?,
                    Rule::merge_memory_stmt => {
                        build_merge_memory(actual).map(Statement::MergeMemory)?
                    }
                    Rule::retract_stmt => build_retract(actual).map(Statement::Retract)?,
                    Rule::history_stmt => build_history(actual).map(Statement::History)?,
                    Rule::traverse_stmt => build_traverse(actual).map(Statement::Traverse)?,
                    Rule::inspect_stmt => build_inspect(actual).map(Statement::Inspect)?,
                    Rule::trace_stmt => build_trace(actual).map(Statement::Trace)?,
                    Rule::explain_causes_stmt => {
                        build_explain_causes(actual).map(Statement::ExplainCauses)?
                    }
                    Rule::what_if_stmt => build_what_if(actual).map(Statement::WhatIf)?,
                    Rule::counterfactual_stmt => {
                        build_counterfactual(actual).map(Statement::Counterfactual)?
                    }
                    Rule::show_policies_stmt => {
                        build_show_policies(actual).map(Statement::ShowPolicies)?
                    }
                    Rule::explain_policy_stmt => {
                        build_explain_policy(actual).map(Statement::ExplainPolicy)?
                    }
                    _ => {
                        return Err(ParseError::simple(format!(
                            "EXPLAIN not supported for {:?}",
                            actual.as_rule()
                        )));
                    }
                });
            }
            _ => {}
        }
    }

    let inner = inner_stmt.ok_or_else(|| ParseError::simple("EXPLAIN requires a statement"))?;

    Ok(Statement::Explain(ExplainStmt {
        analyze,
        inner: Box::new(inner),
    }))
}

fn build_explain_causes(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<ExplainCausesStmt, ParseError> {
    let mut target = String::new();
    let mut namespace = None;
    let mut depth = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_literal => target = extract_string_value(inner)?,
            Rule::parameter => target = inner.as_str().to_string(),
            Rule::namespace_clause => namespace = Some(extract_namespace(inner)),
            Rule::causes_depth_clause => {
                depth = Some(extract_integer_from_clause(inner)?);
            }
            _ => {}
        }
    }

    Ok(ExplainCausesStmt {
        target,
        namespace,
        depth,
    })
}

fn build_what_if(pair: pest::iterators::Pair<'_, Rule>) -> Result<WhatIfStmt, ParseError> {
    let mut intervention = String::new();
    let mut outcome = String::new();
    let mut namespace = None;
    let mut got_first = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_literal | Rule::parameter => {
                let val = if inner.as_rule() == Rule::string_literal {
                    extract_string_value(inner)?
                } else {
                    inner.as_str().to_string()
                };
                if !got_first {
                    intervention = val;
                    got_first = true;
                } else {
                    outcome = val;
                }
            }
            Rule::namespace_clause => namespace = Some(extract_namespace(inner)),
            _ => {}
        }
    }

    Ok(WhatIfStmt {
        intervention,
        outcome,
        namespace,
    })
}

fn build_counterfactual(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<CounterfactualStmt, ParseError> {
    let mut antecedent = String::new();
    let mut consequent = String::new();
    let mut namespace = None;
    let mut got_first = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_literal | Rule::parameter => {
                let val = if inner.as_rule() == Rule::string_literal {
                    extract_string_value(inner)?
                } else {
                    inner.as_str().to_string()
                };
                if !got_first {
                    antecedent = val;
                    got_first = true;
                } else {
                    consequent = val;
                }
            }
            Rule::namespace_clause => namespace = Some(extract_namespace(inner)),
            _ => {}
        }
    }

    Ok(CounterfactualStmt {
        antecedent,
        consequent,
        namespace,
    })
}

fn build_set_assignment_list(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<Vec<SetAssignment>, ParseError> {
    let mut assignments = vec![];
    for child in pair.into_inner() {
        if child.as_rule() == Rule::set_assignment {
            assignments.push(build_set_assignment(child)?);
        }
    }
    Ok(assignments)
}

fn build_set_assignment(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<SetAssignment, ParseError> {
    let mut field = String::new();
    let mut value = SetValue::Int(0);

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => field = inner.as_str().to_string(),
            Rule::set_value => {
                let child = inner
                    .into_inner()
                    .next()
                    .ok_or_else(|| ParseError::simple("empty set value"))?;
                value = match child.as_rule() {
                    Rule::set_function => build_set_function(child)?,
                    Rule::float_literal => {
                        let text = child.as_str();
                        SetValue::Float(text.parse().map_err(|_| {
                            ParseError::simple(format!("invalid float in SET: '{text}'"))
                        })?)
                    }
                    Rule::integer_literal => {
                        let text = child.as_str();
                        SetValue::Int(text.parse().map_err(|_| {
                            ParseError::simple(format!("invalid integer in SET: '{text}'"))
                        })?)
                    }
                    Rule::string_literal => SetValue::String(extract_string_value(child)?),
                    _ => SetValue::Int(0),
                };
            }
            _ => {}
        }
    }

    Ok(SetAssignment { field, value })
}

fn build_set_function(pair: pest::iterators::Pair<'_, Rule>) -> Result<SetValue, ParseError> {
    let raw = pair.as_str();
    let is_max = raw.len() >= 3 && raw[..3].eq_ignore_ascii_case("max");
    let mut field = String::new();
    let mut val = 0.0;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => field = inner.as_str().to_string(),
            Rule::float_literal => {
                let t = inner.as_str();
                val = t.parse().map_err(|_| {
                    ParseError::simple(format!("invalid float in SET function: '{t}'"))
                })?;
            }
            Rule::integer_literal => {
                let t = inner.as_str();
                val = t.parse().map_err(|_| {
                    ParseError::simple(format!("invalid integer in SET function: '{t}'"))
                })?;
            }
            _ => {}
        }
    }

    Ok(if is_max {
        SetValue::Max(field, val)
    } else {
        SetValue::Min(field, val)
    })
}

// ── Multi-modal helpers ────────────────────────────────────────────────

fn build_modality_list(pair: pest::iterators::Pair<'_, Rule>) -> Result<Vec<String>, ParseError> {
    build_named_list(
        pair,
        Rule::modality_list,
        Rule::modality_name,
        "MODALITY clause missing modality list",
    )
}

fn build_evidence_role_list(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<Vec<String>, ParseError> {
    build_named_list(
        pair,
        Rule::evidence_role_list,
        Rule::evidence_role_name,
        "RESOURCE_ROLE clause missing evidence role list",
    )
}

fn build_hydration_mode_list(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<Vec<String>, ParseError> {
    build_named_list(
        pair,
        Rule::hydration_mode_list,
        Rule::hydration_mode_name,
        "HYDRATION clause missing hydration mode list",
    )
}

fn build_artifact_kind_list(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<Vec<String>, ParseError> {
    build_named_list(
        pair,
        Rule::artifact_kind_list,
        Rule::artifact_kind_name,
        "ARTIFACT clause missing artifact kind list",
    )
}

fn build_named_list(
    pair: pest::iterators::Pair<'_, Rule>,
    list_rule: Rule,
    item_rule: Rule,
    missing_message: &str,
) -> Result<Vec<String>, ParseError> {
    let list = pair
        .into_inner()
        .find(|p| p.as_rule() == list_rule)
        .ok_or_else(|| ParseError::simple(missing_message))?;
    Ok(list
        .into_inner()
        .filter(|p| p.as_rule() == item_rule)
        .map(|p| p.as_str().to_lowercase())
        .collect())
}

// ── CREATE REALM / DROP REALM ──────────────────

fn build_create_realm(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<CreateRealmStmt, ParseError> {
    let mut name = String::new();
    let mut description = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_literal if name.is_empty() => {
                name = extract_string_value(inner)?;
            }
            Rule::realm_description => {
                for child in inner.into_inner() {
                    if child.as_rule() == Rule::string_literal {
                        description = Some(extract_string_value(child)?);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(CreateRealmStmt { name, description })
}

fn build_drop_realm(pair: pest::iterators::Pair<'_, Rule>) -> Result<DropRealmStmt, ParseError> {
    let mut name = String::new();
    let mut confirm = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::string_literal if name.is_empty() => {
                name = extract_string_value(inner)?;
            }
            Rule::confirm_flag => confirm = true,
            _ => {}
        }
    }

    Ok(DropRealmStmt { name, confirm })
}

// ── GRANT / REVOKE ─────────────────────────────

fn build_action_list(pair: pest::iterators::Pair<'_, Rule>) -> Vec<String> {
    pair.into_inner()
        .filter(|p| p.as_rule() == Rule::action_name)
        .map(|p| p.as_str().to_lowercase())
        .collect()
}

fn build_grant_target(pair: pest::iterators::Pair<'_, Rule>) -> Result<GrantTarget, ParseError> {
    let raw = pair.as_str();
    let is_namespace = raw.to_ascii_lowercase().contains("namespace");
    let string_val = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::string_literal)
        .map(extract_string_value)
        .transpose()?
        .unwrap_or_default();

    Ok(if is_namespace {
        GrantTarget::Namespace(string_val)
    } else {
        GrantTarget::Realm(string_val)
    })
}

fn build_principal_ref(pair: pest::iterators::Pair<'_, Rule>) -> Result<PrincipalRef, ParseError> {
    let raw = pair.as_str();
    let is_team = raw.to_ascii_lowercase().contains("team");
    let string_val = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::string_literal)
        .map(extract_string_value)
        .transpose()?
        .unwrap_or_default();

    Ok(if is_team {
        PrincipalRef::Team(string_val)
    } else {
        PrincipalRef::Agent(string_val)
    })
}

fn build_grant(pair: pest::iterators::Pair<'_, Rule>) -> Result<GrantStmt, ParseError> {
    let mut actions = Vec::new();
    let mut target = None;
    let mut principal = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::action_list => actions = build_action_list(inner),
            Rule::grant_target => target = Some(build_grant_target(inner)?),
            Rule::principal_ref => principal = Some(build_principal_ref(inner)?),
            _ => {}
        }
    }

    Ok(GrantStmt {
        actions,
        target: target
            .ok_or_else(|| ParseError::simple("GRANT requires ON NAMESPACE/REALM clause"))?,
        principal: principal
            .ok_or_else(|| ParseError::simple("GRANT requires TO AGENT/TEAM clause"))?,
    })
}

fn build_revoke(pair: pest::iterators::Pair<'_, Rule>) -> Result<RevokeStmt, ParseError> {
    let mut actions = Vec::new();
    let mut target = None;
    let mut principal = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::action_list => actions = build_action_list(inner),
            Rule::grant_target => target = Some(build_grant_target(inner)?),
            Rule::principal_ref => principal = Some(build_principal_ref(inner)?),
            _ => {}
        }
    }

    Ok(RevokeStmt {
        actions,
        target: target
            .ok_or_else(|| ParseError::simple("REVOKE requires ON NAMESPACE/REALM clause"))?,
        principal: principal
            .ok_or_else(|| ParseError::simple("REVOKE requires FROM AGENT/TEAM clause"))?,
    })
}

// ── SHOW POLICIES / EXPLAIN POLICY ────────────

fn build_show_policies(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<ShowPoliciesStmt, ParseError> {
    let mut principal = None;

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::principal_ref {
            principal = Some(build_principal_ref(inner)?);
        }
    }

    Ok(ShowPoliciesStmt { principal })
}

fn build_explain_policy(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<ExplainPolicyStmt, ParseError> {
    let mut principal = None;
    let mut resource_type = String::new();
    let mut resource_name = String::new();
    let mut action = String::new();

    let raw = pair.as_str();
    // Detect resource type from the raw text (ON NAMESPACE vs ON REALM).
    let raw_lower = raw.to_ascii_lowercase();
    if raw_lower.contains("namespace") {
        resource_type = "namespace".to_string();
    } else if raw_lower.contains("realm") {
        resource_type = "realm".to_string();
    }

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::principal_ref => principal = Some(build_principal_ref(inner)?),
            Rule::string_literal if resource_name.is_empty() => {
                resource_name = extract_string_value(inner)?;
            }
            Rule::action_name => action = inner.as_str().to_lowercase(),
            _ => {}
        }
    }

    Ok(ExplainPolicyStmt {
        principal: principal
            .ok_or_else(|| ParseError::simple("EXPLAIN POLICY requires FOR AGENT/TEAM clause"))?,
        resource_type,
        resource_name,
        action,
    })
}

// ── SET TIER_POLICY ────────────────────────────

fn build_set_tier_policy(
    pair: pest::iterators::Pair<'_, Rule>,
) -> Result<SetTierPolicyStmt, ParseError> {
    let mut field = String::new();
    let mut value = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::tier_policy_field => {
                field = inner.as_str().to_lowercase();
            }
            Rule::tier_policy_value => {
                let val_inner = inner
                    .into_inner()
                    .next()
                    .ok_or_else(|| ParseError::simple("missing tier policy value"))?;
                value = Some(match val_inner.as_rule() {
                    Rule::string_literal => TierPolicyValue::Str(extract_string_value(val_inner)?),
                    Rule::float_literal => {
                        let v: f64 = val_inner
                            .as_str()
                            .parse()
                            .map_err(|_| ParseError::simple("invalid float in SET TIER_POLICY"))?;
                        TierPolicyValue::Float(v)
                    }
                    Rule::integer_literal => {
                        let v: i64 = val_inner.as_str().parse().map_err(|_| {
                            ParseError::simple("invalid integer in SET TIER_POLICY")
                        })?;
                        TierPolicyValue::Int(v)
                    }
                    _ => {
                        return Err(ParseError::simple("unexpected tier policy value type"));
                    }
                });
            }
            _ => {}
        }
    }

    Ok(SetTierPolicyStmt {
        field,
        value: value.ok_or_else(|| ParseError::simple("missing value in SET TIER_POLICY"))?,
    })
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_recall() {
        let stmt = parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.layers, vec![Layer::Episodic]);
                assert_eq!(r.about, "test");
                assert!(r.limit.is_none());
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_full_recall() {
        let q = r#"
            RECALL semantic, episodic
              ABOUT "vector database optimization"
              INVOLVING "HNSW", "benchmark"
              AFTER "2026-03-01"
              EXPAND GRAPH DEPTH 2 MIN_WEIGHT 0.3 ACTIVATION spreading
              FOLLOW CAUSES DEPTH 3
              WHERE importance > 0.4
              WHERE confidence > 0.8
              AS NARRATIVE
              BUDGET 4096
              NAMESPACE shared_knowledge
              CONSISTENCY linearizable
              LIMIT 20
        "#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.layers, vec![Layer::Semantic, Layer::Episodic]);
                assert_eq!(r.about, "vector database optimization");
                assert_eq!(r.involving.unwrap(), vec!["HNSW", "benchmark"]);
                assert_eq!(r.temporal, Some(TemporalClause::After("2026-03-01".into())));

                let ex = r.expand.unwrap();
                assert_eq!(ex.depth, 2);
                assert_eq!(ex.min_weight, Some(0.3));
                assert_eq!(ex.activation, Some(ActivationModeAst::Spreading));

                assert_eq!(r.follow_causes, Some(3));
                assert_eq!(r.where_clauses.len(), 2);
                assert_eq!(r.output_format, Some(OutputFormat::Narrative));
                assert_eq!(r.budget, Some(4096));
                assert_eq!(r.namespace, Some("shared_knowledge".into()));
                assert_eq!(r.consistency, Some(ConsistencyLevel::Linearizable));
                assert_eq!(r.limit, Some(20));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_with_budget() {
        let q = r#"THINK ABOUT "How should I optimize HNSW?" BUDGET 4096"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert!(t.about.contains("HNSW"));
                assert_eq!(t.budget, Some(4096));
                assert_eq!(t.mode, RetrievalMode::Local);
                assert_eq!(t.community_depth, None);
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_global() {
        let q = r#"THINK GLOBAL ABOUT "summarize themes""#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert!(t.about.contains("themes"));
                assert_eq!(t.mode, RetrievalMode::Global);
                assert_eq!(t.community_depth, None);
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_hybrid_with_community_depth() {
        let q = r#"THINK ABOUT "cross-domain links" MODE hybrid COMMUNITY_DEPTH 3"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert!(t.about.contains("cross-domain"));
                assert_eq!(t.mode, RetrievalMode::Hybrid);
                assert_eq!(t.community_depth, Some(3));
                assert!(!t.hybrid);
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_query_text_hybrid_clause() {
        let q = r#"THINK ABOUT "cross-domain links" HYBRID"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert_eq!(t.mode, RetrievalMode::Local);
                assert!(t.hybrid);
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_mode_local_explicit() {
        let q = r#"THINK ABOUT "x" MODE local"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert_eq!(t.mode, RetrievalMode::Local);
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_think_global() {
        let q = r#"THINK GLOBAL ABOUT "themes" BUDGET 2048 COMMUNITY_DEPTH 2"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn roundtrip_think_hybrid() {
        let q = r#"THINK ABOUT "links" MODE hybrid COMMUNITY_DEPTH 5"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn roundtrip_think_query_text_hybrid() {
        let q = r#"THINK ABOUT "links" HYBRID"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn parse_think_mode_raptor() {
        let q = r#"THINK ABOUT "architecture overview" MODE raptor"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert!(t.about.contains("architecture"));
                assert_eq!(t.mode, RetrievalMode::Raptor);
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_mode_adaptive() {
        let q = r#"THINK ABOUT "deployment strategies" MODE adaptive"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert!(t.about.contains("deployment"));
                assert_eq!(t.mode, RetrievalMode::Adaptive);
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_think_raptor() {
        let q = r#"THINK ABOUT "overview" MODE raptor COMMUNITY_DEPTH 3"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn roundtrip_think_adaptive() {
        let q = r#"THINK ABOUT "analysis" MODE adaptive"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn parse_correct() {
        let q = r#"CORRECT "some_id" SET description = "updated", confidence = 0.9 REASON "fix" OBSERVED AT "2026-01-01T00:00:00Z" CAUSED BY "cause_id" NAMESPACE custom"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Correct(c) => {
                assert_eq!(c.target, SemanticTargetRef::Memory("some_id".into()));
                assert_eq!(c.updates.len(), 2);
                assert_eq!(c.reason.as_deref(), Some("fix"));
                assert_eq!(c.observed_at.as_deref(), Some("2026-01-01T00:00:00Z"));
                assert_eq!(c.caused_by.as_deref(), Some("cause_id"));
                assert_eq!(c.namespace.as_deref(), Some("custom"));
            }
            other => panic!("expected Correct, got {other:?}"),
        }
    }

    #[test]
    fn parse_supersede() {
        let q = r#"SUPERSEDE LOGICAL "some_id" SET description = "replacement", confidence = 0.8 REASON "new authority" OBSERVED AT "2026-02-01T00:00:00Z" CAUSED BY "cause_id" NAMESPACE custom"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Supersede(s) => {
                assert_eq!(s.target, SemanticTargetRef::Logical("some_id".into()));
                assert_eq!(s.updates.len(), 2);
                assert_eq!(s.reason.as_deref(), Some("new authority"));
                assert_eq!(s.observed_at.as_deref(), Some("2026-02-01T00:00:00Z"));
                assert_eq!(s.caused_by.as_deref(), Some("cause_id"));
                assert_eq!(s.namespace.as_deref(), Some("custom"));
            }
            other => panic!("expected Supersede, got {other:?}"),
        }
    }

    #[test]
    fn parse_merge_memory() {
        let q = r#"MERGE MEMORY "source_a", REVISION "source_b" INTO LOGICAL "target_id" SET description = "canonical", confidence = 0.95 REASON "deduplicate" OBSERVED AT "2026-03-01T00:00:00Z" CAUSED BY "cause_id" NAMESPACE custom"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::MergeMemory(m) => {
                assert_eq!(
                    m.sources,
                    vec![
                        SemanticTargetRef::Memory("source_a".into()),
                        SemanticTargetRef::Revision("source_b".into()),
                    ]
                );
                assert_eq!(m.target, SemanticTargetRef::Logical("target_id".into()));
                assert_eq!(m.updates.len(), 2);
                assert_eq!(m.reason.as_deref(), Some("deduplicate"));
                assert_eq!(m.observed_at.as_deref(), Some("2026-03-01T00:00:00Z"));
                assert_eq!(m.caused_by.as_deref(), Some("cause_id"));
                assert_eq!(m.namespace.as_deref(), Some("custom"));
            }
            other => panic!("expected MergeMemory, got {other:?}"),
        }
    }

    #[test]
    fn parse_retract() {
        let q = r#"RETRACT REVISION "some_id" REASON "obsolete" OBSERVED AT "2026-01-01" CAUSED BY "cause_id" NAMESPACE custom"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Retract(r) => {
                assert_eq!(r.target, SemanticTargetRef::Revision("some_id".into()));
                assert_eq!(r.reason.as_deref(), Some("obsolete"));
                assert_eq!(r.observed_at.as_deref(), Some("2026-01-01"));
                assert_eq!(r.caused_by.as_deref(), Some("cause_id"));
                assert_eq!(r.namespace.as_deref(), Some("custom"));
            }
            other => panic!("expected Retract, got {other:?}"),
        }
    }

    #[test]
    fn parse_remember_is_unsupported() {
        let err = parse(r#"REMEMBER episode CONTENT "event happened""#).unwrap_err();
        assert!(err.message.contains("REMEMBER is not supported"));
    }

    #[test]
    fn parse_forget_is_unsupported() {
        let err = parse(r#"FORGET "01J000000000000000000000""#).unwrap_err();
        assert!(err.message.contains("FORGET is not supported"));
    }

    #[test]
    fn parse_connect_is_unsupported() {
        let q = r#"CONNECT "HNSW_indexing" TO "approximate_nearest_neighbors" AS related_to WEIGHT 0.9"#;
        let err = parse(q).unwrap_err();
        assert!(err.message.contains("CONNECT is not supported"));
    }

    #[test]
    fn parse_consolidate_is_unsupported() {
        let err = parse("CONSOLIDATE WHERE episodic.access_count > 5").unwrap_err();
        assert!(err.message.contains("CONSOLIDATE is not supported"));
    }

    #[test]
    fn parse_watch_is_unsupported() {
        let err = parse("WATCH ALL FORMAT json").unwrap_err();
        assert!(err.message.contains("WATCH is not supported"));
    }

    #[test]
    fn parse_explain_analyze_forget_is_unsupported() {
        let err = parse(r#"EXPLAIN ANALYZE FORGET "01J000000000000000000000""#).unwrap_err();
        assert!(err.message.contains("FORGET is not supported"));
    }

    #[test]
    fn parse_inspect() {
        let q = r#"INSPECT LOGICAL "record_id""#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Inspect(i) => {
                assert_eq!(i.target, SemanticTargetRef::Logical("record_id".into()));
            }
            other => panic!("expected Inspect, got {other:?}"),
        }
    }

    #[test]
    fn parse_history() {
        let q = r#"HISTORY REVISION "record_id" NAMESPACE custom"#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::History(h) => {
                assert_eq!(h.target, SemanticTargetRef::Revision("record_id".into()));
                assert_eq!(h.namespace.as_deref(), Some("custom"));
            }
            other => panic!("expected History, got {other:?}"),
        }
    }

    #[test]
    fn parse_trace() {
        let q = r#"TRACE LOGICAL "semantic:caching_best_practices""#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Trace(t) => {
                assert_eq!(
                    t.target,
                    SemanticTargetRef::Logical("semantic:caching_best_practices".into())
                );
            }
            other => panic!("expected Trace, got {other:?}"),
        }
    }

    #[test]
    fn parse_error_unknown_verb() {
        let err = parse("SELECT * FROM memories").unwrap_err();
        assert!(err.message.contains("unknown verb"));
        assert!(err.message.contains("RECALL"));
    }

    #[test]
    fn parse_error_unterminated_string() {
        let err = parse(r#"RECALL episodic ABOUT "unterminated"#).unwrap_err();
        assert!(err.line >= 1);
        assert!(err.column >= 1);
    }

    #[test]
    fn parse_case_insensitive() {
        let q1 = parse(r#"recall episodic about "test""#).unwrap();
        let q2 = parse(r#"RECALL EPISODIC ABOUT "test""#).unwrap();
        let q3 = parse(r#"Recall Episodic About "test""#).unwrap();
        assert_eq!(q1, q2);
        assert_eq!(q2, q3);
    }

    #[test]
    fn parse_with_comments() {
        let q = "-- this is a comment\nRECALL episodic ABOUT \"test\"";
        let stmt = parse(q).unwrap();
        assert!(matches!(stmt, Statement::Recall(_)));
    }

    #[test]
    fn parse_single_quoted_strings() {
        let q = "RECALL episodic ABOUT 'test query'";
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.about, "test query"),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_multiline_query() {
        let q = "RECALL episodic\n  ABOUT \"test\"\n  LIMIT 10";
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.about, "test");
                assert_eq!(r.limit, Some(10));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_empty_input() {
        let err = parse("").unwrap_err();
        assert!(err.line >= 1);
    }

    #[test]
    fn parse_between_temporal() {
        let q = r#"RECALL episodic ABOUT "test" BETWEEN "2026-03-01" AND "2026-03-15""#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(
                    r.temporal,
                    Some(TemporalClause::Between {
                        start: "2026-03-01".into(),
                        end: "2026-03-15".into(),
                    })
                );
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_concept_9_4_semantic_search() {
        let q = r#"
            RECALL semantic, episodic
              ABOUT "vector database optimization"
              EXPAND GRAPH DEPTH 2 MIN_WEIGHT 0.3 ACTIVATION spreading
              WHERE importance > 0.4
              LIMIT 20
        "#;
        let stmt = parse(q).unwrap();
        assert!(matches!(stmt, Statement::Recall(_)));
    }

    #[test]
    fn parse_concept_9_4_temporal_narrative() {
        // This concept query has INVOLVING but no ABOUT, which the grammar requires.
        // For now, test with ABOUT added.
        let q = r#"
            RECALL episodic
              ABOUT "deployment and production events"
              INVOLVING "deployment", "production"
              BETWEEN "2026-03-01" AND "2026-03-15"
              AS NARRATIVE
        "#;
        let stmt = parse(q).unwrap();
        assert!(matches!(stmt, Statement::Recall(_)));
    }

    #[test]
    fn parse_concept_9_4_causal_chain() {
        let q = r#"
            RECALL episodic
              ABOUT "production outage"
              FOLLOW CAUSES DEPTH 3
              AS CAUSAL_CHAIN
        "#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.follow_causes, Some(3));
                assert_eq!(r.output_format, Some(OutputFormat::CausalChain));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_concept_9_4_think() {
        let q = r#"
            THINK
              ABOUT "How should I optimize HNSW for high-dimensional embeddings?"
              EXPAND GRAPH DEPTH 2 ACTIVATION spreading
              BUDGET 4096
        "#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert_eq!(t.budget, Some(4096));
                assert!(t.expand.is_some());
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_concept_9_4_connect_is_unsupported() {
        let q = r#"
            CONNECT "HNSW_indexing" TO "approximate_nearest_neighbors"
              AS related_to
              WEIGHT 0.9
        "#;
        let err = parse(q).unwrap_err();
        assert!(err.message.contains("CONNECT is not supported"));
    }

    #[test]
    fn parse_concept_9_4_trace() {
        let q = r#"TRACE "semantic:caching_best_practices""#;
        let stmt = parse(q).unwrap();
        assert!(matches!(stmt, Statement::Trace(_)));
    }

    #[test]
    fn parse_concept_9_4_cross_agent() {
        let q = r#"
            RECALL semantic
              ABOUT "API rate limiting patterns"
              WHERE confidence > 0.8
              NAMESPACE shared_knowledge
        "#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.namespace, Some("shared_knowledge".into()));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_concept_9_4_consistency() {
        let q = r#"
            RECALL semantic
              ABOUT "compliance rules"
              CONSISTENCY linearizable
        "#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.consistency, Some(ConsistencyLevel::Linearizable));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_recall() {
        let q = r#"RECALL episodic ABOUT "test" LIMIT 10"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn roundtrip_think() {
        let q = r#"THINK ABOUT "optimize queries" BUDGET 4096 LIMIT 5"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn roundtrip_history() {
        let q = r#"HISTORY "id" NAMESPACE custom"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn roundtrip_inspect() {
        let q = r#"INSPECT "id""#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn roundtrip_trace() {
        let q = r#"TRACE "id""#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn roundtrip_full_recall() {
        let q = r#"RECALL semantic, episodic ABOUT "optimization" INVOLVING "HNSW" AFTER "2026-03-01" EXPAND GRAPH DEPTH 2 MIN_WEIGHT 0.3 ACTIVATION spreading WHERE importance > 0.4 BUDGET 4096 NAMESPACE test LIMIT 20"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    #[test]
    fn fuzz_no_panics() {
        // Quick fuzz — random-ish strings should never panic, only return Err.
        let inputs = [
            "",
            "   ",
            "\n\n",
            "SELECT * FROM x",
            "RECALL",
            "RECALL episodic",
            "RECALL episodic ABOUT",
            "RECALL ABOUT \"x\"",
            "THINK",
            "REMEMBER",
            "FORGET",
            "CONNECT",
            "INSPECT",
            "HISTORY",
            "TRACE",
            "CONSOLIDATE",
            "😀 unicode",
            "RECALL episodic ABOUT \"x\" LIMIT -1",
            "RECALL episodic ABOUT \"x\" LIMIT 999999999999",
        ];
        let long_input = "A".repeat(10000);
        let mut inputs_vec: Vec<&str> = inputs.to_vec();
        inputs_vec.push(long_input.as_str());
        for input in inputs_vec {
            let _ = parse(input); // must not panic
        }
    }

    // ── Aggregations & Projections ─────────────────────────

    #[test]
    fn parse_group_by_count() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" GROUP BY entity_type COUNT"#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                let gb = r.group_by.unwrap();
                assert_eq!(gb.field, "entity_type");
                assert_eq!(gb.function, AggFunction::Count);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_group_by_avg() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" GROUP BY importance AVG"#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                let gb = r.group_by.unwrap();
                assert_eq!(gb.field, "importance");
                assert_eq!(gb.function, AggFunction::Avg);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_select_projection() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" SELECT id, summary, importance"#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                let proj = r.projection.unwrap();
                assert_eq!(proj, vec!["id", "summary", "importance"]);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_format_json() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" FORMAT json"#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.result_format.unwrap(), OutputFormat::Json);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_format_csv() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" FORMAT csv"#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.result_format.unwrap(), OutputFormat::Csv);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_group_by_with_no_results_still_parses() {
        // GROUP BY with empty results is a runtime concern, but parsing should succeed.
        let stmt =
            parse(r#"RECALL episodic ABOUT "nonexistent" GROUP BY entity_type COUNT LIMIT 0"#)
                .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert!(r.group_by.is_some());
                assert_eq!(r.limit, Some(0));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_select_single_field() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" SELECT id"#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.projection.unwrap(), vec!["id"]);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_combined_group_by_and_format() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "test" GROUP BY entity_type COUNT FORMAT json LIMIT 10"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert!(r.group_by.is_some());
                assert_eq!(r.result_format.unwrap(), OutputFormat::Json);
                assert_eq!(r.limit, Some(10));
            }
            _ => panic!("expected Recall"),
        }
    }

    // ── Subqueries & Time-Travel ──

    #[test]
    fn parse_as_of_clause() {
        let stmt =
            parse(r#"RECALL episodic ABOUT "deployment" AS OF "2026-03-01T12:00:00Z" LIMIT 5"#)
                .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(
                    r.as_of.unwrap(),
                    RecallSnapshotAst::Unqualified("2026-03-01T12:00:00Z".to_string())
                );
                assert_eq!(r.limit, Some(5));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_explicit_observed_as_of_clause() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "deployment" AS OF OBSERVED "2026-03-01T12:00:00Z" LIMIT 5"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(
                    r.as_of.unwrap(),
                    RecallSnapshotAst::Observed("2026-03-01T12:00:00Z".to_string())
                );
                assert_eq!(r.limit, Some(5));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_recorded_as_of_clause() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "deployment" AS OF RECORDED "2026-03-01T12:00:00Z" LIMIT 5"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(
                    r.as_of.unwrap(),
                    RecallSnapshotAst::Recorded("2026-03-01T12:00:00Z".to_string())
                );
                assert_eq!(r.limit, Some(5));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_revision_as_of_clause() {
        let stmt = parse(
            r#"RECALL semantic ABOUT "deployment" AS OF REVISION "01HW7N0Z5CH9R1R7Z4S4V5Y4QF" LIMIT 5"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(
                    r.as_of.unwrap(),
                    RecallSnapshotAst::Revision("01HW7N0Z5CH9R1R7Z4S4V5Y4QF".to_string())
                );
                assert_eq!(r.limit, Some(5));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_in_subquery() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "outage" WHERE entity IN (RECALL semantic ABOUT "critical services") LIMIT 10"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.subquery_filters.len(), 1);
                assert_eq!(r.subquery_filters[0].field, "entity");
                assert_eq!(r.subquery_filters[0].subquery.about, "critical services");
                assert_eq!(r.subquery_filters[0].subquery.layers, vec![Layer::Semantic]);
                assert_eq!(r.limit, Some(10));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_subquery_with_temporal() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "bugs" WHERE entity IN (RECALL episodic ABOUT "releases" AFTER "2026-01-01" LIMIT 5)"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.subquery_filters.len(), 1);
                let sq = &r.subquery_filters[0].subquery;
                assert_eq!(sq.about, "releases");
                assert!(sq.temporal.is_some());
                assert_eq!(sq.limit, Some(5));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_subquery_with_involving() {
        let stmt = parse(
            r#"RECALL semantic ABOUT "patterns" WHERE entity IN (RECALL episodic ABOUT "events" INVOLVING "auth", "db")"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.subquery_filters.len(), 1);
                let sq = &r.subquery_filters[0].subquery;
                assert_eq!(
                    sq.involving.as_ref().unwrap(),
                    &vec!["auth".to_string(), "db".to_string()]
                );
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_as_of_with_where() {
        let stmt =
            parse(r#"RECALL episodic ABOUT "events" AS OF "2026-06-01" WHERE importance > 0.5"#)
                .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(
                    r.as_of.unwrap(),
                    RecallSnapshotAst::Unqualified("2026-06-01".to_string())
                );
                assert_eq!(r.where_clauses.len(), 1);
                assert_eq!(r.where_clauses[0].field, "importance");
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_where_with_both_condition_and_subquery() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "test" WHERE importance > 0.5 WHERE entity IN (RECALL semantic ABOUT "services")"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.where_clauses.len(), 1);
                assert_eq!(r.subquery_filters.len(), 1);
            }
            _ => panic!("expected Recall"),
        }
    }

    // ── TRAVERSE, Batch FORGET, Upsert REMEMBER ──

    #[test]
    fn parse_traverse_minimal() {
        let stmt = parse(r#"TRAVERSE FROM "node1" DEPTH 3"#).unwrap();
        match stmt {
            Statement::Traverse(t) => {
                assert_eq!(t.from, "node1");
                assert_eq!(t.depth, 3);
                assert!(t.via.is_none());
                assert!(t.where_clauses.is_empty());
                assert!(t.limit.is_none());
            }
            other => panic!("expected Traverse, got {other:?}"),
        }
    }

    #[test]
    fn parse_traverse_with_via() {
        let stmt =
            parse(r#"TRAVERSE FROM "concept_a" VIA causes, related_to DEPTH 2 LIMIT 10"#).unwrap();
        match stmt {
            Statement::Traverse(t) => {
                assert_eq!(t.from, "concept_a");
                assert_eq!(t.via.as_ref().unwrap(), &["causes", "related_to"]);
                assert_eq!(t.depth, 2);
                assert_eq!(t.limit, Some(10));
            }
            other => panic!("expected Traverse, got {other:?}"),
        }
    }

    #[test]
    fn parse_traverse_with_where() {
        let stmt =
            parse(r#"TRAVERSE FROM "root" DEPTH 5 WHERE weight > 0.5 WHERE confidence > 0.3"#)
                .unwrap();
        match stmt {
            Statement::Traverse(t) => {
                assert_eq!(t.depth, 5);
                assert_eq!(t.where_clauses.len(), 2);
                assert_eq!(t.where_clauses[0].field, "weight");
                assert_eq!(t.where_clauses[1].field, "confidence");
            }
            other => panic!("expected Traverse, got {other:?}"),
        }
    }

    #[test]
    fn roundtrip_traverse() {
        let q = r#"TRAVERSE FROM "node1" VIA causes DEPTH 3 LIMIT 10"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    // ── Parameter placeholder tests ────────────────────────────────────

    #[test]
    fn parse_positional_param_in_about() {
        let stmt = parse(r#"RECALL episodic ABOUT $1 LIMIT 10"#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.about, "$1"),
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_named_param_in_about() {
        let stmt = parse(r#"RECALL episodic ABOUT $query LIMIT 5"#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.about, "$query"),
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_param_in_where_condition() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" WHERE importance > $threshold"#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.where_clauses.len(), 1);
                assert_eq!(
                    r.where_clauses[0].value,
                    ConditionValue::Param("$threshold".into())
                );
            }
            _ => panic!("expected Recall"),
        }
    }

    // ── EXPLAIN ──

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ExpectedExplainInner {
        Recall,
        RecallEvents,
        Think,
        Correct,
        Supersede,
        MergeMemory,
        Retract,
        History,
        Traverse,
        Inspect,
        Trace,
        ExplainCauses,
        WhatIf,
        Counterfactual,
        ShowPolicies,
        ExplainPolicy,
    }

    fn explain_inner_matches(statement: &Statement, expected: ExpectedExplainInner) -> bool {
        match expected {
            ExpectedExplainInner::Recall => matches!(statement, Statement::Recall(_)),
            ExpectedExplainInner::RecallEvents => matches!(statement, Statement::RecallEvents(_)),
            ExpectedExplainInner::Think => matches!(statement, Statement::Think(_)),
            ExpectedExplainInner::Correct => matches!(statement, Statement::Correct(_)),
            ExpectedExplainInner::Supersede => matches!(statement, Statement::Supersede(_)),
            ExpectedExplainInner::MergeMemory => matches!(statement, Statement::MergeMemory(_)),
            ExpectedExplainInner::Retract => matches!(statement, Statement::Retract(_)),
            ExpectedExplainInner::History => matches!(statement, Statement::History(_)),
            ExpectedExplainInner::Traverse => matches!(statement, Statement::Traverse(_)),
            ExpectedExplainInner::Inspect => matches!(statement, Statement::Inspect(_)),
            ExpectedExplainInner::Trace => matches!(statement, Statement::Trace(_)),
            ExpectedExplainInner::ExplainCauses => {
                matches!(statement, Statement::ExplainCauses(_))
            }
            ExpectedExplainInner::WhatIf => matches!(statement, Statement::WhatIf(_)),
            ExpectedExplainInner::Counterfactual => {
                matches!(statement, Statement::Counterfactual(_))
            }
            ExpectedExplainInner::ShowPolicies => matches!(statement, Statement::ShowPolicies(_)),
            ExpectedExplainInner::ExplainPolicy => {
                matches!(statement, Statement::ExplainPolicy(_))
            }
        }
    }

    fn assert_explain_shape(
        query: &str,
        expected_analyze: bool,
        expected_inner: ExpectedExplainInner,
    ) {
        let stmt = parse(query).unwrap();
        match stmt {
            Statement::Explain(e) => {
                assert_eq!(
                    e.analyze, expected_analyze,
                    "unexpected analyze flag for `{query}`"
                );
                assert!(
                    explain_inner_matches(e.inner.as_ref(), expected_inner),
                    "unexpected inner statement for `{query}`: {:?}",
                    e.inner
                );
            }
            other => panic!("expected Explain, got {other:?} for `{query}`"),
        }
    }

    #[test]
    fn parse_explain_statement_matrix() {
        for (query, expected_inner) in [
            (
                r#"EXPLAIN RECALL episodic ABOUT "test""#,
                ExpectedExplainInner::Recall,
            ),
            (
                r#"EXPLAIN RECALL EVENTS LIMIT 10"#,
                ExpectedExplainInner::RecallEvents,
            ),
            (
                r#"EXPLAIN THINK ABOUT "reasoning""#,
                ExpectedExplainInner::Think,
            ),
            (
                r#"EXPLAIN CORRECT "01ARZ3NDEKTSV4RRFFQ69G5FAV" SET description = "updated""#,
                ExpectedExplainInner::Correct,
            ),
            (
                r#"EXPLAIN SUPERSEDE "01ARZ3NDEKTSV4RRFFQ69G5FAV" SET description = "replacement""#,
                ExpectedExplainInner::Supersede,
            ),
            (
                r#"EXPLAIN RETRACT "01ARZ3NDEKTSV4RRFFQ69G5FAV" REASON "obsolete""#,
                ExpectedExplainInner::Retract,
            ),
            (
                r#"EXPLAIN HISTORY "01ARZ3NDEKTSV4RRFFQ69G5FAV" NAMESPACE custom"#,
                ExpectedExplainInner::History,
            ),
            (
                r#"EXPLAIN TRAVERSE FROM "01ARZ3NDEKTSV4RRFFQ69G5FAV" DEPTH 3"#,
                ExpectedExplainInner::Traverse,
            ),
            (
                r#"EXPLAIN INSPECT LOGICAL "01ARZ3NDEKTSV4RRFFQ69G5FAV""#,
                ExpectedExplainInner::Inspect,
            ),
            (
                r#"EXPLAIN TRACE LOGICAL "01ARZ3NDEKTSV4RRFFQ69G5FAV""#,
                ExpectedExplainInner::Trace,
            ),
            (
                r#"EXPLAIN EXPLAIN CAUSES "incident" DEPTH 2"#,
                ExpectedExplainInner::ExplainCauses,
            ),
            (
                r#"EXPLAIN WHAT_IF "increase timeout" THEN "fewer errors""#,
                ExpectedExplainInner::WhatIf,
            ),
            (
                r#"EXPLAIN COUNTERFACTUAL "cause" THEN "effect""#,
                ExpectedExplainInner::Counterfactual,
            ),
            (
                r#"EXPLAIN SHOW POLICIES FOR AGENT "agent-007""#,
                ExpectedExplainInner::ShowPolicies,
            ),
            (
                r#"EXPLAIN EXPLAIN POLICY FOR AGENT "agent-007" ON NAMESPACE "default" ACTION recall"#,
                ExpectedExplainInner::ExplainPolicy,
            ),
        ] {
            assert_explain_shape(query, false, expected_inner);
        }
    }

    #[test]
    fn parse_explain_analyze_statement_matrix() {
        for (query, expected_inner) in [
            (
                r#"EXPLAIN ANALYZE RECALL episodic ABOUT "test""#,
                ExpectedExplainInner::Recall,
            ),
            (
                r#"EXPLAIN ANALYZE RECALL EVENTS LIMIT 10"#,
                ExpectedExplainInner::RecallEvents,
            ),
            (
                r#"EXPLAIN ANALYZE THINK ABOUT "reasoning""#,
                ExpectedExplainInner::Think,
            ),
            (
                r#"EXPLAIN ANALYZE CORRECT "01ARZ3NDEKTSV4RRFFQ69G5FAV" SET description = "updated""#,
                ExpectedExplainInner::Correct,
            ),
            (
                r#"EXPLAIN ANALYZE SUPERSEDE "01ARZ3NDEKTSV4RRFFQ69G5FAV" SET description = "replacement""#,
                ExpectedExplainInner::Supersede,
            ),
            (
                r#"EXPLAIN ANALYZE MERGE MEMORY "01ARZ3NDEKTSV4RRFFQ69G5FAA" INTO "01ARZ3NDEKTSV4RRFFQ69G5FAV""#,
                ExpectedExplainInner::MergeMemory,
            ),
            (
                r#"EXPLAIN ANALYZE RETRACT "01ARZ3NDEKTSV4RRFFQ69G5FAV" REASON "obsolete""#,
                ExpectedExplainInner::Retract,
            ),
            (
                r#"EXPLAIN ANALYZE HISTORY "01ARZ3NDEKTSV4RRFFQ69G5FAV" NAMESPACE custom"#,
                ExpectedExplainInner::History,
            ),
            (
                r#"EXPLAIN ANALYZE TRAVERSE FROM "01ARZ3NDEKTSV4RRFFQ69G5FAV" DEPTH 3"#,
                ExpectedExplainInner::Traverse,
            ),
            (
                r#"EXPLAIN ANALYZE INSPECT LOGICAL "01ARZ3NDEKTSV4RRFFQ69G5FAV""#,
                ExpectedExplainInner::Inspect,
            ),
            (
                r#"EXPLAIN ANALYZE TRACE LOGICAL "01ARZ3NDEKTSV4RRFFQ69G5FAV""#,
                ExpectedExplainInner::Trace,
            ),
            (
                r#"EXPLAIN ANALYZE EXPLAIN CAUSES "incident" DEPTH 2"#,
                ExpectedExplainInner::ExplainCauses,
            ),
            (
                r#"EXPLAIN ANALYZE WHAT_IF "increase timeout" THEN "fewer errors""#,
                ExpectedExplainInner::WhatIf,
            ),
            (
                r#"EXPLAIN ANALYZE COUNTERFACTUAL "cause" THEN "effect""#,
                ExpectedExplainInner::Counterfactual,
            ),
            (
                r#"EXPLAIN ANALYZE SHOW POLICIES FOR AGENT "agent-007""#,
                ExpectedExplainInner::ShowPolicies,
            ),
            (
                r#"EXPLAIN ANALYZE EXPLAIN POLICY FOR AGENT "agent-007" ON NAMESPACE "default" ACTION recall"#,
                ExpectedExplainInner::ExplainPolicy,
            ),
        ] {
            assert_explain_shape(query, true, expected_inner);
        }
    }

    #[test]
    fn parse_explain_rejects_non_wrappable_statement_classes() {
        for prefix in ["EXPLAIN", "EXPLAIN ANALYZE"] {
            for inner in [
                r#"REMEMBER episode CONTENT "data""#,
                "WATCH ALL FORMAT json",
                r#"CONNECT "01ARZ3NDEKTSV4RRFFQ69G5FAV" TO "01ARZ3NDEKTSV4RRFFQ69G5FAA" AS causes"#,
                r#"GRANT recall ON NAMESPACE "default" TO AGENT "agent-007""#,
                r#"REVOKE forget ON NAMESPACE "sensitive" FROM AGENT "rogue""#,
                "SET TIER_POLICY semantic_archive_threshold = 0.2",
                r#"CONSOLIDATE WHERE episodic.access_count > 5"#,
                r#"CREATE REALM "analytics""#,
                r#"DROP REALM "analytics" CONFIRM"#,
                "SHOW CLUSTER",
            ] {
                let query = format!("{prefix} {inner}");
                assert!(
                    parse(&query).is_err(),
                    "`{query}` should be rejected by the EXPLAIN wrapper allowlist"
                );
            }
        }
    }

    #[test]
    fn parse_modality_filter() {
        let stmt = parse(r#"RECALL episodic ABOUT "login page" MODALITY image, text"#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.about, "login page");
                let mods = r.modality.unwrap();
                assert_eq!(mods, vec!["image".to_string(), "text".to_string()]);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_modality_single() {
        let stmt = parse(r#"RECALL episodic ABOUT "diagrams" MODALITY image"#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                let mods = r.modality.unwrap();
                assert_eq!(mods, vec!["image".to_string()]);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_recall_without_modality_is_none() {
        let stmt = parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert!(r.modality.is_none());
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_modality_with_other_clauses() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "query" WHERE importance > 0.5 MODALITY code, text LIMIT 10"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                let mods = r.modality.unwrap();
                assert_eq!(mods, vec!["code".to_string(), "text".to_string()]);
                assert_eq!(r.limit, Some(10));
                assert_eq!(r.where_clauses.len(), 1);
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_modality_extended_profiles() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "artifact" MODALITY video, document, composite, external LIMIT 10"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                let mods = r.modality.unwrap();
                assert_eq!(
                    mods,
                    vec![
                        "video".to_string(),
                        "document".to_string(),
                        "composite".to_string(),
                        "external".to_string(),
                    ]
                );
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_resource_aware_recall_clauses() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "artifact" MODALITY image RESOURCE_ROLE source, proof HYDRATION preview, full ARTIFACT preview, caption LIMIT 5"#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.modality.unwrap(), vec!["image".to_string()]);
                assert_eq!(
                    r.resource_roles.unwrap(),
                    vec!["source".to_string(), "proof".to_string()]
                );
                assert_eq!(
                    r.hydration_modes.unwrap(),
                    vec!["preview".to_string(), "full".to_string()]
                );
                assert_eq!(
                    r.artifact_kinds.unwrap(),
                    vec!["preview".to_string(), "caption".to_string()]
                );
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn parse_recall_without_resource_aware_clauses_is_none() {
        let stmt = parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert!(r.resource_roles.is_none());
                assert!(r.hydration_modes.is_none());
                assert!(r.artifact_kinds.is_none());
            }
            _ => panic!("expected Recall"),
        }
    }

    // ── Realm & Policy statement parsing ──

    #[test]
    fn parse_create_realm() {
        let stmt = parse(r#"CREATE REALM "analytics""#).unwrap();
        match stmt {
            Statement::CreateRealm(r) => {
                assert_eq!(r.name, "analytics");
                assert!(r.description.is_none());
            }
            _ => panic!("expected CreateRealm"),
        }
    }

    #[test]
    fn parse_create_realm_with_description() {
        let stmt = parse(r#"CREATE REALM "analytics" DESCRIPTION "For analytics data""#).unwrap();
        match stmt {
            Statement::CreateRealm(r) => {
                assert_eq!(r.name, "analytics");
                assert_eq!(r.description.as_deref(), Some("For analytics data"));
            }
            _ => panic!("expected CreateRealm"),
        }
    }

    #[test]
    fn parse_drop_realm() {
        let stmt = parse(r#"DROP REALM "analytics""#).unwrap();
        match stmt {
            Statement::DropRealm(r) => {
                assert_eq!(r.name, "analytics");
                assert!(!r.confirm);
            }
            _ => panic!("expected DropRealm"),
        }
    }

    #[test]
    fn parse_drop_realm_confirm() {
        let stmt = parse(r#"DROP REALM "analytics" CONFIRM"#).unwrap();
        match stmt {
            Statement::DropRealm(r) => {
                assert_eq!(r.name, "analytics");
                assert!(r.confirm);
            }
            _ => panic!("expected DropRealm"),
        }
    }

    #[test]
    fn parse_grant_single_action() {
        let stmt = parse(r#"GRANT recall ON NAMESPACE "default" TO AGENT "agent-007""#).unwrap();
        match stmt {
            Statement::Grant(g) => {
                assert_eq!(g.actions, vec!["recall"]);
                assert_eq!(g.target, GrantTarget::Namespace("default".into()));
                assert_eq!(g.principal, PrincipalRef::Agent("agent-007".into()));
            }
            _ => panic!("expected Grant"),
        }
    }

    #[test]
    fn parse_grant_multiple_actions() {
        let stmt =
            parse(r#"GRANT recall, remember, think ON REALM "prod" TO TEAM "data-scientists""#)
                .unwrap();
        match stmt {
            Statement::Grant(g) => {
                assert_eq!(g.actions, vec!["recall", "remember", "think"]);
                assert_eq!(g.target, GrantTarget::Realm("prod".into()));
                assert_eq!(g.principal, PrincipalRef::Team("data-scientists".into()));
            }
            _ => panic!("expected Grant"),
        }
    }

    #[test]
    fn parse_revoke() {
        let stmt = parse(r#"REVOKE forget ON NAMESPACE "sensitive" FROM AGENT "rogue""#).unwrap();
        match stmt {
            Statement::Revoke(r) => {
                assert_eq!(r.actions, vec!["forget"]);
                assert_eq!(r.target, GrantTarget::Namespace("sensitive".into()));
                assert_eq!(r.principal, PrincipalRef::Agent("rogue".into()));
            }
            _ => panic!("expected Revoke"),
        }
    }

    #[test]
    fn parse_show_policies() {
        let stmt = parse(r#"SHOW POLICIES"#).unwrap();
        match stmt {
            Statement::ShowPolicies(s) => {
                assert!(s.principal.is_none());
            }
            _ => panic!("expected ShowPolicies"),
        }
    }

    #[test]
    fn parse_show_policies_for_agent() {
        let stmt = parse(r#"SHOW POLICIES FOR AGENT "agent-007""#).unwrap();
        match stmt {
            Statement::ShowPolicies(s) => {
                assert_eq!(s.principal, Some(PrincipalRef::Agent("agent-007".into())));
            }
            _ => panic!("expected ShowPolicies"),
        }
    }

    #[test]
    fn parse_explain_policy() {
        let stmt =
            parse(r#"EXPLAIN POLICY FOR AGENT "agent-007" ON NAMESPACE "default" ACTION recall"#)
                .unwrap();
        match stmt {
            Statement::ExplainPolicy(e) => {
                assert_eq!(e.principal, PrincipalRef::Agent("agent-007".into()));
                assert_eq!(e.resource_type, "namespace");
                assert_eq!(e.resource_name, "default");
                assert_eq!(e.action, "recall");
            }
            _ => panic!("expected ExplainPolicy"),
        }
    }

    #[test]
    fn parse_explain_policy_on_realm() {
        let stmt =
            parse(r#"EXPLAIN POLICY FOR TEAM "analysts" ON REALM "prod" ACTION remember"#).unwrap();
        match stmt {
            Statement::ExplainPolicy(e) => {
                assert_eq!(e.principal, PrincipalRef::Team("analysts".into()));
                assert_eq!(e.resource_type, "realm");
                assert_eq!(e.resource_name, "prod");
                assert_eq!(e.action, "remember");
            }
            _ => panic!("expected ExplainPolicy"),
        }
    }

    #[test]
    fn roundtrip_create_realm() {
        let stmt = parse(r#"CREATE REALM "test-realm" DESCRIPTION "A test realm""#).unwrap();
        let display = stmt.to_string();
        assert!(display.contains("CREATE REALM"));
        assert!(display.contains("test-realm"));
        assert!(display.contains("DESCRIPTION"));
    }

    #[test]
    fn roundtrip_grant() {
        let stmt = parse(r#"GRANT recall, think ON NAMESPACE "ns1" TO AGENT "a1""#).unwrap();
        let display = stmt.to_string();
        assert!(display.contains("GRANT recall, think"));
        assert!(display.contains("NAMESPACE"));
        assert!(display.contains("AGENT"));
    }

    #[test]
    fn recall_events_basic() {
        let stmt = parse(r#"RECALL EVENTS WHERE event_type = "access_denied""#).unwrap();
        match stmt {
            Statement::RecallEvents(r) => {
                assert_eq!(r.where_clauses.len(), 1);
                assert_eq!(r.where_clauses[0].field, "event_type");
            }
            _ => panic!("expected RecallEvents"),
        }
    }

    #[test]
    fn recall_events_multiple_filters() {
        let stmt = parse(
            r#"RECALL EVENTS WHERE agent = "agent-007" WHERE event_type = "access_denied" LIMIT 100"#,
        )
        .unwrap();
        match stmt {
            Statement::RecallEvents(r) => {
                assert_eq!(r.where_clauses.len(), 2);
                assert_eq!(r.limit, Some(100));
            }
            _ => panic!("expected RecallEvents"),
        }
    }

    #[test]
    fn recall_events_with_temporal() {
        let stmt = parse(
            r#"RECALL EVENTS WHERE event_type = "policy_changed" AFTER "2026-01-01" LIMIT 50"#,
        )
        .unwrap();
        match stmt {
            Statement::RecallEvents(r) => {
                assert!(r.temporal.is_some());
                assert_eq!(r.limit, Some(50));
            }
            _ => panic!("expected RecallEvents"),
        }
    }

    #[test]
    fn roundtrip_recall_events() {
        let stmt = parse(
            r#"RECALL EVENTS WHERE agent_id = "a1" WHERE event_type = "access_denied" NAMESPACE test LIMIT 10"#,
        )
        .unwrap();
        let display = stmt.to_string();
        assert!(display.contains("RECALL EVENTS"));
        assert!(display.contains("LIMIT 10"));
    }

    // ── SHOW CLUSTER ─────────────────────────────
    #[test]
    fn parse_show_cluster() {
        let stmt = parse("SHOW CLUSTER").unwrap();
        assert!(matches!(stmt, Statement::ShowCluster));
    }

    #[test]
    fn parse_show_cluster_status() {
        let stmt = parse("SHOW CLUSTER STATUS").unwrap();
        assert!(matches!(stmt, Statement::ShowCluster));
    }

    #[test]
    fn parse_show_cluster_case_insensitive() {
        let stmt = parse("show cluster").unwrap();
        assert!(matches!(stmt, Statement::ShowCluster));
    }

    #[test]
    fn roundtrip_show_cluster() {
        let stmt = parse("SHOW CLUSTER").unwrap();
        assert_eq!(stmt.to_string(), "SHOW CLUSTER");
    }

    #[test]
    fn parse_recall_hybrid() {
        let q = r#"RECALL episodic ABOUT "semantic search" LIMIT 10 HYBRID"#;
        let stmt = parse(q).unwrap();
        if let Statement::Recall(r) = &stmt {
            assert!(r.hybrid);
            assert_eq!(r.about, "semantic search");
            assert_eq!(r.limit, Some(10));
        } else {
            panic!("expected Recall");
        }
    }

    #[test]
    fn parse_recall_without_hybrid() {
        let q = r#"RECALL episodic ABOUT "query" LIMIT 5"#;
        let stmt = parse(q).unwrap();
        if let Statement::Recall(r) = &stmt {
            assert!(!r.hybrid);
        } else {
            panic!("expected Recall");
        }
    }

    #[test]
    fn parse_recall_hybrid_case_insensitive() {
        let q = r#"RECALL episodic ABOUT "test" hybrid"#;
        let stmt = parse(q).unwrap();
        if let Statement::Recall(r) = &stmt {
            assert!(r.hybrid);
        } else {
            panic!("expected Recall");
        }
    }

    #[test]
    fn roundtrip_recall_hybrid() {
        let q = r#"RECALL episodic ABOUT "hybrid test" LIMIT 10 HYBRID"#;
        let stmt1 = parse(q).unwrap();
        let rendered = stmt1.to_string();
        assert!(rendered.contains("HYBRID"));
        let stmt2 = parse(&rendered).unwrap();
        assert_eq!(stmt1, stmt2);
    }

    // ── String unescape & injection tests ──────────────────

    #[test]
    fn invalid_escape_sequence_returns_error() {
        let q = r#"RECALL episodic ABOUT "hello\qworld""#;
        let result = parse(q);
        assert!(result.is_err(), "expected parse error for \\q escape");
    }

    #[test]
    fn valid_escape_newline_succeeds() {
        let q = r#"RECALL episodic ABOUT "hello\nworld""#;
        let stmt = parse(q).unwrap();
        if let Statement::Recall(r) = &stmt {
            assert_eq!(r.about, "hello\nworld");
        } else {
            panic!("expected Recall");
        }
    }

    #[test]
    fn valid_escape_tab_succeeds() {
        let q = r#"RECALL episodic ABOUT "col1\tcol2""#;
        let stmt = parse(q).unwrap();
        if let Statement::Recall(r) = &stmt {
            assert_eq!(r.about, "col1\tcol2");
        } else {
            panic!("expected Recall");
        }
    }

    #[test]
    fn valid_escape_backslash_succeeds() {
        let q = r#"RECALL episodic ABOUT "path\\to\\file""#;
        let stmt = parse(q).unwrap();
        if let Statement::Recall(r) = &stmt {
            assert_eq!(r.about, "path\\to\\file");
        } else {
            panic!("expected Recall");
        }
    }

    #[test]
    fn valid_escape_quote_succeeds() {
        let q = r#"RECALL episodic ABOUT "say \"hello\"" "#;
        let stmt = parse(q).unwrap();
        if let Statement::Recall(r) = &stmt {
            assert_eq!(r.about, "say \"hello\"");
        } else {
            panic!("expected Recall");
        }
    }

    // ── PPR/PageRank grammar tests ─────────────────────────

    #[test]
    fn parse_recall_activation_ppr() {
        let q = r#"RECALL episodic ABOUT "test" EXPAND GRAPH DEPTH 3 ACTIVATION PPR"#;
        let stmt = parse(q).unwrap();
        if let Statement::Recall(r) = &stmt {
            let expand = r.expand.as_ref().expect("should have expand clause");
            assert_eq!(expand.activation, Some(ActivationModeAst::Ppr));
            assert_eq!(expand.depth, 3);
        } else {
            panic!("expected Recall");
        }
    }

    #[test]
    fn parse_recall_activation_pagerank() {
        let q = r#"RECALL episodic ABOUT "test" EXPAND GRAPH DEPTH 2 ACTIVATION PAGERANK"#;
        let stmt = parse(q).unwrap();
        if let Statement::Recall(r) = &stmt {
            let expand = r.expand.as_ref().expect("should have expand clause");
            assert_eq!(expand.activation, Some(ActivationModeAst::Ppr));
            assert_eq!(expand.depth, 2);
        } else {
            panic!("expected Recall");
        }
    }

    // ── Numeric parse error tests ──────────────────────────

    #[test]
    fn limit_overflow_returns_error() {
        let q = r#"RECALL episodic ABOUT "test" LIMIT 99999999999999999999"#;
        let result = parse(q);
        assert!(result.is_err(), "expected error for overflow LIMIT");
    }

    #[test]
    fn limit_valid_still_works() {
        let q = r#"RECALL episodic ABOUT "test" LIMIT 10"#;
        let stmt = parse(q).unwrap();
        if let Statement::Recall(r) = &stmt {
            assert_eq!(r.limit, Some(10));
        } else {
            panic!("expected Recall");
        }
    }

    // ── Query limits tests ─────────────────────────────────

    #[test]
    fn query_too_large_returns_error() {
        let limits = QueryLimits {
            max_query_length: 50,
            ..Default::default()
        };
        let q = &"x".repeat(100);
        let result = parse_with_limits(q, &limits);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("exceeds maximum"));
    }

    #[test]
    fn expand_depth_exceeds_limit() {
        let limits = QueryLimits {
            max_expand_depth: 5,
            ..Default::default()
        };
        let q = r#"RECALL episodic ABOUT "test" EXPAND GRAPH DEPTH 6"#;
        let result = parse_with_limits(q, &limits);
        let err = result.unwrap_err();
        assert!(
            err.message.contains("DEPTH") || err.message.contains("depth"),
            "expected depth error, got: {}",
            err.message
        );
    }

    #[test]
    fn limit_exceeds_max_returns_error() {
        let limits = QueryLimits {
            max_limit: 100,
            ..Default::default()
        };
        let q = r#"RECALL episodic ABOUT "test" LIMIT 200"#;
        let result = parse_with_limits(q, &limits);
        assert!(result.is_err());
        let msg = result.unwrap_err().message.to_lowercase();
        assert!(msg.contains("limit") || msg.contains("exceed"));
    }

    #[test]
    fn normal_query_with_default_limits_succeeds() {
        let limits = QueryLimits::default();
        let q = r#"RECALL episodic ABOUT "test" LIMIT 100"#;
        let stmt = parse_with_limits(q, &limits).unwrap();
        if let Statement::Recall(r) = &stmt {
            assert_eq!(r.limit, Some(100));
        } else {
            panic!("expected Recall");
        }
    }

    // ── DEPTH clause tests ─────────────────────────────────────────────

    #[test]
    fn parse_recall_depth_auto() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" DEPTH AUTO"#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.depth_mode, Some(DepthModeAst::Auto)),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_depth_full() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" DEPTH FULL"#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.depth_mode, Some(DepthModeAst::Full)),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_depth_summary() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" DEPTH SUMMARY"#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.depth_mode, Some(DepthModeAst::Summary)),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_depth_full() {
        let stmt = parse(r#"THINK ABOUT "test" DEPTH FULL"#).unwrap();
        match stmt {
            Statement::Think(t) => assert_eq!(t.depth_mode, Some(DepthModeAst::Full)),
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_depth_omitted_is_none() {
        let stmt = parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.depth_mode, None),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    // ── WITH PROSPECTIVE clause tests ──────────────────────────────────

    #[test]
    fn parse_recall_with_prospective_on() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" WITH PROSPECTIVE ON"#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.with_prospective, Some(true)),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_with_prospective_off() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" WITH PROSPECTIVE OFF"#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.with_prospective, Some(false)),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_with_prospective_on() {
        let stmt = parse(r#"THINK ABOUT "test" WITH PROSPECTIVE ON"#).unwrap();
        match stmt {
            Statement::Think(t) => assert_eq!(t.with_prospective, Some(true)),
            other => panic!("expected Think, got {other:?}"),
        }
    }

    // ── WITH MCFA_DEFENSE clause tests ─────────────────────────────────

    #[test]
    fn parse_recall_with_mcfa_on() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" WITH MCFA_DEFENSE ON"#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.with_mcfa, Some(true)),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_with_mcfa_off() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" WITH MCFA_DEFENSE OFF"#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.with_mcfa, Some(false)),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_with_mcfa_off() {
        let stmt = parse(r#"THINK ABOUT "test" WITH MCFA_DEFENSE OFF"#).unwrap();
        match stmt {
            Statement::Think(t) => assert_eq!(t.with_mcfa, Some(false)),
            other => panic!("expected Think, got {other:?}"),
        }
    }

    // ── WITH CONFLICTS clause tests ────────────────────────────────────

    #[test]
    fn parse_recall_with_conflicts() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" WITH CONFLICTS"#).unwrap();
        match stmt {
            Statement::Recall(r) => assert!(r.with_conflicts),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_without_conflicts_default() {
        let stmt = parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        match stmt {
            Statement::Recall(r) => assert!(!r.with_conflicts),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    // ── TOPIC clause tests (Story 5.2) ─────────────────────────────────

    #[test]
    fn parse_recall_topic() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" TOPIC "deployment""#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.topic, Some("deployment".to_string()));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_topic_with_temporal() {
        let stmt = parse(
            r#"RECALL episodic ABOUT "test" BETWEEN "2026-01-01" AND "2026-06-01" TOPIC "deployment""#,
        )
        .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.topic, Some("deployment".to_string()));
                assert_eq!(
                    r.temporal,
                    Some(TemporalClause::Between {
                        start: "2026-01-01".into(),
                        end: "2026-06-01".into()
                    })
                );
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_topic_omitted_is_none() {
        let stmt = parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        match stmt {
            Statement::Recall(r) => assert_eq!(r.topic, None),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    // ── MODE ITERATIVE clause tests (Story 5.4) ───────────────────────

    #[test]
    fn parse_think_mode_iterative() {
        let stmt = parse(r#"THINK ABOUT "test" BUDGET 4096 MODE ITERATIVE MAX_HOPS 3"#).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert_eq!(t.mode, RetrievalMode::Iterative);
                assert_eq!(t.max_hops, Some(3));
                assert_eq!(t.budget, Some(4096));
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_mode_iterative_without_max_hops() {
        let stmt = parse(r#"THINK ABOUT "test" MODE ITERATIVE"#).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert_eq!(t.mode, RetrievalMode::Iterative);
                assert_eq!(t.max_hops, None);
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_mode_iterative_max_hops_validation_zero() {
        let result = parse(r#"THINK ABOUT "test" MODE ITERATIVE MAX_HOPS 0"#);
        assert!(result.is_err());
        let msg = result.unwrap_err().message;
        assert!(msg.contains("MAX_HOPS must be between 1 and 5"));
    }

    #[test]
    fn parse_think_mode_iterative_max_hops_validation_too_high() {
        let result = parse(r#"THINK ABOUT "test" MODE ITERATIVE MAX_HOPS 6"#);
        assert!(result.is_err());
        let msg = result.unwrap_err().message;
        assert!(msg.contains("MAX_HOPS must be between 1 and 5"));
    }

    #[test]
    fn parse_think_mode_iterative_max_hops_boundary_valid() {
        let stmt = parse(r#"THINK ABOUT "test" MODE ITERATIVE MAX_HOPS 1"#).unwrap();
        match stmt {
            Statement::Think(t) => assert_eq!(t.max_hops, Some(1)),
            other => panic!("expected Think, got {other:?}"),
        }

        let stmt = parse(r#"THINK ABOUT "test" MODE ITERATIVE MAX_HOPS 5"#).unwrap();
        match stmt {
            Statement::Think(t) => assert_eq!(t.max_hops, Some(5)),
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_non_iterative_is_default() {
        let stmt = parse(r#"THINK ABOUT "test""#).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert_eq!(t.mode, RetrievalMode::Local);
                assert_eq!(t.max_hops, None);
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_mode_adaptive_clause() {
        let stmt = parse(r#"THINK ABOUT "test" MODE ADAPTIVE"#).unwrap();
        match stmt {
            Statement::Think(t) => assert_eq!(t.mode, RetrievalMode::Adaptive),
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_mode_raptor_clause() {
        let stmt = parse(r#"THINK ABOUT "test" MODE RAPTOR"#).unwrap();
        match stmt {
            Statement::Think(t) => assert_eq!(t.mode, RetrievalMode::Raptor),
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_mode_hybrid_clause() {
        let stmt = parse(r#"THINK ABOUT "test" MODE HYBRID"#).unwrap();
        match stmt {
            Statement::Think(t) => assert_eq!(t.mode, RetrievalMode::Hybrid),
            other => panic!("expected Think, got {other:?}"),
        }
    }

    // ── EVENTS clause tests (Story 5.1) ───────────────────────────────

    #[test]
    fn parse_recall_events_between() {
        let stmt = parse(r#"RECALL EVENTS BETWEEN "2026-03-01" AND "2026-03-15""#).unwrap();
        match stmt {
            Statement::RecallEvents(re) => {
                assert_eq!(
                    re.temporal,
                    Some(TemporalClause::Between {
                        start: "2026-03-01".into(),
                        end: "2026-03-15".into()
                    })
                );
                assert_eq!(re.entity_filter, None);
            }
            other => panic!("expected RecallEvents, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_events_for_entity() {
        let stmt = parse(r#"RECALL EVENTS FOR "nginx""#).unwrap();
        match stmt {
            Statement::RecallEvents(re) => {
                assert_eq!(re.entity_filter, Some("nginx".to_string()));
            }
            other => panic!("expected RecallEvents, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_events_for_entity_with_temporal() {
        let stmt =
            parse(r#"RECALL EVENTS FOR "nginx" BETWEEN "2026-03-01" AND "2026-03-15""#).unwrap();
        match stmt {
            Statement::RecallEvents(re) => {
                assert_eq!(re.entity_filter, Some("nginx".to_string()));
                assert_eq!(
                    re.temporal,
                    Some(TemporalClause::Between {
                        start: "2026-03-01".into(),
                        end: "2026-03-15".into()
                    })
                );
            }
            other => panic!("expected RecallEvents, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_events_where_subject() {
        let stmt = parse(r#"RECALL EVENTS WHERE subject = "user_login""#).unwrap();
        match stmt {
            Statement::RecallEvents(re) => {
                assert_eq!(re.where_clauses.len(), 1);
                assert_eq!(re.where_clauses[0].field, "subject");
                assert_eq!(
                    re.where_clauses[0].value,
                    ConditionValue::String("user_login".into())
                );
            }
            other => panic!("expected RecallEvents, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_events_for_with_where_and_limit() {
        let stmt = parse(r#"RECALL EVENTS FOR "nginx" WHERE verb = "crashed" LIMIT 10"#).unwrap();
        match stmt {
            Statement::RecallEvents(re) => {
                assert_eq!(re.entity_filter, Some("nginx".to_string()));
                assert_eq!(re.where_clauses.len(), 1);
                assert_eq!(re.where_clauses[0].field, "verb");
                assert_eq!(re.limit, Some(10));
            }
            other => panic!("expected RecallEvents, got {other:?}"),
        }
    }

    // ── Combined clause tests ──────────────────────────────────────────

    #[test]
    fn parse_recall_all_new_clauses() {
        let q = r#"
            RECALL episodic
              ABOUT "deployment"
              DEPTH FULL
              TOPIC "k8s"
              WITH PROSPECTIVE ON
              WITH MCFA_DEFENSE OFF
              WITH CONFLICTS
              LIMIT 20
        "#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.depth_mode, Some(DepthModeAst::Full));
                assert_eq!(r.topic, Some("k8s".to_string()));
                assert_eq!(r.with_prospective, Some(true));
                assert_eq!(r.with_mcfa, Some(false));
                assert!(r.with_conflicts);
                assert_eq!(r.limit, Some(20));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_all_new_clauses() {
        let q = r#"
            THINK ABOUT "optimize HNSW"
              DEPTH SUMMARY
              WITH PROSPECTIVE OFF
              WITH MCFA_DEFENSE ON
              BUDGET 4096
              MODE ITERATIVE MAX_HOPS 2
              HYBRID
        "#;
        let stmt = parse(q).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert_eq!(t.depth_mode, Some(DepthModeAst::Summary));
                assert_eq!(t.with_prospective, Some(false));
                assert_eq!(t.with_mcfa, Some(true));
                assert_eq!(t.mode, RetrievalMode::Iterative);
                assert_eq!(t.max_hops, Some(2));
                assert_eq!(t.budget, Some(4096));
                assert!(t.hybrid);
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    // ── Display round-trip tests ───────────────────────────────────────

    #[test]
    fn display_recall_topic() {
        let q = r#"RECALL episodic ABOUT "test" TOPIC "deployment""#;
        let stmt = parse(q).unwrap();
        let displayed = stmt.to_string();
        assert!(
            displayed.contains("TOPIC \"deployment\""),
            "got: {displayed}"
        );
    }

    #[test]
    fn display_think_mode_iterative() {
        let q = r#"THINK ABOUT "test" MODE ITERATIVE MAX_HOPS 3"#;
        let stmt = parse(q).unwrap();
        let displayed = stmt.to_string();
        assert!(displayed.contains("MODE iterative"), "got: {displayed}");
        assert!(displayed.contains("MAX_HOPS 3"), "got: {displayed}");
    }

    #[test]
    fn display_recall_events_for() {
        let q = r#"RECALL EVENTS FOR "nginx""#;
        let stmt = parse(q).unwrap();
        let displayed = stmt.to_string();
        assert!(displayed.contains("FOR \"nginx\""), "got: {displayed}");
    }

    #[test]
    fn display_recall_depth_mode() {
        let q = r#"RECALL episodic ABOUT "test" DEPTH FULL"#;
        let stmt = parse(q).unwrap();
        let displayed = stmt.to_string();
        assert!(displayed.contains("DEPTH FULL"), "got: {displayed}");
    }

    #[test]
    fn display_recall_with_clauses() {
        let q = r#"RECALL episodic ABOUT "test" WITH PROSPECTIVE ON WITH MCFA_DEFENSE OFF WITH CONFLICTS"#;
        let stmt = parse(q).unwrap();
        let displayed = stmt.to_string();
        assert!(
            displayed.contains("WITH PROSPECTIVE ON"),
            "got: {displayed}"
        );
        assert!(
            displayed.contains("WITH MCFA_DEFENSE OFF"),
            "got: {displayed}"
        );
        assert!(displayed.contains("WITH CONFLICTS"), "got: {displayed}");
    }

    // ── MAX_HOPS only valid with ITERATIVE ─────────────────────────────

    #[test]
    fn parse_think_max_hops_without_iterative_rejected() {
        // MAX_HOPS should not parse with non-iterative modes since grammar
        // only allows max_hops_clause after retrieval_mode within mode_clause.
        // MODE LOCAL MAX_HOPS 3 should fail at grammar level.
        let result = parse(r#"THINK ABOUT "test" MODE LOCAL MAX_HOPS 3"#);
        assert!(
            result.is_err(),
            "MAX_HOPS with LOCAL mode should be rejected"
        );
    }

    // ── WITH PROVENANCE DEPTH clause tests (Story 4.2) ─────────────────

    #[test]
    fn parse_recall_with_provenance_depth() {
        let stmt = parse(r#"RECALL semantic ABOUT "test" WITH PROVENANCE DEPTH 2"#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.provenance_depth, Some(2));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_provenance_depth_omitted_is_none() {
        let stmt = parse(r#"RECALL semantic ABOUT "test""#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.provenance_depth, None);
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_think_with_provenance_depth() {
        let stmt = parse(r#"THINK ABOUT "test" WITH PROVENANCE DEPTH 3"#).unwrap();
        match stmt {
            Statement::Think(t) => {
                assert_eq!(t.provenance_depth, Some(3));
            }
            other => panic!("expected Think, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_provenance_with_conflicts_combo() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" WITH CONFLICTS WITH PROVENANCE DEPTH 1"#)
            .unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert!(r.with_conflicts);
                assert_eq!(r.provenance_depth, Some(1));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    // ── SET TIER_POLICY parse tests ─────────────────────────────────

    #[test]
    fn parse_set_tier_policy_string_value() {
        let stmt = parse("SET TIER_POLICY working_to_episodic_ttl = '2h'").unwrap();
        match stmt {
            Statement::SetTierPolicy(s) => {
                assert_eq!(s.field, "working_to_episodic_ttl");
                assert_eq!(s.value, TierPolicyValue::Str("2h".into()));
            }
            other => panic!("expected SetTierPolicy, got {other:?}"),
        }
    }

    #[test]
    fn parse_set_tier_policy_float_value() {
        let stmt = parse("SET TIER_POLICY episodic_to_semantic_threshold = 0.85").unwrap();
        match stmt {
            Statement::SetTierPolicy(s) => {
                assert_eq!(s.field, "episodic_to_semantic_threshold");
                assert_eq!(s.value, TierPolicyValue::Float(0.85));
            }
            other => panic!("expected SetTierPolicy, got {other:?}"),
        }
    }

    #[test]
    fn parse_set_tier_policy_integer_value() {
        let stmt = parse("SET TIER_POLICY working_to_episodic_ttl = 3600").unwrap();
        match stmt {
            Statement::SetTierPolicy(s) => {
                assert_eq!(s.field, "working_to_episodic_ttl");
                assert_eq!(s.value, TierPolicyValue::Int(3600));
            }
            other => panic!("expected SetTierPolicy, got {other:?}"),
        }
    }

    #[test]
    fn parse_set_tier_policy_case_insensitive() {
        let stmt = parse("set tier_policy procedural_min_success_rate = 0.5").unwrap();
        match stmt {
            Statement::SetTierPolicy(s) => {
                assert_eq!(s.field, "procedural_min_success_rate");
                assert_eq!(s.value, TierPolicyValue::Float(0.5));
            }
            other => panic!("expected SetTierPolicy, got {other:?}"),
        }
    }

    #[test]
    fn parse_set_tier_policy_display_roundtrip() {
        let stmt = parse("SET TIER_POLICY semantic_archive_threshold = 0.2").unwrap();
        let display = format!("{stmt}");
        assert_eq!(display, "SET TIER_POLICY semantic_archive_threshold = 0.2");
    }

    // ── FROM REALM clause tests (Story 6.2) ───────────────────────────

    #[test]
    fn parse_recall_from_realm_single() {
        let stmt = parse(r#"RECALL episodic ABOUT "test" FROM REALM "production""#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.from_realms, Some(vec!["production".to_string()]));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_from_realm_multiple() {
        let stmt =
            parse(r#"RECALL episodic ABOUT "test" FROM REALM "production", "staging""#).unwrap();
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(
                    r.from_realms,
                    Some(vec!["production".to_string(), "staging".to_string()])
                );
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_without_from_realm() {
        let stmt = parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        match stmt {
            Statement::Recall(r) => assert!(r.from_realms.is_none()),
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn parse_recall_from_realm_display_roundtrip() {
        let stmt =
            parse(r#"RECALL episodic ABOUT "test" FROM REALM "production", "staging" LIMIT 10"#)
                .unwrap();
        let display = format!("{stmt}");
        assert!(display.contains("FROM REALM"));
        assert!(display.contains("\"production\""));
        assert!(display.contains("\"staging\""));
    }

    // ── Pearl's 3-Rung Causal Statements ───────────────────────────────

    #[test]
    fn parse_explain_causes_basic() {
        let stmt = parse(r#"EXPLAIN CAUSES "deployment failure""#).unwrap();
        match stmt {
            Statement::ExplainCauses(ec) => {
                assert_eq!(ec.target, "deployment failure");
                assert_eq!(ec.namespace, None);
                assert_eq!(ec.depth, None);
            }
            other => panic!("expected ExplainCauses, got {other:?}"),
        }
    }

    #[test]
    fn parse_explain_causes_with_depth() {
        let stmt = parse(r#"EXPLAIN CAUSES "server crash" DEPTH 5"#).unwrap();
        match stmt {
            Statement::ExplainCauses(ec) => {
                assert_eq!(ec.target, "server crash");
                assert_eq!(ec.depth, Some(5));
            }
            other => panic!("expected ExplainCauses, got {other:?}"),
        }
    }

    #[test]
    fn parse_explain_causes_with_namespace() {
        let stmt = parse(r#"EXPLAIN CAUSES "deployment failure" NAMESPACE ops"#).unwrap();
        match stmt {
            Statement::ExplainCauses(ec) => {
                assert_eq!(ec.target, "deployment failure");
                assert_eq!(ec.namespace, Some("ops".into()));
                assert_eq!(ec.depth, None);
            }
            other => panic!("expected ExplainCauses, got {other:?}"),
        }
    }

    #[test]
    fn parse_explain_causes_full() {
        let stmt = parse(r#"EXPLAIN CAUSES "deployment failure" NAMESPACE ops DEPTH 3"#).unwrap();
        match stmt {
            Statement::ExplainCauses(ec) => {
                assert_eq!(ec.target, "deployment failure");
                assert_eq!(ec.namespace, Some("ops".into()));
                assert_eq!(ec.depth, Some(3));
            }
            other => panic!("expected ExplainCauses, got {other:?}"),
        }
    }

    #[test]
    fn parse_what_if_basic() {
        let stmt = parse(r#"WHAT_IF "increase timeout" THEN "fewer errors""#).unwrap();
        match stmt {
            Statement::WhatIf(wi) => {
                assert_eq!(wi.intervention, "increase timeout");
                assert_eq!(wi.outcome, "fewer errors");
                assert_eq!(wi.namespace, None);
            }
            other => panic!("expected WhatIf, got {other:?}"),
        }
    }

    #[test]
    fn parse_what_if_with_namespace() {
        let stmt =
            parse(r#"WHAT_IF "increase timeout" THEN "fewer errors" NAMESPACE prod"#).unwrap();
        match stmt {
            Statement::WhatIf(wi) => {
                assert_eq!(wi.intervention, "increase timeout");
                assert_eq!(wi.outcome, "fewer errors");
                assert_eq!(wi.namespace, Some("prod".into()));
            }
            other => panic!("expected WhatIf, got {other:?}"),
        }
    }

    #[test]
    fn parse_counterfactual_basic() {
        let stmt = parse(r#"COUNTERFACTUAL "if deploy had not happened" THEN "outage""#).unwrap();
        match stmt {
            Statement::Counterfactual(cf) => {
                assert_eq!(cf.antecedent, "if deploy had not happened");
                assert_eq!(cf.consequent, "outage");
                assert_eq!(cf.namespace, None);
            }
            other => panic!("expected Counterfactual, got {other:?}"),
        }
    }

    #[test]
    fn parse_counterfactual_with_namespace() {
        let stmt = parse(
            r#"COUNTERFACTUAL "if deploy had not happened" THEN "outage" NAMESPACE production"#,
        )
        .unwrap();
        match stmt {
            Statement::Counterfactual(cf) => {
                assert_eq!(cf.antecedent, "if deploy had not happened");
                assert_eq!(cf.consequent, "outage");
                assert_eq!(cf.namespace, Some("production".into()));
            }
            other => panic!("expected Counterfactual, got {other:?}"),
        }
    }

    #[test]
    fn parse_explain_causes_display_roundtrip() {
        let stmt = parse(r#"EXPLAIN CAUSES "failure" NAMESPACE ops DEPTH 3"#).unwrap();
        let display = format!("{stmt}");
        assert!(display.contains("EXPLAIN CAUSES"));
        assert!(display.contains("failure"));
    }

    #[test]
    fn parse_what_if_display_roundtrip() {
        let stmt = parse(r#"WHAT_IF "intervention" THEN "outcome""#).unwrap();
        let display = format!("{stmt}");
        assert!(display.contains("WHAT_IF"));
        assert!(display.contains("intervention"));
        assert!(display.contains("outcome"));
    }

    #[test]
    fn parse_counterfactual_display_roundtrip() {
        let stmt = parse(r#"COUNTERFACTUAL "cause" THEN "effect""#).unwrap();
        let display = format!("{stmt}");
        assert!(display.contains("COUNTERFACTUAL"));
        assert!(display.contains("cause"));
        assert!(display.contains("effect"));
    }
}
