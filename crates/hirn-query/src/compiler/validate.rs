//! Untyped-AST semantic validation — value ranges, field whitelists, and
//! format checks that go beyond what the PEG grammar can enforce.
//!
//! This is Stage 2a of the compilation pipeline: [`validate`] runs at the top
//! of `typed_ast::analyze`, so every execution path (text compile, bound
//! prepared statements) gets the same checks exactly once. It is also called
//! directly by the engine's prepared-statement front-end to reject invalid
//! parameter-free templates at `prepare()` time.
//!
//! Parameter placeholders (`$name`) are skipped by value checks — they are
//! validated after `bind()` substitutes concrete values, when the bound AST
//! is compiled for execution.

use std::collections::HashSet;

use crate::parser::ast::*;

/// A semantic error discovered during validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisError {
    pub message: String,
    pub kind: AnalysisErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisErrorKind {
    /// Unknown field name in WHERE clause.
    UnknownField,
    /// Type mismatch (e.g., comparing importance with a string).
    TypeMismatch,
    /// Invalid temporal format.
    InvalidTemporal,
    /// Value out of range (e.g., importance > 1.0).
    ValueOutOfRange,
    /// Missing required clause.
    MissingRequired,
    /// Unknown relation type for CONNECT.
    UnknownRelation,
    /// Invalid layer for operation.
    InvalidLayer,
}

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "analysis error: {}", self.message)
    }
}

impl std::error::Error for AnalysisError {}

/// Known WHERE-clause fields and their expected value types.
const NUMERIC_FIELDS: &[&str] = &[
    "importance",
    "confidence",
    "surprise",
    "access_count",
    "evidence_count",
    "relevance_score",
    "success_rate",
    "invocation_count",
    "trust",
    "episodic.access_count",
];

/// Validate a parsed statement for semantic correctness.
///
/// Returns a list of errors (empty = valid).
pub fn validate(stmt: &Statement) -> Vec<AnalysisError> {
    match stmt {
        Statement::Recall(r) => validate_recall(r),
        Statement::Think(t) => validate_think(t),
        Statement::Correct(c) => validate_correct(c),
        Statement::Supersede(s) => validate_supersede(s),
        Statement::MergeMemory(m) => validate_merge_memory(m),
        Statement::Retract(r) => validate_retract(r),
        Statement::Inspect(_) | Statement::History(_) | Statement::Trace(_) => vec![],
        Statement::Traverse(t) => validate_traverse(t),
        Statement::Explain(e) => validate(&e.inner),
        Statement::CreateRealm(_)
        | Statement::DropRealm(_)
        | Statement::Grant(_)
        | Statement::Revoke(_)
        | Statement::ShowPolicies(_)
        | Statement::ExplainPolicy(_)
        | Statement::RecallEvents(_)
        | Statement::ShowCluster
        | Statement::SetTierPolicy(_)
        | Statement::ExplainCauses(_)
        | Statement::WhatIf(_)
        | Statement::Counterfactual(_) => vec![],
    }
}

fn semantic_target_is_empty(target: &SemanticTargetRef) -> bool {
    target.raw_value().trim().is_empty()
}

fn validate_recall(r: &RecallStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if r.about.is_blank_literal() {
        errors.push(AnalysisError {
            message: "ABOUT clause cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    if r.layers.is_empty() {
        errors.push(AnalysisError {
            message: "RECALL requires at least one layer".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    errors.extend(validate_recall_where_clauses(&r.where_clauses));
    errors.extend(validate_temporal(r.temporal.as_ref()));
    errors.extend(validate_expand(r.expand.as_ref()));
    errors.extend(validate_budget(r.budget));
    errors
}

fn validate_think(t: &ThinkStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if t.about.is_blank_literal() {
        errors.push(AnalysisError {
            message: "THINK ABOUT clause cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    errors.extend(validate_recall_where_clauses(&t.where_clauses));
    errors.extend(validate_temporal(t.temporal.as_ref()));
    errors.extend(validate_expand(t.expand.as_ref()));
    errors.extend(validate_budget(t.budget));
    errors
}

fn validate_correct(c: &CorrectStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if semantic_target_is_empty(&c.target) {
        errors.push(AnalysisError {
            message: "CORRECT target cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    errors.extend(validate_semantic_updates(&c.updates, "CORRECT", true));
    errors.extend(validate_semantic_observed_at(
        c.observed_at.as_ref(),
        "CORRECT",
    ));

    errors
}

fn validate_semantic_updates(
    updates: &[SetAssignment],
    verb: &str,
    require_updates: bool,
) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if require_updates && updates.is_empty() {
        errors.push(AnalysisError {
            message: format!("{verb} requires at least one field assignment"),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    for update in updates {
        match update.field.as_str() {
            "description" => {
                if !matches!(update.value, SetValue::String(_)) {
                    errors.push(AnalysisError {
                        message: format!("{verb} description requires a string value"),
                        kind: AnalysisErrorKind::TypeMismatch,
                    });
                }
            }
            "confidence" => {
                let value = match update.value {
                    SetValue::Float(v) => Some(v),
                    SetValue::Int(v) => Some(v as f64),
                    _ => None,
                };

                if let Some(value) = value {
                    if !(0.0..=1.0).contains(&value) {
                        errors.push(AnalysisError {
                            message: format!(
                                "{verb} confidence must be between 0.0 and 1.0, got {value}"
                            ),
                            kind: AnalysisErrorKind::ValueOutOfRange,
                        });
                    }
                } else {
                    errors.push(AnalysisError {
                        message: format!("{verb} confidence requires a numeric value"),
                        kind: AnalysisErrorKind::TypeMismatch,
                    });
                }
            }
            "evidence_count" => match update.value {
                SetValue::Int(v) if v >= 0 => {}
                SetValue::Int(v) => errors.push(AnalysisError {
                    message: format!("{verb} evidence_count must be non-negative, got {v}"),
                    kind: AnalysisErrorKind::ValueOutOfRange,
                }),
                _ => errors.push(AnalysisError {
                    message: format!("{verb} evidence_count requires a non-negative integer"),
                    kind: AnalysisErrorKind::TypeMismatch,
                }),
            },
            other => errors.push(AnalysisError {
                message: format!(
                    "unknown {verb} field '{other}' (allowed: description, confidence, evidence_count)"
                ),
                kind: AnalysisErrorKind::UnknownField,
            }),
        }
    }

    errors
}

fn validate_semantic_observed_at(observed_at: Option<&String>, verb: &str) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if let Some(observed_at) = observed_at
        && !is_valid_temporal(observed_at)
    {
        errors.push(AnalysisError {
            message: format!("invalid {verb} OBSERVED AT temporal format: '{observed_at}'"),
            kind: AnalysisErrorKind::InvalidTemporal,
        });
    }

    errors
}

fn validate_supersede(s: &SupersedeStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if semantic_target_is_empty(&s.target) {
        errors.push(AnalysisError {
            message: "SUPERSEDE target cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    errors.extend(validate_semantic_updates(&s.updates, "SUPERSEDE", true));
    errors.extend(validate_semantic_observed_at(
        s.observed_at.as_ref(),
        "SUPERSEDE",
    ));

    errors
}

fn validate_merge_memory(m: &MergeMemoryStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if m.sources.is_empty() {
        errors.push(AnalysisError {
            message: "MERGE MEMORY requires at least one source memory".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    if semantic_target_is_empty(&m.target) {
        errors.push(AnalysisError {
            message: "MERGE MEMORY target cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    let mut seen_sources = HashSet::new();
    for source in &m.sources {
        let normalized = source.raw_value().trim();
        if normalized.is_empty() {
            errors.push(AnalysisError {
                message: "MERGE MEMORY source cannot be empty".into(),
                kind: AnalysisErrorKind::MissingRequired,
            });
            continue;
        }

        let canonical = source.to_string();
        if !seen_sources.insert(canonical.clone()) {
            errors.push(AnalysisError {
                message: format!("MERGE MEMORY source '{}' is duplicated", source.raw_value()),
                kind: AnalysisErrorKind::ValueOutOfRange,
            });
        }

        if canonical == m.target.to_string() {
            errors.push(AnalysisError {
                message: format!(
                    "MERGE MEMORY source '{}' cannot also be the target",
                    source.raw_value()
                ),
                kind: AnalysisErrorKind::ValueOutOfRange,
            });
        }
    }

    errors.extend(validate_semantic_updates(&m.updates, "MERGE MEMORY", false));
    errors.extend(validate_semantic_observed_at(
        m.observed_at.as_ref(),
        "MERGE MEMORY",
    ));

    errors
}

fn validate_retract(r: &RetractStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if semantic_target_is_empty(&r.target) {
        errors.push(AnalysisError {
            message: "RETRACT target cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    if let Some(ref observed_at) = r.observed_at {
        if !is_valid_temporal(observed_at) {
            errors.push(AnalysisError {
                message: format!("invalid OBSERVED AT temporal format: '{observed_at}'"),
                kind: AnalysisErrorKind::InvalidTemporal,
            });
        }
    }

    errors
}

fn validate_traverse(t: &TraverseStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if t.from.trim().is_empty() {
        errors.push(AnalysisError {
            message: "TRAVERSE FROM cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    if t.depth == 0 {
        errors.push(AnalysisError {
            message: "TRAVERSE DEPTH must be at least 1".into(),
            kind: AnalysisErrorKind::ValueOutOfRange,
        });
    }

    errors.extend(validate_where_clauses(&t.where_clauses));
    errors
}

/// WHERE fields that RECALL and THINK actually support end-to-end (the plan
/// compiler translates exactly these into physical-column predicates). Any
/// other field is silently dropped downstream, which makes results either
/// over-return or come back empty — so unknown fields are rejected up front
/// (fail closed), matching how RECALL EVENTS and CORRECT/SUPERSEDE SET already
/// reject unknown fields (R-49).
const RECALL_THINK_WHERE_FIELDS: &[&str] = &[
    "importance",
    "confidence",
    "success_rate",
    "surprise",
    "access_count",
    "evidence_count",
    "invocation_count",
    // `trust` is enforced as a post-load filter (provenance trust score) and
    // `relevance_score` is an alias for importance — both are supported by the
    // engine's recall filter path (see hirn-engine `ql::read_support`).
    "trust",
    "relevance_score",
];

/// Strip an optional `<layer>.` table qualifier (e.g. `episodic.access_count`)
/// so qualified and bare field names validate identically — the engine's recall
/// filter accepts both forms.
fn unqualified_field(field: &str) -> &str {
    field.rsplit('.').next().unwrap_or(field)
}

fn validate_where_clauses(clauses: &[WhereCondition]) -> Vec<AnalysisError> {
    validate_where_clauses_inner(clauses, None)
}

/// Like [`validate_where_clauses`], but additionally rejects any field not in
/// `whitelist` as an `UnknownField`. Used for RECALL/THINK where the set of
/// supported filter fields is closed.
fn validate_recall_where_clauses(clauses: &[WhereCondition]) -> Vec<AnalysisError> {
    validate_where_clauses_inner(clauses, Some(RECALL_THINK_WHERE_FIELDS))
}

fn validate_where_clauses_inner(
    clauses: &[WhereCondition],
    whitelist: Option<&[&str]>,
) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    for wc in clauses {
        // Reject fields outside the supported set (fail closed) so they are
        // never silently dropped by the plan compiler.
        if let Some(allowed) = whitelist
            && !allowed.contains(&unqualified_field(&wc.field))
        {
            errors.push(AnalysisError {
                message: format!(
                    "unknown WHERE field '{}' (supported: {})",
                    wc.field,
                    allowed.join(", ")
                ),
                kind: AnalysisErrorKind::UnknownField,
            });
            // Skip further type/range checks for a field we do not support.
            continue;
        }

        // Check that numeric fields are compared with numeric values.
        if NUMERIC_FIELDS.contains(&wc.field.as_str())
            && matches!(wc.value, ConditionValue::String(_))
        {
            errors.push(AnalysisError {
                message: format!("field '{}' expects a numeric value, got string", wc.field),
                kind: AnalysisErrorKind::TypeMismatch,
            });
        }

        // Check numeric range for known bounded fields.
        match wc.field.as_str() {
            "importance" | "confidence" | "trust" | "relevance_score" | "success_rate" => {
                let v = match &wc.value {
                    ConditionValue::Float(v) => Some(*v),
                    ConditionValue::Int(v) => Some(*v as f64),
                    _ => None,
                };
                if let Some(v) = v {
                    if !(0.0..=1.0).contains(&v) {
                        errors.push(AnalysisError {
                            message: format!(
                                "field '{}' threshold should be between 0.0 and 1.0, got {}",
                                wc.field, v
                            ),
                            kind: AnalysisErrorKind::ValueOutOfRange,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    errors
}

fn validate_temporal(temporal: Option<&TemporalClause>) -> Vec<AnalysisError> {
    let Some(tc) = temporal else { return vec![] };
    let mut errors = Vec::new();

    let timestamps = match tc {
        TemporalClause::After(s) => vec![s.as_str()],
        TemporalClause::Before(s) => vec![s.as_str()],
        TemporalClause::Between { start, end } => vec![start.as_str(), end.as_str()],
    };

    for ts in timestamps {
        if !is_valid_temporal(ts) {
            errors.push(AnalysisError {
                message: format!(
                    "invalid temporal format: '{ts}' (expected YYYY-MM-DD or RFC 3339)"
                ),
                kind: AnalysisErrorKind::InvalidTemporal,
            });
        }
    }

    errors
}

fn validate_expand(expand: Option<&ExpandClause>) -> Vec<AnalysisError> {
    let Some(ex) = expand else { return vec![] };
    let mut errors = Vec::new();

    if ex.depth == 0 {
        errors.push(AnalysisError {
            message: "EXPAND GRAPH DEPTH must be at least 1".into(),
            kind: AnalysisErrorKind::ValueOutOfRange,
        });
    }

    if let Some(mw) = ex.min_weight {
        if !(0.0..=1.0).contains(&mw) {
            errors.push(AnalysisError {
                message: format!("MIN_WEIGHT must be between 0.0 and 1.0, got {mw}"),
                kind: AnalysisErrorKind::ValueOutOfRange,
            });
        }
    }

    errors
}

fn validate_budget(budget: Option<usize>) -> Vec<AnalysisError> {
    if let Some(b) = budget {
        if b == 0 {
            return vec![AnalysisError {
                message: "BUDGET must be greater than 0".into(),
                kind: AnalysisErrorKind::ValueOutOfRange,
            }];
        }
    }
    vec![]
}

fn is_valid_temporal(s: &str) -> bool {
    use chrono::NaiveDate;
    // Accept YYYY-MM-DD.
    if NaiveDate::parse_from_str(s, "%Y-%m-%d").is_ok() {
        return true;
    }
    // Accept RFC 3339 / ISO 8601.
    if chrono::DateTime::parse_from_rfc3339(s).is_ok() {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn valid_recall_passes() {
        let stmt = parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        assert!(validate(&stmt).is_empty());
    }

    #[test]
    fn recall_with_valid_where() {
        let stmt = parse(r#"RECALL episodic ABOUT "x" WHERE importance > 0.5"#).unwrap();
        assert!(validate(&stmt).is_empty());
    }

    #[test]
    fn recall_with_out_of_range_importance() {
        let stmt = parse(r#"RECALL episodic ABOUT "x" WHERE importance > 2.0"#).unwrap();
        let errors = validate(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::ValueOutOfRange);
    }

    #[test]
    fn recall_with_invalid_temporal() {
        let stmt = parse(r#"RECALL episodic ABOUT "x" AFTER "not-a-date""#).unwrap();
        let errors = validate(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::InvalidTemporal);
    }

    #[test]
    fn parameterized_where_value_is_skipped() {
        // $threshold is validated after bind(), not at template validation.
        let stmt = parse(r#"RECALL episodic ABOUT $1 WHERE importance > $threshold"#).unwrap();
        assert!(validate(&stmt).is_empty());
    }

    #[test]
    fn recall_unknown_where_field_is_rejected() {
        // R-49: an unknown RECALL WHERE field is dropped by the plan compiler
        // (over-returns) or fails the engine post-load filter (empty). Reject
        // it up front instead of silently mis-answering.
        let stmt = parse(r#"RECALL episodic ABOUT "x" WHERE nonexistent = "y""#).unwrap();
        let errors = validate(&stmt);
        assert_eq!(errors.len(), 1, "errors: {errors:?}");
        assert_eq!(errors[0].kind, AnalysisErrorKind::UnknownField);
    }

    #[test]
    fn recall_engine_supported_fields_and_qualifiers_pass() {
        // `trust` (post-load provenance filter) and `relevance_score` (an
        // importance alias) ARE supported by the engine's recall filter path, as
        // is a `<layer>.` table qualifier — none may be rejected as unknown.
        for query in [
            r#"RECALL episodic ABOUT "x" WHERE trust < 0.95"#,
            r#"RECALL episodic ABOUT "x" WHERE relevance_score > 0.5"#,
            r#"RECALL episodic ABOUT "x" WHERE episodic.access_count >= 2"#,
        ] {
            let stmt = parse(query).unwrap();
            let errors = validate(&stmt);
            assert!(
                !errors
                    .iter()
                    .any(|e| e.kind == AnalysisErrorKind::UnknownField),
                "query '{query}' must not report an unknown field, got: {errors:?}"
            );
        }
    }

    #[test]
    fn think_unknown_where_field_is_rejected() {
        let stmt = parse(r#"THINK ABOUT "x" WHERE bogus > 1"#).unwrap();
        let errors = validate(&stmt);
        assert!(
            errors
                .iter()
                .any(|e| e.kind == AnalysisErrorKind::UnknownField),
            "errors: {errors:?}"
        );
    }

    #[test]
    fn recall_supported_where_fields_pass() {
        for field in [
            "importance",
            "confidence",
            "success_rate",
            "surprise",
            "access_count",
            "evidence_count",
            "invocation_count",
        ] {
            let stmt = parse(&format!(r#"RECALL episodic ABOUT "x" WHERE {field} > 1"#)).unwrap();
            let errors = validate(&stmt);
            assert!(
                !errors
                    .iter()
                    .any(|e| e.kind == AnalysisErrorKind::UnknownField),
                "field '{field}' must be accepted, got: {errors:?}"
            );
        }
    }

    #[test]
    fn traverse_where_field_is_not_whitelisted() {
        // TRAVERSE filters on traversal-output columns (e.g. `weight`, `depth`)
        // — these must NOT be rejected by the RECALL/THINK whitelist.
        let stmt = parse(r#"TRAVERSE FROM "node1" VIA causes DEPTH 2 WHERE weight > 0.5"#).unwrap();
        let errors = validate(&stmt);
        assert!(
            !errors
                .iter()
                .any(|e| e.kind == AnalysisErrorKind::UnknownField),
            "traverse WHERE fields must not be whitelisted: {errors:?}"
        );
    }

    #[test]
    fn correct_unknown_field_is_rejected() {
        let stmt = parse(r#"CORRECT "x" SET unsupported = 1"#).unwrap();
        let errors = validate(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::UnknownField);
    }

    #[test]
    fn supersede_unknown_field_is_rejected() {
        let stmt = parse(r#"SUPERSEDE "x" SET unsupported = 1"#).unwrap();
        let errors = validate(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::UnknownField);
    }

    #[test]
    fn retract_invalid_observed_at_is_rejected() {
        let stmt = parse(r#"RETRACT "x" OBSERVED AT "not-a-date""#).unwrap();
        let errors = validate(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::InvalidTemporal);
    }

    #[test]
    fn think_valid_passes() {
        let stmt = parse(r#"THINK ABOUT "test" BUDGET 4096"#).unwrap();
        assert!(validate(&stmt).is_empty());
    }

    #[test]
    fn think_global_valid() {
        let stmt = parse(r#"THINK GLOBAL ABOUT "test""#).unwrap();
        assert!(validate(&stmt).is_empty());
    }

    #[test]
    fn budget_zero_rejected() {
        let stmt = parse(r#"RECALL episodic ABOUT "x" BUDGET 0"#).unwrap();
        let errors = validate(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::ValueOutOfRange);
    }

    #[test]
    fn valid_temporal_formats() {
        assert!(is_valid_temporal("2026-03-01"));
        assert!(is_valid_temporal("2026-03-01T12:00:00Z"));
        assert!(is_valid_temporal("2026-03-01T12:00:00+01:00"));
        assert!(!is_valid_temporal("not-a-date"));
        assert!(!is_valid_temporal("March 1st"));
    }

    #[test]
    fn between_with_valid_dates() {
        let stmt =
            parse(r#"RECALL episodic ABOUT "x" BETWEEN "2026-01-01" AND "2026-03-01""#).unwrap();
        assert!(validate(&stmt).is_empty());
    }

    #[test]
    fn traverse_valid() {
        let stmt = parse(r#"TRAVERSE FROM "node1" DEPTH 3"#).unwrap();
        assert!(validate(&stmt).is_empty());
    }

    #[test]
    fn traverse_with_via_and_where() {
        let stmt = parse(r#"TRAVERSE FROM "node1" VIA causes DEPTH 2 WHERE weight > 0.5"#).unwrap();
        assert!(validate(&stmt).is_empty());
    }

    #[test]
    fn explain_valid_recall_no_warnings() {
        let stmt = parse(r#"EXPLAIN RECALL episodic ABOUT "test""#).unwrap();
        assert!(validate(&stmt).is_empty());
    }

    #[test]
    fn explain_analyze_delegates_to_inner() {
        // EXPLAIN ANALYZE on a query with an invalid range should still report the inner warning
        let stmt = parse(r#"EXPLAIN ANALYZE RECALL episodic ABOUT "test" WHERE importance > 2.0"#)
            .unwrap();
        let warnings = validate(&stmt);
        assert!(
            warnings
                .iter()
                .any(|w| matches!(w.kind, AnalysisErrorKind::ValueOutOfRange)),
            "should propagate inner analysis warnings: {warnings:?}"
        );
    }

    #[test]
    fn expand_min_weight_out_of_range_rejected() {
        let stmt =
            parse(r#"RECALL episodic ABOUT "x" EXPAND GRAPH DEPTH 2 MIN_WEIGHT 1.5"#).unwrap();
        let errors = validate(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::ValueOutOfRange);
    }
}
